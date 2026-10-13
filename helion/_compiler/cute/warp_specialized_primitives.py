# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CuTe DSL compile-time values intentionally have implicit Python annotations.
# ruff: noqa: ANN001, ANN202

"""Low-level synchronization, addressing, and copy primitives for CTA pipelines."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims


@cute.jit
def shared_pointer(base, byte_offset, dtype: cutlass.Constexpr):
    """Return a typed pointer into a CTA's byte-addressed shared allocation."""

    return cutlass.inttoptr(base + byte_offset, 3, dtype)


@cute.jit
def staged_mbarrier_pointer(base, byte_offset, stage):
    """Return one eight-byte mbarrier slot from a packed stage array."""

    return shared_pointer(base, byte_offset + stage * 8, cutlass.Int64)


@cute.jit
def wait_mbarrier(pointer, phase):
    """Wait for an mbarrier parity using the nonblocking TRY instruction."""

    while not prims.mbarrier_wait_parity(pointer, phase, prims.MBarrierWait.TRY):
        pass


@cute.jit
def arrive_mbarrier(pointer):
    """Publish one warp's writes and elect one lane to arrive."""

    prims.bar_warp_sync(cute.arch.FULL_MASK)
    if prims.elect_sync():
        prims.mbarrier_arrive(pointer)


@cute.jit
def elect_arrive_mbarrier(pointer):
    """Elect one lane to arrive without adding a warp synchronization."""

    if prims.elect_sync():
        prims.mbarrier_arrive(pointer)


@cute.jit
def fence_async_shared_and_arrive(pointer):
    """Publish async-proxy shared writes before one elected arrival."""

    prims.fence_proxy(
        prims.Proxy.ASYNC_SHARED,
        space=prims.SharedSpace.shared_cta,
    )
    elect_arrive_mbarrier(pointer)


@cute.jit
def tcgen05_commit_mbarrier(pointer):
    """Commit prior tcgen05 work and signal its completion barrier."""

    if prims.elect_sync():
        prims.tcgen05_commit(pointer, group=prims.CTAGroup.CTA_1)


@cute.jit
def expect_mbarrier_tx(pointer, byte_count):
    """Elect one lane to arm a transaction-counted mbarrier."""

    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(pointer, byte_count)


@cute.jit
def wait_and_flip_mbarrier(pointer, phase):
    """Wait for one parity and return the next producer/consumer phase."""

    wait_mbarrier(pointer, phase)
    return phase ^ cutlass.Int32(1)


@cute.jit
def initialize_mbarrier_region(
    shared_base,
    byte_offset,
    stages: cutlass.Constexpr,
    arrivals: cutlass.Constexpr,
    lane,
):
    """Cooperatively initialize a contiguous array of mbarriers with one warp."""

    for stage in cutlass.range(lane, stages, 32, unroll=1):
        prims.mbarrier_init(
            staged_mbarrier_pointer(shared_base, byte_offset, stage), arrivals
        )


@cute.jit
def named_barrier_sync(barrier_id, thread_count: cutlass.Constexpr):
    """Synchronize a statically-sized cooperating thread team."""

    prims.barrier_cta_sync(barrier_id, thread_count=thread_count)


@cute.jit
def fence_async_shared():
    """Publish ordinary shared writes to the async proxy."""

    cute.arch.fence_view_async_shared()


@cute.jit
def advance_ring_stage(
    stage,
    increment: cutlass.Constexpr,
    stages: cutlass.Constexpr,
):
    """Advance a stage cursor and return whether it wrapped."""

    next_stage = stage + cutlass.Int32(increment)
    wrapped = cutlass.Int32(next_stage >= cutlass.Int32(stages))
    return next_stage - wrapped * cutlass.Int32(stages), wrapped


def swizzle_b16_index(
    row,
    column,
    row_bytes: cutlass.Constexpr,
    columns_per_tile: cutlass.Constexpr,
    rows_per_tile: cutlass.Constexpr,
    phase_mask: cutlass.Constexpr,
):
    """Map a row/column pair into a swizzled 16-bit shared-memory tile."""

    byte = (
        (column // columns_per_tile) * rows_per_tile * row_bytes
        + row * row_bytes
        + (column % columns_per_tile) * 2
    )
    return byte ^ (((byte >> 7) & phase_mask) << 4)


def swizzle_b16_element_index(logical_element, phase_mask: cutlass.Constexpr):
    """Apply a byte-addressed XOR swizzle and return a 16-bit element index."""

    byte_offset = logical_element * 2
    return (byte_offset ^ (((byte_offset >> 7) & phase_mask) << 4)) // 2


def segmented_swizzle_b16_element_index(
    row,
    column,
    rows_per_segment: cutlass.Constexpr,
    columns_per_segment: cutlass.Constexpr,
    columns_per_group: cutlass.Constexpr,
    phase_mask: cutlass.Constexpr,
):
    """Map a logical row/column into independently swizzled column segments."""

    segment = column // columns_per_segment
    segment_column = column - segment * columns_per_segment
    group = segment_column // columns_per_group
    column_in_group = segment_column - group * columns_per_group
    row_phase = row & phase_mask
    return (
        segment * rows_per_segment * columns_per_segment
        + row * columns_per_segment
        + ((group ^ row_phase) * columns_per_group)
        + column_in_group
    )


def matrix_16x16_transposed_lane_coordinates(lane):
    """Return the row and column base owned by one STSM/LDMatrix lane."""

    matrix_id = lane // 8
    row_in_matrix = lane & 7
    row = (matrix_id // 2) * 8 + row_in_matrix
    column = (matrix_id & 1) * 8
    return row, column


@cute.jit
def copy_b16x8_async(destination, source, valid):
    """Copy one aligned 16-byte vector and zero-fill a predicated-off lane."""

    aligned_destination = cute.make_ptr(
        destination.dtype,
        destination.toint(),
        destination.memspace,
        assumed_align=16,
    )
    aligned_source = cute.make_ptr(
        source.dtype,
        source.toint(),
        source.memspace,
        assumed_align=16,
    )
    copy_size = cutlass.Int32(0)
    if valid:
        copy_size = cutlass.Int32(16)
    cute.arch.cp_async_shared_global(
        aligned_destination,
        aligned_source,
        16,
        "cg",
        cp_size=copy_size,
    )


@cute.jit
def issue_tmem_smem_mma_slices(
    operand_smem,
    tmem_raw_address,
    completion_mbarrier,
    accumulator_column,
    input_column,
    input_dtype: cutlass.Constexpr,
    n_dim: cutlass.Constexpr,
    m_dim: cutlass.Constexpr,
    k_atom: cutlass.Constexpr,
    element_bytes: cutlass.Constexpr,
    k_block_begin: cutlass.Constexpr,
    k_block_end: cutlass.Constexpr,
    leading_bytes: cutlass.Constexpr,
    stride_bytes: cutlass.Constexpr,
    swizzle_k_phases: cutlass.Constexpr,
    swizzle_bytes: cutlass.Constexpr,
    tile_rows: cutlass.Constexpr,
    initial_scale_d: cutlass.Constexpr,
    commit: cutlass.Constexpr,
):
    """Issue a compile-time range of TMEM-by-SMEM tcgen05 MMA slices."""

    accumulator = cutlass.inttoptr(
        tmem_raw_address + accumulator_column,
        6,
        cutlass.Float32,
    )
    instruction = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=input_dtype,
        b_dtype=input_dtype,
        n_dim=n_dim,
        m_dim=m_dim,
        b_major=0,
    )
    descriptor = prims.Tcgen05SmemDesc.build(
        operand_smem.subview(0),
        leading_byte_offset=leading_bytes,
        stride_byte_offset=stride_bytes,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    for k_block in cutlass.range_constexpr(k_block_begin, k_block_end):
        scale_d = initial_scale_d or k_block != k_block_begin
        operand_offset = (k_block % swizzle_k_phases) * k_atom * element_bytes + (
            k_block // swizzle_k_phases
        ) * tile_rows * swizzle_bytes
        tmem_input = prims.make_tmem_ptr(tmem_raw_address, cutlass.Int8).subview(
            input_column + k_block * (k_atom // 2)
        )
        if prims.elect_sync():
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                accumulator,
                tmem_input,
                descriptor.advance_start_address(operand_offset),
                instruction,
                scale_d,
            )
    if cutlass.const_expr(commit):
        tcgen05_commit_mbarrier(completion_mbarrier)


__all__ = [
    "advance_ring_stage",
    "arrive_mbarrier",
    "copy_b16x8_async",
    "elect_arrive_mbarrier",
    "expect_mbarrier_tx",
    "fence_async_shared",
    "fence_async_shared_and_arrive",
    "initialize_mbarrier_region",
    "issue_tmem_smem_mma_slices",
    "matrix_16x16_transposed_lane_coordinates",
    "named_barrier_sync",
    "segmented_swizzle_b16_element_index",
    "shared_pointer",
    "staged_mbarrier_pointer",
    "swizzle_b16_element_index",
    "swizzle_b16_index",
    "tcgen05_commit_mbarrier",
    "wait_and_flip_mbarrier",
    "wait_mbarrier",
]
