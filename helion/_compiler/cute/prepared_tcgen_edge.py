# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CuTe compile-time values intentionally have implicit types.
# ruff: noqa: ANN001, ANN202

"""Native execution of a bound, already-prepared TCgen communication edge.

This module establishes no graph, owner, seed, readiness or reuse authority.
The compiler-side bound operation must retain that authority across issue,
read, the original typed point body, and publication. Instruction leaves have
no K loop, completion wait, visibility fence or publication arrival.

The descriptor and CuTe-fragment representations preserve their original
instruction families. All coordinates and allocation bases are supplied by
the original selected layout; no workload layout is selected here.
"""

from __future__ import annotations

from typing import Literal

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
import cutlass.experimental.primitives as prims

from .affine_recurrence_primitives import fmul2
from .affine_recurrence_primitives import pack_input_b16x2_to_i32
from .warp_specialized_primitives import elect_arrive_mbarrier
from .warp_specialized_primitives import matrix_16x16_transposed_lane_coordinates
from .warp_specialized_primitives import segmented_swizzle_b16_element_index


@cute.jit
def _prepare_descriptor_issue(
    a,
    b,
    accumulator,
    operation,
    B_SWIZZLE: cutlass.Constexpr[int] = 128,
    B_MAJOR: cutlass.Constexpr[Literal[0, 1]] = 0,
    B_BASE_BYTES: cutlass.Constexpr[int] = 0,
):
    """Original descriptor construction, after the selected input wait/fence."""
    raw, stage, origin, stride = accumulator
    column = origin + stage * stride
    accumulator = cutlass.inttoptr(raw + column, 6, cutlass.Float32)
    dtype, rows, columns = operation
    operation = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=dtype,
        b_dtype=dtype,
        n_dim=columns,
        m_dim=rows,
        b_major=B_MAJOR,
    )
    pointer, leading, stride = b
    assert B_SWIZZLE in (32, 128)
    assert B_MAJOR in (0, 1) and B_BASE_BYTES >= 0
    b = prims.Tcgen05SmemDesc.build(
        pointer.subview(0),
        leading_byte_offset=leading,
        stride_byte_offset=stride,
        layout=(
            prims.Tcgen05SmemSwizzle.SWIZZLE_128B
            if B_SWIZZLE == 128
            else prims.Tcgen05SmemSwizzle.SWIZZLE_32B
        ),
    )
    if cutlass.const_expr(B_BASE_BYTES):
        b = b.advance_start_address(B_BASE_BYTES)
    return a, b, accumulator, operation


@cute.jit
def _read_companion(companion):
    """Original transposed x4 pointer construction and shared load, after LD."""
    pointer, row_base, lane, geometry = companion
    rows, columns, group, mask = geometry
    row, column = matrix_16x16_transposed_lane_coordinates(lane)
    column = row_base + column
    index = segmented_swizzle_b16_element_index(row, column, rows, columns, group, mask)
    return prims.ldmatrix(pointer.subview(index).data_ptr(), 4, prims.MMALayout.COL)


@cute.jit
def execute_prepared_wait(completion, phase, DESCRIPTOR: cutlass.Constexpr[bool]):
    """Complete the original event at its original role-local program point."""
    if cutlass.const_expr(DESCRIPTOR):
        while not prims.mbarrier_wait_parity(completion, phase, prims.MBarrierWait.TRY):
            pass
    else:
        cute.arch.mbarrier_wait(completion, phase)


@cute.jit
def _issue_atom(
    a,
    b,
    accumulator,
    operation,
    k,
    scale_d,
    DESCRIPTOR: cutlass.Constexpr[bool],
    TMEM_A: cutlass.Constexpr[bool],
    K_ATOM: cutlass.Constexpr[int],
    B_ADVANCE: cutlass.Constexpr,
):
    if cutlass.const_expr(DESCRIPTOR):
        raw, column = a
        element_bytes, phases, tile_rows, swizzle_bytes = B_ADVANCE
        offset = (k % phases) * K_ATOM * element_bytes + (
            k // phases
        ) * tile_rows * swizzle_bytes
        tmem_input = prims.make_tmem_ptr(raw, cutlass.Int8).subview(
            column + k * (K_ATOM // 2)
        )
        if prims.elect_sync():
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                accumulator,
                tmem_input,
                b.advance_start_address(offset),
                operation,
                scale_d,
            )
    else:
        if cutlass.const_expr(TMEM_A):
            lhs = a[None, None, k, 0]
        else:
            lhs = a[None, None, k]
        cute.gemm(operation, accumulator, lhs, b[None, None, k], accumulator)


@cute.jit
def execute_prepared_issue(
    a,
    b,
    accumulator,
    operation,
    issuer,
    completion,
    completion_phase,
    input_ready,
    input_phase,
    DESCRIPTOR: cutlass.Constexpr[bool],
    TMEM_A: cutlass.Constexpr[bool],
    K_BEGIN: cutlass.Constexpr[int],
    K_END: cutlass.Constexpr[int],
    K_ATOM: cutlass.Constexpr[int],
    B_ADVANCE: cutlass.Constexpr,
    INITIALIZED: cutlass.Constexpr[bool],
    COMMIT: cutlass.Constexpr[bool],
    WAIT_AFTER: cutlass.Constexpr[bool],
    B_SWIZZLE: cutlass.Constexpr[int] = 128,
    B_MAJOR: cutlass.Constexpr[Literal[0, 1]] = 0,
    B_BASE_BYTES: cutlass.Constexpr[int] = 0,
):
    """Original input wait, ordered issue/accumulation, commit and optional wait.

    A split range can commit only on its final span. A different reader role
    waits later in execute_prepared_read. No issue completion is inferred from
    the return of an asynchronous span. The caller retains its original raw
    input ring wait, outside this local packed-operand readiness event.
    """
    assert K_BEGIN >= 0 and K_END > K_BEGIN
    assert not WAIT_AFTER or COMMIT
    if cutlass.const_expr(input_ready is not None):
        assert DESCRIPTOR
        execute_prepared_wait(input_ready, input_phase, True)
        input_phase = input_phase ^ cutlass.Int32(1)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    # Carry only the typed fragment atom through the issuer region. A caller's
    # pre-existing ThrMma slice shares its trait; branch-local mutation must not
    # leave that alias holding an SSA value defined inside the issuer predicate.
    fragment_operation = None
    if cutlass.const_expr(not DESCRIPTOR):
        fragment_operation = operation
    if issuer:
        if cutlass.const_expr(DESCRIPTOR):
            issue_a, issue_b, issue_accumulator, issue_operation = (
                _prepare_descriptor_issue(
                    a, b, accumulator, operation, B_SWIZZLE, B_MAJOR, B_BASE_BYTES
                )
            )
        else:
            assert fragment_operation is not None
            issue_a, issue_b, issue_accumulator, issue_operation = (
                a,
                b,
                accumulator,
                fragment_operation,
            )
            fragment_operation.set(tcgen05.Field.ACCUMULATE, INITIALIZED)
        for k in cutlass.range_constexpr(K_BEGIN, K_END):
            _issue_atom(
                issue_a,
                issue_b,
                issue_accumulator,
                issue_operation,
                k,
                INITIALIZED or k != K_BEGIN,
                DESCRIPTOR,
                TMEM_A,
                K_ATOM,
                B_ADVANCE,
            )
            if cutlass.const_expr(not DESCRIPTOR):
                assert fragment_operation is not None
                fragment_operation.set(tcgen05.Field.ACCUMULATE, True)
        if cutlass.const_expr(COMMIT):
            if cutlass.const_expr(DESCRIPTOR):
                if prims.elect_sync():
                    prims.tcgen05_commit(completion, group=prims.CTAGroup.CTA_1)
            else:
                with cute.arch.elect_one():
                    tcgen05.commit(completion)
    if cutlass.const_expr(WAIT_AFTER):
        assert not DESCRIPTOR
        execute_prepared_wait(completion, completion_phase, False)
    return input_phase


@cute.jit
def _continuation_control(
    event,
    phase,
    issuer,
    OPCODE: cutlass.Constexpr[int],
    FENCE_AFTER: cutlass.Constexpr[bool],
    DESCRIPTOR: cutlass.Constexpr[bool],
):
    """Closed native wait/arrival/commit instructions, not a phase callback."""
    assert OPCODE in (0, 1, 3, 6)
    if cutlass.const_expr(OPCODE in (0, 1)):
        execute_prepared_wait(event, phase, DESCRIPTOR)
        if cutlass.const_expr(OPCODE == 1):
            phase = phase ^ cutlass.Int32(1)
        if cutlass.const_expr(FENCE_AFTER):
            assert DESCRIPTOR
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    elif cutlass.const_expr(OPCODE == 3):
        assert DESCRIPTOR and not FENCE_AFTER
        elect_arrive_mbarrier(event)
    else:
        assert not FENCE_AFTER
        if issuer:
            if cutlass.const_expr(DESCRIPTOR):
                if prims.elect_sync():
                    prims.tcgen05_commit(event, group=prims.CTAGroup.CTA_1)
            else:
                with cute.arch.elect_one():
                    tcgen05.commit(event)
    return phase


@cute.jit
def execute_prepared_continuation(
    PROGRAM: cutlass.Constexpr,
    issue_ports,
    events,
    phases,
    iteration,
    issuer,
    DESCRIPTOR: cutlass.Constexpr[bool],
):
    """Execute only the closed action payload lowered by the shared BodyProgram.

    The original caller binds every pointer, generation, native map and issuer.
    A previous-iteration branch can only wait/release; it cannot manufacture a
    seed or change a current generation. Return values are updated wait phases,
    not GPU completion or permission to reuse an accumulator.
    """
    current_phases = list(phases)
    for action_index in cutlass.range_constexpr(len(PROGRAM)):
        instruction = PROGRAM[action_index]
        if cutlass.const_expr(instruction[0] == 5):
            assert DESCRIPTOR
            if iteration > 0:
                previous_iteration = iteration - cutlass.Int32(1)
                for previous_index in cutlass.range_constexpr(len(instruction[1])):
                    previous = instruction[1][previous_index]
                    assert previous[0] in (0, 3)
                    previous_base, previous_period = events[previous[1]]
                    _continuation_control(
                        previous_base.subview(previous_iteration % previous_period),
                        (previous_iteration // previous_period) % 2,
                        issuer,
                        previous[0],
                        False,
                        DESCRIPTOR,
                    )
        elif cutlass.const_expr(instruction[0] == 4):
            port = issue_ports[instruction[1]]
            execute_prepared_issue(
                port[0],
                port[1],
                port[2],
                port[3],
                issuer,
                port[4],
                port[5],
                None,
                None,
                DESCRIPTOR,
                port[6],
                port[7] + instruction[2],
                port[7] + instruction[3],
                port[8],
                port[9],
                instruction[4],
                instruction[5],
                instruction[6],
                port[10],
            )
        else:
            current_phases[instruction[1]] = _continuation_control(
                events[instruction[1]],
                current_phases[instruction[1]],
                issuer,
                instruction[0],
                instruction[2],
                DESCRIPTOR,
            )
    return tuple(current_phases)


@cute.jit
def execute_prepared_read(
    source,
    values,
    copy,
    companion,
    completion,
    phase,
    DESCRIPTOR: cutlass.Constexpr[bool],
    LOAD_SHAPE: cutlass.Constexpr,
    LOAD_COUNT: cutlass.Constexpr[int],
):
    """Wait/read the original FP32 result, including its ordered companion load.

    Original typed point math remains at the callsite after this read and
    before execute_prepared_publication. Reading alone never publishes or
    releases the destination owner.
    """
    if cutlass.const_expr(DESCRIPTOR):
        # A role-local caller may wait before its original pointer construction.
        # That wait is the same bound operation, not a second readiness owner.
        if cutlass.const_expr(completion is not None):
            execute_prepared_wait(completion, phase, True)
        values = prims.tcgen05_ld(LOAD_SHAPE, source, num=LOAD_COUNT)
        if cutlass.const_expr(companion is not None):
            companion = _read_companion(companion)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    else:
        assert companion is None and completion is None
        cute.copy(copy, source, values)
        cute.arch.fence_view_async_tmem_load()
    return values, companion


@cute.jit
def pack_layout_f_state(values, dtype: cutlass.Constexpr):
    """Original 32-FP32 to 16-packed-word Layout-F register permutation.

    No memory, synchronization or source-owner replacement occurs here. The
    original FP32 values remain available to their independent scale consumer.
    """
    packed = cutlass.Array(cutlass.Int32, 16, alignment=16)
    for repeat in cutlass.range_constexpr(8):
        source = (repeat ^ 1) * 4
        destination = repeat * 2
        packed[destination] = pack_input_b16x2_to_i32(
            values[source], values[source + 1], dtype
        )
        packed[destination + 1] = pack_input_b16x2_to_i32(
            values[source + 2], values[source + 3], dtype
        )
    return packed


@cute.jit
def scale_layout_f_state(values, coefficients, KEY_BEGIN: cutlass.Constexpr[int]):
    """Original same-coordinate FP32 scale, while a packed store may pend.

    The caller binds the original complete coefficient owner/readiness and
    keeps the exact load alignment and fmul2 primitive. No cast or reduction
    is introduced by this arithmetic leaf.
    """
    pointer = coefficients.data_ptr()
    scaled = cutlass.Array(cutlass.Float32, 32, alignment=16)
    lane_pair = (cute.arch.lane_idx() % 4) * 2
    for repeat in cutlass.range_constexpr(8):
        register = repeat * 4
        column = KEY_BEGIN + repeat * 8 + lane_pair
        scale = (pointer + column).load(count=2, alignment=8)
        scaled[register], scaled[register + 1] = fmul2(
            (values[register], values[register + 1]), (scale[0], scale[1])
        )
        scaled[register + 2], scaled[register + 3] = fmul2(
            (values[register + 2], values[register + 3]), (scale[0], scale[1])
        )
    return scaled


@cute.jit
def execute_prepared_store(
    values,
    destination,
    copy,
    DESCRIPTOR: cutlass.Constexpr[bool],
    STORE_SHAPE: cutlass.Constexpr,
):
    """Issue only the already-bound store; its owner retains completion debt.

    In particular, descriptor snapshot stores may remain outstanding while
    the original FP32 registers are scaled. A returned store is not permission
    to publish readiness or recycle either owner.
    """
    if cutlass.const_expr(DESCRIPTOR):
        assert copy is None
        prims.tcgen05_st(STORE_SHAPE, destination, values)
    else:
        cute.copy(copy, values, destination)


@cute.jit
def execute_prepared_store_completion(
    DESCRIPTOR: cutlass.Constexpr[bool],
):
    """Retire the original store and perform its original visibility fence."""
    if cutlass.const_expr(DESCRIPTOR):
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    else:
        cute.arch.fence_view_async_tmem_store()


@cute.jit
def execute_prepared_publication(
    values,
    destination,
    copy,
    participant_barrier,
    arrival,
    DESCRIPTOR: cutlass.Constexpr[bool],
    STORE_SHAPE: cutlass.Constexpr,
):
    """Original native publication, completion, visibility and role contribution."""
    if cutlass.const_expr(DESCRIPTOR):
        assert participant_barrier is None
        execute_prepared_store(values, destination, copy, True, STORE_SHAPE)
        execute_prepared_store_completion(True)
        if prims.elect_sync():
            prims.mbarrier_arrive(arrival)
    else:
        assert arrival is None
        participant_barrier.arrive_and_wait()
        execute_prepared_store(values, destination, copy, False, STORE_SHAPE)
        execute_prepared_store_completion(False)
        participant_barrier.arrive_and_wait()
