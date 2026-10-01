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

from typing import Any
from typing import Literal
from typing import cast

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
import cutlass.experimental.primitives as prims

from .affine_recurrence_primitives import ffma2
from .affine_recurrence_primitives import fmul2
from .affine_recurrence_primitives import materialize_fp32
from .affine_recurrence_primitives import pack_bf16x2_inline
from .affine_recurrence_primitives import pack_input_b16x2_to_i32
from .affine_recurrence_primitives import pack_typed_b16x2_to_i32
from .warp_specialized_primitives import arrive_mbarrier
from .warp_specialized_primitives import elect_arrive_mbarrier
from .warp_specialized_primitives import fence_async_shared
from .warp_specialized_primitives import matrix_16x16_transposed_lane_coordinates
from .warp_specialized_primitives import named_barrier_sync
from .warp_specialized_primitives import segmented_swizzle_b16_element_index
from .warp_specialized_primitives import swizzle_b16_index


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
    ISSUER_ELECTED: cutlass.Constexpr[bool] = False,
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
        elected = True
        if cutlass.const_expr(not ISSUER_ELECTED):
            elected = prims.elect_sync()
        if elected:
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
    ISSUER_ELECTED: cutlass.Constexpr[bool] = False,
):
    """Original input wait, ordered issue/accumulation, commit and optional wait.

    A split range can commit only on its final span. A different reader role
    waits later in execute_prepared_read. No issue completion is inferred from
    the return of an asynchronous span. The caller retains its original raw
    input ring wait, outside this local packed-operand readiness event.
    ISSUER_ELECTED retains an existing whole-span election supplied by the
    physical caller; the default preserves the original per-atom election.
    """
    assert not ISSUER_ELECTED or DESCRIPTOR
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
                ISSUER_ELECTED,
            )
            if cutlass.const_expr(not DESCRIPTOR):
                assert fragment_operation is not None
                fragment_operation.set(tcgen05.Field.ACCUMULATE, True)
        if cutlass.const_expr(COMMIT):
            if cutlass.const_expr(DESCRIPTOR):
                elected = True
                if cutlass.const_expr(not ISSUER_ELECTED):
                    elected = prims.elect_sync()
                if elected:
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
    COMPANION_TMEM: cutlass.Constexpr[bool] = False,
    DEFER_WAIT: cutlass.Constexpr[bool] = False,
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
            if cutlass.const_expr(COMPANION_TMEM):
                companion = prims.tcgen05_ld(LOAD_SHAPE, companion, num=LOAD_COUNT)
            else:
                companion = _read_companion(companion)
        if cutlass.const_expr(not DEFER_WAIT):
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    else:
        assert companion is None and completion is None
        cute.copy(copy, source, values)
        cute.arch.fence_view_async_tmem_load()
    return values, companion


@cute.jit
def execute_prepared_capture_release(
    empty, issuer, BARRIER: cutlass.Constexpr[int], COUNT: cutlass.Constexpr[int]
):
    """Release a captured native value after all original readers have joined."""
    named_barrier_sync(BARRIER, COUNT)
    if issuer:
        arrive_mbarrier(empty)


@cute.jit
def pack_output_half(values, CAST: cutlass.Constexpr = None):
    packed = cutlass.Array(cutlass.Int32, 8, alignment=16)
    for pair in cutlass.range_constexpr(8):
        if cutlass.const_expr(CAST is not None):
            packed[pair] = pack_typed_b16x2_to_i32(
                CAST(values[pair * 2]), CAST(values[pair * 2 + 1]), cutlass.BFloat16
            )
        else:
            packed[pair] = pack_bf16x2_inline(values[pair * 2], values[pair * 2 + 1])
    return packed


@cute.jit
def store_output_half(
    destination,
    packed,
    row_block,
    lane,
    HALF: cutlass.Constexpr,
    MAP: cutlass.Constexpr,
):
    row, column = matrix_16x16_transposed_lane_coordinates(lane)
    for token_group in cutlass.range_constexpr(2):
        value = row_block * 32 + HALF * 16 + column
        token = token_group * 16 + row
        offset = swizzle_b16_index(token, value, *MAP)
        prims.stmatrix(
            destination + offset // 2,
            [
                packed[token_group * 4],
                packed[token_group * 4 + 1],
                packed[token_group * 4 + 2],
                packed[token_group * 4 + 3],
            ],
            prims.MMALayout.COL,
            shape=prims.StoreShape.M8N8,
        )


@cute.jit
def execute_prepared_output(
    PROGRAM: cutlass.Constexpr,
    source,
    companion,
    destination,
    empty,
    out,
    descriptor,
    begin,
    seqlen,
    head,
    iteration,
    row_block,
    lane,
    STATE_PUBLICATIONS: cutlass.Constexpr = None,
):
    """Execute scheduled capture, conversion, publication and completion leaves.

    No loop/event generation or pointer ownership is selected here. Capture
    and EMPTY release precede the original two-slot TMA reuse wait; conversion
    occurs only after that wait, with the original per-half register lifetime.
    """
    output_cast = cast("Any", None)
    if cutlass.const_expr(STATE_PUBLICATIONS is not None):
        output_cast = STATE_PUBLICATIONS[5]
    values = cast("Any", None)
    second = None
    packed = cast("Any", None)
    for index in cutlass.range_constexpr(len(PROGRAM)):
        instruction = PROGRAM[index]
        if cutlass.const_expr(instruction[0] == 0):
            values, second = execute_prepared_read(
                source,
                None,
                None,
                companion,
                None,
                None,
                instruction[1],
                instruction[2],
                instruction[3],
                instruction[5],
            )
        elif cutlass.const_expr(instruction[0] == 3):
            if cutlass.const_expr(instruction[1] == 0):
                packed = pack_output_half(values, output_cast)
            else:
                packed = pack_output_half(second, output_cast)
        elif cutlass.const_expr(instruction[0] == 1):
            store_output_half(
                destination, packed, row_block, lane, instruction[1], instruction[2:]
            )
        elif cutlass.const_expr(instruction[0] == 4):
            execute_prepared_capture_release(
                empty, row_block == 0, instruction[1], instruction[2]
            )
        elif cutlass.const_expr(instruction[0] == 5):
            if row_block == 0:
                if iteration >= instruction[3]:
                    prims.cp_async_bulk_wait_group(instruction[4], read=True)
            named_barrier_sync(instruction[1], instruction[2])
        elif cutlass.const_expr(instruction[0] == 2):
            named_barrier_sync(instruction[1], instruction[2])
            if row_block == 0:
                fence_async_shared()
                if prims.elect_sync():
                    execute_prepared_tma_segment(
                        destination,
                        descriptor,
                        (0, cutlass.Int32(begin) + iteration * instruction[3], head, 0),
                    )
                    complete_prepared_tma_store(False)
        else:
            assert instruction[0] == 6
            for column in cutlass.range_constexpr(instruction[1]):
                token = iteration * instruction[1] + column
                if token < seqlen:
                    if cutlass.const_expr(output_cast is not None):
                        converted = output_cast(values[column])
                    else:
                        converted = cutlass.BFloat16(values[column])
                    store_prepared_output_element(
                        out,
                        (0, begin + cutlass.Int64(token), head, row_block * 32 + lane),
                        converted,
                    )


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
    FENCE: cutlass.Constexpr[bool] = True,
):
    """Retire the original store and perform its original visibility fence."""
    if cutlass.const_expr(DESCRIPTOR):
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        if cutlass.const_expr(FENCE):
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    else:
        cute.arch.fence_view_async_tmem_store()


@cute.jit
def read_linear_state_abi(
    source,
    coordinates,
    BEGIN: cutlass.Constexpr,
    COUNT: cutlass.Constexpr,
    valid=True,
):
    """Read exact external FP32 state, or the original absent-input zeros."""
    values = cutlass.Array(cutlass.Float32, COUNT, alignment=16)
    for column in cutlass.range_constexpr(COUNT):
        value = cutlass.Float32(0.0)
        if cutlass.const_expr(source is not None):
            if valid:
                value = source[
                    coordinates[0], coordinates[1], coordinates[2], BEGIN + column
                ]
        values[column] = value
    return values


@cute.jit
def store_linear_state_abi(
    values,
    destination,
    coordinates,
    BEGIN: cutlass.Constexpr,
    COUNT: cutlass.Constexpr,
    valid=True,
):
    for column in cutlass.range_constexpr(COUNT):
        if valid:
            destination[
                coordinates[0], coordinates[1], coordinates[2], BEGIN + column
            ] = values[column]


@cute.jit
def execute_prepared_state_abi(
    PROGRAM: cutlass.Constexpr, initial, final, state, coordinates, valid=True
):
    """Original root ABI effects; no allocation, loop or DONE authority."""
    values = cast("Any", None)
    for index in cutlass.range_constexpr(len(PROGRAM)):
        instruction = PROGRAM[index]
        if cutlass.const_expr(instruction[0] == 0):
            if cutlass.const_expr(instruction[1] == 1):
                values, companion = execute_prepared_read(
                    state + instruction[2] * instruction[5],
                    None,
                    None,
                    None,
                    None,
                    None,
                    instruction[3],
                    instruction[4],
                    instruction[5],
                )
            else:
                values = read_linear_state_abi(
                    initial,
                    coordinates,
                    instruction[2] * instruction[5],
                    instruction[5],
                    valid,
                )
        elif cutlass.const_expr(instruction[0] == 1):
            if cutlass.const_expr(instruction[1] == 0):
                execute_prepared_store(
                    values[0 : instruction[5]],
                    state + instruction[2] * instruction[5],
                    None,
                    instruction[3],
                    instruction[6],
                )
            else:
                store_linear_state_abi(
                    values,
                    final,
                    coordinates,
                    instruction[2] * instruction[5],
                    instruction[5],
                    valid,
                )
        else:
            assert instruction[0] == 2 and instruction[1] == 0
            # Original initialization completes stores without a publication
            # fence; later state-input publication retains that fence.
            execute_prepared_store_completion(instruction[3], False)


@cute.jit
def pack_linear_state(values, CAST: cutlass.Constexpr = None):
    """Original adjacent-pair BF16 packing, retaining the input FP32 registers."""
    packed = cutlass.Array(cutlass.Int32, 16, alignment=16)
    for pair in cutlass.range_constexpr(16):
        if cutlass.const_expr(CAST is not None):
            packed[pair] = pack_typed_b16x2_to_i32(
                CAST(values[pair * 2]), CAST(values[pair * 2 + 1]), cutlass.BFloat16
            )
        else:
            packed[pair] = pack_bf16x2_inline(values[pair * 2], values[pair * 2 + 1])
    return packed


@cute.jit
def pack_affine_residual_half(
    prediction,
    raw,
    coefficient,
    row,
    scale,
    remaining,
    HALF: cutlass.Constexpr[int],
    MASK_TAIL: cutlass.Constexpr[bool],
    ROWS: cutlass.Constexpr[int],
    STATE_PUBLICATIONS: cutlass.Constexpr = None,
):
    """Original affine residual rounding, with masked rows explicit before pack."""
    values = cutlass.Array(cutlass.Float32, 16, alignment=16)
    beta = cutlass.Array(cutlass.Float32, 16, alignment=16)
    for column in cutlass.range_constexpr(16):
        token = HALF * 16 + column
        values[column] = cutlass.Float32((raw + token * ROWS + row).load())
        beta[column] = (coefficient + token).load()
    packed = cutlass.Array(cutlass.Int32, 8, alignment=16)
    for pair in cutlass.range_constexpr(8):
        token = HALF * 16 + pair * 2
        if cutlass.const_expr(STATE_PUBLICATIONS is not None):
            low, high = STATE_PUBLICATIONS[1](
                (prediction[token], prediction[token + 1]),
                (values[pair * 2], values[pair * 2 + 1]),
                (beta[pair * 2], beta[pair * 2 + 1]),
                scale,
            )
        else:
            low, high = ffma2(
                (prediction[token], prediction[token + 1]),
                (-scale, -scale),
                (values[pair * 2], values[pair * 2 + 1]),
            )
            low, high = fmul2((low, high), (beta[pair * 2], beta[pair * 2 + 1]))
        if cutlass.const_expr(MASK_TAIL):
            if token >= remaining:
                low = cutlass.Float32(0.0)
            if token + 1 >= remaining:
                high = cutlass.Float32(0.0)
        if cutlass.const_expr(STATE_PUBLICATIONS is not None):
            if cutlass.const_expr(MASK_TAIL):
                low = materialize_fp32(low)
                high = materialize_fp32(high)
            packed[pair] = pack_typed_b16x2_to_i32(
                STATE_PUBLICATIONS[3](low),
                STATE_PUBLICATIONS[3](high),
                cutlass.BFloat16,
            )
        else:
            packed[pair] = pack_bf16x2_inline(low, high)
    return packed


@cute.jit
def execute_prepared_state_product(
    PROGRAM: cutlass.Constexpr,
    source,
    destination,
    raw,
    coefficient,
    row,
    scale,
    remaining,
    ready,
    STATE_PUBLICATIONS: cutlass.Constexpr = None,
):
    """Apply common state effects at the original solved-operand consumer cut."""
    values = cast("Any", None)
    packed = cast("Any", None)
    for index in cutlass.range_constexpr(len(PROGRAM)):
        instruction = PROGRAM[index]
        if cutlass.const_expr(instruction[0] == 0):
            values, unused_companion = execute_prepared_read(
                source,
                None,
                None,
                None,
                None,
                None,
                instruction[3],
                instruction[4],
                instruction[5],
            )
        elif cutlass.const_expr(instruction[0] == 3):
            packed = pack_affine_residual_half(
                values,
                raw,
                coefficient,
                row,
                scale,
                remaining,
                instruction[1],
                instruction[2],
                instruction[3],
                STATE_PUBLICATIONS,
            )
        elif cutlass.const_expr(instruction[0] == 4):
            if cutlass.const_expr(STATE_PUBLICATIONS is not None):
                packed = pack_linear_state(values, STATE_PUBLICATIONS[4])
            else:
                packed = pack_linear_state(values)
        elif cutlass.const_expr(instruction[0] == 1):
            execute_prepared_store(
                packed[0 : instruction[2]],
                destination + instruction[1],
                None,
                instruction[3],
                instruction[6],
            )
        elif cutlass.const_expr(instruction[0] == 2):
            execute_prepared_store_completion(instruction[3])
        else:
            assert instruction[0] == 5
            for event in cutlass.range_constexpr(instruction[1]):
                arrive_mbarrier(ready[event])


@cute.jit
def scale_linear_state(
    values, coefficients, STATE_PUBLICATIONS: cutlass.Constexpr = None
):
    """Original two16-column scalar coefficient reads and packed FP32 multiply."""
    scaled = cutlass.Array(cutlass.Float32, 32, alignment=16)
    for half in cutlass.range_constexpr(2):
        gamma = cutlass.Array(cutlass.Float32, 16, alignment=16)
        for column in cutlass.range_constexpr(16):
            gamma[column] = (coefficients + half * 16 + column).load()
        for pair in cutlass.range_constexpr(8):
            index = half * 16 + pair * 2
            if cutlass.const_expr(STATE_PUBLICATIONS is not None):
                lo, hi = STATE_PUBLICATIONS[0](
                    (values[index], values[index + 1]),
                    (gamma[pair * 2], gamma[pair * 2 + 1]),
                )
            else:
                lo, hi = fmul2(
                    (values[index], values[index + 1]),
                    (gamma[pair * 2], gamma[pair * 2 + 1]),
                )
            scaled[index] = lo
            scaled[index + 1] = hi
    return scaled


@cute.jit
def execute_prepared_linear_state(
    PROGRAM: cutlass.Constexpr,
    source,
    packed_target,
    coefficients,
    ready,
    STATE_PUBLICATIONS: cutlass.Constexpr = None,
):
    """Interpret bound state effects; native transfers share the root/DV2 leaves."""
    original = None
    packed = cast("Any", None)
    scaled = cast("Any", None)
    for index in cutlass.range_constexpr(len(PROGRAM)):
        instruction = PROGRAM[index]
        if cutlass.const_expr(instruction[0] == 0):
            original, companion = execute_prepared_read(
                source,
                None,
                None,
                None,
                None,
                None,
                instruction[2],
                instruction[3],
                instruction[4],
            )
        elif cutlass.const_expr(instruction[0] == 1):
            if cutlass.const_expr(instruction[1] == 1):
                execute_prepared_store(
                    packed[0:16],
                    packed_target,
                    None,
                    instruction[2],
                    instruction[5],
                )
            else:
                execute_prepared_store(
                    scaled[0:32],
                    source,
                    None,
                    instruction[2],
                    instruction[5],
                )
        elif cutlass.const_expr(instruction[0] == 2):
            execute_prepared_store_completion(instruction[2])
        elif cutlass.const_expr(instruction[0] == 3):
            if cutlass.const_expr(STATE_PUBLICATIONS is not None):
                packed = pack_linear_state(original, STATE_PUBLICATIONS[2])
            else:
                packed = pack_linear_state(original)
        elif cutlass.const_expr(instruction[0] == 4):
            scaled = scale_linear_state(original, coefficients, STATE_PUBLICATIONS)
        else:
            assert instruction[0] == 5
            arrive_mbarrier(ready)


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


@cute.jit
def execute_prepared_tma_segment(source, descriptor, coordinates):
    """Store one original segment under the caller's existing elected issuer."""
    prims.cp_async_bulk_tensor_global_shared_cta(
        descriptor.get_ptr(), source, coordinates
    )


@cute.jit
def complete_prepared_tma_store(WAIT: cutlass.Constexpr[bool]):
    """Commit the original store group; optionally retire its source read debt."""
    prims.cp_async_bulk_commit_group()
    if cutlass.const_expr(WAIT):
        prims.cp_async_bulk_wait_group(0, read=True)


@cute.jit
def store_prepared_output_element(destination, coordinates, value):
    """Store an already-typed original output at its selected external index."""
    destination[coordinates[0], coordinates[1], coordinates[2], coordinates[3]] = value
