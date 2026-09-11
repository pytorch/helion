# ruff: noqa: ANN001, ANN202
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""SM100 BT16 KDA recurrence schedule.

Adapted from FlashInfer's ``kda_chunked_bt16.py`` at pinned commit
``f67bc2ed555c1ad6a764ad68f7aa9622178e9eae``.  Helion owns the matcher,
workspace layout, schedule selection, and launch integration around this device
schedule.  The recurrence keeps FP32 state in TMEM and splits the 16 warps into
output-drain, state-left, state-right, two tcgen05 issuers, TMA, and service
roles.  The Helion schedule uses an eight-slot raw-input ring, a six-slot TMA
completion ring, two TMEM output accumulators, rank-4 factor loads, and a
seven-slot asynchronous output-SMEM ring.  These choices are explicit in the
wrapper plan, with workspace-layout and device-ABI checks kept separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import cast

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as prims


def packed_f32x2_binary(
    op: Callable,
    lhs: tuple[cutlass.Float32, cutlass.Float32],
    rhs: tuple[cutlass.Float32, cutlass.Float32],
) -> tuple[cutlass.Float32, cutlass.Float32]:
    """Apply a CUTLASS packed-FP32 primitive to two scalar pairs."""

    lhs_vec = cutlass.Vector.from_elements(lhs, cutlass.Float32)
    rhs_vec = cutlass.Vector.from_elements(rhs, cutlass.Float32)
    result = op(lhs_vec, rhs_vec, ftz=False, rnd="rn")
    return cutlass.Float32(result[0]), cutlass.Float32(result[1])


def fmul2(lhs, rhs):
    return packed_f32x2_binary(prims.mul_packed_f32x2, lhs, rhs)


@cute.jit
def sub_b16x2_input_dtype(
    lhs: cutlass.Int32,
    rhs: cutlass.Int32,
    input_dtype: cutlass.Constexpr,
) -> cutlass.Int32:
    """Subtract two packed pairs using the compile-time input dtype."""

    if cutlass.const_expr(input_dtype is cutlass.BFloat16):
        return cast(
            "cutlass.Int32",
            prims.inline_ptx_hl(
                "sub.bf16x2 {$w0}, {$r0}, {$r1};",
                write_only_types=[cutlass.Int32],
                read_only_args=[lhs, rhs],
            ),
        )
    return cast(
        "cutlass.Int32",
        prims.inline_ptx_hl(
            "sub.f16x2 {$w0}, {$r0}, {$r1};",
            write_only_types=[cutlass.Int32],
            read_only_args=[lhs, rhs],
        ),
    )


@cute.jit
def pack_input_b16x2_to_i32(
    value0: cutlass.Float32,
    value1: cutlass.Float32,
    input_dtype: cutlass.Constexpr,
):
    """Pack two FP32 values through the compile-time input 16-bit dtype."""

    return (
        cutlass.Vector.from_elements(
            (value0, value1),
            cutlass.Float32,
        )
        .to(input_dtype)
        .bitcast(cutlass.Int32)[0]
    )


BT: int = 16

DK: int = 128

DV: int = 128

THREADS_PER_WARP: int = 32

THREADS_PER_CTA: int = 16 * THREADS_PER_WARP

TMEM_USER_WARP_COUNT: int = 5

TMEM_USER_THREADS: int = TMEM_USER_WARP_COUNT * THREADS_PER_WARP

NBAR_TMEM_LIFECYCLE_ID: int = 2

NBAR_OUTPUT_DRAIN_ID: int = 8

OUTPUT_DRAIN_THREADS: int = 4 * THREADS_PER_WARP

KDA_CG1_REGS: int = 136

KDA_SERVICE_REGS: int = 56

TCGEN05_F16_K_ATOM: int = 16

TCGEN05_F16_ELEM_BYTES: int = 2

TCGEN05_SW128_BYTES: int = 128

TCGEN05_SW128_K_PHASES_PER_SLICE: int = 4

TCGEN05_STATE_K_B_LEADING_BYTES: int = 16

TCGEN05_STATE_K_B_STRIDE_BYTES: int = 1024

TCGEN05_STATE_K_B_K_STEP_BYTES: int = TCGEN05_F16_K_ATOM * TCGEN05_F16_ELEM_BYTES

TCGEN05_STATE_INPUT_LOAD_COLS: int = 16

TCGEN05_STATE_INPUT_PACKED_COLS: int = TCGEN05_STATE_INPUT_LOAD_COLS // 2

TCGEN05_STATE_K_TMEM_ROW_BLOCKS: int = DV // THREADS_PER_WARP


def _tcgen05_accumulator_tmem_cols(n_dim: int) -> int:
    """Return TMEM columns for an FP32 `[128, n_dim]` accumulator tile."""

    if n_dim <= 0:
        raise ValueError(f"n_dim must be positive, got {n_dim}")
    if n_dim % 8 != 0:
        raise ValueError(f"n_dim must be a multiple of 8, got {n_dim}")
    return n_dim


def _tcgen05_f16_input_tmem_cols(k_dim: int) -> int:
    """Return TMEM columns for an F16 `[128, k_dim]` A-input staging tile."""

    if k_dim <= 0:
        raise ValueError(f"k_dim must be positive, got {k_dim}")
    if k_dim % 2 != 0:
        raise ValueError(f"k_dim must be even for packed F16 TMEM, got {k_dim}")
    return k_dim // 2


KDA_TMEM_N16_ACC_COLS: int = _tcgen05_accumulator_tmem_cols(BT)

KDA_TMEM_N128_ACC_COLS: int = _tcgen05_accumulator_tmem_cols(DK)

KDA_TMEM_STATE_COLS: int = KDA_TMEM_N128_ACC_COLS

KDA_TMEM_STATE_AS_INPUT_COLS: int = _tcgen05_f16_input_tmem_cols(DK)

KDA_TMEM_SHARED_INPUT_COLS: int = _tcgen05_f16_input_tmem_cols(BT)

KDA_TMEM_SHARED_INPUT_STAGE_COUNT: int = 2

KDA_TMEM_SHARED_ACC_STAGE_COUNT: int = 2

KDA_TMEM_STATE_COL_OFFSET: int = 0

KDA_TMEM_STATE_AS_INPUT_COL_OFFSET: int = (
    KDA_TMEM_STATE_COL_OFFSET + KDA_TMEM_STATE_COLS
)

KDA_TMEM_SHARED_INPUT_COL_OFFSET: int = (
    KDA_TMEM_STATE_AS_INPUT_COL_OFFSET + KDA_TMEM_STATE_AS_INPUT_COLS
)

KDA_TMEM_QSTATE_ACC_COL_OFFSET: int = (
    KDA_TMEM_SHARED_INPUT_COL_OFFSET
    + KDA_TMEM_SHARED_INPUT_STAGE_COUNT * KDA_TMEM_SHARED_INPUT_COLS
)

KDA_TMEM_SHARED_ACC_COL_OFFSET: int = (
    KDA_TMEM_QSTATE_ACC_COL_OFFSET + KDA_TMEM_N16_ACC_COLS
)

KDA_TMEM_QSTATE_ACC_STAGE1_COL_OFFSET: int = (
    KDA_TMEM_SHARED_ACC_COL_OFFSET
    + KDA_TMEM_SHARED_ACC_STAGE_COUNT * KDA_TMEM_N16_ACC_COLS
)

KDA_TMEM_QSTATE_ACC_STAGE_STRIDE_COLS: int = (
    KDA_TMEM_QSTATE_ACC_STAGE1_COL_OFFSET - KDA_TMEM_QSTATE_ACC_COL_OFFSET
)


@cute.jit
def cta_sync() -> None:
    """Synchronize all threads in the CTA."""

    prims.barrier_cta_sync(0, thread_count=THREADS_PER_CTA)


@cute.jit
def tmem_user_sync() -> None:
    """Named barrier for CG1 plus the tcgen05 warp during TMEM lifecycle setup."""

    prims.barrier_cta_sync(NBAR_TMEM_LIFECYCLE_ID, thread_count=TMEM_USER_THREADS)


@cute.jit
def output_drain_sync() -> None:
    """Synchronize the four warps that cooperatively drain output TMEM."""

    prims.barrier_cta_sync(NBAR_OUTPUT_DRAIN_ID, thread_count=OUTPUT_DRAIN_THREADS)


@cute.jit
def is_compute_group0_warp(warp_idx) -> cutlass.Boolean:
    """Return whether this warp belongs to CG0 preprocessing."""

    return (warp_idx >= ROLES.compute_group0_first) & (
        warp_idx <= ROLES.compute_group0_last
    )


@cute.jit
def is_compute_group1_warp(warp_idx) -> cutlass.Boolean:
    """Return whether this warp belongs to CG1 value/final-state work."""

    return (warp_idx >= ROLES.compute_group1_first) & (
        warp_idx <= ROLES.compute_group1_last
    )


@cute.jit
def is_tmem_user_warp(warp_idx) -> cutlass.Boolean:
    """Return whether this warp needs the allocated TMEM base pointer."""

    return is_compute_group1_warp(warp_idx) | (warp_idx == ROLES.tcgen05_mma)


@cute.jit
def is_service_warpgroup(warp_idx) -> cutlass.Boolean:
    """Return whether this warp belongs to the non-CG0/CG1 service warpgroup."""

    return (warp_idx >= ROLES.super_mma) & (warp_idx <= ROLES.epilogue)


O_OUT_OFFSET: int = 0

O_ELEM_BYTES: int = TCGEN05_F16_ELEM_BYTES

O_TMA_SWIZZLE_BYTES: int = 128

O_TMA_SWIZZLE_ELEMS: int = O_TMA_SWIZZLE_BYTES // O_ELEM_BYTES

O_TMA_SWIZZLE_GROUP_BYTES: int = 16

O_TMA_SWIZZLE_GROUP_ELEMS: int = O_TMA_SWIZZLE_GROUP_BYTES // O_ELEM_BYTES

O_TMA_SWIZZLE_ROW_MASK: int = (O_TMA_SWIZZLE_ELEMS // O_TMA_SWIZZLE_GROUP_ELEMS) - 1

O_TMA_SWIZZLE_ALIGNMENT_BYTES: int = O_TMA_SWIZZLE_BYTES * (
    O_TMA_SWIZZLE_ELEMS // O_TMA_SWIZZLE_GROUP_ELEMS
)

TCGEN05_VALUE_PAIRWISE_B_LEADING_BYTES: int = 16

TCGEN05_VALUE_PAIRWISE_B_STRIDE_BYTES: int = 8 * BT * TCGEN05_F16_ELEM_BYTES

TCGEN05_FINAL_STATE_B_N_GROUP_ELEMS: int = TCGEN05_SW128_BYTES // TCGEN05_F16_ELEM_BYTES

TCGEN05_FINAL_STATE_B_LEADING_BYTES: int = (
    BT * TCGEN05_FINAL_STATE_B_N_GROUP_ELEMS * TCGEN05_F16_ELEM_BYTES
)

TCGEN05_FINAL_STATE_B_STRIDE_BYTES: int = (
    8 * TCGEN05_FINAL_STATE_B_N_GROUP_ELEMS * TCGEN05_F16_ELEM_BYTES
)

TCGEN05_FINAL_STATE_TMEM_LOAD_COLS: int = 32

RAW_F16_TMA_SWIZZLE_BYTES: int = 128

RAW_F16_TMA_SWIZZLE_ELEMS: int = RAW_F16_TMA_SWIZZLE_BYTES // TCGEN05_F16_ELEM_BYTES

RAW_F16_TMA_SWIZZLE_GROUP_BYTES: int = 16

RAW_F16_TMA_SWIZZLE_GROUP_ELEMS: int = (
    RAW_F16_TMA_SWIZZLE_GROUP_BYTES // TCGEN05_F16_ELEM_BYTES
)

RAW_F16_TMA_SWIZZLE_ROW_MASK: int = (
    RAW_F16_TMA_SWIZZLE_ELEMS // RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
) - 1

RAW_F16_TMA_SEGMENTS: int = DK // RAW_F16_TMA_SWIZZLE_ELEMS

RAW_F16_TMA_SEGMENT_ELEMS: int = BT * RAW_F16_TMA_SWIZZLE_ELEMS

RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES: int = RAW_F16_TMA_SWIZZLE_BYTES * (
    RAW_F16_TMA_SWIZZLE_ELEMS // RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
)


@dataclass(frozen=True)
class WarpRoles:
    """Warp assignment for the BT=16 fully fused KDA kernel."""

    compute_group0_first: int = 0
    compute_group0_last: int = 7
    compute_group1_first: int = 8
    compute_group1_last: int = 11
    super_mma: int = 12
    tcgen05_mma: int = 13
    tma_load: int = 14
    epilogue: int = 15


ROLES = WarpRoles()


@cute.jit
def raw_f16_s128_smem_index(token_coord, dim):
    """Return the physical s128 SMEM index for raw F16 q/k/v staging."""

    segment = dim // RAW_F16_TMA_SWIZZLE_ELEMS
    segment_dim = dim - segment * RAW_F16_TMA_SWIZZLE_ELEMS
    col_group = segment_dim // RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
    col_in_group = segment_dim - col_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
    row_swizzle = token_coord & RAW_F16_TMA_SWIZZLE_ROW_MASK
    return (
        segment * RAW_F16_TMA_SEGMENT_ELEMS
        + token_coord * RAW_F16_TMA_SWIZZLE_ELEMS
        + ((col_group ^ row_swizzle) * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS)
        + col_in_group
    )


@cute.jit
def o_smem_swizzle_128b_elem_index(
    o_stage_base,
    value_dim,
    token_coord,
):
    """Return the physical W128 SMEM index for one staged output element."""

    segment = value_dim // O_TMA_SWIZZLE_ELEMS
    segment_value_dim = value_dim - segment * O_TMA_SWIZZLE_ELEMS
    col_group = segment_value_dim // O_TMA_SWIZZLE_GROUP_ELEMS
    col_in_group = segment_value_dim - col_group * O_TMA_SWIZZLE_GROUP_ELEMS
    row_swizzle = token_coord & O_TMA_SWIZZLE_ROW_MASK
    return (
        o_stage_base
        + O_OUT_OFFSET
        + segment * BT * O_TMA_SWIZZLE_ELEMS
        + token_coord * O_TMA_SWIZZLE_ELEMS
        + ((col_group ^ row_swizzle) * O_TMA_SWIZZLE_GROUP_ELEMS)
        + col_in_group
    )


@cute.jit
def o_smem_stmatrix_128b_ptr(
    o_smem,
    o_stage_base,
    value_dim_base,
    lane,
):
    """Return the per-lane W128 row-start pointer for one 16x16 STSM.T tile."""

    matrix_id = lane // 8
    row_in_matrix = lane & 7
    token_block = matrix_id // 2
    value_block = matrix_id & 1
    token_coord = token_block * 8 + row_in_matrix
    value_dim = value_dim_base + value_block * 8
    smem_idx = o_smem_swizzle_128b_elem_index(
        o_stage_base,
        value_dim,
        token_coord,
    )
    return o_smem.subview(smem_idx).data_ptr()


@cute.jit
def raw_v_ldmatrix_trans_ptr(raw_v_smem, value_dim_base, lane):
    """Return the per-lane row-start pointer for raw V `ldmatrix.x4.trans`."""

    matrix_id = lane // 8
    row_in_matrix = lane & 7
    token_block = matrix_id // 2
    value_block = matrix_id & 1
    token_coord = token_block * 8 + row_in_matrix
    value_dim = value_dim_base + value_block * 8
    smem_idx = raw_f16_s128_smem_index(token_coord, value_dim)
    return raw_v_smem.subview(smem_idx).data_ptr()


@cute.jit
def tma_transfer_wait(tma_mbar, tma_phase) -> None:
    """Spin until one tma_mbar ring slot's expect_tx transaction completes."""

    while not prims.mbarrier_wait_parity(
        tma_mbar,
        tma_phase,
        prims.MBarrierWait.TRY,
    ):
        pass


@cute.jit
def output_ready_arrive(output_ready_mbar) -> None:
    """Signal that this CG1 warp has finished staging output SMEM."""

    if prims.elect_sync():
        prims.mbarrier_arrive(output_ready_mbar)


@cute.jit
def output_ready_wait(output_ready_mbar, output_ready_phase):
    """Wait until all CG1 warps have staged the output tile."""

    while not prims.mbarrier_wait_parity(
        output_ready_mbar,
        output_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return output_ready_phase ^ cutlass.Int32(1)


@cute.jit
def raw_ready_arrive(raw_ready_mbar) -> None:
    """Signal that raw SMEM inputs are ready (k2-chain relay only)."""

    if prims.elect_sync():
        prims.mbarrier_arrive(raw_ready_mbar)


@cute.jit
def raw_ready_wait(raw_ready_mbar, raw_ready_phase):
    """Generic mbarrier parity spin-wait.

    The engine-class kernels observe raw readiness directly on the
    chunk's tma_mbar ring slot (consumer-direct wait); the k2 chain
    kernels still wait their raw_ready relay ring, and the ws_stored /
    The deltas_issued relay ring uses this helper too.
    """

    while not prims.mbarrier_wait_parity(
        raw_ready_mbar,
        raw_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return raw_ready_phase ^ cutlass.Int32(1)


@cute.jit
def raw_consumed_arrive(raw_consumed_mbar) -> None:
    """Signal that one participating warp has finished raw-slot reads."""

    if prims.elect_sync():
        prims.mbarrier_arrive(raw_consumed_mbar)


@cute.jit
def raw_consumed_wait(raw_consumed_mbar, raw_consumed_phase):
    """Wait until the previous chunk no longer reads raw input SMEM."""

    while not prims.mbarrier_wait_parity(
        raw_consumed_mbar,
        raw_consumed_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return raw_consumed_phase ^ cutlass.Int32(1)


@cute.jit
def state_input_ready_arrive(state_input_ready_mbar) -> None:
    """Signal that this CG1 warp has packed state_as_input TMEM."""

    if prims.elect_sync():
        prims.mbarrier_arrive(state_input_ready_mbar)


@cute.jit
def state_input_ready_wait(state_input_ready_mbar, state_input_ready_phase):
    """Wait until all CG1 warps have packed state_as_input TMEM."""

    while not prims.mbarrier_wait_parity(
        state_input_ready_mbar,
        state_input_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return state_input_ready_phase ^ cutlass.Int32(1)


@cute.jit
def update_ready_arrive(update_ready_mbar) -> None:
    """Signal that this CG1 warp has staged update for qkv MMA."""

    if prims.elect_sync():
        prims.mbarrier_arrive(update_ready_mbar)


@cute.jit
def update_ready_wait(update_ready_mbar, update_ready_phase):
    """Wait until all CG1 warps have staged update for qkv MMA."""

    while not prims.mbarrier_wait_parity(
        update_ready_mbar,
        update_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return update_ready_phase ^ cutlass.Int32(1)


@cute.jit
def final_state_stored_arrive(final_state_stored_mbar) -> None:
    """Signal that this CG1 warp has finished draining final_state TMEM."""

    if prims.elect_sync():
        prims.mbarrier_arrive(final_state_stored_mbar)


@cute.jit
def final_state_stored_wait(final_state_stored_mbar, final_state_stored_phase):
    """Wait until CG1 has drained final_state before TMEM deallocation."""

    while not prims.mbarrier_wait_parity(
        final_state_stored_mbar,
        final_state_stored_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return final_state_stored_phase ^ cutlass.Int32(1)


@cute.jit
def pack_output_b16x2_to_i32(
    value0: cutlass.Float32,
    value1: cutlass.Float32,
    output_dtype: cutlass.Constexpr,
):
    """Pack two FP32 output values through the compile-time 16-bit output dtype."""

    return (
        cutlass.Vector.from_elements(
            (value0, value1),
            cutlass.Float32,
        )
        .to(output_dtype)
        .bitcast(cutlass.Int32)[0]
    )


@cute.jit
def tcgen05_qstate_acc_tmem_col_offset(qstate_acc_stage):
    """Return the runtime TMEM column offset for one qstate acc stage."""

    return (
        KDA_TMEM_QSTATE_ACC_COL_OFFSET
        + qstate_acc_stage * KDA_TMEM_QSTATE_ACC_STAGE_STRIDE_COLS
    )


@cute.jit
def tcgen05_shared_acc_tmem_col_offset(shared_acc_stage):
    """Return the runtime TMEM column offset for one shared acc stage."""

    return KDA_TMEM_SHARED_ACC_COL_OFFSET + shared_acc_stage * KDA_TMEM_N16_ACC_COLS


@cute.jit
def tcgen05_shared_input_tmem_col_offset(shared_input_stage):
    """Return the runtime TMEM column offset for one shared input stage."""

    return (
        KDA_TMEM_SHARED_INPUT_COL_OFFSET
        + shared_input_stage * KDA_TMEM_SHARED_INPUT_COLS
    )


@cute.jit
def advance_ring_stage(
    stage,
    step: cutlass.Constexpr,
    stage_count: cutlass.Constexpr,
):
    """Advance a runtime ring index without division or a wrap branch."""

    next_stage = stage + cutlass.Int32(step)
    wrapped = cutlass.Int32(next_stage >= cutlass.Int32(stage_count))
    return next_stage - wrapped * cutlass.Int32(stage_count), wrapped


@cute.jit
def tcgen05_store_initial_state_tmem(
    tmem_raw_addr,
    state: cute.Tensor,
    sequence,
    bidy,
    dv_half,
    warp_idx,
    lane,
) -> None:
    """Initialize recurrent TMEM from the exact DV2 BF16 state."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    row_id = base_row_id + tmem_sp * THREADS_PER_WARP
    value_dim = (
        dv_half * cutlass.Int32(DV_HALF)
        + tmem_sp * ROWS_PER_WARP
        + (lane % ROWS_PER_WARP)
    )
    valid_lane = lane < ROWS_PER_WARP
    for key_block_start in cutlass.range_constexpr(
        0,
        DK,
        TCGEN05_FINAL_STATE_TMEM_LOAD_COLS,
    ):
        state_block = cutlass.Array(
            cutlass.Float32,
            TCGEN05_FINAL_STATE_TMEM_LOAD_COLS,
            alignment=16,
        )
        for col in cutlass.range_constexpr(TCGEN05_FINAL_STATE_TMEM_LOAD_COLS):
            key_dim = key_block_start + col
            state_value = cutlass.Float32(0.0)
            if valid_lane:
                state_value = cast(
                    "cutlass.Numeric",
                    state[sequence, bidy, value_dim, key_dim],
                ).to(cutlass.Float32)
            state_block[col] = state_value

        projection_col_id = base_col_id + KDA_TMEM_STATE_COL_OFFSET + key_block_start
        block_addr = (row_id << 16) | projection_col_id
        block_ptr = cutlass.inttoptr(block_addr, 6, cutlass.Float32)
        prims.tcgen05_st(
            "32x32b",
            block_ptr,
            state_block[0:TCGEN05_FINAL_STATE_TMEM_LOAD_COLS],
        )

    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)


@cute.jit
def tcgen05_stage_state_input_dv2_half_tmem(
    tmem_raw_addr,
    warp_idx,
    input_dtype: cutlass.Constexpr,
    HALF: cutlass.Constexpr,
) -> cutlass.Array:
    """Load one FP32 state half and asynchronously pack its MMA-A image."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS
    row_addr = (base_row_id + tmem_sp * THREADS_PER_WARP) << 16
    state_half_col = (
        base_col_id
        + KDA_TMEM_STATE_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    )
    state_ptr = cutlass.inttoptr(
        row_addr | state_half_col,
        6,
        cutlass.Float32,
    )
    state = prims.tcgen05_ld("16x256b", state_ptr, num=8)
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    packed_state = cutlass.Array(cutlass.Int32, 16, alignment=16)
    for dst_repeat in cutlass.range_constexpr(8):
        src_base = (dst_repeat ^ 1) * 4
        dst_base = dst_repeat * 2
        packed_state[dst_base] = pack_input_b16x2_to_i32(
            state[src_base],
            state[src_base + 1],
            input_dtype,
        )
        packed_state[dst_base + 1] = pack_input_b16x2_to_i32(
            state[src_base + 2],
            state[src_base + 3],
            input_dtype,
        )

    packed_col = (
        base_col_id
        + KDA_TMEM_STATE_AS_INPUT_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_PACKED_COLS
    )
    packed_ptr = prims.make_tmem_ptr(
        (base_row_id << 16) | packed_col,
        cutlass.Int8,
    )
    prims.tcgen05_st("16x128b", packed_ptr, packed_state[0:16])
    return state


@cute.jit
def tcgen05_rescale_state_dv2_half_regs(
    tmem_raw_addr,
    state_scale_f32_smem,
    warp_idx,
    state,
    state_input_ready_mbar,
    HALF: cutlass.Constexpr,
) -> None:
    """Overlap true-FP32 decay with the outstanding packed-state store."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS
    row_addr = (base_row_id + tmem_sp * THREADS_PER_WARP) << 16
    state_half_col = (
        base_col_id
        + KDA_TMEM_STATE_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    )
    state_ptr = cutlass.inttoptr(
        row_addr | state_half_col,
        6,
        cutlass.Float32,
    )
    state_scale_f32_ptr = state_scale_f32_smem.data_ptr()
    key_half_start: cutlass.Constexpr = HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    scaled_state = cutlass.Array(cutlass.Float32, 32, alignment=16)
    lane_col_pair = (cute.arch.lane_idx() % 4) * 2
    for repeat in cutlass.range_constexpr(8):
        reg_base = repeat * 4
        scale_dim = key_half_start + repeat * 8 + lane_col_pair
        scale = (state_scale_f32_ptr + scale_dim).load(count=2, alignment=8)
        scaled_state[reg_base], scaled_state[reg_base + 1] = fmul2(
            (state[reg_base], state[reg_base + 1]),
            (scale[0], scale[1]),
        )
        scaled_state[reg_base + 2], scaled_state[reg_base + 3] = fmul2(
            (state[reg_base + 2], state[reg_base + 3]),
            (scale[0], scale[1]),
        )

    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    state_input_ready_arrive(state_input_ready_mbar)
    prims.tcgen05_st("16x256b", state_ptr, scaled_state[0:32])
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_issue_state_projection_mma(
    tcgen05_decay_smem,
    tmem_raw_addr,
    acc_ready_mbar,
    tmem_col_offset,
    input_dtype: cutlass.Constexpr,
    K_BLOCK_BEGIN: cutlass.Constexpr,
    K_BLOCK_END: cutlass.Constexpr,
    INITIAL_SCALE_D: cutlass.Constexpr,
    COMMIT: cutlass.Constexpr,
    M_DIM: cutlass.Constexpr,
) -> None:
    """Issue state*decay K-slices through tcgen05, optionally committing."""

    tmem_ptr = cutlass.inttoptr(
        tmem_raw_addr + tmem_col_offset,
        6,
        cutlass.Float32,
    )
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=input_dtype,
        b_dtype=input_dtype,
        n_dim=BT,
        m_dim=M_DIM,
        b_major=0,
    )
    desc_k_decay = prims.Tcgen05SmemDesc.build(
        tcgen05_decay_smem.subview(0),
        leading_byte_offset=TCGEN05_STATE_K_B_LEADING_BYTES,
        stride_byte_offset=TCGEN05_STATE_K_B_STRIDE_BYTES,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )

    for k_block in cutlass.range_constexpr(K_BLOCK_BEGIN, K_BLOCK_END):
        scale_d = INITIAL_SCALE_D or k_block != K_BLOCK_BEGIN
        k_decay_offset = (
            k_block % TCGEN05_SW128_K_PHASES_PER_SLICE
        ) * TCGEN05_STATE_K_B_K_STEP_BYTES + (
            k_block // TCGEN05_SW128_K_PHASES_PER_SLICE
        ) * BT * TCGEN05_SW128_BYTES
        state_a_tmem = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Int8).subview(
            KDA_TMEM_STATE_AS_INPUT_COL_OFFSET + k_block * (TCGEN05_F16_K_ATOM // 2)
        )
        if prims.elect_sync():
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                tmem_ptr,
                state_a_tmem,
                desc_k_decay.advance_start_address(k_decay_offset),
                idesc,
                scale_d,
            )

    if cutlass.const_expr(COMMIT):
        if prims.elect_sync():
            prims.tcgen05_commit(acc_ready_mbar, group=prims.CTAGroup.CTA_1)


@cute.jit
def tcgen05_issue_state_k_mma(
    tcgen05_k_decay_smem,
    tmem_raw_addr,
    acc_ready_mbar,
    shared_acc_stage,
    input_dtype: cutlass.Constexpr,
    K_BLOCK_BEGIN: cutlass.Constexpr,
    K_BLOCK_END: cutlass.Constexpr,
    INITIAL_SCALE_D: cutlass.Constexpr,
    COMMIT: cutlass.Constexpr,
    M_DIM: cutlass.Constexpr,
) -> None:
    """Issue a K-slice range of state*k into a scheduled shared_acc stage."""

    tcgen05_issue_state_projection_mma(
        tcgen05_k_decay_smem,
        tmem_raw_addr,
        acc_ready_mbar,
        tcgen05_shared_acc_tmem_col_offset(shared_acc_stage),
        input_dtype,
        K_BLOCK_BEGIN,
        K_BLOCK_END,
        INITIAL_SCALE_D,
        COMMIT,
        M_DIM,
    )


@cute.jit
def tcgen05_wait_acc_buffer_ready(
    acc_ready_mbar,
    acc_ready_phase,
):
    """Wait until a producer commit has filled the corresponding TMEM acc tile."""

    while not prims.mbarrier_wait_parity(
        acc_ready_mbar,
        acc_ready_phase,
        prims.MBarrierWait.TRY,
    ):
        pass
    return acc_ready_phase ^ cutlass.Int32(1)


@cute.jit
def tcgen05_rhs_token_pair_from_16x256b_fragment(
    fragment,
    reg_idx: cutlass.Constexpr,
):
    """Select the lane-local state*k pair needed by the RHS TMEM store."""

    if cutlass.const_expr(reg_idx == 0):
        return fragment[4], fragment[5]
    if cutlass.const_expr(reg_idx == 1):
        return fragment[6], fragment[7]
    if cutlass.const_expr(reg_idx == 2):
        return fragment[0], fragment[1]
    return fragment[2], fragment[3]


@cute.jit
def tcgen05_store_final_state_tmem(
    tmem_raw_addr,
    state_col_offset,
    state: cute.Tensor,
    sequence,
    bidy,
    dv_half,
    warp_idx,
    lane,
) -> None:
    """Store the live recurrent TMEM state to the exact DV2 BF16 state."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    row_id = base_row_id + tmem_sp * THREADS_PER_WARP
    value_dim = (
        dv_half * cutlass.Int32(DV_HALF)
        + tmem_sp * ROWS_PER_WARP
        + (lane % ROWS_PER_WARP)
    )
    valid_lane = lane < ROWS_PER_WARP
    for key_block_start in cutlass.range_constexpr(
        0,
        DK,
        TCGEN05_FINAL_STATE_TMEM_LOAD_COLS,
    ):
        projection_col_id = base_col_id + state_col_offset + key_block_start
        block_addr = (row_id << 16) | projection_col_id
        block_ptr = cutlass.inttoptr(block_addr, 6, cutlass.Float32)
        loaded = prims.tcgen05_ld(
            "32x32b",
            block_ptr,
            num=TCGEN05_FINAL_STATE_TMEM_LOAD_COLS,
        )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

        for col in cutlass.range_constexpr(TCGEN05_FINAL_STATE_TMEM_LOAD_COLS):
            key_dim = key_block_start + col
            if valid_lane:
                state[sequence, bidy, value_dim, key_dim] = loaded[col].to(
                    state.element_type
                )


K2_RAW_STAGE_COUNT: int = 8

K2_TMA_MBAR_STAGE_COUNT: int = 6

TILE_ELEMS: int = BT * DK

DIAG_REC_ELEMS: int = DK  # per-chunk fp32 diag record

QK_REC_ELEMS: int = BT * BT  # per-chunk SW32-permuted QK' record

TMEM_ALLOC_COLS: int = 512

K2_QSTATE_STAGE_COUNT: int = 2

K2_OUTPUT_SMEM_STAGE_COUNT: int = 7

DV_HALF: int = DV // 2

ROWS_PER_WARP: int = DV_HALF // 4  # Layout F: 16 rows per quadrant

V_TILE_ELEMS: int = BT * DV_HALF  # one 64-elem s128 TMA segment

K2_O_SMEM_STAGE_SIZE: int = BT * DV_HALF

K2_O_SMEM_TILE_SIZE: int = K2_OUTPUT_SMEM_STAGE_COUNT * K2_O_SMEM_STAGE_SIZE

K2_TX_BYTES: int = (
    3 * TILE_ELEMS * 2  # kd + w + qd (bf16, full DK)
    + V_TILE_ELEMS * 2  # v (bf16, DV half)
    + DIAG_REC_ELEMS * 4  # diag (fp32)
    + QK_REC_ELEMS * 2  # qk' (bf16)
)


@cute.jit
def tma_chain_load_tile(
    tma_desc: cutlass.GridConstant[cuda.TensorMap],
    tile_smem,
    row_start,
    head_idx,
    tma_mbar,
) -> None:
    """Issue one rank-4 TMA for a complete [16, DK] workspace tile."""

    tma_coord = (
        cutlass.Int32(0),
        row_start,
        head_idx,
        cutlass.Int32(0),
    )
    prims.cp_async_bulk_tensor_shared_cta_global(
        tile_smem.subview(0),
        tma_desc.get_ptr(),
        tma_coord,
        tma_mbar,
    )


@cute.jit
def tcgen05_commit(mbar) -> None:
    """Commit the tensor pipe's progress to an mbarrier (single elected lane)."""

    if prims.elect_sync():
        prims.tcgen05_commit(mbar, group=prims.CTAGroup.CTA_1)


@cute.jit
def tcgen05_chain_issue_delta_half_mma(
    b_smem,
    tmem_raw_addr,
    a_input_col,
    input_dtype: cutlass.Constexpr,
    HALF: cutlass.Constexpr,
) -> None:
    """M=64 delta-half MMA issue."""

    half_n = DK // 2
    tmem_ptr = cutlass.inttoptr(
        tmem_raw_addr + KDA_TMEM_STATE_COL_OFFSET + HALF * half_n,
        6,
        cutlass.Float32,
    )
    desc_b = prims.Tcgen05SmemDesc.build(
        b_smem.subview(0),
        leading_byte_offset=TCGEN05_FINAL_STATE_B_LEADING_BYTES,
        stride_byte_offset=TCGEN05_FINAL_STATE_B_STRIDE_BYTES,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=input_dtype,
        b_dtype=input_dtype,
        n_dim=half_n,
        m_dim=DV_HALF,
        b_major=1,
    )
    half_byte_offset = HALF * TCGEN05_FINAL_STATE_B_LEADING_BYTES
    a_tmem = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Int8).subview(a_input_col)
    if prims.elect_sync():
        prims.tcgen05_mma(
            prims.Tcgen05MMAKind.F16,
            prims.CTAGroup.CTA_1,
            tmem_ptr,
            a_tmem,
            desc_b.advance_start_address(half_byte_offset),
            idesc,
            True,
        )


@cute.jit
def tcgen05_chain_issue_qkv_mma(
    qk_stage_smem,
    tmem_raw_addr,
    a_input_col,
    qstate_acc_stage,
    acc_ready_mbar,
    input_dtype: cutlass.Constexpr,
    COMMIT: cutlass.Constexpr,
) -> None:
    """M=64 qkv MMA issue."""

    tmem_ptr = cutlass.inttoptr(
        tmem_raw_addr + tcgen05_qstate_acc_tmem_col_offset(qstate_acc_stage),
        6,
        cutlass.Float32,
    )
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=input_dtype,
        b_dtype=input_dtype,
        n_dim=BT,
        m_dim=DV_HALF,
        b_major=0,
    )
    lhs_tmem = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Int8).subview(a_input_col)
    desc_pairwise = prims.Tcgen05SmemDesc.build(
        qk_stage_smem.subview(0),
        leading_byte_offset=TCGEN05_VALUE_PAIRWISE_B_LEADING_BYTES,
        stride_byte_offset=TCGEN05_VALUE_PAIRWISE_B_STRIDE_BYTES,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_32B,
    )
    if prims.elect_sync():
        prims.tcgen05_mma(
            prims.Tcgen05MMAKind.F16,
            prims.CTAGroup.CTA_1,
            tmem_ptr,
            lhs_tmem,
            desc_pairwise,
            idesc,
            True,
        )
        if cutlass.const_expr(COMMIT):
            prims.tcgen05_commit(acc_ready_mbar, group=prims.CTAGroup.CTA_1)


@cute.jit
def tcgen05_chain_stage_vmx_input_tmem(
    tmem_raw_addr,
    raw_v_smem,
    warp_idx,
    lane,
    shared_acc_stage,
    shared_input_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """M=64 (v - X) fused repack staging.

    ONE v ldmatrix ([16,64] half tile) + ONE 16x256b X
    fragment read + sub.b16x2 + ONE 16x128b store.
    """

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    projection_col_id = base_col_id + tcgen05_shared_acc_tmem_col_offset(
        shared_acc_stage
    )
    input_col_id = base_col_id + tcgen05_shared_input_tmem_col_offset(
        shared_input_stage
    )
    value_dim_base = tmem_sp * ROWS_PER_WARP

    row_id0 = base_row_id + tmem_sp * THREADS_PER_WARP
    block_ptr0 = cutlass.inttoptr(
        (row_id0 << 16) | projection_col_id, 6, cutlass.Float32
    )
    state_k0 = prims.tcgen05_ld("16x256b", block_ptr0, num=2)

    raw_v_regs0 = prims.ldmatrix(
        raw_v_ldmatrix_trans_ptr(raw_v_smem, value_dim_base, lane),
        4,
        prims.MMALayout.COL,
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    packed0 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
    for reg_idx in cutlass.range_constexpr(4):
        raw_matrix: cutlass.Constexpr[int] = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
        val0, val1 = tcgen05_rhs_token_pair_from_16x256b_fragment(
            state_k0,
            reg_idx,
        )
        packed0[reg_idx] = sub_b16x2_input_dtype(
            raw_v_regs0[raw_matrix],
            pack_input_b16x2_to_i32(val0, val1, input_dtype),
            input_dtype,
        )

    input_block_addr0 = (base_row_id << 16) | input_col_id
    input_block_ptr0 = prims.make_tmem_ptr(input_block_addr0, cutlass.Int8)
    prims.tcgen05_st("16x128b", input_block_ptr0, packed0[0:4])
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_chain_load_qstate_output_regs(
    tmem_raw_addr,
    warp_idx,
    qstate_acc_stage,
    scale: cutlass.Float32,
    output_dtype: cutlass.Constexpr,
):
    """Load, scale, and pack one warp's output quadrant from TMEM."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    projection_col_id = base_col_id + tcgen05_qstate_acc_tmem_col_offset(
        qstate_acc_stage
    )
    row_id0 = base_row_id + tmem_sp * THREADS_PER_WARP
    block_addr0 = (row_id0 << 16) | projection_col_id
    block_ptr0 = cutlass.inttoptr(block_addr0, 6, cutlass.Float32)
    loaded0 = prims.tcgen05_ld(
        "16x256b",
        block_ptr0,
        num=2,
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    stsm_regs0 = cutlass.Array(
        cutlass.Int32,
        4,
        space=cutlass.AddressSpace.rmem,
    )
    for reg_idx in cutlass.range_constexpr(4):
        scaled0_0, scaled0_1 = fmul2(
            (loaded0[2 * reg_idx], loaded0[2 * reg_idx + 1]),
            (scale, scale),
        )
        stsm_regs0[reg_idx] = pack_output_b16x2_to_i32(
            scaled0_0,
            scaled0_1,
            output_dtype,
        )
    return stsm_regs0


@cute.jit
def tcgen05_chain_store_output_smem(
    o_smem,
    warp_idx,
    lane,
    o_stage_base,
    stsm_regs0,
) -> None:
    """Store one warp's packed output quadrant into the output-SMEM ring."""

    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS
    value_dim_base = tmem_sp * ROWS_PER_WARP
    smem_dst0 = o_smem_stmatrix_128b_ptr(
        o_smem,
        o_stage_base,
        value_dim_base,
        lane,
    )
    prims.stmatrix(
        smem_dst0,
        stsm_regs0.data_ptr().load(count=4, alignment=4),
        prims.MMALayout.COL,
        shape=prims.StoreShape.M8N8,
    )
    cute.arch.fence_view_async_shared()


@cute.jit
def tma_chain_stage_load_inputs(
    tma_desc_kd: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_qd: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_diag: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_qk: cutlass.GridConstant[cuda.TensorMap],
    kd_stage,
    w_stage,
    qd_stage,
    v_stage,
    diag_stage,
    qk_stage,
    head_idx,
    dv_half,
    ws_row_start,
    ws_chunk,
    v_row_start,
    tma_mbar,
    tma_tx_bytes: cutlass.Constexpr,
) -> None:
    """DV2 chunk operand transaction: full kd/w/qd, HALF v (one segment)."""

    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(tma_mbar, tma_tx_bytes)
    if prims.elect_sync():
        tma_chain_load_tile(tma_desc_kd, kd_stage, ws_row_start, head_idx, tma_mbar)
        tma_chain_load_tile(tma_desc_w, w_stage, ws_row_start, head_idx, tma_mbar)
        tma_chain_load_tile(tma_desc_qd, qd_stage, ws_row_start, head_idx, tma_mbar)
        v_coord = (
            dv_half * cutlass.Int32(DV_HALF),
            v_row_start,
            head_idx,
            cutlass.Int32(0),
        )
        prims.cp_async_bulk_tensor_shared_cta_global(
            v_stage.subview(0),
            tma_desc_v.get_ptr(),
            v_coord,
            tma_mbar,
        )
        diag_coord = (
            cutlass.Int32(0),
            ws_chunk,
            head_idx,
            cutlass.Int32(0),
        )
        prims.cp_async_bulk_tensor_shared_cta_global(
            diag_stage.subview(0),
            tma_desc_diag.get_ptr(),
            diag_coord,
            tma_mbar,
        )
        prims.cp_async_bulk_tensor_shared_cta_global(
            qk_stage.subview(0),
            tma_desc_qk.get_ptr(),
            diag_coord,
            tma_mbar,
        )


@cute.jit
def epilogue_chain_stage_store(
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    o_smem,
    sequence_start,
    head_idx,
    dv_half,
    chunk_start,
    o_stage_base,
) -> None:
    """Store the staged `[BT, 64]` half-output tile (one s128 segment)."""

    global_chunk_start = sequence_start + chunk_start
    if prims.elect_sync():
        o_coord = (
            dv_half * cutlass.Int32(DV_HALF),
            global_chunk_start,
            head_idx,
            cutlass.Int32(0),
        )
        prims.cp_async_bulk_tensor_global_shared_cta(
            tma_desc_o.get_ptr(),
            o_smem.subview(o_stage_base + O_OUT_OFFSET),
            o_coord,
        )
        prims.cp_async_bulk_commit_group()
    prims.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def epilogue_chain_tail_store(
    out,
    o_smem,
    sequence_start,
    head_idx,
    dv_half,
    chunk_start,
    seqlen,
    o_stage_base,
    lane,
) -> None:
    """Store a partial packed-sequence half-tail without crossing its boundary."""

    valid_tokens = seqlen - chunk_start
    value_base = dv_half * cutlass.Int32(DV_HALF)
    for elem_iter in cutlass.range_constexpr((BT * DV_HALF) // THREADS_PER_WARP):
        linear_idx = elem_iter * THREADS_PER_WARP + lane
        token_coord = linear_idx // DV_HALF
        value_dim = linear_idx - token_coord * DV_HALF
        if token_coord < valid_tokens:
            smem_idx = o_smem_swizzle_128b_elem_index(
                o_stage_base,
                value_dim,
                token_coord,
            )
            out[
                0,
                sequence_start + chunk_start + token_coord,
                head_idx,
                value_base + value_dim,
            ] = o_smem[smem_idx]
    prims.bar_warp_sync(cute.arch.FULL_MASK)


@cute.kernel
def kernel_chain_dv2(
    tma_desc_kd: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_qd: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_diag: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_qk: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    v: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_chunks: cute.Tensor,
    state: cute.Tensor,
    out: cute.Tensor,
    head_base: cutlass.Int32,
    SCALE: cutlass.Float32,
) -> None:
    """kernel 2, DV-split: each CTA owns half the hidden dimension.

    Grid `(num_sequences, launch_heads * 2, 1)`; bidy = head * 2 + half.
    Identical schedule and mbar topology to the base chain with M=64
    MMAs (PTX Layout F: 16 rows per warp quadrant, lane alignment 0);
    kd/W/QK'/diag are read identically by both halves, v/o/state are
    DV-split (one 64-elem s128 TMA segment at value offset half*64).
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    dv_half = bidy % 2
    bidy = head_base + bidy // 2
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % THREADS_PER_WARP

    sequence_start = cutlass.Int32(cu_seqlens[bidx])
    sequence_end = cutlass.Int32(cu_seqlens[bidx + 1])
    seqlen = sequence_end - sequence_start
    num_chunks = cute.ceil_div(seqlen, BT)
    chunk_base = cutlass.Int32(cu_chunks[bidx])
    input_dtype = v.element_type

    # --- mbarriers (identical topology to the base chain) ----------------
    tma_mbar = cutlass.Array(
        cutlass.Int64,
        K2_TMA_MBAR_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    raw_ready_mbar = cutlass.Array(
        cutlass.Int64,
        K2_RAW_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    raw_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        K2_RAW_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    state_input_ready_l_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    state_input_ready_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    u_input_ready_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    update_ready_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    shared_acc_ready_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    k_restore_consumed_l_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    k_restore_consumed_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    qstate_acc_ready_mbar = cutlass.Array(
        cutlass.Int64,
        K2_QSTATE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    stateq_done_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    output_ready_mbar = cutlass.Array(
        cutlass.Int64,
        K2_QSTATE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    final_state_stored_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    tmem_ptr_i32 = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )

    # --- SMEM rings (v halved to one segment; o halved) --------------------
    kd_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    w_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    qd_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    v_raw_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * V_TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    diag_raw_smem = cutlass.Array(
        cutlass.Float32,
        K2_RAW_STAGE_COUNT * DIAG_REC_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    qk_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * QK_REC_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    o_smem = cutlass.Array(
        out.element_type,
        K2_O_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=O_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )

    # --- init (identical to the base chain) ------------------------------
    if warp_idx == ROLES.tma_load:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(K2_TMA_MBAR_STAGE_COUNT):
                prims.mbarrier_init(tma_mbar.subview(stage), 1)
            for stage in cutlass.range_constexpr(K2_RAW_STAGE_COUNT):
                prims.mbarrier_init(raw_ready_mbar.subview(stage), 1)
                prims.mbarrier_init(
                    raw_consumed_mbar.subview(stage),
                    5,
                )
    elif warp_idx == ROLES.tcgen05_mma:
        if prims.elect_sync():
            prims.mbarrier_init(state_input_ready_l_mbar, 4)
            prims.mbarrier_init(state_input_ready_mbar, 4)
            for stage in cutlass.range_constexpr(2):
                prims.mbarrier_init(u_input_ready_mbar.subview(stage), 4)
                prims.mbarrier_init(shared_acc_ready_mbar.subview(stage), 1)
                prims.mbarrier_init(k_restore_consumed_l_mbar.subview(stage), 1)
                prims.mbarrier_init(k_restore_consumed_mbar.subview(stage), 1)
                prims.mbarrier_init(stateq_done_mbar.subview(stage), 1)
            # CG1 fuses the right-half pack/scale while idle CG0 warps 4-7
            # fuse the left half. Both groups must finish before delta.
            prims.mbarrier_init(update_ready_mbar, 8)
            for stage in cutlass.range_constexpr(K2_QSTATE_STAGE_COUNT):
                prims.mbarrier_init(qstate_acc_ready_mbar.subview(stage), 1)
            prims.mbarrier_init(final_state_stored_mbar, 8)
    elif warp_idx == ROLES.epilogue:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(K2_QSTATE_STAGE_COUNT):
                prims.mbarrier_init(output_ready_mbar.subview(stage), 4)
    prims.fence_mbarrier_init()
    cta_sync()

    if is_tmem_user_warp(warp_idx):
        if warp_idx == ROLES.tcgen05_mma:
            prims.tcgen05_alloc(tmem_ptr_i32, TMEM_ALLOC_COLS, group="cta_1")
        tmem_user_sync()
        if warp_idx == ROLES.tcgen05_mma:
            prims.tcgen05_relinquish_alloc_permit(group="cta_1")
        tmem_user_sync()
    cta_sync()

    if is_service_warpgroup(warp_idx):
        prims.setmaxregister(KDA_SERVICE_REGS, prims.SetMaxRegisterAction.DECREASE)

    # ======================= warp 14: TMA ring =========================
    if warp_idx == ROLES.tma_load:
        raw_stage = cutlass.Int32(0)
        raw_consumed_phase = cutlass.Int32(1)
        issue_mbar_slot = cutlass.Int32(0)
        wait_mbar_slot = cutlass.Int32(0)
        ready_stage = cutlass.Int32(0)
        tma_phase = cutlass.Int32(0)
        for chunk in cutlass.range(num_chunks, unroll=1):
            ws_chunk = chunk_base + chunk
            ws_row_start = ws_chunk * cutlass.Int32(BT)
            v_row_start = sequence_start + chunk * cutlass.Int32(BT)
            raw_consumed_wait(
                raw_consumed_mbar.subview(raw_stage),
                raw_consumed_phase,
            )
            tma_chain_stage_load_inputs(
                tma_desc_kd,
                tma_desc_w,
                tma_desc_qd,
                tma_desc_v,
                tma_desc_diag,
                tma_desc_qk,
                kd_smem.subview(raw_stage * TILE_ELEMS),
                w_smem.subview(raw_stage * TILE_ELEMS),
                qd_smem.subview(raw_stage * TILE_ELEMS),
                v_raw_smem.subview(raw_stage * V_TILE_ELEMS),
                diag_raw_smem.subview(raw_stage * DIAG_REC_ELEMS),
                qk_smem.subview(raw_stage * QK_REC_ELEMS),
                bidy,
                dv_half,
                ws_row_start,
                ws_chunk,
                v_row_start,
                tma_mbar.subview(issue_mbar_slot),
                K2_TX_BYTES,
            )
            issue_mbar_slot, _ = advance_ring_stage(
                issue_mbar_slot, 1, K2_TMA_MBAR_STAGE_COUNT
            )
            if chunk >= cutlass.Int32(K2_TMA_MBAR_STAGE_COUNT - 1):
                tma_transfer_wait(tma_mbar.subview(wait_mbar_slot), tma_phase)
                wait_mbar_slot, wait_wrapped = advance_ring_stage(
                    wait_mbar_slot, 1, K2_TMA_MBAR_STAGE_COUNT
                )
                tma_phase = tma_phase ^ wait_wrapped
                raw_ready_arrive(raw_ready_mbar.subview(ready_stage))
                ready_stage, _ = advance_ring_stage(ready_stage, 1, K2_RAW_STAGE_COUNT)
            raw_stage, raw_wrapped = advance_ring_stage(
                raw_stage, 1, K2_RAW_STAGE_COUNT
            )
            raw_consumed_phase = raw_consumed_phase ^ raw_wrapped
        tma_full = cutlass.Int32(
            num_chunks >= cutlass.Int32(K2_TMA_MBAR_STAGE_COUNT - 1)
        )
        tma_tail = (
            tma_full * cutlass.Int32(K2_TMA_MBAR_STAGE_COUNT - 1)
            + (cutlass.Int32(1) - tma_full) * num_chunks
        )
        for _tail in cutlass.range(tma_tail, unroll=1):
            tma_transfer_wait(tma_mbar.subview(wait_mbar_slot), tma_phase)
            wait_mbar_slot, wait_wrapped = advance_ring_stage(
                wait_mbar_slot, 1, K2_TMA_MBAR_STAGE_COUNT
            )
            tma_phase = tma_phase ^ wait_wrapped
            raw_ready_arrive(raw_ready_mbar.subview(ready_stage))
            ready_stage, _ = advance_ring_stage(ready_stage, 1, K2_RAW_STAGE_COUNT)

    # ====== warp 12: output-group issuer (OUT_ISSUER12) / flag relay ======
    elif warp_idx == ROLES.super_mma:
        tmem_raw_addr = tmem_ptr_i32.load()
        si_phase12 = cutlass.Int32(0)
        upd_phase12 = cutlass.Int32(0)
        for chunk in cutlass.range(num_chunks, unroll=1):
            raw_stage = chunk % K2_RAW_STAGE_COUNT
            qstate_stage = chunk % K2_QSTATE_STAGE_COUNT
            kr_stage = chunk % 2
            acc_stage = chunk % 2
            if chunk > 0:
                prev12 = chunk - cutlass.Int32(1)
                tcgen05_wait_acc_buffer_ready(
                    qstate_acc_ready_mbar.subview(prev12 % K2_QSTATE_STAGE_COUNT),
                    (prev12 // K2_QSTATE_STAGE_COUNT) % 2,
                )
                raw_consumed_arrive(
                    raw_consumed_mbar.subview(prev12 % K2_RAW_STAGE_COUNT)
                )
            raw_ready_wait(
                raw_ready_mbar.subview(raw_stage),
                (chunk // K2_RAW_STAGE_COUNT) % 2,
            )
            si_phase12 = state_input_ready_wait(state_input_ready_mbar, si_phase12)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            output_ready_wait(
                output_ready_mbar.subview(qstate_stage),
                (chunk // K2_QSTATE_STAGE_COUNT + cutlass.Int32(1)) % 2,
            )
            tcgen05_issue_state_projection_mma(
                qd_smem.subview(raw_stage * TILE_ELEMS),
                tmem_raw_addr,
                stateq_done_mbar.subview(kr_stage),
                tcgen05_qstate_acc_tmem_col_offset(qstate_stage),
                input_dtype,
                0,
                DK // TCGEN05_F16_K_ATOM,
                False,
                True,
                DV_HALF,
            )
            upd_phase12 = update_ready_wait(update_ready_mbar, upd_phase12)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            xpack_col12 = tcgen05_shared_input_tmem_col_offset(acc_stage)
            tcgen05_chain_issue_qkv_mma(
                qk_smem.subview(raw_stage * QK_REC_ELEMS),
                tmem_raw_addr,
                xpack_col12,
                qstate_stage,
                qstate_acc_ready_mbar.subview(qstate_stage),
                input_dtype,
                True,
            )

    # =============== warp 13: tcgen05 issuer (the chain owner) =============
    elif warp_idx == ROLES.tcgen05_mma:
        tmem_raw_addr = tmem_ptr_i32.load()
        si_l_phase = cutlass.Int32(0)
        si_phase = cutlass.Int32(0)
        upd_phase = cutlass.Int32(0)
        for chunk in cutlass.range(num_chunks, unroll=1):
            raw_stage = chunk % K2_RAW_STAGE_COUNT
            raw_phase = (chunk // K2_RAW_STAGE_COUNT) % 2
            acc_stage = chunk % 2
            qstate_stage = chunk % K2_QSTATE_STAGE_COUNT
            kr_stage = chunk % 2
            kd_stage_smem = kd_smem.subview(raw_stage * TILE_ELEMS)
            w_stage_smem = w_smem.subview(raw_stage * TILE_ELEMS)

            raw_ready_wait(raw_ready_mbar.subview(raw_stage), raw_phase)

            si_l_phase = state_input_ready_wait(state_input_ready_l_mbar, si_l_phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tcgen05_issue_state_k_mma(
                kd_stage_smem,
                tmem_raw_addr,
                shared_acc_ready_mbar.subview(acc_stage),
                acc_stage,
                input_dtype,
                0,
                (DK // TCGEN05_F16_K_ATOM) // 2,
                False,
                False,
                DV_HALF,
            )
            si_phase = state_input_ready_wait(state_input_ready_mbar, si_phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tcgen05_issue_state_k_mma(
                kd_stage_smem,
                tmem_raw_addr,
                shared_acc_ready_mbar.subview(acc_stage),
                acc_stage,
                input_dtype,
                (DK // TCGEN05_F16_K_ATOM) // 2,
                DK // TCGEN05_F16_K_ATOM,
                True,
                True,
                DV_HALF,
            )

            xpack_col = tcgen05_shared_input_tmem_col_offset(acc_stage)
            upd_phase = update_ready_wait(update_ready_mbar, upd_phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tcgen05_chain_issue_delta_half_mma(
                w_stage_smem,
                tmem_raw_addr,
                xpack_col,
                input_dtype,
                0,
            )
            tcgen05_commit(k_restore_consumed_l_mbar.subview(kr_stage))
            tcgen05_chain_issue_delta_half_mma(
                w_stage_smem,
                tmem_raw_addr,
                xpack_col,
                input_dtype,
                1,
            )
            tcgen05_commit(k_restore_consumed_mbar.subview(kr_stage))

        final_state_stored_wait(final_state_stored_mbar, cutlass.Int32(0))
        tmem_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Float32)
        prims.tcgen05_dealloc(tmem_ptr, TMEM_ALLOC_COLS, group="cta_1")

    # ================= CG1 (warps 8-11): TMEM epilogues ====================
    elif is_compute_group1_warp(warp_idx):
        prims.setmaxregister(KDA_CG1_REGS, prims.SetMaxRegisterAction.INCREASE)
        tmem_raw_addr = tmem_ptr_i32.load()
        tcgen05_store_initial_state_tmem(
            tmem_raw_addr,
            state,
            bidx,
            bidy,
            dv_half,
            warp_idx,
            lane,
        )
        if num_chunks > 0:
            diag_raw_stage = diag_raw_smem.subview(0)
            state_left = tcgen05_stage_state_input_dv2_half_tmem(
                tmem_raw_addr,
                warp_idx,
                input_dtype,
                0,
            )
            raw_ready_wait(raw_ready_mbar.subview(0), 0)
            tcgen05_rescale_state_dv2_half_regs(
                tmem_raw_addr,
                diag_raw_stage,
                warp_idx,
                state_left,
                state_input_ready_l_mbar,
                0,
            )
            state_right = tcgen05_stage_state_input_dv2_half_tmem(
                tmem_raw_addr,
                warp_idx,
                input_dtype,
                1,
            )
            tcgen05_rescale_state_dv2_half_regs(
                tmem_raw_addr,
                diag_raw_stage,
                warp_idx,
                state_right,
                state_input_ready_mbar,
                1,
            )
            tcgen05_wait_acc_buffer_ready(shared_acc_ready_mbar.subview(0), 0)
            vmx_lane = cute.arch.lane_idx()
            tcgen05_chain_stage_vmx_input_tmem(
                tmem_raw_addr,
                v_raw_smem.subview(0),
                warp_idx,
                vmx_lane,
                0,
                0,
                input_dtype,
            )
            update_ready_arrive(update_ready_mbar)
        for chunk in cutlass.range(1, num_chunks, 1, unroll=1):
            prev = chunk - cutlass.Int32(1)
            raw_stage = chunk % K2_RAW_STAGE_COUNT
            diag_raw_stage = diag_raw_smem.subview(raw_stage * DIAG_REC_ELEMS)
            acc_stage = chunk % 2
            prev_kr_phase = (prev // 2) % 2
            tcgen05_wait_acc_buffer_ready(
                stateq_done_mbar.subview(prev % 2), prev_kr_phase
            )
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_mbar.subview(prev % 2), prev_kr_phase
            )
            state_right = tcgen05_stage_state_input_dv2_half_tmem(
                tmem_raw_addr,
                warp_idx,
                input_dtype,
                1,
            )
            raw_consumed_arrive(raw_consumed_mbar.subview(prev % K2_RAW_STAGE_COUNT))
            raw_ready_wait(
                raw_ready_mbar.subview(raw_stage),
                (chunk // K2_RAW_STAGE_COUNT) % 2,
            )
            tcgen05_rescale_state_dv2_half_regs(
                tmem_raw_addr,
                diag_raw_stage,
                warp_idx,
                state_right,
                state_input_ready_mbar,
                1,
            )
            tcgen05_wait_acc_buffer_ready(
                shared_acc_ready_mbar.subview(acc_stage), (chunk // 2) % 2
            )
            vmx_lane = cute.arch.lane_idx()
            tcgen05_chain_stage_vmx_input_tmem(
                tmem_raw_addr,
                v_raw_smem.subview(raw_stage * V_TILE_ELEMS),
                warp_idx,
                vmx_lane,
                acc_stage,
                acc_stage,
                input_dtype,
            )
            update_ready_arrive(update_ready_mbar)
        if num_chunks > 0:
            last = num_chunks - cutlass.Int32(1)
            last_kr_phase = (last // 2) % 2
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_l_mbar.subview(last % 2), last_kr_phase
            )
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_mbar.subview(last % 2), last_kr_phase
            )
        final_lane = cute.arch.lane_idx()
        tcgen05_store_final_state_tmem(
            tmem_raw_addr,
            KDA_TMEM_STATE_COL_OFFSET,
            state,
            bidx,
            bidy,
            dv_half,
            warp_idx,
            final_lane,
        )
        final_state_stored_arrive(final_state_stored_mbar)
    elif is_compute_group0_warp(warp_idx):
        if warp_idx < 4:
            prims.setmaxregister(KDA_CG1_REGS, prims.SetMaxRegisterAction.INCREASE)
            tmem_raw_addr = tmem_ptr_i32.load()
            for chunk in cutlass.range(num_chunks, unroll=1):
                qstate_stage = chunk % K2_QSTATE_STAGE_COUNT
                qstate_phase = (chunk // K2_QSTATE_STAGE_COUNT) % 2
                output_stage = chunk % K2_OUTPUT_SMEM_STAGE_COUNT
                tcgen05_wait_acc_buffer_ready(
                    qstate_acc_ready_mbar.subview(qstate_stage), qstate_phase
                )
                output_regs = tcgen05_chain_load_qstate_output_regs(
                    tmem_raw_addr,
                    warp_idx,
                    qstate_stage,
                    SCALE,
                    out.element_type,
                )
                # Release the two-stage TMEM accumulator as soon as its values
                # are in registers.  Output-SMEM reuse is independently
                # protected by the seven-deep async TMA store group.
                output_ready_arrive(output_ready_mbar.subview(qstate_stage))
                if warp_idx == 0:
                    prims.cp_async_bulk_wait_group(6, read=True)
                output_drain_sync()
                output_stage_base = output_stage * K2_O_SMEM_STAGE_SIZE
                tcgen05_chain_store_output_smem(
                    o_smem,
                    warp_idx,
                    lane,
                    output_stage_base,
                    output_regs,
                )
                output_drain_sync()
                output_chunk_start = chunk * BT
                if seqlen >= output_chunk_start + BT:
                    if warp_idx == 0:
                        epilogue_chain_stage_store(
                            tma_desc_o,
                            o_smem,
                            sequence_start,
                            bidy,
                            dv_half,
                            output_chunk_start,
                            output_stage_base,
                        )
                else:
                    if warp_idx == 0:
                        epilogue_chain_tail_store(
                            out,
                            o_smem,
                            sequence_start,
                            bidy,
                            dv_half,
                            output_chunk_start,
                            seqlen,
                            output_stage_base,
                            lane,
                        )
            if warp_idx == 0:
                prims.cp_async_bulk_wait_group(0, read=True)
            output_drain_sync()
            final_state_stored_arrive(final_state_stored_mbar)
        else:
            # Chunk 0 is initialized by CG1. Afterwards, warps 4-7 own the
            # fused left-half state pack and FP32 decay while CG1 owns right.
            prims.setmaxregister(KDA_CG1_REGS, prims.SetMaxRegisterAction.INCREASE)
            tmem_raw_addr = tmem_ptr_i32.load()
            if num_chunks > 0:
                state_input_ready_wait(
                    state_input_ready_l_mbar,
                    cutlass.Int32(0),
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                update_ready_arrive(update_ready_mbar)
            for chunk in cutlass.range(1, num_chunks, 1, unroll=1):
                prev = chunk - cutlass.Int32(1)
                raw_stage = chunk % K2_RAW_STAGE_COUNT
                prev_kr_phase = (prev // 2) % 2
                tcgen05_wait_acc_buffer_ready(
                    stateq_done_mbar.subview(prev % 2),
                    prev_kr_phase,
                )
                tcgen05_wait_acc_buffer_ready(
                    k_restore_consumed_l_mbar.subview(prev % 2),
                    prev_kr_phase,
                )
                state_left = tcgen05_stage_state_input_dv2_half_tmem(
                    tmem_raw_addr,
                    warp_idx,
                    input_dtype,
                    0,
                )
                raw_ready_wait(
                    raw_ready_mbar.subview(raw_stage),
                    (chunk // K2_RAW_STAGE_COUNT) % 2,
                )
                diag_raw_stage = diag_raw_smem.subview(raw_stage * DIAG_REC_ELEMS)
                tcgen05_rescale_state_dv2_half_regs(
                    tmem_raw_addr,
                    diag_raw_stage,
                    warp_idx,
                    state_left,
                    state_input_ready_l_mbar,
                    0,
                )
                update_ready_arrive(update_ready_mbar)


@cute.jit
def host_chain_dv2(
    v: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_chunks: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_w: cute.Tensor,
    ws_qk: cute.Tensor,
    ws_diag: cute.Tensor,
    state: cute.Tensor,
    out: cute.Tensor,
    stream,
    head_base: cutlass.Int32,
    launch_heads: cutlass.Int32,
    SCALE: cutlass.Float32,
    THREADS: cutlass.Constexpr,
) -> None:
    """DV2 host: identical tensor maps to the base chain host, doubled grid-y."""

    cu_seqlens_shape = cast("tuple[int | cutlass.Integer, ...]", cu_seqlens.shape)
    v_shape = cast("tuple[int | cutlass.Integer, ...]", v.shape)
    ws_kd_shape = cast("tuple[int | cutlass.Integer, ...]", ws_kd.shape)
    ws_qk_shape = cast("tuple[int | cutlass.Integer, ...]", ws_qk.shape)
    num_sequences = cu_seqlens_shape[0] - 1
    seqlen = v_shape[1]
    heads = v_shape[2]
    ws_rows = ws_kd_shape[2]
    num_chunks_total = ws_qk_shape[2]
    # Keep singleton TensorMap modes canonical for the same reason as K1.
    tile_head_stride = DK
    diag_head_stride = DIAG_REC_ELEMS
    qk_head_stride = QK_REC_ELEMS
    if heads != cutlass.Int32(1):
        tile_head_stride = DK * ws_rows
        diag_head_stride = DIAG_REC_ELEMS * num_chunks_total
        qk_head_stride = QK_REC_ELEMS * num_chunks_total
    factor_layout = cute.make_layout(
        (RAW_F16_TMA_SWIZZLE_ELEMS, ws_rows, heads, RAW_F16_TMA_SEGMENTS),
        stride=(1, DK, tile_head_stride, RAW_F16_TMA_SWIZZLE_ELEMS),
    )
    v_layout = cute.make_layout(
        (DV, seqlen, heads, 1),
        stride=(1, DV * heads, DV, DV * seqlen * heads),
    )
    diag_layout = cute.make_layout(
        (DIAG_REC_ELEMS, num_chunks_total, heads, 1),
        stride=(
            1,
            DIAG_REC_ELEMS,
            diag_head_stride,
            DIAG_REC_ELEMS,
        ),
    )
    qk_layout = cute.make_layout(
        (QK_REC_ELEMS, num_chunks_total, heads, 1),
        stride=(
            1,
            QK_REC_ELEMS,
            qk_head_stride,
            QK_REC_ELEMS,
        ),
    )
    factor_box = (
        RAW_F16_TMA_SWIZZLE_ELEMS,
        BT,
        1,
        RAW_F16_TMA_SEGMENTS,
    )
    value_box = (RAW_F16_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    tma_desc_kd = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_kd.iterator, factor_layout),
        box_dims=factor_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_w = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_w.iterator, factor_layout),
        box_dims=factor_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_qd = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_qd.iterator, factor_layout),
        box_dims=factor_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_v = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(v.iterator, v_layout),
        box_dims=value_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_diag = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_diag.iterator, diag_layout),
        box_dims=(DIAG_REC_ELEMS, 1, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_desc_qk = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_qk.iterator, qk_layout),
        box_dims=(QK_REC_ELEMS, 1, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_desc_o = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(out.iterator, v_layout),
        box_dims=(O_TMA_SWIZZLE_ELEMS, BT, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    kernel_chain_dv2(
        tma_desc_kd,
        tma_desc_w,
        tma_desc_qd,
        tma_desc_v,
        tma_desc_diag,
        tma_desc_qk,
        tma_desc_o,
        v,
        cu_seqlens,
        cu_chunks,
        state,
        out,
        head_base,
        SCALE,
    ).launch(
        grid=(num_sequences, launch_heads * 2, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
        preferred_smem_carveout=100,
    )


__all__ = ["host_chain_dv2", "kernel_chain_dv2"]
