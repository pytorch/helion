# ruff: noqa: ANN001, ANN202
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Asynchronous BT16 prefill schedule for the explicit residual/inverse DAG.

Adapted from FlashInfer v0.7.0 (4d75a33f19aa),
flashinfer/kda_kernels/kda_chunked_bt16.py. Only device primitives and the fused
host launch are retained; this module has no FlashInfer runtime dependency.
The admission matcher must prove the native rounding DAG. This schedule cannot
replace the differently rounded five-factor carrier by reassociation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as prims

from . import warp_specialized_primitives as pipeline_primitives
from .affine_recurrence_primitives import accumulator_coordinate as acc_coord
from .affine_recurrence_primitives import f16_round
from .affine_recurrence_primitives import fmul2
from .affine_recurrence_primitives import mma_blockdiag_8x8_f16
from .affine_recurrence_primitives import mma_m16n8k16_f16
from .affine_recurrence_primitives import movmatrix_b16_inline as movmatrix_b16
from .affine_recurrence_primitives import pack_f16x2
from .affine_recurrence_primitives import pack_input_b16x2_to_i32
from .affine_recurrence_primitives import pack_output_b16x2_to_i32
from .affine_recurrence_primitives import packed_f32x2_binary as packed_f32x2_binary
from .affine_recurrence_primitives import sub_b16x2_input_dtype
from .warp_specialized_plan import chained_recurrence_tmem_layout
from .warp_specialized_primitives import advance_ring_stage
from .warp_specialized_primitives import matrix_16x16_transposed_lane_coordinates
from .warp_specialized_primitives import named_barrier_sync
from .warp_specialized_primitives import segmented_swizzle_b16_element_index

pairwise_ready_arrive = pipeline_primitives.fence_async_shared_and_arrive
pairwise_ready_wait = pipeline_primitives.wait_and_flip_mbarrier
pairwise_consumed_arrive = pipeline_primitives.tcgen05_commit_mbarrier
pairwise_consumed_wait = pipeline_primitives.wait_and_flip_mbarrier
output_ready_arrive = pipeline_primitives.elect_arrive_mbarrier
output_ready_wait = pipeline_primitives.wait_and_flip_mbarrier
q_k_restore_ready_arrive = pipeline_primitives.fence_async_shared_and_arrive
q_k_restore_ready_wait = pipeline_primitives.wait_and_flip_mbarrier
cg0_k_ready_arrive = pipeline_primitives.fence_async_shared_and_arrive
cg0_k_ready_wait = pipeline_primitives.wait_and_flip_mbarrier
diag_ready_arrive = pipeline_primitives.fence_async_shared_and_arrive
diag_ready_wait = pipeline_primitives.wait_and_flip_mbarrier
raw_ready_wait = pipeline_primitives.wait_and_flip_mbarrier
checkpoint_read_done_arrive = pipeline_primitives.elect_arrive_mbarrier
checkpoint_read_done_wait = pipeline_primitives.wait_mbarrier
raw_consumed_arrive = pipeline_primitives.elect_arrive_mbarrier
raw_consumed_wait = pipeline_primitives.wait_and_flip_mbarrier
state_input_ready_arrive = pipeline_primitives.elect_arrive_mbarrier
state_input_ready_wait = pipeline_primitives.wait_and_flip_mbarrier
operand_smem_consumed_arrive = pipeline_primitives.elect_arrive_mbarrier
operand_smem_consumed_wait = pipeline_primitives.wait_and_flip_mbarrier
rhs_ready_arrive = pipeline_primitives.elect_arrive_mbarrier
rhs_ready_wait = pipeline_primitives.wait_and_flip_mbarrier
update_ready_arrive = pipeline_primitives.elect_arrive_mbarrier
update_ready_wait = pipeline_primitives.wait_and_flip_mbarrier
output_consumed_arrive = pipeline_primitives.elect_arrive_mbarrier
output_consumed_wait = pipeline_primitives.wait_and_flip_mbarrier
final_state_stored_arrive = pipeline_primitives.elect_arrive_mbarrier
final_state_stored_wait = pipeline_primitives.wait_and_flip_mbarrier
tcgen05_wait_acc_buffer_ready = pipeline_primitives.wait_and_flip_mbarrier

LOG2_E: float = 1.4426950408889634


@cute.jit
def mul_f16x2(value: cutlass.Int32, scale: cutlass.Int32) -> cutlass.Int32:
    """Multiply two packed FP16 pairs."""

    # pyrefly: ignore [bad-return]
    return prims.inline_ptx_hl(
        "mul.f16x2 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Int32],
        read_only_args=[value, scale],
    )


@cute.jit
def mul_b16x2_input_dtype(
    lhs: cutlass.Int32,
    rhs: cutlass.Int32,
    input_dtype: cutlass.Constexpr,
) -> cutlass.Int32:
    """Multiply two packed pairs using the compile-time input dtype."""

    if cutlass.const_expr(input_dtype is cutlass.BFloat16):
        return prims.mul_bf16x2(lhs, rhs)
    return mul_f16x2(lhs, rhs)


@cute.jit
def safe_gate_log2_increment_prehalved(
    half_scaled_gate: cutlass.Float32,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Float32,
) -> cutlass.Float32:
    """Apply the safe-gate transform after folding its exact half scales."""

    if cutlass.const_expr(SAFE_GATE):
        half_scale = cutlass.Float32(GATE_SCALE_LOG2 * 0.5)
        tanh_value = cute.math.tanh(half_scaled_gate, approx=True)
        return tanh_value * half_scale + half_scale
    return half_scaled_gate


@cute.jit
def softplus_log2_f32(value: cutlass.Float32) -> cutlass.Float32:
    """Compute ``softplus(value) * log2(e)`` with predicated MUFU ops.

    Evaluates the FLA non-safe activation directly in the log2 domain so the
    prefix scan never leaves base-2 units.  Matches Triton's ``x > 20``
    overflow guard: above the threshold ``softplus(x) ~= x`` and the result is
    just ``x * log2(e)`` (the ``@!p`` ops are skipped).
    """

    # pyrefly: ignore [bad-return]
    return prims.inline_ptx_hl(
        """
        {
            .reg .pred p;
            setp.gt.f32 p, {$r0}, 20.0;
            mul.f32 {$w0}, {$r0}, 1.4426950408889634;
            @!p ex2.approx.ftz.f32 {$w0}, {$w0};
            @!p add.f32 {$w0}, {$w0}, 1.0;
            @!p lg2.approx.ftz.f32 {$w0}, {$w0};
        }
        """,
        write_only_types=[cutlass.Float32],
        read_only_args=[value],
    )


BT: int = 16


DK: int = 128


DV: int = 128
DV_HALF: int = DV // 2
ROWS_PER_WARP: int = DV_HALF // 4


L2_NORM_EPS: float = 1.0e-12


THREADS_PER_WARP: int = 32


THREADS_PER_CTA: int = 16 * THREADS_PER_WARP


CG0_GROUP_COUNT: int = 2


CG0_WARPS_PER_GROUP: int = 4


CG0_THREADS_PER_GROUP: int = CG0_WARPS_PER_GROUP * THREADS_PER_WARP


NBAR_CG0_GROUP0_ID: int = 1


TMEM_USER_WARP_COUNT: int = 5


TMEM_USER_THREADS: int = TMEM_USER_WARP_COUNT * THREADS_PER_WARP


NBAR_TMEM_LIFECYCLE_ID: int = 2


NBAR_CG0_GROUP1_ID: int = 3


NBAR_CG0_GROUP2_ID: int = 4


KDA_CG0_REGS: int = 160


KDA_CG1_REGS: int = 136


KDA_SERVICE_REGS: int = 56


TCGEN05_F16_K_ATOM: int = 16


TCGEN05_F16_ELEM_BYTES: int = 2


TCGEN05_F16_A_TMEM_PAIR_XOR: int = 4


TCGEN05_SW128_BYTES: int = 128


TCGEN05_SW128_K_PHASES_PER_SLICE: int = 4


TCGEN05_STATE_K_B_LEADING_BYTES: int = 16


TCGEN05_STATE_K_B_STRIDE_BYTES: int = 1024


TCGEN05_STATE_K_B_K_STEP_BYTES: int = TCGEN05_F16_K_ATOM * TCGEN05_F16_ELEM_BYTES


TCGEN05_SW32_BT_HALF_XOR: int = BT // 2


PAIRWISE_SW32_ROW_STRIDE: int = BT


PAIRWISE_SW32_COL_XOR: int = TCGEN05_SW32_BT_HALF_XOR


PAIRWISE_SW32_TILE_ELEMS: int = BT * BT


TCGEN05_TMEM_LOAD_COLS: int = BT


TCGEN05_STATE_INPUT_LOAD_COLS: int = 16


TCGEN05_STATE_INPUT_PACKED_COLS: int = TCGEN05_STATE_INPUT_LOAD_COLS // 2


TCGEN05_STATE_K_TMEM_ROW_BLOCKS: int = DV // THREADS_PER_WARP


KDA_TMEM_SHARED_INPUT_STAGE_COUNT: int = 2


KDA_TMEM_QSTATE_ACC_STAGE_COUNT: int = 2


KDA_TMEM_SHARED_ACC_STAGE_COUNT: int = 2

_TMEM_LAYOUT = chained_recurrence_tmem_layout(
    state_width=DK,
    step_width=BT,
    factor_input_stages=KDA_TMEM_SHARED_INPUT_STAGE_COUNT,
    auxiliary_accumulator_stages=KDA_TMEM_SHARED_ACC_STAGE_COUNT,
)
KDA_TMEM_N16_ACC_COLS = _TMEM_LAYOUT.region("primary_accumulator").columns
KDA_TMEM_N128_ACC_COLS = _TMEM_LAYOUT.region("state").columns
KDA_TMEM_STATE_COLS = KDA_TMEM_N128_ACC_COLS
KDA_TMEM_STATE_AS_INPUT_COLS = _TMEM_LAYOUT.region("state_input").columns
KDA_TMEM_SHARED_INPUT_COLS = _TMEM_LAYOUT.region("factor_input").columns
KDA_TMEM_STATE_COL_OFFSET = _TMEM_LAYOUT.region("state").column_offset
KDA_TMEM_FINAL_STATE_ACC_COL_OFFSET = KDA_TMEM_STATE_COL_OFFSET
KDA_TMEM_STATE_AS_INPUT_COL_OFFSET = _TMEM_LAYOUT.region("state_input").column_offset
KDA_TMEM_SHARED_INPUT_COL_OFFSET = _TMEM_LAYOUT.region("factor_input").column_offset
KDA_TMEM_QSTATE_ACC_COL_OFFSET = _TMEM_LAYOUT.region(
    "primary_accumulator"
).column_offset
KDA_TMEM_SHARED_ACC_COL_OFFSET = _TMEM_LAYOUT.region(
    "auxiliary_accumulator"
).column_offset
KDA_TMEM_QSTATE_ACC_STAGE1_COL_OFFSET = _TMEM_LAYOUT.region(
    "secondary_accumulator"
).column_offset


KDA_TMEM_QSTATE_ACC_STAGE_STRIDE_COLS: int = (
    KDA_TMEM_QSTATE_ACC_STAGE1_COL_OFFSET - KDA_TMEM_QSTATE_ACC_COL_OFFSET
)


KDA_TMEM_LAYOUT_COLS = _TMEM_LAYOUT.required_columns
KDA_TMEM_ALLOC_COLS = _TMEM_LAYOUT.allocated_columns


@cute.jit
def cta_sync() -> None:
    """Synchronize all threads in the CTA."""

    named_barrier_sync(0, THREADS_PER_CTA)


@cute.jit
def cg0_sync(cg0_group_id) -> None:
    """Synchronize one four-warp CG0 producer group."""

    if cg0_group_id == 0:
        named_barrier_sync(
            NBAR_CG0_GROUP0_ID,
            CG0_THREADS_PER_GROUP,
        )
    elif cg0_group_id == 1:
        named_barrier_sync(
            NBAR_CG0_GROUP1_ID,
            CG0_THREADS_PER_GROUP,
        )
    else:
        named_barrier_sync(
            NBAR_CG0_GROUP2_ID,
            CG0_THREADS_PER_GROUP,
        )


@cute.jit
def tmem_user_sync() -> None:
    """Named barrier for CG1 plus the tcgen05 warp during TMEM lifecycle setup."""

    named_barrier_sync(NBAR_TMEM_LIFECYCLE_ID, TMEM_USER_THREADS)


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


@cute.jit
def warp_group_sum_8(value: cutlass.Float32) -> cutlass.Float32:
    """Reduce independent 8-lane row groups inside one warp."""

    value = value + cutlass.Float32(
        prims.shfl_sync(cute.arch.FULL_MASK, value, 4, 0x1F, prims.Shfl.BFLY)
    )
    value = value + cutlass.Float32(
        prims.shfl_sync(cute.arch.FULL_MASK, value, 2, 0x1F, prims.Shfl.BFLY)
    )
    return value + cutlass.Float32(
        prims.shfl_sync(cute.arch.FULL_MASK, value, 1, 0x1F, prims.Shfl.BFLY)
    )


@cute.jit
def mma_input_dtype(
    value: cutlass.Float32,
    input_dtype: cutlass.Constexpr,
) -> cutlass.Float32:
    """Round a scalar through the compile-time input dtype for MMA reuse."""

    return value.to(input_dtype).to(cutlass.Float32)


@cute.jit
def pairwise_eye(row_coord, col_coord) -> cutlass.Float32:
    """Return the 16x16 identity value used by inverse product factors."""

    return cutlass.Float32(1.0) if row_coord == col_coord else cutlass.Float32(0.0)


@cute.jit
def pairwise_sw32_smem_index(offset: cutlass.Constexpr, row_coord, col_coord):
    """Return the SW32 physical index for a logical pairwise tile element."""

    storage_col_coord = col_coord ^ PAIRWISE_SW32_COL_XOR
    return offset + tcgen05_swizzle_32b_elem_index(
        row_coord * PAIRWISE_SW32_ROW_STRIDE + storage_col_coord,
        TCGEN05_F16_ELEM_BYTES,
    )


@cute.jit
def pairwise_stmatrix_m8n8x4_ptr(
    pairwise_smem,
    offset: cutlass.Constexpr,
    lane,
):
    """Return the row-start pointer for a 16x16 F16 pairwise STSM store."""

    matrix_id = lane // 8
    row_coord = lane & 7
    col_coord = cutlass.Int32(0)
    if matrix_id & 1:
        row_coord = row_coord + cutlass.Int32(8)
    if matrix_id >= 2:
        col_coord = cutlass.Int32(8)
    return pairwise_smem.subview(
        pairwise_sw32_smem_index(offset, row_coord, col_coord)
    ).data_ptr()


@cute.jit
def tcgen05_swizzle_128b_elem_index(
    linear_elem_idx,
    elem_bytes: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    """Return the K-box-major SW128 physical element index."""

    elems_per_128b = 128 // elem_bytes
    row_coord = linear_elem_idx // DK
    col_coord = linear_elem_idx - row_coord * DK
    slice_coord = col_coord // elems_per_128b
    col_in_slice = col_coord - slice_coord * elems_per_128b
    slice_linear_idx = row_coord * elems_per_128b + col_in_slice
    byte_offset = slice_linear_idx * elem_bytes
    swizzle_mask = ((byte_offset >> 7) & 0x7) << 4
    return slice_coord * rows * elems_per_128b + (
        (byte_offset ^ swizzle_mask) // elem_bytes
    )


@cute.jit
def tcgen05_swizzle_32b_elem_index(
    linear_elem_idx,
    elem_bytes: cutlass.Constexpr,
):
    """Return the SW32 physical element index for a row-major logical tile."""

    byte_offset = linear_elem_idx * elem_bytes
    # SW32 is CuTe Swizzle<1,4,3>: xor address bit 7 into bit 4.
    swizzle_mask = ((byte_offset >> 7) & 0x1) << 4
    return (byte_offset ^ swizzle_mask) // elem_bytes


@cute.jit
def tcgen05_decay_b_key_storage_dim_runtime(token_coord, key_dim):
    """Return the runtime key coordinate for tcgen05 SW128 decay operands.

    Only the constant half-atom interleave applies.  The former extra
    token-dependent XOR (``(token & 2) * K_ATOM``) was matched by the warp
    ldmatrix reader (self-consistent, so the intra-chunk path stayed clean)
    but NOT by the tcgen05 state MMA, which reads this tile through a
    standard-layout SMEM descriptor: tokens with ``t % 4 in {2, 3}`` read a
    decay row displaced by 32 key channels every chunk.  The error scales
    with state persistence (large-negative ``dt_bias``), which is why random
    small-gate tests never caught it.
    """

    key_mask = cutlass.Int32(TCGEN05_F16_K_ATOM // 2)
    return key_dim ^ key_mask


O_STAGE_COUNT: int = 2


O_STAGE_COLS: int = DV


O_OUT_OFFSET: int = 0


O_ELEM_BYTES: int = TCGEN05_F16_ELEM_BYTES


O_TMA_SWIZZLE_BYTES: int = 128


O_TMA_SWIZZLE_ELEMS: int = O_TMA_SWIZZLE_BYTES // O_ELEM_BYTES


O_TMA_SWIZZLE_GROUP_BYTES: int = 16


O_TMA_SWIZZLE_GROUP_ELEMS: int = O_TMA_SWIZZLE_GROUP_BYTES // O_ELEM_BYTES


O_TMA_SWIZZLE_ROW_MASK: int = (O_TMA_SWIZZLE_ELEMS // O_TMA_SWIZZLE_GROUP_ELEMS) - 1


O_TMA_SEGMENTS: int = DV // O_TMA_SWIZZLE_ELEMS


O_TMA_SWIZZLE_ALIGNMENT_BYTES: int = O_TMA_SWIZZLE_BYTES * (
    O_TMA_SWIZZLE_ELEMS // O_TMA_SWIZZLE_GROUP_ELEMS
)


O_SMEM_STAGE_SIZE: int = BT * O_STAGE_COLS


O_SMEM_TILE_SIZE: int = O_STAGE_COUNT * O_SMEM_STAGE_SIZE


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


DECAY_STAGE_COUNT: int = 2


Q_K_RESTORE_READY_STAGE_COUNT: int = 3


TCGEN05_K_DECAY_STAGE_SIZE: int = DK * BT


TCGEN05_Q_DECAY_STAGE_SIZE: int = DK * BT


TCGEN05_K_RESTORE_STAGE_SIZE: int = DK * BT


TCGEN05_K_DECAY_SMEM_TILE_SIZE: int = DECAY_STAGE_COUNT * TCGEN05_K_DECAY_STAGE_SIZE


TCGEN05_Q_DECAY_SMEM_TILE_SIZE: int = DECAY_STAGE_COUNT * TCGEN05_Q_DECAY_STAGE_SIZE


TCGEN05_K_RESTORE_SMEM_TILE_SIZE: int = DECAY_STAGE_COUNT * TCGEN05_K_RESTORE_STAGE_SIZE


RAW_STAGE_COUNT: int = 8


TMA_MBAR_STAGE_COUNT: int = RAW_STAGE_COUNT


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


RAW_F16_TAIL_ZERO_LANES: int = DK // RAW_F16_TMA_SWIZZLE_GROUP_ELEMS


RAW_F32_ELEM_BYTES: int = 4


RAW_F32_TMA_SWIZZLE_BYTES: int = 128


RAW_F32_TMA_SWIZZLE_ELEMS: int = RAW_F32_TMA_SWIZZLE_BYTES // RAW_F32_ELEM_BYTES


RAW_F32_TMA_SWIZZLE_GROUP_BYTES: int = 16


RAW_F32_TMA_SWIZZLE_GROUP_ELEMS: int = (
    RAW_F32_TMA_SWIZZLE_GROUP_BYTES // RAW_F32_ELEM_BYTES
)


CG0_TOKEN_ROWS_PER_WARP: int = BT // CG0_WARPS_PER_GROUP


RAW_F32_TMA_SWIZZLE_ROW_MASK: int = (
    RAW_F32_TMA_SWIZZLE_ELEMS // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
) - 1


RAW_F32_TMA_SEGMENTS: int = DK // RAW_F32_TMA_SWIZZLE_ELEMS


RAW_F32_TMA_SEGMENT_ELEMS: int = BT * RAW_F32_TMA_SWIZZLE_ELEMS


RAW_F32_TMA_SWIZZLE_ALIGNMENT_BYTES: int = RAW_F32_TMA_SWIZZLE_BYTES * (
    RAW_F32_TMA_SWIZZLE_ELEMS // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
)


RAW_F32_TAIL_ZERO_LANES: int = DK // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS


RAW_Q_STAGE_SIZE: int = DK * BT


RAW_K_STAGE_SIZE: int = DK * BT


RAW_V_STAGE_SIZE: int = DV * BT


RAW_GATE_STAGE_SIZE: int = DK * BT


GATE_EXCHANGE_STAGE_COUNT: int = 4


GATE_EXCHANGE_STAGE_SIZE: int = DK * BT


RAW_BETA_STAGE_SIZE: int = BT


RAW_DT_BIAS_STAGE_SIZE: int = DK


RAW_DT_BIAS_A_LOG_EXP_OFFSET: int = RAW_DT_BIAS_STAGE_SIZE


RAW_Q_SMEM_TILE_SIZE: int = RAW_STAGE_COUNT * RAW_Q_STAGE_SIZE


RAW_K_SMEM_TILE_SIZE: int = RAW_STAGE_COUNT * RAW_K_STAGE_SIZE


RAW_V_SMEM_TILE_SIZE: int = RAW_STAGE_COUNT * RAW_V_STAGE_SIZE


RAW_GATE_SMEM_TILE_SIZE: int = RAW_STAGE_COUNT * RAW_GATE_STAGE_SIZE


GATE_EXCHANGE_SMEM_TILE_SIZE: int = GATE_EXCHANGE_STAGE_COUNT * GATE_EXCHANGE_STAGE_SIZE


RAW_BETA_SMEM_TILE_SIZE: int = RAW_STAGE_COUNT * RAW_BETA_STAGE_SIZE


RAW_DT_BIAS_SMEM_TILE_SIZE: int = RAW_DT_BIAS_STAGE_SIZE + 1


def gate_dtype_is_f32(gate_dtype) -> bool:
    """True when the gate rides the historical FP32 memory format."""

    return gate_dtype is cutlass.Float32


K_INV_STAGE_SIZE: int = BT * DK


K_INV_SMEM_TILE_SIZE: int = DECAY_STAGE_COUNT * K_INV_STAGE_SIZE


PAIRWISE_SMEM_QK_OFFSET: int = 0


PAIRWISE_SMEM_AINV_OFFSET: int = PAIRWISE_SMEM_QK_OFFSET + PAIRWISE_SW32_TILE_ELEMS


PAIRWISE_SMEM_STAGE_SIZE: int = PAIRWISE_SMEM_AINV_OFFSET + PAIRWISE_SW32_TILE_ELEMS


PAIRWISE_STAGE_COUNT: int = 2


PAIRWISE_SMEM_TILE_SIZE: int = PAIRWISE_STAGE_COUNT * PAIRWISE_SMEM_STAGE_SIZE


SUPER_MMA_ATOM_N: int = 8


SUPER_MMA_ATOM_K: int = 16


SUPER_MMA_K_BLOCKS: int = DK // SUPER_MMA_ATOM_K


SUPER_MMA_ACCUMULATORS_PER_LANE: int = 4


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

    return segmented_swizzle_b16_element_index(
        token_coord,
        dim,
        BT,
        RAW_F16_TMA_SWIZZLE_ELEMS,
        RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        RAW_F16_TMA_SWIZZLE_ROW_MASK,
    )


@cute.jit
def raw_f32_s128_smem_index(token_coord, dim):
    """Return the physical s128 SMEM index for raw F32 gate staging."""

    segment = dim // RAW_F32_TMA_SWIZZLE_ELEMS
    segment_dim = dim - segment * RAW_F32_TMA_SWIZZLE_ELEMS
    col_group = segment_dim // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
    col_in_group = segment_dim - col_group * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
    row_swizzle = token_coord & RAW_F32_TMA_SWIZZLE_ROW_MASK
    return (
        segment * RAW_F32_TMA_SEGMENT_ELEMS
        + token_coord * RAW_F32_TMA_SWIZZLE_ELEMS
        + ((col_group ^ row_swizzle) * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS)
        + col_in_group
    )


def raw_f32_exchange_smem_index(token_coord, dim):
    """Return a conflict-free SMEM index for the CG0 gate-prefix exchange.

    The TMA s128 swizzle XORs banks only within one 128-byte segment, so the
    vectorized prefix re-reads hit the same bank group once per segment
    (2-way conflicts). The exchange scratch is FP32 and either reuses the
    raw-gate stage after its raw values are consumed (FP32 gate) or is a
    dedicated `gate_exchange_smem` buffer (16-bit gate), so both sides of the
    exchange can add a segment XOR that spreads segments across bank groups;
    column-wise prefix stores stay conflict-free because the segment term is
    constant per warp.

    NOTE the layout math stays on the RAW_F32 s128 geometry on BOTH gate paths:
    this buffer is FP32 and independent of the TMA landing layout.
    """

    segment = dim // RAW_F32_TMA_SWIZZLE_ELEMS
    segment_dim = dim - segment * RAW_F32_TMA_SWIZZLE_ELEMS
    col_group = segment_dim // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
    col_in_group = segment_dim - col_group * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
    row_swizzle = token_coord & RAW_F32_TMA_SWIZZLE_ROW_MASK
    return (
        segment * RAW_F32_TMA_SEGMENT_ELEMS
        + token_coord * RAW_F32_TMA_SWIZZLE_ELEMS
        + ((col_group ^ row_swizzle ^ segment) * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS)
        + col_in_group
    )


@cute.jit
def k_inv_s128_smem_index(token_coord, key_dim):
    """Return the physical s128 SMEM index for the K-inverse auxiliary-MMA RHS."""

    return raw_f16_s128_smem_index(token_coord, key_dim)


@cute.jit
def o_smem_swizzle_128b_elem_index(
    o_stage_base,
    value_dim,
    token_coord,
):
    """Return the physical W128 SMEM index for one staged output element."""

    return (
        o_stage_base
        + O_OUT_OFFSET
        + segmented_swizzle_b16_element_index(
            token_coord,
            value_dim,
            BT,
            O_TMA_SWIZZLE_ELEMS,
            O_TMA_SWIZZLE_GROUP_ELEMS,
            O_TMA_SWIZZLE_ROW_MASK,
        )
    )


@cute.jit
def o_smem_stmatrix_128b_ptr(
    o_smem,
    o_stage_base,
    value_dim_base,
    lane,
):
    """Return the per-lane W128 row-start pointer for one 16x16 STSM.T tile."""

    token_coord, value_offset = matrix_16x16_transposed_lane_coordinates(lane)
    value_dim = value_dim_base + value_offset
    smem_idx = o_smem_swizzle_128b_elem_index(
        o_stage_base,
        value_dim,
        token_coord,
    )
    return o_smem.subview(smem_idx).data_ptr()


@cute.jit
def raw_v_ldmatrix_trans_ptr(raw_v_smem, value_dim_base, lane):
    """Return the per-lane row-start pointer for raw V `ldmatrix.x4.trans`."""

    token_coord, value_offset = matrix_16x16_transposed_lane_coordinates(lane)
    value_dim = value_dim_base + value_offset
    smem_idx = raw_f16_s128_smem_index(token_coord, value_dim)
    return raw_v_smem.subview(smem_idx).data_ptr()


BETA_TILE_STAGE_ELEMS: int = 256  # bf16; >= max(8*BT group, heads*(BT+g) pair)


BETA_TILE_STAGE_COUNT: int = 8


BETA_TMA_LOOKAHEAD: int = 4


@cute.jit
def tma_stage_load_inputs(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_gate: cutlass.GridConstant[cuda.TensorMap],
    beta_tile_stage,
    beta_tile_mbar,
    beta_tile_phase,
    beta_g,
    heads32,
    raw_q_smem,
    raw_k_smem,
    raw_v_smem,
    raw_gate_smem,
    raw_beta_smem,
    sequence_start,
    head_idx,
    lane,
    chunk_start,
    seqlen,
    tma_mbar,
    tma_tx_bytes: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    """Overlap raw TMA with beta readiness and scalar sigmoid publication.

    expect_tx leaves the slot's one arrival pending. Only the final release
    arrival, after beta stores and a full warp sync, can complete the phase;
    consumers still wait for both raw transactions and beta publication.
    """

    global_chunk_start = sequence_start + chunk_start
    if prims.elect_sync():
        prims.mbarrier_expect_tx(tma_mbar, tma_tx_bytes)
    if prims.elect_sync():
        for segment in cutlass.range_constexpr(RAW_F16_TMA_SEGMENTS):
            tma_coord = (
                cutlass.Int32(segment * RAW_F16_TMA_SWIZZLE_ELEMS),
                global_chunk_start,
                head_idx,
                cutlass.Int32(0),
            )
            smem_offset = segment * RAW_F16_TMA_SEGMENT_ELEMS
            prims.cp_async_bulk_tensor_shared_cta_global(
                raw_q_smem.subview(smem_offset),
                tma_desc_q.get_ptr(),
                tma_coord,
                tma_mbar,
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                raw_k_smem.subview(smem_offset),
                tma_desc_k.get_ptr(),
                tma_coord,
                tma_mbar,
            )
            prims.cp_async_bulk_tensor_shared_cta_global(
                raw_v_smem.subview(smem_offset),
                tma_desc_v.get_ptr(),
                tma_coord,
                tma_mbar,
            )
            if cutlass.const_expr(not gate_dtype_is_f32(gate_dtype)):
                # 16-bit gate: it rides the SAME 16-bit s128 box/swizzle family
                # as q/k/v, so it folds into this loop (2 x 128 B segments
                # instead of the FP32 path's 4 x 128 B segments).
                prims.cp_async_bulk_tensor_shared_cta_global(
                    raw_gate_smem.subview(smem_offset),
                    tma_desc_gate.get_ptr(),
                    tma_coord,
                    tma_mbar,
                )

        if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
            for segment in cutlass.range_constexpr(RAW_F32_TMA_SEGMENTS):
                tma_coord = (
                    cutlass.Int32(segment * RAW_F32_TMA_SWIZZLE_ELEMS),
                    global_chunk_start,
                    head_idx,
                    cutlass.Int32(0),
                )
                smem_offset = segment * RAW_F32_TMA_SEGMENT_ELEMS
                prims.cp_async_bulk_tensor_shared_cta_global(
                    raw_gate_smem.subview(smem_offset),
                    tma_desc_gate.get_ptr(),
                    tma_coord,
                    tma_mbar,
                )

    tma_transfer_wait(beta_tile_mbar, beta_tile_phase)
    if lane < BT:
        token_idx = chunk_start + lane
        beta_value = cutlass.Float32(0.0)
        if token_idx < seqlen:
            # The tile is a contiguous [token][head-slot] slab staged by TMA:
            # g == 1 -> 8-head group rows (slot = head % 8, token = lane);
            # g > 1  -> whole packed rows from token (start - start % g)
            #           (slot = head, token = start % g + lane).
            b_idx = lane * cutlass.Int32(8) + head_idx % cutlass.Int32(8)
            if beta_g > cutlass.Int32(1):
                b_idx = (sequence_start % beta_g + lane) * heads32 + head_idx
            beta_logit = beta_tile_stage[b_idx].to(cutlass.Float32)
            half = cutlass.Float32(0.5)
            beta_value = cute.math.tanh(beta_logit * half, approx=True) * half + half
        raw_beta_smem[lane] = beta_value
    prims.bar_warp_sync(cute.arch.FULL_MASK)
    if prims.elect_sync():
        prims.mbarrier_arrive(tma_mbar)


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
def cg0_zero_tail_raw_operands(
    raw_q_smem,
    raw_k_smem,
    raw_v_smem,
    raw_gate_smem,
    lane,
    chunk_start,
    seqlen,
    input_dtype: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    """Zero padded tail rows after TMA has produced the raw input tile.

    `gate_dtype` is carried separately from `input_dtype` because the gate has
    its own ABI dtype: a 16-bit gate shares the q/k/v s128 geometry (and can be
    zeroed in the same lane block), an FP32 gate keeps its own wider geometry.
    """

    raw_q_ptr = raw_q_smem.data_ptr()
    raw_k_ptr = raw_k_smem.data_ptr()
    raw_v_ptr = raw_v_smem.data_ptr()
    raw_gate_ptr = raw_gate_smem.data_ptr()
    f16_zero = input_dtype(0.0)
    f16_zero_vec = cutlass.Vector.from_elements(
        (
            f16_zero,
            f16_zero,
            f16_zero,
            f16_zero,
            f16_zero,
            f16_zero,
            f16_zero,
            f16_zero,
        ),
        input_dtype,
    )
    if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
        f32_zero = cutlass.Float32(0.0)
        gate_zero_vec = cutlass.Vector.from_elements(
            (f32_zero, f32_zero, f32_zero, f32_zero),
            cutlass.Float32,
        )
    else:
        gate_zero = gate_dtype(0.0)
        gate_zero_vec = cutlass.Vector.from_elements(
            (
                gate_zero,
                gate_zero,
                gate_zero,
                gate_zero,
                gate_zero,
                gate_zero,
                gate_zero,
                gate_zero,
            ),
            gate_dtype,
        )
    for row in cutlass.range_constexpr(BT):
        token_idx = chunk_start + cutlass.Int32(row)
        if token_idx >= seqlen:
            if lane < RAW_F16_TAIL_ZERO_LANES:
                f16_dim_base = lane * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
                f16_idx = raw_f16_s128_smem_index(row, f16_dim_base)
                (raw_q_ptr + f16_idx).store(
                    f16_zero_vec,
                    alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
                )
                (raw_k_ptr + f16_idx).store(
                    f16_zero_vec,
                    alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
                )
                (raw_v_ptr + f16_idx).store(
                    f16_zero_vec,
                    alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
                )
                if cutlass.const_expr(not gate_dtype_is_f32(gate_dtype)):
                    (raw_gate_ptr + f16_idx).store(
                        gate_zero_vec,
                        alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
                    )
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                if lane < RAW_F32_TAIL_ZERO_LANES:
                    f32_dim_base = lane * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
                    f32_idx = raw_f32_s128_smem_index(row, f32_dim_base)
                    (raw_gate_ptr + f32_idx).store(
                        gate_zero_vec,
                        alignment=RAW_F32_TMA_SWIZZLE_GROUP_BYTES,
                    )


@cute.jit
def cg0_materialize_decay_operands(
    raw_q_smem,
    raw_k_smem,
    raw_gate_smem,
    gate_exchange_smem,
    a_log_exp,
    dt_bias_value,
    k_inv_smem,
    tcgen05_k_decay_smem,
    tcgen05_q_decay_smem,
    tcgen05_k_restore_smem,
    cg0_k_ready_stage_mbar,
    cg0_k_half_ready_stage_mbar,
    diag_ready_stage_mbar,
    operand_smem_consumed_stage_mbar,
    k_restore_consumed_stage_mbar,
    chunk,
    seqlen,
    input_dtype: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Float32,
    FULL_CHUNKS: cutlass.Constexpr,
    cg0_group_id,
    cg0_local_warp,
    lane,
) -> None:
    """Materialize safe-gate KDA decay operands for one key dimension.

    `FULL_CHUNKS` is the engine peel's per-call selector (True for a guard-free
    interior chunk, False for the peeled partial tail), NOT a compile-time
    specialization: the guard-free interior skips the masked-tail fixup below.
    """

    row_group_start = cg0_local_warp * CG0_TOKEN_ROWS_PER_WARP
    lane_row_group = lane // RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
    lane_in_row_group = lane - lane_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
    decay_row = row_group_start + lane_row_group

    raw_q_ptr = raw_q_smem.data_ptr()
    raw_k_ptr = raw_k_smem.data_ptr()
    # The FP32 gate-prefix exchange buffer.  With an FP32 gate the caller
    # passes the raw-gate stage itself (the historical aliasing); with a 16-bit
    # gate it is the dedicated `gate_exchange_smem` tile.
    g_prefix_ptr = gate_exchange_smem.data_ptr()
    k_inv_ptr = k_inv_smem.data_ptr()
    tcgen05_k_decay_ptr = tcgen05_k_decay_smem.data_ptr()
    tcgen05_q_decay_ptr = tcgen05_q_decay_smem.data_ptr()
    tcgen05_k_restore_ptr = tcgen05_k_restore_smem.data_ptr()

    prefix_dim = cg0_local_warp * THREADS_PER_WARP + lane
    g_prefix_regs = cutlass.Array(
        cutlass.Float32,
        BT,
        alignment=16,
    )
    if cutlass.const_expr(SAFE_GATE):
        # Fold tanh's exact 0.5 scale into the chunk-uniform coefficient.
        a_log_exp_half = a_log_exp * cutlass.Float32(0.5)
        for row_pair in cutlass.range_constexpr(BT // 2):
            row0 = row_pair * 2
            row1 = row0 + 1
            # Only the MEMORY FORMAT of the raw gate depends on gate_dtype:
            # a 16-bit gate lands in the q/k s128 geometry and is widened to
            # FP32 right here -- ALL gate arithmetic below stays FP32.
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                prefix_idx0 = raw_f32_s128_smem_index(row0, prefix_dim)
                prefix_idx1 = raw_f32_s128_smem_index(row1, prefix_dim)
                gate0 = raw_gate_smem[prefix_idx0]
                gate1 = raw_gate_smem[prefix_idx1]
            else:
                prefix_idx0 = raw_f16_s128_smem_index(row0, prefix_dim)
                prefix_idx1 = raw_f16_s128_smem_index(row1, prefix_dim)
                gate0 = raw_gate_smem[prefix_idx0].to(cutlass.Float32)
                gate1 = raw_gate_smem[prefix_idx1].to(cutlass.Float32)
            gate0 = a_log_exp_half * (gate0 + dt_bias_value)
            gate1 = a_log_exp_half * (gate1 + dt_bias_value)
            gate0 = safe_gate_log2_increment_prehalved(
                gate0,
                SAFE_GATE,
                GATE_SCALE_LOG2,
            )
            gate1 = safe_gate_log2_increment_prehalved(
                gate1,
                SAFE_GATE,
                GATE_SCALE_LOG2,
            )
            gate_pair = cutlass.Vector.from_elements((gate0, gate1), cutlass.Float32)
            g_prefix_regs[row0] = gate_pair[0]
            g_prefix_regs[row1] = gate_pair[1]
    else:
        # FLA-compatible non-safe gate: accept raw gate logits + A_log +
        # dt_bias and compute the log2-domain decay increment in-kernel as
        # ``-exp(A_log) * softplus(raw_gate + dt_bias) * log2(e)``.  a_log_exp
        # already carries ``exp2(A_log * log2(e)) == exp(A_log)`` and the
        # softplus is evaluated in base-2 units (softplus_log2_f32).
        for row in cutlass.range_constexpr(BT):
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                prefix_idx = raw_f32_s128_smem_index(row, prefix_dim)
                gate = raw_gate_smem[prefix_idx]
            else:
                prefix_idx = raw_f16_s128_smem_index(row, prefix_dim)
                gate = raw_gate_smem[prefix_idx].to(cutlass.Float32)
            gate = -a_log_exp * softplus_log2_f32(gate + dt_bias_value)
            g_prefix_regs[row] = gate

    if cutlass.const_expr(not FULL_CHUNKS):
        # tail-block peeling: interior (full) chunks run the lean
        # unconditional gate stream above; only a sequence's genuinely
        # partial tail chunk (warp-uniform runtime condition, one compare
        # per chunk) pays the masked zeroing fixup below.
        tail_valid_rows = seqlen - chunk * cutlass.Int32(BT)
        if tail_valid_rows < cutlass.Int32(BT):
            tail_mask_pt = cutlass.vector.create_mask([BT], [tail_valid_rows])
            for row_pair_pt in cutlass.range_constexpr(BT // 2):
                row0_pt = row_pair_pt * 2
                row1_pt = row0_pt + 1
                gate_pair_pt = cutlass.Vector.from_elements(
                    (g_prefix_regs[row0_pt], g_prefix_regs[row1_pt]),
                    cutlass.Float32,
                )
                gate_pair_pt = cutlass.vector.where(
                    tail_mask_pt[row0_pt : row1_pt + 1], gate_pair_pt, 0.0
                )
                g_prefix_regs[row0_pt] = gate_pair_pt[0]
                g_prefix_regs[row1_pt] = gate_pair_pt[1]

    prefix_acc = cutlass.Float32(0.0)
    for row_pair in cutlass.range_constexpr(BT // 2):
        row0 = row_pair * 2
        row1 = row0 + 1
        # The scalar scan has the same dependency depth as packed fadd2 but
        # avoids the register copy needed to construct its first input pair.
        prefix0 = prefix_acc + g_prefix_regs[row0]
        prefix1 = prefix0 + g_prefix_regs[row1]
        g_prefix_regs[row0] = prefix0
        g_prefix_regs[row1] = prefix1
        prefix_acc = prefix1

    for row in cutlass.range_constexpr(BT):
        g_prefix_regs[row] = cute.math.exp2(g_prefix_regs[row], fastmath=True)

    for row in cutlass.range_constexpr(BT):
        prefix_idx = raw_f32_exchange_smem_index(row, prefix_dim)
        gate_exchange_smem[prefix_idx] = g_prefix_regs[row]

    cg0_sync(cg0_group_id)
    diag_ready_arrive(diag_ready_stage_mbar)

    k_inv_regs = cutlass.Array(
        input_dtype,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    k_restore_all_regs = cutlass.Array(
        input_dtype,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    raw_q_regs = cutlass.Array(
        cutlass.Float32,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    raw_k_regs = cutlass.Array(
        cutlass.Float32,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    q_sum_sq = cutlass.Float32(0.0)
    k_sum_sq = cutlass.Float32(0.0)
    for dim_half in cutlass.range_constexpr(2):
        dim_base = (
            dim_half * (DK // 2) + lane_in_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        )
        reg_base = dim_half * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        raw_f16_idx = raw_f16_s128_smem_index(decay_row, dim_base)
        raw_q_vec = (raw_q_ptr + raw_f16_idx).load(
            count=RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        raw_k_vec = (raw_k_ptr + raw_f16_idx).load(
            count=RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        raw_q_vec_f32 = raw_q_vec.to(cutlass.Float32)
        raw_k_vec_f32 = raw_k_vec.to(cutlass.Float32)
        for dim_offset in cutlass.range_constexpr(RAW_F16_TMA_SWIZZLE_GROUP_ELEMS):
            q_val = raw_q_vec_f32[dim_offset]
            k_val = raw_k_vec_f32[dim_offset]
            raw_q_regs[reg_base + dim_offset] = q_val
            raw_k_regs[reg_base + dim_offset] = k_val
            q_sum_sq = q_sum_sq + q_val * q_val
            k_sum_sq = k_sum_sq + k_val * k_val

    q_sum_sq = warp_group_sum_8(q_sum_sq)
    k_sum_sq = warp_group_sum_8(k_sum_sq)
    norm_floor_sq = cutlass.Float32(L2_NORM_EPS * L2_NORM_EPS)
    q_inv_norm = cute.math.rsqrt(
        cute.math.max(q_sum_sq, norm_floor_sq, ftz=True),
        fastmath=True,
    )
    k_inv_norm = cute.math.rsqrt(
        cute.math.max(k_sum_sq, norm_floor_sq, ftz=True),
        fastmath=True,
    )

    exp_g_regs = cutlass.Array(
        cutlass.Float32,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    exp_g_last_regs = cutlass.Array(
        cutlass.Float32,
        2 * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
        alignment=16,
    )
    for dim_half in cutlass.range_constexpr(2):
        dim_base = (
            dim_half * (DK // 2) + lane_in_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        )
        reg_base = dim_half * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        exp_neg_g_regs = cutlass.Array(
            cutlass.Float32,
            RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=16,
        )
        for f32_group in cutlass.range_constexpr(
            RAW_F16_TMA_SWIZZLE_GROUP_ELEMS // RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
        ):
            f32_dim_base = dim_base + f32_group * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
            g_prefix_idx = raw_f32_exchange_smem_index(decay_row, f32_dim_base)
            exp_g_vec = (g_prefix_ptr + g_prefix_idx).load(
                count=RAW_F32_TMA_SWIZZLE_GROUP_ELEMS,
                alignment=RAW_F32_TMA_SWIZZLE_GROUP_BYTES,
            )
            exp_g_last_idx = raw_f32_exchange_smem_index(BT - 1, f32_dim_base)
            exp_g_last_vec = (g_prefix_ptr + exp_g_last_idx).load(
                count=RAW_F32_TMA_SWIZZLE_GROUP_ELEMS,
                alignment=RAW_F32_TMA_SWIZZLE_GROUP_BYTES,
            )
            half_reg_base = f32_group * RAW_F32_TMA_SWIZZLE_GROUP_ELEMS
            f32_reg_base = reg_base + half_reg_base
            exp_g_regs[f32_reg_base] = exp_g_vec[0]
            exp_g_regs[f32_reg_base + 1] = exp_g_vec[1]
            exp_g_regs[f32_reg_base + 2] = exp_g_vec[2]
            exp_g_regs[f32_reg_base + 3] = exp_g_vec[3]
            exp_neg_g_regs[half_reg_base] = cute.math.rcp(
                exp_g_vec[0], approx=True, ftz=True
            )
            exp_neg_g_regs[half_reg_base + 1] = cute.math.rcp(
                exp_g_vec[1], approx=True, ftz=True
            )
            exp_neg_g_regs[half_reg_base + 2] = cute.math.rcp(
                exp_g_vec[2], approx=True, ftz=True
            )
            exp_neg_g_regs[half_reg_base + 3] = cute.math.rcp(
                exp_g_vec[3], approx=True, ftz=True
            )
            exp_g_last_regs[f32_reg_base] = exp_g_last_vec[0]
            exp_g_last_regs[f32_reg_base + 1] = exp_g_last_vec[1]
            exp_g_last_regs[f32_reg_base + 2] = exp_g_last_vec[2]
            exp_g_last_regs[f32_reg_base + 3] = exp_g_last_vec[3]

        k_decay_vec_regs = cutlass.Array(
            input_dtype,
            RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        for pair_idx in cutlass.range_constexpr(RAW_F16_TMA_SWIZZLE_GROUP_ELEMS // 2):
            dim0 = pair_idx * 2
            dim1 = dim0 + 1
            raw_reg_idx0 = reg_base + dim0
            raw_reg_idx1 = reg_base + dim1
            k_value0, k_value1 = fmul2(
                (raw_k_regs[raw_reg_idx0], raw_k_regs[raw_reg_idx1]),
                (k_inv_norm, k_inv_norm),
            )
            k_decay0, k_decay1 = fmul2(
                (k_value0, k_value1),
                (exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1]),
            )
            k_inv0, k_inv1 = fmul2(
                (k_value0, k_value1),
                (exp_neg_g_regs[dim0], exp_neg_g_regs[dim1]),
            )
            k_inv_regs[reg_base + dim0] = k_inv0.to(input_dtype)
            k_inv_regs[reg_base + dim1] = k_inv1.to(input_dtype)
            k_restore0, k_restore1 = fmul2(
                (k_inv0, k_inv1),
                (
                    exp_g_last_regs[reg_base + dim0],
                    exp_g_last_regs[reg_base + dim1],
                ),
            )
            k_restore_all_regs[reg_base + dim0] = k_restore0.to(input_dtype)
            k_restore_all_regs[reg_base + dim1] = k_restore1.to(input_dtype)
            k_decay_vec_regs[dim0] = k_decay0.to(input_dtype)
            k_decay_vec_regs[dim1] = k_decay1.to(input_dtype)

        k_inv_vec = cutlass.Vector.from_elements(
            (
                k_inv_regs[reg_base],
                k_inv_regs[reg_base + 1],
                k_inv_regs[reg_base + 2],
                k_inv_regs[reg_base + 3],
                k_inv_regs[reg_base + 4],
                k_inv_regs[reg_base + 5],
                k_inv_regs[reg_base + 6],
                k_inv_regs[reg_base + 7],
            ),
            input_dtype,
        )
        k_decay_vec = cutlass.Vector.from_elements(
            (
                k_decay_vec_regs[0],
                k_decay_vec_regs[1],
                k_decay_vec_regs[2],
                k_decay_vec_regs[3],
                k_decay_vec_regs[4],
                k_decay_vec_regs[5],
                k_decay_vec_regs[6],
                k_decay_vec_regs[7],
            ),
            input_dtype,
        )
        if cutlass.const_expr(dim_half == 0):
            operand_smem_consumed_phase = ((chunk // DECAY_STAGE_COUNT) + 1) % 2
            operand_smem_consumed_wait(
                operand_smem_consumed_stage_mbar,
                operand_smem_consumed_phase,
            )
        k_inv_swizzled_idx = k_inv_s128_smem_index(decay_row, dim_base)
        (k_inv_ptr + k_inv_swizzled_idx).store(
            k_inv_vec,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        decay_storage_dim_base = tcgen05_decay_b_key_storage_dim_runtime(
            decay_row,
            dim_base,
        )
        decay_linear_idx_base = decay_row * DK + decay_storage_dim_base
        decay_swizzled_idx_base = tcgen05_swizzle_128b_elem_index(
            decay_linear_idx_base,
            TCGEN05_F16_ELEM_BYTES,
            BT,
        )
        (tcgen05_k_decay_ptr + decay_swizzled_idx_base).store(
            k_decay_vec,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        if cutlass.const_expr(dim_half == 0):
            # Publish the first half-DK of k_inv/k_decay early: warp 12's
            # KK K-blocks 0..3 only read key dims [0, 64).
            cg0_k_ready_arrive(cg0_k_half_ready_stage_mbar)
    cg0_k_ready_arrive(cg0_k_ready_stage_mbar)

    for dim_half in cutlass.range_constexpr(2):
        dim_base = (
            dim_half * (DK // 2) + lane_in_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        )
        reg_base = dim_half * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        q_decay_vec_regs = cutlass.Array(
            input_dtype,
            RAW_F16_TMA_SWIZZLE_GROUP_ELEMS,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )
        for pair_idx in cutlass.range_constexpr(RAW_F16_TMA_SWIZZLE_GROUP_ELEMS // 2):
            dim0 = pair_idx * 2
            dim1 = dim0 + 1
            raw_reg_idx0 = reg_base + dim0
            raw_reg_idx1 = reg_base + dim1
            q_value0, q_value1 = fmul2(
                (raw_q_regs[raw_reg_idx0], raw_q_regs[raw_reg_idx1]),
                (q_inv_norm, q_inv_norm),
            )
            q_decay0, q_decay1 = fmul2(
                (q_value0, q_value1),
                (exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1]),
            )
            q_decay_vec_regs[dim0] = q_decay0.to(input_dtype)
            q_decay_vec_regs[dim1] = q_decay1.to(input_dtype)

        q_decay_vec = cutlass.Vector.from_elements(
            (
                q_decay_vec_regs[0],
                q_decay_vec_regs[1],
                q_decay_vec_regs[2],
                q_decay_vec_regs[3],
                q_decay_vec_regs[4],
                q_decay_vec_regs[5],
                q_decay_vec_regs[6],
                q_decay_vec_regs[7],
            ),
            input_dtype,
        )
        decay_storage_dim_base = tcgen05_decay_b_key_storage_dim_runtime(
            decay_row,
            dim_base,
        )
        decay_linear_idx_base = decay_row * DK + decay_storage_dim_base
        decay_swizzled_idx_base = tcgen05_swizzle_128b_elem_index(
            decay_linear_idx_base,
            TCGEN05_F16_ELEM_BYTES,
            BT,
        )
        (tcgen05_q_decay_ptr + decay_swizzled_idx_base).store(
            q_decay_vec,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )

    k_restore_consumed_phase = ((chunk // DECAY_STAGE_COUNT) + 1) % 2
    operand_smem_consumed_wait(
        k_restore_consumed_stage_mbar,
        k_restore_consumed_phase,
    )

    for dim_half in cutlass.range_constexpr(2):
        dim_base = (
            dim_half * (DK // 2) + lane_in_row_group * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        )
        reg_base = dim_half * RAW_F16_TMA_SWIZZLE_GROUP_ELEMS
        storage_row = decay_row ^ TCGEN05_SW32_BT_HALF_XOR
        k_restore_idx = raw_f16_s128_smem_index(storage_row, dim_base)
        k_restore_vec = cutlass.Vector.from_elements(
            (
                k_restore_all_regs[reg_base],
                k_restore_all_regs[reg_base + 1],
                k_restore_all_regs[reg_base + 2],
                k_restore_all_regs[reg_base + 3],
                k_restore_all_regs[reg_base + 4],
                k_restore_all_regs[reg_base + 5],
                k_restore_all_regs[reg_base + 6],
                k_restore_all_regs[reg_base + 7],
            ),
            input_dtype,
        )
        (tcgen05_k_restore_ptr + k_restore_idx).store(
            k_restore_vec,
            alignment=RAW_F16_TMA_SWIZZLE_GROUP_BYTES,
        )


@cute.jit
def ptx_mma_m16n8k16_b16_f32(
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    c0,
    c1,
    c2,
    c3,
    input_dtype: cutlass.Constexpr,
):
    """Issue `mma.sync.aligned.m16n8k16.row.col.f32.{f16|bf16}.{f16|bf16}.f32`."""

    if cutlass.const_expr(
        input_dtype != cutlass.Float16 and input_dtype != cutlass.BFloat16
    ):
        raise TypeError(f"Invalid auxiliary-MMA input dtype: {input_dtype}")
    input_tag = "f16" if cutlass.const_expr(input_dtype == cutlass.Float16) else "bf16"

    return cute.arch.inline_ptx(
        f"mma.sync.aligned.m16n8k16.row.col.f32.{input_tag}.{input_tag}.f32"
        " {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        write_only_types=[
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
        ],
        read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
    )


@cute.jit
def super_mma_accumulator_row(lane, accum_idx: cutlass.Constexpr) -> cutlass.Int32:
    """Row coordinate for one `m16n8k16` accumulator element."""

    row = lane // 4
    if cutlass.const_expr(accum_idx >= 2):
        row = row + cutlass.Int32(8)
    return row


@cute.jit
def super_mma_accumulator_col(
    lane,
    n_block: cutlass.Constexpr,
    accum_idx: cutlass.Constexpr,
) -> cutlass.Int32:
    """Column coordinate for one `m16n8k16` accumulator element."""

    col = n_block * SUPER_MMA_ATOM_N + 2 * (lane % 4)
    if cutlass.const_expr((accum_idx % 2) == 1):
        col = col + cutlass.Int32(1)
    return col


@cute.jit
def super_mma_store_pairwise_tile_stmatrix_x4(
    pairwise_smem,
    dst_offset: cutlass.Constexpr,
    lane,
    n0_acc,
    n1_acc,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Store one 16x16 pairwise tile through `stmatrix.m8n8.x4.b16`."""

    prims.stmatrix(
        pairwise_stmatrix_m8n8x4_ptr(pairwise_smem, dst_offset, lane),
        [
            pack_input_b16x2_to_i32(n0_acc[0], n0_acc[1], input_dtype),
            pack_input_b16x2_to_i32(n0_acc[2], n0_acc[3], input_dtype),
            pack_input_b16x2_to_i32(n1_acc[0], n1_acc[1], input_dtype),
            pack_input_b16x2_to_i32(n1_acc[2], n1_acc[3], input_dtype),
        ],
        prims.MMALayout.ROW,
        shape=prims.StoreShape.M8N8,
    )


@cute.jit
def super_mma_strict_lower_beta_value(
    value: cutlass.Float32,
    row_coord,
    col_coord,
    beta_scale: cutlass.Float32,
) -> cutlass.Float32:
    """Apply `tril(x, -1) * beta[row]` to one pairwise accumulator value."""

    lower = value if row_coord > col_coord else cutlass.Float32(0.0)
    return lower * beta_scale


@cute.jit
def super_mma_build_l_fragment(
    raw_beta_smem,
    lane,
    n0_acc,
    n1_acc,
    l_frag,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Build the packed `L = beta * tril(KK, -1)` registers into `l_frag`."""

    row_lo = super_mma_accumulator_row(lane, 0)
    row_hi = super_mma_accumulator_row(lane, 2)
    n0_col0 = super_mma_accumulator_col(lane, 0, 0)
    n0_col1 = super_mma_accumulator_col(lane, 0, 1)
    n0_col2 = super_mma_accumulator_col(lane, 0, 2)
    n0_col3 = super_mma_accumulator_col(lane, 0, 3)
    n1_col0 = super_mma_accumulator_col(lane, 1, 0)
    n1_col1 = super_mma_accumulator_col(lane, 1, 1)
    n1_col2 = super_mma_accumulator_col(lane, 1, 2)
    n1_col3 = super_mma_accumulator_col(lane, 1, 3)
    beta_lo = raw_beta_smem[row_lo].to(cutlass.Float32)
    beta_hi = raw_beta_smem[row_hi].to(cutlass.Float32)

    l_frag[0] = pack_input_b16x2_to_i32(
        super_mma_strict_lower_beta_value(n0_acc[0], row_lo, n0_col0, beta_lo),
        super_mma_strict_lower_beta_value(n0_acc[1], row_lo, n0_col1, beta_lo),
        input_dtype,
    )
    l_frag[1] = pack_input_b16x2_to_i32(
        super_mma_strict_lower_beta_value(n0_acc[2], row_hi, n0_col2, beta_hi),
        super_mma_strict_lower_beta_value(n0_acc[3], row_hi, n0_col3, beta_hi),
        input_dtype,
    )
    l_frag[2] = pack_input_b16x2_to_i32(
        super_mma_strict_lower_beta_value(n1_acc[0], row_lo, n1_col0, beta_lo),
        super_mma_strict_lower_beta_value(n1_acc[1], row_lo, n1_col1, beta_lo),
        input_dtype,
    )
    l_frag[3] = pack_input_b16x2_to_i32(
        super_mma_strict_lower_beta_value(n1_acc[2], row_hi, n1_col2, beta_hi),
        super_mma_strict_lower_beta_value(n1_acc[3], row_hi, n1_col3, beta_hi),
        input_dtype,
    )


@cute.jit
def super_mma_sw128_decay_ldmatrix_index(
    row_coord,
    col_offset,
    k_block: cutlass.Constexpr,
):
    """Return the SW128 row-segment index used by the auxiliary-MMA warp.

    The decay tile is stored in the tcgen05 B-operand K-box-major layout with
    only the constant half-atom interleave; the tcgen05 descriptor reads the
    standard SW128 layout, so no row-dependent storage K-block xor exists (the
    former ``(row & 2) * K_ATOM`` term here matched a write-side xor that the
    tcgen05 state MMA never saw — both are removed together).  ldmatrix reads
    the 16B row segment at that physical K-block.
    """

    key_mask = cutlass.Int32(TCGEN05_F16_K_ATOM // 2)
    logical_key = k_block * TCGEN05_F16_K_ATOM + col_offset
    storage_key = logical_key ^ key_mask
    elems_per_128b = cutlass.Int32(TCGEN05_SW128_BYTES // TCGEN05_F16_ELEM_BYTES)
    storage_slice = storage_key // elems_per_128b
    key_in_slice = storage_key - storage_slice * elems_per_128b
    storage_phase = key_in_slice // cutlass.Int32(TCGEN05_F16_K_ATOM)
    storage_col_offset = key_in_slice - storage_phase * cutlass.Int32(
        TCGEN05_F16_K_ATOM
    )
    row_byte = row_coord * cutlass.Int32(TCGEN05_SW128_BYTES)
    phase_byte = storage_phase * cutlass.Int32(
        TCGEN05_F16_K_ATOM * TCGEN05_F16_ELEM_BYTES
    )
    col_byte = storage_col_offset * cutlass.Int32(TCGEN05_F16_ELEM_BYTES)
    byte_in_slice = row_byte + phase_byte + col_byte
    swizzle_mask = (row_coord & cutlass.Int32(7)) << 4
    elems_per_slice = cutlass.Int32(BT) * elems_per_128b
    return storage_slice * elems_per_slice + (
        (byte_in_slice ^ swizzle_mask) // cutlass.Int32(TCGEN05_F16_ELEM_BYTES)
    )


@cute.jit
def super_mma_load_decay_lhs_fragment(
    tcgen05_decay_smem,
    lane,
    k_block: cutlass.Constexpr,
):
    """Load the A operand fragment for one `m16n8k16` K phase.

    A is logically `[BT, DK]` and physically stored in the tcgen05 SW128 decay
    operand layout. The register order matches the direct-Q fragment order used
    by the CUTLASS primitives FMHA examples: rows 0/8 crossed with K cols 0/8.
    """

    lane_div8 = lane // 8
    lane_mod8 = lane % 8
    row_offset = cutlass.Int32(8) if (lane_div8 % 2) else cutlass.Int32(0)
    col_offset = cutlass.Int32(8) if (lane_div8 // 2) else cutlass.Int32(0)
    row_coord = lane_mod8 + row_offset
    swizzled_idx = super_mma_sw128_decay_ldmatrix_index(
        row_coord,
        col_offset,
        k_block,
    )
    ptr = tcgen05_decay_smem.subview(swizzled_idx)
    return prims.ldmatrix(ptr.data_ptr(), 4, prims.MMALayout.ROW)


@cute.jit
def super_mma_load_k_inv_rhs_fragment(
    k_inv_smem,
    lane,
    k_block: cutlass.Constexpr,
):
    """Load the B operand fragment for one `m16n8k16` K phase.

    B is stored token-major `[BT, DK]`, which is column-major for the logical
    `[DK, BT]` MMA RHS. The `ldmatrix.x4` return is split into two N fragments:
    registers 0/1 for columns 0..7, and registers 2/3 for columns 8..15.
    """

    lane_div8 = lane // 8
    lane_div16 = lane // 16
    lane_mod8 = lane % 8
    row_offset = cutlass.Int32(8) if lane_div16 else cutlass.Int32(0)
    col_offset = cutlass.Int32(8) if (lane_div8 % 2) else cutlass.Int32(0)
    row_coord = lane_mod8 + row_offset
    col_coord = k_block * SUPER_MMA_ATOM_K + col_offset
    ptr = k_inv_smem.subview(k_inv_s128_smem_index(row_coord, col_coord))
    return prims.ldmatrix(ptr.data_ptr(), 4, prims.MMALayout.ROW)


@cute.jit
def super_mma_qk_causal_value(
    value: cutlass.Float32,
    lane,
    n_block: cutlass.Constexpr,
    accum_idx: cutlass.Constexpr,
) -> cutlass.Float32:
    """Zero an accumulator outside the inclusive-lower QK tile."""

    row_coord = super_mma_accumulator_row(lane, accum_idx)
    col_coord = super_mma_accumulator_col(lane, n_block, accum_idx)
    return value if row_coord >= col_coord else cutlass.Float32(0.0)


@cute.jit
def super_mma_stage_kk_blocks(
    tcgen05_k_decay_smem,
    k_inv_smem,
    lane,
    input_dtype: cutlass.Constexpr,
    k_block_lo: cutlass.Constexpr,
    k_block_hi: cutlass.Constexpr,
    kk_n0_acc,
    kk_n1_acc,
):
    """Accumulate KK m16n8k16 K-blocks [k_block_lo, k_block_hi).

    The KK product is split at the half-DK boundary so warp 12 can run
    K-blocks 0..3 on the `cg0_k_half_ready` arrival (CG0's dim_half==0
    stores cover key dims [0, 64): k_inv segment 0 and, because the decay
    storage-key xor mask is 8 or 40, k_decay SW128 slice 0) while CG0 is
    still staging the second half.
    """

    for k_block_off in cutlass.range_constexpr(k_block_hi - k_block_lo):
        k_block = k_block_lo + k_block_off
        rhs_vec = super_mma_load_k_inv_rhs_fragment(
            k_inv_smem,
            lane,
            k_block,
        )
        kk_lhs_vec = super_mma_load_decay_lhs_fragment(
            tcgen05_k_decay_smem,
            lane,
            k_block,
        )

        kk_n0_d0, kk_n0_d1, kk_n0_d2, kk_n0_d3 = ptx_mma_m16n8k16_b16_f32(
            kk_lhs_vec[0],
            kk_lhs_vec[1],
            kk_lhs_vec[2],
            kk_lhs_vec[3],
            rhs_vec[0],
            rhs_vec[1],
            kk_n0_acc[0],
            kk_n0_acc[1],
            kk_n0_acc[2],
            kk_n0_acc[3],
            input_dtype,
        )
        kk_n0_acc[0] = kk_n0_d0
        kk_n0_acc[1] = kk_n0_d1
        kk_n0_acc[2] = kk_n0_d2
        kk_n0_acc[3] = kk_n0_d3
        kk_n1_d0, kk_n1_d1, kk_n1_d2, kk_n1_d3 = ptx_mma_m16n8k16_b16_f32(
            kk_lhs_vec[0],
            kk_lhs_vec[1],
            kk_lhs_vec[2],
            kk_lhs_vec[3],
            rhs_vec[2],
            rhs_vec[3],
            kk_n1_acc[0],
            kk_n1_acc[1],
            kk_n1_acc[2],
            kk_n1_acc[3],
            input_dtype,
        )
        kk_n1_acc[0] = kk_n1_d0
        kk_n1_acc[1] = kk_n1_d1
        kk_n1_acc[2] = kk_n1_d2
        kk_n1_acc[3] = kk_n1_d3


@cute.jit
def super_mma_stage_qk(
    tcgen05_q_decay_smem,
    k_inv_smem,
    pairwise_smem,
    lane,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Produce the causal QK tile consumed by the qkv tcgen05 MMA."""

    qk_n0_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    qk_n1_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    for accum_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        qk_n0_acc[accum_idx] = cutlass.Float32(0.0)
        qk_n1_acc[accum_idx] = cutlass.Float32(0.0)

    for k_block in cutlass.range_constexpr(SUPER_MMA_K_BLOCKS):
        rhs_vec = super_mma_load_k_inv_rhs_fragment(
            k_inv_smem,
            lane,
            k_block,
        )
        qk_lhs_vec = super_mma_load_decay_lhs_fragment(
            tcgen05_q_decay_smem,
            lane,
            k_block,
        )

        qk_n0_d0, qk_n0_d1, qk_n0_d2, qk_n0_d3 = ptx_mma_m16n8k16_b16_f32(
            qk_lhs_vec[0],
            qk_lhs_vec[1],
            qk_lhs_vec[2],
            qk_lhs_vec[3],
            rhs_vec[0],
            rhs_vec[1],
            qk_n0_acc[0],
            qk_n0_acc[1],
            qk_n0_acc[2],
            qk_n0_acc[3],
            input_dtype,
        )
        qk_n0_acc[0] = qk_n0_d0
        qk_n0_acc[1] = qk_n0_d1
        qk_n0_acc[2] = qk_n0_d2
        qk_n0_acc[3] = qk_n0_d3
        qk_n1_d0, qk_n1_d1, qk_n1_d2, qk_n1_d3 = ptx_mma_m16n8k16_b16_f32(
            qk_lhs_vec[0],
            qk_lhs_vec[1],
            qk_lhs_vec[2],
            qk_lhs_vec[3],
            rhs_vec[2],
            rhs_vec[3],
            qk_n1_acc[0],
            qk_n1_acc[1],
            qk_n1_acc[2],
            qk_n1_acc[3],
            input_dtype,
        )
        qk_n1_acc[0] = qk_n1_d0
        qk_n1_acc[1] = qk_n1_d1
        qk_n1_acc[2] = qk_n1_d2
        qk_n1_acc[3] = qk_n1_d3

    qk_n0_acc[0] = super_mma_qk_causal_value(qk_n0_acc[0], lane, 0, 0)
    qk_n0_acc[1] = super_mma_qk_causal_value(qk_n0_acc[1], lane, 0, 1)
    qk_n0_acc[2] = super_mma_qk_causal_value(qk_n0_acc[2], lane, 0, 2)
    qk_n0_acc[3] = super_mma_qk_causal_value(qk_n0_acc[3], lane, 0, 3)
    qk_n1_acc[0] = super_mma_qk_causal_value(qk_n1_acc[0], lane, 1, 0)
    qk_n1_acc[1] = super_mma_qk_causal_value(qk_n1_acc[1], lane, 1, 1)
    qk_n1_acc[2] = super_mma_qk_causal_value(qk_n1_acc[2], lane, 1, 2)
    qk_n1_acc[3] = super_mma_qk_causal_value(qk_n1_acc[3], lane, 1, 3)

    super_mma_store_pairwise_tile_stmatrix_x4(
        pairwise_smem,
        PAIRWISE_SMEM_QK_OFFSET,
        lane,
        qk_n0_acc,
        qk_n1_acc,
        input_dtype,
    )


@cute.jit
def super_mma_pairwise_product_from_ab_regs(
    lhs_frag,
    rhs_frag,
    n0_out,
    n1_out,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Compute `lhs @ rhs` accumulators from packed A-layout fragments.

    Results land in the caller-provided `n0_out`/`n1_out` accumulator arrays.
    """

    rhs_b0 = movmatrix_b16(rhs_frag[0])
    rhs_b1 = movmatrix_b16(rhs_frag[1])
    rhs_b2 = movmatrix_b16(rhs_frag[2])
    rhs_b3 = movmatrix_b16(rhs_frag[3])
    zero_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    for accum_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        zero_acc[accum_idx] = cutlass.Float32(0.0)
    n0_out[0], n0_out[1], n0_out[2], n0_out[3] = ptx_mma_m16n8k16_b16_f32(
        lhs_frag[0],
        lhs_frag[1],
        lhs_frag[2],
        lhs_frag[3],
        rhs_b0,
        rhs_b1,
        zero_acc[0],
        zero_acc[1],
        zero_acc[2],
        zero_acc[3],
        input_dtype,
    )
    n1_out[0], n1_out[1], n1_out[2], n1_out[3] = ptx_mma_m16n8k16_b16_f32(
        lhs_frag[0],
        lhs_frag[1],
        lhs_frag[2],
        lhs_frag[3],
        rhs_b2,
        rhs_b3,
        zero_acc[0],
        zero_acc[1],
        zero_acc[2],
        zero_acc[3],
        input_dtype,
    )


@cute.jit
def super_mma_initial_inverse_from_fragment(
    l_values,
    lane,
    n_block: cutlass.Constexpr,
    accum_idx: cutlass.Constexpr,
) -> cutlass.Float32:
    """Return one accumulator element of `I - L` from packed L registers."""

    row_coord = super_mma_accumulator_row(lane, accum_idx)
    col_coord = super_mma_accumulator_col(lane, n_block, accum_idx)
    l_value_idx: cutlass.Constexpr[int] = (
        n_block * SUPER_MMA_ACCUMULATORS_PER_LANE + accum_idx
    )
    return pairwise_eye(row_coord, col_coord) - l_values[l_value_idx]


@cute.jit
def super_mma_pack_pairwise_accumulator(
    n0_acc,
    n1_acc,
    out_frag,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Round one 16x16 pairwise accumulator tile into an A-layout b16 fragment.

    The packed registers land in the caller-provided `out_frag` array.
    """

    out_frag[0] = pack_input_b16x2_to_i32(n0_acc[0], n0_acc[1], input_dtype)
    out_frag[1] = pack_input_b16x2_to_i32(n0_acc[2], n0_acc[3], input_dtype)
    out_frag[2] = pack_input_b16x2_to_i32(n1_acc[0], n1_acc[1], input_dtype)
    out_frag[3] = pack_input_b16x2_to_i32(n1_acc[2], n1_acc[3], input_dtype)


@cute.jit
def super_mma_square_pairwise_fragment(
    src_frag,
    out_frag,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Pack b16 registers for `src @ src` into `out_frag` without SMEM roundtrip."""

    n0_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    n1_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_pairwise_product_from_ab_regs(
        src_frag,
        src_frag,
        n0_acc,
        n1_acc,
        input_dtype,
    )
    super_mma_pack_pairwise_accumulator(
        n0_acc,
        n1_acc,
        out_frag,
        input_dtype,
    )


@cute.jit
def super_mma_update_inverse_with_power_regs(
    rhs_frag,
    n0_acc,
    n1_acc,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Update `inv += inv @ Lpow` in place, keeping `inv` in registers."""

    a_frag = cutlass.Array(
        cutlass.Int32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_pack_pairwise_accumulator(
        n0_acc,
        n1_acc,
        a_frag,
        input_dtype,
    )
    p0_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    p1_acc = cutlass.Array(
        cutlass.Float32,
        SUPER_MMA_ACCUMULATORS_PER_LANE,
        alignment=16,
    )
    super_mma_pairwise_product_from_ab_regs(
        a_frag,
        rhs_frag,
        p0_acc,
        p1_acc,
        input_dtype,
    )

    for accum_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        n0_acc[accum_idx] = (
            mma_input_dtype(n0_acc[accum_idx], input_dtype) + p0_acc[accum_idx]
        )
    for accum_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        n1_acc[accum_idx] = (
            mma_input_dtype(n1_acc[accum_idx], input_dtype) + p1_acc[accum_idx]
        )


@cute.jit
def super_mma_stage_blockwise_inverse(
    pairwise_smem, lane, l_frag, input_dtype: cutlass.Constexpr
) -> None:
    """Block diagonal sparsity lowers this inverse to six warp MMA instructions."""
    l_vec = cutlass.Vector.from_elements(
        (l_frag[0], l_frag[1], l_frag[2], l_frag[3]), cutlass.Int32
    )
    l_values = l_vec.bitcast(input_dtype).to(cutlass.Float32)
    diagonal = [cutlass.Float32(0.0) for _ in range(4)]
    diagonal_slots = (0, 1, 6, 7)
    for index in cutlass.range_constexpr(4):
        slot = diagonal_slots[index]
        row, col = acc_coord(lane, slot)
        eye = cutlass.Float32(0.0)
        if row == col:
            eye = cutlass.Float32(1.0)
        diagonal[index] = eye - f16_round(l_values[slot])

    d0 = pack_f16x2(l_values[0], l_values[1])
    d3 = pack_f16x2(l_values[6], l_values[7])
    d2 = mma_blockdiag_8x8_f16(d0, d3, d0, d3)
    d2_0 = pack_f16x2(d2[0], d2[1])
    d2_3 = pack_f16x2(d2[2], d2[3])

    diagonal_0 = pack_f16x2(diagonal[0], diagonal[1])
    diagonal_3 = pack_f16x2(diagonal[2], diagonal[3])
    product = mma_blockdiag_8x8_f16(diagonal_0, diagonal_3, d2_0, d2_3)
    for index in cutlass.range_constexpr(4):
        diagonal[index] = f16_round(diagonal[index]) + product[index]

    d4 = mma_blockdiag_8x8_f16(d2_0, d2_3, d2_0, d2_3)
    d4_0 = pack_f16x2(d4[0], d4[1])
    d4_3 = pack_f16x2(d4[2], d4[3])
    diagonal_0 = pack_f16x2(diagonal[0], diagonal[1])
    diagonal_3 = pack_f16x2(diagonal[2], diagonal[3])
    product = mma_blockdiag_8x8_f16(diagonal_0, diagonal_3, d4_0, d4_3)
    for index in cutlass.range_constexpr(4):
        diagonal[index] = f16_round(diagonal[index]) + product[index]

    # T1 = Binv @ A21 and X21 = -T1 @ Binv each have only the
    # lower-left output quadrant live, so each needs one N=8 MMA.
    zero_i32 = cutlass.Int32(0)
    zero_f32 = cutlass.Float32(0.0)
    binv_0 = pack_f16x2(diagonal[0], diagonal[1])
    binv_3 = pack_f16x2(diagonal[2], diagonal[3])
    a21_1 = pack_f16x2(l_values[2], l_values[3])
    t1 = mma_m16n8k16_f16(
        binv_0,
        zero_i32,
        zero_i32,
        binv_3,
        zero_i32,
        movmatrix_b16(a21_1),
        zero_f32,
        zero_f32,
        zero_f32,
        zero_f32,
    )
    correction_1 = pack_f16x2(t1[2], t1[3])
    correction = mma_m16n8k16_f16(
        zero_i32,
        correction_1,
        zero_i32,
        zero_i32,
        movmatrix_b16(binv_0),
        zero_i32,
        zero_f32,
        zero_f32,
        zero_f32,
        zero_f32,
    )

    inverse = (
        diagonal[0],
        diagonal[1],
        -correction[2],
        -correction[3],
        zero_f32,
        zero_f32,
        diagonal[2],
        diagonal[3],
    )
    n0_acc = cutlass.Array(cutlass.Float32, 4, alignment=16)
    n1_acc = cutlass.Array(cutlass.Float32, 4, alignment=16)
    for i in cutlass.range_constexpr(4):
        n0_acc[i] = inverse[i]
        n1_acc[i] = inverse[i + 4]
    super_mma_store_pairwise_tile_stmatrix_x4(
        pairwise_smem, PAIRWISE_SMEM_AINV_OFFSET, lane, n0_acc, n1_acc, input_dtype
    )


@cute.jit
def super_mma_stage_pairwise_pipeline(
    tcgen05_k_decay_smem,
    k_inv_smem,
    pairwise_smem,
    raw_beta_smem,
    cg0_k_ready_stage_mbar,
    cg0_k_ready_phase,
    lane,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Run the KK/L/inverse sequence inside the auxiliary-MMA warp.

    The caller has already waited `cg0_k_half_ready` (first half-DK of
    k_inv/k_decay staged); the full `cg0_k_ready` wait happens here, between
    the two KK half-products.
    """

    kk_n0_acc = cutlass.Array(
        cutlass.Float32, SUPER_MMA_ACCUMULATORS_PER_LANE, alignment=16
    )
    kk_n1_acc = cutlass.Array(
        cutlass.Float32, SUPER_MMA_ACCUMULATORS_PER_LANE, alignment=16
    )
    for acc_idx in cutlass.range_constexpr(SUPER_MMA_ACCUMULATORS_PER_LANE):
        kk_n0_acc[acc_idx] = cutlass.Float32(0.0)
        kk_n1_acc[acc_idx] = cutlass.Float32(0.0)
    super_mma_stage_kk_blocks(
        tcgen05_k_decay_smem,
        k_inv_smem,
        lane,
        input_dtype,
        0,
        SUPER_MMA_K_BLOCKS // 2,
        kk_n0_acc,
        kk_n1_acc,
    )
    cg0_k_ready_wait(
        cg0_k_ready_stage_mbar,
        cg0_k_ready_phase,
    )
    super_mma_stage_kk_blocks(
        tcgen05_k_decay_smem,
        k_inv_smem,
        lane,
        input_dtype,
        SUPER_MMA_K_BLOCKS // 2,
        SUPER_MMA_K_BLOCKS,
        kk_n0_acc,
        kk_n1_acc,
    )
    l_frag = cutlass.Array(cutlass.Int32, SUPER_MMA_ACCUMULATORS_PER_LANE, alignment=16)
    super_mma_build_l_fragment(
        raw_beta_smem,
        lane,
        kk_n0_acc,
        kk_n1_acc,
        l_frag,
        input_dtype,
    )
    super_mma_stage_blockwise_inverse(
        pairwise_smem,
        lane,
        l_frag,
        input_dtype,
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
    """Return the runtime TMEM column offset for one shared_acc stage."""

    return KDA_TMEM_SHARED_ACC_COL_OFFSET + shared_acc_stage * KDA_TMEM_N16_ACC_COLS


@cute.jit
def tcgen05_shared_acc_stage_from_event(shared_acc_event_id):
    """Map one shared-acc event to its ring-buffer stage."""

    return shared_acc_event_id % KDA_TMEM_SHARED_ACC_STAGE_COUNT


@cute.jit
def tcgen05_shared_acc_phase_from_event(shared_acc_event_id):
    """Map one shared-acc event to the selected stage's barrier phase."""

    return (shared_acc_event_id // KDA_TMEM_SHARED_ACC_STAGE_COUNT) % 2


@cute.jit
def tcgen05_shared_input_tmem_col_offset(shared_input_stage):
    """Return the runtime TMEM column offset for one shared input stage."""

    return (
        KDA_TMEM_SHARED_INPUT_COL_OFFSET
        + shared_input_stage * KDA_TMEM_SHARED_INPUT_COLS
    )


@cute.jit
def tcgen05_store_initial_state_tmem(
    tmem_raw_addr,
    initial_state: cute.Tensor | None,
    state_ckpt: cute.Tensor | None,
    ckpt_slot,
    state_slot,
    bidy,
    dv_half,
    warp_idx,
    lane,
    HALF: cutlass.Constexpr,
) -> None:
    """Initialize recurrent TMEM from the optional external VK state.

    HALF=True is the DV2 chain form (Layout F): this warp's quadrant lanes
    0..15 hold state rows dv_half*64 + 16q .. +15; the alignment-16 lanes are
    zero-filled junk (kept written so the first pack reads defined data).
    """

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    row_id = base_row_id + tmem_sp * THREADS_PER_WARP
    if cutlass.const_expr(HALF):
        value_dim = (
            dv_half * cutlass.Int32(DV_HALF)
            + tmem_sp * ROWS_PER_WARP
            + (lane % ROWS_PER_WARP)
        )
        valid_lane = lane < ROWS_PER_WARP
    else:
        value_dim = tmem_sp * THREADS_PER_WARP + lane
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
            if cutlass.const_expr(initial_state is not None):
                if cutlass.const_expr(HALF):
                    # pyrefly: ignore [unbound-name]
                    if valid_lane:
                        # pyrefly: ignore [missing-attribute, unsupported-operation]
                        state_value = initial_state[
                            state_slot, bidy, value_dim, key_dim
                        ].to(cutlass.Float32)
                else:
                    # pyrefly: ignore [missing-attribute, unsupported-operation]
                    state_value = initial_state[
                        state_slot, bidy, value_dim, key_dim
                    ].to(cutlass.Float32)
            state_block[col] = state_value
            if cutlass.const_expr(state_ckpt is not None):
                if (ckpt_slot >= cutlass.Int32(0)) & (
                    # pyrefly: ignore [missing-attribute]
                    ckpt_slot < cutlass.Int32(state_ckpt.shape[0])
                ):
                    if cutlass.const_expr(HALF):
                        # pyrefly: ignore [unbound-name]
                        if valid_lane:
                            # pyrefly: ignore [unsupported-operation]
                            state_ckpt[ckpt_slot, bidy, value_dim, key_dim] = (
                                # pyrefly: ignore [missing-attribute]
                                state_value.to(state_ckpt.element_type)
                            )
                    else:
                        # pyrefly: ignore [unsupported-operation]
                        state_ckpt[ckpt_slot, bidy, value_dim, key_dim] = (
                            # pyrefly: ignore [missing-attribute]
                            state_value.to(state_ckpt.element_type)
                        )

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
def pack_state_input_x16(state_block, input_dtype: cutlass.Constexpr):
    """Pack one 16-column FP32 state fragment into an 8-column A fragment."""

    packed_state = cutlass.Array(
        cutlass.Int32,
        TCGEN05_STATE_INPUT_PACKED_COLS,
        alignment=16,
    )
    for packed_col in cutlass.range_constexpr(TCGEN05_STATE_INPUT_PACKED_COLS):
        source_pair = packed_col ^ TCGEN05_F16_A_TMEM_PAIR_XOR
        key_dim0 = source_pair * 2
        key_dim1 = key_dim0 + 1
        packed_state[packed_col] = pack_input_b16x2_to_i32(
            state_block[key_dim0],
            state_block[key_dim1],
            input_dtype,
        )
    return packed_state


@cute.jit
def tcgen05_stage_state_input_tmem(
    tmem_raw_addr,
    warp_idx,
    output_consumed_mbar,
    output_consumed_phase,
    input_dtype: cutlass.Constexpr,
    WAIT_OUTPUT_CONSUMED: cutlass.Constexpr,
    HALF: cutlass.Constexpr,
    DEFER_STORE_WAIT: cutlass.Constexpr,
):
    """Pack one 64-column half of the recurrent state as a tcgen05 A operand.

    ``HALF`` selects state columns ``[HALF*64, HALF*64+64)``. Splitting the
    pack lets the left half start as soon as the left half of the previous
    chunk's final-state delta MMA has committed, overlapping with the right
    delta half. Steady-state chunks overlap the prior output-stage reuse
    wait with the asynchronous state loads.
    """

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    row_addr = (base_row_id + tmem_sp * THREADS_PER_WARP) << 16
    state_col_id = (
        base_col_id
        + KDA_TMEM_STATE_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    )
    state_ptr0 = cutlass.inttoptr(row_addr | state_col_id, 6, cutlass.Float32)
    state_ptr1 = cutlass.inttoptr(
        row_addr | (state_col_id + TCGEN05_STATE_INPUT_LOAD_COLS),
        6,
        cutlass.Float32,
    )
    state_ptr2 = cutlass.inttoptr(
        row_addr | (state_col_id + 2 * TCGEN05_STATE_INPUT_LOAD_COLS),
        6,
        cutlass.Float32,
    )
    state_ptr3 = cutlass.inttoptr(
        row_addr | (state_col_id + 3 * TCGEN05_STATE_INPUT_LOAD_COLS),
        6,
        cutlass.Float32,
    )

    state0 = prims.tcgen05_ld("32x32b", state_ptr0, num=TCGEN05_STATE_INPUT_LOAD_COLS)
    state1 = prims.tcgen05_ld("32x32b", state_ptr1, num=TCGEN05_STATE_INPUT_LOAD_COLS)
    state2 = prims.tcgen05_ld("32x32b", state_ptr2, num=TCGEN05_STATE_INPUT_LOAD_COLS)
    state3 = prims.tcgen05_ld("32x32b", state_ptr3, num=TCGEN05_STATE_INPUT_LOAD_COLS)
    if cutlass.const_expr(WAIT_OUTPUT_CONSUMED):
        output_consumed_wait(
            output_consumed_mbar,
            output_consumed_phase,
        )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    packed_col_id = (
        base_col_id
        + KDA_TMEM_STATE_AS_INPUT_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_PACKED_COLS
    )
    packed_row_addr = base_row_id << 16
    packed_ptr0 = prims.make_tmem_ptr(packed_row_addr | packed_col_id, cutlass.Int8)
    packed_ptr1 = prims.make_tmem_ptr(
        packed_row_addr + packed_col_id + TCGEN05_STATE_INPUT_PACKED_COLS,
        cutlass.Int8,
    )
    packed_ptr2 = prims.make_tmem_ptr(
        packed_row_addr + packed_col_id + 2 * TCGEN05_STATE_INPUT_PACKED_COLS,
        cutlass.Int8,
    )
    packed_ptr3 = prims.make_tmem_ptr(
        packed_row_addr + packed_col_id + 3 * TCGEN05_STATE_INPUT_PACKED_COLS,
        cutlass.Int8,
    )
    packed0 = pack_state_input_x16(state0, input_dtype)
    prims.tcgen05_st("32x32b", packed_ptr0, packed0[0:TCGEN05_STATE_INPUT_PACKED_COLS])
    packed1 = pack_state_input_x16(state1, input_dtype)
    prims.tcgen05_st("32x32b", packed_ptr1, packed1[0:TCGEN05_STATE_INPUT_PACKED_COLS])
    packed2 = pack_state_input_x16(state2, input_dtype)
    prims.tcgen05_st("32x32b", packed_ptr2, packed2[0:TCGEN05_STATE_INPUT_PACKED_COLS])
    packed3 = pack_state_input_x16(state3, input_dtype)
    prims.tcgen05_st("32x32b", packed_ptr3, packed3[0:TCGEN05_STATE_INPUT_PACKED_COLS])

    if cutlass.const_expr(not DEFER_STORE_WAIT):
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    return state0, state1, state2, state3


@cute.jit
def tcgen05_scale_state_x16_regs(
    state_block,
    state_scale_f32_ptr,
    key_block_start: cutlass.Constexpr,
    EXCHANGE_LAYOUT: cutlass.Constexpr,
):
    """Build one true-FP32 scaled state fragment without issuing its store."""

    scaled_state = cutlass.Array(
        cutlass.Float32,
        TCGEN05_STATE_INPUT_LOAD_COLS,
        alignment=16,
    )
    for vec_idx in cutlass.range_constexpr(TCGEN05_STATE_INPUT_LOAD_COLS // 4):
        reg_base = vec_idx * 4
        scale_dim: cutlass.Constexpr = key_block_start + reg_base
        scale_idx = cutlass.Int32(scale_dim)
        if cutlass.const_expr(EXCHANGE_LAYOUT):
            scale_idx = raw_f32_exchange_smem_index(BT - 1, scale_dim)
        scale = (state_scale_f32_ptr + scale_idx).load(count=4, alignment=16)
        scaled_state[reg_base], scaled_state[reg_base + 1] = fmul2(
            (state_block[reg_base], state_block[reg_base + 1]),
            (scale[0], scale[1]),
        )
        scaled_state[reg_base + 2], scaled_state[reg_base + 3] = fmul2(
            (state_block[reg_base + 2], state_block[reg_base + 3]),
            (scale[2], scale[3]),
        )
    return scaled_state


@cute.jit
def tcgen05_rescale_state_x32(
    state_block,
    state_block_ptr,
    state_scale_f32_ptr,
    key_block_start: cutlass.Constexpr,
    EXCHANGE_LAYOUT: cutlass.Constexpr,
) -> None:
    """Apply one 32-element FP32 decay fragment and store the state block."""

    block_cols: cutlass.Constexpr = 2 * TCGEN05_STATE_INPUT_LOAD_COLS
    scaled_state = cutlass.Array(cutlass.Float32, block_cols, alignment=16)
    for vec_idx in cutlass.range_constexpr(block_cols // 4):
        reg_base = vec_idx * 4
        scale_dim: cutlass.Constexpr = key_block_start + reg_base
        scale_idx = cutlass.Int32(scale_dim)
        if cutlass.const_expr(EXCHANGE_LAYOUT):
            scale_idx = raw_f32_exchange_smem_index(BT - 1, scale_dim)
        scale = (state_scale_f32_ptr + scale_idx).load(count=4, alignment=16)
        scaled_state[reg_base], scaled_state[reg_base + 1] = fmul2(
            (state_block[reg_base], state_block[reg_base + 1]),
            (scale[0], scale[1]),
        )
        scaled_state[reg_base + 2], scaled_state[reg_base + 3] = fmul2(
            (state_block[reg_base + 2], state_block[reg_base + 3]),
            (scale[2], scale[3]),
        )
    prims.tcgen05_st("32x32b", state_block_ptr, scaled_state[0:block_cols])


@cute.jit
def tcgen05_publish_projection_then_rescale_state_regs(
    tmem_raw_addr,
    state_scale_f32_smem,
    state_input_ready_mbar,
    warp_idx,
    state0,
    state1,
    state2,
    state3,
    HALF: cutlass.Constexpr,
    EXCHANGE_LAYOUT: cutlass.Constexpr,
) -> None:
    """Hide a projection-input store behind true-FP32 state scaling.

    The packed projection columns and live recurrent-state columns do not
    alias.  Build the scaled fragments while the preceding packed stores are
    outstanding, then publish those packed columns before issuing the scaled
    live-state stores.  The final wait/fence remains a separate update join.
    """

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS
    row_addr = (base_row_id + tmem_sp * THREADS_PER_WARP) << 16
    state_col_id = (
        base_col_id
        + KDA_TMEM_STATE_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    )
    state_ptr0 = cutlass.inttoptr(row_addr | state_col_id, 6, cutlass.Float32)
    state_ptr1 = cutlass.inttoptr(
        row_addr | (state_col_id + TCGEN05_STATE_INPUT_LOAD_COLS),
        6,
        cutlass.Float32,
    )
    state_ptr2 = cutlass.inttoptr(
        row_addr | (state_col_id + 2 * TCGEN05_STATE_INPUT_LOAD_COLS),
        6,
        cutlass.Float32,
    )
    state_ptr3 = cutlass.inttoptr(
        row_addr | (state_col_id + 3 * TCGEN05_STATE_INPUT_LOAD_COLS),
        6,
        cutlass.Float32,
    )
    state_scale_f32_ptr = state_scale_f32_smem.data_ptr()
    key_half_start: cutlass.Constexpr = HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    scaled0 = tcgen05_scale_state_x16_regs(
        state0,
        state_scale_f32_ptr,
        key_half_start,
        EXCHANGE_LAYOUT,
    )
    scaled1 = tcgen05_scale_state_x16_regs(
        state1,
        state_scale_f32_ptr,
        key_half_start + TCGEN05_STATE_INPUT_LOAD_COLS,
        EXCHANGE_LAYOUT,
    )
    scaled2 = tcgen05_scale_state_x16_regs(
        state2,
        state_scale_f32_ptr,
        key_half_start + 2 * TCGEN05_STATE_INPUT_LOAD_COLS,
        EXCHANGE_LAYOUT,
    )
    scaled3 = tcgen05_scale_state_x16_regs(
        state3,
        state_scale_f32_ptr,
        key_half_start + 3 * TCGEN05_STATE_INPUT_LOAD_COLS,
        EXCHANGE_LAYOUT,
    )

    # This wait retires only the already-issued packed projection stores.
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    state_input_ready_arrive(state_input_ready_mbar)

    prims.tcgen05_st(
        "32x32b",
        state_ptr0,
        scaled0[0:TCGEN05_STATE_INPUT_LOAD_COLS],
    )
    prims.tcgen05_st(
        "32x32b",
        state_ptr1,
        scaled1[0:TCGEN05_STATE_INPUT_LOAD_COLS],
    )
    prims.tcgen05_st(
        "32x32b",
        state_ptr2,
        scaled2[0:TCGEN05_STATE_INPUT_LOAD_COLS],
    )
    prims.tcgen05_st(
        "32x32b",
        state_ptr3,
        scaled3[0:TCGEN05_STATE_INPUT_LOAD_COLS],
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_pack_rescale_state_half_tmem(
    tmem_raw_addr,
    state_scale_f32_smem,
    state_input_ready_mbar,
    warp_idx,
    input_dtype: cutlass.Constexpr,
    HALF: cutlass.Constexpr,
    EXCHANGE_LAYOUT: cutlass.Constexpr,
) -> None:
    """Pack and FP32-decay one state half from a single pair of TMEM loads.

    The owning CG0 group publishes the packed left projection operand and the
    decayed live state together.  This removes both CG1's duplicate left-half
    load/pack and CG0's former scale-only reload.
    """

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS
    row_addr = (base_row_id + tmem_sp * THREADS_PER_WARP) << 16
    block_cols: cutlass.Constexpr = 2 * TCGEN05_STATE_INPUT_LOAD_COLS
    state_col_id = (
        base_col_id
        + KDA_TMEM_STATE_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    )
    state_ptr0 = cutlass.inttoptr(row_addr | state_col_id, 6, cutlass.Float32)
    state_ptr1 = cutlass.inttoptr(
        row_addr | (state_col_id + block_cols), 6, cutlass.Float32
    )

    packed_col_id = (
        base_col_id
        + KDA_TMEM_STATE_AS_INPUT_COL_OFFSET
        + HALF * 4 * TCGEN05_STATE_INPUT_PACKED_COLS
    )
    packed_row_addr = base_row_id << 16
    packed_ptr0 = prims.make_tmem_ptr(packed_row_addr | packed_col_id, cutlass.Int8)
    packed_ptr2 = prims.make_tmem_ptr(
        packed_row_addr | (packed_col_id + 2 * TCGEN05_STATE_INPUT_PACKED_COLS),
        cutlass.Int8,
    )

    state0 = prims.tcgen05_ld("32x32b", state_ptr0, num=block_cols)
    state1 = prims.tcgen05_ld("32x32b", state_ptr1, num=block_cols)
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    packed0 = pack_state_input_x16(state0[0:TCGEN05_STATE_INPUT_LOAD_COLS], input_dtype)
    packed1 = pack_state_input_x16(
        state0[TCGEN05_STATE_INPUT_LOAD_COLS:block_cols], input_dtype
    )
    # Adjacent packed fragments share one contiguous TMEM store. Preserve
    # each fragment's operand permutation while halving store issue count.
    combined0 = cutlass.Array(
        cutlass.Int32, 2 * TCGEN05_STATE_INPUT_PACKED_COLS, alignment=16
    )
    for i in cutlass.range_constexpr(TCGEN05_STATE_INPUT_PACKED_COLS):
        combined0[i] = packed0[i]
        combined0[i + TCGEN05_STATE_INPUT_PACKED_COLS] = packed1[i]
    prims.tcgen05_st(
        "32x32b", packed_ptr0, combined0[0 : 2 * TCGEN05_STATE_INPUT_PACKED_COLS]
    )
    packed2 = pack_state_input_x16(state1[0:TCGEN05_STATE_INPUT_LOAD_COLS], input_dtype)
    packed3 = pack_state_input_x16(
        state1[TCGEN05_STATE_INPUT_LOAD_COLS:block_cols], input_dtype
    )
    combined1 = cutlass.Array(
        cutlass.Int32, 2 * TCGEN05_STATE_INPUT_PACKED_COLS, alignment=16
    )
    for i in cutlass.range_constexpr(TCGEN05_STATE_INPUT_PACKED_COLS):
        combined1[i] = packed2[i]
        combined1[i + TCGEN05_STATE_INPUT_PACKED_COLS] = packed3[i]
    prims.tcgen05_st(
        "32x32b", packed_ptr2, combined1[0 : 2 * TCGEN05_STATE_INPUT_PACKED_COLS]
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    state_input_ready_arrive(state_input_ready_mbar)

    state_scale_f32_ptr = state_scale_f32_smem.data_ptr()
    key_half_start: cutlass.Constexpr = HALF * 4 * TCGEN05_STATE_INPUT_LOAD_COLS
    tcgen05_rescale_state_x32(
        state0,
        state_ptr0,
        state_scale_f32_ptr,
        key_half_start,
        EXCHANGE_LAYOUT,
    )
    tcgen05_rescale_state_x32(
        state1,
        state_ptr1,
        state_scale_f32_ptr,
        key_half_start + block_cols,
        EXCHANGE_LAYOUT,
    )
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

    pipeline_primitives.issue_tmem_smem_mma_slices(
        tcgen05_decay_smem,
        tmem_raw_addr,
        acc_ready_mbar,
        tmem_col_offset,
        KDA_TMEM_STATE_AS_INPUT_COL_OFFSET,
        input_dtype,
        BT,
        M_DIM,
        TCGEN05_F16_K_ATOM,
        TCGEN05_F16_ELEM_BYTES,
        K_BLOCK_BEGIN,
        K_BLOCK_END,
        TCGEN05_STATE_K_B_LEADING_BYTES,
        TCGEN05_STATE_K_B_STRIDE_BYTES,
        TCGEN05_SW128_K_PHASES_PER_SLICE,
        TCGEN05_SW128_BYTES,
        BT,
        INITIAL_SCALE_D,
        COMMIT,
    )


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
def tcgen05_issue_state_q_mma(
    tcgen05_q_decay_smem,
    tmem_raw_addr,
    operand_smem_consumed_mbar,
    qstate_acc_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Issue state*q and release q_decay when the tensor core consumes it."""

    tcgen05_issue_state_projection_mma(
        tcgen05_q_decay_smem,
        tmem_raw_addr,
        operand_smem_consumed_mbar,
        tcgen05_qstate_acc_tmem_col_offset(qstate_acc_stage),
        input_dtype,
        0,
        DK // TCGEN05_F16_K_ATOM,
        False,
        True,
        DV,
    )


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
def tcgen05_stage_rhs_input_tmem(
    tmem_raw_addr,
    raw_v_smem,
    raw_beta_smem,
    warp_idx,
    lane,
    shared_acc_stage,
    shared_input_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Produce RHS = beta * (v - state*k) into shared_input TMEM by 16-row tiles."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    projection_col_id = base_col_id + tcgen05_shared_acc_tmem_col_offset(
        shared_acc_stage
    )
    input_col_id = base_col_id + tcgen05_shared_input_tmem_col_offset(
        shared_input_stage
    )
    value_dim_base = tmem_sp * THREADS_PER_WARP

    row_id0 = base_row_id + value_dim_base
    block_addr0 = (row_id0 << 16) | projection_col_id
    block_ptr0 = cutlass.inttoptr(block_addr0, 6, cutlass.Float32)
    state_k0 = prims.tcgen05_ld(
        "16x256b",
        block_ptr0,
        num=2,
    )

    row_id1 = row_id0 + 16
    block_addr1 = (row_id1 << 16) | projection_col_id
    block_ptr1 = cutlass.inttoptr(block_addr1, 6, cutlass.Float32)
    state_k1 = prims.tcgen05_ld(
        "16x256b",
        block_ptr1,
        num=2,
    )

    raw_v_regs0 = prims.ldmatrix(
        raw_v_ldmatrix_trans_ptr(
            raw_v_smem,
            value_dim_base,
            lane,
        ),
        4,
        prims.MMALayout.COL,
    )
    raw_v_regs1 = prims.ldmatrix(
        raw_v_ldmatrix_trans_ptr(
            raw_v_smem,
            value_dim_base + 16,
            lane,
        ),
        4,
        prims.MMALayout.COL,
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    packed_rhs0 = cutlass.Array(
        cutlass.Int32,
        4,
        space=cutlass.AddressSpace.rmem,
    )
    for reg_idx in cutlass.range_constexpr(4):
        packed_col = (reg_idx // 2) * 4 + (lane & 3)
        source_pair = packed_col ^ TCGEN05_F16_A_TMEM_PAIR_XOR
        token0 = source_pair * 2
        token1 = token0 + 1
        beta0 = raw_beta_smem[token0].to(cutlass.Float32)
        beta1 = raw_beta_smem[token1].to(cutlass.Float32)
        raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
        state_k_val0, state_k_val1 = tcgen05_rhs_token_pair_from_16x256b_fragment(
            state_k0,
            reg_idx,
        )
        beta_pair = pack_input_b16x2_to_i32(beta0, beta1, input_dtype)
        state_k_pair = pack_input_b16x2_to_i32(
            state_k_val0,
            state_k_val1,
            input_dtype,
        )
        diff_pair = sub_b16x2_input_dtype(
            raw_v_regs0[raw_matrix],
            state_k_pair,
            input_dtype,
        )
        packed_rhs0[reg_idx] = mul_b16x2_input_dtype(
            beta_pair,
            diff_pair,
            input_dtype,
        )

    packed_rhs1 = cutlass.Array(
        cutlass.Int32,
        4,
        space=cutlass.AddressSpace.rmem,
    )
    for reg_idx in cutlass.range_constexpr(4):
        packed_col = (reg_idx // 2) * 4 + (lane & 3)
        source_pair = packed_col ^ TCGEN05_F16_A_TMEM_PAIR_XOR
        token0 = source_pair * 2
        token1 = token0 + 1
        beta0 = raw_beta_smem[token0].to(cutlass.Float32)
        beta1 = raw_beta_smem[token1].to(cutlass.Float32)
        raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
        state_k_val0, state_k_val1 = tcgen05_rhs_token_pair_from_16x256b_fragment(
            state_k1,
            reg_idx,
        )
        beta_pair = pack_input_b16x2_to_i32(beta0, beta1, input_dtype)
        state_k_pair = pack_input_b16x2_to_i32(
            state_k_val0,
            state_k_val1,
            input_dtype,
        )
        diff_pair = sub_b16x2_input_dtype(
            raw_v_regs1[raw_matrix],
            state_k_pair,
            input_dtype,
        )
        packed_rhs1[reg_idx] = mul_b16x2_input_dtype(
            beta_pair,
            diff_pair,
            input_dtype,
        )

    input_block_addr0 = (base_row_id << 16) | input_col_id
    input_block_ptr0 = prims.make_tmem_ptr(input_block_addr0, cutlass.Int8)
    prims.tcgen05_st("16x128b", input_block_ptr0, packed_rhs0[0:4])

    input_block_addr1 = ((base_row_id + 16) << 16) | input_col_id
    input_block_ptr1 = prims.make_tmem_ptr(input_block_addr1, cutlass.Int8)
    prims.tcgen05_st("16x128b", input_block_ptr1, packed_rhs1[0:4])

    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_stage_update_input_tmem(
    tmem_raw_addr,
    warp_idx,
    shared_acc_stage,
    shared_input_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Move update from shared_acc TMEM into packed shared_input TMEM."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    row_id = base_row_id + tmem_sp * THREADS_PER_WARP
    projection_col_id = base_col_id + tcgen05_shared_acc_tmem_col_offset(
        shared_acc_stage
    )
    block_addr = (row_id << 16) | projection_col_id
    block_ptr = cutlass.inttoptr(block_addr, 6, cutlass.Float32)
    update = prims.tcgen05_ld(
        "32x32b",
        block_ptr,
        num=TCGEN05_TMEM_LOAD_COLS,
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    packed_update = cutlass.Array(
        cutlass.Int32,
        KDA_TMEM_SHARED_INPUT_COLS,
        alignment=16,
    )
    for packed_col in cutlass.range_constexpr(KDA_TMEM_SHARED_INPUT_COLS):
        source_pair = packed_col ^ TCGEN05_F16_A_TMEM_PAIR_XOR
        token0 = source_pair * 2
        token1 = token0 + 1
        packed_update[packed_col] = pack_input_b16x2_to_i32(
            update[token0],
            update[token1],
            input_dtype,
        )

    col_id = base_col_id + tcgen05_shared_input_tmem_col_offset(shared_input_stage)
    input_block_addr = (base_row_id << 16) | col_id
    input_block_ptr = prims.make_tmem_ptr(input_block_addr, cutlass.Int8)
    prims.tcgen05_st(
        "32x32b",
        input_block_ptr,
        packed_update[0:KDA_TMEM_SHARED_INPUT_COLS],
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_issue_value_pairwise_mma(
    pairwise_stage_smem,
    pairwise_tile_offset: cutlass.Constexpr,
    tmem_raw_addr,
    acc_ready_mbar,
    shared_input_stage,
    tmem_col_offset,
    scale_d: cutlass.Constexpr,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Issue a `[DV,BT] @ [BT,BT]` value-side tcgen05 MMA.

    The A operand is a staged value tile (`rhs` or `update`) in the scheduled
    shared_input TMEM slot.  The B operand is a pairwise tile staged as
    `[N=token_i, K=token_j]` for tcgen05.
    """

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
        m_dim=DV,
        b_major=0,
    )
    lhs_tmem = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Int8).subview(
        tcgen05_shared_input_tmem_col_offset(shared_input_stage)
    )
    desc_pairwise = prims.Tcgen05SmemDesc.build(
        pairwise_stage_smem.subview(pairwise_tile_offset),
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
            scale_d,
        )
        prims.tcgen05_commit(acc_ready_mbar, group=prims.CTAGroup.CTA_1)


@cute.jit
def tcgen05_issue_update_mma(
    pairwise_stage_smem,
    tmem_raw_addr,
    acc_ready_mbar,
    shared_acc_stage,
    shared_input_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Issue update = A_inv @ rhs into a scheduled shared_acc TMEM stage."""

    tcgen05_issue_value_pairwise_mma(
        pairwise_stage_smem,
        PAIRWISE_SMEM_AINV_OFFSET,
        tmem_raw_addr,
        acc_ready_mbar,
        shared_input_stage,
        tcgen05_shared_acc_tmem_col_offset(shared_acc_stage),
        False,
        input_dtype,
    )


@cute.jit
def tcgen05_issue_qkv_mma(
    pairwise_stage_smem,
    tmem_raw_addr,
    acc_ready_mbar,
    shared_input_stage,
    qstate_acc_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Accumulate qkv = qk @ update into the live qstate_acc TMEM slot."""

    tcgen05_issue_value_pairwise_mma(
        pairwise_stage_smem,
        PAIRWISE_SMEM_QK_OFFSET,
        tmem_raw_addr,
        acc_ready_mbar,
        shared_input_stage,
        tcgen05_qstate_acc_tmem_col_offset(qstate_acc_stage),
        True,
        input_dtype,
    )


@cute.jit
def tcgen05_load_qstate_output_tmem(
    tmem_raw_addr,
    o_smem,
    warp_idx,
    lane,
    o_stage_base,
    qstate_acc_stage,
    scale: cutlass.Float32,
    output_dtype: cutlass.Constexpr,
) -> None:
    """Drain `state_q + qkv` from qstate_acc TMEM with STSM.T output staging."""

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    projection_col_id = base_col_id + tcgen05_qstate_acc_tmem_col_offset(
        qstate_acc_stage
    )
    value_dim_base = tmem_sp * THREADS_PER_WARP

    row_id0 = base_row_id + value_dim_base
    row_id1 = row_id0 + 16
    block_addr0 = (row_id0 << 16) | projection_col_id
    block_addr1 = (row_id1 << 16) | projection_col_id
    block_ptr0 = cutlass.inttoptr(block_addr0, 6, cutlass.Float32)
    block_ptr1 = cutlass.inttoptr(block_addr1, 6, cutlass.Float32)
    loaded0 = prims.tcgen05_ld(
        "16x256b",
        block_ptr0,
        num=2,
    )
    loaded1 = prims.tcgen05_ld(
        "16x256b",
        block_ptr1,
        num=2,
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)

    stsm_regs0 = cutlass.Array(
        cutlass.Int32,
        4,
        space=cutlass.AddressSpace.rmem,
    )
    stsm_regs1 = cutlass.Array(
        cutlass.Int32,
        4,
        space=cutlass.AddressSpace.rmem,
    )
    for reg_idx in cutlass.range_constexpr(4):
        scaled0_0, scaled0_1 = fmul2(
            (loaded0[2 * reg_idx], loaded0[2 * reg_idx + 1]),
            (scale, scale),
        )
        scaled1_0, scaled1_1 = fmul2(
            (loaded1[2 * reg_idx], loaded1[2 * reg_idx + 1]),
            (scale, scale),
        )
        stsm_regs0[reg_idx] = pack_output_b16x2_to_i32(
            scaled0_0,
            scaled0_1,
            output_dtype,
        )
        stsm_regs1[reg_idx] = pack_output_b16x2_to_i32(
            scaled1_0,
            scaled1_1,
            output_dtype,
        )

    smem_dst0 = o_smem_stmatrix_128b_ptr(
        o_smem,
        o_stage_base,
        value_dim_base,
        lane,
    )
    smem_dst1 = o_smem_stmatrix_128b_ptr(
        o_smem,
        o_stage_base,
        value_dim_base + 16,
        lane,
    )
    prims.stmatrix(
        smem_dst0,
        stsm_regs0.data_ptr().load(count=4, alignment=4),
        prims.MMALayout.COL,
        shape=prims.StoreShape.M8N8,
    )
    prims.stmatrix(
        smem_dst1,
        stsm_regs1.data_ptr().load(count=4, alignment=4),
        prims.MMALayout.COL,
        shape=prims.StoreShape.M8N8,
    )
    cute.arch.fence_view_async_shared()


@cute.jit
def epilogue_stage_store(
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    o_smem,
    sequence_start,
    head_idx,
    chunk_start,
    o_stage_base,
) -> None:
    """Store the staged `[BT, DV]` output tile to global memory with TMA."""

    global_chunk_start = sequence_start + chunk_start
    if prims.elect_sync():
        for value_segment in cutlass.range_constexpr(O_TMA_SEGMENTS):
            segment_base = (
                o_stage_base + O_OUT_OFFSET + value_segment * BT * O_TMA_SWIZZLE_ELEMS
            )
            o_coord = (
                cutlass.Int32(value_segment * O_TMA_SWIZZLE_ELEMS),
                global_chunk_start,
                head_idx,
                cutlass.Int32(0),
            )
            prims.cp_async_bulk_tensor_global_shared_cta(
                tma_desc_o.get_ptr(),
                o_smem.subview(segment_base),
                o_coord,
            )
        prims.cp_async_bulk_commit_group()
        prims.cp_async_bulk_wait_group(0, read=True)
    prims.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def epilogue_tail_store(
    out,
    o_smem,
    sequence_start,
    head_idx,
    chunk_start,
    seqlen,
    o_stage_base,
    lane,
) -> None:
    """Store a partial packed-sequence tail without crossing its boundary."""

    valid_tokens = seqlen - chunk_start
    for elem_iter in cutlass.range_constexpr((BT * DV) // THREADS_PER_WARP):
        linear_idx = elem_iter * THREADS_PER_WARP + lane
        token_coord = linear_idx // DV
        value_dim = linear_idx - token_coord * DV
        if token_coord < valid_tokens:
            smem_idx = o_smem_swizzle_128b_elem_index(
                o_stage_base,
                value_dim,
                token_coord,
            )
            out[0, sequence_start + chunk_start + token_coord, head_idx, value_dim] = (
                o_smem[smem_idx]
            )
    prims.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def epilogue_wait_and_store_full_output(
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    o_smem,
    output_ready_mbar,
    output_consumed_mbar,
    sequence_start,
    head_idx,
    output_chunk,
    O_STAGES: cutlass.Constexpr,
):
    """Drain one full staged output chunk from SMEM with TMA."""

    output_chunk_start = output_chunk * BT
    o_stage = output_chunk % O_STAGES
    o_stage_base = o_stage * O_SMEM_STAGE_SIZE
    output_ready_wait(
        output_ready_mbar.subview(o_stage),
        (output_chunk // O_STAGES) % 2,
    )
    epilogue_stage_store(
        tma_desc_o,
        o_smem,
        sequence_start,
        head_idx,
        output_chunk_start,
        o_stage_base,
    )
    output_consumed_arrive(output_consumed_mbar.subview(o_stage))


@cute.jit
def epilogue_wait_and_store_final_output(
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    out,
    o_smem,
    output_ready_mbar,
    output_consumed_mbar,
    sequence_start,
    head_idx,
    seqlen,
    output_chunk,
    lane,
    O_STAGES: cutlass.Constexpr,
):
    """Drain the final output chunk, guarding a partial packed tail."""

    if seqlen % BT == 0:
        epilogue_wait_and_store_full_output(
            tma_desc_o,
            o_smem,
            output_ready_mbar,
            output_consumed_mbar,
            sequence_start,
            head_idx,
            output_chunk,
            O_STAGES,
        )
    else:
        output_chunk_start = output_chunk * BT
        o_stage = output_chunk % O_STAGES
        o_stage_base = o_stage * O_SMEM_STAGE_SIZE
        output_ready_wait(
            output_ready_mbar.subview(o_stage),
            (output_chunk // O_STAGES) % 2,
        )
        epilogue_tail_store(
            out,
            o_smem,
            sequence_start,
            head_idx,
            output_chunk_start,
            seqlen,
            o_stage_base,
            lane,
        )
        output_consumed_arrive(output_consumed_mbar.subview(o_stage))


@cute.jit
def tcgen05_issue_final_state_delta_mma(
    tcgen05_k_restore_smem,
    tmem_raw_addr,
    k_restore_consumed_l_mbar,
    k_restore_consumed_mbar,
    shared_input_stage,
    input_dtype: cutlass.Constexpr,
) -> None:
    """Accumulate final_state += update @ k_restore in two N=64 halves.

    The left half commits to its own mbarrier so the next chunk's left
    state pack can begin while the right half is still in flight; the
    right commit (which orders after every prior MMA) keeps the original
    full-completion contract for CG0 and the right pack.
    """

    half_n = DK // 2
    tmem_ptr_l = cutlass.inttoptr(
        tmem_raw_addr + KDA_TMEM_FINAL_STATE_ACC_COL_OFFSET,
        6,
        cutlass.Float32,
    )
    tmem_ptr_r = cutlass.inttoptr(
        tmem_raw_addr + KDA_TMEM_FINAL_STATE_ACC_COL_OFFSET + half_n,
        6,
        cutlass.Float32,
    )
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=input_dtype,
        b_dtype=input_dtype,
        n_dim=half_n,
        m_dim=DV,
        b_major=1,
    )
    update_tmem = prims.make_tmem_ptr(tmem_raw_addr, cutlass.Int8).subview(
        tcgen05_shared_input_tmem_col_offset(shared_input_stage)
    )
    desc_k_restore = prims.Tcgen05SmemDesc.build(
        tcgen05_k_restore_smem.subview(0),
        leading_byte_offset=TCGEN05_FINAL_STATE_B_LEADING_BYTES,
        stride_byte_offset=TCGEN05_FINAL_STATE_B_STRIDE_BYTES,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )

    if prims.elect_sync():
        prims.tcgen05_mma(
            prims.Tcgen05MMAKind.F16,
            prims.CTAGroup.CTA_1,
            tmem_ptr_l,
            update_tmem,
            desc_k_restore,
            idesc,
            True,
        )
        prims.tcgen05_commit(
            k_restore_consumed_l_mbar,
            group=prims.CTAGroup.CTA_1,
        )
        prims.tcgen05_mma(
            prims.Tcgen05MMAKind.F16,
            prims.CTAGroup.CTA_1,
            tmem_ptr_r,
            update_tmem,
            desc_k_restore.advance_start_address(TCGEN05_FINAL_STATE_B_LEADING_BYTES),
            idesc,
            True,
        )
        prims.tcgen05_commit(
            k_restore_consumed_mbar,
            group=prims.CTAGroup.CTA_1,
        )


@cute.jit
def tcgen05_store_final_state_tmem(
    tmem_raw_addr,
    state_col_offset,
    final_state: cute.Tensor,
    bidx,
    bidy,
    dv_half,
    warp_idx,
    lane,
    HALF: cutlass.Constexpr,
) -> None:
    """Store the live recurrent TMEM state to the VK final_state tensor.

    HALF=True is the DV2 chain form: each M=64 half stores its own rows
    (Layout F -- only the quadrant lanes 0..15 carry state).
    """

    base_col_id = tmem_raw_addr & 0xFFFF
    base_row_id = tmem_raw_addr >> 16
    tmem_sp = warp_idx % TCGEN05_STATE_K_TMEM_ROW_BLOCKS

    row_id = base_row_id + tmem_sp * THREADS_PER_WARP
    if cutlass.const_expr(HALF):
        value_dim = (
            dv_half * cutlass.Int32(DV_HALF)
            + tmem_sp * ROWS_PER_WARP
            + (lane % ROWS_PER_WARP)
        )
        valid_lane = lane < ROWS_PER_WARP
    else:
        value_dim = tmem_sp * THREADS_PER_WARP + lane
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
            if cutlass.const_expr(HALF):
                # pyrefly: ignore [unbound-name]
                if valid_lane:
                    final_state[bidx, bidy, value_dim, key_dim] = loaded[col].to(
                        final_state.element_type
                    )
            else:
                final_state[bidx, bidy, value_dim, key_dim] = loaded[col].to(
                    final_state.element_type
                )


@cute.jit
def length_ordered_sequence(cu_seqlens, sequence_slot, tidx):
    """Stable longest-first task selection with one candidate per active lane.

    This performs runtime ordering inside the CTA, so no extra launch or
    global workspace is required. Equal lengths retain source order. Threads
    stride over candidates when the sequence count exceeds the CTA width.
    """
    selected = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )
    sequences = cutlass.Int32(cu_seqlens.shape[0] - 1)
    for candidate in cutlass.range(tidx, sequences, 512):
        length = cutlass.Int32(cu_seqlens[candidate + 1]) - cutlass.Int32(
            cu_seqlens[candidate]
        )
        rank = cutlass.Int32(0)
        for other in cutlass.range(sequences):
            other_length = cutlass.Int32(cu_seqlens[other + 1]) - cutlass.Int32(
                cu_seqlens[other]
            )
            if (other_length > length) | (
                (other_length == length) & (other < candidate)
            ):
                rank += cutlass.Int32(1)
        if rank == sequence_slot:
            selected[0] = candidate
    cute.arch.barrier()
    return selected[0]


@cute.kernel
def kernel(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_gate: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_beta: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    seq_order: cute.Tensor | None,
    state_indices: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    SCALE: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    checkpoint_stride_chunks: cutlass.Int32,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Float32,
    gate_dtype: cutlass.Constexpr,
    TASK_ORDER: cutlass.Constexpr = "identity",
    SEGMENT_BEGIN: cutlass.Constexpr = 0,
    SEGMENT_SIZE: cutlass.Constexpr = -1,
    # Python literal defaults are converted by CuTe inside the JIT boundary.
    # pyrefly: ignore [bad-function-definition]
    SEQUENCE_BEGIN: cutlass.Int32 = 0,
    GROUPED: cutlass.Constexpr = False,
) -> None:
    """BT=16 KDA forward kernel.

    Grid: `(heads, num_sequences, 1)`. Each CTA owns one packed sequence/head
    and iterates over `ceil(sequence_length / 16)` chunks in order.
    """

    tidx, _, _ = cute.arch.thread_idx()
    # Put heads in grid-x so every sequence's head CTAs are contiguous in the
    # linear launch order. sequence_slot follows the automatic longest-first
    # permutation; bidx/bidy remain the original sequence/head indices below.
    bidy, sequence_slot, _ = cute.arch.block_idx()
    if cutlass.const_expr(GROUPED):
        sequence_slot += SEQUENCE_BEGIN
    bidx = sequence_slot
    if cutlass.const_expr(seq_order is not None):
        # pyrefly: ignore [unsupported-operation]
        bidx = cutlass.Int32(seq_order[sequence_slot])
    elif cutlass.const_expr(TASK_ORDER == "longest_first"):
        bidx = length_ordered_sequence(cu_seqlens, sequence_slot, tidx)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % THREADS_PER_WARP

    sequence_start = cutlass.Int32(cu_seqlens[bidx])
    # beta TMA transport shape: heads from grid-x; g = 8/gcd(heads, 8) via the
    # lowest set bit (g==1 -> 8-head-group box, g>1 -> pair-packed rows).
    heads32 = cutlass.Int32(cute.arch.grid_dim()[0])
    beta_lsb = heads32 & (-heads32)
    beta_g = cutlass.Int32(8) // cutlass.min(beta_lsb, cutlass.Int32(8))
    sequence_end = cutlass.Int32(cu_seqlens[bidx + 1])
    # Clamp the offset before adding it to the sequence base. Widen constants
    # first so even a rounded prefix offset beyond Int32 cannot wrap.
    if cutlass.const_expr(SEGMENT_BEGIN > 0):
        sequence_start += cutlass.Int32(
            cutlass.min(
                cutlass.Int64(SEGMENT_BEGIN),
                cutlass.Int64(sequence_end - sequence_start),
            )
        )
    if cutlass.const_expr(SEGMENT_SIZE >= 0):
        sequence_end = sequence_start + cutlass.Int32(
            cutlass.min(
                cutlass.Int64(SEGMENT_SIZE),
                cutlass.Int64(sequence_end - sequence_start),
            )
        )
    seqlen = sequence_end - sequence_start
    num_chunks = cute.ceil_div(seqlen, BT)
    input_dtype = q.element_type
    if cutlass.const_expr(
        k.element_type != input_dtype or v.element_type != input_dtype
    ):
        raise TypeError(
            "KDA CUTLASS primitives kernel expects q/k/v to use the same 16-bit dtype"
        )
    if cutlass.const_expr(raw_gate.element_type != gate_dtype):
        raise TypeError(
            "KDA CUTLASS primitives kernel expects raw_gate to match gate_dtype"
        )
    if cutlass.const_expr(beta.element_type != cutlass.BFloat16):
        raise TypeError("KDA CUTLASS primitives kernel expects beta logits to use BF16")
    if cutlass.const_expr(
        cu_seqlens.element_type not in (cutlass.Int32, cutlass.Int64)
    ):
        raise TypeError("fused prefill expects int32 or int64 sequence offsets")
    if cutlass.const_expr(initial_state is not None):
        if cutlass.const_expr(
            # pyrefly: ignore [missing-attribute]
            initial_state.element_type not in (cutlass.BFloat16, cutlass.Float32)
        ):
            raise TypeError(
                "KDA CUTLASS primitives kernel input state dtype must be BF16 or FP32"
            )
    if cutlass.const_expr(final_state is not None):
        if cutlass.const_expr(
            # pyrefly: ignore [missing-attribute]
            final_state.element_type not in (cutlass.BFloat16, cutlass.Float32)
        ):
            raise TypeError(
                "KDA CUTLASS primitives kernel output state dtype must be BF16 or FP32"
            )
    if cutlass.const_expr(initial_state is not None and final_state is not None):
        # pyrefly: ignore [missing-attribute]
        if cutlass.const_expr(initial_state.element_type != final_state.element_type):
            raise TypeError(
                "KDA CUTLASS primitives kernel expects matching state input/output dtypes"
            )
    # Buffers are declaration-ordered and intentionally non-aliased.
    tma_mbar = cutlass.Array(
        cutlass.Int64,
        TMA_MBAR_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    qstate_acc_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    output_ready_mbar = cutlass.Array(
        cutlass.Int64,
        O_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    q_k_restore_ready_mbar = cutlass.Array(
        cutlass.Int64,
        Q_K_RESTORE_READY_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    cg0_k_ready_mbar = cutlass.Array(
        cutlass.Int64,
        DECAY_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    cg0_k_half_ready_mbar = cutlass.Array(
        cutlass.Int64,
        DECAY_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    diag_ready_mbar = cutlass.Array(
        cutlass.Int64,
        RAW_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    raw_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        RAW_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    state_input_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    state_input_ready_l_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    initial_state_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    checkpoint_read_done_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    operand_smem_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        DECAY_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    rhs_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    update_ready_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    output_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        O_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    final_state_stored_mbar = cutlass.Array(
        cutlass.Int64,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    pairwise_ready_mbar = cutlass.Array(
        cutlass.Int64,
        PAIRWISE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    qk_ready_mbar = cutlass.Array(
        cutlass.Int64,
        PAIRWISE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    pairwise_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        PAIRWISE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    k_restore_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        DECAY_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    k_restore_consumed_l_mbar = cutlass.Array(
        cutlass.Int64,
        DECAY_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    shared_acc_ready_mbar = cutlass.Array(
        cutlass.Int64,
        KDA_TMEM_SHARED_ACC_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    # The hand-written K-box-major SW128 mapping is normalized to phase 0,
    # so both tcgen05 and ldmatrix can share 1KB-aligned operand buffers.
    tmem_ptr_i32 = cutlass.Array(
        cutlass.Int32,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=4,
    )
    tcgen05_k_decay_smem = cutlass.Array(
        input_dtype,
        TCGEN05_K_DECAY_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    tcgen05_q_decay_smem = cutlass.Array(
        input_dtype,
        TCGEN05_Q_DECAY_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    tcgen05_k_restore_smem = cutlass.Array(
        input_dtype,
        TCGEN05_K_RESTORE_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    pairwise_smem = cutlass.Array(
        input_dtype,
        PAIRWISE_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    raw_q_smem = cutlass.Array(
        q.element_type,
        RAW_Q_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    raw_k_smem = cutlass.Array(
        k.element_type,
        RAW_K_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    raw_v_smem = cutlass.Array(
        v.element_type,
        RAW_V_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    # Gate SMEM.  Sizes above are ELEMENT counts, so the ring's byte size
    # follows gate_dtype (FP32: 8 x 8 KB = 64 KB; BF16: 8 x 4 KB = 32 KB).
    # Under FP32 the exp2(g_prefix) exchange keeps aliasing this ring exactly
    # as it always did and NOTHING extra is allocated; a 16-bit gate cannot
    # host an FP32 exchange, so it gets a dedicated 4-deep FP32 ring (16 KB).
    if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
        raw_gate_smem = cutlass.Array(
            cutlass.Float32,
            RAW_GATE_SMEM_TILE_SIZE,
            space=cutlass.AddressSpace.smem,
            alignment=RAW_F32_TMA_SWIZZLE_ALIGNMENT_BYTES,
        )
        gate_exchange_smem = None
    else:
        raw_gate_smem = cutlass.Array(
            gate_dtype,
            RAW_GATE_SMEM_TILE_SIZE,
            space=cutlass.AddressSpace.smem,
            alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
        )
        gate_exchange_smem = cutlass.Array(
            cutlass.Float32,
            GATE_EXCHANGE_SMEM_TILE_SIZE,
            space=cutlass.AddressSpace.smem,
            alignment=RAW_F32_TMA_SWIZZLE_ALIGNMENT_BYTES,
        )
    k_inv_smem = cutlass.Array(
        input_dtype,
        K_INV_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    o_smem = cutlass.Array(
        out.element_type,
        O_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        # The scalar CG1 store computes W128 offsets relative to this buffer.
        # Align to the full s128b period so absolute SMEM address bits do not
        # add a hidden phase to the TMA store-side swizzle.
        alignment=O_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    raw_dt_bias_smem = cutlass.Array(
        cutlass.Float32,
        RAW_DT_BIAS_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    raw_beta_smem = cutlass.Array(
        cutlass.Float32,
        RAW_BETA_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    beta_tile_smem = cutlass.Array(
        cutlass.BFloat16,
        BETA_TILE_STAGE_COUNT * BETA_TILE_STAGE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    beta_tile_mbar = cutlass.Array(
        cutlass.Int64,
        BETA_TILE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    # Actual bytes one beta tile transfers (the ring stage is padded to a
    # fixed stride; expect_tx must be the real box size).
    beta_tile_tx = cutlass.Int32(8 * BT * 2)
    if beta_g > cutlass.Int32(1):
        beta_tile_tx = heads32 * (cutlass.Int32(BT) + beta_g) * cutlass.Int32(2)
    tma_tx_bytes = cutlass.const_expr(
        DK * BT * q.element_type.width // 8
        + DK * BT * k.element_type.width // 8
        + DV * BT * v.element_type.width // 8
        + DK * BT * raw_gate.element_type.width // 8
    )

    if warp_idx == ROLES.tma_load:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(TMA_MBAR_STAGE_COUNT):
                prims.mbarrier_init(tma_mbar.subview(stage), 1)
            for stage in cutlass.range_constexpr(RAW_STAGE_COUNT):
                # Four owner-CG0 warps consume q/k/gate, four CG1 warps
                # consume v/beta, and the standalone auxiliary-MMA warp consumes
                # beta while constructing the pairwise tile.
                prims.mbarrier_init(raw_consumed_mbar.subview(stage), 9)
            for stage in cutlass.range_constexpr(BETA_TILE_STAGE_COUNT):
                prims.mbarrier_init(beta_tile_mbar.subview(stage), 1)
    elif warp_idx == ROLES.tcgen05_mma:
        if prims.elect_sync():
            prims.mbarrier_init(qstate_acc_ready_mbar, 1)
            for stage in cutlass.range_constexpr(KDA_TMEM_SHARED_ACC_STAGE_COUNT):
                prims.mbarrier_init(shared_acc_ready_mbar.subview(stage), 1)
            prims.mbarrier_init(state_input_ready_mbar, 4)
            prims.mbarrier_init(state_input_ready_l_mbar, 4)
            prims.mbarrier_init(initial_state_ready_mbar, 4)
            if cutlass.const_expr(state_ckpt is not None):
                prims.mbarrier_init(checkpoint_read_done_mbar, 4)
            for stage in cutlass.range_constexpr(DECAY_STAGE_COUNT):
                prims.mbarrier_init(operand_smem_consumed_mbar.subview(stage), 3)
                prims.mbarrier_init(k_restore_consumed_mbar.subview(stage), 1)
                prims.mbarrier_init(k_restore_consumed_l_mbar.subview(stage), 1)
            prims.mbarrier_init(rhs_ready_mbar, 4)
            prims.mbarrier_init(update_ready_mbar, 8)
            prims.mbarrier_init(final_state_stored_mbar, 4)
    elif warp_idx == ROLES.super_mma:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(PAIRWISE_STAGE_COUNT):
                prims.mbarrier_init(pairwise_ready_mbar.subview(stage), 1)
                prims.mbarrier_init(qk_ready_mbar.subview(stage), 1)
                prims.mbarrier_init(pairwise_consumed_mbar.subview(stage), 1)
            for stage in cutlass.range_constexpr(Q_K_RESTORE_READY_STAGE_COUNT):
                prims.mbarrier_init(q_k_restore_ready_mbar.subview(stage), 4)
            for stage in cutlass.range_constexpr(DECAY_STAGE_COUNT):
                prims.mbarrier_init(cg0_k_ready_mbar.subview(stage), 4)
                prims.mbarrier_init(cg0_k_half_ready_mbar.subview(stage), 4)
            for stage in cutlass.range_constexpr(RAW_STAGE_COUNT):
                prims.mbarrier_init(diag_ready_mbar.subview(stage), 4)
    elif warp_idx == ROLES.epilogue:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(O_STAGE_COUNT):
                prims.mbarrier_init(output_ready_mbar.subview(stage), 4)
                prims.mbarrier_init(output_consumed_mbar.subview(stage), 1)
    prims.fence_mbarrier_init()
    cta_sync()
    if is_tmem_user_warp(warp_idx):
        if warp_idx == ROLES.tcgen05_mma:
            prims.tcgen05_alloc(tmem_ptr_i32, KDA_TMEM_ALLOC_COLS, group="cta_1")
        tmem_user_sync()
        if warp_idx == ROLES.tcgen05_mma:
            prims.tcgen05_relinquish_alloc_permit(group="cta_1")
        tmem_user_sync()
    qstate_acc_ready_phase = cutlass.Int32(0)
    state_input_ready_phase = cutlass.Int32(0)
    state_input_ready_l_phase = cutlass.Int32(0)
    rhs_ready_phase = cutlass.Int32(0)
    update_ready_phase = cutlass.Int32(0)
    final_state_stored_phase = cutlass.Int32(0)

    # Actual SMEM/TMEM buffers for the BT=16 schedule:
    #   q/k/v              : 16 x 128 each
    #   gate_log2          : 16 x 128
    #   beta               : 16
    #   q/k inverse norm   : 16 each, staged once per decay stage
    #   exp_g_last         : 128, CG0-local and staged once per decay stage
    #   state decay        : row 15 of the fp32 gate-prefix exchange tile
    #   q_decay/k_decay    : tcgen05 SW128 operands shared with auxiliary MMA
    #   k_restore          : tcgen05 SW128 N-major final-state operand
    #   k_inv              : 16 x 128 token-major for auxiliary-MMA RHS
    #   A inverse/QK       : 16 x 16 each, plus transposed tcgen05 operands
    #   state              : external/kernel ABI is VK `[DV, DK]`; reference
    #                        math can view it as KV `[DK, DV]` by transposing.
    #                        The TS A-staging path keeps VK in TMEM so state*k
    #                        is `[DV, DK] @ [DK, BT] -> [DV, BT]` with M=128.

    if is_service_warpgroup(warp_idx):
        prims.setmaxregister(KDA_SERVICE_REGS, prims.SetMaxRegisterAction.DECREASE)
        if warp_idx == ROLES.tma_load:
            # Gate constants are needed by BOTH the safe sigmoid and the
            # FLA non-safe softplus activations (a_log_exp == exp(A_log),
            # dt_bias per key dim).  Materialize unconditionally; for
            # SAFE_GATE=True this traces exactly as before (byte-identical
            # generated code), the non-safe path now reads the same staged constants.
            if lane == 0:
                raw_dt_bias_smem[RAW_DT_BIAS_A_LOG_EXP_OFFSET] = cute.math.exp2(
                    # pyrefly: ignore [missing-attribute]
                    a_log[bidy].to(cutlass.Float32) * LOG2_E,
                    fastmath=True,
                )
            for dim_group in cutlass.range_constexpr(DK // THREADS_PER_WARP):
                dim = dim_group * THREADS_PER_WARP + lane
                # pyrefly: ignore [missing-attribute]
                raw_dt_bias_smem[dim] = dt_bias[bidy, dim].to(cutlass.Float32)

            # Prime the beta tile ring: LOOKAHEAD tiles in flight before the
            # first consumption so the wait below is never exposed.  Group
            # coords walk (head-group, token); pair coords walk packed rows
            # from the g-aligned floor of the sequence start.
            beta_row0 = sequence_start // beta_g
            beta_rows_per_chunk = cutlass.Int32(BT) // beta_g
            beta_head_group = (bidy // cutlass.Int32(8)) * cutlass.Int32(8)
            for pre in cutlass.range_constexpr(BETA_TMA_LOOKAHEAD):
                if cutlass.Int32(pre) < num_chunks:
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            beta_tile_mbar.subview(pre % BETA_TILE_STAGE_COUNT),
                            beta_tile_tx,
                        )
                        beta_c0 = beta_head_group
                        beta_c1 = sequence_start + cutlass.Int32(pre * BT)
                        if beta_g > cutlass.Int32(1):
                            beta_c0 = cutlass.Int32(0)
                            beta_c1 = (
                                beta_row0 + cutlass.Int32(pre) * beta_rows_per_chunk
                            )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            beta_tile_smem.subview(
                                (pre % BETA_TILE_STAGE_COUNT) * BETA_TILE_STAGE_ELEMS
                            ),
                            tma_desc_beta.get_ptr(),
                            (beta_c0, beta_c1),
                            beta_tile_mbar.subview(pre % BETA_TILE_STAGE_COUNT),
                        )
            raw_stage = cutlass.Int32(0)
            raw_consumed_phase = cutlass.Int32(1)
            for chunk in cutlass.range(num_chunks, unroll=1):
                chunk_start = chunk * BT
                beta_tile_slot = chunk % BETA_TILE_STAGE_COUNT
                raw_q_stage = raw_q_smem.subview(raw_stage * RAW_Q_STAGE_SIZE)
                raw_k_stage = raw_k_smem.subview(raw_stage * RAW_K_STAGE_SIZE)
                raw_v_stage = raw_v_smem.subview(raw_stage * RAW_V_STAGE_SIZE)
                raw_gate_stage = raw_gate_smem.subview(raw_stage * RAW_GATE_STAGE_SIZE)
                raw_beta_stage = raw_beta_smem.subview(raw_stage * RAW_BETA_STAGE_SIZE)
                raw_consumed_wait(
                    raw_consumed_mbar.subview(raw_stage),
                    raw_consumed_phase,
                )
                # TMA issues q/k/v/gate into typed SMEM against the chunk's
                # tma_mbar ring slot (consumers wait that slot's parity
                # directly). The helper waits for beta only after raw issue;
                # raw_consumed throttles issue to <= RAW_STAGE_COUNT in flight.
                tma_stage_load_inputs(
                    tma_desc_q,
                    tma_desc_k,
                    tma_desc_v,
                    tma_desc_gate,
                    beta_tile_smem.subview(beta_tile_slot * BETA_TILE_STAGE_ELEMS),
                    beta_tile_mbar.subview(beta_tile_slot),
                    (chunk // BETA_TILE_STAGE_COUNT) % 2,
                    beta_g,
                    heads32,
                    raw_q_stage,
                    raw_k_stage,
                    raw_v_stage,
                    raw_gate_stage,
                    raw_beta_stage,
                    sequence_start,
                    bidy,
                    lane,
                    chunk_start,
                    seqlen,
                    tma_mbar.subview(raw_stage),
                    tma_tx_bytes,
                    gate_dtype,
                )
                beta_next = chunk + BETA_TMA_LOOKAHEAD
                if beta_next < num_chunks:
                    beta_next_slot = beta_next % BETA_TILE_STAGE_COUNT
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            beta_tile_mbar.subview(beta_next_slot),
                            beta_tile_tx,
                        )
                        beta_c0 = beta_head_group
                        beta_c1 = sequence_start + beta_next * BT
                        if beta_g > cutlass.Int32(1):
                            beta_c0 = cutlass.Int32(0)
                            beta_c1 = beta_row0 + beta_next * beta_rows_per_chunk
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            beta_tile_smem.subview(
                                beta_next_slot * BETA_TILE_STAGE_ELEMS
                            ),
                            tma_desc_beta.get_ptr(),
                            (beta_c0, beta_c1),
                            beta_tile_mbar.subview(beta_next_slot),
                        )
                raw_stage, raw_wrapped = advance_ring_stage(
                    raw_stage,
                    1,
                    RAW_STAGE_COUNT,
                )
                raw_consumed_phase = raw_consumed_phase ^ raw_wrapped

        elif warp_idx == ROLES.super_mma:
            raw_stage = cutlass.Int32(0)
            for chunk in cutlass.range(num_chunks, unroll=1):
                decay_stage = chunk % DECAY_STAGE_COUNT
                pairwise_stage = chunk % PAIRWISE_STAGE_COUNT
                raw_beta_stage = raw_beta_smem.subview(raw_stage * RAW_BETA_STAGE_SIZE)
                k_inv_stage = k_inv_smem.subview(decay_stage * K_INV_STAGE_SIZE)
                tcgen05_k_decay_stage = tcgen05_k_decay_smem.subview(
                    decay_stage * TCGEN05_K_DECAY_STAGE_SIZE
                )
                pairwise_stage_smem = pairwise_smem.subview(
                    pairwise_stage * PAIRWISE_SMEM_STAGE_SIZE
                )

                pairwise_consumed_wait(
                    pairwise_consumed_mbar.subview(pairwise_stage),
                    ((chunk // PAIRWISE_STAGE_COUNT) + 1) % 2,
                )
                cg0_k_ready_wait(
                    cg0_k_half_ready_mbar.subview(decay_stage),
                    (chunk // DECAY_STAGE_COUNT) % 2,
                )
                # KDA schedule owner: standalone auxiliary-MMA warp.  The first
                # KK half runs on the half-DK arrival; the pipeline waits the
                # full cg0_k_ready on its own before the second half.
                super_mma_stage_pairwise_pipeline(
                    tcgen05_k_decay_stage,
                    k_inv_stage,
                    pairwise_stage_smem,
                    raw_beta_stage,
                    cg0_k_ready_mbar.subview(decay_stage),
                    (chunk // DECAY_STAGE_COUNT) % 2,
                    lane,
                    input_dtype,
                )
                pairwise_ready_arrive(pairwise_ready_mbar.subview(pairwise_stage))
                operand_smem_consumed_arrive(
                    operand_smem_consumed_mbar.subview(decay_stage)
                )
                raw_consumed_arrive(raw_consumed_mbar.subview(raw_stage))
                raw_stage, _ = advance_ring_stage(raw_stage, 1, RAW_STAGE_COUNT)

        elif warp_idx == ROLES.tcgen05_mma:
            tmem_raw_addr = tmem_ptr_i32.load()
            shared_acc_event_id = cutlass.Int32(0)
            late_operand_stage = cutlass.Int32(0)
            q_k_restore_ready_phase = cutlass.Int32(0)
            for chunk in cutlass.range(num_chunks, unroll=1):
                o_stage = chunk % O_STAGE_COUNT
                qstate_acc_stage = chunk % KDA_TMEM_QSTATE_ACC_STAGE_COUNT
                decay_stage = chunk % DECAY_STAGE_COUNT
                q_k_restore_ready_stage = late_operand_stage
                pairwise_stage = chunk % PAIRWISE_STAGE_COUNT
                shared_input_stage = o_stage
                tcgen05_k_decay_stage = tcgen05_k_decay_smem.subview(
                    decay_stage * TCGEN05_K_DECAY_STAGE_SIZE
                )
                tcgen05_q_decay_stage = tcgen05_q_decay_smem.subview(
                    decay_stage * TCGEN05_Q_DECAY_STAGE_SIZE
                )
                tcgen05_k_restore_stage = tcgen05_k_restore_smem.subview(
                    decay_stage * TCGEN05_K_RESTORE_STAGE_SIZE
                )
                pairwise_stage_smem = pairwise_smem.subview(
                    pairwise_stage * PAIRWISE_SMEM_STAGE_SIZE
                )

                cg0_k_ready_wait(
                    cg0_k_ready_mbar.subview(decay_stage),
                    (chunk // DECAY_STAGE_COUNT) % 2,
                )

                state_input_ready_l_phase = state_input_ready_wait(
                    state_input_ready_l_mbar,
                    state_input_ready_l_phase,
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                state_k_acc_stage = tcgen05_shared_acc_stage_from_event(
                    shared_acc_event_id
                )
                tcgen05_issue_state_k_mma(
                    tcgen05_k_decay_stage,
                    tmem_raw_addr,
                    shared_acc_ready_mbar.subview(state_k_acc_stage),
                    state_k_acc_stage,
                    input_dtype,
                    0,
                    (DK // TCGEN05_F16_K_ATOM) // 2,
                    False,
                    False,
                    DV,
                )
                state_input_ready_phase = state_input_ready_wait(
                    state_input_ready_mbar,
                    state_input_ready_phase,
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                tcgen05_issue_state_k_mma(
                    tcgen05_k_decay_stage,
                    tmem_raw_addr,
                    shared_acc_ready_mbar.subview(state_k_acc_stage),
                    state_k_acc_stage,
                    input_dtype,
                    (DK // TCGEN05_F16_K_ATOM) // 2,
                    DK // TCGEN05_F16_K_ATOM,
                    True,
                    True,
                    DV,
                )
                shared_acc_event_id += cutlass.Int32(1)

                q_k_restore_ready_wait(
                    q_k_restore_ready_mbar.subview(q_k_restore_ready_stage),
                    q_k_restore_ready_phase,
                )

                qstate_acc_reuse_phase = (
                    chunk // KDA_TMEM_QSTATE_ACC_STAGE_COUNT + cutlass.Int32(1)
                ) % cutlass.Int32(2)
                output_ready_wait(
                    output_ready_mbar.subview(qstate_acc_stage),
                    qstate_acc_reuse_phase,
                )

                # State*Q tile producer.  This uses the dedicated qstate_acc slot
                # because the tile stays live until qkv is fused into output.
                tcgen05_issue_state_q_mma(
                    tcgen05_q_decay_stage,
                    tmem_raw_addr,
                    operand_smem_consumed_mbar.subview(decay_stage),
                    qstate_acc_stage,
                    input_dtype,
                )
                pairwise_ready_wait(
                    pairwise_ready_mbar.subview(pairwise_stage),
                    (chunk // PAIRWISE_STAGE_COUNT) % 2,
                )
                rhs_ready_phase = rhs_ready_wait(
                    rhs_ready_mbar,
                    rhs_ready_phase,
                )
                # KDA schedule owner: tcgen05-MMA warp.
                #
                # Update tile producer:
                #   update = A_inv @ rhs
                update_acc_stage = tcgen05_shared_acc_stage_from_event(
                    shared_acc_event_id
                )
                tcgen05_issue_update_mma(
                    pairwise_stage_smem,
                    tmem_raw_addr,
                    shared_acc_ready_mbar.subview(update_acc_stage),
                    update_acc_stage,
                    shared_input_stage,
                    input_dtype,
                )
                shared_acc_event_id += cutlass.Int32(1)

                update_ready_phase = update_ready_wait(
                    update_ready_mbar,
                    update_ready_phase,
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                # Final-state producer:
                #   final_state_acc += update @ k_restore
                tcgen05_issue_final_state_delta_mma(
                    tcgen05_k_restore_stage,
                    tmem_raw_addr,
                    k_restore_consumed_l_mbar.subview(decay_stage),
                    k_restore_consumed_mbar.subview(decay_stage),
                    shared_input_stage,
                    input_dtype,
                )

                pairwise_ready_wait(
                    qk_ready_mbar.subview(pairwise_stage),
                    (chunk // PAIRWISE_STAGE_COUNT) % 2,
                )
                # Output tile producer:
                #   staged_o = (state_q + qk @ update) * scale
                tcgen05_issue_qkv_mma(
                    pairwise_stage_smem,
                    tmem_raw_addr,
                    qstate_acc_ready_mbar,
                    shared_input_stage,
                    qstate_acc_stage,
                    input_dtype,
                )
                pairwise_consumed_arrive(pairwise_consumed_mbar.subview(pairwise_stage))
                late_operand_stage, late_operand_wrapped = advance_ring_stage(
                    late_operand_stage,
                    1,
                    Q_K_RESTORE_READY_STAGE_COUNT,
                )
                q_k_restore_ready_phase = q_k_restore_ready_phase ^ late_operand_wrapped

            final_state_stored_phase = final_state_stored_wait(
                final_state_stored_mbar,
                final_state_stored_phase,
            )
            tmem_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Float32)
            prims.tcgen05_dealloc(tmem_ptr, KDA_TMEM_ALLOC_COLS, group="cta_1")

        elif warp_idx == ROLES.epilogue:
            q_k_restore_ready_stage = cutlass.Int32(0)
            q_k_restore_ready_phase = cutlass.Int32(0)
            for chunk in cutlass.range(num_chunks, unroll=1):
                decay_stage = chunk % DECAY_STAGE_COUNT
                pairwise_stage = chunk % PAIRWISE_STAGE_COUNT
                k_inv_stage = k_inv_smem.subview(decay_stage * K_INV_STAGE_SIZE)
                tcgen05_q_decay_stage = tcgen05_q_decay_smem.subview(
                    decay_stage * TCGEN05_Q_DECAY_STAGE_SIZE
                )
                pairwise_stage_smem = pairwise_smem.subview(
                    pairwise_stage * PAIRWISE_SMEM_STAGE_SIZE
                )

                pairwise_consumed_wait(
                    pairwise_consumed_mbar.subview(pairwise_stage),
                    ((chunk // PAIRWISE_STAGE_COUNT) + 1) % 2,
                )
                q_k_restore_ready_wait(
                    q_k_restore_ready_mbar.subview(q_k_restore_ready_stage),
                    q_k_restore_ready_phase,
                )
                super_mma_stage_qk(
                    tcgen05_q_decay_stage,
                    k_inv_stage,
                    pairwise_stage_smem,
                    lane,
                    input_dtype,
                )
                pairwise_ready_arrive(qk_ready_mbar.subview(pairwise_stage))
                operand_smem_consumed_arrive(
                    operand_smem_consumed_mbar.subview(decay_stage)
                )
                q_k_restore_ready_stage, q_k_restore_wrapped = advance_ring_stage(
                    q_k_restore_ready_stage,
                    1,
                    Q_K_RESTORE_READY_STAGE_COUNT,
                )
                q_k_restore_ready_phase = q_k_restore_ready_phase ^ q_k_restore_wrapped

                if chunk > 0:
                    output_chunk = chunk - cutlass.Int32(1)
                    epilogue_wait_and_store_full_output(
                        tma_desc_o,
                        o_smem,
                        output_ready_mbar,
                        output_consumed_mbar,
                        sequence_start,
                        bidy,
                        output_chunk,
                        O_STAGE_COUNT,
                    )
            if num_chunks > 0:
                output_chunk = num_chunks - cutlass.Int32(1)
                epilogue_wait_and_store_final_output(
                    tma_desc_o,
                    out,
                    o_smem,
                    output_ready_mbar,
                    output_consumed_mbar,
                    sequence_start,
                    bidy,
                    seqlen,
                    output_chunk,
                    lane,
                    O_STAGE_COUNT,
                )
    elif is_compute_group0_warp(warp_idx):
        prims.setmaxregister(KDA_CG0_REGS, prims.SetMaxRegisterAction.INCREASE)
        cg0_warp = warp_idx - ROLES.compute_group0_first
        cg0_group_id = cg0_warp // CG0_WARPS_PER_GROUP
        cg0_local_warp = cg0_warp % CG0_WARPS_PER_GROUP
        cg0_a_log_exp = cutlass.Float32(1.0)
        cg0_dt_bias_value = cutlass.Float32(0.0)
        tmem_raw_addr = cutlass.Int32(0)
        if num_chunks > 0:
            raw_ready_wait(tma_mbar.subview(0), 0)
            prefix_dim = cg0_local_warp * THREADS_PER_WARP + lane
            cg0_a_log_exp = raw_dt_bias_smem[RAW_DT_BIAS_A_LOG_EXP_OFFSET]
            cg0_dt_bias_value = raw_dt_bias_smem[prefix_dim]
            state_input_ready_wait(initial_state_ready_mbar, cutlass.Int32(0))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            # The allocator publishes one immutable TMEM base for the CTA.
            # Cache it after the initial-state handoff instead of reloading
            # the SMEM pointer in every steady and peeled CG0 iteration.
            tmem_raw_addr = tmem_ptr_i32.load()
        raw_stage = cg0_group_id
        raw_ready_phase = cutlass.Int32(0)
        late_operand_stage = cg0_group_id
        cg0_ckpt_stride = cutlass.Int32(0)
        if cutlass.const_expr(state_ckpt is not None):
            cg0_ckpt_stride = checkpoint_stride_chunks
        # v27_peel: STRUCTURAL in-kernel tail-peel of the CG0 producer loop.
        # The interior mainloop runs the GUARD-FREE body over chunks
        # [cg0_group_id, num_chunks-1): NO per-chunk `chunk_start+BT>seqlen`
        # compare and NO tail zeroing (materialize called with the FULL branch).
        # The single genuinely-partial tail chunk (num_chunks-1) is peeled out
        # once below and run with the masked tail fixup.  This makes ONE kernel
        # correct for both aligned and ragged inputs (the peeled tail is a
        # runtime no-op when the sequence is BT-aligned).  Ring-state locals
        # (raw_stage / phase, late_operand_stage / phase) are loop-carried and
        # left positioned for the peeled chunk by the mainloop.
        for chunk in cutlass.range(
            cg0_group_id,
            num_chunks - cutlass.Int32(1),
            CG0_GROUP_COUNT,
            unroll=1,
        ):
            decay_stage = chunk % DECAY_STAGE_COUNT
            q_k_restore_ready_stage = late_operand_stage
            raw_q_stage = raw_q_smem.subview(raw_stage * RAW_Q_STAGE_SIZE)
            raw_k_stage = raw_k_smem.subview(raw_stage * RAW_K_STAGE_SIZE)
            raw_gate_stage = raw_gate_smem.subview(raw_stage * RAW_GATE_STAGE_SIZE)
            # FP32 exp2(g_prefix) exchange tile.  With an FP32 gate it IS the
            # raw-gate stage (historical aliasing, zero cost); with a 16-bit gate
            # it comes from the dedicated 4-deep gate_exchange ring.
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                gate_exchange_stage = raw_gate_stage
            else:
                # pyrefly: ignore [missing-attribute]
                gate_exchange_stage = gate_exchange_smem.subview(
                    (chunk % GATE_EXCHANGE_STAGE_COUNT) * GATE_EXCHANGE_STAGE_SIZE
                )
            k_inv_stage = k_inv_smem.subview(decay_stage * K_INV_STAGE_SIZE)
            tcgen05_k_decay_stage = tcgen05_k_decay_smem.subview(
                decay_stage * TCGEN05_K_DECAY_STAGE_SIZE
            )
            tcgen05_q_decay_stage = tcgen05_q_decay_smem.subview(
                decay_stage * TCGEN05_Q_DECAY_STAGE_SIZE
            )
            tcgen05_k_restore_stage = tcgen05_k_restore_smem.subview(
                decay_stage * TCGEN05_K_RESTORE_STAGE_SIZE
            )
            raw_ready_wait(
                tma_mbar.subview(raw_stage),
                raw_ready_phase,
            )

            # KDA schedule owner: compute warp group 0.
            #
            # The decay materialization stages full gate prefixes once, then
            # issues the two gate MUFU families for the full KDA operand set:
            #   exp2(g_prefix)      : 16 * 128
            #   exp2(-g_prefix)     : 16 * 128
            cg0_materialize_decay_operands(
                raw_q_stage,
                raw_k_stage,
                raw_gate_stage,
                gate_exchange_stage,
                cg0_a_log_exp,
                cg0_dt_bias_value,
                k_inv_stage,
                tcgen05_k_decay_stage,
                tcgen05_q_decay_stage,
                tcgen05_k_restore_stage,
                cg0_k_ready_mbar.subview(decay_stage),
                cg0_k_half_ready_mbar.subview(decay_stage),
                diag_ready_mbar.subview(raw_stage),
                operand_smem_consumed_mbar.subview(decay_stage),
                k_restore_consumed_mbar.subview(decay_stage),
                chunk,
                seqlen,
                input_dtype,
                gate_dtype,
                SAFE_GATE,
                GATE_SCALE_LOG2,
                True,
                cg0_group_id,
                cg0_local_warp,
                lane,
            )
            q_k_restore_ready_arrive(
                q_k_restore_ready_mbar.subview(q_k_restore_ready_stage)
            )
            if chunk > 0:
                prev_state_chunk = chunk - cutlass.Int32(1)
                tcgen05_wait_acc_buffer_ready(
                    k_restore_consumed_l_mbar.subview(
                        prev_state_chunk % DECAY_STAGE_COUNT
                    ),
                    (prev_state_chunk // DECAY_STAGE_COUNT) % 2,
                )
            if cutlass.const_expr(state_ckpt is not None):
                if (chunk > cutlass.Int32(0)) & (chunk % cg0_ckpt_stride == 0):
                    checkpoint_read_done_wait(
                        checkpoint_read_done_mbar,
                        ((chunk // cg0_ckpt_stride) + cutlass.Int32(1)) % 2,
                    )
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tcgen05_pack_rescale_state_half_tmem(
                tmem_raw_addr,
                gate_exchange_stage,
                state_input_ready_l_mbar,
                warp_idx,
                input_dtype,
                0,
                True,
            )
            raw_consumed_arrive(raw_consumed_mbar.subview(raw_stage))
            update_ready_arrive(update_ready_mbar)
            raw_stage, raw_wrapped = advance_ring_stage(
                raw_stage,
                CG0_GROUP_COUNT,
                RAW_STAGE_COUNT,
            )
            raw_ready_phase = raw_ready_phase ^ raw_wrapped
            late_operand_stage, late_operand_wrapped = advance_ring_stage(
                late_operand_stage,
                CG0_GROUP_COUNT,
                Q_K_RESTORE_READY_STAGE_COUNT,
            )

        # Peeled final (tail) chunk num_chunks-1, executed once by the CG0 group
        # that owns it.  num_chunks==1 -> the single chunk IS the tail (group 0);
        # num_chunks==0 -> no work.  materialize runs the tail branch (its
        # built-in `tail_valid<BT` runtime test is a no-op when BT-aligned, so
        # aligned output is bit-identical to the guard-free FULL body).
        if num_chunks > cutlass.Int32(0):
            if (num_chunks - cutlass.Int32(1)) % CG0_GROUP_COUNT == cg0_group_id:
                chunk = num_chunks - cutlass.Int32(1)
                chunk_start = chunk * BT
                decay_stage = chunk % DECAY_STAGE_COUNT
                q_k_restore_ready_stage = late_operand_stage
                raw_q_stage = raw_q_smem.subview(raw_stage * RAW_Q_STAGE_SIZE)
                raw_k_stage = raw_k_smem.subview(raw_stage * RAW_K_STAGE_SIZE)
                raw_v_stage = raw_v_smem.subview(raw_stage * RAW_V_STAGE_SIZE)
                raw_gate_stage = raw_gate_smem.subview(raw_stage * RAW_GATE_STAGE_SIZE)
                # FP32 exp2(g_prefix) exchange tile.  With an FP32 gate it IS the
                # raw-gate stage (historical aliasing, zero cost); with a 16-bit gate
                # it comes from the dedicated 4-deep gate_exchange ring.
                if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                    gate_exchange_stage = raw_gate_stage
                else:
                    # pyrefly: ignore [missing-attribute]
                    gate_exchange_stage = gate_exchange_smem.subview(
                        (chunk % GATE_EXCHANGE_STAGE_COUNT) * GATE_EXCHANGE_STAGE_SIZE
                    )
                k_inv_stage = k_inv_smem.subview(decay_stage * K_INV_STAGE_SIZE)
                tcgen05_k_decay_stage = tcgen05_k_decay_smem.subview(
                    decay_stage * TCGEN05_K_DECAY_STAGE_SIZE
                )
                tcgen05_q_decay_stage = tcgen05_q_decay_smem.subview(
                    decay_stage * TCGEN05_Q_DECAY_STAGE_SIZE
                )
                tcgen05_k_restore_stage = tcgen05_k_restore_smem.subview(
                    decay_stage * TCGEN05_K_RESTORE_STAGE_SIZE
                )
                raw_ready_wait(
                    tma_mbar.subview(raw_stage),
                    raw_ready_phase,
                )

                if chunk_start + cutlass.Int32(BT) > seqlen:
                    if cg0_local_warp == 0:
                        cg0_zero_tail_raw_operands(
                            raw_q_stage,
                            raw_k_stage,
                            raw_v_stage,
                            raw_gate_stage,
                            lane,
                            chunk_start,
                            seqlen,
                            q.element_type,
                            gate_dtype,
                        )
                    cg0_sync(cg0_group_id)

                cg0_materialize_decay_operands(
                    raw_q_stage,
                    raw_k_stage,
                    raw_gate_stage,
                    gate_exchange_stage,
                    cg0_a_log_exp,
                    cg0_dt_bias_value,
                    k_inv_stage,
                    tcgen05_k_decay_stage,
                    tcgen05_q_decay_stage,
                    tcgen05_k_restore_stage,
                    cg0_k_ready_mbar.subview(decay_stage),
                    cg0_k_half_ready_mbar.subview(decay_stage),
                    diag_ready_mbar.subview(raw_stage),
                    operand_smem_consumed_mbar.subview(decay_stage),
                    k_restore_consumed_mbar.subview(decay_stage),
                    chunk,
                    seqlen,
                    input_dtype,
                    gate_dtype,
                    SAFE_GATE,
                    GATE_SCALE_LOG2,
                    False,
                    cg0_group_id,
                    cg0_local_warp,
                    lane,
                )
                q_k_restore_ready_arrive(
                    q_k_restore_ready_mbar.subview(q_k_restore_ready_stage)
                )
                if chunk > 0:
                    prev_state_chunk = chunk - cutlass.Int32(1)
                    tcgen05_wait_acc_buffer_ready(
                        k_restore_consumed_l_mbar.subview(
                            prev_state_chunk % DECAY_STAGE_COUNT
                        ),
                        (prev_state_chunk // DECAY_STAGE_COUNT) % 2,
                    )
                if cutlass.const_expr(state_ckpt is not None):
                    if (chunk > cutlass.Int32(0)) & (chunk % cg0_ckpt_stride == 0):
                        checkpoint_read_done_wait(
                            checkpoint_read_done_mbar,
                            ((chunk // cg0_ckpt_stride) + cutlass.Int32(1)) % 2,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                tcgen05_pack_rescale_state_half_tmem(
                    tmem_raw_addr,
                    gate_exchange_stage,
                    state_input_ready_l_mbar,
                    warp_idx,
                    input_dtype,
                    0,
                    True,
                )
                raw_consumed_arrive(raw_consumed_mbar.subview(raw_stage))
                update_ready_arrive(update_ready_mbar)

    elif is_compute_group1_warp(warp_idx):
        prims.setmaxregister(KDA_CG1_REGS, prims.SetMaxRegisterAction.INCREASE)
        tmem_raw_addr = tmem_ptr_i32.load()
        # Only CG1 touches recurrent state. Keep the pool lookup outside the
        # chunk loop and out of producer/MMAs warps.
        state_slot = bidx
        if cutlass.const_expr(state_indices is not None):
            # pyrefly: ignore [unsupported-operation]
            state_slot = cutlass.Int32(state_indices[bidx])
        ckpt_slot = cutlass.Int32(0)
        if cutlass.const_expr(state_ckpt is not None):
            ckpt_stride = checkpoint_stride_chunks
            ckpt_next = ckpt_stride
            # pyrefly: ignore [unsupported-operation]
            ckpt_slot = cutlass.Int32(cu_ckpts[bidx])
        if cutlass.const_expr(checkpoint_state_indices is not None):
            tcgen05_store_initial_state_tmem(
                tmem_raw_addr,
                initial_state,
                None,
                cutlass.Int32(0),
                state_slot,
                bidy,
                cutlass.Int32(0),
                warp_idx,
                lane,
                HALF=False,
            )
        else:
            tcgen05_store_initial_state_tmem(
                tmem_raw_addr,
                initial_state,
                state_ckpt,
                ckpt_slot,
                state_slot,
                bidy,
                cutlass.Int32(0),
                warp_idx,
                lane,
                HALF=False,
            )
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        state_input_ready_arrive(initial_state_ready_mbar)
        shared_acc_event_id = cutlass.Int32(0)
        if cutlass.const_expr(
            state_ckpt is not None and checkpoint_state_indices is None
        ):
            # cu_ckpts supplies the per-sequence base slot offsets.  A
            # loop-carried counter replaces a per-chunk div/mod.
            ckpt_slot += cutlass.Int32(1)

        if num_chunks > 0:
            shared_input_stage = cutlass.Int32(0)
            raw_v_stage = raw_v_smem.subview(0)
            raw_beta_stage = raw_beta_smem.subview(0)
            raw_gate_stage = raw_gate_smem.subview(0)
            # FP32 exp2(g_prefix) exchange tile for the CG1 prologue (chunk 0).
            # With an FP32 gate it IS the raw-gate stage (historical aliasing,
            # zero cost); with a 16-bit gate it is stage 0 of the dedicated
            # 4-deep gate_exchange ring (chunk 0 -> stage 0).
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                gate_exchange_stage = raw_gate_stage
            else:
                # pyrefly: ignore [missing-attribute]
                gate_exchange_stage = gate_exchange_smem.subview(0)

            state0, state1, state2, state3 = tcgen05_stage_state_input_tmem(
                tmem_raw_addr,
                warp_idx,
                output_consumed_mbar.subview(0),
                cutlass.Int32(0),
                input_dtype,
                False,
                1,
                True,
            )
            # Decay the retained right half as soon as CG0 publishes the FP32
            # diagonal; K/Q materialization continues independently.
            diag_ready_wait(diag_ready_mbar.subview(0), 0)
            tcgen05_publish_projection_then_rescale_state_regs(
                tmem_raw_addr,
                gate_exchange_stage,
                state_input_ready_mbar,
                warp_idx,
                state0,
                state1,
                state2,
                state3,
                1,
                True,
            )

            state_k_acc_stage = tcgen05_shared_acc_stage_from_event(shared_acc_event_id)
            state_k_acc_phase = tcgen05_shared_acc_phase_from_event(shared_acc_event_id)
            tcgen05_wait_acc_buffer_ready(
                shared_acc_ready_mbar.subview(state_k_acc_stage),
                state_k_acc_phase,
            )
            rhs_lane = cute.arch.lane_idx()
            tcgen05_stage_rhs_input_tmem(
                tmem_raw_addr,
                raw_v_stage,
                raw_beta_stage,
                warp_idx,
                rhs_lane,
                state_k_acc_stage,
                shared_input_stage,
                input_dtype,
            )
            shared_acc_event_id += cutlass.Int32(1)
            rhs_ready_arrive(rhs_ready_mbar)

            raw_consumed_arrive(raw_consumed_mbar.subview(0))
            post_scale_lane = cute.arch.lane_idx()

            update_acc_stage = tcgen05_shared_acc_stage_from_event(shared_acc_event_id)
            update_acc_phase = tcgen05_shared_acc_phase_from_event(shared_acc_event_id)
            tcgen05_wait_acc_buffer_ready(
                shared_acc_ready_mbar.subview(update_acc_stage),
                update_acc_phase,
            )
            tcgen05_stage_update_input_tmem(
                tmem_raw_addr,
                warp_idx,
                update_acc_stage,
                shared_input_stage,
                input_dtype,
            )
            shared_acc_event_id += cutlass.Int32(1)
            update_ready_arrive(update_ready_mbar)

            if cutlass.const_expr(state_ckpt is not None):
                # Peeled chunk 0's checkpoint (fires only for stride 1, i.e. a
                # checkpoint every BT tokens).
                # pyrefly: ignore [unbound-name]
                checkpoint_due = ckpt_next == cutlass.Int32(1)
                if cutlass.const_expr(checkpoint_state_indices is not None):
                    checkpoint_due = checkpoint_due & (cutlass.Int32(BT) <= seqlen)
                else:
                    checkpoint_due = checkpoint_due & (cutlass.Int32(1) < num_chunks)
                if checkpoint_due:
                    tcgen05_wait_acc_buffer_ready(k_restore_consumed_mbar.subview(0), 0)
                    checkpoint_output_slot = ckpt_slot
                    if cutlass.const_expr(checkpoint_state_indices is not None):
                        checkpoint_output_slot = cutlass.Int32(
                            # pyrefly: ignore [unsupported-operation]
                            checkpoint_state_indices[ckpt_slot]
                        )
                    if (checkpoint_output_slot >= cutlass.Int32(0)) & (
                        # pyrefly: ignore [missing-attribute]
                        checkpoint_output_slot < cutlass.Int32(state_ckpt.shape[0])
                    ):
                        tcgen05_store_final_state_tmem(
                            tmem_raw_addr,
                            KDA_TMEM_FINAL_STATE_ACC_COL_OFFSET,
                            state_ckpt,
                            checkpoint_output_slot,
                            bidy,
                            cutlass.Int32(0),
                            warp_idx,
                            post_scale_lane,
                            HALF=False,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    checkpoint_read_done_arrive(checkpoint_read_done_mbar)
                    ckpt_slot += cutlass.Int32(1)
                    # pyrefly: ignore [unbound-name]
                    ckpt_next += ckpt_stride

        # Peel chunk 0 so the steady-state loop always drains a prior output.
        # Each actual shared-acc commit advances the event ring independently
        # of which GEMM produced it.
        raw_stage = cutlass.Int32(1)
        for chunk in cutlass.range(1, num_chunks, 1, unroll=1):
            o_stage = chunk % O_STAGE_COUNT
            decay_stage = chunk % DECAY_STAGE_COUNT
            shared_input_stage = o_stage
            raw_v_stage = raw_v_smem.subview(raw_stage * RAW_V_STAGE_SIZE)
            raw_beta_stage = raw_beta_smem.subview(raw_stage * RAW_BETA_STAGE_SIZE)
            raw_gate_stage = raw_gate_smem.subview(raw_stage * RAW_GATE_STAGE_SIZE)
            # FP32 exp2(g_prefix) exchange tile.  With an FP32 gate it IS the
            # raw-gate stage (historical aliasing, zero cost); with a 16-bit gate
            # it comes from the dedicated 4-deep gate_exchange ring.
            if cutlass.const_expr(gate_dtype_is_f32(gate_dtype)):
                gate_exchange_stage = raw_gate_stage
            else:
                # pyrefly: ignore [missing-attribute]
                gate_exchange_stage = gate_exchange_smem.subview(
                    (chunk % GATE_EXCHANGE_STAGE_COUNT) * GATE_EXCHANGE_STAGE_SIZE
                )

            prev_output_chunk = chunk - cutlass.Int32(1)
            prev_o_stage = prev_output_chunk % O_STAGE_COUNT
            prev_qstate_acc_stage = prev_output_chunk % KDA_TMEM_QSTATE_ACC_STAGE_COUNT
            prev_o_stage_base = prev_o_stage * O_SMEM_STAGE_SIZE
            prev_decay_stage = prev_output_chunk % DECAY_STAGE_COUNT
            prev_k_restore_phase = (prev_output_chunk // DECAY_STAGE_COUNT) % 2
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_mbar.subview(prev_decay_stage),
                prev_k_restore_phase,
            )
            state0, state1, state2, state3 = tcgen05_stage_state_input_tmem(
                tmem_raw_addr,
                warp_idx,
                output_consumed_mbar.subview(prev_o_stage),
                ((prev_output_chunk // O_STAGE_COUNT) + 1) % 2,
                input_dtype,
                True,
                1,
                True,
            )
            # The diagonal is complete before k_decay/q_decay/k_restore; use
            # that narrower dependency to hide right-half FP32 scaling.
            diag_ready_wait(
                diag_ready_mbar.subview(raw_stage),
                (chunk // RAW_STAGE_COUNT) % 2,
            )
            tcgen05_publish_projection_then_rescale_state_regs(
                tmem_raw_addr,
                gate_exchange_stage,
                state_input_ready_mbar,
                warp_idx,
                state0,
                state1,
                state2,
                state3,
                1,
                True,
            )
            pre_scale_lane = cute.arch.lane_idx()
            qstate_acc_ready_phase = tcgen05_wait_acc_buffer_ready(
                qstate_acc_ready_mbar,
                qstate_acc_ready_phase,
            )
            tcgen05_load_qstate_output_tmem(
                tmem_raw_addr,
                o_smem,
                warp_idx,
                pre_scale_lane,
                prev_o_stage_base,
                prev_qstate_acc_stage,
                SCALE,
                out.element_type,
            )
            output_ready_arrive(output_ready_mbar.subview(prev_o_stage))

            state_k_acc_stage = tcgen05_shared_acc_stage_from_event(shared_acc_event_id)
            state_k_acc_phase = tcgen05_shared_acc_phase_from_event(shared_acc_event_id)
            tcgen05_wait_acc_buffer_ready(
                shared_acc_ready_mbar.subview(state_k_acc_stage),
                state_k_acc_phase,
            )
            # KDA schedule owner: compute warp group 1.
            #
            # Epilogue for tcgen05 state*k, staged directly as the next
            # tcgen05 A operand:
            #   shared_input = beta_i * (v_i - state*k_i)
            tcgen05_stage_rhs_input_tmem(
                tmem_raw_addr,
                raw_v_stage,
                raw_beta_stage,
                warp_idx,
                pre_scale_lane,
                state_k_acc_stage,
                shared_input_stage,
                input_dtype,
            )
            shared_acc_event_id += cutlass.Int32(1)
            rhs_ready_arrive(rhs_ready_mbar)

            raw_consumed_arrive(raw_consumed_mbar.subview(raw_stage))
            post_scale_lane = cute.arch.lane_idx()

            update_acc_stage = tcgen05_shared_acc_stage_from_event(shared_acc_event_id)
            update_acc_phase = tcgen05_shared_acc_phase_from_event(shared_acc_event_id)
            tcgen05_wait_acc_buffer_ready(
                shared_acc_ready_mbar.subview(update_acc_stage),
                update_acc_phase,
            )
            tcgen05_stage_update_input_tmem(
                tmem_raw_addr,
                warp_idx,
                update_acc_stage,
                shared_input_stage,
                input_dtype,
            )
            shared_acc_event_id += cutlass.Int32(1)
            update_ready_arrive(update_ready_mbar)

            if cutlass.const_expr(state_ckpt is not None):
                # State checkpoint: at every ckpt_stride-th chunk boundary the
                # TMEM accumulator holds exactly the state a run truncated here
                # would emit as final_state.  Wait the SAME parity slot the
                # post-loop final store waits (the chunk's state-update MMA has
                # committed), then reuse the final-state store routine with the
                # flat checkpoint slot standing in for the sequence index.
                # pyrefly: ignore [unbound-name]
                checkpoint_due = chunk + cutlass.Int32(1) == ckpt_next
                if cutlass.const_expr(checkpoint_state_indices is not None):
                    checkpoint_due = checkpoint_due & (
                        (chunk + cutlass.Int32(1)) * cutlass.Int32(BT) <= seqlen
                    )
                else:
                    checkpoint_due = checkpoint_due & (
                        chunk + cutlass.Int32(1) < num_chunks
                    )
                if checkpoint_due:
                    tcgen05_wait_acc_buffer_ready(
                        k_restore_consumed_mbar.subview(chunk % DECAY_STAGE_COUNT),
                        (chunk // DECAY_STAGE_COUNT) % 2,
                    )
                    checkpoint_output_slot = ckpt_slot
                    if cutlass.const_expr(checkpoint_state_indices is not None):
                        checkpoint_output_slot = cutlass.Int32(
                            # pyrefly: ignore [unsupported-operation]
                            checkpoint_state_indices[ckpt_slot]
                        )
                    if (checkpoint_output_slot >= cutlass.Int32(0)) & (
                        # pyrefly: ignore [missing-attribute]
                        checkpoint_output_slot < cutlass.Int32(state_ckpt.shape[0])
                    ):
                        tcgen05_store_final_state_tmem(
                            tmem_raw_addr,
                            KDA_TMEM_FINAL_STATE_ACC_COL_OFFSET,
                            state_ckpt,
                            checkpoint_output_slot,
                            bidy,
                            cutlass.Int32(0),
                            warp_idx,
                            post_scale_lane,
                            HALF=False,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    checkpoint_read_done_arrive(checkpoint_read_done_mbar)
                    ckpt_slot += cutlass.Int32(1)
                    ckpt_next += ckpt_stride

            raw_stage, _ = advance_ring_stage(raw_stage, 1, RAW_STAGE_COUNT)

        if num_chunks > 0:
            final_lane = cute.arch.lane_idx()
            output_chunk = num_chunks - cutlass.Int32(1)
            last_decay_stage = output_chunk % DECAY_STAGE_COUNT
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_mbar.subview(last_decay_stage),
                (output_chunk // DECAY_STAGE_COUNT) % 2,
            )
            final_o_stage = output_chunk % O_STAGE_COUNT
            final_qstate_acc_stage = output_chunk % KDA_TMEM_QSTATE_ACC_STAGE_COUNT
            final_o_stage_base = final_o_stage * O_SMEM_STAGE_SIZE
            output_consumed_wait(
                output_consumed_mbar.subview(final_o_stage),
                ((output_chunk // O_STAGE_COUNT) + 1) % 2,
            )

            qstate_acc_ready_phase = tcgen05_wait_acc_buffer_ready(
                qstate_acc_ready_mbar,
                qstate_acc_ready_phase,
            )
            tcgen05_load_qstate_output_tmem(
                tmem_raw_addr,
                o_smem,
                warp_idx,
                final_lane,
                final_o_stage_base,
                final_qstate_acc_stage,
                SCALE,
                out.element_type,
            )
            output_ready_arrive(output_ready_mbar.subview(final_o_stage))

        if cutlass.const_expr(final_state is not None):
            final_lane = cute.arch.lane_idx()
            tcgen05_store_final_state_tmem(
                tmem_raw_addr,
                KDA_TMEM_FINAL_STATE_ACC_COL_OFFSET,
                final_state,
                state_slot,
                bidy,
                cutlass.Int32(0),
                warp_idx,
                final_lane,
                HALF=False,
            )
        final_state_stored_arrive(final_state_stored_mbar)


@cute.jit
def host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    seq_order: cute.Tensor | None,
    state_indices: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    stream,
    SCALE: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    checkpoint_stride_chunks: cutlass.Int32,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Float32,
    THREADS: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
    TASK_ORDER: cutlass.Constexpr = "identity",
    SEGMENT_BEGIN: cutlass.Constexpr = 0,
    SEGMENT_SIZE: cutlass.Constexpr = -1,
    # Python literal defaults are converted by CuTe inside the JIT boundary.
    # pyrefly: ignore [bad-function-definition]
    SEQUENCE_BEGIN: cutlass.Int32 = 0,
    SEQUENCE_COUNT: cutlass.Constexpr = 0,
) -> None:
    # pyrefly: ignore [bad-index]
    packed_batch = q.shape[0]
    # pyrefly: ignore [bad-index, unsupported-operation]
    num_sequences = cu_seqlens.shape[0] - 1
    if cutlass.const_expr(SEQUENCE_COUNT > 0):
        num_sequences = SEQUENCE_COUNT
    # pyrefly: ignore [bad-index]
    seqlen = q.shape[1]
    # pyrefly: ignore [bad-index]
    heads = q.shape[2]
    # Token-major activations: memory order [T, H, D] (token stride = D*heads),
    # so packed [1, T_total, H, D] / batched [B, T, H, D] callers are read
    # directly with no transpose.
    # Keep the unused outer stride representable by CuTe IR for packed inputs.
    # CuTe layouts currently lower strides through signed Int32, so the natural
    # D*T*H batch stride overflows once a singleton packed token buffer exceeds
    # that range even though the batch coordinate is always zero.
    qk_batch_stride = DK
    v_batch_stride = DV
    if packed_batch != cutlass.Int32(1):
        # pyrefly: ignore [unsupported-operation]
        qk_batch_stride = DK * seqlen * heads
        # pyrefly: ignore [unsupported-operation]
        v_batch_stride = DV * seqlen * heads
    qk_layout = cute.make_layout(
        (DK, seqlen, heads, packed_batch),
        stride=(1, DK * heads, DK, qk_batch_stride),
    )
    v_layout = cute.make_layout(
        (DV, seqlen, heads, packed_batch),
        stride=(1, DV * heads, DV, v_batch_stride),
    )
    q_tma = cute.make_tensor(q.iterator, qk_layout)
    k_tma = cute.make_tensor(k.iterator, qk_layout)
    v_tma = cute.make_tensor(v.iterator, v_layout)
    gate_tma = cute.make_tensor(raw_gate.iterator, qk_layout)
    out_tma = cute.make_tensor(out.iterator, v_layout)
    raw_f16_tma_box = (RAW_F16_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    raw_f32_tma_box = (RAW_F32_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    o_tma_box = (O_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    tma_desc_q = cuda.create_tensor_map_tiled_from_view(
        q_tma,
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_k = cuda.create_tensor_map_tiled_from_view(
        k_tma,
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_v = cuda.create_tensor_map_tiled_from_view(
        v_tma,
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    # The gate box/swizzle family follows `gate_dtype`: FP32 keeps the wide
    # 4 x 128 B box, a 16-bit gate uses the same family as q/k/v.
    gate_tma_box = raw_f32_tma_box if gate_dtype_is_f32(gate_dtype) else raw_f16_tma_box
    tma_desc_gate = cuda.create_tensor_map_tiled_from_view(
        gate_tma,
        box_dims=gate_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    # beta transport: one descriptor family over the SAME contiguous [T, H]
    # memory, runtime-shaped by g = 8/gcd(heads, 8) (see the constants block).
    # g == 1: view (heads, T) box (8, BT) -- per-head-group 16B rows.
    # g > 1 : packed view (g*heads, ceil(T/g)) box (g*heads, BT/g + 1).
    # pyrefly: ignore [bad-index]
    beta_heads = cutlass.Int32(beta.shape[2])
    beta_lsb = beta_heads & (-beta_heads)
    beta_g = cutlass.Int32(8) // cutlass.min(beta_lsb, cutlass.Int32(8))
    beta_is_pair = cutlass.Int32(1)
    if beta_g == cutlass.Int32(1):
        beta_is_pair = cutlass.Int32(0)
    # pyrefly: ignore [bad-index]
    beta_rows = (cutlass.Int32(beta.shape[1]) + beta_g - cutlass.Int32(1)) // beta_g
    beta_inner = beta_g * beta_heads
    beta_box_inner = (
        cutlass.Int32(8) * (cutlass.Int32(1) - beta_is_pair) + beta_inner * beta_is_pair
    )
    beta_box_outer = (
        cutlass.Int32(BT) * (cutlass.Int32(1) - beta_is_pair)
        + (cutlass.Int32(BT) // beta_g + cutlass.Int32(1)) * beta_is_pair
    )
    beta_tma_view = cute.make_tensor(
        beta.iterator,
        cute.make_layout((beta_inner, beta_rows), stride=(1, beta_inner)),
    )
    tma_desc_beta = cuda.create_tensor_map_tiled_from_view(
        beta_tma_view,
        box_dims=(beta_box_inner, beta_box_outer),
        stride_order=(0, 1),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_desc_o = cuda.create_tensor_map_tiled_from_view(
        out_tma,
        box_dims=o_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    kernel(
        tma_desc_q,
        tma_desc_k,
        tma_desc_v,
        tma_desc_gate,
        tma_desc_beta,
        tma_desc_o,
        q,
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        seq_order,
        state_indices,
        initial_state,
        out,
        final_state,
        SCALE,
        state_ckpt,
        cu_ckpts,
        checkpoint_state_indices,
        checkpoint_stride_chunks,
        SAFE_GATE,
        GATE_SCALE_LOG2,
        gate_dtype,
        TASK_ORDER,
        SEGMENT_BEGIN,
        SEGMENT_SIZE,
        SEQUENCE_BEGIN,
        SEQUENCE_COUNT > 0,
    ).launch(
        grid=(heads, num_sequences, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )
