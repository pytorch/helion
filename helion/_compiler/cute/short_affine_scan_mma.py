# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CuTe compile-time values intentionally have implicit types.
# ruff: noqa: ANN001, ANN202

"""BF16 warp-MMA primitives for a short direct affine scan.

This module contains only the physical operations shared by direct affine
scan lowerings.  It knows about warp fragments and shared-memory layouts, but
not about an originating FX graph, application tensor names, pointer formulas,
or a complete kernel schedule.  Callers must prove alignment, bounds, aliasing,
and synchronization before using these helpers.

The two supported projection shapes are composed from native M16N8K16 BF16
instructions:

* M16N8 accommodates up to four pairs of affine factors.
* M16N16 accommodates up to eight pairs and permits role-major placement.

Coefficient placement is described by three strides.  This keeps the device
code independent of a policy choice between interleaved
``[target, source, role]`` and role-major ``[role, target, source]`` storage.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op

from .affine_recurrence_primitives import _ldmatrix
from .affine_recurrence_primitives import ldmatrix_x2_trans
from .affine_recurrence_primitives import ldmatrix_x4_trans
from .affine_recurrence_primitives import mma_m16n8k16_bf16
from .affine_recurrence_primitives import pack_bf16x2
from .affine_recurrence_primitives import store_u32x4_if_valid as _store_u32x4_if_valid
from .affine_recurrence_primitives import vec8_bf16

MMA_M = 16
MMA_N = 8
MMA_K = 16
DIRECT_FACTOR_COLUMNS_M16N8 = MMA_N
DIRECT_FACTOR_COLUMNS_M16N16 = 2 * MMA_N
DIRECT_STATE_SEGMENT_FEATURES = 64


def _swizzle_128b_b16(logical_element):
    """Apply CUTLASS's 128-byte XOR swizzle to one 16-bit element index."""

    byte_offset = logical_element * 2
    return (byte_offset ^ (((byte_offset >> 7) & 7) << 4)) // 2


def direct_factor_index(feature, column, column_extent):
    """S128 index of logical ``factor[feature, column]``."""

    return _swizzle_128b_b16(feature * column_extent + column)


def direct_state_index(row, feature, row_extent):
    """S128 index for a row-major state split into 64-feature segments."""

    segment = feature // DIRECT_STATE_SEGMENT_FEATURES
    local_feature = feature - segment * DIRECT_STATE_SEGMENT_FEATURES
    logical = row * DIRECT_STATE_SEGMENT_FEATURES + local_feature
    return segment * row_extent * DIRECT_STATE_SEGMENT_FEATURES + _swizzle_128b_b16(
        logical
    )


def direct_factor_column(
    step,
    role,
    target_stride,
    role_stride,
):
    """Map a logical ``(step, role)`` to a factor-image column."""

    return step * target_stride + role * role_stride


def direct_coefficient_index(
    target,
    source,
    role,
    row_stride,
    source_stride,
    role_stride,
):
    """Map a logical coefficient to an arbitrary affine table layout."""

    return target * row_stride + source * source_stride + role * role_stride


def direct_factor_b_fragment_index(lane, feature_block, column_extent):
    """S128 stage address for an MMA B fragment."""

    feature = feature_block * MMA_K + lane % MMA_K
    column = lane // MMA_K * MMA_N if column_extent == 2 * MMA_N else 0
    return direct_factor_index(feature, column, column_extent)


def direct_state_a_fragment_index(lane, row_base, feature_block, row_extent):
    """S128 state-stage address for one M16K16 MMA A fragment."""

    row = row_base + lane % MMA_M
    feature = feature_block * MMA_K + lane // MMA_M * 8
    return direct_state_index(row, feature, row_extent)


@dsl_user_op
def _ldmatrix_x4(smem_ptr, *, loc=None, ip=None):
    """Load the four b16 matrices forming one M16K16 A fragment."""

    return _ldmatrix(".x4", "", smem_ptr, 4, loc=loc, ip=ip)


def _vector_index(step, feature, feature_extent):
    return step * feature_extent + feature


def _precompute_feature(lane, element, feature_extent):
    return lane * (feature_extent // 32) + element


@cute.jit
def precompute_affine_from_buffers_bf16(
    prediction_smem,
    observation_smem,
    diagonal_smem,
    update_smem,
    update_scale_smem,
    factor_smem,
    coefficient_smem,
    lane,
    target,
    STEP_COUNT: cutlass.Constexpr[int],
    FEATURE_EXTENT: cutlass.Constexpr[int],
    FACTOR_COLUMN_EXTENT: cutlass.Constexpr[int],
    FACTOR_TARGET_STRIDE: cutlass.Constexpr[int],
    FACTOR_ROLE_STRIDE: cutlass.Constexpr[int],
    COEFFICIENT_ROW_STRIDE: cutlass.Constexpr[int],
    COEFFICIENT_SOURCE_STRIDE: cutlass.Constexpr[int],
    COEFFICIENT_ROLE_STRIDE: cutlass.Constexpr[int],
):
    """Build one affine factor pair and its coefficient row.

    All semantic inputs have already been materialized into step-major shared
    buffers.  The six physical strides describe either interleaved or
    role-major factor and coefficient layouts without a shape-specific branch.
    The caller must invoke this helper with a full active warp and a
    warp-uniform ``target`` because both dot products use warp reductions.
    """

    values_per_lane = FEATURE_EXTENT // 32
    prediction = cute.make_rmem_tensor(values_per_lane, cutlass.Float32)
    observation = cute.make_rmem_tensor(values_per_lane, cutlass.Float32)
    suffix = cute.make_rmem_tensor(values_per_lane, cutlass.Float32)
    prediction_column = direct_factor_column(
        target,
        0,
        FACTOR_TARGET_STRIDE,
        FACTOR_ROLE_STRIDE,
    )
    observation_column = direct_factor_column(
        target,
        1,
        FACTOR_TARGET_STRIDE,
        FACTOR_ROLE_STRIDE,
    )

    for element in cutlass.range_constexpr(values_per_lane):
        feature = _precompute_feature(lane, element, FEATURE_EXTENT)
        target_index = _vector_index(target, feature, FEATURE_EXTENT)
        prediction[element] = cutlass.Float32(prediction_smem[target_index])
        observation[element] = cutlass.Float32(observation_smem[target_index])
        prefix = cutlass.Float32(1.0)
        for diagonal_index in cutlass.range_constexpr(STEP_COUNT):
            if diagonal_index <= target:
                prefix *= cutlass.Float32(
                    diagonal_smem[
                        _vector_index(diagonal_index, feature, FEATURE_EXTENT)
                    ]
                )
        factor_smem[
            direct_factor_index(feature, prediction_column, FACTOR_COLUMN_EXTENT)
        ] = (prefix * prediction[element]).to(cutlass.BFloat16)
        factor_smem[
            direct_factor_index(feature, observation_column, FACTOR_COLUMN_EXTENT)
        ] = (prefix * observation[element]).to(cutlass.BFloat16)
        suffix[element] = cutlass.Float32(1.0)

    for source_offset in cutlass.range_constexpr(STEP_COUNT):
        source = target - source_offset
        if source >= 0:
            prediction_dot = cutlass.Float32(0.0)
            observation_dot = cutlass.Float32(0.0)
            for element in cutlass.range_constexpr(values_per_lane):
                feature = _precompute_feature(lane, element, FEATURE_EXTENT)
                update = cutlass.Float32(
                    update_smem[_vector_index(source, feature, FEATURE_EXTENT)]
                )
                weighted_update = update * suffix[element]
                prediction_dot += prediction[element] * weighted_update
                observation_dot += observation[element] * weighted_update
            prediction_dot = cute.arch.warp_reduction_sum(
                prediction_dot,
                threads_in_group=32,
            )
            observation_dot = cute.arch.warp_reduction_sum(
                observation_dot,
                threads_in_group=32,
            )
            if lane == 0:
                update_scale = cutlass.Float32(update_scale_smem[source])
                if source < target:
                    coefficient_smem[
                        direct_coefficient_index(
                            target,
                            source,
                            0,
                            COEFFICIENT_ROW_STRIDE,
                            COEFFICIENT_SOURCE_STRIDE,
                            COEFFICIENT_ROLE_STRIDE,
                        )
                    ] = update_scale * prediction_dot
                coefficient_smem[
                    direct_coefficient_index(
                        target,
                        source,
                        1,
                        COEFFICIENT_ROW_STRIDE,
                        COEFFICIENT_SOURCE_STRIDE,
                        COEFFICIENT_ROLE_STRIDE,
                    )
                ] = update_scale * observation_dot
            if source > 0:
                for element in cutlass.range_constexpr(values_per_lane):
                    feature = _precompute_feature(lane, element, FEATURE_EXTENT)
                    suffix[element] *= cutlass.Float32(
                        diagonal_smem[_vector_index(source, feature, FEATURE_EXTENT)]
                    )


@cute.jit
def stage_state_tile8x8_async_bf16(
    source_ptr,
    source_base_index,
    source_row_stride,
    state_smem,
    state_row_base,
    state_feature,
    valid,
    ROW_EXTENT: cutlass.Constexpr[int],
):
    """Issue one lane's eight aligned state copies without waiting.

    Invalid sources use ``cp_size=0`` zero fill.  The caller owns the wait and
    the CTA barrier, which permits independent work to overlap the copies.
    """

    aligned_source = cute.make_ptr(
        source_ptr.dtype,
        source_ptr.toint(),
        source_ptr.memspace,
        assumed_align=16,
    )
    aligned_state = cute.make_ptr(
        state_smem.dtype,
        state_smem.toint(),
        state_smem.memspace,
        assumed_align=16,
    )
    copy_size = cutlass.Int32(0)
    if valid:
        copy_size = cutlass.Int32(16)
    for row_offset in cutlass.range_constexpr(8):
        source_index = cutlass.Int32(source_base_index) + cutlass.Int32(
            row_offset
        ) * cutlass.Int32(source_row_stride)
        state_index = direct_state_index(
            state_row_base + cutlass.Int32(row_offset),
            state_feature,
            ROW_EXTENT,
        )
        cute.arch.cp_async_shared_global(
            aligned_state + cute.assume(cutlass.Int32(state_index), divby=8),
            aligned_source + source_index,
            16,
            "cg",
            cp_size=copy_size,
        )
    cute.arch.cp_async_commit_group()


@cute.jit
def _retain_state_history_bf16(
    state_smem,
    lane,
    row_base,
    ROW_EXTENT: cutlass.Constexpr[int],
):
    """Reload one lane's 8x8 row-owned state tile as FP32 history."""

    history = cute.make_rmem_tensor(64, cutlass.Float32)
    half_warp = lane // 16
    lane_in_half = lane % 16
    aligned_state = cute.make_ptr(
        state_smem.dtype,
        state_smem.toint(),
        state_smem.memspace,
        assumed_align=16,
    )
    for row_offset in cutlass.range_constexpr(8):
        state_index = direct_state_index(
            row_base + half_warp * 8 + row_offset,
            lane_in_half * 8,
            ROW_EXTENT,
        )
        state_values = vec8_bf16(aligned_state, state_index)
        for element in cutlass.range_constexpr(8):
            history[row_offset * 8 + element] = cutlass.Float32(state_values[element])
    return history


@cute.jit
def _project_m16n8_bf16(
    state_smem,
    factor_smem,
    lane,
    row_base,
    ROW_EXTENT: cutlass.Constexpr[int],
    FEATURE_EXTENT: cutlass.Constexpr[int],
):
    accumulator = (
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
    )
    for feature_block in cutlass.range_constexpr(FEATURE_EXTENT // MMA_K):
        state_fragment = _ldmatrix_x4(
            state_smem
            + direct_state_a_fragment_index(
                lane,
                row_base,
                feature_block,
                ROW_EXTENT,
            )
        )
        factor_fragment = ldmatrix_x2_trans(
            factor_smem
            + direct_factor_b_fragment_index(
                lane,
                feature_block,
                DIRECT_FACTOR_COLUMNS_M16N8,
            )
        )
        accumulator = mma_m16n8k16_bf16(
            state_fragment[0],
            state_fragment[1],
            state_fragment[2],
            state_fragment[3],
            factor_fragment[0],
            factor_fragment[1],
            accumulator[0],
            accumulator[1],
            accumulator[2],
            accumulator[3],
        )
    return accumulator


@cute.jit
def project_retain_affine_m16n8_bf16(
    state_smem,
    factor_smem,
    lane,
    row_base,
    ROW_EXTENT: cutlass.Constexpr[int],
    FEATURE_EXTENT: cutlass.Constexpr[int],
):
    """Project one D128 state tile onto eight columns and retain its FP32 rows."""

    accumulator = _project_m16n8_bf16(
        state_smem,
        factor_smem,
        lane,
        row_base,
        ROW_EXTENT,
        FEATURE_EXTENT,
    )
    history = _retain_state_history_bf16(
        state_smem,
        lane,
        row_base,
        ROW_EXTENT,
    )
    return accumulator, history


@cute.jit
def project_retain_affine_m16n16_bf16(
    state_smem,
    factor_smem,
    lane,
    row_base,
    ROW_EXTENT: cutlass.Constexpr[int],
    FEATURE_EXTENT: cutlass.Constexpr[int],
):
    """Project one D128 state tile onto sixteen columns and retain FP32 rows."""

    accumulator = (
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
    )
    for feature_block in cutlass.range_constexpr(FEATURE_EXTENT // MMA_K):
        state_fragment = _ldmatrix_x4(
            state_smem
            + direct_state_a_fragment_index(
                lane,
                row_base,
                feature_block,
                ROW_EXTENT,
            )
        )
        factor_fragment = ldmatrix_x4_trans(
            factor_smem
            + direct_factor_b_fragment_index(
                lane,
                feature_block,
                DIRECT_FACTOR_COLUMNS_M16N16,
            )
        )
        low = mma_m16n8k16_bf16(
            state_fragment[0],
            state_fragment[1],
            state_fragment[2],
            state_fragment[3],
            factor_fragment[0],
            factor_fragment[1],
            accumulator[0],
            accumulator[1],
            accumulator[2],
            accumulator[3],
        )
        high = mma_m16n8k16_bf16(
            state_fragment[0],
            state_fragment[1],
            state_fragment[2],
            state_fragment[3],
            factor_fragment[2],
            factor_fragment[3],
            accumulator[4],
            accumulator[5],
            accumulator[6],
            accumulator[7],
        )
        accumulator = (*low, *high)
    history = _retain_state_history_bf16(
        state_smem,
        lane,
        row_base,
        ROW_EXTENT,
    )
    return accumulator, history


def _projection_value(accumulator, lane, column, high_row):
    """Broadcast one accumulator cell within its four-lane row group."""

    n_block = column // MMA_N
    local_column = column - n_block * MMA_N
    source_lane = lane - lane % 4 + local_column // 2
    source_slot = n_block * 4 + local_column % 2 + high_row * 2
    return cutlass.Float32(
        cute.arch.shuffle_sync(
            accumulator[source_slot],
            source_lane,
            mask=0xFFFFFFFF,
        )
    )


@cute.jit
def consume_affine_steps(
    projection_accumulator,
    row_input_low,
    row_input_high,
    row_residual_smem,
    coefficient_smem,
    lane,
    row_base,
    prediction_projection_columns,
    observation_projection_columns,
    STEP_COUNT: cutlass.Constexpr[int],
    ROW_EXTENT: cutlass.Constexpr[int],
    COEFFICIENT_ROW_STRIDE: cutlass.Constexpr[int],
    COEFFICIENT_SOURCE_STRIDE: cutlass.Constexpr[int],
    COEFFICIENT_ROLE_STRIDE: cutlass.Constexpr[int],
):
    """Solve a triangular affine system and form all row observations.

    Projection columns and coefficient strides are independent physical-plan
    inputs.  Residuals remain unscaled because coefficients already include
    each source update scale.
    """

    lane_quad = lane % 4
    row_low = row_base + lane // 4
    row_high = row_low + 8
    residual_low = cute.make_rmem_tensor(STEP_COUNT, cutlass.Float32)
    residual_high = cute.make_rmem_tensor(STEP_COUNT, cutlass.Float32)
    for target in cutlass.range_constexpr(STEP_COUNT):
        prediction_low = _projection_value(
            projection_accumulator,
            lane,
            prediction_projection_columns[target],
            0,
        )
        prediction_high = _projection_value(
            projection_accumulator,
            lane,
            prediction_projection_columns[target],
            1,
        )
        residual_low[target] = cutlass.Float32(0.0)
        residual_high[target] = cutlass.Float32(0.0)
        if lane_quad == 0:
            solved_low = cutlass.Float32(row_input_low[target]) - prediction_low
            solved_high = cutlass.Float32(row_input_high[target]) - prediction_high
            for source in cutlass.range_constexpr(target):
                coefficient = cutlass.Float32(
                    coefficient_smem[
                        direct_coefficient_index(
                            target,
                            source,
                            0,
                            COEFFICIENT_ROW_STRIDE,
                            COEFFICIENT_SOURCE_STRIDE,
                            COEFFICIENT_ROLE_STRIDE,
                        )
                    ]
                )
                solved_low -= coefficient * residual_low[source]
                solved_high -= coefficient * residual_high[source]
            residual_low[target] = solved_low
            residual_high[target] = solved_high
            row_residual_smem[target * ROW_EXTENT + row_low] = solved_low
            row_residual_smem[target * ROW_EXTENT + row_high] = solved_high

    cute.arch.sync_warp()
    quad_source = lane - lane_quad
    output_low = cute.make_rmem_tensor(STEP_COUNT, cutlass.Float32)
    output_high = cute.make_rmem_tensor(STEP_COUNT, cutlass.Float32)
    for target in cutlass.range_constexpr(STEP_COUNT):
        residual_low[target] = cutlass.Float32(
            cute.arch.shuffle_sync(
                residual_low[target],
                quad_source,
                mask=0xFFFFFFFF,
            )
        )
        residual_high[target] = cutlass.Float32(
            cute.arch.shuffle_sync(
                residual_high[target],
                quad_source,
                mask=0xFFFFFFFF,
            )
        )
        value_low = _projection_value(
            projection_accumulator,
            lane,
            observation_projection_columns[target],
            0,
        )
        value_high = _projection_value(
            projection_accumulator,
            lane,
            observation_projection_columns[target],
            1,
        )
        for source in cutlass.range_constexpr(target + 1):
            coefficient = cutlass.Float32(
                coefficient_smem[
                    direct_coefficient_index(
                        target,
                        source,
                        1,
                        COEFFICIENT_ROW_STRIDE,
                        COEFFICIENT_SOURCE_STRIDE,
                        COEFFICIENT_ROLE_STRIDE,
                    )
                ]
            )
            value_low += coefficient * residual_low[source]
            value_high += coefficient * residual_high[source]
        output_low[target] = value_low
        output_high[target] = value_high
    return output_low, output_high


@cute.jit
def load_checkpoint_factors_bf16(
    diagonal_smem,
    update_smem,
    lane,
    target,
    FEATURE_EXTENT: cutlass.Constexpr[int],
):
    """Load one lane's eight D128 affine checkpoint factors once per step."""

    feature_base = (lane % 16) * 8
    diagonal_values = cute.make_rmem_tensor(8, cutlass.Float32)
    update_values = cute.make_rmem_tensor(8, cutlass.Float32)
    for element in cutlass.range_constexpr(8):
        vector_index = target * FEATURE_EXTENT + feature_base + element
        diagonal_values[element] = cutlass.Float32(diagonal_smem[vector_index])
        update_values[element] = cutlass.Float32(update_smem[vector_index])
    return diagonal_values, update_values


@cute.jit
def checkpoint_affine_row_bf16(
    history,
    row_residual_smem,
    diagonal_values,
    update_values,
    update_scale_smem,
    lane,
    row_base,
    target,
    row_offset,
    ROW_EXTENT: cutlass.Constexpr[int],
):
    """Advance one retained FP32 row and return four packed BF16 pairs."""

    packed_row = cute.make_rmem_tensor(4, cutlass.Uint32)
    half_warp = lane // 16
    row = row_base + half_warp * 8 + row_offset
    scaled_residual = cutlass.Float32(
        row_residual_smem[target * ROW_EXTENT + row]
    ) * cutlass.Float32(update_scale_smem[target])
    for pair in cutlass.range_constexpr(4):
        element = pair * 2
        history_index = row_offset * 8 + element
        low = (
            history[history_index] * diagonal_values[element]
            + scaled_residual * update_values[element]
        )
        high = (
            history[history_index + 1] * diagonal_values[element + 1]
            + scaled_residual * update_values[element + 1]
        )
        history[history_index] = low
        history[history_index + 1] = high
        packed_row[pair] = pack_bf16x2(low, high)
    return history, packed_row


@cute.jit
def store_packed_b16x8_if_valid(
    pointer,
    value0,
    value1,
    value2,
    value3,
    slot,
    slot_extent,
):
    """Store four packed words when ``slot`` is in range."""

    _store_u32x4_if_valid(
        pointer,
        value0,
        value1,
        value2,
        value3,
        slot,
        slot_extent,
    )


__all__ = [
    "DIRECT_FACTOR_COLUMNS_M16N8",
    "DIRECT_FACTOR_COLUMNS_M16N16",
    "DIRECT_STATE_SEGMENT_FEATURES",
    "MMA_K",
    "MMA_M",
    "MMA_N",
    "checkpoint_affine_row_bf16",
    "consume_affine_steps",
    "direct_coefficient_index",
    "direct_factor_b_fragment_index",
    "direct_factor_column",
    "direct_factor_index",
    "direct_state_a_fragment_index",
    "direct_state_index",
    "load_checkpoint_factors_bf16",
    "precompute_affine_from_buffers_bf16",
    "project_retain_affine_m16n8_bf16",
    "project_retain_affine_m16n16_bf16",
    "stage_state_tile8x8_async_bf16",
    "store_packed_b16x8_if_valid",
]
