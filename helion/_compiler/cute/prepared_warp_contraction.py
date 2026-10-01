# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CuTe compile-time values intentionally have implicit types.
# ruff: noqa: ANN001, ANN202

"""One ordered K executor for already prepared native warp operands.

The two closed operand representations preserve the original instruction
backends. Neither leaf owns a K loop, shared publication, or synchronization.
Callers retain all graph, lifetime, team and input-completion authority.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.experimental.primitives as prims

from .affine_recurrence_primitives import _ldmatrix
from .affine_recurrence_primitives import ldmatrix_x2_trans
from .affine_recurrence_primitives import ldmatrix_x4_trans
from .affine_recurrence_primitives import mma_m16n8k16_bf16
from .affine_recurrence_primitives import mma_m16n8k16_f16
from .affine_recurrence_primitives import movmatrix_b16
from .warp_specialized_primitives import segmented_swizzle_b16_element_index
from .warp_specialized_primitives import shared_pointer
from .warp_specialized_primitives import swizzle_b16_element_index
from .warp_specialized_primitives import swizzle_b16_index


@dsl_user_op
def _load_x4(pointer, *, loc=None, ip=None):
    return _ldmatrix(".x4", "", pointer, 4, loc=loc, ip=ip)


@cute.jit
def _load_a(
    source,
    k,
    PACKED: cutlass.Constexpr[bool],
    B_TRANSPOSED_PACKED: cutlass.Constexpr[bool] = False,
):
    if cutlass.const_expr(PACKED):
        if cutlass.const_expr(B_TRANSPOSED_PACKED):
            base, offset, lane, row_base, row_extent = source
            row = row_base + lane % 16
            feature = k * 16 + lane // 16 * 8
            byte = swizzle_b16_index(row, feature, 128, 64, row_extent, 7)
            result = prims.ldmatrix(
                shared_pointer(base, offset + byte, cutlass.BFloat16),
                4,
                prims.MMALayout.ROW,
            )
        else:
            # (original shared pointer, lane, row origin, complete row extent)
            pointer, lane, row_base, row_extent = source
            row = row_base + lane % 16
            feature = k * 16 + lane // 16 * 8
            index = segmented_swizzle_b16_element_index(
                row, feature, row_extent, 64, 8, 7
            )
            result = _load_x4(pointer + index)
    else:
        # (original CopyAtom/partition/retile, original MMA fragment)
        copy, shared, registers, fragment = source
        cute.copy(copy, shared[None, None, k], registers[None, None, 0])
        result = fragment[None, None, 0]
    return result


@cute.jit
def _load_b(
    source,
    k,
    PACKED: cutlass.Constexpr[bool],
    N_ATOMS: cutlass.Constexpr[int],
    B_TRANSPOSED_PACKED: cutlass.Constexpr[bool] = False,
):
    if cutlass.const_expr(PACKED):
        if cutlass.const_expr(B_TRANSPOSED_PACKED):
            # Native byte-addressed N-by-K image; preserve the original address
            # expression and ldmatrix backend as well as its logical mapping.
            base, offset, lane, column_base, column_extent = source
            row = column_base + lane // 16 * 8 + lane % 8
            feature = k * 16 + (lane // 8) % 2 * 8
            byte = swizzle_b16_index(row, feature, 128, 64, column_extent, 7)
            result = prims.ldmatrix(
                shared_pointer(base, offset + byte, cutlass.BFloat16),
                4,
                prims.MMALayout.ROW,
            )
        else:
            pointer, lane = source
            feature = k * 16 + lane % 16
            column = lane // 16 * 8 if cutlass.const_expr(N_ATOMS == 2) else 0
            index = swizzle_b16_element_index(feature * (N_ATOMS * 8) + column, 7)
            if cutlass.const_expr(N_ATOMS == 2):
                result = ldmatrix_x4_trans(pointer + index)
            else:
                result = ldmatrix_x2_trans(pointer + index)
    else:
        copy, shared, registers, fragment = source
        cute.copy(copy, shared[None, None, k], registers[None, None, 0])
        result = fragment[None, None, 0]
    return result


@cute.jit
def _packed_atom(a, b, c, DTYPE_KIND: cutlass.Constexpr[int]):
    if cutlass.const_expr(DTYPE_KIND == 1):
        return mma_m16n8k16_f16(
            a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]
        )
    return mma_m16n8k16_bf16(a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3])


@cute.jit
def _issue(
    a,
    b,
    accumulator,
    mma,
    PACKED: cutlass.Constexpr[bool],
    N_ATOMS: cutlass.Constexpr[int],
    DTYPE_KIND: cutlass.Constexpr[int] = 0,
):
    if cutlass.const_expr(PACKED):
        low = _packed_atom(
            a,
            (b[0], b[1]),
            (accumulator[0], accumulator[1], accumulator[2], accumulator[3]),
            DTYPE_KIND,
        )
        if cutlass.const_expr(N_ATOMS == 2):
            high = _packed_atom(
                a,
                (b[2], b[3]),
                (accumulator[4], accumulator[5], accumulator[6], accumulator[7]),
                DTYPE_KIND,
            )
            accumulator = (*low, *high)
        else:
            accumulator = low
    else:
        cute.gemm(mma, accumulator, a, b, accumulator)
    return accumulator


@cute.jit
def execute_prepared_warp_fragment(a, b, accumulator, PORT: cutlass.Constexpr):
    """One already prepared physical atom; caller owns sparse logical support."""
    dtype_kind, n_atoms = PORT
    assert dtype_kind in (0, 1) and n_atoms in (1, 2)
    return _issue(a, b, accumulator, None, True, n_atoms, dtype_kind)


@cute.jit
def execute_prepared_blockdiag_fragment(a0, a3, b0, b3, PORT: cutlass.Constexpr):
    """Original compressed two8x8 view, including its exact zero padding."""
    zero_i32 = cutlass.Int32(0)
    zero_f32 = cutlass.Float32(0.0)
    return execute_prepared_warp_fragment(
        (a0, zero_i32, zero_i32, a3),
        (movmatrix_b16(b0), movmatrix_b16(b3)),
        (zero_f32, zero_f32, zero_f32, zero_f32),
        PORT,
    )


@cute.jit
def execute_prepared_warp_k(
    a_source,
    b_source,
    accumulator,
    mma,
    K_STEPS: cutlass.Constexpr[int],
    PACKED: cutlass.Constexpr[bool],
    N_ATOMS: cutlass.Constexpr[int],
    B_TRANSPOSED_PACKED: cutlass.Constexpr[bool] = False,
):
    """Increasing K, A then B, original ordered issues, original FP32 result."""
    assert not B_TRANSPOSED_PACKED or (PACKED and N_ATOMS == 2)
    assert K_STEPS > 0
    assert N_ATOMS == 1 or (PACKED and N_ATOMS == 2)
    for k in cutlass.range_constexpr(K_STEPS):
        a = _load_a(a_source, k, PACKED, B_TRANSPOSED_PACKED)
        b = _load_b(b_source, k, PACKED, N_ATOMS, B_TRANSPOSED_PACKED)
        accumulator = _issue(a, b, accumulator, mma, PACKED, N_ATOMS)
    return accumulator
