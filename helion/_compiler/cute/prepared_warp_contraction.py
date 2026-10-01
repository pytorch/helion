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

from .affine_recurrence_primitives import _ldmatrix
from .affine_recurrence_primitives import ldmatrix_x2_trans
from .affine_recurrence_primitives import ldmatrix_x4_trans
from .affine_recurrence_primitives import mma_m16n8k16_bf16
from .warp_specialized_primitives import segmented_swizzle_b16_element_index
from .warp_specialized_primitives import swizzle_b16_element_index


@dsl_user_op
def _load_x4(pointer, *, loc=None, ip=None):
    return _ldmatrix(".x4", "", pointer, 4, loc=loc, ip=ip)


@cute.jit
def _load_a(source, k, PACKED: cutlass.Constexpr[bool]):
    if cutlass.const_expr(PACKED):
        # (original shared pointer, lane, row origin, complete row extent)
        pointer, lane, row_base, row_extent = source
        row = row_base + lane % 16
        feature = k * 16 + lane // 16 * 8
        index = segmented_swizzle_b16_element_index(row, feature, row_extent, 64, 8, 7)
        result = _load_x4(pointer + index)
    else:
        # (original CopyAtom/partition/retile, original MMA fragment)
        copy, shared, registers, fragment = source
        cute.copy(copy, shared[None, None, k], registers[None, None, 0])
        result = fragment[None, None, 0]
    return result


@cute.jit
def _load_b(
    source, k, PACKED: cutlass.Constexpr[bool], N_ATOMS: cutlass.Constexpr[int]
):
    if cutlass.const_expr(PACKED):
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
def _issue(
    a,
    b,
    accumulator,
    mma,
    PACKED: cutlass.Constexpr[bool],
    N_ATOMS: cutlass.Constexpr[int],
):
    if cutlass.const_expr(PACKED):
        low = mma_m16n8k16_bf16(
            a[0],
            a[1],
            a[2],
            a[3],
            b[0],
            b[1],
            accumulator[0],
            accumulator[1],
            accumulator[2],
            accumulator[3],
        )
        if cutlass.const_expr(N_ATOMS == 2):
            high = mma_m16n8k16_bf16(
                a[0],
                a[1],
                a[2],
                a[3],
                b[2],
                b[3],
                accumulator[4],
                accumulator[5],
                accumulator[6],
                accumulator[7],
            )
            accumulator = (*low, *high)
        else:
            accumulator = low
    else:
        cute.gemm(mma, accumulator, a, b, accumulator)
    return accumulator


@cute.jit
def execute_prepared_warp_k(
    a_source,
    b_source,
    accumulator,
    mma,
    K_STEPS: cutlass.Constexpr[int],
    PACKED: cutlass.Constexpr[bool],
    N_ATOMS: cutlass.Constexpr[int],
):
    """Increasing K, A then B, original ordered issues, original FP32 result."""
    assert K_STEPS > 0
    assert N_ATOMS == 1 or (PACKED and N_ATOMS == 2)
    for k in cutlass.range_constexpr(K_STEPS):
        a = _load_a(a_source, k, PACKED)
        b = _load_b(b_source, k, PACKED, N_ATOMS)
        accumulator = _issue(a, b, accumulator, mma, PACKED, N_ATOMS)
    return accumulator
