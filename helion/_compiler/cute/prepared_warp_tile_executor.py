# ruff: noqa: ANN001, ANN202
"""One runtime tile body around the common original ordered-K executor."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from .affine_recurrence_primitives import pack_typed_b16x2_to_i32
from .prepared_warp_contraction import execute_prepared_warp_k
from .warp_specialized_primitives import named_barrier_sync
from .warp_specialized_primitives import shared_pointer
from .warp_specialized_primitives import swizzle_b16_index


@cute.jit
def prepared_warp_tile_join(team, PROGRAM: cutlass.Constexpr):
    barrier_base, participants = PROGRAM[5]
    named_barrier_sync(barrier_base + team, participants)


@cute.jit
def publish_prepared_warp_tile(
    smem_base,
    stage_bytes,
    row_base,
    col_base,
    lane,
    PORT: cutlass.Constexpr,
    STORE: cutlass.Constexpr,
    PUBLICATION: cutlass.Constexpr,
):
    lhs_offset, rhs_offset, k_steps, n_atoms, row_extent = PORT
    zero = cutlass.Float32(0.0)
    acc = (zero, zero, zero, zero, zero, zero, zero, zero)
    if row_base >= col_base:
        acc = execute_prepared_warp_k(
            (smem_base, lhs_offset + stage_bytes, lane, row_base, row_extent),
            (smem_base, rhs_offset + stage_bytes, lane, col_base, row_extent),
            acc,
            None,
            k_steps,
            True,
            n_atoms,
            B_TRANSPOSED_PACKED=True,
        )
    packed = []
    for pair in cutlass.range_constexpr(4):
        index = pair * 2
        row = row_base + lane // 4 + ((index % 4) // 2) * 8
        col = col_base + (index // 4) * 8 + (lane % 4) * 2
        low = PUBLICATION(acc[index], row, col)
        high = PUBLICATION(acc[index + 1], row, col + 1)
        packed.append(pack_typed_b16x2_to_i32(low, high, cutlass.BFloat16))
    offset, origin, row_bytes, columns_per_tile, rows_per_tile, phase_mask = STORE
    for pair in cutlass.range_constexpr(2):
        row = col_base + pair * 8 + lane % 8
        col = origin + row_base + (lane // 8) * 8
        byte = swizzle_b16_index(
            row, col, row_bytes, columns_per_tile, rows_per_tile, phase_mask
        )
        ptr = shared_pointer(smem_base, offset + stage_bytes + byte, cutlass.BFloat16)
        prims.stmatrix(
            ptr,
            [packed[pair * 2], packed[pair * 2 + 1]],
            prims.MMALayout.COL,
            shape=prims.StoreShape.M8N8,
        )


@cute.jit
def _warp_tile_count(
    local_warp,
    COUNTS: cutlass.Constexpr,
    OWNER: cutlass.Constexpr = 0,
):
    result = cutlass.Int32(0)
    if cutlass.const_expr(len(COUNTS) == OWNER):
        result = cutlass.Int32(0)
    elif cutlass.const_expr(COUNTS[OWNER] == 0):
        result = _warp_tile_count(local_warp, COUNTS, OWNER + 1)
    elif local_warp == OWNER:
        result = cutlass.Int32(COUNTS[OWNER])
    else:
        result = _warp_tile_count(local_warp, COUNTS, OWNER + 1)
    return result


@cute.jit
def execute_prepared_warp_tiles(
    smem_base,
    stage_bytes,
    local_warp,
    lane,
    PROGRAM: cutlass.Constexpr,
    PUBLICATION: cutlass.Constexpr,
):
    counts, default_axes, overrides, operand, store, join, axis_program = PROGRAM
    axes = []
    for index in cutlass.range_constexpr(len(axis_program)):
        divisor, modulus, scale = axis_program[index]
        value = local_warp // divisor
        if cutlass.const_expr(modulus is not None):
            value = value % modulus
        axes.append(value * scale)
    tile_count = cutlass.Int32(counts[0])
    if cutlass.const_expr(counts != (counts[0],) * len(counts)):
        tile_count = _warp_tile_count(local_warp, counts)
    for tile in cutlass.range(tile_count, unroll=1):
        row = axes[default_axes[0]]
        column = axes[default_axes[1]]
        for group in cutlass.range_constexpr(len(overrides)):
            owner, selections = overrides[group]
            if local_warp == owner:
                for selection in cutlass.range_constexpr(len(selections)):
                    ordinal, row_axis, column_axis = selections[selection]
                    if tile == ordinal:
                        row = axes[row_axis]
                        column = axes[column_axis]
        publish_prepared_warp_tile(
            smem_base,
            stage_bytes,
            row,
            column,
            lane,
            operand,
            store,
            PUBLICATION,
        )
