"""Exact-tree row reductions over the 32 lanes of a complete CUDA warp."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import cast

import cutlass
from cutlass._mlir.dialects import arith
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def _select_float32(
    condition: cutlass.Boolean,
    when_true: cutlass.Float32,
    when_false: cutlass.Float32,
    *,
    loc: object = None,
    ip: object = None,
) -> cutlass.Float32:
    return cutlass.Float32(
        arith.select(
            cutlass.Boolean(condition).ir_value(loc=loc, ip=ip),
            cutlass.Float32(when_true).ir_value(loc=loc, ip=ip),
            cutlass.Float32(when_false).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


def warp_row_sum(
    values: Sequence[cutlass.Float32], lane: cutlass.Int32
) -> tuple[cutlass.Int32, cutlass.Float32]:
    """Sum 1/2/4/8/16/32 FP32 rows with the ascending-XOR butterfly tree.

    Every lane supplies one value per row, and all 32 lanes must participate.
    Each transpose stage halves the live row list while combining the same
    subtrees as separate per-row reductions with XOR masks 1,2,4,8,16. Only
    commutative operand swaps occur; the addition tree is not reassociated.

    Return the owned row and its sum. Lanes below len(values) own distinct rows
    in bit-reversed order; callers must ignore the duplicate results in other
    lanes. The caller handles any cross-warp reduction and synchronization.
    """
    count = len(values)
    assert 1 <= count <= 32 and count & (count - 1) == 0
    row_bits = count.bit_length() - 1
    current = list(values)
    for shift in range(row_bits):
        half = len(current) // 2
        upper = (lane & (1 << shift)) != 0
        next_values = []
        for index in range(half):
            send = _select_float32(upper, current[index], current[index + half])
            keep = _select_float32(upper, current[index + half], current[index])
            next_values.append(keep + cute.arch.shuffle_sync_bfly(send, 1 << shift))
        current = next_values
    total = current[0]
    for shift in range(row_bits, 5):
        total = total + cute.arch.shuffle_sync_bfly(total, 1 << shift)
    # Preserve the lane's scalar type (or its vector shape in CPU emulation).
    row = cast("cutlass.Int32", lane & 0)
    for shift in range(row_bits):
        shifted = cast("cutlass.Int32", lane >> shift)
        bit = cast("cutlass.Int32", shifted & 1)
        reversed_bit = cast("cutlass.Int32", bit << (row_bits - shift - 1))
        row = cast("cutlass.Int32", row | reversed_bit)
    return row, cast("cutlass.Float32", total)
