"""Address the proved packed or folded query/output layout of attention."""

from __future__ import annotations

from typing import cast

import cutlass
import cutlass.cute as cute


@cute.jit
def load_query_element(
    query: cute.Tensor,
    item: cutlass.Int32,
    row: cutlass.Int32,
    column: cutlass.Int32,
    heads: cutlass.Constexpr,
    group: cutlass.Constexpr,
    layouts: cutlass.Constexpr,
) -> cutlass.BFloat16:
    if cutlass.const_expr(layouts is None):
        result = query[item, row, column]
    else:
        strides = layouts[0]
        offset = (
            cutlass.Int64(item // heads) * strides[0]
            + cutlass.Int64(row // group) * strides[1]
            + cutlass.Int64(item % heads * group + row % group) * strides[2]
            + cutlass.Int64(column) * strides[3]
        )
        result = query[offset]
    return cast("cutlass.BFloat16", result)


@cute.jit
def store_output_element(
    output: cute.Tensor,
    item: cutlass.Int32,
    row: cutlass.Int32,
    column: cutlass.Int32,
    value: cutlass.BFloat16,
    heads: cutlass.Constexpr,
    group: cutlass.Constexpr,
    layouts: cutlass.Constexpr,
) -> None:
    if cutlass.const_expr(layouts is None):
        output[item, row, column] = value
    else:
        strides = layouts[1]
        offset = (
            cutlass.Int64(item // heads) * strides[0]
            + cutlass.Int64(row // group) * strides[1]
            + cutlass.Int64(item % heads * group + row % group) * strides[2]
            + cutlass.Int64(column) * strides[3]
        )
        output[offset] = value


@cute.jit
def store_lse_element(
    lse: cute.Tensor,
    item: cutlass.Int32,
    row: cutlass.Int32,
    value: cutlass.Float32,
    heads: cutlass.Constexpr,
    group: cutlass.Constexpr,
    layouts: cutlass.Constexpr,
) -> None:
    if cutlass.const_expr(layouts is None):
        lse[item, row] = value
    else:
        strides = layouts[2]
        offset = (
            cutlass.Int64(item // heads) * strides[0]
            + cutlass.Int64(row // group) * strides[1]
            + cutlass.Int64(item % heads * group + row % group) * strides[2]
        )
        lse[offset] = value
