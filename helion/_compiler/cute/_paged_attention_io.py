"""Address the proved packed or folded query/output layout of attention."""

from __future__ import annotations

from typing import cast

import cutlass
import cutlass.cute as cute

from .device.ptx import l2_policy_evict_last
from .device.ptx import stg_bf16_l2
from .device.ptx import stg_cs_bf16
from .device.ptx import stg_cs_f32
from .device.ptx import stg_f32_l2


def folded_offset(
    strides: tuple[int, ...],
    item: cutlass.Int32,
    column: cutlass.Int32,
    heads: int,
    group: int,
    row: cutlass.Int32 | None = None,
) -> cutlass.Int64:
    """The index of element ``(item, row, column)`` in the proved folded
    layout of the query or the output, by its storage strides: request
    ``item // heads``, query ``row // group``, head ``item % heads * group +
    row % group`` and the column, each times its stride.

    Without ``row``, the index of the item's row 0, which every row of the
    item shares; ``folded_row_offset`` is what a row adds to it. The one
    statement of that layout: the query load, the plain output store and the
    stores under a cache policy all address through it.
    """
    request = cutlass.Int64(item // heads) * strides[0]
    if row is None:
        return (
            request
            + cutlass.Int64(item % heads * group) * strides[2]
            + cutlass.Int64(column) * strides[3]
        )
    return (
        request
        + cutlass.Int64(row // group) * strides[1]
        + cutlass.Int64(item % heads * group + row % group) * strides[2]
        + cutlass.Int64(column) * strides[3]
    )


def folded_row_offset(strides: tuple[int, ...], row: int, group: int) -> int:
    """What row ``row`` of an item, known when the kernel is traced, adds
    to the index of the item's row 0 (``folded_offset`` without a row)."""
    return (row // group) * strides[1] + (row % group) * strides[2]


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
        result = query[folded_offset(layouts[0], item, column, heads, group, row)]
    return cast("cutlass.BFloat16", result)


@cute.jit
def store_output_element(
    output: cute.Tensor,
    item: cutlass.Int32,
    row: cutlass.Int32,
    column: cutlass.Int32,
    value: cutlass.Numeric,
    heads: cutlass.Constexpr,
    group: cutlass.Constexpr,
    layouts: cutlass.Constexpr,
) -> None:
    if cutlass.const_expr(layouts is None):
        output[item, row, column] = value
    else:
        output[folded_offset(layouts[1], item, column, heads, group, row)] = value


@cute.jit
def store_output_rows(
    output: cute.Tensor,
    item: cutlass.Int32,
    column: cutlass.Int32,
    values: tuple,
    heads: cutlass.Constexpr,
    group: cutlass.Constexpr,
    layouts: cutlass.Constexpr,
    policy: cutlass.Constexpr,
) -> None:
    """Store ``values[row]`` to the element ``store_output_element`` stores
    for ``(item, row, column)``, each row of the tuple, under ``policy``, a
    cache policy of ``cute_store_policy``:

    - ``streaming``: the streaming cache operator (``st.global.cs``,
      evict-first), for an output that is written once and read by no CTA
      of the launch;
    - ``l2_evict_last``: the L2 evict-last cache hint
      (``st.global.L2::cache_hint``), whose policy word the calling thread
      makes once, ahead of its stores (``createpolicy``, as
      ``_paged_attention_partition.store_part`` does for the partition's
      workspace). The lines an output's first write brings into L2 are then
      the last ones the K/V stream of the following calls displaces. The
      word is made at run time, so this form does not need the fixed
      descriptor of the packet flushes' ``l2_evict_last``.

    The addresses are those elements': the same index, in the tensor's own
    strided layout or by the proved storage strides, times the element's
    size from the tensor's first element. The index is affine in the row,
    so the part every row shares is formed once. A value is the fp32 value
    ahead of the output's conversion. A BF16 output rounds it to nearest
    even inside the store's instruction pair, which is what the DSL's cast
    to BF16 ahead of a plain store does (``cvt.rn.bf16.f32``): the bytes
    written are the plain store's. An fp32 output stores it as it is.
    Count and order of the stores are the plain ones' as well: either
    policy is a hint about the lines, which a device may ignore.
    """
    assert cutlass.const_expr(policy in ("streaming", "l2_evict_last"))
    assert cutlass.const_expr(
        output.element_type in (cutlass.BFloat16, cutlass.Float32)
    )
    half = output.element_type is cutlass.BFloat16
    streamed = stg_cs_bf16 if half else stg_cs_f32
    hinted = stg_bf16_l2 if half else stg_f32_l2
    width = 2 if half else 4
    address = cutlass.Int64(cast("cute.Pointer", output.iterator).toint())
    word = None
    if cutlass.const_expr(policy == "l2_evict_last"):
        word = l2_policy_evict_last()
    # The item's first element and what a row adds to it: the output's own
    # row stride, or the folded layout's strides.
    step = strides = None
    if cutlass.const_expr(layouts is None):
        first = cutlass.Int64(cute.crd2idx((item, 0, column), output.layout))
        step = cutlass.Int64(cute.crd2idx((0, 1, 0), output.layout))
    else:
        strides = layouts[1]
        first = folded_offset(strides, item, column, heads, group)
    for row in cutlass.range_constexpr(len(values)):
        if cutlass.const_expr(strides is None):
            assert step is not None
            target = address + (first + step * row) * width
        else:
            assert strides is not None
            folded = folded_row_offset(strides, row, group)
            target = address + (first + folded) * width
        if cutlass.const_expr(word is None):
            streamed(target, values[row])
        else:
            hinted(target, values[row], word)


@cute.jit
def store_row_element(
    output: cute.Tensor,
    item: cutlass.Int32,
    row: cutlass.Int32,
    value: cutlass.Float32,
    heads: cutlass.Constexpr,
    group: cutlass.Constexpr,
    strides: cutlass.Constexpr,
) -> None:
    if cutlass.const_expr(strides is None):
        output[item, row] = value
    else:
        offset = (
            cutlass.Int64(item // heads) * strides[0]
            + cutlass.Int64(row // group) * strides[1]
            + cutlass.Int64(item % heads * group + row % group) * strides[2]
        )
        output[offset] = value
