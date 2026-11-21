"""Boundary L2 prefetch for one-unit CTAs of the paired attention schedule.

A unit is one CTA's work: one request, the two heads of its pair. The device
dispatches a grid's CTAs in index order onto SMs as they free up, so which
unit starts next on an SM, and roughly when, follows from the grid. The
control warp that has no other work on this schedule (no value-metadata
transport, one unit per CTA) requests the first K pages of such a unit into
L2 while its own CTA still runs or has just started
(``paged_attention_prefetch`` has the two forms and when a bind has one).

A request is ``cp.async.bulk.prefetch.L2`` of a byte range: it returns no
data and changes no state a program can observe, so it cannot change a
result or see a predecessor's pending write. What the role does read is
real: the served unit's bounds (the kernel's own scalar prelude evaluated
for that unit, hence its length) and its first page ids. Both are inputs
the launch reads anyway, for that very unit, in tensors the capture proof
lists as loaded, so a launch that may read its inputs ahead of its
dependency wait may make these reads there too. Page ids are taken only
from the served unit's own range, inside the page table, and a request is
formed only for an id inside the page pool.

One request costs the issuing warp about 0.1 us whichever lane issues it:
a warp issues its requests one after another. A page is two requests, the
raw rows of both heads and their packed scales.

The sibling form can also serve in a launch without a head start, to a
depth of its own (``cold_depth`` of ``prefetch_boundary_units``): the same
reads and requests by the same CTAs, where no launch ahead left SMs idle.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from ._mma_dag_tcgen05_runtime import mbar_wait_sleep
from ._paged_attention_compute import logical_attention_key_shift
from ._paged_attention_partial_pipeline import attention_page_coordinates

# Lanes of a control warp that serve a unit each in the sibling form. The
# CTAs on idle SMs have a head start of a few microseconds, which a warp
# spends on about a hundred requests: with one lane per waiting unit, grids
# that leave few SMs idle (136 to 147 units on 148 SMs) issue 170 to 500
# requests per warp and measured up to 4.7 us more per call, where six
# lanes measured from no change to 0.7 us less.
SIBLING_LANES = 6
# The launch's wait-mode argument holds, above its low byte, the number of
# its first CTAs that start while the launch ahead still runs
# (``runtime.cute.capture_pdl.WAIT_MODE_HEAD_START_SHIFT``).
HEAD_START_SHIFT = 8


@cute.jit
def _request_page(
    raw: cute.Tensor,
    scales: cute.Tensor,
    page: cutlass.Int32,
    first_head: cutlass.Int32,
    shift: cutlass.Int32,
    layout: cutlass.Constexpr,
) -> None:
    """Request one page of an operand, K or V, into L2: its raw rows of both
    heads and their packed scales.

    The forms of one unit per CTA request K pages; the balanced partition's
    range-ahead role requests K and V pages. ``raw`` and ``scales`` are the
    operand's tensors, and ``layout`` holds the byte strides of a page in
    the raw and in the scale tensor
    (``paged_attention_prefetch.boundary_prefetch_layout``, for either
    operand): a key's two heads are adjacent 128-byte rows, a head's 128
    scale words 512 bytes. ``shift`` is the loader's key offset for this
    tile: row ``r`` of its box holds key ``shift + r``, so a positive shift
    leaves out leading keys and a negative one trailing keys, and the raw
    range is the part of the box the loader reads. The caller checked
    ``page`` against the operand's pool.
    """
    raw_page, scale_page, _pages = layout
    skipped = cutlass.Int32(0)
    if shift > 0:
        skipped = shift
    kept = cutlass.Int32(128) - skipped
    if shift < 0:
        kept = cutlass.Int32(128) + shift
    prims.cp_async_bulk_prefetch(
        raw.iterator
        + (
            cutlass.Int64(page) * raw_page
            + cutlass.Int64(first_head) * 128
            + cutlass.Int64(skipped) * 256
        ),
        kept * 256,
    )
    prims.cp_async_bulk_prefetch(
        scales.iterator
        + (cutlass.Int64(page) * scale_page + cutlass.Int64(first_head) * 512),
        1024,
    )


@cute.jit
def prefetch_boundary_units(
    pages: cute.Tensor,
    key: cute.Tensor,
    key_scales: cute.Tensor,
    unit: cutlass.Int32,
    lane: cutlass.Int32,
    capture_pdl_wait_mode: cutlass.Int32,
    signal: cute.Pointer | None,
    unit_bounds: cutlass.Constexpr,
    unit_bounds_args: tuple,
    form: cutlass.Constexpr,
    depth: cutlass.Constexpr,
    units: cutlass.Constexpr,
    capacity: cutlass.Constexpr,
    layout: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    window: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    page_walk: cutlass.Constexpr,
    key_clip: cutlass.Constexpr,
    cold_depth: cutlass.Constexpr = 0,
) -> None:
    """Request the first ``depth`` K pages of the unit this lane serves.

    ``wave_ahead``: lane zero of unit ``u`` serves unit ``u + capacity``,
    the unit that starts on this SM when this one ends. It resolves that
    unit at once and holds the requests until ``signal`` completes: the
    loader arrives there behind its last request, where the SM's memory
    traffic stops. A unit past the grid's end belongs to another launch,
    and nothing is requested for it.

    ``sibling``: the launch's first ``head`` CTAs run on SMs the launch
    ahead left idle while the others wait for its SMs, ``head`` being what
    the host installed above the wait mode: zero unless this launch
    overlaps a launch of the same kernel that left SMs idle. Lane ``l`` of
    unit ``u < head`` serves unit ``u + head * (l + 1)``, so the first
    ``SIBLING_LANES * head`` waiting units have one server each, and
    requests at once.

    ``cold_depth`` (sibling form, a grid of fewer units than SMs): a launch
    whose head start is zero, a cold one, lets its first
    ``capacity - units`` CTAs serve as those of an overlapped launch with
    that head start would, the first ``cold_depth`` pages only: the CTAs
    that an overlapped launch of this grid would start on idle SMs. Their
    requests help no unit of this launch, whose waiting units start long
    behind them; they pay where a later launch reads the same pages while
    L2 still holds them. A cold launch in the early wait mode has waited
    for its dependencies at the kernel's entry, so every read and request
    of the role is behind that wait. One in the before-writes mode (the
    host installed that mode and no head start) reads the served units'
    bounds and page ids and issues the requests ahead of its dependency
    wait, which that mode grants for the tensors the capture proof lists
    as loaded, as an overlapped launch's role does.
    """
    assert cutlass.const_expr(form in ("wave_ahead", "sibling"))
    assert cutlass.const_expr(depth > 0 and capacity > 1 and units > 0)
    assert cutlass.const_expr((signal is not None) == (form == "wave_ahead"))
    assert cutlass.const_expr(type(cold_depth) is int and cold_depth >= 0)
    assert cutlass.const_expr(
        cold_depth == 0 or (form == "sibling" and units < capacity)
    )
    _raw_page, _scale_page, pool = layout
    target = unit + capacity
    serving = lane == 0
    # Pages a serving lane resolves, and of those the ones this launch
    # requests: all of them unless a cold launch has a depth of its own.
    resolved = max(depth, cold_depth)
    limit = cutlass.Int32(depth)
    if cutlass.const_expr(form == "sibling"):
        head = capture_pdl_wait_mode >> HEAD_START_SHIFT
        if cutlass.const_expr(cold_depth > 0):
            cold = cutlass.Int32(head == 0)
            head = head + cutlass.Int32(capacity - units) * cold
            limit = limit + cutlass.Int32(cold_depth - depth) * cold
        target = unit + head * (lane + 1)
        serving = (unit < head) & (lane < SIBLING_LANES)
    if serving & (target < units):
        item, first, last, length = unit_bounds(target, *unit_bounds_args)
        count = cutlass.Int64(0)
        if last > first:
            count = last - first
        request = item // heads
        first_head = item % heads
        table_pages = cute.size(pages, mode=[1])
        # No page: an id outside the pool, which no request is formed for.
        physical = []
        shifts = []
        for index in cutlass.range_constexpr(resolved):
            logical_page, logical_key_base = attention_page_coordinates(
                first, last, cutlass.Int64(index), page_walk
            )
            page = cutlass.Int32(-1)
            shift = cutlass.Int32(0)
            if (index < count) & (logical_page >= 0) & (logical_page < table_pages):
                page = cutlass.Int32(pages[request, logical_page])
                shift = logical_attention_key_shift(
                    logical_key_base,
                    128,
                    length,
                    query_columns,
                    query_len,
                    group,
                    window,
                    coordinate_dtype,
                    key_clip,
                )
            if cutlass.const_expr(cold_depth > 0 and index >= min(depth, cold_depth)):
                # Beyond the depth of one of the two launch kinds: the other
                # kind forms no request for the page.
                if index >= limit:
                    page = cutlass.Int32(-1)
            physical.append(page)
            shifts.append(shift)
        if cutlass.const_expr(signal is not None):
            mbar_wait_sleep(signal, cutlass.Int32(0))
        for index in cutlass.range_constexpr(resolved):
            if (physical[index] >= 0) & (physical[index] < pool):
                _request_page(
                    key,
                    key_scales,
                    physical[index],
                    first_head,
                    shifts[index],
                    layout,
                )
