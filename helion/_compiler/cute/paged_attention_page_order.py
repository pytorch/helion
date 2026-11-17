"""The order in which a paired paged-attention CTA streams a unit's pages.

``cute_mma_dag_page_order`` names it. ``source``, the default and the
meaning of a missing key, streams the pages in the order the source's loop
visits them and renders the kernel without the option. ``bucketed`` streams
a unit's fully visible pages in the order of their physical page's bucket,
one of 32 equal ranges of the page pool, and within a bucket in source
order; the unit's first four pages and its last twelve keep their places.
Every CTA of a launch then reads one region of the pool at a time, in step
with the others, which shortens the page period of a memory-bound stream
(measured on a B200: 1.199 to 1.187 us per page).

The order is a permutation of whole tiles, and only of tiles for which the
template's own test proves that every key is visible to every row
(``logical_attention_tile_is_interior``). Such a tile has no mask and no
key clip, so a step that streams another interior page computes what that
page's own step would have: in real arithmetic the result does not depend
on the order. In floating point it does. The recurrence rounds each
probability to BF16 against the maximum of the pages streamed so far and
adds the tiles' products and denominators to FP32 accumulators in stream
order, so another order rounds at other points. That is a reassociation
only ``fast_math`` licenses, and the knob exists only under it.

``reversed`` walks a work item's pages newest first: step ``s`` of a walk
over the pages ``[first, last)`` takes page ``last - 1 - s``, where the
source's loop takes ``first + s``. It is offered to a kernel with a sliding
window, whose every work item streams the same few pages and whose newest
pages hold the scores a decode step's rows are largest on. Walked first,
they set the running maximum early, and the older tiles rescale less often
(the measurements are in this package's README.md, under "Special-purpose
emitters"). The walk is a work item's: with one unit per CTA that is the
CTA's, and on a persistent partition of a grid of more units than SMs every
unit of a CTA's stream is walked so, a unit in parts by its parts
(``cute_mma_dag.paged_attention_reversed_order_partition``). The walk's
last page is then the item's oldest, the one the window clips: of its keys
only those inside the window are visible.
``cute_mma_dag_clipped_page_lead`` moves that page ahead in the walk: with
a lead of ``p`` the walk takes it when it has ``p`` pages left, itself
included, and the newer pages behind that step move back by one. Zero, the
default and the meaning of a missing key, leaves the page where the
reversed order has it, last.

Here every tile moves, also the ones with a mask or a key clip, and each
keeps its own logical coordinates: the loader, the softmax's masks and a
boundary prefetch take a step's page and key base from one function of the
template (``attention_page_coordinates``), so a step's tile is the tile
its page has in source order, with that tile's mask and clip. The walk is
therefore a permutation of the source's iterations and nothing else. In
real arithmetic the result does not depend on it; in floating point it
changes which tiles raise the running maximum (and so which tiles rescale
the accumulators) and the sequence of the FP32 additions: the bucketed
order's class, which only ``fast_math`` licenses.

This module states the orders' constants and static facts once: the config
contract, the planner and the search surface say from them whether a
kernel is offered an order, and the device code and the shared-memory
table size the sort from them. It imports nothing of the compiler or the
DSL, so both can import it.
"""

from __future__ import annotations

from typing import NamedTuple

MMA_DAG_PAGE_ORDER_KEY = "cute_mma_dag_page_order"
MMA_DAG_PAGE_ORDER_SOURCE = "source"
MMA_DAG_PAGE_ORDER_BUCKETED = "bucketed"
MMA_DAG_PAGE_ORDER_REVERSED = "reversed"
MMA_DAG_PAGE_ORDER_CHOICES: tuple[str, ...] = (
    MMA_DAG_PAGE_ORDER_SOURCE,
    MMA_DAG_PAGE_ORDER_BUCKETED,
    MMA_DAG_PAGE_ORDER_REVERSED,
)

# ``cute_mma_dag_clipped_page_lead``: the pages a reversed walk has left,
# the page itself included, when it takes the work item's oldest page, the
# one a sliding window clips. Zero (and one) is the reversed order itself,
# the page last. A walk of ``n`` pages takes the page at step
# ``max(n - max(lead, 1), 0)``: a work item of fewer pages than the lead
# takes it first, as a stream shorter than the lead of the tail release
# releases at its first page.
#
# Which of the two gains depends on the case: a lead of 8 beside a raw V
# ring of three pages (``cute_mma_dag_value_ring``) on some, the page last
# beside the default ring on others. Beside that ring no other lead was
# better than 8 where leads were compared, so the choices are these two.
# (The measurements: this package's README.md, under "Special-purpose
# emitters", where it describes ``cute_mma_dag_page_order="reversed"``.)
MMA_DAG_CLIPPED_PAGE_LEAD_KEY = "cute_mma_dag_clipped_page_lead"
MMA_DAG_CLIPPED_PAGE_LEAD_DEFAULT = 0
MMA_DAG_CLIPPED_PAGE_LEAD_CHOICES: tuple[int, ...] = (0, 8)


class ReversedWalk(NamedTuple):
    """What the paired body hands the template's page mapping
    (``attention_page_coordinates``) for
    ``cute_mma_dag_page_order="reversed"``, in the place of a source
    reflection's dtype: the walk and its clipped page's lead."""

    clipped_page_lead: int


# What names a walk to the template's page mapping, the ``page_walk`` every
# role of the paired body hands on: ``None`` for the source's loop, the DSL's
# integer type of the source's own reflection, or a ``ReversedWalk``.
PageWalk = type | ReversedWalk | None


# What a kernel without the pair proof lacks for the reversed walk.
REVERSED_ORDER_NEEDS_PAIR = (
    "the paired attention schedule, which this kernel has no pair proof for"
)


def reversed_order_missing(
    *, paired: bool, fast_math: bool, windowed: bool, reflected: bool
) -> str:
    """What a kernel lacks for the reversed walk, from its static facts;
    empty where it lacks nothing.

    ``paired``: the bind has the paired schedule, the only body with the
    walk.
    ``windowed`` is the mask's fact: the source's visibility predicate has
    a lower bound, ``key_position >= query_position - window``. It says
    nothing of the loop's bounds, which stay the source's expressions; the
    page the lead positions is the walk's oldest, the loop's first, which
    is the window's clipped page wherever the loop begins at the window's
    edge. ``reflected``: the source's loop index is already the typed
    reflection ``begin + end - 1 - iteration``, a walk newest first.
    """
    if not paired:
        return REVERSED_ORDER_NEEDS_PAIR
    if not fast_math:
        return (
            "fast_math: the walk changes the order in which the tiles are "
            "accumulated and which of them rescale, so results differ from "
            "the source order's within rounding"
        )
    if not windowed:
        return (
            "a sliding window in the kernel's mask (a visibility bound "
            "key_position >= query_position - window): the walk is offered "
            "where every work item streams a window's few pages"
        )
    if reflected:
        return (
            "a source loop that visits its pages oldest first: this kernel's "
            "loop index is already reflected, and its source order is the "
            "walk newest first"
        )
    return ""


# Keys of a page: the paired schedule's tile.
PAGE_KEYS = 128
# Buckets of the page pool: one per lane of the sorting warp, whose lane
# ``b`` owns bucket ``b``'s counter.
PAGE_ORDER_BUCKETS = 32
# Pages the loader streams in source order before it takes the order: the
# sorting warp starts behind the setup rendezvous, beside the loader, and
# the stream should not wait for it. Measured on a B200 (SDG eval harness,
# 128 requests of 211 and 391 pages, twelve cases over two input banks,
# handwritten / Helion with the longer lead and the deeper ring): 0.9978
# with a head of two pages, 0.9999 with four (0.5 to 0.7 us less per call:
# behind two pages the stream waits for the sort) and 0.9989 with eight,
# the same as four in the mean of the samples.
PAGE_ORDER_HEAD = 4
# Pages at the end of a unit that keep their source order. The newest pages
# are the ones a decode step's scores are largest on; in bucket order they
# land in mid-stream, where each raises the running maximum and rescales
# what was accumulated. Measured on a B200 with the order's prototype (SDG
# eval harness, 128 requests without a window, twelve cases over two input
# banks, handwritten / Helion): a tail of 2 pages read 0.9897 with the two
# cases whose newest pages hold large scores 1 to 3 us slower than the
# source order, 6 pages 0.9913 with one of them still slower, 12 pages
# 0.9932 with none. No measured class had another tail win, so it is a
# constant.
PAGE_ORDER_TAIL = 12
# Interior pages one sort orders: thirteen rounds of the warp's 32 lanes. A
# unit with more has its first 416 ordered and streams the rest in source
# order.
PAGE_ORDER_CAPACITY = 416
# Fewest interior pages worth ordering. The order gains 0.012 to 0.015 us
# per ordered page (3.0 us over 197 pages), and a sort the stream waits for
# costs it 0.4 to 0.7 us (behind a head of two pages), so below about 32
# pages it cannot pay; three ordered pages of a 17-page window cost 1.4 to
# 2.05 us. A unit with fewer streams in source order, and a kernel none of
# whose units can have as many is not offered the order.
PAGE_ORDER_MIN_SORTED = 32


def page_order_shift(page_extent: int) -> int:
    """Bits to drop from a physical page id for its bucket: the pool of
    ``page_extent`` pages, the K tensor's page extent, in
    ``PAGE_ORDER_BUCKETS`` equal ranges of a power of two pages. A pool of
    up to 32 pages has a bucket per page."""
    assert page_extent > 0
    return max(0, (page_extent - 1).bit_length() - 5)


def page_order_bucket(page: int, shift: int) -> int:
    """The bucket of physical page ``page``. An id outside the pool takes
    the bucket of its low bits: the order is defined for every id."""
    return (page >> shift) & (PAGE_ORDER_BUCKETS - 1)


def page_order_sortable_bound(
    page_limit: int, window_left: int | None, key_width: int = PAGE_KEYS
) -> int:
    """Most pages of one unit the order can permute, from the kernel's
    static facts.

    A unit streams at most ``page_limit`` pages, the page table's extent,
    of which the head and the tail keep their places. Under a window of
    ``window_left`` keys behind a row's position, the keys every row of a
    tile sees span at most ``window_left + 1``, so at most that many keys'
    whole pages are interior. One sort orders ``PAGE_ORDER_CAPACITY``.
    """
    sortable = page_limit - PAGE_ORDER_HEAD - PAGE_ORDER_TAIL
    if window_left is not None:
        sortable = min(sortable, (window_left + 1) // key_width)
    return max(0, min(sortable, PAGE_ORDER_CAPACITY))


def page_order_offered(page_limit: int, window_left: int | None) -> bool:
    """Whether some unit of the kernel can have enough interior pages for
    the order to pay (``PAGE_ORDER_MIN_SORTED``). A 17-page window has at
    most sixteen."""
    return page_order_sortable_bound(page_limit, window_left) >= PAGE_ORDER_MIN_SORTED
