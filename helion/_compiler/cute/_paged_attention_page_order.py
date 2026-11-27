"""Bucketed page order for one-unit CTAs of the paired attention schedule.

The control warp that has no other work with one unit per CTA and no
value-metadata transport orders the unit's interior pages while the loader
streams the unit's first pages: it leaves the physical page ids of a run of
stream steps in shared memory, sorted by the bucket of the pool they lie in
and within a bucket in source order (``paged_attention_page_order`` says
what the order is for and what it changes). The loader then streams those
steps from that table instead of the page table.

What may be reordered is decided here, on the device, by the template's own
test. The candidates are the stream steps behind the loader's head and ahead
of the unit's tail, at most one sort's capacity. Each lane evaluates
``logical_attention_tile_is_interior`` for its candidates with the step's
own coordinates, the arguments the softmax and the key clip take. The steps
that pass must form one run: the warp reduces their number, the first and
the last over its lanes, and the run holds exactly when the span from the
first to the last has that number of steps. A unit whose interior candidates
are not one run, or too few to pay for the sort, is not reordered at all. So
every step of the run is an
interior tile: it has no mask and no key clip, the loader requests it with
shift zero, and the softmax treats it as it treats any other interior tile.
Only the page such a step streams changes, to that of another step of the
run.

The sort is a stable counting sort and a function of the page ids alone:
two launches over the same inputs stream the same order. Lane ``b`` owns
bucket ``b``. The lanes count their ids into the buckets with shared-memory
atomics, a prefix sum over the lanes gives each bucket its first position,
and the ids are placed one round of 32 at a time: the lanes of a round OR
their lane bit into their bucket's word, each takes the bucket's next
position plus the number of lower lanes in the word, and the bucket's
lowest lane then advances the position. Rounds follow the source order and
a round's lanes are in source order, so equal buckets keep it. The warp
synchronizes between those three steps, and between two synchronizations no
lane loads or stores a word that another lane stores: the counts and the
words are updated atomically in one step and only read in the next, and the
one lane per bucket that advances a position is the only one to touch it.
Each of those synchronizations (``bar.warp.sync``) is there for the memory
order it gives the warp's lanes, which share these words with no other
barrier between a lane's store and another lane's load; none is there for
the body's final join. One instance orders nothing: the synchronization
behind the last round's stores, which the rounds' common body renders and
no later load or store of another lane's word follows.

The warp publishes the run and arrives at one more pipeline barrier, every
lane behind its own stores; the loader's lane waits there behind its head
pages, for a unit that has more pages than those. The barrier belongs to
this role alone: no named barrier of another role is involved.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from ._paged_attention_compute import logical_attention_tile_is_interior
from ._paged_attention_partial_pipeline import attention_page_coordinates
from .paged_attention_page_order import PAGE_KEYS
from .paged_attention_page_order import PAGE_ORDER_BUCKETS
from .paged_attention_page_order import PAGE_ORDER_CAPACITY
from .paged_attention_page_order import PAGE_ORDER_HEAD
from .paged_attention_page_order import PAGE_ORDER_MIN_SORTED
from .paged_attention_page_order import PAGE_ORDER_TAIL

# Rounds of the warp's 32 lanes that cover one sort's capacity.
PAGE_ORDER_ROUNDS = PAGE_ORDER_CAPACITY // 32
# The words of the span the sorting warp publishes: its run's first step, its stop.
SPAN_FIRST = 0
SPAN_STOP = 1
SPAN_WORDS = 2


@cute.jit
def order_unit_pages(
    pages: cute.Tensor,
    order: cute.Tensor,
    buckets: cute.Pointer,
    members: cute.Pointer,
    span: cute.Tensor,
    ready: cute.Pointer,
    request: cutlass.Int32,
    first_page: cutlass.Int64,
    last_page: cutlass.Int64,
    count: cutlass.Int64,
    length: cutlass.Int32,
    lane: cutlass.Int32,
    shift: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    window: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    page_walk: cutlass.Constexpr,
) -> None:
    """Leave the bucketed order of the unit's interior run in ``order`` and
    the run's stream steps ``[first, stop)`` in ``span``, then arrive at
    ``ready``. All 32 lanes of the warp run it.

    ``order[i]`` is the physical page that stream step ``first + i``
    requests. A unit that is not reordered gets the empty run.
    """
    assert cutlass.const_expr(PAGE_ORDER_CAPACITY == 32 * PAGE_ORDER_ROUNDS)
    assert cutlass.const_expr(PAGE_ORDER_BUCKETS == 32)
    assert cutlass.const_expr(type(shift) is int and 0 <= shift < 32)
    # The stream steps behind the head and ahead of the tail, at most one
    # sort's capacity. Lane ``l`` holds candidate ``32 * round + l``.
    candidates = cutlass.Int32(0)
    if count > PAGE_ORDER_HEAD + PAGE_ORDER_TAIL:
        candidates = cutlass.Int32(
            cutlass.min(
                count - (PAGE_ORDER_HEAD + PAGE_ORDER_TAIL),
                cutlass.Int64(PAGE_ORDER_CAPACITY),
            )
        )
    # Each lane's candidates: the page id of every interior one, which of
    # its rounds they are, how many, and its first and last.
    ids = []
    inside = cutlass.Int32(0)
    own = cutlass.Int32(0)
    lowest = cutlass.Int32(PAGE_ORDER_CAPACITY)
    highest = cutlass.Int32(-1)
    for round_index in cutlass.range_constexpr(PAGE_ORDER_ROUNDS):
        index = round_index * 32 + lane
        page = cutlass.Int32(0)
        # The warp skips the rounds past the candidates together.
        if round_index * 32 < candidates:
            if index < candidates:
                logical_page, logical_key_base = attention_page_coordinates(
                    first_page,
                    last_page,
                    cutlass.Int64(PAGE_ORDER_HEAD + index),
                    page_walk,
                )
                if logical_attention_tile_is_interior(
                    logical_key_base,
                    PAGE_KEYS,
                    length,
                    query_columns,
                    query_len,
                    group,
                    window,
                    coordinate_dtype,
                ):
                    # A page id of the unit's own range, which the loader
                    # would read for this step.
                    page = cutlass.Int32(pages[request, logical_page])
                    inside = cutlass.Int32(inside | cutlass.Int32(1 << round_index))
                    own = cutlass.Int32(own + 1)
                    lowest = cutlass.Int32(cutlass.min(lowest, index))
                    highest = index
        ids.append(page)
    # The warp's interior candidates: their number, the first and the last,
    # by butterfly reductions that leave each result in every lane.
    total = own
    first = lowest
    last = highest
    for bit in cutlass.range_constexpr(5):
        total = cutlass.Int32(
            total + cutlass.Int32(cute.arch.shuffle_sync_bfly(total, 1 << bit))
        )
        first = cutlass.Int32(
            cutlass.min(
                first, cutlass.Int32(cute.arch.shuffle_sync_bfly(first, 1 << bit))
            )
        )
        last = cutlass.Int32(
            cutlass.max(
                last, cutlass.Int32(cute.arch.shuffle_sync_bfly(last, 1 << bit))
            )
        )
    # One run: as many interior candidates as the span from the first to the
    # last holds, and enough of them to pay for the sort.
    ordered = (last - first + 1 == total) & (total >= PAGE_ORDER_MIN_SORTED)
    if ordered:
        counts = cute.make_tensor(buckets, cute.make_layout((PAGE_ORDER_BUCKETS,)))
        lanes = cute.make_tensor(members, cute.make_layout((PAGE_ORDER_BUCKETS,)))
        counts[lane] = cutlass.Int32(0)
        # Behind every lane's store of its bucket's zero: the lanes add to
        # one another's counters next.
        cute.arch.sync_warp()
        for round_index in cutlass.range_constexpr(PAGE_ORDER_ROUNDS):
            if (inside & (1 << round_index)) != 0:
                cute.arch.atomic_add(
                    buckets + ((ids[round_index] >> shift) & (PAGE_ORDER_BUCKETS - 1)),
                    cutlass.Int32(1),
                    scope="cta",
                )
        # Behind every lane's additions: each lane reads its own counter,
        # which the others added to.
        cute.arch.sync_warp()
        mine = cutlass.Int32(counts[lane])
        prefix = mine
        for bit in cutlass.range_constexpr(5):
            lower = cutlass.Int32(cute.arch.shuffle_sync_up(prefix, 1 << bit))
            if lane >= (1 << bit):
                prefix = prefix + lower
        # Bucket ``lane``'s first position, and no lane in its word yet.
        counts[lane] = prefix - mine
        lanes[lane] = cutlass.Int32(0)
        # Behind every lane's store of its bucket's position and empty
        # word: the rounds' lanes update and read other buckets' next.
        cute.arch.sync_warp()
        lane_bit = cutlass.Int32(1) << lane
        below = cutlass.Int32(cute.arch.lanemask_lt())
        for round_index in cutlass.range_constexpr(PAGE_ORDER_ROUNDS):
            # The warp skips the rounds past the candidates together.
            if round_index * 32 < candidates:
                bucket = (ids[round_index] >> shift) & (PAGE_ORDER_BUCKETS - 1)
                word = cutlass.Int32(0)
                placed = (inside & (1 << round_index)) != 0
                if placed:
                    cute.arch.atomic_or(members + bucket, lane_bit, scope="cta")
                # Behind the round's bits: every lane of a bucket reads the
                # word the others set theirs in.
                cute.arch.sync_warp()
                # The round's lanes of a bucket read its word and its next
                # position, which no lane writes here.
                if placed:
                    word = cutlass.Int32(lanes[bucket])
                    order[
                        cutlass.Int32(counts[bucket])
                        + cutlass.Int32(cute.arch.popc(word & below))
                    ] = ids[round_index]
                # Behind those reads: one lane of a bucket now stores what
                # the others read.
                cute.arch.sync_warp()
                # The bucket's lowest lane of this round, by the word it
                # read, advances the position and clears the word: one lane
                # per bucket touches either, and no other lane reads them
                # until the warp has synchronized again.
                if placed & ((word & below) == 0):
                    counts[bucket] = cutlass.Int32(counts[bucket]) + cutlass.Int32(
                        cute.arch.popc(word)
                    )
                    lanes[bucket] = cutlass.Int32(0)
                # Behind that store: the next round's lanes set their bits
                # in the cleared word and read the advanced position. Behind
                # the last round this one orders nothing: from there on no
                # lane loads or stores a word another lane stored (lane 0
                # stores the run, which the loader reads behind all 32
                # arrivals). It stays, so that every round is one body.
                cute.arch.sync_warp()
    if lane == 0:
        run_first = cutlass.Int32(0)
        run_stop = cutlass.Int32(0)
        if ordered:
            run_first = cutlass.Int32(PAGE_ORDER_HEAD) + first
            run_stop = run_first + total
        span[SPAN_FIRST] = run_first
        span[SPAN_STOP] = run_stop
    # Every lane arrives behind its own stores: the loader's lane, which
    # waits for all 32 arrivals, then reads the run and the order.
    prims.mbarrier_arrive(ready)
