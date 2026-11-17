"""Two independent attention heads per CTA with shared conversion warpgroups.

Each 128-thread compute group owns a 256-column TMEM slab. Four K and four V
conversion warps alternate the two heads; the controls issue QK, TMA and PV,
and optionally classify value metadata.
Physical warp modulo four selects the TMEM row quarter, so all eight converter
warps keep their full groups. Named barriers 1/2 and 3/4 belong to the separate
compute groups. Register quotas stay inside disjoint branches, each taken by
whole warpgroups from one instruction: by default 144 for compute, 72 for
either conversion and 48 for all four control warps, which is the CTA's whole
launch pool (640 threads at 96 registers: 61,440). ``register_quotas`` names
another split of that pool in the same role order.
Optional barrier 5 retires the allocation before output stores: both 128-thread
compute groups arrive after their final reads and the 32-thread QK role frees it.
Every role then returns when its own work ends; without that retirement, or
when the caller needs one common end, all 640 threads join before the release.
Barrier 6 is the setup rendezvous. The loader warp initializes the pipeline
barriers and only arrives, so its page stream starts while the other roles
finish their setup; the QK role owns the query barrier and copies.
With a key clip the loader issues each boundary tile's K, V and scale boxes at
the key offset the compute groups derive for that tile's mask. Both sides
compute that offset from kernel arguments and compile-time constants alone, so
the loader needs nothing the setup rendezvous publishes.

The helpers that walk a work item take its range: its first step within the
CTA's work item (``step0``, for page and key coordinates) and the pages the CTA
streamed before it (``g0``, or its parity ``phase``, for ring slots and barrier
phases). With one work item per CTA the body runs each of them over one range,
the whole item, so both are zero. The other helpers follow the page stream and
take its length.

A persistent partition runs several units (work items) in one CTA. The loader,
the QK issuer and the compute groups then walk the CTA's units and run their
helper once per unit; the conversion warps and the PV issuer follow the page
stream and only need its total length. All barriers keep cycling across units.
The roles keep their warps, so each takes the register quota of the role it
extends and every split of the pool fits a persistent CTA as it fits a
one-unit CTA: the walking compute groups the compute quota, the conversion
warps theirs, and the loader, the issuers and the spare warp the control
quota.
A compute group is the only role that stops at a unit boundary: it stores the
unit's outputs and resets its panels and row state while the other roles
already stage the next unit's first pages. Every unit's queries go into one
of the query stages, whose barriers the loader owns with the pipeline's: the
QK role requests those of the CTA's first unit with pages, as it does with
one unit per CTA, and the loader those of every later one, ahead of the
unit's pages. Which units a CTA runs follows from all units' lengths
(``_paged_attention_partition``): the threads scan their shares of them into
two plan tables and arrive at barrier 7, where only the loader waits. The
loader finds the CTA's range from the tables, leaves it in shared memory and
then arrives at the setup rendezvous, which publishes it to the other roles.
So the loader waits for nothing but the lengths, and no other role starts
before the plan exists. All of that is the head of the loader's own role,
behind its register quota, and the stream starts from the plan in the
loader's registers. A persistent CTA's four control warps take that quota
from one instruction behind the scan, ahead of the setup rendezvous
(``setmaxnreg`` has every warp of a warpgroup execute the same one), so the
loader holds it while the other warps still scan. With one unit per scanning
thread the scan keeps the lengths it loaded, and every role evaluates its
units from them (``_scanned_bounds``): a persistent CTA then loads each
length once, as a one-unit CTA does.

The balanced partition gives a CTA a share of the pages instead of whole
units: the walking roles clip each unit to the CTA's slots, so the CTA's
first and last unit can be parts of units that neighbouring CTAs run the
rest of. A part runs as a unit does, from an empty state at the unit's own
steps, and ends differently: its compute group exchanges the part's
unnormalized result with the unit's other parts through a workspace of the
launch, and one of them merges and stores the unit
(``_paged_attention_partition``). The group reads and writes that
workspace only behind the launch's dependency wait. The CTA's spare warp,
which has no other work on this schedule, makes the arrival of the one
part per CTA that is not its unit's lowest, so that the compute group does
not stall the stream behind it for the arrival's memory round trip.

The loader stages two pages of raw K and, by default, two of raw V per head;
``value_ring`` names another depth of the raw V ring, whose slots and
barriers the loader and the V conversion warps then take in turns of that
depth. The body's shared memory is one table
(``paged_attention_shared_memory``), which the config contract replays
against the device before it admits a depth.

With one unit per CTA and without the value-metadata transport that warp is
idle, and a bucketed page order gives it a role
(``_paged_attention_page_order``): it sorts the unit's interior pages by
the region of the pool they lie in and hands the order to the loader at one
more pipeline barrier, behind the loader's first pages. A unit of no more
pages than those has no step that could take the order, and its loader
does not wait for the warp.

A reversed page order has no role and no state of its own: it is another
mapping from a work item's step to its page and key base
(``attention_page_coordinates``), which the loader, the compute groups and
a boundary prefetch all take from the one walk the body hands them, so a
step is the same tile to each.

A boundary prefetch gives that warp a role as well, behind the sort
(``_paged_attention_prefetch``): it requests the first K pages of a unit
that has not started into L2. Where the unit served is the one that takes
over this CTA's SM, the requests wait at one more pipeline barrier, at which
the loader arrives behind its last request. Only some lanes of the warp
serve a unit: where the body ends with its all-thread join, the join waits
for them, as a barrier waits for the lanes of its warp.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as experimental_cuda
import cutlass.experimental.primitives as prims
import cutlass.utils as utils

from ._mma_dag_tcgen05_runtime import mbar_wait
from ._mma_dag_tcgen05_runtime import mbar_wait_sleep
from ._paged_attention_compute import logical_attention_key_shift
from ._paged_attention_compute import normalize_output
from ._paged_attention_compute import output_normalization_factor
from ._paged_attention_io import load_query_element
from ._paged_attention_io import store_output_element
from ._paged_attention_io import store_output_rows
from ._paged_attention_io import store_row_element
from ._paged_attention_page_order import SPAN_FIRST
from ._paged_attention_page_order import SPAN_STOP
from ._paged_attention_page_order import SPAN_WORDS
from ._paged_attention_page_order import order_unit_pages
from ._paged_attention_partial_compute import load_query_panel
from ._paged_attention_partial_compute import mma_128xnx128
from ._paged_attention_partial_compute import online_softmax_per_key
from ._paged_attention_partial_compute import store_query_panel
from ._paged_attention_partial_pipeline import attention_page_coordinates
from ._paged_attention_partition import MAIL_ITEM
from ._paged_attention_partition import MAIL_KIND
from ._paged_attention_partition import MAIL_LAST
from ._paged_attention_partition import MAIL_LOWEST
from ._paged_attention_partition import MAIL_NONE
from ._paged_attention_partition import MAIL_PARTS
from ._paged_attention_partition import PART_MERGE
from ._paged_attention_partition import PART_MERGE_PAIR
from ._paged_attention_partition import PART_POST
from ._paged_attention_partition import PART_PUBLISH
from ._paged_attention_partition import PLAN_ACTIVE
from ._paged_attention_partition import PLAN_BEGIN
from ._paged_attention_partition import PLAN_END
from ._paged_attention_partition import PLAN_FIRST
from ._paged_attention_partition import PLAN_HEADER
from ._paged_attention_partition import PLAN_LAST
from ._paged_attention_partition import PLAN_PAGES
from ._paged_attention_partition import PLAN_SKEW
from ._paged_attention_partition import PLAN_SLOTS
from ._paged_attention_partition import POST_BARRIER
from ._paged_attention_partition import POST_THREADS
from ._paged_attention_partition import POSTED_BARRIER
from ._paged_attention_partition import WALK_END
from ._paged_attention_partition import WALK_HEADER
from ._paged_attention_partition import WALK_ITEM
from ._paged_attention_partition import WALK_LOWEST
from ._paged_attention_partition import WALK_PAGES
from ._paged_attention_partition import WALK_PARTS
from ._paged_attention_partition import WALK_POSTED
from ._paged_attention_partition import WALK_ROLE
from ._paged_attention_partition import WALK_STEP
from ._paged_attention_partition import WALK_UNIT
from ._paged_attention_partition import arrive_part
from ._paged_attention_partition import merge_pair_other
from ._paged_attention_partition import merge_pair_own
from ._paged_attention_partition import merge_pair_rows
from ._paged_attention_partition import merge_parts
from ._paged_attention_partition import pair_partial
from ._paged_attention_partition import partition_plan
from ._paged_attention_partition import partition_scan
from ._paged_attention_partition import post_parts
from ._paged_attention_partition import split_arrival
from ._paged_attention_partition import store_part
from ._paged_attention_partition import unit_page_count
from ._paged_attention_partition import unit_segment
from ._paged_attention_partition import workspace_address
from ._paged_attention_prefetch import prefetch_boundary_units
from ._paged_attention_reference_compute import online_softmax_reference
from ._paged_attention_runtime import _load_words4
from ._paged_attention_runtime import _packed_scale_layout_supported
from ._paged_attention_runtime import enqueue_paged_query
from ._paged_attention_runtime import enqueue_paged_tma_operand
from ._paged_attention_runtime import stage_positive_zero_value_tile
from ._paged_attention_runtime import stage_scaled_s8_rows_to_tmem
from ._paged_attention_runtime import stage_scaled_s8_transpose_to_tmem
from .device.warp_reduce import warp_row_sum
from .paged_attention_page_order import MMA_DAG_CLIPPED_PAGE_LEAD_CHOICES
from .paged_attention_page_order import MMA_DAG_PAGE_ORDER_BUCKETED
from .paged_attention_page_order import MMA_DAG_PAGE_ORDER_CHOICES
from .paged_attention_page_order import MMA_DAG_PAGE_ORDER_REVERSED
from .paged_attention_page_order import MMA_DAG_PAGE_ORDER_SOURCE
from .paged_attention_page_order import PAGE_ORDER_CAPACITY
from .paged_attention_page_order import PAGE_ORDER_HEAD
from .paged_attention_page_order import ReversedWalk
from .paged_attention_registers import PAIRED_DEFAULT_REGISTER_QUOTAS
from .paged_attention_registers import paired_register_quotas_reason
from .paged_attention_release import MMA_DAG_RELEASE_LEAD_DEFAULT
from .paged_attention_shared_memory import KEY_RING_PAGES
from .paged_attention_shared_memory import MAIL_WORDS
from .paged_attention_shared_memory import PAIRED_DEFAULT_VALUE_RING
from .paged_attention_shared_memory import PLAN_WORDS
from .paged_attention_shared_memory import WALK_WORDS
from .paged_attention_shared_memory import WHOLE_UNIT_WORDS
from .paged_attention_shared_memory import SharedArray
from .paged_attention_shared_memory import pipeline_barrier_plan
from .paged_attention_shared_memory import pipeline_parts
from .paged_attention_shared_memory import pipeline_shared_bytes
from .paged_attention_shared_memory import pipeline_shared_memory
from .paged_attention_shared_memory import plan_geometry


@cute.jit
def _wait_non_mma(
    barrier: cute.Pointer,
    phase: cutlass.Int32 | cutlass.Int64,
    sleep_waits: cutlass.Constexpr,
) -> None:
    """Keep issuer waits prompt while optionally suspending the other roles."""
    if cutlass.const_expr(sleep_waits):
        mbar_wait_sleep(barrier, phase)
    else:
        mbar_wait(barrier, phase)


@cute.jit
def _scanned_bounds(
    unit: cutlass.Int32,
    lengths: cute.Tensor | None,
    unit_bounds: cutlass.Constexpr,
    unit_bounds_scanned: cutlass.Constexpr,
    unit_bounds_args: tuple,
) -> tuple[cutlass.Int32, cutlass.Int64, cutlass.Int64, cutlass.Int32]:
    """A unit's item, loop bounds and length for a role of a persistent CTA.

    The plan scan loaded every unit's length. Where it kept them
    (``lengths``: one unit per scanning thread, and a prelude whose loads
    all read that length), a role evaluates the unit from that word of shared
    memory instead of loading the length again: the same value and the same
    arithmetic, without a global load ahead of what the role does next.
    Every caller runs behind a barrier that follows the scan.
    """
    if cutlass.const_expr(lengths is not None):
        assert lengths is not None
        return unit_bounds_scanned(
            unit, cutlass.Int32(lengths[unit]), *unit_bounds_args
        )
    return unit_bounds(unit, *unit_bounds_args)


@cute.jit
def _unit_range(
    unit: cutlass.Int32,
    header: cutlass.Int32,
    begin: cutlass.Int32 | None,
    stop: cutlass.Int32 | None,
    plan: cute.Tensor | None,
    lengths: cute.Tensor | None,
    unit_bounds: cutlass.Constexpr,
    unit_bounds_scanned: cutlass.Constexpr,
    unit_bounds_args: tuple,
    page_limit: cutlass.Constexpr,
    partition: cutlass.Constexpr,
) -> tuple[
    cutlass.Int32,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int32,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int64,
]:
    """What a walking role of a persistent CTA runs of a unit.

    Returns the unit's item, loop bounds and length (``_scanned_bounds``),
    the pages the unit streams, and the range of them this CTA runs: its
    first step within the unit and its page count. A CTA of whole units runs
    all of them, from step zero. Under the balanced partition it runs the
    unit's steps in its slots (``unit_segment``): ``header`` is the unit's
    header slot, which the caller advances by the unit's pages plus one, and
    the slots are ``[begin, stop)``. A role that does not keep them in
    registers passes ``plan`` instead, and they are read from it behind the
    unit's bounds.

    The loader, the QK issuer and the compute groups walk the same units
    and pair their barrier phases by the pages they count, so all three
    take every unit's range from here and none evaluates it itself.
    """
    item, first_page, last_page, length = _scanned_bounds(
        unit, lengths, unit_bounds, unit_bounds_scanned, unit_bounds_args
    )
    unit_pages = unit_page_count(first_page, last_page, page_limit)
    step0 = cutlass.Int64(0)
    count = unit_pages
    if cutlass.const_expr(partition == "balanced"):
        if cutlass.const_expr(plan is not None):
            assert plan is not None
            begin = cutlass.Int32(plan[PLAN_BEGIN])
            stop = cutlass.Int32(plan[PLAN_END])
        assert begin is not None and stop is not None
        count, step0 = unit_segment(header, unit_pages, begin, stop)
    return item, first_page, last_page, length, unit_pages, step0, count


@cute.jit
def _compute_begin(
    output_address: cutlass.Int32,
    denominator_address: cutlass.Int32 | None,
    pair_head: cutlass.Int32,
    query_columns: cutlass.Constexpr,
    probability_schedule: cutlass.Constexpr,
) -> tuple[cute.Tensor, cutlass.Boolean]:
    """Zero one group's O panel and return a work item's empty denominators."""
    zero = cutlass.Array(cutlass.Float32, query_columns, alignment=16)
    for query in cutlass.range_constexpr(query_columns):
        zero[query] = cutlass.Float32(0)
    store_query_panel(output_address, zero, query_columns)
    if cutlass.const_expr(probability_schedule == "early_tmem"):
        store_query_panel(denominator_address, zero, query_columns)
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier(barrier_id=2 + pair_head * 2, number_of_threads=128)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    partial_columns = 1 if probability_schedule == "early_tmem" else query_columns
    # The TMEM schedule loads its real partials inside the recurrence. Keep a
    # typed placeholder for CuTe's shared call signature; it emits no storage.
    partials = cute.make_rmem_tensor((partial_columns,), cutlass.Float32)
    for query in cutlass.range_constexpr(partial_columns):
        partials[query] = cutlass.Float32(0)
    reference_invalid = cutlass.Boolean(True)
    return partials, reference_invalid


@cute.jit
def _compute_pages(
    probability: cute.Tensor,
    state: cute.Tensor,
    scratch: cute.Tensor,
    qk_done: cute.Pointer,
    score_free: cute.Pointer,
    probability_ready: cute.Pointer,
    pv_done: cute.Pointer,
    base: cutlass.Int32,
    pair_head: cutlass.Int32,
    thread: cutlass.Int32,
    lane: cutlass.Int32,
    output_address: cutlass.Int32,
    denominator_address: cutlass.Int32 | None,
    partials: cute.Tensor,
    reference_invalid: cutlass.Boolean,
    first_page: cutlass.Int64,
    last_page: cutlass.Int64,
    phase: cutlass.Int32,
    step0: cutlass.Int64,
    count: cutlass.Int64,
    length: cutlass.Int32,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    window: cutlass.Constexpr,
    score_scale: cutlass.Float32,
    fast_math: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    reference_scores: cutlass.Constexpr,
    base2_scores: cutlass.Constexpr,
    page_walk: cutlass.Constexpr,
    sleep_waits: cutlass.Constexpr,
    coefficients_capped: cutlass.Constexpr,
    probability_schedule: cutlass.Constexpr,
    rowwise_rescale_threshold: cutlass.Constexpr,
    headwide_rescale: cutlass.Constexpr,
    key_clip: cutlass.Constexpr,
    walk: cute.Tensor | None = None,
) -> None:
    """Run the recurrence over ``count`` pages of a work item, from ``step0``.

    Page and key coordinates are those of the item's step ``step0 + step``.
    The barriers alternate with every page the CTA streams, so their phases
    are ``phase + step``, where only the parity of ``phase`` matters: that of
    the pages the CTA streamed before this range. In the nonserial schedules
    the range's last PV is still outstanding on return; ``_compute_reduce``
    waits for it.

    With ``walk``, the group's record in shared memory, the first step is
    read from it at every page instead: the softmax leaves no register for
    one more value that only the coordinates need.
    """
    for step in cutlass.range(count):
        if cutlass.const_expr(walk is not None):
            assert walk is not None
            step0 = cutlass.Int64(cutlass.Int32(walk[WALK_STEP]))
        _logical_page, logical_key_base = attention_page_coordinates(
            first_page, last_page, step0 + step, page_walk
        )
        _wait_non_mma(qk_done, (phase + step) & 1, sleep_waits)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        if cutlass.const_expr(reference_scores):
            maximum_changed, reference_invalid = online_softmax_reference(
                base + 128,
                probability,
                state,
                scratch,
                thread,
                logical_key_base,
                length,
                query_len,
                group,
                window,
                score_scale,
                1 + pair_head * 2,
                partials,
                reference_invalid,
                fast_math,
                score_free,
                coordinate_dtype,
                query_columns,
                base2_scores,
                convergent_branches=True,
                group_barrier=False,
                probability_schedule=probability_schedule,
                denominator_address=denominator_address,
                previous_pv=pv_done,
                previous_phase=(phase + step - 1) & 1,
                have_previous=step > 0,
                previous_sleep=sleep_waits,
                rowwise_rescale_threshold=rowwise_rescale_threshold,
                headwide_rescale=headwide_rescale,
                key_clip=key_clip,
            )
        else:
            # Only reference-score publication opts out here. Keep the per-key
            # helper's default group barrier until that elision is measured.
            maximum_changed = online_softmax_per_key(
                base + 128,
                probability,
                state,
                scratch,
                thread,
                logical_key_base,
                length,
                query_len,
                group,
                window,
                score_scale,
                1 + pair_head * 2,
                partials,
                fast_math,
                score_free,
                coordinate_dtype,
                query_columns,
                base2_scores,
                coefficients_capped,
                key_clip=key_clip,
            )
        if maximum_changed:
            previous = load_query_panel(output_address, query_columns)
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            rescaled = cutlass.Array(cutlass.Float32, query_columns, alignment=16)
            for query in cutlass.range_constexpr(query_columns):
                rescaled[query] = previous[query] * state[2, query]
            store_query_panel(output_address, rescaled, query_columns)
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        # Each warp owns its P columns and O rows. The softmax helper's
        # proxy fence and the TMEM fence/sync below publish both through
        # four arrivals. Serial waits below; early schedules wait for prior
        # PV inside the next softmax call before rewriting either panel.
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.sync_warp()
        if lane == 0:
            prims.mbarrier_arrive(probability_ready)
        if cutlass.const_expr(probability_schedule == "serial"):
            _wait_non_mma(pv_done, (phase + step) & 1, sleep_waits)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _reduce_rows(
    state: cute.Tensor,
    scratch: cute.Tensor,
    pair_head: cutlass.Int32,
    thread: cutlass.Int32,
    warp: cutlass.Int32,
    lane: cutlass.Int32,
    denominator_address: cutlass.Int32 | None,
    partials: cute.Tensor,
    query_columns: cutlass.Constexpr,
    output_reciprocal: cutlass.Constexpr,
    output_scale_log2: cutlass.Constexpr,
    maximum_output: cute.Tensor | None,
    probability_schedule: cutlass.Constexpr,
) -> cutlass.Float32:
    """Reduce a range's denominators into its row state: every row owner's
    total into ``state`` row 1 and, for the normalized epilogue, its
    normalization into row 3. Returns the calling thread's row total, zero
    for a thread that owns no row.

    The head of both reductions (``_compute_reduce``, ``_reduce_part``). It
    reads neither shared P nor O, so it runs inside the wait for the
    range's last PV; the caller's next group barrier publishes the rows.
    """
    if cutlass.const_expr(probability_schedule == "early_tmem"):
        partials = load_query_panel(denominator_address, query_columns)
    # Transpose row ownership while retaining the exact ascending-XOR sum tree.
    if cutlass.const_expr(query_columns >= 32):
        row, row_total = warp_row_sum(
            [cutlass.Float32(partials[query]) for query in range(32)], lane
        )
        scratch[warp, row] = row_total
    if cutlass.const_expr(query_columns == 8 or query_columns == 40):
        offset = query_columns - 8
        tail_row, tail_total = warp_row_sum(
            [cutlass.Float32(partials[offset + query]) for query in range(8)], lane
        )
        if lane < 8:
            scratch[warp, offset + tail_row] = tail_total
    cute.arch.barrier(barrier_id=1 + pair_head * 2, number_of_threads=128)
    total = cutlass.Float32(0)
    if thread < query_columns:
        total = cutlass.Float32(scratch[0, thread]) + cutlass.Float32(
            scratch[1, thread]
        )
        total = total + cutlass.Float32(scratch[2, thread])
        total = total + cutlass.Float32(scratch[3, thread])
        state[1, thread] = total
        if cutlass.const_expr(maximum_output is None):
            safe_total = total
            if total == 0:
                safe_total = cutlass.Float32(1)
            state[3, thread] = output_normalization_factor(
                safe_total, output_reciprocal, output_scale_log2=output_scale_log2
            )
    return total


@cute.jit
def _drain_last_pv(
    pv_done: cute.Pointer,
    phase: cutlass.Int32,
    count: cutlass.Int64,
    sleep_waits: cutlass.Constexpr,
    probability_schedule: cutlass.Constexpr,
) -> None:
    """Wait for the last PV of a range ``_compute_pages`` ran with ``phase``
    and ``count``: the nonserial schedules leave it outstanding. Every read
    of the range's O panel follows this wait."""
    if cutlass.const_expr(probability_schedule != "serial"):
        if count > 0:
            _wait_non_mma(pv_done, (phase + count - 1) & 1, sleep_waits)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)


@cute.jit
def _compute_reduce(
    state: cute.Tensor,
    scratch: cute.Tensor,
    pv_done: cute.Pointer,
    pair_head: cutlass.Int32,
    thread: cutlass.Int32,
    warp: cutlass.Int32,
    lane: cutlass.Int32,
    output_address: cutlass.Int32,
    denominator_address: cutlass.Int32 | None,
    partials: cute.Tensor,
    phase: cutlass.Int32,
    count: cutlass.Int64,
    query_columns: cutlass.Constexpr,
    output_reciprocal: cutlass.Constexpr,
    output_scale_log2: cutlass.Constexpr,
    sleep_waits: cutlass.Constexpr,
    maximum_output: cute.Tensor | None,
    probability_schedule: cutlass.Constexpr,
) -> cute.Tensor | cutlass.Array:
    """Reduce a work item's denominators into its row state; return its O panel.

    ``phase`` and ``count`` are those ``_compute_pages`` ran the range with:
    they select the phase of its last PV.
    """
    # The loop leaves the final denominators. In the nonserial schedules its
    # last PV may still be accumulating O: reduce the denominators inside that
    # wait, since the reduction reads neither shared P nor O.
    _reduce_rows(
        state,
        scratch,
        pair_head,
        thread,
        warp,
        lane,
        denominator_address,
        partials,
        query_columns,
        output_reciprocal,
        output_scale_log2,
        maximum_output,
        probability_schedule,
    )
    cute.arch.barrier(barrier_id=1 + pair_head * 2, number_of_threads=128)
    _drain_last_pv(pv_done, phase, count, sleep_waits, probability_schedule)
    # O is final once the last PV completed (or was zeroed for an empty range).
    result = load_query_panel(output_address, query_columns)
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    return result


@cute.jit
def _reduce_part(
    state: cute.Tensor,
    scratch: cute.Tensor,
    pv_done: cute.Pointer,
    pair_head: cutlass.Int32,
    thread: cutlass.Int32,
    warp: cutlass.Int32,
    lane: cutlass.Int32,
    output_address: cutlass.Int32,
    denominator_address: cutlass.Int32 | None,
    partials: cute.Tensor,
    phase: cutlass.Int32,
    count: cutlass.Int64,
    query_columns: cutlass.Constexpr,
    output_reciprocal: cutlass.Constexpr,
    output_scale_log2: cutlass.Constexpr,
    sleep_waits: cutlass.Constexpr,
    maximum_output: cute.Tensor | None,
    probability_schedule: cutlass.Constexpr,
    walk: cute.Tensor,
    split: cutlass.Boolean,
    workspace: cutlass.Int64,
    record_bytes: cutlass.Constexpr,
    base2_scores: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
) -> cute.Tensor:
    """``_compute_reduce`` for the balanced partition, whose range can be a
    part of a split unit: reduce the range's denominators into its row
    state and return its O panel in registers the caller can fold other
    parts into.

    For a part (``split``) one thread left the part's role in ``walk``, the
    group's record, ahead of this call. If that role is to merge a unit in
    two parts, the other part's loads start behind the first barrier here,
    the row owners fold its statistics into the row state between the two
    barriers, and the panel returned is the unit's: the loads and the
    weights' barrier then overlap the wait for this part's last PV, which
    the reduction of any range overlaps.
    """
    assert cutlass.const_expr(maximum_output is None)
    total = _reduce_rows(
        state,
        scratch,
        pair_head,
        thread,
        warp,
        lane,
        denominator_address,
        partials,
        query_columns,
        output_reciprocal,
        output_scale_log2,
        maximum_output,
        probability_schedule,
    )
    # Merging a unit in two parts: the other part's panel, the unit's first
    # term, and its row statistics.
    pair = cutlass.Boolean(False)
    other_maximum = cutlass.Float32(float("-inf"))
    other_denominator = cutlass.Float32(0)
    panel = cute.make_rmem_tensor((query_columns,), cutlass.Float32)
    for query in cutlass.range_constexpr(query_columns):
        panel[query] = cutlass.Float32(0)
    if split:
        pair = cutlass.Int32(walk[WALK_ROLE]) == PART_MERGE_PAIR
    if pair:
        other_maximum, other_denominator = pair_partial(
            workspace, panel, pair_head, thread, record_bytes, query_columns
        )
        if thread < query_columns:
            # The owner read its row sums above and nothing else reads
            # the scratch rows before the next barrier.
            merge_pair_rows(
                state,
                scratch,
                thread,
                total,
                other_maximum,
                other_denominator,
                base2_scores,
                fast_math,
                output_reciprocal,
                output_scale_log2,
            )
    cute.arch.barrier(barrier_id=1 + pair_head * 2, number_of_threads=128)
    if pair:
        # The barrier published the weights, and the other part's loads
        # land while the last PV still runs.
        merge_pair_other(scratch, panel, query_columns)
    _drain_last_pv(pv_done, phase, count, sleep_waits, probability_schedule)
    # O is final once the last PV completed (or was zeroed for an empty range).
    # This part's own term of a unit in two parts is folded onto the
    # other's eight columns at a time, so the two panels never are in
    # registers together.
    if pair:
        for chunk in cutlass.range_constexpr(query_columns // 8):
            columns = prims.tcgen05_ld(
                "32x32b",
                cutlass.inttoptr(output_address + chunk * 8, 6, cutlass.Float32),
                num=8,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            merge_pair_own(scratch, panel, columns, chunk * 8, 8)
    else:
        own = load_query_panel(output_address, query_columns)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        for query in cutlass.range_constexpr(query_columns):
            panel[query] = cutlass.Float32(own[query])
    return panel


@cute.jit
def _compute_store(
    output: cute.Tensor,
    lse: cute.Tensor | None,
    state: cute.Tensor,
    result: cute.Tensor | cutlass.Array,
    tile: cutlass.Int32,
    thread: cutlass.Int32,
    query_rows: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    group: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    io_layouts: cutlass.Constexpr,
    base2_scores: cutlass.Constexpr,
    output_reciprocal: cutlass.Constexpr,
    maximum_output: cute.Tensor | None,
    denominator_output: cute.Tensor | None,
    statistic_layouts: cutlass.Constexpr,
    store_policy: cutlass.Constexpr = "default",
) -> None:
    """Store a work item's outputs for ``tile`` from its O panel and row state.

    ``store_policy`` is the config's ``cute_store_policy``. ``streaming``
    stores the O elements with the streaming cache operator and
    ``l2_evict_last`` with the L2 evict-last cache hint, under one policy
    word the thread makes ahead of its stores: the same values at the same
    addresses, as many stores in the same order (``store_output_rows``),
    each value formed ahead of the first store, since a store's instruction
    is opaque to the compiler and would keep the rows' loads apart.
    ``default`` keeps the plain stores, and the per-row stores (LSE, raw
    statistics) are plain under each policy.
    """
    if cutlass.const_expr(store_policy != "default"):
        values = []
        for query in cutlass.range_constexpr(query_rows):
            if cutlass.const_expr(maximum_output is not None):
                values.append(cutlass.Float32(result[query]))
            else:
                values.append(
                    normalize_output(result[query], state[3, query], output_reciprocal)
                )
        store_output_rows(
            output, tile, thread, tuple(values), heads, group, io_layouts, store_policy
        )
    else:
        for query in cutlass.range_constexpr(query_columns):
            if cutlass.const_expr(query < query_rows):
                if cutlass.const_expr(maximum_output is not None):
                    value = cutlass.Float32(result[query])
                else:
                    value = cutlass.BFloat16(
                        normalize_output(
                            result[query], state[3, query], output_reciprocal
                        )
                    )
                store_output_element(
                    output,
                    tile,
                    query,
                    thread,
                    value,
                    heads,
                    group,
                    io_layouts,
                )
    if cutlass.const_expr(maximum_output is not None):
        assert maximum_output is not None and denominator_output is not None
        if thread < query_rows:
            assert cutlass.const_expr(statistic_layouts is not None)
            store_row_element(
                maximum_output,
                tile,
                thread,
                cutlass.Float32(state[0, thread]),
                heads,
                group,
                statistic_layouts[0],
            )
            store_row_element(
                denominator_output,
                tile,
                thread,
                cutlass.Float32(state[1, thread]),
                heads,
                group,
                statistic_layouts[1],
            )
    if cutlass.const_expr(lse is not None):
        assert lse is not None
        if thread < query_rows:
            maximum = cutlass.Float32(state[0, thread])
            if cutlass.const_expr(base2_scores):
                maximum = maximum * cutlass.Float32(math.log(2))
            store_row_element(
                lse,
                tile,
                thread,
                maximum + cute.math.log(state[1, thread], fastmath=fast_math),
                heads,
                group,
                None if io_layouts is None else io_layouts[2],
            )


@cute.jit
def _compute_group(
    base: cutlass.Int32,
    query_columns: cutlass.Constexpr,
    probability_schedule: cutlass.Constexpr,
    output_offset: cutlass.Constexpr,
    compute_registers: cutlass.Constexpr,
) -> tuple[
    cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32 | None
]:
    """A compute role's entry: take the group's register quota and return
    this thread's index, warp and lane in its group of 128 with the TMEM
    addresses of the group's O panel and, under the TMEM probability
    schedule, of its denominators."""
    cute.arch.setmaxregister_increase(compute_registers)
    thread = cute.arch.thread_idx()[0] % 128
    warp = cute.arch.make_warp_uniform(thread // 32)
    lane = thread % 32
    output_address = (warp * 32 << 16) | (base + output_offset)
    denominator_address = None
    if cutlass.const_expr(probability_schedule == "early_tmem"):
        denominator_address = (warp * 32 << 16) | (base + 128 + 2 * query_columns)
    return thread, warp, lane, output_address, denominator_address


@cute.jit
def _arrive_at_retirement() -> None:
    """A compute thread's arrival at the allocation's early retirement,
    behind its final read of the slab: barrier 5, which the QK role waits
    at before it frees the allocation (``_retire_tensor_memory``)."""
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_arrive(5, thread_count=288)


@cute.jit
def _compute_head(
    output: cute.Tensor,
    lse: cute.Tensor | None,
    probability: cute.Tensor,
    state: cute.Tensor,
    scratch: cute.Tensor,
    qk_done: cute.Pointer,
    score_free: cute.Pointer,
    probability_ready: cute.Pointer,
    pv_done: cute.Pointer,
    base: cutlass.Int32,
    tile: cutlass.Int32,
    pair_head: cutlass.Int32,
    first_page: cutlass.Int64,
    last_page: cutlass.Int64,
    count: cutlass.Int64,
    length: cutlass.Int32,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    query_rows: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    window: cutlass.Constexpr,
    score_scale: cutlass.Float32,
    fast_math: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    io_layouts: cutlass.Constexpr,
    reference_scores: cutlass.Constexpr,
    base2_scores: cutlass.Constexpr,
    page_walk: cutlass.Constexpr,
    output_reciprocal: cutlass.Constexpr,
    output_scale_log2: cutlass.Constexpr,
    sleep_waits: cutlass.Constexpr,
    capture_pdl_wait_mode: cutlass.Int32,
    maximum_output: cute.Tensor | None,
    denominator_output: cute.Tensor | None,
    statistic_layouts: cutlass.Constexpr,
    coefficients_capped: cutlass.Constexpr,
    probability_schedule: cutlass.Constexpr,
    output_offset: cutlass.Constexpr,
    rowwise_rescale_threshold: cutlass.Constexpr,
    early_tmem_retirement: cutlass.Constexpr,
    headwide_rescale: cutlass.Constexpr,
    key_clip: cutlass.Constexpr,
    compute_registers: cutlass.Constexpr,
    store_policy: cutlass.Constexpr = "default",
) -> None:
    """Run one compute group over the CTA's work item and store its outputs."""
    thread, warp, lane, output_address, denominator_address = _compute_group(
        base, query_columns, probability_schedule, output_offset, compute_registers
    )
    partials, reference_invalid = _compute_begin(
        output_address,
        denominator_address,
        pair_head,
        query_columns,
        probability_schedule,
    )
    _compute_pages(
        probability,
        state,
        scratch,
        qk_done,
        score_free,
        probability_ready,
        pv_done,
        base,
        pair_head,
        thread,
        lane,
        output_address,
        denominator_address,
        partials,
        reference_invalid,
        first_page,
        last_page,
        cutlass.Int32(0),
        cutlass.Int64(0),
        count,
        length,
        query_len,
        group,
        window,
        score_scale,
        fast_math,
        coordinate_dtype,
        query_columns,
        reference_scores,
        base2_scores,
        page_walk,
        sleep_waits,
        coefficients_capped,
        probability_schedule,
        rowwise_rescale_threshold,
        headwide_rescale,
        key_clip,
    )
    result = _compute_reduce(
        state,
        scratch,
        pv_done,
        pair_head,
        thread,
        warp,
        lane,
        output_address,
        denominator_address,
        partials,
        cutlass.Int32(0),
        count,
        query_columns,
        output_reciprocal,
        output_scale_log2,
        sleep_waits,
        maximum_output,
        probability_schedule,
    )
    if cutlass.const_expr(early_tmem_retirement):
        # Both heads finish their last PV and O reads before either slab is
        # freed. Output math/stores use only the retained register copy.
        _arrive_at_retirement()
    # All global stores of this audited schedule are below this point. Even an
    # empty attention interval reaches it. Other roles only read global memory.
    if capture_pdl_wait_mode != 0:
        cute.arch.griddepcontrol_wait()
    _compute_store(
        output,
        lse,
        state,
        result,
        tile,
        thread,
        query_rows,
        heads,
        group,
        fast_math,
        query_columns,
        io_layouts,
        base2_scores,
        output_reciprocal,
        maximum_output,
        denominator_output,
        statistic_layouts,
        store_policy,
    )
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def _compute_units(
    output: cute.Tensor,
    lse: cute.Tensor | None,
    probability: cute.Tensor,
    state: cute.Tensor,
    scratch: cute.Tensor,
    qk_done: cute.Pointer,
    score_free: cute.Pointer,
    probability_ready: cute.Pointer,
    pv_done: cute.Pointer,
    base: cutlass.Int32,
    pair_head: cutlass.Int32,
    walk: cute.Tensor,
    mail: cute.Tensor | None,
    plan: cute.Tensor,
    workspace: cutlass.Int64,
    first_unit: cutlass.Int32,
    last_unit: cutlass.Int32,
    lengths: cute.Tensor | None,
    unit_bounds: cutlass.Constexpr,
    unit_bounds_scanned: cutlass.Constexpr,
    unit_bounds_args: tuple,
    page_limit: cutlass.Constexpr,
    partition: cutlass.Constexpr,
    record_bytes: cutlass.Constexpr,
    slot_bits: cutlass.Constexpr,
    skewed: cutlass.Constexpr,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    query_rows: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    window: cutlass.Constexpr,
    score_scale: cutlass.Float32,
    fast_math: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    io_layouts: cutlass.Constexpr,
    reference_scores: cutlass.Constexpr,
    base2_scores: cutlass.Constexpr,
    page_walk: cutlass.Constexpr,
    output_reciprocal: cutlass.Constexpr,
    output_scale_log2: cutlass.Constexpr,
    sleep_waits: cutlass.Constexpr,
    capture_pdl_wait_mode: cutlass.Int32,
    maximum_output: cute.Tensor | None,
    denominator_output: cute.Tensor | None,
    statistic_layouts: cutlass.Constexpr,
    coefficients_capped: cutlass.Constexpr,
    probability_schedule: cutlass.Constexpr,
    output_offset: cutlass.Constexpr,
    rowwise_rescale_threshold: cutlass.Constexpr,
    early_tmem_retirement: cutlass.Constexpr,
    headwide_rescale: cutlass.Constexpr,
    key_clip: cutlass.Constexpr,
    compute_registers: cutlass.Constexpr,
    store_policy: cutlass.Constexpr = "default",
) -> None:
    """Run one compute group over the CTA's units and store each unit's outputs.

    Every unit runs what a CTA of its own would: empty panels and row state,
    the recurrence over its pages, the reduction and the stores. Only the
    barrier phases continue from the pages the CTA streamed before it.

    The recurrence runs at the register quota's limit, so what the group
    needs only after a unit's pages (the unit, the end of the range and the
    unit's first item) waits in ``walk``, the group's record in shared
    memory, instead of in registers.

    Under the balanced partition the CTA runs the steps of a unit whose
    page slots lie in its range (``plan``): all of them, or a part of a
    unit that other CTAs run the rest of. A part runs the same recurrence
    from an empty state and then, instead of storing, exchanges its
    unnormalized panel and row statistics with the unit's other parts
    through ``workspace`` (``_paged_attention_partition``): it publishes
    them, or, as the last part to arrive, merges all parts and stores the
    unit. The workspace is global memory other launches use too, so a part
    waits for the launch's dependencies before its first access to it; the
    role of the part is decided by one thread and reaches the group through
    ``walk`` at the reduction's first barrier.

    The group meets at named barriers of the aligned form (``bar.sync``).
    A barrier instruction "causes executing thread to wait for all
    non-exited threads from its warp" (PTX ISA 9.5, section 9.7.15.1), so a
    block that only some lanes of a warp run ahead of a group barrier (one
    thread's record, the row owners' state and stores) needs no warp
    synchronization of its own.
    """
    thread, warp, lane, output_address, denominator_address = _compute_group(
        base, query_columns, probability_schedule, output_offset, compute_registers
    )
    phase = cutlass.Int32(0)
    unit = first_unit
    end = last_unit
    # The header slot of the unit: only the balanced partition's clip reads it.
    header = cutlass.Int32(0)
    if cutlass.const_expr(partition == "balanced"):
        # A part's exchange needs most of the quota once its pages are done.
        # The range read again here, behind the quota, is not kept in
        # registers from the role's start through every unit.
        unit = cutlass.Int32(plan[PLAN_FIRST])
        end = cutlass.Int32(plan[PLAN_LAST])
        header = cutlass.Int32(plan[PLAN_HEADER])
    while unit < end:
        # ``count`` is the pages of this CTA's steps, ``unit_pages`` those of
        # the unit. The CTA's slots stay in ``plan``, not in registers.
        item, first_page, last_page, length, unit_pages, step0, count = _unit_range(
            unit,
            header,
            None,
            None,
            plan,
            lengths,
            unit_bounds,
            unit_bounds_scanned,
            unit_bounds_args,
            page_limit,
            partition,
        )
        partials, reference_invalid = _compute_begin(
            output_address,
            denominator_address,
            pair_head,
            query_columns,
            probability_schedule,
        )
        # The stores of the previous unit read every thread's normalization
        # row: the group barrier above orders them before these rewrites, and
        # the one below orders the rewrites before the first softmax, which
        # reads the rows across threads.
        if thread < query_columns:
            state[0, thread] = cutlass.Float32(float("-inf"))
            state[1, thread] = cutlass.Float32(0)
            state[2, thread] = cutlass.Float32(0)
            state[3, thread] = cutlass.Float32(1)
        if thread == 0:
            # One writer per record, between the unit's two group barriers.
            walk[WALK_UNIT] = unit
            walk[WALK_END] = end
            walk[WALK_ITEM] = item
            if cutlass.const_expr(partition == "balanced"):
                walk[WALK_HEADER] = header
                walk[WALK_PAGES] = cutlass.Int32(unit_pages)
                walk[WALK_STEP] = cutlass.Int32(step0)
                if unit == cutlass.Int32(plan[PLAN_FIRST]):
                    # The group has posted nothing yet; ``split_arrival``
                    # notes the part it posts.
                    walk[WALK_POSTED] = cutlass.Int32(0)
        cute.arch.barrier(barrier_id=2 + pair_head * 2, number_of_threads=128)
        _compute_pages(
            probability,
            state,
            scratch,
            qk_done,
            score_free,
            probability_ready,
            pv_done,
            base,
            pair_head,
            thread,
            lane,
            output_address,
            denominator_address,
            partials,
            reference_invalid,
            first_page,
            last_page,
            phase,
            step0,
            count,
            length,
            query_len,
            group,
            window,
            score_scale,
            fast_math,
            coordinate_dtype,
            query_columns,
            reference_scores,
            base2_scores,
            page_walk,
            sleep_waits,
            coefficients_capped,
            probability_schedule,
            rowwise_rescale_threshold,
            headwide_rescale,
            key_clip,
            walk if partition == "balanced" else None,
        )
        unit = cutlass.Int32(walk[WALK_UNIT])
        end = cutlass.Int32(walk[WALK_END])
        item = cutlass.Int32(walk[WALK_ITEM])
        split = cutlass.Boolean(False)
        if cutlass.const_expr(partition == "balanced"):
            # Only the balanced partition posts to the spare warp.
            assert mail is not None
            header = cutlass.Int32(walk[WALK_HEADER])
            planned_pages = cutlass.Int32(walk[WALK_PAGES])
            # A part: other CTAs run the unit's other steps.
            split = cutlass.Int32(count) != planned_pages
            if split:
                # The first access to the workspace is the read of the
                # unit's counter, so the wait comes first. The reduction
                # below still overlaps the part's last PV.
                if capture_pdl_wait_mode != 0:
                    cute.arch.griddepcontrol_wait()
                # One lane decides, and the reduction's first group barrier
                # publishes what it wrote. Keep this block free of waits
                # for another CTA, as everything in the exchange is
                # (``split_arrival``): the deciding lane must not wait on
                # another CTA, and a source test holds the block to that. A
                # part that must wait for another CTA waits as a whole warp
                # or leaves it to the spare warp.
                if thread == 0:
                    split_arrival(
                        workspace,
                        walk,
                        mail,
                        plan,
                        header,
                        planned_pages,
                        cutlass.Int32(walk[WALK_STEP]),
                        cutlass.Int32(count),
                        item,
                        pair_head,
                        record_bytes,
                        slot_bits,
                        skewed,
                    )
            header = header + planned_pages + 1
        if cutlass.const_expr(partition == "balanced"):
            result = _reduce_part(
                state,
                scratch,
                pv_done,
                pair_head,
                thread,
                warp,
                lane,
                output_address,
                denominator_address,
                partials,
                phase,
                count,
                query_columns,
                output_reciprocal,
                output_scale_log2,
                sleep_waits,
                maximum_output,
                probability_schedule,
                walk,
                split,
                workspace,
                record_bytes,
                base2_scores,
                fast_math,
            )
        else:
            result = _compute_reduce(
                state,
                scratch,
                pv_done,
                pair_head,
                thread,
                warp,
                lane,
                output_address,
                denominator_address,
                partials,
                phase,
                count,
                query_columns,
                output_reciprocal,
                output_scale_log2,
                sleep_waits,
                maximum_output,
                probability_schedule,
            )
        if cutlass.const_expr(early_tmem_retirement):
            # The last unit's O read was this thread's final use of the slab.
            if unit == end - 1:
                _arrive_at_retirement()
        # Every global store of a unit is below its wait. A repeated wait
        # returns at once.
        if capture_pdl_wait_mode != 0:
            cute.arch.griddepcontrol_wait()
        # A whole unit stores its outputs; a part of a split unit stores
        # the unit's only if it merged the parts.
        stores = cutlass.Boolean(True)
        if cutlass.const_expr(partition == "balanced"):
            if split:
                # The reduction's barriers published the part's role, and
                # the reduction merged a unit in two parts already.
                role = cutlass.Int32(walk[WALK_ROLE])
                if role != PART_MERGE_PAIR:
                    # Also a part that merges a unit in more parts stores
                    # its own: the fold reads every part from the workspace.
                    store_part(
                        workspace,
                        walk,
                        state,
                        result,
                        pair_head,
                        thread,
                        record_bytes,
                        query_columns,
                    )
                    if role == PART_POST:
                        # The spare warp joins these stores here and arrives
                        # for them; this group goes on with its next unit.
                        prims.barrier_cta_arrive(
                            POST_BARRIER + pair_head, thread_count=POST_THREADS
                        )
                    if role == PART_PUBLISH:
                        role = arrive_part(
                            workspace, walk, pair_head, thread, record_bytes
                        )
                    stores = role == PART_MERGE
                    if stores:
                        merge_parts(
                            workspace,
                            cutlass.Int32(walk[WALK_LOWEST]),
                            cutlass.Int32(walk[WALK_PARTS]),
                            state,
                            scratch,
                            result,
                            pair_head,
                            thread,
                            record_bytes,
                            query_columns,
                            base2_scores,
                            fast_math,
                            output_reciprocal,
                            output_scale_log2,
                        )
        if stores:
            _compute_store(
                output,
                lse,
                state,
                result,
                item + pair_head,
                thread,
                query_rows,
                heads,
                group,
                fast_math,
                query_columns,
                io_layouts,
                base2_scores,
                output_reciprocal,
                maximum_output,
                denominator_output,
                statistic_layouts,
                store_policy,
            )
        # Only the parity of the pages streamed so far selects a phase.
        phase = (phase + cutlass.Int32(count & 1)) & 1
        unit = unit + 1
    no_units = first_unit >= last_unit
    if cutlass.const_expr(partition == "balanced"):
        no_units = cutlass.Int32(plan[PLAN_FIRST]) >= cutlass.Int32(plan[PLAN_LAST])
    if no_units:
        # A CTA beyond the units still joins the allocation's retirement and
        # waits for its predecessor before it exits.
        if cutlass.const_expr(early_tmem_retirement):
            _arrive_at_retirement()
        if capture_pdl_wait_mode != 0:
            cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(partition == "balanced"):
        assert mail is not None
        # The CTA's first unit is the one it can run a part of that is not
        # the unit's lowest: its range then starts behind the unit's first
        # page slot. That part was posted to the spare warp, and the thread
        # that decided so noted it in the group's record
        # (``split_arrival``), which the barriers of every unit since have
        # published. A group without one posts that, so the warp hears from
        # each group once, by one decision.
        posted = cutlass.Boolean(False)
        if not no_units:
            posted = cutlass.Int32(walk[WALK_POSTED]) != 0
        if not posted:
            if thread == 0:
                mail[MAIL_KIND] = cutlass.Int32(MAIL_NONE)
            prims.barrier_cta_arrive(
                POST_BARRIER + pair_head, thread_count=POST_THREADS
            )
        prims.barrier_cta_sync(POSTED_BARRIER + pair_head, thread_count=POST_THREADS)
        if cutlass.Int32(mail[MAIL_KIND]) == MAIL_LAST:
            # The posted part arrived last, behind the unit's lowest part:
            # this group merges the unit, from the workspace alone.
            late = cute.make_rmem_tensor((query_columns,), cutlass.Float32)
            for query in cutlass.range_constexpr(query_columns):
                late[query] = cutlass.Float32(0)
            merge_parts(
                workspace,
                cutlass.Int32(mail[MAIL_LOWEST]),
                cutlass.Int32(mail[MAIL_PARTS]),
                state,
                scratch,
                late,
                pair_head,
                thread,
                record_bytes,
                query_columns,
                base2_scores,
                fast_math,
                output_reciprocal,
                output_scale_log2,
            )
            _compute_store(
                output,
                lse,
                state,
                late,
                cutlass.Int32(mail[MAIL_ITEM]) + pair_head,
                thread,
                query_rows,
                heads,
                group,
                fast_math,
                query_columns,
                io_layouts,
                base2_scores,
                output_reciprocal,
                maximum_output,
                denominator_output,
                statistic_layouts,
                store_policy,
            )
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


def _ring_slot(
    step: cutlass.Int64 | int, head: cutlass.Int32 | int, depth: int
) -> cutlass.Int64 | cutlass.Int32 | int:
    """The slot of a raw ring of ``depth`` pages per head, K's or V's, that
    the stream's step ``step`` stages head ``head`` in: the pages take the
    depth's slots in turn, two heads each. Plain Python over a static depth,
    so each call traces the expression of its depth alone, the default's
    unchanged."""
    if depth == PAIRED_DEFAULT_VALUE_RING:
        return 2 * (step & 1) + head
    return 2 * (step % depth) + head


def _ring_phase(step: cutlass.Int64 | int, depth: int) -> cutlass.Int64 | int:
    """The phase of that slot's barriers at step ``step``: its parity counts
    the earlier steps that used the slot."""
    if depth == PAIRED_DEFAULT_VALUE_RING:
        return step // 2 & 1
    return (step // depth) & 1


@cute.jit
def _prepare_value_metadata(
    scales: cute.Pointer,
    free: cute.Pointer,
    flags: cute.Tensor,
    ready: cute.Pointer,
    source: cute.Tensor,
    step: cutlass.Int64,
    pair_head: cutlass.Int32,
    physical: cutlass.Int32,
    first_head: cutlass.Int32,
    sleep_waits: cutlass.Constexpr,
    read_overlap: cutlass.Constexpr = False,
) -> None:
    """Publish a scale tile, owned until every lane of all four V warps releases it.

    Flags encode copied nonzero scales (0), a positive-zero value tile (1),
    or an invalid descriptor coordinate requiring the original two TMAs (2).
    The source is the proved (key, head, subpage, page) Int32-word view.
    """
    lane = cute.arch.lane_idx()
    ring = 2 * (step & 1) + pair_head
    if cutlass.const_expr(not read_overlap):
        _wait_non_mma(free + ring, (step // 2 & 1) ^ 1, sleep_waits)
    # Match the original TMA's Int32 head arithmetic before widening offsets.
    encoded_head = cutlass.Int32(first_head + pair_head)
    valid = (
        (cutlass.Int64(physical) >= 0)
        & (cutlass.Int64(physical) < cute.size(source, mode=[3]))
        & (cutlass.Int64(encoded_head) >= 0)
        & (cutlass.Int64(encoded_head) < cute.size(source, mode=[1]))
    )
    if valid:
        offset = cute.crd2idx(
            (
                cutlass.Int64(lane) * 4,
                cutlass.Int64(encoded_head),
                cutlass.Int64(0),
                cutlass.Int64(physical),
            ),
            source.layout,
        )
        # All outer strides and the starting key are multiples of four words.
        words = _load_words4((source.iterator + cute.assume(offset, divby=4)).align(16))
        zero = cute.arch.vote_all_sync((words[0] | words[1] | words[2] | words[3]) == 0)
        # Global inputs remain immutable throughout this kernel. Only those
        # reads may overlap the previous consumer; every shared effect below
        # still waits for all 128 reader releases on this exact slot.
        if cutlass.const_expr(read_overlap):
            _wait_non_mma(free + ring, (step // 2 & 1) ^ 1, sleep_waits)
        shared = cute.make_tensor(
            cute.recast_ptr(scales + ring * 512, dtype=cutlass.Int32),
            cute.make_layout((128,)),
        )
        if not zero:
            for index in cutlass.range_constexpr(4):
                shared[lane * 4 + index] = words[index]
            # A later invalid coordinate reuses this slot through TMA's async
            # proxy. Each writer orders its generic stores before that handoff;
            # the warp join and release/acquire barriers alone do not do so.
            cute.arch.fence_proxy(kind="async.shared", space="cta")
        if lane == 0:
            flags[ring] = cutlass.Int32(zero)
    else:
        if cutlass.const_expr(read_overlap):
            _wait_non_mma(free + ring, (step // 2 & 1) ^ 1, sleep_waits)
        if lane == 0:
            flags[ring] = cutlass.Int32(2)
    # Publish each writer directly. The lane-zero release preceded by a warp
    # join passes numerical tests, but its SASS has no WARPSYNC and racecheck
    # reports other lanes' accesses. All 32 arrivals make ownership explicit;
    # this is not evidence of a numerical failure or invalid PTX ordering.
    prims.mbarrier_arrive(ready + ring)


@cute.jit
def _enqueue_operand_tile(
    raw: cute.Pointer,
    scales: cute.Pointer,
    ready: cute.Pointer,
    free: cute.Pointer,
    operand_map: experimental_cuda.TensorMap,
    scale_map: experimental_cuda.TensorMap,
    step: cutlass.Int64,
    pair_head: cutlass.Int32,
    physical: cutlass.Int32,
    first_head: cutlass.Int32,
    l2_hints: cutlass.Constexpr = ("", ""),
    sleep_waits: cutlass.Constexpr = False,
    metadata_ready: cute.Pointer | None = None,
    metadata_flags: cute.Tensor | None = None,
    key: cutlass.Int32 = cutlass.Int32(0),  # noqa: B008
    depth: cutlass.Constexpr = PAIRED_DEFAULT_VALUE_RING,
) -> None:
    """Request one raw tile of the stream's step ``step`` into its slot of a
    ring of ``depth`` pages per head, once the slot is free."""
    # The value-metadata actor walks the ring at its default depth.
    assert cutlass.const_expr(
        depth == PAIRED_DEFAULT_VALUE_RING or metadata_ready is None
    )
    ring = _ring_slot(step, pair_head, depth)
    if cutlass.const_expr(metadata_ready is not None):
        # Acquiring metadata also acquires the preceding raw-free phase. The
        # loader retains the original V order for every lag/drain schedule.
        assert metadata_ready is not None and metadata_flags is not None
        _wait_non_mma(metadata_ready + ring, _ring_phase(step, depth), sleep_waits)
        flag = metadata_flags[ring]
        if flag == 1:
            prims.mbarrier_arrive(ready + ring)
        elif flag == 2:
            enqueue_paged_tma_operand(
                operand_map,
                scale_map,
                raw + cute.assume(ring * 16384, divby=16384),
                scales + ring * 512,
                ready + ring,
                physical,
                cutlass.Int32(0),
                first_head + pair_head,
                128,
                l2_hints=l2_hints,
            )
        else:
            enqueue_paged_tma_operand(
                operand_map,
                scale_map,
                raw + cute.assume(ring * 16384, divby=16384),
                scales + ring * 512,
                ready + ring,
                physical,
                cutlass.Int32(0),
                first_head + pair_head,
                128,
                l2_hints=l2_hints,
                include_scales=False,
            )
    else:
        _wait_non_mma(free + ring, _ring_phase(step, depth) ^ 1, sleep_waits)
        enqueue_paged_tma_operand(
            operand_map,
            scale_map,
            raw + cute.assume(ring * 16384, divby=16384),
            scales + ring * 512,
            ready + ring,
            physical,
            cutlass.Int32(0),
            first_head + pair_head,
            128,
            l2_hints=l2_hints,
            key=key,
        )


def _allocate_pipeline_shared_memory(
    smem: utils.SmemAllocator, arrays: tuple[SharedArray, ...], parts: frozenset[str]
) -> SimpleNamespace:
    """Allocate the body's shared-memory table ``arrays``
    (``paged_attention_shared_memory.pipeline_shared_memory``, which the
    config contract replays against the device) in its order, for a body
    with the optional ``parts``: one attribute per array, ``None`` for those
    of a part the body lacks. An entry names its element type as the DSL
    does, and the table's byte accounting is of that type. The body
    allocates nothing else."""
    allocated = {}
    for array in arrays:
        dtype = getattr(cutlass, array.dtype)
        assert dtype.width == 8 * array.width
        allocated[array.name] = (
            smem.allocate_array(dtype, array.count, byte_alignment=array.alignment)
            if array.guard is None or array.guard in parts
            else None
        )
    return SimpleNamespace(**allocated)


@cute.jit
def _convert_keys(
    raw_k: cute.Pointer,
    scale_k: cute.Pointer,
    raw_k_ready: cute.Pointer,
    raw_k_free: cute.Pointer,
    key_ready: cute.Pointer,
    key_free: cute.Pointer,
    raw_layout: cute.ComposedLayout,
    scale_layout: cute.Layout,
    base: cutlass.Int32,
    row: cutlass.Int32,
    lane: cutlass.Int32,
    count: cutlass.Int64,
    key_post_scale_log2: cutlass.Constexpr,
    sleep_waits: cutlass.Constexpr,
) -> None:
    """Convert the K tiles of the ``count`` pages the CTA streams."""
    for step in cutlass.range(count):
        for pair_head in cutlass.range(2, unroll=1):
            head = cute.arch.make_warp_uniform(pair_head)
            # The raw K ring keeps its own depth, which the loader's K
            # requests name.
            ring = _ring_slot(step, head, KEY_RING_PAGES)
            _wait_non_mma(
                raw_k_ready + ring,
                _ring_phase(step, KEY_RING_PAGES),
                sleep_waits,
            )
            _wait_non_mma(key_free + head, (step & 1) ^ 1, sleep_waits)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            staged_scales = cute.make_tensor(
                (scale_k + ring * 512).align(16), scale_layout
            )
            assert _packed_scale_layout_supported(staged_scales)
            stage_scaled_s8_rows_to_tmem(
                cute.make_tensor(
                    raw_k + cute.assume(ring * 16384, divby=16384), raw_layout
                ),
                staged_scales,
                row,
                base + head * 256,
                128,
                32,
                post_scale_log2=key_post_scale_log2,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.sync_warp()
            if lane == 0:
                prims.mbarrier_arrive(key_ready + head)
                prims.mbarrier_arrive(raw_k_free + ring)


@cute.jit
def _convert_values(
    raw_v: cute.Pointer,
    scale_v: cute.Pointer,
    raw_v_ready: cute.Pointer,
    raw_v_free: cute.Pointer,
    value_ready: cute.Pointer,
    value_free: cute.Pointer,
    raw_layout: cute.ComposedLayout,
    scale_layout: cute.Layout,
    metadata_flags: cute.Tensor | None,
    value_metadata_scales: cute.Tensor | None,
    base: cutlass.Int32,
    dimension: cutlass.Int32,
    lane: cutlass.Int32,
    count: cutlass.Int64,
    value_post_scale_log2: cutlass.Constexpr,
    value_positive_zero: cutlass.Constexpr,
    sleep_waits: cutlass.Constexpr,
    value_ring: cutlass.Constexpr = PAIRED_DEFAULT_VALUE_RING,
) -> None:
    """Convert the V tiles of the ``count`` pages the CTA streams, from a
    raw ring of ``value_ring`` pages per head."""
    for step in cutlass.range(count):
        for pair_head in cutlass.range(2, unroll=1):
            head = cute.arch.make_warp_uniform(pair_head)
            ring = _ring_slot(step, head, value_ring)
            _wait_non_mma(
                raw_v_ready + ring, _ring_phase(step, value_ring), sleep_waits
            )
            _wait_non_mma(value_free + head, (step & 1) ^ 1, sleep_waits)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            staged_scales = cute.make_tensor(
                (scale_v + ring * 512).align(16), scale_layout
            )
            assert _packed_scale_layout_supported(staged_scales)
            zero_tile = False
            if cutlass.const_expr(value_metadata_scales is not None):
                assert metadata_flags is not None
                zero_tile = metadata_flags[ring] == 1
            if zero_tile:
                stage_positive_zero_value_tile(dimension, base + head * 256 + 64)
            else:
                stage_scaled_s8_transpose_to_tmem(
                    cute.make_tensor(
                        raw_v + cute.assume(ring * 16384, divby=16384), raw_layout
                    ),
                    staged_scales,
                    dimension,
                    lane,
                    base + head * 256 + 64,
                    128,
                    post_scale_log2=value_post_scale_log2,
                    positive_zero=value_positive_zero,
                )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.sync_warp()
            if lane == 0:
                prims.mbarrier_arrive(value_ready + head)
            if cutlass.const_expr(value_metadata_scales is not None):
                # Release each lane's flag/scale reads explicitly for the
                # metadata actor, as with the producer's release above.
                # This opt-in path uses 128 arrivals instead of four;
                # TMEM publication keeps its four warp arrivals above.
                prims.mbarrier_arrive(raw_v_free + ring)
            elif lane == 0:
                prims.mbarrier_arrive(raw_v_free + ring)


@cute.jit
def _issue_qk(
    query_raw: cute.Pointer,
    key_ready: cute.Pointer,
    key_free: cute.Pointer,
    score_free: cute.Pointer,
    score_ready: cute.Pointer,
    base: cutlass.Int32,
    g0: cutlass.Int64,
    count: cutlass.Int64,
    query_columns: cutlass.Constexpr,
) -> None:
    """Issue QK for ``count`` pages against the query panels at ``query_raw``.

    ``g0`` is the number of pages the CTA streamed before these: the barrier
    phases follow the stream, not the work item.
    """
    for step in cutlass.range(count):
        for head in cutlass.range_constexpr(2):
            mbar_wait(score_free + head, ((g0 + step) & 1) ^ 1)
            mbar_wait(key_ready + head, (g0 + step) & 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            mma_128xnx128(
                base + head * 256,
                base + head * 256 + 128,
                query_raw + head * (query_columns * 128),
                score_ready + head,
                False,
                query_columns,
            )
            if prims.elect_sync():
                prims.tcgen05_commit(key_free + head, group=prims.CTAGroup.CTA_1)


@cute.jit
def _retire_tensor_memory(base: cutlass.Int32) -> None:
    """The QK role's early retirement of the allocation at ``base``.

    One allocation-wide phase at barrier 5: the two compute groups arrive
    after their final reads, and this warp, which has issued its last MMA,
    waits for them and frees the allocation.
    """
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(5, thread_count=288)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    prims.tcgen05_dealloc(
        cutlass.inttoptr(base, 6, cutlass.Int32),
        512,
        group=prims.CTAGroup.CTA_1,
    )


@cute.jit
def _issue_qk_units(
    query_map: experimental_cuda.TensorMap,
    query_raw: cute.Pointer,
    query_ready: cute.Pointer,
    query_free: cute.Pointer,
    key_ready: cute.Pointer,
    key_free: cute.Pointer,
    score_free: cute.Pointer,
    score_ready: cute.Pointer,
    base: cutlass.Int32,
    lane: cutlass.Int32,
    first_unit: cutlass.Int32,
    last_unit: cutlass.Int32,
    begin: cutlass.Int32,
    stop: cutlass.Int32,
    first_header: cutlass.Int32,
    lengths: cute.Tensor | None,
    unit_bounds: cutlass.Constexpr,
    unit_bounds_scanned: cutlass.Constexpr,
    unit_bounds_args: tuple,
    page_limit: cutlass.Constexpr,
    partition: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    q_stages: cutlass.Constexpr,
    early_tmem_retirement: cutlass.Constexpr,
) -> None:
    """Issue QK for the CTA's units, then retire the allocation if asked to.

    Under the balanced partition a unit's pages are those of its steps in
    the CTA's slots ``[begin, stop)``; ``first_header`` is the header slot
    of the first unit. The units that stream pages take the query stages in
    turn; a unit without pages issues no QK and has no queries. The first of them gets
    its queries from this role, as a one-unit CTA's does, so that the
    loader's first requests are pages; ``_load_units`` fills the stages
    from the second on. A stage is released once the unit's last MMA of
    both heads has retired, so the loader refills it only when nothing
    reads it any more. Every query copy requested, here or by the loader,
    is waited for here, so none is outstanding when this role frees TMEM
    and returns.
    """
    streamed = cutlass.Int64(0)
    loaded = cutlass.Int32(0)
    header = first_header
    for unit in cutlass.range(first_unit, last_unit, 1):
        item, _first_page, _last_page, _length, unit_pages, _step0, count = _unit_range(
            unit,
            header,
            begin,
            stop,
            None,
            lengths,
            unit_bounds,
            unit_bounds_scanned,
            unit_bounds_args,
            page_limit,
            partition,
        )
        if cutlass.const_expr(partition == "balanced"):
            header = header + cutlass.Int32(unit_pages) + 1
        if count > 0:
            stage = loaded % q_stages
            if loaded == 0:
                # The first fill of the first stage. The loader initialized
                # the stage's barriers ahead of the setup rendezvous, and
                # only its second fill, behind this unit's release, touches
                # the stage again.
                if lane == 0:
                    enqueue_paged_query(
                        query_map,
                        query_raw,
                        query_ready,
                        item // heads,
                        item % heads,
                        query_columns,
                    )
            mbar_wait(query_ready + stage, (loaded // q_stages) & 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            _issue_qk(
                query_raw + cute.assume(stage * (2 * query_columns * 128), divby=512),
                key_ready,
                key_free,
                score_free,
                score_ready,
                base,
                streamed,
                count,
                query_columns,
            )
            if prims.elect_sync():
                prims.tcgen05_commit(query_free + stage, group=prims.CTAGroup.CTA_1)
            streamed = streamed + count
            loaded = loaded + 1
    if cutlass.const_expr(early_tmem_retirement):
        # Also when the CTA has no unit or only empty ones.
        _retire_tensor_memory(base)


@cute.jit
def _tail_release_step(
    total: cutlass.Int64, release_lead: cutlass.Constexpr
) -> cutlass.Int64:
    """The stream step at which a loader that streams ``total`` pages releases
    the launch's dependents ``release_lead`` pages ahead of its end.

    A stream shorter than the lead releases at its first page: the step is
    clamped at zero, where the bare difference would never meet a step and
    the release would wait for the end of the stream, later than the default
    lead's. Both values are Int64, and a nonnegative total minus the lead
    cannot overflow.
    """
    release_step = total - release_lead
    if release_step < 0:
        release_step = cutlass.Int64(0)
    return release_step


@cute.jit
def _release_at_tail(
    stream_step: cutlass.Int64,
    total: cutlass.Int64,
    release_step: cutlass.Int64 | None,
) -> None:
    """Release the launch's dependents when the loader, which streams
    ``total`` pages, is at the stream step of its tail release:
    ``release_step`` (``_tail_release_step``), or without one the default
    lead's bare difference. Both values are Int64; a nonnegative total minus
    four cannot overflow, including empty ranges."""
    if cutlass.const_expr(release_step is None):
        if stream_step == total - MMA_DAG_RELEASE_LEAD_DEFAULT:
            cute.arch.griddepcontrol_launch_dependents()
    else:
        if stream_step == release_step:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _ordered_page(
    pages: cute.Tensor,
    page_order: cute.Tensor | None,
    request: cutlass.Int32,
    logical_page: cutlass.Int64,
    step: cutlass.Int64,
    order_first: cutlass.Int64 | None,
    order_stop: cutlass.Int64 | None,
) -> cutlass.Int32:
    """The physical page that a work item's step ``step`` requests: the page
    table's in source order (no ``page_order``), and under a bucketed page
    order inside the ordered run ``[order_first, order_stop)`` the page the
    sorting warp left for the step, elsewhere the page table's as well.
    Every step of the run is an interior tile
    (``_paged_attention_page_order``), so the step keeps its own
    coordinates, an all-visible mask and a key shift of zero whichever page
    of the run it streams."""
    if cutlass.const_expr(page_order is None):
        physical = cutlass.Int32(pages[request, logical_page])
    else:
        assert page_order is not None
        assert order_first is not None and order_stop is not None
        physical = cutlass.Int32(0)
        if (step >= order_first) & (step < order_stop):
            physical = cutlass.Int32(page_order[cutlass.Int32(step - order_first)])
        else:
            physical = cutlass.Int32(pages[request, logical_page])
    return physical


@cute.jit
def _loader_step(
    pages: cute.Tensor,
    page_order: cute.Tensor | None,
    request: cutlass.Int32,
    first_page: cutlass.Int64,
    last_page: cutlass.Int64,
    step0: cutlass.Int64,
    step: cutlass.Int64,
    order_first: cutlass.Int64 | None,
    order_stop: cutlass.Int64 | None,
    length: cutlass.Int32,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    window: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    page_walk: cutlass.Constexpr,
    key_clip: cutlass.Constexpr,
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """What the loader needs of a work item's step ``step0 + step`` before
    it requests the step's tiles: the physical page (``_ordered_page``, at
    the logical page the walk gives the step) and the key shift of that
    page's K and V tiles under ``key_clip``. Both forms of the loader's
    loop, V with its K and V behind it, begin a step with this."""
    logical_page, logical_key_base = attention_page_coordinates(
        first_page, last_page, step0 + step, page_walk
    )
    physical = _ordered_page(
        pages,
        page_order,
        request,
        logical_page,
        step0 + step,
        order_first,
        order_stop,
    )
    key_shift = logical_attention_key_shift(
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
    return physical, key_shift


@cute.jit
def _load_pages(
    pages: cute.Tensor,
    raw_k: cute.Pointer,
    scale_k: cute.Pointer,
    raw_k_ready: cute.Pointer,
    raw_k_free: cute.Pointer,
    kmap: experimental_cuda.TensorMap,
    ksmap: experimental_cuda.TensorMap,
    raw_v: cute.Pointer,
    scale_v: cute.Pointer,
    raw_v_ready: cute.Pointer,
    raw_v_free: cute.Pointer,
    vmap: experimental_cuda.TensorMap,
    vsmap: experimental_cuda.TensorMap,
    metadata_ready: cute.Pointer | None,
    metadata_flags: cute.Tensor | None,
    request: cutlass.Int32,
    first_head: cutlass.Int32,
    first_page: cutlass.Int64,
    last_page: cutlass.Int64,
    g0: cutlass.Int64,
    step0: cutlass.Int64,
    count: cutlass.Int64,
    total: cutlass.Int64,
    length: cutlass.Int32,
    previous_page: cutlass.Int32,
    older_page: cutlass.Int32,
    previous_shift: cutlass.Int32,
    older_shift: cutlass.Int32,
    previous_head: cutlass.Int32,
    older_head: cutlass.Int32,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    window: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    page_walk: cutlass.Constexpr,
    v_lag: cutlass.Constexpr,
    key_clip: cutlass.Constexpr,
    tma_load_hints: cutlass.Constexpr,
    sleep_waits: cutlass.Constexpr,
    capture_release_tail: cutlass.Constexpr,
    capture_release_lead: cutlass.Constexpr = MMA_DAG_RELEASE_LEAD_DEFAULT,
    value_ring: cutlass.Constexpr = PAIRED_DEFAULT_VALUE_RING,
    page_order: cute.Tensor | None = None,
    order_first: cutlass.Int64 | None = None,
    order_stop: cutlass.Int64 | None = None,
) -> tuple[
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
]:
    """Request the K and V tiles of ``count`` pages of a work item, from ``step0``.

    Page coordinates and key offsets are those of the item's step
    ``step0 + step``, so a clip applies at the item's own first and last
    page; ring slots and barrier phases follow ``g0 + step``, with ``g0``
    the number of pages the CTA streamed before these and ``total`` the
    number it streams in all; the raw V ring holds ``value_ring`` pages per
    head, the raw K ring two. With a V lag, ``previous_page`` and
    ``older_page`` are the last two pages streamed before this range,
    ``previous_shift`` and ``older_shift`` their key offsets and
    ``previous_head`` and ``older_head`` the first heads of their work
    items, all of which a lagged V tile keeps. The state of the last two
    pages streamed is returned: their V tiles are still to be requested, by
    a later range or by ``_drain_values``.

    With ``capture_release_tail`` the loader releases the launch's
    dependents when the stream has ``capture_release_lead`` pages left. The
    default lead of four keeps its test of the bare difference, which a
    stream of fewer than four pages never meets (the caller's release behind
    the loop covers it). Any other lead releases at ``_tail_release_step``,
    which a shorter stream meets at its first page: no later than the
    default lead would.

    With ``page_order`` the steps of the run ``[order_first, order_stop)``
    request the pages a bucketed order left for them (``_ordered_page``);
    a lagged V tile keeps the page its K tile requested.
    """
    release_step = None
    if cutlass.const_expr(
        capture_release_tail and capture_release_lead != MMA_DAG_RELEASE_LEAD_DEFAULT
    ):
        release_step = _tail_release_step(total, capture_release_lead)
    if cutlass.const_expr(v_lag == 0):
        for step in cutlass.range(count):
            if cutlass.const_expr(capture_release_tail):
                _release_at_tail(g0 + step, total, release_step)
            physical, key_shift = _loader_step(
                pages,
                page_order,
                request,
                first_page,
                last_page,
                step0,
                step,
                order_first,
                order_stop,
                length,
                query_len,
                group,
                window,
                coordinate_dtype,
                query_columns,
                page_walk,
                key_clip,
            )
            for head in cutlass.range_constexpr(2):
                _enqueue_operand_tile(
                    raw_k,
                    scale_k,
                    raw_k_ready,
                    raw_k_free,
                    kmap,
                    ksmap,
                    g0 + step,
                    cutlass.Int32(head),
                    physical,
                    first_head,
                    l2_hints=tma_load_hints[:2],
                    sleep_waits=sleep_waits,
                    key=key_shift,
                    depth=KEY_RING_PAGES,
                )
            for head in cutlass.range_constexpr(2):
                _enqueue_operand_tile(
                    raw_v,
                    scale_v,
                    raw_v_ready,
                    raw_v_free,
                    vmap,
                    vsmap,
                    g0 + step,
                    cutlass.Int32(head),
                    physical,
                    first_head,
                    metadata_ready=metadata_ready,
                    metadata_flags=metadata_flags,
                    l2_hints=tma_load_hints[2:],
                    sleep_waits=sleep_waits,
                    key=key_shift,
                    depth=value_ring,
                )
    else:
        # K(i) precedes V(i-lag). Iterating steps then two heads
        # avoids overflow from doubling an INT64 step count.
        # A bounded drain also avoids adding lag to that count.
        # Each delayed V tile reuses its K tile's page and shift.
        for step in cutlass.range(count):
            if cutlass.const_expr(capture_release_tail):
                _release_at_tail(g0 + step, total, release_step)
            physical, key_shift = _loader_step(
                pages,
                page_order,
                request,
                first_page,
                last_page,
                step0,
                step,
                order_first,
                order_stop,
                length,
                query_len,
                group,
                window,
                coordinate_dtype,
                query_columns,
                page_walk,
                key_clip,
            )
            for head in cutlass.range_constexpr(2):
                _enqueue_operand_tile(
                    raw_k,
                    scale_k,
                    raw_k_ready,
                    raw_k_free,
                    kmap,
                    ksmap,
                    g0 + step,
                    cutlass.Int32(head),
                    physical,
                    first_head,
                    l2_hints=tma_load_hints[:2],
                    sleep_waits=sleep_waits,
                    key=key_shift,
                    depth=KEY_RING_PAGES,
                )
                distance = (v_lag + 1 - head) // 2
                if cutlass.const_expr(distance == 0) or g0 + step >= distance:
                    delayed_page = physical
                    delayed_shift = key_shift
                    delayed_head = first_head
                    if cutlass.const_expr(distance == 1):
                        delayed_page = previous_page
                        delayed_shift = previous_shift
                        delayed_head = previous_head
                    elif cutlass.const_expr(distance == 2):
                        delayed_page = older_page
                        delayed_shift = older_shift
                        delayed_head = older_head
                    _enqueue_operand_tile(
                        raw_v,
                        scale_v,
                        raw_v_ready,
                        raw_v_free,
                        vmap,
                        vsmap,
                        g0 + step - distance,
                        cutlass.Int32((head - v_lag) & 1),
                        delayed_page,
                        delayed_head,
                        metadata_ready=metadata_ready,
                        metadata_flags=metadata_flags,
                        l2_hints=tma_load_hints[2:],
                        sleep_waits=sleep_waits,
                        key=delayed_shift,
                        depth=value_ring,
                    )
            older_page = previous_page
            previous_page = physical
            older_shift = previous_shift
            previous_shift = key_shift
            older_head = previous_head
            previous_head = first_head
    return (
        previous_page,
        older_page,
        previous_shift,
        older_shift,
        previous_head,
        older_head,
    )


@cute.jit
def _load_ordered_pages(
    pages: cute.Tensor,
    raw_k: cute.Pointer,
    scale_k: cute.Pointer,
    raw_k_ready: cute.Pointer,
    raw_k_free: cute.Pointer,
    kmap: experimental_cuda.TensorMap,
    ksmap: experimental_cuda.TensorMap,
    raw_v: cute.Pointer,
    scale_v: cute.Pointer,
    raw_v_ready: cute.Pointer,
    raw_v_free: cute.Pointer,
    vmap: experimental_cuda.TensorMap,
    vsmap: experimental_cuda.TensorMap,
    page_order: cute.Tensor,
    page_order_span: cute.Tensor,
    page_order_ready: cute.Pointer,
    request: cutlass.Int32,
    first_head: cutlass.Int32,
    first_page: cutlass.Int64,
    last_page: cutlass.Int64,
    count: cutlass.Int64,
    length: cutlass.Int32,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    window: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    page_walk: cutlass.Constexpr,
    v_lag: cutlass.Constexpr,
    key_clip: cutlass.Constexpr,
    tma_load_hints: cutlass.Constexpr,
    sleep_waits: cutlass.Constexpr,
    capture_release_tail: cutlass.Constexpr,
    capture_release_lead: cutlass.Constexpr,
    value_ring: cutlass.Constexpr,
) -> tuple[
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
]:
    """Request the pages of a one-unit CTA's work item under a bucketed
    page order: the item's first ``PAGE_ORDER_HEAD`` pages in source order,
    which need nothing of the sorting warp, then the rest once that warp
    has arrived at ``page_order_ready``, the steps of the run it left in
    ``page_order_span`` from ``page_order``. An item of no more pages than
    the head has no step behind it: its loader neither waits for the
    sorting warp nor reads what that warp leaves, and its run is empty. The
    stream is one: ring slots, barrier phases, the lagging V tiles and the
    tail release follow its steps as in source order. Returns what
    ``_load_pages`` returns.
    """
    head = cutlass.Int64(PAGE_ORDER_HEAD)
    if count < head:
        head = count
    (
        previous_page,
        older_page,
        previous_shift,
        older_shift,
        previous_head,
        older_head,
    ) = _load_pages(
        pages,
        raw_k,
        scale_k,
        raw_k_ready,
        raw_k_free,
        kmap,
        ksmap,
        raw_v,
        scale_v,
        raw_v_ready,
        raw_v_free,
        vmap,
        vsmap,
        None,
        None,
        request,
        first_head,
        first_page,
        last_page,
        cutlass.Int64(0),
        cutlass.Int64(0),
        head,
        count,
        length,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        first_head,
        first_head,
        query_len,
        group,
        window,
        coordinate_dtype,
        query_columns,
        page_walk,
        v_lag,
        key_clip,
        tma_load_hints,
        sleep_waits,
        capture_release_tail,
        capture_release_lead,
        value_ring,
    )
    # The empty run, which a step never lies in: all an item without a step
    # behind the head needs, so only a longer one depends on the sorting warp.
    order_first = cutlass.Int64(0)
    order_stop = cutlass.Int64(0)
    if count > head:
        # One arrival per lane of the sorting warp, each behind its stores:
        # the run and the order are complete, also for a unit that is not
        # reordered.
        _wait_non_mma(page_order_ready, cutlass.Int32(0), sleep_waits)
        order_first = cutlass.Int64(cutlass.Int32(page_order_span[SPAN_FIRST]))
        order_stop = cutlass.Int64(cutlass.Int32(page_order_span[SPAN_STOP]))
    return _load_pages(
        pages,
        raw_k,
        scale_k,
        raw_k_ready,
        raw_k_free,
        kmap,
        ksmap,
        raw_v,
        scale_v,
        raw_v_ready,
        raw_v_free,
        vmap,
        vsmap,
        None,
        None,
        request,
        first_head,
        first_page,
        last_page,
        head,
        head,
        count - head,
        count,
        length,
        previous_page,
        older_page,
        previous_shift,
        older_shift,
        previous_head,
        older_head,
        query_len,
        group,
        window,
        coordinate_dtype,
        query_columns,
        page_walk,
        v_lag,
        key_clip,
        tma_load_hints,
        sleep_waits,
        capture_release_tail,
        capture_release_lead,
        value_ring,
        page_order,
        order_first,
        order_stop,
    )


@cute.jit
def _drain_values(
    raw_v: cute.Pointer,
    scale_v: cute.Pointer,
    raw_v_ready: cute.Pointer,
    raw_v_free: cute.Pointer,
    vmap: experimental_cuda.TensorMap,
    vsmap: experimental_cuda.TensorMap,
    metadata_ready: cute.Pointer | None,
    metadata_flags: cute.Tensor | None,
    count: cutlass.Int64,
    previous_page: cutlass.Int32,
    older_page: cutlass.Int32,
    previous_shift: cutlass.Int32,
    older_shift: cutlass.Int32,
    previous_head: cutlass.Int32,
    older_head: cutlass.Int32,
    v_lag: cutlass.Constexpr,
    tma_load_hints: cutlass.Constexpr,
    sleep_waits: cutlass.Constexpr,
    value_ring: cutlass.Constexpr = PAIRED_DEFAULT_VALUE_RING,
) -> None:
    """Request the V tiles still lagging after the CTA streamed ``count`` pages."""
    # Drain at most three V tiles from already-loaded page
    # IDs. Guards precede subtraction, including empty ranges.
    for tail in cutlass.range_constexpr(v_lag):
        distance = (v_lag - tail + 1) // 2
        if count >= distance:
            delayed_page = previous_page
            delayed_shift = previous_shift
            delayed_head = previous_head
            if cutlass.const_expr(distance == 2):
                delayed_page = older_page
                delayed_shift = older_shift
                delayed_head = older_head
            _enqueue_operand_tile(
                raw_v,
                scale_v,
                raw_v_ready,
                raw_v_free,
                vmap,
                vsmap,
                count - distance,
                cutlass.Int32((tail - v_lag) & 1),
                delayed_page,
                delayed_head,
                metadata_ready=metadata_ready,
                metadata_flags=metadata_flags,
                l2_hints=tma_load_hints[2:],
                sleep_waits=sleep_waits,
                key=delayed_shift,
                depth=value_ring,
            )


@cute.jit
def _load_units(
    pages: cute.Tensor,
    raw_k: cute.Pointer,
    scale_k: cute.Pointer,
    raw_k_ready: cute.Pointer,
    raw_k_free: cute.Pointer,
    kmap: experimental_cuda.TensorMap,
    ksmap: experimental_cuda.TensorMap,
    raw_v: cute.Pointer,
    scale_v: cute.Pointer,
    raw_v_ready: cute.Pointer,
    raw_v_free: cute.Pointer,
    vmap: experimental_cuda.TensorMap,
    vsmap: experimental_cuda.TensorMap,
    query_map: experimental_cuda.TensorMap,
    query_raw: cute.Pointer,
    query_ready: cute.Pointer,
    query_free: cute.Pointer,
    first_unit: cutlass.Int32,
    last_unit: cutlass.Int32,
    total: cutlass.Int64,
    begin: cutlass.Int32,
    stop: cutlass.Int32,
    first_header: cutlass.Int32,
    lengths: cute.Tensor | None,
    unit_bounds: cutlass.Constexpr,
    unit_bounds_scanned: cutlass.Constexpr,
    unit_bounds_args: tuple,
    page_limit: cutlass.Constexpr,
    partition: cutlass.Constexpr,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    window: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    q_stages: cutlass.Constexpr,
    page_walk: cutlass.Constexpr,
    v_lag: cutlass.Constexpr,
    key_clip: cutlass.Constexpr,
    tma_load_hints: cutlass.Constexpr,
    sleep_waits: cutlass.Constexpr,
    capture_release_tail: cutlass.Constexpr,
    capture_release_lead: cutlass.Constexpr = MMA_DAG_RELEASE_LEAD_DEFAULT,
    value_ring: cutlass.Constexpr = PAIRED_DEFAULT_VALUE_RING,
) -> None:
    """Request the K and V tiles of the CTA's units as one stream, and the
    queries of every unit behind the first one with pages.

    The pages of consecutive units follow each other in the rings without a
    gap. Every unit's pages get its own steps, from zero, so a key clip
    shifts the tiles at the unit's own window edges and never at the
    boundary between two units; a lagged V tile is requested with the page,
    key offset and first head of its own unit. ``total`` is the number of
    pages of all units; only the tail release reads it, with the lead
    ``capture_release_lead`` in pages of the CTA's whole stream.

    Under the balanced partition the CTA streams the pages of a unit's
    steps in its slots ``[begin, stop)``, at those steps: the first or the
    last unit can be a part of a unit that other CTAs run the rest of, and
    its pages keep the coordinates and key offsets they have in the unit.
    ``first_header`` is the header slot of the first unit.
    """
    streamed = cutlass.Int64(0)
    loaded = cutlass.Int32(0)
    previous_page = cutlass.Int32(0)
    older_page = cutlass.Int32(0)
    previous_shift = cutlass.Int32(0)
    older_shift = cutlass.Int32(0)
    previous_head = cutlass.Int32(0)
    older_head = cutlass.Int32(0)
    header = first_header
    for unit in cutlass.range(first_unit, last_unit, 1):
        item, first_page, last_page, length, unit_pages, step0, count = _unit_range(
            unit,
            header,
            begin,
            stop,
            None,
            lengths,
            unit_bounds,
            unit_bounds_scanned,
            unit_bounds_args,
            page_limit,
            partition,
        )
        if cutlass.const_expr(partition == "balanced"):
            header = header + cutlass.Int32(unit_pages) + 1
        if count > 0:
            # A unit without pages issues no QK and needs no queries, and
            # the CTA's first unit with pages gets them from the QK role
            # (``_issue_qk_units``), so the stream starts with pages. The
            # release phase of a stage's first use is complete from the
            # start.
            stage = loaded % q_stages
            if loaded > 0:
                _wait_non_mma(
                    query_free + stage, ((loaded // q_stages) & 1) ^ 1, sleep_waits
                )
                enqueue_paged_query(
                    query_map,
                    query_raw
                    + cute.assume(stage * (2 * query_columns * 128), divby=512),
                    query_ready + stage,
                    item // heads,
                    item % heads,
                    query_columns,
                )
            loaded = loaded + 1
            (
                previous_page,
                older_page,
                previous_shift,
                older_shift,
                previous_head,
                older_head,
            ) = _load_pages(
                pages,
                raw_k,
                scale_k,
                raw_k_ready,
                raw_k_free,
                kmap,
                ksmap,
                raw_v,
                scale_v,
                raw_v_ready,
                raw_v_free,
                vmap,
                vsmap,
                None,
                None,
                item // heads,
                item % heads,
                first_page,
                last_page,
                streamed,
                step0,
                count,
                total,
                length,
                previous_page,
                older_page,
                previous_shift,
                older_shift,
                previous_head,
                older_head,
                query_len,
                group,
                window,
                coordinate_dtype,
                query_columns,
                page_walk,
                v_lag,
                key_clip,
                tma_load_hints,
                sleep_waits,
                capture_release_tail,
                capture_release_lead,
                value_ring,
            )
            streamed = streamed + count
    _drain_values(
        raw_v,
        scale_v,
        raw_v_ready,
        raw_v_free,
        vmap,
        vsmap,
        None,
        None,
        streamed,
        previous_page,
        older_page,
        previous_shift,
        older_shift,
        previous_head,
        older_head,
        v_lag,
        tma_load_hints,
        sleep_waits,
        value_ring,
    )
    if cutlass.const_expr(capture_release_tail):
        # Every CTA releases, also one with a short range or without units.
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _planned_units(
    plan: cute.Tensor,
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int64]:
    """The CTA's units ``[first, last)`` and the pages it streams over them.

    What the loader planned and left in shared memory ahead of its arrival
    at the setup rendezvous: only a role that waited there reads it.
    """
    return (
        cutlass.Int32(plan[PLAN_FIRST]),
        cutlass.Int32(plan[PLAN_LAST]),
        cutlass.Int64(cutlass.Int32(plan[PLAN_PAGES])),
    )


@cute.jit
def _issue_pv(
    probability_raw: cute.Pointer,
    probability_ready: cute.Pointer,
    value_ready: cute.Pointer,
    value_free: cute.Pointer,
    pv_done: cute.Pointer,
    base: cutlass.Int32,
    count: cutlass.Int64,
    query_columns: cutlass.Constexpr,
    output_offset: cutlass.Constexpr,
) -> None:
    """Issue PV for the ``count`` pages the CTA streams."""
    for step in cutlass.range(count):
        for head in cutlass.range_constexpr(2):
            mbar_wait(probability_ready + head, step & 1)
            mbar_wait(value_ready + head, step & 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            mma_128xnx128(
                base + head * 256 + 64,
                base + head * 256 + output_offset,
                probability_raw + head * (query_columns * 128),
                pv_done + head,
                True,
                query_columns,
                b_mn_major=True,
            )
            if prims.elect_sync():
                prims.tcgen05_commit(value_free + head, group=prims.CTAGroup.CTA_1)


@cute.jit
def paged_attention_paired_body(
    q: cute.Tensor,
    pages: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor | None,
    tile: cutlass.Int32,
    first_page: cutlass.Int64,
    last_page: cutlass.Int64,
    length: cutlass.Int32,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    query_rows: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    window: cutlass.Constexpr,
    score_scale: cutlass.Float32,
    fast_math: cutlass.Constexpr,
    kmap: experimental_cuda.TensorMap,
    ksmap: experimental_cuda.TensorMap,
    vmap: experimental_cuda.TensorMap,
    vsmap: experimental_cuda.TensorMap,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr = 40,
    io_layouts: cutlass.Constexpr = None,
    reference_scores: cutlass.Constexpr = False,
    base2_scores: cutlass.Constexpr = False,
    reflected_dtype: cutlass.Constexpr = None,
    v_lag: cutlass.Constexpr = 0,
    output_reciprocal: cutlass.Constexpr = False,
    key_post_scale_log2: cutlass.Constexpr = 0,
    value_post_scale_log2: cutlass.Constexpr = 0,
    output_scale_log2: cutlass.Constexpr = 0,
    tma_load_hints: cutlass.Constexpr = ("", "", "", ""),
    sleep_waits: cutlass.Constexpr = False,
    query_map: experimental_cuda.TensorMap | None = None,
    # A typed immutable scalar keeps unguarded calls constant-foldable.
    capture_pdl_wait_mode: cutlass.Int32 = cutlass.Int32(0),  # noqa: B008
    capture_release_tail: cutlass.Constexpr = False,
    capture_release_lead: cutlass.Constexpr = MMA_DAG_RELEASE_LEAD_DEFAULT,
    maximum_output: cute.Tensor | None = None,
    denominator_output: cute.Tensor | None = None,
    statistic_layouts: cutlass.Constexpr = None,
    coefficients_capped: cutlass.Constexpr = False,
    probability_schedule: cutlass.Constexpr = "serial",
    value_positive_zero: cutlass.Constexpr = False,
    value_metadata_scales: cute.Tensor | None = None,
    value_metadata_read_overlap: cutlass.Constexpr = False,
    rowwise_rescale_threshold: cutlass.Constexpr = None,
    early_tmem_retirement: cutlass.Constexpr = False,
    headwide_rescale: cutlass.Constexpr = False,
    join_before_exit: cutlass.Constexpr = False,
    key_clip: cutlass.Constexpr = "off",
    partition: cutlass.Constexpr = "item",
    units: cutlass.Constexpr = 0,
    page_limit: cutlass.Constexpr = 0,
    q_stages: cutlass.Constexpr = 1,
    slot_bits: cutlass.Constexpr = 32,
    unit_bounds: cutlass.Constexpr = None,
    unit_bounds_args: tuple = (),
    unit_bounds_scanned: cutlass.Constexpr = None,
    partition_workspace: cute.Tensor | None = None,
    partition_record_bytes: cutlass.Constexpr = 0,
    partition_skew: cutlass.Constexpr = 0,
    register_quotas: cutlass.Constexpr = PAIRED_DEFAULT_REGISTER_QUOTAS,
    boundary_prefetch: cutlass.Constexpr = 0,
    boundary_prefetch_form: cutlass.Constexpr = None,
    boundary_prefetch_capacity: cutlass.Constexpr = 0,
    boundary_prefetch_key: cute.Tensor | None = None,
    boundary_prefetch_key_scales: cute.Tensor | None = None,
    boundary_prefetch_layout: cutlass.Constexpr = None,
    value_ring: cutlass.Constexpr = PAIRED_DEFAULT_VALUE_RING,
    shared_memory_capacity: cutlass.Constexpr = 0,
    page_order: cutlass.Constexpr = MMA_DAG_PAGE_ORDER_SOURCE,
    page_order_shift: cutlass.Constexpr = 0,
    clipped_page_lead: cutlass.Constexpr = 0,
    store_policy: cutlass.Constexpr = "default",
) -> None:
    assert cutlass.const_expr(partition in ("item", "whole", "balanced"))
    # The cache policy of the O stores: plain, the streaming operator, or
    # the L2 evict-last hint.
    assert cutlass.const_expr(store_policy in ("default", "streaming", "l2_evict_last"))
    assert cutlass.const_expr(page_order in MMA_DAG_PAGE_ORDER_CHOICES)
    # The lead is the reversed walk's: no other order has a page to move.
    assert cutlass.const_expr(
        clipped_page_lead in MMA_DAG_CLIPPED_PAGE_LEAD_CHOICES
        and (clipped_page_lead == 0 or page_order == MMA_DAG_PAGE_ORDER_REVERSED)
    )
    if cutlass.const_expr(page_order == MMA_DAG_PAGE_ORDER_REVERSED):
        # The source's iterations newest first, each with its own tile:
        # another order of the recurrence, which only fast math licenses,
        # for a kernel with a window whose source loop is not itself
        # reflected. The walk is a work item's: on a persistent partition
        # every unit of a CTA's stream is walked newest first, and a part
        # of a unit keeps its steps of the unit's walk. The value-metadata
        # actor was not examined under it.
        assert cutlass.const_expr(fast_math)
        assert cutlass.const_expr(window is not None and reflected_dtype is None)
        assert cutlass.const_expr(value_metadata_scales is None)
    # The walk every role maps its steps through: the source's (its
    # reflection's dtype or none) or the reversed one. The loader, the
    # compute groups and a boundary prefetch all hand this one value to
    # ``attention_page_coordinates``.
    page_walk = (
        ReversedWalk(clipped_page_lead)
        if page_order == MMA_DAG_PAGE_ORDER_REVERSED
        else reflected_dtype
    )
    bucketed = page_order == MMA_DAG_PAGE_ORDER_BUCKETED
    if cutlass.const_expr(bucketed):
        # Another order of whole interior tiles rounds the recurrence at
        # other points, which only fast math licenses. The control warp
        # that sorts is the value-metadata actor's and the spare one of a
        # persistent partition, whose stream is several units.
        assert cutlass.const_expr(fast_math and partition == "item")
        assert cutlass.const_expr(value_metadata_scales is None)
    # Only a body that releases from its loader's tail has a lead to choose,
    # and a lead below the default's four pages would release later than it.
    assert cutlass.const_expr(type(capture_release_lead) is int)
    assert cutlass.const_expr(capture_release_lead >= MMA_DAG_RELEASE_LEAD_DEFAULT)
    assert cutlass.const_expr(
        capture_release_tail or capture_release_lead == MMA_DAG_RELEASE_LEAD_DEFAULT
    )
    if cutlass.const_expr(boundary_prefetch > 0):
        # The idle control warp of a one-unit CTA requests the first
        # ``boundary_prefetch`` K pages of the unit ``unit_bounds`` evaluates
        # for it: nothing else of this body changes.
        assert cutlass.const_expr(partition == "item" and units > 0)
        assert cutlass.const_expr(unit_bounds is not None)
        assert cutlass.const_expr(value_metadata_scales is None)
        assert cutlass.const_expr(boundary_prefetch_form in ("wave_ahead", "sibling"))
    else:
        assert cutlass.const_expr(boundary_prefetch_form is None)
    if cutlass.const_expr(partition == "item"):
        assert cutlass.const_expr(q_stages == 1)
    else:
        # ``tile``, the page range and ``length`` are unused: every role gets
        # them per unit from ``unit_bounds(unit, *unit_bounds_args)``.
        assert cutlass.const_expr(units > 0 and q_stages in (1, 2))
        assert cutlass.const_expr(unit_bounds is not None)
        assert cutlass.const_expr(query_map is not None)
        assert cutlass.const_expr(value_metadata_scales is None)
    # The CTAs a balanced launch's workspace has a record for, and its base.
    workspace_ctas = 0
    workspace = cutlass.Int64(0)
    if cutlass.const_expr(partition == "balanced"):
        # Parts of a unit are merged, which reassociates the source's sums,
        # and only the normalized epilogue has a merge.
        assert cutlass.const_expr(fast_math and maximum_output is None)
        assert partition_workspace is not None
        assert cutlass.const_expr(partition_record_bytes > 0)
        workspace_ctas = cute.size(partition_workspace) // partition_record_bytes
        assert cutlass.const_expr(type(workspace_ctas) is int and workspace_ctas > 0)
        workspace = workspace_address(partition_workspace)
    else:
        assert cutlass.const_expr(partition_workspace is None)
        assert cutlass.const_expr(partition_skew == 0)
    assert cutlass.const_expr(rowwise_rescale_threshold is None or reference_scores)
    # An increase the CTA's pool cannot grant would wait forever.
    quotas_reason = paired_register_quotas_reason(register_quotas)
    assert cutlass.const_expr(quotas_reason is None), quotas_reason
    compute_registers, key_registers, value_registers, control_registers = (
        register_quotas
    )
    assert cutlass.const_expr(v_lag in (0, 1, 2, 3))
    assert cutlass.const_expr(key_clip in ("off", "leading", "trailing", "both"))
    # The metadata actor copies unshifted scale words for the shifted raw tile.
    assert cutlass.const_expr(key_clip == "off" or value_metadata_scales is None)
    assert cutlass.const_expr(query_columns in (8, 32, 40))
    assert cutlass.const_expr(
        not value_metadata_read_overlap or value_metadata_scales is not None
    )
    if cutlass.const_expr(value_metadata_scales is not None):
        assert cutlass.const_expr(
            value_positive_zero and -22 <= value_post_scale_log2 <= -14
        )
        assert cutlass.const_expr(cute.size(value_metadata_scales, mode=[0]) == 128)
        assert cutlass.const_expr(cute.size(value_metadata_scales, mode=[2]) == 1)
    assert cutlass.const_expr((maximum_output is None) == (denominator_output is None))
    if cutlass.const_expr(maximum_output is not None):
        assert cutlass.const_expr(
            base2_scores
            and coefficients_capped
            and not reference_scores
            and lse is None
            and not output_reciprocal
            and output_scale_log2 == 0
        )
    else:
        assert cutlass.const_expr(not coefficients_capped)
        assert cutlass.const_expr(not base2_scores or reference_scores)
    assert cutlass.const_expr(
        not headwide_rescale
        or (
            rowwise_rescale_threshold is not None
            and reference_scores
            and query_rows == query_columns
        )
    )
    assert cutlass.const_expr(probability_schedule in ("serial", "early", "early_tmem"))
    assert cutlass.const_expr(probability_schedule == "serial" or reference_scores)
    # The early_tmem layout packs S/O/L after K[0:64], V[64:128], fitting each
    # head's 256-column slab for N=8,32,40. Other modes have just S/O.
    # O is the schedule-dependent offset shared by the PV issuer and the
    # compute warps that read/rescale it. L is local to the compute helper.
    output_offset = 128 + query_columns if probability_schedule == "early_tmem" else 192
    thread = cute.arch.thread_idx()[0]
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = thread % 32
    # The selected transport proves an even first item and even heads/request.
    first_head = tile % heads
    request = tile // heads
    count = cutlass.Int64(0)
    if last_page > first_page:
        count = last_page - first_page
    # Dynamic shared allocations reduce address setup across the warp roles.
    # The table lists them, stated once for the body it is: a deeper V ring
    # is checked against the device with it, and the body allocates it and
    # nothing else.
    shared_parts = pipeline_parts(
        query_map is not None,
        value_metadata_scales is not None,
        units if partition != "item" else 0,
        unit_bounds_scanned is not None,
        partition == "balanced",
        bucketed,
    )
    shared_table = pipeline_shared_memory(
        query_columns,
        value_metadata_scales is not None,
        q_stages if partition != "item" else 0,
        units if partition != "item" else 0,
        partition == "balanced",
        boundary_prefetch_form == "wave_ahead",
        value_ring,
        bucketed,
    )
    if cutlass.const_expr(value_ring != PAIRED_DEFAULT_VALUE_RING):
        # The value-metadata actor walks the V ring at its default depth.
        assert cutlass.const_expr(type(value_ring) is int and value_ring > 2)
        assert cutlass.const_expr(value_metadata_scales is None)
        # A CTA whose shared memory the device does not grant never
        # launches. The config contract admits a depth by replaying this
        # table against the device; the render repeats it for the table of
        # the body it actually is, on the capacity it was given.
        shared_bytes = pipeline_shared_bytes(shared_table, shared_parts)
        assert cutlass.const_expr(
            shared_memory_capacity == 0 or shared_bytes <= shared_memory_capacity
        ), (shared_bytes, shared_memory_capacity)
    shared = _allocate_pipeline_shared_memory(
        utils.SmemAllocator(), shared_table, shared_parts
    )
    raw_k = shared.raw_k
    raw_v = shared.raw_v
    scale_k = shared.scale_k
    scale_v = shared.scale_v
    raw_layout = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3), 0, cute.make_layout((128, 128), stride=(128, 1))
    )
    scale_layout = cute.make_layout((128, 4), stride=(4, 1))
    shared_q_raw = shared.shared_q_raw
    probability_raw = shared.probability_raw
    state_raw = shared.state_raw
    scratch_raw = shared.scratch_raw
    slot = shared.slot
    # One block holds every pipeline barrier so the loader warp can initialize
    # it with one instruction per 32 barriers: lane L owns barrier 32 * r + L.
    barrier_offsets, _barrier_count, barrier_rounds = pipeline_barrier_plan(
        value_metadata_scales is not None,
        q_stages if partition != "item" else 0,
        boundary_prefetch_form == "wave_ahead",
        value_ring,
        bucketed,
    )
    barriers = shared.barriers
    raw_k_ready = barriers + barrier_offsets["raw_k_ready"]
    raw_k_free = barriers + barrier_offsets["raw_k_free"]
    raw_v_ready = barriers + barrier_offsets["raw_v_ready"]
    raw_v_free = barriers + barrier_offsets["raw_v_free"]
    key_ready = barriers + barrier_offsets["key_ready"]
    key_free = barriers + barrier_offsets["key_free"]
    value_ready = barriers + barrier_offsets["value_ready"]
    value_free = barriers + barrier_offsets["value_free"]
    score_free = barriers + barrier_offsets["score_free"]
    score_ready = barriers + barrier_offsets["score_ready"]
    probability_ready = barriers + barrier_offsets["probability_ready"]
    pv_done = barriers + barrier_offsets["pv_done"]
    # ``None`` without a query map; a persistent partition's are in the block.
    query_ready = shared.query_ready
    query_free = None
    # Per compute group: what it needs again after a unit's pages and, under
    # the balanced partition, the role of a part of a split unit.
    walk_raw = shared.walk_raw
    walk_words = WALK_WORDS if partition == "balanced" else WHOLE_UNIT_WORDS
    # What each compute group posts to the spare warp (balanced only).
    mail_raw = shared.mail_raw
    plan_warps = 0
    plan_threads = 0
    plan_prefix = None
    plan_totals = None
    plan_lengths = None
    plan = None
    if cutlass.const_expr(partition != "item"):
        # The loader requests the queries of every unit but the first, so
        # each stage's barriers are among those it initializes; they follow
        # the pipeline's.
        query_ready = barriers + barrier_offsets["query_ready"]
        query_free = barriers + barrier_offsets["query_free"]
        # The plan tables, one word per scanning thread and one per warp,
        # and the plan: the CTA's first unit, the end of its units and the
        # pages it streams over them.
        _plan_chunk, plan_warps = plan_geometry(units)
        plan_prefix = cute.make_tensor(
            shared.plan_prefix, cute.make_layout((32 * plan_warps,))
        )
        plan_totals = cute.make_tensor(shared.plan_totals, cute.make_layout((32,)))
        plan_words = PLAN_WORDS if partition == "balanced" else WHOLE_UNIT_WORDS
        plan = cute.make_tensor(shared.plan, cute.make_layout((plan_words,)))
        if cutlass.const_expr(shared.plan_lengths is not None):
            # One more word per scanning thread: the length it loaded for
            # its unit, from which every role's walk evaluates the unit.
            plan_lengths = cute.make_tensor(
                shared.plan_lengths, cute.make_layout((32 * plan_warps,))
            )
    metadata_flags = None
    metadata_ready = None
    if cutlass.const_expr(value_metadata_scales is not None):
        metadata_flags = cute.make_tensor(shared.metadata_flags, cute.make_layout((4,)))
        metadata_ready = barriers + barrier_offsets["metadata_ready"]
    prefetch_signal = None
    if cutlass.const_expr(boundary_prefetch_form == "wave_ahead"):
        prefetch_signal = barriers + barrier_offsets["prefetch_signal"]
    # A bucketed page order: the order of one sort, the run it covers and
    # the barrier at which the sorting warp hands both to the loader.
    page_order_table = None
    page_order_span = None
    page_order_ready = None
    if cutlass.const_expr(bucketed):
        page_order_table = cute.make_tensor(
            shared.page_order, cute.make_layout((PAGE_ORDER_CAPACITY,))
        )
        page_order_span = cute.make_tensor(
            shared.page_order_span, cute.make_layout((SPAN_WORDS,))
        )
        page_order_ready = barriers + barrier_offsets["page_order_ready"]
    if warp == 17:
        # The loader starts the page stream without waiting for the other
        # roles, so it initializes the barriers that stream needs itself.
        for init_round in cutlass.range_constexpr(len(barrier_rounds)):
            expected = cutlass.Int32(1)
            for term in cutlass.range_constexpr(len(barrier_rounds[init_round])):
                extra, lanes = barrier_rounds[init_round][term]
                expected = expected + extra * ((cutlass.Int32(lanes) >> lane) & 1)
            prims.mbarrier_init(barriers + (32 * init_round + lane), expected)
        prims.fence_mbarrier_init()
        # Lane zero alone issues the page stream, against barriers the other
        # lanes initialized, and program order relates only one thread's own
        # operations: join the warp behind every lane's fence first, as the
        # conversion warps do before a lane-zero action.
        cute.arch.sync_warp()
    if cutlass.const_expr(partition != "item"):
        # The CTA's plan comes first: the loader's first request waits for
        # it, and it for every unit's length. Each scanning thread sums the
        # slots of its share of the units into the plan tables, as a
        # one-unit CTA loads its length, and arrives at barrier 7. Only the
        # loader waits there, in its role below. It finds the CTA's units
        # from the tables while the other roles finish their setup or
        # already wait at the setup rendezvous, where its arrival publishes
        # the plan to them.
        assert plan_prefix is not None and plan_totals is not None
        assert plan is not None
        # The scanning warps and, if it is not one of them, the loader's.
        plan_threads = 32 * (plan_warps + (0 if plan_warps > 17 else 1))
        partition_scan(
            plan_prefix,
            plan_totals,
            thread,
            lane,
            units,
            page_limit,
            unit_bounds,
            unit_bounds_args,
            plan_lengths,
        )
        if warp != 17:
            if warp < plan_warps:
                prims.barrier_cta_arrive(7, thread_count=plan_threads)
        if warp >= 16:
            # The control warps' register quota. ``setmaxnreg`` has every
            # warp of a warpgroup execute the same instruction, so the four
            # take theirs here, from one, and not each in its role. It
            # precedes the setup rendezvous because the loader's role starts
            # ahead of that and runs behind the quota from its first
            # statement; no warp waits for another on its way here.
            cute.arch.setmaxregister_decrease(control_registers)
    if warp == 16:
        # The whole warp allocates before any of its lanes takes another path.
        prims.tcgen05_alloc(slot, 512, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
        if cutlass.const_expr(query_map is not None and partition == "item"):
            # Only the QK issuer waits for shared Q, so its warp owns that
            # barrier and the copies; the setup rendezvous below publishes
            # the barrier to the warp's other lanes.
            if lane == 0:
                assert query_ready is not None
                prims.mbarrier_init(query_ready, 1)
                prims.fence_mbarrier_init()
                enqueue_paged_query(
                    query_map,
                    shared_q_raw,
                    query_ready,
                    request,
                    first_head,
                    query_columns,
                )
    if cutlass.const_expr(partition == "item"):
        # On a persistent partition each compute group resets its own rows
        # at every unit instead (``_compute_units``).
        if thread < 2 * query_columns:
            h = thread // query_columns
            query = thread % query_columns
            state = cute.make_tensor(
                state_raw + h * (4 * query_columns),
                cute.make_layout((4, query_columns), stride=(query_columns, 1)),
            )
            state[0, query] = cutlass.Float32(float("-inf"))
            state[1, query] = cutlass.Float32(0)
            state[2, query] = cutlass.Float32(0)
            state[3, query] = cutlass.Float32(1)
    if cutlass.const_expr(query_map is None):
        shared_q = cute.make_tensor(
            shared_q_raw, cute.make_layout((2 * query_columns * 128,))
        )
        for start in cutlass.range(0, 2 * query_columns * 128, 640, unroll_full=True):
            element = start + thread
            if element < 2 * query_columns * 128:
                pair_head = element // (query_columns * 128)
                inner = element % (query_columns * 128)
                query_row = inner // 128
                dimension = inner % 128
                offset = (
                    pair_head * query_columns * 128
                    + query_row * 64
                    + ((dimension % 64) ^ ((query_row % 8) * 8))
                    + (dimension // 64) * (query_columns * 64)
                )
                value = cutlass.BFloat16(0)
                if query_row < query_rows:
                    value = load_query_element(
                        q,
                        tile + pair_head,
                        query_row,
                        dimension,
                        heads,
                        group,
                        io_layouts,
                    )
                shared_q[offset] = value
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    # Setup rendezvous. The loader only arrives: each of its lanes initialized
    # and fenced its barriers and the warp then synchronized above, so lane
    # zero's stream follows every lane's initialization; it reads no state
    # another role prepares (not the TMEM base, the row state or shared Q);
    # and every consumer of what it loads waits on one of those barriers. The
    # other 608 threads proceed only after all 640 arrivals, each of which
    # follows its lane's fence, hence after the loader's initialization. A
    # persistent CTA's loader arrives from its role, where it plans first,
    # so there the 608 also proceed only after it wrote the plan.
    # The QK role's allocation writes the TMEM slot, so only a thread that
    # waited here reads it: the loader, which has no use for the base, would
    # race with that write.
    base = cutlass.Int32(0)
    if warp == 17:
        if cutlass.const_expr(partition == "item"):
            prims.barrier_cta_arrive(6, thread_count=640)
    else:
        prims.barrier_cta_sync(6, thread_count=640)
        base = cute.make_tensor(slot, cute.make_layout((1,)))[0]
    if cutlass.const_expr(query_map is not None and partition == "item"):
        # Only the QK issuer consumes shared Q; the other roles start on KV
        # without it. Empty KV loops still wait here, so the query copies
        # have landed before this role frees TMEM and returns.
        if warp == 16:
            mbar_wait(query_ready, cutlass.Int32(0))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if cutlass.const_expr(partition != "item"):
        assert walk_raw is not None and plan is not None
        assert query_ready is not None and query_free is not None
        assert plan_prefix is not None and plan_totals is not None
        if warp == 17:
            # The loader's role runs behind the control warps' quota, which
            # its warp took above while the other warps still scanned. It
            # waits for the scans at barrier 7, finds the CTA's units,
            # leaves the plan to the other roles and starts the stream from
            # its own copy of it.
            prims.barrier_cta_sync(7, thread_count=plan_threads)
            (
                planned_first,
                planned_last,
                planned_pages,
                planned_begin,
                planned_end,
                planned_header,
                planned_slots,
                planned_active,
                planned_skew,
            ) = partition_plan(
                plan_prefix,
                plan_totals,
                lane,
                units,
                page_limit,
                slot_bits,
                unit_bounds,
                unit_bounds_args,
                partition,
                partition_skew,
                workspace_ctas,
            )
            if lane == 0:
                plan[PLAN_FIRST] = planned_first
                plan[PLAN_LAST] = planned_last
                plan[PLAN_PAGES] = cutlass.Int32(planned_pages)
                if cutlass.const_expr(partition == "balanced"):
                    plan[PLAN_BEGIN] = planned_begin
                    plan[PLAN_END] = planned_end
                    plan[PLAN_HEADER] = planned_header
                    plan[PLAN_SLOTS] = planned_slots
                    plan[PLAN_ACTIVE] = planned_active
                    plan[PLAN_SKEW] = planned_skew
            # Lane zero's arrival follows its writes, and the rendezvous
            # needs every arrival: it publishes the plan to the other roles.
            prims.barrier_cta_arrive(6, thread_count=640)
            if lane == 0:
                _load_units(
                    pages,
                    raw_k,
                    scale_k,
                    raw_k_ready,
                    raw_k_free,
                    kmap,
                    ksmap,
                    raw_v,
                    scale_v,
                    raw_v_ready,
                    raw_v_free,
                    vmap,
                    vsmap,
                    query_map,
                    shared_q_raw,
                    query_ready,
                    query_free,
                    planned_first,
                    planned_last,
                    planned_pages,
                    planned_begin,
                    planned_end,
                    planned_header,
                    plan_lengths,
                    unit_bounds,
                    unit_bounds_scanned,
                    unit_bounds_args,
                    page_limit,
                    partition,
                    query_len,
                    group,
                    heads,
                    window,
                    coordinate_dtype,
                    query_columns,
                    q_stages,
                    page_walk,
                    v_lag,
                    key_clip,
                    tma_load_hints,
                    sleep_waits,
                    capture_release_tail,
                    capture_release_lead,
                    value_ring,
                )
        elif warp < 8:
            first_unit, last_unit, _pages = _planned_units(plan)
            head = cute.arch.make_warp_uniform(warp // 4)
            _compute_units(
                output,
                lse,
                cute.make_tensor(
                    probability_raw + head * query_columns * 128,
                    cute.make_layout((query_columns * 128,)),
                ),
                cute.make_tensor(
                    state_raw + head * 4 * query_columns,
                    cute.make_layout((4, query_columns), stride=(query_columns, 1)),
                ),
                cute.make_tensor(
                    scratch_raw + head * 4 * query_columns,
                    cute.make_layout((4, query_columns), stride=(query_columns, 1)),
                ),
                score_ready + head,
                score_free + head,
                probability_ready + head,
                pv_done + head,
                base + head * 256,
                head,
                cute.make_tensor(
                    walk_raw + head * walk_words, cute.make_layout((walk_words,))
                ),
                cute.make_tensor(
                    mail_raw + head * MAIL_WORDS, cute.make_layout((MAIL_WORDS,))
                )
                if partition == "balanced"
                else None,
                plan,
                workspace,
                first_unit,
                last_unit,
                plan_lengths,
                unit_bounds,
                unit_bounds_scanned,
                unit_bounds_args,
                page_limit,
                partition,
                partition_record_bytes,
                slot_bits,
                partition_skew != 0,
                query_len,
                group,
                query_rows,
                heads,
                window,
                score_scale,
                fast_math,
                coordinate_dtype,
                query_columns,
                io_layouts,
                reference_scores,
                base2_scores,
                page_walk,
                output_reciprocal,
                output_scale_log2,
                sleep_waits,
                capture_pdl_wait_mode,
                maximum_output,
                denominator_output,
                statistic_layouts,
                coefficients_capped,
                probability_schedule,
                output_offset,
                rowwise_rescale_threshold,
                early_tmem_retirement,
                headwide_rescale,
                key_clip,
                compute_registers,
                store_policy,
            )
        elif warp < 16:
            # The conversion warps, like the PV issuer, only follow the stream.
            _first, _last, count = _planned_units(plan)
            # Both conversions have one quota in every admitted split
            # (``MMA_DAG_REGISTER_QUOTAS_CHOICES``), so one instruction ahead
            # of the branch serves them.
            assert cutlass.const_expr(key_registers == value_registers)
            cute.arch.setmaxregister_decrease(key_registers)
            if warp < 12:
                _convert_keys(
                    raw_k,
                    scale_k,
                    raw_k_ready,
                    raw_k_free,
                    key_ready,
                    key_free,
                    raw_layout,
                    scale_layout,
                    base,
                    (warp % 4) * 32 + lane,
                    lane,
                    count,
                    key_post_scale_log2,
                    sleep_waits,
                )
            else:
                _convert_values(
                    raw_v,
                    scale_v,
                    raw_v_ready,
                    raw_v_free,
                    value_ready,
                    value_free,
                    raw_layout,
                    scale_layout,
                    None,
                    None,
                    base,
                    (warp % 4) * 32,
                    lane,
                    count,
                    value_post_scale_log2,
                    value_positive_zero,
                    sleep_waits,
                    value_ring,
                )
        else:
            first_unit, last_unit, count = _planned_units(plan)
            if warp == 16:
                planned_begin = cutlass.Int32(0)
                planned_end = cutlass.Int32(0)
                planned_header = cutlass.Int32(0)
                if cutlass.const_expr(partition == "balanced"):
                    planned_begin = cutlass.Int32(plan[PLAN_BEGIN])
                    planned_end = cutlass.Int32(plan[PLAN_END])
                    planned_header = cutlass.Int32(plan[PLAN_HEADER])
                _issue_qk_units(
                    query_map,
                    shared_q_raw,
                    query_ready,
                    query_free,
                    key_ready,
                    key_free,
                    score_free,
                    score_ready,
                    base,
                    lane,
                    first_unit,
                    last_unit,
                    planned_begin,
                    planned_end,
                    planned_header,
                    plan_lengths,
                    unit_bounds,
                    unit_bounds_scanned,
                    unit_bounds_args,
                    page_limit,
                    partition,
                    heads,
                    query_columns,
                    q_stages,
                    early_tmem_retirement,
                )
            elif warp == 18:
                _issue_pv(
                    probability_raw,
                    probability_ready,
                    value_ready,
                    value_free,
                    pv_done,
                    base,
                    count,
                    query_columns,
                    output_offset,
                )
            elif warp == 19:
                if cutlass.const_expr(partition == "balanced"):
                    # The spare warp arrives for the parts the compute
                    # groups post, off their path.
                    assert mail_raw is not None
                    post_parts(
                        workspace,
                        cute.make_tensor(mail_raw, cute.make_layout((2 * MAIL_WORDS,))),
                        lane,
                        capture_pdl_wait_mode,
                        partition_record_bytes,
                    )
    elif warp < 8:
        head = cute.arch.make_warp_uniform(warp // 4)
        probability = cute.make_tensor(
            probability_raw + head * query_columns * 128,
            cute.make_layout((query_columns * 128,)),
        )
        state = cute.make_tensor(
            state_raw + head * 4 * query_columns,
            cute.make_layout((4, query_columns), stride=(query_columns, 1)),
        )
        scratch = cute.make_tensor(
            scratch_raw + head * 4 * query_columns,
            cute.make_layout((4, query_columns), stride=(query_columns, 1)),
        )
        _compute_head(
            output,
            lse,
            probability,
            state,
            scratch,
            score_ready + head,
            score_free + head,
            probability_ready + head,
            pv_done + head,
            base + head * 256,
            tile + head,
            head,
            first_page,
            last_page,
            count,
            length,
            query_len,
            group,
            query_rows,
            heads,
            window,
            score_scale,
            fast_math,
            coordinate_dtype,
            query_columns,
            io_layouts,
            reference_scores,
            base2_scores,
            page_walk,
            output_reciprocal,
            output_scale_log2,
            sleep_waits,
            capture_pdl_wait_mode,
            maximum_output,
            denominator_output,
            statistic_layouts,
            coefficients_capped,
            probability_schedule,
            output_offset,
            rowwise_rescale_threshold,
            early_tmem_retirement,
            headwide_rescale,
            key_clip,
            compute_registers,
            store_policy,
        )
    elif warp < 12:
        cute.arch.setmaxregister_decrease(key_registers)
        row = (warp % 4) * 32 + lane
        _convert_keys(
            raw_k,
            scale_k,
            raw_k_ready,
            raw_k_free,
            key_ready,
            key_free,
            raw_layout,
            scale_layout,
            base,
            row,
            lane,
            count,
            key_post_scale_log2,
            sleep_waits,
        )
    elif warp < 16:
        cute.arch.setmaxregister_decrease(value_registers)
        dimension = (warp % 4) * 32
        _convert_values(
            raw_v,
            scale_v,
            raw_v_ready,
            raw_v_free,
            value_ready,
            value_free,
            raw_layout,
            scale_layout,
            metadata_flags,
            value_metadata_scales,
            base,
            dimension,
            lane,
            count,
            value_post_scale_log2,
            value_positive_zero,
            sleep_waits,
            value_ring,
        )
    else:
        cute.arch.setmaxregister_decrease(control_registers)
        if warp == 16:
            _issue_qk(
                shared_q_raw,
                key_ready,
                key_free,
                score_free,
                score_ready,
                base,
                cutlass.Int64(0),
                count,
                query_columns,
            )
            if cutlass.const_expr(early_tmem_retirement):
                # Zero-page tiles take the same path: each compute group
                # arrives after reading its zeroed panel.
                _retire_tensor_memory(base)
        elif warp == 17:
            if lane == 0:
                if cutlass.const_expr(bucketed):
                    assert page_order_table is not None
                    assert page_order_span is not None
                    assert page_order_ready is not None
                    (
                        previous_page,
                        older_page,
                        previous_shift,
                        older_shift,
                        previous_head,
                        older_head,
                    ) = _load_ordered_pages(
                        pages,
                        raw_k,
                        scale_k,
                        raw_k_ready,
                        raw_k_free,
                        kmap,
                        ksmap,
                        raw_v,
                        scale_v,
                        raw_v_ready,
                        raw_v_free,
                        vmap,
                        vsmap,
                        page_order_table,
                        page_order_span,
                        page_order_ready,
                        request,
                        first_head,
                        first_page,
                        last_page,
                        count,
                        length,
                        query_len,
                        group,
                        window,
                        coordinate_dtype,
                        query_columns,
                        page_walk,
                        v_lag,
                        key_clip,
                        tma_load_hints,
                        sleep_waits,
                        capture_release_tail,
                        capture_release_lead,
                        value_ring,
                    )
                else:
                    (
                        previous_page,
                        older_page,
                        previous_shift,
                        older_shift,
                        previous_head,
                        older_head,
                    ) = _load_pages(
                        pages,
                        raw_k,
                        scale_k,
                        raw_k_ready,
                        raw_k_free,
                        kmap,
                        ksmap,
                        raw_v,
                        scale_v,
                        raw_v_ready,
                        raw_v_free,
                        vmap,
                        vsmap,
                        metadata_ready,
                        metadata_flags,
                        request,
                        first_head,
                        first_page,
                        last_page,
                        cutlass.Int64(0),
                        cutlass.Int64(0),
                        count,
                        count,
                        length,
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                        first_head,
                        first_head,
                        query_len,
                        group,
                        window,
                        coordinate_dtype,
                        query_columns,
                        page_walk,
                        v_lag,
                        key_clip,
                        tma_load_hints,
                        sleep_waits,
                        capture_release_tail,
                        capture_release_lead,
                        value_ring,
                    )
                _drain_values(
                    raw_v,
                    scale_v,
                    raw_v_ready,
                    raw_v_free,
                    vmap,
                    vsmap,
                    metadata_ready,
                    metadata_flags,
                    count,
                    previous_page,
                    older_page,
                    previous_shift,
                    older_shift,
                    previous_head,
                    older_head,
                    v_lag,
                    tma_load_hints,
                    sleep_waits,
                    value_ring,
                )
                if cutlass.const_expr(capture_release_tail):
                    # Every CTA releases even for short or zero-trip ranges.
                    # Repeating a long range's earlier release is harmless.
                    cute.arch.griddepcontrol_launch_dependents()
                if cutlass.const_expr(prefetch_signal is not None):
                    # Behind the loader's last request, lagging V tiles
                    # included: from here this SM's memory traffic only
                    # drains, and the unit that takes the SM over issues its
                    # first request a few microseconds later. A prototype
                    # measured a third less gain for requests issued one
                    # page earlier, and a loss for requests at entry. One
                    # arrival per CTA, also for a unit without pages.
                    prims.mbarrier_arrive(prefetch_signal)
        elif warp == 18:
            _issue_pv(
                probability_raw,
                probability_ready,
                value_ready,
                value_free,
                pv_done,
                base,
                count,
                query_columns,
                output_offset,
            )
        elif warp == 19:
            if cutlass.const_expr(value_metadata_scales is not None):
                for step in cutlass.range(count):
                    logical_page, _logical_key_base = attention_page_coordinates(
                        first_page, last_page, step, page_walk
                    )
                    physical = cutlass.Int32(pages[request, logical_page])
                    for head in cutlass.range_constexpr(2):
                        _prepare_value_metadata(
                            scale_v,
                            raw_v_free,
                            metadata_flags,
                            metadata_ready,
                            value_metadata_scales,
                            step,
                            cutlass.Int32(head),
                            physical,
                            first_head,
                            sleep_waits,
                            read_overlap=value_metadata_read_overlap,
                        )
            if cutlass.const_expr(bucketed):
                assert page_order_table is not None
                assert page_order_span is not None
                assert page_order_ready is not None
                # Ahead of a boundary prefetch: the loader waits for the
                # order behind its first pages.
                order_unit_pages(
                    pages,
                    page_order_table,
                    shared.page_order_buckets,
                    shared.page_order_members,
                    page_order_span,
                    page_order_ready,
                    request,
                    first_page,
                    last_page,
                    count,
                    length,
                    lane,
                    page_order_shift,
                    query_columns,
                    query_len,
                    group,
                    window,
                    coordinate_dtype,
                    page_walk,
                )
            if cutlass.const_expr(boundary_prefetch > 0):
                assert boundary_prefetch_key is not None
                assert boundary_prefetch_key_scales is not None
                # The CTA's unit is its pair of leading items.
                prefetch_boundary_units(
                    pages,
                    boundary_prefetch_key,
                    boundary_prefetch_key_scales,
                    tile // 2,
                    lane,
                    capture_pdl_wait_mode,
                    prefetch_signal,
                    unit_bounds,
                    unit_bounds_args,
                    boundary_prefetch_form,
                    boundary_prefetch,
                    units,
                    boundary_prefetch_capacity,
                    boundary_prefetch_layout,
                    heads,
                    query_columns,
                    query_len,
                    group,
                    window,
                    coordinate_dtype,
                    page_walk,
                    key_clip,
                )
    # With the allocation retired above, nothing else needs a common end: the
    # conversion warps consumed every raw tile, the issuers' MMAs completed
    # before the last O read, and the QK role waited for the query copies and
    # frees TMEM itself. Joining would only keep finished warps resident until
    # the last store. A caller that releases dependents after the body
    # (``join_before_exit``) still gets one common end.
    if cutlass.const_expr(join_before_exit or not early_tmem_retirement):
        cute.arch.barrier()
    if cutlass.const_expr(not early_tmem_retirement):
        if warp == 16:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(
                cutlass.inttoptr(base, 6, cutlass.Int32),
                512,
                group=prims.CTAGroup.CTA_1,
            )
