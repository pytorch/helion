"""Device helpers for persistent partitions of the paired attention schedule.

A persistent CTA runs a contiguous range of units, each a pair of adjacent
leading items, as one page stream. The range follows from the launch grid and
the loop bounds of all units, which the generated ``unit_bounds`` function
evaluates: a pure function of the unit and the kernel's immutable inputs, so
every role that walks the range gets the same trip counts.

Slots. A unit that streams ``n`` pages owns ``n + 1`` consecutive slots, a
header and one per page, in unit order; ``T`` is their total. The first
``G = min(grid, T)`` CTAs of the launch are active and CTA ``c`` owns the
slots ``[c * T // G, (c + 1) * T // G)``: an equal share of the work whatever
the units' lengths, and at least one slot. The ``whole`` partition gives a
CTA the units whose header slot lies in its range, so every unit has one
owner and runs there as it would alone; a CTA whose range lies inside one
long unit owns none.

Plan. The exact slot sums need every unit's length. Each thread sums the
slots of its own units and its warp scans them (``partition_scan``): two
small tables in shared memory, written once. Behind a barrier that publishes
them a warp reads the warps' totals and the one warp row its range starts
and ends in, and finds the range with two divisions (``partition_plan``).
The loads of the scan are one per thread, so they overlap, and the plan
costs the same for any number of units up to one per thread.

With one unit per thread the scan also keeps each unit's length in a third
table. A role that reaches a unit later evaluates its bounds from that word
instead of loading the length again: a global load ahead of the first page
is the longest step of a CTA's start once memory is busy.

The ``balanced`` partition gives a CTA the page slots of its range
themselves. Slot ``header + 1 + j`` of a unit with ``n`` pages is its loop
step ``n - 1 - j``, so a unit that crosses a range boundary is split into
parts that consecutive CTAs run, the lowest CTA its last steps
(``unit_segment``). A unit without pages runs where its header is. Every
part runs the recurrence from an empty state; the parts exchange their
unnormalized results through a workspace the launcher owns and one of them
merges them (``split_arrival``, ``store_part``, ``arrive_part``,
``merge_parts``):

- A record of the workspace belongs to one CTA: two counters, one per
  head, and four slots for a partial (panel and row statistics), two per
  head. The counter of a split unit is the one of its lowest CTA. A CTA is
  the lowest of at most one split unit and a higher part of at most one
  other, hence the two slots per head.
- Nobody waits for another CTA. A part stores its partial, arrives with an
  atomic add and merges only if that arrival was the last. The unit's lowest
  part, which runs the unit's last steps as its CTA's last pages and so
  normally arrives last, first reads the counter once: if every other part
  has arrived, it merges without arriving. Whichever part merges folds all
  of them in source order, from the unit's first steps to its last, so the
  result does not depend on the order of arrival.
- An arrival takes about a page's time while the CTA's memory traffic is
  busy, and the compute group that waited for it would stall the stream
  behind it. A CTA has at most one part that is not its unit's lowest, the
  first one it runs, and the group does not arrive for that one itself: it
  stores the partial and posts it to the CTA's spare warp
  (``post_parts``), which joins the group's stores at a named barrier,
  arrives for them and tells the group at the end of its walk whether the
  arrival was the last. Only then, which takes the unit's lowest part to
  finish first, the group merges that unit after all its other work. The
  warp hears from each group exactly once: the thread that decides a part's
  role notes a post in the group's record, and at the end of its walk a
  group without that note posts that it has no part.
- The merger resets the counter, so all counters are zero whenever no
  launch that uses the workspace is in flight. Every access to the
  workspace follows the launch's dependency wait in the thread that makes
  it.
- What one thread of a compute group decides, and what its row owners
  store or weigh, reaches the group at its next named barrier
  (``bar.sync``). A barrier waits for the lanes of its warp (the PTX ISA,
  quoted in ``_compute_units``), so no block of some lanes needs a warp
  synchronization. The spare warp and the groups meet at barriers of the
  non-aligned form, which they reach at different instructions.
- With ``cute_mma_dag_partition_look="spare"`` the lowest part's one look
  at the counter is the spare warp's too (``look_parts``). A group says at
  the start of that part how many other parts the unit has and arrives at
  a named barrier without waiting; a group that runs no such part says so
  at the end of its walk. The warp, behind its posts, looks once per head
  with the load, the fence and the reset the group's thread makes
  otherwise, leaves in the group's record what that thread would (the
  part's role, the unit's lowest CTA and its parts) and, for a unit in two
  parts that is complete, stages the other part's row statistics in shared
  memory. The group meets the warp at a second named barrier, which is the
  first barrier of the part's reduction: no thread of the group decides
  behind the part's pages, and every thread reads the role and the staged
  pairs (``pair_partial``) behind that meeting. The look comes earlier
  than the group's own would, so it finds the other parts arrived less
  often; the part then publishes and arrives, and the last arrival merges,
  as ever.
  For that complete unit in two parts the warp also stages the other
  part's panel, behind its answer (``stage_panels``): a bulk copy per
  head of the panel's leading columns into a tile of the raw K ring, once
  the K conversion warps have converted the stream's last K tile, which
  each says by one arrival at a barrier of its own. The lowest part is the
  CTA's last range, so the ring has no later use. The group then reads its
  own panel with one load, waits for the copies and folds element by
  element with the other part's value from that tile
  (``merge_pair_staged``): the fused operations of ``merge_pair_other``
  and ``merge_pair_own`` in their order, so the same bits. The columns
  the tile does not hold (the last eight of forty) are loaded as ever.
  A group that met the warp there has heard the verdict of its post and
  does not meet the warp again at the end of its walk, so its late merge
  of a posted unit starts with a barrier of the group: the merge's row
  owners rewrite the weights and the row state (``merge_parts``) that the
  warps still in the lowest part's epilogue read.

A range skew of ``k`` slots moves the range boundaries so that the shares
fall linearly from the mean plus ``k`` (CTA 0) to the mean minus ``k``. A
launch applies at most a thirty-second of its mean share (``SKEW_SHARE``),
whatever the lengths are: no range is empty, and the longest CTA runs at
most a thirty-second longer than its equal share.
"""

from __future__ import annotations

from typing import cast

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from ._paged_attention_compute import output_normalization_factor
from .device.ptx import atom_add_acq_rel_gpu_u32
from .device.ptx import cp_bulk_global_to_shared
from .device.ptx import fence_acq_rel_gpu
from .device.ptx import l2_policy_evict_first
from .device.ptx import l2_policy_evict_last
from .device.ptx import ld_relaxed_gpu_u32
from .device.ptx import ldcg_v2_f32
from .device.ptx import ldcg_v4_f32
from .device.ptx import stg_u32_l2
from .device.ptx import stg_v2_f32_l2
from .device.ptx import stg_v4_f32_l2
from .device.sync import mbar_wait_sleep
from .paged_attention_partition_bounds import RECORD_HEADER_BYTES
from .paged_attention_partition_bounds import SKEW_SHARE
from .paged_attention_partition_bounds import partial_slot_bytes
from .paged_attention_partition_bounds import plan_grid_bound
from .paged_attention_shared_memory import MAIL_WORDS
from .paged_attention_shared_memory import PANEL_QUAD_BYTES
from .paged_attention_shared_memory import PLAN_WORDS
from .paged_attention_shared_memory import RAW_TILE_BYTES
from .paged_attention_shared_memory import WALK_WORDS
from .paged_attention_shared_memory import partition_look_panel_columns
from .paged_attention_shared_memory import plan_geometry

# Words of the plan a balanced CTA's loader leaves in shared memory: its
# units, pages and slot range, the first unit's header slot, then the slots
# of all units, the active CTAs and the applied skew. The plan's length, a
# walk record's and a post's are stated with the body's shared-memory table
# (``paged_attention_shared_memory``), as is the scan's geometry
# (``plan_geometry``).
PLAN_FIRST = 0
PLAN_LAST = 1
PLAN_PAGES = 2
PLAN_BEGIN = 3
PLAN_END = 4
PLAN_HEADER = 5
PLAN_SLOTS = 6
PLAN_ACTIVE = 7
PLAN_SKEW = 8
assert PLAN_SKEW < PLAN_WORDS
# What a part of a split unit does once its pages are done. A merge of a
# unit in two parts by the part that did not publish is the common one: its
# loads start inside the part's reduction (``pair_partial``).
PART_DONE = 0
PART_PUBLISH = 1
PART_MERGE = 2
PART_MERGE_PAIR = 3
PART_POST = 4
# What a compute group posts to the CTA's spare warp, four words per head:
# the kind, then for a part to arrive for the unit's lowest CTA, its parts
# and its first item. The spare warp replaces the kind with its verdict.
MAIL_KIND = 0
MAIL_LOWEST = 1
MAIL_PARTS = 2
MAIL_ITEM = 3
assert MAIL_ITEM < MAIL_WORDS
MAIL_NONE = 1
MAIL_PART = 2
MAIL_ARRIVED = 3
MAIL_LAST = 4
# Named barriers of a compute group and the spare warp: the group arrives
# at the first when it has posted, the spare warp at the second when it has
# dealt with the post. Head ``h`` uses the barrier ``+ h``.
POST_BARRIER = 8
POSTED_BARRIER = 10
POST_THREADS = 160
# A compute group's record in shared memory, sixteen words: what its walk
# needs again after a unit's pages (the unit, the end of the CTA's units,
# the unit's first item, its header slot, its pages and the first step this
# CTA runs), then the exchange of a part of a split unit (its role, the
# unit's lowest CTA and its parts) and whether the group has posted a part
# to the spare warp, which the end of its walk reads.
WALK_UNIT = 0
WALK_END = 1
WALK_ITEM = 2
WALK_HEADER = 3
WALK_PAGES = 4
WALK_STEP = 5
WALK_ROLE = 8
WALK_LOWEST = 9
WALK_PARTS = 10
WALK_POSTED = 11
# And, where the spare warp looks for the unit's lowest part
# (``look_parts``): the other parts of that unit, which the group asks the
# warp to look for (zero: this group runs no such part), and whether the
# group has asked. The warp's answer is that part's role, lowest CTA and
# parts, which it writes for the group (``WALK_ROLE``, ``WALK_LOWEST``,
# ``WALK_PARTS``). Words 7 and 13 to 15 of a record are unused.
WALK_LOOK = 6
WALK_LOOKED = 12
assert WALK_LOOKED < WALK_WORDS
# The named barriers of that look: the group arrives at the first when it
# has asked, the spare warp at the second, where the group meets it, when
# it has answered. Head ``h`` uses the barrier ``+ h``.
LOOK_BARRIER = 12
MEETING_BARRIER = 14
# The bound of a launch's skew (``SKEW_SHARE``) and of its active CTAs under
# Int32 slot products (``plan_grid_bound``) are the compile-time side's too:
# ``paged_attention_partition_bounds`` states each once.


def workspace_address(workspace: cute.Tensor) -> cutlass.Int64:
    """The address of a balanced launch's workspace: of its first record."""
    return cutlass.Int64(cast("cute.Pointer", workspace.iterator).toint())


def _load_pair(address: cutlass.Int64) -> tuple[cutlass.Float32, cutlass.Float32]:
    """Two fp32 values another CTA stored, from L2."""
    return cast("tuple[cutlass.Float32, cutlass.Float32]", ldcg_v2_f32(address))


def _load_quad(
    address: cutlass.Int64,
) -> tuple[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32]:
    """Four fp32 values another CTA stored, from L2."""
    return cast(
        "tuple[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32]",
        ldcg_v4_f32(address),
    )


@cute.jit
def _slot_boundary(
    cta: cutlass.Int32,
    slots: cutlass.Int32,
    active: cutlass.Int32,
    skew: cutlass.Int32,
    slot_bits: cutlass.Constexpr,
    skewed: cutlass.Constexpr,
) -> cutlass.Int32:
    """The first slot of active CTA ``cta``; ``slots`` for ``cta == active``.

    ``skew`` is the applied skew: a fraction of the mean share, so the
    boundaries increase with the CTA, and zero for a single CTA.
    """
    boundary = cutlass.Int32(0)
    if cutlass.const_expr(slot_bits == 32):
        # ``cta * slots`` is at most ``plan_grid_bound * slots``, and the
        # skew's product a quarter of that.
        boundary = cutlass.Int32(cutlass.Uint32(cta * slots) // cutlass.Uint32(active))
        if cutlass.const_expr(skewed):
            if skew > 0:
                boundary = boundary + cutlass.Int32(
                    cutlass.Uint32(skew * cta * (active - cta))
                    // cutlass.Uint32(active - 1)
                )
    else:
        boundary = cutlass.Int32(
            cutlass.Int64(cta) * cutlass.Int64(slots) // cutlass.Int64(active)
        )
        if cutlass.const_expr(skewed):
            if skew > 0:
                boundary = boundary + cutlass.Int32(
                    cutlass.Int64(skew)
                    * cutlass.Int64(cta)
                    * cutlass.Int64(active - cta)
                    // cutlass.Int64(active - 1)
                )
    return boundary


@cute.jit
def _slot_cta(
    slot: cutlass.Int32,
    slots: cutlass.Int32,
    active: cutlass.Int32,
    skew: cutlass.Int32,
    slot_bits: cutlass.Constexpr,
    skewed: cutlass.Constexpr,
) -> cutlass.Int32:
    """The active CTA whose range holds ``slot``.

    Without a skew the ranges are the equal shares, whose owner is one
    division. A skew moves the boundary of CTA ``c`` up by ``skew * c *
    (active - c) // (active - 1)`` slots, about ``skew * active / 4`` at the
    middle CTAs, so the owner is the equal shares' owner or a CTA that many
    slots below it. The loop walks down one CTA at a time, two divisions
    each: at most ``1 + (skew * (active + 1) / 4 + 1) / share`` steps for a
    mean share of ``share`` slots, which the applied skew's bound of
    ``share / SKEW_SHARE`` keeps near ``active / 128`` (two on 148 CTAs).
    One thread of a compute group runs it, once per part of a split unit
    (``split_arrival``), while the group waits for it at its next barrier.
    """
    cta = cutlass.Int32(0)
    if cutlass.const_expr(slot_bits == 32):
        cta = cutlass.Int32(
            cutlass.Uint32((slot + 1) * active - 1) // cutlass.Uint32(slots)
        )
    else:
        cta = cutlass.Int32(
            (cutlass.Int64(slot + 1) * cutlass.Int64(active) - 1)
            // cutlass.Int64(slots)
        )
    if cutlass.const_expr(skewed):
        if skew > 0:
            boundary = _slot_boundary(cta, slots, active, skew, slot_bits, skewed)
            while boundary > slot:
                cta = cta - 1
                boundary = _slot_boundary(cta, slots, active, skew, slot_bits, skewed)
    return cta


@cute.jit
def unit_segment(
    header: cutlass.Int32,
    count: cutlass.Int64,
    begin: cutlass.Int32,
    end: cutlass.Int32,
) -> tuple[cutlass.Int64, cutlass.Int64]:
    """The steps of a unit that a CTA of the balanced partition runs.

    ``header`` is the unit's header slot, ``count`` its pages and
    ``[begin, end)`` the CTA's slots. Returns the pages the CTA runs and
    their first step: slot ``header + 1 + j`` is step ``count - 1 - j``.
    """
    low = header + 1
    if begin > low:
        low = begin
    high = header + cutlass.Int32(count) + 1
    if end < high:
        high = end
    pages = cutlass.Int64(0)
    step0 = cutlass.Int64(0)
    if high > low:
        pages = cutlass.Int64(high - low)
        step0 = count - cutlass.Int64(high - header - 1)
    return pages, step0


@cute.jit
def unit_page_count(
    first_page: cutlass.Int64,
    last_page: cutlass.Int64,
    page_limit: cutlass.Constexpr,
) -> cutlass.Int64:
    """Pages a unit streams: its trip count, at most the page table's extent.

    A source that stays inside its page table never has more trips than the
    table has pages, so the clamp is the identity for it. It bounds what a
    CTA's roles sum, and a difference that wrapped counts as no trip.
    """
    count = cutlass.Int64(0)
    if last_page > first_page:
        count = last_page - first_page
    if count > page_limit:
        count = cutlass.Int64(page_limit)
    if count < 0:
        count = cutlass.Int64(0)
    return count


@cute.jit
def _unit_slots(
    unit: cutlass.Int32,
    page_limit: cutlass.Constexpr,
    unit_bounds: cutlass.Constexpr,
    unit_bounds_args: tuple,
) -> cutlass.Int32:
    """Slots of a unit: its header and one per page."""
    _item, first_page, last_page, _length = unit_bounds(unit, *unit_bounds_args)
    return cutlass.Int32(unit_page_count(first_page, last_page, page_limit)) + 1


@cute.jit
def partition_scan(
    prefix: cute.Tensor,
    totals: cute.Tensor,
    thread: cutlass.Int32,
    lane: cutlass.Int32,
    units: cutlass.Constexpr,
    page_limit: cutlass.Constexpr,
    unit_bounds: cutlass.Constexpr,
    unit_bounds_args: tuple,
    lengths: cute.Tensor | None = None,
) -> None:
    """Fill the plan tables. Every thread of the CTA calls this once.

    ``prefix[t]`` becomes the slots of the units scanned by thread ``t`` and
    by the lower lanes of its warp, ``totals[w]`` those of warp ``w``. Each
    word has one writer; a barrier of the caller publishes them. The slots
    of all units fit an Int32 (``units * (page_limit + 1) < 2**31``).

    With one unit per thread the scan can keep what it loaded: ``lengths[u]``
    becomes unit ``u``'s length, the fourth value of ``unit_bounds``, for
    the roles that evaluate the unit again behind that barrier.
    """
    chunk, warps = plan_geometry(units)
    assert cutlass.const_expr(units * (page_limit + 1) < 2**31)
    assert cutlass.const_expr(lengths is None or chunk == 1)
    if thread < warps * 32:
        slots = cutlass.Int32(0)
        if cutlass.const_expr(chunk == 1):
            if thread < units:
                _item, first_page, last_page, length = unit_bounds(
                    thread, *unit_bounds_args
                )
                slots = (
                    cutlass.Int32(unit_page_count(first_page, last_page, page_limit))
                    + 1
                )
                if cutlass.const_expr(lengths is not None):
                    assert lengths is not None
                    lengths[thread] = length
        else:
            for index in cutlass.range(chunk):
                unit = thread * chunk + index
                if unit < units:
                    slots = slots + _unit_slots(
                        unit, page_limit, unit_bounds, unit_bounds_args
                    )
        for shift in cutlass.range_constexpr(5):
            lower = cutlass.Int32(cute.arch.shuffle_sync_up(slots, 1 << shift))
            if lane >= (1 << shift):
                slots = slots + lower
        prefix[thread] = slots
        if lane == 31:
            totals[thread // 32] = slots


@cute.jit
def _locate_slot(
    prefix: cute.Tensor,
    base: cutlass.Int32,
    lane: cutlass.Int32,
    slot: cutlass.Int32,
    units: cutlass.Constexpr,
    page_limit: cutlass.Constexpr,
    unit_bounds: cutlass.Constexpr,
    unit_bounds_args: tuple,
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
    """The unit that holds ``slot``, its header slot and the next unit's.

    All 32 lanes call this with the same ``slot`` below the total. ``base``
    is the lane's value of the warps' exclusive slot sums: lane ``w`` holds
    the slots ahead of scanning warp ``w``.
    """
    chunk, warps = plan_geometry(units)
    ahead = base <= slot
    if lane >= warps:
        ahead = cutlass.Boolean(False)
    # The last warp that starts at or before the slot holds it: the sums do
    # not decrease, and only trailing warps are empty.
    warp = cutlass.Int32(cute.arch.popc(cute.arch.vote_ballot_sync(ahead))) - 1
    header = cutlass.Int32(cute.arch.shuffle_sync(base, warp))
    row = cutlass.Int32(prefix[warp * 32 + lane])
    # Lanes whose units end at or before the slot; the next lane holds it.
    thread = cutlass.Int32(
        cute.arch.popc(cute.arch.vote_ballot_sync(header + row <= slot))
    )
    following = header + cutlass.Int32(cute.arch.shuffle_sync(row, thread))
    previous = thread - 1
    if thread == 0:
        previous = cutlass.Int32(0)
    before = cutlass.Int32(cute.arch.shuffle_sync(row, previous))
    if thread > 0:
        header = header + before
    unit = (warp * 32 + thread) * chunk
    if cutlass.const_expr(chunk > 1):
        # The thread scanned several units: walk them up to the slot's.
        slots = _unit_slots(unit, page_limit, unit_bounds, unit_bounds_args)
        for _ in cutlass.range(chunk - 1):
            if header + slots <= slot:
                if unit + 1 < units:
                    header = header + slots
                    unit = unit + 1
                    slots = _unit_slots(unit, page_limit, unit_bounds, unit_bounds_args)
        following = header + slots
    return unit, header, following


@cute.jit
def partition_plan(
    prefix: cute.Tensor,
    totals: cute.Tensor,
    lane: cutlass.Int32,
    units: cutlass.Constexpr,
    page_limit: cutlass.Constexpr,
    slot_bits: cutlass.Constexpr,
    unit_bounds: cutlass.Constexpr,
    unit_bounds_args: tuple,
    partition: cutlass.Constexpr = "whole",
    skew: cutlass.Constexpr = 0,
    workspace_ctas: cutlass.Constexpr = 0,
) -> tuple[
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int64,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
    cutlass.Int32,
]:
    """This CTA's units ``[first, last)`` and the pages it streams over them,
    then its slots ``[begin, end)``, the header slot of unit ``first``, the
    slots of all units, the active CTAs and the applied skew.

    All 32 lanes of a warp call this behind a barrier that follows every
    thread's ``partition_scan``; any warp gets the same values. A CTA beyond
    the active ones gets an empty range of units, as does one that runs no
    unit: under ``whole`` when its slots hold no unit's header, under
    ``balanced`` when its only slot is the header of a unit with pages.

    ``slot_bits`` is 32 when ``plan_grid_bound`` admits every grid the
    launch can have: the active CTAs are then capped at that bound, so a
    larger grid only idles CTAs. Otherwise the slot products are Int64.
    ``workspace_ctas`` caps them at the records of a balanced launch's
    workspace. ``skew`` moves the boundaries (``_slot_boundary``); what is
    applied is at most the mean share over ``SKEW_SHARE``.
    """
    assert cutlass.const_expr(slot_bits in (32, 64))
    assert cutlass.const_expr(partition in ("whole", "balanced"))
    assert cutlass.const_expr(skew >= 0 and (skew == 0 or partition == "balanced"))
    assert cutlass.const_expr(workspace_ctas > 0 or partition == "whole")
    skewed = skew != 0
    _chunk, warps = plan_geometry(units)
    total = cutlass.Int32(0)
    if lane < warps:
        total = cutlass.Int32(totals[lane])
    base = total
    for shift in cutlass.range_constexpr(5):
        if cutlass.const_expr((1 << shift) < warps):
            lower = cutlass.Int32(cute.arch.shuffle_sync_up(base, 1 << shift))
            if lane >= (1 << shift):
                base = base + lower
    slots = cutlass.Int32(cute.arch.shuffle_sync(base, warps - 1))
    base = base - total
    cta = cutlass.Int32(cute.arch.block_idx()[0])
    active = cutlass.Int32(cute.arch.grid_dim()[0])
    if active > slots:
        active = slots
    if cutlass.const_expr(slot_bits == 32):
        bound = plan_grid_bound(units, page_limit)
        if active > bound:
            active = cutlass.Int32(bound)
    if cutlass.const_expr(partition == "balanced"):
        if active > workspace_ctas:
            active = cutlass.Int32(workspace_ctas)
    applied = cutlass.Int32(0)
    if cutlass.const_expr(skewed):
        # Every share is at least the mean's floor less the skew, which is
        # at most a thirty-second of that floor: no range is empty.
        applied = cutlass.Int32(
            cutlass.Uint32(slots)
            // cutlass.Uint32(active)
            // cutlass.Uint32(SKEW_SHARE)
        )
        if applied > skew:
            applied = cutlass.Int32(skew)
        if active == 1:
            applied = cutlass.Int32(0)
    first = cutlass.Int32(units)
    last = cutlass.Int32(units)
    pages = cutlass.Int32(0)
    begin = cutlass.Int32(0)
    end = cutlass.Int32(0)
    header = cutlass.Int32(0)
    if cta < active:
        begin = _slot_boundary(cta, slots, active, applied, slot_bits, skewed)
        end = _slot_boundary(cta + 1, slots, active, applied, slot_bits, skewed)
        unit, header, following = _locate_slot(
            prefix, base, lane, begin, units, page_limit, unit_bounds, unit_bounds_args
        )
        last_unit, last_header, last_following = _locate_slot(
            prefix,
            base,
            lane,
            end - 1,
            units,
            page_limit,
            unit_bounds,
            unit_bounds_args,
        )
        if cutlass.const_expr(partition == "balanced"):
            # Every unit with a page slot in the range, and the units
            # without pages whose header is in it. A range that ends with
            # the header of a unit with pages leaves that unit to later CTAs.
            first = unit
            last = last_unit + 1
            if last_header == end - 1:
                if last_following - last_header > 1:
                    last = last_unit
            headers = last_unit - unit + 1
            if header < begin:
                headers = headers - 1
            pages = (end - begin) - headers
        else:
            # The range starts at a unit's header or inside the unit, which
            # then belongs to an earlier CTA.
            first = unit
            if header < begin:
                first = unit + 1
                header = following
            last = last_unit + 1
            # Each owned unit has one slot that is not a page.
            pages = (last_following - header) - (last - first)
    return (
        first,
        last,
        cutlass.Int64(pages),
        begin,
        end,
        header,
        slots,
        active,
        applied,
    )


@cute.jit
def _count_arrival(
    workspace: cutlass.Int64,
    lowest: cutlass.Int32,
    others: cutlass.Int32,
    head: cutlass.Int32 | int,
    record_bytes: cutlass.Constexpr,
    arrive: cutlass.Constexpr,
) -> cutlass.Boolean:
    """One thread's access to the counter of a split unit: the word of head
    ``head`` in the record of the unit's lowest CTA, ``lowest``. Returns
    whether every part of the unit has now arrived, and resets the counter
    if so: the caller's group merges, and no part reads the counter again.

    ``others`` is the unit's parts less one. With ``arrive`` the thread adds
    its part's arrival, which releases the stores its group joined ahead of
    it, and that arrival is the last if the counter held ``others``. Without
    it the thread only looks, once, for a part that merges without arriving
    if the counter already holds ``others``; it then acquires what those
    arrivals released. Nothing here waits.
    """
    counter = workspace + cutlass.Int64(lowest) * record_bytes + cutlass.Int64(head) * 4
    if cutlass.const_expr(arrive):
        fence_acq_rel_gpu()
        seen = cutlass.Int32(atom_add_acq_rel_gpu_u32(counter, 1))
    else:
        seen = cutlass.Int32(ld_relaxed_gpu_u32(counter))
    last = seen == others
    if last:
        if cutlass.const_expr(not arrive):
            fence_acq_rel_gpu()
        # A counter's line is no partial's (``store_part``): one line a CTA,
        # which the next launch's arrivals and looks touch again. It keeps
        # the evict-last hint.
        stg_u32_l2(counter, 0, l2_policy_evict_last())
    return last


@cute.jit
def _note_post(
    walk: cute.Tensor,
    mail: cute.Tensor,
    lowest: cutlass.Int32,
    highest: cutlass.Int32,
    item: cutlass.Int32,
) -> None:
    """The deciding thread's note of a part it posts (``split_arrival``):
    the note in the group's record ``walk`` and, in ``mail``, what the
    spare warp needs to arrive for the part, a unit of the CTAs ``lowest``
    to ``highest``, and the unit's first item."""
    walk[WALK_POSTED] = cutlass.Int32(1)
    mail[MAIL_KIND] = cutlass.Int32(MAIL_PART)
    mail[MAIL_LOWEST] = lowest
    mail[MAIL_PARTS] = highest - lowest + 1
    mail[MAIL_ITEM] = item


@cute.jit
def split_arrival(
    workspace: cutlass.Int64,
    walk: cute.Tensor,
    mail: cute.Tensor,
    plan: cute.Tensor,
    header: cutlass.Int32,
    pages: cutlass.Int32,
    step0: cutlass.Int32,
    count: cutlass.Int32,
    item: cutlass.Int32,
    pair_head: cutlass.Int32,
    record_bytes: cutlass.Constexpr,
    slot_bits: cutlass.Constexpr,
    skewed: cutlass.Constexpr,
    posts: cutlass.Constexpr = False,
) -> None:
    """Decide what this part of a split unit does: one thread of the group.

    The unit has ``pages`` pages from header slot ``header``; this CTA ran
    its steps ``[step0, step0 + count)``. The unit's parts are the CTAs
    that hold its page slots, consecutive ones from the CTA with its last
    step up to the CTA with its first. A part that is not the lowest is
    posted to the spare warp (``PART_POST``): ``mail`` receives what that
    warp needs to arrive for it and ``item``, the unit's first item, for a
    merge at the end of the walk, and the group's record notes the post
    (``WALK_POSTED``). Nothing else decides a post: the group arrives at
    the post's barrier by the role, and at the end of its walk a group
    whose record has no note posts that it has no part. The lowest part
    first reads the unit's counter: if it shows that every other part
    arrived, this part merges (``PART_MERGE``, or ``PART_MERGE_PAIR`` for a
    unit in two parts). It then acquires their partials and resets the
    counter, which no other part reads again. Otherwise it publishes itself
    (``PART_PUBLISH``). The group's record ``walk`` receives the role, the
    unit's lowest CTA and its parts; a group barrier of the caller then
    publishes them. The caller's thread has passed the launch's dependency
    wait. Nothing here waits for another CTA: the counter is read once, and
    the only loop is the walk over CTAs that finds a part's owner under a
    skew (``_slot_cta``), which reads no memory.

    ``posts``: the caller's part is one to post, and the thread need not
    have waited. Where the spare warp looks (``look_parts``) that warp
    decides for a unit's lowest part and leaves the role, the lowest CTA
    and the parts in ``walk`` itself, so the group's thread comes here for
    a part to post alone, which reads the plan and no workspace.
    """
    cta = cutlass.Int32(cute.arch.block_idx()[0])
    slots = cutlass.Int32(plan[PLAN_SLOTS])
    active = cutlass.Int32(plan[PLAN_ACTIVE])
    skew = cutlass.Int32(plan[PLAN_SKEW])
    highest = cta
    if step0 != 0:
        highest = _slot_cta(header + pages, slots, active, skew, slot_bits, skewed)
    lowest = cta
    role = cutlass.Int32(PART_PUBLISH)
    if cutlass.const_expr(posts):
        lowest = _slot_cta(header + 1, slots, active, skew, slot_bits, skewed)
        role = cutlass.Int32(PART_POST)
        _note_post(walk, mail, lowest, highest, item)
    else:
        if step0 + count != pages:
            lowest = _slot_cta(header + 1, slots, active, skew, slot_bits, skewed)
            role = cutlass.Int32(PART_POST)
            _note_post(walk, mail, lowest, highest, item)
        else:
            if _count_arrival(
                workspace, cta, highest - cta, pair_head, record_bytes, arrive=False
            ):
                role = cutlass.Int32(PART_MERGE)
                if highest == cta + 1:
                    role = cutlass.Int32(PART_MERGE_PAIR)
    walk[WALK_ROLE] = role
    walk[WALK_LOWEST] = lowest
    walk[WALK_PARTS] = highest - lowest + 1


@cute.jit
def _partial_slot(
    workspace: cutlass.Int64,
    cta: cutlass.Int32,
    lowest: cutlass.Int32,
    pair_head: cutlass.Int32,
    record_bytes: cutlass.Constexpr,
) -> cutlass.Int64:
    """Where part ``cta`` of the unit whose lowest CTA is ``lowest`` publishes.

    A CTA's second pair of slots holds its part of the unit it is the lowest
    CTA of; the first pair its part of a unit that began in a lower CTA.
    """
    slot_bytes = (record_bytes - RECORD_HEADER_BYTES) // 4
    index = pair_head
    if cta == lowest:
        index = pair_head + 2
    return (
        workspace
        + cutlass.Int64(cta) * record_bytes
        + RECORD_HEADER_BYTES
        + cutlass.Int64(index) * slot_bytes
    )


@cute.jit
def store_part(
    workspace: cutlass.Int64,
    walk: cute.Tensor,
    state: cute.Tensor,
    panel: cute.Tensor,
    pair_head: cutlass.Int32,
    thread: cutlass.Int32,
    record_bytes: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
) -> None:
    """Store this part of a split unit into this CTA's slot for it.

    All 128 threads of the group call this with their part's panel in
    ``panel`` and the rows' maxima and denominators in ``state``. A group
    that goes on to merge the unit reads its own part back with the others.
    """
    panel_bytes, slot_bytes = partial_slot_bytes(query_columns)
    assert cutlass.const_expr(record_bytes == RECORD_HEADER_BYTES + 4 * slot_bytes)
    cta = cutlass.Int32(cute.arch.block_idx()[0])
    lowest = cutlass.Int32(walk[WALK_LOWEST])
    slot = _partial_slot(workspace, cta, lowest, pair_head, record_bytes)
    # A partial's hint is a trade, which the measurements in the README
    # decide. In the common case, a unit in two parts that its lowest part
    # merges, the other part's partial waits for the merge while the CTAs
    # stream the rest of their shares through L2 (a CTA runs a part that
    # is not its unit's lowest first, and a lowest part last): under this
    # hint the merge's loads may miss L2. Against that, one merge uses a
    # partial. (Only the panel's quads are read once on every path: the
    # general merge reads every part, its own included, and the rows'
    # pairs twice, and a spare warp's look reads the pairs as well.) Lines
    # kept in L2 for that are taken from the key and value stream of every
    # CTA, and how much they cost depends on where the workspace lies
    # among the process's allocations: these lines are the first the
    # stream may displace. One policy word a thread, ahead of its stores.
    policy = l2_policy_evict_first()
    for quad in cutlass.range_constexpr(query_columns // 4):
        stg_v4_f32_l2(
            slot + cutlass.Int64(quad * 128 + thread) * 16,
            cutlass.Float32(panel[quad * 4]),
            cutlass.Float32(panel[quad * 4 + 1]),
            cutlass.Float32(panel[quad * 4 + 2]),
            cutlass.Float32(panel[quad * 4 + 3]),
            policy,
        )
    if thread < query_columns:
        stg_v2_f32_l2(
            slot + panel_bytes + cutlass.Int64(thread) * 8,
            cutlass.Float32(state[0, thread]),
            cutlass.Float32(state[1, thread]),
            policy,
        )


@cute.jit
def arrive_part(
    workspace: cutlass.Int64,
    walk: cute.Tensor,
    pair_head: cutlass.Int32,
    thread: cutlass.Int32,
    record_bytes: cutlass.Constexpr,
) -> cutlass.Int32:
    """Arrive with the part ``store_part`` stored; return ``PART_MERGE`` for
    the last arrival and ``PART_DONE`` otherwise.

    The group's barrier joins the stores of its threads and one of them
    releases them with the arrival: a part that sees this arrival, or a
    later one, sees the partial. The last arrival resets the counter and
    its group merges; every other part is done with the unit. The second
    barrier publishes the role.
    """
    cute.arch.barrier(barrier_id=1 + pair_head * 2, number_of_threads=128)
    if thread == 0:
        role = cutlass.Int32(PART_DONE)
        if _count_arrival(
            workspace,
            cutlass.Int32(walk[WALK_LOWEST]),
            cutlass.Int32(walk[WALK_PARTS]) - 1,
            pair_head,
            record_bytes,
            arrive=True,
        ):
            role = cutlass.Int32(PART_MERGE)
        walk[WALK_ROLE] = role
    # The arrival is a memory round trip in one lane; the barrier below
    # publishes the role that lane wrote.
    cute.arch.barrier(barrier_id=1 + pair_head * 2, number_of_threads=128)
    return cutlass.Int32(walk[WALK_ROLE])


@cute.jit
def _part_weight(
    maximum: cutlass.Float32,
    safe_maximum: cutlass.Float32,
    base2_scores: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
) -> cutlass.Float32:
    """``b**(m - M)`` for a part's row; zero for a row that saw no key."""
    weight = cutlass.Float32(0)
    if maximum != float("-inf"):
        if cutlass.const_expr(base2_scores):
            weight = cute.math.exp2(maximum - safe_maximum, fastmath=fast_math)
        else:
            weight = cute.math.exp(maximum - safe_maximum, fastmath=fast_math)
    return weight


@cute.jit
def _fused_sum(
    value: cutlass.Float32, weight: cutlass.Float32, total: cutlass.Float32
) -> cutlass.Float32:
    """``value * weight + total`` as one fused operation, wherever it is."""
    fused, _unused = cute.arch.fma_packed_f32x2(
        (value, cutlass.Float32(0)),
        (weight, cutlass.Float32(0)),
        (total, cutlass.Float32(0)),
    )
    return fused


@cute.jit
def _fold_part(
    panel: cute.Tensor,
    values: cute.Tensor | cutlass.Array,
    weights: cute.Tensor,
    row: cutlass.Constexpr,
    first: cutlass.Constexpr,
    count: cutlass.Constexpr,
    initial: cutlass.Boolean,
) -> None:
    """``panel[first + i] = values[i] * weights[row, first + i] + panel[first
    + i]`` for ``count`` values, fused; onto zero when ``initial``: a unit's
    first term, whose sum so far is not in ``panel`` yet."""
    for pair in cutlass.range_constexpr(count // 2):
        query = first + pair * 2
        t0 = cutlass.Float32(0)
        t1 = cutlass.Float32(0)
        if not initial:
            t0 = cutlass.Float32(panel[query])
            t1 = cutlass.Float32(panel[query + 1])
        panel[query], panel[query + 1] = cute.arch.fma_packed_f32x2(
            (
                cutlass.Float32(values[pair * 2]),
                cutlass.Float32(values[pair * 2 + 1]),
            ),
            (
                cutlass.Float32(weights[row, query]),
                cutlass.Float32(weights[row, query + 1]),
            ),
            (t0, t1),
        )


@cute.jit
def pair_partial(
    workspace: cutlass.Int64,
    panel: cute.Tensor,
    pair_head: cutlass.Int32,
    thread: cutlass.Int32,
    record_bytes: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    staged: cute.Tensor | None,
    capture_pdl_wait_mode: cutlass.Int32,
) -> tuple[cutlass.Float32, cutlass.Float32]:
    """Load the other part of a unit in two parts that this CTA is the lowest
    part of: its panel into ``panel``, and for a row owner the row's
    maximum and denominator.

    The other part is the next CTA's, in that CTA's first pair of slots.
    The caller's group has acquired it. The loads start here so that they
    complete while the group still waits for its own part's last PV.

    ``staged``: the spare warp acquired the part and left every row's pair
    in shared memory (``look_parts``), which the group has met since. A row
    owner reads its pair there: the same two values. That warp also copies
    the panel's leading columns into shared memory (``stage_panels``), and
    the fold reads them there (``merge_pair_staged``): only the columns
    behind them are loaded here, the last eight of forty. The caller's
    thread has then not waited for the launch's dependencies behind this
    part's pages, as no thread of a group reads the counter where the spare
    warp looks: the loads are its first access to the workspace, and the
    wait (``capture_pdl_wait_mode``) stands in front of them.
    """
    panel_bytes, _slot_bytes = partial_slot_bytes(query_columns)
    cta = cutlass.Int32(cute.arch.block_idx()[0])
    slot = _partial_slot(workspace, cta + 1, cta, pair_head, record_bytes)
    # The quads of columns the stage holds: none where the group looks.
    held = partition_look_panel_columns(query_columns) // 4 if staged is not None else 0
    if cutlass.const_expr(staged is not None and held < query_columns // 4):
        if capture_pdl_wait_mode != 0:
            cute.arch.griddepcontrol_wait()
    for quad in cutlass.range_constexpr(held, query_columns // 4):
        query = quad * 4
        panel[query], panel[query + 1], panel[query + 2], panel[query + 3] = _load_quad(
            slot + cutlass.Int64(quad * 128 + thread) * 16
        )
    maximum = cutlass.Float32(float("-inf"))
    denominator = cutlass.Float32(0)
    if thread < query_columns:
        if cutlass.const_expr(staged is not None):
            assert staged is not None
            pair = (pair_head * query_columns + thread) * 2
            maximum = cutlass.Float32(staged[pair])
            denominator = cutlass.Float32(staged[pair + 1])
        else:
            maximum, denominator = _load_pair(
                slot + panel_bytes + cutlass.Int64(thread) * 8
            )
    return maximum, denominator


@cute.jit
def merge_pair_rows(
    state: cute.Tensor,
    scratch: cute.Tensor,
    thread: cutlass.Int32,
    own_denominator: cutlass.Float32,
    other_maximum: cutlass.Float32,
    other_denominator: cutlass.Float32,
    base2_scores: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
    output_reciprocal: cutlass.Constexpr,
    output_scale_log2: cutlass.Constexpr,
) -> None:
    """A row owner's share of merging a unit in two parts: the weights of
    the other part and of this CTA's into ``scratch`` rows 0 and 1, and the
    unit's maximum, denominator and normalization into ``state`` rows 0, 1
    and 3. The same operations, in the same order, as ``merge_parts``.
    """
    own_maximum = cutlass.Float32(state[0, thread])
    maximum = cute.arch.fmax(own_maximum, other_maximum, nan=True)
    safe_maximum = maximum
    if maximum == float("-inf"):
        safe_maximum = cutlass.Float32(0.0)
    other_weight = _part_weight(other_maximum, safe_maximum, base2_scores, fast_math)
    own_weight = _part_weight(own_maximum, safe_maximum, base2_scores, fast_math)
    denominator = _fused_sum(other_denominator, other_weight, cutlass.Float32(0))
    denominator = _fused_sum(own_denominator, own_weight, denominator)
    scratch[0, thread] = other_weight
    scratch[1, thread] = own_weight
    state[0, thread] = maximum
    state[1, thread] = denominator
    safe_total = denominator
    if denominator == 0:
        safe_total = cutlass.Float32(1)
    state[3, thread] = output_normalization_factor(
        safe_total, output_reciprocal, output_scale_log2=output_scale_log2
    )


@cute.jit
def merge_pair_other(
    scratch: cute.Tensor, panel: cute.Tensor, query_columns: cutlass.Constexpr
) -> None:
    """Weight the other part's panel in ``panel``: a unit's first term."""
    _fold_part(panel, panel, scratch, 0, 0, query_columns, cutlass.Boolean(True))


@cute.jit
def merge_pair_own(
    scratch: cute.Tensor,
    panel: cute.Tensor,
    values: cute.Tensor | cutlass.Array,
    first: cutlass.Constexpr,
    count: cutlass.Constexpr,
) -> None:
    """Fold ``count`` columns of this CTA's part, from column ``first``,
    onto the other part's term in ``panel``: a unit's last term."""
    _fold_part(panel, values, scratch, 1, first, count, cutlass.Boolean(False))


@cute.jit
def merge_pair_loaded(
    scratch: cute.Tensor, panel: cute.Tensor, query_columns: cutlass.Constexpr
) -> None:
    """Weight the columns of the other part's panel that ``pair_partial``
    loaded into ``panel`` where the spare warp staged the leading ones
    (``stage_panels``): those columns' share of ``merge_pair_other``, a
    unit's first term, the same fused operation onto zero."""
    first = partition_look_panel_columns(query_columns)
    for pair in cutlass.range_constexpr(first // 2, query_columns // 2):
        query = pair * 2
        panel[query], panel[query + 1] = cute.arch.fma_packed_f32x2(
            (cutlass.Float32(panel[query]), cutlass.Float32(panel[query + 1])),
            (
                cutlass.Float32(scratch[0, query]),
                cutlass.Float32(scratch[0, query + 1]),
            ),
            (cutlass.Float32(0), cutlass.Float32(0)),
        )


@cute.jit
def merge_pair_staged(
    scratch: cute.Tensor,
    panel: cute.Tensor,
    own: cute.Tensor | cutlass.Array,
    stage: cute.Tensor,
    thread: cutlass.Int32,
    query_columns: cutlass.Constexpr,
) -> None:
    """Fold a unit in two parts whose other part's panel the spare warp
    staged (``stage_panels``): ``own`` is this CTA's part, all columns of
    it, and ``stage`` the head's raw K tile (``stage_tile``), which holds
    the other part's leading columns as the workspace does, a quad of
    columns per thread.

    Per element the unit's first term, the other part's value times its
    weight onto zero, and its last, this part's value times its weight
    onto the first: ``merge_pair_other`` and ``merge_pair_own``, the same
    two fused operations with the same operands in the same order, so the
    same bits. A column the stage does not hold has its first term in
    ``panel`` already (``merge_pair_loaded``). The caller's thread has
    waited for the copies.
    """
    staged = partition_look_panel_columns(query_columns)
    for pair in cutlass.range_constexpr(query_columns // 2):
        query = pair * 2
        first0 = cutlass.Float32(panel[query])
        first1 = cutlass.Float32(panel[query + 1])
        if cutlass.const_expr(query < staged):
            word = ((query // 4) * 128 + thread) * 4 + query % 4
            first0, first1 = cute.arch.fma_packed_f32x2(
                (cutlass.Float32(stage[word]), cutlass.Float32(stage[word + 1])),
                (
                    cutlass.Float32(scratch[0, query]),
                    cutlass.Float32(scratch[0, query + 1]),
                ),
                (cutlass.Float32(0), cutlass.Float32(0)),
            )
        panel[query], panel[query + 1] = cute.arch.fma_packed_f32x2(
            (cutlass.Float32(own[query]), cutlass.Float32(own[query + 1])),
            (
                cutlass.Float32(scratch[1, query]),
                cutlass.Float32(scratch[1, query + 1]),
            ),
            (first0, first1),
        )


@cute.jit
def merge_parts(
    workspace: cutlass.Int64,
    lowest: cutlass.Int32,
    parts: cutlass.Int32,
    state: cute.Tensor,
    scratch: cute.Tensor,
    panel: cute.Tensor,
    pair_head: cutlass.Int32,
    thread: cutlass.Int32,
    record_bytes: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    base2_scores: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
    output_reciprocal: cutlass.Constexpr,
    output_scale_log2: cutlass.Constexpr,
) -> None:
    """Fold the parts of a split unit into ``panel``, its unnormalized panel.

    Every part kept, per row, a maximum ``m``, a denominator ``l`` and its
    panel ``O``, the sums of ``b**s`` and ``b**s * v`` over its keys divided
    by ``b**m`` (by one for a row that saw no key, whose ``m`` is minus
    infinity). With ``M`` the largest ``m``, the unit's denominator and
    panel are the sums of ``l * b**(m - M)`` and ``O * b**(m - M)``, each
    term one fused multiply-add onto the sum so far, the first onto zero.
    The terms follow the source: the unit's first steps (its highest CTA)
    first, its last steps (its lowest CTA) last. So the bits are the same
    whichever part merges. The row owners leave ``M``, the denominator and
    the normalization in ``state`` rows 0, 1 and 3, as a unit that ran in
    one CTA does.

    The unit's parts are those of the ``parts`` CTAs from ``lowest``. The
    caller's group has acquired the other parts' partials and stored its
    own (``store_part``): every part is read from the workspace, so the
    fold holds one panel in registers and needs nothing the group kept
    from its own part.

    The weights of up to four parts go through the four ``scratch`` rows
    between two group barriers, so a unit in up to four parts takes one.
    """
    assert cutlass.const_expr(fast_math)
    panel_bytes, _slot_bytes = partial_slot_bytes(query_columns)
    maximum = cutlass.Float32(float("-inf"))
    denominator = cutlass.Float32(0)
    if thread < query_columns:
        for other in cutlass.range(lowest, lowest + parts, 1):
            other_maximum, _other_denominator = _load_pair(
                _partial_slot(workspace, other, lowest, pair_head, record_bytes)
                + panel_bytes
                + cutlass.Int64(thread) * 8
            )
            maximum = cute.arch.fmax(maximum, other_maximum, nan=True)
    safe_maximum = maximum
    if maximum == float("-inf"):
        safe_maximum = cutlass.Float32(0.0)
    # Term ``folded + index`` is the part of CTA ``highest - (folded +
    # index)``.
    highest = lowest + parts - 1
    folded = cutlass.Int32(0)
    while folded < parts:
        if thread < query_columns:
            for index in cutlass.range_constexpr(4):
                if folded + index < parts:
                    part_maximum, part_denominator = _load_pair(
                        _partial_slot(
                            workspace,
                            highest - (folded + index),
                            lowest,
                            pair_head,
                            record_bytes,
                        )
                        + panel_bytes
                        + cutlass.Int64(thread) * 8
                    )
                    weight = _part_weight(
                        part_maximum, safe_maximum, base2_scores, fast_math
                    )
                    denominator = _fused_sum(part_denominator, weight, denominator)
                    scratch[index, thread] = weight
            if folded + 4 >= parts:
                state[0, thread] = maximum
                state[1, thread] = denominator
                safe_total = denominator
                if denominator == 0:
                    safe_total = cutlass.Float32(1)
                state[3, thread] = output_normalization_factor(
                    safe_total, output_reciprocal, output_scale_log2=output_scale_log2
                )
        # The row owners wrote the weights; the barrier publishes them to
        # every thread's fold.
        cute.arch.barrier(barrier_id=1 + pair_head * 2, number_of_threads=128)
        for index in cutlass.range_constexpr(4):
            if folded + index < parts:
                slot = _partial_slot(
                    workspace,
                    highest - (folded + index),
                    lowest,
                    pair_head,
                    record_bytes,
                )
                for quad in cutlass.range_constexpr(query_columns // 4):
                    values = cute.make_rmem_tensor((4,), cutlass.Float32)
                    values[0], values[1], values[2], values[3] = _load_quad(
                        slot + cutlass.Int64(quad * 128 + thread) * 16
                    )
                    if cutlass.const_expr(index == 0):
                        _fold_part(panel, values, scratch, 0, quad * 4, 4, folded == 0)
                    else:
                        _fold_part(
                            panel,
                            values,
                            scratch,
                            index,
                            quad * 4,
                            4,
                            cutlass.Boolean(False),
                        )
        folded = folded + 4
        if folded < parts:
            # The folds read the weight rows the owners rewrite next.
            cute.arch.barrier(barrier_id=1 + pair_head * 2, number_of_threads=128)


@cute.jit
def post_parts(
    workspace: cutlass.Int64,
    mail: cute.Tensor,
    lane: cutlass.Int32,
    capture_pdl_wait_mode: cutlass.Int32,
    record_bytes: cutlass.Constexpr,
) -> None:
    """Arrive for the parts the two compute groups post: the CTA's spare warp.

    Each group posts once: the part of a split unit that is not the unit's
    lowest, or that it has none. All 128 threads of the group arrive at the
    head's named barrier behind their stores of the part and this warp
    waits there, so the barrier joins the stores and one lane releases them
    with the arrival, as a group that arrives for itself does
    (``arrive_part``). The verdict replaces the post's kind: the last
    arrival resets the unit's counter, and the group merges the unit when
    it reads the verdict at the end of its walk. It reads it behind the
    second barrier, where it waits for this warp. A group that met the
    warp behind its lowest part under the spare warp's look
    (``look_parts``) does not wait there: the warp came to that meeting
    behind both verdicts, and the group's merge starts with a barrier of
    the group (``_compute_units``).
    The arrival is this warp's access to the workspace, so the lane waits
    for the launch's dependencies first. Both barriers are the non-aligned
    form, as the warp and the groups reach them at different instructions,
    and a barrier waits for the lanes of its warp: the warp's second
    arrival follows that lane's verdict.
    """
    for head in cutlass.range_constexpr(2):
        prims.barrier_cta_sync(POST_BARRIER + head, thread_count=POST_THREADS)
        if lane == 0:
            if cutlass.Int32(mail[head * MAIL_WORDS + MAIL_KIND]) == MAIL_PART:
                if capture_pdl_wait_mode != 0:
                    cute.arch.griddepcontrol_wait()
                verdict = cutlass.Int32(MAIL_ARRIVED)
                if _count_arrival(
                    workspace,
                    cutlass.Int32(mail[head * MAIL_WORDS + MAIL_LOWEST]),
                    cutlass.Int32(mail[head * MAIL_WORDS + MAIL_PARTS]) - 1,
                    head,
                    record_bytes,
                    arrive=True,
                ):
                    verdict = cutlass.Int32(MAIL_LAST)
                mail[head * MAIL_WORDS + MAIL_KIND] = verdict
        prims.barrier_cta_arrive(POSTED_BARRIER + head, thread_count=POST_THREADS)


@cute.jit
def look_parts(
    workspace: cutlass.Int64,
    walks: cute.Tensor,
    stage: cute.Tensor,
    lane: cutlass.Int32,
    capture_pdl_wait_mode: cutlass.Int32,
    record_bytes: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
) -> tuple[cutlass.Boolean, cutlass.Boolean]:
    """Look at the counter of the split unit this CTA runs the lowest part
    of, for both compute groups: the CTA's spare warp, behind its posts
    (``post_parts``), under ``cute_mma_dag_partition_look="spare"``.

    Each group asks once. All of its threads arrive at the head's first
    named barrier, without waiting, when the unit's lowest part starts,
    its record (``walks``, the two groups' records) holding the unit's
    other parts; a group that runs no such part asks for none at the end
    of its walk, which lets this warp finish. Lanes 0 and 1 look for heads
    0 and 1 at once: the one relaxed load of the counter and, if it shows
    every other part, the fence and the reset (``_count_arrival`` without
    an arrival), what one thread of the group does where the group looks.
    The lane leaves the answer in the group's record: the words that thread
    would have written (``split_arrival``), the part's role (to merge, as a
    pair for a unit in two parts, or to publish), this CTA as the unit's
    lowest and the unit's parts. The group arrives for the look when its
    part starts, so the look finds fewer units complete than one behind
    the part's pages; a unit it finds incomplete is published and merged
    by its last arrival.

    For a unit in two parts that is complete, which is the role to merge
    as a pair in the record of a group that asked, the warp's lanes load
    the other part's row statistics, a maximum and a denominator per row,
    into ``stage``: lane ``l`` the rows ``l`` and ``32 + l``. The lowest
    part's row owners read them there ahead of their reduction's second
    barrier (``pair_partial``) instead of loading them behind their own
    pages.

    Every lane that reads the workspace waits for the launch's dependencies
    itself. The warp then arrives at the head's second named barrier, where
    the group waits as the first barrier of its part's reduction: a barrier
    waits for the lanes of its warp, so the arrival follows the record's
    words and the staged rows. Nothing here waits for another CTA.

    Returns, per head, whether the warp staged the rows: the group asked
    and its unit is a complete pair. The warp then also stages the other
    part's panel (``stage_panels``); it tells from these values, since a
    group it has answered may rewrite its record.
    """
    assert cutlass.const_expr(query_columns in (8, 32, 40))
    for head in cutlass.range_constexpr(2):
        prims.barrier_cta_sync(LOOK_BARRIER + head, thread_count=POST_THREADS)
    request0 = cutlass.Int32(walks[WALK_LOOK])
    request1 = cutlass.Int32(walks[WALK_WORDS + WALK_LOOK])
    if lane < 2:
        request = request0
        if lane == 1:
            request = request1
        if request != 0:
            if capture_pdl_wait_mode != 0:
                cute.arch.griddepcontrol_wait()
            cta = cutlass.Int32(cute.arch.block_idx()[0])
            role = cutlass.Int32(PART_PUBLISH)
            if _count_arrival(
                workspace, cta, request, lane, record_bytes, arrive=False
            ):
                role = cutlass.Int32(PART_MERGE)
                if request == 1:
                    role = cutlass.Int32(PART_MERGE_PAIR)
            # What the group's own look leaves in its record
            # (``split_arrival``): the group's threads read these words
            # behind the meeting, and none of them decides.
            walks[lane * WALK_WORDS + WALK_ROLE] = role
            walks[lane * WALK_WORDS + WALK_LOWEST] = cta
            walks[lane * WALK_WORDS + WALK_PARTS] = request + 1
    # The roles are two lanes' words: every lane reads them behind this.
    cute.arch.sync_warp()
    panel_bytes, _slot_bytes = partial_slot_bytes(query_columns)
    paired = []
    for head in cutlass.range_constexpr(2):
        # The role is this warp's answer where the group asked. The record
        # of a group that asked for no look holds what its own thread left
        # there, or what the shared memory held: never an answer.
        pair = (
            cutlass.Int32(walks[head * WALK_WORDS + WALK_ROLE]) == PART_MERGE_PAIR
        ) & (cutlass.Int32(walks[head * WALK_WORDS + WALK_LOOK]) != 0)
        paired.append(pair)
        if pair:
            if capture_pdl_wait_mode != 0:
                cute.arch.griddepcontrol_wait()
            cta = cutlass.Int32(cute.arch.block_idx()[0])
            pairs = _partial_slot(workspace, cta + 1, cta, head, record_bytes)
            pairs = pairs + panel_bytes
            for first in cutlass.range_constexpr(0, query_columns, 32):
                row = first + lane
                if row < query_columns:
                    maximum, denominator = _load_pair(pairs + cutlass.Int64(row) * 8)
                    stage[(head * query_columns + row) * 2] = maximum
                    stage[(head * query_columns + row) * 2 + 1] = denominator
    if request0 != 0:
        prims.barrier_cta_arrive(MEETING_BARRIER, thread_count=POST_THREADS)
    if request1 != 0:
        prims.barrier_cta_arrive(MEETING_BARRIER + 1, thread_count=POST_THREADS)
    return paired[0], paired[1]


@cute.jit
def stage_tile(plan: cute.Tensor, head: cutlass.Int32 | int) -> cutlass.Int32:
    """The raw K tile in which the spare warp stages the other part's panel
    for head ``head`` (``stage_panels``): the ring's position behind the
    stream's last step, the tiles the stream's next step would take if it
    had one. ``plan`` is the CTA's plan, which holds the pages it streams:
    a role reads it behind the setup rendezvous, or behind a barrier that
    a role which waited there has arrived at."""
    return 2 * (cutlass.Int32(plan[PLAN_PAGES]) & 1) + head


@cute.jit
def stage_panels(
    workspace: cutlass.Int64,
    slots: cute.Pointer,
    plan: cute.Tensor,
    paired: tuple[cutlass.Boolean, cutlass.Boolean],
    converted: cute.Pointer,
    staged: cute.Pointer,
    lane: cutlass.Int32,
    capture_pdl_wait_mode: cutlass.Int32,
    record_bytes: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
) -> None:
    """Stage the other part's panel of a complete unit in two parts for the
    lowest part's fold, in the raw K ring: the CTA's spare warp, behind its
    look (``look_parts``, whose answer ``paired`` is: the heads whose
    group merges a pair).

    The other part's partial lies in the next CTA's record, the panel
    first, a quad of columns per thread (``store_part``), and the look
    acquired it. One bulk copy per head moves the panel's leading columns
    (``partition_look_panel_columns``: 32, or all of a narrower panel) into
    that head's raw K tile of ``slots`` (``stage_tile``): the bytes as the
    workspace holds them. Lane 0 issues the copies, behind the
    launch's dependency wait as every access to the workspace, and they
    complete on ``staged``, where the lowest part's group waits in front
    of its fold (``merge_pair_staged``). The warp has answered the group by
    then, so the meeting does not wait for the copies.

    The ring is the stream's until its last K tile is converted: the loader
    requests a tile into a slot, the four K conversion warps read it there,
    and the slot's own barriers pair those two by the parity of the slot's
    uses, which says nothing to a role that is not in step with them. So
    each K conversion warp arrives at ``converted`` once, behind its share
    of the last tile of the CTA's stream, and lane 0 waits for that phase,
    the barrier's only one: all four have then read every tile the loader
    requested, and the loader requests no more. The lowest part is the
    CTA's last range (its unit goes on in the next CTA, so the CTA's slots
    end inside it): the stream ends with that part's pages, the group's
    fold stands behind them, and no later range has a use for the ring.
    The plan that names the tile was published by the setup rendezvous,
    which the K conversions passed long before their arrival here.
    Nothing here waits for another CTA.
    """
    size = partition_look_panel_columns(query_columns) // 4 * PANEL_QUAD_BYTES
    assert cutlass.const_expr(0 < size <= RAW_TILE_BYTES and size % 16 == 0)
    copies = cutlass.Int32(0)
    for head in cutlass.range_constexpr(2):
        if paired[head]:
            copies = copies + 1
    if lane == 0:
        if copies != 0:
            if capture_pdl_wait_mode != 0:
                cute.arch.griddepcontrol_wait()
            # A wait of many pages, off every other role's path: suspended.
            mbar_wait_sleep(converted, 0)
            cta = cutlass.Int32(cute.arch.block_idx()[0])
            # The copies read through the async proxy what the other CTA's
            # threads stored and the looking lanes acquired.
            cute.arch.fence_proxy("async.global")
            prims.mbarrier_arrive_expect_tx(staged, copies * size)
            for head in cutlass.range_constexpr(2):
                if paired[head]:
                    tile = stage_tile(plan, head)
                    cp_bulk_global_to_shared(
                        slots
                        + cute.assume(tile * RAW_TILE_BYTES, divby=RAW_TILE_BYTES),
                        _partial_slot(workspace, cta + 1, cta, head, record_bytes),
                        size,
                        staged,
                    )
