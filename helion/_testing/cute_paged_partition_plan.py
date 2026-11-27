"""Reference model, decode mirror and device observer of the partition plan.

``reference_partition`` states which loop steps of which units each CTA of a
launch runs, straight from the slot definitions. ``planner_tables``,
``planner_plans`` and ``planner_units`` repeat, lane by lane, what the device
code in ``helion._compiler.cute._paged_attention_partition`` computes, so a
CPU test compares the two; ``observe_plans`` runs that device code on a GPU
and returns what every warp of every CTA found. ``simulate_merges`` runs the
balanced partition's exchange between the parts of split units as
interleaved step machines, for any order of dispatch and any number of CTAs
resident at once.
"""

from __future__ import annotations

import dataclasses
import importlib
import random
from typing import TYPE_CHECKING

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

from helion._compiler.cute._paged_attention_partition import SKEW_SHARE
from helion._compiler.cute._paged_attention_partition import partition_plan
from helion._compiler.cute._paged_attention_partition import partition_scan
from helion._compiler.cute._paged_attention_partition import plan_geometry
from helion._compiler.cute._paged_attention_partition import plan_grid_bound

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence

# ``whole`` gives a CTA the units whose header slot it owns. ``balanced``
# gives it the page slots themselves, so a unit can be split among CTAs.
PARTITIONS = ("whole", "balanced")
PLAN_WARPS = 20
# What ``partition_plan`` returns.
PLAN_VALUES = 9


@dataclasses.dataclass(frozen=True)
class PartitionSegment:
    """The loop steps ``[step, step + count)`` of ``unit`` that one CTA runs."""

    unit: int
    step: int
    count: int
    # The unit's pages: its trip count.
    pages: int

    @property
    def head(self) -> bool:
        """It holds the unit's first iteration (or the unit has none)."""
        return self.step == 0

    @property
    def tail(self) -> bool:
        """It holds the unit's last iteration (or the unit has none)."""
        return self.step + self.count == self.pages

    @property
    def whole(self) -> bool:
        return self.head and self.tail


def unit_headers(pages: Sequence[int]) -> list[int]:
    """Every unit's header slot, then the slots of all units.

    A unit with ``n`` pages owns ``n + 1`` consecutive slots: its header and,
    in descending step order, one per page (slot ``header + 1 + j`` is loop
    step ``n - 1 - j``).
    """
    headers = [0]
    for count in pages:
        headers.append(headers[-1] + count + 1)
    return headers


def active_ctas(
    pages: Sequence[int],
    grid: int,
    *,
    grid_bound: int | None = None,
    workspace_ctas: int | None = None,
) -> int:
    """The CTAs of a launch of ``grid`` that share the slots: at most one per
    slot, ``grid_bound`` and, for a balanced launch, ``workspace_ctas``."""
    active = min(grid, unit_headers(pages)[-1])
    if grid_bound is not None:
        active = min(active, grid_bound)
    if workspace_ctas is not None:
        active = min(active, workspace_ctas)
    return active


def applied_skew(total: int, active: int, skew: int) -> int:
    """The skew a launch applies: at most a thirty-second of the mean share."""
    if active == 1:
        return 0
    return min(skew, total // active // 32)


def slot_boundary(cta: int, total: int, active: int, skew: int) -> int:
    """The first slot of active CTA ``cta`` under the applied ``skew``.

    Equal shares, moved up by a parabola that is zero at both ends: the
    shares then fall linearly from the mean plus the skew to the mean less
    the skew.
    """
    boundary = cta * total // active
    if skew:
        boundary += skew * cta * (active - cta) // (active - 1)
    return boundary


def slot_ranges(
    pages: Sequence[int],
    grid: int,
    *,
    grid_bound: int | None = None,
    workspace_ctas: int | None = None,
    skew: int = 0,
) -> list[tuple[int, int]]:
    """The slots ``[begin, end)`` of each CTA of a launch of ``grid`` CTAs.

    The active CTAs share the slots, equally without a skew; the others get
    an empty range at the end.
    """
    total = unit_headers(pages)[-1]
    active = active_ctas(
        pages, grid, grid_bound=grid_bound, workspace_ctas=workspace_ctas
    )
    skew = applied_skew(total, active, skew)
    return [
        (
            slot_boundary(cta, total, active, skew),
            slot_boundary(cta + 1, total, active, skew),
        )
        if cta < active
        else (total, total)
        for cta in range(grid)
    ]


def reference_partition(
    pages: Sequence[int],
    grid: int,
    partition: str,
    *,
    grid_bound: int | None = None,
    workspace_ctas: int | None = None,
    skew: int = 0,
) -> list[list[PartitionSegment]]:
    """Each CTA's segments, in unit order, by the definitions.

    ``whole``: a unit runs entirely in the CTA that owns its header slot.
    ``balanced``: a CTA runs the steps whose page slots it owns; a unit
    without pages runs (as an empty segment) where its header is.
    """
    assert partition in PARTITIONS
    headers = unit_headers(pages)
    result = []
    for begin, end in slot_ranges(
        pages, grid, grid_bound=grid_bound, workspace_ctas=workspace_ctas, skew=skew
    ):
        segments = []
        for unit, count in enumerate(pages):
            header = headers[unit]
            if partition == "whole" or count == 0:
                if begin <= header < end:
                    segments.append(PartitionSegment(unit, 0, count, count))
                continue
            low = max(begin, header + 1)
            high = min(end, header + 1 + count)
            if high > low:
                segments.append(
                    PartitionSegment(
                        unit, count - (high - header - 1), high - low, count
                    )
                )
        result.append(segments)
    return result


def planner_tables(pages: Sequence[int]) -> tuple[list[int], list[int]]:
    """The tables ``partition_scan`` leaves in shared memory.

    ``prefix[t]``: the slots of the units thread ``t`` and the lower lanes
    of its warp scanned. ``totals[w]``: those of warp ``w``.
    """
    units = len(pages)
    chunk, warps = plan_geometry(units)
    prefix = []
    totals = []
    for warp in range(warps):
        slots = [
            sum(
                pages[unit] + 1
                for unit in range(t * chunk, min((t + 1) * chunk, units))
            )
            for t in range(warp * 32, warp * 32 + 32)
        ]
        # The warp's inclusive scan: five shuffle rounds.
        for shift in range(5):
            distance = 1 << shift
            slots = [
                slots[lane] + (slots[lane - distance] if lane >= distance else 0)
                for lane in range(32)
            ]
        prefix.extend(slots)
        totals.append(slots[31])
    return prefix, totals


def _popcount(votes: Iterable[bool]) -> int:
    """The set bits of a warp's ballot."""
    return sum(1 for vote in votes if vote)


def planner_units(
    pages: Sequence[int],
    grid: int,
    slot_bits: int,
    page_limit: int,
) -> list[tuple[int, int, int]]:
    """What ``partition_plan`` returns for ``whole`` in each CTA of a launch
    of ``grid``: its units ``[first, last)`` and the pages it streams."""
    return [plan[:3] for plan in planner_plans(pages, grid, slot_bits, page_limit)]


def _mirror_boundary(cta: int, slots: int, active: int, skew: int, bits: int) -> int:
    """``_slot_boundary``: the products are Int32 when ``bits`` is 32."""
    if bits == 32:
        assert cta * slots < 2**31 and skew * cta * (active - cta) < 2**31
    boundary = cta * slots // active
    if skew > 0:
        boundary += skew * cta * (active - cta) // (active - 1)
    return boundary


def planner_slot_cta(slot: int, slots: int, active: int, skew: int, bits: int) -> int:
    """``_slot_cta``: the active CTA whose range holds ``slot``."""
    if bits == 32:
        assert (slot + 1) * active - 1 < 2**31
    cta = ((slot + 1) * active - 1) // slots
    if skew > 0:
        while _mirror_boundary(cta, slots, active, skew, bits) > slot:
            cta -= 1
    return cta


def planner_plans(
    pages: Sequence[int],
    grid: int,
    slot_bits: int,
    page_limit: int,
    partition: str = "whole",
    skew: int = 0,
    workspace_ctas: int | None = None,
) -> list[tuple[int, int, int, int, int, int, int, int, int]]:
    """What ``partition_plan`` returns in each CTA of a launch of ``grid``.

    Its units ``[first, last)`` and the pages it streams over them, its
    slots ``[begin, end)``, the header slot of unit ``first``, the slots of
    all units, the active CTAs and the applied skew.
    """
    assert partition in PARTITIONS and (partition == "balanced" or skew == 0)
    units = len(pages)
    chunk, warps = plan_geometry(units)
    prefix, totals = planner_tables(pages)
    total = [totals[lane] if lane < warps else 0 for lane in range(32)]
    base = list(total)
    for shift in range(5):
        distance = 1 << shift
        if distance < warps:
            base = [
                base[lane] + (base[lane - distance] if lane >= distance else 0)
                for lane in range(32)
            ]
    slots = base[warps - 1]
    base = [base[lane] - total[lane] for lane in range(32)]
    assert slots < 2**31
    active = min(grid, slots)
    if slot_bits == 32:
        active = min(active, plan_grid_bound(units, page_limit))
        # The device multiplies in Int32.
        assert active * slots < 2**31
    if partition == "balanced":
        assert workspace_ctas is not None
        active = min(active, workspace_ctas)
    applied = 0
    if skew:
        applied = min(skew, slots // active // SKEW_SHARE)
        if active == 1:
            applied = 0

    def locate(slot: int) -> tuple[int, int, int]:
        warp = _popcount(lane < warps and base[lane] <= slot for lane in range(32)) - 1
        header = base[warp]
        row = prefix[warp * 32 : warp * 32 + 32]
        thread = _popcount(header + row[lane] <= slot for lane in range(32))
        following = header + row[thread]
        if thread > 0:
            header += row[thread - 1]
        unit = (warp * 32 + thread) * chunk
        if chunk > 1:
            weight = pages[unit] + 1
            for _ in range(chunk - 1):
                if header + weight <= slot and unit + 1 < units:
                    header += weight
                    unit += 1
                    weight = pages[unit] + 1
            following = header + weight
        return unit, header, following

    plans = []
    for cta in range(grid):
        if cta >= active:
            plans.append((units, units, 0, 0, 0, 0, slots, active, applied))
            continue
        begin = _mirror_boundary(cta, slots, active, applied, slot_bits)
        end = _mirror_boundary(cta + 1, slots, active, applied, slot_bits)
        unit, header, following = locate(begin)
        last_unit, last_header, last_following = locate(end - 1)
        first = unit
        last = last_unit + 1
        if partition == "balanced":
            if last_header == end - 1 and last_following - last_header > 1:
                last = last_unit
            headers = last_unit - unit + 1 - (header < begin)
            count = (end - begin) - headers
        else:
            if header < begin:
                first = unit + 1
                header = following
            count = (last_following - header) - (last - first)
        plans.append((first, last, count, begin, end, header, slots, active, applied))
    return plans


def split_units(
    segments: Sequence[Sequence[PartitionSegment]],
) -> dict[int, list[int]]:
    """The units a balanced launch runs in parts: each with its CTAs, from
    the one that runs its last steps (the lowest) upwards."""
    parts: dict[int, list[int]] = {}
    for cta, owned in enumerate(segments):
        for segment in owned:
            if not segment.whole:
                parts.setdefault(segment.unit, []).append(cta)
    return parts


def fold_order(ctas: Sequence[int]) -> tuple[int, ...]:
    """The order a merger folds a split unit's parts in: source order, from
    the highest CTA, which ran the unit's first steps, to the lowest."""
    return tuple(reversed(ctas))


def simulate_merges(
    pages: Sequence[int],
    grid: int,
    capacity: int,
    seed: int,
    *,
    workspace_ctas: int | None = None,
    skew: int = 0,
) -> dict[int, tuple[int, ...]]:
    """Run a balanced launch's exchange of split units for one random
    schedule; return, per unit, the CTAs whose parts its stored result holds,
    in the order they were folded.

    Every CTA is a step machine over its segments in unit order, with a
    second machine for its spare warp. A part of a split unit is stored
    into its slot and arrives: one is added to the unit's counter (the
    counter of the unit's lowest CTA), and the part whose arrival is the
    last merges. A part that is not its unit's lowest is posted to the
    spare warp, which arrives for it at any later time, while the CTA goes
    on; the CTA reads the verdict when its segments are done and merges
    then if that arrival was the last. The lowest part first reads the
    counter once: if it shows every other part, it resets the counter and
    merges without arriving; otherwise it arrives itself. A merger folds
    the parts from the unit's highest CTA down. It reads them from the
    slots, its own included, which it stores first; only the merger of a
    unit in two parts that did not arrive takes its own part, the last
    term, from its registers. At most ``capacity`` CTAs are resident,
    dispatched in a random order, and each step of the schedule advances a
    random machine of a resident CTA by one action, so reads, stores and
    arrivals of different CTAs interleave in every order a device could
    produce.

    Raises on a read of an unpublished slot, a second writer of a slot or of
    a unit's result, a counter that is not zero at the end or a unit without
    a result. No machine ever waits for another CTA's.
    """
    rng = random.Random(seed)
    segments = reference_partition(
        pages, grid, "balanced", workspace_ctas=workspace_ctas, skew=skew
    )
    parts = split_units(segments)
    counters: dict[int, int] = {}
    slots: dict[tuple[int, int], int] = {}
    results: dict[int, tuple[int, ...]] = {}
    machines: list[Iterator[None]] = []

    def fold(cta: int, unit: int, own_stored: bool) -> Iterator[None]:
        ctas = parts[unit]
        folded = []
        for other in fold_order(ctas):
            yield  # one part's loads
            if other != cta or own_stored:
                key = (other, int(other == ctas[0]))
                assert slots.get(key) == unit, (
                    f"CTA {cta} read slot {key} of unit {unit} "
                    f"before its part was published"
                )
            folded.append(other)
        assert unit not in results, f"unit {unit} stored twice"
        results[unit] = tuple(folded)

    def store(cta: int, unit: int) -> None:
        # A CTA's second slot holds its part of the unit it is lowest in.
        mine = (cta, int(cta == parts[unit][0]))
        assert mine not in slots, f"slot {mine} written twice"
        slots[mine] = unit

    def arrive(unit: int) -> bool:
        lowest = parts[unit][0]
        arrived = counters.get(lowest, 0)
        counters[lowest] = arrived + 1
        if arrived == len(parts[unit]) - 1:
            counters[lowest] = 0
            return True
        return False

    def spare_warp(unit: int, verdict: list[bool]) -> Iterator[None]:
        yield  # between the group's stores and the arrival
        verdict.append(arrive(unit))

    def machine(cta: int) -> Iterator[None]:
        posted = None
        verdict: list[bool] = []
        for segment in segments[cta]:
            unit = segment.unit
            yield  # the segment's pages
            if segment.whole:
                assert unit not in results, f"unit {unit} stored twice"
                results[unit] = (cta,)
                continue
            ctas = parts[unit]
            lowest = ctas[0]
            assert ctas == list(range(lowest, lowest + len(ctas))), ctas
            if cta != lowest:
                assert posted is None, "a CTA posts at most one part"
                store(cta, unit)
                posted = unit
                machines.append(spare_warp(unit, verdict))
                continue
            merge = counters.get(lowest, 0) == len(ctas) - 1
            yield  # between the read and what follows it
            # The merger of a unit in two parts that did not arrive keeps
            # its own part in registers; every other one stores it.
            stored = not (merge and len(ctas) == 2)
            if merge:
                counters[lowest] = 0
            if stored:
                store(cta, unit)
            if not merge:
                yield  # between the stores and the arrival
                merge = arrive(unit)
            if merge:
                yield from fold(cta, unit, stored)
        if posted is not None:
            while not verdict:
                yield  # the spare warp of this CTA has not arrived yet
            if verdict[0]:
                yield from fold(cta, posted, True)

    waiting = list(range(grid))
    rng.shuffle(waiting)
    dispatched: list[Iterator[None]] = []
    resident = 0
    while waiting or machines:
        while waiting and resident < capacity:
            runner = machine(waiting.pop())
            machines.append(runner)
            dispatched.append(runner)
            resident += 1
        runner = rng.choice(machines)
        try:
            next(runner)
        except StopIteration:
            machines.remove(runner)
            # A spare warp's machine ends before its CTA's.
            resident -= any(runner is other for other in dispatched)
    assert not any(counters.values()), counters
    assert len(results) == len(pages), sorted(set(range(len(pages))) - set(results))
    for unit, ctas in parts.items():
        assert results[unit] == fold_order(ctas), (unit, results[unit], ctas)
    return results


@cute.jit
def _observer_bounds(
    unit: cutlass.Int32, pages: cute.Tensor
) -> tuple[cutlass.Int32, cutlass.Int64, cutlass.Int64, cutlass.Int32]:
    """A unit that streams the pages ``[0, pages[unit])``."""
    return (
        cutlass.Int32(unit * 2),
        cutlass.Int64(0),
        cutlass.Int64(pages[unit]),
        cutlass.Int32(0),
    )


@cute.kernel
def plan_observer_kernel(
    pages: cute.Tensor,
    observed: cute.Tensor,
    units: cutlass.Constexpr,
    page_limit: cutlass.Constexpr,
    slot_bits: cutlass.Constexpr,
    partition: cutlass.Constexpr,
    skew: cutlass.Constexpr,
    workspace_ctas: cutlass.Constexpr,
) -> None:
    """Scan, rendezvous, plan: every warp reports what it found."""
    thread = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = thread % 32
    warps = plan_geometry(units)[1]
    prefix = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, 32 * warps, alignment=4),
        cute.make_layout((32 * warps,)),
    )
    totals = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Int32, 32, alignment=4),
        cute.make_layout((32,)),
    )
    partition_scan(
        prefix, totals, thread, lane, units, page_limit, _observer_bounds, (pages,)
    )
    cute.arch.sync_threads()
    plan = partition_plan(
        prefix,
        totals,
        lane,
        units,
        page_limit,
        slot_bits,
        _observer_bounds,
        (pages,),
        partition,
        skew,
        workspace_ctas,
    )
    if lane == 0:
        cta = cutlass.Int32(cute.arch.block_idx()[0])
        for index in cutlass.range_constexpr(PLAN_VALUES):
            observed[cta, thread // 32, index] = cutlass.Int32(plan[index])


@cute.jit
def launch_plan_observer(
    pages: cute.Tensor,
    observed: cute.Tensor,
    grid: cutlass.Int32,
    stream: object,
    units: cutlass.Constexpr,
    page_limit: cutlass.Constexpr,
    slot_bits: cutlass.Constexpr,
    partition: cutlass.Constexpr,
    skew: cutlass.Constexpr,
    workspace_ctas: cutlass.Constexpr,
) -> None:
    plan_observer_kernel(
        pages, observed, units, page_limit, slot_bits, partition, skew, workspace_ctas
    ).launch(grid=(grid, 1, 1), block=(PLAN_WARPS * 32, 1, 1), stream=stream)


def observe_plans(
    units: int,
    page_limit: int,
    slot_bits: int,
    device: torch.device | str,
    *,
    largest_grid: int = 1024,
    partition: str = "whole",
    skew: int = 0,
    workspace_ctas: int = 0,
) -> Callable[[Sequence[int], int], list[list[tuple[int, ...]]]]:
    """Compile the observer for ``units`` units of up to ``page_limit`` pages.

    The returned function takes each unit's pages and a launch grid and
    returns, per CTA, what each of its twenty warps got from
    ``partition_plan``: the first three values (the units and the pages)
    for ``whole``, all nine for ``balanced``. A count below zero or above
    ``page_limit`` is what the device clamps (``unit_page_count``): it plans
    as zero or as ``page_limit`` pages.
    """
    cuda = importlib.import_module("cuda.bindings.driver")
    pages = torch.zeros(units, dtype=torch.int32, device=device)
    observed = torch.empty(
        (largest_grid, PLAN_WARPS, PLAN_VALUES), dtype=torch.int32, device=device
    )
    values = 3 if partition == "whole" else PLAN_VALUES
    args = [from_dlpack(tensor, assumed_align=4) for tensor in (pages, observed)]
    compiled = cute.compile(
        launch_plan_observer,
        *args,
        cutlass.Int32(1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        units,
        page_limit,
        slot_bits,
        partition,
        skew,
        workspace_ctas,
    )

    def observe(counts: Sequence[int], grid: int) -> list[list[tuple[int, ...]]]:
        assert len(counts) == units
        assert 0 < grid <= largest_grid
        pages.copy_(pages.new_tensor(list(counts)))
        observed.fill_(-1)
        compiled(
            *args,
            cutlass.Int32(grid),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        torch.cuda.synchronize()
        return [
            [tuple(plan[:values]) for plan in warps]
            for warps in observed[:grid].cpu().tolist()
        ]

    return observe
