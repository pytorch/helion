from __future__ import annotations

import dataclasses
from functools import cache
from functools import cached_property
import heapq
import itertools
import operator
from typing import cast

import sympy

from .. import exc
from .tile_dependency import CoordinateDomain
from .tile_dependency import CoordinateRelation
from .tile_dependency import DependencyObligation
from .tile_dependency import TileDependencyGraph
from .tile_dependency import consumer_to_preceding_site_relation
from .tile_dependency import coordinate_axis_symbol
from .tile_dependency import instantiate_symbolic_dependencies
from .tile_dependency import nested_logical_axes

WorkerInterval = tuple[int, int]


def _normalize_intervals(
    intervals: tuple[WorkerInterval, ...] | list[WorkerInterval],
) -> tuple[WorkerInterval, ...]:
    """Return a sorted, disjoint union of half-open integer intervals."""
    result: list[WorkerInterval] = []
    for begin, end in sorted(intervals):
        if begin >= end:
            continue
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((begin, end))
    return tuple(result)


def _subtract_intervals(
    minuend: tuple[WorkerInterval, ...],
    subtrahend: tuple[WorkerInterval, ...],
) -> tuple[WorkerInterval, ...]:
    """Return ``minuend - subtrahend`` using only interval endpoints."""
    result: list[WorkerInterval] = []
    subtrahend = _normalize_intervals(subtrahend)
    for begin, end in _normalize_intervals(minuend):
        cursor = begin
        for remove_begin, remove_end in subtrahend:
            if remove_end <= cursor:
                continue
            if remove_begin >= end:
                break
            if cursor < remove_begin:
                result.append((cursor, min(remove_begin, end)))
            cursor = max(cursor, remove_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return tuple(result)


def _interval_cardinality(intervals: tuple[WorkerInterval, ...]) -> int:
    """Return the exact number of integers in disjoint half-open intervals."""
    return sum(end - begin for begin, end in _normalize_intervals(intervals))


@dataclasses.dataclass(frozen=True)
class WorkerScheduleSegment:
    """One symbolic task-family run in a static persistent-worker schedule.

    ``task_order`` maps dense task-order indices to logical tasks.
    ``dispatch_offset`` places those indices in a linearized range over
    ``worker_count`` workers::

        dispatch_index = dispatch_offset + task_order_index
        worker = worker_begin + dispatch_index % worker_count
        worker_step = dispatch_index // worker_count

    Several segments can describe arbitrary numbers of waves without
    materializing one schedule entry per runtime task.
    """

    root: int
    task_order: CoordinateRelation
    worker_begin: int
    worker_count: int
    dispatch_offset: int

    def __post_init__(self) -> None:
        if self.root < 0:
            raise ValueError(f"root must be nonnegative, got {self.root}")
        if self.worker_begin < 0:
            raise ValueError(
                f"worker_begin must be nonnegative, got {self.worker_begin}"
            )
        if self.worker_count <= 0:
            raise ValueError(f"worker_count must be positive, got {self.worker_count}")
        if self.dispatch_offset < 0:
            raise ValueError(
                f"dispatch_offset must be nonnegative, got {self.dispatch_offset}"
            )
        if (
            self.task_order.source_domain.kind != "task_order"
            or self.task_order.target_domain.kind != "site"
            or not self.task_order.pieces
        ):
            raise ValueError(
                "symbolic worker schedule relation has incompatible domains"
            )

    @property
    def task_count(self) -> int:
        """Number of dense task-order indices represented by this segment."""
        return self.task_order.source_domain.size

    def dispatch_index(self, task_order_index: int) -> int:
        """Return the linearized dispatch index for one ordered task."""
        if not 0 <= task_order_index < self.task_count:
            raise IndexError(task_order_index)
        return self.dispatch_offset + task_order_index

    def occupies(self, worker: int, worker_step: int) -> bool:
        """Return whether this segment occupies one step on a worker."""
        worker_offset = worker - self.worker_begin
        if not 0 <= worker_offset < self.worker_count or worker_step < 0:
            return False
        dispatch_index = worker_step * self.worker_count + worker_offset
        task_order_index = dispatch_index - self.dispatch_offset
        return 0 <= task_order_index < self.task_count

    def worker_step_bounds(self, worker: int) -> tuple[int, int] | None:
        """Return this segment's first and last occupied step on one worker."""
        worker_offset = worker - self.worker_begin
        if not 0 <= worker_offset < self.worker_count:
            return None
        begin = self.dispatch_offset
        end = begin + self.task_count
        first = begin + (worker_offset - begin) % self.worker_count
        if first >= end:
            return None
        last = first + (end - 1 - first) // self.worker_count * self.worker_count
        return first // self.worker_count, last // self.worker_count

    def workers(self) -> frozenset[int]:
        """Materialize active workers for diagnostics and small-domain tests."""
        return frozenset(
            worker
            for begin, end in self.worker_intervals()
            for worker in range(begin, end)
        )

    def worker_step_runs(self) -> tuple[tuple[int, int, int, int], ...]:
        """Return symbolic ``(worker begin, end, first step, last step)`` runs.

        A dense dispatch has only constant-many changes: at the first-dispatch
        wrap and the final-dispatch wrap.  Partitioning at those endpoints
        proves the bounds for arbitrary task and worker counts without
        iterating either domain.  Production optimized schedules impose the
        narrower wave-aligned form when constructing their rank certificate.
        """
        worker_count = self.worker_count
        first_residue = self.dispatch_offset % worker_count
        active_count = min(self.task_count, worker_count)
        first_end = first_residue + active_count
        support = (
            ((first_residue, first_end),)
            if first_end <= worker_count
            else ((0, first_end - worker_count), (first_residue, worker_count))
        )
        final_residue = (self.dispatch_offset + self.task_count - 1) % worker_count
        structural_cuts = {0, worker_count, first_residue, final_residue + 1}
        structural_cuts.update(
            endpoint for interval in support for endpoint in interval
        )
        cuts = sorted(structural_cuts)
        runs: list[tuple[int, int, int, int]] = []
        for begin, end in itertools.pairwise(cuts):
            if begin >= end or not any(
                support_begin <= begin and end <= support_end
                for support_begin, support_end in support
            ):
                continue
            begin_bounds = self.worker_step_bounds(self.worker_begin + begin)
            end_bounds = self.worker_step_bounds(self.worker_begin + end - 1)
            if begin_bounds is None or begin_bounds != end_bounds:
                raise AssertionError("worker-step structural partition is incomplete")
            runs.append(
                (
                    self.worker_begin + begin,
                    self.worker_begin + end,
                    *begin_bounds,
                )
            )
        return tuple(runs)

    def worker_intervals(self) -> tuple[WorkerInterval, ...]:
        """Return the active-worker support without worker enumeration."""
        return _normalize_intervals(
            [(begin, end) for begin, end, _first, _last in self.worker_step_runs()]
        )


def _flat_domain_index_expression(domain: CoordinateDomain) -> sympy.Expr:
    """Return the canonical flattened index of a logical coordinate."""
    result: sympy.Expr = sympy.Integer(0)
    multiplier = 1
    for axis in domain.axis_order:
        result += coordinate_axis_symbol(axis) * multiplier  # pyrefly: ignore[unsupported-operation]
        multiplier *= domain.axis_counts[axis]
    return sympy.simplify(result)


def _task_order_slice(
    task_order: CoordinateRelation,
    begin: int,
    count: int,
) -> CoordinateRelation | None:
    """Return one dense slice of a symbolic task traversal."""
    if begin < 0 or count <= 0 or begin + count > task_order.source_domain.size:
        return None
    axes = (
        *task_order.source_domain.axis_order,
        *task_order.target_domain.axis_order,
    )
    slice_axis = max(axes, default=0) + 1
    slice_domain = CoordinateDomain(
        axis_order=(slice_axis,),
        axis_counts_items=((slice_axis, count),),
        kind="task_order",
    )
    flat_index = coordinate_axis_symbol(slice_axis) + begin  # pyrefly: ignore[unsupported-operation]
    coordinates: list[sympy.Expr] = []
    stride = 1
    source_cuts = {0, count}
    for axis in task_order.source_domain.axis_order:
        axis_count = task_order.source_domain.axis_counts[axis]
        coordinates.append(
            cast(
                "sympy.Expr",
                sympy.Mod(
                    sympy.floor(flat_index / stride),  # pyrefly: ignore[unsupported-operation]
                    axis_count,
                ),
            )
        )
        if axis_count > 1:
            period = stride * axis_count
            boundary = (begin // period + 1) * period
            while boundary < begin + count:
                source_cuts.add(boundary - begin)
                boundary += period
        stride *= axis_count
    slice_to_source = CoordinateRelation.point_map(
        slice_domain,
        task_order.source_domain,
        tuple(
            (
                ((slice_axis, piece_begin, piece_end, 1),),
                tuple(coordinates),
            )
            for piece_begin, piece_end in itertools.pairwise(sorted(source_cuts))
        ),
    )
    return slice_to_source.then(task_order)


class _WorkerScheduleChronologyError(ValueError):
    """Raised when segment tuple order disagrees with a resident strand."""


@dataclasses.dataclass(frozen=True)
class WorkerSchedule:
    """Compressed, globally ordered execution stream for persistent workers.

    Tuple order is executable program order.  Worker ``w`` executes exactly
    the subsequence of segments that assign it at least one task.  The modeled
    worker steps must therefore increase along every such worker strand.
    """

    worker_count: int
    segments: tuple[WorkerScheduleSegment, ...]

    def __post_init__(self) -> None:
        if self.worker_count <= 0:
            raise ValueError(f"worker_count must be positive, got {self.worker_count}")
        prior_runs: list[tuple[int, int, int]] = []
        for segment in self.segments:
            if segment.worker_begin + segment.worker_count > self.worker_count:
                raise ValueError(
                    "worker schedule segment exceeds the resident worker domain"
                )
            for begin, end, first_step, last_step in segment.worker_step_runs():
                for prior_begin, prior_end, prior_last_step in prior_runs:
                    if (
                        max(begin, prior_begin) < min(end, prior_end)
                        and first_step <= prior_last_step
                    ):
                        overlap_begin = max(begin, prior_begin)
                        raise _WorkerScheduleChronologyError(
                            "worker schedule tuple order disagrees with worker-step "
                            f"order for worker {overlap_begin}: {first_step} follows "
                            f"{prior_last_step}"
                        )
                prior_runs.append((begin, end, last_step))

    def root_at(self, worker: int, worker_step: int) -> int | None:
        """Return the task family occupying one step on a worker."""
        roots = tuple(
            segment.root
            for segment in self.segments
            if segment.occupies(worker, worker_step)
        )
        if len(roots) > 1:
            raise AssertionError(
                f"worker {worker} step {worker_step} has multiple tasks"
            )
        return roots[0] if roots else None

    def segments_for_root(self, root: int) -> tuple[WorkerScheduleSegment, ...]:
        """Return the compressed static relation for one task family."""
        return tuple(segment for segment in self.segments if segment.root == root)

    def workers_for_root(self, root: int) -> frozenset[int]:
        """Materialize one root's worker support for diagnostics/tests."""
        return frozenset(
            worker
            for begin, end in self.worker_intervals_for_root(root)
            for worker in range(begin, end)
        )

    def worker_intervals_for_root(self, root: int) -> tuple[WorkerInterval, ...]:
        """Return one root's exact active-worker support symbolically."""
        return root_barrier_publication_plan(self, root).participant_intervals

    def active_worker_count_for_root(self, root: int) -> int:
        """Return one root's exact active-worker count from interval lengths."""
        return root_barrier_publication_plan(self, root).resident_arrival_count

    def _placement_axes(self) -> tuple[int, int]:
        maximum = max(
            (
                axis
                for segment in self.segments
                for domain in (
                    segment.task_order.source_domain,
                    segment.task_order.target_domain,
                )
                for axis in domain.axis_order
            ),
            default=0,
        )
        worker_axis = maximum + 1
        return worker_axis, worker_axis + 1

    @cached_property
    def placement_domain(self) -> CoordinateDomain:
        """The worker and worker-step coordinates of static execution."""
        worker_axis, worker_step_axis = self._placement_axes()
        maximum_worker_step = max(
            (
                segment.dispatch_index(segment.task_count - 1) // segment.worker_count
                for segment in self.segments
            ),
            default=0,
        )
        return CoordinateDomain(
            axis_order=(worker_axis, worker_step_axis),
            axis_counts_items=(
                (worker_axis, self.worker_count),
                (worker_step_axis, maximum_worker_step + 1),
            ),
            kind="worker",
        )

    @cached_property
    def worker_step_domain(self) -> CoordinateDomain:
        """The projected worker-step coordinate used for readiness math."""
        worker_step_axis = self.placement_domain.axis_order[1]
        return CoordinateDomain(
            axis_order=(worker_step_axis,),
            axis_counts_items=(
                (worker_step_axis, self.placement_domain.axis_counts[worker_step_axis]),
            ),
            kind="value",
        )

    def placement_relation(
        self,
        segment: WorkerScheduleSegment,
    ) -> CoordinateRelation:
        """Map one segment's task order to workers and worker steps."""
        relation = segment.task_order
        task_order_index = _flat_domain_index_expression(relation.source_domain)
        dispatch_index = segment.dispatch_offset + task_order_index  # pyrefly: ignore[unsupported-operation]
        worker = segment.worker_begin + sympy.Mod(  # pyrefly: ignore[unsupported-operation]
            dispatch_index,
            segment.worker_count,
        )
        worker_step = sympy.floor(dispatch_index / segment.worker_count)
        return CoordinateRelation.point_map(
            relation.source_domain,
            self.placement_domain,
            (  # pyrefly: ignore[bad-argument-type]
                (
                    tuple(
                        (axis, 0, relation.source_domain.axis_counts[axis], 1)
                        for axis in relation.source_domain.axis_order
                    ),
                    (worker, worker_step),
                ),
            ),
        )

    def worker_step_relation(
        self,
        segment: WorkerScheduleSegment,
    ) -> CoordinateRelation | None:
        """Project one symbolic placement relation to worker step."""
        return self.placement_relation(segment).project_target(self.worker_step_domain)

    def last_worker_steps_for_root(self, root: int) -> dict[int, int]:
        """Return each participating worker's final occupied step."""
        result: dict[int, int] = {}
        for segment in self.segments_for_root(root):
            begin = segment.dispatch_offset
            end = begin + segment.task_count
            for local_worker in range(segment.worker_count):
                first = begin + (local_worker - begin) % segment.worker_count
                if first >= end:
                    continue
                last = first + (end - 1 - first) // segment.worker_count * (
                    segment.worker_count
                )
                worker = segment.worker_begin + local_worker
                result[worker] = max(
                    result.get(worker, -1),
                    last // segment.worker_count,
                )
        return result

    def worker_step_bounds_for_root(self, root: int) -> tuple[int, int] | None:
        """Return the first and last occupied worker steps for one root."""
        segments = self.segments_for_root(root)
        if not segments:
            return None
        worker_steps = tuple(
            (
                segment.dispatch_index(0) // segment.worker_count,
                segment.dispatch_index(segment.task_count - 1) // segment.worker_count,
            )
            for segment in segments
        )
        return (
            min(begin for begin, _end in worker_steps),
            max(end for _begin, end in worker_steps),
        )

    def dense_assignment(self, root: int) -> tuple[int, int, int, int] | None:
        """Return one root's dense worker and schedule interval.

        The tuple contains ``worker_begin``, ``worker_count``,
        ``dispatch_offset``, and ``task_count``.  A root split across different
        worker ranges or separated schedule intervals is not dense.
        """
        segments = sorted(
            self.segments_for_root(root),
            key=lambda segment: segment.dispatch_offset,
        )
        if not segments:
            return None
        worker_begin = segments[0].worker_begin
        worker_count = segments[0].worker_count
        dispatch_offset = segments[0].dispatch_offset
        schedule_end = dispatch_offset
        for segment in segments:
            if (
                segment.worker_begin != worker_begin
                or segment.worker_count != worker_count
                or segment.dispatch_offset != schedule_end
            ):
                return None
            schedule_end += segment.task_count
        return (
            worker_begin,
            worker_count,
            dispatch_offset,
            schedule_end - dispatch_offset,
        )

    def contiguous_global_interval(self, root: int) -> tuple[int, int] | None:
        """Return one dense global schedule interval without task expansion."""
        assignment = self.dense_assignment(root)
        if assignment is None:
            return None
        worker_begin, worker_count, dispatch_offset, task_count = assignment
        if worker_begin or worker_count != self.worker_count:
            return None
        return dispatch_offset, dispatch_offset + task_count

    def without_roots(self, roots: frozenset[int]) -> WorkerSchedule:
        """Remove complete locally executed families without task expansion."""
        if not roots:
            return self
        return WorkerSchedule(
            worker_count=self.worker_count,
            segments=tuple(
                segment for segment in self.segments if segment.root not in roots
            ),
        )

    def replacing_root(
        self,
        root: int,
        segments: tuple[WorkerScheduleSegment, ...],
    ) -> WorkerSchedule:
        """Return a schedule with one task family's placement replaced."""
        result: list[WorkerScheduleSegment] = []
        inserted = False
        for segment in self.segments:
            if segment.root != root:
                result.append(segment)
            elif not inserted:
                result.extend(segments)
                inserted = True
        if not inserted:
            result.extend(segments)
        return WorkerSchedule(worker_count=self.worker_count, segments=tuple(result))


@dataclasses.dataclass(frozen=True)
class RootBarrierPublication:
    """Workers that publish one root after a particular segment occurrence."""

    segment_index: int
    worker_intervals: tuple[WorkerInterval, ...]


@dataclasses.dataclass(frozen=True)
class RootBarrierPublicationPlan:
    """The sole symbolic derivation of root-barrier publication ownership."""

    root: int
    participant_intervals: tuple[WorkerInterval, ...]
    publications: tuple[RootBarrierPublication, ...]

    @property
    def resident_arrival_count(self) -> int:
        """Return the number of resident publications without enumeration."""
        return sum(
            _interval_cardinality(publication.worker_intervals)
            for publication in self.publications
        )


@cache
def root_barrier_publication_plan(
    worker_schedule: WorkerSchedule,
    root: int,
) -> RootBarrierPublicationPlan:
    """Assign every participating worker to its final root occurrence once.

    The reverse scan and interval subtraction are the authoritative derivation
    used for both the barrier arrival count and the codegen emission sites.
    No worker or task is enumerated.
    """
    later_workers: tuple[WorkerInterval, ...] = ()
    reverse_publications: list[RootBarrierPublication] = []
    for segment_index in reversed(range(len(worker_schedule.segments))):
        segment = worker_schedule.segments[segment_index]
        if segment.root != root:
            continue
        segment_workers = segment.worker_intervals()
        publication_workers = _subtract_intervals(segment_workers, later_workers)
        if publication_workers:
            reverse_publications.append(
                RootBarrierPublication(segment_index, publication_workers)
            )
        later_workers = _normalize_intervals([*later_workers, *segment_workers])

    publications = tuple(reversed(reverse_publications))
    published_intervals = _normalize_intervals(
        [
            interval
            for publication in publications
            for interval in publication.worker_intervals
        ]
    )
    published_count = sum(
        _interval_cardinality(publication.worker_intervals)
        for publication in publications
    )
    if published_intervals != later_workers or published_count != _interval_cardinality(
        later_workers
    ):
        raise AssertionError("root publication ownership is not an exact partition")
    return RootBarrierPublicationPlan(root, later_workers, publications)


def build_baseline_worker_schedule(
    root_domains: tuple[CoordinateDomain, ...],
    root_task_orders: tuple[CoordinateRelation, ...],
    worker_count: int,
) -> WorkerSchedule:
    """Represent the existing source-ordered persistent task order exactly."""
    if worker_count <= 0:
        raise ValueError(f"worker_count must be positive, got {worker_count}")
    segments: list[WorkerScheduleSegment] = []
    worker_step_begin = 0
    if len(root_domains) != len(root_task_orders):
        raise ValueError("root domains and task orders must have equal length")
    for root, (domain, task_order) in enumerate(
        zip(root_domains, root_task_orders, strict=True)
    ):
        task_count = domain.size
        active_workers = min(worker_count, task_count)
        segments.append(
            WorkerScheduleSegment(
                root=root,
                task_order=task_order,
                worker_begin=0,
                worker_count=active_workers,
                dispatch_offset=worker_step_begin * active_workers,
            )
        )
        worker_step_begin += (task_count + worker_count - 1) // worker_count
    return WorkerSchedule(worker_count=worker_count, segments=tuple(segments))


def _family_placements_at_worker_step(
    worker_schedule: WorkerSchedule,
    *,
    root: int,
    task_domain: CoordinateDomain,
    task_order: CoordinateRelation,
    worker_step: int,
    unavailable_workers: frozenset[int] = frozenset(),
) -> tuple[WorkerSchedule, ...]:
    """Return dense placements for one complete family in free worker runs."""
    if task_domain.size > worker_schedule.worker_count:
        return ()
    # Preserve every prerequisite implied by earlier roots while looking for
    # idle capacity. ``replacing_root`` separately verifies that the proposed
    # placement agrees with the executable segment-tuple chronology.
    source_order_unavailable_workers = frozenset(
        worker
        for preceding_root in range(root)
        for worker, last_worker_step in worker_schedule.last_worker_steps_for_root(
            preceding_root
        ).items()
        if last_worker_step >= worker_step
    )
    free_workers = [
        worker
        for worker in range(worker_schedule.worker_count)
        if worker not in unavailable_workers
        and worker not in source_order_unavailable_workers
        and (
            (occupant_root := worker_schedule.root_at(worker, worker_step)) is None
            or occupant_root == root
        )
    ]
    result: list[WorkerSchedule] = []
    run_end = len(free_workers)
    while run_end:
        run_begin = run_end - 1
        while run_begin and free_workers[run_begin - 1] == free_workers[run_begin] - 1:
            run_begin -= 1
        if run_end - run_begin >= task_domain.size:
            worker_begin = free_workers[run_end - task_domain.size]
            candidate_segment = WorkerScheduleSegment(
                root=root,
                task_order=task_order,
                worker_begin=worker_begin,
                worker_count=task_domain.size,
                dispatch_offset=worker_step * task_domain.size,
            )
            try:
                candidate = worker_schedule.replacing_root(
                    root,
                    (candidate_segment,),
                )
            except _WorkerScheduleChronologyError:
                # This is a speculative placement.  Earlier scheduling passes
                # can already have moved an independent later root ahead in
                # worker time, while ``replacing_root`` preserves the replaced
                # root's tuple position.  In that case the candidate has no
                # executable linearization in the current tuple order.  Keep
                # searching rather than invalidating the conservative plan.
                pass
            else:
                result.append(candidate)
        run_end = run_begin
    return tuple(result)


def place_ready_families(
    readiness_graph: ReadinessGraph,
    original_schedule: WorkerSchedule,
    worker_schedule: WorkerSchedule,
    continuations: tuple[FinalArrivalContinuation, ...],
) -> tuple[WorkerSchedule, tuple[FinalArrivalContinuation, ...]]:
    """Move complete ready families into idle capacity during a producer tail.

    Final-arrival execution is useful when no separate workers are available.
    When a complete consumer family fits on workers that are free while some
    of its static ancestors still have queued work, a direct event wait avoids
    extending those producer streams.  This is derived from schedule liveness,
    independent of the roots' operations or graph topology.
    """
    result = worker_schedule
    remaining_continuations = continuations
    continuation_by_root = _continuations_by_consumer_root(
        readiness_graph, continuations
    )
    candidate_roots = sorted(
        {
            readiness_graph.event(continuation.event_id)
            .consumers[continuation.consumer_index]
            .consumer_root
            for continuation in remaining_continuations
        }
    )
    for root in candidate_roots:
        task_domain = readiness_graph.root_domains[root]
        if task_domain.size > result.worker_count:
            continue
        root_continuations = tuple(
            continuation
            for continuation in remaining_continuations
            if readiness_graph.event(continuation.event_id)
            .consumers[continuation.consumer_index]
            .consumer_root
            == root
        )
        if len(root_continuations) != 1:
            continue
        continuation = root_continuations[0]
        continuation_event = readiness_graph.event(continuation.event_id)
        continuation_consumer = continuation_event.consumers[
            continuation.consumer_index
        ]
        ready_after = _event_ready_after_worker_steps(
            readiness_graph,
            continuation_event,
            worker_schedule=result,
            continuation_by_root=continuation_by_root,
        )
        if ready_after is None:
            continue
        ready_after_worker_steps, prerequisite_roots = ready_after
        consumer_ready_after = continuation_consumer.keys_by_consumer.then(
            ready_after_worker_steps
        )
        readiness_bounds = (
            None
            if consumer_ready_after is None
            else consumer_ready_after.value_bounds()
        )
        if readiness_bounds is None:
            continue
        prerequisite_worker_steps: set[tuple[int, int]] = set()
        for prerequisite_root in prerequisite_roots:
            last_worker_steps = result.last_worker_steps_for_root(prerequisite_root)
            prerequisite_worker_steps.update(last_worker_steps.items())
        if not prerequisite_worker_steps:
            continue

        original_bounds = original_schedule.worker_step_bounds_for_root(root)
        if original_bounds is None:
            continue
        original_worker_step = original_bounds[0]
        remaining_after_placement = tuple(
            continuation
            for continuation in remaining_continuations
            if continuation not in root_continuations
        )
        for worker_step in range(readiness_bounds[0] + 1, original_worker_step):
            unfinished_workers = frozenset(
                worker
                for worker, prerequisite_worker_step in prerequisite_worker_steps
                if prerequisite_worker_step >= worker_step
            )
            if not unfinished_workers:
                break
            candidate = next(
                (
                    candidate
                    for candidate in _family_placements_at_worker_step(
                        result,
                        root=root,
                        task_domain=task_domain,
                        task_order=readiness_graph.root_task_orders[root],
                        worker_step=worker_step,
                        unavailable_workers=unfinished_workers,
                    )
                ),
                None,
            )
            if candidate is None:
                continue
            result = candidate
            remaining_continuations = remaining_after_placement
            continuation_by_root.pop(root)
            break
    return result, remaining_continuations


def build_worker_schedule(
    readiness_graph: ReadinessGraph,
    *,
    worker_count: int,
) -> tuple[
    WorkerSchedule,
    tuple[FinalArrivalContinuation, ...],
    tuple[ReadinessCounterPlan, ...],
    ReadinessGraph,
]:
    """Derive local and static task placement for one worker count."""
    baseline = build_baseline_worker_schedule(
        readiness_graph.root_domains,
        readiness_graph.root_task_orders,
        worker_count,
    )
    nested_wait_roots = frozenset(
        readiness_consumer.consumer_root
        for event in readiness_graph.events
        for readiness_consumer in event.consumers
        if readiness_consumer.consumer_site_id is not None
    )
    continuations = choose_final_arrival_continuations(
        readiness_graph,
        baseline,
        excluded_roots=nested_wait_roots,
    )
    ordered = order_continuation_producers_by_readiness_key(
        readiness_graph,
        baseline,
        continuations,
    )
    continuations = choose_final_arrival_continuations(
        readiness_graph,
        ordered,
        excluded_roots=nested_wait_roots,
    )
    continuation_roots = frozenset(
        readiness_consumer.consumer_root
        for continuation in continuations
        for readiness_consumer in (
            readiness_graph.event(continuation.event_id).consumers[
                continuation.consumer_index
            ],
        )
    )
    schedule = ordered.without_roots(continuation_roots)
    schedule, nested_loop_counters = place_nested_loop_consumers(
        readiness_graph,
        schedule,
        continuations,
    )
    nested_obligations = frozenset(
        obligation
        for counter in nested_loop_counters
        for readiness_consumer in counter.consumers
        for obligation in readiness_consumer.covered_obligations
    )
    scheduled_readiness_graph = _without_root_consumers_for_obligations(
        readiness_graph, nested_obligations
    )
    schedule, continuations = place_ready_families(
        scheduled_readiness_graph,
        ordered,
        schedule,
        continuations,
    )
    return schedule, continuations, nested_loop_counters, scheduled_readiness_graph


@dataclasses.dataclass(frozen=True)
class FinalArrivalContinuation:
    """A consumer task executed by whichever producer makes the final arrival."""

    event_id: int
    consumer_index: int


def _continuations_by_consumer_root(
    readiness_graph: ReadinessGraph,
    continuations: tuple[FinalArrivalContinuation, ...],
) -> dict[int, FinalArrivalContinuation]:
    """Index final-arrival continuations by their consumer root."""
    return {
        readiness_graph.event(continuation.event_id)
        .consumers[continuation.consumer_index]
        .consumer_root: continuation
        for continuation in continuations
    }


@dataclasses.dataclass(frozen=True)
class ReadinessProducer:
    """One producer execution site's requirements for a readiness event."""

    producer_root: int
    producers_by_key: CoordinateRelation
    producer_site_id: int | None = None

    @cached_property
    def _publication_and_arrival_count_relations(
        self,
    ) -> tuple[CoordinateRelation | None, CoordinateRelation | None]:
        """Derive publication and arrival counts from one target-set proof."""
        return self.producers_by_key.derive_converse_and_target_counts()

    @property
    def keys_by_producer(self) -> CoordinateRelation | None:
        """Return the derived publication relation, when representable."""
        return self._publication_and_arrival_count_relations[0]

    @property
    def arrival_count_by_key(self) -> CoordinateRelation | None:
        """Return the exact symbolic number of arrivals per readiness key."""
        return self._publication_and_arrival_count_relations[1]


def _uniform_arrival_count(
    producers: tuple[ReadinessProducer, ...],
) -> int | None:
    """Return one constant arrival count for an event, when it has one."""
    total = 0
    for readiness_producer in producers:
        cardinality = readiness_producer.arrival_count_by_key
        count = None if cardinality is None else cardinality.constant_value()
        if count is None:
            return None
        total += count
    return total


@dataclasses.dataclass(frozen=True)
class ReadinessConsumer:
    """A consumer execution site's symbolic requirements from one event."""

    consumer_root: int
    keys_by_consumer: CoordinateRelation
    covered_obligations: frozenset[DependencyObligation] = frozenset()
    consumer_site_id: int | None = None


def _readiness_key_domain(
    producers: tuple[ReadinessProducer, ...],
    consumers: tuple[ReadinessConsumer, ...],
) -> CoordinateDomain:
    """Validate and return the shared readiness-key domain of one event."""
    if not producers:
        raise ValueError("an event requires at least one producer")
    readiness_key_domain = producers[0].producers_by_key.source_domain
    if any(
        readiness_producer.producers_by_key.source_domain != readiness_key_domain
        for readiness_producer in producers[1:]
    ) or any(
        readiness_consumer.keys_by_consumer.target_domain != readiness_key_domain
        for readiness_consumer in consumers
    ):
        raise ValueError("event relations must share one readiness-key domain")
    return readiness_key_domain


@dataclasses.dataclass(frozen=True)
class ReadinessEvent:
    """One symbolic readiness event shared by scheduling and lowering."""

    producers: tuple[ReadinessProducer, ...]
    consumers: tuple[ReadinessConsumer, ...]

    def __post_init__(self) -> None:
        readiness_key_domain = _readiness_key_domain(self.producers, self.consumers)
        if readiness_key_domain.kind != "event":
            raise ValueError("readiness-key domain must have event kind")
        if readiness_key_domain.identity is None or readiness_key_domain.identity < 0:
            raise ValueError("readiness-key domain must have a nonnegative identity")
        if readiness_key_domain.axis_order != tuple(
            range(len(readiness_key_domain.axis_order))
        ):
            raise ValueError("readiness-key axes must use canonical local indices")
        if readiness_key_domain.block_sizes_items:
            raise ValueError("readiness-key domains must not inherit site block sizes")

    @property
    def readiness_key_domain(self) -> CoordinateDomain:
        """Return the readiness-key domain owned by every event relation."""
        return self.producers[0].producers_by_key.source_domain

    @property
    def event_id(self) -> int:
        """Return the event identity owned by its readiness-key domain."""
        identity = self.readiness_key_domain.identity
        assert identity is not None
        return identity

    @property
    def readiness_key_count(self) -> int:
        return self.readiness_key_domain.size

    @property
    def root_barrier_producer_root(self) -> int | None:
        if (
            self.readiness_key_count == 1
            and len(self.producers) == 1
            and self.producers[0].producer_site_id is None
            and self.producers[0].producers_by_key.is_total()
        ):
            return self.producers[0].producer_root
        return None


@dataclasses.dataclass(frozen=True)
class ReadinessGraph:
    """Configured symbolic readiness DAG and root task orders."""

    root_task_orders: tuple[CoordinateRelation, ...]
    events: tuple[ReadinessEvent, ...]

    def __post_init__(self) -> None:
        for task_order in self.root_task_orders:
            if (
                task_order.source_domain.size != task_order.target_domain.size
                or task_order.source_domain.kind != "task_order"
                or task_order.target_domain.kind != "site"
                or not task_order.pieces
            ):
                raise ValueError(
                    "each root task order must have compatible typed domains"
                )
        if tuple(event.event_id for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("event IDs must be dense and source ordered")

    @property
    def root_domains(self) -> tuple[CoordinateDomain, ...]:
        """Return the task domains owned by the configured root task orders."""
        return tuple(task_order.target_domain for task_order in self.root_task_orders)

    def event(self, event_id: int) -> ReadinessEvent:
        return self.events[event_id]


def _supports_readiness_counter_lowering(
    readiness_producer: ReadinessProducer,
) -> bool:
    """Keep scheduler eligibility identical to counted-event code generation."""
    publication = readiness_producer.keys_by_producer
    return (
        readiness_producer.arrival_count_by_key is not None
        and publication is not None
        and publication.canonical_single_valued() is not None
    )


def _canonical_readiness_key_domain(domain: CoordinateDomain) -> CoordinateDomain:
    """Name quotient coordinates locally rather than borrowing site axes."""
    if domain.kind != "event" or domain.identity is not None:
        raise AssertionError("event quotient domain must be unidentified")
    return CoordinateDomain(
        axis_order=tuple(range(len(domain.axis_order))),
        axis_counts_items=tuple(
            (event_axis, count)
            for event_axis, (_site_axis, count) in enumerate(domain.axis_counts_items)
        ),
        kind="event",
    )


def _record_readiness_event(
    pending: dict[
        tuple[CoordinateDomain, tuple[ReadinessProducer, ...]],
        ReadinessEvent,
    ],
    *,
    readiness_key_domain: CoordinateDomain,
    producers: tuple[ReadinessProducer, ...],
    consumers: tuple[ReadinessConsumer, ...],
    require_counter_lowering: bool = False,
) -> bool:
    """Canonicalize and group one semantic event by producer partition.

    Counter-lowering admission runs only after the final event identity is
    assigned, so later scheduling phases reuse the same relation proofs.
    """
    if _readiness_key_domain(producers, consumers) != readiness_key_domain:
        raise AssertionError("event relations do not share their quotient domain")
    canonical_domain = _canonical_readiness_key_domain(readiness_key_domain)
    canonical_producers: list[ReadinessProducer] = []
    for readiness_producer in producers:
        producers_by_key = readiness_producer.producers_by_key.rename_source_axes(
            canonical_domain
        )
        if producers_by_key is None:
            raise AssertionError("event relation does not match its quotient geometry")
        canonical_producers.append(
            dataclasses.replace(
                readiness_producer,
                producers_by_key=producers_by_key,
            )
        )
    canonical_consumers: list[ReadinessConsumer] = []
    for readiness_consumer in consumers:
        keys_by_consumer = readiness_consumer.keys_by_consumer.rename_target_axes(
            canonical_domain
        )
        if keys_by_consumer is None:
            raise AssertionError("event relation does not match its quotient geometry")
        canonical_consumers.append(
            dataclasses.replace(
                readiness_consumer,
                keys_by_consumer=keys_by_consumer,
            )
        )
    canonical_producers_tuple = tuple(canonical_producers)
    signature = canonical_domain, canonical_producers_tuple
    previous_event = pending.get(signature)
    if previous_event is None:
        event_id = len(pending)
        identified_domain = dataclasses.replace(canonical_domain, identity=event_id)
        identified_producers: list[ReadinessProducer] = []
        for readiness_producer in canonical_producers_tuple:
            producers_by_key = readiness_producer.producers_by_key.rename_source_axes(
                identified_domain
            )
            if producers_by_key is None:
                raise AssertionError(
                    "event identity assignment changed readiness-key geometry"
                )
            identified_producers.append(
                dataclasses.replace(
                    readiness_producer,
                    producers_by_key=producers_by_key,
                )
            )
        event_producers = tuple(identified_producers)
        previous_consumers: tuple[ReadinessConsumer, ...] = ()
    else:
        event_id = previous_event.event_id
        identified_domain = previous_event.readiness_key_domain
        event_producers = previous_event.producers
        previous_consumers = previous_event.consumers

    if require_counter_lowering and any(
        not _supports_readiness_counter_lowering(readiness_producer)
        for readiness_producer in event_producers
    ):
        return False

    grouped_consumers = list(previous_consumers)
    for canonical_consumer in canonical_consumers:
        keys_by_consumer = canonical_consumer.keys_by_consumer.rename_target_axes(
            identified_domain
        )
        if keys_by_consumer is None:
            raise AssertionError(
                "event identity assignment changed readiness-key geometry"
            )
        readiness_consumer = dataclasses.replace(
            canonical_consumer,
            keys_by_consumer=keys_by_consumer,
        )
        matching_index = next(
            (
                index
                for index, previous in enumerate(grouped_consumers)
                if previous.consumer_root == readiness_consumer.consumer_root
                and previous.consumer_site_id == readiness_consumer.consumer_site_id
                and previous.keys_by_consumer == readiness_consumer.keys_by_consumer
            ),
            None,
        )
        if matching_index is None:
            grouped_consumers.append(readiness_consumer)
            continue
        previous = grouped_consumers[matching_index]
        grouped_consumers[matching_index] = dataclasses.replace(
            previous,
            covered_obligations=(
                previous.covered_obligations | readiness_consumer.covered_obligations
            ),
        )
    pending[signature] = ReadinessEvent(
        producers=event_producers,
        consumers=tuple(grouped_consumers),
    )
    return True


def _without_root_consumers_for_obligations(
    readiness_graph: ReadinessGraph,
    covered_obligations: frozenset[DependencyObligation],
) -> ReadinessGraph:
    """Remove root-entry consumers covered by selected nested-loop waits."""
    if not covered_obligations:
        return readiness_graph
    events: list[ReadinessEvent] = []
    for event in readiness_graph.events:
        consumers: list[ReadinessConsumer] = []
        for readiness_consumer in event.consumers:
            if readiness_consumer.consumer_site_id is not None:
                consumers.append(readiness_consumer)
                continue
            remaining = readiness_consumer.covered_obligations - covered_obligations
            if remaining:
                consumers.append(
                    dataclasses.replace(
                        readiness_consumer,
                        covered_obligations=remaining,
                    )
                )
        events.append(dataclasses.replace(event, consumers=tuple(consumers)))
    return dataclasses.replace(readiness_graph, events=tuple(events))


@dataclasses.dataclass(frozen=True)
class ReadinessCounterPlan:
    """A readiness-key space receiving arrivals from one or more roots.

    Each producer has an independently proved readiness-key-to-producer
    relation. The
    expected count is derived from its producer sets, so the event represents
    both ordinary continuations and generic multi-predecessor joins.
    Consumers are independent of event identity. ``continuation_consumer_index``
    identifies the optional consumer executed by the final arriving producer.
    """

    producers: tuple[ReadinessProducer, ...]
    consumers: tuple[ReadinessConsumer, ...]
    continuation_consumer_index: int | None = None

    def __post_init__(self) -> None:
        _readiness_key_domain(self.producers, self.consumers)

    @property
    def readiness_key_domain(self) -> CoordinateDomain:
        """Return the readiness-key domain owned by every event relation."""
        return self.producers[0].producers_by_key.source_domain

    @property
    def continuation_consumer(self) -> ReadinessConsumer | None:
        if self.continuation_consumer_index is None:
            return None
        return self.consumers[self.continuation_consumer_index]

    @property
    def readiness_key_count(self) -> int:
        """Return the complete readiness-key count."""
        return self.readiness_key_domain.size

    def uniform_arrival_count(self) -> int | None:
        """Return constant fan-in without enumerating readiness keys."""
        return _uniform_arrival_count(self.producers)


@dataclasses.dataclass(frozen=True)
class _EmittedPrerequisite:
    """One consumer prerequisite exactly as synchronization emits it.

    A prerequisite is either one whole-root barrier edge or one consumer arm
    of an exact counter plan.  Candidate generation and symbolic acceptance
    consume this same descriptor sequence, so they cannot silently disagree
    about which emitted dependencies exist.
    """

    barrier_producer_root: int | None = None
    barrier_consumer_root: int | None = None
    counter_plan: ReadinessCounterPlan | None = None
    counter_consumer_index: int | None = None

    def __post_init__(self) -> None:
        is_barrier = (
            self.barrier_producer_root is not None
            and self.barrier_consumer_root is not None
            and self.counter_plan is None
            and self.counter_consumer_index is None
        )
        is_counter = (
            self.barrier_producer_root is None
            and self.barrier_consumer_root is None
            and self.counter_plan is not None
            and self.counter_consumer_index is not None
            and 0 <= self.counter_consumer_index < len(self.counter_plan.consumers)
        )
        if not (is_barrier ^ is_counter):
            raise ValueError("prerequisite must describe exactly one emitted mechanism")

    @property
    def consumer_root(self) -> int:
        """Return the logical root whose execution is guarded."""
        if self.barrier_consumer_root is not None:
            return self.barrier_consumer_root
        assert self.counter_plan is not None
        assert self.counter_consumer_index is not None
        return self.counter_plan.consumers[self.counter_consumer_index].consumer_root

    @property
    def counter_consumer(self) -> ReadinessConsumer | None:
        """Return the exact-counter consumer, or ``None`` for a barrier."""
        if self.counter_plan is None:
            return None
        assert self.counter_consumer_index is not None
        return self.counter_plan.consumers[self.counter_consumer_index]


@cache
def _emitted_prerequisites(
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
) -> tuple[_EmittedPrerequisite, ...]:
    """Return the canonical dependency view shared by proposal and proof."""
    result = [
        _EmittedPrerequisite(
            barrier_producer_root=producer_root,
            barrier_consumer_root=consumer_root,
        )
        for producer_root, consumer_root in sorted(root_barrier_edges)
    ]
    for plan in readiness_counters:
        for consumer_index in range(len(plan.consumers)):
            if consumer_index == plan.continuation_consumer_index:
                continue
            result.append(
                _EmittedPrerequisite(
                    counter_plan=plan,
                    counter_consumer_index=consumer_index,
                )
            )
    return tuple(result)


@dataclasses.dataclass(frozen=True)
class _NestedLoopReadiness:
    """Configured readiness of one nested loop in task-local program order."""

    event: ReadinessEvent
    readiness_consumer: ReadinessConsumer
    ready_after_worker_step: CoordinateRelation
    prerequisite_worker_steps: frozenset[tuple[int, int]]


def _merge_relations_by_root(
    relations: tuple[tuple[int, CoordinateRelation], ...],
) -> tuple[tuple[int, CoordinateRelation], ...] | None:
    merged: dict[int, CoordinateRelation] = {}
    for root, relation in relations:
        previous = merged.get(root)
        if previous is None:
            merged[root] = relation
            continue
        union = previous.union(relation)
        if union is None:
            return None
        merged[root] = union
    return tuple(sorted(merged.items()))


def _static_producer_relations(
    readiness_graph: ReadinessGraph,
    *,
    root: int,
    site_id: int | None,
    readiness_keys: CoordinateRelation,
    continuation_by_root: dict[int, FinalArrivalContinuation],
    visiting: frozenset[int] = frozenset(),
) -> tuple[tuple[int, CoordinateRelation], ...] | None:
    """Contract continuations to relations from statically scheduled roots."""
    root_domain = readiness_graph.root_domains[root]
    root_keys = (
        readiness_keys
        if site_id is None
        else readiness_keys.project_source(root_domain)
    )
    if root_keys is None:
        return None
    continuation = continuation_by_root.get(root)
    if continuation is None:
        return ((root, root_keys),)
    if root in visiting:
        return None
    continuation_event = readiness_graph.event(continuation.event_id)
    continuation_consumer = continuation_event.consumers[continuation.consumer_index]
    converse_consumer = continuation_consumer.keys_by_consumer.converse()
    key_to_target = (
        None if converse_consumer is None else converse_consumer.then(root_keys)
    )
    if key_to_target is None:
        return None
    expanded: list[tuple[int, CoordinateRelation]] = []
    for readiness_producer in continuation_event.producers:
        publication = readiness_producer.keys_by_producer
        upstream_keys = None if publication is None else publication.then(key_to_target)
        if upstream_keys is None:
            return None
        upstream = _static_producer_relations(
            readiness_graph,
            root=readiness_producer.producer_root,
            site_id=readiness_producer.producer_site_id,
            readiness_keys=upstream_keys,
            continuation_by_root=continuation_by_root,
            visiting=visiting | frozenset((root,)),
        )
        if upstream is None:
            return None
        expanded.extend(upstream)
    return _merge_relations_by_root(tuple(expanded))


def _event_static_producers(
    readiness_graph: ReadinessGraph,
    event: ReadinessEvent,
    continuation_by_root: dict[int, FinalArrivalContinuation],
) -> tuple[tuple[int, CoordinateRelation], ...] | None:
    expanded: list[tuple[int, CoordinateRelation]] = []
    for readiness_producer in event.producers:
        publication = readiness_producer.keys_by_producer
        if publication is None:
            return None
        static_relations = _static_producer_relations(
            readiness_graph,
            root=readiness_producer.producer_root,
            site_id=readiness_producer.producer_site_id,
            readiness_keys=publication,
            continuation_by_root=continuation_by_root,
        )
        if static_relations is None:
            return None
        expanded.extend(static_relations)
    return _merge_relations_by_root(tuple(expanded))


def _transitive_static_prerequisite_roots(
    readiness_graph: ReadinessGraph,
    static_relations: tuple[tuple[int, CoordinateRelation], ...],
    continuation_by_root: dict[int, FinalArrivalContinuation],
) -> frozenset[int] | None:
    """Close static producers through waits earlier in task-local program order."""
    roots = {root for root, _relation in static_relations}
    pending = list(roots)
    while pending:
        consumer_root = pending.pop()
        for event in readiness_graph.events:
            if not any(
                readiness_consumer.consumer_root == consumer_root
                for readiness_consumer in event.consumers
            ):
                continue
            upstream = _event_static_producers(
                readiness_graph,
                event,
                continuation_by_root,
            )
            if upstream is None:
                return None
            for producer_root, _relation in upstream:
                if producer_root == consumer_root:
                    return None
                if producer_root not in roots:
                    roots.add(producer_root)
                    pending.append(producer_root)
    return frozenset(roots)


def _event_ready_after_worker_steps(
    readiness_graph: ReadinessGraph,
    event: ReadinessEvent,
    *,
    worker_schedule: WorkerSchedule,
    continuation_by_root: dict[int, FinalArrivalContinuation],
) -> tuple[CoordinateRelation, frozenset[int]] | None:
    """Return when each readiness key becomes ready and its static producers."""
    static_relations = _event_static_producers(
        readiness_graph,
        event,
        continuation_by_root,
    )
    if static_relations is None:
        return None
    prerequisite_roots = _transitive_static_prerequisite_roots(
        readiness_graph,
        static_relations,
        continuation_by_root,
    )
    if prerequisite_roots is None:
        return None
    maxima: list[CoordinateRelation] = []
    for root, keys_by_task in static_relations:
        root_domain = readiness_graph.root_domains[root]
        if keys_by_task.source_domain != root_domain:
            return None
        for segment in worker_schedule.segments_for_root(root):
            task_order = segment.task_order
            if task_order.target_domain != root_domain:
                return None
            keys_by_task_order = task_order.then(keys_by_task)
            converse = (
                None if keys_by_task_order is None else keys_by_task_order.converse()
            )
            worker_steps = worker_schedule.worker_step_relation(segment)
            maximum = (
                None
                if converse is None or worker_steps is None
                else converse.max_target_value_by_source(worker_steps)
            )
            if maximum is None:
                return None
            maxima.append(maximum)
    if not maxima:
        return None
    combined = maxima[0]
    for relation in maxima[1:]:
        union = combined.union(relation)
        if union is None:
            return None
        combined = union
    identity = CoordinateRelation.identity(
        worker_schedule.worker_step_domain,
        worker_schedule.worker_step_domain,
    )
    maximum = combined.max_target_value_by_source(identity)
    return None if maximum is None else (maximum, prerequisite_roots)


def _nested_loop_readiness(
    readiness_graph: ReadinessGraph,
    event: ReadinessEvent,
    readiness_consumer: ReadinessConsumer,
    *,
    worker_schedule: WorkerSchedule,
    continuation_by_root: dict[int, FinalArrivalContinuation],
) -> _NestedLoopReadiness | None:
    """Map each nested-loop iteration to its prerequisite worker step."""
    assert readiness_consumer.consumer_site_id is not None
    domain = readiness_consumer.keys_by_consumer.source_domain
    nested_axes = nested_logical_axes(
        readiness_graph.root_domains[readiness_consumer.consumer_root], domain
    )
    if len(nested_axes) != 1:
        return None

    event_ready_after = _event_ready_after_worker_steps(
        readiness_graph,
        event,
        worker_schedule=worker_schedule,
        continuation_by_root=continuation_by_root,
    )
    if event_ready_after is None:
        return None
    ready_after_worker_steps, prerequisite_roots = event_ready_after
    iteration_ready_after_worker_step = readiness_consumer.keys_by_consumer.then(
        ready_after_worker_steps
    )
    if (
        iteration_ready_after_worker_step is None
        or not iteration_ready_after_worker_step.is_total_function()
    ):
        return None
    prerequisite_worker_steps: set[tuple[int, int]] = set()
    for root in prerequisite_roots:
        last_worker_steps = worker_schedule.last_worker_steps_for_root(root)
        prerequisite_worker_steps.update(last_worker_steps.items())
    return _NestedLoopReadiness(
        event=event,
        readiness_consumer=readiness_consumer,
        ready_after_worker_step=iteration_ready_after_worker_step,
        prerequisite_worker_steps=frozenset(prerequisite_worker_steps),
    )


def _segmented_nested_loop_counter(
    readiness_graph: ReadinessGraph,
    event: ReadinessEvent,
    readiness_consumer: ReadinessConsumer,
    boundaries: tuple[int, ...],
) -> ReadinessCounterPlan | None:
    """Coarsen one exact nested dependency into contiguous loop segments."""
    consumer_site_id = readiness_consumer.consumer_site_id
    assert consumer_site_id is not None
    domain = readiness_consumer.keys_by_consumer.source_domain
    nested_axes = nested_logical_axes(
        readiness_graph.root_domains[readiness_consumer.consumer_root], domain
    )
    if len(nested_axes) != 1:
        return None
    (nested_axis,) = nested_axes
    segments = tuple(itertools.pairwise(boundaries))
    if not segments or any(begin >= end for begin, end in segments):
        return None
    used_axes = readiness_consumer.keys_by_consumer.source_axes_affecting_targets()
    if used_axes is None or nested_axis not in used_axes:
        return None
    reduced_domain = CoordinateDomain(
        axis_order=used_axes,
        axis_counts_items=tuple((axis, domain.axis_counts[axis]) for axis in used_axes),
        block_sizes_items=tuple(
            (axis, domain.block_sizes[axis])
            for axis in used_axes
            if axis in domain.block_sizes
        ),
        kind="site",
        identity=domain.identity,
    )
    outer_axes = tuple(axis for axis in used_axes if axis != nested_axis)
    readiness_key_domain = CoordinateDomain(
        axis_order=tuple(range(len(outer_axes) + 1)),
        axis_counts_items=(
            (0, len(segments)),
            *(
                (event_axis, reduced_domain.axis_counts[source_axis])
                for event_axis, source_axis in enumerate(outer_axes, start=1)
            ),
        ),
        kind="event",
    )
    keys_by_reduced_iteration = CoordinateRelation.point_map(
        reduced_domain,
        readiness_key_domain,
        tuple(
            (
                tuple(
                    (
                        (axis, segment_begin, segment_end, 1)
                        if axis == nested_axis
                        else (axis, 0, reduced_domain.axis_counts[axis], 1)
                    )
                    for axis in reduced_domain.axis_order
                ),
                (
                    sympy.Integer(stage),
                    *(coordinate_axis_symbol(axis) for axis in outer_axes),
                ),
            )
            for stage, (segment_begin, segment_end) in enumerate(segments)
        ),
    )
    # Remove task-local coordinates that provably do not affect the producer
    # set before taking the converse. The unreduced relation may be many-to-one
    # even though the reduced relation is exactly invertible.
    reduced_consumer = readiness_consumer.keys_by_consumer.project_source(
        reduced_domain
    )
    reduced_converse = None if reduced_consumer is None else reduced_consumer.converse()
    key_coarsening = (
        None
        if reduced_converse is None
        else reduced_converse.then(keys_by_reduced_iteration)
    )
    if key_coarsening is None:
        return None
    # This is a scheduling-derived coarsening of an already lowerable event,
    # not a second dependency fact. Derive producer publication from the
    # authoritative producer sets, compose it with the segment map, then
    # take the exact converse back into the representation owned by the plan.
    publication_relations = tuple(
        (
            None
            if readiness_producer.keys_by_producer is None
            else readiness_producer.keys_by_producer.then(key_coarsening)
        )
        for readiness_producer in event.producers
    )
    if any(relation is None for relation in publication_relations):
        return None
    producers_by_key_relations = tuple(
        None if relation is None else relation.converse()
        for relation in publication_relations
    )
    if any(relation is None for relation in producers_by_key_relations):
        return None
    keys_by_consumer = keys_by_reduced_iteration.lift_source(domain)
    if keys_by_consumer is None:
        return None

    return ReadinessCounterPlan(
        producers=tuple(
            ReadinessProducer(
                producer_root=readiness_producer.producer_root,
                producer_site_id=readiness_producer.producer_site_id,
                producers_by_key=relation,
            )
            for readiness_producer, relation in zip(
                event.producers,
                producers_by_key_relations,
                strict=True,
            )
            if relation is not None
        ),
        consumers=(
            ReadinessConsumer(
                consumer_root=readiness_consumer.consumer_root,
                covered_obligations=readiness_consumer.covered_obligations,
                consumer_site_id=readiness_consumer.consumer_site_id,
                keys_by_consumer=keys_by_consumer,
            ),
        ),
    )


def _split_nested_loop_at_readiness(
    readiness_graph: ReadinessGraph,
    nested_readiness: _NestedLoopReadiness,
    *,
    consumer_worker_step: int,
) -> ReadinessCounterPlan | None:
    """Split a nested loop at the first iteration not yet ready."""
    domain = nested_readiness.readiness_consumer.keys_by_consumer.source_domain
    consumer_site_id = nested_readiness.readiness_consumer.consumer_site_id
    assert consumer_site_id is not None
    nested_axes = nested_logical_axes(
        readiness_graph.root_domains[nested_readiness.readiness_consumer.consumer_root],
        domain,
    )
    if len(nested_axes) != 1:
        return None
    (nested_axis,) = nested_axes
    nested_iterations_per_task = domain.axis_counts[nested_axis]

    def ready(nested_iteration: int) -> bool | None:
        value_bounds = nested_readiness.ready_after_worker_step.value_bounds(
            {nested_axis: nested_iteration}
        )
        if value_bounds is None:
            return None
        # Producers at the same worker step execute concurrently on other
        # workers. They permit placement at this step, but are not ready at
        # admission: their consumer iterations belong after the split.
        # ``prerequisite_worker_steps`` separately prevents self-deadlock on a
        # producer's own worker.
        return value_bounds[1] < consumer_worker_step

    first_ready = ready(0)
    last_ready = ready(nested_iterations_per_task - 1)
    if first_ready is None or last_ready is None:
        return None
    if not first_ready:
        split_iteration = 0
    elif last_ready:
        split_iteration = nested_iterations_per_task
    else:
        lower = 0
        upper = nested_iterations_per_task - 1
        while lower + 1 < upper:
            midpoint = (lower + upper) // 2
            midpoint_ready = ready(midpoint)
            if midpoint_ready is None:
                return None
            if midpoint_ready:
                lower = midpoint
            else:
                upper = midpoint
        split_iteration = upper
    boundaries = tuple(sorted({0, split_iteration, nested_iterations_per_task}))
    return _segmented_nested_loop_counter(
        readiness_graph,
        nested_readiness.event,
        nested_readiness.readiness_consumer,
        boundaries,
    )


def _nested_loop_entry_counter(
    readiness_graph: ReadinessGraph,
    event: ReadinessEvent,
    readiness_consumer: ReadinessConsumer,
) -> ReadinessCounterPlan | None:
    """Coarsen exact iteration readiness to one wait per owning root task."""
    consumer_site_id = readiness_consumer.consumer_site_id
    assert consumer_site_id is not None
    domain = readiness_consumer.keys_by_consumer.source_domain
    nested_axes = nested_logical_axes(
        readiness_graph.root_domains[readiness_consumer.consumer_root], domain
    )
    if len(nested_axes) != 1:
        return None
    nested_iterations_per_task = domain.axis_counts[nested_axes[0]]
    return _segmented_nested_loop_counter(
        readiness_graph,
        event,
        readiness_consumer,
        (0, nested_iterations_per_task),
    )


def place_nested_loop_consumers(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
    continuations: tuple[FinalArrivalContinuation, ...],
) -> tuple[WorkerSchedule, tuple[ReadinessCounterPlan, ...]]:
    """Place root tasks with nested waits and derive their readiness counters.

    Exact nested-iteration dependencies remain the semantic source of truth.
    This pass uses only worker steps and task-local program order to select one
    split point for the original nested loop.
    It does not inspect operation kinds or recognize a graph topology.
    """
    consumers_by_root: dict[
        int,
        list[tuple[ReadinessEvent, ReadinessConsumer]],
    ] = {}
    continuation_by_root = _continuations_by_consumer_root(
        readiness_graph, continuations
    )
    for event in readiness_graph.events:
        for readiness_consumer in event.consumers:
            if readiness_consumer.consumer_site_id is not None:
                consumers_by_root.setdefault(
                    readiness_consumer.consumer_root, []
                ).append((event, readiness_consumer))

    result = worker_schedule
    plans: list[ReadinessCounterPlan] = []
    for consumer_root, event_consumers in sorted(consumers_by_root.items()):
        task_domain = readiness_graph.root_domains[consumer_root]

        # A preceding site may already carry every dependency obligation needed by
        # a later site.  The implication was proved from DeviceIR program
        # order when the readiness graph was built, so the later wait is redundant.
        uncovered_consumers: list[tuple[ReadinessEvent, ReadinessConsumer]] = []
        preceding_obligations: set[DependencyObligation] = set()
        for event, readiness_consumer in sorted(
            event_consumers,
            key=lambda item: (
                item[1].consumer_site_id
                if item[1].consumer_site_id is not None
                else -1,
                item[0].event_id,
            ),
        ):
            if (
                readiness_consumer.covered_obligations
                and readiness_consumer.covered_obligations <= (preceding_obligations)
            ):
                continue
            uncovered_consumers.append((event, readiness_consumer))
            preceding_obligations.update(readiness_consumer.covered_obligations)

        nested_loop_entry_plans = tuple(
            plan
            for event, readiness_consumer in uncovered_consumers
            if (
                plan := _nested_loop_entry_counter(
                    readiness_graph, event, readiness_consumer
                )
            )
            is not None
        )
        if task_domain.size > result.worker_count:
            plans.extend(nested_loop_entry_plans)
            continue

        nested_readiness = tuple(
            _nested_loop_readiness(
                readiness_graph,
                event,
                readiness_consumer,
                worker_schedule=result,
                continuation_by_root=continuation_by_root,
            )
            for event, readiness_consumer in uncovered_consumers
        )
        if not nested_readiness or any(item is None for item in nested_readiness):
            plans.extend(nested_loop_entry_plans)
            continue
        ordered_readiness = tuple(item for item in nested_readiness if item is not None)

        current_consumer_bounds = result.worker_step_bounds_for_root(consumer_root)
        if current_consumer_bounds is None:
            plans.extend(nested_loop_entry_plans)
            continue
        original_worker_step = current_consumer_bounds[0]
        readiness_bounds = tuple(
            item.ready_after_worker_step.value_bounds() for item in ordered_readiness
        )
        if any(bounds is None for bounds in readiness_bounds):
            plans.extend(nested_loop_entry_plans)
            continue
        earliest_ready_after_step = min(
            bounds[0] for bounds in readiness_bounds if bounds is not None
        )
        prerequisite_worker_steps = frozenset(
            placement
            for item in ordered_readiness
            for placement in item.prerequisite_worker_steps
        )
        chosen: tuple[WorkerSchedule, tuple[ReadinessCounterPlan, ...]] | None = None
        for worker_step in range(earliest_ready_after_step + 1, original_worker_step):
            busy_workers = frozenset(
                worker
                for worker, prerequisite_worker_step in prerequisite_worker_steps
                if prerequisite_worker_step >= worker_step
            )
            candidate = next(
                (
                    candidate
                    for candidate in _family_placements_at_worker_step(
                        result,
                        root=consumer_root,
                        task_domain=task_domain,
                        task_order=readiness_graph.root_task_orders[consumer_root],
                        worker_step=worker_step,
                        unavailable_workers=busy_workers,
                    )
                ),
                None,
            )
            if candidate is None:
                continue
            milestone_results = tuple(
                _split_nested_loop_at_readiness(
                    readiness_graph,
                    item,
                    consumer_worker_step=worker_step,
                )
                for item in ordered_readiness
            )
            if any(item is None for item in milestone_results):
                continue
            chosen = (
                candidate,
                tuple(plan for plan in milestone_results if plan is not None),
            )
            break
        if chosen is None:
            plans.extend(nested_loop_entry_plans)
            continue
        result, nested_loop_plans = chosen
        plans.extend(nested_loop_plans)
    return result, tuple(plans)


@dataclasses.dataclass(frozen=True)
class StaticPipelinePlan:
    """Pure graph-derived choices consumed by persistent-kernel lowering."""

    worker_schedule: WorkerSchedule
    readiness_counters: tuple[ReadinessCounterPlan, ...]
    root_barrier_edges: frozenset[tuple[int, int]]
    transient_source_root: int | None = None


def choose_final_arrival_continuations(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
    *,
    excluded_roots: frozenset[int] = frozenset(),
) -> tuple[FinalArrivalContinuation, ...]:
    """Choose final-arrival execution from complete exact task readiness.

    A one-task family has no task-level parallelism to expose, so it remains in
    the static schedule. Downstream event granularity does not change whether
    an otherwise exact-ready family is eligible for a continuation.
    """
    return tuple(
        continuation
        for continuation in derive_final_arrival_continuations(
            readiness_graph, worker_schedule
        )
        if (
            readiness_consumer := readiness_graph.event(
                continuation.event_id
            ).consumers[continuation.consumer_index]
        ).consumer_root
        not in excluded_roots
        and readiness_graph.root_domains[readiness_consumer.consumer_root].size > 1
    )


def choose_readiness_counters(
    readiness_graph: ReadinessGraph,
    continuations: tuple[FinalArrivalContinuation, ...],
    *,
    excluded_obligations: frozenset[DependencyObligation] = frozenset(),
) -> tuple[ReadinessCounterPlan, ...]:
    """Select root-entry events representable by readiness counters.

    Nested consumers keep their execution-site lowering. Excluding one
    consumer does not discard independent consumers of the same semantic
    event. A whole-root one-key event remains on the aggregated root-barrier
    path, while a one-key event covering only a strict producer subset retains
    its exact counter. Unsupported relations monotonically fall back to a root
    barrier during coverage selection.
    """
    continuation_consumers = {
        (continuation.event_id, continuation.consumer_index)
        for continuation in continuations
    }
    selected: list[ReadinessCounterPlan] = []
    for event in readiness_graph.events:
        if event.root_barrier_producer_root is not None or any(
            not _supports_readiness_counter_lowering(readiness_producer)
            for readiness_producer in event.producers
        ):
            continue
        retained_consumers: list[ReadinessConsumer] = []
        continuation_consumer_indices: list[int] = []
        for consumer_index, readiness_consumer in enumerate(event.consumers):
            if (
                readiness_consumer.consumer_site_id is not None
                or readiness_consumer.keys_by_consumer.canonical_single_valued() is None
            ):
                continue
            remaining = readiness_consumer.covered_obligations - excluded_obligations
            is_continuation = (
                event.event_id,
                consumer_index,
            ) in continuation_consumers
            if not is_continuation and not remaining:
                continue
            if is_continuation:
                continuation_consumer_indices.append(len(retained_consumers))
            retained_consumers.append(
                ReadinessConsumer(
                    consumer_root=readiness_consumer.consumer_root,
                    keys_by_consumer=readiness_consumer.keys_by_consumer,
                    covered_obligations=(
                        readiness_consumer.covered_obligations
                        if is_continuation
                        else frozenset(remaining)
                    ),
                )
            )
        if len(continuation_consumer_indices) > 1:
            raise ValueError("one readiness counter cannot have multiple continuations")
        continuation_consumer_index = (
            continuation_consumer_indices[0] if continuation_consumer_indices else None
        )
        # A one-key event may cover a strict producer subset. Whole-root
        # one-key events were rejected through ``root_barrier_producer_root``.
        if not retained_consumers:
            continue
        selected.append(
            ReadinessCounterPlan(
                producers=event.producers,
                consumers=tuple(retained_consumers),
                continuation_consumer_index=continuation_consumer_index,
            )
        )
    selected_continuation_count = sum(
        counter_plan.continuation_consumer_index is not None
        for counter_plan in selected
    )
    if selected_continuation_count != len(continuation_consumers):
        raise AssertionError(
            "not every final-arrival continuation has a readiness counter"
        )
    return tuple(selected)


def build_readiness_events(
    dependency_graph: TileDependencyGraph,
    *,
    root_domains: tuple[CoordinateDomain, ...],
    site_domains: tuple[CoordinateDomain | None, ...],
    publishable_site_ids: frozenset[int] | None = None,
) -> tuple[ReadinessEvent, ...]:
    """Build canonical symbolic readiness events from memory dependencies.

    This is the sole event-construction path. It never constructs a per-task
    producer set. Unsupported relations coarsen to one root-barrier
    event for the affected root pair.
    """
    symbolic_dependencies = instantiate_symbolic_dependencies(
        dependency_graph,
        root_domains=root_domains,
        site_domains=site_domains,
    )
    site_by_id = {site.site_id: site for site in dependency_graph.execution_sites}
    exact_dependencies = tuple(
        dependency
        for dependency in symbolic_dependencies
        if dependency.producers_by_consumer is not None
        and dependency.producers_by_consumer.source_axes_affecting_targets() is not None
    )
    all_obligations_by_pair: dict[tuple[int, int], set[DependencyObligation]] = {}
    for edge in dependency_graph.edges:
        pair = (edge.producer_root, edge.consumer_root)
        for access_dependency in edge.access_dependencies:
            all_obligations_by_pair.setdefault(pair, set()).update(
                dependency_graph.dependency_obligations(access_dependency)
            )

    implied_obligations: dict[DependencyObligation, set[DependencyObligation]] = {}
    for preceding_dependency in exact_dependencies:
        preceding_site_id = preceding_dependency.consumer_site_id
        if preceding_site_id is None or site_by_id[preceding_site_id].is_root:
            continue
        preceding_producers = preceding_dependency.producers_by_consumer
        assert preceding_producers is not None
        preceding_obligation = (
            preceding_dependency.dependency_id,
            preceding_dependency.producer_site_id,
            preceding_site_id,
        )
        for later_dependency in exact_dependencies:
            later_site_id = later_dependency.consumer_site_id
            later_producers = later_dependency.producers_by_consumer
            if (
                later_dependency is preceding_dependency
                or later_site_id is None
                or later_producers is None
                or preceding_dependency.consumer_root != later_dependency.consumer_root
                or preceding_dependency.producer_root != later_dependency.producer_root
                or preceding_dependency.producer_site_id
                != later_dependency.producer_site_id
                or preceding_producers.target_domain != later_producers.target_domain
            ):
                continue
            preceding = consumer_to_preceding_site_relation(
                dependency_graph,
                site_domains=site_domains,
                preceding_site_id=preceding_site_id,
                consumer_site_id=later_site_id,
                consumer_access_id=later_dependency.consumer_access_id,
            )
            acquired = (
                None if preceding is None else preceding.then(preceding_producers)
            )
            if acquired is not None and acquired.covers(later_producers):
                implied_obligations.setdefault(preceding_obligation, set()).add(
                    (
                        later_dependency.dependency_id,
                        later_dependency.producer_site_id,
                        later_site_id,
                    )
                )

    exact_relations: dict[
        tuple[int, int | None, CoordinateDomain],
        dict[
            tuple[int, int | None, CoordinateDomain],
            list[tuple[CoordinateRelation, DependencyObligation]],
        ],
    ] = {}

    def add_exact_relation(
        *,
        producer_root: int,
        producer_site_id: int | None,
        consumer_root: int,
        consumer_site_id: int | None,
        relation: CoordinateRelation,
        covered_obligations: frozenset[DependencyObligation],
    ) -> None:
        consumer = (consumer_root, consumer_site_id, relation.source_domain)
        producer = (producer_root, producer_site_id, relation.target_domain)
        exact_relations.setdefault(consumer, {}).setdefault(producer, []).extend(
            (relation, obligation) for obligation in covered_obligations
        )

    for dependency in exact_dependencies:
        relation = dependency.producers_by_consumer
        assert relation is not None
        obligation = (
            dependency.dependency_id,
            dependency.producer_site_id,
            dependency.consumer_site_id,
        )
        exact_obligations = frozenset(
            (obligation, *implied_obligations.get(obligation, ()))
        )
        producer_site = (
            None
            if dependency.producer_site_id is None
            else site_by_id[dependency.producer_site_id]
        )
        consumer_site = (
            None
            if dependency.consumer_site_id is None
            else site_by_id[dependency.consumer_site_id]
        )
        producer_is_root = producer_site is None or producer_site.is_root
        consumer_is_root = consumer_site is None or consumer_site.is_root
        producer_site_is_usable = producer_is_root or (
            producer_site is not None
            and producer_site.can_split_loop
            and (
                publishable_site_ids is None
                or dependency.producer_site_id in publishable_site_ids
            )
        )
        consumer_site_is_usable = consumer_is_root or (
            consumer_site is not None and consumer_site.can_split_loop
        )
        if producer_site_is_usable and consumer_site_is_usable:
            add_exact_relation(
                producer_root=dependency.producer_root,
                producer_site_id=(
                    None if producer_is_root else dependency.producer_site_id
                ),
                consumer_root=dependency.consumer_root,
                consumer_site_id=(
                    None if consumer_is_root else dependency.consumer_site_id
                ),
                relation=relation,
                covered_obligations=exact_obligations,
            )

        if producer_is_root and consumer_is_root:
            continue
        root_relation = relation
        if not consumer_is_root:
            projected = root_relation.project_source(
                root_domains[dependency.consumer_root]
            )
            if projected is None:
                continue
            root_relation = projected
        if not producer_is_root:
            projected = root_relation.project_target(
                root_domains[dependency.producer_root]
            )
            if projected is None:
                continue
            root_relation = projected
        add_exact_relation(
            producer_root=dependency.producer_root,
            producer_site_id=None,
            consumer_root=dependency.consumer_root,
            consumer_site_id=None,
            relation=root_relation,
            covered_obligations=exact_obligations,
        )

    pending_events: dict[
        tuple[CoordinateDomain, tuple[ReadinessProducer, ...]],
        ReadinessEvent,
    ] = {}
    represented_obligations: set[DependencyObligation] = set()

    def record_readiness_event(
        *,
        readiness_key_domain: CoordinateDomain,
        producers: tuple[ReadinessProducer, ...],
        consumers: tuple[ReadinessConsumer, ...],
        require_counter_lowering: bool = False,
    ) -> bool:
        if not _record_readiness_event(
            pending_events,
            readiness_key_domain=readiness_key_domain,
            producers=producers,
            consumers=consumers,
            require_counter_lowering=require_counter_lowering,
        ):
            return False
        for readiness_consumer in consumers:
            represented_obligations.update(readiness_consumer.covered_obligations)
        return True

    def add_producer_key_events(
        *,
        consumer_root: int,
        consumer_site_id: int | None,
        relations: list[
            tuple[
                tuple[int, int | None, CoordinateDomain],
                CoordinateRelation,
                frozenset[DependencyObligation],
            ]
        ],
    ) -> None:
        """Keep finer readiness keys when a consumer quotient needs fanout."""
        for producer, relation, obligations in relations:
            producer_root, producer_site_id, producer_domain = producer
            readiness_key_domain = dataclasses.replace(
                producer_domain,
                kind="event",
                identity=None,
            )
            keys_by_consumer = relation.rename_target_axes(readiness_key_domain)
            if keys_by_consumer is None:
                raise AssertionError("producer-keyed readiness geometry must match")
            record_readiness_event(
                readiness_key_domain=readiness_key_domain,
                producers=(
                    ReadinessProducer(
                        producer_root=producer_root,
                        producer_site_id=producer_site_id,
                        producers_by_key=CoordinateRelation.identity(
                            readiness_key_domain,
                            producer_domain,
                        ),
                    ),
                ),
                consumers=(
                    ReadinessConsumer(
                        consumer_root=consumer_root,
                        consumer_site_id=consumer_site_id,
                        keys_by_consumer=keys_by_consumer,
                        covered_obligations=obligations,
                    ),
                ),
            )

    for consumer, producers in sorted(
        exact_relations.items(),
        key=lambda item: (
            item[0][0],
            -1 if item[0][1] is None else item[0][1],
        ),
    ):
        consumer_root, consumer_site_id, consumer_domain = consumer
        merged_relations: list[
            tuple[
                tuple[int, int | None, CoordinateDomain],
                CoordinateRelation,
                frozenset[DependencyObligation],
            ]
        ] = []
        readiness_key_axis_set: set[int] = set()
        quotient_is_supported = True
        for producer, relation_points in sorted(
            producers.items(),
            key=lambda item: (
                item[0][0],
                -1 if item[0][1] is None else item[0][1],
            ),
        ):
            relation, first_point = relation_points[0]
            obligations = {first_point}
            for next_relation, obligation in relation_points[1:]:
                union = relation.union(next_relation)
                if union is None:
                    quotient_is_supported = False
                    break
                relation = union
                obligations.add(obligation)
            if not quotient_is_supported:
                break
            used_axes = relation.source_axes_affecting_targets()
            if used_axes is None:
                quotient_is_supported = False
                break
            readiness_key_axis_set.update(used_axes)
            merged_relations.append((producer, relation, frozenset(obligations)))

        if not quotient_is_supported:
            add_producer_key_events(
                consumer_root=consumer_root,
                consumer_site_id=consumer_site_id,
                relations=[
                    (producer, relation, frozenset((obligation,)))
                    for producer, relation_points in sorted(
                        producers.items(),
                        key=lambda item: (
                            item[0][0],
                            -1 if item[0][1] is None else item[0][1],
                        ),
                    )
                    for relation, obligation in relation_points
                ],
            )
            continue

        if any(
            left_points & right_points
            for left_index, (_left, _left_relation, left_points) in enumerate(
                merged_relations
            )
            for _right, _right_relation, right_points in merged_relations[
                left_index + 1 :
            ]
        ):
            # The same memory obligation was represented at more than one
            # producer site. These are alternative synchronization points,
            # not independent arrivals to one joined event.
            add_producer_key_events(
                consumer_root=consumer_root,
                consumer_site_id=consumer_site_id,
                relations=merged_relations,
            )
            continue

        readiness_key_axes = tuple(
            axis
            for axis in consumer_domain.axis_order
            if axis in readiness_key_axis_set
        )
        consumer_counts = consumer_domain.axis_counts
        consumer_blocks = consumer_domain.block_sizes
        readiness_key_domain = CoordinateDomain(
            axis_order=readiness_key_axes,
            axis_counts_items=tuple(
                (axis, consumer_counts[axis]) for axis in readiness_key_axes
            ),
            block_sizes_items=tuple(
                (axis, consumer_blocks[axis])
                for axis in readiness_key_axes
                if axis in consumer_blocks
            ),
            kind="event",
        )
        keys_by_consumer = CoordinateRelation.projection(
            consumer_domain, readiness_key_domain
        )
        if keys_by_consumer is None:
            add_producer_key_events(
                consumer_root=consumer_root,
                consumer_site_id=consumer_site_id,
                relations=merged_relations,
            )
            continue
        event_producers: list[ReadinessProducer] = []
        covered_obligations: set[DependencyObligation] = set()
        for producer, relation, relation_points in merged_relations:
            producer_root, producer_site_id, _producer_domain = producer
            producers_by_key = relation.factor_through(keys_by_consumer)
            if producers_by_key is None:
                break
            event_producers.append(
                ReadinessProducer(
                    producer_root=producer_root,
                    producer_site_id=producer_site_id,
                    producers_by_key=producers_by_key,
                )
            )
            covered_obligations.update(relation_points)
        else:
            if not record_readiness_event(
                readiness_key_domain=readiness_key_domain,
                producers=tuple(event_producers),
                consumers=(
                    ReadinessConsumer(
                        consumer_root=consumer_root,
                        consumer_site_id=consumer_site_id,
                        keys_by_consumer=keys_by_consumer,
                        covered_obligations=frozenset(covered_obligations),
                    ),
                ),
                require_counter_lowering=True,
            ):
                add_producer_key_events(
                    consumer_root=consumer_root,
                    consumer_site_id=consumer_site_id,
                    relations=merged_relations,
                )
            continue

        if len(event_producers) != len(merged_relations):
            add_producer_key_events(
                consumer_root=consumer_root,
                consumer_site_id=consumer_site_id,
                relations=merged_relations,
            )
            continue

    failed_consumers_by_producer: dict[int, dict[int, set[DependencyObligation]]] = {}
    for (
        producer_root,
        consumer_root,
    ), obligations in all_obligations_by_pair.items():
        remaining_obligations = obligations - represented_obligations
        if not remaining_obligations:
            continue
        failed_consumers_by_producer.setdefault(producer_root, {})[consumer_root] = (
            remaining_obligations
        )
    for producer_root, obligations_by_consumer in sorted(
        failed_consumers_by_producer.items()
    ):
        readiness_key_domain = CoordinateDomain(
            axis_order=(),
            axis_counts_items=(),
            kind="event",
        )
        producer_domain = root_domains[producer_root]
        consumers: list[ReadinessConsumer] = []
        for consumer_root, obligations in sorted(obligations_by_consumer.items()):
            consumers.append(
                ReadinessConsumer(
                    consumer_root=consumer_root,
                    consumer_site_id=None,
                    keys_by_consumer=CoordinateRelation.total(
                        root_domains[consumer_root],
                        readiness_key_domain,
                    ),
                    covered_obligations=frozenset(obligations),
                )
            )
        _record_readiness_event(
            pending_events,
            readiness_key_domain=readiness_key_domain,
            producers=(
                ReadinessProducer(
                    producer_root=producer_root,
                    producer_site_id=None,
                    producers_by_key=CoordinateRelation.total(
                        readiness_key_domain,
                        producer_domain,
                    ),
                ),
            ),
            consumers=tuple(consumers),
        )
    return tuple(pending_events.values())


def build_readiness_graph(
    dependency_graph: TileDependencyGraph,
    *,
    root_task_orders: tuple[CoordinateRelation, ...],
    site_domains: tuple[CoordinateDomain | None, ...],
    publishable_site_ids: frozenset[int] | None = None,
) -> ReadinessGraph:
    """Bind the symbolic readiness DAG for one selected configuration."""
    root_domains = tuple(task_order.target_domain for task_order in root_task_orders)
    events = build_readiness_events(
        dependency_graph,
        root_domains=root_domains,
        site_domains=site_domains,
        publishable_site_ids=publishable_site_ids,
    )
    return ReadinessGraph(
        root_task_orders=root_task_orders,
        events=events,
    )


def derive_final_arrival_continuations(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
) -> tuple[FinalArrivalContinuation, ...]:
    """Select complete one-task-per-readiness-key continuations."""
    required_obligations_by_root: dict[int, set[DependencyObligation]] = {}
    for event in readiness_graph.events:
        for readiness_consumer in event.consumers:
            if readiness_consumer.consumer_site_id is None:
                required_obligations_by_root.setdefault(
                    readiness_consumer.consumer_root, set()
                ).update(readiness_consumer.covered_obligations)

    candidates: list[
        tuple[
            int,
            int,
            int,
            ReadinessEvent,
            ReadinessConsumer,
            tuple[tuple[int, CoordinateRelation], ...],
        ]
    ] = []
    for event in readiness_graph.events:
        if (
            event.root_barrier_producer_root is not None
            or len(event.consumers) != 1
            or any(
                readiness_producer.producer_site_id is not None
                for readiness_producer in event.producers
            )
        ):
            continue
        if any(
            not _supports_readiness_counter_lowering(readiness_producer)
            for readiness_producer in event.producers
        ):
            continue
        consumer_index = 0
        readiness_consumer = event.consumers[consumer_index]
        if readiness_consumer.consumer_site_id is not None:
            continue
        fan_in = _uniform_arrival_count(event.producers)
        if fan_in is None or fan_in <= 0:
            continue
        converse_consumer = readiness_consumer.keys_by_consumer.converse()
        if (
            not readiness_consumer.covered_obligations.issuperset(
                required_obligations_by_root.get(readiness_consumer.consumer_root, ())
            )
            or not readiness_consumer.keys_by_consumer.is_total_function()
            or converse_consumer is None
            or not converse_consumer.is_total_function()
        ):
            continue
        producer_relations = _merge_relations_by_root(
            tuple(
                (readiness_producer.producer_root, publication)
                for readiness_producer in event.producers
                if (publication := readiness_producer.keys_by_producer) is not None
            )
        )
        if producer_relations is None or len(producer_relations) != len(
            {item.producer_root for item in event.producers}
        ):
            continue

        candidates.append(
            (
                readiness_consumer.consumer_root,
                event.event_id,
                consumer_index,
                event,
                readiness_consumer,
                producer_relations,
            )
        )

    conflicting_candidates: set[tuple[int, int]] = set()
    candidates_by_consumer_root: dict[int, list[tuple[int, int]]] = {}
    for consumer_root, event_id, consumer_index, *_rest in candidates:
        candidates_by_consumer_root.setdefault(consumer_root, []).append(
            (event_id, consumer_index)
        )
    for root_candidates in candidates_by_consumer_root.values():
        if len(root_candidates) > 1:
            conflicting_candidates.update(root_candidates)
    candidates_by_producer_root: dict[
        int,
        list[tuple[int, CoordinateRelation]],
    ] = {}
    for candidate_index, candidate in enumerate(candidates):
        for producer_root, relation in candidate[-1]:
            candidates_by_producer_root.setdefault(producer_root, []).append(
                (candidate_index, relation)
            )
    for root_candidates in candidates_by_producer_root.values():
        for (left_index, left), (right_index, right) in itertools.combinations(
            root_candidates, 2
        ):
            if not left.has_disjoint_source_support(right):
                conflicting_candidates.update(
                    (
                        (candidates[left_index][1], candidates[left_index][2]),
                        (candidates[right_index][1], candidates[right_index][2]),
                    )
                )

    possible_workers_by_root = {
        root: worker_schedule.workers_for_root(root)
        for root in range(len(readiness_graph.root_domains))
    }

    result: list[FinalArrivalContinuation] = []
    for (
        _consumer_root,
        event_id,
        consumer_index,
        event,
        readiness_consumer,
        _producer_relations,
    ) in sorted(candidates, key=operator.itemgetter(slice(3))):
        if (event_id, consumer_index) in conflicting_candidates:
            continue
        possible_workers = frozenset(
            worker
            for readiness_producer in event.producers
            for worker in possible_workers_by_root[readiness_producer.producer_root]
        )
        if not possible_workers:
            continue
        possible_workers_by_root[readiness_consumer.consumer_root] = possible_workers
        result.append(
            FinalArrivalContinuation(
                event_id=event_id,
                consumer_index=consumer_index,
            )
        )
    return tuple(result)


def order_continuation_producers_by_readiness_key(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
    continuations: tuple[FinalArrivalContinuation, ...],
) -> WorkerSchedule:
    """Order eligible static producers by readiness key.

    Readiness-key-major ordering completes one readiness key at a time so
    final-arrival work becomes ready as early as possible. It is legal only
    when one producer compactly enumerates a complete static task family; all
    other families keep their existing task order.
    """
    continuation_roots = {
        readiness_graph.event(continuation.event_id)
        .consumers[continuation.consumer_index]
        .consumer_root
        for continuation in continuations
    }
    replacement_by_root: dict[int, tuple[WorkerScheduleSegment, ...]] = {}

    for continuation in continuations:
        event = readiness_graph.event(continuation.event_id)
        if len(event.producers) != 1:
            continue
        readiness_producer = event.producers[0]
        if readiness_producer.producer_site_id is not None:
            continue
        root = readiness_producer.producer_root
        if root in continuation_roots or root in replacement_by_root:
            continue
        task_domain = readiness_graph.root_domains[root]
        task_order = readiness_producer.producers_by_key.enumerate_targets_by_source()
        if (
            task_order is None
            or task_order.target_domain != task_domain
            or task_order.source_domain.size != task_domain.size
        ):
            continue

        schedule_interval = worker_schedule.contiguous_global_interval(root)
        if (
            schedule_interval is None
            or schedule_interval[1] - schedule_interval[0] != task_domain.size
        ):
            continue
        replacement_by_root[root] = (
            WorkerScheduleSegment(
                root=root,
                task_order=task_order,
                worker_begin=0,
                worker_count=worker_schedule.worker_count,
                dispatch_offset=schedule_interval[0],
            ),
        )

    if not replacement_by_root:
        return worker_schedule
    segments: list[WorkerScheduleSegment] = []
    inserted_roots: set[int] = set()
    for segment in worker_schedule.segments:
        replacement = replacement_by_root.get(segment.root)
        if replacement is None:
            segments.append(segment)
        elif segment.root not in inserted_roots:
            segments.extend(replacement)
            inserted_roots.add(segment.root)
    return WorkerSchedule(worker_schedule.worker_count, tuple(segments))


_MAX_GLOBAL_LIST_TASKS = 4096
_MAX_GLOBAL_LIST_EDGES = 2_000_000
_MAX_GLOBAL_LIST_RELATION_ITEMS = 2_000_000
_MAX_GLOBAL_LIST_SEGMENTS = 4096


def _ordered_root_tasks(
    worker_schedule: WorkerSchedule,
    root: int,
) -> tuple[tuple[int, WorkerScheduleSegment, int], ...] | None:
    """Materialize one bounded root order with its originating segment index."""
    result: list[tuple[int, WorkerScheduleSegment, int]] = []
    seen: set[int] = set()
    for segment in worker_schedule.segments_for_root(root):
        for local_index, logical_tasks in enumerate(segment.task_order.materialize()):
            if len(logical_tasks) != 1:
                return None
            (logical_task,) = logical_tasks
            if logical_task in seen:
                return None
            seen.add(logical_task)
            result.append((logical_task, segment, local_index))
    domain = (
        worker_schedule.segments_for_root(root)[0].task_order.target_domain
        if result
        else None
    )
    if domain is None or seen != set(range(domain.size)):
        return None
    return tuple(result)


def _combined_producer_tasks_by_key(
    plan: ReadinessCounterPlan,
    producer_root: int,
    producer_domain: CoordinateDomain,
) -> CoordinateRelation | None:
    """Return the distinct root tasks behind each key for one producer root."""
    result: CoordinateRelation | None = None
    found = False
    for producer in plan.producers:
        if producer.producer_root != producer_root:
            continue
        found = True
        projected = producer.producers_by_key.project_target(producer_domain)
        if projected is None:
            return None
        result = projected if result is None else result.union(projected)
        if result is None:
            return None
    return result if found else None


def _producer_tasks_by_key(
    readiness_graph: ReadinessGraph,
    plan: ReadinessCounterPlan,
    materialization_budget: list[int],
    *,
    excluded_producer_roots: frozenset[int] = frozenset(),
) -> tuple[frozenset[tuple[int, int]], ...] | None:
    """Project one emitted counter's producers onto owning root CTAs."""
    if plan.readiness_key_count > materialization_budget[0]:
        return None
    materialization_budget[0] -= plan.readiness_key_count
    result: list[set[tuple[int, int]]] = [
        set() for _ in range(plan.readiness_key_count)
    ]
    producer_roots = tuple(
        dict.fromkeys(producer.producer_root for producer in plan.producers)
    )
    for producer_root in producer_roots:
        if producer_root in excluded_producer_roots:
            continue
        root_domain = readiness_graph.root_domains[producer_root]
        relation = _combined_producer_tasks_by_key(
            plan,
            producer_root,
            root_domain,
        )
        if relation is None:
            return None
        materialized = _materialize_relation_bounded(relation, materialization_budget)
        if materialized is None:
            return None
        for readiness_key, logical_tasks in enumerate(materialized):
            result[readiness_key].update(
                (producer_root, logical_task) for logical_task in logical_tasks
            )
    return tuple(frozenset(tasks) for tasks in result)


def _materialize_relation_bounded(
    relation: CoordinateRelation,
    materialization_budget: list[int],
) -> tuple[frozenset[int], ...] | None:
    """Materialize a relation without exceeding a shared item budget."""
    source_size = relation.source_domain.size
    if (
        source_size > materialization_budget[0]
        or relation.target_domain.size > _MAX_GLOBAL_LIST_RELATION_ITEMS
    ):
        return None
    materialization_budget[0] -= source_size
    result: list[frozenset[int]] = []
    for source_index in range(source_size):
        targets = relation.targets(source_index)
        if len(targets) > materialization_budget[0]:
            return None
        materialization_budget[0] -= len(targets)
        result.append(targets)
    return tuple(result)


def _consumer_keys_by_root_task(
    readiness_graph: ReadinessGraph,
    consumer: ReadinessConsumer,
    *,
    entry_only: bool,
    materialization_budget: list[int],
) -> tuple[frozenset[int], ...] | None:
    """Materialize exact readiness keys at CTA entry or over the whole CTA."""
    root_domain = readiness_graph.root_domains[consumer.consumer_root]
    relation = consumer.keys_by_consumer
    if consumer.consumer_site_id is None:
        if relation.source_domain != root_domain:
            return None
        return _materialize_relation_bounded(relation, materialization_budget)
    if not entry_only:
        projected = relation.project_source(root_domain)
        return (
            None
            if projected is None
            else _materialize_relation_bounded(projected, materialization_budget)
        )

    nested_axes = nested_logical_axes(root_domain, relation.source_domain)
    if len(nested_axes) != 1 or any(
        axis not in relation.source_domain.axis_counts
        for axis in root_domain.axis_order
    ):
        return None
    (nested_axis,) = nested_axes
    if (
        root_domain.size > materialization_budget[0]
        or relation.target_domain.size > _MAX_GLOBAL_LIST_RELATION_ITEMS
    ):
        return None
    materialization_budget[0] -= root_domain.size
    result: list[frozenset[int]] = []
    for logical_task in range(root_domain.size):
        coordinates = root_domain.coordinates(logical_task)
        coordinates[nested_axis] = 0
        targets = frozenset(
            relation.target_domain.index(
                dict(
                    zip(
                        relation.target_domain.axis_order,
                        target_coordinates,
                        strict=True,
                    )
                )
            )
            for target_coordinates in relation.target_coordinates(coordinates)
        )
        if len(targets) > materialization_budget[0]:
            return None
        materialization_budget[0] -= len(targets)
        result.append(targets)
    return tuple(result)


def _semantic_task_successors(
    readiness_graph: ReadinessGraph,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    tasks: tuple[tuple[int, int], ...],
    *,
    nested_entry_only: bool,
    external_producer_roots: frozenset[int] = frozenset(),
) -> tuple[tuple[int, ...], ...] | None:
    """Build a bounded CTA DAG from the synchronization that codegen emits."""
    index_by_task = {task: index for index, task in enumerate(tasks)}
    successors: list[set[int]] = [set() for _ in tasks]
    edge_attempt_count = 0
    materialization_budget = [_MAX_GLOBAL_LIST_RELATION_ITEMS]
    producers_by_key_by_plan: dict[
        ReadinessCounterPlan,
        tuple[frozenset[tuple[int, int]], ...],
    ] = {}

    def add_edge(producer: tuple[int, int], consumer: tuple[int, int]) -> bool:
        nonlocal edge_attempt_count
        edge_attempt_count += 1
        if edge_attempt_count > _MAX_GLOBAL_LIST_EDGES:
            return False
        producer_index = index_by_task.get(producer)
        consumer_index = index_by_task.get(consumer)
        if consumer_index is None:
            return False
        if producer_index is None:
            # A wait-free transient source is executed by an earlier ticket
            # role, outside the resident WorkerSchedule.  Its emitted
            # synchronization remains in the kernel, but it contributes no
            # resident strand edge to this bounded proposal DAG.
            return producer[0] in external_producer_roots
        if (
            producer_index == consumer_index
            or consumer_index in successors[producer_index]
        ):
            return True
        successors[producer_index].add(consumer_index)
        return True

    for prerequisite in _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    ):
        if prerequisite.barrier_producer_root is not None:
            producer_root = prerequisite.barrier_producer_root
            consumer_root = prerequisite.consumer_root
            if producer_root in external_producer_roots:
                continue
            for producer_task in range(
                readiness_graph.root_domains[producer_root].size
            ):
                for consumer_task in range(
                    readiness_graph.root_domains[consumer_root].size
                ):
                    if not add_edge(
                        (producer_root, producer_task),
                        (consumer_root, consumer_task),
                    ):
                        return None
            continue

        plan = prerequisite.counter_plan
        consumer = prerequisite.counter_consumer
        assert plan is not None and consumer is not None
        producers_by_key = producers_by_key_by_plan.get(plan)
        if producers_by_key is None:
            producers_by_key = _producer_tasks_by_key(
                readiness_graph,
                plan,
                materialization_budget,
                excluded_producer_roots=external_producer_roots,
            )
            if producers_by_key is None:
                return None
            producers_by_key_by_plan[plan] = producers_by_key
        all_producers = frozenset().union(*producers_by_key)
        required_keys = _consumer_keys_by_root_task(
            readiness_graph,
            consumer,
            entry_only=nested_entry_only,
            materialization_budget=materialization_budget,
        )
        if required_keys is None:
            if nested_entry_only:
                return None
            required_producers_by_task = (
                all_producers
                for _ in range(
                    readiness_graph.root_domains[consumer.consumer_root].size
                )
            )
        else:
            required_producers_by_task = (
                frozenset(
                    producer_task
                    for readiness_key in keys
                    for producer_task in producers_by_key[readiness_key]
                )
                for keys in required_keys
            )
        for consumer_task, required_producers in enumerate(required_producers_by_task):
            for producer_task in required_producers:
                if not add_edge(
                    producer_task,
                    (consumer.consumer_root, consumer_task),
                ):
                    return None
    return tuple(tuple(sorted(items)) for items in successors)


def _topological_order(
    successors: tuple[tuple[int, ...], ...],
) -> tuple[int, ...] | None:
    indegree = [0] * len(successors)
    for task_successors in successors:
        for successor in task_successors:
            indegree[successor] += 1
    ready = [task for task, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    result: list[int] = []
    while ready:
        task = heapq.heappop(ready)
        result.append(task)
        for successor in successors[task]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, successor)
    return tuple(result) if len(result) == len(successors) else None


def _validate_worker_schedule_tasks(
    worker_schedule: WorkerSchedule,
    root_task_orders: tuple[CoordinateRelation, ...],
    *,
    excluded_roots: frozenset[int] = frozenset(),
) -> bool:
    """Prove exact ownership for every root represented by resident segments."""
    if any(root < 0 or root >= len(root_task_orders) for root in excluded_roots):
        return False
    for root, reference_task_order in enumerate(root_task_orders):
        root_domain = reference_task_order.target_domain
        segments = worker_schedule.segments_for_root(root)
        if root in excluded_roots:
            if segments:
                return False
            continue
        if not segments:
            return False
        if sum(segment.task_count for segment in segments) != root_domain.size:
            return False
        traversal = _root_schedule_traversal(segments, reference_task_order)
        if traversal is None or traversal.logical_task_to_scheduled_ordinal is None:
            return False
    return True


@dataclasses.dataclass(frozen=True)
class _ScheduledRootTraversal:
    """One symbolic view of the executable traversal for a scheduled root.

    The segment tuple is the source of truth.  This certificate is its cached
    symbolic derivation: codegen consumes the forward relation and ordinal
    ranges, while optimized-schedule legality checks require the exact inverse
    relation.  Some existing PID permutations cannot express either direction
    as one relation; codegen still consumes these certified ranges, but they
    are ineligible for optimized schedule acceptance.  Keeping every available
    view here prevents either side from rebuilding a subtly different
    traversal.
    """

    segment_ordinal_ranges: tuple[tuple[WorkerScheduleSegment, int, int], ...]
    scheduled_ordinal_to_logical_task: CoordinateRelation | None
    logical_task_to_scheduled_ordinal: CoordinateRelation | None
    matches_reference: bool


def _logical_task_to_reference_ordinal(
    reference_task_order: CoordinateRelation,
    ordinal_domain: CoordinateDomain,
) -> CoordinateRelation | None:
    """Map a logical CTA to its canonical PID ordinal symbolically."""
    if reference_task_order.source_domain.size != ordinal_domain.size:
        return None
    root_to_reference = reference_task_order.converse()
    root_to_reference = (
        None
        if root_to_reference is None
        else root_to_reference.canonical_single_valued()
    )
    reference_to_ordinal = CoordinateRelation.point_map(
        reference_task_order.source_domain,
        ordinal_domain,
        (
            (
                tuple(
                    (
                        axis,
                        0,
                        reference_task_order.source_domain.axis_counts[axis],
                        1,
                    )
                    for axis in reference_task_order.source_domain.axis_order
                ),
                (_flat_domain_index_expression(reference_task_order.source_domain),),
            ),
        ),
    )
    result = (
        None
        if root_to_reference is None
        else root_to_reference.then(reference_to_ordinal)
    )
    return result if result is not None and result.is_total_function() else None


@cache
def _root_schedule_traversal(
    segments: tuple[WorkerScheduleSegment, ...],
    reference_task_order: CoordinateRelation,
) -> _ScheduledRootTraversal | None:
    """Derive the one symbolic traversal certificate consumed downstream.

    Codegen maps each concatenated segment ordinal through the corresponding
    ``segment.task_order`` before recovering the root's canonical PID.  The
    schedule therefore need not reproduce the reference traversal.  When the
    union of directly composed inverse relations is representable and total,
    equal cardinality proves exact-once ownership without materializing either
    domain.  Optimized candidates require that inverse.  Existing conservative
    PID permutations may retain only the segment ranges used by codegen.
    """
    root_domain = reference_task_order.target_domain
    if (
        not segments
        or sum(segment.task_count for segment in segments) != root_domain.size
    ):
        return None
    ordinal_axis = (
        max(
            (
                *root_domain.axis_order,
                *(
                    axis
                    for segment in segments
                    for axis in segment.task_order.source_domain.axis_order
                ),
            ),
            default=0,
        )
        + 1
    )
    ordinal_domain = CoordinateDomain(
        axis_order=(ordinal_axis,),
        axis_counts_items=((ordinal_axis, root_domain.size),),
        kind="task_order",
    )
    root_to_scheduled: CoordinateRelation | None = None
    scheduled_to_root: CoordinateRelation | None = None
    inverse_relation_supported = True
    forward_relation_supported = True
    segment_ordinal_ranges: list[tuple[WorkerScheduleSegment, int, int]] = []
    scheduled_ordinal = coordinate_axis_symbol(ordinal_axis)
    ordinal_begin = 0
    for segment in segments:
        if (
            segment.task_order.target_domain != root_domain
            or not segment.task_order.is_total_function()
        ):
            return None
        root_to_local = segment.task_order.converse()
        root_to_local = (
            None if root_to_local is None else root_to_local.canonical_single_valued()
        )
        local_to_scheduled = CoordinateRelation.point_map(
            segment.task_order.source_domain,
            ordinal_domain,
            (
                (
                    tuple(
                        (
                            axis,
                            0,
                            segment.task_order.source_domain.axis_counts[axis],
                            1,
                        )
                        for axis in segment.task_order.source_domain.axis_order
                    ),
                    (
                        ordinal_begin  # pyrefly: ignore[unsupported-operation]
                        + _flat_domain_index_expression(
                            segment.task_order.source_domain
                        ),
                    ),
                ),
            ),
        )
        inverse_piece = (
            None if root_to_local is None else root_to_local.then(local_to_scheduled)
        )
        local_flat_index = scheduled_ordinal - ordinal_begin  # pyrefly: ignore[unsupported-operation]
        local_coordinates: list[sympy.Expr] = []
        local_stride = 1
        for axis in segment.task_order.source_domain.axis_order:
            axis_count = segment.task_order.source_domain.axis_counts[axis]
            local_coordinates.append(
                cast(
                    "sympy.Expr",
                    sympy.Mod(
                        sympy.floor(local_flat_index / local_stride),  # pyrefly: ignore[unsupported-operation]
                        axis_count,
                    ),
                )
            )
            local_stride *= axis_count
        scheduled_to_local = CoordinateRelation.point_map(
            ordinal_domain,
            segment.task_order.source_domain,
            (
                (
                    (
                        (
                            ordinal_axis,
                            ordinal_begin,
                            ordinal_begin + segment.task_count,
                            1,
                        ),
                    ),
                    tuple(local_coordinates),
                ),
            ),
        )
        forward_piece = scheduled_to_local.then(segment.task_order)
        if inverse_piece is None or not inverse_relation_supported:
            inverse_relation_supported = False
            root_to_scheduled = None
        else:
            root_to_scheduled = (
                inverse_piece
                if root_to_scheduled is None
                else root_to_scheduled.union(inverse_piece)
            )
            if root_to_scheduled is None:
                inverse_relation_supported = False
        if forward_piece is None or not forward_relation_supported:
            forward_relation_supported = False
            scheduled_to_root = None
        else:
            scheduled_to_root = (
                forward_piece
                if scheduled_to_root is None
                else scheduled_to_root.union(forward_piece)
            )
            if scheduled_to_root is None:
                forward_relation_supported = False
        segment_ordinal_ranges.append(
            (segment, ordinal_begin, ordinal_begin + segment.task_count)
        )
        ordinal_begin += segment.task_count
    if (
        reference_task_order.source_domain.size != root_domain.size
        or not reference_task_order.is_total_function()
    ):
        return None
    if root_to_scheduled is not None and not root_to_scheduled.is_total_function():
        root_to_scheduled = None

    scheduled_to_root = (
        None
        if scheduled_to_root is None
        else scheduled_to_root.canonical_single_valued()
    )
    if scheduled_to_root is not None and not scheduled_to_root.is_total_function():
        scheduled_to_root = None
    reference_ordinal = _logical_task_to_reference_ordinal(
        reference_task_order,
        ordinal_domain,
    )

    return _ScheduledRootTraversal(
        segment_ordinal_ranges=tuple(segment_ordinal_ranges),
        scheduled_ordinal_to_logical_task=scheduled_to_root,
        logical_task_to_scheduled_ordinal=root_to_scheduled,
        matches_reference=(
            root_to_scheduled is not None
            and reference_ordinal is not None
            and root_to_scheduled.is_pointwise_equal_to(reference_ordinal)
        ),
    )


def root_schedule_matches_reference(
    segments: tuple[WorkerScheduleSegment, ...],
    reference_task_order: CoordinateRelation,
) -> bool:
    """Prove that a segmented traversal preserves canonical PID order."""
    traversal = _root_schedule_traversal(segments, reference_task_order)
    return traversal is not None and traversal.matches_reference


def _root_task_step_relation(
    segments: tuple[WorkerScheduleSegment, ...],
    reference_task_order: CoordinateRelation,
    rank_domain: CoordinateDomain,
    *,
    ticket_worker_count: int | None = None,
) -> CoordinateRelation | None:
    """Map logical CTAs to steps through the sole scheduled traversal."""
    traversal = _root_schedule_traversal(
        segments,
        reference_task_order,
    )
    if traversal is None:
        return None
    task_to_scheduled = traversal.logical_task_to_scheduled_ordinal
    if task_to_scheduled is None:
        return None
    scheduled_ordinal_domain = task_to_scheduled.target_domain
    if ticket_worker_count is not None and traversal.matches_reference:
        reference_ordinal = _logical_task_to_reference_ordinal(
            reference_task_order, scheduled_ordinal_domain
        )
        if reference_ordinal is not None:
            # Preserve the actual traversal as the authority; equality merely
            # permits its compact canonical form to avoid carrying slice
            # boundaries into subsequent frontier algebra.
            task_to_scheduled = reference_ordinal
    (scheduled_axis,) = scheduled_ordinal_domain.axis_order
    scheduled_ordinal = coordinate_axis_symbol(scheduled_axis)
    step_pieces: list[
        tuple[tuple[tuple[int, int, int, int], ...], tuple[sympy.Expr, ...]]
    ] = []
    if ticket_worker_count is not None:
        step_pieces.append(
            (
                (
                    (
                        scheduled_axis,
                        0,
                        scheduled_ordinal_domain.size,
                        1,
                    ),
                ),
                (
                    cast(
                        "sympy.Expr",
                        sympy.floor(  # pyrefly: ignore[bad-argument-type]
                            scheduled_ordinal / ticket_worker_count  # pyrefly: ignore[unsupported-operation]
                        ),
                    ),
                ),
            )
        )
    else:
        for segment, ordinal_begin, ordinal_end in traversal.segment_ordinal_ranges:
            step_expression = cast(
                "sympy.Expr",
                sympy.floor(  # pyrefly: ignore[bad-argument-type]
                    (segment.dispatch_offset + scheduled_ordinal - ordinal_begin)  # pyrefly: ignore[unsupported-operation]
                    / segment.worker_count
                ),
            )
            step_pieces.append(
                (
                    (
                        (
                            scheduled_axis,
                            ordinal_begin,
                            ordinal_end,
                            1,
                        ),
                    ),
                    (step_expression,),
                )
            )
    steps_by_scheduled_order = CoordinateRelation.point_map(
        scheduled_ordinal_domain,
        rank_domain,
        tuple(step_pieces),
    )
    task_steps = task_to_scheduled.then(steps_by_scheduled_order)
    return (
        task_steps
        if task_steps is not None and task_steps.is_total_function()
        else None
    )


def _task_step_relations(
    worker_schedule: WorkerSchedule,
    readiness_graph: ReadinessGraph,
    *,
    excluded_roots: frozenset[int] = frozenset(),
) -> tuple[CoordinateRelation | None, ...] | None:
    """Return symbolic logical-task-to-rank functions for resident roots."""
    maximum_step = max(
        (
            last_step
            for segment in worker_schedule.segments
            for _begin, _end, _first_step, last_step in segment.worker_step_runs()
        ),
        default=0,
    )
    rank_axis = (
        max(
            (
                axis
                for task_order in (
                    *readiness_graph.root_task_orders,
                    *(segment.task_order for segment in worker_schedule.segments),
                )
                for domain in (task_order.source_domain, task_order.target_domain)
                for axis in domain.axis_order
            ),
            default=0,
        )
        + 1
    )
    rank_domain = CoordinateDomain(
        axis_order=(rank_axis,),
        axis_counts_items=((rank_axis, maximum_step + 1),),
        kind="value",
    )
    result: list[CoordinateRelation | None] = []
    for root, reference_task_order in enumerate(readiness_graph.root_task_orders):
        segments = worker_schedule.segments_for_root(root)
        if root in excluded_roots:
            if segments:
                return None
            result.append(None)
            continue
        if any(segment.dispatch_offset % segment.worker_count for segment in segments):
            return None
        task_steps = _root_task_step_relation(
            segments,
            reference_task_order,
            rank_domain,
        )
        if task_steps is None:
            return None
        result.append(task_steps)
    return tuple(result)


def _keys_by_consumer_root_task(
    readiness_graph: ReadinessGraph,
    consumer: ReadinessConsumer,
) -> CoordinateRelation | None:
    """Project every nested checkpoint requirement onto its owning CTA."""
    root_domain = readiness_graph.root_domains[consumer.consumer_root]
    if consumer.keys_by_consumer.source_domain == root_domain:
        return consumer.keys_by_consumer
    return consumer.keys_by_consumer.project_source(root_domain)


def _keys_at_first_consumer_checkpoint(
    readiness_graph: ReadinessGraph,
    consumer: ReadinessConsumer,
) -> CoordinateRelation | None:
    """Map each owning CTA to the keys waited on at nested iteration zero."""
    root_domain = readiness_graph.root_domains[consumer.consumer_root]
    site_domain = consumer.keys_by_consumer.source_domain
    if site_domain == root_domain:
        return consumer.keys_by_consumer
    nested_axes = nested_logical_axes(root_domain, site_domain)
    if len(nested_axes) != 1:
        return None
    root_axes = frozenset(root_domain.axis_order)
    root_to_checkpoint = CoordinateRelation.point_map(
        root_domain,
        site_domain,
        (
            (
                tuple(
                    (axis, 0, root_domain.axis_counts[axis], 1)
                    for axis in root_domain.axis_order
                ),
                tuple(
                    coordinate_axis_symbol(axis)
                    if axis in root_axes
                    else sympy.Integer(0)
                    for axis in site_domain.axis_order
                ),
            ),
        ),
    )
    return root_to_checkpoint.then(consumer.keys_by_consumer)


def _producer_frontier_by_consumer_task(
    readiness_graph: ReadinessGraph,
    *,
    worker_schedule: WorkerSchedule,
    consumer_keys: CoordinateRelation,
    producer: ReadinessProducer,
    producer_steps: CoordinateRelation,
) -> CoordinateRelation | None:
    """Map each consumer CTA to the latest required producer rank."""
    producer_domain = readiness_graph.root_domains[producer.producer_root]
    tasks_by_key = producer.producers_by_key.project_target(producer_domain)
    if tasks_by_key is None:
        return None
    required_tasks = consumer_keys.then(tasks_by_key)
    direct_frontier = (
        None
        if required_tasks is None
        else required_tasks.max_target_value_by_source(producer_steps)
    )
    if (
        direct_frontier is not None
        and direct_frontier.canonical_single_valued() is not None
    ):
        return direct_frontier

    # Keep keys explicit when direct consumer-to-task composition is outside
    # the supported affine subset.  Each segment still obtains its rank from
    # ``producer_steps``; no second schedule or rank formula is constructed.
    keys_by_task = producer.keys_by_producer
    if keys_by_task is None:
        return None
    if keys_by_task.source_domain != producer_domain:
        keys_by_task = keys_by_task.project_source(producer_domain)
    if keys_by_task is None:
        return None
    maxima: list[CoordinateRelation] = []
    for segment in worker_schedule.segments_for_root(producer.producer_root):
        root_to_order = segment.task_order.converse()
        root_to_order = (
            None if root_to_order is None else root_to_order.canonical_single_valued()
        )
        direct_order_by_key = (
            None if root_to_order is None else tasks_by_key.then(root_to_order)
        )
        order_by_key = direct_order_by_key
        if direct_order_by_key is None:
            keys_by_order = segment.task_order.then(keys_by_task)
            order_by_key = None if keys_by_order is None else keys_by_order.converse()
        steps_by_order = segment.task_order.then(producer_steps)
        maximum = (
            None
            if order_by_key is None or steps_by_order is None
            else order_by_key.max_target_value_by_source(steps_by_order)
        )
        if maximum is None:
            return None
        maxima.append(maximum)
    if not maxima:
        return None
    key_frontier = maxima[0]
    for maximum in maxima[1:]:
        key_frontier = key_frontier.union(maximum)
        if key_frontier is None:
            return None
    identity = CoordinateRelation.identity(
        producer_steps.target_domain,
        producer_steps.target_domain,
    )
    key_frontier = key_frontier.max_target_value_by_source(identity)
    if key_frontier is None or key_frontier.canonical_single_valued() is None:
        return None
    frontier = (
        consumer_keys.then(key_frontier)
        if consumer_keys.is_single_valued()
        else consumer_keys.max_target_value_by_source(key_frontier)
    )
    # Each producer arm is proved separately.  A union of partial frontiers
    # must never masquerade as a total combined prerequisite.
    return (
        frontier
        if frontier is not None and frontier.canonical_single_valued() is not None
        else None
    )


def _all_tasks_frontier(
    consumer_domain: CoordinateDomain,
    producer_steps: CoordinateRelation,
) -> CoordinateRelation | None:
    """Return a conservative symbolic frontier for a whole-root barrier."""
    if not producer_steps.is_total_function():
        return None
    bounds = producer_steps.value_bounds()
    if bounds is None:
        return None
    rank_domain = producer_steps.target_domain
    return CoordinateRelation.point_map(
        consumer_domain,
        rank_domain,
        (
            (
                tuple(
                    (axis, 0, consumer_domain.axis_counts[axis], 1)
                    for axis in consumer_domain.axis_order
                ),
                (sympy.Integer(bounds[1]),),
            ),
        ),
    )


def _has_symbolic_worker_rank(
    worker_schedule: WorkerSchedule,
) -> bool:
    """Prove every resident strand edge strictly increases worker step."""
    prior_runs: list[tuple[int, int, int]] = []
    for segment in worker_schedule.segments:
        if segment.dispatch_offset % segment.worker_count:
            return False
        for begin, end, first_step, last_step in segment.worker_step_runs():
            if any(
                max(begin, prior_begin) < min(end, prior_end)
                and first_step <= prior_last_step
                for prior_begin, prior_end, prior_last_step in prior_runs
            ):
                return False
            prior_runs.append((begin, end, last_step))
    return True


def _schedule_is_progress_safe(
    worker_schedule: WorkerSchedule,
    readiness_graph: ReadinessGraph,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    *,
    transient_source_root: int | None = None,
) -> bool:
    """Prove progress with an affine rank, without constructing a CTA DAG.

    Resident strand edges and ordinary semantic edges strictly increase the
    scalar worker-step rank. A source-first transient root occupies an earlier
    ticket role and has no waits. Its edges into resident work are therefore
    progress-safe even when the resident CTA reaches its emitted wait before
    the source finishes; all source tickets have already been issued.
    """
    excluded_roots = (
        frozenset()
        if transient_source_root is None
        else frozenset((transient_source_root,))
    )
    if (
        transient_source_root is not None
        and worker_schedule.segments_for_root(transient_source_root)
    ) or not _has_symbolic_worker_rank(worker_schedule):
        return False
    prerequisites = _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    )
    if transient_source_root is not None and (
        any(
            prerequisite.consumer_root == transient_source_root
            for prerequisite in prerequisites
        )
    ):
        return False
    task_steps = _task_step_relations(
        worker_schedule,
        readiness_graph,
        excluded_roots=excluded_roots,
    )
    if task_steps is None:
        return False

    for prerequisite in prerequisites:
        if prerequisite.barrier_producer_root is not None:
            producer_root = prerequisite.barrier_producer_root
            consumer_root = prerequisite.consumer_root
            if producer_root == transient_source_root:
                # Every source ticket is issued before any resident ticket.
                # The barrier still gates completion and visibility at
                # runtime, but launch-stage order proves progress.
                continue
            consumer_domain = readiness_graph.root_domains[consumer_root]
            producer_steps = task_steps[producer_root]
            consumer_steps = task_steps[consumer_root]
            if producer_steps is None or consumer_steps is None:
                return False
            frontier = _all_tasks_frontier(
                consumer_domain,
                producer_steps,
            )
            if (
                frontier is None
                or not frontier.is_total_function()
                or not frontier.is_pointwise_strictly_less_than(consumer_steps)
            ):
                return False
            continue

        plan = prerequisite.counter_plan
        consumer = prerequisite.counter_consumer
        assert plan is not None and consumer is not None
        consumer_keys = _keys_by_consumer_root_task(
            readiness_graph,
            consumer,
        )
        if consumer_keys is None:
            return False
        for producer in plan.producers:
            if producer.producer_root == transient_source_root:
                # This arm is external to the resident WorkerSchedule.  In a
                # mixed join, every other resident arm is still proved below.
                continue
            producer_steps = task_steps[producer.producer_root]
            consumer_steps = task_steps[consumer.consumer_root]
            if producer_steps is None or consumer_steps is None:
                return False
            frontier = _producer_frontier_by_consumer_task(
                readiness_graph,
                worker_schedule=worker_schedule,
                consumer_keys=consumer_keys,
                producer=producer,
                producer_steps=producer_steps,
            )
            if frontier is None:
                return False
            if frontier.is_pointwise_strictly_less_than_where_defined(consumer_steps):
                continue
            return False
    return True


@dataclasses.dataclass(frozen=True)
class _PlacedRun:
    root: int
    source_segment: WorkerScheduleSegment
    source_begin: int
    task_count: int
    worker_begin: int
    worker_count: int
    worker_step: int


def _global_unit_list_schedule(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    *,
    transient_source_root: int | None = None,
) -> WorkerSchedule | None:
    """Apply deterministic unit-weight list scheduling to a bounded CTA DAG.

    This is a topology-only Graham list scheduler.  It has no latency or
    resource cost model: structural slack, bottom level, and source/task order
    are the only priorities.  Large or unrepresentable graphs retain their
    existing schedule.
    """
    scheduled_roots = tuple(
        root
        for root in range(len(readiness_graph.root_domains))
        if root != transient_source_root
    )
    total_tasks = sum(
        readiness_graph.root_domains[root].size for root in scheduled_roots
    )
    if total_tasks > _MAX_GLOBAL_LIST_TASKS:
        return None
    tasks = tuple(
        (root, logical_task)
        for root in scheduled_roots
        for logical_task in range(readiness_graph.root_domains[root].size)
    )
    if len(tasks) != total_tasks:
        raise AssertionError("task enumeration disagrees with root domains")

    root_position: dict[tuple[int, int], int] = {}
    origin_by_task: dict[tuple[int, int], tuple[WorkerScheduleSegment, int]] = {}
    for root in scheduled_roots:
        ordered = _ordered_root_tasks(worker_schedule, root)
        if ordered is None:
            return None
        for position, (logical_task, source_segment, source_index) in enumerate(
            ordered
        ):
            root_position[(root, logical_task)] = position
            origin_by_task[(root, logical_task)] = (source_segment, source_index)

    successors = _semantic_task_successors(
        readiness_graph,
        readiness_counters,
        root_barrier_edges,
        tasks,
        nested_entry_only=True,
        external_producer_roots=(
            frozenset()
            if transient_source_root is None
            else frozenset((transient_source_root,))
        ),
    )
    if successors is None:
        return None
    topological_order = _topological_order(successors)
    if topological_order is None:
        return None
    indegree = [0] * len(tasks)
    for task_successors in successors:
        for successor in task_successors:
            indegree[successor] += 1

    bottom_level = [1] * len(tasks)
    earliest_start = [0] * len(tasks)
    for task in reversed(topological_order):
        bottom_level[task] = 1 + max(
            (bottom_level[successor] for successor in successors[task]),
            default=0,
        )
    for task in topological_order:
        finish = earliest_start[task] + 1
        for successor in successors[task]:
            earliest_start[successor] = max(earliest_start[successor], finish)
    source_order_makespan = sum(
        (readiness_graph.root_domains[root].size + worker_schedule.worker_count - 1)
        // worker_schedule.worker_count
        for root in scheduled_roots
    )

    def priority(task_index: int) -> tuple[int, int, int, int, int]:
        root, logical_task = tasks[task_index]
        slack = (
            source_order_makespan
            - earliest_start[task_index]
            - bottom_level[task_index]
        )
        return (
            slack,
            -bottom_level[task_index],
            root,
            root_position[(root, logical_task)],
            task_index,
        )

    ready = [
        (*priority(task_index), task_index)
        for task_index, degree in enumerate(indegree)
        if degree == 0
    ]
    heapq.heapify(ready)
    placed_runs: list[_PlacedRun] = []
    scheduled_count = 0
    worker_step = 0
    while scheduled_count < len(tasks):
        ready_items = [heapq.heappop(ready) for _ in range(len(ready))]
        selected_items = ready_items[: worker_schedule.worker_count]
        for item in ready_items[worker_schedule.worker_count :]:
            heapq.heappush(ready, item)
        selected = [item[-1] for item in selected_items]
        if not selected:
            return None

        root_order = tuple(dict.fromkeys(tasks[index][0] for index in selected))
        worker_begin = 0
        for root in root_order:
            root_tasks = sorted(
                (index for index in selected if tasks[index][0] == root),
                key=lambda index: root_position[tasks[index]],
            )
            run_begin = 0
            while run_begin < len(root_tasks):
                first_index = root_tasks[run_begin]
                first_task = tasks[first_index]
                source_segment, source_begin = origin_by_task[first_task]
                run_end = run_begin + 1
                while run_end < len(root_tasks):
                    previous_task = tasks[root_tasks[run_end - 1]]
                    current_task = tasks[root_tasks[run_end]]
                    current_segment, current_source_index = origin_by_task[current_task]
                    if (
                        root_position[current_task] != root_position[previous_task] + 1
                        or current_segment != source_segment
                        or current_source_index != source_begin + run_end - run_begin
                    ):
                        break
                    run_end += 1
                run_count = run_end - run_begin
                placed_runs.append(
                    _PlacedRun(
                        root=root,
                        source_segment=source_segment,
                        source_begin=source_begin,
                        task_count=run_count,
                        worker_begin=worker_begin,
                        worker_count=run_count,
                        worker_step=worker_step,
                    )
                )
                worker_begin += run_count
                run_begin = run_end

        scheduled_count += len(selected)
        for task_index in selected:
            for successor in successors[task_index]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heapq.heappush(ready, (*priority(successor), successor))
        worker_step += 1

    merged_runs: list[_PlacedRun] = []
    for run in placed_runs:
        if merged_runs:
            previous = merged_runs[-1]
            if (
                previous.root == run.root
                and previous.source_segment == run.source_segment
                and previous.source_begin + previous.task_count == run.source_begin
                and previous.worker_begin == run.worker_begin
                and previous.worker_count == run.worker_count
                and previous.worker_step + previous.task_count // previous.worker_count
                == run.worker_step
            ):
                merged_runs[-1] = dataclasses.replace(
                    previous,
                    task_count=previous.task_count + run.task_count,
                )
                continue
        merged_runs.append(run)
    if len(merged_runs) > _MAX_GLOBAL_LIST_SEGMENTS:
        return None

    segments: list[WorkerScheduleSegment] = []
    for run in merged_runs:
        task_order = _task_order_slice(
            run.source_segment.task_order,
            run.source_begin,
            run.task_count,
        )
        if task_order is None:
            return None
        segments.append(
            WorkerScheduleSegment(
                root=run.root,
                task_order=task_order,
                worker_begin=run.worker_begin,
                worker_count=run.worker_count,
                dispatch_offset=run.worker_step * run.worker_count,
            )
        )
    try:
        result = WorkerSchedule(worker_schedule.worker_count, tuple(segments))
    except ValueError:
        return None
    if not _validate_worker_schedule_tasks(
        result,
        readiness_graph.root_task_orders,
        excluded_roots=(
            frozenset()
            if transient_source_root is None
            else frozenset((transient_source_root,))
        ),
    ):
        return None
    if not _schedule_is_progress_safe(
        result,
        readiness_graph,
        readiness_counters,
        root_barrier_edges,
        transient_source_root=transient_source_root,
    ):
        return None
    return result


def _transient_source_candidate(
    readiness_graph: ReadinessGraph,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    *,
    worker_count: int,
) -> int | None:
    """Find the sole oversubscribed source in the emitted prerequisite graph."""
    prerequisites = _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    )
    consumer_roots = {prerequisite.consumer_root for prerequisite in prerequisites}
    producer_roots: set[int] = set()
    for prerequisite in prerequisites:
        if prerequisite.barrier_producer_root is not None:
            producer_roots.add(prerequisite.barrier_producer_root)
            continue
        assert prerequisite.counter_plan is not None
        producer_roots.update(
            producer.producer_root for producer in prerequisite.counter_plan.producers
        )
    source_roots = tuple(sorted(producer_roots - consumer_roots))
    if len(source_roots) != 1:
        return None
    (source_root,) = source_roots
    if readiness_graph.root_domains[
        source_root
    ].size <= worker_count or not _has_strict_partial_source_signal(
        readiness_graph,
        source_root,
        readiness_counters,
        root_barrier_edges,
    ):
        return None
    return source_root


def _source_ticket_order(
    readiness_graph: ReadinessGraph,
    source_root: int,
) -> CoordinateRelation | None:
    """Return the exact logical-task-to-ticket map for a transient source."""
    task_order = readiness_graph.root_task_orders[source_root]
    ticket_axis = (
        max(
            (
                *task_order.source_domain.axis_order,
                *task_order.target_domain.axis_order,
            ),
            default=0,
        )
        + 1
    )
    ticket_domain = CoordinateDomain(
        axis_order=(ticket_axis,),
        axis_counts_items=((ticket_axis, task_order.target_domain.size),),
        kind="task_order",
    )
    logical_to_ticket = _logical_task_to_reference_ordinal(
        task_order,
        ticket_domain,
    )
    ticket_to_logical = (
        None if logical_to_ticket is None else logical_to_ticket.converse()
    )
    ticket_to_logical = (
        None
        if ticket_to_logical is None
        else ticket_to_logical.canonical_single_valued()
    )
    if (
        logical_to_ticket is None
        or not logical_to_ticket.is_total_function()
        or ticket_to_logical is None
        or not ticket_to_logical.is_total_function()
    ):
        return None
    return logical_to_ticket


def _has_strict_partial_source_signal(
    readiness_graph: ReadinessGraph,
    source_root: int,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
) -> bool:
    """Find an emitted wait with a strict-subset source contribution."""
    source_domain = readiness_graph.root_domains[source_root]
    root_order_edges = set(root_barrier_edges)
    for prerequisite in _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    ):
        plan = prerequisite.counter_plan
        consumer = prerequisite.counter_consumer
        if plan is None or consumer is None:
            continue
        if _is_ordered_by_root_barrier(
            source_root,
            consumer.consumer_root,
            root_order_edges,
        ):
            continue
        source_tasks_by_key = _combined_producer_tasks_by_key(
            plan,
            source_root,
            source_domain,
        )
        if source_tasks_by_key is None:
            continue
        source_count_by_key = source_tasks_by_key.target_count_by_source()
        source_count_by_wait = (
            None
            if source_count_by_key is None
            else consumer.keys_by_consumer.then(source_count_by_key)
        )
        bounds = (
            None
            if source_count_by_wait is None
            else source_count_by_wait.value_bounds()
        )
        if (
            source_count_by_wait is not None
            and source_count_by_wait.pieces
            and bounds is not None
            and bounds[0] > 0
            and bounds[1] < source_domain.size
        ):
            return True
    return False


def _has_valid_transient_source_schedule(
    worker_schedule: WorkerSchedule,
    readiness_graph: ReadinessGraph,
    source_root: int,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
) -> bool:
    """Prove the disjoint source-ticket and resident-schedule roles."""
    if not 0 <= source_root < len(readiness_graph.root_task_orders):
        return False
    if worker_schedule.segments_for_root(source_root):
        return False
    if _source_ticket_order(readiness_graph, source_root) is None:
        return False
    prerequisites = _emitted_prerequisites(
        readiness_counters,
        root_barrier_edges,
    )
    if any(prerequisite.consumer_root == source_root for prerequisite in prerequisites):
        return False
    return _validate_worker_schedule_tasks(
        worker_schedule,
        readiness_graph.root_task_orders,
        excluded_roots=frozenset((source_root,)),
    ) and _schedule_is_progress_safe(
        worker_schedule,
        readiness_graph,
        readiness_counters,
        root_barrier_edges,
        transient_source_root=source_root,
    )


def build_static_pipeline_plan(
    *,
    dependency_graph: TileDependencyGraph,
    root_task_orders: tuple[CoordinateRelation, ...],
    site_domains: tuple[CoordinateDomain | None, ...],
    worker_count: int,
    publishable_site_ids: frozenset[int] | None = None,
    allow_transient_source: bool = False,
) -> StaticPipelinePlan:
    """Derive all generic readiness strategies without inspecting root bodies."""
    readiness_graph = build_readiness_graph(
        dependency_graph,
        root_task_orders=root_task_orders,
        site_domains=site_domains,
        publishable_site_ids=publishable_site_ids,
    )
    try:
        (
            worker_schedule,
            continuations,
            nested_loop_counters,
            readiness_graph,
        ) = build_worker_schedule(
            readiness_graph,
            worker_count=worker_count,
        )
    except ValueError as error:
        raise exc.InvalidConfig(
            f"the num_sm_multiplier grid of {worker_count} workers does not "
            "admit a progress-safe cross-loop schedule"
        ) from error

    nested_loop_obligations = frozenset(
        obligation
        for plan in nested_loop_counters
        for readiness_consumer in plan.consumers
        for obligation in readiness_consumer.covered_obligations
    )
    readiness_counters = (
        *choose_readiness_counters(
            readiness_graph,
            continuations,
            excluded_obligations=nested_loop_obligations,
        ),
        *nested_loop_counters,
    )
    covered_obligations = frozenset(
        obligation
        for counter_plan in readiness_counters
        for readiness_consumer in counter_plan.consumers
        for obligation in readiness_consumer.covered_obligations
    )
    # Recompute coverage from the mechanisms that will actually be emitted.
    # Dependency analysis may prove a finer relation than the selected emitter
    # can materialize. Such a relation must monotonically coarsen to root
    # barrier; retaining a task-ready classification without an emitter
    # would remove the dependency entirely.
    root_barrier_edges = _select_root_barrier_edges(
        dependency_graph=dependency_graph,
        covered_obligations=covered_obligations,
    )
    root_order_edges = set(root_barrier_edges)
    retained_readiness_counters: list[ReadinessCounterPlan] = []
    for counter_plan in readiness_counters:
        retained_consumer_indices = tuple(
            consumer_index
            for consumer_index, readiness_consumer in enumerate(counter_plan.consumers)
            if consumer_index == counter_plan.continuation_consumer_index
            or not all(
                _is_ordered_by_root_barrier(
                    readiness_producer.producer_root,
                    readiness_consumer.consumer_root,
                    root_order_edges,
                )
                for readiness_producer in counter_plan.producers
            )
        )
        if not retained_consumer_indices:
            continue
        retained_readiness_counters.append(
            dataclasses.replace(
                counter_plan,
                consumers=tuple(
                    counter_plan.consumers[index] for index in retained_consumer_indices
                ),
                continuation_consumer_index=(
                    retained_consumer_indices.index(
                        counter_plan.continuation_consumer_index
                    )
                    if counter_plan.continuation_consumer_index is not None
                    else None
                ),
            )
        )
    readiness_counters = tuple(retained_readiness_counters)
    covered_obligations = frozenset(
        obligation
        for counter_plan in readiness_counters
        for readiness_consumer in counter_plan.consumers
        for obligation in readiness_consumer.covered_obligations
    )
    _validate_schedule_coverage(
        dependency_graph=dependency_graph,
        covered_obligations=covered_obligations,
        root_barrier_edges=root_barrier_edges,
    )
    transient_source_root: int | None = None
    if not continuations:
        globally_scheduled = None
        transient_candidate = (
            _transient_source_candidate(
                readiness_graph,
                readiness_counters,
                root_barrier_edges,
                worker_count=worker_count,
            )
            if allow_transient_source
            else None
        )
        if transient_candidate is not None:
            transient_schedule = _global_unit_list_schedule(
                readiness_graph,
                worker_schedule,
                readiness_counters,
                root_barrier_edges,
                transient_source_root=transient_candidate,
            )
            if transient_schedule is not None and _has_valid_transient_source_schedule(
                transient_schedule,
                readiness_graph,
                transient_candidate,
                readiness_counters,
                root_barrier_edges,
            ):
                globally_scheduled = transient_schedule
                transient_source_root = transient_candidate
        if globally_scheduled is None:
            globally_scheduled = _global_unit_list_schedule(
                readiness_graph,
                worker_schedule,
                readiness_counters,
                root_barrier_edges,
            )
        if globally_scheduled is not None:
            worker_schedule = globally_scheduled
    return StaticPipelinePlan(
        worker_schedule=worker_schedule,
        readiness_counters=readiness_counters,
        root_barrier_edges=root_barrier_edges,
        transient_source_root=transient_source_root,
    )


def _validate_schedule_coverage(
    *,
    dependency_graph: TileDependencyGraph,
    covered_obligations: frozenset[DependencyObligation],
    root_barrier_edges: frozenset[tuple[int, int]],
) -> None:
    """Verify that every dependence has an emitted synchronization path."""
    root_order_edges = set(root_barrier_edges)
    for dependency in dependency_graph.edges:
        pair = (dependency.producer_root, dependency.consumer_root)
        if _is_ordered_by_root_barrier(*pair, root_order_edges):
            continue
        uncovered = tuple(
            obligation
            for access_dependency in dependency.access_dependencies
            for obligation in dependency_graph.dependency_obligations(access_dependency)
            if obligation not in covered_obligations
        )
        if not uncovered:
            continue
        raise exc.CrossLoopSchedulingError(
            f"{dependency.producer_root}->{dependency.consumer_root} through "
            f"allocations {sorted(dependency.tensor_names)!r} has no cross-loop "
            f"synchronization path for dependencies {uncovered!r}"
        )


def _select_root_barrier_edges(
    *,
    dependency_graph: TileDependencyGraph,
    covered_obligations: frozenset[DependencyObligation],
) -> frozenset[tuple[int, int]]:
    """Choose the minimal source-ordered root-barrier fallback edges."""
    selected_edges: set[tuple[int, int]] = set()
    ordered_root_edges: set[tuple[int, int]] = set()
    for dependency in sorted(
        dependency_graph.edges,
        key=lambda edge: (
            edge.consumer_root - edge.producer_root,
            edge.producer_root,
            edge.consumer_root,
        ),
    ):
        pair = (dependency.producer_root, dependency.consumer_root)
        if all(
            dependency_graph.dependency_obligations(access_dependency)
            <= covered_obligations
            for access_dependency in dependency.access_dependencies
        ):
            continue
        if _is_ordered_by_root_barrier(*pair, ordered_root_edges):
            continue
        selected_edges.add(pair)
        ordered_root_edges.add(pair)
    return frozenset(selected_edges)


def _is_ordered_by_root_barrier(
    producer: int,
    consumer: int,
    edges: set[tuple[int, int]],
) -> bool:
    """Return whether whole-root ordering transitively covers one pair."""
    pending = [producer]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current == consumer:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(target for source, target in edges if source == current)
    return False
