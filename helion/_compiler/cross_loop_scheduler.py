from __future__ import annotations

import dataclasses
from functools import cached_property
import itertools
import operator

import sympy

from .. import exc
from .tile_dependency import CoordinateDomain
from .tile_dependency import CoordinateRelation
from .tile_dependency import DependencyObligation
from .tile_dependency import TileDependencyGraph
from .tile_dependency import instantiate_symbolic_dependencies
from .tile_dependency import logical_axis_symbol
from .tile_dependency import nested_logical_axes
from .tile_dependency import preceding_site_relation


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


def _flat_domain_index_expression(domain: CoordinateDomain) -> sympy.Expr:
    """Return the canonical flattened index of a logical coordinate."""
    result: sympy.Expr = sympy.Integer(0)
    multiplier = 1
    for axis in domain.axis_order:
        result += logical_axis_symbol(axis) * multiplier  # pyrefly: ignore[unsupported-operation]
        multiplier *= domain.axis_counts[axis]
    return sympy.simplify(result)


@dataclasses.dataclass(frozen=True)
class WorkerSchedule:
    """Compressed static execution order for all persistent workers."""

    worker_count: int
    segments: tuple[WorkerScheduleSegment, ...]

    def __post_init__(self) -> None:
        if self.worker_count <= 0:
            raise ValueError(f"worker_count must be positive, got {self.worker_count}")
        for segment in self.segments:
            if segment.worker_begin + segment.worker_count > self.worker_count:
                raise ValueError(
                    "worker schedule segment exceeds the resident worker domain"
                )

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
        """Return the compact worker support of one statically placed family."""
        return frozenset(
            segment.worker_begin
            + (segment.dispatch_offset + task_order_index) % segment.worker_count
            for segment in self.segments_for_root(root)
            for task_order_index in range(min(segment.task_count, segment.worker_count))
        )

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
    # Root bodies are emitted in source order. A modeled placement is therefore
    # executable only if no earlier root still occupies this worker step or
    # a later one; codegen cannot move this root ahead of that earlier body.
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
            result.append(
                worker_schedule.replacing_root(
                    root,
                    (
                        WorkerScheduleSegment(
                            root=root,
                            task_order=task_order,
                            worker_begin=worker_begin,
                            worker_count=task_domain.size,
                            dispatch_offset=worker_step * task_domain.size,
                        ),
                    ),
                )
            )
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
            readiness_graph.event(continuation.event_index)
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
            if readiness_graph.event(continuation.event_index)
            .consumers[continuation.consumer_index]
            .consumer_root
            == root
        )
        if len(root_continuations) != 1:
            continue
        continuation = root_continuations[0]
        continuation_event = readiness_graph.event(continuation.event_index)
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
    ordered = order_continuation_producers_by_key(
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
            readiness_graph.event(continuation.event_index).consumers[
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

    event_index: int
    consumer_index: int


def _continuations_by_consumer_root(
    readiness_graph: ReadinessGraph,
    continuations: tuple[FinalArrivalContinuation, ...],
) -> dict[int, FinalArrivalContinuation]:
    """Index final-arrival continuations by their consumer root."""
    return {
        readiness_graph.event(continuation.event_index)
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
    def root_barrier_source(self) -> int | None:
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


def _readiness_producer_is_lowerable(
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


def _add_readiness_event_candidate(
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
    """Canonicalize and group one event candidate by producer partition.

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
        not _readiness_producer_is_lowerable(readiness_producer)
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
    continuation_event = readiness_graph.event(continuation.event_index)
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
    if static_relations is None or any(
        not relation.has_total_source() for _root, relation in static_relations
    ):
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
                    *(logical_axis_symbol(axis) for axis in outer_axes),
                ),
            )
            for stage, (segment_begin, segment_end) in enumerate(segments)
        ),
    )
    converse_consumer = readiness_consumer.keys_by_consumer.converse()
    reduced_converse = (
        None
        if converse_consumer is None
        else converse_consumer.project_target(reduced_domain)
    )
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


def _nested_loop_entry_event(
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
                plan := _nested_loop_entry_event(
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
                continuation.event_index
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
    event. A one-key event is a root barrier and remains on the aggregated
    root-barrier path unless it owns a final-arrival continuation. Unsupported
    relations monotonically fall back to a root barrier during coverage
    selection.
    """
    continuation_consumers = {
        (continuation.event_index, continuation.consumer_index)
        for continuation in continuations
    }
    selected: list[ReadinessCounterPlan] = []
    for event in readiness_graph.events:
        if event.root_barrier_source is not None or any(
            not _readiness_producer_is_lowerable(readiness_producer)
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
        if not retained_consumers or (
            continuation_consumer_index is None and event.readiness_key_count <= 1
        ):
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
        if dependency.relation is not None
        and dependency.relation.source_axes_affecting_targets() is not None
    )
    all_obligations_by_pair: dict[tuple[int, int], set[DependencyObligation]] = {}
    for edge in dependency_graph.edges:
        pair = (edge.producer_root, edge.consumer_root)
        for access_dependency in edge.access_dependencies:
            all_obligations_by_pair.setdefault(pair, set()).update(
                dependency_graph.dependency_obligations(access_dependency)
            )

    implied_obligations: dict[DependencyObligation, set[DependencyObligation]] = {}
    for source in exact_dependencies:
        source_site_id = source.consumer_site_id
        if source_site_id is None or site_by_id[source_site_id].is_root:
            continue
        source_relation = source.relation
        assert source_relation is not None
        source_obligation = (
            source.dependency_id,
            source.producer_site_id,
            source_site_id,
        )
        for later in exact_dependencies:
            later_site_id = later.consumer_site_id
            later_relation = later.relation
            if (
                later is source
                or later_site_id is None
                or later_relation is None
                or source.consumer_root != later.consumer_root
                or source.producer_root != later.producer_root
                or source.producer_site_id != later.producer_site_id
                or source_relation.target_domain != later_relation.target_domain
            ):
                continue
            preceding = preceding_site_relation(
                dependency_graph,
                site_domains=site_domains,
                source_site_id=source_site_id,
                consumer_site_id=later_site_id,
                consumer_access_id=later.consumer_access_id,
            )
            acquired = None if preceding is None else preceding.then(source_relation)
            if acquired is not None and acquired.covers(later_relation):
                implied_obligations.setdefault(source_obligation, set()).add(
                    (
                        later.dependency_id,
                        later.producer_site_id,
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
        relation = dependency.relation
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

    def record_event_candidate(
        *,
        readiness_key_domain: CoordinateDomain,
        producers: tuple[ReadinessProducer, ...],
        consumers: tuple[ReadinessConsumer, ...],
        require_counter_lowering: bool = False,
    ) -> bool:
        if not _add_readiness_event_candidate(
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
            record_event_candidate(
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
            if not record_event_candidate(
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
        _add_readiness_event_candidate(
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
            event.root_barrier_source is not None
            or len(event.consumers) != 1
            or any(
                readiness_producer.producer_site_id is not None
                for readiness_producer in event.producers
            )
        ):
            continue
        if any(
            not _readiness_producer_is_lowerable(readiness_producer)
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
                event_index=event_id,
                consumer_index=consumer_index,
            )
        )
    return tuple(result)


def order_continuation_producers_by_key(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
    continuations: tuple[FinalArrivalContinuation, ...],
) -> WorkerSchedule:
    """Order eligible static producers by readiness key.

    Key-major ordering completes one readiness key at a time so final-arrival work
    becomes ready as early as possible. It is legal only when one producer
    compactly enumerates a complete static task family; all other families keep
    their existing task order.
    """
    continuation_roots = {
        readiness_graph.event(continuation.event_index)
        .consumers[continuation.consumer_index]
        .consumer_root
        for continuation in continuations
    }
    replacement_by_root: dict[int, tuple[WorkerScheduleSegment, ...]] = {}

    for continuation in continuations:
        event = readiness_graph.event(continuation.event_index)
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


def build_static_pipeline_plan(
    *,
    dependency_graph: TileDependencyGraph,
    root_task_orders: tuple[CoordinateRelation, ...],
    site_domains: tuple[CoordinateDomain | None, ...],
    worker_count: int,
    publishable_site_ids: frozenset[int] | None = None,
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
    return StaticPipelinePlan(
        worker_schedule=worker_schedule,
        readiness_counters=readiness_counters,
        root_barrier_edges=root_barrier_edges,
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
