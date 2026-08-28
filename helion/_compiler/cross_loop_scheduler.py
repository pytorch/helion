from __future__ import annotations

import dataclasses
from functools import cached_property
import itertools
import operator

import sympy

from .. import exc
from .tile_dependency import DependencyPoint
from .tile_dependency import LogicalDomain
from .tile_dependency import LogicalRelation
from .tile_dependency import TileDependencyGraph
from .tile_dependency import instantiate_symbolic_dependencies
from .tile_dependency import logical_axis_symbol
from .tile_dependency import nested_logical_axes
from .tile_dependency import preceding_scope_relation


@dataclasses.dataclass(frozen=True)
class WorkerScheduleSegment:
    """One symbolic task-family run in a static persistent-worker schedule.

    ``task_relation`` maps dense segment ordinals to logical tasks.
    ``schedule_begin`` places those ordinals in a linearized range over
    ``worker_count`` workers::

        offset = schedule_begin + ordinal
        worker = worker_begin + offset % worker_count
        position = offset // worker_count

    Several segments can describe arbitrary numbers of waves without
    materializing one schedule entry per runtime task.
    """

    root: int
    task_relation: LogicalRelation
    worker_begin: int
    worker_count: int
    schedule_begin: int

    def __post_init__(self) -> None:
        if self.root < 0:
            raise ValueError(f"root must be nonnegative, got {self.root}")
        if self.worker_begin < 0:
            raise ValueError(
                f"worker_begin must be nonnegative, got {self.worker_begin}"
            )
        if self.worker_count <= 0:
            raise ValueError(f"worker_count must be positive, got {self.worker_count}")
        if self.schedule_begin < 0:
            raise ValueError(
                f"schedule_begin must be nonnegative, got {self.schedule_begin}"
            )
        if (
            self.task_relation.source_domain.size <= 0
            or self.task_relation.source_domain.kind != "worker"
            or self.task_relation.target_domain.kind != "scope"
            or not self.task_relation.pieces
        ):
            raise ValueError(
                "symbolic worker schedule relation has incompatible domains"
            )

    @property
    def task_count(self) -> int:
        """Number of dense ordinals represented by this segment."""
        return self.task_relation.source_domain.size

    def schedule_for_offset(self, task_offset: int) -> int:
        """Return the linearized worker-stream position for one task offset."""
        if not 0 <= task_offset < self.task_count:
            raise IndexError(task_offset)
        return self.schedule_begin + task_offset

    def occupies(self, worker: int, position: int) -> bool:
        """Return whether this segment occupies one worker-stream position."""
        worker_offset = worker - self.worker_begin
        if not 0 <= worker_offset < self.worker_count or position < 0:
            return False
        schedule_offset = position * self.worker_count + worker_offset
        schedule_delta = schedule_offset - self.schedule_begin
        return 0 <= schedule_delta < self.task_count


def _flat_domain_index_expression(domain: LogicalDomain) -> sympy.Expr:
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

    def root_at(self, worker: int, position: int) -> int | None:
        """Return the task family occupying one worker-stream position."""
        roots = tuple(
            segment.root
            for segment in self.segments
            if segment.occupies(worker, position)
        )
        if len(roots) > 1:
            raise AssertionError(
                f"worker {worker} position {position} has multiple tasks"
            )
        return roots[0] if roots else None

    def segments_for_root(self, root: int) -> tuple[WorkerScheduleSegment, ...]:
        """Return the compressed static relation for one task family."""
        return tuple(segment for segment in self.segments if segment.root == root)

    def workers_for_root(self, root: int) -> frozenset[int]:
        """Return the compact worker support of one statically placed family."""
        return frozenset(
            worker
            for segment in self.segments_for_root(root)
            for worker in range(
                segment.worker_begin,
                segment.worker_begin + segment.worker_count,
            )
        )

    def _placement_axes(self) -> tuple[int, int]:
        maximum = max(
            (
                axis
                for segment in self.segments
                for domain in (
                    segment.task_relation.source_domain,
                    segment.task_relation.target_domain,
                )
                for axis in domain.axis_order
            ),
            default=0,
        )
        worker_axis = maximum + 1
        return worker_axis, worker_axis + 1

    @cached_property
    def placement_domain(self) -> LogicalDomain:
        """The worker/stream-position coordinates of static execution."""
        worker_axis, position_axis = self._placement_axes()
        maximum_position = max(
            (
                segment.schedule_for_offset(segment.task_count - 1)
                // segment.worker_count
                for segment in self.segments
            ),
            default=0,
        )
        return LogicalDomain(
            axis_order=(worker_axis, position_axis),
            axis_counts_items=(
                (worker_axis, self.worker_count),
                (position_axis, maximum_position + 1),
            ),
            kind="worker",
        )

    @cached_property
    def position_domain(self) -> LogicalDomain:
        """The projected stream-position coordinate used for frontier math."""
        position_axis = self.placement_domain.axis_order[1]
        return LogicalDomain(
            axis_order=(position_axis,),
            axis_counts_items=(
                (position_axis, self.placement_domain.axis_counts[position_axis]),
            ),
            kind="value",
        )

    def placement_relation(
        self,
        segment: WorkerScheduleSegment,
    ) -> LogicalRelation:
        """Map one segment's ordinal coordinates to worker and stream position."""
        relation = segment.task_relation
        ordinal = _flat_domain_index_expression(relation.source_domain)
        schedule_offset = segment.schedule_begin + ordinal  # pyrefly: ignore[unsupported-operation]
        worker = segment.worker_begin + sympy.Mod(  # pyrefly: ignore[unsupported-operation]
            schedule_offset,
            segment.worker_count,
        )
        position = sympy.floor(schedule_offset / segment.worker_count)
        return LogicalRelation.point_map(
            relation.source_domain,
            self.placement_domain,
            (  # pyrefly: ignore[bad-argument-type]
                (
                    tuple(
                        (axis, 0, relation.source_domain.axis_counts[axis], 1)
                        for axis in relation.source_domain.axis_order
                    ),
                    (worker, position),
                ),
            ),
        )

    def position_relation(
        self,
        segment: WorkerScheduleSegment,
    ) -> LogicalRelation | None:
        """Project one symbolic placement relation to stream position."""
        return self.placement_relation(segment).project_target(self.position_domain)

    def last_positions_for_root(self, root: int) -> dict[int, int]:
        """Return each participating worker's final stream position."""
        result: dict[int, int] = {}
        for segment in self.segments_for_root(root):
            begin = segment.schedule_begin
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

    def position_bounds_for_root(self, root: int) -> tuple[int, int] | None:
        """Return the first and last occupied stream positions for one root."""
        segments = self.segments_for_root(root)
        if not segments:
            return None
        positions = tuple(
            (
                segment.schedule_for_offset(0) // segment.worker_count,
                segment.schedule_for_offset(segment.task_count - 1)
                // segment.worker_count,
            )
            for segment in segments
        )
        return (
            min(begin for begin, _end in positions),
            max(end for _begin, end in positions),
        )

    def contiguous_global_interval(self, root: int) -> tuple[int, int] | None:
        """Return one dense global schedule interval without task expansion."""
        segments = sorted(
            self.segments_for_root(root),
            key=lambda segment: segment.schedule_begin,
        )
        if not segments:
            return None
        begin = segments[0].schedule_begin
        end = begin
        for segment in segments:
            if (
                segment.worker_begin != 0
                or segment.worker_count != self.worker_count
                or segment.schedule_begin != end
            ):
                return None
            end += segment.task_count
        return begin, end

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
    root_domains: tuple[LogicalDomain, ...],
    root_traversals: tuple[LogicalRelation, ...],
    worker_count: int,
) -> WorkerSchedule:
    """Represent the existing source-ordered persistent traversal exactly."""
    if worker_count <= 0:
        raise ValueError(f"worker_count must be positive, got {worker_count}")
    segments: list[WorkerScheduleSegment] = []
    position_begin = 0
    if len(root_domains) != len(root_traversals):
        raise ValueError("root domains and traversals must have equal length")
    for root, (domain, traversal) in enumerate(
        zip(root_domains, root_traversals, strict=True)
    ):
        task_count = domain.size
        if task_count <= 0:
            continue
        active_workers = min(worker_count, task_count)
        segments.append(
            WorkerScheduleSegment(
                root=root,
                task_relation=traversal,
                worker_begin=0,
                worker_count=active_workers,
                schedule_begin=position_begin * active_workers,
            )
        )
        position_begin += (task_count + worker_count - 1) // worker_count
    return WorkerSchedule(worker_count=worker_count, segments=tuple(segments))


def _family_placements_at_position(
    worker_schedule: WorkerSchedule,
    *,
    root: int,
    task_domain: LogicalDomain,
    task_traversal: LogicalRelation,
    position: int,
    unavailable_workers: frozenset[int] = frozenset(),
) -> tuple[WorkerSchedule, ...]:
    """Return dense placements for one complete family in free worker runs."""
    if task_domain.size > worker_schedule.worker_count:
        return ()
    free_workers = [
        worker
        for worker in range(worker_schedule.worker_count)
        if worker not in unavailable_workers
        and (
            (occupant_root := worker_schedule.root_at(worker, position)) is None
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
                            task_relation=task_traversal,
                            worker_begin=worker_begin,
                            worker_count=task_domain.size,
                            schedule_begin=position * task_domain.size,
                        ),
                    ),
                )
            )
        run_end = run_begin
    return tuple(result)


def place_ready_families(
    event_graph: EventGraph,
    original_schedule: WorkerSchedule,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...],
) -> tuple[WorkerSchedule, tuple[LocalTrigger, ...]]:
    """Move complete ready families into idle capacity during a producer tail.

    Final-arrival execution is useful when no separate workers are available.
    When a complete consumer family fits on workers that are free while some
    of its static ancestors still have queued work, a direct event wait avoids
    extending those producer streams.  This is derived from schedule liveness,
    independent of the roots' operations or graph topology.
    """
    result = worker_schedule
    remaining_triggers = local_triggers
    local_trigger_by_root = _index_local_triggers(event_graph, local_triggers)
    candidate_roots = sorted(
        {
            event_graph.event(trigger.event_index).uses[trigger.use_index].consumer_root
            for trigger in remaining_triggers
        }
    )
    for root in candidate_roots:
        task_domain = event_graph.root_domains[root]
        if task_domain.size > result.worker_count:
            continue
        root_triggers = tuple(
            trigger
            for trigger in remaining_triggers
            if event_graph.event(trigger.event_index)
            .uses[trigger.use_index]
            .consumer_root
            == root
        )
        if len(root_triggers) != 1:
            continue
        trigger = root_triggers[0]
        trigger_event = event_graph.event(trigger.event_index)
        trigger_use = trigger_event.uses[trigger.use_index]
        completion = _event_completion_positions(
            event_graph,
            trigger_event,
            worker_schedule=result,
            local_trigger_by_root=local_trigger_by_root,
        )
        if completion is None:
            continue
        completion_positions, prerequisite_roots = completion
        readiness = trigger_use.keys.then(completion_positions)
        readiness_bounds = None if readiness is None else readiness.value_bounds()
        if readiness_bounds is None:
            continue
        ancestor_placements: set[tuple[int, int]] = set()
        for prerequisite_root in prerequisite_roots:
            last_positions = result.last_positions_for_root(prerequisite_root)
            ancestor_placements.update(last_positions.items())
        if not ancestor_placements:
            continue

        original_bounds = original_schedule.position_bounds_for_root(root)
        if original_bounds is None:
            continue
        original_position = original_bounds[0]
        remaining_without_root = tuple(
            trigger for trigger in remaining_triggers if trigger not in root_triggers
        )
        for position in range(readiness_bounds[0] + 1, original_position):
            unfinished_workers = frozenset(
                worker
                for worker, ancestor_position in ancestor_placements
                if ancestor_position >= position
            )
            if not unfinished_workers:
                break
            candidate = next(
                (
                    candidate
                    for candidate in _family_placements_at_position(
                        result,
                        root=root,
                        task_domain=task_domain,
                        task_traversal=event_graph.root_traversals[root],
                        position=position,
                        unavailable_workers=unfinished_workers,
                    )
                ),
                None,
            )
            if candidate is None:
                continue
            result = candidate
            remaining_triggers = remaining_without_root
            local_trigger_by_root.pop(root)
            break
    return result, remaining_triggers


def build_worker_schedule(
    event_graph: EventGraph,
    *,
    worker_count: int,
) -> tuple[
    WorkerSchedule,
    tuple[LocalTrigger, ...],
    tuple[CountedEventPlan, ...],
    EventGraph,
]:
    """Derive local and static task placement for one worker count."""
    baseline = build_baseline_worker_schedule(
        event_graph.root_domains,
        event_graph.root_traversals,
        worker_count,
    )
    nested_wait_roots = frozenset(
        use.consumer_root
        for event in event_graph.events
        for use in event.uses
        if use.consumer_scope_id is not None
    )
    local_triggers = choose_local_triggers(
        event_graph,
        baseline,
        worker_limit=worker_count,
        excluded_roots=nested_wait_roots,
    )
    ordered = order_local_contributors_by_key(
        event_graph,
        baseline,
        local_triggers,
    )
    local_triggers = choose_local_triggers(
        event_graph,
        ordered,
        worker_limit=worker_count,
        excluded_roots=nested_wait_roots,
    )
    local_roots = frozenset(
        use.consumer_root
        for trigger in local_triggers
        for use in (event_graph.event(trigger.event_index).uses[trigger.use_index],)
    )
    schedule = ordered.without_roots(local_roots)
    schedule, nested_scope_events = place_nested_scope_consumers(
        event_graph,
        schedule,
        local_triggers,
    )
    nested_dependency_points = frozenset(
        dependency_point
        for event in nested_scope_events
        for use in event.uses
        for dependency_point in use.dependency_points
    )
    scheduled_event_graph = _without_root_uses_for_dependencies(
        event_graph, nested_dependency_points
    )
    schedule, local_triggers = place_ready_families(
        scheduled_event_graph,
        ordered,
        schedule,
        local_triggers,
    )
    return schedule, local_triggers, nested_scope_events, scheduled_event_graph


@dataclasses.dataclass(frozen=True)
class LocalTrigger:
    """A task use executed by whichever contributor makes the final arrival."""

    event_index: int
    use_index: int


def _index_local_triggers(
    event_graph: EventGraph,
    local_triggers: tuple[LocalTrigger, ...],
) -> dict[int, LocalTrigger]:
    """Index final-arrival triggers by their consumer root."""
    return {
        event_graph.event(trigger.event_index)
        .uses[trigger.use_index]
        .consumer_root: trigger
        for trigger in local_triggers
    }


@dataclasses.dataclass(frozen=True)
class EventContribution:
    """A producer execution scope's symbolic contribution to one event."""

    producer_root: int
    predecessors: LogicalRelation
    producer_scope_id: int | None = None

    @cached_property
    def _readiness_relations(
        self,
    ) -> tuple[LogicalRelation | None, LogicalRelation | None]:
        """Derive publication and fan-in from one fiber analysis."""
        return self.predecessors.fiber_analysis()

    @property
    def producer_to_keys(self) -> LogicalRelation | None:
        """Return the derived publication relation, when representable."""
        return self._readiness_relations[0]

    @property
    def arrivals_per_key(self) -> LogicalRelation | None:
        """Return the exact symbolic number of arrivals for each event key."""
        return self._readiness_relations[1]


def _uniform_arrivals(
    contributions: tuple[EventContribution, ...],
) -> int | None:
    """Return one constant arrival count for an event, when it has one."""
    total = 0
    for contribution in contributions:
        cardinality = contribution.arrivals_per_key
        count = None if cardinality is None else cardinality.constant_value()
        if count is None:
            return None
        total += count
    return total


@dataclasses.dataclass(frozen=True)
class EventUse:
    """A consumer execution scope's symbolic requirements from one event."""

    consumer_root: int
    keys: LogicalRelation
    dependency_points: frozenset[DependencyPoint] = frozenset()
    consumer_scope_id: int | None = None


def _event_key_domain(
    contributions: tuple[EventContribution, ...],
    uses: tuple[EventUse, ...],
) -> LogicalDomain:
    """Validate and return the shared key domain of one event."""
    if not contributions:
        raise ValueError("an event requires at least one contributor")
    key_domain = contributions[0].predecessors.source_domain
    if any(
        contributor.predecessors.source_domain != key_domain
        for contributor in contributions[1:]
    ) or any(use.keys.target_domain != key_domain for use in uses):
        raise ValueError("event relations must share one key domain")
    return key_domain


@dataclasses.dataclass(frozen=True)
class KeyedEvent:
    """One symbolic readiness event shared by scheduling and lowering."""

    contributions: tuple[EventContribution, ...]
    uses: tuple[EventUse, ...]

    def __post_init__(self) -> None:
        key_domain = _event_key_domain(self.contributions, self.uses)
        if key_domain.kind != "event":
            raise ValueError("event key domain must have event kind")
        if key_domain.identity is None or key_domain.identity < 0:
            raise ValueError("event key domain must have a nonnegative identity")
        if key_domain.axis_order != tuple(range(len(key_domain.axis_order))):
            raise ValueError("event key axes must use canonical local ordinals")
        if key_domain.block_sizes_items:
            raise ValueError("event key domains must not inherit scope block sizes")

    @property
    def key_domain(self) -> LogicalDomain:
        """Return the key domain owned by every event relation."""
        return self.contributions[0].predecessors.source_domain

    @property
    def event_id(self) -> int:
        """Return the event identity owned by its key domain."""
        identity = self.key_domain.identity
        assert identity is not None
        return identity

    @property
    def key_count(self) -> int:
        return self.key_domain.size

    @property
    def family_done_root(self) -> int | None:
        if (
            self.key_count == 1
            and len(self.contributions) == 1
            and self.contributions[0].producer_scope_id is None
            and self.contributions[0].predecessors.is_total()
        ):
            return self.contributions[0].producer_root
        return None


@dataclasses.dataclass(frozen=True)
class EventGraph:
    """Configured symbolic readiness DAG and its execution-scope domains."""

    root_traversals: tuple[LogicalRelation, ...]
    scope_domains: tuple[LogicalDomain | None, ...]
    events: tuple[KeyedEvent, ...]

    def __post_init__(self) -> None:
        for traversal in self.root_traversals:
            if (
                traversal.source_domain.size != traversal.target_domain.size
                or traversal.source_domain.kind != "worker"
                or traversal.target_domain.kind != "scope"
                or not traversal.pieces
            ):
                raise ValueError(
                    "each root traversal must have compatible typed domains"
                )
        if tuple(event.event_id for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("event IDs must be dense and source ordered")

    @property
    def root_domains(self) -> tuple[LogicalDomain, ...]:
        """Return the task domains owned by the physical root traversals."""
        return tuple(traversal.target_domain for traversal in self.root_traversals)

    def event(self, event_id: int) -> KeyedEvent:
        return self.events[event_id]

    def scope_domain(self, scope_id: int) -> LogicalDomain:
        domain = self.scope_domains[scope_id]
        if domain is None:
            raise ValueError(f"execution scope {scope_id} has no configured domain")
        return domain

    def nested_axes(self, root: int, scope_id: int) -> tuple[int, ...]:
        return nested_logical_axes(
            self.root_domains[root],
            self.scope_domain(scope_id),
        )

    def source_traversal(self, root: int, scope_id: int | None) -> tuple[int, ...]:
        root_axes = self.root_domains[root].axis_order
        if scope_id is None:
            return root_axes
        return (*self.nested_axes(root, scope_id), *root_axes)

    def required_keys_by_strand(self, use: EventUse) -> LogicalRelation | None:
        """Project a checkpoint's requirements onto its owning root strands.

        Projection failure is a legality failure for strand-level scheduling;
        it must not trigger enumeration of the nested action domain.
        """
        root_domain = self.root_domains[use.consumer_root]
        if use.consumer_scope_id is None:
            if use.keys.source_domain != root_domain:
                raise ValueError("root event use has the wrong source domain")
            return use.keys
        return use.keys.project_source(root_domain)


def _counted_contribution_is_lowerable(contribution: EventContribution) -> bool:
    """Keep scheduler eligibility identical to counted-event code generation."""
    publication = contribution.producer_to_keys
    return (
        contribution.arrivals_per_key is not None
        and publication is not None
        and publication.canonical_single_valued() is not None
    )


def _canonical_event_domain(domain: LogicalDomain) -> LogicalDomain:
    """Name quotient coordinates locally rather than borrowing scope axes."""
    return LogicalDomain(
        axis_order=tuple(range(len(domain.axis_order))),
        axis_counts_items=tuple(
            (event_axis, count)
            for event_axis, (_scope_axis, count) in enumerate(domain.axis_counts_items)
        ),
        kind="event",
    )


def _canonical_event_use_relation(
    relation: LogicalRelation,
    key_domain: LogicalDomain,
) -> LogicalRelation:
    """Express one event use in its event-local coordinate chart."""
    old_domain = relation.target_domain
    if (
        old_domain.kind != "event"
        or old_domain.identity is not None
        or tuple(old_domain.axis_counts.values())
        != tuple(key_domain.axis_counts.values())
    ):
        raise AssertionError("event relation does not match its quotient geometry")
    renamed_axes = dict(zip(old_domain.axis_order, key_domain.axis_order, strict=True))
    return LogicalRelation(
        source_domain=relation.source_domain,
        target_domain=key_domain,
        pieces=tuple(
            dataclasses.replace(
                piece,
                target_ranges=tuple(
                    (renamed_axes[axis], begin, end, step)
                    for axis, begin, end, step in piece.target_ranges
                ),
            )
            for piece in relation.pieces
        ),
    )


def _canonical_event_predecessors(
    relation: LogicalRelation,
    key_domain: LogicalDomain,
) -> LogicalRelation:
    """Express key-to-producer fibers in event-local coordinates."""
    old_domain = relation.source_domain
    if (
        old_domain.kind != "event"
        or old_domain.identity is not None
        or tuple(old_domain.axis_counts.values())
        != tuple(key_domain.axis_counts.values())
    ):
        raise AssertionError("event relation does not match its quotient geometry")
    renamed_axes = dict(zip(old_domain.axis_order, key_domain.axis_order, strict=True))
    substitutions = {
        logical_axis_symbol(axis): logical_axis_symbol(renamed_axes[axis])
        for axis in old_domain.axis_order
    }
    return LogicalRelation(
        source_domain=key_domain,
        target_domain=relation.target_domain,
        pieces=tuple(
            dataclasses.replace(
                piece,
                source_bounds_items=tuple(
                    (renamed_axes[axis], begin, end, step)
                    for axis, begin, end, step in piece.source_bounds_items
                ),
                target_ranges=tuple(
                    (
                        axis,
                        begin.xreplace(substitutions),
                        end.xreplace(substitutions),
                        step,
                    )
                    for axis, begin, end, step in piece.target_ranges
                ),
            )
            for piece in relation.pieces
        ),
    )


def _add_event_candidate(
    pending: dict[
        tuple[LogicalDomain, tuple[EventContribution, ...]],
        KeyedEvent,
    ],
    *,
    key_domain: LogicalDomain,
    contributions: tuple[EventContribution, ...],
    uses: tuple[EventUse, ...],
) -> None:
    """Group fanout by producer partition in final event-local coordinates."""
    canonical_domain = _canonical_event_domain(key_domain)
    canonical_contributions = tuple(
        dataclasses.replace(
            contribution,
            predecessors=_canonical_event_predecessors(
                contribution.predecessors,
                canonical_domain,
            ),
        )
        for contribution in contributions
    )
    canonical_uses = tuple(
        dataclasses.replace(
            use,
            keys=_canonical_event_use_relation(use.keys, canonical_domain),
        )
        for use in uses
    )
    signature = canonical_domain, canonical_contributions
    previous_event = pending.get(signature)
    if previous_event is None:
        event_id = len(pending)
        identified_domain = dataclasses.replace(canonical_domain, identity=event_id)
        identified_contributions: list[EventContribution] = []
        for contribution in canonical_contributions:
            predecessors = contribution.predecessors.retype_source(identified_domain)
            if predecessors is None:
                raise AssertionError("event identity assignment changed key geometry")
            identified_contributions.append(
                dataclasses.replace(contribution, predecessors=predecessors)
            )
        event_contributions = tuple(identified_contributions)
        previous_uses: tuple[EventUse, ...] = ()
    else:
        event_id = previous_event.event_id
        identified_domain = previous_event.key_domain
        event_contributions = previous_event.contributions
        previous_uses = previous_event.uses

    grouped_uses = list(previous_uses)
    for canonical_use in canonical_uses:
        keys = canonical_use.keys.retarget(identified_domain)
        if keys is None:
            raise AssertionError("event identity assignment changed key geometry")
        use = dataclasses.replace(canonical_use, keys=keys)
        matching_index = next(
            (
                index
                for index, previous in enumerate(grouped_uses)
                if previous.consumer_root == use.consumer_root
                and previous.consumer_scope_id == use.consumer_scope_id
                and previous.keys == use.keys
            ),
            None,
        )
        if matching_index is None:
            grouped_uses.append(use)
            continue
        previous = grouped_uses[matching_index]
        grouped_uses[matching_index] = dataclasses.replace(
            previous,
            dependency_points=previous.dependency_points | use.dependency_points,
        )
    pending[signature] = KeyedEvent(
        contributions=event_contributions,
        uses=tuple(grouped_uses),
    )


def _without_root_uses_for_dependencies(
    event_graph: EventGraph,
    dependency_points: frozenset[DependencyPoint],
) -> EventGraph:
    """Remove root-entry alternatives covered by selected action checkpoints."""
    if not dependency_points:
        return event_graph
    events: list[KeyedEvent] = []
    for event in event_graph.events:
        uses: list[EventUse] = []
        for use in event.uses:
            if use.consumer_scope_id is not None:
                uses.append(use)
                continue
            remaining = use.dependency_points - dependency_points
            if remaining:
                uses.append(dataclasses.replace(use, dependency_points=remaining))
        events.append(dataclasses.replace(event, uses=tuple(uses)))
    return dataclasses.replace(event_graph, events=tuple(events))


@dataclasses.dataclass(frozen=True)
class CountedEventPlan:
    """A logical key space receiving contributions from one or more roots.

    Each contributor has an independently proved key-to-predecessor relation.
    The expected count is derived by summing its fibers; the event therefore
    represents both ordinary continuations and generic multi-predecessor joins.
    Consumer uses are independent of event identity. ``local_trigger_use``
    identifies the optional use executed by the final arriving contributor.
    """

    contributions: tuple[EventContribution, ...]
    uses: tuple[EventUse, ...]
    local_trigger_use: int | None = None

    def __post_init__(self) -> None:
        _event_key_domain(self.contributions, self.uses)

    @property
    def key_domain(self) -> LogicalDomain:
        """Return the key domain owned by every event relation."""
        return self.contributions[0].predecessors.source_domain

    @property
    def local_use(self) -> EventUse | None:
        if self.local_trigger_use is None:
            return None
        return self.uses[self.local_trigger_use]

    @property
    def key_count(self) -> int:
        """Return the complete event-key domain used by producers or consumers."""
        return self.key_domain.size

    def uniform_arrivals(self) -> int | None:
        """Return constant fan-in without enumerating event keys."""
        return _uniform_arrivals(self.contributions)


@dataclasses.dataclass(frozen=True)
class _ScopeReadiness:
    """Configured readiness of one nested scope on its owning task strands."""

    event: KeyedEvent
    use: EventUse
    domain: LogicalDomain
    readiness: LogicalRelation
    ancestor_placements: frozenset[tuple[int, int]]


def _merge_relations_by_root(
    relations: tuple[tuple[int, LogicalRelation], ...],
) -> tuple[tuple[int, LogicalRelation], ...] | None:
    merged: dict[int, LogicalRelation] = {}
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


def _static_contribution_relations(
    event_graph: EventGraph,
    *,
    root: int,
    scope_id: int | None,
    keys: LogicalRelation,
    local_trigger_by_root: dict[int, LocalTrigger],
    visiting: frozenset[int] = frozenset(),
) -> tuple[tuple[int, LogicalRelation], ...] | None:
    """Contract local execution to relations from statically scheduled roots."""
    root_domain = event_graph.root_domains[root]
    root_keys = keys if scope_id is None else keys.project_source(root_domain)
    if root_keys is None:
        return None
    trigger = local_trigger_by_root.get(root)
    if trigger is None:
        return ((root, root_keys),)
    if root in visiting:
        return None
    trigger_event = event_graph.event(trigger.event_index)
    trigger_use = trigger_event.uses[trigger.use_index]
    inverse_use = trigger_use.keys.inverse()
    key_to_target = None if inverse_use is None else inverse_use.then(root_keys)
    if key_to_target is None:
        return None
    expanded: list[tuple[int, LogicalRelation]] = []
    for contribution in trigger_event.contributions:
        publication = contribution.producer_to_keys
        upstream_keys = None if publication is None else publication.then(key_to_target)
        if upstream_keys is None:
            return None
        upstream = _static_contribution_relations(
            event_graph,
            root=contribution.producer_root,
            scope_id=contribution.producer_scope_id,
            keys=upstream_keys,
            local_trigger_by_root=local_trigger_by_root,
            visiting=visiting | frozenset((root,)),
        )
        if upstream is None:
            return None
        expanded.extend(upstream)
    return _merge_relations_by_root(tuple(expanded))


def _event_static_contributions(
    event_graph: EventGraph,
    event: KeyedEvent,
    local_trigger_by_root: dict[int, LocalTrigger],
) -> tuple[tuple[int, LogicalRelation], ...] | None:
    expanded: list[tuple[int, LogicalRelation]] = []
    for contribution in event.contributions:
        publication = contribution.producer_to_keys
        if publication is None:
            return None
        static_relations = _static_contribution_relations(
            event_graph,
            root=contribution.producer_root,
            scope_id=contribution.producer_scope_id,
            keys=publication,
            local_trigger_by_root=local_trigger_by_root,
        )
        if static_relations is None:
            return None
        expanded.extend(static_relations)
    return _merge_relations_by_root(tuple(expanded))


def _transitive_static_prerequisite_roots(
    event_graph: EventGraph,
    static_relations: tuple[tuple[int, LogicalRelation], ...],
    local_trigger_by_root: dict[int, LocalTrigger],
) -> frozenset[int] | None:
    """Close static contributors through waits earlier on their task strands."""
    roots = {root for root, _relation in static_relations}
    pending = list(roots)
    while pending:
        consumer_root = pending.pop()
        for event in event_graph.events:
            if not any(use.consumer_root == consumer_root for use in event.uses):
                continue
            upstream = _event_static_contributions(
                event_graph,
                event,
                local_trigger_by_root,
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


def _event_completion_positions(
    event_graph: EventGraph,
    event: KeyedEvent,
    *,
    worker_schedule: WorkerSchedule,
    local_trigger_by_root: dict[int, LocalTrigger],
) -> tuple[LogicalRelation, frozenset[int]] | None:
    """Return event-key frontiers and their ultimate static contributors."""
    static_relations = _event_static_contributions(
        event_graph,
        event,
        local_trigger_by_root,
    )
    if static_relations is None or any(
        not relation.has_total_source() for _root, relation in static_relations
    ):
        return None
    prerequisite_roots = _transitive_static_prerequisite_roots(
        event_graph,
        static_relations,
        local_trigger_by_root,
    )
    if prerequisite_roots is None:
        return None
    maxima: list[LogicalRelation] = []
    for root, keys in static_relations:
        root_domain = event_graph.root_domains[root]
        if keys.source_domain != root_domain:
            return None
        for segment in worker_schedule.segments_for_root(root):
            task_relation = segment.task_relation
            if task_relation.target_domain != root_domain:
                return None
            ordinal_keys = task_relation.then(keys)
            inverse = None if ordinal_keys is None else ordinal_keys.inverse()
            positions = worker_schedule.position_relation(segment)
            maximum = (
                None
                if inverse is None or positions is None
                else inverse.fiber_maximum(positions)
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
    identity = LogicalRelation.identity(
        worker_schedule.position_domain,
        worker_schedule.position_domain,
    )
    maximum = combined.fiber_maximum(identity)
    return None if maximum is None else (maximum, prerequisite_roots)


def _scope_readiness(
    event_graph: EventGraph,
    event: KeyedEvent,
    use: EventUse,
    *,
    worker_schedule: WorkerSchedule,
    local_trigger_by_root: dict[int, LocalTrigger],
) -> _ScopeReadiness | None:
    """Project one exact action relation onto static worker completion positions."""
    assert use.consumer_scope_id is not None
    domain = event_graph.scope_domain(use.consumer_scope_id)
    nested_axes = event_graph.nested_axes(
        use.consumer_root,
        use.consumer_scope_id,
    )
    if len(nested_axes) != 1 or use.keys.source_domain != domain:
        return None

    completion = _event_completion_positions(
        event_graph,
        event,
        worker_schedule=worker_schedule,
        local_trigger_by_root=local_trigger_by_root,
    )
    if completion is None:
        return None
    completion_positions, prerequisite_roots = completion
    action_readiness = use.keys.then(completion_positions)
    if action_readiness is None or not action_readiness.is_total_function():
        return None
    ancestor_placements: set[tuple[int, int]] = set()
    for root in prerequisite_roots:
        last_positions = worker_schedule.last_positions_for_root(root)
        ancestor_placements.update(last_positions.items())
    return _ScopeReadiness(
        event=event,
        use=use,
        domain=domain,
        readiness=action_readiness,
        ancestor_placements=frozenset(ancestor_placements),
    )


def _segmented_scope_event(
    event_graph: EventGraph,
    event: KeyedEvent,
    use: EventUse,
    boundaries: tuple[int, ...],
) -> CountedEventPlan | None:
    """Coarsen one exact nested dependency into contiguous action segments."""
    consumer_scope_id = use.consumer_scope_id
    assert consumer_scope_id is not None
    domain = event_graph.scope_domain(consumer_scope_id)
    nested_axes = event_graph.nested_axes(
        use.consumer_root,
        consumer_scope_id,
    )
    if len(nested_axes) != 1:
        return None
    (nested_axis,) = nested_axes
    segments = tuple(itertools.pairwise(boundaries))
    if not segments or any(begin >= end for begin, end in segments):
        return None
    used_axes = use.keys.source_axes_used()
    if used_axes is None or nested_axis not in used_axes:
        return None
    reduced_domain = LogicalDomain(
        axis_order=used_axes,
        axis_counts_items=tuple((axis, domain.axis_counts[axis]) for axis in used_axes),
        block_sizes_items=tuple(
            (axis, domain.block_sizes[axis])
            for axis in used_axes
            if axis in domain.block_sizes
        ),
        kind="scope",
        identity=domain.identity,
    )
    outer_axes = tuple(axis for axis in used_axes if axis != nested_axis)
    key_domain = LogicalDomain(
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
    stage_keys = LogicalRelation.point_map(
        reduced_domain,
        key_domain,
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
    inverse_use = use.keys.inverse()
    reduced_inverse = (
        None if inverse_use is None else inverse_use.project_target(reduced_domain)
    )
    coarsening = None if reduced_inverse is None else reduced_inverse.then(stage_keys)
    if coarsening is None:
        return None
    # This is a scheduling-derived coarsening of an already lowerable event,
    # not a second dependency fact. Derive producer publication from the
    # authoritative predecessor fibers, compose it with the stage map, then
    # invert the exact result back into the representation owned by the plan.
    publication_relations = tuple(
        (
            None
            if contribution.producer_to_keys is None
            else contribution.producer_to_keys.then(coarsening)
        )
        for contribution in event.contributions
    )
    if any(relation is None for relation in publication_relations):
        return None
    predecessor_relations = tuple(
        None if relation is None else relation.inverse()
        for relation in publication_relations
    )
    if any(relation is None for relation in predecessor_relations):
        return None
    action_keys = stage_keys.lift_source(domain)
    if action_keys is None:
        return None

    return CountedEventPlan(
        contributions=tuple(
            EventContribution(
                producer_root=contribution.producer_root,
                producer_scope_id=contribution.producer_scope_id,
                predecessors=relation,
            )
            for contribution, relation in zip(
                event.contributions,
                predecessor_relations,
                strict=True,
            )
            if relation is not None
        ),
        uses=(
            EventUse(
                consumer_root=use.consumer_root,
                dependency_points=use.dependency_points,
                consumer_scope_id=use.consumer_scope_id,
                keys=action_keys,
            ),
        ),
    )


def _scope_milestones(
    event_graph: EventGraph,
    readiness: _ScopeReadiness,
    *,
    consumer_position: int,
) -> CountedEventPlan | None:
    """Split a nested scope loop at the selected schedule frontier."""
    domain = readiness.domain
    consumer_scope_id = readiness.use.consumer_scope_id
    assert consumer_scope_id is not None
    nested_axes = event_graph.nested_axes(
        readiness.use.consumer_root,
        consumer_scope_id,
    )
    if len(nested_axes) != 1:
        return None
    (nested_axis,) = nested_axes
    actions_per_strand = domain.axis_counts[nested_axis]

    def ready(action_offset: int) -> bool | None:
        value_bounds = readiness.readiness.value_bounds({nested_axis: action_offset})
        if value_bounds is None:
            return None
        # Producers at the same stream position execute concurrently on other
        # workers. They permit placement at this position, but are not ready at
        # admission: their consumer actions belong after the single frontier.
        # ``ancestor_placements`` separately prevents self-deadlock on a
        # producer's own worker.
        return value_bounds[1] < consumer_position

    first_ready = ready(0)
    last_ready = ready(actions_per_strand - 1)
    if first_ready is None or last_ready is None:
        return None
    if not first_ready:
        frontier = 0
    elif last_ready:
        frontier = actions_per_strand
    else:
        lower = 0
        upper = actions_per_strand - 1
        while lower + 1 < upper:
            midpoint = (lower + upper) // 2
            midpoint_ready = ready(midpoint)
            if midpoint_ready is None:
                return None
            if midpoint_ready:
                lower = midpoint
            else:
                upper = midpoint
        frontier = upper
    boundaries = tuple(sorted({0, frontier, actions_per_strand}))
    return _segmented_scope_event(
        event_graph,
        readiness.event,
        readiness.use,
        boundaries,
    )


def _scope_entry_event(
    event_graph: EventGraph,
    event: KeyedEvent,
    use: EventUse,
) -> CountedEventPlan | None:
    """Coarsen exact action readiness to one wait per owning task strand."""
    consumer_scope_id = use.consumer_scope_id
    assert consumer_scope_id is not None
    domain = event_graph.scope_domain(consumer_scope_id)
    nested_axes = event_graph.nested_axes(
        use.consumer_root,
        consumer_scope_id,
    )
    if len(nested_axes) != 1:
        return None
    actions_per_strand = domain.axis_counts[nested_axes[0]]
    return _segmented_scope_event(
        event_graph,
        event,
        use,
        (0, actions_per_strand),
    )


def place_nested_scope_consumers(
    event_graph: EventGraph,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...],
) -> tuple[WorkerSchedule, tuple[CountedEventPlan, ...]]:
    """Place strands with nested waits and derive their milestone events.

    Exact action dependencies remain the semantic source of truth. This pass
    uses only worker positions and same-strand action order to select one
    admission frontier for the original nested loop.
    It does not inspect operation kinds or recognize a graph topology.
    """
    uses_by_consumer: dict[
        int,
        list[tuple[KeyedEvent, EventUse]],
    ] = {}
    local_trigger_by_root = _index_local_triggers(event_graph, local_triggers)
    for event in event_graph.events:
        for use in event.uses:
            if use.consumer_scope_id is not None:
                uses_by_consumer.setdefault(use.consumer_root, []).append((event, use))

    result = worker_schedule
    plans: list[CountedEventPlan] = []
    for consumer_root, event_uses in sorted(uses_by_consumer.items()):
        task_domain = event_graph.root_domains[consumer_root]

        # A preceding scope may already carry every dependency point needed by
        # a later scope.  The implication was proved from DeviceIR program
        # order when the event graph was built, so the later wait is redundant.
        uncovered_event_uses: list[tuple[KeyedEvent, EventUse]] = []
        preceding_dependency_points: set[DependencyPoint] = set()
        for event, use in sorted(
            event_uses,
            key=lambda item: (
                item[1].consumer_scope_id
                if item[1].consumer_scope_id is not None
                else -1,
                item[0].event_id,
            ),
        ):
            if use.dependency_points and use.dependency_points <= (
                preceding_dependency_points
            ):
                continue
            uncovered_event_uses.append((event, use))
            preceding_dependency_points.update(use.dependency_points)

        scope_entry_plans = tuple(
            plan
            for event, use in uncovered_event_uses
            if (plan := _scope_entry_event(event_graph, event, use)) is not None
        )
        if task_domain.size > result.worker_count:
            plans.extend(scope_entry_plans)
            continue

        readiness = tuple(
            _scope_readiness(
                event_graph,
                event,
                use,
                worker_schedule=result,
                local_trigger_by_root=local_trigger_by_root,
            )
            for event, use in uncovered_event_uses
        )
        if not readiness or any(item is None for item in readiness):
            plans.extend(scope_entry_plans)
            continue
        ordered_readiness = tuple(item for item in readiness if item is not None)

        current_consumer_bounds = result.position_bounds_for_root(consumer_root)
        if current_consumer_bounds is None:
            plans.extend(scope_entry_plans)
            continue
        original_position = current_consumer_bounds[0]
        readiness_bounds = tuple(
            item.readiness.value_bounds() for item in ordered_readiness
        )
        if any(bounds is None for bounds in readiness_bounds):
            plans.extend(scope_entry_plans)
            continue
        earliest_readiness = min(
            bounds[0] for bounds in readiness_bounds if bounds is not None
        )
        ancestor_placements = frozenset(
            placement
            for item in ordered_readiness
            for placement in item.ancestor_placements
        )
        chosen: tuple[WorkerSchedule, tuple[CountedEventPlan, ...]] | None = None
        for position in range(earliest_readiness + 1, original_position):
            busy_workers = frozenset(
                worker
                for worker, ancestor_position in ancestor_placements
                if ancestor_position >= position
            )
            candidate = next(
                (
                    candidate
                    for candidate in _family_placements_at_position(
                        result,
                        root=consumer_root,
                        task_domain=task_domain,
                        task_traversal=event_graph.root_traversals[consumer_root],
                        position=position,
                        unavailable_workers=busy_workers,
                    )
                ),
                None,
            )
            if candidate is None:
                continue
            milestone_results = tuple(
                _scope_milestones(
                    event_graph,
                    item,
                    consumer_position=position,
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
            plans.extend(scope_entry_plans)
            continue
        result, scope_plans = chosen
        plans.extend(scope_plans)
    return result, tuple(plans)


@dataclasses.dataclass(frozen=True)
class CrossLoopSchedule:
    """Pure graph-derived choices consumed by persistent-kernel lowering."""

    worker_schedule: WorkerSchedule
    counted_events: tuple[CountedEventPlan, ...]
    root_completion_edges: frozenset[tuple[int, int]]


def choose_local_triggers(
    event_graph: EventGraph,
    worker_schedule: WorkerSchedule,
    *,
    worker_limit: int,
    excluded_roots: frozenset[int] = frozenset(),
) -> tuple[LocalTrigger, ...]:
    """Choose final-arrival execution from complete exact task readiness.

    A one-task family has no task-level parallelism to expose, so it remains in
    the static schedule. A family larger than the resident grid also remains
    static when its completion is consumed coarsely.
    """
    family_done_roots = {
        family_done_root
        for event in event_graph.events
        if (family_done_root := event.family_done_root) is not None and event.uses
    }
    return tuple(
        trigger
        for trigger in derive_local_triggers(event_graph, worker_schedule)
        if (
            use := event_graph.event(trigger.event_index).uses[trigger.use_index]
        ).consumer_root
        not in excluded_roots
        and event_graph.root_domains[use.consumer_root].size > 1
        and not (
            use.consumer_root in family_done_roots
            and event_graph.root_domains[use.consumer_root].size > worker_limit
        )
    )


def choose_counted_events(
    event_graph: EventGraph,
    local_triggers: tuple[LocalTrigger, ...],
    *,
    excluded_dependency_points: frozenset[DependencyPoint] = frozenset(),
) -> tuple[CountedEventPlan, ...]:
    """Select root-entry events representable by the counted-event emitter.

    Nested consumers keep their program-point lowering. Excluding one use does
    not discard independent uses of the same semantic event. A one-key event is
    whole-family completion and remains on the aggregated root-completion path
    unless it owns a local trigger. Unsupported relations monotonically fall
    back to root completion during coverage selection.
    """
    local_uses = {
        (trigger.event_index, trigger.use_index) for trigger in local_triggers
    }
    selected: list[CountedEventPlan] = []
    for event in event_graph.events:
        if (
            event.family_done_root is not None
            or not event.key_count
            or any(
                not _counted_contribution_is_lowerable(contribution)
                for contribution in event.contributions
            )
        ):
            continue
        retained_uses: list[EventUse] = []
        selected_local_uses: list[int] = []
        for use_index, use in enumerate(event.uses):
            if (
                use.consumer_scope_id is not None
                or use.keys.canonical_single_valued() is None
            ):
                continue
            remaining = use.dependency_points - excluded_dependency_points
            is_local = (event.event_id, use_index) in local_uses
            if not is_local and not remaining:
                continue
            if is_local:
                selected_local_uses.append(len(retained_uses))
            retained_uses.append(
                EventUse(
                    consumer_root=use.consumer_root,
                    keys=use.keys,
                    dependency_points=(
                        use.dependency_points if is_local else frozenset(remaining)
                    ),
                )
            )
        if len(selected_local_uses) > 1:
            raise ValueError("one counted event cannot have multiple local executors")
        local_trigger_use = selected_local_uses[0] if selected_local_uses else None
        if not retained_uses or (local_trigger_use is None and event.key_count <= 1):
            continue
        selected.append(
            CountedEventPlan(
                contributions=event.contributions,
                uses=tuple(retained_uses),
                local_trigger_use=local_trigger_use,
            )
        )
    selected_local_count = sum(
        event.local_trigger_use is not None for event in selected
    )
    if selected_local_count != len(local_uses):
        raise AssertionError("not every selected local trigger has a lowering event")
    return tuple(selected)


def build_keyed_events(
    dependency_graph: TileDependencyGraph,
    *,
    root_domains: tuple[LogicalDomain, ...],
    scope_domains: tuple[LogicalDomain | None, ...],
    publishable_scope_ids: frozenset[int] | None = None,
) -> tuple[KeyedEvent, ...]:
    """Build the canonical symbolic event graph from memory dependencies.

    This is the sole event-construction path. It never constructs a per-task
    predecessor set. Unsupported relations coarsen to one family-completion
    event for the affected root pair.
    """
    symbolic_dependencies = instantiate_symbolic_dependencies(
        dependency_graph,
        root_domains=root_domains,
        scope_domains=scope_domains,
    )
    scope_by_id = {scope.scope_id: scope for scope in dependency_graph.execution_scopes}
    exact_dependencies = tuple(
        dependency
        for dependency in symbolic_dependencies
        if dependency.relation is not None
        and dependency.relation.source_axes_used() is not None
    )
    all_dependency_points_by_pair: dict[tuple[int, int], set[DependencyPoint]] = {}
    for edge in dependency_graph.edges:
        pair = (edge.producer_root, edge.consumer_root)
        for access_dependency in edge.access_dependencies:
            all_dependency_points_by_pair.setdefault(pair, set()).update(
                dependency_graph.dependency_points(access_dependency)
            )

    implied_points: dict[DependencyPoint, set[DependencyPoint]] = {}
    for source in exact_dependencies:
        source_scope_id = source.consumer_scope_id
        if source_scope_id is None or scope_by_id[source_scope_id].is_root:
            continue
        source_relation = source.relation
        assert source_relation is not None
        source_point = (
            source.dependency_id,
            source.producer_scope_id,
            source_scope_id,
        )
        for later in exact_dependencies:
            later_scope_id = later.consumer_scope_id
            later_relation = later.relation
            if (
                later is source
                or later_scope_id is None
                or later_relation is None
                or source.consumer_root != later.consumer_root
                or source.producer_root != later.producer_root
                or source.producer_scope_id != later.producer_scope_id
                or source_relation.target_domain != later_relation.target_domain
            ):
                continue
            preceding = preceding_scope_relation(
                dependency_graph,
                scope_domains=scope_domains,
                source_scope_id=source_scope_id,
                consumer_scope_id=later_scope_id,
                consumer_access_id=later.consumer_access_id,
            )
            acquired = None if preceding is None else preceding.then(source_relation)
            if acquired is not None and acquired.covers(later_relation):
                implied_points.setdefault(source_point, set()).add(
                    (
                        later.dependency_id,
                        later.producer_scope_id,
                        later_scope_id,
                    )
                )

    exact_relations: dict[
        tuple[int, int | None, LogicalDomain],
        dict[
            tuple[int, int | None, LogicalDomain],
            list[tuple[LogicalRelation, DependencyPoint]],
        ],
    ] = {}

    def add_exact_relation(
        *,
        producer_root: int,
        producer_scope_id: int | None,
        consumer_root: int,
        consumer_scope_id: int | None,
        relation: LogicalRelation,
        dependency_points: frozenset[DependencyPoint],
    ) -> None:
        consumer = (consumer_root, consumer_scope_id, relation.source_domain)
        producer = (producer_root, producer_scope_id, relation.target_domain)
        exact_relations.setdefault(consumer, {}).setdefault(producer, []).extend(
            (relation, dependency_point) for dependency_point in dependency_points
        )

    for dependency in exact_dependencies:
        relation = dependency.relation
        assert relation is not None
        dependency_point = (
            dependency.dependency_id,
            dependency.producer_scope_id,
            dependency.consumer_scope_id,
        )
        exact_points = frozenset(
            (dependency_point, *implied_points.get(dependency_point, ()))
        )
        producer_scope = (
            None
            if dependency.producer_scope_id is None
            else scope_by_id[dependency.producer_scope_id]
        )
        consumer_scope = (
            None
            if dependency.consumer_scope_id is None
            else scope_by_id[dependency.consumer_scope_id]
        )
        producer_is_root = producer_scope is None or producer_scope.is_root
        consumer_is_root = consumer_scope is None or consumer_scope.is_root
        producer_scope_is_usable = producer_is_root or (
            producer_scope is not None
            and producer_scope.segmentable
            and (
                publishable_scope_ids is None
                or dependency.producer_scope_id in publishable_scope_ids
            )
        )
        consumer_scope_is_usable = consumer_is_root or (
            consumer_scope is not None and consumer_scope.segmentable
        )
        if producer_scope_is_usable and consumer_scope_is_usable:
            add_exact_relation(
                producer_root=dependency.producer_root,
                producer_scope_id=(
                    None if producer_is_root else dependency.producer_scope_id
                ),
                consumer_root=dependency.consumer_root,
                consumer_scope_id=(
                    None if consumer_is_root else dependency.consumer_scope_id
                ),
                relation=relation,
                dependency_points=exact_points,
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
            producer_scope_id=None,
            consumer_root=dependency.consumer_root,
            consumer_scope_id=None,
            relation=root_relation,
            dependency_points=exact_points,
        )

    pending_events: dict[
        tuple[LogicalDomain, tuple[EventContribution, ...]],
        KeyedEvent,
    ] = {}
    represented_dependency_points: set[DependencyPoint] = set()

    def record_event_candidate(
        *,
        key_domain: LogicalDomain,
        contributions: tuple[EventContribution, ...],
        uses: tuple[EventUse, ...],
    ) -> None:
        _add_event_candidate(
            pending_events,
            key_domain=key_domain,
            contributions=contributions,
            uses=uses,
        )
        for use in uses:
            represented_dependency_points.update(use.dependency_points)

    def add_producer_keyed_events(
        *,
        consumer_root: int,
        consumer_scope_id: int | None,
        relations: list[
            tuple[
                tuple[int, int | None, LogicalDomain],
                LogicalRelation,
                frozenset[DependencyPoint],
            ]
        ],
    ) -> None:
        """Keep a finer exact key when a consumer quotient needs fanout."""
        for producer, relation, dependency_points in relations:
            producer_root, producer_scope_id, producer_domain = producer
            key_domain = dataclasses.replace(
                producer_domain,
                kind="event",
                identity=None,
            )
            use_relation = relation.retarget(key_domain)
            if use_relation is None:
                raise AssertionError("producer-key event geometry must match")
            record_event_candidate(
                key_domain=key_domain,
                contributions=(
                    EventContribution(
                        producer_root=producer_root,
                        producer_scope_id=producer_scope_id,
                        predecessors=LogicalRelation.identity(
                            key_domain,
                            producer_domain,
                        ),
                    ),
                ),
                uses=(
                    EventUse(
                        consumer_root=consumer_root,
                        consumer_scope_id=consumer_scope_id,
                        keys=use_relation,
                        dependency_points=dependency_points,
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
        consumer_root, consumer_scope_id, consumer_domain = consumer
        merged_relations: list[
            tuple[
                tuple[int, int | None, LogicalDomain],
                LogicalRelation,
                frozenset[DependencyPoint],
            ]
        ] = []
        key_axes: set[int] = set()
        quotient_is_supported = True
        for producer, relation_points in sorted(
            producers.items(),
            key=lambda item: (
                item[0][0],
                -1 if item[0][1] is None else item[0][1],
            ),
        ):
            relation, first_point = relation_points[0]
            dependency_points = {first_point}
            for next_relation, dependency_point in relation_points[1:]:
                union = relation.union(next_relation)
                if union is None:
                    quotient_is_supported = False
                    break
                relation = union
                dependency_points.add(dependency_point)
            if not quotient_is_supported:
                break
            used_axes = relation.source_axes_used()
            if used_axes is None:
                quotient_is_supported = False
                break
            key_axes.update(used_axes)
            merged_relations.append((producer, relation, frozenset(dependency_points)))

        if not quotient_is_supported:
            add_producer_keyed_events(
                consumer_root=consumer_root,
                consumer_scope_id=consumer_scope_id,
                relations=[
                    (producer, relation, frozenset((dependency_point,)))
                    for producer, relation_points in sorted(
                        producers.items(),
                        key=lambda item: (
                            item[0][0],
                            -1 if item[0][1] is None else item[0][1],
                        ),
                    )
                    for relation, dependency_point in relation_points
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
            # producer scope. These are alternative synchronization points,
            # not independent arrivals to one joined event.
            add_producer_keyed_events(
                consumer_root=consumer_root,
                consumer_scope_id=consumer_scope_id,
                relations=merged_relations,
            )
            continue

        ordered_key_axes = tuple(
            axis for axis in consumer_domain.axis_order if axis in key_axes
        )
        consumer_counts = consumer_domain.axis_counts
        consumer_blocks = consumer_domain.block_sizes
        key_domain = LogicalDomain(
            axis_order=ordered_key_axes,
            axis_counts_items=tuple(
                (axis, consumer_counts[axis]) for axis in ordered_key_axes
            ),
            block_sizes_items=tuple(
                (axis, consumer_blocks[axis])
                for axis in ordered_key_axes
                if axis in consumer_blocks
            ),
            kind="event",
        )
        use_relation = LogicalRelation.projection(consumer_domain, key_domain)
        if use_relation is None:
            add_producer_keyed_events(
                consumer_root=consumer_root,
                consumer_scope_id=consumer_scope_id,
                relations=merged_relations,
            )
            continue
        contributions: list[EventContribution] = []
        dependency_points: set[DependencyPoint] = set()
        for producer, relation, relation_points in merged_relations:
            producer_root, producer_scope_id, _producer_domain = producer
            predecessors = relation.factor_through(use_relation)
            if predecessors is None:
                break
            contributions.append(
                EventContribution(
                    producer_root=producer_root,
                    producer_scope_id=producer_scope_id,
                    predecessors=predecessors,
                )
            )
            dependency_points.update(relation_points)
        else:
            if all(
                _counted_contribution_is_lowerable(contribution)
                for contribution in contributions
            ):
                record_event_candidate(
                    key_domain=key_domain,
                    contributions=tuple(contributions),
                    uses=(
                        EventUse(
                            consumer_root=consumer_root,
                            consumer_scope_id=consumer_scope_id,
                            keys=use_relation,
                            dependency_points=frozenset(dependency_points),
                        ),
                    ),
                )
            else:
                add_producer_keyed_events(
                    consumer_root=consumer_root,
                    consumer_scope_id=consumer_scope_id,
                    relations=merged_relations,
                )
            continue

        if len(contributions) != len(merged_relations):
            add_producer_keyed_events(
                consumer_root=consumer_root,
                consumer_scope_id=consumer_scope_id,
                relations=merged_relations,
            )
            continue

    failed_consumers_by_producer: dict[int, dict[int, set[DependencyPoint]]] = {}
    for (
        producer_root,
        consumer_root,
    ), dependency_points in all_dependency_points_by_pair.items():
        remaining_points = dependency_points - represented_dependency_points
        if not remaining_points:
            continue
        failed_consumers_by_producer.setdefault(producer_root, {})[consumer_root] = (
            remaining_points
        )
    for producer_root, consumer_points in sorted(failed_consumers_by_producer.items()):
        key_domain = LogicalDomain(
            axis_order=(),
            axis_counts_items=(),
            kind="event",
        )
        producer_domain = root_domains[producer_root]
        uses: list[EventUse] = []
        for consumer_root, dependency_points in sorted(consumer_points.items()):
            uses.append(
                EventUse(
                    consumer_root=consumer_root,
                    consumer_scope_id=None,
                    keys=LogicalRelation.total(
                        root_domains[consumer_root],
                        key_domain,
                    ),
                    dependency_points=frozenset(dependency_points),
                )
            )
        _add_event_candidate(
            pending_events,
            key_domain=key_domain,
            contributions=(
                EventContribution(
                    producer_root=producer_root,
                    producer_scope_id=None,
                    predecessors=LogicalRelation.total(
                        key_domain,
                        producer_domain,
                    ),
                ),
            ),
            uses=tuple(uses),
        )
    return tuple(pending_events.values())


def build_event_graph(
    dependency_graph: TileDependencyGraph,
    *,
    root_traversals: tuple[LogicalRelation, ...],
    scope_domains: tuple[LogicalDomain | None, ...],
    publishable_scope_ids: frozenset[int] | None = None,
) -> EventGraph:
    """Bind the symbolic readiness DAG for one selected configuration."""
    root_domains = tuple(traversal.target_domain for traversal in root_traversals)
    events = build_keyed_events(
        dependency_graph,
        root_domains=root_domains,
        scope_domains=scope_domains,
        publishable_scope_ids=publishable_scope_ids,
    )
    return EventGraph(
        root_traversals=root_traversals,
        scope_domains=scope_domains,
        events=events,
    )


def derive_local_triggers(
    event_graph: EventGraph,
    worker_schedule: WorkerSchedule,
) -> tuple[LocalTrigger, ...]:
    """Select complete one-task-per-key uses for final-arrival execution."""
    required_points_by_root: dict[int, set[DependencyPoint]] = {}
    for event in event_graph.events:
        for use in event.uses:
            if use.consumer_scope_id is None:
                required_points_by_root.setdefault(use.consumer_root, set()).update(
                    use.dependency_points
                )

    candidates: list[
        tuple[
            int,
            int,
            int,
            KeyedEvent,
            EventUse,
            tuple[tuple[int, LogicalRelation], ...],
        ]
    ] = []
    for event in event_graph.events:
        if (
            event.family_done_root is not None
            or len(event.uses) != 1
            or any(
                contribution.producer_scope_id is not None
                for contribution in event.contributions
            )
        ):
            continue
        if any(
            not _counted_contribution_is_lowerable(contribution)
            for contribution in event.contributions
        ):
            continue
        use_index = 0
        use = event.uses[use_index]
        if use.consumer_scope_id is not None:
            continue
        fan_in = _uniform_arrivals(event.contributions)
        if not event.key_count or fan_in is None or fan_in <= 0:
            continue
        inverse_use = use.keys.inverse()
        if (
            not use.dependency_points.issuperset(
                required_points_by_root.get(use.consumer_root, ())
            )
            or not use.keys.is_total_function()
            or inverse_use is None
            or not inverse_use.is_total_function()
        ):
            continue
        producer_relations = _merge_relations_by_root(
            tuple(
                (contribution.producer_root, publication)
                for contribution in event.contributions
                if (publication := contribution.producer_to_keys) is not None
            )
        )
        if producer_relations is None or len(producer_relations) != len(
            {item.producer_root for item in event.contributions}
        ):
            continue

        candidates.append(
            (
                use.consumer_root,
                event.event_id,
                use_index,
                event,
                use,
                producer_relations,
            )
        )

    conflicting_candidates: set[tuple[int, int]] = set()
    candidates_by_consumer_root: dict[int, list[tuple[int, int]]] = {}
    for consumer_root, event_id, use_index, *_rest in candidates:
        candidates_by_consumer_root.setdefault(consumer_root, []).append(
            (event_id, use_index)
        )
    for root_candidates in candidates_by_consumer_root.values():
        if len(root_candidates) > 1:
            conflicting_candidates.update(root_candidates)
    candidates_by_producer_root: dict[
        int,
        list[tuple[int, LogicalRelation]],
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
        for root in range(len(event_graph.root_domains))
    }

    result: list[LocalTrigger] = []
    for (
        _consumer_root,
        event_id,
        use_index,
        event,
        use,
        _producer_relations,
    ) in sorted(candidates, key=operator.itemgetter(slice(3))):
        if (event_id, use_index) in conflicting_candidates:
            continue
        possible_workers = frozenset(
            worker
            for contribution in event.contributions
            for worker in possible_workers_by_root[contribution.producer_root]
        )
        if not possible_workers:
            continue
        possible_workers_by_root[use.consumer_root] = possible_workers
        result.append(
            LocalTrigger(
                event_index=event_id,
                use_index=use_index,
            )
        )
    return tuple(result)


def order_local_contributors_by_key(
    event_graph: EventGraph,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...],
) -> WorkerSchedule:
    """Order eligible static contributors by ready key.

    Key-major ordering completes one event key at a time so final-arrival work
    becomes ready as early as possible. It is legal only when one contribution
    compactly enumerates a complete static task family; all other families keep
    their existing traversal.
    """
    local_roots = {
        event_graph.event(trigger.event_index).uses[trigger.use_index].consumer_root
        for trigger in local_triggers
    }
    replacement_by_root: dict[int, tuple[WorkerScheduleSegment, ...]] = {}

    for trigger in local_triggers:
        event = event_graph.event(trigger.event_index)
        if len(event.contributions) != 1:
            continue
        contribution = event.contributions[0]
        if contribution.producer_scope_id is not None:
            continue
        root = contribution.producer_root
        if root in local_roots or root in replacement_by_root:
            continue
        task_domain = event_graph.root_domains[root]
        task_relation = contribution.predecessors.fiber_enumeration()
        if (
            task_relation is None
            or task_relation.target_domain != task_domain
            or task_relation.source_domain.size != task_domain.size
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
                task_relation=task_relation,
                worker_begin=0,
                worker_count=worker_schedule.worker_count,
                schedule_begin=schedule_interval[0],
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


def build_cross_loop_schedule(
    *,
    dependency_plan: TileDependencyGraph,
    root_traversals: tuple[LogicalRelation, ...],
    scope_domains: tuple[LogicalDomain | None, ...],
    worker_count: int,
    publishable_scope_ids: frozenset[int] | None = None,
) -> CrossLoopSchedule:
    """Derive all generic readiness strategies without inspecting root bodies."""
    event_graph = build_event_graph(
        dependency_plan,
        root_traversals=root_traversals,
        scope_domains=scope_domains,
        publishable_scope_ids=publishable_scope_ids,
    )
    try:
        (
            worker_schedule,
            local_triggers,
            nested_scope_events,
            event_graph,
        ) = build_worker_schedule(
            event_graph,
            worker_count=worker_count,
        )
    except ValueError as error:
        raise exc.InvalidConfig(
            f"the num_sm_multiplier grid of {worker_count} workers does not "
            "admit a progress-safe cross-loop schedule"
        ) from error

    nested_scope_dependency_points = frozenset(
        dependency_point
        for plan in nested_scope_events
        for use in plan.uses
        for dependency_point in use.dependency_points
    )
    counted_events = (
        *choose_counted_events(
            event_graph,
            local_triggers,
            excluded_dependency_points=nested_scope_dependency_points,
        ),
        *nested_scope_events,
    )
    covered_dependency_points = frozenset(
        dependency_point
        for event in counted_events
        for use in event.uses
        for dependency_point in use.dependency_points
    )
    # Recompute coverage from the mechanisms that will actually be emitted.
    # Dependency analysis may prove a finer relation than the selected emitter
    # can materialize. Such a relation must monotonically coarsen to root
    # completion; retaining a task-ready classification without an emitter
    # would remove the dependency entirely.
    root_completion_edges = _select_root_completion_edges(
        dependency_graph=dependency_plan,
        covered_dependency_points=covered_dependency_points,
    )
    root_order_edges = set(root_completion_edges)
    retained_counted_events: list[CountedEventPlan] = []
    for event in counted_events:
        retained_use_indices = tuple(
            use_index
            for use_index, use in enumerate(event.uses)
            if use_index == event.local_trigger_use
            or not all(
                _is_ordered_by_root_completion(
                    contributor.producer_root,
                    use.consumer_root,
                    root_order_edges,
                )
                for contributor in event.contributions
            )
        )
        if not retained_use_indices:
            continue
        retained_counted_events.append(
            dataclasses.replace(
                event,
                uses=tuple(event.uses[index] for index in retained_use_indices),
                local_trigger_use=(
                    retained_use_indices.index(event.local_trigger_use)
                    if event.local_trigger_use is not None
                    else None
                ),
            )
        )
    counted_events = tuple(retained_counted_events)
    covered_dependency_points = frozenset(
        dependency_point
        for event in counted_events
        for use in event.uses
        for dependency_point in use.dependency_points
    )
    _validate_schedule_coverage(
        dependency_graph=dependency_plan,
        covered_dependency_points=covered_dependency_points,
        root_completion_edges=root_completion_edges,
    )
    return CrossLoopSchedule(
        worker_schedule=worker_schedule,
        counted_events=counted_events,
        root_completion_edges=root_completion_edges,
    )


def _validate_schedule_coverage(
    *,
    dependency_graph: TileDependencyGraph,
    covered_dependency_points: frozenset[DependencyPoint],
    root_completion_edges: frozenset[tuple[int, int]],
) -> None:
    """Verify that every dependence has an emitted synchronization path."""
    root_order_edges = set(root_completion_edges)
    for dependency in dependency_graph.edges:
        pair = (dependency.producer_root, dependency.consumer_root)
        if _is_ordered_by_root_completion(*pair, root_order_edges):
            continue
        uncovered = tuple(
            dependency_point
            for access_dependency in dependency.access_dependencies
            for dependency_point in dependency_graph.dependency_points(
                access_dependency
            )
            if dependency_point not in covered_dependency_points
        )
        if not uncovered:
            continue
        raise exc.CrossLoopSchedulingError(
            f"{dependency.producer_root}->{dependency.consumer_root} through "
            f"allocations {sorted(dependency.tensor_names)!r} has no cross-loop "
            f"synchronization path for dependencies {uncovered!r}"
        )


def _select_root_completion_edges(
    *,
    dependency_graph: TileDependencyGraph,
    covered_dependency_points: frozenset[DependencyPoint],
) -> frozenset[tuple[int, int]]:
    """Choose the minimal source-ordered root-completion fallback edges."""
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
            dependency_graph.dependency_points(access_dependency)
            <= covered_dependency_points
            for access_dependency in dependency.access_dependencies
        ):
            continue
        if _is_ordered_by_root_completion(*pair, ordered_root_edges):
            continue
        selected_edges.add(pair)
        ordered_root_edges.add(pair)
    return frozenset(selected_edges)


def _is_ordered_by_root_completion(
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
