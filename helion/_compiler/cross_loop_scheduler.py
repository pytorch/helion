from __future__ import annotations

import dataclasses
from functools import cached_property
import itertools
import math
import operator

import sympy

from .. import exc
from .tile_dependency import DependencyPoint
from .tile_dependency import LogicalDomain
from .tile_dependency import LogicalRelation
from .tile_dependency import TileDependencyGraph
from .tile_dependency import instantiate_root_domains
from .tile_dependency import instantiate_scope_domains
from .tile_dependency import instantiate_symbolic_dependencies
from .tile_dependency import logical_axis_symbol
from .tile_dependency import preceding_scope_relation

CROSS_LOOP_NUM_WORKERS_CONFIG = "cross_loop_num_workers"
CROSS_LOOP_NUM_WORKERS_DEFAULT = 0


@dataclasses.dataclass(frozen=True)
class TaskStreamSegment:
    """One contiguous command range for a logical root task family."""

    root: int
    command_begin: int
    configured_traversal: LogicalRelation
    ordering_partition: LogicalRelation | None = None
    task_traversal: LogicalRelation = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if self.root < 0:
            raise ValueError(f"root must be nonnegative, got {self.root}")
        if self.command_begin < 0:
            raise ValueError(
                f"command_begin must be nonnegative, got {self.command_begin}"
            )
        traversal = self.configured_traversal
        if self.ordering_partition is not None:
            publication = self.ordering_partition.publication_converse()
            traversal = self.ordering_partition.fiber_enumeration()
            if (
                publication is None
                or not publication.is_total_function()
                or traversal is None
                or not traversal.is_total_function()
                or traversal.source_domain.size
                != self.ordering_partition.target_domain.size
            ):
                raise ValueError(
                    "task stream ordering partition must cover each task once"
                )
            if _same_flat_task_order(traversal, self.configured_traversal):
                traversal = self.configured_traversal
        else:
            inverse = traversal.inverse()
            if inverse is None or not inverse.is_total_function():
                raise ValueError(
                    "configured task traversal must cover each task exactly once"
                )
        if (
            traversal.source_domain.kind != "worker"
            or not traversal.pieces
            or not traversal.is_total_function()
        ):
            raise ValueError("task stream traversal must map every command ordinal")
        object.__setattr__(self, "task_traversal", traversal)

    @property
    def command_count(self) -> int:
        return self.task_traversal.source_domain.size

    @property
    def command_end(self) -> int:
        return self.command_begin + self.command_count


@dataclasses.dataclass(frozen=True)
class TaskStream:
    """A symbolic dependency-safe linearization with no worker placement."""

    segments: tuple[TaskStreamSegment, ...]

    def __post_init__(self) -> None:
        command_begin = 0
        roots: set[int] = set()
        for segment in self.segments:
            if segment.command_begin != command_begin:
                raise ValueError("task stream command ranges must be contiguous")
            if segment.root in roots:
                raise ValueError("one root must occupy one task stream segment")
            roots.add(segment.root)
            command_begin = segment.command_end

    @property
    def command_count(self) -> int:
        return self.segments[-1].command_end if self.segments else 0

    def segment_for_root(self, root: int) -> TaskStreamSegment | None:
        return next(
            (segment for segment in self.segments if segment.root == root), None
        )


def _flat_domain_index_expression(domain: LogicalDomain) -> sympy.Expr:
    """Return the canonical flattened index of a logical coordinate."""
    result: sympy.Expr = sympy.Integer(0)
    multiplier = 1
    for axis in domain.axis_order:
        if domain.axis_counts[axis] > 1:
            result += logical_axis_symbol(axis) * multiplier  # pyrefly: ignore[unsupported-operation]
        multiplier *= domain.axis_counts[axis]
    return sympy.simplify(result)


def _flat_domain_relation(domain: LogicalDomain) -> LogicalRelation:
    """Map a Cartesian domain to its canonical scalar ordinal."""
    value_axis = min((*domain.axis_order, 0)) - 1
    value_domain = LogicalDomain(
        axis_order=(value_axis,),
        axis_counts_items=((value_axis, domain.size),),
        kind="value",
    )
    return LogicalRelation.point_map(
        domain,
        value_domain,
        (
            (
                tuple(
                    (axis, 0, domain.axis_counts[axis], 1) for axis in domain.axis_order
                ),
                (_flat_domain_index_expression(domain),),
            ),
        ),
    )


def _same_flat_task_order(
    left: LogicalRelation,
    right: LogicalRelation,
) -> bool:
    """Return whether two traversals visit the same flattened task sequence."""
    if left == right:
        return True
    if (
        left.target_domain != right.target_domain
        or left.source_domain.size != right.source_domain.size
    ):
        return False

    def ordinal_map(traversal: LogicalRelation) -> LogicalRelation | None:
        source_flat = _flat_domain_relation(traversal.source_domain)
        ordinal_to_source = source_flat.inverse()
        source_to_target = (
            None if ordinal_to_source is None else ordinal_to_source.then(traversal)
        )
        return (
            None
            if source_to_target is None
            else source_to_target.then(_flat_domain_relation(traversal.target_domain))
        )

    def is_identity(relation: LogicalRelation | None) -> bool:
        if (
            relation is None
            or relation.source_domain.size != relation.target_domain.size
            or len(relation.source_domain.axis_order) != 1
            or len(relation.target_domain.axis_order) != 1
            or len(relation.pieces) != 1
        ):
            return False
        source_axis = relation.source_domain.axis_order[0]
        target_axis = relation.target_domain.axis_order[0]
        (piece,) = relation.pieces
        if piece.source_bounds_items != (
            (source_axis, 0, relation.source_domain.size, 1),
        ):
            return False
        if len(piece.target_ranges) != 1:
            return False
        axis, begin, end, step = piece.target_ranges[0]
        return (
            axis == target_axis
            and step == 1
            and sympy.simplify(  # pyrefly: ignore[unsupported-operation]
                begin  # pyrefly: ignore[unsupported-operation]
                - logical_axis_symbol(source_axis)
            )
            == 0
            and sympy.simplify(end - begin - 1)  # pyrefly: ignore[unsupported-operation]
            == 0
        )

    return is_identity(ordinal_map(left)) and is_identity(ordinal_map(right))


@dataclasses.dataclass(frozen=True)
class LocalTrigger:
    """A task use executed by whichever contributor makes the final arrival."""

    event_index: int
    use_index: int


@dataclasses.dataclass(frozen=True)
class EventContribution:
    """A producer execution scope's symbolic contribution to one event."""

    producer_root: int
    predecessors: LogicalRelation
    producer_scope_id: int | None = None

    @cached_property
    def producer_to_keys(self) -> LogicalRelation | None:
        """Return the derived publication relation, when representable."""
        return self.predecessors.publication_converse()

    @cached_property
    def arrivals_per_key(self) -> LogicalRelation | None:
        """Return the exact symbolic number of arrivals for each event key."""
        return self.predecessors.fiber_cardinality()

    @property
    def expected_arrivals(self) -> int:
        count = (
            None
            if self.arrivals_per_key is None
            else self.arrivals_per_key.constant_value()
        )
        if count is None:
            raise ValueError("symbolic contributor has nonuniform fan-in")
        return count


@dataclasses.dataclass(frozen=True)
class EventUse:
    """A consumer execution scope's symbolic requirements from one event."""

    consumer_root: int
    keys: LogicalRelation
    dependency_points_by_contribution: tuple[frozenset[DependencyPoint], ...]
    consumer_scope_id: int | None = None

    @cached_property
    def dependency_points(self) -> frozenset[DependencyPoint]:
        """Return all dependency facts represented by this event use."""
        return frozenset().union(*self.dependency_points_by_contribution)

    def select_contributions(self, indices: tuple[int, ...]) -> EventUse:
        """Restrict provenance to the selected event contributors."""
        return dataclasses.replace(
            self,
            dependency_points_by_contribution=tuple(
                self.dependency_points_by_contribution[index] for index in indices
            ),
        )

    def exclude_dependency_points(
        self,
        excluded: frozenset[DependencyPoint],
    ) -> EventUse:
        """Remove already synchronized facts without losing provenance."""
        return dataclasses.replace(
            self,
            dependency_points_by_contribution=tuple(
                points - excluded for points in self.dependency_points_by_contribution
            ),
        )


@dataclasses.dataclass(frozen=True)
class KeyedEvent:
    """One symbolic readiness event shared by scheduling and lowering."""

    event_id: int
    key_domain: LogicalDomain
    contributions: tuple[EventContribution, ...]
    uses: tuple[EventUse, ...]

    def __post_init__(self) -> None:
        if self.key_domain.kind != "event":
            raise ValueError("event key domain must have event kind")
        if self.key_domain.identity != self.event_id:
            raise ValueError("event key domain identity must match its event ID")
        if self.key_domain.axis_order != tuple(range(len(self.key_domain.axis_order))):
            raise ValueError("event key axes must use canonical local ordinals")
        if self.key_domain.block_sizes_items:
            raise ValueError("event key domains must not inherit scope block sizes")
        if any(
            contribution.predecessors.source_domain != self.key_domain
            for contribution in self.contributions
        ) or any(use.keys.target_domain != self.key_domain for use in self.uses):
            raise ValueError("event relations must use the event key domain")
        if any(
            len(use.dependency_points_by_contribution) != len(self.contributions)
            for use in self.uses
        ):
            raise ValueError("event-use provenance must align with contributors")

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

    @property
    def is_family_done(self) -> bool:
        return self.family_done_root is not None


@dataclasses.dataclass(frozen=True)
class EventGraph:
    """Configured symbolic readiness DAG and its execution-scope domains."""

    root_domains: tuple[LogicalDomain, ...]
    root_traversals: tuple[LogicalRelation, ...]
    scope_domains: tuple[LogicalDomain | None, ...]
    events: tuple[KeyedEvent, ...]

    def __post_init__(self) -> None:
        if len(self.root_domains) != len(self.root_traversals):
            raise ValueError("event graph root domains must match traversals")
        for domain, traversal in zip(
            self.root_domains,
            self.root_traversals,
            strict=True,
        ):
            if (
                traversal.target_domain != domain
                or traversal.source_domain.size != domain.size
                or traversal.source_domain.kind != "worker"
                or not traversal.pieces
            ):
                raise ValueError(
                    "each root traversal must have compatible typed domains"
                )
        if tuple(event.event_id for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("event IDs must be dense and source ordered")

    def event(self, event_id: int) -> KeyedEvent:
        return self.events[event_id]

    def events_contributed_by(self, root: int) -> tuple[KeyedEvent, ...]:
        return tuple(
            event
            for event in self.events
            if any(
                contribution.producer_root == root
                for contribution in event.contributions
            )
        )

    def uses_for_root(self, root: int) -> tuple[EventUse, ...]:
        return tuple(
            use
            for event in self.events
            for use in event.uses
            if use.consumer_root == root
        )

    def scope_domain(self, scope_id: int) -> LogicalDomain:
        domain = self.scope_domains[scope_id]
        if domain is None:
            raise ValueError(f"execution scope {scope_id} has no configured domain")
        return domain

    def uniform_expected_arrivals(self, event: KeyedEvent) -> int | None:
        """Return constant fan-in without enumerating event keys."""
        total = 0
        for contribution in event.contributions:
            cardinality = contribution.arrivals_per_key
            count = None if cardinality is None else cardinality.constant_value()
            if count is None:
                return None
            total += count
        return total


def _task_stream_root_order(event_graph: EventGraph) -> tuple[int, ...]:
    """Return the stable root-level topological order of the event DAG."""
    root_count = len(event_graph.root_domains)
    successors = [set() for _ in range(root_count)]
    indegrees = [0] * root_count
    for event in event_graph.events:
        producers = {contribution.producer_root for contribution in event.contributions}
        consumers = {use.consumer_root for use in event.uses}
        for producer in producers:
            for consumer in consumers:
                if producer == consumer or consumer in successors[producer]:
                    continue
                successors[producer].add(consumer)
                indegrees[consumer] += 1

    ready = [root for root, indegree in enumerate(indegrees) if indegree == 0]
    ordered: list[int] = []
    while ready:
        root = ready.pop(0)
        ordered.append(root)
        for consumer in sorted(successors[root]):
            indegrees[consumer] -= 1
            if indegrees[consumer] == 0:
                ready.append(consumer)
        ready.sort()
    if len(ordered) != root_count:
        raise ValueError("event graph contains a root dependency cycle")
    return tuple(ordered)


def _task_stream_root_partition(
    event_graph: EventGraph,
    root: int,
    candidate_plans: tuple[CountedEventPlan, ...] | None = None,
) -> LogicalRelation | None:
    """Choose one exact key-major traversal when the event DAG proves it."""
    domain = event_graph.root_domains[root]
    candidates: list[tuple[LogicalRelation, LogicalRelation]] = []
    event_contributions = (
        tuple((event.key_count, event.contributions) for event in event_graph.events)
        if candidate_plans is None
        else tuple((plan.key_count, plan.contributors) for plan in candidate_plans)
    )
    for key_count, contributions in event_contributions:
        if key_count <= 1:
            continue
        for contribution in contributions:
            if (
                contribution.producer_root != root
                or contribution.producer_scope_id is not None
            ):
                continue
            partition = contribution.predecessors
            publication = partition.publication_converse()
            traversal = partition.fiber_enumeration()
            if (
                traversal is None
                or traversal.target_domain != domain
                or traversal.source_domain.size != domain.size
                or not traversal.is_total_function()
                or publication is None
                or not publication.is_total_function()
                or any(traversal == candidate for candidate, _ in candidates)
            ):
                continue
            candidates.append((traversal, partition))
    if len(candidates) == 1:
        return candidates[0][1]
    return None


def build_task_stream(
    event_graph: EventGraph,
    *,
    excluded_roots: frozenset[int] = frozenset(),
    candidate_plans: tuple[CountedEventPlan, ...] | None = None,
) -> TaskStream:
    """Build a compact dependency-safe stream directly from the event DAG."""
    segments: list[TaskStreamSegment] = []
    command_begin = 0
    for root in _task_stream_root_order(event_graph):
        if root in excluded_roots:
            continue
        segment = TaskStreamSegment(
            root=root,
            command_begin=command_begin,
            configured_traversal=event_graph.root_traversals[root],
            ordering_partition=_task_stream_root_partition(
                event_graph,
                root,
                candidate_plans,
            ),
        )
        segments.append(segment)
        command_begin = segment.command_end
    return TaskStream(segments=tuple(segments))


def validate_task_stream(
    event_graph: EventGraph,
    task_stream: TaskStream,
    local_triggers: tuple[LocalTrigger, ...],
    *,
    excluded_roots: frozenset[int] = frozenset(),
) -> None:
    """Verify that a retained stream is a linear extension of the final DAG."""
    actual_roots = tuple(segment.root for segment in task_stream.segments)
    expected_roots = frozenset(range(len(event_graph.root_domains))) - excluded_roots
    if frozenset(actual_roots) != expected_roots:
        raise ValueError("task stream does not contain every scheduled root")
    rank = {root: index for index, root in enumerate(actual_roots)}
    for segment in task_stream.segments:
        domain = event_graph.root_domains[segment.root]
        if (
            segment.task_traversal.target_domain != domain
            or segment.command_count != domain.size
            or not segment.task_traversal.is_total_function()
        ):
            raise ValueError("task stream traversal does not cover its root")
    for event in event_graph.events:
        consumer_roots = {
            use.consumer_root
            for use in event.uses
            if use.consumer_root not in excluded_roots
        }
        if not consumer_roots:
            continue
        producer_roots = _scheduled_owner_roots(
            event_graph,
            event.contributions,
            local_triggers,
        )
        if producer_roots is None:
            raise ValueError("local-trigger ownership contains a cycle")
        for consumer_root in consumer_roots:
            if any(
                producer_root != consumer_root
                and rank[producer_root] >= rank[consumer_root]
                for producer_root in producer_roots
            ):
                raise ValueError("task stream no longer orders the final event DAG")


def _scheduled_owner_roots(
    event_graph: EventGraph,
    contributions: tuple[EventContribution, ...],
    local_triggers: tuple[LocalTrigger, ...],
) -> frozenset[int] | None:
    """Resolve command owners through local triggers without coordinate maps."""
    local_trigger_by_root = {
        event_graph.event(trigger.event_index)
        .uses[trigger.use_index]
        .consumer_root: trigger
        for trigger in local_triggers
    }

    def expand(root: int, visiting: frozenset[int]) -> frozenset[int] | None:
        trigger = local_trigger_by_root.get(root)
        if trigger is None:
            return frozenset((root,))
        if root in visiting:
            return None
        trigger_event = event_graph.event(trigger.event_index)
        result: set[int] = set()
        for contribution in trigger_event.contributions:
            expanded = expand(
                contribution.producer_root,
                visiting | frozenset((root,)),
            )
            if expanded is None:
                return None
            result.update(expanded)
        return frozenset(result)

    result: set[int] = set()
    for contribution in contributions:
        expanded = expand(contribution.producer_root, frozenset())
        if expanded is None:
            return None
        result.update(expanded)
    return frozenset(result)


def _counted_contribution_is_lowerable(contribution: EventContribution) -> bool:
    """Keep scheduler eligibility identical to counted-event code generation."""
    publication = contribution.producer_to_keys
    return (
        contribution.arrivals_per_key is not None
        and publication is not None
        and publication.canonical_single_valued() is not None
    )


def _scope_is_lowerable(
    scope_id: int | None,
    lowerable_scope_ids: frozenset[int] | None,
) -> bool:
    """Return whether one execution scope has a concrete lowering site."""
    return (
        scope_id is None
        or lowerable_scope_ids is None
        or scope_id in lowerable_scope_ids
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


def _merge_event_use(uses: list[EventUse], use: EventUse) -> None:
    """Merge equivalent consumer uses while preserving contributor provenance."""
    matching_index = next(
        (
            index
            for index, previous in enumerate(uses)
            if previous.consumer_root == use.consumer_root
            and previous.consumer_scope_id == use.consumer_scope_id
            and previous.keys == use.keys
        ),
        None,
    )
    if matching_index is None:
        uses.append(use)
        return
    previous = uses[matching_index]
    if len(previous.dependency_points_by_contribution) != len(
        use.dependency_points_by_contribution
    ):
        raise AssertionError("merged event-use provenance does not align")
    uses[matching_index] = dataclasses.replace(
        previous,
        dependency_points_by_contribution=tuple(
            left | right
            for left, right in zip(
                previous.dependency_points_by_contribution,
                use.dependency_points_by_contribution,
                strict=True,
            )
        ),
    )


def _add_event_candidate(
    pending: dict[
        tuple[LogicalDomain, tuple[EventContribution, ...]],
        list[EventUse],
    ],
    *,
    key_domain: LogicalDomain,
    contributions: tuple[EventContribution, ...],
    uses: tuple[EventUse, ...],
) -> None:
    """Group fanout by the complete producer partition before assigning IDs."""
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
    grouped_uses = pending.setdefault(
        (canonical_domain, canonical_contributions),
        [],
    )
    for use in canonical_uses:
        _merge_event_use(grouped_uses, use)


def _finalize_keyed_events(
    pending: dict[
        tuple[LogicalDomain, tuple[EventContribution, ...]],
        list[EventUse],
    ],
) -> tuple[KeyedEvent, ...]:
    """Assign deterministic IDs after the readiness quotient is complete."""

    events: list[KeyedEvent] = []
    for event_id, ((key_domain, contributions), uses) in enumerate(pending.items()):
        identified_domain = dataclasses.replace(key_domain, identity=event_id)
        events.append(
            KeyedEvent(
                event_id=event_id,
                key_domain=identified_domain,
                contributions=tuple(
                    dataclasses.replace(
                        contribution,
                        predecessors=dataclasses.replace(
                            contribution.predecessors,
                            source_domain=identified_domain,
                        ),
                    )
                    for contribution in contributions
                ),
                uses=tuple(
                    dataclasses.replace(
                        use,
                        keys=dataclasses.replace(
                            use.keys,
                            target_domain=identified_domain,
                        ),
                    )
                    for use in uses
                ),
            )
        )
    return tuple(events)


@dataclasses.dataclass(frozen=True)
class CountedEventPlan:
    """A logical key space receiving contributions from one or more roots.

    Each contributor has an independently proved key-to-predecessor relation.
    The expected count is derived by summing its fibers; the event therefore
    represents both ordinary continuations and generic multi-predecessor joins.
    Consumer uses are independent of event identity. ``local_trigger_use``
    identifies the optional use executed by the final arriving contributor.
    """

    contributors: tuple[EventContribution, ...]
    uses: tuple[EventUse, ...]
    key_domain: LogicalDomain
    local_trigger_use: int | None = None

    def __post_init__(self) -> None:
        if any(
            len(use.dependency_points_by_contribution) != len(self.contributors)
            for use in self.uses
        ):
            raise ValueError("event-plan provenance must align with contributors")

    @property
    def local_use(self) -> EventUse | None:
        if self.local_trigger_use is None:
            return None
        return self.uses[self.local_trigger_use]

    @property
    def key_count(self) -> int:
        """Return the complete event-key domain used by producers or consumers."""
        return self.key_domain.size

    @property
    def expected_arrivals(self) -> int:
        return sum(contributor.expected_arrivals for contributor in self.contributors)

    @property
    def is_single_contributor(self) -> bool:
        return len(self.contributors) == 1

    @property
    def single_contributor(self) -> EventContribution:
        if not self.is_single_contributor:
            raise ValueError("keyed event has multiple contributors")
        return self.contributors[0]

    @property
    def producer_root(self) -> int:
        return self.single_contributor.producer_root

    @property
    def family_done_root(self) -> int | None:
        """Return the completed producer family for a one-key total event."""
        if (
            self.key_count == 1
            and self.is_single_contributor
            and self.single_contributor.producer_scope_id is None
            and self.single_contributor.predecessors.is_total()
        ):
            return self.single_contributor.producer_root
        return None


def _natural_counted_plans_for_use(
    event: KeyedEvent,
    use: EventUse,
    contribution_indices: tuple[int, ...],
) -> tuple[CountedEventPlan, ...]:
    """Refactor exact obligations through their smallest coordinate quotient.

    A semantic event uses the joint quotient needed by all of its producer
    contributions.  That joint key space can make an individual contribution
    look like producer fanout even when the contribution has a smaller,
    directly publishable quotient.  Recover that quotient from the exact
    relation ``consumer -> producer`` without returning to access analysis.
    """
    grouped: dict[
        tuple[int, ...],
        list[tuple[int, EventContribution, LogicalRelation]],
    ] = {}
    for index in contribution_indices:
        contribution = event.contributions[index]
        relation = use.keys.then(contribution.predecessors)
        if relation is None:
            continue
        used_axes = relation.source_axes_used()
        if used_axes is None:
            continue
        grouped.setdefault(used_axes, []).append((index, contribution, relation))

    plans: list[CountedEventPlan] = []
    consumer_domain = use.keys.source_domain
    for used_axes, entries in grouped.items():
        key_domain = LogicalDomain(
            axis_order=used_axes,
            axis_counts_items=tuple(
                (axis, consumer_domain.axis_counts[axis]) for axis in used_axes
            ),
            block_sizes_items=tuple(
                (axis, consumer_domain.block_sizes[axis])
                for axis in used_axes
                if axis in consumer_domain.block_sizes
            ),
            kind="event",
        )
        quotient = LogicalRelation.projection(consumer_domain, key_domain)
        if quotient is None:
            continue
        canonical_domain = _canonical_event_domain(key_domain)
        canonical_use = _canonical_event_use_relation(quotient, canonical_domain)
        selected_indices: list[int] = []
        contributions: list[EventContribution] = []
        producer_key_entries: list[tuple[int, EventContribution, LogicalRelation]] = []
        for index, contribution, relation in entries:
            predecessors = relation.factor_through(quotient)
            if predecessors is None:
                producer_key_entries.append((index, contribution, relation))
                continue
            lowered = dataclasses.replace(
                contribution,
                predecessors=_canonical_event_predecessors(
                    predecessors,
                    canonical_domain,
                ),
            )
            if not _counted_contribution_is_lowerable(lowered):
                producer_key_entries.append((index, contribution, relation))
                continue
            selected_indices.append(index)
            contributions.append(lowered)
        if contributions:
            plans.append(
                CountedEventPlan(
                    contributors=tuple(contributions),
                    uses=(
                        dataclasses.replace(
                            use.select_contributions(tuple(selected_indices)),
                            keys=canonical_use,
                        ),
                    ),
                    key_domain=canonical_domain,
                )
            )
        for index, contribution, relation in producer_key_entries:
            if relation.canonical_single_valued() is None:
                continue
            # A functional dependency relation is itself an exact quotient:
            # use the producer coordinate as K, so publication is identity.
            # Producers outside the relation's image merely publish unused
            # keys; partial consumer support remains guarded by the use map.
            producer_key_domain = dataclasses.replace(
                relation.target_domain,
                kind="event",
                identity=None,
            )
            producer_key_use = relation.retarget(producer_key_domain)
            if producer_key_use is None:
                continue
            producer_key_predecessors = LogicalRelation.identity(
                producer_key_domain,
                relation.target_domain,
            )
            canonical_producer_domain = _canonical_event_domain(producer_key_domain)
            plans.append(
                CountedEventPlan(
                    contributors=(
                        dataclasses.replace(
                            contribution,
                            predecessors=_canonical_event_predecessors(
                                producer_key_predecessors,
                                canonical_producer_domain,
                            ),
                        ),
                    ),
                    uses=(
                        dataclasses.replace(
                            use.select_contributions((index,)),
                            keys=_canonical_event_use_relation(
                                producer_key_use,
                                canonical_producer_domain,
                            ),
                        ),
                    ),
                    key_domain=canonical_producer_domain,
                )
            )
    return tuple(plans)


def _coalesce_counted_event_plans(
    plans: tuple[CountedEventPlan, ...],
) -> tuple[CountedEventPlan, ...]:
    """Share one counter/publication plan across equivalent fanout uses."""
    grouped: dict[
        tuple[LogicalDomain, tuple[EventContribution, ...]],
        list[EventUse],
    ] = {}
    for plan in plans:
        if plan.local_trigger_use is not None:
            raise ValueError("neutral event plans cannot contain local triggers")
        uses = grouped.setdefault((plan.key_domain, plan.contributors), [])
        for use in plan.uses:
            _merge_event_use(uses, use)
    return tuple(
        CountedEventPlan(
            contributors=contributors,
            uses=tuple(uses),
            key_domain=key_domain,
        )
        for (key_domain, contributors), uses in grouped.items()
    )


def derive_counted_event_plans(
    event_graph: EventGraph,
    *,
    lowerable_scope_ids: frozenset[int] | None = None,
) -> tuple[CountedEventPlan, ...]:
    """Derive emitter-ready plans from the semantic event graph.

    Fully lowerable joint events are preserved.  Mixed events are decomposed
    per use and re-factored through each contribution's natural consumer
    quotient.  Both paths therefore remain views of the same proved relation.
    """
    plans: list[CountedEventPlan] = []
    for event in event_graph.events:
        eligible_contribution_indices = tuple(
            index
            for index, contribution in enumerate(event.contributions)
            if _scope_is_lowerable(
                contribution.producer_scope_id,
                lowerable_scope_ids,
            )
        )
        retained_uses = tuple(
            (use_index, use)
            for use_index, use in enumerate(event.uses)
            if _scope_is_lowerable(use.consumer_scope_id, lowerable_scope_ids)
        )
        if not eligible_contribution_indices or not retained_uses:
            continue
        joint_contributions_are_lowerable = len(eligible_contribution_indices) == len(
            event.contributions
        ) and all(
            _counted_contribution_is_lowerable(contribution)
            for contribution in event.contributions
        )
        joint_uses = tuple(
            (use_index, use)
            for use_index, use in retained_uses
            if joint_contributions_are_lowerable
            and use.keys.canonical_single_valued() is not None
        )
        if joint_uses:
            plans.append(
                CountedEventPlan(
                    contributors=event.contributions,
                    uses=tuple(use for _use_index, use in joint_uses),
                    key_domain=event.key_domain,
                )
            )
        joint_use_indices = {use_index for use_index, _use in joint_uses}
        for use_index, use in retained_uses:
            if use_index in joint_use_indices:
                continue
            plans.extend(
                _natural_counted_plans_for_use(
                    event,
                    use,
                    eligible_contribution_indices,
                )
            )
    return _coalesce_counted_event_plans(tuple(plans))


def _with_local_triggers(
    event_graph: EventGraph,
    plans: tuple[CountedEventPlan, ...],
    local_triggers: tuple[LocalTrigger, ...],
) -> tuple[CountedEventPlan, ...]:
    """Attach semantic final-arrival choices to their unchanged joint plans."""
    result = list(plans)
    for trigger in local_triggers:
        event = event_graph.event(trigger.event_index)
        use = event.uses[trigger.use_index]
        matches = tuple(
            (plan_index, use_index)
            for plan_index, plan in enumerate(result)
            if plan.key_domain == event.key_domain
            and plan.contributors == event.contributions
            for use_index, candidate_use in enumerate(plan.uses)
            if candidate_use == use
        )
        if len(matches) != 1:
            raise ValueError("local trigger has no unique complete event plan")
        plan_index, use_index = matches[0]
        if result[plan_index].local_trigger_use is not None:
            raise ValueError("one counted event cannot have multiple local executors")
        result[plan_index] = dataclasses.replace(
            result[plan_index],
            local_trigger_use=use_index,
        )
    return tuple(result)


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


def _scheduled_predecessors(
    event_graph: EventGraph,
    contributions: tuple[EventContribution, ...],
    local_triggers: tuple[LocalTrigger, ...],
) -> tuple[tuple[int, LogicalRelation], ...] | None:
    """Contract on-ready roots while preserving key-to-task predecessors."""
    local_trigger_by_root = {
        event_graph.event(trigger.event_index)
        .uses[trigger.use_index]
        .consumer_root: trigger
        for trigger in local_triggers
    }

    def expand(
        root: int,
        scope_id: int | None,
        predecessors: LogicalRelation,
        visiting: frozenset[int],
    ) -> tuple[tuple[int, LogicalRelation], ...] | None:
        root_domain = event_graph.root_domains[root]
        root_predecessors = (
            predecessors
            if scope_id is None
            else predecessors.project_target(root_domain)
        )
        if root_predecessors is None:
            return None
        trigger = local_trigger_by_root.get(root)
        if trigger is None:
            return ((root, root_predecessors),)
        if scope_id is not None or root in visiting:
            return None
        trigger_event = event_graph.event(trigger.event_index)
        trigger_use = trigger_event.uses[trigger.use_index]
        trigger_keys = root_predecessors.then(trigger_use.keys)
        if trigger_keys is None:
            return None
        result: list[tuple[int, LogicalRelation]] = []
        for contribution in trigger_event.contributions:
            upstream = trigger_keys.then(contribution.predecessors)
            if upstream is None:
                return None
            expanded = expand(
                contribution.producer_root,
                contribution.producer_scope_id,
                upstream,
                visiting | frozenset((root,)),
            )
            if expanded is None:
                return None
            result.extend(expanded)
        return _merge_relations_by_root(tuple(result))

    expanded: list[tuple[int, LogicalRelation]] = []
    for contribution in contributions:
        result = expand(
            contribution.producer_root,
            contribution.producer_scope_id,
            contribution.predecessors,
            frozenset(),
        )
        if result is None:
            return None
        expanded.extend(result)
    return _merge_relations_by_root(tuple(expanded))


def _coarsen_scope_event(
    event_graph: EventGraph,
    plan: CountedEventPlan,
    use: EventUse,
    *,
    nested_axis: int,
    boundaries: tuple[int, ...],
) -> CountedEventPlan | None:
    """Compose one exact nested event with a contiguous action partition."""
    consumer_scope_id = use.consumer_scope_id
    assert consumer_scope_id is not None
    domain = event_graph.scope_domain(consumer_scope_id)
    if len(boundaries) < 2 or boundaries[0] != 0:
        return None
    nested_count = domain.axis_counts[nested_axis]
    if boundaries[-1] != nested_count or any(
        left >= right for left, right in itertools.pairwise(boundaries)
    ):
        return None
    segments = tuple(itertools.pairwise(boundaries))
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
        for contribution in plan.contributors
    )
    predecessor_relations = tuple(
        None if relation is None else relation.inverse()
        for relation in publication_relations
    )
    action_keys = stage_keys.lift_source(domain)
    if action_keys is None:
        return None

    lowered_contributions: list[EventContribution] = []
    for contribution, relation in zip(
        plan.contributors, predecessor_relations, strict=True
    ):
        if relation is None:
            return None
        lowered = EventContribution(
            producer_root=contribution.producer_root,
            producer_scope_id=contribution.producer_scope_id,
            predecessors=relation,
        )
        if not _counted_contribution_is_lowerable(lowered):
            return None
        lowered_contributions.append(lowered)

    return CountedEventPlan(
        contributors=tuple(lowered_contributions),
        uses=(
            dataclasses.replace(
                use,
                consumer_scope_id=use.consumer_scope_id,
                keys=action_keys,
            ),
        ),
        key_domain=key_domain,
    )


def _task_stream_prefix_task_count(
    event_graph: EventGraph,
    plan: CountedEventPlan,
    task_stream: TaskStream,
    local_triggers: tuple[LocalTrigger, ...],
) -> int | None:
    """Prove that stage zero owns an exact prefix of one stream segment."""
    if not plan.key_domain.axis_order:
        return None
    stage_axis = plan.key_domain.axis_order[0]
    if plan.key_domain.axis_counts[stage_axis] <= 1:
        return None
    scheduled = _scheduled_predecessors(
        event_graph,
        plan.contributors,
        local_triggers,
    )
    if scheduled is None or len(scheduled) != 1:
        return None
    producer_root, predecessors = scheduled[0]
    segment = task_stream.segment_for_root(producer_root)
    if segment is None or segment.ordering_partition is None:
        return None

    # Relate the coarsened event keys back to the finer key partition that
    # induced the actual TaskStream traversal.  This proves compatibility even
    # when the two events use different, but nested, readiness granularities.
    ordered_groups = segment.ordering_partition.overlapping_sources(predecessors)
    if ordered_groups is None:
        return None
    stage_domain = LogicalDomain(
        axis_order=(stage_axis,),
        axis_counts_items=((stage_axis, plan.key_domain.axis_counts[stage_axis]),),
        kind="event",
        identity=plan.key_domain.identity,
    )
    # Union outer-key fibers before flattening the producer partition. A fixed
    # outer coordinate may select a strided subset of flattened groups even
    # though the union across that coordinate is one dense stream prefix.
    stage_ordered_groups = ordered_groups.project_source(stage_domain)
    stage_groups = (
        None
        if stage_ordered_groups is None
        else stage_ordered_groups.then(
            _flat_domain_relation(segment.ordering_partition.source_domain)
        )
    )
    stage_predecessors = predecessors.project_source(stage_domain)
    if stage_groups is None or stage_predecessors is None:
        return None

    group_cardinality = stage_groups.fiber_cardinality()
    predecessor_cardinality = stage_predecessors.fiber_cardinality()
    group_maximum = stage_groups.fiber_maximum(
        LogicalRelation.identity(
            stage_groups.target_domain,
            stage_groups.target_domain,
        )
    )
    if (
        group_cardinality is None
        or predecessor_cardinality is None
        or group_maximum is None
    ):
        return None

    fixed_stage = {stage_axis: 0}
    group_count_bounds = group_cardinality.value_bounds(fixed_stage)
    predecessor_count_bounds = predecessor_cardinality.value_bounds(fixed_stage)
    maximum_bounds = group_maximum.value_bounds(fixed_stage)
    if (
        group_count_bounds is None
        or group_count_bounds[0] != group_count_bounds[1]
        or predecessor_count_bounds is None
        or predecessor_count_bounds[0] != predecessor_count_bounds[1]
        or maximum_bounds is None
        or maximum_bounds[0] != maximum_bounds[1]
    ):
        return None
    group_count = group_count_bounds[0]
    predecessor_count = predecessor_count_bounds[0]
    tasks_per_group_relation = segment.ordering_partition.fiber_cardinality()
    tasks_per_group = (
        None
        if tasks_per_group_relation is None
        else tasks_per_group_relation.constant_value()
    )
    if (
        group_count <= 0
        or tasks_per_group is None
        or maximum_bounds[0] != group_count - 1
        or predecessor_count != group_count * tasks_per_group
    ):
        return None
    return predecessor_count


def _nested_plan_consumer_roots(
    dependency_graph: TileDependencyGraph,
    event_graph: EventGraph,
    candidate_plans: tuple[CountedEventPlan, ...],
) -> frozenset[int]:
    """Return roots with at least one actually lowerable nested checkpoint."""
    result: set[int] = set()
    for plan in candidate_plans:
        for use in plan.uses:
            if use.consumer_scope_id is None:
                continue
            scope = dependency_graph.execution_scopes[use.consumer_scope_id]
            if len(scope.local_axis_order) != 1:
                continue
            (nested_axis,) = scope.local_axis_order
            nested_count = event_graph.scope_domain(use.consumer_scope_id).axis_counts[
                nested_axis
            ]
            if (
                _coarsen_scope_event(
                    event_graph,
                    plan,
                    use,
                    nested_axis=nested_axis,
                    boundaries=(0, nested_count),
                )
                is not None
            ):
                result.add(use.consumer_root)
    return frozenset(result)


def choose_task_stream_scope_events(
    dependency_graph: TileDependencyGraph,
    event_graph: EventGraph,
    task_stream: TaskStream,
    local_triggers: tuple[LocalTrigger, ...],
    *,
    resident_wave_size: int,
    lowerable_scope_ids: frozenset[int] | None = None,
    candidate_plans: tuple[CountedEventPlan, ...] | None = None,
) -> tuple[CountedEventPlan, ...]:
    """Derive nested-loop wait partitions without physical worker placement.

    An exact scope-to-event relation is always the correctness source.  When
    its ultimate scheduled producer is traversed key-major, one resident wave
    gives a natural first readiness prefix.  Coarsening at that prefix keeps
    polling outside the hot inner loop while preserving the exact dependency.
    Unsupported shapes conservatively use one whole-range milestone.
    """
    if candidate_plans is None:
        candidate_plans = _with_local_triggers(
            event_graph,
            derive_counted_event_plans(
                event_graph,
                lowerable_scope_ids=lowerable_scope_ids,
            ),
            local_triggers,
        )
    uses_by_consumer: dict[
        int,
        list[tuple[int, CountedEventPlan, EventUse]],
    ] = {}
    for plan_index, plan in enumerate(candidate_plans):
        for use in plan.uses:
            if use.consumer_scope_id is not None:
                uses_by_consumer.setdefault(use.consumer_root, []).append(
                    (plan_index, plan, use)
                )

    plans: list[CountedEventPlan] = []
    for _consumer_root, plan_uses in sorted(uses_by_consumer.items()):
        covered_dependency_points: set[DependencyPoint] = set()
        for _plan_index, source_plan, use in sorted(
            plan_uses,
            key=lambda item: (
                item[2].consumer_scope_id
                if item[2].consumer_scope_id is not None
                else -1,
                item[0],
            ),
        ):
            if use.dependency_points and use.dependency_points <= (
                covered_dependency_points
            ):
                continue
            consumer_scope_id = use.consumer_scope_id
            assert consumer_scope_id is not None
            consumer_scope = dependency_graph.execution_scopes[consumer_scope_id]
            if len(consumer_scope.local_axis_order) != 1:
                continue
            (nested_axis,) = consumer_scope.local_axis_order
            scope_domain = event_graph.scope_domain(consumer_scope_id)
            used_axes = use.keys.source_axes_used()
            if (
                used_axes is None
                or nested_axis not in used_axes
                or not use.keys.is_total_function()
            ):
                continue
            nested_count = scope_domain.axis_counts[nested_axis]
            outer_axes = tuple(axis for axis in used_axes if axis != nested_axis)
            outer_key_count = math.prod(
                scope_domain.axis_counts[axis] for axis in outer_axes
            )
            boundaries = (0, nested_count)

            scheduled_predecessors = _scheduled_predecessors(
                event_graph,
                source_plan.contributors,
                local_triggers,
            )
            if scheduled_predecessors is not None and len(scheduled_predecessors) == 1:
                producer_root, predecessors = scheduled_predecessors[0]
                cardinality = predecessors.fiber_cardinality()
                fan_in = None if cardinality is None else cardinality.constant_value()
                if (
                    fan_in is not None
                    and fan_in > 0
                    and source_plan.key_count == outer_key_count * nested_count
                ):
                    ready_prefix = min(
                        nested_count,
                        resident_wave_size // (outer_key_count * fan_in),
                    )
                    candidate_boundaries = tuple(
                        sorted({0, ready_prefix, nested_count})
                    )
                    if len(candidate_boundaries) > 2:
                        candidate = _coarsen_scope_event(
                            event_graph,
                            source_plan,
                            use,
                            nested_axis=nested_axis,
                            boundaries=candidate_boundaries,
                        )
                        prefix_tasks = (
                            None
                            if candidate is None
                            else _task_stream_prefix_task_count(
                                event_graph,
                                candidate,
                                task_stream,
                                local_triggers,
                            )
                        )
                        if (
                            prefix_tasks is not None
                            and prefix_tasks <= resident_wave_size
                        ):
                            boundaries = candidate_boundaries

            plan = _coarsen_scope_event(
                event_graph,
                source_plan,
                use,
                nested_axis=nested_axis,
                boundaries=boundaries,
            )
            if plan is not None:
                plans.append(plan)
                covered_dependency_points.update(plan.uses[0].dependency_points)
    return tuple(plans)


@dataclasses.dataclass(frozen=True)
class CrossLoopSchedule:
    """Pure graph-derived choices consumed by persistent-kernel lowering."""

    event_graph: EventGraph
    task_stream: TaskStream
    local_triggers: tuple[LocalTrigger, ...]
    counted_events: tuple[CountedEventPlan, ...]
    resident_wave_size: int

    @property
    def root_completion_edges(self) -> frozenset[tuple[int, int]]:
        """Return blocking whole-family fallbacks in the selected schedule."""
        return frozenset(
            (family_done_root, use.consumer_root)
            for plan in self.counted_events
            if plan.local_trigger_use is None
            if (family_done_root := plan.family_done_root) is not None
            for use in plan.uses
            if use.consumer_scope_id is None
        )


def select_root_completion_plans(
    dependency_graph: TileDependencyGraph,
    event_graph: EventGraph,
    edges: frozenset[tuple[int, int]],
) -> tuple[CountedEventPlan, ...]:
    """Coarsen selected semantic dependencies to one-key family completion."""
    plans: list[CountedEventPlan] = []
    consumers_by_producer: dict[int, list[int]] = {}
    for producer_root, consumer_root in sorted(edges):
        consumers_by_producer.setdefault(producer_root, []).append(consumer_root)
    for producer_root, consumer_roots in consumers_by_producer.items():
        key_domain = LogicalDomain(
            axis_order=(),
            axis_counts_items=(),
            kind="event",
        )
        contribution = EventContribution(
            producer_root=producer_root,
            predecessors=LogicalRelation.total(
                key_domain,
                event_graph.root_domains[producer_root],
            ),
        )
        uses = tuple(
            EventUse(
                consumer_root=consumer_root,
                keys=LogicalRelation.total(
                    event_graph.root_domains[consumer_root],
                    key_domain,
                ),
                dependency_points_by_contribution=(
                    frozenset(
                        dependency_point
                        for dependency in dependency_graph.edges_between(
                            producer_root,
                            consumer_root,
                        )
                        for access_dependency in dependency.access_dependencies
                        for dependency_point in dependency_graph.dependency_points(
                            access_dependency
                        )
                    ),
                ),
            )
            for consumer_root in consumer_roots
        )
        plans.append(
            CountedEventPlan(
                contributors=(contribution,),
                uses=uses,
                key_domain=key_domain,
            )
        )
    return tuple(plans)


def choose_local_triggers(
    event_graph: EventGraph,
    *,
    excluded_roots: frozenset[int] = frozenset(),
) -> tuple[LocalTrigger, ...]:
    """Choose final-arrival execution from complete exact task readiness."""
    return tuple(
        trigger
        for trigger in derive_local_triggers(event_graph)
        if event_graph.event(trigger.event_index).uses[trigger.use_index].consumer_root
        not in excluded_roots
        and event_graph.uniform_expected_arrivals(
            event_graph.event(trigger.event_index)
        )
        is not None
    )


def choose_counted_events(
    event_graph: EventGraph,
    local_triggers: tuple[LocalTrigger, ...],
    *,
    lowerable_scope_ids: frozenset[int] | None = None,
    candidate_plans: tuple[CountedEventPlan, ...] | None = None,
) -> tuple[CountedEventPlan, ...]:
    """Select root-entry events representable by the counted-event emitter.

    Nested consumers keep their program-point lowering. Excluding one use does
    not discard independent uses of the same semantic event. Whole-family
    completion is selected later and lowered through the same counted-event
    path. Unsupported relations monotonically coarsen during coverage
    selection.
    """
    if candidate_plans is None:
        candidate_plans = _with_local_triggers(
            event_graph,
            derive_counted_event_plans(
                event_graph,
                lowerable_scope_ids=lowerable_scope_ids,
            ),
            local_triggers,
        )
    selected: list[CountedEventPlan] = []
    for plan in candidate_plans:
        if not plan.key_count:
            continue
        retained_uses: list[EventUse] = []
        selected_local_use: int | None = None
        for use_index, use in enumerate(plan.uses):
            if (
                use.consumer_scope_id is not None
                or use.keys.canonical_single_valued() is None
            ):
                continue
            is_local = use_index == plan.local_trigger_use
            if plan.family_done_root is not None and not is_local:
                continue
            if is_local:
                if selected_local_use is not None:
                    raise ValueError(
                        "one counted event cannot have multiple local executors"
                    )
                selected_local_use = len(retained_uses)
            retained_uses.append(dataclasses.replace(use, consumer_scope_id=None))
        if not retained_uses:
            continue
        selected.append(
            dataclasses.replace(
                plan,
                uses=tuple(retained_uses),
                local_trigger_use=selected_local_use,
            )
        )
    selected_local_count = sum(plan.local_trigger_use is not None for plan in selected)
    if selected_local_count != len(local_triggers):
        raise AssertionError("not every selected local trigger has a lowering event")
    return tuple(selected)


def _exclude_covered_points(
    plan: CountedEventPlan,
    covered_points: frozenset[DependencyPoint],
) -> CountedEventPlan | None:
    """Remove already synchronized obligations from a candidate event plan."""
    filtered_uses: list[tuple[EventUse, bool]] = []
    for use_index, use in enumerate(plan.uses):
        is_local = use_index == plan.local_trigger_use
        filtered = use if is_local else use.exclude_dependency_points(covered_points)
        if not filtered.dependency_points and not is_local:
            continue
        filtered_uses.append((filtered, is_local))
    if not filtered_uses:
        return None

    retained_contribution_indices = tuple(
        index
        for index in range(len(plan.contributors))
        if any(
            use.dependency_points_by_contribution[index]
            for use, _is_local in filtered_uses
        )
    )
    if not retained_contribution_indices:
        return None
    retained_uses = tuple(
        use.select_contributions(retained_contribution_indices)
        for use, _is_local in filtered_uses
    )
    local_trigger_use = next(
        (index for index, (_use, is_local) in enumerate(filtered_uses) if is_local),
        None,
    )
    return dataclasses.replace(
        plan,
        contributors=tuple(
            plan.contributors[index] for index in retained_contribution_indices
        ),
        uses=retained_uses,
        local_trigger_use=local_trigger_use,
    )


def select_event_plans(
    dependency_graph: TileDependencyGraph,
    event_graph: EventGraph,
    task_stream: TaskStream,
    local_triggers: tuple[LocalTrigger, ...],
    *,
    resident_wave_size: int,
    lowerable_scope_ids: frozenset[int] | None = None,
    candidate_plans: tuple[CountedEventPlan, ...] | None = None,
) -> tuple[tuple[CountedEventPlan, ...], frozenset[tuple[int, int]]]:
    """Choose one lowering mechanism for every dependency point.

    The priority is local final-arrival, nested exact/coarsened readiness,
    root keyed readiness, then whole-family completion. The semantic event
    graph is never rewritten; this function only selects executable views.
    """
    if candidate_plans is None:
        candidate_plans = _with_local_triggers(
            event_graph,
            derive_counted_event_plans(
                event_graph,
                lowerable_scope_ids=lowerable_scope_ids,
            ),
            local_triggers,
        )
    nested_plans = choose_task_stream_scope_events(
        dependency_graph,
        event_graph,
        task_stream,
        local_triggers,
        resident_wave_size=resident_wave_size,
        lowerable_scope_ids=lowerable_scope_ids,
        candidate_plans=candidate_plans,
    )
    nested_points = frozenset(
        point
        for plan in nested_plans
        for use in plan.uses
        for point in use.dependency_points
    )
    root_plans = tuple(
        selected
        for plan in choose_counted_events(
            event_graph,
            local_triggers,
            lowerable_scope_ids=lowerable_scope_ids,
            candidate_plans=candidate_plans,
        )
        if (selected := _exclude_covered_points(plan, nested_points)) is not None
    )
    fine_plans = (*root_plans, *nested_plans)
    fine_points = frozenset(
        point
        for plan in fine_plans
        for use in plan.uses
        for point in use.dependency_points
    )
    root_completion_edges = _select_root_completion_edges(
        dependency_graph=dependency_graph,
        covered_dependency_points=fine_points,
    )
    root_order_edges = set(root_completion_edges)

    selected_fine_plans: list[CountedEventPlan] = []
    for plan in fine_plans:
        retained_use_indices = tuple(
            use_index
            for use_index, use in enumerate(plan.uses)
            if use_index == plan.local_trigger_use
            or not all(
                _is_ordered_by_root_completion(
                    contributor.producer_root,
                    use.consumer_root,
                    root_order_edges,
                )
                for contributor in plan.contributors
            )
        )
        if not retained_use_indices:
            continue
        selected_fine_plans.append(
            dataclasses.replace(
                plan,
                uses=tuple(plan.uses[index] for index in retained_use_indices),
                local_trigger_use=(
                    retained_use_indices.index(plan.local_trigger_use)
                    if plan.local_trigger_use is not None
                    else None
                ),
            )
        )

    selected_fine_points = frozenset(
        point
        for plan in selected_fine_plans
        for use in plan.uses
        for point in use.dependency_points
    )
    _validate_schedule_coverage(
        dependency_graph=dependency_graph,
        covered_dependency_points=selected_fine_points,
        root_completion_edges=root_completion_edges,
    )
    return (
        (
            *selected_fine_plans,
            *select_root_completion_plans(
                dependency_graph,
                event_graph,
                root_completion_edges,
            ),
        ),
        root_completion_edges,
    )


def build_keyed_events(
    dependency_graph: TileDependencyGraph,
    *,
    axis_geometry: dict[int, tuple[int, int]],
) -> tuple[KeyedEvent, ...] | None:
    """Build the canonical symbolic event graph from memory dependencies.

    This is the sole event-construction path. It never constructs a per-task
    predecessor set. Unsupported relations coarsen to one family-completion
    event for the affected root pair.
    """
    root_domains = instantiate_root_domains(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    if any(domain is None for domain in root_domains):
        return None
    concrete_root_domains = tuple(
        domain for domain in root_domains if domain is not None
    )
    symbolic_dependencies = instantiate_symbolic_dependencies(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    scope_by_id = {scope.scope_id: scope for scope in dependency_graph.execution_scopes}
    relational_dependencies = tuple(
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

    scope_domains = instantiate_scope_domains(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    implied_points: dict[DependencyPoint, set[DependencyPoint]] = {}
    for source in relational_dependencies:
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
        for later in relational_dependencies:
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

    relations_by_consumer: dict[
        tuple[int, int | None, LogicalDomain],
        dict[
            tuple[int, int | None, LogicalDomain],
            list[tuple[LogicalRelation, DependencyPoint]],
        ],
    ] = {}
    represented_root_points_by_pair: dict[tuple[int, int], set[DependencyPoint]] = {}

    def add_relation(
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
        relations_by_consumer.setdefault(consumer, {}).setdefault(producer, []).extend(
            (relation, dependency_point) for dependency_point in dependency_points
        )

    def mark_root_points(
        *,
        producer_root: int,
        producer_scope_id: int | None,
        consumer_root: int,
        consumer_scope_id: int | None,
        dependency_points: frozenset[DependencyPoint],
    ) -> None:
        if producer_scope_id is None and consumer_scope_id is None:
            represented_root_points_by_pair.setdefault(
                (producer_root, consumer_root), set()
            ).update(dependency_points)

    for dependency in relational_dependencies:
        relation = dependency.relation
        assert relation is not None
        dependency_point = (
            dependency.dependency_id,
            dependency.producer_scope_id,
            dependency.consumer_scope_id,
        )
        represented_points = frozenset(
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
        add_relation(
            producer_root=dependency.producer_root,
            producer_scope_id=(
                None if producer_is_root else dependency.producer_scope_id
            ),
            consumer_root=dependency.consumer_root,
            consumer_scope_id=(
                None if consumer_is_root else dependency.consumer_scope_id
            ),
            relation=relation,
            dependency_points=represented_points,
        )

        if producer_is_root and consumer_is_root:
            continue
        root_relation = relation
        if not consumer_is_root:
            projected = root_relation.project_source(
                concrete_root_domains[dependency.consumer_root]
            )
            if projected is None:
                continue
            root_relation = projected
        if not producer_is_root:
            projected = root_relation.project_target(
                concrete_root_domains[dependency.producer_root]
            )
            if projected is None:
                continue
            root_relation = projected
        add_relation(
            producer_root=dependency.producer_root,
            producer_scope_id=None,
            consumer_root=dependency.consumer_root,
            consumer_scope_id=None,
            relation=root_relation,
            dependency_points=represented_points,
        )

    pending_events: dict[
        tuple[LogicalDomain, tuple[EventContribution, ...]],
        list[EventUse],
    ] = {}

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
            _add_event_candidate(
                pending_events,
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
                        dependency_points_by_contribution=(dependency_points,),
                    ),
                ),
            )
            if use_relation.canonical_single_valued() is not None:
                mark_root_points(
                    producer_root=producer_root,
                    producer_scope_id=producer_scope_id,
                    consumer_root=consumer_root,
                    consumer_scope_id=consumer_scope_id,
                    dependency_points=dependency_points,
                )

    for consumer, producers in sorted(
        relations_by_consumer.items(),
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
        merge_failed = False
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
                    merge_failed = True
                    break
                relation = union
                dependency_points.add(dependency_point)
            if merge_failed:
                break
            used_axes = relation.source_axes_used()
            if used_axes is None:
                merge_failed = True
                break
            key_axes.update(used_axes)
            merged_relations.append((producer, relation, frozenset(dependency_points)))
        if merge_failed:
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
            continue
        contributions: list[EventContribution] = []
        dependency_points_by_contribution: list[frozenset[DependencyPoint]] = []
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
            dependency_points_by_contribution.append(relation_points)
        else:
            _add_event_candidate(
                pending_events,
                key_domain=key_domain,
                contributions=tuple(contributions),
                uses=(
                    EventUse(
                        consumer_root=consumer_root,
                        consumer_scope_id=consumer_scope_id,
                        keys=use_relation,
                        dependency_points_by_contribution=tuple(
                            dependency_points_by_contribution
                        ),
                    ),
                ),
            )
            for (
                producer_root,
                producer_scope_id,
                _producer_domain,
            ), _relation, relation_points in merged_relations:
                mark_root_points(
                    producer_root=producer_root,
                    producer_scope_id=producer_scope_id,
                    consumer_root=consumer_root,
                    consumer_scope_id=consumer_scope_id,
                    dependency_points=relation_points,
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
        remaining_points = dependency_points - represented_root_points_by_pair.get(
            (producer_root, consumer_root), set()
        )
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
        producer_domain = concrete_root_domains[producer_root]
        uses: list[EventUse] = []
        for consumer_root, dependency_points in sorted(consumer_points.items()):
            uses.append(
                EventUse(
                    consumer_root=consumer_root,
                    consumer_scope_id=None,
                    keys=LogicalRelation.total(
                        concrete_root_domains[consumer_root],
                        key_domain,
                    ),
                    dependency_points_by_contribution=(frozenset(dependency_points),),
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
    return _finalize_keyed_events(pending_events)


def build_event_graph(
    dependency_graph: TileDependencyGraph,
    *,
    root_domains: tuple[LogicalDomain, ...],
    root_traversals: tuple[LogicalRelation, ...],
    axis_geometry: dict[int, tuple[int, int]],
) -> EventGraph:
    """Bind the symbolic readiness DAG for one selected configuration."""
    configured_root_domains = instantiate_root_domains(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    if (
        any(domain is None for domain in configured_root_domains)
        or tuple(domain for domain in configured_root_domains if domain is not None)
        != root_domains
    ):
        raise ValueError("configured root domains disagree with the dependency graph")
    events = build_keyed_events(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    if events is None:
        raise ValueError("configured dependency graph has dynamic root domains")
    return EventGraph(
        root_domains=root_domains,
        root_traversals=root_traversals,
        scope_domains=instantiate_scope_domains(
            dependency_graph,
            axis_geometry=axis_geometry,
        ),
        events=events,
    )


def derive_local_triggers(
    event_graph: EventGraph,
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
        if len(event.uses) != 1 or any(
            contribution.producer_scope_id is not None
            for contribution in event.contributions
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
        fan_in = event_graph.uniform_expected_arrivals(event)
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

    result: list[LocalTrigger] = []
    for (
        _consumer_root,
        event_id,
        use_index,
        _event,
        _use,
        _producer_relations,
    ) in sorted(candidates, key=operator.itemgetter(slice(3))):
        if (event_id, use_index) in conflicting_candidates:
            continue
        result.append(
            LocalTrigger(
                event_index=event_id,
                use_index=use_index,
            )
        )
    return tuple(result)


def build_cross_loop_schedule(
    *,
    dependency_plan: TileDependencyGraph,
    root_domains: tuple[LogicalDomain, ...],
    root_traversals: tuple[LogicalRelation, ...],
    axis_geometry: dict[int, tuple[int, int]],
    default_resident_wave_size: int,
    requested_resident_wave_size: int = CROSS_LOOP_NUM_WORKERS_DEFAULT,
    lowerable_scope_ids: frozenset[int] | None = None,
) -> CrossLoopSchedule:
    """Derive all generic readiness strategies without inspecting root bodies."""
    event_graph = build_event_graph(
        dependency_plan,
        root_domains=root_domains,
        root_traversals=root_traversals,
        axis_geometry=axis_geometry,
    )
    semantic_root_edges = frozenset(
        (contribution.producer_root, use.consumer_root)
        for event in event_graph.events
        for contribution in event.contributions
        for use in event.uses
        if contribution.producer_root != use.consumer_root
    )
    if requested_resident_wave_size < 0:
        raise exc.InvalidConfig(
            f"{CROSS_LOOP_NUM_WORKERS_CONFIG} must be nonnegative, got "
            f"{requested_resident_wave_size}"
        )
    resident_wave_size = (
        default_resident_wave_size
        if requested_resident_wave_size == CROSS_LOOP_NUM_WORKERS_DEFAULT
        else requested_resident_wave_size
    )
    try:
        prelocal_plans = derive_counted_event_plans(
            event_graph,
            lowerable_scope_ids=lowerable_scope_ids,
        )
        nested_wait_roots = _nested_plan_consumer_roots(
            dependency_plan,
            event_graph,
            prelocal_plans,
        )
        local_triggers = choose_local_triggers(
            event_graph,
            excluded_roots=nested_wait_roots,
        )
        candidate_plans = _with_local_triggers(
            event_graph,
            prelocal_plans,
            local_triggers,
        )
        locally_executed_roots = frozenset(
            event_graph.event(trigger.event_index).uses[trigger.use_index].consumer_root
            for trigger in local_triggers
        )
        task_stream = build_task_stream(
            event_graph,
            excluded_roots=locally_executed_roots,
            candidate_plans=candidate_plans,
        )
        counted_events, root_completion_edges = select_event_plans(
            dependency_plan,
            event_graph,
            task_stream,
            local_triggers,
            resident_wave_size=resident_wave_size,
            lowerable_scope_ids=lowerable_scope_ids,
            candidate_plans=candidate_plans,
        )
    except ValueError as error:
        raise exc.InvalidConfig(
            f"{CROSS_LOOP_NUM_WORKERS_CONFIG}={requested_resident_wave_size} does not "
            "admit a progress-safe cross-loop schedule"
        ) from error
    if not root_completion_edges <= semantic_root_edges:
        raise AssertionError("root completion introduced a new dependency edge")
    locally_executed_roots = frozenset(
        event_graph.event(trigger.event_index).uses[trigger.use_index].consumer_root
        for trigger in local_triggers
    )
    validate_task_stream(
        event_graph,
        task_stream,
        local_triggers,
        excluded_roots=locally_executed_roots,
    )
    return CrossLoopSchedule(
        event_graph=event_graph,
        task_stream=task_stream,
        local_triggers=local_triggers,
        counted_events=counted_events,
        resident_wave_size=resident_wave_size,
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
