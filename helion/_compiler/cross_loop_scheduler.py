from __future__ import annotations

import contextlib
import dataclasses
from functools import cached_property
import heapq
import itertools
from typing import TYPE_CHECKING
from typing import cast

from .. import exc
from .tile_dependency import CoordinateDomain
from .tile_dependency import CoordinateRelation
from .tile_dependency import DenseTaskOrder
from .tile_dependency import DependencyObligation
from .tile_dependency import Incidence
from .tile_dependency import KeyPartition
from .tile_dependency import TileDependencyGraph
from .tile_dependency import consumer_to_preceding_site_relation
from .tile_dependency import coordinate_axis_symbol
from .tile_dependency import instantiate_symbolic_dependencies
from .tile_dependency import nested_logical_axes

if TYPE_CHECKING:
    from collections.abc import Callable

    _StaticProducerResolver = Callable[
        [tuple["ReadinessProducer", ...]], tuple[tuple[int, Incidence], ...] | None
    ]

ObligationsByRootPair = tuple[
    tuple[tuple[int, int], frozenset[DependencyObligation]], ...
]
_MAX_SYMBOLIC_RELATION_WORK = 2_000_000


def _new_relation_work_budget() -> Callable[[int], bool]:
    remaining = _MAX_SYMBOLIC_RELATION_WORK

    def charge(amount: int) -> bool:
        nonlocal remaining
        remaining -= amount
        return remaining >= 0

    return charge


@dataclasses.dataclass(frozen=True)
class FinalArrivalContinuation:
    """A consumer task executed by whichever producer makes the final arrival."""

    event_id: int
    consumer_index: int


@dataclasses.dataclass(frozen=True)
class ReadinessProducer:
    """One producer execution site's requirements for a readiness event."""

    producer_root: int
    incidence: Incidence
    producer_site_id: int | None = None


def _covered_obligations(
    counters: tuple[ReadinessCounterPlan, ...],
) -> frozenset[DependencyObligation]:
    return frozenset(
        obligation
        for plan in counters
        for consumer in plan.consumers
        for obligation in consumer.covered_obligations
    )


def _arrival_count_bounds(
    producers: tuple[ReadinessProducer, ...],
) -> tuple[int, int] | None:
    minimum = 0
    maximum = 0
    for readiness_producer in producers:
        cardinality = readiness_producer.incidence.count_by_key
        if cardinality is None:
            return None
        bounds = cardinality.value_bounds()
        if bounds is None:
            return None
        minimum += bounds[0]
        maximum += bounds[1]
    return minimum, maximum


@dataclasses.dataclass(frozen=True)
class ReadinessConsumer:
    """A consumer execution site's symbolic requirements from one event."""

    consumer_root: int
    incidence: Incidence
    consumer_id: int
    covered_obligations: frozenset[DependencyObligation] = frozenset()
    consumer_site_id: int | None = None

    def __post_init__(self) -> None:
        if self.incidence.keys_by_item is None:
            raise ValueError("consumer incidence requires both orientations")

    @property
    def keys_by_consumer(self) -> CoordinateRelation:
        return cast("CoordinateRelation", self.incidence.keys_by_item)


def nested_wait_placement(
    root_domain: CoordinateDomain,
    consumer: ReadinessConsumer,
) -> tuple[int, tuple[int, ...] | None] | None:
    """Return the nested axis and static starts, or ``None`` for every iteration."""
    domain = consumer.keys_by_consumer.source_domain
    nested_axes = nested_logical_axes(root_domain, domain)
    if len(nested_axes) != 1:
        return None
    (nested_axis,) = nested_axes
    nested_symbol = coordinate_axis_symbol(nested_axis)
    if not consumer.keys_by_consumer.has_total_source() or any(
        nested_symbol in expression.free_symbols
        for piece in consumer.keys_by_consumer.pieces
        for _axis, begin, end, _step in piece.target_ranges
        for expression in (begin, end)
    ):
        return nested_axis, None
    extent = domain.axis_counts[nested_axis]
    boundaries = tuple(
        sorted(
            {
                int(boundary)
                for piece in consumer.keys_by_consumer.pieces
                for axis, begin, end, _step in piece.source_bounds_items
                if axis == nested_axis
                for boundary in (begin, end)
                if 0 < boundary < extent
            }
        )
    )
    return nested_axis, (0, *boundaries)


def _readiness_key_domain(
    producers: tuple[ReadinessProducer, ...],
    consumers: tuple[ReadinessConsumer, ...],
) -> CoordinateDomain:
    if not producers:
        raise ValueError("an event requires at least one producer")
    readiness_key_domain = producers[0].incidence.items_by_key.source_domain
    if any(
        readiness_producer.incidence.items_by_key.source_domain != readiness_key_domain
        for readiness_producer in producers[1:]
    ) or any(
        readiness_consumer.incidence.items_by_key.source_domain != readiness_key_domain
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
        consumer_ids = tuple(consumer.consumer_id for consumer in self.consumers)
        if consumer_ids != tuple(range(len(self.consumers))):
            raise ValueError("consumer IDs must be dense and source ordered")
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
        return self.producers[0].incidence.items_by_key.source_domain

    @property
    def event_id(self) -> int:
        """Return the event identity owned by its readiness-key domain."""
        identity = self.readiness_key_domain.identity
        assert identity is not None
        return identity

    @property
    def root_barrier_producer_root(self) -> int | None:
        if (
            self.readiness_key_domain.size == 1
            and len(self.producers) == 1
            and self.producers[0].producer_site_id is None
            and self.producers[0].incidence.items_by_key.is_total()
        ):
            return self.producers[0].producer_root
        return None


@dataclasses.dataclass(frozen=True)
class ReadinessGraph:
    """Configured semantic readiness DAG over fixed-capacity root domains."""

    root_domains: tuple[CoordinateDomain, ...]
    events: tuple[ReadinessEvent, ...]

    def __post_init__(self) -> None:
        if tuple(event.event_id for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("event IDs must be dense and source ordered")

    def event(self, event_id: int) -> ReadinessEvent:
        return self.events[event_id]


def _supports_readiness_counter_lowering(
    readiness_producer: ReadinessProducer,
) -> bool:
    return (
        readiness_producer.incidence.keys_by_item is not None
        and readiness_producer.incidence.count_by_key is not None
        and readiness_producer.incidence.keys_by_item.is_single_valued()
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
) -> None:
    """Canonicalize and group an event by its producer partition."""
    canonical_domain = CoordinateDomain(
        tuple(range(len(readiness_key_domain.axis_order))),
        tuple(
            (axis, count)
            for axis, (_old_axis, count) in enumerate(
                readiness_key_domain.axis_counts_items
            )
        ),
        kind="event",
    )

    def rename(incidence: Incidence, domain: CoordinateDomain) -> Incidence:
        result = incidence.rename_domains(domain, incidence.items_by_key.target_domain)
        if result is None:
            raise AssertionError("event relation does not match its quotient geometry")
        return result

    canonical_producers_tuple = tuple(
        dataclasses.replace(
            producer, incidence=rename(producer.incidence, canonical_domain)
        )
        for producer in producers
    )
    canonical_consumers = tuple(
        dataclasses.replace(
            consumer, incidence=rename(consumer.incidence, canonical_domain)
        )
        for consumer in consumers
    )
    signature = canonical_domain, canonical_producers_tuple
    previous_event = pending.get(signature)
    if previous_event is None:
        event_id = len(pending)
        identified_domain = dataclasses.replace(canonical_domain, identity=event_id)
        event_producers = tuple(
            dataclasses.replace(
                producer, incidence=rename(producer.incidence, identified_domain)
            )
            for producer in canonical_producers_tuple
        )
        previous_consumers: tuple[ReadinessConsumer, ...] = ()
    else:
        event_id = previous_event.event_id
        identified_domain = previous_event.readiness_key_domain
        event_producers = previous_event.producers
        previous_consumers = previous_event.consumers

    grouped_consumers = list(previous_consumers)
    for canonical_consumer in canonical_consumers:
        readiness_consumer = dataclasses.replace(
            canonical_consumer,
            incidence=rename(canonical_consumer.incidence, identified_domain),
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
            grouped_consumers.append(
                dataclasses.replace(
                    readiness_consumer, consumer_id=len(grouped_consumers)
                )
            )
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


@dataclasses.dataclass(frozen=True)
class ReadinessCounterPlan:
    """One emitted readiness-key space and its optional continuation."""

    producers: tuple[ReadinessProducer, ...]
    consumers: tuple[ReadinessConsumer, ...]
    continuation_consumer_index: int | None = None

    def __post_init__(self) -> None:
        _readiness_key_domain(self.producers, self.consumers)
        if any(consumer.consumer_id < 0 for consumer in self.consumers):
            raise ValueError("counter consumers require stable event-local IDs")
        if self.continuation_consumer_index is not None and not (
            0 <= self.continuation_consumer_index < len(self.consumers)
        ):
            raise ValueError("continuation consumer index is out of range")

    @property
    def readiness_key_domain(self) -> CoordinateDomain:
        """Return the readiness-key domain owned by every event relation."""
        return self.producers[0].incidence.items_by_key.source_domain

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
        bounds = _arrival_count_bounds(self.producers)
        return None if bounds is None or bounds[0] != bounds[1] else bounds[0]

    def is_root_barrier_equivalent(
        self,
        root_domains: tuple[CoordinateDomain, ...],
        charge: Callable[[int], bool],
    ) -> bool:
        """Return whether this counter is exactly a set of root barriers."""
        if (
            self.continuation_consumer_index is not None
            or self.readiness_key_count != 1
            or len(self.producers) != 1
        ):
            return False
        producer = self.producers[0]
        producer_domain = root_domains[producer.producer_root]
        count = producer.incidence.count_by_key
        if (
            producer.producer_site_id is not None
            or producer.incidence.items_by_key.target_domain != producer_domain
            or count is None
            or count.value_bounds() != (producer_domain.size, producer_domain.size)
            or any(
                consumer.consumer_site_id is not None
                or consumer.consumer_root <= producer.producer_root
                or consumer.incidence.items_by_key.target_domain
                != root_domains[consumer.consumer_root]
                for consumer in self.consumers
            )
        ):
            return False
        return charge(
            sum(
                len(consumer.keys_by_consumer.pieces) ** 2
                for consumer in self.consumers
            )
        ) and all(
            consumer.keys_by_consumer.is_total_function() for consumer in self.consumers
        )


def _emitted_final_arrival_continuations(
    readiness_graph: ReadinessGraph,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
) -> tuple[FinalArrivalContinuation, ...] | None:
    result: list[FinalArrivalContinuation] = []
    consumer_roots: set[int] = set()
    for plan in readiness_counters:
        index = plan.continuation_consumer_index
        if index is None:
            continue
        event_id = plan.readiness_key_domain.identity
        consumer_id = plan.consumers[index].consumer_id
        if event_id is None or not 0 <= event_id < len(readiness_graph.events):
            return None
        continuation = FinalArrivalContinuation(event_id, consumer_id)
        event = readiness_graph.event(continuation.event_id)
        if not 0 <= continuation.consumer_index < len(event.consumers):
            return None
        consumer_root = event.consumers[continuation.consumer_index].consumer_root
        if (
            consumer_root != plan.consumers[index].consumer_root
            or consumer_root in consumer_roots
        ):
            return None
        consumer_roots.add(consumer_root)
        result.append(continuation)
    return tuple(result)


def _contract_static_producer_incidences(
    readiness_graph: ReadinessGraph,
    queries: tuple[tuple[int, int | None, Incidence], ...],
    continuation_by_root: dict[int, FinalArrivalContinuation],
    charge: Callable[[int], bool],
) -> tuple[tuple[int, Incidence], ...] | None:
    """Contract continuations with one bounded, topology-safe traversal."""
    memo: dict[
        tuple[int, int | None, Incidence],
        tuple[tuple[int, Incidence], ...] | None,
    ] = {}

    def compose(
        left: Incidence,
        right: Incidence,
    ) -> Incidence | None:
        if not charge(
            1 + len(left.items_by_key.pieces) * len(right.items_by_key.pieces)
        ):
            return None
        result = left.then(right)
        if result is not None and not charge(len(result.items_by_key.pieces)):
            return None
        return result

    def merge(
        relations: tuple[tuple[int, Incidence], ...],
    ) -> tuple[tuple[int, Incidence], ...] | None:
        grouped: dict[int, list[Incidence]] = {}
        for root, incidence in relations:
            grouped.setdefault(root, []).append(incidence)
        merged: list[tuple[int, Incidence]] = []
        for root, group in sorted(grouped.items()):
            if not charge(1 + sum(len(item.items_by_key.pieces) for item in group)):
                return None
            incidence = Incidence.union_all(tuple(group))
            if incidence is None or not charge(len(incidence.items_by_key.pieces)):
                return None
            merged.append((root, incidence))
        return tuple(merged)

    def expand(
        key: tuple[int, int | None, Incidence],
        active: frozenset[tuple[int, int]],
    ) -> tuple[tuple[int, Incidence], ...] | None:
        if key in memo:
            return memo[key]
        if not charge(1):
            return None
        root, site_id, readiness_keys = key
        root_domain = readiness_graph.root_domains[root]
        root_keys = (
            readiness_keys
            if site_id is None
            else readiness_keys.project_items(root_domain)
        )
        if root_keys is None:
            memo[key] = None
            return None
        continuation = continuation_by_root.get(root)
        if continuation is None:
            result = ((root, root_keys),)
            memo[key] = result
            return result

        topology_key = (root, continuation.event_id)
        if topology_key in active:
            memo[key] = None
            return None
        if not 0 <= continuation.event_id < len(readiness_graph.events):
            memo[key] = None
            return None
        event = readiness_graph.event(continuation.event_id)
        if not 0 <= continuation.consumer_index < len(event.consumers):
            memo[key] = None
            return None
        consumer = event.consumers[continuation.consumer_index]
        consumer_keys = consumer.incidence.reversed()
        if consumer.consumer_root != root or consumer_keys is None:
            memo[key] = None
            return None
        target_to_upstream_keys = compose(root_keys, consumer_keys)
        if target_to_upstream_keys is None:
            memo[key] = None
            return None

        expanded: list[tuple[int, Incidence]] = []
        next_active = active | {topology_key}
        for producer in event.producers:
            if not charge(1):
                return None
            upstream_keys = compose(target_to_upstream_keys, producer.incidence)
            if upstream_keys is None:
                memo[key] = None
                return None
            upstream = expand(
                (
                    producer.producer_root,
                    producer.producer_site_id,
                    upstream_keys,
                ),
                next_active,
            )
            if upstream is None:
                memo[key] = None
                return None
            expanded.extend(upstream)
        result = merge(tuple(expanded))
        memo[key] = result
        return result

    expanded: list[tuple[int, Incidence]] = []
    for query in queries:
        result = expand(query, frozenset())
        if result is None:
            return None
        expanded.extend(result)
    return merge(tuple(expanded))


def _nested_readiness_frontier(
    readiness_graph: ReadinessGraph,
    event: ReadinessEvent,
    consumer: ReadinessConsumer,
    *,
    pipeline_plan: StaticPipelinePlan,
    static_producers: _StaticProducerResolver,
    charge: Callable[[int], bool],
) -> CoordinateRelation | None:
    assert consumer.consumer_site_id is not None
    domain = consumer.keys_by_consumer.source_domain
    nested_axes = nested_logical_axes(
        readiness_graph.root_domains[consumer.consumer_root], domain
    )
    if len(nested_axes) != 1:
        return None
    static_relations = static_producers(event.producers)
    if static_relations is None:
        return None
    piece_count = sum(
        len(incidence.items_by_key.pieces) for _root, incidence in static_relations
    )
    if not charge(
        1 + (domain.size + pipeline_plan.wave_domain.size) * max(1, piece_count) ** 2
    ):
        return None
    maxima: list[CoordinateRelation] = []
    for root, incidence in static_relations:
        if incidence.items_by_key.target_domain != readiness_graph.root_domains[root]:
            return None
        maximum = _maximum_root_wave_by_key(
            pipeline_plan, root, incidence.items_by_key, charge
        )
        if maximum is None:
            return None
        maxima.append(maximum)
    if not maxima:
        return None
    combined = CoordinateRelation.union_all(tuple(maxima))
    if combined is None:
        return None
    ready_by_key = combined.max_target_value_by_source(
        CoordinateRelation.identity(combined.target_domain, combined.target_domain)
    )
    if ready_by_key is None:
        return None
    ready = consumer.keys_by_consumer.then(ready_by_key)
    if ready is None or not ready.is_total_function():
        return None

    nested_axis = nested_axes[0]
    affecting_axes = ready.source_axes_affecting_targets()
    if affecting_axes is None:
        return None
    retained_axes = tuple(
        axis
        for axis in domain.axis_order
        if axis == nested_axis or axis in affecting_axes
    )
    counts = domain.axis_count_expressions
    reduced_domain = CoordinateDomain(
        axis_order=retained_axes,
        axis_counts_items=tuple((axis, counts[axis]) for axis in retained_axes),
        block_sizes_items=tuple(
            (axis, domain.block_sizes[axis])
            for axis in retained_axes
            if axis in domain.block_sizes
        ),
        kind=domain.kind,
        identity=domain.identity,
        _allow_empty=domain._allow_empty,
    )
    reduced = ready.project_source(reduced_domain)
    recomposed = None if reduced is None else reduced.lift_source(domain)
    if (
        reduced is None
        or recomposed is None
        or not recomposed.covers(ready)
        or not ready.covers(recomposed)
        or not reduced.is_total_function()
    ):
        return None
    nested_domain = CoordinateDomain(
        axis_order=(nested_axis,),
        axis_counts_items=((nested_axis, counts[nested_axis]),),
        block_sizes_items=(
            ((nested_axis, domain.block_sizes[nested_axis]),)
            if nested_axis in domain.block_sizes
            else ()
        ),
        kind=domain.kind,
        identity=domain.identity,
        _allow_empty=domain._allow_empty,
    )
    if reduced_domain == nested_domain:
        return reduced
    projection = KeyPartition.projection(reduced_domain, nested_domain)
    return (
        None
        if projection is None
        else projection.fine_keys_by_coarse_key.max_target_value_by_source(reduced)
    )


def _segmented_nested_loop_counter(
    readiness_graph: ReadinessGraph,
    event: ReadinessEvent,
    readiness_consumer: ReadinessConsumer,
    boundaries: tuple[int, ...] | None,
    charge: Callable[[int], bool],
) -> ReadinessCounterPlan | None:
    """Coarsen one exact nested dependency into contiguous loop segments."""
    assert readiness_consumer.consumer_site_id is not None
    domain = readiness_consumer.keys_by_consumer.source_domain
    nested_axes = nested_logical_axes(
        readiness_graph.root_domains[readiness_consumer.consumer_root], domain
    )
    if len(nested_axes) != 1:
        return None
    nested_axis = nested_axes[0]
    nested_extent = domain.axis_counts[nested_axis]
    if boundaries is None:
        normalized_boundaries = (0, nested_extent)
    else:
        try:
            normalized_boundaries = tuple(int(boundary) for boundary in boundaries)
        except (TypeError, ValueError):
            return None
    segments = tuple(itertools.pairwise(normalized_boundaries))
    if not segments:
        return None
    used_axes = readiness_consumer.keys_by_consumer.source_axes_affecting_targets()
    if used_axes is None or nested_axis not in used_axes:
        return None
    domain_counts = domain.axis_count_expressions
    reduced_domain = CoordinateDomain(
        axis_order=used_axes,
        axis_counts_items=tuple((axis, domain_counts[axis]) for axis in used_axes),
        block_sizes_items=tuple(
            (axis, domain.block_sizes[axis])
            for axis in used_axes
            if axis in domain.block_sizes
        ),
        kind="site",
        identity=domain.identity,
        _allow_empty=domain._allow_empty,
    )
    outer_axes = tuple(axis for axis in used_axes if axis != nested_axis)
    reduced_counts = reduced_domain.axis_count_expressions
    readiness_key_domain = CoordinateDomain(
        axis_order=tuple(range(len(outer_axes) + 1)),
        axis_counts_items=(
            (0, len(segments)),
            *(
                (event_axis, reduced_counts[source_axis])
                for event_axis, source_axis in enumerate(outer_axes, start=1)
            ),
        ),
        kind="event",
        identity=event.event_id,
        _allow_empty=reduced_domain._allow_empty,
    )
    segment_partition = KeyPartition.contiguous_segments(
        reduced_domain, readiness_key_domain, nested_axis, segments
    )
    reduced_consumers = readiness_consumer.incidence.project_items(reduced_domain)
    if segment_partition is None or reduced_consumers is None:
        return None
    segment_incidence = segment_partition.as_incidence()
    root_projection = KeyPartition.projection(domain, reduced_domain)
    reduced_keys = reduced_consumers.keys_by_item
    if (
        reduced_keys is None
        or root_projection is None
        or not charge(
            1
            + len(reduced_consumers.items_by_key.pieces) ** 2
            + len(reduced_keys.pieces) ** 2
            + len(segment_incidence.items_by_key.pieces)
            * (
                len(reduced_keys.pieces)
                + len(root_projection.fine_keys_by_coarse_key.pieces)
            )
            + len(segment_partition.coarse_key_by_fine_key.pieces)
            * (
                len(reduced_consumers.items_by_key.pieces)
                + len(root_projection.coarse_key_by_fine_key.pieces)
            )
        )
        or not reduced_keys.has_total_source()
    ):
        return None
    partition = segment_partition.rekey_fine(reduced_consumers)
    consumer_incidence = segment_incidence.then(root_projection.as_incidence())
    if partition is None or consumer_incidence is None:
        return None
    selected_consumer = dataclasses.replace(readiness_consumer, consumer_id=0)
    lowered = _coarsen_event(
        ReadinessEvent(event.producers, (selected_consumer,)),
        partition,
        known_consumer_incidences={0: consumer_incidence},
        charge=charge,
    )
    if lowered is None:
        return None
    lowered_consumer = dataclasses.replace(
        lowered[1][0], consumer_id=readiness_consumer.consumer_id
    )
    return ReadinessCounterPlan(
        producers=lowered[0],
        consumers=(lowered_consumer,),
    )


def collect_nested_loop_scheduling_counters(
    readiness_graph: ReadinessGraph,
    charge: Callable[[int], bool],
) -> tuple[ReadinessCounterPlan, ...]:
    """Select one exact schedule-independent plan for each nested wait."""
    root_domains = readiness_graph.root_domains
    result: list[ReadinessCounterPlan] = []
    event_consumers = sorted(
        (
            (event, consumer)
            for event in readiness_graph.events
            for consumer in event.consumers
            if consumer.consumer_site_id is not None
        ),
        key=lambda item: (
            item[1].consumer_root,
            cast("int", item[1].consumer_site_id),
            item[0].event_id,
        ),
    )
    preceding_obligations: set[DependencyObligation] = set()
    previous_root: int | None = None
    for event, readiness_consumer in event_consumers:
        if readiness_consumer.consumer_root != previous_root:
            previous_root = readiness_consumer.consumer_root
            preceding_obligations.clear()
        if (
            readiness_consumer.covered_obligations
            and readiness_consumer.covered_obligations <= preceding_obligations
        ):
            continue
        preceding_obligations.update(readiness_consumer.covered_obligations)
        exact = ReadinessCounterPlan(
            producers=event.producers,
            consumers=(readiness_consumer,),
        )
        if _supports_emitted_counter_plan_lowering(exact, root_domains):
            result.append(exact)
            continue
        entry = _segmented_nested_loop_counter(
            readiness_graph,
            event,
            readiness_consumer,
            None,
            charge,
        )
        if entry is not None and _supports_emitted_counter_plan_lowering(
            entry,
            root_domains,
        ):
            result.append(entry)
    return tuple(result)


def _compact_nested_loop_counters_for_schedule(
    readiness_graph: ReadinessGraph,
    pipeline_plan: StaticPipelinePlan,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    static_producers: _StaticProducerResolver,
    charge: Callable[[int], bool],
) -> tuple[ReadinessCounterPlan, ...]:
    """Strength-reduce exact nested counters on an accepted placement."""
    scheduled_roots = frozenset(pipeline_plan.resident_roots)
    task_steps: tuple[CoordinateRelation | None, ...] | None = None
    task_steps_computed = False
    result: list[ReadinessCounterPlan] = []
    for plan in readiness_counters:
        if len(plan.consumers) != 1 or plan.continuation_consumer_index is not None:
            result.append(plan)
            continue
        consumer = plan.consumers[0]
        event_id = plan.readiness_key_domain.identity
        if event_id is None or not 0 <= event_id < len(readiness_graph.events):
            result.append(plan)
            continue
        event = readiness_graph.event(event_id)
        if (
            not 0 <= consumer.consumer_id < len(event.consumers)
            or event.consumers[consumer.consumer_id] is not consumer
            or plan.producers is not event.producers
            or consumer.consumer_site_id is None
        ):
            result.append(plan)
            continue
        consumer_root = consumer.consumer_root
        if consumer_root not in scheduled_roots:
            result.append(plan)
            continue
        consumer_first_wave = (
            pipeline_plan.static_base(consumer_root) // pipeline_plan.worker_count
        )
        entry = _segmented_nested_loop_counter(
            readiness_graph, event, consumer, None, charge
        )
        entry_consumer = None if entry is None else entry.consumers[0]
        entry_consumer_keys = (
            None
            if entry_consumer is None
            else _keys_by_consumer_root_task(readiness_graph, entry_consumer)
        )
        entry_is_smaller = (
            entry is not None and plan.readiness_key_count > entry.readiness_key_count
        )
        terminal_producers = (
            None if entry is None else static_producers(entry.producers)
        )
        active_roots = (
            None
            if terminal_producers is None
            else tuple(
                root
                for root, incidence in terminal_producers
                if incidence.items_by_key.source_support_is_empty() is not True
            )
        )
        same_root_producer = active_roots is not None and consumer_root in active_roots
        root_order_precedes = active_roots is not None and all(
            root in scheduled_roots and root <= consumer_root for root in active_roots
        )
        if entry_is_smaller and same_root_producer and not task_steps_computed:
            task_steps = _task_step_relations(pipeline_plan, charge)
            task_steps_computed = True
        same_root_rank_precedes = not same_root_producer or (
            entry is not None
            and entry_consumer is not None
            and entry_consumer_keys is not None
            and task_steps is not None
            and _counter_prerequisite_has_progress_precedence(
                pipeline_plan=pipeline_plan,
                terminal_producers=cast(
                    "tuple[tuple[int, Incidence], ...]", terminal_producers
                ),
                consumer_root=entry_consumer.consumer_root,
                consumer_keys=cast(
                    "CoordinateRelation", entry_consumer_keys.keys_by_item
                ),
                task_steps=task_steps,
                charge=charge,
            )
        )
        entry_has_progress_precedence = (
            entry is not None
            and entry_consumer is not None
            and entry_consumer_keys is not None
            and entry_consumer_keys.keys_by_item is not None
            and entry_consumer_keys.keys_by_item.is_total_function()
            and entry_is_smaller
            and entry_consumer.covered_obligations == consumer.covered_obligations
            and _supports_emitted_counter_plan_lowering(
                entry,
                readiness_graph.root_domains,
            )
            and root_order_precedes
            and same_root_rank_precedes
        )
        if entry_has_progress_precedence:
            assert entry is not None
            result.append(entry)
            continue
        if active_roots is not None and any(
            root not in scheduled_roots for root in active_roots
        ):
            result.append(plan)
            continue
        frontier = _nested_readiness_frontier(
            readiness_graph,
            event,
            consumer,
            pipeline_plan=pipeline_plan,
            static_producers=static_producers,
            charge=charge,
        )
        split = (
            None
            if frontier is None or not charge(1 + len(frontier.pieces))
            else frontier.prefix_before(consumer_first_wave)
        )
        if split is None:
            result.append(plan)
            continue
        compact = _segmented_nested_loop_counter(
            readiness_graph,
            event,
            consumer,
            tuple(sorted({0, split, frontier.source_domain.size})),
            charge,
        )
        if (
            compact is None
            or (
                entry is not None
                and compact.readiness_key_count == entry.readiness_key_count
            )
            or compact.readiness_key_count >= plan.readiness_key_count
            or compact.consumers[0].covered_obligations != consumer.covered_obligations
            or not _supports_emitted_counter_plan_lowering(
                compact,
                readiness_graph.root_domains,
            )
        ):
            result.append(plan)
            continue
        result.append(compact)
    return tuple(result)


@dataclasses.dataclass(frozen=True)
class StaticPipelinePlan:
    """The finalized root-indexed static execution and synchronization plan."""

    worker_count: int
    execution_orders: tuple[DenseTaskOrder, ...]
    body_orders: tuple[DenseTaskOrder, ...]
    readiness_counters: tuple[ReadinessCounterPlan, ...]
    root_barrier_edges: frozenset[tuple[int, int]]

    def __post_init__(self) -> None:
        if self.worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if len(self.execution_orders) != len(self.body_orders):
            raise ValueError("every root requires one execution and body order")
        root_count = len(self.execution_orders)
        root_domains = self.root_domains
        if any(domain.size_expr.is_zero is True for domain in root_domains):
            raise ValueError("pipeline plan requires positive root capacity")
        for root, (order, body_order) in enumerate(
            zip(self.execution_orders, self.body_orders, strict=True)
        ):
            if (
                body_order.tasks_by_ordinal.target_domain
                != order.tasks_by_ordinal.target_domain
            ):
                raise ValueError(
                    f"body PID ABI for root {root} disagrees with its task domain"
                )
        for producer_root, consumer_root in self.root_barrier_edges:
            if not (
                0 <= producer_root < root_count and 0 <= consumer_root < root_count
            ):
                raise ValueError("root-barrier edge references an unknown root")
            if producer_root >= consumer_root:
                raise ValueError(
                    "root-barrier edge must be a strict source-ordered dependency"
                )

        continuation_roots: list[int] = []
        for counter in self.readiness_counters:
            if not _supports_emitted_counter_plan_lowering(
                counter,
                root_domains,
            ):
                raise ValueError("readiness counter has no fixed exact lowering")
            for consumer_index, consumer in enumerate(counter.consumers):
                if consumer_index == counter.continuation_consumer_index:
                    continuation_roots.append(consumer.consumer_root)
        if len(set(continuation_roots)) != len(continuation_roots):
            raise ValueError("one root cannot have multiple continuation owners")

    @property
    def root_domains(self) -> tuple[CoordinateDomain, ...]:
        return tuple(
            order.tasks_by_ordinal.target_domain for order in self.execution_orders
        )

    @property
    def body_pid_by_task(self) -> tuple[CoordinateRelation, ...]:
        return tuple(order.ordinal_by_task for order in self.body_orders)

    @cached_property
    def continuation_roots(self) -> frozenset[int]:
        return frozenset(
            consumer.consumer_root
            for counter in self.readiness_counters
            if (consumer := counter.continuation_consumer) is not None
        )

    @property
    def resident_roots(self) -> tuple[int, ...]:
        return tuple(
            root
            for root in range(len(self.execution_orders))
            if root not in self.continuation_roots
        )

    def static_base(self, root: int) -> int:
        """Return one root's immutable W-aligned ownership base."""
        return sum(self._padded_task_count(index) for index in range(root))

    def _padded_task_count(self, root: int) -> int:
        tasks = self.execution_orders[root].task_count
        return -(-tasks // self.worker_count) * self.worker_count

    @property
    def static_slot_count(self) -> int:
        return sum(
            self._padded_task_count(root) for root in range(len(self.execution_orders))
        )

    @cached_property
    def wave_domain(self) -> CoordinateDomain:
        return CoordinateDomain.scalar(
            self.static_slot_count // self.worker_count,
            kind="value",
        )

    def root_barrier_arrival_count(self, root: int) -> int:
        """Return the exact number of physical publishers for one root."""
        continuation = next(
            (
                consumer
                for counter in self.readiness_counters
                if (consumer := counter.continuation_consumer) is not None
                and consumer.consumer_root == root
            ),
            None,
        )
        if continuation is not None:
            return continuation.keys_by_consumer.source_domain.size
        return min(self.worker_count, self.execution_orders[root].task_count)


def _nested_counter_acquire_count(
    plan: StaticPipelinePlan, consumer: ReadinessConsumer
) -> int | None:
    """Count waits emitted by the shared nested-loop placement policy."""
    placement = nested_wait_placement(
        plan.root_domains[consumer.consumer_root], consumer
    )
    if placement is None:
        return None
    _nested_axis, segment_starts = placement
    if segment_starts is not None:
        return plan.root_domains[consumer.consumer_root].size * len(segment_starts)
    return (
        consumer.keys_by_consumer.source_domain.size
        if consumer.keys_by_consumer.has_total_source()
        else None
    )


def _without_wave_dominated_nested_counters(
    plan: StaticPipelinePlan,
) -> tuple[ReadinessCounterPlan, ...]:
    """Prefer a barrier when a one-wave producer has strictly cheaper sync."""
    retained = []
    for counter in plan.readiness_counters:
        if len(counter.producers) != 1 or len(counter.consumers) != 1:
            retained.append(counter)
            continue
        producer, consumer = counter.producers[0], counter.consumers[0]
        producer_tasks = plan.root_domains[producer.producer_root].size
        arrivals = counter.uniform_arrival_count()
        acquires = _nested_counter_acquire_count(plan, consumer)
        if (
            producer.producer_site_id is not None
            or consumer.consumer_site_id is None
            or producer.producer_root >= consumer.consumer_root
            or producer.producer_root in plan.continuation_roots
            or producer_tasks > plan.worker_count
            or arrivals is None
            or acquires is None
        ):
            retained.append(counter)
            continue
        counter_work = (counter.readiness_key_count * arrivals, acquires)
        barrier_work = (
            plan.root_barrier_arrival_count(producer.producer_root),
            min(
                plan.worker_count,
                plan.root_domains[consumer.consumer_root].size,
            ),
        )
        if not (
            barrier_work != counter_work
            and all(
                left <= right
                for left, right in zip(barrier_work, counter_work, strict=True)
            )
        ):
            retained.append(counter)
    return tuple(retained)


def _root_task_wave_relation(
    plan: StaticPipelinePlan,
    root: int,
    charge: Callable[[int], bool],
) -> CoordinateRelation | None:
    if root in plan.continuation_roots:
        return None
    order = plan.execution_orders[root]
    ordinal_to_wave = CoordinateRelation.scalar_floor_div(
        order.tasks_by_ordinal.source_domain,
        plan.worker_count,
        offset=plan.static_base(root) // plan.worker_count,
        target_domain=plan.wave_domain,
    )
    if not charge(1 + len(order.ordinal_by_task.pieces) * len(ordinal_to_wave.pieces)):
        return None
    result = order.ordinal_by_task.then(ordinal_to_wave)
    return result if result is None or charge(len(result.pieces)) else None


def _maximum_root_wave_by_key(
    plan: StaticPipelinePlan,
    root: int,
    tasks_by_key: CoordinateRelation | None,
    charge: Callable[[int], bool],
) -> CoordinateRelation | None:
    task_waves = _root_task_wave_relation(plan, root, charge)
    if (
        task_waves is None
        or tasks_by_key is None
        or not charge(1 + len(tasks_by_key.pieces) * len(task_waves.pieces))
    ):
        return None
    result = tasks_by_key.max_target_value_by_source(task_waves)
    return result if result is None or charge(len(result.pieces)) else None


def _root_causal_prerequisite_relations(
    readiness_graph: ReadinessGraph,
    charge: Callable[[int], bool],
) -> tuple[tuple[tuple[int, int], Incidence], ...]:
    """Derive exact root-task happens-before relations from root events."""

    def compose(left: Incidence, right: Incidence) -> Incidence | None:
        if not charge(
            1 + len(left.items_by_key.pieces) * len(right.items_by_key.pieces)
        ):
            return None
        result = left.then(right)
        return (
            result
            if result is None or charge(len(result.items_by_key.pieces))
            else None
        )

    def record(
        relations: dict[tuple[int, int], Incidence],
        key: tuple[int, int],
        relation: Incidence,
    ) -> None:
        previous = relations.get(key)
        if previous is None:
            if charge(len(relation.items_by_key.pieces)):
                relations[key] = relation
            return
        if not charge(
            1 + len(previous.items_by_key.pieces) + len(relation.items_by_key.pieces)
        ):
            return
        combined = Incidence.union_all((previous, relation))
        if combined is not None and charge(len(combined.items_by_key.pieces)):
            relations[key] = combined

    direct: dict[tuple[int, int], Incidence] = {}

    for event in readiness_graph.events:
        for consumer in event.consumers:
            consumer_domain = readiness_graph.root_domains[consumer.consumer_root]
            consumer_incidence = consumer.incidence.project_items(consumer_domain)
            consumer_keys = (
                None if consumer_incidence is None else consumer_incidence.reversed()
            )
            if consumer_keys is None:
                continue
            for producer in event.producers:
                if producer.producer_site_id is not None:
                    continue
                producer_domain = readiness_graph.root_domains[producer.producer_root]
                producer_tasks = producer.incidence.project_items(producer_domain)
                relation = (
                    None
                    if producer_tasks is None
                    else compose(consumer_keys, producer_tasks)
                )
                if relation is not None:
                    record(
                        direct,
                        (consumer.consumer_root, producer.producer_root),
                        relation,
                    )

    root_count = len(readiness_graph.root_domains)
    successors = [set() for _ in range(root_count)]
    direct_by_consumer: dict[int, list[tuple[int, Incidence]]] = {}
    for (consumer_root, producer_root), relation in direct.items():
        if consumer_root == producer_root:
            continue
        successors[producer_root].add(consumer_root)
        direct_by_consumer.setdefault(consumer_root, []).append(
            (producer_root, relation)
        )
    order = _deterministic_topological_order(successors)
    if order is None:
        return tuple(sorted(direct.items()))

    closure: dict[tuple[int, int], Incidence] = {}
    for consumer_root in order:
        for producer_root, relation in direct_by_consumer.get(consumer_root, ()):
            record(closure, (consumer_root, producer_root), relation)
            for (
                (intermediate_root, ancestor_root),
                ancestor_relation,
            ) in tuple(closure.items()):
                if intermediate_root != producer_root:
                    continue
                transitive = compose(relation, ancestor_relation)
                if transitive is not None:
                    record(
                        closure,
                        (consumer_root, ancestor_root),
                        transitive,
                    )
    return tuple(sorted(closure.items()))


def _causally_maximal_event_producers(
    producers: tuple[ReadinessProducer, ...],
    causal_relations: dict[tuple[int, int], Incidence],
    charge: Callable[[int], bool],
) -> tuple[tuple[int, Incidence], ...] | None:
    """Keep every producer arm not proved upstream of another event arm."""
    grouped: dict[int, list[Incidence]] = {}
    for producer in producers:
        if producer.producer_site_id is not None:
            return None
        grouped.setdefault(producer.producer_root, []).append(producer.incidence)
    producers_by_root: list[tuple[int, Incidence]] = []
    for root, incidences in sorted(grouped.items()):
        incidence = (
            incidences[0]
            if len(incidences) == 1
            else (
                Incidence.union_all(tuple(incidences))
                if charge(1 + sum(len(item.items_by_key.pieces) for item in incidences))
                else None
            )
        )
        if incidence is None:
            return None
        producers_by_root.append((root, incidence))
    if not producers_by_root:
        return None
    result: list[tuple[int, Incidence]] = []
    for producer_root, incidence in producers_by_root:
        producers_for_key = incidence.items_by_key
        dominated: CoordinateRelation | None = None
        for later_root, later_incidence in producers_by_root:
            if later_root == producer_root:
                continue
            later_to_producer = causal_relations.get((later_root, producer_root))
            causally_prior_incidence = (
                None
                if later_to_producer is None
                or not charge(
                    1
                    + len(later_incidence.items_by_key.pieces)
                    * len(later_to_producer.items_by_key.pieces)
                )
                else later_incidence.then(later_to_producer)
            )
            causally_prior = (
                None
                if causally_prior_incidence is None
                else causally_prior_incidence.items_by_key
            )
            if causally_prior is None:
                continue
            combined = (
                causally_prior
                if dominated is None
                else (
                    CoordinateRelation.union_all((dominated, causally_prior))
                    if charge(1 + len(dominated.pieces) + len(causally_prior.pieces))
                    else None
                )
            )
            if combined is not None:
                dominated = combined
        if dominated is None or not dominated.covers(producers_for_key):
            result.append((producer_root, incidence))
    return tuple(result)


def _continuation_dominance_owner(
    continuation: FinalArrivalContinuation,
    counter: ReadinessCounterPlan,
    *,
    plan: StaticPipelinePlan,
    slot_domain: CoordinateDomain,
    execution_by_slot_by_root: dict[int, CoordinateRelation],
    virtual_placements: dict[int, Incidence],
    removed_roots: frozenset[int],
    virtual_supports: tuple[CoordinateRelation, ...],
    causal_relations: dict[tuple[int, int], Incidence],
    charge: Callable[[int], bool],
) -> Incidence | None:
    """Prove one continuation is a non-displacing ownership strength reduction."""
    matching_consumers = tuple(
        consumer
        for consumer in counter.consumers
        if consumer.consumer_id == continuation.consumer_index
    )
    if len(matching_consumers) != 1:
        return None
    (consumer,) = matching_consumers
    maximal = _causally_maximal_event_producers(
        counter.producers, causal_relations, charge
    )
    if not maximal:
        return None
    piece_count = sum(
        len(incidence.items_by_key.pieces)
        + len(() if incidence.keys_by_item is None else incidence.keys_by_item.pieces)
        for _root, incidence in maximal
    )
    occupied_pieces = sum(
        len(execution.pieces)
        for execution in (*execution_by_slot_by_root.values(), *virtual_supports)
    )
    if not charge(
        1 + plan.worker_count * max(1, piece_count) + piece_count * occupied_pieces
    ):
        return None
    slot_incidences: list[Incidence] = []
    for producer_root, incidence in maximal:
        if producer_root in removed_roots:
            shifted = incidence.then(virtual_placements[producer_root])
        elif producer_root in plan.resident_roots:
            reindexed = incidence.reindex_items(plan.execution_orders[producer_root])
            shifted = (
                None
                if reindexed is None
                else reindexed.shift_items_scalar(
                    plan.static_base(producer_root), ambient_domain=slot_domain
                )
            )
        else:
            return None
        if shifted is None:
            return None
        slot_incidences.append(shifted)
    combined = Incidence.union_all(tuple(slot_incidences))
    last_owners = (
        None if combined is None else combined.last_item_by_residue(plan.worker_count)
    )
    shifted_owners = (
        None
        if last_owners is None
        else last_owners.shift_items_scalar(
            plan.worker_count, ambient_domain=slot_domain
        )
    )
    placement = (
        None
        if shifted_owners is None
        else cast("Incidence", consumer.incidence.reversed()).then(shifted_owners)
    )
    virtual_execution = None if placement is None else placement.keys_by_item
    if virtual_execution is None or not virtual_execution.is_single_valued():
        return None
    ignored = removed_roots | frozenset((consumer.consumer_root,))
    if any(
        root not in ignored
        and (
            (occupied := execution_by_slot_by_root.get(root)) is None
            or not virtual_execution.has_disjoint_source_support(occupied)
        )
        for root in plan.resident_roots
    ) or any(
        not virtual_execution.has_disjoint_source_support(previous)
        for previous in virtual_supports
    ):
        return None
    return placement


def choose_final_arrival_continuations(
    readiness_graph: ReadinessGraph,
    candidates: tuple[FinalArrivalContinuation, ...],
    plan: StaticPipelinePlan,
    *,
    excluded_roots: frozenset[int] = frozenset(),
    causal_relations: dict[tuple[int, int], Incidence],
    charge: Callable[[int], bool],
) -> tuple[FinalArrivalContinuation, ...]:
    """Select exact non-displacing continuation ownership in scalar slots."""
    excluded_roots |= frozenset(
        consumer.consumer_root
        for event in readiness_graph.events
        for consumer in event.consumers
        if consumer.consumer_site_id is not None
    )
    slot_domain = CoordinateDomain.scalar(
        plan.static_slot_count + (len(candidates) + 1) * plan.worker_count,
        kind="worker",
    )
    if not charge(
        1
        + sum(
            len(plan.execution_orders[root].tasks_by_ordinal.pieces)
            for root in plan.resident_roots
        )
    ):
        return ()
    execution_by_slot_by_root: dict[int, CoordinateRelation] = {}
    for root in plan.resident_roots:
        execution = plan.execution_orders[root].tasks_by_ordinal.shift_source_scalar(
            plan.static_base(root), ambient_domain=slot_domain
        )
        if execution is None:
            return ()
        execution_by_slot_by_root[root] = execution
    virtual_placements: dict[int, Incidence] = {}
    counters_by_event = {
        counter.readiness_key_domain.identity: counter
        for counter in plan.readiness_counters
    }
    result: list[FinalArrivalContinuation] = []
    removed_roots: set[int] = set()
    virtual_supports: list[CoordinateRelation] = []
    for continuation in candidates:
        consumer_root = (
            readiness_graph.event(continuation.event_id)
            .consumers[continuation.consumer_index]
            .consumer_root
        )
        if consumer_root in excluded_roots or consumer_root in removed_roots:
            continue
        counter = counters_by_event.get(continuation.event_id)
        if counter is None:
            continue
        dominance = _continuation_dominance_owner(
            continuation,
            counter,
            plan=plan,
            slot_domain=slot_domain,
            execution_by_slot_by_root=execution_by_slot_by_root,
            virtual_placements=virtual_placements,
            removed_roots=frozenset(removed_roots),
            virtual_supports=tuple(virtual_supports),
            causal_relations=causal_relations,
            charge=charge,
        )
        if dominance is None:
            continue
        virtual_execution = dominance.keys_by_item
        assert virtual_execution is not None
        result.append(continuation)
        removed_roots.add(consumer_root)
        execution_by_slot_by_root[consumer_root] = virtual_execution
        virtual_placements[consumer_root] = dominance
        virtual_supports.append(virtual_execution)
    return tuple(result)


def _assign_final_arrival_continuations(
    readiness_graph: ReadinessGraph,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    continuations: tuple[FinalArrivalContinuation, ...],
) -> tuple[ReadinessCounterPlan, ...] | None:
    selected = {
        (continuation.event_id, continuation.consumer_index)
        for continuation in continuations
    }
    if len(selected) != len(continuations):
        return None
    assigned: set[tuple[int, int]] = set()
    result: list[ReadinessCounterPlan] = []
    for plan in readiness_counters:
        event_id = plan.readiness_key_domain.identity
        if event_id is not None and not 0 <= event_id < len(readiness_graph.events):
            return None
        matches = tuple(
            (index, (event_id, consumer.consumer_id))
            for index, consumer in enumerate(plan.consumers)
            if (event_id, consumer.consumer_id) in selected
        )
        if not matches:
            result.append(plan)
            continue
        if (
            len(matches) != 1
            or plan.continuation_consumer_index is not None
            or matches[0][1] in assigned
        ):
            return None
        plan_index, selection = matches[0]
        result.append(dataclasses.replace(plan, continuation_consumer_index=plan_index))
        assigned.add(selection)
    return tuple(result) if assigned == selected else None


def _coarsen_event(
    event: ReadinessEvent,
    partition: KeyPartition,
    *,
    known_incidences: dict[int, Incidence] | None = None,
    known_consumer_incidences: dict[int, Incidence] | None = None,
    charge: Callable[[int], bool],
) -> tuple[tuple[ReadinessProducer, ...], tuple[ReadinessConsumer, ...]] | None:
    """Coarsen every event arm through one certified key partition."""
    known_incidences = known_incidences or {}
    known_consumer_incidences = known_consumer_incidences or {}
    if (
        partition.coarse_key_by_fine_key.source_domain != event.readiness_key_domain
        or partition.coarse_key_by_fine_key.target_domain.identity != event.event_id
    ):
        return None

    to_coarsen = tuple(
        producer.incidence
        for index, producer in enumerate(event.producers)
        if index not in known_incidences
    ) + tuple(
        consumer.incidence
        for index, consumer in enumerate(event.consumers)
        if index not in known_consumer_incidences
    )
    if not charge(
        sum(
            1
            + (
                incidence.items_by_key.target_domain.size
                + len(partition.fine_keys_by_coarse_key.pieces)
                + len(partition.coarse_key_by_fine_key.pieces)
            )
            * max(
                len(incidence.items_by_key.pieces),
                len(
                    ()
                    if incidence.keys_by_item is None
                    else incidence.keys_by_item.pieces
                ),
            )
            for incidence in to_coarsen
        )
    ):
        return None

    producers: list[ReadinessProducer] = []
    for index, producer in enumerate(event.producers):
        incidence = known_incidences.get(index) or producer.incidence.coarsen(partition)
        if incidence is None:
            return None
        lowered = dataclasses.replace(producer, incidence=incidence)
        if not _supports_readiness_counter_lowering(lowered):
            return None
        producers.append(lowered)
    consumers: list[ReadinessConsumer] = []
    for index, consumer in enumerate(event.consumers):
        incidence = known_consumer_incidences.get(index) or consumer.incidence.coarsen(
            partition
        )
        if incidence is None:
            return None
        consumers.append(dataclasses.replace(consumer, incidence=incidence))
    return tuple(producers), tuple(consumers)


def choose_readiness_counters(
    readiness_graph: ReadinessGraph,
    continuations: tuple[FinalArrivalContinuation, ...],
    *,
    excluded_obligations: frozenset[DependencyObligation] = frozenset(),
    charge: Callable[[int], bool],
) -> tuple[ReadinessCounterPlan, ...]:
    """Select root-entry events representable by readiness counters."""
    continuation_consumers = {
        (continuation.event_id, continuation.consumer_index)
        for continuation in continuations
    }
    selected: list[ReadinessCounterPlan] = []
    for event in readiness_graph.events:
        if event.root_barrier_producer_root is not None:
            continue
        if not charge(
            1
            + sum(
                len(producer.incidence.items_by_key.pieces)
                + len(
                    ()
                    if producer.incidence.keys_by_item is None
                    else producer.incidence.keys_by_item.pieces
                )
                ** 2
                for producer in event.producers
            )
        ):
            continue
        lowering_relations = (
            (event.producers, event.consumers)
            if all(
                _supports_readiness_counter_lowering(producer)
                for producer in event.producers
            )
            else None
        )
        candidates: list[tuple[KeyPartition, dict[int, Incidence]]] = []
        if lowering_relations is None:
            for producer_index, producer in enumerate(event.producers):
                if _supports_readiness_counter_lowering(producer):
                    continue
                publication = producer.incidence.keys_by_item
                quotient = (
                    None
                    if publication is None or not charge(1 + len(publication.pieces))
                    else KeyPartition.from_fixed_width_publication(publication)
                )
                if quotient is not None:
                    partition, incidence = quotient
                    candidates.append((partition, {producer_index: incidence}))
            for partition, known_incidences in candidates:
                lowering_relations = _coarsen_event(
                    event,
                    partition,
                    known_incidences=known_incidences,
                    charge=charge,
                )
                if lowering_relations is not None:
                    break
        if lowering_relations is None:
            continue
        lowered_producers, lowered_consumers = lowering_relations
        retained_consumers: list[ReadinessConsumer] = []
        continuation_consumer_indices: list[int] = []
        for consumer_index, readiness_consumer in enumerate(lowered_consumers):
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
                dataclasses.replace(
                    readiness_consumer,
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
        if not retained_consumers:
            continue
        candidate = ReadinessCounterPlan(
            producers=lowered_producers,
            consumers=tuple(retained_consumers),
            continuation_consumer_index=continuation_consumer_index,
        )
        if not _supports_emitted_counter_plan_lowering(
            candidate,
            readiness_graph.root_domains,
        ):
            continue
        if candidate.is_root_barrier_equivalent(readiness_graph.root_domains, charge):
            continue
        selected.append(candidate)
    selected_continuation_count = sum(
        counter_plan.continuation_consumer_index is not None
        for counter_plan in selected
    )
    if selected_continuation_count != len(continuation_consumers):
        raise AssertionError(
            "not every final-arrival continuation has a readiness counter"
        )
    return tuple(selected)


def _build_readiness_events(
    dependency_graph: TileDependencyGraph,
    *,
    root_domains: tuple[CoordinateDomain, ...],
    site_domains: tuple[CoordinateDomain | None, ...],
    publishable_site_ids: frozenset[int] | None = None,
    prove_nonnegative: Callable[[object], bool] | None = None,
    charge: Callable[[int], bool],
) -> tuple[ReadinessEvent, ...]:
    """Build canonical readiness events from the dependency graph."""
    symbolic_dependencies = instantiate_symbolic_dependencies(
        dependency_graph,
        root_domains=root_domains,
        site_domains=site_domains,
        prove_nonnegative=prove_nonnegative,
    )
    site_by_id = {site.site_id: site for site in dependency_graph.execution_sites}
    exact_dependencies = tuple(
        dependency
        for dependency in symbolic_dependencies
        if dependency.incidence is not None
        and dependency.incidence.items_by_key.source_axes_affecting_targets()
        is not None
        and charge(1 + len(dependency.incidence.items_by_key.pieces))
    )
    obligations_by_root_pair = dependency_graph.obligations_by_root_pair()
    all_obligations_by_pair = {
        pair: set(obligations) for pair, obligations in obligations_by_root_pair
    }

    implied_obligations: dict[DependencyObligation, set[DependencyObligation]] = {}
    for preceding_dependency in exact_dependencies:
        preceding_site_id = preceding_dependency.consumer_site_id
        if preceding_site_id is None or site_by_id[preceding_site_id].is_root:
            continue
        preceding_incidence = preceding_dependency.incidence
        assert preceding_incidence is not None
        preceding_producers = preceding_incidence.items_by_key
        preceding_obligation = (
            preceding_dependency.dependency_id,
            preceding_dependency.producer_site_id,
            preceding_site_id,
        )
        for later_dependency in exact_dependencies:
            later_site_id = later_dependency.consumer_site_id
            later_incidence = later_dependency.incidence
            later_producers = (
                None if later_incidence is None else later_incidence.items_by_key
            )
            if (
                later_dependency is preceding_dependency
                or later_site_id is None
                or later_producers is None
                or preceding_dependency.consumer_root != later_dependency.consumer_root
                or preceding_dependency.producer_root != later_dependency.producer_root
                or preceding_dependency.producer_site_id
                != later_dependency.producer_site_id
                or preceding_producers.target_domain != later_producers.target_domain
                or not charge(
                    1 + len(preceding_producers.pieces) * len(later_producers.pieces)
                )
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
            list[tuple[Incidence, DependencyObligation]],
        ],
    ] = {}

    def add_exact_relation(
        *,
        producer_root: int,
        producer_site_id: int | None,
        consumer_root: int,
        consumer_site_id: int | None,
        incidence: Incidence,
        covered_obligations: frozenset[DependencyObligation],
    ) -> None:
        relation = incidence.items_by_key
        consumer = (consumer_root, consumer_site_id, relation.source_domain)
        producer = (producer_root, producer_site_id, relation.target_domain)
        exact_relations.setdefault(consumer, {}).setdefault(producer, []).extend(
            (incidence, obligation) for obligation in covered_obligations
        )

    for dependency in exact_dependencies:
        incidence = dependency.incidence
        assert incidence is not None
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
                incidence=incidence,
                covered_obligations=exact_obligations,
            )

        if producer_is_root and consumer_is_root:
            continue
        root_incidence = incidence
        if not producer_is_root:
            projected = root_incidence.project_items(
                root_domains[dependency.producer_root]
            )
            if projected is None:
                continue
            root_incidence = projected
        if not consumer_is_root:
            projected = root_incidence.project_keys(
                root_domains[dependency.consumer_root]
            )
            if projected is None:
                continue
            root_incidence = projected
        add_exact_relation(
            producer_root=dependency.producer_root,
            producer_site_id=None,
            consumer_root=dependency.consumer_root,
            consumer_site_id=None,
            incidence=root_incidence,
            covered_obligations=exact_obligations,
        )

    pending_events: dict[
        tuple[CoordinateDomain, tuple[ReadinessProducer, ...]],
        ReadinessEvent,
    ] = {}

    def add_producer_key_events(
        *,
        consumer_root: int,
        consumer_site_id: int | None,
        relations: list[
            tuple[
                tuple[int, int | None, CoordinateDomain],
                Incidence,
                frozenset[DependencyObligation],
            ]
        ],
    ) -> None:
        """Keep finer readiness keys when a consumer quotient needs fanout."""
        for producer, incidence, obligations in relations:
            producer_root, producer_site_id, producer_domain = producer
            readiness_key_domain = dataclasses.replace(
                producer_domain,
                kind="event",
                identity=None,
            )
            key_incidence = incidence.reversed()
            key_incidence = (
                None
                if key_incidence is None
                else key_incidence.rename_domains(
                    readiness_key_domain, incidence.items_by_key.source_domain
                )
            )
            if key_incidence is None:
                continue
            _record_readiness_event(
                pending_events,
                readiness_key_domain=readiness_key_domain,
                producers=(
                    ReadinessProducer(
                        producer_root,
                        Incidence.complete(
                            readiness_key_domain, producer_domain, identity=True
                        ),
                        producer_site_id,
                    ),
                ),
                consumers=(
                    ReadinessConsumer(
                        consumer_root,
                        key_incidence,
                        0,
                        obligations,
                        consumer_site_id,
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
                Incidence,
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
            incidence, first_point = relation_points[0]
            obligations = {first_point}
            for next_incidence, obligation in relation_points[1:]:
                if not charge(
                    1
                    + len(incidence.items_by_key.pieces)
                    + len(next_incidence.items_by_key.pieces)
                ):
                    quotient_is_supported = False
                    break
                union = Incidence.union_all((incidence, next_incidence))
                if union is None:
                    quotient_is_supported = False
                    break
                incidence = union
                obligations.add(obligation)
            if not quotient_is_supported:
                break
            used_axes = incidence.items_by_key.source_axes_affecting_targets()
            if used_axes is None:
                quotient_is_supported = False
                break
            readiness_key_axis_set.update(used_axes)
            merged_relations.append((producer, incidence, frozenset(obligations)))

        if not quotient_is_supported:
            add_producer_key_events(
                consumer_root=consumer_root,
                consumer_site_id=consumer_site_id,
                relations=[
                    (producer, incidence, frozenset((obligation,)))
                    for producer, relation_points in sorted(
                        producers.items(),
                        key=lambda item: (
                            item[0][0],
                            -1 if item[0][1] is None else item[0][1],
                        ),
                    )
                    for incidence, obligation in relation_points
                ],
            )
            continue

        if any(
            left_points & right_points
            for left_index, (_left, _left_incidence, left_points) in enumerate(
                merged_relations
            )
            for _right, _right_incidence, right_points in merged_relations[
                left_index + 1 :
            ]
        ):
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
        consumer_counts = consumer_domain.axis_count_expressions
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
        partition = KeyPartition.projection(consumer_domain, readiness_key_domain)
        if partition is None:
            add_producer_key_events(
                consumer_root=consumer_root,
                consumer_site_id=consumer_site_id,
                relations=merged_relations,
            )
            continue
        consumer_group_order = DenseTaskOrder.from_pid(
            consumer_domain,
            tuple(
                axis
                for axis in consumer_domain.axis_order
                if axis not in readiness_key_axis_set
            )
            + readiness_key_axes,
        )
        event_producers: list[ReadinessProducer] = []
        covered_obligations: set[DependencyObligation] = set()
        for producer, incidence, relation_points in merged_relations:
            producer_root, producer_site_id, _producer_domain = producer
            if not charge(
                1
                + len(incidence.items_by_key.pieces)
                * len(partition.fine_keys_by_coarse_key.pieces)
            ):
                break
            producer_incidence = incidence.coarsen(partition)
            if producer_incidence is None:
                break
            event_producers.append(
                ReadinessProducer(producer_root, producer_incidence, producer_site_id)
            )
            covered_obligations.update(relation_points)
        else:
            _record_readiness_event(
                pending_events,
                readiness_key_domain=readiness_key_domain,
                producers=tuple(event_producers),
                consumers=(
                    ReadinessConsumer(
                        consumer_root,
                        partition.as_incidence(grouped_items=consumer_group_order),
                        0,
                        frozenset(covered_obligations),
                        consumer_site_id,
                    ),
                ),
            )
            continue

        add_producer_key_events(
            consumer_root=consumer_root,
            consumer_site_id=consumer_site_id,
            relations=merged_relations,
        )

    represented_obligations = {
        obligation
        for event in pending_events.values()
        for consumer in event.consumers
        for obligation in consumer.covered_obligations
    }
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
        for consumer_id, (consumer_root, obligations) in enumerate(
            sorted(obligations_by_consumer.items())
        ):
            consumers.append(
                ReadinessConsumer(
                    consumer_root,
                    Incidence.complete(
                        readiness_key_domain, root_domains[consumer_root]
                    ),
                    consumer_id,
                    frozenset(obligations),
                )
            )
        _record_readiness_event(
            pending_events,
            readiness_key_domain=readiness_key_domain,
            producers=(
                ReadinessProducer(
                    producer_root,
                    Incidence.complete(readiness_key_domain, producer_domain),
                ),
            ),
            consumers=tuple(consumers),
        )
    events = tuple(pending_events.values())
    event_obligations = frozenset(
        obligation
        for event in events
        for consumer in event.consumers
        for obligation in consumer.covered_obligations
    )
    manifest_obligations = frozenset(
        obligation
        for _pair, obligations in obligations_by_root_pair
        for obligation in obligations
    )
    if event_obligations != manifest_obligations:
        raise AssertionError(
            "readiness events must cover the dependency manifest exactly"
        )
    return events


def derive_final_arrival_continuations(
    readiness_graph: ReadinessGraph,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
) -> tuple[FinalArrivalContinuation, ...]:
    """Derive complete one-task-per-readiness-key continuation candidates."""
    required_obligations_by_root: dict[int, set[DependencyObligation]] = {}
    for event in readiness_graph.events:
        for readiness_consumer in event.consumers:
            required_obligations_by_root.setdefault(
                readiness_consumer.consumer_root, set()
            ).update(readiness_consumer.covered_obligations)

    candidates: list[tuple[int, int]] = []
    for plan in readiness_counters:
        event_id = plan.readiness_key_domain.identity
        if (
            event_id is None
            or not 0 <= event_id < len(readiness_graph.events)
            or len(plan.consumers) != 1
            or any(producer.producer_site_id is not None for producer in plan.producers)
        ):
            continue
        if len(readiness_graph.event(event_id).consumers) != 1:
            continue
        (readiness_consumer,) = plan.consumers
        if readiness_consumer.consumer_site_id is not None:
            continue
        candidate_plan = dataclasses.replace(plan, continuation_consumer_index=0)
        if not _supports_emitted_counter_plan_lowering(
            candidate_plan,
            readiness_graph.root_domains,
        ):
            continue
        if not readiness_consumer.covered_obligations.issuperset(
            required_obligations_by_root.get(readiness_consumer.consumer_root, ())
        ):
            continue
        candidates.append((readiness_consumer.consumer_root, event_id))
    return tuple(
        FinalArrivalContinuation(event_id, 0) for _root, event_id in sorted(candidates)
    )


def _task_step_relations(
    pipeline_plan: StaticPipelinePlan,
    charge: Callable[[int], bool],
) -> tuple[CoordinateRelation | None, ...] | None:
    """Return logical-task-to-wave functions from schedule converses."""
    result: list[CoordinateRelation | None] = []
    for root in range(len(pipeline_plan.execution_orders)):
        if root in pipeline_plan.continuation_roots:
            result.append(None)
            continue
        task_steps = _root_task_wave_relation(pipeline_plan, root, charge)
        if task_steps is None:
            return None
        result.append(task_steps)
    return tuple(result)


def _keys_by_consumer_root_task(
    readiness_graph: ReadinessGraph,
    consumer: ReadinessConsumer,
) -> Incidence | None:
    """Project every nested checkpoint requirement onto its owning CTA."""
    root_domain = readiness_graph.root_domains[consumer.consumer_root]
    if consumer.keys_by_consumer.source_domain == root_domain:
        return consumer.incidence
    return consumer.incidence.project_items(root_domain)


def _consumer_major_producer_order(
    readiness_graph: ReadinessGraph,
    pipeline_plan: StaticPipelinePlan,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    *,
    static_producers: _StaticProducerResolver,
    excluded_roots: frozenset[int],
    charge: Callable[[int], bool],
) -> StaticPipelinePlan:
    """Prepare all exact root-local traversals atomically."""
    incidence_pieces = sum(
        len(endpoint.incidence.items_by_key.pieces)
        + len(
            ()
            if endpoint.incidence.keys_by_item is None
            else endpoint.incidence.keys_by_item.pieces
        )
        for counter in readiness_counters
        for endpoint in (*counter.producers, *counter.consumers)
    )
    task_count = sum(order.task_count for order in pipeline_plan.execution_orders)
    if not charge(1 + incidence_pieces * (incidence_pieces + task_count)):
        return pipeline_plan

    def exact_task_order(
        incidence: Incidence | None,
        root: int,
    ) -> DenseTaskOrder | None:
        task_domain = readiness_graph.root_domains[root]
        if incidence is None or incidence.items_by_key.target_domain != task_domain:
            return None
        return incidence.grouped_items

    def unique_orders(
        candidates: dict[int, list[DenseTaskOrder | None]],
    ) -> dict[int, DenseTaskOrder]:
        result: dict[int, DenseTaskOrder] = {}
        for root, root_candidates in candidates.items():
            if not root_candidates or any(item is None for item in root_candidates):
                continue
            order = pipeline_plan.execution_orders[root]
            if (
                root in pipeline_plan.continuation_roots
                or order.task_count < pipeline_plan.worker_count
                or order.task_count != readiness_graph.root_domains[root].size
            ):
                continue
            exact_candidates = cast("list[DenseTaskOrder]", root_candidates)
            reference_ordinal = exact_candidates[0].ordinal_by_task
            if any(
                not candidate.ordinal_by_task.covers(reference_ordinal)
                or not reference_ordinal.covers(candidate.ordinal_by_task)
                for candidate in exact_candidates[1:]
            ):
                continue
            result[root] = exact_candidates[0]
        return result

    incoming_counter_roots: set[int] = set()
    admission_candidates: dict[int, list[DenseTaskOrder | None]] = {}
    for consumer in (
        consumer
        for counter in readiness_counters
        for consumer_index, consumer in enumerate(counter.consumers)
        if consumer_index != counter.continuation_consumer_index
    ):
        consumer_root = consumer.consumer_root
        if consumer_root in excluded_roots:
            continue
        incoming_counter_roots.add(consumer_root)
        consumer_keys = _keys_by_consumer_root_task(readiness_graph, consumer)
        if consumer.consumer_site_id is not None:
            root_domain = readiness_graph.root_domains[consumer_root]
            site_domain = consumer.keys_by_consumer.source_domain
            nested_axes = nested_logical_axes(root_domain, site_domain)
            placement = (
                None
                if len(nested_axes) != 1
                else Incidence.fixed_embedding(
                    root_domain, site_domain, {nested_axes[0]: 0}
                )
            )
            placement = None if placement is None else placement.reversed()
            consumer_keys = (
                None if placement is None else consumer.incidence.then(placement)
            )
        admission_candidates.setdefault(consumer_root, []).append(
            exact_task_order(consumer_keys, consumer_root)
        )
    admission_orders = unique_orders(admission_candidates)

    completion_candidates: dict[int, list[DenseTaskOrder | None]] = {}
    for plan in readiness_counters:
        static_relations = static_producers(plan.producers)
        if static_relations is None:
            continue
        for consumer in plan.consumers:
            consumer_root = consumer.consumer_root
            consumer_order = admission_orders.get(
                consumer_root, pipeline_plan.execution_orders[consumer_root]
            )
            consumer_keys = _keys_by_consumer_root_task(readiness_graph, consumer)
            ordered_consumer_keys = (
                None
                if consumer_keys is None
                else consumer_keys.reindex_items(consumer_order)
            )
            ordered_consumer_keys = (
                None
                if ordered_consumer_keys is None
                else ordered_consumer_keys.reversed()
            )
            for producer_root, incidence in static_relations:
                if (
                    producer_root in excluded_roots
                    or producer_root in incoming_counter_roots
                ):
                    continue
                producers_by_consumer = (
                    None
                    if ordered_consumer_keys is None
                    else ordered_consumer_keys.then(incidence)
                )
                completion_candidates.setdefault(producer_root, []).append(
                    None
                    if producers_by_consumer is None
                    else producers_by_consumer.with_key_major_order().grouped_items
                )
    replacements = {**admission_orders, **unique_orders(completion_candidates)}
    if not replacements:
        return pipeline_plan
    try:
        return dataclasses.replace(
            pipeline_plan,
            execution_orders=tuple(
                itertools.starmap(
                    replacements.get, enumerate(pipeline_plan.execution_orders)
                )
            ),
        )
    except ValueError:
        return pipeline_plan


def _counter_prerequisite_has_progress_precedence(
    *,
    pipeline_plan: StaticPipelinePlan,
    terminal_producers: tuple[tuple[int, Incidence], ...],
    consumer_root: int,
    consumer_keys: CoordinateRelation,
    task_steps: tuple[CoordinateRelation | None, ...],
    charge: Callable[[int], bool],
) -> bool:
    """Prove strict static worker-rank precedence for one counter wait."""
    consumer_steps = task_steps[consumer_root]
    if consumer_steps is None:
        return False
    for producer_root, incidence in terminal_producers:
        if task_steps[producer_root] is None:
            return False
        key_frontier = _maximum_root_wave_by_key(
            pipeline_plan, producer_root, incidence.items_by_key, charge
        )
        if key_frontier is None or not charge(
            1
            + (
                len(consumer_keys.pieces)
                + len(key_frontier.pieces)
                + len(consumer_steps.pieces)
            )
            ** 2
        ):
            return False
        values = consumer_keys.then(key_frontier)
        if values is None:
            frontier = consumer_keys.max_target_value_by_source(key_frontier)
        else:
            frontier = values.max_target_value_by_source(
                CoordinateRelation.identity(
                    key_frontier.target_domain,
                    key_frontier.target_domain,
                )
            )
        if frontier is None or frontier.canonical_single_valued() is None:
            return False
        if frontier.source_support_is_empty() is True:
            continue
        if not frontier.is_pointwise_strictly_less_than_where_defined(consumer_steps):
            return False
    return True


def _deterministic_topological_order(
    successors: list[set[int]],
) -> tuple[int, ...] | None:
    indegree = [0] * len(successors)
    for node_successors in successors:
        for successor in node_successors:
            indegree[successor] += 1
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for successor in sorted(successors[node]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, successor)
    return tuple(order) if len(order) == len(successors) else None


def _schedule_is_progress_safe(
    pipeline_plan: StaticPipelinePlan,
    readiness_graph: ReadinessGraph,
    obligations_by_root_pair: ObligationsByRootPair,
    continuation_by_root: dict[int, FinalArrivalContinuation],
    static_producers: _StaticProducerResolver,
    charge: Callable[[int], bool],
) -> bool:
    """Prove cross-root order and same-root worker-rank progress."""
    continuation_plans = {
        (plan.readiness_key_domain.identity, consumer.consumer_id): (plan, consumer)
        for plan in pipeline_plan.readiness_counters
        if plan.continuation_consumer_index is not None
        for consumer in (plan.consumers[plan.continuation_consumer_index],)
    }
    scheduled_roots = frozenset(pipeline_plan.resident_roots)
    for continuation in continuation_by_root.values():
        counter_and_consumer = continuation_plans.get(
            (continuation.event_id, continuation.consumer_index)
        )
        counter, consumer = (
            (None, None) if counter_and_consumer is None else counter_and_consumer
        )
        required_obligations = frozenset(
            obligation
            for (
                producer_root,
                consumer_root,
            ), obligations in obligations_by_root_pair
            if consumer_root == (-1 if consumer is None else consumer.consumer_root)
            for obligation in obligations
        )
        terminal_producers = (
            None if counter is None else static_producers(counter.producers)
        )
        if (
            consumer is None
            or not consumer.keys_by_consumer.is_total_function()
            or not required_obligations <= consumer.covered_obligations
            or terminal_producers is None
            or any(
                producer_root not in scheduled_roots
                or producer_root >= consumer.consumer_root
                for producer_root, incidence in terminal_producers
                if incidence.items_by_key.source_support_is_empty() is not True
            )
        ):
            return False

    task_steps: tuple[CoordinateRelation | None, ...] | None = None
    prerequisites = itertools.chain(
        (
            (consumer_root, producer_root, None, None)
            for producer_root, consumer_root in sorted(pipeline_plan.root_barrier_edges)
        ),
        (
            (consumer.consumer_root, None, plan, consumer)
            for plan in pipeline_plan.readiness_counters
            for index, consumer in enumerate(plan.consumers)
            if index != plan.continuation_consumer_index
        ),
    )
    for consumer_root, barrier_producer_root, plan, consumer in prerequisites:
        if consumer_root not in scheduled_roots:
            return False
        if barrier_producer_root is not None:
            producer_root = barrier_producer_root
            producer_domain = readiness_graph.root_domains[producer_root]
            root_incidence = Incidence.complete(
                producer_domain, producer_domain, identity=True
            )
            terminal_producers = (
                ((producer_root, root_incidence),)
                if producer_root not in continuation_by_root
                else _contract_static_producer_incidences(
                    readiness_graph,
                    ((producer_root, None, root_incidence),),
                    continuation_by_root,
                    charge,
                )
            )
        else:
            assert plan is not None and consumer is not None
            terminal_producers = static_producers(plan.producers)
        if terminal_producers is None:
            return False
        same_root = False
        for producer_root, incidence in terminal_producers:
            if incidence.items_by_key.source_support_is_empty() is True:
                continue
            if producer_root not in scheduled_roots:
                return False
            if producer_root < consumer_root:
                continue
            if producer_root != consumer_root:
                return False
            same_root = True

        if not same_root:
            continue
        if plan is None:
            return False
        assert consumer is not None
        consumer_keys = _keys_by_consumer_root_task(readiness_graph, consumer)
        if consumer_keys is None:
            return False
        if task_steps is None:
            task_steps = _task_step_relations(pipeline_plan, charge)
        if task_steps is None or not _counter_prerequisite_has_progress_precedence(
            pipeline_plan=pipeline_plan,
            terminal_producers=terminal_producers,
            consumer_root=consumer.consumer_root,
            consumer_keys=cast("CoordinateRelation", consumer_keys.keys_by_item),
            task_steps=task_steps,
            charge=charge,
        ):
            return False
    return True


def _supports_emitted_counter_plan_lowering(
    plan: ReadinessCounterPlan,
    root_domains: tuple[CoordinateDomain, ...],
) -> bool:
    """Return whether a counter has one exact supported concrete lowering."""
    try:
        if (
            plan.readiness_key_count <= 0
            or _arrival_count_bounds(plan.producers) is None
        ):
            return False
    except ValueError:
        return False

    def endpoint_has_supported_domain(
        root: int,
        site_id: int | None,
        domain: CoordinateDomain,
    ) -> bool:
        if not 0 <= root < len(root_domains):
            return False
        root_domain = root_domains[root]
        if site_id is None:
            return domain == root_domain
        root_counts = root_domain.axis_count_expressions
        site_counts = domain.axis_count_expressions
        return (
            domain.kind == "site"
            and domain.identity == site_id
            and len(nested_logical_axes(root_domain, domain)) == 1
            and all(
                axis in site_counts and site_counts[axis] == count
                for axis, count in root_counts.items()
            )
        )

    continuation_index = plan.continuation_consumer_index
    if not plan.consumers or (
        continuation_index is not None
        and not 0 <= continuation_index < len(plan.consumers)
    ):
        return False
    for producer in plan.producers:
        if not endpoint_has_supported_domain(
            producer.producer_root,
            producer.producer_site_id,
            producer.incidence.items_by_key.target_domain,
        ) or not _supports_readiness_counter_lowering(producer):
            return False
    for consumer in plan.consumers:
        if (
            not endpoint_has_supported_domain(
                consumer.consumer_root,
                consumer.consumer_site_id,
                consumer.keys_by_consumer.source_domain,
            )
            or consumer.keys_by_consumer.canonical_single_valued() is None
        ):
            return False

    if any(consumer.consumer_site_id is not None for consumer in plan.consumers) and (
        len(plan.consumers) != 1 or continuation_index is not None
    ):
        return False

    if continuation_index is None:
        return True

    continuation_consumer = plan.consumers[continuation_index]
    fan_in = plan.uniform_arrival_count()
    return (
        fan_in is not None
        and fan_in > 0
        and continuation_consumer.keys_by_consumer.is_total_function()
        and continuation_consumer.incidence.items_by_key.is_total_function()
    )


def _finalize_emitted_synchronization(
    *,
    readiness_graph: ReadinessGraph,
    obligations_by_root_pair: ObligationsByRootPair,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
) -> tuple[tuple[ReadinessCounterPlan, ...], frozenset[tuple[int, int]]]:
    """Select fallback barriers and remove counter consumers they subsume."""
    readiness_counters = tuple(
        plan
        for plan in readiness_counters
        if _supports_emitted_counter_plan_lowering(plan, readiness_graph.root_domains)
    )
    covered_obligations = _covered_obligations(readiness_counters)
    root_barrier_edges = _select_root_barrier_edges(
        obligations_by_root_pair=obligations_by_root_pair,
        covered_obligations=covered_obligations,
    )
    root_order_edges = _root_barrier_reachability(root_barrier_edges)
    retained: list[ReadinessCounterPlan] = []
    for plan in readiness_counters:
        kept = tuple(
            (index, consumer)
            for index, consumer in enumerate(plan.consumers)
            if index == plan.continuation_consumer_index
            or not all(
                (producer.producer_root, consumer.consumer_root) in root_order_edges
                for producer in plan.producers
            )
        )
        if not kept:
            continue
        retained.append(
            dataclasses.replace(
                plan,
                consumers=tuple(consumer for _index, consumer in kept),
                continuation_consumer_index=next(
                    (
                        new
                        for new, (old, _consumer) in enumerate(kept)
                        if old == plan.continuation_consumer_index
                    ),
                    None,
                ),
            )
        )
    retained_counters = tuple(retained)
    retained_obligations = _covered_obligations(retained_counters)
    for pair, obligations in obligations_by_root_pair:
        if pair in root_order_edges:
            continue
        uncovered = tuple(sorted(obligations - retained_obligations))
        if uncovered:
            raise exc.CrossLoopSchedulingError(
                f"{pair[0]}->{pair[1]} has no cross-loop "
                f"synchronization path for dependencies {uncovered!r}"
            )
    return retained_counters, root_barrier_edges


def _try_finalize_pipeline_proposal(
    *,
    readiness_graph: ReadinessGraph,
    obligations_by_root_pair: ObligationsByRootPair,
    configured_orders: tuple[DenseTaskOrder, ...],
    worker_count: int,
    readiness_counters: tuple[ReadinessCounterPlan, ...],
    root_barrier_edges: frozenset[tuple[int, int]],
    charge: Callable[[int], bool],
) -> StaticPipelinePlan | None:
    """Freeze one canonical placement, then lower its final counters."""
    emitted_continuations = _emitted_final_arrival_continuations(
        readiness_graph,
        readiness_counters,
    )
    if emitted_continuations is None:
        return None
    continuation_by_root = {
        readiness_graph.events[continuation.event_id]
        .consumers[continuation.consumer_index]
        .consumer_root: continuation
        for continuation in emitted_continuations
    }
    continuation_roots = frozenset(continuation_by_root)
    static_producer_cache: dict[
        tuple[ReadinessProducer, ...], tuple[tuple[int, Incidence], ...] | None
    ] = {}

    def static_producers(
        producers: tuple[ReadinessProducer, ...],
    ) -> tuple[tuple[int, Incidence], ...] | None:
        if producers not in static_producer_cache:
            static_producer_cache[producers] = (
                None
                if not charge(len(producers))
                else _contract_static_producer_incidences(
                    readiness_graph,
                    tuple(
                        (
                            producer.producer_root,
                            producer.producer_site_id,
                            producer.incidence,
                        )
                        for producer in producers
                    ),
                    continuation_by_root,
                    charge,
                )
            )
        return static_producer_cache[producers]

    try:
        ownership_base = StaticPipelinePlan(
            worker_count=worker_count,
            execution_orders=configured_orders,
            body_orders=configured_orders,
            readiness_counters=readiness_counters,
            root_barrier_edges=root_barrier_edges,
        )
    except (ValueError, exc.CrossLoopSchedulingError):
        return None

    exact_covered_obligations = _covered_obligations(readiness_counters)

    def finalize(candidate: StaticPipelinePlan) -> StaticPipelinePlan | None:
        if not _schedule_is_progress_safe(
            candidate,
            readiness_graph,
            obligations_by_root_pair,
            continuation_by_root,
            static_producers,
            charge,
        ):
            return None
        compact_counters = _compact_nested_loop_counters_for_schedule(
            readiness_graph,
            candidate,
            readiness_counters,
            static_producers,
            charge,
        )
        for emitted_counters in dict.fromkeys((compact_counters, readiness_counters)):
            if (
                _covered_obligations(emitted_counters) != exact_covered_obligations
                or any(
                    not _supports_emitted_counter_plan_lowering(
                        plan, readiness_graph.root_domains
                    )
                    for plan in emitted_counters
                )
                or _emitted_final_arrival_continuations(
                    readiness_graph,
                    emitted_counters,
                )
                != emitted_continuations
                or (
                    emitted_counters != readiness_counters
                    and not _schedule_is_progress_safe(
                        dataclasses.replace(
                            candidate,
                            readiness_counters=emitted_counters,
                        ),
                        readiness_graph,
                        obligations_by_root_pair,
                        continuation_by_root,
                        static_producers,
                        charge,
                    )
                )
            ):
                continue
            try:
                return dataclasses.replace(
                    candidate, readiness_counters=emitted_counters
                )
            except (ValueError, exc.CrossLoopSchedulingError):
                continue
        return None

    baseline = finalize(ownership_base)
    if baseline is None:
        return None
    prepared = _consumer_major_producer_order(
        readiness_graph,
        ownership_base,
        readiness_counters,
        static_producers=static_producers,
        excluded_roots=continuation_roots,
        charge=charge,
    )
    return baseline if prepared == ownership_base else finalize(prepared) or baseline


def build_static_pipeline_plan(
    *,
    dependency_graph: TileDependencyGraph,
    root_task_orders: tuple[DenseTaskOrder, ...],
    site_domains: tuple[CoordinateDomain | None, ...],
    worker_count: int,
    publishable_site_ids: frozenset[int] | None = None,
    continuation_ineligible_roots: frozenset[int] = frozenset(),
    prove_nonnegative: Callable[[object], bool] | None = None,
) -> StaticPipelinePlan:
    """Derive all generic readiness strategies without inspecting root bodies."""
    charge = _new_relation_work_budget()
    root_identities = tuple(
        order.tasks_by_ordinal.target_domain.identity for order in root_task_orders
    )
    if any(
        order.tasks_by_ordinal.source_domain.kind != "task_order"
        or order.tasks_by_ordinal.target_domain.kind != "site"
        or order.tasks_by_ordinal.source_domain.identity
        != order.tasks_by_ordinal.target_domain.identity
        or order.task_count <= 0
        for order in root_task_orders
    ) or len({identity for identity in root_identities if identity is not None}) != sum(
        identity is not None for identity in root_identities
    ):
        raise exc.InvalidConfig(
            "cross_loop_pipeline='static' requires a fixed task "
            "capacity and task order; specialize the schedule-affecting "
            "capacity while keeping symbolic layout maps and runtime metadata "
            "unspecialized"
        )
    root_domains = tuple(
        order.tasks_by_ordinal.target_domain for order in root_task_orders
    )
    readiness_graph = ReadinessGraph(
        root_domains,
        _build_readiness_events(
            dependency_graph,
            root_domains=root_domains,
            site_domains=site_domains,
            publishable_site_ids=publishable_site_ids,
            prove_nonnegative=prove_nonnegative,
            charge=charge,
        ),
    )
    obligations_by_root_pair = dependency_graph.obligations_by_root_pair()
    nested_loop_counters = collect_nested_loop_scheduling_counters(
        readiness_graph, charge
    )
    nested_loop_obligations = _covered_obligations(nested_loop_counters)

    all_resident_candidate_counters = (
        *choose_readiness_counters(
            readiness_graph,
            (),
            excluded_obligations=nested_loop_obligations,
            charge=charge,
        ),
        *nested_loop_counters,
    )
    try:
        all_resident_counters, all_resident_barriers = (
            _finalize_emitted_synchronization(
                readiness_graph=readiness_graph,
                obligations_by_root_pair=obligations_by_root_pair,
                readiness_counters=all_resident_candidate_counters,
            )
        )
    except (ValueError, exc.CrossLoopSchedulingError) as error:
        raise exc.InvalidConfig(
            f"the num_sm_multiplier grid of {worker_count} workers does not "
            "admit complete cross-loop synchronization"
        ) from error
    all_resident_plan = _try_finalize_pipeline_proposal(
        readiness_graph=readiness_graph,
        obligations_by_root_pair=obligations_by_root_pair,
        configured_orders=root_task_orders,
        worker_count=worker_count,
        readiness_counters=all_resident_counters,
        root_barrier_edges=all_resident_barriers,
        charge=charge,
    )
    if all_resident_plan is None and all_resident_counters:
        with contextlib.suppress(ValueError, exc.CrossLoopSchedulingError):
            all_resident_counters, all_resident_barriers = (
                _finalize_emitted_synchronization(
                    readiness_graph=readiness_graph,
                    obligations_by_root_pair=obligations_by_root_pair,
                    readiness_counters=(),
                )
            )
            all_resident_plan = _try_finalize_pipeline_proposal(
                readiness_graph=readiness_graph,
                obligations_by_root_pair=obligations_by_root_pair,
                configured_orders=root_task_orders,
                worker_count=worker_count,
                readiness_counters=all_resident_counters,
                root_barrier_edges=all_resident_barriers,
                charge=charge,
            )
    if all_resident_plan is None:
        raise exc.InvalidConfig(
            f"the num_sm_multiplier grid of {worker_count} workers does not "
            "admit a progress-safe all-resident cross-loop schedule"
        )

    cheaper_counters = _without_wave_dominated_nested_counters(all_resident_plan)
    if cheaper_counters != all_resident_plan.readiness_counters:
        with contextlib.suppress(ValueError, exc.CrossLoopSchedulingError):
            cheaper_counters, cheaper_barriers = _finalize_emitted_synchronization(
                readiness_graph=readiness_graph,
                obligations_by_root_pair=obligations_by_root_pair,
                readiness_counters=cheaper_counters,
            )
            cheaper_plan = _try_finalize_pipeline_proposal(
                readiness_graph=readiness_graph,
                obligations_by_root_pair=obligations_by_root_pair,
                configured_orders=root_task_orders,
                worker_count=worker_count,
                readiness_counters=cheaper_counters,
                root_barrier_edges=cheaper_barriers,
                charge=charge,
            )
            if cheaper_plan is not None:
                all_resident_counters = cheaper_counters
                all_resident_barriers = cheaper_barriers
                all_resident_plan = cheaper_plan

    continuation_candidates = derive_final_arrival_continuations(
        readiness_graph, all_resident_counters
    )
    causal_relations = dict(
        _root_causal_prerequisite_relations(readiness_graph, charge)
        if continuation_candidates
        else ()
    )
    continuations = choose_final_arrival_continuations(
        readiness_graph,
        continuation_candidates,
        all_resident_plan,
        excluded_roots=continuation_ineligible_roots,
        causal_relations=causal_relations,
        charge=charge,
    )
    continuation_counters = _assign_final_arrival_continuations(
        readiness_graph,
        all_resident_counters,
        continuations,
    )
    proposal = all_resident_plan
    if continuations and continuation_counters is not None:
        candidate = _try_finalize_pipeline_proposal(
            readiness_graph=readiness_graph,
            obligations_by_root_pair=obligations_by_root_pair,
            configured_orders=root_task_orders,
            worker_count=worker_count,
            readiness_counters=continuation_counters,
            root_barrier_edges=all_resident_barriers,
            charge=charge,
        )
        final_continuations = (
            None
            if candidate is None
            else _emitted_final_arrival_continuations(
                readiness_graph, candidate.readiness_counters
            )
        )
        if (
            candidate is not None
            and final_continuations is not None
            and choose_final_arrival_continuations(
                readiness_graph,
                final_continuations,
                candidate,
                excluded_roots=continuation_ineligible_roots,
                causal_relations=causal_relations,
                charge=charge,
            )
            == final_continuations
        ):
            proposal = candidate
    return proposal


def _select_root_barrier_edges(
    *,
    obligations_by_root_pair: ObligationsByRootPair,
    covered_obligations: frozenset[DependencyObligation],
) -> frozenset[tuple[int, int]]:
    """Choose the minimal source-ordered root-barrier fallback edges."""
    selected_edges: set[tuple[int, int]] = set()
    for pair, obligations in sorted(
        obligations_by_root_pair,
        key=lambda item: (
            item[0][1] - item[0][0],
            item[0][0],
            item[0][1],
        ),
    ):
        producer_root, consumer_root = pair
        if obligations <= covered_obligations:
            continue
        if producer_root >= consumer_root:
            raise exc.CrossLoopSchedulingError(
                "a whole-root barrier can cover only a strict source-ordered "
                f"dependency, got {producer_root}->{consumer_root}"
            )
        if pair in _root_barrier_reachability(selected_edges):
            continue
        selected_edges.add(pair)
    return frozenset(selected_edges)


def _root_barrier_reachability(
    edges: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> set[tuple[int, int]]:
    closure = set(edges)
    for middle in sorted({root for edge in edges for root in edge}):
        incoming = {source for source, target in closure if target == middle}
        outgoing = {target for source, target in closure if source == middle}
        closure.update(itertools.product(incoming, outgoing))
    return closure
