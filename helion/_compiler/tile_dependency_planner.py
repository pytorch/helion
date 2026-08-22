from __future__ import annotations

import dataclasses
import math
from operator import itemgetter
from typing import Literal

from .. import exc
from .cross_loop_dependencies import AffinePredecessorAxis
from .cross_loop_dependencies import AllocationRegion
from .cross_loop_dependencies import CrossLoopAccess
from .cross_loop_dependencies import CrossLoopDependencyEdge
from .cross_loop_dependencies import CrossLoopDependencyPlan
from .cross_loop_dependencies import TaskFamily
from .cross_loop_dependencies import TileDependencyKind
from .cross_loop_dependencies import UniformTaskPartition
from .cross_loop_dependencies import WaitSpec
from .cross_loop_dependencies import access_task_region
from .cross_loop_dependencies import allocation_regions_may_overlap
from .cross_loop_dependencies import prove_uniform_task_partition

TILE_DEPENDENCY_FRONTIER_CONFIG = "tile_dependency_frontier"
TILE_DEPENDENCY_FRONTIER_DEFAULT = -1


@dataclasses.dataclass(frozen=True)
class AccessProgramPoint:
    """A lowered access location with its logical loop coordinates.

    The dependency proof names axes by source-level block ID.  The expressions
    here are used only to materialize an already-proven event key at the
    explicit access marker; they are never used to infer dependency geometry.
    """

    coordinate_items: tuple[tuple[int, str], ...] | None
    loop_id: int | None
    loop_depth: int
    root_statement_index: int

    @property
    def coordinates(self) -> dict[int, str] | None:
        if self.coordinate_items is None:
            return None
        return dict(self.coordinate_items)


@dataclasses.dataclass(frozen=True)
class InstantiatedTaskFamily:
    """A logical task family instantiated for one kernel configuration."""

    logical_axis_order: tuple[int, ...]
    physical_axis_order: tuple[int, ...]
    axis_counts_items: tuple[tuple[int, int], ...]
    block_sizes_items: tuple[tuple[int, int], ...]

    @property
    def axis_counts(self) -> dict[int, int]:
        return dict(self.axis_counts_items)

    @property
    def block_sizes(self) -> dict[int, int]:
        return dict(self.block_sizes_items)

    @property
    def task_count(self) -> int:
        return math.prod(count for _, count in self.axis_counts_items)


@dataclasses.dataclass(frozen=True)
class AccessCohortPlan:
    """A contiguous coarsening of one access-local readiness relation."""

    event_id: int
    producer_root: int
    consumer_root: int
    axes: tuple[AffinePredecessorAxis, ...]
    access_ids: tuple[int, ...]
    producer_stream_axis: int
    consumer_stream_axis: int
    consumer_loop_id: int
    consumer_stream_coordinate: str | None
    stream_count: int
    stream_task_granularity: int
    stage_sizes: tuple[int, ...]
    outer_producer_axes: tuple[int, ...]

    @property
    def is_per_coordinate(self) -> bool:
        return self.consumer_stream_coordinate is not None

    @property
    def milestone_count(self) -> int:
        return self.stream_count if self.is_per_coordinate else len(self.stage_sizes)

    @property
    def stage_offsets(self) -> tuple[int, ...]:
        if self.is_per_coordinate:
            raise AssertionError("per-coordinate readiness has no static stages")
        result = [0]
        for size in self.stage_sizes:
            result.append(result[-1] + size)
        return tuple(result)


@dataclasses.dataclass(frozen=True)
class TaskToKeySegment:
    """One compact run of a flattened task-to-event-key mapping."""

    task_begin: int
    task_count: int
    tasks_per_key: int
    first_key: int
    key_stride: int
    key_period: int | None = None


@dataclasses.dataclass(frozen=True)
class KeyedEventContributorPlan:
    """One root's contributions to a keyed readiness event."""

    source_event_ids: tuple[int, ...]
    producer_root: int
    task_to_key: tuple[int | None, ...]
    partition: UniformTaskPartition | None = None

    @property
    def task_to_key_segments(self) -> tuple[TaskToKeySegment, ...]:
        return _compress_task_to_key(self.task_to_key)

    @property
    def expected_arrivals(self) -> int:
        arrivals: dict[int, int] = {}
        for key in self.task_to_key:
            if key is not None:
                arrivals[key] = arrivals.get(key, 0) + 1
        counts = set(arrivals.values())
        if len(counts) != 1:
            raise ValueError("keyed event contributor has nonuniform fan-in")
        return counts.pop()


@dataclasses.dataclass(frozen=True)
class KeyedEventPlan:
    """A logical key space receiving contributions from one or more roots.

    Each contributor has an independently proved task-to-key relation. The
    expected count is derived by summing those relations; the event therefore
    represents both ordinary continuations and generic multi-predecessor joins.
    ``on_ready_root`` remains dispatch policy rather than dependency semantics.
    """

    key_root: int
    contributors: tuple[KeyedEventContributorPlan, ...]
    consumer_key_by_task: tuple[int, ...]
    on_ready_root: int | None = None

    @property
    def consumer_task_to_key_segments(self) -> tuple[TaskToKeySegment, ...]:
        return _compress_task_to_key(self.consumer_key_by_task)

    @property
    def source_event_ids(self) -> tuple[int, ...]:
        return tuple(
            event_id
            for contributor in self.contributors
            for event_id in contributor.source_event_ids
        )

    @property
    def key_tasks(self) -> int:
        return max(self.consumer_key_by_task, default=-1) + 1

    @property
    def expected_arrivals(self) -> int:
        arrivals = [0] * self.key_tasks
        for contributor in self.contributors:
            for key in contributor.task_to_key:
                if key is not None:
                    arrivals[key] += 1
        counts = set(arrivals)
        if len(counts) != 1:
            raise ValueError("keyed event has nonuniform total fan-in")
        return counts.pop()

    @property
    def is_single_contributor(self) -> bool:
        return len(self.contributors) == 1

    @property
    def single_contributor(self) -> KeyedEventContributorPlan:
        if not self.is_single_contributor:
            raise ValueError("keyed event has multiple contributors")
        return self.contributors[0]

    @property
    def event_id(self) -> int:
        event_ids = self.single_contributor.source_event_ids
        if len(event_ids) != 1:
            raise ValueError("keyed event has no source event")
        return event_ids[0]

    @property
    def producer_root(self) -> int:
        return self.single_contributor.producer_root

    @property
    def partition(self) -> UniformTaskPartition:
        partition = self.single_contributor.partition
        if partition is None:
            raise ValueError("keyed event has no uniform partition")
        return partition


@dataclasses.dataclass(frozen=True)
class ReadinessFrontierPlan:
    """A counted event followed by a concurrently admitted consumer root."""

    event: KeyedEventPlan
    downstream_cohort: AccessCohortPlan
    worker_count: int
    downstream_tasks: int
    initial_stream_tasks: int

    @property
    def tail_producer_tasks(self) -> int:
        return self.event.partition.producer_tasks - self.worker_count

    @property
    def downstream_worker_begin(self) -> int:
        return self.worker_count - self.downstream_tasks


@dataclasses.dataclass(frozen=True)
class ReadinessFrontierSelection:
    """One selected frontier, or its conservative root-completion fallback."""

    plans: tuple[ReadinessFrontierPlan, ...] = ()
    root_completion_key_roots: frozenset[int] = frozenset()


@dataclasses.dataclass(frozen=True)
class RootCompletionWait:
    """A wait for a producer root, placed at entry or a specific access."""

    producer_root: int
    consumer_root: int
    consumer_access_id: int | None
    placement: Literal["root_entry", "access"]


@dataclasses.dataclass(frozen=True)
class GenericSchedulePlan:
    """Pure graph-derived choices consumed by persistent-kernel lowering."""

    task_ready_edges: frozenset[tuple[int, int]]
    root_completion_edges: frozenset[tuple[int, int]]
    task_waits_by_root: dict[int, tuple[WaitSpec, ...]]
    access_cohorts: tuple[AccessCohortPlan, ...]
    counted_events: tuple[KeyedEventPlan, ...]
    readiness_frontiers: tuple[ReadinessFrontierPlan, ...]
    worker_limit: int


def _compress_task_to_key(
    task_to_key: tuple[int | None, ...],
) -> tuple[TaskToKeySegment, ...]:
    runs: list[tuple[int, int, int]] = []
    task = 0
    while task < len(task_to_key):
        key = task_to_key[task]
        if key is None:
            task += 1
            continue
        begin = task
        task += 1
        while task < len(task_to_key) and task_to_key[task] == key:
            task += 1
        runs.append((begin, task - begin, key))

    segments: list[TaskToKeySegment] = []
    run = 0
    while run < len(runs):
        begin, tasks_per_key, first_key = runs[run]
        end_run = run + 1
        key_stride = 0
        if end_run < len(runs):
            next_begin, next_length, next_key = runs[end_run]
            if next_begin == begin + tasks_per_key and next_length == tasks_per_key:
                key_stride = next_key - first_key
                end_run += 1
                while end_run < len(runs):
                    candidate_begin, candidate_length, candidate_key = runs[end_run]
                    if (
                        candidate_begin != begin + (end_run - run) * tasks_per_key
                        or candidate_length != tasks_per_key
                        or candidate_key != first_key + (end_run - run) * key_stride
                    ):
                        break
                    end_run += 1
        segments.append(
            TaskToKeySegment(
                task_begin=begin,
                task_count=(end_run - run) * tasks_per_key,
                tasks_per_key=tasks_per_key,
                first_key=first_key,
                key_stride=key_stride,
            )
        )
        run = end_run
    repeated: list[TaskToKeySegment] = []
    for segment in segments:
        if repeated:
            previous = repeated[-1]
            period = previous.key_period or previous.task_count
            if (
                segment.task_begin == previous.task_begin + previous.task_count
                and segment.task_count == period
                and segment.tasks_per_key == previous.tasks_per_key
                and segment.first_key == previous.first_key
                and segment.key_stride == previous.key_stride
                and segment.key_period is None
            ):
                repeated[-1] = dataclasses.replace(
                    previous,
                    task_count=previous.task_count + segment.task_count,
                    key_period=period,
                )
                continue
        repeated.append(segment)
    return tuple(repeated)


def _task_coordinates(
    task: int,
    family: InstantiatedTaskFamily,
) -> dict[int, int]:
    coordinates: dict[int, int] = {}
    remainder = task
    for block_id in family.logical_axis_order:
        count = family.axis_counts[block_id]
        coordinates[block_id] = remainder % count
        remainder //= count
    if remainder:
        raise AssertionError("task exceeds its logical coordinate domain")
    return coordinates


def _edge_predecessor_sets(
    edge: CrossLoopDependencyEdge,
    *,
    task_families: tuple[InstantiatedTaskFamily, ...],
    access_by_id: dict[int, CrossLoopAccess],
) -> tuple[frozenset[int], ...] | None:
    """Evaluate conservative task overlap with an interval sweep.

    The prior implementation compared every producer task with every consumer
    task and needed an arbitrary work cutoff.  Allocation-address intervals
    provide a complete candidate index: sorting them makes planning cost
    proportional to the task domains plus the overlaps the relation actually
    contains.  Unknown intervals are not guessed; they retain root completion.
    """
    producer = task_families[edge.producer_root]
    consumer = task_families[edge.consumer_root]
    region_cache: dict[tuple[int, int], AllocationRegion] = {}

    def task_region(
        access_id: int,
        task: int,
        family: InstantiatedTaskFamily,
    ) -> AllocationRegion:
        key = (access_id, task)
        if key not in region_cache:
            access = access_by_id[access_id]
            region_cache[key] = access_task_region(
                access,
                task_coordinates=_task_coordinates(task, family),
                block_sizes=family.block_sizes,
            )
        return region_cache[key]

    def indexed_regions(
        access_id: int,
        family: InstantiatedTaskFamily,
        task_count: int,
        dependency_region: AllocationRegion,
    ) -> list[tuple[int, int, int, AllocationRegion]] | None:
        result: list[tuple[int, int, int, AllocationRegion]] = []
        for task in range(task_count):
            region = task_region(access_id, task, family)
            interval = region.address_interval
            if interval is None:
                return None
            if interval[0] < interval[1] and allocation_regions_may_overlap(
                region, dependency_region
            ):
                result.append((interval[0], interval[1], task, region))
        result.sort(key=itemgetter(0, 1, 2))
        return result

    predecessors = [set() for _ in range(consumer.task_count)]
    for dependency in edge.access_dependencies:
        producer_regions = indexed_regions(
            dependency.producer_access_id,
            producer,
            producer.task_count,
            dependency.region,
        )
        consumer_regions = indexed_regions(
            dependency.consumer_access_id,
            consumer,
            consumer.task_count,
            dependency.region,
        )
        if producer_regions is None or consumer_regions is None:
            return None

        # Regions are half-open, so ends precede starts at the same address.
        # Producer starts precede consumer starts so equal-start pairs are
        # emitted exactly once.
        producer_end = 0
        consumer_end = 1
        producer_start = 2
        consumer_start = 3
        sweep_events: list[tuple[int, int, int, AllocationRegion]] = []
        for begin, end, task, region in producer_regions:
            sweep_events.extend(
                (
                    (begin, producer_start, task, region),
                    (end, producer_end, task, region),
                )
            )
        for begin, end, task, region in consumer_regions:
            sweep_events.extend(
                (
                    (begin, consumer_start, task, region),
                    (end, consumer_end, task, region),
                )
            )
        sweep_events.sort(key=itemgetter(0, 1, 2))

        active_producers: dict[int, AllocationRegion] = {}
        active_consumers: dict[int, AllocationRegion] = {}
        for _, event_kind, task, region in sweep_events:
            if event_kind == producer_end:
                active_producers.pop(task)
            elif event_kind == consumer_end:
                active_consumers.pop(task)
            elif event_kind == producer_start:
                for consumer_task, consumer_region in active_consumers.items():
                    if allocation_regions_may_overlap(region, consumer_region):
                        predecessors[consumer_task].add(task)
                active_producers[task] = region
            else:
                for producer_task, producer_region in active_producers.items():
                    if allocation_regions_may_overlap(producer_region, region):
                        predecessors[task].add(producer_task)
                active_consumers[task] = region
    return tuple(frozenset(tasks) for tasks in predecessors)


def derive_singleton_worker_affinity(
    *,
    dependency_plan: CrossLoopDependencyPlan,
    task_families: tuple[InstantiatedTaskFamily, ...],
    singleton_roots: frozenset[int],
    worker_limit: int,
) -> dict[int, int]:
    """Assign singleton roots away from their predecessors' active workers."""
    if worker_limit <= 0:
        raise ValueError(f"worker_limit must be positive, got {worker_limit}")
    predecessors_by_root: dict[int, set[int]] = {}
    for edge in dependency_plan.edges:
        predecessors_by_root.setdefault(edge.consumer_root, set()).add(
            edge.producer_root
        )

    result: dict[int, int] = {}
    for singleton_index, root in enumerate(sorted(singleton_roots)):
        predecessor_workers = max(
            (
                min(task_families[producer].task_count, worker_limit)
                for producer in predecessors_by_root.get(root, ())
            ),
            default=0,
        )
        idle_workers = worker_limit - predecessor_workers
        if idle_workers > 0:
            result[root] = predecessor_workers + singleton_index % idle_workers
        else:
            result[root] = worker_limit - 1 - singleton_index % worker_limit
    return result


def build_generic_schedule_plan(
    *,
    dependency_plan: CrossLoopDependencyPlan,
    task_families: tuple[InstantiatedTaskFamily, ...],
    available_access_ids_by_root: tuple[frozenset[int], ...],
    access_program_points: dict[int, AccessProgramPoint],
    axis_geometry: dict[int, tuple[int, int]],
    excluded_roots: frozenset[int],
    preordered_edges: frozenset[tuple[int, int]],
    physical_worker_limit: int,
    frontier_index: int = TILE_DEPENDENCY_FRONTIER_DEFAULT,
) -> GenericSchedulePlan:
    """Derive all generic readiness strategies without inspecting root bodies."""
    task_ready_edges, waits_by_root, root_wait_candidates = _select_available_waits(
        dependency_plan=dependency_plan,
        task_families=task_families,
        available_access_ids_by_root=available_access_ids_by_root,
        access_program_points=access_program_points,
        axis_geometry=axis_geometry,
        excluded_roots=excluded_roots,
    )
    candidate_counted_events = _derive_counted_events(
        dependency_plan=dependency_plan,
        waits_by_root=waits_by_root,
        task_families=task_families,
        physical_worker_limit=physical_worker_limit,
    )
    access_cohorts = _derive_access_cohorts(
        dependency_plan=dependency_plan,
        waits_by_root=waits_by_root,
        task_families=task_families,
        axis_geometry=axis_geometry,
        access_program_points=access_program_points,
        physical_worker_limit=physical_worker_limit,
        coarse_producer_roots=frozenset(
            plan.key_root for plan in candidate_counted_events
        ),
    )
    cohort_pairs = {(plan.producer_root, plan.consumer_root) for plan in access_cohorts}
    waits_by_pair: dict[tuple[int, int], list[WaitSpec]] = {}
    for consumer_root, waits in waits_by_root.items():
        for wait in waits:
            producer_root = dependency_plan.event(wait.event_id).producer_root
            waits_by_pair.setdefault((producer_root, consumer_root), []).append(wait)
    access_fallback_pairs = {
        pair
        for pair, waits in waits_by_pair.items()
        if pair not in cohort_pairs
        and waits
        and all(wait.placement == "access" for wait in waits)
    }
    if access_fallback_pairs:
        task_ready_edges = frozenset(task_ready_edges - access_fallback_pairs)
        waits_by_root = {
            root: tuple(
                wait
                for wait in waits
                if (
                    dependency_plan.event(wait.event_id).producer_root,
                    root,
                )
                not in access_fallback_pairs
            )
            for root, waits in waits_by_root.items()
        }
        waits_by_root = {root: waits for root, waits in waits_by_root.items() if waits}
        root_wait_candidates = (
            *root_wait_candidates,
            *(
                RootCompletionWait(producer, consumer, None, "root_entry")
                for producer, consumer in sorted(access_fallback_pairs)
            ),
        )
    root_wait_pairs_before_elision = {
        (wait.producer_root, wait.consumer_root) for wait in root_wait_candidates
    }
    root_wait_candidates = _elide_redundant_root_waits(
        root_wait_candidates,
        access_cohorts=access_cohorts,
        task_families=task_families,
        dependency_plan=dependency_plan,
        access_program_points=access_program_points,
    )
    root_wait_pairs_after_elision = {
        (wait.producer_root, wait.consumer_root) for wait in root_wait_candidates
    }
    task_ready_edges = frozenset(
        set(task_ready_edges)
        | (root_wait_pairs_before_elision - root_wait_pairs_after_elision)
    )
    waits_by_root = {
        root: tuple(
            wait
            for wait in waits
            if (
                dependency_plan.event(wait.event_id).producer_root,
                wait.consumer_root,
            )
            in task_ready_edges
        )
        for root, waits in waits_by_root.items()
    }
    waits_by_root = {root: waits for root, waits in waits_by_root.items() if waits}
    access_cohorts = tuple(
        plan
        for plan in access_cohorts
        if (plan.producer_root, plan.consumer_root) in task_ready_edges
    )
    counted_events = _derive_counted_events(
        dependency_plan=dependency_plan,
        waits_by_root=waits_by_root,
        task_families=task_families,
        physical_worker_limit=physical_worker_limit,
    )
    coalesced_events = _derive_coalesced_keyed_events(
        dependency_plan=dependency_plan,
        task_families=task_families,
        excluded_roots=excluded_roots,
        excluded_consumer_roots=frozenset(
            plan.consumer_root for plan in access_cohorts
        ),
        existing_events=counted_events,
        physical_worker_limit=physical_worker_limit,
    )
    counted_events = (*counted_events, *coalesced_events)
    coalesced_pairs = {
        (contributor.producer_root, event.key_root)
        for event in coalesced_events
        for contributor in event.contributors
    }
    if coalesced_pairs:
        task_ready_edges = frozenset(set(task_ready_edges) | coalesced_pairs)
        root_wait_candidates = tuple(
            wait
            for wait in root_wait_candidates
            if (wait.producer_root, wait.consumer_root) not in coalesced_pairs
        )
    on_ready_waits = {
        (plan.on_ready_root, event_id)
        for plan in counted_events
        if plan.on_ready_root is not None
        for event_id in plan.source_event_ids
    }
    keyed_event_waits = {
        (plan.key_root, event_id)
        for plan in counted_events
        for event_id in plan.source_event_ids
    }
    coarsened_access_waits = {
        (plan.consumer_root, plan.event_id, access_id)
        for plan in access_cohorts
        for access_id in plan.access_ids
    }
    retained_waits = {
        root: tuple(
            wait
            for wait in waits
            if (
                root,
                wait.event_id,
                wait.consumer_access_id,
            )
            not in coarsened_access_waits
            and (root, wait.event_id) not in on_ready_waits
            and (root, wait.event_id) not in keyed_event_waits
        )
        for root, waits in waits_by_root.items()
    }
    retained_waits = {root: waits for root, waits in retained_waits.items() if waits}
    frontier_selection = _select_readiness_frontier(
        dependency_plan=dependency_plan,
        counted_events=counted_events,
        access_cohorts=access_cohorts,
        task_families=task_families,
        frontier_index=frontier_index,
    )
    readiness_frontiers = frontier_selection.plans
    if frontier_selection.root_completion_key_roots:
        fallback_event_pairs = {
            (contributor.producer_root, event.key_root)
            for event in counted_events
            if event.key_root in frontier_selection.root_completion_key_roots
            for contributor in event.contributors
        }
        task_ready_edges = frozenset(task_ready_edges - fallback_event_pairs)
        root_wait_candidates = (
            *root_wait_candidates,
            *(
                RootCompletionWait(producer, consumer, None, "root_entry")
                for producer, consumer in sorted(fallback_event_pairs)
            ),
        )
        counted_events = tuple(
            event
            for event in counted_events
            if event.key_root not in frontier_selection.root_completion_key_roots
        )
    selected_frontier_cohorts = {plan.downstream_cohort for plan in readiness_frontiers}
    inactive_coarse_pairs = {
        (plan.producer_root, plan.consumer_root)
        for plan in access_cohorts
        if not plan.is_per_coordinate and plan not in selected_frontier_cohorts
    }
    if inactive_coarse_pairs:
        task_ready_edges = frozenset(task_ready_edges - inactive_coarse_pairs)
        root_wait_candidates = (
            *root_wait_candidates,
            *(
                RootCompletionWait(producer, consumer, None, "root_entry")
                for producer, consumer in sorted(inactive_coarse_pairs)
            ),
        )
        access_cohorts = tuple(
            plan
            for plan in access_cohorts
            if (plan.producer_root, plan.consumer_root) not in inactive_coarse_pairs
        )
    worker_limit = physical_worker_limit
    if readiness_frontiers:
        worker_limit = max(plan.worker_count for plan in readiness_frontiers)
        replacement_cohorts = {
            plan.downstream_cohort: dataclasses.replace(
                plan.downstream_cohort,
                stage_sizes=(
                    plan.initial_stream_tasks,
                    sum(plan.downstream_cohort.stage_sizes) - plan.initial_stream_tasks,
                ),
            )
            for plan in readiness_frontiers
        }
        access_cohorts = tuple(
            replacement_cohorts.get(plan, plan) for plan in access_cohorts
        )
        readiness_frontiers = tuple(
            dataclasses.replace(
                plan,
                downstream_cohort=replacement_cohorts[plan.downstream_cohort],
            )
            for plan in readiness_frontiers
        )

    root_waits = _select_root_completion_waits(
        dependencies=dependency_plan.edges,
        fully_task_ready_edges=task_ready_edges,
        root_wait_candidates=root_wait_candidates,
        preordered_edges=preordered_edges,
    )
    root_completion_edges = frozenset(
        (wait.producer_root, wait.consumer_root) for wait in root_waits
    )
    structural_task_pairs = {
        (contributor.producer_root, event.key_root)
        for event in counted_events
        for contributor in event.contributors
    }
    structural_task_pairs.update(
        (cohort.producer_root, cohort.consumer_root) for cohort in access_cohorts
    )
    root_order_edges = set(root_completion_edges) | set(preordered_edges)
    redundant_task_wait_pairs = {
        (dependency_plan.event(wait.event_id).producer_root, consumer_root)
        for consumer_root, waits in retained_waits.items()
        for wait in waits
        if (
            dependency_plan.event(wait.event_id).producer_root,
            consumer_root,
        )
        not in structural_task_pairs
        and _is_ordered_by_root_completion(
            dependency_plan.event(wait.event_id).producer_root,
            consumer_root,
            root_order_edges,
        )
    }
    if redundant_task_wait_pairs:
        task_ready_edges = frozenset(task_ready_edges - redundant_task_wait_pairs)
        retained_waits = {
            root: tuple(
                wait
                for wait in waits
                if (
                    dependency_plan.event(wait.event_id).producer_root,
                    root,
                )
                not in redundant_task_wait_pairs
            )
            for root, waits in retained_waits.items()
        }
        retained_waits = {
            root: waits for root, waits in retained_waits.items() if waits
        }
    return GenericSchedulePlan(
        task_ready_edges=task_ready_edges,
        root_completion_edges=root_completion_edges,
        task_waits_by_root=retained_waits,
        access_cohorts=access_cohorts,
        counted_events=counted_events,
        readiness_frontiers=readiness_frontiers,
        worker_limit=worker_limit,
    )


def _select_root_completion_waits(
    *,
    dependencies: tuple[CrossLoopDependencyEdge, ...],
    fully_task_ready_edges: frozenset[tuple[int, int]],
    root_wait_candidates: tuple[RootCompletionWait, ...],
    preordered_edges: frozenset[tuple[int, int]],
) -> tuple[RootCompletionWait, ...]:
    """Choose the minimal source-ordered root-completion fallback edges."""

    waits_by_pair: dict[tuple[int, int], list[RootCompletionWait]] = {}
    for wait in root_wait_candidates:
        waits_by_pair.setdefault((wait.producer_root, wait.consumer_root), []).append(
            wait
        )
    root_completion_waits: list[RootCompletionWait] = []
    ordered_root_edges: set[tuple[int, int]] = set()
    for dependency in sorted(
        dependencies,
        key=lambda edge: (
            edge.consumer_root - edge.producer_root,
            edge.producer_root,
            edge.consumer_root,
        ),
    ):
        pair = (dependency.producer_root, dependency.consumer_root)
        if pair in fully_task_ready_edges or pair in preordered_edges:
            continue
        if _is_ordered_by_root_completion(*pair, ordered_root_edges):
            continue
        root_completion_waits.extend(
            waits_by_pair.get(
                pair,
                (
                    RootCompletionWait(
                        producer_root=dependency.producer_root,
                        consumer_root=dependency.consumer_root,
                        consumer_access_id=None,
                        placement="root_entry",
                    ),
                ),
            )
        )
        ordered_root_edges.add(pair)
    return tuple(dict.fromkeys(root_completion_waits))


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


def _select_available_waits(
    *,
    dependency_plan: CrossLoopDependencyPlan,
    task_families: tuple[InstantiatedTaskFamily, ...],
    available_access_ids_by_root: tuple[frozenset[int], ...],
    access_program_points: dict[int, AccessProgramPoint],
    axis_geometry: dict[int, tuple[int, int]],
    excluded_roots: frozenset[int],
) -> tuple[
    frozenset[tuple[int, int]],
    dict[int, tuple[WaitSpec, ...]],
    tuple[RootCompletionWait, ...],
]:
    kinds_by_pair: dict[tuple[int, int], set[TileDependencyKind]] = {}
    for dependency in dependency_plan.edges:
        pair = (dependency.producer_root, dependency.consumer_root)
        kinds_by_pair.setdefault(pair, set()).update(dependency.kinds)

    waits_by_pair: dict[tuple[int, int], list[WaitSpec]] = {}
    for wait in dependency_plan.waits:
        event = dependency_plan.event(wait.event_id)
        waits_by_pair.setdefault((event.producer_root, wait.consumer_root), []).append(
            wait
        )

    fully_task_ready_pairs: set[tuple[int, int]] = set()
    waits_by_root: dict[int, list[WaitSpec]] = {}
    root_waits: list[RootCompletionWait] = []
    for pair in kinds_by_pair:
        producer_root, consumer_root = pair
        producer = task_families[producer_root]
        consumer = task_families[consumer_root]
        pair_waits = waits_by_pair.get(pair, ())
        task_waits = tuple(
            wait
            for wait in pair_waits
            if dependency_plan.event(wait.event_id).granularity == "task"
        )
        declared_root_waits = tuple(
            wait
            for wait in pair_waits
            if dependency_plan.event(wait.event_id).granularity == "root"
        )
        can_use_tasks = (
            producer_root not in excluded_roots
            and consumer_root not in excluded_roots
            # A one-task event is exactly root completion. Keep one canonical
            # representation so singleton producers do not allocate and poll
            # a second, equivalent task-event array.
            and producer.task_count > 1
            and consumer.task_count > 0
            and bool(task_waits)
        )
        producer_axes = set(producer.logical_axis_order)
        consumer_axes = set(consumer.logical_axis_order)
        for wait in task_waits:
            predecessor_map = wait.predecessor_map
            if (
                predecessor_map is None
                or {axis.producer_block_id for axis in predecessor_map.axes}
                != producer_axes
            ):
                can_use_tasks = False
                break
            required_consumer_axes = {
                axis.consumer_block_id for axis in predecessor_map.axes
            }
            if wait.placement == "root_entry":
                if not required_consumer_axes <= consumer_axes:
                    can_use_tasks = False
                    break
                continue
            access_id = wait.consumer_access_id
            program_point = (
                access_program_points.get(access_id) if access_id is not None else None
            )
            coordinates = program_point.coordinates if program_point else None
            if (
                access_id is None
                or access_id not in available_access_ids_by_root[consumer_root]
                or coordinates is None
                or not required_consumer_axes <= coordinates.keys()
                or any(axis not in axis_geometry for axis in required_consumer_axes)
            ):
                can_use_tasks = False
                break

        if not can_use_tasks:
            root_waits.append(
                RootCompletionWait(producer_root, consumer_root, None, "root_entry")
            )
            continue

        selected_waits = waits_by_root.setdefault(consumer_root, [])
        existing = {
            (
                wait.event_id,
                wait.predecessor_map.axes,
                wait.consumer_access_id if wait.placement == "access" else None,
            )
            for wait in selected_waits
            if wait.predecessor_map is not None
        }
        for wait in task_waits:
            assert wait.predecessor_map is not None
            key = (
                wait.event_id,
                wait.predecessor_map.axes,
                wait.consumer_access_id if wait.placement == "access" else None,
            )
            if key not in existing:
                selected_waits.append(wait)
                existing.add(key)

        if not declared_root_waits:
            fully_task_ready_pairs.add(pair)
            continue
        for wait in declared_root_waits:
            access_id = wait.consumer_access_id
            if (
                wait.placement == "access"
                and access_id is not None
                and access_id in available_access_ids_by_root[consumer_root]
            ):
                root_waits.append(
                    RootCompletionWait(
                        producer_root,
                        consumer_root,
                        access_id,
                        "access",
                    )
                )
            else:
                root_waits.append(
                    RootCompletionWait(producer_root, consumer_root, None, "root_entry")
                )

    return (
        frozenset(fully_task_ready_pairs),
        {root: tuple(waits) for root, waits in waits_by_root.items()},
        tuple(dict.fromkeys(root_waits)),
    )


def _elide_redundant_root_waits(
    waits: tuple[RootCompletionWait, ...],
    *,
    access_cohorts: tuple[AccessCohortPlan, ...],
    task_families: tuple[InstantiatedTaskFamily, ...],
    dependency_plan: CrossLoopDependencyPlan,
    access_program_points: dict[int, AccessProgramPoint],
) -> tuple[RootCompletionWait, ...]:
    """Drop late root waits dominated by a complete one-consumer stream.

    A per-coordinate cohort with one consumer task waits for every producer
    task while traversing a direct top-level loop.  Once that loop exits, a
    later top-level access in the same opaque root is already protected by the
    same producer completion and needs no second, whole-root event.
    """
    access_by_id = {access.access_id: access for access in dependency_plan.accesses}
    covered_pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for cohort in access_cohorts:
        if (
            not cohort.is_per_coordinate
            or task_families[cohort.consumer_root].task_count != 1
        ):
            continue
        points = [access_program_points[access_id] for access_id in cohort.access_ids]
        if any(point.loop_depth != 1 for point in points):
            continue
        covered_pairs[(cohort.producer_root, cohort.consumer_root)] = (
            max(point.root_statement_index for point in points),
            max(
                access_by_id[access_id].memory_op_index
                for access_id in cohort.access_ids
            ),
        )

    result: list[RootCompletionWait] = []
    for wait in waits:
        covered = covered_pairs.get((wait.producer_root, wait.consumer_root))
        access_id = wait.consumer_access_id
        point = access_program_points.get(access_id) if access_id is not None else None
        access = access_by_id.get(access_id) if access_id is not None else None
        if (
            wait.placement == "access"
            and covered is not None
            and point is not None
            and access is not None
            and point.root_statement_index > covered[0]
            and access.memory_op_index > covered[1]
        ):
            continue
        result.append(wait)
    return tuple(result)


def _derive_access_cohorts(
    *,
    dependency_plan: CrossLoopDependencyPlan,
    waits_by_root: dict[int, tuple[WaitSpec, ...]],
    task_families: tuple[InstantiatedTaskFamily, ...],
    axis_geometry: dict[int, tuple[int, int]],
    access_program_points: dict[int, AccessProgramPoint],
    physical_worker_limit: int,
    coarse_producer_roots: frozenset[int],
) -> tuple[AccessCohortPlan, ...]:
    result: list[AccessCohortPlan] = []
    for consumer_root, waits in waits_by_root.items():
        waits_by_event: dict[int, list[WaitSpec]] = {}
        for wait in waits:
            if wait.placement == "access":
                waits_by_event.setdefault(wait.event_id, []).append(wait)
        for event_id, event_waits in waits_by_event.items():
            predecessor_maps = [
                wait.predecessor_map
                for wait in event_waits
                if wait.predecessor_map is not None
            ]
            if len(predecessor_maps) != len(event_waits):
                continue
            axes = predecessor_maps[0].axes
            axis_pairs = tuple(
                (axis.producer_block_id, axis.consumer_block_id) for axis in axes
            )
            if any(
                tuple(
                    (axis.producer_block_id, axis.consumer_block_id)
                    for axis in mapping.axes
                )
                != axis_pairs
                for mapping in predecessor_maps[1:]
            ):
                continue

            producer_root = dependency_plan.event(event_id).producer_root
            producer = task_families[producer_root]
            consumer = task_families[consumer_root]
            producer_counts = producer.axis_counts
            producer_blocks = producer.block_sizes
            consumer_counts = consumer.axis_counts
            axes_by_producer = {axis.producer_block_id: axis for axis in axes}
            if len(axes_by_producer) != len(axes) or set(axes_by_producer) != set(
                producer.logical_axis_order
            ):
                continue

            nested_axes = [
                axis for axis in axes if axis.consumer_block_id not in consumer_counts
            ]
            if len(nested_axes) != 1:
                continue
            stream_axis = nested_axes[0]
            valid = True
            stream_task_granularity: int | None = None
            for predecessor_map in predecessor_maps:
                seen_consumer_axes: set[int] = set()
                for axis in predecessor_map.axes:
                    consumer_geometry = axis_geometry.get(axis.consumer_block_id)
                    if consumer_geometry is None:
                        valid = False
                        break
                    consumer_count, consumer_block = consumer_geometry
                    producer_block = producer_blocks[axis.producer_block_id]
                    producer_width = 1 if axis.producer_is_scalar else producer_block
                    consumer_width = 1 if axis.consumer_is_scalar else consumer_block
                    if axis.consumer_block_id in seen_consumer_axes or (
                        axis.producer_offset != axis.consumer_offset
                    ):
                        valid = False
                        break
                    is_stream_axis = (
                        axis.producer_block_id == stream_axis.producer_block_id
                        and axis.consumer_block_id == stream_axis.consumer_block_id
                    )
                    if not is_stream_axis:
                        if (
                            producer_counts[axis.producer_block_id] != consumer_count
                            or producer_width != consumer_width
                        ):
                            valid = False
                            break
                        seen_consumer_axes.add(axis.consumer_block_id)
                        continue
                    if axis.producer_is_scalar or axis.consumer_is_scalar:
                        if (
                            axis.producer_is_scalar != axis.consumer_is_scalar
                            or producer_counts[axis.producer_block_id] != consumer_count
                            or producer_width != consumer_width
                        ):
                            valid = False
                            break
                        ratio = 1
                    else:
                        if consumer_width % producer_width:
                            valid = False
                            break
                        ratio = consumer_width // producer_width
                        if producer_counts[axis.producer_block_id] != (
                            consumer_count * ratio
                        ):
                            valid = False
                            break
                    if (
                        stream_task_granularity is not None
                        and stream_task_granularity != ratio
                    ):
                        valid = False
                        break
                    stream_task_granularity = ratio
                    seen_consumer_axes.add(axis.consumer_block_id)
                if not valid:
                    break
            if not valid or stream_task_granularity is None:
                continue

            stream_count = producer_counts[stream_axis.producer_block_id]
            if stream_count <= 1:
                continue
            access_ids = tuple(
                wait.consumer_access_id
                for wait in event_waits
                if wait.consumer_access_id is not None
            )
            if len(access_ids) != len(event_waits):
                continue
            program_points = tuple(
                access_program_points[access_id]
                for access_id in access_ids
                if access_id in access_program_points
            )
            loop_ids = {point.loop_id for point in program_points}
            if (
                len(program_points) != len(access_ids)
                or None in loop_ids
                or len(loop_ids) != 1
                or any(point.coordinates is None for point in program_points)
            ):
                continue
            consumer_loop_id = next(iter(loop_ids))
            assert consumer_loop_id is not None
            mapped_consumer_root_axes = {
                axis.consumer_block_id
                for axis in axes
                if axis.consumer_block_id in consumer_counts
            }
            consumer_fanout = math.prod(
                count
                for block_id, count in consumer_counts.items()
                if block_id not in mapped_consumer_root_axes
            )
            stream_coordinate_expressions = {
                coordinates.get(stream_axis.consumer_block_id)
                for program_point in program_points
                if (coordinates := program_point.coordinates) is not None
            }
            per_coordinate = (
                stream_task_granularity == 1
                and consumer_fanout == 1
                and producer.task_count > physical_worker_limit
                and len(stream_coordinate_expressions) == 1
                and None not in stream_coordinate_expressions
            )
            if not per_coordinate and producer_root not in coarse_producer_roots:
                continue
            if per_coordinate:
                consumer_stream_coordinate = next(iter(stream_coordinate_expressions))
                assert consumer_stream_coordinate is not None
                stage_sizes: tuple[int, ...] = ()
            else:
                consumer_stream_coordinate = None
                stage_sizes = (stream_count,)
            result.append(
                AccessCohortPlan(
                    event_id=event_id,
                    producer_root=producer_root,
                    consumer_root=consumer_root,
                    axes=axes,
                    access_ids=access_ids,
                    producer_stream_axis=stream_axis.producer_block_id,
                    consumer_stream_axis=stream_axis.consumer_block_id,
                    consumer_loop_id=consumer_loop_id,
                    consumer_stream_coordinate=consumer_stream_coordinate,
                    stream_count=stream_count,
                    stream_task_granularity=stream_task_granularity,
                    stage_sizes=stage_sizes,
                    outer_producer_axes=tuple(
                        block_id
                        for block_id in producer.logical_axis_order
                        if block_id != stream_axis.producer_block_id
                    ),
                )
            )
    plan_count_by_consumer: dict[int, int] = {}
    for plan in result:
        plan_count_by_consumer[plan.consumer_root] = (
            plan_count_by_consumer.get(plan.consumer_root, 0) + 1
        )
    return tuple(
        plan for plan in result if plan_count_by_consumer[plan.consumer_root] == 1
    )


def _derive_counted_events(
    *,
    dependency_plan: CrossLoopDependencyPlan,
    waits_by_root: dict[int, tuple[WaitSpec, ...]],
    task_families: tuple[InstantiatedTaskFamily, ...],
    physical_worker_limit: int,
) -> tuple[KeyedEventPlan, ...]:
    result: list[KeyedEventPlan] = []
    for consumer_root, waits in waits_by_root.items():
        if not waits or any(wait.placement != "root_entry" for wait in waits):
            continue
        incoming_producer_roots = {
            edge.producer_root
            for edge in dependency_plan.edges
            if edge.consumer_root == consumer_root
        }
        waits_by_event: dict[int, list[WaitSpec]] = {}
        for wait in waits:
            waits_by_event.setdefault(wait.event_id, []).append(wait)
        if {
            dependency_plan.event(event_id).producer_root for event_id in waits_by_event
        } != incoming_producer_roots:
            continue
        consumer = task_families[consumer_root]
        if (
            any(
                event.producer_root == consumer_root and event.granularity == "root"
                for event in dependency_plan.events
            )
            and consumer.task_count > physical_worker_limit
        ):
            continue
        contributors: list[KeyedEventContributorPlan] = []
        for event_id, event_waits in sorted(waits_by_event.items()):
            event = dependency_plan.event(event_id)
            if event.granularity != "task":
                break
            producer_root = event.producer_root
            producer = task_families[producer_root]
            predecessor_maps = tuple(
                wait.predecessor_map
                for wait in event_waits
                if wait.predecessor_map is not None
            )
            if len(predecessor_maps) != len(event_waits):
                break
            partition = prove_uniform_task_partition(
                predecessor_maps,
                consumer_axis_order=consumer.logical_axis_order,
                consumer_axis_counts=consumer.axis_counts,
                producer_axis_order=producer.logical_axis_order,
                producer_axis_counts=producer.axis_counts,
                block_sizes={**producer.block_sizes, **consumer.block_sizes},
            )
            if partition is None:
                break
            contributors.append(
                KeyedEventContributorPlan(
                    source_event_ids=(event_id,),
                    producer_root=producer_root,
                    task_to_key=partition.producer_key_by_task,
                    partition=partition,
                )
            )
        if len(contributors) != len(waits_by_event):
            continue
        if len(contributors) == 1 and contributors[0].expected_arrivals == 1:
            continue
        result.append(
            KeyedEventPlan(
                key_root=consumer_root,
                contributors=tuple(contributors),
                consumer_key_by_task=tuple(range(consumer.task_count)),
                on_ready_root=consumer_root,
            )
        )
    on_ready_roots = {plan.key_root for plan in result}

    def has_simple_reverse(
        plan: KeyedEventPlan,
        contributor: KeyedEventContributorPlan,
    ) -> bool:
        """Whether lowering can enumerate keys reached by one producer task."""
        partition = contributor.partition
        if partition is None:
            return True
        if any(axis.scale not in (-1, 0, 1) for axis in partition.outer_axes):
            return False
        partition_count = task_families[plan.key_root].axis_counts[
            partition.partition_consumer_block_id
        ]
        if partition_count == 1:
            return True
        return (
            len(partition.segments) == 1
            and partition.partition_consumer_stride > 0
            and partition.segments[0].length == partition.partition_consumer_stride
        )

    filtered: list[KeyedEventPlan] = []
    for plan in result:
        if all(
            contributor.producer_root not in on_ready_roots
            or has_simple_reverse(plan, contributor)
            for contributor in plan.contributors
        ):
            filtered.append(plan)
    return tuple(filtered)


def _derive_coalesced_keyed_events(
    *,
    dependency_plan: CrossLoopDependencyPlan,
    task_families: tuple[InstantiatedTaskFamily, ...],
    excluded_roots: frozenset[int],
    excluded_consumer_roots: frozenset[int],
    existing_events: tuple[KeyedEventPlan, ...],
    physical_worker_limit: int,
) -> tuple[KeyedEventPlan, ...]:
    """Coalesce repeated consumer predecessor sets into logical event keys."""
    existing_key_roots = {event.key_root for event in existing_events}
    existing_on_ready_roots = {
        event.on_ready_root
        for event in existing_events
        if event.on_ready_root is not None
    }
    access_by_id = {access.access_id: access for access in dependency_plan.accesses}
    edges_by_consumer: dict[int, list[CrossLoopDependencyEdge]] = {}
    for edge in dependency_plan.edges:
        edges_by_consumer.setdefault(edge.consumer_root, []).append(edge)

    result: list[KeyedEventPlan] = []
    for consumer_root, incoming_edges in edges_by_consumer.items():
        if (
            consumer_root in existing_key_roots
            or consumer_root in excluded_roots
            or consumer_root in excluded_consumer_roots
        ):
            continue
        producer_roots = tuple(sorted({edge.producer_root for edge in incoming_edges}))
        if not producer_roots or any(
            edge.kinds != frozenset((TileDependencyKind.READ_AFTER_WRITE,))
            or edge.producer_root in excluded_roots
            for edge in incoming_edges
        ):
            continue
        consumer = task_families[consumer_root]
        if consumer.task_count <= 1:
            continue

        valid = True
        predecessors_by_producer: dict[int, list[set[int]]] = {
            producer_root: [set() for _ in range(consumer.task_count)]
            for producer_root in producer_roots
        }
        for edge in incoming_edges:
            edge_predecessors = _edge_predecessor_sets(
                edge,
                task_families=task_families,
                access_by_id=access_by_id,
            )
            if edge_predecessors is None:
                valid = False
                break
            aggregate = predecessors_by_producer[edge.producer_root]
            for consumer_task, predecessors in enumerate(edge_predecessors):
                aggregate[consumer_task].update(predecessors)
        if not valid:
            continue

        key_by_signature: dict[tuple[frozenset[int], ...], int] = {}
        consumer_key_by_task: list[int] = []
        signatures: list[tuple[frozenset[int], ...]] = []
        for consumer_task in range(consumer.task_count):
            signature = tuple(
                frozenset(predecessors_by_producer[producer_root][consumer_task])
                for producer_root in producer_roots
            )
            if any(not predecessors for predecessors in signature):
                valid = False
                break
            key = key_by_signature.setdefault(signature, len(key_by_signature))
            consumer_key_by_task.append(key)
            if key == len(signatures):
                signatures.append(signature)
        if not valid or not signatures:
            continue
        if (
            len(producer_roots) == 1
            and len(signatures) == consumer.task_count
            and producer_roots[0] not in existing_on_ready_roots
        ):
            continue

        contributors: list[KeyedEventContributorPlan] = []
        for producer_index, producer_root in enumerate(producer_roots):
            producer = task_families[producer_root]
            task_to_key: list[int | None] = [None] * producer.task_count
            for key, signature in enumerate(signatures):
                for producer_task in signature[producer_index]:
                    previous = task_to_key[producer_task]
                    if previous is not None and previous != key:
                        valid = False
                        break
                    task_to_key[producer_task] = key
                if not valid:
                    break
            if not valid:
                break
            mapping = tuple(task_to_key)
            segments = _compress_task_to_key(mapping)
            if not segments:
                valid = False
                break
            source_event_ids = tuple(
                sorted(
                    {
                        wait.event_id
                        for wait in dependency_plan.waits
                        if wait.consumer_root == consumer_root
                        and dependency_plan.event(wait.event_id).producer_root
                        == producer_root
                    }
                )
            )
            contributors.append(
                KeyedEventContributorPlan(
                    source_event_ids=source_event_ids,
                    producer_root=producer_root,
                    task_to_key=mapping,
                )
            )
        consumer_mapping = tuple(consumer_key_by_task)
        consumer_segments = _compress_task_to_key(consumer_mapping)
        if not valid or not consumer_segments:
            continue
        if (
            len(signatures) == 1
            and len(contributors) == 1
            and all(key is not None for key in contributors[0].task_to_key)
        ):
            continue
        event = KeyedEventPlan(
            key_root=consumer_root,
            contributors=tuple(contributors),
            consumer_key_by_task=consumer_mapping,
            on_ready_root=(
                consumer_root
                if len(signatures) == consumer.task_count
                and consumer_mapping == tuple(range(consumer.task_count))
                else None
            ),
        )
        if (
            event.on_ready_root is not None
            and any(
                completion.producer_root == consumer_root
                and completion.granularity == "root"
                for completion in dependency_plan.events
            )
            and consumer.task_count > physical_worker_limit
        ):
            continue
        try:
            if event.expected_arrivals <= 0:
                continue
        except ValueError:
            continue
        result.append(event)
    return tuple(result)


def _select_readiness_frontier(
    *,
    dependency_plan: CrossLoopDependencyPlan,
    counted_events: tuple[KeyedEventPlan, ...],
    access_cohorts: tuple[AccessCohortPlan, ...],
    task_families: tuple[InstantiatedTaskFamily, ...],
    frontier_index: int,
) -> ReadinessFrontierSelection:
    candidate_families: list[
        tuple[KeyedEventPlan, tuple[ReadinessFrontierPlan, ...]]
    ] = []
    for event in counted_events:
        if event.on_ready_root is None or not event.is_single_contributor:
            continue
        contributor = event.single_contributor
        partition = contributor.partition
        if partition is None:
            continue
        matching_cohorts = [
            cohort
            for cohort in access_cohorts
            if cohort.producer_root == event.on_ready_root
        ]
        if len(matching_cohorts) != 1:
            continue
        cohort = matching_cohorts[0]
        if cohort.is_per_coordinate:
            continue
        downstream_root = cohort.consumer_root
        if {
            edge.producer_root
            for edge in dependency_plan.edges
            if edge.consumer_root == downstream_root
        } != {event.on_ready_root}:
            continue
        mapping_family = task_families[event.on_ready_root]
        if (
            not mapping_family.logical_axis_order
            or mapping_family.logical_axis_order[-1] != cohort.producer_stream_axis
            or mapping_family.logical_axis_order[:-1] != cohort.outer_producer_axes
        ):
            continue
        mapping_counts = mapping_family.axis_counts
        outer_count = math.prod(
            mapping_counts[block_id] for block_id in cohort.outer_producer_axes
        )
        stream_tasks = mapping_counts[cohort.producer_stream_axis]
        if partition.producer_tasks != (
            outer_count * stream_tasks * event.expected_arrivals
        ):
            continue
        downstream_tasks = task_families[downstream_root].task_count
        if downstream_tasks <= 0:
            continue

        minimum_workers = (partition.producer_tasks + downstream_tasks + 1) // 2
        worker_scale = outer_count * event.expected_arrivals
        minimum_frontier = (minimum_workers + worker_scale - 1) // worker_scale
        granularity = cohort.stream_task_granularity
        minimum_frontier = (
            (minimum_frontier + granularity - 1) // granularity * granularity
        )
        candidates = tuple(
            ReadinessFrontierPlan(
                event=event,
                downstream_cohort=cohort,
                worker_count=worker_scale * initial_stream_tasks,
                downstream_tasks=downstream_tasks,
                initial_stream_tasks=initial_stream_tasks,
            )
            for initial_stream_tasks in range(
                minimum_frontier,
                stream_tasks,
                granularity,
            )
            if (
                partition.producer_tasks - worker_scale * initial_stream_tasks
                <= worker_scale * initial_stream_tasks - downstream_tasks
            )
        )
        candidate_families.append((event, candidates))

    if frontier_index < -1:
        raise exc.InvalidConfig(
            f"{TILE_DEPENDENCY_FRONTIER_CONFIG} must be at least -1, got "
            f"{frontier_index}"
        )
    if frontier_index == -1:
        return ReadinessFrontierSelection(
            root_completion_key_roots=frozenset(
                event.key_root for event, _candidates in candidate_families
            )
        )
    if len(candidate_families) != 1:
        raise exc.InvalidConfig(
            f"{TILE_DEPENDENCY_FRONTIER_CONFIG} selects candidate "
            f"{frontier_index}, but this configuration has no unique "
            "readiness-frontier component"
        )
    event, candidates = candidate_families[0]
    if frontier_index >= len(candidates):
        raise exc.InvalidConfig(
            f"{TILE_DEPENDENCY_FRONTIER_CONFIG} selects candidate "
            f"{frontier_index}, but this configuration has only "
            f"{len(candidates)} readiness frontiers"
        )
    return ReadinessFrontierSelection(plans=(candidates[frontier_index],))


def potential_readiness_frontier_stream_axes(
    dependency_plan: CrossLoopDependencyPlan,
    task_families: tuple[TaskFamily, ...],
) -> frozenset[int]:
    """Return producer axes that can parameterize a readiness frontier."""
    producer_by_event = {
        event.event_id: event.producer_root
        for event in dependency_plan.events
        if event.granularity == "task"
    }
    root_entry_consumers = {
        wait.consumer_root
        for wait in dependency_plan.waits
        if wait.placement == "root_entry"
        and wait.predecessor_map is not None
        and wait.event_id in producer_by_event
    }
    result: set[int] = set()
    for wait in dependency_plan.waits:
        predecessor_map = wait.predecessor_map
        if (
            wait.placement != "access"
            or predecessor_map is None
            or producer_by_event.get(wait.event_id) not in root_entry_consumers
        ):
            continue
        consumer_axes = set(task_families[wait.consumer_root].logical_axis_order)
        nested_axes = {
            axis.producer_block_id
            for axis in predecessor_map.axes
            if axis.consumer_block_id not in consumer_axes
        }
        if len(nested_axes) == 1:
            result.update(nested_axes)
    return frozenset(result)
