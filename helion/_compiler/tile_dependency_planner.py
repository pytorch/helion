from __future__ import annotations

import dataclasses
import math
from typing import Literal

from .cross_loop_dependencies import AffinePredecessorAxis
from .cross_loop_dependencies import CrossLoopDependencyEdge
from .cross_loop_dependencies import CrossLoopDependencyPlan
from .cross_loop_dependencies import TileDependencyKind
from .cross_loop_dependencies import UniformTaskPartition
from .cross_loop_dependencies import WaitSpec
from .cross_loop_dependencies import prove_uniform_task_partition


@dataclasses.dataclass(frozen=True)
class AccessProgramPoint:
    """A lowered access location with its logical loop coordinates.

    The dependency proof names axes by source-level block ID.  The expressions
    here are used only to materialize an already-proven event key at the
    explicit access marker; they are never used to infer dependency geometry.
    """

    access_id: int
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

    root: int
    logical_axis_order: tuple[int, ...]
    physical_axis_order: tuple[int, ...]
    axis_counts_items: tuple[tuple[int, int], ...]
    block_sizes_items: tuple[tuple[int, int], ...]
    has_nontrivial_pid_remap: bool = False

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
class TaskContinuationPlan:
    """A consumer task elected by the last task in a proven partition."""

    event_id: int
    producer_root: int
    consumer_root: int
    partition: UniformTaskPartition

    @property
    def producer_tasks(self) -> int:
        return self.partition.producer_tasks

    @property
    def consumer_tasks(self) -> int:
        return self.partition.consumer_tasks

    @property
    def fanin(self) -> int:
        return self.partition.fanin


@dataclasses.dataclass(frozen=True)
class TaskContinuationPipelinePlan:
    """A continuation followed by a concurrently admitted consumer root."""

    continuation: TaskContinuationPlan
    cohort: AccessCohortPlan
    worker_count: int
    consumer_tasks: int
    initial_stream_tasks: int

    @property
    def tail_producer_tasks(self) -> int:
        return self.continuation.producer_tasks - self.worker_count

    @property
    def consumer_worker_begin(self) -> int:
        return self.worker_count - self.consumer_tasks


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
    root_waits_by_root: dict[int, tuple[RootCompletionWait, ...]]
    task_waits_by_root: dict[int, tuple[WaitSpec, ...]]
    access_cohorts: tuple[AccessCohortPlan, ...]
    continuations: tuple[TaskContinuationPlan, ...]
    continuation_pipelines: tuple[TaskContinuationPipelinePlan, ...]
    worker_limit: int


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
    access_cohorts = _derive_access_cohorts(
        dependency_plan=dependency_plan,
        waits_by_root=waits_by_root,
        task_families=task_families,
        axis_geometry=axis_geometry,
        access_program_points=access_program_points,
        physical_worker_limit=physical_worker_limit,
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
    continuations = _derive_task_continuations(
        dependency_plan=dependency_plan,
        waits_by_root=waits_by_root,
        task_families=task_families,
    )
    continuation_waits = {(plan.consumer_root, plan.event_id) for plan in continuations}
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
            and (root, wait.event_id) not in continuation_waits
        )
        for root, waits in waits_by_root.items()
    }
    retained_waits = {root: waits for root, waits in retained_waits.items() if waits}
    pipelines = _derive_task_continuation_pipelines(
        dependency_plan=dependency_plan,
        task_continuations=continuations,
        access_cohorts=access_cohorts,
        task_families=task_families,
        physical_worker_limit=physical_worker_limit,
    )
    worker_limit = physical_worker_limit
    if pipelines:
        worker_limit = max(plan.worker_count for plan in pipelines)
        replacement_cohorts = {
            plan.cohort: dataclasses.replace(
                plan.cohort,
                stage_sizes=(
                    plan.initial_stream_tasks,
                    sum(plan.cohort.stage_sizes) - plan.initial_stream_tasks,
                ),
            )
            for plan in pipelines
        }
        access_cohorts = tuple(
            replacement_cohorts.get(plan, plan) for plan in access_cohorts
        )
        pipelines = tuple(
            dataclasses.replace(
                plan,
                cohort=replacement_cohorts[plan.cohort],
            )
            for plan in pipelines
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
    root_waits_by_root: dict[int, list[RootCompletionWait]] = {}
    for wait in root_waits:
        root_waits_by_root.setdefault(wait.consumer_root, []).append(wait)

    return GenericSchedulePlan(
        task_ready_edges=task_ready_edges,
        root_completion_edges=root_completion_edges,
        root_waits_by_root={
            root: tuple(waits) for root, waits in root_waits_by_root.items()
        },
        task_waits_by_root=retained_waits,
        access_cohorts=access_cohorts,
        continuations=continuations,
        continuation_pipelines=pipelines,
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

    def is_ordered(
        producer: int,
        consumer: int,
        edges: set[tuple[int, int]],
    ) -> bool:
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

    waits_by_pair: dict[tuple[int, int], list[RootCompletionWait]] = {}
    for wait in root_wait_candidates:
        waits_by_pair.setdefault((wait.producer_root, wait.consumer_root), []).append(
            wait
        )
    root_completion_waits: list[RootCompletionWait] = []
    ordered_edges = set(preordered_edges)
    for dependency in sorted(
        dependencies,
        key=lambda edge: (
            edge.consumer_root - edge.producer_root,
            edge.producer_root,
            edge.consumer_root,
        ),
    ):
        pair = (dependency.producer_root, dependency.consumer_root)
        if pair in fully_task_ready_edges:
            ordered_edges.add(pair)
            continue
        if is_ordered(*pair, ordered_edges):
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
        ordered_edges.add(pair)
    return tuple(dict.fromkeys(root_completion_waits))


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
    for pair, kinds in kinds_by_pair.items():
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
            and producer.task_count > 0
            and consumer.task_count > 0
            and not producer.has_nontrivial_pid_remap
            and not consumer.has_nontrivial_pid_remap
            and kinds == {TileDependencyKind.READ_AFTER_WRITE}
            and bool(task_waits)
        )
        producer_axes = set(producer.physical_axis_order)
        consumer_axes = set(consumer.physical_axis_order)
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
                producer.physical_axis_order
            ):
                continue

            nested_axes = [
                axis for axis in axes if axis.consumer_block_id not in consumer_counts
            ]
            if len(nested_axes) != 1:
                continue
            stream_axis = nested_axes[0]
            valid = True
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
                    if (
                        axis.consumer_block_id in seen_consumer_axes
                        or producer_counts[axis.producer_block_id] != consumer_count
                        or producer_width != consumer_width
                        or axis.producer_offset != axis.consumer_offset
                    ):
                        valid = False
                        break
                    seen_consumer_axes.add(axis.consumer_block_id)
                if not valid:
                    break
            if not valid:
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
                consumer_fanout == 1
                and producer.task_count > physical_worker_limit
                and len(stream_coordinate_expressions) == 1
                and None not in stream_coordinate_expressions
            )
            if per_coordinate:
                consumer_stream_coordinate = next(iter(stream_coordinate_expressions))
                assert consumer_stream_coordinate is not None
                stage_sizes: tuple[int, ...] = ()
            else:
                consumer_stream_coordinate = None
                first_stage = 1 << ((stream_count // 2).bit_length() - 1)
                mutable_stage_sizes = [first_stage]
                remaining = stream_count - first_stage
                while remaining:
                    stage = 1 << (remaining.bit_length() - 1)
                    mutable_stage_sizes.append(stage)
                    remaining -= stage
                stage_sizes = tuple(mutable_stage_sizes)
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
                    stage_sizes=stage_sizes,
                    outer_producer_axes=tuple(
                        block_id
                        for block_id in producer.physical_axis_order
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


def _derive_task_continuations(
    *,
    dependency_plan: CrossLoopDependencyPlan,
    waits_by_root: dict[int, tuple[WaitSpec, ...]],
    task_families: tuple[InstantiatedTaskFamily, ...],
) -> tuple[TaskContinuationPlan, ...]:
    result: list[TaskContinuationPlan] = []
    for consumer_root, waits in waits_by_root.items():
        if not waits or any(wait.placement != "root_entry" for wait in waits):
            continue
        event_ids = {wait.event_id for wait in waits}
        if len(event_ids) != 1:
            continue
        event_id = event_ids.pop()
        producer_root = dependency_plan.event(event_id).producer_root
        if {
            edge.producer_root
            for edge in dependency_plan.edges
            if edge.consumer_root == consumer_root
        } != {producer_root}:
            continue
        if any(
            event.producer_root == consumer_root and event.granularity == "root"
            for event in dependency_plan.events
        ):
            continue
        producer = task_families[producer_root]
        consumer = task_families[consumer_root]
        block_sizes = {**producer.block_sizes, **consumer.block_sizes}
        predecessor_maps = tuple(
            wait.predecessor_map for wait in waits if wait.predecessor_map is not None
        )
        if len(predecessor_maps) != len(waits):
            continue
        partition = prove_uniform_task_partition(
            predecessor_maps,
            consumer_axis_order=consumer.physical_axis_order,
            consumer_axis_counts=consumer.axis_counts,
            producer_axis_order=producer.physical_axis_order,
            producer_axis_counts=producer.axis_counts,
            block_sizes=block_sizes,
        )
        if partition is None:
            continue
        result.append(
            TaskContinuationPlan(
                event_id=event_id,
                producer_root=producer_root,
                consumer_root=consumer_root,
                partition=partition,
            )
        )
    continuation_count_by_root: dict[int, int] = {}
    for plan in result:
        for root in (plan.producer_root, plan.consumer_root):
            continuation_count_by_root[root] = (
                continuation_count_by_root.get(root, 0) + 1
            )
    return tuple(
        plan
        for plan in result
        if continuation_count_by_root[plan.consumer_root] == 1
        and continuation_count_by_root[plan.producer_root] == 1
    )


def _derive_task_continuation_pipelines(
    *,
    dependency_plan: CrossLoopDependencyPlan,
    task_continuations: tuple[TaskContinuationPlan, ...],
    access_cohorts: tuple[AccessCohortPlan, ...],
    task_families: tuple[InstantiatedTaskFamily, ...],
    physical_worker_limit: int,
) -> tuple[TaskContinuationPipelinePlan, ...]:
    result: list[TaskContinuationPipelinePlan] = []
    for continuation in task_continuations:
        matching_cohorts = [
            cohort
            for cohort in access_cohorts
            if cohort.producer_root == continuation.consumer_root
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
        } != {continuation.consumer_root}:
            continue
        if any(
            event.producer_root in (continuation.producer_root, downstream_root)
            and event.granularity == "root"
            for event in dependency_plan.events
        ):
            continue

        producer_tasks = continuation.producer_tasks
        consumer_tasks = task_families[downstream_root].task_count
        minimum_workers = (producer_tasks + consumer_tasks + 1) // 2
        worker_count = (
            (minimum_workers + continuation.fanin - 1)
            // continuation.fanin
            * continuation.fanin
        )
        if (
            worker_count > physical_worker_limit
            or worker_count >= producer_tasks
            or consumer_tasks <= 0
        ):
            continue
        tail_producer_tasks = producer_tasks - worker_count
        consumer_worker_begin = worker_count - consumer_tasks
        if tail_producer_tasks > consumer_worker_begin:
            continue

        mapping_family = task_families[continuation.consumer_root]
        if (
            not mapping_family.physical_axis_order
            or mapping_family.physical_axis_order[-1] != cohort.producer_stream_axis
            or mapping_family.physical_axis_order[:-1] != cohort.outer_producer_axes
        ):
            continue
        mapping_counts = mapping_family.axis_counts
        outer_count = math.prod(
            mapping_counts[block_id] for block_id in cohort.outer_producer_axes
        )
        initial_map_tasks = worker_count // continuation.fanin
        if initial_map_tasks % outer_count:
            continue
        initial_stream_tasks = initial_map_tasks // outer_count
        stream_tasks = mapping_counts[cohort.producer_stream_axis]
        if not 0 < initial_stream_tasks < stream_tasks:
            continue

        result.append(
            TaskContinuationPipelinePlan(
                continuation=continuation,
                cohort=cohort,
                worker_count=worker_count,
                consumer_tasks=consumer_tasks,
                initial_stream_tasks=initial_stream_tasks,
            )
        )
    return tuple(result)
