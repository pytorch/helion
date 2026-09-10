from __future__ import annotations

import contextlib
import copy
import dataclasses
import itertools
import pickle
from typing import TYPE_CHECKING
from typing import Literal
from unittest import mock

import sympy
import torch
from torch.utils._sympy.functions import Max as SymbolicMax
from torch.utils._sympy.functions import Min as SymbolicMin

import helion
from helion._compiler import cross_loop_scheduler
from helion._compiler import tile_dependency
from helion._compiler.cross_loop_scheduler import FinalArrivalContinuation
from helion._compiler.cross_loop_scheduler import ReadinessConsumer
from helion._compiler.cross_loop_scheduler import ReadinessCounterPlan
from helion._compiler.cross_loop_scheduler import ReadinessEvent
from helion._compiler.cross_loop_scheduler import ReadinessGraph
from helion._compiler.cross_loop_scheduler import ReadinessProducer
from helion._compiler.cross_loop_scheduler import WorkerSchedule
from helion._compiler.cross_loop_scheduler import WorkerScheduleSegment
from helion._compiler.cross_loop_scheduler import _event_ready_after_worker_steps
from helion._compiler.cross_loop_scheduler import _flat_task_order_relation
from helion._compiler.cross_loop_scheduler import _global_unit_list_schedule
from helion._compiler.cross_loop_scheduler import _has_valid_transient_source_schedule
from helion._compiler.cross_loop_scheduler import _nested_loop_entry_counter
from helion._compiler.cross_loop_scheduler import _root_schedule_traversal
from helion._compiler.cross_loop_scheduler import _segmented_nested_loop_counter
from helion._compiler.cross_loop_scheduler import _select_root_barrier_edges
from helion._compiler.cross_loop_scheduler import _task_order_ordinal_domain
from helion._compiler.cross_loop_scheduler import _task_order_slice
from helion._compiler.cross_loop_scheduler import _validate_worker_schedule_tasks
from helion._compiler.cross_loop_scheduler import (
    build_baseline_worker_schedule as _build_baseline_worker_schedule,
)
from helion._compiler.cross_loop_scheduler import (
    build_readiness_events as _build_readiness_events,
)
from helion._compiler.cross_loop_scheduler import (
    build_readiness_graph as _build_readiness_graph,
)
from helion._compiler.cross_loop_scheduler import (
    build_static_pipeline_plan as _build_static_pipeline_plan,
)
from helion._compiler.cross_loop_scheduler import choose_final_arrival_continuations
from helion._compiler.cross_loop_scheduler import choose_readiness_counters
from helion._compiler.cross_loop_scheduler import derive_final_arrival_continuations
from helion._compiler.cross_loop_scheduler import (
    order_continuation_producers_by_readiness_key,
)
from helion._compiler.cross_loop_scheduler import place_nested_loop_consumers
from helion._compiler.tile_dependency import CoordinateDomain
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import ExecutionSite
from helion._compiler.tile_dependency import TileAccess
from helion._compiler.tile_dependency import _CoordinateRelationPiece
from helion._compiler.tile_dependency import (
    build_tile_dependency_graph as _build_tile_dependency_graph,
)
from helion._compiler.tile_dependency import coordinate_axis_symbol
from helion._compiler.tile_dependency import instantiate_coordinate_domains
from helion._compiler.tile_dependency import instantiate_symbolic_dependencies
from helion._compiler.tile_dependency import pid_task_order
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfNotTriton
from helion._testing import skipIfRefEager
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextlib.contextmanager
def _forbid_schedule_enumeration() -> Iterator[None]:
    """Make any materialized acceptance check fail loudly."""
    cross_loop_scheduler._root_schedule_traversal.cache_clear()
    cross_loop_scheduler.root_barrier_publication_plan.cache_clear()
    error = AssertionError("production acceptance must remain symbolic")
    with (
        mock.patch.object(CoordinateRelation, "materialize", side_effect=error),
        mock.patch.object(CoordinateRelation, "targets", side_effect=error),
        mock.patch.object(WorkerSchedule, "workers_for_root", side_effect=error),
        mock.patch.object(WorkerSchedule, "root_at", side_effect=error),
    ):
        yield


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def streamed_singleton_reduction(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty((batch,), dtype=torch.float32, device=x.device)

    for producer_batch, producer_width in hl.tile([batch, width]):
        tmp[producer_batch, producer_width] = x[producer_batch, producer_width] + 1
    for consumer_batch in hl.tile(batch, block_size=1):
        acc = hl.zeros([consumer_batch], dtype=torch.float32)
        for reduction_width in hl.tile(width, block_size=16):
            acc = acc + torch.sum(
                tmp[consumer_batch, reduction_width].to(torch.float32), dim=-1
            )
        out[consumer_batch] = acc + tmp[consumer_batch, 0].to(torch.float32)
    return out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def nested_store_chain(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for producer_batch in hl.tile(batch, block_size=1):
        for producer_width in hl.tile(width, block_size=16):
            tmp[producer_batch, producer_width] = x[producer_batch, producer_width] + 1
    for consumer_batch, consumer_width in hl.tile([batch, width], block_size=[1, 16]):
        out[consumer_batch, consumer_width] = tmp[consumer_batch, consumer_width] * 2
    return out


def readiness_consumer_source_order(
    readiness_graph: ReadinessGraph,
    readiness_consumer: ReadinessConsumer,
) -> tuple[int, ...]:
    """Return one consumer's exhaustive source order for test materialization."""
    root_axes = readiness_graph.root_domains[
        readiness_consumer.consumer_root
    ].axis_order
    if readiness_consumer.consumer_site_id is None:
        return root_axes
    nested_axes = tuple(
        axis
        for axis in readiness_consumer.keys_by_consumer.source_domain.axis_order
        if axis not in root_axes
    )
    return (*nested_axes, *root_axes)


def required_keys_by_task(
    readiness_graph: ReadinessGraph,
    readiness_consumer: ReadinessConsumer,
) -> CoordinateRelation | None:
    """Project a test readiness consumer onto its owning root tasks."""
    root_domain = readiness_graph.root_domains[readiness_consumer.consumer_root]
    if readiness_consumer.consumer_site_id is None:
        if readiness_consumer.keys_by_consumer.source_domain != root_domain:
            raise ValueError("root readiness consumer has the wrong source domain")
        return readiness_consumer.keys_by_consumer
    return readiness_consumer.keys_by_consumer.project_source(root_domain)


def segment_task_at_index(
    segment: WorkerScheduleSegment,
    task_order_index: int,
) -> int:
    """Materialize one task-order index for small scheduler tests."""
    if not 0 <= task_order_index < segment.task_count:
        raise IndexError(task_order_index)
    logical_order = segment.logical_task_order
    if logical_order is None:
        raise AssertionError("segment has no dense diagnostic traversal")
    source_coordinates = logical_order.source_domain.coordinates(task_order_index)
    targets = logical_order.target_coordinates(source_coordinates)
    if len(targets) != 1:
        raise AssertionError("task-order index does not map to one logical task")
    return logical_order.target_domain.index(
        dict(
            zip(
                logical_order.target_domain.axis_order,
                next(iter(targets)),
                strict=True,
            )
        )
    )


def segment_placement(
    segment: WorkerScheduleSegment,
    task: int,
) -> tuple[int, int] | None:
    """Materialize one task's placement for small scheduler tests."""
    if segment.is_normalized:
        converse = segment.task_order.converse()
        placements = (
            frozenset()
            if converse is None
            else converse.target_coordinates(
                segment.task_order.target_domain.coordinates(task)
            )
        )
        if len(placements) > 1:
            raise AssertionError("symbolic schedule maps one task more than once")
        if not placements:
            return None
        coordinates = dict(
            zip(
                segment.task_order.source_domain.axis_order,
                next(iter(placements)),
                strict=True,
            )
        )
        _launch_stage_axis, worker_axis, wave_axis = (
            segment.task_order.source_domain.axis_order
        )
        return coordinates[worker_axis], coordinates[wave_axis]

    converse = segment.task_order.converse()
    task_order_indices = (
        converse.targets(task)
        if converse is not None
        else frozenset(
            task_order_index
            for task_order_index in range(segment.task_count)
            if segment_task_at_index(segment, task_order_index) == task
        )
    )
    if len(task_order_indices) > 1:
        raise AssertionError("symbolic schedule maps one task more than once")
    if not task_order_indices:
        return None
    dispatch_index = segment.dispatch_index(next(iter(task_order_indices)))
    return (
        segment.worker_begin + dispatch_index % segment.worker_count,
        dispatch_index // segment.worker_count,
    )


def segment_task_at(
    segment: WorkerScheduleSegment,
    worker: int,
    worker_step: int,
) -> int | None:
    """Materialize the task at one segment worker step for small tests."""
    if segment.is_normalized:
        launch_stage_axis, worker_axis, wave_axis = (
            segment.task_order.source_domain.axis_order
        )
        targets = segment.task_order.target_coordinates(
            {
                launch_stage_axis: 1,
                worker_axis: worker,
                wave_axis: worker_step,
            }
        )
        if len(targets) > 1:
            raise AssertionError("one worker wave maps to multiple tasks")
        if not targets:
            return None
        return segment.task_order.target_domain.index(
            dict(
                zip(
                    segment.task_order.target_domain.axis_order,
                    next(iter(targets)),
                    strict=True,
                )
            )
        )

    worker_offset = worker - segment.worker_begin
    if not 0 <= worker_offset < segment.worker_count or worker_step < 0:
        return None
    task_order_index = (
        worker_step * segment.worker_count + worker_offset - segment.dispatch_offset
    )
    if not 0 <= task_order_index < segment.task_count:
        return None
    return segment_task_at_index(segment, task_order_index)


def placement(
    schedule: WorkerSchedule,
    root: int,
    task: int,
) -> tuple[int, int] | None:
    """Materialize one task's placement for small scheduler tests."""
    placements = tuple(
        result
        for segment in schedule.segments_for_root(root)
        if (result := segment_placement(segment, task)) is not None
    )
    if len(placements) > 1:
        raise AssertionError(f"task ({root}, {task}) has multiple placements")
    return placements[0] if placements else None


def task_at(
    schedule: WorkerSchedule,
    worker: int,
    worker_step: int,
) -> tuple[int, int] | None:
    """Materialize the task at one worker step for small tests."""
    tasks = tuple(
        (segment.root, task)
        for segment in schedule.segments
        if (task := segment_task_at(segment, worker, worker_step)) is not None
    )
    if len(tasks) > 1:
        raise AssertionError(f"worker {worker} step {worker_step} has multiple tasks")
    return tasks[0] if tasks else None


def task_order(schedule: WorkerSchedule, root: int) -> tuple[int, ...]:
    """Materialize one root's order for small scheduler tests."""
    placed_tasks: list[tuple[int, int]] = []
    for segment in schedule.segments_for_root(root):
        for task_order_index in range(segment.task_count):
            dispatch_index = segment.dispatch_index(task_order_index)
            task = segment_task_at_index(segment, task_order_index)
            placed_tasks.append((dispatch_index, task))
    placed_tasks.sort()
    if any(
        left_offset == right_offset
        for (left_offset, _), (right_offset, _) in itertools.pairwise(placed_tasks)
    ):
        raise AssertionError(f"root {root} has overlapping schedule segments")
    return tuple(task for _offset, task in placed_tasks)


def _materialized_producer_tasks_by_key(
    readiness_graph: ReadinessGraph,
    event_id: int,
) -> tuple[frozenset[tuple[int, int]], ...]:
    event = readiness_graph.event(event_id)
    result: list[set[tuple[int, int]]] = [
        set() for _ in range(event.readiness_key_count)
    ]
    for readiness_producer in event.producers:
        key_to_tasks = readiness_producer.producers_by_key.project_target(
            readiness_graph.root_domains[readiness_producer.producer_root]
        )
        if key_to_tasks is None:
            raise ValueError("readiness producer cannot be projected onto root tasks")
        tasks_by_key = key_to_tasks.materialize()
        for readiness_key, tasks in enumerate(tasks_by_key):
            result[readiness_key].update(
                (readiness_producer.producer_root, task) for task in tasks
            )
    return tuple(frozenset(tasks) for tasks in result)


def _continuation_producers(
    readiness_graph: ReadinessGraph,
    continuations: tuple[FinalArrivalContinuation, ...],
) -> dict[tuple[int, int], frozenset[tuple[int, int]]]:
    result: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for continuation in continuations:
        event = readiness_graph.event(continuation.event_id)
        readiness_consumer = event.consumers[continuation.consumer_index]
        producers_by_key = _materialized_producer_tasks_by_key(
            readiness_graph,
            continuation.event_id,
        )
        for consumer_task, required_keys in enumerate(
            readiness_consumer.keys_by_consumer.materialize(
                source_axis_order=readiness_consumer_source_order(
                    readiness_graph, readiness_consumer
                )
            )
        ):
            if len(required_keys) != 1:
                raise ValueError(
                    "a final-arrival continuation requires one readiness key per task"
                )
            readiness_key = next(iter(required_keys))
            task = (readiness_consumer.consumer_root, consumer_task)
            if task in result:
                raise ValueError(
                    f"task {task} has multiple final-arrival continuations"
                )
            result[task] = producers_by_key[readiness_key]
    return result


def _static_ancestors(
    task: tuple[int, int],
    *,
    worker_schedule: WorkerSchedule,
    continuation_producers: dict[tuple[int, int], frozenset[tuple[int, int]]],
    cache: dict[tuple[int, int], frozenset[tuple[int, int]]],
    visiting: frozenset[tuple[int, int]] = frozenset(),
) -> frozenset[tuple[int, int]]:
    if task in cache:
        return cache[task]
    if placement(worker_schedule, *task) is not None:
        result = frozenset((task,))
    elif task in visiting:
        raise ValueError("final-arrival continuation graph contains a cycle")
    elif (producer_tasks := continuation_producers.get(task)) is None:
        result = frozenset()
    else:
        result = frozenset(
            ancestor
            for producer_task in producer_tasks
            for ancestor in _static_ancestors(
                producer_task,
                worker_schedule=worker_schedule,
                continuation_producers=continuation_producers,
                cache=cache,
                visiting=visiting | frozenset((task,)),
            )
        )
    cache[task] = result
    return result


def validate_worker_schedule(
    readiness_graph: ReadinessGraph,
    worker_schedule: WorkerSchedule,
    continuations: tuple[FinalArrivalContinuation, ...] = (),
) -> None:
    """Exhaustively validate small schedules without entering production."""
    task_nodes = {
        (root, task)
        for root, domain in enumerate(readiness_graph.root_domains)
        for task in range(domain.size)
    }
    continuation_producers = _continuation_producers(readiness_graph, continuations)
    static_tasks = task_nodes - continuation_producers.keys()
    tasks_by_worker: list[list[tuple[int, tuple[int, int]]]] = [
        [] for _ in range(worker_schedule.worker_count)
    ]
    for root, task in sorted(task_nodes):
        task_placement = placement(worker_schedule, root, task)
        if (root, task) in continuation_producers:
            if task_placement is not None:
                raise ValueError(
                    f"locally executed task ({root}, {task}) also has a static placement"
                )
            continue
        if task_placement is None:
            raise ValueError(f"task ({root}, {task}) has no static placement")
        worker, worker_step = task_placement
        tasks_by_worker[worker].append((worker_step, (root, task)))

    graph_nodes = {("task", root, task) for root, task in static_tasks}
    successors: dict[tuple[str, int, int], set[tuple[str, int, int]]] = {
        node: set() for node in graph_nodes
    }
    indegree = dict.fromkeys(graph_nodes, 0)
    static_ancestors_cache: dict[
        tuple[int, int],
        frozenset[tuple[int, int]],
    ] = {}

    def add_edge(
        producer: tuple[str, int, int],
        consumer: tuple[str, int, int],
    ) -> None:
        if producer not in successors:
            successors[producer] = set()
            indegree[producer] = 0
        if consumer not in successors:
            successors[consumer] = set()
            indegree[consumer] = 0
        if producer == consumer or consumer in successors[producer]:
            return
        successors[producer].add(consumer)
        indegree[consumer] += 1

    for worker_tasks in tasks_by_worker:
        worker_tasks.sort()
        if any(
            left_worker_step == right_worker_step
            for (left_worker_step, _), (right_worker_step, _) in itertools.pairwise(
                worker_tasks
            )
        ):
            raise ValueError("multiple tasks occupy one worker step")
        for (_, producer), (_, consumer) in itertools.pairwise(worker_tasks):
            add_edge(("task", *producer), ("task", *consumer))

    for event in readiness_graph.events:
        producers_by_key = _materialized_producer_tasks_by_key(
            readiness_graph,
            event.event_id,
        )
        consumers_by_key: list[set[tuple[int, int]]] = [
            set() for _ in range(event.readiness_key_count)
        ]
        for readiness_consumer in event.consumers:
            keys_by_task = required_keys_by_task(readiness_graph, readiness_consumer)
            if keys_by_task is None:
                raise ValueError(
                    "readiness consumer cannot be projected onto root tasks"
                )
            for consumer_task, required_keys in enumerate(keys_by_task.materialize()):
                consumer = (readiness_consumer.consumer_root, consumer_task)
                if consumer in continuation_producers:
                    continue
                for readiness_key in required_keys:
                    consumers_by_key[readiness_key].add(consumer)
        for readiness_key, consumers in enumerate(consumers_by_key):
            if not consumers:
                continue
            event_node = ("event", event.event_id, readiness_key)
            for producer in producers_by_key[readiness_key]:
                ancestors = _static_ancestors(
                    producer,
                    worker_schedule=worker_schedule,
                    continuation_producers=continuation_producers,
                    cache=static_ancestors_cache,
                )
                if not ancestors:
                    raise ValueError(f"task {producer} has no executor")
                for ancestor in ancestors:
                    add_edge(("task", *ancestor), event_node)
            for consumer in consumers:
                add_edge(event_node, ("task", *consumer))

    ready = [task for task, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        task = ready.pop()
        visited += 1
        for successor in successors[task]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if visited != len(indegree):
        blocked = sorted(node for node, degree in indegree.items() if degree)
        raise ValueError(
            f"worker schedule contains a dependency/order cycle involving {blocked[:8]}"
        )


def _domain(
    *axis_specs: tuple[int, ...],
    kind: Literal[
        "site", "allocation", "event", "task_order", "worker", "value"
    ] = "site",
    identity: int | None = None,
) -> CoordinateDomain:
    return CoordinateDomain(
        tuple(axis for axis, *_ in axis_specs),
        tuple((axis, count) for axis, count, *_ in axis_specs),
        tuple(
            (axis_spec[0], axis_spec[2])
            for axis_spec in axis_specs
            if len(axis_spec) == 3
        ),
        kind=kind,
        identity=identity,
    )


def _full_point_map(
    source: CoordinateDomain,
    target: CoordinateDomain,
    *target_coordinates: sympy.Expr,
) -> CoordinateRelation:
    return CoordinateRelation.point_map(
        source,
        target,
        (
            (
                tuple(
                    (axis, 0, source.axis_count_expressions[axis], 1)
                    for axis in source.axis_order
                ),
                target_coordinates,
            ),
        ),
    )


def _symbolic_nested_counter_graph(
    batch_size: int | sympy.Expr,
    query_size: int | sympy.Expr,
    nested_extent: int | sympy.Expr,
) -> tuple[ReadinessGraph, ReadinessEvent, ReadinessConsumer]:
    """Build one exact B x Q x nested readiness relation for counter tests."""
    producer_domain, consumer_domain = _identify_root_domains(
        (
            CoordinateDomain(
                (10, 11, 12),
                ((10, batch_size), (11, query_size), (12, nested_extent)),
                ((10, 1), (11, 1), (12, 1)),
                _allow_empty=True,
            ),
            CoordinateDomain(
                (20, 21),
                ((20, batch_size), (21, query_size)),
                ((20, 1), (21, 1)),
                _allow_empty=True,
            ),
        )
    )
    consumer_site_domain = CoordinateDomain(
        (20, 21, 22),
        ((20, batch_size), (21, query_size), (22, nested_extent)),
        ((20, 1), (21, 1), (22, 1)),
        identity=7,
        _allow_empty=True,
    )
    readiness_key_domain = CoordinateDomain(
        (0, 1, 2),
        ((0, batch_size), (1, query_size), (2, nested_extent)),
        kind="event",
        identity=0,
        _allow_empty=True,
    )
    producer = _readiness_producer_from_publication(
        producer_root=0,
        publication=_full_point_map(
            producer_domain,
            readiness_key_domain,
            coordinate_axis_symbol(10),
            coordinate_axis_symbol(11),
            coordinate_axis_symbol(12),
        ),
    )
    consumer = ReadinessConsumer(
        consumer_root=1,
        consumer_site_id=7,
        keys_by_consumer=_full_point_map(
            consumer_site_domain,
            readiness_key_domain,
            coordinate_axis_symbol(20),
            coordinate_axis_symbol(21),
            coordinate_axis_symbol(22),
        ),
        covered_obligations=frozenset(((0, None, 7),)),
    )
    event = ReadinessEvent((producer,), (consumer,))
    return _readiness_graph((producer_domain, consumer_domain), event), event, consumer


def _axis_geometry(
    root_domains: tuple[CoordinateDomain, ...],
) -> dict[int, tuple[int, int]]:
    return {
        axis: (domain.axis_counts[axis], domain.block_sizes[axis])
        for domain in root_domains
        for axis in domain.axis_order
    }


def _identify_root_domains(
    root_domains: tuple[CoordinateDomain, ...],
) -> tuple[CoordinateDomain, ...]:
    return tuple(
        dataclasses.replace(domain, identity=root)
        for root, domain in enumerate(root_domains)
    )


def _configured_domains(
    graph,
    axis_geometry: dict[int, tuple[int, int]],
) -> tuple[tuple[CoordinateDomain, ...], tuple[CoordinateDomain | None, ...]]:
    configured_roots, site_domains = instantiate_coordinate_domains(
        graph,
        axis_geometry=axis_geometry,
    )
    assert all(domain is not None for domain in configured_roots)
    return (
        tuple(domain for domain in configured_roots if domain is not None),
        site_domains,
    )


def _default_root_task_orders(
    root_domains: tuple[CoordinateDomain, ...],
    pid_axis_orders: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[CoordinateRelation, ...]:
    if pid_axis_orders is None:
        pid_axis_orders = tuple(domain.axis_order for domain in root_domains)
    return tuple(
        itertools.starmap(
            pid_task_order,
            zip(
                root_domains,
                pid_axis_orders,
                strict=True,
            ),
        )
    )


def _readiness_graph(
    root_domains: tuple[CoordinateDomain, ...],
    *events: ReadinessEvent,
) -> ReadinessGraph:
    return ReadinessGraph(
        root_task_orders=_default_root_task_orders(root_domains),
        events=events,
    )


def _segment(
    root: int,
    task_order: CoordinateRelation,
    *,
    workers: tuple[int, int],
    dispatch_offset: int,
) -> WorkerScheduleSegment:
    return WorkerScheduleSegment(
        root=root,
        task_order=task_order,
        worker_begin=workers[0],
        worker_count=workers[1],
        dispatch_offset=dispatch_offset,
    )


def _schedule(worker_count: int, *segments: WorkerScheduleSegment) -> WorkerSchedule:
    return WorkerSchedule(worker_count=worker_count, segments=segments)


def _access(
    *,
    root: int,
    allocation_id: int = 0,
    kind: Literal["load", "store"],
    shape: tuple[int, ...] = (128,),
    strides: tuple[int, ...] | None = None,
    block_ids: tuple[int | None, ...] = (0,),
    scales: tuple[int, ...] | None = None,
    offsets: tuple[int | None, ...] | None = None,
    scalar: tuple[bool, ...] | None = None,
    full_slice: tuple[bool, ...] | None = None,
    static_extents: tuple[int | None, ...] | None = None,
    masked: bool = False,
    tensor_name: str = "tmp",
    storage_offset: int = 0,
    layout_is_symbolically_exact: bool = True,
    affine_subscript_ranges=None,
) -> TileAccess:
    if strides is None:
        stride = 1
        reversed_strides = []
        for extent in reversed(shape):
            reversed_strides.append(stride)
            stride *= extent
        strides = tuple(reversed(reversed_strides))
    if scales is None:
        scales = (1,) * len(block_ids)
    if offsets is None:
        offsets = (0,) * len(block_ids)
    return TileAccess(
        access_id=-1,
        memory_op_index=-1,
        graph_id=root,
        root=root,
        allocation_id=allocation_id,
        kind=kind,
        tensor_name=tensor_name,
        tensor_shape=shape,
        tensor_strides=strides,
        storage_offset=storage_offset,
        subscript_dims=tuple(range(len(block_ids))),
        subscript_affine_block_ids=block_ids,
        subscript_index_scales=scales,
        subscript_offsets=offsets,
        subscript_is_scalar=scalar or tuple(False for _ in block_ids),
        has_explicit_mask=masked,
        subscript_is_full_slice=full_slice or tuple(False for _ in block_ids),
        subscript_static_extents=static_extents or (),
        layout_is_symbolically_exact=layout_is_symbolically_exact,
        affine_subscript_ranges=affine_subscript_ranges,
    )


def _dependency_graph(
    root_axes: list[list[int]],
    *accesses: TileAccess,
):
    return _build_tile_dependency_graph(
        tuple(
            dataclasses.replace(
                access,
                access_id=access_id,
                memory_op_index=access_id,
            )
            for access_id, access in enumerate(accesses)
        ),
        root_axes,
    )


def _one_dimensional_domains(
    *,
    producer_count: int = 8,
    consumer_count: int = 8,
    producer_block: int = 16,
    consumer_block: int = 16,
) -> tuple[CoordinateDomain, CoordinateDomain]:
    return (
        _domain((10, producer_count, producer_block)),
        _domain((20, consumer_count, consumer_block)),
    )


def _configured_readiness_graph(
    graph,
    root_domains: tuple[CoordinateDomain, ...],
    *,
    axis_geometry: dict[int, tuple[int, int]] | None = None,
    pid_axis_orders: tuple[tuple[int, ...], ...] | None = None,
    publishable_site_ids: frozenset[int] | None = None,
) -> ReadinessGraph:
    if axis_geometry is None:
        axis_geometry = _axis_geometry(root_domains)
    configured_root_domains, site_domains = _configured_domains(graph, axis_geometry)
    return _build_readiness_graph(
        graph,
        root_task_orders=_default_root_task_orders(
            configured_root_domains,
            pid_axis_orders,
        ),
        site_domains=site_domains,
        publishable_site_ids=publishable_site_ids,
    )


def _configured_readiness_events(
    graph,
    *,
    axis_geometry: dict[int, tuple[int, int]],
    publishable_site_ids: frozenset[int] | None = None,
):
    root_domains, site_domains = _configured_domains(graph, axis_geometry)
    return _build_readiness_events(
        graph,
        root_domains=root_domains,
        site_domains=site_domains,
        publishable_site_ids=publishable_site_ids,
    )


def _baseline_worker_schedule(
    root_domains: tuple[CoordinateDomain, ...],
    worker_count: int,
    *,
    root_task_orders: tuple[CoordinateRelation, ...] | None = None,
    pid_axis_orders: tuple[tuple[int, ...], ...] | None = None,
) -> WorkerSchedule:
    root_domains = _identify_root_domains(root_domains)
    if root_task_orders is None:
        root_task_orders = _default_root_task_orders(
            root_domains,
            pid_axis_orders,
        )
    return _build_baseline_worker_schedule(
        root_domains,
        root_task_orders,
        worker_count,
    )


def _repeated_phase_worker_schedule(
    root_domains: tuple[CoordinateDomain, ...],
    worker_count: int,
) -> WorkerSchedule:
    """Build the old repeated-frontier relation as a scheduler test input."""
    root_domains = _identify_root_domains(root_domains)
    task_count = root_domains[0].size_expr
    assert all(
        sympy.simplify(domain.size_expr - task_count) == 0
        for domain in root_domains
    )
    minimum_axis = min(
        axis for domain in root_domains for axis in domain.axis_order
    )
    period = len(root_domains)
    schedule_domain = cross_loop_scheduler._worker_schedule_domain(
        worker_count,
        sympy.simplify(
            period
            * cross_loop_scheduler._ceildiv_nonnegative_expression(
                task_count,
                worker_count,
            )
        ),
        (minimum_axis - 3, minimum_axis - 2, minimum_axis - 1),
    )
    return WorkerSchedule(
        worker_count,
        tuple(
            WorkerScheduleSegment(
                root=root,
                task_order=cross_loop_scheduler._parametric_event_frontier_relation(
                    schedule_domain,
                    domain,
                    root,
                    period,
                    worker_count,
                ),
                worker_begin=0,
                worker_count=worker_count,
                dispatch_offset=0,
            )
            for root, domain in enumerate(root_domains)
        ),
    )


def _configured_static_pipeline_plan(
    *,
    dependency_graph,
    root_domains: tuple[CoordinateDomain, ...],
    axis_geometry: dict[int, tuple[int, int]],
    root_task_orders: tuple[CoordinateRelation, ...] | None = None,
    pid_axis_orders: tuple[tuple[int, ...], ...] | None = None,
    **kwargs,
):
    root_domains = _identify_root_domains(root_domains)
    if root_task_orders is None:
        root_task_orders = _default_root_task_orders(
            root_domains,
            pid_axis_orders,
        )
    site_domains = instantiate_coordinate_domains(
        dependency_graph,
        axis_geometry=axis_geometry,
    )[1]
    return _build_static_pipeline_plan(
        dependency_graph=dependency_graph,
        root_task_orders=root_task_orders,
        site_domains=site_domains,
        **kwargs,
    )


def _one_dimensional_task_range(
    domain: CoordinateDomain,
    begin: int,
    count: int,
) -> CoordinateRelation:
    (axis,) = domain.axis_order
    task_order_domain = _domain((axis, count), kind="task_order")
    return _full_point_map(
        task_order_domain, domain, coordinate_axis_symbol(axis) + begin
    )


def _expected_arrivals(
    readiness_key_domain: CoordinateDomain,
    producers: tuple[ReadinessProducer, ...],
) -> tuple[int, ...]:
    result = [0] * readiness_key_domain.size
    for readiness_producer in producers:
        for readiness_key, producer_tasks in enumerate(
            readiness_producer.producers_by_key.materialize()
        ):
            result[readiness_key] += len(producer_tasks)
    return tuple(result)


def _readiness_producer_from_publication(
    producer_root: int,
    publication: CoordinateRelation,
    producer_site_id: int | None = None,
) -> ReadinessProducer:
    producers_by_key = publication.converse()
    assert producers_by_key is not None
    return ReadinessProducer(
        producer_root=producer_root,
        producer_site_id=producer_site_id,
        producers_by_key=producers_by_key,
    )


def _publication(readiness_producer: ReadinessProducer) -> CoordinateRelation:
    publication = readiness_producer.keys_by_producer
    assert publication is not None
    return publication


class TestCrossLoopScheduler(TestCase):
    def test_scalar_frontier_monotonicity_is_proved_symbolically(self) -> None:
        ordinal_domain = _domain((10, 8, 1), kind="task_order")
        frontier_domain = _domain((20, 64, 1), kind="task_order")
        ordinal = coordinate_axis_symbol(10)

        floor_frontier = _full_point_map(
            ordinal_domain,
            frontier_domain,
            16 * sympy.floor(ordinal / 4) + 15,
        )
        wrapping_frontier = _full_point_map(
            ordinal_domain,
            frontier_domain,
            sympy.Mod(ordinal, 4),
        )
        reversed_boundary = CoordinateRelation.point_map(
            ordinal_domain,
            frontier_domain,
            (
                (((10, 0, 4, 1),), (ordinal,)),
                (((10, 4, 8, 1),), (ordinal - 4,)),
            ),
        )

        self.assertTrue(
            cross_loop_scheduler._scalar_relation_is_nondecreasing(floor_frontier)
        )
        self.assertFalse(
            cross_loop_scheduler._scalar_relation_is_nondecreasing(wrapping_frontier)
        )
        self.assertFalse(
            cross_loop_scheduler._scalar_relation_is_nondecreasing(reversed_boundary)
        )

    def test_transient_source_has_launch_stage_zero_relation(
        self,
    ) -> None:
        producer_domain, branch_domain, sink_domain = _identify_root_domains(
            (
                _domain((10, 5, 1)),
                _domain((20, 2, 1)),
                _domain((30, 1, 1)),
            )
        )
        branch_key_domain = _domain((0, 2), kind="event", identity=0)
        branch_event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation(
                        branch_key_domain,
                        producer_domain,
                        (
                            _CoordinateRelationPiece(
                                ((0, 0, 1, 1),),
                                ((10, sympy.Integer(0), sympy.Integer(2), 1),),
                            ),
                            _CoordinateRelationPiece(
                                ((0, 1, 2, 1),),
                                ((10, sympy.Integer(2), sympy.Integer(5), 1),),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=_full_point_map(
                        branch_domain,
                        branch_key_domain,
                        coordinate_axis_symbol(20),
                    ),
                ),
            ),
        )
        sink_key_domain = _domain((0, 1), kind="event", identity=1)
        sink_event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=1,
                    producers_by_key=_full_point_map(
                        sink_key_domain,
                        branch_domain,
                        sympy.Integer(0),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=2,
                    keys_by_consumer=_full_point_map(
                        sink_domain,
                        sink_key_domain,
                        sympy.Integer(0),
                    ),
                ),
            ),
        )
        readiness_graph = _readiness_graph(
            (producer_domain, branch_domain, sink_domain),
            branch_event,
            sink_event,
        )
        baseline = _baseline_worker_schedule(
            readiness_graph.root_domains,
            worker_count=2,
        )

        readiness_counters = (
            ReadinessCounterPlan(
                producers=branch_event.producers,
                consumers=branch_event.consumers,
            ),
            ReadinessCounterPlan(
                producers=sink_event.producers,
                consumers=sink_event.consumers,
            ),
        )
        scheduled = _global_unit_list_schedule(
            readiness_graph,
            baseline,
            readiness_counters,
            frozenset(),
            transient_source_root=0,
        )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        with _forbid_schedule_enumeration():
            self.assertTrue(
                _validate_worker_schedule_tasks(
                    scheduled,
                    readiness_graph.root_task_orders,
                )
            )
            self.assertTrue(
                cross_loop_scheduler._schedule_is_progress_safe(
                    scheduled,
                    readiness_graph,
                    readiness_counters,
                    frozenset(),
                    transient_source_root=0,
                )
            )
            self.assertTrue(
                _has_valid_transient_source_schedule(
                    scheduled,
                    readiness_graph,
                    0,
                    readiness_counters,
                    frozenset(),
                )
            )
        source_segments = scheduled.segments_for_root(0)
        self.assertEqual(len(source_segments), 1)
        launch_stage_axis = scheduled.placement_domain.axis_order[0]
        self.assertTrue(
            all(
                next(
                    (begin, end, step)
                    for axis, begin, end, step in piece.source_bounds_items
                    if axis == launch_stage_axis
                )
                == (0, 1, 1)
                for piece in source_segments[0].task_order.pieces
            )
        )
        self.assertIsNone(source_segments[0].worker_step_bounds(0))
        self.assertIsNone(scheduled.worker_step_bounds_for_root(0))
        self.assertEqual(scheduled.workers_for_root(0), frozenset())
        self.assertEqual(scheduled.active_worker_count_for_root(0), 0)
        with _forbid_schedule_enumeration():
            source_publication = cross_loop_scheduler.root_barrier_publication_plan(
                scheduled,
                0,
                readiness_counters,
            )
        self.assertEqual(source_publication.participant_intervals, ())
        self.assertIsNone(source_publication.participant_order)
        self.assertEqual(source_publication.publications, ())
        self.assertEqual(source_publication.resident_arrival_count, 0)
        self.assertEqual(source_publication.continuation_arrival_count, 0)
        self.assertEqual(source_publication.source_stage_arrival_count, 5)
        self.assertEqual(source_publication.real_arrival_count, 5)
        self.assertEqual(source_publication.effective_arrival_count, 5)
        self.assertEqual(source_publication.maximum_arrival_count, 5)
        self.assertEqual(placement(scheduled, 1, 0), (0, 0))
        self.assertEqual(placement(scheduled, 2, 0), (0, 1))

        with (
            _forbid_schedule_enumeration(),
            self.assertRaisesRegex(
                ValueError,
                "worker schedule does not own each logical task once",
            ),
        ):
            WorkerSchedule(
                scheduled.worker_count,
                (
                    *scheduled.segments,
                    WorkerScheduleSegment(
                        root=0,
                        task_order=readiness_graph.root_task_orders[0],
                        worker_begin=0,
                        worker_count=2,
                        dispatch_offset=4,
                    ),
                ),
            )

    def test_transient_source_inlets_precede_downstream_waits(self) -> None:
        source, early, late, early_sink, late_sink = _identify_root_domains(
            (
                _domain((10, 4, 1)),
                _domain((20, 1, 1)),
                _domain((30, 2, 1)),
                _domain((40, 1, 1)),
                _domain((50, 1, 1)),
            )
        )
        source_key_domain = _domain((0, 2), kind="event", identity=0)
        source_event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation(
                        source_key_domain,
                        source,
                        (
                            _CoordinateRelationPiece(
                                ((0, 0, 1, 1),),
                                ((10, sympy.Integer(0), sympy.Integer(2), 1),),
                            ),
                            _CoordinateRelationPiece(
                                ((0, 1, 2, 1),),
                                ((10, sympy.Integer(2), sympy.Integer(4), 1),),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=_full_point_map(
                        early,
                        source_key_domain,
                        sympy.Integer(0),
                    ),
                ),
                ReadinessConsumer(
                    consumer_root=2,
                    keys_by_consumer=_full_point_map(
                        late,
                        source_key_domain,
                        sympy.Integer(1),
                    ),
                ),
            ),
        )

        def sink_event(
            producer_root: int,
            producer_domain: CoordinateDomain,
            consumer_root: int,
            consumer_domain: CoordinateDomain,
            identity: int,
        ) -> ReadinessEvent:
            key_domain = _domain((0, 1), kind="event", identity=identity)
            return ReadinessEvent(
                producers=(
                    ReadinessProducer(
                        producer_root=producer_root,
                        producers_by_key=CoordinateRelation.total(
                            key_domain,
                            producer_domain,
                        ),
                    ),
                ),
                consumers=(
                    ReadinessConsumer(
                        consumer_root=consumer_root,
                        keys_by_consumer=_full_point_map(
                            consumer_domain,
                            key_domain,
                            sympy.Integer(0),
                        ),
                    ),
                ),
            )

        early_event = sink_event(1, early, 3, early_sink, 1)
        late_event = sink_event(2, late, 4, late_sink, 2)
        graph = _readiness_graph(
            (source, early, late, early_sink, late_sink),
            source_event,
            early_event,
            late_event,
        )
        counters = tuple(
            ReadinessCounterPlan(event.producers, event.consumers)
            for event in (source_event, early_event, late_event)
        )

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                _baseline_worker_schedule(graph.root_domains, worker_count=2),
                counters,
                frozenset(),
                transient_source_root=0,
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        self.assertEqual(task_at(scheduled, 0, 1), (2, 1))
        self.assertEqual(task_at(scheduled, 1, 1), (3, 0))
        self.assertTrue(
            _validate_worker_schedule_tasks(
                scheduled,
                graph.root_task_orders,
            )
        )

    def test_transient_source_ticket_proof_is_independent_of_wave_remainder(
        self,
    ) -> None:
        for source_count in (1, 2, 3, 4, 9, 10_000):
            with self.subTest(source_count=source_count):
                source_domain, consumer_domain = _identify_root_domains(
                    (
                        _domain((10, source_count, 1)),
                        _domain((20, 1, 1)),
                    )
                )
                key_domain = _domain((0, 1), kind="event", identity=0)
                event = ReadinessEvent(
                    producers=(
                        ReadinessProducer(
                            producer_root=0,
                            producers_by_key=CoordinateRelation.total(
                                key_domain,
                                source_domain,
                            ),
                        ),
                    ),
                    consumers=(
                        ReadinessConsumer(
                            consumer_root=1,
                            keys_by_consumer=_full_point_map(
                                consumer_domain,
                                key_domain,
                                sympy.Integer(0),
                            ),
                        ),
                    ),
                )
                readiness_graph = _readiness_graph(
                    (source_domain, consumer_domain),
                    event,
                )
                readiness_counters = (
                    ReadinessCounterPlan(event.producers, event.consumers),
                )
                with _forbid_schedule_enumeration():
                    scheduled = _global_unit_list_schedule(
                        readiness_graph,
                        _baseline_worker_schedule(
                            readiness_graph.root_domains,
                            worker_count=2,
                        ),
                        readiness_counters,
                        frozenset(),
                        transient_source_root=0,
                    )

                    self.assertIsNotNone(scheduled)
                    assert scheduled is not None
                    self.assertIsNotNone(
                        cross_loop_scheduler._source_ticket_order(
                            scheduled,
                            0,
                        )
                    )
                    self.assertTrue(
                        _validate_worker_schedule_tasks(
                            scheduled,
                            readiness_graph.root_task_orders,
                        )
                    )
                    self.assertTrue(
                        cross_loop_scheduler._schedule_is_progress_safe(
                            scheduled,
                            readiness_graph,
                            readiness_counters,
                            frozenset(),
                            transient_source_root=0,
                        )
                    )
                self.assertEqual(len(scheduled.segments_for_root(0)), 1)
                self.assertEqual(placement(scheduled, 1, 0), (0, 0))

    def test_transient_source_selection_requires_oversubscription(self) -> None:
        for source_count, expected in ((2, None), (3, 0), (4, 0), (9, 0)):
            with self.subTest(source_count=source_count):
                source_domain, consumer_domain = _identify_root_domains(
                    (
                        _domain((10, source_count, 1)),
                        _domain((20, 1, 1)),
                    )
                )
                key_domain = _domain((0, 1), kind="event", identity=0)
                event = ReadinessEvent(
                    producers=(
                        ReadinessProducer(
                            producer_root=0,
                            producers_by_key=CoordinateRelation(
                                key_domain,
                                source_domain,
                                (
                                    _CoordinateRelationPiece(
                                        ((0, 0, 1, 1),),
                                        (
                                            (
                                                10,
                                                sympy.Integer(0),
                                                sympy.Integer(1),
                                                1,
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    consumers=(
                        ReadinessConsumer(
                            consumer_root=1,
                            keys_by_consumer=_full_point_map(
                                consumer_domain,
                                key_domain,
                                sympy.Integer(0),
                            ),
                        ),
                    ),
                )
                readiness_graph = _readiness_graph(
                    (source_domain, consumer_domain),
                    event,
                )
                readiness_counters = (
                    ReadinessCounterPlan(event.producers, event.consumers),
                )
                self.assertEqual(
                    cross_loop_scheduler._transient_source_candidate(
                        readiness_graph,
                        readiness_counters,
                        frozenset(),
                        worker_count=2,
                    ),
                    expected,
                )

    def test_transient_source_signal_declines_full_frontiers(self) -> None:
        source_domain, consumer_domain = _identify_root_domains(
            (
                _domain((10, 5, 1)),
                _domain((20, 1, 1)),
            )
        )
        key_domain = _domain((0, 1), kind="event", identity=0)
        consumer = ReadinessConsumer(
            consumer_root=1,
            keys_by_consumer=_full_point_map(
                consumer_domain,
                key_domain,
                sympy.Integer(0),
            ),
        )
        full_event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation.total(
                        key_domain,
                        source_domain,
                    ),
                ),
            ),
            consumers=(consumer,),
        )
        readiness_graph = _readiness_graph(
            (source_domain, consumer_domain),
            full_event,
        )
        schedule = _schedule(
            2,
            _segment(
                1,
                readiness_graph.root_task_orders[1],
                workers=(0, 1),
                dispatch_offset=0,
            ),
        )
        schedule = cross_loop_scheduler._with_transient_source_schedule_segment(
            schedule,
            readiness_graph.root_task_orders,
            0,
        )
        self.assertIsNotNone(schedule)
        assert schedule is not None

        with _forbid_schedule_enumeration():
            full_plan = ReadinessCounterPlan(full_event.producers, (consumer,))
            self.assertFalse(
                cross_loop_scheduler._has_strict_partial_source_signal(
                    readiness_graph,
                    0,
                    (full_plan,),
                    frozenset(),
                )
            )
            self.assertIsNone(
                cross_loop_scheduler._transient_source_candidate(
                    readiness_graph,
                    (full_plan,),
                    frozenset(),
                    worker_count=2,
                )
            )
            self.assertTrue(
                _has_valid_transient_source_schedule(
                    schedule,
                    readiness_graph,
                    0,
                    (full_plan,),
                    frozenset(),
                )
            )
            self.assertTrue(
                _has_valid_transient_source_schedule(
                    schedule,
                    readiness_graph,
                    0,
                    (),
                    frozenset(((0, 1),)),
                )
            )
            partial_producer = ReadinessProducer(
                producer_root=0,
                producers_by_key=CoordinateRelation(
                    key_domain,
                    source_domain,
                    (
                        _CoordinateRelationPiece(
                            ((0, 0, 1, 1),),
                            ((10, sympy.Integer(0), sympy.Integer(2), 1),),
                        ),
                    ),
                ),
            )
            partial_plan = ReadinessCounterPlan((partial_producer,), (consumer,))
            self.assertFalse(
                cross_loop_scheduler._has_strict_partial_source_signal(
                    readiness_graph,
                    0,
                    (partial_plan,),
                    frozenset(((0, 1),)),
                )
            )
            self.assertIsNone(
                cross_loop_scheduler._transient_source_candidate(
                    readiness_graph,
                    (partial_plan,),
                    frozenset(((0, 1),)),
                    worker_count=2,
                )
            )
            self.assertTrue(
                _has_valid_transient_source_schedule(
                    schedule,
                    readiness_graph,
                    0,
                    (partial_plan,),
                    frozenset(((0, 1),)),
                )
            )

    def test_transient_source_combines_producer_arms_before_subset_proof(
        self,
    ) -> None:
        source_domain, consumer_domain = _identify_root_domains(
            (
                _domain((10, 4, 1)),
                _domain((20, 1, 1)),
            )
        )
        key_domain = _domain((0, 1), kind="event", identity=0)
        arms = tuple(
            ReadinessProducer(
                producer_root=0,
                producers_by_key=CoordinateRelation(
                    key_domain,
                    source_domain,
                    (
                        _CoordinateRelationPiece(
                            ((0, 0, 1, 1),),
                            ((10, sympy.Integer(begin), sympy.Integer(end), 1),),
                        ),
                    ),
                ),
            )
            for begin, end in ((0, 2), (2, 4))
        )
        consumer = ReadinessConsumer(
            consumer_root=1,
            keys_by_consumer=_full_point_map(
                consumer_domain,
                key_domain,
                sympy.Integer(0),
            ),
        )
        event = ReadinessEvent(producers=arms, consumers=(consumer,))
        readiness_graph = _readiness_graph(
            (source_domain, consumer_domain),
            event,
        )

        with _forbid_schedule_enumeration():
            plan = ReadinessCounterPlan(arms, (consumer,))
            self.assertFalse(
                cross_loop_scheduler._has_strict_partial_source_signal(
                    readiness_graph,
                    0,
                    (plan,),
                    frozenset(),
                )
            )
            self.assertIsNone(
                cross_loop_scheduler._transient_source_candidate(
                    readiness_graph,
                    (plan,),
                    frozenset(),
                    worker_count=2,
                )
            )

    def test_global_list_schedule_starts_ready_critical_consumer(self) -> None:
        producer_domain, branch_domain, sink_domain = _identify_root_domains(
            (
                _domain((10, 4, 1)),
                _domain((20, 2, 1)),
                _domain((30, 1, 1)),
            )
        )
        branch_key_domain = _domain((0, 2), kind="event", identity=0)
        branch_key = coordinate_axis_symbol(0)
        branch_event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation(
                        branch_key_domain,
                        producer_domain,
                        (
                            _CoordinateRelationPiece(
                                ((0, 0, 2, 1),),
                                ((10, 2 * branch_key, 2 * branch_key + 2, 1),),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=_full_point_map(
                        branch_domain,
                        branch_key_domain,
                        coordinate_axis_symbol(20),
                    ),
                ),
            ),
        )
        sink_key_domain = _domain((0, 1), kind="event", identity=1)
        sink_event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=1,
                    producers_by_key=CoordinateRelation(
                        sink_key_domain,
                        branch_domain,
                        (
                            _CoordinateRelationPiece(
                                ((0, 0, 1, 1),),
                                ((20, sympy.Integer(0), sympy.Integer(1), 1),),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=2,
                    keys_by_consumer=_full_point_map(
                        sink_domain,
                        sink_key_domain,
                        sympy.Integer(0),
                    ),
                ),
            ),
        )
        readiness_graph = _readiness_graph(
            (producer_domain, branch_domain, sink_domain),
            branch_event,
            sink_event,
        )
        baseline = _baseline_worker_schedule(
            readiness_graph.root_domains,
            worker_count=2,
        )

        readiness_counters = (
            ReadinessCounterPlan(
                producers=branch_event.producers,
                consumers=branch_event.consumers,
            ),
            ReadinessCounterPlan(
                producers=sink_event.producers,
                consumers=sink_event.consumers,
            ),
        )
        scheduled = _global_unit_list_schedule(
            readiness_graph,
            baseline,
            readiness_counters,
            frozenset(),
        )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        self.assertEqual(
            tuple(segment.root for segment in scheduled.segments),
            (0, 1, 0, 2, 0, 1),
        )
        self.assertEqual(placement(scheduled, 1, 0), (0, 1))
        self.assertEqual(placement(scheduled, 0, 2), (1, 1))
        with _forbid_schedule_enumeration():
            self.assertTrue(
                _validate_worker_schedule_tasks(
                    scheduled,
                    readiness_graph.root_task_orders,
                )
            )
            self.assertTrue(
                cross_loop_scheduler._schedule_is_progress_safe(
                    scheduled,
                    readiness_graph,
                    readiness_counters,
                    frozenset(),
                )
            )
        validate_worker_schedule(readiness_graph, scheduled)

    def test_event_closing_bonus_matches_effective_criticality(self) -> None:
        branch_b, branch_a, join, sink, side = _identify_root_domains(
            (
                _domain((10, 2, 1)),
                _domain((20, 2, 1)),
                _domain((30, 1, 1)),
                _domain((40, 1, 1)),
                _domain((50, 1, 1)),
            )
        )
        join_keys = _domain((0, 1), kind="event", identity=0)
        join_event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation.total(join_keys, branch_b),
                ),
                ReadinessProducer(
                    producer_root=1,
                    producers_by_key=CoordinateRelation.total(join_keys, branch_a),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=2,
                    keys_by_consumer=_full_point_map(
                        join,
                        join_keys,
                        sympy.Integer(0),
                    ),
                ),
            ),
        )
        sink_keys = _domain((0, 1), kind="event", identity=1)
        sink_event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=2,
                    producers_by_key=CoordinateRelation.total(sink_keys, join),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=3,
                    keys_by_consumer=_full_point_map(
                        sink,
                        sink_keys,
                        sympy.Integer(0),
                    ),
                ),
            ),
        )
        side_keys = _domain((0, 1), kind="event", identity=2)
        side_event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=1,
                    producers_by_key=CoordinateRelation(
                        side_keys,
                        branch_a,
                        (
                            _CoordinateRelationPiece(
                                ((0, 0, 1, 1),),
                                ((20, sympy.Integer(0), sympy.Integer(1), 1),),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=4,
                    keys_by_consumer=_full_point_map(
                        side,
                        side_keys,
                        sympy.Integer(0),
                    ),
                ),
            ),
        )
        graph = _readiness_graph(
            (branch_b, branch_a, join, sink, side),
            join_event,
            sink_event,
            side_event,
        )
        plans = tuple(
            ReadinessCounterPlan(event.producers, event.consumers)
            for event in graph.events
        )

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                _baseline_worker_schedule(graph.root_domains, worker_count=1),
                plans,
                frozenset(),
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        # A0 releases only the slack side branch C.  That must not outrank B0,
        # whose class matches A's but whose canonical root order comes first.
        self.assertEqual(task_at(scheduled, 0, 0), (0, 0))
        validate_worker_schedule(graph, scheduled)

    def test_global_list_schedule_finishes_nearly_ready_event(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (
                _domain((10, 2, 1), (11, 2, 1)),
                _domain((20, 2, 1)),
            )
        )
        key_domain = _domain((0, 2), kind="event", identity=0)
        event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation(
                        key_domain,
                        producer_domain,
                        (
                            _CoordinateRelationPiece(
                                ((0, 0, 2, 1),),
                                (
                                    (
                                        10,
                                        coordinate_axis_symbol(0),
                                        coordinate_axis_symbol(0) + 1,
                                        1,
                                    ),
                                    (11, sympy.Integer(0), sympy.Integer(2), 1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=_full_point_map(
                        consumer_domain,
                        key_domain,
                        coordinate_axis_symbol(20),
                    ),
                ),
            ),
        )
        graph = _readiness_graph((producer_domain, consumer_domain), event)
        plan = ReadinessCounterPlan(event.producers, event.consumers)
        baseline = _baseline_worker_schedule(graph.root_domains, worker_count=2)

        with mock.patch.object(
            cross_loop_scheduler,
            "WorkerSchedule",
            side_effect=ValueError("rejected speculative normalization"),
        ) as constructor:
            fallback = cross_loop_scheduler._consumer_major_producer_order(
                graph,
                baseline,
                (plan,),
                frozenset(),
                excluded_roots=frozenset(),
            )
        self.assertTrue(constructor.called)
        self.assertIs(fallback, baseline)

        proposed = cross_loop_scheduler._event_frontier_list_schedule(
            graph,
            baseline,
            (plan,),
            frozenset(),
        )
        self.assertIsNotNone(proposed)
        assert proposed is not None
        # Source order alternates keys: 0, 1, 0, 1.  Once task 0 is selected,
        # task 2 becomes the final producer of key 0 and takes the second slot.
        producer_zero = placement(proposed, 0, 0)
        producer_two = placement(proposed, 0, 2)
        producer_one = placement(proposed, 0, 1)
        consumer_zero = placement(proposed, 1, 0)
        assert None not in (
            producer_zero,
            producer_two,
            producer_one,
            consumer_zero,
        )
        assert producer_zero is not None
        assert producer_two is not None
        assert producer_one is not None
        assert consumer_zero is not None
        self.assertEqual(producer_zero[1], 0)
        self.assertEqual(producer_two[1], 0)
        self.assertGreater(producer_one[1], 0)
        self.assertGreater(consumer_zero[1], 0)
        self.assertEqual(proposed.worker_step_domain.size, 4)
        validate_worker_schedule(graph, proposed)

        # The proposal exposes the desired early event, but costs one extra
        # unit-task wave.  Selection retains the equally valid baseline.
        scheduled = _global_unit_list_schedule(
            graph,
            baseline,
            (plan,),
            frozenset(),
        )
        self.assertIs(scheduled, baseline)
        self.assertEqual(baseline.worker_step_domain.size, 3)

    def test_schedule_horizon_matches_concrete_worker_step_runs(self) -> None:
        (domain,) = _identify_root_domains((_domain((10, 3, 1)),))
        task_order = pid_task_order(domain, domain.axis_order)
        schedules = (
            _baseline_worker_schedule((domain,), worker_count=2),
            _schedule(
                2,
                _segment(
                    0,
                    task_order,
                    workers=(0, 2),
                    dispatch_offset=4,
                ),
            ),
        )

        for schedule in schedules:
            old_horizon = (
                max(
                    (
                        final_wave
                        for segment in schedule.segments
                        for _begin, _end, _first_wave, final_wave in (
                            segment.worker_step_runs()
                        )
                    ),
                    default=-1,
                )
                + 1
            )
            with _forbid_schedule_enumeration():
                horizon = (
                    cross_loop_scheduler._resident_schedule_occupied_wave_count(
                        schedule
                    )
                )
            self.assertEqual(horizon, old_horizon)

    def test_symbolic_schedule_horizon_is_zero_safe_under_substitution(self) -> None:
        task_count = sympy.Symbol("task_count", integer=True, nonnegative=True)
        domains = tuple(
            CoordinateDomain(
                (axis,),
                ((axis, task_count),),
                ((axis, 16),),
                kind="site",
                identity=root,
                _allow_empty=True,
            )
            for root, axis in enumerate((10, 20, 30))
        )
        schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            domains,
            _default_root_task_orders(domains),
            worker_count=4,
        )

        with _forbid_schedule_enumeration():
            symbolic_horizon = (
                cross_loop_scheduler._resident_schedule_occupied_wave_count(schedule)
            )
        self.assertIsNotNone(symbolic_horizon)
        assert symbolic_horizon is not None

        for concrete_count in (0, 1, 4, 5, 9):
            concrete_schedule = WorkerSchedule(
                schedule.worker_count,
                tuple(
                    dataclasses.replace(
                        segment,
                        task_order=segment.task_order.substitute_parameters(
                            {task_count: concrete_count}
                        ),
                    )
                    for segment in schedule.segments
                ),
            )
            old_horizon = (
                max(
                    (
                        final_wave
                        for segment in concrete_schedule.segments
                        for _begin, _end, _first_wave, final_wave in (
                            segment.worker_step_runs()
                        )
                    ),
                    default=-1,
                )
                + 1
            )
            self.assertEqual(
                int(symbolic_horizon.subs(task_count, concrete_count)),
                old_horizon,
            )

    def test_symbolic_horizon_comparison_matches_concrete_and_keeps_ties(
        self,
    ) -> None:
        def old_horizon(schedule: WorkerSchedule) -> int:
            return (
                max(
                    (
                        final_wave
                        for segment in schedule.segments
                        for _begin, _end, _first_wave, final_wave in (
                            segment.worker_step_runs()
                        )
                    ),
                    default=-1,
                )
                + 1
            )

        for task_count in (1, 3, 4, 5, 8, 9):
            with self.subTest(task_count=task_count):
                domains = _identify_root_domains(
                    tuple(_domain((axis, task_count, 16)) for axis in (10, 20, 30))
                )
                events: list[ReadinessEvent] = []
                for event_id, (producer_root, consumer_root) in enumerate(
                    ((0, 1), (1, 2))
                ):
                    key_domain = _domain(
                        (0, task_count),
                        kind="event",
                        identity=event_id,
                    )
                    events.append(
                        ReadinessEvent(
                            producers=(
                                ReadinessProducer(
                                    producer_root=producer_root,
                                    producers_by_key=_full_point_map(
                                        key_domain,
                                        domains[producer_root],
                                        coordinate_axis_symbol(0),
                                    ),
                                ),
                            ),
                            consumers=(
                                ReadinessConsumer(
                                    consumer_root=consumer_root,
                                    keys_by_consumer=_full_point_map(
                                        domains[consumer_root],
                                        key_domain,
                                        coordinate_axis_symbol(
                                            domains[consumer_root].axis_order[0]
                                        ),
                                    ),
                                ),
                            ),
                        )
                    )
                graph = _readiness_graph(domains, *events)
                plans = tuple(
                    ReadinessCounterPlan(event.producers, event.consumers)
                    for event in events
                )
                baseline = _baseline_worker_schedule(domains, worker_count=4)
                candidate = cross_loop_scheduler._event_frontier_list_schedule(
                    graph,
                    baseline,
                    plans,
                    frozenset(),
                )
                self.assertIsNotNone(candidate)
                assert candidate is not None
                expected = (
                    candidate
                    if old_horizon(candidate) <= old_horizon(baseline)
                    else baseline
                )
                with (
                    _forbid_schedule_enumeration(),
                    mock.patch.object(
                        cross_loop_scheduler,
                        "_event_frontier_list_schedule",
                        return_value=candidate,
                    ),
                ):
                    actual = _global_unit_list_schedule(
                        graph,
                        baseline,
                        plans,
                        frozenset(),
                    )
                self.assertIs(actual, expected)

        delayed_domains = _identify_root_domains(
            (_domain((10, 3, 1)), _domain((20, 3, 1)))
        )
        delayed_orders = _default_root_task_orders(delayed_domains)
        delayed_graph = _readiness_graph(delayed_domains)
        delayed_baseline = _baseline_worker_schedule(
            delayed_domains,
            worker_count=2,
        )
        delayed_candidate = _schedule(
            2,
            _segment(
                0,
                delayed_orders[0],
                workers=(0, 2),
                dispatch_offset=0,
            ),
            _segment(
                1,
                delayed_orders[1],
                workers=(0, 2),
                dispatch_offset=6,
            ),
        )
        self.assertGreater(
            old_horizon(delayed_candidate),
            old_horizon(delayed_baseline),
        )
        with mock.patch.object(
            cross_loop_scheduler,
            "_event_frontier_list_schedule",
            return_value=delayed_candidate,
        ):
            selected = _global_unit_list_schedule(
                delayed_graph,
                delayed_baseline,
                (),
                frozenset(((0, 1),)),
            )
        self.assertIs(selected, delayed_baseline)

        dynamic_count = sympy.Symbol(
            "dynamic_count",
            integer=True,
            nonnegative=True,
        )
        dynamic_domains = tuple(
            CoordinateDomain(
                (axis,),
                ((axis, dynamic_count),),
                ((axis, 16),),
                kind="site",
                identity=root,
                _allow_empty=True,
            )
            for root, axis in enumerate((10, 20, 30))
        )
        dynamic_graph = _readiness_graph(dynamic_domains)
        dynamic_baseline = cross_loop_scheduler._build_root_major_worker_schedule(
            dynamic_domains,
            _default_root_task_orders(dynamic_domains),
            worker_count=4,
        )
        dynamic_candidate = _repeated_phase_worker_schedule(
            dynamic_domains,
            worker_count=4,
        )
        with (
            _forbid_schedule_enumeration(),
            mock.patch.object(
                cross_loop_scheduler,
                "_event_frontier_list_schedule",
                return_value=dynamic_candidate,
            ),
        ):
            selected = _global_unit_list_schedule(
                dynamic_graph,
                dynamic_baseline,
                (),
                frozenset(((0, 1), (1, 2))),
            )
        self.assertIs(selected, dynamic_baseline)

        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        tied_domains = tuple(
            CoordinateDomain(
                (axis,),
                ((axis, 4 * batch),),
                ((axis, 16),),
                kind="site",
                identity=root,
                _allow_empty=True,
            )
            for root, axis in enumerate((10, 20, 30))
        )
        tied_graph = _readiness_graph(tied_domains)
        tied_baseline = cross_loop_scheduler._build_root_major_worker_schedule(
            tied_domains,
            _default_root_task_orders(tied_domains),
            worker_count=4,
        )
        tied_candidate = _repeated_phase_worker_schedule(
            tied_domains,
            worker_count=4,
        )
        with (
            _forbid_schedule_enumeration(),
            mock.patch.object(
                cross_loop_scheduler,
                "_event_frontier_list_schedule",
                return_value=tied_candidate,
            ),
        ):
            selected = _global_unit_list_schedule(
                tied_graph,
                tied_baseline,
                (),
                frozenset(((0, 1), (1, 2))),
            )
        self.assertIs(selected, tied_candidate)

    def test_event_frontier_waits_for_every_join_arm(self) -> None:
        first_domain, second_domain, consumer_domain = _identify_root_domains(
            (
                _domain((10, 4, 1)),
                _domain((20, 4, 1)),
                _domain((30, 1, 1)),
            )
        )
        key_domain = _domain((0, 1), kind="event", identity=0)
        event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation.total(
                        key_domain,
                        first_domain,
                    ),
                ),
                ReadinessProducer(
                    producer_root=1,
                    producers_by_key=CoordinateRelation.total(
                        key_domain,
                        second_domain,
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=2,
                    keys_by_consumer=_full_point_map(
                        consumer_domain,
                        key_domain,
                        sympy.Integer(0),
                    ),
                ),
            ),
        )
        graph = _readiness_graph(
            (first_domain, second_domain, consumer_domain),
            event,
        )
        plan = ReadinessCounterPlan(event.producers, event.consumers)

        scheduled = _global_unit_list_schedule(
            graph,
            _baseline_worker_schedule(graph.root_domains, worker_count=8),
            (plan,),
            frozenset(),
        )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        self.assertEqual(placement(scheduled, 2, 0)[1], 1)
        self.assertTrue(
            all(
                placement(scheduled, root, task)[1] == 0
                for root in (0, 1)
                for task in range(4)
            )
        )
        validate_worker_schedule(graph, scheduled)

    def test_event_frontier_defers_newly_released_join_to_next_wave(self) -> None:
        first_domain, second_domain, consumer_domain, independent_domain = (
            _identify_root_domains(
                (
                    _domain((10, 2, 1)),
                    _domain((20, 2, 1)),
                    _domain((30, 1, 1)),
                    _domain((40, 1, 1)),
                )
            )
        )
        key_domain = _domain((0, 1), kind="event", identity=0)
        event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation.total(
                        key_domain,
                        first_domain,
                    ),
                ),
                ReadinessProducer(
                    producer_root=1,
                    producers_by_key=CoordinateRelation.total(
                        key_domain,
                        second_domain,
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=2,
                    keys_by_consumer=_full_point_map(
                        consumer_domain,
                        key_domain,
                        sympy.Integer(0),
                    ),
                ),
            ),
        )
        graph = _readiness_graph(
            (
                first_domain,
                second_domain,
                consumer_domain,
                independent_domain,
            ),
            event,
        )
        plan = ReadinessCounterPlan(event.producers, event.consumers)

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                _baseline_worker_schedule(graph.root_domains, worker_count=5),
                (plan,),
                frozenset(),
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        self.assertEqual(placement(scheduled, 3, 0)[1], 0)
        self.assertEqual(placement(scheduled, 2, 0)[1], 1)
        self.assertTrue(
            all(
                placement(scheduled, root, task)[1] == 0
                for root in (0, 1)
                for task in range(2)
            )
        )
        validate_worker_schedule(graph, scheduled)

    def test_event_frontier_root_barrier_waits_for_later_producer_wave(
        self,
    ) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (
                _domain((10, 6, 1)),
                _domain((20, 1, 1)),
            )
        )
        graph = _readiness_graph((producer_domain, consumer_domain))

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                _baseline_worker_schedule(graph.root_domains, worker_count=4),
                (),
                frozenset(((0, 1),)),
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        self.assertEqual(placement(scheduled, 0, 5)[1], 1)
        self.assertEqual(placement(scheduled, 1, 0)[1], 2)
        validate_worker_schedule(graph, scheduled)

    def test_event_frontier_uses_every_nested_iteration_key(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (
                _domain((10, 4, 1)),
                _domain((20, 1, 1)),
            )
        )
        nested_domain = _domain((20, 1, 1), (21, 2, 1), identity=7)
        key_domain = _domain((0, 2), kind="event", identity=0)
        producer_key = coordinate_axis_symbol(0)
        event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation(
                        key_domain,
                        producer_domain,
                        (
                            _CoordinateRelationPiece(
                                ((0, 0, 2, 1),),
                                (
                                    (
                                        10,
                                        2 * producer_key,
                                        2 * producer_key + 2,
                                        1,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    consumer_site_id=7,
                    keys_by_consumer=_full_point_map(
                        nested_domain,
                        key_domain,
                        coordinate_axis_symbol(21),
                    ),
                ),
            ),
        )
        graph = _readiness_graph((producer_domain, consumer_domain), event)
        plan = ReadinessCounterPlan(event.producers, event.consumers)

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                _baseline_worker_schedule(graph.root_domains, worker_count=3),
                (plan,),
                frozenset(),
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        self.assertEqual(placement(scheduled, 0, 3)[1], 1)
        self.assertEqual(placement(scheduled, 1, 0)[1], 2)
        validate_worker_schedule(graph, scheduled)

    def test_global_list_schedule_contracts_final_arrival_continuation(self) -> None:
        producer_domain, continuation_domain, independent_domain, sink_domain = (
            _identify_root_domains(
                (
                    _domain((10, 2, 1)),
                    _domain((20, 2, 1)),
                    _domain((30, 2, 1)),
                    _domain((40, 2, 1)),
                )
            )
        )
        first_keys = _domain((0, 2), kind="event", identity=0)
        first_event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=_full_point_map(
                        first_keys,
                        producer_domain,
                        coordinate_axis_symbol(0),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=_full_point_map(
                        continuation_domain,
                        first_keys,
                        coordinate_axis_symbol(20),
                    ),
                ),
            ),
        )
        second_keys = _domain((0, 2), kind="event", identity=1)
        second_event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=1,
                    producers_by_key=_full_point_map(
                        second_keys,
                        continuation_domain,
                        coordinate_axis_symbol(0),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=3,
                    keys_by_consumer=_full_point_map(
                        sink_domain,
                        second_keys,
                        coordinate_axis_symbol(40),
                    ),
                ),
            ),
        )
        graph = _readiness_graph(
            (
                producer_domain,
                continuation_domain,
                independent_domain,
                sink_domain,
            ),
            first_event,
            second_event,
        )
        continuation = FinalArrivalContinuation(event_id=0, consumer_index=0)
        plans = (
            ReadinessCounterPlan(
                first_event.producers,
                first_event.consumers,
                continuation_consumer_index=0,
            ),
            ReadinessCounterPlan(second_event.producers, second_event.consumers),
        )
        baseline = _baseline_worker_schedule(
            graph.root_domains,
            worker_count=4,
        ).without_roots(frozenset((1,)))

        scheduled = _global_unit_list_schedule(
            graph,
            baseline,
            plans,
            frozenset(),
        )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        self.assertEqual(scheduled.segments_for_root(1), ())
        self.assertEqual(placement(scheduled, 0, 0), (0, 0))
        self.assertEqual(placement(scheduled, 2, 0), (2, 0))
        self.assertEqual(placement(scheduled, 3, 0), (0, 1))
        with _forbid_schedule_enumeration():
            self.assertTrue(
                cross_loop_scheduler._schedule_is_progress_safe(
                    scheduled,
                    graph,
                    plans,
                    frozenset(),
                )
            )
        validate_worker_schedule(graph, scheduled, (continuation,))

    def test_independent_large_domain_returns_without_enumeration(self) -> None:
        (domain,) = _identify_root_domains((_domain((10, 50_000_000, 1)),))
        graph = _readiness_graph((domain,))
        baseline = _baseline_worker_schedule(graph.root_domains, worker_count=148)

        with (
            _forbid_schedule_enumeration(),
            mock.patch.object(
                cross_loop_scheduler,
                "_has_acyclic_symbolic_segment_precedence",
                side_effect=AssertionError("production proof built a segment DAG"),
            ),
        ):
            scheduled = _global_unit_list_schedule(
                graph,
                baseline,
                (),
                frozenset(),
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        self.assertIs(scheduled, baseline)
        self.assertEqual(
            sum(segment.task_count for segment in scheduled.segments),
            50_000_000,
        )

    def test_worker_schedule_normalizes_dense_runs_to_one_schedule_domain(
        self,
    ) -> None:
        first_domain, second_domain = _identify_root_domains(
            (_domain((10, 5, 1)), _domain((20, 3, 1)))
        )
        schedule = _schedule(
            4,
            _segment(
                0,
                pid_task_order(first_domain, first_domain.axis_order),
                workers=(0, 4),
                dispatch_offset=0,
            ),
            _segment(
                1,
                pid_task_order(second_domain, second_domain.axis_order),
                workers=(0, 4),
                dispatch_offset=8,
            ),
        )

        self.assertTrue(all(segment.is_normalized for segment in schedule.segments))
        self.assertTrue(
            all(
                segment.task_order.source_domain == schedule.placement_domain
                for segment in schedule.segments
            )
        )
        self.assertEqual(
            [placement(schedule, 0, task) for task in range(first_domain.size)],
            [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1)],
        )
        self.assertEqual(
            [placement(schedule, 1, task) for task in range(second_domain.size)],
            [(0, 2), (1, 2), (2, 2)],
        )

    def test_normalized_segment_task_count_uses_relation_support(self) -> None:
        target_domain = _identify_root_domains((_domain((10, 3, 1)),))[0]
        schedule_domain = CoordinateDomain(
            axis_order=(-3, -2, -1),
            axis_counts_items=((-3, 2), (-2, 4), (-1, 3)),
            kind="worker",
        )
        worker = coordinate_axis_symbol(-2)
        relation = CoordinateRelation.point_map(
            schedule_domain,
            target_domain,
            (
                (
                    ((-3, 1, 2, 1), (-2, 1, 4, 1), (-1, 1, 2, 1)),
                    (worker - 1,),
                ),
            ),
        )
        segment = WorkerScheduleSegment(0, relation, 1, 3, 3)
        schedule = WorkerSchedule(4, (segment,))

        self.assertEqual(schedule.segments[0].task_count, 3)
        self.assertEqual(schedule.segments[0].task_order.source_domain.size, 24)
        self.assertEqual(
            [placement(schedule, 0, task) for task in range(target_domain.size)],
            [(1, 1), (2, 1), (3, 1)],
        )
        root_placement = cross_loop_scheduler._root_task_placement_relation(
            schedule,
            0,
        )
        self.assertIsNotNone(root_placement)
        assert root_placement is not None
        self.assertTrue(root_placement.is_total_function())

    def test_worker_schedule_rejects_equal_cardinality_with_duplicate_tasks(
        self,
    ) -> None:
        target_domain = _identify_root_domains((_domain((10, 2, 1)),))[0]
        schedule_domain = CoordinateDomain(
            axis_order=(-3, -2, -1),
            axis_counts_items=((-3, 2), (-2, 2), (-1, 1)),
            kind="worker",
        )
        duplicate = CoordinateRelation.point_map(
            schedule_domain,
            target_domain,
            (
                (
                    ((-3, 1, 2, 1), (-2, 0, 2, 1), (-1, 0, 1, 1)),
                    (sympy.Integer(0),),
                ),
            ),
        )

        self.assertEqual(duplicate.source_support_cardinality(), 2)
        with self.assertRaisesRegex(ValueError, "own each logical task once"):
            WorkerSchedule(
                2,
                (WorkerScheduleSegment(0, duplicate, 0, 2, 0),),
            )

    def test_worker_schedule_rejects_symbolic_overlapping_support(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        first_domain, second_domain = _identify_root_domains(
            (
                CoordinateDomain((10,), ((10, batch),), ((10, 1),)),
                CoordinateDomain((20,), ((20, batch),), ((20, 1),)),
            )
        )
        schedule_domain = CoordinateDomain(
            axis_order=(-3, -2, -1),
            axis_counts_items=((-3, 2), (-2, 1), (-1, batch)),
            kind="worker",
        )
        wave = coordinate_axis_symbol(-1)

        def placement(domain: CoordinateDomain) -> CoordinateRelation:
            return CoordinateRelation.point_map(
                schedule_domain,
                domain,
                (
                    (
                        ((-3, 1, 2, 1), (-2, 0, 1, 1), (-1, 0, batch, 1)),
                        (wave,),
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "support overlaps"):
            WorkerSchedule(
                1,
                (
                    WorkerScheduleSegment(0, placement(first_domain), 0, 1, 0),
                    WorkerScheduleSegment(1, placement(second_domain), 0, 1, 0),
                ),
            )

    def test_static_pipeline_plan_owns_configured_root_orders(self) -> None:
        (domain,) = _identify_root_domains((_domain((10, 2, 1)),))
        order = pid_task_order(domain, domain.axis_order)
        schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (domain,),
            (order,),
            2,
        )
        plan = cross_loop_scheduler.StaticPipelinePlan(
            worker_schedule=schedule,
            root_task_orders=(order,),
            readiness_counters=(),
            root_barrier_edges=frozenset(),
        )

        self.assertEqual(plan.root_task_orders, (order,))
        self.assertEqual(dataclasses.replace(plan).root_task_orders, (order,))
        publication = plan.root_barrier_publication_plans[0]
        self.assertIsNotNone(publication)
        self.assertIs(
            publication,
            plan.root_barrier_publication_plans[0],
        )
        self.assertEqual(
            publication,
            cross_loop_scheduler.root_barrier_publication_plan(
                schedule,
                0,
            ),
        )

        order_axis = order.source_domain.axis_order[0]
        duplicate = CoordinateRelation.point_map(
            order.source_domain,
            domain,
            (
                (
                    ((order_axis, 0, 2, 1),),
                    (sympy.Integer(0),),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "not an exact bijection"):
            dataclasses.replace(plan, root_task_orders=(duplicate,))

    def test_static_pipeline_plan_rejects_unlowerable_counter(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (_domain((10, 2, 1)), _domain((20, 2, 1)))
        )
        root_task_orders = tuple(
            pid_task_order(domain, domain.axis_order)
            for domain in (producer_domain, consumer_domain)
        )
        worker_schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (producer_domain, consumer_domain),
            root_task_orders,
            2,
        )
        readiness_key_domain = _domain((0, 2), kind="event", identity=0)
        malformed = ReadinessCounterPlan(
            producers=(
                _readiness_producer_from_publication(
                    producer_root=0,
                    publication=_full_point_map(
                        producer_domain,
                        readiness_key_domain,
                        coordinate_axis_symbol(10),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=CoordinateRelation.total(
                        consumer_domain,
                        readiness_key_domain,
                    ),
                ),
            ),
        )

        self.assertIsNone(
            malformed.consumers[0].keys_by_consumer.canonical_single_valued()
        )
        with self.assertRaisesRegex(ValueError, "no exact lowering"):
            cross_loop_scheduler.StaticPipelinePlan(
                worker_schedule=worker_schedule,
                root_task_orders=root_task_orders,
                readiness_counters=(malformed,),
                root_barrier_edges=frozenset(),
            )

    def test_static_pipeline_plan_rejects_resident_continuation_owner(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (_domain((10, 2, 1)), _domain((20, 2, 1)))
        )
        root_task_orders = _default_root_task_orders((producer_domain, consumer_domain))
        worker_schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (producer_domain, consumer_domain),
            root_task_orders,
            2,
        )
        readiness_key_domain = _domain((0, 2), kind="event", identity=0)
        continuation = ReadinessCounterPlan(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=_full_point_map(
                        readiness_key_domain,
                        producer_domain,
                        coordinate_axis_symbol(0),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=_full_point_map(
                        consumer_domain,
                        readiness_key_domain,
                        coordinate_axis_symbol(20),
                    ),
                ),
            ),
            continuation_consumer_index=0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "continuation roots must not retain resident ownership",
        ):
            cross_loop_scheduler.StaticPipelinePlan(
                worker_schedule=worker_schedule,
                root_task_orders=root_task_orders,
                readiness_counters=(continuation,),
                root_barrier_edges=frozenset(),
            )

    def test_counter_lowering_rejects_mixed_and_continuation_nested_plans(
        self,
    ) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (_domain((10, 2, 1)), _domain((20, 2, 1)))
        )
        root_task_orders = tuple(
            pid_task_order(domain, domain.axis_order)
            for domain in (producer_domain, consumer_domain)
        )
        worker_schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (producer_domain, consumer_domain),
            root_task_orders,
            2,
        )
        readiness_key_domain = _domain((0, 2), kind="event", identity=0)
        producer = _readiness_producer_from_publication(
            producer_root=0,
            publication=_full_point_map(
                producer_domain,
                readiness_key_domain,
                coordinate_axis_symbol(10),
            ),
        )
        root_consumer = ReadinessConsumer(
            consumer_root=1,
            keys_by_consumer=_full_point_map(
                consumer_domain,
                readiness_key_domain,
                coordinate_axis_symbol(20),
            ),
        )
        nested_site_id = 100
        nested_domain = CoordinateDomain(
            (20, 21),
            ((20, 2), (21, 2)),
            kind="site",
            identity=nested_site_id,
        )
        nested_consumer = ReadinessConsumer(
            consumer_root=1,
            consumer_site_id=nested_site_id,
            keys_by_consumer=CoordinateRelation.point_map(
                nested_domain,
                readiness_key_domain,
                (
                    (
                        ((20, 0, 2, 1), (21, 0, 2, 1)),
                        (coordinate_axis_symbol(20),),
                    ),
                ),
            ),
        )
        invalid_plans = (
            ReadinessCounterPlan(
                producers=(producer,),
                consumers=(root_consumer, nested_consumer),
            ),
            ReadinessCounterPlan(
                producers=(producer,),
                consumers=(nested_consumer,),
                continuation_consumer_index=0,
            ),
        )

        for plan in invalid_plans:
            with self.subTest(plan=plan):
                self.assertFalse(
                    cross_loop_scheduler._supports_exact_counter_plan_lowering(
                        plan,
                        (producer_domain, consumer_domain),
                    )
                )
                with self.assertRaisesRegex(ValueError, "no exact lowering"):
                    cross_loop_scheduler.StaticPipelinePlan(
                        worker_schedule=worker_schedule,
                        root_task_orders=root_task_orders,
                        readiness_counters=(plan,),
                        root_barrier_edges=frozenset(),
                    )

    def test_counter_lowering_rejects_invalid_nested_endpoint_geometry(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (_domain((10, 2, 1)), _domain((20, 2, 1)))
        )
        root_domains = (producer_domain, consumer_domain)
        readiness_key_domain = _domain((0, 2), kind="event", identity=0)
        key = coordinate_axis_symbol(0)
        root_producer = _readiness_producer_from_publication(
            producer_root=0,
            publication=_full_point_map(
                producer_domain,
                readiness_key_domain,
                coordinate_axis_symbol(10),
            ),
        )
        root_consumer = ReadinessConsumer(
            consumer_root=1,
            keys_by_consumer=_full_point_map(
                consumer_domain,
                readiness_key_domain,
                coordinate_axis_symbol(20),
            ),
        )

        def nested_consumer(site_domain: CoordinateDomain) -> ReadinessConsumer:
            return ReadinessConsumer(
                consumer_root=1,
                consumer_site_id=site_domain.identity,
                keys_by_consumer=CoordinateRelation.point_map(
                    site_domain,
                    readiness_key_domain,
                    (
                        (
                            tuple(
                                (axis, 0, count, 1)
                                for axis, count in site_domain.axis_counts_items
                            ),
                            (sympy.Mod(coordinate_axis_symbol(20), 2),),
                        ),
                    ),
                ),
            )

        def nested_producer(site_domain: CoordinateDomain) -> ReadinessProducer:
            return ReadinessProducer(
                producer_root=0,
                producer_site_id=site_domain.identity,
                producers_by_key=CoordinateRelation(
                    source_domain=readiness_key_domain,
                    target_domain=site_domain,
                    pieces=(
                        _CoordinateRelationPiece(
                            source_bounds_items=((0, 0, 2, 1),),
                            target_ranges=tuple(
                                (axis, key, key + 1, 1)
                                if axis == 10
                                else (axis, sympy.Integer(0), count, 1)
                                for axis, count in site_domain.axis_counts_items
                            ),
                        ),
                    ),
                ),
            )

        invalid_consumer_domains = (
            CoordinateDomain(
                (20, 21, 22),
                ((20, 2), (21, 2), (22, 2)),
                kind="site",
                identity=101,
            ),
            CoordinateDomain(
                (20, 21),
                ((20, 3), (21, 2)),
                kind="site",
                identity=102,
            ),
        )
        invalid_producer_domains = (
            CoordinateDomain(
                (10, 11, 12),
                ((10, 2), (11, 2), (12, 2)),
                kind="site",
                identity=201,
            ),
            CoordinateDomain(
                (10, 11),
                ((10, 3), (11, 2)),
                kind="site",
                identity=202,
            ),
        )
        invalid_plans = (
            *(
                ReadinessCounterPlan(
                    producers=(root_producer,),
                    consumers=(nested_consumer(site_domain),),
                )
                for site_domain in invalid_consumer_domains
            ),
            *(
                ReadinessCounterPlan(
                    producers=(nested_producer(site_domain),),
                    consumers=(root_consumer,),
                )
                for site_domain in invalid_producer_domains
            ),
        )

        for plan in invalid_plans:
            with self.subTest(plan=plan):
                self.assertTrue(
                    all(
                        cross_loop_scheduler._supports_readiness_counter_lowering(
                            producer
                        )
                        for producer in plan.producers
                    )
                )
                self.assertTrue(
                    all(
                        consumer.keys_by_consumer.canonical_single_valued()
                        is not None
                        for consumer in plan.consumers
                    )
                )
                self.assertFalse(
                    cross_loop_scheduler._supports_exact_counter_plan_lowering(
                        plan,
                        root_domains,
                    )
                )

    def test_common_counter_legality_is_specialization_invariant(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)

        def counter(
            extent: int | sympy.Expr,
        ) -> tuple[ReadinessCounterPlan, tuple[CoordinateDomain, ...]]:
            root_domains = _identify_root_domains(
                (
                    CoordinateDomain(
                        (10,),
                        ((10, extent),),
                        ((10, 1),),
                        _allow_empty=True,
                    ),
                    CoordinateDomain(
                        (20,),
                        ((20, extent),),
                        ((20, 1),),
                        _allow_empty=True,
                    ),
                )
            )
            readiness_key_domain = CoordinateDomain(
                (0,),
                ((0, extent),),
                kind="event",
                identity=0,
                _allow_empty=True,
            )
            return (
                ReadinessCounterPlan(
                    producers=(
                        ReadinessProducer(
                            producer_root=0,
                            producers_by_key=CoordinateRelation.point_map(
                                readiness_key_domain,
                                root_domains[0],
                                (
                                    (
                                        ((0, 0, extent, 1),),
                                        (coordinate_axis_symbol(0),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    consumers=(
                        ReadinessConsumer(
                            consumer_root=1,
                            keys_by_consumer=CoordinateRelation.point_map(
                                root_domains[1],
                                readiness_key_domain,
                                (
                                    (
                                        ((20, 0, extent, 1),),
                                        (coordinate_axis_symbol(20),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                root_domains,
            )

        with _forbid_schedule_enumeration():
            for extent in (batch, 0, 1, 4):
                with self.subTest(extent=extent):
                    plan, root_domains = counter(extent)
                    self.assertTrue(
                        cross_loop_scheduler._supports_exact_counter_plan_lowering(
                            plan,
                            root_domains,
                        )
                    )

    def test_epoch_framing_rejects_unbounded_static_sibling(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        root_domains = _identify_root_domains(
            (
                _domain((10, 1, 1)),
                _domain((20, 1, 1)),
                CoordinateDomain((30,), ((30, batch),), ((30, 1),)),
                CoordinateDomain((40,), ((40, batch),), ((40, 1),)),
            )
        )
        root_task_orders = _default_root_task_orders(root_domains)
        worker_schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            root_domains,
            root_task_orders,
            4,
        )

        static_key_domain = _domain((0, 1), kind="event", identity=0)
        static_counter = ReadinessCounterPlan(
            producers=(
                _readiness_producer_from_publication(
                    producer_root=0,
                    publication=_full_point_map(
                        root_domains[0],
                        static_key_domain,
                        sympy.Integer(0),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=_full_point_map(
                        root_domains[1],
                        static_key_domain,
                        sympy.Integer(0),
                    ),
                ),
            ),
        )
        dynamic_key_domain = CoordinateDomain(
            (0,),
            ((0, batch),),
            kind="event",
            identity=1,
        )
        dynamic_counter = ReadinessCounterPlan(
            producers=(
                ReadinessProducer(
                    producer_root=2,
                    producers_by_key=CoordinateRelation.point_map(
                        dynamic_key_domain,
                        root_domains[2],
                        (
                            (
                                ((0, 0, batch, 1),),
                                (coordinate_axis_symbol(0),),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=3,
                    keys_by_consumer=CoordinateRelation.point_map(
                        root_domains[3],
                        dynamic_key_domain,
                        (
                            (
                                ((40, 0, batch, 1),),
                                (coordinate_axis_symbol(40),),
                            ),
                        ),
                    ),
                ),
            ),
        )

        self.assertTrue(
            cross_loop_scheduler._supports_exact_counter_plan_lowering(
                static_counter,
                root_domains,
            )
        )
        self.assertEqual(static_counter.arrival_count_bounds(), (1, 1))
        self.assertTrue(
            cross_loop_scheduler._supports_current_parameterized_renderer(
                dynamic_counter,
                root_domains,
            )
        )
        original_arrival_count_bounds = ReadinessCounterPlan.arrival_count_bounds

        def mixed_bounds(plan: ReadinessCounterPlan) -> tuple[int, int] | None:
            return (
                None
                if plan is static_counter
                else original_arrival_count_bounds(plan)
            )

        with (
            mock.patch.object(
                ReadinessCounterPlan,
                "arrival_count_bounds",
                autospec=True,
                side_effect=mixed_bounds,
            ),
            self.assertRaisesRegex(ValueError, "current renderer"),
        ):
            cross_loop_scheduler.StaticPipelinePlan(
                worker_schedule=worker_schedule,
                root_task_orders=root_task_orders,
                readiness_counters=(static_counter, dynamic_counter),
                root_barrier_edges=frozenset(),
            )

    def test_dense_schedule_rejects_logical_order_with_empty_targets(self) -> None:
        target_domain = _identify_root_domains((_domain((10, 1, 1)),))[0]
        order_domain = CoordinateDomain(
            axis_order=(20,),
            axis_counts_items=((20, 2),),
            kind="task_order",
        )
        logical_order = CoordinateRelation.point_map(
            order_domain,
            target_domain,
            (
                (
                    ((20, 0, 2, 1),),
                    (coordinate_axis_symbol(20) + 1,),
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "logical task order cannot be flattened exactly",
        ):
            WorkerSchedule(
                2,
                (WorkerScheduleSegment(0, logical_order, 0, 2, 0),),
            )

    def test_worker_schedule_ignores_stale_dense_compatibility_fields(self) -> None:
        target_domain = _identify_root_domains((_domain((10, 4, 1)),))[0]
        schedule_domain = CoordinateDomain(
            axis_order=(-3, -2, -1),
            axis_counts_items=((-3, 2), (-2, 4), (-1, 2)),
            kind="worker",
        )
        worker = coordinate_axis_symbol(-2)
        relation = CoordinateRelation.point_map(
            schedule_domain,
            target_domain,
            (
                (
                    ((-3, 1, 2, 1), (-2, 0, 4, 1), (-1, 0, 1, 1)),
                    (worker,),
                ),
            ),
        )

        schedule = WorkerSchedule(
            4,
            (
                # The relation owns four workers in wave zero. These stale
                # fields describe a different in-domain four-slot bijection:
                # workers 0/1 across waves zero and one. Bijection alone must
                # not make that incompatible dispatch certificate valid.
                WorkerScheduleSegment(0, relation, 0, 2, 0),
            ),
        )

        self.assertEqual(schedule.segments[0].task_count, 4)
        self.assertEqual(schedule.workers_for_root(0), frozenset(range(4)))
        self.assertIsNone(schedule.segments[0].logical_task_order)
        self.assertIsNone(schedule.dense_assignment(0))

    def test_task_order_slice_preserves_symbolic_traversal(self) -> None:
        (domain,) = _identify_root_domains((_domain((10, 2, 1), (11, 3, 1)),))
        task_order = pid_task_order(domain, (11, 10))

        middle = _task_order_slice(task_order, 1, 4)

        self.assertIsNotNone(middle)
        assert middle is not None
        self.assertEqual(middle.materialize(), task_order.materialize()[1:5])
        self.assertIsNone(_task_order_slice(task_order, -1, 1))
        self.assertIsNone(_task_order_slice(task_order, 0, 0))
        self.assertIsNone(_task_order_slice(task_order, 4, 3))

    def test_task_order_slice_accepts_symbolic_begin_and_count(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        query = sympy.Symbol("query", integer=True, nonnegative=True)
        domain = CoordinateDomain(
            (10,),
            ((10, batch + query),),
            identity=0,
        )
        task_order = pid_task_order(domain, domain.axis_order)
        variants = (
            ("direct", task_order),
            ("deepcopy", copy.deepcopy(task_order)),
            ("pickle", pickle.loads(pickle.dumps(task_order))),
        )

        with (
            mock.patch.object(
                CoordinateDomain,
                "size",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError("symbolic slice must not request size"),
            ),
            mock.patch.object(
                CoordinateDomain,
                "axis_counts",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "symbolic slice must not request concrete axis counts"
                ),
            ),
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError("symbolic slice must not enumerate"),
            ),
            mock.patch.object(
                CoordinateRelation,
                "_factored_source_support_converse",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "slice composition must retain its exact converse"
                ),
            ),
        ):
            slices = []
            for name, variant in variants:
                with self.subTest(roundtrip=name):
                    sliced = _task_order_slice(variant, batch, query)
                    self.assertIsNotNone(sliced)
                    assert sliced is not None
                    self.assertEqual(sliced.source_domain.size_expr, query)
                    self.assertLessEqual(len(sliced.pieces), len(variant.pieces))
                    self.assertIsNotNone(
                        tile_dependency._memoized_exact_converse(sliced)
                    )
                    converse = sliced.converse()
                    self.assertIsNotNone(converse)
                    assert converse is not None
                    self.assertTrue(converse.is_single_valued())
                    slices.append(sliced)

        for concrete_batch, concrete_query in ((0, 0), (0, 3), (2, 0), (2, 3)):
            substitutions = {batch: concrete_batch, query: concrete_query}
            expected = task_order.substitute_parameters(substitutions).materialize()[
                concrete_batch : concrete_batch + concrete_query
            ]
            for sliced in slices:
                concrete = sliced.substitute_parameters(substitutions)
                self.assertEqual(concrete.materialize(), expected)
                self.assertIsNotNone(concrete.converse())
                if concrete_query == 0:
                    self.assertTrue(concrete.source_domain.size_expr.is_zero)
                    self.assertFalse(concrete.pieces)
                    self.assertIsNotNone(
                        tile_dependency._memoized_exact_converse(concrete)
                    )

    def test_task_order_slice_handles_unaligned_symbolic_bq_prefix(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        query = sympy.Symbol("query", integer=True, nonnegative=True)
        domain = CoordinateDomain(
            (10, 11, 12, 13),
            ((10, 5), (11, 3), (12, batch), (13, query)),
            identity=0,
        )
        task_order = pid_task_order(domain, (11, 10, 12, 13))
        ordinal_begin = batch * query
        task_count = 14 * batch * query
        variants = (
            ("direct", task_order),
            ("deepcopy", copy.deepcopy(task_order)),
            ("pickle", pickle.loads(pickle.dumps(task_order))),
        )

        with (
            mock.patch.object(
                CoordinateDomain,
                "size",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError("symbolic slice must not request size"),
            ),
            mock.patch.object(
                CoordinateDomain,
                "axis_counts",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "symbolic slice must not request concrete axis counts"
                ),
            ),
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError("symbolic slice must not enumerate"),
            ),
            mock.patch.object(
                CoordinateRelation,
                "_factored_source_support_converse",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "slice composition must retain its exact converse"
                ),
            ),
        ):
            slices = []
            for name, variant in variants:
                with self.subTest(roundtrip=name):
                    sliced = _task_order_slice(
                        variant,
                        ordinal_begin,
                        task_count,
                    )
                    self.assertIsNotNone(sliced)
                    assert sliced is not None
                    self.assertEqual(sliced.source_domain.size_expr, task_count)
                    self.assertEqual(len(sliced.pieces), 1)
                    self.assertIsNotNone(
                        tile_dependency._memoized_exact_converse(sliced)
                    )
                    self.assertIsNotNone(sliced.converse())
                    slices.append(sliced)

        for concrete_batch, concrete_query in ((0, 3), (2, 0), (1, 1), (2, 3)):
            substitutions = {batch: concrete_batch, query: concrete_query}
            concrete_order = task_order.substitute_parameters(substitutions)
            concrete_begin = concrete_batch * concrete_query
            concrete_count = 14 * concrete_begin
            expected = concrete_order.materialize()[
                concrete_begin : concrete_begin + concrete_count
            ]
            for sliced in slices:
                concrete_slice = sliced.substitute_parameters(substitutions)
                self.assertEqual(concrete_slice.materialize(), expected)
                self.assertIsNotNone(concrete_slice.converse())
                if not concrete_count:
                    self.assertFalse(concrete_slice.pieces)
            if concrete_count:
                direct = _task_order_slice(
                    concrete_order,
                    concrete_begin,
                    concrete_count,
                )
                self.assertIsNotNone(direct)
                assert direct is not None
                self.assertEqual(expected, direct.materialize())

    def test_task_order_slice_declines_unproved_symbolic_interval(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        query = sympy.Symbol("query", integer=True, nonnegative=True)
        domain = CoordinateDomain(
            (10,),
            ((10, batch * query),),
            identity=0,
        )
        task_order = pid_task_order(domain, domain.axis_order)

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("unsupported slice must decline symbolically"),
        ):
            self.assertIsNone(_task_order_slice(task_order, batch, query))

        positive_batch = sympy.Symbol("positive_batch", integer=True, positive=True)
        positive_query = sympy.Symbol("positive_query", integer=True, positive=True)
        dynamic_count = positive_batch + positive_query
        manual_source = CoordinateDomain(
            (20,),
            ((20, dynamic_count),),
            kind="task_order",
        )
        manual_target = CoordinateDomain(
            (10,),
            ((10, dynamic_count),),
            identity=0,
        )
        source_ordinal = coordinate_axis_symbol(20)
        unsupported_order = CoordinateRelation.point_map(
            manual_source,
            manual_target,
            (
                (
                    ((20, 0, dynamic_count, 1),),
                    (sympy.Mod(source_ordinal + 1, dynamic_count),),
                ),
            ),
        )
        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("unsupported order must not be enumerated"),
        ):
            self.assertIsNone(
                _task_order_slice(
                    unsupported_order,
                    positive_batch,
                    positive_query,
                )
            )

    def test_task_order_slice_retains_exact_converse_before_worker_normalization(
        self,
    ) -> None:
        domain = CoordinateDomain(
            (10, 11),
            ((10, 2), (11, 3)),
            identity=0,
        )
        task_order = pid_task_order(domain, domain.axis_order)
        task_order_converse = task_order.derive_converse_and_target_counts()[0]
        self.assertIsNotNone(task_order_converse)
        assert task_order_converse is not None
        tile_dependency._remember_exact_converse(task_order, task_order_converse)

        with mock.patch.object(
            CoordinateRelation,
            "_factored_source_support_converse",
            new_callable=mock.PropertyMock,
            side_effect=AssertionError("slice proof must be retained early"),
        ):
            prefix = _task_order_slice(task_order, 0, 4)
            suffix = _task_order_slice(task_order, 4, 2)
            leading = _task_order_slice(task_order, 0, 1)
            middle = _task_order_slice(task_order, 1, 4)
            trailing = _task_order_slice(task_order, 5, 1)
            self.assertTrue(
                all(
                    relation is not None
                    and tile_dependency._memoized_exact_converse(relation) is not None
                    for relation in (prefix, suffix, leading, middle, trailing)
                )
            )
            assert prefix is not None and suffix is not None
            assert leading is not None and middle is not None and trailing is not None
            WorkerSchedule(
                4,
                (
                    WorkerScheduleSegment(0, prefix, 0, 4, 0),
                    WorkerScheduleSegment(0, suffix, 0, 4, 4),
                ),
            )
            WorkerSchedule(
                4,
                (
                    WorkerScheduleSegment(0, leading, 0, 4, 0),
                    WorkerScheduleSegment(0, middle, 0, 4, 1),
                    WorkerScheduleSegment(0, trailing, 0, 4, 5),
                ),
            )

        renamed_domain = CoordinateDomain(
            (20,),
            ((20, middle.source_domain.size),),
            kind="task_order",
        )
        proved_renamed = middle.rename_source_axes(renamed_domain)
        self.assertIsNotNone(proved_renamed)
        assert proved_renamed is not None
        self.assertIsNotNone(tile_dependency._memoized_exact_converse(proved_renamed))
        proved_coalesced = middle.coalesce_adjacent_source_boxes(
            fold_static_offsets=True
        )
        self.assertIsNotNone(tile_dependency._memoized_exact_converse(proved_coalesced))

        unproved = dataclasses.replace(middle)
        renamed = unproved.rename_source_axes(renamed_domain)
        self.assertIsNotNone(renamed)
        assert renamed is not None
        self.assertIsNone(tile_dependency._memoized_exact_converse(renamed))

        split = CoordinateRelation(
            source_domain=middle.source_domain,
            target_domain=middle.target_domain,
            pieces=middle.pieces,
        )
        coalesced = split.coalesce_adjacent_source_boxes(fold_static_offsets=True)
        self.assertIsNone(tile_dependency._memoized_exact_converse(coalesced))

    def test_piece_aligned_task_order_slice_retains_exact_converse(self) -> None:
        source = CoordinateDomain(
            (-1, 0),
            ((-1, 4), (0, 3)),
            kind="task_order",
            identity=0,
        )
        target = CoordinateDomain((10,), ((10, 12),), identity=0)
        inner = coordinate_axis_symbol(-1)
        outer = coordinate_axis_symbol(0)
        task_order = CoordinateRelation(
            source,
            target,
            (
                _CoordinateRelationPiece(
                    ((-1, 0, 2, 1), (0, 0, 3, 1)),
                    ((10, 2 * outer + inner, 2 * outer + inner + 1, 1),),
                ),
                _CoordinateRelationPiece(
                    ((-1, 2, 4, 1), (0, 0, 3, 1)),
                    ((10, 2 * outer + inner + 4, 2 * outer + inner + 5, 1),),
                ),
            ),
        )
        task_order_converse = task_order.derive_converse_and_target_counts()[0]
        self.assertIsNotNone(task_order_converse)
        assert task_order_converse is not None
        tile_dependency._remember_exact_converse(task_order, task_order_converse)

        with (
            mock.patch.object(
                CoordinateRelation,
                "_factored_source_support_converse",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "aligned slice proof must be retained early"
                ),
            ),
            mock.patch.object(
                cross_loop_scheduler,
                "_flat_task_order_relation",
                side_effect=AssertionError(
                    "concrete manual slice must use its fallback"
                ),
            ),
        ):
            sliced = _task_order_slice(task_order, 1, 4)
            self.assertIsNotNone(sliced)
            assert sliced is not None
            self.assertIsNotNone(tile_dependency._memoized_exact_converse(sliced))
            self.assertIsNotNone(sliced.converse())
        self.assertEqual(
            sliced.materialize(),
            task_order.materialize()[1:5],
        )

    def test_task_order_slice_preserves_piecewise_dense_bijection(self) -> None:
        target_domain = _domain((10, 1, 1), (11, 1536, 1), identity=17)
        source_domain = _domain((-1, 16), (0, 96), kind="task_order")
        inner = coordinate_axis_symbol(-1)
        outer = coordinate_axis_symbol(0)
        task_order = CoordinateRelation(
            source_domain,
            target_domain,
            (
                _CoordinateRelationPiece(
                    ((-1, 0, 8, 1), (0, 0, 96, 1)),
                    (
                        (10, sympy.Integer(0), sympy.Integer(1), 1),
                        (
                            11,
                            8 * outer + sympy.Mod(inner, 8),
                            8 * outer + sympy.Mod(inner, 8) + 1,
                            1,
                        ),
                    ),
                ),
                _CoordinateRelationPiece(
                    ((-1, 8, 16, 1), (0, 0, 96, 1)),
                    (
                        (10, sympy.Integer(0), sympy.Integer(1), 1),
                        (
                            11,
                            8 * outer + sympy.Mod(inner, 8) + 768,
                            8 * outer + sympy.Mod(inner, 8) + 769,
                            1,
                        ),
                    ),
                ),
            ),
        )

        prefix = _task_order_slice(task_order, 0, 1184)
        suffix = _task_order_slice(task_order, 1184, 352)

        self.assertIsNotNone(prefix)
        self.assertIsNotNone(suffix)
        assert prefix is not None and suffix is not None
        self.assertEqual(prefix.source_domain.shape, (16, 74))
        self.assertEqual(suffix.source_domain.shape, (16, 22))
        self.assertIsNotNone(prefix.converse())
        self.assertIsNotNone(suffix.converse())
        with mock.patch.object(
            tile_dependency,
            "_MAX_RELATION_PIECES",
            15,
        ):
            self.assertIsNone(_task_order_slice(task_order, 1, 1184))
        schedule = WorkerSchedule(
            1184,
            (
                WorkerScheduleSegment(0, prefix, 0, 1184, 0),
                WorkerScheduleSegment(0, suffix, 0, 352, 352),
            ),
        )
        self.assertTrue(_validate_worker_schedule_tasks(schedule, (task_order,)))

    def test_mixed_radix_flattening_stays_compact_and_symbolic(self) -> None:
        target_domain = _domain((10, 1, 1), (11, 1536, 1), identity=17)
        source_domain = _domain((-1, 16), (0, 96), kind="task_order")
        inner = coordinate_axis_symbol(-1)
        outer = coordinate_axis_symbol(0)
        task_order = CoordinateRelation.point_map(
            source_domain,
            target_domain,
            (
                (
                    ((-1, 0, 8, 1), (0, 0, 96, 1)),
                    (sympy.Integer(0), 8 * outer + inner),
                ),
                (
                    ((-1, 8, 16, 1), (0, 0, 96, 1)),
                    (sympy.Integer(0), 8 * outer + inner + 760),
                ),
            ),
        )
        ordinal_domain = _task_order_ordinal_domain(task_order)

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("flattening must remain symbolic"),
        ):
            flattened = _flat_task_order_relation(task_order, ordinal_domain)

        self.assertIsNotNone(flattened)
        assert flattened is not None
        self.assertEqual(len(flattened.pieces), 1)
        self.assertTrue(flattened.is_total_function())
        self.assertIsNotNone(flattened.converse())

        one_dimensional_target = _domain((20, 16, 1), identity=18)
        one_dimensional = pid_task_order(
            one_dimensional_target,
            one_dimensional_target.axis_order,
        )
        one_dimensional_ordinal = _task_order_ordinal_domain(one_dimensional)
        with mock.patch.object(
            CoordinateRelation,
            "coalesce_adjacent_source_boxes",
            side_effect=AssertionError("the native flattening path must win"),
        ):
            native = _flat_task_order_relation(
                one_dimensional,
                one_dimensional_ordinal,
            )
        self.assertIsNotNone(native)

    def test_woven_task_order_has_symbolic_flat_traversal_certificate(self) -> None:
        target = _domain(
            (20, 2, 1),
            (21, 8, 1),
            (22, 44, 1),
            identity=17,
        )
        source = _domain((-1, 32), (0, 2), (1, 11), kind="task_order")
        inner = coordinate_axis_symbol(-1)
        batch = coordinate_axis_symbol(0)
        outer = coordinate_axis_symbol(1)
        woven = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((-1, 0, 32, 1), (0, 0, 2, 1), (1, 0, 11, 1)),
                    (
                        batch,
                        sympy.Mod(inner, 8),
                        4 * outer + sympy.floor(sympy.Mod(inner, 32) / 8),
                    ),
                ),
            ),
        )
        reference = pid_task_order(target, target.axis_order)
        schedule = WorkerSchedule(
            444,
            (WorkerScheduleSegment(0, woven, 0, 444, 0),),
        )

        with _forbid_schedule_enumeration():
            self.assertTrue(_validate_worker_schedule_tasks(schedule, (reference,)))

    def test_reflected_woven_worker_schedule_has_traversal_certificate(self) -> None:
        (target,) = _identify_root_domains((_domain((20, 1, 1), (21, 1536, 1)),))
        schedule_domain = CoordinateDomain(
            axis_order=(-3, -2, -1),
            axis_counts_items=((-3, 2), (-2, 1184), (-1, 14)),
            kind="worker",
        )
        worker = coordinate_axis_symbol(-2)
        wave = coordinate_axis_symbol(-1)
        logical_n = (
            592 * wave
            + sympy.Mod(worker, 8)
            + 8 * sympy.floor(worker / 16)
            - 768 * sympy.floor(sympy.Mod(worker, 16) / 8)
            - 6336
        )
        targets = (
            (20, sympy.Integer(0), sympy.Integer(1), 1),
            (21, logical_n, logical_n + 1, 1),
        )
        placement = CoordinateRelation(
            schedule_domain,
            target,
            (
                _CoordinateRelationPiece(
                    ((-3, 1, 2, 1), (-2, 0, 1184, 1), (-1, 12, 13, 1)),
                    targets,
                ),
                _CoordinateRelationPiece(
                    ((-3, 1, 2, 1), (-2, 0, 352, 1), (-1, 13, 14, 1)),
                    targets,
                ),
            ),
        )
        schedule = WorkerSchedule(
            1184,
            (
                WorkerScheduleSegment(
                    0,
                    placement,
                    0,
                    1184,
                    12 * 1184,
                ),
            ),
        )
        reference = pid_task_order(target, target.axis_order)

        with _forbid_schedule_enumeration():
            traversal = _root_schedule_traversal(schedule.segments, reference)
            self.assertIsNotNone(traversal)
            assert traversal is not None
            self.assertIsNotNone(traversal.scheduled_ordinal_to_logical_task)
            self.assertIsNotNone(traversal.logical_task_to_scheduled_ordinal)
            assert traversal.scheduled_ordinal_to_logical_task is not None
            assert traversal.logical_task_to_scheduled_ordinal is not None
            self.assertTrue(
                traversal.scheduled_ordinal_to_logical_task.is_total_function()
            )
            self.assertTrue(
                traversal.logical_task_to_scheduled_ordinal.is_total_function()
            )
            self.assertTrue(_validate_worker_schedule_tasks(schedule, (reference,)))

    def test_segmented_traversal_proves_exact_once_without_materializing(self) -> None:
        target = _domain((20, 2, 1), (21, 11, 1), identity=19)
        reference = pid_task_order(target, target.axis_order)
        prefix = _task_order_slice(reference, 0, 19)
        suffix = _task_order_slice(reference, 19, 3)
        duplicate = _task_order_slice(reference, 16, 3)
        self.assertIsNotNone(prefix)
        self.assertIsNotNone(suffix)
        self.assertIsNotNone(duplicate)
        assert prefix is not None and suffix is not None and duplicate is not None
        valid = WorkerSchedule(
            22,
            (
                WorkerScheduleSegment(0, prefix, 0, 19, 0),
                WorkerScheduleSegment(0, suffix, 19, 3, 0),
            ),
        )
        with _forbid_schedule_enumeration():
            self.assertTrue(_validate_worker_schedule_tasks(valid, (reference,)))
            traversal = _root_schedule_traversal(valid.segments, reference)
            self.assertIsNotNone(traversal)
            assert traversal is not None
            self.assertTrue(traversal.matches_reference)
        with (
            _forbid_schedule_enumeration(),
            self.assertRaisesRegex(
                ValueError,
                "worker schedule does not own each logical task once",
            ),
        ):
            WorkerSchedule(
                22,
                (
                    WorkerScheduleSegment(0, prefix, 0, 19, 0),
                    WorkerScheduleSegment(0, duplicate, 19, 3, 0),
                ),
            )

    def test_worker_schedule_tuple_order_must_match_each_worker_strand(self) -> None:
        first_domain, second_domain = _identify_root_domains(
            (_domain((10, 1, 1)), _domain((20, 1, 1)))
        )

        with self.assertRaisesRegex(
            ValueError,
            "tuple order disagrees with worker-step order",
        ):
            _schedule(
                1,
                _segment(
                    1,
                    pid_task_order(second_domain, second_domain.axis_order),
                    workers=(0, 1),
                    dispatch_offset=1,
                ),
                _segment(
                    0,
                    pid_task_order(first_domain, first_domain.axis_order),
                    workers=(0, 1),
                    dispatch_offset=0,
                ),
            )

    def test_ready_family_skips_nonmonotone_speculative_placement(self) -> None:
        _first_domain, candidate_domain, later_domain = _identify_root_domains(
            (
                _domain((10, 1, 1)),
                _domain((20, 1, 1)),
                _domain((30, 1, 1)),
            )
        )
        candidate_order = pid_task_order(candidate_domain, candidate_domain.axis_order)
        later_order = pid_task_order(later_domain, later_domain.axis_order)
        schedule = _schedule(
            3,
            _segment(2, later_order, workers=(2, 1), dispatch_offset=9),
            _segment(1, candidate_order, workers=(0, 1), dispatch_offset=10),
        )

        placements = cross_loop_scheduler._family_placements_at_worker_step(
            schedule,
            root=1,
            task_domain=candidate_domain,
            task_order=candidate_order,
            worker_step=5,
            unavailable_workers=frozenset((1,)),
        )

        self.assertEqual(len(placements), 1)
        self.assertEqual(placements[0].root_at(0, 5), 1)
        self.assertIsNone(placements[0].root_at(2, 5))

    def test_worker_schedule_skips_unassigned_segments_without_ordering(self) -> None:
        first_domain, second_domain = _identify_root_domains(
            (_domain((10, 1, 1)), _domain((20, 1, 1)))
        )

        schedule = _schedule(
            2,
            _segment(
                1,
                pid_task_order(second_domain, second_domain.axis_order),
                workers=(1, 1),
                dispatch_offset=1,
            ),
            _segment(
                0,
                pid_task_order(first_domain, first_domain.axis_order),
                workers=(0, 1),
                dispatch_offset=0,
            ),
        )

        self.assertEqual(schedule.root_at(0, 0), 0)
        self.assertEqual(schedule.root_at(1, 1), 1)

    def test_root_publication_plan_partitions_final_worker_occurrences(self) -> None:
        first_domain, second_domain = _identify_root_domains(
            (_domain((10, 6, 1)), _domain((20, 4, 1)))
        )
        first_order = pid_task_order(first_domain, first_domain.axis_order)
        first_prefix = _task_order_slice(first_order, 0, 4)
        first_suffix = _task_order_slice(first_order, 4, 2)
        assert first_prefix is not None and first_suffix is not None
        schedule = _schedule(
            4,
            _segment(0, first_prefix, workers=(0, 4), dispatch_offset=0),
            _segment(
                1,
                pid_task_order(second_domain, second_domain.axis_order),
                workers=(0, 4),
                dispatch_offset=4,
            ),
            _segment(0, first_suffix, workers=(0, 2), dispatch_offset=4),
        )

        with _forbid_schedule_enumeration():
            publication = cross_loop_scheduler.root_barrier_publication_plan(
                schedule,
                0,
            )

        self.assertEqual(publication.participant_intervals, ((0, 4),))
        self.assertEqual(publication.resident_arrival_count, 4)
        self.assertEqual(
            tuple(
                (item.segment_index, item.worker_intervals)
                for item in publication.publications
            ),
            ((0, ((2, 4),)), (2, ((0, 2),))),
        )

    def test_root_publication_plan_owns_continuation_arrival_count(self) -> None:
        producer_domain, continuation_domain = _identify_root_domains(
            (_domain((10, 4, 1)), _domain((20, 3, 1)))
        )
        readiness_key_domain = _domain((0, 3), kind="event", identity=0)
        counter = ReadinessCounterPlan(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=_full_point_map(
                        readiness_key_domain,
                        producer_domain,
                        sympy.Integer(0),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=_full_point_map(
                        continuation_domain,
                        readiness_key_domain,
                        coordinate_axis_symbol(20),
                    ),
                ),
            ),
            continuation_consumer_index=0,
        )
        schedule = _schedule(
            4,
            _segment(
                0,
                pid_task_order(producer_domain, producer_domain.axis_order),
                workers=(0, 4),
                dispatch_offset=0,
            ),
        )

        with _forbid_schedule_enumeration():
            publication = cross_loop_scheduler.root_barrier_publication_plan(
                schedule,
                1,
                (counter,),
            )

        self.assertEqual(publication.participant_intervals, ())
        self.assertIsNone(publication.participant_order)
        self.assertEqual(publication.publications, ())
        self.assertEqual(publication.resident_arrival_count, 0)
        self.assertEqual(publication.continuation_arrival_count, 3)
        self.assertEqual(publication.source_stage_arrival_count, 0)
        self.assertEqual(publication.real_arrival_count, 3)
        self.assertEqual(publication.effective_arrival_count, 3)
        self.assertEqual(publication.maximum_arrival_count, 3)

    def test_parameterized_continuation_only_publication_declines_without_bound(
        self,
    ) -> None:
        task_count = sympy.Symbol("task_count", integer=True, nonnegative=True)
        producer_domain, continuation_domain = _identify_root_domains(
            (
                CoordinateDomain((10,), ((10, task_count),), ((10, 16),)),
                CoordinateDomain((20,), ((20, task_count),), ((20, 16),)),
            )
        )
        readiness_key_domain = CoordinateDomain(
            (0,),
            ((0, task_count),),
            ((0, 16),),
            kind="event",
            identity=0,
        )
        counter = ReadinessCounterPlan(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation.point_map(
                        readiness_key_domain,
                        producer_domain,
                        (
                            (
                                ((0, 0, task_count, 1),),
                                (coordinate_axis_symbol(0),),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=CoordinateRelation.point_map(
                        continuation_domain,
                        readiness_key_domain,
                        (
                            (
                                ((20, 0, task_count, 1),),
                                (coordinate_axis_symbol(20),),
                            ),
                        ),
                    ),
                ),
            ),
            continuation_consumer_index=0,
        )
        schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (producer_domain,),
            (pid_task_order(producer_domain, producer_domain.axis_order),),
            4,
        )

        with (
            _forbid_schedule_enumeration(),
            self.assertRaisesRegex(
                ValueError,
                "positive bounded owner set",
            ),
        ):
            cross_loop_scheduler.root_barrier_publication_plan(
                schedule,
                1,
                (counter,),
            )

    def test_parametric_root_major_schedule_is_exact_after_substitution(self) -> None:
        first_count = sympy.Symbol("first_count", integer=True, nonnegative=True)
        second_count = sympy.Symbol("second_count", integer=True, nonnegative=True)
        root_domains = _identify_root_domains(
            (
                CoordinateDomain((10,), ((10, first_count),), ((10, 16),)),
                CoordinateDomain((20,), ((20, second_count),), ((20, 32),)),
            )
        )
        root_task_orders = _default_root_task_orders(root_domains)

        with _forbid_schedule_enumeration():
            schedule = cross_loop_scheduler._build_root_major_worker_schedule(
                root_domains,
                root_task_orders,
                4,
            )
            geometry = cross_loop_scheduler._parametric_root_major_schedule_geometry(
                schedule
            )
            publications = tuple(
                cross_loop_scheduler.root_barrier_publication_plan(schedule, root)
                for root in range(2)
            )

        self.assertIsNotNone(geometry)
        assert geometry is not None
        self.assertEqual(len(schedule.segments), 2)
        self.assertTrue(
            all(len(segment.task_order.pieces) == 3 for segment in schedule.segments)
        )
        self.assertEqual(
            tuple(plan.participant_intervals for plan in publications),
            ((), ()),
        )
        self.assertEqual(
            tuple(plan.resident_arrival_count for plan in publications),
            (SymbolicMin(4, first_count), SymbolicMin(4, second_count)),
        )
        self.assertEqual(
            tuple(plan.effective_arrival_count for plan in publications),
            (
                SymbolicMax(1, SymbolicMin(4, first_count)),
                SymbolicMax(1, SymbolicMin(4, second_count)),
            ),
        )
        self.assertEqual(
            tuple(plan.maximum_arrival_count for plan in publications),
            (4, 4),
        )
        self.assertTrue(all(plan.unit_contribution == 1 for plan in publications))
        self.assertTrue(
            all(plan.participant_order is not None for plan in publications)
        )
        with self.assertRaisesRegex(ValueError, "must be max\\(real, 1\\)"):
            dataclasses.replace(
                publications[0],
                effective_arrival_count=publications[0].real_arrival_count,
            )
        with self.assertRaisesRegex(ValueError, "exceeds its epoch bound"):
            dataclasses.replace(publications[0], maximum_arrival_count=3)
        self.assertTrue(
            cross_loop_scheduler._parameterized_prerequisites_follow_root_order(
                (),
                frozenset(((0, 1),)),
            )
        )
        self.assertFalse(
            cross_loop_scheduler._parameterized_prerequisites_follow_root_order(
                (),
                frozenset(((1, 0),)),
            )
        )

        launch_stage_axis, worker_axis, wave_axis = schedule.placement_domain.axis_order
        for concrete_counts in (
            (0, 0),
            (1, 3),
            (3, 3),
            (4, 5),
            (5, 8),
            (9, 1),
        ):
            substitutions = dict(
                zip((first_count, second_count), concrete_counts, strict=True)
            )
            first_slot = 0
            for root, (segment, _symbolic_first_slot, _task_count) in enumerate(
                geometry
            ):
                publication = publications[root]
                participant_order = publication.participant_order
                assert participant_order is not None
                concrete_participants = participant_order.substitute_parameters(
                    {
                        symbol: substitutions[symbol]
                        for symbol in participant_order.parameter_symbols
                    }
                )
                actual_participants = {
                    worker: next(iter(targets))
                    for worker in range(4)
                    if (
                        targets := concrete_participants.target_coordinates(
                            {worker_axis: worker}
                        )
                    )
                }
                effective_count = max(min(concrete_counts[root], 4), 1)
                self.assertEqual(
                    actual_participants,
                    {
                        (first_slot + ordinal) % 4: (ordinal,)
                        for ordinal in range(effective_count)
                    },
                )
                self.assertEqual(
                    int(
                        sympy.sympify(publication.real_arrival_count).xreplace(
                            substitutions
                        )
                    ),
                    min(concrete_counts[root], 4),
                )
                self.assertEqual(
                    int(
                        sympy.sympify(publication.effective_arrival_count).xreplace(
                            substitutions
                        )
                    ),
                    effective_count,
                )
                relation = segment.task_order.substitute_parameters(substitutions)
                actual_owners: dict[int, tuple[int, int]] = {}
                wave_count = relation.source_domain.axis_counts[wave_axis]
                assert isinstance(wave_count, int)
                for wave in range(wave_count):
                    self.assertFalse(
                        relation.target_coordinates(
                            {
                                launch_stage_axis: 0,
                                worker_axis: 0,
                                wave_axis: wave,
                            }
                        )
                    )
                    for worker in range(4):
                        targets = relation.target_coordinates(
                            {
                                launch_stage_axis: 1,
                                worker_axis: worker,
                                wave_axis: wave,
                            }
                        )
                        self.assertLessEqual(len(targets), 1)
                        for (task,) in targets:
                            self.assertNotIn(task, actual_owners)
                            actual_owners[task] = (worker, wave)
                expected_count = concrete_counts[root]
                self.assertEqual(
                    actual_owners,
                    {
                        task: (
                            (first_slot + task) % 4,
                            (first_slot + task) // 4,
                        )
                        for task in range(expected_count)
                    },
                )
                first_slot += expected_count

    def test_parametric_packed_root_major_handles_empty_and_excluded_roots(
        self,
    ) -> None:
        counts = tuple(
            sympy.Symbol(f"count_{root}", integer=True, nonnegative=True)
            for root in range(4)
        )
        root_domains = _identify_root_domains(
            tuple(
                CoordinateDomain(
                    (10 + root,),
                    ((10 + root, count),),
                    ((10 + root, 1),),
                )
                for root, count in enumerate(counts)
            )
        )
        with _forbid_schedule_enumeration():
            schedule = cross_loop_scheduler._build_root_major_worker_schedule(
                root_domains,
                _default_root_task_orders(root_domains),
                7,
                excluded_roots=frozenset((1,)),
            )
            geometry = cross_loop_scheduler._parametric_root_major_schedule_geometry(
                schedule
            )
        self.assertIsNotNone(geometry)
        assert geometry is not None
        self.assertEqual(tuple(item[0].root for item in geometry), (0, 2, 3))

        launch_axis, worker_axis, wave_axis = schedule.placement_domain.axis_order
        for concrete_counts in (
            (0, 0, 0, 0),
            (1, 2, 3, 4),
            (7, 0, 7, 1),
            (8, 13, 2, 20),
        ):
            substitutions = dict(zip(counts, concrete_counts, strict=True))
            expected_first_slot = 0
            expected_wave_count = (
                sum(concrete_counts[root] for root in (0, 2, 3)) + 6
            ) // 7
            for segment, symbolic_first_slot, _task_count in geometry:
                self.assertEqual(
                    int(sympy.simplify(symbolic_first_slot.xreplace(substitutions))),
                    expected_first_slot,
                )
                relation = segment.task_order.substitute_parameters(substitutions)
                self.assertEqual(
                    relation.source_domain.axis_counts[wave_axis],
                    expected_wave_count,
                )
                owners: dict[int, tuple[int, int]] = {}
                for wave in range(expected_wave_count):
                    for worker in range(7):
                        for (task,) in relation.target_coordinates(
                            {
                                launch_axis: 1,
                                worker_axis: worker,
                                wave_axis: wave,
                            }
                        ):
                            self.assertNotIn(task, owners)
                            owners[task] = (worker, wave)
                task_count = concrete_counts[segment.root]
                self.assertEqual(
                    owners,
                    {
                        task: (
                            (expected_first_slot + task) % 7,
                            (expected_first_slot + task) // 7,
                        )
                        for task in range(task_count)
                    },
                )
                expected_first_slot += task_count

    def test_parametric_root_major_supports_multiaxis_pid_orders(self) -> None:
        batch = sympy.Symbol("batch", integer=True, positive=True)
        root_domains = _identify_root_domains(
            (
                CoordinateDomain(
                    (10, 11),
                    ((10, batch), (11, 3)),
                    ((10, 1), (11, 16)),
                ),
                CoordinateDomain(
                    (20, 21, 22),
                    ((20, 2), (21, batch), (22, 2)),
                    ((20, 8), (21, 1), (22, 32)),
                ),
            )
        )
        task_axis_orders = ((11, 10), (21, 22, 20))
        root_task_orders = tuple(
            itertools.starmap(
                pid_task_order,
                zip(root_domains, task_axis_orders, strict=True),
            )
        )

        with _forbid_schedule_enumeration():
            schedule = cross_loop_scheduler._build_root_major_worker_schedule(
                root_domains,
                root_task_orders,
                7,
            )
            geometry = cross_loop_scheduler._parametric_root_major_schedule_geometry(
                schedule
            )

        self.assertIsNotNone(geometry)
        launch_axis, worker_axis, wave_axis = schedule.placement_domain.axis_order
        for concrete_batch in (1, 2, 9):
            substitutions = {batch: concrete_batch}
            actual: list[tuple[int, int, tuple[int, ...]]] = []
            for segment in schedule.segments:
                relation = segment.task_order.substitute_parameters(substitutions)
                for wave in range(relation.source_domain.axis_counts[wave_axis]):
                    for worker in range(7):
                        for target in relation.target_coordinates(
                            {
                                launch_axis: 1,
                                worker_axis: worker,
                                wave_axis: wave,
                            }
                        ):
                            actual.append((wave * 7 + worker, segment.root, target))

            expected: list[tuple[int, int, tuple[int, ...]]] = []
            slot = 0
            for root, domain in enumerate(root_domains):
                concrete_domain = domain.substitute_parameters(substitutions)
                for task in range(concrete_domain.size):
                    coordinates = concrete_domain.coordinates(
                        task,
                        linearization_order=task_axis_orders[root],
                    )
                    expected.append(
                        (
                            slot,
                            root,
                            tuple(
                                coordinates[axis] for axis in concrete_domain.axis_order
                            ),
                        )
                    )
                    slot += 1
            self.assertEqual(sorted(actual), expected)

    def test_root_major_symbolic_and_constant_orders_are_identical(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        symbolic_domain = CoordinateDomain(
            (10, 11),
            ((10, batch), (11, 3)),
            ((10, 1), (11, 16)),
            identity=0,
        )
        symbolic_order = pid_task_order(symbolic_domain, (11, 10))

        with _forbid_schedule_enumeration():
            symbolic_schedule = cross_loop_scheduler._build_root_major_worker_schedule(
                (symbolic_domain,),
                (symbolic_order,),
                4,
            )
            concrete_schedules = tuple(
                cross_loop_scheduler._build_root_major_worker_schedule(
                    (concrete_domain,),
                    (symbolic_order.substitute_parameters({batch: concrete_batch}),),
                    4,
                )
                for concrete_batch in (0, 1, 2, 5)
                for concrete_domain in (
                    symbolic_domain.substitute_parameters({batch: concrete_batch}),
                )
            )

        for concrete_batch, concrete_schedule in zip(
            (0, 1, 2, 5), concrete_schedules, strict=True
        ):
            self.assertEqual(
                symbolic_schedule.segments[0]
                .task_order.substitute_parameters({batch: concrete_batch})
                .materialize(),
                concrete_schedule.segments[0].task_order.materialize(),
            )

    def test_root_major_preserves_piecewise_configured_orders(self) -> None:
        l2_domain = _domain((10, 4, 1), (11, 3, 1), identity=0)
        l2_order = pid_task_order(
            l2_domain,
            l2_domain.axis_order,
            l2_group_size=2,
        )

        reflected_domain = _domain((20, 2, 1), (21, 4, 1), identity=0)
        reflected_source = dataclasses.replace(
            reflected_domain,
            kind="task_order",
        )
        reflected_order = CoordinateRelation.point_map(
            reflected_source,
            reflected_domain,
            (
                (
                    ((20, 0, 2, 1), (21, 0, 4, 1)),
                    (
                        coordinate_axis_symbol(20),
                        3 - coordinate_axis_symbol(21),
                    ),
                ),
            ),
        )

        woven_domain = _domain(
            (30, 2, 1),
            (31, 2, 1),
            (32, 4, 1),
            identity=0,
        )
        woven_source = CoordinateDomain(
            (-1, 0, 1),
            ((-1, 4), (0, 2), (1, 2)),
            kind="task_order",
            identity=0,
        )
        inner = coordinate_axis_symbol(-1)
        woven_order = CoordinateRelation.point_map(
            woven_source,
            woven_domain,
            (
                (
                    ((-1, 0, 4, 1), (0, 0, 2, 1), (1, 0, 2, 1)),
                    (
                        coordinate_axis_symbol(0),
                        sympy.Mod(inner, 2),
                        2 * coordinate_axis_symbol(1) + sympy.floor(inner / 2),
                    ),
                ),
            ),
        )

        for domain, configured_order in (
            (l2_domain, l2_order),
            (reflected_domain, reflected_order),
            (woven_domain, woven_order),
        ):
            with (
                self.subTest(configured_order=configured_order),
                _forbid_schedule_enumeration(),
            ):
                schedule = cross_loop_scheduler._build_root_major_worker_schedule(
                    (domain,),
                    (configured_order,),
                    4,
                )
                self.assertTrue(schedule.segments[0].task_order.is_single_valued())
            relation = schedule.segments[0].task_order
            launch_axis, worker_axis, wave_axis = relation.source_domain.axis_order
            actual: list[int] = []
            for ordinal in range(domain.size):
                wave, worker = divmod(ordinal, 4)
                targets = relation.target_coordinates(
                    {
                        launch_axis: 1,
                        worker_axis: worker,
                        wave_axis: wave,
                    }
                )
                self.assertEqual(len(targets), 1)
                target = next(iter(targets))
                actual.append(
                    domain.index(dict(zip(domain.axis_order, target, strict=True)))
                )
            self.assertEqual(
                actual,
                [next(iter(targets)) for targets in configured_order.materialize()],
            )

    def test_root_major_symbolic_configured_orders_match_substitution(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)

        reflected_domain = CoordinateDomain(
            (10, 11),
            ((10, batch), (11, 4)),
            ((10, 1), (11, 1)),
            identity=0,
        )
        reflected_source = dataclasses.replace(
            reflected_domain,
            kind="task_order",
        )
        reflected_order = CoordinateRelation.point_map(
            reflected_source,
            reflected_domain,
            (
                (
                    ((10, 0, batch, 1), (11, 0, 4, 1)),
                    (
                        coordinate_axis_symbol(10),
                        3 - coordinate_axis_symbol(11),
                    ),
                ),
            ),
        )

        woven_domain = CoordinateDomain(
            (20, 21, 22),
            ((20, batch), (21, 2), (22, 4)),
            ((20, 1), (21, 1), (22, 1)),
            identity=1,
        )
        woven_source = CoordinateDomain(
            (-1, 20, 21),
            ((-1, 4), (20, batch), (21, 2)),
            kind="task_order",
            identity=1,
        )
        inner = coordinate_axis_symbol(-1)
        woven_order = CoordinateRelation.point_map(
            woven_source,
            woven_domain,
            (
                (
                    ((-1, 0, 4, 1), (20, 0, batch, 1), (21, 0, 2, 1)),
                    (
                        coordinate_axis_symbol(20),
                        coordinate_axis_symbol(21),
                        2 * sympy.Mod(inner, 2) + sympy.floor(inner / 2),
                    ),
                ),
            ),
        )

        l2_domain = CoordinateDomain(
            (30, 31, 32),
            ((30, 4), (31, 3), (32, batch)),
            ((30, 1), (31, 1), (32, 1)),
            identity=2,
        )
        l2_order = pid_task_order(
            l2_domain,
            l2_domain.axis_order,
            l2_group_size=2,
        )

        for name, domain, configured_order in (
            ("reflected", reflected_domain, reflected_order),
            ("woven", woven_domain, woven_order),
            ("l2", l2_domain, l2_order),
        ):
            with self.subTest(order=name), _forbid_schedule_enumeration():
                symbolic = cross_loop_scheduler._build_root_major_worker_schedule(
                    (domain,),
                    (configured_order,),
                    7,
                )
            for concrete_batch in (0, 1, 8):
                concrete_domain = domain.substitute_parameters(
                    {batch: concrete_batch}
                )
                concrete_order = configured_order.substitute_parameters(
                    {batch: concrete_batch}
                )
                with self.subTest(order=name, batch=concrete_batch):
                    concrete = cross_loop_scheduler._build_root_major_worker_schedule(
                        (concrete_domain,),
                        (concrete_order,),
                        7,
                    )
                    self.assertEqual(
                        symbolic.segments[0]
                        .task_order.substitute_parameters({batch: concrete_batch})
                        .materialize(),
                        concrete.segments[0].task_order.materialize(),
                    )

    def test_parametric_event_frontier_schedule_uses_exact_fan_in_one_counter(
        self,
    ) -> None:
        task_count = sympy.Symbol("task_count", integer=True, nonnegative=True)
        dependency_graph = _dependency_graph(
            [[10], [20]],
            _access(
                root=0,
                kind="store",
                shape=(8192,),
                block_ids=(10,),
            ),
            _access(
                root=1,
                kind="load",
                shape=(8192,),
                block_ids=(20,),
            ),
        )
        root_domains = (
            CoordinateDomain((10,), ((10, task_count),), ((10, 16),)),
            CoordinateDomain((20,), ((20, task_count),), ((20, 16),)),
        )

        with _forbid_schedule_enumeration():
            plan = _configured_static_pipeline_plan(
                dependency_graph=dependency_graph,
                root_domains=root_domains,
                axis_geometry={
                    10: (task_count, 16),
                    20: (task_count, 16),
                },
                worker_count=4,
            )

        self.assertEqual(plan.root_barrier_edges, frozenset())
        self.assertEqual(len(plan.readiness_counters), 1)
        (counter,) = plan.readiness_counters
        self.assertEqual(counter.readiness_key_count_expr, task_count)
        self.assertEqual(counter.uniform_arrival_count(), 1)
        self.assertTrue(
            cross_loop_scheduler._supports_parameterized_fan_in_one_counter(counter)
        )
        schedule_geometry = (
            cross_loop_scheduler._parametric_event_frontier_schedule_geometry(
                plan.worker_schedule
            )
        )
        self.assertIsNotNone(schedule_geometry)
        assert schedule_geometry is not None
        self.assertEqual(
            tuple(
                (segment.root, phase) for segment, phase, _count in schedule_geometry
            ),
            ((0, 0), (1, 1)),
        )
        with self.assertRaisesRegex(
            ValueError,
            "event-frontier schedules do not prove exact",
        ):
            cross_loop_scheduler.root_barrier_publication_plan(
                plan.worker_schedule,
                0,
            )

        for concrete_count in (0, 1, 3, 4, 5, 11):
            publication = counter.producers[0].keys_by_producer
            self.assertIsNotNone(publication)
            assert publication is not None
            concrete_publication = publication.substitute_parameters(
                {task_count: concrete_count}
            )
            concrete_waits = counter.consumers[
                0
            ].keys_by_consumer.substitute_parameters({task_count: concrete_count})
            expected = tuple(frozenset((index,)) for index in range(concrete_count))
            self.assertEqual(concrete_publication.materialize(), expected)
            self.assertEqual(concrete_waits.materialize(), expected)
            for segment, phase, _count in schedule_geometry:
                relation = segment.task_order.substitute_parameters(
                    {task_count: concrete_count}
                )
                launch_axis, worker_axis, wave_axis = relation.source_domain.axis_order
                actual: dict[int, tuple[int, int]] = {}
                for wave in range(relation.source_domain.axis_counts[wave_axis]):
                    for worker in range(4):
                        targets = relation.target_coordinates(
                            {
                                launch_axis: 1,
                                worker_axis: worker,
                                wave_axis: wave,
                            }
                        )
                        for (task,) in targets:
                            self.assertNotIn(task, actual)
                            actual[task] = (worker, wave)
                self.assertEqual(
                    actual,
                    {
                        task: (task % 4, 2 * (task // 4) + phase)
                        for task in range(concrete_count)
                    },
                )

    def test_parametric_event_frontier_declines_ambiguous_fork(self) -> None:
        task_count = sympy.Symbol("task_count", integer=True, nonnegative=True)
        dependency_graph = _dependency_graph(
            [[10], [20], [30]],
            _access(
                root=0,
                allocation_id=0,
                kind="store",
                shape=(8192,),
                block_ids=(10,),
                tensor_name="first_tmp",
            ),
            _access(
                root=0,
                allocation_id=1,
                kind="store",
                shape=(8192,),
                block_ids=(10,),
                tensor_name="second_tmp",
            ),
            _access(
                root=1,
                allocation_id=0,
                kind="load",
                shape=(8192,),
                block_ids=(20,),
                tensor_name="first_tmp",
            ),
            _access(
                root=2,
                allocation_id=1,
                kind="load",
                shape=(8192,),
                block_ids=(30,),
                tensor_name="second_tmp",
            ),
        )
        root_domains = tuple(
            CoordinateDomain((axis,), ((axis, task_count),), ((axis, 16),))
            for axis in (10, 20, 30)
        )

        with _forbid_schedule_enumeration():
            plan = _configured_static_pipeline_plan(
                dependency_graph=dependency_graph,
                root_domains=root_domains,
                axis_geometry=dict.fromkeys((10, 20, 30), (task_count, 16)),
                worker_count=4,
            )

        self.assertEqual(len(plan.readiness_counters), 1)
        self.assertEqual(
            tuple(
                consumer.consumer_root
                for consumer in plan.readiness_counters[0].consumers
            ),
            (1, 2),
        )
        self.assertEqual(plan.root_barrier_edges, frozenset())
        self.assertIsNone(
            cross_loop_scheduler._parametric_event_frontier_schedule_geometry(
                plan.worker_schedule
            )
        )
        self.assertIsNotNone(
            cross_loop_scheduler._parametric_root_major_schedule_geometry(
                plan.worker_schedule
            )
        )

    def test_parametric_event_frontier_matches_concrete_list_scheduler(self) -> None:
        task_count = sympy.Symbol("task_count", integer=True, nonnegative=True)
        dependency_graph = _dependency_graph(
            [[10], [20], [30]],
            _access(
                root=0,
                allocation_id=0,
                kind="store",
                shape=(8192,),
                block_ids=(10,),
                tensor_name="first_tmp",
            ),
            _access(
                root=1,
                allocation_id=0,
                kind="load",
                shape=(8192,),
                block_ids=(20,),
                tensor_name="first_tmp",
            ),
            _access(
                root=1,
                allocation_id=1,
                kind="store",
                shape=(8192,),
                block_ids=(20,),
                tensor_name="second_tmp",
            ),
            _access(
                root=2,
                allocation_id=1,
                kind="load",
                shape=(8192,),
                block_ids=(30,),
                tensor_name="second_tmp",
            ),
        )
        symbolic_domains = tuple(
            CoordinateDomain((axis,), ((axis, task_count),), ((axis, 16),))
            for axis in (10, 20, 30)
        )

        with _forbid_schedule_enumeration():
            symbolic_plan = _configured_static_pipeline_plan(
                dependency_graph=dependency_graph,
                root_domains=symbolic_domains,
                axis_geometry=dict.fromkeys((10, 20, 30), (task_count, 16)),
                worker_count=4,
            )
        symbolic_geometry = (
            cross_loop_scheduler._parametric_event_frontier_schedule_geometry(
                symbolic_plan.worker_schedule
            )
        )
        self.assertIsNotNone(symbolic_geometry)
        assert symbolic_geometry is not None

        for concrete_count in (1, 3, 4, 5, 8, 9):
            concrete_domains = tuple(
                _domain((axis, concrete_count, 16)) for axis in (10, 20, 30)
            )
            readiness_graph = _configured_readiness_graph(
                dependency_graph,
                concrete_domains,
                axis_geometry=dict.fromkeys((10, 20, 30), (concrete_count, 16)),
            )
            initial_schedule = _baseline_worker_schedule(
                concrete_domains,
                4,
                root_task_orders=readiness_graph.root_task_orders,
            )
            readiness_counters, root_barrier_edges = (
                cross_loop_scheduler._finalize_emitted_synchronization(
                    dependency_graph=dependency_graph,
                    root_domains=readiness_graph.root_domains,
                    readiness_counters=choose_readiness_counters(
                        readiness_graph,
                        (),
                    ),
                )
            )
            concrete_schedule = cross_loop_scheduler._event_frontier_list_schedule(
                readiness_graph,
                initial_schedule,
                readiness_counters,
                root_barrier_edges,
            )
            self.assertIsNotNone(concrete_schedule)
            assert concrete_schedule is not None

            concrete_relations = tuple(
                (
                    segment.root,
                    segment.task_order.substitute_parameters(
                        {task_count: concrete_count}
                    ),
                )
                for segment, _phase, _count in symbolic_geometry
            )
            wave_count = 3 * ((concrete_count + 3) // 4)
            for wave in range(wave_count):
                for worker in range(4):
                    symbolic_tasks: list[tuple[int, int]] = []
                    for root, relation in concrete_relations:
                        launch_axis, worker_axis, wave_axis = (
                            relation.source_domain.axis_order
                        )
                        symbolic_tasks.extend(
                            (root, task)
                            for (task,) in relation.target_coordinates(
                                {
                                    launch_axis: 1,
                                    worker_axis: worker,
                                    wave_axis: wave,
                                }
                            )
                        )
                    self.assertLessEqual(len(symbolic_tasks), 1)
                    self.assertEqual(
                        symbolic_tasks[0] if symbolic_tasks else None,
                        task_at(concrete_schedule, worker, wave),
                    )

    def test_parametric_counter_declines_unproved_dynamic_layout(self) -> None:
        task_count = sympy.Symbol("task_count", integer=True, nonnegative=True)
        dependency_graph = _dependency_graph(
            [[10], [20]],
            _access(
                root=0,
                kind="store",
                shape=(8192,),
                block_ids=(10,),
                layout_is_symbolically_exact=False,
            ),
            _access(
                root=1,
                kind="load",
                shape=(8192,),
                block_ids=(20,),
                layout_is_symbolically_exact=False,
            ),
        )
        root_domains = (
            CoordinateDomain((10,), ((10, task_count),), ((10, 16),)),
            CoordinateDomain((20,), ((20, task_count),), ((20, 16),)),
        )

        with _forbid_schedule_enumeration():
            plan = _configured_static_pipeline_plan(
                dependency_graph=dependency_graph,
                root_domains=root_domains,
                axis_geometry={
                    10: (task_count, 16),
                    20: (task_count, 16),
                },
                worker_count=4,
            )

        self.assertEqual(plan.readiness_counters, ())
        self.assertEqual(plan.root_barrier_edges, frozenset(((0, 1),)))
        self.assertIsNone(
            cross_loop_scheduler._parametric_event_frontier_schedule_geometry(
                plan.worker_schedule
            )
        )
        self.assertIsNotNone(
            cross_loop_scheduler._parametric_root_major_schedule_geometry(
                plan.worker_schedule
            )
        )

    def test_parametric_fixed_fan_in_uses_final_arrival_continuation(self) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        dependency_graph = _dependency_graph(
            [[10], [20]],
            _access(
                root=0,
                kind="store",
                shape=(8192,),
                block_ids=(10,),
            ),
            _access(
                root=1,
                kind="load",
                shape=(8192,),
                block_ids=(20,),
            ),
        )
        root_domains = (
            CoordinateDomain((10,), ((10, 2 * key_count),), ((10, 16),)),
            CoordinateDomain((20,), ((20, key_count),), ((20, 32),)),
        )

        with _forbid_schedule_enumeration():
            plan = _configured_static_pipeline_plan(
                dependency_graph=dependency_graph,
                root_domains=root_domains,
                axis_geometry={
                    10: (2 * key_count, 16),
                    20: (key_count, 32),
                },
                worker_count=4,
            )

        self.assertEqual(plan.root_barrier_edges, frozenset())
        self.assertEqual(len(plan.readiness_counters), 1)
        (counter,) = plan.readiness_counters
        self.assertEqual(counter.readiness_key_count_expr, key_count)
        self.assertEqual(counter.uniform_arrival_count(), 2)
        self.assertEqual(counter.continuation_consumer_index, 0)
        self.assertTrue(
            cross_loop_scheduler._supports_current_parameterized_renderer(
                counter,
                tuple(
                    task_order.target_domain for task_order in plan.root_task_orders
                ),
            )
        )
        self.assertFalse(
            cross_loop_scheduler._supports_parameterized_fan_in_one_counter(counter)
        )
        self.assertEqual(
            tuple(segment.root for segment in plan.worker_schedule.segments),
            (0,),
        )
        schedule_geometry = (
            cross_loop_scheduler._parametric_root_major_schedule_geometry(
                plan.worker_schedule
            )
        )
        self.assertIsNotNone(schedule_geometry)

        publication = counter.producers[0].keys_by_producer
        self.assertIsNotNone(publication)
        assert publication is not None
        for concrete_count in (0, 1, 3, 5):
            concrete_publication = publication.substitute_parameters(
                {key_count: concrete_count}
            )
            self.assertEqual(
                concrete_publication.materialize(),
                tuple(
                    frozenset((producer_index // 2,))
                    for producer_index in range(2 * concrete_count)
                ),
            )

    def test_parametric_counter_allows_multiple_consumers_per_key(self) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        consumers_per_key = 16
        producers_per_key = 4
        producer_domain = CoordinateDomain(
            (10,),
            ((10, producers_per_key * key_count),),
            kind="site",
            identity=0,
        )
        consumer_domain = CoordinateDomain(
            (20,),
            ((20, consumers_per_key * key_count),),
            kind="site",
            identity=1,
        )
        readiness_key_domain = CoordinateDomain(
            (0,),
            ((0, key_count),),
            kind="event",
            identity=0,
        )
        key = coordinate_axis_symbol(0)
        consumer = coordinate_axis_symbol(20)
        counter = ReadinessCounterPlan(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation(
                        source_domain=readiness_key_domain,
                        target_domain=producer_domain,
                        pieces=(
                            _CoordinateRelationPiece(
                                source_bounds_items=((0, 0, key_count, 1),),
                                target_ranges=(
                                    (
                                        10,
                                        producers_per_key * key,
                                        producers_per_key * (key + 1),
                                        1,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=CoordinateRelation.point_map(
                        consumer_domain,
                        readiness_key_domain,
                        (
                            (
                                ((20, 0, consumers_per_key * key_count, 1),),
                                (sympy.floor(consumer / consumers_per_key),),
                            ),
                        ),
                    ),
                ),
            ),
        )

        with _forbid_schedule_enumeration():
            self.assertEqual(counter.uniform_arrival_count(), producers_per_key)
            self.assertIsNone(counter.continuation_consumer_index)
            self.assertTrue(
                cross_loop_scheduler._supports_current_parameterized_renderer(
                    counter,
                    (producer_domain, consumer_domain),
                )
            )
            self.assertFalse(
                cross_loop_scheduler._supports_parameterized_fan_in_one_counter(counter)
            )

        one_producer_domain = CoordinateDomain(
            (10,),
            ((10, key_count),),
            kind="site",
            identity=0,
        )
        one_producer_counter = dataclasses.replace(
            counter,
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation.point_map(
                        readiness_key_domain,
                        one_producer_domain,
                        (
                            (
                                ((0, 0, key_count, 1),),
                                (key,),
                            ),
                        ),
                    ),
                ),
            ),
        )
        with _forbid_schedule_enumeration():
            self.assertEqual(one_producer_counter.uniform_arrival_count(), 1)
            self.assertTrue(
                cross_loop_scheduler._supports_current_parameterized_renderer(
                    one_producer_counter,
                    (one_producer_domain, consumer_domain),
                )
            )
            self.assertFalse(
                cross_loop_scheduler._supports_parameterized_fan_in_one_counter(
                    one_producer_counter
                )
            )

    def test_parametric_counter_allows_partial_consumer_domain(self) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        producer_domain = CoordinateDomain(
            (10,),
            ((10, key_count),),
            kind="site",
            identity=0,
        )
        consumer_domain = CoordinateDomain(
            (20,),
            ((20, 2 * key_count),),
            kind="site",
            identity=1,
        )
        readiness_key_domain = CoordinateDomain(
            (0,),
            ((0, key_count),),
            kind="event",
            identity=0,
        )
        key = coordinate_axis_symbol(0)
        consumer = coordinate_axis_symbol(20)
        counter = ReadinessCounterPlan(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation.point_map(
                        readiness_key_domain,
                        producer_domain,
                        ((((0, 0, key_count, 1),), (key,)),),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=CoordinateRelation.point_map(
                        consumer_domain,
                        readiness_key_domain,
                        ((((20, 0, key_count, 1),), (consumer,)),),
                    ),
                ),
            ),
        )

        with _forbid_schedule_enumeration():
            self.assertFalse(counter.consumers[0].keys_by_consumer.is_total_function())
            self.assertTrue(
                cross_loop_scheduler._supports_current_parameterized_renderer(
                    counter,
                    (producer_domain, consumer_domain),
                )
            )
            self.assertFalse(
                cross_loop_scheduler._supports_parameterized_fan_in_one_counter(counter)
            )

    def test_parametric_counter_allows_multi_axis_keys(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        key_domain = CoordinateDomain(
            (0, 1),
            ((0, batch), (1, 2)),
            kind="event",
            identity=0,
        )
        producer_domain = CoordinateDomain(
            (10, 11, 12),
            ((10, batch), (11, 4), (12, 3)),
            kind="site",
            identity=0,
        )
        consumer_domain = CoordinateDomain(
            (20, 21, 22),
            ((20, batch), (21, 6), (22, 5)),
            kind="site",
            identity=1,
        )
        key_batch = coordinate_axis_symbol(0)
        key_group = coordinate_axis_symbol(1)
        consumer_batch = coordinate_axis_symbol(20)
        consumer_group = coordinate_axis_symbol(21)
        counter = ReadinessCounterPlan(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation(
                        source_domain=key_domain,
                        target_domain=producer_domain,
                        pieces=(
                            _CoordinateRelationPiece(
                                source_bounds_items=(
                                    (0, 0, batch, 1),
                                    (1, 0, 2, 1),
                                ),
                                target_ranges=(
                                    (10, key_batch, key_batch + 1, 1),
                                    (11, 2 * key_group, 2 * key_group + 2, 1),
                                    (12, sympy.Integer(0), sympy.Integer(3), 1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=CoordinateRelation.point_map(
                        consumer_domain,
                        key_domain,
                        (
                            (
                                (
                                    (20, 0, batch, 1),
                                    (21, 0, 6, 1),
                                    (22, 0, 5, 1),
                                ),
                                (
                                    consumer_batch,
                                    sympy.floor(consumer_group / 3),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

        with _forbid_schedule_enumeration():
            self.assertEqual(counter.uniform_arrival_count(), 6)
            self.assertTrue(
                cross_loop_scheduler._supports_current_parameterized_renderer(
                    counter,
                    (producer_domain, consumer_domain),
                )
            )
            self.assertFalse(
                cross_loop_scheduler._supports_parameterized_fan_in_one_counter(counter)
            )

    def test_parametric_counter_allows_bounded_nonuniform_fan_in(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        key_domain = CoordinateDomain(
            (0, 1),
            ((0, batch), (1, 2)),
            kind="event",
            identity=0,
        )
        producer_domain = CoordinateDomain(
            (10, 11),
            ((10, batch), (11, 3)),
            kind="site",
            identity=0,
        )
        consumer_domain = CoordinateDomain(
            (20, 21),
            ((20, batch), (21, 2)),
            kind="site",
            identity=1,
        )
        key_batch = coordinate_axis_symbol(0)
        producer_relation = CoordinateRelation(
            source_domain=key_domain,
            target_domain=producer_domain,
            pieces=(
                _CoordinateRelationPiece(
                    ((0, 0, batch, 1), (1, 0, 1, 1)),
                    (
                        (10, key_batch, key_batch + 1, 1),
                        (11, sympy.Integer(0), sympy.Integer(2), 1),
                    ),
                ),
                _CoordinateRelationPiece(
                    ((0, 0, batch, 1), (1, 1, 2, 1)),
                    (
                        (10, key_batch, key_batch + 1, 1),
                        (11, sympy.Integer(2), sympy.Integer(3), 1),
                    ),
                ),
            ),
        )
        counter = ReadinessCounterPlan(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=producer_relation,
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=CoordinateRelation.point_map(
                        consumer_domain,
                        key_domain,
                        (
                            (
                                ((20, 0, batch, 1), (21, 0, 2, 1)),
                                (
                                    coordinate_axis_symbol(20),
                                    coordinate_axis_symbol(21),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

        with _forbid_schedule_enumeration():
            arrival_count = counter.producers[0].arrival_count_by_key
            self.assertIsNotNone(arrival_count)
            assert arrival_count is not None
            self.assertTrue(arrival_count.is_total_function())
            self.assertIs(arrival_count.canonical_single_valued(), arrival_count)
            self.assertIsNone(counter.uniform_arrival_count())
            self.assertEqual(counter.arrival_count_bounds(), (1, 2))
            self.assertTrue(
                cross_loop_scheduler._supports_current_parameterized_renderer(
                    counter,
                    (producer_domain, consumer_domain),
                )
            )
            self.assertFalse(
                cross_loop_scheduler._supports_parameterized_fan_in_one_counter(counter)
            )

    def test_static_root_symbolic_nested_counter_falls_back_before_codegen(
        self,
    ) -> None:
        dependency_graph = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", shape=(1,), block_ids=(10,)),
            _access(root=1, kind="load", shape=(1,), block_ids=(20,)),
        )
        dependency = dependency_graph.edges[0].access_dependencies[0]
        (obligation,) = dependency_graph.dependency_obligations(dependency)
        producer_root = _domain((10, 1, 1))
        consumer_root = _domain((20, 1, 1))
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        key_domain = CoordinateDomain(
            (0,),
            ((0, key_count),),
            kind="event",
            identity=0,
            _allow_empty=True,
        )
        producer_site = CoordinateDomain(
            (10, 11),
            ((10, 1), (11, key_count)),
            kind="site",
            identity=100,
            _allow_empty=True,
        )
        counter = ReadinessCounterPlan(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation.point_map(
                        key_domain,
                        producer_site,
                        (
                            (
                                ((0, 0, key_count, 1),),
                                (
                                    sympy.Integer(0),
                                    coordinate_axis_symbol(0),
                                ),
                            ),
                        ),
                    ),
                    producer_site_id=100,
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=_full_point_map(
                        consumer_root,
                        key_domain,
                        sympy.Integer(0),
                    ),
                    covered_obligations=frozenset((obligation,)),
                ),
            ),
        )

        with _forbid_schedule_enumeration():
            self.assertTrue(counter.parameter_symbols)
            self.assertEqual(counter.arrival_count_bounds(), (1, 1))
            self.assertTrue(
                cross_loop_scheduler._supports_exact_counter_plan_lowering(
                    counter,
                    (producer_root, consumer_root),
                )
            )
            self.assertFalse(
                cross_loop_scheduler._supports_current_parameterized_renderer(
                    counter,
                    (producer_root, consumer_root),
                )
            )
            counters, barriers = cross_loop_scheduler._finalize_emitted_synchronization(
                dependency_graph=dependency_graph,
                root_domains=(producer_root, consumer_root),
                readiness_counters=(counter,),
            )

        self.assertEqual(counters, ())
        self.assertEqual(barriers, frozenset(((0, 1),)))
        readiness_graph = _readiness_graph(
            (producer_root, consumer_root),
            ReadinessEvent(counter.producers, counter.consumers),
        )
        baseline = _baseline_worker_schedule(
            readiness_graph.root_domains,
            worker_count=2,
        )
        plan_schedule = _build_baseline_worker_schedule(
            readiness_graph.root_domains,
            readiness_graph.root_task_orders,
            worker_count=2,
        )
        with _forbid_schedule_enumeration():
            with self.assertRaisesRegex(ValueError, "current renderer"):
                cross_loop_scheduler.StaticPipelinePlan(
                    worker_schedule=plan_schedule,
                    root_task_orders=readiness_graph.root_task_orders,
                    readiness_counters=(counter,),
                    root_barrier_edges=frozenset(),
                )
            self.assertTrue(
                cross_loop_scheduler._schedule_is_progress_safe(
                    baseline,
                    readiness_graph,
                    counters,
                    barriers,
                )
            )

    def test_parametric_fixed_fan_in_non_sink_uses_ordinary_counter(self) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        dependency_graph = _dependency_graph(
            [[10], [20], [30]],
            _access(
                root=0,
                allocation_id=0,
                kind="store",
                shape=(8192,),
                block_ids=(10,),
            ),
            _access(
                root=1,
                allocation_id=0,
                kind="load",
                shape=(8192,),
                block_ids=(20,),
            ),
            _access(
                root=1,
                allocation_id=1,
                kind="store",
                shape=(8192,),
                block_ids=(20,),
            ),
            _access(
                root=2,
                allocation_id=1,
                kind="load",
                shape=(8192,),
                block_ids=(30,),
            ),
        )
        root_domains = (
            CoordinateDomain((10,), ((10, 2 * key_count),), ((10, 16),)),
            CoordinateDomain((20,), ((20, key_count),), ((20, 32),)),
            CoordinateDomain((30,), ((30, key_count),), ((30, 32),)),
        )

        with _forbid_schedule_enumeration():
            plan = _configured_static_pipeline_plan(
                dependency_graph=dependency_graph,
                root_domains=root_domains,
                axis_geometry={
                    10: (2 * key_count, 16),
                    20: (key_count, 32),
                    30: (key_count, 32),
                },
                worker_count=4,
            )

        self.assertEqual(
            tuple(segment.root for segment in plan.worker_schedule.segments),
            (0, 1, 2),
        )
        self.assertFalse(
            any(
                counter.continuation_consumer_index is not None
                for counter in plan.readiness_counters
            )
        )
        self.assertEqual(plan.root_barrier_edges, frozenset())
        self.assertEqual(
            tuple(
                counter.uniform_arrival_count() for counter in plan.readiness_counters
            ),
            (2, 1),
        )
        self.assertTrue(
            all(
                cross_loop_scheduler._supports_current_parameterized_renderer(
                    counter,
                    tuple(
                        task_order.target_domain
                        for task_order in plan.root_task_orders
                    ),
                )
                for counter in plan.readiness_counters
            )
        )

    def test_pipeline_derives_continuation_candidates_once(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)

        for task_count in (4, batch):
            dependency_graph = _dependency_graph(
                [[10], [20]],
                _access(
                    root=0,
                    kind="store",
                    shape=(32 * task_count,),
                    block_ids=(10,),
                ),
                _access(
                    root=1,
                    kind="load",
                    shape=(32 * task_count,),
                    block_ids=(20,),
                ),
            )
            with (
                self.subTest(task_count=task_count),
                mock.patch.object(
                    cross_loop_scheduler,
                    "derive_final_arrival_continuations",
                    wraps=derive_final_arrival_continuations,
                ) as derive_candidates,
            ):
                plan = _configured_static_pipeline_plan(
                    dependency_graph=dependency_graph,
                    root_domains=(
                        _domain((10, 2 * task_count, 16)),
                        _domain((20, task_count, 32)),
                    ),
                    axis_geometry={
                        10: (2 * task_count, 16),
                        20: (task_count, 32),
                    },
                    worker_count=4,
                )

            self.assertEqual(derive_candidates.call_count, 1)
            (counter,) = plan.readiness_counters
            self.assertEqual(counter.continuation_consumer_index, 0)
            self.assertEqual(counter.uniform_arrival_count(), 2)

    def test_parametric_counter_declines_unsupported_access_scale(self) -> None:
        task_count = sympy.Symbol("task_count", integer=True, nonnegative=True)
        dependency_graph = _dependency_graph(
            [[10], [20]],
            _access(
                root=0,
                kind="store",
                shape=(8192,),
                block_ids=(10,),
                scales=(2,),
            ),
            _access(
                root=1,
                kind="load",
                shape=(8192,),
                block_ids=(20,),
            ),
        )
        root_domains = (
            CoordinateDomain((10,), ((10, task_count),), ((10, 16),)),
            CoordinateDomain((20,), ((20, task_count),), ((20, 16),)),
        )

        with _forbid_schedule_enumeration():
            plan = _configured_static_pipeline_plan(
                dependency_graph=dependency_graph,
                root_domains=root_domains,
                axis_geometry={
                    10: (task_count, 16),
                    20: (task_count, 16),
                },
                worker_count=4,
            )

        self.assertEqual(plan.readiness_counters, ())
        self.assertEqual(plan.root_barrier_edges, frozenset(((0, 1),)))

    def test_worker_schedule_task_coverage_rejects_duplicate_and_missing(self) -> None:
        (domain,) = _identify_root_domains((_domain((10, 2, 1)),))
        task_order = pid_task_order(domain, domain.axis_order)
        first = _task_order_slice(task_order, 0, 1)
        assert first is not None
        with (
            _forbid_schedule_enumeration(),
            self.assertRaisesRegex(
                ValueError,
                "worker schedule does not own each logical task once",
            ),
        ):
            _schedule(
                1,
                _segment(0, first, workers=(0, 1), dispatch_offset=0),
                _segment(0, first, workers=(0, 1), dispatch_offset=1),
            )

    def test_worker_schedule_accepts_symbolic_permuted_traversal(self) -> None:
        (domain,) = _identify_root_domains((_domain((10, 2, 1), (11, 3, 1)),))
        reference = pid_task_order(domain, (11, 10))
        permuted = pid_task_order(domain, (10, 11))
        schedule = _schedule(
            2,
            _segment(0, permuted, workers=(0, 2), dispatch_offset=0),
        )

        with _forbid_schedule_enumeration():
            self.assertTrue(_validate_worker_schedule_tasks(schedule, (reference,)))
            self.assertFalse(
                cross_loop_scheduler.root_schedule_matches_reference(
                    schedule.segments,
                    reference,
                )
            )

    def test_transient_ticket_order_is_a_symbolic_bijection(self) -> None:
        (domain,) = _identify_root_domains((_domain((10, 5, 1)),))
        reference = pid_task_order(domain, domain.axis_order)
        readiness_graph = ReadinessGraph(
            root_task_orders=(reference,),
            events=(),
        )
        schedule = cross_loop_scheduler._with_transient_source_schedule_segment(
            _baseline_worker_schedule(readiness_graph.root_domains, worker_count=2),
            readiness_graph.root_task_orders,
            0,
        )
        self.assertIsNotNone(schedule)
        assert schedule is not None

        with _forbid_schedule_enumeration():
            logical_to_ticket = cross_loop_scheduler._source_ticket_order(
                schedule,
                0,
            )
        self.assertIsNotNone(logical_to_ticket)
        assert logical_to_ticket is not None
        self.assertEqual(
            tuple(next(iter(ticket)) for ticket in logical_to_ticket.materialize()),
            (0, 1, 2, 3, 4),
        )

    def test_transient_ticket_order_preserves_l2_pid_mapping(self) -> None:
        (domain,) = _identify_root_domains((_domain((10, 4, 1), (11, 3, 1)),))
        reference = pid_task_order(
            domain,
            domain.axis_order,
            l2_group_size=2,
        )
        readiness_graph = ReadinessGraph(
            root_task_orders=(reference,),
            events=(),
        )
        schedule = cross_loop_scheduler._with_transient_source_schedule_segment(
            _baseline_worker_schedule(readiness_graph.root_domains, worker_count=2),
            readiness_graph.root_task_orders,
            0,
        )
        self.assertIsNotNone(schedule)
        assert schedule is not None

        with _forbid_schedule_enumeration():
            logical_to_ticket = cross_loop_scheduler._source_ticket_order(
                schedule,
                0,
            )
        self.assertIsNotNone(logical_to_ticket)
        assert logical_to_ticket is not None
        logical_task_by_ticket = tuple(
            next(iter(tasks)) for tasks in reference.materialize()
        )
        ticket_by_logical_task = tuple(
            next(iter(tickets)) for tickets in logical_to_ticket.materialize()
        )
        self.assertEqual(
            tuple(
                ticket_by_logical_task[logical_task]
                for logical_task in logical_task_by_ticket
            ),
            tuple(range(domain.size)),
        )

    def test_transient_ticket_order_keeps_uncoalesced_woven_proof(self) -> None:
        (domain,) = _identify_root_domains((_domain((10, 2, 1), (11, 256, 1)),))
        reference = pid_task_order(domain, domain.axis_order)
        schedule = cross_loop_scheduler._with_transient_source_schedule_segment(
            _baseline_worker_schedule((domain,), worker_count=148),
            (reference,),
            0,
        )
        self.assertIsNotNone(schedule)
        assert schedule is not None

        with _forbid_schedule_enumeration():
            source_segment = cross_loop_scheduler._transient_source_schedule_segment(
                schedule,
                0,
            )
            logical_to_ticket = cross_loop_scheduler._source_ticket_order(
                schedule,
                0,
            )

        self.assertIsNotNone(source_segment)
        self.assertIsNotNone(logical_to_ticket)
        assert source_segment is not None
        assert logical_to_ticket is not None
        self.assertEqual(source_segment.task_count, 512)
        self.assertTrue(logical_to_ticket.is_total_function())
        self.assertTrue(
            cross_loop_scheduler.root_schedule_matches_reference(
                (source_segment,),
                reference,
            )
        )

    def test_transient_ticket_order_preserves_existing_segment_permutation(
        self,
    ) -> None:
        (domain,) = _identify_root_domains((_domain((10, 2, 1), (11, 3, 1)),))
        reference = pid_task_order(domain, (10, 11))
        permuted = pid_task_order(domain, (11, 10))
        schedule = cross_loop_scheduler._with_transient_source_schedule_segment(
            _baseline_worker_schedule((domain,), worker_count=2),
            (permuted,),
            0,
        )
        self.assertIsNotNone(schedule)
        assert schedule is not None
        source_relation = schedule.segments_for_root(0)[0].task_order

        retained = cross_loop_scheduler._with_transient_source_schedule_segment(
            schedule,
            (reference,),
            0,
        )

        self.assertIs(retained, schedule)
        assert retained is not None
        self.assertEqual(retained.segments_for_root(0)[0].task_order, source_relation)
        with _forbid_schedule_enumeration():
            self.assertTrue(_validate_worker_schedule_tasks(retained, (reference,)))
            logical_to_ticket = cross_loop_scheduler._source_ticket_order(
                retained,
                0,
            )
        self.assertIsNotNone(logical_to_ticket)
        assert logical_to_ticket is not None
        logical_task_by_ticket = tuple(
            next(iter(tasks)) for tasks in permuted.materialize()
        )
        ticket_by_logical_task = tuple(
            next(iter(tickets)) for tickets in logical_to_ticket.materialize()
        )
        self.assertEqual(
            tuple(
                ticket_by_logical_task[logical_task]
                for logical_task in logical_task_by_ticket
            ),
            tuple(range(domain.size)),
        )

    def test_progress_proves_disjoint_partial_producer_arms_independently(
        self,
    ) -> None:
        first_domain, second_domain, consumer_domain = _identify_root_domains(
            (
                _domain((10, 1, 1)),
                _domain((20, 1, 1)),
                _domain((30, 2, 1)),
            )
        )
        key_domain = _domain((0, 2), kind="event", identity=0)
        event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation(
                        key_domain,
                        first_domain,
                        (
                            _CoordinateRelationPiece(
                                ((0, 0, 1, 1),),
                                ((10, sympy.Integer(0), sympy.Integer(1), 1),),
                            ),
                        ),
                    ),
                ),
                ReadinessProducer(
                    producer_root=1,
                    producers_by_key=CoordinateRelation(
                        key_domain,
                        second_domain,
                        (
                            _CoordinateRelationPiece(
                                ((0, 1, 2, 1),),
                                ((20, sympy.Integer(0), sympy.Integer(1), 1),),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=2,
                    keys_by_consumer=_full_point_map(
                        consumer_domain,
                        key_domain,
                        coordinate_axis_symbol(30),
                    ),
                ),
            ),
        )
        readiness_graph = _readiness_graph(
            (first_domain, second_domain, consumer_domain),
            event,
        )
        plan = ReadinessCounterPlan(event.producers, event.consumers)
        root_orders = readiness_graph.root_task_orders
        safe = _schedule(
            2,
            _segment(0, root_orders[0], workers=(0, 1), dispatch_offset=0),
            _segment(1, root_orders[1], workers=(1, 1), dispatch_offset=0),
            _segment(2, root_orders[2], workers=(0, 2), dispatch_offset=2),
        )
        late_second_arm = _schedule(
            2,
            _segment(0, root_orders[0], workers=(0, 1), dispatch_offset=0),
            _segment(2, root_orders[2], workers=(0, 2), dispatch_offset=2),
            _segment(1, root_orders[1], workers=(1, 1), dispatch_offset=2),
        )
        transient_safe = _schedule(
            2,
            _segment(1, root_orders[1], workers=(0, 1), dispatch_offset=0),
            _segment(2, root_orders[2], workers=(0, 2), dispatch_offset=2),
        )
        transient_late_resident_arm = _schedule(
            2,
            _segment(2, root_orders[2], workers=(0, 2), dispatch_offset=0),
            _segment(1, root_orders[1], workers=(1, 1), dispatch_offset=1),
        )
        transient_safe = cross_loop_scheduler._with_transient_source_schedule_segment(
            transient_safe,
            root_orders,
            0,
        )
        transient_late_resident_arm = (
            cross_loop_scheduler._with_transient_source_schedule_segment(
                transient_late_resident_arm,
                root_orders,
                0,
            )
        )
        self.assertIsNotNone(transient_safe)
        self.assertIsNotNone(transient_late_resident_arm)
        assert transient_safe is not None
        assert transient_late_resident_arm is not None

        with _forbid_schedule_enumeration():
            self.assertTrue(
                cross_loop_scheduler._schedule_is_progress_safe(
                    safe,
                    readiness_graph,
                    (plan,),
                    frozenset(),
                )
            )
            self.assertFalse(
                cross_loop_scheduler._schedule_is_progress_safe(
                    late_second_arm,
                    readiness_graph,
                    (plan,),
                    frozenset(),
                )
            )
            self.assertTrue(
                cross_loop_scheduler._schedule_is_progress_safe(
                    transient_safe,
                    readiness_graph,
                    (plan,),
                    frozenset(),
                    transient_source_root=0,
                )
            )
            self.assertFalse(
                cross_loop_scheduler._schedule_is_progress_safe(
                    transient_late_resident_arm,
                    readiness_graph,
                    (plan,),
                    frozenset(),
                    transient_source_root=0,
                )
            )

    def test_progress_rejects_cycle_across_disjoint_worker_waits(self) -> None:
        domains = _identify_root_domains(
            tuple(_domain((10 + root, 1, 1)) for root in range(4))
        )

        def event(producer_root: int, consumer_root: int, event_id: int):
            key_domain = _domain((0, 1), kind="event", identity=event_id)
            zero = sympy.Integer(0)
            return ReadinessEvent(
                producers=(
                    ReadinessProducer(
                        producer_root,
                        _full_point_map(key_domain, domains[producer_root], zero),
                    ),
                ),
                consumers=(
                    ReadinessConsumer(
                        consumer_root,
                        _full_point_map(domains[consumer_root], key_domain, zero),
                    ),
                ),
            )

        first = event(0, 1, 0)
        second = event(2, 3, 1)
        readiness_graph = _readiness_graph(domains, first, second)
        # Worker 0 executes B -> C; worker 1 executes D -> A.  The two
        # individually disjoint waits close A -> B -> C -> D -> A.
        schedule = _schedule(
            2,
            _segment(
                1,
                readiness_graph.root_task_orders[1],
                workers=(0, 1),
                dispatch_offset=0,
            ),
            _segment(
                3,
                readiness_graph.root_task_orders[3],
                workers=(1, 1),
                dispatch_offset=0,
            ),
            _segment(
                2,
                readiness_graph.root_task_orders[2],
                workers=(0, 1),
                dispatch_offset=1,
            ),
            _segment(
                0,
                readiness_graph.root_task_orders[0],
                workers=(1, 1),
                dispatch_offset=1,
            ),
        )
        plans = (
            ReadinessCounterPlan(first.producers, first.consumers),
            ReadinessCounterPlan(second.producers, second.consumers),
        )

        with _forbid_schedule_enumeration():
            self.assertFalse(
                cross_loop_scheduler._schedule_is_progress_safe(
                    schedule,
                    readiness_graph,
                    plans,
                    frozenset(),
                )
            )

    def test_progress_uses_exact_segment_support_for_partial_producer(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (_domain((10, 2, 1)), _domain((20, 1, 1)))
        )
        key_domain = _domain((0, 1), kind="event", identity=0)
        zero = sympy.Integer(0)

        def graph_and_plan(producer_task: int):
            event = ReadinessEvent(
                producers=(
                    ReadinessProducer(
                        0,
                        _full_point_map(
                            key_domain,
                            producer_domain,
                            sympy.Integer(producer_task),
                        ),
                    ),
                ),
                consumers=(
                    ReadinessConsumer(
                        1,
                        _full_point_map(consumer_domain, key_domain, zero),
                    ),
                ),
            )
            graph = _readiness_graph((producer_domain, consumer_domain), event)
            return graph, ReadinessCounterPlan(event.producers, event.consumers)

        safe_graph, safe_plan = graph_and_plan(0)
        safe = _schedule(
            2,
            _segment(
                0,
                _one_dimensional_task_range(producer_domain, 0, 1),
                workers=(0, 1),
                dispatch_offset=0,
            ),
            _segment(
                1,
                safe_graph.root_task_orders[1],
                workers=(1, 1),
                dispatch_offset=0,
            ),
            _segment(
                0,
                _one_dimensional_task_range(producer_domain, 1, 1),
                workers=(1, 1),
                dispatch_offset=1,
            ),
        )
        unsafe_graph, unsafe_plan = graph_and_plan(1)

        with _forbid_schedule_enumeration():
            self.assertTrue(
                cross_loop_scheduler._schedule_is_progress_safe(
                    safe,
                    safe_graph,
                    (safe_plan,),
                    frozenset(),
                )
            )
            self.assertFalse(
                cross_loop_scheduler._schedule_is_progress_safe(
                    safe,
                    unsafe_graph,
                    (unsafe_plan,),
                    frozenset(),
                )
            )
            with mock.patch.object(
                cross_loop_scheduler,
                "_MAX_GLOBAL_LIST_EDGES",
                2,
            ):
                self.assertFalse(
                    cross_loop_scheduler._schedule_is_progress_safe(
                        safe,
                        safe_graph,
                        (safe_plan,),
                        frozenset(),
                    )
                )

    def test_nested_counter_projects_unused_owning_task_axis(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (_domain((10, 8, 1)), _domain((20, 2, 1), (22, 2, 1)))
        )
        consumer_site_domain = _domain(
            (20, 2, 1),
            (22, 2, 1),
            (21, 4, 1),
            identity=7,
        )
        readiness_key_domain = _domain((0, 8), kind="event", identity=0)
        readiness_consumer = ReadinessConsumer(
            consumer_root=1,
            consumer_site_id=7,
            keys_by_consumer=_full_point_map(
                consumer_site_domain,
                readiness_key_domain,
                coordinate_axis_symbol(21) + 4 * coordinate_axis_symbol(22),
            ),
            covered_obligations=frozenset(((0, None, 7),)),
        )
        event = ReadinessEvent(
            producers=(
                _readiness_producer_from_publication(
                    producer_root=0,
                    publication=_full_point_map(
                        producer_domain,
                        readiness_key_domain,
                        coordinate_axis_symbol(10),
                    ),
                ),
            ),
            consumers=(readiness_consumer,),
        )
        readiness_graph = _readiness_graph(
            (producer_domain, consumer_domain),
            event,
        )

        reduced_domain = CoordinateDomain(
            axis_order=(22, 21),
            axis_counts_items=((22, 2), (21, 4)),
            block_sizes_items=((22, 1), (21, 1)),
            kind="site",
            identity=7,
        )
        direct_converse = readiness_consumer.keys_by_consumer.converse()
        self.assertTrue(
            direct_converse is None
            or direct_converse.project_target(reduced_domain) is None
        )
        projected = readiness_consumer.keys_by_consumer.project_source(reduced_domain)
        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertIsNotNone(projected.converse())
        plan = _segmented_nested_loop_counter(
            readiness_graph,
            event,
            readiness_consumer,
            (0, 4),
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(
            plan.producers[0].producers_by_key.materialize(),
            (frozenset((0, 1, 2, 3)), frozenset((4, 5, 6, 7))),
        )
        self.assertEqual(
            plan.consumers[0].keys_by_consumer.materialize(),
            (frozenset((0,)), frozenset((0,)), frozenset((1,)), frozenset((1,))) * 4,
        )
        self.assertEqual(
            plan.consumers[0].covered_obligations,
            readiness_consumer.covered_obligations,
        )

    def test_nested_counter_supports_symbolic_bq_and_boundaries(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        query = sympy.Symbol("query", integer=True, nonnegative=True)
        half = sympy.Symbol("half", integer=True, positive=True)
        graph, event, consumer = _symbolic_nested_counter_graph(
            batch,
            query,
            2 * half,
        )

        with (
            mock.patch.object(
                CoordinateDomain,
                "axis_counts",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "symbolic nested counters must not request concrete axis counts"
                ),
            ),
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError(
                    "symbolic nested counters must not enumerate relations"
                ),
            ),
        ):
            plan = _segmented_nested_loop_counter(
                graph,
                event,
                consumer,
                (0, half, 2 * half),
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.readiness_key_domain.shape_expr, (2, batch, query))
        self.assertEqual(plan.parameter_symbols, frozenset((batch, query, half)))

        for concrete_batch, concrete_query, concrete_half in (
            (0, 3, 2),
            (2, 0, 3),
            (2, 2, 1),
            (2, 3, 2),
            (3, 2, 3),
        ):
            substitutions = {
                batch: concrete_batch,
                query: concrete_query,
                half: concrete_half,
            }
            dynamic_producer = plan.producers[0].producers_by_key.substitute_parameters(
                substitutions
            )
            dynamic_consumer = plan.consumers[0].keys_by_consumer.substitute_parameters(
                substitutions
            )
            if not concrete_batch or not concrete_query:
                self.assertFalse(dynamic_producer.pieces)
                self.assertFalse(dynamic_consumer.pieces)
                self.assertEqual(dynamic_producer.source_domain.size, 0)
                self.assertEqual(dynamic_consumer.source_domain.size, 0)
                continue

            static_graph, static_event, static_consumer = (
                _symbolic_nested_counter_graph(
                    concrete_batch,
                    concrete_query,
                    2 * concrete_half,
                )
            )
            static_plan = _segmented_nested_loop_counter(
                static_graph,
                static_event,
                static_consumer,
                (0, concrete_half, 2 * concrete_half),
            )
            self.assertIsNotNone(static_plan)
            assert static_plan is not None
            self.assertEqual(
                dynamic_producer.materialize(),
                static_plan.producers[0].producers_by_key.materialize(),
            )
            self.assertEqual(
                dynamic_consumer.materialize(),
                static_plan.consumers[0].keys_by_consumer.materialize(),
            )

    def test_nested_counter_falls_back_per_producer_arm(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        query = sympy.Symbol("query", integer=True, nonnegative=True)
        first_producer, second_producer, consumer_root = _identify_root_domains(
            (
                CoordinateDomain(
                    (10, 11, 12),
                    ((10, batch), (11, query), (12, 8)),
                    ((10, 1), (11, 1), (12, 1)),
                    _allow_empty=True,
                ),
                CoordinateDomain(
                    (30, 31, 32),
                    ((30, batch), (31, query), (32, 8)),
                    ((30, 1), (31, 1), (32, 1)),
                    _allow_empty=True,
                ),
                CoordinateDomain(
                    (20, 21),
                    ((20, batch), (21, query)),
                    ((20, 1), (21, 1)),
                    _allow_empty=True,
                ),
            )
        )
        semantic_key_domain = CoordinateDomain(
            (0, 1, 2),
            ((0, batch), (1, query), (2, 8)),
            kind="event",
            identity=0,
            _allow_empty=True,
        )
        producers = tuple(
            _readiness_producer_from_publication(
                producer_root=root,
                publication=_full_point_map(
                    domain,
                    semantic_key_domain,
                    *(coordinate_axis_symbol(axis) for axis in domain.axis_order),
                ),
            )
            for root, domain in enumerate((first_producer, second_producer))
        )
        consumer_site = CoordinateDomain(
            (20, 21, 22),
            ((20, batch), (21, query), (22, 8)),
            ((20, 1), (21, 1), (22, 1)),
            identity=7,
            _allow_empty=True,
        )
        consumer = ReadinessConsumer(
            consumer_root=2,
            consumer_site_id=7,
            keys_by_consumer=_full_point_map(
                consumer_site,
                semantic_key_domain,
                coordinate_axis_symbol(20),
                coordinate_axis_symbol(21),
                coordinate_axis_symbol(22),
            ),
            covered_obligations=frozenset(((0, None, 7), (1, None, 7))),
        )
        event = ReadinessEvent(producers, (consumer,))
        graph = _readiness_graph(
            (first_producer, second_producer, consumer_root),
            event,
        )

        original_converse = CoordinateRelation.converse
        original_axis_counts = CoordinateDomain._concrete_axis_counts
        declined_publications = 0

        def decline_second_publication(
            relation: CoordinateRelation,
        ) -> CoordinateRelation | None:
            nonlocal declined_publications
            if (
                relation.source_domain == second_producer
                and relation.target_domain.kind == "event"
                and relation.target_domain.identity is None
            ):
                declined_publications += 1
                return None
            return original_converse(relation)

        def reject_runtime_axis_counts(domain: CoordinateDomain) -> dict[int, int]:
            if domain.parameter_symbols:
                raise AssertionError(
                    "symbolic nested counters must not concretize runtime axes"
                )
            return original_axis_counts(domain)

        with (
            mock.patch.object(
                CoordinateRelation,
                "converse",
                decline_second_publication,
            ),
            mock.patch.object(
                CoordinateDomain,
                "_concrete_axis_counts",
                reject_runtime_axis_counts,
            ),
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError(
                    "symbolic nested counters must not enumerate relations"
                ),
            ),
        ):
            plan = _segmented_nested_loop_counter(
                graph,
                event,
                consumer,
                (0, 4, 8),
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(declined_publications, 1)
        self.assertEqual(len(plan.producers), 2)
        self.assertTrue(
            cross_loop_scheduler._supports_exact_counter_plan_lowering(
                plan,
                graph.root_domains,
            )
        )

        for concrete_batch, concrete_query in ((0, 3), (2, 0), (2, 3)):
            substitutions = {batch: concrete_batch, query: concrete_query}
            concrete_relations = tuple(
                producer.producers_by_key.substitute_parameters(substitutions)
                for producer in plan.producers
            )
            concrete_consumer = plan.consumers[
                0
            ].keys_by_consumer.substitute_parameters(substitutions)
            if not concrete_batch or not concrete_query:
                self.assertTrue(
                    all(not relation.pieces for relation in concrete_relations)
                )
                self.assertFalse(concrete_consumer.pieces)
                continue

            for relation in concrete_relations:
                target_axes = relation.target_domain.axis_order
                for key_index, actual in enumerate(relation.materialize()):
                    key = relation.source_domain.coordinates(key_index)
                    expected = frozenset(
                        relation.target_domain.index(
                            {
                                target_axes[0]: key[1],
                                target_axes[1]: key[2],
                                target_axes[2]: nested_iteration,
                            }
                        )
                        for nested_iteration in range(4 * key[0], 4 * key[0] + 4)
                    )
                    self.assertEqual(actual, expected)

            for consumer_index, actual in enumerate(concrete_consumer.materialize()):
                coordinates = concrete_consumer.source_domain.coordinates(
                    consumer_index
                )
                expected_key = concrete_consumer.target_domain.index(
                    {
                        0: coordinates[22] // 4,
                        1: coordinates[20],
                        2: coordinates[21],
                    }
                )
                self.assertEqual(actual, frozenset((expected_key,)))

    def test_nested_loop_entry_counter_accepts_positive_symbolic_extent(
        self,
    ) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        query = sympy.Symbol("query", integer=True, nonnegative=True)
        nested_extent = sympy.Symbol("nested_extent", integer=True, positive=True)
        graph, event, consumer = _symbolic_nested_counter_graph(
            batch,
            query,
            nested_extent,
        )

        with mock.patch.object(
            CoordinateDomain,
            "axis_counts",
            new_callable=mock.PropertyMock,
            side_effect=AssertionError(
                "symbolic nested counters must not request concrete axis counts"
            ),
        ):
            plan = _nested_loop_entry_counter(graph, event, consumer)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.readiness_key_domain.shape_expr, (1, batch, query))
        zero_batch = {
            batch: 0,
            query: 3,
            nested_extent: 5,
        }
        self.assertFalse(
            plan.producers[0]
            .producers_by_key.substitute_parameters(zero_batch)
            .pieces
        )
        self.assertFalse(
            plan.consumers[0]
            .keys_by_consumer.substitute_parameters(zero_batch)
            .pieces
        )

    def test_nested_counter_preserves_static_nonuniform_segments(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        query = sympy.Symbol("query", integer=True, nonnegative=True)
        graph, event, consumer = _symbolic_nested_counter_graph(batch, query, 8)

        concrete_axis_counts = CoordinateDomain._concrete_axis_counts

        def reject_runtime_axis_counts(domain: CoordinateDomain) -> dict[int, int]:
            if domain.parameter_symbols:
                raise AssertionError(
                    "symbolic nested counters must not concretize runtime axes"
                )
            return concrete_axis_counts(domain)

        with mock.patch.object(
            CoordinateDomain,
            "_concrete_axis_counts",
            reject_runtime_axis_counts,
        ):
            plan = _segmented_nested_loop_counter(
                graph,
                event,
                consumer,
                (0, 1, 5, 8),
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.readiness_key_domain.shape_expr, (3, batch, query))
        self.assertTrue(
            cross_loop_scheduler._supports_exact_counter_plan_lowering(
                plan,
                graph.root_domains,
            )
        )
        concrete_producers = plan.producers[0].producers_by_key.substitute_parameters(
            {batch: 2, query: 2}
        )
        self.assertEqual(
            sorted(
                len(producers)
                for producers in concrete_producers.materialize()
            ),
            [1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4],
        )
        self.assertEqual(plan.arrival_count_bounds(), (1, 4))

    def test_nested_counter_declines_unproved_or_empty_inner_segments(self) -> None:
        maybe_empty = sympy.Symbol(
            "maybe_empty",
            integer=True,
            nonnegative=True,
        )
        graph, event, consumer = _symbolic_nested_counter_graph(
            2,
            3,
            maybe_empty + 2,
        )

        with mock.patch.object(
            CoordinateDomain,
            "axis_counts",
            new_callable=mock.PropertyMock,
            side_effect=AssertionError(
                "symbolic nested counters must not request concrete axis counts"
            ),
        ):
            self.assertIsNone(
                _segmented_nested_loop_counter(
                    graph,
                    event,
                    consumer,
                    (0, maybe_empty, maybe_empty + 2),
                )
            )

        empty_graph, empty_event, empty_consumer = _symbolic_nested_counter_graph(
            2,
            3,
            maybe_empty,
        )
        self.assertIsNone(
            _nested_loop_entry_counter(
                empty_graph,
                empty_event,
                empty_consumer,
            )
        )

    def test_nested_counter_declines_diagonal_owning_task_axis(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (_domain((10, 4, 1)), _domain((20, 2, 1)))
        )
        consumer_site_domain = _domain(
            (20, 2, 1),
            (21, 4, 1),
            identity=7,
        )
        readiness_key_domain = _domain((0, 4), kind="event", identity=0)
        readiness_consumer = ReadinessConsumer(
            consumer_root=1,
            consumer_site_id=7,
            keys_by_consumer=_full_point_map(
                consumer_site_domain,
                readiness_key_domain,
                sympy.Mod(
                    coordinate_axis_symbol(20) + coordinate_axis_symbol(21),
                    4,
                ),
            ),
            covered_obligations=frozenset(((0, None, 7),)),
        )
        event = ReadinessEvent(
            producers=(
                _readiness_producer_from_publication(
                    producer_root=0,
                    publication=_full_point_map(
                        producer_domain,
                        readiness_key_domain,
                        coordinate_axis_symbol(10),
                    ),
                ),
            ),
            consumers=(readiness_consumer,),
        )
        readiness_graph = _readiness_graph(
            (producer_domain, consumer_domain),
            event,
        )

        self.assertIsNone(readiness_consumer.keys_by_consumer.converse())
        self.assertIsNone(
            _segmented_nested_loop_counter(
                readiness_graph,
                event,
                readiness_consumer,
                (0, 2, 4),
            )
        )

    def test_partial_source_event_has_a_structural_ready_frontier(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (_domain((10, 8, 1)), _domain((20, 2, 1)))
        )
        key_domain = _domain((0, 2), kind="event", identity=0)
        key = coordinate_axis_symbol(0)
        event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation(
                        key_domain,
                        producer_domain,
                        (
                            _CoordinateRelationPiece(
                                ((0, 0, 2, 1),),
                                ((10, 2 * key, 2 * key + 2, 1),),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=_full_point_map(
                        consumer_domain,
                        key_domain,
                        coordinate_axis_symbol(20),
                    ),
                ),
            ),
        )
        readiness_graph = _readiness_graph(
            (producer_domain, consumer_domain),
            event,
        )
        baseline = _build_baseline_worker_schedule(
            readiness_graph.root_domains,
            readiness_graph.root_task_orders,
            worker_count=2,
        )

        result = _event_ready_after_worker_steps(
            readiness_graph,
            event,
            worker_schedule=baseline,
            continuation_by_root={},
        )

        self.assertIsNotNone(result)
        assert result is not None
        ready_after, prerequisite_roots = result
        self.assertEqual(ready_after.materialize(), (frozenset((0,)), frozenset((1,))))
        self.assertEqual(prerequisite_roots, frozenset((0,)))

    def test_strict_subset_one_key_event_keeps_exact_counter(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (_domain((10, 8, 1)), _domain((20, 1, 1)))
        )
        key_domain = _domain((0, 1), kind="event", identity=0)
        event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation(
                        key_domain,
                        producer_domain,
                        (
                            _CoordinateRelationPiece(
                                ((0, 0, 1, 1),),
                                ((10, sympy.Integer(2), sympy.Integer(6), 1),),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=_full_point_map(
                        consumer_domain,
                        key_domain,
                        sympy.Integer(0),
                    ),
                    covered_obligations=frozenset(((0, None, None),)),
                ),
            ),
        )
        readiness_graph = _readiness_graph(
            (producer_domain, consumer_domain),
            event,
        )

        selected = choose_readiness_counters(readiness_graph, ())

        self.assertIsNone(event.root_barrier_producer_root)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].uniform_arrival_count(), 4)

    def test_large_affine_schedule_never_materializes_task_orders(self) -> None:
        size = 319_488
        plan = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", shape=(size,), block_ids=(10,)),
            _access(root=1, kind="load", shape=(size,), block_ids=(20,)),
        )
        root_domains = _one_dimensional_domains(
            producer_count=size // 16,
            consumer_count=size // 512,
            producer_block=16,
            consumer_block=512,
        )

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("production scheduling expanded a relation"),
        ):
            schedule = _configured_static_pipeline_plan(
                dependency_graph=plan,
                root_domains=root_domains,
                axis_geometry={10: (size // 16, 16), 20: (size // 512, 512)},
                worker_count=148,
            )

        self.assertLessEqual(
            sum(
                len(readiness_producer.producers_by_key.pieces)
                for event in schedule.readiness_counters
                for readiness_producer in event.producers
            ),
            4,
        )

    def test_symbolic_readiness_event_keeps_relations_compact(self) -> None:
        plan = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", shape=(65,), block_ids=(10,)),
            _access(root=1, kind="load", shape=(65,), block_ids=(20,)),
        )

        events = _configured_readiness_events(
            plan,
            axis_geometry={10: (5, 16), 20: (3, 32)},
        )

        self.assertIsNotNone(events)
        assert events is not None
        (event,) = events
        self.assertEqual(event.readiness_key_count, 3)
        self.assertEqual(len(event.producers[0].producers_by_key.pieces), 1)
        self.assertEqual(len(event.consumers[0].keys_by_consumer.pieces), 1)
        self.assertEqual(
            event.consumers[0].keys_by_consumer.materialize(),
            (frozenset((0,)), frozenset((1,)), frozenset((2,))),
        )
        self.assertEqual(
            _publication(event.producers[0]).materialize(),
            (
                frozenset((0,)),
                frozenset((0,)),
                frozenset((1,)),
                frozenset((1,)),
                frozenset((2,)),
            ),
        )

    def test_symbolic_readiness_event_joins_multiple_producers(self) -> None:
        plan = _dependency_graph(
            [[10], [20], [30]],
            _access(root=0, kind="store", shape=(64,), block_ids=(10,)),
            _access(
                root=1, allocation_id=1, kind="store", shape=(64,), block_ids=(20,)
            ),
            _access(root=2, kind="load", shape=(64,), block_ids=(30,)),
            _access(root=2, allocation_id=1, kind="load", shape=(64,), block_ids=(30,)),
        )

        events = _configured_readiness_events(
            plan,
            axis_geometry={10: (4, 16), 20: (4, 16), 30: (2, 32)},
        )

        self.assertIsNotNone(events)
        assert events is not None
        (event,) = events
        self.assertEqual(event.readiness_key_count, 2)
        self.assertEqual(len(event.producers), 2)
        self.assertEqual(
            tuple(
                _publication(readiness_producer).materialize()
                for readiness_producer in event.producers
            ),
            (
                (
                    frozenset((0,)),
                    frozenset((0,)),
                    frozenset((1,)),
                    frozenset((1,)),
                ),
            )
            * 2,
        )
        self.assertEqual(len(event.consumers[0].covered_obligations), 2)

    def test_symbolic_readiness_event_drops_irrelevant_consumer_axis(self) -> None:
        plan = _dependency_graph(
            [[10, 11], [20, 21, 22]],
            _access(root=0, kind="store", shape=(2, 64), block_ids=(10, 11)),
            _access(root=1, kind="load", shape=(2, 64), block_ids=(20, 22)),
        )

        events = _configured_readiness_events(
            plan,
            axis_geometry={
                10: (2, 1),
                11: (4, 16),
                20: (2, 1),
                21: (4, 1),
                22: (4, 16),
            },
        )

        self.assertIsNotNone(events)
        assert events is not None
        (event,) = events
        self.assertEqual(event.readiness_key_domain.axis_order, (0, 1))
        self.assertEqual(event.readiness_key_domain.block_sizes_items, ())
        self.assertEqual(event.readiness_key_count, 8)
        for consumer_task in range(
            event.consumers[0].keys_by_consumer.source_domain.size
        ):
            consumer_coordinates = event.consumers[
                0
            ].keys_by_consumer.source_domain.coordinates(consumer_task)
            expected_key = consumer_coordinates[20] + 2 * consumer_coordinates[22]
            self.assertEqual(
                event.consumers[0].keys_by_consumer.targets(consumer_task),
                frozenset((expected_key,)),
            )

    def test_symbolic_readiness_event_coalesces_equivalent_fanout(self) -> None:
        plan = _dependency_graph(
            [[10, 11], [20, 21, 22], [30, 31, 32]],
            _access(root=0, kind="store", shape=(2, 64), block_ids=(10, 11)),
            _access(root=1, kind="load", shape=(2, 64), block_ids=(20, 22)),
            _access(root=2, kind="load", shape=(2, 64), block_ids=(30, 32)),
        )

        events = _configured_readiness_events(
            plan,
            axis_geometry={
                10: (2, 1),
                11: (4, 16),
                20: (2, 1),
                21: (3, 7),
                22: (4, 16),
                30: (2, 1),
                31: (5, 11),
                32: (4, 16),
            },
        )

        self.assertIsNotNone(events)
        assert events is not None
        (event,) = events
        self.assertEqual(event.readiness_key_domain.axis_order, (0, 1))
        self.assertEqual(event.readiness_key_count, 8)
        self.assertEqual(
            {
                readiness_consumer.consumer_root
                for readiness_consumer in event.consumers
            },
            {1, 2},
        )

    def test_symbolic_readiness_event_does_not_coalesce_swapped_axes(self) -> None:
        plan = _dependency_graph(
            [[10, 11], [20, 21], [30, 31]],
            _access(root=0, kind="store", shape=(2, 2), block_ids=(10, 11)),
            _access(root=1, kind="load", shape=(2, 2), block_ids=(20, 21)),
            _access(root=2, kind="load", shape=(2, 2), block_ids=(31, 30)),
        )

        events = _configured_readiness_events(
            plan,
            axis_geometry=dict.fromkeys((10, 11, 20, 21, 30, 31), (2, 1)),
        )

        self.assertIsNotNone(events)
        assert events is not None
        self.assertEqual(len(events), 2)
        self.assertEqual(
            [
                {
                    readiness_consumer.consumer_root
                    for readiness_consumer in event.consumers
                }
                for event in events
            ],
            [{1}, {2}],
        )
        self.assertNotEqual(
            events[0].producers[0].producers_by_key,
            events[1].producers[0].producers_by_key,
        )

    def test_symbolic_readiness_event_uses_one_chart_for_multi_producer_join(
        self,
    ) -> None:
        plan = _dependency_graph(
            [[10, 11], [20, 21], [30, 31], [40, 41]],
            _access(root=0, kind="store", shape=(2, 2), block_ids=(10, 11)),
            _access(
                root=1, allocation_id=1, kind="store", shape=(2, 2), block_ids=(20, 21)
            ),
            _access(root=2, kind="load", shape=(2, 2), block_ids=(30, 31)),
            _access(
                root=2, allocation_id=1, kind="load", shape=(2, 2), block_ids=(30, 31)
            ),
            _access(root=3, kind="load", shape=(2, 2), block_ids=(40, 41)),
            _access(
                root=3, allocation_id=1, kind="load", shape=(2, 2), block_ids=(41, 40)
            ),
        )

        events = _configured_readiness_events(
            plan,
            axis_geometry=dict.fromkeys((10, 11, 20, 21, 30, 31, 40, 41), (2, 1)),
        )

        self.assertIsNotNone(events)
        assert events is not None
        self.assertEqual(len(events), 2)
        self.assertTrue(all(len(event.producers) == 2 for event in events))
        self.assertEqual(
            [
                {
                    readiness_consumer.consumer_root
                    for readiness_consumer in event.consumers
                }
                for event in events
            ],
            [{2}, {3}],
        )

    def test_symbolic_readiness_event_unions_disjoint_producer_ranges(self) -> None:
        plan = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", shape=(64,), block_ids=(10,)),
            _access(root=1, kind="load", shape=(64,), block_ids=(20,)),
            _access(root=1, kind="load", shape=(64,), block_ids=(20,), offsets=(32,)),
        )

        events = _configured_readiness_events(
            plan,
            axis_geometry={10: (32, 2), 20: (2, 16)},
        )

        self.assertIsNotNone(events)
        assert events is not None
        (event,) = events
        self.assertEqual(event.readiness_key_count, 2)
        self.assertEqual(len(event.producers), 1)
        self.assertEqual(len(event.producers[0].producers_by_key.pieces), 2)
        expected = tuple(frozenset((producer // 8 % 2,)) for producer in range(32))
        self.assertEqual(_publication(event.producers[0]).materialize(), expected)

    def test_unsupported_symbolic_event_coarsens_to_family_done(self) -> None:
        plan = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", shape=(64,), block_ids=(10,)),
            _access(root=1, kind="load", shape=(64,), block_ids=(20,), masked=True),
        )

        events = _configured_readiness_events(
            plan,
            axis_geometry={10: (4, 16), 20: (2, 32)},
        )

        self.assertIsNotNone(events)
        assert events is not None
        (event,) = events
        self.assertEqual(event.readiness_key_count, 1)
        self.assertEqual(event.producers[0].producer_root, 0)
        self.assertEqual(
            _publication(event.producers[0]).materialize(),
            (frozenset((0,)),) * 4,
        )
        self.assertEqual(
            event.consumers[0].keys_by_consumer.materialize(),
            (frozenset((0,)),) * 2,
        )

    def test_unsupported_event_quotient_does_not_coarsen_unrelated_edges(
        self,
    ) -> None:
        plan = _dependency_graph(
            [[10], [20], [30], [40]],
            _access(root=0, kind="store", shape=(64,), block_ids=(10,)),
            _access(root=1, kind="load", shape=(64,), block_ids=(20,)),
            _access(root=1, kind="load", shape=(64,), block_ids=(20,)),
            _access(
                root=2, allocation_id=1, kind="store", shape=(64,), block_ids=(30,)
            ),
            _access(root=3, allocation_id=1, kind="load", shape=(64,), block_ids=(40,)),
        )

        original_union = CoordinateRelation.union
        failed_once = False

        def fail_first_union(
            left: CoordinateRelation,
            right: CoordinateRelation,
        ) -> CoordinateRelation | None:
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                return None
            return original_union(left, right)

        with mock.patch.object(CoordinateRelation, "union", fail_first_union):
            events = _configured_readiness_events(
                plan,
                axis_geometry={
                    10: (4, 16),
                    20: (2, 32),
                    30: (4, 16),
                    40: (2, 32),
                },
            )

        unrelated = [
            event
            for event in events
            if any(
                readiness_producer.producer_root == 2
                for readiness_producer in event.producers
            )
        ]
        self.assertEqual(len(unrelated), 1)
        self.assertEqual(unrelated[0].readiness_key_count, 2)
        self.assertIsNone(unrelated[0].root_barrier_producer_root)

    @skipIfNotCUDA()
    @skipIfNotTriton("cross-loop scheduling is currently Triton-only")
    @skipIfRefEager("compiled DeviceIR is unavailable in ref eager mode")
    def test_device_ir_sites_preserve_nested_producer_and_consumer_axes(
        self,
    ) -> None:
        x = torch.empty((2, 64), device=DEVICE, dtype=torch.float32)

        producer_ir = nested_store_chain.bind((x,)).host_function.device_ir
        assert producer_ir.tile_dependency_graph is not None
        producer_graph = producer_ir.tile_dependency_graph
        producer_store = next(
            access
            for access in producer_graph.accesses
            if access.root == 0 and access.kind == "store"
        )
        (producer_site,) = producer_graph.sites_for_access(producer_store.access_id)
        self.assertEqual(producer_site.kind, "loop")
        self.assertEqual(len(producer_site.callsite_path), 1)
        self.assertEqual(
            producer_site.logical_axis_order,
            (
                *producer_ir.task_families[0].logical_axis_order,
                *producer_site.local_axis_order,
            ),
        )
        self.assertTrue(producer_site.executes_unconditionally)
        self.assertTrue(producer_site.can_split_loop)

        producer_outer_axis = producer_ir.task_families[0].logical_axis_order[0]
        consumer_batch_axis, consumer_width_axis = producer_ir.task_families[
            1
        ].logical_axis_order
        producer_domains = (
            _domain((producer_outer_axis, 2, 1)),
            _domain((consumer_batch_axis, 2, 1), (consumer_width_axis, 4, 16)),
        )
        producer_axis_geometry = {
            producer_outer_axis: (2, 1),
            producer_site.local_axis_order[0]: (4, 16),
            consumer_batch_axis: (2, 1),
            consumer_width_axis: (4, 16),
        }
        producer_events = _configured_readiness_graph(
            producer_graph,
            root_domains=producer_domains,
            axis_geometry=producer_axis_geometry,
        )
        producer_event = next(
            event
            for event in producer_events.events
            if any(
                readiness_producer.producer_site_id == producer_site.site_id
                for readiness_producer in event.producers
            )
        )
        self.assertEqual(producer_event.readiness_key_count, 8)
        self.assertEqual(
            _expected_arrivals(
                producer_event.readiness_key_domain, producer_event.producers
            ),
            (1,) * 8,
        )
        self.assertEqual(producer_event.consumers[0].consumer_site_id, None)

        synchronous_events = _configured_readiness_graph(
            producer_graph,
            root_domains=producer_domains,
            axis_geometry=producer_axis_geometry,
            publishable_site_ids=frozenset(),
        )
        self.assertFalse(
            any(
                readiness_producer.producer_site_id is not None
                for event in synchronous_events.events
                for readiness_producer in event.producers
            )
        )

        consumer_ir = streamed_singleton_reduction.bind((x,)).host_function.device_ir
        assert consumer_ir.tile_dependency_graph is not None
        consumer_graph = consumer_ir.tile_dependency_graph
        consumer_load = next(
            access
            for access in consumer_graph.accesses
            if access.root == 1
            and access.kind == "load"
            and any(
                site.kind == "loop"
                for site in consumer_graph.sites_for_access(access.access_id)
            )
        )
        (consumer_site,) = consumer_graph.sites_for_access(consumer_load.access_id)
        self.assertEqual(consumer_site.kind, "loop")
        self.assertEqual(len(consumer_site.callsite_path), 1)
        self.assertEqual(
            consumer_site.logical_axis_order,
            (
                *consumer_ir.task_families[1].logical_axis_order,
                *consumer_site.local_axis_order,
            ),
        )
        self.assertTrue(consumer_site.executes_unconditionally)
        self.assertTrue(consumer_site.can_split_loop)

        producer_batch_axis, producer_width_axis = consumer_ir.task_families[
            0
        ].logical_axis_order
        consumer_outer_axis = consumer_ir.task_families[1].logical_axis_order[0]
        consumer_domains = (
            _domain((producer_batch_axis, 2, 1), (producer_width_axis, 4, 16)),
            _domain((consumer_outer_axis, 2, 1)),
        )
        consumer_axis_geometry = {
            producer_batch_axis: (2, 1),
            producer_width_axis: (4, 16),
            consumer_outer_axis: (2, 1),
            consumer_site.local_axis_order[0]: (4, 16),
        }
        consumer_events = _configured_readiness_graph(
            consumer_graph,
            root_domains=consumer_domains,
            axis_geometry=consumer_axis_geometry,
        )
        nested_event = next(
            event
            for event in consumer_events.events
            if any(
                readiness_consumer.consumer_site_id == consumer_site.site_id
                for readiness_consumer in event.consumers
            )
        )
        self.assertEqual(nested_event.readiness_key_count, 8)
        self.assertEqual(
            _expected_arrivals(
                nested_event.readiness_key_domain, nested_event.producers
            ),
            (1,) * 8,
        )
        (nested_keys_by_consumer,) = nested_event.consumers
        nested_keys = nested_keys_by_consumer.keys_by_consumer.materialize(
            source_axis_order=readiness_consumer_source_order(
                consumer_events, nested_keys_by_consumer
            )
        )
        self.assertTrue(all(len(keys) == 1 for keys in nested_keys))
        self.assertEqual(
            {next(iter(keys)) for keys in nested_keys},
            set(range(8)),
        )

    def test_semantic_readiness_graph_composes_arbitrary_chain_depth(self) -> None:
        graph = _dependency_graph(
            [[10], [20], [30], [40]],
            _access(root=0, kind="store", block_ids=(10,)),
            _access(root=1, kind="load", block_ids=(20,)),
            _access(root=1, allocation_id=1, kind="store", block_ids=(20,)),
            _access(root=2, allocation_id=1, kind="load", block_ids=(30,)),
            _access(root=2, allocation_id=2, kind="store", block_ids=(30,)),
            _access(root=3, allocation_id=2, kind="load", block_ids=(40,)),
        )

        self.assertEqual(len(graph.task_families), 4)
        configured = _configured_readiness_graph(
            graph,
            tuple(_domain((block_id, 4, 1)) for block_id in (10, 20, 30, 40)),
        )
        self.assertEqual(len(configured.events), 3)
        self.assertEqual(
            tuple(
                _expected_arrivals(event.readiness_key_domain, event.producers)
                for event in configured.events
            ),
            ((1, 1, 1, 1),) * 3,
        )
        baseline = _build_baseline_worker_schedule(
            configured.root_domains,
            configured.root_task_orders,
            worker_count=4,
        )
        continuations = derive_final_arrival_continuations(configured)
        self.assertEqual(
            tuple(
                configured.event(continuation.event_id)
                .consumers[continuation.consumer_index]
                .consumer_root
                for continuation in continuations
            ),
            (1, 2, 3),
        )
        validate_worker_schedule(
            configured,
            baseline.without_roots(frozenset((1, 2, 3))),
            continuations,
        )

    def test_final_arrival_continuations_allow_disjoint_uses_of_one_producer_family(
        self,
    ) -> None:
        domains = tuple(
            _domain((axis, count), identity=root)
            for root, (axis, count) in enumerate(((10, 4), (20, 2), (30, 2)))
        )
        key_domains = tuple(
            _domain((0, 2), kind="event", identity=event) for event in range(2)
        )

        def keys(
            domain: CoordinateDomain,
            readiness_key_domain: CoordinateDomain,
            begin: int,
            end: int,
        ) -> CoordinateRelation:
            (axis,) = domain.axis_order
            return CoordinateRelation.point_map(
                domain,
                readiness_key_domain,
                (
                    (
                        ((axis, begin, end, 1),),
                        (coordinate_axis_symbol(axis) - begin,),
                    ),
                ),
            )

        readiness_graph = ReadinessGraph(
            root_task_orders=tuple(
                pid_task_order(domain, domain.axis_order) for domain in domains
            ),
            events=(
                ReadinessEvent(
                    producers=(
                        _readiness_producer_from_publication(
                            0,
                            keys(domains[0], key_domains[0], 0, 2),
                        ),
                    ),
                    consumers=(
                        ReadinessConsumer(1, keys(domains[1], key_domains[0], 0, 2)),
                    ),
                ),
                ReadinessEvent(
                    producers=(
                        _readiness_producer_from_publication(
                            0,
                            keys(domains[0], key_domains[1], 2, 4),
                        ),
                        _readiness_producer_from_publication(
                            1,
                            keys(domains[1], key_domains[1], 0, 2),
                        ),
                    ),
                    consumers=(
                        ReadinessConsumer(2, keys(domains[2], key_domains[1], 0, 2)),
                    ),
                ),
            ),
        )
        continuations = derive_final_arrival_continuations(readiness_graph)

        self.assertEqual(
            tuple(
                readiness_graph.event(continuation.event_id)
                .consumers[continuation.consumer_index]
                .consumer_root
                for continuation in continuations
            ),
            (1, 2),
        )

        overlapping = dataclasses.replace(
            readiness_graph,
            events=(
                readiness_graph.events[0],
                dataclasses.replace(
                    readiness_graph.events[1],
                    producers=(
                        _readiness_producer_from_publication(
                            0,
                            keys(domains[0], key_domains[1], 1, 3),
                        ),
                        readiness_graph.events[1].producers[1],
                    ),
                ),
            ),
        )
        self.assertEqual(derive_final_arrival_continuations(overlapping), ())

    def test_final_arrival_continuation_requires_counter_lowerability(
        self,
    ) -> None:
        producer_domain = _domain((10, 4), identity=0)
        consumer_domain = _domain((20, 2), identity=1)
        readiness_key_domain = _domain((0, 2), kind="event", identity=0)
        readiness_producer = ReadinessProducer(
            producer_root=0,
            producers_by_key=CoordinateRelation(
                readiness_key_domain,
                producer_domain,
                (
                    _CoordinateRelationPiece(
                        ((0, 0, 2, 1),),
                        (
                            (
                                10,
                                2 * coordinate_axis_symbol(0),
                                2 * coordinate_axis_symbol(0) + 1,
                                1,
                            ),
                        ),
                    ),
                ),
            ),
        )
        readiness_graph = ReadinessGraph(
            root_task_orders=(
                pid_task_order(producer_domain, (10,)),
                pid_task_order(consumer_domain, (20,)),
            ),
            events=(
                ReadinessEvent(
                    (readiness_producer,),
                    (
                        ReadinessConsumer(
                            1,
                            _full_point_map(
                                consumer_domain,
                                readiness_key_domain,
                                coordinate_axis_symbol(20),
                            ),
                        ),
                    ),
                ),
            ),
        )
        publication = readiness_producer.keys_by_producer
        self.assertIsNotNone(publication)
        assert publication is not None
        self.assertIsNone(publication.canonical_single_valued())
        self.assertEqual(
            derive_final_arrival_continuations(readiness_graph), ()
        )
        self.assertEqual(choose_readiness_counters(readiness_graph, ()), ())

    def test_final_arrival_candidates_use_lowered_event_relations(self) -> None:
        producer_domain = _domain((10, 2), identity=0)
        readiness_key_domain = _domain((0, 4), kind="event", identity=0)
        readiness_key = coordinate_axis_symbol(0)
        producer = ReadinessProducer(
            producer_root=0,
            producers_by_key=_full_point_map(
                readiness_key_domain,
                producer_domain,
                sympy.floor(readiness_key / 2),
            ),
        )
        obligation = (0, None, None)

        self.assertFalse(
            cross_loop_scheduler._supports_readiness_counter_lowering(producer)
        )
        for consumer_count, expected_candidate in ((2, True), (4, False)):
            with self.subTest(consumer_count=consumer_count):
                consumer_domain = _domain((20, consumer_count), identity=1)
                consumer = ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=(
                        CoordinateRelation(
                            consumer_domain,
                            readiness_key_domain,
                            (
                                _CoordinateRelationPiece(
                                    ((20, 0, 2, 1),),
                                    (
                                        (
                                            0,
                                            2 * coordinate_axis_symbol(20),
                                            2 * coordinate_axis_symbol(20) + 1,
                                            1,
                                        ),
                                    ),
                                ),
                                _CoordinateRelationPiece(
                                    ((20, 0, 2, 1),),
                                    (
                                        (
                                            0,
                                            2 * coordinate_axis_symbol(20) + 1,
                                            2 * coordinate_axis_symbol(20) + 2,
                                            1,
                                        ),
                                    ),
                                ),
                            ),
                        )
                        if consumer_count == 2
                        else _full_point_map(
                            consumer_domain,
                            readiness_key_domain,
                            coordinate_axis_symbol(20),
                        )
                    ),
                    covered_obligations=frozenset((obligation,)),
                )
                semantic_converse = consumer.keys_by_consumer.converse()
                semantic_is_bijection = (
                    consumer.keys_by_consumer.is_total_function()
                    and semantic_converse is not None
                    and semantic_converse.is_total_function()
                )
                self.assertNotEqual(semantic_is_bijection, expected_candidate)

                event = ReadinessEvent((producer,), (consumer,))
                lowering_relations = cross_loop_scheduler._counter_lowering_relations(
                    event
                )
                self.assertIsNotNone(lowering_relations)
                assert lowering_relations is not None
                lowered_producers, lowered_consumers = lowering_relations
                (lowered_consumer,) = lowered_consumers
                lowered_converse = lowered_consumer.keys_by_consumer.converse()
                self.assertEqual(
                    lowered_consumer.keys_by_consumer.is_total_function()
                    and lowered_converse is not None
                    and lowered_converse.is_total_function(),
                    expected_candidate,
                )
                candidate_plan = ReadinessCounterPlan(
                    producers=lowered_producers,
                    consumers=lowered_consumers,
                    continuation_consumer_index=0,
                )
                readiness_graph = _readiness_graph(
                    (producer_domain, consumer_domain),
                    event,
                )

                self.assertEqual(
                    cross_loop_scheduler._supports_exact_counter_plan_lowering(
                        candidate_plan,
                        readiness_graph.root_domains,
                    ),
                    expected_candidate,
                )
                candidates = derive_final_arrival_continuations(readiness_graph)
                expected_candidates = (
                    (FinalArrivalContinuation(event_id=0, consumer_index=0),)
                    if expected_candidate
                    else ()
                )
                self.assertEqual(candidates, expected_candidates)
                if expected_candidate:
                    selected = choose_readiness_counters(
                        readiness_graph,
                        candidates,
                    )
                    self.assertEqual(
                        cross_loop_scheduler._emitted_final_arrival_continuations(
                            readiness_graph,
                            selected,
                        ),
                        candidates,
                    )

    def test_final_arrival_root_event_must_cover_nested_obligations(self) -> None:
        producer_domain = _domain((10, 4), identity=0)
        consumer_domain = _domain((20, 2), identity=1)
        nested_domain = _domain((20, 2), (21, 1), identity=2)
        readiness_key_domain = _domain((0, 2), kind="event", identity=0)
        nested_key_domain = dataclasses.replace(readiness_key_domain, identity=1)

        def producer(key_domain: CoordinateDomain) -> ReadinessProducer:
            return _readiness_producer_from_publication(
                producer_root=0,
                producer_site_id=None,
                publication=_full_point_map(
                    producer_domain,
                    key_domain,
                    sympy.floor(coordinate_axis_symbol(10) / 2),
                ),
            )

        root_obligation = (0, None, None)
        nested_obligation = (1, None, 2)

        def graph(root_coverage: frozenset[tuple[int, int | None, int | None]]):
            return _readiness_graph(
                (producer_domain, consumer_domain),
                ReadinessEvent(
                    producers=(producer(readiness_key_domain),),
                    consumers=(
                        ReadinessConsumer(
                            consumer_root=1,
                            consumer_site_id=None,
                            keys_by_consumer=_full_point_map(
                                consumer_domain,
                                readiness_key_domain,
                                coordinate_axis_symbol(20),
                            ),
                            covered_obligations=root_coverage,
                        ),
                    ),
                ),
                ReadinessEvent(
                    producers=(producer(nested_key_domain),),
                    consumers=(
                        ReadinessConsumer(
                            consumer_root=1,
                            consumer_site_id=2,
                            keys_by_consumer=_full_point_map(
                                nested_domain,
                                nested_key_domain,
                                coordinate_axis_symbol(20),
                            ),
                            covered_obligations=frozenset((nested_obligation,)),
                        ),
                    ),
                ),
            )

        incomplete = graph(frozenset((root_obligation,)))
        self.assertEqual(derive_final_arrival_continuations(incomplete), ())

        complete = graph(frozenset((root_obligation, nested_obligation)))
        continuations = derive_final_arrival_continuations(complete)
        self.assertEqual(
            continuations,
            (FinalArrivalContinuation(event_id=0, consumer_index=0),),
        )
        (counter,) = choose_readiness_counters(complete, continuations)
        self.assertEqual(counter.continuation_consumer_index, 0)
        self.assertEqual(len(counter.consumers), 1)
        self.assertIsNone(counter.consumers[0].consumer_site_id)

    def test_root_projection_unions_all_nested_sites_or_keeps_barrier(self) -> None:
        def dependency_graph(
            *, second_outer_stride: int = 16, second_inner_stride: int = 4
        ):
            graph = _dependency_graph(
                [[10], [20]],
                _access(root=0, kind="store", shape=(64,), block_ids=(10,)),
                _access(
                    root=1,
                    kind="load",
                    shape=(64,),
                    block_ids=(None,),
                    offsets=(None,),
                    affine_subscript_ranges=((((20, 16, 1), (21, 4, 1)), 0, 4, 1),),
                ),
                _access(
                    root=1,
                    kind="load",
                    shape=(64,),
                    block_ids=(None,),
                    offsets=(None,),
                    affine_subscript_ranges=(
                        (
                            (
                                (20, second_outer_stride, 1),
                                (22, second_inner_stride, 1),
                            ),
                            0,
                            4,
                            1,
                        ),
                    ),
                ),
            )
            return dataclasses.replace(
                graph,
                execution_sites=(
                    ExecutionSite(0, 0, 0, (), None, "root", (10,), (10,), True, False),
                    ExecutionSite(1, 1, 1, (), None, "root", (20,), (20,), True, False),
                    ExecutionSite(
                        2,
                        1,
                        2,
                        ((0, 0),),
                        1,
                        "loop",
                        (21,),
                        (20, 21),
                        True,
                        False,
                    ),
                    ExecutionSite(
                        3,
                        1,
                        3,
                        ((1, 0),),
                        1,
                        "loop",
                        (22,),
                        (20, 22),
                        True,
                        False,
                    ),
                ),
                site_ids_by_access=((0,), (2,), (3,)),
            )

        root_domains = (_domain((10, 64, 1)), _domain((20, 3, 1)))
        axis_geometry = {10: (64, 1), 20: (3, 1), 21: (4, 1), 22: (4, 1)}
        complete_graph = dependency_graph()
        complete = _configured_readiness_graph(
            complete_graph,
            root_domains,
            axis_geometry=axis_geometry,
        )
        self.assertEqual(len(complete.events), 1)
        root_event = next(
            event
            for event in complete.events
            if any(consumer.consumer_site_id is None for consumer in event.consumers)
        )
        root_consumer = next(
            consumer
            for consumer in root_event.consumers
            if consumer.consumer_site_id is None
        )
        self.assertEqual(len(root_event.consumers), 1)
        all_obligations = frozenset(
            obligation
            for edge in complete_graph.edges
            for dependency in edge.access_dependencies
            for obligation in complete_graph.dependency_obligations(dependency)
        )
        self.assertEqual(root_consumer.covered_obligations, all_obligations)
        self.assertEqual(
            _expected_arrivals(
                root_event.readiness_key_domain,
                root_event.producers,
            ),
            (16, 16, 16),
        )
        self.assertEqual(
            len(derive_final_arrival_continuations(complete)),
            1,
        )

        # Width four with a stride-five nested axis is valid at each nested
        # site, but its root projection is gapped and cannot join the event.
        incomplete_graph = dependency_graph(
            second_outer_stride=19,
            second_inner_stride=5,
        )
        incomplete = _configured_readiness_graph(
            incomplete_graph,
            root_domains,
            axis_geometry=axis_geometry,
        )
        incomplete_root = next(
            event
            for event in incomplete.events
            if event.root_barrier_producer_root is None
        )
        fallback = next(
            event
            for event in incomplete.events
            if event.root_barrier_producer_root is not None
        )
        incomplete_root_consumer = incomplete_root.consumers[0]
        self.assertEqual(len(incomplete_root_consumer.covered_obligations), 1)
        self.assertEqual(fallback.root_barrier_producer_root, 0)
        self.assertEqual(len(fallback.consumers), 1)
        self.assertEqual(
            fallback.consumers[0].covered_obligations,
            all_obligations - incomplete_root_consumer.covered_obligations,
        )

    def test_large_final_arrival_continuation_ignores_downstream_event_granularity(
        self,
    ) -> None:
        graph = _dependency_graph(
            [[10], [20], [30]],
            _access(root=0, kind="store", block_ids=(10,)),
            _access(root=1, kind="load", block_ids=(20,)),
            _access(
                root=1,
                allocation_id=1,
                kind="store",
                block_ids=(20,),
                layout_is_symbolically_exact=False,
            ),
            _access(
                root=2,
                allocation_id=1,
                kind="load",
                block_ids=(30,),
                layout_is_symbolically_exact=False,
            ),
        )
        root_domains = _identify_root_domains(
            (
                _domain((10, 8, 1)),
                _domain((20, 8, 1)),
                _domain((30, 1, 1)),
            )
        )
        readiness_graph = _configured_readiness_graph(graph, root_domains)
        baseline = _build_baseline_worker_schedule(
            root_domains,
            readiness_graph.root_task_orders,
            worker_count=4,
        )

        continuations = choose_final_arrival_continuations(
            readiness_graph,
            derive_final_arrival_continuations(readiness_graph),
            baseline,
        )

        self.assertGreater(root_domains[1].size, baseline.worker_count)
        self.assertEqual(readiness_graph.events[1].root_barrier_producer_root, 1)
        self.assertEqual(len(continuations), 1)
        readiness_consumer = readiness_graph.event(continuations[0].event_id).consumers[
            continuations[0].consumer_index
        ]
        self.assertEqual(readiness_consumer.consumer_root, 1)

    def test_semantic_readiness_graph_represents_diamond_without_path_matching(
        self,
    ) -> None:
        graph = _dependency_graph(
            [[10], [20], [30], [40]],
            _access(root=0, kind="store", block_ids=(10,)),
            _access(root=0, allocation_id=1, kind="store", block_ids=(10,)),
            _access(root=1, kind="load", block_ids=(20,)),
            _access(root=1, allocation_id=2, kind="store", block_ids=(20,)),
            _access(root=2, allocation_id=1, kind="load", block_ids=(30,)),
            _access(root=2, allocation_id=3, kind="store", block_ids=(30,)),
            _access(root=3, allocation_id=2, kind="load", block_ids=(40,)),
            _access(root=3, allocation_id=3, kind="load", block_ids=(40,)),
        )

        configured = _configured_readiness_graph(
            graph,
            tuple(_domain((block_id, 4, 1)) for block_id in (10, 20, 30, 40)),
        )
        (root_zero_event,) = tuple(
            event
            for event in configured.events
            if any(
                readiness_producer.producer_root == 0
                for readiness_producer in event.producers
            )
        )
        self.assertEqual(
            {
                readiness_consumer.consumer_root
                for readiness_consumer in root_zero_event.consumers
            },
            {1, 2},
        )
        continuations = derive_final_arrival_continuations(configured)
        self.assertEqual(
            {
                configured.event(continuation.event_id)
                .consumers[continuation.consumer_index]
                .consumer_root
                for continuation in continuations
            },
            {3},
        )

    def test_exact_and_family_done_dependencies_can_share_one_consumer(
        self,
    ) -> None:
        graph = _dependency_graph(
            [[10], [20], [30]],
            _access(
                root=0,
                kind="store",
                block_ids=(10,),
                layout_is_symbolically_exact=False,
            ),
            _access(root=1, allocation_id=1, kind="store", block_ids=(20,)),
            _access(
                root=2,
                kind="load",
                block_ids=(30,),
                layout_is_symbolically_exact=False,
            ),
            _access(root=2, allocation_id=1, kind="load", block_ids=(30,)),
        )

        configured = _configured_readiness_graph(
            graph,
            tuple(_domain((block_id, 4, 1)) for block_id in (10, 20, 30)),
        )
        configured_uses = tuple(
            readiness_consumer
            for event in configured.events
            for readiness_consumer in event.consumers
            if readiness_consumer.consumer_root == 2
        )
        self.assertEqual(len(configured_uses), 2)
        family_event = next(
            event
            for event in configured.events
            if event.root_barrier_producer_root is not None
        )
        self.assertEqual(family_event.root_barrier_producer_root, 0)
        self.assertEqual(
            _expected_arrivals(
                family_event.readiness_key_domain, family_event.producers
            ),
            (4,),
        )
        self.assertEqual(derive_final_arrival_continuations(configured), ())

    def test_dependency_coverage_distinguishes_producer_callsites(self) -> None:
        graph = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", block_ids=(10,)),
            _access(root=1, kind="load", block_ids=(20,)),
        )
        sites = (
            ExecutionSite(0, 0, 0, (), None, "root", (), (10,), True, False),
            ExecutionSite(
                1,
                0,
                0,
                ((0, 0),),
                None,
                "root",
                (),
                (10,),
                False,
                False,
            ),
            ExecutionSite(2, 1, 1, (), None, "root", (), (20,), True, False),
        )
        graph = dataclasses.replace(
            graph,
            execution_sites=sites,
            site_ids_by_access=((0, 1), (2,)),
        )
        access_dependency = graph.edges[0].access_dependencies[0]
        self.assertEqual(
            graph.dependency_obligations(access_dependency),
            frozenset(
                (
                    (access_dependency.dependency_id, 0, 2),
                    (access_dependency.dependency_id, 1, 2),
                )
            ),
        )
        axis_geometry = {10: (4, 32), 20: (4, 32)}
        configured_root_domains, configured_site_domains = (
            instantiate_coordinate_domains(
                graph,
                axis_geometry=axis_geometry,
            )
        )
        exact_dependencies = instantiate_symbolic_dependencies(
            graph,
            root_domains=configured_root_domains,
            site_domains=configured_site_domains,
        )
        self.assertEqual(len(exact_dependencies), 1)

        events = _configured_readiness_events(graph, axis_geometry=axis_geometry)

        self.assertIsNotNone(events)
        assert events is not None
        self.assertEqual(len(events), 2)
        exact_event = next(
            event for event in events if event.root_barrier_producer_root is None
        )
        family_event = next(
            event for event in events if event.root_barrier_producer_root is not None
        )
        dependency_id = access_dependency.dependency_id
        self.assertEqual(
            exact_event.consumers[0].covered_obligations,
            frozenset(((dependency_id, 0, 2),)),
        )
        self.assertEqual(
            family_event.consumers[0].covered_obligations,
            frozenset(((dependency_id, 1, 2),)),
        )

        root_domains = tuple(
            domain for domain in configured_root_domains if domain is not None
        )
        readiness_graph = ReadinessGraph(
            root_task_orders=tuple(
                pid_task_order(domain, domain.axis_order) for domain in root_domains
            ),
            events=events,
        )
        self.assertEqual(
            derive_final_arrival_continuations(readiness_graph), ()
        )
        readiness_counters = choose_readiness_counters(readiness_graph, ())
        covered_obligations = frozenset(
            obligation
            for counter_plan in readiness_counters
            for readiness_consumer in counter_plan.consumers
            for obligation in readiness_consumer.covered_obligations
        )
        self.assertEqual(
            _select_root_barrier_edges(
                dependency_graph=graph,
                covered_obligations=covered_obligations,
            ),
            frozenset(((0, 1),)),
        )

    def test_baseline_worker_schedule_preserves_source_order(self) -> None:
        root_domains = (
            _domain((10, 3, 1)),
            _domain((20, 5, 1)),
        )

        schedule = _baseline_worker_schedule(root_domains, worker_count=4)

        self.assertEqual(placement(schedule, 0, 0), (0, 0))
        self.assertEqual(placement(schedule, 0, 2), (2, 0))
        self.assertEqual(placement(schedule, 1, 0), (0, 1))
        self.assertEqual(placement(schedule, 1, 4), (0, 2))
        self.assertEqual(task_at(schedule, 3, 0), None)
        self.assertEqual(task_at(schedule, 3, 1), (1, 3))

    def test_root_task_orders_require_compatible_domains(self) -> None:
        task_domain = _domain((10, 2), identity=0)
        task_order_domain = _domain((-1, 1), kind="task_order", identity=0)
        wrong_size = _full_point_map(task_order_domain, task_domain, sympy.Integer(0))

        with self.assertRaisesRegex(ValueError, "compatible typed domains"):
            ReadinessGraph(
                root_task_orders=(wrong_size,),
                events=(),
            )
        with self.assertRaisesRegex(ValueError, "incompatible domains"):
            converse = wrong_size.converse()
            assert converse is not None
            _segment(0, converse, workers=(0, 2), dispatch_offset=0)

    def test_baseline_worker_schedule_preserves_pid_task_order(self) -> None:
        root_domains = (_domain((10, 4, 1), identity=0),)
        task_order_domain = _domain((-1, 4), kind="task_order", identity=0)
        task_order_relation = CoordinateRelation.point_map(
            task_order_domain,
            root_domains[0],
            tuple(
                (
                    ((-1, task_order_index, task_order_index + 1, 1),),
                    (sympy.Integer(task),),
                )
                for task_order_index, task in enumerate((0, 2, 1, 3))
            ),
        )

        schedule = _baseline_worker_schedule(
            root_domains,
            worker_count=2,
            root_task_orders=(task_order_relation,),
        )

        self.assertEqual(placement(schedule, 0, 0), (0, 0))
        self.assertEqual(placement(schedule, 0, 2), (1, 0))
        self.assertEqual(placement(schedule, 0, 1), (0, 1))
        self.assertEqual(placement(schedule, 0, 3), (1, 1))

    def test_worker_schedule_segment_uses_symbolic_order_across_rounds(self) -> None:
        task_axis = 10
        task_order_axis = 20
        task_domain = _domain((task_axis, 15), identity=2)
        task_order_domain = _domain((task_order_axis, 3), kind="task_order")
        segment = _segment(
            2,
            _full_point_map(
                task_order_domain,
                task_domain,
                10 + 2 * coordinate_axis_symbol(task_order_axis),
            ),
            workers=(2, 2),
            dispatch_offset=0,
        )

        self.assertEqual(segment_placement(segment, 10), (2, 0))
        self.assertEqual(segment_placement(segment, 12), (3, 0))
        self.assertEqual(segment_placement(segment, 14), (2, 1))
        self.assertEqual(segment_placement(segment, 11), None)
        self.assertEqual(segment_task_at(segment, 2, 1), 14)

    def test_worker_support_excludes_unused_segment_capacity(self) -> None:
        task_domain = _domain((10, 2), identity=0)
        schedule = _schedule(
            6,
            _segment(
                0,
                _one_dimensional_task_range(task_domain, 0, 2),
                workers=(1, 4),
                dispatch_offset=2,
            ),
        )

        self.assertEqual(schedule.workers_for_root(0), frozenset((3, 4)))
        self.assertEqual(schedule.dense_assignment(0), (1, 4, 2, 2))
        self.assertIsNone(schedule.contiguous_global_interval(0))

    def test_continuation_producers_preserve_key_major_order(self) -> None:
        root_domains = (
            _domain((10, 4, 1)),
            _domain((20, 2, 1)),
        )
        producer_domain, consumer_domain = _identify_root_domains(root_domains)
        readiness_key_domain = _domain((0, 2), kind="event", identity=0)
        producer_axis = coordinate_axis_symbol(10)
        consumer_axis = coordinate_axis_symbol(20)
        readiness_graph = _readiness_graph(
            (producer_domain, consumer_domain),
            ReadinessEvent(
                producers=(
                    _readiness_producer_from_publication(
                        producer_root=0,
                        producer_site_id=None,
                        publication=_full_point_map(
                            producer_domain,
                            readiness_key_domain,
                            sympy.floor(producer_axis / 2),
                        ),
                    ),
                ),
                consumers=(
                    ReadinessConsumer(
                        consumer_root=1,
                        consumer_site_id=None,
                        keys_by_consumer=_full_point_map(
                            consumer_domain, readiness_key_domain, consumer_axis
                        ),
                    ),
                ),
            ),
        )
        baseline = _baseline_worker_schedule(
            readiness_graph.root_domains,
            worker_count=2,
        )
        continuations = derive_final_arrival_continuations(readiness_graph)

        schedule = order_continuation_producers_by_readiness_key(
            readiness_graph,
            baseline,
            continuations,
        )

        self.assertEqual(task_order(schedule, 0), (0, 1, 2, 3))

    def test_worker_schedule_detects_dependency_order_cycle(self) -> None:
        graph = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", block_ids=(10,)),
            _access(root=1, kind="load", block_ids=(20,)),
        )
        root_domains = (
            _domain((10, 1, 1)),
            _domain((20, 1, 1)),
        )
        readiness_graph = _configured_readiness_graph(graph, root_domains)

        validate_worker_schedule(
            readiness_graph,
            _baseline_worker_schedule(
                readiness_graph.root_domains,
                worker_count=1,
            ),
        )
        reversed_schedule = _schedule(
            1,
            _segment(
                1,
                readiness_graph.root_task_orders[1],
                workers=(0, 1),
                dispatch_offset=0,
            ),
            _segment(
                0,
                readiness_graph.root_task_orders[0],
                workers=(0, 1),
                dispatch_offset=1,
            ),
        )
        with self.assertRaisesRegex(ValueError, "dependency/order cycle"):
            validate_worker_schedule(readiness_graph, reversed_schedule)

    def test_readiness_counter_supports_independent_consumers(self) -> None:
        producer_domain = _domain((10, 2), identity=0)
        first_consumer = _domain((20, 1), identity=1)
        second_consumer = _domain((30, 2), identity=2)
        readiness_key_domain = _domain(kind="event", identity=0)
        event = ReadinessCounterPlan(
            producers=(
                _readiness_producer_from_publication(
                    producer_root=0,
                    publication=CoordinateRelation.total(
                        producer_domain, readiness_key_domain
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=1,
                    keys_by_consumer=CoordinateRelation.total(
                        first_consumer, readiness_key_domain
                    ),
                ),
                ReadinessConsumer(
                    consumer_root=2,
                    keys_by_consumer=CoordinateRelation.total(
                        second_consumer, readiness_key_domain
                    ),
                ),
            ),
        )

        self.assertEqual(event.readiness_key_count, 1)
        self.assertEqual(event.uniform_arrival_count(), 2)
        self.assertIsNone(event.continuation_consumer)
        self.assertEqual(
            tuple(
                readiness_consumer.consumer_root
                for readiness_consumer in event.consumers
            ),
            (1, 2),
        )

    def test_readiness_counter_selection_keeps_independent_direct_consumers(
        self,
    ) -> None:
        root_domains = tuple(_domain((axis, 2, 1)) for axis in (10, 20, 30))
        root_domains = _identify_root_domains(root_domains)
        readiness_key_domain = _domain((0, 2), kind="event", identity=0)
        readiness_graph = _readiness_graph(
            root_domains,
            ReadinessEvent(
                producers=(
                    _readiness_producer_from_publication(
                        producer_root=0,
                        producer_site_id=None,
                        publication=_full_point_map(
                            root_domains[0],
                            readiness_key_domain,
                            coordinate_axis_symbol(10),
                        ),
                    ),
                ),
                consumers=tuple(
                    ReadinessConsumer(
                        consumer_root=root,
                        consumer_site_id=None,
                        keys_by_consumer=_full_point_map(
                            root_domains[root],
                            readiness_key_domain,
                            coordinate_axis_symbol(10 * (root + 1)),
                        ),
                        covered_obligations=frozenset(((root - 1, None, None),)),
                    )
                    for root in (1, 2)
                ),
            ),
        )
        (selected,) = choose_readiness_counters(
            readiness_graph,
            (),
            excluded_obligations=frozenset(((0, None, None),)),
        )

        self.assertEqual(selected.readiness_key_count, 2)
        self.assertEqual(
            tuple(
                readiness_consumer.consumer_root
                for readiness_consumer in selected.consumers
            ),
            (2,),
        )

    def test_readiness_counter_lowering_is_derived_from_the_semantic_graph(
        self,
    ) -> None:
        root_domains = (
            _domain((10, 4, 1)),
            _domain((20, 2, 2)),
        )
        root_domains = _identify_root_domains(root_domains)
        readiness_key_domain = _domain((0, 2), kind="event", identity=0)
        readiness_graph = _readiness_graph(
            root_domains,
            ReadinessEvent(
                producers=(
                    _readiness_producer_from_publication(
                        producer_root=0,
                        producer_site_id=None,
                        publication=_full_point_map(
                            root_domains[0],
                            readiness_key_domain,
                            sympy.floor(coordinate_axis_symbol(10) / 2),
                        ),
                    ),
                ),
                consumers=(
                    ReadinessConsumer(
                        consumer_root=1,
                        consumer_site_id=None,
                        keys_by_consumer=_full_point_map(
                            root_domains[1],
                            readiness_key_domain,
                            coordinate_axis_symbol(20),
                        ),
                    ),
                ),
            ),
        )
        continuations = derive_final_arrival_continuations(readiness_graph)

        (lowered,) = choose_readiness_counters(readiness_graph, continuations)

        self.assertEqual(
            _publication(lowered.producers[0]).materialize(),
            (
                frozenset((0,)),
                frozenset((0,)),
                frozenset((1,)),
                frozenset((1,)),
            ),
        )
        self.assertEqual(
            lowered.consumers[0].keys_by_consumer.materialize(),
            (frozenset((0,)), frozenset((1,))),
        )
        self.assertEqual(lowered.uniform_arrival_count(), 2)
        self.assertEqual(lowered.continuation_consumer_index, 0)

    def test_semantic_event_identity_does_not_depend_on_counter_lowering(
        self,
    ) -> None:
        dependency_graph = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", shape=(64,), block_ids=(10,)),
            _access(root=1, kind="load", shape=(64,), block_ids=(20,)),
        )
        root_domains = (
            _domain((10, 4, 16)),
            _domain((20, 2, 32)),
        )
        with mock.patch.object(
            cross_loop_scheduler,
            "_supports_readiness_counter_lowering",
            side_effect=AssertionError("semantic graph consulted lowering policy"),
        ):
            readiness_graph = _configured_readiness_graph(
                dependency_graph,
                root_domains,
            )

        (event,) = readiness_graph.events
        self.assertEqual(event.readiness_key_count, 2)
        self.assertIsNone(event.root_barrier_producer_root)
        with mock.patch.object(
            cross_loop_scheduler,
            "_supports_readiness_counter_lowering",
            return_value=False,
        ):
            selected = choose_readiness_counters(readiness_graph, ())
        self.assertEqual(selected, ())
        counters, barriers = cross_loop_scheduler._finalize_emitted_synchronization(
            dependency_graph=dependency_graph,
            root_domains=readiness_graph.root_domains,
            readiness_counters=selected,
        )
        self.assertEqual(counters, ())
        self.assertEqual(barriers, frozenset(((0, 1),)))

    def test_producer_set_quotient_is_lowering_only(self) -> None:
        dependency_graph = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", shape=(64,), block_ids=(10,)),
            _access(root=1, kind="load", shape=(64,), block_ids=(20,)),
        )
        root_domains = (
            _domain((10, 4, 16)),
            _domain((20, 64, 1)),
        )
        readiness_graph = _configured_readiness_graph(
            dependency_graph,
            root_domains,
        )

        (event,) = readiness_graph.events
        self.assertEqual(event.readiness_key_count, 64)
        self.assertFalse(
            cross_loop_scheduler._supports_readiness_counter_lowering(
                event.producers[0]
            )
        )
        plan = _configured_static_pipeline_plan(
            dependency_graph=dependency_graph,
            root_domains=root_domains,
            axis_geometry={10: (4, 16), 20: (64, 1)},
            worker_count=4,
        )

        self.assertEqual(plan.root_barrier_edges, frozenset())
        (counter,) = plan.readiness_counters
        self.assertEqual(counter.readiness_key_count, 4)
        self.assertEqual(counter.uniform_arrival_count(), 1)

    def test_counter_quotient_is_invariant_to_adjacent_source_pieces(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (_domain((10, 4, 1)), _domain((20, 64, 1)))
        )
        readiness_key_domain = _domain((0, 64), kind="event", identity=0)
        readiness_key = coordinate_axis_symbol(0)
        obligation = (0, None, None)

        def graph(source_intervals: tuple[tuple[int, int], ...]) -> ReadinessGraph:
            producer = ReadinessProducer(
                producer_root=0,
                producers_by_key=CoordinateRelation(
                    readiness_key_domain,
                    producer_domain,
                    tuple(
                        _CoordinateRelationPiece(
                            ((0, begin, end, 1),),
                            (
                                (
                                    10,
                                    sympy.floor(readiness_key / 16),
                                    sympy.floor(readiness_key / 16) + 1,
                                    1,
                                ),
                            ),
                        )
                        for begin, end in source_intervals
                    ),
                ),
            )
            consumer = ReadinessConsumer(
                consumer_root=1,
                keys_by_consumer=_full_point_map(
                    consumer_domain,
                    readiness_key_domain,
                    coordinate_axis_symbol(20),
                ),
                covered_obligations=frozenset((obligation,)),
            )
            return _readiness_graph(
                (producer_domain, consumer_domain),
                ReadinessEvent((producer,), (consumer,)),
            )

        canonical = choose_readiness_counters(graph(((0, 64),)), ())
        split = choose_readiness_counters(graph(((0, 32), (32, 64))), ())

        self.assertEqual(split, canonical)
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0].readiness_key_count, 4)

    def test_dynamic_counter_quotient_matches_specialized_controls(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)

        def configured_plan(batch_size: int | sympy.Expr):
            dependency_graph = _dependency_graph(
                [[10], [20]],
                _access(
                    root=0,
                    kind="store",
                    shape=(64 * batch_size,),
                    block_ids=(10,),
                ),
                _access(
                    root=1,
                    kind="load",
                    shape=(64 * batch_size,),
                    block_ids=(20,),
                ),
            )
            return _configured_static_pipeline_plan(
                dependency_graph=dependency_graph,
                root_domains=(
                    _domain((10, 4 * batch_size, 16)),
                    _domain((20, 64 * batch_size, 1)),
                ),
                axis_geometry={
                    10: (4 * batch_size, 16),
                    20: (64 * batch_size, 1),
                },
                worker_count=4,
            )

        with _forbid_schedule_enumeration():
            dynamic_plan = configured_plan(batch)
        self.assertEqual(dynamic_plan.root_barrier_edges, frozenset())
        (dynamic_counter,) = dynamic_plan.readiness_counters
        self.assertEqual(dynamic_counter.readiness_key_count_expr, 4 * batch)
        self.assertEqual(dynamic_counter.uniform_arrival_count(), 1)

        for batch_size in (0, 1, 3):
            substitutions = {batch: batch_size}
            dynamic_producer = dynamic_counter.producers[
                0
            ].producers_by_key.substitute_parameters(substitutions)
            dynamic_consumer = dynamic_counter.consumers[
                0
            ].keys_by_consumer.substitute_parameters(substitutions)
            self.assertEqual(dynamic_producer.source_domain.size, 4 * batch_size)
            self.assertEqual(dynamic_consumer.source_domain.size, 64 * batch_size)
            if batch_size == 0:
                self.assertEqual(dynamic_producer.pieces, ())
                self.assertEqual(dynamic_consumer.pieces, ())
                continue

            static_plan = configured_plan(batch_size)
            self.assertEqual(static_plan.root_barrier_edges, frozenset())
            (static_counter,) = static_plan.readiness_counters
            self.assertEqual(
                dynamic_producer.materialize(),
                static_counter.producers[0].producers_by_key.materialize(),
            )
            self.assertEqual(
                dynamic_consumer.materialize(),
                static_counter.consumers[0].keys_by_consumer.materialize(),
            )

    def test_nested_readiness_traversal_keeps_semantic_prerequisites(self) -> None:
        root_domains = tuple(
            _domain((axis, 2, 1), identity=root)
            for root, axis in enumerate((10, 20, 30))
        )
        first_key_domain = _domain((0, 2), kind="event", identity=0)
        second_key_domain = _domain((0, 2), kind="event", identity=1)
        first_obligation = (0, None, None)

        def event(
            producer_root: int,
            consumer_root: int,
            key_domain: CoordinateDomain,
            obligation: tuple[int, int | None, int | None],
        ) -> ReadinessEvent:
            producer_axis = root_domains[producer_root].axis_order[0]
            consumer_axis = root_domains[consumer_root].axis_order[0]
            return ReadinessEvent(
                producers=(
                    _readiness_producer_from_publication(
                        producer_root,
                        _full_point_map(
                            root_domains[producer_root],
                            key_domain,
                            coordinate_axis_symbol(producer_axis),
                        ),
                    ),
                ),
                consumers=(
                    ReadinessConsumer(
                        consumer_root,
                        _full_point_map(
                            root_domains[consumer_root],
                            key_domain,
                            coordinate_axis_symbol(consumer_axis),
                        ),
                        covered_obligations=frozenset((obligation,)),
                    ),
                ),
            )

        first_event = event(0, 1, first_key_domain, first_obligation)
        second_event = event(1, 2, second_key_domain, (1, None, None))
        readiness_graph = _readiness_graph(
            root_domains,
            first_event,
            second_event,
        )
        baseline = _baseline_worker_schedule(root_domains, worker_count=2)

        unfiltered = _event_ready_after_worker_steps(
            readiness_graph,
            second_event,
            worker_schedule=baseline,
            continuation_by_root={},
        )
        self.assertIsNotNone(unfiltered)
        assert unfiltered is not None
        self.assertEqual(unfiltered[1], frozenset((0, 1)))
        self.assertIs(readiness_graph.events[0], first_event)

    def test_nonstatic_layout_falls_back_to_root_readiness(self) -> None:
        plan = _dependency_graph(
            [[10], [20]],
            _access(
                root=0,
                kind="store",
                block_ids=(10,),
                layout_is_symbolically_exact=False,
            ),
            _access(root=1, kind="load", block_ids=(20,)),
        )

        (event,) = _configured_readiness_graph(plan, _one_dimensional_domains()).events
        self.assertIsNotNone(event.root_barrier_producer_root)
        self.assertEqual(event.root_barrier_producer_root, 0)

    def test_fanout_keeps_one_edge_per_consumer(self) -> None:
        plan = _dependency_graph(
            [[0], [1], [2]],
            _access(root=0, kind="store"),
            _access(root=1, kind="load", block_ids=(1,)),
            _access(root=2, kind="load", block_ids=(2,)),
        )

        self.assertEqual(
            [(edge.producer_root, edge.consumer_root) for edge in plan.edges],
            [(0, 1), (0, 2)],
        )
        configured = _configured_readiness_graph(
            plan,
            (
                _domain((0, 8, 16)),
                _domain((1, 8, 16)),
                _domain((2, 8, 16)),
            ),
        )
        (event,) = tuple(
            event
            for event in configured.events
            if any(
                readiness_producer.producer_root == 0
                for readiness_producer in event.producers
            )
        )
        self.assertIsNone(event.root_barrier_producer_root)
        self.assertEqual(
            {
                readiness_consumer.consumer_root
                for readiness_consumer in event.consumers
            },
            {1, 2},
        )

    def test_mixed_accesses_retain_exact_and_conservative_readiness(self) -> None:
        plan = _dependency_graph(
            [[0], [1]],
            _access(root=0, kind="store"),
            _access(root=1, kind="load", block_ids=(1,)),
            _access(root=0, allocation_id=1, kind="store"),
            _access(root=1, allocation_id=1, kind="load", block_ids=(1,), scales=(-1,)),
        )

        self.assertEqual(len(plan.edges), 2)
        configured = _configured_readiness_graph(
            plan,
            (
                _domain((0, 8, 16)),
                _domain((1, 8, 16)),
            ),
        )
        self.assertEqual(len(configured.events), 2)
        family_event = next(
            event
            for event in configured.events
            if event.root_barrier_producer_root is not None
        )
        self.assertEqual(family_event.root_barrier_producer_root, 0)

    def test_mixed_exact_and_unknown_accesses_use_root_barrier(self) -> None:
        dependency_graph = _dependency_graph(
            [[10, 11], [20]],
            _access(root=0, kind="store", shape=(1, 64), block_ids=(10, 11)),
            _access(root=1, kind="load", shape=(1, 64), block_ids=(20, 21)),
            _access(
                root=1,
                kind="load",
                shape=(1, 64),
                block_ids=(None, None),
                offsets=(None, None),
                masked=True,
            ),
        )
        schedule = _configured_static_pipeline_plan(
            dependency_graph=dependency_graph,
            root_domains=(
                _domain((10, 1, 1), (11, 4, 16)),
                _domain((20, 1, 1)),
            ),
            axis_geometry={10: (1, 1), 11: (4, 16), 20: (1, 1), 21: (4, 16)},
            worker_count=2,
        )

        self.assertEqual(schedule.root_barrier_edges, frozenset(((0, 1),)))

    def test_singleton_producer_uses_root_barrier(self) -> None:
        dependency_graph = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", block_ids=(10,)),
            _access(root=1, kind="load", block_ids=(20,)),
        )
        schedule = _configured_static_pipeline_plan(
            dependency_graph=dependency_graph,
            root_domains=(
                _domain((10, 1, 128)),
                _domain((20, 4, 32)),
            ),
            axis_geometry={10: (1, 128), 20: (4, 32)},
            worker_count=4,
        )

        self.assertEqual(schedule.root_barrier_edges, frozenset(((0, 1),)))

    def test_root_barrier_path_elides_redundant_exact_task_wait(self) -> None:
        dependency_graph = _dependency_graph(
            [[10], [20], [30], [40]],
            _access(root=0, kind="store", block_ids=(10,)),
            _access(root=1, kind="load", block_ids=(None,), offsets=(None,)),
            _access(root=1, allocation_id=1, kind="store", block_ids=(20,)),
            _access(
                root=2, allocation_id=1, kind="load", block_ids=(None,), offsets=(None,)
            ),
            _access(root=2, allocation_id=2, kind="store", block_ids=(30,)),
            _access(
                root=3, allocation_id=2, kind="load", block_ids=(None,), offsets=(None,)
            ),
            _access(root=3, kind="load", block_ids=(40,)),
        )
        root_domains = tuple(_domain((10 + root * 10, 8, 16)) for root in range(4))

        schedule = _configured_static_pipeline_plan(
            dependency_graph=dependency_graph,
            root_domains=root_domains,
            axis_geometry={
                10: (8, 16),
                20: (8, 16),
                30: (8, 16),
                40: (8, 16),
            },
            worker_count=8,
        )

        self.assertEqual(
            schedule.root_barrier_edges,
            frozenset(((0, 1), (1, 2), (2, 3))),
        )

    def test_worker_schedule_derives_access_ready_overlap(self) -> None:
        dependency_graph = _dependency_graph(
            [[10, 11], [20, 21], [30]],
            _access(root=0, kind="store", shape=(1, 128), block_ids=(10, 11)),
            _access(root=1, kind="load", shape=(1, 128), block_ids=(20, 21)),
            _access(
                root=1,
                allocation_id=1,
                kind="store",
                shape=(1, 128),
                block_ids=(20, 21),
            ),
            _access(
                root=2, allocation_id=1, kind="load", shape=(1, 128), block_ids=(30, 31)
            ),
        )
        dependency_graph = dataclasses.replace(
            dependency_graph,
            execution_sites=(
                ExecutionSite(
                    0, 0, 0, (), None, "root", (10, 11), (10, 11), True, False
                ),
                ExecutionSite(
                    1, 1, 1, (), None, "root", (20, 21), (20, 21), True, False
                ),
                ExecutionSite(2, 2, 2, (), None, "root", (30,), (30,), True, False),
                ExecutionSite(
                    3,
                    2,
                    3,
                    ((0, 0),),
                    2,
                    "loop",
                    (31,),
                    (30, 31),
                    True,
                    True,
                ),
            ),
            site_ids_by_access=((0,), (1,), (1,), (3,)),
        )
        root_domains = (
            _domain((10, 1, 1), (11, 8, 16)),
            _domain((20, 1, 1), (21, 4, 32)),
            _domain((30, 1, 1)),
        )
        kwargs = {
            "dependency_graph": dependency_graph,
            "root_domains": root_domains,
            "axis_geometry": {
                10: (1, 1),
                11: (8, 16),
                20: (1, 1),
                21: (4, 32),
                30: (1, 1),
                31: (4, 32),
            },
            "worker_count": 8,
        }

        schedule = _configured_static_pipeline_plan(**{**kwargs, "worker_count": 6})

        root_events = tuple(
            plan
            for plan in schedule.readiness_counters
            if all(
                readiness_consumer.consumer_site_id is None
                for readiness_consumer in plan.consumers
            )
        )
        self.assertEqual(len(root_events), 1)
        event = root_events[0]
        self.assertEqual(
            (
                event.producers[0].producer_root,
                event.consumers[0].consumer_root,
                event.continuation_consumer.consumer_root
                if event.continuation_consumer is not None
                else None,
                event.uniform_arrival_count(),
            ),
            (0, 1, 1, 2),
        )
        local_events = tuple(
            plan
            for plan in schedule.readiness_counters
            if plan.continuation_consumer is not None
        )
        self.assertEqual(len(local_events), 1)
        self.assertEqual(local_events[0].continuation_consumer_index, 0)
        assert local_events[0].continuation_consumer is not None
        self.assertEqual(local_events[0].continuation_consumer.consumer_root, 1)
        self.assertEqual(schedule.worker_schedule.worker_count, 6)
        nested_loop_events = tuple(
            plan
            for plan in schedule.readiness_counters
            if any(
                readiness_consumer.consumer_site_id is not None
                for readiness_consumer in plan.consumers
            )
        )
        self.assertEqual(len(nested_loop_events), 1)
        self.assertEqual(
            _expected_arrivals(
                nested_loop_events[0].readiness_key_domain,
                nested_loop_events[0].producers,
            ),
            (3, 1),
        )
        consumer_placement = placement(schedule.worker_schedule, 2, 0)
        producer_placement = placement(schedule.worker_schedule, 0, 6)
        self.assertIsNotNone(consumer_placement)
        self.assertIsNotNone(producer_placement)
        assert consumer_placement is not None and producer_placement is not None
        self.assertEqual((consumer_placement[1], producer_placement[1]), (1, 1))
        self.assertNotEqual(consumer_placement[0], producer_placement[0])

        exact = _configured_static_pipeline_plan(**{**kwargs, "worker_count": 7})
        self.assertEqual(exact.worker_schedule.worker_count, 7)
        self.assertNotEqual(exact.worker_schedule, schedule.worker_schedule)

        default_schedule = _configured_static_pipeline_plan(**kwargs)
        self.assertEqual(len(default_schedule.readiness_counters), 2)
        self.assertEqual(
            default_schedule.root_barrier_edges,
            frozenset(),
        )
        short_domains = (
            dataclasses.replace(
                root_domains[0],
                axis_counts_items=((10, 1), (11, 4)),
                block_sizes_items=((10, 1), (11, 32)),
            ),
            dataclasses.replace(
                root_domains[1],
                axis_counts_items=((20, 1), (21, 2)),
                block_sizes_items=((20, 1), (21, 64)),
            ),
            root_domains[2],
        )
        short_schedule = _configured_static_pipeline_plan(
            **{
                **kwargs,
                "root_domains": short_domains,
                "axis_geometry": {
                    10: (1, 1),
                    11: (4, 32),
                    20: (1, 1),
                    21: (2, 64),
                    30: (1, 1),
                    31: (2, 64),
                },
            }
        )
        self.assertEqual(len(short_schedule.readiness_counters), 2)
        self.assertEqual(
            short_schedule.root_barrier_edges,
            frozenset(),
        )

    def test_nested_split_nested_loop_at_readiness_follow_worker_readiness(
        self,
    ) -> None:
        root_domains = (
            _domain((10, 4, 1)),
            _domain((20, 1, 1)),
        )
        root_domains = _identify_root_domains(root_domains)
        nested_loop_domain = _domain((20, 1, 1), (21, 4, 1), identity=7)
        readiness_key_domain = _domain((0, 4), kind="event", identity=0)
        readiness_graph = _readiness_graph(
            root_domains,
            ReadinessEvent(
                producers=(
                    _readiness_producer_from_publication(
                        producer_root=0,
                        producer_site_id=None,
                        publication=_full_point_map(
                            root_domains[0],
                            readiness_key_domain,
                            coordinate_axis_symbol(10),
                        ),
                    ),
                ),
                consumers=(
                    ReadinessConsumer(
                        consumer_root=1,
                        consumer_site_id=7,
                        keys_by_consumer=_full_point_map(
                            nested_loop_domain,
                            readiness_key_domain,
                            coordinate_axis_symbol(21),
                        ),
                    ),
                ),
            ),
        )
        schedule = _schedule(
            4,
            _segment(
                0,
                _one_dimensional_task_range(root_domains[0], 0, 3),
                workers=(0, 3),
                dispatch_offset=0,
            ),
            _segment(
                0,
                _one_dimensional_task_range(root_domains[0], 3, 1),
                workers=(0, 1),
                dispatch_offset=1,
            ),
            _segment(
                1,
                readiness_graph.root_task_orders[1],
                workers=(3, 1),
                dispatch_offset=2,
            ),
        )

        placed, plans = place_nested_loop_consumers(readiness_graph, schedule, ())

        self.assertEqual(placement(placed, 1, 0), (3, 1))
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(
            _expected_arrivals(plan.readiness_key_domain, plan.producers),
            (3, 1),
        )
        self.assertEqual(
            _publication(plan.producers[0]).materialize(),
            (
                frozenset((0,)),
                frozenset((0,)),
                frozenset((0,)),
                frozenset((1,)),
            ),
        )
        self.assertEqual(
            plan.consumers[0].keys_by_consumer.materialize(),
            (
                frozenset((0,)),
                frozenset((0,)),
                frozenset((0,)),
                frozenset((1,)),
            ),
        )
        self.assertEqual(plan.consumers[0].consumer_site_id, 7)

    def test_nested_nested_loop_entry_counter_survives_without_early_placement(
        self,
    ) -> None:
        producer_domain = _domain((10, 2, 1), (11, 4, 1), identity=0)
        consumer_domain = _domain((20, 2, 1), identity=1)
        nested_loop_domain = _domain((20, 2, 1), (21, 4, 1), identity=7)
        readiness_key_domain = _domain((0, 2), (1, 4), kind="event", identity=0)
        readiness_graph = ReadinessGraph(
            root_task_orders=(
                pid_task_order(producer_domain, (10, 11)),
                pid_task_order(consumer_domain, (20,)),
            ),
            events=(
                ReadinessEvent(
                    producers=(
                        _readiness_producer_from_publication(
                            producer_root=0,
                            publication=_full_point_map(
                                producer_domain,
                                readiness_key_domain,
                                coordinate_axis_symbol(10),
                                coordinate_axis_symbol(11),
                            ),
                        ),
                    ),
                    consumers=(
                        ReadinessConsumer(
                            consumer_root=1,
                            consumer_site_id=7,
                            keys_by_consumer=_full_point_map(
                                nested_loop_domain,
                                readiness_key_domain,
                                coordinate_axis_symbol(20),
                                coordinate_axis_symbol(21),
                            ),
                        ),
                    ),
                ),
            ),
        )
        for worker_count in (1, 2):
            with self.subTest(worker_count=worker_count):
                schedule = _build_baseline_worker_schedule(
                    readiness_graph.root_domains,
                    readiness_graph.root_task_orders,
                    worker_count=worker_count,
                )

                placed, plans = place_nested_loop_consumers(
                    readiness_graph,
                    schedule,
                    (),
                )

                self.assertEqual(placed, schedule)
                self.assertEqual(len(plans), 1)
                self.assertEqual(plans[0].readiness_key_count, 2)
                self.assertEqual(
                    _expected_arrivals(
                        plans[0].readiness_key_domain,
                        plans[0].producers,
                    ),
                    (4, 4),
                )
                self.assertEqual(plans[0].consumers[0].consumer_site_id, 7)

    def test_nested_site_identity_readiness_uses_one_split_point(self) -> None:
        """Per-iteration readiness is coarsened to one compact split point."""
        root_domains = (
            _domain((10, 4, 1)),
            _domain((20, 1, 1)),
        )
        root_domains = _identify_root_domains(root_domains)
        nested_loop_domain = _domain((20, 1, 1), (21, 4, 1), identity=7)
        readiness_key_domain = _domain((0, 4), kind="event", identity=0)
        readiness_graph = _readiness_graph(
            root_domains,
            ReadinessEvent(
                producers=(
                    _readiness_producer_from_publication(
                        producer_root=0,
                        publication=_full_point_map(
                            root_domains[0],
                            readiness_key_domain,
                            coordinate_axis_symbol(10),
                        ),
                    ),
                ),
                consumers=(
                    ReadinessConsumer(
                        consumer_root=1,
                        consumer_site_id=7,
                        keys_by_consumer=_full_point_map(
                            nested_loop_domain,
                            readiness_key_domain,
                            coordinate_axis_symbol(21),
                        ),
                    ),
                ),
            ),
        )
        schedule = _schedule(
            4,
            _segment(
                0,
                readiness_graph.root_task_orders[0],
                workers=(0, 1),
                dispatch_offset=0,
            ),
            _segment(
                1,
                readiness_graph.root_task_orders[1],
                workers=(3, 1),
                dispatch_offset=5,
            ),
        )

        placed, plans = place_nested_loop_consumers(readiness_graph, schedule, ())

        self.assertEqual(placement(placed, 1, 0), (3, 1))
        self.assertEqual(len(plans), 1)
        self.assertEqual(
            _expected_arrivals(plans[0].readiness_key_domain, plans[0].producers),
            (1, 3),
        )

    def test_nested_loop_placement_keeps_transitive_worker_liveness(
        self,
    ) -> None:
        """A moved wait must not block an upstream prerequisite on its worker."""
        root_domains = (
            _domain((10, 4, 1)),
            _domain((20, 4, 1)),
            _domain((30, 1, 1)),
        )
        root_domains = _identify_root_domains(root_domains)
        nested_loop_domain = _domain((30, 1, 1), (31, 4, 1), identity=7)

        def identity_keys(
            source_domain: CoordinateDomain,
            source_axis: int,
            event_id: int,
        ) -> tuple[CoordinateDomain, CoordinateRelation]:
            readiness_key_domain = _domain((0, 4), kind="event", identity=event_id)
            return readiness_key_domain, CoordinateRelation.point_map(
                source_domain,
                readiness_key_domain,
                (
                    (
                        tuple(
                            (axis, 0, source_domain.axis_counts[axis], 1)
                            for axis in source_domain.axis_order
                        ),
                        (coordinate_axis_symbol(source_axis),),
                    ),
                ),
            )

        first_keys, a_to_first = identity_keys(root_domains[0], 10, 0)
        _, first_use = identity_keys(root_domains[1], 20, 0)
        second_keys, b_to_second = identity_keys(root_domains[1], 20, 1)
        _, nested_keys_by_consumer = identity_keys(nested_loop_domain, 31, 1)
        readiness_graph = _readiness_graph(
            root_domains,
            ReadinessEvent(
                producers=(_readiness_producer_from_publication(0, a_to_first),),
                consumers=(ReadinessConsumer(1, first_use),),
            ),
            ReadinessEvent(
                producers=(_readiness_producer_from_publication(1, b_to_second),),
                consumers=(
                    ReadinessConsumer(2, nested_keys_by_consumer, consumer_site_id=7),
                ),
            ),
        )

        def task_segment(
            root: int,
            task_begin: int,
            task_count: int,
            worker: int,
            worker_step: int,
        ) -> WorkerScheduleSegment:
            return _segment(
                root,
                _one_dimensional_task_range(root_domains[root], task_begin, task_count),
                workers=(worker, task_count),
                dispatch_offset=worker_step * task_count,
            )

        schedule = _schedule(
            4,
            task_segment(0, 0, 3, 0, 0),
            task_segment(0, 3, 1, 3, 3),
            task_segment(1, 0, 3, 0, 1),
            task_segment(1, 3, 1, 0, 4),
            task_segment(2, 0, 1, 3, 6),
        )

        placed, plans = place_nested_loop_consumers(readiness_graph, schedule, ())

        # Worker 3 looks idle at step 2, but its A task at step 3 is a
        # prerequisite of B task 3. Placing C there would form C -> B -> A
        # while A remains later on C's blocked worker. Worker 2 is safe.
        self.assertEqual(placement(placed, 2, 0), (2, 2))
        self.assertEqual(len(plans), 1)
        validate_worker_schedule(readiness_graph, placed)

        # P0 -> C -> P1 on worker 3, combined with P1 -> B1 -> C, is the
        # resident nested-wait cycle that a first-checkpoint-only certificate
        # would miss.  The symbolic full-frontier proof must reject it.
        unsafe = _schedule(
            4,
            task_segment(0, 0, 3, 0, 0),
            task_segment(1, 0, 3, 0, 1),
            task_segment(2, 0, 1, 3, 2),
            task_segment(0, 3, 1, 3, 3),
            task_segment(1, 3, 1, 0, 4),
        )
        counter_plans = tuple(
            ReadinessCounterPlan(event.producers, event.consumers)
            for event in readiness_graph.events
        )
        with _forbid_schedule_enumeration():
            self.assertFalse(
                cross_loop_scheduler._schedule_is_progress_safe(
                    unsafe,
                    readiness_graph,
                    counter_plans,
                    frozenset(),
                )
            )

    def test_nested_loop_placement_preserves_source_order_on_each_worker(
        self,
    ) -> None:
        root_domains = _identify_root_domains(
            (
                _domain((10, 5, 1)),
                _domain((20, 4, 1)),
                _domain((30, 1, 1)),
            )
        )
        nested_loop_domain = _domain((30, 1, 1), (31, 5, 1), identity=7)
        nested_key_domain = _domain((0, 5), kind="event", identity=0)
        producer_to_key = _full_point_map(
            root_domains[0], nested_key_domain, coordinate_axis_symbol(10)
        )
        producers_by_key = producer_to_key.converse()
        assert producers_by_key is not None
        keys_by_nested_iteration = _full_point_map(
            nested_loop_domain, nested_key_domain, coordinate_axis_symbol(31)
        )
        family_done_domain = _domain(kind="event", identity=1)
        readiness_graph = _readiness_graph(
            root_domains,
            ReadinessEvent(
                producers=(ReadinessProducer(0, producers_by_key),),
                consumers=(
                    ReadinessConsumer(2, keys_by_nested_iteration, consumer_site_id=7),
                ),
            ),
            ReadinessEvent(
                producers=(
                    ReadinessProducer(
                        1, CoordinateRelation.total(family_done_domain, root_domains[1])
                    ),
                ),
                consumers=(
                    ReadinessConsumer(
                        2, CoordinateRelation.total(root_domains[2], family_done_domain)
                    ),
                ),
            ),
        )
        baseline = _build_baseline_worker_schedule(
            readiness_graph.root_domains,
            readiness_graph.root_task_orders,
            worker_count=4,
        )

        placed, plans = place_nested_loop_consumers(readiness_graph, baseline, ())

        self.assertEqual(placement(baseline, 2, 0), (0, 3))
        self.assertEqual(placement(placed, 2, 0), (0, 3))
        self.assertEqual(len(plans), 1)
        validate_worker_schedule(readiness_graph, placed)

    def test_nested_split_nested_loop_at_readiness_compose_sibling_sites(self) -> None:
        root_domains = (
            _domain((10, 4, 1)),
            _domain((20, 4, 1)),
            _domain((30, 1, 1)),
        )
        root_domains = _identify_root_domains(root_domains)
        nested_loop_domains = tuple(
            _domain((30, 1, 1), (nested_axis, 4, 1), identity=site_id)
            for site_id, nested_axis in ((7, 31), (8, 32))
        )
        site_domains: tuple[CoordinateDomain | None, ...] = (
            *(None for _ in range(7)),
            *nested_loop_domains,
        )
        events = []
        for producer_root, site_id, nested_axis in ((0, 7, 31), (1, 8, 32)):
            readiness_key_domain = _domain((0, 4), kind="event", identity=producer_root)
            events.append(
                ReadinessEvent(
                    producers=(
                        _readiness_producer_from_publication(
                            producer_root=producer_root,
                            producer_site_id=None,
                            publication=_full_point_map(
                                root_domains[producer_root],
                                readiness_key_domain,
                                coordinate_axis_symbol(
                                    root_domains[producer_root].axis_order[0]
                                ),
                            ),
                        ),
                    ),
                    consumers=(
                        ReadinessConsumer(
                            consumer_root=2,
                            consumer_site_id=site_id,
                            keys_by_consumer=_full_point_map(
                                site_domains[site_id],
                                readiness_key_domain,
                                coordinate_axis_symbol(nested_axis),
                            ),
                            covered_obligations=frozenset(
                                ((producer_root, None, site_id),)
                            ),
                        ),
                    ),
                )
            )
        readiness_graph = _readiness_graph(root_domains, *events)
        schedule = _schedule(
            4,
            _segment(
                0,
                _one_dimensional_task_range(root_domains[0], 0, 4),
                workers=(0, 4),
                dispatch_offset=0,
            ),
            _segment(
                1,
                _one_dimensional_task_range(root_domains[1], 0, 3),
                workers=(0, 4),
                dispatch_offset=4,
            ),
            _segment(
                1,
                _one_dimensional_task_range(root_domains[1], 3, 1),
                workers=(0, 4),
                dispatch_offset=8,
            ),
            _segment(
                2,
                readiness_graph.root_task_orders[2],
                workers=(3, 1),
                dispatch_offset=3,
            ),
        )

        placed, plans = place_nested_loop_consumers(readiness_graph, schedule, ())

        self.assertEqual(placement(placed, 2, 0), (3, 1))
        self.assertEqual(len(plans), 2)
        plans_by_site = {plan.consumers[0].consumer_site_id: plan for plan in plans}
        self.assertEqual(
            _expected_arrivals(
                plans_by_site[7].readiness_key_domain,
                plans_by_site[7].producers,
            ),
            (4,),
        )
        self.assertEqual(
            plans_by_site[7].consumers[0].keys_by_consumer.materialize(),
            (frozenset((0,)),) * 4,
        )
        self.assertEqual(
            _expected_arrivals(
                plans_by_site[8].readiness_key_domain,
                plans_by_site[8].producers,
            ),
            (4,),
        )
        self.assertEqual(
            plans_by_site[8].consumers[0].keys_by_consumer.materialize(),
            (
                frozenset((0,)),
                frozenset((0,)),
                frozenset((0,)),
                frozenset((0,)),
            ),
        )

    def test_multi_producer_join_uses_one_readiness_event(self) -> None:
        dependency_graph = _dependency_graph(
            [[10], [20], [30]],
            _access(root=0, kind="store", block_ids=(10,)),
            _access(root=1, allocation_id=1, kind="store", block_ids=(20,)),
            _access(root=2, kind="load", block_ids=(30,)),
            _access(root=2, allocation_id=1, kind="load", block_ids=(30,)),
        )
        root_domains = tuple(
            _domain((block_id, 8, 16)) for root, block_id in enumerate((10, 20, 30))
        )

        schedule = _configured_static_pipeline_plan(
            dependency_graph=dependency_graph,
            root_domains=root_domains,
            axis_geometry={10: (8, 16), 20: (8, 16), 30: (8, 16)},
            worker_count=8,
        )

        self.assertEqual(schedule.root_barrier_edges, frozenset())
        self.assertEqual(len(schedule.readiness_counters), 1)
        event = schedule.readiness_counters[0]
        self.assertEqual(event.consumers[0].consumer_root, 2)
        self.assertEqual(event.uniform_arrival_count(), 2)
        self.assertEqual(
            [
                (
                    readiness_producer.producer_root,
                    readiness_producer.arrival_count_by_key.constant_value()
                    if readiness_producer.arrival_count_by_key is not None
                    else None,
                )
                for readiness_producer in event.producers
            ],
            [(0, 1), (1, 1)],
        )

    def test_repeated_join_producers_coalesce_consumer_tasks(self) -> None:
        dependency_graph = _dependency_graph(
            [[10], [30], [22, 20, 21]],
            _access(root=0, kind="store", shape=(32,), block_ids=(10,)),
            _access(root=1, allocation_id=1, kind="store", shape=(8,), block_ids=(30,)),
            _access(root=2, kind="load", shape=(8, 4), block_ids=(20, 21)),
            _access(root=2, allocation_id=1, kind="load", shape=(8,), block_ids=(20,)),
        )
        root_domains = (
            _domain((10, 32, 1)),
            _domain((30, 8, 1)),
            _domain((22, 4, 1), (20, 8, 1), (21, 1, 4)),
        )

        schedule = _configured_static_pipeline_plan(
            dependency_graph=dependency_graph,
            root_domains=root_domains,
            axis_geometry={
                10: (32, 1),
                20: (8, 1),
                21: (1, 4),
                22: (4, 1),
                30: (8, 1),
            },
            worker_count=32,
        )

        self.assertEqual(schedule.root_barrier_edges, frozenset())
        self.assertEqual(len(schedule.readiness_counters), 1)
        self.assertFalse(
            any(
                event.continuation_consumer is not None
                for event in schedule.readiness_counters
            )
        )
        event = schedule.readiness_counters[0]
        self.assertIsNone(event.continuation_consumer)
        self.assertEqual(event.readiness_key_count, 8)
        self.assertEqual(event.uniform_arrival_count(), 5)
        self.assertEqual(
            event.consumers[0].keys_by_consumer.materialize(),
            tuple(frozenset((i // 4,)) for i in range(32)),
        )
        self.assertEqual(
            [
                readiness_producer.arrival_count_by_key.constant_value()
                if readiness_producer.arrival_count_by_key is not None
                else None
                for readiness_producer in event.producers
            ],
            [4, 1],
        )

    def test_large_flattened_ready_groups_have_no_task_product_cutoff(self) -> None:
        heads = 513
        width = 4
        splits = 4
        elements = heads * width
        dependency_graph = _dependency_graph(
            [[10], [22, 20, 21]],
            _access(root=0, kind="store", shape=(elements,), block_ids=(10,)),
            _access(root=1, kind="load", shape=(heads, width), block_ids=(20, 21)),
        )
        root_domains = (
            _domain((10, elements, 1)),
            _domain((22, splits, 1), (20, heads, 1), (21, 1, width)),
        )

        schedule = _configured_static_pipeline_plan(
            dependency_graph=dependency_graph,
            root_domains=root_domains,
            axis_geometry={
                10: (elements, 1),
                20: (heads, 1),
                21: (1, width),
                22: (splits, 1),
            },
            worker_count=128,
        )

        self.assertEqual(schedule.root_barrier_edges, frozenset())
        self.assertEqual(len(schedule.readiness_counters), 1)
        event = schedule.readiness_counters[0]
        self.assertEqual(event.readiness_key_count, heads)
        self.assertEqual(event.uniform_arrival_count(), width)
        self.assertEqual(
            event.consumers[0].keys_by_consumer.materialize(),
            tuple(frozenset((task // splits,)) for task in range(heads * splits)),
        )
        self.assertEqual(len(event.producers[0].producers_by_key.pieces), 1)
        self.assertEqual(len(event.consumers[0].keys_by_consumer.pieces), 1)

    def test_strided_ready_groups_use_exact_coordinates_with_overlapping_hulls(
        self,
    ) -> None:
        columns = 8
        splits = 4
        dependency_graph = _dependency_graph(
            [[10], [22, 20]],
            _access(
                root=0,
                kind="store",
                shape=(2, columns),
                block_ids=(None, 10),
                full_slice=(True, False),
            ),
            _access(
                root=1,
                kind="load",
                shape=(2, columns),
                block_ids=(None, 20),
                full_slice=(True, False),
            ),
        )
        root_domains = (
            _domain((10, columns, 1)),
            _domain((22, splits, 1), (20, columns, 1)),
        )

        schedule = _configured_static_pipeline_plan(
            dependency_graph=dependency_graph,
            root_domains=root_domains,
            axis_geometry={10: (columns, 1), 20: (columns, 1), 22: (splits, 1)},
            worker_count=32,
        )

        self.assertEqual(schedule.root_barrier_edges, frozenset())
        self.assertEqual(len(schedule.readiness_counters), 1)
        event = schedule.readiness_counters[0]
        self.assertEqual(event.readiness_key_count, columns)
        self.assertEqual(event.uniform_arrival_count(), 1)
        self.assertEqual(
            event.consumers[0].keys_by_consumer.materialize(),
            tuple(frozenset((task // splits,)) for task in range(columns * splits)),
        )

    def test_multiple_access_events_in_one_root_fall_back_together(self) -> None:
        dependency_graph = _dependency_graph(
            [[10, 11], [20, 21], [30]],
            # Deliberately incomplete affine metadata forces conservative fallback.
            _access(
                root=0,
                kind="store",
                shape=(1, 128),
                block_ids=(10, 11),
                scales=(1,),
                offsets=(0,),
            ),
            _access(
                root=1,
                allocation_id=1,
                kind="store",
                shape=(1, 128),
                block_ids=(20, 21),
                scales=(1,),
                offsets=(0,),
            ),
            _access(
                root=2,
                kind="load",
                shape=(1, 128),
                block_ids=(30, 31),
                scales=(1,),
                offsets=(0,),
            ),
            _access(
                root=2,
                allocation_id=1,
                kind="load",
                shape=(1, 128),
                block_ids=(30, 32),
                scales=(1,),
                offsets=(0,),
            ),
        )
        root_domains = (
            _domain((10, 1, 1), (11, 8, 16)),
            _domain((20, 1, 1), (21, 8, 16)),
            _domain((30, 1, 1)),
        )

        schedule = _configured_static_pipeline_plan(
            dependency_graph=dependency_graph,
            root_domains=root_domains,
            axis_geometry={
                10: (1, 1),
                11: (8, 16),
                20: (1, 1),
                21: (8, 16),
                30: (1, 1),
                31: (8, 16),
                32: (8, 16),
            },
            worker_count=4,
        )

        self.assertEqual(
            schedule.root_barrier_edges,
            frozenset(((0, 2), (1, 2))),
        )

    def test_worker_schedule_handles_independent_components(self) -> None:
        accesses: list[TileAccess] = []
        root_domains: list[CoordinateDomain] = []
        axis_geometry: dict[int, tuple[int, int]] = {}
        for component in range(2):
            root_base = component * 3
            block_base = 10 + component * 30
            access_base = component * 4
            allocation_base = component * 2
            accesses.extend(
                (
                    _access(
                        root=root_base,
                        allocation_id=allocation_base,
                        kind="store",
                        shape=(1, 128),
                        block_ids=(block_base, block_base + 1),
                    ),
                    _access(
                        root=root_base + 1,
                        allocation_id=allocation_base,
                        kind="load",
                        shape=(1, 128),
                        block_ids=(block_base + 10, block_base + 11),
                    ),
                    _access(
                        root=root_base + 1,
                        allocation_id=allocation_base + 1,
                        kind="store",
                        shape=(1, 128),
                        block_ids=(block_base + 10, block_base + 11),
                    ),
                    _access(
                        root=root_base + 2,
                        allocation_id=allocation_base + 1,
                        kind="load",
                        shape=(1, 128),
                        block_ids=(block_base + 20, block_base + 21),
                    ),
                )
            )
            root_domains.extend(
                (
                    _domain((block_base, 1, 1), (block_base + 1, 8, 16)),
                    _domain((block_base + 10, 1, 1), (block_base + 11, 4, 32)),
                    _domain((block_base + 20, 1, 1)),
                )
            )
            axis_geometry.update(
                {
                    block_base: (1, 1),
                    block_base + 1: (8, 16),
                    block_base + 10: (1, 1),
                    block_base + 11: (4, 32),
                    block_base + 20: (1, 1),
                    block_base + 21: (4, 32),
                }
            )

        dependency_graph = _dependency_graph(
            [list(domain.axis_order) for domain in root_domains], *tuple(accesses)
        )
        root_sites = tuple(
            ExecutionSite(
                site_id=root,
                root=root,
                graph_id=root,
                callsite_path=(),
                parent_site_id=None,
                kind="root",
                local_axis_order=domain.axis_order,
                logical_axis_order=domain.axis_order,
                executes_unconditionally=True,
                can_split_loop=False,
            )
            for root, domain in enumerate(root_domains)
        )
        nested_loops = tuple(
            ExecutionSite(
                site_id=6 + component,
                root=component * 3 + 2,
                graph_id=6 + component,
                callsite_path=((0, 0),),
                parent_site_id=component * 3 + 2,
                kind="loop",
                local_axis_order=(10 + component * 30 + 21,),
                logical_axis_order=(
                    10 + component * 30 + 20,
                    10 + component * 30 + 21,
                ),
                executes_unconditionally=True,
                can_split_loop=True,
            )
            for component in range(2)
        )
        site_ids_by_access: list[tuple[int, ...]] = [()] * len(accesses)
        for component in range(2):
            root_base = component * 3
            access_base = component * 4
            site_ids_by_access[access_base] = (root_base,)
            site_ids_by_access[access_base + 1] = (root_base + 1,)
            site_ids_by_access[access_base + 2] = (root_base + 1,)
            site_ids_by_access[access_base + 3] = (6 + component,)
        dependency_graph = dataclasses.replace(
            dependency_graph,
            execution_sites=(*root_sites, *nested_loops),
            site_ids_by_access=tuple(site_ids_by_access),
        )
        kwargs = {
            "dependency_graph": dependency_graph,
            "root_domains": tuple(root_domains),
            "axis_geometry": axis_geometry,
            "worker_count": 8,
        }

        schedule = _configured_static_pipeline_plan(**kwargs)
        self.assertEqual(len(schedule.readiness_counters), 4)
        self.assertEqual(
            schedule.root_barrier_edges,
            frozenset(),
        )
        overlapped = _configured_static_pipeline_plan(**{**kwargs, "worker_count": 6})
        self.assertEqual(overlapped.worker_schedule.worker_count, 6)
        nested_loop_events = tuple(
            plan
            for plan in overlapped.readiness_counters
            if any(
                readiness_consumer.consumer_site_id is not None
                for readiness_consumer in plan.consumers
            )
        )
        self.assertEqual(
            [
                (
                    plan.producers[0].producer_root,
                    plan.consumers[0].consumer_root,
                    _expected_arrivals(plan.readiness_key_domain, plan.producers),
                )
                for plan in nested_loop_events
            ],
            [(1, 2, (3, 1)), (4, 5, (3, 1))],
        )
        self.assertEqual(overlapped.root_barrier_edges, frozenset())
        first_sink_placement = placement(overlapped.worker_schedule, 2, 0)
        second_sink_placement = placement(overlapped.worker_schedule, 5, 0)
        self.assertIsNotNone(first_sink_placement)
        self.assertIsNotNone(second_sink_placement)
        assert first_sink_placement is not None and second_sink_placement is not None
        self.assertEqual(
            sorted((first_sink_placement[1], second_sink_placement[1])),
            [2, 3],
        )
        for source_root, sink_placement in (
            (0, first_sink_placement),
            (3, second_sink_placement),
        ):
            source_bounds = overlapped.worker_schedule.worker_step_bounds_for_root(
                source_root
            )
            self.assertIsNotNone(source_bounds)
            assert source_bounds is not None
            self.assertLess(source_bounds[1], sink_placement[1])
