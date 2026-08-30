from __future__ import annotations

import dataclasses
import itertools
from typing import Literal
from unittest import mock

import sympy
import torch

import helion
from helion._compiler.cross_loop_scheduler import FinalArrivalContinuation
from helion._compiler.cross_loop_scheduler import ReadinessConsumer
from helion._compiler.cross_loop_scheduler import ReadinessCounterPlan
from helion._compiler.cross_loop_scheduler import ReadinessEvent
from helion._compiler.cross_loop_scheduler import ReadinessGraph
from helion._compiler.cross_loop_scheduler import ReadinessProducer
from helion._compiler.cross_loop_scheduler import WorkerSchedule
from helion._compiler.cross_loop_scheduler import WorkerScheduleSegment
from helion._compiler.cross_loop_scheduler import _select_root_barrier_edges
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
import helion.language as hl


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
    source_coordinates = segment.task_order.source_domain.coordinates(task_order_index)
    targets = segment.task_order.target_coordinates(source_coordinates)
    if len(targets) != 1:
        raise AssertionError("task-order index does not map to one logical task")
    return segment.task_order.target_domain.index(
        dict(
            zip(
                segment.task_order.target_domain.axis_order,
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
                    (axis, 0, source.axis_counts[axis], 1) for axis in source.axis_order
                ),
                target_coordinates,
            ),
        ),
    )


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
    layout_is_static: bool = True,
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
        layout_is_static=layout_is_static,
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
        continuations = derive_final_arrival_continuations(configured, baseline)
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
        baseline = _build_baseline_worker_schedule(
            domains,
            readiness_graph.root_task_orders,
            worker_count=4,
        )

        continuations = derive_final_arrival_continuations(readiness_graph, baseline)

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
        self.assertEqual(derive_final_arrival_continuations(overlapping, baseline), ())

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
        baseline = _build_baseline_worker_schedule(
            readiness_graph.root_domains,
            readiness_graph.root_task_orders,
            worker_count=4,
        )

        publication = readiness_producer.keys_by_producer
        self.assertIsNotNone(publication)
        assert publication is not None
        self.assertIsNone(publication.canonical_single_valued())
        self.assertEqual(
            derive_final_arrival_continuations(readiness_graph, baseline), ()
        )
        self.assertEqual(choose_readiness_counters(readiness_graph, ()), ())

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
                layout_is_static=False,
            ),
            _access(
                root=2,
                allocation_id=1,
                kind="load",
                block_ids=(30,),
                layout_is_static=False,
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

        continuations = choose_final_arrival_continuations(readiness_graph, baseline)

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
        continuations = derive_final_arrival_continuations(
            configured,
            _build_baseline_worker_schedule(
                configured.root_domains,
                configured.root_task_orders,
                worker_count=4,
            ),
        )
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
            _access(root=0, kind="store", block_ids=(10,), layout_is_static=False),
            _access(root=1, allocation_id=1, kind="store", block_ids=(20,)),
            _access(root=2, kind="load", block_ids=(30,), layout_is_static=False),
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
        baseline = _build_baseline_worker_schedule(
            configured.root_domains,
            configured.root_task_orders,
            worker_count=4,
        )
        self.assertEqual(derive_final_arrival_continuations(configured, baseline), ())

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
        baseline = _build_baseline_worker_schedule(
            root_domains,
            readiness_graph.root_task_orders,
            worker_count=4,
        )
        self.assertEqual(
            derive_final_arrival_continuations(readiness_graph, baseline), ()
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
        continuations = derive_final_arrival_continuations(readiness_graph, baseline)

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
        baseline = _baseline_worker_schedule(
            root_domains,
            worker_count=4,
        )
        continuations = derive_final_arrival_continuations(readiness_graph, baseline)

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

    def test_nonstatic_layout_falls_back_to_root_readiness(self) -> None:
        plan = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", block_ids=(10,), layout_is_static=False),
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
        self.assertEqual(placement(schedule.worker_schedule, 2, 0), (5, 1))
        self.assertEqual(placement(schedule.worker_schedule, 0, 6), (0, 1))

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
        self.assertEqual(placement(overlapped.worker_schedule, 2, 0), (5, 1))
        self.assertEqual(placement(overlapped.worker_schedule, 5, 0), (5, 5))
