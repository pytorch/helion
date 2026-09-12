from __future__ import annotations

import contextlib
import copy
import dataclasses
import itertools
import pickle
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast
from unittest import mock

import sympy
import torch

import helion
from helion import exc
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
from helion._compiler.cross_loop_scheduler import _has_valid_source_ticket_schedule
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
    build_readiness_graph as _build_readiness_graph,
)
from helion._compiler.cross_loop_scheduler import (
    build_static_pipeline_plan as _build_static_pipeline_plan,
)
from helion._compiler.cross_loop_scheduler import choose_final_arrival_continuations
from helion._compiler.cross_loop_scheduler import choose_readiness_counters
from helion._compiler.cross_loop_scheduler import derive_final_arrival_continuations
from helion._compiler.tile_dependency import CoordinateDomain
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import DependencyObligation
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
    obligations_by_root_pair: tuple[
        tuple[tuple[int, int], frozenset[DependencyObligation]], ...
    ]
    | None = None,
) -> ReadinessGraph:
    return ReadinessGraph(
        root_task_orders=_default_root_task_orders(root_domains),
        events=events,
        obligations_by_root_pair=obligations_by_root_pair,
    )


def _pointwise_root_readiness_event(
    root_domains: tuple[CoordinateDomain, ...],
    producer_root: int,
    consumer_root: int,
    event_id: int,
) -> ReadinessEvent:
    """Build one same-index root-level event for small quotient tests."""
    producer_domain = root_domains[producer_root]
    consumer_domain = root_domains[consumer_root]
    if (
        len(producer_domain.axis_order) != 1
        or len(consumer_domain.axis_order) != 1
        or producer_domain.size_expr != consumer_domain.size_expr
    ):
        raise ValueError("pointwise test readiness requires equal 1-D roots")
    event_domain = CoordinateDomain(
        axis_order=(0,),
        axis_counts_items=((0, producer_domain.size_expr),),
        kind="event",
        identity=event_id,
        _allow_empty=producer_domain.size_expr.is_zero is not False,
    )
    consumer_axis = consumer_domain.axis_order[0]
    producer = ReadinessProducer(
        producer_root=producer_root,
        producers_by_key=_full_point_map(
            event_domain,
            producer_domain,
            coordinate_axis_symbol(0),
        ),
    )
    consumer = ReadinessConsumer(
        consumer_root=consumer_root,
        keys_by_consumer=_full_point_map(
            consumer_domain,
            event_domain,
            coordinate_axis_symbol(consumer_axis),
        ),
    )
    return ReadinessEvent((producer,), (consumer,))


def _whole_consumer_subset_readiness_event(
    root_domains: tuple[CoordinateDomain, ...],
    producer_root: int,
    consumer_root: int,
    event_id: int,
) -> ReadinessEvent:
    """Build one lowerable event whose consumer is one complete cohort."""
    producer_domain = root_domains[producer_root]
    consumer_domain = root_domains[consumer_root]
    if (
        producer_domain.size_expr.free_symbols
        or producer_domain.size_expr.is_integer is not True
        or int(producer_domain.size_expr) <= 1
    ):
        raise ValueError("subset-producer test event needs at least two tasks")
    (producer_axis,) = producer_domain.axis_order
    event_domain = _domain((0, 1), kind="event", identity=event_id)
    producers_by_key = CoordinateRelation(
        event_domain,
        producer_domain,
        (
            _CoordinateRelationPiece(
                ((0, 0, 1, 1),),
                ((producer_axis, sympy.Integer(0), sympy.Integer(1), 1),),
            ),
        ),
    )
    return ReadinessEvent(
        producers=(
            ReadinessProducer(
                producer_root=producer_root,
                producers_by_key=producers_by_key,
            ),
        ),
        consumers=(
            ReadinessConsumer(
                consumer_root=consumer_root,
                keys_by_consumer=_full_point_map(
                    consumer_domain,
                    event_domain,
                    sympy.Integer(0),
                ),
            ),
        ),
    )


def _whole_root_readiness_event(
    root_domains: tuple[CoordinateDomain, ...],
    producer_root: int,
    consumer_root: int,
    event_id: int,
) -> ReadinessEvent:
    """Build one event released only after the complete producer root."""
    event_domain = _domain((0, 1), kind="event", identity=event_id)
    return ReadinessEvent(
        producers=(
            _readiness_producer_from_publication(
                producer_root,
                _full_point_map(
                    root_domains[producer_root],
                    event_domain,
                    sympy.Integer(0),
                ),
            ),
        ),
        consumers=(
            ReadinessConsumer(
                consumer_root=consumer_root,
                keys_by_consumer=_full_point_map(
                    root_domains[consumer_root],
                    event_domain,
                    sympy.Integer(0),
                ),
            ),
        ),
    )


def _whole_consumer_join_readiness_event(
    root_domains: tuple[CoordinateDomain, ...],
    producer_roots: tuple[int, ...],
    consumer_root: int,
    event_id: int,
) -> ReadinessEvent:
    """Build a one-cohort consumer released by complete producer roots."""
    if len(producer_roots) < 2:
        raise ValueError("join test event needs at least two producer roots")
    event_domain = _domain((0, 1), kind="event", identity=event_id)
    return ReadinessEvent(
        producers=tuple(
            _readiness_producer_from_publication(
                producer_root,
                _full_point_map(
                    root_domains[producer_root],
                    event_domain,
                    sympy.Integer(0),
                ),
            )
            for producer_root in producer_roots
        ),
        consumers=(
            ReadinessConsumer(
                consumer_root=consumer_root,
                keys_by_consumer=_full_point_map(
                    root_domains[consumer_root],
                    event_domain,
                    sympy.Integer(0),
                ),
            ),
        ),
    )


def _source_ticket_inlet_problem(
    source_count: int,
) -> tuple[ReadinessGraph, tuple[ReadinessCounterPlan, ...]]:
    """Build two source-ticket inlets followed by symmetric sinks."""
    if source_count < 3:
        raise ValueError("source count must leave a nonempty late inlet")
    source, early, late, early_sink, late_sink = _identify_root_domains(
        (
            _domain((10, source_count, 1)),
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
                            ((10, sympy.Integer(2), sympy.Integer(source_count), 1),),
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

    events = (
        source_event,
        sink_event(1, early, 3, early_sink, 1),
        sink_event(2, late, 4, late_sink, 2),
    )
    graph = _readiness_graph(
        (source, early, late, early_sink, late_sink),
        *events,
    )
    return graph, tuple(
        ReadinessCounterPlan(event.producers, event.consumers) for event in events
    )


def _partial_release_chain_problem(
    *,
    consumer_cohort_width: int,
    consumer_cohort_count: int = 2,
    leading_task_count: int = 5,
) -> tuple[ReadinessGraph, WorkerSchedule, tuple[ReadinessCounterPlan, ...]]:
    """Build a generic chain whose first consumer cohort reaches a tail hole."""
    if (
        consumer_cohort_width <= 0
        or consumer_cohort_count <= 0
        or leading_task_count <= 0
    ):
        raise ValueError("cohort geometry must be positive")
    root_domains = _identify_root_domains(
        (
            _domain((10, leading_task_count, 1)),
            _domain((20, 4, 1)),
            _domain((30, 4, 1)),
            _domain((40, consumer_cohort_count * consumer_cohort_width, 1)),
            _domain((50, 4, 1)),
        )
    )

    def subset_event(
        producer_root: int,
        consumer_begin: int,
        consumer_end: int,
        event_id: int,
    ) -> ReadinessEvent:
        key_domain = _domain((0, 1), kind="event", identity=event_id)
        producer_domain = root_domains[producer_root]
        consumer_domain = root_domains[2]
        return ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root,
                    CoordinateRelation(
                        key_domain,
                        producer_domain,
                        (
                            _CoordinateRelationPiece(
                                ((0, 0, 1, 1),),
                                (
                                    (
                                        producer_domain.axis_order[0],
                                        sympy.Integer(0),
                                        producer_domain.size_expr,
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
                    2,
                    CoordinateRelation(
                        consumer_domain,
                        key_domain,
                        (
                            _CoordinateRelationPiece(
                                (
                                    (
                                        consumer_domain.axis_order[0],
                                        consumer_begin,
                                        consumer_end,
                                        1,
                                    ),
                                ),
                                ((0, 0, 1, 1),),
                            ),
                        ),
                    ),
                ),
            ),
        )

    first_release = subset_event(0, 0, 2, 0)
    second_release = subset_event(1, 2, 4, 1)
    key_domain = _domain((0, 2), kind="event", identity=2)
    key = coordinate_axis_symbol(0)
    consumer_task = coordinate_axis_symbol(40)
    downstream = ReadinessEvent(
        producers=(
            ReadinessProducer(
                2,
                CoordinateRelation(
                    key_domain,
                    root_domains[2],
                    (
                        _CoordinateRelationPiece(
                            ((0, 0, 2, 1),),
                            ((30, 2 * key, 2 * key + 2, 1),),
                        ),
                    ),
                ),
            ),
        ),
        consumers=(
            ReadinessConsumer(
                3,
                _full_point_map(
                    root_domains[3],
                    key_domain,
                    sympy.floor(consumer_task / consumer_cohort_width),
                ),
            ),
        ),
    )
    graph = _readiness_graph(
        root_domains,
        first_release,
        second_release,
        downstream,
    )
    plans = tuple(
        ReadinessCounterPlan(event.producers, event.consumers) for event in graph.events
    )
    return (
        graph,
        _baseline_worker_schedule(graph.root_domains, worker_count=4),
        plans,
    )


def _retirement_frontier_chain_problem(
    producer_count: int,
    *,
    worker_count: int = 8,
    fan_in: int = 2,
    nested_consumer_count: int = 0,
) -> tuple[ReadinessGraph, WorkerSchedule, tuple[ReadinessCounterPlan, ...]]:
    """Build a fixed-width fan-in chain for retirement-window tests."""
    if (
        worker_count <= 0
        or fan_in <= 0
        or producer_count <= 0
        or producer_count % fan_in
    ):
        raise ValueError("producer geometry must contain exact full fan-in groups")
    intermediate_count = producer_count // fan_in
    root_domains = _identify_root_domains(
        (
            _domain((10, producer_count, 1)),
            _domain((20, intermediate_count, 1)),
            *(
                (_domain((30, nested_consumer_count, 1)),)
                if nested_consumer_count
                else ()
            ),
        )
    )
    key_domain = _domain((0, intermediate_count), kind="event", identity=0)
    key = coordinate_axis_symbol(0)
    producer_event = ReadinessEvent(
        producers=(
            ReadinessProducer(
                producer_root=0,
                producers_by_key=CoordinateRelation(
                    key_domain,
                    root_domains[0],
                    (
                        _CoordinateRelationPiece(
                            ((0, 0, intermediate_count, 1),),
                            ((10, fan_in * key, fan_in * key + fan_in, 1),),
                        ),
                    ),
                ),
            ),
        ),
        consumers=(
            ReadinessConsumer(
                consumer_root=1,
                keys_by_consumer=_full_point_map(
                    root_domains[1],
                    key_domain,
                    coordinate_axis_symbol(20),
                ),
            ),
        ),
    )
    events = [producer_event]
    if nested_consumer_count:
        nested_key_domain = _domain(
            (0, intermediate_count),
            kind="event",
            identity=1,
        )
        nested_domain = _domain(
            (30, nested_consumer_count, 1),
            (31, intermediate_count, 1),
            identity=71,
        )
        nested_event = ReadinessEvent(
            producers=(
                _readiness_producer_from_publication(
                    1,
                    _full_point_map(
                        root_domains[1],
                        nested_key_domain,
                        coordinate_axis_symbol(20),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=2,
                    consumer_site_id=71,
                    keys_by_consumer=_full_point_map(
                        nested_domain,
                        nested_key_domain,
                        coordinate_axis_symbol(31),
                    ),
                ),
            ),
        )
        events.append(nested_event)
    graph = _readiness_graph(root_domains, *events)
    plans = tuple(
        ReadinessCounterPlan(event.producers, event.consumers) for event in events
    )
    return (
        graph,
        _baseline_worker_schedule(graph.root_domains, worker_count=worker_count),
        plans,
    )


def _retirement_backfill_problem(
    *,
    descendant_blocks_tail: bool,
) -> tuple[ReadinessGraph, WorkerSchedule, tuple[ReadinessCounterPlan, ...]]:
    """Build a rotated retirement window with an independent ready branch."""
    root_domains = _identify_root_domains(
        (
            _domain((10, 4, 1)),
            _domain((20, 10, 1)),
            _domain((30, 2, 1)),
            _domain((40, 2 if descendant_blocks_tail else 3, 1)),
            _domain((50, 1 if descendant_blocks_tail else 2, 1)),
        )
    )
    prefix = _whole_root_readiness_event(root_domains, 0, 1, 0)
    anchor_key_domain = _domain((0, 2), kind="event", identity=1)
    anchor_release = ReadinessEvent(
        producers=(
            ReadinessProducer(
                producer_root=1,
                producers_by_key=CoordinateRelation(
                    anchor_key_domain,
                    root_domains[1],
                    (
                        _CoordinateRelationPiece(
                            ((0, 0, 1, 1),),
                            ((20, 0, 8, 1),),
                        ),
                        _CoordinateRelationPiece(
                            ((0, 1, 2, 1),),
                            ((20, 8, 10, 1),),
                        ),
                    ),
                ),
            ),
        ),
        consumers=(
            ReadinessConsumer(
                consumer_root=2,
                keys_by_consumer=_full_point_map(
                    root_domains[2],
                    anchor_key_domain,
                    coordinate_axis_symbol(30),
                ),
            ),
        ),
    )
    branch_key_domain = _domain((0, 1), kind="event", identity=2)
    branch_producer_root = 2 if descendant_blocks_tail else 3
    branch_consumer_root = 3 if descendant_blocks_tail else 4
    producer_axis = root_domains[branch_producer_root].axis_order[0]
    branch_release = ReadinessEvent(
        producers=(
            ReadinessProducer(
                producer_root=branch_producer_root,
                producers_by_key=CoordinateRelation(
                    branch_key_domain,
                    root_domains[branch_producer_root],
                    (
                        _CoordinateRelationPiece(
                            ((0, 0, 1, 1),),
                            (
                                (
                                    producer_axis,
                                    0,
                                    (
                                        1
                                        if descendant_blocks_tail
                                        else root_domains[
                                            branch_producer_root
                                        ].axis_counts[producer_axis]
                                    ),
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
                consumer_root=branch_consumer_root,
                keys_by_consumer=_full_point_map(
                    root_domains[branch_consumer_root],
                    branch_key_domain,
                    sympy.Integer(0),
                ),
            ),
        ),
    )
    graph = _readiness_graph(
        root_domains,
        prefix,
        anchor_release,
        branch_release,
    )
    plans = tuple(
        ReadinessCounterPlan(event.producers, event.consumers) for event in graph.events
    )
    return (
        graph,
        _baseline_worker_schedule(graph.root_domains, worker_count=8),
        plans,
    )


def _qwen_shaped_retirement_problem() -> tuple[
    ReadinessGraph,
    WorkerSchedule,
    tuple[ReadinessCounterPlan, ...],
    frozenset[tuple[int, int]],
]:
    """Build the concrete Qwen attention-tail geometry without model code."""
    root_domains = _identify_root_domains(
        (
            _domain((10, 2930, 1)),
            _domain((20, 1, 1), (21, 1536, 16)),
            _domain((30, 1, 1), (31, 96, 1)),
            _domain((40, 1, 1), (41, 512, 1)),
        )
    )
    producer_order_domain = _domain(
        (50, 1),
        (51, 16),
        (52, 96),
        kind="task_order",
        identity=root_domains[1].identity,
    )
    order_batch = coordinate_axis_symbol(50)
    order_inner = coordinate_axis_symbol(51)
    order_iteration = coordinate_axis_symbol(52)
    producer_order = CoordinateRelation.point_map(
        producer_order_domain,
        root_domains[1],
        (
            (
                ((50, 0, 1, 1), (51, 0, 8, 1), (52, 0, 96, 1)),
                (
                    order_batch,
                    8 * order_iteration + order_inner,
                ),
            ),
            (
                ((50, 0, 1, 1), (51, 8, 16, 1), (52, 0, 96, 1)),
                (
                    order_batch,
                    8 * order_iteration + order_inner + 760,
                ),
            ),
        ),
    )
    reduction_keys = _domain((0, 1), (1, 96), kind="event", identity=0)
    key_batch = coordinate_axis_symbol(0)
    key_iteration = coordinate_axis_symbol(1)
    producer_event = ReadinessEvent(
        producers=(
            ReadinessProducer(
                producer_root=1,
                producers_by_key=CoordinateRelation(
                    reduction_keys,
                    root_domains[1],
                    (
                        _CoordinateRelationPiece(
                            ((0, 0, 1, 1), (1, 0, 96, 1)),
                            (
                                (20, key_batch, key_batch + 1, 1),
                                (
                                    21,
                                    8 * key_iteration,
                                    8 * key_iteration + 8,
                                    1,
                                ),
                            ),
                        ),
                        _CoordinateRelationPiece(
                            ((0, 0, 1, 1), (1, 0, 96, 1)),
                            (
                                (20, key_batch, key_batch + 1, 1),
                                (
                                    21,
                                    8 * key_iteration + 768,
                                    8 * key_iteration + 776,
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
                consumer_root=2,
                keys_by_consumer=_full_point_map(
                    root_domains[2],
                    reduction_keys,
                    coordinate_axis_symbol(30),
                    coordinate_axis_symbol(31),
                ),
            ),
        ),
    )
    nested_keys = _domain((0, 1), (1, 96), kind="event", identity=1)
    nested_domain = _domain(
        (40, 1, 1),
        (41, 512, 1),
        (42, 96, 1),
        identity=71,
    )
    nested_event = ReadinessEvent(
        producers=(
            _readiness_producer_from_publication(
                2,
                _full_point_map(
                    root_domains[2],
                    nested_keys,
                    coordinate_axis_symbol(30),
                    coordinate_axis_symbol(31),
                ),
            ),
        ),
        consumers=(
            ReadinessConsumer(
                consumer_root=3,
                consumer_site_id=71,
                keys_by_consumer=_full_point_map(
                    nested_domain,
                    nested_keys,
                    coordinate_axis_symbol(40),
                    coordinate_axis_symbol(42),
                ),
            ),
        ),
    )
    graph = ReadinessGraph(
        root_task_orders=(
            pid_task_order(root_domains[0], root_domains[0].axis_order),
            producer_order,
            pid_task_order(root_domains[2], root_domains[2].axis_order),
            pid_task_order(root_domains[3], root_domains[3].axis_order),
        ),
        events=(producer_event, nested_event),
    )
    plans = tuple(
        ReadinessCounterPlan(event.producers, event.consumers) for event in graph.events
    )
    return (
        graph,
        _build_baseline_worker_schedule(
            graph.root_domains,
            graph.root_task_orders,
            worker_count=1184,
        ),
        plans,
        frozenset(((0, 1),)),
    )


def _conservative_static_arm_problem() -> tuple[
    ReadinessGraph,
    WorkerSchedule,
    tuple[ReadinessCounterPlan, ...],
]:
    """Build one unrenderable frontier beside an independent exact chain."""
    root_domains = _identify_root_domains(
        (
            _domain((10, 8, 1)),
            _domain((20, 2, 1)),
            _domain((30, 2, 1)),
            _domain((40, 2, 1), (41, 2, 1)),
        )
    )
    key_domain = _domain((0, 2), (1, 2), kind="event", identity=0)
    key_outer = coordinate_axis_symbol(0)
    key_inner = coordinate_axis_symbol(1)
    producer = ReadinessProducer(
        producer_root=0,
        producers_by_key=CoordinateRelation(
            key_domain,
            root_domains[0],
            (
                _CoordinateRelationPiece(
                    ((0, 0, 2, 1), (1, 0, 2, 1)),
                    (
                        (
                            10,
                            4 * key_inner + 2 * key_outer,
                            4 * key_inner + 2 * key_outer + 2,
                            1,
                        ),
                    ),
                ),
            ),
        ),
    )
    affected_event = ReadinessEvent(
        producers=(producer,),
        consumers=(
            ReadinessConsumer(
                consumer_root=3,
                keys_by_consumer=_full_point_map(
                    root_domains[3],
                    key_domain,
                    coordinate_axis_symbol(40),
                    coordinate_axis_symbol(41),
                ),
            ),
        ),
    )
    exact_event = _pointwise_root_readiness_event(
        root_domains,
        producer_root=1,
        consumer_root=2,
        event_id=1,
    )
    root_task_orders = _default_root_task_orders(root_domains)
    graph = ReadinessGraph(
        root_task_orders=root_task_orders,
        events=(affected_event, exact_event),
    )
    plans = tuple(
        ReadinessCounterPlan(event.producers, event.consumers) for event in graph.events
    )
    return (
        graph,
        _baseline_worker_schedule(
            root_domains,
            worker_count=12,
            root_task_orders=root_task_orders,
        ),
        plans,
    )


def _branching_continuation_problem() -> tuple[
    ReadinessGraph,
    WorkerSchedule,
    tuple[ReadinessCounterPlan, ...],
]:
    """Build a reconvergent continuation diamond with two static leaves."""
    root_domains = _identify_root_domains(
        tuple(_domain((10 + 10 * root, 1, 1)) for root in range(8))
    )
    event_specs = (
        ((0, 1), 2),
        ((0, 1), 3),
        ((2, 3), 4),
        ((2, 3), 5),
        ((4, 5), 6),
        ((6,), 7),
    )
    events = tuple(
        (
            _whole_consumer_join_readiness_event(
                root_domains,
                producer_roots,
                consumer_root,
                event_id,
            )
            if len(producer_roots) > 1
            else _whole_root_readiness_event(
                root_domains,
                producer_roots[0],
                consumer_root,
                event_id,
            )
        )
        for event_id, (producer_roots, consumer_root) in enumerate(event_specs)
    )
    graph = _readiness_graph(root_domains, *events)
    plans = tuple(
        ReadinessCounterPlan(
            event.producers,
            event.consumers,
            continuation_consumer_index=(0 if event_id < len(events) - 1 else None),
        )
        for event_id, event in enumerate(events)
    )
    prepared = _schedule(
        2,
        *(
            _segment(
                root,
                graph.root_task_orders[root],
                workers=(0, 2),
                dispatch_offset=slot,
            )
            for slot, root in enumerate((0, 1, 7))
        ),
    )
    return graph, prepared, plans


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
    return _build_readiness_graph(
        graph,
        root_task_orders=_default_root_task_orders(root_domains),
        site_domains=site_domains,
        publishable_site_ids=publishable_site_ids,
    ).events


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

    def test_source_ticket_has_launch_stage_zero_relation(
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
        source_schedule = cross_loop_scheduler._with_source_ticket_schedule_segment(
            baseline,
            readiness_graph.root_task_orders,
            0,
        )
        self.assertIsNotNone(source_schedule)
        assert source_schedule is not None
        scheduled = _global_unit_list_schedule(
            readiness_graph,
            source_schedule,
            readiness_counters,
            frozenset(),
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
                )
            )
            self.assertTrue(
                _has_valid_source_ticket_schedule(
                    scheduled,
                    readiness_graph.root_task_orders,
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

    def test_source_ticket_inlets_precede_downstream_waits(self) -> None:
        graph, counters = _source_ticket_inlet_problem(4)
        source_schedule = cross_loop_scheduler._with_source_ticket_schedule_segment(
            _baseline_worker_schedule(graph.root_domains, worker_count=2),
            graph.root_task_orders,
            0,
        )
        self.assertIsNotNone(source_schedule)
        assert source_schedule is not None

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                source_schedule,
                counters,
                frozenset(),
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

    def test_progress_rejects_a_second_launch_stage_source(self) -> None:
        graph, counters = _source_ticket_inlet_problem(4)
        schedule = cross_loop_scheduler._with_source_ticket_schedule_segment(
            _baseline_worker_schedule(graph.root_domains, worker_count=2),
            graph.root_task_orders,
            0,
        )
        self.assertIsNotNone(schedule)
        assert schedule is not None
        bogus_source_segment = cross_loop_scheduler._normalize_dense_schedule_segment(
            WorkerScheduleSegment(
                root=1,
                task_order=graph.root_task_orders[1],
                worker_begin=0,
                worker_count=2,
                dispatch_offset=4,
            ),
            schedule.placement_domain,
            launch_stage=0,
        )
        invalid = WorkerSchedule(
            schedule.worker_count,
            (
                *schedule.without_roots(frozenset((1,))).segments,
                bogus_source_segment,
            ),
        )

        with _forbid_schedule_enumeration():
            self.assertTrue(
                _validate_worker_schedule_tasks(
                    invalid,
                    graph.root_task_orders,
                )
            )
            self.assertFalse(
                _has_valid_source_ticket_schedule(
                    invalid,
                    graph.root_task_orders,
                    counters,
                    frozenset(),
                )
            )
            self.assertFalse(
                cross_loop_scheduler._schedule_is_progress_safe(
                    invalid,
                    graph,
                    counters,
                    frozenset(),
                )
            )

    def test_event_frontier_handles_trailing_source_ticket_waves(self) -> None:
        graph, counters = _source_ticket_inlet_problem(9)
        dispatch_offset = 0
        resident_segments: list[WorkerScheduleSegment] = []
        # Put the late inlet first in C. The source frontier should prove that
        # the early inlet is the strict first candidate rather than silently
        # returning C when launch-stage-zero padding extends the wave domain.
        for root in (2, 1, 4, 3):
            resident_segments.append(
                _segment(
                    root,
                    graph.root_task_orders[root],
                    workers=(0, 2),
                    dispatch_offset=dispatch_offset,
                )
            )
            dispatch_offset += graph.root_domains[root].size
        resident_schedule = _schedule(2, *resident_segments)
        source_schedule = cross_loop_scheduler._with_source_ticket_schedule_segment(
            resident_schedule,
            graph.root_task_orders,
            0,
        )
        self.assertIsNotNone(source_schedule)
        assert source_schedule is not None
        self.assertGreater(
            source_schedule.worker_step_domain.size,
            resident_schedule.worker_step_domain.size,
        )
        self.assertEqual(task_at(source_schedule, 0, 0), (2, 0))

        with _forbid_schedule_enumeration():
            scheduled = cross_loop_scheduler._event_frontier_list_schedule(
                graph,
                source_schedule,
                counters,
                frozenset(),
                pipeline_depth=2,
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        self.assertIsNot(scheduled, source_schedule)
        self.assertEqual(task_at(scheduled, 0, 0), (1, 0))
        self.assertEqual(
            scheduled.segments_for_root(0),
            source_schedule.segments_for_root(0),
        )
        with _forbid_schedule_enumeration():
            self.assertTrue(
                cross_loop_scheduler._schedule_is_progress_safe(
                    scheduled,
                    graph,
                    counters,
                    frozenset(),
                )
            )
            self.assertTrue(
                _has_valid_source_ticket_schedule(
                    scheduled,
                    graph.root_task_orders,
                    counters,
                    frozenset(),
                )
            )

    def test_source_ticket_proof_is_independent_of_wave_remainder(
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
                source_schedule = (
                    cross_loop_scheduler._with_source_ticket_schedule_segment(
                        _baseline_worker_schedule(
                            readiness_graph.root_domains,
                            worker_count=2,
                        ),
                        readiness_graph.root_task_orders,
                        0,
                    )
                )
                self.assertIsNotNone(source_schedule)
                assert source_schedule is not None
                with _forbid_schedule_enumeration():
                    scheduled = _global_unit_list_schedule(
                        readiness_graph,
                        source_schedule,
                        readiness_counters,
                        frozenset(),
                    )

                    self.assertIsNotNone(scheduled)
                    assert scheduled is not None
                    source_segment = (
                        cross_loop_scheduler._source_ticket_schedule_segment(scheduled)
                    )
                    self.assertIsNotNone(source_segment)
                    assert source_segment is not None
                    self.assertIsNotNone(
                        cross_loop_scheduler._source_segment_ticket_order(
                            source_segment,
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
                        )
                    )
                self.assertEqual(len(scheduled.segments_for_root(0)), 1)
                self.assertEqual(placement(scheduled, 1, 0), (0, 0))

    def test_source_ticket_selection_requires_oversubscription(self) -> None:
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
                    cross_loop_scheduler._source_ticket_candidate(
                        readiness_graph,
                        readiness_counters,
                        frozenset(),
                        worker_count=2,
                    ),
                    expected,
                )

    def test_invalid_source_ticket_candidate_is_rejected_before_freeze(self) -> None:
        root_domains = _identify_root_domains(
            (_domain((10, 3, 1)), _domain((20, 2, 1)))
        )
        dependency_graph = _dependency_graph([[10], [20]])

        def progress_safe(
            worker_schedule: WorkerSchedule,
            *_args: object,
            **_kwargs: object,
        ) -> bool:
            return (
                cross_loop_scheduler._source_ticket_schedule_segment(worker_schedule)
                is None
            )

        with (
            mock.patch.object(
                cross_loop_scheduler,
                "_source_ticket_candidate",
                return_value=0,
            ),
            mock.patch.object(
                cross_loop_scheduler,
                "_has_valid_source_ticket_schedule",
                return_value=True,
            ),
            mock.patch.object(
                cross_loop_scheduler,
                "_schedule_is_progress_safe",
                side_effect=progress_safe,
            ),
        ):
            plan = _configured_static_pipeline_plan(
                dependency_graph=dependency_graph,
                root_domains=root_domains,
                axis_geometry={10: (3, 1), 20: (2, 1)},
                worker_count=2,
                supports_source_ticket_launch=True,
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertIsNone(
            cross_loop_scheduler._source_ticket_schedule_segment(plan.worker_schedule)
        )
        self.assertEqual(
            tuple(segment.root for segment in plan.worker_schedule.segments),
            (0, 1),
        )
        self.assertTrue(
            all(segment.launch_stage == 1 for segment in plan.worker_schedule.segments)
        )

    def test_unrepresentable_source_frontier_keeps_configured_traversals(self) -> None:
        root_domains = _identify_root_domains(
            (_domain((10, 3, 1)), _domain((20, 2, 1)))
        )
        dependency_graph = _dependency_graph([[10], [20]])
        graph = _readiness_graph(root_domains)

        with (
            mock.patch.object(
                cross_loop_scheduler,
                "_source_ticket_candidate",
                return_value=0,
            ),
            mock.patch.object(
                cross_loop_scheduler,
                "_has_valid_source_ticket_schedule",
                return_value=True,
            ),
            mock.patch.object(
                cross_loop_scheduler,
                "_source_ticket_frontiers",
                return_value=None,
            ),
        ):
            plan = _configured_static_pipeline_plan(
                dependency_graph=dependency_graph,
                root_domains=root_domains,
                axis_geometry={10: (3, 1), 20: (2, 1)},
                worker_count=2,
                supports_source_ticket_launch=True,
                cross_loop_pipeline_depth=2,
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        source_segment = cross_loop_scheduler._source_ticket_schedule_segment(
            plan.worker_schedule
        )
        self.assertIsNotNone(source_segment)
        assert source_segment is not None
        self.assertEqual(source_segment.root, 0)
        (resident_segment,) = plan.worker_schedule.segments_for_root(1)
        resident_order = resident_segment.logical_task_order
        self.assertIsNotNone(resident_order)
        assert resident_order is not None
        self.assertEqual(
            resident_order.materialize(),
            graph.root_task_orders[1].materialize(),
        )

    def test_source_ticket_signal_declines_full_frontiers(self) -> None:
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
        schedule = cross_loop_scheduler._with_source_ticket_schedule_segment(
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
                cross_loop_scheduler._source_ticket_candidate(
                    readiness_graph,
                    (full_plan,),
                    frozenset(),
                    worker_count=2,
                )
            )
            self.assertTrue(
                _has_valid_source_ticket_schedule(
                    schedule,
                    readiness_graph.root_task_orders,
                    (full_plan,),
                    frozenset(),
                )
            )
            self.assertTrue(
                _has_valid_source_ticket_schedule(
                    schedule,
                    readiness_graph.root_task_orders,
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
                cross_loop_scheduler._source_ticket_candidate(
                    readiness_graph,
                    (partial_plan,),
                    frozenset(((0, 1),)),
                    worker_count=2,
                )
            )
            self.assertTrue(
                _has_valid_source_ticket_schedule(
                    schedule,
                    readiness_graph.root_task_orders,
                    (partial_plan,),
                    frozenset(((0, 1),)),
                )
            )

    def test_source_ticket_combines_producer_arms_before_subset_proof(
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
                cross_loop_scheduler._source_ticket_candidate(
                    readiness_graph,
                    (plan,),
                    frozenset(),
                    worker_count=2,
                )
            )

    def test_global_list_schedule_does_not_displace_unfinished_ancestor(
        self,
    ) -> None:
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
        depth_one = _global_unit_list_schedule(
            readiness_graph,
            baseline,
            readiness_counters,
            frozenset(),
            pipeline_depth=1,
        )
        # This fixture's configured traversal is already its prepared C.
        self.assertIs(depth_one, baseline)

        scheduled = _global_unit_list_schedule(
            readiness_graph,
            baseline,
            readiness_counters,
            frozenset(),
            pipeline_depth=2,
        )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        # The first consumer is critical once its key closes, but the producer
        # has no terminal-wave hole. It must not displace unfinished ancestor
        # work merely to start the critical path early.
        for root, task_count in enumerate((4, 2, 1)):
            for task in range(task_count):
                self.assertEqual(
                    placement(scheduled, root, task),
                    placement(baseline, root, task),
                )
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

    def test_global_list_schedule_does_not_split_cohort_into_terminal_hole(
        self,
    ) -> None:
        graph, baseline, plans = _partial_release_chain_problem(
            consumer_cohort_width=2,
        )

        scheduled = _global_unit_list_schedule(
            graph,
            baseline,
            plans,
            frozenset(),
            pipeline_depth=3,
        )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        terminal_wave = placement(scheduled, 0, 4)[1]
        # The first producer cohort consumes two of the root's three terminal
        # holes and releases a two-task consumer cohort.  The one remaining
        # lane cannot justify splitting that dependent cohort while the next
        # producer cohort is still unassigned.
        self.assertEqual(
            {placement(scheduled, 2, task)[1] for task in (0, 1)},
            {terminal_wave},
        )
        self.assertTrue(
            all(placement(scheduled, 2, task)[1] > terminal_wave for task in (2, 3))
        )
        self.assertTrue(
            all(placement(scheduled, 3, task)[1] > terminal_wave for task in (0, 1))
        )
        # The producer arm that will release the next upstream cohort may use
        # the lane; preserving atomicity does not require leaving it idle.
        self.assertTrue(
            any(placement(scheduled, 1, task)[1] == terminal_wave for task in range(4))
        )
        validate_worker_schedule(graph, scheduled)

    def test_global_list_schedule_admits_complete_cohort_into_terminal_hole(
        self,
    ) -> None:
        graph, baseline, plans = _partial_release_chain_problem(
            consumer_cohort_width=1,
        )

        scheduled = _global_unit_list_schedule(
            graph,
            baseline,
            plans,
            frozenset(),
            pipeline_depth=3,
        )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        terminal_wave = placement(scheduled, 0, 4)[1]
        producer_workers = {placement(scheduled, 2, task)[0] for task in (0, 1)}
        consumer_placement = placement(scheduled, 3, 0)
        self.assertEqual(
            {placement(scheduled, 2, task)[1] for task in (0, 1)},
            {terminal_wave},
        )
        self.assertEqual(consumer_placement[1], terminal_wave)
        self.assertNotIn(consumer_placement[0], producer_workers)
        self.assertTrue(
            all(placement(scheduled, 2, task)[1] > terminal_wave for task in (2, 3))
        )
        validate_worker_schedule(graph, scheduled)

    def test_global_list_schedule_commits_oversized_ready_cohort_across_waves(
        self,
    ) -> None:
        graph, baseline, plans = _partial_release_chain_problem(
            consumer_cohort_width=5,
        )

        scheduled = _global_unit_list_schedule(
            graph,
            baseline,
            plans,
            frozenset(),
            pipeline_depth=3,
        )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None

        def slot(root: int, task: int) -> int:
            task_placement = placement(scheduled, root, task)
            self.assertIsNotNone(task_placement)
            assert task_placement is not None
            worker, wave = task_placement
            return wave * scheduled.worker_count + worker

        producer_slots = [
            slot(root, task)
            for root in (0, 1, 2)
            for task in range(graph.root_domains[root].size)
        ]
        consumer_slots = [slot(3, task) for task in range(5)]
        self.assertGreater(consumer_slots[0], max(producer_slots))
        self.assertEqual(
            consumer_slots,
            list(range(consumer_slots[0], consumer_slots[0] + 5)),
        )
        self.assertNotEqual(
            consumer_slots[0] // scheduled.worker_count,
            consumer_slots[-1] // scheduled.worker_count,
        )
        self.assertTrue(
            all(
                not consumer_slots[0] <= slot(4, task) <= consumer_slots[-1]
                for task in range(graph.root_domains[4].size)
            )
        )
        validate_worker_schedule(graph, scheduled)

    def test_event_frontier_backfills_retired_lanes_with_nested_chain(self) -> None:
        graph, baseline, plans = _retirement_frontier_chain_problem(
            12,
            nested_consumer_count=4,
        )

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                baseline,
                plans,
                frozenset(),
                pipeline_depth=2,
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        # Producer ownership is the fixed side of the transaction.
        self.assertEqual(
            [placement(scheduled, 0, task) for task in range(12)],
            [placement(baseline, 0, task) for task in range(12)],
        )
        # Four complete fan-in-two cohorts become ready before the producer's
        # terminal wave.  The remaining two stay in their prepared rank.
        self.assertEqual(
            {placement(scheduled, 1, task)[1] for task in range(4)},
            {1},
        )
        self.assertEqual(
            {placement(scheduled, 1, task)[1] for task in range(4, 6)},
            {2},
        )
        # The held producer frontier is published after the one-use window;
        # its late intermediate cohort and nested consumer share the next rank.
        self.assertEqual(
            {placement(scheduled, 2, task)[1] for task in range(4)},
            {2},
        )
        self.assertEqual(len(scheduled.segments_for_root(1)), 2)
        validate_worker_schedule(graph, scheduled)

    def test_retirement_frontier_matches_qwen_complementary_lanes(self) -> None:
        graph, baseline, plans, root_barriers = _qwen_shaped_retirement_problem()

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                baseline,
                plans,
                root_barriers,
                pipeline_depth=2,
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        # The producer remains the selected dense interval [2930, 4466).
        self.assertEqual(
            tuple(
                segment.resident_slot_interval
                for segment in scheduled.segments_for_root(1)
            ),
            ((2930, 4466),),
        )
        traversal = _root_schedule_traversal(
            baseline.segments_for_root(1),
            graph.root_task_orders[1],
        )
        self.assertIsNotNone(traversal)
        assert traversal is not None
        self.assertIsNotNone(traversal.logical_task_to_scheduled_ordinal)
        assert traversal.logical_task_to_scheduled_ordinal is not None
        frontier = cross_loop_scheduler._maximum_value_by_key(
            graph.events[0].producers[0].keys_by_producer,
            traversal.logical_task_to_scheduled_ordinal,
        )
        self.assertIsNotNone(frontier)
        assert frontier is not None
        reduction_keys = graph.events[0].producers[0].producers_by_key.source_domain
        reduction_batch_axis, reduction_iteration_axis = reduction_keys.axis_order
        self.assertTrue(
            frontier.is_pointwise_equal_to(
                CoordinateRelation.point_map(
                    reduction_keys,
                    traversal.logical_task_to_scheduled_ordinal.target_domain,
                    (
                        (
                            (
                                (reduction_batch_axis, 0, 1, 1),
                                (reduction_iteration_axis, 0, 96, 1),
                            ),
                            (
                                16 * coordinate_axis_symbol(reduction_iteration_axis)
                                + 15,
                            ),
                        ),
                    ),
                )
            )
        )
        # Its 1184-task drain frontier releases 74 complete fan-in-16
        # cohorts.  The complementary 22 cohorts remain after the one-use
        # low-count-lane window, independent of these model-derived numbers.
        self.assertEqual(
            [placement(scheduled, 2, task) for task in range(74)],
            [(worker, 3) for worker in range(914, 988)],
        )
        self.assertEqual(
            [placement(scheduled, 2, task) for task in range(74, 96)],
            [(worker, 4) for worker in range(562, 584)],
        )
        reduction_segments = scheduled.segments_for_root(2)
        self.assertEqual(
            tuple(segment.resident_slot_interval for segment in reduction_segments),
            ((4466, 4540), (5298, 5320)),
        )
        self.assertEqual(
            tuple(segment.task_count for segment in reduction_segments),
            (74, 22),
        )
        # The nested consumer uses the next rank of the same retirement
        # window.  Keeping its rank-sized action intact lets the ordinary
        # nested-counter quotient observe the earlier 74-task prefix.
        self.assertEqual(
            [placement(scheduled, 3, task) for task in range(512)],
            [(worker, 4) for worker in range(512)],
        )
        self.assertIsNotNone(
            cross_loop_scheduler._packed_schedule_segment_geometry(scheduled)
        )
        compact = cross_loop_scheduler._compact_nested_loop_counters_for_schedule(
            graph,
            scheduled,
            plans,
        )
        self.assertEqual(compact[1].readiness_key_count, 2)
        self.assertEqual(
            [
                piece.target_ranges
                for piece in compact[1].producers[0].producers_by_key.pieces
            ],
            [
                ((30, 0, 1, 1), (31, 0, 74, 1)),
                ((30, 0, 1, 1), (31, 74, 96, 1)),
            ],
        )
        validate_worker_schedule(graph, scheduled)

    def test_retirement_frontier_declines_full_worker_round(self) -> None:
        graph, baseline, plans = _retirement_frontier_chain_problem(16)

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                baseline,
                plans,
                frozenset(),
                pipeline_depth=2,
            )

        # A producer with no low-count lanes has no retirement window.
        self.assertEqual(scheduled, baseline)
        validate_worker_schedule(graph, scheduled)

    def test_event_frontier_uses_idle_lanes_for_subwave_producer(self) -> None:
        graph, baseline, plans = _retirement_frontier_chain_problem(
            6,
            fan_in=3,
        )

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                baseline,
                plans,
                frozenset(),
                pipeline_depth=2,
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        self.assertEqual(
            [placement(scheduled, 0, task) for task in range(6)],
            [placement(baseline, 0, task) for task in range(6)],
        )
        # All producers are resident in the first wave.  Their complete
        # two-task consumer cohort can therefore use the two otherwise-idle
        # lanes without splitting the cohort or delaying a producer.
        self.assertEqual(
            [placement(scheduled, 1, task) for task in range(2)],
            [(6, 0), (7, 0)],
        )
        self.assertLess(
            cross_loop_scheduler._resident_schedule_occupied_wave_count(scheduled),
            cross_loop_scheduler._resident_schedule_occupied_wave_count(baseline),
        )
        validate_worker_schedule(graph, scheduled)

    def test_retirement_frontier_fits_only_complete_actions(self) -> None:
        def schedule(consumer_count: int) -> tuple[WorkerSchedule, WorkerSchedule]:
            producer_domain, consumer_domain = _identify_root_domains(
                (_domain((10, 10, 1)), _domain((20, consumer_count, 1)))
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
                                    ((10, 0, 8, 1),),
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
            graph = _readiness_graph(
                (producer_domain, consumer_domain),
                event,
            )
            plans = (ReadinessCounterPlan(event.producers, event.consumers),)
            baseline = _baseline_worker_schedule(graph.root_domains, worker_count=8)
            candidate = cross_loop_scheduler._event_frontier_list_schedule(
                graph,
                baseline,
                plans,
                frozenset(),
                pipeline_depth=2,
            )
            self.assertIsNotNone(candidate)
            assert candidate is not None
            validate_worker_schedule(graph, candidate)
            return baseline, candidate

        exact_baseline, exact = schedule(6)
        self.assertEqual(
            [placement(exact, 1, task) for task in range(6)],
            [(worker, 1) for worker in range(2, 8)],
        )
        self.assertLess(
            cross_loop_scheduler._resident_schedule_occupied_wave_count(exact),
            cross_loop_scheduler._resident_schedule_occupied_wave_count(exact_baseline),
        )

        too_wide_baseline, too_wide = schedule(7)
        self.assertEqual(
            [placement(too_wide, 1, task) for task in range(7)],
            [
                *((worker, 1) for worker in range(2, 8)),
                (0, 2),
            ],
        )
        # No retirement window opens when no complete action fits.  The
        # ordinary selector keeps the seven-task cohort as one committed run;
        # crossing a rank boundary does not split the action or extend the
        # prepared three-wave horizon.
        self.assertEqual(
            cross_loop_scheduler._resident_schedule_occupied_wave_count(too_wide),
            cross_loop_scheduler._resident_schedule_occupied_wave_count(
                too_wide_baseline
            ),
        )

    def test_retirement_frontier_claims_nested_wait_on_disjoint_lane(self) -> None:
        graph, baseline, plans = _retirement_frontier_chain_problem(
            10,
            nested_consumer_count=2,
        )

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                baseline,
                plans,
                frozenset(),
                pipeline_depth=2,
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        self.assertEqual(
            [placement(scheduled, 0, task) for task in range(10)],
            [placement(baseline, 0, task) for task in range(10)],
        )
        intermediate = {placement(scheduled, 1, task) for task in range(4)}
        nested = {placement(scheduled, 2, task) for task in range(2)}
        self.assertEqual({wave for _worker, wave in intermediate}, {1})
        self.assertEqual({wave for _worker, wave in nested}, {1})
        self.assertFalse(
            {worker for worker, _wave in intermediate}
            & {worker for worker, _wave in nested}
        )
        self.assertEqual(placement(scheduled, 1, 4)[1], 2)
        validate_worker_schedule(graph, scheduled)

    def test_retirement_frontier_backfills_complete_incomparable_branch(
        self,
    ) -> None:
        graph, baseline, plans = _retirement_backfill_problem(
            descendant_blocks_tail=False
        )

        with _forbid_schedule_enumeration():
            depth_two = _global_unit_list_schedule(
                graph,
                baseline,
                plans,
                frozenset(),
                pipeline_depth=2,
            )
            depth_three = _global_unit_list_schedule(
                graph,
                baseline,
                plans,
                frozenset(),
                pipeline_depth=3,
            )

        self.assertIsNotNone(depth_two)
        self.assertIsNotNone(depth_three)
        assert depth_two is not None and depth_three is not None
        anchor_placement = [placement(depth_two, 1, task) for task in range(10)]
        self.assertEqual(
            anchor_placement,
            [(4, 0), (5, 0), (6, 0), (7, 0)] + [(worker, 1) for worker in range(6)],
        )
        self.assertEqual(
            [placement(depth_three, 1, task) for task in range(10)],
            anchor_placement,
        )
        # Root 1's rotated ten-task run ends at slot 14 and reserves slots
        # [14, 20). Its first one-task descendant uses slot 14. The complete
        # independent root 3 then occupies [15, 18) instead of wasting the
        # otherwise dead reservation.
        self.assertEqual(
            [placement(depth_two, 3, task) for task in range(3)],
            [(7, 1), (0, 2), (1, 2)],
        )
        # Root 4 is one causal level deeper. Depth two leaves it outside the
        # reservation; depth three admits its complete two-task suffix.
        self.assertEqual(
            [placement(depth_two, 4, task) for task in range(2)],
            [(5, 2), (6, 2)],
        )
        self.assertEqual(
            [placement(depth_three, 4, task) for task in range(2)],
            [(2, 2), (3, 2)],
        )
        validate_worker_schedule(graph, depth_two)
        validate_worker_schedule(graph, depth_three)

    def test_rank_deferred_descendant_blocks_retirement_backfill(self) -> None:
        graph, baseline, plans = _retirement_backfill_problem(
            descendant_blocks_tail=True
        )

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                baseline,
                plans,
                frozenset(),
                pipeline_depth=2,
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        # Root 3 is a ready two-task descendant at slot 15, where only one
        # lane remains in the current rank. It must start at slot 16; unrelated
        # root 4 may not steal slot 15 while that action is rank-deferred.
        self.assertEqual(
            [placement(scheduled, 3, task) for task in range(2)],
            [(0, 2), (1, 2)],
        )
        unrelated_slot = placement(scheduled, 4, 0)
        self.assertIsNotNone(unrelated_slot)
        assert unrelated_slot is not None
        self.assertNotEqual(unrelated_slot, (7, 1))
        validate_worker_schedule(graph, scheduled)

    def test_retirement_frontier_uses_one_admission_frontier_for_all_waits(
        self,
    ) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (_domain((10, 10, 1)), _domain((20, 2, 1)))
        )
        key_domain = _domain((0, 1), kind="event", identity=0)
        producer = _readiness_producer_from_publication(
            0,
            _full_point_map(
                producer_domain,
                key_domain,
                sympy.Integer(0),
            ),
        )
        root_consumer = ReadinessConsumer(
            consumer_root=1,
            keys_by_consumer=_full_point_map(
                consumer_domain,
                key_domain,
                sympy.Integer(0),
            ),
        )
        nested_domain = _domain((20, 2, 1), (21, 1, 1), identity=71)
        nested_consumer = ReadinessConsumer(
            consumer_root=1,
            consumer_site_id=71,
            keys_by_consumer=_full_point_map(
                nested_domain,
                key_domain,
                sympy.Integer(0),
            ),
        )
        event = ReadinessEvent((producer,), (root_consumer, nested_consumer))
        graph = _readiness_graph((producer_domain, consumer_domain), event)
        plan = ReadinessCounterPlan(event.producers, event.consumers)
        baseline = _baseline_worker_schedule(graph.root_domains, worker_count=8)

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                baseline,
                (plan,),
                frozenset(),
                pipeline_depth=2,
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        # The drain prefix releases no complete key, so no retirement window
        # opens. Ordinary placement may still use disjoint lanes in the same
        # rank; the final segment-DAG proof is authoritative for both waits.
        self.assertEqual(
            [placement(scheduled, 1, task) for task in range(2)],
            [(2, 1), (3, 1)],
        )
        validate_worker_schedule(graph, scheduled)

    def test_retirement_frontier_segments_do_not_scale_with_key_count(self) -> None:
        segment_counts: list[int] = []
        cohort_proof_counts: list[int] = []
        for producer_count in (12, 1204):
            graph, baseline, plans = _retirement_frontier_chain_problem(producer_count)
            cohort_interval = cross_loop_scheduler._cohort_interval_at_cursor
            with (
                _forbid_schedule_enumeration(),
                mock.patch.object(
                    cross_loop_scheduler,
                    "_cohort_interval_at_cursor",
                    wraps=cohort_interval,
                ) as cohort_interval_spy,
            ):
                scheduled = _global_unit_list_schedule(
                    graph,
                    baseline,
                    plans,
                    frozenset(),
                    pipeline_depth=2,
                )
            self.assertIsNotNone(scheduled)
            assert scheduled is not None
            self.assertEqual(
                [placement(scheduled, 0, task) for task in range(producer_count)],
                [placement(baseline, 0, task) for task in range(producer_count)],
            )
            producer_terminal_wave = producer_count // 8
            self.assertEqual(
                {placement(scheduled, 1, task)[1] for task in range(4)},
                {producer_terminal_wave},
            )
            segment_counts.append(len(scheduled.segments_for_root(1)))
            cohort_proof_counts.append(cohort_interval_spy.call_count)
            validate_worker_schedule(graph, scheduled)

        # The adjacent moved prefix and canonical suffix merge back into one
        # segment; neither schedule representation nor proof work scales with
        # the number of readiness keys behind the retirement stratum.
        self.assertEqual(segment_counts, [1, 1])
        self.assertEqual(cohort_proof_counts[0], cohort_proof_counts[1])

    def test_retirement_frontier_uses_rotated_prepared_interval(self) -> None:
        root_domains = _identify_root_domains(
            (
                _domain((10, 3, 1)),
                _domain((20, 10, 1)),
                _domain((30, 5, 1)),
            )
        )
        key_domain = _domain((0, 5), kind="event", identity=0)
        key = coordinate_axis_symbol(0)
        event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=1,
                    producers_by_key=CoordinateRelation(
                        key_domain,
                        root_domains[1],
                        (
                            _CoordinateRelationPiece(
                                ((0, 0, 5, 1),),
                                ((20, 2 * key, 2 * key + 2, 1),),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=2,
                    keys_by_consumer=_full_point_map(
                        root_domains[2],
                        key_domain,
                        coordinate_axis_symbol(30),
                    ),
                ),
            ),
        )
        graph = _readiness_graph(root_domains, event)
        plan = ReadinessCounterPlan(event.producers, event.consumers)
        baseline = _baseline_worker_schedule(root_domains, worker_count=8)

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                baseline,
                (plan,),
                frozenset(((0, 1),)),
                pipeline_depth=2,
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        self.assertEqual(
            [placement(scheduled, 1, task) for task in range(10)],
            [
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (7, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
            ],
        )
        # The one-use low-count-lane window [13,19) wraps across the absolute
        # wave boundary while preserving the rotated producer placement.
        self.assertEqual(
            [placement(scheduled, 2, task) for task in range(4)],
            [(5, 1), (6, 1), (7, 1), (0, 2)],
        )
        self.assertEqual(placement(scheduled, 2, 4), (3, 2))
        validate_worker_schedule(graph, scheduled)

    def test_event_frontier_fast_forwards_large_committed_suffix(self) -> None:
        def schedule(
            consumer_count: int,
        ) -> tuple[WorkerSchedule, int]:
            root_domains = _identify_root_domains(
                (
                    _domain((10, 1, 1)),
                    _domain((20, consumer_count, 1)),
                )
            )
            event = _whole_root_readiness_event(
                root_domains,
                producer_root=0,
                consumer_root=1,
                event_id=0,
            )
            graph = _readiness_graph(root_domains, event)
            plan = ReadinessCounterPlan(event.producers, event.consumers)
            prepared = _schedule(
                4,
                _segment(
                    0,
                    graph.root_task_orders[0],
                    workers=(0, 4),
                    dispatch_offset=0,
                ),
                _segment(
                    1,
                    graph.root_task_orders[1],
                    workers=(0, 4),
                    dispatch_offset=1,
                ),
            )
            placed_run_type = cross_loop_scheduler._PlacedRun
            with (
                _forbid_schedule_enumeration(),
                mock.patch.object(
                    cross_loop_scheduler,
                    "_PlacedRun",
                    wraps=placed_run_type,
                ) as placed_run_spy,
            ):
                result = cross_loop_scheduler._event_frontier_list_schedule(
                    graph,
                    prepared,
                    (plan,),
                    frozenset(),
                    pipeline_depth=2,
                )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertIsNot(result, prepared)
            return result, placed_run_spy.call_count

        small_count = 13
        large_count = 16_385
        small, small_run_count = schedule(small_count)
        large, large_run_count = schedule(large_count)

        # The committed consumer suffix has the same prefix/interior/tail
        # structure in both schedules. Its construction work must therefore
        # be independent of the number of full waves in the interior.
        self.assertEqual(large_run_count, small_run_count)
        self.assertLessEqual(large_run_count, 4)
        self.assertEqual(
            [
                (
                    segment.root,
                    segment.worker_begin,
                    segment.worker_count,
                    segment.dispatch_offset,
                )
                for segment in large.segments
            ],
            [
                (
                    segment.root,
                    segment.worker_begin,
                    segment.worker_count,
                    segment.dispatch_offset,
                )
                for segment in small.segments
            ],
        )
        for task in (0, 1, 2, 3, small_count - 1):
            self.assertEqual(
                placement(large, 1, task),
                placement(small, 1, task),
            )
        self.assertEqual(
            placement(large, 1, large_count - 1),
            (
                large_count % large.worker_count,
                large_count // large.worker_count,
            ),
        )
        self.assertEqual(
            large.worker_step_domain.size,
            (large_count + large.worker_count) // large.worker_count,
        )

    def test_event_frontier_preflight_declines_before_chooser(self) -> None:
        root_domains = _identify_root_domains(
            (
                _domain((10, 1, 1)),
                _domain((20, 4, 1)),
            )
        )
        event = _whole_root_readiness_event(root_domains, 0, 1, 0)
        graph = _readiness_graph(root_domains, event)
        plan = ReadinessCounterPlan(event.producers, event.consumers)
        prepared = _schedule(
            4,
            _segment(
                0,
                graph.root_task_orders[0],
                workers=(0, 4),
                dispatch_offset=0,
            ),
            _segment(
                1,
                graph.root_task_orders[1],
                workers=(0, 4),
                dispatch_offset=1,
            ),
        )

        with (
            _forbid_schedule_enumeration(),
            mock.patch.object(cross_loop_scheduler, "_MAX_GLOBAL_LIST_WORK", 1),
            mock.patch.object(
                cross_loop_scheduler,
                "_root_schema_criticality",
                side_effect=AssertionError("chooser ran after preflight decline"),
            ) as criticality,
        ):
            scheduled = cross_loop_scheduler._event_frontier_list_schedule(
                graph,
                prepared,
                (plan,),
                frozenset(),
                pipeline_depth=2,
            )

        self.assertIs(scheduled, prepared)
        criticality.assert_not_called()

    def test_event_frontier_preflights_too_many_admission_cohorts(self) -> None:
        cohort_count = tile_dependency._MAX_RELATION_PIECES + 1
        root_domains = _identify_root_domains(
            (
                _domain((10, cohort_count, 1)),
                _domain((20, cohort_count, 1)),
            )
        )
        event = _pointwise_root_readiness_event(root_domains, 0, 1, 0)
        graph = _readiness_graph(root_domains, event)
        plan = ReadinessCounterPlan(event.producers, event.consumers)
        prepared = _schedule(
            4,
            _segment(
                0,
                graph.root_task_orders[0],
                workers=(0, 4),
                dispatch_offset=0,
            ),
            _segment(
                1,
                graph.root_task_orders[1],
                workers=(0, 4),
                dispatch_offset=cohort_count,
            ),
        )

        with (
            _forbid_schedule_enumeration(),
            mock.patch.object(
                cross_loop_scheduler,
                "_root_schema_criticality",
                side_effect=AssertionError("chooser ran after preflight decline"),
            ) as criticality,
        ):
            scheduled = cross_loop_scheduler._event_frontier_list_schedule(
                graph,
                prepared,
                (plan,),
                frozenset(),
                pipeline_depth=2,
            )

        self.assertIs(scheduled, prepared)
        criticality.assert_not_called()

    def test_global_list_schedule_does_not_spill_ready_suffix_past_ancestor(
        self,
    ) -> None:
        graph, baseline, plans = _partial_release_chain_problem(
            consumer_cohort_width=5,
            consumer_cohort_count=1,
        )

        scheduled = cross_loop_scheduler._event_frontier_list_schedule(
            graph,
            baseline,
            plans,
            frozenset(),
            pipeline_depth=3,
        )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None

        def slot(root: int, task: int) -> int:
            worker, wave = placement(scheduled, root, task)
            return wave * scheduled.worker_count + worker

        # The entire consumer suffix is ready after producer tasks 0 and 1,
        # but it is wider than the three-lane tail. Starting it there would
        # push the unfinished producer into a later wave. Finish the ancestor,
        # then commit the consumer suffix across waves.
        producer_slots = [slot(2, task) for task in range(4)]
        consumer_slots = [slot(3, task) for task in range(5)]
        self.assertGreater(consumer_slots[0], max(producer_slots))
        self.assertEqual(
            consumer_slots,
            list(range(consumer_slots[0], consumer_slots[0] + 5)),
        )
        validate_worker_schedule(graph, scheduled)

    def test_global_list_schedule_does_not_spill_past_transitive_ancestor(
        self,
    ) -> None:
        graph, _baseline, _plans = _partial_release_chain_problem(
            consumer_cohort_width=1,
            consumer_cohort_count=1,
        )
        descendant_domain = _domain((60, 5, 1), identity=5)
        root_task_orders = (
            *graph.root_task_orders,
            pid_task_order(descendant_domain, descendant_domain.axis_order),
        )
        root_domains = tuple(
            task_order.target_domain for task_order in root_task_orders
        )
        final_event = _whole_root_readiness_event(
            root_domains,
            producer_root=3,
            consumer_root=5,
            event_id=3,
        )
        graph = ReadinessGraph(
            root_task_orders=root_task_orders,
            events=(*graph.events, final_event),
        )
        plans = tuple(
            ReadinessCounterPlan(event.producers, event.consumers)
            for event in graph.events
        )
        dispatch_offset = 0
        prepared_segments: list[WorkerScheduleSegment] = []
        for root, root_domain in enumerate(root_domains):
            prepared_segments.append(
                _segment(
                    root,
                    root_task_orders[root],
                    workers=(0, 4),
                    dispatch_offset=dispatch_offset,
                )
            )
            dispatch_offset += root_domain.size
        prepared = _schedule(
            4,
            *prepared_segments,
        )

        scheduled = cross_loop_scheduler._event_frontier_list_schedule(
            graph,
            prepared,
            plans,
            frozenset(),
            pipeline_depth=4,
        )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None

        def slot(root: int, task: int) -> int:
            task_placement = placement(scheduled, root, task)
            self.assertIsNotNone(task_placement)
            assert task_placement is not None
            worker, wave = task_placement
            return wave * scheduled.worker_count + worker

        terminal_wave = placement(scheduled, 0, 4)[1]
        self.assertEqual(
            {placement(scheduled, 2, task)[1] for task in (0, 1)},
            {terminal_wave},
        )
        self.assertEqual(placement(scheduled, 3, 0)[1], terminal_wave)

        # Root 3 completes and releases root 5 in the terminal wave, but the
        # five-task action cannot fit that wave. Unlike the independent root
        # backfill exercised above, root 5 is transitively downstream of the
        # still-unfinished roots 1 and 2. It must not become a committed run
        # across later waves until those crossed ancestors are assigned.
        transitive_ancestor_slots = [
            slot(root, task)
            for root in (1, 2)
            for task in range(graph.root_domains[root].size)
        ]
        descendant_slots = [slot(5, task) for task in range(5)]
        self.assertGreater(
            min(descendant_slots),
            max(transitive_ancestor_slots),
        )
        validate_worker_schedule(graph, scheduled)

    def test_global_list_schedule_does_not_treat_order_pieces_as_cohorts(
        self,
    ) -> None:
        graph, _baseline, plans = _partial_release_chain_problem(
            consumer_cohort_width=4,
            leading_task_count=4,
        )
        reference_order = graph.root_task_orders[3]
        (order_axis,) = reference_order.source_domain.axis_order
        (task_axis,) = reference_order.target_domain.axis_order
        order_index = coordinate_axis_symbol(order_axis)
        split_order = CoordinateRelation(
            reference_order.source_domain,
            reference_order.target_domain,
            (
                _CoordinateRelationPiece(
                    ((order_axis, 0, 2, 1),),
                    ((task_axis, order_index, order_index + 1, 1),),
                ),
                _CoordinateRelationPiece(
                    ((order_axis, 2, 8, 1),),
                    ((task_axis, order_index, order_index + 1, 1),),
                ),
            ),
        )
        self.assertEqual(len(split_order.pieces), 2)
        consumer_cohort = cross_loop_scheduler._readiness_equivalent_cohort_relation(
            split_order,
            graph.events[-1].consumers[0].keys_by_consumer,
        )
        self.assertIsNotNone(consumer_cohort)
        assert consumer_cohort is not None
        self.assertEqual(
            cross_loop_scheduler._cohort_interval_end_at_cursor(
                consumer_cohort[0],
                0,
            ),
            4,
        )
        graph = dataclasses.replace(
            graph,
            root_task_orders=(
                *graph.root_task_orders[:3],
                split_order,
                *graph.root_task_orders[4:],
            ),
        )
        baseline = _baseline_worker_schedule(
            graph.root_domains,
            worker_count=4,
            root_task_orders=graph.root_task_orders,
        )

        scheduled = _global_unit_list_schedule(
            graph,
            baseline,
            plans,
            frozenset(),
            pipeline_depth=3,
        )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        producer_cohort_wave = placement(scheduled, 2, 0)[1]
        # The first producer cohort leaves two lanes, exactly the width of the
        # first relation piece but only half the semantic consumer cohort. The
        # other upstream root fills those lanes; the four-task consumer
        # stays atomic and begins later.
        self.assertEqual(
            {placement(scheduled, 2, task)[1] for task in (0, 1)},
            {producer_cohort_wave},
        )
        self.assertTrue(
            all(
                placement(scheduled, 2, task)[1] > producer_cohort_wave
                for task in (2, 3)
            )
        )
        self.assertTrue(
            all(
                placement(scheduled, 3, task)[1] > producer_cohort_wave
                for task in range(8)
            )
        )
        validate_worker_schedule(graph, scheduled)

    def test_global_list_schedule_validates_pipeline_depth(self) -> None:
        (domain,) = _identify_root_domains((_domain((10, 1, 1)),))
        graph = _readiness_graph((domain,))
        baseline = _baseline_worker_schedule(graph.root_domains, worker_count=1)

        for pipeline_depth in (0, 5, True, "2"):
            with (
                self.subTest(pipeline_depth=pipeline_depth),
                self.assertRaisesRegex(
                    ValueError,
                    "pipeline depth must be an integer between 1 and 4",
                ),
            ):
                _global_unit_list_schedule(
                    graph,
                    baseline,
                    (),
                    frozenset(),
                    pipeline_depth=cast("int", pipeline_depth),
                )

    def test_global_list_schedule_bounds_causal_pull_depth(self) -> None:
        root_domains = _identify_root_domains(
            tuple(_domain((10 + 10 * root, 4, 1)) for root in range(5))
        )
        graph = _readiness_graph(
            root_domains,
            _whole_root_readiness_event(root_domains, 0, 2, 0),
            _whole_root_readiness_event(root_domains, 2, 3, 1),
            _whole_root_readiness_event(root_domains, 3, 4, 2),
        )
        plans = tuple(
            ReadinessCounterPlan(event.producers, event.consumers)
            for event in graph.events
        )
        baseline = _baseline_worker_schedule(graph.root_domains, worker_count=4)
        prepared = cross_loop_scheduler._consumer_major_producer_order(
            graph,
            baseline,
            plans,
            frozenset(),
            excluded_roots=frozenset(),
        )

        schedules: dict[int, WorkerSchedule] = {}
        with _forbid_schedule_enumeration():
            for pipeline_depth in range(1, 5):
                scheduled = _global_unit_list_schedule(
                    graph,
                    prepared,
                    plans,
                    frozenset(),
                    pipeline_depth=pipeline_depth,
                )
                self.assertIsNotNone(scheduled)
                assert scheduled is not None
                schedules[pipeline_depth] = scheduled

        self.assertIs(schedules[1], prepared)
        for pipeline_depth, schedule in schedules.items():
            validate_worker_schedule(graph, schedule)
            first_wave = {root: placement(schedule, root, 0)[1] for root in range(5)}
            if pipeline_depth >= 2:
                self.assertLess(first_wave[2], first_wave[1])
            else:
                self.assertGreater(first_wave[2], first_wave[1])
            if pipeline_depth >= 3:
                self.assertLess(first_wave[3], first_wave[1])
            else:
                self.assertGreater(first_wave[3], first_wave[1])
            if pipeline_depth == 4:
                self.assertLess(first_wave[4], first_wave[1])
            else:
                self.assertGreater(first_wave[4], first_wave[1])

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

    def test_global_list_schedule_exact_tie_preserves_prepared_root_order(
        self,
    ) -> None:
        root_domains = _identify_root_domains(
            tuple(_domain((10 + 10 * root, 1, 1)) for root in range(4))
        )
        graph = _readiness_graph(
            root_domains,
            _whole_root_readiness_event(root_domains, 0, 2, 0),
            _whole_root_readiness_event(root_domains, 1, 3, 1),
        )
        plans = tuple(
            ReadinessCounterPlan(event.producers, event.consumers)
            for event in graph.events
        )
        prepared_root_order = (1, 0, 3, 2)
        prepared = _schedule(
            1,
            *(
                _segment(
                    root,
                    graph.root_task_orders[root],
                    workers=(0, 1),
                    dispatch_offset=worker_step,
                )
                for worker_step, root in enumerate(prepared_root_order)
            ),
        )
        criticality = cross_loop_scheduler._root_schema_criticality(
            len(root_domains),
            frozenset(((0, 2), (1, 3))),
        )
        self.assertIsNotNone(criticality)
        assert criticality is not None
        self.assertEqual(criticality[0], criticality[1])
        self.assertEqual(criticality[2], criticality[3])
        self.assertTrue(
            cross_loop_scheduler._schedule_is_progress_safe(
                prepared,
                graph,
                plans,
                frozenset(),
            )
        )
        self.assertIs(
            _global_unit_list_schedule(
                graph,
                prepared,
                plans,
                frozenset(),
                pipeline_depth=1,
            ),
            prepared,
        )

        scheduled = cross_loop_scheduler._event_frontier_list_schedule(
            graph,
            prepared,
            plans,
            frozenset(),
            pipeline_depth=2,
        )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        # Both first actions have identical static priority and symmetrically
        # release one child. The prepared packed order, not numeric root ID,
        # is therefore the final deterministic tie-breaker. Once root 1 wins
        # that tie, its newly released child strictly outranks root 0.
        self.assertEqual(task_at(scheduled, 0, 0), (1, 0))
        self.assertEqual(
            [task_at(scheduled, 0, worker_step)[0] for worker_step in range(4)],
            [1, 3, 0, 2],
        )
        validate_worker_schedule(graph, scheduled)

    def test_global_list_schedule_prioritizes_a_released_consumer(self) -> None:
        root_domains = _identify_root_domains(
            tuple(_domain((10 + 10 * root, 2, 1)) for root in range(5))
        )
        producer_to_candidate = _whole_consumer_subset_readiness_event(
            root_domains,
            0,
            2,
            0,
        )
        control_graph = _readiness_graph(root_domains, producer_to_candidate)
        release_graph = _readiness_graph(
            root_domains,
            producer_to_candidate,
            _whole_consumer_join_readiness_event(
                root_domains,
                (0, 1),
                4,
                1,
            ),
        )

        def schedule(graph: ReadinessGraph) -> WorkerSchedule:
            plans = tuple(
                ReadinessCounterPlan(event.producers, event.consumers)
                for event in graph.events
            )
            result = _global_unit_list_schedule(
                graph,
                _baseline_worker_schedule(graph.root_domains, worker_count=4),
                plans,
                frozenset(),
                pipeline_depth=2,
            )
            self.assertIsNotNone(result)
            assert result is not None
            return result

        with _forbid_schedule_enumeration():
            control = schedule(control_graph)
            released = schedule(release_graph)
        validate_worker_schedule(control_graph, control)
        validate_worker_schedule(release_graph, released)

        # With no release, canonical root 3 precedes root 4. Once roots 0 and
        # 1 complete the join, root 4 becomes ready and is selected first.
        self.assertLess(placement(control, 3, 0)[1], placement(control, 4, 0)[1])
        self.assertLess(
            placement(released, 4, 0)[1],
            placement(released, 3, 0)[1],
        )

    def test_global_list_schedule_continues_an_active_event(self) -> None:
        root_domains = _identify_root_domains(
            tuple(
                _domain((10 + 10 * root, 4 if root == 1 else 2, 1)) for root in range(8)
            )
        )
        graph = _readiness_graph(
            root_domains,
            _whole_root_readiness_event(root_domains, 0, 2, 0),
            _whole_root_readiness_event(root_domains, 0, 3, 1),
            _whole_consumer_join_readiness_event(
                root_domains,
                (0, 3, 4),
                6,
                2,
            ),
            _whole_consumer_join_readiness_event(
                root_domains,
                (2, 5),
                7,
                3,
            ),
        )
        plans = tuple(
            ReadinessCounterPlan(event.producers, event.consumers)
            for event in graph.events
        )

        with _forbid_schedule_enumeration():
            scheduled = _global_unit_list_schedule(
                graph,
                _baseline_worker_schedule(graph.root_domains, worker_count=2),
                plans,
                frozenset(),
                pipeline_depth=2,
            )

        self.assertIsNotNone(scheduled)
        assert scheduled is not None
        validate_worker_schedule(graph, scheduled)
        # Roots 2 and 3 have the same structural depth. Root 3 contributes to
        # the join already opened by root 0, so it precedes canonical root 2.
        self.assertLess(
            placement(scheduled, 3, 0)[1],
            placement(scheduled, 2, 0)[1],
        )

    def test_global_list_schedule_ignores_disjoint_event_keys(self) -> None:
        root_domains = _identify_root_domains(
            (
                _domain((10, 2, 1)),
                _domain((20, 4, 1)),
                _domain((30, 2, 1)),
                _domain((40, 2, 1)),
            )
        )
        event_domain = _domain((0, 2), kind="event", identity=0)
        relevant_producer = _readiness_producer_from_publication(
            0,
            _full_point_map(
                root_domains[0],
                event_domain,
                sympy.Integer(1),
            ),
        )
        disjoint_producer = _readiness_producer_from_publication(
            2,
            _full_point_map(
                root_domains[2],
                event_domain,
                sympy.Integer(0),
            ),
        )
        consumer = ReadinessConsumer(
            consumer_root=3,
            keys_by_consumer=_full_point_map(
                root_domains[3],
                event_domain,
                sympy.Integer(1),
            ),
        )

        def graph_and_schedule(
            producers: tuple[ReadinessProducer, ...],
        ) -> tuple[ReadinessGraph, WorkerSchedule]:
            event = ReadinessEvent(producers, (consumer,))
            graph = _readiness_graph(root_domains, event)
            scheduled = _global_unit_list_schedule(
                graph,
                _baseline_worker_schedule(graph.root_domains, worker_count=4),
                (ReadinessCounterPlan(event.producers, event.consumers),),
                frozenset(),
                pipeline_depth=2,
            )
            self.assertIsNotNone(scheduled)
            assert scheduled is not None
            return graph, scheduled

        with _forbid_schedule_enumeration():
            control_graph, control = graph_and_schedule((relevant_producer,))
            graph, scheduled = graph_and_schedule(
                (relevant_producer, disjoint_producer)
            )

        for root, task_count in enumerate((2, 4, 2, 2)):
            for task in range(task_count):
                self.assertEqual(
                    placement(scheduled, root, task),
                    placement(control, root, task),
                )
        validate_worker_schedule(control_graph, control)
        validate_worker_schedule(graph, scheduled)

    def test_global_list_schedule_uses_exact_static_fanout_frontier(self) -> None:
        graph, baseline, plans = _conservative_static_arm_problem()
        producer_traversal = _root_schedule_traversal(
            baseline.segments_for_root(0),
            graph.root_task_orders[0],
        )
        self.assertIsNotNone(producer_traversal)
        assert producer_traversal is not None
        producer_order = producer_traversal.scheduled_ordinal_to_logical_task
        self.assertIsNotNone(producer_order)
        assert producer_order is not None
        keys_by_producer = plans[0].producers[0].keys_by_producer
        self.assertIsNotNone(keys_by_producer)
        assert keys_by_producer is not None
        # Publication derivation retains its authoritative inverse, so this
        # fan-out arm now has the exact scalar frontier used by scheduling.
        self.assertIsNotNone(keys_by_producer.converse())
        ordered_producer_keys = producer_order.then(keys_by_producer)
        self.assertIsNotNone(ordered_producer_keys)
        assert ordered_producer_keys is not None
        ordinal_identity = CoordinateRelation.identity(
            producer_order.source_domain,
            producer_order.source_domain,
        )
        self.assertIsNotNone(
            cross_loop_scheduler._maximum_value_by_key(
                ordered_producer_keys,
                ordinal_identity,
            )
        )
        self.assertTrue(
            all(
                cross_loop_scheduler._supports_exact_counter_plan_lowering(
                    plan,
                    graph.root_domains,
                )
                for plan in plans
            )
        )

        proposal = cross_loop_scheduler._event_frontier_list_schedule(
            graph,
            baseline,
            plans,
            frozenset(),
            pipeline_depth=2,
        )

        self.assertIsNotNone(proposal)
        assert proposal is not None

        def slot(root: int, task: int) -> int:
            task_placement = placement(proposal, root, task)
            self.assertIsNotNone(task_placement)
            assert task_placement is not None
            worker, wave = task_placement
            return wave * proposal.worker_count + worker

        affected_producer_slots = [slot(0, task) for task in range(8)]
        affected_consumer_slots = [slot(3, task) for task in range(4)]
        self.assertGreater(
            min(affected_consumer_slots),
            max(affected_producer_slots),
        )

        # The exact fan-out frontier must not disable the unrelated chain. Its
        # consumer moves up from the baseline's next wave and shares a wave
        # with its own producer.
        exact_producer_wave = placement(proposal, 1, 0)[1]
        exact_consumer_wave = placement(proposal, 2, 0)[1]
        self.assertEqual(exact_consumer_wave, exact_producer_wave)
        self.assertLess(
            exact_consumer_wave,
            placement(baseline, 2, 0)[1],
        )

        final_plan = cross_loop_scheduler.StaticPipelinePlan(
            worker_schedule=proposal,
            root_task_orders=graph.root_task_orders,
            readiness_counters=plans,
            root_barrier_edges=frozenset(),
        )
        self.assertEqual(final_plan.readiness_counters, plans)
        self.assertEqual(final_plan.root_barrier_edges, frozenset())
        with mock.patch.object(
            cross_loop_scheduler,
            "_has_acyclic_symbolic_segment_precedence",
            wraps=cross_loop_scheduler._has_acyclic_symbolic_segment_precedence,
        ) as segment_precedence_proof:
            self.assertTrue(
                cross_loop_scheduler._schedule_is_progress_safe(
                    final_plan.worker_schedule,
                    graph,
                    final_plan.readiness_counters,
                    final_plan.root_barrier_edges,
                )
            )
        segment_precedence_proof.assert_called_once()
        validate_worker_schedule(graph, final_plan.worker_schedule)

    def test_readiness_cohort_relation_preserves_dynamic_native_order(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        task_domain = _identify_root_domains((_domain((10, batch, 1), (11, 3, 1)),))[0]
        task_order = pid_task_order(task_domain, (11, 10))
        key_domain = CoordinateDomain(
            axis_order=(0, 1),
            axis_counts_items=((0, batch), (1, 2)),
            kind="event",
            identity=0,
            _allow_empty=True,
        )
        keys_by_task = _full_point_map(
            task_domain,
            key_domain,
            coordinate_axis_symbol(10),
            sympy.floor(coordinate_axis_symbol(11) / 2),
        )

        with _forbid_schedule_enumeration():
            cohort = cross_loop_scheduler._readiness_equivalent_cohort_relation(
                task_order,
                keys_by_task,
            )
        self.assertIsNotNone(cohort)
        assert cohort is not None
        cohort_by_order, event_keys_by_cohort = cohort
        order_points_by_cohort = cohort_by_order.converse()
        self.assertIsNotNone(order_points_by_cohort)
        assert order_points_by_cohort is not None
        self.assertEqual(
            event_keys_by_cohort,
            CoordinateRelation.identity(key_domain, key_domain),
        )
        self.assertEqual(order_points_by_cohort, cohort_by_order.converse())
        task_keys = task_order.then(keys_by_task)
        self.assertIsNotNone(task_keys)
        assert task_keys is not None
        for concrete_batch in (0, 1, 3):
            with self.subTest(concrete_batch=concrete_batch):
                substitutions = {batch: concrete_batch}
                self.assertEqual(
                    cohort_by_order.substitute_parameters(substitutions).materialize(),
                    task_keys.substitute_parameters(substitutions).materialize(),
                )
                concrete_inverse = order_points_by_cohort.substitute_parameters(
                    substitutions
                )
                self.assertEqual(
                    tuple(len(points) for points in concrete_inverse.materialize()),
                    (2,) * concrete_batch + (1,) * concrete_batch,
                )

    def test_readiness_cohort_relation_factors_fixed_fanout(self) -> None:
        task_count = sympy.Symbol(
            "task_count",
            integer=True,
            nonnegative=True,
        )
        task_domain = _identify_root_domains((_domain((10, task_count, 1)),))[0]
        task_order = pid_task_order(task_domain, (10,))
        key_domain = CoordinateDomain(
            axis_order=(0,),
            axis_counts_items=((0, 2 * task_count),),
            kind="event",
            identity=0,
            _allow_empty=True,
        )
        task = coordinate_axis_symbol(10)
        keys_by_task = CoordinateRelation(
            source_domain=task_domain,
            target_domain=key_domain,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=((10, 0, task_count, 1),),
                    target_ranges=((0, 2 * task, 2 * task + 2, 1),),
                ),
            ),
        )

        with _forbid_schedule_enumeration():
            cohort = cross_loop_scheduler._readiness_equivalent_cohort_relation(
                task_order,
                keys_by_task,
            )
        self.assertIsNotNone(cohort)
        assert cohort is not None
        cohort_by_order, event_keys_by_cohort = cohort
        order_points_by_cohort = cohort_by_order.converse()
        self.assertIsNotNone(order_points_by_cohort)
        assert order_points_by_cohort is not None
        self.assertTrue(cohort_by_order.is_total_function())
        for concrete_count in (0, 1, 3):
            with self.subTest(concrete_count=concrete_count):
                substitutions = {task_count: concrete_count}
                concrete_keys = event_keys_by_cohort.substitute_parameters(
                    substitutions
                )
                self.assertEqual(
                    concrete_keys.materialize(),
                    tuple(
                        frozenset((2 * index, 2 * index + 1))
                        for index in range(concrete_count)
                    ),
                )
                self.assertEqual(
                    tuple(
                        len(points)
                        for points in order_points_by_cohort.substitute_parameters(
                            substitutions
                        ).materialize()
                    ),
                    (1,) * concrete_count,
                )

    def test_readiness_cohort_relation_declines_partial_task_support(self) -> None:
        task_domain = _identify_root_domains((_domain((10, 3, 1)),))[0]
        task_order = pid_task_order(task_domain, (10,))
        key_domain = _domain((0, 3), kind="event", identity=0)
        task = coordinate_axis_symbol(10)
        partial_keys = CoordinateRelation.point_map(
            task_domain,
            key_domain,
            ((((10, 0, 2, 1),), (task,)),),
        )

        with _forbid_schedule_enumeration():
            self.assertIsNone(
                cross_loop_scheduler._readiness_equivalent_cohort_relation(
                    task_order,
                    partial_keys,
                )
            )

    def test_cohort_interval_end_handles_partial_point_fibers(self) -> None:
        task_domain = _identify_root_domains((_domain((10, 6, 1)),))[0]
        task_order = pid_task_order(task_domain, (10,))
        key_domain = _domain((0, 1), kind="event", identity=0)
        partial_keys = CoordinateRelation.point_map(
            task_domain,
            key_domain,
            (
                (((10, 1, 3, 1),), (sympy.Integer(0),)),
                (((10, 3, 5, 1),), (sympy.Integer(0),)),
            ),
        )
        ordered_keys = task_order.then(partial_keys)
        self.assertIsNotNone(ordered_keys)
        assert ordered_keys is not None

        with _forbid_schedule_enumeration():
            self.assertEqual(
                tuple(
                    cross_loop_scheduler._cohort_interval_end_at_cursor(
                        ordered_keys,
                        cursor,
                    )
                    for cursor in range(6)
                ),
                (1, 5, 5, 5, 5, 6),
            )

    def test_readiness_cohort_relation_declines_invalid_constant_fiber(self) -> None:
        task_domain = _identify_root_domains((_domain((10, 2, 1)),))[0]
        task_order = pid_task_order(task_domain, (10,))

        for key_count, key in (
            (1, sympy.Integer(1)),
            (
                sympy.Symbol("key_count", integer=True, nonnegative=True),
                sympy.Integer(0),
            ),
        ):
            with self.subTest(key_count=key_count, key=key):
                key_domain = CoordinateDomain(
                    axis_order=(0,),
                    axis_counts_items=((0, key_count),),
                    kind="event",
                    identity=0,
                    _allow_empty=True,
                )
                keys_by_task = _full_point_map(
                    task_domain,
                    key_domain,
                    key,
                )

                with _forbid_schedule_enumeration():
                    self.assertIsNone(
                        cross_loop_scheduler._readiness_equivalent_cohort_relation(
                            task_order,
                            keys_by_task,
                        )
                    )

    def test_progress_rejects_final_arrival_continuation_cycle(self) -> None:
        root_domains = _identify_root_domains(
            (_domain((10, 1, 1)), _domain((20, 1, 1)))
        )
        forward = _pointwise_root_readiness_event(root_domains, 0, 1, 0)
        backward = _pointwise_root_readiness_event(root_domains, 1, 0, 1)
        graph = _readiness_graph(root_domains, forward, backward)
        plans = (
            ReadinessCounterPlan(
                forward.producers,
                forward.consumers,
                continuation_consumer_index=0,
            ),
            ReadinessCounterPlan(
                backward.producers,
                backward.consumers,
                continuation_consumer_index=0,
            ),
        )

        with _forbid_schedule_enumeration():
            self.assertFalse(
                cross_loop_scheduler._schedule_is_progress_safe(
                    _schedule(1),
                    graph,
                    plans,
                    frozenset(),
                )
            )

    def test_plan_construction_uses_one_ownership_pipeline(self) -> None:
        dependency_graph = _dependency_graph([[10], [20]])
        root_domains = _identify_root_domains(
            (_domain((10, 3, 1)), _domain((20, 2, 1)))
        )

        with (
            mock.patch.object(
                cross_loop_scheduler,
                "_global_unit_list_schedule",
                wraps=cross_loop_scheduler._global_unit_list_schedule,
            ) as list_schedule,
            mock.patch.object(
                cross_loop_scheduler,
                "_finalize_emitted_synchronization",
                wraps=cross_loop_scheduler._finalize_emitted_synchronization,
            ) as finalize,
            _forbid_schedule_enumeration(),
        ):
            plan = _configured_static_pipeline_plan(
                dependency_graph=dependency_graph,
                root_domains=root_domains,
                axis_geometry=_axis_geometry(root_domains),
                worker_count=4,
            )

        self.assertEqual(finalize.call_count, 1)
        self.assertEqual(list_schedule.call_count, 1)
        self.assertEqual(
            tuple(segment.root for segment in plan.worker_schedule.segments),
            (0, 1),
        )
        self.assertEqual(plan.readiness_counters, ())
        self.assertEqual(plan.root_barrier_edges, frozenset())

    def test_event_frontier_preserves_prepared_readiness_order(self) -> None:
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
        prepared = cross_loop_scheduler._consumer_major_producer_order(
            graph,
            baseline,
            (plan,),
            frozenset(),
            excluded_roots=frozenset(),
        )

        proposed = cross_loop_scheduler._event_frontier_list_schedule(
            graph,
            prepared,
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
        self.assertLessEqual(
            proposed.worker_step_domain.size,
            baseline.worker_step_domain.size,
        )
        validate_worker_schedule(graph, proposed)

        # The selected schedule preserves that root-local event preparation
        # without increasing the occupied-wave horizon.
        scheduled = _global_unit_list_schedule(
            graph,
            prepared,
            (plan,),
            frozenset(),
        )
        self.assertEqual(placement(scheduled, 0, 2)[1], 0)
        self.assertGreater(placement(scheduled, 0, 1)[1], 0)
        self.assertEqual(baseline.worker_step_domain.size, 3)

    def test_event_frontier_materializes_wrapped_woven_root(self) -> None:
        root_domains = _identify_root_domains(
            (
                _domain((10, 4, 1)),
                _domain((20, 16, 1)),
                _domain((30, 2, 1)),
            )
        )
        woven_source = CoordinateDomain(
            axis_order=(40, 41),
            axis_counts_items=((40, 4), (41, 4)),
            kind="task_order",
            identity=root_domains[1].identity,
        )
        inner = coordinate_axis_symbol(40)
        outer = coordinate_axis_symbol(41)
        woven_order = CoordinateRelation.point_map(
            woven_source,
            root_domains[1],
            (
                (
                    ((40, 0, 2, 1), (41, 0, 4, 1)),
                    (2 * outer + inner,),
                ),
                (
                    ((40, 2, 4, 1), (41, 0, 4, 1)),
                    (2 * outer + inner + 6,),
                ),
            ),
        )
        event = _whole_root_readiness_event(root_domains, 1, 2, 0)
        graph = ReadinessGraph(
            root_task_orders=(
                pid_task_order(root_domains[0], root_domains[0].axis_order),
                woven_order,
                pid_task_order(root_domains[2], root_domains[2].axis_order),
            ),
            events=(event,),
        )
        counter = ReadinessCounterPlan(event.producers, event.consumers)
        baseline = _build_baseline_worker_schedule(
            graph.root_domains,
            graph.root_task_orders,
            worker_count=10,
        )

        with _forbid_schedule_enumeration():
            proposed = cross_loop_scheduler._event_frontier_list_schedule(
                graph,
                baseline,
                (counter,),
                frozenset(((0, 1),)),
                pipeline_depth=2,
            )

        self.assertIsNotNone(proposed)
        assert proposed is not None
        self.assertIsNot(proposed, baseline)
        self.assertEqual(
            tuple(segment.resident_slot_interval for segment in proposed.segments),
            ((0, 4), (4, 20), (20, 22)),
        )
        self.assertEqual(len(proposed.segments_for_root(1)), 1)
        self.assertIsNotNone(
            cross_loop_scheduler._packed_schedule_segment_geometry(proposed)
        )
        self.assertEqual(
            task_order(proposed, 1),
            (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15),
        )
        validate_worker_schedule(graph, proposed)

        # If the generic packed reconstruction declines, already-proved
        # adjacent run relations still form one exact segment when each slice
        # has a representable inverse.  This keeps a representational proof
        # decline from cloning the root body at the wave boundary.
        simple_graph = dataclasses.replace(
            graph,
            root_task_orders=(
                graph.root_task_orders[0],
                pid_task_order(root_domains[1], root_domains[1].axis_order),
                graph.root_task_orders[2],
            ),
        )
        simple_baseline = _build_baseline_worker_schedule(
            simple_graph.root_domains,
            simple_graph.root_task_orders,
            worker_count=10,
        )
        packed_relation = cross_loop_scheduler._packed_root_major_task_order_relation
        declined_combined_runs = 0

        def decline_combined_run(*args: Any, **kwargs: Any):
            nonlocal declined_combined_runs
            if (
                args[1].target_domain == root_domains[1]
                and kwargs.get("ordinal_begin") == 0
                and kwargs.get("task_count") == 16
            ):
                declined_combined_runs += 1
                return None
            return packed_relation(*args, **kwargs)

        with (
            mock.patch.object(
                cross_loop_scheduler,
                "_packed_root_major_task_order_relation",
                side_effect=decline_combined_run,
            ),
            _forbid_schedule_enumeration(),
        ):
            relation_union_fallback = (
                cross_loop_scheduler._event_frontier_list_schedule(
                    simple_graph,
                    simple_baseline,
                    (counter,),
                    frozenset(((0, 1),)),
                    pipeline_depth=2,
                )
            )

        self.assertIsNotNone(relation_union_fallback)
        assert relation_union_fallback is not None
        self.assertEqual(declined_combined_runs, 1)
        self.assertEqual(len(relation_union_fallback.segments_for_root(1)), 1)
        self.assertEqual(
            relation_union_fallback.segments_for_root(1)[0].resident_slot_interval,
            (4, 20),
        )
        self.assertEqual(task_order(relation_union_fallback, 1), tuple(range(16)))
        validate_worker_schedule(simple_graph, relation_union_fallback)

    def test_packed_segment_geometry_accepts_split_root_interior_gaps(
        self,
    ) -> None:
        root_domains = _identify_root_domains(
            (_domain((10, 4, 1)), _domain((20, 2, 1)))
        )
        root_orders = _default_root_task_orders(root_domains)

        def segment_stream(
            wave_count: int,
            starts: tuple[int, int, int],
        ) -> tuple[WorkerScheduleSegment, ...]:
            schedule_domain = cross_loop_scheduler._worker_schedule_domain(
                4,
                wave_count,
                (-3, -2, -1),
            )
            result: list[WorkerScheduleSegment] = []
            for root, task_order, ordinal_begin, task_count, first_slot in zip(
                (0, 1, 0),
                (root_orders[0], root_orders[1], root_orders[0]),
                (0, 0, 2),
                (2, 2, 2),
                starts,
                strict=True,
            ):
                relation = cross_loop_scheduler._packed_root_major_task_order_relation(
                    schedule_domain,
                    task_order,
                    sympy.Integer(first_slot),
                    4,
                    ordinal_begin=ordinal_begin,
                    task_count=task_count,
                )
                self.assertIsNotNone(relation)
                assert relation is not None
                result.append(
                    WorkerScheduleSegment(
                        root,
                        relation,
                        0,
                        4,
                        first_slot,
                    )
                )
            return tuple(result)

        gapped = cross_loop_scheduler._packed_schedule_segment_geometry_from_parts(
            4,
            segment_stream(2, (0, 4, 6)),
        )
        self.assertIsNotNone(gapped)
        assert gapped is not None
        self.assertEqual(
            tuple(
                (first_slot, task_count) for _segment, first_slot, task_count in gapped
            ),
            ((0, 2), (4, 2), (6, 2)),
        )
        # A gap between one-shot roots remains on the simpler legacy renderer;
        # the relation renderer is needed only when a split root resumes.
        self.assertIsNone(
            cross_loop_scheduler._packed_schedule_segment_geometry_from_parts(
                2,
                segment_stream(2, (0, 4, 6))[:2],
            )
        )
        self.assertIsNone(
            cross_loop_scheduler._packed_schedule_segment_geometry_from_parts(
                4,
                segment_stream(2, (0, 1, 6)),
            )
        )
        self.assertIsNone(
            cross_loop_scheduler._packed_schedule_segment_geometry_from_parts(
                4,
                segment_stream(2, (4, 0, 6)),
            )
        )
        self.assertIsNone(
            cross_loop_scheduler._packed_schedule_segment_geometry_from_parts(
                4,
                segment_stream(3, (0, 4, 6)),
            )
        )

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
                horizon = cross_loop_scheduler._resident_schedule_occupied_wave_count(
                    schedule
                )
            self.assertEqual(horizon, old_horizon)

    def test_horizon_comparison_matches_concrete(self) -> None:
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
                prepared = cross_loop_scheduler._consumer_major_producer_order(
                    graph,
                    baseline,
                    plans,
                    frozenset(),
                    excluded_roots=frozenset(),
                )
                candidate = cross_loop_scheduler._event_frontier_list_schedule(
                    graph,
                    prepared,
                    plans,
                    frozenset(),
                )
                self.assertIsNotNone(candidate)
                assert candidate is not None
                expected = (
                    candidate
                    if old_horizon(candidate) <= old_horizon(prepared)
                    else prepared
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
                        prepared,
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

    def test_event_frontier_fills_tail_with_newly_released_join(self) -> None:
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
        self.assertEqual(placement(scheduled, 2, 0), (4, 0))
        self.assertEqual(placement(scheduled, 3, 0)[1], 1)
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
        # The barrier consumer may wait in the producer's terminal wave on a
        # disjoint worker; the segment-precedence proof rejects cyclic cases.
        self.assertEqual(placement(scheduled, 1, 0), (2, 1))
        self.assertTrue(
            cross_loop_scheduler._schedule_is_progress_safe(
                scheduled,
                graph,
                (),
                frozenset(((0, 1),)),
            )
        )
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
        self.assertEqual(placement(scheduled, 1, 0), (1, 1))
        self.assertTrue(
            cross_loop_scheduler._schedule_is_progress_safe(
                scheduled,
                graph,
                (plan,),
                frozenset(),
            )
        )
        validate_worker_schedule(graph, scheduled)

    def test_merge_relations_by_root_batches_exact_disjoint_union(self) -> None:
        source_domain = _domain((10, 4, 1), identity=0)
        target_domain = _domain((0, 4), kind="event", identity=0)
        source = coordinate_axis_symbol(10)
        relations = tuple(
            CoordinateRelation.point_map(
                source_domain,
                target_domain,
                (
                    (
                        ((10, begin, end, 1),),
                        (source,),
                    ),
                ),
            )
            for begin, end in ((0, 2), (2, 4))
        )
        converses = tuple(relation.converse() for relation in relations)
        self.assertTrue(all(converse is not None for converse in converses))
        expected = relations[0].union(relations[1])
        self.assertIsNotNone(expected)
        assert expected is not None

        with mock.patch.object(
            CoordinateRelation,
            "union",
            side_effect=AssertionError("batch merge used sequential union"),
        ):
            merged = cross_loop_scheduler._merge_relations_by_root(
                ((7, relations[0]), (7, relations[1])),
            )

        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertEqual(tuple(root for root, _relation in merged), (7,))
        merged_relation = merged[0][1]
        self.assertEqual(merged_relation.materialize(), expected.materialize())
        merged_converse = tile_dependency._memoized_exact_converse(merged_relation)
        self.assertIsNotNone(merged_converse)
        assert merged_converse is not None
        expected_converse = expected.converse()
        self.assertIsNotNone(expected_converse)
        assert expected_converse is not None
        self.assertEqual(
            merged_converse.materialize(),
            expected_converse.materialize(),
        )
        with mock.patch.object(cross_loop_scheduler, "_MAX_GLOBAL_LIST_WORK", 5):
            forward_only = cross_loop_scheduler._merge_relations_by_root(
                ((7, relations[0]), (7, relations[1])),
            )
        self.assertIsNotNone(forward_only)
        assert forward_only is not None
        self.assertEqual(forward_only[0][1].materialize(), expected.materialize())
        self.assertIsNone(tile_dependency._memoized_exact_converse(forward_only[0][1]))

    def test_compose_exact_relations_rejects_domain_mismatch(self) -> None:
        source = _domain((10, 2), identity=0)
        middle = _domain((20, 2), identity=1)
        other_middle = _domain((21, 2), identity=2)
        target = _domain((30, 2), kind="event", identity=0)
        first = _full_point_map(
            source,
            middle,
            coordinate_axis_symbol(10),
        )
        following = _full_point_map(
            other_middle,
            target,
            coordinate_axis_symbol(21),
        )

        self.assertIsNone(
            cross_loop_scheduler._compose_exact_relations(first, following)
        )

    def test_identity_on_relation_source_support_rejects_clipped_target(
        self,
    ) -> None:
        source = _domain((10, 2), kind="worker")
        target = _domain((20, 1), kind="event", identity=0)
        clipped = _full_point_map(
            source,
            target,
            coordinate_axis_symbol(10),
        )

        self.assertIsNone(
            cross_loop_scheduler._identity_on_relation_source_support(clipped)
        )

    def test_static_producer_preflight_handles_wide_ordinary_producers(
        self,
    ) -> None:
        producer_count = 257
        root_domains = _identify_root_domains(
            tuple(_domain((10 + root, 1, 1)) for root in range(producer_count + 1))
        )
        event = _whole_consumer_join_readiness_event(
            root_domains,
            tuple(range(producer_count)),
            producer_count,
            0,
        )
        graph = _readiness_graph(root_domains, event)
        queries = cross_loop_scheduler._readiness_producer_queries(event.producers)
        self.assertIsNotNone(queries)
        assert queries is not None

        with _forbid_schedule_enumeration():
            preflight = cross_loop_scheduler._static_producer_contraction_preflight(
                graph,
                queries,
                {},
                cross_loop_scheduler._MAX_GLOBAL_LIST_WORK,
            )
        self.assertIsNotNone(preflight)
        assert preflight is not None
        self.assertGreater(preflight, len(queries))
        with _forbid_schedule_enumeration():
            contracted = cross_loop_scheduler._contract_static_producer_relations(
                graph,
                queries,
                {},
                preflight=preflight,
            )
            declined = cross_loop_scheduler._static_producer_contraction_preflight(
                graph,
                queries,
                {},
                preflight - 1,
            )

        self.assertIsNotNone(contracted)
        assert contracted is not None
        self.assertEqual(
            tuple(root for root, _relation in contracted),
            tuple(range(producer_count)),
        )
        self.assertEqual(contracted[0][1], queries[0][2])
        self.assertEqual(contracted[-1][1], queries[-1][2])
        self.assertIsNone(declined)

    def test_static_producer_contraction_memoizes_reconvergent_diamond(
        self,
    ) -> None:
        graph, _prepared, plans = _branching_continuation_problem()
        continuations = cross_loop_scheduler._emitted_final_arrival_continuations(
            graph,
            plans,
        )
        self.assertIsNotNone(continuations)
        assert continuations is not None
        continuation_by_root = cross_loop_scheduler._continuations_by_consumer_root(
            graph,
            continuations,
        )
        queries = cross_loop_scheduler._readiness_producer_queries(
            plans[-1].producers,
        )
        self.assertIsNotNone(queries)
        assert queries is not None
        preflight = cross_loop_scheduler._static_producer_contraction_preflight(
            graph,
            queries,
            continuation_by_root,
            cross_loop_scheduler._MAX_GLOBAL_LIST_WORK,
        )
        self.assertIsNotNone(preflight)
        assert preflight is not None
        merge_relations = cross_loop_scheduler._merge_relations_by_root

        with (
            _forbid_schedule_enumeration(),
            mock.patch.object(
                cross_loop_scheduler,
                "_merge_relations_by_root",
                wraps=merge_relations,
            ) as merge_spy,
        ):
            contracted = cross_loop_scheduler._contract_static_producer_relations(
                graph,
                queries,
                continuation_by_root,
                preflight=preflight,
            )

        self.assertIsNotNone(contracted)
        assert contracted is not None
        self.assertEqual(tuple(root for root, _relation in contracted), (0, 1))
        # Five unique continuation nodes plus the final producer-set merge are
        # computed once. The two reconvergent paths through roots 2 and 3 do
        # not repeat their relation unions inside this local transaction.
        self.assertEqual(merge_spy.call_count, 6)

    def test_static_producer_contraction_is_converse_cache_invariant(self) -> None:
        graph, _prepared, _plans = _branching_continuation_problem()
        cold_graph = pickle.loads(pickle.dumps(graph))
        warm_graph = pickle.loads(pickle.dumps(graph))
        continuation_by_root = {
            2: FinalArrivalContinuation(0, 0),
            3: FinalArrivalContinuation(1, 0),
            4: FinalArrivalContinuation(2, 0),
            5: FinalArrivalContinuation(3, 0),
            6: FinalArrivalContinuation(4, 0),
        }
        self.assertTrue(
            all(
                tile_dependency._memoized_exact_converse(producer.producers_by_key)
                is None
                for event in cold_graph.events
                for producer in event.producers
            )
        )
        for event in warm_graph.events:
            for producer in event.producers:
                self.assertIsNotNone(producer.producers_by_key.converse())
            for consumer in event.consumers:
                self.assertIsNotNone(consumer.keys_by_consumer.converse())

        cold_queries = cross_loop_scheduler._readiness_producer_queries(
            cold_graph.events[-1].producers,
        )
        warm_queries = cross_loop_scheduler._readiness_producer_queries(
            warm_graph.events[-1].producers,
        )
        self.assertIsNotNone(cold_queries)
        self.assertIsNotNone(warm_queries)
        assert cold_queries is not None and warm_queries is not None
        self.assertEqual(cold_queries, warm_queries)
        # Query resolution deterministically seeds the authoritative
        # publication inverse even when no unrelated earlier proof warmed it.
        self.assertTrue(
            all(
                tile_dependency._memoized_exact_converse(readiness_keys) is not None
                for _root, _site_id, readiness_keys in cold_queries
            )
        )

        with _forbid_schedule_enumeration():
            cold_preflight = (
                cross_loop_scheduler._static_producer_contraction_preflight(
                    cold_graph,
                    cold_queries,
                    continuation_by_root,
                    cross_loop_scheduler._MAX_GLOBAL_LIST_WORK,
                )
            )
            warm_preflight = (
                cross_loop_scheduler._static_producer_contraction_preflight(
                    warm_graph,
                    warm_queries,
                    continuation_by_root,
                    cross_loop_scheduler._MAX_GLOBAL_LIST_WORK,
                )
            )
        self.assertIsNotNone(cold_preflight)
        self.assertIsNotNone(warm_preflight)
        assert cold_preflight is not None and warm_preflight is not None
        self.assertEqual(cold_preflight, warm_preflight)

        with _forbid_schedule_enumeration():
            cold_contracted = cross_loop_scheduler._contract_static_producer_relations(
                cold_graph,
                cold_queries,
                continuation_by_root,
                preflight=cold_preflight,
            )
            warm_contracted = cross_loop_scheduler._contract_static_producer_relations(
                warm_graph,
                warm_queries,
                continuation_by_root,
                preflight=warm_preflight,
            )
        self.assertEqual(cold_contracted, warm_contracted)
        self.assertIsNotNone(cold_contracted)
        assert cold_contracted is not None
        self.assertEqual(
            tuple((root, relation.materialize()) for root, relation in cold_contracted),
            ((0, (frozenset((0,)),)), (1, (frozenset((0,)),))),
        )

    def test_static_producer_contraction_handles_deep_one_piece_chain(
        self,
    ) -> None:
        continuation_count = 1_001
        root_domains = _identify_root_domains(
            tuple(_domain((10 + root, 1, 1)) for root in range(continuation_count + 2))
        )
        events = tuple(
            _whole_root_readiness_event(
                root_domains,
                producer_root=consumer_root - 1,
                consumer_root=consumer_root,
                event_id=consumer_root - 1,
            )
            for consumer_root in range(1, continuation_count + 2)
        )
        graph = _readiness_graph(root_domains, *events)
        continuation_by_root = {
            root: FinalArrivalContinuation(root - 1, 0)
            for root in range(1, continuation_count + 1)
        }
        queries = cross_loop_scheduler._readiness_producer_queries(
            events[-1].producers,
        )
        self.assertIsNotNone(queries)
        assert queries is not None

        with _forbid_schedule_enumeration():
            preflight = cross_loop_scheduler._static_producer_contraction_preflight(
                graph,
                queries,
                continuation_by_root,
                cross_loop_scheduler._MAX_GLOBAL_LIST_WORK,
            )
        self.assertIsNotNone(preflight)
        assert preflight is not None
        self.assertLess(preflight, cross_loop_scheduler._MAX_GLOBAL_LIST_WORK)
        merge_relations = cross_loop_scheduler._merge_relations_by_root
        with (
            _forbid_schedule_enumeration(),
            mock.patch.object(
                cross_loop_scheduler,
                "_merge_relations_by_root",
                wraps=merge_relations,
            ) as merge_spy,
        ):
            contracted = cross_loop_scheduler._contract_static_producer_relations(
                graph,
                queries,
                continuation_by_root,
                preflight=preflight,
            )

        self.assertIsNotNone(contracted)
        assert contracted is not None
        self.assertEqual(tuple(root for root, _relation in contracted), (0,))
        self.assertEqual(merge_spy.call_count, continuation_count + 1)
        self.assertIsNotNone(contracted[0][1].converse())

    def test_event_frontier_continuation_budget_declines_transactionally(
        self,
    ) -> None:
        graph, prepared, plans = _branching_continuation_problem()

        with (
            _forbid_schedule_enumeration(),
            mock.patch.object(cross_loop_scheduler, "_MAX_GLOBAL_LIST_WORK", 64),
            mock.patch.object(
                cross_loop_scheduler,
                "_contract_static_producer_relations",
                side_effect=AssertionError("contraction ran after preflight decline"),
            ) as contraction_spy,
            mock.patch.object(
                cross_loop_scheduler,
                "_root_schema_criticality",
                side_effect=AssertionError("chooser ran after contraction decline"),
            ) as criticality,
        ):
            scheduled = cross_loop_scheduler._event_frontier_list_schedule(
                graph,
                prepared,
                plans,
                frozenset(),
                pipeline_depth=2,
            )

        self.assertIs(scheduled, prepared)
        contraction_spy.assert_not_called()
        criticality.assert_not_called()

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
        # Each continuation runs on its final producer strand, allowing the
        # corresponding sink to fill a later lane in the same wave. Exact
        # event cohorts need not wait for the unrelated producer key.
        self.assertEqual(
            {placement(scheduled, root, task)[1] for root in (0, 3) for task in (0, 1)},
            {0},
        )
        self.assertTrue(
            all(
                placement(scheduled, 0, task)[0] < placement(scheduled, 3, task)[0]
                for task in (0, 1)
            )
        )
        self.assertEqual(placement(scheduled, 2, 0), (0, 1))
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
        schedule = _baseline_worker_schedule(
            (domain,),
            worker_count=2,
            root_task_orders=(order,),
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
        worker_schedule = _baseline_worker_schedule(
            (producer_domain, consumer_domain),
            worker_count=2,
            root_task_orders=root_task_orders,
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
        worker_schedule = _baseline_worker_schedule(
            (producer_domain, consumer_domain),
            worker_count=2,
            root_task_orders=root_task_orders,
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
        worker_schedule = _baseline_worker_schedule(
            (producer_domain, consumer_domain),
            worker_count=2,
            root_task_orders=root_task_orders,
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
                        consumer.keys_by_consumer.canonical_single_valued() is not None
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

    def test_parameterized_counter_is_not_emitted(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        producer_domain = CoordinateDomain(
            (10,),
            ((10, batch),),
            kind="site",
            identity=0,
        )
        consumer_domain = CoordinateDomain(
            (20,),
            ((20, batch),),
            kind="site",
            identity=1,
        )
        key_domain = CoordinateDomain(
            (0,),
            ((0, batch),),
            kind="event",
            identity=0,
        )
        counter = ReadinessCounterPlan(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=CoordinateRelation.point_map(
                        key_domain,
                        producer_domain,
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
                    consumer_root=1,
                    keys_by_consumer=CoordinateRelation.point_map(
                        consumer_domain,
                        key_domain,
                        (
                            (
                                ((20, 0, batch, 1),),
                                (coordinate_axis_symbol(20),),
                            ),
                        ),
                    ),
                ),
            ),
        )

        self.assertTrue(
            cross_loop_scheduler._supports_exact_counter_plan_lowering(
                counter,
                (producer_domain, consumer_domain),
            )
        )
        self.assertFalse(
            cross_loop_scheduler._supports_emitted_counter_plan_lowering(
                counter,
                (producer_domain, consumer_domain),
            )
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
        self.assertEqual(
            cross_loop_scheduler.root_barrier_publication_plan(
                schedule,
                0,
            ).participant_intervals,
            ((0, 4),),
        )
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

    def test_task_order_slice_preserves_rank_three_leading_cohort(self) -> None:
        batch = sympy.Symbol("batch", integer=True, positive=True)
        query = sympy.Symbol("query", integer=True, positive=True)
        domain = CoordinateDomain(
            (10, 11, 12),
            ((10, batch), (11, query), (12, 2)),
            identity=0,
        )
        # The configured traversal is H-fastest, then B, then Q.  This is the
        # shape that exposed the old rank-two-only slicing assumption.
        task_order = pid_task_order(domain, (12, 10, 11))
        task_count = 2 * batch * query

        with _forbid_schedule_enumeration():
            prefix = _task_order_slice(task_order, 0, 2)
            suffix = _task_order_slice(task_order, 2, task_count - 2)

        self.assertIsNotNone(prefix)
        self.assertIsNotNone(suffix)
        assert prefix is not None and suffix is not None
        self.assertEqual(prefix.source_support_cardinality(), 2)
        self.assertEqual(suffix.source_support_cardinality(), task_count - 2)
        prefix_inverse = tile_dependency._memoized_exact_converse(prefix)
        suffix_inverse = tile_dependency._memoized_exact_converse(suffix)
        self.assertIsNotNone(prefix_inverse)
        self.assertIsNotNone(suffix_inverse)
        assert prefix_inverse is not None and suffix_inverse is not None
        self.assertTrue(prefix_inverse.is_single_valued())
        self.assertTrue(suffix_inverse.is_single_valued())

        for concrete_batch, concrete_query in ((1, 1), (1, 2), (2, 1), (2, 3)):
            with self.subTest(batch=concrete_batch, query=concrete_query):
                substitutions = {batch: concrete_batch, query: concrete_query}
                concrete_order = task_order.substitute_parameters(substitutions)
                expected = concrete_order.materialize()
                concrete_prefix = prefix.substitute_parameters(substitutions)
                concrete_suffix = suffix.substitute_parameters(substitutions)
                self.assertEqual(concrete_prefix.materialize(), expected[:2])
                self.assertEqual(concrete_suffix.materialize(), expected[2:])
                self.assertEqual(
                    concrete_prefix.materialize() + concrete_suffix.materialize(),
                    expected,
                )

    def test_task_order_slice_preserves_grouped_rank_three_leading_cohort(
        self,
    ) -> None:
        batch = sympy.Symbol("batch", integer=True, positive=True)
        query = sympy.Symbol("query", integer=True, positive=True)
        domain = CoordinateDomain(
            (10, 11, 12, 13),
            ((10, 5), (11, 3), (12, batch), (13, query)),
            identity=0,
        )
        task_order = pid_task_order(
            domain,
            domain.axis_order,
            l2_group_size=2,
        )
        cohort_count = sympy.Integer(15)
        task_count = task_order.source_domain.size_expr
        worker_count = 148
        first_slot = sympy.Integer(11)
        schedule_domain = cross_loop_scheduler._worker_schedule_domain(
            worker_count,
            cross_loop_scheduler._ceildiv_nonnegative_expression(
                first_slot + task_count,
                worker_count,
            ),
            (-3, -2, -1),
        )

        with _forbid_schedule_enumeration():
            prefix = _task_order_slice(task_order, 0, cohort_count)
            suffix = _task_order_slice(
                task_order,
                cohort_count,
                task_count - cohort_count,
            )
            packed_prefix = cross_loop_scheduler._packed_root_major_task_order_relation(
                schedule_domain,
                task_order,
                first_slot,
                worker_count,
                ordinal_begin=0,
                task_count=cohort_count,
            )
            packed_suffix = cross_loop_scheduler._packed_root_major_task_order_relation(
                schedule_domain,
                task_order,
                first_slot + cohort_count,
                worker_count,
                ordinal_begin=cohort_count,
                task_count=task_count - cohort_count,
            )

        self.assertIsNotNone(prefix)
        self.assertIsNotNone(suffix)
        self.assertIsNotNone(packed_prefix)
        self.assertIsNotNone(packed_suffix)
        assert prefix is not None and suffix is not None
        assert packed_prefix is not None and packed_suffix is not None
        self.assertEqual(prefix.source_support_cardinality(), cohort_count)
        self.assertEqual(
            suffix.source_support_cardinality(),
            task_count - cohort_count,
        )
        self.assertTrue(prefix.is_total_function())
        self.assertTrue(suffix.is_total_function())
        prefix_inverse = prefix.converse()
        suffix_inverse = suffix.converse()
        self.assertIsNotNone(prefix_inverse)
        self.assertIsNotNone(suffix_inverse)
        assert prefix_inverse is not None and suffix_inverse is not None
        self.assertTrue(prefix_inverse.is_single_valued())
        self.assertTrue(suffix_inverse.is_single_valued())
        with _forbid_schedule_enumeration():
            WorkerSchedule(
                worker_count,
                (
                    WorkerScheduleSegment(0, packed_prefix, 0, worker_count, 0),
                    WorkerScheduleSegment(0, packed_suffix, 0, worker_count, 0),
                ),
            )

        for concrete_batch, concrete_query in ((1, 1), (1, 2), (2, 1), (2, 3)):
            with self.subTest(batch=concrete_batch, query=concrete_query):
                substitutions = {batch: concrete_batch, query: concrete_query}
                expected = task_order.substitute_parameters(substitutions).materialize()
                concrete_prefix = prefix.substitute_parameters(
                    substitutions
                ).materialize()
                concrete_suffix = suffix.substitute_parameters(
                    substitutions
                ).materialize()
                self.assertEqual(concrete_prefix, expected[:15])
                self.assertEqual(concrete_suffix, expected[15:])
                self.assertEqual(concrete_prefix + concrete_suffix, expected)

        zero_capable_batch = sympy.Symbol(
            "zero_capable_batch",
            integer=True,
            nonnegative=True,
        )
        zero_capable_query = sympy.Symbol(
            "zero_capable_query",
            integer=True,
            nonnegative=True,
        )
        zero_capable_domain = CoordinateDomain(
            (20, 21, 22, 23),
            (
                (20, 5),
                (21, 3),
                (22, zero_capable_batch),
                (23, zero_capable_query),
            ),
            identity=1,
        )
        zero_capable_order = pid_task_order(
            zero_capable_domain,
            zero_capable_domain.axis_order,
            l2_group_size=2,
        )
        with _forbid_schedule_enumeration():
            self.assertIsNone(_task_order_slice(zero_capable_order, 0, 15))

    def test_task_order_slice_preflights_rank_three_cohort_piece_budget(self) -> None:
        batch = sympy.Symbol("batch", integer=True, positive=True)
        query = sympy.Symbol("query", integer=True, positive=True)
        domain = CoordinateDomain(
            (10, 11, 12),
            ((10, batch), (11, query), (12, 2)),
            identity=0,
        )
        task_order = pid_task_order(domain, (12, 10, 11))
        original_budget_check = tile_dependency._relation_product_is_within_budget

        def reject_rank_three_suffix(*factor_sizes: int) -> bool:
            if factor_sizes == (2, 3):
                return False
            return original_budget_check(*factor_sizes)

        with (
            mock.patch.object(
                tile_dependency,
                "_relation_product_is_within_budget",
                side_effect=reject_rank_three_suffix,
            ) as budget_check,
            mock.patch.object(
                cross_loop_scheduler,
                "_flat_domain_index_expression",
                side_effect=AssertionError(
                    "over-budget cohort must decline before slab construction"
                ),
            ),
            _forbid_schedule_enumeration(),
        ):
            self.assertIsNotNone(
                _task_order_slice(
                    task_order,
                    2,
                    2 * batch * query - 2,
                )
            )
        self.assertIn(mock.call(2, 3), budget_check.call_args_list)

    def test_packed_relation_reuses_constructive_interval_proof(self) -> None:
        batch = sympy.Symbol("batch", integer=True, positive=True)
        worker_count = 148
        first_slot = 2 * batch
        left_count = sympy.Integer(3)
        right_count = 2 * batch + 1
        final_slot = first_slot + left_count + right_count
        schedule_domain = cross_loop_scheduler._worker_schedule_domain(
            worker_count,
            cross_loop_scheduler._ceildiv_nonnegative_expression(
                final_slot,
                worker_count,
            ),
            (-3, -2, -1),
        )
        left_domain, right_domain = _identify_root_domains(
            (
                _domain((10, left_count, 1)),
                _domain((20, right_count, 1)),
            )
        )
        left = cross_loop_scheduler._packed_root_major_task_order_relation(
            schedule_domain,
            pid_task_order(left_domain, left_domain.axis_order),
            first_slot,
            worker_count,
        )
        right = cross_loop_scheduler._packed_root_major_task_order_relation(
            schedule_domain,
            pid_task_order(right_domain, right_domain.axis_order),
            first_slot + left_count,
            worker_count,
        )
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        assert left is not None and right is not None

        with (
            mock.patch.object(
                tile_dependency,
                "_source_boxes_are_disjoint",
                side_effect=AssertionError("must reuse the packed interval proof"),
            ),
            mock.patch.object(
                CoordinateRelation,
                "is_single_valued",
                side_effect=AssertionError("must reuse the construction proof"),
            ),
        ):
            self.assertTrue(left.has_disjoint_source_support(right))
            self.assertEqual(left.source_support_cardinality(), left_count)
            self.assertEqual(right.source_support_cardinality(), right_count)
            self.assertEqual(
                tile_dependency._dense_linear_source_support_interval(
                    left,
                    (-2, -1),
                ),
                (first_slot, first_slot + left_count),
            )
            self.assertEqual(
                tile_dependency._dense_linear_source_support_interval(
                    right,
                    (-2, -1),
                ),
                (first_slot + left_count, final_slot),
            )

        for concrete_batch in (1, 73, 74, 75):
            with self.subTest(batch=concrete_batch):
                substitutions = {batch: concrete_batch}
                concrete_left = left.substitute_parameters(substitutions)
                concrete_right = right.substitute_parameters(substitutions)
                concrete_first = 2 * concrete_batch
                self.assertEqual(
                    tile_dependency._dense_linear_source_support_interval(
                        concrete_left,
                        (-2, -1),
                    ),
                    (concrete_first, concrete_first + 3),
                )
                self.assertEqual(
                    tile_dependency._dense_linear_source_support_interval(
                        concrete_right,
                        (-2, -1),
                    ),
                    (
                        concrete_first + 3,
                        concrete_first + 3 + 2 * concrete_batch + 1,
                    ),
                )

        concrete_schedule = schedule_domain.substitute_parameters({batch: 1})
        overlap_domain = _identify_root_domains((_domain((30, 2, 1)),))[0]
        overlap = cross_loop_scheduler._packed_root_major_task_order_relation(
            concrete_schedule,
            pid_task_order(overlap_domain, overlap_domain.axis_order),
            4,
            worker_count,
        )
        self.assertIsNotNone(overlap)
        assert overlap is not None
        concrete_left = left.substitute_parameters({batch: 1})
        self.assertFalse(concrete_left.has_disjoint_source_support(overlap))

        mismatched_worker_domain = cross_loop_scheduler._worker_schedule_domain(
            8,
            2,
            (-3, -2, -1),
        )
        with self.assertRaisesRegex(ValueError, "disagrees with its worker domain"):
            cross_loop_scheduler._packed_root_major_relation(
                mismatched_worker_domain,
                left_domain,
                sympy.Integer(0),
                4,
            )

        invalid_stage_domain = CoordinateDomain(
            (-3, -2, -1),
            ((-3, 1), (-2, 4), (-1, 1)),
            kind="worker",
        )
        invalid_stage_relation = CoordinateRelation.point_map(
            invalid_stage_domain,
            _domain((40, 4, 1)),
            (
                (
                    ((-3, 1, 2, 1), (-2, 0, 4, 1), (-1, 0, 1, 1)),
                    (coordinate_axis_symbol(-2),),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "out-of-domain projection"):
            tile_dependency._remember_dense_source_support_interval(
                invalid_stage_relation,
                (-2, -1),
                0,
                4,
            )

        two_axis_domain = _domain((50, 2, 1), (51, 2, 1))
        with self.assertRaisesRegex(ValueError, "must permute"):
            cross_loop_scheduler._packed_root_major_relation(
                cross_loop_scheduler._worker_schedule_domain(
                    4,
                    1,
                    (-3, -2, -1),
                ),
                two_axis_domain,
                sympy.Integer(0),
                4,
                (50, 50, 51),
            )

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
            unaligned = _task_order_slice(task_order, 1, 1184)
        self.assertIsNotNone(unaligned)
        assert unaligned is not None
        self.assertLessEqual(len(unaligned.pieces), 15)
        self.assertIsNotNone(unaligned.converse())
        self.assertEqual(
            unaligned.materialize(),
            task_order.materialize()[1:1185],
        )
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

        self.assertEqual(task_at(schedule, 0, 0), (0, 0))
        self.assertEqual(task_at(schedule, 1, 1), (1, 0))

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

    def test_relation_schedule_root_publication_uses_final_worker_occurrence(
        self,
    ) -> None:
        root_domains = _identify_root_domains(
            (
                _domain((10, 6, 1)),
                _domain((20, 2, 1)),
                _domain((30, 2, 1)),
            )
        )
        root_task_orders = _default_root_task_orders(root_domains)
        first_prefix = _task_order_slice(root_task_orders[0], 0, 4)
        first_suffix = _task_order_slice(root_task_orders[0], 4, 2)
        assert first_prefix is not None and first_suffix is not None
        schedule = _schedule(
            4,
            _segment(0, first_prefix, workers=(0, 4), dispatch_offset=0),
            _segment(
                1,
                root_task_orders[1],
                workers=(0, 2),
                dispatch_offset=2,
            ),
            _segment(0, first_suffix, workers=(2, 2), dispatch_offset=2),
            _segment(
                2,
                root_task_orders[2],
                workers=(0, 2),
                dispatch_offset=4,
            ),
        )
        self.assertIsNone(cross_loop_scheduler._root_major_schedule_geometry(schedule))
        self.assertIsNotNone(
            cross_loop_scheduler._packed_schedule_segment_geometry(schedule)
        )
        plan = cross_loop_scheduler.StaticPipelinePlan(
            worker_schedule=schedule,
            root_task_orders=root_task_orders,
            readiness_counters=(),
            root_barrier_edges=frozenset(((0, 2),)),
        )

        publication = plan.root_barrier_publication_plans[0]
        self.assertIsNotNone(publication)
        assert publication is not None
        self.assertEqual(publication.participant_intervals, ((0, 4),))
        self.assertEqual(publication.resident_arrival_count, 4)
        self.assertEqual(
            tuple(
                (item.segment_index, item.worker_intervals)
                for item in publication.publications
            ),
            ((0, ((0, 2),)), (2, ((2, 4),))),
        )

        publication_segment_by_worker: dict[int, int] = {}
        for item in publication.publications:
            for begin, end in item.worker_intervals:
                for worker in range(begin, end):
                    self.assertNotIn(worker, publication_segment_by_worker)
                    publication_segment_by_worker[worker] = item.segment_index
        self.assertEqual(set(publication_segment_by_worker), set(range(4)))
        for worker, publication_segment in publication_segment_by_worker.items():
            final_occurrence = max(
                segment_index
                for segment_index, segment in enumerate(schedule.segments)
                if segment.root == 0
                and any(
                    begin <= worker < end for begin, end in segment.worker_intervals()
                )
            )
            self.assertEqual(publication_segment, final_occurrence)

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

    def test_baseline_preserves_piecewise_configured_orders(self) -> None:
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
                schedule = _baseline_worker_schedule(
                    (domain,),
                    worker_count=4,
                    root_task_orders=(configured_order,),
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

    def test_static_pipeline_rejects_parameterized_task_capacity(self) -> None:
        task_count = sympy.Symbol("task_count", integer=True, nonnegative=True)
        dependency_graph = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", shape=(8192,), block_ids=(10,)),
            _access(root=1, kind="load", shape=(8192,), block_ids=(20,)),
        )
        root_domains = (
            CoordinateDomain((10,), ((10, task_count),), ((10, 16),)),
            CoordinateDomain((20,), ((20, task_count),), ((20, 16),)),
        )

        with (
            _forbid_schedule_enumeration(),
            self.assertRaisesRegex(exc.InvalidConfig, "fixed task capacity"),
        ):
            _configured_static_pipeline_plan(
                dependency_graph=dependency_graph,
                root_domains=root_domains,
                axis_geometry={
                    10: (task_count, 16),
                    20: (task_count, 16),
                },
                worker_count=4,
            )

    def test_unproved_fine_mapping_uses_root_barrier(self) -> None:
        task_count = 4
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
        self.assertIsNotNone(
            cross_loop_scheduler._root_major_schedule_geometry(plan.worker_schedule)
        )

    def test_symbolic_multi_consumer_counter_is_not_emitted(self) -> None:
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
            self.assertFalse(
                cross_loop_scheduler._supports_emitted_counter_plan_lowering(
                    counter,
                    (producer_domain, consumer_domain),
                )
            )
            self.assertFalse(
                counter.producers[0].producers_by_key.is_positional_bijection()
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
            self.assertFalse(
                cross_loop_scheduler._supports_emitted_counter_plan_lowering(
                    one_producer_counter,
                    (one_producer_domain, consumer_domain),
                )
            )
            self.assertFalse(
                one_producer_counter.consumers[
                    0
                ].keys_by_consumer.is_positional_bijection()
            )

    def test_symbolic_partial_consumer_counter_is_not_emitted(self) -> None:
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
            self.assertFalse(
                cross_loop_scheduler._supports_emitted_counter_plan_lowering(
                    counter,
                    (producer_domain, consumer_domain),
                )
            )
            self.assertFalse(
                counter.consumers[0].keys_by_consumer.is_positional_bijection()
            )

    def test_symbolic_multi_axis_counter_is_not_emitted(self) -> None:
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
            self.assertFalse(
                cross_loop_scheduler._supports_emitted_counter_plan_lowering(
                    counter,
                    (producer_domain, consumer_domain),
                )
            )
            self.assertEqual(
                len(counter.producers[0].producers_by_key.source_domain.axis_order),
                2,
            )

    def test_symbolic_nonuniform_counter_is_not_emitted(self) -> None:
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
            self.assertEqual(
                cross_loop_scheduler._arrival_count_bounds(counter.producers),
                (1, 2),
            )
            self.assertFalse(
                cross_loop_scheduler._supports_emitted_counter_plan_lowering(
                    counter,
                    (producer_domain, consumer_domain),
                )
            )
            self.assertEqual(len(arrival_count.pieces), 2)

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
        readiness_graph = _readiness_graph(
            (producer_root, consumer_root),
            ReadinessEvent(counter.producers, counter.consumers),
            obligations_by_root_pair=(((0, 1), frozenset((obligation,))),),
        )

        with _forbid_schedule_enumeration():
            self.assertTrue(counter.parameter_symbols)
            self.assertEqual(
                cross_loop_scheduler._arrival_count_bounds(counter.producers),
                (1, 1),
            )
            self.assertTrue(
                cross_loop_scheduler._supports_exact_counter_plan_lowering(
                    counter,
                    (producer_root, consumer_root),
                )
            )
            self.assertFalse(
                cross_loop_scheduler._supports_emitted_counter_plan_lowering(
                    counter,
                    (producer_root, consumer_root),
                )
            )
            counters, barriers = cross_loop_scheduler._finalize_emitted_synchronization(
                readiness_graph=readiness_graph,
                readiness_counters=(counter,),
            )

        self.assertEqual(counters, ())
        self.assertEqual(barriers, frozenset(((0, 1),)))
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
            with self.assertRaisesRegex(ValueError, "readiness state is parameterized"):
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

    def test_pipeline_derives_continuation_candidates_once(self) -> None:
        for task_count in (4, 7):
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
            self.assertIsNone(counter.continuation_consumer_index)
            self.assertEqual(counter.uniform_arrival_count(), 2)
            self.assertTrue(plan.worker_schedule.segments_for_root(1))

    def test_unsupported_access_scale_uses_root_barrier(self) -> None:
        task_count = 4
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
            traversal = _root_schedule_traversal(schedule.segments, reference)
        self.assertIsNotNone(traversal)
        assert traversal is not None
        self.assertFalse(traversal.matches_reference)

    def test_source_ticket_order_is_a_symbolic_bijection(self) -> None:
        (domain,) = _identify_root_domains((_domain((10, 5, 1)),))
        reference = pid_task_order(domain, domain.axis_order)
        readiness_graph = ReadinessGraph(
            root_task_orders=(reference,),
            events=(),
        )
        schedule = cross_loop_scheduler._with_source_ticket_schedule_segment(
            _baseline_worker_schedule(readiness_graph.root_domains, worker_count=2),
            readiness_graph.root_task_orders,
            0,
        )
        self.assertIsNotNone(schedule)
        assert schedule is not None

        with _forbid_schedule_enumeration():
            source_segment = cross_loop_scheduler._source_ticket_schedule_segment(
                schedule
            )
            assert source_segment is not None
            logical_to_ticket = cross_loop_scheduler._source_segment_ticket_order(
                source_segment
            )
        self.assertIsNotNone(logical_to_ticket)
        assert logical_to_ticket is not None
        self.assertEqual(
            tuple(next(iter(ticket)) for ticket in logical_to_ticket.materialize()),
            (0, 1, 2, 3, 4),
        )

    def test_source_ticket_order_preserves_l2_pid_mapping(self) -> None:
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
        schedule = cross_loop_scheduler._with_source_ticket_schedule_segment(
            _baseline_worker_schedule(readiness_graph.root_domains, worker_count=2),
            readiness_graph.root_task_orders,
            0,
        )
        self.assertIsNotNone(schedule)
        assert schedule is not None

        with _forbid_schedule_enumeration():
            source_segment = cross_loop_scheduler._source_ticket_schedule_segment(
                schedule
            )
            assert source_segment is not None
            logical_to_ticket = cross_loop_scheduler._source_segment_ticket_order(
                source_segment
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

    def test_source_ticket_order_keeps_uncoalesced_woven_proof(self) -> None:
        (domain,) = _identify_root_domains((_domain((10, 2, 1), (11, 256, 1)),))
        reference = pid_task_order(domain, domain.axis_order)
        schedule = cross_loop_scheduler._with_source_ticket_schedule_segment(
            _baseline_worker_schedule((domain,), worker_count=148),
            (reference,),
            0,
        )
        self.assertIsNotNone(schedule)
        assert schedule is not None

        with _forbid_schedule_enumeration():
            source_segment = cross_loop_scheduler._source_ticket_schedule_segment(
                schedule,
            )
            assert source_segment is not None
            logical_to_ticket = cross_loop_scheduler._source_segment_ticket_order(
                source_segment,
            )

        self.assertIsNotNone(source_segment)
        self.assertIsNotNone(logical_to_ticket)
        assert source_segment is not None
        assert logical_to_ticket is not None
        self.assertEqual(source_segment.task_count, 512)
        self.assertTrue(logical_to_ticket.is_total_function())
        traversal = _root_schedule_traversal((source_segment,), reference)
        self.assertIsNotNone(traversal)
        assert traversal is not None
        self.assertTrue(traversal.matches_reference)

    def test_source_ticket_order_preserves_existing_segment_permutation(
        self,
    ) -> None:
        (domain,) = _identify_root_domains((_domain((10, 2, 1), (11, 3, 1)),))
        reference = pid_task_order(domain, (10, 11))
        permuted = pid_task_order(domain, (11, 10))
        schedule = cross_loop_scheduler._with_source_ticket_schedule_segment(
            _baseline_worker_schedule((domain,), worker_count=2),
            (permuted,),
            0,
        )
        self.assertIsNotNone(schedule)
        assert schedule is not None
        source_relation = schedule.segments_for_root(0)[0].task_order

        retained = cross_loop_scheduler._with_source_ticket_schedule_segment(
            schedule,
            (reference,),
            0,
        )

        self.assertIs(retained, schedule)
        assert retained is not None
        self.assertEqual(retained.segments_for_root(0)[0].task_order, source_relation)
        with _forbid_schedule_enumeration():
            self.assertTrue(_validate_worker_schedule_tasks(retained, (reference,)))
            source_segment = cross_loop_scheduler._source_ticket_schedule_segment(
                retained
            )
            assert source_segment is not None
            logical_to_ticket = cross_loop_scheduler._source_segment_ticket_order(
                source_segment,
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
        source_ticket_safe = _schedule(
            2,
            _segment(1, root_orders[1], workers=(0, 1), dispatch_offset=0),
            _segment(2, root_orders[2], workers=(0, 2), dispatch_offset=2),
        )
        source_ticket_late_resident_arm = _schedule(
            2,
            _segment(2, root_orders[2], workers=(0, 2), dispatch_offset=0),
            _segment(1, root_orders[1], workers=(1, 1), dispatch_offset=1),
        )
        source_ticket_safe = cross_loop_scheduler._with_source_ticket_schedule_segment(
            source_ticket_safe,
            root_orders,
            0,
        )
        source_ticket_late_resident_arm = (
            cross_loop_scheduler._with_source_ticket_schedule_segment(
                source_ticket_late_resident_arm,
                root_orders,
                0,
            )
        )
        self.assertIsNotNone(source_ticket_safe)
        self.assertIsNotNone(source_ticket_late_resident_arm)
        assert source_ticket_safe is not None
        assert source_ticket_late_resident_arm is not None

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
                    source_ticket_safe,
                    readiness_graph,
                    (plan,),
                    frozenset(),
                )
            )
            self.assertFalse(
                cross_loop_scheduler._schedule_is_progress_safe(
                    source_ticket_late_resident_arm,
                    readiness_graph,
                    (plan,),
                    frozenset(),
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
                "_MAX_GLOBAL_LIST_WORK",
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
            plan.producers[0].producers_by_key.substitute_parameters(zero_batch).pieces
        )
        self.assertFalse(
            plan.consumers[0].keys_by_consumer.substitute_parameters(zero_batch).pieces
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
            sorted(len(producers) for producers in concrete_producers.materialize()),
            [1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4],
        )
        self.assertEqual(
            cross_loop_scheduler._arrival_count_bounds(plan.producers),
            (1, 4),
        )

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

    def test_nested_frontier_derives_qwen_split_from_worker_schedule(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (
                _domain((20, 1, 1), (21, 1536, 1)),
                _domain((30, 1, 1), (31, 16, 1)),
            )
        )
        producer_order_domain = _domain(
            (10, 1),
            (11, 16),
            (12, 96),
            kind="task_order",
            identity=0,
        )
        order_batch = coordinate_axis_symbol(10)
        inner = coordinate_axis_symbol(11)
        iteration = coordinate_axis_symbol(12)
        producer_order = CoordinateRelation.point_map(
            producer_order_domain,
            producer_domain,
            (
                (
                    ((10, 0, 1, 1), (11, 0, 8, 1), (12, 0, 96, 1)),
                    (order_batch, 8 * iteration + inner),
                ),
                (
                    ((10, 0, 1, 1), (11, 8, 16, 1), (12, 0, 96, 1)),
                    (order_batch, 8 * iteration + inner + 760),
                ),
            ),
        )
        key_domain = _domain((0, 1), (1, 96), kind="event", identity=0)
        key_batch = coordinate_axis_symbol(0)
        key_iteration = coordinate_axis_symbol(1)
        producers_by_key = CoordinateRelation(
            key_domain,
            producer_domain,
            (
                _CoordinateRelationPiece(
                    ((0, 0, 1, 1), (1, 0, 96, 1)),
                    (
                        (20, key_batch, key_batch + 1, 1),
                        (21, 8 * key_iteration, 8 * key_iteration + 8, 1),
                    ),
                ),
                _CoordinateRelationPiece(
                    ((0, 0, 1, 1), (1, 0, 96, 1)),
                    (
                        (20, key_batch, key_batch + 1, 1),
                        (
                            21,
                            8 * key_iteration + 768,
                            8 * key_iteration + 776,
                            1,
                        ),
                    ),
                ),
            ),
        )
        nested_domain = _domain(
            (30, 1, 1),
            (31, 16, 1),
            (32, 96, 1),
            identity=7,
        )
        consumer = ReadinessConsumer(
            consumer_root=1,
            consumer_site_id=7,
            keys_by_consumer=_full_point_map(
                nested_domain,
                key_domain,
                coordinate_axis_symbol(30),
                coordinate_axis_symbol(32),
            ),
        )
        event = ReadinessEvent(
            producers=(ReadinessProducer(0, producers_by_key),),
            consumers=(consumer,),
        )
        graph = ReadinessGraph(
            root_task_orders=(
                producer_order,
                pid_task_order(consumer_domain, consumer_domain.axis_order),
            ),
            events=(event,),
        )
        accepted_schedule = _schedule(
            1184,
            _segment(
                0,
                producer_order,
                workers=(0, 1184),
                dispatch_offset=0,
            ),
        )
        event_frontier = _event_ready_after_worker_steps(
            graph,
            event,
            worker_schedule=accepted_schedule,
            continuation_by_root={},
        )
        self.assertIsNotNone(event_frontier)
        assert event_frontier is not None
        ready_after_worker_step = consumer.keys_by_consumer.then(event_frontier[0])
        self.assertIsNotNone(ready_after_worker_step)
        assert ready_after_worker_step is not None
        nested_readiness = cross_loop_scheduler._NestedLoopReadiness(
            event,
            consumer,
            ready_after_worker_step,
            frozenset(),
        )
        frontier = cross_loop_scheduler._uniform_nested_readiness_frontier(
            ready_after_worker_step,
            32,
        )
        self.assertIsNotNone(frontier)
        assert frontier is not None
        self.assertEqual(
            cross_loop_scheduler._nested_ready_prefix_boundaries(frontier, 1),
            (0, 74, 96),
        )

        with (
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError("nested frontier must not enumerate"),
            ),
            mock.patch.object(
                CoordinateRelation,
                "value_bounds",
                side_effect=AssertionError("exact preimage must precede the oracle"),
            ),
        ):
            plan = cross_loop_scheduler._split_nested_loop_at_readiness(
                graph,
                nested_readiness,
                consumer_worker_step=1,
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.readiness_key_domain.shape, (2,))
        self.assertEqual(
            sorted(set(_expected_arrivals(plan.readiness_key_domain, plan.producers))),
            [352, 1184],
        )

    def test_nested_quotient_uses_final_root_order_without_rescheduling(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (
                _domain((10, 8, 1)),
                _domain((20, 1, 1)),
            )
        )
        key_domain = _domain((0, 4), kind="event", identity=0)
        key = coordinate_axis_symbol(0)
        producer_arms = tuple(
            ReadinessProducer(
                0,
                CoordinateRelation(
                    key_domain,
                    producer_domain,
                    (
                        _CoordinateRelationPiece(
                            ((0, 0, 4, 1),),
                            ((10, key + offset, key + offset + 1, 1),),
                        ),
                    ),
                ),
            )
            for offset in (0, 4)
        )
        nested_domain = _domain((20, 1, 1), (21, 4, 1), identity=7)
        consumer = ReadinessConsumer(
            consumer_root=1,
            consumer_site_id=7,
            keys_by_consumer=_full_point_map(
                nested_domain,
                key_domain,
                coordinate_axis_symbol(21),
            ),
            covered_obligations=frozenset(((0, None, 7),)),
        )
        event = ReadinessEvent(producers=producer_arms, consumers=(consumer,))
        graph = _readiness_graph((producer_domain, consumer_domain), event)
        exact = (ReadinessCounterPlan(event.producers, event.consumers),)
        self.assertEqual(
            cross_loop_scheduler.collect_nested_loop_scheduling_counters(graph),
            exact,
        )

        # The scratch traversal completes two keys before wave one.  The final
        # traversal groups the two producers of each key together and completes
        # three.  The emitted quotient must describe the latter without being
        # an input to the root-order chooser.
        scratch = _schedule(
            8,
            _segment(
                0,
                graph.root_task_orders[0],
                workers=(0, 6),
                dispatch_offset=0,
            ),
            _segment(
                1,
                graph.root_task_orders[1],
                workers=(6, 1),
                dispatch_offset=1,
            ),
        )
        order_domain = _domain(
            (30, 2),
            (31, 4),
            kind="task_order",
            identity=0,
        )
        final_order = _full_point_map(
            order_domain,
            producer_domain,
            4 * coordinate_axis_symbol(30) + coordinate_axis_symbol(31),
        )
        final = _schedule(
            8,
            _segment(0, final_order, workers=(0, 6), dispatch_offset=0),
            _segment(
                1,
                graph.root_task_orders[1],
                workers=(6, 1),
                dispatch_offset=1,
            ),
        )
        scratch_counters = (
            cross_loop_scheduler._compact_nested_loop_counters_for_schedule(
                graph,
                scratch,
                exact,
            )
        )
        final_counters = (
            cross_loop_scheduler._compact_nested_loop_counters_for_schedule(
                graph,
                final,
                exact,
            )
        )
        original_consumer_steps = WorkerSchedule.worker_step_bounds_for_root

        def span_two_consumer_waves(
            schedule: WorkerSchedule,
            root: int,
        ) -> tuple[int, int] | None:
            bounds = original_consumer_steps(schedule, root)
            if schedule is final and root == consumer.consumer_root:
                assert bounds is not None
                return bounds[0], bounds[0] + 1
            return bounds

        with mock.patch.object(
            WorkerSchedule,
            "worker_step_bounds_for_root",
            span_two_consumer_waves,
        ):
            multiwave_counters = (
                cross_loop_scheduler._compact_nested_loop_counters_for_schedule(
                    graph,
                    final,
                    exact,
                )
            )
        self.assertEqual(
            _expected_arrivals(
                scratch_counters[0].readiness_key_domain,
                scratch_counters[0].producers,
            ),
            (4, 4),
        )
        self.assertEqual(
            _expected_arrivals(
                final_counters[0].readiness_key_domain,
                final_counters[0].producers,
            ),
            (6, 2),
        )
        self.assertEqual(multiwave_counters, final_counters)

        scheduling_inputs: list[tuple[ReadinessCounterPlan, ...]] = []

        def prepare_from_exact(
            _readiness_graph: ReadinessGraph,
            _worker_schedule: WorkerSchedule,
            readiness_counters: tuple[ReadinessCounterPlan, ...],
            _root_barrier_edges: frozenset[tuple[int, int]],
            **_kwargs: object,
        ) -> WorkerSchedule:
            scheduling_inputs.append(readiness_counters)
            return final

        def place_from_exact(
            _readiness_graph: ReadinessGraph,
            _worker_schedule: WorkerSchedule,
            readiness_counters: tuple[ReadinessCounterPlan, ...],
            _root_barrier_edges: frozenset[tuple[int, int]],
            **_kwargs: object,
        ) -> None:
            scheduling_inputs.append(readiness_counters)
            return None

        with (
            mock.patch.object(
                cross_loop_scheduler,
                "_consumer_major_producer_order",
                side_effect=prepare_from_exact,
            ) as prepare,
            mock.patch.object(
                cross_loop_scheduler,
                "_global_unit_list_schedule",
                side_effect=place_from_exact,
            ),
        ):
            selected = cross_loop_scheduler._try_finalize_pipeline_proposal(
                readiness_graph=graph,
                worker_count=scratch.worker_count,
                readiness_counters=exact,
                root_barrier_edges=frozenset(),
                source_ticket_root=None,
                pipeline_depth=1,
            )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(scheduling_inputs, [exact, exact])
        self.assertEqual(selected.worker_schedule, final)
        self.assertEqual(
            _expected_arrivals(
                selected.readiness_counters[0].readiness_key_domain,
                selected.readiness_counters[0].producers,
            ),
            (6, 2),
        )
        self.assertEqual(len(selected.readiness_counters[0].producers), 2)
        self.assertEqual(
            cross_loop_scheduler._arrival_count_bounds(
                selected.readiness_counters[0].producers
            ),
            (2, 6),
        )
        self.assertEqual(
            selected.readiness_counters[0].consumers[0].covered_obligations,
            consumer.covered_obligations,
        )
        publication_counts = [0] * producer_domain.size
        for producer in selected.readiness_counters[0].producers:
            publication = producer.producers_by_key.converse()
            self.assertIsNotNone(publication)
            assert publication is not None
            for task_index, keys in enumerate(publication.materialize()):
                publication_counts[task_index] += len(keys)
        self.assertEqual(publication_counts, [1] * producer_domain.size)

        entry = _nested_loop_entry_counter(graph, event, consumer)
        self.assertIsNotNone(entry)
        assert entry is not None
        original_progress = cross_loop_scheduler._schedule_is_progress_safe

        def reject_compact_progress(
            worker_schedule: WorkerSchedule,
            readiness_graph: ReadinessGraph,
            readiness_counters: tuple[ReadinessCounterPlan, ...],
            root_barrier_edges: frozenset[tuple[int, int]],
        ) -> bool:
            # Model the final progress proof finding that the stronger entry
            # wait has introduced a same-strand cycle.  It was not an input to
            # scheduling, so the transaction must retain the same placement
            # and retry the exact per-iteration prerequisite.
            if readiness_counters == (entry,):
                return False
            return original_progress(
                worker_schedule,
                readiness_graph,
                readiness_counters,
                root_barrier_edges,
            )

        with (
            mock.patch.object(
                cross_loop_scheduler,
                "_consumer_major_producer_order",
                return_value=final,
            ) as fallback_prepare,
            mock.patch.object(
                cross_loop_scheduler,
                "_global_unit_list_schedule",
                return_value=None,
            ),
            mock.patch.object(
                cross_loop_scheduler,
                "_compact_nested_loop_counters_for_schedule",
                return_value=(entry,),
            ),
            mock.patch.object(
                cross_loop_scheduler,
                "_schedule_is_progress_safe",
                side_effect=reject_compact_progress,
            ),
        ):
            exact_fallback = cross_loop_scheduler._try_finalize_pipeline_proposal(
                readiness_graph=graph,
                worker_count=scratch.worker_count,
                readiness_counters=exact,
                root_barrier_edges=frozenset(),
                source_ticket_root=None,
                pipeline_depth=1,
            )

        self.assertIsNotNone(exact_fallback)
        assert exact_fallback is not None
        self.assertEqual(fallback_prepare.call_count, 1)
        self.assertEqual(exact_fallback.worker_schedule, final)
        self.assertEqual(exact_fallback.readiness_counters, exact)

    def test_nested_entry_quotient_requires_progress_precedence(
        self,
    ) -> None:
        graph, event, consumer = _symbolic_nested_counter_graph(8, 1, 4)
        exact = (ReadinessCounterPlan(event.producers, event.consumers),)
        entry = _nested_loop_entry_counter(graph, event, consumer)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(exact[0].readiness_key_count, 32)
        self.assertEqual(entry.readiness_key_count, 8)
        self.assertEqual(
            cross_loop_scheduler._arrival_count_bounds(entry.producers),
            (4, 4),
        )

        prior_wave = _schedule(
            32,
            _segment(
                0,
                graph.root_task_orders[0],
                workers=(0, 32),
                dispatch_offset=0,
            ),
            _segment(
                1,
                graph.root_task_orders[1],
                workers=(0, 8),
                dispatch_offset=8,
            ),
        )
        same_wave = _schedule(
            40,
            _segment(
                0,
                graph.root_task_orders[0],
                workers=(0, 32),
                dispatch_offset=0,
            ),
            _segment(
                1,
                graph.root_task_orders[1],
                workers=(32, 8),
                dispatch_offset=0,
            ),
        )
        oversubscribed_resident = _schedule(
            8,
            _segment(
                1,
                graph.root_task_orders[1],
                workers=(0, 8),
                dispatch_offset=0,
            ),
        )
        source_ticket = cross_loop_scheduler._with_source_ticket_schedule_segment(
            oversubscribed_resident,
            graph.root_task_orders,
            0,
        )
        self.assertIsNotNone(source_ticket)
        assert source_ticket is not None

        with _forbid_schedule_enumeration():
            prior_wave_compact = (
                cross_loop_scheduler._compact_nested_loop_counters_for_schedule(
                    graph,
                    prior_wave,
                    exact,
                )
            )
            same_wave_compact = (
                cross_loop_scheduler._compact_nested_loop_counters_for_schedule(
                    graph,
                    same_wave,
                    exact,
                )
            )
            source_ticket_compact = (
                cross_loop_scheduler._compact_nested_loop_counters_for_schedule(
                    graph,
                    source_ticket,
                    exact,
                )
            )

        # A resident producer must be on a prior rank before its wait can move
        # to loop entry. A certified source-stage producer may instead use its
        # earlier ticket-issue phase, as checked below.
        self.assertEqual(prior_wave_compact, (entry,))
        # The same-wave plan is live by the segment-DAG proof, but hoisting its
        # wait would erase legal overlap.  Keep the exact per-iteration keys.
        self.assertTrue(
            cross_loop_scheduler._schedule_is_progress_safe(
                same_wave,
                graph,
                (entry,),
                frozenset(),
            )
        )
        self.assertEqual(same_wave_compact, exact)
        # Source tickets are all issued before any resident ticket.  The
        # coarser entry wait still gates completion, but cannot occupy the GPU
        # before an outstanding source task has begun and thereby deadlock it.
        # This case deliberately has 32 source tasks but only eight resident
        # workers, matching the oversubscribed source condition used by MLA.
        self.assertEqual(source_ticket_compact, (entry,))
        self.assertTrue(
            cross_loop_scheduler._schedule_is_progress_safe(
                source_ticket,
                graph,
                source_ticket_compact,
                frozenset(),
            )
        )

    def test_nested_quotient_uses_earliest_real_multiwave_admission(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (
                _domain((10, 5, 1)),
                _domain((20, 8, 1)),
            )
        )
        key_domain = _domain((0, 5), kind="event", identity=0)
        producer = _readiness_producer_from_publication(
            0,
            _full_point_map(
                producer_domain,
                key_domain,
                coordinate_axis_symbol(10),
            ),
        )
        producers = (producer,)
        nested_domain = _domain((20, 8, 1), (21, 5, 1), identity=7)
        consumer = ReadinessConsumer(
            consumer_root=1,
            consumer_site_id=7,
            keys_by_consumer=_full_point_map(
                nested_domain,
                key_domain,
                coordinate_axis_symbol(21),
            ),
            covered_obligations=frozenset(((0, None, 7),)),
        )
        event = ReadinessEvent(producers=producers, consumers=(consumer,))
        graph = _readiness_graph((producer_domain, consumer_domain), event)
        exact = (ReadinessCounterPlan(event.producers, event.consumers),)

        # Producer and consumer use disjoint resident lanes and both span
        # multiple waves. At the consumer's earliest wave only keys 0 and 1
        # are ready; at its latest wave keys 0--3 are ready. The uniform
        # quotient must use the earliest admission frontier so every owning
        # CTA can execute the same two-stage wait safely.
        schedule = _schedule(
            8,
            _segment(
                0,
                graph.root_task_orders[0],
                workers=(0, 2),
                dispatch_offset=0,
            ),
            _segment(
                1,
                graph.root_task_orders[1],
                workers=(2, 6),
                dispatch_offset=6,
            ),
        )
        self.assertEqual(schedule.worker_step_bounds_for_root(0), (0, 2))
        self.assertEqual(schedule.worker_step_bounds_for_root(1), (1, 2))

        nested_readiness = cross_loop_scheduler._nested_loop_readiness(
            graph,
            event,
            consumer,
            worker_schedule=schedule,
            continuation_by_root={},
        )
        self.assertIsNotNone(nested_readiness)
        assert nested_readiness is not None
        frontier = cross_loop_scheduler._uniform_nested_readiness_frontier(
            nested_readiness.ready_after_worker_step,
            21,
        )
        self.assertIsNotNone(frontier)
        assert frontier is not None
        self.assertEqual(
            cross_loop_scheduler._nested_ready_prefix_boundaries(frontier, 1),
            (0, 2, 5),
        )
        earliest = cross_loop_scheduler._split_nested_loop_at_readiness(
            graph,
            nested_readiness,
            consumer_worker_step=1,
        )
        latest = cross_loop_scheduler._split_nested_loop_at_readiness(
            graph,
            nested_readiness,
            consumer_worker_step=2,
        )
        self.assertIsNotNone(earliest)
        self.assertIsNotNone(latest)
        assert earliest is not None and latest is not None
        self.assertEqual(earliest.readiness_key_count, 2)
        self.assertEqual(latest.readiness_key_count, 2)
        self.assertEqual(
            _expected_arrivals(earliest.readiness_key_domain, earliest.producers),
            (2, 3),
        )
        self.assertEqual(
            _expected_arrivals(latest.readiness_key_domain, latest.producers),
            (4, 1),
        )

        compact = cross_loop_scheduler._compact_nested_loop_counters_for_schedule(
            graph,
            schedule,
            exact,
        )

        self.assertEqual(len(compact), 1)
        self.assertEqual(compact[0].readiness_key_count, 2)
        self.assertEqual(
            _expected_arrivals(
                compact[0].readiness_key_domain,
                compact[0].producers,
            ),
            (2, 3),
        )
        self.assertEqual(
            cross_loop_scheduler._arrival_count_bounds(compact[0].producers),
            (2, 3),
        )
        self.assertEqual(len(compact[0].producers), 1)
        self.assertEqual(
            compact[0].consumers[0].covered_obligations,
            consumer.covered_obligations,
        )
        publication_counts = [0] * producer_domain.size
        for producer in compact[0].producers:
            publication = producer.producers_by_key.converse()
            self.assertIsNotNone(publication)
            assert publication is not None
            for task_index, keys in enumerate(publication.materialize()):
                publication_counts[task_index] += len(keys)
        self.assertEqual(publication_counts, [1] * producer_domain.size)
        self.assertTrue(
            cross_loop_scheduler._schedule_is_progress_safe(
                schedule,
                graph,
                compact,
                frozenset(),
            )
        )

    def test_nested_frontier_factors_runtime_empty_outer_domain(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        query = sympy.Symbol("query", integer=True, nonnegative=True)
        graph, event, consumer = _symbolic_nested_counter_graph(batch, query, 96)
        source_domain = consumer.keys_by_consumer.source_domain
        wave_domain = _domain((40, 2), kind="value")
        iteration = coordinate_axis_symbol(22)
        ready_after_worker_step = _full_point_map(
            source_domain,
            wave_domain,
            sympy.floor((16 * iteration + 15) / 1184),
        )
        nested_readiness = cross_loop_scheduler._NestedLoopReadiness(
            event,
            consumer,
            ready_after_worker_step,
            frozenset(),
        )
        concrete_axis_counts = CoordinateDomain._concrete_axis_counts

        def reject_runtime_axis_counts(domain: CoordinateDomain) -> dict[int, int]:
            if domain.parameter_symbols:
                raise AssertionError("runtime outer axes must remain symbolic")
            return concrete_axis_counts(domain)

        with (
            mock.patch.object(
                CoordinateDomain,
                "_concrete_axis_counts",
                reject_runtime_axis_counts,
            ),
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError("runtime outer axes must not enumerate"),
            ),
        ):
            plan = cross_loop_scheduler._split_nested_loop_at_readiness(
                graph,
                nested_readiness,
                consumer_worker_step=1,
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.readiness_key_domain.shape_expr, (2, batch, query))
        for zero in ({batch: 0, query: 3}, {batch: 2, query: 0}):
            self.assertFalse(
                plan.producers[0].producers_by_key.substitute_parameters(zero).pieces
            )
            self.assertFalse(
                plan.consumers[0].keys_by_consumer.substitute_parameters(zero).pieces
            )
        concrete = {batch: 2, query: 3}
        concrete_producers = plan.producers[0].producers_by_key.substitute_parameters(
            concrete
        )
        arrivals = tuple(
            len(producers) for producers in concrete_producers.materialize()
        )
        self.assertEqual(arrivals.count(74), 6)
        self.assertEqual(arrivals.count(22), 6)

    def test_nested_frontier_declines_nonuniform_empty_or_nonprefix_fibers(
        self,
    ) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source_domain = CoordinateDomain(
            (20, 21),
            ((20, batch), (21, 4)),
            kind="site",
            _allow_empty=True,
        )
        wave_domain = CoordinateDomain(
            (30,),
            ((30, batch + 4),),
            kind="value",
            _allow_empty=True,
        )
        outer = coordinate_axis_symbol(20)
        nested = coordinate_axis_symbol(21)
        outer_dependent = _full_point_map(
            source_domain,
            wave_domain,
            outer + nested,
        )
        self.assertIsNone(
            cross_loop_scheduler._uniform_nested_readiness_frontier(
                outer_dependent,
                21,
            )
        )

        concrete_source = _domain((20, 1), (21, 4), kind="site")
        nonmonotone = CoordinateRelation.point_map(
            concrete_source,
            _domain((30, 2), kind="value"),
            (
                (((20, 0, 1, 1), (21, 0, 2, 1)), (sympy.Integer(0),)),
                (((20, 0, 1, 1), (21, 2, 3, 1)), (sympy.Integer(1),)),
                (((20, 0, 1, 1), (21, 3, 4, 1)), (sympy.Integer(0),)),
            ),
        )
        frontier = cross_loop_scheduler._uniform_nested_readiness_frontier(
            nonmonotone,
            21,
        )
        self.assertIsNotNone(frontier)
        assert frontier is not None
        self.assertIsNone(
            cross_loop_scheduler._nested_ready_prefix_boundaries(frontier, 1)
        )
        self.assertIsNone(
            cross_loop_scheduler._concrete_nested_ready_prefix_boundaries(
                frontier,
                1,
            )
        )

    def test_nested_frontier_matches_concrete_prefix_oracle(self) -> None:
        source_domain = _domain((20, 3), (21, 17), kind="site")
        wave_domain = _domain((30, 8), kind="value")
        outer = coordinate_axis_symbol(20)
        nested = coordinate_axis_symbol(21)
        ready_after_worker_step = _full_point_map(
            source_domain,
            wave_domain,
            sympy.floor((2 * nested + outer) / 5),
        )
        frontier = cross_loop_scheduler._uniform_nested_readiness_frontier(
            ready_after_worker_step,
            21,
        )
        self.assertIsNotNone(frontier)
        assert frontier is not None
        concrete_values = ready_after_worker_step.materialize()

        for consumer_worker_step in range(9):
            with self.subTest(consumer_worker_step=consumer_worker_step):
                expected = 0
                for nested_iteration in range(17):
                    if all(
                        next(iter(concrete_values[outer_index + 3 * nested_iteration]))
                        < consumer_worker_step
                        for outer_index in range(3)
                    ):
                        expected += 1
                    else:
                        break
                expected_boundaries = tuple(sorted({0, expected, 17}))
                self.assertEqual(
                    cross_loop_scheduler._nested_ready_prefix_boundaries(
                        frontier,
                        consumer_worker_step,
                    ),
                    expected_boundaries,
                )

        monotone_but_not_invertible = _full_point_map(
            _domain((21, 17), kind="site"),
            wave_domain,
            sympy.Max(
                sympy.floor(nested / 4),
                sympy.floor(nested / 6),
            ),
        )
        self.assertIsNone(
            cross_loop_scheduler._nested_ready_prefix_boundaries(
                monotone_but_not_invertible,
                3,
            )
        )
        self.assertEqual(
            cross_loop_scheduler._concrete_nested_ready_prefix_boundaries(
                monotone_but_not_invertible,
                3,
            ),
            (0, 12, 17),
        )

        symbolic_split = sympy.Symbol(
            "symbolic_split",
            integer=True,
            positive=True,
        )
        symbolic_source = CoordinateDomain(
            (21,),
            ((21, symbolic_split + 2),),
            kind="site",
        )
        symbolic_frontier = CoordinateRelation.point_map(
            symbolic_source,
            _domain((30, 2), kind="value"),
            (
                (((21, 0, symbolic_split, 1),), (sympy.Integer(0),)),
                (
                    ((21, symbolic_split, symbolic_split + 2, 1),),
                    (sympy.Integer(1),),
                ),
            ),
        )
        self.assertEqual(
            cross_loop_scheduler._nested_ready_prefix_boundaries(
                symbolic_frontier,
                1,
            ),
            (0, symbolic_split, symbolic_split + 2),
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
        self.assertEqual(derive_final_arrival_continuations(readiness_graph), ())
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

    def test_final_arrival_continuation_rejects_cross_key_worker_strands(
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

        candidates = derive_final_arrival_continuations(readiness_graph)
        continuations = choose_final_arrival_continuations(
            readiness_graph,
            candidates,
            baseline,
        )

        self.assertEqual(len(candidates), 1)
        self.assertGreater(root_domains[0].size, baseline.worker_count)
        self.assertGreater(root_domains[1].size, baseline.worker_count)
        self.assertEqual(readiness_graph.events[1].root_barrier_producer_root, 1)
        # Every worker executes producers for more than one readiness key.
        # The final-arrival owner is therefore not a one-use strand, so the
        # consumer remains resident regardless of its downstream barrier.
        self.assertEqual(continuations, ())

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
        readiness_graph = _configured_readiness_graph(
            graph,
            root_domains,
            axis_geometry=axis_geometry,
        )
        self.assertEqual(readiness_graph.events, events)
        self.assertEqual(
            readiness_graph.obligations_by_root_pair,
            (
                (
                    (0, 1),
                    frozenset(
                        (
                            (dependency_id, 0, 2),
                            (dependency_id, 1, 2),
                        )
                    ),
                ),
            ),
        )
        self.assertEqual(derive_final_arrival_continuations(readiness_graph), ())
        readiness_counters = choose_readiness_counters(readiness_graph, ())
        covered_obligations = frozenset(
            obligation
            for counter_plan in readiness_counters
            for readiness_consumer in counter_plan.consumers
            for obligation in readiness_consumer.covered_obligations
        )
        self.assertEqual(
            _select_root_barrier_edges(
                readiness_graph=readiness_graph,
                covered_obligations=covered_obligations,
            ),
            frozenset(((0, 1),)),
        )

    def test_dependency_manifest_is_required_and_exact_for_finalization(self) -> None:
        root_domains = _identify_root_domains(
            (_domain((10, 1, 1)), _domain((20, 1, 1)))
        )
        obligation = (0, None, None)
        event = _whole_root_readiness_event(root_domains, 0, 1, 0)
        event = dataclasses.replace(
            event,
            consumers=(
                dataclasses.replace(
                    event.consumers[0],
                    covered_obligations=frozenset((obligation,)),
                ),
            ),
        )
        synthetic = _readiness_graph(root_domains, event)
        with self.assertRaisesRegex(ValueError, "dependency-obligation manifest"):
            cross_loop_scheduler._finalize_emitted_synchronization(
                readiness_graph=synthetic,
                readiness_counters=(),
            )
        with self.assertRaisesRegex(ValueError, "dependency-obligation manifest"):
            cross_loop_scheduler._validate_schedule_coverage(
                readiness_graph=synthetic,
                covered_obligations=frozenset(),
                root_barrier_edges=frozenset(),
            )
        with self.assertRaisesRegex(ValueError, "cover the dependency manifest"):
            _readiness_graph(
                root_domains,
                event,
                obligations_by_root_pair=(),
            )
        with self.assertRaisesRegex(ValueError, "multiple root pairs"):
            _readiness_graph(
                root_domains,
                event,
                obligations_by_root_pair=(
                    ((0, 1), frozenset((obligation,))),
                    ((1, 0), frozenset((obligation,))),
                ),
            )

    def test_root_barrier_declines_non_forward_dependency(self) -> None:
        graph = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", block_ids=(10,)),
            _access(root=1, kind="load", block_ids=(20,)),
        )
        graph = dataclasses.replace(
            graph,
            edges=(dataclasses.replace(graph.edges[0], consumer_root=0),),
        )
        root_domains = (
            _domain((10, 1, 1), identity=0),
            _domain((20, 1, 1), identity=1),
        )
        access_dependency = graph.edges[0].access_dependencies[0]
        obligations = graph.dependency_obligations(access_dependency)
        event = _whole_root_readiness_event(root_domains, 0, 0, 0)
        event = dataclasses.replace(
            event,
            consumers=(
                dataclasses.replace(
                    event.consumers[0],
                    covered_obligations=obligations,
                ),
            ),
        )
        readiness_graph = _readiness_graph(
            root_domains,
            event,
            obligations_by_root_pair=(((0, 0), obligations),),
        )

        with self.assertRaisesRegex(
            exc.CrossLoopSchedulingError,
            "strict source-ordered dependency",
        ):
            _select_root_barrier_edges(
                readiness_graph=readiness_graph,
                covered_obligations=frozenset(),
            )

        root_task_orders = _default_root_task_orders(root_domains)
        with self.assertRaisesRegex(ValueError, "strict source-ordered dependency"):
            cross_loop_scheduler.StaticPipelinePlan(
                worker_schedule=_build_baseline_worker_schedule(
                    root_domains,
                    root_task_orders,
                    worker_count=2,
                ),
                root_task_orders=root_task_orders,
                readiness_counters=(),
                root_barrier_edges=frozenset(((0, 0),)),
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

    def test_baseline_worker_schedule_compacts_around_excluded_roots(self) -> None:
        root_domains = _identify_root_domains(
            (
                _domain((10, 3, 1)),
                _domain((20, 5, 1)),
                _domain((30, 2, 1)),
            )
        )
        schedule = _build_baseline_worker_schedule(
            root_domains,
            _default_root_task_orders(root_domains),
            worker_count=4,
            excluded_roots=frozenset((1,)),
        )

        self.assertEqual(
            tuple(segment.root for segment in schedule.segments),
            (0, 2),
        )
        self.assertEqual(placement(schedule, 2, 0), (0, 1))
        self.assertEqual(placement(schedule, 2, 1), (1, 1))

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

        self.assertEqual(
            cross_loop_scheduler.root_barrier_publication_plan(
                schedule,
                0,
            ).participant_intervals,
            ((3, 5),),
        )
        self.assertEqual(schedule.dense_assignment(0), (1, 4, 2, 2))
        self.assertIsNone(schedule.contiguous_global_interval(0))

    def test_root_local_preparation_orders_continuation_producers_atomically(
        self,
    ) -> None:
        root_domains = (
            _domain((10, 2, 1), (11, 2, 1)),
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
                            producer_axis,
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
        event = readiness_graph.events[0]
        continuation_plan = ReadinessCounterPlan(
            event.producers,
            event.consumers,
            continuation_consumer_index=0,
        )
        resident = baseline.without_roots(frozenset((1,)))
        schedule = cross_loop_scheduler._consumer_major_producer_order(
            readiness_graph,
            resident,
            (continuation_plan,),
            frozenset(),
            excluded_roots=frozenset((1,)),
        )

        self.assertEqual(task_order(resident, 0), (0, 1, 2, 3))
        self.assertEqual(task_order(schedule, 0), (0, 2, 1, 3))

        # Root-local ordering is one speculative transaction. Failure to
        # normalize the alternate exact traversal must retain the canonical
        # schedule rather than invalidate continuation ownership or the plan.
        with mock.patch.object(
            cross_loop_scheduler,
            "WorkerSchedule",
            side_effect=ValueError("unsupported alternate traversal"),
        ):
            declined = cross_loop_scheduler._consumer_major_producer_order(
                readiness_graph,
                resident,
                (continuation_plan,),
                frozenset(),
                excluded_roots=frozenset((1,)),
            )
        self.assertIs(declined, resident)

    def test_unsafe_root_local_preparation_retains_frozen_ownership(self) -> None:
        producer_domain, consumer_domain = _identify_root_domains(
            (_domain((10, 2, 1)), _domain((20, 1, 1)))
        )
        key_domain = _domain((0, 1), kind="event", identity=0)
        zero = sympy.Integer(0)
        event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    0,
                    _full_point_map(key_domain, producer_domain, zero),
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
        canonical = _schedule(
            2,
            _segment(
                0,
                _one_dimensional_task_range(producer_domain, 0, 1),
                workers=(0, 1),
                dispatch_offset=0,
            ),
            _segment(
                1,
                graph.root_task_orders[1],
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
        unsafe_permutation = _schedule(
            2,
            _segment(
                0,
                _one_dimensional_task_range(producer_domain, 1, 1),
                workers=(0, 1),
                dispatch_offset=0,
            ),
            _segment(
                1,
                graph.root_task_orders[1],
                workers=(1, 1),
                dispatch_offset=0,
            ),
            _segment(
                0,
                _one_dimensional_task_range(producer_domain, 0, 1),
                workers=(1, 1),
                dispatch_offset=1,
            ),
        )
        plan = ReadinessCounterPlan(event.producers, event.consumers)
        self.assertTrue(
            cross_loop_scheduler._schedule_is_progress_safe(
                canonical,
                graph,
                (plan,),
                frozenset(),
            )
        )
        self.assertFalse(
            cross_loop_scheduler._schedule_is_progress_safe(
                unsafe_permutation,
                graph,
                (plan,),
                frozenset(),
            )
        )

        with mock.patch.object(
            cross_loop_scheduler,
            "_consumer_major_producer_order",
            return_value=unsafe_permutation,
        ):
            baseline = _baseline_worker_schedule(
                graph.root_domains,
                worker_count=2,
            )
            selected = cross_loop_scheduler._try_finalize_pipeline_proposal(
                readiness_graph=graph,
                worker_count=baseline.worker_count,
                readiness_counters=(plan,),
                root_barrier_edges=frozenset(),
                source_ticket_root=None,
                pipeline_depth=1,
            )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.worker_schedule, baseline)

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
            readiness_graph=readiness_graph,
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

    def test_counter_quotient_supports_fixed_capacity_specializations(self) -> None:
        def configured_plan(batch_size: int):
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

        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size), _forbid_schedule_enumeration():
                plan = configured_plan(batch_size)
            self.assertEqual(plan.root_barrier_edges, frozenset())
            (counter,) = plan.readiness_counters
            self.assertEqual(counter.readiness_key_count, 4 * batch_size)
            self.assertEqual(counter.uniform_arrival_count(), 1)
            self.assertFalse(counter.parameter_symbols)

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

    def test_unknown_edge_does_not_discard_independent_exact_counter(self) -> None:
        dependency_graph = _dependency_graph(
            [[10], [20], [30]],
            _access(
                root=0,
                allocation_id=0,
                kind="store",
                shape=(64,),
                block_ids=(10,),
            ),
            _access(
                root=1,
                allocation_id=0,
                kind="load",
                shape=(64,),
                block_ids=(20,),
            ),
            _access(
                root=1,
                allocation_id=1,
                kind="store",
                shape=(64,),
                block_ids=(20,),
            ),
            _access(
                root=2,
                allocation_id=1,
                kind="load",
                shape=(64,),
                block_ids=(None,),
                offsets=(None,),
            ),
        )
        plan = _configured_static_pipeline_plan(
            dependency_graph=dependency_graph,
            root_domains=tuple(_domain((axis, 4, 16)) for axis in (10, 20, 30)),
            axis_geometry=dict.fromkeys((10, 20, 30), (4, 16)),
            worker_count=4,
        )

        self.assertEqual(plan.root_barrier_edges, frozenset(((1, 2),)))
        self.assertTrue(
            any(
                producer.producer_root == 0
                and any(consumer.consumer_root == 1 for consumer in counter.consumers)
                for counter in plan.readiness_counters
                for producer in counter.producers
            )
        )
        self.assertFalse(
            any(counter.parameter_symbols for counter in plan.readiness_counters)
        )

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

    def test_worker_schedule_derives_coalesced_nested_checkpoint_waits(self) -> None:
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
            (0, 1, None, 2),
        )
        local_events = tuple(
            plan
            for plan in schedule.readiness_counters
            if plan.continuation_consumer is not None
        )
        self.assertEqual(local_events, ())
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
            (4,),
        )
        self.assertEqual(
            tuple(segment.root for segment in schedule.worker_schedule.segments),
            (0, 1, 2),
        )
        readiness_graph = _configured_readiness_graph(
            dependency_graph,
            root_domains,
            axis_geometry=kwargs["axis_geometry"],
        )
        self.assertTrue(
            cross_loop_scheduler._schedule_is_progress_safe(
                schedule.worker_schedule,
                readiness_graph,
                schedule.readiness_counters,
                schedule.root_barrier_edges,
            )
        )

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

    def test_multi_producer_join_uses_one_common_static_tail_quotient(
        self,
    ) -> None:
        first_producer, second_producer, consumer = _identify_root_domains(
            (
                _domain((10, 6, 1)),
                _domain((20, 2, 1)),
                _domain((30, 6, 1)),
            )
        )
        semantic_keys = _domain((0, 6), kind="event", identity=0)
        semantic_key = coordinate_axis_symbol(0)
        event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=_full_point_map(
                        semantic_keys,
                        first_producer,
                        semantic_key,
                    ),
                ),
                _readiness_producer_from_publication(
                    producer_root=1,
                    publication=CoordinateRelation(
                        second_producer,
                        semantic_keys,
                        (
                            _CoordinateRelationPiece(
                                ((20, 0, 2, 1),),
                                (
                                    (
                                        0,
                                        2 * coordinate_axis_symbol(20),
                                        2 * coordinate_axis_symbol(20) + 2,
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
                    consumer_root=2,
                    keys_by_consumer=_full_point_map(
                        consumer,
                        semantic_keys,
                        coordinate_axis_symbol(30),
                    ),
                    covered_obligations=frozenset(((0, 0, 2), (1, 0, 2))),
                ),
            ),
        )
        graph = _readiness_graph(
            (first_producer, second_producer, consumer),
            event,
        )

        with _forbid_schedule_enumeration():
            plans = choose_readiness_counters(graph, ())

        self.assertEqual(event.readiness_key_count, 6)
        self.assertEqual(len(plans), 1)
        (plan,) = plans
        self.assertEqual(plan.readiness_key_count, 3)
        self.assertEqual(
            cross_loop_scheduler._arrival_count_bounds(plan.producers),
            (2, 3),
        )
        self.assertIsNone(plan.uniform_arrival_count())
        self.assertEqual(
            _expected_arrivals(plan.readiness_key_domain, plan.producers), (3, 3, 2)
        )
        self.assertEqual(
            plan.producers[0].producers_by_key.materialize(),
            (
                frozenset((0, 1)),
                frozenset((2, 3)),
                frozenset((4, 5)),
            ),
        )
        self.assertEqual(
            plan.producers[1].producers_by_key.materialize(),
            (frozenset((0,)), frozenset((1,)), frozenset()),
        )
        self.assertEqual(
            plan.producers[0].keys_by_producer.materialize(),
            tuple(frozenset((producer_task // 2,)) for producer_task in range(6)),
        )
        self.assertEqual(
            plan.producers[1].keys_by_producer.materialize(),
            (frozenset((0,)), frozenset((1,))),
        )
        self.assertEqual(
            plan.consumers[0].keys_by_consumer.materialize(),
            tuple(frozenset((consumer_task // 2,)) for consumer_task in range(6)),
        )
        self.assertEqual(
            plan.consumers[0].covered_obligations,
            event.consumers[0].covered_obligations,
        )

    def test_common_key_quotient_declines_cross_group_publication(self) -> None:
        first_producer, grouped_producer, crossing_producer, consumer = (
            _identify_root_domains(
                (
                    _domain((10, 6, 1)),
                    _domain((20, 2, 1)),
                    _domain((30, 1, 1)),
                    _domain((40, 6, 1)),
                )
            )
        )
        semantic_keys = _domain((0, 6), kind="event", identity=0)
        event = ReadinessEvent(
            producers=(
                ReadinessProducer(
                    producer_root=0,
                    producers_by_key=_full_point_map(
                        semantic_keys,
                        first_producer,
                        coordinate_axis_symbol(0),
                    ),
                ),
                _readiness_producer_from_publication(
                    producer_root=1,
                    publication=CoordinateRelation(
                        grouped_producer,
                        semantic_keys,
                        (
                            _CoordinateRelationPiece(
                                ((20, 0, 2, 1),),
                                (
                                    (
                                        0,
                                        2 * coordinate_axis_symbol(20),
                                        2 * coordinate_axis_symbol(20) + 2,
                                        1,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                _readiness_producer_from_publication(
                    producer_root=2,
                    publication=CoordinateRelation(
                        crossing_producer,
                        semantic_keys,
                        (
                            _CoordinateRelationPiece(
                                ((30, 0, 1, 1),),
                                ((0, sympy.Integer(1), sympy.Integer(3), 1),),
                            ),
                        ),
                    ),
                ),
            ),
            consumers=(
                ReadinessConsumer(
                    consumer_root=3,
                    keys_by_consumer=_full_point_map(
                        consumer,
                        semantic_keys,
                        coordinate_axis_symbol(40),
                    ),
                    covered_obligations=frozenset(((0, 0, 3), (1, 0, 3), (2, 0, 3))),
                ),
            ),
        )
        graph = _readiness_graph(
            (first_producer, grouped_producer, crossing_producer, consumer),
            event,
        )

        with _forbid_schedule_enumeration():
            self.assertIsNone(cross_loop_scheduler._counter_lowering_relations(event))
            plans = choose_readiness_counters(graph, ())

        self.assertEqual(plans, ())

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
            [(1, 2, (4,)), (4, 5, (4,))],
        )
        self.assertEqual(overlapped.root_barrier_edges, frozenset())
        first_sink_placement = placement(overlapped.worker_schedule, 2, 0)
        second_sink_placement = placement(overlapped.worker_schedule, 5, 0)
        self.assertIsNotNone(first_sink_placement)
        self.assertIsNotNone(second_sink_placement)
        assert first_sink_placement is not None and second_sink_placement is not None
        readiness_graph = _configured_readiness_graph(
            dependency_graph,
            tuple(root_domains),
            axis_geometry=axis_geometry,
        )
        self.assertTrue(
            cross_loop_scheduler._schedule_is_progress_safe(
                overlapped.worker_schedule,
                readiness_graph,
                overlapped.readiness_counters,
                overlapped.root_barrier_edges,
            )
        )
