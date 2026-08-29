from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helion._compiler.cross_loop_scheduler import FinalArrivalContinuation
    from helion._compiler.cross_loop_scheduler import ReadinessConsumer
    from helion._compiler.cross_loop_scheduler import ReadinessGraph
    from helion._compiler.cross_loop_scheduler import WorkerSchedule
    from helion._compiler.cross_loop_scheduler import WorkerScheduleSegment
    from helion._compiler.tile_dependency import CoordinateRelation


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
    event_index: int,
) -> tuple[frozenset[tuple[int, int]], ...]:
    event = readiness_graph.event(event_index)
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
        event = readiness_graph.event(continuation.event_index)
        readiness_consumer = event.consumers[continuation.consumer_index]
        producers_by_key = _materialized_producer_tasks_by_key(
            readiness_graph,
            continuation.event_index,
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
