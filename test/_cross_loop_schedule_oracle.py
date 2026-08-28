from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helion._compiler.cross_loop_scheduler import EventGraph
    from helion._compiler.cross_loop_scheduler import EventUse
    from helion._compiler.cross_loop_scheduler import LocalTrigger
    from helion._compiler.cross_loop_scheduler import WorkerSchedule
    from helion._compiler.cross_loop_scheduler import WorkerScheduleSegment
    from helion._compiler.tile_dependency import LogicalRelation


def event_source_traversal(
    event_graph: EventGraph,
    use: EventUse,
) -> tuple[int, ...]:
    """Return one event use's exhaustive source order for test materialization."""
    root_axes = event_graph.root_domains[use.consumer_root].axis_order
    if use.consumer_scope_id is None:
        return root_axes
    nested_axes = tuple(
        axis for axis in use.keys.source_domain.axis_order if axis not in root_axes
    )
    return (*nested_axes, *root_axes)


def required_keys_by_strand(
    event_graph: EventGraph,
    use: EventUse,
) -> LogicalRelation | None:
    """Project a test event use onto its owning root task strands."""
    root_domain = event_graph.root_domains[use.consumer_root]
    if use.consumer_scope_id is None:
        if use.keys.source_domain != root_domain:
            raise ValueError("root event use has the wrong source domain")
        return use.keys
    return use.keys.project_source(root_domain)


def segment_task_for_offset(
    segment: WorkerScheduleSegment,
    task_offset: int,
) -> int:
    """Materialize one segment ordinal for small scheduler tests."""
    if not 0 <= task_offset < segment.task_count:
        raise IndexError(task_offset)
    source_coordinates = segment.task_relation.source_domain.coordinates(task_offset)
    targets = segment.task_relation.target_coordinates(source_coordinates)
    if len(targets) != 1:
        raise AssertionError("symbolic schedule ordinal does not map to one task")
    return segment.task_relation.target_domain.index(
        dict(
            zip(
                segment.task_relation.target_domain.axis_order,
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
    inverse = segment.task_relation.inverse()
    offsets = (
        inverse.targets(task)
        if inverse is not None
        else frozenset(
            offset
            for offset in range(segment.task_count)
            if segment_task_for_offset(segment, offset) == task
        )
    )
    if len(offsets) > 1:
        raise AssertionError("symbolic schedule maps one task more than once")
    if not offsets:
        return None
    schedule_offset = segment.schedule_for_offset(next(iter(offsets)))
    return (
        segment.worker_begin + schedule_offset % segment.worker_count,
        schedule_offset // segment.worker_count,
    )


def segment_task_at(
    segment: WorkerScheduleSegment,
    worker: int,
    position: int,
) -> int | None:
    """Materialize the task at one segment position for small tests."""
    worker_offset = worker - segment.worker_begin
    if not 0 <= worker_offset < segment.worker_count or position < 0:
        return None
    schedule_delta = (
        position * segment.worker_count + worker_offset - segment.schedule_begin
    )
    if not 0 <= schedule_delta < segment.task_count:
        return None
    return segment_task_for_offset(segment, schedule_delta)


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
    position: int,
) -> tuple[int, int] | None:
    """Materialize the task at one worker position for small tests."""
    tasks = tuple(
        (segment.root, task)
        for segment in schedule.segments
        if (task := segment_task_at(segment, worker, position)) is not None
    )
    if len(tasks) > 1:
        raise AssertionError(f"worker {worker} position {position} has multiple tasks")
    return tasks[0] if tasks else None


def task_order(schedule: WorkerSchedule, root: int) -> tuple[int, ...]:
    """Materialize one root's order for small scheduler tests."""
    placed_tasks: list[tuple[int, int]] = []
    for segment in schedule.segments_for_root(root):
        for task_offset in range(segment.task_count):
            schedule_offset = segment.schedule_for_offset(task_offset)
            task = segment_task_for_offset(segment, task_offset)
            placed_tasks.append((schedule_offset, task))
    placed_tasks.sort()
    if any(
        left_offset == right_offset
        for (left_offset, _), (right_offset, _) in itertools.pairwise(placed_tasks)
    ):
        raise AssertionError(f"root {root} has overlapping schedule segments")
    return tuple(task for _offset, task in placed_tasks)


def _materialized_contributor_tasks_by_key(
    event_graph: EventGraph,
    event_index: int,
) -> tuple[frozenset[tuple[int, int]], ...]:
    event = event_graph.event(event_index)
    result: list[set[tuple[int, int]]] = [set() for _ in range(event.key_count)]
    for contribution in event.contributions:
        key_to_strands = contribution.predecessors.project_target(
            event_graph.root_domains[contribution.producer_root]
        )
        if key_to_strands is None:
            raise ValueError(
                "event contribution cannot be projected onto producer strands"
            )
        tasks_by_key = key_to_strands.materialize()
        for key, tasks in enumerate(tasks_by_key):
            result[key].update((contribution.producer_root, task) for task in tasks)
    return tuple(frozenset(tasks) for tasks in result)


def _local_trigger_predecessors(
    event_graph: EventGraph,
    local_triggers: tuple[LocalTrigger, ...],
) -> dict[tuple[int, int], frozenset[tuple[int, int]]]:
    result: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for trigger in local_triggers:
        event = event_graph.event(trigger.event_index)
        use = event.uses[trigger.use_index]
        contributors_by_key = _materialized_contributor_tasks_by_key(
            event_graph,
            trigger.event_index,
        )
        for consumer_task, required_keys in enumerate(
            use.keys.materialize(
                source_traversal=event_source_traversal(event_graph, use)
            )
        ):
            if len(required_keys) != 1:
                raise ValueError("a local trigger requires exactly one key per task")
            key = next(iter(required_keys))
            task = (use.consumer_root, consumer_task)
            if task in result:
                raise ValueError(f"task {task} has multiple local triggers")
            result[task] = contributors_by_key[key]
    return result


def _static_ancestors(
    task: tuple[int, int],
    *,
    worker_schedule: WorkerSchedule,
    local_predecessors: dict[tuple[int, int], frozenset[tuple[int, int]]],
    cache: dict[tuple[int, int], frozenset[tuple[int, int]]],
    visiting: frozenset[tuple[int, int]] = frozenset(),
) -> frozenset[tuple[int, int]]:
    if task in cache:
        return cache[task]
    if placement(worker_schedule, *task) is not None:
        result = frozenset((task,))
    elif task in visiting:
        raise ValueError("local trigger graph contains a cycle")
    elif (predecessors := local_predecessors.get(task)) is None:
        result = frozenset()
    else:
        result = frozenset(
            ancestor
            for predecessor in predecessors
            for ancestor in _static_ancestors(
                predecessor,
                worker_schedule=worker_schedule,
                local_predecessors=local_predecessors,
                cache=cache,
                visiting=visiting | frozenset((task,)),
            )
        )
    cache[task] = result
    return result


def validate_worker_schedule(
    event_graph: EventGraph,
    worker_schedule: WorkerSchedule,
    local_triggers: tuple[LocalTrigger, ...] = (),
) -> None:
    """Exhaustively validate small schedules without entering production."""
    task_nodes = {
        (root, task)
        for root, domain in enumerate(event_graph.root_domains)
        for task in range(domain.size)
    }
    local_predecessors = _local_trigger_predecessors(event_graph, local_triggers)
    static_tasks = task_nodes - local_predecessors.keys()
    tasks_by_worker: list[list[tuple[int, tuple[int, int]]]] = [
        [] for _ in range(worker_schedule.worker_count)
    ]
    for root, task in sorted(task_nodes):
        task_placement = placement(worker_schedule, root, task)
        if (root, task) in local_predecessors:
            if task_placement is not None:
                raise ValueError(
                    f"locally executed task ({root}, {task}) also has a static placement"
                )
            continue
        if task_placement is None:
            raise ValueError(f"task ({root}, {task}) has no static placement")
        worker, position = task_placement
        tasks_by_worker[worker].append((position, (root, task)))

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
            left_position == right_position
            for (left_position, _), (right_position, _) in itertools.pairwise(
                worker_tasks
            )
        ):
            raise ValueError("multiple tasks occupy one worker schedule position")
        for (_, producer), (_, consumer) in itertools.pairwise(worker_tasks):
            add_edge(("task", *producer), ("task", *consumer))

    for event in event_graph.events:
        contributors_by_key = _materialized_contributor_tasks_by_key(
            event_graph,
            event.event_id,
        )
        consumers_by_key: list[set[tuple[int, int]]] = [
            set() for _ in range(event.key_count)
        ]
        for use in event.uses:
            strand_keys = required_keys_by_strand(event_graph, use)
            if strand_keys is None:
                raise ValueError("event use cannot be projected onto consumer strands")
            for consumer_task, required_keys in enumerate(strand_keys.materialize()):
                consumer = (use.consumer_root, consumer_task)
                if consumer in local_predecessors:
                    continue
                for key in required_keys:
                    consumers_by_key[key].add(consumer)
        for key, consumers in enumerate(consumers_by_key):
            if not consumers:
                continue
            event_node = ("event", event.event_id, key)
            for producer in contributors_by_key[key]:
                ancestors = _static_ancestors(
                    producer,
                    worker_schedule=worker_schedule,
                    local_predecessors=local_predecessors,
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
