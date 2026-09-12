#!/usr/bin/env python3
"""Inspect the production scheduler on small Qwen- and Gemma-shaped DAGs.

This is intentionally a CPU-only structural probe.  It uses the compiler's
real ``ReadinessGraph``, counter plans, root task orders, and global list
scheduler, then independently materializes the small result to check:

* every logical CTA is owned exactly once;
* every readiness producer has an earlier global slot than its consumer;
* same-worker order plus readiness edges is acyclic;
* the list schedule does not worsen unit-body critical-path makespan; and
* event consumers begin as soon as their worker strand permits after release.

The model shapes preserve the important topology rather than GPU work:

* Qwen FFN: interleaved gate/up split-K -> activation -> down tiles -> reduce.
* Gemma A4B: interleaved routed-expert split-K -> activation -> down tiles ->
  per-token routed reduction.

Run from the repository root, for example::

    python benchmarks/cross_loop_scheduling/model_shaped_scheduler_check.py
    python benchmarks/cross_loop_scheduling/model_shaped_scheduler_check.py \
        --case qwen --workers 8 --show-edges
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
from pathlib import Path
import sys
from typing import NamedTuple

import sympy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from helion._compiler import cross_loop_scheduler as scheduler  # noqa: E402
from helion._compiler.tile_dependency import (  # noqa: E402
    CoordinateDomain,
    CoordinateRelation,
    _CoordinateRelationPiece,
    coordinate_axis_symbol,
    pid_task_order,
)


Task = tuple[int, int]
Slot = tuple[int, int, int]


class ShapeCase(NamedTuple):
    name: str
    root_names: tuple[str, ...]
    graph: scheduler.ReadinessGraph
    counters: tuple[scheduler.ReadinessCounterPlan, ...]


class ScheduleMetrics(NamedTuple):
    completion: int
    critical_handoffs: int
    occupied_waves: int
    event_release_delays: tuple[int, ...]


def _domain(
    *axis_counts: tuple[int, int],
    kind: str = "site",
    identity: int | None = None,
) -> CoordinateDomain:
    return CoordinateDomain(
        axis_order=tuple(axis for axis, _count in axis_counts),
        axis_counts_items=axis_counts,
        kind=kind,
        identity=identity,
    )


def _point_map(
    source: CoordinateDomain,
    target: CoordinateDomain,
    *coordinates: sympy.Expr,
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
                coordinates,
            ),
        ),
    )


def _box_by_key(
    key_domain: CoordinateDomain,
    task_domain: CoordinateDomain,
    target_ranges: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
) -> CoordinateRelation:
    return CoordinateRelation(
        source_domain=key_domain,
        target_domain=task_domain,
        pieces=(
            _CoordinateRelationPiece(
                source_bounds_items=tuple(
                    (axis, 0, key_domain.axis_count_expressions[axis], 1)
                    for axis in key_domain.axis_order
                ),
                target_ranges=target_ranges,
            ),
        ),
    )


def _graph(
    domains: tuple[CoordinateDomain, ...],
    events: tuple[scheduler.ReadinessEvent, ...],
) -> tuple[
    scheduler.ReadinessGraph,
    tuple[scheduler.ReadinessCounterPlan, ...],
]:
    identified = tuple(
        dataclasses.replace(domain, identity=root)
        for root, domain in enumerate(domains)
    )
    # Event relations are built against these identified objects by callers.
    assert identified == domains
    task_orders = tuple(
        pid_task_order(domain, domain.axis_order) for domain in identified
    )
    graph = scheduler.ReadinessGraph(task_orders, events)
    counters = tuple(
        scheduler.ReadinessCounterPlan(event.producers, event.consumers)
        for event in events
    )
    return graph, counters


def qwen_ffn_case(
    *,
    tokens: int,
    gate_splits: int,
    down_tiles: int,
) -> ShapeCase:
    gate = _domain((10, tokens), (11, gate_splits), identity=0)
    activation = _domain((20, tokens), identity=1)
    down = _domain((30, tokens), (31, down_tiles), identity=2)
    reduction = _domain((40, tokens), identity=3)

    token = coordinate_axis_symbol(0)
    gate_event_domain = _domain((0, tokens), kind="event", identity=0)
    gate_event = scheduler.ReadinessEvent(
        producers=(
            scheduler.ReadinessProducer(
                producer_root=0,
                producers_by_key=_box_by_key(
                    gate_event_domain,
                    gate,
                    (
                        (10, token, token + 1, 1),
                        (11, sympy.Integer(0), sympy.Integer(gate_splits), 1),
                    ),
                ),
            ),
        ),
        consumers=(
            scheduler.ReadinessConsumer(
                consumer_root=1,
                keys_by_consumer=_point_map(
                    activation,
                    gate_event_domain,
                    coordinate_axis_symbol(20),
                ),
            ),
        ),
    )

    activation_event_domain = _domain((0, tokens), kind="event", identity=1)
    activation_event = scheduler.ReadinessEvent(
        producers=(
            scheduler.ReadinessProducer(
                producer_root=1,
                producers_by_key=_box_by_key(
                    activation_event_domain,
                    activation,
                    ((20, token, token + 1, 1),),
                ),
            ),
        ),
        consumers=(
            scheduler.ReadinessConsumer(
                consumer_root=2,
                keys_by_consumer=_point_map(
                    down,
                    activation_event_domain,
                    coordinate_axis_symbol(30),
                ),
            ),
        ),
    )

    down_event_domain = _domain((0, tokens), kind="event", identity=2)
    down_event = scheduler.ReadinessEvent(
        producers=(
            scheduler.ReadinessProducer(
                producer_root=2,
                producers_by_key=_box_by_key(
                    down_event_domain,
                    down,
                    (
                        (30, token, token + 1, 1),
                        (31, sympy.Integer(0), sympy.Integer(down_tiles), 1),
                    ),
                ),
            ),
        ),
        consumers=(
            scheduler.ReadinessConsumer(
                consumer_root=3,
                keys_by_consumer=_point_map(
                    reduction,
                    down_event_domain,
                    coordinate_axis_symbol(40),
                ),
            ),
        ),
    )

    graph, counters = _graph(
        (gate, activation, down, reduction),
        (gate_event, activation_event, down_event),
    )
    return ShapeCase(
        "qwen_ffn",
        ("gate/up", "silu*up", "down", "reduce"),
        graph,
        counters,
    )


def gemma_a4b_case(
    *,
    batch: int,
    routes: int,
    gate_splits: int,
    down_tiles: int,
) -> ShapeCase:
    gate = _domain(
        (10, batch),
        (11, routes),
        (12, gate_splits),
        identity=0,
    )
    activation = _domain((20, batch), (21, routes), identity=1)
    down = _domain((30, batch), (31, routes), (32, down_tiles), identity=2)
    reduction = _domain((40, batch), identity=3)

    token = coordinate_axis_symbol(0)
    route = coordinate_axis_symbol(1)
    gate_event_domain = _domain(
        (0, batch),
        (1, routes),
        kind="event",
        identity=0,
    )
    gate_event = scheduler.ReadinessEvent(
        producers=(
            scheduler.ReadinessProducer(
                producer_root=0,
                producers_by_key=_box_by_key(
                    gate_event_domain,
                    gate,
                    (
                        (10, token, token + 1, 1),
                        (11, route, route + 1, 1),
                        (12, sympy.Integer(0), sympy.Integer(gate_splits), 1),
                    ),
                ),
            ),
        ),
        consumers=(
            scheduler.ReadinessConsumer(
                consumer_root=1,
                keys_by_consumer=_point_map(
                    activation,
                    gate_event_domain,
                    coordinate_axis_symbol(20),
                    coordinate_axis_symbol(21),
                ),
            ),
        ),
    )

    activation_event_domain = _domain(
        (0, batch),
        (1, routes),
        kind="event",
        identity=1,
    )
    activation_event = scheduler.ReadinessEvent(
        producers=(
            scheduler.ReadinessProducer(
                producer_root=1,
                producers_by_key=_box_by_key(
                    activation_event_domain,
                    activation,
                    (
                        (20, token, token + 1, 1),
                        (21, route, route + 1, 1),
                    ),
                ),
            ),
        ),
        consumers=(
            scheduler.ReadinessConsumer(
                consumer_root=2,
                keys_by_consumer=_point_map(
                    down,
                    activation_event_domain,
                    coordinate_axis_symbol(30),
                    coordinate_axis_symbol(31),
                ),
            ),
        ),
    )

    reduction_event_domain = _domain((0, batch), kind="event", identity=2)
    reduction_event = scheduler.ReadinessEvent(
        producers=(
            scheduler.ReadinessProducer(
                producer_root=2,
                producers_by_key=_box_by_key(
                    reduction_event_domain,
                    down,
                    (
                        (30, token, token + 1, 1),
                        (31, sympy.Integer(0), sympy.Integer(routes), 1),
                        (32, sympy.Integer(0), sympy.Integer(down_tiles), 1),
                    ),
                ),
            ),
        ),
        consumers=(
            scheduler.ReadinessConsumer(
                consumer_root=3,
                keys_by_consumer=_point_map(
                    reduction,
                    reduction_event_domain,
                    coordinate_axis_symbol(40),
                ),
            ),
        ),
    )

    graph, counters = _graph(
        (gate, activation, down, reduction),
        (gate_event, activation_event, reduction_event),
    )
    return ShapeCase(
        "gemma_a4b",
        ("expert gate/up", "expert silu", "expert down", "token reduce"),
        graph,
        counters,
    )


def _placements(schedule: scheduler.WorkerSchedule) -> dict[Task, Slot]:
    placements: dict[Task, Slot] = {}
    occupied_slots: set[Slot] = set()
    for segment in schedule.segments:
        relation = segment.task_order
        source_domain = relation.source_domain
        for source_index in range(source_domain.size):
            source = source_domain.coordinates(source_index)
            targets = relation.target_coordinates(source)
            if not targets:
                continue
            if len(targets) != 1:
                raise AssertionError("one schedule slot maps to multiple CTAs")
            slot = tuple(source[axis] for axis in source_domain.axis_order)
            assert len(slot) == 3
            typed_slot = (slot[0], slot[1], slot[2])
            if typed_slot in occupied_slots:
                raise AssertionError(f"duplicate occupied slot {typed_slot}")
            occupied_slots.add(typed_slot)
            task = relation.target_domain.index(
                dict(
                    zip(
                        relation.target_domain.axis_order,
                        next(iter(targets)),
                        strict=True,
                    )
                )
            )
            logical = (segment.root, task)
            if logical in placements:
                raise AssertionError(f"duplicate logical CTA {logical}")
            placements[logical] = typed_slot
    return placements


def _event_edges(
    graph: scheduler.ReadinessGraph,
) -> tuple[tuple[Task, Task, int, int], ...]:
    edges: set[tuple[Task, Task, int, int]] = set()
    for event in graph.events:
        producers_by_key: list[set[Task]] = [
            set() for _ in range(event.readiness_key_count)
        ]
        for producer in event.producers:
            for key, producer_tasks in enumerate(
                producer.producers_by_key.materialize()
            ):
                producers_by_key[key].update(
                    (producer.producer_root, task) for task in producer_tasks
                )
        for consumer in event.consumers:
            for consumer_task, keys in enumerate(
                consumer.keys_by_consumer.materialize()
            ):
                for key in keys:
                    for producer_task in producers_by_key[key]:
                        edges.add(
                            (
                                producer_task,
                                (consumer.consumer_root, consumer_task),
                                event.event_id,
                                key,
                            )
                        )
    return tuple(sorted(edges))


def _evaluate(
    case: ShapeCase,
    schedule: scheduler.WorkerSchedule,
) -> ScheduleMetrics:
    placements = _placements(schedule)
    expected_tasks = {
        (root, task)
        for root, domain in enumerate(case.graph.root_domains)
        for task in range(domain.size)
    }
    if placements.keys() != expected_tasks:
        missing = sorted(expected_tasks - placements.keys())
        extra = sorted(placements.keys() - expected_tasks)
        raise AssertionError(f"coverage mismatch: missing={missing[:8]}, extra={extra[:8]}")

    # (predecessor, handoff charge, edge description)
    incoming: dict[Task, dict[Task, tuple[int, str]]] = {
        task: {} for task in expected_tasks
    }
    tasks_by_strand: dict[tuple[int, int], list[tuple[int, Task]]] = defaultdict(list)
    for task, (stage, worker, wave) in placements.items():
        tasks_by_strand[(stage, worker)].append((wave, task))
    for strand_tasks in tasks_by_strand.values():
        strand_tasks.sort()
        for (_wave_a, predecessor), (_wave_b, consumer) in zip(
            strand_tasks,
            strand_tasks[1:],
        ):
            incoming[consumer][predecessor] = (0, "strand")

    event_edges = _event_edges(case.graph)
    for producer, consumer, event_id, key in event_edges:
        producer_slot = placements[producer]
        consumer_slot = placements[consumer]
        producer_global_slot = producer_slot[2] * schedule.worker_count + producer_slot[1]
        consumer_global_slot = consumer_slot[2] * schedule.worker_count + consumer_slot[1]
        if producer_global_slot >= consumer_global_slot:
            raise AssertionError(
                "readiness rank violation: "
                f"event={event_id} key={key} producer={producer}@{producer_slot} "
                f"consumer={consumer}@{consumer_slot}"
            )
        charge = int(producer_slot[:2] != consumer_slot[:2])
        previous = incoming[consumer].get(producer)
        if previous is None or charge > previous[0]:
            incoming[consumer][producer] = (charge, f"event {event_id}:{key}")

    successors: dict[Task, set[Task]] = {task: set() for task in expected_tasks}
    indegree = dict.fromkeys(expected_tasks, 0)
    for consumer, predecessors in incoming.items():
        for predecessor in predecessors:
            if consumer not in successors[predecessor]:
                successors[predecessor].add(consumer)
                indegree[consumer] += 1
    ready = sorted(task for task, degree in indegree.items() if degree == 0)
    scores: dict[Task, tuple[int, int]] = {}
    while ready:
        task = ready.pop(0)
        predecessor_scores = [
            (scores[predecessor][0], scores[predecessor][1] + charge)
            for predecessor, (charge, _description) in incoming[task].items()
        ]
        best_before = max(predecessor_scores, default=(0, 0))
        scores[task] = (best_before[0] + 1, best_before[1])
        for successor in sorted(successors[task]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
        ready.sort()
    if len(scores) != len(expected_tasks):
        blocked = sorted(task for task, degree in indegree.items() if degree)
        raise AssertionError(f"schedule contains a cycle involving {blocked[:8]}")

    release_delays: list[int] = []
    edge_groups: dict[tuple[int, int, Task], list[Task]] = defaultdict(list)
    for producer, consumer, event_id, key in event_edges:
        edge_groups[(event_id, key, consumer)].append(producer)
    for producers_and_consumer in edge_groups.items():
        (_event_id, _key, consumer), producers = producers_and_consumer
        producer_completion = max(scores[producer][0] for producer in producers)
        consumer_start = scores[consumer][0] - 1
        release_delays.append(consumer_start - producer_completion)
    if any(delay < 0 for delay in release_delays):
        raise AssertionError("a consumer starts before its readiness event completes")

    final_score = max(scores.values(), default=(0, 0))
    occupied_waves = 1 + max((slot[2] for slot in placements.values()), default=-1)
    return ScheduleMetrics(
        completion=final_score[0],
        critical_handoffs=final_score[1],
        occupied_waves=occupied_waves,
        event_release_delays=tuple(release_delays),
    )


def _task_label(
    case: ShapeCase,
    task: Task,
) -> str:
    root, ordinal = task
    domain = case.graph.root_domains[root]
    coordinates = domain.coordinates(ordinal)
    coordinate_text = ",".join(
        str(coordinates[axis]) for axis in domain.axis_order
    )
    return f"r{root}:{coordinate_text}"


def _print_timeline(
    case: ShapeCase,
    schedule: scheduler.WorkerSchedule,
    *,
    title: str,
) -> None:
    placements = _placements(schedule)
    by_slot = {slot: task for task, slot in placements.items()}
    maximum_wave = max((slot[2] for slot in placements.values()), default=-1)
    print(f"\n{title}")
    print("wave | " + " | ".join(f"w{worker}" for worker in range(schedule.worker_count)))
    print("-" * (8 + 13 * schedule.worker_count))
    for wave in range(maximum_wave + 1):
        cells = []
        for worker in range(schedule.worker_count):
            task = by_slot.get((1, worker, wave))
            cells.append("." if task is None else _task_label(case, task))
        print(f"{wave:>4} | " + " | ".join(f"{cell:<9}" for cell in cells))


def _print_event_summary(
    case: ShapeCase,
    schedule: scheduler.WorkerSchedule,
) -> None:
    placements = _placements(schedule)
    grouped: dict[tuple[int, int, Task], list[Task]] = defaultdict(list)
    for producer, consumer, event_id, key in _event_edges(case.graph):
        grouped[(event_id, key, consumer)].append(producer)
    print("\nevent releases (placement waves)")
    for (event_id, key, consumer), producers in sorted(grouped.items()):
        producer_wave = max(placements[task][2] for task in producers)
        consumer_wave = placements[consumer][2]
        print(
            f"  e{event_id} key={key:<2} -> {_task_label(case, consumer):<9} "
            f"producers_done=w{producer_wave}, consumer=w{consumer_wave}, "
            f"wave_gap={consumer_wave - producer_wave}"
        )


def run_case(case: ShapeCase, *, workers: int, show_edges: bool) -> None:
    baseline = scheduler.build_baseline_worker_schedule(
        case.graph.root_domains,
        case.graph.root_task_orders,
        workers,
    )
    scheduled = scheduler._global_unit_list_schedule(
        case.graph,
        baseline,
        case.counters,
        frozenset(),
    )
    if scheduled is None:
        raise AssertionError(f"{case.name}: production list scheduler declined")

    baseline_metrics = _evaluate(case, baseline)
    scheduled_metrics = _evaluate(case, scheduled)
    # ``completion`` is the unit-body makespan oracle.  Cross-strand handoffs
    # remain useful diagnostics, but they are not an optimization objective:
    # the production event-frontier policy may deliberately trade an extra
    # handoff for an earlier complete readiness cohort.
    if scheduled_metrics.completion > baseline_metrics.completion:
        raise AssertionError(
            f"{case.name}: list schedule worsened unit-body makespan: "
            f"baseline={baseline_metrics.completion}, "
            f"scheduled={scheduled_metrics.completion}"
        )
    if scheduled_metrics.occupied_waves > baseline_metrics.occupied_waves:
        raise AssertionError(
            f"{case.name}: list schedule grew the occupied-wave horizon"
        )

    print(f"\n{'=' * 80}\n{case.name}")
    print("roots: " + ", ".join(f"r{i}={name}" for i, name in enumerate(case.root_names)))
    print(
        "baseline: "
        f"completion={baseline_metrics.completion}, "
        f"critical_handoffs={baseline_metrics.critical_handoffs}, "
        f"waves={baseline_metrics.occupied_waves}"
    )
    print(
        "scheduled: "
        f"completion={scheduled_metrics.completion}, "
        f"critical_handoffs={scheduled_metrics.critical_handoffs}, "
        f"waves={scheduled_metrics.occupied_waves}, "
        f"zero-delay releases={sum(delay == 0 for delay in scheduled_metrics.event_release_delays)}"
        f"/{len(scheduled_metrics.event_release_delays)}"
    )
    _print_timeline(case, baseline, title="baseline root-major")
    _print_timeline(case, scheduled, title="production list schedule")
    _print_event_summary(case, scheduled)
    if show_edges:
        print("\nreadiness edges")
        for producer, consumer, event_id, key in _event_edges(case.graph):
            print(
                f"  e{event_id}:{key} {_task_label(case, producer)} -> "
                f"{_task_label(case, consumer)}"
            )
    print(f"\nPASS: {case.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "qwen", "gemma"), default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--show-edges", action="store_true")
    parser.add_argument("--qwen-tokens", type=int, default=4)
    parser.add_argument("--qwen-gate-splits", type=int, default=6)
    parser.add_argument("--qwen-down-tiles", type=int, default=3)
    parser.add_argument("--gemma-batch", type=int, default=2)
    parser.add_argument("--gemma-routes", type=int, default=2)
    parser.add_argument("--gemma-gate-splits", type=int, default=4)
    parser.add_argument("--gemma-down-tiles", type=int, default=2)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    cases: list[ShapeCase] = []
    if args.case in ("all", "qwen"):
        cases.append(
            qwen_ffn_case(
                tokens=args.qwen_tokens,
                gate_splits=args.qwen_gate_splits,
                down_tiles=args.qwen_down_tiles,
            )
        )
    if args.case in ("all", "gemma"):
        cases.append(
            gemma_a4b_case(
                batch=args.gemma_batch,
                routes=args.gemma_routes,
                gate_splits=args.gemma_gate_splits,
                down_tiles=args.gemma_down_tiles,
            )
        )
    for case in cases:
        run_case(case, workers=args.workers, show_edges=args.show_edges)


if __name__ == "__main__":
    main()
