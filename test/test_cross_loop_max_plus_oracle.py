"""Concrete test oracle for the scheduler's unit-weight max-plus objective.

The helpers deliberately materialize schedule and readiness relations and must
remain test-only.  Root bodies are atomic; nested-site timing is rejected.
Single-body final-arrival ownership is modeled exactly, while correlated
multi-body ownership remains an explicit gap.
"""

from __future__ import annotations

import itertools
from typing import TypeAlias

from helion._compiler.cross_loop_scheduler import FinalArrivalContinuation
from helion._compiler.cross_loop_scheduler import ReadinessConsumer
from helion._compiler.cross_loop_scheduler import ReadinessEvent
from helion._compiler.cross_loop_scheduler import ReadinessGraph
from helion._compiler.cross_loop_scheduler import ReadinessProducer
from helion._compiler.cross_loop_scheduler import WorkerSchedule
from helion._compiler.cross_loop_scheduler import WorkerScheduleSegment
from helion._compiler.tile_dependency import CoordinateDomain
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import _CoordinateRelationPiece
from helion._compiler.tile_dependency import _remember_exact_converse
from helion._compiler.tile_dependency import pid_task_order
from helion._testing import TestCase

_Node: TypeAlias = tuple[int, int]
_Slot: TypeAlias = tuple[int, int, int]
_Score: TypeAlias = tuple[int, int]
_Edge: TypeAlias = tuple[_Node, _Node]
_WeightedEdge: TypeAlias = tuple[_Node, _Node, int]


class _OracleInputError(ValueError):
    pass


class _ContinuationOracleGap(ValueError):
    pass


def _domains(task_counts: tuple[int, ...]) -> tuple[CoordinateDomain, ...]:
    return tuple(
        CoordinateDomain(
            (100 + root,),
            ((100 + root, task_count),),
            ((100 + root, 1),),
            kind="site",
            identity=root,
        )
        for root, task_count in enumerate(task_counts)
    )


def _key_to_tasks(
    key_domain: CoordinateDomain,
    task_domain: CoordinateDomain,
    tasks: tuple[int, ...],
) -> CoordinateRelation:
    (task_axis,) = task_domain.axis_order
    return CoordinateRelation(
        key_domain,
        task_domain,
        tuple(
            _CoordinateRelationPiece(
                ((0, 0, 1, 1),),
                ((task_axis, task, task + 1, 1),),
            )
            for task in tasks
        ),
    )


def _tasks_to_key(
    task_domain: CoordinateDomain,
    key_domain: CoordinateDomain,
    tasks: tuple[int, ...],
) -> CoordinateRelation:
    (task_axis,) = task_domain.axis_order
    return CoordinateRelation.point_map(
        task_domain,
        key_domain,
        tuple(
            (
                ((task_axis, task, task + 1, 1),),
                (0,),
            )
            for task in tasks
        ),
    )


def _readiness_graph(
    task_counts: tuple[int, ...],
    event_edges: tuple[tuple[tuple[_Node, ...], tuple[_Node, ...]], ...],
) -> ReadinessGraph:
    root_domains = _domains(task_counts)
    events: list[ReadinessEvent] = []
    for event_id, (producer_nodes, consumer_nodes) in enumerate(event_edges):
        key_domain = CoordinateDomain(
            (0,),
            ((0, 1),),
            kind="event",
            identity=event_id,
        )
        producers_by_root: dict[int, list[int]] = {}
        consumers_by_root: dict[int, list[int]] = {}
        for root, task in producer_nodes:
            producers_by_root.setdefault(root, []).append(task)
        for root, task in consumer_nodes:
            consumers_by_root.setdefault(root, []).append(task)
        events.append(
            ReadinessEvent(
                producers=tuple(
                    ReadinessProducer(
                        root,
                        _key_to_tasks(
                            key_domain,
                            root_domains[root],
                            tuple(sorted(tasks)),
                        ),
                    )
                    for root, tasks in sorted(producers_by_root.items())
                ),
                consumers=tuple(
                    ReadinessConsumer(
                        root,
                        _tasks_to_key(
                            root_domains[root],
                            key_domain,
                            tuple(sorted(tasks)),
                        ),
                    )
                    for root, tasks in sorted(consumers_by_root.items())
                ),
            )
        )
    return ReadinessGraph(
        tuple(pid_task_order(domain, domain.axis_order) for domain in root_domains),
        tuple(events),
    )


def _schedule(
    graph: ReadinessGraph,
    placements: dict[_Node, tuple[int, int]],
    *,
    worker_count: int,
) -> WorkerSchedule:
    wave_count = max((wave for _worker, wave in placements.values()), default=0) + 1
    placement_domain = CoordinateDomain(
        (-3, -2, -1),
        ((-3, 2), (-2, worker_count), (-1, wave_count)),
        kind="worker",
    )
    segments: list[WorkerScheduleSegment] = []
    for root, root_domain in enumerate(graph.root_domains):
        root_placements = tuple(
            sorted(
                (
                    task,
                    placements[(root, task)],
                )
                for task in range(root_domain.size)
                if (root, task) in placements
            )
        )
        if not root_placements:
            continue
        (task_axis,) = root_domain.axis_order
        task_order = CoordinateRelation.point_map(
            placement_domain,
            root_domain,
            tuple(
                (
                    (
                        (-3, 1, 2, 1),
                        (-2, worker, worker + 1, 1),
                        (-1, wave, wave + 1, 1),
                    ),
                    (task,),
                )
                for task, (worker, wave) in root_placements
            ),
        )
        converse = CoordinateRelation.point_map(
            root_domain,
            placement_domain,
            tuple(
                (
                    ((task_axis, task_index, task_index + 1, 1),),
                    (1, worker, wave),
                )
                for task_index, (worker, wave) in root_placements
            ),
        )
        _remember_exact_converse(task_order, converse)
        segments.append(
            WorkerScheduleSegment(
                root=root,
                task_order=task_order,
                worker_begin=0,
                worker_count=worker_count,
                dispatch_offset=0,
            )
        )
    return WorkerSchedule(worker_count, tuple(segments))


def _all_nodes(graph: ReadinessGraph) -> frozenset[_Node]:
    return frozenset(
        (root, task)
        for root, domain in enumerate(graph.root_domains)
        for task in range(domain.size)
    )


def _materialized_schedule(
    schedule: WorkerSchedule,
) -> tuple[dict[_Node, _Slot], frozenset[_Edge], dict[_Node, _Node]]:
    owners: dict[_Node, _Slot] = {}
    nodes_by_slot: dict[_Slot, _Node] = {}
    for segment in schedule.segments:
        relation = segment.task_order
        for source_index, targets in enumerate(relation.materialize()):
            if not targets:
                continue
            if len(targets) != 1:
                raise _OracleInputError("one schedule slot owns multiple bodies")
            (task,) = targets
            coordinates = relation.source_domain.coordinates(source_index)
            slot = tuple(
                coordinates[axis] for axis in relation.source_domain.axis_order
            )
            node = (segment.root, task)
            if node in owners:
                raise _OracleInputError(f"duplicate ownership for {node}")
            if slot in nodes_by_slot:
                raise _OracleInputError(f"duplicate schedule slot {slot}")
            owners[node] = slot
            nodes_by_slot[slot] = node

    strand_edges: set[_Edge] = set()
    successor: dict[_Node, _Node] = {}
    nodes_by_strand: dict[tuple[int, int], list[tuple[int, _Node]]] = {}
    for node, (launch_stage, worker, wave) in owners.items():
        nodes_by_strand.setdefault((launch_stage, worker), []).append((wave, node))
    for entries in nodes_by_strand.values():
        ordered = [node for _wave, node in sorted(entries)]
        for predecessor, following in itertools.pairwise(ordered):
            strand_edges.add((predecessor, following))
            successor[predecessor] = following
    return owners, frozenset(strand_edges), successor


def _materialized_readiness(graph: ReadinessGraph) -> frozenset[_Edge]:
    edges: set[_Edge] = set()
    for event in graph.events:
        producers_by_key = tuple(
            (producer.producer_root, producer.producers_by_key.materialize())
            for producer in event.producers
            if producer.producer_site_id is None
        )
        if len(producers_by_key) != len(event.producers):
            raise _OracleInputError("nested producer timing is outside this oracle")
        for consumer in event.consumers:
            if consumer.consumer_site_id is not None:
                raise _OracleInputError("nested consumer timing is outside this oracle")
            for consumer_task, keys in enumerate(
                consumer.keys_by_consumer.materialize()
            ):
                for key in keys:
                    for producer_root, tasks_by_key in producers_by_key:
                        edges.update(
                            (
                                (producer_root, producer_task),
                                (consumer.consumer_root, consumer_task),
                            )
                            for producer_task in tasks_by_key[key]
                        )
    return frozenset(edges)


def _weighted_edges(
    readiness_edges: frozenset[_Edge],
    strand_edges: frozenset[_Edge],
    owners: dict[_Node, _Slot],
    *,
    inline_nodes: frozenset[_Node] = frozenset(),
    insertion_edges: frozenset[_Edge] = frozenset(),
) -> frozenset[_WeightedEdge]:
    weights: dict[_Edge, int] = dict.fromkeys(strand_edges | insertion_edges, 0)
    for producer, consumer in readiness_edges:
        if producer not in owners or consumer not in owners:
            raise _OracleInputError("readiness edge has missing ownership")
        producer_strand = owners[producer][:2]
        consumer_strand = owners[consumer][:2]
        handoff = int(
            consumer not in inline_nodes and producer_strand != consumer_strand
        )
        weights[(producer, consumer)] = max(
            weights.get((producer, consumer), 0),
            handoff,
        )
    return frozenset(
        (producer, consumer, weight) for (producer, consumer), weight in weights.items()
    )


def _score_weighted_dag(
    nodes: frozenset[_Node],
    edges: frozenset[_WeightedEdge],
) -> _Score:
    predecessors: dict[_Node, list[tuple[_Node, int]]] = {node: [] for node in nodes}
    successors: dict[_Node, list[_Node]] = {node: [] for node in nodes}
    indegree = dict.fromkeys(nodes, 0)
    for producer, consumer, handoff in edges:
        if producer not in nodes or consumer not in nodes:
            raise _OracleInputError("edge names an unknown body")
        predecessors[consumer].append((producer, handoff))
        successors[producer].append(consumer)
        indegree[consumer] += 1

    ready = sorted(node for node, count in indegree.items() if count == 0)
    scores: dict[_Node, _Score] = {}
    while ready:
        node = ready.pop(0)
        best_predecessor = max(
            (
                (scores[predecessor][0], scores[predecessor][1] + handoff)
                for predecessor, handoff in predecessors[node]
            ),
            default=(0, 0),
        )
        scores[node] = (best_predecessor[0] + 1, best_predecessor[1])
        for following in successors[node]:
            indegree[following] -= 1
            if indegree[following] == 0:
                ready.append(following)
                ready.sort()
    if len(scores) != len(nodes):
        raise _OracleInputError("combined readiness/strand graph contains a cycle")
    sinks = tuple(node for node in nodes if not successors[node])
    return max((scores[node] for node in sinks), default=(0, 0))


def _score_resident(graph: ReadinessGraph, schedule: WorkerSchedule) -> _Score:
    nodes = _all_nodes(graph)
    owners, strand_edges, _successor = _materialized_schedule(schedule)
    missing = nodes - owners.keys()
    extra = owners.keys() - nodes
    if missing or extra:
        raise _OracleInputError(
            f"schedule ownership mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    readiness_edges = _materialized_readiness(graph)
    return _score_weighted_dag(
        nodes,
        _weighted_edges(readiness_edges, strand_edges, owners),
    )


def _reachable(
    source: _Node,
    target: _Node,
    edges: frozenset[_Edge],
) -> bool:
    successors: dict[_Node, set[_Node]] = {}
    for predecessor, following in edges:
        successors.setdefault(predecessor, set()).add(following)
    pending = list(successors.get(source, ()))
    seen: set[_Node] = set()
    while pending:
        node = pending.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        pending.extend(successors.get(node, ()))
    return False


def _continuation_scores(
    graph: ReadinessGraph,
    schedule_without_consumer: WorkerSchedule,
    continuation: FinalArrivalContinuation,
) -> tuple[tuple[_Node, _Score], ...]:
    event = graph.event(continuation.event_id)
    consumer = event.consumers[continuation.consumer_index]
    consumer_nodes = frozenset(
        (consumer.consumer_root, task)
        for task in range(graph.root_domains[consumer.consumer_root].size)
    )
    if len(consumer_nodes) != 1:
        raise _ContinuationOracleGap(
            "multi-body continuation ownership correlation is not modeled"
        )
    (consumer_node,) = consumer_nodes

    nodes = _all_nodes(graph)
    owners, strand_edges, successor = _materialized_schedule(schedule_without_consumer)
    missing = nodes - owners.keys()
    extra = owners.keys() - nodes
    if missing != consumer_nodes or extra:
        raise _OracleInputError(
            f"continuation ownership mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    readiness_edges = _materialized_readiness(graph)

    keys = consumer.keys_by_consumer.materialize()[consumer_node[1]]
    if len(keys) != 1:
        raise _ContinuationOracleGap("one inline body must consume exactly one key")
    (key,) = keys
    required_producers = frozenset(
        (producer.producer_root, producer_task)
        for producer in event.producers
        for producer_task in producer.producers_by_key.materialize()[key]
    )
    if not required_producers <= owners.keys() or not required_producers:
        raise _OracleInputError("continuation has missing producer ownership")

    causal_edges = readiness_edges | strand_edges
    # Check the unowned structural graph before asking which producer can win.
    _score_weighted_dag(
        nodes,
        frozenset(
            (producer, consumer_node_, 0) for producer, consumer_node_ in causal_edges
        ),
    )
    possible_publishers = tuple(
        sorted(
            producer
            for producer in required_producers
            if not any(
                producer != other and _reachable(producer, other, causal_edges)
                for other in required_producers
            )
        )
    )
    if not possible_publishers:
        raise _OracleInputError("continuation has no causally maximal publisher")

    alternatives: list[tuple[_Node, _Score]] = []
    for publisher in possible_publishers:
        alternative_owners = {**owners, consumer_node: owners[publisher]}
        alternative_strand_edges = set(strand_edges)
        insertion_edges: set[_Edge] = {(publisher, consumer_node)}
        if publisher in successor:
            following = successor[publisher]
            alternative_strand_edges.remove((publisher, following))
            insertion_edges.add((consumer_node, following))
        score = _score_weighted_dag(
            nodes,
            _weighted_edges(
                readiness_edges,
                frozenset(alternative_strand_edges),
                alternative_owners,
                inline_nodes=consumer_nodes,
                insertion_edges=frozenset(insertion_edges),
            ),
        )
        alternatives.append((publisher, score))
    return tuple(alternatives)


def _score_continuation(
    graph: ReadinessGraph,
    schedule_without_consumer: WorkerSchedule,
    continuation: FinalArrivalContinuation,
) -> _Score:
    return max(
        score
        for _publisher, score in _continuation_scores(
            graph,
            schedule_without_consumer,
            continuation,
        )
    )


def _brute_force_score(
    nodes: frozenset[_Node],
    edges: frozenset[_WeightedEdge],
) -> _Score:
    unweighted = frozenset((source, target) for source, target, _weight in edges)
    valid_order = next(
        (
            order
            for order in itertools.permutations(nodes)
            if all(
                order.index(source) < order.index(target)
                for source, target in unweighted
            )
        ),
        None,
    )
    if valid_order is None:
        raise _OracleInputError("cycle")
    successors: dict[_Node, list[tuple[_Node, int]]] = {node: [] for node in nodes}
    for source, target, weight in edges:
        successors[source].append((target, weight))

    best = (0, 0)

    def visit(node: _Node, score: _Score) -> None:
        nonlocal best
        score = (score[0] + 1, score[1])
        best = max(best, score)
        for following, handoff in successors[node]:
            visit(following, (score[0], score[1] + handoff))

    for node in nodes:
        visit(node, (0, 0))
    return best


class TestConcreteMaxPlusOracle(TestCase):
    def test_exhaustive_tiny_dags_and_schedules(self) -> None:
        nodes = frozenset((root, 0) for root in range(3))
        possible_readiness = ((0, 1), (0, 2), (1, 2))
        slots = ((0, 0), (1, 0), (0, 1), (1, 1))
        for edge_mask in range(1 << len(possible_readiness)):
            readiness = frozenset(
                ((source, 0), (target, 0))
                for bit, (source, target) in enumerate(possible_readiness)
                if edge_mask & (1 << bit)
            )
            graph = _readiness_graph(
                (1, 1, 1),
                tuple(((source,), (target,)) for source, target in sorted(readiness)),
            )
            for occupied_slots in itertools.permutations(slots, len(nodes)):
                placements = dict(zip(sorted(nodes), occupied_slots, strict=True))
                schedule = _schedule(graph, placements, worker_count=2)
                owners = {
                    node: (1, worker, wave)
                    for node, (worker, wave) in placements.items()
                }
                expected_strand_edges: set[_Edge] = set()
                for worker in range(2):
                    ordered = [
                        node
                        for node, (_worker, _wave) in sorted(
                            placements.items(),
                            key=lambda item: item[1][1],
                        )
                        if placements[node][0] == worker
                    ]
                    expected_strand_edges.update(itertools.pairwise(ordered))
                weights: dict[_Edge, int] = dict.fromkeys(expected_strand_edges, 0)
                for source, target in readiness:
                    weights[(source, target)] = int(
                        owners[source][:2] != owners[target][:2]
                    )
                weighted = frozenset(
                    (source, target, handoff)
                    for (source, target), handoff in weights.items()
                )
                try:
                    expected = _brute_force_score(nodes, weighted)
                except _OracleInputError:
                    with self.assertRaisesRegex(_OracleInputError, "cycle"):
                        _score_resident(graph, schedule)
                else:
                    self.assertEqual(_score_resident(graph, schedule), expected)

    def test_no_global_slot_order_and_input_validation(self) -> None:
        graph = _readiness_graph(
            (1, 1),
            ((((0, 0),), ((1, 0),)),),
        )
        schedule = _schedule(
            graph,
            {(0, 0): (0, 1), (1, 0): (1, 0)},
            worker_count=2,
        )
        self.assertEqual(_score_resident(graph, schedule), (2, 1))

        missing = _schedule(graph, {(0, 0): (0, 0)}, worker_count=2)
        with self.assertRaisesRegex(_OracleInputError, "missing"):
            _score_resident(graph, missing)

        cyclic = _readiness_graph(
            (1, 1),
            (
                (((0, 0),), ((1, 0),)),
                (((1, 0),), ((0, 0),)),
            ),
        )
        with self.assertRaisesRegex(_OracleInputError, "cycle"):
            _score_resident(cyclic, schedule)

    def test_final_publishers_are_mutually_exclusive_alternatives(self) -> None:
        graph = _readiness_graph(
            (1, 1, 1),
            (((((0, 0), (1, 0))), ((2, 0),)),),
        )
        candidate = _schedule(
            graph,
            {(0, 0): (0, 0), (1, 0): (1, 0)},
            worker_count=2,
        )
        alternatives = _continuation_scores(
            graph,
            candidate,
            FinalArrivalContinuation(0, 0),
        )
        self.assertEqual(
            tuple(publisher for publisher, _score in alternatives), ((0, 0), (1, 0))
        )
        self.assertEqual(
            tuple(score for _publisher, score in alternatives), ((2, 0), (2, 0))
        )

    def test_final_publishers_use_causal_maxima_not_global_wave(self) -> None:
        graph = _readiness_graph(
            (1, 1, 1, 1),
            (((((0, 0), (1, 0), (2, 0))), ((3, 0),)),),
        )
        candidate = _schedule(
            graph,
            {
                (0, 0): (0, 0),
                (1, 0): (0, 1),
                (2, 0): (1, 5),
            },
            worker_count=2,
        )
        alternatives = _continuation_scores(
            graph,
            candidate,
            FinalArrivalContinuation(0, 0),
        )
        self.assertEqual(
            tuple(publisher for publisher, _score in alternatives),
            ((1, 0), (2, 0)),
        )

    def test_multi_body_continuation_is_an_explicit_gap(self) -> None:
        graph = _readiness_graph(
            (1, 2),
            (((((0, 0),)), ((1, 0), (1, 1))),),
        )
        candidate = _schedule(graph, {(0, 0): (0, 0)}, worker_count=2)
        with self.assertRaisesRegex(_ContinuationOracleGap, "multi-body"):
            _score_continuation(
                graph,
                candidate,
                FinalArrivalContinuation(0, 0),
            )

    def test_flashmla_fan_in_early_release_improves_completion(self) -> None:
        producers = tuple((0, task) for task in range(4))
        graph = _readiness_graph(
            (4, 1, 1, 1, 1, 1, 1),
            (
                (producers, ((1, 0),)),
                ((((1, 0),)), ((2, 0),)),
            ),
        )
        baseline = _schedule(
            graph,
            {
                **{(0, task): (task, 0) for task in range(4)},
                (3, 0): (0, 1),
                (4, 0): (1, 1),
                (5, 0): (2, 1),
                (6, 0): (3, 1),
                (1, 0): (0, 2),
                (2, 0): (0, 3),
            },
            worker_count=4,
        )
        early_release = _schedule(
            graph,
            {
                **{(0, task): (task, 0) for task in range(4)},
                (1, 0): (0, 1),
                (2, 0): (0, 2),
                (3, 0): (1, 1),
                (4, 0): (2, 1),
                (5, 0): (3, 1),
                (6, 0): (1, 2),
            },
            worker_count=4,
        )
        self.assertEqual(_score_resident(graph, baseline), (4, 0))
        self.assertEqual(_score_resident(graph, early_release), (3, 1))
        self.assertLess(
            _score_resident(graph, early_release), _score_resident(graph, baseline)
        )

    def test_qwen_chain_branch_ordering_and_resident_join(self) -> None:
        graph = _readiness_graph(
            (1, 1, 1, 1, 1, 1),
            (
                ((((0, 0),)), ((1, 0), (2, 0))),
                (((1, 0), (2, 0)), ((4, 0),)),
                ((((4, 0),)), ((5, 0),)),
            ),
        )
        good = _schedule(
            graph,
            {
                (0, 0): (0, 0),
                (1, 0): (0, 1),
                (3, 0): (0, 2),
                (2, 0): (1, 0),
                (4, 0): (2, 0),
                (5, 0): (2, 1),
            },
            worker_count=3,
        )
        delayed_attention = _schedule(
            graph,
            {
                (0, 0): (0, 0),
                (3, 0): (0, 1),
                (1, 0): (0, 2),
                (2, 0): (1, 0),
                (4, 0): (2, 0),
                (5, 0): (2, 1),
            },
            worker_count=3,
        )
        continuation_schedule = _schedule(
            graph,
            {
                (0, 0): (0, 0),
                (1, 0): (0, 1),
                (3, 0): (0, 2),
                (2, 0): (1, 0),
                (5, 0): (2, 1),
            },
            worker_count=3,
        )
        good_score = _score_resident(graph, good)
        self.assertEqual(good_score, (4, 2))
        self.assertEqual(_score_resident(graph, delayed_attention), (5, 1))
        continuation_score = _score_continuation(
            graph,
            continuation_schedule,
            FinalArrivalContinuation(1, 0),
        )
        # The full objective ties, so the specified resident tie-break keeps
        # the join (the compact analogue of Qwen root 13) resident.
        self.assertEqual(continuation_score, good_score)

    def test_gemma_routed_expert_reduction_stays_resident(self) -> None:
        graph = _readiness_graph(
            (1, 1, 1, 1, 1, 1),
            (
                (((0, 0), (1, 0)), ((3, 0),)),
                ((((3, 0),)), ((4, 0),)),
            ),
        )
        resident = _schedule(
            graph,
            {
                (0, 0): (0, 0),
                (2, 0): (0, 1),
                (5, 0): (0, 2),
                (1, 0): (1, 0),
                (3, 0): (2, 0),
                (4, 0): (2, 1),
            },
            worker_count=3,
        )
        continuation_schedule = _schedule(
            graph,
            {
                (0, 0): (0, 0),
                (2, 0): (0, 1),
                (5, 0): (0, 2),
                (1, 0): (1, 0),
                (4, 0): (2, 1),
            },
            worker_count=3,
        )
        resident_score = _score_resident(graph, resident)
        alternatives = _continuation_scores(
            graph,
            continuation_schedule,
            FinalArrivalContinuation(0, 0),
        )
        continuation_score = max(score for _publisher, score in alternatives)
        self.assertEqual(resident_score, (3, 1))
        self.assertEqual(
            alternatives,
            (((0, 0), (4, 0)), ((1, 0), (3, 1))),
        )
        self.assertEqual(continuation_score[0], 4)
        self.assertLess(resident_score, continuation_score)

    def test_completion_first_fan_in_frontier_improves_makespan(self) -> None:
        graph = _readiness_graph(
            (1, 1, 1, 1, 1, 1, 1),
            (
                (((0, 0), (1, 0)), ((2, 0),)),
                ((((2, 0),)), ((3, 0),)),
                ((((3, 0),)), ((4, 0),)),
            ),
        )
        spread = _schedule(
            graph,
            {
                (0, 0): (0, 0),
                (5, 0): (1, 0),
                (6, 0): (0, 1),
                (1, 0): (1, 1),
                (2, 0): (0, 2),
                (3, 0): (0, 3),
                (4, 0): (0, 4),
            },
            worker_count=2,
        )
        completion_first = _schedule(
            graph,
            {
                (0, 0): (0, 0),
                (1, 0): (1, 0),
                (2, 0): (0, 1),
                (3, 0): (0, 2),
                (4, 0): (0, 3),
                (5, 0): (1, 1),
                (6, 0): (1, 2),
            },
            worker_count=2,
        )
        self.assertEqual(_score_resident(graph, spread), (5, 1))
        self.assertEqual(_score_resident(graph, completion_first), (4, 1))

    def test_muse_topology_shaped_multiplicity_prefers_group_completion(
        self,
    ) -> None:
        # Two groups each have two producers, two activation slices, and two
        # down-projection CTAs.  This preserves the multiplicities that make
        # finishing one fan-in frontier useful while the next group proceeds.
        graph = _readiness_graph(
            (4, 4, 4),
            (
                (((0, 0), (0, 1)), ((1, 0), (1, 1))),
                (((1, 0), (1, 1)), ((2, 0), (2, 1))),
                (((0, 2), (0, 3)), ((1, 2), (1, 3))),
                (((1, 2), (1, 3)), ((2, 2), (2, 3))),
            ),
        )
        spread = _schedule(
            graph,
            {
                (0, 0): (0, 0),
                (0, 2): (1, 0),
                (0, 1): (2, 0),
                (0, 3): (0, 1),
                (1, 0): (1, 1),
                (1, 2): (2, 1),
                (1, 1): (0, 2),
                (1, 3): (1, 2),
                (2, 0): (2, 2),
                (2, 2): (0, 3),
                (2, 1): (1, 3),
                (2, 3): (2, 3),
            },
            worker_count=3,
        )
        completion_first = _schedule(
            graph,
            {
                (0, 3): (0, 0),
                (0, 0): (1, 0),
                (0, 1): (2, 0),
                (1, 0): (0, 1),
                (1, 1): (1, 1),
                (0, 2): (2, 1),
                (1, 3): (0, 2),
                (2, 0): (1, 2),
                (1, 2): (2, 2),
                (2, 1): (0, 3),
                (2, 3): (1, 3),
                (2, 2): (2, 3),
            },
            worker_count=3,
        )
        self.assertEqual(_score_resident(graph, spread), (5, 1))
        self.assertEqual(_score_resident(graph, completion_first), (4, 2))

    def test_muse_unit_objective_exposes_grouping_gap(self) -> None:
        # Two fan-in groups, each followed by activation and down-projection.
        graph = _readiness_graph(
            (1, 1, 1, 1, 1, 1, 1, 1),
            (
                (((0, 0), (1, 0)), ((4, 0),)),
                ((((4, 0),)), ((6, 0),)),
                (((2, 0), (3, 0)), ((5, 0),)),
                ((((5, 0),)), ((7, 0),)),
            ),
        )
        spread = _schedule(
            graph,
            {
                (0, 0): (0, 0),
                (2, 0): (1, 0),
                (1, 0): (0, 1),
                (3, 0): (1, 1),
                (4, 0): (0, 2),
                (5, 0): (1, 2),
                (6, 0): (0, 3),
                (7, 0): (1, 3),
            },
            worker_count=2,
        )
        grouped = _schedule(
            graph,
            {
                (0, 0): (0, 0),
                (1, 0): (1, 0),
                (4, 0): (0, 1),
                (6, 0): (0, 2),
                (2, 0): (1, 1),
                (3, 0): (1, 2),
                (5, 0): (1, 3),
                (7, 0): (1, 4),
            },
            worker_count=2,
        )
        # This compact equal-cost rendering exposes a limit of the objective:
        # serializing the second group outweighs the earlier first activation.
        # Keep it as a negative control instead of claiming the measured Muse
        # preference from an intentionally incomplete body-cost model.
        self.assertEqual(_score_resident(graph, spread), (4, 0))
        self.assertEqual(_score_resident(graph, grouped), (5, 0))
