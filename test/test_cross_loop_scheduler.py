from __future__ import annotations

import dataclasses
import itertools
from typing import Any
from typing import Literal
from typing import cast
from unittest import mock

import sympy

from helion import exc
from helion._compiler import cross_loop_scheduler
from helion._compiler.cross_loop_scheduler import ReadinessConsumer
from helion._compiler.cross_loop_scheduler import ReadinessCounterPlan
from helion._compiler.cross_loop_scheduler import ReadinessEvent
from helion._compiler.cross_loop_scheduler import ReadinessGraph
from helion._compiler.cross_loop_scheduler import ReadinessProducer
from helion._compiler.cross_loop_scheduler import StaticPipelinePlan
from helion._compiler.cross_loop_scheduler import _segmented_nested_loop_counter
from helion._compiler.cross_loop_scheduler import _select_root_barrier_edges
from helion._compiler.cross_loop_scheduler import build_static_pipeline_plan
from helion._compiler.cross_loop_scheduler import choose_final_arrival_continuations
from helion._compiler.cross_loop_scheduler import choose_readiness_counters
from helion._compiler.cross_loop_scheduler import derive_final_arrival_continuations
from helion._compiler.tile_dependency import CoordinateDomain
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import DenseTaskOrder
from helion._compiler.tile_dependency import DependencyObligation
from helion._compiler.tile_dependency import Incidence
from helion._compiler.tile_dependency import KeyPartition
from helion._compiler.tile_dependency import TileAccess
from helion._compiler.tile_dependency import TileDependencyGraph
from helion._compiler.tile_dependency import _CoordinateRelationPiece
from helion._compiler.tile_dependency import build_tile_dependency_graph
from helion._compiler.tile_dependency import coordinate_axis_symbol
from helion._compiler.tile_dependency import instantiate_coordinate_domains
from helion._testing import TestCase


def _domain(
    *axis_specs: tuple[Any, ...],
    kind: Literal[
        "site", "allocation", "event", "task_order", "worker", "value"
    ] = "site",
    identity: int | None = None,
    allow_empty: bool = False,
) -> CoordinateDomain:
    return CoordinateDomain(
        tuple(spec[0] for spec in axis_specs),
        tuple((spec[0], spec[1]) for spec in axis_specs),
        tuple((spec[0], spec[2]) for spec in axis_specs if len(spec) == 3),
        kind=kind,
        identity=identity,
        _allow_empty=allow_empty,
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


def _point_pairs(
    source: CoordinateDomain,
    target: CoordinateDomain,
    pairs: tuple[tuple[int, int], ...],
) -> CoordinateRelation:
    """Build a finite 1-D relation without materializing its result."""
    (source_axis,) = source.axis_order
    return CoordinateRelation.point_map(
        source,
        target,
        tuple(
            (((source_axis, source_point, source_point + 1, 1),), (target_point,))
            for source_point, target_point in pairs
        ),
    )


def _dense(
    domain: CoordinateDomain,
    axis_order: tuple[int, ...] | None = None,
) -> DenseTaskOrder:
    order = DenseTaskOrder.from_pid(domain, axis_order or domain.axis_order)
    assert order is not None
    return order


def _static_producers(
    producers: tuple[ReadinessProducer, ...],
) -> tuple[tuple[int, Incidence], ...]:
    return tuple((producer.producer_root, producer.incidence) for producer in producers)


def _coordinates(domain: CoordinateDomain, index: int) -> dict[int, int]:
    result: dict[int, int] = {}
    for axis in domain.axis_order:
        index, result[axis] = divmod(index, domain.axis_counts[axis])
    return result


def _point_relation(
    source: CoordinateDomain,
    target: CoordinateDomain,
    pairs: tuple[tuple[int, int], ...],
) -> CoordinateRelation:
    return CoordinateRelation.point_map(
        source,
        target,
        tuple(
            (
                tuple(
                    (axis, value, value + 1, 1)
                    for axis, value in _coordinates(source, source_index).items()
                ),
                tuple(_coordinates(target, target_index).values()),
            )
            for source_index, target_index in pairs
        ),
    )


def _relation_pairs(relation: CoordinateRelation) -> tuple[tuple[int, int], ...]:
    pairs: set[tuple[int, int]] = set()
    target_counts = relation.target_domain.axis_counts
    for source_index in range(relation.source_domain.size):
        source = _coordinates(relation.source_domain, source_index)
        substitutions = {
            coordinate_axis_symbol(axis): sympy.Integer(value)
            for axis, value in source.items()
        }
        for piece in relation.pieces:
            if not all(
                (begin_value := int(sympy.sympify(begin).xreplace(substitutions)))
                <= source[axis]
                < int(sympy.sympify(end).xreplace(substitutions))
                and (source[axis] - begin_value) % int(step) == 0
                for axis, begin, end, step in piece.source_bounds_items
            ):
                continue
            ranges = tuple(
                range(
                    max(0, int(sympy.sympify(begin).xreplace(substitutions))),
                    min(
                        target_counts[axis],
                        int(sympy.sympify(end).xreplace(substitutions)),
                    ),
                    int(step),
                )
                for axis, begin, end, step in piece.target_ranges
            )
            for point in itertools.product(*ranges):
                target_index = 0
                stride = 1
                by_axis = dict(
                    zip(
                        (axis for axis, _begin, _end, _step in piece.target_ranges),
                        point,
                        strict=True,
                    )
                )
                for axis in relation.target_domain.axis_order:
                    target_index += by_axis[axis] * stride
                    stride *= target_counts[axis]
                pairs.add((source_index, target_index))
    return tuple(sorted(pairs))


def _reverse_relation(relation: CoordinateRelation) -> CoordinateRelation:
    return _point_relation(
        relation.target_domain,
        relation.source_domain,
        tuple((target, source) for source, target in _relation_pairs(relation)),
    )


def _incidence(items_by_key: CoordinateRelation, *, grouped: bool = False) -> Incidence:
    pairs = _relation_pairs(items_by_key)
    counts = tuple(
        sum(source == key for source, _target in pairs)
        for key in range(items_by_key.source_domain.size)
    )
    count_axis = (
        max(
            (
                *items_by_key.source_domain.axis_order,
                *items_by_key.target_domain.axis_order,
            ),
            default=-1,
        )
        + 1
    )
    count_domain = CoordinateDomain.scalar(
        max(counts, default=0) + 1,
        axis=count_axis,
        kind="value",
    )
    incidence = Incidence._from_constructed(
        items_by_key,
        keys_by_item=_point_relation(
            items_by_key.target_domain,
            items_by_key.source_domain,
            tuple((target, source) for source, target in pairs),
        ),
        count_by_key=_point_relation(
            items_by_key.source_domain,
            count_domain,
            tuple(enumerate(counts)),
        ),
    )
    return incidence.with_key_major_order() if grouped else incidence


def _producer(
    root: int,
    items_by_key: CoordinateRelation,
    *,
    site_id: int | None = None,
) -> ReadinessProducer:
    return ReadinessProducer(
        producer_root=root,
        incidence=_incidence(items_by_key, grouped=True),
        producer_site_id=site_id,
    )


def _consumer(
    root: int,
    keys_by_consumer: CoordinateRelation,
    *,
    consumers_by_key: CoordinateRelation | None = None,
    consumer_id: int = 0,
    site_id: int | None = None,
    obligations: frozenset[DependencyObligation] = frozenset(),
) -> ReadinessConsumer:
    if consumers_by_key is None:
        consumers_by_key = _reverse_relation(keys_by_consumer)
        incidence = _incidence(consumers_by_key)
    else:
        incidence = Incidence._from_constructed(
            consumers_by_key,
            keys_by_item=keys_by_consumer,
        )
    return ReadinessConsumer(
        consumer_root=root,
        incidence=incidence,
        consumer_id=consumer_id,
        covered_obligations=obligations,
        consumer_site_id=site_id,
    )


def _pointwise_event(
    root_domains: tuple[CoordinateDomain, ...],
    producer_root: int,
    consumer_root: int,
    event_id: int,
    *,
    obligations: frozenset[DependencyObligation] = frozenset(),
) -> ReadinessEvent:
    producer_domain = root_domains[producer_root]
    consumer_domain = root_domains[consumer_root]
    if producer_domain.size != consumer_domain.size:
        raise ValueError("pointwise fixture requires equal task counts")
    event_domain = CoordinateDomain.scalar(
        producer_domain.size,
        kind="event",
        identity=event_id,
    )
    return ReadinessEvent(
        producers=(
            _producer(
                producer_root,
                _full_point_map(
                    event_domain,
                    producer_domain,
                    coordinate_axis_symbol(event_domain.axis_order[0]),
                ),
            ),
        ),
        consumers=(
            _consumer(
                consumer_root,
                _full_point_map(
                    consumer_domain,
                    event_domain,
                    coordinate_axis_symbol(consumer_domain.axis_order[0]),
                ),
                obligations=obligations,
            ),
        ),
    )


def _plan(
    root_domains: tuple[CoordinateDomain, ...],
    worker_count: int,
    *,
    counters: tuple[ReadinessCounterPlan, ...] = (),
    barriers: frozenset[tuple[int, int]] = frozenset(),
    execution_orders: tuple[DenseTaskOrder, ...] | None = None,
    body_orders: tuple[DenseTaskOrder, ...] | None = None,
    dispatch_mode: cross_loop_scheduler.CrossLoopDispatchMode = "static",
) -> StaticPipelinePlan:
    orders = execution_orders or tuple(_dense(domain) for domain in root_domains)
    return StaticPipelinePlan(
        worker_count=worker_count,
        execution_orders=orders,
        body_orders=body_orders if body_orders is not None else orders,
        readiness_counters=counters,
        root_barrier_edges=barriers,
        dispatch_mode=dispatch_mode,
    )


def _access(
    *,
    root: int,
    kind: Literal["load", "store"],
    block_id: int,
    allocation_id: int = 0,
    exact: bool = True,
) -> TileAccess:
    return TileAccess(
        access_id=-1,
        memory_op_index=-1,
        graph_id=root,
        root=root,
        allocation_id=allocation_id,
        kind=kind,
        tensor_name=f"tmp_{allocation_id}",
        tensor_shape=(128,),
        tensor_strides=(1,),
        storage_offset=0,
        subscript_dims=(0,),
        subscript_affine_block_ids=(block_id,),
        subscript_index_scales=(1,),
        subscript_offsets=(0,),
        subscript_is_scalar=(False,),
        has_explicit_mask=False,
        layout_is_symbolically_exact=exact,
        subscript_is_full_slice=(False,),
        subscript_static_extents=(),
    )


def _dependency_graph(
    root_axes: list[list[int]],
    *accesses: TileAccess,
) -> TileDependencyGraph:
    return build_tile_dependency_graph(
        tuple(
            dataclasses.replace(access, access_id=index, memory_op_index=index)
            for index, access in enumerate(accesses)
        ),
        root_axes,
    )


def _configured_readiness_graph(
    dependency_graph: TileDependencyGraph,
    axis_geometry: dict[int, tuple[int, int]],
) -> ReadinessGraph:
    roots, sites = instantiate_coordinate_domains(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    assert all(root is not None for root in roots)
    root_domains = tuple(root for root in roots if root is not None)
    return ReadinessGraph(
        root_domains,
        cross_loop_scheduler._build_readiness_events(
            dependency_graph,
            root_domains=root_domains,
            site_domains=sites,
            publishable_site_ids=None,
            prove_nonnegative=None,
            charge=cross_loop_scheduler._new_relation_work_budget(),
        ),
    )


class TestCrossLoopScheduler(TestCase):
    def test_dispatch_mode_changes_only_physical_root_ownership(self) -> None:
        roots = tuple(
            _domain((axis, count, 1), identity=root)
            for root, (axis, count) in enumerate(((10, 5), (20, 3)))
        )
        static = _plan(roots, 4, barriers=frozenset(((0, 1),)))
        dynamic = dataclasses.replace(static, dispatch_mode="dynamic")

        self.assertEqual(static.execution_orders, dynamic.execution_orders)
        self.assertEqual(static.readiness_counters, dynamic.readiness_counters)
        self.assertEqual(static.root_barrier_arrival_count(0), 4)
        self.assertEqual(dynamic.root_barrier_arrival_count(0), 5)
        with self.assertRaisesRegex(ValueError, "invalid cross-loop dispatch mode"):
            dataclasses.replace(static, dispatch_mode=cast("Any", "invalid"))

    def test_dynamic_dispatch_rejects_same_root_inter_task_wait(self) -> None:
        root = _domain((10, 2, 1), identity=0)
        keys = CoordinateDomain.scalar(1, kind="event", identity=0)
        event = ReadinessEvent(
            (_producer(0, _point_pairs(keys, root, ((0, 0),))),),
            (_consumer(0, _point_pairs(root, keys, ((1, 0),))),),
        )
        counter = ReadinessCounterPlan(event.producers, event.consumers)
        graph = ReadinessGraph((root,), (event,))
        static = _plan((root,), 1, counters=(counter,))
        dynamic = dataclasses.replace(static, dispatch_mode="dynamic")

        def is_safe(plan: StaticPipelinePlan) -> bool:
            return cross_loop_scheduler._schedule_is_progress_safe(
                plan,
                graph,
                (),
                {},
                _static_producers,
                cross_loop_scheduler._new_relation_work_budget(),
            )

        self.assertTrue(is_safe(static))
        self.assertFalse(is_safe(dynamic))

    def test_static_geometry_uses_all_root_padded_prefixes(self) -> None:
        roots = tuple(
            _domain((axis, count, 1), identity=root)
            for root, (axis, count) in enumerate(((10, 5), (20, 3), (30, 2)))
        )
        plan = _plan(roots, 4)

        self.assertEqual(tuple(plan.static_base(root) for root in range(3)), (0, 8, 12))
        self.assertEqual(plan.static_slot_count, 16)
        self.assertEqual(plan.resident_roots, (0, 1, 2))
        self.assertEqual(plan.wave_domain.size, 4)

    def test_continuation_keeps_later_prefix_and_uses_key_count_arrivals(
        self,
    ) -> None:
        roots = tuple(
            _domain((axis, count, 1), identity=root)
            for root, (axis, count) in enumerate(((10, 8), (20, 2), (30, 2)))
        )
        keys = CoordinateDomain.scalar(2, kind="event", identity=0)
        event = ReadinessEvent(
            (
                _producer(
                    0,
                    _point_pairs(
                        keys,
                        roots[0],
                        tuple((task // 4, task) for task in range(8)),
                    ),
                ),
            ),
            (
                _consumer(
                    1,
                    _point_pairs(roots[1], keys, ((0, 0), (1, 1))),
                ),
            ),
        )
        counter = ReadinessCounterPlan(event.producers, event.consumers, 0)
        plan = _plan(roots, 4, counters=(counter,))

        self.assertEqual(plan.continuation_roots, frozenset((1,)))
        self.assertEqual(plan.resident_roots, (0, 2))
        self.assertEqual(tuple(plan.static_base(root) for root in range(3)), (0, 8, 12))
        self.assertEqual(plan.static_slot_count, 16)
        self.assertEqual(plan.root_barrier_arrival_count(0), 4)
        self.assertEqual(plan.root_barrier_arrival_count(1), 2)

    def test_execution_reorder_preserves_configured_body_pid_abi(self) -> None:
        domain = _domain((10, 2, 1), (11, 3, 1), identity=0)
        configured = _dense(domain, (10, 11))
        execution = _dense(domain, (11, 10))
        plan = _plan(
            (domain,),
            4,
            execution_orders=(execution,),
            body_orders=(configured,),
        )

        self.assertEqual(
            plan.body_orders[0].ordinal_by_task, configured.ordinal_by_task
        )
        self.assertNotEqual(
            plan.body_orders[0].ordinal_by_task, execution.ordinal_by_task
        )

    def test_symbolic_counter_map_with_concrete_capacity_is_allowed(self) -> None:
        roots = (
            _domain((10, 8, 1), identity=0),
            _domain((20, 8, 1), identity=1),
        )
        keys = CoordinateDomain.scalar(8, kind="event", identity=0)
        offset = sympy.Symbol("schedule_offset", integer=True)
        symbolic_keys = _full_point_map(
            roots[1],
            keys,
            sympy.Mod(coordinate_axis_symbol(20) + offset, 8),
        )
        symbolic_consumers = _full_point_map(
            keys,
            roots[1],
            sympy.Mod(coordinate_axis_symbol(0) - offset, 8),
        )
        event = ReadinessEvent(
            (_producer(0, _full_point_map(keys, roots[0], coordinate_axis_symbol(0))),),
            (_consumer(1, symbolic_keys, consumers_by_key=symbolic_consumers),),
        )
        counter = ReadinessCounterPlan(event.producers, event.consumers)
        plan = _plan(
            roots,
            4,
            counters=(counter,),
        )
        self.assertEqual(
            plan.readiness_counters[0].consumers[0].keys_by_consumer, symbolic_keys
        )

    def test_parameterized_task_capacity_is_not_a_dense_order(self) -> None:
        task_count = sympy.Symbol("task_count", integer=True, positive=True)
        domain = _domain((10, task_count, 1), identity=0)
        self.assertIsNone(DenseTaskOrder.from_pid(domain, domain.axis_order))

    def test_large_fixed_capacity_schedule_remains_structural(self) -> None:
        task_count = 50_000_000
        graph = _dependency_graph([[10]])
        domain = _domain((10, task_count, 1), identity=0)
        order = _dense(domain)
        for mode in ("static", "dynamic"):
            with self.subTest(mode=mode):
                plan = build_static_pipeline_plan(
                    dependency_graph=graph,
                    root_task_orders=(order,),
                    site_domains=(),
                    worker_count=148,
                    cross_loop_dispatch_mode=mode,
                )
                self.assertEqual(plan.execution_orders[0].task_count, task_count)
                self.assertLessEqual(
                    len(plan.execution_orders[0].tasks_by_ordinal.pieces), 2
                )
                expected_arrivals = 148 if mode == "static" else task_count
                self.assertEqual(plan.root_barrier_arrival_count(0), expected_arrivals)

    def test_configured_orders_require_unique_root_identities(self) -> None:
        graph = _dependency_graph([[10], [20]])
        roots = (
            _domain((10, 2, 1), identity=0),
            _domain((20, 2, 1), identity=0),
        )
        with self.assertRaisesRegex(exc.InvalidConfig, "fixed task capacity"):
            build_static_pipeline_plan(
                dependency_graph=graph,
                root_task_orders=tuple(_dense(root) for root in roots),
                site_domains=(),
                worker_count=2,
            )

    def test_event_and_consumer_ids_are_dense_and_stable(self) -> None:
        roots = (
            _domain((10, 2, 1), identity=0),
            _domain((20, 2, 1), identity=1),
        )
        event = _pointwise_event(roots, 0, 1, 0)
        self.assertEqual(event.event_id, 0)
        self.assertEqual(event.consumers[0].consumer_id, 0)

        with self.assertRaisesRegex(ValueError, "consumer IDs"):
            dataclasses.replace(
                event,
                consumers=(dataclasses.replace(event.consumers[0], consumer_id=1),),
            )
        with self.assertRaisesRegex(ValueError, "event IDs"):
            ReadinessGraph(roots, (_pointwise_event(roots, 0, 1, 1),))

    def test_filtered_counter_keeps_stable_event_consumer_id(self) -> None:
        roots = tuple(
            _domain((axis, 2, 1), identity=root)
            for root, axis in enumerate((10, 20, 30))
        )
        event = ReadinessEvent(
            _pointwise_event(roots, 0, 1, 0).producers,
            tuple(
                dataclasses.replace(
                    _pointwise_event(roots, 0, consumer_root, 0).consumers[0],
                    consumer_id=consumer_id,
                    covered_obligations=frozenset(((consumer_id, None, None),)),
                )
                for consumer_id, consumer_root in enumerate((1, 2))
            ),
        )
        graph = ReadinessGraph(roots, (event,))
        continuation = cross_loop_scheduler.FinalArrivalContinuation(0, 1)

        (counter,) = choose_readiness_counters(
            graph,
            (continuation,),
            excluded_obligations=frozenset(((0, None, None),)),
            charge=cross_loop_scheduler._new_relation_work_budget(),
        )

        self.assertEqual(tuple(item.consumer_id for item in counter.consumers), (1,))
        self.assertEqual(counter.continuation_consumer_index, 0)
        self.assertEqual(
            cross_loop_scheduler._emitted_final_arrival_continuations(
                graph, (counter,)
            ),
            (continuation,),
        )

    def test_pointwise_final_arrival_continuation_is_selected(self) -> None:
        roots = (
            _domain((10, 8, 1), identity=0),
            _domain((20, 8, 1), identity=1),
        )
        event = _pointwise_event(roots, 0, 1, 0)
        graph = ReadinessGraph(roots, (event,))
        candidates = derive_final_arrival_continuations(
            graph, (ReadinessCounterPlan(event.producers, event.consumers),)
        )
        plan = _plan(
            roots,
            8,
            counters=(ReadinessCounterPlan(event.producers, event.consumers),),
        )

        self.assertEqual(len(candidates), 1)
        charge = cross_loop_scheduler._new_relation_work_budget()
        self.assertEqual(
            choose_final_arrival_continuations(
                graph,
                candidates,
                plan,
                causal_relations=dict(
                    cross_loop_scheduler._root_causal_prerequisite_relations(
                        graph, charge
                    )
                ),
                charge=charge,
            ),
            candidates,
        )

    def test_final_arrival_continuation_rejects_cross_key_worker_strands(self) -> None:
        roots = (
            _domain((10, 8, 1), identity=0),
            _domain((20, 8, 1), identity=1),
        )
        event = _pointwise_event(roots, 0, 1, 0)
        graph = ReadinessGraph(roots, (event,))
        candidates = derive_final_arrival_continuations(
            graph, (ReadinessCounterPlan(event.producers, event.consumers),)
        )
        plan = _plan(
            roots,
            4,
            counters=(ReadinessCounterPlan(event.producers, event.consumers),),
        )

        self.assertEqual(len(candidates), 1)
        charge = cross_loop_scheduler._new_relation_work_budget()
        self.assertEqual(
            choose_final_arrival_continuations(
                graph,
                candidates,
                plan,
                causal_relations=dict(
                    cross_loop_scheduler._root_causal_prerequisite_relations(
                        graph, charge
                    )
                ),
                charge=charge,
            ),
            (),
        )

    def test_root_local_preparation_keeps_order_without_composed_grouping(
        self,
    ) -> None:
        producer = _domain((10, 2, 1), (11, 2, 1), identity=0)
        consumer = _domain((20, 2, 1), identity=1)
        roots = (producer, consumer)
        keys = CoordinateDomain.scalar(2, kind="event", identity=0)
        producer_keys = _domain((10, 2), kind="event", identity=0)
        producer_partition = KeyPartition.projection(producer, producer_keys)
        assert producer_partition is not None
        producer_incidence = producer_partition.as_incidence(
            grouped_items=_dense(producer, (11, 10))
        ).rename_domains(keys, producer)
        assert producer_incidence is not None
        event = ReadinessEvent(
            (ReadinessProducer(0, producer_incidence),),
            (
                _consumer(
                    1,
                    _full_point_map(
                        consumer,
                        keys,
                        coordinate_axis_symbol(20),
                    ),
                ),
            ),
        )
        graph = ReadinessGraph(roots, (event,))
        continuation = ReadinessCounterPlan(
            event.producers,
            event.consumers,
            continuation_consumer_index=0,
        )
        baseline = _plan(roots, 2, counters=(continuation,))
        prepared = cross_loop_scheduler._consumer_major_producer_order(
            graph,
            baseline,
            (continuation,),
            static_producers=_static_producers,
            excluded_roots=frozenset((1,)),
            charge=cross_loop_scheduler._new_relation_work_budget(),
        )

        self.assertEqual(baseline.execution_orders[0], _dense(producer, (10, 11)))
        actual = prepared.execution_orders[0]
        self.assertEqual(
            _relation_pairs(actual.ordinal_by_task),
            _relation_pairs(baseline.execution_orders[0].ordinal_by_task),
        )
        self.assertEqual(prepared.body_orders, baseline.body_orders)

        configured_consumer = _dense(consumer)
        ordinal_domain = configured_consumer.tasks_by_ordinal.source_domain
        ordinal_axis = ordinal_domain.axis_order[0]
        reflected_consumer = DenseTaskOrder._from_constructed(
            _full_point_map(
                ordinal_domain,
                consumer,
                1 - coordinate_axis_symbol(ordinal_axis),
            ),
            _full_point_map(
                consumer,
                ordinal_domain,
                1 - coordinate_axis_symbol(20),
            ),
        )
        reflected_baseline = _plan(
            roots,
            2,
            counters=(continuation,),
            execution_orders=(baseline.execution_orders[0], reflected_consumer),
            body_orders=baseline.body_orders,
        )
        reflected = cross_loop_scheduler._consumer_major_producer_order(
            graph,
            reflected_baseline,
            (continuation,),
            static_producers=_static_producers,
            excluded_roots=frozenset((1,)),
            charge=cross_loop_scheduler._new_relation_work_budget(),
        )
        self.assertEqual(
            _relation_pairs(reflected.execution_orders[0].ordinal_by_task),
            _relation_pairs(baseline.execution_orders[0].ordinal_by_task),
        )
        self.assertEqual(reflected.body_orders, baseline.body_orders)

        original_replace = dataclasses.replace

        def decline_alternate(value: object, **changes: object) -> object:
            if isinstance(value, StaticPipelinePlan) and "execution_orders" in changes:
                raise ValueError("unsupported alternate traversal")
            return original_replace(value, **changes)

        with mock.patch.object(
            cross_loop_scheduler.dataclasses,
            "replace",
            side_effect=decline_alternate,
        ):
            declined = cross_loop_scheduler._consumer_major_producer_order(
                graph,
                baseline,
                (continuation,),
                static_producers=_static_producers,
                excluded_roots=frozenset((1,)),
                charge=cross_loop_scheduler._new_relation_work_budget(),
            )
        self.assertIs(declined, baseline)

    def test_partial_consumer_declines_final_arrival_continuation(self) -> None:
        roots = (
            _domain((10, 2, 1), identity=0),
            _domain((20, 2, 1), identity=1),
        )
        event_domain = CoordinateDomain.scalar(2, kind="event", identity=0)
        event = ReadinessEvent(
            (_producer(0, _point_pairs(event_domain, roots[0], ((0, 0), (1, 1)))),),
            (_consumer(1, _point_pairs(roots[1], event_domain, ((0, 0),))),),
        )

        self.assertEqual(
            derive_final_arrival_continuations(
                ReadinessGraph(roots, (event,)),
                (ReadinessCounterPlan(event.producers, event.consumers),),
            ),
            (),
        )

    def test_nonuniform_arrival_count_is_retained_exactly(self) -> None:
        roots = (
            _domain((10, 5, 1), identity=0),
            _domain((20, 2, 1), identity=1),
        )
        keys = CoordinateDomain.scalar(2, kind="event", identity=0)
        event = ReadinessEvent(
            (
                _producer(
                    0,
                    _point_pairs(
                        keys,
                        roots[0],
                        ((0, 0), (0, 1), (1, 2), (1, 3), (1, 4)),
                    ),
                ),
            ),
            (_consumer(1, _point_pairs(roots[1], keys, ((0, 0), (1, 1)))),),
        )
        counter = ReadinessCounterPlan(event.producers, event.consumers)

        self.assertIsNone(counter.uniform_arrival_count())
        count_by_key = counter.producers[0].incidence.count_by_key
        self.assertIsNotNone(count_by_key)
        assert count_by_key is not None
        self.assertEqual(count_by_key.value_bounds(), (2, 3))
        _plan(roots, 4, counters=(counter,))

    def test_counter_lowering_rejects_uncounted_publications(self) -> None:
        # Two heads of three row tiles; each head's consumers read two tiles.
        producers = _domain((0, 2), (1, 3), identity=0)
        heads = _domain((2, 2), kind="event", identity=0)
        head, key = coordinate_axis_symbol(0), coordinate_axis_symbol(2)
        fibers = CoordinateRelation(
            heads,
            producers,
            (
                _CoordinateRelationPiece(
                    ((2, 0, 2, 1),), ((0, key, key + 1, 1), (1, 0, 2, 1))
                ),
            ),
        )
        count = CoordinateRelation.point_map(
            heads,
            CoordinateDomain.scalar(3, axis=3, kind="value"),
            ((((2, 0, 2, 1),), (sympy.Integer(2),)),),
        )

        def producer(published_rows: int) -> ReadinessProducer:
            publication = CoordinateRelation.point_map(
                producers,
                heads,
                ((((0, 0, 2, 1), (1, 0, published_rows, 1)), (head,)),),
            )
            return ReadinessProducer(
                0,
                Incidence._from_constructed(
                    fibers, keys_by_item=publication, count_by_key=count
                ),
            )

        # Three arrivals per head never equal an exact wait for two.
        self.assertFalse(
            cross_loop_scheduler._supports_readiness_counter_lowering(producer(3))
        )
        self.assertTrue(
            cross_loop_scheduler._supports_readiness_counter_lowering(producer(2))
        )
        # A count that may go negative (an unclipped block tail) is no target.
        negative = CoordinateRelation.point_map(
            heads,
            CoordinateDomain.scalar(3, axis=3, kind="value"),
            ((((2, 0, 2, 1),), (key - 1,)),),
        )
        self.assertFalse(
            cross_loop_scheduler._supports_readiness_counter_lowering(
                ReadinessProducer(
                    0,
                    Incidence._from_constructed(
                        fibers,
                        keys_by_item=producer(2).incidence.keys_by_item,
                        count_by_key=negative,
                    ),
                )
            )
        )

    def test_nested_entry_counter_publishes_only_read_producer_tiles(self) -> None:
        # Producer (head, row tile) for 2 heads x 3 row tiles; each consumer
        # task's nested loop reads row tiles 0 and 1 of its head.  The fine
        # publication item -> (head, tile) is clipped to the two read tiles.
        roots = (
            _domain((0, 2), (1, 3), identity=0),
            _domain((2, 2), (3, 2), identity=1),
        )
        site = _domain((2, 2), (3, 2), (4, 2), identity=5)
        keys = _domain((0, 2), (1, 2), kind="event", identity=0)
        full_keys = ((0, 0, 2, 1), (1, 0, 2, 1))
        axis = coordinate_axis_symbol
        producer = Incidence._from_constructed(
            CoordinateRelation.point_map(
                keys, roots[0], ((full_keys, (axis(0), axis(1))),)
            ),
            keys_by_item=CoordinateRelation.point_map(
                roots[0], keys, ((((0, 0, 2, 1), (1, 0, 3, 1)), (axis(0), axis(1))),)
            ),
            count_by_key=CoordinateRelation.point_map(
                keys,
                CoordinateDomain.scalar(2, axis=5, kind="value"),
                ((full_keys, (sympy.Integer(1),)),),
            ),
        )
        consumer = Incidence._from_constructed(
            CoordinateRelation(
                keys,
                site,
                (
                    _CoordinateRelationPiece(
                        full_keys,
                        (
                            (2, axis(0), axis(0) + 1, 1),
                            (3, 0, 2, 1),
                            (4, axis(1), axis(1) + 1, 1),
                        ),
                    ),
                ),
            ),
            keys_by_item=CoordinateRelation.point_map(
                site,
                keys,
                ((((2, 0, 2, 1), (3, 0, 2, 1), (4, 0, 2, 1)), (axis(2), axis(4))),),
            ),
        )
        event = ReadinessEvent(
            (ReadinessProducer(0, producer),),
            (ReadinessConsumer(1, consumer, 0, frozenset(), 5),),
        )
        entry = _segmented_nested_loop_counter(
            ReadinessGraph(roots, (event,)),
            event,
            event.consumers[0],
            None,
            cross_loop_scheduler._new_relation_work_budget(),
        )

        assert entry is not None
        self.assertEqual(entry.uniform_arrival_count(), 2)
        publication = entry.producers[0].incidence.keys_by_item
        assert publication is not None
        # Producer index head + 2 * tile publishes to its head; tile 2 is silent.
        self.assertEqual(_relation_pairs(publication), ((0, 0), (1, 1), (2, 0), (3, 1)))

    def test_paired_incidence_composition_preserves_widening_chain(self) -> None:
        first = _domain((0, 1), (1, 4), kind="event", identity=0)
        middle = _domain((2, 4), (3, 4), identity=1)
        last = _domain((4, 8), (5, 4), identity=2)
        first_to_middle = CoordinateRelation(
            first,
            middle,
            (
                _CoordinateRelationPiece(
                    ((0, 0, 1, 1), (1, 0, 4, 1)),
                    (
                        (2, 0, 4, 1),
                        (
                            3,
                            coordinate_axis_symbol(1),
                            coordinate_axis_symbol(1) + 1,
                            1,
                        ),
                    ),
                ),
            ),
        )
        middle_to_last = CoordinateRelation(
            middle,
            last,
            (
                _CoordinateRelationPiece(
                    ((2, 0, 4, 1), (3, 0, 4, 1)),
                    (
                        (
                            4,
                            2 * coordinate_axis_symbol(2),
                            2 * coordinate_axis_symbol(2) + 2,
                            1,
                        ),
                        (
                            5,
                            coordinate_axis_symbol(3),
                            coordinate_axis_symbol(3) + 1,
                            1,
                        ),
                    ),
                ),
            ),
        )

        composed = _incidence(first_to_middle).then(_incidence(middle_to_last))

        self.assertIsNotNone(composed)
        assert composed is not None and composed.keys_by_item is not None
        expected = tuple((y, x + 8 * y) for y in range(4) for x in range(8))
        self.assertEqual(_relation_pairs(composed.items_by_key), expected)
        self.assertEqual(
            _relation_pairs(composed.keys_by_item),
            tuple((target, source) for source, target in expected),
        )

    def test_nested_entry_and_segmented_counters_preserve_identity(self) -> None:
        producer_root = _domain((10, 2, 1), (11, 4, 1), identity=0)
        consumer_root = _domain((20, 2, 1), identity=1)
        consumer_site = _domain((20, 2, 1), (21, 4, 1), identity=7)
        keys = _domain((0, 2), (1, 4), kind="event", identity=0)
        obligation = (0, None, 7)
        event = ReadinessEvent(
            (
                _producer(
                    0,
                    _full_point_map(
                        keys,
                        producer_root,
                        coordinate_axis_symbol(0),
                        coordinate_axis_symbol(1),
                    ),
                ),
            ),
            (
                _consumer(
                    1,
                    _full_point_map(
                        consumer_site,
                        keys,
                        coordinate_axis_symbol(20),
                        coordinate_axis_symbol(21),
                    ),
                    site_id=7,
                    obligations=frozenset((obligation,)),
                ),
            ),
        )
        graph = ReadinessGraph((producer_root, consumer_root), (event,))
        charge = cross_loop_scheduler._new_relation_work_budget()
        entry = _segmented_nested_loop_counter(
            graph, event, event.consumers[0], None, charge
        )
        segmented = _segmented_nested_loop_counter(
            graph, event, event.consumers[0], (0, 2, 4), charge
        )

        self.assertIsNotNone(entry)
        self.assertIsNotNone(segmented)
        assert entry is not None and segmented is not None
        self.assertEqual(
            (entry.readiness_key_domain.size, segmented.readiness_key_domain.size),
            (2, 4),
        )
        for counter in (entry, segmented):
            self.assertEqual(counter.readiness_key_domain.identity, event.event_id)
            self.assertEqual(counter.consumers[0].consumer_id, 0)
            self.assertEqual(
                counter.consumers[0].covered_obligations,
                frozenset((obligation,)),
            )

    def test_nested_entry_pulls_back_many_to_one_tail_partition(self) -> None:
        producer_root = _domain((10, 5, 1), (11, 4, 1), identity=0)
        consumer_root = _domain((21, 4, 1), identity=1)
        consumer_site = _domain((21, 4, 1), (22, 2, 1), identity=7)
        keys = _domain((0, 5), (1, 4), kind="event", identity=0)
        slot = coordinate_axis_symbol(21)
        nested = coordinate_axis_symbol(22)
        key_row = coordinate_axis_symbol(0)
        key_slot = coordinate_axis_symbol(1)
        keys_by_consumer = CoordinateRelation(
            consumer_site,
            keys,
            (
                _CoordinateRelationPiece(
                    ((21, 0, 4, 1), (22, 0, 2, 1)),
                    ((0, 3 * nested, 3 * (nested + 1), 1), (1, slot, slot + 1, 1)),
                ),
            ),
        )
        consumers_by_key = CoordinateRelation.point_map(
            keys,
            consumer_site,
            (
                (((0, 0, 3, 1), (1, 0, 4, 1)), (key_slot, sympy.Integer(0))),
                (((0, 3, 5, 1), (1, 0, 4, 1)), (key_slot, sympy.Integer(1))),
            ),
        )
        event = ReadinessEvent(
            (
                _producer(
                    0,
                    _full_point_map(
                        keys,
                        producer_root,
                        key_row,
                        key_slot,
                    ),
                ),
            ),
            (
                _consumer(
                    1,
                    keys_by_consumer,
                    consumers_by_key=consumers_by_key,
                    site_id=7,
                ),
            ),
        )
        graph = ReadinessGraph((producer_root, consumer_root), (event,))

        counter = _segmented_nested_loop_counter(
            graph,
            event,
            event.consumers[0],
            None,
            cross_loop_scheduler._new_relation_work_budget(),
        )

        self.assertIsNotNone(counter)
        assert counter is not None
        self.assertEqual(counter.readiness_key_domain.size, 4)
        self.assertEqual(counter.uniform_arrival_count(), 5)
        plan = _plan(
            (producer_root, consumer_root),
            20,
            counters=(counter,),
        )
        self.assertEqual(
            cross_loop_scheduler.nested_wait_placement(
                consumer_root, counter.consumers[0]
            ),
            (22, (0,)),
        )
        self.assertEqual(
            cross_loop_scheduler._without_wave_dominated_nested_counters(
                plan, "static"
            ),
            (counter,),
        )

    def test_nested_counter_admission_uses_emitted_wait_count(self) -> None:
        producer_root = _domain((10, 2, 1), identity=0)
        consumer_root = _domain((20, 3, 1), identity=1)
        consumer_site = _domain((20, 3, 1), (21, 2, 1), identity=7)

        def counter(*, per_iteration: bool) -> ReadinessCounterPlan:
            keys = CoordinateDomain.scalar(
                2 if per_iteration else 1, kind="event", identity=0
            )
            key = coordinate_axis_symbol(0)
            nested = coordinate_axis_symbol(21)
            producer = _producer(
                0,
                CoordinateRelation(
                    keys,
                    producer_root,
                    (
                        _CoordinateRelationPiece(
                            ((0, 0, keys.size, 1),),
                            ((10, 0, 2, 1),)
                            if not per_iteration
                            else ((10, key, key + 1, 1),),
                        ),
                    ),
                ),
            )
            keys_by_consumer = _full_point_map(
                consumer_site,
                keys,
                nested if per_iteration else sympy.Integer(0),
            )
            consumers_by_key = CoordinateRelation(
                keys,
                consumer_site,
                (
                    _CoordinateRelationPiece(
                        ((0, 0, keys.size, 1),),
                        (
                            (20, 0, 3, 1),
                            (21, key, key + 1, 1) if per_iteration else (21, 0, 2, 1),
                        ),
                    ),
                ),
            )
            consumer = _consumer(
                1,
                keys_by_consumer,
                consumers_by_key=consumers_by_key,
                site_id=7,
            )
            return ReadinessCounterPlan((producer,), (consumer,))

        repeated = counter(per_iteration=True)
        hoisted = counter(per_iteration=False)
        repeated_plan = _plan((producer_root, consumer_root), 4, counters=(repeated,))
        hoisted_plan = _plan((producer_root, consumer_root), 4, counters=(hoisted,))

        self.assertEqual(
            cross_loop_scheduler._without_wave_dominated_nested_counters(
                repeated_plan, "static"
            ),
            (),
        )
        self.assertEqual(
            cross_loop_scheduler._without_wave_dominated_nested_counters(
                hoisted_plan, "static"
            ),
            (hoisted,),
        )

    def test_nested_counter_admission_counts_clipped_uniform_fanout(self) -> None:
        producer_root = _domain((10, 2, 1), identity=0)
        consumer_root = _domain((20, 3, 1), identity=1)
        consumer_site = _domain((20, 3, 1), (21, 2, 1), identity=7)

        def counter(
            consumer_pairs: tuple[tuple[int, int], ...],
            key_count: int,
            site: CoordinateDomain = consumer_site,
        ) -> ReadinessCounterPlan:
            keys = CoordinateDomain.scalar(key_count, kind="event", identity=0)
            producer_pairs = (
                ((0, 0), (0, 1))
                if key_count == 1
                else tuple((key, key) for key in range(key_count))
            )
            keys_by_consumer = _point_relation(site, keys, consumer_pairs)
            consumer = _consumer(1, keys_by_consumer, site_id=7)
            return ReadinessCounterPlan(
                (_producer(0, _point_relation(keys, producer_root, producer_pairs)),),
                (consumer,),
            )

        expensive = counter(((0, 0), (1, 0), (3, 0), (4, 0)), 1)
        useful = counter(((0, 0), (3, 0)), 1)
        nonuniform = counter(((0, 0), (3, 0), (1, 1)), 2)
        count = expensive.consumers[0].incidence.count_by_key
        assert count is not None
        self.assertEqual(count.value_bounds(), (4, 4))
        self.assertEqual(
            cross_loop_scheduler.nested_wait_placement(
                consumer_root, expensive.consumers[0]
            ),
            (21, None),
        )
        self.assertEqual(
            cross_loop_scheduler._without_wave_dominated_nested_counters(
                _plan((producer_root, consumer_root), 4, counters=(expensive,)),
                "static",
            ),
            (),
        )
        for retained in (useful, nonuniform):
            self.assertEqual(
                cross_loop_scheduler._without_wave_dominated_nested_counters(
                    _plan((producer_root, consumer_root), 4, counters=(retained,)),
                    "static",
                ),
                (retained,),
            )

        large_consumer_root = _domain((20, 6, 1), identity=1)
        large_consumer_site = _domain((20, 6, 1), (21, 2, 1), identity=7)
        dynamic_sensitive = counter(
            tuple((consumer, 0) for consumer in range(6)), 1, large_consumer_site
        )
        large_plan = _plan(
            (producer_root, large_consumer_root),
            4,
            counters=(dynamic_sensitive,),
        )
        self.assertEqual(
            cross_loop_scheduler._without_wave_dominated_nested_counters(
                large_plan, "static"
            ),
            (),
        )
        self.assertEqual(
            cross_loop_scheduler._without_wave_dominated_nested_counters(
                large_plan, "dynamic"
            ),
            (dynamic_sensitive,),
        )

    def test_nested_counter_admission_derives_missing_consumer_count(self) -> None:
        producer_root = _domain((10, 2, 1), identity=0)
        consumer_root = _domain(identity=1)
        keys = CoordinateDomain.scalar(1, kind="event", identity=0)
        consumer_site = CoordinateDomain.scalar(6, axis=21, kind="site", identity=7)
        consumers_by_key = CoordinateRelation(
            keys,
            consumer_site,
            (_CoordinateRelationPiece(((0, 0, 1, 1),), ((21, 1, 5, 1),)),),
        )
        keys_by_consumer = CoordinateRelation.point_map(
            consumer_site,
            keys,
            (((((21, 1, 5, 1),), (sympy.Integer(0),))),),
        )
        incidence = Incidence._from_constructed(
            consumers_by_key, keys_by_item=keys_by_consumer
        )
        self.assertIsNone(incidence.count_by_key)
        derived = incidence.with_key_major_order().count_by_key
        assert derived is not None
        self.assertEqual(derived.value_bounds(), (4, 4))
        counter = ReadinessCounterPlan(
            (_producer(0, _point_relation(keys, producer_root, ((0, 0), (0, 1)))),),
            (ReadinessConsumer(1, incidence, 0, consumer_site_id=7),),
        )
        plan = _plan((producer_root, consumer_root), 4, counters=(counter,))
        self.assertEqual(
            cross_loop_scheduler._without_wave_dominated_nested_counters(
                plan, "static"
            ),
            (),
        )

    def test_coarsened_whole_root_event_uses_barrier(self) -> None:
        producer_root = _domain((10, 1, 1), identity=0)
        consumer_root = _domain((20, 6, 1), identity=1)
        keys = CoordinateDomain.scalar(4, kind="event", identity=0)
        producer_items = CoordinateRelation(
            keys,
            producer_root,
            (_CoordinateRelationPiece(((0, 0, 4, 1),), ((10, 0, 1, 1),)),),
        )
        base_incidence = _incidence(producer_items, grouped=True)
        publication = CoordinateRelation(
            producer_root,
            keys,
            (
                _CoordinateRelationPiece(
                    ((10, 0, 1, 1),),
                    (
                        (
                            0,
                            4 * coordinate_axis_symbol(10),
                            4 * (coordinate_axis_symbol(10) + 1),
                            1,
                        ),
                    ),
                ),
            ),
        )
        producer = ReadinessProducer(
            producer_root=0,
            incidence=Incidence._from_constructed(
                producer_items,
                keys_by_item=publication,
                count_by_key=base_incidence.count_by_key,
                grouped_items=base_incidence.grouped_items,
            ),
        )
        self.assertIsNotNone(KeyPartition.from_fixed_width_publication(publication))

        def consumer(end: int) -> ReadinessConsumer:
            return _consumer(
                1,
                CoordinateRelation(
                    consumer_root,
                    keys,
                    (_CoordinateRelationPiece(((20, 0, end, 1),), ((0, 0, 4, 1),)),),
                ),
                consumers_by_key=CoordinateRelation(
                    keys,
                    consumer_root,
                    (_CoordinateRelationPiece(((0, 0, 4, 1),), ((20, 0, end, 1),)),),
                ),
                obligations=frozenset(((0, None, None),)),
            )

        def selected(end: int) -> tuple[ReadinessCounterPlan, ...]:
            graph = ReadinessGraph(
                (producer_root, consumer_root),
                (ReadinessEvent((producer,), (consumer(end),)),),
            )
            return choose_readiness_counters(
                graph,
                (),
                charge=cross_loop_scheduler._new_relation_work_budget(),
            )

        with mock.patch.object(
            ReadinessCounterPlan,
            "is_root_barrier_equivalent",
            return_value=False,
        ):
            self.assertEqual(len(selected(6)), 1)
        self.assertEqual(selected(6), ())

        coarse_keys = CoordinateDomain.scalar(1, kind="event", identity=0)
        coarse_producer = _producer(
            0,
            CoordinateRelation(
                coarse_keys,
                producer_root,
                (_CoordinateRelationPiece(((0, 0, 1, 1),), ((10, 0, 1, 1),)),),
            ),
        )
        full_consumer = ReadinessConsumer(
            consumer_root=1,
            incidence=Incidence.complete(coarse_keys, consumer_root),
            consumer_id=0,
        )
        counter = ReadinessCounterPlan((coarse_producer,), (full_consumer,))
        roots = (producer_root, consumer_root)
        charge = cross_loop_scheduler._new_relation_work_budget()
        self.assertTrue(counter.is_root_barrier_equivalent(roots, charge))
        partial_consumer = _consumer(
            1,
            CoordinateRelation.point_map(
                consumer_root,
                coarse_keys,
                (((((20, 0, 3, 1),), (sympy.Integer(0),))),),
            ),
        )
        self.assertFalse(
            dataclasses.replace(
                counter, consumers=(partial_consumer,)
            ).is_root_barrier_equivalent(roots, charge)
        )
        self.assertFalse(
            dataclasses.replace(
                counter,
                consumers=(dataclasses.replace(full_consumer, consumer_site_id=7),),
            ).is_root_barrier_equivalent(roots, charge)
        )
        self.assertFalse(
            dataclasses.replace(
                counter, continuation_consumer_index=0
            ).is_root_barrier_equivalent(roots, charge)
        )

    def test_two_producer_nested_frontier_uses_shared_wave_domain(self) -> None:
        roots = (
            _domain((10, 2, 1), identity=0),
            _domain((20, 6, 1), identity=1),
            _domain((30, 1, 1), identity=2),
        )
        consumer_site = _domain((30, 1, 1), (31, 2, 1), identity=7)
        keys = CoordinateDomain.scalar(2, kind="event", identity=0)
        event = ReadinessEvent(
            (
                _producer(0, _point_pairs(keys, roots[0], ((0, 0), (1, 1)))),
                _producer(
                    1,
                    _point_pairs(
                        keys,
                        roots[1],
                        ((0, 0), (0, 1), (0, 2), (1, 3), (1, 4), (1, 5)),
                    ),
                ),
            ),
            (
                _consumer(
                    2,
                    _full_point_map(
                        consumer_site,
                        keys,
                        coordinate_axis_symbol(31),
                    ),
                    site_id=7,
                ),
            ),
        )
        graph = ReadinessGraph(roots, (event,))
        plan = _plan(roots, 2)

        frontier = cross_loop_scheduler._nested_readiness_frontier(
            graph,
            event,
            event.consumers[0],
            pipeline_plan=plan,
            static_producers=_static_producers,
            charge=cross_loop_scheduler._new_relation_work_budget(),
        )

        self.assertIsNotNone(frontier)
        assert frontier is not None
        self.assertEqual(frontier.target_domain, plan.wave_domain)
        self.assertEqual(frontier.prefix_before(3), 1)

    def test_exact_dependency_builds_one_counter_event(self) -> None:
        graph = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", block_id=10),
            _access(root=1, kind="load", block_id=20),
        )
        readiness = _configured_readiness_graph(graph, {10: (8, 16), 20: (8, 16)})

        self.assertEqual(len(readiness.events), 1)
        self.assertIsNone(readiness.events[0].root_barrier_producer_root)
        counters = choose_readiness_counters(
            readiness,
            (),
            charge=cross_loop_scheduler._new_relation_work_budget(),
        )
        self.assertEqual(len(counters), 1)
        self.assertEqual(counters[0].uniform_arrival_count(), 1)

    def test_unproved_mapping_builds_root_barrier_event(self) -> None:
        graph = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", block_id=10, exact=False),
            _access(root=1, kind="load", block_id=20, exact=False),
        )
        roots, sites = instantiate_coordinate_domains(
            graph,
            axis_geometry={10: (4, 16), 20: (4, 16)},
        )
        assert all(root is not None for root in roots)
        root_domains = tuple(root for root in roots if root is not None)
        readiness = ReadinessGraph(
            root_domains,
            cross_loop_scheduler._build_readiness_events(
                graph,
                root_domains=root_domains,
                site_domains=sites,
                charge=cross_loop_scheduler._new_relation_work_budget(),
            ),
        )

        self.assertEqual(len(readiness.events), 1)
        self.assertEqual(readiness.events[0].root_barrier_producer_root, 0)
        self.assertEqual(
            choose_readiness_counters(
                readiness,
                (),
                charge=cross_loop_scheduler._new_relation_work_budget(),
            ),
            (),
        )
        plan = build_static_pipeline_plan(
            dependency_graph=graph,
            root_task_orders=tuple(_dense(root) for root in root_domains),
            site_domains=sites,
            worker_count=4,
        )
        self.assertEqual(plan.readiness_counters, ())
        self.assertEqual(plan.root_barrier_edges, frozenset(((0, 1),)))

        dynamic = build_static_pipeline_plan(
            dependency_graph=graph,
            root_task_orders=tuple(_dense(root) for root in root_domains),
            site_domains=sites,
            worker_count=2,
            cross_loop_dispatch_mode="dynamic",
        )
        self.assertEqual(dynamic.dispatch_mode, "dynamic")
        self.assertEqual(dynamic.root_barrier_edges, plan.root_barrier_edges)
        self.assertEqual(dynamic.root_barrier_arrival_count(0), 4)

    def test_exhausted_relation_budget_uses_root_barrier(self) -> None:
        graph = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", block_id=10),
            _access(root=1, kind="load", block_id=20),
        )
        roots, sites = instantiate_coordinate_domains(
            graph, axis_geometry={10: (8, 16), 20: (8, 16)}
        )
        assert all(root is not None for root in roots)
        root_domains = tuple(root for root in roots if root is not None)
        with mock.patch.object(cross_loop_scheduler, "_MAX_SYMBOLIC_RELATION_WORK", 0):
            plan = build_static_pipeline_plan(
                dependency_graph=graph,
                root_task_orders=tuple(_dense(root) for root in root_domains),
                site_domains=sites,
                worker_count=4,
            )

        self.assertEqual(plan.readiness_counters, ())
        self.assertEqual(plan.root_barrier_edges, frozenset(((0, 1),)))

    def test_relation_budget_degrades_monotonically_to_root_barrier(self) -> None:
        budgets = (0, 1, 2, 4, 8, 12, 16, 32, 64, 128, 192)
        for mode_index, mode in enumerate(("static", "dynamic")):
            exact_counter_seen = False
            for index, budget in enumerate(budgets):
                producer_axis = 100 + 100 * mode_index + 2 * index
                consumer_axis = producer_axis + 1
                with self.subTest(mode=mode, budget=budget):
                    graph = _dependency_graph(
                        [[producer_axis], [consumer_axis]],
                        _access(root=0, kind="store", block_id=producer_axis),
                        _access(root=1, kind="load", block_id=consumer_axis),
                    )
                    roots, sites = instantiate_coordinate_domains(
                        graph,
                        axis_geometry={
                            producer_axis: (4, 16),
                            consumer_axis: (4, 16),
                        },
                    )
                    assert all(root is not None for root in roots)
                    root_domains = tuple(root for root in roots if root is not None)
                    with mock.patch.object(
                        cross_loop_scheduler,
                        "_MAX_SYMBOLIC_RELATION_WORK",
                        budget,
                    ):
                        plan = build_static_pipeline_plan(
                            dependency_graph=graph,
                            root_task_orders=tuple(
                                _dense(root) for root in root_domains
                            ),
                            site_domains=sites,
                            worker_count=4,
                            cross_loop_dispatch_mode=mode,
                        )

                    self.assertEqual(plan.dispatch_mode, mode)
                    outcome = (
                        bool(plan.readiness_counters),
                        bool(plan.root_barrier_edges),
                    )
                    self.assertIn(outcome, ((False, True), (True, False)))
                    if outcome[0]:
                        exact_counter_seen = True
                    else:
                        self.assertFalse(exact_counter_seen)
                    if mode == "dynamic":
                        self.assertEqual(plan.root_barrier_arrival_count(0), 4)

    def test_multi_producer_join_uses_one_readiness_event(self) -> None:
        graph = _dependency_graph(
            [[10], [20], [30]],
            _access(root=0, kind="store", block_id=10),
            _access(root=1, allocation_id=1, kind="store", block_id=20),
            _access(root=2, kind="load", block_id=30),
            _access(root=2, allocation_id=1, kind="load", block_id=30),
        )
        readiness = _configured_readiness_graph(
            graph,
            {10: (8, 16), 20: (8, 16), 30: (8, 16)},
        )

        self.assertEqual(len(readiness.events), 1)
        self.assertEqual(
            tuple(producer.producer_root for producer in readiness.events[0].producers),
            (0, 1),
        )
        (counter,) = choose_readiness_counters(
            readiness,
            (),
            charge=cross_loop_scheduler._new_relation_work_budget(),
        )
        self.assertEqual(counter.uniform_arrival_count(), 2)

    def _transitive_chain_plan(
        self,
        mode: cross_loop_scheduler.CrossLoopDispatchMode,
        *,
        middle_reads_source: bool = True,
        consumer_axis_geometry: tuple[int, int] = (8, 16),
    ) -> tuple[StaticPipelinePlan, TileDependencyGraph]:
        """Source S (8 tasks) -> middle M (reads all of S, keyed output) ->
        consumer C (reads M's tile r and all of S)."""

        def full_load(root: int, allocation_id: int) -> TileAccess:
            return dataclasses.replace(
                _access(
                    root=root, kind="load", block_id=0, allocation_id=allocation_id
                ),
                subscript_affine_block_ids=(None,),
                subscript_is_full_slice=(True,),
            )

        accesses = [_access(root=0, kind="store", block_id=10)]
        if middle_reads_source:
            accesses.append(full_load(1, 0))
        accesses += [
            _access(root=1, allocation_id=1, kind="store", block_id=20),
            _access(root=2, allocation_id=1, kind="load", block_id=30),
            full_load(2, 0),
        ]
        graph = _dependency_graph([[10], [20], [30]], *accesses)
        roots, sites = instantiate_coordinate_domains(
            graph,
            axis_geometry={10: (8, 16), 20: (8, 16), 30: consumer_axis_geometry},
        )
        assert all(root is not None for root in roots)
        root_domains = tuple(root for root in roots if root is not None)
        plan = build_static_pipeline_plan(
            dependency_graph=graph,
            root_task_orders=tuple(_dense(root) for root in root_domains),
            site_domains=sites,
            worker_count=32,
            cross_loop_dispatch_mode=mode,
        )
        return plan, graph

    def test_keyed_wait_certifies_what_its_producer_waited_on(self) -> None:
        # C's keyed wait on M covers C's read of S: every M task waited on all of
        # S before publishing, so no S -> C barrier is needed (S publishes one
        # only for M).
        for mode in ("static", "dynamic"):
            with self.subTest(mode=mode):
                plan, graph = self._transitive_chain_plan(mode)
                self.assertNotIn((0, 2), plan.root_barrier_edges)
                pairs = dict(graph.obligations_by_root_pair())
                carriers = [
                    consumer
                    for counter in plan.readiness_counters
                    for consumer in counter.consumers
                    if consumer.consumer_root == 2
                    and {p.producer_root for p in counter.producers} == {1}
                ]
                self.assertEqual(len(carriers), 1)
                # The keyed consumer now also names C's read of S, so the
                # lowering places its wait before that read.
                self.assertLessEqual(pairs[(0, 2)], carriers[0].covered_obligations)

    def test_transitive_cover_needs_the_middle_root_to_wait(self) -> None:
        # Negative control: M does not read S, so M's publication certifies
        # nothing about S and C keeps its own S -> C synchronization.
        for mode in ("static", "dynamic"):
            with self.subTest(mode=mode):
                plan, graph = self._transitive_chain_plan(
                    mode, middle_reads_source=False
                )
                pairs = dict(graph.obligations_by_root_pair())
                carried = frozenset(
                    obligation
                    for counter in plan.readiness_counters
                    for consumer in counter.consumers
                    if consumer.consumer_root == 2
                    and 0 not in {p.producer_root for p in counter.producers}
                    for obligation in consumer.covered_obligations
                )
                self.assertFalse(pairs[(0, 2)] & carried)
                self.assertIn((0, 2), plan.root_barrier_edges)

    def test_transitive_cover_rejects_consumers_without_a_key(self) -> None:
        # Negative control: C's tasks 8.. lie past M's output, so they have no
        # key and skip the (guarded) keyed wait, yet still read all of S: the
        # certified relation does not cover them and they keep the S -> C
        # barrier.
        for mode in ("static", "dynamic"):
            with self.subTest(mode=mode):
                plan, _graph = self._transitive_chain_plan(
                    mode, consumer_axis_geometry=(16, 16)
                )
                self.assertIn((0, 2), plan.root_barrier_edges)

    def test_dispatch_order_must_permute_the_dynamic_resident_roots(self) -> None:
        # Two independent chains A -> B and C -> D: without a reordering
        # feature the ticket stream keeps source order (empty dispatch order),
        # and a dispatch order must be a permutation of a dynamic plan's
        # resident roots.
        def full_load(root: int, allocation_id: int) -> TileAccess:
            return dataclasses.replace(
                _access(
                    root=root, kind="load", block_id=0, allocation_id=allocation_id
                ),
                subscript_affine_block_ids=(None,),
                subscript_is_full_slice=(True,),
            )

        graph = _dependency_graph(
            [[10], [20], [30], [40]],
            _access(root=0, kind="store", block_id=10),
            full_load(1, 0),
            _access(root=2, allocation_id=1, kind="store", block_id=30),
            full_load(3, 1),
        )
        roots, sites = instantiate_coordinate_domains(
            graph,
            axis_geometry={10: (8, 16), 20: (8, 16), 30: (32, 4), 40: (32, 4)},
        )
        assert all(root is not None for root in roots)
        root_domains = tuple(root for root in roots if root is not None)
        orders = tuple(_dense(root) for root in root_domains)
        dynamic = build_static_pipeline_plan(
            dependency_graph=graph,
            root_task_orders=orders,
            site_domains=sites,
            worker_count=8,
            cross_loop_dispatch_mode="dynamic",
        )
        self.assertEqual(dynamic.dispatch_order, ())
        static = build_static_pipeline_plan(
            dependency_graph=graph,
            root_task_orders=orders,
            site_domains=sites,
            worker_count=8,
        )
        self.assertEqual(static.dispatch_order, ())
        dataclasses.replace(dynamic, dispatch_order=(2, 0, 3, 1))
        with self.assertRaisesRegex(ValueError, "permute"):
            dataclasses.replace(static, dispatch_order=(1, 0, 2, 3))
        with self.assertRaisesRegex(ValueError, "permute"):
            dataclasses.replace(dynamic, dispatch_order=(1, 0, 2))

    def test_sink_lpt_order_claims_long_task_sinks_first(self) -> None:
        # A (8 tasks) feeds two sinks that each read all of it: B (cheap tasks)
        # then C (expensive tasks) in source order, and an unrelated chain
        # D -> E written after them.  Only the trailing run of sinks moves, and
        # only when every cost in it is known.
        def full_load(root: int, allocation_id: int) -> TileAccess:
            return dataclasses.replace(
                _access(
                    root=root, kind="load", block_id=0, allocation_id=allocation_id
                ),
                subscript_affine_block_ids=(None,),
                subscript_is_full_slice=(True,),
            )

        graph = _dependency_graph(
            [[10], [20], [30]],
            _access(root=0, kind="store", block_id=10),
            full_load(1, 0),
            full_load(2, 0),
        )
        roots, sites = instantiate_coordinate_domains(
            graph, axis_geometry={10: (8, 16), 20: (8, 16), 30: (8, 16)}
        )
        orders = tuple(_dense(root) for root in roots if root is not None)

        def plan(
            costs: tuple[int | None, ...], mode: str = "dynamic", lpt: bool = True
        ) -> StaticPipelinePlan:
            return build_static_pipeline_plan(
                dependency_graph=graph,
                root_task_orders=orders,
                site_domains=sites,
                worker_count=8,
                cross_loop_dispatch_mode=cast("Any", mode),
                sink_lpt_order=lpt,
                root_task_costs=costs,
            )

        self.assertEqual(plan((64, 4096, 65536)).dispatch_order, (0, 2, 1))
        # Already longest first, ties, an unknown cost, the setting off, and a
        # static pipeline: source order.
        self.assertEqual(plan((64, 65536, 4096)).dispatch_order, ())
        self.assertEqual(plan((64, 4096, 4096)).dispatch_order, ())
        self.assertEqual(plan((64, None, 65536)).dispatch_order, ())
        self.assertEqual(plan((64, 4096, 65536), lpt=False).dispatch_order, ())
        self.assertEqual(plan((64, 4096, 65536), mode="static").dispatch_order, ())
        # The producer is never part of the moved run, whatever its cost.
        self.assertEqual(plan((1 << 30, 4096, 65536)).dispatch_order, (0, 2, 1))

        # A sink followed by a producer -> consumer pair: the trailing run is
        # just the last consumer, so nothing moves (B keeps its slot ahead of D).
        chained = _dependency_graph(
            [[10], [20], [30], [40]],
            _access(root=0, kind="store", block_id=10),
            full_load(1, 0),
            _access(root=2, allocation_id=1, kind="store", block_id=30),
            full_load(3, 1),
        )
        roots, sites = instantiate_coordinate_domains(
            chained,
            axis_geometry={10: (8, 16), 20: (8, 16), 30: (8, 16), 40: (8, 16)},
        )
        chained_plan = build_static_pipeline_plan(
            dependency_graph=chained,
            root_task_orders=tuple(_dense(root) for root in roots if root is not None),
            site_domains=sites,
            worker_count=8,
            cross_loop_dispatch_mode="dynamic",
            sink_lpt_order=True,
            root_task_costs=(64, 65536, 64, 4096),
        )
        self.assertEqual(chained_plan.dispatch_order, ())
        # Every producer precedes its consumers in a reordered stream.
        reordered = plan((64, 4096, 65536))
        position = {root: index for index, root in enumerate(reordered.dispatch_order)}
        for (producer, consumer), _obligations in graph.obligations_by_root_pair():
            self.assertLess(position[producer], position[consumer])

    def test_lpt_tail_reorders_the_producers_of_a_join(self) -> None:
        # A (8 tasks) feeds B (cheap) and C (expensive), which both feed a join
        # J that reads all of them (an in-kernel finalize).  B and C now have a
        # consumer, but J waits for every task of both (root barriers), so
        # only the makespan of B and C counts: C goes first, J stays last.
        def full_load(root: int, allocation_id: int) -> TileAccess:
            return dataclasses.replace(
                _access(
                    root=root, kind="load", block_id=0, allocation_id=allocation_id
                ),
                subscript_affine_block_ids=(None,),
                subscript_is_full_slice=(True,),
            )

        geometry = {10: (8, 16), 20: (8, 16), 30: (8, 16), 40: (8, 16)}

        def plan(
            join_b: TileAccess, costs: tuple[int | None, ...], sources: bool = False
        ) -> StaticPipelinePlan:
            accesses = [
                _access(root=0, kind="store", block_id=10),
                *(() if sources else (full_load(1, 0), full_load(2, 0))),
                _access(root=1, allocation_id=1, kind="store", block_id=20),
                _access(root=2, allocation_id=2, kind="store", block_id=30),
                join_b,
                full_load(3, 2),
            ]
            graph = _dependency_graph([[10], [20], [30], [40]], *accesses)
            roots, sites = instantiate_coordinate_domains(graph, axis_geometry=geometry)
            built = build_static_pipeline_plan(
                dependency_graph=graph,
                root_task_orders=tuple(
                    _dense(root) for root in roots if root is not None
                ),
                site_domains=sites,
                worker_count=8,
                cross_loop_dispatch_mode="dynamic",
                sink_lpt_order=True,
                root_task_costs=costs,
            )
            order = built.dispatch_order or built.resident_roots
            position = {root: index for index, root in enumerate(order)}
            for (producer, consumer), _obligations in graph.obligations_by_root_pair():
                self.assertLess(position[producer], position[consumer])
            return built

        costs = (64, 4096, 65536, 1024)
        self.assertEqual(plan(full_load(3, 1), costs).dispatch_order, (0, 2, 1, 3))
        # The longest path counts, not the root's own task: a cheap B whose
        # join is expensive still trails C only by rank.
        self.assertEqual(
            plan(full_load(3, 1), (64, 4096, 65536, 1 << 20)).dispatch_order,
            (0, 2, 1, 3),
        )
        self.assertEqual(
            plan(full_load(3, 1), (64, 65536, 4096, 1024)).dispatch_order, ()
        )
        # J reads B tile by tile (a keyed readiness counter): B's finish order
        # gates J's tasks, so B is no makespan-only root and nothing moves.
        keyed = _access(root=3, allocation_id=1, kind="load", block_id=40)
        self.assertEqual(plan(keyed, costs).dispatch_order, ())
        # B and C without producers are sources: their order is the author's
        # first-wave priority, so only the join is in the tail run.
        self.assertEqual(plan(full_load(3, 1), costs, sources=True).dispatch_order, ())

        # Longest path, not own task: A feeds B (cheap) and C (a mid-size
        # sink); D (expensive tasks) reads all of B.  B heads the path B -> D,
        # so the stream is A B D C (by task cost alone it would be A C B D).
        chain = _dependency_graph(
            [[10], [20], [30], [40]],
            _access(root=0, kind="store", block_id=10),
            full_load(1, 0),
            full_load(2, 0),
            _access(root=1, allocation_id=1, kind="store", block_id=20),
            full_load(3, 1),
        )
        roots, sites = instantiate_coordinate_domains(chain, axis_geometry=geometry)
        chain_plan = build_static_pipeline_plan(
            dependency_graph=chain,
            root_task_orders=tuple(_dense(root) for root in roots if root is not None),
            site_domains=sites,
            worker_count=8,
            cross_loop_dispatch_mode="dynamic",
            sink_lpt_order=True,
            root_task_costs=(64, 4096, 65536, 1 << 20),
        )
        self.assertEqual(chain_plan.dispatch_order, (0, 1, 3, 2))

    def test_source_root_task_groups_bound_first_wave_bytes(self) -> None:
        # S (192 tasks, no producer) feeds C (32 tasks, reads all of S); R (6
        # tasks) is a small source.  With 296 workers S's first wave puts
        # 192 * cost bytes in flight; above the budget it claims ceil(bytes /
        # budget) consecutive tasks per ticket.  Consumers are never grouped.
        def full_load(root: int, allocation_id: int) -> TileAccess:
            return dataclasses.replace(
                _access(
                    root=root, kind="load", block_id=0, allocation_id=allocation_id
                ),
                subscript_affine_block_ids=(None,),
                subscript_is_full_slice=(True,),
            )

        graph = _dependency_graph(
            [[10], [20], [30]],
            _access(root=0, kind="store", block_id=10),
            full_load(1, 0),
            _access(root=2, allocation_id=1, kind="store", block_id=30),
        )
        roots, sites = instantiate_coordinate_domains(
            graph, axis_geometry={10: (192, 16), 20: (32, 16), 30: (6, 16)}
        )
        orders = tuple(_dense(root) for root in roots if root is not None)

        def groups(
            costs: tuple[int | None, ...],
            budget: int,
            mode: str = "dynamic",
            max_group: int | None = None,
        ) -> dict[int, int]:
            plan = build_static_pipeline_plan(
                dependency_graph=graph,
                root_task_orders=orders,
                site_domains=sites,
                worker_count=296,
                cross_loop_dispatch_mode=cast("Any", mode),
            )
            extra = {} if max_group is None else {"max_group": max_group}
            return cross_loop_scheduler.source_root_task_groups(
                plan, graph.obligations_by_root_pair(), costs, budget, **extra
            )

        kib = 1024
        self.assertEqual(groups((72 * kib, 1 << 30, 2 * kib), 8 << 20), {0: 2})
        self.assertEqual(
            groups((72 * kib, 1 << 30, 2 * kib), 4 << 20, max_group=8), {0: 4}
        )
        # The default cap: however far over the budget, a ticket claims at
        # most SOURCE_ROOT_MAX_GROUP (2) tasks (g=3..7 serialization cliffs).
        self.assertEqual(cross_loop_scheduler.SOURCE_ROOT_MAX_GROUP, 2)
        self.assertEqual(groups((72 * kib, 1 << 30, 2 * kib), 4 << 20), {0: 2})
        self.assertEqual(groups((72 * kib, 1 << 30, 2 * kib), 2 << 20), {0: 2})
        self.assertEqual(groups((1 << 40, 1 << 30, 1 << 40), 8 << 20), {0: 2, 2: 2})
        # A cap of 1 (or less) disables grouping.
        self.assertEqual(groups((72 * kib, 1 << 30, 2 * kib), 4 << 20, max_group=1), {})
        self.assertEqual(groups((72 * kib, 1 << 30, 2 * kib), 4 << 20, max_group=0), {})
        # Under the budget, an unknown cost, the budget off, a static pipeline.
        self.assertEqual(groups((40 * kib, 1 << 30, 2 * kib), 8 << 20), {})
        self.assertEqual(groups((None, 1 << 30, 2 * kib), 8 << 20), {})
        self.assertEqual(groups((72 * kib, 1 << 30, 2 * kib), 0), {})
        self.assertEqual(
            groups((72 * kib, 1 << 30, 2 * kib), 8 << 20, mode="static"), {}
        )
        # A group never exceeds the root's task count.
        self.assertEqual(
            groups((1 << 40, 1 << 30, 1 << 40), 8 << 20, max_group=1 << 20),
            {0: 192, 2: 6},
        )

        # A source whose consumer starts on part of it (here C reads S tile by
        # tile: a keyed counter) is never grouped -- a grouped task finishes
        # later than on its own slot, and C's tasks would wait for it.
        keyed = _dependency_graph(
            [[10], [20]],
            _access(root=0, kind="store", block_id=10),
            _access(root=1, kind="load", block_id=20),
        )
        roots, sites = instantiate_coordinate_domains(
            keyed, axis_geometry={10: (192, 16), 20: (192, 16)}
        )
        keyed_plan = build_static_pipeline_plan(
            dependency_graph=keyed,
            root_task_orders=tuple(_dense(root) for root in roots if root is not None),
            site_domains=sites,
            worker_count=296,
            cross_loop_dispatch_mode="dynamic",
        )
        self.assertTrue(keyed_plan.readiness_counters)
        self.assertEqual(
            cross_loop_scheduler.source_root_task_groups(
                keyed_plan,
                keyed.obligations_by_root_pair(),
                (72 * kib, 1 << 30),
                8 << 20,
            ),
            {},
        )

    def test_root_barrier_selection_is_forward_and_transitively_minimal(self) -> None:
        obligations = (
            ((0, 1), frozenset(((0, None, None),))),
            ((1, 2), frozenset(((1, None, None),))),
            ((0, 2), frozenset(((2, None, None),))),
        )
        self.assertEqual(
            _select_root_barrier_edges(
                obligations_by_root_pair=obligations,
                covered_obligations=frozenset(),
            ),
            frozenset(((0, 1), (1, 2))),
        )
        with self.assertRaisesRegex(exc.CrossLoopSchedulingError, "source-ordered"):
            _select_root_barrier_edges(
                obligations_by_root_pair=(((1, 0), frozenset(((3, None, None),))),),
                covered_obligations=frozenset(),
            )
