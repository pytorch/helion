from __future__ import annotations

import dataclasses
from unittest import mock

import sympy
import torch

from test._cross_loop_schedule_oracle import event_source_traversal
from test._cross_loop_schedule_oracle import placement
from test._cross_loop_schedule_oracle import segment_placement
from test._cross_loop_schedule_oracle import segment_task_at
from test._cross_loop_schedule_oracle import task_at
from test._cross_loop_schedule_oracle import task_order
from test._cross_loop_schedule_oracle import validate_worker_schedule
from test._cross_loop_test_kernels import nested_store_chain
from test._cross_loop_test_kernels import streamed_singleton_reduction
from test._cross_loop_test_utils import _access
from test._cross_loop_test_utils import _configured_event_graph
from test._cross_loop_test_utils import _default_root_traversals
from test._cross_loop_test_utils import _event_contribution_from_publication
from test._cross_loop_test_utils import _expected_arrivals
from test._cross_loop_test_utils import _identify_root_domains
from test._cross_loop_test_utils import _one_dimensional_domains
from test._cross_loop_test_utils import _one_dimensional_task_range
from test._cross_loop_test_utils import _publication
from test._cross_loop_test_utils import build_baseline_worker_schedule
from test._cross_loop_test_utils import build_cross_loop_schedule
from test._cross_loop_test_utils import build_keyed_events

from helion._compiler.cross_loop_scheduler import CountedEventPlan
from helion._compiler.cross_loop_scheduler import EventContribution
from helion._compiler.cross_loop_scheduler import EventGraph
from helion._compiler.cross_loop_scheduler import EventUse
from helion._compiler.cross_loop_scheduler import KeyedEvent
from helion._compiler.cross_loop_scheduler import WorkerSchedule
from helion._compiler.cross_loop_scheduler import WorkerScheduleSegment
from helion._compiler.cross_loop_scheduler import _select_root_completion_edges
from helion._compiler.cross_loop_scheduler import (
    build_baseline_worker_schedule as _build_baseline_worker_schedule,
)
from helion._compiler.cross_loop_scheduler import choose_counted_events
from helion._compiler.cross_loop_scheduler import choose_local_triggers
from helion._compiler.cross_loop_scheduler import derive_local_triggers
from helion._compiler.cross_loop_scheduler import order_local_contributors_by_key
from helion._compiler.cross_loop_scheduler import place_nested_scope_consumers
from helion._compiler.tile_dependency import ExecutionScope
from helion._compiler.tile_dependency import LogicalDomain
from helion._compiler.tile_dependency import LogicalRelation
from helion._compiler.tile_dependency import TileAccess
from helion._compiler.tile_dependency import _LogicalRelationPiece
from helion._compiler.tile_dependency import build_tile_dependency_graph
from helion._compiler.tile_dependency import instantiate_logical_domains
from helion._compiler.tile_dependency import instantiate_symbolic_dependencies
from helion._compiler.tile_dependency import logical_axis_symbol
from helion._compiler.tile_dependency import physical_traversal_relation
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import skipIfNotCUDA


class TestCrossLoopScheduler(TestCase):
    def test_large_affine_schedule_never_materializes_task_relations(self) -> None:
        size = 319_488
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(size,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(size,),
                    block_ids=(20,),
                ),
            ),
            [[10], [20]],
        )
        root_domains = _one_dimensional_domains(
            producer_count=size // 16,
            consumer_count=size // 512,
            producer_block=16,
            consumer_block=512,
        )

        with mock.patch.object(
            LogicalRelation,
            "materialize",
            side_effect=AssertionError("production scheduling expanded a relation"),
        ):
            schedule = build_cross_loop_schedule(
                dependency_plan=plan,
                root_domains=root_domains,
                axis_geometry={10: (size // 16, 16), 20: (size // 512, 512)},
                worker_count=148,
            )

        self.assertLessEqual(
            sum(
                len(contributor.predecessors.pieces)
                for event in schedule.counted_events
                for contributor in event.contributions
            ),
            4,
        )

    def test_symbolic_keyed_event_keeps_relations_compact(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(65,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(65,),
                    block_ids=(20,),
                ),
            ),
            [[10], [20]],
        )

        events = build_keyed_events(
            plan,
            axis_geometry={10: (5, 16), 20: (3, 32)},
        )

        self.assertIsNotNone(events)
        assert events is not None
        (event,) = events
        self.assertEqual(event.key_count, 3)
        self.assertEqual(len(event.contributions[0].predecessors.pieces), 1)
        self.assertEqual(len(event.uses[0].keys.pieces), 1)
        self.assertEqual(
            event.uses[0].keys.materialize(),
            (frozenset((0,)), frozenset((1,)), frozenset((2,))),
        )
        self.assertEqual(
            _publication(event.contributions[0]).materialize(),
            (
                frozenset((0,)),
                frozenset((0,)),
                frozenset((1,)),
                frozenset((1,)),
                frozenset((2,)),
            ),
        )

    def test_symbolic_keyed_event_joins_multiple_producers(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(64,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=1,
                    kind="store",
                    shape=(64,),
                    block_ids=(20,),
                ),
                _access(
                    2,
                    root=2,
                    allocation_id=0,
                    kind="load",
                    shape=(64,),
                    block_ids=(30,),
                ),
                _access(
                    3,
                    root=2,
                    allocation_id=1,
                    kind="load",
                    shape=(64,),
                    block_ids=(30,),
                ),
            ),
            [[10], [20], [30]],
        )

        events = build_keyed_events(
            plan,
            axis_geometry={10: (4, 16), 20: (4, 16), 30: (2, 32)},
        )

        self.assertIsNotNone(events)
        assert events is not None
        (event,) = events
        self.assertEqual(event.key_count, 2)
        self.assertEqual(len(event.contributions), 2)
        self.assertEqual(
            tuple(
                _publication(contribution).materialize()
                for contribution in event.contributions
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
        self.assertEqual(len(event.uses[0].dependency_points), 2)

    def test_symbolic_keyed_event_drops_irrelevant_consumer_axis(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(2, 64),
                    strides=(64, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(2, 64),
                    strides=(64, 1),
                    block_ids=(20, 22),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
            ),
            [[10, 11], [20, 21, 22]],
        )

        events = build_keyed_events(
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
        self.assertEqual(event.key_domain.axis_order, (0, 1))
        self.assertEqual(event.key_domain.block_sizes_items, ())
        self.assertEqual(event.key_count, 8)
        for consumer_task in range(event.uses[0].keys.source_domain.size):
            consumer_coordinates = event.uses[0].keys.source_domain.coordinates(
                consumer_task
            )
            expected_key = consumer_coordinates[20] + 2 * consumer_coordinates[22]
            self.assertEqual(
                event.uses[0].keys.targets(consumer_task),
                frozenset((expected_key,)),
            )

    def test_symbolic_keyed_event_coalesces_equivalent_fanout(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(2, 64),
                    strides=(64, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(2, 64),
                    strides=(64, 1),
                    block_ids=(20, 22),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    2,
                    root=2,
                    allocation_id=0,
                    kind="load",
                    shape=(2, 64),
                    strides=(64, 1),
                    block_ids=(30, 32),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
            ),
            [[10, 11], [20, 21, 22], [30, 31, 32]],
        )

        events = build_keyed_events(
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
        self.assertEqual(event.key_domain.axis_order, (0, 1))
        self.assertEqual(event.key_count, 8)
        self.assertEqual({use.consumer_root for use in event.uses}, {1, 2})

    def test_symbolic_keyed_event_does_not_coalesce_swapped_axes(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(2, 2),
                    strides=(2, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(2, 2),
                    strides=(2, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    2,
                    root=2,
                    allocation_id=0,
                    kind="load",
                    shape=(2, 2),
                    strides=(2, 1),
                    block_ids=(31, 30),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
            ),
            [[10, 11], [20, 21], [30, 31]],
        )

        events = build_keyed_events(
            plan,
            axis_geometry=dict.fromkeys((10, 11, 20, 21, 30, 31), (2, 1)),
        )

        self.assertIsNotNone(events)
        assert events is not None
        self.assertEqual(len(events), 2)
        self.assertEqual(
            [{use.consumer_root for use in event.uses} for event in events],
            [{1}, {2}],
        )
        self.assertNotEqual(
            events[0].contributions[0].predecessors,
            events[1].contributions[0].predecessors,
        )

    def test_symbolic_keyed_event_uses_one_chart_for_multi_producer_join(
        self,
    ) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(2, 2),
                    strides=(2, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=1,
                    kind="store",
                    shape=(2, 2),
                    strides=(2, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    2,
                    root=2,
                    allocation_id=0,
                    kind="load",
                    shape=(2, 2),
                    strides=(2, 1),
                    block_ids=(30, 31),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    3,
                    root=2,
                    allocation_id=1,
                    kind="load",
                    shape=(2, 2),
                    strides=(2, 1),
                    block_ids=(30, 31),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    4,
                    root=3,
                    allocation_id=0,
                    kind="load",
                    shape=(2, 2),
                    strides=(2, 1),
                    block_ids=(40, 41),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    5,
                    root=3,
                    allocation_id=1,
                    kind="load",
                    shape=(2, 2),
                    strides=(2, 1),
                    block_ids=(41, 40),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
            ),
            [[10, 11], [20, 21], [30, 31], [40, 41]],
        )

        events = build_keyed_events(
            plan,
            axis_geometry=dict.fromkeys((10, 11, 20, 21, 30, 31, 40, 41), (2, 1)),
        )

        self.assertIsNotNone(events)
        assert events is not None
        self.assertEqual(len(events), 2)
        self.assertTrue(all(len(event.contributions) == 2 for event in events))
        self.assertEqual(
            [{use.consumer_root for use in event.uses} for event in events],
            [{2}, {3}],
        )

    def test_symbolic_keyed_event_unions_disjoint_producer_ranges(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(64,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(64,),
                    block_ids=(20,),
                ),
                _access(
                    2,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(64,),
                    block_ids=(20,),
                    offsets=(32,),
                ),
            ),
            [[10], [20]],
        )

        events = build_keyed_events(
            plan,
            axis_geometry={10: (32, 2), 20: (2, 16)},
        )

        self.assertIsNotNone(events)
        assert events is not None
        (event,) = events
        self.assertEqual(event.key_count, 2)
        self.assertEqual(len(event.contributions), 1)
        self.assertEqual(len(event.contributions[0].predecessors.pieces), 2)
        expected = tuple(frozenset((producer // 8 % 2,)) for producer in range(32))
        self.assertEqual(_publication(event.contributions[0]).materialize(), expected)

    def test_unsupported_symbolic_event_coarsens_to_family_done(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(64,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(64,),
                    block_ids=(20,),
                    masked=True,
                ),
            ),
            [[10], [20]],
        )

        events = build_keyed_events(
            plan,
            axis_geometry={10: (4, 16), 20: (2, 32)},
        )

        self.assertIsNotNone(events)
        assert events is not None
        (event,) = events
        self.assertEqual(event.key_count, 1)
        self.assertEqual(event.contributions[0].producer_root, 0)
        self.assertEqual(
            _publication(event.contributions[0]).materialize(),
            (frozenset((0,)),) * 4,
        )
        self.assertEqual(
            event.uses[0].keys.materialize(),
            (frozenset((0,)),) * 2,
        )

    def test_unsupported_event_quotient_does_not_coarsen_unrelated_edges(
        self,
    ) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(64,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(64,),
                    block_ids=(20,),
                ),
                _access(
                    2,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(64,),
                    block_ids=(20,),
                ),
                _access(
                    3,
                    root=2,
                    allocation_id=1,
                    kind="store",
                    shape=(64,),
                    block_ids=(30,),
                ),
                _access(
                    4,
                    root=3,
                    allocation_id=1,
                    kind="load",
                    shape=(64,),
                    block_ids=(40,),
                ),
            ),
            [[10], [20], [30], [40]],
        )

        original_union = LogicalRelation.union
        failed_once = False

        def fail_first_union(
            left: LogicalRelation,
            right: LogicalRelation,
        ) -> LogicalRelation | None:
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                return None
            return original_union(left, right)

        with mock.patch.object(LogicalRelation, "union", fail_first_union):
            events = build_keyed_events(
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
                contribution.producer_root == 2 for contribution in event.contributions
            )
        ]
        self.assertEqual(len(unrelated), 1)
        self.assertEqual(unrelated[0].key_count, 2)
        self.assertIsNone(unrelated[0].family_done_root)

    @skipIfNotCUDA()
    def test_device_ir_scopes_preserve_nested_producer_and_consumer_axes(
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
        (producer_scope,) = producer_graph.scopes_for_access(producer_store.access_id)
        self.assertEqual(producer_scope.kind, "loop")
        self.assertEqual(len(producer_scope.callsite_path), 1)
        self.assertEqual(
            producer_scope.logical_axis_order,
            (
                *producer_ir.task_families[0].logical_axis_order,
                *producer_scope.local_axis_order,
            ),
        )
        self.assertTrue(producer_scope.guaranteed)
        self.assertTrue(producer_scope.segmentable)

        producer_outer_axis = producer_ir.task_families[0].logical_axis_order[0]
        consumer_batch_axis, consumer_width_axis = producer_ir.task_families[
            1
        ].logical_axis_order
        producer_domains = (
            LogicalDomain(
                (producer_outer_axis,),
                ((producer_outer_axis, 2),),
                ((producer_outer_axis, 1),),
            ),
            LogicalDomain(
                (consumer_batch_axis, consumer_width_axis),
                ((consumer_batch_axis, 2), (consumer_width_axis, 4)),
                ((consumer_batch_axis, 1), (consumer_width_axis, 16)),
            ),
        )
        producer_axis_geometry = {
            producer_outer_axis: (2, 1),
            producer_scope.local_axis_order[0]: (4, 16),
            consumer_batch_axis: (2, 1),
            consumer_width_axis: (4, 16),
        }
        producer_events = _configured_event_graph(
            producer_graph,
            root_domains=producer_domains,
            axis_geometry=producer_axis_geometry,
        )
        producer_event = next(
            event
            for event in producer_events.events
            if any(
                contribution.producer_scope_id == producer_scope.scope_id
                for contribution in event.contributions
            )
        )
        self.assertEqual(producer_event.key_count, 8)
        self.assertEqual(
            _expected_arrivals(producer_event.key_domain, producer_event.contributions),
            (1,) * 8,
        )
        self.assertEqual(producer_event.uses[0].consumer_scope_id, None)

        synchronous_events = _configured_event_graph(
            producer_graph,
            root_domains=producer_domains,
            axis_geometry=producer_axis_geometry,
            publishable_scope_ids=frozenset(),
        )
        self.assertFalse(
            any(
                contribution.producer_scope_id is not None
                for event in synchronous_events.events
                for contribution in event.contributions
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
                scope.kind == "loop"
                for scope in consumer_graph.scopes_for_access(access.access_id)
            )
        )
        (consumer_scope,) = consumer_graph.scopes_for_access(consumer_load.access_id)
        self.assertEqual(consumer_scope.kind, "loop")
        self.assertEqual(len(consumer_scope.callsite_path), 1)
        self.assertEqual(
            consumer_scope.logical_axis_order,
            (
                *consumer_ir.task_families[1].logical_axis_order,
                *consumer_scope.local_axis_order,
            ),
        )
        self.assertTrue(consumer_scope.guaranteed)
        self.assertTrue(consumer_scope.segmentable)

        producer_batch_axis, producer_width_axis = consumer_ir.task_families[
            0
        ].logical_axis_order
        consumer_outer_axis = consumer_ir.task_families[1].logical_axis_order[0]
        consumer_domains = (
            LogicalDomain(
                (producer_batch_axis, producer_width_axis),
                ((producer_batch_axis, 2), (producer_width_axis, 4)),
                ((producer_batch_axis, 1), (producer_width_axis, 16)),
            ),
            LogicalDomain(
                (consumer_outer_axis,),
                ((consumer_outer_axis, 2),),
                ((consumer_outer_axis, 1),),
            ),
        )
        consumer_axis_geometry = {
            producer_batch_axis: (2, 1),
            producer_width_axis: (4, 16),
            consumer_outer_axis: (2, 1),
            consumer_scope.local_axis_order[0]: (4, 16),
        }
        consumer_events = _configured_event_graph(
            consumer_graph,
            root_domains=consumer_domains,
            axis_geometry=consumer_axis_geometry,
        )
        nested_event = next(
            event
            for event in consumer_events.events
            if any(
                use.consumer_scope_id == consumer_scope.scope_id for use in event.uses
            )
        )
        self.assertEqual(nested_event.key_count, 8)
        self.assertEqual(
            _expected_arrivals(nested_event.key_domain, nested_event.contributions),
            (1,) * 8,
        )
        (nested_use,) = nested_event.uses
        nested_keys = nested_use.keys.materialize(
            source_traversal=event_source_traversal(consumer_events, nested_use)
        )
        self.assertTrue(all(len(keys) == 1 for keys in nested_keys))
        self.assertEqual(
            {next(iter(keys)) for keys in nested_keys},
            set(range(8)),
        )

    def test_semantic_event_graph_composes_arbitrary_chain_depth(self) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store", block_ids=(10,)),
                _access(1, root=1, allocation_id=0, kind="load", block_ids=(20,)),
                _access(2, root=1, allocation_id=1, kind="store", block_ids=(20,)),
                _access(3, root=2, allocation_id=1, kind="load", block_ids=(30,)),
                _access(4, root=2, allocation_id=2, kind="store", block_ids=(30,)),
                _access(5, root=3, allocation_id=2, kind="load", block_ids=(40,)),
            ),
            [[10], [20], [30], [40]],
        )

        self.assertEqual(len(graph.task_families), 4)
        configured = _configured_event_graph(
            graph,
            tuple(
                LogicalDomain((block_id,), ((block_id, 4),), ((block_id, 1),))
                for block_id in (10, 20, 30, 40)
            ),
        )
        self.assertEqual(len(configured.events), 3)
        self.assertEqual(
            tuple(
                _expected_arrivals(event.key_domain, event.contributions)
                for event in configured.events
            ),
            ((1, 1, 1, 1),) * 3,
        )
        baseline = _build_baseline_worker_schedule(
            configured.root_domains,
            configured.root_traversals,
            worker_count=4,
        )
        local_triggers = derive_local_triggers(configured, baseline)
        self.assertEqual(
            tuple(
                configured.event(trigger.event_index)
                .uses[trigger.use_index]
                .consumer_root
                for trigger in local_triggers
            ),
            (1, 2, 3),
        )
        validate_worker_schedule(
            configured,
            baseline.without_roots(frozenset((1, 2, 3))),
            local_triggers,
        )

    def test_local_triggers_allow_disjoint_uses_of_one_producer_family(self) -> None:
        domains = tuple(
            LogicalDomain((axis,), ((axis, count),), identity=root)
            for root, (axis, count) in enumerate(((10, 4), (20, 2), (30, 2)))
        )
        key_domains = tuple(
            LogicalDomain((0,), ((0, 2),), kind="event", identity=event)
            for event in range(2)
        )

        def keys(
            domain: LogicalDomain,
            key_domain: LogicalDomain,
            begin: int,
            end: int,
        ) -> LogicalRelation:
            (axis,) = domain.axis_order
            return LogicalRelation.point_map(
                domain,
                key_domain,
                (
                    (
                        ((axis, begin, end, 1),),
                        (logical_axis_symbol(axis) - begin,),
                    ),
                ),
            )

        event_graph = EventGraph(
            root_traversals=tuple(
                physical_traversal_relation(domain, domain.axis_order)
                for domain in domains
            ),
            events=(
                KeyedEvent(
                    contributions=(
                        _event_contribution_from_publication(
                            0,
                            keys(domains[0], key_domains[0], 0, 2),
                        ),
                    ),
                    uses=(EventUse(1, keys(domains[1], key_domains[0], 0, 2)),),
                ),
                KeyedEvent(
                    contributions=(
                        _event_contribution_from_publication(
                            0,
                            keys(domains[0], key_domains[1], 2, 4),
                        ),
                        _event_contribution_from_publication(
                            1,
                            keys(domains[1], key_domains[1], 0, 2),
                        ),
                    ),
                    uses=(EventUse(2, keys(domains[2], key_domains[1], 0, 2)),),
                ),
            ),
        )
        baseline = _build_baseline_worker_schedule(
            domains,
            event_graph.root_traversals,
            worker_count=4,
        )

        triggers = derive_local_triggers(event_graph, baseline)

        self.assertEqual(
            tuple(
                event_graph.event(trigger.event_index)
                .uses[trigger.use_index]
                .consumer_root
                for trigger in triggers
            ),
            (1, 2),
        )

        overlapping = dataclasses.replace(
            event_graph,
            events=(
                event_graph.events[0],
                dataclasses.replace(
                    event_graph.events[1],
                    contributions=(
                        _event_contribution_from_publication(
                            0,
                            keys(domains[0], key_domains[1], 1, 3),
                        ),
                        event_graph.events[1].contributions[1],
                    ),
                ),
            ),
        )
        self.assertEqual(derive_local_triggers(overlapping, baseline), ())

    def test_local_trigger_requires_counted_event_lowerability(self) -> None:
        producer_domain = LogicalDomain((10,), ((10, 4),), identity=0)
        consumer_domain = LogicalDomain((20,), ((20, 2),), identity=1)
        key_domain = LogicalDomain((0,), ((0, 2),), kind="event", identity=0)
        contribution = EventContribution(
            producer_root=0,
            predecessors=LogicalRelation(
                key_domain,
                producer_domain,
                (
                    _LogicalRelationPiece(
                        ((0, 0, 2, 1),),
                        (
                            (
                                10,
                                2 * logical_axis_symbol(0),
                                2 * logical_axis_symbol(0) + 1,
                                1,
                            ),
                        ),
                    ),
                ),
            ),
        )
        event_graph = EventGraph(
            root_traversals=(
                physical_traversal_relation(producer_domain, (10,)),
                physical_traversal_relation(consumer_domain, (20,)),
            ),
            events=(
                KeyedEvent(
                    (contribution,),
                    (
                        EventUse(
                            1,
                            LogicalRelation.point_map(
                                consumer_domain,
                                key_domain,
                                (
                                    (
                                        ((20, 0, 2, 1),),
                                        (logical_axis_symbol(20),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        baseline = _build_baseline_worker_schedule(
            event_graph.root_domains,
            event_graph.root_traversals,
            worker_count=4,
        )

        publication = contribution.producer_to_keys
        self.assertIsNotNone(publication)
        assert publication is not None
        self.assertIsNone(publication.canonical_single_valued())
        self.assertEqual(derive_local_triggers(event_graph, baseline), ())
        self.assertEqual(choose_counted_events(event_graph, ()), ())

    def test_large_local_trigger_ignores_downstream_event_granularity(self) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store", block_ids=(10,)),
                _access(1, root=1, allocation_id=0, kind="load", block_ids=(20,)),
                _access(
                    2,
                    root=1,
                    allocation_id=1,
                    kind="store",
                    block_ids=(20,),
                    layout_is_static=False,
                ),
                _access(
                    3,
                    root=2,
                    allocation_id=1,
                    kind="load",
                    block_ids=(30,),
                    layout_is_static=False,
                ),
            ),
            [[10], [20], [30]],
        )
        root_domains = _identify_root_domains(
            (
                LogicalDomain((10,), ((10, 8),), ((10, 1),)),
                LogicalDomain((20,), ((20, 8),), ((20, 1),)),
                LogicalDomain((30,), ((30, 1),), ((30, 1),)),
            )
        )
        event_graph = _configured_event_graph(graph, root_domains)
        baseline = _build_baseline_worker_schedule(
            root_domains,
            event_graph.root_traversals,
            worker_count=4,
        )

        triggers = choose_local_triggers(event_graph, baseline)

        self.assertGreater(root_domains[1].size, baseline.worker_count)
        self.assertEqual(event_graph.events[1].family_done_root, 1)
        self.assertEqual(len(triggers), 1)
        use = event_graph.event(triggers[0].event_index).uses[triggers[0].use_index]
        self.assertEqual(use.consumer_root, 1)

    def test_semantic_event_graph_represents_diamond_without_path_matching(
        self,
    ) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store", block_ids=(10,)),
                _access(1, root=0, allocation_id=1, kind="store", block_ids=(10,)),
                _access(2, root=1, allocation_id=0, kind="load", block_ids=(20,)),
                _access(3, root=1, allocation_id=2, kind="store", block_ids=(20,)),
                _access(4, root=2, allocation_id=1, kind="load", block_ids=(30,)),
                _access(5, root=2, allocation_id=3, kind="store", block_ids=(30,)),
                _access(6, root=3, allocation_id=2, kind="load", block_ids=(40,)),
                _access(7, root=3, allocation_id=3, kind="load", block_ids=(40,)),
            ),
            [[10], [20], [30], [40]],
        )

        configured = _configured_event_graph(
            graph,
            tuple(
                LogicalDomain((block_id,), ((block_id, 4),), ((block_id, 1),))
                for block_id in (10, 20, 30, 40)
            ),
        )
        (root_zero_event,) = tuple(
            event
            for event in configured.events
            if any(
                contribution.producer_root == 0 for contribution in event.contributions
            )
        )
        self.assertEqual(
            {use.consumer_root for use in root_zero_event.uses},
            {1, 2},
        )
        local_triggers = derive_local_triggers(
            configured,
            _build_baseline_worker_schedule(
                configured.root_domains,
                configured.root_traversals,
                worker_count=4,
            ),
        )
        self.assertEqual(
            {
                configured.event(trigger.event_index)
                .uses[trigger.use_index]
                .consumer_root
                for trigger in local_triggers
            },
            {3},
        )

    def test_exact_and_family_done_dependencies_can_share_one_consumer(
        self,
    ) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    block_ids=(10,),
                    layout_is_static=False,
                ),
                _access(1, root=1, allocation_id=1, kind="store", block_ids=(20,)),
                _access(
                    2,
                    root=2,
                    allocation_id=0,
                    kind="load",
                    block_ids=(30,),
                    layout_is_static=False,
                ),
                _access(3, root=2, allocation_id=1, kind="load", block_ids=(30,)),
            ),
            [[10], [20], [30]],
        )

        configured = _configured_event_graph(
            graph,
            tuple(
                LogicalDomain((block_id,), ((block_id, 4),), ((block_id, 1),))
                for block_id in (10, 20, 30)
            ),
        )
        configured_uses = tuple(
            use
            for event in configured.events
            for use in event.uses
            if use.consumer_root == 2
        )
        self.assertEqual(len(configured_uses), 2)
        family_event = next(
            event for event in configured.events if event.family_done_root is not None
        )
        self.assertEqual(family_event.family_done_root, 0)
        self.assertEqual(
            _expected_arrivals(family_event.key_domain, family_event.contributions),
            (4,),
        )
        baseline = _build_baseline_worker_schedule(
            configured.root_domains,
            configured.root_traversals,
            worker_count=4,
        )
        self.assertEqual(derive_local_triggers(configured, baseline), ())

    def test_dependency_coverage_distinguishes_producer_callsites(self) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        scopes = (
            ExecutionScope(0, 0, 0, (), None, "root", (), (10,), True, False),
            ExecutionScope(
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
            ExecutionScope(2, 1, 1, (), None, "root", (), (20,), True, False),
        )
        graph = dataclasses.replace(
            graph,
            execution_scopes=scopes,
            scope_ids_by_access=((0, 1), (2,)),
        )
        access_dependency = graph.edges[0].access_dependencies[0]
        self.assertEqual(
            graph.dependency_points(access_dependency),
            frozenset(
                (
                    (access_dependency.dependency_id, 0, 2),
                    (access_dependency.dependency_id, 1, 2),
                )
            ),
        )
        axis_geometry = {10: (4, 32), 20: (4, 32)}
        configured_root_domains, configured_scope_domains = instantiate_logical_domains(
            graph,
            axis_geometry=axis_geometry,
        )
        exact_dependencies = instantiate_symbolic_dependencies(
            graph,
            root_domains=configured_root_domains,
            scope_domains=configured_scope_domains,
        )
        self.assertEqual(len(exact_dependencies), 1)

        events = build_keyed_events(graph, axis_geometry=axis_geometry)

        self.assertIsNotNone(events)
        assert events is not None
        self.assertEqual(len(events), 2)
        exact_event = next(event for event in events if event.family_done_root is None)
        family_event = next(
            event for event in events if event.family_done_root is not None
        )
        dependency_id = access_dependency.dependency_id
        self.assertEqual(
            exact_event.uses[0].dependency_points,
            frozenset(((dependency_id, 0, 2),)),
        )
        self.assertEqual(
            family_event.uses[0].dependency_points,
            frozenset(((dependency_id, 1, 2),)),
        )

        root_domains = tuple(
            domain for domain in configured_root_domains if domain is not None
        )
        event_graph = EventGraph(
            root_traversals=tuple(
                physical_traversal_relation(domain, domain.axis_order)
                for domain in root_domains
            ),
            events=events,
        )
        baseline = _build_baseline_worker_schedule(
            root_domains,
            event_graph.root_traversals,
            worker_count=4,
        )
        self.assertEqual(derive_local_triggers(event_graph, baseline), ())
        counted_events = choose_counted_events(event_graph, ())
        covered_points = frozenset(
            point
            for event in counted_events
            for use in event.uses
            for point in use.dependency_points
        )
        self.assertEqual(
            _select_root_completion_edges(
                dependency_graph=graph,
                covered_dependency_points=covered_points,
            ),
            frozenset(((0, 1),)),
        )

    def test_baseline_worker_schedule_preserves_source_order(self) -> None:
        root_domains = (
            LogicalDomain((10,), ((10, 3),), ((10, 1),)),
            LogicalDomain((20,), ((20, 5),), ((20, 1),)),
        )

        schedule = build_baseline_worker_schedule(root_domains, worker_count=4)

        self.assertEqual(placement(schedule, 0, 0), (0, 0))
        self.assertEqual(placement(schedule, 0, 2), (2, 0))
        self.assertEqual(placement(schedule, 1, 0), (0, 1))
        self.assertEqual(placement(schedule, 1, 4), (0, 2))
        self.assertEqual(task_at(schedule, 3, 0), None)
        self.assertEqual(task_at(schedule, 3, 1), (1, 3))

    def test_schedule_traversals_require_compatible_domains(self) -> None:
        task_domain = LogicalDomain((10,), ((10, 2),), identity=0)
        ordinal_domain = LogicalDomain(
            (-1,),
            ((-1, 1),),
            kind="worker",
            identity=0,
        )
        wrong_size = LogicalRelation.point_map(
            ordinal_domain,
            task_domain,
            ((((-1, 0, 1, 1),), (sympy.Integer(0),)),),
        )

        with self.assertRaisesRegex(ValueError, "compatible typed domains"):
            EventGraph(
                root_traversals=(wrong_size,),
                events=(),
            )
        with self.assertRaisesRegex(ValueError, "incompatible domains"):
            inverse = wrong_size.inverse()
            assert inverse is not None
            WorkerScheduleSegment(
                root=0,
                task_relation=inverse,
                worker_begin=0,
                worker_count=2,
                schedule_begin=0,
            )

    def test_baseline_worker_schedule_preserves_physical_traversal(self) -> None:
        root_domains = (LogicalDomain((10,), ((10, 4),), ((10, 1),), identity=0),)
        ordinal_domain = LogicalDomain(
            (-1,),
            ((-1, 4),),
            kind="worker",
            identity=0,
        )
        traversal = LogicalRelation.point_map(
            ordinal_domain,
            root_domains[0],
            tuple(
                (((-1, ordinal, ordinal + 1, 1),), (sympy.Integer(task),))
                for ordinal, task in enumerate((0, 2, 1, 3))
            ),
        )

        schedule = build_baseline_worker_schedule(
            root_domains,
            worker_count=2,
            root_traversals=(traversal,),
        )

        self.assertEqual(placement(schedule, 0, 0), (0, 0))
        self.assertEqual(placement(schedule, 0, 2), (1, 0))
        self.assertEqual(placement(schedule, 0, 1), (0, 1))
        self.assertEqual(placement(schedule, 0, 3), (1, 1))

    def test_worker_schedule_segment_uses_symbolic_order_across_rounds(self) -> None:
        task_axis = 10
        ordinal_axis = 20
        task_domain = LogicalDomain(
            (task_axis,),
            ((task_axis, 15),),
            identity=2,
        )
        ordinal_domain = LogicalDomain(
            (ordinal_axis,),
            ((ordinal_axis, 3),),
            kind="worker",
        )
        segment = WorkerScheduleSegment(
            root=2,
            task_relation=LogicalRelation.point_map(
                ordinal_domain,
                task_domain,
                (
                    (
                        ((ordinal_axis, 0, 3, 1),),
                        (10 + 2 * logical_axis_symbol(ordinal_axis),),
                    ),
                ),
            ),
            worker_begin=2,
            worker_count=2,
            schedule_begin=0,
        )

        self.assertEqual(segment_placement(segment, 10), (2, 0))
        self.assertEqual(segment_placement(segment, 12), (3, 0))
        self.assertEqual(segment_placement(segment, 14), (2, 1))
        self.assertEqual(segment_placement(segment, 11), None)
        self.assertEqual(segment_task_at(segment, 2, 1), 14)

    def test_worker_support_excludes_unused_segment_capacity(self) -> None:
        task_domain = LogicalDomain((10,), ((10, 2),), identity=0)
        schedule = WorkerSchedule(
            worker_count=6,
            segments=(
                WorkerScheduleSegment(
                    root=0,
                    task_relation=_one_dimensional_task_range(task_domain, 0, 2),
                    worker_begin=1,
                    worker_count=4,
                    schedule_begin=2,
                ),
            ),
        )

        self.assertEqual(schedule.workers_for_root(0), frozenset((3, 4)))
        self.assertEqual(schedule.dense_assignment(0), (1, 4, 2, 2))
        self.assertIsNone(schedule.contiguous_global_interval(0))

    def test_local_contributors_preserve_key_major_order(self) -> None:
        root_domains = (
            LogicalDomain((10,), ((10, 4),), ((10, 1),)),
            LogicalDomain((20,), ((20, 2),), ((20, 1),)),
        )
        producer_domain, consumer_domain = _identify_root_domains(root_domains)
        key_domain = LogicalDomain(
            (0,),
            ((0, 2),),
            kind="event",
            identity=0,
        )
        producer_axis = logical_axis_symbol(10)
        consumer_axis = logical_axis_symbol(20)
        event_graph = EventGraph(
            root_traversals=_default_root_traversals(
                (producer_domain, consumer_domain)
            ),
            events=(
                KeyedEvent(
                    contributions=(
                        _event_contribution_from_publication(
                            producer_root=0,
                            producer_scope_id=None,
                            publication=LogicalRelation.point_map(
                                producer_domain,
                                key_domain,
                                (
                                    (
                                        ((10, 0, 4, 1),),
                                        (sympy.floor(producer_axis / 2),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    uses=(
                        EventUse(
                            consumer_root=1,
                            consumer_scope_id=None,
                            keys=LogicalRelation.point_map(
                                consumer_domain,
                                key_domain,
                                (
                                    (
                                        ((20, 0, 2, 1),),
                                        (consumer_axis,),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        baseline = build_baseline_worker_schedule(
            event_graph.root_domains,
            worker_count=2,
        )
        triggers = derive_local_triggers(event_graph, baseline)

        schedule = order_local_contributors_by_key(
            event_graph,
            baseline,
            triggers,
        )

        self.assertEqual(task_order(schedule, 0), (0, 1, 2, 3))

    def test_worker_schedule_detects_dependency_order_cycle(self) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store", block_ids=(10,)),
                _access(1, root=1, allocation_id=0, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        root_domains = (
            LogicalDomain((10,), ((10, 1),), ((10, 1),)),
            LogicalDomain((20,), ((20, 1),), ((20, 1),)),
        )
        event_graph = _configured_event_graph(graph, root_domains)

        validate_worker_schedule(
            event_graph,
            build_baseline_worker_schedule(
                event_graph.root_domains,
                worker_count=1,
            ),
        )
        reversed_schedule = WorkerSchedule(
            worker_count=1,
            segments=(
                WorkerScheduleSegment(
                    root=1,
                    task_relation=event_graph.root_traversals[1],
                    worker_begin=0,
                    worker_count=1,
                    schedule_begin=0,
                ),
                WorkerScheduleSegment(
                    root=0,
                    task_relation=event_graph.root_traversals[0],
                    worker_begin=0,
                    worker_count=1,
                    schedule_begin=1,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "dependency/order cycle"):
            validate_worker_schedule(event_graph, reversed_schedule)

    def test_counted_event_supports_independent_consumer_uses(self) -> None:
        producer_domain = LogicalDomain((10,), ((10, 2),), identity=0)
        first_consumer = LogicalDomain((20,), ((20, 1),), identity=1)
        second_consumer = LogicalDomain((30,), ((30, 2),), identity=2)
        key_domain = LogicalDomain((), (), kind="event", identity=0)
        event = CountedEventPlan(
            contributions=(
                _event_contribution_from_publication(
                    producer_root=0,
                    publication=LogicalRelation.total(producer_domain, key_domain),
                ),
            ),
            uses=(
                EventUse(
                    consumer_root=1,
                    keys=LogicalRelation.total(first_consumer, key_domain),
                ),
                EventUse(
                    consumer_root=2,
                    keys=LogicalRelation.total(second_consumer, key_domain),
                ),
            ),
        )

        self.assertEqual(event.key_count, 1)
        self.assertEqual(event.uniform_arrivals(), 2)
        self.assertIsNone(event.local_use)
        self.assertEqual(tuple(use.consumer_root for use in event.uses), (1, 2))

    def test_counted_event_selection_keeps_independent_direct_uses(self) -> None:
        root_domains = tuple(
            LogicalDomain((axis,), ((axis, 2),), ((axis, 1),)) for axis in (10, 20, 30)
        )
        root_domains = _identify_root_domains(root_domains)
        key_domain = LogicalDomain((0,), ((0, 2),), kind="event", identity=0)
        event_graph = EventGraph(
            root_traversals=_default_root_traversals(root_domains),
            events=(
                KeyedEvent(
                    contributions=(
                        _event_contribution_from_publication(
                            producer_root=0,
                            producer_scope_id=None,
                            publication=LogicalRelation.point_map(
                                root_domains[0],
                                key_domain,
                                (
                                    (
                                        ((10, 0, 2, 1),),
                                        (logical_axis_symbol(10),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    uses=tuple(
                        EventUse(
                            consumer_root=root,
                            consumer_scope_id=None,
                            keys=LogicalRelation.point_map(
                                root_domains[root],
                                key_domain,
                                (
                                    (
                                        ((10 * (root + 1), 0, 2, 1),),
                                        (logical_axis_symbol(10 * (root + 1)),),
                                    ),
                                ),
                            ),
                            dependency_points=frozenset(((root - 1, None, None),)),
                        )
                        for root in (1, 2)
                    ),
                ),
            ),
        )
        (selected,) = choose_counted_events(
            event_graph,
            (),
            excluded_dependency_points=frozenset(((0, None, None),)),
        )

        self.assertEqual(selected.key_count, 2)
        self.assertEqual(tuple(use.consumer_root for use in selected.uses), (2,))

    def test_counted_event_lowering_is_derived_from_the_semantic_graph(self) -> None:
        root_domains = (
            LogicalDomain((10,), ((10, 4),), ((10, 1),)),
            LogicalDomain((20,), ((20, 2),), ((20, 2),)),
        )
        root_domains = _identify_root_domains(root_domains)
        key_domain = LogicalDomain((0,), ((0, 2),), kind="event", identity=0)
        event_graph = EventGraph(
            root_traversals=_default_root_traversals(root_domains),
            events=(
                KeyedEvent(
                    contributions=(
                        _event_contribution_from_publication(
                            producer_root=0,
                            producer_scope_id=None,
                            publication=LogicalRelation.point_map(
                                root_domains[0],
                                key_domain,
                                (
                                    (
                                        ((10, 0, 4, 1),),
                                        (sympy.floor(logical_axis_symbol(10) / 2),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    uses=(
                        EventUse(
                            consumer_root=1,
                            consumer_scope_id=None,
                            keys=LogicalRelation.point_map(
                                root_domains[1],
                                key_domain,
                                (
                                    (
                                        ((20, 0, 2, 1),),
                                        (logical_axis_symbol(20),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        baseline = build_baseline_worker_schedule(
            root_domains,
            worker_count=4,
        )
        triggers = derive_local_triggers(event_graph, baseline)

        (lowered,) = choose_counted_events(event_graph, triggers)

        self.assertEqual(
            _publication(lowered.contributions[0]).materialize(),
            (
                frozenset((0,)),
                frozenset((0,)),
                frozenset((1,)),
                frozenset((1,)),
            ),
        )
        self.assertEqual(
            lowered.uses[0].keys.materialize(),
            (frozenset((0,)), frozenset((1,))),
        )
        self.assertEqual(lowered.uniform_arrivals(), 2)
        self.assertEqual(lowered.local_trigger_use, 0)

    def test_nonstatic_layout_falls_back_to_root_readiness(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    block_ids=(10,),
                    layout_is_static=False,
                ),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        (event,) = _configured_event_graph(plan, _one_dimensional_domains()).events
        self.assertIsNotNone(event.family_done_root)
        self.assertEqual(event.family_done_root, 0)

    def test_fanout_keeps_one_edge_per_consumer(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store"),
                _access(1, root=1, kind="load", block_ids=(1,)),
                _access(2, root=2, kind="load", block_ids=(2,)),
            ),
            [[0], [1], [2]],
        )

        self.assertEqual(
            [(edge.producer_root, edge.consumer_root) for edge in plan.edges],
            [(0, 1), (0, 2)],
        )
        configured = _configured_event_graph(
            plan,
            (
                LogicalDomain((0,), ((0, 8),), ((0, 16),)),
                LogicalDomain((1,), ((1, 8),), ((1, 16),)),
                LogicalDomain((2,), ((2, 8),), ((2, 16),)),
            ),
        )
        (event,) = tuple(
            event
            for event in configured.events
            if any(
                contribution.producer_root == 0 for contribution in event.contributions
            )
        )
        self.assertIsNone(event.family_done_root)
        self.assertEqual(
            {use.consumer_root for use in event.uses},
            {1, 2},
        )

    def test_mixed_accesses_retain_exact_and_conservative_readiness(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store"),
                _access(1, root=1, allocation_id=0, kind="load", block_ids=(1,)),
                _access(2, root=0, allocation_id=1, kind="store"),
                _access(
                    3,
                    root=1,
                    allocation_id=1,
                    kind="load",
                    block_ids=(1,),
                    scales=(-1,),
                ),
            ),
            [[0], [1]],
        )

        self.assertEqual(len(plan.edges), 2)
        configured = _configured_event_graph(
            plan,
            (
                LogicalDomain((0,), ((0, 8),), ((0, 16),)),
                LogicalDomain((1,), ((1, 8),), ((1, 16),)),
            ),
        )
        self.assertEqual(len(configured.events), 2)
        family_event = next(
            event for event in configured.events if event.family_done_root is not None
        )
        self.assertEqual(family_event.family_done_root, 0)

    def test_mixed_exact_and_unknown_accesses_use_root_completion(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(1, 64),
                    strides=(64, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(1, 64),
                    strides=(64, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    2,
                    root=1,
                    kind="load",
                    shape=(1, 64),
                    strides=(64, 1),
                    block_ids=(None, None),
                    scales=(1, 1),
                    offsets=(None, None),
                    masked=True,
                ),
            ),
            [[10, 11], [20]],
        )
        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            root_domains=(
                LogicalDomain((10, 11), ((10, 1), (11, 4)), ((10, 1), (11, 16))),
                LogicalDomain((20,), ((20, 1),), ((20, 1),)),
            ),
            axis_geometry={10: (1, 1), 11: (4, 16), 20: (1, 1), 21: (4, 16)},
            worker_count=2,
        )

        self.assertEqual(schedule.root_completion_edges, frozenset(((0, 1),)))

    def test_singleton_producer_uses_root_completion(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            root_domains=(
                LogicalDomain((10,), ((10, 1),), ((10, 128),)),
                LogicalDomain((20,), ((20, 4),), ((20, 32),)),
            ),
            axis_geometry={10: (1, 128), 20: (4, 32)},
            worker_count=4,
        )

        self.assertEqual(schedule.root_completion_edges, frozenset(((0, 1),)))

    def test_root_completion_path_elides_redundant_exact_task_wait(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store", block_ids=(10,)),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    block_ids=(None,),
                    offsets=(None,),
                ),
                _access(2, root=1, allocation_id=1, kind="store", block_ids=(20,)),
                _access(
                    3,
                    root=2,
                    allocation_id=1,
                    kind="load",
                    block_ids=(None,),
                    offsets=(None,),
                ),
                _access(4, root=2, allocation_id=2, kind="store", block_ids=(30,)),
                _access(
                    5,
                    root=3,
                    allocation_id=2,
                    kind="load",
                    block_ids=(None,),
                    offsets=(None,),
                ),
                _access(6, root=3, allocation_id=0, kind="load", block_ids=(40,)),
            ),
            [[10], [20], [30], [40]],
        )
        root_domains = tuple(
            LogicalDomain(
                (10 + root * 10,), ((10 + root * 10, 8),), ((10 + root * 10, 16),)
            )
            for root in range(4)
        )

        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
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
            schedule.root_completion_edges,
            frozenset(((0, 1), (1, 2), (2, 3))),
        )

    def test_worker_schedule_derives_access_ready_overlap(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    2,
                    root=1,
                    allocation_id=1,
                    kind="store",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    3,
                    root=2,
                    allocation_id=1,
                    kind="load",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(30, 31),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
            ),
            [[10, 11], [20, 21], [30]],
        )
        dependency_plan = dataclasses.replace(
            dependency_plan,
            execution_scopes=(
                ExecutionScope(
                    0, 0, 0, (), None, "root", (10, 11), (10, 11), True, False
                ),
                ExecutionScope(
                    1, 1, 1, (), None, "root", (20, 21), (20, 21), True, False
                ),
                ExecutionScope(2, 2, 2, (), None, "root", (30,), (30,), True, False),
                ExecutionScope(
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
            scope_ids_by_access=((0,), (1,), (1,), (3,)),
        )
        root_domains = (
            LogicalDomain((10, 11), ((10, 1), (11, 8)), ((10, 1), (11, 16))),
            LogicalDomain((20, 21), ((20, 1), (21, 4)), ((20, 1), (21, 32))),
            LogicalDomain((30,), ((30, 1),), ((30, 1),)),
        )
        kwargs = {
            "dependency_plan": dependency_plan,
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

        schedule = build_cross_loop_schedule(**{**kwargs, "worker_count": 6})

        root_events = tuple(
            plan
            for plan in schedule.counted_events
            if all(use.consumer_scope_id is None for use in plan.uses)
        )
        self.assertEqual(len(root_events), 1)
        event = root_events[0]
        self.assertEqual(
            (
                event.contributions[0].producer_root,
                event.uses[0].consumer_root,
                event.local_use.consumer_root if event.local_use is not None else None,
                event.uniform_arrivals(),
            ),
            (0, 1, 1, 2),
        )
        local_events = tuple(
            plan for plan in schedule.counted_events if plan.local_use is not None
        )
        self.assertEqual(len(local_events), 1)
        self.assertEqual(local_events[0].local_trigger_use, 0)
        assert local_events[0].local_use is not None
        self.assertEqual(local_events[0].local_use.consumer_root, 1)
        self.assertEqual(schedule.worker_schedule.worker_count, 6)
        nested_scope_events = tuple(
            plan
            for plan in schedule.counted_events
            if any(use.consumer_scope_id is not None for use in plan.uses)
        )
        self.assertEqual(len(nested_scope_events), 1)
        self.assertEqual(
            _expected_arrivals(
                nested_scope_events[0].key_domain,
                nested_scope_events[0].contributions,
            ),
            (3, 1),
        )
        self.assertEqual(placement(schedule.worker_schedule, 2, 0), (5, 1))
        self.assertEqual(placement(schedule.worker_schedule, 0, 6), (0, 1))

        exact = build_cross_loop_schedule(**{**kwargs, "worker_count": 7})
        self.assertEqual(exact.worker_schedule.worker_count, 7)
        self.assertNotEqual(exact.worker_schedule, schedule.worker_schedule)

        default_schedule = build_cross_loop_schedule(**kwargs)
        self.assertEqual(len(default_schedule.counted_events), 2)
        self.assertEqual(
            default_schedule.root_completion_edges,
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
        short_schedule = build_cross_loop_schedule(
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
        self.assertEqual(len(short_schedule.counted_events), 2)
        self.assertEqual(
            short_schedule.root_completion_edges,
            frozenset(),
        )

    def test_nested_scope_milestones_follow_worker_readiness(self) -> None:
        root_domains = (
            LogicalDomain((10,), ((10, 4),), ((10, 1),)),
            LogicalDomain((20,), ((20, 1),), ((20, 1),)),
        )
        root_domains = _identify_root_domains(root_domains)
        action_domain = LogicalDomain(
            (20, 21),
            ((20, 1), (21, 4)),
            ((20, 1), (21, 1)),
            identity=7,
        )
        key_domain = LogicalDomain((0,), ((0, 4),), kind="event", identity=0)
        event_graph = EventGraph(
            root_traversals=_default_root_traversals(root_domains),
            events=(
                KeyedEvent(
                    contributions=(
                        _event_contribution_from_publication(
                            producer_root=0,
                            producer_scope_id=None,
                            publication=LogicalRelation.point_map(
                                root_domains[0],
                                key_domain,
                                (
                                    (
                                        ((10, 0, 4, 1),),
                                        (logical_axis_symbol(10),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    uses=(
                        EventUse(
                            consumer_root=1,
                            consumer_scope_id=7,
                            keys=LogicalRelation.point_map(
                                action_domain,
                                key_domain,
                                (
                                    (
                                        ((20, 0, 1, 1), (21, 0, 4, 1)),
                                        (logical_axis_symbol(21),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        schedule = WorkerSchedule(
            worker_count=4,
            segments=(
                WorkerScheduleSegment(
                    root=0,
                    task_relation=_one_dimensional_task_range(root_domains[0], 0, 3),
                    worker_begin=0,
                    worker_count=3,
                    schedule_begin=0,
                ),
                WorkerScheduleSegment(
                    root=0,
                    task_relation=_one_dimensional_task_range(root_domains[0], 3, 1),
                    worker_begin=0,
                    worker_count=1,
                    schedule_begin=1,
                ),
                WorkerScheduleSegment(
                    root=1,
                    task_relation=event_graph.root_traversals[1],
                    worker_begin=3,
                    worker_count=1,
                    schedule_begin=2,
                ),
            ),
        )

        placed, plans = place_nested_scope_consumers(event_graph, schedule, ())

        self.assertEqual(placement(placed, 1, 0), (3, 1))
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(
            _expected_arrivals(plan.key_domain, plan.contributions),
            (3, 1),
        )
        self.assertEqual(
            _publication(plan.contributions[0]).materialize(),
            (
                frozenset((0,)),
                frozenset((0,)),
                frozenset((0,)),
                frozenset((1,)),
            ),
        )
        self.assertEqual(
            plan.uses[0].keys.materialize(),
            (
                frozenset((0,)),
                frozenset((0,)),
                frozenset((0,)),
                frozenset((1,)),
            ),
        )
        self.assertEqual(plan.uses[0].consumer_scope_id, 7)

    def test_nested_scope_entry_event_survives_without_early_placement(self) -> None:
        producer_domain = LogicalDomain(
            (10, 11),
            ((10, 2), (11, 4)),
            ((10, 1), (11, 1)),
            identity=0,
        )
        consumer_domain = LogicalDomain(
            (20,),
            ((20, 2),),
            ((20, 1),),
            identity=1,
        )
        action_domain = LogicalDomain(
            (20, 21),
            ((20, 2), (21, 4)),
            ((20, 1), (21, 1)),
            identity=7,
        )
        key_domain = LogicalDomain(
            (0, 1),
            ((0, 2), (1, 4)),
            kind="event",
            identity=0,
        )
        event_graph = EventGraph(
            root_traversals=(
                physical_traversal_relation(producer_domain, (10, 11)),
                physical_traversal_relation(consumer_domain, (20,)),
            ),
            events=(
                KeyedEvent(
                    contributions=(
                        _event_contribution_from_publication(
                            producer_root=0,
                            publication=LogicalRelation.point_map(
                                producer_domain,
                                key_domain,
                                (
                                    (
                                        (
                                            (10, 0, 2, 1),
                                            (11, 0, 4, 1),
                                        ),
                                        (
                                            logical_axis_symbol(10),
                                            logical_axis_symbol(11),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    uses=(
                        EventUse(
                            consumer_root=1,
                            consumer_scope_id=7,
                            keys=LogicalRelation.point_map(
                                action_domain,
                                key_domain,
                                (
                                    (
                                        (
                                            (20, 0, 2, 1),
                                            (21, 0, 4, 1),
                                        ),
                                        (
                                            logical_axis_symbol(20),
                                            logical_axis_symbol(21),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        for worker_count in (1, 2):
            with self.subTest(worker_count=worker_count):
                schedule = _build_baseline_worker_schedule(
                    event_graph.root_domains,
                    event_graph.root_traversals,
                    worker_count=worker_count,
                )

                placed, plans = place_nested_scope_consumers(
                    event_graph,
                    schedule,
                    (),
                )

                self.assertEqual(placed, schedule)
                self.assertEqual(len(plans), 1)
                self.assertEqual(plans[0].key_count, 2)
                self.assertEqual(
                    _expected_arrivals(
                        plans[0].key_domain,
                        plans[0].contributions,
                    ),
                    (4, 4),
                )
                self.assertEqual(plans[0].uses[0].consumer_scope_id, 7)

    def test_nested_scope_identity_readiness_uses_one_admission_frontier(self) -> None:
        """Per-iteration readiness is coarsened to one compact frontier."""
        root_domains = (
            LogicalDomain((10,), ((10, 4),), ((10, 1),)),
            LogicalDomain((20,), ((20, 1),), ((20, 1),)),
        )
        root_domains = _identify_root_domains(root_domains)
        action_domain = LogicalDomain(
            (20, 21),
            ((20, 1), (21, 4)),
            ((20, 1), (21, 1)),
            identity=7,
        )
        key_domain = LogicalDomain((0,), ((0, 4),), kind="event", identity=0)
        event_graph = EventGraph(
            root_traversals=_default_root_traversals(root_domains),
            events=(
                KeyedEvent(
                    contributions=(
                        _event_contribution_from_publication(
                            producer_root=0,
                            publication=LogicalRelation.point_map(
                                root_domains[0],
                                key_domain,
                                (
                                    (
                                        ((10, 0, 4, 1),),
                                        (logical_axis_symbol(10),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    uses=(
                        EventUse(
                            consumer_root=1,
                            consumer_scope_id=7,
                            keys=LogicalRelation.point_map(
                                action_domain,
                                key_domain,
                                (
                                    (
                                        ((20, 0, 1, 1), (21, 0, 4, 1)),
                                        (logical_axis_symbol(21),),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        schedule = WorkerSchedule(
            worker_count=4,
            segments=(
                WorkerScheduleSegment(
                    root=0,
                    task_relation=event_graph.root_traversals[0],
                    worker_begin=0,
                    worker_count=1,
                    schedule_begin=0,
                ),
                WorkerScheduleSegment(
                    root=1,
                    task_relation=event_graph.root_traversals[1],
                    worker_begin=3,
                    worker_count=1,
                    schedule_begin=5,
                ),
            ),
        )

        placed, plans = place_nested_scope_consumers(event_graph, schedule, ())

        self.assertEqual(placement(placed, 1, 0), (3, 1))
        self.assertEqual(len(plans), 1)
        self.assertEqual(
            _expected_arrivals(plans[0].key_domain, plans[0].contributions),
            (1, 3),
        )

    def test_nested_scope_placement_keeps_transitive_worker_liveness(
        self,
    ) -> None:
        """A moved wait must not block an upstream prerequisite on its worker."""
        root_domains = (
            LogicalDomain((10,), ((10, 4),), ((10, 1),)),
            LogicalDomain((20,), ((20, 4),), ((20, 1),)),
            LogicalDomain((30,), ((30, 1),), ((30, 1),)),
        )
        root_domains = _identify_root_domains(root_domains)
        action_domain = LogicalDomain(
            (30, 31),
            ((30, 1), (31, 4)),
            ((30, 1), (31, 1)),
            identity=7,
        )

        def identity_keys(
            source_domain: LogicalDomain,
            source_axis: int,
            event_id: int,
        ) -> tuple[LogicalDomain, LogicalRelation]:
            key_domain = LogicalDomain(
                (0,),
                ((0, 4),),
                kind="event",
                identity=event_id,
            )
            return key_domain, LogicalRelation.point_map(
                source_domain,
                key_domain,
                (
                    (
                        tuple(
                            (axis, 0, source_domain.axis_counts[axis], 1)
                            for axis in source_domain.axis_order
                        ),
                        (logical_axis_symbol(source_axis),),
                    ),
                ),
            )

        first_keys, a_to_first = identity_keys(root_domains[0], 10, 0)
        _, first_use = identity_keys(root_domains[1], 20, 0)
        second_keys, b_to_second = identity_keys(root_domains[1], 20, 1)
        _, nested_use = identity_keys(action_domain, 31, 1)
        event_graph = EventGraph(
            root_traversals=_default_root_traversals(root_domains),
            events=(
                KeyedEvent(
                    contributions=(
                        _event_contribution_from_publication(0, a_to_first),
                    ),
                    uses=(EventUse(1, first_use),),
                ),
                KeyedEvent(
                    contributions=(
                        _event_contribution_from_publication(1, b_to_second),
                    ),
                    uses=(EventUse(2, nested_use, consumer_scope_id=7),),
                ),
            ),
        )

        def task_segment(
            root: int,
            task_begin: int,
            task_count: int,
            worker: int,
            position: int,
        ) -> WorkerScheduleSegment:
            return WorkerScheduleSegment(
                root=root,
                task_relation=_one_dimensional_task_range(
                    root_domains[root], task_begin, task_count
                ),
                worker_begin=worker,
                worker_count=task_count,
                schedule_begin=position * task_count,
            )

        schedule = WorkerSchedule(
            worker_count=4,
            segments=(
                task_segment(0, 0, 3, 0, 0),
                task_segment(0, 3, 1, 3, 3),
                task_segment(1, 0, 3, 0, 1),
                task_segment(1, 3, 1, 0, 4),
                task_segment(2, 0, 1, 3, 6),
            ),
        )

        placed, plans = place_nested_scope_consumers(event_graph, schedule, ())

        # Worker 3 looks idle at position 2, but its A task at position 3 is a
        # prerequisite of B task 3. Placing C there would form C -> B -> A
        # while A remains later on C's blocked worker. Worker 2 is safe.
        self.assertEqual(placement(placed, 2, 0), (2, 2))
        self.assertEqual(len(plans), 1)
        validate_worker_schedule(event_graph, placed)

    def test_nested_scope_placement_preserves_source_order_on_each_worker(
        self,
    ) -> None:
        root_domains = _identify_root_domains(
            (
                LogicalDomain((10,), ((10, 5),), ((10, 1),)),
                LogicalDomain((20,), ((20, 4),), ((20, 1),)),
                LogicalDomain((30,), ((30, 1),), ((30, 1),)),
            )
        )
        action_domain = LogicalDomain(
            (30, 31),
            ((30, 1), (31, 5)),
            ((30, 1), (31, 1)),
            identity=7,
        )
        nested_key_domain = LogicalDomain(
            (0,),
            ((0, 5),),
            kind="event",
            identity=0,
        )
        producer_to_key = LogicalRelation.point_map(
            root_domains[0],
            nested_key_domain,
            (
                (
                    ((10, 0, 5, 1),),
                    (logical_axis_symbol(10),),
                ),
            ),
        )
        predecessors = producer_to_key.inverse()
        assert predecessors is not None
        action_to_key = LogicalRelation.point_map(
            action_domain,
            nested_key_domain,
            (
                (
                    ((30, 0, 1, 1), (31, 0, 5, 1)),
                    (logical_axis_symbol(31),),
                ),
            ),
        )
        family_done_domain = LogicalDomain(
            (),
            (),
            kind="event",
            identity=1,
        )
        event_graph = EventGraph(
            root_traversals=_default_root_traversals(root_domains),
            events=(
                KeyedEvent(
                    contributions=(EventContribution(0, predecessors),),
                    uses=(EventUse(2, action_to_key, consumer_scope_id=7),),
                ),
                KeyedEvent(
                    contributions=(
                        EventContribution(
                            1,
                            LogicalRelation.total(
                                family_done_domain,
                                root_domains[1],
                            ),
                        ),
                    ),
                    uses=(
                        EventUse(
                            2,
                            LogicalRelation.total(
                                root_domains[2],
                                family_done_domain,
                            ),
                        ),
                    ),
                ),
            ),
        )
        baseline = _build_baseline_worker_schedule(
            event_graph.root_domains,
            event_graph.root_traversals,
            worker_count=4,
        )

        placed, plans = place_nested_scope_consumers(event_graph, baseline, ())

        self.assertEqual(placement(baseline, 2, 0), (0, 3))
        self.assertEqual(placement(placed, 2, 0), (0, 3))
        self.assertEqual(len(plans), 1)
        validate_worker_schedule(event_graph, placed)

    def test_nested_scope_milestones_compose_sibling_scopes(self) -> None:
        root_domains = (
            LogicalDomain((10,), ((10, 4),), ((10, 1),)),
            LogicalDomain((20,), ((20, 4),), ((20, 1),)),
            LogicalDomain((30,), ((30, 1),), ((30, 1),)),
        )
        root_domains = _identify_root_domains(root_domains)
        action_domains = tuple(
            LogicalDomain(
                (30, nested_axis),
                ((30, 1), (nested_axis, 4)),
                ((30, 1), (nested_axis, 1)),
                identity=scope_id,
            )
            for scope_id, nested_axis in ((7, 31), (8, 32))
        )
        scope_domains: tuple[LogicalDomain | None, ...] = (
            *(None for _ in range(7)),
            *action_domains,
        )
        events = []
        for producer_root, scope_id, nested_axis in ((0, 7, 31), (1, 8, 32)):
            key_domain = LogicalDomain(
                (0,),
                ((0, 4),),
                kind="event",
                identity=producer_root,
            )
            events.append(
                KeyedEvent(
                    contributions=(
                        _event_contribution_from_publication(
                            producer_root=producer_root,
                            producer_scope_id=None,
                            publication=LogicalRelation.point_map(
                                root_domains[producer_root],
                                key_domain,
                                (
                                    (
                                        (
                                            (
                                                root_domains[producer_root].axis_order[
                                                    0
                                                ],
                                                0,
                                                4,
                                                1,
                                            ),
                                        ),
                                        (
                                            logical_axis_symbol(
                                                root_domains[producer_root].axis_order[
                                                    0
                                                ]
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    uses=(
                        EventUse(
                            consumer_root=2,
                            consumer_scope_id=scope_id,
                            keys=LogicalRelation.point_map(
                                scope_domains[scope_id],
                                key_domain,
                                (
                                    (
                                        ((30, 0, 1, 1), (nested_axis, 0, 4, 1)),
                                        (logical_axis_symbol(nested_axis),),
                                    ),
                                ),
                            ),
                            dependency_points=frozenset(
                                ((producer_root, None, scope_id),)
                            ),
                        ),
                    ),
                )
            )
        event_graph = EventGraph(
            root_traversals=_default_root_traversals(root_domains),
            events=tuple(events),
        )
        schedule = WorkerSchedule(
            worker_count=4,
            segments=(
                WorkerScheduleSegment(
                    root=0,
                    task_relation=_one_dimensional_task_range(root_domains[0], 0, 4),
                    worker_begin=0,
                    worker_count=4,
                    schedule_begin=0,
                ),
                WorkerScheduleSegment(
                    root=1,
                    task_relation=_one_dimensional_task_range(root_domains[1], 0, 3),
                    worker_begin=0,
                    worker_count=4,
                    schedule_begin=4,
                ),
                WorkerScheduleSegment(
                    root=1,
                    task_relation=_one_dimensional_task_range(root_domains[1], 3, 1),
                    worker_begin=0,
                    worker_count=4,
                    schedule_begin=8,
                ),
                WorkerScheduleSegment(
                    root=2,
                    task_relation=event_graph.root_traversals[2],
                    worker_begin=3,
                    worker_count=1,
                    schedule_begin=3,
                ),
            ),
        )

        placed, plans = place_nested_scope_consumers(event_graph, schedule, ())

        self.assertEqual(placement(placed, 2, 0), (3, 1))
        self.assertEqual(len(plans), 2)
        plans_by_scope = {plan.uses[0].consumer_scope_id: plan for plan in plans}
        self.assertEqual(
            _expected_arrivals(
                plans_by_scope[7].key_domain,
                plans_by_scope[7].contributions,
            ),
            (4,),
        )
        self.assertEqual(
            plans_by_scope[7].uses[0].keys.materialize(),
            (frozenset((0,)),) * 4,
        )
        self.assertEqual(
            _expected_arrivals(
                plans_by_scope[8].key_domain,
                plans_by_scope[8].contributions,
            ),
            (4,),
        )
        self.assertEqual(
            plans_by_scope[8].uses[0].keys.materialize(),
            (
                frozenset((0,)),
                frozenset((0,)),
                frozenset((0,)),
                frozenset((0,)),
            ),
        )

    def test_multi_producer_join_uses_one_keyed_event(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(0, root=0, allocation_id=0, kind="store", block_ids=(10,)),
                _access(1, root=1, allocation_id=1, kind="store", block_ids=(20,)),
                _access(2, root=2, allocation_id=0, kind="load", block_ids=(30,)),
                _access(3, root=2, allocation_id=1, kind="load", block_ids=(30,)),
            ),
            [[10], [20], [30]],
        )
        root_domains = tuple(
            LogicalDomain((block_id,), ((block_id, 8),), ((block_id, 16),))
            for root, block_id in enumerate((10, 20, 30))
        )

        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            root_domains=root_domains,
            axis_geometry={10: (8, 16), 20: (8, 16), 30: (8, 16)},
            worker_count=8,
        )

        self.assertEqual(schedule.root_completion_edges, frozenset())
        self.assertEqual(len(schedule.counted_events), 1)
        event = schedule.counted_events[0]
        self.assertEqual(event.uses[0].consumer_root, 2)
        self.assertEqual(event.uniform_arrivals(), 2)
        self.assertEqual(
            [
                (
                    contributor.producer_root,
                    contributor.arrivals_per_key.constant_value()
                    if contributor.arrivals_per_key is not None
                    else None,
                )
                for contributor in event.contributions
            ],
            [(0, 1), (1, 1)],
        )

    def test_repeated_join_predecessors_coalesce_consumer_tasks(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(32,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=1,
                    kind="store",
                    shape=(8,),
                    block_ids=(30,),
                ),
                _access(
                    2,
                    root=2,
                    allocation_id=0,
                    kind="load",
                    shape=(8, 4),
                    strides=(4, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    3,
                    root=2,
                    allocation_id=1,
                    kind="load",
                    shape=(8,),
                    block_ids=(20,),
                ),
            ),
            [[10], [30], [22, 20, 21]],
        )
        root_domains = (
            LogicalDomain((10,), ((10, 32),), ((10, 1),)),
            LogicalDomain((30,), ((30, 8),), ((30, 1),)),
            LogicalDomain(
                (22, 20, 21), ((22, 4), (20, 8), (21, 1)), ((22, 1), (20, 1), (21, 4))
            ),
        )

        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
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

        self.assertEqual(schedule.root_completion_edges, frozenset())
        self.assertEqual(len(schedule.counted_events), 1)
        self.assertFalse(
            any(event.local_use is not None for event in schedule.counted_events)
        )
        event = schedule.counted_events[0]
        self.assertIsNone(event.local_use)
        self.assertEqual(event.key_count, 8)
        self.assertEqual(event.uniform_arrivals(), 5)
        self.assertEqual(
            event.uses[0].keys.materialize(),
            tuple(frozenset((i // 4,)) for i in range(32)),
        )
        self.assertEqual(
            [
                contributor.arrivals_per_key.constant_value()
                if contributor.arrivals_per_key is not None
                else None
                for contributor in event.contributions
            ],
            [4, 1],
        )

    def test_large_flattened_ready_groups_have_no_task_product_cutoff(self) -> None:
        heads = 513
        width = 4
        splits = 4
        elements = heads * width
        dependency_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(elements,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(heads, width),
                    strides=(width, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
            ),
            [[10], [22, 20, 21]],
        )
        root_domains = (
            LogicalDomain((10,), ((10, elements),), ((10, 1),)),
            LogicalDomain(
                (22, 20, 21),
                ((22, splits), (20, heads), (21, 1)),
                ((22, 1), (20, 1), (21, width)),
            ),
        )

        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            root_domains=root_domains,
            axis_geometry={
                10: (elements, 1),
                20: (heads, 1),
                21: (1, width),
                22: (splits, 1),
            },
            worker_count=128,
        )

        self.assertEqual(schedule.root_completion_edges, frozenset())
        self.assertEqual(len(schedule.counted_events), 1)
        event = schedule.counted_events[0]
        self.assertEqual(event.key_count, heads)
        self.assertEqual(event.uniform_arrivals(), width)
        self.assertEqual(
            event.uses[0].keys.materialize(),
            tuple(frozenset((task // splits,)) for task in range(heads * splits)),
        )
        self.assertEqual(len(event.contributions[0].predecessors.pieces), 1)
        self.assertEqual(len(event.uses[0].keys.pieces), 1)

    def test_strided_ready_groups_use_exact_coordinates_with_overlapping_hulls(
        self,
    ) -> None:
        columns = 8
        splits = 4
        dependency_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(2, columns),
                    strides=(columns, 1),
                    block_ids=(None, 10),
                    scales=(1, 1),
                    offsets=(0, 0),
                    full_slice=(True, False),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    shape=(2, columns),
                    strides=(columns, 1),
                    block_ids=(None, 20),
                    scales=(1, 1),
                    offsets=(0, 0),
                    full_slice=(True, False),
                ),
            ),
            [[10], [22, 20]],
        )
        root_domains = (
            LogicalDomain((10,), ((10, columns),), ((10, 1),)),
            LogicalDomain((22, 20), ((22, splits), (20, columns)), ((22, 1), (20, 1))),
        )

        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
            root_domains=root_domains,
            axis_geometry={10: (columns, 1), 20: (columns, 1), 22: (splits, 1)},
            worker_count=32,
        )

        self.assertEqual(schedule.root_completion_edges, frozenset())
        self.assertEqual(len(schedule.counted_events), 1)
        event = schedule.counted_events[0]
        self.assertEqual(event.key_count, columns)
        self.assertEqual(event.uniform_arrivals(), 1)
        self.assertEqual(
            event.uses[0].keys.materialize(),
            tuple(frozenset((task // splits,)) for task in range(columns * splits)),
        )

    def test_multiple_access_events_in_one_root_fall_back_together(self) -> None:
        dependency_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(10, 11),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=1,
                    kind="store",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(20, 21),
                ),
                _access(
                    2,
                    root=2,
                    allocation_id=0,
                    kind="load",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(30, 31),
                ),
                _access(
                    3,
                    root=2,
                    allocation_id=1,
                    kind="load",
                    shape=(1, 128),
                    strides=(128, 1),
                    block_ids=(30, 32),
                ),
            ),
            [[10, 11], [20, 21], [30]],
        )
        root_domains = (
            LogicalDomain((10, 11), ((10, 1), (11, 8)), ((10, 1), (11, 16))),
            LogicalDomain((20, 21), ((20, 1), (21, 8)), ((20, 1), (21, 16))),
            LogicalDomain((30,), ((30, 1),), ((30, 1),)),
        )

        schedule = build_cross_loop_schedule(
            dependency_plan=dependency_plan,
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
            schedule.root_completion_edges,
            frozenset(((0, 2), (1, 2))),
        )

    def test_worker_schedule_handles_independent_components(self) -> None:
        accesses: list[TileAccess] = []
        root_domains: list[LogicalDomain] = []
        axis_geometry: dict[int, tuple[int, int]] = {}
        for component in range(2):
            root_base = component * 3
            block_base = 10 + component * 30
            access_base = component * 4
            allocation_base = component * 2
            accesses.extend(
                (
                    _access(
                        access_base,
                        root=root_base,
                        allocation_id=allocation_base,
                        kind="store",
                        shape=(1, 128),
                        strides=(128, 1),
                        block_ids=(block_base, block_base + 1),
                        scales=(1, 1),
                        offsets=(0, 0),
                    ),
                    _access(
                        access_base + 1,
                        root=root_base + 1,
                        allocation_id=allocation_base,
                        kind="load",
                        shape=(1, 128),
                        strides=(128, 1),
                        block_ids=(block_base + 10, block_base + 11),
                        scales=(1, 1),
                        offsets=(0, 0),
                    ),
                    _access(
                        access_base + 2,
                        root=root_base + 1,
                        allocation_id=allocation_base + 1,
                        kind="store",
                        shape=(1, 128),
                        strides=(128, 1),
                        block_ids=(block_base + 10, block_base + 11),
                        scales=(1, 1),
                        offsets=(0, 0),
                    ),
                    _access(
                        access_base + 3,
                        root=root_base + 2,
                        allocation_id=allocation_base + 1,
                        kind="load",
                        shape=(1, 128),
                        strides=(128, 1),
                        block_ids=(block_base + 20, block_base + 21),
                        scales=(1, 1),
                        offsets=(0, 0),
                    ),
                )
            )
            root_domains.extend(
                (
                    LogicalDomain(
                        (block_base, block_base + 1),
                        ((block_base, 1), (block_base + 1, 8)),
                        ((block_base, 1), (block_base + 1, 16)),
                    ),
                    LogicalDomain(
                        (block_base + 10, block_base + 11),
                        (
                            (block_base + 10, 1),
                            (block_base + 11, 4),
                        ),
                        (
                            (block_base + 10, 1),
                            (block_base + 11, 32),
                        ),
                    ),
                    LogicalDomain(
                        (block_base + 20,),
                        ((block_base + 20, 1),),
                        ((block_base + 20, 1),),
                    ),
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

        dependency_plan = build_tile_dependency_graph(
            tuple(accesses),
            [list(domain.axis_order) for domain in root_domains],
        )
        root_scopes = tuple(
            ExecutionScope(
                scope_id=root,
                root=root,
                graph_id=root,
                callsite_path=(),
                parent_scope_id=None,
                kind="root",
                local_axis_order=domain.axis_order,
                logical_axis_order=domain.axis_order,
                guaranteed=True,
                segmentable=False,
            )
            for root, domain in enumerate(root_domains)
        )
        nested_scopes = tuple(
            ExecutionScope(
                scope_id=6 + component,
                root=component * 3 + 2,
                graph_id=6 + component,
                callsite_path=((0, 0),),
                parent_scope_id=component * 3 + 2,
                kind="loop",
                local_axis_order=(10 + component * 30 + 21,),
                logical_axis_order=(
                    10 + component * 30 + 20,
                    10 + component * 30 + 21,
                ),
                guaranteed=True,
                segmentable=True,
            )
            for component in range(2)
        )
        scope_ids_by_access: list[tuple[int, ...]] = [()] * len(accesses)
        for component in range(2):
            root_base = component * 3
            access_base = component * 4
            scope_ids_by_access[access_base] = (root_base,)
            scope_ids_by_access[access_base + 1] = (root_base + 1,)
            scope_ids_by_access[access_base + 2] = (root_base + 1,)
            scope_ids_by_access[access_base + 3] = (6 + component,)
        dependency_plan = dataclasses.replace(
            dependency_plan,
            execution_scopes=(*root_scopes, *nested_scopes),
            scope_ids_by_access=tuple(scope_ids_by_access),
        )
        kwargs = {
            "dependency_plan": dependency_plan,
            "root_domains": tuple(root_domains),
            "axis_geometry": axis_geometry,
            "worker_count": 8,
        }

        schedule = build_cross_loop_schedule(**kwargs)
        self.assertEqual(len(schedule.counted_events), 4)
        self.assertEqual(
            schedule.root_completion_edges,
            frozenset(),
        )
        overlapped = build_cross_loop_schedule(**{**kwargs, "worker_count": 6})
        self.assertEqual(overlapped.worker_schedule.worker_count, 6)
        nested_scope_events = tuple(
            plan
            for plan in overlapped.counted_events
            if any(use.consumer_scope_id is not None for use in plan.uses)
        )
        self.assertEqual(
            [
                (
                    plan.contributions[0].producer_root,
                    plan.uses[0].consumer_root,
                    _expected_arrivals(plan.key_domain, plan.contributions),
                )
                for plan in nested_scope_events
            ],
            [(1, 2, (3, 1)), (4, 5, (3, 1))],
        )
        self.assertEqual(overlapped.root_completion_edges, frozenset())
        self.assertEqual(placement(overlapped.worker_schedule, 2, 0), (5, 1))
        self.assertEqual(placement(overlapped.worker_schedule, 5, 0), (5, 5))
