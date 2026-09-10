from __future__ import annotations

import copy
import dataclasses
import pickle
from unittest import mock

import sympy

from helion._compiler import cross_loop_scheduler
from helion._compiler.cross_loop_scheduler import RootBarrierPublication
from helion._compiler.cross_loop_scheduler import RootBarrierPublicationPlan
from helion._compiler.cross_loop_scheduler import WorkerSchedule
from helion._compiler.cross_loop_scheduler import WorkerScheduleSegment
from helion._compiler.tile_dependency import CoordinateDomain
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import _memoized_exact_converse
from helion._compiler.tile_dependency import coordinate_axis_symbol
from helion._compiler.tile_dependency import pid_task_order
from helion._testing import TestCase


def _dynamic_task_orders(
    extent: sympy.Symbol,
) -> tuple[tuple[str, CoordinateDomain, CoordinateRelation], ...]:
    canonical_domain = CoordinateDomain(
        (10, 11),
        ((10, 2), (11, extent)),
        identity=0,
    )
    canonical = pid_task_order(canonical_domain, canonical_domain.axis_order)

    permuted_domain = CoordinateDomain(
        (20, 21, 22),
        ((20, 2), (21, 2), (22, extent)),
        identity=0,
    )
    permuted = pid_task_order(permuted_domain, (21, 20, 22))

    reflected_domain = CoordinateDomain(
        (30, 31),
        ((30, 4), (31, extent)),
        identity=0,
    )
    reflected_source = CoordinateDomain(
        (29, 31),
        ((29, 4), (31, extent)),
        kind="task_order",
        identity=0,
    )
    reflected = CoordinateRelation.point_map(
        reflected_source,
        reflected_domain,
        (
            (
                ((29, 0, 4, 1), (31, 0, extent, 1)),
                (
                    3 - coordinate_axis_symbol(29),
                    coordinate_axis_symbol(31),
                ),
            ),
        ),
    )

    woven_domain = CoordinateDomain(
        (40, 41, 42),
        ((40, 2), (41, 4), (42, extent)),
        identity=0,
    )
    woven_source = CoordinateDomain(
        (39, 38, 42),
        ((39, 4), (38, 2), (42, extent)),
        kind="task_order",
        identity=0,
    )
    inner = coordinate_axis_symbol(39)
    woven = CoordinateRelation.point_map(
        woven_source,
        woven_domain,
        (
            (
                ((39, 0, 4, 1), (38, 0, 2, 1), (42, 0, extent, 1)),
                (
                    sympy.Mod(inner, 2),
                    2 * coordinate_axis_symbol(38) + sympy.floor(inner / 2),
                    coordinate_axis_symbol(42),
                ),
            ),
        ),
    )

    l2_domain = CoordinateDomain(
        (50, 51, 52),
        ((50, 4), (51, 3), (52, extent)),
        identity=0,
    )
    l2 = pid_task_order(
        l2_domain,
        l2_domain.axis_order,
        l2_group_size=2,
    )
    return (
        ("canonical", canonical_domain, canonical),
        ("permuted", permuted_domain, permuted),
        ("reflected", reflected_domain, reflected),
        ("woven", woven_domain, woven),
        ("l2", l2_domain, l2),
    )


class TestExactConverseSemantics(TestCase):
    def test_composition_and_union_retain_only_existing_exact_converses(
        self,
    ) -> None:
        source = CoordinateDomain((10,), ((10, 4),), kind="worker")
        ordinal = CoordinateDomain((20,), ((20, 4),), kind="task_order")
        target = CoordinateDomain((30,), ((30, 4),), identity=0)
        source_coordinate = coordinate_axis_symbol(10)
        ordinal_coordinate = coordinate_axis_symbol(20)
        first = CoordinateRelation.point_map(
            source,
            ordinal,
            ((((10, 0, 4, 1),), (source_coordinate,)),),
        )
        following = CoordinateRelation.point_map(
            ordinal,
            target,
            ((((20, 0, 4, 1),), (3 - ordinal_coordinate,)),),
        )

        with mock.patch.object(
            CoordinateRelation,
            "converse",
            side_effect=AssertionError("composition must not initiate a proof"),
        ):
            unproved = first.then(following)
        self.assertIsNotNone(unproved)
        assert unproved is not None
        self.assertIsNone(_memoized_exact_converse(unproved))

        self.assertIsNotNone(first.converse())
        self.assertIsNotNone(following.converse())
        with mock.patch.object(
            CoordinateRelation,
            "converse",
            side_effect=AssertionError("composition must use retained proofs"),
        ):
            composed = first.then(following)
        self.assertIsNotNone(composed)
        assert composed is not None
        composed_converse = _memoized_exact_converse(composed)
        self.assertIsNotNone(composed_converse)
        assert composed_converse is not None
        self.assertEqual(
            composed_converse.materialize(),
            (frozenset((3,)), frozenset((2,)), frozenset((1,)), frozenset((0,))),
        )

        left = CoordinateRelation.point_map(
            source,
            target,
            ((((10, 0, 2, 1),), (source_coordinate,)),),
        )
        right = CoordinateRelation.point_map(
            source,
            target,
            ((((10, 2, 4, 1),), (source_coordinate,)),),
        )
        self.assertIsNotNone(left.converse())
        self.assertIsNotNone(right.converse())
        with mock.patch.object(
            CoordinateRelation,
            "converse",
            side_effect=AssertionError("union must use retained proofs"),
        ):
            combined = left.union(right)
        self.assertIsNotNone(combined)
        assert combined is not None
        self.assertIsNotNone(_memoized_exact_converse(combined))
        self.assertTrue(combined.is_bijection_from_source_support())

        # Derived proof state is not part of equality and is not copied by a
        # value transformation whose new forward relation has not been proved.
        transformed = dataclasses.replace(composed)
        self.assertEqual(transformed, composed)
        self.assertEqual(hash(transformed), hash(composed))
        self.assertIsNone(_memoized_exact_converse(transformed))
        for rebuilt in (copy.deepcopy(composed), pickle.loads(pickle.dumps(composed))):
            self.assertEqual(rebuilt, composed)
            self.assertEqual(hash(rebuilt), hash(composed))
            self.assertIsNone(_memoized_exact_converse(rebuilt))

    def test_manual_relation_without_construction_witness_uses_bounded_fallback(
        self,
    ) -> None:
        source = CoordinateDomain((10,), ((10, 6),), kind="task_order")
        target = CoordinateDomain((20,), ((20, 6),), identity=0)
        permutation = (2, 5, 1, 4, 0, 3)
        derived = CoordinateRelation.point_map(
            source,
            target,
            tuple(
                (
                    ((10, source_index, source_index + 1, 1),),
                    (sympy.Integer(target_index),),
                )
                for source_index, target_index in enumerate(permutation)
            ),
        )

        # Reconstruct from semantic fields so no constructor provenance or
        # propagated converse can be required for acceptance.
        manual = CoordinateRelation(
            source_domain=derived.source_domain,
            target_domain=derived.target_domain,
            pieces=derived.pieces,
        )
        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("the bounded proof must stay symbolic"),
        ):
            self.assertTrue(manual.is_bijection_from_source_support())
            converse = manual.converse()
            self.assertIsNotNone(converse)
            assert converse is not None
            self.assertTrue(converse.is_total_function())

        self.assertEqual(
            converse.materialize(),
            tuple(
                frozenset((permutation.index(target_index),))
                for target_index in range(len(permutation))
            ),
        )

    def test_packed_composition_preserves_supported_dynamic_task_orders(
        self,
    ) -> None:
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)
        worker_count = 2

        for name, domain, task_order in _dynamic_task_orders(extent):
            with self.subTest(order=name):
                with mock.patch.object(
                    CoordinateRelation,
                    "materialize",
                    side_effect=AssertionError(
                        "symbolic task-order proof must not enumerate"
                    ),
                ):
                    self.assertTrue(task_order.is_bijection_from_source_support())
                    task_order_converse = task_order.converse()
                    self.assertIsNotNone(task_order_converse)
                    assert task_order_converse is not None
                    self.assertTrue(task_order_converse.is_total_function())
                    schedule = cross_loop_scheduler._build_root_major_worker_schedule(
                        (domain,),
                        (task_order,),
                        worker_count,
                    )
                    relation = schedule.segments[0].task_order
                    self.assertTrue(relation.is_bijection_from_source_support())
                    converse = relation.converse()
                    self.assertIsNotNone(converse)
                    assert converse is not None
                    self.assertTrue(converse.is_total_function())

                for concrete_extent in (0, 1, worker_count, worker_count + 1):
                    concrete_order = task_order.substitute_parameters(
                        {extent: concrete_extent}
                    )
                    concrete_relation = relation.substitute_parameters(
                        {extent: concrete_extent}
                    )
                    self.assertTrue(
                        concrete_relation.is_bijection_from_source_support(),
                        (name, concrete_extent),
                    )
                    concrete_converse = concrete_relation.converse()
                    self.assertIsNotNone(
                        concrete_converse,
                        (name, concrete_extent),
                    )
                    assert concrete_converse is not None
                    self.assertTrue(concrete_converse.is_total_function())

                    expected = tuple(
                        tuple(
                            concrete_order.target_domain.coordinates(
                                next(iter(targets))
                            )[axis]
                            for axis in concrete_order.target_domain.axis_order
                        )
                        for targets in concrete_order.materialize()
                    )
                    launch_axis, worker_axis, wave_axis = (
                        concrete_relation.source_domain.axis_order
                    )
                    actual = []
                    for ordinal in range(concrete_order.source_domain.size):
                        targets = concrete_relation.target_coordinates(
                            {
                                launch_axis: 1,
                                worker_axis: ordinal % worker_count,
                                wave_axis: ordinal // worker_count,
                            }
                        )
                        self.assertEqual(len(targets), 1)
                        actual.append(next(iter(targets)))
                    self.assertEqual(tuple(actual), expected)

    def test_multisegment_root_requires_exact_union_ownership(self) -> None:
        target = CoordinateDomain((10,), ((10, 6),), identity=0)
        task_order = pid_task_order(target, target.axis_order)
        prefix = cross_loop_scheduler._task_order_slice(task_order, 0, 4)
        suffix = cross_loop_scheduler._task_order_slice(task_order, 4, 2)
        duplicate = cross_loop_scheduler._task_order_slice(task_order, 2, 2)
        self.assertIsNotNone(prefix)
        self.assertIsNotNone(suffix)
        self.assertIsNotNone(duplicate)
        assert prefix is not None and suffix is not None and duplicate is not None

        valid = WorkerSchedule(
            6,
            (
                WorkerScheduleSegment(0, prefix, 0, 4, 0),
                WorkerScheduleSegment(0, suffix, 4, 2, 0),
            ),
        )
        first, second = (segment.task_order for segment in valid.segments)
        self.assertTrue(first.has_disjoint_source_support(second))
        combined = first.union(second)
        self.assertIsNotNone(combined)
        assert combined is not None
        self.assertTrue(combined.is_bijection_from_source_support())
        combined_converse = combined.converse()
        self.assertIsNotNone(combined_converse)
        assert combined_converse is not None
        self.assertTrue(combined_converse.is_total_function())

        with self.assertRaisesRegex(ValueError, "own each logical task once"):
            WorkerSchedule(
                6,
                (WorkerScheduleSegment(0, prefix, 0, 4, 0),),
            )
        with self.assertRaisesRegex(ValueError, "own each logical task once"):
            WorkerSchedule(
                6,
                (
                    WorkerScheduleSegment(0, prefix, 0, 4, 0),
                    WorkerScheduleSegment(0, duplicate, 4, 2, 0),
                ),
            )

    def test_worker_schedule_requires_disjoint_source_support(self) -> None:
        schedule_domain = CoordinateDomain(
            (-3, -2, -1),
            ((-3, 2), (-2, 6), (-1, 1)),
            kind="worker",
        )
        left_target = CoordinateDomain((10,), ((10, 4),), identity=0)
        right_target = CoordinateDomain((20,), ((20, 2),), identity=1)
        worker = coordinate_axis_symbol(-2)

        def placement(
            target: CoordinateDomain,
            begin: int,
            end: int,
        ) -> CoordinateRelation:
            return CoordinateRelation.point_map(
                schedule_domain,
                target,
                (
                    (
                        ((-3, 1, 2, 1), (-2, begin, end, 1), (-1, 0, 1, 1)),
                        (worker - begin,),
                    ),
                ),
            )

        left = placement(left_target, 0, 4)
        adjacent = placement(right_target, 4, 6)
        overlap = placement(right_target, 3, 5)
        self.assertTrue(left.has_disjoint_source_support(adjacent))
        self.assertFalse(left.has_disjoint_source_support(overlap))
        WorkerSchedule(
            6,
            (
                WorkerScheduleSegment(0, left, 0, 4, 0),
                WorkerScheduleSegment(1, adjacent, 4, 2, 0),
            ),
        )
        with self.assertRaisesRegex(ValueError, "support overlaps"):
            WorkerSchedule(
                6,
                (
                    WorkerScheduleSegment(0, left, 0, 4, 0),
                    WorkerScheduleSegment(1, overlap, 3, 2, 0),
                ),
            )

    def test_schedule_bijection_rejects_padded_and_out_of_domain_support(
        self,
    ) -> None:
        schedule_domain = CoordinateDomain(
            (-3, -2, -1),
            ((-3, 2), (-2, 4), (-1, 1)),
            kind="worker",
        )
        target = CoordinateDomain((10,), ((10, 3),), identity=0)
        worker = coordinate_axis_symbol(-2)
        valid = CoordinateRelation.point_map(
            schedule_domain,
            target,
            (
                (
                    ((-3, 1, 2, 1), (-2, 1, 4, 1), (-1, 0, 1, 1)),
                    (worker - 1,),
                ),
            ),
        )
        self.assertTrue(valid.is_bijection_from_source_support())
        self.assertIsNotNone(valid.converse())

        invalid_relations = {
            "padded target tail": CoordinateRelation.point_map(
                schedule_domain,
                target,
                (
                    (
                        ((-3, 1, 2, 1), (-2, 1, 4, 1), (-1, 0, 1, 1)),
                        (worker,),
                    ),
                ),
            ),
            "out-of-domain source": CoordinateRelation.point_map(
                schedule_domain,
                target,
                (
                    (
                        ((-3, 1, 2, 1), (-2, -1, 2, 1), (-1, 0, 1, 1)),
                        (worker + 1,),
                    ),
                ),
            ),
        }
        for name, invalid in invalid_relations.items():
            with self.subTest(case=name):
                # Replacing a relation after proving its converse must not
                # retain a stale proof for the changed forward relation.
                transformed = dataclasses.replace(valid, pieces=invalid.pieces)
                self.assertFalse(transformed.is_bijection_from_source_support())
                with self.assertRaisesRegex(
                    ValueError,
                    "own each logical task once",
                ):
                    WorkerSchedule(
                        4,
                        (WorkerScheduleSegment(0, transformed, 0, 4, 0),),
                    )

    def test_root_barrier_participant_order_requires_support_bijection(self) -> None:
        workers = CoordinateDomain((-1,), ((-1, 4),), kind="worker")
        arrivals = CoordinateDomain(
            (-2,),
            ((-2, 2),),
            kind="value",
            identity=0,
        )

        def participant_order(
            assignments: tuple[tuple[int, int], ...],
        ) -> CoordinateRelation:
            return CoordinateRelation.point_map(
                workers,
                arrivals,
                tuple(
                    (
                        ((-1, worker, worker + 1, 1),),
                        (sympy.Integer(arrival),),
                    )
                    for worker, arrival in assignments
                ),
            )

        valid_order = participant_order(((1, 0), (3, 1)))
        self.assertTrue(valid_order.is_bijection_from_source_support())
        valid = RootBarrierPublicationPlan(
            root=0,
            participant_intervals=(),
            participant_order=valid_order,
            publications=(RootBarrierPublication(0, ()),),
            resident_arrival_count=2,
            continuation_arrival_count=0,
            source_stage_arrival_count=0,
            real_arrival_count=2,
            effective_arrival_count=2,
            maximum_arrival_count=4,
        )

        invalid_orders = {
            "missing arrival": participant_order(((1, 0),)),
            "duplicate arrival": participant_order(((1, 0), (3, 0))),
            "padded arrival": participant_order(((1, 0), (3, 2))),
        }
        for name, invalid_order in invalid_orders.items():
            with self.subTest(case=name):
                self.assertFalse(invalid_order.is_bijection_from_source_support())
                with self.assertRaises(ValueError):
                    dataclasses.replace(valid, participant_order=invalid_order)

    def test_runtime_empty_root_has_one_bijective_synthetic_participant(
        self,
    ) -> None:
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)
        worker_count = 4
        root = CoordinateDomain((10,), ((10, extent),), identity=0)
        task_order = pid_task_order(root, root.axis_order)
        schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (root,),
            (task_order,),
            worker_count,
        )
        publication = cross_loop_scheduler.root_barrier_publication_plan(
            schedule,
            0,
        )
        participant_order = publication.participant_order
        self.assertIsNotNone(participant_order)
        assert participant_order is not None
        self.assertTrue(participant_order.is_bijection_from_source_support())

        for concrete_extent in (0, 1, worker_count - 1, worker_count, worker_count + 1):
            concrete = participant_order.substitute_parameters(
                {extent: concrete_extent}
            )
            expected = max(1, min(worker_count, concrete_extent))
            self.assertEqual(concrete.target_domain.size, expected)
            self.assertEqual(concrete.source_support_cardinality(), expected)
            self.assertTrue(concrete.is_bijection_from_source_support())
            converse = concrete.converse()
            self.assertIsNotNone(converse)
            assert converse is not None
            self.assertTrue(converse.is_total_function())
