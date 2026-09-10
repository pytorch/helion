from __future__ import annotations

import copy
import dataclasses
import pickle
from unittest import mock

import sympy
from torch.utils._sympy.functions import Max as SymbolicMax
from torch.utils._sympy.functions import Min as SymbolicMin

from helion._compiler import cross_loop_scheduler
from helion._compiler.cross_loop_scheduler import RootBarrierPublication
from helion._compiler.cross_loop_scheduler import RootBarrierPublicationPlan
from helion._compiler.cross_loop_scheduler import WorkerSchedule
from helion._compiler.cross_loop_scheduler import WorkerScheduleSegment
from helion._compiler.tile_dependency import CoordinateDomain
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import _bounded_parameter_expression_interval
from helion._compiler.tile_dependency import _is_provably_nonnegative
from helion._compiler.tile_dependency import _memoized_exact_converse
from helion._compiler.tile_dependency import _remember_exact_converse
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
    @staticmethod
    def _partial_worker_task_order(
        *,
        target_identity: int = 0,
    ) -> tuple[
        CoordinateDomain,
        CoordinateDomain,
        CoordinateRelation,
    ]:
        schedule_domain = CoordinateDomain(
            (-3, -2, -1),
            ((-3, 2), (-2, 4), (-1, 2)),
            kind="worker",
        )
        widened_domain = CoordinateDomain(
            (-3, -2, -1),
            ((-3, 2), (-2, 4), (-1, 3)),
            kind="worker",
        )
        target_domain = CoordinateDomain(
            (10,),
            ((10, 3),),
            identity=target_identity,
        )
        worker = coordinate_axis_symbol(-2)
        task = coordinate_axis_symbol(10)
        task_order = CoordinateRelation.point_map(
            schedule_domain,
            target_domain,
            (
                (
                    ((-3, 1, 2, 1), (-2, 0, 3, 1), (-1, 0, 1, 1)),
                    (worker,),
                ),
            ),
        )
        exact_converse = CoordinateRelation.point_map(
            target_domain,
            schedule_domain,
            (
                (
                    ((10, 0, 3, 1),),
                    (sympy.Integer(1), task, sympy.Integer(0)),
                ),
            ),
        )
        _remember_exact_converse(task_order, exact_converse)
        return schedule_domain, widened_domain, task_order

    def test_source_domain_rebase_retains_exact_converse_without_reproof(
        self,
    ) -> None:
        old_domain, widened_domain, task_order = self._partial_worker_task_order()

        with mock.patch.object(
            CoordinateRelation,
            "_factored_source_support_converse",
            new_callable=mock.PropertyMock,
            side_effect=AssertionError("source rebasing must retain its converse"),
        ):
            rebased = task_order.rebase_source_domain(widened_domain)
            self.assertIsNotNone(rebased)
            assert rebased is not None
            self.assertEqual(rebased.pieces, task_order.pieces)
            self.assertTrue(rebased.is_bijection_from_source_support())
            converse = rebased.converse()
            self.assertIsNotNone(converse)
            assert converse is not None
            self.assertEqual(converse.target_domain, widened_domain)
            self.assertTrue(converse.is_total_function())

        launch_axis, worker_axis, wave_axis = widened_domain.axis_order
        for launch_stage in range(2):
            for worker in range(4):
                self.assertEqual(
                    rebased.target_coordinates(
                        {
                            launch_axis: launch_stage,
                            worker_axis: worker,
                            wave_axis: 2,
                        }
                    ),
                    frozenset(),
                )
        self.assertEqual(task_order.source_domain, old_domain)

    def test_reorder_symbolic_source_axes_retains_exact_converse(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = CoordinateDomain(
            (10, 11),
            ((10, 3), (11, batch)),
            kind="task_order",
            _allow_empty=True,
        )
        target = CoordinateDomain(
            (20, 21),
            ((20, batch), (21, 3)),
            identity=0,
            _allow_empty=True,
        )
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((10, 0, 3, 1), (11, 0, batch, 1)),
                    (coordinate_axis_symbol(11), coordinate_axis_symbol(10)),
                ),
            ),
        )
        self.assertIsNotNone(relation.converse())

        with mock.patch.object(
            CoordinateRelation,
            "_factored_source_support_converse",
            new_callable=mock.PropertyMock,
            side_effect=AssertionError("source reorder must retain its converse"),
        ):
            reordered = relation.reorder_source_axes((11, 10))
            self.assertIsNotNone(reordered)
            assert reordered is not None
            self.assertEqual(reordered.source_domain.shape_expr, (batch, 3))
            self.assertTrue(reordered.source_domain._allow_empty)
            self.assertTrue(reordered.is_bijection_from_source_support())
            converse = reordered.converse()
            self.assertIsNotNone(converse)
            assert converse is not None
            self.assertEqual(converse.target_domain, reordered.source_domain)

        for concrete_batch in (0, 1, 4):
            with self.subTest(batch=concrete_batch):
                concrete = reordered.substitute_parameters({batch: concrete_batch})
                original = relation.substitute_parameters({batch: concrete_batch})
                self.assertTrue(concrete.is_bijection_from_source_support())
                self.assertEqual(
                    concrete.materialize(source_axis_order=(10, 11)),
                    original.materialize(source_axis_order=(10, 11)),
                )

    def test_source_support_ordinalization_retains_constructed_inverse(self) -> None:
        _old_domain, _widened_domain, task_order = self._partial_worker_task_order()

        ordinalization = task_order._ordinalized_source_support
        self.assertIsNotNone(ordinalization)
        assert ordinalization is not None
        inverse = _memoized_exact_converse(ordinalization)
        self.assertIsNotNone(inverse)
        assert inverse is not None
        self.assertTrue(inverse.is_total_function())

    def test_normalized_segment_widening_does_not_reconstruct_converse(
        self,
    ) -> None:
        _old_domain, widened_domain, task_order = self._partial_worker_task_order()
        segment = WorkerScheduleSegment(
            root=0,
            task_order=task_order,
            worker_begin=0,
            worker_count=4,
            dispatch_offset=0,
        )

        with mock.patch.object(
            CoordinateRelation,
            "_factored_source_support_converse",
            new_callable=mock.PropertyMock,
            side_effect=AssertionError("normalization must retain its converse"),
        ):
            normalized = cross_loop_scheduler._normalize_dense_schedule_segment(
                segment,
                widened_domain,
            )
            self.assertIsNotNone(_memoized_exact_converse(normalized.task_order))
            WorkerSchedule(4, (normalized,))

    def test_transient_source_widening_retains_resident_converse(self) -> None:
        _old_domain, widened_domain, resident_order = self._partial_worker_task_order(
            target_identity=1
        )
        resident_target = resident_order.target_domain
        resident_schedule = WorkerSchedule(
            4,
            (WorkerScheduleSegment(1, resident_order, 0, 4, 0),),
        )

        source_domain = CoordinateDomain((20,), ((20, 9),), identity=0)
        source_order = pid_task_order(source_domain, source_domain.axis_order)
        resident_reference = pid_task_order(
            resident_target,
            resident_target.axis_order,
        )
        source_segment = cross_loop_scheduler._normalize_dense_schedule_segment(
            WorkerScheduleSegment(0, source_order, 0, 4, 0),
            widened_domain,
            launch_stage=0,
        )
        self.assertIsNotNone(source_segment.task_order.converse())
        normalize = cross_loop_scheduler._normalize_dense_schedule_segment

        def use_prepared_source_segment(
            segment: WorkerScheduleSegment,
            schedule_domain: CoordinateDomain,
            *,
            launch_stage: int = 1,
        ) -> WorkerScheduleSegment:
            if not segment.is_normalized and segment.root == 0 and launch_stage == 0:
                self.assertEqual(schedule_domain, widened_domain)
                return source_segment
            return normalize(
                segment,
                schedule_domain,
                launch_stage=launch_stage,
            )

        with (
            mock.patch.object(
                cross_loop_scheduler,
                "_normalize_dense_schedule_segment",
                side_effect=use_prepared_source_segment,
            ),
            mock.patch.object(
                CoordinateRelation,
                "_factored_source_support_converse",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "transient widening must retain resident converses"
                ),
            ),
        ):
            result = cross_loop_scheduler._with_transient_source_schedule_segment(
                resident_schedule,
                (source_order, resident_reference),
                0,
            )
            self.assertIsNotNone(result)
            assert result is not None
            widened_resident = result.segments_for_root(1)[0].task_order
            self.assertEqual(widened_resident.source_domain, widened_domain)
            self.assertIsNotNone(_memoized_exact_converse(widened_resident))
            self.assertTrue(widened_resident.is_bijection_from_source_support())

    def test_static_quotient_bounds_are_valid_for_signed_integer_base(self) -> None:
        value = sympy.Symbol("value", integer=True)

        def quotient(numerator: sympy.Expr, divisor: int) -> sympy.Expr:
            return sympy.floor(numerator / divisor)  # pyrefly: ignore[bad-return]

        bounded = 1 + quotient(value, 4) - quotient(value + 2, 4)
        remainder_complement = 5 + 4 * quotient(value, 4) - value
        negative = quotient(value - 1, 4) - quotient(value, 4)
        oversized_offset = 1 + quotient(value, 4) - quotient(value + 5, 4)
        mismatched_divisor = 1 + quotient(value, 4) - quotient(value + 2, 5)

        self.assertEqual(
            _bounded_parameter_expression_interval(bounded),
            (0, 1),
        )
        self.assertEqual(
            _bounded_parameter_expression_interval(remainder_complement),
            (2, 5),
        )
        self.assertTrue(_is_provably_nonnegative(bounded, None))
        self.assertTrue(_is_provably_nonnegative(remainder_complement, None))
        self.assertFalse(_is_provably_nonnegative(negative, None))
        self.assertFalse(_is_provably_nonnegative(oversized_offset, None))
        self.assertFalse(_is_provably_nonnegative(mismatched_divisor, None))

    def test_full_source_composition_does_not_restore_clipped_points(self) -> None:
        source = CoordinateDomain((10,), ((10, 2),), kind="worker")
        middle = CoordinateDomain((20,), ((20, 1),), kind="task_order")
        target = CoordinateDomain((30,), ((30, 1),), kind="site")
        source_coordinate = coordinate_axis_symbol(10)
        middle_coordinate = coordinate_axis_symbol(20)
        first = CoordinateRelation.point_map(
            source,
            middle,
            ((((10, 0, 2, 1),), (source_coordinate,)),),
        )
        first_inverse = CoordinateRelation.point_map(
            middle,
            source,
            ((((20, 0, 1, 1),), (middle_coordinate,)),),
        )
        _remember_exact_converse(first, first_inverse)
        following = CoordinateRelation.point_map(
            middle,
            target,
            ((((20, 0, 1, 1),), (sympy.Integer(0),)),),
        )
        following_inverse = CoordinateRelation.point_map(
            target,
            middle,
            ((((30, 0, 1, 1),), (sympy.Integer(0),)),),
        )
        _remember_exact_converse(following, following_inverse)

        self.assertEqual(
            first.materialize(),
            (frozenset((0,)), frozenset()),
        )
        self.assertIsNone(first.then(following))

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

    def test_unaligned_packed_interval_preserves_symbolic_support(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        worker_count = 7
        domains = (
            CoordinateDomain((10,), ((10, 3 * batch),), identity=0),
            CoordinateDomain((20,), ((20, 4 * batch),), identity=1),
        )
        task_orders = tuple(
            pid_task_order(domain, domain.axis_order) for domain in domains
        )

        with (
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError(
                    "symbolic support proof must not enumerate"
                ),
            ),
            mock.patch.object(
                CoordinateRelation,
                "_factored_source_support_converse",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "ordinary packed construction must retain its converse"
                ),
            ),
            mock.patch.object(
                CoordinateRelation,
                "_ordinalized_source_support",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "ordinary packed disjointness must use source geometry"
                ),
            ),
        ):
            schedule = cross_loop_scheduler._build_root_major_worker_schedule(
                domains,
                task_orders,
                worker_count,
            )
            relation = schedule.segments[1].task_order
            self.assertTrue(relation.is_bijection_from_source_support())
            converse = relation.converse()
            self.assertIsNotNone(converse)
            assert converse is not None
            self.assertTrue(converse.is_total_function())

        for concrete_batch in (0, 1, 8):
            concrete = relation.substitute_parameters({batch: concrete_batch})
            launch_axis, worker_axis, wave_axis = concrete.source_domain.axis_order
            occupied_slots: list[int] = []
            targets: list[int] = []
            for wave in range(concrete_batch):
                for worker in range(worker_count):
                    mapped = concrete.target_coordinates(
                        {
                            launch_axis: 1,
                            worker_axis: worker,
                            wave_axis: wave,
                        }
                    )
                    if mapped:
                        occupied_slots.append(wave * worker_count + worker)
                        targets.append(next(iter(mapped))[0])
            self.assertEqual(
                occupied_slots,
                list(range(3 * concrete_batch, 7 * concrete_batch)),
            )
            self.assertEqual(targets, list(range(4 * concrete_batch)))

    def test_worker_schedule_validation_uses_symbolic_relation_counts(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        worker_count = 7
        domains = (
            CoordinateDomain((10,), ((10, 3 * batch),), identity=0),
            CoordinateDomain((20,), ((20, 4 * batch),), identity=1),
        )
        task_orders = tuple(
            pid_task_order(domain, domain.axis_order) for domain in domains
        )
        schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            domains,
            task_orders,
            worker_count,
        )
        cross_loop_scheduler._root_task_placement_relation.cache_clear()

        with (
            mock.patch.object(
                WorkerScheduleSegment,
                "task_count",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "symbolic validation must not request a concrete task count"
                ),
            ),
            mock.patch.object(
                CoordinateDomain,
                "size",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "symbolic validation must not request a concrete domain size"
                ),
            ),
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError(
                    "symbolic validation must not enumerate runtime tasks"
                ),
            ),
        ):
            self.assertTrue(
                cross_loop_scheduler._validate_worker_schedule_tasks(
                    schedule,
                    task_orders,
                )
            )

    def test_symbolic_traversal_helpers_avoid_concrete_counts_after_roundtrip(
        self,
    ) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        worker_count = 7
        domain = CoordinateDomain(
            (10, 11),
            ((10, 2), (11, batch)),
            identity=0,
        )
        reference = pid_task_order(domain, domain.axis_order)
        schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (domain,),
            (reference,),
            worker_count,
        )
        variants = (
            ("direct", schedule, reference),
            ("deepcopy", copy.deepcopy(schedule), copy.deepcopy(reference)),
            ("pickle", *pickle.loads(pickle.dumps((schedule, reference)))),
        )

        with (
            mock.patch.object(
                WorkerScheduleSegment,
                "task_count",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "symbolic traversal must not request a concrete task count"
                ),
            ),
            mock.patch.object(
                CoordinateDomain,
                "size",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "symbolic traversal must not request a concrete domain size"
                ),
            ),
            mock.patch.object(
                CoordinateDomain,
                "axis_counts",
                new_callable=mock.PropertyMock,
                side_effect=AssertionError(
                    "symbolic traversal must not request concrete axis counts"
                ),
            ),
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError(
                    "symbolic traversal must not enumerate runtime tasks"
                ),
            ),
        ):
            for name, variant_schedule, variant_reference in variants:
                with self.subTest(roundtrip=name):
                    cross_loop_scheduler._root_schedule_traversal.cache_clear()
                    (segment,) = variant_schedule.segments
                    logical_order = segment.logical_task_order
                    self.assertIsNotNone(logical_order)
                    assert logical_order is not None
                    self.assertTrue(logical_order.is_total_function())
                    logical_to_order = logical_order.converse()
                    self.assertIsNotNone(logical_to_order)
                    assert logical_to_order is not None
                    self.assertTrue(logical_to_order.is_single_valued())

                    worker_step_domain = variant_schedule.worker_step_domain
                    task_to_wave = segment.logical_task_wave_relation(
                        worker_step_domain
                    )
                    self.assertIsNotNone(task_to_wave)
                    assert task_to_wave is not None
                    self.assertTrue(task_to_wave.is_total_function())

                    ordinal_domain = CoordinateDomain(
                        (30,),
                        ((30, 8 * batch),),
                        kind="task_order",
                    )
                    forward = cross_loop_scheduler._flat_task_order_relation(
                        variant_reference,
                        ordinal_domain,
                        ordinal_begin=6 * batch,
                    )
                    inverse = cross_loop_scheduler._logical_task_to_order_ordinal(
                        variant_reference,
                        ordinal_domain,
                        ordinal_begin=6 * batch,
                    )
                    self.assertIsNotNone(forward)
                    self.assertIsNotNone(inverse)
                    assert forward is not None
                    assert inverse is not None
                    forward_converse = forward.converse()
                    self.assertIsNotNone(forward_converse)
                    assert forward_converse is not None
                    self.assertTrue(
                        forward_converse.is_pointwise_equal_on_same_support(inverse)
                    )

                    # Only one B-sized interval remains after this offset, so
                    # a 2*B traversal must be rejected symbolically.
                    self.assertIsNone(
                        cross_loop_scheduler._flat_task_order_relation(
                            variant_reference,
                            ordinal_domain,
                            ordinal_begin=7 * batch,
                        )
                    )
                    self.assertIsNone(
                        cross_loop_scheduler._logical_task_to_order_ordinal(
                            variant_reference,
                            ordinal_domain,
                            ordinal_begin=7 * batch,
                        )
                    )

                    traversal = cross_loop_scheduler._root_schedule_traversal(
                        variant_schedule.segments,
                        variant_reference,
                    )
                    self.assertIsNotNone(traversal)
                    assert traversal is not None
                    self.assertIsNotNone(
                        traversal.scheduled_ordinal_to_logical_task
                    )
                    self.assertIsNotNone(
                        traversal.logical_task_to_scheduled_ordinal
                    )
                    self.assertTrue(traversal.matches_reference)

        logical_order = schedule.segments[0].logical_task_order
        assert logical_order is not None
        for concrete_batch in (0, 1, 8):
            substitutions = {batch: concrete_batch}
            self.assertEqual(
                logical_order.substitute_parameters(substitutions).materialize(),
                reference.substitute_parameters(substitutions).materialize(),
            )

    def test_static_empty_traversal_helpers_are_exact_after_roundtrip(self) -> None:
        domain = CoordinateDomain(
            (10,),
            ((10, 0),),
            identity=0,
            _allow_empty=True,
        )
        reference = pid_task_order(domain, domain.axis_order)
        ordinal_domain = cross_loop_scheduler._task_order_ordinal_domain(reference)
        schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (domain,),
            (reference,),
            4,
        )
        variants = (
            ("direct", schedule, reference, ordinal_domain),
            (
                "deepcopy",
                copy.deepcopy(schedule),
                copy.deepcopy(reference),
                copy.deepcopy(ordinal_domain),
            ),
            (
                "pickle",
                *pickle.loads(
                    pickle.dumps((schedule, reference, ordinal_domain))
                ),
            ),
        )

        for name, variant_schedule, variant_reference, variant_ordinal in variants:
            with self.subTest(roundtrip=name):
                forward = cross_loop_scheduler._flat_task_order_relation(
                    variant_reference,
                    variant_ordinal,
                )
                inverse = cross_loop_scheduler._logical_task_to_order_ordinal(
                    variant_reference,
                    variant_ordinal,
                )
                self.assertIsNotNone(forward)
                self.assertIsNotNone(inverse)
                assert forward is not None
                assert inverse is not None
                self.assertFalse(forward.pieces)
                self.assertFalse(inverse.pieces)
                self.assertTrue(forward.is_total_function())
                self.assertTrue(inverse.is_total_function())
                self.assertEqual(forward.converse(), inverse)

                (segment,) = variant_schedule.segments
                logical_order = segment.logical_task_order
                self.assertIsNotNone(logical_order)
                assert logical_order is not None
                self.assertFalse(logical_order.pieces)
                worker_steps = variant_schedule.worker_step_domain
                self.assertTrue(worker_steps.size_expr.is_zero)
                task_to_wave = segment.logical_task_wave_relation(worker_steps)
                self.assertIsNotNone(task_to_wave)
                assert task_to_wave is not None
                self.assertFalse(task_to_wave.pieces)

                cross_loop_scheduler._root_schedule_traversal.cache_clear()
                traversal = cross_loop_scheduler._root_schedule_traversal(
                    variant_schedule.segments,
                    variant_reference,
                )
                self.assertIsNotNone(traversal)
                assert traversal is not None
                self.assertTrue(traversal.matches_reference)

    def test_runtime_empty_middle_root_preserves_adjacent_packed_support(
        self,
    ) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        worker_count = 4
        symbolic_domains = (
            CoordinateDomain((10,), ((10, 3),), identity=0),
            CoordinateDomain((20,), ((20, batch),), identity=1),
            CoordinateDomain((30,), ((30, 2),), identity=2),
        )
        symbolic = cross_loop_scheduler._build_root_major_worker_schedule(
            symbolic_domains,
            tuple(
                pid_task_order(domain, domain.axis_order) for domain in symbolic_domains
            ),
            worker_count,
        )

        for left_index, left in enumerate(symbolic.segments):
            self.assertTrue(left.task_order.is_bijection_from_source_support())
            for right in symbolic.segments[left_index + 1 :]:
                self.assertTrue(
                    left.task_order.has_disjoint_source_support(right.task_order)
                )

        for concrete_batch in (0, 1, worker_count - 1, worker_count, worker_count + 1):
            substitutions = {batch: concrete_batch}
            concrete_domains = tuple(
                domain.substitute_parameters(substitutions)
                for domain in symbolic_domains
            )
            concrete = cross_loop_scheduler._build_root_major_worker_schedule(
                concrete_domains,
                tuple(
                    pid_task_order(domain, domain.axis_order)
                    for domain in concrete_domains
                ),
                worker_count,
            )
            self.assertEqual(
                tuple(
                    segment.task_order.substitute_parameters(
                        substitutions
                    ).materialize()
                    for segment in symbolic.segments
                ),
                tuple(
                    segment.task_order.materialize() for segment in concrete.segments
                ),
            )

    def test_root_major_builder_retains_geometry_for_all_consumers(self) -> None:
        batch = sympy.Symbol("batch", integer=True, positive=True)
        worker_count = 7
        domains = (
            CoordinateDomain((10, 11), ((10, batch), (11, 3)), identity=0),
            CoordinateDomain(
                (20, 21, 22),
                ((20, 2), (21, batch), (22, 2)),
                identity=1,
            ),
        )
        task_orders = (
            pid_task_order(domains[0], (11, 10)),
            pid_task_order(domains[1], (21, 22, 20)),
        )
        schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            domains,
            task_orders,
            worker_count,
        )
        cross_loop_scheduler.root_barrier_publication_plan.cache_clear()

        with (
            mock.patch.object(
                cross_loop_scheduler,
                "_parametric_root_major_schedule_geometry_from_parts",
                side_effect=AssertionError("builder geometry must be retained"),
            ),
            mock.patch.object(
                CoordinateRelation,
                "has_same_source_support",
                side_effect=AssertionError("participant lowering must not re-prove"),
            ),
            mock.patch.object(
                CoordinateRelation,
                "is_pointwise_equal_on_same_support",
                side_effect=AssertionError("participant lowering must not re-prove"),
            ),
        ):
            geometry = cross_loop_scheduler._parametric_root_major_schedule_geometry(
                schedule
            )
            self.assertIsNotNone(geometry)
            for root in range(len(domains)):
                for _ in range(2):
                    publication = (
                        cross_loop_scheduler.root_barrier_publication_plan(
                            schedule,
                            root,
                        )
                    )
                    self.assertIsNotNone(publication.participant_order)
                    assert publication.participant_order is not None
                    self.assertTrue(
                        publication.participant_order.is_bijection_from_source_support()
                    )

    def test_root_major_geometry_survives_copy_without_proof_provenance(self) -> None:
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)
        worker_count = 2
        for name, domain, task_order in _dynamic_task_orders(extent):
            with self.subTest(order=name):
                schedule = cross_loop_scheduler._build_root_major_worker_schedule(
                    (domain,),
                    (task_order,),
                    worker_count,
                )
                for rebuilt in (
                    copy.deepcopy(schedule),
                    pickle.loads(pickle.dumps(schedule)),
                ):
                    self.assertEqual(rebuilt, schedule)
                    cross_loop_scheduler._root_task_placement_relation.cache_clear()
                    rebuilt_placement = (
                        cross_loop_scheduler._root_task_placement_relation(rebuilt, 0)
                    )
                    original_placement = (
                        cross_loop_scheduler._root_task_placement_relation(schedule, 0)
                    )
                    self.assertIsNotNone(rebuilt_placement)
                    self.assertEqual(rebuilt_placement, original_placement)
                    geometry = (
                        cross_loop_scheduler._parametric_root_major_schedule_geometry(
                            rebuilt
                        )
                    )
                    self.assertIsNotNone(geometry)
                    cross_loop_scheduler.root_barrier_publication_plan.cache_clear()
                    publication = (
                        cross_loop_scheduler.root_barrier_publication_plan(rebuilt, 0)
                    )
                    self.assertIsNotNone(publication.participant_order)
                    self.assertEqual(
                        publication.resident_arrival_count,
                        SymbolicMin(worker_count, domain.size_expr),
                    )
                    self.assertEqual(
                        publication.effective_arrival_count,
                        SymbolicMax(
                            1,
                            SymbolicMin(worker_count, domain.size_expr),
                        ),
                    )

    def test_unaligned_root_major_copy_recovers_conditional_support(self) -> None:
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)
        worker_count = 7
        prefix = CoordinateDomain(
            (10,),
            ((10, 3 * extent),),
            identity=0,
        )
        _name, reflected_domain, reflected_order = next(
            item for item in _dynamic_task_orders(extent) if item[0] == "reflected"
        )
        reflected_domain = dataclasses.replace(reflected_domain, identity=1)
        reflected_order = dataclasses.replace(
            reflected_order,
            source_domain=dataclasses.replace(
                reflected_order.source_domain,
                identity=1,
            ),
            target_domain=reflected_domain,
        )
        schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (prefix, reflected_domain),
            (pid_task_order(prefix, prefix.axis_order), reflected_order),
            worker_count,
        )

        for rebuilt in (
            copy.deepcopy(schedule),
            pickle.loads(pickle.dumps(schedule)),
        ):
            geometry = cross_loop_scheduler._parametric_root_major_schedule_geometry(
                rebuilt
            )
            self.assertIsNotNone(geometry)
            assert geometry is not None
            self.assertEqual(
                tuple((first_slot, task_count) for _, first_slot, task_count in geometry),
                ((0, 3 * extent), (3 * extent, 4 * extent)),
            )
            for root, task_count in enumerate((3 * extent, 4 * extent)):
                cross_loop_scheduler.root_barrier_publication_plan.cache_clear()
                publication = cross_loop_scheduler.root_barrier_publication_plan(
                    rebuilt,
                    root,
                )
                self.assertIsNotNone(publication.participant_order)
                self.assertEqual(
                    publication.resident_arrival_count,
                    SymbolicMin(worker_count, task_count),
                )
                self.assertEqual(
                    publication.effective_arrival_count,
                    SymbolicMax(1, SymbolicMin(worker_count, task_count)),
                )

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

        # Dense-support intervals are comparable only after both relations
        # use the same source-axis basis.  Dropping each relation's own fixed
        # axis would incorrectly make these overlapping supports look like
        # adjacent scalar intervals.
        crossed_domain = CoordinateDomain((0, 1), ((0, 2), (1, 4)))
        crossed_left = CoordinateRelation.point_map(
            crossed_domain,
            CoordinateDomain((10,), ((10, 2),)),
            (
                (
                    ((0, 1, 2, 1), (1, 2, 4, 1)),
                    (coordinate_axis_symbol(1) - 2,),
                ),
            ),
        )
        crossed_right = CoordinateRelation.point_map(
            crossed_domain,
            CoordinateDomain((20,), ((20, 2),)),
            (
                (
                    ((0, 0, 2, 1), (1, 2, 3, 1)),
                    (coordinate_axis_symbol(0),),
                ),
            ),
        )
        self.assertIsNotNone(crossed_left.converse())
        self.assertIsNotNone(crossed_right.converse())
        self.assertFalse(crossed_left.has_disjoint_source_support(crossed_right))

        with mock.patch.object(
            cross_loop_scheduler,
            "_parametric_root_major_schedule_geometry_from_parts",
            side_effect=AssertionError(
                "derived geometry must not replace relation validation"
            ),
        ):
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

        concrete_root = root.substitute_parameters({extent: 0})
        concrete_order = pid_task_order(concrete_root, concrete_root.axis_order)
        concrete_schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (concrete_root,),
            (concrete_order,),
            worker_count,
        )
        concrete_publication = cross_loop_scheduler.root_barrier_publication_plan(
            concrete_schedule,
            0,
        )
        self.assertEqual(concrete_publication.real_arrival_count, 0)
        self.assertEqual(concrete_publication.effective_arrival_count, 1)
        self.assertIsNotNone(concrete_publication.participant_order)
        assert concrete_publication.participant_order is not None
        self.assertTrue(
            concrete_publication.participant_order.is_bijection_from_source_support()
        )

    def test_empty_root_preserves_its_packed_synthetic_occurrence(self) -> None:
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)
        worker_count = 4
        prefix = CoordinateDomain((10,), ((10, 3),), identity=0)
        dynamic_empty = CoordinateDomain((20,), ((20, extent),), identity=1)
        symbolic_schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (prefix, dynamic_empty),
            (
                pid_task_order(prefix, prefix.axis_order),
                pid_task_order(dynamic_empty, dynamic_empty.axis_order),
            ),
            worker_count,
        )
        symbolic_publication = cross_loop_scheduler.root_barrier_publication_plan(
            symbolic_schedule,
            1,
        )
        self.assertIsNotNone(symbolic_publication.participant_order)
        assert symbolic_publication.participant_order is not None

        concrete_empty = dynamic_empty.substitute_parameters({extent: 0})
        concrete_schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (prefix, concrete_empty),
            (
                pid_task_order(prefix, prefix.axis_order),
                pid_task_order(concrete_empty, concrete_empty.axis_order),
            ),
            worker_count,
        )
        concrete_publication = cross_loop_scheduler.root_barrier_publication_plan(
            concrete_schedule,
            1,
        )
        self.assertIsNotNone(concrete_publication.participant_order)
        assert concrete_publication.participant_order is not None

        specialized = symbolic_publication.participant_order.substitute_parameters(
            {extent: 0}
        )
        self.assertEqual(
            concrete_publication.participant_order.materialize(),
            specialized.materialize(),
        )
        self.assertEqual(
            specialized.materialize(),
            (frozenset(), frozenset(), frozenset(), frozenset((0,))),
        )
