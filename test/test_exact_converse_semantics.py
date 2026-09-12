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
from helion._compiler.tile_dependency import _bounded_parameter_expression_interval
from helion._compiler.tile_dependency import _CoordinateRelationPiece
from helion._compiler.tile_dependency import _is_provably_nonnegative
from helion._compiler.tile_dependency import _memoized_exact_converse
from helion._compiler.tile_dependency import _remember_exact_converse
from helion._compiler.tile_dependency import coordinate_axis_symbol
from helion._compiler.tile_dependency import pid_task_order
from helion._testing import TestCase


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

    def test_relation_transform_exact_converse_provenance(self) -> None:
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)
        source = CoordinateDomain(
            (10, 11),
            ((10, 2), (11, extent)),
            kind="task_order",
            _allow_empty=True,
        )
        target = CoordinateDomain(
            (20, 21),
            ((20, extent), (21, 2)),
            identity=0,
            _allow_empty=True,
        )
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((10, 0, 2, 1), (11, 0, extent, 1)),
                    (coordinate_axis_symbol(11), coordinate_axis_symbol(10)),
                ),
            ),
        )
        self.assertIsNotNone(relation.converse())

        renamed_target = CoordinateDomain(
            (30, 31),
            ((30, extent), (31, 2)),
            identity=1,
            _allow_empty=True,
        )
        renamed = relation.rename_target_axes(renamed_target)
        reordered = relation.reorder_source_axes((11, 10))
        self.assertIsNotNone(renamed)
        self.assertIsNotNone(reordered)
        assert renamed is not None and reordered is not None
        self.assertIsNotNone(_memoized_exact_converse(renamed))
        self.assertIsNotNone(_memoized_exact_converse(reordered))

        def assert_cached_converse_is_exact(
            transformed: CoordinateRelation,
            concrete_extent: int,
        ) -> None:
            concrete = transformed.substitute_parameters({extent: concrete_extent})
            converse = _memoized_exact_converse(concrete)
            self.assertIsNotNone(converse)
            assert converse is not None
            forward = concrete.materialize()
            expected = tuple(
                frozenset(
                    source_index
                    for source_index, targets in enumerate(forward)
                    if target_index in targets
                )
                for target_index in range(concrete.target_domain.size)
            )
            self.assertEqual(converse.materialize(), expected)

        for concrete_extent in (0, 1, 4):
            with self.subTest(transform="rename_target", extent=concrete_extent):
                assert_cached_converse_is_exact(renamed, concrete_extent)
            with self.subTest(transform="reorder_source", extent=concrete_extent):
                assert_cached_converse_is_exact(reordered, concrete_extent)
            with self.subTest(transform="substitute", extent=concrete_extent):
                assert_cached_converse_is_exact(relation, concrete_extent)

        projected_target = relation.project_target(
            CoordinateDomain(
                (20,),
                ((20, extent),),
                identity=0,
                _allow_empty=True,
            )
        )
        projected_source = relation.project_source(
            CoordinateDomain(
                (11,),
                ((11, extent),),
                kind="task_order",
                _allow_empty=True,
            )
        )
        narrow_source = CoordinateDomain(
            (11,),
            ((11, extent),),
            kind="task_order",
            _allow_empty=True,
        )
        narrow_target = CoordinateDomain(
            (20,),
            ((20, extent),),
            identity=0,
            _allow_empty=True,
        )
        narrow = CoordinateRelation.point_map(
            narrow_source,
            narrow_target,
            (
                (
                    ((11, 0, extent, 1),),
                    (coordinate_axis_symbol(11),),
                ),
            ),
        )
        self.assertIsNotNone(narrow.converse())
        lifted_source = narrow.lift_source(source)
        replaced = dataclasses.replace(relation)
        for name, transformed in (
            ("project_target", projected_target),
            ("project_source", projected_source),
            ("lift_source", lifted_source),
            ("dataclasses.replace", replaced),
        ):
            with self.subTest(transform=name):
                self.assertIsNotNone(transformed)
                assert transformed is not None
                self.assertIsNone(_memoized_exact_converse(transformed))

    def test_source_and_target_coalescing_retain_exact_converse(self) -> None:
        source = CoordinateDomain((10,), ((10, 4),), kind="task_order")
        target = CoordinateDomain((20,), ((20, 4),), identity=0)
        source_coordinate = coordinate_axis_symbol(10)
        source_partitioned = CoordinateRelation.point_map(
            source,
            target,
            (
                (((10, 0, 2, 1),), (source_coordinate,)),
                (((10, 2, 4, 1),), (source_coordinate,)),
            ),
        )
        self.assertIsNotNone(source_partitioned.converse())
        source_coalesced = source_partitioned.coalesce_adjacent_source_boxes()
        self.assertEqual(len(source_coalesced.pieces), 1)

        singleton_source = CoordinateDomain(
            (30,),
            ((30, 1),),
            kind="task_order",
        )
        target_partitioned = CoordinateRelation(
            singleton_source,
            target,
            (
                _CoordinateRelationPiece(
                    ((30, 0, 1, 1),),
                    ((20, sympy.Integer(0), sympy.Integer(2), 1),),
                ),
                _CoordinateRelationPiece(
                    ((30, 0, 1, 1),),
                    ((20, sympy.Integer(2), sympy.Integer(4), 1),),
                ),
            ),
        )
        self.assertIsNotNone(target_partitioned.converse())
        target_coalesced = target_partitioned.coalesce_adjacent_target_boxes()
        self.assertEqual(len(target_coalesced.pieces), 1)

        for name, transformed in (
            ("source", source_coalesced),
            ("target", target_coalesced),
        ):
            with self.subTest(coalescing=name):
                converse = _memoized_exact_converse(transformed)
                self.assertIsNotNone(converse)
                assert converse is not None
                forward = transformed.materialize()
                expected = tuple(
                    frozenset(
                        source_index
                        for source_index, targets in enumerate(forward)
                        if target_index in targets
                    )
                    for target_index in range(transformed.target_domain.size)
                )
                self.assertEqual(converse.materialize(), expected)

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
