from __future__ import annotations

import copy
import dataclasses
import pickle
from unittest import mock

import sympy

from helion._compiler.cross_loop_scheduler import WorkerScheduleSegment
from helion._compiler.tile_dependency import CoordinateDomain
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import _bounded_parameter_expression_interval
from helion._compiler.tile_dependency import _CoordinateRelationPiece
from helion._compiler.tile_dependency import _is_provably_nonnegative
from helion._compiler.tile_dependency import _memoized_exact_converse
from helion._compiler.tile_dependency import _remember_exact_converse
from helion._compiler.tile_dependency import coordinate_axis_symbol
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
        self.assertIsNotNone(renamed)
        assert renamed is not None
        self.assertIsNotNone(_memoized_exact_converse(renamed))

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

    def test_local_order_rejects_padded_and_out_of_domain_support(
        self,
    ) -> None:
        local_order = CoordinateDomain(
            (10,),
            ((10, 3),),
            kind="task_order",
        )
        target = CoordinateDomain(
            (20,),
            ((20, 3),),
            kind="site",
            identity=0,
        )
        ordinal = coordinate_axis_symbol(10)
        valid = CoordinateRelation.point_map(
            local_order,
            target,
            (
                (
                    ((10, 0, 3, 1),),
                    (ordinal,),
                ),
            ),
        )
        self.assertTrue(valid.is_bijection_from_source_support())
        self.assertIsNotNone(valid.converse())
        WorkerScheduleSegment(root=0, task_order=valid)

        invalid_relations = {
            "padded target tail": CoordinateRelation.point_map(
                local_order,
                target,
                (
                    (
                        ((10, 0, 3, 1),),
                        (ordinal + 1,),
                    ),
                ),
            ),
            "out-of-domain source": CoordinateRelation.point_map(
                local_order,
                target,
                (
                    (
                        ((10, -1, 2, 1),),
                        (ordinal + 1,),
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
                    "not an exact local bijection",
                ):
                    WorkerScheduleSegment(root=0, task_order=transformed)
