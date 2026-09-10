from __future__ import annotations

import copy
import math
import pickle
from unittest import mock

import sympy
from torch.utils._sympy.functions import FloorDiv

from helion._compiler import cross_loop_scheduler
from helion._compiler import tile_dependency
from helion._compiler.tile_dependency import CoordinateDomain
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import _flat_static_inner_dynamic_outer_converse
from helion._compiler.tile_dependency import coordinate_axis_symbol
from helion._compiler.tile_dependency import pid_task_order
from helion._testing import TestCase

_FIRST_AXIS = 10
_SECOND_AXIS = 20
_BATCH_AXIS = 30
_QUERY_AXIS = 40


def _domain(
    axes: tuple[int, ...],
    counts: tuple[int | sympy.Expr, ...],
    *,
    identity: int = 0,
    allow_empty: bool = False,
) -> CoordinateDomain:
    return CoordinateDomain(
        axes,
        tuple(zip(axes, counts, strict=True)),
        tuple((axis, 1) for axis in axes),
        identity=identity,
        _allow_empty=allow_empty,
    )


def _coordinates_in_linear_order(
    axes: tuple[int, ...],
    counts: dict[int, int],
) -> tuple[dict[int, int], ...]:
    """Enumerate a small test domain with ``axes[0]`` varying fastest."""
    coordinates = []
    for ordinal in range(math.prod(counts[axis] for axis in axes)):
        remainder = ordinal
        point: dict[int, int] = {}
        for axis in axes:
            point[axis] = remainder % counts[axis]
            remainder //= counts[axis]
        coordinates.append(point)
    return tuple(coordinates)


def _expected_l2_targets(
    domain: CoordinateDomain,
    pid_axis_order: tuple[int, ...],
    group_size: int,
) -> tuple[int, ...]:
    """Independent oracle for Helion's existing grouped-PID traversal."""
    counts = domain.axis_counts
    first_axis, second_axis, *outer_axes = pid_axis_order
    first_count = counts[first_axis]
    second_count = counts[second_axis]
    if first_count == 0 or second_count == 0:
        return ()
    actual_group_size = min(group_size, first_count)
    result: list[int] = []
    for outer in _coordinates_in_linear_order(tuple(outer_axes), counts):
        for first_begin in range(0, first_count, actual_group_size):
            first_end = min(first_begin + actual_group_size, first_count)
            for second in range(second_count):
                for first in range(first_begin, first_end):
                    result.append(
                        domain.index(
                            {
                                **outer,
                                first_axis: first,
                                second_axis: second,
                            }
                        )
                    )
    return tuple(result)


def _singleton_targets(relation: CoordinateRelation) -> tuple[int, ...]:
    result: list[int] = []
    for targets in relation.materialize():
        if not targets:
            continue
        if len(targets) != 1:
            raise AssertionError(f"expected point map, got {targets}")
        result.append(next(iter(targets)))
    return tuple(result)


def _assert_exact_bijection(
    test: TestCase,
    relation: CoordinateRelation,
    expected_cardinality: int | sympy.Expr,
) -> CoordinateRelation:
    cardinality = relation.source_support_cardinality()
    test.assertIsNotNone(cardinality)
    assert cardinality is not None
    test.assertEqual(sympy.simplify(cardinality - expected_cardinality), 0)
    test.assertTrue(relation.is_single_valued())
    test.assertTrue(relation.is_bijection_from_source_support())
    converse = relation.converse()
    test.assertIsNotNone(converse)
    assert converse is not None
    test.assertTrue(converse.is_total_function())
    return converse


class TestRaggedL2Schedule(TestCase):
    def test_nested_quotient_remainder_identity_is_simplified_first(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        first = sympy.Symbol("first", integer=True, nonnegative=True)
        second = sympy.Symbol("second", integer=True, nonnegative=True)
        dividend = (
            3 * batch
            + 2 * second
            + 6 * FloorDiv(first, 2)
            + sympy.Mod(first, 2)
        )
        expression = 7 * FloorDiv(dividend, 7) + sympy.Mod(dividend, 7)

        self.assertEqual(
            tile_dependency._simplify_integer_quotients(expression),
            tile_dependency._simplify_integer_quotients(dividend),
        )

    def test_ragged_l2_dynamic_outer_matches_existing_traversal(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        domain = _domain(
            (_FIRST_AXIS, _SECOND_AXIS, _BATCH_AXIS),
            (5, 3, batch),
            identity=1,
        )
        task_order = pid_task_order(
            domain,
            domain.axis_order,
            l2_group_size=2,
        )

        # The representation must stay independent of the runtime batch and
        # of the nine boxes used by the historical traversal implementation.
        self.assertEqual(len(task_order.pieces), 1)
        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic L2 proofs must not enumerate"),
        ):
            converse = _assert_exact_bijection(self, task_order, 15 * batch)
            ordinal_domain = CoordinateDomain(
                (40,),
                ((40, 15 * batch),),
                kind="task_order",
            )
            flat_order = cross_loop_scheduler._flat_task_order_relation(
                task_order,
                ordinal_domain,
            )
            self.assertIsNotNone(flat_order)
            assert flat_order is not None
            flat_converse = _assert_exact_bijection(
                self,
                flat_order,
                15 * batch,
            )
        self.assertLessEqual(len(converse.pieces), 2)
        self.assertEqual(len(flat_order.pieces), 1)
        self.assertLessEqual(len(flat_converse.pieces), 2)

        for concrete_batch in (0, 1, 2, 8):
            with self.subTest(batch=concrete_batch):
                concrete = task_order.substitute_parameters({batch: concrete_batch})
                concrete_domain = domain.substitute_parameters({batch: concrete_batch})
                expected = _expected_l2_targets(
                    concrete_domain,
                    concrete_domain.axis_order,
                    2,
                )
                self.assertEqual(_singleton_targets(concrete), expected)
                self.assertEqual(
                    _singleton_targets(
                        flat_order.substitute_parameters(
                            {batch: concrete_batch}
                        )
                    ),
                    expected,
                )
                self.assertEqual(len(expected), 15 * concrete_batch)
                self.assertEqual(len(set(expected)), len(expected))
                self.assertEqual(sorted(expected), list(range(15 * concrete_batch)))
                _assert_exact_bijection(self, concrete, 15 * concrete_batch)

    def test_ragged_l2_multiple_dynamic_outer_axes_recover_after_roundtrip(
        self,
    ) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        query = sympy.Symbol("query", integer=True, nonnegative=True)
        domain = _domain(
            (_FIRST_AXIS, _SECOND_AXIS, _BATCH_AXIS, _QUERY_AXIS),
            (5, 3, batch, query),
            identity=1,
        )
        task_order = pid_task_order(
            domain,
            domain.axis_order,
            l2_group_size=2,
        )
        ordinal_domain = CoordinateDomain(
            (50,),
            ((50, 15 * batch * query),),
            kind="task_order",
        )
        flat_order = cross_loop_scheduler._flat_task_order_relation(
            task_order,
            ordinal_domain,
        )
        self.assertIsNotNone(flat_order)
        assert flat_order is not None

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic L2 proofs must not enumerate"),
        ):
            for name, relation in (
                ("direct", task_order),
                ("direct_deepcopy", copy.deepcopy(task_order)),
                ("direct_pickle", pickle.loads(pickle.dumps(task_order))),
                ("flat", flat_order),
                ("flat_deepcopy", copy.deepcopy(flat_order)),
                ("flat_pickle", pickle.loads(pickle.dumps(flat_order))),
            ):
                with self.subTest(roundtrip=name):
                    converse = _assert_exact_bijection(
                        self,
                        relation,
                        15 * batch * query,
                    )
                    self.assertLessEqual(len(converse.pieces), 2)

        for concrete_batch, concrete_query in ((0, 2), (1, 1), (2, 3)):
            with self.subTest(batch=concrete_batch, query=concrete_query):
                substitutions = {
                    batch: concrete_batch,
                    query: concrete_query,
                }
                concrete_domain = domain.substitute_parameters(substitutions)
                expected = _expected_l2_targets(
                    concrete_domain,
                    concrete_domain.axis_order,
                    2,
                )
                self.assertEqual(
                    _singleton_targets(
                        task_order.substitute_parameters(substitutions)
                    ),
                    expected,
                )
                self.assertEqual(
                    _singleton_targets(
                        flat_order.substitute_parameters(substitutions)
                    ),
                    expected,
                )

    def test_flat_ragged_l2_infers_permuted_dynamic_axis_order(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        query = sympy.Symbol("query", integer=True, nonnegative=True)
        domain = _domain(
            (_BATCH_AXIS, _FIRST_AXIS, _QUERY_AXIS, _SECOND_AXIS),
            (batch, 5, query, 3),
            identity=1,
        )
        pid_axis_order = (
            _FIRST_AXIS,
            _SECOND_AXIS,
            _QUERY_AXIS,
            _BATCH_AXIS,
        )
        task_order = pid_task_order(
            domain,
            pid_axis_order,
            l2_group_size=2,
        )
        ordinal_domain = CoordinateDomain(
            (50,),
            ((50, 15 * query * batch),),
            kind="task_order",
        )
        flat_order = cross_loop_scheduler._flat_task_order_relation(
            task_order,
            ordinal_domain,
        )
        self.assertIsNotNone(flat_order)
        assert flat_order is not None

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("axis-order proof must not enumerate"),
        ):
            for name, relation in (
                ("deepcopy", copy.deepcopy(flat_order)),
                ("pickle", pickle.loads(pickle.dumps(flat_order))),
            ):
                with self.subTest(roundtrip=name):
                    _assert_exact_bijection(
                        self,
                        relation,
                        15 * query * batch,
                    )

        substitutions = {batch: 2, query: 3}
        concrete_domain = domain.substitute_parameters(substitutions)
        expected = _expected_l2_targets(
            concrete_domain,
            pid_axis_order,
            2,
        )
        self.assertEqual(
            _singleton_targets(flat_order.substitute_parameters(substitutions)),
            expected,
        )

    def test_flat_dynamic_outer_axis_order_declines_when_ambiguous(self) -> None:
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)
        source_axis = 10
        source = CoordinateDomain(
            (source_axis,),
            ((source_axis, extent * extent),),
            kind="task_order",
        )
        target = CoordinateDomain(
            (20, 30),
            ((20, extent), (30, extent)),
            identity=1,
        )
        ordinal = coordinate_axis_symbol(source_axis)
        repeated_digit = ordinal - FloorDiv(ordinal, extent) * extent
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((source_axis, 0, extent * extent, 1),),
                    (repeated_digit, repeated_digit),
                ),
            ),
        )

        # Both target axes match the first mixed-radix digit.  Picking either
        # by target-axis order would be an arbitrary and unsound tie-break.
        self.assertIsNone(_flat_static_inner_dynamic_outer_converse(relation))

        # All public proof entry points share one bounded negative result on a
        # relation instance.  Round trips intentionally retain only semantic
        # fields, so each rebuilt relation performs its own single attempt.
        proof_cache = "_flat_static_inner_dynamic_outer_converse_proof"
        factored_cache = "_factored_source_support_converse"
        original_derivation = _flat_static_inner_dynamic_outer_converse
        with mock.patch.object(
            tile_dependency,
            "_flat_static_inner_dynamic_outer_converse",
            wraps=original_derivation,
        ) as derivation:
            self.assertIsNone(relation.converse())
            self.assertIsNone(relation.derive_converse_and_target_counts()[0])
            self.assertFalse(relation.is_bijection_from_source_support())
            self.assertEqual(derivation.call_count, 1)
        self.assertIn(proof_cache, relation.__dict__)

        for name, rebuilt in (
            ("deepcopy", copy.deepcopy(relation)),
            ("pickle", pickle.loads(pickle.dumps(relation))),
        ):
            with self.subTest(roundtrip=name):
                self.assertNotIn(proof_cache, rebuilt.__dict__)
                self.assertNotIn(factored_cache, rebuilt.__dict__)
                with mock.patch.object(
                    tile_dependency,
                    "_flat_static_inner_dynamic_outer_converse",
                    wraps=original_derivation,
                ) as derivation:
                    self.assertIsNone(rebuilt.converse())
                    self.assertIsNone(
                        rebuilt.derive_converse_and_target_counts()[0]
                    )
                    self.assertEqual(derivation.call_count, 1)

    def test_ragged_l2_relation_proof_recovers_after_copy_and_pickle(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        domain = _domain(
            (_FIRST_AXIS, _SECOND_AXIS, _BATCH_AXIS),
            (5, 3, batch),
            identity=1,
        )
        original = pid_task_order(
            domain,
            domain.axis_order,
            l2_group_size=2,
        )

        for name, rebuilt in (
            ("deepcopy", copy.deepcopy(original)),
            ("pickle", pickle.loads(pickle.dumps(original))),
        ):
            with self.subTest(roundtrip=name):
                self.assertEqual(rebuilt, original)
                with mock.patch.object(
                    CoordinateRelation,
                    "materialize",
                    side_effect=AssertionError(
                        "proof recovery must not enumerate runtime CTAs"
                    ),
                ):
                    converse = _assert_exact_bijection(self, rebuilt, 15 * batch)
                self.assertLessEqual(len(converse.pieces), 2)
                for concrete_batch in (0, 1, 2, 8):
                    concrete = rebuilt.substitute_parameters(
                        {batch: concrete_batch}
                    )
                    concrete_domain = domain.substitute_parameters(
                        {batch: concrete_batch}
                    )
                    self.assertEqual(
                        _singleton_targets(concrete),
                        _expected_l2_targets(
                            concrete_domain,
                            concrete_domain.axis_order,
                            2,
                        ),
                    )

    def test_ragged_l2_survives_unaligned_packed_prefix(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        worker_count = 7
        prefix = _domain((1,), (3 * batch,), identity=0)
        l2_domain = _domain(
            (_FIRST_AXIS, _SECOND_AXIS, _BATCH_AXIS),
            (5, 3, batch),
            identity=1,
        )
        task_orders = (
            pid_task_order(prefix, prefix.axis_order),
            pid_task_order(
                l2_domain,
                l2_domain.axis_order,
                l2_group_size=2,
            ),
        )
        with (
            mock.patch.object(
                cross_loop_scheduler,
                "_parametric_root_major_schedule_geometry_from_parts",
                side_effect=AssertionError(
                    "root-major geometry must be retained after validation"
                ),
            ),
            mock.patch.object(
                tile_dependency,
                "_ordinalized_source_supports_are_disjoint",
                side_effect=AssertionError(
                    "dense source intervals must prove packed disjointness"
                ),
            ),
        ):
            schedule = cross_loop_scheduler._build_root_major_worker_schedule(
                (prefix, l2_domain),
                task_orders,
                worker_count,
            )

        self.assertEqual(len(schedule.segments), 2)
        self.assertIn(
            cross_loop_scheduler._DERIVED_ROOT_MAJOR_GEOMETRY_ATTRIBUTE,
            schedule.__dict__,
        )
        self.assertLessEqual(len(schedule.segments[0].task_order.pieces), 3)
        self.assertLessEqual(len(schedule.segments[1].task_order.pieces), 3)
        for name, rebuilt in (
            ("original", schedule),
            ("deepcopy", copy.deepcopy(schedule)),
            ("pickle", pickle.loads(pickle.dumps(schedule))),
        ):
            with self.subTest(roundtrip=name):
                cross_loop_scheduler._root_task_placement_relation.cache_clear()
                cross_loop_scheduler._root_schedule_traversal.cache_clear()
                cross_loop_scheduler.root_barrier_publication_plan.cache_clear()
                with mock.patch.object(
                    CoordinateRelation,
                    "materialize",
                    side_effect=AssertionError(
                        "schedule proof recovery must not enumerate runtime CTAs"
                    ),
                ):
                    self.assertTrue(
                        cross_loop_scheduler._validate_worker_schedule_tasks(
                            rebuilt,
                            task_orders,
                        )
                    )
                    for root in range(2):
                        placement = (
                            cross_loop_scheduler._root_task_placement_relation(
                                rebuilt,
                                root,
                            )
                        )
                        self.assertIsNotNone(placement)
                        assert placement is not None
                        self.assertTrue(placement.is_total_function())
                        publication = (
                            cross_loop_scheduler.root_barrier_publication_plan(
                                rebuilt,
                                root,
                            )
                        )
                        self.assertIsNotNone(publication.participant_order)
                        assert publication.participant_order is not None
                        self.assertTrue(
                            publication.participant_order
                            .is_bijection_from_source_support()
                        )

                for concrete_batch in (0, 1, 2, 8):
                    concrete_relations = tuple(
                        segment.task_order.substitute_parameters(
                            {batch: concrete_batch}
                        )
                        for segment in rebuilt.segments
                    )
                    prefix_targets = _singleton_targets(concrete_relations[0])
                    l2_targets = _singleton_targets(concrete_relations[1])
                    concrete_l2_domain = l2_domain.substitute_parameters(
                        {batch: concrete_batch}
                    )
                    self.assertEqual(
                        prefix_targets,
                        tuple(range(3 * concrete_batch)),
                    )
                    self.assertEqual(
                        l2_targets,
                        _expected_l2_targets(
                            concrete_l2_domain,
                            concrete_l2_domain.axis_order,
                            2,
                        ),
                    )
                    for relation, target_count in zip(
                        concrete_relations,
                        (3 * concrete_batch, 15 * concrete_batch),
                        strict=True,
                    ):
                        targets = _singleton_targets(relation)
                        self.assertEqual(len(targets), target_count)
                        self.assertEqual(len(set(targets)), target_count)
                        self.assertEqual(sorted(targets), list(range(target_count)))
                        _assert_exact_bijection(self, relation, target_count)

                    occupied_supports = tuple(
                        {
                            source_index
                            for source_index, targets in enumerate(
                                relation.materialize()
                            )
                            if targets
                        }
                        for relation in concrete_relations
                    )
                    self.assertTrue(
                        occupied_supports[0].isdisjoint(occupied_supports[1])
                    )

    def test_two_dynamic_axis_l2_survives_unaligned_packed_prefix(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        query = sympy.Symbol("query", integer=True, nonnegative=True)
        worker_count = 7
        prefix = _domain((1,), (3,), identity=0)
        l2_domain = _domain(
            (_FIRST_AXIS, _SECOND_AXIS, _BATCH_AXIS, _QUERY_AXIS),
            (5, 3, batch, query),
            identity=1,
        )
        task_orders = (
            pid_task_order(prefix, prefix.axis_order),
            pid_task_order(
                l2_domain,
                l2_domain.axis_order,
                l2_group_size=2,
            ),
        )
        schedule = cross_loop_scheduler._build_root_major_worker_schedule(
            (prefix, l2_domain),
            task_orders,
            worker_count,
        )

        self.assertEqual(len(schedule.segments), 2)
        self.assertLessEqual(len(schedule.segments[0].task_order.pieces), 3)
        self.assertLessEqual(len(schedule.segments[1].task_order.pieces), 3)
        for name, rebuilt in (
            ("original", schedule),
            ("deepcopy", copy.deepcopy(schedule)),
            ("pickle", pickle.loads(pickle.dumps(schedule))),
        ):
            with self.subTest(roundtrip=name):
                cross_loop_scheduler._root_task_placement_relation.cache_clear()
                cross_loop_scheduler._root_schedule_traversal.cache_clear()
                cross_loop_scheduler.root_barrier_publication_plan.cache_clear()
                with mock.patch.object(
                    CoordinateRelation,
                    "materialize",
                    side_effect=AssertionError(
                        "schedule proof recovery must not enumerate runtime CTAs"
                    ),
                ):
                    self.assertTrue(
                        cross_loop_scheduler._validate_worker_schedule_tasks(
                            rebuilt,
                            task_orders,
                        )
                    )
                    for root in range(2):
                        placement = (
                            cross_loop_scheduler._root_task_placement_relation(
                                rebuilt,
                                root,
                            )
                        )
                        self.assertIsNotNone(placement)
                        assert placement is not None
                        self.assertTrue(placement.is_total_function())
                        publication = (
                            cross_loop_scheduler.root_barrier_publication_plan(
                                rebuilt,
                                root,
                            )
                        )
                        self.assertIsNotNone(publication.participant_order)
                        assert publication.participant_order is not None
                        self.assertTrue(
                            publication.participant_order
                            .is_bijection_from_source_support()
                        )

                for concrete_batch, concrete_query in (
                    (0, 2),
                    (2, 0),
                    (1, 1),
                    (2, 3),
                ):
                    with self.subTest(
                        batch=concrete_batch,
                        query=concrete_query,
                    ):
                        substitutions = {
                            batch: concrete_batch,
                            query: concrete_query,
                        }
                        concrete_relations = tuple(
                            segment.task_order.substitute_parameters(substitutions)
                            for segment in rebuilt.segments
                        )
                        self.assertEqual(
                            _singleton_targets(concrete_relations[0]),
                            (0, 1, 2),
                        )
                        concrete_l2_domain = l2_domain.substitute_parameters(
                            substitutions
                        )
                        l2_targets = _singleton_targets(concrete_relations[1])
                        expected_l2_targets = _expected_l2_targets(
                            concrete_l2_domain,
                            concrete_l2_domain.axis_order,
                            2,
                        )
                        self.assertEqual(l2_targets, expected_l2_targets)
                        self.assertEqual(
                            len(l2_targets),
                            15 * concrete_batch * concrete_query,
                        )
                        _assert_exact_bijection(
                            self,
                            concrete_relations[1],
                            15 * concrete_batch * concrete_query,
                        )
                        occupied_supports = tuple(
                            {
                                source_index
                                for source_index, targets in enumerate(
                                    relation.materialize()
                                )
                                if targets
                            }
                            for relation in concrete_relations
                        )
                        self.assertTrue(
                            occupied_supports[0].isdisjoint(occupied_supports[1])
                        )

    def test_l2_edge_geometries_have_exact_constant_size_proofs(self) -> None:
        for first_count, second_count, group_size in (
            (0, 3, 2),
            (5, 0, 2),
            (4, 3, 2),
            (5, 3, 5),
            (5, 3, 8),
        ):
            with self.subTest(
                first_count=first_count,
                second_count=second_count,
                group_size=group_size,
            ):
                domain = _domain(
                    (_FIRST_AXIS, _SECOND_AXIS),
                    (first_count, second_count),
                    allow_empty=first_count == 0 or second_count == 0,
                )
                task_order = pid_task_order(
                    domain,
                    domain.axis_order,
                    l2_group_size=group_size,
                )
                expected = _expected_l2_targets(
                    domain,
                    domain.axis_order,
                    group_size,
                )
                self.assertEqual(_singleton_targets(task_order), expected)
                self.assertEqual(len(task_order.pieces), 0 if not expected else 1)
                with mock.patch.object(
                    CoordinateRelation,
                    "materialize",
                    side_effect=AssertionError("L2 proof must stay structural"),
                ):
                    converse = _assert_exact_bijection(
                        self,
                        task_order,
                        first_count * second_count,
                    )
                self.assertLessEqual(len(converse.pieces), 2)

    def test_l2_small_shape_oracle_and_large_shape_piece_bound(self) -> None:
        for first_count in range(1, 9):
            for second_count in range(1, 5):
                for group_size in sorted({1, 2, 3, first_count, first_count + 2}):
                    with self.subTest(
                        first_count=first_count,
                        second_count=second_count,
                        group_size=group_size,
                    ):
                        domain = _domain(
                            (_FIRST_AXIS, _SECOND_AXIS),
                            (first_count, second_count),
                        )
                        task_order = pid_task_order(
                            domain,
                            domain.axis_order,
                            l2_group_size=group_size,
                        )
                        self.assertEqual(len(task_order.pieces), 1)
                        self.assertEqual(
                            _singleton_targets(task_order),
                            _expected_l2_targets(
                                domain,
                                domain.axis_order,
                                group_size,
                            ),
                        )

        # This shape exceeded the historical group_count * second_count
        # witness limit. Its semantic representation is still constant-size.
        large_domain = _domain(
            (_FIRST_AXIS, _SECOND_AXIS),
            (4097, 3),
        )
        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("large L2 order must not enumerate"),
        ):
            large_order = pid_task_order(
                large_domain,
                large_domain.axis_order,
                l2_group_size=2,
            )
            self.assertEqual(len(large_order.pieces), 1)
            large_converse = _assert_exact_bijection(
                self,
                large_order,
                4097 * 3,
            )
        self.assertLessEqual(len(large_converse.pieces), 2)

    def test_ragged_l2_preserves_axis_permutation(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        domain = _domain(
            (_BATCH_AXIS, _FIRST_AXIS, _SECOND_AXIS),
            (batch, 5, 3),
        )
        pid_axis_order = (_FIRST_AXIS, _SECOND_AXIS, _BATCH_AXIS)
        task_order = pid_task_order(
            domain,
            pid_axis_order,
            l2_group_size=2,
        )

        self.assertEqual(len(task_order.pieces), 1)
        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("permuted symbolic proof must not enumerate"),
        ):
            _assert_exact_bijection(self, task_order, 15 * batch)
        for concrete_batch in (0, 1, 2, 8):
            concrete_domain = domain.substitute_parameters({batch: concrete_batch})
            concrete_order = task_order.substitute_parameters(
                {batch: concrete_batch}
            )
            expected = _expected_l2_targets(
                concrete_domain,
                pid_axis_order,
                2,
            )
            self.assertEqual(_singleton_targets(concrete_order), expected)
            self.assertEqual(sorted(expected), list(range(15 * concrete_batch)))

    def test_bounded_binary_floordiv_selector_is_generic(self) -> None:
        source = CoordinateDomain((1,), ((1, 6),), kind="task_order")
        target = CoordinateDomain((2,), ((2, 6),), identity=0)
        coordinate = coordinate_axis_symbol(1)
        selector = FloorDiv(coordinate, 3)
        selected = coordinate + selector * (8 - 2 * coordinate)
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((1, 0, 6, 1),),
                    (selected,),
                ),
            ),
        )

        self.assertEqual(_singleton_targets(relation), (0, 1, 2, 5, 4, 3))
        for name, rebuilt in (
            ("direct", relation),
            ("deepcopy", copy.deepcopy(relation)),
            ("pickle", pickle.loads(pickle.dumps(relation))),
        ):
            with self.subTest(roundtrip=name):
                with mock.patch.object(
                    CoordinateRelation,
                    "materialize",
                    side_effect=AssertionError(
                        "bounded selector proof must not enumerate"
                    ),
                ):
                    converse = _assert_exact_bijection(self, rebuilt, 6)
                self.assertLessEqual(len(converse.pieces), 2)
                self.assertEqual(
                    _singleton_targets(converse),
                    (0, 1, 2, 5, 4, 3),
                )

    def test_nonbinary_floordiv_selector_declines_conservatively(self) -> None:
        source = CoordinateDomain((1,), ((1, 9),), kind="task_order")
        target = CoordinateDomain((2,), ((2, 9),), identity=0)
        coordinate = coordinate_axis_symbol(1)
        selector = FloorDiv(coordinate, 3)
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((1, 0, 9, 1),),
                    (coordinate + 6 - 6 * selector,),
                ),
            ),
        )

        # This happens to be a permutation, but proving a three-way selector
        # is outside the bounded binary-selector grammar. Do not silently
        # expand it into one relation piece per source point.
        self.assertEqual(
            sorted(_singleton_targets(relation)),
            list(range(9)),
        )
        semantic_copy = CoordinateRelation(
            relation.source_domain,
            relation.target_domain,
            relation.pieces,
        )
        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("unsupported selectors must not enumerate"),
        ):
            self.assertIsNone(semantic_copy.converse())
            self.assertFalse(semantic_copy.is_bijection_from_source_support())
        self.assertEqual(len(semantic_copy.pieces), 1)
