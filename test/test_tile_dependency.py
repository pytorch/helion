from __future__ import annotations

import copy
import dataclasses
import itertools
import math
import pickle
import random
from typing import Any
from typing import Literal
from unittest import mock

import sympy
import torch
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import Max as SymbolicMax
from torch.utils._sympy.functions import Min as SymbolicMin

import helion
from helion import exc
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cross_loop_scheduler import WorkerScheduleSegment
from helion._compiler.device_ir_analysis import DeviceIRAnalysis
from helion._compiler.tile_dependency import AllocationRegion
from helion._compiler.tile_dependency import CoordinateDomain
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import ExecutionSite
from helion._compiler.tile_dependency import TaskAxis
from helion._compiler.tile_dependency import TaskFamily
from helion._compiler.tile_dependency import TileAccess
from helion._compiler.tile_dependency import TileDependency
from helion._compiler.tile_dependency import TileDependencyGraph
from helion._compiler.tile_dependency import TileDependencyKind
from helion._compiler.tile_dependency import _bounded_parameter_expression_interval
from helion._compiler.tile_dependency import _coalesce_adjacent_target_boxes
from helion._compiler.tile_dependency import _CoordinateRelationPiece
from helion._compiler.tile_dependency import _dense_linear_overlap_relation
from helion._compiler.tile_dependency import _dense_linear_source_support_interval
from helion._compiler.tile_dependency import _dense_mixed_radix_converse
from helion._compiler.tile_dependency import _is_provably_nonnegative
from helion._compiler.tile_dependency import _layout_is_injective
from helion._compiler.tile_dependency import _logical_expression_bounds
from helion._compiler.tile_dependency import _memoized_exact_converse
from helion._compiler.tile_dependency import (
    _piecewise_single_source_mixed_radix_converse,
)
from helion._compiler.tile_dependency import (
    _piecewise_source_grouped_mixed_radix_converse,
)
from helion._compiler.tile_dependency import _positive_integer_shift_is_nonnegative
from helion._compiler.tile_dependency import _relation_source_cells
from helion._compiler.tile_dependency import _remember_exact_converse
from helion._compiler.tile_dependency import _simplify_integer_quotients
from helion._compiler.tile_dependency import _simplify_logical_expression
from helion._compiler.tile_dependency import _symbolic_linear_access_relation
from helion._compiler.tile_dependency import _symbolic_producers_by_consumer
from helion._compiler.tile_dependency import _target_box_expression_extreme
from helion._compiler.tile_dependency import (
    _target_box_expression_extreme_proof_uncached,
)
from helion._compiler.tile_dependency import allocation_regions_may_overlap
from helion._compiler.tile_dependency import build_tile_dependency_graph
from helion._compiler.tile_dependency import coordinate_axis_symbol
from helion._compiler.tile_dependency import instantiate_coordinate_domains
from helion._compiler.tile_dependency import instantiate_symbolic_dependencies
from helion._compiler.tile_dependency import owner_roots_by_graph_id
from helion._compiler.tile_dependency import pid_task_order
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def cartesian_affine_stage(x: torch.Tensor) -> torch.Tensor:
    batch, width = x.size()
    out = torch.empty_like(x)

    for tile_batch, tile_width in hl.tile([batch, width]):
        out[tile_batch, tile_width] = x[tile_batch, tile_width] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def flattened_affine_stage(x: torch.Tensor) -> torch.Tensor:
    partial = torch.empty((8, 2, 4), device=x.device, dtype=x.dtype)
    partial_storage = partial.view(-1)
    out = torch.empty((2, 8), device=x.device, dtype=x.dtype)
    for tile_split, tile_group, tile_head in hl.tile([8, 2, 4], block_size=[1, 1, 4]):
        partial[tile_split, tile_group, tile_head] = x[tile_group, tile_head][
            None, :, :
        ]
    for tile_chunk, tile_head in hl.tile([2, 8], block_size=[1, 1]):
        split = tile_chunk.index[:, None] * 4 + hl.arange(4)[None, :]
        offsets = split[:, :, None] * 8 + tile_head.index[None, None, :]
        out[tile_chunk, tile_head] = torch.sum(partial_storage[offsets], dim=1)
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def flattened_qwen_attention_stages(x: torch.Tensor) -> torch.Tensor:
    partial = torch.empty((128, 2, 4, 128), device=x.device, dtype=x.dtype)
    partial_storage = partial.view(-1)
    chunk = torch.empty((16, 8, 128), device=x.device, dtype=x.dtype)
    chunk_storage = chunk.view(-1)
    out = torch.empty((8, 128), device=x.device, dtype=x.dtype)
    for tile_split, tile_group, tile_head in hl.tile([128, 2, 4], block_size=[1, 1, 4]):
        partial[tile_split, tile_group, tile_head, :] = x[tile_group, tile_head][
            None, :, :, None
        ]
    for tile_chunk, tile_head in hl.tile([16, 8], block_size=[1, 1]):
        split = tile_chunk.index[:, None] * 8 + hl.arange(8)[None, :]
        base = split[:, :, None] * 8 + tile_head.index[None, None, :]
        offsets = base[:, :, :, None] * 128 + hl.arange(128)[None, None, None, :]
        chunk[tile_chunk, tile_head, :] = torch.sum(partial_storage[offsets], dim=1)
    for tile_head in hl.tile(8, block_size=1):
        chunk_index = hl.arange(16)
        offsets = (chunk_index[:, None] * 8 + tile_head.index[None, :])[
            :, :, None
        ] * 128 + hl.arange(128)[None, None, :]
        out[tile_head, :] = torch.sum(chunk_storage[offsets], dim=0)
    return out


def _axis_geometry(
    root_domains: tuple[CoordinateDomain, ...],
) -> dict[int, tuple[int, int]]:
    return {
        axis: (domain.axis_counts[axis], domain.block_sizes[axis])
        for domain in root_domains
        for axis in domain.axis_order
    }


def _domain(
    *axis_specs: tuple[Any, ...],
    kind: Literal[
        "site", "allocation", "event", "task_order", "worker", "value"
    ] = "site",
    identity: int | None = None,
    allow_empty: bool = False,
) -> CoordinateDomain:
    """Build a test domain from compact ``(axis, count[, block])`` specs."""
    return CoordinateDomain(
        tuple(axis for axis, *_ in axis_specs),
        tuple((axis, count) for axis, count, *_ in axis_specs),
        tuple(
            (axis_spec[0], axis_spec[2])
            for axis_spec in axis_specs
            if len(axis_spec) == 3
        ),
        kind=kind,
        identity=identity,
        _allow_empty=allow_empty,
    )


def _configured_domains(
    graph,
    axis_geometry: dict[int, tuple[int, int]],
) -> tuple[tuple[CoordinateDomain, ...], tuple[CoordinateDomain | None, ...]]:
    configured_roots, site_domains = instantiate_coordinate_domains(
        graph,
        axis_geometry=axis_geometry,
    )
    assert all(domain is not None for domain in configured_roots)
    return (
        tuple(domain for domain in configured_roots if domain is not None),
        site_domains,
    )


def _access(
    access_id: int,
    *,
    root: int,
    allocation_id: int = 0,
    kind: Literal["load", "store"],
    shape: tuple[int | sympy.Expr, ...] = (128,),
    strides: tuple[int | sympy.Expr, ...] = (1,),
    block_ids: tuple[int | None, ...] = (0,),
    scales: tuple[int, ...] = (1,),
    offsets: tuple[int | None, ...] = (0,),
    scalar: tuple[bool, ...] | None = None,
    full_slice: tuple[bool, ...] | None = None,
    static_extents: tuple[int | None, ...] | None = None,
    masked: bool = False,
    tensor_name: str = "tmp",
    storage_offset: int | sympy.Expr = 0,
    layout_is_symbolically_exact: bool = True,
    affine_subscript_ranges=None,
) -> TileAccess:
    return TileAccess(
        access_id=access_id,
        memory_op_index=access_id,
        graph_id=root,
        root=root,
        allocation_id=allocation_id,
        kind=kind,
        tensor_name=tensor_name,
        tensor_shape=shape,
        tensor_strides=strides,
        storage_offset=storage_offset,
        subscript_dims=tuple(range(len(block_ids))),
        subscript_affine_block_ids=block_ids,
        subscript_index_scales=scales,
        subscript_offsets=offsets,
        subscript_is_scalar=scalar or tuple(False for _ in block_ids),
        has_explicit_mask=masked,
        subscript_is_full_slice=full_slice or tuple(False for _ in block_ids),
        subscript_static_extents=static_extents or (),
        layout_is_symbolically_exact=layout_is_symbolically_exact,
        affine_subscript_ranges=affine_subscript_ranges,
    )


def _root_producers_by_consumer(
    plan,
    root_domains: tuple[CoordinateDomain, ...],
    pair: tuple[int, int] = (0, 1),
) -> tuple[frozenset[int], ...] | None:
    axis_geometry = _axis_geometry(root_domains)
    configured_root_domains, site_domains = _configured_domains(plan, axis_geometry)
    relations = tuple(
        dependency.producers_by_consumer
        for dependency in instantiate_symbolic_dependencies(
            plan,
            root_domains=configured_root_domains,
            site_domains=site_domains,
        )
        if (dependency.producer_root, dependency.consumer_root) == pair
        and dependency.producer_site_id is None
        and dependency.consumer_site_id is None
    )
    if not relations or any(relation is None for relation in relations):
        return None
    concrete = tuple(relation for relation in relations if relation is not None)
    result = concrete[0]
    for relation in concrete[1:]:
        union = result.union(relation)
        if union is None:
            return None
        result = union
    return result.materialize(
        source_axis_order=root_domains[pair[1]].axis_order,
        target_axis_order=root_domains[pair[0]].axis_order,
    )


def _symbolic_root_relation(
    plan,
    axis_geometry: dict[int, tuple[int, int]],
    *,
    prove_nonnegative=None,
):
    root_domains, site_domains = _configured_domains(plan, axis_geometry)
    dependencies = instantiate_symbolic_dependencies(
        plan,
        root_domains=root_domains,
        site_domains=site_domains,
        prove_nonnegative=prove_nonnegative,
    )
    self_relations = tuple(
        dependency.producers_by_consumer
        for dependency in dependencies
        if dependency.producer_root == 0 and dependency.consumer_root == 1
    )
    assert len(self_relations) == 1
    return self_relations[0]


def _one_dimensional_domains(
    *,
    producer_count: int = 8,
    consumer_count: int = 8,
    producer_block: int = 16,
    consumer_block: int = 16,
) -> tuple[CoordinateDomain, CoordinateDomain]:
    return (
        _domain((10, producer_count, producer_block)),
        _domain((20, consumer_count, consumer_block)),
    )


def _bounded_coordinate_relation(size: int) -> CoordinateRelation:
    source = _domain((10, 2), (11, size), kind="site")
    target = _domain((20, 4), kind="allocation")
    retained = coordinate_axis_symbol(10)
    bounded = coordinate_axis_symbol(11)
    value = retained + 2 * sympy.Mod(sympy.floor(bounded / 64), 2)
    return _full_point_map(source, target, value)


_RelationBounds = tuple[int, Any, Any, int]
_PointMapPiece = tuple[
    tuple[_RelationBounds, ...],
    tuple[Any, ...],
]


def _piece(
    source_bounds: tuple[_RelationBounds, ...],
    *target_ranges: _RelationBounds,
) -> _CoordinateRelationPiece:
    return _CoordinateRelationPiece(source_bounds, target_ranges)


def _point_map(
    source: CoordinateDomain,
    target: CoordinateDomain,
    *pieces: _PointMapPiece,
) -> CoordinateRelation:
    """Build a point map without the otherwise repetitive outer tuple."""
    return CoordinateRelation.point_map(source, target, pieces)


def _relation(
    source: CoordinateDomain,
    target: CoordinateDomain,
    *pieces: _CoordinateRelationPiece,
) -> CoordinateRelation:
    """Build a relation without the otherwise repetitive outer tuple."""
    return CoordinateRelation(source, target, pieces)


def _full_point_map(
    source: CoordinateDomain,
    target: CoordinateDomain,
    *target_coordinates: sympy.Expr,
) -> CoordinateRelation:
    return _point_map(
        source,
        target,
        (
            tuple(
                (axis, 0, source.axis_count_expressions[axis], 1)
                for axis in source.axis_order
            ),
            target_coordinates,
        ),
    )


def _dependency_kinds(edge: TileDependency) -> frozenset[TileDependencyKind]:
    return frozenset(dependency.kind for dependency in edge.access_dependencies)


class TestTileDependency(TestCase):
    def test_unresolved_allocation_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            exc.CrossLoopSchedulingError,
            "allocation identity is unavailable",
        ):
            build_tile_dependency_graph(
                (_access(0, root=0, allocation_id=-1, kind="store"),),
                [[0], [1]],
            )

    def test_unresolved_allocation_is_allowed_across_source_phases(self) -> None:
        plan = build_tile_dependency_graph(
            (_access(0, root=0, allocation_id=-1, kind="store"),),
            [[0], [1]],
            root_phases=(0, 1),
        )

        self.assertEqual(plan.edges, ())

    def test_coordinate_domain_separates_geometry_from_linearization_order(
        self,
    ) -> None:
        domain = _domain((10, 2, 4), (20, 3, 8))
        self.assertEqual(domain.coordinates(3), {10: 1, 20: 1})
        self.assertEqual(
            domain.coordinates(3, linearization_order=(20, 10)),
            {10: 1, 20: 0},
        )
        self.assertEqual(domain.index({10: 1, 20: 1}), 3)
        self.assertEqual(
            domain.index({10: 1, 20: 0}, linearization_order=(20, 10)),
            3,
        )
        self.assertEqual(domain.size_expr, 6)
        self.assertEqual(domain.concrete_size, 6)
        with self.assertRaisesRegex(ValueError, "axis counts must be positive"):
            _domain((10, 0))
        unknown_sign = sympy.Symbol("unknown_sign", integer=True)
        with self.assertRaisesRegex(ValueError, "axis counts must be positive"):
            _domain((10, unknown_sign))
        noninteger = sympy.Symbol("noninteger", nonnegative=True)
        with self.assertRaisesRegex(ValueError, "must be an integer expression"):
            _domain((10, noninteger))

    def test_positive_integer_shift_proves_bounded_polynomials(self) -> None:
        batch = sympy.Symbol("batch", integer=True, positive=True)
        query = sympy.Symbol("query", integer=True, positive=True)

        for expression in (
            15 * batch * query - 15,
            15 * batch * query - 11,
            15 * batch * query - 15 * batch + 3,
        ):
            with self.subTest(expression=expression):
                self.assertTrue(_is_provably_nonnegative(expression, None))

        self.assertFalse(
            _is_provably_nonnegative(
                15 * batch * query - 15 * batch - 1,
                None,
            )
        )
        zero_capable_batch = sympy.Symbol(
            "zero_capable_batch",
            integer=True,
            nonnegative=True,
        )
        self.assertFalse(_is_provably_nonnegative(15 * zero_capable_batch - 15, None))

        accepted_domain = _domain((10, 15 * batch * query - 15))
        self.assertEqual(accepted_domain.size_expr, 15 * batch * query - 15)
        zero_domain = accepted_domain.substitute_parameters({batch: 1, query: 1})
        self.assertEqual(zero_domain.shape_expr, (0,))
        self.assertTrue(zero_domain._allow_empty)
        with self.assertRaisesRegex(ValueError, "axis counts must be positive"):
            _domain((10, 15 * zero_capable_batch - 15))
        with self.assertRaisesRegex(ValueError, "axis counts must be positive"):
            _domain((10, 15 * batch * query - 15 * batch - 1))

        with mock.patch(
            "helion._compiler.tile_dependency._relation_product_is_within_budget",
            return_value=False,
        ):
            self.assertFalse(
                _positive_integer_shift_is_nonnegative(
                    15 * batch * query - 11,
                )
            )
        with mock.patch.object(
            sympy,
            "Poly",
            side_effect=AssertionError("over-budget shift must not build a polynomial"),
        ):
            self.assertFalse(
                _positive_integer_shift_is_nonnegative((batch + query) ** 9)
            )
            ordinary_x = sympy.Symbol("ordinary_x", integer=True)
            ordinary_y = sympy.Symbol("ordinary_y", integer=True)
            self.assertFalse(
                _positive_integer_shift_is_nonnegative(
                    (batch + 1) * (ordinary_x + ordinary_y) ** 100_000
                )
            )

    def test_symbolic_coordinate_domain_substitution(self) -> None:
        width = 8
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)
        task_count = sympy.floor((extent + width - 1) / width)
        domain = _domain((10, task_count, width), (20, 2, 1))

        self.assertEqual(domain.size_expr, 2 * task_count)
        self.assertEqual(domain.parameter_symbols, frozenset((extent,)))
        with self.assertRaisesRegex(ValueError, "coordinate-domain size is symbolic"):
            _ = domain.concrete_size
        with self.assertRaisesRegex(ValueError, "coordinate-domain size is symbolic"):
            domain.coordinates(0)
        with self.assertRaisesRegex(
            ValueError,
            "coordinate-domain axis count is symbolic",
        ):
            domain.index({10: 0, 20: 0})

        for value in (0, 1, width - 1, width, width + 1):
            expected_count = (value + width - 1) // width
            concrete = domain.substitute_parameters({extent: value})
            self.assertEqual(concrete.axis_counts, {10: expected_count, 20: 2})
            self.assertEqual(concrete.size, 2 * expected_count)

        with self.assertRaisesRegex(
            ValueError,
            "missing coordinate-domain parameters: extent",
        ):
            domain.substitute_parameters({})

    def test_symbolic_coordinate_relation_substitution(self) -> None:
        width = 8
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)
        task_count = sympy.floor((extent + width - 1) / width)
        source = _domain((10, task_count), kind="task_order")
        target = _domain((20, task_count), kind="site")
        source_coordinate = coordinate_axis_symbol(10)
        relation = _relation(
            source,
            target,
            _piece(
                ((10, 0, task_count, 1),),
                (20, source_coordinate, source_coordinate + 1, 1),
            ),
        )

        self.assertEqual(relation.parameter_symbols, frozenset((extent,)))
        with self.assertRaisesRegex(ValueError, "coordinate-domain size is symbolic"):
            relation.materialize()
        with self.assertRaisesRegex(ValueError, "relation source bound is symbolic"):
            relation.target_coordinates({10: 0})
        with self.assertRaisesRegex(
            ValueError,
            "coordinate symbols are not parameters",
        ):
            relation.substitute_parameters({source_coordinate: 0, extent: width})

        for value in (0, 1, width - 1, width, width + 1):
            expected_count = (value + width - 1) // width
            concrete = relation.substitute_parameters({extent: value})
            if expected_count:
                self.assertEqual(
                    concrete.pieces[0].source_bounds_items,
                    ((10, 0, expected_count, 1),),
                )
            else:
                self.assertFalse(concrete.pieces)
            self.assertEqual(
                concrete.materialize(),
                tuple(frozenset((index,)) for index in range(expected_count)),
            )

    def test_relation_substitution_prunes_empty_conditional_piece(self) -> None:
        count = sympy.Symbol("count", integer=True, nonnegative=True)
        source = _domain((10, 4), kind="worker")
        target = _domain((20, 4), kind="site")
        coordinate = coordinate_axis_symbol(10)
        relation = _point_map(source, target, (((10, 0, count, 1),), (coordinate,)))

        empty = relation.substitute_parameters({count: 0})
        nonempty = relation.substitute_parameters({count: 3})

        self.assertFalse(empty.pieces)
        self.assertEqual(empty.materialize(), (frozenset(),) * 4)
        self.assertEqual(
            nonempty.materialize(),
            (
                frozenset((0,)),
                frozenset((1,)),
                frozenset((2,)),
                frozenset(),
            ),
        )

    def test_piecewise_dense_point_converse_is_exact_transpose(self) -> None:
        inner = coordinate_axis_symbol(10)
        outer = coordinate_axis_symbol(11)
        source = _domain((10, 4), (11, 3), kind="task_order")
        target = _domain((20, 12), kind="site")
        relation = _point_map(
            source,
            target,
            (((10, 0, 2, 1), (11, 0, 3, 1)), (2 * outer + inner,)),
            (((10, 2, 4, 1), (11, 0, 3, 1)), (2 * outer + inner + 4,)),
        )

        converse = relation.converse()
        assert converse is not None
        expected: list[set[int]] = [set() for _ in range(target.size)]
        for source_index, target_indices in enumerate(relation.materialize()):
            for target_index in target_indices:
                expected[target_index].add(source_index)
        self.assertEqual(
            converse.materialize(),
            tuple(frozenset(indices) for indices in expected),
        )
        derived, target_counts = relation.derive_converse_and_target_counts()
        self.assertIsNotNone(derived)
        self.assertIsNotNone(target_counts)
        assert derived is not None
        self.assertEqual(derived.materialize(), converse.materialize())

    def test_adjacent_static_offset_point_maps_fold_exactly(self) -> None:
        inner = coordinate_axis_symbol(10)
        outer = coordinate_axis_symbol(11)
        source = _domain((10, 16), (11, 3), kind="task_order")
        target = _domain((20, 48), kind="site")
        relation = _point_map(
            source,
            target,
            (((10, 0, 8, 1), (11, 0, 3, 1)), (8 * outer + inner,)),
            (((10, 8, 16, 1), (11, 0, 3, 1)), (8 * outer + inner + 16,)),
        )

        self.assertEqual(len(relation.coalesce_adjacent_source_boxes().pieces), 2)
        compact = relation.coalesce_adjacent_source_boxes(fold_static_offsets=True)

        self.assertEqual(len(compact.pieces), 1)
        self.assertTrue(compact.is_total_function())
        self.assertTrue(compact.is_pointwise_equal_to(relation))
        self.assertEqual(compact.materialize(), relation.materialize())

        many_source = _domain((10, 8), (11, 2), kind="task_order")
        many_target = _domain((20, 16), kind="site")
        ordered_pieces = tuple(
            _piece(
                ((10, 2 * part, 2 * part + 2, 1), (11, 0, 2, 1)),
                (20, 2 * outer + inner + 2 * part, 2 * outer + inner + 2 * part + 1, 1),
            )
            for part in range(4)
        )
        for permutation in (
            tuple(reversed(ordered_pieces)),
            tuple(ordered_pieces[index] for index in (2, 0, 3, 1)),
        ):
            shuffled = CoordinateRelation(many_source, many_target, permutation)
            shuffled_compact = shuffled.coalesce_adjacent_source_boxes(
                fold_static_offsets=True
            )
            self.assertEqual(len(shuffled_compact.pieces), 1)
            self.assertTrue(shuffled_compact.is_pointwise_equal_to(shuffled))

        unequal = _point_map(
            _domain((10, 15), (11, 3), kind="task_order"),
            target,
            (((10, 0, 8, 1), (11, 0, 3, 1)), (8 * outer + inner,)),
            (((10, 8, 15, 1), (11, 0, 3, 1)), (8 * outer + inner + 16,)),
        )
        self.assertEqual(
            len(
                unequal.coalesce_adjacent_source_boxes(fold_static_offsets=True).pieces
            ),
            2,
        )

    def test_piecewise_point_converse_declines_non_dense_layout(self) -> None:
        inner = coordinate_axis_symbol(10)
        outer = coordinate_axis_symbol(11)
        source = _domain((10, 2), (11, 3), kind="task_order")
        target = _domain((20, 8), kind="site")
        relation = _full_point_map(source, target, 3 * outer + inner)

        self.assertIsNone(relation.converse())

    def test_piecewise_mixed_radix_point_converse_is_exact_transpose(self) -> None:
        ordinal = coordinate_axis_symbol(10)
        source = _domain((10, 12), kind="task_order")
        target = _domain((20, 2), (21, 3), (22, 2), kind="site")

        def relation(second_batch: int) -> CoordinateRelation:
            return _point_map(
                source,
                target,
                (
                    ((10, 0, 6, 1),),
                    (sympy.Integer(0), sympy.Mod(ordinal, 3), sympy.floor(ordinal / 3)),
                ),
                (
                    ((10, 6, 12, 1),),
                    (
                        sympy.Integer(second_batch),
                        sympy.Mod(ordinal, 3),
                        sympy.floor(ordinal / 3) - 2,
                    ),
                ),
            )

        bijection = relation(1)
        converse = bijection.converse()
        assert converse is not None
        expected: list[set[int]] = [set() for _ in range(target.size)]
        for source_index, target_indices in enumerate(bijection.materialize()):
            for target_index in target_indices:
                expected[target_index].add(source_index)
        self.assertEqual(
            converse.materialize(),
            tuple(frozenset(indices) for indices in expected),
        )

        # Mapping the second source interval onto the first interval's target
        # box would make the inverse multi-valued, so this specialized proof
        # must decline it.
        self.assertIsNone(relation(0).converse())

    def test_mixed_radix_stride_inference_handles_reordered_digits(self) -> None:
        ordinal = coordinate_axis_symbol(10)
        source = _domain((10, 256), kind="task_order")
        target = CoordinateDomain(
            tuple(range(20, 28)),
            tuple((axis, 2) for axis in range(20, 28)),
            kind="site",
        )
        reordered = _full_point_map(
            source,
            target,
            sympy.floor(ordinal / 128),
            sympy.Mod(sympy.floor(ordinal / 64), 2),
            sympy.Mod(sympy.floor(ordinal / 32), 2),
            sympy.Mod(sympy.floor(ordinal / 16), 2),
            sympy.Mod(sympy.floor(ordinal / 8), 2),
            sympy.Mod(sympy.floor(ordinal / 4), 2),
            sympy.Mod(sympy.floor(ordinal / 2), 2),
            sympy.Mod(ordinal, 2),
        )

        inner = coordinate_axis_symbol(10)
        outer = coordinate_axis_symbol(11)
        grouped_source = _domain((10, 6), (11, 4), kind="task_order")
        grouped_target = _domain((20, 3), (21, 2), (22, 2), (23, 2), kind="site")
        grouped = _full_point_map(
            grouped_source,
            grouped_target,
            sympy.floor(inner / 2),
            sympy.Mod(inner, 2),
            sympy.floor(outer / 2),
            sympy.Mod(outer, 2),
        )

        with mock.patch.object(
            itertools,
            "permutations",
            side_effect=AssertionError("mixed-radix proof must not search orders"),
        ):
            converses = (
                _piecewise_single_source_mixed_radix_converse(reordered),
                _piecewise_source_grouped_mixed_radix_converse(grouped),
            )
        for relation, converse in zip(
            (reordered, grouped),
            converses,
            strict=True,
        ):
            self.assertIsNotNone(converse)
            assert converse is not None
            expected: list[set[int]] = [
                set() for _ in range(relation.target_domain.size)
            ]
            for source_index, target_indices in enumerate(relation.materialize()):
                for target_index in target_indices:
                    expected[target_index].add(source_index)
            self.assertEqual(
                converse.materialize(),
                tuple(frozenset(indices) for indices in expected),
            )

    def test_mixed_radix_stride_inference_declines_invalid_chains(self) -> None:
        ordinal = coordinate_axis_symbol(10)

        def relation(
            source_count: int,
            target_counts: tuple[int, int],
            expressions: tuple[sympy.Expr, sympy.Expr],
        ) -> CoordinateRelation:
            return _point_map(
                _domain((10, source_count), kind="task_order"),
                CoordinateDomain(
                    (20, 21),
                    tuple(zip((20, 21), target_counts, strict=True)),
                    kind="site",
                ),
                (((10, 0, source_count, 1),), expressions),
            )

        ambiguous = relation(
            4,
            (2, 2),
            (sympy.Mod(ordinal, 2), sympy.Mod(ordinal, 2)),
        )
        non_mixed_radix = relation(
            6,
            (2, 3),
            (sympy.Mod(ordinal + 1, 2), sympy.floor(ordinal / 2)),
        )
        for malformed in (ambiguous, non_mixed_radix):
            with self.subTest(relation=malformed):
                self.assertIsNone(
                    _piecewise_single_source_mixed_radix_converse(malformed)
                )
                self.assertIsNone(
                    _piecewise_source_grouped_mixed_radix_converse(malformed)
                )

    def test_woven_mixed_radix_converse_is_exact_and_symbolic(self) -> None:
        ordinal = coordinate_axis_symbol(10)
        source = _domain((10, 704), kind="task_order")
        target = _domain((20, 2), (21, 8), (22, 44), kind="site")
        relation = _full_point_map(
            source,
            target,
            sympy.Mod(sympy.floor(ordinal / 32), 2),
            sympy.Mod(ordinal, 8),
            4 * sympy.floor(ordinal / 64) + sympy.floor(sympy.Mod(ordinal, 32) / 8),
        )

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("converse proof must remain symbolic"),
        ):
            converse = relation.converse()
        self.assertIsNotNone(converse)
        assert converse is not None
        self.assertTrue(converse.is_total_function())

        expected: list[set[int]] = [set() for _ in range(target.size)]
        for source_index, target_indices in enumerate(relation.materialize()):
            for target_index in target_indices:
                expected[target_index].add(source_index)
        self.assertEqual(
            converse.materialize(),
            tuple(frozenset(indices) for indices in expected),
        )

    def test_reflected_woven_mixed_radix_converse_is_exact(self) -> None:
        ordinal = coordinate_axis_symbol(10)
        source = _domain((10, 1536), kind="task_order")
        target = _domain((20, 1536), kind="site")
        reflected_bit = sympy.floor(sympy.Mod(ordinal, 16) / 8)
        relation = _full_point_map(
            source,
            target,
            sympy.Mod(ordinal, 8)
            + 8 * sympy.floor(ordinal / 16)
            + 768 * (1 - reflected_bit),
        )

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("converse proof must remain symbolic"),
        ):
            converse = relation.converse()
        self.assertIsNotNone(converse)
        assert converse is not None
        self.assertTrue(converse.is_total_function())

        # The preceding woven test checks a complete concrete transpose. Keep
        # this larger Qwen-shaped case symbolic and sample every discontinuity
        # introduced by reflection instead of materializing all 1,536 points.
        for source_index in (0, 7, 8, 15, 16, 767, 768, 775, 776, 1535):
            (target_index,) = relation.targets(source_index)
            self.assertEqual(converse.targets(target_index), frozenset((source_index,)))

        malformed = _full_point_map(
            source,
            target,
            sympy.Mod(ordinal, 8)
            + 8 * sympy.floor(ordinal / 16)
            + 767 * (1 - reflected_bit),
        )
        self.assertIsNone(malformed.converse())

    def test_woven_mixed_radix_converse_rejects_reused_input_digit(self) -> None:
        ordinal = coordinate_axis_symbol(10)
        source = _domain((10, 8), kind="task_order")
        target = _domain((20, 2), (21, 4), kind="site")
        noninjective = _full_point_map(
            source, target, sympy.Mod(ordinal, 2), sympy.Mod(ordinal, 4)
        )

        self.assertIsNone(noninjective.converse())

    def test_relation_axis_renaming_preserves_positional_coordinates(self) -> None:
        source = _domain((10, 2), (20, 3))
        target = _domain((30, 2), (40, 3))
        source_10 = coordinate_axis_symbol(10)
        source_20 = coordinate_axis_symbol(20)
        relation = _full_point_map(
            source,
            target,
            sympy.Mod(source_10 + source_20, 2),
            sympy.floor((source_10 + 2 * source_20) / 2),
        )
        renamed_source = _domain((20, 2, 4), (10, 3, 8), kind="task_order", identity=7)
        renamed_target = _domain((40, 2), (30, 3), kind="event", identity=4)

        renamed = relation.rename_source_axes(renamed_source)
        assert renamed is not None
        renamed = renamed.rename_target_axes(renamed_target)
        assert renamed is not None

        self.assertEqual(renamed.source_domain, renamed_source)
        self.assertEqual(renamed.target_domain, renamed_target)
        self.assertEqual(renamed.materialize(), relation.materialize())
        self.assertIsNone(relation.rename_source_axes(_domain((0, 2), (1, 4))))
        self.assertIsNone(relation.rename_target_axes(_domain((0, 2), (1, 4))))

    def test_configured_roots_reuse_their_execution_site_domains(self) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        graph = dataclasses.replace(
            graph,
            execution_sites=(
                ExecutionSite(0, 0, 0, (), None, "root", (), (10,), True, False),
                ExecutionSite(1, 1, 1, (), None, "root", (), (20,), True, False),
            ),
            site_ids_by_access=((0,), (1,)),
        )

        root_domains, site_domains = instantiate_coordinate_domains(
            graph,
            axis_geometry={10: (8, 16), 20: (4, 32)},
        )

        root_sites = tuple(site for site in graph.execution_sites if site.is_root)
        self.assertEqual(len(root_sites), 2)
        for site in root_sites:
            self.assertIs(root_domains[site.root], site_domains[site.site_id])

    def test_symbolic_geometry_instantiates_coordinate_domains(self) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        graph = dataclasses.replace(
            graph,
            execution_sites=(
                ExecutionSite(0, 0, 0, (), None, "root", (), (10,), True, False),
                ExecutionSite(1, 1, 1, (), None, "root", (), (20,), True, False),
            ),
            site_ids_by_access=((0,), (1,)),
        )
        task_count = sympy.Symbol("task_count", integer=True, nonnegative=True)

        root_domains, site_domains = instantiate_coordinate_domains(
            graph,
            axis_geometry={10: (task_count, 16), 20: (task_count + 1, 32)},
        )

        self.assertEqual(root_domains[0].size_expr, task_count)
        self.assertEqual(root_domains[1].size_expr, task_count + 1)
        self.assertIs(root_domains[0], site_domains[0])
        self.assertIs(root_domains[1], site_domains[1])
        self.assertEqual(
            root_domains[0].substitute_parameters({task_count: 0}).size,
            0,
        )
        zero_roots, _zero_sites = instantiate_coordinate_domains(
            graph,
            axis_geometry={10: (0, 16), 20: (1, 32)},
        )
        self.assertIsNone(zero_roots[0])

    def test_symbolic_positional_bijection_derives_inverse_and_unit_fan_in(
        self,
    ) -> None:
        task_count = sympy.Symbol("task_count", integer=True, nonnegative=True)
        key_domain = _domain((0, task_count), kind="event", identity=0)
        producer_domain = _domain((10, task_count, 16), kind="site", identity=0)
        relation = _full_point_map(
            key_domain, producer_domain, coordinate_axis_symbol(0)
        )

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic proof must not enumerate"),
        ):
            converse, target_counts = relation.derive_converse_and_target_counts()
            self.assertTrue(relation.is_positional_bijection())
            self.assertTrue(relation.is_total_function())
            self.assertIs(relation.canonical_single_valued(), relation)
            self.assertIsNotNone(converse)
            self.assertIsNotNone(target_counts)
            assert converse is not None and target_counts is not None
            self.assertTrue(converse.is_positional_bijection())
            self.assertEqual(target_counts.constant_value(), 1)

        for concrete_count in (0, 1, 3, 4, 5):
            concrete = relation.substitute_parameters({task_count: concrete_count})
            concrete_converse = converse.substitute_parameters(
                {task_count: concrete_count}
            )
            expected = tuple(frozenset((index,)) for index in range(concrete_count))
            self.assertEqual(concrete.materialize(), expected)
            self.assertEqual(concrete_converse.materialize(), expected)

    def test_symbolic_fixed_width_partition_derives_inverse_and_fan_in(
        self,
    ) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        key_domain = _domain((0, key_count), kind="event", identity=0)
        producer_domain = _domain((10, 2 * key_count, 16), kind="site", identity=0)
        key = coordinate_axis_symbol(0)
        relation = _relation(
            key_domain,
            producer_domain,
            _piece(((0, 0, key_count, 1),), (10, 2 * key, 2 * key + 2, 1)),
        )

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic proof must not enumerate"),
        ):
            converse, target_counts = relation.derive_converse_and_target_counts()
            self.assertIsNotNone(converse)
            self.assertIsNotNone(target_counts)
            assert converse is not None and target_counts is not None
            self.assertEqual(converse, relation.converse())
            self.assertEqual(target_counts.constant_value(), 2)

        for concrete_count in (0, 1, 3, 5):
            concrete = relation.substitute_parameters({key_count: concrete_count})
            concrete_converse = converse.substitute_parameters(
                {key_count: concrete_count}
            )
            self.assertEqual(
                concrete.materialize(),
                tuple(
                    frozenset((2 * key_index, 2 * key_index + 1))
                    for key_index in range(concrete_count)
                ),
            )
            self.assertEqual(
                concrete_converse.materialize(),
                tuple(
                    frozenset((producer_index // 2,))
                    for producer_index in range(2 * concrete_count)
                ),
            )

    def test_symbolic_total_relation_derives_inverse_and_target_count(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        key_domain = _domain((0, 1), kind="event", identity=0)
        producer_domain = _domain(
            (10, 2 * batch), kind="site", identity=0, allow_empty=True
        )
        relation = CoordinateRelation.total(key_domain, producer_domain)

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic proof must not enumerate"),
        ):
            converse, target_counts = relation.derive_converse_and_target_counts()

        self.assertIsNotNone(converse)
        self.assertIsNotNone(target_counts)
        assert converse is not None and target_counts is not None
        self.assertEqual(converse, relation.converse())
        for concrete_batch in (0, 1, 3):
            with self.subTest(concrete_batch=concrete_batch):
                substitutions = {batch: concrete_batch}
                self.assertEqual(
                    converse.substitute_parameters(substitutions).materialize(),
                    (frozenset((0,)),) * (2 * concrete_batch),
                )
                self.assertEqual(
                    target_counts.substitute_parameters(substitutions).materialize(),
                    (frozenset((2 * concrete_batch,)),),
                )

    def test_symbolic_repeated_fiber_producer_set_quotient(self) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        consumers_per_key = 16
        producers_per_key = 4
        consumer_domain = _domain((20, consumers_per_key * key_count), kind="site")
        producer_domain = _domain((10, producers_per_key * key_count), kind="site")
        consumer = coordinate_axis_symbol(20)
        producer_begin = producers_per_key * sympy.floor(consumer / consumers_per_key)
        relation = _relation(
            consumer_domain,
            producer_domain,
            _piece(
                ((20, 0, consumers_per_key * key_count, 1),),
                (10, producer_begin, producer_begin + producers_per_key, 1),
            ),
        )

        with (
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError("symbolic proof must not enumerate"),
            ),
            mock.patch.object(
                CoordinateRelation,
                "targets",
                side_effect=AssertionError("symbolic proof must not enumerate"),
            ),
        ):
            quotient = relation.producer_set_quotient()
            assert quotient is not None
            keys_by_consumer, producers_by_key = quotient
            self.assertEqual(keys_by_consumer.target_domain.size_expr, key_count)
            self.assertTrue(keys_by_consumer.is_total_function())
            _publication, producer_count = (
                producers_by_key.derive_converse_and_target_counts()
            )
            self.assertIsNotNone(producer_count)
            assert producer_count is not None
            self.assertEqual(producer_count.constant_value(), producers_per_key)

        for concrete_count in (0, 1, 3):
            concrete_keys = keys_by_consumer.substitute_parameters(
                {key_count: concrete_count}
            )
            concrete_producers = producers_by_key.substitute_parameters(
                {key_count: concrete_count}
            )
            self.assertEqual(
                concrete_keys.materialize(),
                tuple(
                    frozenset((consumer_index // consumers_per_key,))
                    for consumer_index in range(consumers_per_key * concrete_count)
                ),
            )
            self.assertEqual(
                concrete_producers.materialize(),
                tuple(
                    frozenset(
                        range(
                            producers_per_key * key_index,
                            producers_per_key * (key_index + 1),
                        )
                    )
                    for key_index in range(concrete_count)
                ),
            )

    def test_symbolic_multi_axis_producer_set_quotient(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        consumer_domain = _domain((20, batch), (21, 6), (22, 5), kind="site")
        producer_domain = _domain((10, batch), (11, 4), (12, 3), kind="site")
        consumer_batch = coordinate_axis_symbol(20)
        consumer_group = coordinate_axis_symbol(21)
        producer_group = 2 * sympy.floor(consumer_group / 3)
        relation = _relation(
            consumer_domain,
            producer_domain,
            _piece(
                ((20, 0, batch, 1), (21, 0, 6, 1), (22, 0, 5, 1)),
                (10, consumer_batch, consumer_batch + 1, 1),
                (11, producer_group, producer_group + 2, 1),
                (12, sympy.Integer(0), sympy.Integer(3), 1),
            ),
        )

        with (
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError("symbolic proof must not enumerate"),
            ),
            mock.patch.object(
                CoordinateRelation,
                "targets",
                side_effect=AssertionError("symbolic proof must not enumerate"),
            ),
        ):
            quotient = relation.producer_set_quotient()
            assert quotient is not None
            keys_by_consumer, producers_by_key = quotient
            self.assertEqual(keys_by_consumer.target_domain.shape_expr, (batch, 2))
            self.assertTrue(keys_by_consumer.is_total_function())
            publication, producer_count = (
                producers_by_key.derive_converse_and_target_counts()
            )
            self.assertIsNotNone(publication)
            self.assertIsNotNone(producer_count)
            assert publication is not None and producer_count is not None
            self.assertTrue(publication.is_total_function())
            self.assertEqual(producer_count.constant_value(), 6)

        for concrete_batch in (1, 2, 9):
            concrete_keys = keys_by_consumer.substitute_parameters(
                {batch: concrete_batch}
            )
            concrete_producers = producers_by_key.substitute_parameters(
                {batch: concrete_batch}
            )
            expected_keys: list[frozenset[int]] = []
            for consumer_index in range(
                consumer_domain.size_expr.subs(batch, concrete_batch)
            ):
                consumer_coordinates = consumer_domain.substitute_parameters(
                    {batch: concrete_batch}
                ).coordinates(consumer_index)
                expected_keys.append(
                    frozenset(
                        (
                            consumer_coordinates[20]
                            + concrete_batch * (consumer_coordinates[21] // 3),
                        )
                    )
                )
            self.assertEqual(concrete_keys.materialize(), tuple(expected_keys))

            concrete_producer_domain = producer_domain.substitute_parameters(
                {batch: concrete_batch}
            )
            expected_producers: list[frozenset[int]] = []
            for key_index in range(2 * concrete_batch):
                batch_index = key_index % concrete_batch
                group_index = key_index // concrete_batch
                expected_producers.append(
                    frozenset(
                        concrete_producer_domain.index(
                            {10: batch_index, 11: producer_group_index, 12: tail}
                        )
                        for producer_group_index in range(
                            2 * group_index, 2 * group_index + 2
                        )
                        for tail in range(3)
                    )
                )
            self.assertEqual(
                concrete_producers.materialize(), tuple(expected_producers)
            )

    def test_symbolic_contiguous_layout_derives_multi_axis_dependencies(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        producer_domain = _domain((10, batch, 1), (11, 4, 16), kind="site")
        consumer_domain = _domain((20, batch, 1), (21, 2, 32), kind="site")
        producer = _access(
            0,
            root=0,
            kind="store",
            shape=(batch, 64),
            strides=(64, 1),
            block_ids=(10, 11),
            scales=(1, 1),
            offsets=(0, 0),
        )
        consumer = _access(
            1,
            root=1,
            kind="load",
            shape=(batch, 64),
            strides=(64, 1),
            block_ids=(20, 21),
            scales=(1, 1),
            offsets=(0, 0),
        )

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic proof must not enumerate"),
        ):
            relation = _symbolic_producers_by_consumer(
                producer_access=producer,
                producer_domain=producer_domain,
                consumer_access=consumer,
                consumer_domain=consumer_domain,
            )
            assert relation is not None
            quotient = relation.producer_set_quotient()
            assert quotient is not None
            keys_by_consumer, producers_by_key = quotient
            self.assertTrue(keys_by_consumer.is_total_function())
            publication, target_counts = (
                producers_by_key.derive_converse_and_target_counts()
            )
            self.assertIsNotNone(publication)
            self.assertIsNotNone(target_counts)
            assert target_counts is not None
            self.assertEqual(target_counts.constant_value(), 2)

        self.assertTrue(_layout_is_injective(((batch, 64), (64, 1), 0)))
        self.assertFalse(_layout_is_injective(((batch, 64), (32, 1), 0)))
        for concrete_batch in (1, 2, 9):
            concrete = relation.substitute_parameters({batch: concrete_batch})
            self.assertEqual(
                concrete.materialize(),
                tuple(
                    frozenset(
                        batch_index + concrete_batch * (2 * column + inner_column)
                        for inner_column in range(2)
                    )
                    for column in range(2)
                    for batch_index in range(concrete_batch)
                ),
            )

    def test_tile_access_canonicalizes_layout_expressions(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        access = _access(
            0,
            root=0,
            kind="store",
            shape=(batch, 64),
            strides=(64, 1),
            storage_offset=0,
            masked=True,
        )

        self.assertEqual(access.tensor_shape, (batch, sympy.Integer(64)))
        self.assertTrue(
            all(isinstance(value, sympy.Expr) for value in access.tensor_shape)
        )
        self.assertTrue(
            all(isinstance(value, sympy.Expr) for value in access.tensor_strides)
        )
        self.assertIsInstance(access.storage_offset, sympy.Integer)
        # A mask makes the access relation unsupported, not its tensor layout.
        self.assertTrue(access.layout_is_symbolically_exact)

    def test_symbolic_flat_view_matches_multidimensional_layout(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        producer_domain = _domain((10, batch, 1), (11, 2, 4), kind="site")
        consumer_domain = _domain((20, batch, 1), (21, 2, 1), kind="site")
        producer = _access(
            0,
            root=0,
            kind="store",
            shape=(batch, 8),
            strides=(8, 1),
            block_ids=(10, 11),
            scales=(1, 1),
            offsets=(0, 0),
        )
        consumer = _access(
            1,
            root=1,
            kind="load",
            shape=(8 * batch,),
            strides=(1,),
            block_ids=(None,),
            affine_subscript_ranges=((((20, 8, 1), (21, 4, 1)), 0, 4, 1),),
        )

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic proof must not enumerate"),
        ):
            relation = _symbolic_producers_by_consumer(
                producer_access=producer,
                producer_domain=producer_domain,
                consumer_access=consumer,
                consumer_domain=consumer_domain,
            )
            self.assertIsNotNone(relation)

        assert relation is not None
        for concrete_batch in (1, 2, 9):
            concrete = relation.substitute_parameters({batch: concrete_batch})
            self.assertEqual(
                concrete.materialize(),
                tuple(frozenset((task,)) for task in range(2 * concrete_batch)),
            )

    def test_symbolic_affine_access_declines_symbolic_target_stride(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        stride = sympy.Symbol("stride", integer=True, positive=True)
        source_domain = _domain((10, batch), kind="site", identity=0)
        allocation_domain = _domain((-1, batch * stride), kind="allocation", identity=0)
        access = _access(
            0,
            root=0,
            kind="load",
            shape=(batch,),
            strides=(stride,),
            block_ids=(None,),
            affine_subscript_ranges=(((((10, 1, 1),), 0, 1, 1)),),
        )

        self.assertIsNone(
            _symbolic_linear_access_relation(
                access,
                source_domain=source_domain,
                allocation_domain=allocation_domain,
            )
        )

    def test_symbolic_positional_product_lifts_static_tail_proof(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        key_domain = _domain((0, batch), (1, 2), kind="event")
        producer_domain = _domain((10, batch), (11, 3), kind="site")
        key_batch = coordinate_axis_symbol(0)
        relation = _relation(
            key_domain,
            producer_domain,
            _piece(
                ((0, 0, batch, 1), (1, 0, 1, 1)),
                (10, key_batch, key_batch + 1, 1),
                (11, sympy.Integer(0), sympy.Integer(2), 1),
            ),
            _piece(
                ((0, 0, batch, 1), (1, 1, 2, 1)),
                (10, key_batch, key_batch + 1, 1),
                (11, sympy.Integer(2), sympy.Integer(3), 1),
            ),
        )

        with (
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError("symbolic proof must not enumerate"),
            ),
            mock.patch.object(
                CoordinateRelation,
                "targets",
                side_effect=AssertionError("symbolic proof must not enumerate"),
            ),
        ):
            converse, target_counts = relation.derive_converse_and_target_counts()
            self.assertIsNotNone(converse)
            self.assertIsNotNone(target_counts)
            assert converse is not None and target_counts is not None
            self.assertTrue(converse.is_total_function())
            self.assertIsNotNone(converse.canonical_single_valued())
            self.assertIsNone(target_counts.constant_value())
            self.assertEqual(target_counts.value_bounds(), (1, 2))

        for concrete_batch in (1, 2, 9):
            substitutions = {batch: concrete_batch}
            concrete_relation = relation.substitute_parameters(substitutions)
            concrete_converse = converse.substitute_parameters(substitutions)
            concrete_counts = target_counts.substitute_parameters(substitutions)
            self.assertEqual(
                concrete_relation.materialize(),
                tuple(
                    frozenset(
                        batch_index + concrete_batch * producer_inner
                        for producer_inner in (
                            range(2) if key_inner == 0 else range(2, 3)
                        )
                    )
                    for key_inner in range(2)
                    for batch_index in range(concrete_batch)
                ),
            )
            self.assertEqual(
                concrete_converse.materialize(),
                tuple(
                    frozenset((batch_index + concrete_batch * (producer_inner // 2),))
                    for producer_inner in range(3)
                    for batch_index in range(concrete_batch)
                ),
            )
            self.assertEqual(
                concrete_counts.materialize(),
                tuple(
                    frozenset((2 if key_inner == 0 else 1,))
                    for key_inner in range(2)
                    for _batch_index in range(concrete_batch)
                ),
            )

    def test_symbolic_value_bounds_preserve_correlated_minimum(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((0, batch), (1, 20), kind="event")
        value = _domain((0, 33), kind="value")
        column = coordinate_axis_symbol(1)
        counts = _full_point_map(
            source,
            value,
            16 * sympy.Max(0, -2 * column + sympy.Min(39, 2 * column + 2)),
        )

        self.assertEqual(counts.value_bounds(), (16, 32))

    def test_symbolic_repeated_fiber_quotient_rejects_inexact_forms(self) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        consumer = coordinate_axis_symbol(20)
        cases = (
            (16 * key_count + 1, 4 * key_count, 4 * sympy.floor(consumer / 16), 1),
            (16 * key_count, 4 * key_count + 1, 4 * sympy.floor(consumer / 16), 1),
            (16 * key_count, 4 * key_count, 4 * sympy.floor(consumer / 16) + 1, 1),
            (16 * key_count, 4 * key_count, 4 * sympy.floor(consumer / 16), 2),
        )
        for source_count, target_count, begin, step in cases:
            with self.subTest(
                source_count=source_count,
                target_count=target_count,
                begin=begin,
                step=step,
            ):
                relation = _relation(
                    _domain((20, source_count), kind="site"),
                    _domain((10, target_count), kind="site"),
                    _piece(((20, 0, source_count, 1),), (10, begin, begin + 4, step)),
                )
                self.assertIsNone(relation.producer_set_quotient())

        fractional_key = _relation(
            _domain((20, 16 * key_count), kind="site"),
            _domain((10, 8), kind="site"),
            _piece(
                ((20, 0, 16 * key_count, 1),),
                (10, sympy.Integer(1), sympy.Integer(5), 1),
            ),
        )
        self.assertIsNone(fractional_key.producer_set_quotient())

    def test_symbolic_nonpartition_relations_do_not_derive_fixed_fan_in(
        self,
    ) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        key_domain = _domain((0, key_count), kind="event")
        key = coordinate_axis_symbol(0)
        cases = (
            (2 * key_count + 1, 2 * key, 2 * key + 2, 1),
            (2 * key_count, 2 * key + 1, 2 * key + 3, 1),
            (2 * key_count, 2 * key, 2 * key + 2, 2),
        )
        for target_count, begin, end, step in cases:
            with self.subTest(
                target_count=target_count,
                begin=begin,
                end=end,
                step=step,
            ):
                producer_domain = _domain((10, target_count), kind="site")
                relation = _relation(
                    key_domain,
                    producer_domain,
                    _piece(((0, 0, key_count, 1),), (10, begin, end, step)),
                )
                converse, target_counts = relation.derive_converse_and_target_counts()
                self.assertIsNone(converse)
                self.assertIsNone(target_counts)

        symbolic_fan_in = sympy.Symbol("fan_in", integer=True, positive=True)
        symbolic_producer_domain = _domain(
            (10, symbolic_fan_in * key_count), kind="site"
        )
        symbolic_width = _relation(
            key_domain,
            symbolic_producer_domain,
            _piece(
                ((0, 0, key_count, 1),),
                (10, symbolic_fan_in * key, symbolic_fan_in * (key + 1), 1),
            ),
        )
        self.assertEqual(
            symbolic_width.derive_converse_and_target_counts(),
            (None, None),
        )

        producer_domain = _domain((10, 2 * key_count), kind="site")
        duplicate_pieces = _relation(
            key_domain,
            producer_domain,
            _piece(((0, 0, key_count, 1),), (10, 2 * key, 2 * key + 2, 1)),
            _piece(((0, 0, key_count, 1),), (10, 2 * key, 2 * key + 2, 1)),
        )
        self.assertEqual(
            duplicate_pieces.derive_converse_and_target_counts(),
            (None, None),
        )

    def test_unproved_symbolic_point_maps_decline_totality(self) -> None:
        task_count = sympy.Symbol("task_count", integer=True, nonnegative=True)
        source = _domain((10, task_count), kind="site")
        target = _domain((20, task_count), kind="event")
        coordinate = coordinate_axis_symbol(10)

        for expression in (
            coordinate + 1,
            sympy.Integer(0),
            sympy.Mod(coordinate, 2),
        ):
            with self.subTest(expression=expression):
                relation = _full_point_map(source, target, expression)
                self.assertFalse(relation.is_total_function())
                self.assertIsNone(relation.converse())

    def test_l2_pid_task_order_preflights_relation_budget(self) -> None:
        domain = _domain((10, 3, 1), (20, 2, 1))

        for budget_name in (
            "_MAX_RELATION_PIECES",
            "_MAX_RELATION_PRODUCT_STATES",
        ):
            with (
                self.subTest(budget_name=budget_name),
                mock.patch(
                    f"helion._compiler.tile_dependency.{budget_name}",
                    1,
                ),
                mock.patch.object(
                    CoordinateRelation,
                    "point_map",
                    side_effect=AssertionError("L2 construction must preflight"),
                ),
                self.assertRaisesRegex(ValueError, "relation budget"),
            ):
                pid_task_order(domain, domain.axis_order, l2_group_size=2)

    def test_binary_floor_selector_does_not_certify_duplicate_targets(self) -> None:
        source = _domain((10, 4), identity=0)
        target = _domain((20, 4), identity=1)
        coordinate = coordinate_axis_symbol(10)
        relation = _full_point_map(source, target, FloorDiv(coordinate, 2))

        converse = relation.converse()
        assert converse is not None
        self.assertFalse(converse.is_total_function())
        self.assertFalse(relation.is_bijection_from_source_support())

    def test_full_source_composition_substitutes_directly(self) -> None:
        source = _domain((10, 3), identity=0)
        intermediate = _domain((20, 3), identity=1)
        target = _domain((30, 3), identity=2)
        source_coordinate = coordinate_axis_symbol(10)
        intermediate_coordinate = coordinate_axis_symbol(20)
        first = _full_point_map(source, intermediate, source_coordinate)
        following = _full_point_map(intermediate, target, 2 - intermediate_coordinate)
        self.assertIsNotNone(first.converse())
        with mock.patch(
            "helion._compiler.tile_dependency._substitute_composed_expression",
            side_effect=AssertionError("full support should substitute directly"),
        ):
            composed = first.then(following)
        self.assertIsNotNone(composed)
        assert composed is not None
        self.assertEqual(
            composed.materialize(),
            (frozenset((2,)), frozenset((1,)), frozenset((0,))),
        )

    def test_composition_clips_semantically_empty_intermediate_boxes(self) -> None:
        source = _domain((10, 3), identity=0)
        intermediate = _domain((20, 3), identity=1)
        target = _domain((30, 1), identity=2)
        source_coordinate = coordinate_axis_symbol(10)
        first = _full_point_map(source, intermediate, source_coordinate + 3)
        following = _relation(
            intermediate,
            target,
            _piece(((20, 3, 6, 1),), (30, sympy.Integer(0), sympy.Integer(1), 1)),
        )

        composed = first.then(following)

        assert composed is not None
        self.assertFalse(composed.pieces)
        self.assertEqual(
            composed.materialize(),
            (frozenset(), frozenset(), frozenset()),
        )

    def test_partial_identity_composes_with_full_set_relation(self) -> None:
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)

        def relations(
            count: int | sympy.Expr,
        ) -> tuple[CoordinateRelation, CoordinateRelation]:
            source = _domain((10, count), kind="site", allow_empty=True)
            intermediate = _domain((10, count), kind="worker", allow_empty=True)
            target = _domain((30, count), kind="value", allow_empty=True)
            coordinate = coordinate_axis_symbol(10)
            return (
                _point_map(
                    source, intermediate, (((10, -3, count + 4, 2),), (coordinate,))
                ),
                _relation(
                    intermediate,
                    target,
                    _piece(
                        ((10, 0, count, 1),), (30, sympy.Integer(0), coordinate + 1, 1)
                    ),
                ),
            )

        first, following = relations(extent)
        composed = first.then(following)
        assert composed is not None
        for concrete_extent in (0, 1, 2, 5, 8):
            with self.subTest(extent=concrete_extent):
                direct_first, direct_following = relations(concrete_extent)
                direct = direct_first.then(direct_following)
                assert direct is not None
                expected = tuple(
                    frozenset(range(source_index + 1))
                    if source_index % 2 == 1
                    else frozenset()
                    for source_index in range(concrete_extent)
                )
                self.assertEqual(
                    composed.substitute_parameters(
                        {extent: concrete_extent}
                    ).materialize(),
                    expected,
                )
                self.assertEqual(direct.materialize(), expected)

    def test_mixed_radix_readiness_quotient_derives_publication(self) -> None:
        for slots in (1, 2, 8, 64):
            with self.subTest(slots=slots):
                consumer_domain = _domain((20, slots, 1), (21, 8, 256), identity=1)
                producer_domain = _domain((10, slots * 256, 16), identity=0)
                readiness_key_domain = dataclasses.replace(
                    consumer_domain,
                    kind="event",
                    identity=None,
                )
                slot = coordinate_axis_symbol(20)
                activation_block = coordinate_axis_symbol(21)
                begin = 256 * slot + 16 * activation_block
                bounds = ((20, 0, slots, 1), (21, 0, 8, 1))
                dependency = _relation(
                    consumer_domain,
                    producer_domain,
                    _piece(bounds, (10, begin, begin + 16, 1)),
                    _piece(bounds, (10, begin + 128, begin + 144, 1)),
                )
                consumer_to_key = CoordinateRelation.projection(
                    consumer_domain,
                    readiness_key_domain,
                )
                assert consumer_to_key is not None

                producers_by_key = dependency.factor_through(consumer_to_key)

                assert producers_by_key is not None
                self.assertEqual(len(producers_by_key.pieces), 2)
                arrival_count_by_key = producers_by_key.target_count_by_source()
                assert arrival_count_by_key is not None
                self.assertEqual(arrival_count_by_key.constant_value(), 32)
                keys_by_producer = producers_by_key.converse()
                assert keys_by_producer is not None
                self.assertEqual(len(keys_by_producer.pieces), 1)
                self.assertTrue(keys_by_producer.is_total_function())
                for producer in (0, 15, 16, 127, 128, 255, slots * 256 - 1):
                    self.assertEqual(
                        keys_by_producer.target_coordinates({10: producer}),
                        frozenset(
                            (
                                (
                                    producer // 256,
                                    producer // 16 % 8,
                                ),
                            )
                        ),
                    )

    def test_mixed_radix_partial_periodic_support_keeps_semantics(self) -> None:
        slots = 8
        readiness_key_domain = _domain((20, slots), (21, 8), kind="event")
        producer_domain = _domain((10, slots * 256), identity=0)
        slot = coordinate_axis_symbol(20)
        activation_block = coordinate_axis_symbol(21)
        begin = 256 * slot + 16 * activation_block
        producers_by_key = _relation(
            readiness_key_domain,
            producer_domain,
            _piece(((20, 0, slots, 1), (21, 0, 8, 1)), (10, begin, begin + 16, 1)),
        )

        arrival_count_by_key = producers_by_key.target_count_by_source()

        assert arrival_count_by_key is not None
        self.assertEqual(arrival_count_by_key.constant_value(), 16)
        self.assertIsNone(producers_by_key.converse())

    def test_mixed_radix_converse_matches_reversed_axis_relation(self) -> None:
        source_domain = _domain((21, 3), (20, 2), kind="site")
        target_domain = _domain((10, 24), kind="allocation")
        inner = coordinate_axis_symbol(21)
        outer = coordinate_axis_symbol(20)
        begin = 2 * inner + 12 * outer
        bounds = ((21, 0, 3, 1), (20, 0, 2, 1))
        targets_by_source = _relation(
            source_domain,
            target_domain,
            _piece(bounds, (10, begin, begin + 2, 1)),
            _piece(bounds, (10, begin + 6, begin + 8, 1)),
        )

        sources_by_target = targets_by_source.converse()

        assert sources_by_target is not None
        expected = {
            target: frozenset(
                source
                for source, targets in enumerate(targets_by_source.materialize())
                if target in targets
            )
            for target in range(target_domain.size)
        }
        self.assertEqual(
            sources_by_target.materialize(),
            tuple(expected[target] for target in range(target_domain.size)),
        )

    def test_converse_normalizes_bounded_shifted_modulo(self) -> None:
        producer = _domain((10, 684), identity=0)
        keys = _domain((0, 3), kind="event", identity=0)
        source = coordinate_axis_symbol(10)
        key = sympy.Mod(sympy.floor(source / 16) + 2, 3)
        producer_to_key = _relation(
            producer,
            keys,
            _piece(((10, 400, 448, 1),), (0, key, key + 1, 1)),
        )

        producers_by_key = producer_to_key.converse()

        assert producers_by_key is not None
        self.assertEqual(
            producers_by_key.materialize(),
            tuple(
                frozenset(
                    range(
                        400 + key_index * 16,
                        400 + (key_index + 1) * 16,
                    )
                )
                for key_index in range(3)
            ),
        )

    def test_converse_of_modulo_projection_is_exact(self) -> None:
        source = _domain((10, 32), (11, 3), kind="task_order")
        keys = _domain((20, 16), kind="event")
        source_inner = coordinate_axis_symbol(10)
        source_to_key = _full_point_map(source, keys, sympy.Mod(source_inner, 16))

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("converse proof must remain symbolic"),
        ):
            keys_to_source = source_to_key.converse()

        self.assertIsNotNone(keys_to_source)
        assert keys_to_source is not None
        expected: list[set[int]] = [set() for _ in range(keys.size)]
        for source_index, key_indices in enumerate(source_to_key.materialize()):
            for key_index in key_indices:
                expected[key_index].add(source_index)
        self.assertEqual(
            keys_to_source.materialize(),
            tuple(frozenset(indices) for indices in expected),
        )

    def test_converse_of_grouped_mixed_radix_task_order_is_exact(self) -> None:
        order = _domain((10, 16), (11, 3), kind="task_order")
        tasks = _domain((20, 3), (21, 2), (22, 4), (23, 2), kind="site")
        inner = coordinate_axis_symbol(10)
        outer = coordinate_axis_symbol(11)
        task_order = _point_map(
            order,
            tasks,
            (
                ((10, 0, 6, 1), (11, 0, 3, 1)),
                (outer, sympy.Integer(0), sympy.Mod(inner, 3), sympy.floor(inner / 3)),
            ),
            (
                ((10, 6, 12, 1), (11, 0, 3, 1)),
                (
                    outer,
                    sympy.Integer(1),
                    sympy.Mod(inner, 3),
                    sympy.Mod(sympy.floor(inner / 3), 2),
                ),
            ),
            (
                ((10, 12, 14, 1), (11, 0, 3, 1)),
                (outer, sympy.Integer(0), sympy.Integer(3), inner - 12),
            ),
            (
                ((10, 14, 16, 1), (11, 0, 3, 1)),
                (outer, sympy.Integer(1), sympy.Integer(3), inner - 14),
            ),
        )

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("converse proof must remain symbolic"),
        ):
            tasks_to_order = task_order.converse()

        self.assertIsNotNone(tasks_to_order)
        assert tasks_to_order is not None
        self.assertTrue(tasks_to_order.is_total_function())
        expected: list[set[int]] = [set() for _ in range(tasks.size)]
        for order_index, task_indices in enumerate(task_order.materialize()):
            for task_index in task_indices:
                expected[task_index].add(order_index)
        self.assertEqual(
            tasks_to_order.materialize(),
            tuple(frozenset(indices) for indices in expected),
        )

    def test_project_source_folds_only_small_bounded_constants(self) -> None:
        small = _bounded_coordinate_relation(64)
        large = _bounded_coordinate_relation(65)
        retained = _domain((10, 2), kind="site")

        projected = small.project_source(retained)

        assert projected is not None
        self.assertEqual(
            projected.materialize(),
            (frozenset((0,)), frozenset((1,))),
        )
        self.assertIsNone(large.project_source(retained))

    def test_project_target_preserves_symbolic_axis_counts(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((10, batch), (11, 4), kind="site")
        target = _domain((20, batch), (21, 4), kind="site")
        retained = _domain((20, batch), kind="site")
        relation = _full_point_map(
            source, target, coordinate_axis_symbol(10), coordinate_axis_symbol(11)
        )

        projected = relation.project_target(retained)

        self.assertIsNotNone(projected)
        self.assertTrue(CoordinateRelation.total(source, target).covers(relation))
        assert projected is not None
        self.assertEqual(
            projected.then(CoordinateRelation.identity(retained, retained)),
            projected,
        )
        concrete = projected.substitute_parameters({batch: 3})
        self.assertEqual(
            concrete.materialize(),
            tuple(frozenset((index % 3,)) for index in range(12)),
        )

    def test_symbolic_set_relation_composes_with_identity(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((10, batch), kind="site")
        target = _domain((20, 4), kind="event")
        relation = CoordinateRelation.total(source, target)

        self.assertIs(
            relation.then(CoordinateRelation.identity(target, target)),
            relation,
        )

    def test_point_composition_uses_piece_bounds_for_modulo(self) -> None:
        order = _domain((10, 8), kind="task_order")
        logical = _domain((20, 1), (21, 8), kind="site")
        key = _domain((30, 2), kind="event")
        ordinal = coordinate_axis_symbol(10)
        task_order = _point_map(
            order,
            logical,
            (((10, 0, 6, 1),), (sympy.Integer(0), sympy.Mod(ordinal, 6))),
            (((10, 6, 8, 1),), (sympy.Integer(0), sympy.Mod(ordinal, 6) + 6)),
        )
        keys_by_task = _point_map(
            logical,
            key,
            (((20, 0, 1, 1), (21, 0, 6, 1)), (sympy.Integer(0),)),
            (((20, 0, 1, 1), (21, 6, 8, 1)), (sympy.Integer(1),)),
        )

        composed = task_order.then(keys_by_task)

        assert composed is not None
        self.assertEqual(
            composed.materialize(),
            (
                frozenset((0,)),
                frozenset((0,)),
                frozenset((0,)),
                frozenset((0,)),
                frozenset((0,)),
                frozenset((0,)),
                frozenset((1,)),
                frozenset((1,)),
            ),
        )

    def test_project_source_keeps_symbolic_outer_axis_and_unions_static_inner(
        self,
    ) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        source = _domain((10, key_count), (11, 4), kind="site")
        retained = _domain((10, key_count), kind="site")
        producer = _domain((20, 4 * key_count), kind="site")
        outer = coordinate_axis_symbol(10)
        inner = coordinate_axis_symbol(11)
        relation = _full_point_map(source, producer, 4 * outer + inner)

        projected = relation.project_source(retained)

        assert projected is not None
        self.assertEqual(
            projected.pieces,
            (_piece(((10, 0, key_count, 1),), (20, 4 * outer, 4 * outer + 4, 1)),),
        )
        for concrete_count in (0, 1, 5):
            concrete = projected.substitute_parameters({key_count: concrete_count})
            self.assertEqual(
                concrete.materialize(),
                tuple(
                    frozenset(range(4 * key, 4 * key + 4))
                    for key in range(concrete_count)
                ),
            )

    def test_project_source_unions_runtime_sized_positional_factor(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((10, batch), (11, 3), kind="site")
        retained = _domain((11, 3), kind="site")
        target = _domain((20, batch), (21, 3), kind="site")
        inner = coordinate_axis_symbol(11)
        relation = _full_point_map(source, target, coordinate_axis_symbol(10), inner)

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("runtime positional factor must stay symbolic"),
        ):
            projected = relation.project_source(retained)

        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertEqual(
            projected.pieces,
            (
                _piece(
                    ((11, 0, 3, 1),),
                    (20, sympy.Integer(0), batch, 1),
                    (21, inner, inner + 1, 1),
                ),
            ),
        )
        for concrete_batch in (0, 1, 5):
            concrete = projected.substitute_parameters({batch: concrete_batch})
            self.assertEqual(
                concrete.materialize(),
                tuple(
                    frozenset(
                        batch_index + concrete_batch * inner_index
                        for batch_index in range(concrete_batch)
                    )
                    for inner_index in range(3)
                ),
            )

    def test_project_source_exact_nested_c4_and_rejects_nonrectangular_union(
        self,
    ) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        source = _domain((10, key_count), (11, 4), kind="site")
        retained = _domain((10, key_count), kind="site")
        producer = _domain((20, 16 * key_count), kind="site")
        key = coordinate_axis_symbol(10)
        split = coordinate_axis_symbol(11)
        begin = 16 * key + 4 * split
        relation = _relation(
            source,
            producer,
            _piece(((10, 0, key_count, 1), (11, 0, 4, 1)), (20, begin, begin + 4, 1)),
        )

        projected = relation.project_source(retained)

        assert projected is not None
        self.assertEqual(
            projected.pieces,
            (_piece(((10, 0, key_count, 1),), (20, 16 * key, 16 * key + 16, 1)),),
        )

        gapped = dataclasses.replace(
            relation,
            pieces=(
                _piece(
                    ((10, 0, key_count, 1), (11, 0, 4, 1)), (20, begin, begin + 2, 1)
                ),
            ),
        )
        self.assertIsNone(gapped.project_source(retained))

        diagonal_target = _domain(
            (20, 16 * key_count), (21, 16 * key_count), kind="site"
        )
        diagonal = _relation(
            source,
            diagonal_target,
            _piece(
                ((10, 0, key_count, 1), (11, 0, 4, 1)),
                (20, begin, begin + 4, 1),
                (21, begin, begin + 4, 1),
            ),
        )
        self.assertIsNone(diagonal.project_source(retained))

        dynamic_inner = sympy.Symbol("dynamic_inner", integer=True, nonnegative=True)
        dynamic_source = _domain((10, key_count), (11, dynamic_inner), kind="site")
        dynamic_relation = dataclasses.replace(
            relation,
            source_domain=dynamic_source,
            pieces=(
                _piece(
                    ((10, 0, key_count, 1), (11, 0, dynamic_inner, 1)),
                    (20, begin, begin + 4, 1),
                ),
            ),
        )
        self.assertIsNone(dynamic_relation.project_source(retained))

    def test_symbolic_fixed_capacity_requires_explicit_bound_proof(self) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(32,),
                    strides=(1,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(16, 2),
                    strides=(2, 1),
                    block_ids=(20, None),
                    offsets=(0, None),
                    full_slice=(False, True),
                ),
            ),
            [[10], [20]],
        )
        geometry = {10: (2 * key_count, 1), 20: (key_count, 1)}

        self.assertIsNone(_symbolic_root_relation(plan, geometry))

        proved_expressions: list[sympy.Expr] = []

        def prove_under_key_bound(expression: sympy.Expr) -> bool:
            proved_expressions.append(expression)
            polynomial = sympy.Poly(expression, key_count)
            return polynomial.degree() <= 1 and all(
                sympy.sympify(expression.subs(key_count, value)).is_nonnegative is True
                for value in (0, 16)
            )

        relation = _symbolic_root_relation(
            plan,
            geometry,
            prove_nonnegative=prove_under_key_bound,
        )

        assert relation is not None
        self.assertIn(32 - 2 * key_count, proved_expressions)
        self.assertEqual(
            relation.substitute_parameters({key_count: 1}).materialize(),
            (frozenset((0, 1)),),
        )
        self.assertIsNone(
            _symbolic_root_relation(
                plan,
                geometry,
                prove_nonnegative=lambda _expression: False,
            )
        )

    def test_dynamic_strided_overlap_eliminates_bounded_modulo(self) -> None:
        key_count = sympy.Symbol("key_count", integer=True, positive=True)
        producer = _domain((10, 64 * key_count), kind="site")
        consumer = _domain((20, 16 * key_count), (21, 8), kind="site")
        allocation = _domain((-1, 16384), kind="allocation", identity=0)
        producer_id = coordinate_axis_symbol(10)
        consumer_id = coordinate_axis_symbol(20)
        group = coordinate_axis_symbol(21)
        producer_access = _relation(
            producer,
            allocation,
            _piece(
                ((10, 0, 64 * key_count, 1),),
                (-1, 16 * producer_id, 16 * producer_id + 16, 1),
            ),
        )
        consumer_begin = (
            consumer_id + 128 * group + 1008 * sympy.floor(consumer_id / 16)
        )
        consumer_access = _relation(
            consumer,
            allocation,
            _piece(
                ((20, 0, 16 * key_count, 1), (21, 0, 8, 1)),
                (-1, consumer_begin, consumer_begin + 128, 16),
            ),
        )

        def prove_under_capacity(expression: sympy.Expr) -> bool:
            expression = sympy.simplify(expression)
            if expression.is_nonnegative is True:
                return True
            polynomial = sympy.Poly(expression, key_count)
            return polynomial.degree() <= 1 and all(
                sympy.sympify(expression.subs(key_count, value)).is_nonnegative is True
                for value in (1, 16)
            )

        relation = _dense_linear_overlap_relation(
            producer_access,
            consumer_access,
            prove_nonnegative=prove_under_capacity,
        )

        assert relation is not None
        expected_begin = 8 * group + 64 * sympy.floor(consumer_id / 16)
        self.assertEqual(
            relation.pieces,
            (
                _piece(
                    ((20, 0, 16 * key_count, 1), (21, 0, 8, 1)),
                    (10, expected_begin, expected_begin + 8, 1),
                ),
            ),
        )
        projected = relation.project_source(_domain((20, 16 * key_count), kind="site"))
        assert projected is not None
        expected_root_begin = 64 * sympy.floor(consumer_id / 16)
        self.assertEqual(
            projected.pieces,
            (
                _piece(
                    ((20, 0, 16 * key_count, 1),),
                    (10, expected_root_begin, expected_root_begin + 64, 1),
                ),
            ),
        )

    def test_dynamic_overlap_coalesces_and_prunes_disjoint_chunks(self) -> None:
        key_count = sympy.Symbol("key_count", integer=True, positive=True)
        producer = _domain((10, 64 * key_count), kind="site")
        consumer = _domain((20, 16 * key_count), (21, 8), kind="site")
        allocation = _domain((-1, 8 * 1024 * 1024), kind="allocation", identity=0)
        producer_id = coordinate_axis_symbol(10)
        consumer_id = coordinate_axis_symbol(20)
        group = coordinate_axis_symbol(21)
        producer_access = _relation(
            producer,
            allocation,
            _piece(
                ((10, 0, 64 * key_count, 1),),
                (-1, 2048 * producer_id, 2048 * producer_id + 2048, 1),
            ),
        )

        def prove_under_capacity(expression: sympy.Expr) -> bool:
            expression = sympy.simplify(expression)
            if expression.is_nonnegative is True:
                return True
            polynomial = sympy.Poly(expression, key_count)
            return polynomial.degree() <= 1 and all(
                sympy.sympify(expression.subs(key_count, value)).is_nonnegative is True
                for value in (1, 16)
            )

        def consumer_access(chunk_offset: int) -> CoordinateRelation:
            base = (
                chunk_offset
                + 128 * consumer_id
                + 129024 * sympy.floor(consumer_id / 16)
                + 16384 * group
            )
            return CoordinateRelation(
                consumer,
                allocation,
                tuple(
                    _piece(
                        ((20, 0, 16 * key_count, 1), (21, 0, 8, 1)),
                        (-1, base + 2048 * lane, base + 2048 * lane + 128, 1),
                    )
                    for lane in range(8)
                ),
            )

        matching = producer_access.overlapping_sources(
            consumer_access(0),
            prove_nonnegative=prove_under_capacity,
        )
        disjoint = producer_access.overlapping_sources(
            consumer_access(2_097_152),
            prove_nonnegative=prove_under_capacity,
        )

        self.assertIsNotNone(matching)
        assert matching is not None
        expected_begin = 8 * group + 64 * sympy.floor(consumer_id / 16)
        self.assertEqual(
            matching.pieces,
            (
                _piece(
                    ((20, 0, 16 * key_count, 1), (21, 0, 8, 1)),
                    (10, expected_begin, expected_begin + 8, 1),
                ),
            ),
        )
        self.assertIsNotNone(disjoint)
        assert disjoint is not None
        self.assertEqual(disjoint.pieces, ())

    def test_overlap_respects_partial_strided_producer_support(self) -> None:
        producer = _domain((10, 8), kind="site")
        consumer = _domain((20, 6), kind="site")
        allocation = _domain((-1, 10), kind="allocation", identity=0)
        producer_coordinate = coordinate_axis_symbol(10)
        consumer_coordinate = coordinate_axis_symbol(20)
        consumer_access = _relation(
            consumer,
            allocation,
            _piece(
                ((20, 0, 6, 1),),
                (-1, consumer_coordinate - 1, consumer_coordinate + 2, 1),
            ),
        )

        for producer_bounds, supported_producers in (
            ((10, 1, 8, 2), (1, 3, 5, 7)),
            # Source membership remains anchored at the raw negative begin.
            # Clipping must not shift this odd lattice onto even coordinates.
            ((10, -3, 12, 2), (1, 3, 5, 7)),
            ((10, -4, 13, 3), (2, 5)),
        ):
            with self.subTest(producer_bounds=producer_bounds):
                producer_access = _relation(
                    producer,
                    allocation,
                    _piece(
                        (producer_bounds,),
                        (-1, producer_coordinate, producer_coordinate + 1, 1),
                    ),
                )
                overlap = producer_access.overlapping_sources(consumer_access)

                assert overlap is not None
                self.assertEqual(
                    overlap.materialize(),
                    tuple(
                        frozenset(
                            producer_index
                            for producer_index in supported_producers
                            if consumer_index - 1 <= producer_index < consumer_index + 2
                        )
                        for consumer_index in range(consumer.size)
                    ),
                )

                disjoint_consumer = _relation(
                    consumer,
                    allocation,
                    _piece(
                        ((20, 0, 6, 1),), (-1, sympy.Integer(8), sympy.Integer(9), 1)
                    ),
                )
                disjoint = producer_access.overlapping_sources(disjoint_consumer)
                assert disjoint is not None
                self.assertEqual(
                    disjoint.materialize(),
                    (frozenset(),) * consumer.size,
                )

        empty = _relation(producer, allocation)
        empty_overlap = empty.overlapping_sources(consumer_access)
        assert empty_overlap is not None
        self.assertEqual(
            empty_overlap.materialize(),
            (frozenset(),) * consumer.size,
        )

    def test_partial_overlap_accepts_proved_unclipped_constant_points(self) -> None:
        extent = sympy.Symbol(
            "constant_overlap_extent",
            integer=True,
            nonnegative=True,
        )
        wave_count = FloorDiv(extent + 3, 4)
        producer = _domain((10, 4), (11, wave_count), kind="worker", allow_empty=True)
        consumer = _domain((20, 3), kind="worker")
        key = _domain((30, 1), kind="event")
        producer_keys = _point_map(
            producer,
            key,
            (
                (
                    (10, 0, sympy.Min(4, extent), 1),
                    (11, 0, sympy.Min(1, wave_count), 1),
                ),
                (sympy.Integer(0),),
            ),
        )
        consumer_keys = _full_point_map(consumer, key, sympy.Integer(0))

        overlap = producer_keys.overlapping_sources(consumer_keys)

        assert overlap is not None
        for concrete_extent in (0, 1, 3, 4, 5, 9):
            substitutions = {extent: concrete_extent}
            concrete_producer = producer_keys.substitute_parameters(substitutions)
            concrete_consumer = consumer_keys.substitute_parameters(substitutions)
            concrete_overlap = overlap.substitute_parameters(substitutions)
            expected = tuple(
                frozenset(
                    producer_index
                    for producer_index, producer_targets in enumerate(
                        concrete_producer.materialize()
                    )
                    if producer_targets & consumer_targets
                )
                for consumer_targets in concrete_consumer.materialize()
            )
            self.assertEqual(concrete_overlap.materialize(), expected)

        consumer_coordinate = coordinate_axis_symbol(20)
        clipped_consumer_keys = _full_point_map(consumer, key, consumer_coordinate - 1)
        self.assertIsNone(producer_keys.overlapping_sources(clipped_consumer_keys))

        invalid_keys = _point_map(
            producer,
            key,
            (
                (
                    (10, 0, sympy.Min(4, extent), 1),
                    (11, 0, sympy.Min(1, wave_count), 1),
                ),
                (sympy.Integer(1),),
            ),
        )
        self.assertIsNone(invalid_keys.overlapping_sources(consumer_keys))

    def test_partial_constant_point_overlap_clipped_consumer_differential(
        self,
    ) -> None:
        generator = random.Random(17)
        accepted = 0
        clipped_declines = 0
        for case in range(128):
            producer_count = generator.randrange(2, 8)
            consumer_count = generator.randrange(2, 6)
            key_count = 1
            producer_begin = generator.randrange(0, producer_count)
            producer_end = generator.randrange(producer_begin + 1, producer_count + 1)
            consumer_is_clipped = bool(generator.randrange(2))
            producer = _domain((10, producer_count), kind="worker")
            consumer = _domain((20, consumer_count), kind="worker")
            key = _domain((30, key_count), kind="event", identity=case)
            consumer_coordinate = coordinate_axis_symbol(20)
            producer_keys = _point_map(
                producer,
                key,
                (((10, producer_begin, producer_end, 1),), (sympy.Integer(0),)),
            )
            consumer_keys = _full_point_map(
                consumer,
                key,
                consumer_coordinate - 1 if consumer_is_clipped else sympy.Integer(0),
            )

            overlap = producer_keys.overlapping_sources(consumer_keys)
            producer_points = producer_keys.materialize()
            consumer_points = consumer_keys.materialize()
            expected = tuple(
                frozenset(
                    producer_index
                    for producer_index, producer_targets in enumerate(producer_points)
                    if producer_targets & consumer_targets
                )
                for consumer_targets in consumer_points
            )
            if overlap is None:
                if consumer_is_clipped:
                    clipped_declines += 1
                continue
            accepted += 1
            self.assertEqual(overlap.materialize(), expected, msg=f"case {case}")
        self.assertGreaterEqual(accepted, 16)
        self.assertGreaterEqual(clipped_declines, 16)

    def test_full_target_overlap_checks_consumer_semantic_nonemptiness(self) -> None:
        producer = _domain((10, 1), kind="worker")
        consumer = _domain((20, 3), kind="worker")
        key = _domain((30, 1), kind="event")
        consumer_coordinate = coordinate_axis_symbol(20)
        producer_keys = CoordinateRelation.total(producer, key)
        conditionally_clipped = _full_point_map(consumer, key, consumer_coordinate - 1)
        always_clipped = _full_point_map(consumer, key, sympy.Integer(2))
        in_domain = _full_point_map(consumer, key, sympy.Integer(0))

        self.assertIsNone(producer_keys.overlapping_sources(conditionally_clipped))
        empty = producer_keys.overlapping_sources(always_clipped)
        overlap = producer_keys.overlapping_sources(in_domain)
        self.assertIsNotNone(empty)
        self.assertIsNotNone(overlap)
        assert empty is not None and overlap is not None
        self.assertEqual(empty.materialize(), (frozenset(),) * 3)
        self.assertEqual(overlap.materialize(), (frozenset((0,)),) * 3)

    def test_partial_stride_overlap_preserves_endpoint_crossing(self) -> None:
        producer = _domain((10, 5), kind="site")
        consumer = _domain((20, 7), kind="site")
        allocation = _domain((-1, 128), kind="allocation", identity=0)
        producer_coordinate = coordinate_axis_symbol(10)
        consumer_coordinate = coordinate_axis_symbol(20)
        producer_access = _relation(
            producer,
            allocation,
            _piece(
                ((10, 1, 4, 3),),
                (-1, producer_coordinate + 110, producer_coordinate + 114, 1),
            ),
        )
        consumer_access = _relation(
            consumer,
            allocation,
            _piece(
                ((20, 0, 7, 1),),
                (-1, 4 * consumer_coordinate + 102, 4 * consumer_coordinate + 107, 1),
            ),
        )

        overlap = producer_access.overlapping_sources(consumer_access)

        assert overlap is not None
        self.assertEqual(
            overlap.materialize(),
            (
                frozenset(),
                frozenset(),
                frozenset((1,)),
                frozenset((1,)),
                frozenset(),
                frozenset(),
                frozenset(),
            ),
        )

    def test_partial_overlap_declines_mixed_clipped_target_pieces(self) -> None:
        producer = _domain((10, 2), kind="site")
        consumer = _domain((20, 2), kind="site")
        allocation = _domain((-1, 4), kind="allocation", identity=0)
        producer_coordinate = coordinate_axis_symbol(10)
        consumer_coordinate = coordinate_axis_symbol(20)
        producer_access = _relation(
            producer,
            allocation,
            _piece(
                ((10, 0, 2, 1),),
                (-1, producer_coordinate + 4, producer_coordinate + 5, 1),
            ),
            _piece(
                ((10, 0, 1, 1),), (-1, producer_coordinate, producer_coordinate + 1, 1)
            ),
        )
        consumer_access = _relation(
            consumer,
            allocation,
            _piece(
                ((20, 0, 2, 1),),
                (-1, consumer_coordinate + 4, consumer_coordinate + 5, 1),
            ),
        )

        # Once any producer piece has partial source support, the generalized
        # overlap path must validate target clipping for every producer piece.
        # Declining is conservative; accepting the raw out-of-domain intervals
        # would invent an overlap for consumer 0.
        self.assertIsNone(producer_access.overlapping_sources(consumer_access))

    def test_partial_strided_overlap_randomized_differential(self) -> None:
        generator = random.Random(0)
        accepted = 0
        for case in range(128):
            producer_count = generator.randrange(1, 9)
            consumer_count = generator.randrange(1, 9)
            allocation_count = generator.randrange(8, 33)
            producer_begin = generator.randrange(-5, producer_count + 4)
            producer_end = generator.randrange(
                producer_begin,
                producer_count + 6,
            )
            producer_step = generator.randrange(2, 5)
            producer_scale = generator.randrange(1, 5)
            producer_offset = generator.randrange(-10, allocation_count + 8)
            producer_width = generator.randrange(1, 7)
            consumer_scale = generator.randrange(1, 6)
            consumer_offset = generator.randrange(-10, allocation_count + 8)
            consumer_width = generator.randrange(1, 7)
            producer = _domain((10, producer_count), kind="site")
            consumer = _domain((20, consumer_count), kind="site")
            allocation = _domain(
                (-1, allocation_count), kind="allocation", identity=case
            )
            producer_coordinate = coordinate_axis_symbol(10)
            consumer_coordinate = coordinate_axis_symbol(20)
            producer_access = _relation(
                producer,
                allocation,
                _piece(
                    ((10, producer_begin, producer_end, producer_step),),
                    (
                        -1,
                        producer_scale * producer_coordinate + producer_offset,
                        producer_scale * producer_coordinate
                        + producer_offset
                        + producer_width,
                        1,
                    ),
                ),
            )
            consumer_access = _relation(
                consumer,
                allocation,
                _piece(
                    ((20, 0, consumer_count, 1),),
                    (
                        -1,
                        consumer_scale * consumer_coordinate + consumer_offset,
                        consumer_scale * consumer_coordinate
                        + consumer_offset
                        + consumer_width,
                        1,
                    ),
                ),
            )

            overlap = producer_access.overlapping_sources(consumer_access)
            if overlap is None:
                continue
            accepted += 1
            producer_points = producer_access.materialize()
            consumer_points = consumer_access.materialize()
            expected = tuple(
                frozenset(
                    producer_index
                    for producer_index, points in enumerate(producer_points)
                    if points & consumer_points[consumer_index]
                )
                for consumer_index in range(consumer_count)
            )
            self.assertEqual(
                overlap.materialize(),
                expected,
                msg=f"random differential case {case}",
            )
        self.assertGreaterEqual(accepted, 16)

    def test_partial_strided_overlap_has_symbolic_substitution_parity(self) -> None:
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)

        def relations(
            count: int | sympy.Expr,
        ) -> tuple[CoordinateRelation, CoordinateRelation]:
            producer = _domain((10, count), kind="site", allow_empty=True)
            consumer = _domain((20, count), kind="site", allow_empty=True)
            allocation = _domain(
                (-1, count), kind="allocation", identity=0, allow_empty=True
            )
            producer_coordinate = coordinate_axis_symbol(10)
            consumer_coordinate = coordinate_axis_symbol(20)
            return (
                _relation(
                    producer,
                    allocation,
                    _piece(
                        ((10, 0, count, 2),),
                        (-1, producer_coordinate, producer_coordinate + 1, 1),
                    ),
                ),
                _relation(
                    consumer,
                    allocation,
                    _piece(
                        ((20, 0, count, 1),),
                        (-1, sympy.Integer(0), consumer_coordinate + 1, 1),
                    ),
                ),
            )

        producer_access, consumer_access = relations(extent)
        overlap = producer_access.overlapping_sources(consumer_access)
        assert overlap is not None

        for concrete_extent in (0, 1, 2, 5, 8):
            with self.subTest(extent=concrete_extent):
                direct_producer, direct_consumer = relations(concrete_extent)
                direct = direct_producer.overlapping_sources(direct_consumer)
                assert direct is not None
                specialized = overlap.substitute_parameters({extent: concrete_extent})
                expected = tuple(
                    frozenset(range(0, consumer_index + 1, 2))
                    for consumer_index in range(concrete_extent)
                )
                self.assertEqual(specialized.materialize(), expected)
                self.assertEqual(direct.materialize(), expected)

        with mock.patch(
            "helion._compiler.tile_dependency._relation_product_is_within_budget",
            return_value=False,
        ):
            self.assertIsNone(producer_access.overlapping_sources(consumer_access))

    def test_target_coalescing_rejects_conditionally_empty_piece(self) -> None:
        source = _domain((20, 3), kind="site")
        coordinate = coordinate_axis_symbol(20)
        pieces = (
            _piece(((20, 0, 3, 1),), (10, coordinate, sympy.Integer(1), 1)),
            _piece(((20, 0, 3, 1),), (10, sympy.Integer(1), sympy.Integer(2), 1)),
        )

        coalesced = _coalesce_adjacent_target_boxes(pieces, source_domain=source)
        self.assertEqual(frozenset(coalesced), frozenset(pieces))
        self.assertEqual(
            _coalesce_adjacent_target_boxes(
                tuple(reversed(pieces)),
                source_domain=source,
            ),
            coalesced,
        )

    def test_logical_simplification_preserves_bounded_floor_correlation(self) -> None:
        source = _domain((20, 2), kind="task_order")
        coordinate = coordinate_axis_symbol(20)
        expression = coordinate + 2 * sympy.floor(
            sympy.Rational(511, 2) - coordinate / 2
        )

        self.assertEqual(
            _simplify_logical_expression(
                expression,
                domain=source,
                source_bounds=((20, 0, 2, 1),),
            ),
            coordinate + 510,
        )

    def test_logical_simplification_reassociates_integer_quotient_digits(
        self,
    ) -> None:
        source = _domain((20, 1536), kind="task_order")
        coordinate = coordinate_axis_symbol(20)
        bounds = ((20, 0, 1536, 1),)

        for coefficient, constant in ((3, 11), (-5, 7)):
            expression = (
                coefficient * 4 * sympy.floor(coordinate / 12)
                + coefficient * sympy.floor(sympy.Mod(coordinate, 12) / 3)
                + constant
            )
            self.assertEqual(
                _simplify_logical_expression(
                    expression,
                    domain=source,
                    source_bounds=bounds,
                ),
                coefficient * sympy.floor(coordinate / 3) + constant,
            )

        # The quotient identity is valid only for an integer dividend.  A
        # foreign real-valued expression must remain untouched.
        real_value = sympy.Symbol("real_value", real=True)
        noninteger_expression = 4 * sympy.floor(real_value / 12) + sympy.floor(
            sympy.Mod(real_value, 12) / 3
        )
        self.assertEqual(
            _simplify_logical_expression(
                noninteger_expression,
                domain=source,
                source_bounds=bounds,
            ),
            noninteger_expression,
        )

    def test_logical_simplification_uses_symbolic_mixed_radix_bounds(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        domain = _domain((20, 2), (21, batch), kind="site")
        chunk = coordinate_axis_symbol(20)
        batch_index = coordinate_axis_symbol(21)
        bounds = ((20, 0, 2, 1), (21, 0, batch, 1))

        self.assertEqual(
            _logical_expression_bounds(
                2 * batch * chunk + batch_index,
                domain=domain,
                source_bounds=bounds,
            ),
            (sympy.Integer(0), 3 * batch - 1),
        )
        self.assertEqual(
            _simplify_logical_expression(
                sympy.Mod(2 * chunk + sympy.floor(batch_index / batch), 4),
                domain=domain,
                source_bounds=bounds,
            ),
            2 * chunk,
        )

    def test_logical_bounds_apply_point_substitution_before_classification(
        self,
    ) -> None:
        source = _domain((20, 2), kind="site")
        source_coordinate = coordinate_axis_symbol(20)
        intermediate_coordinate = coordinate_axis_symbol(10)

        self.assertEqual(
            _logical_expression_bounds(
                sympy.floor(intermediate_coordinate / 2),
                domain=source,
                source_bounds=((20, 0, 2, 1),),
                symbol_substitutions={intermediate_coordinate: source_coordinate},
            ),
            (sympy.Integer(0), sympy.Integer(0)),
        )

    def test_logical_simplification_proves_symbolic_dominance_safely(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        domain = _domain((20, batch), kind="site")
        coordinate = coordinate_axis_symbol(20)
        bounds = ((20, 0, batch, 1),)

        ambiguous_minimum = sympy.Min(coordinate, sympy.floor(batch / 2))
        ambiguous_maximum = sympy.Max(coordinate, batch - 2)
        self.assertEqual(
            _simplify_logical_expression(
                ambiguous_minimum,
                domain=domain,
                source_bounds=bounds,
            ),
            ambiguous_minimum,
        )
        self.assertEqual(
            _simplify_logical_expression(
                ambiguous_maximum,
                domain=domain,
                source_bounds=bounds,
            ),
            ambiguous_maximum,
        )
        self.assertEqual(
            _simplify_logical_expression(
                sympy.Min(coordinate, batch),
                domain=domain,
                source_bounds=bounds,
            ),
            coordinate,
        )
        self.assertEqual(
            _simplify_logical_expression(
                sympy.Max(0, batch - coordinate - 1),
                domain=domain,
                source_bounds=bounds,
            ),
            batch - coordinate - 1,
        )

    def test_symbolic_tail_relation_uses_exact_support_cardinality(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((20, batch + 1), kind="site")
        target = _domain((10, 2), kind="event")
        relation = _relation(
            source,
            target,
            _piece(((20, 0, batch, 1),), (10, sympy.Integer(0), sympy.Integer(1), 1)),
            _piece(
                ((20, batch, batch + 1, 1),),
                (10, sympy.Integer(1), sympy.Integer(2), 1),
            ),
        )

        self.assertTrue(relation.is_single_valued())
        self.assertEqual(relation.canonical_single_valued(), relation)
        self.assertTrue(relation.has_total_source())
        self.assertTrue(relation.is_total_function())
        self.assertEqual(
            sympy.simplify(relation.source_support_cardinality() - batch - 1),
            0,
        )
        for concrete_batch in (0, 1, 8):
            concrete = relation.substitute_parameters({batch: concrete_batch})
            self.assertTrue(
                all(len(targets) == 1 for targets in concrete.materialize())
            )

    def test_compile_environment_nonnegative_proof_uses_shape_ranges(self) -> None:
        env = CompileEnvironment(torch.device("cpu"), helion.Settings(backend="triton"))
        with env:
            batch_size = env.create_unbacked_symint(hint=8)
            batch = batch_size._sympy_()
            env.shape_env.constrain_symbol_range(batch, 0, 16)
            guards_before = tuple(env.shape_env.guards)

            self.assertTrue(env.known_nonnegative(256 - 16 * batch))
            self.assertFalse(env.known_nonnegative(255 - 16 * batch))
            self.assertFalse(
                env.known_nonnegative(
                    1 - sympy.Symbol("foreign", integer=True, nonnegative=True)
                )
            )
            self.assertEqual(tuple(env.shape_env.guards), guards_before)

    def test_factor_through_folds_only_small_bounded_constants(self) -> None:
        small = _bounded_coordinate_relation(64)
        large = _bounded_coordinate_relation(65)
        key_domain = _domain((10, 2), kind="event")
        small_quotient = CoordinateRelation.projection(
            small.source_domain,
            key_domain,
        )
        large_quotient = CoordinateRelation.projection(
            large.source_domain,
            key_domain,
        )
        assert small_quotient is not None and large_quotient is not None

        factored = small.factor_through(small_quotient)

        assert factored is not None
        self.assertEqual(
            factored.materialize(),
            (frozenset((0,)), frozenset((1,))),
        )
        self.assertIsNone(large.factor_through(large_quotient))

    def test_source_axes_fold_only_small_bounded_constants(self) -> None:
        self.assertEqual(
            _bounded_coordinate_relation(64).source_axes_affecting_targets(),
            (10,),
        )
        self.assertEqual(
            _bounded_coordinate_relation(65).source_axes_affecting_targets(),
            (10, 11),
        )

    def test_source_axes_ignore_known_domain_parameters(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        foreign = sympy.Symbol("foreign", integer=True, nonnegative=True)
        source = _domain((10, batch), (11, 4), kind="site")
        target = _domain((20, batch), (21, batch), kind="site")
        source_batch = coordinate_axis_symbol(10)

        relation = _relation(
            source,
            target,
            _piece(
                ((10, 0, batch, 1), (11, 0, 4, 1)),
                (20, source_batch, source_batch + 1, 1),
                (21, sympy.Integer(0), batch, 1),
            ),
        )
        self.assertEqual(relation.source_axes_affecting_targets(), (10,))

        unknown_parameter = dataclasses.replace(
            relation,
            pieces=(
                dataclasses.replace(
                    relation.pieces[0],
                    target_ranges=(
                        relation.pieces[0].target_ranges[0],
                        (21, sympy.Integer(0), foreign, 1),
                    ),
                ),
            ),
        )
        self.assertIsNone(unknown_parameter.source_axes_affecting_targets())

    def test_dense_converse_requires_full_singleton_extra_axes(self) -> None:
        source = _domain((0, 2), kind="event")
        source_index = coordinate_axis_symbol(0)
        expected = (frozenset((0,)),) * 4 + (frozenset((1,)),) * 4
        for name, extra_axis_count, extra_axis_end, expected_converse in (
            ("full singleton", 1, 1, expected),
            ("empty singleton", 1, 0, None),
            ("nontrivial axis", 2, 2, None),
        ):
            with self.subTest(case=name):
                target = _domain((10, extra_axis_count), (11, 8), kind="site")
                relation = _relation(
                    source,
                    target,
                    _piece(
                        ((0, 0, 2, 1),),
                        (10, sympy.Integer(0), extra_axis_end, 1),
                        (11, 4 * source_index, 4 * source_index + 4, 1),
                    ),
                )
                target_counts = relation.target_count_by_source()
                assert target_counts is not None
                converse = _dense_mixed_radix_converse(relation, target_counts)
                if expected_converse is None:
                    self.assertIsNone(converse)
                else:
                    self.assertIsNotNone(converse)
                    assert converse is not None
                    self.assertEqual(converse.materialize(), expected_converse)

    def test_target_enumeration_preserves_multi_piece_bijection(self) -> None:
        producer = _domain((10, 8), identity=0)
        keys = _domain((0, 2), kind="event", identity=0)
        producer_to_key = _point_map(
            producer,
            keys,
            (((10, 0, 4, 1),), (sympy.Integer(0),)),
            (((10, 4, 8, 1),), (sympy.Integer(1),)),
        )
        converse = producer_to_key.converse()
        assert converse is not None
        task_order = converse.enumerate_targets_by_source()
        assert task_order is not None
        self.assertEqual(
            tuple(next(iter(targets)) for targets in task_order.materialize()),
            tuple(range(producer.size)),
        )

        tail_producer = _domain((10, 7), identity=0)
        tail_relation = _point_map(
            tail_producer,
            keys,
            (((10, 0, 4, 1),), (sympy.Integer(0),)),
            (((10, 4, 7, 1),), (sympy.Integer(1),)),
        )
        tail_converse = tail_relation.converse()
        assert tail_converse is not None
        self.assertIsNone(tail_converse.enumerate_targets_by_source())

    def test_target_enumeration_accepts_symbolic_full_domain_fibers(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        cohorts = _domain((0, batch), kind="event", allow_empty=True)
        order = _domain((10, batch), (11, 2), kind="task_order", allow_empty=True)
        cohort = coordinate_axis_symbol(0)
        order_points_by_cohort = _relation(
            cohorts,
            order,
            _piece(
                ((0, 0, batch, 1),),
                (10, cohort, cohort + 1, 1),
                (11, sympy.Integer(0), sympy.Integer(2), 1),
            ),
        )

        with (
            mock.patch.object(
                CoordinateRelation,
                "materialize",
                side_effect=AssertionError("symbolic enumeration must not materialize"),
            ),
            mock.patch.object(
                CoordinateRelation,
                "targets",
                side_effect=AssertionError("symbolic enumeration must not enumerate"),
            ),
        ):
            task_order = order_points_by_cohort.enumerate_targets_by_source()
            assert task_order is not None
            self.assertTrue(task_order.is_bijection_from_source_support())

        for concrete_batch in (0, 1, 3):
            concrete = task_order.substitute_parameters({batch: concrete_batch})
            self.assertEqual(
                sorted(next(iter(targets)) for targets in concrete.materialize()),
                list(range(2 * concrete_batch)),
            )

    def test_symbolic_target_enumeration_declines_clipped_strided_range(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((0, batch), kind="event", allow_empty=True)
        target = _domain((10, 3), kind="site")
        relation = _relation(
            source,
            target,
            _piece(((0, 0, batch, 1),), (10, -1, 3, 2)),
        )

        self.assertIsNone(relation.enumerate_targets_by_source())

    def test_symbolic_dependency_preserves_unequal_tile_range(self) -> None:
        elements = 65_536
        plan = build_tile_dependency_graph(
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
                    shape=(elements,),
                    block_ids=(20,),
                ),
            ),
            [[10], [20]],
        )
        axis_geometry = {
            10: (elements // 16, 16),
            20: (elements // 32, 32),
        }

        relation = _symbolic_root_relation(plan, axis_geometry)

        assert relation is not None
        self.assertEqual(len(relation.pieces), 1)
        self.assertEqual(relation.targets(0), frozenset((0, 1)))
        self.assertEqual(relation.targets(123), frozenset((246, 247)))
        self.assertEqual(
            relation.targets(elements // 32 - 1),
            frozenset((elements // 16 - 2, elements // 16 - 1)),
        )

        cardinality = relation.target_count_by_source()
        assert cardinality is not None
        self.assertEqual(
            cardinality.materialize(),
            tuple(frozenset((2,)) for _ in range(elements // 32)),
        )

    def test_symbolic_target_count_preserves_tail_pieces(self) -> None:
        elements = 65
        plan = build_tile_dependency_graph(
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
                    shape=(elements,),
                    block_ids=(20,),
                ),
            ),
            [[10], [20]],
        )
        relation = _symbolic_root_relation(
            plan,
            {
                10: ((elements + 15) // 16, 16),
                20: ((elements + 23) // 24, 24),
            },
        )

        assert relation is not None
        cardinality = relation.target_count_by_source()
        assert cardinality is not None
        self.assertEqual(
            cardinality.materialize(),
            tuple(frozenset((len(targets),)) for targets in relation.materialize()),
        )

    def test_target_count_accepts_exact_partial_strided_source_support(self) -> None:
        source = _domain((10, 6), kind="site")
        target = _domain((20, 6), kind="site")
        source_index = coordinate_axis_symbol(10)
        source_bounds = ((10, 0, 6, 2),)
        relation = _relation(
            source,
            target,
            _piece(source_bounds, (20, sympy.Integer(0), source_index, 1)),
        )
        support = _point_map(
            source,
            source,
            (source_bounds, (source_index,)),
            (((10, 5, 6, 1),), (source_index,)),
        )

        counts = relation.target_count_by_source(source_support=support)

        assert counts is not None
        self.assertEqual(
            counts.materialize(),
            (
                frozenset((0,)),
                frozenset(),
                frozenset((2,)),
                frozenset(),
                frozenset((4,)),
                frozenset((0,)),
            ),
        )

    def test_target_count_with_symbolic_support_including_empty(self) -> None:
        count = sympy.Symbol("count", integer=True, nonnegative=True)
        source = _domain((10, count), kind="site", allow_empty=True)
        target = _domain((20, count), kind="site", allow_empty=True)
        source_index = coordinate_axis_symbol(10)
        source_bounds = ((10, 0, count, 1),)
        relation = _relation(
            source,
            target,
            _piece(source_bounds, (20, sympy.Integer(0), source_index, 1)),
        )
        support = _point_map(source, source, (source_bounds, (source_index,)))

        counts = relation.target_count_by_source(source_support=support)

        assert counts is not None
        for concrete_count in (0, 1, 5):
            with self.subTest(count=concrete_count):
                concrete = counts.substitute_parameters({count: concrete_count})
                self.assertEqual(
                    concrete.materialize(),
                    tuple(frozenset((index,)) for index in range(concrete_count)),
                )

    def test_target_count_accepts_exact_symbolic_support_carrier(self) -> None:
        count = sympy.Symbol("count", integer=True, nonnegative=True)
        source = _domain((10, count), kind="site", allow_empty=True)
        target = _domain((20, count), kind="site", allow_empty=True)
        carrier_target = _domain((30, count), kind="task_order", allow_empty=True)
        source_index = coordinate_axis_symbol(10)
        carrier_index = coordinate_axis_symbol(30)
        relation = _relation(
            source,
            target,
            _piece(((10, 0, count, 1),), (20, sympy.Integer(0), source_index, 1)),
        )
        carrier = _point_map(
            source,
            carrier_target,
            (((10, 0, sympy.Min(1, count), 1),), (source_index,)),
            (((10, 1, count, 1),), (source_index,)),
        )
        carrier_converse = _full_point_map(carrier_target, source, carrier_index)
        _remember_exact_converse(carrier, carrier_converse)

        counts = relation.target_count_by_source(source_support=carrier)

        assert counts is not None
        for concrete_count in (0, 1, 4):
            with self.subTest(count=concrete_count):
                concrete = counts.substitute_parameters({count: concrete_count})
                self.assertEqual(
                    concrete.materialize(),
                    tuple(frozenset((index,)) for index in range(concrete_count)),
                )

    def test_target_count_carrier_declines_crossing_relation_boundary(self) -> None:
        source = _domain((10, 4), kind="site")
        target = _domain((20, 4), kind="site")
        carrier_target = _domain((30, 2), kind="task_order")
        source_index = coordinate_axis_symbol(10)
        carrier_index = coordinate_axis_symbol(30)
        relation = _relation(
            source,
            target,
            _piece(((10, 0, 2, 1),), (20, sympy.Integer(0), sympy.Integer(1), 1)),
            _piece(((10, 2, 4, 1),), (20, sympy.Integer(0), sympy.Integer(2), 1)),
        )
        carrier = _point_map(
            source, carrier_target, (((10, 1, 3, 1),), (source_index - 1,))
        )
        carrier_converse = _full_point_map(carrier_target, source, carrier_index + 1)
        _remember_exact_converse(carrier, carrier_converse)

        self.assertIsNone(relation.target_count_by_source(source_support=carrier))

    def test_target_count_with_support_rejects_overlapping_targets(self) -> None:
        source = _domain((10, 3), kind="site")
        target = _domain((20, 4), kind="site")
        source_bounds = ((10, 0, 3, 1),)
        relation = _relation(
            source,
            target,
            _piece(source_bounds, (20, sympy.Integer(0), sympy.Integer(3), 1)),
            _piece(source_bounds, (20, sympy.Integer(2), sympy.Integer(4), 1)),
        )
        support = CoordinateRelation.identity(source, source)

        self.assertIsNone(relation.target_count_by_source(source_support=support))

    def test_target_count_rejects_invalid_or_overlapping_source_support(self) -> None:
        source = _domain((10, 4), kind="site")
        target = _domain((20, 4), kind="site")
        source_index = coordinate_axis_symbol(10)
        relation = _relation(
            source,
            target,
            _piece(((10, 0, 4, 1),), (20, sympy.Integer(0), source_index, 1)),
        )
        nonidentity_support = _full_point_map(
            source, source, sympy.Mod(source_index + 1, 4)
        )
        overlapping_support = _point_map(
            source,
            source,
            (((10, 0, 3, 1),), (source_index,)),
            (((10, 2, 4, 1),), (source_index,)),
        )
        piecewise_relation = _relation(
            source,
            target,
            _piece(((10, 0, 2, 1),), (20, sympy.Integer(0), sympy.Integer(1), 1)),
            _piece(((10, 2, 4, 1),), (20, sympy.Integer(0), sympy.Integer(2), 1)),
        )
        crossing_support = _point_map(
            source, source, (((10, 1, 3, 1),), (source_index,))
        )

        self.assertIsNone(
            relation.target_count_by_source(source_support=nonidentity_support)
        )
        self.assertIsNone(
            relation.target_count_by_source(source_support=overlapping_support)
        )
        self.assertIsNone(
            piecewise_relation.target_count_by_source(source_support=crossing_support)
        )
        with mock.patch(
            "helion._compiler.tile_dependency._relation_product_is_within_budget",
            return_value=False,
        ):
            self.assertIsNone(
                relation.target_count_by_source(
                    source_support=CoordinateRelation.identity(source, source)
                )
            )

    def test_symbolic_muse_group_widths_keep_affine_fan_in(self) -> None:
        producer_block = 256
        for groups, group_width in ((16, 1248), (13, 1536)):
            with self.subTest(groups=groups, group_width=group_width):
                elements = groups * group_width
                plan = build_tile_dependency_graph(
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
                            shape=(elements,),
                            block_ids=(20,),
                        ),
                    ),
                    [[10], [20]],
                )
                relation = _symbolic_root_relation(
                    plan,
                    {
                        10: ((elements + producer_block - 1) // producer_block, 256),
                        20: (groups, group_width),
                    },
                )

                assert relation is not None
                self.assertLessEqual(len(relation.pieces), 3)
                cardinality = relation.target_count_by_source()
                assert cardinality is not None
                expected = tuple(
                    frozenset(
                        (
                            (math.ceil((group + 1) * group_width / producer_block))
                            - (group * group_width // producer_block),
                        )
                    )
                    for group in range(groups)
                )
                self.assertEqual(cardinality.materialize(), expected)
                if group_width == 1536:
                    self.assertEqual(set(expected), {frozenset((6,))})
                else:
                    self.assertGreater(len(set(expected)), 1)

    def test_symbolic_static_contiguous_index_range_keeps_exact_support(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    allocation_id=0,
                    kind="store",
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    allocation_id=0,
                    kind="load",
                    block_ids=(None,),
                    offsets=(32,),
                    static_extents=(64,),
                ),
            ),
            [[10], [20]],
        )
        relation = _symbolic_root_relation(
            plan,
            {
                10: (8, 16),
                20: (1, 1),
            },
        )

        assert relation is not None
        self.assertEqual(relation.materialize(), (frozenset((2, 3, 4, 5)),))

    def test_relation_coverage_preserves_stride_phase(self) -> None:
        source = _domain((10, 8), identity=0)
        key = _domain((0, 8), kind="event", identity=0)
        even_sources = _relation(
            source,
            key,
            _piece(((10, 0, 8, 2),), (0, sympy.Integer(0), sympy.Integer(1), 1)),
        )
        odd_sources = _relation(
            source,
            key,
            _piece(((10, 1, 8, 2),), (0, sympy.Integer(0), sympy.Integer(1), 1)),
        )

        self.assertFalse(even_sources.covers(odd_sources))
        source_union = even_sources.union(odd_sources)
        assert source_union is not None
        self.assertEqual(source_union.materialize(), (frozenset((0,)),) * 8)

        singleton = _domain((20, 1), identity=1)
        even_targets = _relation(
            singleton,
            key,
            _piece(((20, 0, 1, 1),), (0, sympy.Integer(0), sympy.Integer(8), 2)),
        )
        odd_targets = _relation(
            singleton,
            key,
            _piece(((20, 0, 1, 1),), (0, sympy.Integer(1), sympy.Integer(8), 2)),
        )

        self.assertFalse(even_targets.covers(odd_targets))
        target_union = even_targets.union(odd_targets)
        assert target_union is not None
        self.assertEqual(target_union.materialize(), (frozenset(range(8)),))

    def test_relation_coverage_factors_runtime_positional_axis(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((10, batch), (11, 2), identity=0)
        target = _domain((20, batch), (21, 2), identity=1)
        outer = coordinate_axis_symbol(10)
        inner = coordinate_axis_symbol(11)
        available = _relation(
            source,
            target,
            _piece(
                ((10, 0, batch, 1), (11, 0, 2, 1)),
                (20, outer, outer + 1, 1),
                (21, sympy.Integer(0), sympy.Integer(2), 1),
            ),
        )
        required = _full_point_map(source, target, outer, inner)

        with mock.patch(
            "helion._compiler.tile_dependency._relation_piece_covers",
            side_effect=AssertionError("runtime identity factor should be reused"),
        ):
            self.assertTrue(available.covers(required))
        self.assertTrue(available.has_total_source())
        self.assertFalse(available.is_single_valued())
        self.assertTrue(required.has_total_source())
        self.assertTrue(required.is_single_valued())
        self.assertFalse(required.covers(available))

    def test_positional_product_does_not_hide_runtime_residual(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((10, batch), (11, 2), identity=0)
        target = _domain((20, batch), (21, 2), identity=1)
        outer = coordinate_axis_symbol(10)
        inner = coordinate_axis_symbol(11)
        relation = _full_point_map(source, target, batch - outer - 1, inner)

        # The static identity axis is not a useful Cartesian factor when the
        # remaining runtime axis is non-positional.
        self.assertIsNone(relation._positional_product)

    def test_relation_operations_decline_before_oversized_products(self) -> None:
        domain = _domain((10, 4), identity=0)
        coordinate = coordinate_axis_symbol(10)
        first = _point_map(
            domain,
            domain,
            (((10, 0, 2, 1),), (coordinate,)),
            (((10, 2, 4, 1),), (coordinate,)),
        )
        following = _point_map(
            domain,
            domain,
            (((10, 0, 1, 1),), (coordinate,)),
            (((10, 1, 4, 1),), (coordinate,)),
        )

        with (
            mock.patch(
                "helion._compiler.tile_dependency._MAX_RELATION_PRODUCT_STATES",
                3,
            ),
            mock.patch(
                "helion._compiler.tile_dependency._relation_piece_covers",
                side_effect=AssertionError("union must check its budget first"),
            ),
        ):
            self.assertIsNone(first.union(following))
        with (
            mock.patch(
                "helion._compiler.tile_dependency._MAX_RELATION_PRODUCT_STATES",
                3,
            ),
            mock.patch(
                "helion._compiler.tile_dependency._substitute_composed_expression",
                side_effect=AssertionError("composition must check its budget first"),
            ),
        ):
            self.assertIsNone(first.then(following))
        with (
            mock.patch(
                "helion._compiler.tile_dependency._MAX_RELATION_PRODUCT_STATES",
                3,
            ),
            mock.patch(
                "helion._compiler.tile_dependency._source_boxes_are_disjoint",
                side_effect=AssertionError("disjointness must check its budget first"),
            ),
        ):
            self.assertFalse(first.has_disjoint_source_support(following))
        with (
            mock.patch(
                "helion._compiler.tile_dependency._MAX_RELATION_PRODUCT_STATES",
                3,
            ),
            mock.patch(
                "helion._compiler.tile_dependency._relation_piece_covers",
                side_effect=AssertionError("coverage must check its budget first"),
            ),
        ):
            self.assertFalse(first.covers(following))

        self.assertEqual(len(first.coalesce_adjacent_source_boxes().pieces), 1)
        with mock.patch(
            "helion._compiler.tile_dependency._MAX_RELATION_PRODUCT_STATES",
            3,
        ):
            self.assertIs(first.coalesce_adjacent_source_boxes(), first)

        two_dimensional = _domain((10, 2), (11, 2), identity=0)
        horizontal = _point_map(
            two_dimensional,
            two_dimensional,
            (
                ((10, 0, 1, 1), (11, 0, 2, 1)),
                (coordinate_axis_symbol(10), coordinate_axis_symbol(11)),
            ),
            (
                ((10, 1, 2, 1), (11, 0, 1, 1)),
                (coordinate_axis_symbol(10), coordinate_axis_symbol(11)),
            ),
        )
        with (
            mock.patch(
                "helion._compiler.tile_dependency._MAX_RELATION_PRODUCT_STATES",
                3,
            ),
            mock.patch(
                "helion._compiler.tile_dependency.itertools.product",
                side_effect=AssertionError(
                    "source cells must check their budget first"
                ),
            ),
        ):
            self.assertIsNone(_relation_source_cells(horizontal))

    def test_relation_union_declines_above_retained_piece_budget(self) -> None:
        source = _domain((10, 2), identity=0)
        target = _domain((20, 2), identity=1)
        coordinate = coordinate_axis_symbol(10)
        left = _point_map(source, target, (((10, 0, 1, 1),), (coordinate,)))
        right = _point_map(source, target, (((10, 1, 2, 1),), (coordinate,)))

        with mock.patch(
            "helion._compiler.tile_dependency._MAX_RELATION_PIECES",
            1,
        ):
            self.assertIsNone(left.union(right))

    def test_adjacent_target_coalescing_with_symbolic_source(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((10, batch), kind="site")
        target = _domain((20, batch), (21, 2), kind="site")
        source_batch = coordinate_axis_symbol(10)

        def half(begin: int) -> CoordinateRelation:
            return _relation(
                source,
                target,
                _piece(
                    ((10, 0, batch, 1),),
                    (20, source_batch, source_batch + 1, 1),
                    (21, sympy.Integer(begin), sympy.Integer(begin + 1), 1),
                ),
            )

        raw_union = half(0).union(half(1))
        assert raw_union is not None
        combined = raw_union.coalesce_adjacent_target_boxes()
        reversed_combined = CoordinateRelation(
            source_domain=raw_union.source_domain,
            target_domain=raw_union.target_domain,
            pieces=tuple(reversed(raw_union.pieces)),
        ).coalesce_adjacent_target_boxes()
        self.assertEqual(reversed_combined, combined)
        with mock.patch(
            "helion._compiler.tile_dependency._MAX_RELATION_PRODUCT_STATES",
            0,
        ):
            budgeted = raw_union.coalesce_adjacent_target_boxes()
        self.assertEqual(len(budgeted.pieces), 2)
        self.assertEqual(len(combined.pieces), 1)
        self.assertEqual(
            combined.pieces[0].target_ranges[-1],
            (21, sympy.Integer(0), sympy.Integer(2), 1),
        )
        for concrete_batch in (1, 2, 9):
            concrete = combined.substitute_parameters({batch: concrete_batch})
            budgeted_concrete = budgeted.substitute_parameters({batch: concrete_batch})
            self.assertEqual(
                concrete.materialize(),
                tuple(
                    frozenset((batch_index, batch_index + concrete_batch))
                    for batch_index in range(concrete_batch)
                ),
            )
            self.assertEqual(budgeted_concrete.materialize(), concrete.materialize())

    def test_target_box_extrema_require_one_attainable_corner(self) -> None:
        target_axis = 20
        target = _domain((target_axis, 4), kind="site")
        source = _domain(kind="event")
        coordinate = coordinate_axis_symbol(target_axis)
        target_ranges = ((target_axis, sympy.Integer(0), sympy.Integer(4), 1),)

        def extreme(expression: sympy.Expr, *, maximize: bool) -> sympy.Expr | None:
            return _target_box_expression_extreme(
                expression,
                target_domain=target,
                target_ranges=target_ranges,
                source_domain=source,
                source_bounds=(),
                maximize=maximize,
            )

        # The separate maxima of ``coordinate`` and ``-floor(coordinate / 2)``
        # occur at opposite endpoints.  Adding them used to return the
        # unattainable upper bound 4 even though the exact maximum is 3.
        anticorrelated = coordinate - sympy.floor(coordinate / 2) + 1
        self.assertEqual(
            max(anticorrelated.subs(coordinate, value) for value in range(4)),
            3,
        )
        self.assertIsNone(extreme(anticorrelated, maximize=True))
        self.assertIsNone(extreme(anticorrelated, maximize=False))

        # Negative coefficients reverse both the requested extremum and its
        # witnessing endpoint.
        self.assertEqual(extreme(-coordinate, maximize=True), 0)
        self.assertEqual(extreme(-coordinate, maximize=False), -3)
        dead_anticorrelated_term = sympy.Mul(0, -coordinate, evaluate=False)
        self.assertEqual(
            extreme(
                sympy.Add(coordinate, dead_anticorrelated_term, evaluate=False),
                maximize=True,
            ),
            3,
        )

        # Correlated children remain provable when their extrema share a
        # corner.  The same rule applies to Min/Max; incompatible child
        # extrema must decline rather than manufacture an unattainable value.
        correlated = coordinate + sympy.floor(coordinate / 2)
        self.assertEqual(extreme(correlated, maximize=True), 4)
        self.assertEqual(extreme(correlated, maximize=False), 0)
        self.assertEqual(
            extreme(
                sympy.Min(coordinate, sympy.floor(coordinate / 2) + 1),
                maximize=True,
            ),
            2,
        )
        self.assertEqual(
            extreme(
                sympy.Max(coordinate, sympy.floor(coordinate / 2) + 1),
                maximize=True,
            ),
            3,
        )
        self.assertIsNone(extreme(sympy.Min(coordinate, 3 - coordinate), maximize=True))
        self.assertIsNone(
            extreme(sympy.Max(coordinate, 3 - coordinate), maximize=False)
        )

    def test_target_box_extrema_modulo_requires_attainable_dividend_bounds(
        self,
    ) -> None:
        target_axis = 20
        target = _domain((target_axis, 4), kind="site")
        source = _domain(kind="event")
        coordinate = coordinate_axis_symbol(target_axis)
        target_ranges = ((target_axis, sympy.Integer(0), sympy.Integer(4), 1),)

        def maximum(expression: sympy.Expr) -> sympy.Expr | None:
            return _target_box_expression_extreme(
                expression,
                target_domain=target,
                target_ranges=target_ranges,
                source_domain=source,
                source_bounds=(),
                maximize=True,
            )

        anticorrelated = coordinate - sympy.floor(coordinate / 2) + 1
        self.assertIsNone(maximum(sympy.Mod(anticorrelated, 10)))
        self.assertIsNone(maximum(sympy.floor(sympy.Mod(anticorrelated, 10) / 2)))
        self.assertEqual(maximum(sympy.Mod(coordinate, 8)), 3)
        self.assertIsNone(maximum(sympy.Mod(coordinate + 1, 4)))

        key_axis = 10
        value_axis = 30
        keys = _domain((key_axis, 1), kind="event")
        values = _domain((value_axis, 2), kind="worker")
        required = _relation(
            keys,
            target,
            _piece(
                ((key_axis, 0, 1, 1),),
                (target_axis, sympy.Integer(0), sympy.Integer(4), 1),
            ),
        )
        fixed_width_value = _full_point_map(
            target, values, sympy.floor(sympy.Mod(anticorrelated, 10) / 2)
        )
        self.assertIsNone(required.max_target_value_by_source(fixed_width_value))

    def test_target_box_extrema_memoizes_nested_modulo_dag(self) -> None:
        target_axis = 20
        target = _domain((target_axis, 4), kind="site")
        source = _domain(kind="event")
        coordinate = coordinate_axis_symbol(target_axis)
        target_ranges = ((target_axis, sympy.Integer(0), sympy.Integer(4), 1),)
        depth = 12
        expression = coordinate
        for modulus in range(16, 16 + depth):
            expression = sympy.Mod(expression, modulus, evaluate=False)

        with mock.patch(
            "helion._compiler.tile_dependency._target_box_expression_extreme_proof_uncached",
            wraps=_target_box_expression_extreme_proof_uncached,
        ) as uncached:
            maximum = _target_box_expression_extreme(
                expression,
                target_domain=target,
                target_ranges=target_ranges,
                source_domain=source,
                source_bounds=(),
                maximize=True,
            )

        self.assertEqual(maximum, 3)
        self.assertLessEqual(uncached.call_count, 2 * depth + 1)

    def test_symbolic_max_target_value_declines_possibly_empty_fiber(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        key_axis = 10
        task_axis = 20
        value_axis = 30
        keys = _domain((key_axis, 1), kind="event")
        tasks = _domain((task_axis, batch), kind="site", allow_empty=True)
        values = _domain((value_axis, 1), kind="worker")
        required = _relation(
            keys,
            tasks,
            _piece(((key_axis, 0, 1, 1),), (task_axis, sympy.Integer(0), batch, 1)),
        )
        value_by_task = _full_point_map(tasks, values, sympy.Integer(0))

        self.assertEqual(
            required.substitute_parameters({batch: 0}).materialize(),
            (frozenset(),),
        )
        self.assertIsNone(required.max_target_value_by_source(value_by_task))
        self.assertIsNone(
            required.extreme_target_value_and_attainers_by_source(
                value_by_task,
                maximize=True,
            )
        )

        concrete_required = required.substitute_parameters({batch: 4})
        concrete_values = value_by_task.substitute_parameters({batch: 4})
        concrete_maximum = concrete_required.max_target_value_by_source(concrete_values)
        assert concrete_maximum is not None
        self.assertEqual(concrete_maximum.materialize(), (frozenset((0,)),))

    def test_extrema_accept_pairwise_disjoint_partial_source_groups(self) -> None:
        extent = sympy.Symbol(
            "partial_extrema_source_extent",
            integer=True,
            nonnegative=True,
        )
        source = _domain((10, extent + 2), kind="event")
        target = _domain((20, 3), kind="site")
        value = _domain((30, 3), kind="value")
        target_coordinate = coordinate_axis_symbol(20)
        required = _relation(
            source,
            target,
            _piece(((10, 0, 1, 1),), (20, 0, 2, 1)),
            _piece(((10, extent + 1, extent + 2, 1),), (20, 1, 3, 1)),
        )
        value_by_target = _full_point_map(target, value, target_coordinate)

        maximum = required.max_target_value_by_source(value_by_target)
        joint = required.extreme_target_value_and_attainers_by_source(
            value_by_target,
            maximize=True,
        )

        self.assertIsNotNone(maximum)
        self.assertIsNotNone(joint)
        assert maximum is not None and joint is not None
        for concrete_extent in (0, 1, 4):
            substitutions = {extent: concrete_extent}
            expected_values = tuple(
                frozenset((1,))
                if index == 0
                else frozenset((2,))
                if index == concrete_extent + 1
                else frozenset()
                for index in range(concrete_extent + 2)
            )
            expected_attainers = tuple(
                frozenset((1,))
                if index == 0
                else frozenset((2,))
                if index == concrete_extent + 1
                else frozenset()
                for index in range(concrete_extent + 2)
            )
            self.assertEqual(
                maximum.substitute_parameters(substitutions).materialize(),
                expected_values,
            )
            concrete_joint = tuple(
                relation.substitute_parameters(substitutions).materialize()
                for relation in joint
            )
            self.assertEqual(
                concrete_joint,
                (expected_values, expected_attainers),
            )

    def test_partial_source_group_extrema_respect_proof_budget(self) -> None:
        value = _domain((30, 1), kind="value")
        for group_count in (128, 256):
            with self.subTest(group_count=group_count):
                source = _domain((10, 2 * group_count), kind="event")
                target = _domain((20, 1), kind="site")
                required = CoordinateRelation(
                    source,
                    target,
                    tuple(
                        _piece(((10, 2 * index, 2 * index + 1, 1),), (20, 0, 1, 1))
                        for index in range(group_count)
                    ),
                )
                values = _full_point_map(target, value, sympy.Integer(0))

                with mock.patch(
                    "helion._compiler.tile_dependency._relation_source_cells",
                    side_effect=AssertionError(
                        "oversized partial source proof must decline before partition"
                    ),
                ):
                    self.assertIsNone(required.max_target_value_by_source(values))

    def test_target_value_extreme_retains_complete_plateaus(self) -> None:
        source = _domain(kind="event")
        target = _domain((20, 4), kind="site")
        values = _domain((30, 3), kind="value")
        target_coordinate = coordinate_axis_symbol(20)
        required = CoordinateRelation.total(source, target)
        value_by_target = _full_point_map(
            target, values, sympy.floor((target_coordinate + 2) / 2)
        )

        maximum = required.extreme_target_value_and_attainers_by_source(
            value_by_target,
            maximize=True,
        )
        minimum = required.extreme_target_value_and_attainers_by_source(
            value_by_target,
            maximize=False,
        )

        self.assertIsNotNone(maximum)
        self.assertIsNotNone(minimum)
        assert maximum is not None
        assert minimum is not None
        self.assertEqual(maximum[0].materialize(), (frozenset((2,)),))
        self.assertEqual(maximum[1].materialize(), (frozenset((2, 3)),))
        self.assertEqual(minimum[0].materialize(), (frozenset((1,)),))
        self.assertEqual(minimum[1].materialize(), (frozenset((0, 1)),))

        empty_relations = (
            _relation(source, target),
            *(
                _relation(source, target, _piece((), (20, begin, end, 1)))
                for begin, end in ((0, 0), (2, 2), (4, 4), (3, 2))
            ),
        )
        for empty in empty_relations:
            empty_result = empty.extreme_target_value_and_attainers_by_source(
                value_by_target,
                maximize=True,
            )
            assert empty_result is not None
            self.assertEqual(
                (empty_result[0].pieces, empty_result[1].pieces),
                ((), ()),
            )

    def test_target_value_extreme_accepts_exact_partial_value_support(self) -> None:
        source_axis = 10
        target_axis = 20
        value_axis = 30
        source = _domain((source_axis, 2), kind="event")
        target = _domain((target_axis, 3), kind="site")
        values = _domain((value_axis, 3), kind="value")

        checked = 0
        for required_bits in range(1 << (source.size * target.size)):
            required = CoordinateRelation(
                source,
                target,
                tuple(
                    _piece(
                        ((source_axis, source_index, source_index + 1, 1),),
                        (target_axis, target_index, target_index + 1, 1),
                    )
                    for source_index in range(source.size)
                    for target_index in range(target.size)
                    if required_bits
                    & (1 << (source_index * target.size + target_index))
                ),
            )
            for support_bits in range(1 << target.size):
                value_by_target = CoordinateRelation.point_map(
                    target,
                    values,
                    tuple(
                        (
                            ((target_axis, target_index, target_index + 1, 1),),
                            (target_index % 2,),
                        )
                        for target_index in range(target.size)
                        if support_bits & (1 << target_index)
                    ),
                )
                reachable = {
                    target_index
                    for source_index in range(source.size)
                    for target_index in range(target.size)
                    if required_bits
                    & (1 << (source_index * target.size + target_index))
                }
                support = {
                    target_index
                    for target_index in range(target.size)
                    if support_bits & (1 << target_index)
                }
                with mock.patch.object(
                    CoordinateRelation,
                    "materialize",
                    side_effect=AssertionError("partial extrema must not enumerate"),
                ):
                    maximum = required.max_target_value_by_source(value_by_target)
                    joint = required.extreme_target_value_and_attainers_by_source(
                        value_by_target,
                        maximize=True,
                    )
                if not reachable <= support:
                    self.assertIsNone(maximum)
                    self.assertIsNone(joint)
                    continue

                self.assertIsNotNone(maximum)
                self.assertIsNotNone(joint)
                assert maximum is not None and joint is not None
                joint_maximum, attainers = joint
                for source_index in range(source.size):
                    targets = tuple(
                        target_index
                        for target_index in range(target.size)
                        if required_bits
                        & (1 << (source_index * target.size + target_index))
                    )
                    expected_value = (
                        frozenset((max(target % 2 for target in targets),))
                        if targets
                        else frozenset()
                    )
                    expected_attainers = frozenset(
                        target_index
                        for target_index in targets
                        if target_index % 2 in expected_value
                    )
                    self.assertEqual(maximum.targets(source_index), expected_value)
                    self.assertEqual(
                        joint_maximum.targets(source_index),
                        expected_value,
                    )
                    self.assertEqual(
                        attainers.targets(source_index),
                        expected_attainers,
                    )
                checked += 1
        self.assertEqual(checked, 125)

    def test_partial_value_support_handles_overlap_and_extra_pieces(self) -> None:
        source = _domain(kind="event")
        target_axis = 20
        target = _domain((target_axis, 5), kind="site")
        value_domain = _domain((30, 16), kind="value")
        target_coordinate = coordinate_axis_symbol(target_axis)
        required = _relation(source, target, _piece((), (target_axis, 1, 4, 1)))
        overlapping_equal_values = _point_map(
            target,
            value_domain,
            (((target_axis, 0, 3, 1),), (target_coordinate + 1,)),
            (((target_axis, 2, 4, 1),), (target_coordinate + 1,)),
            (((target_axis, 2, 4, 1),), (target_coordinate + 1,)),
        )

        result = required.extreme_target_value_and_attainers_by_source(
            overlapping_equal_values,
            maximize=True,
        )

        assert result is not None
        self.assertEqual(result[0].materialize(), (frozenset((4,)),))
        self.assertEqual(result[1].materialize(), (frozenset((3,)),))

        missing_one = _point_map(
            target,
            value_domain,
            (((target_axis, 0, 2, 1),), (target_coordinate,)),
            (((target_axis, 3, 4, 1),), (target_coordinate,)),
        )
        self.assertIsNone(required.max_target_value_by_source(missing_one))
        self.assertIsNone(
            required.extreme_target_value_and_attainers_by_source(
                missing_one,
                maximize=True,
            )
        )
        conflicting_overlap = _point_map(
            target,
            value_domain,
            (((target_axis, 0, 3, 1),), (target_coordinate,)),
            (((target_axis, 2, 4, 1),), (target_coordinate + 1,)),
        )
        self.assertIsNone(
            required.extreme_target_value_and_attainers_by_source(
                conflicting_overlap,
                maximize=True,
            )
        )

    def test_partial_value_support_substitutes_symbolic_boundaries(self) -> None:
        extent = sympy.Symbol("partial_value_extent", integer=True, positive=True)
        source_axis = 10
        target_axis = 20
        source = _domain((source_axis, 2), kind="event")
        target = _domain((target_axis, extent + 6), kind="site")
        value_domain = _domain((30, extent + 7), kind="value")
        target_coordinate = coordinate_axis_symbol(target_axis)
        required = _relation(
            source,
            target,
            _piece(((source_axis, 0, 1, 1),), (target_axis, -2, extent + 3, 1)),
            _piece(((source_axis, 1, 2, 1),), (target_axis, 1, extent + 4, 1)),
        )
        value_by_target = _point_map(
            target,
            value_domain,
            (((target_axis, 0, 2, 1),), (target_coordinate + 1,)),
            (((target_axis, 2, extent + 4, 1),), (target_coordinate + 1,)),
        )

        symbolic = required.extreme_target_value_and_attainers_by_source(
            value_by_target,
            maximize=True,
        )

        assert symbolic is not None
        for concrete_extent in (1, 2, 5):
            substitutions = {extent: concrete_extent}
            concrete = tuple(
                relation.substitute_parameters(substitutions) for relation in symbolic
            )
            self.assertEqual(
                concrete[0].materialize(),
                (
                    frozenset((concrete_extent + 3,)),
                    frozenset((concrete_extent + 4,)),
                ),
            )
            self.assertEqual(
                concrete[1].materialize(),
                (
                    frozenset((concrete_extent + 2,)),
                    frozenset((concrete_extent + 3,)),
                ),
            )

        maybe_empty_extent = sympy.Symbol(
            "partial_value_maybe_empty",
            integer=True,
            nonnegative=True,
        )
        maybe_empty_target = _domain((target_axis, maybe_empty_extent + 1), kind="site")
        conditionally_nonempty = _relation(
            source,
            maybe_empty_target,
            _piece(((source_axis, 0, 1, 1),), (target_axis, 0, maybe_empty_extent, 1)),
        )
        conditional_values = _point_map(
            maybe_empty_target,
            _domain((30, maybe_empty_extent + 2), kind="value"),
            (((target_axis, 0, maybe_empty_extent, 1),), (target_coordinate + 1,)),
        )
        self.assertIsNone(
            conditionally_nonempty.extreme_target_value_and_attainers_by_source(
                conditional_values,
                maximize=True,
            )
        )

    def test_partial_value_support_respects_relation_budget(self) -> None:
        source = _domain(kind="event")
        target_axis = 20
        target = _domain((target_axis, 4), kind="site")
        values = _domain((30, 4), kind="value")
        target_coordinate = coordinate_axis_symbol(target_axis)
        required = _relation(source, target, _piece((), (target_axis, 0, 3, 1)))
        partial = _point_map(
            target,
            values,
            (((target_axis, 0, 1, 1),), (target_coordinate,)),
            (((target_axis, 1, 3, 1),), (target_coordinate,)),
        )
        self.assertIsNotNone(required.max_target_value_by_source(partial))
        with mock.patch(
            "helion._compiler.tile_dependency._MAX_RELATION_PRODUCT_STATES",
            3,
        ):
            self.assertIsNone(required.max_target_value_by_source(partial))

    def test_partial_value_extrema_rejects_out_of_domain_scalar(self) -> None:
        source = _domain(kind="event")
        target = _domain((20, 2), kind="site")
        scalar = _domain((30, 1), kind="value")
        required = _relation(source, target, _piece((), (20, 0, 1, 1)))
        invalid_values = _point_map(
            target, scalar, (((20, 0, 1, 1),), (sympy.Integer(5),))
        )

        self.assertIsNone(required.max_target_value_by_source(invalid_values))
        self.assertIsNone(
            required.extreme_target_value_and_attainers_by_source(
                invalid_values,
                maximize=True,
            )
        )

    def test_target_value_extreme_retains_tied_boxes_and_affine_face(self) -> None:
        source = _domain(kind="event")
        target = _domain((20, 3), (21, 4), kind="site")
        constant_values = _domain((30, 8), kind="value")
        affine_values = _domain((31, 3), kind="value")
        required = CoordinateRelation(
            source,
            target,
            tuple(_piece((), (20, row, row + 1, 1), (21, 0, 4, 1)) for row in range(3)),
        )
        full_target_bounds = ((20, 0, 3, 1), (21, 0, 4, 1))
        constant = _point_map(
            target, constant_values, (full_target_bounds, (sympy.Integer(7),))
        )
        affine = _point_map(
            target, affine_values, (full_target_bounds, (coordinate_axis_symbol(20),))
        )

        constant_maximum = required.extreme_target_value_and_attainers_by_source(
            constant,
            maximize=True,
        )
        affine_maximum = required.extreme_target_value_and_attainers_by_source(
            affine,
            maximize=True,
        )

        self.assertIsNotNone(constant_maximum)
        self.assertIsNotNone(affine_maximum)
        assert constant_maximum is not None
        assert affine_maximum is not None
        self.assertEqual(
            constant_maximum[1].target_coordinates({}),
            frozenset(itertools.product(range(3), range(4))),
        )
        self.assertEqual(
            affine_maximum[1].target_coordinates({}),
            frozenset((2, column) for column in range(4)),
        )

    def test_target_value_extreme_partitions_affine_source_crossing(self) -> None:
        source_axis = 10
        outer_axis = 11
        target_axis = 20
        scale = sympy.Symbol("scale", integer=True, positive=True)
        source = _domain((source_axis, 4), (outer_axis, 2), kind="event")
        target = _domain((target_axis, 4), kind="site")
        values = _domain((30, 4 * scale), kind="value")
        source_coordinate = coordinate_axis_symbol(source_axis)
        outer_coordinate = coordinate_axis_symbol(outer_axis)
        target_coordinate = coordinate_axis_symbol(target_axis)
        source_bounds = ((source_axis, 0, 4, 1), (outer_axis, 0, 2, 1))
        pieces = (
            _piece(
                source_bounds,
                (target_axis, source_coordinate, source_coordinate + 1, 1),
            ),
            _piece(
                source_bounds,
                (target_axis, 3 - source_coordinate, 4 - source_coordinate, 1),
            ),
        )
        value_by_target = _full_point_map(target, values, scale * target_coordinate)

        results = []
        for ordered_pieces in (pieces, tuple(reversed(pieces))):
            required = CoordinateRelation(source, target, ordered_pieces)
            result = required.extreme_target_value_and_attainers_by_source(
                value_by_target,
                maximize=True,
            )
            assert result is not None
            results.append(result)
        self.assertEqual(results[0], results[1])
        for scale_value in (1, 3):
            value_relation, attainer_relation = (
                relation.substitute_parameters({scale: scale_value})
                for relation in results[0]
            )
            for source_coordinate_value in range(4):
                for outer_coordinate_value in range(2):
                    coordinates = {
                        source_axis: source_coordinate_value,
                        outer_axis: outer_coordinate_value,
                    }
                    expected = max(
                        source_coordinate_value,
                        3 - source_coordinate_value,
                    )
                    self.assertEqual(
                        value_relation.target_coordinates(coordinates),
                        frozenset(((scale_value * expected,),)),
                    )
                    self.assertEqual(
                        attainer_relation.target_coordinates(coordinates),
                        frozenset(((expected,),)),
                    )

        multi_axis_pieces = (
            pieces[0],
            _piece(
                source_bounds, (target_axis, outer_coordinate, outer_coordinate + 1, 1)
            ),
        )
        self.assertIsNone(
            CoordinateRelation(
                source, target, multi_axis_pieces
            ).extreme_target_value_and_attainers_by_source(
                value_by_target,
                maximize=True,
            )
        )

        with mock.patch(
            "helion._compiler.tile_dependency._MAX_RELATION_PRODUCT_STATES",
            4,
        ):
            self.assertIsNone(
                CoordinateRelation(
                    source, target, pieces
                ).extreme_target_value_and_attainers_by_source(
                    value_by_target,
                    maximize=True,
                )
            )

    def test_target_value_extreme_handles_identity_axis_alias(self) -> None:
        domain = _domain((20, 4), kind="site")
        values = _domain((30, 4), kind="value")
        value_by_target = _full_point_map(domain, values, coordinate_axis_symbol(20))

        result = CoordinateRelation.identity(
            domain,
            domain,
        ).extreme_target_value_and_attainers_by_source(
            value_by_target,
            maximize=True,
        )

        assert result is not None
        expected = tuple(frozenset((coordinate,)) for coordinate in range(4))
        self.assertEqual(result[0].materialize(), expected)
        self.assertEqual(result[1].materialize(), expected)

    def test_target_value_extreme_clips_to_semantic_target_domain(self) -> None:
        source = _domain(kind="event")
        target = _domain((20, 3), kind="site")
        values = _domain((30, 21), kind="value")
        target_coordinate = coordinate_axis_symbol(20)
        required = _relation(source, target, _piece((), (20, 0, 10, 1)))
        value_by_target = _point_map(
            target, values, (((20, -10, 10, 1),), (target_coordinate + 10,))
        )

        maximum = required.extreme_target_value_and_attainers_by_source(
            value_by_target,
            maximize=True,
        )
        minimum = required.extreme_target_value_and_attainers_by_source(
            value_by_target,
            maximize=False,
        )
        legacy_maximum = required.max_target_value_by_source(value_by_target)

        self.assertIsNotNone(maximum)
        self.assertIsNotNone(minimum)
        self.assertIsNotNone(legacy_maximum)
        assert maximum is not None
        assert minimum is not None
        assert legacy_maximum is not None
        self.assertEqual(maximum[0].materialize(), (frozenset((12,)),))
        self.assertEqual(maximum[1].materialize(), (frozenset((2,)),))
        self.assertEqual(minimum[0].materialize(), (frozenset((10,)),))
        self.assertEqual(minimum[1].materialize(), (frozenset((0,)),))
        self.assertEqual(legacy_maximum.materialize(), maximum[0].materialize())

        extent = sympy.Symbol("extent", integer=True, positive=True)
        symbolic_target = _domain((20, extent + 2), kind="site")
        symbolic_values = _domain((30, extent + 12), kind="value")
        strided_required = _relation(
            source,
            symbolic_target,
            _piece((), (20, -3, extent + 9, 2)),
        )
        symbolic_value_by_target = _full_point_map(
            symbolic_target, symbolic_values, target_coordinate + 10
        )
        symbolic_maximum = (
            strided_required.extreme_target_value_and_attainers_by_source(
                symbolic_value_by_target,
                maximize=True,
            )
        )
        symbolic_minimum = (
            strided_required.extreme_target_value_and_attainers_by_source(
                symbolic_value_by_target,
                maximize=False,
            )
        )
        self.assertIsNotNone(symbolic_maximum)
        self.assertIsNotNone(symbolic_minimum)
        assert symbolic_maximum is not None
        assert symbolic_minimum is not None
        for concrete_extent in range(1, 6):
            substitutions = {extent: concrete_extent}
            semantic_targets = tuple(range(0, concrete_extent + 2, 2))
            self.assertEqual(
                symbolic_maximum[0].substitute_parameters(substitutions).materialize(),
                (frozenset((max(semantic_targets) + 10,)),),
            )
            self.assertEqual(
                symbolic_maximum[1].substitute_parameters(substitutions).materialize(),
                (frozenset((max(semantic_targets),)),),
            )
            self.assertEqual(
                symbolic_minimum[0].substitute_parameters(substitutions).materialize(),
                (frozenset((min(semantic_targets) + 10,)),),
            )
            self.assertEqual(
                symbolic_minimum[1].substitute_parameters(substitutions).materialize(),
                (frozenset((min(semantic_targets),)),),
            )

    def test_target_value_extreme_declines_unrepresentable_guards_and_levels(
        self,
    ) -> None:
        parameter = sympy.Symbol("parameter", integer=True, positive=True)
        source = _domain(kind="event")
        target = _domain((20, 2), kind="site")
        values = _domain((30, parameter + 5), kind="value")
        required = _relation(
            source,
            target,
            _piece((), (20, sympy.Integer(0), sympy.Integer(1), 1)),
            _piece((), (20, sympy.Integer(1), sympy.Integer(2), 1)),
        )
        parameter_crossing = _point_map(
            target,
            values,
            (((20, 0, 1, 1),), (parameter,)),
            (((20, 1, 2, 1),), (sympy.Integer(4),)),
        )

        self.assertIsNotNone(required.max_target_value_by_source(parameter_crossing))
        self.assertIsNone(
            required.extreme_target_value_and_attainers_by_source(
                parameter_crossing,
                maximize=True,
            )
        )

        square = _domain((40, 4), (41, 4), kind="site")
        diagonal_values = _domain((50, 2), kind="value")
        row = coordinate_axis_symbol(40)
        column = coordinate_axis_symbol(41)
        nonrectangular = _full_point_map(
            square, diagonal_values, sympy.floor((row + column) / 4)
        )
        self.assertIsNone(
            CoordinateRelation.total(
                source, square
            ).extreme_target_value_and_attainers_by_source(
                nonrectangular,
                maximize=True,
            )
        )

    def test_symbolic_max_target_value_reduces_schedule_positions(self) -> None:
        elements = 128
        plan = build_tile_dependency_graph(
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
                    shape=(elements,),
                    block_ids=(20,),
                ),
            ),
            [[10], [20]],
        )
        relation = _symbolic_root_relation(
            plan,
            {10: (8, 16), 20: (4, 32)},
        )
        assert relation is not None
        producer_axis = relation.target_domain.axis_order[0]
        value_domain = _domain((0, 4), kind="value")
        worker_steps = _point_map(
            relation.target_domain,
            value_domain,
            (
                ((producer_axis, 0, 8, 1),),
                (sympy.floor((coordinate_axis_symbol(producer_axis) + 3) / 4),),
            ),
        )

        maximum = relation.max_target_value_by_source(worker_steps)

        assert maximum is not None
        self.assertEqual(
            maximum.materialize(),
            tuple(
                frozenset(
                    (max(max(worker_steps.targets(task)) for task in producer_tasks),)
                )
                for producer_tasks in relation.materialize()
            ),
        )

    def test_symbolic_max_target_value_handles_strided_symbolic_begin(self) -> None:
        consumer_axis = 57
        producer_axis = 29
        consumers = _domain((consumer_axis, 64), kind="task_order")
        producers = _domain((producer_axis, 8), kind="task_order")
        required_producers = _relation(
            consumers,
            producers,
            _piece(
                ((consumer_axis, 0, 64, 1),),
                (
                    producer_axis,
                    sympy.Mod(coordinate_axis_symbol(consumer_axis), 4),
                    sympy.Integer(8),
                    4,
                ),
            ),
        )

        maximum = required_producers.max_target_value_by_source(
            CoordinateRelation.identity(producers, producers)
        )

        assert maximum is not None
        self.assertEqual(
            maximum.materialize(),
            tuple(frozenset((4 + consumer % 4,)) for consumer in range(64)),
        )

    def test_max_target_value_respects_relation_product_budget(self) -> None:
        source = _domain((10, 2), kind="task_order")
        target = _domain((20, 2), kind="task_order")
        values = _domain((0, 2), kind="value")
        source_coordinate = coordinate_axis_symbol(10)
        target_coordinate = coordinate_axis_symbol(20)
        source_pieces = (
            ((10, 0, 1, 1),),
            ((10, 1, 2, 1),),
        )
        target_pieces = (
            ((20, 0, 1, 1),),
            ((20, 1, 2, 1),),
        )
        required = CoordinateRelation.point_map(
            source,
            target,
            tuple((bounds, (source_coordinate,)) for bounds in source_pieces),
        )
        value_by_target = CoordinateRelation.point_map(
            target,
            values,
            tuple((bounds, (target_coordinate,)) for bounds in target_pieces),
        )

        self.assertIsNotNone(required.max_target_value_by_source(value_by_target))
        self.assertIsNotNone(
            required.extreme_target_value_and_attainers_by_source(
                value_by_target,
                maximize=True,
            )
        )
        with mock.patch(
            "helion._compiler.tile_dependency._MAX_RELATION_PRODUCT_STATES",
            1,
        ):
            self.assertIsNone(required.max_target_value_by_source(value_by_target))
            self.assertIsNone(
                required.extreme_target_value_and_attainers_by_source(
                    value_by_target,
                    maximize=True,
                )
            )
        with mock.patch(
            "helion._compiler.tile_dependency._MAX_RELATION_PIECES",
            1,
        ):
            self.assertIsNone(required.max_target_value_by_source(value_by_target))
            self.assertIsNone(
                required.extreme_target_value_and_attainers_by_source(
                    value_by_target,
                    maximize=True,
                )
            )

    def test_symbolic_max_target_value_respects_stride_alignment(self) -> None:
        consumer_axis = 57
        producer_axis = 29
        value_axis = -1
        consumers = _domain((consumer_axis, 1), kind="task_order")
        producers = _domain((producer_axis, 8), kind="task_order")
        values = _domain((value_axis, 108), kind="value")
        required_producers = _relation(
            consumers,
            producers,
            _piece(
                ((consumer_axis, 0, 1, 1),),
                (producer_axis, sympy.Integer(1), sympy.Integer(8), 2),
            ),
        )
        producer_values = _relation(
            producers,
            values,
            _piece(
                ((producer_axis, 0, 8, 2),),
                (
                    value_axis,
                    coordinate_axis_symbol(producer_axis) + 100,
                    coordinate_axis_symbol(producer_axis) + 101,
                    1,
                ),
            ),
            _piece(
                ((producer_axis, 1, 8, 2),),
                (
                    value_axis,
                    coordinate_axis_symbol(producer_axis),
                    coordinate_axis_symbol(producer_axis) + 1,
                    1,
                ),
            ),
        )

        maximum = required_producers.max_target_value_by_source(producer_values)

        assert maximum is not None
        self.assertEqual(maximum.materialize(), (frozenset((7,)),))

    def test_symbolic_max_target_value_composes_qwen_mixed_radix_quotient(
        self,
    ) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        order = _domain((10, batch), (11, 16), (12, 96), kind="task_order")
        tasks = _domain((20, batch), (21, 1536), kind="site")
        order_batch = coordinate_axis_symbol(10)
        inner = coordinate_axis_symbol(11)
        iteration = coordinate_axis_symbol(12)
        task_order = _point_map(
            order,
            tasks,
            (
                ((10, 0, batch, 1), (11, 0, 8, 1), (12, 0, 96, 1)),
                (order_batch, 8 * iteration + inner),
            ),
            (
                ((10, 0, batch, 1), (11, 8, 16, 1), (12, 0, 96, 1)),
                (order_batch, 8 * iteration + inner + 760),
            ),
        )
        waves = _domain((30, 2), kind="worker")
        wave_by_order = _full_point_map(
            order, waves, sympy.floor((16 * iteration + inner) / 1184)
        )
        keys = _domain((40, batch), (41, 96), kind="event")
        key_batch = coordinate_axis_symbol(40)
        key_iteration = coordinate_axis_symbol(41)
        required_producers = _relation(
            keys,
            tasks,
            _piece(
                ((40, 0, batch, 1), (41, 0, 96, 1)),
                (20, key_batch, key_batch + 1, 1),
                (21, 8 * key_iteration, 8 * key_iteration + 8, 1),
            ),
            _piece(
                ((40, 0, batch, 1), (41, 0, 96, 1)),
                (20, key_batch, key_batch + 1, 1),
                (21, 8 * key_iteration + 768, 8 * key_iteration + 776, 1),
            ),
        )

        positional_product = task_order._positional_product
        assert positional_product is not None
        self.assertEqual(positional_product[0], ((10, 20),))
        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic frontier must not enumerate"),
        ):
            tasks_to_order = task_order.converse()
            assert tasks_to_order is not None
            wave_by_task = tasks_to_order.then(wave_by_order)
            assert wave_by_task is not None
            maximum = required_producers.max_target_value_by_source(wave_by_task)

        self.assertIsNotNone(maximum)
        assert maximum is not None
        self.assertEqual(len(maximum.pieces), 1)
        (_axis, frontier, _end, _step) = maximum.pieces[0].target_ranges[0]
        self.assertEqual(
            frontier,
            sympy.floor(key_iteration / 74 + sympy.Rational(15, 1184)),
        )
        for concrete_batch in (0, 1, 2):
            substitutions = {batch: concrete_batch}
            concrete_required = required_producers.substitute_parameters(substitutions)
            concrete_values = wave_by_task.substitute_parameters(substitutions)
            concrete_maximum = maximum.substitute_parameters(substitutions)
            values = concrete_values.materialize()
            oracle = tuple(
                frozenset((max(max(values[task]) for task in producer_tasks),))
                for producer_tasks in concrete_required.materialize()
            )
            self.assertEqual(concrete_maximum.materialize(), oracle)
            self.assertEqual(
                sum(next(iter(value)) for value in oracle),
                22 * concrete_batch,
            )

    def test_symbolic_max_target_value_aligned_modulo_catalog(self) -> None:
        for modulus in (4, 6, 8, 12):
            for width in range(1, modulus + 1):
                if modulus % width:
                    continue
                with self.subTest(modulus=modulus, width=width):
                    group_count = modulus // width
                    keys = _domain((10, group_count), kind="event")
                    tasks = _domain((20, width), (21, group_count), kind="site")
                    key = coordinate_axis_symbol(10)
                    inner = coordinate_axis_symbol(20)
                    group = coordinate_axis_symbol(21)
                    required = _relation(
                        keys,
                        tasks,
                        _piece(
                            ((10, 0, group_count, 1),),
                            (20, sympy.Integer(0), sympy.Integer(width), 1),
                            (21, key, key + 1, 1),
                        ),
                    )
                    quotient_width = 2
                    values = _domain(
                        (30, (modulus + quotient_width - 1) // quotient_width),
                        kind="worker",
                    )
                    value_by_task = _full_point_map(
                        tasks,
                        values,
                        sympy.floor(
                            sympy.Mod(inner + width * group, modulus) / quotient_width
                        ),
                    )

                    maximum = required.max_target_value_by_source(value_by_task)

                    assert maximum is not None
                    values_by_task = value_by_task.materialize()
                    oracle = tuple(
                        frozenset(
                            (max(max(values_by_task[task]) for task in tasks_for_key),)
                        )
                        for tasks_for_key in required.materialize()
                    )
                    self.assertEqual(maximum.materialize(), oracle)

        batch = sympy.Symbol("batch", integer=True, positive=True)
        keys = _domain((40, batch), kind="event")
        tasks = _domain((50, batch), (51, 4), kind="site")
        key_batch = coordinate_axis_symbol(40)
        task_batch = coordinate_axis_symbol(50)
        inner = coordinate_axis_symbol(51)
        required = _relation(
            keys,
            tasks,
            _piece(
                ((40, 0, batch, 1),),
                (50, key_batch, key_batch + 1, 1),
                (51, sympy.Integer(0), sympy.Integer(4), 1),
            ),
        )
        values = _domain((60, 8), kind="worker")
        value_by_task = _point_map(
            tasks,
            values,
            (
                ((50, 0, batch, 1), (51, 0, 4, 1)),
                (sympy.Mod(inner + 4 * sympy.Mod(task_batch, 2), 8),),
            ),
        )
        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic outer parity must not enumerate"),
        ):
            maximum = required.max_target_value_by_source(value_by_task)
        self.assertIsNotNone(maximum)
        assert maximum is not None
        for concrete_batch in (1, 2, 5):
            substitutions = {batch: concrete_batch}
            concrete_required = required.substitute_parameters(substitutions)
            concrete_values = value_by_task.substitute_parameters(substitutions)
            concrete_maximum = maximum.substitute_parameters(substitutions)
            values_by_task = concrete_values.materialize()
            oracle = tuple(
                frozenset((max(max(values_by_task[task]) for task in tasks_for_key),))
                for tasks_for_key in concrete_required.materialize()
            )
            self.assertEqual(concrete_maximum.materialize(), oracle)

    def test_symbolic_max_target_value_declines_wrapping_modulo_fiber(
        self,
    ) -> None:
        keys = _domain((10, 4), kind="event")
        tasks = _domain((20, 8), (21, 4), kind="site")
        key = coordinate_axis_symbol(10)
        head = coordinate_axis_symbol(20)
        column = coordinate_axis_symbol(21)
        required_producers = _relation(
            keys,
            tasks,
            _piece(
                ((10, 0, 4, 1),),
                (20, sympy.Integer(0), sympy.Integer(8), 1),
                (21, key, key + 1, 1),
            ),
        )
        values = _domain((30, 4), kind="worker")
        cyclic_wave = _full_point_map(
            tasks, values, sympy.floor(sympy.Mod(head + 8 * column + 1, 32) / 8)
        )

        # The shifted mixed-radix rank wraps within the final fiber.  Its
        # maximum would require another source partition, so the bounded
        # extrema proof must decline instead of selecting an endpoint.
        self.assertIsNone(required_producers.max_target_value_by_source(cyclic_wave))

        for modulus in (4, 6, 8, 12):
            for width in range(2, modulus + 1):
                if modulus % width:
                    continue
                with self.subTest(modulus=modulus, width=width):
                    group_count = modulus // width
                    small_keys = _domain((40, group_count), kind="event")
                    small_tasks = _domain((50, width), (51, group_count), kind="site")
                    small_key = coordinate_axis_symbol(40)
                    small_inner = coordinate_axis_symbol(50)
                    small_group = coordinate_axis_symbol(51)
                    small_required = _relation(
                        small_keys,
                        small_tasks,
                        _piece(
                            ((40, 0, group_count, 1),),
                            (50, sympy.Integer(0), sympy.Integer(width), 1),
                            (51, small_key, small_key + 1, 1),
                        ),
                    )
                    small_values = _domain((60, modulus), kind="worker")
                    wrapping_value = _full_point_map(
                        small_tasks,
                        small_values,
                        sympy.Mod(small_inner + width * small_group + 1, modulus),
                    )
                    self.assertIsNone(
                        small_required.max_target_value_by_source(wrapping_value)
                    )

    def test_out_of_domain_point_map_is_not_total(self) -> None:
        source = _domain((10, 6), identity=0)
        target = _domain((0, 2), kind="event", identity=0)
        relation = _full_point_map(
            source, target, sympy.floor(coordinate_axis_symbol(10) / 2)
        )

        self.assertFalse(relation.has_total_source())
        self.assertFalse(relation.is_total_function())
        self.assertEqual(relation.materialize()[-2:], (frozenset(), frozenset()))

    def test_symbolic_quotient_point_map_respects_ceildiv_domain(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((10, batch), (11, 3), identity=0)
        outer = coordinate_axis_symbol(10)
        inner = coordinate_axis_symbol(11)
        numerator = 3 * outer + inner

        def relation(target_count: sympy.Expr) -> CoordinateRelation:
            target = _domain((20, target_count), kind="worker")
            return _full_point_map(source, target, FloorDiv(numerator, 4))

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic bound proof must not enumerate"),
        ):
            self.assertTrue(relation(FloorDiv(3 * batch + 3, 4)).is_total_function())
            self.assertFalse(relation(FloorDiv(3 * batch + 2, 4)).is_total_function())

    def test_out_of_domain_source_support_is_not_counted_as_total(self) -> None:
        source = _domain((10, 2), identity=0)
        target = _domain((20, 1), identity=1)
        relation = _point_map(source, target, (((10, -1, 1, 1),), (sympy.Integer(0),)))

        self.assertIsNone(relation.source_support_cardinality())
        self.assertFalse(relation.is_total_function())

    def test_symbolic_source_support_cardinality_is_exact_without_enumeration(
        self,
    ) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((10, batch), (11, 4), identity=0)
        target = _domain((20, batch), (21, 4), identity=1)
        outer = coordinate_axis_symbol(10)
        inner = coordinate_axis_symbol(11)
        relation = _point_map(
            source,
            target,
            (((10, 0, batch, 1), (11, 0, 2, 1)), (outer, inner)),
            (((10, 0, batch, 1), (11, 2, 4, 1)), (outer, inner)),
        )

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic support must not enumerate"),
        ):
            cardinality = relation.source_support_cardinality()
        self.assertEqual(sympy.simplify(cardinality - 4 * batch), 0)
        self.assertTrue(relation.is_total_function())
        for concrete_batch in (0, 1, 31):
            concrete = relation.substitute_parameters({batch: concrete_batch})
            self.assertEqual(
                concrete.source_support_cardinality(),
                4 * concrete_batch,
            )

    def test_symbolic_strided_source_support_cardinality_is_exact(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((10, 2 * batch), identity=0)
        target = _domain((20, 1), identity=1)
        relation = _relation(
            source,
            target,
            _piece(
                ((10, 0, 2 * batch, 2),), (20, sympy.Integer(0), sympy.Integer(1), 1)
            ),
        )

        cardinality = relation.source_support_cardinality()

        self.assertIsNotNone(cardinality)
        self.assertEqual(sympy.simplify(cardinality - batch), 0)

    def test_packed_partial_bijections_and_adjacent_support_are_symbolic(
        self,
    ) -> None:
        left_count = sympy.Symbol("left_count", integer=True, nonnegative=True)
        right_count = sympy.Symbol("right_count", integer=True, nonnegative=True)
        worker_count = 7
        worker_axis, wave_axis = 11, 12
        schedule = _domain(
            (10, 2),
            (worker_axis, worker_count),
            (
                wave_axis,
                FloorDiv(left_count + right_count + worker_count - 1, worker_count),
            ),
            kind="worker",
        )

        def packed(
            target_axis: int,
            first_slot: sympy.Expr,
            count: sympy.Expr,
        ) -> CoordinateRelation:
            worker = coordinate_axis_symbol(worker_axis)
            wave = coordinate_axis_symbol(wave_axis)
            first_wave = FloorDiv(first_slot, worker_count)
            first_worker = sympy.Mod(first_slot, worker_count)
            first_count = SymbolicMin(count, worker_count - first_worker)
            remaining = SymbolicMax(count - first_count, 0)
            full_waves = FloorDiv(remaining, worker_count)
            tail_count = sympy.Mod(remaining, worker_count)
            middle_begin = first_wave + 1
            middle_end = middle_begin + full_waves
            final_end = middle_end + FloorDiv(
                tail_count + worker_count - 1,
                worker_count,
            )
            target = _domain((target_axis, count), kind="task_order")
            logical_task = wave * worker_count + worker - first_slot
            return _point_map(
                schedule,
                target,
                (
                    (
                        (10, 1, 2, 1),
                        (worker_axis, first_worker, first_worker + first_count, 1),
                        (wave_axis, first_wave, first_wave + 1, 1),
                    ),
                    (logical_task,),
                ),
                (
                    (
                        (10, 1, 2, 1),
                        (worker_axis, 0, worker_count, 1),
                        (wave_axis, middle_begin, middle_end, 1),
                    ),
                    (logical_task,),
                ),
                (
                    (
                        (10, 1, 2, 1),
                        (worker_axis, 0, tail_count, 1),
                        (wave_axis, middle_end, final_end, 1),
                    ),
                    (logical_task,),
                ),
            )

        left = packed(20, sympy.Integer(0), left_count)
        right = packed(21, left_count, right_count)
        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic traversal proof must not enumerate"),
        ):
            self.assertEqual(left.source_support_cardinality(), left_count)
            self.assertEqual(right.source_support_cardinality(), right_count)
            self.assertTrue(left.is_bijection_from_source_support())
            self.assertTrue(right.is_bijection_from_source_support())
            self.assertTrue(left.has_disjoint_source_support(right))

        for concrete_left, concrete_right in ((0, 0), (1, 6), (7, 1), (8, 13)):
            substitutions = {
                left_count: concrete_left,
                right_count: concrete_right,
            }
            for relation, expected_count in (
                (left, concrete_left),
                (right, concrete_right),
            ):
                concrete = relation.substitute_parameters(substitutions)
                self.assertEqual(concrete.source_support_cardinality(), expected_count)

    def test_singleton_set_converse_preserves_exact_symbolic_support(self) -> None:
        count = sympy.Symbol("count", integer=True, positive=True)
        marker = _domain(kind="value")
        order = _domain((10, 2 * count + 2), kind="task_order")
        selected = _relation(
            marker,
            order,
            _piece((), (10, 2 * count, 2 * count + 2, 1)),
        )

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic support proof must not enumerate"),
        ):
            membership = selected.converse()
            assert membership is not None
            self.assertEqual(membership.source_support_cardinality(), 2)
            self.assertEqual(
                _dense_linear_source_support_interval(membership, (10,)),
                (2 * count, 2 * count + 2),
            )

        clipped = _relation(
            marker,
            _domain((10, count), kind="task_order"),
            _piece((), (10, -1, count, 2)),
        )
        self.assertIsNone(clipped.converse())

    def test_partial_bijection_dense_interval_and_ordinalization_are_exact(
        self,
    ) -> None:
        rows = sympy.Symbol("rows", integer=True, nonnegative=True)
        stage_axis, worker_axis, wave_axis = 10, 11, 12
        schedule = _domain(
            (stage_axis, 2), (worker_axis, 4), (wave_axis, rows + 2), kind="worker"
        )
        logical = _domain((20, 4 * rows + 4), kind="site")
        worker = coordinate_axis_symbol(worker_axis)
        wave = coordinate_axis_symbol(wave_axis)
        logical_task = coordinate_axis_symbol(20)
        support_bounds = (
            (stage_axis, 1, 2, 1),
            (worker_axis, 0, 4, 1),
            (wave_axis, 1, rows + 1, 1),
        )
        relation = _point_map(
            schedule, logical, (support_bounds, (4 * (wave - 1) + worker,))
        )
        converse = _point_map(
            logical,
            schedule,
            (
                ((20, 0, 4 * rows, 1),),
                (
                    sympy.Integer(1),
                    sympy.Mod(logical_task, 4),
                    FloorDiv(logical_task, 4) + 1,
                ),
            ),
        )
        _remember_exact_converse(relation, converse)
        ordinal_domain = _domain((30, 4 * rows), kind="task_order", allow_empty=True)
        support = _point_map(
            schedule, ordinal_domain, (support_bounds, (sympy.Integer(0),))
        )

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("partial support proof must not enumerate"),
        ):
            self.assertEqual(
                _dense_linear_source_support_interval(
                    relation,
                    (worker_axis, wave_axis),
                ),
                (sympy.Integer(4), 4 * rows + 4),
            )
            ordinalization = support._ordinalized_source_support
            assert ordinalization is not None
            self.assertIsNotNone(ordinalization.converse())

        hole_target = _domain((40, 2), kind="site")
        hole = _point_map(
            schedule,
            hole_target,
            (
                ((stage_axis, 1, 2, 1), (worker_axis, 0, 1, 1), (wave_axis, 1, 2, 1)),
                (sympy.Integer(0),),
            ),
            (
                ((stage_axis, 1, 2, 1), (worker_axis, 2, 3, 1), (wave_axis, 1, 2, 1)),
                (sympy.Integer(1),),
            ),
        )
        hole_inverse = _point_map(
            hole_target,
            schedule,
            (((40, 0, 1, 1),), (sympy.Integer(1), 0, 1)),
            (((40, 1, 2, 1),), (sympy.Integer(1), 2, 1)),
        )
        _remember_exact_converse(hole, hole_inverse)
        self.assertIsNone(
            _dense_linear_source_support_interval(
                hole,
                (worker_axis, wave_axis),
            )
        )

        projected_source = _domain((50, 2), (51, 3), kind="worker")
        projected_target = _domain((60, 3), kind="site")
        omitted = coordinate_axis_symbol(50)
        projected = _point_map(
            projected_source,
            projected_target,
            (((50, 0, 2, 1), (51, 0, 1, 1)), (omitted,)),
            (((50, 0, 1, 1), (51, 2, 3, 1)), (sympy.Integer(2),)),
        )
        projected_inverse = _point_map(
            projected_target,
            projected_source,
            (((60, 0, 2, 1),), (coordinate_axis_symbol(60), 0)),
            (((60, 2, 3, 1),), (sympy.Integer(0), 2)),
        )
        _remember_exact_converse(projected, projected_inverse)
        self.assertIsNone(_dense_linear_source_support_interval(projected, (51,)))

    def test_event_frontier_partial_bijection_is_symbolic(self) -> None:
        count = sympy.Symbol("count", integer=True, nonnegative=True)
        worker_count, period, phase = 7, 3, 2
        source = _domain(
            (10, 2),
            (11, worker_count),
            (12, period * FloorDiv(count + worker_count - 1, worker_count)),
            kind="worker",
        )
        target = _domain((20, count), kind="task_order")
        worker = coordinate_axis_symbol(11)
        wave = coordinate_axis_symbol(12)
        full_waves = FloorDiv(count, worker_count)
        tail_count = sympy.Mod(count, worker_count)
        tail_wave = phase + period * full_waves
        relation = _point_map(
            source,
            target,
            (
                (
                    (10, 1, 2, 1),
                    (11, 0, worker_count, 1),
                    (12, phase, tail_wave, period),
                ),
                (sympy.floor(wave / period) * worker_count + worker,),
            ),
            (
                (
                    (10, 1, 2, 1),
                    (11, 0, tail_count, 1),
                    (
                        12,
                        tail_wave,
                        tail_wave
                        + FloorDiv(tail_count + worker_count - 1, worker_count),
                        1,
                    ),
                ),
                (sympy.floor(wave / period) * worker_count + worker,),
            ),
        )

        with mock.patch.object(
            CoordinateRelation,
            "materialize",
            side_effect=AssertionError("symbolic traversal proof must not enumerate"),
        ):
            self.assertEqual(relation.source_support_cardinality(), count)
            self.assertTrue(relation.is_bijection_from_source_support())
            self.assertIsNotNone(relation.converse())
        for concrete_count in (0, 1, 6, 7, 8):
            concrete = relation.substitute_parameters({count: concrete_count})
            self.assertEqual(concrete.source_support_cardinality(), concrete_count)
            self.assertTrue(concrete.is_bijection_from_source_support())

    def test_equal_cardinality_does_not_prove_bijection(self) -> None:
        source = _domain((10, 2), identity=0)
        target = _domain((20, 2), identity=1)
        relation = _point_map(
            source,
            target,
            (((10, 0, 1, 1),), (sympy.Integer(0),)),
            (((10, 1, 2, 1),), (sympy.Integer(0),)),
        )

        self.assertEqual(relation.source_support_cardinality(), 2)
        self.assertFalse(relation.is_bijection_from_source_support())

    def test_point_map_equality_on_reordered_symbolic_support(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        source = _domain((10, batch), (11, 4), identity=0)
        target = _domain((20, batch), (21, 4), identity=1)
        outer = coordinate_axis_symbol(10)
        inner = coordinate_axis_symbol(11)
        pieces = (
            (((10, 0, batch, 1), (11, 0, 2, 1)), (outer, inner)),
            (((10, 0, batch, 1), (11, 2, 4, 1)), (outer, inner)),
        )
        reference = CoordinateRelation.point_map(source, target, pieces)
        equivalent = CoordinateRelation.point_map(
            source,
            target,
            tuple(
                (
                    bounds,
                    (mapped_outer + FloorDiv(inner, 4), mapped_inner),
                )
                for bounds, (mapped_outer, mapped_inner) in reversed(pieces)
            ),
        )
        different_map = _point_map(
            source, target, pieces[0], (pieces[1][0], (outer, inner - 1))
        )
        different_support = _point_map(source, target, pieces[0])

        self.assertTrue(reference.is_pointwise_equal_on_same_support(equivalent))
        self.assertTrue(equivalent.is_pointwise_equal_on_same_support(reference))
        self.assertFalse(reference.is_pointwise_equal_on_same_support(different_map))
        self.assertFalse(
            reference.is_pointwise_equal_on_same_support(different_support)
        )

    def test_empty_target_is_not_counted_as_source_support(self) -> None:
        source = _domain((10, 2), identity=0)
        target = _domain((20, 1), identity=1)
        coordinate = coordinate_axis_symbol(10)
        relation = _full_point_map(source, target, coordinate + 1)

        self.assertEqual(relation.materialize(), (frozenset(), frozenset()))
        self.assertIsNone(relation.source_support_cardinality())
        self.assertFalse(relation.is_total_function())

    def test_pointwise_order_where_defined_is_proved_on_common_affine_partition(
        self,
    ) -> None:
        source = _domain((10, 8), identity=0)
        values = _domain((0, 32), kind="value")
        coordinate = coordinate_axis_symbol(10)
        left = _point_map(
            source,
            values,
            (((10, 0, 4, 1),), (coordinate,)),
            (((10, 4, 8, 1),), (coordinate + 2,)),
        )
        right = _point_map(
            source,
            values,
            (((10, 0, 2, 1),), (coordinate + 1,)),
            (((10, 2, 4, 1),), (coordinate + 3,)),
            (((10, 4, 8, 1),), (coordinate + 3,)),
        )

        self.assertTrue(left.is_pointwise_equal_to(left))

        partial_right = _point_map(
            source, values, (((10, 0, 7, 1),), (coordinate + 1,))
        )
        partial_left = _point_map(source, values, (((10, 0, 7, 1),), (coordinate,)))
        self.assertTrue(
            partial_left.is_pointwise_strictly_less_than_where_defined(right)
        )
        self.assertFalse(partial_right.is_pointwise_equal_to(right))

    def test_pointwise_strict_order_uses_symbolic_support_cardinality(self) -> None:
        count = sympy.Symbol("count", integer=True, nonnegative=True)
        source = _domain((10, count), identity=0, allow_empty=True)
        values = _domain((0, 2 * count + 1), kind="value")
        coordinate = coordinate_axis_symbol(10)
        left = _full_point_map(source, values, coordinate)
        right = _full_point_map(source, values, coordinate + count)

        self.assertTrue(left.is_pointwise_strictly_less_than_where_defined(right))
        self.assertFalse(right.is_pointwise_strictly_less_than_where_defined(left))

    def test_pointwise_strict_order_respects_relation_piece_budget(self) -> None:
        source = _domain((10, 2), identity=0)
        values = _domain((0, 4), kind="value")
        coordinate = coordinate_axis_symbol(10)
        source_pieces = (
            ((10, 0, 1, 1),),
            ((10, 1, 2, 1),),
        )
        left = CoordinateRelation.point_map(
            source,
            values,
            tuple((bounds, (coordinate,)) for bounds in source_pieces),
        )
        right = CoordinateRelation.point_map(
            source,
            values,
            tuple((bounds, (coordinate + 1,)) for bounds in source_pieces),
        )

        self.assertTrue(left.is_pointwise_strictly_less_than_where_defined(right))
        with mock.patch(
            "helion._compiler.tile_dependency._MAX_RELATION_PIECES",
            1,
        ):
            self.assertFalse(left.is_pointwise_strictly_less_than_where_defined(right))

    def test_partitioned_total_function_avoids_global_canonicalization(self) -> None:
        source = _domain((10, 128), identity=0)
        target = _domain((20, 128), identity=1)
        relation = CoordinateRelation.point_map(
            source,
            target,
            tuple(
                (
                    ((10, index, index + 1, 1),),
                    (sympy.Integer(index),),
                )
                for index in range(source.size)
            ),
        )

        with mock.patch.object(
            CoordinateRelation,
            "canonical_single_valued",
            side_effect=AssertionError("slow fallback should not run"),
        ):
            self.assertTrue(relation.is_total_function())

    def test_symbolic_dependency_matches_enumerated_overlap(self) -> None:
        for elements, producer_block, consumer_block in (
            (1, 1, 1),
            (31, 8, 16),
            (33, 16, 8),
            (65, 16, 24),
            (127, 32, 48),
        ):
            with self.subTest(
                elements=elements,
                producer_block=producer_block,
                consumer_block=consumer_block,
            ):
                plan = build_tile_dependency_graph(
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
                            shape=(elements,),
                            block_ids=(20,),
                        ),
                    ),
                    [[10], [20]],
                )
                producer_count = (elements + producer_block - 1) // producer_block
                consumer_count = (elements + consumer_block - 1) // consumer_block
                root_domains = _one_dimensional_domains(
                    producer_count=producer_count,
                    consumer_count=consumer_count,
                    producer_block=producer_block,
                    consumer_block=consumer_block,
                )
                relation = _symbolic_root_relation(
                    plan,
                    {
                        10: (producer_count, producer_block),
                        20: (consumer_count, consumer_block),
                    },
                )

                assert relation is not None
                self.assertEqual(
                    relation.materialize(),
                    _root_producers_by_consumer(plan, root_domains),
                )

    def test_symbolic_dependency_keeps_batch_axis(self) -> None:
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
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
            ),
            [[10, 11], [20, 21]],
        )
        axis_geometry = {
            10: (2, 1),
            11: (4, 16),
            20: (2, 1),
            21: (2, 32),
        }

        relation = _symbolic_root_relation(plan, axis_geometry)

        assert relation is not None
        consumer = relation.source_domain
        producer = relation.target_domain
        for consumer_task in range(consumer.size):
            coordinates = consumer.coordinates(consumer_task)
            expected = frozenset(
                producer.index(
                    {
                        10: coordinates[20],
                        11: 2 * coordinates[21] + offset,
                    }
                )
                for offset in range(2)
            )
            self.assertEqual(relation.targets(consumer_task), expected)

    def test_flattened_qwen_merge_matches_multidimensional_dependency(self) -> None:
        domains = (
            _domain((15, 128, 1), (16, 16, 1), (17, 1, 4)),
            _domain((20, 16, 1), (21, 64, 1)),
        )
        lse_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(128, 16, 4),
                    strides=(64, 4, 1),
                    block_ids=(15, 16, 17),
                    scales=(1, 1, 1),
                    offsets=(0, 0, 0),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(8192,),
                    strides=(1,),
                    block_ids=(None,),
                    offsets=(None,),
                    affine_subscript_ranges=((((20, 512, 1), (21, 1, 1)), 0, 512, 64),),
                ),
            ),
            [[15, 16, 17], [20, 21]],
        )
        output_ranges = tuple(
            (((20, 65536, 1), (21, 128, 1)), offset, offset + 128, 1)
            for offset in range(0, 65536, 8192)
        )
        output_plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(128, 16, 4, 128),
                    strides=(8192, 512, 128, 1),
                    block_ids=(15, 16, 17, None),
                    scales=(1, 1, 1, 1),
                    offsets=(0, 0, 0, None),
                    full_slice=(False, False, False, True),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(1048576,),
                    strides=(1,),
                    block_ids=(None,),
                    offsets=(None,),
                    affine_subscript_ranges=output_ranges,
                ),
            ),
            [[15, 16, 17], [20, 21]],
        )

        lse = _root_producers_by_consumer(lse_plan, domains)
        output = _root_producers_by_consumer(output_plan, domains)

        self.assertEqual(lse, output)
        assert lse is not None
        for consumer_task, producers in enumerate(lse):
            coordinates = domains[1].coordinates(consumer_task)
            chunk = coordinates[20]
            head = coordinates[21]
            expected = frozenset(
                split_index + 128 * (head // 4)
                for split_index in range(8 * chunk, 8 * chunk + 8)
            )
            self.assertEqual(producers, expected)

        relation = _symbolic_root_relation(
            lse_plan,
            _axis_geometry(domains),
        )
        assert relation is not None
        quotient = relation.producer_set_quotient()
        assert quotient is not None
        keys_by_consumer, producers_by_key = quotient
        self.assertEqual(keys_by_consumer.target_domain.shape, (16, 16))
        producer_count = producers_by_key.target_count_by_source()
        assert producer_count is not None
        self.assertEqual(producer_count.constant_value(), 8)

    def test_strided_flattened_hull_is_not_treated_as_contiguous(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(16,),
                    strides=(1,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(16,),
                    strides=(1,),
                    block_ids=(None,),
                    offsets=(None,),
                    affine_subscript_ranges=((((20, 0, 1),), 0, 16, 2),),
                ),
            ),
            [[10], [20]],
        )
        domains = (
            _domain((10, 16, 1)),
            _domain((20, 1, 1)),
        )

        self.assertIsNone(_root_producers_by_consumer(plan, domains))

    def test_flattened_affine_range_rejects_negative_first_lane(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(16,),
                    strides=(1,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(16,),
                    strides=(1,),
                    block_ids=(None,),
                    offsets=(None,),
                    affine_subscript_ranges=((((20, 1, 1),), -1, 2, 1),),
                ),
            ),
            [[10], [20]],
        )
        domains = (
            _domain((10, 16, 1)),
            _domain((20, 1, 1)),
        )

        self.assertIsNone(_root_producers_by_consumer(plan, domains))

    @skipIfNotCUDA()
    @skipIfRefEager("compiled DeviceIR is unavailable in ref eager mode")
    def test_device_analysis_retains_flattened_affine_iota_pattern(self) -> None:
        x = torch.randn((2, 4), device=DEVICE)
        bound = flattened_affine_stage.bind((x,))
        assert bound.host_function is not None
        device_ir = bound.host_function.device_ir
        with bound.env, bound.host_function:
            accesses = DeviceIRAnalysis.build(device_ir, bound.env).tile_accesses(
                device_ir,
                bound.env,
                bound.host_function,
            )

        flattened_loads = tuple(
            access
            for access in accesses
            if access.kind == "load"
            and access.tensor_shape == (64,)
            and access.affine_subscript_ranges is not None
        )
        self.assertEqual(len(flattened_loads), 1)
        (flattened_load,) = flattened_loads
        assert flattened_load.affine_subscript_ranges is not None
        self.assertEqual(
            tuple(
                (begin, end, step)
                for _coefficients, begin, end, step in (
                    flattened_load.affine_subscript_ranges
                )
            ),
            ((0, 32, 8),),
        )
        self.assertEqual(
            tuple(
                coefficient
                for coefficients, _begin, _end, _step in (
                    flattened_load.affine_subscript_ranges
                )
                for _axis, coefficient, _divisor in coefficients
            ),
            (32, 1),
        )

    @skipIfNotCUDA()
    @skipIfRefEager("compiled DeviceIR is unavailable in ref eager mode")
    def test_device_analysis_retains_qwen_value_and_final_patterns(self) -> None:
        x = torch.randn((2, 4), device=DEVICE)
        bound = flattened_qwen_attention_stages.bind((x,))
        assert bound.host_function is not None
        device_ir = bound.host_function.device_ir
        with bound.env, bound.host_function:
            accesses = DeviceIRAnalysis.build(device_ir, bound.env).tile_accesses(
                device_ir,
                bound.env,
                bound.host_function,
            )

        flattened_loads = tuple(
            access
            for access in accesses
            if access.kind == "load" and access.affine_subscript_ranges is not None
        )
        self.assertEqual(
            tuple(access.tensor_shape for access in flattened_loads),
            ((131072,), (16384,)),
        )
        self.assertEqual(
            tuple(
                len(access.affine_subscript_ranges or ()) for access in flattened_loads
            ),
            (8, 16),
        )
        self.assertEqual(
            tuple(
                (end - begin) // step
                for access in flattened_loads
                for _coefficients, begin, end, step in (
                    access.affine_subscript_ranges or ()
                )
            ),
            (128,) * 24,
        )

    @skipIfNotCUDA()
    @skipIfRefEager("compiled DeviceIR is unavailable in ref eager mode")
    def test_shared_device_graph_preserves_every_root_owner(self) -> None:
        x = torch.empty((2, 64), device=DEVICE, dtype=torch.float32)
        bound = cartesian_affine_stage.bind((x,))
        assert bound.host_function is not None
        device_ir = bound.host_function.device_ir
        shared_graph_id = device_ir.root_ids[0]
        shared_family = device_ir.task_families[0]
        shared_grid_block_ids = device_ir.grid_block_ids[0]
        original_root_ids = device_ir.root_ids
        original_task_families = device_ir.task_families
        original_grid_block_ids = device_ir.grid_block_ids
        try:
            device_ir.root_ids = [shared_graph_id, shared_graph_id]
            device_ir.task_families = [shared_family, shared_family]
            device_ir.grid_block_ids = [shared_grid_block_ids, shared_grid_block_ids]
            owners = owner_roots_by_graph_id(device_ir)
            self.assertEqual(owners[shared_graph_id], (0, 1))
            with bound.env, bound.host_function:
                analysis = DeviceIRAnalysis.build(device_ir, bound.env)
                accesses = analysis.tile_accesses(
                    device_ir,
                    bound.env,
                    bound.host_function,
                )
            self.assertEqual(
                sorted((access.root, access.kind) for access in accesses),
                [(0, "load"), (0, "store"), (1, "load"), (1, "store")],
            )
            dependency_graph = build_tile_dependency_graph(
                accesses,
                device_ir=device_ir,
            )
            self.assertTrue(dependency_graph.edges_between(0, 1))
            self.assertTrue(
                all(
                    all(site.root == access.root for site in sites)
                    for access in dependency_graph.accesses
                    for sites in (dependency_graph.sites_for_access(access.access_id),)
                )
            )
        finally:
            device_ir.root_ids = original_root_ids
            device_ir.task_families = original_task_families
            device_ir.grid_block_ids = original_grid_block_ids

    def test_noninjective_regions_are_not_coordinate_disjoint(self) -> None:
        for layout, left_interval, right_interval, second_dimension in (
            (((2, 1), (0, 1), 0), (0, 1), (0, 1), (0, 1)),
            (((2, 2), (1, 1), 0), (0, 2), (1, 3), (0, 2)),
        ):
            with self.subTest(layout=layout):
                left = AllocationRegion(
                    left_interval,
                    False,
                    layout,
                    ((0, 1), second_dimension),
                    True,
                )
                right = AllocationRegion(
                    right_interval,
                    False,
                    layout,
                    ((1, 2), second_dimension),
                    True,
                )

                self.assertTrue(allocation_regions_may_overlap(left, right))

    def test_multidimensional_storage_offset_falls_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(4, 4),
                    strides=(4, 1),
                    block_ids=(10, 11),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(3, 3),
                    strides=(4, 1),
                    block_ids=(20, 21),
                    storage_offset=5,
                ),
            ),
            [[10, 11], [20, 21]],
        )

        root_domains = (
            _domain((10, 4, 1), (11, 4, 1)),
            _domain((20, 3, 1), (21, 3, 1)),
        )
        self.assertIsNone(_root_producers_by_consumer(plan, root_domains))

    def test_one_dimensional_storage_offset_remains_task_ready(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(128,),
                    strides=(1,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(64,),
                    strides=(1,),
                    block_ids=(20,),
                    storage_offset=32,
                ),
            ),
            [[10], [20]],
        )

        self.assertEqual(
            _root_producers_by_consumer(
                plan,
                _one_dimensional_domains(
                    producer_count=8,
                    consumer_count=4,
                ),
            ),
            tuple(frozenset((task + 2,)) for task in range(4)),
        )

    def test_source_phase_boundary_satisfies_allocation_dependency(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            task_families=(
                TaskFamily((TaskAxis(10, None),)),
                TaskFamily((TaskAxis(20, None),)),
            ),
            root_phases=(0, 1),
        )

        self.assertEqual(plan.edges, ())

    def test_edge_retains_every_alias_of_the_allocation(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    tensor_name="base",
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    tensor_name="producer_view",
                    block_ids=(1,),
                ),
                _access(
                    2,
                    root=1,
                    kind="store",
                    tensor_name="producer_view",
                    block_ids=(1,),
                ),
                _access(
                    3,
                    root=2,
                    kind="load",
                    tensor_name="consumer_view",
                    block_ids=(2,),
                ),
            ),
            [[0], [1], [2]],
        )

        edge = plan.edges_between(1, 2)[0]
        self.assertEqual(edge.allocation_id, 0)
        self.assertEqual(
            edge.tensor_names,
            frozenset(("base", "producer_view", "consumer_view")),
        )

    def test_identity_mapping_is_task_ready(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        self.assertEqual(len(plan.edges), 1)
        edge = plan.edges[0]
        self.assertEqual(
            _dependency_kinds(edge),
            frozenset((TileDependencyKind.READ_AFTER_WRITE,)),
        )
        self.assertEqual(
            _root_producers_by_consumer(plan, _one_dimensional_domains()),
            tuple(frozenset((task,)) for task in range(8)),
        )

    def test_aligned_in_place_update_is_task_ready(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
                _access(2, root=1, kind="store", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        edge = plan.edges[0]
        self.assertEqual(
            _dependency_kinds(edge),
            frozenset(
                (
                    TileDependencyKind.READ_AFTER_WRITE,
                    TileDependencyKind.WRITE_AFTER_WRITE,
                )
            ),
        )
        self.assertEqual(
            _root_producers_by_consumer(plan, _one_dimensional_domains()),
            tuple(frozenset((task,)) for task in range(8)),
        )

    def test_aligned_write_after_read_is_task_ready(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="load", block_ids=(10,)),
                _access(1, root=1, kind="store", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        edge = plan.edges[0]
        self.assertEqual(
            _dependency_kinds(edge),
            frozenset((TileDependencyKind.WRITE_AFTER_READ,)),
        )
        self.assertEqual(
            _root_producers_by_consumer(plan, _one_dimensional_domains()),
            tuple(frozenset((task,)) for task in range(8)),
        )

    def test_unproven_reversed_access_falls_back_to_root(self) -> None:
        for kind in ("load", "store"):
            with self.subTest(kind=kind):
                plan = build_tile_dependency_graph(
                    (
                        _access(0, root=0, kind="store", block_ids=(10,)),
                        _access(
                            1,
                            root=1,
                            kind=kind,
                            block_ids=(20,),
                            scales=(-1,),
                        ),
                    ),
                    [[10], [20]],
                )

                self.assertIsNone(
                    _root_producers_by_consumer(plan, _one_dimensional_domains())
                )

    def test_batch_axis_is_part_of_task_mapping(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(2, 128),
                    strides=(128, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(2, 128),
                    strides=(128, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
            ),
            [[10, 11], [20, 21]],
        )

        root_domains = (
            _domain((10, 2, 1), (11, 4, 1)),
            _domain((20, 2, 1), (21, 4, 1)),
        )
        relation = _root_producers_by_consumer(plan, root_domains)
        assert relation is not None
        consumer_task = 1 + 2 * 2
        (producer_task,) = relation[consumer_task]
        self.assertEqual(
            root_domains[0].coordinates(producer_task),
            {10: 1, 11: 2},
        )

    def test_size_one_view_dimensions_are_normalized(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(32, 128),
                    strides=(128, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(1, 32, 128),
                    strides=(4096, 128, 1),
                    block_ids=(None, 20, 21),
                    scales=(1, 1, 1),
                    offsets=(0, 0, 0),
                    scalar=(True, False, False),
                    full_slice=(True, False, False),
                ),
            ),
            [[10, 11], [20, 21]],
        )

        root_domains = (
            _domain((10, 32, 1), (11, 8, 16)),
            _domain((20, 32, 1), (21, 8, 16)),
        )
        self.assertEqual(
            _root_producers_by_consumer(plan, root_domains),
            tuple(frozenset((task,)) for task in range(256)),
        )

    def test_dense_nontrivial_reshape_maps_exactly(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(32, 128),
                    strides=(128, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(4096,),
                    strides=(1,),
                    block_ids=(20,),
                    scales=(1,),
                    offsets=(0,),
                ),
            ),
            [[10, 11], [20]],
        )

        root_domains = (
            _domain((10, 32, 1), (11, 8, 16)),
            _domain((20, 256, 16)),
        )
        relation = _root_producers_by_consumer(plan, root_domains)
        producer_domain, consumer_domain = root_domains
        self.assertEqual(
            relation,
            tuple(
                frozenset(
                    (
                        producer_domain.index(
                            {
                                10: (consumer_task * 16) // 128,
                                11: ((consumer_task * 16) % 128) // 16,
                            }
                        ),
                    )
                )
                for consumer_task in range(consumer_domain.size)
            ),
        )

    def test_dense_reshape_ignores_common_symbolic_storage_origin(self) -> None:
        storage_origin = sympy.Symbol(
            "input_storage_offset",
            integer=True,
            nonnegative=True,
        )
        common_offset = storage_origin + 7

        def make_plan(offset: int | sympy.Expr) -> TileDependencyGraph:
            return build_tile_dependency_graph(
                (
                    _access(
                        0,
                        root=0,
                        kind="store",
                        shape=(1, 32, 128),
                        strides=(4096, 128, 1),
                        block_ids=(10, 11, 12),
                        scales=(1, 1, 1),
                        offsets=(0, 0, 0),
                        storage_offset=offset,
                    ),
                    _access(
                        1,
                        root=1,
                        kind="load",
                        shape=(1, 4096),
                        strides=(4096, 1),
                        block_ids=(20, 21),
                        scales=(1, 1),
                        offsets=(0, 0),
                        storage_offset=offset,
                    ),
                ),
                [[10, 11, 12], [20, 21]],
            )

        root_domains = (
            _domain((10, 1, 1), (11, 32, 1), (12, 1, 128)),
            _domain((20, 1, 1), (21, 32, 128)),
        )
        zero_offset = make_plan(0)
        symbolic_offset = make_plan(common_offset)

        self.assertEqual(
            tuple(access.storage_offset for access in symbolic_offset.accesses),
            (common_offset, common_offset),
        )
        configured_roots, site_domains = _configured_domains(
            symbolic_offset,
            _axis_geometry(root_domains),
        )
        (symbolic_relation,) = tuple(
            dependency.producers_by_consumer
            for dependency in instantiate_symbolic_dependencies(
                symbolic_offset,
                root_domains=configured_roots,
                site_domains=site_domains,
            )
        )
        self.assertIsNotNone(symbolic_relation)
        assert symbolic_relation is not None
        self.assertNotIn(storage_origin, symbolic_relation.parameter_symbols)
        self.assertEqual(
            _root_producers_by_consumer(symbolic_offset, root_domains),
            _root_producers_by_consumer(zero_offset, root_domains),
        )

    def test_dense_reshape_does_not_cancel_unequal_symbolic_offsets(self) -> None:
        storage_origin = sympy.Symbol(
            "input_storage_offset",
            integer=True,
            nonnegative=True,
        )
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(1, 32, 128),
                    strides=(4096, 128, 1),
                    block_ids=(10, 11, 12),
                    scales=(1, 1, 1),
                    offsets=(0, 0, 0),
                    storage_offset=storage_origin,
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(1, 4096),
                    strides=(4096, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                    storage_offset=storage_origin + 1,
                ),
            ),
            [[10, 11, 12], [20, 21]],
        )
        root_domains = (
            _domain((10, 1, 1), (11, 32, 1), (12, 1, 128)),
            _domain((20, 1, 1), (21, 32, 128)),
        )

        self.assertIsNone(_root_producers_by_consumer(plan, root_domains))

    def test_unequal_tiles_map_to_every_overlapping_producer(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        self.assertEqual(
            _root_producers_by_consumer(
                plan,
                _one_dimensional_domains(
                    producer_count=8,
                    consumer_count=2,
                    producer_block=16,
                    consumer_block=64,
                ),
            ),
            (frozenset((0, 1, 2, 3)), frozenset((4, 5, 6, 7))),
        )

    def test_root_relation_uses_coordinates_not_flattened_pid_runs(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(2, 256),
                    strides=(256, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(2, 256),
                    strides=(256, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    2,
                    root=1,
                    kind="load",
                    shape=(2, 256),
                    strides=(256, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 128),
                ),
            ),
            [[10, 11], [20, 21]],
        )
        root_domains = (
            _domain((10, 2, 1), (11, 16, 16)),
            _domain((20, 2, 1), (21, 4, 32)),
        )
        relation = _root_producers_by_consumer(plan, root_domains)
        assert relation is not None
        self.assertEqual(len(relation), 8)
        self.assertEqual({len(producer_tasks) for producer_tasks in relation}, {4})
        self.assertEqual(frozenset().union(*relation), frozenset(range(32)))
        self.assertTrue(
            all(
                left.isdisjoint(right)
                for left, right in itertools.combinations(relation, 2)
            )
        )

    def test_allocation_overlap_relation_is_authoritative(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(2, 256),
                    strides=(256, 1),
                    block_ids=(10, 11),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(2, 256),
                    strides=(256, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 0),
                ),
                _access(
                    2,
                    root=1,
                    kind="load",
                    shape=(2, 256),
                    strides=(256, 1),
                    block_ids=(20, 21),
                    scales=(1, 1),
                    offsets=(0, 128),
                ),
            ),
            [[10, 11], [20, 21]],
        )
        root_domains = (
            _domain((10, 2, 1), (11, 16, 16)),
            _domain((20, 2, 1), (21, 4, 32)),
        )
        actual = _root_producers_by_consumer(plan, root_domains)
        assert actual is not None
        for consumer_task, producer_tasks in enumerate(actual):
            coordinates = root_domains[1].coordinates(consumer_task)
            batch = coordinates[20]
            group = coordinates[21]
            self.assertEqual(
                producer_tasks,
                frozenset(
                    batch + producer_group * 2
                    for producer_group in (
                        2 * group,
                        2 * group + 1,
                        8 + 2 * group,
                        9 + 2 * group,
                    )
                ),
            )

    def test_root_relation_accepts_non_power_of_two_fanin(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", shape=(96,), block_ids=(10,)),
                _access(1, root=1, kind="load", shape=(96,), block_ids=(20,)),
            ),
            [[10], [20]],
        )
        self.assertEqual(
            _root_producers_by_consumer(
                plan,
                _one_dimensional_domains(
                    producer_count=6,
                    consumer_count=2,
                    producer_block=16,
                    consumer_block=48,
                ),
            ),
            (frozenset((0, 1, 2)), frozenset((3, 4, 5))),
        )

    def test_root_relation_accepts_overlapping_and_partial_domains(
        self,
    ) -> None:
        overlapping = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
                _access(
                    2,
                    root=1,
                    kind="load",
                    block_ids=(20,),
                    offsets=(16,),
                ),
            ),
            [[10], [20]],
        )
        self.assertEqual(
            _root_producers_by_consumer(
                overlapping,
                _one_dimensional_domains(
                    producer_count=8,
                    consumer_count=4,
                    producer_block=16,
                    consumer_block=32,
                ),
            ),
            (
                frozenset((0, 1, 2)),
                frozenset((2, 3, 4)),
                frozenset((4, 5, 6)),
                frozenset((6, 7)),
            ),
        )

        identity = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        prefix = _root_producers_by_consumer(
            identity,
            _one_dimensional_domains(
                producer_count=8,
                consumer_count=3,
                producer_block=16,
                consumer_block=32,
            ),
        )
        assert prefix is not None
        self.assertEqual(
            prefix, (frozenset((0, 1)), frozenset((2, 3)), frozenset((4, 5)))
        )
        self.assertEqual(frozenset().union(*prefix), frozenset(range(6)))

        suffix = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(
                    1,
                    root=1,
                    kind="load",
                    block_ids=(20,),
                    offsets=(32,),
                ),
            ),
            [[10], [20]],
        )
        suffix_relation = _root_producers_by_consumer(
            suffix,
            _one_dimensional_domains(
                producer_count=8,
                consumer_count=3,
                producer_block=16,
                consumer_block=32,
            ),
        )
        self.assertEqual(
            suffix_relation,
            (frozenset((2, 3)), frozenset((4, 5)), frozenset((6, 7))),
        )

    def test_tile_id_indices_use_scalar_extent(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    block_ids=(10,),
                    scalar=(True,),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    block_ids=(20,),
                    scalar=(True,),
                ),
            ),
            [[10], [20]],
        )
        self.assertEqual(
            _root_producers_by_consumer(
                plan,
                _one_dimensional_domains(
                    producer_count=4,
                    consumer_count=4,
                    producer_block=128,
                    consumer_block=128,
                ),
            ),
            tuple(frozenset((task,)) for task in range(4)),
        )

    def test_distinct_fixed_scalar_regions_do_not_alias(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(8,),
                    block_ids=(None,),
                    offsets=(3,),
                    scalar=(True,),
                    static_extents=(1,),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(8,),
                    block_ids=(None,),
                    offsets=(4,),
                    scalar=(True,),
                    static_extents=(1,),
                ),
            ),
            [[10], [20]],
        )

        self.assertEqual(plan.edges, ())

    def test_dynamic_scalar_offset_remains_conservative(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(8,),
                    block_ids=(None,),
                    offsets=(None,),
                    scalar=(True,),
                    static_extents=(1,),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(8,),
                    block_ids=(None,),
                    offsets=(4,),
                    scalar=(True,),
                    static_extents=(1,),
                ),
            ),
            [[10], [20]],
        )

        self.assertEqual(
            tuple((edge.producer_root, edge.consumer_root) for edge in plan.edges),
            ((0, 1),),
        )

    def test_masked_fixed_scalar_store_does_not_kill_previous_writer(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(8,),
                    block_ids=(None,),
                    offsets=(3,),
                    scalar=(True,),
                    static_extents=(1,),
                ),
                _access(
                    1,
                    root=1,
                    kind="store",
                    shape=(8,),
                    block_ids=(None,),
                    offsets=(3,),
                    scalar=(True,),
                    static_extents=(1,),
                    masked=True,
                ),
                _access(
                    2,
                    root=2,
                    kind="load",
                    shape=(8,),
                    block_ids=(None,),
                    offsets=(3,),
                    scalar=(True,),
                    static_extents=(1,),
                ),
            ),
            [[10], [20], [30]],
        )

        self.assertEqual(
            tuple((edge.producer_root, edge.consumer_root) for edge in plan.edges),
            ((0, 1), (0, 2), (1, 2)),
        )

    def test_multiple_stores_fall_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=0, kind="store", block_ids=(10,)),
                _access(2, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        self.assertEqual(
            _root_producers_by_consumer(plan, _one_dimensional_domains()),
            tuple(frozenset((task,)) for task in range(8)),
        )

    def test_masked_store_falls_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,), masked=True),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )

        self.assertIsNone(_root_producers_by_consumer(plan, _one_dimensional_domains()))

    def test_nonzero_or_dynamic_grid_start_falls_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
            noncanonical_task_origin_block_ids=frozenset((10,)),
        )

        self.assertIsNone(_root_producers_by_consumer(plan, _one_dimensional_domains()))

    def test_tracks_latest_writer_and_intervening_readers(self) -> None:
        task_families = tuple(
            TaskFamily(
                axes=(TaskAxis(root, 128),),
            )
            for root in range(4)
        )
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store"),
                _access(1, root=1, kind="load", block_ids=(1,)),
                _access(2, root=2, kind="store", block_ids=(2,)),
                _access(3, root=3, kind="load", block_ids=(3,)),
            ),
            task_families=task_families,
        )

        self.assertEqual(
            [
                (edge.producer_root, edge.consumer_root, _dependency_kinds(edge))
                for edge in plan.edges
            ],
            [
                (0, 1, frozenset((TileDependencyKind.READ_AFTER_WRITE,))),
                (0, 2, frozenset((TileDependencyKind.WRITE_AFTER_WRITE,))),
                (1, 2, frozenset((TileDependencyKind.WRITE_AFTER_READ,))),
                (2, 3, frozenset((TileDependencyKind.READ_AFTER_WRITE,))),
            ],
        )

    def test_partial_write_retains_uncovered_reaching_definition(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    shape=(96,),
                    block_ids=(10,),
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    shape=(96,),
                    block_ids=(20,),
                ),
                _access(
                    2,
                    root=1,
                    kind="store",
                    shape=(96,),
                    block_ids=(20,),
                ),
                _access(
                    3,
                    root=2,
                    kind="load",
                    shape=(96,),
                    block_ids=(30,),
                ),
            ),
            task_families=(
                TaskFamily((TaskAxis(10, 96),)),
                TaskFamily((TaskAxis(20, 64),)),
                TaskFamily((TaskAxis(30, 96),)),
            ),
        )

        self.assertEqual(
            [
                (
                    edge.producer_root,
                    edge.consumer_root,
                    _dependency_kinds(edge),
                    tuple(
                        dependency.region.address_interval
                        for dependency in edge.access_dependencies
                    ),
                )
                for edge in plan.edges
            ],
            [
                (
                    0,
                    1,
                    frozenset(
                        (
                            TileDependencyKind.READ_AFTER_WRITE,
                            TileDependencyKind.WRITE_AFTER_WRITE,
                        )
                    ),
                    ((0, 64), (0, 64)),
                ),
                (
                    0,
                    2,
                    frozenset((TileDependencyKind.READ_AFTER_WRITE,)),
                    ((64, 96),),
                ),
                (
                    1,
                    2,
                    frozenset((TileDependencyKind.READ_AFTER_WRITE,)),
                    ((0, 64),),
                ),
            ],
        )

    def test_alias_names_share_an_allocation_dependency(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(
                    0,
                    root=0,
                    kind="store",
                    tensor_name="base",
                ),
                _access(
                    1,
                    root=1,
                    kind="load",
                    tensor_name="view",
                    block_ids=(1,),
                ),
            ),
            [[0], [1]],
        )

        self.assertEqual(len(plan.edges), 1)
        self.assertEqual(plan.edges[0].tensor_names, frozenset(("base", "view")))


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
        schedule_domain = _domain((-3, 2), (-2, 4), (-1, 2), kind="worker")
        widened_domain = _domain((-3, 2), (-2, 4), (-1, 3), kind="worker")
        target_domain = _domain((10, 3), identity=target_identity)
        worker = coordinate_axis_symbol(-2)
        task = coordinate_axis_symbol(10)
        task_order = _point_map(
            schedule_domain,
            target_domain,
            (((-3, 1, 2, 1), (-2, 0, 3, 1), (-1, 0, 1, 1)), (worker,)),
        )
        exact_converse = _full_point_map(
            target_domain,
            schedule_domain,
            sympy.Integer(1),
            task,
            sympy.Integer(0),
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
            assert rebased is not None
            self.assertEqual(rebased.pieces, task_order.pieces)
            self.assertTrue(rebased.is_bijection_from_source_support())
            converse = rebased.converse()
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
        ordinalization = task_order._ordinalized_source_support
        assert ordinalization is not None
        ordinal_inverse = _memoized_exact_converse(ordinalization)
        assert ordinal_inverse is not None
        self.assertTrue(ordinal_inverse.is_total_function())

    def test_relation_transform_exact_converse_provenance(self) -> None:
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)
        source = _domain((10, 2), (11, extent), kind="task_order", allow_empty=True)
        target = _domain((20, extent), (21, 2), identity=0, allow_empty=True)
        relation = _full_point_map(
            source,
            target,
            coordinate_axis_symbol(11),
            coordinate_axis_symbol(10),
        )
        self.assertIsNotNone(relation.converse())

        renamed_target = _domain((30, extent), (31, 2), identity=1, allow_empty=True)
        renamed = relation.rename_target_axes(renamed_target)
        assert renamed is not None
        self.assertIsNotNone(_memoized_exact_converse(renamed))

        def assert_cached_converse_is_exact(
            transformed: CoordinateRelation,
            concrete_extent: int,
        ) -> None:
            concrete = transformed.substitute_parameters({extent: concrete_extent})
            converse = _memoized_exact_converse(concrete)
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
            _domain((20, extent), identity=0, allow_empty=True)
        )
        projected_source = relation.project_source(
            _domain((11, extent), kind="task_order", allow_empty=True)
        )
        narrow_source = _domain((11, extent), kind="task_order", allow_empty=True)
        narrow_target = _domain((20, extent), identity=0, allow_empty=True)
        narrow = _full_point_map(
            narrow_source,
            narrow_target,
            coordinate_axis_symbol(11),
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
        source = _domain((10, 4), kind="task_order")
        target = _domain((20, 4), identity=0)
        source_coordinate = coordinate_axis_symbol(10)
        source_partitioned = _point_map(
            source,
            target,
            (((10, 0, 2, 1),), (source_coordinate,)),
            (((10, 2, 4, 1),), (source_coordinate,)),
        )
        self.assertIsNotNone(source_partitioned.converse())
        source_coalesced = source_partitioned.coalesce_adjacent_source_boxes()
        self.assertEqual(len(source_coalesced.pieces), 1)

        singleton_source = _domain((30, 1), kind="task_order")
        target_partitioned = _relation(
            singleton_source,
            target,
            _piece(((30, 0, 1, 1),), (20, sympy.Integer(0), sympy.Integer(2), 1)),
            _piece(((30, 0, 1, 1),), (20, sympy.Integer(2), sympy.Integer(4), 1)),
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
        source = _domain((10, 2), kind="worker")
        middle = _domain((20, 1), kind="task_order")
        target = _domain((30, 1), kind="site")
        source_coordinate = coordinate_axis_symbol(10)
        middle_coordinate = coordinate_axis_symbol(20)
        first = _full_point_map(
            source,
            middle,
            source_coordinate,
        )
        first_inverse = _full_point_map(
            middle,
            source,
            middle_coordinate,
        )
        _remember_exact_converse(first, first_inverse)
        following = _full_point_map(
            middle,
            target,
            sympy.Integer(0),
        )
        following_inverse = _full_point_map(
            target,
            middle,
            sympy.Integer(0),
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
        source = _domain((10, 4), kind="worker")
        ordinal = _domain((20, 4), kind="task_order")
        target = _domain((30, 4), identity=0)
        source_coordinate = coordinate_axis_symbol(10)
        ordinal_coordinate = coordinate_axis_symbol(20)
        first = _full_point_map(
            source,
            ordinal,
            source_coordinate,
        )
        following = _full_point_map(
            ordinal,
            target,
            3 - ordinal_coordinate,
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
        assert composed_converse is not None
        self.assertEqual(
            composed_converse.materialize(),
            (frozenset((3,)), frozenset((2,)), frozenset((1,)), frozenset((0,))),
        )

        left = _point_map(source, target, (((10, 0, 2, 1),), (source_coordinate,)))
        right = _point_map(source, target, (((10, 2, 4, 1),), (source_coordinate,)))
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
        source = _domain((10, 6), kind="task_order")
        target = _domain((20, 6), identity=0)
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
        local_order = _domain((10, 3), kind="task_order")
        target = _domain((20, 3), kind="site", identity=0)
        ordinal = coordinate_axis_symbol(10)
        valid = _full_point_map(
            local_order,
            target,
            ordinal,
        )
        self.assertTrue(valid.is_bijection_from_source_support())
        self.assertIsNotNone(valid.converse())
        WorkerScheduleSegment(root=0, task_order=valid)

        invalid_relations = {
            "padded target tail": _full_point_map(
                local_order,
                target,
                ordinal + 1,
            ),
            "out-of-domain source": _point_map(
                local_order, target, (((10, -1, 2, 1),), (ordinal + 1,))
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


_FIRST_AXIS = 10
_SECOND_AXIS = 20
_BATCH_AXIS = 30


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
    assert cardinality is not None
    test.assertEqual(sympy.simplify(cardinality - expected_cardinality), 0)
    test.assertTrue(relation.is_single_valued())
    test.assertTrue(relation.is_bijection_from_source_support())
    converse = relation.converse()
    assert converse is not None
    test.assertTrue(converse.is_total_function())
    return converse


class TestRaggedL2Schedule(TestCase):
    def test_nested_quotient_remainder_identity_is_simplified_first(self) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        first = sympy.Symbol("first", integer=True, nonnegative=True)
        second = sympy.Symbol("second", integer=True, nonnegative=True)
        dividend = 3 * batch + 2 * second + 6 * FloorDiv(first, 2) + sympy.Mod(first, 2)
        expression = 7 * FloorDiv(dividend, 7) + sympy.Mod(dividend, 7)

        self.assertEqual(
            _simplify_integer_quotients(expression),
            _simplify_integer_quotients(dividend),
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
                    (_FIRST_AXIS, first_count, 1),
                    (_SECOND_AXIS, second_count, 1),
                    identity=0,
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
                            (_FIRST_AXIS, first_count, 1),
                            (_SECOND_AXIS, second_count, 1),
                            identity=0,
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
            (_FIRST_AXIS, 4097, 1),
            (_SECOND_AXIS, 3, 1),
            identity=0,
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

    def test_symbolic_outer_l2_preserves_permutation_and_concrete_inverse(
        self,
    ) -> None:
        batch = sympy.Symbol("batch", integer=True, nonnegative=True)
        pid_axis_order = (_FIRST_AXIS, _SECOND_AXIS, _BATCH_AXIS)
        for first_count in (4, 5):
            domain = _domain(
                (_BATCH_AXIS, batch, 1),
                (_FIRST_AXIS, first_count, 1),
                (_SECOND_AXIS, 3, 1),
                identity=0,
            )
            task_order = pid_task_order(
                domain,
                pid_axis_order,
                l2_group_size=2,
            )

            self.assertEqual(len(task_order.pieces), 1)
            if first_count == 4:
                with mock.patch.object(
                    CoordinateRelation,
                    "materialize",
                    side_effect=AssertionError("symbolic L2 proof must not enumerate"),
                ):
                    _assert_exact_bijection(self, task_order, 12 * batch)
            for concrete_batch in (0, 1, 2, 8):
                with self.subTest(
                    first_count=first_count,
                    batch=concrete_batch,
                ):
                    concrete_domain = domain.substitute_parameters(
                        {batch: concrete_batch}
                    )
                    concrete_order = task_order.substitute_parameters(
                        {batch: concrete_batch}
                    )
                    expected = _expected_l2_targets(
                        concrete_domain,
                        pid_axis_order,
                        2,
                    )
                    self.assertEqual(_singleton_targets(concrete_order), expected)
                    converse = _assert_exact_bijection(
                        self,
                        concrete_order,
                        first_count * 3 * concrete_batch,
                    )
                    self.assertEqual(
                        _singleton_targets(converse),
                        tuple(expected.index(task) for task in range(len(expected))),
                    )

    def test_bounded_binary_floordiv_selector_is_generic(self) -> None:
        source = _domain((1, 6), kind="task_order")
        target = _domain((2, 6), identity=0)
        coordinate = coordinate_axis_symbol(1)
        selector = FloorDiv(coordinate, 3)
        selected = coordinate + selector * (8 - 2 * coordinate)
        relation = _full_point_map(
            source,
            target,
            selected,
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
        source = _domain((1, 9), kind="task_order")
        target = _domain((2, 9), identity=0)
        coordinate = coordinate_axis_symbol(1)
        selector = FloorDiv(coordinate, 3)
        relation = _full_point_map(
            source,
            target,
            coordinate + 6 - 6 * selector,
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
