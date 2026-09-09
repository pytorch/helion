from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Literal
from unittest import mock

import sympy
import torch

import helion
from helion import exc
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.device_ir_analysis import DeviceIRAnalysis
from helion._compiler.tile_dependency import AllocationRegion
from helion._compiler.tile_dependency import CoordinateDomain
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import ExecutionSite
from helion._compiler.tile_dependency import TaskAxis
from helion._compiler.tile_dependency import TaskFamily
from helion._compiler.tile_dependency import TileAccess
from helion._compiler.tile_dependency import TileDependency
from helion._compiler.tile_dependency import TileDependencyKind
from helion._compiler.tile_dependency import _CoordinateRelationPiece
from helion._compiler.tile_dependency import _dense_mixed_radix_converse
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
    shape: tuple[int, ...] = (128,),
    strides: tuple[int, ...] = (1,),
    block_ids: tuple[int | None, ...] = (0,),
    scales: tuple[int, ...] = (1,),
    offsets: tuple[int | None, ...] = (0,),
    scalar: tuple[bool, ...] | None = None,
    full_slice: tuple[bool, ...] | None = None,
    static_extents: tuple[int | None, ...] | None = None,
    masked: bool = False,
    tensor_name: str = "tmp",
    storage_offset: int = 0,
    layout_is_static: bool = True,
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
        layout_is_static=layout_is_static,
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
        CoordinateDomain(
            (10,),
            ((10, producer_count),),
            ((10, producer_block),),
        ),
        CoordinateDomain(
            (20,),
            ((20, consumer_count),),
            ((20, consumer_block),),
        ),
    )


def _bounded_coordinate_relation(size: int) -> CoordinateRelation:
    source = CoordinateDomain(
        (10, 11),
        ((10, 2), (11, size)),
        kind="site",
    )
    target = CoordinateDomain((20,), ((20, 4),), kind="allocation")
    retained = coordinate_axis_symbol(10)
    bounded = coordinate_axis_symbol(11)
    value = retained + 2 * sympy.Mod(sympy.floor(bounded / 64), 2)
    return CoordinateRelation.point_map(
        source,
        target,
        (
            (
                ((10, 0, 2, 1), (11, 0, size, 1)),
                (value,),
            ),
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
        domain = CoordinateDomain(
            (10, 20),
            ((10, 2), (20, 3)),
            ((10, 4), (20, 8)),
        )
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
            CoordinateDomain((10,), ((10, 0),))
        unknown_sign = sympy.Symbol("unknown_sign", integer=True)
        with self.assertRaisesRegex(ValueError, "axis counts must be positive"):
            CoordinateDomain((10,), ((10, unknown_sign),))
        noninteger = sympy.Symbol("noninteger", nonnegative=True)
        with self.assertRaisesRegex(ValueError, "must be an integer expression"):
            CoordinateDomain((10,), ((10, noninteger),))

    def test_symbolic_coordinate_domain_substitution(self) -> None:
        width = 8
        extent = sympy.Symbol("extent", integer=True, nonnegative=True)
        task_count = sympy.floor((extent + width - 1) / width)
        domain = CoordinateDomain(
            (10, 20),
            ((10, task_count), (20, 2)),
            ((10, width), (20, 1)),
        )

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
        source = CoordinateDomain((10,), ((10, task_count),), kind="task_order")
        target = CoordinateDomain((20,), ((20, task_count),), kind="site")
        source_coordinate = coordinate_axis_symbol(10)
        relation = CoordinateRelation(
            source_domain=source,
            target_domain=target,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=((10, 0, task_count, 1),),
                    target_ranges=((20, source_coordinate, source_coordinate + 1, 1),),
                ),
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
            self.assertEqual(
                concrete.pieces[0].source_bounds_items,
                ((10, 0, expected_count, 1),),
            )
            self.assertEqual(
                concrete.materialize(),
                tuple(frozenset((index,)) for index in range(expected_count)),
            )

    def test_piecewise_dense_point_converse_is_exact_transpose(self) -> None:
        inner = coordinate_axis_symbol(10)
        outer = coordinate_axis_symbol(11)
        source = CoordinateDomain(
            (10, 11),
            ((10, 4), (11, 3)),
            kind="task_order",
        )
        target = CoordinateDomain((20,), ((20, 12),), kind="site")
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((10, 0, 2, 1), (11, 0, 3, 1)),
                    (2 * outer + inner,),
                ),
                (
                    ((10, 2, 4, 1), (11, 0, 3, 1)),
                    (2 * outer + inner + 4,),
                ),
            ),
        )

        converse = relation.converse()
        self.assertIsNotNone(converse)
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
        source = CoordinateDomain(
            (10, 11),
            ((10, 16), (11, 3)),
            kind="task_order",
        )
        target = CoordinateDomain((20,), ((20, 48),), kind="site")
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((10, 0, 8, 1), (11, 0, 3, 1)),
                    (8 * outer + inner,),
                ),
                (
                    ((10, 8, 16, 1), (11, 0, 3, 1)),
                    (8 * outer + inner + 16,),
                ),
            ),
        )

        self.assertEqual(len(relation.coalesce_adjacent_source_boxes().pieces), 2)
        compact = relation.coalesce_adjacent_source_boxes(fold_static_offsets=True)

        self.assertEqual(len(compact.pieces), 1)
        self.assertTrue(compact.is_total_function())
        self.assertTrue(compact.is_pointwise_equal_to(relation))
        self.assertEqual(compact.materialize(), relation.materialize())

        many_source = CoordinateDomain(
            (10, 11),
            ((10, 8), (11, 2)),
            kind="task_order",
        )
        many_target = CoordinateDomain((20,), ((20, 16),), kind="site")
        ordered_pieces = tuple(
            _CoordinateRelationPiece(
                ((10, 2 * part, 2 * part + 2, 1), (11, 0, 2, 1)),
                (
                    (
                        20,
                        2 * outer + inner + 2 * part,
                        2 * outer + inner + 2 * part + 1,
                        1,
                    ),
                ),
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

        unequal = CoordinateRelation.point_map(
            CoordinateDomain((10, 11), ((10, 15), (11, 3)), kind="task_order"),
            target,
            (
                (
                    ((10, 0, 8, 1), (11, 0, 3, 1)),
                    (8 * outer + inner,),
                ),
                (
                    ((10, 8, 15, 1), (11, 0, 3, 1)),
                    (8 * outer + inner + 16,),
                ),
            ),
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
        source = CoordinateDomain(
            (10, 11),
            ((10, 2), (11, 3)),
            kind="task_order",
        )
        target = CoordinateDomain((20,), ((20, 8),), kind="site")
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((10, 0, 2, 1), (11, 0, 3, 1)),
                    (3 * outer + inner,),
                ),
            ),
        )

        self.assertIsNone(relation.converse())

    def test_piecewise_mixed_radix_point_converse_is_exact_transpose(self) -> None:
        ordinal = coordinate_axis_symbol(10)
        source = CoordinateDomain((10,), ((10, 12),), kind="task_order")
        target = CoordinateDomain(
            (20, 21, 22),
            ((20, 2), (21, 3), (22, 2)),
            kind="site",
        )

        def relation(second_batch: int) -> CoordinateRelation:
            return CoordinateRelation.point_map(
                source,
                target,
                (
                    (
                        ((10, 0, 6, 1),),
                        (
                            sympy.Integer(0),
                            sympy.Mod(ordinal, 3),
                            sympy.floor(ordinal / 3),
                        ),
                    ),
                    (
                        ((10, 6, 12, 1),),
                        (
                            sympy.Integer(second_batch),
                            sympy.Mod(ordinal, 3),
                            sympy.floor(ordinal / 3) - 2,
                        ),
                    ),
                ),
            )

        bijection = relation(1)
        converse = bijection.converse()
        self.assertIsNotNone(converse)
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

    def test_woven_mixed_radix_converse_is_exact_and_symbolic(self) -> None:
        ordinal = coordinate_axis_symbol(10)
        source = CoordinateDomain((10,), ((10, 704),), kind="task_order")
        target = CoordinateDomain(
            (20, 21, 22),
            ((20, 2), (21, 8), (22, 44)),
            kind="site",
        )
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((10, 0, 704, 1),),
                    (
                        sympy.Mod(sympy.floor(ordinal / 32), 2),
                        sympy.Mod(ordinal, 8),
                        4 * sympy.floor(ordinal / 64)
                        + sympy.floor(sympy.Mod(ordinal, 32) / 8),
                    ),
                ),
            ),
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

    def test_woven_mixed_radix_converse_rejects_reused_input_digit(self) -> None:
        ordinal = coordinate_axis_symbol(10)
        source = CoordinateDomain((10,), ((10, 8),), kind="task_order")
        target = CoordinateDomain((20, 21), ((20, 2), (21, 4)), kind="site")
        noninjective = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((10, 0, 8, 1),),
                    (
                        sympy.Mod(ordinal, 2),
                        sympy.Mod(ordinal, 4),
                    ),
                ),
            ),
        )

        self.assertIsNone(noninjective.converse())

    def test_relation_axis_renaming_preserves_positional_coordinates(self) -> None:
        source = CoordinateDomain((10, 20), ((10, 2), (20, 3)))
        target = CoordinateDomain((30, 40), ((30, 2), (40, 3)))
        source_10 = coordinate_axis_symbol(10)
        source_20 = coordinate_axis_symbol(20)
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((10, 0, 2, 1), (20, 0, 3, 1)),
                    (
                        sympy.Mod(source_10 + source_20, 2),
                        sympy.floor((source_10 + 2 * source_20) / 2),
                    ),
                ),
            ),
        )
        renamed_source = CoordinateDomain(
            (20, 10),
            ((20, 2), (10, 3)),
            ((20, 4), (10, 8)),
            kind="task_order",
            identity=7,
        )
        renamed_target = CoordinateDomain(
            (40, 30),
            ((40, 2), (30, 3)),
            kind="event",
            identity=4,
        )

        renamed = relation.rename_source_axes(renamed_source)
        self.assertIsNotNone(renamed)
        assert renamed is not None
        renamed = renamed.rename_target_axes(renamed_target)
        self.assertIsNotNone(renamed)
        assert renamed is not None

        self.assertEqual(renamed.source_domain, renamed_source)
        self.assertEqual(renamed.target_domain, renamed_target)
        self.assertEqual(renamed.materialize(), relation.materialize())
        self.assertIsNone(
            relation.rename_source_axes(CoordinateDomain((0, 1), ((0, 2), (1, 4))))
        )
        self.assertIsNone(
            relation.rename_target_axes(CoordinateDomain((0, 1), ((0, 2), (1, 4))))
        )

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
        key_domain = CoordinateDomain(
            (0,),
            ((0, task_count),),
            kind="event",
            identity=0,
        )
        producer_domain = CoordinateDomain(
            (10,),
            ((10, task_count),),
            ((10, 16),),
            kind="site",
            identity=0,
        )
        relation = CoordinateRelation.point_map(
            key_domain,
            producer_domain,
            (
                (
                    ((0, 0, task_count, 1),),
                    (coordinate_axis_symbol(0),),
                ),
            ),
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
        key_domain = CoordinateDomain(
            (0,),
            ((0, key_count),),
            kind="event",
            identity=0,
        )
        producer_domain = CoordinateDomain(
            (10,),
            ((10, 2 * key_count),),
            ((10, 16),),
            kind="site",
            identity=0,
        )
        key = coordinate_axis_symbol(0)
        relation = CoordinateRelation(
            source_domain=key_domain,
            target_domain=producer_domain,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=((0, 0, key_count, 1),),
                    target_ranges=((10, 2 * key, 2 * key + 2, 1),),
                ),
            ),
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

    def test_symbolic_nonpartition_relations_do_not_derive_fixed_fan_in(
        self,
    ) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        key_domain = CoordinateDomain((0,), ((0, key_count),), kind="event")
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
                producer_domain = CoordinateDomain(
                    (10,),
                    ((10, target_count),),
                    kind="site",
                )
                relation = CoordinateRelation(
                    source_domain=key_domain,
                    target_domain=producer_domain,
                    pieces=(
                        _CoordinateRelationPiece(
                            source_bounds_items=((0, 0, key_count, 1),),
                            target_ranges=((10, begin, end, step),),
                        ),
                    ),
                )
                converse, target_counts = relation.derive_converse_and_target_counts()
                self.assertIsNone(converse)
                self.assertIsNone(target_counts)

        symbolic_fan_in = sympy.Symbol("fan_in", integer=True, positive=True)
        symbolic_producer_domain = CoordinateDomain(
            (10,),
            ((10, symbolic_fan_in * key_count),),
            kind="site",
        )
        symbolic_width = CoordinateRelation(
            source_domain=key_domain,
            target_domain=symbolic_producer_domain,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=((0, 0, key_count, 1),),
                    target_ranges=(
                        (
                            10,
                            symbolic_fan_in * key,
                            symbolic_fan_in * (key + 1),
                            1,
                        ),
                    ),
                ),
            ),
        )
        self.assertEqual(
            symbolic_width.derive_converse_and_target_counts(),
            (None, None),
        )

        producer_domain = CoordinateDomain(
            (10,),
            ((10, 2 * key_count),),
            kind="site",
        )
        duplicate_pieces = CoordinateRelation(
            source_domain=key_domain,
            target_domain=producer_domain,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=((0, 0, key_count, 1),),
                    target_ranges=((10, 2 * key, 2 * key + 2, 1),),
                ),
                _CoordinateRelationPiece(
                    source_bounds_items=((0, 0, key_count, 1),),
                    target_ranges=((10, 2 * key, 2 * key + 2, 1),),
                ),
            ),
        )
        self.assertEqual(
            duplicate_pieces.derive_converse_and_target_counts(),
            (None, None),
        )

    def test_unproved_symbolic_point_maps_decline_totality(self) -> None:
        task_count = sympy.Symbol("task_count", integer=True, nonnegative=True)
        source = CoordinateDomain((10,), ((10, task_count),), kind="site")
        target = CoordinateDomain((20,), ((20, task_count),), kind="event")
        coordinate = coordinate_axis_symbol(10)

        for expression in (
            coordinate + 1,
            sympy.Integer(0),
            sympy.Mod(coordinate, 2),
        ):
            with self.subTest(expression=expression):
                relation = CoordinateRelation.point_map(
                    source,
                    target,
                    (
                        (
                            ((10, 0, task_count, 1),),
                            (expression,),
                        ),
                    ),
                )
                self.assertFalse(relation.is_total_function())
                self.assertIsNone(relation.converse())

    def test_symbolic_pid_task_order_preserves_l2_tail_group(self) -> None:
        domain = CoordinateDomain(
            (10, 20, 30),
            ((10, 5), (20, 3), (30, 2)),
            ((10, 1), (20, 1), (30, 1)),
        )
        relation = pid_task_order(
            domain,
            domain.axis_order,
            l2_group_size=2,
        )
        one_outer_slice = (0, 1, 5, 6, 10, 11, 2, 3, 7, 8, 12, 13, 4, 9, 14)
        expected = (*one_outer_slice, *(task + 15 for task in one_outer_slice))

        self.assertEqual(
            tuple(next(iter(targets)) for targets in relation.materialize()),
            expected,
        )
        converse = relation.converse()
        self.assertIsNotNone(converse)
        assert converse is not None
        self.assertEqual(
            tuple(next(iter(targets)) for targets in converse.materialize()),
            tuple(expected.index(task) for task in range(len(expected))),
        )

    def test_symbolic_pid_task_order_preserves_axis_permutation(self) -> None:
        domain = CoordinateDomain(
            (10, 20, 30),
            ((10, 2), (20, 3), (30, 4)),
            ((10, 1), (20, 1), (30, 1)),
        )
        task_order = pid_task_order(domain, (20, 10, 30))
        pid_to_logical = tuple(
            next(iter(targets)) for targets in task_order.materialize()
        )

        self.assertEqual(sorted(pid_to_logical), list(range(domain.size)))
        converse = task_order.converse()
        self.assertIsNotNone(converse)
        assert converse is not None
        logical_to_pid = tuple(
            next(iter(targets)) for targets in converse.materialize()
        )
        self.assertEqual(
            tuple(pid_to_logical[pid_task] for pid_task in logical_to_pid),
            tuple(range(domain.size)),
        )

    def test_mixed_radix_readiness_quotient_derives_publication(self) -> None:
        for slots in (1, 2, 8, 64):
            with self.subTest(slots=slots):
                consumer_domain = CoordinateDomain(
                    (20, 21),
                    ((20, slots), (21, 8)),
                    ((20, 1), (21, 256)),
                    identity=1,
                )
                producer_domain = CoordinateDomain(
                    (10,),
                    ((10, slots * 256),),
                    ((10, 16),),
                    identity=0,
                )
                readiness_key_domain = dataclasses.replace(
                    consumer_domain,
                    kind="event",
                    identity=None,
                )
                slot = coordinate_axis_symbol(20)
                activation_block = coordinate_axis_symbol(21)
                begin = 256 * slot + 16 * activation_block
                bounds = ((20, 0, slots, 1), (21, 0, 8, 1))
                dependency = CoordinateRelation(
                    consumer_domain,
                    producer_domain,
                    (
                        _CoordinateRelationPiece(
                            bounds,
                            ((10, begin, begin + 16, 1),),
                        ),
                        _CoordinateRelationPiece(
                            bounds,
                            ((10, begin + 128, begin + 144, 1),),
                        ),
                    ),
                )
                consumer_to_key = CoordinateRelation.projection(
                    consumer_domain,
                    readiness_key_domain,
                )
                assert consumer_to_key is not None

                producers_by_key = dependency.factor_through(consumer_to_key)

                self.assertIsNotNone(producers_by_key)
                assert producers_by_key is not None
                self.assertEqual(len(producers_by_key.pieces), 2)
                arrival_count_by_key = producers_by_key.target_count_by_source()
                self.assertIsNotNone(arrival_count_by_key)
                assert arrival_count_by_key is not None
                self.assertEqual(arrival_count_by_key.constant_value(), 32)
                keys_by_producer = producers_by_key.converse()
                self.assertIsNotNone(keys_by_producer)
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
        readiness_key_domain = CoordinateDomain(
            (20, 21),
            ((20, slots), (21, 8)),
            kind="event",
        )
        producer_domain = CoordinateDomain((10,), ((10, slots * 256),), identity=0)
        slot = coordinate_axis_symbol(20)
        activation_block = coordinate_axis_symbol(21)
        begin = 256 * slot + 16 * activation_block
        producers_by_key = CoordinateRelation(
            readiness_key_domain,
            producer_domain,
            (
                _CoordinateRelationPiece(
                    ((20, 0, slots, 1), (21, 0, 8, 1)),
                    ((10, begin, begin + 16, 1),),
                ),
            ),
        )

        arrival_count_by_key = producers_by_key.target_count_by_source()

        self.assertIsNotNone(arrival_count_by_key)
        assert arrival_count_by_key is not None
        self.assertEqual(arrival_count_by_key.constant_value(), 16)
        self.assertIsNone(producers_by_key.converse())

    def test_mixed_radix_converse_matches_reversed_axis_relation(self) -> None:
        source_domain = CoordinateDomain(
            (21, 20),
            ((21, 3), (20, 2)),
            kind="site",
        )
        target_domain = CoordinateDomain((10,), ((10, 24),), kind="allocation")
        inner = coordinate_axis_symbol(21)
        outer = coordinate_axis_symbol(20)
        begin = 2 * inner + 12 * outer
        bounds = ((21, 0, 3, 1), (20, 0, 2, 1))
        targets_by_source = CoordinateRelation(
            source_domain,
            target_domain,
            (
                _CoordinateRelationPiece(bounds, ((10, begin, begin + 2, 1),)),
                _CoordinateRelationPiece(bounds, ((10, begin + 6, begin + 8, 1),)),
            ),
        )

        sources_by_target = targets_by_source.converse()

        self.assertIsNotNone(sources_by_target)
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
        producer = CoordinateDomain((10,), ((10, 684),), identity=0)
        keys = CoordinateDomain((0,), ((0, 3),), kind="event", identity=0)
        source = coordinate_axis_symbol(10)
        key = sympy.Mod(sympy.floor(source / 16) + 2, 3)
        producer_to_key = CoordinateRelation(
            producer,
            keys,
            (
                _CoordinateRelationPiece(
                    ((10, 400, 448, 1),),
                    ((0, key, key + 1, 1),),
                ),
            ),
        )

        producers_by_key = producer_to_key.converse()

        self.assertIsNotNone(producers_by_key)
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
        source = CoordinateDomain(
            (10, 11),
            ((10, 32), (11, 3)),
            kind="task_order",
        )
        keys = CoordinateDomain((20,), ((20, 16),), kind="event")
        source_inner = coordinate_axis_symbol(10)
        source_to_key = CoordinateRelation.point_map(
            source,
            keys,
            (
                (
                    ((10, 0, 32, 1), (11, 0, 3, 1)),
                    (sympy.Mod(source_inner, 16),),
                ),
            ),
        )

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
        order = CoordinateDomain(
            (10, 11),
            ((10, 16), (11, 3)),
            kind="task_order",
        )
        tasks = CoordinateDomain(
            (20, 21, 22, 23),
            ((20, 3), (21, 2), (22, 4), (23, 2)),
            kind="site",
        )
        inner = coordinate_axis_symbol(10)
        outer = coordinate_axis_symbol(11)
        task_order = CoordinateRelation.point_map(
            order,
            tasks,
            (
                (
                    ((10, 0, 6, 1), (11, 0, 3, 1)),
                    (
                        outer,
                        sympy.Integer(0),
                        sympy.Mod(inner, 3),
                        sympy.floor(inner / 3),
                    ),
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
        retained = CoordinateDomain((10,), ((10, 2),), kind="site")

        projected = small.project_source(retained)

        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertEqual(
            projected.materialize(),
            (frozenset((0,)), frozenset((1,))),
        )
        self.assertIsNone(large.project_source(retained))

    def test_project_source_keeps_symbolic_outer_axis_and_unions_static_inner(
        self,
    ) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        source = CoordinateDomain(
            (10, 11),
            ((10, key_count), (11, 4)),
            kind="site",
        )
        retained = CoordinateDomain(
            (10,),
            ((10, key_count),),
            kind="site",
        )
        producer = CoordinateDomain(
            (20,),
            ((20, 4 * key_count),),
            kind="site",
        )
        outer = coordinate_axis_symbol(10)
        inner = coordinate_axis_symbol(11)
        relation = CoordinateRelation.point_map(
            source,
            producer,
            (
                (
                    ((10, 0, key_count, 1), (11, 0, 4, 1)),
                    (4 * outer + inner,),
                ),
            ),
        )

        projected = relation.project_source(retained)

        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertEqual(
            projected.pieces,
            (
                _CoordinateRelationPiece(
                    ((10, 0, key_count, 1),),
                    ((20, 4 * outer, 4 * outer + 4, 1),),
                ),
            ),
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

    def test_project_source_exact_nested_c4_and_rejects_nonrectangular_union(
        self,
    ) -> None:
        key_count = sympy.Symbol("key_count", integer=True, nonnegative=True)
        source = CoordinateDomain(
            (10, 11),
            ((10, key_count), (11, 4)),
            kind="site",
        )
        retained = CoordinateDomain((10,), ((10, key_count),), kind="site")
        producer = CoordinateDomain(
            (20,),
            ((20, 16 * key_count),),
            kind="site",
        )
        key = coordinate_axis_symbol(10)
        split = coordinate_axis_symbol(11)
        begin = 16 * key + 4 * split
        relation = CoordinateRelation(
            source,
            producer,
            (
                _CoordinateRelationPiece(
                    ((10, 0, key_count, 1), (11, 0, 4, 1)),
                    ((20, begin, begin + 4, 1),),
                ),
            ),
        )

        projected = relation.project_source(retained)

        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertEqual(
            projected.pieces,
            (
                _CoordinateRelationPiece(
                    ((10, 0, key_count, 1),),
                    ((20, 16 * key, 16 * key + 16, 1),),
                ),
            ),
        )

        gapped = dataclasses.replace(
            relation,
            pieces=(
                _CoordinateRelationPiece(
                    ((10, 0, key_count, 1), (11, 0, 4, 1)),
                    ((20, begin, begin + 2, 1),),
                ),
            ),
        )
        self.assertIsNone(gapped.project_source(retained))

        diagonal_target = CoordinateDomain(
            (20, 21),
            ((20, 16 * key_count), (21, 16 * key_count)),
            kind="site",
        )
        diagonal = CoordinateRelation(
            source,
            diagonal_target,
            (
                _CoordinateRelationPiece(
                    ((10, 0, key_count, 1), (11, 0, 4, 1)),
                    (
                        (20, begin, begin + 4, 1),
                        (21, begin, begin + 4, 1),
                    ),
                ),
            ),
        )
        self.assertIsNone(diagonal.project_source(retained))

        dynamic_inner = sympy.Symbol("dynamic_inner", integer=True, nonnegative=True)
        dynamic_source = CoordinateDomain(
            (10, 11),
            ((10, key_count), (11, dynamic_inner)),
            kind="site",
        )
        dynamic_relation = dataclasses.replace(
            relation,
            source_domain=dynamic_source,
            pieces=(
                _CoordinateRelationPiece(
                    ((10, 0, key_count, 1), (11, 0, dynamic_inner, 1)),
                    ((20, begin, begin + 4, 1),),
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

        self.assertIsNotNone(relation)
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
        key_domain = CoordinateDomain((10,), ((10, 2),), kind="event")
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

        self.assertIsNotNone(factored)
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

    def test_dense_converse_lifts_full_singleton_target_axis(self) -> None:
        source = CoordinateDomain((0,), ((0, 2),), kind="event")
        target = CoordinateDomain(
            (10, 11),
            ((10, 1), (11, 8)),
            kind="site",
        )
        source_index = coordinate_axis_symbol(0)
        relation = CoordinateRelation(
            source,
            target,
            (
                _CoordinateRelationPiece(
                    ((0, 0, 2, 1),),
                    (
                        (10, sympy.Integer(0), sympy.Integer(1), 1),
                        (11, 4 * source_index, 4 * source_index + 4, 1),
                    ),
                ),
            ),
        )
        target_counts = relation.target_count_by_source()
        assert target_counts is not None

        converse = _dense_mixed_radix_converse(relation, target_counts)

        self.assertIsNotNone(converse)
        assert converse is not None
        self.assertEqual(
            converse.materialize(),
            (frozenset((0,)),) * 4 + (frozenset((1,)),) * 4,
        )

    def test_dense_converse_rejects_nonfull_singleton_target_axis(self) -> None:
        source = CoordinateDomain((0,), ((0, 2),), kind="event")
        target = CoordinateDomain(
            (10, 11),
            ((10, 1), (11, 8)),
            kind="site",
        )
        source_index = coordinate_axis_symbol(0)
        relation = CoordinateRelation(
            source,
            target,
            (
                _CoordinateRelationPiece(
                    ((0, 0, 2, 1),),
                    (
                        (10, sympy.Integer(0), sympy.Integer(0), 1),
                        (11, 4 * source_index, 4 * source_index + 4, 1),
                    ),
                ),
            ),
        )
        target_counts = relation.target_count_by_source()
        assert target_counts is not None

        self.assertIsNone(_dense_mixed_radix_converse(relation, target_counts))

    def test_dense_converse_rejects_multiple_nontrivial_target_axes(self) -> None:
        source = CoordinateDomain((0,), ((0, 2),), kind="event")
        target = CoordinateDomain(
            (10, 11),
            ((10, 2), (11, 8)),
            kind="site",
        )
        source_index = coordinate_axis_symbol(0)
        relation = CoordinateRelation(
            source,
            target,
            (
                _CoordinateRelationPiece(
                    ((0, 0, 2, 1),),
                    (
                        (10, sympy.Integer(0), sympy.Integer(2), 1),
                        (11, 4 * source_index, 4 * source_index + 4, 1),
                    ),
                ),
            ),
        )
        target_counts = relation.target_count_by_source()
        assert target_counts is not None

        self.assertIsNone(_dense_mixed_radix_converse(relation, target_counts))

    def test_target_enumeration_preserves_multi_piece_bijection(self) -> None:
        producer = CoordinateDomain((10,), ((10, 8),), identity=0)
        keys = CoordinateDomain((0,), ((0, 2),), kind="event", identity=0)
        producer_to_key = CoordinateRelation.point_map(
            producer,
            keys,
            (
                (((10, 0, 4, 1),), (sympy.Integer(0),)),
                (((10, 4, 8, 1),), (sympy.Integer(1),)),
            ),
        )
        converse = producer_to_key.converse()
        self.assertIsNotNone(converse)
        assert converse is not None
        task_order = converse.enumerate_targets_by_source()
        self.assertIsNotNone(task_order)
        assert task_order is not None
        self.assertEqual(
            tuple(next(iter(targets)) for targets in task_order.materialize()),
            tuple(range(producer.size)),
        )

        tail_producer = CoordinateDomain((10,), ((10, 7),), identity=0)
        tail_relation = CoordinateRelation.point_map(
            tail_producer,
            keys,
            (
                (((10, 0, 4, 1),), (sympy.Integer(0),)),
                (((10, 4, 7, 1),), (sympy.Integer(1),)),
            ),
        )
        tail_converse = tail_relation.converse()
        self.assertIsNotNone(tail_converse)
        assert tail_converse is not None
        self.assertIsNone(tail_converse.enumerate_targets_by_source())

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

        self.assertIsNotNone(relation)
        assert relation is not None
        self.assertEqual(len(relation.pieces), 1)
        self.assertEqual(relation.targets(0), frozenset((0, 1)))
        self.assertEqual(relation.targets(123), frozenset((246, 247)))
        self.assertEqual(
            relation.targets(elements // 32 - 1),
            frozenset((elements // 16 - 2, elements // 16 - 1)),
        )

        cardinality = relation.target_count_by_source()
        self.assertIsNotNone(cardinality)
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

        self.assertIsNotNone(relation)
        assert relation is not None
        cardinality = relation.target_count_by_source()
        self.assertIsNotNone(cardinality)
        assert cardinality is not None
        self.assertEqual(
            cardinality.materialize(),
            tuple(frozenset((len(targets),)) for targets in relation.materialize()),
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

                self.assertIsNotNone(relation)
                assert relation is not None
                self.assertLessEqual(len(relation.pieces), 3)
                cardinality = relation.target_count_by_source()
                self.assertIsNotNone(cardinality)
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

        self.assertIsNotNone(relation)
        assert relation is not None
        self.assertEqual(relation.materialize(), (frozenset((2, 3, 4, 5)),))

    def test_relation_coverage_preserves_stride_phase(self) -> None:
        source = CoordinateDomain((10,), ((10, 8),), identity=0)
        key = CoordinateDomain((0,), ((0, 8),), kind="event", identity=0)
        even_sources = CoordinateRelation(
            source,
            key,
            (
                _CoordinateRelationPiece(
                    ((10, 0, 8, 2),),
                    ((0, sympy.Integer(0), sympy.Integer(1), 1),),
                ),
            ),
        )
        odd_sources = CoordinateRelation(
            source,
            key,
            (
                _CoordinateRelationPiece(
                    ((10, 1, 8, 2),),
                    ((0, sympy.Integer(0), sympy.Integer(1), 1),),
                ),
            ),
        )

        self.assertFalse(even_sources.covers(odd_sources))
        source_union = even_sources.union(odd_sources)
        self.assertIsNotNone(source_union)
        assert source_union is not None
        self.assertEqual(source_union.materialize(), (frozenset((0,)),) * 8)

        singleton = CoordinateDomain((20,), ((20, 1),), identity=1)
        even_targets = CoordinateRelation(
            singleton,
            key,
            (
                _CoordinateRelationPiece(
                    ((20, 0, 1, 1),),
                    ((0, sympy.Integer(0), sympy.Integer(8), 2),),
                ),
            ),
        )
        odd_targets = CoordinateRelation(
            singleton,
            key,
            (
                _CoordinateRelationPiece(
                    ((20, 0, 1, 1),),
                    ((0, sympy.Integer(1), sympy.Integer(8), 2),),
                ),
            ),
        )

        self.assertFalse(even_targets.covers(odd_targets))
        target_union = even_targets.union(odd_targets)
        self.assertIsNotNone(target_union)
        assert target_union is not None
        self.assertEqual(target_union.materialize(), (frozenset(range(8)),))

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
        self.assertIsNotNone(relation)
        assert relation is not None
        producer_axis = relation.target_domain.axis_order[0]
        value_domain = CoordinateDomain(
            (0,),
            ((0, 4),),
            kind="value",
        )
        worker_steps = CoordinateRelation.point_map(
            relation.target_domain,
            value_domain,
            (
                (
                    ((producer_axis, 0, 8, 1),),
                    (sympy.floor((coordinate_axis_symbol(producer_axis) + 3) / 4),),
                ),
            ),
        )

        maximum = relation.max_target_value_by_source(worker_steps)

        self.assertIsNotNone(maximum)
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
        consumers = CoordinateDomain(
            (consumer_axis,),
            ((consumer_axis, 64),),
            kind="task_order",
        )
        producers = CoordinateDomain(
            (producer_axis,),
            ((producer_axis, 8),),
            kind="task_order",
        )
        required_producers = CoordinateRelation(
            source_domain=consumers,
            target_domain=producers,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=((consumer_axis, 0, 64, 1),),
                    target_ranges=(
                        (
                            producer_axis,
                            sympy.Mod(
                                coordinate_axis_symbol(consumer_axis),
                                4,
                            ),
                            sympy.Integer(8),
                            4,
                        ),
                    ),
                ),
            ),
        )

        maximum = required_producers.max_target_value_by_source(
            CoordinateRelation.identity(producers, producers)
        )

        self.assertIsNotNone(maximum)
        assert maximum is not None
        self.assertEqual(
            maximum.materialize(),
            tuple(frozenset((4 + consumer % 4,)) for consumer in range(64)),
        )

    def test_symbolic_max_target_value_respects_stride_alignment(self) -> None:
        consumer_axis = 57
        producer_axis = 29
        value_axis = -1
        consumers = CoordinateDomain(
            (consumer_axis,),
            ((consumer_axis, 1),),
            kind="task_order",
        )
        producers = CoordinateDomain(
            (producer_axis,),
            ((producer_axis, 8),),
            kind="task_order",
        )
        values = CoordinateDomain(
            (value_axis,),
            ((value_axis, 108),),
            kind="value",
        )
        required_producers = CoordinateRelation(
            source_domain=consumers,
            target_domain=producers,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=((consumer_axis, 0, 1, 1),),
                    target_ranges=(
                        (
                            producer_axis,
                            sympy.Integer(1),
                            sympy.Integer(8),
                            2,
                        ),
                    ),
                ),
            ),
        )
        producer_values = CoordinateRelation(
            source_domain=producers,
            target_domain=values,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=((producer_axis, 0, 8, 2),),
                    target_ranges=(
                        (
                            value_axis,
                            coordinate_axis_symbol(producer_axis) + 100,
                            coordinate_axis_symbol(producer_axis) + 101,
                            1,
                        ),
                    ),
                ),
                _CoordinateRelationPiece(
                    source_bounds_items=((producer_axis, 1, 8, 2),),
                    target_ranges=(
                        (
                            value_axis,
                            coordinate_axis_symbol(producer_axis),
                            coordinate_axis_symbol(producer_axis) + 1,
                            1,
                        ),
                    ),
                ),
            ),
        )

        maximum = required_producers.max_target_value_by_source(producer_values)

        self.assertIsNotNone(maximum)
        assert maximum is not None
        self.assertEqual(maximum.materialize(), (frozenset((7,)),))

    def test_out_of_domain_point_map_is_not_total(self) -> None:
        source = CoordinateDomain((10,), ((10, 6),), identity=0)
        target = CoordinateDomain((0,), ((0, 2),), kind="event", identity=0)
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((10, 0, 6, 1),),
                    (sympy.floor(coordinate_axis_symbol(10) / 2),),
                ),
            ),
        )

        self.assertFalse(relation.has_total_source())
        self.assertFalse(relation.is_total_function())
        self.assertEqual(relation.materialize()[-2:], (frozenset(), frozenset()))

    def test_out_of_domain_source_support_is_not_counted_as_total(self) -> None:
        source = CoordinateDomain((10,), ((10, 2),), identity=0)
        target = CoordinateDomain((20,), ((20, 1),), identity=1)
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((10, -1, 1, 1),),
                    (sympy.Integer(0),),
                ),
            ),
        )

        self.assertIsNone(relation.source_support_cardinality())
        self.assertFalse(relation.is_total_function())

    def test_empty_target_is_not_counted_as_source_support(self) -> None:
        source = CoordinateDomain((10,), ((10, 2),), identity=0)
        target = CoordinateDomain((20,), ((20, 1),), identity=1)
        coordinate = coordinate_axis_symbol(10)
        relation = CoordinateRelation.point_map(
            source,
            target,
            (
                (
                    ((10, 0, 2, 1),),
                    (coordinate + 1,),
                ),
            ),
        )

        self.assertEqual(relation.materialize(), (frozenset(), frozenset()))
        self.assertIsNone(relation.source_support_cardinality())
        self.assertFalse(relation.is_total_function())

    def test_pointwise_strict_order_is_proved_on_common_affine_partition(
        self,
    ) -> None:
        source = CoordinateDomain((10,), ((10, 8),), identity=0)
        values = CoordinateDomain((0,), ((0, 32),), kind="value")
        coordinate = coordinate_axis_symbol(10)
        left = CoordinateRelation.point_map(
            source,
            values,
            (
                (((10, 0, 4, 1),), (coordinate,)),
                (((10, 4, 8, 1),), (coordinate + 2,)),
            ),
        )
        right = CoordinateRelation.point_map(
            source,
            values,
            (
                (((10, 0, 2, 1),), (coordinate + 1,)),
                (((10, 2, 4, 1),), (coordinate + 3,)),
                (((10, 4, 8, 1),), (coordinate + 3,)),
            ),
        )

        self.assertTrue(left.is_pointwise_strictly_less_than(right))
        self.assertTrue(left.is_pointwise_equal_to(left))
        self.assertFalse(right.is_pointwise_strictly_less_than(left))
        self.assertFalse(left.is_pointwise_strictly_less_than(left))

        partial_right = CoordinateRelation.point_map(
            source,
            values,
            ((((10, 0, 7, 1),), (coordinate + 1,)),),
        )
        self.assertFalse(left.is_pointwise_strictly_less_than(partial_right))
        partial_left = CoordinateRelation.point_map(
            source,
            values,
            ((((10, 0, 7, 1),), (coordinate,)),),
        )
        self.assertTrue(
            partial_left.is_pointwise_strictly_less_than_where_defined(right)
        )
        self.assertFalse(partial_right.is_pointwise_equal_to(right))

    def test_partitioned_total_function_avoids_global_canonicalization(self) -> None:
        source = CoordinateDomain((10,), ((10, 128),), identity=0)
        target = CoordinateDomain((20,), ((20, 128),), identity=1)
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

                self.assertIsNotNone(relation)
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

        self.assertIsNotNone(relation)
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
            CoordinateDomain(
                (15, 16, 17),
                ((15, 128), (16, 16), (17, 1)),
                ((15, 1), (16, 1), (17, 4)),
            ),
            CoordinateDomain(
                (20, 21),
                ((20, 16), (21, 64)),
                ((20, 1), (21, 1)),
            ),
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
                    affine_subscript_ranges=((((20, 512), (21, 1)), 0, 512, 64),),
                ),
            ),
            [[15, 16, 17], [20, 21]],
        )
        output_ranges = tuple(
            (((20, 65536), (21, 128)), offset, offset + 128, 1)
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
        self.assertIsNotNone(quotient)
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
                    affine_subscript_ranges=((((20, 0),), 0, 16, 2),),
                ),
            ),
            [[10], [20]],
        )
        domains = (
            CoordinateDomain((10,), ((10, 16),), ((10, 1),)),
            CoordinateDomain((20,), ((20, 1),), ((20, 1),)),
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
                for _axis, coefficient in coefficients
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
            CoordinateDomain((10, 11), ((10, 4), (11, 4)), ((10, 1), (11, 1))),
            CoordinateDomain((20, 21), ((20, 3), (21, 3)), ((20, 1), (21, 1))),
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

    def test_unproven_write_hazard_falls_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(
                    1,
                    root=1,
                    kind="store",
                    block_ids=(20,),
                    scales=(-1,),
                ),
            ),
            [[10], [20]],
        )

        relation = _root_producers_by_consumer(plan, _one_dimensional_domains())
        self.assertIsNone(relation)

    def test_reversed_mapping_falls_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(
                    1,
                    root=1,
                    kind="load",
                    block_ids=(20,),
                    scales=(-1,),
                ),
            ),
            [[10], [20]],
        )

        relation = _root_producers_by_consumer(plan, _one_dimensional_domains())
        self.assertIsNone(relation)

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
            CoordinateDomain((10, 11), ((10, 2), (11, 4)), ((10, 1), (11, 1))),
            CoordinateDomain((20, 21), ((20, 2), (21, 4)), ((20, 1), (21, 1))),
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
            CoordinateDomain((10, 11), ((10, 32), (11, 8)), ((10, 1), (11, 16))),
            CoordinateDomain((20, 21), ((20, 32), (21, 8)), ((20, 1), (21, 16))),
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
            CoordinateDomain((10, 11), ((10, 32), (11, 8)), ((10, 1), (11, 16))),
            CoordinateDomain((20,), ((20, 256),), ((20, 16),)),
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
            CoordinateDomain((10, 11), ((10, 2), (11, 16)), ((10, 1), (11, 16))),
            CoordinateDomain((20, 21), ((20, 2), (21, 4)), ((20, 1), (21, 32))),
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
            CoordinateDomain((10, 11), ((10, 2), (11, 16)), ((10, 1), (11, 16))),
            CoordinateDomain((20, 21), ((20, 2), (21, 4)), ((20, 1), (21, 32))),
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
