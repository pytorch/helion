from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Literal
from unittest import mock

import sympy
import torch

from test._cross_loop_test_kernels import cartesian_affine_chain

from helion._compiler.device_ir import _collect_memory_op_facts
from helion._compiler.tile_dependency import AllocationRegion
from helion._compiler.tile_dependency import ExecutionScope
from helion._compiler.tile_dependency import LogicalDomain
from helion._compiler.tile_dependency import LogicalRelation
from helion._compiler.tile_dependency import LogicalTaskAxis
from helion._compiler.tile_dependency import TaskFamily
from helion._compiler.tile_dependency import TileAccess
from helion._compiler.tile_dependency import TileDependency
from helion._compiler.tile_dependency import TileDependencyKind
from helion._compiler.tile_dependency import _LogicalRelationPiece
from helion._compiler.tile_dependency import allocation_regions_may_overlap
from helion._compiler.tile_dependency import build_tile_dependency_graph
from helion._compiler.tile_dependency import instantiate_logical_domains
from helion._compiler.tile_dependency import instantiate_symbolic_dependencies
from helion._compiler.tile_dependency import logical_axis_symbol
from helion._compiler.tile_dependency import owner_roots_by_graph_id
from helion._compiler.tile_dependency import physical_traversal_relation
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import skipIfNotCUDA


def _axis_geometry(
    root_domains: tuple[LogicalDomain, ...],
) -> dict[int, tuple[int, int]]:
    return {
        axis: (domain.axis_counts[axis], domain.block_sizes[axis])
        for domain in root_domains
        for axis in domain.axis_order
    }


def _configured_domains(
    graph,
    axis_geometry: dict[int, tuple[int, int]],
) -> tuple[tuple[LogicalDomain, ...], tuple[LogicalDomain | None, ...]]:
    configured_roots, scope_domains = instantiate_logical_domains(
        graph,
        axis_geometry=axis_geometry,
    )
    assert all(domain is not None for domain in configured_roots)
    return (
        tuple(domain for domain in configured_roots if domain is not None),
        scope_domains,
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
    )


def _root_predecessors(
    plan,
    root_domains: tuple[LogicalDomain, ...],
    pair: tuple[int, int] = (0, 1),
) -> tuple[frozenset[int], ...] | None:
    axis_geometry = _axis_geometry(root_domains)
    configured_root_domains, scope_domains = _configured_domains(plan, axis_geometry)
    relations = tuple(
        dependency.relation
        for dependency in instantiate_symbolic_dependencies(
            plan,
            root_domains=configured_root_domains,
            scope_domains=scope_domains,
        )
        if (dependency.producer_root, dependency.consumer_root) == pair
        and dependency.producer_scope_id is None
        and dependency.consumer_scope_id is None
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
        source_traversal=root_domains[pair[1]].axis_order,
        target_traversal=root_domains[pair[0]].axis_order,
    )


def _symbolic_root_relation(
    plan,
    axis_geometry: dict[int, tuple[int, int]],
):
    root_domains, scope_domains = _configured_domains(plan, axis_geometry)
    dependencies = instantiate_symbolic_dependencies(
        plan,
        root_domains=root_domains,
        scope_domains=scope_domains,
    )
    self_relations = tuple(
        dependency.relation
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
) -> tuple[LogicalDomain, LogicalDomain]:
    return (
        LogicalDomain(
            (10,),
            ((10, producer_count),),
            ((10, producer_block),),
        ),
        LogicalDomain(
            (20,),
            ((20, consumer_count),),
            ((20, consumer_block),),
        ),
    )


def _dependency_kinds(edge: TileDependency) -> frozenset[TileDependencyKind]:
    return frozenset(dependency.kind for dependency in edge.access_dependencies)


class TestTileDependency(TestCase):
    def test_logical_domain_separates_geometry_from_traversal(self) -> None:
        domain = LogicalDomain(
            (10, 20),
            ((10, 2), (20, 3)),
            ((10, 4), (20, 8)),
        )
        self.assertEqual(domain.coordinates(3), {10: 1, 20: 1})
        self.assertEqual(
            domain.coordinates(3, traversal=(20, 10)),
            {10: 1, 20: 0},
        )
        self.assertEqual(domain.index({10: 1, 20: 1}), 3)
        self.assertEqual(
            domain.index({10: 1, 20: 0}, traversal=(20, 10)),
            3,
        )

    def test_relation_axis_renaming_preserves_positional_coordinates(self) -> None:
        source = LogicalDomain((10, 20), ((10, 2), (20, 3)))
        target = LogicalDomain((30, 40), ((30, 2), (40, 3)))
        source_10 = logical_axis_symbol(10)
        source_20 = logical_axis_symbol(20)
        relation = LogicalRelation.point_map(
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
        renamed_source = LogicalDomain(
            (20, 10),
            ((20, 2), (10, 3)),
            ((20, 4), (10, 8)),
            kind="worker",
            identity=7,
        )
        renamed_target = LogicalDomain(
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
            relation.rename_source_axes(LogicalDomain((0, 1), ((0, 2), (1, 4))))
        )
        self.assertIsNone(
            relation.rename_target_axes(LogicalDomain((0, 1), ((0, 2), (1, 4))))
        )

    def test_configured_roots_reuse_their_execution_scope_domains(self) -> None:
        graph = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        graph = dataclasses.replace(
            graph,
            execution_scopes=(
                ExecutionScope(0, 0, 0, (), None, "root", (), (10,), True, False),
                ExecutionScope(1, 1, 1, (), None, "root", (), (20,), True, False),
            ),
            scope_ids_by_access=((0,), (1,)),
        )

        root_domains, scope_domains = instantiate_logical_domains(
            graph,
            axis_geometry={10: (8, 16), 20: (4, 32)},
        )

        root_scopes = tuple(scope for scope in graph.execution_scopes if scope.is_root)
        self.assertEqual(len(root_scopes), 2)
        for scope in root_scopes:
            self.assertIs(root_domains[scope.root], scope_domains[scope.scope_id])

    def test_symbolic_physical_traversal_preserves_l2_tail_group(self) -> None:
        domain = LogicalDomain(
            (10, 20, 30),
            ((10, 5), (20, 3), (30, 2)),
            ((10, 1), (20, 1), (30, 1)),
        )
        relation = physical_traversal_relation(
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
        inverse = relation.inverse()
        self.assertIsNotNone(inverse)
        assert inverse is not None
        self.assertEqual(
            tuple(next(iter(targets)) for targets in inverse.materialize()),
            tuple(expected.index(task) for task in range(len(expected))),
        )

    def test_symbolic_physical_traversal_preserves_axis_permutation(self) -> None:
        domain = LogicalDomain(
            (10, 20, 30),
            ((10, 2), (20, 3), (30, 4)),
            ((10, 1), (20, 1), (30, 1)),
        )
        traversal = physical_traversal_relation(domain, (20, 10, 30))
        physical_to_logical = tuple(
            next(iter(targets)) for targets in traversal.materialize()
        )

        self.assertEqual(sorted(physical_to_logical), list(range(domain.size)))
        inverse = traversal.inverse()
        self.assertIsNotNone(inverse)
        assert inverse is not None
        logical_to_physical = tuple(
            next(iter(targets)) for targets in inverse.materialize()
        )
        self.assertEqual(
            tuple(physical_to_logical[physical] for physical in logical_to_physical),
            tuple(range(domain.size)),
        )

    def test_mixed_radix_predecessor_quotient_derives_publication(self) -> None:
        for slots in (1, 2, 8, 64):
            with self.subTest(slots=slots):
                consumer_domain = LogicalDomain(
                    (20, 21),
                    ((20, slots), (21, 8)),
                    ((20, 1), (21, 256)),
                    identity=1,
                )
                producer_domain = LogicalDomain(
                    (10,),
                    ((10, slots * 256),),
                    ((10, 16),),
                    identity=0,
                )
                key_domain = dataclasses.replace(
                    consumer_domain,
                    kind="event",
                    identity=None,
                )
                slot = logical_axis_symbol(20)
                activation_block = logical_axis_symbol(21)
                begin = 256 * slot + 16 * activation_block
                bounds = ((20, 0, slots, 1), (21, 0, 8, 1))
                dependency = LogicalRelation(
                    consumer_domain,
                    producer_domain,
                    (
                        _LogicalRelationPiece(
                            bounds,
                            ((10, begin, begin + 16, 1),),
                        ),
                        _LogicalRelationPiece(
                            bounds,
                            ((10, begin + 128, begin + 144, 1),),
                        ),
                    ),
                )
                consumer_to_key = LogicalRelation.projection(
                    consumer_domain,
                    key_domain,
                )
                assert consumer_to_key is not None

                predecessors = dependency.factor_through(consumer_to_key)

                self.assertIsNotNone(predecessors)
                assert predecessors is not None
                self.assertEqual(len(predecessors.pieces), 2)
                arrivals = predecessors.fiber_cardinality()
                self.assertIsNotNone(arrivals)
                assert arrivals is not None
                self.assertEqual(arrivals.constant_value(), 32)
                publication = predecessors.publication_converse()
                self.assertIsNotNone(publication)
                assert publication is not None
                self.assertEqual(len(publication.pieces), 1)
                self.assertTrue(publication.is_total_function())
                for producer in (0, 15, 16, 127, 128, 255, slots * 256 - 1):
                    self.assertEqual(
                        publication.target_coordinates({10: producer}),
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
        key_domain = LogicalDomain(
            (20, 21),
            ((20, slots), (21, 8)),
            kind="event",
        )
        producer_domain = LogicalDomain((10,), ((10, slots * 256),), identity=0)
        slot = logical_axis_symbol(20)
        activation_block = logical_axis_symbol(21)
        begin = 256 * slot + 16 * activation_block
        predecessors = LogicalRelation(
            key_domain,
            producer_domain,
            (
                _LogicalRelationPiece(
                    ((20, 0, slots, 1), (21, 0, 8, 1)),
                    ((10, begin, begin + 16, 1),),
                ),
            ),
        )

        arrivals = predecessors.fiber_cardinality()

        self.assertIsNotNone(arrivals)
        assert arrivals is not None
        self.assertEqual(arrivals.constant_value(), 16)
        self.assertIsNone(predecessors.publication_converse())

    def test_mixed_radix_converse_matches_reversed_axis_relation(self) -> None:
        key_domain = LogicalDomain(
            (21, 20),
            ((21, 3), (20, 2)),
            kind="event",
        )
        producer_domain = LogicalDomain((10,), ((10, 24),), identity=0)
        inner = logical_axis_symbol(21)
        outer = logical_axis_symbol(20)
        begin = 2 * inner + 12 * outer
        bounds = ((21, 0, 3, 1), (20, 0, 2, 1))
        predecessors = LogicalRelation(
            key_domain,
            producer_domain,
            (
                _LogicalRelationPiece(bounds, ((10, begin, begin + 2, 1),)),
                _LogicalRelationPiece(bounds, ((10, begin + 6, begin + 8, 1),)),
            ),
        )

        publication = predecessors.publication_converse()

        self.assertIsNotNone(publication)
        assert publication is not None
        expected = {
            producer: frozenset(
                key
                for key, producers in enumerate(predecessors.materialize())
                if producer in producers
            )
            for producer in range(producer_domain.size)
        }
        self.assertEqual(
            publication.materialize(),
            tuple(expected[producer] for producer in range(producer_domain.size)),
        )

    def test_fiber_enumeration_preserves_multi_piece_bijection(self) -> None:
        producer = LogicalDomain((10,), ((10, 8),), identity=0)
        keys = LogicalDomain((0,), ((0, 2),), kind="event", identity=0)
        producer_to_key = LogicalRelation.point_map(
            producer,
            keys,
            (
                (((10, 0, 4, 1),), (sympy.Integer(0),)),
                (((10, 4, 8, 1),), (sympy.Integer(1),)),
            ),
        )
        inverse = producer_to_key.inverse()
        self.assertIsNotNone(inverse)
        assert inverse is not None
        traversal = inverse.fiber_enumeration()
        self.assertIsNotNone(traversal)
        assert traversal is not None
        self.assertEqual(
            tuple(next(iter(targets)) for targets in traversal.materialize()),
            tuple(range(producer.size)),
        )

        tail_producer = LogicalDomain((10,), ((10, 7),), identity=0)
        tail_relation = LogicalRelation.point_map(
            tail_producer,
            keys,
            (
                (((10, 0, 4, 1),), (sympy.Integer(0),)),
                (((10, 4, 7, 1),), (sympy.Integer(1),)),
            ),
        )
        tail_inverse = tail_relation.inverse()
        self.assertIsNotNone(tail_inverse)
        assert tail_inverse is not None
        self.assertIsNone(tail_inverse.fiber_enumeration())

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

        cardinality = relation.fiber_cardinality()
        self.assertIsNotNone(cardinality)
        assert cardinality is not None
        self.assertEqual(
            cardinality.materialize(),
            tuple(frozenset((2,)) for _ in range(elements // 32)),
        )

    def test_symbolic_fiber_cardinality_preserves_tail_pieces(self) -> None:
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
        cardinality = relation.fiber_cardinality()
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
                cardinality = relation.fiber_cardinality()
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
        source = LogicalDomain((10,), ((10, 8),), identity=0)
        key = LogicalDomain((0,), ((0, 8),), kind="event", identity=0)
        even_sources = LogicalRelation(
            source,
            key,
            (
                _LogicalRelationPiece(
                    ((10, 0, 8, 2),),
                    ((0, sympy.Integer(0), sympy.Integer(1), 1),),
                ),
            ),
        )
        odd_sources = LogicalRelation(
            source,
            key,
            (
                _LogicalRelationPiece(
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

        singleton = LogicalDomain((20,), ((20, 1),), identity=1)
        even_targets = LogicalRelation(
            singleton,
            key,
            (
                _LogicalRelationPiece(
                    ((20, 0, 1, 1),),
                    ((0, sympy.Integer(0), sympy.Integer(8), 2),),
                ),
            ),
        )
        odd_targets = LogicalRelation(
            singleton,
            key,
            (
                _LogicalRelationPiece(
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

    def test_symbolic_fiber_maximum_reduces_schedule_positions(self) -> None:
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
        value_domain = LogicalDomain(
            (0,),
            ((0, 4),),
            kind="value",
        )
        positions = LogicalRelation.point_map(
            relation.target_domain,
            value_domain,
            (
                (
                    ((producer_axis, 0, 8, 1),),
                    (sympy.floor((logical_axis_symbol(producer_axis) + 3) / 4),),
                ),
            ),
        )

        maximum = relation.fiber_maximum(positions)

        self.assertIsNotNone(maximum)
        assert maximum is not None
        self.assertEqual(
            maximum.materialize(),
            tuple(
                frozenset((max(max(positions.targets(task)) for task in producers),))
                for producers in relation.materialize()
            ),
        )

    def test_out_of_domain_point_map_is_not_total(self) -> None:
        source = LogicalDomain((10,), ((10, 6),), identity=0)
        target = LogicalDomain((0,), ((0, 2),), kind="event", identity=0)
        relation = LogicalRelation.point_map(
            source,
            target,
            (
                (
                    ((10, 0, 6, 1),),
                    (sympy.floor(logical_axis_symbol(10) / 2),),
                ),
            ),
        )

        self.assertFalse(relation.has_total_source())
        self.assertFalse(relation.is_total_function())
        self.assertEqual(relation.materialize()[-2:], (frozenset(), frozenset()))

    def test_partitioned_total_function_avoids_global_canonicalization(self) -> None:
        source = LogicalDomain((10,), ((10, 128),), identity=0)
        target = LogicalDomain((20,), ((20, 128),), identity=1)
        relation = LogicalRelation.point_map(
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
            LogicalRelation,
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
                    relation.materialize(), _root_predecessors(plan, root_domains)
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

    @skipIfNotCUDA()
    def test_shared_device_graph_preserves_every_root_owner(self) -> None:
        x = torch.empty((2, 64), device=DEVICE, dtype=torch.float32)
        bound = cartesian_affine_chain.bind((x,))
        assert bound.host_function is not None
        device_ir = bound.host_function.device_ir
        shared_graph_id = device_ir.root_ids[0]
        shared_family = device_ir.task_families[0]
        original_root_ids = device_ir.root_ids
        original_task_families = device_ir.task_families
        try:
            device_ir.root_ids = [shared_graph_id, shared_graph_id]
            device_ir.task_families = [shared_family, shared_family]
            owners = owner_roots_by_graph_id(device_ir)
            self.assertEqual(owners[shared_graph_id], (0, 1))
            with bound.env, bound.host_function:
                _facts, _liveness, accesses = _collect_memory_op_facts(device_ir)
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
                    all(scope.root == access.root for scope in scopes)
                    for access in dependency_graph.accesses
                    for scopes in (
                        dependency_graph.scopes_for_access(access.access_id),
                    )
                )
            )
        finally:
            device_ir.root_ids = original_root_ids
            device_ir.task_families = original_task_families

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
            LogicalDomain((10, 11), ((10, 4), (11, 4)), ((10, 1), (11, 1))),
            LogicalDomain((20, 21), ((20, 3), (21, 3)), ((20, 1), (21, 1))),
        )
        self.assertIsNone(_root_predecessors(plan, root_domains))

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
            _root_predecessors(
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
                TaskFamily((LogicalTaskAxis(10, None),)),
                TaskFamily((LogicalTaskAxis(20, None),)),
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
            _root_predecessors(plan, _one_dimensional_domains()),
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
            _root_predecessors(plan, _one_dimensional_domains()),
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
            _root_predecessors(plan, _one_dimensional_domains()),
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

        relation = _root_predecessors(plan, _one_dimensional_domains())
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

        relation = _root_predecessors(plan, _one_dimensional_domains())
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
            LogicalDomain((10, 11), ((10, 2), (11, 4)), ((10, 1), (11, 1))),
            LogicalDomain((20, 21), ((20, 2), (21, 4)), ((20, 1), (21, 1))),
        )
        relation = _root_predecessors(plan, root_domains)
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
            LogicalDomain((10, 11), ((10, 32), (11, 8)), ((10, 1), (11, 16))),
            LogicalDomain((20, 21), ((20, 32), (21, 8)), ((20, 1), (21, 16))),
        )
        self.assertEqual(
            _root_predecessors(plan, root_domains),
            tuple(frozenset((task,)) for task in range(256)),
        )

    def test_nontrivial_reshape_still_falls_back_to_root(self) -> None:
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
            LogicalDomain((10, 11), ((10, 32), (11, 8)), ((10, 1), (11, 16))),
            LogicalDomain((20,), ((20, 256),), ((20, 16),)),
        )
        relation = _root_predecessors(plan, root_domains)
        self.assertIsNone(relation)

    def test_unequal_tiles_map_to_every_overlapping_producer(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
        )
        self.assertEqual(
            _root_predecessors(
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
            LogicalDomain((10, 11), ((10, 2), (11, 16)), ((10, 1), (11, 16))),
            LogicalDomain((20, 21), ((20, 2), (21, 4)), ((20, 1), (21, 32))),
        )
        relation = _root_predecessors(plan, root_domains)
        assert relation is not None
        self.assertEqual(len(relation), 8)
        self.assertEqual({len(predecessors) for predecessors in relation}, {4})
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
            LogicalDomain((10, 11), ((10, 2), (11, 16)), ((10, 1), (11, 16))),
            LogicalDomain((20, 21), ((20, 2), (21, 4)), ((20, 1), (21, 32))),
        )
        actual = _root_predecessors(plan, root_domains)
        assert actual is not None
        for consumer_task, predecessors in enumerate(actual):
            coordinates = root_domains[1].coordinates(consumer_task)
            batch = coordinates[20]
            group = coordinates[21]
            self.assertEqual(
                predecessors,
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
            _root_predecessors(
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
            _root_predecessors(
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
        prefix = _root_predecessors(
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
        suffix_relation = _root_predecessors(
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
            _root_predecessors(
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
            _root_predecessors(plan, _one_dimensional_domains()),
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

        self.assertIsNone(_root_predecessors(plan, _one_dimensional_domains()))

    def test_nonzero_or_dynamic_grid_start_falls_back_to_root(self) -> None:
        plan = build_tile_dependency_graph(
            (
                _access(0, root=0, kind="store", block_ids=(10,)),
                _access(1, root=1, kind="load", block_ids=(20,)),
            ),
            [[10], [20]],
            noncanonical_task_origin_block_ids=frozenset((10,)),
        )

        self.assertIsNone(_root_predecessors(plan, _one_dimensional_domains()))

    def test_tracks_latest_writer_and_intervening_readers(self) -> None:
        task_families = tuple(
            TaskFamily(
                axes=(LogicalTaskAxis(root, 128),),
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
                TaskFamily((LogicalTaskAxis(10, 96),)),
                TaskFamily((LogicalTaskAxis(20, 64),)),
                TaskFamily((LogicalTaskAxis(30, 96),)),
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
