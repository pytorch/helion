from __future__ import annotations

import copy
import dataclasses
import itertools
import math
import pickle
from typing import Literal
from unittest import mock

import sympy
import torch
from torch.utils._sympy.functions import FloorDiv

import helion
from helion import exc
from helion._compiler.device_ir_analysis import DeviceIRAnalysis
from helion._compiler.tile_dependency import AllocationRegion
from helion._compiler.tile_dependency import CoordinateDomain
from helion._compiler.tile_dependency import CoordinateRelation
from helion._compiler.tile_dependency import DenseTaskOrder
from helion._compiler.tile_dependency import ExecutionSite
from helion._compiler.tile_dependency import Incidence
from helion._compiler.tile_dependency import KeyPartition
from helion._compiler.tile_dependency import TaskAxis
from helion._compiler.tile_dependency import TaskFamily
from helion._compiler.tile_dependency import TileAccess
from helion._compiler.tile_dependency import TileDependency
from helion._compiler.tile_dependency import TileDependencyKind
from helion._compiler.tile_dependency import _access_layout
from helion._compiler.tile_dependency import _CoordinateRelationPiece
from helion._compiler.tile_dependency import _simplify_logical_expression
from helion._compiler.tile_dependency import _symbolic_access_map
from helion._compiler.tile_dependency import _transpose_projection_or_singletons
from helion._compiler.tile_dependency import allocation_regions_may_overlap
from helion._compiler.tile_dependency import build_tile_dependency_graph
from helion._compiler.tile_dependency import coordinate_axis_symbol
from helion._compiler.tile_dependency import instantiate_coordinate_domains
from helion._compiler.tile_dependency import instantiate_symbolic_dependencies
from helion._compiler.tile_dependency import owner_roots_by_graph_id
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


def _incidence(items_by_key: CoordinateRelation, *, grouped: bool = False) -> Incidence:
    """Materialize small fixtures into an explicit paired capability bundle."""
    fibers = _materialize(items_by_key)
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
    count_by_key = CoordinateRelation.point_map(
        items_by_key.source_domain,
        CoordinateDomain.scalar(
            max(map(len, fibers), default=0) + 1, axis=count_axis, kind="value"
        ),
        tuple(
            (
                tuple(
                    (axis, value, value + 1, 1)
                    for axis, value in _coordinates(
                        items_by_key.source_domain, source
                    ).items()
                ),
                (len(targets),),
            )
            for source, targets in enumerate(fibers)
        ),
    )
    keys_by_item = _transpose_projection_or_singletons(items_by_key)
    if keys_by_item is None:
        pairs = [
            (target, source)
            for source, targets in enumerate(fibers)
            for target in targets
        ]
        if len({target for target, _source in pairs}) != len(pairs):
            return Incidence._from_constructed(items_by_key)
        keys_by_item = CoordinateRelation.point_map(
            items_by_key.target_domain,
            items_by_key.source_domain,
            tuple(
                (
                    tuple(
                        (axis, value, value + 1, 1)
                        for axis, value in _coordinates(
                            items_by_key.target_domain, target
                        ).items()
                    ),
                    tuple(_coordinates(items_by_key.source_domain, source).values()),
                )
                for target, source in pairs
            ),
        )
    incidence = Incidence._from_constructed(
        items_by_key,
        keys_by_item=keys_by_item,
        count_by_key=count_by_key,
    )
    return incidence.with_key_major_order() if grouped else incidence


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
        layout_is_symbolically_exact=layout_is_static,
    )


def _root_producers_by_consumer(
    plan,
    root_domains: tuple[CoordinateDomain, ...],
    pair: tuple[int, int] = (0, 1),
) -> tuple[frozenset[int], ...] | None:
    axis_geometry = _axis_geometry(root_domains)
    configured_root_domains, site_domains = _configured_domains(plan, axis_geometry)
    relations = tuple(
        None if dependency.incidence is None else dependency.incidence.items_by_key
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
    result = CoordinateRelation.union_all(concrete)
    if result is None:
        return None
    return _materialize(
        result,
        source_axis_order=root_domains[pair[1]].axis_order,
        target_axis_order=root_domains[pair[0]].axis_order,
    )


def _coordinates(
    domain: CoordinateDomain,
    index: int,
    axis_order: tuple[int, ...] | None = None,
) -> dict[int, int]:
    order = axis_order or domain.axis_order
    if not 0 <= index < domain.size:
        raise ValueError("coordinate index is outside the domain")
    result = {}
    for axis in order:
        count = domain.axis_counts[axis]
        result[axis], index = index % count, index // count
    return result


def _index(
    domain: CoordinateDomain,
    coordinates: dict[int, int],
    axis_order: tuple[int, ...] | None = None,
) -> int:
    result, stride = 0, 1
    for axis in axis_order or domain.axis_order:
        result += coordinates[axis] * stride
        stride *= domain.axis_counts[axis]
    return result


def _targets(
    relation: CoordinateRelation,
    source_index: int,
    *,
    source_axis_order: tuple[int, ...] | None = None,
    target_axis_order: tuple[int, ...] | None = None,
) -> frozenset[int]:
    source = _coordinates(relation.source_domain, source_index, source_axis_order)
    substitutions = {
        coordinate_axis_symbol(axis): sympy.Integer(value)
        for axis, value in source.items()
    }
    result = set()
    for piece in relation.pieces:
        if not all(
            int(begin) <= source[axis] < int(end)
            and (source[axis] - int(begin)) % step == 0
            for axis, begin, end, step in piece.source_bounds_items
        ):
            continue
        ranges = []
        for axis, begin, end, step in piece.target_ranges:
            begin_value = int(begin.xreplace(substitutions))
            end_value = int(end.xreplace(substitutions))
            count = relation.target_domain.axis_counts[axis]
            ranges.append(
                tuple(
                    value
                    for value in range(begin_value, end_value, step)
                    if 0 <= value < count
                )
            )
        result.update(
            _index(
                relation.target_domain,
                dict(zip(relation.target_domain.axis_order, point, strict=True)),
                target_axis_order,
            )
            for point in itertools.product(*ranges)
        )
    return frozenset(result)


def _materialize(
    relation: CoordinateRelation,
    *,
    source_axis_order: tuple[int, ...] | None = None,
    target_axis_order: tuple[int, ...] | None = None,
) -> tuple[frozenset[int], ...]:
    return tuple(
        _targets(
            relation,
            index,
            source_axis_order=source_axis_order,
            target_axis_order=target_axis_order,
        )
        for index in range(relation.source_domain.size)
    )


def _symbolic_root_relation(
    plan,
    axis_geometry: dict[int, tuple[int, int]],
):
    root_domains, site_domains = _configured_domains(plan, axis_geometry)
    dependencies = instantiate_symbolic_dependencies(
        plan,
        root_domains=root_domains,
        site_domains=site_domains,
    )
    self_relations = tuple(
        None if dependency.incidence is None else dependency.incidence.items_by_key
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
        self.assertEqual(_coordinates(domain, 3), {10: 1, 20: 1})
        self.assertEqual(
            _coordinates(domain, 3, (20, 10)),
            {10: 1, 20: 0},
        )
        self.assertEqual(_index(domain, {10: 1, 20: 1}), 3)
        self.assertEqual(
            _index(domain, {10: 1, 20: 0}, (20, 10)),
            3,
        )

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
        self.assertEqual(_materialize(renamed), _materialize(relation))
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
                ExecutionSite(0, 0, 0, (), None, "root", (10,), True, False),
                ExecutionSite(1, 1, 1, (), None, "root", (20,), True, False),
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

    def test_symbolic_pid_task_order_preserves_l2_tail_group(self) -> None:
        domain = CoordinateDomain(
            (10, 20, 30),
            ((10, 5), (20, 3), (30, 2)),
            ((10, 1), (20, 1), (30, 1)),
        )
        dense = DenseTaskOrder.from_pid(
            domain,
            domain.axis_order,
            l2_group_size=2,
        )
        assert dense is not None
        relation = dense.tasks_by_ordinal
        one_outer_slice = (0, 1, 5, 6, 10, 11, 2, 3, 7, 8, 12, 13, 4, 9, 14)
        expected = (*one_outer_slice, *(task + 15 for task in one_outer_slice))

        self.assertEqual(
            tuple(next(iter(targets)) for targets in _materialize(relation)),
            expected,
        )
        converse = dense.ordinal_by_task
        self.assertEqual(
            tuple(next(iter(targets)) for targets in _materialize(converse)),
            tuple(expected.index(task) for task in range(len(expected))),
        )

    def test_symbolic_pid_task_order_preserves_axis_permutation(self) -> None:
        domain = CoordinateDomain(
            (10, 20, 30),
            ((10, 2), (20, 3), (30, 4)),
            ((10, 1), (20, 1), (30, 1)),
        )
        dense = DenseTaskOrder.from_pid(domain, (20, 10, 30))
        assert dense is not None
        task_order = dense.tasks_by_ordinal
        pid_to_logical = tuple(
            next(iter(targets)) for targets in _materialize(task_order)
        )

        self.assertEqual(sorted(pid_to_logical), list(range(domain.size)))
        converse = dense.ordinal_by_task
        logical_to_pid = tuple(
            next(iter(targets)) for targets in _materialize(converse)
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

                producers_by_key = dependency.rename_source_axes(readiness_key_domain)

                self.assertIsNotNone(producers_by_key)
                assert producers_by_key is not None
                self.assertEqual(len(producers_by_key.pieces), 2)
                producer = coordinate_axis_symbol(10)
                keys_by_producer = CoordinateRelation.point_map(
                    producer_domain,
                    readiness_key_domain,
                    (
                        (
                            ((10, 0, slots * 256, 1),),
                            (
                                FloorDiv(producer, 256),
                                sympy.Mod(FloorDiv(producer, 16), 8),
                            ),
                        ),
                    ),
                )
                incidence = Incidence._from_constructed(
                    producers_by_key,
                    keys_by_item=keys_by_producer,
                    count_by_key=CoordinateRelation.point_map(
                        readiness_key_domain,
                        CoordinateDomain.scalar(33, axis=22, kind="value"),
                        ((bounds, (sympy.Integer(32),)),),
                    ),
                ).with_key_major_order()
                arrival_count_by_key = incidence.count_by_key
                self.assertIsNotNone(arrival_count_by_key)
                assert arrival_count_by_key is not None
                self.assertEqual(arrival_count_by_key.value_bounds(), (32, 32))
                self.assertEqual(len(keys_by_producer.pieces), 1)
                self.assertTrue(keys_by_producer.is_total_function())
                for producer in (0, 15, 16, 127, 128, 255, slots * 256 - 1):
                    self.assertEqual(
                        frozenset(
                            tuple(
                                _coordinates(keys_by_producer.target_domain, target)[
                                    axis
                                ]
                                for axis in keys_by_producer.target_domain.axis_order
                            )
                            for target in _targets(keys_by_producer, producer)
                        ),
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

        incidence = _incidence(producers_by_key)
        assert incidence.count_by_key is not None
        self.assertEqual(incidence.count_by_key.value_bounds(), (16, 16))
        self.assertIsNotNone(incidence.keys_by_item)

    def test_target_enumeration_preserves_multi_piece_bijection(self) -> None:
        producer = CoordinateDomain((10,), ((10, 8),), identity=0)
        keys = CoordinateDomain((0,), ((0, 2),), kind="event", identity=0)
        converse = CoordinateRelation(
            keys,
            producer,
            (
                _CoordinateRelationPiece(((0, 0, 1, 1),), ((10, 0, 4, 1),)),
                _CoordinateRelationPiece(((0, 1, 2, 1),), ((10, 4, 8, 1),)),
            ),
        )
        incidence = _incidence(converse, grouped=True)
        task_order = incidence.grouped_items
        self.assertIsNotNone(task_order)
        assert task_order is not None
        self.assertEqual(
            tuple(
                next(iter(targets))
                for targets in _materialize(task_order.tasks_by_ordinal)
            ),
            tuple(range(producer.size)),
        )

        tail_producer = CoordinateDomain((10,), ((10, 7),), identity=0)
        tail_converse = CoordinateRelation(
            keys,
            tail_producer,
            (
                _CoordinateRelationPiece(((0, 0, 1, 1),), ((10, 0, 4, 1),)),
                _CoordinateRelationPiece(((0, 1, 2, 1),), ((10, 4, 7, 1),)),
            ),
        )
        tail_order = _incidence(tail_converse, grouped=True).grouped_items
        self.assertIsNotNone(tail_order)
        assert tail_order is not None
        self.assertEqual(
            tuple(
                next(iter(targets))
                for targets in _materialize(tail_order.tasks_by_ordinal)
            ),
            tuple(range(tail_producer.size)),
        )

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
        self.assertEqual(_targets(relation, 0), frozenset((0, 1)))
        self.assertEqual(_targets(relation, 123), frozenset((246, 247)))
        self.assertEqual(
            _targets(relation, elements // 32 - 1),
            frozenset((elements // 16 - 2, elements // 16 - 1)),
        )

        cardinality = _incidence(relation).count_by_key
        self.assertIsNotNone(cardinality)
        assert cardinality is not None
        self.assertEqual(
            _materialize(cardinality),
            tuple(frozenset((2,)) for _ in range(elements // 32)),
        )

    def test_symbolic_tail_relation_preserves_exact_pieces(self) -> None:
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
        self.assertEqual(tuple(map(len, _materialize(relation))), (2, 2, 2))

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
                expected = tuple(
                    frozenset(
                        (
                            (math.ceil((group + 1) * group_width / producer_block))
                            - (group * group_width // producer_block),
                        )
                    )
                    for group in range(groups)
                )
                actual = tuple(
                    frozenset((len(targets),)) for targets in _materialize(relation)
                )
                self.assertEqual(actual, expected)
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
        self.assertEqual(_materialize(relation), (frozenset((2, 3, 4, 5)),))

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
        source_union = CoordinateRelation.union_all((even_sources, odd_sources))
        self.assertIsNotNone(source_union)
        assert source_union is not None
        self.assertEqual(_materialize(source_union), (frozenset((0,)),) * 8)

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
        target_union = CoordinateRelation.union_all((even_targets, odd_targets))
        self.assertIsNotNone(target_union)
        assert target_union is not None
        self.assertEqual(_materialize(target_union), (frozenset(range(8)),))

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
        self.assertEqual(_materialize(relation)[-2:], (frozenset(), frozenset()))

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
                    _materialize(relation),
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
            coordinates = _coordinates(consumer, consumer_task)
            expected = frozenset(
                _index(
                    producer,
                    {
                        10: coordinates[20],
                        11: 2 * coordinates[21] + offset,
                    },
                )
                for offset in range(2)
            )
            self.assertEqual(_targets(relation, consumer_task), expected)

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
            self.assertTrue(
                any(
                    edge.producer_root == 0 and edge.consumer_root == 1
                    for edge in dependency_graph.edges
                )
            )
            self.assertTrue(
                all(
                    all(site.root == access.root for site in sites)
                    for access in dependency_graph.accesses
                    for sites in (
                        tuple(
                            dependency_graph.execution_sites[site_id]
                            for site_id in dependency_graph.site_ids_by_access[
                                access.access_id
                            ]
                        ),
                    )
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

        edge = next(
            edge
            for edge in plan.edges
            if edge.producer_root == 1 and edge.consumer_root == 2
        )
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
            _coordinates(root_domains[0], producer_task),
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
            CoordinateDomain((10, 11), ((10, 32), (11, 8)), ((10, 1), (11, 16))),
            CoordinateDomain((20,), ((20, 256),), ((20, 16),)),
        )
        relation = _root_producers_by_consumer(plan, root_domains)
        self.assertIsNotNone(relation)
        assert relation is not None
        self.assertTrue(all(len(producers) == 1 for producers in relation))

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
            coordinates = _coordinates(root_domains[1], consumer_task)
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

    def test_dense_qwen_order_and_capability_bundles(self) -> None:
        tasks = CoordinateDomain((10, 20), ((10, 16), (20, 96)), identity=0)
        order = DenseTaskOrder.from_pid(tasks, tasks.axis_order, l2_group_size=8)
        assert order is not None
        expected = {0: 0, 7: 7, 8: 16, 767: 1527, 768: 8, 1535: 1535}
        for ordinal, task in expected.items():
            self.assertEqual(_targets(order.tasks_by_ordinal, ordinal), {task})
            self.assertEqual(_targets(order.ordinal_by_task, task), {ordinal})

        key = CoordinateDomain((0,), ((0, 4),), kind="event", identity=2)
        item = CoordinateDomain((1,), ((1, 4),), identity=3)
        coordinate = coordinate_axis_symbol(0)
        relation = CoordinateRelation.point_map(
            key, item, (((((0, 0, 4, 1),), (coordinate,))),)
        )
        incidence = _incidence(relation, grouped=True)
        for value in (order, incidence):
            with self.assertRaises(TypeError):
                dataclasses.replace(value)
            self.assertEqual(copy.deepcopy(value), value)
            self.assertEqual(pickle.loads(pickle.dumps(value)), value)

    def test_grouped_incidence_requires_complete_unique_item_coverage(self) -> None:
        keys = CoordinateDomain.scalar(2, kind="event")
        items = CoordinateDomain.scalar(2, axis=1)
        counts = CoordinateDomain.scalar(2, axis=2, kind="value")
        items_by_key = CoordinateRelation.point_map(
            keys,
            items,
            (((((0, 0, 1, 1),), (0,))), ((((0, 1, 2, 1),), (0,)))),
        )
        keys_by_item = CoordinateRelation(
            items,
            keys,
            (_CoordinateRelationPiece(((1, 0, 1, 1),), ((0, 0, 2, 1),)),),
        )
        count_by_key = CoordinateRelation.point_map(
            keys, counts, (((((0, 0, 2, 1),), (1,))),)
        )
        incidence = Incidence._from_constructed(
            items_by_key, keys_by_item=keys_by_item, count_by_key=count_by_key
        ).with_key_major_order()
        self.assertIsNone(incidence.grouped_items)

    def test_relation_construction_deduplicates_identical_pieces(self) -> None:
        source = CoordinateDomain.scalar(2)
        target = CoordinateDomain.scalar(3, axis=1)
        piece = _CoordinateRelationPiece(
            ((0, 0, 2, 1),),
            ((1, 0, 3, 1),),
        )
        relation = CoordinateRelation(source, target, (piece, piece))
        self.assertEqual(relation.pieces, (piece,))

        incidence = Incidence.from_fibers(relation)
        assert incidence.keys_by_item is not None
        assert incidence.count_by_key is not None
        self.assertEqual(
            _materialize(incidence.keys_by_item),
            (frozenset((0, 1)),) * 3,
        )
        self.assertEqual(
            _materialize(incidence.count_by_key),
            (frozenset((3,)),) * 2,
        )

    def test_symbolic_uniform_fibers_keep_count_without_dense_order(self) -> None:
        batch = sympy.Symbol("batch", integer=True, positive=True)
        keys = CoordinateDomain((0,), ((0, batch),), kind="event")
        items = CoordinateDomain((1,), ((1, 2 * batch),), kind="site")
        key = coordinate_axis_symbol(0)
        incidence = Incidence.from_fibers(
            CoordinateRelation(
                keys,
                items,
                (
                    _CoordinateRelationPiece(
                        ((0, 0, batch, 1),),
                        ((1, 2 * key, 2 * key + 2, 1),),
                    ),
                ),
            )
        )
        assert incidence.count_by_key is not None
        self.assertEqual(incidence.count_by_key.value_bounds(), (2, 2))
        self.assertIsNone(incidence.grouped_items)

        item = coordinate_axis_symbol(1)
        reflected = Incidence.from_fibers(
            CoordinateRelation(
                keys,
                items,
                (
                    _CoordinateRelationPiece(
                        ((0, 0, batch, 1),),
                        ((1, 2 * (batch - 1 - key), 2 * (batch - key), 1),),
                    ),
                ),
            ),
            keys_by_item=CoordinateRelation.point_map(
                items,
                keys,
                (((((1, 0, 2 * batch, 1),), (batch - 1 - FloorDiv(item, 2),))),),
            ),
        )
        assert reflected.count_by_key is not None
        self.assertEqual(reflected.count_by_key.value_bounds(), (2, 2))
        self.assertIsNone(reflected.grouped_items)

    def test_key_major_order_declines_unproved_alternating_order(self) -> None:
        keys = CoordinateDomain.scalar(2, kind="event")
        items = CoordinateDomain.scalar(6, axis=1)
        counts = CoordinateDomain.scalar(4, axis=2, kind="value")
        item = coordinate_axis_symbol(1)
        compact_reverse = CoordinateRelation.point_map(
            items,
            keys,
            (((((1, 0, 6, 1),), (1 - sympy.Mod(item, 2),))),),
        )
        explicit_reverse = CoordinateRelation.point_map(
            items,
            keys,
            tuple(
                ((((1, index, index + 1, 1),), (owner,)))
                for index, owner in enumerate((1, 0, 1, 0, 1, 0))
            ),
        )
        for keys_by_item in (compact_reverse, explicit_reverse):
            with self.subTest(keys_by_item=keys_by_item):
                incidence = Incidence._from_constructed(
                    CoordinateRelation(
                        keys,
                        items,
                        (
                            _CoordinateRelationPiece(((0, 0, 1, 1),), ((1, 1, 6, 2),)),
                            _CoordinateRelationPiece(((0, 1, 2, 1),), ((1, 0, 6, 2),)),
                        ),
                    ),
                    keys_by_item=keys_by_item,
                    count_by_key=CoordinateRelation.point_map(
                        keys,
                        counts,
                        (((((0, 0, 2, 1),), (3,))),),
                    ),
                ).with_key_major_order()
                self.assertIsNone(incidence.grouped_items)
                assert incidence.count_by_key is not None
                self.assertEqual(
                    _materialize(incidence.count_by_key),
                    (frozenset((3,)), frozenset((3,))),
                )

    def test_key_major_order_packs_uniform_disjoint_spans(self) -> None:
        keys = CoordinateDomain.scalar(2, kind="event")
        items = CoordinateDomain.scalar(8, axis=1)
        key = coordinate_axis_symbol(0)
        item = coordinate_axis_symbol(1)
        incidence = Incidence._from_constructed(
            CoordinateRelation(
                keys,
                items,
                (
                    _CoordinateRelationPiece(
                        ((0, 0, 2, 1),),
                        ((1, 2 * key, 2 * key + 2, 1),),
                    ),
                    _CoordinateRelationPiece(
                        ((0, 0, 2, 1),),
                        ((1, 2 * key + 4, 2 * key + 6, 1),),
                    ),
                ),
            ),
            keys_by_item=CoordinateRelation.point_map(
                items,
                keys,
                (
                    (((1, 0, 4, 1),), (FloorDiv(item, 2),)),
                    (((1, 4, 8, 1),), (FloorDiv(item - 4, 2),)),
                ),
            ),
        ).with_key_major_order()
        assert incidence.count_by_key is not None
        assert incidence.grouped_items is not None
        self.assertEqual(incidence.count_by_key.value_bounds(), (4, 4))
        self.assertEqual(
            tuple(
                next(iter(values))
                for values in _materialize(incidence.grouped_items.tasks_by_ordinal)
            ),
            (0, 1, 4, 5, 2, 3, 6, 7),
        )

    def test_uniform_count_rejects_out_of_domain_fibers(self) -> None:
        keys = CoordinateDomain.scalar(1, kind="event")
        items = CoordinateDomain.scalar(1, axis=1)
        incidence = Incidence._from_constructed(
            CoordinateRelation(
                keys,
                items,
                (_CoordinateRelationPiece(((0, 0, 1, 1),), ((1, -1, 1, 1),)),),
            ),
            keys_by_item=CoordinateRelation.point_map(
                items,
                keys,
                (((((1, 0, 1, 1),), (0,))),),
            ),
        ).with_key_major_order()
        self.assertIsNone(incidence.count_by_key)
        self.assertIsNone(incidence.grouped_items)

        grid = CoordinateDomain((1, 2), ((1, 2), (2, 2)))
        aliased_rank = Incidence._from_constructed(
            CoordinateRelation(
                keys,
                grid,
                (
                    _CoordinateRelationPiece(
                        ((0, 0, 1, 1),),
                        ((1, 2, 3, 1), (2, 0, 1, 1)),
                    ),
                ),
            ),
            keys_by_item=CoordinateRelation(grid, keys, ()),
        ).with_key_major_order()
        self.assertIsNone(aliased_rank.count_by_key)
        self.assertIsNone(aliased_rank.grouped_items)

    def test_logical_simplification_preserves_nested_modulo(self) -> None:
        domain = CoordinateDomain.scalar(6)
        value = coordinate_axis_symbol(0)
        expression = 1 + sympy.Mod(
            4 * sympy.Mod(value, 3, evaluate=False), 6, evaluate=False
        )
        simplified = _simplify_logical_expression(
            expression,
            domain=domain,
            source_bounds=((0, 0, 6, 1),),
        )
        self.assertEqual(
            tuple(int(simplified.xreplace({value: index})) for index in range(6)),
            (1, 5, 3, 1, 5, 3),
        )

    def test_relation_cartesian_work_declines_before_expansion(self) -> None:
        scalar = CoordinateDomain.scalar(65)
        points = CoordinateRelation.point_map(
            scalar,
            scalar,
            tuple(((((0, value, value + 1, 1),), (value,))) for value in range(65)),
        )
        self.assertIsNone(points.then(points))

        square = CoordinateDomain((0, 1), ((0, 65), (1, 65)))
        value = CoordinateDomain.scalar(1, axis=2, kind="value")
        pieces = tuple(
            _CoordinateRelationPiece(bounds, ((2, 0, 1, 1),))
            for axis in (0, 1)
            for index in range(65)
            for bounds in (
                tuple(
                    (coordinate_axis, index, index + 1, 1)
                    if coordinate_axis == axis
                    else (coordinate_axis, 0, 65, 1)
                    for coordinate_axis in square.axis_order
                ),
            )
        )
        self.assertFalse(CoordinateRelation(square, value, pieces).has_total_source())

    def test_fixed_width_partition_keeps_empty_tail_counts(self) -> None:
        producer = CoordinateDomain((10,), ((10, 4),), identity=0)
        fine = CoordinateDomain((20,), ((20, 10),), kind="event", identity=7)
        task = coordinate_axis_symbol(10)
        publication = CoordinateRelation(
            producer,
            fine,
            (
                _CoordinateRelationPiece(
                    ((10, 0, 4, 1),), ((20, 2 * task, 2 * task + 2, 1),)
                ),
            ),
        )
        result = KeyPartition.from_fixed_width_publication(publication)
        assert result is not None
        partition, incidence = result
        self.assertEqual(partition.coarse_key_by_fine_key.target_domain.identity, 7)
        assert incidence.count_by_key is not None
        self.assertEqual(
            _materialize(incidence.count_by_key),
            (
                frozenset((1,)),
                frozenset((1,)),
                frozenset((1,)),
                frozenset((1,)),
                frozenset((0,)),
            ),
        )

    def test_separable_block_and_quotient_fibers_coarsen_exactly(self) -> None:
        producers = CoordinateDomain((0, 1), ((0, 2), (1, 4)), identity=0)
        fine_keys = CoordinateDomain(
            (10, 11), ((10, 4), (11, 2)), kind="event", identity=7
        )
        producer0 = coordinate_axis_symbol(0)
        producer1 = coordinate_axis_symbol(1)
        publication = CoordinateRelation(
            producers,
            fine_keys,
            (
                _CoordinateRelationPiece(
                    ((0, 0, 2, 1), (1, 0, 4, 1)),
                    (
                        (10, 2 * producer0, 2 * producer0 + 2, 1),
                        (11, FloorDiv(producer1, 2), FloorDiv(producer1, 2) + 1, 1),
                    ),
                ),
            ),
        )

        fine = Incidence.from_fibers(publication)
        assert fine.keys_by_item is not None
        assert fine.count_by_key is not None
        self.assertEqual(fine.count_by_key.value_bounds(), (2, 2))
        self.assertFalse(fine.keys_by_item.is_single_valued())

        result = KeyPartition.from_fixed_width_publication(publication)
        assert result is not None
        partition, coarse = result
        self.assertEqual(
            partition.coarse_key_by_fine_key.target_domain.axis_counts_items,
            ((10, sympy.Integer(2)), (11, sympy.Integer(2))),
        )
        assert coarse.keys_by_item is not None
        assert coarse.count_by_key is not None
        self.assertTrue(coarse.keys_by_item.is_single_valued())
        self.assertEqual(coarse.count_by_key.value_bounds(), (2, 2))

    def test_incidence_composition_rejects_cross_key_worker_fibers(self) -> None:
        keys = CoordinateDomain((0,), ((0, 2),), kind="event", identity=0)
        slots = CoordinateDomain((10,), ((10, 8),), identity=1)
        key = coordinate_axis_symbol(0)
        items = CoordinateRelation(
            keys,
            slots,
            (
                _CoordinateRelationPiece(
                    ((0, 0, 2, 1),), ((10, 2 * key, 2 * key + 2, 1),)
                ),
            ),
        )
        incidence = _incidence(items)
        self.assertIsNone(incidence.last_item_by_residue(2))
        owners = incidence.last_item_by_residue(4)
        self.assertIsNotNone(owners)
        assert owners is not None and owners.keys_by_item is not None
        self.assertEqual(
            _materialize(owners.keys_by_item)[:4],
            (frozenset((0,)), frozenset((0,)), frozenset((1,)), frozenset((1,))),
        )

    def test_symbolic_layout_and_relation_parameters_survive(self) -> None:
        extent = sympy.Symbol("extent", integer=True, positive=True)
        source = CoordinateDomain((10,), ((10, extent),), identity=0)
        target = CoordinateDomain((20,), ((20, extent),), identity=1)
        coordinate = coordinate_axis_symbol(10)
        relation = CoordinateRelation.point_map(
            source, target, (((((10, 0, extent, 1),), (coordinate,))),)
        )
        access = TileAccess(
            0,
            0,
            0,
            0,
            0,
            "load",
            "x",
            (extent,),
            (1,),
            0,
            (0,),
            (10,),
            (1,),
            (0,),
            (False,),
            False,
            True,
        )
        self.assertIn(extent, relation.source_domain.parameter_symbols)
        self.assertEqual(access.tensor_shape, (extent,))
        self.assertEqual(access, dataclasses.replace(access))

        tiled = TileAccess(
            1,
            1,
            0,
            0,
            0,
            "load",
            "x",
            (extent, 16),
            (16, 1),
            0,
            (1,),
            (10,),
            (1,),
            (0,),
            (False,),
            False,
            True,
            subscript_is_full_slice=(False,),
        )
        allocation = CoordinateDomain(
            (-1,), ((-1, 16 * extent),), kind="allocation", identity=0
        )
        linear_map = _symbolic_access_map(
            tiled,
            layout=_access_layout(tiled, None),
            source_domain=CoordinateDomain((10,), ((10, 1),), ((10, 16),)),
            allocation_domain=allocation,
        )
        assert linear_map is not None
        linear, _codec = linear_map
        self.assertIn(extent, linear.target_domain.parameter_symbols)
