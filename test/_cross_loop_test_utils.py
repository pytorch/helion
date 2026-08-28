from __future__ import annotations

import dataclasses
import itertools
from typing import Literal

from helion._compiler.cross_loop_scheduler import EventContribution
from helion._compiler.cross_loop_scheduler import EventGraph
from helion._compiler.cross_loop_scheduler import WorkerSchedule
from helion._compiler.cross_loop_scheduler import (
    build_baseline_worker_schedule as _build_baseline_worker_schedule,
)
from helion._compiler.cross_loop_scheduler import (
    build_cross_loop_schedule as _build_cross_loop_schedule,
)
from helion._compiler.cross_loop_scheduler import (
    build_event_graph as _build_event_graph,
)
from helion._compiler.tile_dependency import LogicalDomain
from helion._compiler.tile_dependency import LogicalRelation
from helion._compiler.tile_dependency import TileAccess
from helion._compiler.tile_dependency import instantiate_root_domains
from helion._compiler.tile_dependency import instantiate_symbolic_dependencies
from helion._compiler.tile_dependency import logical_axis_symbol
from helion._compiler.tile_dependency import physical_traversal_relation


def _axis_geometry(
    root_domains: tuple[LogicalDomain, ...],
) -> dict[int, tuple[int, int]]:
    return {
        axis: (domain.axis_counts[axis], domain.block_sizes[axis])
        for domain in root_domains
        for axis in domain.axis_order
    }


def _identify_root_domains(
    root_domains: tuple[LogicalDomain, ...],
) -> tuple[LogicalDomain, ...]:
    return tuple(
        dataclasses.replace(domain, identity=root)
        for root, domain in enumerate(root_domains)
    )


def _default_root_traversals(
    root_domains: tuple[LogicalDomain, ...],
    physical_axis_orders: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[LogicalRelation, ...]:
    if physical_axis_orders is None:
        physical_axis_orders = tuple(domain.axis_order for domain in root_domains)
    return tuple(
        itertools.starmap(
            physical_traversal_relation,
            zip(
                root_domains,
                physical_axis_orders,
                strict=True,
            ),
        )
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
    relations = tuple(
        dependency.relation
        for dependency in instantiate_symbolic_dependencies(
            plan,
            axis_geometry=axis_geometry,
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
    dependencies = instantiate_symbolic_dependencies(
        plan,
        axis_geometry=axis_geometry,
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


def _configured_event_graph(
    graph,
    root_domains: tuple[LogicalDomain, ...],
    *,
    axis_geometry: dict[int, tuple[int, int]] | None = None,
    physical_axis_orders: tuple[tuple[int, ...], ...] | None = None,
    publishable_scope_ids: frozenset[int] | None = None,
) -> EventGraph:
    if axis_geometry is None:
        axis_geometry = _axis_geometry(root_domains)
    configured_domains = instantiate_root_domains(
        graph,
        axis_geometry=axis_geometry,
    )
    assert all(domain is not None for domain in configured_domains)
    configured_root_domains = tuple(
        domain for domain in configured_domains if domain is not None
    )
    return _build_event_graph(
        graph,
        root_domains=configured_root_domains,
        root_traversals=_default_root_traversals(
            configured_root_domains,
            physical_axis_orders,
        ),
        axis_geometry=axis_geometry,
        publishable_scope_ids=publishable_scope_ids,
    )


def build_baseline_worker_schedule(
    root_domains: tuple[LogicalDomain, ...],
    worker_count: int,
    *,
    root_traversals: tuple[LogicalRelation, ...] | None = None,
    physical_axis_orders: tuple[tuple[int, ...], ...] | None = None,
) -> WorkerSchedule:
    root_domains = _identify_root_domains(root_domains)
    if root_traversals is None:
        root_traversals = _default_root_traversals(
            root_domains,
            physical_axis_orders,
        )
    return _build_baseline_worker_schedule(
        root_domains,
        root_traversals,
        worker_count,
    )


def build_cross_loop_schedule(
    *,
    root_domains: tuple[LogicalDomain, ...],
    root_traversals: tuple[LogicalRelation, ...] | None = None,
    physical_axis_orders: tuple[tuple[int, ...], ...] | None = None,
    **kwargs,
):
    root_domains = _identify_root_domains(root_domains)
    if root_traversals is None:
        root_traversals = _default_root_traversals(
            root_domains,
            physical_axis_orders,
        )
    return _build_cross_loop_schedule(
        root_domains=root_domains,
        root_traversals=root_traversals,
        **kwargs,
    )


def _one_dimensional_task_range(
    domain: LogicalDomain,
    begin: int,
    count: int,
) -> LogicalRelation:
    (axis,) = domain.axis_order
    ordinal_domain = LogicalDomain((axis,), ((axis, count),), kind="worker")
    return LogicalRelation.point_map(
        ordinal_domain,
        domain,
        (
            (
                ((axis, 0, count, 1),),
                (logical_axis_symbol(axis) + begin,),
            ),
        ),
    )


def _expected_arrivals(
    key_domain: LogicalDomain,
    contributions: tuple[EventContribution, ...],
) -> tuple[int, ...]:
    result = [0] * key_domain.size
    for contribution in contributions:
        cardinality = contribution.predecessors.fiber_cardinality()
        assert cardinality is not None
        for key, values in enumerate(cardinality.materialize()):
            assert len(values) == 1
            result[key] += next(iter(values))
    return tuple(result)


def _event_contribution_from_publication(
    producer_root: int,
    publication: LogicalRelation,
    producer_scope_id: int | None = None,
) -> EventContribution:
    predecessors = publication.inverse()
    assert predecessors is not None
    return EventContribution(
        producer_root=producer_root,
        producer_scope_id=producer_scope_id,
        predecessors=predecessors,
    )


def _publication(contribution: EventContribution) -> LogicalRelation:
    publication = contribution.producer_to_keys
    assert publication is not None
    return publication
