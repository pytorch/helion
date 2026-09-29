"""In-band (LL) lowering of tile readiness.

A symmetric allocation qualifies when one root writes every element exactly
once per launch through its local view and later roots read it only through
peer views.  The producer then also pushes each element, tagged with the launch
epoch, into every peer's mailbox, and a peer-view load becomes a poll of the
reader's own mailbox.  The covered dependencies need no readiness counters,
completion counters, or terminal drain.

Mailbox layout per allocation: ``[2 parity][world source][numel]`` uint64 words
holding ``epoch << 32 | element bits``.  Parity double buffering is WAR-safe
because every rank reads every source each launch (so no rank can run two
launches ahead of a reader) and kernels on one stream never overlap.

A rank-local allocation with the same write-once producer, read only by later
roots, can use the same protocol through a launcher-owned ``[numel]`` mailbox
(``settings.local_ll``).  One slot suffices: launches on a stream never overlap,
so every read of a launch finishes before the next launch pushes.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import TYPE_CHECKING

import sympy
import torch

from .compile_environment import CompileEnvironment
from .tile_dependency import CoordinateRelation
from .tile_dependency import TileDependencyKind

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .device_ir import DeviceIR
    from .tile_dependency import TileAccess
    from .tile_dependency import TileDependencyGraph

# Stamped on memory-op FX nodes by tile-access analysis.  Graph copies made for
# codegen keep node meta, so codegen can map a node back to its TileAccess.
TILE_ACCESS_KEY_META = "helion_tile_access_key"

AccessKey = tuple[int, int]


@dataclasses.dataclass(frozen=True)
class LLAllocation:
    """One symmetric allocation whose readiness travels with its data."""

    allocation_id: int
    tensor_name: str | None
    dtype: torch.dtype
    numel: int
    # ``None`` for a rank-local allocation (one slot, no peers or parity).
    world_size: int | None
    # First word of this allocation's region: [2][world][numel] in the
    # symmetric mailbox, or [numel] in the local mailbox.
    mailbox_offset: int
    producer_root: int

    @property
    def is_local(self) -> bool:
        return self.world_size is None

    def slot_words(self) -> int:
        return self.numel if self.world_size is None else self.world_size * self.numel

    def mailbox_words(self) -> int:
        return self.numel if self.world_size is None else 2 * self.slot_words()


@dataclasses.dataclass(frozen=True)
class DistributedLLPlan:
    """Per-config LL decisions consumed by memory-op and schedule codegen."""

    # ``None`` when every allocation is rank-local.
    world_size: int | None
    allocations: tuple[LLAllocation, ...]
    stores: dict[AccessKey, LLAllocation]
    # LL load -> (allocation, source rank named by a peer view; ``None`` local).
    loads: dict[AccessKey, tuple[LLAllocation, int | None]]
    covered_dependency_ids: frozenset[int]
    consumer_roots: frozenset[int]

    @property
    def mailbox_words(self) -> int:
        """Words of the symmetric (peer-pushed) mailbox."""
        return sum(
            allocation.mailbox_words()
            for allocation in self.allocations
            if not allocation.is_local
        )

    @property
    def local_mailbox_words(self) -> int:
        return sum(
            allocation.mailbox_words()
            for allocation in self.allocations
            if allocation.is_local
        )


def _as_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, sympy.Expr) and value.is_Integer:
        return int(value)
    return None


def _concrete_layout(
    access: TileAccess,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    shape = tuple(_as_int(size) for size in access.tensor_shape)
    strides = tuple(_as_int(stride) for stride in access.tensor_strides)
    if (
        not access.layout_is_symbolically_exact
        or _as_int(access.storage_offset) != 0
        or any(size is None or size <= 0 for size in shape)
        or any(stride is None for stride in strides)
    ):
        return None
    return tuple(s for s in shape if s is not None), tuple(
        s for s in strides if s is not None
    )


def _is_dense(shape: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    expected = 1
    for size, stride in sorted(
        zip(shape, strides, strict=True), key=operator.itemgetter(1)
    ):
        if size != 1 and stride != expected:
            return False
        expected *= size
    return True


def _owner_rank(access: TileAccess, world_size: int) -> int | None:
    """Constant owner rank of a peer view; ``None`` for the local view."""
    relation = access.owner_rank_relation
    assert relation is not None
    if relation == CoordinateRelation.identity(
        relation.source_domain, relation.target_domain
    ):
        return None
    (piece,) = relation.pieces
    ((_axis, begin, _end, _step),) = piece.target_ranges
    rank = _as_int(begin)
    if rank is None or not 0 <= rank < world_size:
        raise AssertionError("symmetric peer view has no constant owner rank")
    return rank


def _node_dtype(device_ir: DeviceIR, access: TileAccess) -> torch.dtype | None:
    node = list(device_ir.graphs[access.graph_id].graph.nodes)[access.graph_node_index]
    tensor = node.args[0].meta.get("val") if node.args else None
    return tensor.dtype if isinstance(tensor, torch.Tensor) else None


def _writes_each_element_once(
    graph: TileDependencyGraph,
    store: TileAccess,
    shape: tuple[int, ...],
    config: Config,
) -> bool:
    """Whether the store's root tasks partition the whole tensor."""
    sites = graph.site_ids_by_access[store.access_id]
    if (
        store.has_explicit_mask
        or not sites
        or any(not graph.execution_sites[site].is_root for site in sites)
        or len(store.subscript_dims) != len(shape)
    ):
        return False
    family = graph.task_families[store.root]
    used_axes: list[int] = []
    for position, size in enumerate(shape):
        if store.subscript_is_full_slice[position]:
            continue
        dense_span = (
            store.subscript_dense_spans[position]
            if position < len(store.subscript_dense_spans)
            else None
        )
        # Lanes of a dense span are not tile-masked, so it needs whole tiles.
        whole_tiles = dense_span is not None
        if dense_span is not None:
            # ``tile.begin * scale + arange(block * scale)`` covers
            # [begin * scale, (begin + block) * scale) of the dimension.
            block_id, scale, offset = dense_span
        else:
            block_id = store.subscript_affine_block_ids[position]
            scale = store.subscript_index_scales[position]
            offset = store.subscript_offsets[position]
            if store.subscript_is_scalar[position] or scale != 1:
                return False
        axis = family.axis(block_id) if block_id is not None else None
        extent = _as_int(axis.extent) if axis is not None else None
        if (
            axis is None
            or extent is None
            or scale < 1
            or offset != 0
            or not axis.canonical_origin
            or extent * scale != size
        ):
            return False
        if whole_tiles:
            block = CompileEnvironment.current().block_sizes[axis.block_id]
            block_size = block.from_config(config)
            if not isinstance(block_size, int) or extent % block_size != 0:
                return False
        used_axes.append(axis.block_id)
    # A root axis the subscript ignores would make tasks rewrite one element.
    return len(set(used_axes)) == len(used_axes) and set(used_axes) == set(
        family.logical_axis_order
    )


@dataclasses.dataclass(frozen=True)
class _Candidate:
    store: TileAccess
    # (load, source rank named by a peer view; ``None`` for a local load).
    loads: tuple[tuple[TileAccess, int | None], ...]
    dtype: torch.dtype
    numel: int
    dependency_ids: frozenset[int]


def _write_once_store(
    graph: TileDependencyGraph,
    device_ir: DeviceIR,
    accesses: list[TileAccess],
    config: Config,
) -> tuple[TileAccess, tuple[tuple[int, ...], tuple[int, ...]], torch.dtype] | None:
    """The allocation's only store, if it writes each dense element once."""
    if any(access.is_atomic for access in accesses):
        return None
    store_accesses = [a for a in accesses if a.kind == "store"]
    if len(store_accesses) != 1:
        return None
    (store,) = store_accesses
    layout = _concrete_layout(store)
    if (
        layout is None
        or not _is_dense(*layout)
        or not _writes_each_element_once(graph, store, layout[0], config)
    ):
        return None
    dtype = _node_dtype(device_ir, store)
    # Words carry the value's bits; bool has no bitcast partner.
    if dtype is None or dtype.is_complex or dtype == torch.bool or dtype.itemsize > 4:
        return None
    return store, layout, dtype


def _loads_follow_store(
    device_ir: DeviceIR,
    store: TileAccess,
    layout: tuple[tuple[int, ...], tuple[int, ...]],
    dtype: torch.dtype,
    loads: list[TileAccess],
) -> bool:
    return bool(loads) and all(
        access.root > store.root
        and _concrete_layout(access) == layout
        and _node_dtype(device_ir, access) == dtype
        for access in loads
    )


def _covered_dependencies(
    graph: TileDependencyGraph,
    store: TileAccess,
    loads: list[TileAccess],
) -> frozenset[int] | None:
    """The hazards of ``loads``; ``None`` unless each is RAW on ``store``."""
    load_ids = {access.access_id for access in loads}
    dependency_ids = {
        dependency.dependency_id
        for edge in graph.edges
        for dependency in edge.access_dependencies
        if dependency.consumer_access_id in load_ids
        or dependency.producer_access_id in load_ids
    }
    if any(
        dependency.producer_access_id != store.access_id
        or dependency.kind is not TileDependencyKind.READ_AFTER_WRITE
        for edge in graph.edges
        for dependency in edge.access_dependencies
        if dependency.dependency_id in dependency_ids
    ):
        return None
    return frozenset(dependency_ids)


def _symmetric_candidate(
    graph: TileDependencyGraph,
    device_ir: DeviceIR,
    accesses: list[TileAccess],
    world_size: int,
    config: Config,
) -> _Candidate | None:
    written = _write_once_store(graph, device_ir, accesses, config)
    if written is None:
        return None
    store, layout, dtype = written
    if _owner_rank(store, world_size) is not None:
        return None
    peer_loads = [
        (access, rank)
        for access in accesses
        if access.kind == "load"
        and (rank := _owner_rank(access, world_size)) is not None
    ]
    if not _loads_follow_store(
        device_ir, store, layout, dtype, [access for access, _rank in peer_loads]
    ):
        return None
    unconditional_sources = {
        rank
        for access, rank in peer_loads
        if all(
            graph.execution_sites[site].executes_unconditionally
            for site in graph.site_ids_by_access[access.access_id]
        )
    }
    if unconditional_sources != set(range(world_size)):
        return None
    dependency_ids = _covered_dependencies(
        graph, store, [access for access, _rank in peer_loads]
    )
    if dependency_ids is None:
        return None
    return _Candidate(
        store=store,
        loads=tuple(peer_loads),
        dtype=dtype,
        numel=math.prod(layout[0]),
        dependency_ids=dependency_ids,
    )


def _local_candidate(
    graph: TileDependencyGraph,
    device_ir: DeviceIR,
    accesses: list[TileAccess],
    config: Config,
) -> _Candidate | None:
    written = _write_once_store(graph, device_ir, accesses, config)
    if written is None:
        return None
    store, layout, dtype = written
    loads = [access for access in accesses if access.kind == "load"]
    if not _loads_follow_store(device_ir, store, layout, dtype, loads):
        return None
    dependency_ids = _covered_dependencies(graph, store, loads)
    if dependency_ids is None:
        return None
    return _Candidate(
        store=store,
        loads=tuple((access, None) for access in loads),
        dtype=dtype,
        numel=math.prod(layout[0]),
        dependency_ids=dependency_ids,
    )


def _symmetric_world_size(
    by_allocation: dict[int, list[TileAccess]],
) -> int | None:
    world_sizes = {
        access.owner_rank_relation.source_domain.size
        for accesses in by_allocation.values()
        for access in accesses
        if access.owner_rank_relation is not None
    }
    return int(next(iter(world_sizes))) if len(world_sizes) == 1 else None


def plan_distributed_ll(
    graph: TileDependencyGraph,
    device_ir: DeviceIR,
    config: Config,
    *,
    symmetric: bool,
    local: bool,
) -> DistributedLLPlan | None:
    """Select allocations whose tile readiness can travel with their data."""
    symmetric_accesses: dict[int, list[TileAccess]] = {}
    local_accesses: dict[int, list[TileAccess]] = {}
    for access in graph.accesses:
        by_allocation = (
            local_accesses if access.owner_rank_relation is None else symmetric_accesses
        )
        by_allocation.setdefault(access.allocation_id, []).append(access)
    world_size = _symmetric_world_size(symmetric_accesses) if symmetric else None
    candidates: list[tuple[int, _Candidate, int | None]] = []
    if world_size is not None:
        for allocation_id, accesses in sorted(symmetric_accesses.items()):
            candidate = _symmetric_candidate(
                graph, device_ir, accesses, world_size, config
            )
            if candidate is not None:
                candidates.append((allocation_id, candidate, world_size))
    if local:
        for allocation_id, accesses in sorted(local_accesses.items()):
            candidate = _local_candidate(graph, device_ir, accesses, config)
            if candidate is not None:
                candidates.append((allocation_id, candidate, None))
    if not candidates:
        return None
    allocations: list[LLAllocation] = []
    stores: dict[AccessKey, LLAllocation] = {}
    loads: dict[AccessKey, tuple[LLAllocation, int | None]] = {}
    covered: set[int] = set()
    consumer_roots: set[int] = set()
    mailbox_offsets = {True: 0, False: 0}
    for allocation_id, candidate, candidate_world_size in candidates:
        is_local = candidate_world_size is None
        allocation = LLAllocation(
            allocation_id=allocation_id,
            tensor_name=candidate.store.tensor_name,
            dtype=candidate.dtype,
            numel=candidate.numel,
            world_size=candidate_world_size,
            mailbox_offset=mailbox_offsets[is_local],
            producer_root=candidate.store.root,
        )
        mailbox_offsets[is_local] += allocation.mailbox_words()
        allocations.append(allocation)
        stores[(candidate.store.graph_id, candidate.store.graph_node_index)] = (
            allocation
        )
        for access, rank in candidate.loads:
            loads[(access.graph_id, access.graph_node_index)] = (allocation, rank)
            consumer_roots.add(access.root)
        covered.update(candidate.dependency_ids)
    return DistributedLLPlan(
        world_size=world_size if mailbox_offsets[False] else None,
        allocations=tuple(allocations),
        stores=stores,
        loads=loads,
        covered_dependency_ids=frozenset(covered),
        consumer_roots=frozenset(consumer_roots),
    )


def without_ll_dependencies(
    graph: TileDependencyGraph,
    plan: DistributedLLPlan | None,
) -> TileDependencyGraph:
    """Drop the hazards whose readiness the LL mailbox words carry."""
    if plan is None:
        return graph
    edges = []
    for edge in graph.edges:
        kept = tuple(
            dependency
            for dependency in edge.access_dependencies
            if dependency.dependency_id not in plan.covered_dependency_ids
        )
        if kept:
            edges.append(dataclasses.replace(edge, access_dependencies=kept))
    return dataclasses.replace(graph, edges=tuple(edges))
