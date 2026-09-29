"""In-band (LL) lowering of cross-rank tile readiness.

A symmetric allocation qualifies when one root writes every element exactly
once per launch through its local view and later roots read it only through
peer views.  The producer then also pushes each element, tagged with the launch
epoch, into every peer's mailbox, and a peer-view load becomes a poll of the
reader's own mailbox.  The covered dependencies need no readiness counters,
completion counters, or terminal drain.

Mailbox layout per allocation: ``[2 parity][world source][words]``. A uint64
word holds a 32-bit epoch and either one element, or two adjacent BF16 elements
when the opt-in multicast path can prove pair alignment. Parity double
buffering is WAR-safe because every rank reads every source each launch (so no
rank can run two launches ahead of a reader) and kernels on one stream never
overlap.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import TYPE_CHECKING

import sympy
import torch

from .tile_dependency import CoordinateRelation
from .tile_dependency import TileDependencyKind

if TYPE_CHECKING:
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
    world_size: int
    # First mailbox word of this allocation's [2][world][words] region.
    mailbox_offset: int
    producer_root: int
    elements_per_word: int = 1

    def words_per_source(self) -> int:
        return (self.numel + self.elements_per_word - 1) // self.elements_per_word

    def slot_words(self) -> int:
        return self.world_size * self.words_per_source()


@dataclasses.dataclass(frozen=True)
class DistributedLLPlan:
    """Per-config LL decisions consumed by memory-op and schedule codegen."""

    world_size: int
    allocations: tuple[LLAllocation, ...]
    stores: dict[AccessKey, LLAllocation]
    # Peer-view load -> (allocation, source rank named by the view).
    loads: dict[AccessKey, tuple[LLAllocation, int]]
    # Pair-packed loads whose index vector itself can be halved before polling.
    packed_loads: frozenset[AccessKey]
    covered_dependency_ids: frozenset[int]
    consumer_roots: frozenset[int]

    @property
    def mailbox_words(self) -> int:
        return sum(2 * allocation.slot_words() for allocation in self.allocations)


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
        block_id = store.subscript_affine_block_ids[position]
        axis = family.axis(block_id) if block_id is not None else None
        if (
            axis is None
            or store.subscript_is_scalar[position]
            or store.subscript_index_scales[position] != 1
            or store.subscript_offsets[position] != 0
            or not axis.canonical_origin
            or _as_int(axis.extent) != size
        ):
            return False
        used_axes.append(axis.block_id)
    # A root axis the subscript ignores would make tasks rewrite one element.
    return len(set(used_axes)) == len(used_axes) and set(used_axes) == set(
        family.logical_axis_order
    )


def _access_is_pair_aligned(
    access: TileAccess,
    shape: tuple[int, ...],
) -> bool:
    """Whether each access invocation consists of aligned adjacent pairs."""

    def even_base(value: object) -> bool:
        expression = sympy.expand(sympy.sympify(value))
        concrete = _as_int(expression)
        if concrete is not None:
            return concrete % 2 == 0
        from .compile_environment import CompileEnvironment
        from .host_function import HostFunction
        from .variable_origin import TileBeginOrigin

        env = CompileEnvironment.current()
        replacements: dict[sympy.Symbol, sympy.Integer] = {}
        for symbol in expression.free_symbols:
            origin_info = HostFunction.current().expr_to_origin.get(symbol)
            origin = origin_info.origin if origin_info is not None else None
            coefficient = expression.coeff(symbol)
            if not isinstance(origin, TileBeginOrigin) or not isinstance(
                coefficient, sympy.Integer
            ):
                return False
            block = env.block_sizes[env.canonical_block_id(origin.block_id)].size_hint()
            if int(coefficient) * block % 2:
                return False
            replacements[symbol] = sympy.Integer(0)
        remainder = _as_int(expression.xreplace(replacements))
        return remainder is not None and remainder % 2 == 0

    if (
        len(shape) != 1
        or shape[0] % 2
        or access.subscript_dims != (0,)
        or access.subscript_is_scalar != (False,)
        or access.subscript_index_scales != (1,)
    ):
        return False
    if access.subscript_is_full_slice == (True,):
        return True
    ranges = access.affine_subscript_ranges
    if ranges is not None and len(ranges) == 1:
        coefficients, begin, end, step = ranges[0]
        extent = _as_int(sympy.simplify(end - begin))
        concrete_coefficients = tuple(
            _as_int(coefficient) for _axis, coefficient, _divisor in coefficients
        )
        if (
            step == 1
            and even_base(begin)
            and extent is not None
            and extent % 2 == 0
            and all(
                coefficient is not None and coefficient % 2 == 0
                for coefficient in concrete_coefficients
            )
        ):
            return True
    if not access.subscript_offsets or access.subscript_offsets[0] is None:
        return False
    if not even_base(access.subscript_offsets[0]):
        return False
    if access.subscript_static_extents:
        static_extent = access.subscript_static_extents[0]
        if static_extent is not None:
            return static_extent % 2 == 0
    (block_id,) = access.subscript_affine_block_ids
    if block_id is None:
        return False
    from .compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    block = env.block_sizes[env.canonical_block_id(block_id)].size_hint()
    return block % 2 == 0


def plan_distributed_ll(
    graph: TileDependencyGraph,
    device_ir: DeviceIR,
    *,
    pack_bf16: bool = False,
) -> DistributedLLPlan | None:
    """Select symmetric allocations whose cross-rank readiness can be in-band."""
    by_allocation: dict[int, list[TileAccess]] = {}
    for access in graph.accesses:
        if access.owner_rank_relation is not None:
            by_allocation.setdefault(access.allocation_id, []).append(access)
    if not by_allocation:
        return None
    world_sizes = {
        access.owner_rank_relation.source_domain.size
        for accesses in by_allocation.values()
        for access in accesses
        if access.owner_rank_relation is not None
    }
    if len(world_sizes) != 1:
        return None
    (world_size,) = world_sizes
    world_size = int(world_size)

    allocations: list[LLAllocation] = []
    stores: dict[AccessKey, LLAllocation] = {}
    loads: dict[AccessKey, tuple[LLAllocation, int]] = {}
    packed_loads: set[AccessKey] = set()
    covered: set[int] = set()
    consumer_roots: set[int] = set()
    mailbox_offset = 0
    for allocation_id, accesses in sorted(by_allocation.items()):
        if any(access.is_atomic for access in accesses):
            continue
        store_accesses = [a for a in accesses if a.kind == "store"]
        if len(store_accesses) != 1:
            continue
        (store,) = store_accesses
        layout = _concrete_layout(store)
        if (
            layout is None
            or not _is_dense(*layout)
            or _owner_rank(store, world_size) is not None
            or not _writes_each_element_once(graph, store, layout[0])
        ):
            continue
        dtype = _node_dtype(device_ir, store)
        if dtype is None or dtype.is_complex or dtype.itemsize > 4:
            continue
        peer_loads = [
            (access, _owner_rank(access, world_size))
            for access in accesses
            if access.kind == "load"
        ]
        peer_loads = [(access, rank) for access, rank in peer_loads if rank is not None]
        if not peer_loads or any(
            access.root <= store.root
            or _concrete_layout(access) != layout
            or _node_dtype(device_ir, access) != dtype
            for access, _rank in peer_loads
        ):
            continue
        unconditional_sources = {
            rank
            for access, rank in peer_loads
            if all(
                graph.execution_sites[site].executes_unconditionally
                for site in graph.site_ids_by_access[access.access_id]
            )
        }
        if unconditional_sources != set(range(world_size)):
            continue
        load_ids = {access.access_id for access, _rank in peer_loads}
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
            continue
        allocation = LLAllocation(
            allocation_id=allocation_id,
            tensor_name=store.tensor_name,
            dtype=dtype,
            numel=math.prod(layout[0]),
            world_size=world_size,
            mailbox_offset=mailbox_offset,
            producer_root=store.root,
            elements_per_word=(
                2
                if pack_bf16
                and dtype == torch.bfloat16
                and _access_is_pair_aligned(store, layout[0])
                else 1
            ),
        )
        mailbox_offset += 2 * allocation.slot_words()
        allocations.append(allocation)
        stores[(store.graph_id, store.graph_node_index)] = allocation
        for access, rank in peer_loads:
            key = access.graph_id, access.graph_node_index
            loads[key] = (allocation, rank)
            if allocation.elements_per_word == 2 and _access_is_pair_aligned(
                access, layout[0]
            ):
                packed_loads.add(key)
            consumer_roots.add(access.root)
        covered.update(dependency_ids)
    if not allocations:
        return None
    return DistributedLLPlan(
        world_size=world_size,
        allocations=tuple(allocations),
        stores=stores,
        loads=loads,
        packed_loads=frozenset(packed_loads),
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
