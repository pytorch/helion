"""Closed static pointwise islands with one complete FP32 row per warp lane.

Planning owns no emission or arithmetic substitutions. Sparse selectors retain
their original source recipes and zero-add trees; every external consumer sees
the original typed panel through one completed shared publication.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import view_ops
from .chained_pointwise_residency import _shape
from .chained_root_native import joined_root
from .chained_root_native import native_facts
from .chained_sparse_reduction import SparseReduction
from .chained_sparse_reduction import plan_sparse_reduction
from .chained_store_expression import _node_facts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from .chained_matmul import ChainedMatmulPlan


KEY = "cute_chained_pointwise_schedule"
MODES = ("shared", "warp_rows", "warp_rows_parallel")
# This is a bound on explicitly allocated FP32 payload slots per participating
# lane, not a prediction of the compiler's final physical register count.
MAX_ROW_REGISTER_SLOTS = 160


@dataclass(frozen=True)
class RowRegisterEvent:
    node: Node
    reduction: SparseReduction | None
    reads: tuple[Node, ...]
    offset: int
    size: int
    last_use: int
    literal_columns: bool = False


@dataclass(frozen=True)
class RowRegisterIsland:
    output: Node
    shape: tuple[int, int]
    first_collective: Node
    stage: int
    reductions: tuple[Node, ...]
    members: tuple[Node, ...]
    events: tuple[RowRegisterEvent, ...]
    external_reads: tuple[Node, ...]
    register_slots: int
    peak_live_slots: int
    name: str

    @property
    def shared_bytes(self) -> int:
        return (4 * self.shape[0] * self.shape[1] + 127) // 128 * 128


@dataclass(frozen=True)
class RowRegisterPlan:
    islands: tuple[RowRegisterIsland, ...]
    facts: tuple[object, ...]
    selection: tuple[object, ...] = ()

    @property
    def shared_bytes(self) -> int:
        return sum(island.shared_bytes for island in self.islands)

    def check(self, plan: ChainedMatmulPlan) -> None:
        from .chained_matmul import _UnsupportedChain

        if (
            plan.region is None
            or self.facts != _facts(plan)
            or self.selection != tuple(_record(item) for item in self.islands)
        ):
            raise _UnsupportedChain("row-register island graph or schedule changed")


def _record(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            (item.name, _record(getattr(value, item.name))) for item in fields(value)
        )
    if isinstance(value, tuple):
        return tuple(_record(item) for item in value)
    return value


def _facts(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    assert plan.region is not None
    return (
        plan.region,
        plan.dots,
        plan.store,
        plan.axes,
        plan.shapes,
        plan.threads,
        plan.strategy,
        native_facts(plan),
        plan.loop,
        plan.contraction_groups,
        plan.pointwise_cache,
        plan.root_vector_cache,
        plan.preparation_pipeline,
        plan.initialized_accumulator,
        plan.late_rhs_reuse,
        plan.root_typed_operands,
        plan.k_schedule,
        plan.row_panel_cache,
        plan.row_subwarp_packing,
        plan.row_frontier,
        _node_facts(tuple(plan.region.graph.nodes)),
    )


def _pointwise(node: Node) -> bool:
    """Closed scalar arithmetic plus coordinate-only singleton views."""
    if node.op != "call_function":
        return False
    if isinstance(node.target, torch._ops.OpOverload):
        return torch.Tag.pointwise in node.target.tags or node.target in (
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.expand.default,
        )
    if node.target in (_tracing_ops._new_var, _tracing_ops._mask_to):
        return True
    return (
        node.target is view_ops.subscript
        and len(node.args) == 2
        and isinstance(node.args[1], (tuple, list))
        and all(item is None or item == slice(None) for item in node.args[1])
    )


def _panel(node: Node) -> tuple[int, int] | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        return None
    shape = _shape(node)
    if len(shape) != 2 or any(
        type(size) is not int or not 1 < size <= 32 for size in shape
    ):
        return None
    return shape[0], shape[1]


def has_row_register_candidate(
    graphs: Sequence[GraphInfo], env: CompileEnvironment
) -> bool:
    """Bound discovery to repeated full static small FP32 panel reductions.

    Closure, sparse selector recipes and the exact live payload are proved only
    after configuration binding. Tunable/unresolved extents grant no fact here.
    """
    from ..compile_environment import FixedBlockSizeSource
    from ..compile_environment import ReductionLoopBlockSizeSource
    from .chained_collectives import classify_collective
    from .chained_matmul import _classify_chained_graph

    graph = _classify_chained_graph(graphs)
    if graph is None or graph.loop is not None:
        return False

    def extent(value: object) -> int | None:
        if type(value) is int:
            return value
        if not isinstance(value, torch.SymInt):
            return None
        block_id = env.get_block_id(value)
        if block_id is None:
            return None
        block = env.block_sizes[env.canonical_block_id(block_id)]
        source = block.block_size_source
        if isinstance(source, FixedBlockSizeSource):
            return source.value if type(source.value) is int else None
        if isinstance(source, ReductionLoopBlockSizeSource):
            return env._search_invariant_extent_for_numel_constraint(block)
        return None

    shapes: Counter[tuple[int, ...]] = Counter()
    for node in graph.region.reductions:
        operation = classify_collective(node)
        if (
            operation is None
            or operation.kind != "sum"
            or operation.source.meta["val"].dtype != torch.float32
        ):
            continue
        shape = tuple(extent(size) for size in operation.source.meta["val"].shape)
        if len(shape) == 2 and all(
            size is not None and 1 < size <= 32 for size in shape
        ):
            shapes[tuple(size for size in shape if size is not None)] += 1
    return any(count >= 2 for count in shapes.values())


def _allocate_events(
    nodes: tuple[Node, ...],
    reductions: dict[Node, SparseReduction],
    reads: dict[Node, tuple[Node, ...]],
    width: int,
) -> tuple[tuple[RowRegisterEvent, ...], int, int]:
    """No output overwrites a source at the same read/publish event."""
    last = {node: index for index, node in enumerate(nodes)}
    for index, node in enumerate(nodes):
        for source in reads[node]:
            last[source] = max(last[source], index)
    last[nodes[-1]] = len(nodes)  # the final shared publication is a real read
    active: list[RowRegisterEvent] = []
    result: list[RowRegisterEvent] = []
    high_water = peak_live = 0
    for index, node in enumerate(nodes):
        active = [event for event in active if event.last_use >= index]
        proof = reductions.get(node)
        size = 1 if proof is not None and proof.axis == 1 else width
        offset = 0
        for event in sorted(active, key=lambda item: item.offset):
            if offset + size <= event.offset:
                break
            offset = event.offset + event.size
        event = RowRegisterEvent(node, proof, reads[node], offset, size, last[node])
        active.append(event)
        result.append(event)
        high_water = max(high_water, offset + size)
        peak_live = max(peak_live, sum(item.size for item in active))
    return tuple(result), high_water, peak_live


def plan_row_registers(plan: ChainedMatmulPlan) -> RowRegisterPlan:
    """Find exclusive static islands without changing any graph or allocation."""
    region = plan.region
    if (
        region is None
        or not joined_root(plan)
        or plan.loop is not None
        or plan.contraction_groups is not None
        or plan.pointwise_cache is not None
        or plan.initialized_accumulator is not None
        or plan.late_rhs_reuse is not None
        or plan.root_typed_operands is not None
        or plan.k_schedule is not None
        or plan.threads < 32
    ):
        return RowRegisterPlan((), () if region is None else _facts(plan))
    nodes = region.nodes
    positions = {node: index for index, node in enumerate(nodes)}
    boundaries = {*plan.dots, *region.scans}
    sparse = {
        node: proof
        for node in region.reductions
        if (proof := plan_sparse_reduction(plan, node)) is not None
        and _panel(proof.source) is not None
    }
    # Every collective, including unsupported ones, remains visible to closure.
    influence: dict[Node, set[Node]] = {}
    ancestors: dict[Node, set[Node]] = {}
    for node in nodes:
        parents = () if node in boundaries else node.all_input_nodes
        ancestors[node] = {node}.union(*(ancestors[parent] for parent in parents))
        influence[node] = set().union(*(influence[parent] for parent in parents))
        if node in region.reductions:
            influence[node].add(node)
    candidates = sorted(
        (
            node
            for node in nodes
            if len(influence[node]) >= 2 and _panel(node) is not None
        ),
        key=lambda node: (len(influence[node]), positions[node]),
        reverse=True,
    )
    islands: list[RowRegisterIsland] = []
    occupied: set[Node] = set()
    dot_positions = [positions[node] for node in plan.dots]
    for output in candidates:
        reductions = influence[output]
        # A fresh root store is already required by ordinary root admission.
        # Keep this movement proof local as well: neither immutable image
        # initialization nor a later recipe may cross an effect/publication.
        if any(
            node.op == "call_function"
            and (node.target is memory_ops.store or node.is_impure())
            for node in nodes[: positions[output]]
        ):
            continue
        shape = _panel(output)
        assert shape is not None
        if reductions & occupied or any(
            node not in sparse or _panel(sparse[node].source) != shape
            for node in reductions
        ):
            continue
        # Do not salvage an earlier prefix when a later selector would extend
        # the same component across an unproved use/fanout. The first version
        # only publishes completed selection islands at their original exit.
        if any(
            output in ancestors[node]
            for node in region.reductions
            if node not in reductions
        ):
            continue
        members = {node for node in ancestors[output] if influence[node] & reductions}
        if any(
            (node not in reductions and not _pointwise(node))
            or (node is not output and any(user not in members for user in node.users))
            for node in members
        ):
            continue
        first = min(reductions, key=positions.__getitem__)
        stage = bisect_left(dot_positions, positions[first])
        if bisect_left(dot_positions, positions[output]) != stage:
            continue
        images = {output, *(sparse[node].selected for node in reductions)}
        panels: set[Node] = set()
        if plan.row_panel_cache:
            # The original closure/readiness proof still owns every dependency.
            # Only immutable, reduction-independent internal panels are added.
            loads = Counter(
                node.args[0]
                for node in ancestors[output]
                if node.op == "call_function"
                and node.target is memory_ops.load
                and node.args[0] in positions
                and node.args[0].op == "call_function"
                and node.args[0] not in boundaries
                and not influence[node.args[0]]
                and _panel(node.args[0]) == shape
            )
            panels = {source for source, count in loads.items() if count > 1}
            images.update(panels)
        if any(_panel(node) != shape for node in images):
            continue
        event_nodes = tuple(node for node in nodes if node in images | reductions)
        # All selected sources are explicit versioned images. Other values can
        # depend on them only through original scalar arithmetic and broadcasts.
        event_set = set(event_nodes)
        reads: dict[Node, tuple[Node, ...]] = {}
        external: set[Node] = set()
        legal = True
        for event in event_nodes:
            if event in boundaries:
                external.add(event)
                reads[event] = ()
                continue
            pending = list(event.all_input_nodes)
            visited: set[Node] = set()
            inputs: set[Node] = set()
            while pending:
                node = pending.pop()
                if node in visited:
                    continue
                visited.add(node)
                if node in event_set:
                    inputs.add(node)
                elif node in boundaries:
                    external.add(node)
                elif node in region.reductions:
                    legal = False
                else:
                    pending.extend(node.all_input_nodes)
            reads[event] = tuple(node for node in event_nodes if node in inputs)
        if not legal or any(positions[node] >= positions[first] for node in external):
            continue
        events, slots, live = _allocate_events(event_nodes, sparse, reads, shape[1])
        if panels:
            events = tuple(
                replace(event, literal_columns=event.node in panels) for event in events
            )
        if (
            slots > MAX_ROW_REGISTER_SLOTS
            or not events
            or events[-1].node is not output
        ):
            continue
        islands.append(
            RowRegisterIsland(
                output,
                shape,
                first,
                stage,
                tuple(node for node in nodes if node in reductions),
                tuple(node for node in nodes if node in members),
                events,
                tuple(node for node in nodes if node in external),
                slots,
                live,
                f"chain_row_register_{len(islands)}",
            )
        )
        occupied.update(reductions)
    if plan.row_panel_cache and not any(
        event.literal_columns for island in islands for event in island.events
    ):
        return RowRegisterPlan((), _facts(plan))
    ordered = tuple(sorted(islands, key=lambda item: positions[item.first_collective]))
    return RowRegisterPlan(
        ordered, _facts(plan), tuple(_record(item) for item in ordered)
    )
