"""Typed pointwise residency for an admitted computed-contraction region.

This policy ranks typed caches by saved arithmetic per aligned byte. Its default
keeps independent ancestry; opt-in nesting accounts only additional work removed
from a selected parent's publication, after paying for the child publication.
The caller owns the byte budget, schedule admission, and host-storage alias
proofs. All existing contraction/collective lifetimes remain conservative.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from dataclasses import replace
from fractions import Fraction
import math
from operator import itemgetter
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ... import exc
from ...language import memory_ops
from ...language import view_ops
from ..compile_environment import CompileEnvironment
from .chained_execution import ChainedExecution

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chained_cache_layout import PointwiseCacheLayouts
    from .chained_matmul import ChainedMatmulPlan
    from .chained_scratch_layout import ScratchLayouts


_DTYPES = {
    torch.float32: "cutlass.Float32",
    torch.float16: "cutlass.Float16",
    torch.bfloat16: "cutlass.BFloat16",
}
_EXPENSIVE = {
    "aten::exp",
    "aten::exp2",
    "aten::log",
    "aten::log2",
    "aten::sigmoid",
    "aten::tanh",
    "aten::rsqrt",
    "aten::sqrt",
    "aten::div",
}


@dataclass(frozen=True)
class PointwiseCacheEntry:
    node: Node
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    first_stage: int
    last_stage: int
    byte_size: int
    allocated_bytes: int
    dependencies: tuple[Node, ...]
    producer_uses: int
    estimated_saved_operations: int


@dataclass(frozen=True)
class PointwiseCachePlan:
    entries: tuple[PointwiseCacheEntry, ...] = ()

    @property
    def shared_bytes(self) -> int:
        return sum(entry.allocated_bytes for entry in self.entries)


def _shape(node: Node) -> tuple[int, ...]:
    from .chained_matmul import _shape as tile_shape

    if CompileEnvironment.has_current():
        return tile_shape(node)
    # Static FX fixtures need no compiler context. Real symbolic shapes are
    # resolved only under the caller's active device-function configuration.
    return tuple(cast("torch.Tensor", node.meta["val"]).shape)


def _pointwise(node: Node) -> bool:
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.pointwise in node.target.tags
    )


def _cost(node: Node) -> int:
    if not _pointwise(node):
        return 0
    target = cast("torch._ops.OpOverload", node.target)
    return 8 if target._schema.name in _EXPENSIVE else 1


def _arithmetic(node: Node) -> bool:
    value = node.meta.get("val")
    return (
        _pointwise(node)
        and isinstance(value, torch.Tensor)
        and value.dtype in _DTYPES
        and node.target
        not in (
            torch.ops.prims.convert_element_type.default,
            torch.ops.aten.where.self,
        )
    )


def _boundaries(plan: ChainedMatmulPlan) -> set[Node]:
    assert plan.region is not None
    return {
        *plan.dots,
        *plan.scans,
        *plan.region.scans,
        *plan.region.reductions,
        *(carry.input for carry in plan.region.carries),
    }


def _ancestors(node: Node, boundaries: set[Node]) -> tuple[set[Node], set[Node]]:
    pending, visited, reads = [node], set(), set()
    while pending:
        current = pending.pop()
        if current in boundaries:
            reads.add(current)
        elif current not in visited:
            visited.add(current)
            pending.extend(current.all_input_nodes)
    return visited, reads


def _producers(plan: ChainedMatmulPlan) -> tuple[tuple[int, Node], ...]:
    """Enumerate expressions actually emitted, including explicit C seeds."""
    roots: list[tuple[int, Node]] = []
    if plan.contraction_groups is None:
        for stage, dot in enumerate(plan.dots):
            roots.extend(
                (stage, value) for value in dot.args[:3] if isinstance(value, Node)
            )
    else:
        for group in plan.contraction_groups:
            first = group.stages[0]
            # A common-left producer is evaluated once for the whole group.
            left = plan.dots[first].args[1 if group.geometries[0].transpose else 0]
            assert isinstance(left, Node)
            roots.append((first, left))
            for member, geometry in zip(group.stages, group.geometries, strict=True):
                right = plan.dots[member].args[0 if geometry.transpose else 1]
                assert isinstance(right, Node)
                roots.append((first, right))
                accumulator = plan.dots[member].args[2]
                if isinstance(accumulator, Node):
                    roots.append((first, accumulator))
    return tuple(roots)


def _finite_consumers(
    node: Node, boundaries: set[Node], *, within: frozenset[Node] | None = None
) -> bool:
    """Require views with proven finite coordinate maps before rebinding a tile.

    Inlining exp outside a tile yields exp(0), while a materialized boundary read
    outside its allocation yields zero. Arbitrary internal gathers therefore
    need a separate coordinate proof; do not silently change those semantics.
    ``within`` restricts the proof to uses actually redirected by a role cut.
    Unchanged producer-side uses must not be mistaken for image consumers.
    """
    pending, seen = list(node.users), set()
    while pending:
        user = pending.pop()
        if within is not None and user not in within:
            continue
        if user in boundaries or user in seen:
            continue
        seen.add(user)
        if user.target is memory_ops.load:
            return False
        if user.target is view_ops.subscript and any(
            selector is not None and selector != slice(None)
            for selector in cast("tuple[object, ...]", user.args[1])
        ):
            return False
        if user.target is torch.ops.aten.view.dtype:
            return False
        pending.extend(user.users)
    return True


def _candidate_ancestry(
    node: Node, boundaries: set[Node], written: set[object]
) -> tuple[set[Node], set[Node]] | None:
    value = node.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim not in (1, 2)
        or value.dtype not in _DTYPES
        or any(isinstance(size, int) and size == 0 for size in value.shape)
        or not _pointwise(node)
        or not _finite_consumers(node, boundaries)
    ):
        return None
    ancestors, dependencies = _ancestors(node, boundaries)
    if (
        any(item.target is torch.ops.aten.view.dtype for item in ancestors)
        or not any(_arithmetic(item) for item in ancestors)
        or not (
            dependencies
            or any(
                item.target is memory_ops.load
                or item.op == "placeholder"
                and isinstance(item.meta.get("val"), torch.Tensor)
                for item in ancestors
            )
        )
        or any(
            item.target is memory_ops.load and item.args[0] in written
            for item in ancestors
        )
    ):
        return None
    return ancestors, dependencies


def _uses(
    producers: tuple[tuple[int, Node], ...], boundaries: set[Node]
) -> dict[Node, list[tuple[int, int]]]:
    result: dict[Node, list[tuple[int, int]]] = {}
    for producer, (stage, root) in enumerate(producers):
        ancestors, _ = _ancestors(root, boundaries)
        for node in ancestors:
            result.setdefault(node, []).append((stage, producer))
    return result


def has_pointwise_residency_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """Structural search fact before physical tiles/groups/budgets are chosen."""
    from .chained_matmul import _classify_chained_graph

    graph = _classify_chained_graph(graphs)
    if graph is None:
        return False
    boundaries = {
        *graph.dots,
        *graph.region.scans,
        *graph.region.reductions,
        *(carry.input for carry in graph.region.carries),
    }
    producers = tuple(
        (stage, value)
        for stage, dot in enumerate(graph.dots)
        for value in dot.args[:3]
        if isinstance(value, Node)
    )
    written: set[object] = {node.args[0] for node in graph.region.stores}
    return any(
        len({stage for stage, _ in consumers}) >= 2
        and _candidate_ancestry(node, boundaries, written) is not None
        for node, consumers in _uses(producers, boundaries).items()
    )


def plan_pointwise_cache(
    plan: ChainedMatmulPlan,
    budget_bytes: int,
    *,
    max_entries: int = 1,
    nested: bool = False,
) -> PointwiseCachePlan:
    """Select independent typed boundaries under one aligned byte budget.

    Reuse is counted across physical producer groups, not logical dots merged
    into one issue. Candidate ancestry stops at resident boundaries. The first
    consumer is a dot/seed producer; earlier collective uses remain unmodified.
    One entry preserves the original winner. Larger bounds greedily follow the
    same ranking, rejecting shared non-boundary ancestry, including host reads.
    Entries retain ranking order; their first-stage metadata orders publication.
    """
    if type(max_entries) is not int or max_entries not in (1, 2, 4):
        raise ValueError("pointwise cache max_entries must be 1, 2, or 4")
    if type(nested) is not bool or (nested and max_entries < 2):
        raise ValueError(
            "nested pointwise caches require bool policy and max_entries >= 2"
        )
    if budget_bytes < 128 or plan.region is None:
        return PointwiseCachePlan()
    boundaries = _boundaries(plan)
    positions = {node: index for index, node in enumerate(plan.region.nodes)}
    uses = _uses(_producers(plan), boundaries)
    written: set[object] = {node.args[0] for node in plan.region.stores}
    choices: list[tuple[Fraction, int, int, PointwiseCacheEntry, set[Node]]] = []
    for node in plan.region.nodes:
        consumers = uses.get(node, ())
        stages = {stage for stage, _ in consumers}
        if len(stages) < 2:
            continue
        ancestry = _candidate_ancestry(node, boundaries, written)
        if ancestry is None:
            continue
        ancestors, dependencies = ancestry
        value = cast("torch.Tensor", node.meta["val"])
        shape = _shape(node)
        elements = math.prod(shape)
        if elements <= 0:
            continue
        byte_size = elements * value.element_size()
        aligned = (byte_size + 127) // 128 * 128
        if aligned > budget_bytes:
            continue
        first, last = min(stages), max(stages)
        # Storage is independent and never recycled, but expose a truthful
        # final expression-use stage, including collectives and epilogues.
        pending, visited = list(node.users), set()
        dot_stages = {dot: stage for stage, dot in enumerate(plan.dots)}
        dot_positions = [positions[dot] for dot in plan.dots]
        while pending:
            user = pending.pop()
            if user in visited:
                continue
            visited.add(user)
            if user in dot_stages:
                last = max(last, dot_stages[user])
            elif user in boundaries:
                last = max(last, bisect_right(dot_positions, positions[user]))
            elif user.op == "output" or user.target is memory_ops.store:
                last = len(plan.dots)
            else:
                pending.extend(user.users)
        carries = {carry.input for carry in plan.region.carries}
        if any(
            dep not in carries and positions[dep] >= positions[plan.dots[first]]
            for dep in dependencies
        ):
            continue
        # Counts are a deterministic selection proxy, not measured latency.
        # Broadcasting may repeat a scalar within one producer; counting it
        # once per output coordinate is a conservative, shape-general policy.
        saved = (len(consumers) - 1) * elements * sum(_cost(item) for item in ancestors)
        entry = PointwiseCacheEntry(
            node=node,
            name="chain_pointwise_cache_0",
            shape=shape,
            dtype=value.dtype,
            first_stage=first,
            last_stage=last,
            byte_size=byte_size,
            allocated_bytes=aligned,
            dependencies=tuple(
                item for item in plan.region.nodes if item in dependencies
            ),
            producer_uses=len(consumers),
            estimated_saved_operations=saved,
        )
        choices.append(
            (Fraction(saved, aligned), saved, positions[node], entry, ancestors)
        )
    if not choices:
        return PointwiseCachePlan()
    # Later exact typed boundaries win a complete tie, retaining source casts.
    if max_entries == 1:
        selected = max(choices, key=itemgetter(slice(3)))[3]
        return PointwiseCachePlan((selected,))
    if nested:
        return _nested_cache_plan(
            choices, boundaries, positions, budget_bytes, max_entries
        )
    entries: list[PointwiseCacheEntry] = []
    occupied: set[Node] = set()
    remaining = budget_bytes
    for _, _, _, entry, ancestors in sorted(
        choices, key=itemgetter(slice(3)), reverse=True
    ):
        if entry.allocated_bytes > remaining or not occupied.isdisjoint(ancestors):
            continue
        entries.append(replace(entry, name=f"chain_pointwise_cache_{len(entries)}"))
        occupied.update(ancestors)
        remaining -= entry.allocated_bytes
        if len(entries) == max_entries:
            break
    return PointwiseCachePlan(tuple(entries))


def _publication_layers(
    entries: Sequence[PointwiseCacheEntry],
) -> tuple[tuple[PointwiseCacheEntry, ...], ...]:
    pending = list(entries)
    layers = []
    while pending:
        nodes = {entry.node for entry in pending}
        layer = tuple(
            entry for entry in pending if nodes.isdisjoint(entry.dependencies)
        )
        if not layer:
            raise ValueError("cyclic pointwise cache dependencies")
        layers.append(layer)
        emitted = {entry.node for entry in layer}
        pending = [entry for entry in pending if entry.node not in emitted]
    return tuple(layers)


def _nested_cache_plan(
    choices: list[tuple[Fraction, int, int, PointwiseCacheEntry, set[Node]]],
    boundaries: set[Node],
    positions: dict[Node, int],
    budget: int,
    limit: int,
) -> PointwiseCachePlan:
    selected: list[PointwiseCacheEntry] = []
    selected_ancestry: dict[Node, set[Node]] = {}
    remaining = list(choices)
    while remaining and len(selected) < limit:
        ranked = []
        for _, saved, position, entry, ancestors in remaining:
            if entry.allocated_bytes > budget:
                continue
            parents = [
                item
                for item in selected
                if not ancestors.isdisjoint(selected_ancestry[item.node])
            ]
            if parents:
                if any(
                    entry.node not in selected_ancestry[parent.node]
                    or len(entry.shape) >= len(parent.shape)
                    or entry.first_stage != parent.first_stage
                    for parent in parents
                ):
                    continue
                # Only account work still performed by selected cache producers.
                # The old standalone consumer score overlaps their prior savings.
                active = boundaries | selected_ancestry.keys()
                child_work, _ = _ancestors(entry.node, active)
                cost = sum(_cost(node) for node in child_work)
                saved = -math.prod(entry.shape) * cost
                for parent in parents:
                    work, _ = _ancestors(parent.node, active - {parent.node})
                    if entry.node in work:
                        saved += math.prod(parent.shape) * cost
                if saved <= 0:
                    continue
            ranked.append(
                (
                    Fraction(saved, entry.allocated_bytes),
                    saved,
                    position,
                    entry,
                    ancestors,
                )
            )
        if not ranked:
            break
        _, saved, _, chosen, ancestors = max(ranked, key=itemgetter(slice(3)))
        selected.append(
            replace(
                chosen,
                name=f"chain_pointwise_cache_{len(selected)}",
                estimated_saved_operations=saved,
            )
        )
        selected_ancestry[chosen.node] = ancestors
        budget -= chosen.allocated_bytes
        remaining = [item for item in remaining if item[3].node is not chosen.node]
    nodes = {entry.node for entry in selected}
    rebound = []
    for entry in selected:
        _, reads = _ancestors(entry.node, boundaries | (nodes - {entry.node}))
        rebound.append(
            replace(entry, dependencies=tuple(sorted(reads, key=positions.__getitem__)))
        )
    return PointwiseCachePlan(
        tuple(entry for layer in _publication_layers(rebound) for entry in layer)
    )


def allocate_pointwise_cache(
    cache: PointwiseCachePlan,
    scratch: ScratchLayouts,
    layouts: PointwiseCacheLayouts | None = None,
) -> list[str]:
    """Independent typed storage; call once outside the lexical loop."""
    if scratch.read_buffers is not None:
        scratch.read_buffers = scratch.read_buffers | frozenset(
            entry.name for entry in cache.entries
        )
    lines = []
    for entry in cache.entries:
        layout = scratch.layout(entry.name, entry.shape, entry.dtype)
        if layouts is not None:
            layout = layouts.layout(entry.shape, entry.dtype, layout)
        lines.append(
            f"{entry.name} = cute.make_tensor(cute.arch.alloc_smem({_DTYPES[entry.dtype]}, {math.prod(entry.shape)}, alignment=128), {layout})"
        )
    return lines


def validate_pointwise_cache(plan: ChainedMatmulPlan | None, budget: object) -> None:
    """Reject a requested budget when a competing or ineffective path won."""
    if type(budget) is int and budget == 0:
        return
    if (
        type(budget) is not int
        or budget <= 0
        or plan is None
        or plan.pointwise_cache is None
        or not plan.pointwise_cache.entries
    ):
        raise exc.BackendUnsupported(
            "cute",
            "pointwise cache requires a positive integer budget and an effective typed residency plan",
        )


def emit_pointwise_cache_before(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    cache: PointwiseCachePlan,
    stage: int,
    *,
    execution: ChainedExecution | None = None,
    selected: frozenset[Node] | None = None,
) -> list[str]:
    """Fill after preceding collectives, then publish before producer ``stage``.

    Emit this code in the loop body for loop plans. No cached binding is exposed
    until its execution-team publication barrier has been appended. Internal source
    masks/casts use ordinary expression emission; do not insert a new zero mask
    at this intermediate boundary. Original consumers retain their own domains.
    ``selected`` may choose individual eligible entries without renaming them.
    """
    from . import chained_matmul as chain

    entries = tuple(entry for entry in cache.entries if entry.first_stage == stage)
    if selected is not None:
        if not selected <= {entry.node for entry in entries}:
            raise chain._UnsupportedChain(
                "selected cache is not eligible at this stage"
            )
        entries = tuple(entry for entry in entries if entry.node in selected)
    if not entries:
        return []
    execution = execution or ChainedExecution(plan.threads)
    lines: list[str] = []
    for layer in _publication_layers(entries):
        lines.extend(
            _emit_pointwise_cache_layer(cg, plan, boundaries, layer, execution)
        )
    return lines


def _emit_pointwise_cache_layer(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    entries: Sequence[PointwiseCacheEntry],
    execution: ChainedExecution,
) -> list[str]:
    from . import chained_matmul as chain

    available = {
        **(plan.loop.boundaries() if plan.loop is not None else {}),
        **boundaries,
    }
    lines: list[str] = []
    for entry in entries:
        if entry.node in available or any(
            dep not in available for dep in entry.dependencies
        ):
            raise chain._UnsupportedChain(
                "pointwise cache requires published source boundaries"
            )
        index = f"{entry.name}_index"
        coords = (
            (index,)
            if len(entry.shape) == 1
            else (f"{index} // {entry.shape[1]}", f"{index} % {entry.shape[1]}")
        )
        expression = chain._Expression(cg, plan, boundaries)
        expression.coordinate_names.add(index)
        value = expression.value(entry.node, coords)
        count = math.prod(entry.shape)
        lines.extend(
            [
                f"for {entry.name}_step in cutlass.range({(count + execution.threads - 1) // execution.threads}, unroll=1):",
                f"    {index} = {execution.thread} + {entry.name}_step * {execution.threads}",
                f"    if {index} < {count}:",
                chain._indent(expression.lines, 8),
                f"        {entry.name}[{', '.join(coords)}] = {_DTYPES[entry.dtype]}({value})",
            ]
        )
    lines.append(execution.sync)
    boundaries.update((entry.node, entry.name) for entry in entries)
    return lines
