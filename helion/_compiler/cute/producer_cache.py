"""Deterministic shared-cache candidates for repeated pure fragment producers."""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING

import torch
from torch._inductor.ir import Reduction
from torch.fx import Graph
from torch.fx import Node

from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import memory_ops
from ...language import view_ops
from ..inductor_lowering import ReductionLowering

if TYPE_CHECKING:
    from collections.abc import Callable

_VIEWS = frozenset(
    (
        view_ops.subscript,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.permute.default,
        torch.ops.aten.view.dtype,
    )
)
_EXPENSIVE = frozenset(
    (
        torch.ops.aten.exp.default,
        torch.ops.aten.exp2.default,
        torch.ops.aten.log.default,
        torch.ops.aten.log2.default,
        torch.ops.aten.tanh.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.sin.default,
        torch.ops.aten.cos.default,
        torch.ops.aten.erf.default,
    )
)


def pure_producer(node: Node) -> bool:
    """Schema-level purity; opaque programs and loads remain existing boundaries."""
    target = node.target
    if target in _VIEWS:
        return True
    return (
        isinstance(target, torch._ops.OpOverload)
        and torch.Tag.pointwise in target.tags
        and torch.Tag.nondeterministic_seeded not in target.tags
        and not target._schema.is_mutable
        and isinstance(node.meta.get("val"), torch.Tensor)
        and not isinstance(node.meta.get("lowering"), ReductionLowering)
    )


def producer_cache_candidates(
    graph: Graph,
    shape: Callable[[object], tuple[int, ...]] | None = None,
) -> frozenset[Node]:
    """Pick costly shared subexpressions without crossing effect/control epochs.

    Only straight-line roots are admitted. Every load is already a shared
    snapshot in this schedule. Local mutations, atomics and nested regions are
    excluded, so evaluating a lazy value at its defining node cannot change
    which storage epoch its consumers observe. The existing allocator retains
    all dependencies and enforces the configured shared-memory limit.
    """
    nodes = list(graph.nodes)
    if any(
        _tracing_ops.is_for_loop_target(n.target)
        or n.target
        in (_tracing_ops._if, _tracing_ops._while_loop, atomic_ops.atomic_add)
        or (
            n.target is memory_ops.store
            and (
                not isinstance(n.args[0], Node)
                or n.args[0].target is not _tracing_ops._host_tensor
            )
        )
        for n in nodes
    ):
        return frozenset()
    pure = {n for n in nodes if pure_producer(n)}

    def consumer_sinks(selected: set[Node]) -> dict[Node, frozenset[Node]]:
        result: dict[Node, frozenset[Node]] = {}
        for node in reversed(nodes):
            result[node] = frozenset(
                sink
                for user in node.users
                for sink in (
                    result[user] if user in pure and user not in selected else (user,)
                )
                if sink.op == "call_function"
            )
        return result

    sinks = consumer_sinks(set())

    def cost(node: Node, selected: set[Node], seen: set[Node]) -> int:
        if node not in pure or node in selected or node in seen:
            return 0
        seen.add(node)
        local = 0 if node.target in _VIEWS else 16 if node.target in _EXPENSIVE else 1
        return local + sum(cost(n, selected, seen) for n in node.all_input_nodes)

    candidates = [
        node
        for node in nodes
        if node in pure
        and node.target not in _VIEWS
        and len(node.users) > 1
        and len(sinks[node]) > 1
        and cost(node, set(), set()) >= 8
    ]
    if shape is None:
        return frozenset(candidates)

    def elements(node: Node) -> int:
        lowering = node.meta.get("lowering")
        if isinstance(lowering, ReductionLowering):
            assert isinstance(lowering.buffer.data, Reduction)
            return math.prod(shape(lowering.buffer.data.ranges)) * math.prod(
                shape(lowering.buffer.data.reduction_ranges)
            )
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            return math.prod(shape(value.shape))
        if node.target is memory_ops.store and isinstance(node.args[2], Node):
            value = node.args[2].meta.get("val")
            if isinstance(value, torch.Tensor):
                return math.prod(shape(value.shape))
        return 1

    selected: set[Node] = set()
    while candidates:
        # Stable graph order breaks ties. Broadcast/reduction expansion is
        # measured in configured physical elements, not shape hints.
        scored = []
        sinks = consumer_sinks(selected)
        for node in candidates:
            work = cost(node, selected, set())
            produced = elements(node)
            consumed = sum(elements(sink) for sink in sinks[node])
            saved = work * (consumed - produced)
            # Keep cheap expressions lazy even when broadcast many times.
            if len(sinks[node]) > 1 and work >= 8 and saved > 4 * (produced + consumed):
                scored.append((saved / produced, node))
        if not scored:
            break
        chosen = max(scored, key=operator.itemgetter(0))[1]
        selected.add(chosen)
        candidates.remove(chosen)
    return frozenset(selected)
