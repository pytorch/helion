"""One typed, contraction-local cache for positional scalar contraction reuse.

The caller supplies the remaining *root-wide* shared-memory budget and current
materialized/aliased boundaries. This planner never changes those boundaries:
publication and temporary rebinding belong to the lexical contraction emitter.
"""

from __future__ import annotations

from fractions import Fraction
import math
from typing import TYPE_CHECKING

import torch

from ...language import memory_ops
from . import chained_pointwise_residency as residency
from .positional_domains import contraction_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import GraphInfo


def has_positional_pointwise_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """A shape-independent search fact; resolved codegen proves reuse/bounds."""
    for info in graphs:
        for node in info.graph.nodes:
            contraction = contraction_of(node)
            if contraction is None:
                continue
            for root in (contraction.lhs, contraction.rhs, contraction.acc):
                if root is None:
                    continue
                pending, seen = [root], set()
                while pending:
                    current = pending.pop()
                    if current in seen:
                        continue
                    seen.add(current)
                    if residency._arithmetic(current):
                        return True
                    if (
                        current.op != "placeholder"
                        and current.target is not memory_ops.load
                        and contraction_of(current) is None
                    ):
                        pending.extend(current.all_input_nodes)
    return False


def plan_contraction_cache(
    node: Node, boundaries: set[Node], budget: int
) -> residency.PointwiseCachePlan:
    """Rank independent candidates by saved scalar arithmetic per aligned byte.

    Scalar dot operands are evaluated for each output element and K iteration;
    C is evaluated once per output element. Publishing a candidate evaluates its
    original expression once per *physical* element, including padded lanes.
    The estimate is only a deterministic ranking, not a performance prediction.
    One candidate per contraction avoids nested-cache cost/lifetime assumptions.
    """
    if budget < 128:
        return residency.PointwiseCachePlan()
    contraction = contraction_of(node)
    assert contraction is not None
    outputs = math.prod(residency._shape(node))
    k_extent = residency._shape(contraction.lhs)[-1]
    producers = (
        (contraction.lhs, outputs * k_extent),
        (contraction.rhs, outputs * k_extent),
        (contraction.acc, outputs),
    )
    # Loads retain their original position-sensitive masks. They are leaves of
    # arithmetic discovery; their index/host metadata is not a cache candidate.
    cuts = boundaries | {
        item
        for item in node.graph.nodes
        if item.op == "placeholder" or item.target is memory_ops.load
    }
    uses: dict[Node, int] = {}
    redirected: set[Node] = {node}
    for root, evaluations in producers:
        if root is not None:
            ancestors, _ = residency._ancestors(root, cuts)
            redirected.update(ancestors)
            for ancestor in ancestors:
                uses[ancestor] = uses.get(ancestor, 0) + evaluations
    entries = []
    for candidate in node.graph.nodes:  # stable topological tie breaking
        value = candidate.meta.get("val")
        if (
            candidate not in uses
            or not residency._pointwise(candidate)
            or not isinstance(value, torch.Tensor)
            or value.dtype not in residency._DTYPES
            or not residency._finite_consumers(
                candidate, {node}, within=frozenset(redirected)
            )
        ):
            continue
        shape = residency._shape(candidate)
        if not shape or any(size <= 0 for size in shape):
            continue
        elements = math.prod(shape)
        byte_size = elements * value.element_size()
        allocated = (byte_size + 127) // 128 * 128
        if allocated > budget or uses[candidate] <= elements:
            continue
        ancestry, dependencies = residency._ancestors(candidate, cuts)
        if (
            not dependencies
            or any(item.target is torch.ops.aten.view.dtype for item in ancestry)
            or not any(residency._arithmetic(item) for item in ancestry)
            # An unmaterialized collective is not a scalar producer. The
            # emitter's ready boundaries normally cut all such dependencies.
            or any(contraction_of(item) is not None for item in ancestry)
        ):
            continue
        cost = sum(residency._cost(item) for item in ancestry)
        saved = (uses[candidate] - elements) * cost
        if saved <= 0:
            continue
        entries.append(
            residency.PointwiseCacheEntry(
                candidate,
                candidate.name,
                shape,
                value.dtype,
                0,
                0,
                byte_size,
                allocated,
                tuple(sorted(dependencies, key=lambda item: item.name)),
                uses[candidate],
                saved,
            )
        )
    if not entries:
        return residency.PointwiseCachePlan()
    winner = max(
        entries,
        key=lambda entry: (
            Fraction(entry.estimated_saved_operations, entry.allocated_bytes),
            -entry.allocated_bytes,
        ),
    )
    return residency.PointwiseCachePlan((winner,))
