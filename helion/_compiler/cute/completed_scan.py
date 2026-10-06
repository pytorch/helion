"""Closed, same-owner consumers of a completed short warp scan.

The proof selects a straight-line interval, not an algorithm. Shared scan
publication remains intact. Only terminal integer extrema are rescheduled;
all floating-point expressions keep their original operation trees.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import inline_asm_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_ops
from ...language import view_ops
from ..inductor_lowering import ReductionLowering

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._inductor.ir import Reduction
    from torch.fx import Graph
    from torch.fx import Node


@dataclass(frozen=True)
class CompletedScan:
    scan: Node
    reductions: tuple[Node, ...]


def pure(node: Node) -> bool:
    target = node.target
    if target in (
        _tracing_ops._mask_to,
        view_ops.subscript,
        torch.ops.aten.alias.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.scalar_tensor.default,
        creation_ops.full,
        torch.ops.aten.full.default,
        torch.ops.prims.iota.default,
    ):
        return True
    return (
        isinstance(target, torch._ops.OpOverload)
        and torch.Tag.pointwise in target.tags
        and torch.Tag.nondeterministic_seeded not in target.tags
        and not target._schema.is_mutable
        and not isinstance(node.meta.get("lowering"), ReductionLowering)
    )


def completed_scan_plans(
    graph: Graph, shape: Callable[[object], tuple[int, ...]] | None = None
) -> tuple[CompletedScan, ...]:
    """Prove closed terminal uses and no effect/mutation in the export interval.

    The optional physical resolver is mandatory at emission. Discovery may
    expose the knob before configuration selects the exact one-row capacity.
    Loads, previous reductions and opaque ASM are existing shared boundaries.
    The flag excludes lazy opaque-producer/private layouts. New opaque
    consumers, captures and index remaps decline; no ASM is replicated.
    """
    nodes = list(graph.nodes)
    order = {node: i for i, node in enumerate(nodes)}

    def terminal(node: Node, scan: Node) -> bool:
        low = node.meta.get("lowering")
        value = node.meta.get("val")
        before = cast("torch.Tensor", scan.meta["val"])
        if shape is None:
            return (
                node.target
                in (torch.ops.aten.amin.default, torch.ops.aten.amax.default)
                and isinstance(value, torch.Tensor)
                and value.dtype in (torch.int32, torch.int64)
            )
        if not (
            isinstance(low, ReductionLowering)
            and low.reduction_type in ("min", "max")
            and isinstance(value, torch.Tensor)
            and value.dtype in (torch.int32, torch.int64)
            and len(cast("Reduction", low.buffer.data).reduction_ranges) == 1
        ):
            return False
        physical = shape(before.shape)
        return (
            math.prod(shape(value.shape)) == 1
            and math.prod(shape(low.buffer.data.ranges)) == 1
            and shape(cast("Reduction", low.buffer.data).reduction_ranges)
            == (physical[-1],)
        )

    def recipe(node: Node, stop: Node, seen: set[Node]) -> bool:
        if node in seen or node is stop:
            return True
        seen.add(node)
        if (
            node.target in (memory_ops.load, inline_asm_ops.inline_asm_elementwise)
            or isinstance(node.meta.get("lowering"), ReductionLowering)
            or node.target
            in (
                scan_ops._associative_scan,
                torch.ops.aten.amin.default,
                torch.ops.aten.amax.default,
            )
        ):
            return True  # Existing shared snapshots, not a reordered producer.
        if node.target in (
            _tracing_ops._get_symnode,
            tile_ops.tile_index,
            tile_ops.tile_begin,
            tile_ops.tile_end,
            tile_ops.tile_id,
        ):
            return True
        return pure(node) and all(recipe(n, stop, seen) for n in node.all_input_nodes)

    result = []
    for scan in nodes:
        value = scan.meta.get("val")
        if not (
            scan.target is scan_ops._associative_scan
            and isinstance(value, torch.Tensor)
            and value.dtype in (torch.float32, torch.int32, torch.int64)
            and value.ndim > 0
            and scan.args[2] in (-1, value.ndim - 1)
            and scan.args[3] is False
            and scan.users
        ):
            continue
        physical: tuple[int, ...] = ()
        if shape is not None:
            physical = shape(value.shape)
            if not (math.prod(physical[:-1]) == 1 and 0 < physical[-1] <= 1024):
                continue
        seen: set[Node] = set()
        pending = list(scan.users)
        reductions: set[Node] = set()
        valid = True
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            if terminal(node, scan):
                reductions.add(node)
                continue
            if (
                node.graph is not graph
                or not pure(node)
                or not node.users
                or not isinstance(node.meta.get("val"), torch.Tensor)
                or (shape is not None and shape(node.meta["val"].shape) != physical)
            ):
                valid = False
                break
            pending.extend(node.users)
        if (
            not valid
            or not reductions
            or len({n.meta["val"].dtype for n in reductions}) != 1
        ):
            continue
        last = max(order[n] for n in reductions)
        # An adjacent independent extremum of the same input may share the
        # CTA fold. Never delay a result past any of its users.
        limit = min(order[u] for n in reductions for u in n.users)
        for node in nodes[last + 1 : limit]:
            if terminal(node, scan):
                ancestors: set[Node] = set()
                if (
                    recipe(cast("Node", node.args[0]), scan, ancestors)
                    and scan.args[1] in ancestors
                    and not reductions.intersection(ancestors)
                    and node.meta["val"].dtype
                    == next(iter(reductions)).meta["val"].dtype
                ):
                    reductions.add(node)
                    last = order[node]
        interval = nodes[order[scan] + 1 : last + 1]
        if (
            len(reductions) > 4
            or any(n not in reductions and not pure(n) for n in interval)
            or any(order[u] <= last for n in reductions for u in n.users)
            or not all(recipe(cast("Node", n.args[0]), scan, set()) for n in reductions)
        ):
            continue
        result.append(
            CompletedScan(scan, tuple(sorted(reductions, key=order.__getitem__)))
        )
    return tuple(result)
