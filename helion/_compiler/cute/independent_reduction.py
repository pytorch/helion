"""Prove when a full reduction and an output tile need distinct coordinates."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.ir import Reduction
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_ops
from ..device_ir import ReductionLoopGraphInfo
from ..inductor_lowering import ReductionLowering

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo


def independent_reduction_coordinates(
    env: CompileEnvironment,
    graphs: list[GraphInfo],
    *,
    include_rolled_outputs: bool = False,
) -> frozenset[Node]:
    """Find reductions whose result is consumed along another stored tile axis.

    This proves a need for complete logical coordinates, not general operation
    support or memory safety. The ordinary fragment admission and emitter still
    check every operation, extent, layout, predicate and shared allocation.
    Only dependencies inside one GraphInfo participate; cross-loop captures
    and arbitrary computed store indices keep their existing paths. The optional
    rolled-output projection is only for conservative metadata specialization
    discovery: a later persistent config may inline that full reduction.
    """

    def shape_axes(node: Node) -> frozenset[int]:
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return frozenset()
        return frozenset(
            axis
            for size in value.shape
            if (axis := env.resolve_block_id(size)) is not None
        )

    def stored_axis(index: object) -> int | None:
        if not isinstance(index, Node):
            return None
        if index.target is tile_ops.tile_id:
            index = index.args[0]
        if not isinstance(index, Node) or index.target is not _tracing_ops._get_symnode:
            return None
        axis = env.resolve_block_id(index.meta.get("val"))
        if axis is None or env.block_sizes[axis].reduction:
            return None
        return axis

    def full_input_axis(
        node: Node, lowering: ReductionLowering, axes: dict[Node, frozenset[int]]
    ) -> bool:
        reduction = lowering.buffer.data
        assert isinstance(reduction, Reduction)
        (extent,) = reduction.reduction_ranges
        for source in node.all_input_nodes:
            if lowering.block_index in axes[source]:
                return True
            value = source.meta.get("val")
            if (
                isinstance(value, torch.Tensor)
                and isinstance(extent, (int, sympy.Integer))
                and int(extent) > 1
                and any(
                    isinstance(size, int) and size == int(extent)
                    for size in value.shape
                )
            ):
                return True
        return False

    axes = {node: shape_axes(node) for info in graphs for node in info.graph.nodes}
    producers = {
        node: (
            lowering.block_index,
            set().union(*(axes[source] for source in node.all_input_nodes)),
        )
        for info in graphs
        for node in info.graph.nodes
        if isinstance(lowering := node.meta.get("lowering"), ReductionLowering)
        and env.block_sizes[lowering.block_index].reduction
        and full_input_axis(node, lowering, axes)
    }
    if include_rolled_outputs:
        by_id = {info.graph_id: info for info in graphs}
        for info in graphs:
            for node in info.graph.nodes:
                if node.target is not operator.getitem or len(node.args) != 2:
                    continue
                call, slot = node.args
                if (
                    not isinstance(call, Node)
                    or not _tracing_ops.is_for_loop_target(call.target)
                    or not isinstance(slot, int)
                ):
                    continue
                child = by_id[cast("int", call.args[0])]
                if not isinstance(child, ReductionLoopGraphInfo):
                    continue
                output = next(iter(child.graph.find_nodes(op="output")))
                values = output.args[0]
                if (
                    isinstance(values, (list, tuple))
                    and 0 <= slot < len(values)
                    and isinstance(value := values[slot], Node)
                    and value in producers
                ):
                    producers[node] = producers[value]

    result: set[Node] = set()
    for info in graphs:
        reductions = {
            node: proof for node, proof in producers.items() if node.graph is info.graph
        }
        for store in info.graph.nodes:
            if store.target is not memory_ops.store:
                continue
            indices, value = store.args[1:3]
            if not isinstance(indices, (list, tuple)) or not isinstance(value, Node):
                continue
            output_axes = {
                axis for index in indices if (axis := stored_axis(index)) is not None
            }
            if not output_axes:
                continue
            # A full statistic must actually feed this store's payload. Stop
            # at global loads: memory aliasing is not a data-dependency proof.
            ancestors: set[Node] = set()
            pending = [value]
            while pending:
                node = pending.pop()
                if node.graph is not info.graph or node in ancestors:
                    continue
                ancestors.add(node)
                if node.target is not memory_ops.load:
                    pending.extend(node.all_input_nodes)
            for reduction, (full_axis, source_axes) in reductions.items():
                if reduction not in ancestors:
                    continue
                independent = output_axes - source_axes - {full_axis}
                if not independent:
                    continue
                # Require the independent tile coordinate on the actual value
                # path, not just an unrelated tiled input or an output index.
                pending = list(reduction.users)
                visited: set[Node] = set()
                while pending:
                    node = pending.pop()
                    if node not in ancestors or node in visited:
                        continue
                    visited.add(node)
                    # Broadcasting a statistic back onto its own full axis
                    # retains the existing native reduction coordinates. Later
                    # row broadcasts do not make that normalization independent.
                    if full_axis in axes[node]:
                        continue
                    if axes[node] & independent:
                        result.add(reduction)
                        break
                    pending.extend(node.users)
    return frozenset(result)
