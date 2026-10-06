"""Prove when a full reduction axis is captured across a device loop."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import _tracing_ops
from ...language import memory_ops
from ..device_ir import ForLoopGraphInfo
from ..device_ir import ReductionLoopGraphInfo
from ..device_ir import control_flow_parent_entries
from ..inductor_lowering import ReductionLowering

if TYPE_CHECKING:
    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo


def captured_reduction_coordinates(
    env: CompileEnvironment,
    graphs: list[GraphInfo],
    *,
    root_graph_id: int | None = None,
) -> frozenset[Node]:
    """Identify full captured tensor reductions that need complete ownership.

    A native scalar recipe can represent one captured element per thread, but
    not the complete axis when lane loops surround the capturing device loop.
    Explicitly tiled reduction axes are excluded: their loads and reduction
    coordinates belong to the same tiled iteration instead of a full capture.
    This proves a coordinate requirement, not operation or storage support.
    """
    parents = control_flow_parent_entries(graphs)
    if not parents:
        return frozenset()
    by_graph = {info.graph: info for info in graphs}
    captures: dict[Node, Node] = {}
    for info in graphs:
        if info.graph_id not in parents:
            continue
        call, slot = parents[info.graph_id]
        captures.update(
            zip(
                info.graph.find_nodes(op="placeholder"),
                cast("list[Node]", call.args[slot]),
                strict=True,
            )
        )

    def root_id(info: GraphInfo) -> int:
        while info.graph_id in parents:
            call, _ = parents[info.graph_id]
            info = by_graph[call.graph]
        return info.graph_id

    def reads_full_capture(node: Node, axis: int) -> bool:
        pending = list(node.all_input_nodes)
        seen: set[Node] = set()
        while pending:
            value = pending.pop()
            if value in seen:
                continue
            seen.add(value)
            if value in captures:
                info = by_graph[value.graph]
                tensor = value.meta.get("val")
                if (
                    isinstance(info, ForLoopGraphInfo)
                    and not isinstance(info, ReductionLoopGraphInfo)
                    and axis not in info.block_ids
                    and isinstance(tensor, torch.Tensor)
                    and any(env.resolve_block_id(size) == axis for size in tensor.shape)
                ):
                    return True
                pending.append(captures[value])
            elif value.target not in (memory_ops.load, _tracing_ops._host_tensor):
                pending.extend(value.all_input_nodes)
        return False

    return frozenset(
        node
        for info in graphs
        if root_graph_id is None or root_id(info) == root_graph_id
        for node in info.graph.nodes
        if isinstance(lowering := node.meta.get("lowering"), ReductionLowering)
        and env.block_sizes[lowering.block_index].reduction
        and reads_full_capture(node, lowering.block_index)
    )
