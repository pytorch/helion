"""Prove when a full reduction axis is captured across a device loop."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import sympy
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
    from ..generate_ast import GenerateAST


def captured_reduction_coordinates(
    env: CompileEnvironment,
    graphs: list[GraphInfo],
    *,
    root_graph_id: int | None = None,
    physical_axes: frozenset[int] = frozenset(),
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
        and lowering.block_index not in physical_axes
        and reads_full_capture(node, lowering.block_index)
    )


def physical_capture_axes(cg: GenerateAST) -> frozenset[int]:
    """Recognize full scalar captures already covered by contiguous warp lanes.

    Each captured vector has one scalar per physical lane, with the complete
    static axis mapped to threadIdx.x. No synthetic lane loop or vector packet
    can straddle the capturing loop. Other coordinate requirements still go
    through the ordinary fragment-support proof.
    """
    from ..compile_environment import CompileEnvironment
    from ..device_ir import RootGraphInfo
    from ..reduction_strategy import PersistentReductionStrategy

    env = CompileEnvironment.current()
    fn = cg.device_function
    graphs = cg.host_function.device_ir.graphs
    if (
        fn.cute_state.simt_cluster_n != 1
        or len(cg.host_function.device_ir.root_ids) != 1
        or any(
            not isinstance(info, (RootGraphInfo, ForLoopGraphInfo)) for info in graphs
        )
    ):
        return frozenset()
    reductions = captured_reduction_coordinates(env, graphs)
    candidates = {node.meta["lowering"].block_index for node in reductions}
    parents = control_flow_parent_entries(graphs)
    result = set()
    for axis in candidates:
        size = env.block_sizes[axis].numel
        if (
            not isinstance(size, (int, sympy.Integer))
            or not 1 <= size <= 32
            or int(size) & (int(size) - 1)
        ):
            continue
        strategies = [s for s in fn.tile_strategy.strategies if axis in s.block_ids]
        if len(strategies) != 1:
            continue
        strategy = strategies[0]
        if (
            not isinstance(strategy, PersistentReductionStrategy)
            or strategy._synthetic_cute_lane_var is not None
            or fn.tile_strategy.thread_axis_for_block_id(axis) != 0
            or fn.tile_strategy.thread_extent_for_block_id(axis) != size
            or fn.resolved_block_size(axis) != size
            or env.config_spec.cute_vector_widths.config_get(
                cast("list[int]", fn.config.config.get("cute_vector_widths", []) or []),
                axis,
                1,
            )
            != 1
        ):
            continue
        seen = False
        valid = True
        for info in graphs:
            if info.graph_id not in parents:
                continue
            for node in info.graph.find_nodes(op="placeholder"):
                value = node.meta.get("val")
                if not isinstance(value, torch.Tensor):
                    continue
                axes = [env.resolve_block_id(dim) for dim in value.shape]
                if axis not in axes:
                    continue
                seen = True
                if sum(bid == axis for bid in axes) != 1 or any(
                    bid != axis and not env.known_equal(dim, 1)
                    for bid, dim in zip(axes, value.shape, strict=True)
                ):
                    valid = False
        if seen and valid:
            result.add(axis)
    return frozenset(result)
