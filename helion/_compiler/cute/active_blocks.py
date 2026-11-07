"""CuTe uses of the logical-axis liveness analysis (``helion._compiler.active_blocks``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo
from ..device_ir import RootGraphInfo
from ..host_function import HostFunction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx.node import Node

    from ..device_function import DeviceFunction
    from ..inductor_lowering import CodegenState
    from ..tile_strategy import TileStrategy


def match_active_reduction_block_id(
    state: CodegenState, node: Node, size: object
) -> int | None:
    """The one active tile block whose extent is ``size``, for a reduction over
    a dim that is not a reduction block (``Backend.reduction_active_block_id``):
    the reduction then runs as a block-level strategy over that tile's threads.
    ``None`` when no block or more than one block matches."""
    env = CompileEnvironment.current()
    candidates: set[int] = set()
    block_id = env.resolve_block_id(size)
    if block_id is not None:
        block_id = env.resolve_codegen_block_id(
            block_id,
            state.codegen,
            node.graph,
        )
        if state.codegen.active_device_loops.get(block_id) or (
            state.codegen.device_function.tile_strategy.thread_axis_for_block_id(
                block_id
            )
            is not None
        ):
            candidates.add(block_id)
    for strategy in state.codegen.device_function.tile_strategy.strategies:
        for candidate_block_id in strategy.block_ids:
            if (
                state.codegen.device_function.tile_strategy.thread_axis_for_block_id(
                    candidate_block_id
                )
                is None
            ):
                continue
            candidate_size = env.block_sizes[candidate_block_id].size
            candidate_source = getattr(
                env.block_sizes[candidate_block_id].block_size_source,
                "value",
                None,
            )
            if (
                isinstance(size, torch.SymInt)
                and isinstance(candidate_source, torch.SymInt)
                and candidate_source._sympy_() == size._sympy_()
            ):
                candidates.add(candidate_block_id)
                continue
            if isinstance(size, (int, torch.SymInt)) and isinstance(
                candidate_size, (int, torch.SymInt)
            ):
                if env.known_equal(candidate_size, size):
                    candidates.add(candidate_block_id)

    seen: set[int] = set()
    for loops in state.codegen.active_device_loops.values():
        for loop_state in loops:
            key = id(loop_state)
            if key in seen:
                continue
            seen.add(key)
            for candidate_block_id, info in loop_state.block_id_to_info.items():
                if isinstance(size, (int, torch.SymInt)) and info.is_end_matching(size):
                    candidates.add(candidate_block_id)
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def inactive_broadcast_alias_block_ids(
    fn: DeviceFunction, strategies: Sequence[TileStrategy]
) -> set[int]:
    """Drop duplicate output-range reservations before physical allocation.

    Register broadcasting can allocate a reduction block whose variable is
    exactly an existing tile variable. The canonical tile already owns
    those coordinates. Nested tile loops may reuse an outer root's tile,
    but cannot establish ownership for sibling or child-only tile axes.
    Never elide executable reductions, free-iota or matrix dimensions.
    """
    env = CompileEnvironment.current()
    spec = env.config_spec
    graphs = fn.codegen.codegen_graphs
    if spec.matmul_facts or any(
        isinstance(node.target, torch._ops.OpOverload)
        and node.target.overloadpacket is torch.ops.aten.arange
        for graph in graphs
        for node in graph.graph.nodes
    ):
        return set()
    tile_blocks = {
        block_id for strategy in strategies for block_id in strategy.block_ids
    }
    if not (
        spec.pointwise_facts
        and all(isinstance(graph, RootGraphInfo) for graph in graphs)
    ):
        # One root's grid tiles dominate its tiled child graphs. Restrict
        # the extension to those outer coordinates: a child loop's tile
        # may be inactive in a sibling, even with an identical extent.
        grid_blocks = HostFunction.current().device_ir.grid_block_ids
        if (
            len(grid_blocks) != 1
            or sum(isinstance(graph, RootGraphInfo) for graph in graphs) != 1
            or not any(type(graph) is ForLoopGraphInfo for graph in graphs)
            or any(
                not isinstance(graph, RootGraphInfo)
                and not (type(graph) is ForLoopGraphInfo and graph.block_ids)
                for graph in graphs
            )
        ):
            return set()
        tile_blocks &= set(grid_blocks[0])
    executable = set(spec.reduction_loops.valid_block_ids())
    if spec.reduction_kernel_fact is not None:
        executable.update(
            reduction.block_id for reduction in spec.reduction_kernel_fact.reductions
        )
    if spec.kernel_matmul_fact is not None:
        executable.update(
            matmul.fact.k_block_id
            for matmul in spec.kernel_matmul_fact.matmuls
            if matmul.fact.k_block_id is not None
        )
    return {
        block.block_id
        for block in env.block_sizes
        if block.reduction
        and block.block_id not in executable
        and (canonical := env.canonical_block_id(block.block_id)) != block.block_id
        and canonical in tile_blocks
        and env.block_sizes[canonical].is_tile
        and block.symbol() == env.block_sizes[canonical].symbol()
        and block.numel == env.block_sizes[canonical].symbol()
    }
