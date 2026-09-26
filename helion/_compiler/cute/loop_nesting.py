"""The tile loops that need CUDA thread axes at the same time."""

from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.utils._pytree import tree_leaves

if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo

TileLoopPath = tuple[tuple[int, ...], ...]


def boundary_only_grid_blocks(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    graphs: Sequence[GraphInfo],
) -> set[int]:
    """Prove 1-D grid tiles whose elements are never observed by the body.

    A coarse outer tile may only provide begin/end/id for an inner tile. Its
    logical width still determines the grid and loop bounds, but distributing
    that width across threads or scalar lanes merely duplicates the body.
    Require direct inner-loop bounds and inspect nested graphs as well. Any
    other width use or device tensor dimension keeps the axis active. Bounds
    used to construct an arange may borrow the grid's element coordinates,
    so they deliberately keep the existing distribution.
    """
    from ...language import _tracing_ops
    from ...language import tile_ops

    candidates = {ids[0] for ids in device_ir.grid_block_ids if len(ids) == 1}
    boundary_uses: set[int] = set()
    other_uses: set[int] = set()
    tensor_values: list[torch.Tensor] = []

    def block_ids_in_size(
        size: int | torch.SymInt, *, include_bounds: bool = False
    ) -> set[int]:
        if not isinstance(size, torch.SymInt):
            return set()
        return {
            block_id
            for symbol in size._sympy_().free_symbols
            if (block_id := env.get_block_id(cast("sympy.Expr", symbol))) is not None
            and block_id in candidates
            # begin/end/id also carry a GridOrigin for this block, but their
            # scalar symbols do not denote the tile's element dimension.
            and (include_bounds or symbol == env.block_sizes[block_id].symbol())
        }

    boundary_ops = {tile_ops.tile_begin, tile_ops.tile_end, tile_ops.tile_id}
    for info in graphs:
        for node in info.graph.nodes:
            # Host allocation shapes can depend on the number of coarse tiles
            # without distributing any device value over the coarse tile.
            if node.target is _tracing_ops._host_tensor:
                continue
            if node.target in {tile_ops.tile_begin, tile_ops.tile_end}:
                size_node = node.args[0]
                assert isinstance(size_node, torch.fx.Node)
                size = size_node.meta["val"]
                if any(
                    not _tracing_ops.is_for_loop_target(user.target)
                    for user in node.users
                ):
                    other_uses.update(block_ids_in_size(size))
            for value in tree_leaves(node.meta.get("val")):
                if isinstance(value, torch.Tensor):
                    tensor_values.append(value)
                elif isinstance(value, torch.SymInt):
                    block_ids = block_ids_in_size(value)
                    if not block_ids or not node.users:
                        continue
                    if all(user.target in boundary_ops for user in node.users):
                        boundary_uses.update(block_ids)
                    else:
                        other_uses.update(block_ids)
    inactive = boundary_uses - other_uses
    if not inactive:
        return set()
    for value in tensor_values:
        for size in value.shape:
            inactive.difference_update(block_ids_in_size(size, include_bounds=True))
    return inactive


def _child_graph_ids(node: torch.fx.Node) -> tuple[int, ...]:
    from ...language import _tracing_ops

    if node.op != "call_function":
        return ()
    if _tracing_ops.is_for_loop_target(node.target):
        return (cast("int", node.args[0]),)
    if node.target is _tracing_ops._if:
        return cast("tuple[int, ...]", node.args[1:3])
    if node.target is _tracing_ops._while_loop:
        result = cast("tuple[int, ...]", node.args[:2])
        if len(node.args) > 3 and isinstance(node.args[3], int):
            result = (*result, node.args[3])
        return result
    return ()


def tile_loop_paths(
    device_ir: DeviceIR,
    graphs: Sequence[GraphInfo],
) -> tuple[TileLoopPath, ...]:
    """Return maximal stacks of simultaneously active non-reduction tiles.

    Sibling loops, branches, and separate grid roots share thread axes. A
    nested tile needs additional axes. Rolled reductions are deliberately
    excluded: their strategies reserve axes independently of tile nesting.
    Read the codegen graphs so reduction rolling and epilogue subtiling are
    reflected in the result, and ignore unreachable graph copies.
    """
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import ReductionLoopGraphInfo

    paths: dict[TileLoopPath, None] = {}

    def visit(graph_id: int, path: TileLoopPath) -> None:
        info = graphs[graph_id]
        if (
            isinstance(info, ForLoopGraphInfo)
            and not isinstance(info, ReductionLoopGraphInfo)
            and info.block_ids
        ):
            path = (*path, tuple(info.block_ids))
        paths[path] = None
        for node in info.graph.nodes:
            for child in _child_graph_ids(node):
                visit(child, path)

    for root_id, block_ids in zip(
        device_ir.root_ids, device_ir.grid_block_ids, strict=True
    ):
        visit(root_id, (tuple(block_ids),))
    return tuple(
        path
        for path in paths
        if not any(
            len(other) > len(path) and other[: len(path)] == path for other in paths
        )
    )


def sibling_row_loop_blocks(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    graphs: Sequence[GraphInfo],
) -> tuple[int, tuple[int, ...]] | None:
    """Prove a row grid with sibling, equally bounded 1-D tile passes.

    Tile IDs can differ between passes even though their logical iteration
    space is the same. This proof allows their layouts and autotuner seeds
    to agree without conflating genuinely nested dimensions.
    """
    from ...language import _tracing_ops
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import control_flow_parent_entries
    from ..device_ir import device_loop_bounds

    if len(device_ir.grid_block_ids) != 1 or len(device_ir.grid_block_ids[0]) != 1:
        return None
    paths = tile_loop_paths(device_ir, graphs)
    if not paths or any(len(path) != 2 or len(path[1]) != 1 for path in paths):
        return None
    (row_block_id,) = device_ir.grid_block_ids[0]
    inner_block_ids = tuple(dict.fromkeys(path[1][0] for path in paths))
    if row_block_id in inner_block_ids:
        return None
    parents = {
        graph_id: entry[0]
        for graph_id, entry in control_flow_parent_entries(graphs).items()
    }
    reference: tuple[int | torch.SymInt, int | torch.SymInt] | None = None
    for info in graphs:
        if not isinstance(info, ForLoopGraphInfo):
            continue
        if len(info.block_ids) != 1 or info.block_ids[0] not in inner_block_ids:
            continue
        parent = parents.get(info.graph_id)
        if parent is None or parent.target is not _tracing_ops._for_loop:
            return None
        bounds = device_loop_bounds(info, parents, info.block_ids[0])
        if bounds is None or not all(
            isinstance(value, (int, torch.SymInt)) for value in bounds
        ):
            return None
        typed_bounds = cast("tuple[int | torch.SymInt, int | torch.SymInt]", bounds)
        if reference is None:
            reference = typed_bounds
        elif not all(
            starmap(env.known_equal, zip(reference, typed_bounds, strict=True))
        ):
            return None
    if reference is None:
        return None
    return row_block_id, inner_block_ids
