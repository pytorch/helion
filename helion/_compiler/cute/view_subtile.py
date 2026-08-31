"""Attach local coordinates to views that split one tiled dimension.

A user-written ``val.view(..., D // F, F)`` turns a tile dimension ``D``
distributed over one thread axis into two logical dimensions. Those new
dimensions carry no intrinsic block id, so without metadata ``hl.split``'s
CuTe lowering reads constant-zero coordinates. A matching ``hl.join`` also
needs the minor coordinate to select the reconstructed element.

This pass records ``{block_id, divisor, modulus}`` mappings that
``cute_reshape._subtile_coord_expr`` expands into block-local coordinates. It
is a no-op unless one tiled dimension is split exactly.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch

from ...language.view_ops import join as hl_join
from ...language.view_ops import split as hl_split
from ..compile_environment import CompileEnvironment
from .cute_reshape import CUTE_DIM_LOCAL_COORD_META

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..device_ir import GraphInfo

_VIEW_TARGETS = (
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
)


def annotate_view_subtiles(graphs: list[GraphInfo], config: Config) -> None:
    """Annotate split views and matching joins with local coordinates."""
    env = CompileEnvironment.current()
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or CUTE_DIM_LOCAL_COORD_META in node.meta:
                continue
            meta: list[object | None] | None = None
            if node.target in _VIEW_TARGETS and _feeds_split(node):
                meta = _split_subtile_coord_meta(node, env, config)
            elif node.target is hl_join:
                meta = _join_subtile_coord_meta(node)
            if meta is not None:
                node.meta[CUTE_DIM_LOCAL_COORD_META] = meta


def _feeds_split(node: torch.fx.Node) -> bool:
    return any(
        user.op == "call_function" and user.target is hl_split for user in node.users
    )


def _split_subtile_coord_meta(
    node: torch.fx.Node,
    env: CompileEnvironment,
    config: Config,
) -> list[object | None] | None:
    """Return coordinate metadata when one tiled dimension becomes two."""
    from .cute_reshape import _get_tile_shape

    output_val = node.meta.get("val")
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return None
    input_val = source.meta.get("val")
    if not isinstance(output_val, torch.Tensor) or not isinstance(
        input_val, torch.Tensor
    ):
        return None
    if output_val.ndim != input_val.ndim + 1:
        return None

    input_shape = _get_tile_shape(input_val, env, config)
    output_shape = _get_tile_shape(output_val, env, config)
    source_meta = source.meta.get(CUTE_DIM_LOCAL_COORD_META)
    input_meta = (
        [*source_meta]
        if isinstance(source_meta, (list, tuple)) and len(source_meta) == input_val.ndim
        else [None] * input_val.ndim
    )
    for dim, input_extent in enumerate(input_shape):
        if (
            input_shape[:dim] != output_shape[:dim]
            or input_shape[dim + 1 :] != output_shape[dim + 2 :]
        ):
            continue
        outer_extent, inner_extent = output_shape[dim : dim + 2]
        if (
            outer_extent < 1
            or inner_extent < 2
            or outer_extent * inner_extent != input_extent
        ):
            continue

        old_coord = input_meta[dim]
        if isinstance(old_coord, dict) and isinstance(old_coord.get("block_id"), int):
            block_id = old_coord["block_id"]
            divisor = old_coord.get("divisor", 1)
            if not isinstance(divisor, int):
                continue
        else:
            block_id = env.get_block_id(input_val.shape[dim])
            divisor = 1
        if block_id is None:
            continue
        block_info = env.block_sizes[block_id]
        if block_info.reduction or env.is_jagged_tile(block_id):
            continue
        block_size = block_info.from_config(config)
        if not isinstance(block_size, int) or block_size % input_extent:
            continue

        return [
            *input_meta[:dim],
            {
                "block_id": block_id,
                "divisor": divisor * inner_extent,
                "modulus": outer_extent,
            },
            {
                "block_id": block_id,
                "divisor": divisor,
                "modulus": inner_extent,
            },
            *input_meta[dim + 1 :],
        ]
    return None


def _join_subtile_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    """Recover the minor selector when ``join`` rebuilds a split block dim."""
    output_val = node.meta.get("val")
    sources = node.args[:2]
    if len(sources) != 2 or not all(
        isinstance(source, torch.fx.Node) for source in sources
    ):
        return None
    left, right = sources
    assert isinstance(left, torch.fx.Node) and isinstance(right, torch.fx.Node)
    input_val = left.meta.get("val")
    right_val = right.meta.get("val")
    if (
        not isinstance(output_val, torch.Tensor)
        or not isinstance(input_val, torch.Tensor)
        or not isinstance(right_val, torch.Tensor)
        or right_val.shape != input_val.shape
        or output_val.ndim != input_val.ndim + 1
        or output_val.shape[-1] != 2
        or input_val.ndim == 0
    ):
        return None

    left_coord = _split_minor_coord_meta(left)
    right_coord = _split_minor_coord_meta(right)
    if left_coord is None or left_coord != right_coord:
        return None
    return [*([None] * input_val.ndim), left_coord]


def _split_minor_coord_meta(node: torch.fx.Node) -> dict[object, object] | None:
    """Trace a pointwise join operand back to the split it reconstructs."""
    if node.op != "call_function":
        return None
    if node.target is operator.getitem and node.args:
        split_node = node.args[0]
        if (
            isinstance(split_node, torch.fx.Node)
            and split_node.target is hl_split
            and split_node.args
            and isinstance(split_node.args[0], torch.fx.Node)
        ):
            split_input_meta = split_node.args[0].meta.get(CUTE_DIM_LOCAL_COORD_META)
            if (
                isinstance(split_input_meta, (list, tuple))
                and split_input_meta
                and isinstance(split_input_meta[-1], dict)
            ):
                return dict(split_input_meta[-1])
        return None

    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    from ..inductor_lowering import PointwiseLowering

    if not isinstance(node.meta.get("lowering"), PointwiseLowering):
        return None
    discovered: list[dict[object, object]] = []
    for input_node in node.all_input_nodes:
        input_value = input_node.meta.get("val")
        if (
            not isinstance(input_value, torch.Tensor)
            or input_value.shape != value.shape
        ):
            continue
        coord = _split_minor_coord_meta(input_node)
        if coord is not None and coord not in discovered:
            discovered.append(coord)
    return discovered[0] if len(discovered) == 1 else None
