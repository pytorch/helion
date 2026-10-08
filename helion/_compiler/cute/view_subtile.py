"""Attach local coordinates to views that split one tiled dimension.

A user-written ``val.view(..., D // F, F)`` turns a tile dimension ``D``
distributed over one thread axis into two logical dimensions. Those new
dimensions carry no intrinsic block id, so without metadata ``hl.split``'s
CuTe lowering cannot tell which pair element a thread holds. A matching
``hl.join`` also needs the minor coordinate to select the reconstructed
element.

This pass records ``{block_id, divisor, modulus}`` mappings that
``cute_reshape._subtile_coord_expr`` expands into block-local coordinates,
and carries them through the permute/unsqueeze/expand/cast views that may
sit between the split view and ``hl.split`` (and after ``hl.join``). It is a
no-op unless one tiled dimension is split exactly.
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

# Views and casts that keep every thread's element in place; split-view
# coordinates propagate through them (reordered or padded with ``None``).
_PASS_THROUGH_TARGETS = (
    torch.ops.aten.permute.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.clone.default,
    torch.ops.aten._to_copy.default,
    torch.ops.prims.convert_element_type.default,
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
            elif node.target in _PASS_THROUGH_TARGETS:
                meta = _propagated_coord_meta(node)
            elif node.target is operator.getitem:
                meta = _split_output_coord_meta(node)
            elif node.target is hl_join:
                meta = _join_subtile_coord_meta(node)
            else:
                meta = _pointwise_coord_meta(node)
            if meta is not None:
                node.meta[CUTE_DIM_LOCAL_COORD_META] = meta


def _feeds_split(node: torch.fx.Node) -> bool:
    for user in node.users:
        if user.op != "call_function":
            continue
        if user.target is hl_split:
            return True
        if user.target in _PASS_THROUGH_TARGETS and _feeds_split(user):
            return True
    return False


def _source_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return None
    meta = source.meta.get(CUTE_DIM_LOCAL_COORD_META)
    return [*meta] if isinstance(meta, (list, tuple)) else None


def _propagated_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    """Carry split-view coordinates through a pass-through view or cast."""
    meta = _source_coord_meta(node)
    if meta is None:
        return None
    if node.target is torch.ops.aten.permute.default:
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dims")
        if not isinstance(dims, (list, tuple)):
            return None
        perm = [dim for dim in dims if isinstance(dim, int)]
        if len(perm) != len(dims) or len(perm) != len(meta):
            return None
        return [meta[dim] for dim in perm]
    if node.target is torch.ops.aten.unsqueeze.default:
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(dim, int):
            return None
        dim %= len(meta) + 1
        return [*meta[:dim], None, *meta[dim:]]
    if node.target is torch.ops.aten.expand.default:
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor) or value.ndim < len(meta):
            return None
        return [*([None] * (value.ndim - len(meta))), *meta]
    return meta


def _split_output_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    """``hl.split`` outputs keep the input's coordinates minus the pair dim."""
    split_node = node.args[0] if node.args else None
    if not isinstance(split_node, torch.fx.Node) or split_node.target is not hl_split:
        return None
    meta = _source_coord_meta(split_node)
    if meta is None or not meta:
        return None
    return meta[:-1]


def _pointwise_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    """A per-thread pointwise op keeps the coordinates of its same-shape inputs.

    Stores consume these so a split output (or arithmetic on it) is written at
    the coordinate the thread actually holds.
    """
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    from ..inductor_lowering import PointwiseLowering

    if not isinstance(node.meta.get("lowering"), PointwiseLowering):
        return None
    discovered: list[list[object | None]] = []
    for input_node in node.all_input_nodes:
        input_value = input_node.meta.get("val")
        meta = input_node.meta.get(CUTE_DIM_LOCAL_COORD_META)
        if (
            not isinstance(input_value, torch.Tensor)
            or input_value.shape != value.shape
            or not isinstance(meta, (list, tuple))
        ):
            continue
        if [*meta] not in discovered:
            discovered.append([*meta])
    if len(discovered) != 1 or not any(
        isinstance(info, dict) for info in discovered[0]
    ):
        return None
    return discovered[0]


def _dim_block_id(env: CompileEnvironment, size: int | torch.SymInt) -> int | None:
    """Block id owning a tile dim.

    A static full-slice extent (``x[tile, :]``) is not a block symbol; like
    load addressing, map it to the unique reduction dim of that size.
    """
    block_id = env.get_block_id(size)
    if block_id is not None:
        return block_id
    candidates = [
        info.block_id
        for info in env.block_sizes
        if info.reduction
        and isinstance(info.size, (int, torch.SymInt))
        and env.known_equal(info.size, size)
    ]
    return candidates[0] if len(candidates) == 1 else None


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
            block_id = _dim_block_id(env, input_val.shape[dim])
            divisor = 1
        if block_id is None or env.is_jagged_tile(block_id):
            continue
        block_size = env.block_sizes[block_id].from_config(config)
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
    """Recover the split input's coordinates when ``join`` rebuilds its shape."""
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

    left_meta = _split_source_coord_meta(left)
    right_meta = _split_source_coord_meta(right)
    if (
        left_meta is None
        or left_meta != right_meta
        or len(left_meta) != output_val.ndim
    ):
        return None
    return left_meta


def _split_source_coord_meta(node: torch.fx.Node) -> list[object | None] | None:
    """Trace a pointwise join operand back to the split input it reconstructs.

    Returns that input's full coordinate list (its last entry is the minor
    pair coordinate the join selects with).
    """
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
                return [*split_input_meta]
        return None

    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    from ..inductor_lowering import PointwiseLowering

    if not isinstance(node.meta.get("lowering"), PointwiseLowering):
        return None
    discovered: list[list[object | None]] = []
    for input_node in node.all_input_nodes:
        input_value = input_node.meta.get("val")
        if (
            not isinstance(input_value, torch.Tensor)
            or input_value.shape != value.shape
        ):
            continue
        meta = _split_source_coord_meta(input_node)
        if meta is not None and meta not in discovered:
            discovered.append(meta)
    return discovered[0] if len(discovered) == 1 else None


def _split_minor_coord_meta(node: torch.fx.Node) -> dict[object, object] | None:
    """Trace a pointwise join operand back to the split's minor coordinate."""
    meta = _split_source_coord_meta(node)
    if meta is None:
        return None
    minor = meta[-1]
    return dict(minor) if isinstance(minor, dict) else None
