"""Attach subtile coordinate metadata to view/reshape nodes that split a
single tiled block dimension by a constexpr factor.

A user-written ``val.view(block_size // F, F)`` (or ``reshape``) turns a 1-D
block tile ``[D]`` distributed over a single thread axis into a 2-D tile
``[D // F, F]``. The new dimensions carry no intrinsic per-thread coordinate,
so without help ``hl.split``'s CuTe codegen reads constant-0 coordinates and
every thread writes the same shared-memory slot.

This pass detects that pattern and records, on the view node, the per-dimension
``{block_id, divisor, modulus}`` mapping that ``cute_reshape._subtile_coord_expr``
expands into ``block_local // divisor`` / ``block_local % modulus``. With the
metadata attached, dim ``i`` of ``[D // F, F]`` maps thread lane ``t`` to
``(t // F, t % F)`` so the flattened source index is ``t`` again — i.e. thread
``t`` writes its own loaded scalar to ``smem[t]``.

The pass is a strict no-op unless every gate matches, so existing reshape/split
kernels are byte-for-byte unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
    """Annotate single-block-dim split views with subtile coordinate metadata."""
    env = CompileEnvironment.current()
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target not in _VIEW_TARGETS:
                continue
            if CUTE_DIM_LOCAL_COORD_META in node.meta:
                continue
            # Only meaningful when the view feeds ``hl.split`` — that is the only
            # consumer of this metadata. Restricting to split consumers keeps the
            # pass from touching unrelated reshape nodes.
            if not _feeds_split(node):
                continue
            meta = _split_subtile_coord_meta(node, env, config)
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
    """Return ``[..{block_id, divisor, modulus}..]`` for a 1-D-block split view.

    Recognizes exactly ``[D] -> [D // F, F]`` where ``D`` is a single tiled
    block dim mapped to one thread axis and ``F`` is a static constexpr factor.
    """
    output_val = node.meta.get("val")
    source = node.args[0]
    if not isinstance(source, torch.fx.Node):
        return None
    input_val = source.meta.get("val")
    if not isinstance(output_val, torch.Tensor) or not isinstance(
        input_val, torch.Tensor
    ):
        return None
    # Only the bare ``[D] -> [D // F, F]`` shape is handled here.
    if input_val.ndim != 1 or output_val.ndim != 2:
        return None

    factor = output_val.shape[1]
    if not isinstance(factor, int) or factor < 2:
        return None

    block_id = env.get_block_id(input_val.shape[0])
    if block_id is None:
        return None
    if env.block_sizes[block_id].reduction or env.is_jagged_tile(block_id):
        return None

    # The view collapses the whole block dim into ``[D // F, F]`` only when the
    # configured block size is an exact multiple of the constexpr factor.
    block_size = env.block_sizes[block_id].from_config(config)
    if not isinstance(block_size, int) or block_size % factor != 0:
        return None

    # dim 0 (size D // F): block_local // F ; dim 1 (size F): block_local % F.
    return [
        {"block_id": block_id, "divisor": factor, "modulus": None},
        {"block_id": block_id, "divisor": 1, "modulus": factor},
    ]
