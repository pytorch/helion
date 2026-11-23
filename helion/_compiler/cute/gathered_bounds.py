"""Gathered coordinates whose bound the masks of their tiles imply.

Role: general-purpose analysis of the memory lowering
(``cute_guarded_lanes``).

A load through a computed coordinate, ``x[tile_n, tile_h.index * 2 + 1,
tile_c]``, is masked by the masks of the tiles its coordinates read and by an
``index < size`` term of each coordinate (a signed int32 compare; nothing
checks ``index >= 0``).  Where the coordinate is an affine function of tile
indices over static extents, its range on the lanes those masks admit is
known: a tile's index lies in ``[begin, begin + numel)`` wherever the tile's
mask holds, and on every lane of a tile whose mask the strategy elided.
When the top of that range is below ``size`` the term repeats the tile masks,
so the mask of the load goes without it: the packet guard then holds nothing
but tile masks and what the program's own ``extra_mask`` adds, which is what
lets the guard of the store that consumes the load stand in for it
(``guarded_lanes``).  Which elements a lane reads does not change: only a
term that is true on every lane the other terms admit is left out.

``proven_gather_positions`` decides this on the device FX graph by signed
interval arithmetic, so a coordinate that can pass the dim (``index + 1``
over the tile's own extent) keeps its term, and so does anything it does not
model: a symbolic extent, a tile that does not begin at a static position, an
intermediate that could leave the int32 range, and every operation other than
``+``, ``-`` and ``*`` of tile indices and integers under ``None`` / ``:``
views.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...language.tile_ops import tile_index
from ...language.view_ops import subscript as view_subscript
from ..compile_environment import CompileEnvironment
from .emitters.memory_access import _cute_index_override
from .emitters.memory_access import _cute_remap_block_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..inductor_lowering import CodegenState

_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_Interval = tuple[int, int]


def _checked(low: int, high: int) -> _Interval | None:
    """``[low, high]`` when every value of it is an int32, as the generated
    index arithmetic is; None for a range a wrapped intermediate could leave."""
    if _INT32_MIN <= low <= high <= _INT32_MAX:
        return low, high
    return None


def _tile_index_interval(
    state: CodegenState, node: torch.fx.Node, blocks: set[int]
) -> _Interval | None:
    """The values ``tile_index`` takes on the lanes its tile's mask admits;
    its block joins ``blocks``."""
    from ..tile_strategy import BlockSizeTileStrategy

    env = CompileEnvironment.current()
    symbol = node.args[0]
    if isinstance(symbol, torch.fx.Node):
        symbol = symbol.meta.get("val")
    if not isinstance(symbol, torch.SymInt):
        return None
    block_id = env.get_block_id(symbol)
    if block_id is None:
        return None
    # An operand index override or a block remap replaces a tile's index
    # while a matmul operand is re-materialized, and ``cute_guarded_lanes``
    # is not offered to a kernel with a matmul (``guarded_lanes_offered``).
    assert (
        _cute_index_override(state, block_id) is None
        and _cute_remap_block_id(state, block_id) == block_id
    )
    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return None
    owner = loops[-1]
    # The strategy masks the lanes past the tile's end, or elided the mask
    # because none exist (``_setup_mask``).
    if not isinstance(owner.strategy, BlockSizeTileStrategy):
        return None
    info = owner.block_id_to_info.get(block_id)
    block = env.block_sizes[block_id]
    # A jagged tile has no extent of its own.
    if not isinstance(block.size, (int, torch.SymInt)):
        return None
    numel = block.numel
    if info is None or not numel.is_Integer:
        return None
    # The tile's first element: a static position, or zero for a loop that
    # records no begin at all.
    if info.begin_expr is not None:
        if not info.begin_expr.is_Integer:
            return None
        begin = int(info.begin_expr)
    elif info.begin_var_name is None:
        begin = 0
    else:
        return None
    extent = int(numel)
    if begin < 0 or extent < 1:
        return None
    if info.end_expr is not None and info.end_expr != begin + extent:
        return None
    blocks.add(block_id)
    return _checked(begin, begin + extent - 1)


def _interval(state: CodegenState, value: object, blocks: set[int]) -> _Interval | None:
    """The range of an integer coordinate on the lanes the masks of the tiles
    it reads admit, or None when it is not modeled; the blocks of those
    tiles join ``blocks``."""
    if type(value) is int:
        return _checked(value, value)
    if not (isinstance(value, torch.fx.Node) and value.op == "call_function"):
        return None
    target = value.target
    if target is tile_index:
        return _tile_index_interval(state, value, blocks)
    if target is view_subscript:
        index = value.args[1] if len(value.args) == 2 else None
        if not (
            isinstance(index, (list, tuple))
            and all(entry is None or entry == slice(None) for entry in index)
        ):
            return None
        return _interval(state, value.args[0], blocks)
    if value.kwargs or len(value.args) != 2:
        return None
    if target not in (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
    ):
        return None
    left = _interval(state, value.args[0], blocks)
    right = _interval(state, value.args[1], blocks)
    if left is None or right is None:
        return None
    if target is torch.ops.aten.add.Tensor:
        return _checked(left[0] + right[0], left[1] + right[1])
    if target is torch.ops.aten.sub.Tensor:
        return _checked(left[0] - right[1], left[1] - right[0])
    products = [a * b for a in left for b in right]
    return _checked(min(products), max(products))


def proven_gather_positions(
    state: CodegenState, tensor: torch.Tensor, subscript: Sequence[object]
) -> frozenset[int]:
    """The positions of ``subscript`` whose gathered coordinate stays below
    the size of its dim of ``tensor`` on every lane the masks of its tiles
    admit (module docstring); empty when the load's FX node is not at hand."""
    env = CompileEnvironment.current()
    fx_node = state.fx_node
    fx_index = (
        fx_node.args[1] if fx_node is not None and len(fx_node.args) > 1 else None
    )
    if not isinstance(fx_index, (list, tuple)) or len(fx_index) != len(subscript):
        return frozenset()
    proven: set[int] = set()
    tensor_dim = 0
    for position, (index, fx_entry) in enumerate(zip(subscript, fx_index, strict=True)):
        if index is None:
            continue
        if (
            isinstance(index, torch.Tensor)
            and index.dtype in (torch.int32, torch.int64)
            and tensor_dim < tensor.ndim
        ):
            size = tensor.shape[tensor_dim]
            blocks: set[int] = set()
            interval = (
                _interval(state, fx_entry, blocks) if isinstance(size, int) else None
            )
            # The term is the signed ``index < size`` alone: a coordinate
            # that can go negative loses a term that was true there anyway.
            if interval is not None and interval[1] < size:
                # The mask of the load holds the masks of the tiles whose
                # sizes are dims of the coordinate (``_cute_combined_mask``),
                # and the operations modeled keep the dim of every tile they
                # read (a broadcast, a ``None`` / ``:`` view): the range was
                # taken over tiles the load masks.
                assert blocks <= {env.get_block_id(dim) for dim in index.shape}
                proven.add(position)
        tensor_dim += 1
    return frozenset(proven)
