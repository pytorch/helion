"""Loads whose masked-off lanes no store writes.

A masked load renders ``value if mask else zero``.  The zero matters to
consumers that see every lane of a tile: a reduction relies on it for its
padded lanes (``remove_unnecessary_masking`` drops the reduction's own
``_mask_to`` when the load's fill already is its identity), and so do dots,
scans and loop carries.  A load whose value reaches nothing but stores,
through the ops lowered as ``PointwiseLowering`` (inductor's pointwise ops:
arithmetic, comparisons, casts, math functions) and ``None``/``:`` views,
gives the zero to no one: each element a store writes is guarded by the
masks of the tiles its subscript indexes, and when that subscript indexes
every tile of the load at the element's own position, a lane whose load is
masked off is a lane no store writes.  A packet load -- one vector
transaction above the V-loop whose masked-off threads read an in-bounds
anchor -- then needs no per-lane select: whatever a masked-off lane holds is
never observed.

``load_fill_is_dead`` decides this on the device FX graph, position by
position (``out[tile_n, tile_m] = x[tile_m, tile_n]`` with equal block
sizes stores lane ``(i, j)`` of the load under the masks of ``(j, i)``, so
it is rejected), and fails closed on every other consumer, subscript form
or ``extra_mask``.  A limitation: that includes ``torch.where``, lowered as
an ``AtenLowering`` the analysis does not follow, although it reads its
inputs only at its own position as a pointwise op does; a masked fill such
as ``out = torch.where(m, x, y)`` keeps the select of each of its loads.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import torch

from ...language._tracing_ops import _get_symnode
from ...language.memory_ops import load
from ...language.memory_ops import store
from ...language.view_ops import subscript
from ..compile_environment import CompileEnvironment
from ..inductor_lowering import PointwiseLowering
from .emitters.memory_access import _cute_is_tile_scalar

if TYPE_CHECKING:
    from collections.abc import Sequence

# Per position of a value on the path: the load dim whose coordinate it is,
# or None (a broadcast position).
_Dims = list[int | None]


def _tile_blocks(index: object) -> list[int] | None:
    """The block id of each dim a subscript keeps, or None for any entry
    other than a tile index, a scalar of a block or an integer.  A scalar
    without a mask keeps no dim: an integer, or a symbol of a block other
    than its tile (``tile.begin``, ``tile.end``, ``tile.id``, an ``hl.grid``
    index), as the subscript's mask leaves it (``_cute_is_tile_scalar``).
    The ids are the masks' own (``env.get_block_id``).  The tiles of one
    ``hl.register_block_size`` share one id whatever their extents, so no id
    tells their subscripts apart; ``_shared_blocks`` rejects them."""
    if not isinstance(index, (list, tuple)):
        return None
    env = CompileEnvironment.current()
    blocks: list[int] = []
    for entry in index:
        if isinstance(entry, int):
            continue
        if not (isinstance(entry, torch.fx.Node) and entry.target is _get_symnode):
            return None
        value = entry.meta.get("val")
        if not isinstance(value, torch.SymInt):
            return None
        block_id = env.get_block_id(value)
        if block_id is None:
            return None
        if _cute_is_tile_scalar(env, value, block_id):
            continue
        blocks.append(block_id)
    return blocks


def _shared_blocks() -> set[int]:
    """The blocks more than one tile walks: a block size registered once and
    handed to two loops, or to two dimensions of one, gives their tiles one
    index symbol, so a subscript cannot tell them apart."""
    # The device IR imports this module's importers (memory_ops): bind at
    # call time.
    from ..device_ir import ForLoopGraphInfo
    from ..host_function import HostFunction

    device_ir = HostFunction.current().device_ir
    walked = [
        *device_ir.grid_block_ids,
        *(
            info.block_ids
            for info in device_ir.graphs
            if isinstance(info, ForLoopGraphInfo)
        ),
    ]
    counts = Counter(block_id for block_ids in walked for block_id in block_ids)
    return {block_id for block_id, count in counts.items() if count > 1}


def _rank(node: torch.fx.Node) -> int | None:
    value = node.meta.get("val")
    return value.ndim if isinstance(value, torch.Tensor) else None


def _aligned(dims: _Dims, rank: int) -> _Dims | None:
    """``dims`` broadcast to ``rank`` (trailing alignment)."""
    if len(dims) > rank:
        return None
    return [None] * (rank - len(dims)) + dims


def _viewed(dims: _Dims, index: Sequence[object]) -> _Dims | None:
    """``dims`` through ``subscript(value, index)`` with ``None`` and ``:``
    entries only."""
    out: _Dims = []
    rest = list(dims)
    for entry in index:
        if entry is None:
            out.append(None)
        elif entry == slice(None) and rest:
            out.append(rest.pop(0))
        else:
            return None
    return out + rest


def load_fill_is_dead(node: object) -> bool:
    """Whether ``node``, a ``load`` without ``extra_mask``, reaches only
    stores (as their value, through ``PointwiseLowering`` ops and
    ``None``/``:`` views) that index each of its tiles at the position its
    value has there, so no store writes a lane the load masks off (module
    docstring)."""
    if not (
        isinstance(node, torch.fx.Node)
        and node.target is load
        and len(node.args) >= 3
        and node.args[2] is None
    ):
        return False
    load_blocks = _tile_blocks(node.args[1])
    if (
        load_blocks is None
        or _rank(node) != len(load_blocks)
        or not _shared_blocks().isdisjoint(load_blocks)
    ):
        return False
    pending: list[tuple[torch.fx.Node, _Dims]] = [(node, list(range(len(load_blocks))))]
    seen: dict[torch.fx.Node, _Dims] = {}
    while pending:
        current, dims = pending.pop()
        if current in seen:
            if seen[current] != dims:
                return False
            continue
        seen[current] = dims
        for user in current.users:
            if user.op != "call_function":
                return False
            if user.target is store:
                # The value only: an index entry other than a tile index is
                # refused by ``_tile_blocks``.
                if (
                    len(user.args) < 3
                    or user.args[2] is not current
                    or user.args[0] is current
                    or (len(user.args) > 3 and user.args[3] is current)
                    or user.kwargs
                ):
                    return False
                store_blocks = _tile_blocks(user.args[1])
                if store_blocks is None:
                    return False
                region = _aligned(dims, len(store_blocks))
                if region is None or any(
                    dim is not None and store_blocks[position] != load_blocks[dim]
                    for position, dim in enumerate(region)
                ):
                    return False
                continue
            rank = _rank(user)
            if rank is None:
                return False
            if user.target is subscript:
                if user.args[0] is not current:
                    return False
                index = user.args[1]
                viewed = (
                    _viewed(dims, index) if isinstance(index, (list, tuple)) else None
                )
                if viewed is None or len(viewed) != rank:
                    return False
                pending.append((user, viewed))
            elif isinstance(user.meta.get("lowering"), PointwiseLowering):
                aligned = _aligned(dims, rank)
                if aligned is None:
                    return False
                pending.append((user, aligned))
            else:
                return False
    return True
