"""Gathered coordinates whose bound the masks of their tiles imply, or the
load's own ``extra_mask`` states.

Role: general-purpose analysis of the memory lowering
(``cute_guarded_lanes``, ``cute_predicated_element_loads``).

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

``mask_bounded_gather_positions`` finds the term in the program's own mask
instead (``cute_predicated_element_loads``): a padded window masks each tap
with ``(col >= 0) & (col < width)`` and gathers through ``col``, so the mask
of the load holds ``col < width`` twice, once from the program and once as
the coordinate's bound.  Where the ``extra_mask`` is a conjunction with a
term ``coordinate < bound`` (or ``<=``) for the gathered coordinate itself
and a ``bound`` no larger than the size of the dim, the bound term repeats
it and the mask of the load goes without it: its test is then the program's
mask alone wherever the tile masks are gone too, which is what lets a
select under that mask give the load its fill
(``predicated_element_loads``).  The term compares the coordinate as an
int32, as the address reads it; the program's term compares it in its own
type, so the two agree only when the coordinate is an int32 or the mask
also keeps it at or above an int32 (``coordinate >= low``).  And the term
must speak of the lane the load reads.  The mask's comparison and the
subscript are ``None`` / ``:`` views of one value; the subscript puts each
tile the coordinate runs over on the axis of the load's value that has
that tile, which is one known axis when the tile is exactly one axis of
the value.  The comparison must sit on the same axis: each term is
followed from the mask through its ``&`` operands and views, dim by dim,
to the axis of the load's value it is broadcast against.  (Helion accepts
a mask over another tile of equal extent, ``ok[:, None]`` for a coordinate
along the last axis; its term is about another lane and states nothing.)
And the limit must be a value of the coordinate's type: the comparison
reads its constant in that type, so one outside it is another number there
(an int32 ``col < -2**31 - 1`` compares against ``2**31 - 1`` and holds
for every column but one; an int64 ``col >= 2**63`` holds always).
Anything else keeps the term: a mask that is not a conjunction of such
comparisons at its top (an ``|``, a ``~``, a comparison against a tensor),
a comparison of another value than the coordinate, a comparison on another
axis than the coordinate's, a limit outside the coordinate's type, an
int64 coordinate with no lower bound, a coordinate whose tile is two axes
of the load's value or none, a dim whose size is not static.
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
        if not _is_broadcast_view(value):
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


# The axis of the load's value that each dim of a mask operand is broadcast
# against.
_Axes = tuple[int, ...]


def _operand_axes(operand: object, axes: _Axes) -> _Axes | None:
    """The axes of ``operand``'s dims where it is broadcast against a
    tensor whose dims lie on ``axes`` (the trailing ones); None when it is
    no tensor node or has more dims."""
    value = operand.meta.get("val") if isinstance(operand, torch.fx.Node) else None
    if not isinstance(value, torch.Tensor) or value.ndim > len(axes):
        return None
    return axes[len(axes) - value.ndim :]


def _is_broadcast_view(node: torch.fx.Node) -> bool:
    """``value[None, :, None]``: a view that keeps every element of its
    operand and only adds unit dims."""
    index = node.args[1] if len(node.args) == 2 else None
    return (
        node.op == "call_function"
        and node.target is view_subscript
        and isinstance(index, (list, tuple))
        and all(entry is None or entry == slice(None) for entry in index)
    )


def _viewed_axes(view: torch.fx.Node, axes: _Axes) -> _Axes | None:
    """The axes of the dims of the operand of the broadcast ``view`` whose
    own dims lie on ``axes``: a ``:`` passes its axis on, a ``None`` adds a
    unit dim, and dims past the subscript keep theirs."""
    index = view.args[1]
    assert isinstance(index, (list, tuple))
    if len(index) > len(axes):
        return None
    kept = tuple(
        axis for entry, axis in zip(index, axes, strict=False) if entry is not None
    )
    kept += axes[len(index) :]
    value = (
        view.args[0].meta.get("val")
        if isinstance(view.args[0], torch.fx.Node)
        else None
    )
    if not isinstance(value, torch.Tensor) or value.ndim != len(kept):
        return None
    return kept


def _conjuncts(mask: object, value_ndim: int) -> list[tuple[torch.fx.Node, _Axes]]:
    """The terms of ``mask`` as a conjunction, the operands of every ``&``
    at its top through ``None`` / ``:`` views (an ``&`` that holds has two
    operands that hold, whatever their type), each with the axes of a
    load's value of ``value_ndim`` dims that its dims are broadcast
    against."""
    terms: list[tuple[torch.fx.Node, _Axes]] = []
    pending = [(mask, _operand_axes(mask, tuple(range(value_ndim))))]
    while pending:
        node, axes = pending.pop()
        if axes is None or not (
            isinstance(node, torch.fx.Node) and node.op == "call_function"
        ):
            continue
        if node.target is view_subscript:
            if _is_broadcast_view(node):
                pending.append((node.args[0], _viewed_axes(node, axes)))
            continue
        if (
            node.target
            in (torch.ops.aten.bitwise_and.Tensor, torch.ops.aten.logical_and.default)
            and len(node.args) == 2
            and not node.kwargs
        ):
            pending.extend(
                (operand, _operand_axes(operand, axes)) for operand in node.args
            )
            continue
        terms.append((node, axes))
    return terms


def _viewed(node: object) -> object:
    """``node`` without the ``None`` / ``:`` views around it."""
    while isinstance(node, torch.fx.Node) and _is_broadcast_view(node):
        node = node.args[0]
    return node


def _coordinate_axes(node: object, axes: _Axes, coordinate: object) -> _Axes | None:
    """The axes of ``coordinate``'s dims when ``node``, whose dims lie on
    ``axes``, is ``coordinate`` under ``None`` / ``:`` views; None when it
    is not."""
    while node is not coordinate:
        if not (isinstance(node, torch.fx.Node) and _is_broadcast_view(node)):
            return None
        viewed = _viewed_axes(node, axes)
        if viewed is None:
            return None
        node, axes = node.args[0], viewed
    return axes


def _stated_range(
    mask: object,
    value_ndim: int,
    coordinate: object,
    lanes: dict[int, int],
    dtype: torch.dtype,
) -> tuple[int | None, int | None]:
    """``(low, bound)``: the largest ``low`` and smallest ``bound`` for which
    ``mask`` has a term ``coordinate >= low`` and ``coordinate < bound``
    (``>`` and ``<=`` against an integer count as the next one) on the
    coordinate's own lanes: ``lanes`` maps each dim of ``coordinate`` that
    runs over a tile to the axis of the load's value (of ``value_ndim``
    dims) the subscript puts it on, and a term whose comparison sits on
    another axis is not counted.  Nor is a term whose integer the
    coordinate's ``dtype`` does not hold: the comparison reads it in that
    type, as another number.  None for a side the mask does not state."""
    held = torch.iinfo(dtype)
    low: int | None = None
    bound: int | None = None
    for term, axes in _conjuncts(mask, value_ndim):
        if len(term.args) != 2 or term.kwargs or type(term.args[1]) is not int:
            continue
        # A comparison with an integer has the dims of its tensor operand.
        operand_axes = _operand_axes(term.args[0], axes)
        if operand_axes is None:
            continue
        placed = _coordinate_axes(term.args[0], operand_axes, coordinate)
        if placed is None or any(placed[dim] != axis for dim, axis in lanes.items()):
            continue
        limit = term.args[1]
        if not held.min <= limit <= held.max:
            continue
        if term.target in (torch.ops.aten.lt.Scalar, torch.ops.aten.le.Scalar):
            if term.target is torch.ops.aten.le.Scalar:
                limit += 1
            bound = limit if bound is None else min(bound, limit)
        elif term.target in (torch.ops.aten.ge.Scalar, torch.ops.aten.gt.Scalar):
            if term.target is torch.ops.aten.gt.Scalar:
                limit += 1
            low = limit if low is None else max(low, limit)
    return low, bound


def _coordinate_lanes(
    coordinate: object, value_tiles: Sequence[int | None]
) -> dict[int, int] | None:
    """Each dim of ``coordinate`` that runs over a tile, with the axis of
    the load's value that has that tile (``value_tiles``: the tile of each
    axis); None when a dim is neither a view's literal unit dim nor a tile
    that is exactly one axis of the value.  (A symbolic dim is not compared
    with one, which would record a guard.)"""
    value = (
        coordinate.meta.get("val") if isinstance(coordinate, torch.fx.Node) else None
    )
    if not isinstance(value, torch.Tensor):
        return None
    env = CompileEnvironment.current()
    lanes: dict[int, int] = {}
    for dim, size in enumerate(value.shape):
        if isinstance(size, int) and size == 1:
            continue
        tile = env.get_block_id(size)
        if tile is None or value_tiles.count(tile) != 1:
            return None
        lanes[dim] = value_tiles.index(tile)
    return lanes


def mask_bounded_gather_positions(
    state: CodegenState, tensor: torch.Tensor, subscript: Sequence[object]
) -> frozenset[int]:
    """The positions of ``subscript`` whose gathered coordinate the load's
    own ``extra_mask`` keeps below the size of its dim of ``tensor`` (module
    docstring); empty without a mask or the load's FX node."""
    fx_node = state.fx_node
    if fx_node is None or len(fx_node.args) < 3 or fx_node.args[2] is None:
        return frozenset()
    fx_index = fx_node.args[1]
    loaded = fx_node.meta.get("val")
    if (
        not isinstance(fx_index, (list, tuple))
        or len(fx_index) != len(subscript)
        or not isinstance(loaded, torch.Tensor)
    ):
        return frozenset()
    env = CompileEnvironment.current()
    # The tile of each axis of the load's value.
    value_tiles = [env.get_block_id(dim) for dim in loaded.shape]
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
            coordinate = _viewed(fx_entry)
            lanes = _coordinate_lanes(coordinate, value_tiles)
            if isinstance(size, int) and lanes is not None:
                low, bound = _stated_range(
                    fx_node.args[2], loaded.ndim, coordinate, lanes, index.dtype
                )
                if (
                    bound is not None
                    and bound <= size
                    # The address reads the coordinate as an int32: the
                    # stated range is the coordinate's own only where that
                    # changes nothing.
                    and (
                        index.dtype is torch.int32
                        or (
                            low is not None
                            and low >= _INT32_MIN
                            and bound <= _INT32_MAX
                        )
                    )
                ):
                    proven.add(position)
        tensor_dim += 1
    return frozenset(proven)
