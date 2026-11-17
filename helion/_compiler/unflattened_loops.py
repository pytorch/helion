"""Pin unflattened the loops of the tiles that ops run along."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ..language.reduce_ops import _reduce
from ..language.scan_ops import _associative_scan
from .compile_environment import CompileEnvironment
from .inductor_lowering import ReductionLowering
from .tile_strategy import FlattenedTileStrategy

if TYPE_CHECKING:
    from .device_ir import DeviceIR

# ATen ops along one dim of their first argument: the position of that dim
# among the arguments (None where it is keyword-only) and its default (None
# for every dim).
_ATEN_OPS_ALONG_A_DIM: dict[object, tuple[int | None, int | None]] = {
    torch.ops.aten.sort.default: (1, -1),
    torch.ops.aten.sort.stable: (None, -1),
    torch.ops.aten.topk.default: (2, -1),
    torch.ops.aten.gather.default: (1, None),
    torch.ops.aten.argmax.default: (1, None),
    torch.ops.aten.argmin.default: (1, None),
}


def pin_unflattened_loops(device_ir: DeviceIR) -> None:
    """Pin unflattened the loop of each tile an op runs along
    (``BlockSizeInfo.pin_unflattened``).

    A flattened tile's dims share one index
    (``Backend.flattened_tile_shares_one_index``), so none of them can be the
    dim of a reduction (``ReductionLowering.block_index``), a sort, top-k,
    gather, argmin or argmax, an ``hl.associative_scan`` or an ``hl.reduce``
    along a dim.  An ``hl.reduce`` over every dim pins all its tile dims where
    the backend reduces one dim at a time
    (``Backend.reduces_every_dim_one_at_a_time``), where its result keeps the
    reduced dims, whose size-1 dims cannot stand for the tile's single one,
    and in a kernel whose flattened tiles keep a dim per block over their one
    index (``FlattenedTileStrategy.compacts_tiles``), where the value holds
    each element once per lane of the other dims.  The ops are found by
    their targets once the graphs' lowerings are prepared, whichever
    lowering a backend gave them.
    """
    env = CompileEnvironment.current()
    compacted = FlattenedTileStrategy.compacts_tiles(device_ir)
    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function":
                continue
            lowering = node.meta.get("lowering")
            if isinstance(lowering, ReductionLowering):
                env.block_sizes[lowering.block_index].pin_unflattened()
            for size in _extents_along(node, compacted):
                env.pin_unflattened_along(size)


def _extents_along(node: Node, compacted: bool) -> list[int | torch.SymInt]:
    """The extents of the dims ``node`` runs along, if it is one of the ops
    ``pin_unflattened_loops`` pins for (``compacted``: whether a flattened
    tile is one dimension of a value).  An op whose lowering reads nothing
    of its operand runs along nothing: an argmax over a dim of size 1 is a
    constant, and ``strip_unused_inputs`` replaced its operand with None."""
    if node.target in _ATEN_OPS_ALONG_A_DIM:
        position, default = _ATEN_OPS_ALONG_A_DIM[node.target]
        dim = (
            node.args[position]
            if position is not None and len(node.args) > position
            else node.kwargs.get("dim", default)
        )
        value = _first_tensor(node.args[0])
    elif node.target is _reduce or node.target is _associative_scan:
        # (combine_graph_id, input, dim, keep_dims or reverse, is_tuple_input)
        dim = node.args[2]
        value = _first_tensor(node.args[1])
        if (
            dim is None
            and compacted
            and not (
                node.args[3]
                or CompileEnvironment.current().backend.reduces_every_dim_one_at_a_time()
            )
        ):
            # An hl.reduce over every dim that reduces the whole value at once.
            return []
    else:
        return []
    if value is None:
        return []
    sizes = value.size()
    if dim is None:
        return [*sizes]
    assert isinstance(dim, int)
    return [sizes[dim]]


def _first_tensor(arg: object) -> torch.Tensor | None:
    """The value of ``arg``, a node or (for ``hl.reduce`` and
    ``hl.associative_scan`` over a tuple) a sequence of them: the first, or
    None for an operand the lowering stripped."""
    if isinstance(arg, (tuple, list)):
        arg = arg[0]
    if arg is None:
        return None
    assert isinstance(arg, Node)
    value = arg.meta["val"]
    assert isinstance(value, torch.Tensor)
    return value
