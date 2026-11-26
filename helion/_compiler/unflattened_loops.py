"""Pin unflattened the loops of the tiles that ops run along."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx import Node

from ..language import atomic_ops
from ..language import memory_ops
from ..language import tile_ops
from ..language._tracing_ops import _for_loop
from ..language._tracing_ops import _for_loop_step
from ..language._tracing_ops import _if
from ..language._tracing_ops import _while_loop
from ..language.reduce_ops import _reduce
from ..language.scan_ops import _associative_scan
from .compile_environment import CompileEnvironment
from .inductor_lowering import ReductionLowering
from .tile_strategy import FlattenedTileStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .device_ir import DeviceIR
    from .device_ir import GraphInfo

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
    lowering a backend gave them.  A tile whose ``tile.block_size`` the
    kernel reads as a value is pinned as well (``block_size_value_block_ids``).
    """
    env = CompileEnvironment.current()
    for block_id in block_size_value_block_ids(device_ir.graphs):
        env.block_sizes[block_id].pin_unflattened()
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


# The memory ops whose second argument is a subscript list (a prefetch, which
# never changes a result, is left out).
_SUBSCRIPTED = (
    memory_ops.load,
    memory_ops.store,
    atomic_ops.atomic_add,
    atomic_ops.atomic_xchg,
    atomic_ops.atomic_and,
    atomic_ops.atomic_or,
    atomic_ops.atomic_xor,
    atomic_ops.atomic_max,
    atomic_ops.atomic_min,
    atomic_ops.atomic_cas,
)
# The control-flow ops, whose nested arguments hold loop bounds, steps and the
# inputs of their bodies, never shapes.
_CONTROL_FLOW = (_for_loop, _for_loop_step, _while_loop, _if)
# The tile ops take the tile itself, not its size as a value.
_TILE_TARGETS = (
    tile_ops.tile_index,
    tile_ops.tile_begin,
    tile_ops.tile_end,
    tile_ops.tile_count,
    tile_ops.tile_id,
)
_SYMBOLIC = (torch.SymInt, torch.SymFloat, torch.SymBool)


def block_size_value_block_ids(graphs: Sequence[GraphInfo]) -> set[int]:
    """Block ids whose block size a kernel reads as a value, directly or
    through symbolic arithmetic: an operand of a tensor op; an entry, or a
    slice bound, of a subscript; a loop bound or step; or an input of a loop
    or branch body.  The size comes from ``tile.block_size`` or from a
    value's own sizes (``v.numel()`` of a tile).  A flattened tile's dims
    share its one block size, so flattened over [4, 16], ``x[tile_m, tile_d]
    + tile_d.block_size`` would add the flat tile's ``_BLOCK_SIZE_0_1``, 64,
    where it adds 16, ``lookup[tile_d.block_size // 2]`` would read
    ``lookup[32]``, and ``hl.grid(tile_d.block_size)`` would run 64 times.
    A size that only shapes (``x.reshape([tile_n, 2, tile_d.block_size //
    2])``) is no value: a flattened tile reads shapes by the tile's own
    blocks.  (A subscript entry that is a block size alone reads as the tile
    it names.)  ``env.get_block_id`` also names the block of a grid index or
    of a tile edge (``tile.begin``) read as a value; neither has a flatten
    choice to pin (``_disable_flatten_get_tile``)."""
    env = CompileEnvironment.current()
    block_ids: set[int] = set()

    def nested_value(node: Node, user: Node) -> bool:
        # A use of ``node`` nested in an argument of ``user`` that reads it
        # as a value: an input or a bound of a control-flow op, or a scalar
        # index or slice bound of a subscript (not the tile a block size
        # alone names).
        if user.target in _CONTROL_FLOW:
            return True
        subscript = user.args[1] if len(user.args) > 1 else None
        if user.target not in _SUBSCRIPTED or not isinstance(subscript, (list, tuple)):
            return False
        return any(
            node in (entry.start, entry.stop, entry.step)
            if isinstance(entry, slice)
            else (
                entry is node
                and not isinstance(node.meta["val"].node.expr, sympy.Symbol)
            )
            for entry in subscript
        )

    def read_as_value(node: Node) -> bool:
        pending, seen = [node], {node}
        while pending:
            current = pending.pop()
            for user in current.users:
                if user.target in _TILE_TARGETS:
                    continue
                if current not in (*user.args, *user.kwargs.values()):
                    if nested_value(current, user):
                        return True
                    continue
                if isinstance(user.meta.get("val"), _SYMBOLIC):
                    # Symbolic arithmetic (``tile_d.block_size // 2``,
                    # ``torch.sym_int(tile_d.block_size / 2)``): its
                    # result's uses decide.
                    if user not in seen:
                        seen.add(user)
                        pending.append(user)
                    continue
                return True
        return False

    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            value = node.meta.get("val")
            if (
                node.op == "call_function"
                and isinstance(value, torch.SymInt)
                and read_as_value(node)
            ):
                for symbol in value.node.expr.free_symbols:
                    if (block_id := env.get_block_id(symbol)) is not None:
                        block_ids.add(env.canonical_block_id(block_id))
    return block_ids


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
