"""Prove that a CuTe branch condition is the same for every thread of a CTA.

A block-wide barrier (``cute.arch.sync_threads()``) deadlocks in a branch some
threads of the CTA skip, so the barrier passes treat a branch as divergent
unless its condition is proven uniform here: then every thread takes the same
side and a barrier inside it is convergent.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch

from ...language import _tracing_ops
from ...language import tile_ops
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..variable_origin import BlockSizeOrigin
from ..variable_origin import GridOrigin
from ..variable_origin import TileBeginOrigin
from ..variable_origin import TileEndOrigin
from ..variable_origin import TileIdOrigin

if TYPE_CHECKING:
    import sympy

# Python scalar operations: uniform operands give a uniform result.
_SCALAR_OPS = frozenset(
    {
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.neg,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.eq,
        operator.ne,
        operator.and_,
        operator.or_,
        operator.xor,
        operator.not_,
        _tracing_ops._and,
        _tracing_ops._or,
        _tracing_ops._not,
        _tracing_ops._new_var,
    }
)


def block_uniform(value: object) -> bool:
    """Whether every thread of a CTA computes the FX value ``value`` alike.

    A positive proof that fails closed: literals, block sizes, host scalars
    (tensor sizes, kernel arguments), the index of a top-level ``hl.grid``,
    the ``begin``/``end``/``id`` of a grid tile (one tile per CTA; these ops
    disable flattening, so the tile offset never comes from a thread's flat
    index) and the Python scalar arithmetic, comparisons and Boolean
    operations over them.  Anything else may differ between threads: a
    loaded or tensor value, a thread's tile index or ``hl.arange`` element,
    a device loop's tile, a placeholder (a loop-carried value may change
    after the first iteration) or an unknown call.
    """
    if isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, (torch.SymInt, torch.SymBool)):
        return _symbolic_block_uniform(value)
    if not isinstance(value, torch.fx.Node) or value.op != "call_function":
        return False
    if value.target is _tracing_ops._get_symnode:
        return _symbolic_block_uniform(value.meta["val"])
    if value.target in (tile_ops.tile_begin, tile_ops.tile_end, tile_ops.tile_id):
        (tile,) = value.args
        return (
            isinstance(tile, torch.fx.Node)
            and tile.target is _tracing_ops._get_symnode
            and _is_grid_block(
                CompileEnvironment.current().get_block_id(tile.meta["val"])
            )
        )
    if value.target in _SCALAR_OPS:
        return not value.kwargs and all(block_uniform(arg) for arg in value.args)
    return False


def _symbolic_block_uniform(value: object) -> bool:
    if isinstance(value, (bool, int, float)):
        return True
    if not isinstance(value, (torch.SymInt, torch.SymBool)):
        return False
    return all(
        _symbol_block_uniform(symbol)
        for symbol in _tracing_ops._val_to_sympy(value).free_symbols
    )


def _symbol_block_uniform(symbol: sympy.Basic) -> bool:
    origin_info = HostFunction.current().expr_to_origin.get(symbol)
    if origin_info is None:
        return False
    origin = origin_info.origin
    if isinstance(origin, BlockSizeOrigin) or origin.is_host():
        return True
    # An ``hl.grid`` index (block size 1 along every dim, never flattened) or
    # a grid tile's begin/end/id.
    return (
        type(origin) is GridOrigin
        or isinstance(origin, (TileBeginOrigin, TileEndOrigin, TileIdOrigin))
    ) and _is_grid_block(origin.block_id)


def _is_grid_block(block_id: int | None) -> bool:
    return block_id is not None and any(
        block_id in block_ids
        for block_ids in HostFunction.current().device_ir.grid_block_ids
    )
