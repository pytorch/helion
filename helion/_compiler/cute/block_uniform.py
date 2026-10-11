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
from ...language import memory_ops
from ...language import tile_ops
from ...language.atomic_ops import ATOMIC_OPS
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..loop_dependency_checker import HOST_UNKNOWN_ROOT
from ..loop_dependency_checker import collect_host_tensor_roots
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
# Elementwise operations on 0-d tensors (a loaded flag compared to a constant).
_TENSOR_OPS = frozenset(
    {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.lt.Scalar,
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.gt.Scalar,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.ge.Scalar,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.eq.Scalar,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.ne.Scalar,
        torch.ops.aten.ne.Tensor,
        torch.ops.aten.logical_and.default,
        torch.ops.aten.logical_or.default,
        torch.ops.aten.logical_not.default,
        torch.ops.prims.convert_element_type.default,
    }
)


def block_uniform(value: object) -> bool:
    """Whether every thread of a CTA computes the FX value ``value`` alike.

    A positive proof that fails closed: literals, block sizes, host scalars
    (tensor sizes, kernel arguments), the index of a top-level ``hl.grid``,
    the ``begin``/``end``/``id`` of a grid tile (one tile per CTA; these ops
    disable flattening, so the tile offset never comes from a thread's flat
    index), a load at such an index of a tensor nothing in the kernel can
    write (``_uniform_load``), and the scalar arithmetic, comparisons and
    Boolean operations over them.  Anything else may differ between
    threads: any other loaded or tensor value, a thread's tile index or
    ``hl.arange`` element, a device loop's tile, a placeholder (a
    loop-carried value may change after the first iteration) or an unknown
    call.
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
    if value.target is memory_ops.load:
        return _uniform_load(value)
    if value.target in _SCALAR_OPS or value.target in _TENSOR_OPS:
        return all(
            block_uniform(arg) for arg in [*value.args, *value.kwargs.values()]
        ) or (
            # A conversion's dtype argument.
            value.target is torch.ops.prims.convert_element_type.default
            and block_uniform(value.args[0])
        )
    return False


def _uniform_load(load: torch.fx.Node) -> bool:
    """Whether every thread issues ``load`` at one element no thread can write.

    The index is literals and uniform scalars only (a tile in it is the
    thread's own index), there is no extra mask, and no store or atomic in
    any graph of the kernel can target the tensor's storage: every written
    tensor's host value is provably distinct by its allocation roots
    (``collect_host_tensor_roots``: a fresh allocation is apart from the
    arguments, while all arguments share one root since the caller may pass
    one tensor twice), or the two storages are inputs the bound kernel's
    cache-key disjointness fact proves apart
    (``runtime_tensors_are_proven_disjoint``; a host view counts as the
    input whose storage it shares).  Then every thread reads the same value
    whenever it loads it.  A distributed kernel is refused: a peer may write
    the tensor during the kernel.
    """
    from .memory_ops import runtime_tensors_are_proven_disjoint
    from .view_ops import _host_tensor_root_name

    tensor, index, extra_mask, *_ = load.args
    if (
        extra_mask is not None
        or not isinstance(tensor, torch.fx.Node)
        or tensor.target is not _tracing_ops._host_tensor
        or not isinstance(index, (list, tuple))
        or not all(_uniform_index(item) for item in index)
    ):
        return False
    env = CompileEnvironment.current()
    if env.process_group_name is not None:
        return False
    host_function = HostFunction.current()
    roots = collect_host_tensor_roots(
        host_function.body, set(host_function.params.arguments)
    )
    unknown = frozenset({HOST_UNKNOWN_ROOT})

    def tensor_roots(node: torch.fx.Node) -> frozenset[str]:
        name = node.args[0]
        assert isinstance(name, str)
        return roots.get(_host_tensor_root_name(name), unknown)

    input_storages = {id(value.untyped_storage()): value for value in env.input_sources}

    def storage_input(node: torch.fx.Node) -> torch.Tensor | None:
        return input_storages.get(id(node.meta["val"].untyped_storage()))

    loaded = tensor_roots(tensor)
    loaded_input = storage_input(tensor)
    for graph_info in host_function.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or (
                node.target is not memory_ops.store and node.target not in ATOMIC_OPS
            ):
                continue
            target = node.args[0]
            if not (
                isinstance(target, torch.fx.Node)
                and target.target is _tracing_ops._host_tensor
            ):
                return False
            written = tensor_roots(target)
            if HOST_UNKNOWN_ROOT not in loaded | written and not loaded & written:
                continue
            written_input = storage_input(target)
            if (
                loaded_input is None
                or written_input is None
                or not runtime_tensors_are_proven_disjoint(
                    env, loaded_input, written_input
                )
            ):
                return False
    return True


def _uniform_index(item: object) -> bool:
    """Whether a subscript element names one element for every thread."""
    if isinstance(item, int):
        return True
    if not isinstance(item, torch.fx.Node) or not block_uniform(item):
        return False
    # A tile in a subscript is the thread's own index, not its block size.
    return not (
        item.target is _tracing_ops._get_symnode
        and isinstance(_symbol_origin(item.meta["val"]), BlockSizeOrigin)
    )


def _symbolic_block_uniform(value: object) -> bool:
    if isinstance(value, (bool, int, float)):
        return True
    if not isinstance(value, (torch.SymInt, torch.SymBool)):
        return False
    return all(
        _symbol_block_uniform(symbol)
        for symbol in _tracing_ops._val_to_sympy(value).free_symbols
    )


def _symbol_origin(value: object) -> object:
    if not isinstance(value, (torch.SymInt, torch.SymBool)):
        return None
    origin_info = HostFunction.current().expr_to_origin.get(
        _tracing_ops._val_to_sympy(value)
    )
    return None if origin_info is None else origin_info.origin


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
