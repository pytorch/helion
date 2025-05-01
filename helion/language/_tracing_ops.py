from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .._compiler.ast_extension import expr_from_string
from .._compiler.host_function import HostFunction
from .._compiler.tile_strategy import TileStrategy
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

"""
This file contains "fake" ops that cannot appear in user program but
are generated while compiling the user program. These ops are used to
generate code for certain constructs.
"""


@_decorators.api()
def _get_symnode(debug_name: str) -> int:
    """FX requires a torch.SymInt to come from an op. This is a fake op is added lazily to work around this."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_get_symnode)
def _(state: CodegenState) -> ast.AST:
    val = state.fx_node.meta["val"]
    assert isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)), val
    if (block_idx := TileStrategy.get_block_index(val)) is not None:
        if state.device_function.tile_strategy.block_size_var(block_idx) is None:
            # this should be unused
            return expr_from_string("block_size_var_optimized_away")
    return state.codegen.lift(
        expr_from_string(state.device_function.sympy_expr(val._sympy_())), dce=True
    )


@_decorators.api()
def _host_tensor(debug_name: str) -> torch.Tensor:
    """Source of a tensor that was allocated on the host and must be passed to the kernel as an arg."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_host_tensor)
def _(state: CodegenState) -> ast.AST:
    return expr_from_string("_host_tensor")  # should be unused


@has_side_effect
@_decorators.api()
def _for_loop(graph_id: int, args: list[object]) -> list[object]:
    """`for` loops are mapped to this op since FX does not support control flow."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_for_loop)
def _(state: CodegenState) -> None:
    return HostFunction.current().device_ir.graphs[state.proxy_arg(0)].codegen(state)


@_decorators.api()
def _phi(lhs: object, rhs: object) -> object:
    """Combine values from different branches of a control flow."""
    raise AssertionError("this should never be called")


@_decorators.register_fake(_phi)
def _(lhs: object, rhs: object) -> object:
    assert isinstance(lhs, torch.Tensor), lhs
    assert isinstance(rhs, torch.Tensor), rhs
    assert lhs.size() == rhs.size()
    assert lhs.dtype == rhs.dtype
    assert lhs.device == rhs.device
    return torch.empty_like(lhs)


@_decorators.codegen(_phi)
def _(state: CodegenState) -> ast.Name:
    lhs = state.ast_arg(0)
    assert isinstance(lhs, ast.Name), lhs
    rhs = state.ast_arg(1)
    assert isinstance(rhs, ast.Name), rhs
    state.device_function.merge_variable_names(lhs.id, rhs.id)
    return lhs


@_decorators.api()
def _inductor_lowering_extra(args: list[object]) -> torch.Tensor:
    """
    When we have an inductor lowering that results in multiple inductor
    buffers, we insert this fake op in the graph to represent intermediate
    values.
    """
    raise AssertionError("this should never be called")
