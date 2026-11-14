from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import TypeVar

import sympy
import torch
from torch.fx import has_side_effect
from torch.fx.experimental.sym_node import SymNode

from .._compiler.ast_extension import create
from .._compiler.ast_extension import expr_from_string
from .._compiler.ast_extension import statement_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.host_function import HostFunction
from .._compiler.variable_origin import BlockSizeOrigin
from ..exc import NotInsideKernel
from . import _decorators
from .tile_proxy import Tile

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState

    _T = TypeVar("_T", bound=object)

"""
This file contains "fake" ops that cannot appear in user program but
are generated while compiling the user program. These ops are used to
generate code for certain constructs.
"""

_symbolic_types = (torch.Tensor, torch.SymInt, torch.SymFloat, torch.SymBool)


def is_for_loop_target(target: object) -> bool:
    return target in (_for_loop, _for_loop_step)


def _val_to_sympy(val: torch.SymInt | torch.SymFloat | torch.SymBool) -> sympy.Expr:
    """Resolve a sym value to its sympy expression, preferring the cached node expr.

    A ``SymBool`` resolves to a sympy boolean rather than an ``Expr``; that case is
    not expected on the symnode codegen paths below but is accepted by the type.
    """
    sym_expr = getattr(getattr(val, "node", None), "_expr", None)
    if isinstance(sym_expr, sympy.Expr):
        return sym_expr
    # pyrefly: ignore [bad-return]
    return val._sympy_()


@_decorators.api()
def _get_symnode(debug_name: str) -> int:
    """FX requires a torch.SymInt to come from an op. This is a fake op is added lazily to work around this."""
    raise AssertionError("this should never be called")


# The value a ``_get_symnode`` capture prints when its traced value stays for
# a shape (``resolve_reduction_sizes``): the one computed from the extents.
EXTENT_VALUE = "helion_extent_value"


def symnode_value(node: torch.fx.Node) -> object:
    """The value a ``_get_symnode`` capture's code prints (``EXTENT_VALUE``)."""
    return node.meta.get(EXTENT_VALUE, node.meta["val"])


@_decorators.codegen(_get_symnode, "common")
def _(state: CodegenState) -> ast.AST:
    # pyrefly: ignore [bad-argument-type]
    val = symnode_value(state.fx_node)

    # A constant: an integer from the config (reduction_loops), or a value
    # computed from a reduction dim's extent (``resolve_reduction_sizes``).
    if isinstance(val, (int, float, bool)):
        return expr_from_string(state.device_function.literal_expr(val))

    assert isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)), val
    sym_expr = _val_to_sympy(val)
    host = HostFunction.current()
    origins = host.expr_to_origin
    origin_info = origins.get(sym_expr)
    # A nested graph can import a scalar registered in an outer device loop.
    # It has config provenance, but no replayable host origin. Preserve its
    # selected value and Python type instead of inventing a runtime argument.
    if origin_info is None:
        resolved = host.resolve_tunable_expr(sym_expr, state.config)
        if isinstance(resolved, (int, float, bool)):
            return expr_from_string(
                CompileEnvironment.current().backend.scalar_literal_expr(resolved)
            )
        if resolved != sym_expr:
            sym_expr = resolved
            if not sym_expr.free_symbols:
                # An integral zero can cancel the remaining runtime symbols.
                # Retain the original expression's scalar type in that case.
                selected = val.node.pytype(sym_expr)
                return expr_from_string(
                    CompileEnvironment.current().backend.scalar_literal_expr(selected)
                )
            origin_info = origins.get(sym_expr)

    if origin_info is not None and isinstance(origin_info.origin, BlockSizeOrigin):
        block_size_var = state.device_function.block_size_var(
            origin_info.origin.block_id
        )
        if block_size_var is None:
            return expr_from_string("1")
        return expr_from_string(block_size_var)
    return state.codegen.lift_symnode(
        expr_from_string(state.sympy_expr(sym_expr)),
        sym_expr,
        dce=True,
        prefix="symnode",
    )


@_decorators.api()
def _host_tensor(debug_name: str) -> torch.Tensor:
    """Source of a tensor that was allocated on the host and must be passed to the kernel as an arg."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_host_tensor, "common")
def _(state: CodegenState) -> ast.AST:
    return expr_from_string("_host_tensor")  # should be unused


@_decorators.api()
def _constant_tensor(value: float, dtype: torch.dtype) -> torch.Tensor:
    """
    Source of a constant scalar tensor created inside a kernel.
    This is generated when torch.tensor(val) is called inside a kernel.
    """
    raise AssertionError("this should never be called")


@_decorators.codegen(_constant_tensor, "common")
def _(state: CodegenState) -> ast.AST:
    value = state.proxy_arg(0)
    dtype = state.proxy_arg(1)
    assert isinstance(value, (int, float, bool))
    assert isinstance(dtype, torch.dtype)
    assert state.fx_node is not None
    shape = ["1"] * state.fx_node.meta["val"].ndim
    backend = CompileEnvironment.current().backend
    return expr_from_string(
        backend.full_expr(shape, backend.scalar_literal_expr(value), dtype)
    )


@has_side_effect
@_decorators.api()
def _for_loop(
    graph_id: int,
    begin: list[int],
    end: list[int],
    args: list[object],
) -> list[object]:
    """`for` loops are mapped to this op since FX does not support control flow."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_for_loop, "common")
def _(state: CodegenState) -> None:
    # pyrefly: ignore[bad-return]
    return state.get_graph(state.proxy_arg(0)).codegen(state)


@has_side_effect
@_decorators.api()
def _for_loop_step(
    graph_id: int,
    begin: list[int],
    end: list[int],
    args: list[object],
    step: list[int | None],
) -> list[object]:
    """Stepped ``for`` loops mapped into FX."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_for_loop_step, "common")
def _(state: CodegenState) -> None:
    # pyrefly: ignore[bad-return]
    return state.get_graph(state.proxy_arg(0)).codegen(state)


@_decorators.api()
def _pre_broadcast_tile(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    """Tile a pre-broadcast tensor along its last dim to match target_size."""
    raise AssertionError("this should never be called")


@_decorators.register_fake(_pre_broadcast_tile)
def _(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    new_shape = [*tensor.shape[:-1], target_size]
    return tensor.new_empty(new_shape)


@has_side_effect
@_decorators.api()
def _while_loop(
    cond_graph_id: int,
    body_graph_id: int,
    args: list[object],
    orelse_graph_id: int | None = None,
) -> list[object]:
    """Represent a while loop in FX since FX lacks native control flow."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_while_loop, "common")
def _(state: CodegenState) -> None:
    # pyrefly: ignore[bad-return]
    return state.get_graph(state.proxy_arg(1)).codegen(state)


@has_side_effect
@_decorators.api()
def _if(
    test: object,
    if_graph_id: int,
    else_graph_id: int,
    if_args: list[object],
    else_args: list[object],
) -> list[object]:
    """`for` loops are mapped to this op since FX does not support control flow."""
    raise AssertionError("this should never be called")


@_decorators.codegen(_if, "common")
def _(state: CodegenState) -> list[object]:
    return state.get_graph(state.proxy_arg(1)).codegen(state)


# Note we can't DCE phi nodes because there may be a loop carry dependency not captured in the outer graph
@has_side_effect
@_decorators.api(allow_host_tensor=True)
def _phi(lhs: object, rhs: object) -> object:
    """Combine values from different branches of a control flow."""
    raise AssertionError("this should never be called")


@_decorators.register_fake(_phi)
def _(lhs: object, rhs: object) -> object:
    if isinstance(lhs, Tile):
        assert isinstance(rhs, Tile)
        assert lhs.block_id == rhs.block_id
        return lhs
    if isinstance(lhs, torch.SymFloat) and isinstance(rhs, torch.SymFloat):
        shape_env = CompileEnvironment.current().shape_env
        with shape_env.ignore_fresh_unbacked_symbols():
            return shape_env.create_unbacked_symfloat()
    assert isinstance(lhs, torch.Tensor), lhs
    assert isinstance(rhs, torch.Tensor), rhs
    assert lhs.size() == rhs.size()
    assert lhs.dtype == rhs.dtype
    assert lhs.device == rhs.device
    result = torch.empty_like(lhs)
    HostFunction.current().compiler_state.record_ssa_copy(result, (lhs, rhs))
    return result


@_decorators.codegen(_phi, "common")
def _(state: CodegenState) -> ast.Name:
    lhs = state.ast_arg(0)
    assert isinstance(lhs, ast.Name), lhs
    rhs = state.ast_arg(1)
    assert isinstance(rhs, ast.Name), rhs
    state.device_function.merge_variable_names(lhs.id, rhs.id)
    return lhs


@_decorators.get_masked_value(_phi)
def _(node: torch.fx.Node) -> float | bool | None:
    lhs, rhs = node.args
    assert isinstance(lhs, torch.fx.Node)
    assert isinstance(rhs, torch.fx.Node)

    from .._compiler.node_masking import cached_masked_value

    lval = cached_masked_value(lhs)
    if lval is not None:
        rval = cached_masked_value(rhs)
        if lval == rval:
            return lval
    return None


@_decorators.api()
def _inductor_lowering_extra(args: list[object]) -> torch.Tensor:
    """
    When we have an inductor lowering that results in multiple inductor
    buffers, we insert this fake op in the graph to represent intermediate
    values.
    """
    raise AssertionError("this should never be called")


@_decorators.api()
def _and(left: object, right: object) -> object:
    raise NotInsideKernel


@_decorators.codegen(_and, "common")
def _(state: CodegenState) -> None:
    # pyrefly: ignore [bad-return]
    return expr_from_string(
        "{lhs} and {rhs}", lhs=state.ast_arg(0), rhs=state.ast_arg(1)
    )


@_decorators.register_fake(_and)
def _(left: object, right: object) -> object:
    if not isinstance(left, _symbolic_types):
        if not left:
            return left
        return right
    if not isinstance(right, _symbolic_types):
        if not right:
            return right
        return left
    env = CompileEnvironment.current()
    if isinstance(left, torch.SymBool) and isinstance(right, torch.SymBool):
        return torch.SymBool(
            SymNode(
                sympy.And(left._sympy_(), right._sympy_()),
                env.shape_env,
                bool,
                hint=None,
            )
        )
    # TODO(jansel): should match the type of the input
    with env.shape_env.ignore_fresh_unbacked_symbols():
        return env.shape_env.create_unbacked_symbool()


@_decorators.api()
def _or(left: object, right: object) -> object:
    raise NotInsideKernel


@_decorators.register_fake(_or)
def _(left: object, right: object) -> object:
    if not isinstance(left, _symbolic_types):
        if left:
            return left
        return right
    if not isinstance(right, _symbolic_types):
        if right:
            return right
        return left
    env = CompileEnvironment.current()
    if isinstance(left, torch.SymBool) and isinstance(right, torch.SymBool):
        return torch.SymBool(
            SymNode(
                sympy.Or(left._sympy_(), right._sympy_()),
                env.shape_env,
                bool,
                hint=None,
            )
        )
    with env.shape_env.ignore_fresh_unbacked_symbols():
        return env.shape_env.create_unbacked_symbool()


@_decorators.codegen(_or, "common")
def _(state: CodegenState) -> None:
    # pyrefly: ignore [bad-return]
    return expr_from_string(
        "{lhs} or {rhs}", lhs=state.ast_arg(0), rhs=state.ast_arg(1)
    )


@_decorators.api()
def _not(left: object) -> object:
    raise NotInsideKernel


@_decorators.register_fake(_not)
def _(left: object) -> object:
    if not isinstance(left, _symbolic_types):
        return not left
    env = CompileEnvironment.current()
    if isinstance(left, torch.SymBool):
        return torch.SymBool(
            SymNode(sympy.Not(left._sympy_()), env.shape_env, bool, hint=None)
        )
    with env.shape_env.ignore_fresh_unbacked_symbols():
        return env.shape_env.create_unbacked_symbool()


@_decorators.codegen(_not, "common")
def _(state: CodegenState) -> ast.AST:
    return expr_from_string(
        "not {lhs}",
        lhs=state.ast_arg(0),
    )


@_decorators.api()
def _mask_to(tensor: torch.Tensor, other: float | bool, /) -> torch.Tensor:
    """
    Set the masked out values of a given tile to a specific value.
    This operation is automatically generated by the compiler when doing a
    dot or reduction operation, and should not need to be called directly
    by users.

    Args:
        tensor: The tensor to apply the mask to.
        other: The value to set the masked out elements to.

    Returns:
        torch.Tensor: A tensor with the masked out elements set to `other`.
    """
    raise NotInsideKernel


@_decorators.register_fake(_mask_to)
def _(tensor: torch.Tensor, other: float) -> torch.Tensor:
    return torch.empty_like(tensor)


@_decorators.get_masked_value(_mask_to)
def _(node: torch.fx.Node) -> float | bool:
    value = node.args[1]
    assert isinstance(value, (int, float, bool))
    return value


@_decorators.api()
def _mask_select(
    tensor: torch.Tensor,
    other: torch.Tensor | float | bool,
    block_ids: tuple[int, ...],
    /,
) -> torch.Tensor:
    """
    ``tensor`` with the values outside the tiles of ``block_ids`` taken from
    ``other``, a tensor of the same shape or a scalar: ``_mask_to`` limited to
    some blocks.  The compiler applies it to the operands of a loop carry's
    update that read a partial tile of the loop, or to the whole update
    (``mask_loop_carries``), so that the tile's padded lanes add nothing to
    the carried value.

    Args:
        tensor: The tensor to apply the mask to.
        other: The tensor whose values the masked out elements take, or the
            value they all take.
        block_ids: The canonical block ids whose tile masks apply.

    Returns:
        torch.Tensor: ``tensor`` where those tile masks are true, else ``other``.
    """
    raise NotInsideKernel


@_decorators.register_fake(_mask_select)
def _(
    tensor: torch.Tensor,
    other: torch.Tensor | float | bool,
    block_ids: tuple[int, ...],
) -> torch.Tensor:
    return torch.empty_like(tensor)


@_decorators.get_masked_value(_mask_select)
def _(node: torch.fx.Node) -> float | bool | None:
    from .._compiler.node_masking import cached_masked_value

    # The masked lanes of other blocks keep the tensor's values.
    tensor, other = node.args[:2]
    assert isinstance(tensor, torch.fx.Node)
    value = cached_masked_value(tensor)
    if isinstance(other, torch.fx.Node):
        other = cached_masked_value(other)
    return value if value == other else None


@_decorators.api(allow_host_tensor=True)
def _new_var(value: _T, /) -> _T:
    """
    Create a shallow copy of a value that is assigned a fresh variable in codegen.

    This is used to ensure phi() node handling works properly when a value is renamed
    without mutation in a loop.  We need to copy the inputs to a loop so that phi nodes
    are handled properly.  Phi nodes will merge variable names from outside the loop,
    but the old value of those variables could have usages.
    """
    raise NotInsideKernel


@_decorators.register_fake(_new_var)
def _(value: _T) -> _T:
    if isinstance(value, torch.Tensor):
        result = torch.empty_like(value)
        HostFunction.current().compiler_state.record_ssa_copy(result, (value,))
        # pyrefly: ignore [bad-return]
        return result
    if isinstance(value, torch.SymInt):
        # pyrefly: ignore [bad-return]
        return CompileEnvironment.current().create_unbacked_symint()
    if isinstance(value, torch.SymFloat):
        shape_env = CompileEnvironment.current().shape_env
        with shape_env.ignore_fresh_unbacked_symbols():
            # pyrefly: ignore [bad-return]
            return shape_env.create_unbacked_symfloat()
    if isinstance(value, (int, float, bool)) or value is None:
        # pyrefly: ignore [bad-return]
        return value
    raise NotImplementedError(f"Unsupported type for _new_var: {type(value)}")


@_decorators.codegen(_new_var, "common")
def _(state: CodegenState) -> ast.AST:
    value = state.ast_arg(0)
    assert isinstance(value, ast.AST)
    varname = state.codegen.tmpvar(
        prefix=value.id if isinstance(value, ast.Name) else "new_var"
    )
    state.add_statement(statement_from_string(f"{varname} = {{expr}}", expr=value))
    return create(ast.Name, id=varname, ctx=ast.Load())


@_decorators.get_masked_value(_new_var)
def _(node: torch.fx.Node) -> float | bool | None:
    from .._compiler.node_masking import cached_masked_value

    (arg,) = node.args
    assert isinstance(arg, torch.fx.Node)
    return cached_masked_value(arg)
