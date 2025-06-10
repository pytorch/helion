from __future__ import annotations

import ast
import builtins
from typing import TYPE_CHECKING

from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import create
from .._compiler.ast_extension import expr_from_string
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState


@has_side_effect
@_decorators.device_func_replacement(builtins.print)
@_decorators.api(is_device_only=False)
def device_print(prefix: str, *values: object) -> None:
    """
    Print values from device code.

    :param prefix: A string prefix for the print statement
    :param values: Tensor values to print
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(device_print)
def _(*values: object, sep: str = " ", end: str = "\n") -> None:
    return None


@_decorators.type_propagation(device_print)
def _(*args: object, origin: object, **kwargs: object) -> object:
    from .._compiler.type_propagation import LiteralType
    from .._compiler.type_propagation import NoType
    from .._compiler.type_propagation import TensorType

    # Check that we have at least one argument (prefix)
    if len(args) == 0:
        raise ValueError("print() requires at least one argument (prefix)")

    # First argument must be the prefix string
    if not (isinstance(args[0], LiteralType) and isinstance(args[0].value, str)):
        raise TypeError(
            f"First argument to print() must be a string prefix, got {args[0]}"
        )

    # For compile-time values like tensor shapes, we should error out
    for i, arg in enumerate(args[1:]):
        if not isinstance(arg, TensorType):
            raise TypeError(
                f"print() only supports runtime tensor values. "
                f"Argument {i + 1} is {arg}, not a tensor. "
                f"Compile-time values like tensor shapes are not supported yet."
            )

    return NoType(origin=origin)


# pyre-fixme[56]
@_decorators.codegen(device_print)
def _(state: CodegenState) -> None:
    prefix = state.proxy_arg(0)
    call_args = [create(ast.Constant, value=prefix)]

    # Handle varargs
    if len(state.proxy_args) > 1:
        assert len(state.ast_args) > 1
        ast_varargs = state.ast_args[1]
        call_args.extend(ast_varargs[0])  # pyre-fixme[16]

    call_expr = create(
        ast.Call,
        func=expr_from_string("tl.device_print"),
        args=call_args,
        keywords=[],
    )
    stmt = create(ast.Expr, value=call_expr)
    state.add_statement(stmt)
