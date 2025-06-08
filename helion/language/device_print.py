"""Device print support for Helion kernels."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import create
from .._compiler.ast_extension import expr_from_string
from . import _decorators

if TYPE_CHECKING:
    import torch

    from .._compiler.inductor_lowering import CodegenState


@has_side_effect
@_decorators.api()
def device_print(prefix: str, *values: torch.Tensor) -> None:
    """
    Print values from device code.

    :param prefix: A string prefix for the print statement
    :param values: Tensor values to print
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(device_print)
def _(prefix: str, *values: object) -> None:
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
        if isinstance(args[0], LiteralType):
            raise TypeError(
                f"First argument to print() must be a string prefix, got {type(args[0].value).__name__}"
            )
        raise TypeError(
            f"First argument to print() must be a string prefix, got {args[0]}"
        )

    # For compile-time values like tensor shapes, we should error out
    for i, arg in enumerate(args[1:]):
        if not isinstance(arg, TensorType):
            raise TypeError(
                f"device_print() only supports runtime tensor values. "
                f"Argument {i + 1} is {arg}, not a tensor. "
                f"Compile-time values like tensor shapes are not supported yet."
            )

    return NoType(origin=origin)


# pyre-fixme[56]
@_decorators.codegen(device_print)
def _(state: CodegenState) -> None:
    if len(state.proxy_args) < 1:
        raise ValueError("device_print requires at least a prefix argument")

    prefix = state.proxy_arg(0)
    if not isinstance(prefix, str):
        raise TypeError(f"device_print prefix must be a string, got {type(prefix)}")

    call_args = [create(ast.Constant, value=prefix)]

    # Handle varargs
    if len(state.proxy_args) > 1:
        assert len(state.ast_args) > 1
        ast_varargs = state.ast_args[1]
        assert len(ast_varargs) == 1, (  # pyre-fixme[6]
            "device_print varargs must be a single tuple"
        )
        call_args.extend(ast_varargs[0])  # pyre-fixme[16]

    call_expr = create(
        ast.Call,
        func=expr_from_string("tl.device_print"),
        args=call_args,
        keywords=[],
    )
    stmt = create(ast.Expr, value=call_expr)
    state.add_statement(stmt)
