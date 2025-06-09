"""Device print support for Helion kernels."""

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
    import torch

    from .._compiler.inductor_lowering import CodegenState


def _fake_print(*args: object, **kwargs: object) -> None:
    """Fake print that doesn't actually print during tracing."""
    pass


@has_side_effect
@_decorators.builtin_replacement(builtins.print, host_fake=_fake_print)
@_decorators.api(is_device_only=True)
def device_print(*values: object, sep: str = " ", end: str = "\n") -> None:
    """
    Print values from device code.

    This function serves as both:
    1. A replacement for Python's builtin print() in device code
    2. The original device_print API (when called with a prefix string as first arg)

    :param values: Values to print (strings, numbers, tensors)
    :param sep: Separator between values (ignored in device code)
    :param end: End character (ignored in device code)
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(device_print)
def _(*values: object, sep: str = " ", end: str = "\n") -> None:
    return None


@_decorators.type_propagation(device_print)
def _(*args: object, origin: object, **kwargs: object) -> object:
    from .._compiler.type_propagation import NoType

    # Accept any arguments - we'll handle validation in codegen if needed
    # This allows both the original device_print API and Python's print()
    return NoType(origin=origin)


# pyre-fixme[56]
@_decorators.codegen(device_print)
def _(state: CodegenState) -> None:
    call_args = []
    
    # Check if this is being called as Python's print() or as device_print()
    # Python's print() will have a complex nested structure from *args/**kwargs handling
    # device_print() will have simple args
    
    if len(state.proxy_args) > 0 and isinstance(state.proxy_args[0], tuple):
        # This is Python's print() - extract from nested structure
        actual_args = state.proxy_args[0]
        if (
            isinstance(actual_args, tuple)
            and len(actual_args) > 0
            and isinstance(actual_args[0], tuple)
        ):
            # This is the print args
            print_args = actual_args[0]
        else:
            print_args = actual_args
            
        # Similarly for AST args
        if len(state.ast_args) > 0 and isinstance(state.ast_args[0], tuple):
            actual_ast_args = state.ast_args[0]
            if (
                isinstance(actual_ast_args, tuple)
                and len(actual_ast_args) > 0
                and isinstance(actual_ast_args[0], tuple)
            ):
                ast_print_args = actual_ast_args[0]
            else:
                ast_print_args = actual_ast_args
        else:
            ast_print_args = state.ast_args
            
        # Process the arguments
        if isinstance(print_args, tuple):
            for i, arg in enumerate(print_args):
                if isinstance(arg, str):
                    # String literal
                    call_args.append(create(ast.Constant, value=arg))
                elif isinstance(arg, (int, float, bool)):
                    # Other literal types - convert to string
                    call_args.append(create(ast.Constant, value=str(arg)))
                elif i < len(ast_print_args) and isinstance(ast_print_args[i], ast.AST):
                    # Use the AST representation for tensors
                    call_args.append(ast_print_args[i])
        elif isinstance(print_args, str):
            # Single string arg
            call_args.append(create(ast.Constant, value=print_args))
    else:
        # This is the original device_print API
        prefix = state.proxy_arg(0)
        call_args = [create(ast.Constant, value=prefix)]

        # Handle varargs
        if len(state.proxy_args) > 1:
            assert len(state.ast_args) > 1
            ast_varargs = state.ast_args[1]
            call_args.extend(ast_varargs[0])  # pyre-fixme[16]

    # If no args, print empty line
    if not call_args:
        call_args = [create(ast.Constant, value="")]

    call_expr = create(
        ast.Call,
        func=expr_from_string("tl.device_print"),
        args=call_args,
        keywords=[],
    )
    stmt = create(ast.Expr, value=call_expr)
    state.add_statement(stmt)


