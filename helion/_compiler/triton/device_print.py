"""Triton-backend codegen for the ops defined in ``helion.language.device_print``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``device_print`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.device_print import device_print
from ..ast_extension import create
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


@_decorators.codegen(device_print, "triton")
def _(state: CodegenState) -> None:
    prefix = state.proxy_arg(0)
    call_args: list[ast.AST] = [create(ast.Constant, value=prefix)]

    # Handle varargs
    if len(state.proxy_args) > 1:
        assert len(state.ast_args) > 1
        # varargs are wrapped in a tuple, extract the elements
        ast_varargs = state.ast_args[1]
        assert isinstance(ast_varargs, (tuple, list)), (
            f"Expected tuple for varargs, got {type(ast_varargs)}"
        )
        call_args.extend(ast_varargs[0])

    call_expr = create(
        ast.Call,
        func=expr_from_string("tl.device_print"),
        args=call_args,
        keywords=[],
    )
    stmt = create(ast.Expr, value=call_expr)
    state.add_statement(stmt)
