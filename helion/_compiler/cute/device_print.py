"""CuTe-backend codegen for the ops defined in ``helion.language.device_print``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
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


@_decorators.codegen(device_print, "cute")
def _(state: CodegenState) -> None:
    prefix = state.proxy_arg(0)
    assert isinstance(prefix, str)

    value_asts: list[ast.AST] = []
    if len(state.proxy_args) > 1:
        assert len(state.ast_args) > 1
        ast_varargs = state.ast_args[1]
        assert isinstance(ast_varargs, (tuple, list)), (
            f"Expected tuple for varargs, got {type(ast_varargs)}"
        )
        value_asts.extend(ast_varargs[0])

    fmt = prefix + " ".join(["{}"] * len(value_asts))
    call_args: list[ast.AST] = [create(ast.Constant, value=fmt), *value_asts]
    call_expr = create(
        ast.Call,
        func=expr_from_string("cute.printf"),
        args=call_args,
        keywords=[],
    )
    stmt = create(ast.Expr, value=call_expr)
    state.add_statement(stmt)
