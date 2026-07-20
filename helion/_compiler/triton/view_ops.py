"""Triton-backend codegen for ops defined in ``helion.language.view_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``view_ops`` imports it at the bottom so registration keeps the
same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.view_ops import join
from ...language.view_ops import split
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(split, "triton")
def _(state: CodegenState) -> list[ast.AST]:
    split_call = expr_from_string("tl.split({tensor})", tensor=state.ast_arg(0))
    return [
        expr_from_string("{value}[0]", value=split_call),
        expr_from_string("{value}[1]", value=split_call),
    ]


@_decorators.codegen(join, "triton")
def _(state: CodegenState) -> ast.AST:
    return expr_from_string(
        "tl.join({tensor0}, {tensor1})",
        tensor0=state.ast_arg(0),
        tensor1=state.ast_arg(1),
    )
