"""FlyDSL-backend codegen for ops defined in ``helion.language.view_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "flydsl")``
registrations; ``_codegen_modules`` imports it so registration keeps the same
eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language.view_ops import subscript

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(subscript, "flydsl")
def _(state: CodegenState) -> ast.AST:
    # FlyDSL per-thread scalars: shape-only subscripts like [:, None] are no-ops.
    return state.ast_arg(0)
