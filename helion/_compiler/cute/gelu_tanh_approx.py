"""CuTe-backend codegen for ops defined in ``helion.language._gelu_tanh_approx``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``_gelu_tanh_approx`` imports it at the bottom so registration
keeps the same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...language import _decorators
from ...language._gelu_tanh_approx import _gelu_erf
from ...language._gelu_tanh_approx import _gelu_tanh_approx
from ...language._gelu_tanh_approx import epilogue_unary_step_template
from ...language._gelu_tanh_approx import gelu_erf_epilogue_unary_step_template
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(_gelu_tanh_approx, "cute", pure_fragment=True)
def _(state: CodegenState) -> ast.AST:
    # Lift once so repeated polynomial references share a single local.
    input_ast = state.codegen.lift(
        state.ast_arg(0), dce=True, prefix="gelu_tanh_approx_in"
    )
    return expr_from_string(epilogue_unary_step_template().format(inner=input_ast.id))


@_decorators.codegen(_gelu_erf, "cute", pure_fragment=True)
def _(state: CodegenState) -> ast.AST:
    input_ast = state.codegen.lift(state.ast_arg(0), dce=True, prefix="gelu_erf_in")
    return expr_from_string(
        gelu_erf_epilogue_unary_step_template().format(inner=input_ast.id)
    )
