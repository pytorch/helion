"""CuTe-backend codegen for ops defined in ``helion.language._gelu_tanh_approx``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``_gelu_tanh_approx`` imports it at the bottom so registration
keeps the same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...language import _decorators
from ...language._gelu_tanh_approx import _gelu_erf
from ...language._gelu_tanh_approx import _gelu_tanh_approx
from ...language._gelu_tanh_approx import epilogue_unary_step_template
from ...language._gelu_tanh_approx import gelu_erf_epilogue_unary_step_template
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


def _half_input_dtype(state: CodegenState) -> torch.dtype | None:
    proxy = state.proxy_args[0]
    if isinstance(proxy, torch.Tensor) and proxy.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return proxy.dtype
    return None


def _render(state: CodegenState, template: str, prefix: str) -> ast.AST:
    """Render ``template`` on the lifted input, computing a 16-bit input's
    GELU in fp32 and rounding it to the input dtype once, as eager does.
    """
    backend = CompileEnvironment.current().backend
    half_dtype = _half_input_dtype(state)
    input_ast = state.ast_arg(0)
    if half_dtype is not None:
        input_ast = backend.cast_ast(input_ast, torch.float32)
    # Same lift-to-single-local rationale as the triton path: see the
    # module docstring and :class:`Tcgen05UnaryEpilogueChain`
    # (``cute_epilogue.py``).
    inner = state.codegen.lift(input_ast, dce=True, prefix=prefix)
    result = expr_from_string(template.format(inner=inner.id))
    if half_dtype is not None:
        result = backend.cast_ast(result, half_dtype)
    return result


@_decorators.codegen(_gelu_tanh_approx, "cute")
def _(state: CodegenState) -> ast.AST:
    return _render(state, epilogue_unary_step_template(), "gelu_tanh_approx_in")


@_decorators.codegen(_gelu_erf, "cute")
def _(state: CodegenState) -> ast.AST:
    return _render(state, gelu_erf_epilogue_unary_step_template(), "gelu_erf_in")
