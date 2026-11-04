"""Metal-backend codegen for ops defined in ``helion.language._gelu_tanh_approx``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "metal")``
registrations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...language import _decorators
from ...language._gelu_tanh_approx import GELU_ERF_INV_SQRT2
from ...language._gelu_tanh_approx import GELU_TANH_APPROX_CUBIC
from ...language._gelu_tanh_approx import GELU_TANH_APPROX_KAPPA
from ...language._gelu_tanh_approx import _gelu_erf
from ...language._gelu_tanh_approx import _gelu_tanh_approx
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState

# ``.`` stands for MSL's ``::`` (the walker prints it back); the precise tanh
# and c10's erf are the ones Inductor's Metal overrides use for torch.tanh and
# torch.erf.
_GELU_TANH_APPROX_EXPR = (
    f"(0.5 * {{x}} * (1.0 + metal.precise.tanh({GELU_TANH_APPROX_KAPPA!r} * ({{x}}"
    f" + {GELU_TANH_APPROX_CUBIC!r} * ({{x}} * {{x}} * {{x}})))))"
)
_GELU_ERF_EXPR = (
    f"(0.5 * {{x}} * (1.0 + c10.metal.erf({{x}} * {GELU_ERF_INV_SQRT2!r})))"
)


def _gelu(state: CodegenState, template: str, prefix: str) -> ast.AST:
    """*template* over the input, evaluated in float as eager does for
    half and bfloat inputs, and rounded back to the input dtype once."""
    backend = CompileEnvironment.current().backend
    x = state.codegen.lift(state.ast_arg(0), dce=True, prefix=prefix).id
    proxy = state.proxy_args[0]
    narrow = isinstance(proxy, torch.Tensor) and proxy.dtype in (
        torch.float16,
        torch.bfloat16,
    )
    if narrow:
        x = state.codegen.lift(
            expr_from_string(backend.cast_expr(x, "float")),
            dce=True,
            prefix=f"{prefix}_fp32",
        ).id
    expr = template.replace("{x}", x)
    if narrow:
        assert isinstance(proxy, torch.Tensor)
        expr = backend.cast_expr(expr, backend.dtype_str(proxy.dtype))
    return expr_from_string(expr)


@_decorators.codegen(_gelu_tanh_approx, "metal")
def _(state: CodegenState) -> ast.AST:
    return _gelu(state, _GELU_TANH_APPROX_EXPR, "gelu_tanh_approx_in")


@_decorators.codegen(_gelu_erf, "metal")
def _(state: CodegenState) -> ast.AST:
    return _gelu(state, _GELU_ERF_EXPR, "gelu_erf_in")
