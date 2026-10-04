"""Pallas-backend codegen for ops defined in ``helion.language._gelu_tanh_approx``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "pallas")``
registrations; ``_gelu_tanh_approx`` imports it at the bottom so registration
keeps the same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...language import _decorators
from ...language._gelu_tanh_approx import GELU_ERF_INV_SQRT2
from ...language._gelu_tanh_approx import GELU_TANH_APPROX_KAPPA
from ...language._gelu_tanh_approx import GELU_TANH_APPROX_LAMBDA
from ...language._gelu_tanh_approx import _gelu_erf
from ...language._gelu_tanh_approx import _gelu_tanh_approx
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(_gelu_tanh_approx, "pallas")
def _(state: CodegenState) -> ast.AST:
    # The tanh polynomial, in f32 for narrower floats (``lax.tanh`` lowers on
    # Mosaic), cast back to the input dtype as ``register_fake`` promises.
    input_ast = state.codegen.lift(
        state.ast_arg(0), dce=True, prefix="gelu_tanh_approx_in"
    )
    x = input_ast.id
    proxy = state.proxy_args[0]
    assert isinstance(proxy, torch.Tensor)
    narrow = proxy.dtype in (torch.float16, torch.bfloat16)
    if narrow:
        x = state.codegen.lift(
            expr_from_string(f"lax.convert_element_type({x}, jnp.float32)"),
            dce=True,
            prefix="gelu_tanh_approx_fp32",
        ).id
    expr = (
        f"(0.5 * {x} * (1.0 + lax.tanh({x} * ({GELU_TANH_APPROX_KAPPA!r}"
        f" + {GELU_TANH_APPROX_LAMBDA!r} * {x} * {x}))))"
    )
    if narrow:
        dtype = CompileEnvironment.current().backend.dtype_str(proxy.dtype)
        expr = f"lax.convert_element_type({expr}, {dtype})"
    return expr_from_string(expr)


@_decorators.codegen(_gelu_erf, "pallas")
def _(state: CodegenState) -> ast.AST:
    # ``jax.nn.gelu(x, approximate=False)`` lowers via ``lax.erfc`` which is
    # unimplemented in Pallas TPU's Mosaic lowering. Render the equivalent
    # ``erf``-based formula directly so the chain only references the
    # TPU-supported ``lax.erf`` primitive.
    input_ast = state.codegen.lift(state.ast_arg(0), dce=True, prefix="gelu_erf_in")
    expr = (
        f"(0.5 * ({input_ast.id}) * "
        f"(1.0 + lax.erf(({input_ast.id}) * {GELU_ERF_INV_SQRT2!r})))"
    )
    return expr_from_string(expr)
