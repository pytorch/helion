"""Triton-backend codegen for ops defined in ``helion.language._gelu_tanh_approx``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``_gelu_tanh_approx`` imports it at the bottom so registration
keeps the same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.utils import triton_type

from ...language import _decorators
from ...language._gelu_tanh_approx import GELU_ERF_INV_SQRT2
from ...language._gelu_tanh_approx import GELU_TANH_APPROX_KAPPA
from ...language._gelu_tanh_approx import GELU_TANH_APPROX_LAMBDA
from ...language._gelu_tanh_approx import _gelu_erf
from ...language._gelu_tanh_approx import _gelu_tanh_approx
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


# The triton template is rendered as two layers: the inner ``{x32}``
# placeholder is fp32 (cast happens at the codegen-site for fp16 /
# bf16 inputs to satisfy ``libdevice.tanh``'s fp32-only contract);
# the outer ``{x}`` is the original-dtype value preserved for the
# leading ``0.5 * x`` factor that has no transcendental component.
# This keeps the fp32 round-trip narrow — only the ``tanh`` argument
# is cast — and matches Helion's existing ``tanh`` lowering shape
# (``inductor_lowering_extra.FP32_FALLBACK_OPS_UNARY``). For fp32
# inputs the codegen treats ``{x32}`` and ``{x}`` as the same local.
_GELU_TANH_APPROX_EXPR_TRITON = (
    f"(0.5 * ({{x}}) * (1.0 + libdevice.tanh(({{x32}}) * ({GELU_TANH_APPROX_KAPPA!r}"
    f" + {GELU_TANH_APPROX_LAMBDA!r} * ({{x32}}) * ({{x32}})))))"
)
_GELU_ERF_EXPR_TRITON = (
    f"(0.5 * ({{x}}) * (1.0 + libdevice.erf(({{x32}}) * {GELU_ERF_INV_SQRT2!r})))"
)


@_decorators.codegen(_gelu_tanh_approx, "triton")
def _(state: CodegenState) -> ast.AST:
    # Lift the input to a local so the four ``{x}`` references all
    # bind to the same Triton SSA name; without the lift, the
    # rendered expression would textually duplicate the inbound
    # expression four times (Triton compile-time CSE handles the
    # values, but generated source size grows quadratically with
    # chain depth, which we explicitly avoid in cute_epilogue.py too).
    input_ast = state.codegen.lift(
        state.ast_arg(0), dce=True, prefix="gelu_tanh_approx_in"
    )
    # ``libdevice.tanh`` is fp32-only on Triton/CUDA. For fp16 / bf16
    # inputs cast the *tanh argument* (``{x32}``) to fp32 (the
    # surrounding multiplies promote through to fp32 too) and then
    # narrow the whole result back to the input dtype before
    # returning, matching ``register_fake``'s ``torch.empty_like(x)``
    # contract. Mirrors the round-trip Helion installs for
    # ``aten.tanh.default`` via
    # ``inductor_lowering_extra.FP32_FALLBACK_OPS_UNARY``.
    proxy = state.proxy_args[0]
    orig_dtype: torch.dtype | None = None
    if isinstance(proxy, torch.Tensor) and proxy.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        orig_dtype = proxy.dtype
    if orig_dtype is not None:
        x32_local = state.codegen.lift(
            expr_from_string(f"{input_ast.id}.to(tl.float32)"),
            dce=True,
            prefix="gelu_tanh_approx_fp32",
        )
        x32_id = x32_local.id
    else:
        x32_id = input_ast.id
    expr = _GELU_TANH_APPROX_EXPR_TRITON.replace("{x}", input_ast.id).replace(
        "{x32}", x32_id
    )
    if orig_dtype is not None:
        # Narrow the fp32 polynomial result back to the input dtype so
        # the FX-level same-dtype contract holds for callers that do
        # not immediately follow with ``.to(x.dtype)``.
        expr = f"({expr}).to({triton_type(orig_dtype)})"
    return expr_from_string(expr)


@_decorators.codegen(_gelu_erf, "triton")
def _(state: CodegenState) -> ast.AST:
    input_ast = state.codegen.lift(state.ast_arg(0), dce=True, prefix="gelu_erf_in")
    proxy = state.proxy_args[0]
    orig_dtype: torch.dtype | None = None
    if isinstance(proxy, torch.Tensor) and proxy.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        orig_dtype = proxy.dtype
    if orig_dtype is not None:
        x32_local = state.codegen.lift(
            expr_from_string(f"{input_ast.id}.to(tl.float32)"),
            dce=True,
            prefix="gelu_erf_fp32",
        )
        x32_id = x32_local.id
    else:
        x32_id = input_ast.id
    expr = _GELU_ERF_EXPR_TRITON.replace("{x}", input_ast.id).replace("{x32}", x32_id)
    if orig_dtype is not None:
        expr = f"({expr}).to({triton_type(orig_dtype)})"
    return expr_from_string(expr)
