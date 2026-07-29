"""Pallas-backend codegen for ops defined in ``helion.language._gelu_tanh_approx``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "pallas")``
registrations; ``_gelu_tanh_approx`` imports it at the bottom so registration
keeps the same eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ... import exc
from ...language import _decorators
from ...language._gelu_tanh_approx import GELU_ERF_INV_SQRT2
from ...language._gelu_tanh_approx import _gelu_erf
from ...language._gelu_tanh_approx import _gelu_tanh_approx
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(_gelu_tanh_approx, "pallas")
def _(state: CodegenState) -> ast.AST:
    # Pallas does not have a ``cute.math.tanh`` analog wired through
    # Helion today; raise a structured ``BackendUnsupported`` so the
    # diagnostic is actionable rather than failing at codegen lookup
    # with a missing-implementation error. Users can spell
    # ``jax.nn.gelu(x, approximate=True)`` directly when targeting
    # Pallas.
    raise exc.BackendUnsupported(
        "pallas",
        "F.gelu(x, approximate='tanh') (cute and triton only)",
    )


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
