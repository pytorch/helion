"""MetalOverrides — thin subclass of Inductor's MPS MetalOverrides.

Reuses Inductor's MetalOverrides for all expression generation (math ops,
casts, comparisons, etc.).  Its results are MSL text, which
``MetalBackend.parse_op_expression`` parses (``msl_expr``) into device-code
AST, so no op needs rewriting into Python syntax.

Overrides:
- ``_special_unary`` / ``_special_binary``: skip the ``V.kernel.headers``
  dependency (Helion includes the c10/metal headers unconditionally).
- ``exp`` / ``truediv``: under the kernel's ``fast_math`` setting, take
  ``metal::fast::exp`` and ``metal::fast::divide``, the operations the Triton
  backend makes approximate under the same setting.
- ``log1p`` / ``expm1``: Inductor's MPS overrides before pytorch#201580 have
  neither, so they would fall back to ``log(1 + x)`` and ``exp(x) - 1``, which
  lose every digit of a small ``x`` (``log1p(1e-8)`` is 0); later ones still
  take ``exp(x) - 1`` for ``|x| >= 0.1``.  log1p takes c10's accurate
  function, as torch's MPS kernels do; expm1 Kahan's formula.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.codegen.mps import MetalOverrides as _InductorMetalOverrides
from torch._inductor.virtualized import V
from torch._inductor.virtualized import ops

from ..compile_environment import CompileEnvironment
from ..inductor_lowering import _is_plain_true_division

if TYPE_CHECKING:
    from torch._inductor.codegen.common import CSEVariable


class MetalOverrides(_InductorMetalOverrides):
    """Helion Metal op overrides.

    Inherits all expression generation from Inductor's MetalOverrides.
    """

    def _special_unary(self, a: object, name: str) -> str:
        # Skip V.kernel.headers.add() — Helion includes c10/metal headers
        # unconditionally in the MSL preamble.
        return f"c10::metal::{name}({a})"

    def _special_binary(self, a: object, b: object, name: str) -> str:
        return f"c10::metal::{name}({a}, {b})"

    @staticmethod
    def exp(x: CSEVariable) -> str:
        if CompileEnvironment.current().settings.fast_math:
            return f"metal::fast::exp({x})"
        return _InductorMetalOverrides.exp(x)

    @staticmethod
    def log1p(x: CSEVariable) -> str:
        return f"c10::metal::log1p(static_cast<float>({x}))"

    @staticmethod
    def expm1(x: CSEVariable) -> str:
        # Kahan's expm1: (e^x - 1) * (x / log(e^x)) cancels the rounding error
        # of e^x, to a few ulp; x / log(e^x) is near 1, so the product only
        # overflows where e^x - 1 does.  Not c10::metal::expm1f, whose
        # argument reduction drops the low half of log(2): about 50-70 ulp off
        # at |x| > log(2) / 2.
        xf = ops.to_dtype(x, torch.float32)
        one = ops.constant(1.0, torch.float32)
        u = ops.exp(xf)
        um1 = ops.sub(u, one)
        result = ops.mul(um1, ops.truediv(xf, ops.log(u)))
        result = ops.where(ops.isinf(u), u, result)
        minus_one = ops.constant(-1.0, torch.float32)
        result = ops.where(ops.eq(um1, minus_one), um1, result)
        return ops.where(ops.eq(u, one), xf, result)

    @staticmethod
    def truediv(a: object, b: object) -> str:
        env = CompileEnvironment.current()
        node = V.current_node
        if not env.settings.fast_math or not _is_plain_true_division(node):
            return _InductorMetalOverrides.truediv(a, b)
        quotient = (
            f"metal::fast::divide(static_cast<float>({a}), static_cast<float>({b}))"
        )
        dtype = node.meta["val"].dtype
        if dtype == torch.float32:
            return quotient
        # MSL never converts float to bfloat implicitly, so a 16-bit quotient
        # is rounded back to its dtype here, as torch rounds it.
        return f"static_cast<{env.backend.dtype_str(dtype)}>({quotient})"


# Inductor binds erf, lgamma, the Bessel functions, ... to its own
# ``_special_unary`` / ``_special_binary`` with ``functools.partialmethod`` when
# its module loads, so the overrides above take effect only once the ops are
# bound again on this class.
MetalOverrides._initialize_special_ops()
