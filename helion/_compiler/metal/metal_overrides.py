"""MetalOverrides — thin subclass of Inductor's MPS MetalOverrides.

Reuses Inductor's MetalOverrides for all expression generation (math ops,
casts, comparisons, etc.).  The C++ namespace syntax (``::`` in expressions
like ``metal::precise::sin(x)``) is handled by replacing ``::`` with ``.``
before Python AST parsing, then converting back in the MSL walker.

Overrides:
- ``_special_unary`` / ``_special_binary``: skip the ``V.kernel.headers``
  dependency (Helion includes the c10/metal headers unconditionally).
- ``exp`` / ``truediv``: under the kernel's ``fast_math`` setting, take
  ``metal::fast::exp`` and ``metal::fast::divide``, the operations the Triton
  backend makes approximate under the same setting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.codegen.mps import MetalOverrides as _InductorMetalOverrides
from torch._inductor.virtualized import V

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
