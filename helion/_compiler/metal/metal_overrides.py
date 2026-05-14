"""MetalOverrides — thin subclass of Inductor's MPS MetalOverrides.

Reuses Inductor's MetalOverrides for all expression generation (math ops,
casts, comparisons, etc.).  The C++ namespace syntax (``::`` in expressions
like ``metal::precise::sin(x)``) is handled by replacing ``::`` with ``.``
before Python AST parsing, then converting back in the MSL walker.

Overrides:
- ``_special_unary`` / ``_special_binary``: skip the ``V.kernel.headers``
  dependency (Helion includes the c10/metal headers unconditionally).
- ``where``: emit Python ternary (``a if cond else b``) instead of C++
  ternary (``cond ? a : b``) so it can be parsed as Python AST.
"""

from __future__ import annotations

from torch._inductor.codegen.mps import MetalOverrides as _InductorMetalOverrides


class MetalOverrides(_InductorMetalOverrides):
    """Helion Metal op overrides.

    Inherits all expression generation from Inductor's MetalOverrides.
    """

    @staticmethod
    def where(a: object, b: object, c: object) -> str:
        # Inductor emits C++ ternary (cond ? a : b) which isn't valid Python.
        # Use Python ternary instead; the walker converts it to C++ ternary.
        return f"({b} if {a} else {c})"

    def _special_unary(self, a: object, name: str) -> str:
        # Skip V.kernel.headers.add() — Helion includes c10/metal headers
        # unconditionally in the MSL preamble.
        return f"c10::metal::{name}({a})"

    def _special_binary(self, a: object, b: object, name: str) -> str:
        return f"c10::metal::{name}({a}, {b})"
