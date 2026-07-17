"""Pallas-backend sympy expression printer.

``HelionPallasPrinter`` extends :class:`~helion._compiler.triton.printer.HelionTritonPrinter`
to emit plain Python operators instead of Triton runtime helpers.  Moved out of
``device_function.py`` so each backend owns its own printer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..triton.printer import HelionTritonPrinter

if TYPE_CHECKING:
    import sympy


class HelionPallasPrinter(HelionTritonPrinter):
    """Pallas printer that emits plain Python operators instead of Triton runtime helpers."""

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        # pyrefly: ignore [missing-attribute]
        return f"(({self._print(lhs)}) // ({self._print(rhs)}))"

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        # Paren both operands to override Python precedence — ``%`` binds
        # tighter than ``+``/``-`` so ``(a + b) % c`` would otherwise
        # print as ``a + b % c``.
        # pyrefly: ignore [missing-attribute]
        return f"(({self._print(lhs)}) % ({self._print(rhs)}))"


def pallas_texpr(expr: sympy.Expr) -> str:
    return HelionPallasPrinter().doprint(expr)
