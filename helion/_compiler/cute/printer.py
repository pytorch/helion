"""Cute-backend sympy expression printer.

``HelionCutePrinter`` extends :class:`~helion._compiler.triton.printer.HelionTritonPrinter`
to avoid Triton runtime helpers in device expressions.  Moved out of
``device_function.py`` so each backend owns its own printer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ..triton.printer import HelionTritonPrinter

if TYPE_CHECKING:
    import sympy


class HelionCutePrinter(HelionTritonPrinter):
    """CuTe printer that avoids Triton runtime helpers in device expressions."""

    def _print_basic_expr(self, expr: sympy.Basic) -> str:
        return self.doprint(cast("sympy.Expr", expr))

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        return f"({self._print_basic_expr(lhs)} // {self._print_basic_expr(rhs)})"

    def _print_CleanDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        return f"({self._print_basic_expr(lhs)} // {self._print_basic_expr(rhs)})"

    def _print_CeilDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        lhs_printed = self._print_basic_expr(lhs)
        rhs_printed = self._print_basic_expr(rhs)
        return f"(({lhs_printed} + {rhs_printed} - 1) // {rhs_printed})"

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        return f"({self._print_basic_expr(lhs)} % {self._print_basic_expr(rhs)})"


def cute_texpr(expr: sympy.Expr) -> str:
    return HelionCutePrinter().doprint(expr)
