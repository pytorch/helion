"""Pallas-backend sympy expression printer.

``HelionPallasPrinter`` extends :class:`~helion._compiler.triton.printer.HelionTritonPrinter`
to emit plain Python operators instead of Triton runtime helpers.  Moved out of
``device_function.py`` so each backend owns its own printer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from torch.utils._sympy.printers import PRECEDENCE

from ..triton.printer import HelionTritonPrinter

if TYPE_CHECKING:
    import sympy


class HelionPallasPrinter(HelionTritonPrinter):
    """Pallas printer that emits plain Python operators instead of Triton runtime helpers."""

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        level = PRECEDENCE["Atom"] - 0.5
        lhs_expr = cast("sympy.Expr", lhs)
        rhs_expr = cast("sympy.Expr", rhs)
        return f"({self.parenthesize(lhs_expr, level)} // {self.parenthesize(rhs_expr, level)})"

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        level = PRECEDENCE["Atom"] - 0.5
        lhs_expr = cast("sympy.Expr", lhs)
        rhs_expr = cast("sympy.Expr", rhs)
        return f"({self.parenthesize(lhs_expr, level)} % {self.parenthesize(rhs_expr, level)})"


def pallas_texpr(expr: sympy.Expr) -> str:
    return HelionPallasPrinter().doprint(expr)
