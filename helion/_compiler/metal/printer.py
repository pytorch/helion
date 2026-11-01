"""Metal-backend sympy expression printer.

Inductor's ``TritonPrinter`` prints a ``FloorDiv`` or ``PythonMod`` whose
operand signs sympy can't prove as a ``triton_helpers`` call, which MSL
doesn't have.  Metal prints such an expression with Python semantics instead,
using PyTorch's ``c10::metal`` helpers, and leaves every other expression as
``HelionTritonPrinter`` prints it.
"""

from __future__ import annotations

from typing import cast

import sympy
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import PythonMod

from ..device_function import DeviceFunction
from ..triton.printer import HelionTritonPrinter
from ..triton.printer import texpr


def _operands(node: sympy.Basic) -> tuple[sympy.Expr, sympy.Expr]:
    return cast("tuple[sympy.Expr, sympy.Expr]", node.args)


def _needs_helper(node: sympy.Basic) -> bool:
    """Whether ``texpr`` prints *node*, a ``FloorDiv`` or ``PythonMod``, as a
    Triton helper call.

    Mirrors the plain-operator cases of ``TritonPrinter._print_FloorDiv`` /
    ``_print_PythonMod`` (both operands nonnegative) and of
    ``HelionTritonPrinter._print_FloorDiv`` (a constexpr divided by an
    integer).
    """
    lhs, rhs = _operands(node)
    if (
        lhs.is_nonnegative  # pyrefly: ignore[missing-attribute]
        and rhs.is_nonnegative  # pyrefly: ignore[missing-attribute]
    ):
        return False
    return not (
        isinstance(node, FloorDiv)
        and isinstance(rhs, sympy.Integer)
        and isinstance(lhs, sympy.Symbol)
        and lhs.name in DeviceFunction.current()._constexpr_args
    )


class _SignedIndexPrinter(HelionTritonPrinter):
    """Prints an expression in 64-bit signed arithmetic.

    Metal's index values (offsets, ``tid``, ``tgid``) are ``uint``, so a
    subtraction can wrap before a division sees it.  Casting every leaf to
    ``long`` makes each intermediate the exact Python integer; a ``uint`` leaf
    up to 2**32 - 1 converts exactly.  The walker lowers ``//`` to
    ``c10::metal::floor_divide``; it and ``c10::metal::remainder`` follow
    Python's floor semantics for any operand signs.
    """

    def _print_Symbol(self, expr: sympy.Symbol) -> str:
        return f"(static_cast<long>({expr.name}))"

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = _operands(expr)
        return f"(({self.doprint(lhs)}) // ({self.doprint(rhs)}))"

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        lhs, rhs = _operands(expr)
        return f"c10.metal.remainder({self.doprint(lhs)}, {self.doprint(rhs)})"


def metal_texpr(expr: sympy.Expr) -> str:
    # The whole expression is signed, not just the division: its other leaves
    # are uint too, and a signed quotient added to a wrapped uint term stays
    # wrong.
    if any(_needs_helper(node) for node in expr.atoms(FloorDiv, PythonMod)):
        return _SignedIndexPrinter().doprint(expr)
    return texpr(expr)
