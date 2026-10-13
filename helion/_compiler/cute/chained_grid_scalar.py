"""Captured scalar grid expressions, distinct from vector tile extents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx import Node
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import PythonMod

from ...language import _tracing_ops
from ..host_function import HostFunction
from ..variable_origin import GridOrigin
from ..variable_origin import TileBeginOrigin
from ..variable_origin import TileIdOrigin

if TYPE_CHECKING:
    from collections.abc import Mapping


def grid_scalar_expression(node: object) -> sympy.Expr | None:
    """Prove a composite uses only known scalar grid origins and integer math.

    A block-size symbol is deliberately excluded: it can represent an entire
    vector selector. Unknown symbols, tile ends/counts, and arbitrary SymPy
    operations retain their previous unsupported behavior.
    """
    if not isinstance(node, Node) or node.target not in (
        _tracing_ops._get_symnode,
        torch.ops.aten.sym_size.int,
    ):
        return None
    value = node.meta.get("val")
    if not isinstance(value, torch.SymInt):
        return None
    expr = value._sympy_()
    if (
        not isinstance(expr, sympy.Expr)
        or not expr.free_symbols
        or isinstance(expr, sympy.Symbol)
    ):
        return None
    origins = HostFunction.current().expr_to_origin
    for symbol in expr.free_symbols:
        info = origins.get(symbol)
        if info is None or type(info.origin) not in (
            GridOrigin,
            TileBeginOrigin,
            TileIdOrigin,
        ):
            return None

    def supported(part: sympy.Basic) -> bool:
        if isinstance(part, (sympy.Integer, sympy.Symbol)):
            return True
        if isinstance(part, (sympy.Add, sympy.Mul, sympy.Min, sympy.Max)):
            return all(supported(arg) for arg in part.args)
        if isinstance(part, (FloorDiv, PythonMod, sympy.Mod)):
            return (
                len(part.args) == 2
                and isinstance(part.args[1], sympy.Integer)
                and part.args[1] != 0
                and supported(part.args[0])
            )
        return False

    return expr if supported(expr) else None


def grid_scalar_bounds(
    expr: sympy.Basic,
    symbols: Mapping[sympy.Symbol, tuple[int, int]],
    limits: tuple[int, int],
) -> tuple[int, int] | None:
    """Bound every emitted integer intermediate, including n-ary partial sums."""

    def checked(bounds: tuple[int, int]) -> tuple[int, int] | None:
        return bounds if limits[0] <= bounds[0] <= bounds[1] <= limits[1] else None

    if isinstance(expr, sympy.Integer):
        return checked((int(expr), int(expr)))
    if isinstance(expr, sympy.Symbol):
        return checked(symbols[expr]) if expr in symbols else None
    arguments = [grid_scalar_bounds(arg, symbols, limits) for arg in expr.args]
    if not arguments or any(arg is None for arg in arguments):
        return None
    bounds = [arg for arg in arguments if arg is not None]
    current = bounds[0]
    if isinstance(expr, (sympy.Min, sympy.Max)):
        select = min if isinstance(expr, sympy.Min) else max
        return checked(
            (select(arg[0] for arg in bounds), select(arg[1] for arg in bounds))
        )
    if isinstance(expr, (sympy.Add, sympy.Mul)):
        for other in bounds[1:]:
            if isinstance(expr, sympy.Add):
                updated = (current[0] + other[0], current[1] + other[1])
            else:
                products = [first * second for first in current for second in other]
                updated = (min(products), max(products))
            if checked(updated) is None:
                return None
            current = updated
        return current
    if len(bounds) != 2 or bounds[1][0] != bounds[1][1] or bounds[1][0] == 0:
        return None
    divisor = bounds[1][0]
    if isinstance(expr, FloorDiv):
        values = [value // divisor for value in current]
        return checked((min(values), max(values)))
    if isinstance(expr, (PythonMod, sympy.Mod)):
        # The C remainder used by the signed correction also divides. Avoid
        # signed minimum / -1 even though its mathematical remainder is zero.
        if divisor == -1 and current[0] == limits[0]:
            return None
        return checked((0, divisor - 1) if divisor > 0 else (divisor + 1, 0))
    return None
