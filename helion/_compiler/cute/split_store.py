"""Proved two-axis split/permuted stores for the MMA-DAG warp owner.

Four tensor dimensions must be a permutation of the quotient/remainder
pairs of two logical coordinates.  Dense positive strides make the four
mixed-radix digits a bijection onto the tensor's storage span.  A tile is
locally affine only when its dimensions divide the two split periods;
these restrictions are checked again by the config validator, not inferred
from the shape of one sampled tile.  No gathers, masks, offsets in the index
DAG, overlapping strides or partial tiles are admitted.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
import itertools
import math
from typing import TYPE_CHECKING

import sympy

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class SplitStoreMap:
    """An immutable proof of a rank-four store's mixed-radix bijection.

    ``order`` names the physical dimensions of (outer row, inner row,
    outer column, inner column).  The pointer passed by the caller already
    includes its storage offset; every offset here is relative to it.
    """

    extents: tuple[int, int]
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    order: tuple[int, ...]

    @property
    def periods(self) -> tuple[int, int]:
        return self.shape[self.order[1]], self.shape[self.order[3]]

    @property
    def inner_strides(self) -> tuple[int, int]:
        return self.strides[self.order[1]], self.strides[self.order[3]]

    def offset(self, row: int, col: int) -> int:
        """Element offset at a logical coordinate (the proof's CPU model)."""
        row_period, col_period = self.periods
        digits = (
            row // row_period,
            row % row_period,
            col // col_period,
            col % col_period,
        )
        return sum(
            digit * self.strides[dim]
            for digit, dim in zip(digits, self.order, strict=True)
        )

    def tile_reason(self, rows: int, cols: int) -> str | None:
        """Why a uniform grid tile could cross a split boundary."""
        for tile, period in zip((rows, cols), self.periods, strict=True):
            if tile <= 0 or period % tile:
                return f"mapped store tile {tile} must divide split period {period}"
        return None

    def base_terms(self, row: str, col: str, elem_bytes: int) -> tuple[str, ...]:
        """Warp-emitter byte terms for a tile's first logical coordinate."""
        terms: list[str] = []
        for coord, outer, inner in (
            (row, self.order[0], self.order[1]),
            (col, self.order[2], self.order[3]),
        ):
            period = self.shape[inner]
            terms.extend(
                (
                    f"cutlass.Int64({coord} // {period}) * {self.strides[outer] * elem_bytes}",
                    f"cutlass.Int64({coord} % {period}) * {self.strides[inner] * elem_bytes}",
                )
            )
        return tuple(terms)


def _rational_bounds(
    expr: sympy.Basic, bounds: Mapping[sympy.Symbol, tuple[int, int]]
) -> tuple[Fraction, Fraction] | None:
    """Exact finite bounds for the proof's rational arithmetic vocabulary.

    Rational coefficients occur inside floor division.  Keep their bounds
    as exact Python fractions, not symbolic values that might be non-finite
    or undecidable.  Unknown symbols, products and operations fail closed.
    """
    if isinstance(expr, sympy.Symbol) and expr in bounds:
        lo, hi = bounds[expr]
        return Fraction(lo), Fraction(hi)
    if isinstance(expr, sympy.Rational):
        value = Fraction(int(expr.p), int(expr.q))
        return value, value
    if isinstance(expr, sympy.Add):
        lo = hi = Fraction(0)
        for arg in expr.args:
            interval = _rational_bounds(arg, bounds)
            if interval is None:
                return None
            lo += interval[0]
            hi += interval[1]
        return lo, hi
    if isinstance(expr, sympy.Mul):
        coefficient, value = expr.as_coeff_Mul()
        if not isinstance(coefficient, sympy.Rational) or value is expr:
            return None
        interval = _rational_bounds(value, bounds)
        if interval is None:
            return None
        scale = Fraction(int(coefficient.p), int(coefficient.q))
        lo, hi = scale * interval[0], scale * interval[1]
        return (lo, hi) if scale >= 0 else (hi, lo)
    if isinstance(expr, sympy.floor):
        interval = _rational_bounds(expr.args[0], bounds)
        if interval is not None:
            return Fraction(math.floor(interval[0])), Fraction(math.floor(interval[1]))
    if isinstance(expr, sympy.Mod):
        modulus = expr.args[1]
        if isinstance(modulus, sympy.Integer):
            radix = int(modulus)
            if radix <= 0:
                return None
            interval = _rational_bounds(expr.args[0], bounds)
            if interval is not None:
                lo, hi = interval
                if 0 <= lo <= hi < radix:
                    return lo, hi
                return Fraction(0), Fraction(radix - 1)
    return None


def integer_bounds(
    expr: sympy.Expr, bounds: Mapping[sympy.Symbol, tuple[int, int]]
) -> tuple[int, int] | None:
    """Finite, exact integer endpoints, or no proof (never truncate a bound)."""
    interval = _rational_bounds(expr, bounds)
    if interval is None:
        return None
    lo, hi = interval
    if lo.denominator != 1 or hi.denominator != 1:
        return None
    return lo.numerator, hi.numerator


def _normalize(
    expr: sympy.Basic, bounds: Mapping[sympy.Symbol, tuple[int, int]]
) -> sympy.Expr | None:
    if not isinstance(expr, sympy.Expr) or _rational_bounds(expr, bounds) is None:
        return None
    if not expr.args:
        return expr
    args: list[sympy.Expr] = []
    for arg in expr.args:
        normalized = _normalize(arg, bounds)
        if normalized is None:
            return None
        args.append(normalized)
    if isinstance(expr, sympy.Add):
        value = sympy.Add(*args)
    elif isinstance(expr, sympy.Mul):
        value = sympy.Mul(*args)
    elif isinstance(expr, sympy.floor):
        value = sympy.floor(args[0])
    elif isinstance(expr, sympy.Mod):
        value = sympy.Mod(*args)
    else:
        return None
    if not isinstance(value, sympy.Expr):
        return None
    if isinstance(value, sympy.Mod):
        dividend, modulus = value.args
        interval = _rational_bounds(dividend, bounds)
        if interval is None or not isinstance(modulus, sympy.Integer):
            return None
        lo, hi = interval
        if 0 <= lo <= hi < int(modulus) and isinstance(dividend, sympy.Expr):
            return dividend
    return value


def prove_split_store_map(
    indices: Sequence[sympy.Expr],
    symbols: tuple[sympy.Symbol, sympy.Symbol],
    extents: tuple[int, int],
    shape: Sequence[int],
    strides: Sequence[int],
) -> SplitStoreMap | None:
    """Prove a permutation of two complete quotient/remainder digit pairs.

    Periods come only from the tensor dimensions, and both digit pairs must
    cover their logical axis exactly.  The bounded simplification removes
    redundant modular wraps of a flattened coordinate; equality with the
    complete digit vocabulary, rather than sampled addresses, proves the
    map.  Degenerate dimensions and non-dense layouts stay unclaimed.
    """
    if len(indices) != 4 or len(shape) != 4 or len(strides) != 4:
        return None
    if symbols[0] == symbols[1] or any(
        sympy.ask(sympy.Q.integer(symbol)) is not True for symbol in symbols
    ):
        return None
    if any(type(size) is not int or size <= 1 for size in (*extents, *shape)):
        return None
    if any(type(stride) is not int or stride <= 0 for stride in strides):
        return None
    if math.prod(shape) != math.prod(extents):
        return None
    span = 1
    for dim in sorted(range(4), key=lambda dim: strides[dim]):
        if strides[dim] != span:
            return None
        span *= shape[dim]
    bounds = {
        symbol: (0, extent - 1) for symbol, extent in zip(symbols, extents, strict=True)
    }
    normalized = tuple(_normalize(expr, bounds) for expr in indices)
    if any(expr is None for expr in normalized):
        return None
    for order in itertools.permutations(range(4)):
        ro, ri, co, ci = order
        if shape[ro] * shape[ri] != extents[0] or shape[co] * shape[ci] != extents[1]:
            continue
        row, col = symbols
        digits = (
            sympy.floor(sympy.Mul(row, sympy.Rational(1, shape[ri]))),
            sympy.Mod(row, shape[ri]),
            sympy.floor(sympy.Mul(col, sympy.Rational(1, shape[ci]))),
            sympy.Mod(col, shape[ci]),
        )
        expected: list[sympy.Expr | None] = [sympy.Integer(0)] * 4
        for dim, digit in zip(order, digits, strict=True):
            if not isinstance(digit, sympy.Expr):
                return None
            expected[dim] = _normalize(digit, bounds)
        if normalized == tuple(expected):
            return SplitStoreMap(extents, tuple(shape), tuple(strides), order)
    return None
