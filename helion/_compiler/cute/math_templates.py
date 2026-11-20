"""Math expressions shared by CuTe scalar and tensor epilogue lowerings."""

from __future__ import annotations

import math

# ``torch.sigmoid`` in FP32: ``1 / (1 + ex2.approx(x * -log2(e)))``, with the
# bits of Triton's ``tl.sigmoid`` (``ex2.approx`` and ``div.full(1, d)``) for
# every FP32 input (compared over all 2**32 of them on an sm_100 device),
# independently of fast_math.  ``d = 1 + e`` is at least 1,
# so the result is at most 1 and underflows for ``d`` above ``2**126``: there
# ``div.full`` takes the reciprocal of ``d / 4`` and scales it back, and the
# bare ``rcp.approx.ftz(d)`` returns 0 for a subnormal result (arguments in
# about (-88.72, -87.34)).  This form scales every ``d``: ``d / 4`` is the
# exactly rounded ``e * 0.25 + 0.25`` (one FFMA), its reciprocal is always a
# normal number, so ``rcp.approx.ftz`` never flushes, and the final ``* 0.25``
# is exact but for the rounding of a subnormal result.  One multiply more
# than the bare reciprocal, no compare and no select.  ``ex2.approx.ftz``
# differs from ``ex2.approx`` only below ``2**-126``, where ``1 + e`` is 1.
# NaNs, infinities and saturation keep their behavior.
# Callers compute in FP32 and retain each original result/store dtype cast.
# This does not change the lowering of an explicit division or other math.
# Fold the negation into log2(e) so packed FP32 epilogues avoid a separate negate.
# The parentheses make the template one operand wherever a caller pastes it.
SIGMOID_TEMPLATE = (
    "(cute.math.rcp(cute.math.exp2(({x}) * -1.4426950408889634, "
    "fastmath=True) * 0.25 + 0.25, approx=True, ftz=True) * 0.25)"
)
# ``1 / (1 + ex2.approx.ftz(x * -log2(e)))`` with the bare ``rcp.approx.ftz``:
# not a sigmoid lowering (it returns 0 where the sigmoid is subnormal) but the
# reciprocal ``div.approx.ftz`` multiplies by, for the callers that mirror a
# ``fast_math`` quotient by ``1 + exp(-x)``.
FAST_MATH_LOGISTIC_RECIPROCAL_TEMPLATE = (
    "cute.math.rcp(1.0 + cute.math.exp2(({x}) * "
    "-1.4426950408889634, fastmath=True), approx=True, ftz=True)"
)


# The signed integer dtypes whose chains of maxima or minima ptxas compiles
# with one operand's negation lost (``VIMNMX3`` for Int32, its packed 16-bit
# form ``VIMNMX3.S16x2`` for Int16), each with the unsigned dtype of its
# width and its sign bit.
_SIGNED_CHAIN_KEYS = {
    "cutlass.Int16": ("cutlass.Uint16", 1 << 15),
    "cutlass.Int32": ("cutlass.Uint32", 1 << 31),
}


def nan_extremum_expr(
    reduction_type: str,
    a: str,
    b: str,
    *,
    float32: bool,
    chain_dtype: str | None = None,
) -> str:
    """``max``/``min`` of ``a`` and ``b`` that propagates NaN, as torch.amax does.

    For fp32 ``cute.arch.fmax``/``fmin`` with ``nan=True`` is one FMNMX.NaN,
    the cost of the NaN-quiet form; ``cute.math.max(propagate_nan=True)``
    lowers to two compares and two selects there, so it only serves the other
    dtypes (integers have no NaN).

    A reduction passes ``chain_dtype``, the CuTe dtype of both operands (the
    text of its constructor).  It accumulates one element or lane after
    another, so its extrema form a chain, ``max(max(max(a, b), c), d)``,
    which ptxas fuses into three-operand instructions (``VIMNMX3``).  In a
    chain of Int32 or Int16 extrema ptxas (CUDA 13.3) loses the negation of
    one operand: ``torch.amax(-x)`` took the maximum of ``-a``, ``b`` and
    ``-c``.  It keeps every negation in an unsigned chain, so an Int32 or
    Int16 extremum of a chain is taken over unsigned keys: flipping the sign
    bit maps the signed order onto the unsigned one exactly (``a < b`` iff
    ``a ^ SIGN < b ^ SIGN`` as unsigned values), and the flips between
    consecutive links cancel.  The elementwise lowering of ``torch.maximum``
    / ``torch.minimum`` and of the clamps passes ``chain_dtype`` for the
    links of a chain of such extrema, which one expression can form too
    (``handler_ops._in_extremum_chain``).  Int8 keeps the plain form: ptxas
    sign-extends each negation before the packed 16-bit instruction, its
    extrema of negated values are right at every layout measured, and the
    keys would slow it down.  Unsigned integers (their chains keep every
    negation), 64-bit integers (a compare and a select) and the floats keep
    the forms above too.
    """
    assert reduction_type in ("max", "min")
    if float32:
        return f"cute.arch.f{reduction_type}({a}, {b}, nan=True)"
    key = _SIGNED_CHAIN_KEYS.get(chain_dtype) if chain_dtype is not None else None
    if key is not None:
        unsigned, sign = key
        sign_bit = f"{unsigned}({sign})"
        return (
            f"{chain_dtype}(cute.math.{reduction_type}("
            f"({unsigned}({a}) ^ {sign_bit}), ({unsigned}({b}) ^ {sign_bit})"
            f") ^ {sign_bit})"
        )
    return f"cute.math.{reduction_type}({a}, {b}, propagate_nan=True)"


def argreduce_candidate_expr(
    index: str, value: str, reduced: str, max_index: str
) -> str:
    """``index`` where ``value`` holds the reduced extremum ``reduced``, else
    ``max_index``: min-reducing the candidates gives argmin/argmax.

    ``reduced`` is NaN exactly when the row holds one; then the NaNs are the
    candidates, as torch picks the first NaN.
    """
    return (
        f"({index}) if ((({value}) == ({reduced})) | "
        f"(({value}) != ({value}))) else ({max_index})"
    )


def float_literal(value: float) -> str:
    """Python source for a float constant: ``repr`` for a finite value;
    ``inf``, ``-inf`` and ``nan`` repr to bare names (undefined in the
    generated module), so they are spelled as ``float(...)`` calls."""
    if math.isinf(value):
        return "float('inf')" if value > 0 else "float('-inf')"
    if math.isnan(value):
        return "float('nan')"
    return repr(value)
