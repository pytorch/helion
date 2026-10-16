"""Math expressions shared by CuTe scalar and tensor epilogue lowerings."""

from __future__ import annotations

# Sigmoid uses the default pointwise lowering's EX2.APPROX + RCP.APPROX
# contract, independently of fast_math. Its FP32 argument rounding already
# dominates the reciprocal error. NaNs, infinities, and saturation retain
# their behavior; denormal outputs flush to zero, as in the pointwise path.
# Callers compute in FP32 and retain each original result/store dtype cast.
# This does not change the lowering of an explicit division or other math.
# Fold the negation into log2(e) so packed FP32 epilogues avoid a separate negate.
SIGMOID_TEMPLATE = (
    "cute.math.rcp(1.0 + cute.math.exp2(({x}) * "
    "-1.4426950408889634, fastmath=True), approx=True, ftz=True)"
)


def nan_extremum_expr(reduction_type: str, a: str, b: str, *, float32: bool) -> str:
    """``max``/``min`` of ``a`` and ``b`` that propagates NaN, as torch.amax does.

    For fp32 ``cute.arch.fmax``/``fmin`` with ``nan=True`` is one FMNMX.NaN,
    the cost of the NaN-quiet form; ``cute.math.max(propagate_nan=True)``
    lowers to two compares and two selects there, so it only serves the other
    dtypes (integers have no NaN).
    """
    assert reduction_type in ("max", "min")
    if float32:
        return f"cute.arch.f{reduction_type}({a}, {b}, nan=True)"
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
