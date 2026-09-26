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
