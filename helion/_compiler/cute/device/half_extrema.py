# pyrefly: ignore-errors
"""The extremum of a 16-bit packet's half-precision lanes, two lanes at a time.

Device helper (see the package doc) for ``half_extremum_folds``, which
replaces a V-loop's lane-by-lane fp32 fold of a bf16 / f16 packet with one
call here.
"""

from __future__ import annotations

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import vector
import cutlass.cute as cute

from .ptx import max_nan_bf16x2
from .ptx import max_nan_f16x2
from .ptx import min_nan_bf16x2
from .ptx import min_nan_f16x2
from .ptx import unpack_f16x2

_HALF2_EXTREMA = {
    (cutlass.BFloat16, "max"): max_nan_bf16x2,
    (cutlass.BFloat16, "min"): min_nan_bf16x2,
    (cutlass.Float16, "max"): max_nan_f16x2,
    (cutlass.Float16, "min"): min_nan_f16x2,
}


def half_packet_extremum_f32(
    packet: object, count: int, half: type, op: str
) -> cutlass.Float32:
    """``cute.arch.fmax`` (``op == "max"``) or ``fmin`` with ``nan=True`` of
    the first ``count`` lanes of a 16-bit ``packet``, each bitcast to
    ``half`` (``cutlass.BFloat16`` / ``cutlass.Float16``) and widened to fp32:
    the lanes compared two to a word with ``max.NaN.bf16x2`` and friends,
    then the last word's halves widened and compared in fp32.

    Bit for bit the lane-by-lane fp32 fold: widening is exact and monotonic,
    both precisions order -0.0 below +0.0 and give the canonical NaN for any
    NaN, and the closing fp32 op canonicalizes a NaN half.  A packet that is
    not a vector of an even number of 16-bit lanes (whole 32-bit words), and
    an odd ``count``, are folded lane by lane.
    """
    fold = cute.arch.fmax if op == "max" else cute.arch.fmin
    values = cutlass.extract_mlir_values(packet)
    vector_type = (
        ir.VectorType(values[0].type)
        if len(values) == 1 and ir.VectorType.isinstance(values[0].type)
        else None
    )
    if (
        vector_type is None
        or vector_type.element_type.width != 16
        or count % 2
        or vector_type.shape[0] % 2
        or count > vector_type.shape[0]
    ):
        result = None
        for lane in range(count):
            value = cutlass.Float32(cutlass.Uint16(packet[lane]).bitcast(half))
            result = value if result is None else fold(result, value, nan=True)
        return result
    words = vector.bitcast(
        ir.VectorType.get([vector_type.shape[0] // 2], cutlass.Uint32.mlir_type),
        values[0],
    )
    level = [
        cutlass.Uint32(vector.extract(words, [], [index]))
        for index in range(count // 2)
    ]
    combine = _HALF2_EXTREMA[half, op]
    while len(level) > 1:
        pairs = [combine(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        level = pairs + level[2 * len(pairs) :]
    (word,) = level
    if half is cutlass.BFloat16:
        low = cutlass.Uint32(word << cutlass.Uint32(16)).bitcast(cutlass.Float32)
        high = cutlass.Uint32(word & cutlass.Uint32(0xFFFF0000)).bitcast(
            cutlass.Float32
        )
    else:
        low, high = unpack_f16x2(word)
    return fold(low, high, nan=True)
