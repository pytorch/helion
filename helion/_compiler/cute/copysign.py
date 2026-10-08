"""Bitwise copysign for CuTe codegen.

Called during ``@cute.kernel`` tracing (plain Python), so no ``@cute.jit``
wrapper is needed.
"""

from __future__ import annotations

import cutlass

# The unsigned type of each float type's bits, and its sign bit.
_BITS = {
    cutlass.Float32: (cutlass.Uint32, 1 << 31),
    cutlass.Float64: (cutlass.Uint64, 1 << 63),
}


def copysign(magnitude: cutlass.Numeric, sign: cutlass.Numeric) -> cutlass.Numeric:
    """``magnitude`` with the sign bit of ``sign``, both Float32 or Float64.

    ``cute.math.copysign`` gives a NaN magnitude the canonical positive NaN;
    eager moves the sign bit onto it like onto any other value.
    """
    float_type = type(magnitude)
    bits, sign_bit = _BITS[float_type]  # pyrefly: ignore[bad-index]
    magnitude_bits = magnitude.bitcast(bits) & bits(sign_bit - 1)
    sign_bits = sign.bitcast(bits) & bits(sign_bit)
    # pyrefly: ignore[unsupported-operation]
    return (magnitude_bits | sign_bits).bitcast(float_type)
