# NOTE: deliberately NO ``from __future__ import annotations`` -- these helpers
# are traced inline by the CuTe DSL from the generated kernel module, which
# imports them as real functions (see ``_flash_runtime.py`` for the same
# convention).
"""Runtime helpers of the flex-envelope score_mod / mask_mod hook.

The renderer (``flash_score_mod.py``) emits calls to these helpers through the
backend's library-import table (``_cute_flex_tanh``, ``_cute_flex_load_vec``,
``_cute_flex_load_window``,
``_cute_flex_fmod``, ``_cute_flex_remainder``); everything else in a rendered
hook is plain CuTe DSL scalar code.
"""

from typing import Any
from typing import cast

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32

from ._mlir_compat import ir

# ``tanh(y) = 1 - 2 / (exp2(2 * log2(e) * y) + 1)``: the identity the flash
# softcap uses (``_flash_runtime.softcap_t2r``).  The exp2 argument is clamped
# to 126 so an overflowing ``exp2`` cannot turn the Newton step NaN; tanh is
# exactly 1.0 in fp32 well before that point.
_TWO_LOG2E = 2.8853900817779268
_EXP2_CLAMP = 126.0


def tanh_f32(y: Float32, approx: bool = False) -> Float32:
    """fp32 ``tanh`` for a generic score_mod.

    ``approx=False`` (the default, IEEE kernels) evaluates the exp2 identity
    with the hardware ``ex2.approx`` the softmax already relies on and a
    Newton-refined ``rcp.approx`` (bit-identical to ``rcp.rn`` on the
    ``[1, 2**126)`` operand range), the same recipe and error class as the
    flash softcap; ``cute.math.tanh`` would expand to the ~17-instruction
    libdevice form.  ``approx=True`` (the ``fast_math`` setting) is the single
    ``tanh.approx.f32`` MUFU op.
    """
    if approx:
        return cute.math.tanh(y, approx=True)
    one = cutlass.Float32(1.0)
    t = cute.arch.fmin(
        y * cutlass.Float32(_TWO_LOG2E), cutlass.Float32(_EXP2_CLAMP), nan=True
    )
    d = cute.arch.exp2(t) + one
    # Negated seed ``nr = -1 / d``: ``v = 1 - d * r`` is then one FMA with
    # positive operands and ``nr' = nr + nr * v``.
    nr = cute.arch.rcp_approx(-d)
    v = cute.math.fma(d, nr, one)
    nr = cute.math.fma(nr, v, nr)
    # ``1 - 2 / d = 1 + 2 * nr``
    return cute.math.fma(cutlass.Float32(2.0), nr, one)


def load_vec(ptr: object, width: int, dtype: type) -> list:
    """One 16-byte-class vector load of ``width`` consecutive ``dtype`` values.

    Returns the lanes as typed DSL scalars.  ``ptr`` must be aligned to the
    vector's byte size; the renderer proves that statically from the aux
    tensor's shape and strides and the chunk base column.
    """
    numeric = cast("Any", dtype)
    vec = cute.arch.load(ptr, ir.VectorType.get([width], numeric.mlir_type))
    return [numeric(cast("Any", vec)[j]) for j in range(width)]


def load_window(
    ptr: object, last_ptr: object, count: int, width: int, dtype: type
) -> list:
    """``count`` consecutive ``width``-lane vectors from ``ptr`` as a flat list.

    The last vector is read at ``last_ptr`` (the renderer clamps it to the
    table end: its lanes are only selected when the thread's window is
    misaligned, in which case it lies inside the table).  ``ptr`` must be
    aligned to the vector's byte size.
    """
    values: list = []
    base = cast("Any", ptr)
    for k in range(count - 1):
        values.extend(load_vec(base + width * k, width, dtype))
    values.extend(load_vec(last_ptr, width, dtype))
    return values


def fmod(a: object, b: object) -> object:
    """C ``fmod`` / ``torch.fmod``: the remainder takes the dividend's sign.

    CuTe's ``%`` is exactly that (``arith.remf`` / ``arith.remsi``); the helper
    exists so the rendered text names the semantics (and so the CPU emulation
    can model it, Python's ``%`` being floor-mod).
    """
    return cast("Any", a) % b


def remainder_f32(a: Float32, b: Float32) -> Float32:
    """``torch.remainder`` on fp32: the remainder takes the divisor's sign.

    ``fmod`` then one sign correction, the same recipe as the integer
    ``python_mod`` helper; ``a - floor(a / b) * b`` would lose up to a few
    ulps through the rounded quotient.
    """
    r = cutlass.Float32(a) % cutlass.Float32(b)
    zero = cutlass.Float32(0.0)
    correct = (r != zero) & ((r < zero) != (cutlass.Float32(b) < zero))
    return cutlass.Float32(cutlass.select_(correct, r + cutlass.Float32(b), r))
