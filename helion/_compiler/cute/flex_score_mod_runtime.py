# NOTE: deliberately NO ``from __future__ import annotations`` -- these helpers
# are traced inline by the CuTe DSL from the generated kernel module, which
# imports them as real functions (see ``_flash_runtime.py`` for the same
# convention).
"""Runtime helpers of the flex-envelope score_mod / mask_mod hook.

The renderer (``flash_score_mod.py``) emits calls to these helpers through the
backend's library-import table (``_cute_flex_tanh``, ``_cute_flex_load_vec``,
``_cute_flex_load_run``, ``_cute_flex_load_window``, ``_cute_flex_coop_lane``,
``_cute_flex_coop_issue``, ``_cute_flex_lane_below``,
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


def load_run(ptr: object, count: int, width: int, dtype: type) -> list:
    """``count`` consecutive ``width``-lane vectors from ``ptr`` as a flat list:
    the contiguous run one group of lanes reads in the strided-run form.
    ``ptr`` must be aligned to the vector's byte size."""
    values: list = []
    base = cast("Any", ptr)
    for k in range(count):
        values.extend(load_vec(base + width * k, width, dtype))
    return values


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


def coop_lane_map(lane: object, width: int) -> tuple[object, object]:
    """``(row, slot)`` of a lane within one cooperative request of ``width``
    bytes per row: the row (of the ``512 // width`` rows the request covers)
    and the 16-byte slot (of ``width // 16``) the lane copies.  Works on
    Python ints and DSL ``Int32`` alike (shifts, masks, adds).

    Row-major (``lane // L``, ``lane % L``) would put, at the padded pitch,
    two rows a multiple of 16 words apart into one 8-lane store phase for
    the 64- and 32-byte widths; the phases here pair rows 4 (64) / 2 (32)
    apart instead, whose pitch offsets land on disjoint bank groups.
    """
    index = cast("Any", lane)
    if width == 128:
        return index >> 3, index & 7
    if width == 64:
        return (index >> 3) + (index & 4), index & 3
    if width == 32:
        return ((index >> 4) << 3) + ((index >> 3) & 1) + (index & 6), index & 1
    raise ValueError(f"unsupported cooperative width {width}")


def coop_issue(
    row_ptr: object,
    col: object,
    slot: object,
    row_step: int,
    width: int,
    units: int,
    per_vector: int,
    dtype: type,
    stage_ptr: object,
    lane: object,
    pitch: int,
    limit: int | None = None,
) -> None:
    """Issue this lane's share of the warp's block copy for one chunk.

    ``row_ptr`` is the table pointer at the first row this lane copies (the
    owner's row base shifted by ``coop_lane_map`` row minus lane rows: the
    renderer adds that), ``col`` the element offset of the chunk's first
    column along the lane dimension, ``slot`` the lane's 16-byte slot,
    ``row_step`` the element distance between consecutive rows, and
    ``stage_ptr`` the chunk's staging slot for this table (the table's
    dtype; ``pitch`` elements per staged row).  The block is ``units``
    pieces of ``width`` bytes per row; piece ``u``, request ``k`` copies
    row ``row + k * rows_per_request`` at bytes ``u * width + slot * 16`` of
    the row's chunk span, as one 16-byte ``cp.async.cg`` into the row's
    staged slot.  With ``limit`` (the lane dimension's last valid vector
    start, unproven tables) each piece's element offset is clamped into the
    table: the owner selects zero for out-of-range columns, so the clamped
    vector's values are never used.  The caller commits the group (one per
    chunk) and waits before the rows are read.
    """
    lanes_per_row = width // 16
    rows_per_request = 32 // lanes_per_row
    unit_elems = width * per_vector // 16
    row, _slot = cast("tuple[Any, Any]", coop_lane_map(lane, width))
    source = cast("Any", col) + cast("Any", slot) * per_vector
    target = cast("Any", stage_ptr) + (row * pitch + cast("Any", slot) * per_vector)
    for u in range(units):
        offset = source + u * unit_elems
        if limit is not None:
            offset = cutlass.min(
                cutlass.max(offset, cutlass.Int32(0)), cutlass.Int32(limit)
            )
        piece = cast("Any", row_ptr) + offset
        for k in range(lanes_per_row):
            cute.arch.cp_async_shared_global(
                target + (u * unit_elems + k * rows_per_request * pitch),
                piece + k * rows_per_request * row_step,
                16,
                "cg",
            )


def lane_below(limit: object, width: int) -> cutlass.Int32:
    """The ``width``-bit word whose bits ``j < limit`` are set (``limit``
    clamped to ``[0, width]``): one bound of a keep-word lane interval.

    ``1 << 32`` is poison in the DSL's ``Int32`` shift, so the full 32-bit
    word is selected rather than shifted into.
    """
    t = cutlass.max(cutlass.Int32(limit), cutlass.Int32(0))
    if width < 32:
        t = cutlass.min(t, cutlass.Int32(width))
        return (cutlass.Int32(1) << t) - cutlass.Int32(1)
    full = t >= cutlass.Int32(32)
    t = cutlass.min(t, cutlass.Int32(31))
    partial = (cutlass.Int32(1) << t) - cutlass.Int32(1)
    return cutlass.Int32(cutlass.select_(full, cutlass.Int32(-1), partial))


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
