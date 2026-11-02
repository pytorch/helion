"""Device helpers for the exact loop-invariant fp32 division hoist.

The generic CuTe lowering renders an fp32 quotient as ``a / b``, which NVVM
emits as ``div.rn.f32``.  ptxas expands every ``div.rn.f32`` into its own
``MUFU.RCP`` + two reciprocal-refinement ``FFMA`` + ``FCHK`` + three
``FFMA`` + a slow-path call skeleton, PER ELEMENT, even when the divisor
is the same for a whole row (a softmax row sum, a kernel-wide
temperature).  The hoist in ``hoist_loop_invariant_recip`` moves the
divisor-only work out of the element loop and keeps three FMA-pipe
instructions plus two FMNMX per element, while the result stays the
IEEE-754 round-to-nearest-even quotient bit for bit.

Per divisor ``b`` (hoisted, once per loop-invariant divisor): ``y = 1.0 / b``,
the correctly rounded reciprocal ``RN(1/b)`` (an IEEE division), and
``lo, hi = exact_div_bounds_f32(b)``, the admitted ``|a|`` window.

Per element ``a`` of the speculated loop::

    q  = RN(a * y)                        # FMUL
    r  = RN(a - b * q)                    # FFMA; exact when admitted
    q' = RN(q + r * y)                    # FFMA
    min_abs = fmin(min_abs, |a|)          # FMNMX (NaN-propagating)
    max_abs = fmax(max_abs, |a|)          # FMNMX

and after the loop, ``if not (lo <= min_abs and max_abs < hi)`` -- some
dividend was a hard case -- the loop-carried values are restored and the
loop runs again as written, ``a / b`` (``div.rn.f32``) per element.

This is the Markstein quotient sequence (P. Markstein, "Computation of
elementary functions on the IBM RISC System/6000 processor", IBM J. Res.
Dev. 34(1), 1990: a correctly rounded reciprocal ``y``, a first quotient
``q`` and one exact-residual correction give the correctly rounded
quotient; see also Cornea, Harrison and Tang, "Scientific Computing on
Itanium-based Systems", Intel Press 2002, ch. 8, and Muller et al.,
"Handbook of Floating-Point Arithmetic", 2nd ed., Birkhauser 2018, sec.
4.7 on Markstein's division iterations).  It is also, instruction for
instruction, the fast path ptxas emits for ``div.rn.f32`` on this GPU
(``MUFU.RCP; FFMA -b,r0,1; FFMA r0,e,r0`` is ``RN(1/b)`` -- the same
sequence ``1.0 / b`` expands to -- followed by ``FFMA a,y,0; FFMA -b,q,a;
FFMA y,r,q``), guarded there by ``FCHK``.

Bit-identity argument (the lowering is enabled by default only because
every piece below is checked by ``test/test_cute_exact_division.py``):

1. *Exhaustive core.*  Let ``p = 24``.  For every pair of significands
   ``(sa, sb)`` in ``[1, 2)`` -- all ``2**23 * 2**23`` pairs, with the
   exponents of both operands fixed to zero -- the sequence above (with
   the admitting predicate true, so the fast path is taken) equals
   ``div.rn.f32``, the IEEE-754 round-to-nearest quotient (PTX ISA,
   "div.rn.f32 ... IEEE 754 compliant").  Verified exhaustively on a B200
   with this module's ``exact_div_fast_f32`` (artifacts of the
   ``exact-division-hoist`` workstream, ``prof/exhaustive.txt``).  The
   sequence is odd in ``a`` and odd in ``b`` (every rounding is
   sign-symmetric and the FMAs negate exactly), so the four sign
   combinations follow from the positive one.

2. *Scaling lemma.*  Write ``a = sa * 2**Ea``, ``b = sb * 2**Eb`` with
   normal ``a`` and ``b``.  Rounding to nearest commutes with scaling by a
   power of two as long as the exact value and its scaled image are both
   in the normal range (the rounding grid is uniform under the scaling
   there), so each step of the real computation is the matching step of
   the exponent-zero computation scaled by a power of two provided every
   intermediate is normal or exactly zero:

   * ``y = RN(1/b)``: ``|1/b|`` lies in ``(2**(-Eb-1), 2**-Eb]``, normal
     when ``Eb <= 125``, i.e. a biased exponent of ``b`` in ``[1, 252]``
     (also exactly the range ptxas gives the ``1.0 / b`` fast path).
   * ``q = RN(a*y)``, ``q' = RN(q + r*y)`` and the reference ``RN(a/b)``:
     all lie in ``(2**(Ea-Eb-1), 2**(Ea-Eb+1))``, normal when
     ``-125 <= Ea - Eb <= 126``; we require ``-124 <= Ea - Eb <= 126``.
   * ``r = RN(a - b*q)``: the exact residual is a multiple of
     ``ulp(b) * ulp(q) >= 2**(Ea-47)`` and at most about ``2**(Ea-21)`` in
     magnitude, so whenever it is nonzero it is normal when
     ``Ea - 47 >= -126``, i.e. ``Ea >= -79`` (biased ``>= 48``).  When it
     is zero in the model it is zero in the real computation (``b*q``
     equals ``a`` exactly in both).

   With biased exponents ``EA`` of ``a`` and ``EB`` of ``b`` the admitted
   set is therefore ``EB in [1, 252]`` and ``EA in [max(48, EB - 124),
   min(254, EB + 126)]``; it excludes zero, subnormal, infinite and NaN
   operands and every quotient that could overflow or leave the normal
   range.  ``exact_div_bounds_f32`` turns the window into the two powers
   of two ``lo = 2**(EAmin-127)`` and ``hi = 2**(EAmax-126)`` (``hi`` is
   ``+inf`` when ``EAmax = 254``), and an empty window (``lo = +inf``,
   ``hi = 0``) when ``b`` itself is not admitted, so the per-element test
   ``lo <= |a| < hi`` is two compares; NaN fails both.

3. *Fallback.*  A loop with some pair outside the window runs again as
   written, ``a / b`` -- the same ``div.rn.f32`` the plain lowering emits
   -- so hard cases (signed zeros, subnormals, infinities, NaNs,
   overflowing or underflowing quotients, ``FCHK``'s own slow-path
   operands) are bit identical by construction.  The redo form
   (``hoist_loop_invariant_recip``): the loop runs the unguarded
   ``exact_div_fast_f32`` while ``cute.arch.fmin``/``fmax`` (``nan=True``)
   track the extrema of ``|a|`` per divisor; afterwards
   ``exact_div_redo_f32`` decides from the extrema whether every dividend
   was admitted.  If not, the loop-carried values are restored and the
   original loop -- plain ``a / b`` -- runs again, overwriting every
   store.  The loop body keeps the plain lowering's statements (two FMNMX
   added per element) and so the shape ptxas compiles.  A loop the redo
   cannot take (it reads memory it writes, contains a barrier, shuffle,
   reduction or branch, ...) keeps the plain ``a / b``.

   A per-element guard (``if not admitted: a / b``, once the ``guard``
   form of ``cute_exact_div_form``) is deliberately NOT offered: the
   per-element branch changed the compiled shape of the loop (no
   unrolling, spills, a BSSY/BRA/BSYNC trio per element) so much that
   ptxas contracted the plain lowering's unmarked fp32 ``mul``/``add``/
   ``sub`` chains around the quotient differently, and the reductions
   over the quotients differed from the plain render by 1-2 ulp on ragged
   shapes -- the quotients themselves were exact.  A recorded config that
   still names it is repaired to ``plain``.

Everything here is exact in IEEE-754 arithmetic; nothing depends on
``fast_math`` (whose fp32 division is the approximate ``cute.math.div``
and never reaches this module).  ``Settings.cute_exact_division`` turns
the lowering off; the config knob ``cute_exact_div_form`` picks the form
(``redo`` / ``plain``) per kernel so the autotuner can keep the plain
``div.rn`` where the redo copy would push a 1024-thread kernel to the
register cap.
"""

from __future__ import annotations

from cutlass import Boolean
from cutlass import Float32
from cutlass import Int32
import cutlass.cute as cute

from .inline_asm_helpers import inline_asm_elementwise

# Biased-exponent limits of the admitted window; see the module docstring.
_B_EXP_MIN = 1  # b normal ...
_B_EXP_MAX = 252  # ... with a normal reciprocal (|b| < 2**126)
_A_EXP_MIN = 48  # |a| >= 2**-79 keeps a nonzero residual normal
_A_EXP_MAX = 254  # a finite
_QUOT_EXP_LOW = -124  # Ea - Eb >= -124: quotient stays normal with margin
_QUOT_EXP_HIGH = 126  # Ea - Eb <= 126: quotient cannot overflow


@cute.jit
def _opaque_f32(x: Float32) -> Float32:
    """``x``, through an empty side-effecting asm (``=f,0`` ties the output
    to the input register; no instruction is emitted).  Applied to the
    hoisted bounds, which ptxas otherwise rematerializes from the divisor
    (the ten-instruction exponent chain) at every read instead of keeping
    two registers live."""
    return inline_asm_elementwise(
        (x,), asm="", constraints="=f,0", dtype=Float32, is_pure=False
    )


@cute.jit
def exact_div_bounds_f32(b: Float32) -> tuple[Float32, Float32]:
    """``(lo, hi)``: the fast path is admitted for ``lo <= |a| < hi``.

    Both are powers of two built from the biased exponent of ``b`` (see
    the scaling lemma in the module docstring); ``hi`` is ``+inf`` when
    every finite ``|a|`` above ``lo`` is admitted.  A divisor outside its
    own window (zero, subnormal, ``|b| >= 2**126``, infinite, NaN) gets
    the empty window ``(+inf, 0)``: no ``|a|`` passes.
    """
    # pyrefly: ignore[unsupported-operation]
    eb = (b.bitcast(Int32) >> Int32(23)) & Int32(0xFF)
    lo_e = Int32(_A_EXP_MIN)
    if eb + _QUOT_EXP_LOW > lo_e:
        lo_e = eb + _QUOT_EXP_LOW
    hi_e = Int32(_A_EXP_MAX)
    if eb + _QUOT_EXP_HIGH < hi_e:
        hi_e = eb + _QUOT_EXP_HIGH
    # pyrefly: ignore[missing-attribute]
    lo = (lo_e << Int32(23)).bitcast(Float32)
    # pyrefly: ignore[missing-attribute]
    hi = ((hi_e + Int32(1)) << Int32(23)).bitcast(Float32)
    if (eb < _B_EXP_MIN) | (eb > _B_EXP_MAX):
        lo = Float32(float("inf"))
        hi = Float32(0.0)
    return _opaque_f32(lo), _opaque_f32(hi)  # pyrefly: ignore[bad-return]


@cute.jit
def exact_div_fast_f32(a: Float32, b: Float32, y: Float32) -> Float32:
    """The fast path: ``RN(q + r*y)`` with ``q = RN(a*y)`` and
    ``r = RN(a - b*q)`` for ``y = RN(1/b)``.  Equal to ``RN(a/b)`` on the
    admitted window (module docstring), which the redo test checks for
    the whole loop afterwards."""
    q = a * y
    r = cute.math.fma(-q, b, a)
    return cute.math.fma(r, y, q)


@cute.jit
def exact_div_redo_f32(
    min_abs: Float32, max_abs: Float32, lo: Float32, hi: Float32
) -> Boolean:
    """Whether some dividend of a speculated loop left the admitted window.

    ``min_abs`` / ``max_abs`` are the running extrema of ``|a|`` over the
    loop (``cute.arch.fmin`` / ``fmax`` with ``nan=True``, so a NaN
    dividend makes them NaN and fails both compares); ``lo`` / ``hi`` are
    the divisor's ``exact_div_bounds_f32``.  True means the original loop
    must run again with ``div.rn.f32``.
    """
    admitted: Boolean = (min_abs >= lo) & (max_abs < hi)
    return not admitted  # pyrefly: ignore[bad-return]
