"""FlashAttention-4-inspired approximate math utilities.

On NVIDIA Blackwell (B200) and newer GPUs, tensor core throughput has
scaled dramatically while Special Function Unit (SFU) throughput has
not.  For kernels where ``exp`` / ``softmax`` is on the critical path
(e.g., attention), the SFU becomes the bottleneck.

FlashAttention-4 demonstrated that a degree-3 polynomial approximation
of ``2^f`` can run on CUDA/FMA cores *in parallel* with SFU work,
achieving ~20% speedup over ``cuDNN`` attention on Blackwell.

This module provides:

* **Polynomial coefficients** for the FA4 ``exp2`` approximation,
  ready to embed in ``hl.inline_triton()`` blocks.
* **``exp2_poly_triton_src``** — a Triton code snippet implementing
  the approximation, suitable for ``hl.inline_triton()`` usage.
* **``exp_to_exp2_scale``** — the ``log2(e)`` constant for converting
  ``exp(x)`` to ``exp2(x * log2e)``, useful for folding into upstream
  multiplies (e.g., attention's ``sm_scale``).

Accuracy
--------
The degree-3 polynomial achieves:

* Max relative error (FP32): 8.77 × 10⁻⁵
* After BF16 rounding: matches ``MUFU.EX2`` to within 1 ULP on >99%
  of inputs in the range ``|f| < 1`` (fractional part after integer
  split).

When to use
-----------
* **Use when** the kernel is SFU-throughput-limited (profile with
  ``ncu`` and check ``sm__pipe_fma_cycles_active`` vs
  ``sm__pipe_su_cycles_active``).
* **Do not use** for general-purpose ``exp`` — Triton's ``tl.exp``
  already maps to ``MUFU.EX2`` and is faster for non-bottlenecked
  kernels.

References
----------
* FlashAttention-4: Algorithm and Kernel Pipelining Co-Design,
  `arXiv:2603.05451 <https://arxiv.org/abs/2603.05451>`_
* "We reverse-engineered Flash Attention 4", Modal Blog,
  https://modal.com/blog/reverse-engineer-flash-attention-4

.. note::

   This module is **experimental** and may change without notice.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Polynomial coefficients for 2^f ≈ p0 + f*(p1 + f*(p2 + f*p3))
# These are the Horner-form coefficients from FlashAttention-4.
# The approximation is evaluated on the *fractional* part f of the input
# after splitting x = n + f  (n integer, |f| < 1).
# ---------------------------------------------------------------------------

POLY_EXP2_P0: float = 1.0
POLY_EXP2_P1: float = 0.69514614
POLY_EXP2_P2: float = 0.22756439
POLY_EXP2_P3: float = 0.07711909

# log2(e) — multiply exp(x) arguments by this to convert to exp2 domain.
# Folding this into an upstream scale factor (e.g., attention's sm_scale)
# saves one FP multiply per element.
EXP_TO_EXP2_SCALE: float = math.log2(math.e)  # 1.4426950408889634

# ---------------------------------------------------------------------------
# Triton code snippet — embed via ``hl.inline_triton()`` or copy into a
# hand-written Triton kernel.  The function takes a *pre-scaled* input
# (i.e., already in the exp2 domain) and returns the approximation.
# ---------------------------------------------------------------------------

EXP2_POLY_TRITON_SRC: str = """\
@triton.jit
def _poly_exp2(x):
    \"\"\"Approximate 2^x using a degree-3 polynomial (3 FMA ops).

    Splits x into integer part n and fractional part f, then
    evaluates 2^f via Horner form and scales by 2^n using
    bit manipulation (ldexp).

    Max relative error (FP32): 8.77e-5.
    \"\"\"
    import triton.language as tl

    # Split into integer and fractional parts
    n = tl.math.floor(x).to(tl.int32)
    f = x - n.to(tl.float32)

    # Horner evaluation: p0 + f*(p1 + f*(p2 + f*p3))
    poly = 0.07711909 + f * 0.0  # p3 (compiler will constant-fold)
    poly = 0.22756439 + f * poly  # p2 + f*p3
    poly = 0.69514614 + f * poly  # p1 + f*(p2 + f*p3)
    poly = 1.0 + f * poly         # p0 + f*(p1 + f*(p2 + f*p3))

    # Scale by 2^n via ldexp (bit manipulation, zero cost)
    return tl.math.ldexp(poly, n)
"""

# ---------------------------------------------------------------------------
# Usage example (for documentation / copy-paste)
# ---------------------------------------------------------------------------

USAGE_EXAMPLE: str = """\
# --- Optimizing attention softmax with exp2 scaling ---
#
# Standard attention:
#   qk = torch.matmul(q, k.T) * sm_scale
#   attn = torch.softmax(qk, dim=-1)   # uses exp() internally
#
# Optimized with exp2 (fold log2e into scale):
#   from helion.experimental.fast_math import EXP_TO_EXP2_SCALE
#   qk = torch.matmul(q, k.T) * (sm_scale * EXP_TO_EXP2_SCALE)
#   attn_unnorm = torch.exp2(qk - qk_max)   # exp2 is 1 HW instruction
#
# This saves one FP multiply per element compared to:
#   torch.exp(qk - qk_max)  # = exp2((qk - qk_max) * log2e)  <- extra multiply
"""
