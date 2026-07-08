"""
Full GLA (Diagonal Decay) Example
==================================

Chunked linear attention with per-dimension (diagonal) decay gates.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .linear_attention_engine import LinearAttentionVariant
from .linear_attention_harness import DTYPE
from .linear_attention_harness import Inputs
from .linear_attention_harness import LinearAttentionExampleHarness
from helion._testing import DEVICE


def _make_inputs(
    B: int,
    H: int,
    T: int,
    D: int,
    DV: int,
    dtype: torch.dtype = DTYPE,
    device: str | torch.device = DEVICE,
    requires_grad: bool = False,
) -> Inputs:
    q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(
        B, H, T, DV, device=device, dtype=dtype, requires_grad=requires_grad
    )
    g = F.logsigmoid(torch.randn(B, H, T, D, device=device, dtype=dtype))
    return Inputs(q=q, k=k, v=v, scale=1.0 / math.sqrt(D), g=g)


VARIANT = LinearAttentionVariant.FULL_GLA

HARNESS = LinearAttentionExampleHarness(
    name="full_gla",
    title="Full GLA",
    variant=VARIANT,
    make_inputs=_make_inputs,
)

# Module API consumed by run_linattn.py: test / benchmark / accuracy.
test = HARNESS.test
benchmark = HARNESS.benchmark
accuracy = HARNESS.accuracy


def main() -> None:
    print(f"=== {HARNESS.title} ===")
    test()
    print()
    benchmark()


if __name__ == "__main__":
    main()
