"""
Gated Delta Rule Example
========================

Gated DeltaNet: L2-normalized keys, scalar gated decay, and rank-1 correction
via beta.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .linear_attention_engine import DecayType
from .linear_attention_engine import LinearAttentionVariant
from .linear_attention_harness import DTYPE
from .linear_attention_harness import Inputs
from .linear_attention_harness import LinearAttentionHarness
from helion._testing import DEVICE

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk,  # pyrefly: ignore[missing-import]
    )
except ImportError:
    _fla_chunk = None
_fla_chunk_gated_delta_rule: Any = _fla_chunk


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
    k = F.normalize(torch.randn(B, H, T, D, device=device, dtype=dtype), dim=-1)
    if requires_grad:
        k = k.detach().requires_grad_(True)
    v = torch.randn(
        B, H, T, DV, device=device, dtype=dtype, requires_grad=requires_grad
    )
    beta = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=dtype))
    g = F.logsigmoid(torch.randn(B, H, T, device=device, dtype=dtype))
    return Inputs(q=q, k=k, v=v, scale=1.0 / math.sqrt(D), g=g, beta=beta)


def _fla_fwd(i: Inputs, scale: float) -> torch.Tensor:
    # FLA arg order: q, k, v, g, beta (g before beta)
    o, _ = _fla_chunk_gated_delta_rule(i.q, i.k, i.v, i.g, i.beta, scale=scale)
    return o


VARIANT = LinearAttentionVariant(decay=DecayType.SCALAR, correction=True)

HARNESS = LinearAttentionHarness(
    name="gated_delta_rule",
    title="Gated Delta Rule",
    variant=VARIANT,
    make_inputs=_make_inputs,
    fla_fwd=_fla_fwd if _fla_chunk else None,
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
