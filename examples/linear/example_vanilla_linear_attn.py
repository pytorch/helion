"""
Vanilla Linear Attention Example
=================================

Chunked linear attention with no decay and no correction term.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .linear_attention_engine import LinearAttentionVariant
from .linear_attention_harness import DTYPE
from .linear_attention_harness import Inputs
from .linear_attention_harness import LinearAttentionExampleHarness
from helion._testing import DEVICE

try:
    from fla.ops.linear_attn import (
        chunk_linear_attn as _fla_chunk,  # pyrefly: ignore[missing-import]
    )
except ImportError:
    _fla_chunk = None
_fla_chunk_linear_attn: Any = _fla_chunk


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
    g = torch.zeros(B, H, T, device=device, dtype=dtype)
    return Inputs(q=q, k=k, v=v, scale=1.0 / math.sqrt(D), g=g)


def _fla_fwd(i: Inputs, scale: float) -> torch.Tensor:
    o = _fla_chunk_linear_attn(i.q, i.k, i.v, scale=scale, normalize=False)
    return o[0] if isinstance(o, tuple) else o


VARIANT = LinearAttentionVariant.VANILLA

HARNESS = LinearAttentionExampleHarness(
    name="vanilla_linear_attn",
    title="Vanilla Linear Attention",
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
