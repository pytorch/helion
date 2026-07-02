"""
Vanilla Linear Attention Example
=================================

Chunked linear attention with no decay and no correction term.
"""

from __future__ import annotations

from typing import Any

import torch

from .linear_attention_engine import chunked_linear_attn
from .linear_attention_harness import Inputs
from .linear_attention_harness import LinearAttentionVariant
from .linear_attention_harness import run_accuracy as _run_accuracy
from .linear_attention_harness import run_benchmark as _run_benchmark
from .linear_attention_harness import run_test as _run_test
from .linear_attention_utils import chunked_linear_attn_reference
from .linear_attention_utils import make_vanilla_linear_attn_inputs
from .linear_attention_utils import naive_recurrent_reference
from helion._testing import DEVICE

try:
    from fla.ops.linear_attn import (
        chunk_linear_attn as _fla_chunk,  # pyrefly: ignore[missing-import]
    )
except ImportError:
    _fla_chunk = None
_fla_chunk_linear_attn: Any = _fla_chunk

# Test/benchmark config
B, H, T, D, DV = 2, 4, 128, 32, 16
C = 32
DTYPE = torch.bfloat16
BENCH_CONFIGS = [(1, 32, 2048, 128, 128), (1, 32, 4096, 128, 128)]
BENCH_C = 64


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
    q, k, v, g, scale = make_vanilla_linear_attn_inputs(
        B, H, T, D, DV, dtype=dtype, device=device, requires_grad=requires_grad
    )
    return Inputs(q=q, k=k, v=v, scale=scale, g=g)


def _helion_fwd(i: Inputs, C: int) -> torch.Tensor:
    return chunked_linear_attn(i.q, i.k, i.v, None, C=C, scale=i.scale)


def _helion_fb(i: Inputs, grad_out: torch.Tensor, C: int) -> None:
    chunked_linear_attn(i.q, i.k, i.v, None, C=C, scale=i.scale).backward(grad_out)


def _fla_fwd(i: Inputs, scale: float) -> torch.Tensor:
    o = _fla_chunk_linear_attn(i.q, i.k, i.v, scale=scale, normalize=False)
    return o[0] if isinstance(o, tuple) else o


def _fla_fb(i: Inputs, go_t: torch.Tensor, scale: float) -> None:
    o = _fla_chunk_linear_attn(i.q, i.k, i.v, scale=scale, normalize=False)
    (o[0] if isinstance(o, tuple) else o).backward(go_t)


def _reference(i: Inputs) -> torch.Tensor:
    assert i.g is not None
    return naive_recurrent_reference(i.q, i.k, i.v, i.g, q_scale=i.scale)


def _chunked_reference(i: Inputs, C: int) -> torch.Tensor:
    assert i.g is not None
    return chunked_linear_attn_reference(i.q * i.scale, i.k, i.v, i.g, C=C)


VARIANT = LinearAttentionVariant(
    name="vanilla_linear_attn",
    title="Vanilla Linear Attention",
    make_inputs=_make_inputs,
    helion_fwd=_helion_fwd,
    helion_fb=_helion_fb,
    reference=_reference,
    chunked_reference=_chunked_reference,
    fla_fwd=_fla_fwd if _fla_chunk else None,
    fla_fb=_fla_fb if _fla_chunk else None,
    dtype=DTYPE,
)


# Module API consumed by run_linattn.py: test / benchmark / accuracy.
def test() -> None:
    _run_test(VARIANT, (B, H, T, D, DV), C)


def benchmark(
    configs: list | None = None,
) -> list[tuple[str, float, float, float, float]]:
    return _run_benchmark(
        VARIANT, configs if configs is not None else BENCH_CONFIGS, BENCH_C
    )


def accuracy(configs: list | None = None) -> list[tuple[bool, bool]]:
    return _run_accuracy(
        VARIANT, configs if configs is not None else BENCH_CONFIGS, BENCH_C
    )


def main() -> None:
    print("=== Vanilla Linear Attention ===")
    test()
    print()
    benchmark()


if __name__ == "__main__":
    main()
