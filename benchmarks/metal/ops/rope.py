"""rope: rotate-half rotary embedding of q (32 heads) and k (8 heads).

cos/sin are LLaMA's default rotary tables (theta 10000), as TritonBench builds
them with ``transformers.LlamaRotaryEmbedding``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from helion._testing import import_example

_examples_rope = import_example("rope")
rope = _examples_rope.rope
rope_fwd = _examples_rope.rope_fwd
rope_pytorch = _examples_rope.rope_pytorch

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (rope_fwd,)
Q_HEADS = 32
KV_HEADS = 8
THETA = 10000.0


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    hidden, t = case.params["hidden"], case.params["T"]
    hd = hidden // Q_HEADS
    inv_freq = 1.0 / THETA ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    freqs = torch.arange(t, dtype=torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)[None]
    return {
        "q": torch.randn(1, Q_HEADS, t, hd, generator=gen),
        "k": torch.randn(1, KV_HEADS, t, hd, generator=gen),
        "cos": emb.cos(),
        "sin": emb.sin(),
    }


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def reference(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, cos, sin = (t.double() for t in (q, k, cos, sin))
    cos, sin = cos[:, None], sin[:, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def eager(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return rope_pytorch(q, k, cos, sin)


def helion(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return rope(q, k, cos, sin)


def mlx(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    import mlx.core as mx

    # Composed from the given tables: mx.fast.rope evaluates cos/sin in-kernel,
    # which drifts from the tables by up to ~3e-4 at long positions.
    cos, sin = cos[:, None], sin[:, None]
    half = q.shape[-1] // 2

    def rotate_half(x: mx.array) -> mx.array:
        return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
