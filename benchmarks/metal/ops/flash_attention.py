"""flash_attention: non-causal ``softmax(q @ k^T * D**-0.5) @ v`` in bf16.

The reference runs in float32 on the GPU, one (batch, head) at a time: a
float64 host reference of the S=8192 row is 3.3 TFLOP.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from helion._testing import import_example

attention = import_example("attention").attention

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (attention,)


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    shape = tuple(case.params[key] for key in ("B", "H", "S", "D"))
    return {
        name: torch.randn(shape, generator=gen, dtype=torch.bfloat16)
        for name in ("q", "k", "v")
    }


def _scale(q: torch.Tensor | mx.array) -> float:
    return 1.0 / math.sqrt(q.shape[-1])


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = torch.empty(q.shape, dtype=torch.float32)
    scale = _scale(q)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            qh, kh, vh = (t[b, h].to("mps", torch.float32) for t in (q, k, v))
            scores = torch.softmax((qh @ kh.T) * scale, dim=-1)
            out[b, h] = (scores @ vh).cpu()
    return out


def eager(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v)


def helion(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return attention(q, k, v)[0]


def mlx(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    import mlx.core as mx

    return mx.fast.scaled_dot_product_attention(q, k, v, scale=_scale(q))
