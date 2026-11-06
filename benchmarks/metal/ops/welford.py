"""welford: bf16 layer norm over the last dim of a (262144, D) input."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import REFERENCE_ROWS
from helion._testing import import_example

_examples_welford = import_example("welford")
eager_layer_norm = _examples_welford.eager_layer_norm
welford = _examples_welford.welford

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (welford,)
EPS = 1e-5


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    s, d = case.params["S"], case.params["D"]
    # TritonBench's (p1, p2, p3) order and draw order: weight, bias, x.
    return {
        "weight": torch.randn(d, generator=gen, dtype=torch.bfloat16),
        "bias": torch.randn(d, generator=gen, dtype=torch.bfloat16),
        "x": torch.randn(s, d, generator=gen, dtype=torch.bfloat16),
    }


def reference(
    weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Float64 per row chunk, stored as float32 (exact far below bf16 ulp)."""
    out = torch.empty(x.shape, dtype=torch.float32)
    w, b = weight.double(), bias.double()
    for start in range(0, x.shape[0], REFERENCE_ROWS * 16):
        rows = x[start : start + REFERENCE_ROWS * 16].double()
        mean = rows.mean(-1, keepdim=True)
        var = (rows - mean).pow(2).mean(-1, keepdim=True)
        out[start : start + rows.shape[0]] = (rows - mean) * torch.rsqrt(
            var + EPS
        ) * w + b
    return out


def eager(weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return eager_layer_norm(weight, bias, x, EPS)


def helion(weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return welford(weight, bias, x, EPS)


def mlx(weight: mx.array, bias: mx.array, x: mx.array) -> mx.array:
    import mlx.core as mx

    return mx.fast.layer_norm(x, weight, bias, EPS)
