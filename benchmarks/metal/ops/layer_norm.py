"""layer_norm: ``y = (x - mean) * rsqrt(var + eps) * weight + bias``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from . import f64
from helion._testing import import_example

_examples_layer_norm = import_example("layer_norm")
layer_norm = _examples_layer_norm.layer_norm
layer_norm_fwd = _examples_layer_norm.layer_norm_fwd

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (layer_norm_fwd,)
EPS = 1e-5


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    m, n = case.params["M"], case.params["N"]
    return {
        "x": -2.3 + 0.5 * torch.randn(m, n, generator=gen),
        "weight": torch.rand(n, generator=gen),
        "bias": torch.rand(n, generator=gen),
    }


def reference(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    x, weight, bias = f64(x, weight, bias)
    mean = x.mean(-1, keepdim=True)
    var = (x - mean).pow(2).mean(-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + EPS) * weight + bias


def eager(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, EPS)


def helion(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return layer_norm(x, [x.shape[-1]], weight, bias, EPS)


def mlx(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    import mlx.core as mx

    return mx.fast.layer_norm(x, weight, bias, EPS)
