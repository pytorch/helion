"""rms_norm: ``y = weight * x * rsqrt(mean(x**2, -1) + eps)``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import f64
from helion._testing import import_example

_examples_rms_norm = import_example("rms_norm")
rms_norm = _examples_rms_norm.rms_norm
rms_norm_fwd = _examples_rms_norm.rms_norm_fwd
rms_norm_pytorch = _examples_rms_norm.rms_norm_pytorch

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (rms_norm_fwd,)
EPS = 1e-6


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    m, h = case.params["M"], case.params["H"]
    return {
        "x": torch.randn(m, h, generator=gen),
        "weight": torch.ones(h),
    }


def reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x, weight = f64(x, weight)
    return weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def eager(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return rms_norm_pytorch(x, weight, EPS)


def helion(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return rms_norm(x, weight, eps=EPS)


def mlx(x: mx.array, weight: mx.array) -> mx.array:
    import mlx.core as mx

    return mx.fast.rms_norm(x, weight, EPS)
