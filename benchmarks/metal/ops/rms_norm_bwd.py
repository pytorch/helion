"""rms_norm-bwd: gradients of rms_norm for x and weight, backward only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import f64
from . import rms_norm as fwd
from helion._testing import import_example

_examples_rms_norm = import_example("rms_norm")
rms_norm_bwd = _examples_rms_norm.rms_norm_bwd
rms_norm_fwd = _examples_rms_norm.rms_norm_fwd

if TYPE_CHECKING:
    from ..manifest import Case

MODE = "bwd"
GRAD_WRT = ("x", "weight")
HELION_KERNELS = (rms_norm_fwd, rms_norm_bwd)


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    host = fwd.inputs(case, gen)
    host["dy"] = 0.1 * torch.randn(host["x"].shape, generator=gen)
    return host


def reference(
    x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    x, weight = (t.requires_grad_() for t in f64(x, weight))
    (dy,) = f64(dy)
    y = fwd.reference(x, weight)
    return torch.autograd.grad(y, (x, weight), dy)


eager = fwd.eager
helion = fwd.helion
mlx = fwd.mlx
