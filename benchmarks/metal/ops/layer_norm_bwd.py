"""layer_norm-bwd: gradients of layer_norm for x, weight and bias."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import f64
from . import layer_norm as fwd
from helion._testing import import_example

_examples_layer_norm = import_example("layer_norm")
layer_norm_bwd = _examples_layer_norm.layer_norm_bwd
layer_norm_fwd = _examples_layer_norm.layer_norm_fwd

if TYPE_CHECKING:
    from ..manifest import Case

MODE = "bwd"
GRAD_WRT = ("x", "weight", "bias")
HELION_KERNELS = (layer_norm_fwd, layer_norm_bwd)


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    host = fwd.inputs(case, gen)
    host["dy"] = 0.1 * torch.randn(host["x"].shape, generator=gen)
    return host


def reference(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, dy: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    x, weight, bias = (t.requires_grad_() for t in f64(x, weight, bias))
    (dy,) = f64(dy)
    y = fwd.reference(x, weight, bias)
    return torch.autograd.grad(y, (x, weight, bias), dy)


eager = fwd.eager
helion = fwd.helion
mlx = fwd.mlx
