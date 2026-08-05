"""Tier-1 corpus workload: layer_norm forward."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE
    from helion._testing import HALF_DTYPE

    from examples.layer_norm import layer_norm

    dim = 2048
    x = -2.3 + 0.5 * torch.randn((2048, dim), device=DEVICE, dtype=HALF_DTYPE)
    weight = torch.randn((dim,), device=DEVICE, dtype=HALF_DTYPE)
    bias = torch.randn((dim,), device=DEVICE, dtype=HALF_DTYPE)
    return layer_norm, torch.nn.functional.layer_norm, (x, [dim], weight, bias, 1e-4)


register("layer_norm-2048x2048", Workload("layer_norm_fwd", 1e-2, 1e-2, _build))
