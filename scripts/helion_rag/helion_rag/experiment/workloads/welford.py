"""Corpus workload: welford layernorm."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE

    from examples.welford import eager_layer_norm
    from examples.welford import welford

    s, d = 2048, 1024
    weight = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    bias = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    x = torch.rand((s, d), device=DEVICE, dtype=torch.float32)
    return welford, eager_layer_norm, (weight, bias, x)


register("welford-2048x1024", Workload("welford", 1e-2, 1e-1, _build))
