"""Tier-1 corpus workload: rms_norm forward."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE

    from examples.rms_norm import rms_norm
    from examples.rms_norm import rms_norm_pytorch

    x = torch.randn((2048, 1024), device=DEVICE, dtype=torch.float32)
    weight = torch.randn((1024,), device=DEVICE, dtype=torch.float32)
    return rms_norm, rms_norm_pytorch, (x, weight)


register("rms_norm-2048x1024", Workload("rms_norm_fwd", 1e-3, 1e-3, _build))
