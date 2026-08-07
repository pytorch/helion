"""Tier-1 corpus workload: matmul forward."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE
    from helion._testing import HALF_DTYPE

    from examples.matmul import matmul

    x = torch.randn((1024, 1024), device=DEVICE, dtype=HALF_DTYPE)
    y = torch.randn((1024, 1024), device=DEVICE, dtype=HALF_DTYPE)
    return matmul, torch.matmul, (x, y)


register("matmul-1024x1024x1024", Workload("matmul", 1e-2, 1e-1, _build))
