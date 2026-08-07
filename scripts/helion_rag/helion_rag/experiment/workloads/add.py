"""Tier-2 workload: elementwise add (no qualifying RAG neighbor -> tier2_miss)."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE

    from examples.add import add

    x = torch.randn((1024, 1024), device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn((1024, 1024), device=DEVICE, dtype=torch.bfloat16)
    return add, torch.add, (x, y)


register("add-1024x1024", Workload("add", 1e-2, 1e-1, _build))
