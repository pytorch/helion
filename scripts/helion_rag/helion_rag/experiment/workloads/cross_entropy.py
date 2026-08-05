"""Tier-1 corpus workload: cross_entropy forward."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE
    from helion._testing import LONG_INT_TYPE

    from examples.cross_entropy import cross_entropy

    n, v = 2048, 4096
    logits = torch.randn((n, v), device=DEVICE, dtype=torch.float32)
    labels = torch.randint(0, v, (n,), device=DEVICE, dtype=LONG_INT_TYPE)
    return cross_entropy, torch.nn.functional.cross_entropy, (logits, labels)


register("cross_entropy-2048x4096", Workload("cross_entropy", 1e-4, 1e-4, _build))
