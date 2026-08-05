"""Tier-1 corpus workload: jsd_forward (Jensen-Shannon divergence)."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE

    from examples.jsd import HelionJSD
    from examples.jsd import TorchJSDBaseline

    bt, v = 2048, 4096
    log_q = torch.randn((bt, v), device=DEVICE).log_softmax(dim=-1)
    log_p = torch.randn((bt, v), device=DEVICE).log_softmax(dim=-1)
    return HelionJSD(), TorchJSDBaseline(), (log_q, log_p)


register("jsd-2048x4096", Workload("jsd_forward", 1e-2, 1e-1, _build))
