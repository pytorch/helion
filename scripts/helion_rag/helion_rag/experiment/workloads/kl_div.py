"""Tier-1 corpus workload: kl_div_forward (KL divergence)."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE

    from examples.kl_div import kl_div_forward

    bt, v = 2048, 4096
    y_pred = torch.randn((bt, v), device=DEVICE).log_softmax(dim=-1)
    y_true = torch.randn((bt, v), device=DEVICE).softmax(dim=-1)
    reference = torch.nn.KLDivLoss(reduction="batchmean").to(DEVICE)
    return kl_div_forward, reference, (y_pred, y_true)


register("kl_div-2048x4096", Workload("kl_div_forward", 1e-2, 1e-1, _build))
