"""Tier-1 corpus workload: softmax two-pass forward."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE
    from helion._testing import HALF_DTYPE

    from examples.softmax import softmax_two_pass

    x = torch.randn((2048, 2048), device=DEVICE, dtype=HALF_DTYPE)
    return softmax_two_pass, lambda t: torch.nn.functional.softmax(t, dim=1), (x,)


register("softmax-2048x2048", Workload("softmax_two_pass", 1e-2, 1e-1, _build))
