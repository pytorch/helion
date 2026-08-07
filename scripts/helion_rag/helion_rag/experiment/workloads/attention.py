"""Corpus workload: attention forward (non-causal)."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE
    from helion._testing import HALF_DTYPE

    from examples.attention import _attention_baseline
    from examples.attention import attention

    z, h, n_ctx, head_dim = 2, 8, 512, 64
    q, k, v = (
        torch.randn((z, h, n_ctx, head_dim), device=DEVICE, dtype=HALF_DTYPE)
        for _ in range(3)
    )
    return attention, _attention_baseline, (q, k, v)


register("attention-2x8x512x64", Workload("attention", 2e-2, 5e-2, _build))
