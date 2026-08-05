"""Corpus workload: rope forward."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE
    from helion._testing import HALF_DTYPE

    from examples.rope import rope_fwd
    from examples.rope import rope_pytorch

    batch, q_heads, k_heads, seq_len, head_dim = 1, 4, 2, 128, 64
    q = torch.randn(
        [batch, q_heads, seq_len, head_dim], device=DEVICE, dtype=HALF_DTYPE
    )
    k = torch.randn(
        [batch, k_heads, seq_len, head_dim], device=DEVICE, dtype=HALF_DTYPE
    )
    angles = torch.randn([batch, seq_len, head_dim], device=DEVICE, dtype=HALF_DTYPE)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return rope_fwd, rope_pytorch, (q, k, cos, sin)


register("rope-1x4x2x128x64", Workload("rope_fwd", 1e-2, 1e-2, _build))
