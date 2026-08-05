"""Corpus workload: mamba2 chunked state kernel."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE
    from helion._testing import HALF_DTYPE

    from examples.mamba2_chunk_state import helion_mamba2_chunk_state_kernel
    from examples.mamba2_chunk_state import ref_chunk_state

    batch, nheads, ngroups = 1, 4, 1
    seqlen, chunk_size, headdim, dstate = 512, 128, 64, 128
    nchunks = (seqlen + chunk_size - 1) // chunk_size

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, dtype=HALF_DTYPE, device=DEVICE)

    B = rand(batch, seqlen, ngroups, dstate)
    x = rand(batch, seqlen, nheads, headdim)
    dt = rand(batch, nheads, nchunks, chunk_size)
    dA_cumsum = rand(batch, nheads, nchunks, chunk_size)
    args = (B, x, dt, dA_cumsum)
    return helion_mamba2_chunk_state_kernel, ref_chunk_state, args


register(
    "mamba2_chunk_state-b1h4s512",
    Workload("helion_mamba2_chunk_state_kernel", 1e-2, 1e-1, _build),
)
