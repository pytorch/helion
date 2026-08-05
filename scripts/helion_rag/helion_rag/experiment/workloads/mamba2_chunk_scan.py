"""Corpus workload: mamba2 chunked scan kernel."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import torch

    from helion._testing import DEVICE
    from helion._testing import HALF_DTYPE

    from examples.mamba2_chunk_scan import helion_mamba2_chunk_scan_kernel
    from examples.mamba2_chunk_scan import ref_chunk_scan

    batch, nheads, ngroups = 1, 4, 1
    seqlen, chunk_size, headdim, dstate = 512, 128, 64, 128
    nchunks = (seqlen + chunk_size - 1) // chunk_size

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=HALF_DTYPE, device=DEVICE)

    cb = randn(batch, nchunks, ngroups, chunk_size, chunk_size)
    x = randn(batch, seqlen, nheads, headdim)
    dt = randn(batch, nheads, nchunks, chunk_size)
    # dA_cumsum kept in [0, 1) so decay exponents stay numerically bounded.
    dA_cumsum = torch.rand(
        batch, nheads, nchunks, chunk_size, dtype=HALF_DTYPE, device=DEVICE
    )
    C = torch.zeros(batch, seqlen, ngroups, dstate, dtype=HALF_DTYPE, device=DEVICE)
    prev_states = torch.zeros(
        batch, nchunks, nheads, headdim, dstate, dtype=HALF_DTYPE, device=DEVICE
    )
    D = torch.zeros(nheads, dtype=HALF_DTYPE, device=DEVICE)
    args = (cb, x, dt, dA_cumsum, C, prev_states, D)
    return helion_mamba2_chunk_scan_kernel, ref_chunk_scan, args


register(
    "mamba2_chunk_scan-b1h4s512",
    Workload("helion_mamba2_chunk_scan_kernel", 1e-2, 1e-1, _build),
)
