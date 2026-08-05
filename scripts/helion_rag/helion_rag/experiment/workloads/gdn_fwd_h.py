"""Corpus workload: gated delta net forward-h kernel."""

from __future__ import annotations

from helion_rag.experiment.workloads import Workload
from helion_rag.experiment.workloads import register


def _build():
    import math

    import torch

    from helion._testing import DEVICE

    from examples.gdn_fwd_h import helion_gdn_fwd_h
    from examples.gdn_fwd_h import ref_gdn_fwd_h

    batch, nheads, seqlen, chunk_size, dhead, dstate = 1, 4, 512, 128, 64, 128

    k = torch.randn(batch, seqlen, nheads, dhead, dtype=torch.bfloat16, device=DEVICE)
    k = torch.nn.functional.rms_norm(k, [dhead])
    w = torch.randn(
        batch,
        seqlen // chunk_size,
        chunk_size,
        nheads,
        dhead,
        dtype=torch.float32,
        device=DEVICE,
    )
    wu, _ws, wv = torch.linalg.svd(w.permute(0, 1, 3, 2, 4), full_matrices=False)
    w = torch.einsum("bnhik,bnhkj->bnhij", wu, wv)
    w = (
        w.permute(0, 1, 3, 2, 4)
        .reshape(batch, seqlen, nheads, dhead)
        .to(torch.bfloat16)
    )
    u = torch.randn(batch, seqlen, nheads, dstate, dtype=torch.bfloat16, device=DEVICE)
    u = torch.nn.functional.rms_norm(u, [dstate])
    g = torch.cumsum(
        0.5
        * math.log(1 / dhead)
        * torch.rand(batch, seqlen, nheads, dtype=torch.float32, device=DEVICE),
        dim=1,
    )
    return helion_gdn_fwd_h, ref_gdn_fwd_h, (k, w, u, g, chunk_size)


register("gdn_fwd_h-b1h4s512", Workload("helion_gdn_fwd_h", 1e-2, 1e-1, _build))
