"""A varying-bound product with bounded addresses for wide loop indices."""

from __future__ import annotations

import torch

import helion.language as hl


def ring_matmul(a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    m, n = a.size(1), hl.specialize(b.size(2))
    batches = hl.specialize(a.size(0))
    hl.specialize(a.size(2))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        start = bounds[0]
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for step in hl.grid(start, bounds[1]):
            offset = step % batches
            left = a[offset, tm, :].to(b.dtype)
            acc = hl.dot(left, b[offset, :, :], acc=acc)
        out[tm, :] = acc
    return out
