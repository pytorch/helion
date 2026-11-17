"""Shared kernels for tensor view lowering tests."""

from __future__ import annotations

import torch

import helion.language as hl


def chunk_unbind_axes(
    x: torch.Tensor, leading_axis: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    n, d, k = x.shape
    d, k = hl.specialize((d, k))
    lo = x.new_empty((n, d // 2, k))
    hi = torch.empty_like(lo)
    for tile in hl.tile(n):
        # pyrefly: ignore [bad-argument-type]
        values = x[tile, :, :].reshape(tile, d, k)
        if leading_axis:
            # Default dim=0 and the unbound Tensor method form.
            transposed = values.permute(1, 0, 2)
            left, right = torch.Tensor.chunk(transposed, 2)
            a = left.permute(1, 0, 2)
            b = right.permute(1, 0, 2)
        else:
            a, b = values.chunk(2, dim=-2)
        grouped = torch.stack((a, b), dim=0)
        c, e = grouped.unbind()
        lo[tile, :, :] = c
        hi[tile, :, :] = e
    return lo, hi
