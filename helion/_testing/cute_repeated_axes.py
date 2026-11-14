"""Repeated-axis tensor kernels for CPU geometry and native value checks."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def full_slice_add(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for batch in hl.tile(x.size(0), block_size=1):
        out[batch, :, :] = x[batch, :, :] + 1.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def grid_full_slice_add(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for batch in hl.grid(x.size(0)):
        out[batch, :, :] = x[batch, :, :] + 1.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def full_slice_tiled_dot(
    q: torch.Tensor, k: torch.Tensor, epilogue: hl.constexpr
) -> torch.Tensor:
    batch_size = q.size(0)
    channels = hl.specialize(q.size(1))
    depth = q.size(2)
    out = torch.empty(
        [batch_size, channels, channels], dtype=torch.float32, device=q.device
    )
    for batch in hl.tile(batch_size):
        acc = hl.zeros([batch, channels, channels], dtype=torch.float32)
        for tile_k in hl.tile(depth):
            left = q[batch, :, tile_k]
            right = k[batch, :, tile_k]
            acc = hl.dot(left, right.transpose(-2, -1), acc=acc)
        if epilogue:
            out[batch, :, :] = acc * 0.5
        else:
            out[batch, :, :] = acc
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def full_slice_row_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[:2], dtype=x.dtype, device=x.device)
    for batch in hl.tile(x.size(0), block_size=1):
        out[batch, :] = (x[batch, :, :] * 2.0).sum(-1)
    return out
