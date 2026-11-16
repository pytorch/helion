"""MMA expressions with root and reduction-loop tile indices."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def masked_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty((y.size(1),), device=x.device, dtype=torch.float32)
    for cols in hl.tile(y.size(1)):
        total = hl.zeros([cols], dtype=torch.float32)
        for rows in hl.tile(x.size(0)):
            score = hl.dot(x[rows, :], y[:, cols])
            visible = (rows.index[:, None] >= cols.index[None, :]) & (
                rows.index[:, None] < x.size(0) // 2
            )
            total = total + torch.where(visible, score, 0).sum(dim=0)
        out[cols] = total
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def causal_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.size(0), y.size(1)), device=x.device, dtype=torch.float32)
    for rows, cols in hl.tile([x.size(0), y.size(1)]):
        total = hl.zeros([rows, cols], dtype=torch.float32)
        for inner in hl.tile(rows.begin + rows.block_size):
            total = hl.dot(x[rows, inner], y[inner, cols], acc=total)
        out[rows, cols] = total
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def recurrent_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.size(0), y.size(1)), device=x.device, dtype=torch.float32)
    for rows, cols in hl.tile([x.size(0), y.size(1)]):
        total = hl.zeros([rows, cols], dtype=torch.float32)
        for inner in hl.tile(x.size(1)):
            total = hl.dot(x[rows, inner], y[inner, cols], acc=total * 0.5)
        out[rows, cols] = total
    return out
