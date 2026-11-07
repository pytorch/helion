"""Configuration-owned scalars used by uniform MMA loop expressions."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def repeated_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    repeats = hl.register_block_size(1, 8)
    out = torch.empty((x.size(0), y.size(1)), device=x.device, dtype=torch.float32)
    for rows, cols in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([rows, cols], dtype=torch.float32)
        for _iteration in hl.grid(repeats):
            acc = hl.dot(x[rows, :], y[:, cols], acc=acc)
        out[rows, cols] = acc / repeats
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def tile_position_bound(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.size(0), y.size(1)), device=x.device, dtype=torch.float32)
    for rows, cols in hl.tile([x.size(0), y.size(1)]):
        acc = hl.zeros([rows, cols], dtype=torch.float32)
        for _iteration in hl.grid(rows.begin):
            acc = hl.dot(x[rows, :], y[:, cols], acc=acc)
        out[rows, cols] = acc
    return out
