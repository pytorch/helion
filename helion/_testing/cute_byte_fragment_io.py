"""Byte-window transport exercised by computed-fragment consumers."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def byte_window_gram(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, rows, _cols = x.shape
    gram = torch.empty((batch, rows, rows), device=x.device, dtype=torch.float32)
    copied = torch.empty_like(x)
    for b in hl.tile(batch, block_size=1):
        row = hl.arange(rows)
        tile = x[b, row, :]
        gram[b, row, :] = hl.dot(tile.float(), tile.float().transpose(-1, -2))
        copied[b, row, :] = tile + 1
    return gram, copied
