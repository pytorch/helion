"""Kernels that read a row's reduction dim size beside the row.

Shared by ``test/test_tensor_descriptor_codegen.py`` (rendered on CPU) and
by ``test/test_tensor_descriptor.py`` and ``test/test_reductions.py`` (compiled
and run on a GPU).
"""

from __future__ import annotations

import torch

import helion.language as hl


def onehot_row_update(x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Each row plus its one-hot column (-1 at ``label``) times the one-hot's
    sum: ``hl.arange(row.size(1))`` spans the row's reduction dim."""
    rows, cols = x.shape
    out = torch.empty([rows, cols], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :].to(torch.float32)
        cols_idx = hl.arange(row.size(1))
        onehot = torch.where(cols_idx[None, :] == label[tile_r][:, None], -1.0, 0.0)
        total = torch.sum(onehot, dim=-1, keepdim=True)
        out[tile_r, :] = row + onehot * total
    return out


def half_row_fill(x: torch.Tensor) -> torch.Tensor:
    """The first half of each output row is the row's sum plus the half's
    width: ``n = row.size(1) // 2`` reaches the nested loop as a captured
    value."""
    out = torch.zeros([x.size(0), x.size(1)], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(x.size(0)):
        row = x[tile_r, :].to(torch.float32)
        n = row.size(1) // 2
        total = row.sum(-1)
        for j in range(n):
            out[tile_r, j] = total + n
    return out
