"""Indexed-storage GEMM fixture for computed-fragment logical domains."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def indexed_gemm(
    values: torch.Tensor, weights: torch.Tensor, output: torch.Tensor, rows: int
) -> torch.Tensor:
    rows = hl.specialize(rows)
    for batch, tile in hl.tile([2, rows], block_size=[1, None]):
        columns = hl.arange(16)
        indices = batch.id * rows * 16 + tile.index[:, None] * 16 + columns[None, :]
        result = hl.dot(values[indices], weights[:, :])
        output[indices] = result
    return output


@helion.kernel(backend="cute", static_shapes=True)
def dot_then_reduce(
    x: torch.Tensor, y: torch.Tensor, kind: hl.constexpr, axis: int
) -> torch.Tensor:
    axis = hl.specialize(axis)
    batches, rows, reduction = x.shape
    columns = y.size(2)
    output = torch.empty((batches, rows, columns), device=x.device, dtype=torch.float32)
    for batch in hl.grid(batches):
        r, k, c = hl.arange(rows), hl.arange(reduction), hl.arange(columns)
        value = hl.dot(x[batch, r, k], y[batch, k, c])
        if kind == "max":
            reduced = torch.amax(value, dim=axis, keepdim=True)
        elif kind == "min":
            reduced = torch.amin(value, dim=axis, keepdim=True)
        else:
            reduced = torch.prod(value, dim=axis, keepdim=True)
        output[batch, r, c] = value + reduced
    return output


@helion.kernel(backend="cute", static_shapes=True)
def reshaped_dot_then_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    batches, elements = x.shape
    rows = elements // 8
    columns = y.size(1)
    output = torch.empty((batches, rows, columns), device=x.device, dtype=torch.float32)
    for batch in hl.grid(batches):
        flat = x[batch, hl.arange(elements)]
        left = flat.reshape(-1, 8)
        k, c = hl.arange(8), hl.arange(columns)
        value = hl.dot(left, y[k, c])
        r = hl.arange(rows)
        output[batch, r, c] = value + torch.amax(value, dim=0, keepdim=True)
    return output
