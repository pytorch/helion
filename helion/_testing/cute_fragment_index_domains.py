"""Indexed-storage GEMM fixture for computed-fragment logical domains."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


def _cumsum_column_tile(x: torch.Tensor, batch: hl.Tile, col: hl.Tile) -> torch.Tensor:
    chunk = x.size(1)
    i = hl.arange(chunk)
    lower = (i[:, None] >= i[None, :]).float()
    tri = lower[None, :, :].broadcast_to([batch, chunk, chunk])  # pyrefly: ignore[no-matching-overload]
    return hl.dot(tri, x[batch, :, col].float())


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def windowed_cumsum(
    x: torch.Tensor, out: torch.Tensor, begin: int, end: int
) -> torch.Tensor:
    begin, end = hl.specialize(begin), hl.specialize(end)
    for batch, col in hl.tile([0, begin], [x.size(0), end], block_size=[1, 16]):
        out[batch, :, col] = _cumsum_column_tile(x, batch, col)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def nested_windowed_cumsum(
    x: torch.Tensor, out: torch.Tensor, begin: int, end: int, fixed_end: hl.constexpr
) -> torch.Tensor:
    begin = hl.specialize(begin)
    if fixed_end:
        end = hl.specialize(end)
    for batch in hl.tile(x.size(0), block_size=1):
        for col in hl.tile(begin, end, block_size=16):
            out[batch, :, col] = _cumsum_column_tile(x, batch, col)
    return out


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
