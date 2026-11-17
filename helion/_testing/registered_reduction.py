"""Registered block extents used as materialized reduction dimensions."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True, autotune_effort="none")
def partitioned_copy_sum(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    parts = hl.register_block_size(1, 8)
    workspace = torch.empty((parts, rows, cols), device=x.device, dtype=x.dtype)
    output = torch.empty_like(x)
    for part, row in hl.grid([parts, rows]):
        # Distinct partitions expose a consumer that repeats one coordinate;
        # their symmetric offsets sum to zero for every configured count.
        workspace[part, row, :] = x[row, :] / parts + (2 * part + 1 - parts)
    hl.barrier()
    for row in hl.grid(rows):
        output[row, :] = torch.sum(workspace[:, row, :], dim=0)
    return output


def registered_scale_sum(x: torch.Tensor) -> torch.Tensor:
    """Bind inside the TileIR CPU fixture, after it activates ENABLE_TILE."""
    scale = hl.register_block_size(1, 8)
    output = torch.empty(x.size(0), device=x.device, dtype=x.dtype)
    for row in hl.grid(x.size(0)):
        output[row] = torch.sum(x[row, :]) / scale
    return output


@helion.kernel(static_shapes=True, autotune_effort="none")
def registered_chunk_max(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    chunk = hl.register_block_size(16, 64)
    parts = helion.cdiv(cols, chunk)
    out = torch.empty((rows, parts), dtype=x.dtype, device=x.device)
    flat = x.view(-1)
    for row, part in hl.grid([rows, parts]):
        column = hl.arange(chunk) + part * chunk
        # PyTorch wheel stubs omit this supported SymInt/Tensor sum.
        # pyrefly: ignore[unsupported-operation]
        value = hl.load(flat, [row * cols + column], extra_mask=column < cols)
        value = torch.where(column < cols, value, float("-inf"))
        out[row, part] = value.amax()
    return out
