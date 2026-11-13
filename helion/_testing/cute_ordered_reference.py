"""A tunable workspace with a recurrent producer and a stable final result."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def partitioned_dot_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    rows, reduction = x.shape
    steps, _, cols = y.shape
    splits = hl.register_block_size(1, 8)
    temporary = torch.empty((splits, rows, cols), device=x.device, dtype=torch.float32)
    output = torch.empty((rows, cols), device=x.device, dtype=torch.float32)
    for split_tile in hl.tile(splits, block_size=1):
        split = split_tile.begin
        r, k, c = hl.arange(rows), hl.arange(reduction), hl.arange(cols)
        left = x[r, k]
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for step in hl.grid(split, steps, splits):
            shifted = left + torch.amax(left, dim=0)[None, :]
            acc = hl.dot(shifted, y[step, k, c], acc=acc)
        temporary[split, r, c] = acc
    hl.barrier()
    for row, col in hl.tile([rows, cols]):
        result = hl.zeros((row, col), dtype=torch.float32)
        for part in hl.grid(splits):
            result += temporary[part, row, col]
        output[row, col] = result
    return output


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def local_histogram_phases(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    temporary = torch.empty((rows, 8), device=x.device, dtype=torch.float32)
    output = torch.empty_like(temporary)
    for row in hl.grid(rows):
        counts = hl.zeros((8,), dtype=torch.float32)
        index = hl.arange(cols)
        hl.atomic_add(counts, [x[row, index] % 8], 1.0)
        temporary[row, :] = counts
    hl.barrier()
    for row in hl.grid(rows):
        for column in hl.grid(8):
            output[row, column] = temporary[row, column] * 2
    return output
