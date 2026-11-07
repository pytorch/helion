"""An MMA state recurrence between ordinary materialized phases."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def partitioned_matmul_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    groups, steps, rows, contraction = x.shape
    cols = y.size(3)
    splits = hl.register_block_size(1, 8)
    chunk = helion.cdiv(steps, splits)
    packed = torch.empty_like(x)
    flat_in = x.reshape(-1)
    flat_packed = packed.reshape(-1)
    partial = torch.empty(
        (groups, splits, rows, cols), device=x.device, dtype=torch.float32
    )
    out = torch.empty((groups, rows, cols), device=x.device, dtype=torch.float32)
    for item in hl.tile(flat_in.numel()):
        flat_packed[item] = flat_in[item] * 0.5
    hl.barrier()
    for group, part, tile_rows in hl.tile(
        [groups, splits, rows], block_size=[1, 1, None]
    ):
        total = hl.zeros([tile_rows, cols], dtype=torch.float32)
        for step in hl.grid(part.id * chunk, (part.id + 1) * chunk):
            total = hl.dot(
                packed[group.id, step, tile_rows, :], y[group.id, step, :, :], acc=total
            )
        partial[group.id, part.id, tile_rows, :] = total
    hl.barrier()
    for group_row, row in hl.grid([groups, rows]):
        out[group_row, row, :] = torch.sum(partial[group_row, :, row, :], dim=0)
    return out


@helion.kernel(
    backend="cute", static_shapes=True, cute_graph_cse=False, autotune_effort="none"
)
def duplicate_mma_loads(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    rows, contraction = x.shape
    cols = y.size(1)
    packed = torch.empty_like(x)
    flat_in = x.reshape(-1)
    flat_packed = packed.reshape(-1)
    partial = torch.empty((rows, cols), device=x.device, dtype=torch.float32)
    out = torch.empty_like(partial)
    for item in hl.tile(flat_in.numel()):
        flat_packed[item] = flat_in[item] * 0.5
    hl.barrier()
    for row in hl.tile(rows):
        total = hl.zeros([row, cols], dtype=torch.float32)
        for inner in hl.tile(contraction):
            total = hl.dot(packed[row, inner], y[inner, :], acc=total)
            total = hl.dot(packed[row, inner], y[inner, :], acc=total)
        partial[row, :] = total
    hl.barrier()
    for row in hl.tile(rows):
        out[row, :] = partial[row, :] * 2.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def two_matmul_phases(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    groups, steps, rows, contraction = x.shape
    cols = y.size(3)
    temporary = torch.empty((groups, steps, rows, cols), device=x.device, dtype=x.dtype)
    out = torch.empty((groups, rows, cols), device=x.device, dtype=torch.float32)
    for group, step, row_tile in hl.tile(
        [groups, steps, rows], block_size=[1, 1, None]
    ):
        temporary[group.id, step.id, row_tile, :] = hl.dot(
            x[group.id, step.id, row_tile, :], y[group.id, step.id, :, :]
        ).to(x.dtype)
    hl.barrier()
    for out_group, out_rows in hl.tile([groups, rows], block_size=[1, None]):
        total = hl.zeros([out_rows, cols], dtype=torch.float32)
        for position in hl.grid(steps):
            total = hl.dot(
                temporary[out_group.id, position, out_rows, :],
                y[out_group.id, position, :, :],
                acc=total,
            )
        out[out_group.id, out_rows, :] = total
    return out
