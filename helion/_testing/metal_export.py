"""Kernels the Metal export tests capture: on CPU tensors to read their
launches (``test_metal_export.py``), and on MPS to run them under MLX
(``test_metal_export_mlx.py``)."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def add_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(
    backend="metal", static_shapes=False, config=helion.Config(block_sizes=[64])
)
def shift(x: torch.Tensor, f: float) -> torch.Tensor:
    """``x * f + 1``: ``f`` binds by value, ``x``'s size and stride are
    host constants."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * f + 1
    return out


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def row_sum(x: torch.Tensor) -> torch.Tensor:
    """A reduction over the threads of a threadgroup, through scratch."""
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tm in hl.tile(m):
        out[tm] = x[tm, :].sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def scatter_even(x: torch.Tensor) -> torch.Tensor:
    """Writes the even positions of an output the host zeroes."""
    (n,) = x.shape
    out = torch.zeros(2 * n, dtype=x.dtype, device=x.device)
    bm = hl.register_block_size(16, 64)
    for g in hl.grid((n + bm - 1) // bm):
        src = hl.arange(bm) + g * bm
        v = hl.load(x, [src], extra_mask=src < n)
        hl.store(out, [2 * src], v, extra_mask=src < n)
    return out


@helion.kernel(backend="metal", static_shapes=True, autotune_effort="none")
def position_named(threadgroup_position_in_grid: torch.Tensor) -> torch.Tensor:
    """``x + 1`` for an input named like one of MLX's position attributes."""
    out = torch.empty_like(threadgroup_position_in_grid)
    for tile in hl.tile(threadgroup_position_in_grid.size(0)):
        out[tile] = threadgroup_position_in_grid[tile] + 1
    return out
