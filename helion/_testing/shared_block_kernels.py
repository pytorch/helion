"""Loops that share a registered block size, so they have the same block ids.

Shared by the CuTe codegen checks (``test/test_cute_shared_block_loops.py``)
and their GPU numerics (``test/test_cute_lowerings.py``): a two-pass softmax
whose two device loops over ``n`` take ``block_n`` and run on one tile
strategy, a root loop over ``n`` beside another root's device loop over it,
and device loops over ``n`` in two roots.
"""

from __future__ import annotations

import torch

import helion.language as hl


def shared_block_softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim of ``[batch, heads, rows, n]``: a running
    max and sum in one loop over ``n``, the normalized values in a second
    over the same registered block size."""
    batch, heads, rows, n = x.shape
    out = torch.empty_like(x)
    block_n = hl.register_block_size(n)
    for tile_b, tile_h, tile_r in hl.tile([batch, heads, rows]):
        row_max = hl.full([tile_b, tile_h, tile_r], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tile_b, tile_h, tile_r], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_n):
            values = x[tile_b, tile_h, tile_r, tile_n]
            new_max = torch.maximum(row_max, torch.amax(values, dim=-1))
            row_sum = row_sum * torch.exp(row_max - new_max) + torch.sum(
                torch.exp(values - new_max[:, :, :, None]), dim=-1
            )
            row_max = new_max
        for tile_n in hl.tile(n, block_size=block_n):
            values = x[tile_b, tile_h, tile_r, tile_n]
            out[tile_b, tile_h, tile_r, tile_n] = (
                torch.exp(values - row_max[:, :, :, None]) / row_sum[:, :, :, None]
            )
    return out


def root_beside_a_device_loop(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``2 * x`` in a root loop over ``block_n``, and ``y + 1`` in a second
    root whose device loop takes the same registered block size."""
    n = x.size(0)
    doubled = torch.empty_like(x)
    shifted = torch.empty_like(y)
    block_n = hl.register_block_size(n)
    for tile_a in hl.tile(n, block_size=block_n):
        doubled[tile_a] = x[tile_a] * 2
    for tile_m in hl.tile(y.size(0)):
        for tile_n in hl.tile(n, block_size=block_n):
            shifted[tile_m, tile_n] = y[tile_m, tile_n] + 1
    return doubled, shifted


def device_loops_of_two_roots(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x + 1`` over a 1-D grid and ``2 * y`` over a 2-D grid, each in a
    device loop over the same registered block size."""
    n = x.size(1)
    shifted = torch.empty_like(x)
    doubled = torch.empty_like(y)
    block_n = hl.register_block_size(n)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(n, block_size=block_n):
            shifted[tile_m, tile_n] = x[tile_m, tile_n] + 1
    for tile_a, tile_b in hl.tile([y.size(0), y.size(1)]):
        for tile_n in hl.tile(n, block_size=block_n):
            doubled[tile_a, tile_b, tile_n] = y[tile_a, tile_b, tile_n] * 2
    return shifted, doubled
