"""Kernels and a config shared by the ``cute_accumulate_stores`` test modules:
a pointwise value stored and summed over tile_r in one loop, the first kernel
of a split column sum that helion.inductor fuses with its producer (an
activation backward and its channel sums)."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# The column reduction's second seed: 16-byte packets of the contiguous
# columns on 4 threads, 32 row threads over 128-row tiles, the sunk vector
# loop and the row lane loop unrolled 8 deep.
SEED_CONFIG = {
    "block_sizes": [1, 32, 128],
    "num_threads": [1, 4, 32],
    "cute_vector_widths": [1, 8, 1],
    "cute_lane_layouts": ["blocked", "blocked", "strided"],
    "cute_vloop_sink": True,
    "cute_lane_unroll": 8,
}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def store_and_column_sums(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``y = x * exp(x)`` stored in x's dtype, and the fp32 sums of y over
    each split's rows (dim 1) of the [splits, rows, columns] input."""
    splits, rows, columns = x.size()
    y = torch.empty_like(x)
    total = torch.empty([splits, columns], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([splits, columns]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            v = x[tile_s, tile_r, tile_c].to(torch.float32)
            value = v * torch.exp(v)
            y[tile_s, tile_r, tile_c] = value.to(x.dtype)
            acc = acc + torch.sum(value, dim=1)
        total[tile_s, tile_c] = acc
    return y, total


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def accumulate_into_stored(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``y += x * exp(x)`` in place and the fp32 sums of ``x * exp(x)`` over
    each split's rows: the loop reads the tensor it stores."""
    splits, rows, columns = x.size()
    total = torch.empty([splits, columns], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([splits, columns]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            v = x[tile_s, tile_r, tile_c].to(torch.float32)
            value = v * torch.exp(v)
            previous = y[tile_s, tile_r, tile_c].to(torch.float32)
            y[tile_s, tile_r, tile_c] = (previous + value).to(y.dtype)
            acc = acc + torch.sum(value, dim=1)
        total[tile_s, tile_c] = acc
    return y, total


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def column_sums(x: torch.Tensor) -> torch.Tensor:
    """The fp32 sums of ``x * exp(x)`` over each split's rows, nothing
    stored per element."""
    splits, rows, columns = x.size()
    total = torch.empty([splits, columns], dtype=torch.float32, device=x.device)
    for tile_s, tile_c in hl.tile([splits, columns]):
        acc = hl.zeros([tile_s, tile_c], dtype=torch.float32)
        for tile_r in hl.tile(rows):
            v = x[tile_s, tile_r, tile_c].to(torch.float32)
            acc = acc + torch.sum(v * torch.exp(v), dim=1)
        total[tile_s, tile_c] = acc
    return total


def store_and_column_sums_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``store_and_column_sums`` in float64 eager."""
    v = x.double()
    value = v * torch.exp(v)
    return value.to(x.dtype), value.sum(1)
