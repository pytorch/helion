"""Kernels the Metal export tests capture (``test_metal_export.py``)."""

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
