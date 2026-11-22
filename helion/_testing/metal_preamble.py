"""Kernels for the c10 preamble's tests (``helion._compiler.metal.c10_preamble``),
each naming c10 helpers of its own kind: a reduction (Helion's reduction
preamble over c10's SIMD reductions), a special function and bfloat math."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="metal", static_shapes=True)
def row_sums(x: torch.Tensor) -> torch.Tensor:
    """Each row's sum."""
    m, _ = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def erf_of(x: torch.Tensor) -> torch.Tensor:
    """``torch.erf`` of each element: ``c10::metal::erf``."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.erf(x[tile])
    return out


@helion.kernel(backend="metal", static_shapes=True)
def bfloat_mul_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``x * y + x`` of bfloat tensors, in bfloat."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] * y[tile] + x[tile]
    return out
