"""Copied loop initials, readonly aliases and escaping values for lifetime tests."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def loop_scans(
    x: torch.Tensor,
    steps: int,
    escape: hl.constexpr,
    readonly_alias: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    escaped = torch.empty_like(x)
    for row in hl.tile(x.size(0), block_size=1):
        initial = torch.cumsum(x[row, :], dim=-1)
        readonly = initial + 7 if readonly_alias else x[row, :] + 1
        state = initial
        for _ in hl.grid(steps):
            state = torch.cumsum(state + readonly, dim=-1)
        out[row, :] = state
        escaped[row, :] = initial + 3 if escape else 0
    return out, escaped


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def loop_tensor_bound(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0), block_size=1):
        state = torch.cumsum(x[row, :], dim=-1)
        bound = torch.sum(counts[hl.arange(counts.numel())])
        for tile in hl.tile(bound, block_size=4):
            increment = torch.sum(x[row, tile], dim=-1, keepdim=True)
            state = torch.cumsum(state + increment + bound.to(torch.float32), dim=-1)
            bound = bound + 1
        out[row, :] = state
    return out
