"""A dropout residual LayerNorm whose word0 mask every reduction sweep needs.

Shared by the CuTe sweep-value codegen tests (``test/test_cute_sweep_values.py``)
and their GPU twin (``test/test_cute_fuse_two_pass_loads.py``): the mean,
variance and store sweeps each need the mask ``hl.rand`` draws once per row,
so the later sweeps can read the first sweep's draw from a fragment.
"""

from __future__ import annotations

import torch

import helion.language as hl


def dropout_layernorm(
    x: torch.Tensor,
    seeds: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    seed_index: hl.constexpr,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, n = x.shape
    keep_out = torch.empty([rows, n], dtype=torch.bool, device=x.device)
    dropped_out = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty([rows, n], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        keep = hl.rand([tile_r, n], seed=seeds[seed_index]) > 0.1
        dropped = (
            keep.to(torch.float32)
            * (residual[tile_r, :] + x[tile_r, :].to(torch.float32))
            * 1.1111111111111112
        )
        mean = torch.mean(dropped, dim=-1, keepdim=True)
        centered = dropped - mean
        var = torch.sum(centered * centered, dim=-1, keepdim=True) / count
        y = weight[None, :] * centered * torch.rsqrt(var + 1e-5) + bias[None, :]
        keep_out[tile_r, :] = keep
        dropped_out[tile_r, :] = dropped
        out[tile_r, :] = y.to(out.dtype)
    return keep_out, dropped_out, out
