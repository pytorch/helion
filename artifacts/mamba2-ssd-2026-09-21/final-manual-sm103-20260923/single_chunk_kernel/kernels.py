"""Fused one-chunk Mamba with FP32 CB, weights and residual arithmetic."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.aot_kernel(backend="cute", static_shapes=True)
def mamba2_ssd_cute_scan_fused(
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    batch, length, heads, dim = x.shape
    groups, state = b.shape[2:]
    out = torch.empty_like(x)
    length = hl.specialize(length)
    state = hl.specialize(state)
    for hb, row, col in hl.tile(
        [batch * heads, length, dim], block_size=[1, None, None]
    ):
        bi = hb.begin // heads
        h = hb.begin % heads
        group = h // (heads // groups)
        q = hl.arange(length)
        dts = dt[bi, q, h].float().clamp(min=0.0)
        decay = hl.cumsum(dts * a[h], dim=0)
        row_decay = decay[row]
        cb = hl.dot(c[bi, row, group, :], b[bi, q, group, :].T)
        weights = (
            cb
            * torch.exp((row_decay[:, None] - decay[None, :]).clamp(max=0.0))
            * dts[None, :]
        )
        weights = torch.where(row.index[:, None] >= q[None, :], weights, 0.0).to(
            x.dtype
        )
        value = hl.dot(weights, x[bi, q, h, col])
        value += x[bi, row, h, col].float() * d[h].float()
        out[bi, row, h, col] = value.to(x.dtype)
    return out


def reference_fp64(
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Independent unrounded FP64 oracle, including the two Mamba clamps."""
    repeats = x.shape[2] // b.shape[2]
    xh = x.double().permute(0, 2, 1, 3)
    bh = b.double().repeat_interleave(repeats, dim=2).permute(0, 2, 1, 3)
    ch = c.double().repeat_interleave(repeats, dim=2).permute(0, 2, 1, 3)
    dth = dt.double().permute(0, 2, 1).clamp(min=0.0)
    decay = (dth * a.double()[None, :, None]).cumsum(-1)
    weights = (ch @ bh.transpose(-1, -2)) * torch.exp(
        (decay[..., :, None] - decay[..., None, :]).clamp(max=0.0)
    )
    weights = torch.tril(weights * dth[..., None, :])
    value = weights @ xh + xh * d.double()[None, :, None, None]
    return value.permute(0, 2, 1, 3)


def tuning_reference(*inputs: torch.Tensor) -> torch.Tensor:
    return reference_fp64(*inputs).to(inputs[0].dtype)
