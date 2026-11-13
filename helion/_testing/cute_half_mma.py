"""Shared Helion kernels for computed-fragment half MMA tests."""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def cast_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    rows = hl.specialize(x.size(1))
    cols = hl.specialize(y.size(2))
    out = torch.empty((x.size(0), rows, cols), dtype=torch.float32, device=x.device)
    for batch in hl.tile(x.size(0), block_size=1):
        out[batch, :, :] = hl.dot(x[batch, :, :].to(y.dtype), y[batch, :, :])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def batched_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    rows = hl.specialize(x.size(1))
    cols = hl.specialize(y.size(2))
    out = torch.empty((x.size(0), rows, cols), dtype=torch.float32, device=x.device)
    for batch in hl.tile(x.size(0), block_size=1):
        out[batch, :, :] = hl.dot(x[batch, :, :], y[batch, :, :])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def computed_dot(x: torch.Tensor, y: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
    rows = hl.specialize(x.size(1))
    cols = hl.specialize(y.size(2))
    reduction = hl.specialize(x.size(2))
    out = torch.empty((x.size(0), rows, cols), dtype=acc.dtype, device=acc.device)
    for batch in hl.grid(x.size(0)):
        r, k, c = hl.arange(rows), hl.arange(reduction), hl.arange(cols)
        left = (x[batch, r, k].float() * 0.5).to(x.dtype)
        value = hl.dot(left, y[batch, k, c], acc=acc[batch, r, c])
        out[batch, r, c] = value + torch.amax(value, dim=0)[None, :]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def recurrent_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    batches, steps, m, k = x.shape
    m = hl.specialize(m)
    k = hl.specialize(k)
    n = hl.specialize(y.size(-1))
    out = torch.empty((batches, m, n), device=x.device, dtype=torch.float32)
    for batch in hl.grid(batches):
        r, contraction, c = hl.arange(m), hl.arange(k), hl.arange(n)
        acc = hl.zeros((m, n), dtype=torch.float32)
        for step in hl.grid(steps):
            left = (x[batch, step, r, contraction].float() * 0.5).to(x.dtype)
            acc = hl.dot(left, y[batch, step, contraction, c], acc=acc * 0.5)
        out[batch, r, c] = acc
    return out
