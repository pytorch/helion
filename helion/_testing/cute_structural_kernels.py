"""Shared authored kernels and CPU inputs for structural-policy tests."""

from __future__ import annotations

import torch

import helion.language as hl

CPU_DEVICE = "cpu"


def _joint_sum(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    rows, columns = x.shape
    batches = offsets.size(0) - 1
    out = torch.empty_like(x)
    source = x.view(-1)
    dest = out.view(-1)
    for row in hl.tile(batches):
        starts = offsets[row]
        ends = offsets[row.index + 1]
        lengths = ends - starts
        total = hl.zeros([row], dtype=torch.float32)
        for column in hl.tile(columns):
            partial = hl.zeros([row, column], dtype=torch.float32)
            for jagged in hl.jagged_tile(lengths):
                index = (starts[:, None] + jagged.index[None, :])[:, :, None] * columns
                index = index + column.index[None, None, :]
                value = hl.load(source, [index])
                partial = partial + (value * 1.25 + 2).sum(dim=1)
            total = total + partial.sum(dim=1)
        scale = total / (lengths.to(torch.float32) * columns)
        for column in hl.tile(columns):
            for jagged in hl.jagged_tile(lengths):
                index = (starts[:, None] + jagged.index[None, :])[:, :, None] * columns
                index = index + column.index[None, None, :]
                value = hl.load(source, [index])
                hl.store(dest, [index], value + scale[:, None, None])
    return out


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty([a.size(0), b.size(1)], dtype=torch.float32, device=a.device)
    for row, column in hl.tile([a.size(0), b.size(1)]):
        product = a[row, :] @ b[:, column]
        out[row, column] = product.float() * 1.003
    return out


def _se_args(
    shape: tuple[int, int, int] = (32, 64, 16), dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, ...]:
    m, n, k = shape
    generator = torch.Generator(device=CPU_DEVICE).manual_seed(27)
    return tuple(
        torch.randn(dimensions, generator=generator, dtype=dtype) * 0.25
        for dimensions in ((m, n), (n, k), (k, n))
    )


def _offset_mm(a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    groups = offsets.size(0) - 1
    for group in hl.grid(groups):
        start = offsets[group]
        end = offsets[group + 1]
        count = end - start
        if count != 0:
            for row, column in hl.tile([count, n]):
                acc = hl.zeros([row, column], dtype=torch.float32)
                for reduction in hl.tile(k):
                    x = a[start + row.index, reduction]
                    y = b[reduction, column]
                    acc = torch.addmm(acc, x, y)
                out[start + row.index, column] = acc.to(out.dtype)
    return out


def _pointwise(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out
