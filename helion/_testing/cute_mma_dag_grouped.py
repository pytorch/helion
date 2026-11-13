"""Matrix products with typed grouped auxiliary operands."""

from __future__ import annotations

import torch

import helion.language as hl


def grouped_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scales: torch.Tensor,
    group: int,
    row_group: hl.constexpr,
    wrap_scale: hl.constexpr,
    index_offset: int = 0,
) -> torch.Tensor:
    group = hl.specialize(group)
    index_offset = hl.specialize(index_offset)
    m, k = a.size()
    n = hl.specialize(b.size(1))
    k = hl.specialize(k)
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for _repeat in hl.grid(2):
            if row_group:
                scale = scales[tm.index // group, :]
            elif index_offset:
                scale = scales[tm, (hl.arange(k) + index_offset) // group]
            else:
                scale = scales[tm, hl.arange(k) // group]
            typed = scale + 1 if wrap_scale else scale
            value = (a[tm, :].float() * torch.exp2(typed.float() - 127)).to(b.dtype)
            acc = hl.dot(value, b[:, :], acc=acc)
        out[tm, :] = acc
    return out


def grouped_loop_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scales: torch.Tensor,
    group: int,
) -> torch.Tensor:
    group = hl.specialize(group)
    m, k = a.size()
    n = hl.specialize(b.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            scale = scales[tm, tk.index // group].float()
            value = (a[tm, tk].float() * torch.exp2(scale - 127)).to(b.dtype)
            acc = hl.dot(value, b[tk, :], acc=acc)
        out[tm, :] = acc
    return out


def grouped_vector_matmul(
    a: torch.Tensor, b: torch.Tensor, scales: torch.Tensor, grouped: hl.constexpr
) -> torch.Tensor:
    m, k = a.size()
    n = hl.specialize(b.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(k):
            scale = scales[tk.index // 3] if grouped else scales[tk]
            value = (a[tm, tk].float() * scale.float()).to(b.dtype)
            acc = hl.dot(value, b[tk, :], acc=acc)
        out[tm, :] = acc
    return out


def grouped_carry_matmul(
    a: torch.Tensor, b: torch.Tensor, seed: torch.Tensor, grouped: hl.constexpr
) -> torch.Tensor:
    m, k = a.size()
    n = hl.specialize(b.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        acc = (seed[tm, hl.arange(n) // 2] if grouped else seed[tm, :]).float()
        for tk in hl.tile(k):
            acc = hl.dot(a[tm, tk], b[tk, :], acc=acc)
        out[tm, :] = acc
    return out
