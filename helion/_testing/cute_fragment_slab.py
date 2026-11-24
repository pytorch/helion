"""Kernels and inputs shared by the slab codegen and CUDA numerical tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel


def _config(bound: BoundKernel, **overrides: object) -> helion.Config:
    spec = bound.config_spec
    return helion.Config.from_dict(dict(spec.default_config()) | overrides)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _varlen_anchored_gram(
    q: torch.Tensor,
    k: torch.Tensor,
    gc: torch.Tensor,
    token_base: torch.Tensor,
    valid_len: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """The two tiers of ``_anchored_gram`` over token-major q / k
    ``[T, H, D]``: chunk ``j`` of head ``h`` reads its rows at
    ``token_base[j]`` (the same static windows shifted by one runtime
    offset), each row live below ``valid_len[j]`` and zero past it; gc is
    chunk-major ``[H * NT, C, D]``."""
    NT = token_base.size(0)
    HNT = gc.size(0)
    C = hl.specialize(gc.size(1))
    D = hl.specialize(gc.size(2))
    A = torch.zeros([HNT, C, C], dtype=q.dtype, device=q.device)
    for tile_r in hl.tile(HNT, block_size=1):
        r = tile_r.begin
        j = r % NT
        h = r // NT
        base = token_base[j]
        vlen = valid_len[j]
        dcols = hl.arange(D)
        rows0 = hl.arange(16)
        cols0 = hl.arange(16)
        m0 = rows0 < vlen
        q0 = torch.where(
            m0[:, None],
            hl.load(q, [base + rows0, h, dcols], extra_mask=m0[:, None]),
            0,
        ).float()
        k0 = torch.where(
            m0[:, None],
            hl.load(k, [base + rows0, h, dcols], extra_mask=m0[:, None]),
            0,
        ).float()
        n0 = gc[r, 0, :]
        e0 = torch.exp2(gc[r, rows0, :] - n0[None, :])
        kg0 = (k0 * torch.exp2(n0[None, :] - gc[r, rows0, :])).transpose(-2, -1)
        a0 = hl.dot(q0 * e0, kg0) * scale
        A[r, rows0, cols0] = a0 * (rows0[:, None] >= cols0[None, :]).to(a0.dtype)

        rows1 = 16 + hl.arange(16)
        cols1 = hl.arange(32)
        m1 = rows1 < vlen
        mc1 = cols1 < vlen
        q1 = torch.where(
            m1[:, None],
            hl.load(q, [base + rows1, h, dcols], extra_mask=m1[:, None]),
            0,
        ).float()
        kc1 = torch.where(
            mc1[:, None],
            hl.load(k, [base + cols1, h, dcols], extra_mask=mc1[:, None]),
            0,
        ).float()
        n1 = gc[r, 16, :]
        e1 = torch.exp2(gc[r, rows1, :] - n1[None, :])
        ec1 = torch.exp2(n1[None, :] - gc[r, cols1, :])
        a1 = hl.dot(q1 * e1, (kc1 * ec1).transpose(-2, -1)) * scale
        A[r, rows1, cols1] = a1 * (rows1[:, None] >= cols1[None, :]).to(a1.dtype)
    return A


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _varlen_contiguous_dim_windows(
    x: torch.Tensor, token_base: torch.Tensor, valid_len: torch.Tensor
) -> torch.Tensor:
    """``_contiguous_dim_windows`` at a runtime offset: ``x [H, D, T]``
    (tokens contiguous) read at ``token_base[j] + rows`` under
    ``(rows < valid_len[j])[None, :]``, the limit broadcast over the
    leading dim."""
    NT = token_base.size(0)
    H = x.size(0)
    D = hl.specialize(x.size(1))
    HNT = H * NT
    out = torch.zeros([HNT, 16, 16], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(HNT, block_size=1):
        r = tile_r.begin
        j = r % NT
        h = r // NT
        base = token_base[j]
        vlen = valid_len[j]
        rows0 = hl.arange(16)
        rows1 = 16 + hl.arange(16)
        cols0 = hl.arange(16)
        dcols = hl.arange(D)
        m0 = rows0 < vlen
        m1 = rows1 < vlen
        a = torch.where(
            m0[None, :],
            hl.load(x, [h, dcols, base + rows0], extra_mask=m0[None, :]),
            0,
        )
        b = torch.where(
            m1[None, :],
            hl.load(x, [h, dcols, base + rows1], extra_mask=m1[None, :]),
            0,
        )
        out[r, rows0, cols0] = hl.dot((a.float() * 2.0).transpose(-2, -1), b.float())
    return out


def _varlen_tables(
    lens: tuple[int, ...], c: int, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(token_base [NT], valid_len [NT], NT)`` of sequences of ``lens``
    tokens cut into chunks of ``c`` rows (the last chunk of a sequence
    ragged)."""
    token_base: list[int] = []
    valid_len: list[int] = []
    start = 0
    for length in lens:
        for i in range(0, length, c):
            token_base.append(start + i)
            valid_len.append(min(length - i, c))
        start += length
    i64 = torch.int64
    return (
        torch.tensor(token_base, dtype=i64, device=device),
        torch.tensor(valid_len, dtype=i64, device=device),
        len(token_base),
    )


def _varlen_gram_inputs(
    lens: tuple[int, ...] = (37, 64, 27),
    h: int = 2,
    d: int = 128,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    token_base, valid_len, nt = _varlen_tables(lens, 32, device)
    total = sum(lens)
    if seed is None:
        q = torch.ones(total, h, d, dtype=torch.bfloat16, device=device)
        k = torch.ones(total, h, d, dtype=torch.bfloat16, device=device)
        gc = torch.zeros(h * nt, 32, d, dtype=torch.float32, device=device)
    else:
        g = torch.Generator(device=device).manual_seed(seed)
        q = torch.randn(total, h, d, device=device, generator=g).to(torch.bfloat16)
        k = torch.randn(total, h, d, device=device, generator=g).to(torch.bfloat16)
        gc = (-torch.rand(h * nt, 32, d, device=device, generator=g) * 0.1).cumsum(1)
    return q, k, gc, token_base, valid_len, 0.125


def _varlen_contiguous_inputs(
    lens: tuple[int, ...] = (37, 64, 27),
    h: int = 2,
    d: int = 32,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_base, valid_len, _nt = _varlen_tables(lens, 32, device)
    total = sum(lens)
    if seed is None:
        x = torch.ones(h, d, total, dtype=torch.float32, device=device)
    else:
        g = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(h, d, total, device=device, generator=g)
    return x, token_base, valid_len
