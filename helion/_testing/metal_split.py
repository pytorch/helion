"""Kernels the Metal split-reduction tests share: the CPU-only modules
(``test/test_metal_split_reduction_heuristic.py``,
``test/test_metal_masked_branch.py``) read their seeds and code and the GPU
one (``test/test_metal_split_reduction_numerics.py``) runs them."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

F32 = torch.float32


@helion.kernel(backend="metal", static_shapes=True)
def split_gemv(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """x @ W.T for few rows of x, W[n, k] = s[n, k // 32] * w[n, k] with w
    int8: K in chunks of a register block, each chunk's products accumulated
    per row and column, the chunks summed after the K loop."""
    m, k = x.size()
    n = w.size(0)
    bk = hl.register_block_size(1, k)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n, bk], dtype=F32)
        for tile_k in hl.tile(k, block_size=bk):
            wk = w[tile_n, tile_k].to(F32) * s[tile_n, tile_k.index // 32].to(F32)
            xk = x[tile_m, tile_k].to(F32)
            acc = acc + xk.unsqueeze(1) * wk.unsqueeze(0)
        out[tile_m, tile_n] = acc.sum(2).to(x.dtype)
    return out


def split_gemv_args(
    m: int, n: int, k: int, device: str = "cpu"
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(m, k, generator=gen).to(torch.float16)
    w = torch.randint(-127, 128, (n, k), generator=gen, dtype=torch.int8)
    s = (torch.rand(n, k // 32, generator=gen) * 0.01).to(torch.float16)
    return tuple(t.to(device) for t in (x, w, s))


def split_gemv_reference(args: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """:func:`split_gemv` in float64 on the CPU."""
    x, w, s = (t.cpu().double() for t in args)
    return x @ (w * s.repeat_interleave(32, 1)).T


@helion.kernel(backend="metal", static_shapes=True)
def reduced_in_loop(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ W.T with each K chunk summed inside the loop, across its threads:
    nothing is reduced after it."""
    m, k = x.size()
    n = w.size(0)
    bk = hl.register_block_size(1, k)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k, block_size=bk):
            xk = x[tile_m, tile_k].to(F32)
            wk = w[tile_n, tile_k].to(F32)
            acc = acc + (xk.unsqueeze(1) * wk.unsqueeze(0)).sum(2)
        out[tile_m, tile_n] = acc
    return out
