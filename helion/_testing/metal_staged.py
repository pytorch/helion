"""Kernels the Metal staged-matmul seed tests share: the CPU-only module
(``test/test_metal_staged_matmul_heuristic.py``) reads their seeds and the
GPU one (``test/test_metal_mpp_matmul.py``) runs them."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

F32 = torch.float32


@helion.kernel(backend="metal", static_shapes=True)
def dequant_mm(x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """x @ W.T, W 4-bit (nibble k % 8 of word k // 8, scaled per 32): the
    weight tile is computed per K tile and staged."""
    m, k = x.size()
    n = wq.size(0)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            words = wq[tile_n, tile_k.index // 8]
            nib = (words >> ((tile_k.index % 8) * 4)[None, :]) & 0xF
            w = (nib.to(F32) - 8.0) * s[tile_n, tile_k.index // 32]
            acc = torch.addmm(acc, x[tile_m, tile_k], w.T.to(x.dtype))
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


def dequant_args(
    m: int, n: int, k: int, device: str = "cpu"
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(m, k, generator=gen).to(torch.bfloat16)
    wq = torch.randint(
        -(2**31), 2**31 - 1, (n, k // 8), generator=gen, dtype=torch.int32
    )
    s = torch.rand(n, k // 32, generator=gen) * 0.1
    return tuple(t.to(device) for t in (x, wq, s))
