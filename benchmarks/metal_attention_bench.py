"""Benchmark: Helion Metal fused attention vs PyTorch SDPA on MPS.

Compares single-dispatch batched fused attention:
  - Eager SDPA: PyTorch built-in (takes [B, H, M, D])
  - Helion fused: Helion-generated composed kernel (takes [B*H, M, D])

Usage (on a Mac with MPS):
    python benchmarks/metal_attention_bench.py
"""

from __future__ import annotations

import math
import time
from typing import Callable

import torch
import torch.nn.functional as F

import helion
import helion.language as hl
from helion.runtime import Config


@helion.kernel(
    backend="metal",
    static_shapes=True,
    config=Config(block_sizes=[1, 32, 32], num_warps=4),
)
def _fused_attention_kernel(
    q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Batched fused attention: single dispatch for all B*H heads.

    Q[B*H, M, D], Kt[B*H, D, N], V[B*H, N, D] -> out[B*H, M, D].
    Metal backend generates a 3D grid with tgid.z for head indexing.
    """
    num_heads = q.size(0)
    m_dim = q.size(1)
    head_dim = hl.specialize(q.size(2))
    n_dim = v.size(1)
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    for tile_b, tile_m in hl.tile([num_heads, m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        for tile_n in hl.tile(n_dim):
            qk = torch.bmm(q[tile_b, tile_m, :], kt[tile_b, :, tile_n])
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            p2 = p.to(v.dtype)
            acc = torch.baddbmm(acc, p2, v[tile_b, tile_n, :])
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out


def bench(
    fn: Callable[..., object],
    *args: torch.Tensor,
    warmup: int = 25,
    rep: int = 100,
) -> float:
    """Wall-clock benchmark with MPS synchronization. Returns median ms."""
    for _ in range(warmup):
        fn(*args)
    torch.mps.synchronize()

    times = []
    for _ in range(rep):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    times.sort()
    return times[len(times) // 2]


def main() -> None:
    device = torch.device("mps")
    batch, heads, head_dim = 2, 8, 64
    seq_lens = [64, 128, 256, 512]

    print(f"{'Config':<26} {'Eager SDPA':>12} {'Helion Fused':>13} {'Speedup':>9}")
    print("-" * 66)

    for seq_len in seq_lens:
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)

        # Pre-compute reshaped inputs (not timed — data layout, not compute)
        num_heads = batch * heads
        q3 = q.reshape(num_heads, seq_len, head_dim).contiguous()
        k3 = k.reshape(num_heads, seq_len, head_dim).contiguous()
        v3 = v.reshape(num_heads, seq_len, head_dim).contiguous()
        kt3 = k3.transpose(1, 2).contiguous()

        # Correctness check
        expected = F.scaled_dot_product_attention(q, k, v)
        result = _fused_attention_kernel(q3, kt3, v3).view_as(q)
        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

        # Benchmark: kernel dispatch only (no reshape overhead)
        t_eager = bench(F.scaled_dot_product_attention, q, k, v)
        t_helion = bench(_fused_attention_kernel, q3, kt3, v3)
        speedup = t_eager / t_helion

        config = f"B={batch} H={heads} S={seq_len} D={head_dim}"
        print(f"{config:<26} {t_eager:>10.3f}ms {t_helion:>11.3f}ms {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
