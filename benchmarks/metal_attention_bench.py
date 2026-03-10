"""Benchmark: Helion Metal fused attention vs PyTorch SDPA on MPS.

Compares:
  1. Eager SDPA (PyTorch built-in, single dispatch)
  2. Helion fused (per-head Helion-generated composed kernel, B*H dispatches)

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


@helion.kernel(backend="metal", static_shapes=True)
def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Fused attention: softmax(Q @ K^T / sqrt(d)) @ V in a single kernel.

    Q[M, D], K[N, D], V[N, D] -> out[M, D].
    Metal backend emits: multi-SG matmul -> threadgroup softmax -> multi-SG matmul.
    """
    m_dim = q.size(0)
    n_dim = k.size(0)
    head_dim = hl.specialize(q.size(1))
    kt = k.transpose(0, 1)
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    for tile_m in hl.tile(m_dim):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        for tile_n in hl.tile(n_dim):
            qk = torch.mm(q[tile_m, :], kt[:, tile_n])
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            p2 = p.to(v.dtype)
            acc = torch.addmm(acc, p2, v[tile_n, :])
            m_i = m_ij
        acc = acc / l_i[:, None]
        out[tile_m, :] = acc.to(out.dtype)
    return out


def helion_attention_batched(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Batched Helion attention. Q/K/V: [B, H, M, D] -> [B, H, M, D].

    Dispatches one Helion-generated composed kernel per head.
    """
    batch, heads, seq_len, head_dim = q.shape
    out = torch.empty_like(q)
    for b in range(batch):
        for h in range(heads):
            out[b, h] = fused_attention(q[b, h], k[b, h], v[b, h])
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

        # Correctness check (per-head against single-head SDPA)
        for b in range(min(batch, 1)):
            for h in range(min(heads, 1)):
                result = fused_attention(q[b, h], k[b, h], v[b, h])
                expected = (
                    F.scaled_dot_product_attention(
                        q[b, h].unsqueeze(0).unsqueeze(0),
                        k[b, h].unsqueeze(0).unsqueeze(0),
                        v[b, h].unsqueeze(0).unsqueeze(0),
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
                torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

        t_eager = bench(F.scaled_dot_product_attention, q, k, v)
        t_helion = bench(helion_attention_batched, q, k, v)
        speedup = t_eager / t_helion

        config = f"B={batch} H={heads} S={seq_len} D={head_dim}"
        print(f"{config:<26} {t_eager:>10.3f}ms {t_helion:>11.3f}ms {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
