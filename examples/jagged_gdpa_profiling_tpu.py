"""
Jagged GDPA TPU Profiling
==========================

Profiles the jagged GDPA forward kernel on TPU across various PFF (KV sequence
length) values to measure TFLOP/s scaling.

Same TFLOP calculation logic as the reference benchmark::

    total_flops = 4 * B * seq_length * pff * H * dim

Tensor shapes::

    q            : [L_q, H, D]
    k, v         : [L_kv, H, D]
    q_offsets    : [B + 1]
    kv_offsets   : [B + 1]
"""

from __future__ import annotations

import time

import torch

import helion
from helion._testing import DEVICE
import helion.language as hl

B = 256
H = 8
dim = 128
seq_length = 3731
dtype = torch.bfloat16


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    qk_scale: float,
) -> torch.Tensor:
    _H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = q_offsets.size(0) - 1
    out = torch.empty_like(q)

    for seq_idx in hl.grid(num_sequences):
        q_start = q_offsets[seq_idx]
        q_end = q_offsets[seq_idx + 1]
        k_start = kv_offsets[seq_idx]
        k_end = kv_offsets[seq_idx + 1]

        for tile_q in hl.tile(q_start, q_end):
            q_blk = q[tile_q, :, :].transpose(0, 1)
            acc = hl.zeros([_H, tile_q, D], dtype=torch.float32)

            for tile_kv in hl.tile(k_start, k_end):
                k_blk = k[tile_kv, :, :].transpose(0, 1)
                v_blk = v[tile_kv, :, :].transpose(0, 1)
                scores = torch.bmm(q_blk, k_blk.transpose(-2, -1)) * qk_scale
                acc = acc + torch.bmm(scores.to(v.dtype), v_blk)

            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)

    return out


def benchmark_pff(pff: int, warmup: int = 50, rep: int = 100) -> float:
    from helion.autotuner.benchmarking import synchronize_device

    device = torch.device(DEVICE)

    lengths_q = torch.full((B,), seq_length, dtype=torch.int32)
    lengths_kv = torch.full((B,), pff, dtype=torch.int32)
    q_offsets = torch.cat([
        torch.zeros(1, dtype=torch.int32),
        torch.cumsum(lengths_q, dim=0).to(torch.int32),
    ]).to(device)
    kv_offsets = torch.cat([
        torch.zeros(1, dtype=torch.int32),
        torch.cumsum(lengths_kv, dim=0).to(torch.int32),
    ]).to(device)
    L_q = int(q_offsets[-1].item())
    L_kv = int(kv_offsets[-1].item())

    q = torch.randn(L_q, H, dim, dtype=dtype, device=device)
    k = torch.randn(L_kv, H, dim, dtype=dtype, device=device)
    v = torch.randn(L_kv, H, dim, dtype=dtype, device=device)

    block_kv = min(256, pff)
    config = helion.Config(
        block_sizes=[2048, block_kv],
        pallas_loop_type="emit_pipeline",
        pallas_pre_broadcast=True,
    )
    kernel = helion.kernel(config=config, static_shapes=True)(attention)

    args = (q, k, v, q_offsets, kv_offsets, seq_length, pff, 1.0)

    for _ in range(warmup):
        synchronize_device(kernel(*args))

    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        synchronize_device(kernel(*args))
        times.append(time.perf_counter() - t0)

    median_s = sorted(times)[len(times) // 2]
    total_flops = 4 * B * seq_length * pff * H * dim
    tflops = total_flops / median_s / 1e12
    return tflops


def main() -> None:
    torch.manual_seed(0)

    print(f"Jagged GDPA TPU Profiling (B={B}, seq={seq_length}, H={H}, dim={dim}, {dtype})")
    print(f"{'pff':>6} {'TFLOP/s':>10}")
    print("-" * 20)

    pff = 128
    while pff <= 8192:
        tflops = benchmark_pff(pff)
        print(f"{pff:>6} {tflops:>10.2f}")
        pff *= 2


if __name__ == "__main__":
    main()
