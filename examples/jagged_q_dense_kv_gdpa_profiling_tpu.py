"""
Jagged Q Dense KV GDPA TPU Profiling
======================================

Profiles the jagged Q dense KV GDPA forward kernel on TPU with fixed KV length (128).
Q is jagged (variable per sequence), KV is loaded in full without tiling.

Same TFLOP calculation logic as the reference benchmark::

    total_flops = 4 * B * seq_length * pff * H * dim

Tensor shapes::

    q            : [L_q, H, D]
    k, v         : [B, N, H, D]
    q_offsets    : [B + 1]
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
pff = 128
dtype = torch.bfloat16


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    qk_scale: float,
) -> torch.Tensor:
    _H = hl.specialize(q.size(1))
    D = hl.specialize(q.size(2))
    num_sequences = q_offsets.size(0) - 1
    out = torch.empty_like(q)

    for seq_idx in hl.grid(num_sequences):
        q_start = q_offsets[seq_idx]
        q_end = q_offsets[seq_idx + 1]

        # [N, H, D] -> [H, N, D]
        k_blk = k[seq_idx, :, :, :].transpose(0, 1)
        v_blk = v[seq_idx, :, :, :].transpose(0, 1)

        for tile_q in hl.tile(q_start, q_end):
            q_blk = q[tile_q, :, :].transpose(0, 1)
            scores = torch.bmm(q_blk, k_blk.transpose(-2, -1)) * qk_scale
            acc = torch.bmm(scores.to(v.dtype), v_blk)
            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)

    return out


def benchmark(warmup: int = 50, rep: int = 100) -> float:
    from helion.autotuner.benchmarking import synchronize_device

    device = torch.device(DEVICE)

    lengths_q = torch.full((B,), seq_length, dtype=torch.int32)
    q_offsets = torch.cat([
        torch.zeros(1, dtype=torch.int32),
        torch.cumsum(lengths_q, dim=0).to(torch.int32),
    ]).to(device)
    L_q = int(q_offsets[-1].item())

    q = torch.randn(L_q, H, dim, dtype=dtype, device=device)
    k = torch.randn(B, pff, H, dim, dtype=dtype, device=device)
    v = torch.randn(B, pff, H, dim, dtype=dtype, device=device)

    config = helion.Config(
        block_sizes=[2048],
        pallas_loop_type="emit_pipeline",
        pallas_pre_broadcast=True,
    )
    kernel = helion.kernel(config=config, static_shapes=True)(attention)

    args = (q, k, v, q_offsets, 1.0)

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

    print(f"Jagged Q Dense KV GDPA TPU Profiling (B={B}, seq={seq_length}, pff={pff}, H={H}, dim={dim}, {dtype})")
    tflops = benchmark()
    print(f"TFLOP/s: {tflops:.2f}")


if __name__ == "__main__":
    main()
