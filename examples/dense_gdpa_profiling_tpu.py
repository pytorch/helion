"""
Dense GDPA TPU Profiling
=========================

Profiles the dense GDPA forward kernel on TPU with fixed Q length (3731)
and fixed KV length (128). All sequences have the same lengths.
KV is loaded in full without tiling.

Same TFLOP calculation logic as the reference benchmark::

    total_flops = 4 * B * seq_length * pff * H * dim

Tensor shapes::

    q            : [B, M, H, D]
    k, v         : [B, N, H, D]
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
    qk_scale: float,
) -> torch.Tensor:
    _H = hl.specialize(q.size(2))
    D = hl.specialize(q.size(3))
    num_sequences = q.size(0)
    M = q.size(1)
    out = torch.empty_like(q)

    for tile_seq, tile_q in hl.tile([num_sequences, M]):
        # [tile_seq, tile_q, H, D] -> [tile_seq*H, tile_q, D]
        q_blk = q[tile_seq, tile_q, :, :].permute(0, 2, 1, 3).flatten(0, 1)
        # [tile_seq, N, H, D] -> [tile_seq*H, N, D]
        k_blk = k[tile_seq, :, :, :].permute(0, 2, 1, 3).flatten(0, 1)
        v_blk = v[tile_seq, :, :, :].permute(0, 2, 1, 3).flatten(0, 1)

        scores = torch.bmm(q_blk, k_blk.transpose(-2, -1)) * qk_scale
        acc = torch.bmm(scores.to(v.dtype), v_blk)
        out[tile_seq, tile_q, :, :] = acc.unflatten(0, [tile_seq, _H]).permute(0, 2, 1, 3).to(out.dtype)

    return out


def benchmark(warmup: int = 50, rep: int = 100) -> float:
    from helion.autotuner.benchmarking import synchronize_device

    device = torch.device(DEVICE)

    q = torch.randn(B, seq_length, H, dim, dtype=dtype, device=device)
    k = torch.randn(B, pff, H, dim, dtype=dtype, device=device)
    v = torch.randn(B, pff, H, dim, dtype=dtype, device=device)

    config = helion.Config(
        block_sizes=[1, 2048],
        pallas_loop_type="emit_pipeline",
        pallas_pre_broadcast=True,
    )
    kernel = helion.kernel(config=config, static_shapes=True)(attention)

    args = (q, k, v, 1.0)

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

    print(f"Dense GDPA TPU Profiling (B={B}, seq={seq_length}, pff={pff}, H={H}, dim={dim}, {dtype})")
    tflops = benchmark()
    print(f"TFLOP/s: {tflops:.2f}")


if __name__ == "__main__":
    main()
