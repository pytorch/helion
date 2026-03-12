"""Benchmark: Helion Metal RMSNorm vs PyTorch eager vs torch.compile (inductor).

Usage (on a Mac with MPS):
    python benchmarks/metal_rmsnorm_bench.py
"""

from __future__ import annotations

import time
from typing import Callable

import torch

import helion
import helion.language as hl


@helion.kernel(backend="metal", autotune_effort="full")
def helion_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        row = x[tile_n, :]
        sq = row * row
        mean_sq = torch.sum(sq, dim=1, keepdim=True) / m
        rms = torch.rsqrt(mean_sq + eps)
        out[tile_n, :] = row * rms * weight[None, :]
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

    sizes = [
        (128, 256),
        (256, 1024),
        (1024, 1024),
        (1024, 4096),
        (4096, 2560),
    ]

    def eager_fn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(x, [x.size(-1)], w, 1e-6)

    compiled_fn = torch.compile(eager_fn, backend="inductor")

    print(
        f"{'(M, N)':<16} {'Eager (ms)':>12} {'Inductor (ms)':>14} {'Helion (ms)':>13} {'vs Eager':>10} {'vs Inductor':>12}"
    )
    print("-" * 80)

    for m, n in sizes:
        x = torch.randn(m, n, device=device, dtype=torch.float32)
        w = torch.randn(n, device=device, dtype=torch.float32)

        # Correctness check
        expected = torch.nn.functional.rms_norm(x, [n], w, 1e-6)
        result = helion_rms_norm(x, w)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

        t_eager = bench(eager_fn, x, w)
        t_inductor = bench(compiled_fn, x, w)
        t_helion = bench(lambda x, w: helion_rms_norm(x, w), x, w)

        speedup_eager = t_eager / t_helion
        speedup_inductor = t_inductor / t_helion

        print(
            f"({m}, {n}){'':<{14 - len(f'({m}, {n})')}} "
            f"{t_eager:>10.3f}   "
            f"{t_inductor:>12.3f}   "
            f"{t_helion:>11.3f}   "
            f"{speedup_eager:>8.2f}x  "
            f"{speedup_inductor:>10.2f}x"
        )


if __name__ == "__main__":
    main()
