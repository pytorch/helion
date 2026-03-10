"""Benchmark: Helion Metal matmul vs PyTorch eager vs torch.compile (inductor).

Usage (on a Mac with MPS):
    python benchmarks/metal_matmul_bench.py
"""

from __future__ import annotations

import time
from typing import Callable

import torch

import helion
import helion.language as hl


@helion.kernel(backend="metal", autotune_effort="full")
def helion_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _k2, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
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

    sizes = [128, 256, 512, 1024]

    def eager_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mm(x, y)

    compiled_fn = torch.compile(eager_fn, backend="inductor")

    print(
        f"{'Size':<10} {'Eager (ms)':>12} {'Inductor (ms)':>14} {'Helion (ms)':>13} {'vs Eager':>10} {'vs Inductor':>12}"
    )
    print("-" * 75)

    for sz in sizes:
        x = torch.randn(sz, sz, device=device, dtype=torch.float32)
        y = torch.randn(sz, sz, device=device, dtype=torch.float32)

        # Correctness check
        expected = torch.mm(x, y)
        result = helion_matmul(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

        t_eager = bench(eager_fn, x, y)
        t_inductor = bench(compiled_fn, x, y)
        t_helion = bench(lambda x, y: helion_matmul(x, y), x, y)

        speedup_eager = t_eager / t_helion
        speedup_inductor = t_inductor / t_helion

        print(
            f"{sz:>4}x{sz:<4} "
            f"{t_eager:>10.3f}   "
            f"{t_inductor:>12.3f}   "
            f"{t_helion:>11.3f}   "
            f"{speedup_eager:>8.2f}x  "
            f"{speedup_inductor:>10.2f}x"
        )


if __name__ == "__main__":
    main()
