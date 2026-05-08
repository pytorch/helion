"""Element-wise vector addition."""

from __future__ import annotations

import math

import torch
import triton.testing as tt

import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel()
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


def main() -> None:
    shapes = [2**i for i in range(19, 29)]

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"{'N':>10s}  {'helion (us)':>12s}  {'torch (us)':>12s}  "
        f"{'speedup':>8s}  {'winner':>8s}"
    )
    print("-" * 60)

    speedups: list[float] = []
    helion_wins = 0
    best_speedup = 0.0
    best_n = 0
    for n in shapes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        vector_add(x, y)  # warmup
        ms_helion = tt.do_bench(
            lambda x=x, y=y: vector_add(x, y),
            warmup=50,
            rep=200,
            return_mode="median",
        )
        ms_torch = tt.do_bench(
            lambda x=x, y=y: x + y,
            warmup=50,
            rep=200,
            return_mode="median",
        )
        speedup = ms_torch / ms_helion if ms_helion > 0 else float("nan")
        speedups.append(speedup)
        winner = "helion" if speedup > 1.0 else "torch" if speedup < 1.0 else "tie"
        if speedup > 1.0:
            helion_wins += 1
        if speedup > best_speedup:
            best_speedup = speedup
            best_n = n
        print(
            f"{n:>10d}  {ms_helion * 1000:>12.2f}  "
            f"{ms_torch * 1000:>12.2f}  {speedup:>7.3f}x  {winner:>8s}"
        )

    geomean = math.exp(
        sum(math.log(s) for s in speedups if s > 0) / max(len(speedups), 1)
    )
    print(
        f"\nHelion faster on {helion_wins}/{len(shapes)} shapes; "
        f"geomean speedup {geomean:.3f}x; "
        f"best speedup {best_speedup:.2f}x at N={best_n}."
    )
    # Machine-parseable summary line consumed by test/test_pretuned_kernels.py.
    print(
        f"SUMMARY: helion_wins={helion_wins} total={len(shapes)} "
        f"geomean={geomean:.4f} best_speedup={best_speedup:.4f}"
    )


if __name__ == "__main__":
    main()
