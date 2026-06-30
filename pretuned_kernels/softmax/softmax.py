"""Row-wise softmax."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton.testing as tt

import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel()
def softmax(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        out[tile_m, :] = torch.nn.functional.softmax(x[tile_m, :], dim=1)
    return out


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return False


def main(verbose: bool = True) -> dict:
    def _p(*args: object) -> None:
        if verbose:
            print(*args)

    triton_tutorial_shapes = [(4096, 128 * i) for i in range(2, 100)]
    realistic_shapes = [
        (4096, 16384),
        (2048, 32768),
    ]
    shapes = list(dict.fromkeys(triton_tutorial_shapes + realistic_shapes))

    _p(f"GPU: {torch.cuda.get_device_name()}")
    _p(
        f"{'M':>5s}  {'N':>6s}  {'helion (us)':>12s}  "
        f"{'torch (us)':>12s}  {'speedup':>8s}"
    )
    _p("-" * 60)

    speedups: list[float] = []
    helion_wins = 0
    best_speedup = 0.0
    best_shape = (0, 0)
    for M, N in shapes:
        x = torch.randn([M, N], device="cuda", dtype=torch.float16)
        softmax(x)  # warmup
        ms_helion = tt.do_bench(
            lambda x=x: softmax(x),
            warmup=50,
            rep=200,
            return_mode="median",
        )
        ms_torch = tt.do_bench(
            lambda x=x: F.softmax(x, dim=1),
            warmup=50,
            rep=200,
            return_mode="median",
        )
        speedup = ms_torch / ms_helion if ms_helion > 0 else float("nan")
        speedups.append(speedup)
        if speedup > 1.0:
            helion_wins += 1
        if speedup > best_speedup:
            best_speedup = speedup
            best_shape = (M, N)
        _p(
            f"{M:>5d}  {N:>6d}  {ms_helion * 1000:>12.2f}  "
            f"{ms_torch * 1000:>12.2f}  {speedup:>7.2f}x"
        )

    geomean = math.exp(
        sum(math.log(s) for s in speedups if s > 0) / max(len(speedups), 1)
    )
    _p(
        f"\nHelion faster on {helion_wins}/{len(shapes)} shapes; "
        f"geomean speedup {geomean:.3f}x; "
        f"best speedup {best_speedup:.2f}x at (M, N)={best_shape}."
    )
    return {
        "helion_wins": helion_wins,
        "total": len(shapes),
        "geomean": round(geomean, 4),
        "best_speedup": round(best_speedup, 4),
    }


if __name__ == "__main__":
    main()
