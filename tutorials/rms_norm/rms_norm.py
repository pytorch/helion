"""Root mean square normalization (forward only)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton.testing as tt

import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel()
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        variance = torch.mean(acc * acc, dim=-1)
        inv_rms = torch.rsqrt(variance + eps)
        out[tile_m, :] = (acc * inv_rms[:, None] * weight[:].to(torch.float32)).to(
            x.dtype
        )
    return out


def _rms_norm_torch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    return F.rms_norm(x, [x.size(1)], weight, eps=eps)


def main() -> None:
    tritonbench_shapes = [
        (2048, 1024),
        (2048, 2048),
        (2048, 4096),
        (2048, 8192),
        (2048, 16384),
        (2048, 32768),
    ]
    tritonbench_npot_shapes = [
        (2048, 48),
        (2048, 96),
        (2048, 127),
        (2048, 768),
        (2048, 1023),
        (2048, 1536),
        (2048, 2047),
        (2048, 3072),
        (2048, 5120),
        (2048, 6144),
    ]
    realistic_shapes = [
        (4096, 3584),
        (4096, 4096),
        (4096, 5120),
        (4096, 7168),
        (4096, 8192),
        (4096, 12288),
        (8192, 4096),
        (8192, 8192),
        (16384, 4096),
        (16384, 8192),
        (145956, 384),
        (380668, 512),
        (589824, 256),
        (1179648, 256),
    ]
    shapes = list(
        dict.fromkeys(tritonbench_shapes + tritonbench_npot_shapes + realistic_shapes)
    )

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"{'M':>8s}  {'N':>6s}  {'helion (us)':>12s}  "
        f"{'torch (us)':>12s}  {'speedup':>8s}"
    )
    print("-" * 63)

    speedups: list[float] = []
    helion_wins = 0
    best_speedup = 0.0
    best_shape = (0, 0)
    for M, N in shapes:
        x = torch.randn([M, N], device="cuda", dtype=torch.bfloat16)
        weight = torch.randn([N], device="cuda", dtype=torch.bfloat16)
        rms_norm(x, weight)  # warmup
        ms_helion = tt.do_bench(
            lambda x=x, weight=weight: rms_norm(x, weight),
            warmup=25,
            rep=100,
            return_mode="median",
        )
        ms_torch = tt.do_bench(
            lambda x=x, weight=weight: _rms_norm_torch(x, weight),
            warmup=25,
            rep=100,
            return_mode="median",
        )
        speedup = ms_torch / ms_helion if ms_helion > 0 else float("nan")
        speedups.append(speedup)
        if speedup > 1.0:
            helion_wins += 1
        if speedup > best_speedup:
            best_speedup = speedup
            best_shape = (M, N)
        print(
            f"{M:>8d}  {N:>6d}  {ms_helion * 1000:>12.2f}  "
            f"{ms_torch * 1000:>12.2f}  {speedup:>7.2f}x"
        )

    geomean = math.exp(
        sum(math.log(s) for s in speedups if s > 0) / max(len(speedups), 1)
    )
    print(
        f"\nHelion faster on {helion_wins}/{len(shapes)} shapes; "
        f"geomean speedup {geomean:.3f}x; "
        f"best speedup {best_speedup:.2f}x at (M, N)={best_shape}."
    )
    print(
        f"SUMMARY: helion_wins={helion_wins} total={len(shapes)} "
        f"geomean={geomean:.4f} best_speedup={best_speedup:.4f}"
    )


if __name__ == "__main__":
    main()
