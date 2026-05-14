"""Layer normalization (forward only)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton.testing as tt

import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel()
def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        mean_val = torch.sum(acc, dim=-1) / n
        centered = acc - mean_val[:, None]
        var_val = torch.sum(centered * centered, dim=-1) / n
        rstd_val = torch.rsqrt(var_val + 1e-5)
        out[tile_m, :] = (
            centered * rstd_val[:, None] * weight[:].to(torch.float32)
            + bias[:].to(torch.float32)
        ).to(x.dtype)
    return out


def main() -> None:
    triton_tutorial_shapes = [(4096, 512 * i) for i in range(2, 32)]
    realistic_shapes = [
        (2048, 3584),
        (8192, 4096),
        (8192, 5120),
        (8192, 7168),
        (2048, 8192),
        (4096, 16384),
        (1024, 36864),
        (1152, 36864),
    ]
    shapes = list(dict.fromkeys(triton_tutorial_shapes + realistic_shapes))

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"{'M':>5s}  {'N':>6s}  {'helion (us)':>12s}  "
        f"{'torch (us)':>12s}  {'speedup':>8s}"
    )
    print("-" * 60)

    speedups: list[float] = []
    helion_wins = 0
    best_speedup = 0.0
    best_shape = (0, 0)
    for M, N in shapes:
        x = torch.randn([M, N], device="cuda", dtype=torch.float16)
        w = torch.randn([N], device="cuda", dtype=torch.float16)
        b = torch.randn([N], device="cuda", dtype=torch.float16)
        layer_norm(x, w, b)  # warmup
        ms_helion = tt.do_bench(
            lambda x=x, w=w, b=b: layer_norm(x, w, b),
            warmup=50,
            rep=200,
            return_mode="median",
        )
        ms_torch = tt.do_bench(
            lambda x=x, N=N, w=w, b=b: F.layer_norm(x, [N], w, b, eps=1e-5),
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
        print(
            f"{M:>5d}  {N:>6d}  {ms_helion * 1000:>12.2f}  "
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
