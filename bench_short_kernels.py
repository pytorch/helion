"""
Focused investigation: can we measure real differences between the
pre-fix and post-fix vLLM configs for super-short kernels?

Uses interleaved_bench (which cancels drift) to get the ground truth,
then checks whether do_bench at various rep levels can see it too.
"""

from __future__ import annotations

import functools
import math
import statistics

import torch

import helion
from helion.autotuner.benchmarking import do_bench
from helion.autotuner.benchmarking import interleaved_bench
import helion.language as hl


@helion.kernel(static_shapes=False)
def silu_mul_fp8(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    original_shape = input.shape
    two_d = hl.specialize(original_shape[-1])
    d = two_d // 2
    output_shape = original_shape[:-1] + (d,)
    input_2d = input.view(-1, original_shape[-1])
    m = input_2d.shape[0]
    out = torch.empty((m, d), device=input.device, dtype=torch.float8_e4m3fn)
    input_part_a = input_2d[:, :d]
    input_part_b = input_2d[:, d:]
    assert scale.numel() == 1
    for tile_m, tile_n in hl.tile([m, d]):
        a_vals = input_part_a[tile_m, tile_n]
        silu_result = torch.nn.functional.silu(a_vals)
        b_vals = input_part_b[tile_m, tile_n]
        result = silu_result * b_vals
        result_f32 = result.to(torch.float32)
        scale_val = hl.load(scale, [0])
        inv_scale = 1.0 / scale_val
        result_scaled = result_f32 * inv_scale
        out[tile_m, tile_n] = result_scaled.to(out.dtype)
    return out.view(output_shape)


def make_fn(num_tokens, isize, config):
    """Compile a config and return a zero-arg callable."""
    inp = torch.randn(num_tokens, 2 * isize, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    silu_mul_fp8.reset()
    bound = silu_mul_fp8.bind((inp, scale))
    compiled = bound.compile_config(config)
    compiled(inp, scale)  # warmup/compile
    return functools.partial(compiled, inp, scale)


# Actual pre-fix vs post-fix vLLM configs (from the PR diff)
CASES = [
    # (label, tokens, isize, old_config, new_config)
    (
        "1x2048",
        1,
        2048,
        {"block_sizes": [1, 16], "num_warps": 16, "num_stages": 2},
        {"block_sizes": [1, 256], "num_warps": 2, "num_stages": 1},
    ),
    (
        "1x4096",
        1,
        4096,
        {"block_sizes": [1, 32], "num_warps": 4, "num_stages": 2},
        {"block_sizes": [1, 256], "num_warps": 2, "num_stages": 1},
    ),
    (
        "2x4096",
        2,
        4096,
        {"block_sizes": [2, 32], "num_warps": 8, "num_stages": 2},
        {"block_sizes": [2, 128], "num_warps": 2, "num_stages": 1},
    ),
    (
        "4x2048",
        4,
        2048,
        {"block_sizes": [1, 256], "num_warps": 32, "num_stages": 2},
        {"block_sizes": [4, 64], "num_warps": 2, "num_stages": 1},
    ),
    (
        "8x4096",
        8,
        4096,
        {"block_sizes": [2, 32], "num_warps": 32, "num_stages": 1},
        {"block_sizes": [8, 32], "num_warps": 2, "num_stages": 1},
    ),
    (
        "16x2048",
        16,
        2048,
        {"block_sizes": [16, 64], "num_warps": 4, "num_stages": 2},
        {"block_sizes": [8, 512], "num_warps": 16, "num_stages": 2},
    ),
    (
        "32x4096",
        32,
        4096,
        {"block_sizes": [32, 16], "num_warps": 1, "num_stages": 2},
        {"block_sizes": [4, 4096], "num_warps": 32, "num_stages": 5},
    ),
    (
        "64x4096",
        64,
        4096,
        {"block_sizes": [2, 16], "num_warps": 1, "num_stages": 1},
        {"block_sizes": [64, 32], "num_warps": 8, "num_stages": 1},
    ),
    (
        "128x4096",
        128,
        4096,
        {"block_sizes": [128, 32], "num_warps": 32, "num_stages": 2},
        {"block_sizes": [128, 32], "num_warps": 8, "num_stages": 1},
    ),
]


def main():
    N = 30  # trials per method

    print("=" * 100)
    print("Ground truth: interleaved_bench on actual vLLM pre-fix vs post-fix configs")
    print("=" * 100)

    # Phase 1: Establish ground truth with interleaved_bench
    print()
    print(
        f"{'case':<12} {'method':<30} {'old_ms':>10} {'new_ms':>10} {'Δ_ms':>10} {'Δ%':>8} {'d':>6} {'old_wins':>10}"
    )
    print("-" * 100)

    for label, tokens, isize, old_cfg, new_cfg in CASES:
        fn_old = make_fn(tokens, isize, old_cfg)
        fn_new = make_fn(tokens, isize, new_cfg)

        # Ground truth: many interleaved trials
        old_times, new_times = [], []
        for _ in range(N):
            t = interleaved_bench([fn_old, fn_new], repeat=1000)
            old_times.append(t[0])
            new_times.append(t[1])

        med_old, med_new = statistics.median(old_times), statistics.median(new_times)
        std_old, std_new = statistics.stdev(old_times), statistics.stdev(new_times)
        pooled = math.sqrt((std_old**2 + std_new**2) / 2) or 1e-9
        d = abs(med_old - med_new) / pooled
        pct = (med_old - med_new) / min(med_old, med_new) * 100
        wins = sum(1 for a, b in zip(old_times, new_times) if a > b)

        print(
            f"{label:<12} {'interleaved(1000)':<30} {med_old:>10.5f} {med_new:>10.5f} "
            f"{med_old - med_new:>+10.5f} {pct:>+7.1f}% {d:>6.1f} {wins:>7}/{N}"
        )

        # Current autotuner: do_bench(rep=50, median)
        old_bench, new_bench = [], []
        for _ in range(N):
            old_bench.append(do_bench(fn_old, warmup=1, rep=50, return_mode="median"))
            new_bench.append(do_bench(fn_new, warmup=1, rep=50, return_mode="median"))
        med_old2, med_new2 = statistics.median(old_bench), statistics.median(new_bench)
        std_old2, std_new2 = statistics.stdev(old_bench), statistics.stdev(new_bench)
        pooled2 = math.sqrt((std_old2**2 + std_new2**2) / 2) or 1e-9
        d2 = abs(med_old2 - med_new2) / pooled2
        pct2 = (med_old2 - med_new2) / min(med_old2, med_new2) * 100
        wins2 = sum(1 for a, b in zip(old_bench, new_bench) if a > b)
        print(
            f"{'':12} {'do_bench(rep=50,median)':<30} {med_old2:>10.5f} {med_new2:>10.5f} "
            f"{med_old2 - med_new2:>+10.5f} {pct2:>+7.1f}% {d2:>6.1f} {wins2:>7}/{N}"
        )

        # Proposed: do_bench with adaptive rep for fast kernels
        est_ms = min(med_old, med_new)
        if est_ms < 0.1:
            adaptive_rep = max(50, min(200, int(5 / est_ms)))
        else:
            adaptive_rep = 50
        old_arep, new_arep = [], []
        for _ in range(N):
            old_arep.append(
                do_bench(fn_old, warmup=1, rep=adaptive_rep, return_mode="median")
            )
            new_arep.append(
                do_bench(fn_new, warmup=1, rep=adaptive_rep, return_mode="median")
            )
        med_old3, med_new3 = statistics.median(old_arep), statistics.median(new_arep)
        std_old3, std_new3 = statistics.stdev(old_arep), statistics.stdev(new_arep)
        pooled3 = math.sqrt((std_old3**2 + std_new3**2) / 2) or 1e-9
        d3 = abs(med_old3 - med_new3) / pooled3
        pct3 = (med_old3 - med_new3) / min(med_old3, med_new3) * 100
        wins3 = sum(1 for a, b in zip(old_arep, new_arep) if a > b)
        print(
            f"{'':12} {f'do_bench(rep={adaptive_rep},median)':<30} {med_old3:>10.5f} {med_new3:>10.5f} "
            f"{med_old3 - med_new3:>+10.5f} {pct3:>+7.1f}% {d3:>6.1f} {wins3:>7}/{N}"
        )

        print()


if __name__ == "__main__":
    main()
