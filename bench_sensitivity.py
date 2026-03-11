"""
Microbenchmark to measure the autotuner's timing sensitivity floor.

We want to answer:
1. What is the minimum timing difference the autotuner can reliably detect?
2. How does this change with kernel runtime (fast vs slow kernels)?
3. Can we sharpen it by increasing rep count, using interleaved bench, etc.?

We test with known-different configs that produce measurably different
performance at larger scales, then measure the autotuner's ability to
distinguish them at the microsecond scale.
"""

from __future__ import annotations

import math
import statistics

import torch

import helion
from helion.autotuner.benchmarking import do_bench
from helion.autotuner.benchmarking import interleaved_bench
import helion.language as hl

# ============================================================================
# Test kernel: simple elementwise with tunable work-per-block
# ============================================================================


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

    assert scale.numel() == 1, "Scale must be a scalar Tensor"

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


def make_callable(num_tokens, intermediate_size, config):
    """Create a compiled callable for a given config.

    Uses compile_config to get independent compiled functions, avoiding
    the BoundKernel cache which would cause different configs to alias.
    """
    import functools

    inp = torch.randn(
        num_tokens, 2 * intermediate_size, device="cuda", dtype=torch.bfloat16
    )
    scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    # Reset to avoid cached BoundKernel from a previous call with same shape
    silu_mul_fp8.reset()
    bound = silu_mul_fp8.bind((inp, scale))
    compiled_fn = bound.compile_config(config)
    # warmup
    compiled_fn(inp, scale)
    return functools.partial(compiled_fn, inp, scale)


# ============================================================================
# Sensitivity measurement
# ============================================================================


def measure_timing_distribution(fn, n_trials=50, rep_ms=50):
    """Collect n_trials independent timing measurements using do_bench."""
    measurements = []
    for _ in range(n_trials):
        t = do_bench(fn, warmup=5, rep=rep_ms, return_mode="median")
        assert isinstance(t, float)
        measurements.append(t)
    return measurements


def measure_timing_distribution_custom(fn, n_trials=50, n_repeat=1000):
    """Collect timing with explicit repeat count (bypassing do_bench's auto-repeat)."""
    from triton import runtime

    di = runtime.driver.active.get_device_interface()
    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # warmup
    for _ in range(100):
        fn()
    di.synchronize()

    measurements = []
    for _ in range(n_trials):
        times = []
        for _ in range(n_repeat):
            runtime.driver.active.clear_cache(cache)
            start = di.Event(enable_timing=True)
            end = di.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            di.synchronize()
            times.append(start.elapsed_time(end))
        measurements.append(statistics.median(times))
    return measurements


def analyze_discrimination(times_a, times_b, label_a="A", label_b="B"):
    """Analyze whether two timing distributions are distinguishable."""
    med_a = statistics.median(times_a)
    med_b = statistics.median(times_b)
    std_a = statistics.stdev(times_a) if len(times_a) > 1 else 0
    std_b = statistics.stdev(times_b) if len(times_b) > 1 else 0

    # Effect size (Cohen's d)
    pooled_std = math.sqrt((std_a**2 + std_b**2) / 2) if (std_a + std_b) > 0 else 1e-9
    cohens_d = abs(med_a - med_b) / pooled_std

    # Simple discrimination: what fraction of A samples < median(B)?
    if med_a < med_b:
        correct = sum(
            1 for a, b in zip(sorted(times_a), sorted(times_b, reverse=True)) if a < b
        )
    else:
        correct = sum(
            1 for a, b in zip(sorted(times_a, reverse=True), sorted(times_b)) if a > b
        )
    n = min(len(times_a), len(times_b))
    accuracy = correct / n if n > 0 else 0

    # Overlap: fraction of times_a that fall within range of times_b
    min_b, max_b = min(times_b), max(times_b)
    overlap = sum(1 for t in times_a if min_b <= t <= max_b) / len(times_a)

    return {
        "med_a": med_a,
        "med_b": med_b,
        "std_a": std_a,
        "std_b": std_b,
        "diff_ms": abs(med_a - med_b),
        "diff_pct": abs(med_a - med_b) / min(med_a, med_b) * 100
        if min(med_a, med_b) > 0
        else 0,
        "cohens_d": cohens_d,
        "pairwise_accuracy": accuracy,
        "overlap": overlap,
    }


def measure_interleaved_discrimination(fn_a, fn_b, n_trials=50, repeat=1000):
    """Use interleaved_bench to compare two functions, repeat n_trials times."""
    results_a = []
    results_b = []
    for _ in range(n_trials):
        timings = interleaved_bench([fn_a, fn_b], repeat=repeat)
        results_a.append(timings[0])
        results_b.append(timings[1])
    return results_a, results_b


def main():
    print("=" * 80)
    print("Autotuner Timing Sensitivity Floor Measurement")
    print("=" * 80)
    print()

    # Test cases: pairs of configs that differ by a known amount
    test_cases = [
        # (label, tokens, isize, config_a, config_b)
        (
            "DRAMATIC: [1,1] vs [128,32] @ 128x4096",
            128,
            4096,
            {"block_sizes": [1, 1], "num_warps": 4, "num_stages": 1},
            {"block_sizes": [128, 32], "num_warps": 8, "num_stages": 1},
        ),
        (
            "MODERATE: [4,8] vs [64,32] @ 64x4096",
            64,
            4096,
            {"block_sizes": [4, 8], "num_warps": 2, "num_stages": 1},
            {"block_sizes": [64, 32], "num_warps": 8, "num_stages": 1},
        ),
        (
            "SUBTLE: [1,32] vs [1,256] @ 1x2048",
            1,
            2048,
            {"block_sizes": [1, 32], "num_warps": 4, "num_stages": 2},
            {"block_sizes": [1, 256], "num_warps": 2, "num_stages": 1},
        ),
        (
            "NOISE: same config, different warps @ 8x4096",
            8,
            4096,
            {"block_sizes": [8, 32], "num_warps": 2, "num_stages": 1},
            {"block_sizes": [8, 32], "num_warps": 4, "num_stages": 1},
        ),
        (
            "TINY: [2,16] vs [2,128] @ 2x4096",
            2,
            4096,
            {"block_sizes": [2, 16], "num_warps": 2, "num_stages": 1},
            {"block_sizes": [2, 128], "num_warps": 2, "num_stages": 1},
        ),
    ]

    # ====================================================================
    # Phase 1: Measure with current autotuner settings (rep=50ms)
    # ====================================================================
    print("Phase 1: Current autotuner settings (do_bench, rep=50ms, median)")
    print("-" * 80)
    n_trials = 30

    for label, tokens, isize, cfg_a, cfg_b in test_cases:
        print(f"\n  {label}")
        fn_a = make_callable(tokens, isize, cfg_a)
        fn_b = make_callable(tokens, isize, cfg_b)

        times_a = measure_timing_distribution(fn_a, n_trials=n_trials, rep_ms=50)
        times_b = measure_timing_distribution(fn_b, n_trials=n_trials, rep_ms=50)

        stats = analyze_discrimination(times_a, times_b, "A", "B")
        print(
            f"    A: {stats['med_a']:.4f}ms ± {stats['std_a']:.4f}  B: {stats['med_b']:.4f}ms ± {stats['std_b']:.4f}"
        )
        print(
            f"    diff={stats['diff_ms']:.4f}ms ({stats['diff_pct']:.1f}%)  Cohen's d={stats['cohens_d']:.2f}  "
            f"pairwise={stats['pairwise_accuracy']:.0%}  overlap={stats['overlap']:.0%}"
        )

    # ====================================================================
    # Phase 2: Measure with higher rep count
    # ====================================================================
    print()
    print()
    print("Phase 2: Higher rep count (do_bench, rep=500ms, median)")
    print("-" * 80)

    for label, tokens, isize, cfg_a, cfg_b in test_cases:
        print(f"\n  {label}")
        fn_a = make_callable(tokens, isize, cfg_a)
        fn_b = make_callable(tokens, isize, cfg_b)

        times_a = measure_timing_distribution(fn_a, n_trials=n_trials, rep_ms=500)
        times_b = measure_timing_distribution(fn_b, n_trials=n_trials, rep_ms=500)

        stats = analyze_discrimination(times_a, times_b, "A", "B")
        print(
            f"    A: {stats['med_a']:.4f}ms ± {stats['std_a']:.4f}  B: {stats['med_b']:.4f}ms ± {stats['std_b']:.4f}"
        )
        print(
            f"    diff={stats['diff_ms']:.4f}ms ({stats['diff_pct']:.1f}%)  Cohen's d={stats['cohens_d']:.2f}  "
            f"pairwise={stats['pairwise_accuracy']:.0%}  overlap={stats['overlap']:.0%}"
        )

    # ====================================================================
    # Phase 3: Interleaved bench (current repeat)
    # ====================================================================
    print()
    print()
    print("Phase 3: Interleaved bench (current autotuner repeat formula)")
    print("-" * 80)

    for label, tokens, isize, cfg_a, cfg_b in test_cases:
        print(f"\n  {label}")
        fn_a = make_callable(tokens, isize, cfg_a)
        fn_b = make_callable(tokens, isize, cfg_b)

        # Use the autotuner's repeat formula: 200 / best_perf_ms, clamped [3, 1000]
        est_ms = do_bench(fn_a, warmup=5, rep=50, return_mode="median")
        assert isinstance(est_ms, float)
        repeat = min(1000, max(3, int(200 / est_ms)))
        print(f"    est_ms={est_ms:.4f}, repeat={repeat}")

        times_a, times_b = measure_interleaved_discrimination(
            fn_a, fn_b, n_trials=n_trials, repeat=repeat
        )

        stats = analyze_discrimination(times_a, times_b, "A", "B")
        print(
            f"    A: {stats['med_a']:.4f}ms ± {stats['std_a']:.4f}  B: {stats['med_b']:.4f}ms ± {stats['std_b']:.4f}"
        )
        print(
            f"    diff={stats['diff_ms']:.4f}ms ({stats['diff_pct']:.1f}%)  Cohen's d={stats['cohens_d']:.2f}  "
            f"pairwise={stats['pairwise_accuracy']:.0%}  overlap={stats['overlap']:.0%}"
        )

    # ====================================================================
    # Phase 4: Interleaved bench with adaptive higher repeat for fast kernels
    # ====================================================================
    print()
    print()
    print("Phase 4: Interleaved bench with ADAPTIVE repeat (sharper for fast kernels)")
    print("-" * 80)

    for label, tokens, isize, cfg_a, cfg_b in test_cases:
        print(f"\n  {label}")
        fn_a = make_callable(tokens, isize, cfg_a)
        fn_b = make_callable(tokens, isize, cfg_b)

        # Adaptive: target a fixed wall-clock budget (e.g., 2s) but ensure
        # minimum repeat scales inversely with kernel time
        est_ms = do_bench(fn_a, warmup=5, rep=50, return_mode="median")
        assert isinstance(est_ms, float)

        # Current formula: repeat = 200 / est_ms → for 0.03ms kernel: 6666 (clamped to 1000)
        # Proposed: target_wall_ms / est_ms, with higher target for fast kernels
        # If kernel < 0.1ms, use 2000ms wall budget; else 200ms
        if est_ms < 0.1:
            target_wall_ms = 2000
        else:
            target_wall_ms = 200
        repeat = min(5000, max(100, int(target_wall_ms / est_ms)))
        print(f"    est_ms={est_ms:.4f}, adaptive_repeat={repeat}")

        times_a, times_b = measure_interleaved_discrimination(
            fn_a, fn_b, n_trials=n_trials, repeat=repeat
        )

        stats = analyze_discrimination(times_a, times_b, "A", "B")
        print(
            f"    A: {stats['med_a']:.4f}ms ± {stats['std_a']:.4f}  B: {stats['med_b']:.4f}ms ± {stats['std_b']:.4f}"
        )
        print(
            f"    diff={stats['diff_ms']:.4f}ms ({stats['diff_pct']:.1f}%)  Cohen's d={stats['cohens_d']:.2f}  "
            f"pairwise={stats['pairwise_accuracy']:.0%}  overlap={stats['overlap']:.0%}"
        )

    # ====================================================================
    # Phase 5: Noise floor measurement — identical function
    # ====================================================================
    print()
    print()
    print("Phase 5: Noise floor — benchmarking identical function against itself")
    print("-" * 80)

    for tokens, isize, cfg in [
        (1, 2048, {"block_sizes": [1, 256], "num_warps": 2, "num_stages": 1}),
        (64, 4096, {"block_sizes": [64, 32], "num_warps": 8, "num_stages": 1}),
        (512, 4096, {"block_sizes": [64, 64], "num_warps": 8, "num_stages": 1}),
    ]:
        fn = make_callable(tokens, isize, cfg)
        est_ms = do_bench(fn, warmup=5, rep=50, return_mode="median")
        assert isinstance(est_ms, float)

        for rep_budget in [50, 200, 500, 2000]:
            times = measure_timing_distribution(fn, n_trials=50, rep_ms=rep_budget)
            med = statistics.median(times)
            std = statistics.stdev(times)
            cv = std / med * 100 if med > 0 else 0
            iqr = statistics.quantiles(times, n=4)
            iqr_range = iqr[2] - iqr[0]
            print(
                f"  tokens={tokens:>4} isize={isize} | rep={rep_budget:>4}ms | "
                f"median={med:.4f}ms  std={std:.4f}ms  CV={cv:.1f}%  IQR={iqr_range:.4f}ms  "
                f"min_detectable~{2 * std:.4f}ms ({2 * std / med * 100:.1f}%)"
            )


if __name__ == "__main__":
    main()
