"""
Focused sensitivity microbenchmark: understand WHY increasing rep doesn't help,
and explore strategies that might actually sharpen discrimination.

Hypothesis: The noise comes from CUDA event timing resolution and system-level
jitter, not from statistical sampling. More reps don't help because the median
is already stable — the issue is that the median itself fluctuates between
trials due to external factors (clock throttling, L2 state, scheduler).

Strategy ideas:
1. Batch-of-batches: run many short do_bench calls, take min-of-medians
2. Interleaved A/B within each batch to cancel external drift
3. Increase the INNER repeat count (more samples per median) vs OUTER trials
4. Use min instead of median (less noise-robust but more stable for fast kernels)
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


def make_callable(num_tokens, intermediate_size, config):
    inp = torch.randn(
        num_tokens, 2 * intermediate_size, device="cuda", dtype=torch.bfloat16
    )
    scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    silu_mul_fp8.reset()
    bound = silu_mul_fp8.bind((inp, scale))
    compiled_fn = bound.compile_config(config)
    compiled_fn(inp, scale)
    return functools.partial(compiled_fn, inp, scale)


def raw_event_bench(fn, n_samples):
    """Collect n_samples raw CUDA event timings. Returns list of ms."""
    from triton import runtime

    di = runtime.driver.active.get_device_interface()
    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # warmup
    for _ in range(50):
        fn()
    di.synchronize()

    times = []
    for _ in range(n_samples):
        runtime.driver.active.clear_cache(cache)
        s = di.Event(enable_timing=True)
        e = di.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        di.synchronize()
        times.append(s.elapsed_time(e))
    return times


def analyze(times_a, times_b, method_name):
    """Quick analysis of two timing distributions."""
    med_a, med_b = statistics.median(times_a), statistics.median(times_b)
    std_a = statistics.stdev(times_a) if len(times_a) > 1 else 0
    std_b = statistics.stdev(times_b) if len(times_b) > 1 else 0
    pooled_std = math.sqrt((std_a**2 + std_b**2) / 2) if (std_a + std_b) > 0 else 1e-9
    d = abs(med_a - med_b) / pooled_std
    # Pairwise: for each trial, does A < B consistently?
    n = min(len(times_a), len(times_b))
    correct = sum(1 for i in range(n) if (times_a[i] < times_b[i]) == (med_a < med_b))
    acc = correct / n

    print(
        f"  {method_name:<35} A={med_a:.5f}±{std_a:.5f}  B={med_b:.5f}±{std_b:.5f}  "
        f"Δ={abs(med_a - med_b):.5f}ms ({abs(med_a - med_b) / min(med_a, med_b) * 100:.1f}%)  "
        f"d={d:.2f}  pair={acc:.0%}"
    )


def main():
    print("=" * 100)
    print("Autotuner Sensitivity: Strategies for Sharper Discrimination")
    print("=" * 100)

    # Test case: MODERATE difference (~15% true diff on do_bench)
    # [4,8] vs [64,32] @ 64x4096
    print()
    print("Test case: [4,8] vs [64,32] @ 64x4096  (MODERATE: should be ~15% diff)")
    print("-" * 100)

    fn_a = make_callable(
        64, 4096, {"block_sizes": [4, 8], "num_warps": 2, "num_stages": 1}
    )
    fn_b = make_callable(
        64, 4096, {"block_sizes": [64, 32], "num_warps": 8, "num_stages": 1}
    )

    N_TRIALS = 30

    # Strategy 1: Current autotuner (do_bench, rep=50, median) × N trials
    print()
    print(
        "Strategy 1: do_bench with various rep budgets (take median of N samples per trial)"
    )
    for rep in [50, 100, 500]:
        results_a, results_b = [], []
        for _ in range(N_TRIALS):
            results_a.append(do_bench(fn_a, warmup=5, rep=rep, return_mode="median"))
            results_b.append(do_bench(fn_b, warmup=5, rep=rep, return_mode="median"))
        analyze(results_a, results_b, f"do_bench(rep={rep}, return=median)")

    # Strategy 2: do_bench with min instead of median
    print()
    print("Strategy 2: do_bench with min return mode (more stable for fast kernels?)")
    for rep in [50, 100, 500]:
        results_a, results_b = [], []
        for _ in range(N_TRIALS):
            results_a.append(do_bench(fn_a, warmup=5, rep=rep, return_mode="min"))
            results_b.append(do_bench(fn_b, warmup=5, rep=rep, return_mode="min"))
        analyze(results_a, results_b, f"do_bench(rep={rep}, return=min)")

    # Strategy 3: Raw event timing — many samples, different aggregations
    print()
    print("Strategy 3: Raw CUDA events — various sample counts and aggregations")
    for n_samples in [100, 1000, 5000, 10000]:
        raw_a = raw_event_bench(fn_a, n_samples)
        raw_b = raw_event_bench(fn_b, n_samples)

        # Try different aggregation strategies
        med_a, med_b = statistics.median(raw_a), statistics.median(raw_b)
        min_a, min_b = min(raw_a), min(raw_b)
        p5_a = sorted(raw_a)[max(0, len(raw_a) * 5 // 100)]
        p5_b = sorted(raw_b)[max(0, len(raw_b) * 5 // 100)]

        print(
            f"  n={n_samples:>5}: median A={med_a:.5f} B={med_b:.5f} Δ={abs(med_a - med_b):.5f} ({abs(med_a - med_b) / min(med_a, med_b) * 100:.1f}%)"
            f"  |  min A={min_a:.5f} B={min_b:.5f} Δ={abs(min_a - min_b):.5f} ({abs(min_a - min_b) / min(min_a, min_b) * 100:.1f}%)"
            f"  |  p5 A={p5_a:.5f} B={p5_b:.5f} Δ={abs(p5_a - p5_b):.5f} ({abs(p5_a - p5_b) / min(p5_a, p5_b) * 100:.1f}%)"
        )

    # Strategy 4: Interleaved bench — controls for drift
    print()
    print("Strategy 4: Interleaved bench (A/B alternating within each trial)")
    for repeat in [100, 500, 1000, 5000]:
        results_a, results_b = [], []
        for _ in range(N_TRIALS):
            t = interleaved_bench([fn_a, fn_b], repeat=repeat)
            results_a.append(t[0])
            results_b.append(t[1])
        analyze(results_a, results_b, f"interleaved(repeat={repeat})")

    # Strategy 5: Adaptive — only spend extra time on fast kernels
    print()
    print(
        "Strategy 5: Adaptive wall-clock budget (2s for <0.1ms kernels, 200ms otherwise)"
    )
    est_ms = statistics.median(
        [do_bench(fn_a, warmup=1, rep=10, return_mode="min") for _ in range(3)]
    )
    assert isinstance(est_ms, float)
    if est_ms < 0.1:
        budget_ms = 2000
    else:
        budget_ms = 200
    adaptive_repeat = min(10000, max(100, int(budget_ms / est_ms)))
    print(f"  est_ms={est_ms:.4f}, budget={budget_ms}ms, repeat={adaptive_repeat}")
    results_a, results_b = [], []
    for _ in range(N_TRIALS):
        t = interleaved_bench([fn_a, fn_b], repeat=adaptive_repeat)
        results_a.append(t[0])
        results_b.append(t[1])
    analyze(results_a, results_b, f"adaptive_interleaved(repeat={adaptive_repeat})")

    # ================================================================
    # Repeat for SUBTLE case: [1,32] vs [1,256] @ 1x2048
    # ================================================================
    print()
    print()
    print("Test case: [1,32] vs [1,256] @ 1x2048  (SUBTLE: ~12% diff)")
    print("-" * 100)

    fn_a = make_callable(
        1, 2048, {"block_sizes": [1, 32], "num_warps": 4, "num_stages": 2}
    )
    fn_b = make_callable(
        1, 2048, {"block_sizes": [1, 256], "num_warps": 2, "num_stages": 1}
    )

    # Quick comparison of key strategies
    print()
    print("Key strategies comparison:")

    for rep in [50, 500]:
        results_a, results_b = [], []
        for _ in range(N_TRIALS):
            results_a.append(do_bench(fn_a, warmup=5, rep=rep, return_mode="median"))
            results_b.append(do_bench(fn_b, warmup=5, rep=rep, return_mode="median"))
        analyze(results_a, results_b, f"do_bench(rep={rep}, median)")

    for rep in [50, 500]:
        results_a, results_b = [], []
        for _ in range(N_TRIALS):
            results_a.append(do_bench(fn_a, warmup=5, rep=rep, return_mode="min"))
            results_b.append(do_bench(fn_b, warmup=5, rep=rep, return_mode="min"))
        analyze(results_a, results_b, f"do_bench(rep={rep}, min)")

    for n_samples in [1000, 10000]:
        raw_a = raw_event_bench(fn_a, n_samples)
        raw_b = raw_event_bench(fn_b, n_samples)
        med_a, med_b = statistics.median(raw_a), statistics.median(raw_b)
        min_a, min_b = min(raw_a), min(raw_b)
        p5_a = sorted(raw_a)[max(0, len(raw_a) * 5 // 100)]
        p5_b = sorted(raw_b)[max(0, len(raw_b) * 5 // 100)]
        print(
            f"  raw(n={n_samples:>5}): median Δ={abs(med_a - med_b):.5f}({abs(med_a - med_b) / min(med_a, med_b) * 100:.1f}%)"
            f"  min Δ={abs(min_a - min_b):.5f}({abs(min_a - min_b) / min(min_a, min_b) * 100:.1f}%)"
            f"  p5 Δ={abs(p5_a - p5_b):.5f}({abs(p5_a - p5_b) / min(p5_a, p5_b) * 100:.1f}%)"
        )

    for repeat in [1000, 5000]:
        results_a, results_b = [], []
        for _ in range(N_TRIALS):
            t = interleaved_bench([fn_a, fn_b], repeat=repeat)
            results_a.append(t[0])
            results_b.append(t[1])
        analyze(results_a, results_b, f"interleaved(repeat={repeat})")

    # Adaptive
    est_ms = statistics.median(
        [do_bench(fn_a, warmup=1, rep=10, return_mode="min") for _ in range(3)]
    )
    assert isinstance(est_ms, float)
    budget_ms = 2000 if est_ms < 0.1 else 200
    adaptive_repeat = min(10000, max(100, int(budget_ms / est_ms)))
    print(
        f"\n  Adaptive: est={est_ms:.4f}ms, budget={budget_ms}ms, repeat={adaptive_repeat}"
    )
    results_a, results_b = [], []
    for _ in range(N_TRIALS):
        t = interleaved_bench([fn_a, fn_b], repeat=adaptive_repeat)
        results_a.append(t[0])
        results_b.append(t[1])
    analyze(results_a, results_b, f"adaptive_interleaved(repeat={adaptive_repeat})")

    # ================================================================
    # Repeat for NOISE case: same block sizes, different warps @ 8x4096
    # ================================================================
    print()
    print()
    print("Test case: warps=2 vs warps=4, same blocks @ 8x4096  (NOISE: ~2.5% diff)")
    print("-" * 100)

    fn_a = make_callable(
        8, 4096, {"block_sizes": [8, 32], "num_warps": 2, "num_stages": 1}
    )
    fn_b = make_callable(
        8, 4096, {"block_sizes": [8, 32], "num_warps": 4, "num_stages": 1}
    )

    print()
    print("Key strategies comparison:")

    for rep in [50, 500]:
        results_a, results_b = [], []
        for _ in range(N_TRIALS):
            results_a.append(do_bench(fn_a, warmup=5, rep=rep, return_mode="min"))
            results_b.append(do_bench(fn_b, warmup=5, rep=rep, return_mode="min"))
        analyze(results_a, results_b, f"do_bench(rep={rep}, min)")

    for repeat in [1000, 5000]:
        results_a, results_b = [], []
        for _ in range(N_TRIALS):
            t = interleaved_bench([fn_a, fn_b], repeat=repeat)
            results_a.append(t[0])
            results_b.append(t[1])
        analyze(results_a, results_b, f"interleaved(repeat={repeat})")

    est_ms = statistics.median(
        [do_bench(fn_a, warmup=1, rep=10, return_mode="min") for _ in range(3)]
    )
    assert isinstance(est_ms, float)
    budget_ms = 2000 if est_ms < 0.1 else 200
    adaptive_repeat = min(10000, max(100, int(budget_ms / est_ms)))
    print(
        f"\n  Adaptive: est={est_ms:.4f}ms, budget={budget_ms}ms, repeat={adaptive_repeat}"
    )
    results_a, results_b = [], []
    for _ in range(N_TRIALS):
        t = interleaved_bench([fn_a, fn_b], repeat=adaptive_repeat)
        results_a.append(t[0])
        results_b.append(t[1])
    analyze(results_a, results_b, f"adaptive_interleaved(repeat={adaptive_repeat})")


if __name__ == "__main__":
    main()
