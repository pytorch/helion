"""
Experiment to test the effect of mutation_distribution on autotuning performance.

Tests 5 kernels with 3 distributions (geometric, log_uniform, harmonic)
and alpha values of 0, 1, 2, and 10.
Each configuration is run 5 times to get confidence intervals.
Uses effort="quick" for all runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
import statistics
import time
from typing import Callable

import torch

# Use quick effort
os.environ["HELION_AUTOTUNE_EFFORT"] = "quick"

import helion
from helion._testing import DEVICE
import helion.language as hl

# ============= Experiment Infrastructure =============


@dataclass
class ExperimentResult:
    kernel_name: str
    distribution: str
    alpha: float
    run_idx: int
    duration_seconds: float
    final_perf_ms: float
    config: dict


def run_single_autotune(
    kernel_factory: Callable,
    args: tuple,
    alpha: float,
    distribution: str,
) -> tuple[float, float, dict]:
    """
    Run a single autotuning session.

    Returns: (duration_seconds, final_perf_ms, config_dict)
    """
    # Create a fresh kernel to avoid caching
    kernel_fn = kernel_factory()

    gc.collect()
    torch.cuda.empty_cache()

    # Run autotuning with timing (force=True to ignore cache)
    start_time = time.perf_counter()
    config = kernel_fn.autotune(
        args,
        force=True,
        mutation_alpha=alpha,
        mutation_distribution=distribution,
    )
    end_time = time.perf_counter()

    duration = end_time - start_time

    # Benchmark the final config to get performance
    # The kernel is already configured after autotune()
    # Warm up
    for _ in range(3):
        kernel_fn(*args)
    torch.cuda.synchronize()

    # Benchmark
    n_iters = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iters):
        kernel_fn(*args)
    end.record()
    torch.cuda.synchronize()

    perf_ms = start.elapsed_time(end) / n_iters

    return duration, perf_ms, config.config


def run_experiment(
    kernel_name: str,
    kernel_factory: Callable,
    args: tuple,
    distributions: list[str],
    alphas: list[float],
    n_runs: int,
) -> list[ExperimentResult]:
    """Run experiment for a single kernel across all distributions and alpha values."""
    results = []

    for distribution in distributions:
        for alpha in alphas:
            # Skip alpha=0 for non-geometric since it's equivalent
            if alpha == 0 and distribution != "geometric":
                continue

            print(f"\n  Testing distribution={distribution}, alpha={alpha}...")
            for run_idx in range(n_runs):
                print(f"    Run {run_idx + 1}/{n_runs}...", end=" ", flush=True)
                try:
                    duration, perf_ms, config = run_single_autotune(
                        kernel_factory, args, alpha, distribution
                    )
                    result = ExperimentResult(
                        kernel_name=kernel_name,
                        distribution=distribution,
                        alpha=alpha,
                        run_idx=run_idx,
                        duration_seconds=duration,
                        final_perf_ms=perf_ms,
                        config=config,
                    )
                    results.append(result)
                    print(f"duration={duration:.2f}s, perf={perf_ms:.4f}ms")
                except Exception as e:
                    print(f"FAILED: {e}")

    return results


def summarize_results(results: list[ExperimentResult]) -> None:
    """Print summary statistics for the results."""
    # Group by kernel, distribution, and alpha
    from collections import defaultdict

    grouped: dict[tuple[str, str, float], list[ExperimentResult]] = defaultdict(list)
    for r in results:
        grouped[(r.kernel_name, r.distribution, r.alpha)].append(r)

    print("\n" + "=" * 90)
    print("SUMMARY RESULTS")
    print("=" * 90)

    # Print by kernel
    kernels = sorted(set(r.kernel_name for r in results))
    distributions = ["geometric", "log_uniform", "harmonic"]
    alphas = sorted(set(r.alpha for r in results))

    for kernel in kernels:
        print(f"\n{kernel}:")
        print("-" * 85)
        print(
            f"{'Distribution':>12} | {'Alpha':>6} | {'Duration (s)':>20} | {'Performance (ms)':>20}"
        )
        print(f"{'':>12} | {'':>6} | {'mean +/- std':>20} | {'mean +/- std':>20}")
        print("-" * 85)

        for distribution in distributions:
            for alpha in alphas:
                # Skip alpha=0 for non-geometric
                if alpha == 0 and distribution != "geometric":
                    continue

                key = (kernel, distribution, alpha)
                if key not in grouped:
                    continue

                runs = grouped[key]
                durations = [r.duration_seconds for r in runs]
                perfs = [r.final_perf_ms for r in runs]

                dur_mean = statistics.mean(durations)
                dur_std = statistics.stdev(durations) if len(durations) > 1 else 0
                perf_mean = statistics.mean(perfs)
                perf_std = statistics.stdev(perfs) if len(perfs) > 1 else 0

                print(
                    f"{distribution:>12} | {alpha:>6.1f} | {dur_mean:>8.2f} +/- {dur_std:<8.2f} | {perf_mean:>8.4f} +/- {perf_std:<8.4f}"
                )


def main():
    print("=" * 90)
    print("MUTATION DISTRIBUTION EXPERIMENT")
    print("=" * 90)
    print(f"Device: {DEVICE}")
    print("Distributions: [geometric, log_uniform, harmonic]")
    print("Alphas: [0, 1, 2, 10]")
    print("Runs per config: 5")
    print("Effort: quick")
    print("=" * 90)

    # Define test cases using kernel factories to avoid caching issues
    distributions = ["geometric", "log_uniform", "harmonic"]
    alphas = [0.0, 1.0, 2.0, 10.0]
    n_runs = 5

    # Kernel factories - return fresh kernel each time
    def make_softmax():
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            block_size_m = hl.register_block_size(m)
            block_size_n = hl.register_block_size(n)
            for tile_m in hl.tile(m, block_size=block_size_m):
                mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
                di = hl.zeros([tile_m], dtype=torch.float32)
                for tile_n in hl.tile(n, block_size=block_size_n):
                    values = x[tile_m, tile_n]
                    local_amax = torch.amax(values, dim=1)
                    mi_next = torch.maximum(mi, local_amax)
                    di = di * torch.exp(mi - mi_next) + torch.exp(
                        values - mi_next[:, None]
                    ).sum(dim=1)
                    mi = mi_next
                for tile_n in hl.tile(n, block_size=block_size_n):
                    values = x[tile_m, tile_n]
                    out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
            return out

        return fn

    def make_matmul():
        @helion.kernel()
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.zeros([m, n], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                for tile_n in hl.tile(n):
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                    out[tile_m, tile_n] = acc.to(x.dtype)
            return out

        return fn

    def make_vector_add():
        @helion.kernel()
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            for tile in hl.tile(n):
                out[tile] = x[tile] + y[tile]
            return out

        return fn

    def make_reduce_sum():
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.zeros([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m], dtype=torch.float32)
                for tile_n in hl.tile(n):
                    acc += x[tile_m, tile_n].sum(dim=1)
                out[tile_m] = acc.to(x.dtype)
            return out

        return fn

    def make_elementwise_mul():
        @helion.kernel()
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                for tile_n in hl.tile(n):
                    out[tile_m, tile_n] = x[tile_m, tile_n] * y[tile_m, tile_n]
            return out

        return fn

    test_cases = [
        (
            "softmax_two_pass",
            make_softmax,
            (torch.randn(2048, 2048, device=DEVICE, dtype=torch.float16),),
        ),
        (
            "matmul_simple",
            make_matmul,
            (
                torch.randn(1024, 512, device=DEVICE, dtype=torch.float16),
                torch.randn(512, 1024, device=DEVICE, dtype=torch.float16),
            ),
        ),
        (
            "vector_add",
            make_vector_add,
            (
                torch.randn(1024 * 1024, device=DEVICE, dtype=torch.float16),
                torch.randn(1024 * 1024, device=DEVICE, dtype=torch.float16),
            ),
        ),
        (
            "reduce_sum",
            make_reduce_sum,
            (torch.randn(4096, 4096, device=DEVICE, dtype=torch.float16),),
        ),
        (
            "elementwise_mul",
            make_elementwise_mul,
            (
                torch.randn(2048, 2048, device=DEVICE, dtype=torch.float16),
                torch.randn(2048, 2048, device=DEVICE, dtype=torch.float16),
            ),
        ),
    ]

    all_results = []

    for kernel_name, kernel_fn, args in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Testing: {kernel_name}")
        print(f"{'=' * 60}")

        results = run_experiment(
            kernel_name, kernel_fn, args, distributions, alphas, n_runs
        )
        all_results.extend(results)

    # Print summary
    summarize_results(all_results)

    # Save raw results to JSON
    output_file = "mutation_distribution_results.json"
    with open(output_file, "w") as f:
        json.dump(
            [
                {
                    "kernel": r.kernel_name,
                    "distribution": r.distribution,
                    "alpha": r.alpha,
                    "run": r.run_idx,
                    "duration_s": r.duration_seconds,
                    "perf_ms": r.final_perf_ms,
                    "config": r.config,
                }
                for r in all_results
            ],
            f,
            indent=2,
        )
    print(f"\nRaw results saved to: {output_file}")


if __name__ == "__main__":
    main()
