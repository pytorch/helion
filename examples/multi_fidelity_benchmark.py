"""
Multi-Fidelity Autotuner Benchmark
===================================

Compares the multi-fidelity autotuner wrapper against baseline search algorithms.
MultiFidelitySearch evaluates candidates at smaller tensor sizes first to filter
out bad configs, then only runs expensive full-size benchmarks on survivors.

Usage:
    HELION_AUTOTUNE_EFFORT=quick python examples/multi_fidelity_benchmark.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch

import helion
import helion.language as hl


@helion.kernel()
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel()
def rms_norm_kernel(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        x_squared = x_tile * x_tile
        mean_x_sq = torch.mean(x_squared, dim=-1)
        inv_rms = torch.rsqrt(mean_x_sq + eps)
        normalized = x_tile * inv_rms[:, None]
        out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(out.dtype)
    return out


def _run_single_benchmark(
    kernel_name: str,
    autotuner: str,
    inner: str | None,
) -> float:
    """Run a single benchmark in a subprocess to avoid cache sharing."""
    env = os.environ.copy()
    env["HELION_AUTOTUNER"] = autotuner
    env["HELION_FORCE_AUTOTUNE"] = "1"
    env["HELION_AUTOTUNE_EFFORT"] = env.get("HELION_AUTOTUNE_EFFORT", "quick")
    if inner is not None:
        env["HELION_MULTI_FIDELITY_INNER"] = inner

    script = f"""
import time, torch
import helion
import helion.language as hl

@helion.kernel()
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out

@helion.kernel()
def rms_norm_kernel(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        x_squared = x_tile * x_tile
        mean_x_sq = torch.mean(x_squared, dim=-1)
        inv_rms = torch.rsqrt(mean_x_sq + eps)
        normalized = x_tile * inv_rms[:, None]
        out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(out.dtype)
    return out

kernels = {{
    "add_kernel": (add_kernel, lambda: (
        torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16),
        torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16),
    )),
    "rms_norm": (rms_norm_kernel, lambda: (
        torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16),
        torch.randn(4096, device="cuda", dtype=torch.bfloat16),
    )),
}}

kernel_fn, make_args = kernels["{kernel_name}"]
args = make_args()
t0 = time.perf_counter()
kernel_fn(*args)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"TIME:{{elapsed:.4f}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    # Parse the TIME output
    for line in result.stdout.splitlines():
        if line.startswith("TIME:"):
            return float(line.split(":")[1])
    # If we didn't find TIME, print stderr for debugging
    print(f"  stderr: {result.stderr[-500:]}" if result.stderr else "")
    return -1.0


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available. Skipping benchmark.")
        return

    kernel_names = ["add_kernel", "rms_norm"]
    baseline_algo = "PatternSearch"
    results: list[tuple[str, float, float, float]] = []

    for name in kernel_names:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {name}")
        print(f"{'=' * 60}")

        print(f"\n--- Baseline: {baseline_algo} ---")
        baseline_time = _run_single_benchmark(name, baseline_algo, None)
        print(f"  Wall time: {baseline_time:.1f}s")

        print(f"\n--- MultiFidelity + {baseline_algo} ---")
        mf_time = _run_single_benchmark(name, "MultiFidelitySearch", baseline_algo)
        print(f"  Wall time: {mf_time:.1f}s")

        speedup = baseline_time / mf_time if mf_time > 0 else 0.0
        results.append((name, baseline_time, mf_time, speedup))

    # Print summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    header = f"{'Kernel':<20} | {'Baseline':>10} | {'MF+Algo':>10} | {'Speedup':>8}"
    print(header)
    print("-" * len(header))
    for name, bt, mt, sp in results:
        print(f"{name:<20} | {bt:>9.1f}s | {mt:>9.1f}s | {sp:>7.2f}x")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
