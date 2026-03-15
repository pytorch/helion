"""
Grid Search Autotuner Benchmark
================================

Compares the GridSearch autotuner against baseline search algorithms.
GridSearch exhaustively enumerates (or samples from) the full configuration
space, benchmarks each candidate, and returns the best one.

Usage:
    HELION_AUTOTUNE_EFFORT=quick python examples/grid_search_benchmark.py
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
) -> float:
    """Run a single benchmark in a subprocess to avoid cache sharing."""
    env = os.environ.copy()
    env["HELION_AUTOTUNER"] = autotuner
    env["HELION_FORCE_AUTOTUNE"] = "1"
    env["HELION_AUTOTUNE_EFFORT"] = env.get("HELION_AUTOTUNE_EFFORT", "quick")

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
    for line in result.stdout.splitlines():
        if line.startswith("TIME:"):
            return float(line.split(":")[1])
    print(f"  stderr: {result.stderr[-500:]}" if result.stderr else "")
    return -1.0


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available. Skipping benchmark.")
        return

    kernel_names = ["add_kernel", "rms_norm"]
    algorithms = ["PatternSearch", "RandomSearch", "GridSearch"]
    results: dict[str, dict[str, float]] = {}

    for name in kernel_names:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {name}")
        print(f"{'=' * 60}")
        results[name] = {}

        for algo in algorithms:
            print(f"\n--- {algo} ---")
            t = _run_single_benchmark(name, algo)
            results[name][algo] = t
            print(f"  Wall time: {t:.1f}s")

    # Print summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    header = f"{'Kernel':<20}"
    for algo in algorithms:
        header += f" | {algo:>14}"
    print(header)
    print("-" * len(header))
    for name in kernel_names:
        row = f"{name:<20}"
        for algo in algorithms:
            t = results[name][algo]
            row += f" | {t:>13.1f}s"
        print(row)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
