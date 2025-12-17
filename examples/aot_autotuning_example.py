#!/usr/bin/env python3
"""
AOT Autotuning Example
======================

This example demonstrates how to use the AOT (Ahead-of-Time) autotuning
workflow for Helion kernels.

The AOT workflow consists of three phases:
1. Collect: Run benchmarks, autotuning each shape individually
2. Measure: Re-run benchmarks, measuring all configs across all shapes
3. Evaluate: Generate heuristics and validate performance

Usage:
    # Run the full workflow using the AOT runner
    python -m helion.autotuner.aot_runner --benchmark "python examples/aot_autotuning_example.py"

    # Or run individual phases:
    HELION_AOT_MODE=collect HELION_AOT_DATA_DIR=./aot_data python examples/aot_autotuning_example.py
    HELION_AOT_MODE=measure HELION_AOT_DATA_DIR=./aot_data python examples/aot_autotuning_example.py
    python -c "from helion.autotuner.heuristic_generator import generate_heuristic; from pathlib import Path; generate_heuristic(Path('./aot_data/measurements_*.csv'), Path('./aot_data'))"
    HELION_AOT_MODE=evaluate HELION_AOT_DATA_DIR=./aot_data python examples/aot_autotuning_example.py
"""

from __future__ import annotations

import os

import torch

import helion
from helion._testing import DEVICE
import helion.language as hl


# Define a simple kernel for demonstration
@helion.kernel
def vector_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale a vector by a constant."""
    n = x.size(0)
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n] * scale
    return out


@helion.kernel
def rms_norm_simple(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Simplified RMS normalization."""
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(x_tile * x_tile, dim=-1) + eps)
        out[tile_m, :] = (x_tile / rms[:, None]).to(out.dtype)
    return out


def benchmark_kernels() -> None:
    """Run benchmarks on various shapes."""
    print(f"AOT Mode: {os.environ.get('HELION_AOT_MODE', 'disabled')}")
    print(f"AOT Data Dir: {os.environ.get('HELION_AOT_DATA_DIR', 'N/A')}")
    print()

    # Test vector_scale with various sizes
    print("=== vector_scale kernel ===")
    for n in [1024, 4096, 16384]:
        x = torch.randn(n, device=DEVICE, dtype=torch.float16)
        result = vector_scale(x, 2.0)
        print(f"  Shape ({n},): output sum = {result.sum().item():.2f}")

    # Test rms_norm_simple with various shapes
    print()
    print("=== rms_norm_simple kernel ===")
    for m, n in [(128, 512), (256, 1024), (512, 2048)]:
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        result = rms_norm_simple(x)
        print(f"  Shape ({m}, {n}): output sum = {result.sum().item():.2f}")


def main() -> None:
    """Main entry point."""
    # Check if we're in AOT mode
    aot_mode = os.environ.get("HELION_AOT_MODE", "disabled")

    if aot_mode == "disabled":
        print("Running in normal mode (no AOT)")
        print("Set HELION_AOT_MODE=collect|measure|evaluate to enable AOT workflow")
        print()

    # Enable AOT cache if in AOT mode
    if aot_mode != "disabled":
        os.environ["HELION_AUTOTUNE_CACHE"] = "AOTAutotuneCache"

    benchmark_kernels()

    if aot_mode != "disabled":
        print()
        print(f"AOT {aot_mode} phase completed!")
        data_dir = os.environ.get("HELION_AOT_DATA_DIR", ".helion_aot")
        print(f"Data saved to: {data_dir}")


if __name__ == "__main__":
    main()
