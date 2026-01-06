"""
AOT Autotuning Example
======================

This example demonstrates how to use the AOT (Ahead-of-Time) autotuning
workflow for Helion kernels. It includes examples of:
- Simple 1D kernels (vector_scale)
- 2D kernels with varying aspect ratios (row_softmax, col_reduce_sum)
- Batch-aware kernels (rms_norm_batched)

The AOT workflow consists of four phases:
1. Collect: Run benchmarks, autotuning each shape individually
2. Measure: Re-run benchmarks, measuring all configs across all shapes
3. Build: Generate heuristics using decision trees
4. Evaluate: Validate performance using the generated heuristics

Usage:
    # Run the full workflow using the AOT runner
    python -m helion.autotuner.aot_runner --benchmark "python examples/aot_example.py"

    # Run only specific kernels
    python -m helion.autotuner.aot_runner --benchmark "python examples/aot_example.py" --kernel row_softmax

    # Or run individual phases manually:
    HELION_AOT_MODE=collect HELION_AOT_DATA_DIR=./aot_data python examples/aot_example.py
    HELION_AOT_MODE=measure HELION_AOT_DATA_DIR=./aot_data python examples/aot_example.py

Using @helion.aot_kernel():
    The simplest way to use AOT autotuning is with the @helion.aot_kernel() decorator:

    @helion.aot_kernel()
    def my_kernel(...):
        ...

    This automatically configures the kernel for AOT autotuning with:
    - AOTAutotuneCache for heuristic-based config selection
    - static_shapes=False for dynamic shape handling
    - aot_key for shape-based specialization
"""

from __future__ import annotations

import argparse
import os

import torch
from triton.testing import do_bench

import helion
from helion._testing import DEVICE
import helion.language as hl


# ============================================================================
# Simple 1D Kernel
# ============================================================================


@helion.aot_kernel()
def vector_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale a vector by a constant."""
    n = x.size(0)
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n] * scale
    return out


# ============================================================================
# 2D Kernels with varying aspect ratios
# These demonstrate multi-config heuristics where different shapes need
# different optimal configurations.
# ============================================================================


@helion.aot_kernel(static_shapes=False)
def row_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Row-wise softmax with explicit 2D tiling.

    The optimal block sizes depend on the matrix shape:
    - Tall matrices benefit from larger row tiles
    - Wide matrices benefit from larger column tiles
    """
    m, n = x.size()
    out = torch.empty_like(x)

    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)

    for tile_m in hl.tile(m, block_size=block_m):
        # First pass: compute max and sum for numerical stability
        row_max = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        row_sum = hl.zeros([tile_m], dtype=torch.float32)

        for tile_n in hl.tile(n, block_size=block_n):
            values = x[tile_m, tile_n].to(torch.float32)
            local_max = torch.amax(values, dim=1)
            new_max = torch.maximum(row_max, local_max)
            row_sum = row_sum * torch.exp(row_max - new_max) + torch.sum(
                torch.exp(values - new_max[:, None]), dim=1
            )
            row_max = new_max

        # Second pass: compute softmax output
        for tile_n in hl.tile(n, block_size=block_n):
            values = x[tile_m, tile_n].to(torch.float32)
            out[tile_m, tile_n] = (
                torch.exp(values - row_max[:, None]) / row_sum[:, None]
            ).to(out.dtype)

    return out


@helion.aot_kernel()
def col_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Column-wise sum reduction with 2D tiling.

    For tall matrices, we want to process many rows in parallel.
    For wide matrices, we want larger column blocks.
    """
    m, n = x.size()
    out = torch.zeros(n, dtype=x.dtype, device=x.device)

    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)

    for tile_n in hl.tile(n, block_size=block_n):
        col_acc = hl.zeros([tile_n], dtype=torch.float32)

        for tile_m in hl.tile(m, block_size=block_m):
            col_acc += torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)

        out[tile_n] = col_acc.to(out.dtype)

    return out


# ============================================================================
# Batch-aware kernels
# The batched parameter tells the autotuner which dimensions are batch
# dimensions, allowing inputs with different batch sizes but the same
# non-batch dimensions to share the same optimized config.
# ============================================================================


@helion.aot_kernel(batched=[[0, None], None])
def rms_norm_batched(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS normalization with batch-aware heuristic.

    The batched=[[0, None], None] parameter means:
    - x (arg 0): 2D tensor where dim 0 is batched, dim 1 is not
    - eps (arg 1): scalar (None)

    This allows different batch sizes to share the same optimized config,
    as long as the hidden dimension is the same.
    """
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(x_tile * x_tile, dim=-1) + eps)
        out[tile_m, :] = (x_tile / rms[:, None]).to(out.dtype)
    return out


# ============================================================================
# Benchmarking
# ============================================================================


def benchmark_vector_scale() -> None:
    """Benchmark vector_scale kernel."""
    print("=== vector_scale kernel ===")
    print(f"{'Shape':>12} {'Time (ms)':>12} {'GB/s':>10}")
    print("-" * 36)
    for n in [1024, 4096, 16384, 65536, 262144, 1048576]:
        x = torch.randn(n, device=DEVICE, dtype=torch.float16)
        vector_scale(x, 2.0)  # Warmup
        time_ms = do_bench(lambda x=x: vector_scale(x, 2.0))
        assert isinstance(time_ms, float)
        total_bytes = x.numel() * x.element_size() * 2  # read + write
        gbps = total_bytes / time_ms * 1e-6
        print(f"{(n,)!s:>12} {time_ms:>12.4f} {gbps:>10.2f}")


def benchmark_row_softmax() -> None:
    """Benchmark row_softmax kernel with various aspect ratios."""
    # Shapes covering different aspect ratios
    shapes = [
        # Tall and skinny (M >> N)
        (8192, 64),
        (4096, 128),
        (2048, 256),
        # Square-ish
        (1024, 1024),
        (2048, 512),
        (512, 2048),
        # Short and wide (M << N)
        (256, 2048),
        (128, 4096),
        (64, 8192),
    ]

    dtypes = [torch.float16, torch.float32]

    print("=== row_softmax kernel ===")
    for dtype in dtypes:
        print(f"\n  dtype={dtype}:")
        print(f"  {'Shape':>16} {'Time (ms)':>12} {'GB/s':>10} {'Correct':>8}")
        print("  " + "-" * 50)
        for m, n in shapes:
            x = torch.randn(m, n, device=DEVICE, dtype=dtype)
            result = row_softmax(x)  # Warmup
            # Verify softmax property: each row sums to 1
            row_sums = result.sum(dim=1)
            correct = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
            time_ms = do_bench(lambda x=x: row_softmax(x))
            assert isinstance(time_ms, float)
            # GB/s: softmax reads 2x + writes 1x = 3 passes
            total_bytes = x.numel() * x.element_size() * 3
            gbps = total_bytes / time_ms * 1e-6
            print(f"  {(m, n)!s:>16} {time_ms:>12.4f} {gbps:>10.2f} {correct!s:>8}")


def benchmark_col_reduce_sum() -> None:
    """Benchmark col_reduce_sum kernel."""
    shapes = [
        (8192, 64),
        (4096, 128),
        (1024, 1024),
        (128, 4096),
        (64, 8192),
    ]

    print("=== col_reduce_sum kernel ===")
    print(f"{'Shape':>16} {'Time (ms)':>12} {'GB/s':>10}")
    print("-" * 40)
    for m, n in shapes:
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
        col_reduce_sum(x)  # Warmup
        time_ms = do_bench(lambda x=x: col_reduce_sum(x))
        assert isinstance(time_ms, float)
        total_bytes = x.numel() * x.element_size() * 2  # read + write
        gbps = total_bytes / time_ms * 1e-6
        print(f"{(m, n)!s:>16} {time_ms:>12.4f} {gbps:>10.2f}")


def benchmark_rms_norm_batched() -> None:
    """Benchmark rms_norm_batched kernel with varying batch sizes."""
    print("=== rms_norm_batched kernel (batch-aware heuristic) ===")
    print(f"{'Shape':>16} {'Time (ms)':>12} {'GB/s':>10}")
    print("-" * 40)
    hidden = 4096
    for batch in [32, 64, 128, 256, 512]:
        x = torch.randn(batch, hidden, device=DEVICE, dtype=torch.float16)
        rms_norm_batched(x)  # Warmup
        time_ms = do_bench(lambda x=x: rms_norm_batched(x))
        assert isinstance(time_ms, float)
        total_bytes = x.numel() * x.element_size() * 2  # read + write
        gbps = total_bytes / time_ms * 1e-6
        print(f"{(batch, hidden)!s:>16} {time_ms:>12.4f} {gbps:>10.2f}")


# Map of kernel names to benchmark functions
KERNEL_BENCHMARKS = {
    "vector_scale": benchmark_vector_scale,
    "row_softmax": benchmark_row_softmax,
    "col_reduce_sum": benchmark_col_reduce_sum,
    "rms_norm_batched": benchmark_rms_norm_batched,
}


def benchmark_kernels(kernels: list[str] | None = None) -> None:
    """Run benchmarks on selected kernels."""
    print(f"AOT Mode: {os.environ.get('HELION_AOT_MODE', 'disabled')}")
    print(f"AOT Data Dir: {os.environ.get('HELION_AOT_DATA_DIR', 'N/A')}")
    print()

    if kernels is None:
        kernels = list(KERNEL_BENCHMARKS.keys())

    for i, kernel_name in enumerate(kernels):
        if kernel_name in KERNEL_BENCHMARKS:
            if i > 0:
                print()
            KERNEL_BENCHMARKS[kernel_name]()
        else:
            print(f"Unknown kernel: {kernel_name}")
            print(f"Available kernels: {', '.join(KERNEL_BENCHMARKS.keys())}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AOT Autotuning Example")
    parser.add_argument(
        "--kernel",
        "-k",
        type=str,
        action="append",
        dest="kernels",
        help="Kernel(s) to benchmark (can be repeated). Default: all",
    )
    args = parser.parse_args()

    # Get kernels from args or environment variable (set by aot_runner --kernel)
    kernels = args.kernels
    if kernels is None:
        env_kernels = os.environ.get("HELION_AOT_KERNELS", "")
        if env_kernels:
            kernels = env_kernels.split(",")

    aot_mode = os.environ.get("HELION_AOT_MODE", "disabled")

    if aot_mode == "disabled":
        print("Running in normal mode (no AOT)")
        print("Set HELION_AOT_MODE=collect|measure|evaluate to enable AOT workflow")
        print()
    else:
        print(f"Running in AOT mode: {aot_mode}")
        print("(Using @helion.aot_kernel() for automatic AOT configuration)")
        print()

    benchmark_kernels(kernels)

    if aot_mode not in ("disabled", "evaluate"):
        print()
        print(f"AOT {aot_mode} phase completed!")
        data_dir = os.environ.get("HELION_AOT_DATA_DIR", ".helion_aot")
        print(f"Data saved to: {data_dir}")


if __name__ == "__main__":
    main()
