"""
AOT Multi-Config Autotuning Example
===================================

This example demonstrates AOT autotuning with a kernel that requires
different configurations for different input shapes. The kernel has
2D block sizes that have different optimal values for:
- Tall-and-skinny matrices (M >> N): Small block_m, large block_n
- Short-and-wide matrices (M << N): Large block_m, small block_n
- Square matrices: Balanced block sizes

This generates heuristics with actual decision tree logic rather than
a single config.

Usage:
    python -m helion.autotuner.aot_runner --benchmark "python examples/aot_multiconfig_example.py"
"""

from __future__ import annotations

import os

import torch
from triton.testing import do_bench

import helion
from helion._testing import DEVICE
import helion.language as hl


@helion.kernel(autotune_cache="AOTAutotuneCache")
def row_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Row-wise softmax with explicit 2D tiling.

    The optimal block sizes depend on the matrix shape:
    - Tall matrices benefit from larger row tiles
    - Wide matrices benefit from larger column tiles
    """
    m, n = x.size()
    out = torch.empty_like(x)

    # Explicit block size registration allows tuning
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
            # Rescale previous sum and add new contributions
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


@helion.kernel()
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


def benchmark_kernels() -> None:
    """Run benchmarks on various shapes and dtypes, reporting GB/s."""
    print(f"AOT Mode: {os.environ.get('HELION_AOT_MODE', 'disabled')}")
    print(f"AOT Data Dir: {os.environ.get('HELION_AOT_DATA_DIR', 'N/A')}")
    print()

    # Define shapes covering different aspect ratios
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

    # Test multiple dtypes - different dtypes often need different tile sizes
    dtypes = [torch.float16, torch.float32]

    print("=== row_softmax kernel ===")
    for dtype in dtypes:
        print(f"\n  dtype={dtype}:")
        print(f"  {'Shape':>16} {'Time (ms)':>12} {'GB/s':>10} {'Correct':>8}")
        print("  " + "-" * 50)
        for m, n in shapes:
            x = torch.randn(m, n, device=DEVICE, dtype=dtype)
            # Warmup
            result = row_softmax(x)
            # Verify softmax property: each row sums to 1
            row_sums = result.sum(dim=1)
            correct = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
            # Benchmark
            time_ms = do_bench(lambda x=x: row_softmax(x))
            # GB/s: softmax reads input twice (for max and for exp) + writes output
            # Actually it reads 2x and writes 1x = 3 passes
            total_bytes = x.numel() * x.element_size() * 3
            gbps = total_bytes / time_ms * 1e-6
            print(f"  {(m, n)!s:>16} {time_ms:>12.4f} {gbps:>10.2f} {correct!s:>8}")


def main() -> None:
    """Main entry point."""
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
