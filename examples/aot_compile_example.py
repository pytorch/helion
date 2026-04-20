"""
AOT Standalone Compilation Example
===================================

Generates a single ``.py`` file from Helion kernels that has **zero Helion
dependencies** at runtime — only ``torch``, ``triton``, and
``triton.language``.

Run the full AOT workflow with ``--standalone`` to produce standalone files::

    python -m helion.experimental.aot_runner --standalone \\
        -- python examples/aot_compile_example.py

This runs collect → measure → build → evaluate, then compiles standalone
Triton code and writes ``examples/aot_compile_example_add_2d_standalone.py``
next to this file.

The standalone file is a drop-in replacement::

    # before  (needs helion installed)
    from examples.aot_compile_example import add_2d

    # after   (pure triton — no helion)
    from examples.aot_compile_example_add_2d_standalone import add_2d
"""

from __future__ import annotations

import argparse
import os

import torch

from helion._testing import DEVICE
from helion._testing import do_bench
import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel()
def add_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise addition of two 2-D tensors."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


KERNEL_BENCHMARKS = {
    "add_2d": "benchmark_add_2d",
}


def benchmark_add_2d() -> None:
    """Benchmark add_2d across a range of shapes."""
    shapes = [
        (64, 128),
        (256, 256),
        (1024, 1024),
        (4096, 4096),
    ]
    print("=== add_2d kernel ===")
    print(f"{'Shape':>16} {'Time (ms)':>12} {'GB/s':>10}")
    print("-" * 40)
    for m, n in shapes:
        x = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
        add_2d(x, y)  # warmup
        time_ms = do_bench(lambda x=x, y=y: add_2d(x, y))
        assert isinstance(time_ms, float)
        total_bytes = x.numel() * x.element_size() * 3  # 2 reads + 1 write
        gbps = total_bytes / time_ms * 1e-6
        print(f"{(m, n)!s:>16} {time_ms:>12.4f} {gbps:>10.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AOT Compile Example")
    parser.add_argument(
        "--kernel",
        "-k",
        type=str,
        action="append",
        dest="kernels",
        help="Kernel(s) to benchmark (can be repeated). Default: all",
    )
    args = parser.parse_args()

    kernels = args.kernels
    if kernels is None:
        env_kernels = os.environ.get("HELION_AOT_KERNELS", "")
        if env_kernels:
            kernels = env_kernels.split(",")

    aot_mode = os.environ.get("HELION_AOT_MODE", "disabled")
    if aot_mode != "disabled":
        print(f"AOT mode: {aot_mode}")
        print()

    targets = kernels or list(KERNEL_BENCHMARKS.keys())
    for name in targets:
        if name in KERNEL_BENCHMARKS:
            globals()[KERNEL_BENCHMARKS[name]]()
        else:
            print(f"Unknown kernel: {name}")


if __name__ == "__main__":
    main()
