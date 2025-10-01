"""
Split Reduction Example
=======================

This example demonstrates a two-phase split reduction strategy for summing a 1D vector.
Phase 1 splits the vector across tiles, each computing a partial sum.
Phase 2 combines the partial sums on the host.
"""

# %%
# Imports
# -------
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


# %%
# Split Reduction Kernel
# ---------------------
@helion.kernel()
def vector_sum_split(x: torch.Tensor) -> torch.Tensor:
    """
    1D vector sum using two-phase split reduction.

    Phase 1: Each tile computes a partial sum and stores it in a temp buffer.
    Phase 2: Sum the temp buffer (happens on host/CPU after kernel).

    This demonstrates the split reduction pattern without atomics.

    Args:
        x: Input tensor of shape [n]

    Returns:
        Temp buffer of partial sums (will be summed by caller)
    """
    n = x.size(0)
    block_size = hl.register_block_size(n)
    num_blocks = (n + block_size - 1) // block_size

    temp = torch.zeros([num_blocks], dtype=x.dtype, device=x.device)

    for tile in hl.tile(n):
        tile_idx = hl.tile_id(tile)
        temp[tile_idx] = x[tile].sum()

    return temp


# %%
# Standard Single-Pass Kernel
# --------------------------
@helion.kernel()
def vector_sum_standard(x: torch.Tensor) -> torch.Tensor:
    """
    Standard single-pass vector sum using atomics.

    Tiles the vector and uses atomic_add to accumulate.

    Args:
        x: Input tensor of shape [n]

    Returns:
        Scalar tensor containing the sum
    """
    n = x.size(0)
    result = torch.zeros([1], dtype=x.dtype, device=x.device)

    for tile in hl.tile(n):
        hl.atomic_add(result, [0], x[tile].sum())

    return result[0]


# %%
# Testing Function
# -------------
def test(n: int, dtype: torch.dtype = torch.float32) -> None:
    """
    Test the split reduction kernel and compare with standard approach.

    Args:
        n: Vector length
        dtype: Data type for the tensors
    """
    x = torch.randn([n], device=DEVICE, dtype=dtype)

    print("Performance:")
    run_example(
        vector_sum_standard,
        {
            "split": lambda x: vector_sum_split(x).sum(),
            "pytorch": lambda x: x.sum(),
        },
        (x,),
    )


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the split reduction test.
    Tests with a 1M element vector using float32.
    """
    test(1_000_000)


if __name__ == "__main__":
    main()
