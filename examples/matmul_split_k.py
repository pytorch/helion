"""
Split-K Matrix Multiplication Example
================================

This example demonstrates how to implement a matrix multiplication kernel using the split-K
algorithm for better parallelism in Helion.
"""

# %%
# Imports
# -------
from __future__ import annotations

import torch

import helion
from helion._testing import run_example
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl


# %%
# Split-K Matrix Multiplication Kernel
# --------------------------------
# static_shapes=True gives a performance boost for matmuls
@helion.kernel(static_shapes=True)
def matmul_split_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication using split-K algorithm for better parallelism.

    Split-K divides the reduction dimension (K) into multiple chunks that can be processed
    in parallel, with results atomically accumulated at the end.

    Args:
        x: First input tensor of shape [M, K]
        y: Second input tensor of shape [K, N]

    Returns:
        Output tensor of shape [M, N] containing the result of matrix multiplication
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.zeros(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
    k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    for tile_m, tile_n, outer_k in hl.tile([m, n, k], block_size=[None, None, k_block]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for inner_k in hl.tile(outer_k.begin, outer_k.end):
            acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
        hl.atomic_add(out, [tile_m, tile_n], acc)
    return out


# %%
# Verification Function
# -------------------
def check(m: int, k: int, n: int) -> None:
    """
    Verify the split-K matmul kernel implementation against PyTorch's native matmul function.

    Args:
        m: First dimension of the first matrix
        k: Second dimension of the first matrix / First dimension of the second matrix
        n: Second dimension of the second matrix
    """
    x = torch.randn([m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, n], device="cuda", dtype=torch.float16)
    run_example(matmul_split_k, torch.matmul, (x, y), atol=1)


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the split-K matmul kernel verification.

    Tests with matrices of shape 64x32768 and 32768x64, which benefits from the split-K approach
    due to the large reduction dimension.
    """
    check(64, 32768, 64)


if __name__ == "__main__":
    main()
