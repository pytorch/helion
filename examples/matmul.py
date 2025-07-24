"""
Matrix Multiplication Example
============================

This example demonstrates how to implement a basic matrix multiplication kernel using Helion.
"""

# %%
# Imports
# -------
from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


# %%
# Matrix Multiplication Kernel
# ---------------------------
# static_shapes=True gives a performance boost for matmuls
@helion.kernel(static_shapes=True)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication between two tensors.

    Args:
        x: First input tensor of shape [M, K]
        y: Second input tensor of shape [K, N]

    Returns:
        Output tensor of shape [M, N] containing the result of matrix multiplication
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


# %%
# Verification Function
# -------------------
def check(m: int, k: int, n: int) -> None:
    """
    Verify the matmul kernel implementation against PyTorch's native matmul function.

    Args:
        m: First dimension of the first matrix
        k: Second dimension of the first matrix / First dimension of the second matrix
        n: Second dimension of the second matrix
    """
    x = torch.randn([m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, n], device="cuda", dtype=torch.float16)
    run_example(matmul, torch.matmul, (x, y))


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the matmul kernel verification with 1024x1024 matrices.
    """
    check(1024, 1024, 1024)


if __name__ == "__main__":
    main()
