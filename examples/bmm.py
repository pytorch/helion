"""
Batch Matrix Multiplication Example
===================================

This example demonstrates how to implement a batch matrix multiplication kernel using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from packaging import version
import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Batch Matrix Multiplication Kernel
# ----------------------------------
# static_shapes=True gives a performance boost for matmuls


# %%
@helion.kernel(static_shapes=True)
def bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs batch matrix multiplication.

    Args:
        A: Input tensor of shape [B, M, K]
        B: Input tensor of shape [B, K, N]

    Returns:
        Output tensor of shape [B, M, N] containing the result of batch matrix multiplication
    """
    # A: [B, M, K], B: [B, K, N], Out: [B, M, N]   # dense bmm
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty(
        [b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype)
    )
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc
    return out


# %%
# Verification Function
# ---------------------


# %%
def check(b: int, m: int, k: int, n: int) -> None:
    """
    Verify the bmm kernel implementation against PyTorch's native bmm function.

    Args:
        b: Batch size
        m: First dimension of the first matrix
        k: Second dimension of the first matrix / First dimension of the second matrix
        n: Second dimension of the second matrix
    """
    x = torch.randn([b, m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([b, k, n], device=DEVICE, dtype=torch.float16)
    run_example(bmm, torch.bmm, (x, y))


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the bmm kernel verification with specific parameters.
    Tests with batch size 16, and matrices of dimensions 512x768 and 768x1024.
    Ensures torch version is at least 2.8 for 16-bit tensor support in baddbmm.
    """
    # torch.baddbmm support for 16-bit tensors requires torch 2.8+
    assert version.parse(torch.__version__.split("+")[0]) >= version.parse("2.8"), (
        "Requires torch 2.8+"
    )
    check(16, 512, 768, 1024)


if __name__ == "__main__":
    main()
