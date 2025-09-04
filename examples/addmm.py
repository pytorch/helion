"""
Helion Addmm Kernel Example
============================
This example demonstrates a Helion kernel implementation of matrix multiplication followed by an addition
It includes correctness checks against the PyTorch baseline and integration with tritonbench.
"""

# %%
# Imports
# -------
from __future__ import annotations

import torch
from torch import Tensor

import helion
from helion._testing import run_example
import helion.language as hl


# %%
# Addmm Kernel
# --------------
@helion.kernel()
def addmm(a: Tensor, mat1: Tensor, mat2: Tensor) -> Tensor:
    """
    Performs a matrix multiplication of the matrices `mat1` and `mat2` and adds `a` to the result.
    Args:
        a (Tensor): Input tensor of shape [m, k].
        mat1 (Tensor): Input tensor of shape [k, n].
        mat2 (Tensor): Input tensor of shape [m, n].
    Returns:
        Tensor: Resulting tensor of shape [m, n].
    """
    m, k = mat1.size()
    k2, n = mat2.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(mat1.dtype, mat2.dtype), device=mat1.device
    )
    a = torch.broadcast_to(a, (m, n))
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                mat1[tile_m, tile_k],
                mat2[tile_k, tile_n],
            )
        out[tile_m, tile_n] = acc + a[tile_m, tile_n]
    return out


# %%
# Verification Function
# -------------------
def check(m: int, n: int, k: int) -> None:
    """
    Verify the add kernel implementation against PyTorch's native addmm function.

    Args:
        m (int): Number of rows in matrix x.
        n (int): Number of columns in matrix y.
        k (int): Number of columns in matrix x and rows in matrix y.
    """
    a = torch.randn([m], device="cuda", dtype=torch.float16)
    mat1 = torch.randn([m, k], device="cuda", dtype=torch.float16)
    mat2 = torch.randn([k, n], device="cuda", dtype=torch.float16)
    run_example(addmm, torch.addmm, (a, mat1, mat2))


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the add kernel verification with a 1024x512 tensor and a 512x1024 tensor.
    """
    check(1024, 1024, 512)


if __name__ == "__main__":
    main()
