"""
BF16 x INT16 GEMM with Helion
============================================================
The kernel performs matrix multiplication where one matrix is in bfloat16 format and the other is in int16 format.
The int16 values are converted to bfloat16 before performing the matrix multiplication.
"""

# %%
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


# %%
@helion.kernel(static_shapes=True)
def _bf16xint16_gemm(x: Tensor, w: Tensor) -> Tensor:
    """
    x is bf16, w is int16.
    """
    M, K = x.shape
    K2, N = w.shape
    assert K == K2, f"size mismatch {K} != {K2}"

    out = torch.empty([M, N], dtype=torch.bfloat16, device=x.device)

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            x_tile = x[tile_m, tile_k]
            w_tile = w[tile_k, tile_n].to(torch.bfloat16)
            acc = hl.dot(x_tile, w_tile, acc=acc)
        out[tile_m, tile_n] = acc.to(torch.bfloat16)

    return out


# %%
@helion.kernel(static_shapes=True)
def _int16xbf16_gemm(x: Tensor, w: Tensor) -> Tensor:
    """
    x is int16, w is bf16.
    """
    M, K = x.shape
    K2, N = w.shape
    assert K == K2, f"size mismatch {K} != {K2}"

    out = torch.empty([M, N], dtype=torch.bfloat16, device=x.device)

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            x_tile = x[tile_m, tile_k].to(torch.bfloat16)
            w_tile = w[tile_k, tile_n]
            acc = hl.dot(x_tile, w_tile, acc=acc)
        out[tile_m, tile_n] = acc.to(torch.bfloat16)

    return out


# %%
def bf16xint16_gemm(x: Tensor, w: Tensor, transpose: bool = False) -> Tensor:
    """
    This function dispatches to the appropriate kernel based on the transpose flag.

    Args:
        x (Tensor): Input tensor.
        w (Tensor): Weight tensor.
        transpose (bool): If True, assumes x is int16 and w is bf16. Default: False.

    Returns:
        Tensor: Output tensor in bfloat16 format.
    """
    if transpose:
        return _int16xbf16_gemm(x, w)
    return _bf16xint16_gemm(x, w)


# %%
def bf16xint16_gemm_tritonbench(
    tb_op: object, x: torch.Tensor, w: torch.Tensor
) -> Callable[[], torch.Tensor]:
    """
    Wrapper for TritonBench compatibility.

    Args:
        tb_op: TritonBench operator instance
        x (torch.Tensor): Input tensor in bfloat16 format.
        w (torch.Tensor): Weight tensor in int16 format.

    Returns:
        Callable that returns output tensor in bfloat16 format.
    """
    # Check if transpose mode based on tritonbench operator
    transpose = getattr(tb_op, "transpose", False)

    def run_kernel() -> torch.Tensor:
        return bf16xint16_gemm(x, w, transpose=transpose)

    return run_kernel


# %%
def reference_bf16xint16_pytorch(
    x: torch.Tensor, w: torch.Tensor, transpose: bool = False
) -> torch.Tensor:
    """
    Reference implementation using PyTorch operations.

    Args:
        x (torch.Tensor): Input tensor.
        w (torch.Tensor): Weight tensor.
        transpose (bool): Transpose mode flag.

    Returns:
        torch.Tensor: Output tensor in bfloat16 format.
    """
    if transpose:
        x_bf16 = x.to(torch.bfloat16)
        return torch.matmul(x_bf16, w)
    w_bf16 = w.to(torch.bfloat16)
    return torch.matmul(x, w_bf16)


# %%
def check(m: int, k: int, n: int) -> None:
    """
    Test the bf16 x int16 GEMM implementation against the PyTorch reference.

    Args:
        m (int): Number of rows.
        k (int): Shared dimension.
        n (int): Number of cols.
    """
    x = torch.randn([m, k], device=DEVICE, dtype=torch.bfloat16)
    w = torch.randint(-(2**15), 2**15 - 1, (k, n), device=DEVICE, dtype=torch.int16)
    run_example(
        bf16xint16_gemm,
        reference_bf16xint16_pytorch,
        (x, w, False),
        rtol=1e-2,
        atol=1e-2,
    )

    x_int16 = torch.randint(
        -(2**15), 2**15 - 1, (m, k), device=DEVICE, dtype=torch.int16
    )
    w_bf16 = torch.randn([k, n], device=DEVICE, dtype=torch.bfloat16)
    run_example(
        bf16xint16_gemm,
        reference_bf16xint16_pytorch,
        (x_int16, w_bf16, True),
        rtol=1e-2,
        atol=1e-2,
    )


# %%
def main() -> None:
    """
    Main entry point that runs the bf16xint16 kernel verification with different tensor sizes.
    """
    check(256, 256, 256)
    check(512, 512, 512)
    check(65536, 1024, 1280)


# %%
if __name__ == "__main__":
    main()
