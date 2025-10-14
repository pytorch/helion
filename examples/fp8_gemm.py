"""
FP8 General Matrix Multiplication (GEMM) with Helion
====================================================
This example demonstrates an FP8 GEMM kernel implemented in Helion. The kernel performs
matrix multiplication on FP8 inputs, accumulating results in FP32 for accuracy, and
outputs in FP16 format. It includes a reference PyTorch implementation using
torch._scaled_mm for correctness comparison, and a test function to validate the kernel.
"""

# %%
from __future__ import annotations

import os
from typing import Callable

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# Override default config to work around Triton tl.dot requirement:
# `AssertionError: Input shapes should have M >= 16, N >= 16 and K >= 32`
config = None
if os.environ.get("HELION_AUTOTUNE_EFFORT") == "none":
    config = helion.Config(block_sizes=[32, 32, 32])


# %%
@helion.kernel(static_shapes=True, config=config)
def fp8_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    FP8 General Matrix Multiplication (GEMM).
    This kernel demonstrates FP8 computation in Helion.
    When lowered to Triton, the tl.dot operation will handle
    FP8 inputs natively and accumulate to FP32.
    Args:
        x (torch.Tensor): Input tensor of shape [m, k] in FP8 format.
        y (torch.Tensor): Input tensor of shape [k, n] in FP8 format.
    Returns:
        torch.Tensor: Output tensor of shape [m, n] in FP16 format.
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    # Output is in FP16 to match tritonbench behavior
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        # Accumulate in FP32 for accuracy
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            # Load FP8 tiles directly - no conversion needed
            x_tile = x[tile_m, tile_k]
            y_tile = y[tile_k, tile_n]
            # Use hl.dot for FP8 GEMM
            acc = hl.dot(x_tile, y_tile, acc=acc)
        out[tile_m, tile_n] = acc.to(torch.float16)
    return out


# %%
def reference_fp8_gemm_pytorch(
    x_fp8: torch.Tensor, y_fp8: torch.Tensor
) -> torch.Tensor:
    """
    Reference implementation using torch._scaled_mm.
    Args:
        x_fp8 (torch.Tensor): Input tensor in FP8 format.
        y_fp8 (torch.Tensor): Input tensor in FP8 format.
    Returns:
        torch.Tensor: Output tensor in FP16 format.
    """
    # torch._scaled_mm requires column-major for second operand
    y_fp8_t = y_fp8.T.contiguous().T
    scale_a = torch.tensor(1.0, device=x_fp8.device)
    scale_b = torch.tensor(1.0, device=x_fp8.device)
    return torch._scaled_mm(
        x_fp8, y_fp8_t, scale_a, scale_b, use_fast_accum=False, out_dtype=torch.float16
    )


# %%
def fp8_gemm_tritonbench(
    tb_op: object,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    """
    Wrapper for TritonBench compatibility.
    Args:
        tb_op: TritonBench operator instance
        a (torch.Tensor): Left input tensor in FP8 format.
        b (torch.Tensor): Right input tensor in FP8 format.
        scale_a (torch.Tensor): Scale factor for tensor a (unused in our implementation).
        scale_b (torch.Tensor): Scale factor for tensor b (unused in our implementation).
    Returns:
        Callable that returns output tensor in FP16 format.
    """
    return lambda: fp8_gemm(a, b)


# %%
def check(m: int, k: int, n: int) -> None:
    """
    Test the FP8 GEMM implementation against the PyTorch reference.
    Args:
        m (int): Number of rows in the left input matrix.
        k (int): Shared dimension.
        n (int): Number of columns in the right input matrix.
    """
    # Create FP8 tensors
    x = torch.randn([m, k], device=DEVICE, dtype=torch.float32)
    y = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
    # Convert to FP8 format (e4m3fn is commonly used for forward pass)
    x_fp8 = x.to(torch.float8_e4m3fn)
    y_fp8 = y.to(torch.float8_e4m3fn)
    run_example(fp8_gemm, reference_fp8_gemm_pytorch, (x_fp8, y_fp8))


# %%
def main() -> None:
    """
    Main function to run tests with different matrix sizes.
    """
    check(256, 256, 256)
    check(512, 512, 512)
    check(1024, 1024, 1024)


# %%
if __name__ == "__main__":
    main()
