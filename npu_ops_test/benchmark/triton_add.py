"""
Pure Triton Ascend Add Kernel
=============================

This file implements a simple element-wise addition kernel using pure Triton
for direct comparison with PyTorch's torch.add on NPU hardware.
"""

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def triton_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise addition.

    Args:
        x_ptr: Pointer to input tensor x
        y_ptr: Pointer to input tensor y
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size for tiling
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition using Triton kernel.

    Args:
        x: First input tensor
        y: Second input tensor

    Returns:
        Output tensor containing x + y
    """
    # Ensure x and y have the same shape
    x, y = torch.broadcast_tensors(x, y)
    output = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    triton_add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def triton_add_2d(x: torch.Tensor, y: torch.Tensor, block_m: int = 128, block_n: int = 128) -> torch.Tensor:
    """
    2D tiled element-wise addition using Triton kernel (optimized for 2D tensors).

    Args:
        x: First input tensor (M, N)
        y: Second input tensor (M, N)
        block_m: Block size for M dimension
        block_n: Block size for N dimension

    Returns:
        Output tensor containing x + y
    """
    x, y = torch.broadcast_tensors(x, y)
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)

    M, N = x.shape
    # Compute grid size
    grid_m = triton.cdiv(M, block_m)
    grid_n = triton.cdiv(N, block_n)
    grid = (grid_m * grid_n,)

    @triton.jit
    def triton_add_2d_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        M,
        N,
        stride_x_m,
        stride_x_n,
        stride_y_m,
        stride_y_n,
        stride_out_m,
        stride_out_n,
        grid_m,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        pid_m = pid // grid_m
        pid_n = pid % grid_m

        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)

        x = tl.load(
            x_ptr + offsets_m[:, None] * stride_x_m + offsets_n[None, :] * stride_x_n,
            mask=mask,
        )
        y = tl.load(
            y_ptr + offsets_m[:, None] * stride_y_m + offsets_n[None, :] * stride_y_n,
            mask=mask,
        )
        output = x + y

        tl.store(
            output_ptr
            + offsets_m[:, None] * stride_out_m
            + offsets_n[None, :] * stride_out_n,
            output,
            mask=mask,
        )

    triton_add_2d_kernel[grid](
        x,
        y,
        output,
        M,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        output.stride(0),
        output.stride(1),
        grid_m,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )

    return output


if __name__ == "__main__":
    # Quick test
    import time

    device = "npu"
    M, N = 1024, 1024
    dtype = torch.bfloat16

    x = torch.randn(M, N, device=device, dtype=dtype)
    y = torch.randn(M, N, device=device, dtype=dtype)

    # Test correctness
    output_triton = triton_add_2d(x, y)
    output_torch = torch.add(x, y)

    print(f"Max error: {torch.max(torch.abs(output_triton - output_torch))}")
    print(f"Correctness check passed: {torch.allclose(output_triton, output_torch, rtol=1e-2, atol=1e-2)}")
