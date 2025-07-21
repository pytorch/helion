"""Custom Triton kernel for FP8 E5M2 matrix multiplication.

Why we need it:
Several kernels in the benchmark uses FP8 E5M2 format for FP8 GEMM inputs.
However, the current PyTorch API for FP8 GEMM (`torch._scaled_mm`, backed by cuBLAS)
doesn't support E5M2 format.
To work around this limitation, we implement a custom FP8 GEMM Triton kernel that can handle E5M2 inputs,
and patch torch.matmul to use this kernel when the inputs are E5M2.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def e5m2_matmul_kernel(
    # Pointers to matrices
    a_ptr,  # noqa: ANN001
    b_ptr,  # noqa: ANN001
    c_ptr,  # noqa: ANN001
    # Matrix dimensions
    M: int,
    N: int,
    K: int,
    # Strides
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    """Triton kernel for FP8 E5M2 matrix multiplication.

    Computes C = A @ B where A and B are in FP8 E5M2 format.
    The output C is in FP32 format for accuracy.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    _ = tl.cdiv(M, BLOCK_SIZE_M)  # num_pid_m not used
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator with zeros (FP32)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop
    for _k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        # Load next block of A and B, with bounds checking
        a = tl.load(
            a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K), other=0.0
        )
        b = tl.load(
            b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0
        )

        # Perform the matrix multiplication with tl.dot
        # tl.dot natively handles FP8 inputs and accumulates to FP32
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k += BLOCK_SIZE_K

    # Write the output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    tl.store(c_ptrs, accumulator, mask=mask)


def e5m2_matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper function to call the Triton kernel for E5M2 matmul.

    Args:
        a: First matrix in FP8 E5M2 format [M, K]
        b: Second matrix in FP8 E5M2 format [K, N]

    Returns:
        Result matrix in FP32 format [M, N]
    """
    # Check inputs
    assert a.dtype == torch.float8_e5m2, f"Expected a to be float8_e5m2, got {a.dtype}"
    assert b.dtype == torch.float8_e5m2, f"Expected b to be float8_e5m2, got {b.dtype}"
    assert a.dim() == 2 and b.dim() == 2, "Only 2D matrices supported"
    assert a.size(1) == b.size(0), (
        f"Matrix dimensions don't match: {a.size()} @ {b.size()}"
    )

    M, K = a.shape
    _, N = b.shape

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Grid configuration
    def grid(META: dict[str, int]) -> tuple[int, ...]:
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    # Launch kernel
    e5m2_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=tl.constexpr(64),
        BLOCK_SIZE_N=tl.constexpr(64),
        BLOCK_SIZE_K=tl.constexpr(32),
    )

    return c


# Register as a torch custom op for torch.compile compatibility
@torch.library.custom_op("helion::e5m2_matmul", mutates_args=())
def e5m2_matmul_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Custom op for E5M2 matrix multiplication."""
    return e5m2_matmul_triton(a, b)


@e5m2_matmul_op.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fake tensor implementation for torch.compile."""
    assert a.size(-1) == b.size(0)
    return torch.empty(a.shape[:-1] + b.shape[1:], device=a.device, dtype=torch.float32)
