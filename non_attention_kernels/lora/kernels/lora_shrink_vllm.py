# SPDX-License-Identifier: Apache-2.0
"""
LoRA shrink kernel (A matrix multiplication).

Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""
import torch

import triton
import triton.language as tl


@triton.jit
def _lora_shrink_kernel(
    input_ptr,
    lora_a_ptr,
    out_ptr,
    M,  # num_tokens
    N,  # lora_rank
    K,  # hidden_dim
    scale,
    input_stride_m,
    input_stride_k,
    lora_stride_n,
    lora_stride_k,
    out_stride_m,
    out_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    LoRA shrink kernel: output = scale * (input @ lora_A.T)

    - input: [M, K] where M=num_tokens, K=hidden_dim
    - lora_A: [N, K] where N=lora_rank, K=hidden_dim
    - output: [M, N]

    Computes: output[m, n] = scale * sum_k(input[m, k] * lora_A[n, k])
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k

        # Load input block [BLOCK_M, BLOCK_K]
        input_ptrs = input_ptr + offs_m[:, None] * input_stride_m + k_offs[None, :] * input_stride_k
        input_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(input_ptrs, mask=input_mask, other=0.0)

        # Load lora_A block [BLOCK_N, BLOCK_K] -> transpose to [BLOCK_K, BLOCK_N]
        lora_ptrs = lora_a_ptr + offs_n[None, :] * lora_stride_n + k_offs[:, None] * lora_stride_k
        lora_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        b = tl.load(lora_ptrs, mask=lora_mask, other=0.0)

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)

    # Apply scale
    acc = acc * scale

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def lora_shrink(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    LoRA shrink operation using Triton kernel: output = x @ lora_a.T * scale

    Args:
        x: Input tensor [num_tokens, hidden_dim]
        lora_a: LoRA A matrix [lora_rank, hidden_dim]
        scale: Scaling factor (default 1.0)

    Returns:
        Tensor [num_tokens, lora_rank]
    """
    M, K = x.shape  # num_tokens, hidden_dim
    N = lora_a.shape[0]  # lora_rank

    output = torch.empty(M, N, device=x.device, dtype=x.dtype)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _lora_shrink_kernel[grid](
        x, lora_a, output,
        M, N, K,
        scale,
        x.stride(0), x.stride(1),
        lora_a.stride(0), lora_a.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output
