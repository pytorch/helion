# SPDX-License-Identifier: Apache-2.0
"""
LoRA expand kernel (B matrix multiplication).

Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""
import torch

import triton
import triton.language as tl


@triton.jit
def _lora_expand_kernel(
    input_ptr,
    lora_b_ptr,
    out_ptr,
    M,  # num_tokens
    N,  # output_dim
    K,  # lora_rank
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
    LoRA expand kernel: output = scale * (input @ lora_B.T)

    - input: [M, K] where M=num_tokens, K=lora_rank
    - lora_B: [N, K] where N=output_dim, K=lora_rank
    - output: [M, N]

    Computes: output[m, n] = scale * sum_k(input[m, k] * lora_B[n, k])
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

        # Load lora_B block [BLOCK_N, BLOCK_K] -> transpose to [BLOCK_K, BLOCK_N]
        lora_ptrs = lora_b_ptr + offs_n[None, :] * lora_stride_n + k_offs[:, None] * lora_stride_k
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


def lora_expand(
    x: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    LoRA expand operation using Triton kernel: output = x @ lora_b.T * scale

    Args:
        x: Input tensor [num_tokens, lora_rank]
        lora_b: LoRA B matrix [output_dim, lora_rank]
        scale: Scaling factor (default 1.0)

    Returns:
        Tensor [num_tokens, output_dim]
    """
    M, K = x.shape  # num_tokens, lora_rank
    N = lora_b.shape[0]  # output_dim

    output = torch.empty(M, N, device=x.device, dtype=x.dtype)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _lora_expand_kernel[grid](
        x, lora_b, output,
        M, N, K,
        scale,
        x.stride(0), x.stride(1),
        lora_b.stride(0), lora_b.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output
