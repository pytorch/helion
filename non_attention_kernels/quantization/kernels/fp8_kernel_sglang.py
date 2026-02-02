# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0
# Source: sglang/srt/layers/quantization/fp8_kernel.py
"""
FP8 quantization kernels from SGLang.

Supports:
- Per-token quantization
- Per-token-group quantization
- Static quantization with pre-computed scales
- Block-wise FP8 matmul
- MXFP8 block-scaled matmul

FP8 (8-bit floating point) quantization provides significant memory savings
while maintaining reasonable accuracy for LLM inference.
"""
from typing import List, Optional, Tuple

import torch

import triton
import triton.language as tl


# FP8 dtype configuration
fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max


@triton.jit
def _per_token_group_quant_8bit(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for float8
    bit8_min,
    bit8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """
    Triton kernel for per-token-group FP8 quantization.

    Converts tensor values into float8 values with per-group scales.

    Grid: (num_groups,) where num_groups = numel // group_size
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_8bit_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    bit8_min,
    bit8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    """
    Triton kernel for per-token-group FP8 quantization with column-major scales.

    Used when scales need to be stored in column-major order for TMA alignment.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id.to(tl.int64) * group_size
    y_q_ptr += g_id.to(tl.int64) * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    if SCALE_UE8M0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.abs(y_s))))
    y_q = tl.clamp(y / y_s, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _static_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_s_repeat_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
    REPEAT_SCALE: tl.constexpr,
):
    """
    Static FP8 quantization kernel using pre-computed scale.

    Grid: (M,) where M = num_rows
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    if REPEAT_SCALE:
        y_s_repeat_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y_s = tl.load(y_s_ptr).to(tl.float32)
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    if REPEAT_SCALE:
        tl.store(y_s_repeat_ptr, y_s)


@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
):
    """
    Block-wise FP8 matmul kernel (W8A8).

    Performs matrix multiplication with block-wise quantization scales.
    A (activation) and B (weight) are both FP8, with scales As and Bs.

    Grid: (num_pid_m * num_pid_n,)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    scale_step_k = BLOCK_SIZE_K // group_k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _per_tensor_quant_mla_fp8_stage1(
    x_ptr,
    x_s_ptr,
    head_size,
    x_stride_h,
    x_stride_s,
    eps,
    fp8_max,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 1 of per-tensor MLA FP8 quantization: compute max absolute value.

    Grid: (num_seq, num_head)
    """
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < head_size

    x_ptr += head_id * x_stride_h + seq_id * x_stride_s
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(x)), eps)

    tl.atomic_max(x_s_ptr, _absmax / fp8_max)


@triton.jit
def _per_tensor_quant_mla_fp8_stage2(
    x_ptr,
    x_s_ptr,
    x_q_ptr,
    num_seq,
    head_size,
    x_stride_h,
    x_stride_s,
    fp8_min,
    fp8_max,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 2 of per-tensor MLA FP8 quantization: apply quantization.

    Grid: (num_seq, num_head)
    """
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < head_size

    x_s = tl.load(x_s_ptr)
    x_s_inv = 1.0 / x_s

    x_ptr += head_id * x_stride_h + seq_id * x_stride_s
    x_q_ptr += head_id * num_seq * head_size + seq_id * head_size

    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    x_q = tl.clamp(x * x_s_inv, fp8_min, fp8_max).to(x_q_ptr.dtype.element_ty)
    tl.store(x_q_ptr + offset, x_q, mask=mask)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = fp8_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform per-token-group FP8 quantization on an input tensor.

    Args:
        x: Input tensor with ndim >= 2
        group_size: The group size for quantization
        eps: Minimum value to avoid division by zero
        dtype: Output dtype (default FP8)

    Returns:
        Tuple of (quantized tensor, scale tensor)
    """
    assert x.shape[-1] % group_size == 0, "Last dim must be divisible by group_size"
    assert x.is_contiguous(), "`x` must be contiguous"

    info = torch.finfo(dtype)
    bit8_max = info.max
    bit8_min = info.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    M = x.numel() // group_size
    N = group_size

    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    _per_token_group_quant_8bit[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        bit8_min=bit8_min,
        bit8_max=bit8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s


def static_quant_fp8(
    x: torch.Tensor,
    x_s: torch.Tensor,
    repeat_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform static FP8 quantization using pre-computed scale.

    Args:
        x: Input tensor with ndim >= 2
        x_s: Pre-computed quantization scale (scalar)
        repeat_scale: Whether to broadcast per-tensor scale to per-channel

    Returns:
        Tuple of (quantized tensor, scale tensor)
    """
    assert x.is_contiguous(), "`x` must be contiguous"
    assert x_s.numel() == 1, "Only supports per-tensor scale"

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]

    if repeat_scale:
        x_s_repeat = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    else:
        x_s_repeat = None

    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    _static_quant_fp8[(M,)](
        x,
        x_q,
        x_s,
        x_s_repeat,
        N,
        N,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        BLOCK=BLOCK,
        REPEAT_SCALE=repeat_scale,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    x_s = x_s_repeat if repeat_scale else x_s
    return x_q, x_s


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Block-wise FP8 matrix multiplication (W8A8).

    Args:
        A: Input activation tensor (FP8)
        B: Weight tensor (FP8)
        As: Per-token-group scale for A
        Bs: Per-block scale for B
        block_size: Block size [block_n, block_k]
        output_dtype: Output data type

    Returns:
        Result tensor of matmul
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]
    assert A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]

    M = A.numel() // A.shape[-1]

    assert B.ndim == 2
    assert B.is_contiguous()
    assert Bs.ndim == 2
    N, K = B.shape

    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    # Default config
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": block_size[0],
        "BLOCK_SIZE_K": block_size[1],
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 3,
    }

    needs_masking = bool(K % config["BLOCK_SIZE_K"] != 0)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_block_fp8_matmul[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
        needs_masking=needs_masking,
    )

    return C
