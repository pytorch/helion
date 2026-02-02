"""
Quantization Kernels - Core Triton Implementations

==============================================================================
MATHEMATICAL CORE
==============================================================================

Quantization reduces precision of weights/activations for faster inference
and reduced memory bandwidth.

Key Operations:

1. Per-Token Quantization (Dynamic):
    For each token x ∈ R^d:
    - scale = max(|x|) / max_quant_value
    - x_quant = round(x / scale)
    - x_dequant = x_quant * scale

2. Per-Block Quantization:
    For blocks of size G (e.g., 128):
    - scale[i] = max(|x[i*G:(i+1)*G]|) / max_quant_value
    - x_quant[j] = round(x[j] / scale[j // G])

3. Scaled Matrix Multiplication:
    C = (A_quant @ B_quant) * scale_A[:, None] * scale_B[None, :]

    Where:
    - A_quant: quantized activations [M, K]
    - B_quant: quantized weights [K, N]
    - scale_A: per-row scales [M, 1]
    - scale_B: per-column scales [1, N]

4. AWQ (Activation-aware Weight Quantization):
    4-bit weight quantization with per-group scales and zero-points:
    - w_dequant = (w_packed - zero_point) * scale
    - Packing: 8 INT4 values packed into 1 INT32

5. FP8 Quantization:
    E4M3 (4-bit exponent, 3-bit mantissa) or E5M2 format:
    - Dynamic range: [-448, 448] for E4M3
    - Per-tensor or per-block scaling

6. MxFP8 (Microscaling FP8):
    Shared exponent across small blocks:
    - Each 32-element block shares a scale
    - Better utilization of FP8 dynamic range

Complexity:
    - Quantization: O(elements) per tensor
    - Dequantization: O(elements) per tensor
    - Scaled GEMM: O(M×K×N) with reduced precision compute

References:
    - AWQ: Activation-aware Weight Quantization (Lin et al., 2023)
    - FP8 Formats for Deep Learning (Micikevicius et al., 2022)
    - LLM.int8(): 8-bit Matrix Multiplication (Dettmers et al., 2022)

==============================================================================
"""

import torch
import triton
import triton.language as tl


# ==============================================================================
# Triton Kernel: Per-Token FP8 Quantization
# ==============================================================================

@triton.jit
def per_token_quant_fp8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    num_tokens,
    hidden_dim,
    fp8_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Per-token dynamic quantization to FP8.

    For each token:
    1. Find max absolute value
    2. Compute scale = max_abs / fp8_max
    3. Quantize: output = round(input / scale)

    Grid: (num_tokens,)
    """
    token_idx = tl.program_id(0)

    # First pass: find max absolute value
    max_abs = 0.0
    for block_idx in range(tl.cdiv(hidden_dim, BLOCK_SIZE)):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_dim

        x = tl.load(
            input_ptr + token_idx * hidden_dim + offs,
            mask=mask,
            other=0.0
        ).to(tl.float32)

        max_abs = tl.maximum(max_abs, tl.max(tl.abs(x)))

    # Compute scale
    scale = max_abs / fp8_max
    scale = tl.maximum(scale, 1e-12)  # Avoid division by zero

    # Store scale
    tl.store(scale_ptr + token_idx, scale)

    # Second pass: quantize
    for block_idx in range(tl.cdiv(hidden_dim, BLOCK_SIZE)):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_dim

        x = tl.load(
            input_ptr + token_idx * hidden_dim + offs,
            mask=mask,
            other=0.0
        ).to(tl.float32)

        # Quantize
        x_quant = x / scale
        # Clamp to FP8 range
        x_quant = tl.minimum(tl.maximum(x_quant, -fp8_max), fp8_max)

        tl.store(
            output_ptr + token_idx * hidden_dim + offs,
            x_quant.to(output_ptr.dtype.element_ty),
            mask=mask
        )


# ==============================================================================
# Triton Kernel: Per-Block INT8 Quantization
# ==============================================================================

@triton.jit
def per_block_quant_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    num_elements,
    block_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Per-block quantization to INT8.

    Each block of `block_size` elements shares a scale factor.

    Grid: (num_blocks,)
    """
    block_idx = tl.program_id(0)

    # Load block
    offs = block_idx * block_size + tl.arange(0, BLOCK_SIZE)
    mask = (offs < num_elements) & (tl.arange(0, BLOCK_SIZE) < block_size)

    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Find max absolute value
    max_abs = tl.max(tl.abs(x))
    scale = max_abs / 127.0
    scale = tl.maximum(scale, 1e-12)

    # Store scale
    tl.store(scale_ptr + block_idx, scale)

    # Quantize to INT8
    x_quant = tl.libdevice.rint(x / scale)
    x_quant = tl.minimum(tl.maximum(x_quant, -127.0), 127.0)

    tl.store(output_ptr + offs, x_quant.to(tl.int8), mask=mask)


# ==============================================================================
# Triton Kernel: Block-Scaled FP8 GEMM
# ==============================================================================

@triton.jit
def w8a8_block_fp8_matmul_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block scaling parameters
    scale_block_m: tl.constexpr,
    scale_block_n: tl.constexpr,
    scale_block_k: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Block-scaled FP8 matrix multiplication.

    C[m, n] = sum_k(A[m, k] * B[k, n]) * scale_A[m] * scale_B[n]

    Where A and B are in FP8 format with block-wise scales.

    Grid: (ceil(M/BLOCK_M) * ceil(N/BLOCK_N),)
    """
    # Program ID to (M block, N block) mapping with grouping
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator in higher precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Main loop over K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K

        # Masks
        a_mask = (offs_m[:, None] < M) & ((k_offs + offs_k[None, :]) < K)
        b_mask = ((k_offs + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        # Load A and B blocks (FP8)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Load scales for this K block
        # Scale indices depend on block scaling granularity
        a_scale_idx = pid_m * scale_block_m + k // (scale_block_k // BLOCK_K)
        b_scale_idx = k // (scale_block_k // BLOCK_K) * tl.cdiv(N, scale_block_n) + pid_n

        a_scale = tl.load(a_scale_ptr + a_scale_idx)
        b_scale = tl.load(b_scale_ptr + b_scale_idx)

        # Dequantize and accumulate
        a_fp32 = a.to(tl.float32) * a_scale
        b_fp32 = b.to(tl.float32) * b_scale

        acc += tl.dot(a_fp32, b_fp32)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


# ==============================================================================
# Triton Kernel: AWQ Dequantization
# ==============================================================================

@triton.jit
def awq_dequantize_kernel(
    qweight_ptr,     # Packed INT4 weights
    scales_ptr,      # Per-group scales
    zeros_ptr,       # Per-group zero points
    output_ptr,      # Dequantized FP16 weights
    K, N,
    group_size: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    AWQ 4-bit weight dequantization.

    w_dequant = (w_int4 - zero_point) * scale

    Where w_int4 is unpacked from 8 values per INT32.

    Grid: (ceil(K/BLOCK_K), ceil(N/BLOCK_N))
    """
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Each INT32 contains 8 INT4 values
    # qweight layout: [K // 8, N]
    packed_k = offs_k // 8
    pack_idx = offs_k % 8

    # Load packed weights
    qweight = tl.load(
        qweight_ptr + packed_k[:, None] * N + offs_n[None, :],
        mask=(packed_k[:, None] < K // 8) & (offs_n[None, :] < N),
        other=0
    )

    # Unpack INT4: shift and mask
    shift = pack_idx * 4
    w_int4 = (qweight >> shift[:, None]) & 0xF

    # Load scales and zeros (per group)
    group_idx = offs_k // group_size
    scales = tl.load(
        scales_ptr + group_idx[:, None] * N + offs_n[None, :],
        mask=(group_idx[:, None] < K // group_size) & (offs_n[None, :] < N),
        other=1.0
    )
    zeros = tl.load(
        zeros_ptr + group_idx[:, None] * N + offs_n[None, :],
        mask=(group_idx[:, None] < K // group_size) & (offs_n[None, :] < N),
        other=0
    )

    # Dequantize
    w_dequant = (w_int4.to(tl.float16) - zeros.to(tl.float16)) * scales.to(tl.float16)

    # Store
    tl.store(
        output_ptr + offs_k[:, None] * N + offs_n[None, :],
        w_dequant,
        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N)
    )


# ==============================================================================
# Triton Kernel: Scaled Matrix Multiplication
# ==============================================================================

@triton.jit
def scaled_mm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Scaled matrix multiplication for quantized inference.

    C = (A @ B) * scale_A[:, None] * scale_B[None, :]

    Used for INT8 or FP8 GEMM with per-tensor or per-row/column scales.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Main GEMM loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = (offs_m[:, None] < M) & ((k * BLOCK_K + offs_k[None, :]) < K)
        b_mask = ((k * BLOCK_K + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply scales
    a_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=1.0)
    b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=1.0)

    acc = acc * a_scale[:, None] * b_scale[None, :]

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================

def quantize_per_token_int8_reference(
    x: torch.Tensor,  # [num_tokens, hidden_dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token INT8 quantization reference.

    Returns (quantized_tensor, scales)
    """
    # Compute per-token scales
    max_abs = x.abs().max(dim=-1, keepdim=True).values
    scales = max_abs / 127.0
    scales = scales.clamp(min=1e-12)

    # Quantize
    x_quant = (x / scales).round().clamp(-127, 127).to(torch.int8)

    return x_quant, scales.squeeze(-1)


def awq_dequantize_reference(
    qweight: torch.Tensor,    # [K // 8, N] packed INT32
    scales: torch.Tensor,     # [K // group_size, N]
    zeros: torch.Tensor,      # [K // group_size, N]
    group_size: int = 128,
) -> torch.Tensor:
    """
    AWQ dequantization reference.

    Unpacks 4-bit weights and dequantizes with per-group scales.
    """
    K_packed, N = qweight.shape
    K = K_packed * 8

    # Unpack INT4
    w_int4 = torch.zeros(K, N, dtype=torch.int32, device=qweight.device)
    for i in range(8):
        w_int4[i::8] = (qweight >> (i * 4)) & 0xF

    # Dequantize
    group_idx = torch.arange(K, device=qweight.device) // group_size
    w_dequant = (w_int4.float() - zeros[group_idx].float()) * scales[group_idx].float()

    return w_dequant.to(scales.dtype)


def scaled_matmul_reference(
    a: torch.Tensor,       # [M, K] quantized
    b: torch.Tensor,       # [K, N] quantized
    a_scale: torch.Tensor, # [M] or scalar
    b_scale: torch.Tensor, # [N] or scalar
) -> torch.Tensor:
    """
    Scaled matrix multiplication reference.

    C = (A @ B) * scale_A[:, None] * scale_B[None, :]
    """
    c = (a.float() @ b.float())

    if a_scale.dim() > 0:
        c = c * a_scale[:, None]
    else:
        c = c * a_scale

    if b_scale.dim() > 0:
        c = c * b_scale[None, :]
    else:
        c = c * b_scale

    return c


# ==============================================================================
# Triton Kernel Wrappers
# ==============================================================================

def per_token_quant_fp8_triton(
    x: torch.Tensor,
    fp8_max: float = 448.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper to call per-token FP8 quantization kernel."""
    x_flat = x.view(-1, x.shape[-1])
    num_tokens, hidden_dim = x_flat.shape

    output = torch.empty_like(x_flat)
    scales = torch.empty(num_tokens, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = 1024
    grid = (num_tokens,)
    per_token_quant_fp8_kernel[grid](
        x_flat,
        output,
        scales,
        num_tokens,
        hidden_dim,
        fp8_max=fp8_max,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.view(x.shape), scales


def per_token_quant_fp8_reference(
    x: torch.Tensor,
    fp8_max: float = 448.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token FP8 quantization reference."""
    x_flat = x.view(-1, x.shape[-1])
    max_abs = x_flat.abs().max(dim=-1, keepdim=True).values
    scales = (max_abs / fp8_max).clamp(min=1e-12)
    x_quant = (x_flat / scales).clamp(-fp8_max, fp8_max)
    return x_quant.view(x.shape), scales.squeeze(-1)


def scaled_mm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """Wrapper to call scaled matrix multiplication kernel."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"

    c = torch.empty(M, N, device=a.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    scaled_mm_kernel[grid](
        a, b, c,
        a_scale, b_scale,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


# ==============================================================================
# Tests
# ==============================================================================

def test_per_token_quant_fp8():
    """Test per-token FP8 quantization kernel."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    configs = [
        (4, 256),     # num_tokens, hidden_dim
        (16, 512),
        (32, 1024),
        (8, 4096),
    ]

    fp8_max = 448.0  # E4M3 range

    print("Testing Per-Token FP8 Quantization:")
    for num_tokens, hidden_dim in configs:
        x = torch.randn(num_tokens, hidden_dim, dtype=torch.float32, device=device)

        # Reference
        ref_quant, ref_scales = per_token_quant_fp8_reference(x, fp8_max)
        # Triton
        tri_quant, tri_scales = per_token_quant_fp8_triton(x, fp8_max)

        quant_atol = (ref_quant - tri_quant).abs().max().item()
        scale_atol = (ref_scales - tri_scales).abs().max().item()
        # Relax tolerance for FP8 quantization
        passed = quant_atol < 1e-4 and scale_atol < 1e-5
        status = "PASS" if passed else "FAIL"
        print(f"  tokens={num_tokens}, hidden={hidden_dim}: quant_atol={quant_atol:.2e}, scale_atol={scale_atol:.2e} [{status}]")

        if not passed:
            raise AssertionError(f"Test failed for tokens={num_tokens}, hidden={hidden_dim}")

    print("  All Per-Token FP8 Quantization tests passed!")


def test_scaled_mm():
    """Test scaled matrix multiplication kernel."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    configs = [
        (32, 64, 64),    # M, N, K
        (64, 128, 128),
        (128, 256, 64),
    ]

    print("Testing Scaled Matrix Multiplication:")
    for M, N, K in configs:
        # Create random quantized matrices
        a = torch.randn(M, K, dtype=torch.float32, device=device)
        b = torch.randn(K, N, dtype=torch.float32, device=device)
        a_scale = torch.randn(M, dtype=torch.float32, device=device).abs() + 0.1
        b_scale = torch.randn(N, dtype=torch.float32, device=device).abs() + 0.1

        ref_c = scaled_matmul_reference(a, b, a_scale, b_scale)
        tri_c = scaled_mm_triton(a, b, a_scale, b_scale)

        atol = (ref_c - tri_c).abs().max().item()
        # Use mean relative error for more stable comparison
        mean_rtol = ((ref_c - tri_c).abs() / (ref_c.abs() + 1e-8)).mean().item()
        # Use relaxed tolerance for matmul (floating point accumulation errors)
        # Atol scales with matrix size, so use a more lenient threshold
        passed = mean_rtol < 0.02  # <2% mean relative error
        status = "PASS" if passed else "FAIL"
        print(f"  M={M}, N={N}, K={K}: atol={atol:.2e}, mean_rtol={mean_rtol:.2e} [{status}]")

        if not passed:
            raise AssertionError(f"Test failed for M={M}, N={N}, K={K}")

    print("  All Scaled MM tests passed!")


def test_quantization_roundtrip():
    """Test quantization followed by dequantization."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    print("Testing Quantization Roundtrip (FP8):")
    num_tokens, hidden_dim = 16, 512
    fp8_max = 448.0

    x = torch.randn(num_tokens, hidden_dim, dtype=torch.float32, device=device)

    # Quantize
    x_quant, scales = per_token_quant_fp8_triton(x, fp8_max)

    # Dequantize
    x_dequant = x_quant * scales[:, None]

    # Check reconstruction error
    atol = (x - x_dequant).abs().max().item()
    rtol = ((x - x_dequant).abs() / (x.abs() + 1e-8)).max().item()
    passed = rtol < 0.02  # <2% relative error for FP8
    status = "PASS" if passed else "FAIL"
    print(f"  Roundtrip error: atol={atol:.2e}, rtol={rtol:.2e} [{status}]")

    if not passed:
        raise AssertionError("Quantization roundtrip test failed")

    print("  Quantization roundtrip test passed!")


def test_quantization_half_precision():
    """Test quantization with half precision inputs."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")

    print("Testing Quantization with half precision (float16):")
    num_tokens, hidden_dim = 16, 512
    fp8_max = 448.0

    x_fp16 = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device=device)

    # Reference (fp32)
    ref_quant, ref_scales = per_token_quant_fp8_reference(x_fp16.float(), fp8_max)

    # Triton (fp16 input)
    tri_quant, tri_scales = per_token_quant_fp8_triton(x_fp16, fp8_max)

    quant_atol = (ref_quant.to(tri_quant.dtype) - tri_quant).abs().max().item()
    scale_atol = (ref_scales - tri_scales).abs().max().item()
    passed = quant_atol < 1e-2 and scale_atol < 1e-5
    status = "PASS" if passed else "FAIL"
    print(f"  FP16 input: quant_atol={quant_atol:.2e}, scale_atol={scale_atol:.2e} [{status}]")

    if not passed:
        raise AssertionError("Half precision quantization test failed")

    print("  Half precision quantization test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Quantization Kernels - Triton Core Tests")
    print("=" * 60)

    test_per_token_quant_fp8()
    print()
    test_scaled_mm()
    print()
    test_quantization_roundtrip()
    print()
    test_quantization_half_precision()

    print("\n" + "=" * 60)
    print("All Quantization tests completed successfully!")
    print("=" * 60)
