"""
Block Scaled Matrix Multiplication with Helion
===============================================
This example demonstrates block-scaled matrix multiplication using Helion's
``hl.dot_scaled`` API, inspired by the Triton block scaled matmul tutorial.
Block scaling (microscaling / MX format) uses per-block scale factors to enable
efficient low-precision computation with higher accuracy:

- Tensors are divided into blocks of 32 elements along the K dimension.
- Each block shares one scale factor in e8m0 format (stored as uint8).
- Elements within each block use a compact format (e.g., FP8 e4m3).
- The hardware computes ``element * 2^(scale - 127)`` during the dot product.

This example uses FP8 e4m3 data format with e8m0 block scales. The same
pattern works with other formats supported by ``hl.dot_scaled``: e2m1 (FP4),
e5m2, bf16, and fp16.

This file includes both the Helion implementation and the original Triton
tutorial kernel for side-by-side comparison.  The Triton tutorial kernel uses
TMA tensor descriptors and pre-shuffled 5D scale layouts for maximum
throughput on Blackwell GPUs.

Note: Requires CUDA compute capability >= 10.0 (NVIDIA B200 / Blackwell+).

Reference: https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
"""

# ruff: noqa: ANN001, ANN201
# Triton JIT functions use tl.constexpr, not Python type annotations.

# %%
from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# One scale factor per 32 K-elements (MX format standard)
SCALE_GROUP_SIZE = 32

# Triton tutorial uses: BLOCK_M=128, BLOCK_N=256, BLOCK_K=128 (for FP8),
# num_stages=4, num_warps=4.  Our kernel does not tile K (full K loaded via
# ':' slice, matching the tl.dot_scaled semantics), so only M and N block
# sizes are needed.  We pick BLOCK_M=128, BLOCK_N=256 to match the tutorial.
config = helion.Config(
    block_sizes=[128, 256],
    num_warps=4,
)


# %%
# Helion Implementation
# ---------------------


# %%
@helion.kernel(static_shapes=False, config=config)
def block_scaled_matmul(
    x: Tensor,
    x_scale: Tensor,
    y: Tensor,
    y_scale: Tensor,
) -> Tensor:
    """
    Block-scaled matrix multiplication using microscaling formats.

    Computes ``C = dequant(x, x_scale) @ dequant(y, y_scale)`` using
    hardware-accelerated scaled dot product instructions (``tl.dot_scaled``
    in Triton).

    Args:
        x: Input matrix of shape ``[M, K]`` in float8_e4m3fn format.
        x_scale: Scale factors of shape ``[M, K // 32]`` in uint8 (e8m0 format).
            Each value encodes a power-of-two scale: ``2^(value - 127)``.
        y: Weight matrix of shape ``[K, N]`` in float8_e4m3fn format.
        y_scale: Scale factors of shape ``[N, K // 32]`` in uint8 (e8m0 format).
            Note: indexed by N (not K) in the first dimension.

    Returns:
        Output matrix of shape ``[M, N]`` in float32 format.
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    assert k % SCALE_GROUP_SIZE == 0, f"K ({k}) must be divisible by {SCALE_GROUP_SIZE}"

    out = torch.empty([m, n], dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc = hl.dot_scaled(
            x[tile_m, :],
            x_scale[tile_m, :],
            "e4m3",
            y[:, tile_n],
            y_scale[tile_n, :],
            "e4m3",
            acc=acc,
        )
        out[tile_m, tile_n] = acc

    return out


# %%
# Triton Tutorial Kernel (original)
# ----------------------------------
# The following is the block-scaled matmul kernel from the Triton tutorial,
# included for side-by-side comparison with the Helion implementation above.
# This kernel uses TMA tensor descriptors with pre-shuffled 5D scale layouts
# for maximum throughput on Blackwell GPUs.
#
# Reference: https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
#
# Key differences vs. the Helion version:
# - TMA tensor descriptors (a_desc, b_desc) for hardware-accelerated loads
# - 5D pre-shuffled scale layout with reshape/transpose for contiguous access
# - Explicit K tiling loop with BLOCK_K=128 and software pipelining
# - Manual management of element packing (ELEM_PER_BYTE for FP4 vs FP8)
# - All abstracted away by Helion's hl.dot_scaled API


# %%
@triton.jit
def triton_block_scaled_matmul_kernel(
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    c_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    output_type: tl.constexpr,
    ELEM_PER_BYTE_A: tl.constexpr,
    ELEM_PER_BYTE_B: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    rep_m: tl.constexpr,
    rep_n: tl.constexpr,
    rep_k: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = (
            scale_a.reshape(rep_m, rep_k, 32, 4, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        )
        scale_b = (
            scale_b.reshape(rep_n, rep_k, 32, 4, 4)
            .trans(0, 3, 2, 1, 4)
            .reshape(BLOCK_N, BLOCK_K // VEC_SIZE)
        )

        if MIXED_PREC:
            accumulator = tl.dot_scaled(
                a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator
            )
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(
                a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator
            )
        else:
            accumulator = tl.dot_scaled(
                a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator
            )

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def _pack_scale_for_tma(
    scale_2d: Tensor, outer_dim: int, K: int, VEC_SIZE: int = 32
) -> Tensor:
    """Convert scale from 2D (outer_dim, K // VEC_SIZE) to pre-shuffled 5D for TMA.

    The pre-shuffled layout matches the Triton tutorial's 5D scale format:
    ``(1, outer_dim // 128, K // VEC_SIZE // 4, 2, 256)``.
    """
    num_chunk_outer = outer_dim // 128
    num_chunk_k = K // VEC_SIZE // 4
    split = scale_2d.reshape(num_chunk_outer, 4, 32, num_chunk_k, 4)
    packed = split.permute(0, 3, 2, 1, 4).contiguous()
    return packed.reshape(1, num_chunk_outer, num_chunk_k, 2, 256)


def triton_block_scaled_matmul(
    x: Tensor, x_scale: Tensor, y: Tensor, y_scale: Tensor
) -> Tensor:
    """Wrapper that launches the original Triton tutorial block-scaled matmul kernel.

    Converts from the simple 2D scale layout used by the Helion kernel to the
    pre-shuffled 5D TMA format expected by the tutorial kernel.
    """
    M, K = x.shape
    K2, N = y.shape
    assert K == K2

    # Tutorial config for FP8 e4m3
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 128
    VEC_SIZE = 32
    ELEM_PER_BYTE_A = 1  # FP8 = 1 byte per element
    ELEM_PER_BYTE_B = 1
    NUM_STAGES = 4

    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    # Data TMA descriptors (B must be in (N, K) layout for the tutorial kernel)
    y_t = y.T.contiguous()
    a_desc = TensorDescriptor.from_tensor(x, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
    b_desc = TensorDescriptor.from_tensor(y_t, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])

    # Scale TMA descriptors: convert 2D -> pre-shuffled 5D
    x_scale_5d = _pack_scale_for_tma(x_scale, M, K, VEC_SIZE)
    y_scale_5d = _pack_scale_for_tma(y_scale, N, K, VEC_SIZE)
    a_scale_desc = TensorDescriptor.from_tensor(x_scale_5d, [1, rep_m, rep_k, 2, 256])
    b_scale_desc = TensorDescriptor.from_tensor(y_scale_5d, [1, rep_n, rep_k, 2, 256])

    # Output
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    c_desc = TensorDescriptor.from_tensor(out, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    triton_block_scaled_matmul_kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        M,
        N,
        K,
        0,  # output_type: 0 = float32
        ELEM_PER_BYTE_A,
        ELEM_PER_BYTE_B,
        VEC_SIZE,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        rep_m,
        rep_n,
        rep_k,
        NUM_STAGES,
    )
    return out


# %%
def reference_block_scaled_matmul(
    x: Tensor, x_scale: Tensor, y: Tensor, y_scale: Tensor
) -> Tensor:
    """
    Reference implementation using explicit dequantization and torch.mm.

    Decodes e8m0 scales (``2^(uint8 - 127)``), applies them element-wise
    to the data, and performs standard matrix multiplication in float32.
    """
    # Decode e8m0 scale factors: value -> 2^(value - 127)
    x_scale_f = torch.pow(2.0, x_scale.to(torch.float32) - 127.0)  # [M, K//32]
    y_scale_f = torch.pow(2.0, y_scale.to(torch.float32) - 127.0)  # [N, K//32]

    # Expand scales to match K dimension (one scale per 32 elements)
    x_scale_f = x_scale_f.repeat_interleave(SCALE_GROUP_SIZE, dim=1)  # [M, K]
    y_scale_f = y_scale_f.repeat_interleave(SCALE_GROUP_SIZE, dim=1)  # [N, K]

    # Dequantize: apply scales to data
    x_f = x.to(torch.float32) * x_scale_f  # [M, K]
    y_f = y.to(torch.float32) * y_scale_f.T  # [K, N]

    return x_f @ y_f  # [M, N]


# %%
def check(m: int, k: int, n: int) -> None:
    """
    Test block-scaled matmul against the reference implementation.

    Args:
        m: Number of rows in the left input matrix.
        k: Shared dimension (must be divisible by 32).
        n: Number of columns in the right input matrix.
    """
    # Create random FP8 input tensors (scale by 0.5 to stay within FP8 range)
    x = (torch.randn(m, k, dtype=torch.float32, device=DEVICE) * 0.5).to(
        torch.float8_e4m3fn
    )
    y = (torch.randn(k, n, dtype=torch.float32, device=DEVICE) * 0.5).to(
        torch.float8_e4m3fn
    )

    # Create scale factors: use 127 (= 2^0 = 1.0) for clean correctness check
    x_scale = torch.full(
        (m, k // SCALE_GROUP_SIZE), 127, device=DEVICE, dtype=torch.uint8
    )
    y_scale = torch.full(
        (n, k // SCALE_GROUP_SIZE), 127, device=DEVICE, dtype=torch.uint8
    )

    print("--- Helion block-scaled matmul ---")
    run_example(
        block_scaled_matmul,
        reference_block_scaled_matmul,
        (x, x_scale, y, y_scale),
        atol=0.5,
        rtol=0.1,
    )
    print("\n--- Triton block-scaled matmul ---")
    run_example(
        triton_block_scaled_matmul,
        reference_block_scaled_matmul,
        (x, x_scale, y, y_scale),
        atol=0.5,
        rtol=0.1,
    )


# %%
def main() -> None:
    # hl.dot_scaled requires CUDA compute capability >= 10.0 (Blackwell+)
    if torch.cuda.get_device_capability() < (10, 0):
        print("Skipping: block_scaled_matmul requires compute capability >= 10.0")
        return
    check(256, 256, 256)
    check(512, 256, 512)


# %%
if __name__ == "__main__":
    main()
