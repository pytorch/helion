"""
NVFP4 (E2M1) General Matrix Multiplication (GEMM) with Helion
==============================================================
This example demonstrates an NVFP4 GEMM kernel implemented in Helion. The kernel
performs matrix multiplication with BFloat16 activations and FP4 (E2M1 format) quantized
weights. The FP4 weights are packed as two values per byte (int8), dequantized to float32
in the kernel, and accumulated in FP32 for accuracy. The output is in BFloat16 format.

FP4 E2M1 format represents values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
using 1 sign bit, 2 exponent bits, and 1 mantissa bit.
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

# FP4 E2M1 lookup table indexed by 4-bit encoding (0-15)
FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,  # positive (sign=0)
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,  # negative (sign=1)
    ],
    dtype=torch.float32,
)


# %%
@helion.kernel(static_shapes=False)
def nvfp4_matmul(A: Tensor, B_packed: Tensor) -> Tensor:
    """
    BFloat16 x NVFP4 (E2M1) General Matrix Multiplication (GEMM).

    This kernel performs matrix multiplication where:
    - A is a bfloat16 matrix of shape [M, K]
    - B_packed is an int8 matrix of shape [K//2, N] containing packed FP4 E2M1 values
      (two 4-bit values packed into each byte, first element in low nibble)

    The kernel dequantizes FP4 weights to float32 using piecewise arithmetic and
    accumulates in FP32 for accuracy.

    Args:
        A (Tensor): Input tensor of shape [M, K] in bfloat16 format.
        B_packed (Tensor): Packed FP4 tensor of shape [K//2, N] in int8 format.

    Returns:
        Tensor: Output tensor of shape [M, N] in bfloat16 format.
    """
    M, K = A.shape
    _, N = B_packed.shape

    C = torch.zeros(M, N, dtype=torch.bfloat16, device=A.device)
    block_size_k_packed = hl.register_block_size(K // 2)

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        for tile_k_packed in hl.tile(K // 2, block_size=block_size_k_packed):
            # Load bf16 activations for the corresponding unpacked K range
            a_tile_begin = tile_k_packed.begin * 2
            a_tile_len = block_size_k_packed * 2
            a_tile = A[tile_m, a_tile_begin : (a_tile_begin + a_tile_len)].to(
                torch.float32
            )

            # Load packed FP4 data from B
            b_tile = B_packed[tile_k_packed, tile_n]

            # Extract low and high nibbles (each is a 4-bit FP4 E2M1 value)
            b_lo = b_tile & 0xF
            b_hi = (b_tile >> 4) & 0xF

            # Dequantize FP4 E2M1 nibbles to float32
            # Extract sign (bit 3) and unsigned magnitude (bits 2-0)
            sign_lo = ((b_lo >> 3) & 1).to(torch.float32)
            u_lo = (b_lo & 0x7).to(torch.float32)
            sign_hi = ((b_hi >> 3) & 1).to(torch.float32)
            u_hi = (b_hi & 0x7).to(torch.float32)

            # FP4 E2M1 unsigned magnitude to float:
            #   u=0 -> 0.0,  u=1 -> 0.5,  u=2 -> 1.0,  u=3 -> 1.5
            #   u=4 -> 2.0,  u=5 -> 3.0,  u=6 -> 4.0,  u=7 -> 6.0
            # Piecewise formula: u<4: u*0.5 | u<6: u-2.0 | else: u*2.0-8.0
            abs_lo = torch.where(
                u_lo < 4,
                u_lo * 0.5,
                torch.where(u_lo < 6, u_lo - 2.0, u_lo * 2.0 - 8.0),
            )
            abs_hi = torch.where(
                u_hi < 4,
                u_hi * 0.5,
                torch.where(u_hi < 6, u_hi - 2.0, u_hi * 2.0 - 8.0),
            )

            # Apply sign: multiply by +1 or -1
            b_lo_f = abs_lo * (1.0 - 2.0 * sign_lo)
            b_hi_f = abs_hi * (1.0 - 2.0 * sign_hi)

            # Interleave low and high to recover original order
            # Stack: [K_packed, 2, N] -> reshape to [K, N]
            b_stacked = torch.stack([b_lo_f, b_hi_f], dim=1)
            b_unpacked = b_stacked.reshape(
                tile_k_packed.block_size * 2, tile_n.block_size
            )

            # Element-wise matrix multiply: [M, K, 1] * [1, K, N] -> sum over K
            a_tile = a_tile.unsqueeze(2)
            b_unpacked = b_unpacked.unsqueeze(0)
            acc = acc + (a_tile * b_unpacked).sum(dim=1)

        C[tile_m, tile_n] = acc.to(torch.bfloat16)

    return C


# %%
def quantize_fp4_e2m1(x: Tensor) -> Tensor:
    """
    Quantize a float tensor to FP4 E2M1 nibble indices (0-15).

    Each value is rounded to the nearest representable FP4 E2M1 value and
    encoded as a 4-bit index: bit 3 = sign, bits 2-0 = magnitude index.

    Args:
        x (Tensor): Input float tensor.

    Returns:
        Tensor: Tensor of uint8 values in [0, 15], one per element.
    """
    sign = (x < 0).to(torch.uint8)
    abs_x = x.abs().clamp(max=6.0)
    # Boundaries between consecutive positive FP4 E2M1 values (midpoints)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=abs_x.dtype
    )
    mag_idx = torch.bucketize(abs_x, boundaries).to(torch.uint8)
    return mag_idx | (sign << 3)


def pack_fp4(indices: Tensor) -> Tensor:
    """
    Pack pairs of FP4 nibble indices along dim 0 into int8 bytes.

    Element at even index goes into the low nibble, odd index into the high nibble.

    Args:
        indices (Tensor): Tensor of shape [K, N] with uint8 values in [0, 15].

    Returns:
        Tensor: Packed tensor of shape [K//2, N] in int8 format.
    """
    K, N = indices.shape
    assert K % 2 == 0, "K dimension must be even for FP4 packing"
    reshaped = indices.reshape(K // 2, 2, N).permute(1, 0, 2)
    return ((reshaped[0] & 0xF) | (reshaped[1] << 4)).to(torch.int8)


def unpack_and_dequantize_fp4(packed: Tensor) -> Tensor:
    """
    Unpack and dequantize packed FP4 E2M1 values to bfloat16.

    Args:
        packed (Tensor): Packed tensor of shape [K//2, N] in int8 format.

    Returns:
        Tensor: Dequantized tensor of shape [K, N] in bfloat16 format.
    """
    lo = (packed & 0xF).to(torch.long)
    hi = ((packed >> 4) & 0xF).to(torch.long)
    lut = FP4_E2M1_LUT.to(device=packed.device)
    lo_f = lut[lo]
    hi_f = lut[hi]
    stacked = torch.stack([lo_f, hi_f], dim=1)
    return stacked.reshape(packed.shape[0] * 2, packed.shape[1]).to(torch.bfloat16)


# %%
def reference_nvfp4_matmul(A: Tensor, B_packed: Tensor) -> Tensor:
    """
    Reference implementation that dequantizes FP4 weights and performs matmul.

    Args:
        A (Tensor): Input tensor in bfloat16 format.
        B_packed (Tensor): Packed FP4 tensor.

    Returns:
        Tensor: Output tensor in bfloat16 format.
    """
    B_dequant = unpack_and_dequantize_fp4(B_packed)
    return torch.matmul(A, B_dequant)


# %%
def nvfp4_gemm_tritonbench(
    tb_op: object, x: torch.Tensor, w: torch.Tensor
) -> Callable[[], torch.Tensor]:
    """
    Wrapper for TritonBench compatibility.

    Args:
        tb_op: TritonBench operator instance
        x (torch.Tensor): Left input tensor in bfloat16 format.
        w (torch.Tensor): Right input tensor to be quantized to FP4.

    Returns:
        Callable that returns output tensor in bfloat16 format.
    """
    x_2d = x.reshape(-1, x.size(-1))
    w_quantized = quantize_fp4_e2m1(w)
    w_packed = pack_fp4(w_quantized)

    def run_kernel() -> torch.Tensor:
        return nvfp4_matmul(x_2d, w_packed)

    return run_kernel


# %%
def check(m: int, k: int, n: int) -> None:
    """
    Test the NVFP4 GEMM implementation against the reference.

    Args:
        m (int): Number of rows in the left input matrix.
        k (int): Shared dimension (must be even).
        n (int): Number of columns in the right input matrix.
    """
    A = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE)
    # Create weights and quantize to FP4
    W = torch.randn(k, n, dtype=torch.bfloat16, device=DEVICE)
    W_quantized = quantize_fp4_e2m1(W)
    W_packed = pack_fp4(W_quantized)

    run_example(
        nvfp4_matmul,
        reference_nvfp4_matmul,
        (A, W_packed),
        rtol=2e-1,
        atol=1.0,
    )
    print(f"Test passed for shapes: M={m}, K={k}, N={n}")


# %%
def main() -> None:
    """
    Main function to run tests with different matrix sizes.
    """
    check(256, 256, 256)
    check(512, 512, 512)


# %%
if __name__ == "__main__":
    main()
