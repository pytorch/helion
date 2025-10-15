"""
INT4 General Matrix Multiplication (GEMM) with Helion
=====================================================
This example demonstrates an INT4 GEMM kernel implemented in Helion. The kernel performs
matrix multiplication where the second matrix B is packed with two 4-bit values per byte.
The kernel unpacks the int4 values, converts to bfloat16, and performs matmul with
the bfloat16 matrix A.
"""

# %%
# Imports
# -------

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
# INT4 GEMM Kernel
# ----------------


# %%
@helion.kernel(static_shapes=False)
def matmul_bf16_int4(A: Tensor, B: Tensor) -> Tensor:
    """
    BFloat16 x INT4 General Matrix Multiplication (GEMM).

    This kernel performs matrix multiplication where:
    - A is a bfloat16 matrix of shape [M, K]
    - B is an int8 matrix of shape [K//2, N] containing packed int4 values
      (two 4-bit values packed into each int8)

    Args:
        A (Tensor): Input tensor of shape [M, K] in bfloat16 format.
        B (Tensor): Packed int4 tensor of shape [K//2, N] in int8 format.

    Returns:
        Tensor: Output tensor of shape [M, N] in bfloat16 format.
    """
    M, K = A.shape
    _, N = B.shape

    C = torch.zeros(M, N, dtype=torch.bfloat16, device=A.device)
    block_size_k_packed = hl.register_block_size(K // 2)

    # Use Helion to tile the computation
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        for tile_k_packed in hl.tile(K // 2, block_size=block_size_k_packed):
            # Load corresponding tiles from A (need to load twice the packed tile size)
            # We need to map tile_k_packed to the corresponding range in A
            a_tile_begin = tile_k_packed.begin * 2
            a_tile_len = block_size_k_packed * 2
            a_tile = A[tile_m, a_tile_begin : (a_tile_begin + a_tile_len)].to(
                torch.float32
            )  # [BLOCK_SIZE_M, BLOCK_SIZE_K]

            # Load packed int8 data from B
            b_tile = B[tile_k_packed, tile_n]  # [BLOCK_SIZE_K//2, BLOCK_SIZE_N]

            # Extract low and high 4-bit values with sign extension
            # Low nibble: sign-extend from 4-bit to 8-bit using left shift then arithmetic right shift
            b_lo = ((b_tile << 4) >> 4).to(torch.int8)  # Sign-extend low 4 bits
            b_hi = (b_tile >> 4).to(torch.int8)  # Sign-extend high 4 bits

            # Stack and reshape to interleave low and high bits
            # Stack along a new dimension to get [BLOCK_SIZE_K//2, 2, BLOCK_SIZE_N]
            b_stacked = torch.stack([b_lo, b_hi], dim=1)

            # Reshape to interleave: [BLOCK_SIZE_K//2, 2, BLOCK_SIZE_N] -> [BLOCK_SIZE_K, BLOCK_SIZE_N]
            # This will place elements in the order: b_lo[0], b_hi[0], b_lo[1], b_hi[1], ...
            b_unpacked = b_stacked.reshape(
                tile_k_packed.block_size * 2, tile_n.block_size
            ).to(torch.float32)

            a_tile = a_tile.unsqueeze(2)  # [BLOCK_SIZE_M, BLOCK_SIZE_K, 1]
            b_unpacked = b_unpacked.unsqueeze(0)
            acc = acc + (a_tile * b_unpacked).sum(dim=1)  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

        C[tile_m, tile_n] = acc.to(torch.bfloat16)

    return C


# %%
# TritonBench Wrapper
# -------------------


# %%
def int4_gemm_tritonbench(tb_op: object, x: torch.Tensor, w: torch.Tensor) -> Callable:
    """
    Wrapper for TritonBench compatibility.

    Args:
        tb_op: TritonBench operator instance
        x (torch.Tensor): Left input tensor in bfloat16 format.
        w (torch.Tensor): Right input tensor of shape [K, N] containing int4 values.
                          Will be packed to int4 format.

    Returns:
        Callable: A function that performs the int4 gemm.
    """

    # Pack w to int4 format (two 4-bit values per int8 byte)
    x_2d = x.reshape(-1, x.size(-1))
    w_int8 = w.to(torch.int8)
    w_reshaped = w_int8.reshape(w.shape[0] // 2, 2, w.shape[1]).permute(1, 0, 2)
    w_packed = ((w_reshaped[0] & 0xF) | (w_reshaped[1] << 4)).to(torch.int8)

    def run_kernel() -> torch.Tensor:
        return matmul_bf16_int4(x_2d, w_packed)

    return run_kernel


# %%
# Verification Function
# ---------------------


# %%
def _pack_int4_matrix(unpacked: torch.Tensor) -> torch.Tensor:
    """
    Pack int4 matrix into int8 container with two values per byte.

    Args:
        unpacked (torch.Tensor): Tensor of shape [K, N] with values in [-8, 7].

    Returns:
        torch.Tensor: Packed tensor of shape [K//2, N] in int8 format.
    """
    k, n = unpacked.shape
    assert k % 2 == 0, "K dimension must be even for int4 packing"
    reshaped = unpacked.reshape(k // 2, 2, n).permute(1, 0, 2)
    return ((reshaped[0] & 0xF) | (reshaped[1] << 4)).to(torch.int8)


def _unpack_int4_matrix(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack an int4 matrix stored as two 4-bit values per int8 byte.

    Args:
        packed (torch.Tensor): Packed tensor of shape [K//2, N] in int8 format.

    Returns:
        torch.Tensor: Unpacked tensor of shape [K, N] in int8 format.
    """
    b_lo = ((packed << 4) >> 4).to(torch.int8)
    b_hi = (packed >> 4).to(torch.int8)
    stacked = torch.stack([b_lo, b_hi], dim=1)
    return stacked.reshape(packed.shape[0] * 2, packed.shape[1])


def reference_matmul_bf16_int4(A: Tensor, B_packed: Tensor) -> Tensor:
    """
    Reference implementation that unpacks the int4 weights and performs matmul.

    Args:
        A (Tensor): Input tensor in bfloat16 format.
        B_packed (Tensor): Packed int4 tensor.

    Returns:
        Tensor: Output tensor in bfloat16 format.
    """
    B_unpacked = _unpack_int4_matrix(B_packed).to(torch.bfloat16)
    return torch.matmul(A, B_unpacked)


def check(m: int, k: int, n: int) -> None:
    """
    Test the INT4 GEMM implementation using the run_example utility.

    Args:
        m (int): Number of rows in the left input matrix.
        k (int): Shared dimension (must be even).
        n (int): Number of columns in the right input matrix.
    """
    A = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE)
    B_unpacked = torch.randint(-8, 8, (k, n), dtype=torch.int8, device=DEVICE)
    B_packed = _pack_int4_matrix(B_unpacked)
    run_example(
        matmul_bf16_int4,
        reference_matmul_bf16_int4,
        (A, B_packed),
        rtol=2e-1,
        atol=1.0,
    )
    print(f"Test passed for shapes: M={m}, K={k}, N={n}")


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main function to run tests with different matrix sizes.
    """
    check(4, 8192, 7168)
    check(8192, 8192, 8192)


# %%
# Run Example
# -----------

# %%
if __name__ == "__main__":
    main()
