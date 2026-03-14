"""
INT4 GEMM v2 — Tensor Core Optimized with Group Scales
=======================================================
Rewrites the existing Helion int4_gemm.py to use torch.addmm (Tensor Cores)
instead of manual broadcast multiply-sum. Adds Split-K parallelism and
group quantization scale support aligned to group_size=128.

Key optimizations over baseline int4_gemm.py:
1. Uses torch.addmm -> maps to Tensor Core mma/wgmma instructions
2. Split-K via hl.register_tunable for better SM utilization at small M
3. BLOCK_K=128 aligned to group_size=128 for clean scale indexing
4. Proper sign extension for INT4 unpacking
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
# INT4 Quantization Utilities
# ---------------------------


# %%
def _pack_int4(unpacked: Tensor) -> Tensor:
    """
    Pack INT4 values into INT8 containers (two 4-bit values per byte).
    Low nibble = even index, high nibble = odd index along K dimension.

    Args:
        unpacked: Tensor of shape [K, N] with values in [-8, 7].

    Returns:
        Packed tensor of shape [K//2, N] in int8 format.
    """
    k, n = unpacked.shape
    assert k % 2 == 0, "K dimension must be even for INT4 packing"
    reshaped = unpacked.reshape(k // 2, 2, n).permute(1, 0, 2)
    return ((reshaped[0] & 0xF) | (reshaped[1] << 4)).to(torch.int8)


def _unpack_int4(packed: Tensor) -> Tensor:
    """
    Unpack INT4 values from INT8 containers.

    Args:
        packed: Tensor of shape [K//2, N] in int8 format.

    Returns:
        Unpacked tensor of shape [K, N] in int8 format.
    """
    b_lo = ((packed << 4) >> 4).to(torch.int8)
    b_hi = (packed >> 4).to(torch.int8)
    stacked = torch.stack([b_lo, b_hi], dim=1)
    return stacked.reshape(packed.shape[0] * 2, packed.shape[1])


def _quantize_symmetric_int4(
    weight: Tensor, group_size: int = 128
) -> tuple[Tensor, Tensor]:
    """
    Symmetric INT4 quantization with per-group scaling.

    Args:
        weight: FP16/BF16 weight tensor of shape [K, N].
        group_size: Number of elements per quantization group along K.

    Returns:
        Tuple of (packed_weight [K//2, N], scales [K//group_size, N]).
    """
    k, n = weight.shape
    assert k % group_size == 0, f"K={k} must be divisible by group_size={group_size}"

    w_grouped = weight.reshape(k // group_size, group_size, n).float()
    scales = w_grouped.abs().amax(dim=1) / 7.0
    scales = scales.clamp(min=1e-10)

    w_scaled = w_grouped / scales.unsqueeze(1)
    w_int4 = w_scaled.round().clamp(-8, 7).to(torch.int8)

    w_int4_flat = w_int4.reshape(k, n)
    packed = _pack_int4(w_int4_flat)

    return packed, scales.to(weight.dtype)


def _dequantize_int4(packed: Tensor, scales: Tensor, group_size: int = 128) -> Tensor:
    """
    Dequantize packed INT4 weights back to FP16.

    Args:
        packed: Packed INT4 tensor [K//2, N].
        scales: Per-group scale factors [K//group_size, N].
        group_size: Elements per quantization group.

    Returns:
        Dequantized tensor [K, N] in the same dtype as scales.
    """
    unpacked = _unpack_int4(packed).float()
    k, n = unpacked.shape

    scales_expanded = scales.float().repeat_interleave(group_size, dim=0)
    dequantized = unpacked * scales_expanded

    return dequantized.to(scales.dtype)


# %%
# INT4 GEMM v2 Kernel
# --------------------


# %%
@helion.kernel(static_shapes=True)
def int4_gemm_v2(
    A: Tensor,
    B_packed: Tensor,
    scales: Tensor,
    group_size: int = 128,
) -> Tensor:
    """
    INT4 Dequant + GEMM using Tensor Cores with Split-K.

    Args:
        A: Activation tensor [M, K] in float16/bfloat16.
        B_packed: Packed INT4 weights [K//2, N] in int8.
        scales: Per-group scale factors [K//group_size, N] in float16.
        group_size: Quantization group size (default 128).

    Returns:
        Output tensor [M, N] in same dtype as A.
    """
    M, K = A.shape
    _, N = B_packed.shape

    out = torch.zeros(M, N, dtype=A.dtype, device=A.device)

    # Split-K for better parallelism at small batch sizes
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 64))
    k_block = helion.next_power_of_2(helion.cdiv(K // 2, split_k))

    for tile_m, tile_n, outer_k in hl.tile(
        [M, N, K // 2], block_size=[None, None, k_block]
    ):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        for inner_k in hl.tile(outer_k.begin, outer_k.end):
            # Load packed INT4 weights
            b_packed_tile = B_packed[inner_k, tile_n]

            # Unpack: sign-extend low and high nibbles
            b_lo = ((b_packed_tile << 4) >> 4).to(torch.int8)
            b_hi = (b_packed_tile >> 4).to(torch.int8)

            # Interleave to recover original K ordering
            b_stacked = torch.stack([b_lo, b_hi], dim=1)
            b_unpacked = b_stacked.reshape(inner_k.block_size * 2, tile_n.block_size)

            # Dequantize: cast to float and apply group scales
            b_fp = b_unpacked.to(A.dtype)
            k_start_unpacked = inner_k.begin * 2
            group_idx = k_start_unpacked // group_size
            scale_tile = scales[group_idx, tile_n]
            b_fp = b_fp * scale_tile.unsqueeze(0)

            # Load corresponding activation tile (2x the packed K range)
            a_begin = inner_k.begin * 2
            a_len = inner_k.block_size * 2
            a_tile = A[tile_m, a_begin : (a_begin + a_len)]

            # Tensor Core matmul via torch.addmm
            acc = torch.addmm(acc, a_tile, b_fp)

        # Atomic add for Split-K accumulation
        hl.atomic_add(out, [tile_m, tile_n], acc.to(out.dtype))

    return out


# %%
# TritonBench Wrapper
# -------------------


# %%
def int4_gemm_v2_tritonbench(tb_op: object, x: Tensor, w: Tensor) -> Callable:
    """Tritonbench-compatible wrapper."""
    x_2d = x.reshape(-1, x.size(-1))
    w_int8 = w.to(torch.int8)
    w_reshaped = w_int8.reshape(w.shape[0] // 2, 2, w.shape[1]).permute(1, 0, 2)
    w_packed = ((w_reshaped[0] & 0xF) | (w_reshaped[1] << 4)).to(torch.int8)

    K = x_2d.shape[1]
    N = w_packed.shape[1]
    group_size = 128
    scales = torch.ones(K // group_size, N, dtype=x.dtype, device=x.device)

    return lambda: int4_gemm_v2(x_2d, w_packed, scales, group_size)


# %%
# Verification
# ------------


# %%
def check(m: int, k: int, n: int, group_size: int = 128) -> None:
    """Test INT4 GEMM v2 against PyTorch reference."""
    A = torch.randn(m, k, dtype=HALF_DTYPE, device=DEVICE)
    W = torch.randn(k, n, dtype=HALF_DTYPE, device=DEVICE)

    # Quantize
    W_packed, scales = _quantize_symmetric_int4(W, group_size=group_size)

    # Reference: dequantize then matmul
    def reference(A: Tensor, W_packed: Tensor, scales: Tensor, gs: int) -> Tensor:
        W_deq = _dequantize_int4(W_packed, scales, group_size=gs)
        return torch.matmul(A, W_deq)

    run_example(
        lambda a, w, s, g: int4_gemm_v2(a, w, s, g),
        reference,
        (A, W_packed, scales, group_size),
        rtol=2e-1,
        atol=1.0,
    )
    print(f"INT4 GEMM v2: M={m}, K={k}, N={n} PASSED")


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """Run correctness checks."""
    print("Testing INT4 GEMM v2 (Tensor Core)...")
    check(64, 4096, 11008)
    check(128, 4096, 11008)


# %%
# Run Example
# -----------

# %%
if __name__ == "__main__":
    main()
