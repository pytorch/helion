"""
Fused INT4 Dequant + GEMM + SiLU Activation
=============================================
Combines INT4 weight dequantization, matrix multiplication, and SiLU
activation into a single kernel pass. This eliminates:
- Separate dequantization kernel
- Separate GEMM kernel
- Separate SiLU activation kernel

Targets the gate_proj path in LLaMA FFN: SiLU(X @ W_gate_int4)
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

# %%
# INT4 Quantization Utilities
# ---------------------------


# %%
def _pack_int4(unpacked: Tensor) -> Tensor:
    """
    Pack INT4 values into INT8 containers (two 4-bit values per byte).

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
# Fused INT4 GEMM + SiLU Kernel
# ------------------------------


# %%
@helion.kernel(static_shapes=True)
def int4_gemm_silu(
    A: Tensor,
    B_packed: Tensor,
    scales: Tensor,
    group_size: int = 128,
) -> Tensor:
    """
    Fused INT4 Dequant + GEMM + SiLU activation.

    Computes: SiLU(A @ dequantize(B_packed, scales))
    In a single kernel pass with SiLU applied in the epilogue
    after full K reduction (no Split-K, since SiLU is nonlinear).

    Args:
        A: Activation tensor [M, K] in float16/bfloat16.
        B_packed: Packed INT4 weights [K//2, N] in int8.
        scales: Per-group scale factors [K//group_size, N] in float16.
        group_size: Quantization group size.

    Returns:
        Output tensor [M, N] with SiLU applied.
    """
    M, K = A.shape
    _, N = B_packed.shape

    out = torch.empty(M, N, dtype=A.dtype, device=A.device)
    block_size_k_packed = hl.register_block_size(K // 2)

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        for inner_k in hl.tile(K // 2, block_size=block_size_k_packed):
            # Load and unpack INT4 weights
            b_packed_tile = B_packed[inner_k, tile_n]
            b_lo = ((b_packed_tile << 4) >> 4).to(torch.int8)
            b_hi = (b_packed_tile >> 4).to(torch.int8)
            b_stacked = torch.stack([b_lo, b_hi], dim=1)
            b_unpacked = b_stacked.reshape(inner_k.block_size * 2, tile_n.block_size)

            # Dequantize: cast to float and apply group scales
            b_fp = b_unpacked.to(A.dtype)
            k_start_unpacked = inner_k.begin * 2
            group_idx = k_start_unpacked // group_size
            scale_tile = scales[group_idx, tile_n]
            b_fp = b_fp * scale_tile.unsqueeze(0)

            # Load activation tile (2x the packed K range)
            a_begin = inner_k.begin * 2
            a_len = inner_k.block_size * 2
            a_tile = A[tile_m, a_begin : (a_begin + a_len)]

            acc = torch.addmm(acc, a_tile, b_fp)

        # SiLU after full K reduction — mathematically correct
        silu_acc = acc * torch.sigmoid(acc)
        out[tile_m, tile_n] = silu_acc.to(out.dtype)

    return out


# %%
# Verification
# ------------


# %%
def check(m: int, k: int, n: int) -> None:
    """Test fused INT4 GEMM + SiLU against reference."""
    A = torch.randn(m, k, dtype=HALF_DTYPE, device=DEVICE)
    W = torch.randn(k, n, dtype=HALF_DTYPE, device=DEVICE)
    group_size = 128

    W_packed, scales = _quantize_symmetric_int4(W, group_size=group_size)

    def reference(A: Tensor, W_packed: Tensor, scales: Tensor, gs: int) -> Tensor:
        W_deq = _dequantize_int4(W_packed, scales, group_size=gs)
        return torch.nn.functional.silu(torch.matmul(A, W_deq))

    run_example(
        lambda a, w, s, g: int4_gemm_silu(a, w, s, g),
        reference,
        (A, W_packed, scales, group_size),
        rtol=2e-1,
        atol=1.0,
    )
    print(f"INT4 GEMM + SiLU: M={m}, K={k}, N={n} PASSED")


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """Run correctness checks."""
    print("Testing INT4 GEMM + SiLU (Fused)...")
    check(64, 4096, 11008)
    check(128, 4096, 11008)


# %%
# Run Example
# -----------

# %%
if __name__ == "__main__":
    main()
