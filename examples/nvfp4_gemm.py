"""
NVFP4 GEMM with Helion
=====================
This example implements a CuTe NVFP4 GEMM for BF16 activations and FP4 E2M1
weights.  The weights use PyTorch's ``torch.float4_e2m1fn_x2`` shell dtype,
which stores two E2M1 values per byte, and E4M3 per-16-value scales in
PyTorch's SWIZZLE_32_4_4 layout.
"""

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


GEMM_CONFIG = helion.Config(
    block_sizes=[16, 16, 8],
    indexing=["pointer"] * 8,
    load_eviction_policies=["last", "last", "last", "last"],
    num_warps=4,
    num_stages=3,
    pid_type="flat",
    range_warp_specializes=[None],
)

# FP4 E2M1 lookup table indexed by 4-bit encoding (0-15).
FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


def swizzled_scale_numel(rows: int, cols: int) -> int:
    return _round_up(rows, 128) * _round_up(cols, 4)


def swizzled_scale_offsets(row: Tensor, col: Tensor, cols: int) -> Tensor:
    num_col_tiles = _ceil_div(cols, 4)
    tile_offset = ((row // 128) * num_col_tiles + col // 4) * 512
    return tile_offset + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4


def swizzle_fp8_scales(scales: Tensor) -> Tensor:
    """Convert logical row-major block scales to PyTorch's SWIZZLE_32_4_4 layout."""
    if scales.dim() == 1:
        logical_scales = scales.reshape(1, scales.shape[0])
    elif scales.dim() == 2:
        logical_scales = scales
    else:
        raise ValueError(f"expected 1D or 2D scales, got {scales.dim()}D")

    rows, cols = logical_scales.shape
    out = torch.zeros(
        swizzled_scale_numel(rows, cols),
        device=logical_scales.device,
        dtype=logical_scales.dtype,
    )
    row = torch.arange(rows, device=logical_scales.device, dtype=torch.int64)[:, None]
    col = torch.arange(cols, device=logical_scales.device, dtype=torch.int64)[None, :]
    out[swizzled_scale_offsets(row, col, cols).reshape(-1)] = logical_scales.reshape(-1)
    return out


def unswizzle_fp8_scales(scales: Tensor, rows: int, cols: int) -> Tensor:
    row = torch.arange(rows, device=scales.device, dtype=torch.int64)[:, None]
    col = torch.arange(cols, device=scales.device, dtype=torch.int64)[None, :]
    return scales.reshape(-1)[swizzled_scale_offsets(row, col, cols)]


def _check_swizzled_scales(
    name: str,
    scales: Tensor,
    rows: int,
    cols: int,
) -> None:
    expected = swizzled_scale_numel(rows, cols)
    if scales.numel() != expected:
        raise ValueError(
            f"{name} must contain {expected} SWIZZLE_32_4_4 scale values "
            f"for logical shape ({rows}, {cols}); got {scales.numel()}"
        )


def _as_fp4x2(tensor: Tensor) -> Tensor:
    if tensor.dtype is torch.float4_e2m1fn_x2:
        return tensor
    if tensor.dtype is torch.uint8:
        return tensor.view(torch.float4_e2m1fn_x2)
    raise TypeError(f"expected uint8 or float4_e2m1fn_x2 tensor, got {tensor.dtype}")


def _fp4_storage(tensor: Tensor) -> Tensor:
    if tensor.dtype is torch.float4_e2m1fn_x2:
        return tensor.view(torch.uint8)
    return tensor


@helion.kernel(static_shapes=True, config=GEMM_CONFIG, backend="cute")
def _nvfp4_matmul_single_pass_kernel(
    a_groups: Tensor,
    b_fp4x2: Tensor,
    weight_scale: Tensor,
    out: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    """Compute ``a_groups @ b_fp4x2`` using generated FP4/FP8 CuTe conversion."""
    M, K_groups, _ = a_groups.shape
    _, _, N = b_fp4x2.shape

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K_groups):
            scale_offsets = swizzled_scale_offsets(
                tile_n.index[None, :],
                tile_k.index[:, None],
                K_groups,
            )
            scale = weight_scale[scale_offsets].to(torch.float32)
            for byte in hl.static_range(8):
                weight_lo, weight_hi = hl.float4_e2m1fn_x2_to_float32(
                    b_fp4x2[tile_k, byte, tile_n]
                )
                a_lo = a_groups[tile_m, tile_k, byte * 2].to(torch.float32)
                a_hi = a_groups[tile_m, tile_k, byte * 2 + 1].to(torch.float32)
                contrib_lo = a_lo.unsqueeze(2) * weight_lo.unsqueeze(0)
                contrib_hi = a_hi.unsqueeze(2) * weight_hi.unsqueeze(0)
                acc = acc + ((contrib_lo + contrib_hi) * scale.unsqueeze(0)).sum(dim=1)
        out[tile_m, tile_n] = (acc * alpha).to(torch.bfloat16)
    return out


def nvfp4_matmul(
    A: Tensor,
    B_packed: Tensor,
    weight_scale: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    """
    Compute ``A @ B_packed`` for BF16 activations and NVFP4 weights.

    Args:
        A: BF16 activation matrix of shape ``[M, K]``.
        B_packed: packed FP4 weight matrix of shape ``[K // 2, N]``.  The tensor
            may be raw ``uint8`` storage or a ``torch.float4_e2m1fn_x2`` view.
        weight_scale: E4M3 scales in SWIZZLE_32_4_4 layout for logical shape
            ``[N, K // 16]``.
        alpha: optional output multiplier.

    Returns:
        BF16 output matrix of shape ``[M, N]``.
    """
    M, K = A.shape
    K_bytes, N = B_packed.shape
    if K % 16 != 0:
        raise ValueError(f"K must be divisible by 16, got {K}")
    if K_bytes * 2 != K:
        raise ValueError(
            f"B_packed shape {tuple(B_packed.shape)} is incompatible with A shape "
            f"{tuple(A.shape)}"
        )
    K_groups = K // 16
    _check_swizzled_scales("weight_scale", weight_scale, N, K_groups)
    b_fp4x2 = _as_fp4x2(B_packed)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    return _nvfp4_matmul_single_pass_kernel(
        A.view(M, K_groups, 16),
        b_fp4x2.view(K_groups, 8, N),
        weight_scale.reshape(-1),
        out,
        alpha,
    )


def _prepare_scaled_mm_inputs(
    A_packed: Tensor,
    B_packed_t: Tensor,
) -> tuple[Tensor, Tensor, int, int, int]:
    A = _as_fp4x2(A_packed)
    B_t = _as_fp4x2(B_packed_t)
    if A.dim() != 2 or B_t.dim() != 2:
        raise ValueError("nvfp4_scaled_matmul expects 2D FP4 matrices")
    M, K_bytes = A.shape
    K_bytes_b, N = B_t.shape
    if K_bytes_b != K_bytes:
        raise ValueError(
            f"B_packed_t shape {tuple(B_t.shape)} is incompatible with "
            f"A_packed shape {tuple(A.shape)}"
        )
    if K_bytes % 8 != 0:
        raise ValueError(f"K must be divisible by 16, got {K_bytes * 2}")
    if A.stride() != (K_bytes, 1):
        A = A.contiguous()
    if B_t.stride() != (1, K_bytes):
        B_t = B_t.T.contiguous().T
    return A, B_t, M, K_bytes * 2, N


def nvfp4_scaled_matmul(
    A_packed: Tensor,
    B_packed_t: Tensor,
    scale_a: Tensor,
    scale_b: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """
    Native Blackwell FP4 x FP4 block-scaled GEMM using ``torch._scaled_mm``.

    ``A_packed`` has shape ``[M, K // 2]``. ``B_packed_t`` is the transposed
    packed RHS with shape ``[K // 2, N]``, matching ``torch._scaled_mm``.
    Scales are E4M3 tensors in PyTorch's SWIZZLE_32_4_4 flat layout for logical
    shapes ``[M, K // 16]`` and ``[N, K // 16]``.
    """
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(
            f"unsupported output dtype for nvfp4_scaled_matmul: {out_dtype}"
        )
    A, B_t, M, K, N = _prepare_scaled_mm_inputs(A_packed, B_packed_t)
    K_groups = K // 16
    _check_swizzled_scales("scale_a", scale_a, M, K_groups)
    _check_swizzled_scales("scale_b", scale_b, N, K_groups)
    if scale_a.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"scale_a must be torch.float8_e4m3fn, got {scale_a.dtype}")
    if scale_b.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"scale_b must be torch.float8_e4m3fn, got {scale_b.dtype}")
    return torch._scaled_mm(
        A,
        B_t,
        scale_a.reshape(-1),
        scale_b.reshape(-1),
        out_dtype=out_dtype,
    )


# %%
def quantize_fp4_e2m1(x: Tensor) -> Tensor:
    """
    Quantize a float tensor to FP4 E2M1 nibble indices (0-15).

    Each value is rounded to the nearest representable FP4 E2M1 value and
    encoded as a 4-bit index: bit 3 = sign, bits 2-0 = magnitude index.
    """
    sign = (x < 0).to(torch.uint8)
    abs_x = x.abs().clamp(max=6.0)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=abs_x.dtype
    )
    mag_idx = torch.bucketize(abs_x, boundaries).to(torch.uint8)
    return mag_idx | (sign << 3)


def pack_fp4(indices: Tensor) -> Tensor:
    """
    Pack pairs of FP4 nibble indices along dim 0 into bytes.

    Element at even index goes into the low nibble, odd index into the high nibble.
    """
    K, N = indices.shape
    assert K % 2 == 0, "K dimension must be even for FP4 packing"
    reshaped = indices.reshape(K // 2, 2, N).permute(1, 0, 2)
    return ((reshaped[0] & 0xF) | (reshaped[1] << 4)).to(torch.uint8)


def pack_fp4_last_dim(indices: Tensor) -> Tensor:
    """
    Pack pairs of FP4 nibble indices along the trailing dimension into bytes.
    """
    M, K = indices.shape
    assert K % 2 == 0, "K dimension must be even for FP4 packing"
    reshaped = indices.reshape(M, K // 2, 2)
    return ((reshaped[:, :, 0] & 0xF) | (reshaped[:, :, 1] << 4)).to(torch.uint8)


def unpack_and_dequantize_fp4(packed: Tensor) -> Tensor:
    """Unpack and dequantize packed FP4 E2M1 values to float32."""
    packed_storage = _fp4_storage(packed)
    lo = (packed_storage & 0xF).to(torch.long)
    hi = ((packed_storage >> 4) & 0xF).to(torch.long)
    lut = FP4_E2M1_LUT.to(device=packed_storage.device)
    lo_f = lut[lo]
    hi_f = lut[hi]
    stacked = torch.stack([lo_f, hi_f], dim=1)
    return stacked.reshape(packed_storage.shape[0] * 2, packed_storage.shape[1])


# %%
def reference_nvfp4_matmul(
    A: Tensor,
    B_packed: Tensor,
    weight_scale: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    """Reference implementation that dequantizes FP4 weights and applies scales."""
    B_dequant = unpack_and_dequantize_fp4(B_packed)
    K, N = B_dequant.shape
    K_groups = K // 16
    group_idx = torch.arange(K, device=A.device) // 16
    col_idx = torch.arange(N, device=A.device)[None, :]
    scale_offsets = swizzled_scale_offsets(col_idx, group_idx[:, None], K_groups)
    scale = weight_scale.reshape(-1)[scale_offsets].to(torch.float32)
    B_dequant = B_dequant * scale
    return (torch.matmul(A.to(torch.float32), B_dequant) * alpha).to(torch.bfloat16)


def reference_nvfp4_scaled_matmul(
    A_packed: Tensor,
    B_packed_t: Tensor,
    scale_a: Tensor,
    scale_b: Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    A, B_t, M, K, N = _prepare_scaled_mm_inputs(A_packed, B_packed_t)
    K_groups = K // 16
    _check_swizzled_scales("scale_a", scale_a, M, K_groups)
    _check_swizzled_scales("scale_b", scale_b, N, K_groups)
    return torch._scaled_mm(
        A,
        B_t,
        scale_a.reshape(-1),
        scale_b.reshape(-1),
        out_dtype=out_dtype,
    )


def make_fp8_scales(shape: tuple[int, ...], device: torch.device) -> Tensor:
    logical_scales = (torch.rand(shape, device=device, dtype=torch.float32) + 0.5).to(
        torch.float8_e4m3fn
    )
    return swizzle_fp8_scales(logical_scales)


def make_random_fp4(shape: tuple[int, int], device: torch.device) -> Tensor:
    """Create random packed FP4 shell tensor with the logical trailing shape."""
    rows, cols = shape
    if cols % 2 != 0:
        raise ValueError(f"FP4 logical trailing dimension must be even, got {cols}")
    storage = torch.randint(0, 2, (rows, cols // 2), device=device, dtype=torch.uint8)
    return storage.view(torch.float4_e2m1fn_x2)


# %%
def nvfp4_gemm_tritonbench(
    tb_op: object, x: torch.Tensor, w: torch.Tensor
) -> Callable[[], torch.Tensor]:
    """Wrapper for TritonBench compatibility."""
    x_2d = x.reshape(-1, x.size(-1))
    w_quantized = quantize_fp4_e2m1(w)
    w_packed = pack_fp4(w_quantized)
    weight_scale = torch.ones(
        swizzled_scale_numel(w_packed.shape[1], x_2d.shape[1] // 16),
        device=w.device,
        dtype=torch.float8_e4m3fn,
    )

    def run_kernel() -> torch.Tensor:
        return nvfp4_matmul(x_2d, w_packed, weight_scale)

    return run_kernel


# %%
def check(m: int, k: int, n: int) -> None:
    """Test the NVFP4 GEMM implementation against the reference."""
    A = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE)
    W = torch.randn(k, n, dtype=torch.bfloat16, device=DEVICE)
    W_quantized = quantize_fp4_e2m1(W)
    W_packed = pack_fp4(W_quantized).view(torch.float4_e2m1fn_x2)
    weight_scale = make_fp8_scales((n, k // 16), DEVICE)

    run_example(
        nvfp4_matmul,
        reference_nvfp4_matmul,
        (A, W_packed, weight_scale),
        rtol=2e-1,
        atol=1.0,
    )
    print(f"Test passed for shapes: M={m}, K={k}, N={n}")


def check_scaled(m: int, k: int, n: int) -> None:
    """Test and benchmark the native FP4 x FP4 block-scaled CuTe path."""
    A = make_random_fp4((m, k), DEVICE)
    B = make_random_fp4((n, k), DEVICE)
    B_t = B.T
    scale_a = make_fp8_scales((m, k // 16), DEVICE)
    scale_b = make_fp8_scales((n, k // 16), DEVICE)

    result = nvfp4_scaled_matmul(A, B_t, scale_a, scale_b)
    expected = reference_nvfp4_scaled_matmul(A, B_t, scale_a, scale_b)
    torch.testing.assert_close(
        result.to(torch.float32),
        expected.to(torch.float32),
        atol=1.0,
        rtol=2e-1,
    )

    from triton.testing import do_bench

    torch.cuda.synchronize()
    fast_ms = do_bench(lambda: nvfp4_scaled_matmul(A, B_t, scale_a, scale_b))
    torch_ms = do_bench(lambda: reference_nvfp4_scaled_matmul(A, B_t, scale_a, scale_b))
    print(f"Native FP4xFP4 passed for shapes: M={m}, K={k}, N={n}")
    print(f"  fast path:   {fast_ms:.4f} ms")
    print(f"  torch ref:   {torch_ms:.4f} ms")


# %%
def main() -> None:
    check(64, 128, 64)
    check(128, 256, 128)
    check_scaled(128, 256, 256)


# %%
if __name__ == "__main__":
    main()
