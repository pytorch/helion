"""
Fast NVFP4 GEMV Kernels
=======================
This example implements low-latency NVFP4 GEMV kernels for decode-style
batch-size-1 inference on Blackwell GPUs.  The kernels target weights stored as
packed E2M1 bytes with E4M3 per-16-value scales in PyTorch's SWIZZLE_32_4_4
layout.

Two variants are provided:

* NVFP4 weights with BF16 input.
* NVFP4 weights with NVFP4 input.
"""

# %%
from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl
from helion.runtime.settings import _get_backend

if TYPE_CHECKING:
    from collections.abc import Callable

BackendName = Literal["triton", "cute"]

BF16IN_CUTE_CONFIG = helion.Config(
    block_sizes=[1, 128],
    indexing=["pointer"] * 8,
    load_eviction_policies=["first", "last", "last", "last", "last", "first"],
    num_threads=[1, 128],
    num_warps=4,
    num_stages=1,
    pid_type="flat",
    range_warp_specializes=[None],
)

FP4IN_CUTE_CONFIG = helion.Config(
    block_sizes=[1, 128],
    indexing=["pointer"] * 5,
    load_eviction_policies=["first", "last", "", "last"],
    num_threads=[1, 64],
    num_warps=2,
    num_stages=3,
    pid_type="flat",
    range_warp_specializes=[None],
)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


def swizzled_scale_numel(rows: int, cols: int) -> int:
    return _round_up(rows, 128) * _round_up(cols, 4)


def swizzled_scale_offsets(
    row: Tensor | int,
    col: Tensor | int,
    cols: int,
) -> Tensor | int:
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
    offsets = cast("Tensor", swizzled_scale_offsets(row, col, cols))
    out[offsets.reshape(-1)] = logical_scales.reshape(-1)
    return out


def unswizzle_fp8_scales(scales: Tensor, rows: int, cols: int) -> Tensor:
    row = torch.arange(rows, device=scales.device, dtype=torch.int64)[:, None]
    col = torch.arange(cols, device=scales.device, dtype=torch.int64)[None, :]
    offsets = cast("Tensor", swizzled_scale_offsets(row, col, cols))
    return scales.reshape(-1)[offsets]


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


def _check_contiguous(name: str, tensor: Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_fp4_weight_storage(name: str, storage: Tensor) -> None:
    if storage.dim() != 2:
        raise ValueError(f"{name} must be a 2D packed FP4 tensor")
    if storage.shape[1] % 8 != 0:
        raise ValueError(
            f"{name} K bytes must be divisible by 8; got {storage.shape[1]}"
        )


def _check_numel(name: str, tensor: Tensor, expected: int) -> None:
    if tensor.numel() != expected:
        raise ValueError(f"{name} must contain {expected} values; got {tensor.numel()}")


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


def _e4m3_byte_to_f32(scale_byte: Tensor) -> Tensor:
    """Decode one E4M3 scale byte to fp32 (Triton inline PTX)."""
    return hl.inline_asm_elementwise(
        """
        {
          .reg .b16 sc, scale_lo, scale_hi;
          .reg .b32 scale_h2;
          mov.b32 {sc, _}, $1;
          cvt.rn.f16x2.e4m3x2 scale_h2, sc;
          mov.b32 {scale_lo, scale_hi}, scale_h2;
          cvt.f32.f16 $0, scale_lo;
        }
        """,
        "=f,r",
        [scale_byte],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def _nvfp4_gemv_bf16in_triton_body(
    weight_bytes: Tensor,  # (M, K_bytes) uint8 packed NVFP4 weight
    x_values: Tensor,  # (K_groups, 16) bf16 activation
    weight_scale_bytes: Tensor,  # int8 view of SWIZZLE_32_4_4 E4M3 scales
    out: Tensor,  # (M,) bf16 output
    alpha: float = 1.0,
) -> Tensor:
    """W4A16 NVFP4 GEMV, Triton backend (fp16 decode, fp32 scale).

    Two coalescing wins make this genuine Helion->Triton output competitive:
    ncu showed a naive decode is not DRAM-bound but stalls on uncoalesced loads:

    * Weight: each 16-value group is one contiguous 8-byte chunk, loaded as one
      64-bit word by ``hl.load_float4_e2m1fn_x16_to_float16``. (Reading it as
      two int32 words would make each load stride-2, halving coalescing.)
    * Activation: load a group's 16 bf16 as one contiguous ``(block_g, 16)`` tile,
      then peel each lane with a masked sum (the DSL cannot subscript a loaded
      tile as ``xt[:, j]``) -- the contiguous load is what coalesces.

    The per-group E4M3 scale stays a swizzled gather: SWIZZLE_32_4_4's max
    contiguous run is 4 bytes, so it cannot be coalesced for M-tiled access.
    """
    M = hl.specialize(weight_bytes.size(0))
    K_bytes = hl.specialize(weight_bytes.size(1))
    K_groups = K_bytes // 8
    block_m = hl.register_block_size(1, 16)
    block_g = hl.register_block_size(K_groups)
    f16 = torch.float16
    for tile_m in hl.tile(M, block_size=block_m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_g in hl.tile(K_groups, block_size=block_g):
            group_offsets = tile_m.index[:, None] * K_groups + tile_g.index[None, :]
            weight_mask = (tile_m.index[:, None] < M) & (
                tile_g.index[None, :] < K_groups
            )
            w = hl.load_float4_e2m1fn_x16_to_float16(
                weight_bytes,
                group_offsets,
                extra_mask=weight_mask,
            )
            xt = x_values[tile_g, :].to(f16)  # (block_g, 16) contiguous coalesced
            lane = hl.arange(16)[None, :]
            contrib = hl.zeros([block_m, block_g], dtype=f16)
            for i in hl.static_range(16):
                xi = torch.where(lane == i, xt, 0.0).sum(-1)[None, :]
                contrib = contrib + w[i] * xi
            scale_offsets = swizzled_scale_offsets(
                tile_m.index[:, None], tile_g.index[None, :], K_groups
            )
            scale = _e4m3_byte_to_f32(weight_scale_bytes[scale_offsets])
            acc = acc + (contrib.to(torch.float32) * scale).sum(-1)
        out[tile_m] = (acc * alpha).to(torch.bfloat16)
    return out


def _nvfp4_gemv_bf16in_body(
    weight_fp4x2: Tensor,
    x_values: Tensor,
    weight_scale: Tensor,
    out: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    M, K_groups, _ = weight_fp4x2.shape
    block_m = hl.register_block_size(1, 8)
    block_k = hl.register_block_size(16, K_groups)

    for tile_m in hl.tile(M, block_size=block_m):
        row = tile_m.begin
        acc = hl.zeros([], dtype=torch.float32)
        for tile_k in hl.tile(K_groups, block_size=block_k):
            contrib = hl.zeros([tile_k], dtype=torch.float32)
            for byte in hl.static_range(8):
                weight_lo, weight_hi = hl.float4_e2m1fn_x2_to_float32(
                    weight_fp4x2[row, tile_k, byte]
                )
                contrib = contrib + weight_lo * x_values[tile_k, byte * 2].to(
                    torch.float32
                )
                contrib = contrib + weight_hi * x_values[tile_k, byte * 2 + 1].to(
                    torch.float32
                )
            scale_offsets = swizzled_scale_offsets(
                cast("int", row),
                tile_k.index,
                K_groups,
            )
            scale = hl.load(
                weight_scale,
                [scale_offsets],
                extra_mask=tile_k.index < K_groups,
            ).to(torch.float32)
            acc = acc + (contrib * scale).sum()
        out[row] = (acc * alpha).to(torch.bfloat16)
    return out


def _nvfp4_gemv_fp4in_body(
    weight_fp4x2: Tensor,
    x_fp4x2: Tensor,
    weight_scale: Tensor,
    x_scale: Tensor,
    out: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    M, K_groups, _ = weight_fp4x2.shape
    block_m = hl.register_block_size(1, 8)
    block_k = hl.register_block_size(16, K_groups)

    for tile_m in hl.tile(M, block_size=block_m):
        row = tile_m.begin
        acc = hl.zeros([], dtype=torch.float32)
        for tile_k in hl.tile(K_groups, block_size=block_k):
            contrib = hl.zeros([tile_k], dtype=torch.float32)
            for byte in hl.static_range(8):
                weight_lo, weight_hi = hl.float4_e2m1fn_x2_to_float32(
                    weight_fp4x2[row, tile_k, byte]
                )
                x_lo, x_hi = hl.float4_e2m1fn_x2_to_float32(x_fp4x2[tile_k, byte])
                contrib = contrib + weight_lo * x_lo + weight_hi * x_hi
            weight_scale_offsets = swizzled_scale_offsets(
                cast("int", row),
                tile_k.index,
                K_groups,
            )
            x_scale_offsets = swizzled_scale_offsets(
                tile_k.index * 0,
                tile_k.index,
                K_groups,
            )
            scale = hl.load(
                weight_scale,
                [weight_scale_offsets],
                extra_mask=tile_k.index < K_groups,
            ).to(torch.float32)
            scale = scale * hl.load(
                x_scale,
                [x_scale_offsets],
                extra_mask=tile_k.index < K_groups,
            ).to(torch.float32)
            acc = acc + (contrib * scale).sum()
        out[row] = (acc * alpha).to(torch.bfloat16)
    return out


def _nvfp4_gemv_fp4in_triton_body(
    weight_bytes: Tensor,  # (M, K_bytes) uint8 packed NVFP4 weight
    x_bytes: Tensor,  # (K_bytes,) uint8 packed NVFP4 activation
    weight_scale_bytes: Tensor,  # int8 view of SWIZZLE_32_4_4 E4M3 scales
    x_scale_bytes: Tensor,  # int8 view of SWIZZLE_32_4_4 E4M3 scales
    out: Tensor,  # (M,) bf16 output
    alpha: float = 1.0,
) -> Tensor:
    """W4A4 NVFP4 GEMV, Triton backend."""
    M = hl.specialize(weight_bytes.size(0))
    K_bytes = hl.specialize(weight_bytes.size(1))
    K_groups = K_bytes // 8
    block_m = hl.register_block_size(1, 16)
    block_g = hl.register_block_size(K_groups)

    for tile_m in hl.tile(M, block_size=block_m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_g in hl.tile(K_groups, block_size=block_g):
            group_offsets = tile_m.index[:, None] * K_groups + tile_g.index[None, :]
            group_mask = tile_g.index < K_groups
            weight_mask = (tile_m.index[:, None] < M) & group_mask[None, :]
            w = hl.load_float4_e2m1fn_x16_to_float16(
                weight_bytes,
                group_offsets,
                extra_mask=weight_mask,
            )
            x = hl.load_float4_e2m1fn_x16_to_float16(
                x_bytes,
                tile_g.index,
                extra_mask=group_mask,
            )
            contrib = hl.zeros([block_m, block_g], dtype=torch.float16)
            for i in hl.static_range(16):
                contrib = contrib + w[i] * x[i][None, :]

            weight_scale_offsets = swizzled_scale_offsets(
                tile_m.index[:, None], tile_g.index[None, :], K_groups
            )
            x_scale_offsets = swizzled_scale_offsets(
                tile_g.index * 0,
                tile_g.index,
                K_groups,
            )
            scale = _e4m3_byte_to_f32(weight_scale_bytes[weight_scale_offsets])
            scale = scale * _e4m3_byte_to_f32(x_scale_bytes[x_scale_offsets])[None, :]
            acc = acc + (contrib.to(torch.float32) * scale).sum(-1)
        out[tile_m] = (acc * alpha).to(torch.bfloat16)
    return out


def _nvfp4_scaled_mm_gemv_body(
    out: Tensor,  # (1, M) bf16 output
    x_bytes: Tensor,  # (1, K_bytes) uint8 packed NVFP4 activation
    weight_bytes: Tensor,  # (M, K_bytes) uint8 packed NVFP4 weight
    x_scale: Tensor,  # (128, round_up(K_groups, 4)) E4M3 swizzled scales
    weight_scale: Tensor,  # (round_up(M, 128), round_up(K_groups, 4)) E4M3 scales
    alpha: Tensor,  # (1,) fp32
) -> Tensor:
    """Single-row NVFP4 scaled matmul with the CUTLASS-style in-place signature."""
    M = hl.specialize(weight_bytes.size(0))
    K_bytes = hl.specialize(weight_bytes.size(1))
    K_groups = K_bytes // 8
    x_scale_bytes = x_scale.view(torch.int8).reshape(-1)
    weight_scale_bytes = weight_scale.view(torch.int8).reshape(-1)
    block_m = hl.register_block_size(1, 16)
    block_g = hl.register_block_size(K_groups)

    for tile_m in hl.tile(M, block_size=block_m):
        alpha_value = alpha[0].to(torch.float32)
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_g in hl.tile(K_groups, block_size=block_g):
            group_offsets = tile_m.index[:, None] * K_groups + tile_g.index[None, :]
            group_mask = tile_g.index < K_groups
            weight_mask = (tile_m.index[:, None] < M) & group_mask[None, :]
            w = hl.load_float4_e2m1fn_x16_to_float16(
                weight_bytes,
                group_offsets,
                extra_mask=weight_mask,
            )
            x = hl.load_float4_e2m1fn_x16_to_float16(
                x_bytes,
                tile_g.index,
                extra_mask=group_mask,
            )
            contrib = hl.zeros([block_m, block_g], dtype=torch.float16)
            for i in hl.static_range(16):
                contrib = contrib + w[i] * x[i][None, :]

            weight_scale_offsets = swizzled_scale_offsets(
                tile_m.index[:, None], tile_g.index[None, :], K_groups
            )
            x_scale_offsets = swizzled_scale_offsets(
                tile_g.index * 0,
                tile_g.index,
                K_groups,
            )
            scale = _e4m3_byte_to_f32(weight_scale_bytes[weight_scale_offsets])
            scale = scale * _e4m3_byte_to_f32(x_scale_bytes[x_scale_offsets])[None, :]
            acc = acc + (contrib.to(torch.float32) * scale).sum(-1)
        out[0, tile_m] = (acc * alpha_value).to(torch.bfloat16)
    return out


# Triton W4A16 uses the coalesced-load body with block_m=16 and block_g=128.
# Triton W4A4 uses the autotuned pretuned config from nvfp4_gemv. The CuTe
# backend remains a Helion DSL fallback for coverage rather than a hand-written
# CuTe fast path.
BF16IN_TRITON_CONFIG = helion.Config(block_sizes=[16, 128], num_warps=4, num_stages=3)
FP4IN_TRITON_CONFIG = helion.Config(
    block_sizes=[4, 256],
    num_warps=2,
    num_stages=3,
    range_multi_buffers=[None, True],
)


def _selected_backend(backend: BackendName | None) -> BackendName:
    selected = _get_backend() if backend is None else backend
    if selected not in ("triton", "cute"):
        raise ValueError(
            f"nvfp4_gemv supports backend='triton' or 'cute', got {selected!r}"
        )
    return selected


@functools.cache
def _nvfp4_gemv_bf16in_triton_kernel() -> helion.Kernel[Tensor]:
    """Coalesced-load W4A16 GEMV, Triton backend (the triton path)."""
    return helion.kernel(
        _nvfp4_gemv_bf16in_triton_body,
        static_shapes=True,
        config=BF16IN_TRITON_CONFIG,
        backend="triton",
    )


@functools.cache
def _nvfp4_gemv_bf16in_cute_kernel() -> helion.Kernel[Tensor]:
    """Portable f32-decode W4A16 GEMV for Helion's CuTe backend."""
    return helion.kernel(
        _nvfp4_gemv_bf16in_body,
        static_shapes=True,
        config=BF16IN_CUTE_CONFIG,
        backend="cute",
    )


@functools.cache
def _nvfp4_gemv_fp4in_triton_kernel() -> helion.Kernel[Tensor]:
    return helion.kernel(
        _nvfp4_gemv_fp4in_triton_body,
        static_shapes=True,
        config=FP4IN_TRITON_CONFIG,
        backend="triton",
    )


@functools.cache
def _nvfp4_gemv_fp4in_cute_kernel() -> helion.Kernel[Tensor]:
    return helion.kernel(
        _nvfp4_gemv_fp4in_body,
        static_shapes=True,
        config=FP4IN_CUTE_CONFIG,
        backend="cute",
    )


@functools.cache
def nvfp4_scaled_mm_gemv_kernel() -> helion.Kernel[Tensor]:
    """Return the single Helion W4A4 GEMV kernel matching stable CUTLASS inputs."""
    return helion.kernel(
        _nvfp4_scaled_mm_gemv_body,
        static_shapes=True,
        config=FP4IN_TRITON_CONFIG,
        backend="triton",
    )


def nvfp4_gemv_bf16in(
    weight_packed: Tensor,
    x_bf16: Tensor,
    weight_scale: Tensor,
    alpha: float = 1.0,
    *,
    backend: BackendName | None = None,
) -> Tensor:
    """Compute ``weight_packed @ x_bf16`` for NVFP4 weights and BF16 input."""
    backend = _selected_backend(backend)
    _check_contiguous("weight_packed", weight_packed)
    _check_contiguous("x_bf16", x_bf16)
    weight_fp4x2 = _as_fp4x2(weight_packed)
    weight_bytes = weight_fp4x2.view(torch.uint8)
    _check_fp4_weight_storage("weight_packed", weight_bytes)
    _check_numel("x_bf16", x_bf16, weight_bytes.shape[1] * 2)
    scale_cols = weight_bytes.shape[1] // 8
    _check_swizzled_scales(
        "weight_scale",
        weight_scale,
        weight_bytes.shape[0],
        scale_cols,
    )
    out = torch.empty(
        weight_bytes.shape[0], dtype=torch.bfloat16, device=weight_bytes.device
    )
    n_rows, k_bytes = weight_bytes.shape
    groups = k_bytes // 8
    if backend == "triton":
        # Coalesced-load Triton kernel: weight stored as bytes, activation as
        # contiguous (groups, 16) bf16, scales as the raw SWIZZLE_32_4_4 bytes.
        return _nvfp4_gemv_bf16in_triton_kernel()(
            weight_bytes,
            x_bf16.view(groups, 16),
            weight_scale.reshape(-1).view(torch.int8),
            out,
            alpha,
        )
    return _nvfp4_gemv_bf16in_cute_kernel()(
        weight_fp4x2.view(n_rows, groups, 8),
        x_bf16.view(groups, 16),
        weight_scale.reshape(-1),
        out,
        alpha,
    )


def nvfp4_gemv_fp4in(
    weight_packed: Tensor,
    x_packed: Tensor,
    weight_scale: Tensor,
    x_scale: Tensor,
    alpha: float = 1.0,
    *,
    backend: BackendName | None = None,
) -> Tensor:
    """Compute ``weight_packed @ x_packed`` for NVFP4 weights and input."""
    backend = _selected_backend(backend)
    _check_contiguous("weight_packed", weight_packed)
    _check_contiguous("x_packed", x_packed)
    weight_fp4x2 = _as_fp4x2(weight_packed)
    x_fp4x2 = _as_fp4x2(x_packed)
    weight_bytes = weight_fp4x2.view(torch.uint8)
    x_bytes = x_fp4x2.view(torch.uint8)
    _check_fp4_weight_storage("weight_packed", weight_bytes)
    _check_numel("x_packed", x_bytes, weight_bytes.shape[1])
    scale_cols = weight_bytes.shape[1] // 8
    _check_swizzled_scales(
        "weight_scale",
        weight_scale,
        weight_bytes.shape[0],
        scale_cols,
    )
    _check_swizzled_scales("x_scale", x_scale, 1, scale_cols)
    out = torch.empty(
        weight_bytes.shape[0], dtype=torch.bfloat16, device=weight_bytes.device
    )
    if backend == "triton":
        return _nvfp4_gemv_fp4in_triton_kernel()(
            weight_bytes,
            x_bytes,
            weight_scale.reshape(-1).view(torch.int8),
            x_scale.reshape(-1).view(torch.int8),
            out,
            alpha,
        )
    return _nvfp4_gemv_fp4in_cute_kernel()(
        weight_fp4x2.view(weight_bytes.shape[0], weight_bytes.shape[1] // 8, 8),
        x_fp4x2.view(weight_bytes.shape[1] // 8, 8),
        weight_scale.reshape(-1),
        x_scale.reshape(-1),
        out,
        alpha,
    )


def _dequant_e2m1(nibbles: Tensor) -> Tensor:
    sign = ((nibbles >> 3) & 1).to(torch.float32)
    u = (nibbles & 0x7).to(torch.float32)
    abs_val = torch.where(
        u < 4.0,
        u * 0.5,
        torch.where(u < 6.0, u - 2.0, u * 2.0 - 8.0),
    )
    return abs_val * (1.0 - 2.0 * sign)


def reference_nvfp4_gemv_bf16in(
    weight_packed: Tensor,
    x_bf16: Tensor,
    weight_scale: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    weight_storage = _fp4_storage(weight_packed)
    M, K_bytes = weight_storage.shape
    weight = weight_storage.to(torch.int32)
    weight_lo = _dequant_e2m1(weight & 0xF)
    weight_hi = _dequant_e2m1((weight >> 4) & 0xF)
    x = x_bf16.to(torch.float32).view(K_bytes, 2)
    scale_cols = K_bytes // 8
    scale_idx = torch.arange(K_bytes, device=weight_storage.device) // 8
    row_idx = torch.arange(M, device=weight_storage.device)[:, None]
    scale_offsets = swizzled_scale_offsets(row_idx, scale_idx[None, :], scale_cols)
    scale = weight_scale.reshape(-1)[scale_offsets].to(torch.float32)
    result = ((weight_lo * x[:, 0] + weight_hi * x[:, 1]) * scale).sum(-1)
    return (result * alpha).to(torch.bfloat16).reshape(M)


def reference_nvfp4_gemv_fp4in(
    weight_packed: Tensor,
    x_packed: Tensor,
    weight_scale: Tensor,
    x_scale: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    weight_storage = _fp4_storage(weight_packed)
    x_storage = _fp4_storage(x_packed)
    M, K_bytes = weight_storage.shape
    weight = weight_storage.to(torch.int32)
    x = x_storage.to(torch.int32)
    weight_lo = _dequant_e2m1(weight & 0xF)
    weight_hi = _dequant_e2m1((weight >> 4) & 0xF)
    x_lo = _dequant_e2m1(x & 0xF)
    x_hi = _dequant_e2m1((x >> 4) & 0xF)
    scale_cols = K_bytes // 8
    scale_idx = torch.arange(K_bytes, device=weight_storage.device) // 8
    row_idx = torch.arange(M, device=weight_storage.device)[:, None]
    weight_scale_offsets = swizzled_scale_offsets(
        row_idx, scale_idx[None, :], scale_cols
    )
    x_scale_offsets = swizzled_scale_offsets(
        torch.zeros_like(scale_idx),
        scale_idx,
        scale_cols,
    )
    scale = weight_scale.reshape(-1)[weight_scale_offsets].to(torch.float32)
    scale = scale * x_scale.reshape(-1)[x_scale_offsets].to(torch.float32)
    result = ((weight_lo * x_lo + weight_hi * x_hi) * scale).sum(-1)
    return (result * alpha).to(torch.bfloat16).reshape(M)


def make_fp8_scales(shape: tuple[int, ...], device: torch.device) -> Tensor:
    logical_scales = (torch.rand(shape, device=device, dtype=torch.float32) + 0.5).to(
        torch.float8_e4m3fn
    )
    return swizzle_fp8_scales(logical_scales)


def check_bf16in(M: int, K_bytes: int, backend: BackendName) -> None:
    weight = torch.randint(0, 256, (M, K_bytes), dtype=torch.uint8, device=DEVICE).view(
        torch.float4_e2m1fn_x2
    )
    x = torch.randn(K_bytes * 2, dtype=torch.bfloat16, device=DEVICE)
    weight_scale = make_fp8_scales((M, K_bytes // 8), DEVICE)
    run_example(
        functools.partial(nvfp4_gemv_bf16in, backend=backend),
        reference_nvfp4_gemv_bf16in,
        (weight, x, weight_scale),
        kernel_name=f"helion-{backend}",
        atol=4.0,
        rtol=2e-1,
    )


def check_fp4in(M: int, K_bytes: int, backend: BackendName) -> None:
    weight = torch.randint(0, 256, (M, K_bytes), dtype=torch.uint8, device=DEVICE).view(
        torch.float4_e2m1fn_x2
    )
    x = torch.randint(0, 256, (K_bytes,), dtype=torch.uint8, device=DEVICE).view(
        torch.float4_e2m1fn_x2
    )
    weight_scale = make_fp8_scales((M, K_bytes // 8), DEVICE)
    x_scale = make_fp8_scales((K_bytes // 8,), DEVICE)
    run_example(
        functools.partial(nvfp4_gemv_fp4in, backend=backend),
        reference_nvfp4_gemv_fp4in,
        (weight, x, weight_scale, x_scale),
        kernel_name=f"helion-{backend}",
        atol=4.0,
        rtol=2e-1,
    )


def nvfp4_gemv_bf16in_tritonbench(
    tb_op: object,
    weight_packed: Tensor,
    x_bf16: Tensor,
    weight_scale: Tensor,
) -> Callable[[], Tensor]:
    return lambda: nvfp4_gemv_bf16in(
        weight_packed, x_bf16, weight_scale, backend="triton"
    )


def nvfp4_gemv_fp4in_tritonbench(
    tb_op: object,
    weight_packed: Tensor,
    x_packed: Tensor,
    weight_scale: Tensor,
    x_scale: Tensor,
) -> Callable[[], Tensor]:
    return lambda: nvfp4_gemv_fp4in(
        weight_packed, x_packed, weight_scale, x_scale, backend="triton"
    )


def main() -> None:
    for backend in ("triton", "cute"):
        check_bf16in(64, 128, backend)
        check_fp4in(64, 128, backend)


if __name__ == "__main__":
    main()
