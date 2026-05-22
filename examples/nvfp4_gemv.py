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

from typing import TYPE_CHECKING
from typing import cast

import cutlass
from cutlass import Int32
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import _normalize_ptr
from cutlass.cute.typing import Float32
from cutlass.cute.typing import Pointer as CutePointer
from cutlass.cute.typing import Tensor as CuteTensor
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op
import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl
from helion.runtime import default_cute_launcher

if TYPE_CHECKING:
    from collections.abc import Callable

    from cutlass._mlir import ir

BF16IN_CONFIG = helion.Config(
    block_sizes=[1, 128],
    indexing=["pointer"] * 8,
    load_eviction_policies=["first", "last", "last", "last", "last", "first"],
    num_threads=[1, 128],
    num_warps=4,
    num_stages=1,
    pid_type="flat",
    range_warp_specializes=[None],
)

FP4IN_CONFIG = helion.Config(
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


if cute is not None and dsl_user_op is not None:

    def _fast_swizzled_scale_offset(
        row: Int32,
        col: Int32,
        scale_cols: int,
    ) -> Int32:
        num_col_tiles = (scale_cols + 3) // 4
        tile = (row >> Int32(7)) * Int32(num_col_tiles) + (col >> Int32(2))
        return (
            tile * Int32(512)
            + (row & Int32(31)) * Int32(16)
            + ((row >> Int32(5)) & Int32(3)) * Int32(4)
            + (col & Int32(3))
        )

    @dsl_user_op
    def _fast_bf16_16bytes_dual_ptr(
        acc: Float32,
        weight_ptr: CutePointer,
        x_ptr: CutePointer,
        scale_ptr: CutePointer,
        *,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> Float32:
        acc_ir = Float32(acc).ir_value(loc=loc, ip=ip)
        weight_ir = _normalize_ptr(weight_ptr, loc=loc, ip=ip)
        x_ir = _normalize_ptr(x_ptr, loc=loc, ip=ip)
        scale_ir = _normalize_ptr(scale_ptr, loc=loc, ip=ip)
        result = llvm.inline_asm(
            T.f32(),
            [acc_ir, weight_ir, x_ir, scale_ir],
            """
            {
              .reg .b8 wb0, wb1, wb2, wb3, wb4, wb5, wb6, wb7;
              .reg .b16 scales, sf0, sf1, b0, b1, h0, h1, lo, hi, sumh;
              .reg .b32 wr0, wr1, wr2, wr3;
              .reg .b32 xr0, xr1, xr2, xr3, xr4, xr5, xr6, xr7;
              .reg .b32 yr0, yr1, yr2, yr3, yr4, yr5, yr6, yr7;
              .reg .b32 w0, w1, w2, w3, w4, w5, w6, w7;
              .reg .b32 z0, z1, z2, z3, z4, z5, z6, z7;
              .reg .b32 x0, x1, x2, x3, x4, x5, x6, x7;
              .reg .b32 y0, y1, y2, y3, y4, y5, y6, y7;
              .reg .b32 prod0, prod1, sf2;
              mov.f32 $0, $1;
              ld.global.L1::no_allocate.v4.u32 {wr0, wr1, wr2, wr3}, [$2];
              ld.global.L1::evict_last.L2::evict_last.v8.u32
                {xr0, xr1, xr2, xr3, xr4, xr5, xr6, xr7}, [$3];
              ld.global.L1::evict_last.L2::evict_last.v8.u32
                {yr0, yr1, yr2, yr3, yr4, yr5, yr6, yr7}, [$3+32];
              ld.global.L1::no_allocate.u16 scales, [$4];
              cvt.rn.f16x2.e4m3x2 sf2, scales;
              mov.b32 {sf0, sf1}, sf2;
              mov.b32 {wb0, wb1, wb2, wb3}, wr0;
              cvt.rn.f16x2.e2m1x2 w0, wb0; cvt.rn.f16x2.e2m1x2 w1, wb1;
              cvt.rn.f16x2.e2m1x2 w2, wb2; cvt.rn.f16x2.e2m1x2 w3, wb3;
              mov.b32 {wb4, wb5, wb6, wb7}, wr1;
              cvt.rn.f16x2.e2m1x2 w4, wb4; cvt.rn.f16x2.e2m1x2 w5, wb5;
              cvt.rn.f16x2.e2m1x2 w6, wb6; cvt.rn.f16x2.e2m1x2 w7, wb7;
              mov.b32 {wb0, wb1, wb2, wb3}, wr2;
              cvt.rn.f16x2.e2m1x2 z0, wb0; cvt.rn.f16x2.e2m1x2 z1, wb1;
              cvt.rn.f16x2.e2m1x2 z2, wb2; cvt.rn.f16x2.e2m1x2 z3, wb3;
              mov.b32 {wb4, wb5, wb6, wb7}, wr3;
              cvt.rn.f16x2.e2m1x2 z4, wb4; cvt.rn.f16x2.e2m1x2 z5, wb5;
              cvt.rn.f16x2.e2m1x2 z6, wb6; cvt.rn.f16x2.e2m1x2 z7, wb7;
              mov.b32 {b0, b1}, xr0; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 x0, {h0, h1};
              mov.b32 {b0, b1}, xr1; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 x1, {h0, h1};
              mov.b32 {b0, b1}, xr2; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 x2, {h0, h1};
              mov.b32 {b0, b1}, xr3; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 x3, {h0, h1};
              mov.b32 {b0, b1}, xr4; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 x4, {h0, h1};
              mov.b32 {b0, b1}, xr5; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 x5, {h0, h1};
              mov.b32 {b0, b1}, xr6; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 x6, {h0, h1};
              mov.b32 {b0, b1}, xr7; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 x7, {h0, h1};
              mov.b32 {b0, b1}, yr0; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 y0, {h0, h1};
              mov.b32 {b0, b1}, yr1; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 y1, {h0, h1};
              mov.b32 {b0, b1}, yr2; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 y2, {h0, h1};
              mov.b32 {b0, b1}, yr3; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 y3, {h0, h1};
              mov.b32 {b0, b1}, yr4; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 y4, {h0, h1};
              mov.b32 {b0, b1}, yr5; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 y5, {h0, h1};
              mov.b32 {b0, b1}, yr6; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 y6, {h0, h1};
              mov.b32 {b0, b1}, yr7; cvt.rn.f16.bf16 h0, b0; cvt.rn.f16.bf16 h1, b1; mov.b32 y7, {h0, h1};
              mul.rn.f16x2 prod0, w0, x0;   mul.rn.f16x2 prod1, z0, y0;
              fma.rn.f16x2 prod0, w1, x1, prod0;   fma.rn.f16x2 prod1, z1, y1, prod1;
              fma.rn.f16x2 prod0, w2, x2, prod0;   fma.rn.f16x2 prod1, z2, y2, prod1;
              fma.rn.f16x2 prod0, w3, x3, prod0;   fma.rn.f16x2 prod1, z3, y3, prod1;
              fma.rn.f16x2 prod0, w4, x4, prod0;   fma.rn.f16x2 prod1, z4, y4, prod1;
              fma.rn.f16x2 prod0, w5, x5, prod0;   fma.rn.f16x2 prod1, z5, y5, prod1;
              fma.rn.f16x2 prod0, w6, x6, prod0;   fma.rn.f16x2 prod1, z6, y6, prod1;
              fma.rn.f16x2 prod0, w7, x7, prod0;   fma.rn.f16x2 prod1, z7, y7, prod1;
              mov.b32 {lo, hi}, prod0; add.rn.f16 sumh, lo, hi;
              fma.rn.f32.f16 $0, sumh, sf0, $0;
              mov.b32 {lo, hi}, prod1; add.rn.f16 sumh, lo, hi;
              fma.rn.f32.f16 $0, sumh, sf1, $0;
            }
            """,
            "=f,f,l,l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return Float32(result)

    @dsl_user_op
    def _fast_fp4_32bytes_ptr(
        acc: Float32,
        weight_ptr: CutePointer,
        x_ptr: CutePointer,
        w_scale_ptr: CutePointer,
        x_scale_ptr: CutePointer,
        *,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> Float32:
        acc_ir = Float32(acc).ir_value(loc=loc, ip=ip)
        weight_ir = _normalize_ptr(weight_ptr, loc=loc, ip=ip)
        x_ir = _normalize_ptr(x_ptr, loc=loc, ip=ip)
        w_scale_ir = _normalize_ptr(w_scale_ptr, loc=loc, ip=ip)
        x_scale_ir = _normalize_ptr(x_scale_ptr, loc=loc, ip=ip)
        result = llvm.inline_asm(
            T.f32(),
            [acc_ir, weight_ir, x_ir, w_scale_ir, x_scale_ir],
            """
            {
              .reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;
              .reg .b16 wsc0, wsc1, xsc0, xsc1, sf0, sf1, lo, hi, sumh;
              .reg .b32 wr0, wr1, wr2, wr3, wr4, wr5, wr6, wr7;
              .reg .b32 xr0, xr1, xr2, xr3, xr4, xr5, xr6, xr7;
              .reg .b32 w0, w1, w2, w3, w4, w5, w6, w7;
              .reg .b32 x0, x1, x2, x3, x4, x5, x6, x7;
              .reg .b32 prod, wsf2, xsf2, sf2;
              mov.f32 $0, $1;
              ld.global.L1::no_allocate.L2::evict_first.v8.b32
                {wr0, wr1, wr2, wr3, wr4, wr5, wr6, wr7}, [$2];
              ld.global.L1::evict_last.L2::evict_last.v8.b32
                {xr0, xr1, xr2, xr3, xr4, xr5, xr6, xr7}, [$3];
              ld.global.L1::no_allocate.v2.b16 {wsc0, wsc1}, [$4];
              ld.global.L1::evict_last.v2.b16 {xsc0, xsc1}, [$5];
              cvt.rn.f16x2.e4m3x2 wsf2, wsc0;
              cvt.rn.f16x2.e4m3x2 xsf2, xsc0;
              mul.rn.f16x2 sf2, wsf2, xsf2;
              mov.b32 {sf0, sf1}, sf2;
              mov.b32 {b0, b1, b2, b3}, wr0;
              mov.b32 {b4, b5, b6, b7}, wr1;
              cvt.rn.f16x2.e2m1x2 w0, b0; cvt.rn.f16x2.e2m1x2 w1, b1;
              cvt.rn.f16x2.e2m1x2 w2, b2; cvt.rn.f16x2.e2m1x2 w3, b3;
              cvt.rn.f16x2.e2m1x2 w4, b4; cvt.rn.f16x2.e2m1x2 w5, b5;
              cvt.rn.f16x2.e2m1x2 w6, b6; cvt.rn.f16x2.e2m1x2 w7, b7;
              mov.b32 {b0, b1, b2, b3}, xr0;
              mov.b32 {b4, b5, b6, b7}, xr1;
              cvt.rn.f16x2.e2m1x2 x0, b0; cvt.rn.f16x2.e2m1x2 x1, b1;
              cvt.rn.f16x2.e2m1x2 x2, b2; cvt.rn.f16x2.e2m1x2 x3, b3;
              cvt.rn.f16x2.e2m1x2 x4, b4; cvt.rn.f16x2.e2m1x2 x5, b5;
              cvt.rn.f16x2.e2m1x2 x6, b6; cvt.rn.f16x2.e2m1x2 x7, b7;
              mul.rn.f16x2 prod, w0, x0; fma.rn.f16x2 prod, w1, x1, prod;
              fma.rn.f16x2 prod, w2, x2, prod; fma.rn.f16x2 prod, w3, x3, prod;
              fma.rn.f16x2 prod, w4, x4, prod; fma.rn.f16x2 prod, w5, x5, prod;
              fma.rn.f16x2 prod, w6, x6, prod; fma.rn.f16x2 prod, w7, x7, prod;
              mov.b32 {lo, hi}, prod; add.rn.f16 sumh, lo, hi;
              fma.rn.f32.f16 $0, sumh, sf0, $0;
              mov.b32 {b0, b1, b2, b3}, wr2;
              mov.b32 {b4, b5, b6, b7}, wr3;
              cvt.rn.f16x2.e2m1x2 w0, b0; cvt.rn.f16x2.e2m1x2 w1, b1;
              cvt.rn.f16x2.e2m1x2 w2, b2; cvt.rn.f16x2.e2m1x2 w3, b3;
              cvt.rn.f16x2.e2m1x2 w4, b4; cvt.rn.f16x2.e2m1x2 w5, b5;
              cvt.rn.f16x2.e2m1x2 w6, b6; cvt.rn.f16x2.e2m1x2 w7, b7;
              mov.b32 {b0, b1, b2, b3}, xr2;
              mov.b32 {b4, b5, b6, b7}, xr3;
              cvt.rn.f16x2.e2m1x2 x0, b0; cvt.rn.f16x2.e2m1x2 x1, b1;
              cvt.rn.f16x2.e2m1x2 x2, b2; cvt.rn.f16x2.e2m1x2 x3, b3;
              cvt.rn.f16x2.e2m1x2 x4, b4; cvt.rn.f16x2.e2m1x2 x5, b5;
              cvt.rn.f16x2.e2m1x2 x6, b6; cvt.rn.f16x2.e2m1x2 x7, b7;
              mul.rn.f16x2 prod, w0, x0; fma.rn.f16x2 prod, w1, x1, prod;
              fma.rn.f16x2 prod, w2, x2, prod; fma.rn.f16x2 prod, w3, x3, prod;
              fma.rn.f16x2 prod, w4, x4, prod; fma.rn.f16x2 prod, w5, x5, prod;
              fma.rn.f16x2 prod, w6, x6, prod; fma.rn.f16x2 prod, w7, x7, prod;
              mov.b32 {lo, hi}, prod; add.rn.f16 sumh, lo, hi;
              fma.rn.f32.f16 $0, sumh, sf1, $0;
              cvt.rn.f16x2.e4m3x2 wsf2, wsc1;
              cvt.rn.f16x2.e4m3x2 xsf2, xsc1;
              mul.rn.f16x2 sf2, wsf2, xsf2;
              mov.b32 {sf0, sf1}, sf2;
              mov.b32 {b0, b1, b2, b3}, wr4;
              mov.b32 {b4, b5, b6, b7}, wr5;
              cvt.rn.f16x2.e2m1x2 w0, b0; cvt.rn.f16x2.e2m1x2 w1, b1;
              cvt.rn.f16x2.e2m1x2 w2, b2; cvt.rn.f16x2.e2m1x2 w3, b3;
              cvt.rn.f16x2.e2m1x2 w4, b4; cvt.rn.f16x2.e2m1x2 w5, b5;
              cvt.rn.f16x2.e2m1x2 w6, b6; cvt.rn.f16x2.e2m1x2 w7, b7;
              mov.b32 {b0, b1, b2, b3}, xr4;
              mov.b32 {b4, b5, b6, b7}, xr5;
              cvt.rn.f16x2.e2m1x2 x0, b0; cvt.rn.f16x2.e2m1x2 x1, b1;
              cvt.rn.f16x2.e2m1x2 x2, b2; cvt.rn.f16x2.e2m1x2 x3, b3;
              cvt.rn.f16x2.e2m1x2 x4, b4; cvt.rn.f16x2.e2m1x2 x5, b5;
              cvt.rn.f16x2.e2m1x2 x6, b6; cvt.rn.f16x2.e2m1x2 x7, b7;
              mul.rn.f16x2 prod, w0, x0; fma.rn.f16x2 prod, w1, x1, prod;
              fma.rn.f16x2 prod, w2, x2, prod; fma.rn.f16x2 prod, w3, x3, prod;
              fma.rn.f16x2 prod, w4, x4, prod; fma.rn.f16x2 prod, w5, x5, prod;
              fma.rn.f16x2 prod, w6, x6, prod; fma.rn.f16x2 prod, w7, x7, prod;
              mov.b32 {lo, hi}, prod; add.rn.f16 sumh, lo, hi;
              fma.rn.f32.f16 $0, sumh, sf0, $0;
              mov.b32 {b0, b1, b2, b3}, wr6;
              mov.b32 {b4, b5, b6, b7}, wr7;
              cvt.rn.f16x2.e2m1x2 w0, b0; cvt.rn.f16x2.e2m1x2 w1, b1;
              cvt.rn.f16x2.e2m1x2 w2, b2; cvt.rn.f16x2.e2m1x2 w3, b3;
              cvt.rn.f16x2.e2m1x2 w4, b4; cvt.rn.f16x2.e2m1x2 w5, b5;
              cvt.rn.f16x2.e2m1x2 w6, b6; cvt.rn.f16x2.e2m1x2 w7, b7;
              mov.b32 {b0, b1, b2, b3}, xr6;
              mov.b32 {b4, b5, b6, b7}, xr7;
              cvt.rn.f16x2.e2m1x2 x0, b0; cvt.rn.f16x2.e2m1x2 x1, b1;
              cvt.rn.f16x2.e2m1x2 x2, b2; cvt.rn.f16x2.e2m1x2 x3, b3;
              cvt.rn.f16x2.e2m1x2 x4, b4; cvt.rn.f16x2.e2m1x2 x5, b5;
              cvt.rn.f16x2.e2m1x2 x6, b6; cvt.rn.f16x2.e2m1x2 x7, b7;
              mul.rn.f16x2 prod, w0, x0; fma.rn.f16x2 prod, w1, x1, prod;
              fma.rn.f16x2 prod, w2, x2, prod; fma.rn.f16x2 prod, w3, x3, prod;
              fma.rn.f16x2 prod, w4, x4, prod; fma.rn.f16x2 prod, w5, x5, prod;
              fma.rn.f16x2 prod, w6, x6, prod; fma.rn.f16x2 prod, w7, x7, prod;
              mov.b32 {lo, hi}, prod; add.rn.f16 sumh, lo, hi;
              fma.rn.f32.f16 $0, sumh, sf1, $0;
            }
            """,
            "=f,f,l,l,l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return Float32(result)

    @cute.kernel
    def _fast_nvfp4_gemv_bf16in_kernel(
        weight: CuteTensor,
        x: CuteTensor,
        weight_scale: CuteTensor,
        out: CuteTensor,
        alpha: Float32,
        k_bytes: cutlass.Constexpr[int],
    ) -> None:
        tid_x, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        tid = Int32(tid_x)
        smem = cute.arch.alloc_smem(Float32, 128, alignment=16)
        partial = Float32(0.0)
        k_bytes_i32 = Int32(k_bytes)  # pyrefly: ignore[bad-argument-type]
        scale_cols = k_bytes // 8  # pyrefly: ignore[unsupported-operation]
        weight_row_ptr = weight.iterator + row * k_bytes_i32
        thread_k_base = tid * Int32(16)
        thread_scale_base = tid * Int32(2)
        num_iters = k_bytes // 2048  # pyrefly: ignore[unsupported-operation]
        for iter_k in cutlass.range(num_iters, unroll_full=True):
            k_tile_base = Int32(iter_k * 2048) + thread_k_base
            scale_tile_base = Int32(iter_k * 256) + thread_scale_base
            scale_offset = _fast_swizzled_scale_offset(
                row,
                scale_tile_base,
                scale_cols,
            )
            partial = _fast_bf16_16bytes_dual_ptr(
                partial,
                weight_row_ptr + k_tile_base,
                x.iterator + k_tile_base * Int32(2),
                weight_scale.iterator + scale_offset,
            )
        cute.arch.store(smem + tid, partial)
        cute.arch.sync_threads()
        block_val = Float32(0.0)
        if tid < Int32(64):
            block_val = partial + cute.arch.load(smem + tid + Int32(64), Float32)
            cute.arch.store(smem + tid, block_val)
        cute.arch.sync_threads()
        if tid < Int32(32):
            block_val += cute.arch.load(smem + tid + Int32(32), Float32)
            block_val += cute.arch.shuffle_sync_down(block_val, 16)
            block_val += cute.arch.shuffle_sync_down(block_val, 8)
            block_val += cute.arch.shuffle_sync_down(block_val, 4)
            block_val += cute.arch.shuffle_sync_down(block_val, 2)
            block_val += cute.arch.shuffle_sync_down(block_val, 1)
        if tid == Int32(0):
            out[row] = (block_val * alpha).to(cutlass.BFloat16)

    @cute.kernel
    def _fast_nvfp4_gemv_fp4in_kernel(
        weight: CuteTensor,
        x: CuteTensor,
        weight_scale: CuteTensor,
        x_scale: CuteTensor,
        out: CuteTensor,
        alpha: Float32,
        k_bytes: cutlass.Constexpr[int],
    ) -> None:
        tid_x, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        tid = Int32(tid_x)
        smem = cute.arch.alloc_smem(Float32, 2, alignment=16)
        partial = Float32(0.0)
        k_bytes_i32 = Int32(k_bytes)  # pyrefly: ignore[bad-argument-type]
        scale_cols = k_bytes // 8  # pyrefly: ignore[unsupported-operation]
        weight_row_ptr = weight.iterator + row * k_bytes_i32
        thread_k_base = tid * Int32(32)
        thread_scale_base = tid * Int32(4)
        num_iters = k_bytes // 2048  # pyrefly: ignore[unsupported-operation]
        for iter_k in cutlass.range(num_iters, unroll_full=True):
            k_tile_base = Int32(iter_k * 2048) + thread_k_base
            scale_tile_base = Int32(iter_k * 256) + thread_scale_base
            scale_offset = _fast_swizzled_scale_offset(
                row,
                scale_tile_base,
                scale_cols,
            )
            x_scale_offset = _fast_swizzled_scale_offset(
                Int32(0),
                scale_tile_base,
                scale_cols,
            )
            partial = _fast_fp4_32bytes_ptr(
                partial,
                weight_row_ptr + k_tile_base,
                x.iterator + k_tile_base,
                weight_scale.iterator + scale_offset,
                x_scale.iterator + x_scale_offset,
            )
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        warp_val = partial
        warp_val += cute.arch.shuffle_sync_down(warp_val, 16)
        warp_val += cute.arch.shuffle_sync_down(warp_val, 8)
        warp_val += cute.arch.shuffle_sync_down(warp_val, 4)
        warp_val += cute.arch.shuffle_sync_down(warp_val, 2)
        warp_val += cute.arch.shuffle_sync_down(warp_val, 1)
        if lane == Int32(0):
            cute.arch.store(smem + warp_id, warp_val)
        cute.arch.sync_threads()
        block_val = Float32(0.0)
        if tid < Int32(32):
            if tid < Int32(2):
                block_val = cute.arch.load(smem + tid, Float32)
            block_val += cute.arch.shuffle_sync_down(block_val, 4)
            block_val += cute.arch.shuffle_sync_down(block_val, 2)
            block_val += cute.arch.shuffle_sync_down(block_val, 1)
        if tid == Int32(0):
            out[row] = (block_val * alpha).to(cutlass.BFloat16)

else:
    _fast_nvfp4_gemv_bf16in_kernel = None
    _fast_nvfp4_gemv_fp4in_kernel = None


def _can_use_fast_cute_path(*tensors: Tensor, k_bytes: int) -> bool:
    return (
        _fast_nvfp4_gemv_bf16in_kernel is not None
        and k_bytes % 2048 == 0
        and all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors)
        and torch.cuda.get_device_capability(tensors[0].device) >= (10, 0)
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


@helion.kernel(static_shapes=True, config=BF16IN_CONFIG, backend="cute")
def _nvfp4_gemv_bf16in_kernel(
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
                row,
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


@helion.kernel(static_shapes=True, config=FP4IN_CONFIG, backend="cute")
def _nvfp4_gemv_fp4in_kernel(
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
                row,
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


def nvfp4_gemv_bf16in(
    weight_packed: Tensor,
    x_bf16: Tensor,
    weight_scale: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    """Compute ``weight_packed @ x_bf16`` for NVFP4 weights and BF16 input."""
    weight_fp4x2 = _as_fp4x2(weight_packed)
    weight_bytes = weight_fp4x2.view(torch.uint8)
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
    if _can_use_fast_cute_path(
        weight_bytes,
        x_bf16,
        weight_scale,
        k_bytes=weight_bytes.shape[1],
    ):
        default_cute_launcher(
            _fast_nvfp4_gemv_bf16in_kernel,
            (weight_bytes.shape[0],),
            weight_bytes,
            x_bf16,
            weight_scale.view(torch.uint8),
            out,
            alpha,
            weight_bytes.shape[1],
            block=(128, 1, 1),
        )
        return out
    return _nvfp4_gemv_bf16in_kernel(
        weight_fp4x2.view(weight_bytes.shape[0], weight_bytes.shape[1] // 8, 8),
        x_bf16.view(weight_bytes.shape[1] // 8, 16),
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
) -> Tensor:
    """Compute ``weight_packed @ x_packed`` for NVFP4 weights and input."""
    weight_fp4x2 = _as_fp4x2(weight_packed)
    x_fp4x2 = _as_fp4x2(x_packed)
    weight_bytes = weight_fp4x2.view(torch.uint8)
    x_bytes = x_fp4x2.view(torch.uint8)
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
    if _can_use_fast_cute_path(
        weight_bytes,
        x_bytes,
        weight_scale,
        x_scale,
        k_bytes=weight_bytes.shape[1],
    ):
        default_cute_launcher(
            _fast_nvfp4_gemv_fp4in_kernel,
            (weight_bytes.shape[0],),
            weight_bytes,
            x_bytes,
            weight_scale.view(torch.uint8),
            x_scale.view(torch.uint8),
            out,
            alpha,
            weight_bytes.shape[1],
            block=(64, 1, 1),
        )
        return out
    return _nvfp4_gemv_fp4in_kernel(
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


def check_bf16in(M: int, K_bytes: int) -> None:
    weight = torch.randint(0, 256, (M, K_bytes), dtype=torch.uint8, device=DEVICE).view(
        torch.float4_e2m1fn_x2
    )
    x = torch.randn(K_bytes * 2, dtype=torch.bfloat16, device=DEVICE)
    weight_scale = make_fp8_scales((M, K_bytes // 8), DEVICE)
    run_example(
        nvfp4_gemv_bf16in,
        reference_nvfp4_gemv_bf16in,
        (weight, x, weight_scale),
        atol=4.0,
        rtol=2e-1,
    )


def check_fp4in(M: int, K_bytes: int) -> None:
    weight = torch.randint(0, 256, (M, K_bytes), dtype=torch.uint8, device=DEVICE).view(
        torch.float4_e2m1fn_x2
    )
    x = torch.randint(0, 256, (K_bytes,), dtype=torch.uint8, device=DEVICE).view(
        torch.float4_e2m1fn_x2
    )
    weight_scale = make_fp8_scales((M, K_bytes // 8), DEVICE)
    x_scale = make_fp8_scales((K_bytes // 8,), DEVICE)
    run_example(
        nvfp4_gemv_fp4in,
        reference_nvfp4_gemv_fp4in,
        (weight, x, weight_scale, x_scale),
        atol=4.0,
        rtol=2e-1,
    )


def nvfp4_gemv_bf16in_tritonbench(
    tb_op: object,
    weight_packed: Tensor,
    x_bf16: Tensor,
    weight_scale: Tensor,
) -> Callable[[], Tensor]:
    return lambda: nvfp4_gemv_bf16in(weight_packed, x_bf16, weight_scale)


def nvfp4_gemv_fp4in_tritonbench(
    tb_op: object,
    weight_packed: Tensor,
    x_packed: Tensor,
    weight_scale: Tensor,
    x_scale: Tensor,
) -> Callable[[], Tensor]:
    return lambda: nvfp4_gemv_fp4in(weight_packed, x_packed, weight_scale, x_scale)


def main() -> None:
    check_bf16in(64, 128)
    check_fp4in(64, 128)


if __name__ == "__main__":
    main()
