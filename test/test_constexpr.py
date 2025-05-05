from __future__ import annotations

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


class TestConstExpr(TestCase):
    maxDiff = 16384

    def test_constexpr_float(self):
        @helion.kernel()
        def fn(x: torch.Tensor, v: hl.constexpr) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = torch.sigmoid(x[tile] + v)
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, 5.0),
        )
        torch.testing.assert_close(result, torch.sigmoid(x + 5.0))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import helion.language as hl
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < x_size_1
    load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    v_0 = 5.0
    v_1 = load + v_0
    v_2 = tl.sigmoid(v_1)
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_2, mask_0[:, None] & mask_1[None, :])

def fn(x: torch.Tensor, v: hl.constexpr):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0) * triton.cdiv(x.size(1), _BLOCK_SIZE_1),](x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor, v: hl.constexpr):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_constexpr_float_wrapped(self):
        @helion.kernel()
        def fn(x: torch.Tensor, v: float) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = torch.sigmoid(x[tile] + v)
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, hl.constexpr(5.0)),
        )
        torch.testing.assert_close(result, torch.sigmoid(x + 5.0))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < x_size_1
    load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    v_0 = 5.0
    v_1 = load + v_0
    v_2 = tl.sigmoid(v_1)
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_2, mask_0[:, None] & mask_1[None, :])

def fn(x: torch.Tensor, v: float):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0) * triton.cdiv(x.size(1), _BLOCK_SIZE_1),](x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor, v: float):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_constexpr_size(self):
        @helion.kernel()
        def fn(x: torch.Tensor, s: hl.constexpr) -> torch.Tensor:
            (b,) = x.size()
            out = torch.empty([b, s], device=x.device, dtype=x.dtype)
            for tile_b, tile_s in hl.tile([b, s]):
                out[tile_b, tile_s] = x[tile_b].view(-1, 1).expand(tile_b, tile_s)
            return out

        x = torch.randn([512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, 16),
        )
        torch.testing.assert_close(result, x.view(-1, 1).expand(512, 16))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import helion.language as hl
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, out_stride_0, out_stride_1, x_stride_0, b, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(b, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < b
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    load = tl.load(x + indices_0 * x_stride_0, mask_0, other=0)
    view = tl.reshape(load, [_BLOCK_SIZE_0, 1])
    expand = tl.broadcast_to(view, [_BLOCK_SIZE_0, _BLOCK_SIZE_1])
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), expand, mask_0[:, None])

def fn(x: torch.Tensor, s: hl.constexpr):
    b, = x.size()
    out = torch.empty([b, 16], device=x.device, dtype=x.dtype)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 16
    _fn_kernel[triton.cdiv(b, _BLOCK_SIZE_0) * triton.cdiv(16, _BLOCK_SIZE_1),](x, out, out.stride(0), out.stride(1), x.stride(0), b, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor, s: hl.constexpr):
    b, = x.size()
    out = torch.empty([b, 16], device=x.device, dtype=x.dtype)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, out.stride(0), out.stride(1), x.stride(0), b, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )
