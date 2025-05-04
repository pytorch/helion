from __future__ import annotations

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


class TestControlFlow(TestCase):
    maxDiff = 16384

    def test_if_arg(self):
        @helion.kernel()
        def fn(x, v):
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code0, result = code_and_output(
            fn,
            (x, 5),
        )
        torch.testing.assert_close(result, torch.sigmoid(x))
        code1, result = code_and_output(
            fn,
            (x, 10),
        )
        torch.testing.assert_close(result, torch.sin(x))
        self.assertEqual(code0, code1)
        self.assertExpectedInline(
            code0,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _fn_kernel(x, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, v, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < x_size_1
    gt = v > 3
    lt = v < 7
    _and = gt and lt
    if _and:
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
        v_0 = tl.sigmoid(load)
        tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_0, mask_0[:, None] & mask_1[None, :])
    _not = not _and
    if _not:
        load_1 = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
        v_1 = tl_math.sin(load_1)
        tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_1, mask_0[:, None] & mask_1[None, :])

def fn(x, v):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0) * triton.cdiv(x.size(1), _BLOCK_SIZE_1),](x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), v, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x, v):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), v, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_constant_true(self):
        @helion.kernel(config={"block_size": 128, "indexing": "block_ptr"})
        def fn(x):
            out = torch.empty_like(x)
            v = 4
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x,),
        )
        torch.testing.assert_close(result, torch.sigmoid(x))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0_1: tl.constexpr):
    offsets_0_1 = tl.program_id(0) * _BLOCK_SIZE_0_1 + tl.arange(0, _BLOCK_SIZE_0_1).to(tl.int32)
    indices_1 = offsets_0_1 % x_size_1
    indices_0 = offsets_0_1 // x_size_1
    mask_0_1 = offsets_0_1 < x_size_0 * x_size_1
    load = tl.load(x + (indices_0 * x_stride_0 + indices_1 * x_stride_1), mask_0_1, other=0)
    v_0 = tl.sigmoid(load)
    tl.store(out + (indices_0 * out_stride_0 + indices_1 * out_stride_1), v_0, mask_0_1)

def fn(x):
    out = torch.empty_like(x)
    v = 4
    _BLOCK_SIZE_0_1 = 128
    _fn_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE_0_1), 1, 1](x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x):
    out = torch.empty_like(x)
    v = 4
    _BLOCK_SIZE_0_1 = 128
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)""",
        )

    def test_constant_false(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "block_ptr"})
        def fn(x):
            out = torch.empty_like(x)
            v = 15
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x,),
        )
        torch.testing.assert_close(result, torch.sin(x))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _fn_kernel(x, out, out_size_0, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    v_0 = tl_math.sin(load)
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_0, boundary_check=[0, 1])

def fn(x):
    out = torch.empty_like(x)
    v = 15
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0) * triton.cdiv(x.size(1), _BLOCK_SIZE_1),](x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x):
    out = torch.empty_like(x)
    v = 15
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )
