from __future__ import annotations

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


class TestViews(TestCase):
    maxDiff = 16384

    def test_softmax_unsqueeze(self):
        @helion.kernel(config={"block_size": 1})
        def softmax(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1).unsqueeze(1)
                exp = torch.exp(values - amax)
                sum_exp = torch.unsqueeze(torch.sum(exp, dim=1), -1)
                out[tile_n, :] = exp / sum_exp
            return out

        x = torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16)
        code, result = code_and_output(softmax, (x,))
        torch.testing.assert_close(
            result, torch.nn.functional.softmax(x, dim=1), rtol=1e-2, atol=1e-1
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _softmax_kernel(x, out, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _m, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < _m
    values = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_1[None, :], other=0)
    v_0 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), values, float('-inf'))
    amax = tl.max(v_0, 1)
    amax_1 = amax[:, None]
    v_1 = values - amax_1
    v_2 = tl_math.exp(v_1)
    v_3 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), v_2, 0)
    sum_1 = tl.sum(v_3, 1)
    sum_exp = sum_1[None, :]
    v_4 = v_2 / sum_exp
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_4, mask_1[None, :])

def softmax(x: torch.Tensor):
    n, _m = x.size()
    out = torch.empty_like(x)
    _RDIM_SIZE_1 = triton.next_power_of_2(_m)
    _softmax_kernel[n,](x, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), _m, _RDIM_SIZE_1, num_warps=4, num_stages=3)
    return out

def _softmax_make_precompiler(x: torch.Tensor):
    n, _m = x.size()
    out = torch.empty_like(x)
    _RDIM_SIZE_1 = triton.next_power_of_2(_m)
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_softmax_kernel)(x, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), _m, _RDIM_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_softmax_view_reshape(self):
        @helion.kernel(config={"block_size": 1})
        def softmax(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1).view(tile_n, 1)
                exp = torch.exp(values - amax)
                sum_exp = torch.reshape(torch.sum(exp, dim=1), [tile_n, 1])
                out[tile_n, :] = exp / sum_exp
            return out

        x = torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16)
        code, result = code_and_output(softmax, (x,))
        torch.testing.assert_close(
            result, torch.nn.functional.softmax(x, dim=1), rtol=1e-2, atol=1e-1
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _softmax_kernel(x, out, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _m, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < _m
    values = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_1[None, :], other=0)
    v_0 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), values, float('-inf'))
    amax = tl.max(v_0, 1)
    amax_1 = tl.reshape(amax, [1, 1])
    v_1 = values - amax_1
    v_2 = tl_math.exp(v_1)
    v_3 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), v_2, 0)
    sum_1 = tl.sum(v_3, 1)
    sum_exp = tl.reshape(sum_1, [1, 1])
    v_4 = v_2 / sum_exp
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_4, mask_1[None, :])

def softmax(x: torch.Tensor):
    n, _m = x.size()
    out = torch.empty_like(x)
    _RDIM_SIZE_1 = triton.next_power_of_2(_m)
    _softmax_kernel[n,](x, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), _m, _RDIM_SIZE_1, num_warps=4, num_stages=3)
    return out

def _softmax_make_precompiler(x: torch.Tensor):
    n, _m = x.size()
    out = torch.empty_like(x)
    _RDIM_SIZE_1 = triton.next_power_of_2(_m)
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_softmax_kernel)(x, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), _m, _RDIM_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_squeeze(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "block_ptr"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                out[tile_n, tile_m] = x[tile_n, tile_m] + y[tile_m, :].squeeze(
                    1
                ).unsqueeze(0)
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024, 1], device=DEVICE),
        )
        code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1][:, 0].unsqueeze(0))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, y, out, out_size_0, out_size_1, x_size_0, x_size_1, y_size_0, y_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    load_1 = tl.load(tl.make_block_ptr(y, [y_size_0, y_size_1], [y_stride_0, y_stride_1], [offset_1, 0], [_BLOCK_SIZE_1, 1], [1, 0]), boundary_check=[0], padding_option='zero')
    squeeze = tl.reshape(load_1, [_BLOCK_SIZE_1])
    unsqueeze = squeeze[None, :]
    v_0 = load + unsqueeze
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_0, boundary_check=[0, 1])

def fn(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0) * triton.cdiv(x.size(1), _BLOCK_SIZE_1),](x, y, out, out.size(0), out.size(1), x.size(0), x.size(1), y.size(0), y.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, y, out, out.size(0), out.size(1), x.size(0), x.size(1), y.size(0), y.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_transpose(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "block_ptr"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                out[tile_n, tile_m] = x[tile_n, tile_m] + y[tile_m, :].transpose(0, 1)
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024, 1], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1].transpose(0, 1))

    def test_expand(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "block_ptr"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                out[tile_n, tile_m] = x[tile_n, tile_m] + y[tile_n, :].expand(
                    tile_n, tile_m
                )
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024, 1], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_expand_as(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "block_ptr"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                a = x[tile_n, tile_m]
                b = y[tile_m].expand_as(a)
                out[tile_n, tile_m] = a + b
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_expand_slicing(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "pointer"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                a = x[tile_n, tile_m]
                b = y[tile_m]
                out[tile_n, tile_m] = a + b[None, :]
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1])

    def test_expand_implicit(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "pointer"})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_n, tile_m in hl.tile(x.size()):
                a = x[tile_n, tile_m]
                b = y[tile_m]
                out[tile_n, tile_m] = a + b
            return out

        args = (
            torch.randn([1024, 1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        _code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + args[1])
