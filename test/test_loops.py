from __future__ import annotations

import functools
from pathlib import Path
import unittest

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import import_path
import helion.language as hl

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")


@helion.kernel
def device_loop_3d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    for tile_a in hl.tile(a):
        for tile_b, tile_c, tile_d in hl.tile([b, c, d]):
            out[tile_a, tile_b, tile_c, tile_d] = torch.sin(
                x[tile_a, tile_b, tile_c, tile_d]
            )
    return out


class TestLoops(TestCase):
    maxDiff = 16384

    def test_pointwise_device_loop(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.pointwise_device_loop,
            args,
            block_sizes=[32, 32],
        )
        torch.testing.assert_close(result, torch.sigmoid(args[0] + 1))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _pointwise_device_loop_kernel(x, out, out_stride_0, out_stride_1, x_stride_0, x_stride_1, n, m, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < n
    for offset_1 in tl.range(0, m.to(tl.int32), _BLOCK_SIZE_1):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < m
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
        v_0 = 1.0
        v_1 = load + v_0
        v_2 = tl.sigmoid(v_1)
        tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_2, mask_0[:, None] & mask_1[None, :])

def pointwise_device_loop(x: torch.Tensor):
    out = torch.empty_like(x)
    n, m = x.shape
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _pointwise_device_loop_kernel[triton.cdiv(n, _BLOCK_SIZE_0),](x, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), n, m, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _pointwise_device_loop_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    n, m = x.shape
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_pointwise_device_loop_kernel)(x, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), n, m, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_3d_device_loop0(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[1, 8, 8, 8],
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _device_loop_3d_kernel(x, out, out_stride_0, out_stride_1, out_stride_2, out_stride_3, x_stride_0, x_stride_1, x_stride_2, x_stride_3, b, c, d, _BLOCK_SIZE_3: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    for offset_1 in tl.range(0, b.to(tl.int32), _BLOCK_SIZE_1):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < b
        for offset_2 in tl.range(0, c.to(tl.int32), _BLOCK_SIZE_2):
            indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
            mask_2 = indices_2 < c
            for offset_3 in tl.range(0, d.to(tl.int32), _BLOCK_SIZE_3):
                indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
                mask_3 = indices_3 < d
                load = tl.load(x + (indices_0[:, None, None, None] * x_stride_0 + indices_1[None, :, None, None] * x_stride_1 + indices_2[None, None, :, None] * x_stride_2 + indices_3[None, None, None, :] * x_stride_3), mask_1[None, :, None, None] & mask_2[None, None, :, None] & mask_3[None, None, None, :], other=0)
                v_0 = tl_math.sin(load)
                tl.store(out + (indices_0[:, None, None, None] * out_stride_0 + indices_1[None, :, None, None] * out_stride_1 + indices_2[None, None, :, None] * out_stride_2 + indices_3[None, None, None, :] * out_stride_3), v_0, mask_1[None, :, None, None] & mask_2[None, None, :, None] & mask_3[None, None, None, :])

def device_loop_3d(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_3 = 8
    _BLOCK_SIZE_2 = 8
    _BLOCK_SIZE_1 = 8
    _device_loop_3d_kernel[a,](x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), b, c, d, _BLOCK_SIZE_3, _BLOCK_SIZE_2, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _device_loop_3d_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_3 = 8
    _BLOCK_SIZE_2 = 8
    _BLOCK_SIZE_1 = 8
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_device_loop_3d_kernel)(x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), b, c, d, _BLOCK_SIZE_3, _BLOCK_SIZE_2, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_3d_device_loop1(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[2, 8, 4, 1],
            loop_order=[1, 0, 2],
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _device_loop_3d_kernel(x, out, out_stride_0, out_stride_1, out_stride_2, out_stride_3, x_stride_0, x_stride_1, x_stride_2, x_stride_3, a, b, c, d, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < a
    for offset_2 in tl.range(0, c.to(tl.int32), _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        mask_2 = indices_2 < c
        for offset_1 in tl.range(0, b.to(tl.int32), _BLOCK_SIZE_1):
            indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
            mask_1 = indices_1 < b
            for offset_3 in tl.range(0, d.to(tl.int32), 1):
                indices_3 = offset_3 + tl.arange(0, 1).to(tl.int32)
                load = tl.load(x + (indices_0[:, None, None, None] * x_stride_0 + indices_1[None, :, None, None] * x_stride_1 + indices_2[None, None, :, None] * x_stride_2 + indices_3[None, None, None, :] * x_stride_3), mask_0[:, None, None, None] & mask_1[None, :, None, None] & mask_2[None, None, :, None], other=0)
                v_0 = tl_math.sin(load)
                tl.store(out + (indices_0[:, None, None, None] * out_stride_0 + indices_1[None, :, None, None] * out_stride_1 + indices_2[None, None, :, None] * out_stride_2 + indices_3[None, None, None, :] * out_stride_3), v_0, mask_0[:, None, None, None] & mask_1[None, :, None, None] & mask_2[None, None, :, None])

def device_loop_3d(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_0 = 2
    _BLOCK_SIZE_1 = 8
    _BLOCK_SIZE_2 = 4
    _device_loop_3d_kernel[triton.cdiv(a, _BLOCK_SIZE_0),](x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), a, b, c, d, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _device_loop_3d_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_0 = 2
    _BLOCK_SIZE_1 = 8
    _BLOCK_SIZE_2 = 4
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_device_loop_3d_kernel)(x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), a, b, c, d, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_3d_device_loop2(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[4, 128, 1, 1],
            flatten_loops=[True],
            loop_order=[2, 0, 1],
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _device_loop_3d_kernel(x, out, out_stride_0, out_stride_1, out_stride_2, out_stride_3, x_stride_0, x_stride_1, x_stride_2, x_stride_3, a, b, c, d, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1_2_3: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < a
    for lid_1_2_3 in tl.range(tl.cdiv(b * c * d, _BLOCK_SIZE_1_2_3)):
        offsets_1_2_3 = lid_1_2_3 * _BLOCK_SIZE_1_2_3 + tl.arange(0, _BLOCK_SIZE_1_2_3).to(tl.int32)
        indices_2 = offsets_1_2_3 % c
        indices_1 = offsets_1_2_3 // c % b
        indices_3 = offsets_1_2_3 // (b * c)
        mask_1_2_3 = offsets_1_2_3 < b * c * d
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1 + indices_2[None, :] * x_stride_2 + indices_3[None, :] * x_stride_3), mask_0[:, None] & mask_1_2_3[None, :], other=0)
        v_0 = tl_math.sin(load)
        tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1 + indices_2[None, :] * out_stride_2 + indices_3[None, :] * out_stride_3), v_0, mask_0[:, None] & mask_1_2_3[None, :])

def device_loop_3d(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_0 = 4
    _BLOCK_SIZE_1_2_3 = 128
    _device_loop_3d_kernel[triton.cdiv(a, _BLOCK_SIZE_0),](x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), a, b, c, d, _BLOCK_SIZE_0, _BLOCK_SIZE_1_2_3, num_warps=4, num_stages=3)
    return out

def _device_loop_3d_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_0 = 4
    _BLOCK_SIZE_1_2_3 = 128
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_device_loop_3d_kernel)(x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), a, b, c, d, _BLOCK_SIZE_0, _BLOCK_SIZE_1_2_3, num_warps=4, num_stages=3)""",
        )

    def test_3d_device_loop3(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[2, 8, 4, 1],
            loop_order=[2, 0, 1],
            indexing="block_ptr",
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _device_loop_3d_kernel(x, out, out_size_0, out_size_1, out_size_2, out_size_3, x_size_0, x_size_1, x_size_2, x_size_3, out_stride_0, out_stride_1, out_stride_2, out_stride_3, x_stride_0, x_stride_1, x_stride_2, x_stride_3, b, c, d, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    for offset_3 in tl.range(0, d.to(tl.int32), 1):
        for offset_1 in tl.range(0, b.to(tl.int32), _BLOCK_SIZE_1):
            for offset_2 in tl.range(0, c.to(tl.int32), _BLOCK_SIZE_2):
                load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1, x_size_2, x_size_3], [x_stride_0, x_stride_1, x_stride_2, x_stride_3], [offset_0, offset_1, offset_2, offset_3], [_BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, 1], [3, 2, 1, 0]), boundary_check=[0, 1, 2, 3], padding_option='zero')
                v_0 = tl_math.sin(load)
                tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1, out_size_2, out_size_3], [out_stride_0, out_stride_1, out_stride_2, out_stride_3], [offset_0, offset_1, offset_2, offset_3], [_BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, 1], [3, 2, 1, 0]), v_0, boundary_check=[0, 1, 2, 3])

def device_loop_3d(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_0 = 2
    _BLOCK_SIZE_2 = 4
    _BLOCK_SIZE_1 = 8
    _device_loop_3d_kernel[triton.cdiv(a, _BLOCK_SIZE_0),](x, out, out.size(0), out.size(1), out.size(2), out.size(3), x.size(0), x.size(1), x.size(2), x.size(3), out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), b, c, d, _BLOCK_SIZE_0, _BLOCK_SIZE_2, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _device_loop_3d_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_0 = 2
    _BLOCK_SIZE_2 = 4
    _BLOCK_SIZE_1 = 8
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_device_loop_3d_kernel)(x, out, out.size(0), out.size(1), out.size(2), out.size(3), x.size(0), x.size(1), x.size(2), x.size(3), out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), b, c, d, _BLOCK_SIZE_0, _BLOCK_SIZE_2, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_loop_fixed_block(self):
        @helion.kernel(config={"block_sizes": [], "indexing": "block_ptr"})
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            a, b, c = x.shape
            for tile_a, tile_b in hl.tile([a, b], block_size=[4, 8]):
                for tile_c in hl.tile(c, block_size=16):
                    out[tile_a, tile_b, tile_c] = torch.sin(x[tile_a, tile_b, tile_c])
            return out

        args = (torch.randn([128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _fn_kernel(x, out, out_size_0, out_size_1, out_size_2, x_size_0, x_size_1, x_size_2, out_stride_0, out_stride_1, out_stride_2, x_stride_0, x_stride_1, x_stride_2, a, c, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_blocks_0 = tl.cdiv(a, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    for offset_2 in tl.range(0, c.to(tl.int32), _BLOCK_SIZE_2):
        load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1, x_size_2], [x_stride_0, x_stride_1, x_stride_2], [offset_0, offset_1, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2], [2, 1, 0]), boundary_check=[0, 1, 2], padding_option='zero')
        v_0 = tl_math.sin(load)
        tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1, out_size_2], [out_stride_0, out_stride_1, out_stride_2], [offset_0, offset_1, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2], [2, 1, 0]), v_0, boundary_check=[0, 1, 2])

def fn(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c = x.shape
    _BLOCK_SIZE_0 = 4
    _BLOCK_SIZE_1 = 8
    _BLOCK_SIZE_2 = 16
    _fn_kernel[triton.cdiv(a, _BLOCK_SIZE_0) * triton.cdiv(b, _BLOCK_SIZE_1),](x, out, out.size(0), out.size(1), out.size(2), x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), a, c, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c = x.shape
    _BLOCK_SIZE_0 = 4
    _BLOCK_SIZE_1 = 8
    _BLOCK_SIZE_2 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, out.size(0), out.size(1), out.size(2), x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), a, c, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_loop_arg_block(self):
        @helion.kernel(config={"block_sizes": [], "indexing": "block_ptr"})
        def fn(x: torch.Tensor, block_size: int) -> torch.Tensor:
            out = torch.empty_like(x)
            (a,) = x.shape
            for tile_a in hl.tile(a, block_size=block_size):
                out[tile_a] = torch.sin(x[tile_a])
            return out

        args = (torch.randn([1024], device=DEVICE), 32)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, torch.sin(args[0]))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _fn_kernel(x, out, out_size_0, x_size_0, out_stride_0, x_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    load = tl.load(tl.make_block_ptr(x, [x_size_0], [x_stride_0], [offset_0], [_BLOCK_SIZE_0], [0]), boundary_check=[0], padding_option='zero')
    v_0 = tl_math.sin(load)
    tl.store(tl.make_block_ptr(out, [out_size_0], [out_stride_0], [offset_0], [_BLOCK_SIZE_0], [0]), v_0, boundary_check=[0])

def fn(x: torch.Tensor, block_size: int):
    out = torch.empty_like(x)
    a, = x.shape
    _BLOCK_SIZE_0 = block_size
    _fn_kernel[triton.cdiv(a, _BLOCK_SIZE_0),](x, out, out.size(0), x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor, block_size: int):
    out = torch.empty_like(x)
    a, = x.shape
    _BLOCK_SIZE_0 = block_size
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, out.size(0), x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_three_level_matmul(self):
        @helion.kernel(static_shapes=True)
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            for tile_m in hl.tile(m):
                for tile_n in hl.tile(n):
                    acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                    for tile_k in hl.tile(k):
                        acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                    out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([256, 512], device=DEVICE),
            torch.randn([512, 128], device=DEVICE),
        )
        code, result = code_and_output(matmul, args, block_sizes=[16, 64, 64])
        torch.testing.assert_close(
            result, functools.reduce(torch.matmul, args), atol=1e-1, rtol=1e-2
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(x, y, out, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    for offset_1 in tl.range(0, 128, _BLOCK_SIZE_1):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
        for offset_2 in tl.range(0, 512, _BLOCK_SIZE_2):
            indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
            acc_copy = acc
            acc_copy_0 = acc_copy
            load = tl.load(x + (indices_0[:, None] * 512 + indices_2[None, :] * 1), None)
            load_1 = tl.load(y + (indices_2[:, None] * 128 + indices_1[None, :] * 1), None)
            acc = tl.dot(load, load_1, acc=acc_copy_0, input_precision='tf32')
        tl.store(out + (indices_0[:, None] * 128 + indices_1[None, :] * 1), acc, None)

def matmul(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = 64
    _matmul_kernel[triton.cdiv(256, _BLOCK_SIZE_0),](x, y, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _matmul_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_matmul_kernel)(x, y, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_data_dependent_bounds1(self):
        @helion.kernel()
        def fn(x: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            bs = hl.register_block_size(x.size(1))
            for tile0 in hl.tile(x.size(0)):
                acc = hl.zeros([tile0, bs])
                for tile1 in hl.tile(end[0], block_size=bs):
                    acc += x[tile0, tile1]
                out[tile0] = acc.sum(-1)
            return out

        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.tensor([200], device=DEVICE, dtype=torch.int64),
        )
        code, result = code_and_output(fn, args, block_sizes=[32, 32])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, end, out, x_size_0, out_stride_0, x_stride_0, x_stride_1, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_1 = pid_0 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < x_size_0
    acc = tl.full([_BLOCK_SIZE_1, _BLOCK_SIZE_0], 0.0, tl.float32)
    load = tl.load(end + tl.zeros([], tl.int32), None)
    for offset_0 in tl.range(0, load.to(tl.int32), _BLOCK_SIZE_0):
        indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
        mask_0 = indices_0 < load
        acc_copy = acc
        acc_copy_0 = acc_copy
        load_1 = tl.load(x + (indices_1[:, None] * x_stride_0 + indices_0[None, :] * x_stride_1), mask_1[:, None] & mask_0[None, :], other=0)
        acc = acc_copy_0 + load_1
    sum_1 = tl.sum(acc, 1)
    tl.store(out + indices_1 * out_stride_0, sum_1, mask_1)

def fn(x: torch.Tensor, end: torch.Tensor):
    out = x.new_empty([x.size(0)])
    bs = 32
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_0 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_1),](x, end, out, x.size(0), out.stride(0), x.stride(0), x.stride(1), _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor, end: torch.Tensor):
    out = x.new_empty([x.size(0)])
    bs = 32
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_0 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, end, out, x.size(0), out.stride(0), x.stride(0), x.stride(1), _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )
        torch.testing.assert_close(result, args[0][:, : args[1][0].item()].sum(-1))

    def test_data_dependent_bounds2(self):
        @helion.kernel()
        def fn(x: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile0 in hl.tile(x.size(0)):
                acc = hl.zeros([tile0])
                for tile1 in hl.tile(end[0]):
                    acc += x[tile0, tile1].sum(-1)
                out[tile0] = acc
            return out

        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.tensor([200], device=DEVICE, dtype=torch.int32),
        )
        code, result = code_and_output(
            fn, args, block_sizes=[32, 32], indexing="block_ptr"
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, end, out, out_size_0, x_size_0, out_stride_0, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    acc = tl.full([_BLOCK_SIZE_0], 0.0, tl.float32)
    load = tl.load(end + tl.zeros([], tl.int32), None)
    for offset_1 in tl.range(0, load.to(tl.int32), _BLOCK_SIZE_1):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < load
        acc_copy = acc
        acc_copy_0 = acc_copy
        load_1 = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
        sum_1 = tl.sum(load_1, 1)
        acc = acc_copy_0 + sum_1
    tl.store(tl.make_block_ptr(out, [out_size_0], [out_stride_0], [offset_0], [_BLOCK_SIZE_0], [0]), acc, boundary_check=[0])

def fn(x: torch.Tensor, end: torch.Tensor):
    out = x.new_empty([x.size(0)])
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, end, out, out.size(0), x.size(0), out.stride(0), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor, end: torch.Tensor):
    out = x.new_empty([x.size(0)])
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, end, out, out.size(0), x.size(0), out.stride(0), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )
        torch.testing.assert_close(result, args[0][:, : args[1][0].item()].sum(-1))

    def test_data_dependent_bounds3(self):
        @helion.kernel()
        def fn(x: torch.Tensor, end0: torch.Tensor, end1: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile0 in hl.tile(x.size(0)):
                acc = hl.zeros([tile0], dtype=x.dtype)
                for tile1, tile2 in hl.tile([end0[0], end1[0]]):
                    # TODO(jansel): make this version work
                    # acc += x[tile0, tile1, tile2].reshape(tile0, -1).sum(-1)
                    acc += x[tile0, tile1, tile2].sum(-1).sum(-1)
                out[tile0] = acc
            return out

        args = (
            torch.randn([32, 256, 256], device=DEVICE, dtype=torch.float64),
            torch.tensor([200], device=DEVICE, dtype=torch.int64),
            torch.tensor([150], device=DEVICE, dtype=torch.int64),
        )
        code, result = code_and_output(fn, args, block_sizes=[32, 32, 32])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, end0, end1, out, x_size_0, out_stride_0, x_stride_0, x_stride_1, x_stride_2, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    acc = tl.full([_BLOCK_SIZE_0], 0.0, tl.float64)
    load = tl.load(end0 + tl.zeros([], tl.int32), None)
    load_1 = tl.load(end1 + tl.zeros([], tl.int32), None)
    for offset_1 in tl.range(0, load.to(tl.int32), _BLOCK_SIZE_1):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < load
        for offset_2 in tl.range(0, load_1.to(tl.int32), _BLOCK_SIZE_2):
            indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
            mask_2 = indices_2 < load_1
            acc_copy = acc
            acc_copy_0 = acc_copy
            load_2 = tl.load(x + (indices_0[:, None, None] * x_stride_0 + indices_1[None, :, None] * x_stride_1 + indices_2[None, None, :] * x_stride_2), mask_0[:, None, None] & mask_1[None, :, None] & mask_2[None, None, :], other=0)
            sum_1 = tl.sum(load_2, 2)
            sum_2 = tl.sum(sum_1, 1)
            acc = acc_copy_0 + sum_2
    tl.store(out + indices_0 * out_stride_0, acc, mask_0)

def fn(x: torch.Tensor, end0: torch.Tensor, end1: torch.Tensor):
    out = x.new_empty([x.size(0)])
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_2 = 32
    _BLOCK_SIZE_1 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, end0, end1, out, x.size(0), out.stride(0), x.stride(0), x.stride(1), x.stride(2), _BLOCK_SIZE_0, _BLOCK_SIZE_2, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor, end0: torch.Tensor, end1: torch.Tensor):
    out = x.new_empty([x.size(0)])
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_2 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, end0, end1, out, x.size(0), out.stride(0), x.stride(0), x.stride(1), x.stride(2), _BLOCK_SIZE_0, _BLOCK_SIZE_2, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )
        torch.testing.assert_close(
            result, args[0][:, : args[1][0].item(), : args[2][0].item()].sum(-1).sum(-1)
        )

    def test_data_dependent_bounds4(self):
        @helion.kernel()
        def fn(x: torch.Tensor, begin: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            bs = hl.register_block_size(8192)
            for tile0 in hl.tile(x.size(0)):
                acc = hl.zeros([tile0, bs])
                for tile1 in hl.tile(begin[0], end[0], block_size=bs):
                    acc += x[tile0, tile1]
                out[tile0] = acc.sum(-1)
            return out

        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.tensor([100], device=DEVICE, dtype=torch.int64),
            torch.tensor([200], device=DEVICE, dtype=torch.int64),
        )
        code, result = code_and_output(fn, args, block_sizes=[32, 32])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, begin, end, out, x_size_0, out_stride_0, x_stride_0, x_stride_1, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_1 = pid_0 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < x_size_0
    acc = tl.full([_BLOCK_SIZE_1, _BLOCK_SIZE_0], 0.0, tl.float32)
    load = tl.load(begin + tl.zeros([], tl.int32), None)
    load_1 = tl.load(end + tl.zeros([], tl.int32), None)
    for offset_0 in tl.range(load.to(tl.int32), load_1.to(tl.int32), _BLOCK_SIZE_0):
        indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
        mask_0 = indices_0 < load_1
        acc_copy = acc
        acc_copy_0 = acc_copy
        load_2 = tl.load(x + (indices_1[:, None] * x_stride_0 + indices_0[None, :] * x_stride_1), mask_1[:, None] & mask_0[None, :], other=0)
        acc = acc_copy_0 + load_2
    sum_1 = tl.sum(acc, 1)
    tl.store(out + indices_1 * out_stride_0, sum_1, mask_1)

def fn(x: torch.Tensor, begin: torch.Tensor, end: torch.Tensor):
    out = x.new_empty([x.size(0)])
    bs = 32
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_0 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_1),](x, begin, end, out, x.size(0), out.stride(0), x.stride(0), x.stride(1), _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor, begin: torch.Tensor, end: torch.Tensor):
    out = x.new_empty([x.size(0)])
    bs = 32
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_0 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, begin, end, out, x.size(0), out.stride(0), x.stride(0), x.stride(1), _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )
        torch.testing.assert_close(
            result, args[0][:, args[1][0].item() : args[2][0].item()].sum(-1)
        )

    def test_data_dependent_bounds5(self):
        @helion.kernel()
        def fn(x: torch.Tensor, begin: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            for tile0 in hl.tile(x.size(0)):
                acc = hl.zeros([tile0])
                for (tile1,) in hl.tile([begin[0]], [end[0]]):
                    acc += x[tile0, tile1].sum(-1)
                out[tile0] = acc
            return out

        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.tensor([100], device=DEVICE, dtype=torch.int64),
            torch.tensor([200], device=DEVICE, dtype=torch.int64),
        )
        code, result = code_and_output(fn, args, block_sizes=[32, 32])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, begin, end, out, x_size_0, out_stride_0, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    acc = tl.full([_BLOCK_SIZE_0], 0.0, tl.float32)
    load = tl.load(begin + tl.zeros([], tl.int32), None)
    load_1 = tl.load(end + tl.zeros([], tl.int32), None)
    for offset_1 in tl.range(load.to(tl.int32), load_1.to(tl.int32), _BLOCK_SIZE_1):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < load_1
        acc_copy = acc
        acc_copy_0 = acc_copy
        load_2 = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
        sum_1 = tl.sum(load_2, 1)
        acc = acc_copy_0 + sum_1
    tl.store(out + indices_0 * out_stride_0, acc, mask_0)

def fn(x: torch.Tensor, begin: torch.Tensor, end: torch.Tensor):
    out = x.new_empty([x.size(0)])
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, begin, end, out, x.size(0), out.stride(0), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor, begin: torch.Tensor, end: torch.Tensor):
    out = x.new_empty([x.size(0)])
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, begin, end, out, x.size(0), out.stride(0), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )
        torch.testing.assert_close(
            result, args[0][:, args[1][0].item() : args[2][0].item()].sum(-1)
        )

    def test_register_block_size_minimum(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            bs = hl.register_block_size(32, 256)
            for tile0 in hl.tile(x.size(0), block_size=bs):
                out[tile0] = x[tile0] + 1
            return out

        args = (torch.randn([1024], device=DEVICE, dtype=torch.float32),)
        code, result = code_and_output(fn, args, block_size=64)
        torch.testing.assert_close(result, args[0] + 1)
        spec = fn.bind(args).config_spec.block_sizes[0]
        self.assertEqual(spec.size_hint, 1024)
        self.assertEqual(spec.min_size, 32)
        self.assertEqual(spec.max_size, 256)

    def test_reorder_with_register_block_size(self):
        @helion.kernel(
            config={
                "block_sizes": [64, 32],
                "indexing": "block_ptr",
                "loop_order": [1, 0],
            }
        )
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            bs0 = hl.register_block_size(1024)
            bs1 = hl.register_block_size(1024)
            for tile0, tile1 in hl.tile(x.size(), block_size=[bs0, bs1]):
                out[tile0, tile1] = x[tile0, tile1] + 1
            return out

        args = (torch.randn([2048, 2048], device=DEVICE),)
        code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + 1)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, out_size_0, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_1, _BLOCK_SIZE_1)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_1 = pid_0 * _BLOCK_SIZE_1
    offset_0 = pid_1 * _BLOCK_SIZE_0
    load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    v_0 = 1.0
    v_1 = load + v_0
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_1, boundary_check=[0, 1])

def fn(x: torch.Tensor):
    out = torch.empty_like(x)
    bs0 = 64
    bs1 = 32
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_0 = 64
    _fn_kernel[triton.cdiv(x.size(1), _BLOCK_SIZE_1) * triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    bs0 = 64
    bs1 = 32
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_0 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_l2_grouping_with_register_block_size(self):
        @helion.kernel(
            config={
                "block_sizes": [32, 16],
                "indexing": "block_ptr",
                "l2_grouping": 8,
            }
        )
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            bs0 = hl.register_block_size(1024)
            bs1 = hl.register_block_size(1024)
            for tile0, tile1 in hl.tile(x.size(), block_size=[bs0, bs1]):
                out[tile0, tile1] = x[tile0, tile1] + 1
            return out

        args = (torch.randn([2048, 2048], device=DEVICE),)
        code, result = code_and_output(fn, args)
        torch.testing.assert_close(result, args[0] + 1)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, out_size_0, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_pid_m = tl.cdiv(x_size_0, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(x_size_1, _BLOCK_SIZE_1)
    num_pid_in_group = 8 * num_pid_n
    group_id = tl.program_id(0) // num_pid_in_group
    first_pid_m = group_id * 8
    group_size_m = min(num_pid_m - first_pid_m, 8)
    pid_0 = first_pid_m + tl.program_id(0) % num_pid_in_group % group_size_m
    pid_1 = tl.program_id(0) % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    v_0 = 1.0
    v_1 = load + v_0
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_1, boundary_check=[0, 1])

def fn(x: torch.Tensor):
    out = torch.empty_like(x)
    bs0 = 32
    bs1 = 16
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 16
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0) * triton.cdiv(x.size(1), _BLOCK_SIZE_1),](x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    bs0 = 32
    bs1 = 16
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_multiple_for_loop_1d(self):
        @helion.kernel
        def addToBoth(a, b, c):
            x0, c0 = a
            x1, c1 = b
            x2, c2 = c
            for tile in hl.tile(x0.size()):
                x0[tile] += c0
            for tile in hl.tile(x1.size()):
                x1[tile] += c1
            for tile in hl.tile(x2.size()):
                x2[tile] += c2
            return x0, x1, x2

        constants = [2, 4, 8]
        args = [(torch.ones(5, device=DEVICE), constants[i]) for i in range(3)]
        eager_results = [t + c for t, c in args]

        code, compiled_result = code_and_output(addToBoth, args)

        assert isinstance(compiled_result, tuple)
        for e, c in zip(eager_results, compiled_result, strict=False):
            torch.testing.assert_close(e, c)

        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import triton
import triton.language as tl

@triton.jit
def _addToBoth_kernel(x0, x1, x2, x0_size_0, x1_size_0, x2_size_0, x0_stride_0, x1_stride_0, x2_stride_0, c0, c1, c2, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    pid_shared = tl.program_id(0)
    if pid_shared < tl.cdiv(x0_size_0, _BLOCK_SIZE_0):
        pid_0 = pid_shared
        offset_0 = pid_0 * _BLOCK_SIZE_0
        indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
        mask_0 = indices_0 < x0_size_0
        load = tl.load(x0 + indices_0 * x0_stride_0, mask_0, other=0)
        v_0 = c0.to(tl.float32)
        v_1 = load + v_0
        tl.store(x0 + indices_0 * x0_stride_0, v_1, mask_0)
    elif pid_shared < tl.cdiv(x0_size_0, _BLOCK_SIZE_0) + tl.cdiv(x1_size_0, _BLOCK_SIZE_1):
        pid_shared -= tl.cdiv(x0_size_0, _BLOCK_SIZE_0)
        pid_1 = pid_shared
        offset_1 = pid_1 * _BLOCK_SIZE_1
        indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
        mask_1 = indices_1 < x1_size_0
        load_1 = tl.load(x1 + indices_1 * x1_stride_0, mask_1, other=0)
        v_2 = c1.to(tl.float32)
        v_3 = load_1 + v_2
        tl.store(x1 + indices_1 * x1_stride_0, v_3, mask_1)
    else:
        pid_shared -= tl.cdiv(x0_size_0, _BLOCK_SIZE_0) + tl.cdiv(x1_size_0, _BLOCK_SIZE_1)
        pid_2 = pid_shared
        offset_2 = pid_2 * _BLOCK_SIZE_2
        indices_2 = (offset_2 + tl.arange(0, _BLOCK_SIZE_2)).to(tl.int32)
        mask_2 = indices_2 < x2_size_0
        load_2 = tl.load(x2 + indices_2 * x2_stride_0, mask_2, other=0)
        v_4 = c2.to(tl.float32)
        v_5 = load_2 + v_4
        tl.store(x2 + indices_2 * x2_stride_0, v_5, mask_2)

def addToBoth(a, b, c):
    x0, c0 = a
    x1, c1 = b
    x2, c2 = c
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 8
    _BLOCK_SIZE_2 = 8
    _addToBoth_kernel[triton.cdiv(x0.size(0), _BLOCK_SIZE_0) + triton.cdiv(x1.size(0), _BLOCK_SIZE_1) + triton.cdiv(x2.size(0), _BLOCK_SIZE_2),](x0, x1, x2, x0.size(0), x1.size(0), x2.size(0), x0.stride(0), x1.stride(0), x2.stride(0), c0, c1, c2, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return (x0, x1, x2)

def _addToBoth_make_precompiler(a, b, c):
    x0, c0 = a
    x1, c1 = b
    x2, c2 = c
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 8
    _BLOCK_SIZE_2 = 8
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_addToBoth_kernel)(x0, x1, x2, x0.size(0), x1.size(0), x2.size(0), x0.stride(0), x1.stride(0), x2.stride(0), c0, c1, c2, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_multiple_for_loop_2d(self):
        @helion.kernel
        def addToBoth(a, b, c):
            x0, c0 = a
            x1, c1 = b
            x2, c2 = c

            a_n, a_m = x0.shape
            b_n, b_m = x1.shape
            c_n, c_m = x2.shape

            for tile_n in hl.tile(a_n):
                for tile_m in hl.tile(a_m):
                    x0[tile_n, tile_m] += c0
            for tile_n in hl.tile(b_n):
                for tile_m in hl.tile(b_m):
                    x1[tile_n, tile_m] += c1
            for tile_n in hl.tile(c_n):
                for tile_m in hl.tile(c_m):
                    x2[tile_n, tile_m] += c2
            return x0, x1, x2

        constants = [2, 4, 8]
        args = [(torch.ones(5, 10, device=DEVICE), constants[i]) for i in range(3)]
        eager_results = [t + c for t, c in args]

        code, compiled_result = code_and_output(addToBoth, args)

        assert isinstance(compiled_result, tuple)
        for e, c in zip(eager_results, compiled_result, strict=False):
            torch.testing.assert_close(e, c)

        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import triton
import triton.language as tl

@triton.jit
def _addToBoth_kernel(x0, x1, x2, x0_stride_0, x0_stride_1, x1_stride_0, x1_stride_1, x2_stride_0, x2_stride_1, a_n, a_m, c0, b_n, b_m, c1, c_n, c_m, c2, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr, _BLOCK_SIZE_4: tl.constexpr, _BLOCK_SIZE_5: tl.constexpr):
    pid_shared = tl.program_id(0)
    if pid_shared < tl.cdiv(a_n, _BLOCK_SIZE_0):
        pid_0 = pid_shared
        offset_0 = pid_0 * _BLOCK_SIZE_0
        indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
        mask_0 = indices_0 < a_n
        for offset_1 in tl.range(0, a_m.to(tl.int32), _BLOCK_SIZE_1):
            indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
            mask_1 = indices_1 < a_m
            load = tl.load(x0 + (indices_0[:, None] * x0_stride_0 + indices_1[None, :] * x0_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
            v_0 = c0.to(tl.float32)
            v_1 = load + v_0
            tl.store(x0 + (indices_0[:, None] * x0_stride_0 + indices_1[None, :] * x0_stride_1), v_1, mask_0[:, None] & mask_1[None, :])
    elif pid_shared < tl.cdiv(a_n, _BLOCK_SIZE_0) + tl.cdiv(b_n, _BLOCK_SIZE_2):
        pid_shared -= tl.cdiv(a_n, _BLOCK_SIZE_0)
        pid_1 = pid_shared
        offset_2 = pid_1 * _BLOCK_SIZE_2
        indices_2 = (offset_2 + tl.arange(0, _BLOCK_SIZE_2)).to(tl.int32)
        mask_2 = indices_2 < b_n
        for offset_3 in tl.range(0, b_m.to(tl.int32), _BLOCK_SIZE_3):
            indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
            mask_3 = indices_3 < b_m
            load_1 = tl.load(x1 + (indices_2[:, None] * x1_stride_0 + indices_3[None, :] * x1_stride_1), mask_2[:, None] & mask_3[None, :], other=0)
            v_2 = c1.to(tl.float32)
            v_3 = load_1 + v_2
            tl.store(x1 + (indices_2[:, None] * x1_stride_0 + indices_3[None, :] * x1_stride_1), v_3, mask_2[:, None] & mask_3[None, :])
    else:
        pid_shared -= tl.cdiv(a_n, _BLOCK_SIZE_0) + tl.cdiv(b_n, _BLOCK_SIZE_2)
        pid_2 = pid_shared
        offset_4 = pid_2 * _BLOCK_SIZE_4
        indices_4 = (offset_4 + tl.arange(0, _BLOCK_SIZE_4)).to(tl.int32)
        mask_4 = indices_4 < c_n
        for offset_5 in tl.range(0, c_m.to(tl.int32), _BLOCK_SIZE_5):
            indices_5 = offset_5 + tl.arange(0, _BLOCK_SIZE_5).to(tl.int32)
            mask_5 = indices_5 < c_m
            load_2 = tl.load(x2 + (indices_4[:, None] * x2_stride_0 + indices_5[None, :] * x2_stride_1), mask_4[:, None] & mask_5[None, :], other=0)
            v_4 = c2.to(tl.float32)
            v_5 = load_2 + v_4
            tl.store(x2 + (indices_4[:, None] * x2_stride_0 + indices_5[None, :] * x2_stride_1), v_5, mask_4[:, None] & mask_5[None, :])

def addToBoth(a, b, c):
    x0, c0 = a
    x1, c1 = b
    x2, c2 = c
    a_n, a_m = x0.shape
    b_n, b_m = x1.shape
    c_n, c_m = x2.shape
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 8
    _BLOCK_SIZE_3 = 16
    _BLOCK_SIZE_4 = 8
    _BLOCK_SIZE_5 = 16
    _addToBoth_kernel[triton.cdiv(a_n, _BLOCK_SIZE_0) + triton.cdiv(b_n, _BLOCK_SIZE_2) + triton.cdiv(c_n, _BLOCK_SIZE_4),](x0, x1, x2, x0.stride(0), x0.stride(1), x1.stride(0), x1.stride(1), x2.stride(0), x2.stride(1), a_n, a_m, c0, b_n, b_m, c1, c_n, c_m, c2, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, _BLOCK_SIZE_3, _BLOCK_SIZE_4, _BLOCK_SIZE_5, num_warps=4, num_stages=3)
    return (x0, x1, x2)

def _addToBoth_make_precompiler(a, b, c):
    x0, c0 = a
    x1, c1 = b
    x2, c2 = c
    a_n, a_m = x0.shape
    b_n, b_m = x1.shape
    c_n, c_m = x2.shape
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 8
    _BLOCK_SIZE_3 = 16
    _BLOCK_SIZE_4 = 8
    _BLOCK_SIZE_5 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_addToBoth_kernel)(x0, x1, x2, x0.stride(0), x0.stride(1), x1.stride(0), x1.stride(1), x2.stride(0), x2.stride(1), a_n, a_m, c0, b_n, b_m, c1, c_n, c_m, c2, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, _BLOCK_SIZE_3, _BLOCK_SIZE_4, _BLOCK_SIZE_5, num_warps=4, num_stages=3)""",
        )

    def test_multiple_for_loop_2d_multiple_tile(self):
        @helion.kernel
        def addToBoth(a, b, c):
            x0, c0 = a
            x1, c1 = b
            x2, c2 = c

            a_n, a_m = x0.shape
            b_n, b_m = x1.shape
            c_n, c_m = x2.shape

            for tile_n, tile_m in hl.tile([a_n, a_m]):
                x0[tile_n, tile_m] += c0
            for tile_n, tile_m in hl.tile([b_n, b_m]):
                x1[tile_n, tile_m] += c1
            for tile_n, tile_m in hl.tile([c_n, c_m]):
                x2[tile_n, tile_m] += c2
            return x0, x1, x2

        constants = [2, 4, 8]
        args = [(torch.ones(5, 10, device=DEVICE), constants[i]) for i in range(3)]
        eager_results = [t + c for t, c in args]

        code, compiled_result = code_and_output(addToBoth, args)

        assert isinstance(compiled_result, tuple)
        for e, c in zip(eager_results, compiled_result, strict=False):
            torch.testing.assert_close(e, c)

        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import triton
import triton.language as tl

@triton.jit
def _addToBoth_kernel(x0, x1, x2, x0_stride_0, x0_stride_1, x1_stride_0, x1_stride_1, x2_stride_0, x2_stride_1, a_n, a_m, c0, b_n, b_m, c1, c_n, c_m, c2, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr, _BLOCK_SIZE_4: tl.constexpr, _BLOCK_SIZE_5: tl.constexpr):
    pid_shared = tl.program_id(0)
    if pid_shared < tl.cdiv(a_n, _BLOCK_SIZE_0) * tl.cdiv(a_m, _BLOCK_SIZE_1):
        num_blocks_0 = tl.cdiv(a_n, _BLOCK_SIZE_0)
        pid_0 = pid_shared % num_blocks_0
        pid_1 = pid_shared // num_blocks_0
        offset_0 = pid_0 * _BLOCK_SIZE_0
        indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
        mask_0 = indices_0 < a_n
        offset_1 = pid_1 * _BLOCK_SIZE_1
        indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
        mask_1 = indices_1 < a_m
        load = tl.load(x0 + (indices_0[:, None] * x0_stride_0 + indices_1[None, :] * x0_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
        v_0 = c0.to(tl.float32)
        v_1 = load + v_0
        tl.store(x0 + (indices_0[:, None] * x0_stride_0 + indices_1[None, :] * x0_stride_1), v_1, mask_0[:, None] & mask_1[None, :])
    elif pid_shared < tl.cdiv(a_n, _BLOCK_SIZE_0) * tl.cdiv(a_m, _BLOCK_SIZE_1) + tl.cdiv(b_n, _BLOCK_SIZE_2) * tl.cdiv(b_m, _BLOCK_SIZE_3):
        pid_shared -= tl.cdiv(a_n, _BLOCK_SIZE_0) * tl.cdiv(a_m, _BLOCK_SIZE_1)
        num_blocks_1 = tl.cdiv(b_n, _BLOCK_SIZE_2)
        pid_2 = pid_shared % num_blocks_1
        pid_3 = pid_shared // num_blocks_1
        offset_2 = pid_2 * _BLOCK_SIZE_2
        indices_2 = (offset_2 + tl.arange(0, _BLOCK_SIZE_2)).to(tl.int32)
        mask_2 = indices_2 < b_n
        offset_3 = pid_3 * _BLOCK_SIZE_3
        indices_3 = (offset_3 + tl.arange(0, _BLOCK_SIZE_3)).to(tl.int32)
        mask_3 = indices_3 < b_m
        load_1 = tl.load(x1 + (indices_2[:, None] * x1_stride_0 + indices_3[None, :] * x1_stride_1), mask_2[:, None] & mask_3[None, :], other=0)
        v_2 = c1.to(tl.float32)
        v_3 = load_1 + v_2
        tl.store(x1 + (indices_2[:, None] * x1_stride_0 + indices_3[None, :] * x1_stride_1), v_3, mask_2[:, None] & mask_3[None, :])
    else:
        pid_shared -= tl.cdiv(a_n, _BLOCK_SIZE_0) * tl.cdiv(a_m, _BLOCK_SIZE_1) + tl.cdiv(b_n, _BLOCK_SIZE_2) * tl.cdiv(b_m, _BLOCK_SIZE_3)
        num_blocks_2 = tl.cdiv(c_n, _BLOCK_SIZE_4)
        pid_4 = pid_shared % num_blocks_2
        pid_5 = pid_shared // num_blocks_2
        offset_4 = pid_4 * _BLOCK_SIZE_4
        indices_4 = (offset_4 + tl.arange(0, _BLOCK_SIZE_4)).to(tl.int32)
        mask_4 = indices_4 < c_n
        offset_5 = pid_5 * _BLOCK_SIZE_5
        indices_5 = (offset_5 + tl.arange(0, _BLOCK_SIZE_5)).to(tl.int32)
        mask_5 = indices_5 < c_m
        load_2 = tl.load(x2 + (indices_4[:, None] * x2_stride_0 + indices_5[None, :] * x2_stride_1), mask_4[:, None] & mask_5[None, :], other=0)
        v_4 = c2.to(tl.float32)
        v_5 = load_2 + v_4
        tl.store(x2 + (indices_4[:, None] * x2_stride_0 + indices_5[None, :] * x2_stride_1), v_5, mask_4[:, None] & mask_5[None, :])

def addToBoth(a, b, c):
    x0, c0 = a
    x1, c1 = b
    x2, c2 = c
    a_n, a_m = x0.shape
    b_n, b_m = x1.shape
    c_n, c_m = x2.shape
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 8
    _BLOCK_SIZE_3 = 16
    _BLOCK_SIZE_4 = 8
    _BLOCK_SIZE_5 = 16
    _addToBoth_kernel[triton.cdiv(a_n, _BLOCK_SIZE_0) * triton.cdiv(a_m, _BLOCK_SIZE_1) + triton.cdiv(b_n, _BLOCK_SIZE_2) * triton.cdiv(b_m, _BLOCK_SIZE_3) + triton.cdiv(c_n, _BLOCK_SIZE_4) * triton.cdiv(c_m, _BLOCK_SIZE_5),](x0, x1, x2, x0.stride(0), x0.stride(1), x1.stride(0), x1.stride(1), x2.stride(0), x2.stride(1), a_n, a_m, c0, b_n, b_m, c1, c_n, c_m, c2, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, _BLOCK_SIZE_3, _BLOCK_SIZE_4, _BLOCK_SIZE_5, num_warps=4, num_stages=3)
    return (x0, x1, x2)

def _addToBoth_make_precompiler(a, b, c):
    x0, c0 = a
    x1, c1 = b
    x2, c2 = c
    a_n, a_m = x0.shape
    b_n, b_m = x1.shape
    c_n, c_m = x2.shape
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 8
    _BLOCK_SIZE_3 = 16
    _BLOCK_SIZE_4 = 8
    _BLOCK_SIZE_5 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_addToBoth_kernel)(x0, x1, x2, x0.stride(0), x0.stride(1), x1.stride(0), x1.stride(1), x2.stride(0), x2.stride(1), a_n, a_m, c0, b_n, b_m, c1, c_n, c_m, c2, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, _BLOCK_SIZE_3, _BLOCK_SIZE_4, _BLOCK_SIZE_5, num_warps=4, num_stages=3)""",
        )

    def test_chebyshev_polynomials(self):
        """Test nested loops with sequential computation - Chebyshev polynomials."""

        def chebyshev_torch(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            # x has shape (B, C)
            # w has shape (N, C), where N corresponds to order of Chebyshev polynomials
            # this function combines building Chebyshev polynomials with x and contracting with w, i.e.
            # 1. (B, C) -> (B, N, C)
            # 2. (B, N, C), (N, C) -> (B, C)
            assert w.size(0) >= 2
            # build weighted Chebyshev polynomials
            T0 = torch.ones_like(x)
            T1 = x
            acc = T0 * w[0] + T1 * w[1]
            for n in range(2, w.size(0)):
                T_new = 2 * x * T1 - T0
                acc = acc + T_new * w[n]
                T0 = T1
                T1 = T_new
            return acc

        @helion.kernel(use_default_config=True)
        def chebyshev_kernel(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            B, C = x.shape
            N, C = w.shape
            hl.specialize(N)
            out = torch.zeros((B, C), device=x.device, dtype=x.dtype)
            assert N >= 2, "assume N>= 2 for simplicity"
            for b_tile, c_tile in hl.tile([B, C]):
                in_x = x[b_tile, c_tile]
                T0 = hl.full((b_tile, c_tile), 1.0, x.dtype)
                T1 = in_x
                acc = w[0, c_tile][None, :] * T0 + w[1, c_tile][None, :] * T1
                two_x = 2.0 * in_x
                for order in hl.tile(2, N, block_size=1):
                    new_T = two_x * T1 - T0
                    acc = acc + w[order, c_tile] * new_T
                    T0 = T1
                    T1 = new_T
                out[b_tile, c_tile] = acc
            return out

        # test tensors
        args = (
            torch.randn(123, 64, device=DEVICE, dtype=torch.float32),
            torch.randn(5, 64, device=DEVICE, dtype=torch.float32),
        )

        code, result = code_and_output(chebyshev_kernel, args)
        expected = chebyshev_torch(args[0], args[1])
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _chebyshev_kernel_kernel(x, w, out, out_stride_0, out_stride_1, w_stride_0, w_stride_1, x_stride_0, x_stride_1, B, C, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(B, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < B
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < C
    in_x = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    T0 = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 1.0, tl.float32)
    in_x_0 = in_x
    load_1 = tl.load(w + (0 * w_stride_0 + indices_1 * w_stride_1), mask_1, other=0)
    subscript = load_1[None, :]
    v_0 = subscript * T0
    load_2 = tl.load(w + (1 * w_stride_0 + indices_1 * w_stride_1), mask_1, other=0)
    subscript_1 = load_2[None, :]
    v_1 = subscript_1 * in_x_0
    v_2 = v_0 + v_1
    v_3 = 2.0
    v_4 = in_x * v_3
    for offset_2 in tl.range(2, 5, 1):
        indices_2 = offset_2 + tl.arange(0, 1).to(tl.int32)
        v_4_copy = v_4
        in_x_0_copy = in_x_0
        T0_copy = T0
        v_2_copy = v_2
        v_4_copy_0 = v_4_copy
        in_x_0_copy_0 = in_x_0_copy
        T0_copy_0 = T0_copy
        v_2_copy_0 = v_2_copy
        v_5 = v_4_copy_0 * in_x_0_copy_0
        v_6 = v_5 - T0_copy_0
        load = tl.load(w + (indices_2[:, None] * w_stride_0 + indices_1[None, :] * w_stride_1), mask_1[None, :], other=0)
        v_7 = load * v_6
        v_2 = v_2_copy_0 + v_7
        T0 = in_x_0_copy_0
        in_x_0 = v_6
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_2, mask_0[:, None] & mask_1[None, :])

def chebyshev_kernel(x: torch.Tensor, w: torch.Tensor):
    B, C = x.shape
    N, C = w.shape
    5
    out = torch.zeros((B, C), device=x.device, dtype=x.dtype)
    assert N >= 2, 'assume N>= 2 for simplicity'
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _chebyshev_kernel_kernel[triton.cdiv(B, _BLOCK_SIZE_0) * triton.cdiv(C, _BLOCK_SIZE_1),](x, w, out, out.stride(0), out.stride(1), w.stride(0), w.stride(1), x.stride(0), x.stride(1), B, C, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _chebyshev_kernel_make_precompiler(x: torch.Tensor, w: torch.Tensor):
    B, C = x.shape
    N, C = w.shape
    5
    out = torch.zeros((B, C), device=x.device, dtype=x.dtype)
    assert N >= 2, 'assume N>= 2 for simplicity'
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_chebyshev_kernel_kernel)(x, w, out, out.stride(0), out.stride(1), w.stride(0), w.stride(1), x.stride(0), x.stride(1), B, C, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_loop_unroll1(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
                for i in [1, 2, 3]:
                    out[tile] += i
            return out

        x = torch.randn(4, device=DEVICE)
        code, output = code_and_output(fn, (x,))
        torch.testing.assert_close(output, x + 6)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, x_size_0, out_stride_0, x_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    load = tl.load(x + indices_0 * x_stride_0, mask_0, other=0)
    tl.store(out + indices_0 * out_stride_0, load, mask_0)
    load_1 = tl.load(out + indices_0 * out_stride_0, mask_0, other=0)
    v_0 = 1.0
    v_1 = load_1 + v_0
    tl.store(out + indices_0 * out_stride_0, v_1, mask_0)
    load_2 = tl.load(out + indices_0 * out_stride_0, mask_0, other=0)
    v_2 = 2.0
    v_3 = load_2 + v_2
    tl.store(out + indices_0 * out_stride_0, v_3, mask_0)
    load_3 = tl.load(out + indices_0 * out_stride_0, mask_0, other=0)
    v_4 = 3.0
    v_5 = load_3 + v_4
    tl.store(out + indices_0 * out_stride_0, v_5, mask_0)

def fn(x: torch.Tensor):
    out = torch.zeros_like(x)
    _BLOCK_SIZE_0 = 4
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, out, x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor):
    out = torch.zeros_like(x)
    _BLOCK_SIZE_0 = 4
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_loop_unroll2(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            a = 1
            b = 2
            c = 3
            for tile in hl.tile(x.size()):
                out[tile] = x[tile]
                for i in (a, b, c):
                    out[tile] += i
            return out

        x = torch.randn(4, device=DEVICE)
        code, output = code_and_output(fn, (x,))
        torch.testing.assert_close(output, x + 6)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, x_size_0, out_stride_0, x_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    load = tl.load(x + indices_0 * x_stride_0, mask_0, other=0)
    tl.store(out + indices_0 * out_stride_0, load, mask_0)
    load_1 = tl.load(out + indices_0 * out_stride_0, mask_0, other=0)
    v_0 = 1.0
    v_1 = load_1 + v_0
    tl.store(out + indices_0 * out_stride_0, v_1, mask_0)
    load_2 = tl.load(out + indices_0 * out_stride_0, mask_0, other=0)
    v_2 = 2.0
    v_3 = load_2 + v_2
    tl.store(out + indices_0 * out_stride_0, v_3, mask_0)
    load_3 = tl.load(out + indices_0 * out_stride_0, mask_0, other=0)
    v_4 = 3.0
    v_5 = load_3 + v_4
    tl.store(out + indices_0 * out_stride_0, v_5, mask_0)

def fn(x: torch.Tensor):
    out = torch.zeros_like(x)
    a = 1
    b = 2
    c = 3
    _BLOCK_SIZE_0 = 4
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, out, x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor):
    out = torch.zeros_like(x)
    a = 1
    b = 2
    c = 3
    _BLOCK_SIZE_0 = 4
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_variable_assignment_phi_nodes(self):
        """Test for phi node issue with variable assignments like U1 = two_x.

        This test ensures that simple variable assignments create new variables
        rather than aliases, preventing phi node issues when the source variable
        gets mutated in loops.
        """

        @helion.kernel(use_default_config=True)
        def kernel_with_assignment(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            B, C = x.shape
            N, _ = w.shape
            hl.specialize(N)
            grad_x = torch.zeros_like(x)

            for b_tile, c_tile in hl.tile([B, C]):
                in_x = x[b_tile, c_tile]
                two_x = 2.0 * in_x

                # This assignment should create a new variable, not an alias
                U1 = two_x
                U0 = hl.full((b_tile, c_tile), 1.0, x.dtype)

                acc = w[0, c_tile] * U0 + w[1, c_tile] * U1

                for order in hl.tile(2, N, block_size=1):
                    acc += w[order, c_tile] * U1
                    U_new = two_x * U1 - U0
                    U0 = U1
                    U1 = U_new

                grad_x[b_tile, c_tile] = acc
            return grad_x

        @helion.kernel(use_default_config=True)
        def kernel_without_assignment(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            B, C = x.shape
            N, _ = w.shape
            hl.specialize(N)
            grad_x = torch.zeros_like(x)

            for b_tile, c_tile in hl.tile([B, C]):
                in_x = x[b_tile, c_tile]
                two_x = 2.0 * in_x

                # Direct use without assignment
                U1 = 2.0 * in_x
                U0 = hl.full((b_tile, c_tile), 1.0, x.dtype)

                acc = w[0, c_tile] * U0 + w[1, c_tile] * U1

                for order in hl.tile(2, N, block_size=1):
                    acc += w[order, c_tile] * U1
                    U_new = two_x * U1 - U0
                    U0 = U1
                    U1 = U_new

                grad_x[b_tile, c_tile] = acc
            return grad_x

        # Test with small tensor
        x = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        w = torch.randn(5, 8, device=DEVICE, dtype=torch.float32)

        code1, result1 = code_and_output(kernel_with_assignment, (x, w))
        code2, result2 = code_and_output(kernel_without_assignment, (x, w))

        # Both should produce identical results
        torch.testing.assert_close(result1, result2, rtol=1e-5, atol=1e-5)

    def test_range_unroll_factors(self):
        @helion.kernel()
        def nested_loop_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            # Outer loop becomes grid (no tl.range)
            for tile_outer in hl.tile(x.size(0)):
                # Inner loop becomes device loop with tl.range
                for tile_inner in hl.tile(x.size(1)):
                    out[tile_outer, tile_inner] = x[tile_outer, tile_inner] + 1
            return out

        # Test configuration validation - that range_unroll_factors works
        args = (torch.randn([64, 32], device=DEVICE),)

        # Test with range_unroll_factors = [0] (no unrolling for device loop)
        code0, result0 = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_unroll_factors=[0]
        )

        # Test with range_unroll_factors = [2] (unroll factor 2 for device loop)
        code2, result2 = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_unroll_factors=[2]
        )

        torch.testing.assert_close(result0, result2)
        torch.testing.assert_close(result0, args[0] + 1)
        self.assertNotEqual(code0, code2)
        self.assertNotIn("loop_unroll_factor", code0)
        self.assertExpectedInline(
            code2,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _nested_loop_kernel_kernel(x, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    for offset_1 in tl.range(0, x_size_1.to(tl.int32), _BLOCK_SIZE_1, loop_unroll_factor=2):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < x_size_1
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
        v_0 = 1.0
        v_1 = load + v_0
        tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_1, mask_0[:, None] & mask_1[None, :])

def nested_loop_kernel(x: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 16
    _nested_loop_kernel_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _nested_loop_kernel_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_nested_loop_kernel_kernel)(x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_range_num_stages(self):
        @helion.kernel()
        def nested_loop_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            # Outer loop becomes grid (no tl.range)
            for tile_outer in hl.tile(x.size(0)):
                # Inner loop becomes device loop with tl.range
                for tile_inner in hl.tile(x.size(1)):
                    out[tile_outer, tile_inner] = x[tile_outer, tile_inner] + 1
            return out

        # Test configuration validation - that range_num_stages works
        args = (torch.randn([64, 32], device=DEVICE),)

        # Test with range_num_stages = [0] (no num_stages for device loop)
        code0, result0 = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_num_stages=[0]
        )

        # Test with range_num_stages = [3] (num_stages=3 for device loop)
        code3, result3 = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_num_stages=[3]
        )

        torch.testing.assert_close(result0, result3)
        torch.testing.assert_close(result0, args[0] + 1)
        self.assertNotEqual(code0, code3)
        # Check that range_num_stages parameter appears in tl.range call
        self.assertNotIn(
            "tl.range(0, x_size_1.to(tl.int32), _BLOCK_SIZE_1, num_stages=", code0
        )
        self.assertIn(
            "tl.range(0, x_size_1.to(tl.int32), _BLOCK_SIZE_1, num_stages=3)", code3
        )

    def test_range_multi_buffers(self):
        @helion.kernel()
        def nested_loop_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            # Outer loop becomes grid (no tl.range)
            for tile_outer in hl.tile(x.size(0)):
                # Inner loop becomes device loop with tl.range
                for tile_inner in hl.tile(x.size(1)):
                    out[tile_outer, tile_inner] = x[tile_outer, tile_inner] + 1
            return out

        # Test configuration validation - that range_multi_buffers works
        args = (torch.randn([64, 32], device=DEVICE),)

        # Test with range_multi_buffers = [None] (no disallow_acc_multi_buffer for device loop)
        code_none, result_none = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_multi_buffers=[None]
        )

        # Test with range_multi_buffers = [True] (disallow_acc_multi_buffer=False for device loop)
        code_true, result_true = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_multi_buffers=[True]
        )

        # Test with range_multi_buffers = [False] (disallow_acc_multi_buffer=True for device loop)
        code_false, result_false = code_and_output(
            nested_loop_kernel, args, block_sizes=[32, 16], range_multi_buffers=[False]
        )

        torch.testing.assert_close(result_none, result_true)
        torch.testing.assert_close(result_none, result_false)
        torch.testing.assert_close(result_none, args[0] + 1)
        self.assertNotEqual(code_none, code_true)
        self.assertNotEqual(code_none, code_false)
        self.assertNotEqual(code_true, code_false)
        # Check that disallow_acc_multi_buffer parameter appears in tl.range call
        self.assertNotIn("disallow_acc_multi_buffer", code_none)
        self.assertIn("disallow_acc_multi_buffer=False", code_true)
        self.assertIn("disallow_acc_multi_buffer=True", code_false)


if __name__ == "__main__":
    unittest.main()
