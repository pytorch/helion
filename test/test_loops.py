from __future__ import annotations

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
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < n
    for offset_1 in range(0, m, _BLOCK_SIZE_1):
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
            block_sizes=[[1], [8, 8, 8]],
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
def _device_loop_3d_kernel(x, out, out_stride_0, out_stride_1, out_stride_2, out_stride_3, x_stride_0, x_stride_1, x_stride_2, x_stride_3, d, c, b, _BLOCK_SIZE_3: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    for offset_1 in range(0, b, _BLOCK_SIZE_1):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < b
        for offset_2 in range(0, c, _BLOCK_SIZE_2):
            indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
            mask_2 = indices_2 < c
            for offset_3 in range(0, d, _BLOCK_SIZE_3):
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
    _device_loop_3d_kernel[a,](x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), d, c, b, _BLOCK_SIZE_3, _BLOCK_SIZE_2, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _device_loop_3d_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_3 = 8
    _BLOCK_SIZE_2 = 8
    _BLOCK_SIZE_1 = 8
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_device_loop_3d_kernel)(x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), d, c, b, _BLOCK_SIZE_3, _BLOCK_SIZE_2, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_3d_device_loop1(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[[2], [8, 4, 1]],
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
def _device_loop_3d_kernel(x, out, out_stride_0, out_stride_1, out_stride_2, out_stride_3, x_stride_0, x_stride_1, x_stride_2, x_stride_3, a, d, b, c, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a
    for offset_2 in range(0, c, _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        mask_2 = indices_2 < c
        for offset_1 in range(0, b, _BLOCK_SIZE_1):
            indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
            mask_1 = indices_1 < b
            for offset_3 in range(0, d, 1):
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
    _device_loop_3d_kernel[triton.cdiv(a, _BLOCK_SIZE_0),](x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), a, d, b, c, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _device_loop_3d_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_0 = 2
    _BLOCK_SIZE_1 = 8
    _BLOCK_SIZE_2 = 4
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_device_loop_3d_kernel)(x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), a, d, b, c, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_3d_device_loop2(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[4, 128],
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
def _device_loop_3d_kernel(x, out, out_stride_0, out_stride_1, out_stride_2, out_stride_3, x_stride_0, x_stride_1, x_stride_2, x_stride_3, a, c, b, d, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1_2_3: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a
    for lid_1_2_3 in range(tl.cdiv(b * c * d, _BLOCK_SIZE_1_2_3)):
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
    _device_loop_3d_kernel[triton.cdiv(a, _BLOCK_SIZE_0),](x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), a, c, b, d, _BLOCK_SIZE_0, _BLOCK_SIZE_1_2_3, num_warps=4, num_stages=3)
    return out

def _device_loop_3d_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_0 = 4
    _BLOCK_SIZE_1_2_3 = 128
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_device_loop_3d_kernel)(x, out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), a, c, b, d, _BLOCK_SIZE_0, _BLOCK_SIZE_1_2_3, num_warps=4, num_stages=3)""",
        )

    def test_3d_device_loop3(self):
        args = (torch.randn([128, 128, 128, 128], device=DEVICE),)
        code, result = code_and_output(
            device_loop_3d,
            args,
            block_sizes=[[2], [8, 4, 1]],
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
def _device_loop_3d_kernel(x, out, out_size_0, out_size_1, out_size_2, out_size_3, x_size_0, x_size_1, x_size_2, x_size_3, out_stride_0, out_stride_1, out_stride_2, out_stride_3, x_stride_0, x_stride_1, x_stride_2, x_stride_3, c, b, d, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    for offset_3 in range(0, d, 1):
        for offset_1 in range(0, b, _BLOCK_SIZE_1):
            for offset_2 in range(0, c, _BLOCK_SIZE_2):
                load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1, x_size_2, x_size_3], [x_stride_0, x_stride_1, x_stride_2, x_stride_3], [offset_0, offset_1, offset_2, offset_3], [_BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, 1], [3, 2, 1, 0]), boundary_check=[0, 1, 2, 3], padding_option='zero')
                v_0 = tl_math.sin(load)
                tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1, out_size_2, out_size_3], [out_stride_0, out_stride_1, out_stride_2, out_stride_3], [offset_0, offset_1, offset_2, offset_3], [_BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, 1], [3, 2, 1, 0]), v_0, boundary_check=[0, 1, 2, 3])

def device_loop_3d(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_0 = 2
    _BLOCK_SIZE_2 = 4
    _BLOCK_SIZE_1 = 8
    _device_loop_3d_kernel[triton.cdiv(a, _BLOCK_SIZE_0),](x, out, out.size(0), out.size(1), out.size(2), out.size(3), x.size(0), x.size(1), x.size(2), x.size(3), out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), c, b, d, _BLOCK_SIZE_0, _BLOCK_SIZE_2, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _device_loop_3d_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    a, b, c, d = x.shape
    _BLOCK_SIZE_0 = 2
    _BLOCK_SIZE_2 = 4
    _BLOCK_SIZE_1 = 8
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_device_loop_3d_kernel)(x, out, out.size(0), out.size(1), out.size(2), out.size(3), x.size(0), x.size(1), x.size(2), x.size(3), out.stride(0), out.stride(1), out.stride(2), out.stride(3), x.stride(0), x.stride(1), x.stride(2), x.stride(3), c, b, d, _BLOCK_SIZE_0, _BLOCK_SIZE_2, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
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
    for offset_2 in range(0, c, _BLOCK_SIZE_2):
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

    @unittest.skip("TODO(jansel): fix this")
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
        self.assertExpectedInline(code, """""")
