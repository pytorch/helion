from __future__ import annotations

from pathlib import Path
import unittest

from expecttest import TestCase
import torch

from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import import_path

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")


class TestGenerateAst(TestCase):
    maxDiff = 16384

    def test_add1d(self):
        args = (torch.randn([4096], device=DEVICE), torch.randn([4096], device=DEVICE))
        code, result = code_and_output(basic_kernels.add, args, block_size=1024)
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x, y, out, x_size_0, out_stride_0, x_stride_0, y_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    load = tl.load(x + indices_0 * x_stride_0, mask_0, other=0)
    load_1 = tl.load(y + indices_0 * y_stride_0, mask_0, other=0)
    v_0 = load + load_1
    tl.store(out + indices_0 * out_stride_0, v_0, mask_0)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 1024
    _add_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, y, out, x.size(0), out.stride(0), x.stride(0), y.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _add_make_precompiler(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 1024
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_add_kernel)(x, y, out, x.size(0), out.stride(0), x.stride(0), y.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_add2d(self):
        args = (
            torch.randn([100, 500], device=DEVICE),
            torch.randn([100, 500], device=DEVICE),
        )
        code, result = code_and_output(basic_kernels.add, args, block_size=1024)
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x, y, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, _BLOCK_SIZE_0_1: tl.constexpr):
    offsets_0_1 = tl.program_id(0) * _BLOCK_SIZE_0_1 + tl.arange(0, _BLOCK_SIZE_0_1).to(tl.int32)
    indices_1 = offsets_0_1 % x_size_1
    indices_0 = offsets_0_1 // x_size_1
    mask_0_1 = offsets_0_1 < x_size_0 * x_size_1
    load = tl.load(x + (indices_0 * x_stride_0 + indices_1 * x_stride_1), mask_0_1, other=0)
    load_1 = tl.load(y + (indices_0 * y_stride_0 + indices_1 * y_stride_1), mask_0_1, other=0)
    v_0 = load + load_1
    tl.store(out + (indices_0 * out_stride_0 + indices_1 * out_stride_1), v_0, mask_0_1)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1 = 1024
    _add_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE_0_1), 1, 1](x, y, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)
    return out

def _add_make_precompiler(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1 = 1024
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_add_kernel)(x, y, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)""",
        )

    def test_add2d_loop_order(self):
        args = (
            torch.randn([100, 500], device=DEVICE),
            torch.randn([100, 500], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_size=1024, loop_order=(1, 0)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x, y, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, _BLOCK_SIZE_0_1: tl.constexpr):
    offsets_0_1 = tl.program_id(0) * _BLOCK_SIZE_0_1 + tl.arange(0, _BLOCK_SIZE_0_1).to(tl.int32)
    indices_0 = offsets_0_1 % x_size_0
    indices_1 = offsets_0_1 // x_size_0
    mask_0_1 = offsets_0_1 < x_size_0 * x_size_1
    load = tl.load(x + (indices_0 * x_stride_0 + indices_1 * x_stride_1), mask_0_1, other=0)
    load_1 = tl.load(y + (indices_0 * y_stride_0 + indices_1 * y_stride_1), mask_0_1, other=0)
    v_0 = load + load_1
    tl.store(out + (indices_0 * out_stride_0 + indices_1 * out_stride_1), v_0, mask_0_1)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1 = 1024
    _add_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE_0_1), 1, 1](x, y, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)
    return out

def _add_make_precompiler(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1 = 1024
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_add_kernel)(x, y, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)""",
        )

    def test_add3d(self):
        args = (
            torch.randn([100, 500, 10], device=DEVICE),
            torch.randn([100, 500, 10], device=DEVICE),
        )
        code, result = code_and_output(basic_kernels.add, args, block_size=1024)
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x, y, out, x_size_0, x_size_1, x_size_2, out_stride_0, out_stride_1, out_stride_2, x_stride_0, x_stride_1, x_stride_2, y_stride_0, y_stride_1, y_stride_2, _BLOCK_SIZE_0_1_2: tl.constexpr):
    offsets_0_1_2 = tl.program_id(0) * _BLOCK_SIZE_0_1_2 + tl.arange(0, _BLOCK_SIZE_0_1_2).to(tl.int32)
    indices_2 = offsets_0_1_2 % x_size_2
    indices_1 = offsets_0_1_2 // x_size_2 % x_size_1
    indices_0 = offsets_0_1_2 // (x_size_1 * x_size_2)
    mask_0_1_2 = offsets_0_1_2 < x_size_0 * x_size_1 * x_size_2
    load = tl.load(x + (indices_0 * x_stride_0 + indices_1 * x_stride_1 + indices_2 * x_stride_2), mask_0_1_2, other=0)
    load_1 = tl.load(y + (indices_0 * y_stride_0 + indices_1 * y_stride_1 + indices_2 * y_stride_2), mask_0_1_2, other=0)
    v_0 = load + load_1
    tl.store(out + (indices_0 * out_stride_0 + indices_1 * out_stride_1 + indices_2 * out_stride_2), v_0, mask_0_1_2)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1_2 = 1024
    _add_kernel[triton.cdiv(x.size(0) * x.size(1) * x.size(2), _BLOCK_SIZE_0_1_2), 1, 1](x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_0_1_2, num_warps=4, num_stages=3)
    return out

def _add_make_precompiler(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1_2 = 1024
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_add_kernel)(x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_0_1_2, num_warps=4, num_stages=3)""",
        )

    def test_add3d_xy_grid(self):
        args = (
            torch.randn([100, 500, 10], device=DEVICE),
            torch.randn([100, 500, 10], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_size=[16, 16, 16], use_yz_grid=True
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x, y, out, x_size_0, x_size_1, x_size_2, out_stride_0, out_stride_1, out_stride_2, x_stride_0, x_stride_1, x_stride_2, y_stride_0, y_stride_1, y_stride_2, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    pid_2 = tl.program_id(2)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < x_size_1
    offset_2 = pid_2 * _BLOCK_SIZE_2
    indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
    mask_2 = indices_2 < x_size_2
    load = tl.load(x + (indices_0[:, None, None] * x_stride_0 + indices_1[None, :, None] * x_stride_1 + indices_2[None, None, :] * x_stride_2), mask_0[:, None, None] & mask_1[None, :, None] & mask_2[None, None, :], other=0)
    load_1 = tl.load(y + (indices_0[:, None, None] * y_stride_0 + indices_1[None, :, None] * y_stride_1 + indices_2[None, None, :] * y_stride_2), mask_0[:, None, None] & mask_1[None, :, None] & mask_2[None, None, :], other=0)
    v_0 = load + load_1
    tl.store(out + (indices_0[:, None, None] * out_stride_0 + indices_1[None, :, None] * out_stride_1 + indices_2[None, None, :] * out_stride_2), v_0, mask_0[:, None, None] & mask_1[None, :, None] & mask_2[None, None, :])

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _add_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0), triton.cdiv(x.size(1), _BLOCK_SIZE_1), triton.cdiv(x.size(2), _BLOCK_SIZE_2)](x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _add_make_precompiler(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_add_kernel)(x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_add3d_reorder(self):
        args = (
            torch.randn([100, 500, 10], device=DEVICE),
            torch.randn([100, 500, 10], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_size=1024, loop_order=(2, 0, 1)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x, y, out, x_size_0, x_size_1, x_size_2, out_stride_0, out_stride_1, out_stride_2, x_stride_0, x_stride_1, x_stride_2, y_stride_0, y_stride_1, y_stride_2, _BLOCK_SIZE_0_1_2: tl.constexpr):
    offsets_0_1_2 = tl.program_id(0) * _BLOCK_SIZE_0_1_2 + tl.arange(0, _BLOCK_SIZE_0_1_2).to(tl.int32)
    indices_1 = offsets_0_1_2 % x_size_1
    indices_0 = offsets_0_1_2 // x_size_1 % x_size_0
    indices_2 = offsets_0_1_2 // (x_size_0 * x_size_1)
    mask_0_1_2 = offsets_0_1_2 < x_size_0 * x_size_1 * x_size_2
    load = tl.load(x + (indices_0 * x_stride_0 + indices_1 * x_stride_1 + indices_2 * x_stride_2), mask_0_1_2, other=0)
    load_1 = tl.load(y + (indices_0 * y_stride_0 + indices_1 * y_stride_1 + indices_2 * y_stride_2), mask_0_1_2, other=0)
    v_0 = load + load_1
    tl.store(out + (indices_0 * out_stride_0 + indices_1 * out_stride_1 + indices_2 * out_stride_2), v_0, mask_0_1_2)

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1_2 = 1024
    _add_kernel[triton.cdiv(x.size(0) * x.size(1) * x.size(2), _BLOCK_SIZE_0_1_2), 1, 1](x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_0_1_2, num_warps=4, num_stages=3)
    return out

def _add_make_precompiler(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1_2 = 1024
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_add_kernel)(x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_0_1_2, num_warps=4, num_stages=3)""",
        )

    def test_add_tilend0(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_size=[8, 16, 32], loop_order=(0, 1, 2)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x, y, out, x_size_0, x_size_1, x_size_2, out_stride_0, out_stride_1, out_stride_2, x_stride_0, x_stride_1, x_stride_2, y_stride_0, y_stride_1, y_stride_2, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_0, _BLOCK_SIZE_0)
    num_blocks_1 = tl.cdiv(x_size_1, _BLOCK_SIZE_1)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0 % num_blocks_1
    pid_2 = tl.program_id(0) // (num_blocks_0 * num_blocks_1)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < x_size_1
    offset_2 = pid_2 * _BLOCK_SIZE_2
    indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
    mask_2 = indices_2 < x_size_2
    load = tl.load(x + (indices_0[:, None, None] * x_stride_0 + indices_1[None, :, None] * x_stride_1 + indices_2[None, None, :] * x_stride_2), mask_0[:, None, None] & mask_1[None, :, None] & mask_2[None, None, :], other=0)
    load_1 = tl.load(y + (indices_0[:, None, None] * y_stride_0 + indices_1[None, :, None] * y_stride_1 + indices_2[None, None, :] * y_stride_2), mask_0[:, None, None] & mask_1[None, :, None] & mask_2[None, None, :], other=0)
    v_0 = load + load_1
    tl.store(out + (indices_0[:, None, None] * out_stride_0 + indices_1[None, :, None] * out_stride_1 + indices_2[None, None, :] * out_stride_2), v_0, mask_0[:, None, None] & mask_1[None, :, None] & mask_2[None, None, :])

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 32
    _add_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0) * triton.cdiv(x.size(1), _BLOCK_SIZE_1) * triton.cdiv(x.size(2), _BLOCK_SIZE_2),](x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _add_make_precompiler(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_add_kernel)(x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_add_tilend1(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_size=[8, 16, 32], loop_order=(2, 1, 0)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x, y, out, x_size_0, x_size_1, x_size_2, out_stride_0, out_stride_1, out_stride_2, x_stride_0, x_stride_1, x_stride_2, y_stride_0, y_stride_1, y_stride_2, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_2, _BLOCK_SIZE_2)
    num_blocks_1 = tl.cdiv(x_size_1, _BLOCK_SIZE_1)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0 % num_blocks_1
    pid_2 = tl.program_id(0) // (num_blocks_0 * num_blocks_1)
    offset_2 = pid_0 * _BLOCK_SIZE_2
    indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
    mask_2 = indices_2 < x_size_2
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < x_size_1
    offset_0 = pid_2 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    load = tl.load(x + (indices_0[:, None, None] * x_stride_0 + indices_1[None, :, None] * x_stride_1 + indices_2[None, None, :] * x_stride_2), mask_0[:, None, None] & mask_1[None, :, None] & mask_2[None, None, :], other=0)
    load_1 = tl.load(y + (indices_0[:, None, None] * y_stride_0 + indices_1[None, :, None] * y_stride_1 + indices_2[None, None, :] * y_stride_2), mask_0[:, None, None] & mask_1[None, :, None] & mask_2[None, None, :], other=0)
    v_0 = load + load_1
    tl.store(out + (indices_0[:, None, None] * out_stride_0 + indices_1[None, :, None] * out_stride_1 + indices_2[None, None, :] * out_stride_2), v_0, mask_0[:, None, None] & mask_1[None, :, None] & mask_2[None, None, :])

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_2 = 32
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_0 = 8
    _add_kernel[triton.cdiv(x.size(2), _BLOCK_SIZE_2) * triton.cdiv(x.size(1), _BLOCK_SIZE_1) * triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_2, _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _add_make_precompiler(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_2 = 32
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_0 = 8
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_add_kernel)(x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_2, _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_add_tilend2(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_size=[1, 32, 32], loop_order=(0, 1, 2)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x, y, out, x_size_0, x_size_1, x_size_2, out_stride_0, out_stride_1, out_stride_2, x_stride_0, x_stride_1, x_stride_2, y_stride_0, y_stride_1, y_stride_2, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_blocks_0 = x_size_0
    num_blocks_1 = tl.cdiv(x_size_1, _BLOCK_SIZE_1)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0 % num_blocks_1
    pid_2 = tl.program_id(0) // (num_blocks_0 * num_blocks_1)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < x_size_1
    offset_2 = pid_2 * _BLOCK_SIZE_2
    indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
    mask_2 = indices_2 < x_size_2
    load = tl.load(x + (indices_0[:, None, None] * x_stride_0 + indices_1[None, :, None] * x_stride_1 + indices_2[None, None, :] * x_stride_2), mask_1[None, :, None] & mask_2[None, None, :], other=0)
    load_1 = tl.load(y + (indices_0[:, None, None] * y_stride_0 + indices_1[None, :, None] * y_stride_1 + indices_2[None, None, :] * y_stride_2), mask_1[None, :, None] & mask_2[None, None, :], other=0)
    v_0 = load + load_1
    tl.store(out + (indices_0[:, None, None] * out_stride_0 + indices_1[None, :, None] * out_stride_1 + indices_2[None, None, :] * out_stride_2), v_0, mask_1[None, :, None] & mask_2[None, None, :])

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_2 = 32
    _add_kernel[x.size(0) * triton.cdiv(x.size(1), _BLOCK_SIZE_1) * triton.cdiv(x.size(2), _BLOCK_SIZE_2),](x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _add_make_precompiler(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_2 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_add_kernel)(x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_add_tilend3(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add,
            args,
            block_size=[1, 32, 1],
            loop_order=(0, 2, 1),
            num_warps=8,
            num_stages=1,
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x, y, out, x_size_0, x_size_1, x_size_2, out_stride_0, out_stride_1, out_stride_2, x_stride_0, x_stride_1, x_stride_2, y_stride_0, y_stride_1, y_stride_2, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = x_size_0
    num_blocks_1 = x_size_2
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0 % num_blocks_1
    pid_2 = tl.program_id(0) // (num_blocks_0 * num_blocks_1)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    offset_2 = pid_1
    indices_2 = offset_2 + tl.zeros([1], tl.int32)
    offset_1 = pid_2 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < x_size_1
    load = tl.load(x + (indices_0[:, None, None] * x_stride_0 + indices_1[None, :, None] * x_stride_1 + indices_2[None, None, :] * x_stride_2), mask_1[None, :, None], other=0)
    load_1 = tl.load(y + (indices_0[:, None, None] * y_stride_0 + indices_1[None, :, None] * y_stride_1 + indices_2[None, None, :] * y_stride_2), mask_1[None, :, None], other=0)
    v_0 = load + load_1
    tl.store(out + (indices_0[:, None, None] * out_stride_0 + indices_1[None, :, None] * out_stride_1 + indices_2[None, None, :] * out_stride_2), v_0, mask_1[None, :, None])

def add(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_1 = 32
    _add_kernel[x.size(0) * x.size(2) * triton.cdiv(x.size(1), _BLOCK_SIZE_1),](x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_1, num_warps=8, num_stages=1)
    return out

def _add_make_precompiler(x, y):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_add_kernel)(x, y, out, x.size(0), x.size(1), x.size(2), out.stride(0), out.stride(1), out.stride(2), x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2), _BLOCK_SIZE_1, num_warps=8, num_stages=1)""",
        )

    def test_torch_ops_pointwise(self):
        args = (
            torch.randn([1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.torch_ops_pointwise,
            args,
            block_size=128,
        )
        torch.testing.assert_close(
            result, torch.sigmoid(torch.add(torch.sin(args[0]), torch.cos(args[1])))
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
def _torch_ops_pointwise_kernel(x, y, out, x_size_0, out_stride_0, x_stride_0, y_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    load = tl.load(x + indices_0 * x_stride_0, mask_0, other=0)
    v_0 = tl_math.sin(load)
    load_1 = tl.load(y + indices_0 * y_stride_0, mask_0, other=0)
    v_1 = tl_math.cos(load_1)
    v_2 = v_0 + v_1
    v_3 = tl.sigmoid(v_2)
    tl.store(out + indices_0 * out_stride_0, v_3, mask_0)

def torch_ops_pointwise(x, y):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 128
    _torch_ops_pointwise_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, y, out, x.size(0), out.stride(0), x.stride(0), y.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _torch_ops_pointwise_make_precompiler(x, y):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 128
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_torch_ops_pointwise_kernel)(x, y, out, x.size(0), out.stride(0), x.stride(0), y.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_hl_zeros_usage(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.hl_zeros_usage,
            args,
            block_size=[32, 32],
        )
        torch.testing.assert_close(result, args[0] * 2)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _hl_zeros_usage_kernel(x, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < x_size_1
    tmp = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    v_0 = tmp + load
    load_1 = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    v_1 = v_0 + load_1
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_1, mask_0[:, None] & mask_1[None, :])

def hl_zeros_usage(x: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _hl_zeros_usage_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0) * triton.cdiv(x.size(1), _BLOCK_SIZE_1),](x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _hl_zeros_usage_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_hl_zeros_usage_kernel)(x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_hl_full_usage(self):
        args = (torch.randn([512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.hl_full_usage,
            args,
            block_size=128,
        )
        torch.testing.assert_close(result, args[0] * 2 + 1)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _hl_full_usage_kernel(x, out, x_size_0, out_stride_0, x_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    tmp = tl.full([_BLOCK_SIZE_0], 1, tl.float32)
    load = tl.load(x + indices_0 * x_stride_0, mask_0, other=0)
    v_0 = tmp + load
    load_1 = tl.load(x + indices_0 * x_stride_0, mask_0, other=0)
    v_1 = v_0 + load_1
    tl.store(out + indices_0 * out_stride_0, v_1, mask_0)

def hl_full_usage(x: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 128
    _hl_full_usage_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, out, x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _hl_full_usage_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 128
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_hl_full_usage_kernel)(x, out, x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_hl_zeros_flat(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.hl_zeros_usage,
            args,
            block_size=128,
        )
        torch.testing.assert_close(result, args[0] * 2)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _hl_zeros_usage_kernel(x, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0_1: tl.constexpr):
    offsets_0_1 = tl.program_id(0) * _BLOCK_SIZE_0_1 + tl.arange(0, _BLOCK_SIZE_0_1).to(tl.int32)
    indices_1 = offsets_0_1 % x_size_1
    indices_0 = offsets_0_1 // x_size_1
    mask_0_1 = offsets_0_1 < x_size_0 * x_size_1
    tmp = tl.full([_BLOCK_SIZE_0_1], 0.0, tl.float32)
    load = tl.load(x + (indices_0 * x_stride_0 + indices_1 * x_stride_1), mask_0_1, other=0)
    v_0 = tmp + load
    load_1 = tl.load(x + (indices_0 * x_stride_0 + indices_1 * x_stride_1), mask_0_1, other=0)
    v_1 = v_0 + load_1
    tl.store(out + (indices_0 * out_stride_0 + indices_1 * out_stride_1), v_1, mask_0_1)

def hl_zeros_usage(x: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1 = 128
    _hl_zeros_usage_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE_0_1), 1, 1](x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)
    return out

def _hl_zeros_usage_make_precompiler(x: torch.Tensor):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1 = 128
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_hl_zeros_usage_kernel)(x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)""",
        )


if __name__ == "__main__":
    unittest.main()
