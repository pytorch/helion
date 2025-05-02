from __future__ import annotations

import unittest

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


@helion.kernel
def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    for tile0, tile1 in hl.tile(out0.size()):
        out0[tile0, tile1] = a[tile0, tile1] + b[tile0, None]
        out1[tile0, tile1] = a[tile0, tile1] + b[None, tile1]
    return out0, out1


def broadcast_fn_ref(a, b):
    out0 = a + b[:, None]
    out1 = a + b[None, :]
    return out0, out1


def _check_broadcast_fn(**config):
    args = [torch.randn(512, 512, device=DEVICE), torch.randn(512, device=DEVICE)]
    code, (out0, out1) = code_and_output(broadcast_fn, args, **config)
    ref0, ref1 = broadcast_fn_ref(*args)
    torch.testing.assert_close(out0, ref0)
    torch.testing.assert_close(out1, ref1)
    return code


class TestBroadcasting(TestCase):
    maxDiff = 16384

    def test_broadcast_no_flatten(self):
        args = [torch.randn(512, 512, device=DEVICE), torch.randn(512, device=DEVICE)]
        assert (
            not broadcast_fn.bind(args).config_spec.block_size_specs[0].allow_flattened
        )

    def test_broadcast1(self):
        code = _check_broadcast_fn(
            block_size=[16, 8],
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _broadcast_fn_kernel(a, b, out0, out1, a_size_0, a_size_1, a_stride_0, a_stride_1, b_stride_0, out0_stride_0, out0_stride_1, out1_stride_0, out1_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(a_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < a_size_1
    load = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_1 = tl.load(b + indices_0[:, None] * b_stride_0, mask_0[:, None], other=0)
    v_0 = load + load_1
    tl.store(out0 + (indices_0[:, None] * out0_stride_0 + indices_1[None, :] * out0_stride_1), v_0, mask_0[:, None] & mask_1[None, :])
    load_2 = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_3 = tl.load(b + indices_1[None, :] * b_stride_0, mask_1[None, :], other=0)
    v_1 = load_2 + load_3
    tl.store(out1 + (indices_0[:, None] * out1_stride_0 + indices_1[None, :] * out1_stride_1), v_1, mask_0[:, None] & mask_1[None, :])

def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 8
    _broadcast_fn_kernel[triton.cdiv(a.size(0), _BLOCK_SIZE_0) * triton.cdiv(a.size(1), _BLOCK_SIZE_1),](a, b, out0, out1, a.size(0), a.size(1), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return (out0, out1)

def _broadcast_fn_make_precompiler(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 8
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_broadcast_fn_kernel)(a, b, out0, out1, a.size(0), a.size(1), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_broadcast2(self):
        code = _check_broadcast_fn(block_size=[16, 8], loop_order=(1, 0))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _broadcast_fn_kernel(a, b, out0, out1, a_size_0, a_size_1, a_stride_0, a_stride_1, b_stride_0, out0_stride_0, out0_stride_1, out1_stride_0, out1_stride_1, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr):
    num_blocks_0 = tl.cdiv(a_size_1, _BLOCK_SIZE_1)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_1 = pid_0 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < a_size_1
    offset_0 = pid_1 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a_size_0
    load = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_1 = tl.load(b + indices_0[:, None] * b_stride_0, mask_0[:, None], other=0)
    v_0 = load + load_1
    tl.store(out0 + (indices_0[:, None] * out0_stride_0 + indices_1[None, :] * out0_stride_1), v_0, mask_0[:, None] & mask_1[None, :])
    load_2 = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_3 = tl.load(b + indices_1[None, :] * b_stride_0, mask_1[None, :], other=0)
    v_1 = load_2 + load_3
    tl.store(out1 + (indices_0[:, None] * out1_stride_0 + indices_1[None, :] * out1_stride_1), v_1, mask_0[:, None] & mask_1[None, :])

def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    _BLOCK_SIZE_1 = 8
    _BLOCK_SIZE_0 = 16
    _broadcast_fn_kernel[triton.cdiv(a.size(1), _BLOCK_SIZE_1) * triton.cdiv(a.size(0), _BLOCK_SIZE_0),](a, b, out0, out1, a.size(0), a.size(1), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return (out0, out1)

def _broadcast_fn_make_precompiler(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    _BLOCK_SIZE_1 = 8
    _BLOCK_SIZE_0 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_broadcast_fn_kernel)(a, b, out0, out1, a.size(0), a.size(1), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_broadcast3(self):
        code = _check_broadcast_fn(
            block_size=[64, 1],
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _broadcast_fn_kernel(a, b, out0, out1, a_size_0, a_stride_0, a_stride_1, b_stride_0, out0_stride_0, out0_stride_1, out1_stride_0, out1_stride_1, _BLOCK_SIZE_0: tl.constexpr):
    num_blocks_0 = tl.cdiv(a_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a_size_0
    offset_1 = pid_1
    indices_1 = offset_1 + tl.zeros([1], tl.int32)
    load = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_1[None, :] * a_stride_1), mask_0[:, None], other=0)
    load_1 = tl.load(b + indices_0[:, None] * b_stride_0, mask_0[:, None], other=0)
    v_0 = load + load_1
    tl.store(out0 + (indices_0[:, None] * out0_stride_0 + indices_1[None, :] * out0_stride_1), v_0, mask_0[:, None])
    load_2 = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_1[None, :] * a_stride_1), mask_0[:, None], other=0)
    load_3 = tl.load(b + indices_1[None, :] * b_stride_0, None)
    v_1 = load_2 + load_3
    tl.store(out1 + (indices_0[:, None] * out1_stride_0 + indices_1[None, :] * out1_stride_1), v_1, mask_0[:, None])

def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    _BLOCK_SIZE_0 = 64
    _broadcast_fn_kernel[triton.cdiv(a.size(0), _BLOCK_SIZE_0) * a.size(1),](a, b, out0, out1, a.size(0), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return (out0, out1)

def _broadcast_fn_make_precompiler(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    _BLOCK_SIZE_0 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_broadcast_fn_kernel)(a, b, out0, out1, a.size(0), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_broadcast4(self):
        code = _check_broadcast_fn(
            block_size=[1, 64],
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _broadcast_fn_kernel(a, b, out0, out1, a_size_0, a_size_1, a_stride_0, a_stride_1, b_stride_0, out0_stride_0, out0_stride_1, out1_stride_0, out1_stride_1, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = a_size_0
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < a_size_1
    load = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_1[None, :] * a_stride_1), mask_1[None, :], other=0)
    load_1 = tl.load(b + indices_0[:, None] * b_stride_0, None)
    v_0 = load + load_1
    tl.store(out0 + (indices_0[:, None] * out0_stride_0 + indices_1[None, :] * out0_stride_1), v_0, mask_1[None, :])
    load_2 = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_1[None, :] * a_stride_1), mask_1[None, :], other=0)
    load_3 = tl.load(b + indices_1[None, :] * b_stride_0, mask_1[None, :], other=0)
    v_1 = load_2 + load_3
    tl.store(out1 + (indices_0[:, None] * out1_stride_0 + indices_1[None, :] * out1_stride_1), v_1, mask_1[None, :])

def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    _BLOCK_SIZE_1 = 64
    _broadcast_fn_kernel[a.size(0) * triton.cdiv(a.size(1), _BLOCK_SIZE_1),](a, b, out0, out1, a.size(0), a.size(1), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return (out0, out1)

def _broadcast_fn_make_precompiler(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    _BLOCK_SIZE_1 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_broadcast_fn_kernel)(a, b, out0, out1, a.size(0), a.size(1), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_broadcast5(self):
        code = _check_broadcast_fn(
            block_size=[32, 32],
            indexing="block_ptr",
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _broadcast_fn_kernel(a, b, out0, out1, a_size_0, a_size_1, b_size_0, out0_size_0, out0_size_1, out1_size_0, out1_size_1, a_stride_0, a_stride_1, b_stride_0, out0_stride_0, out0_stride_1, out1_stride_0, out1_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(a_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    load = tl.load(tl.make_block_ptr(a, [a_size_0, a_size_1], [a_stride_0, a_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    load_1 = tl.reshape(tl.load(tl.make_block_ptr(b, [b_size_0], [b_stride_0], [offset_0], [_BLOCK_SIZE_0], [0]), boundary_check=[0], padding_option='zero'), [_BLOCK_SIZE_0, 1])
    v_0 = load + load_1
    tl.store(tl.make_block_ptr(out0, [out0_size_0, out0_size_1], [out0_stride_0, out0_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_0, boundary_check=[0, 1])
    load_2 = tl.load(tl.make_block_ptr(a, [a_size_0, a_size_1], [a_stride_0, a_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    load_3 = tl.reshape(tl.load(tl.make_block_ptr(b, [b_size_0], [b_stride_0], [offset_1], [_BLOCK_SIZE_1], [0]), boundary_check=[0], padding_option='zero'), [1, _BLOCK_SIZE_1])
    v_1 = load_2 + load_3
    tl.store(tl.make_block_ptr(out1, [out1_size_0, out1_size_1], [out1_stride_0, out1_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_1, boundary_check=[0, 1])

def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _broadcast_fn_kernel[triton.cdiv(a.size(0), _BLOCK_SIZE_0) * triton.cdiv(a.size(1), _BLOCK_SIZE_1),](a, b, out0, out1, a.size(0), a.size(1), b.size(0), out0.size(0), out0.size(1), out1.size(0), out1.size(1), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return (out0, out1)

def _broadcast_fn_make_precompiler(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_broadcast_fn_kernel)(a, b, out0, out1, a.size(0), a.size(1), b.size(0), out0.size(0), out0.size(1), out1.size(0), out1.size(1), a.stride(0), a.stride(1), b.stride(0), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_constexpr_index(self):
        @helion.kernel
        def fn(a, idx1):
            out0 = torch.empty_like(a)
            out1 = torch.empty_like(a)
            out2 = torch.empty_like(a)
            idx0 = 11
            for tile0, tile1 in hl.tile(out0.size()):
                out0[tile0, tile1] = a[tile0, tile1] + a[tile0, 3, None]
                out1[tile0, tile1] = a[tile0, tile1] + a[idx0, tile1][None, :]
                out2[tile0, tile1] = a[tile0, tile1] + a[tile0, idx1, None]
            return out0, out1, out2

        args = (torch.randn(512, 512, device=DEVICE), 123)
        code, (out0, out1, out2) = code_and_output(fn, args, block_size=[16, 16])
        torch.testing.assert_close(out0, args[0] + args[0][:, 3, None])
        torch.testing.assert_close(out1, args[0] + args[0][11, None, :])
        torch.testing.assert_close(out2, args[0] + args[0][:, args[1], None])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(a, out0, out1, out2, a_size_0, a_size_1, a_stride_0, a_stride_1, out0_stride_0, out0_stride_1, out1_stride_0, out1_stride_1, out2_stride_0, out2_stride_1, idx1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(a_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < a_size_1
    load = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_1 = tl.load(a + (indices_0[:, None] * a_stride_0 + 3 * a_stride_1), mask_0[:, None], other=0)
    v_0 = load + load_1
    tl.store(out0 + (indices_0[:, None] * out0_stride_0 + indices_1[None, :] * out0_stride_1), v_0, mask_0[:, None] & mask_1[None, :])
    load_2 = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_3 = tl.load(a + (11 * a_stride_0 + indices_1 * a_stride_1), mask_1, other=0)
    subscript = load_3[None, :]
    v_1 = load_2 + subscript
    tl.store(out1 + (indices_0[:, None] * out1_stride_0 + indices_1[None, :] * out1_stride_1), v_1, mask_0[:, None] & mask_1[None, :])
    load_4 = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_1[None, :] * a_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    load_5 = tl.load(a + (indices_0[:, None] * a_stride_0 + tl.full([1], idx1, tl.int32)[None, :] * a_stride_1), mask_0[:, None], other=0)
    v_2 = load_4 + load_5
    tl.store(out2 + (indices_0[:, None] * out2_stride_0 + indices_1[None, :] * out2_stride_1), v_2, mask_0[:, None] & mask_1[None, :])

def fn(a, idx1):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    out2 = torch.empty_like(a)
    idx0 = 11
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _fn_kernel[triton.cdiv(a.size(0), _BLOCK_SIZE_0) * triton.cdiv(a.size(1), _BLOCK_SIZE_1),](a, out0, out1, out2, a.size(0), a.size(1), a.stride(0), a.stride(1), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), out2.stride(0), out2.stride(1), idx1, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return (out0, out1, out2)

def _fn_make_precompiler(a, idx1):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    out2 = torch.empty_like(a)
    idx0 = 11
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(a, out0, out1, out2, a.size(0), a.size(1), a.stride(0), a.stride(1), out0.stride(0), out0.stride(1), out1.stride(0), out1.stride(1), out2.stride(0), out2.stride(1), idx1, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )


if __name__ == "__main__":
    unittest.main()
