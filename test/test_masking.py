from __future__ import annotations

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


class TestMasking(TestCase):
    maxDiff = 16384

    def test_mask_dot(self):
        @helion.kernel(config={"block_sizes": [[32, 32], 32]}, dot_precision="ieee")
        def add1mm(x, y):
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n])
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k] + 1, y[tile_k, tile_n] + 1)
                out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([100, 100], device=DEVICE),
            torch.randn([100, 100], device=DEVICE),
        )
        code, result = code_and_output(
            add1mm,
            args,
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _add1mm_kernel(x, y, out, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, m, n, k, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_blocks_0 = tl.cdiv(m, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < m
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < n
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in range(0, k.to(tl.int32), _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        mask_2 = indices_2 < k
        acc_copy = acc
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_2[None, :] * x_stride_1), mask_0[:, None] & mask_2[None, :], other=0)
        v_0 = 1.0
        v_1 = load + v_0
        load_1 = tl.load(y + (indices_2[:, None] * y_stride_0 + indices_1[None, :] * y_stride_1), mask_2[:, None] & mask_1[None, :], other=0)
        v_2 = 1.0
        v_3 = load_1 + v_2
        _mask_to = tl.where(mask_0[:, None] & mask_2[None, :], v_1, 0)
        _mask_to_1 = tl.where(mask_2[:, None] & mask_1[None, :], v_3, 0)
        acc = tl.dot(_mask_to, _mask_to_1, acc=acc_copy, input_precision='ieee')
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), acc, mask_0[:, None] & mask_1[None, :])

def add1mm(x, y):
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], device=x.device)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_2 = 32
    _add1mm_kernel[triton.cdiv(m, _BLOCK_SIZE_0) * triton.cdiv(n, _BLOCK_SIZE_1),](x, y, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _add1mm_make_precompiler(x, y):
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], device=x.device)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_2 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_add1mm_kernel)(x, y, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )
        torch.testing.assert_close(result, (args[0] + 1) @ (args[1] + 1))

    def test_no_mask_views0(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m, 1], device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :][:, :, None].sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, args[0].sum(dim=1, keepdim=True))
        self.assertNotIn("tl.where", code)

    def test_no_mask_views1(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].transpose(0, 1).sum(dim=0)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, args[0].sum(dim=1))
        self.assertNotIn("tl.where", code)

    def test_no_mask_full0(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                v = torch.zeros_like(x[tile_m, :])
                out[tile_m] = v.sum(-1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, torch.zeros_like(args[0]).sum(dim=1))
        self.assertNotIn("tl.where", code)

    def test_no_mask_full1(self):
        @helion.kernel(config={"block_size": [32, 32]})
        def fn(x):
            m, n = x.size()
            hl.specialize(n)
            out = torch.empty([m], device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                v = hl.zeros([tile_m, tile_n])
                out[tile_m] = v.sum(-1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, torch.zeros_like(args[0]).sum(dim=1))
        self.assertNotIn("tl.where", code)

    def test_mask_offset(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = (x[tile_m, :] + 1).sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, (args[0] + 1).sum(dim=1))
        self.assertIn("tl.where", code)

    def test_no_mask_inductor_ops(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                x0 = x[tile_m, :] + 1
                x1 = x0 - 1
                # +1-1 cancels out, so no masking needed
                out[tile_m] = x1.sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, args[0].sum(dim=1))
        self.assertNotIn("tl.where", code)

    def test_loop_carry_masking(self):
        @helion.kernel(config={"block_sizes": [32, 32]})
        def fn(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for _ in hl.tile(n, block_size=block_n):
                    # The first iteration, this doesn't need a mask -- but the second does
                    acc += acc.sum(dim=1, keepdim=True)
                    acc += 1
                out[tile_m] = acc.sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        self.assertIn("tl.where", code)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(out, out_stride_0, m, n, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_1 = pid_0 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < m
    acc = tl.full([_BLOCK_SIZE_1, _BLOCK_SIZE_0], 0.0, tl.float32)
    for offset_0 in range(0, n.to(tl.int32), _BLOCK_SIZE_0):
        indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
        mask_0 = indices_0 < n
        acc_copy = acc
        _mask_to = tl.where(mask_1[:, None] & mask_0[None, :], acc_copy, 0)
        sum_1 = tl.reshape(tl.sum(_mask_to, 1), [_BLOCK_SIZE_1, 1])
        v_0 = acc_copy + sum_1
        v_1 = 1.0
        acc = v_0 + v_1
    _mask_to_1 = tl.where(tl.broadcast_to(mask_1[:, None], [_BLOCK_SIZE_1, _BLOCK_SIZE_0]), acc, 0)
    sum_2 = tl.sum(_mask_to_1, 1)
    tl.store(out + indices_1 * out_stride_0, sum_2, mask_1)

def fn(x):
    m, n = x.size()
    block_n = 32
    out = torch.empty([m], device=x.device)
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_0 = 32
    _fn_kernel[triton.cdiv(m, _BLOCK_SIZE_1),](out, out.stride(0), m, n, _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x):
    m, n = x.size()
    block_n = 32
    out = torch.empty([m], device=x.device)
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_0 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(out, out.stride(0), m, n, _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_tile_index_does_not_mask(self):
        @helion.kernel(config={"block_sizes": [32, 32], "indexing": "block_ptr"})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            block_size_n = hl.register_block_size(n)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_size_n])
                for tile_n in hl.tile(0, n, block_size_n):
                    acc += x[tile_m.index, tile_n.index]
                out[tile_m.index] = acc.sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, args[0].sum(dim=1))
        self.assertNotIn("tl.where", code)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, out_size_0, x_size_0, x_size_1, out_stride_0, x_stride_0, x_stride_1, n, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_1 = pid_0 * _BLOCK_SIZE_1
    acc = tl.full([_BLOCK_SIZE_1, _BLOCK_SIZE_0], 0.0, tl.float32)
    for offset_0 in range(0, n.to(tl.int32), _BLOCK_SIZE_0):
        acc_copy = acc
        load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_1, offset_0], [_BLOCK_SIZE_1, _BLOCK_SIZE_0], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        acc = acc_copy + load
    sum_1 = tl.sum(acc, 1)
    tl.store(tl.make_block_ptr(out, [out_size_0], [out_stride_0], [offset_1], [_BLOCK_SIZE_1], [0]), sum_1, boundary_check=[0])

def fn(x):
    m, n = x.size()
    out = torch.empty([m], device=x.device)
    block_size_n = 32
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_0 = 32
    _fn_kernel[triton.cdiv(m, _BLOCK_SIZE_1),](x, out, out.size(0), x.size(0), x.size(1), out.stride(0), x.stride(0), x.stride(1), n, _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x):
    m, n = x.size()
    out = torch.empty([m], device=x.device)
    block_size_n = 32
    _BLOCK_SIZE_1 = 32
    _BLOCK_SIZE_0 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, out.size(0), x.size(0), x.size(1), out.stride(0), x.stride(0), x.stride(1), n, _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )
