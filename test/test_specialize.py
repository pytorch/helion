from __future__ import annotations

import math
import unittest

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion.exc import ShapeSpecializingAllocation
import helion.language as hl


class TestSpecialize(TestCase):
    maxDiff = 163842

    def test_sqrt_does_not_specialize(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            scale = 1.0 / math.sqrt(x.size(-1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * scale
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_sizes=[32, 1], flatten_loop=True)
        torch.testing.assert_close(result, x / math.sqrt(x.size(-1)))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, scale, _BLOCK_SIZE_0_1: tl.constexpr):
    offsets_0_1 = tl.program_id(0) * _BLOCK_SIZE_0_1 + tl.arange(0, _BLOCK_SIZE_0_1).to(tl.int32)
    indices_1 = offsets_0_1 % x_size_1
    indices_0 = offsets_0_1 // x_size_1
    mask_0_1 = offsets_0_1 < x_size_0 * x_size_1
    load = tl.load(x + (indices_0 * x_stride_0 + indices_1 * x_stride_1), mask_0_1, other=0)
    v_0 = load * scale
    tl.store(out + (indices_0 * out_stride_0 + indices_1 * out_stride_1), v_0, mask_0_1)

def fn(x: torch.Tensor):
    scale = 1.0 / math.sqrt(x.size(-1))
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1 = 32
    _fn_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE_0_1), 1, 1](x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), scale, _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor):
    scale = 1.0 / math.sqrt(x.size(-1))
    out = torch.empty_like(x)
    _BLOCK_SIZE_0_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), scale, _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)""",
        )

    def test_specialize_host(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(-1))
            scale = 1.0 / math.sqrt(x.size(-1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * scale
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_sizes=[32, 32])
        torch.testing.assert_close(result, x / math.sqrt(x.size(-1)))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, x_size_0, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < 500
    load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    v_0 = 0.044721359549995794
    v_1 = load * v_0
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_1, mask_0[:, None] & mask_1[None, :])

def fn(x: torch.Tensor):
    500
    scale = 1.0 / math.sqrt(x.size(-1))
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0) * triton.cdiv(x.size(1), _BLOCK_SIZE_1),](x, out, x.size(0), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor):
    500
    scale = 1.0 / math.sqrt(x.size(-1))
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_dynamic_size_block_errors(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.zeros([tile, x.size(1)])
                acc += x[tile, :] + 1
                out[tile, :] = acc
            return out

        x = torch.randn([512, 512], device=DEVICE)
        with self.assertRaises(ShapeSpecializingAllocation):
            code_and_output(fn, (x,), block_size=16)

    def test_dynamic_size_block_specialize(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.zeros([tile, x.size(1)])
                acc += x[tile, :] + 1
                out[tile, :] = acc
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x + 1)
        self.assertFalse(fn.bind((x,)).config_spec.reduction_loop_specs[0].allow_loop)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, x_size_0, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    acc = tl.full([_BLOCK_SIZE_0, 512], 0.0, tl.float32)
    load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None], other=0)
    v_0 = 1.0
    v_1 = load + v_0
    v_2 = acc + v_1
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_2, mask_0[:, None])

def fn(x: torch.Tensor):
    512
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _RDIM_SIZE_1 = 512
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, out, x.size(0), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _RDIM_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor):
    512
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _RDIM_SIZE_1 = 512
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _RDIM_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_dynamic_size_block_non_power_of_two(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.zeros([tile, helion.next_power_of_2(x.size(1))])
                acc += x[tile, :] + 1
                out[tile, :] = acc
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x + 1)
        self.assertFalse(fn.bind((x,)).config_spec.reduction_loop_specs[0].allow_loop)
        self.assertIs(
            fn.bind((x,)),
            fn.bind((torch.zeros_like(x),)),
        )
        self.assertIsNot(
            fn.bind((x,)),
            fn.bind((torch.zeros_like(x[:, 1:]),)),
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, x_size_0, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < 500
    acc = tl.full([_BLOCK_SIZE_0, 512], 0.0, tl.float32)
    load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    v_0 = 1.0
    v_1 = load + v_0
    v_2 = acc + v_1
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_2, mask_0[:, None] & mask_1[None, :])

def fn(x: torch.Tensor):
    500
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _RDIM_SIZE_1 = 512
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, out, x.size(0), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _RDIM_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor):
    500
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _RDIM_SIZE_1 = 512
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _RDIM_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_specialize_reduce(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x.sum(-1))
        self.assertTrue(fn.bind((x,)).config_spec.reduction_loop_specs[0].allow_loop)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, x_size_0, out_stride_0, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < 500
    load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    sum_1 = tl.sum(load, 1)
    tl.store(out + indices_0 * out_stride_0, sum_1, mask_0)

def fn(x: torch.Tensor):
    500
    out = x.new_empty([x.size(0)])
    _BLOCK_SIZE_0 = 32
    _RDIM_SIZE_1 = 512
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](x, out, x.size(0), out.stride(0), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _RDIM_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor):
    500
    out = x.new_empty([x.size(0)])
    _BLOCK_SIZE_0 = 32
    _RDIM_SIZE_1 = 512
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), out.stride(0), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _RDIM_SIZE_1, num_warps=4, num_stages=3)""",
        )


if __name__ == "__main__":
    unittest.main()
