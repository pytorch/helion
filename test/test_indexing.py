from __future__ import annotations

import unittest

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


class TestIndexing(TestCase):
    maxDiff = 16384

    def test_arange(self):
        @helion.kernel
        def arange(length: int, device: torch.device) -> torch.Tensor:
            out = torch.empty([length], dtype=torch.int32, device=device)
            for tile in hl.tile(length):
                out[tile] = tile.index
            return out

        code, result = code_and_output(
            arange,
            (100, DEVICE),
            block_size=32,
        )
        torch.testing.assert_close(
            result, torch.arange(0, 100, device=DEVICE, dtype=torch.int32)
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _arange_kernel(out, out_stride_0, length, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < length
    tl.store(out + indices_0 * out_stride_0, indices_0, mask_0)

def arange(length: int, device: torch.device):
    out = torch.empty([length], dtype=torch.int32, device=device)
    _BLOCK_SIZE_0 = 32
    _arange_kernel[triton.cdiv(length, _BLOCK_SIZE_0),](out, out.stride(0), length, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _arange_make_precompiler(length: int, device: torch.device):
    out = torch.empty([length], dtype=torch.int32, device=device)
    _BLOCK_SIZE_0 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_arange_kernel)(out, out.stride(0), length, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_pairwise_add(self):
        @helion.kernel()
        def pairwise_add(x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0) - 1])
            for tile in hl.tile(out.size(0)):
                out[tile] = x[tile] + x[tile.index + 1]
            return out

        x = torch.randn([500], device=DEVICE)
        code, result = code_and_output(
            pairwise_add,
            (x,),
            block_size=32,
        )
        torch.testing.assert_close(result, x[:-1] + x[1:])
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _pairwise_add_kernel(out, x, out_size_0, out_stride_0, x_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < out_size_0
    load = tl.load(x + indices_0 * x_stride_0, mask_0, other=0)
    v_0 = tl.full([], 1, tl.int32)
    v_1 = indices_0 + v_0
    load_1 = tl.load(x + v_1 * x_stride_0, mask_0, other=0)
    v_2 = load + load_1
    tl.store(out + indices_0 * out_stride_0, v_2, mask_0)

def pairwise_add(x: torch.Tensor):
    out = x.new_empty([x.size(0) - 1])
    _BLOCK_SIZE_0 = 32
    _pairwise_add_kernel[triton.cdiv(out.size(0), _BLOCK_SIZE_0),](out, x, out.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _pairwise_add_make_precompiler(x: torch.Tensor):
    out = x.new_empty([x.size(0) - 1])
    _BLOCK_SIZE_0 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_pairwise_add_kernel)(out, x, out.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_mask_store(self):
        @helion.kernel
        def masked_store(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(out.size(0)):
                hl.store(out, [tile], x[tile], extra_mask=(tile.index % 2) == 0)
            return out

        x = torch.randn([200], device=DEVICE)
        code, result = code_and_output(
            masked_store,
            (x,),
            block_size=16,
        )
        torch.testing.assert_close(
            result, torch.where(torch.arange(200, device=DEVICE) % 2 == 0, x, 0)
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_compat import libdevice

@triton.jit
def _masked_store_kernel(out, x, x_size_0, out_stride_0, x_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    load = tl.load(x + indices_0 * x_stride_0, mask_0, other=0)
    v_0 = tl.full([], 2, tl.int32)
    v_1 = indices_0 % v_0
    v_2 = tl.full([], 0, tl.int32)
    v_3 = v_1 != v_2
    v_4 = libdevice.signbit(v_1) != 0 if v_1.dtype is tl.float32 else v_1 < 0
    v_5 = libdevice.signbit(v_0) != 0 if v_0.dtype is tl.float32 else v_0 < 0
    v_6 = v_4 != v_5
    v_7 = v_3 & v_6
    v_8 = v_1 + v_0
    v_9 = tl.where(v_7, v_8, v_1)
    v_10 = tl.full([], 0, tl.int32)
    v_11 = v_9 == v_10
    tl.store(out + indices_0 * out_stride_0, load, mask_0 & v_11)

def masked_store(x: torch.Tensor):
    out = torch.zeros_like(x)
    _BLOCK_SIZE_0 = 16
    _masked_store_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](out, x, x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _masked_store_make_precompiler(x: torch.Tensor):
    out = torch.zeros_like(x)
    _BLOCK_SIZE_0 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_masked_store_kernel)(out, x, x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_mask_load(self):
        @helion.kernel
        def masked_load(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(out.size(0)):
                out[tile] = hl.load(x, [tile], extra_mask=(tile.index % 2) == 0)
            return out

        x = torch.randn([200], device=DEVICE)
        code, result = code_and_output(
            masked_load,
            (x,),
            block_size=16,
        )
        torch.testing.assert_close(
            result, torch.where(torch.arange(200, device=DEVICE) % 2 == 0, x, 0)
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_compat import libdevice

@triton.jit
def _masked_load_kernel(out, x, x_size_0, out_stride_0, x_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    v_0 = tl.full([], 2, tl.int32)
    v_1 = indices_0 % v_0
    v_2 = tl.full([], 0, tl.int32)
    v_3 = v_1 != v_2
    v_4 = libdevice.signbit(v_1) != 0 if v_1.dtype is tl.float32 else v_1 < 0
    v_5 = libdevice.signbit(v_0) != 0 if v_0.dtype is tl.float32 else v_0 < 0
    v_6 = v_4 != v_5
    v_7 = v_3 & v_6
    v_8 = v_1 + v_0
    v_9 = tl.where(v_7, v_8, v_1)
    v_10 = tl.full([], 0, tl.int32)
    v_11 = v_9 == v_10
    load = tl.load(x + indices_0 * x_stride_0, mask_0 & v_11, other=0)
    tl.store(out + indices_0 * out_stride_0, load, mask_0)

def masked_load(x: torch.Tensor):
    out = torch.zeros_like(x)
    _BLOCK_SIZE_0 = 16
    _masked_load_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0),](out, x, x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _masked_load_make_precompiler(x: torch.Tensor):
    out = torch.zeros_like(x)
    _BLOCK_SIZE_0 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_masked_load_kernel)(out, x, x.size(0), out.stride(0), x.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_tile_begin_end(self):
        @helion.kernel
        def tile_range_copy(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile in hl.tile(x.size(0)):
                for inner_tile in hl.tile(tile.begin, tile.end):
                    out[inner_tile] = x[inner_tile]
            return out

        x = torch.randn([100], device=DEVICE)
        code, result = code_and_output(
            tile_range_copy,
            (x,),
            block_size=[32, 16],
        )
        torch.testing.assert_close(result, x)
        code, result = code_and_output(
            tile_range_copy,
            (x,),
            block_size=[1, 1],
        )
        torch.testing.assert_close(result, x)

    def test_tile_block_size(self):
        @helion.kernel
        def test_block_size_access(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile in hl.tile(x.size(0)):
                out[tile] = tile.block_size
            return out

        x = torch.randn([64], device=DEVICE)
        code, result = code_and_output(
            test_block_size_access,
            (x,),
            block_size=16,
        )
        expected = torch.full_like(x, 16, dtype=torch.int32)
        torch.testing.assert_close(result, expected)
        code, result = code_and_output(
            test_block_size_access,
            (x,),
            block_size=1,
        )
        expected = torch.full_like(x, 1, dtype=torch.int32)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
