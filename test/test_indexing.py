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
            (100, torch.device("cuda")),
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
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
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
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
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


if __name__ == "__main__":
    unittest.main()
