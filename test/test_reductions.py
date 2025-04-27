from __future__ import annotations

from typing import Callable
import unittest

from expecttest import TestCase
import torch

import helion
from helion._testing import code_and_output
import helion.language as hl


@helion.kernel()
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty(
        [n],
        dtype=x.dtype,
        device=x.device,
    )
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n, :].sum(-1)
    return out


@helion.kernel()
def sum_kernel_keepdims(x: torch.Tensor) -> torch.Tensor:
    _n, m = x.size()
    out = torch.empty(
        [1, m],
        dtype=x.dtype,
        device=x.device,
    )
    for tile_m in hl.tile(m):
        out[:, tile_m] = x[:, tile_m].sum(0, keepdim=True)
    return out


@helion.kernel(config={"block_sizes": [1]})
def reduce_kernel(
    x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], out_dtype=torch.float32
) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty(
        [n],
        dtype=out_dtype,
        device=x.device,
    )
    for tile_n in hl.tile(n):
        out[tile_n] = fn(x[tile_n, :], dim=-1)
    return out


class TestReductions(TestCase):
    maxDiff = 16384

    def test_sum(self):
        args = (torch.randn([512, 512], device="cuda"),)
        code, output = code_and_output(sum_kernel, args, block_size=1)
        torch.testing.assert_close(output, args[0].sum(-1))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _sum_kernel_kernel(x, out, out_stride_0, x_stride_0, x_stride_1, _m, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < _m
    load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_1[None, :], other=0)
    sum_1 = tl.sum(load, 1)
    tl.store(out + indices_0 * out_stride_0, sum_1, None)

def sum_kernel(x: torch.Tensor):
    n, _m = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    _RDIM_SIZE_1 = triton.next_power_of_2(_m)
    _sum_kernel_kernel[n,](x, out, out.stride(0), x.stride(0), x.stride(1), _m, _RDIM_SIZE_1, num_warps=4, num_stages=3)
    return out""",
        )

    def test_sum_keepdims(self):
        args = (torch.randn([512, 512], device="cuda"),)
        code, output = code_and_output(
            sum_kernel_keepdims, args, block_size=16, indexing="block_ptr"
        )
        torch.testing.assert_close(output, args[0].sum(0, keepdim=True))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _sum_kernel_keepdims_kernel(x, out, out_size_0, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [0, offset_0], [_RDIM_SIZE_1, _BLOCK_SIZE_0], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    sum_1 = tl.reshape(tl.sum(load, 0), [1, _BLOCK_SIZE_0])
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [0, offset_0], [1, _BLOCK_SIZE_0], [1, 0]), sum_1, boundary_check=[1])

def sum_kernel_keepdims(x: torch.Tensor):
    _n, m = x.size()
    out = torch.empty([1, m], dtype=x.dtype, device=x.device)
    _BLOCK_SIZE_0 = 16
    _RDIM_SIZE_1 = triton.next_power_of_2(_n)
    _sum_kernel_keepdims_kernel[triton.cdiv(m, _BLOCK_SIZE_0),](x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _RDIM_SIZE_1, num_warps=4, num_stages=3)
    return out""",
        )

    def test_argmin_argmax(self):
        for fn in (torch.argmin, torch.argmax):
            args = (torch.randn([512, 512], device="cuda"), fn, torch.int64)
            code, output = code_and_output(
                reduce_kernel, args, block_size=16, indexing="block_ptr"
            )
            torch.testing.assert_close(output, args[1](args[0], dim=-1))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

@triton.jit
def _reduce_kernel_kernel(x, out, out_size_0, x_size_0, x_size_1, out_stride_0, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, 0], [_BLOCK_SIZE_0, _RDIM_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    argmax = triton_helpers.max_with_index(load, tl.broadcast_to(indices_1[None, :], [_BLOCK_SIZE_0, _RDIM_SIZE_1]), 1)[1].to(tl.int64)
    tl.store(tl.make_block_ptr(out, [out_size_0], [out_stride_0], [offset_0], [_BLOCK_SIZE_0], [0]), argmax, boundary_check=[0])

def reduce_kernel(x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], out_dtype=torch.float32):
    n, _m = x.size()
    out = torch.empty([n], dtype=out_dtype, device=x.device)
    _BLOCK_SIZE_0 = 16
    _RDIM_SIZE_1 = triton.next_power_of_2(_m)
    _reduce_kernel_kernel[triton.cdiv(n, _BLOCK_SIZE_0),](x, out, out.size(0), x.size(0), x.size(1), out.stride(0), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _RDIM_SIZE_1, num_warps=4, num_stages=3)
    return out""",
        )

    def test_reduction_functions(self):
        for block_size in (1, 16):
            for indexing in ("block_ptr", "pointer"):
                for fn in (
                    torch.amax,
                    torch.amin,
                    torch.prod,
                    torch.sum,
                ):
                    args = (torch.randn([512, 512], device="cuda"), fn)
                    _, output = code_and_output(
                        reduce_kernel, args, block_size=block_size, indexing=indexing
                    )
                    torch.testing.assert_close(output, fn(args[0], dim=-1))

    @unittest.skip("need to fix handling of multiple inductor buffers")
    def test_mean(self):
        args = (torch.randn([512, 512], device="cuda"), torch.mean, torch.int64)
        code, output = code_and_output(
            reduce_kernel, args, block_size=8, index="block_ptr"
        )
        torch.testing.assert_close(output, args[1](args[0], dim=-1))
        self.assertExpectedInline(
            code,
            """
            """,
        )
