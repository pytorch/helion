from __future__ import annotations

from typing import TYPE_CHECKING

from expecttest import TestCase
import torch

import helion
from helion._testing import code_and_output
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


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
    v_0 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), load, 0)
    sum_1 = tl.sum(v_0, 1)
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
        torch.testing.assert_close(
            output, args[0].sum(0, keepdim=True), rtol=2e-05, atol=1e-05
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _sum_kernel_keepdims_kernel(x, out, out_size_0, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _n, _BLOCK_SIZE_0: tl.constexpr, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < _n
    load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [0, offset_0], [_RDIM_SIZE_1, _BLOCK_SIZE_0], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    v_0 = tl.where(tl.broadcast_to(mask_1[:, None], [_RDIM_SIZE_1, _BLOCK_SIZE_0]), load, 0)
    sum_1 = tl.reshape(tl.sum(v_0, 0), [1, _BLOCK_SIZE_0])
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [0, offset_0], [1, _BLOCK_SIZE_0], [1, 0]), sum_1, boundary_check=[1])

def sum_kernel_keepdims(x: torch.Tensor):
    _n, m = x.size()
    out = torch.empty([1, m], dtype=x.dtype, device=x.device)
    _BLOCK_SIZE_0 = 16
    _RDIM_SIZE_1 = triton.next_power_of_2(_n)
    _sum_kernel_keepdims_kernel[triton.cdiv(m, _BLOCK_SIZE_0),](x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _n, _BLOCK_SIZE_0, _RDIM_SIZE_1, num_warps=4, num_stages=3)
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
def _reduce_kernel_kernel(x, out, out_size_0, x_size_0, x_size_1, out_stride_0, x_stride_0, x_stride_1, _m, _BLOCK_SIZE_0: tl.constexpr, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < _m
    load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, 0], [_BLOCK_SIZE_0, _RDIM_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    v_0 = tl.where(tl.broadcast_to(mask_1[None, :], [_BLOCK_SIZE_0, _RDIM_SIZE_1]), load, float('-inf'))
    argmax = triton_helpers.max_with_index(v_0, tl.broadcast_to(indices_1[None, :], [_BLOCK_SIZE_0, _RDIM_SIZE_1]), 1)[1].to(tl.int64)
    tl.store(tl.make_block_ptr(out, [out_size_0], [out_stride_0], [offset_0], [_BLOCK_SIZE_0], [0]), argmax, boundary_check=[0])

def reduce_kernel(x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], out_dtype=torch.float32):
    n, _m = x.size()
    out = torch.empty([n], dtype=out_dtype, device=x.device)
    _BLOCK_SIZE_0 = 16
    _RDIM_SIZE_1 = triton.next_power_of_2(_m)
    _reduce_kernel_kernel[triton.cdiv(n, _BLOCK_SIZE_0),](x, out, out.size(0), x.size(0), x.size(1), out.stride(0), x.stride(0), x.stride(1), _m, _BLOCK_SIZE_0, _RDIM_SIZE_1, num_warps=4, num_stages=3)
    return out""",
        )

    def test_reduction_functions(self):
        for reduction_loop in (None, 16):
            for block_size in (1, 16):
                for indexing in ("block_ptr", "pointer"):
                    for fn in (
                        torch.amax,
                        torch.amin,
                        torch.prod,
                        torch.sum,
                        torch.mean,
                    ):
                        args = (torch.randn([512, 512], device="cuda"), fn)
                        _, output = code_and_output(
                            reduce_kernel,
                            args,
                            block_size=block_size,
                            indexing=indexing,
                            reduction_loop=reduction_loop,
                        )
                        torch.testing.assert_close(
                            output, fn(args[0], dim=-1), rtol=1e-3, atol=1e-3
                        )

    def test_mean(self):
        args = (torch.randn([512, 512], device="cuda"), torch.mean, torch.float32)
        self.assertExpectedInline(
            reduce_kernel.bind(args)._debug_str(),
            """\
def reduce_kernel(x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], out_dtype=torch.float32):
    # Call: SequenceType((SymIntType(s77), SymIntType(s27))) SourceOrigin(location=<SourceLocation test_reductions.py:46>)
    # Attribute: TensorAttributeType AttributeOrigin(value=ArgumentOrigin(name='x'), key='size')
    # Name: TensorType([x_size0, x_size1], torch.float32) ArgumentOrigin(name='x')
    n, _m = x.size()
    # Call: TensorType([x_size0], torch.float32) SourceOrigin(location=<SourceLocation test_reductions.py:47>)
    # Attribute: CallableType(_VariableFunctionsClass.empty) AttributeOrigin(value=GlobalOrigin(name='torch'), key='empty')
    # Name: PythonModuleType(torch) GlobalOrigin(name='torch')
    # List: SequenceType([SymIntType(s77)]) SourceOrigin(location=<SourceLocation test_reductions.py:48>)
    # Name: SymIntType(s77) GetItemOrigin(value=SourceOrigin(location=<SourceLocation test_reductions.py:46>), key=0)
    # Name: LiteralType(torch.float32) ArgumentOrigin(name='out_dtype')
    # Attribute: LiteralType(device(type='cuda', index=0)) AttributeOrigin(value=ArgumentOrigin(name='x'), key='device')
    # Name: TensorType([x_size0, x_size1], torch.float32) ArgumentOrigin(name='x')
    # For: loop_type=GRID
    out = torch.empty([n], dtype=out_dtype, device=x.device)
    # Call: IterType(TileIndexType(0)) SourceOrigin(location=<SourceLocation test_reductions.py:52>)
    # Attribute: CallableType(tile) AttributeOrigin(value=GlobalOrigin(name='hl'), key='tile')
    # Name: PythonModuleType(helion.language) GlobalOrigin(name='hl')
    # Name: SymIntType(s77) GetItemOrigin(value=SourceOrigin(location=<SourceLocation test_reductions.py:46>), key=0)
    for tile_n in hl.tile(n):
        # Subscript: TensorType([block_size_0], torch.float32) DeviceOrigin(location=<SourceLocation test_reductions.py:53>)
        # Name: TensorType([x_size0], torch.float32) SourceOrigin(location=<SourceLocation test_reductions.py:47>)
        # Name: TileIndexType(0) SourceOrigin(location=<SourceLocation test_reductions.py:52>)
        # Call: TensorType([block_size_0], torch.float32) DeviceOrigin(location=<SourceLocation test_reductions.py:53>)
        # Name: CallableType(_VariableFunctionsClass.mean) ArgumentOrigin(name='fn')
        # Subscript: TensorType([block_size_0, x_size1], torch.float32) DeviceOrigin(location=<SourceLocation test_reductions.py:53>)
        # Name: TensorType([x_size0, x_size1], torch.float32) ArgumentOrigin(name='x')
        # Name: TileIndexType(0) SourceOrigin(location=<SourceLocation test_reductions.py:52>)
        # Slice: SliceType(LiteralType(None):LiteralType(None):LiteralType(None)) DeviceOrigin(location=<SourceLocation test_reductions.py:53>)
        # UnaryOp: LiteralType(-1) DeviceOrigin(location=<SourceLocation test_reductions.py:53>)
        # Constant: LiteralType(1) DeviceOrigin(location=<SourceLocation test_reductions.py:53>)
        out[tile_n] = fn(x[tile_n, :], dim=-1)
    return out

def root_graph_0():
    # File: .../test_reductions.py:53 in reduce_kernel, code: out[tile_n] = fn(x[tile_n, :], dim=-1)
    x: "f32[s77, s27]" = helion_language__tracing_ops__host_tensor('x')
    block_size_0: "Sym(u0)" = helion_language__tracing_ops__get_symnode('block_size_0')
    load: "f32[u0, u1]" = helion_language_memory_ops_load(x, [block_size_0, slice(None, None, None)]);  x = None
    mean_extra: "f32[u0]" = helion_language__tracing_ops__inductor_lowering_extra([load]);  load = None
    mean: "f32[u0]" = torch.ops.aten.mean.dim(None, [-1], _extra_args = [mean_extra]);  mean_extra = None
    out: "f32[s77]" = helion_language__tracing_ops__host_tensor('out')
    store = helion_language_memory_ops_store(out, [block_size_0], mean);  out = block_size_0 = mean = store = None
    return None

def reduction_loop_1(x: "f32[s77, s27]"):
    # File: .../test_reductions.py:53 in reduce_kernel, code: out[tile_n] = fn(x[tile_n, :], dim=-1)
    block_size_0: "Sym(u0)" = helion_language__tracing_ops__get_symnode('block_size_0')
    load: "f32[u0, u1]" = helion_language_memory_ops_load(x, [block_size_0, slice(None, None, None)]);  x = block_size_0 = None
    mean_extra: "f32[u0]" = helion_language__tracing_ops__inductor_lowering_extra([load]);  load = None
    return [mean_extra]

def root_graph_2():
    # File: .../test_reductions.py:53 in reduce_kernel, code: out[tile_n] = fn(x[tile_n, :], dim=-1)
    x: "f32[s77, s27]" = helion_language__tracing_ops__host_tensor('x')
    block_size_0: "Sym(u0)" = helion_language__tracing_ops__get_symnode('block_size_0')
    _for_loop = helion_language__tracing_ops__for_loop(1, [x]);  x = None
    getitem: "f32[u0]" = _for_loop[0];  _for_loop = None
    mean: "f32[u0]" = torch.ops.aten.mean.dim(None, [-1], _extra_args = [getitem]);  getitem = None
    out: "f32[s77]" = helion_language__tracing_ops__host_tensor('out')
    store = helion_language_memory_ops_store(out, [block_size_0], mean);  out = block_size_0 = mean = store = None
    return None""",
        )
        code, output = code_and_output(
            reduce_kernel, args, block_size=8, indexing="block_ptr"
        )
        torch.testing.assert_close(output, args[1](args[0], dim=-1))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _reduce_kernel_kernel(x, out, out_size_0, x_size_0, x_size_1, out_stride_0, x_stride_0, x_stride_1, _m, _BLOCK_SIZE_0: tl.constexpr, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < _m
    load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, 0], [_BLOCK_SIZE_0, _RDIM_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    v_0 = tl.where(tl.broadcast_to(mask_1[None, :], [_BLOCK_SIZE_0, _RDIM_SIZE_1]), load, 0)
    mean_extra = tl.sum(v_0, 1)
    v_1 = mean_extra / _m.to(tl.float32)
    tl.store(tl.make_block_ptr(out, [out_size_0], [out_stride_0], [offset_0], [_BLOCK_SIZE_0], [0]), v_1, boundary_check=[0])

def reduce_kernel(x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], out_dtype=torch.float32):
    n, _m = x.size()
    out = torch.empty([n], dtype=out_dtype, device=x.device)
    _BLOCK_SIZE_0 = 8
    _RDIM_SIZE_1 = triton.next_power_of_2(_m)
    _reduce_kernel_kernel[triton.cdiv(n, _BLOCK_SIZE_0),](x, out, out.size(0), x.size(0), x.size(1), out.stride(0), x.stride(0), x.stride(1), _m, _BLOCK_SIZE_0, _RDIM_SIZE_1, num_warps=4, num_stages=3)
    return out""",
        )

    def test_sum_looped(self):
        args = (torch.randn([512, 512], device="cuda"),)
        code, output = code_and_output(
            sum_kernel, args, block_size=2, reduction_loop=64
        )
        torch.testing.assert_close(output, args[0].sum(-1))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _sum_kernel_kernel(x, out, out_stride_0, x_stride_0, x_stride_1, n, _m, _BLOCK_SIZE_0: tl.constexpr, _REDUCTION_BLOCK_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < n
    sum_1_acc = tl.full([_BLOCK_SIZE_0, _REDUCTION_BLOCK_1], 0, tl.float32)
    for roffset_1 in range(0, _m, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < _m
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + rindex_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
        v_0 = tl.where(tl.broadcast_to(mask_1[None, :], [_BLOCK_SIZE_0, _REDUCTION_BLOCK_1]), load, 0)
        v_1 = sum_1_acc + v_0
        sum_1_acc = v_1
    sum_1 = tl.sum(sum_1_acc, 1)
    tl.store(out + indices_0 * out_stride_0, sum_1, mask_0)

def sum_kernel(x: torch.Tensor):
    n, _m = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    _BLOCK_SIZE_0 = 2
    _REDUCTION_BLOCK_1 = 64
    _sum_kernel_kernel[triton.cdiv(n, _BLOCK_SIZE_0),](x, out, out.stride(0), x.stride(0), x.stride(1), n, _m, _BLOCK_SIZE_0, _REDUCTION_BLOCK_1, num_warps=4, num_stages=3)
    return out""",
        )

    def test_argmin_argmax_looped(self):
        for fn in (torch.argmin, torch.argmax):
            args = (torch.randn([512, 512], device="cuda"), fn, torch.int64)
            code, output = code_and_output(
                reduce_kernel,
                args,
                block_size=1,
                indexing="block_ptr",
                reduction_loop=16,
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
def _reduce_kernel_kernel(x, out, out_size_0, x_size_0, x_size_1, out_stride_0, x_stride_0, x_stride_1, _m, _REDUCTION_BLOCK_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    argmax_acc = tl.full([1, _REDUCTION_BLOCK_1], float('-inf'), tl.float32)
    argmax_acc_index = tl.full([1, _REDUCTION_BLOCK_1], 2147483647, tl.int32)
    for roffset_1 in range(0, _m, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < _m
        load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, roffset_1], [1, _REDUCTION_BLOCK_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        v_0 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _REDUCTION_BLOCK_1]), load, float('-inf'))
        argmax_acc, argmax_acc_index = triton_helpers.maximum_with_index(argmax_acc, argmax_acc_index, v_0, tl.broadcast_to(rindex_1[None, :], [1, _REDUCTION_BLOCK_1]))
    argmax = triton_helpers.max_with_index(argmax_acc, argmax_acc_index, 1)[1].to(tl.int64)
    tl.store(tl.make_block_ptr(out, [out_size_0], [out_stride_0], [offset_0], [1], [0]), argmax, boundary_check=[0])

def reduce_kernel(x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], out_dtype=torch.float32):
    n, _m = x.size()
    out = torch.empty([n], dtype=out_dtype, device=x.device)
    _REDUCTION_BLOCK_1 = 16
    _reduce_kernel_kernel[n,](x, out, out.size(0), x.size(0), x.size(1), out.stride(0), x.stride(0), x.stride(1), _m, _REDUCTION_BLOCK_1, num_warps=4, num_stages=3)
    return out""",
        )
