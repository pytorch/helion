from __future__ import annotations

from pathlib import Path

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import import_path
import helion.language as hl

basic_kernels = import_path(Path(__file__).parent / "data/basic_kernels.py")


global_tensor = torch.randn([512], device=DEVICE)


@helion.kernel
def sin_func_arg(a, fn) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile in hl.tile(a.size()):
        out[tile] = fn(torch.sin(a[tile]), tile)
    return out


class TestClosures(TestCase):
    maxDiff = 16384

    def test_add_global(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, out = code_and_output(basic_kernels.use_globals, args)
        torch.testing.assert_close(
            out,
            torch.sin(args[0] + basic_kernels.global_tensor[None, :])
            + basic_kernels.global_float,
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

import helion._testing.basic_kernels as _source_module

@triton.jit
def _use_globals_kernel(a, _source_module_attr_global_tensor, out, a_size_0, a_size_1, _source_module_attr_global_tensor_stride_0, a_stride_0, a_stride_1, out_stride_0, out_stride_1, _source_module_attr_global_float, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
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
    load_1 = tl.load(_source_module_attr_global_tensor + indices_1[None, :] * _source_module_attr_global_tensor_stride_0, mask_1[None, :], other=0)
    v_0 = load + load_1
    v_1 = tl_math.sin(v_0)
    v_2 = v_1 + _source_module_attr_global_float
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_2, mask_0[:, None] & mask_1[None, :])

def use_globals(a):
    out = _source_module.empty_like(a)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _use_globals_kernel[triton.cdiv(a.size(0), _BLOCK_SIZE_0) * triton.cdiv(a.size(1), _BLOCK_SIZE_1),](a, _source_module.global_tensor, out, a.size(0), a.size(1), _source_module.global_tensor.stride(0), a.stride(0), a.stride(1), out.stride(0), out.stride(1), _source_module.global_float, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _use_globals_make_precompiler(a):
    out = _source_module.empty_like(a)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_use_globals_kernel)(a, _source_module.global_tensor, out, a.size(0), a.size(1), _source_module.global_tensor.stride(0), a.stride(0), a.stride(1), out.stride(0), out.stride(1), _source_module.global_float, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_fn_arg_with_global(self):
        def fn_with_global(x, tile) -> torch.Tensor:
            return x + global_tensor[tile]

        args = (torch.randn([512], device=DEVICE), fn_with_global)
        code, out = code_and_output(sin_func_arg, args)
        torch.testing.assert_close(out, args[0].sin() + global_tensor)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

import test_closures as _source_module

@triton.jit
def _sin_func_arg_kernel(a, _source_module_attr_global_tensor, out, a_size_0, _source_module_attr_global_tensor_stride_0, a_stride_0, out_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a_size_0
    load = tl.load(a + indices_0 * a_stride_0, mask_0, other=0)
    v_0 = tl_math.sin(load)
    load_1 = tl.load(_source_module_attr_global_tensor + indices_0 * _source_module_attr_global_tensor_stride_0, mask_0, other=0)
    v_1 = v_0 + load_1
    tl.store(out + indices_0 * out_stride_0, v_1, mask_0)

def sin_func_arg(a, fn):
    out = torch.empty_like(a)
    _BLOCK_SIZE_0 = 512
    _sin_func_arg_kernel[triton.cdiv(a.size(0), _BLOCK_SIZE_0),](a, _source_module.global_tensor, out, a.size(0), _source_module.global_tensor.stride(0), a.stride(0), out.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _sin_func_arg_make_precompiler(a, fn):
    out = torch.empty_like(a)
    _BLOCK_SIZE_0 = 512
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_sin_func_arg_kernel)(a, _source_module.global_tensor, out, a.size(0), _source_module.global_tensor.stride(0), a.stride(0), out.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_fn_arg_with_global_different_file(self):
        args = (torch.randn([512], device=DEVICE), basic_kernels.add_global_float)
        code, out = code_and_output(sin_func_arg, args)
        torch.testing.assert_close(out, args[0].sin() + basic_kernels.global_float)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

import helion._testing.basic_kernels as _global_source0

@triton.jit
def _sin_func_arg_kernel(a, out, a_size_0, a_stride_0, out_stride_0, _global_source0_attr_global_float, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a_size_0
    load = tl.load(a + indices_0 * a_stride_0, mask_0, other=0)
    v_0 = tl_math.sin(load)
    v_1 = v_0 + _global_source0_attr_global_float
    tl.store(out + indices_0 * out_stride_0, v_1, mask_0)

def sin_func_arg(a, fn):
    out = torch.empty_like(a)
    _BLOCK_SIZE_0 = 512
    _sin_func_arg_kernel[triton.cdiv(a.size(0), _BLOCK_SIZE_0),](a, out, a.size(0), a.stride(0), out.stride(0), _global_source0.global_float, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _sin_func_arg_make_precompiler(a, fn):
    out = torch.empty_like(a)
    _BLOCK_SIZE_0 = 512
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_sin_func_arg_kernel)(a, out, a.size(0), a.stride(0), out.stride(0), _global_source0.global_float, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_fn_arg_with_closure(self):
        def fn_with_closure(x, tile) -> torch.Tensor:
            return x + closure_tensor[tile]

        closure_tensor = torch.randn([512], device=DEVICE)
        args = (torch.randn([512], device=DEVICE), fn_with_closure)
        code, out = code_and_output(sin_func_arg, args)
        torch.testing.assert_close(out, args[0].sin() + closure_tensor)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _sin_func_arg_kernel(a, fn_closure_0, out, a_size_0, a_stride_0, fn_closure_0_stride_0, out_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a_size_0
    load = tl.load(a + indices_0 * a_stride_0, mask_0, other=0)
    v_0 = tl_math.sin(load)
    load_1 = tl.load(fn_closure_0 + indices_0 * fn_closure_0_stride_0, mask_0, other=0)
    v_1 = v_0 + load_1
    tl.store(out + indices_0 * out_stride_0, v_1, mask_0)

def sin_func_arg(a, fn):
    out = torch.empty_like(a)
    _BLOCK_SIZE_0 = 512
    _sin_func_arg_kernel[triton.cdiv(a.size(0), _BLOCK_SIZE_0),](a, fn.__closure__[0].cell_contents, out, a.size(0), a.stride(0), fn.__closure__[0].cell_contents.stride(0), out.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _sin_func_arg_make_precompiler(a, fn):
    out = torch.empty_like(a)
    _BLOCK_SIZE_0 = 512
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_sin_func_arg_kernel)(a, fn.__closure__[0].cell_contents, out, a.size(0), a.stride(0), fn.__closure__[0].cell_contents.stride(0), out.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_fn_arg_with_nested_closure(self):
        def fn_with_closure_a(x, tile) -> torch.Tensor:
            return x + closure_tensor[tile]

        def fn_with_closure_b(x, tile) -> torch.Tensor:
            return fn_with_closure_a(x, tile) + int_closure

        closure_tensor = torch.randn([512], device=DEVICE)
        int_closure = 42
        args = (torch.randn([512], device=DEVICE), fn_with_closure_b)
        code, out = code_and_output(sin_func_arg, args)
        torch.testing.assert_close(out, args[0].sin() + closure_tensor + int_closure)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _sin_func_arg_kernel(a, fn_closure_0_closure_0, out, a_size_0, a_stride_0, fn_closure_0_closure_0_stride_0, out_stride_0, fn_closure_1, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a_size_0
    load = tl.load(a + indices_0 * a_stride_0, mask_0, other=0)
    v_0 = tl_math.sin(load)
    load_1 = tl.load(fn_closure_0_closure_0 + indices_0 * fn_closure_0_closure_0_stride_0, mask_0, other=0)
    v_1 = v_0 + load_1
    v_2 = fn_closure_1.to(tl.float32)
    v_3 = v_1 + v_2
    tl.store(out + indices_0 * out_stride_0, v_3, mask_0)

def sin_func_arg(a, fn):
    out = torch.empty_like(a)
    _BLOCK_SIZE_0 = 512
    _sin_func_arg_kernel[triton.cdiv(a.size(0), _BLOCK_SIZE_0),](a, fn.__closure__[0].cell_contents.__closure__[0].cell_contents, out, a.size(0), a.stride(0), fn.__closure__[0].cell_contents.__closure__[0].cell_contents.stride(0), out.stride(0), fn.__closure__[1].cell_contents, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _sin_func_arg_make_precompiler(a, fn):
    out = torch.empty_like(a)
    _BLOCK_SIZE_0 = 512
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_sin_func_arg_kernel)(a, fn.__closure__[0].cell_contents.__closure__[0].cell_contents, out, a.size(0), a.stride(0), fn.__closure__[0].cell_contents.__closure__[0].cell_contents.stride(0), out.stride(0), fn.__closure__[1].cell_contents, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_fn_called_on_host(self):
        def alloc(x):
            return torch.empty_like(x)

        @helion.kernel
        def call_func_arg_on_host(a, alloc) -> torch.Tensor:
            out = alloc(a)
            for tile in hl.tile(a.size()):
                out[tile] = a[tile].sin()
            return out

        args = (torch.randn([512], device=DEVICE), alloc)
        code, out = code_and_output(call_func_arg_on_host, args)
        torch.testing.assert_close(out, args[0].sin())
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

import test_closures as _source_module

@triton.jit
def _call_func_arg_on_host_kernel(a, out, a_size_0, a_stride_0, out_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a_size_0
    load = tl.load(a + indices_0 * a_stride_0, mask_0, other=0)
    v_0 = tl_math.sin(load)
    tl.store(out + indices_0 * out_stride_0, v_0, mask_0)

def call_func_arg_on_host(a, alloc):
    out = alloc(a)
    _BLOCK_SIZE_0 = 512
    _call_func_arg_on_host_kernel[triton.cdiv(a.size(0), _BLOCK_SIZE_0),](a, out, a.size(0), a.stride(0), out.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _call_func_arg_on_host_make_precompiler(a, alloc):
    out = alloc(a)
    _BLOCK_SIZE_0 = 512
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_call_func_arg_on_host_kernel)(a, out, a.size(0), a.stride(0), out.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )
