from __future__ import annotations

from pathlib import Path

from expecttest import TestCase
import torch

from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import import_path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
examples_dir = Path(__file__).parent.parent / "examples"


def run_example(
    name: str,
    args: tuple[torch.Tensor, ...],
    expected: torch.Tensor,
    fn_name: str | None = None,
    skip_accuracy=False,
    **kwargs: object,
) -> str:
    code, result = code_and_output(
        getattr(import_path(examples_dir / f"{name}.py"), fn_name or name),
        args,
        **kwargs,
    )
    skip_accuracy or torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-2)
    return code


class TestExamples(TestCase):
    maxDiff = 16384

    def test_add(self):
        args = (
            torch.randn([512, 512], device=DEVICE, dtype=torch.float32),
            torch.randn([512], device=DEVICE, dtype=torch.float16),
        )
        self.assertExpectedInline(
            run_example("add", args, sum(args), block_size=128),
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
    v_0 = load_1.to(tl.float32)
    v_1 = load + v_0
    tl.store(out + (indices_0 * out_stride_0 + indices_1 * out_stride_1), v_1, mask_0_1)

def add(x: torch.Tensor, y: torch.Tensor):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(x.shape, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0_1 = 128
    _add_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE_0_1), 1, 1](x, y, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)
    return out

def _add_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(x.shape, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0_1 = 128
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_add_kernel)(x, y, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)""",
        )

    def test_matmul(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        self.assertExpectedInline(
            run_example(
                "matmul",
                args,
                args[0] @ args[1],
                block_sizes=[[16, 16], 16],
                l2_grouping=4,
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(x, y, out, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, m, n, k, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_pid_m = tl.cdiv(m, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(n, _BLOCK_SIZE_1)
    num_pid_in_group = 4 * num_pid_n
    group_id = tl.program_id(0) // num_pid_in_group
    first_pid_m = group_id * 4
    group_size_m = min(num_pid_m - first_pid_m, 4)
    pid_0 = first_pid_m + tl.program_id(0) % num_pid_in_group % group_size_m
    pid_1 = tl.program_id(0) % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < m
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < n
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in range(0, k, _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        mask_2 = indices_2 < k
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_2[None, :] * x_stride_1), mask_0[:, None] & mask_2[None, :], other=0)
        load_1 = tl.load(y + (indices_2[:, None] * y_stride_0 + indices_1[None, :] * y_stride_1), mask_2[:, None] & mask_1[None, :], other=0)
        acc = tl.dot(load, load_1, acc=acc, input_precision='tf32')
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), acc, mask_0[:, None] & mask_1[None, :])

def matmul(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_kernel[triton.cdiv(m, _BLOCK_SIZE_0) * triton.cdiv(n, _BLOCK_SIZE_1),](x, y, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _matmul_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_matmul_kernel)(x, y, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_template_via_closure0(self):
        bias = torch.randn([1, 1024], device=DEVICE, dtype=torch.float16)
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            lambda acc, tile: torch.relu(acc + bias[tile]),
        )
        self.assertExpectedInline(
            run_example(
                "template_via_closure",
                args,
                torch.relu(args[0] @ args[1] + bias),
                fn_name="matmul_with_epilogue",
                block_sizes=[[64, 64], [16]],
                loop_orders=[[0, 1]],
                num_warps=2,
                num_stages=4,
                indexing="pointer",
                l2_grouping=64,
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

import test_examples as _global_source0

@triton.jit
def _matmul_with_epilogue_kernel(x, y, epilogue_closure_0, out, epilogue_closure_0_stride_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, m, n, k, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_pid_m = tl.cdiv(m, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(n, _BLOCK_SIZE_1)
    num_pid_in_group = 64 * num_pid_n
    group_id = tl.program_id(0) // num_pid_in_group
    first_pid_m = group_id * 64
    group_size_m = min(num_pid_m - first_pid_m, 64)
    pid_0 = first_pid_m + tl.program_id(0) % num_pid_in_group % group_size_m
    pid_1 = tl.program_id(0) % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < m
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < n
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in range(0, k, _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        mask_2 = indices_2 < k
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_2[None, :] * x_stride_1), mask_0[:, None] & mask_2[None, :], other=0)
        load_1 = tl.load(y + (indices_2[:, None] * y_stride_0 + indices_1[None, :] * y_stride_1), mask_2[:, None] & mask_1[None, :], other=0)
        acc = tl.dot(load, load_1, acc=acc, input_precision='tf32')
    load_2 = tl.load(epilogue_closure_0 + indices_1[None, :] * epilogue_closure_0_stride_1, mask_1[None, :], other=0)
    v_0 = load_2.to(tl.float32)
    v_1 = acc + v_0
    v_2 = tl.full([], 0, tl.int32)
    v_3 = triton_helpers.maximum(v_2, v_1)
    v_4 = v_3.to(tl.float16)
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_4, mask_0[:, None] & mask_1[None, :])

def matmul_with_epilogue(x: Tensor, y: Tensor, epilogue: Callable[[Tensor, list[Tensor]], Tensor]):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 64
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = 16
    _matmul_with_epilogue_kernel[triton.cdiv(m, _BLOCK_SIZE_0) * triton.cdiv(n, _BLOCK_SIZE_1),](x, y, epilogue.__closure__[0].cell_contents, out, epilogue.__closure__[0].cell_contents.stride(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2, num_stages=4)
    return out

def _matmul_with_epilogue_make_precompiler(x: Tensor, y: Tensor, epilogue: Callable[[Tensor, list[Tensor]], Tensor]):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 64
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_matmul_with_epilogue_kernel)(x, y, epilogue.__closure__[0].cell_contents, out, epilogue.__closure__[0].cell_contents.stride(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2, num_stages=4)""",
        )

    def test_template_via_closure1(self):
        bias = torch.randn([1, 1024], device=DEVICE, dtype=torch.float16)
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            lambda acc, tile: torch.relu(acc + bias[tile]),
        )
        self.assertExpectedInline(
            run_example(
                "template_via_closure",
                args,
                torch.relu(args[0] @ args[1] + bias),
                fn_name="matmul_with_epilogue",
                block_sizes=[[64, 64], [16]],
                loop_orders=[[0, 1]],
                num_warps=2,
                num_stages=4,
                indexing="block_ptr",
                l2_grouping=64,
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

import test_examples as _global_source0

@triton.jit
def _matmul_with_epilogue_kernel(x, y, epilogue_closure_0, out, epilogue_closure_0_size_0, epilogue_closure_0_size_1, out_size_0, out_size_1, x_size_0, x_size_1, y_size_0, y_size_1, epilogue_closure_0_stride_0, epilogue_closure_0_stride_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, m, n, k, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_pid_m = tl.cdiv(m, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(n, _BLOCK_SIZE_1)
    num_pid_in_group = 64 * num_pid_n
    group_id = tl.program_id(0) // num_pid_in_group
    first_pid_m = group_id * 64
    group_size_m = min(num_pid_m - first_pid_m, 64)
    pid_0 = first_pid_m + tl.program_id(0) % num_pid_in_group % group_size_m
    pid_1 = tl.program_id(0) % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in range(0, k, _BLOCK_SIZE_2):
        load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_2], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        load_1 = tl.load(tl.make_block_ptr(y, [y_size_0, y_size_1], [y_stride_0, y_stride_1], [offset_2, offset_1], [_BLOCK_SIZE_2, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        acc = tl.dot(load, load_1, acc=acc, input_precision='tf32')
    load_2 = tl.load(tl.make_block_ptr(epilogue_closure_0, [epilogue_closure_0_size_0, epilogue_closure_0_size_1], [epilogue_closure_0_stride_0, epilogue_closure_0_stride_1], [0, offset_1], [1, _BLOCK_SIZE_1], [1, 0]), boundary_check=[1], padding_option='zero')
    v_0 = load_2.to(tl.float32)
    v_1 = acc + v_0
    v_2 = tl.full([], 0, tl.int32)
    v_3 = triton_helpers.maximum(v_2, v_1)
    v_4 = v_3.to(tl.float16)
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_4, boundary_check=[0, 1])

def matmul_with_epilogue(x: Tensor, y: Tensor, epilogue: Callable[[Tensor, list[Tensor]], Tensor]):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 64
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = 16
    _matmul_with_epilogue_kernel[triton.cdiv(m, _BLOCK_SIZE_0) * triton.cdiv(n, _BLOCK_SIZE_1),](x, y, epilogue.__closure__[0].cell_contents, out, epilogue.__closure__[0].cell_contents.size(0), epilogue.__closure__[0].cell_contents.size(1), out.size(0), out.size(1), x.size(0), x.size(1), y.size(0), y.size(1), epilogue.__closure__[0].cell_contents.stride(0), epilogue.__closure__[0].cell_contents.stride(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2, num_stages=4)
    return out

def _matmul_with_epilogue_make_precompiler(x: Tensor, y: Tensor, epilogue: Callable[[Tensor, list[Tensor]], Tensor]):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 64
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_matmul_with_epilogue_kernel)(x, y, epilogue.__closure__[0].cell_contents, out, epilogue.__closure__[0].cell_contents.size(0), epilogue.__closure__[0].cell_contents.size(1), out.size(0), out.size(1), x.size(0), x.size(1), y.size(0), y.size(1), epilogue.__closure__[0].cell_contents.stride(0), epilogue.__closure__[0].cell_contents.stride(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2, num_stages=4)""",
        )

    def test_template_via_closure2(self):
        args = (
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            torch.randn([1024, 1024], device=DEVICE, dtype=torch.float16),
            lambda x, _: torch.nn.functional.relu(x),
        )
        self.assertExpectedInline(
            run_example(
                "template_via_closure",
                args,
                torch.relu(args[0] @ args[1]),
                fn_name="matmul_with_epilogue",
                block_sizes=[[64, 64], [16]],
                loop_orders=[[0, 1]],
                num_warps=2,
                num_stages=4,
                indexing="block_ptr",
                l2_grouping=64,
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

import test_examples as _global_source0

@triton.jit
def _matmul_with_epilogue_kernel(x, y, out, out_size_0, out_size_1, x_size_0, x_size_1, y_size_0, y_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, m, n, k, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_pid_m = tl.cdiv(m, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(n, _BLOCK_SIZE_1)
    num_pid_in_group = 64 * num_pid_n
    group_id = tl.program_id(0) // num_pid_in_group
    first_pid_m = group_id * 64
    group_size_m = min(num_pid_m - first_pid_m, 64)
    pid_0 = first_pid_m + tl.program_id(0) % num_pid_in_group % group_size_m
    pid_1 = tl.program_id(0) % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in range(0, k, _BLOCK_SIZE_2):
        load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_2], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        load_1 = tl.load(tl.make_block_ptr(y, [y_size_0, y_size_1], [y_stride_0, y_stride_1], [offset_2, offset_1], [_BLOCK_SIZE_2, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        acc = tl.dot(load, load_1, acc=acc, input_precision='tf32')
    v_0 = tl.full([], 0, tl.int32)
    v_1 = triton_helpers.maximum(v_0, acc)
    v_2 = v_1.to(tl.float16)
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_2, boundary_check=[0, 1])

def matmul_with_epilogue(x: Tensor, y: Tensor, epilogue: Callable[[Tensor, list[Tensor]], Tensor]):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 64
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = 16
    _matmul_with_epilogue_kernel[triton.cdiv(m, _BLOCK_SIZE_0) * triton.cdiv(n, _BLOCK_SIZE_1),](x, y, out, out.size(0), out.size(1), x.size(0), x.size(1), y.size(0), y.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2, num_stages=4)
    return out

def _matmul_with_epilogue_make_precompiler(x: Tensor, y: Tensor, epilogue: Callable[[Tensor, list[Tensor]], Tensor]):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 64
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_matmul_with_epilogue_kernel)(x, y, out, out.size(0), out.size(1), x.size(0), x.size(1), y.size(0), y.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2, num_stages=4)""",
        )

    def test_softmax(self):
        args = (torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),)
        self.assertExpectedInline(
            run_example(
                "softmax",
                args,
                torch.nn.functional.softmax(*args, dim=1),
                block_size=1,
                num_warps=4,
                num_stages=1,
                indexing="block_ptr",
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _softmax_kernel(x, out, out_size_0, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _m, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < _m
    load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, 0], [1, _RDIM_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    v_0 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), load, float('-inf'))
    amax = tl.reshape(tl.max(v_0, 1), [1, 1])
    v_1 = load - amax
    v_2 = tl_math.exp(v_1)
    v_3 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), v_2, 0)
    sum_1 = tl.reshape(tl.sum(v_3, 1), [1, 1])
    v_4 = v_2 / sum_1
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, 0], [1, _RDIM_SIZE_1], [1, 0]), v_4, boundary_check=[0, 1])

def softmax(x: torch.Tensor):
    n, _m = x.size()
    out = torch.empty_like(x)
    _RDIM_SIZE_1 = triton.next_power_of_2(_m)
    _softmax_kernel[n,](x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _m, _RDIM_SIZE_1, num_warps=4, num_stages=1)
    return out

def _softmax_make_precompiler(x: torch.Tensor):
    n, _m = x.size()
    out = torch.empty_like(x)
    _RDIM_SIZE_1 = triton.next_power_of_2(_m)
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_softmax_kernel)(x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _m, _RDIM_SIZE_1, num_warps=4, num_stages=1)""",
        )

    def test_softmax_looped(self):
        args = (torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),)
        self.assertExpectedInline(
            run_example(
                "softmax",
                args,
                torch.nn.functional.softmax(*args, dim=1),
                block_size=1,
                num_warps=4,
                num_stages=1,
                indexing="block_ptr",
                reduction_loop=32,
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _softmax_kernel(x, out, out_size_0, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _m, _REDUCTION_BLOCK_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    amax_acc = tl.full([1, _REDUCTION_BLOCK_1], float('-inf'), tl.float32)
    for roffset_1 in range(0, _m, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < _m
        load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, roffset_1], [1, _REDUCTION_BLOCK_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        v_0 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _REDUCTION_BLOCK_1]), load, float('-inf'))
        v_1 = triton_helpers.maximum(amax_acc, v_0)
        amax_acc = v_1
    amax = tl.reshape(tl.max(amax_acc, 1), [1, 1])
    sum_1_acc = tl.full([1, _REDUCTION_BLOCK_1], 0, tl.float32)
    for roffset_1 in range(0, _m, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < _m
        load_1 = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, roffset_1], [1, _REDUCTION_BLOCK_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        v_2 = load_1 - amax
        v_3 = tl_math.exp(v_2)
        v_4 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _REDUCTION_BLOCK_1]), v_3, 0)
        v_5 = sum_1_acc + v_4
        sum_1_acc = v_5
    sum_1 = tl.reshape(tl.sum(sum_1_acc, 1), [1, 1])
    for roffset_1 in range(0, _m, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < _m
        load_2 = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, roffset_1], [1, _REDUCTION_BLOCK_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        v_6 = load_2 - amax
        v_7 = tl_math.exp(v_6)
        v_8 = v_7 / sum_1
        tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, roffset_1], [1, _REDUCTION_BLOCK_1], [1, 0]), v_8, boundary_check=[0, 1])

def softmax(x: torch.Tensor):
    n, _m = x.size()
    out = torch.empty_like(x)
    _REDUCTION_BLOCK_1 = 32
    _softmax_kernel[n,](x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _m, _REDUCTION_BLOCK_1, num_warps=4, num_stages=1)
    return out

def _softmax_make_precompiler(x: torch.Tensor):
    n, _m = x.size()
    out = torch.empty_like(x)
    _REDUCTION_BLOCK_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_softmax_kernel)(x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _m, _REDUCTION_BLOCK_1, num_warps=4, num_stages=1)""",
        )

    def test_softmax_decomposed(self):
        args = (torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),)
        self.assertExpectedInline(
            run_example(
                "softmax",
                args,
                torch.nn.functional.softmax(*args, dim=1),
                fn_name="softmax_decomposed",
                block_size=1,
                num_warps=4,
                num_stages=1,
                indexing="block_ptr",
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _softmax_decomposed_kernel(x, out, out_size_0, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _m, _RDIM_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < _m
    values = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, 0], [1, _RDIM_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    v_0 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), values, float('-inf'))
    amax = tl.reshape(tl.max(v_0, 1), [1, 1])
    v_1 = values - amax
    v_2 = tl_math.exp(v_1)
    v_3 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), v_2, 0)
    sum_exp = tl.reshape(tl.sum(v_3, 1), [1, 1])
    v_4 = v_2 / sum_exp
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, 0], [1, _RDIM_SIZE_1], [1, 0]), v_4, boundary_check=[0, 1])

def softmax_decomposed(x: torch.Tensor):
    n, _m = x.size()
    out = torch.empty_like(x)
    _RDIM_SIZE_1 = triton.next_power_of_2(_m)
    _softmax_decomposed_kernel[n,](x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _m, _RDIM_SIZE_1, num_warps=4, num_stages=1)
    return out

def _softmax_decomposed_make_precompiler(x: torch.Tensor):
    n, _m = x.size()
    out = torch.empty_like(x)
    _RDIM_SIZE_1 = triton.next_power_of_2(_m)
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_softmax_decomposed_kernel)(x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _m, _RDIM_SIZE_1, num_warps=4, num_stages=1)""",
        )
