from __future__ import annotations

from pathlib import Path
import unittest

from expecttest import TestCase
from packaging import version
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
def _matmul_kernel(x, y, out, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_pid_m = tl.cdiv(128, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(128, _BLOCK_SIZE_1)
    num_pid_in_group = 4 * num_pid_n
    group_id = tl.program_id(0) // num_pid_in_group
    first_pid_m = group_id * 4
    group_size_m = min(num_pid_m - first_pid_m, 4)
    pid_0 = first_pid_m + tl.program_id(0) % num_pid_in_group % group_size_m
    pid_1 = tl.program_id(0) % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in range(0, 128, _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        acc_copy = acc
        load = tl.load(x + (indices_0[:, None] * 128 + indices_2[None, :] * 1), None)
        load_1 = tl.load(y + (indices_2[:, None] * 128 + indices_1[None, :] * 1), None)
        acc = tl.dot(load, load_1, acc=acc_copy, input_precision='tf32')
    tl.store(out + (indices_0[:, None] * 128 + indices_1[None, :] * 1), acc, None)

def matmul(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_kernel[triton.cdiv(128, _BLOCK_SIZE_0) * triton.cdiv(128, _BLOCK_SIZE_1),](x, y, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
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
    return make_precompiler(_matmul_kernel)(x, y, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    @unittest.skipIf(
        version.parse(torch.__version__.split("+")[0]) < version.parse("2.8"),
        "Requires torch 2.8+",
    )
    def test_bmm(self):
        args = (
            torch.randn([16, 512, 768], device=DEVICE, dtype=torch.float16),
            torch.randn([16, 768, 1024], device=DEVICE, dtype=torch.float16),
        )
        self.assertExpectedInline(
            run_example(
                "bmm",
                args,
                torch.bmm(args[0], args[1]),
                block_sizes=[[16, 16, 16], 16],
                l2_grouping=4,
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _bmm_kernel(A, B, out, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    num_blocks_0 = tl.cdiv(16, _BLOCK_SIZE_0)
    num_blocks_1 = tl.cdiv(512, _BLOCK_SIZE_1)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0 % num_blocks_1
    pid_2 = tl.program_id(0) // (num_blocks_0 * num_blocks_1)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    offset_2 = pid_2 * _BLOCK_SIZE_2
    indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2], 0.0, tl.float32)
    for offset_3 in range(0, 768, _BLOCK_SIZE_3):
        indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
        acc_copy = acc
        load = tl.load(A + (indices_0[:, None, None] * 393216 + indices_1[None, :, None] * 768 + indices_3[None, None, :] * 1), None)
        load_1 = tl.load(B + (indices_0[:, None, None] * 786432 + indices_3[None, :, None] * 1024 + indices_2[None, None, :] * 1), None)
        acc = tl.dot(load, load_1, acc=acc_copy, input_precision='tf32')
    v_0 = acc.to(tl.float16)
    tl.store(out + (indices_0[:, None, None] * 524288 + indices_1[None, :, None] * 1024 + indices_2[None, None, :] * 1), v_0, None)

def bmm(A: torch.Tensor, B: torch.Tensor):
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty([b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype))
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _BLOCK_SIZE_3 = 16
    _bmm_kernel[triton.cdiv(16, _BLOCK_SIZE_0) * triton.cdiv(512, _BLOCK_SIZE_1) * triton.cdiv(1024, _BLOCK_SIZE_2),](A, B, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, _BLOCK_SIZE_3, num_warps=4, num_stages=3)
    return out

def _bmm_make_precompiler(A: torch.Tensor, B: torch.Tensor):
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty([b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype))
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _BLOCK_SIZE_3 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_bmm_kernel)(A, B, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, _BLOCK_SIZE_3, num_warps=4, num_stages=3)""",
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
def _matmul_with_epilogue_kernel(x, y, epilogue_closure_0, out, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_pid_m = tl.cdiv(1024, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(1024, _BLOCK_SIZE_1)
    num_pid_in_group = 64 * num_pid_n
    group_id = tl.program_id(0) // num_pid_in_group
    first_pid_m = group_id * 64
    group_size_m = min(num_pid_m - first_pid_m, 64)
    pid_0 = first_pid_m + tl.program_id(0) % num_pid_in_group % group_size_m
    pid_1 = tl.program_id(0) % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in range(0, 1024, _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        acc_copy = acc
        load = tl.load(x + (indices_0[:, None] * 1024 + indices_2[None, :] * 1), None)
        load_1 = tl.load(y + (indices_2[:, None] * 1024 + indices_1[None, :] * 1), None)
        acc = tl.dot(load, load_1, acc=acc_copy, input_precision='tf32')
    load_2 = tl.load(epilogue_closure_0 + indices_1[None, :] * 1, None)
    v_0 = load_2.to(tl.float32)
    v_1 = acc + v_0
    v_2 = tl.full([], 0, tl.int32)
    v_3 = triton_helpers.maximum(v_2, v_1)
    v_4 = v_3.to(tl.float16)
    tl.store(out + (indices_0[:, None] * 1024 + indices_1[None, :] * 1), v_4, None)

def matmul_with_epilogue(x: Tensor, y: Tensor, epilogue: Callable[[Tensor, list[Tensor]], Tensor]):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 64
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = 16
    _matmul_with_epilogue_kernel[triton.cdiv(1024, _BLOCK_SIZE_0) * triton.cdiv(1024, _BLOCK_SIZE_1),](x, y, epilogue.__closure__[0].cell_contents, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2, num_stages=4)
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
    return make_precompiler(_matmul_with_epilogue_kernel)(x, y, epilogue.__closure__[0].cell_contents, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2, num_stages=4)""",
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
def _matmul_with_epilogue_kernel(x, y, epilogue_closure_0, out, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_pid_m = tl.cdiv(1024, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(1024, _BLOCK_SIZE_1)
    num_pid_in_group = 64 * num_pid_n
    group_id = tl.program_id(0) // num_pid_in_group
    first_pid_m = group_id * 64
    group_size_m = min(num_pid_m - first_pid_m, 64)
    pid_0 = first_pid_m + tl.program_id(0) % num_pid_in_group % group_size_m
    pid_1 = tl.program_id(0) % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in range(0, 1024, _BLOCK_SIZE_2):
        acc_copy = acc
        load = tl.load(tl.make_block_ptr(x, [1024, 1024], [1024, 1], [offset_0, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_2], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        load_1 = tl.load(tl.make_block_ptr(y, [1024, 1024], [1024, 1], [offset_2, offset_1], [_BLOCK_SIZE_2, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        acc = tl.dot(load, load_1, acc=acc_copy, input_precision='tf32')
    load_2 = tl.load(tl.make_block_ptr(epilogue_closure_0, [1, 1024], [1024, 1], [0, offset_1], [1, _BLOCK_SIZE_1], [1, 0]), boundary_check=[1], padding_option='zero')
    v_0 = load_2.to(tl.float32)
    v_1 = acc + v_0
    v_2 = tl.full([], 0, tl.int32)
    v_3 = triton_helpers.maximum(v_2, v_1)
    v_4 = v_3.to(tl.float16)
    tl.store(tl.make_block_ptr(out, [1024, 1024], [1024, 1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_4, boundary_check=[0, 1])

def matmul_with_epilogue(x: Tensor, y: Tensor, epilogue: Callable[[Tensor, list[Tensor]], Tensor]):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 64
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = 16
    _matmul_with_epilogue_kernel[triton.cdiv(1024, _BLOCK_SIZE_0) * triton.cdiv(1024, _BLOCK_SIZE_1),](x, y, epilogue.__closure__[0].cell_contents, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2, num_stages=4)
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
    return make_precompiler(_matmul_with_epilogue_kernel)(x, y, epilogue.__closure__[0].cell_contents, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2, num_stages=4)""",
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
def _matmul_with_epilogue_kernel(x, y, out, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_pid_m = tl.cdiv(1024, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(1024, _BLOCK_SIZE_1)
    num_pid_in_group = 64 * num_pid_n
    group_id = tl.program_id(0) // num_pid_in_group
    first_pid_m = group_id * 64
    group_size_m = min(num_pid_m - first_pid_m, 64)
    pid_0 = first_pid_m + tl.program_id(0) % num_pid_in_group % group_size_m
    pid_1 = tl.program_id(0) % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in range(0, 1024, _BLOCK_SIZE_2):
        acc_copy = acc
        load = tl.load(tl.make_block_ptr(x, [1024, 1024], [1024, 1], [offset_0, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_2], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        load_1 = tl.load(tl.make_block_ptr(y, [1024, 1024], [1024, 1], [offset_2, offset_1], [_BLOCK_SIZE_2, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        acc = tl.dot(load, load_1, acc=acc_copy, input_precision='tf32')
    v_0 = tl.full([], 0, tl.int32)
    v_1 = triton_helpers.maximum(v_0, acc)
    v_2 = v_1.to(tl.float16)
    tl.store(tl.make_block_ptr(out, [1024, 1024], [1024, 1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_2, boundary_check=[0, 1])

def matmul_with_epilogue(x: Tensor, y: Tensor, epilogue: Callable[[Tensor, list[Tensor]], Tensor]):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 64
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = 16
    _matmul_with_epilogue_kernel[triton.cdiv(1024, _BLOCK_SIZE_0) * triton.cdiv(1024, _BLOCK_SIZE_1),](x, y, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2, num_stages=4)
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
    return make_precompiler(_matmul_with_epilogue_kernel)(x, y, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2, num_stages=4)""",
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
        amax_copy = amax
        load_1 = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, roffset_1], [1, _REDUCTION_BLOCK_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        v_2 = load_1 - amax_copy
        v_3 = tl_math.exp(v_2)
        v_4 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _REDUCTION_BLOCK_1]), v_3, 0)
        v_5 = sum_1_acc + v_4
        sum_1_acc = v_5
    sum_1 = tl.reshape(tl.sum(sum_1_acc, 1), [1, 1])
    for roffset_1 in range(0, _m, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < _m
        amax_copy_1 = amax
        sum_1_copy = sum_1
        load_2 = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, roffset_1], [1, _REDUCTION_BLOCK_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        v_6 = load_2 - amax_copy_1
        v_7 = tl_math.exp(v_6)
        v_8 = v_7 / sum_1_copy
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

    def test_softmax_two_pass(self):
        args = (torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),)
        self.assertExpectedInline(
            run_example(
                "softmax",
                args,
                torch.nn.functional.softmax(*args, dim=1),
                fn_name="softmax_two_pass",
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _softmax_two_pass_kernel(x, out, out_stride_0, out_stride_1, x_stride_0, x_stride_1, n, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    mi = tl.full([1], float('-inf'), tl.float32)
    di = tl.full([1], 0.0, tl.float32)
    for offset_2 in range(0, n, _BLOCK_SIZE_1):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_2 < n
        mi_copy = mi
        di_copy = di
        values = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_2[None, :] * x_stride_1), mask_1[None, :], other=0)
        v_0 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _BLOCK_SIZE_1]), values, float('-inf'))
        local_amax = tl.max(v_0, 1)
        mi = triton_helpers.maximum(mi_copy, local_amax)
        v_2 = mi_copy - mi
        v_3 = tl_math.exp(v_2)
        v_4 = di_copy * v_3
        subscript = mi[:, None]
        v_5 = values - subscript
        v_6 = tl_math.exp(v_5)
        v_7 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _BLOCK_SIZE_1]), v_6, 0)
        sum_1 = tl.sum(v_7, 1)
        di = v_4 + sum_1
    for offset_2 in range(0, n, _BLOCK_SIZE_1):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_2 = indices_2 < n
        mi_copy_1 = mi
        di_copy_1 = di
        values = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_2[None, :] * x_stride_1), mask_2[None, :], other=0)
        subscript_1 = mi_copy_1[:, None]
        v_9 = values - subscript_1
        v_10 = tl_math.exp(v_9)
        subscript_2 = di_copy_1[:, None]
        v_11 = v_10 / subscript_2
        tl.store(out + (indices_0[:, None] * out_stride_0 + indices_2[None, :] * out_stride_1), v_11, mask_2[None, :])

def softmax_two_pass(x: torch.Tensor):
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = 1
    block_size_n = 128
    _BLOCK_SIZE_1 = 128
    _softmax_two_pass_kernel[m,](x, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), n, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _softmax_two_pass_make_precompiler(x: torch.Tensor):
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = 1
    block_size_n = 128
    _BLOCK_SIZE_1 = 128
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_softmax_two_pass_kernel)(x, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), n, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_softmax_two_pass_block_ptr(self):
        args = (torch.randn([1024, 1024], device=DEVICE, dtype=torch.float32),)
        self.assertExpectedInline(
            run_example(
                "softmax",
                args,
                torch.nn.functional.softmax(*args, dim=1),
                fn_name="softmax_two_pass",
                block_sizes=[8, 64],
                indexing="block_ptr",
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _softmax_two_pass_kernel(x, out, out_size_0, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, n, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    mi = tl.full([_BLOCK_SIZE_0], float('-inf'), tl.float32)
    di = tl.full([_BLOCK_SIZE_0], 0.0, tl.float32)
    for offset_2 in range(0, n, _BLOCK_SIZE_1):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_2 < n
        mi_copy = mi
        di_copy = di
        values = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        v_0 = tl.where(tl.broadcast_to(mask_1[None, :], [_BLOCK_SIZE_0, _BLOCK_SIZE_1]), values, float('-inf'))
        local_amax = tl.max(v_0, 1)
        mi = triton_helpers.maximum(mi_copy, local_amax)
        v_2 = mi_copy - mi
        v_3 = tl_math.exp(v_2)
        v_4 = di_copy * v_3
        subscript = mi[:, None]
        v_5 = values - subscript
        v_6 = tl_math.exp(v_5)
        v_7 = tl.where(tl.broadcast_to(mask_1[None, :], [_BLOCK_SIZE_0, _BLOCK_SIZE_1]), v_6, 0)
        sum_1 = tl.sum(v_7, 1)
        di = v_4 + sum_1
    for offset_2 in range(0, n, _BLOCK_SIZE_1):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mi_copy_1 = mi
        di_copy_1 = di
        values = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        subscript_1 = mi_copy_1[:, None]
        v_9 = values - subscript_1
        v_10 = tl_math.exp(v_9)
        subscript_2 = di_copy_1[:, None]
        v_11 = v_10 / subscript_2
        tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_11, boundary_check=[0, 1])

def softmax_two_pass(x: torch.Tensor):
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = 8
    block_size_n = 64
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 64
    _softmax_two_pass_kernel[triton.cdiv(m, _BLOCK_SIZE_0),](x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), n, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _softmax_two_pass_make_precompiler(x: torch.Tensor):
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = 8
    block_size_n = 64
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_softmax_two_pass_kernel)(x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), n, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_embedding_pointers(self):
        args = (
            torch.randint(0, 1024, [8, 128], device=DEVICE, dtype=torch.int32),
            torch.randn([1024, 256], device=DEVICE, dtype=torch.float16),
        )
        self.assertExpectedInline(
            run_example(
                "embedding",
                args,
                torch.nn.functional.embedding(*args),
                block_size=[1, 256],
                indexing="pointer",
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _embedding_kernel(x_flat, weight, out, x_size_0, x_size_1, out_stride_0, out_stride_1, weight_stride_0, weight_stride_1, x_flat_stride_0, embedding_dim, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = x_size_0 * x_size_1
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < embedding_dim
    load = tl.load(x_flat + indices_0 * x_flat_stride_0, None)
    load_1 = tl.load(weight + (load[:, None] * weight_stride_0 + indices_1[None, :] * weight_stride_1), mask_1[None, :], other=0)
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), load_1, mask_1[None, :])

def embedding(x: torch.Tensor, weight: torch.Tensor):
    x_flat = x.reshape(-1)
    _, embedding_dim = weight.size()
    out = torch.empty([x_flat.size(0), embedding_dim], dtype=weight.dtype, device=weight.device)
    _BLOCK_SIZE_1 = 256
    _embedding_kernel[x.size(0) * x.size(1) * triton.cdiv(embedding_dim, _BLOCK_SIZE_1),](x_flat, weight, out, x.size(0), x.size(1), out.stride(0), out.stride(1), weight.stride(0), weight.stride(1), x_flat.stride(0), embedding_dim, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out.view(*x.size(), embedding_dim)

def _embedding_make_precompiler(x: torch.Tensor, weight: torch.Tensor):
    x_flat = x.reshape(-1)
    _, embedding_dim = weight.size()
    out = torch.empty([x_flat.size(0), embedding_dim], dtype=weight.dtype, device=weight.device)
    _BLOCK_SIZE_1 = 256
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_embedding_kernel)(x_flat, weight, out, x.size(0), x.size(1), out.stride(0), out.stride(1), weight.stride(0), weight.stride(1), x_flat.stride(0), embedding_dim, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_embedding_block_ptr(self):
        args = (
            torch.randint(0, 1024, [8, 128], device=DEVICE, dtype=torch.int32),
            torch.randn([1024, 256], device=DEVICE, dtype=torch.float16),
        )
        self.assertExpectedInline(
            run_example(
                "embedding",
                args,
                torch.nn.functional.embedding(*args),
                block_size=[8, 64],
                indexing="block_ptr",
                use_yz_grid=True,
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _embedding_kernel(x_flat, weight, out, out_size_0, out_size_1, x_size_0, x_size_1, x_flat_size_0, out_stride_0, out_stride_1, weight_stride_0, weight_stride_1, x_flat_stride_0, embedding_dim, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < x_size_0 * x_size_1
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < embedding_dim
    load = tl.load(tl.make_block_ptr(x_flat, [x_flat_size_0], [x_flat_stride_0], [offset_0], [_BLOCK_SIZE_0], [0]), boundary_check=[0], padding_option='zero')
    load_1 = tl.load(weight + (load[:, None] * weight_stride_0 + indices_1[None, :] * weight_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), load_1, boundary_check=[0, 1])

def embedding(x: torch.Tensor, weight: torch.Tensor):
    x_flat = x.reshape(-1)
    _, embedding_dim = weight.size()
    out = torch.empty([x_flat.size(0), embedding_dim], dtype=weight.dtype, device=weight.device)
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 64
    _embedding_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE_0), triton.cdiv(embedding_dim, _BLOCK_SIZE_1)](x_flat, weight, out, out.size(0), out.size(1), x.size(0), x.size(1), x_flat.size(0), out.stride(0), out.stride(1), weight.stride(0), weight.stride(1), x_flat.stride(0), embedding_dim, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out.view(*x.size(), embedding_dim)

def _embedding_make_precompiler(x: torch.Tensor, weight: torch.Tensor):
    x_flat = x.reshape(-1)
    _, embedding_dim = weight.size()
    out = torch.empty([x_flat.size(0), embedding_dim], dtype=weight.dtype, device=weight.device)
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_embedding_kernel)(x_flat, weight, out, out.size(0), out.size(1), x.size(0), x.size(1), x_flat.size(0), out.stride(0), out.stride(1), weight.stride(0), weight.stride(1), x_flat.stride(0), embedding_dim, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_attention_pointer(self):
        args = (
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
            torch.randn(1, 32, 512, 64, dtype=torch.float32, device=DEVICE),
        )
        self.assertExpectedInline(
            run_example(
                "attention",
                args,
                torch.nn.functional.scaled_dot_product_attention(*args),
                block_sizes=[64, 64],
                indexing="pointer",
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_compat import libdevice

import helion._testing.attention as _source_module

@triton.jit
def _attention_kernel(q_view, k_view, v_view, out, _BLOCK_SIZE_1: tl.constexpr, _RDIM_SIZE_3: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_blocks_0 = 32
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    indices_4 = tl.arange(0, _RDIM_SIZE_3).to(tl.int32)
    m_i = tl.full([1, _BLOCK_SIZE_1], float('-inf'), tl.float32)
    l_i = tl.full([1, _BLOCK_SIZE_1], 1.0, tl.float32)
    acc = tl.full([1, _BLOCK_SIZE_1, 64], 0.0, tl.float32)
    q = tl.load(q_view + (indices_0[:, None, None] * 32768 + indices_1[None, :, None] * 64 + indices_4[None, None, :] * 1), None)
    for offset_3 in range(0, 512, _BLOCK_SIZE_2):
        indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        q_copy = q
        m_i_copy = m_i
        l_i_copy = l_i
        acc_copy = acc
        k = tl.load(k_view + (indices_0[:, None, None] * 32768 + indices_4[None, :, None] * 1 + indices_3[None, None, :] * 64), None)
        qk = tl.dot(q_copy, k, input_precision='tf32')
        amax = tl.max(qk, 2)
        v_0 = 0.18033688
        v_1 = amax * v_0
        m_i = triton_helpers.maximum(m_i_copy, v_1)
        v_3 = 0.18033688
        v_4 = qk * v_3
        subscript = m_i[:, :, None]
        v_5 = v_4 - subscript
        v_6 = libdevice.exp2(v_5)
        l_ij = tl.sum(v_6, 2)
        v_7 = m_i_copy - m_i
        v_8 = libdevice.exp2(v_7)
        v_9 = l_i_copy * v_8
        l_i = v_9 + l_ij
        subscript_1 = v_8[:, :, None]
        v_11 = acc_copy * subscript_1
        v = tl.load(v_view + (indices_0[:, None, None] * 32768 + indices_3[None, :, None] * 64 + indices_4[None, None, :] * 1), None)
        acc = tl.dot(v_6, v, acc=v_11, input_precision='tf32')
    subscript_2 = l_i[:, :, None]
    v_12 = acc / subscript_2
    tl.store(out + (indices_0[:, None, None] * 32768 + indices_1[None, :, None] * 64 + indices_4[None, None, :] * 1), v_12, None)

def attention(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = q_in.size(-1)
    assert n_dim == v_in.size(-2)
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / _source_module.math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 64
    _RDIM_SIZE_3 = triton.next_power_of_2(64)
    _BLOCK_SIZE_2 = 64
    _attention_kernel[32 * triton.cdiv(512, _BLOCK_SIZE_1),](q_view, k_view, v_view, out, _BLOCK_SIZE_1, _RDIM_SIZE_3, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out.view(q_in.size())

def _attention_make_precompiler(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = q_in.size(-1)
    assert n_dim == v_in.size(-2)
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / _source_module.math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 64
    _RDIM_SIZE_3 = triton.next_power_of_2(64)
    _BLOCK_SIZE_2 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_attention_kernel)(q_view, k_view, v_view, out, _BLOCK_SIZE_1, _RDIM_SIZE_3, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_attention_block_pointer(self):
        args = (
            torch.randn(2, 32, 1024, 64, dtype=torch.float16, device=DEVICE),
            torch.randn(2, 32, 512, 64, dtype=torch.float16, device=DEVICE),
            torch.randn(2, 32, 512, 64, dtype=torch.float16, device=DEVICE),
        )
        self.assertExpectedInline(
            run_example(
                "attention",
                args,
                torch.nn.functional.scaled_dot_product_attention(*args),
                block_sizes=[128, 64],
                indexing="block_ptr",
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_compat import libdevice

import helion._testing.attention as _source_module

@triton.jit
def _attention_kernel(q_view, k_view, v_view, out, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_blocks_0 = 64
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    m_i = tl.full([1, _BLOCK_SIZE_1], float('-inf'), tl.float32)
    l_i = tl.full([1, _BLOCK_SIZE_1], 1.0, tl.float32)
    acc = tl.full([1, _BLOCK_SIZE_1, 64], 0.0, tl.float32)
    q = tl.load(tl.make_block_ptr(q_view, [64, 1024, 64], [65536, 64, 1], [offset_0, offset_1, 0], [1, _BLOCK_SIZE_1, 64], [2, 1, 0]), boundary_check=[0, 1, 2], padding_option='zero')
    for offset_3 in range(0, 512, _BLOCK_SIZE_2):
        q_copy = q
        m_i_copy = m_i
        l_i_copy = l_i
        acc_copy = acc
        k = tl.load(tl.make_block_ptr(k_view, [64, 64, 512], [32768, 1, 64], [offset_0, 0, offset_3], [1, 64, _BLOCK_SIZE_2], [2, 0, 1]), boundary_check=[0, 1, 2], padding_option='zero')
        qk = tl.dot(q_copy, k, input_precision='tf32')
        amax = tl.max(qk, 2)
        v_0 = tl.full([], 0.18033688, tl.float16)
        v_1 = amax * v_0
        v_2 = v_1.to(tl.float32)
        m_i = triton_helpers.maximum(m_i_copy, v_2)
        v_4 = tl.full([], 0.18033688, tl.float16)
        v_5 = qk * v_4
        subscript = m_i[:, :, None]
        v_6 = v_5.to(tl.float32)
        v_7 = v_6 - subscript
        v_8 = libdevice.exp2(v_7)
        l_ij = tl.sum(v_8, 2)
        v_9 = m_i_copy - m_i
        v_10 = libdevice.exp2(v_9)
        v_11 = l_i_copy * v_10
        l_i = v_11 + l_ij
        subscript_1 = v_10[:, :, None]
        v_13 = acc_copy * subscript_1
        v = tl.load(tl.make_block_ptr(v_view, [64, 512, 64], [32768, 64, 1], [offset_0, offset_3, 0], [1, _BLOCK_SIZE_2, 64], [2, 1, 0]), boundary_check=[0, 1, 2], padding_option='zero')
        v_14 = v_8.to(tl.float16)
        acc = tl.dot(v_14, v, acc=v_13, input_precision='tf32')
    subscript_2 = l_i[:, :, None]
    v_15 = acc / subscript_2
    v_16 = v_15.to(tl.float16)
    tl.store(tl.make_block_ptr(out, [64, 1024, 64], [65536, 64, 1], [offset_0, offset_1, 0], [1, _BLOCK_SIZE_1, 64], [2, 1, 0]), v_16, boundary_check=[0, 1, 2])

def attention(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = q_in.size(-1)
    assert n_dim == v_in.size(-2)
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / _source_module.math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 128
    _RDIM_SIZE_3 = triton.next_power_of_2(64)
    _BLOCK_SIZE_2 = 64
    _attention_kernel[64 * triton.cdiv(1024, _BLOCK_SIZE_1),](q_view, k_view, v_view, out, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out.view(q_in.size())

def _attention_make_precompiler(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = q_in.size(-1)
    assert n_dim == v_in.size(-2)
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / _source_module.math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 128
    _RDIM_SIZE_3 = triton.next_power_of_2(64)
    _BLOCK_SIZE_2 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_attention_kernel)(q_view, k_view, v_view, out, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )
