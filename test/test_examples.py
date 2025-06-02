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
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
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

import test.test_examples as _global_source0

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
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
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

import test.test_examples as _global_source0

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

import test.test_examples as _global_source0

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
    _mask_to = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), load, float('-inf'))
    amax = tl.reshape(tl.max(_mask_to, 1), [1, 1])
    v_0 = load - amax
    v_1 = tl_math.exp(v_0)
    _mask_to_1 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), v_1, 0)
    sum_1 = tl.reshape(tl.sum(_mask_to_1, 1), [1, 1])
    v_2 = v_1 / sum_1
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, 0], [1, _RDIM_SIZE_1], [1, 0]), v_2, boundary_check=[0, 1])

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
        _mask_to = tl.where(tl.broadcast_to(mask_1[None, :], [1, _REDUCTION_BLOCK_1]), load, float('-inf'))
        v_0 = triton_helpers.maximum(amax_acc, _mask_to)
        amax_acc = v_0
    amax = tl.reshape(tl.max(amax_acc, 1), [1, 1])
    sum_1_acc = tl.full([1, _REDUCTION_BLOCK_1], 0, tl.float32)
    for roffset_1 in range(0, _m, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < _m
        amax_copy = amax
        load_1 = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, roffset_1], [1, _REDUCTION_BLOCK_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        v_1 = load_1 - amax_copy
        v_2 = tl_math.exp(v_1)
        _mask_to_1 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _REDUCTION_BLOCK_1]), v_2, 0)
        v_3 = sum_1_acc + _mask_to_1
        sum_1_acc = v_3
    sum_1 = tl.reshape(tl.sum(sum_1_acc, 1), [1, 1])
    for roffset_1 in range(0, _m, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < _m
        amax_copy_1 = amax
        sum_1_copy = sum_1
        load_2 = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, roffset_1], [1, _REDUCTION_BLOCK_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        v_4 = load_2 - amax_copy_1
        v_5 = tl_math.exp(v_4)
        v_6 = v_5 / sum_1_copy
        tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, roffset_1], [1, _REDUCTION_BLOCK_1], [1, 0]), v_6, boundary_check=[0, 1])

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
    _mask_to = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), values, float('-inf'))
    amax = tl.reshape(tl.max(_mask_to, 1), [1, 1])
    v_0 = values - amax
    v_1 = tl_math.exp(v_0)
    _mask_to_1 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _RDIM_SIZE_1]), v_1, 0)
    sum_exp = tl.reshape(tl.sum(_mask_to_1, 1), [1, 1])
    v_2 = v_1 / sum_exp
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, 0], [1, _RDIM_SIZE_1], [1, 0]), v_2, boundary_check=[0, 1])

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
    for offset_2 in range(0, n.to(tl.int32), _BLOCK_SIZE_1):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_2 < n
        mi_copy = mi
        di_copy = di
        values = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_2[None, :] * x_stride_1), mask_1[None, :], other=0)
        _mask_to = tl.where(tl.broadcast_to(mask_1[None, :], [1, _BLOCK_SIZE_1]), values, float('-inf'))
        local_amax = tl.max(_mask_to, 1)
        mi = triton_helpers.maximum(mi_copy, local_amax)
        v_1 = mi_copy - mi
        v_2 = tl_math.exp(v_1)
        v_3 = di_copy * v_2
        subscript = mi[:, None]
        v_4 = values - subscript
        v_5 = tl_math.exp(v_4)
        _mask_to_1 = tl.where(tl.broadcast_to(mask_1[None, :], [1, _BLOCK_SIZE_1]), v_5, 0)
        sum_1 = tl.sum(_mask_to_1, 1)
        di = v_3 + sum_1
    for offset_2 in range(0, n.to(tl.int32), _BLOCK_SIZE_1):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_2 = indices_2 < n
        mi_copy_1 = mi
        di_copy_1 = di
        values = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_2[None, :] * x_stride_1), mask_2[None, :], other=0)
        subscript_1 = mi_copy_1[:, None]
        v_7 = values - subscript_1
        v_8 = tl_math.exp(v_7)
        subscript_2 = di_copy_1[:, None]
        v_9 = v_8 / subscript_2
        tl.store(out + (indices_0[:, None] * out_stride_0 + indices_2[None, :] * out_stride_1), v_9, mask_2[None, :])

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
def _softmax_two_pass_kernel(x, out, out_size_0, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, m, n, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < m
    mi = tl.full([_BLOCK_SIZE_0], float('-inf'), tl.float32)
    di = tl.full([_BLOCK_SIZE_0], 0.0, tl.float32)
    for offset_2 in range(0, n.to(tl.int32), _BLOCK_SIZE_1):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_2 < n
        mi_copy = mi
        di_copy = di
        values = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        _mask_to = tl.where(mask_0[:, None] & mask_1[None, :], values, float('-inf'))
        local_amax = tl.max(_mask_to, 1)
        mi = triton_helpers.maximum(mi_copy, local_amax)
        v_1 = mi_copy - mi
        v_2 = tl_math.exp(v_1)
        v_3 = di_copy * v_2
        subscript = mi[:, None]
        v_4 = values - subscript
        v_5 = tl_math.exp(v_4)
        _mask_to_1 = tl.where(mask_0[:, None] & mask_1[None, :], v_5, 0)
        sum_1 = tl.sum(_mask_to_1, 1)
        di = v_3 + sum_1
    for offset_2 in range(0, n.to(tl.int32), _BLOCK_SIZE_1):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mi_copy_1 = mi
        di_copy_1 = di
        values = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        subscript_1 = mi_copy_1[:, None]
        v_7 = values - subscript_1
        v_8 = tl_math.exp(v_7)
        subscript_2 = di_copy_1[:, None]
        v_9 = v_8 / subscript_2
        tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_9, boundary_check=[0, 1])

def softmax_two_pass(x: torch.Tensor):
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = 8
    block_size_n = 64
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 64
    _softmax_two_pass_kernel[triton.cdiv(m, _BLOCK_SIZE_0),](x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), m, n, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _softmax_two_pass_make_precompiler(x: torch.Tensor):
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = 8
    block_size_n = 64
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_softmax_two_pass_kernel)(x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), m, n, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
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
def _embedding_kernel(x_flat, weight, out, x_flat_size_0, out_stride_0, out_stride_1, weight_stride_0, weight_stride_1, x_flat_stride_0, embedding_dim, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = x_flat_size_0
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < embedding_dim
    load = tl.load(x_flat + indices_0 * x_flat_stride_0, None)
    load_1 = tl.load(weight + (load[:, None] * weight_stride_0 + indices_1[None, :] * weight_stride_1), mask_1[None, :], other=0)
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), load_1, mask_1[None, :])

def embedding(x: torch.Tensor, weight: torch.Tensor):
    x_flat = x.reshape(-1)
    _, embedding_dim = weight.size()
    out = torch.empty([x_flat.size(0), embedding_dim], dtype=weight.dtype, device=weight.device)
    _BLOCK_SIZE_1 = 256
    _embedding_kernel[x_flat.size(0) * triton.cdiv(embedding_dim, _BLOCK_SIZE_1),](x_flat, weight, out, x_flat.size(0), out.stride(0), out.stride(1), weight.stride(0), weight.stride(1), x_flat.stride(0), embedding_dim, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out.view(*x.size(), embedding_dim)

def _embedding_make_precompiler(x: torch.Tensor, weight: torch.Tensor):
    x_flat = x.reshape(-1)
    _, embedding_dim = weight.size()
    out = torch.empty([x_flat.size(0), embedding_dim], dtype=weight.dtype, device=weight.device)
    _BLOCK_SIZE_1 = 256
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_embedding_kernel)(x_flat, weight, out, x_flat.size(0), out.stride(0), out.stride(1), weight.stride(0), weight.stride(1), x_flat.stride(0), embedding_dim, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
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
def _embedding_kernel(x_flat, weight, out, out_size_0, out_size_1, x_flat_size_0, out_stride_0, out_stride_1, weight_stride_0, weight_stride_1, x_flat_stride_0, embedding_dim, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_flat_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
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
    _embedding_kernel[triton.cdiv(x_flat.size(0), _BLOCK_SIZE_0), triton.cdiv(embedding_dim, _BLOCK_SIZE_1)](x_flat, weight, out, out.size(0), out.size(1), x_flat.size(0), out.stride(0), out.stride(1), weight.stride(0), weight.stride(1), x_flat.stride(0), embedding_dim, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out.view(*x.size(), embedding_dim)

def _embedding_make_precompiler(x: torch.Tensor, weight: torch.Tensor):
    x_flat = x.reshape(-1)
    _, embedding_dim = weight.size()
    out = torch.empty([x_flat.size(0), embedding_dim], dtype=weight.dtype, device=weight.device)
    _BLOCK_SIZE_0 = 8
    _BLOCK_SIZE_1 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_embedding_kernel)(x_flat, weight, out, out.size(0), out.size(1), x_flat.size(0), out.stride(0), out.stride(1), weight.stride(0), weight.stride(1), x_flat.stride(0), embedding_dim, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
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

import math
import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_compat import libdevice

@triton.jit
def _attention_kernel(q_view, k_view, v_view, out, _BLOCK_SIZE_1: tl.constexpr, _RDIM_SIZE_2: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    num_blocks_0 = 32
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    indices_4 = tl.arange(0, _RDIM_SIZE_2).to(tl.int32)
    m_i = tl.full([1, _BLOCK_SIZE_1], float('-inf'), tl.float32)
    l_i = tl.full([1, _BLOCK_SIZE_1], 1.0, tl.float32)
    acc = tl.full([1, _BLOCK_SIZE_1, 64], 0.0, tl.float32)
    q = tl.load(q_view + (indices_0[:, None, None] * 32768 + indices_1[None, :, None] * 64 + indices_4[None, None, :] * 1), None)
    for offset_2 in range(0, 512, _BLOCK_SIZE_3):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
        q_copy = q
        m_i_copy = m_i
        l_i_copy = l_i
        acc_copy = acc
        k = tl.load(k_view + (indices_0[:, None, None] * 32768 + indices_4[None, :, None] * 1 + indices_2[None, None, :] * 64), None)
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
        v = tl.load(v_view + (indices_0[:, None, None] * 32768 + indices_2[None, :, None] * 64 + indices_4[None, None, :] * 1), None)
        acc = tl.dot(v_6, v, acc=v_11, input_precision='tf32')
    subscript_2 = l_i[:, :, None]
    v_12 = acc / subscript_2
    tl.store(out + (indices_0[:, None, None] * 32768 + indices_1[None, :, None] * 64 + indices_4[None, None, :] * 1), v_12, None)

def attention(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = 64
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 64
    _RDIM_SIZE_2 = 64
    _BLOCK_SIZE_3 = 64
    _attention_kernel[32 * triton.cdiv(512, _BLOCK_SIZE_1),](q_view, k_view, v_view, out, _BLOCK_SIZE_1, _RDIM_SIZE_2, _BLOCK_SIZE_3, num_warps=4, num_stages=3)
    return out.view(q_in.size())

def _attention_make_precompiler(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = 64
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 64
    _RDIM_SIZE_2 = 64
    _BLOCK_SIZE_3 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_attention_kernel)(q_view, k_view, v_view, out, _BLOCK_SIZE_1, _RDIM_SIZE_2, _BLOCK_SIZE_3, num_warps=4, num_stages=3)""",
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

import math
import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_compat import libdevice

@triton.jit
def _attention_kernel(q_view, k_view, v_view, out, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    num_blocks_0 = 64
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    m_i = tl.full([1, _BLOCK_SIZE_1], float('-inf'), tl.float32)
    l_i = tl.full([1, _BLOCK_SIZE_1], 1.0, tl.float32)
    acc = tl.full([1, _BLOCK_SIZE_1, 64], 0.0, tl.float32)
    q = tl.load(tl.make_block_ptr(q_view, [64, 1024, 64], [65536, 64, 1], [offset_0, offset_1, 0], [1, _BLOCK_SIZE_1, 64], [2, 1, 0]), boundary_check=[0, 1, 2], padding_option='zero')
    for offset_2 in range(0, 512, _BLOCK_SIZE_3):
        q_copy = q
        m_i_copy = m_i
        l_i_copy = l_i
        acc_copy = acc
        k = tl.load(tl.make_block_ptr(k_view, [64, 64, 512], [32768, 1, 64], [offset_0, 0, offset_2], [1, 64, _BLOCK_SIZE_3], [2, 0, 1]), boundary_check=[0, 1, 2], padding_option='zero')
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
        v = tl.load(tl.make_block_ptr(v_view, [64, 512, 64], [32768, 64, 1], [offset_0, offset_2, 0], [1, _BLOCK_SIZE_3, 64], [2, 1, 0]), boundary_check=[0, 1, 2], padding_option='zero')
        v_14 = v_8.to(tl.float16)
        acc = tl.dot(v_14, v, acc=v_13, input_precision='tf32')
    subscript_2 = l_i[:, :, None]
    v_15 = acc / subscript_2
    v_16 = v_15.to(tl.float16)
    tl.store(tl.make_block_ptr(out, [64, 1024, 64], [65536, 64, 1], [offset_0, offset_1, 0], [1, _BLOCK_SIZE_1, 64], [2, 1, 0]), v_16, boundary_check=[0, 1, 2])

def attention(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = 64
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 128
    _RDIM_SIZE_2 = 64
    _BLOCK_SIZE_3 = 64
    _attention_kernel[64 * triton.cdiv(1024, _BLOCK_SIZE_1),](q_view, k_view, v_view, out, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)
    return out.view(q_in.size())

def _attention_make_precompiler(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = 64
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 128
    _RDIM_SIZE_2 = 64
    _BLOCK_SIZE_3 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_attention_kernel)(q_view, k_view, v_view, out, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)""",
        )

    def test_attention_dynamic(self):
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
                fn_name="attention_dynamic",
            ),
            """\
from __future__ import annotations

import math
import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_compat import libdevice

@triton.jit
def _attention_kernel(q_view, k_view, v_view, out, k_view_size_0, k_view_size_2, out_size_0, out_size_1, q_in_size_1, q_view_size_0, q_view_size_1, v_view_size_0, v_view_size_1, k_view_stride_0, k_view_stride_1, k_view_stride_2, out_stride_0, out_stride_1, out_stride_2, q_view_stride_0, q_view_stride_1, q_view_stride_2, v_view_stride_0, v_view_stride_1, v_view_stride_2, m_dim, n_dim, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    num_blocks_0 = q_in_size_1
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < m_dim
    m_i = tl.full([1, _BLOCK_SIZE_1], float('-inf'), tl.float32)
    l_i = tl.full([1, _BLOCK_SIZE_1], 1.0, tl.float32)
    acc = tl.full([1, _BLOCK_SIZE_1, 64], 0.0, tl.float32)
    q = tl.load(tl.make_block_ptr(q_view, [q_view_size_0, q_view_size_1, 64], [q_view_stride_0, q_view_stride_1, q_view_stride_2], [offset_0, offset_1, 0], [1, _BLOCK_SIZE_1, 64], [2, 1, 0]), boundary_check=[0, 1, 2], padding_option='zero')
    for offset_2 in range(0, n_dim.to(tl.int32), _BLOCK_SIZE_3):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
        mask_3 = indices_2 < n_dim
        q_copy = q
        m_i_copy = m_i
        l_i_copy = l_i
        acc_copy = acc
        k = tl.load(tl.make_block_ptr(k_view, [k_view_size_0, 64, k_view_size_2], [k_view_stride_0, k_view_stride_1, k_view_stride_2], [offset_0, 0, offset_2], [1, 64, _BLOCK_SIZE_3], [2, 0, 1]), boundary_check=[0, 1, 2], padding_option='zero')
        qk = tl.dot(q_copy, k, input_precision='tf32')
        _mask_to_2 = tl.where(tl.broadcast_to(mask_1[None, :, None] & mask_3[None, None, :], [1, _BLOCK_SIZE_1, _BLOCK_SIZE_3]), qk, float('-inf'))
        amax = tl.max(_mask_to_2, 2)
        v_0 = 0.18033688
        v_1 = amax * v_0
        m_i = triton_helpers.maximum(m_i_copy, v_1)
        v_3 = 0.18033688
        v_4 = qk * v_3
        subscript = m_i[:, :, None]
        v_5 = v_4 - subscript
        v_6 = libdevice.exp2(v_5)
        _mask_to_3 = tl.where(tl.broadcast_to(mask_1[None, :, None] & mask_3[None, None, :], [1, _BLOCK_SIZE_1, _BLOCK_SIZE_3]), v_6, 0)
        l_ij = tl.sum(_mask_to_3, 2)
        v_7 = m_i_copy - m_i
        v_8 = libdevice.exp2(v_7)
        v_9 = l_i_copy * v_8
        l_i = v_9 + l_ij
        subscript_1 = v_8[:, :, None]
        v_11 = acc_copy * subscript_1
        v = tl.load(tl.make_block_ptr(v_view, [v_view_size_0, v_view_size_1, 64], [v_view_stride_0, v_view_stride_1, v_view_stride_2], [offset_0, offset_2, 0], [1, _BLOCK_SIZE_3, 64], [2, 1, 0]), boundary_check=[0, 1, 2], padding_option='zero')
        acc = tl.dot(_mask_to_3, v, acc=v_11, input_precision='tf32')
    subscript_2 = l_i[:, :, None]
    v_12 = acc / subscript_2
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1, 64], [out_stride_0, out_stride_1, out_stride_2], [offset_0, offset_1, 0], [1, _BLOCK_SIZE_1, 64], [2, 1, 0]), v_12, boundary_check=[0, 1, 2])

def attention(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = 64
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 32
    _RDIM_SIZE_2 = 64
    _BLOCK_SIZE_3 = 16
    _attention_kernel[q_in.size(1) * triton.cdiv(m_dim, _BLOCK_SIZE_1),](q_view, k_view, v_view, out, k_view.size(0), k_view.size(2), out.size(0), out.size(1), q_in.size(1), q_view.size(0), q_view.size(1), v_view.size(0), v_view.size(1), k_view.stride(0), k_view.stride(1), k_view.stride(2), out.stride(0), out.stride(1), out.stride(2), q_view.stride(0), q_view.stride(1), q_view.stride(2), v_view.stride(0), v_view.stride(1), v_view.stride(2), m_dim, n_dim, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=1, num_stages=2)
    return out.view(q_in.size())

def _attention_make_precompiler(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = 64
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    _BLOCK_SIZE_1 = 32
    _RDIM_SIZE_2 = 64
    _BLOCK_SIZE_3 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_attention_kernel)(q_view, k_view, v_view, out, k_view.size(0), k_view.size(2), out.size(0), out.size(1), q_in.size(1), q_view.size(0), q_view.size(1), v_view.size(0), v_view.size(1), k_view.stride(0), k_view.stride(1), k_view.stride(2), out.stride(0), out.stride(1), out.stride(2), q_view.stride(0), q_view.stride(1), q_view.stride(2), v_view.stride(0), v_view.stride(1), v_view.stride(2), m_dim, n_dim, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=1, num_stages=2)""",
        )

    def test_concat(self):
        args = (
            torch.randn(512, 500, device=DEVICE),
            torch.randn(512, 512, device=DEVICE),
        )
        self.assertExpectedInline(
            run_example(
                "concatenate",
                args,
                torch.cat(args, dim=1),
                fn_name="concat2d_dim1",
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _concat2d_dim1_kernel(out, x, y, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr):
    num_blocks_0 = tl.cdiv(out_size_1, _BLOCK_SIZE_1)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_1 = pid_0 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < out_size_1
    offset_0 = pid_1 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    v_0 = x_size_1.to(tl.int32)
    v_1 = indices_1 < v_0
    subscript = v_1[None, :]
    x_part = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :] & subscript, other=0)
    v_2 = x_size_1.to(tl.int32)
    v_3 = indices_1 - v_2
    v_4 = x_size_1.to(tl.int32)
    v_5 = indices_1 >= v_4
    subscript_1 = v_5[None, :]
    y_part = tl.load(y + (indices_0[:, None] * y_stride_0 + v_3[None, :] * y_stride_1), mask_0[:, None] & mask_1[None, :] & subscript_1, other=0)
    v_6 = x_size_1.to(tl.int32)
    v_7 = indices_1 < v_6
    subscript_2 = v_7[None, :]
    v_8 = tl.where(subscript_2, x_part, y_part)
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_8, mask_0[:, None] & mask_1[None, :])

def concat2d_dim1(x: torch.Tensor, y: torch.Tensor):
    assert x.size(0) == y.size(0)
    out = torch.empty([x.size(0), x.size(1) + y.size(1)], dtype=x.dtype, device=x.device)
    _BLOCK_SIZE_1 = 1024
    _BLOCK_SIZE_0 = 4
    _concat2d_dim1_kernel[triton.cdiv(out.size(1), _BLOCK_SIZE_1) * triton.cdiv(x.size(0), _BLOCK_SIZE_0),](out, x, y, out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=2, num_stages=3)
    return out

def _concat2d_dim1_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    assert x.size(0) == y.size(0)
    out = torch.empty([x.size(0), x.size(1) + y.size(1)], dtype=x.dtype, device=x.device)
    _BLOCK_SIZE_1 = 1024
    _BLOCK_SIZE_0 = 4
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_concat2d_dim1_kernel)(out, x, y, out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=2, num_stages=3)""",
        )

    def test_concat_block_ptr(self):
        args = (
            torch.randn(222, 100, device=DEVICE),
            torch.randn(222, 151, device=DEVICE),
        )
        self.assertExpectedInline(
            run_example(
                "concatenate",
                args,
                torch.cat(args, dim=1),
                fn_name="concat2d_dim1",
                indexing="block_ptr",
                block_size=[128, 64],
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _concat2d_dim1_kernel(x, out, y, out_size_0, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < out_size_1
    v_0 = x_size_1.to(tl.int32)
    v_1 = indices_1 < v_0
    subscript = v_1[None, :]
    x_part = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :] & subscript, other=0)
    v_2 = x_size_1.to(tl.int32)
    v_3 = indices_1 - v_2
    v_4 = x_size_1.to(tl.int32)
    v_5 = indices_1 >= v_4
    subscript_1 = v_5[None, :]
    y_part = tl.load(y + (indices_0[:, None] * y_stride_0 + v_3[None, :] * y_stride_1), mask_0[:, None] & mask_1[None, :] & subscript_1, other=0)
    v_6 = x_size_1.to(tl.int32)
    v_7 = indices_1 < v_6
    subscript_2 = v_7[None, :]
    v_8 = tl.where(subscript_2, x_part, y_part)
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_8, boundary_check=[0, 1])

def concat2d_dim1(x: torch.Tensor, y: torch.Tensor):
    assert x.size(0) == y.size(0)
    out = torch.empty([x.size(0), x.size(1) + y.size(1)], dtype=x.dtype, device=x.device)
    _BLOCK_SIZE_0 = 128
    _BLOCK_SIZE_1 = 64
    _concat2d_dim1_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0) * triton.cdiv(out.size(1), _BLOCK_SIZE_1),](x, out, y, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _concat2d_dim1_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    assert x.size(0) == y.size(0)
    out = torch.empty([x.size(0), x.size(1) + y.size(1)], dtype=x.dtype, device=x.device)
    _BLOCK_SIZE_0 = 128
    _BLOCK_SIZE_1 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_concat2d_dim1_kernel)(x, out, y, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_jagged_dense_add(self):
        mod = import_path(examples_dir / "jagged_dense_add.py")
        args = (
            *mod.random_jagged_2d(500, 5000, device=DEVICE),
            torch.randn(500, 5000, device=DEVICE),
        )
        self.assertExpectedInline(
            run_example(
                "jagged_dense_add",
                args,
                mod.jagged_dense_add_2d_reference(*args),
                fn_name="jagged_dense_add_2d",
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _jagged_dense_add_2d_kernel(x_offsets, x_data, y, out, out_size_0, out_size_1, x_offsets_size_0, y_size_0, y_size_1, out_stride_0, out_stride_1, x_data_stride_0, x_offsets_stride_0, y_stride_0, y_stride_1, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    starts = tl.load(tl.make_block_ptr(x_offsets, [x_offsets_size_0], [x_offsets_stride_0], [offset_0], [1], [0]), boundary_check=[0], padding_option='zero')
    v_0 = tl.full([], 1, tl.int32)
    v_1 = indices_0 + v_0
    ends = tl.load(x_offsets + v_1 * x_offsets_stride_0, None)
    v_2 = ends - starts
    max_nnz = tl.max(v_2, 0)
    for offset_1 in range(0, max_nnz.to(tl.int32), _BLOCK_SIZE_1):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < max_nnz
        starts_copy = starts
        v_2_copy = v_2
        subscript = starts_copy[:, None]
        subscript_1 = indices_1[None, :]
        v_3 = subscript_1.to(tl.int64)
        v_4 = subscript + v_3
        subscript_2 = indices_1[None, :]
        subscript_3 = v_2_copy[:, None]
        v_5 = subscript_2.to(tl.int64)
        v_6 = v_5 < subscript_3
        x_slice = tl.load(x_data + v_4 * x_data_stride_0, mask_1[None, :] & v_6, other=0)
        load_1 = tl.load(tl.make_block_ptr(y, [y_size_0, y_size_1], [y_stride_0, y_stride_1], [offset_0, offset_1], [1, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        v_7 = load_1 + x_slice
        tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_1], [1, _BLOCK_SIZE_1], [1, 0]), v_7, boundary_check=[0, 1])
    for offset_2 in range(max_nnz.to(tl.int32), y_size_1.to(tl.int32), _BLOCK_SIZE_2):
        load = tl.load(tl.make_block_ptr(y, [y_size_0, y_size_1], [y_stride_0, y_stride_1], [offset_0, offset_2], [1, _BLOCK_SIZE_2], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_2], [1, _BLOCK_SIZE_2], [1, 0]), load, boundary_check=[0, 1])

def jagged_dense_add_2d(x_data: torch.Tensor, x_offsets: torch.Tensor, y: torch.Tensor):
    \"\"\"
    Add a jagged-prefix sparse tensor (x_data, x_offsets) to a dense matrix y
    and return the dense result.

    Args
    ----
    x_data    : 1-D tensor holding all non-zero elements row-by-row.
    x_offsets : (num_rows + 1) tensor.  Row i is the slice
                x_data[x_offsets[i] : x_offsets[i+1]] (length K_i).
    y: (num_rows, N) tensor, N >= max(K_i).

    Returns
    -------
    result : dense + jagged, shape (num_rows, N).
    \"\"\"
    num_rows = y.size(0)
    assert (*x_offsets.size(),) == (num_rows + 1,)
    out = torch.zeros_like(y)
    _BLOCK_SIZE_1 = 512
    _BLOCK_SIZE_2 = 512
    _jagged_dense_add_2d_kernel[num_rows,](x_offsets, x_data, y, out, out.size(0), out.size(1), x_offsets.size(0), y.size(0), y.size(1), out.stride(0), out.stride(1), x_data.stride(0), x_offsets.stride(0), y.stride(0), y.stride(1), _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=8, num_stages=4)
    return out

def _jagged_dense_add_2d_make_precompiler(x_data: torch.Tensor, x_offsets: torch.Tensor, y: torch.Tensor):
    \"\"\"
    Add a jagged-prefix sparse tensor (x_data, x_offsets) to a dense matrix y
    and return the dense result.

    Args
    ----
    x_data    : 1-D tensor holding all non-zero elements row-by-row.
    x_offsets : (num_rows + 1) tensor.  Row i is the slice
                x_data[x_offsets[i] : x_offsets[i+1]] (length K_i).
    y: (num_rows, N) tensor, N >= max(K_i).

    Returns
    -------
    result : dense + jagged, shape (num_rows, N).
    \"\"\"
    num_rows = y.size(0)
    assert (*x_offsets.size(),) == (num_rows + 1,)
    out = torch.zeros_like(y)
    _BLOCK_SIZE_1 = 512
    _BLOCK_SIZE_2 = 512
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_jagged_dense_add_2d_kernel)(x_offsets, x_data, y, out, out.size(0), out.size(1), x_offsets.size(0), y.size(0), y.size(1), out.stride(0), out.stride(1), x_data.stride(0), x_offsets.stride(0), y.stride(0), y.stride(1), _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=8, num_stages=4)""",
        )

    @unittest.skip("TODO(yf225): fix occasional numerical error")
    def test_moe_matmul_ogs(self):
        mod = import_path(examples_dir / "moe_matmul_ogs.py")

        B = 1000  # tokens / rows
        K = 500  # hidden size
        N = 200  # output size
        n_experts = 30
        A = torch.randn(B, K, device=DEVICE, dtype=torch.float16)
        W = torch.randn(n_experts, K, N, device=DEVICE, dtype=torch.float16)
        top1_expert_per_token = torch.randint(n_experts, (B,), device=DEVICE)

        args = (A, W, top1_expert_per_token)
        helion_kernel_args = mod.moe_matmul_ogs_helion_kernel_args_gen(
            A, W, top1_expert_per_token
        )
        self.assertExpectedInline(
            run_example(
                "moe_matmul_ogs",
                helion_kernel_args,
                mod.moe_matmul_ogs_reference(*args),
                block_sizes=[[16, 16], 16],
                l2_grouping=4,
            ),
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _moe_matmul_ogs_kernel(expert_token_offsets, expert_token_counts, sorted_to_orig_token_idx, A, W, C, A_stride_0, A_stride_1, C_stride_0, C_stride_1, W_stride_0, W_stride_1, W_stride_2, expert_token_counts_stride_0, expert_token_offsets_stride_0, sorted_to_orig_token_idx_stride_0, max_T_per_expert, N, K, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    start = tl.load(expert_token_offsets + indices_0 * expert_token_offsets_stride_0, None)
    num_tokens = tl.load(expert_token_counts + indices_0 * expert_token_counts_stride_0, None)
    for offset_1 in range(0, max_T_per_expert.to(tl.int32), _BLOCK_SIZE_1):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < max_T_per_expert
        for offset_2 in range(0, N.to(tl.int32), _BLOCK_SIZE_2):
            indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
            mask_2 = indices_2 < N
            num_tokens_copy = num_tokens
            start_copy = start
            v_0 = num_tokens_copy[None]
            v_1 = indices_1 < v_0
            v_2 = tl.full([], 0, tl.int32)
            v_3 = v_2[None]
            v_4 = tl.where(v_1, indices_1, v_3)
            v_5 = start_copy[None]
            v_6 = v_5 + v_4
            squeeze = tl.reshape(v_6, [_BLOCK_SIZE_1])
            expert_orig_token_indices = tl.load(sorted_to_orig_token_idx + squeeze * sorted_to_orig_token_idx_stride_0, mask_1, other=0)
            acc = tl.full([_BLOCK_SIZE_1, _BLOCK_SIZE_2], 0.0, tl.float32)
            for offset_3 in range(0, K.to(tl.int32), _BLOCK_SIZE_3):
                indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
                mask_3 = indices_3 < K
                expert_orig_token_indices_copy = expert_orig_token_indices
                acc_copy = acc
                A_frag = tl.load(A + (expert_orig_token_indices_copy[:, None] * A_stride_0 + indices_3[None, :] * A_stride_1), mask_1[:, None] & mask_3[None, :], other=0)
                W_frag = tl.load(W + (indices_0 * W_stride_0 + indices_3[:, None] * W_stride_1 + indices_2[None, :] * W_stride_2), mask_3[:, None] & mask_2[None, :], other=0)
                acc = tl.dot(A_frag, W_frag, acc=acc_copy, input_precision='tf32')
            existing_values = tl.load(C + (expert_orig_token_indices[:, None] * C_stride_0 + indices_2[None, :] * C_stride_1), mask_1[:, None] & mask_2[None, :], other=0)
            view = tl.reshape(v_1, [_BLOCK_SIZE_1, 1])
            mask_2d = tl.broadcast_to(view, [_BLOCK_SIZE_1, _BLOCK_SIZE_2])
            v_7 = acc.to(tl.float16)
            v_8 = tl.where(mask_2d, v_7, existing_values)
            tl.store(C + (expert_orig_token_indices[:, None] * C_stride_0 + indices_2[None, :] * C_stride_1), v_8, mask_1[:, None] & mask_2[None, :])

def moe_matmul_ogs(A: torch.Tensor, W: torch.Tensor, expert_token_counts: torch.Tensor, expert_token_offsets: torch.Tensor, sorted_to_orig_token_idx: torch.Tensor, max_T_per_expert_tensor: torch.Tensor):
    T, K = A.shape
    E, _, N = W.shape
    max_T_per_expert = max_T_per_expert_tensor.numel()
    C = torch.zeros(T, N, dtype=torch.promote_types(A.dtype, W.dtype), device=A.device)
    _BLOCK_SIZE_2 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_3 = 16
    _moe_matmul_ogs_kernel[E,](expert_token_offsets, expert_token_counts, sorted_to_orig_token_idx, A, W, C, A.stride(0), A.stride(1), C.stride(0), C.stride(1), W.stride(0), W.stride(1), W.stride(2), expert_token_counts.stride(0), expert_token_offsets.stride(0), sorted_to_orig_token_idx.stride(0), max_T_per_expert, N, K, _BLOCK_SIZE_2, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)
    return C

def _moe_matmul_ogs_make_precompiler(A: torch.Tensor, W: torch.Tensor, expert_token_counts: torch.Tensor, expert_token_offsets: torch.Tensor, sorted_to_orig_token_idx: torch.Tensor, max_T_per_expert_tensor: torch.Tensor):
    T, K = A.shape
    E, _, N = W.shape
    max_T_per_expert = max_T_per_expert_tensor.numel()
    C = torch.zeros(T, N, dtype=torch.promote_types(A.dtype, W.dtype), device=A.device)
    _BLOCK_SIZE_2 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_3 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_moe_matmul_ogs_kernel)(expert_token_offsets, expert_token_counts, sorted_to_orig_token_idx, A, W, C, A.stride(0), A.stride(1), C.stride(0), C.stride(1), W.stride(0), W.stride(1), W.stride(2), expert_token_counts.stride(0), expert_token_offsets.stride(0), sorted_to_orig_token_idx.stride(0), max_T_per_expert, N, K, _BLOCK_SIZE_2, _BLOCK_SIZE_1, _BLOCK_SIZE_3, num_warps=4, num_stages=3)""",
        )


if __name__ == "__main__":
    unittest.main()
