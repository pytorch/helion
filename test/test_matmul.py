from __future__ import annotations

from pathlib import Path
import unittest

from expecttest import TestCase
import torch

import helion
from helion import Config
from helion._compat import get_triton_tensor_descriptor_import_path
from helion._compat import supports_tensor_descriptor
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import import_path
import helion.language as hl

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
examples_dir = Path(__file__).parent.parent / "examples"
examples_matmul = import_path(examples_dir / "matmul.py").matmul


@helion.kernel
def matmul_with_addmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel
def matmul_without_addmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True)
def matmul_static_shapes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc += torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


class TestMatmul(TestCase):
    maxDiff = 16384

    def test_matmul0(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_without_addmm,
            args,
            block_sizes=[[16, 16], 16],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_without_addmm_kernel(x, y, out, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, m, n, k, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
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
        acc_copy = acc
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_2[None, :] * x_stride_1), mask_0[:, None] & mask_2[None, :], other=0)
        load_1 = tl.load(y + (indices_2[:, None] * y_stride_0 + indices_1[None, :] * y_stride_1), mask_2[:, None] & mask_1[None, :], other=0)
        mm = tl.dot(load, load_1, input_precision='tf32')
        acc = acc_copy + mm
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), acc, mask_0[:, None] & mask_1[None, :])

def matmul_without_addmm(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_without_addmm_kernel[triton.cdiv(m, _BLOCK_SIZE_0) * triton.cdiv(n, _BLOCK_SIZE_1),](x, y, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _matmul_without_addmm_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_matmul_without_addmm_kernel)(x, y, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_matmul1(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            examples_matmul,
            args,
            block_sizes=[[16, 16], 16],
            loop_order=[1, 0],
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(x, y, out, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_blocks_0 = tl.cdiv(128, _BLOCK_SIZE_1)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_1 = pid_0 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    offset_0 = pid_1 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
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
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_kernel[triton.cdiv(128, _BLOCK_SIZE_1) * triton.cdiv(128, _BLOCK_SIZE_0),](x, y, out, _BLOCK_SIZE_1, _BLOCK_SIZE_0, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _matmul_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_2 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_matmul_kernel)(x, y, out, _BLOCK_SIZE_1, _BLOCK_SIZE_0, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_matmul3(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_with_addmm,
            args,
            block_sizes=[[16, 16], 16],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_with_addmm_kernel(x, y, out, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, m, n, k, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
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
        acc_copy = acc
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_2[None, :] * x_stride_1), mask_0[:, None] & mask_2[None, :], other=0)
        load_1 = tl.load(y + (indices_2[:, None] * y_stride_0 + indices_1[None, :] * y_stride_1), mask_2[:, None] & mask_1[None, :], other=0)
        acc = tl.dot(load, load_1, acc=acc_copy, input_precision='tf32')
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), acc, mask_0[:, None] & mask_1[None, :])

def matmul_with_addmm(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_with_addmm_kernel[triton.cdiv(m, _BLOCK_SIZE_0) * triton.cdiv(n, _BLOCK_SIZE_1),](x, y, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _matmul_with_addmm_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_matmul_with_addmm_kernel)(x, y, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_matmul_block_ptr(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            examples_matmul,
            args,
            block_sizes=[[16, 16], 16],
            l2_grouping=4,
            indexing="block_ptr",
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedInline(
            code,
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
    offset_1 = pid_1 * _BLOCK_SIZE_1
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in range(0, 128, _BLOCK_SIZE_2):
        acc_copy = acc
        load = tl.load(tl.make_block_ptr(x, [128, 128], [128, 1], [offset_0, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_2], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        load_1 = tl.load(tl.make_block_ptr(y, [128, 128], [128, 1], [offset_2, offset_1], [_BLOCK_SIZE_2, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
        acc = tl.dot(load, load_1, acc=acc_copy, input_precision='tf32')
    tl.store(tl.make_block_ptr(out, [128, 128], [128, 1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), acc, boundary_check=[0, 1])

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

    @unittest.skipIf(not supports_tensor_descriptor(), "TensorDescriptor not supported")
    def test_matmul_tensor_descriptor(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        config = Config(
            block_sizes=[[16, 16], 16],
            l2_grouping=4,
            indexing="tensor_descriptor",
        )
        # Note TensorDescriptor doesn't run on older cards
        code = examples_matmul.bind(args).to_triton_code(config)
        self.assertExpectedInline(
            code,
            f"""\
from __future__ import annotations

import torch
import triton
import triton.language as tl
{get_triton_tensor_descriptor_import_path()}

@triton.jit
def _matmul_kernel(x_desc, y_desc, out_desc, m, n, k, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_pid_m = tl.cdiv(m, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(n, _BLOCK_SIZE_1)
    num_pid_in_group = 4 * num_pid_n
    group_id = tl.program_id(0) // num_pid_in_group
    first_pid_m = group_id * 4
    group_size_m = min(num_pid_m - first_pid_m, 4)
    pid_0 = first_pid_m + tl.program_id(0) % num_pid_in_group % group_size_m
    pid_1 = tl.program_id(0) % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in range(0, k, _BLOCK_SIZE_2):
        load = x_desc.load([offset_0, offset_2])
        load_1 = y_desc.load([offset_2, offset_1])
        acc = tl.dot(load, load_1, acc=acc, input_precision='tf32')
    out_desc.store([offset_0, offset_1], acc)

def matmul(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {{k}} != {{k2}}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_kernel[triton.cdiv(m, _BLOCK_SIZE_0) * triton.cdiv(n, _BLOCK_SIZE_1),](TensorDescriptor.from_tensor(x, [_BLOCK_SIZE_0, _BLOCK_SIZE_2]), TensorDescriptor.from_tensor(y, [_BLOCK_SIZE_2, _BLOCK_SIZE_1]), TensorDescriptor.from_tensor(out, [_BLOCK_SIZE_0, _BLOCK_SIZE_1]), m, n, k, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out""",
        )

    def test_matmul_static_shapes0(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_static_shapes,
            args,
            block_sizes=[[16, 16], 16],
            l2_grouping=4,
            indexing="pointer",
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_static_shapes_kernel(x, y, out, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
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
        mm = tl.dot(load, load_1, input_precision='tf32')
        acc = acc_copy + mm
    tl.store(out + (indices_0[:, None] * 128 + indices_1[None, :] * 1), acc, None)

def matmul_static_shapes(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_static_shapes_kernel[triton.cdiv(128, _BLOCK_SIZE_0) * triton.cdiv(128, _BLOCK_SIZE_1),](x, y, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _matmul_static_shapes_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_matmul_static_shapes_kernel)(x, y, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    def test_matmul_static_shapes1(self):
        args = (
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_static_shapes,
            args,
            block_sizes=[[16, 16], 16],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_static_shapes_kernel(x, y, out, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
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
        mm = tl.dot(load, load_1, input_precision='tf32')
        acc = acc_copy + mm
    tl.store(out + (indices_0[:, None] * 128 + indices_1[None, :] * 1), acc, None)

def matmul_static_shapes(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_static_shapes_kernel[triton.cdiv(128, _BLOCK_SIZE_0) * triton.cdiv(128, _BLOCK_SIZE_1),](x, y, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out

def _matmul_static_shapes_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_matmul_static_shapes_kernel)(x, y, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)""",
        )

    @unittest.skip("need to debug correctness issue")
    def test_matmul_static_shapes2(self):
        args = (
            torch.randn([128, 127], device=DEVICE, dtype=torch.float32),
            torch.randn([127, 128], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_static_shapes,
            args,
            block_sizes=[[16, 16], 16],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_static_shapes_kernel(x, y, out, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
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
    for offset_2 in range(0, 127, _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        mask_2 = indices_2 < 127
        load = tl.load(x + (indices_0[:, None] * 127 + indices_2[None, :] * 1), mask_2[None, :], other=0)
        load_1 = tl.load(y + (indices_2[:, None] * 128 + indices_1[None, :] * 1), mask_2[:, None], other=0)
        mm = tl.dot(load, load_1, input_precision='tf32')
        acc = acc + mm
    tl.store(out + (indices_0[:, None] * 128 + indices_1[None, :] * 1), acc, None)

def matmul_static_shapes(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_static_shapes_kernel[triton.cdiv(128, _BLOCK_SIZE_0) * triton.cdiv(128, _BLOCK_SIZE_1),](x, y, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out""",
        )

    @unittest.skip("need to debug correctness issue")
    def test_matmul_static_shapes3(self):
        args = (
            torch.randn([127, 128], device=DEVICE, dtype=torch.float32),
            torch.randn([128, 127], device=DEVICE, dtype=torch.float32),
        )
        code, output = code_and_output(
            matmul_static_shapes,
            args,
            block_sizes=[[16, 16], 16],
            l2_grouping=4,
        )
        torch.testing.assert_close(output, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertExpectedInline(
            code,
            """\
import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_static_shapes_kernel(x, y, out, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr):
    num_pid_m = tl.cdiv(127, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(127, _BLOCK_SIZE_1)
    num_pid_in_group = 4 * num_pid_n
    group_id = tl.program_id(0) // num_pid_in_group
    first_pid_m = group_id * 4
    group_size_m = min(num_pid_m - first_pid_m, 4)
    pid_0 = first_pid_m + tl.program_id(0) % num_pid_in_group % group_size_m
    pid_1 = tl.program_id(0) % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < 127
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < 127
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in range(0, 128, _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        load = tl.load(x + (indices_0[:, None] * 128 + indices_2[None, :] * 1), mask_0[:, None], other=0)
        load_1 = tl.load(y + (indices_2[:, None] * 127 + indices_1[None, :] * 1), mask_1[None, :], other=0)
        mm = tl.dot(load, load_1, input_precision='tf32')
        acc = acc + mm
    tl.store(out + (indices_0[:, None] * 127 + indices_1[None, :] * 1), acc, mask_0[:, None] & mask_1[None, :])

def matmul_static_shapes(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _BLOCK_SIZE_2 = 16
    _matmul_static_shapes_kernel[triton.cdiv(127, _BLOCK_SIZE_0) * triton.cdiv(127, _BLOCK_SIZE_1),](x, y, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=4, num_stages=3)
    return out""",
        )
