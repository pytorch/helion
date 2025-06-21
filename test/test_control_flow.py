from __future__ import annotations

import unittest

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


class TestControlFlow(TestCase):
    maxDiff = 16384

    def test_if_arg(self):
        @helion.kernel()
        def fn(x, v):
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code0, result = code_and_output(
            fn,
            (x, 5),
        )
        torch.testing.assert_close(result, torch.sigmoid(x))
        code1, result = code_and_output(
            fn,
            (x, 10),
        )
        torch.testing.assert_close(result, torch.sin(x))
        self.assertEqual(code0, code1)
        self.assertExpectedInline(
            code0,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _fn_kernel(x, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, v, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < x_size_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < x_size_1
    gt = v > 3
    lt = v < 7
    _and = gt and lt
    if _and:
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
        v_0 = tl.sigmoid(load)
        tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_0, mask_0[:, None] & mask_1[None, :])
    _not = not _and
    if _not:
        load_1 = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
        v_1 = tl_math.sin(load_1)
        tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_1, mask_0[:, None] & mask_1[None, :])

def fn(x, v):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0) * triton.cdiv(x.size(1), _BLOCK_SIZE_1),](x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), v, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x, v):
    out = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), v, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_if_arg_one_element_tensor(self):
        @helion.kernel
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.zeros_like(x)

            for idx in hl.grid(x.shape[0]):
                # Since `y[idx]` is a one-element tensor, comparing it against 0 will also create a one-element tensor.
                if y[idx] != 0:
                    output[idx] = x[idx] * 2
                if (
                    y[idx] == 0
                ):  # TODO(yf225): `else:` raises MLIR error in Triton, so we use a second if.
                    output[idx] = x[idx]

            return output

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        y = torch.tensor([0, 1, 0, 1], device=DEVICE, dtype=torch.int32)
        expected = torch.tensor([1.0, 4.0, 3.0, 8.0], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, y),
        )
        torch.testing.assert_close(result, expected)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, y, output, output_stride_0, x_stride_0, y_stride_0):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    load = tl.load(y + tl.full([1], offset_0, tl.int32) * y_stride_0, None)
    v_0 = tl.full([], 0, tl.int32)
    v_1 = load != v_0
    if tl.sum(v_1):
        load_1 = tl.load(x + tl.full([1], offset_0, tl.int32) * x_stride_0, None)
        v_2 = 2.0
        v_3 = load_1 * v_2
        tl.store(output + tl.full([1], offset_0, tl.int32) * output_stride_0, v_3, None)
    load_2 = tl.load(y + tl.full([1], offset_0, tl.int32) * y_stride_0, None)
    v_4 = tl.full([], 0, tl.int32)
    v_5 = load_2 == v_4
    if tl.sum(v_5):
        load_3 = tl.load(x + tl.full([1], offset_0, tl.int32) * x_stride_0, None)
        tl.store(output + tl.full([1], offset_0, tl.int32) * output_stride_0, load_3, None)

def fn(x: torch.Tensor, y: torch.Tensor):
    output = torch.zeros_like(x)
    _fn_kernel[x.size(0),](x, y, output, output.stride(0), x.stride(0), y.stride(0), num_warps=4, num_stages=3)
    return output

def _fn_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    output = torch.zeros_like(x)
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, y, output, output.stride(0), x.stride(0), y.stride(0), num_warps=4, num_stages=3)""",
        )

    def test_constant_true(self):
        @helion.kernel(
            config={
                "block_sizes": [128, 1],
                "flatten_loop": True,
                "indexing": "block_ptr",
            }
        )
        def fn(x):
            out = torch.empty_like(x)
            v = 4
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x,),
        )
        torch.testing.assert_close(result, torch.sigmoid(x))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0_1: tl.constexpr):
    offsets_0_1 = tl.program_id(0) * _BLOCK_SIZE_0_1 + tl.arange(0, _BLOCK_SIZE_0_1).to(tl.int32)
    indices_1 = offsets_0_1 % x_size_1
    indices_0 = offsets_0_1 // x_size_1
    mask_0_1 = offsets_0_1 < x_size_0 * x_size_1
    load = tl.load(x + (indices_0 * x_stride_0 + indices_1 * x_stride_1), mask_0_1, other=0)
    v_0 = tl.sigmoid(load)
    tl.store(out + (indices_0 * out_stride_0 + indices_1 * out_stride_1), v_0, mask_0_1)

def fn(x):
    out = torch.empty_like(x)
    v = 4
    _BLOCK_SIZE_0_1 = 128
    _fn_kernel[triton.cdiv(x.size(0) * x.size(1), _BLOCK_SIZE_0_1), 1, 1](x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x):
    out = torch.empty_like(x)
    v = 4
    _BLOCK_SIZE_0_1 = 128
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0_1, num_warps=4, num_stages=3)""",
        )

    def test_constant_false(self):
        @helion.kernel(config={"block_size": [32, 32], "indexing": "block_ptr"})
        def fn(x):
            out = torch.empty_like(x)
            v = 15
            for tile in hl.tile(x.size()):
                if 3 < v < 7:
                    out[tile] = torch.sigmoid(x[tile])
                else:
                    out[tile] = torch.sin(x[tile])
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x,),
        )
        torch.testing.assert_close(result, torch.sin(x))
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math

@triton.jit
def _fn_kernel(x, out, out_size_0, out_size_1, x_size_0, x_size_1, out_stride_0, out_stride_1, x_stride_0, x_stride_1, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(x_size_0, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    load = tl.load(tl.make_block_ptr(x, [x_size_0, x_size_1], [x_stride_0, x_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), boundary_check=[0, 1], padding_option='zero')
    v_0 = tl_math.sin(load)
    tl.store(tl.make_block_ptr(out, [out_size_0, out_size_1], [out_stride_0, out_stride_1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]), v_0, boundary_check=[0, 1])

def fn(x):
    out = torch.empty_like(x)
    v = 15
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _fn_kernel[triton.cdiv(x.size(0), _BLOCK_SIZE_0) * triton.cdiv(x.size(1), _BLOCK_SIZE_1),](x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x):
    out = torch.empty_like(x)
    v = 15
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, out.size(0), out.size(1), x.size(0), x.size(1), out.stride(0), out.stride(1), x.stride(0), x.stride(1), _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )

    def test_error_in_non_taken_branch(self):
        def mul_relu_block_back_spec(x, y, dz):
            z = torch.relu(x * y[:, None])
            grad_x, grad_y = torch.autograd.grad(z, [x, y], dz, retain_graph=True)
            return grad_x, grad_y

        @helion.kernel(config=helion.Config(block_sizes=[32, 32]))
        def mul_relu_block_backward_kernel(
            x: torch.Tensor,
            y: torch.Tensor,
            dz: torch.Tensor,
            use_atomics: hl.constexpr = False,
        ):
            # Get tensor sizes
            m, n = x.shape
            # Create output tensor for gradients
            dx = torch.empty_like(x)

            if use_atomics:
                dy = torch.zeros_like(y)
            else:
                dy = torch.empty_like(x)

            # Use Helion to tile the computation
            for tile_i, tile_j in hl.tile([m, n]):
                # Get input tiles
                x_tile = x[tile_i, tile_j]
                y_tile = y[tile_i]
                dz_tile = dz[tile_i, tile_j]

                # For ReLU, gradient is 1 where input > 0, 0 otherwise
                relu_mask = (x_tile * y_tile[:, None]) > 0
                # Chain rule: dx = dz * relu_grad * y
                relu_grad = torch.where(relu_mask, 1, 0)
                dx[tile_i, tile_j] = dz_tile * relu_grad * y_tile[:, None]

                # Chain rule: dy = dz * relu_grad * x -> backwards of broadcast(sum)
                if use_atomics:
                    local_dy_grad = torch.sum(dz_tile * relu_grad * x_tile, dim=1)
                    hl.atomic_add(dy, [tile_i], local_dy_grad)
                else:
                    local_dy_grad = dz_tile * relu_grad * x_tile
                    dy[tile_i, tile_j] = local_dy_grad

            if use_atomics:
                return dx, dy
            return dx, dy.sum(axis=-1)

        x = torch.randn(512, 1024, device=DEVICE, requires_grad=True)
        y = torch.randn(512, device=DEVICE, requires_grad=True)
        dz = torch.randn(512, 1024, device=DEVICE)
        expected = mul_relu_block_back_spec(x, y, dz)
        torch.testing.assert_close(
            mul_relu_block_backward_kernel(x, y, dz, False),
            expected,
            atol=1e-4,
            rtol=1e-4,
        )
        code, output = code_and_output(
            mul_relu_block_backward_kernel,
            (x, y, dz, True),
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import helion.language as hl
import triton
import triton.language as tl

@triton.jit
def _mul_relu_block_backward_kernel_kernel(x, y, dz, dx, dy, dx_stride_0, dx_stride_1, dy_stride_0, dz_stride_0, dz_stride_1, x_stride_0, x_stride_1, y_stride_0, m, n, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(m, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < m
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < n
    x_tile = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    y_tile = tl.load(y + indices_0 * y_stride_0, mask_0, other=0)
    dz_tile = tl.load(dz + (indices_0[:, None] * dz_stride_0 + indices_1[None, :] * dz_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    subscript = y_tile[:, None]
    v_0 = x_tile * subscript
    v_1 = 0.0
    v_2 = v_0 > v_1
    v_3 = tl.full([], 0, tl.int64)
    v_4 = tl.full([], 1, tl.int64)
    v_5 = v_4[None, None]
    v_6 = v_3[None, None]
    v_7 = tl.where(v_2, v_5, v_6)
    v_8 = v_7.to(tl.float32)
    v_9 = dz_tile * v_8
    subscript_1 = y_tile[:, None]
    v_10 = v_9 * subscript_1
    tl.store(dx + (indices_0[:, None] * dx_stride_0 + indices_1[None, :] * dx_stride_1), v_10, mask_0[:, None] & mask_1[None, :])
    v_11 = v_7.to(tl.float32)
    v_12 = dz_tile * v_11
    v_13 = v_12 * x_tile
    local_dy_grad = tl.sum(v_13, 1)
    tl.atomic_add(dy + indices_0 * dy_stride_0, local_dy_grad, mask=mask_0, sem='relaxed')

def mul_relu_block_backward_kernel(x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor, use_atomics: hl.constexpr=False):
    m, n = x.shape
    dx = torch.empty_like(x)
    if True:
        dy = torch.zeros_like(y)
    else:
        dy = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    _mul_relu_block_backward_kernel_kernel[triton.cdiv(m, _BLOCK_SIZE_0) * triton.cdiv(n, _BLOCK_SIZE_1),](x, y, dz, dx, dy, dx.stride(0), dx.stride(1), dy.stride(0), dz.stride(0), dz.stride(1), x.stride(0), x.stride(1), y.stride(0), m, n, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    if True:
        return (dx, dy)
    return (dx, dy.sum(axis=-1))

def _mul_relu_block_backward_kernel_make_precompiler(x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor, use_atomics: hl.constexpr=False):
    m, n = x.shape
    dx = torch.empty_like(x)
    if True:
        dy = torch.zeros_like(y)
    else:
        dy = torch.empty_like(x)
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_1 = 32
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_mul_relu_block_backward_kernel_kernel)(x, y, dz, dx, dy, dx.stride(0), dx.stride(1), dy.stride(0), dz.stride(0), dz.stride(1), x.stride(0), x.stride(1), y.stride(0), m, n, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
        )
        torch.testing.assert_close(
            output,
            expected,
            atol=1e-4,
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
