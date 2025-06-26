from __future__ import annotations

import unittest

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion.autotuner import EnumFragment
from helion.autotuner import IntegerFragment
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl


class TestRegisterTunable(TestCase):
    maxDiff = 10000

    def test_power_of_two_fragment_basic(self):
        @helion.kernel(use_default_config=True)
        def kernel_with_tunable(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)

            # Register a tunable parameter for block size
            block_size = hl.register_tunable("foo", PowerOfTwoFragment(16, 256))

            for tile_n in hl.tile([n], block_size=[block_size * 2]):
                out[tile_n] = x[tile_n] * 2.0

            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(kernel_with_tunable, (x,))
        expected = x * 2.0
        torch.testing.assert_close(result, expected)
        self.assertIsInstance(
            kernel_with_tunable.bind((x,)).config_spec.user_defined_tunables["foo"],
            PowerOfTwoFragment,
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

import test.test_register_tunable as _source_module

@triton.jit
def _kernel_with_tunable_kernel(x, out, out_stride_0, x_stride_0, n, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < n
    load = tl.load(x + indices_0 * x_stride_0, mask_0, other=0)
    v_0 = 2.0
    v_1 = load * v_0
    tl.store(out + indices_0 * out_stride_0, v_1, mask_0)

def kernel_with_tunable(x: torch.Tensor):
    n, = x.size()
    out = torch.empty_like(x)
    block_size = 16
    _BLOCK_SIZE_0 = 2 * block_size
    _kernel_with_tunable_kernel[triton.cdiv(n, _BLOCK_SIZE_0),](x, out, out.stride(0), x.stride(0), n, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _kernel_with_tunable_make_precompiler(x: torch.Tensor):
    n, = x.size()
    out = torch.empty_like(x)
    block_size = 16
    _BLOCK_SIZE_0 = 2 * block_size
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_kernel_with_tunable_kernel)(x, out, out.stride(0), x.stride(0), n, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_integer_fragment(self):
        @helion.kernel()
        def kernel_with_int_param(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)
            # Register an integer tunable parameter
            multiplier = hl.register_tunable("multiplier", IntegerFragment(1, 10, 3))
            for tile_n in hl.tile([n]):
                out[tile_n] = x[tile_n] * multiplier
            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            kernel_with_int_param, (x,), block_size=64, multiplier=4
        )
        expected = x * 4
        torch.testing.assert_close(result, expected)
        self.assertExpectedInline(
            repr(kernel_with_int_param.bind((x,)).config_spec.default_config()),
            """helion.Config(block_sizes=[128], num_warps=4, num_stages=3, indexing='pointer', multiplier=3)""",
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

import test.test_register_tunable as _source_module

@triton.jit
def _kernel_with_int_param_kernel(x, out, out_stride_0, x_stride_0, n, multiplier, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < n
    load = tl.load(x + indices_0 * x_stride_0, mask_0, other=0)
    v_0 = multiplier.to(tl.float32)
    v_1 = load * v_0
    tl.store(out + indices_0 * out_stride_0, v_1, mask_0)

def kernel_with_int_param(x: torch.Tensor):
    n, = x.size()
    out = torch.empty_like(x)
    multiplier = 4
    _BLOCK_SIZE_0 = 64
    _kernel_with_int_param_kernel[triton.cdiv(n, _BLOCK_SIZE_0),](x, out, out.stride(0), x.stride(0), n, multiplier, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _kernel_with_int_param_make_precompiler(x: torch.Tensor):
    n, = x.size()
    out = torch.empty_like(x)
    multiplier = 4
    _BLOCK_SIZE_0 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_kernel_with_int_param_kernel)(x, out, out.stride(0), x.stride(0), n, multiplier, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_enum_fragment(self):
        @helion.kernel(config={"operation": 2})
        def kernel_with_enum(x: torch.Tensor) -> torch.Tensor:
            (n,) = x.size()
            out = torch.empty_like(x)

            # Register an enum tunable parameter
            operation = hl.register_tunable("operation", EnumFragment((1, 2, 4)))

            for tile_n in hl.tile([n], block_size=[64]):
                out[tile_n] = x[tile_n] * operation

            return out

        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        result = kernel_with_enum(x)
        expected = x * 2.0
        torch.testing.assert_close(result, expected)

    def test_tensor_allocated_with_block_size(self):
        @helion.kernel()
        def fn(x: torch.Tensor):
            m = x.size(0)
            block_m = hl.register_block_size(m)
            tiles_m = (m + block_m - 1) // block_m  # cdiv
            partial = torch.zeros(tiles_m, dtype=x.dtype, device=x.device)
            for tile in hl.tile(m, block_size=block_m):
                partial[tile.begin // block_m] = x[tile].sum()
            return partial.sum()

        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(fn, (x,), block_size=64)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers

@triton.jit
def _fn_kernel(x, partial, partial_stride_0, x_stride_0, m, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < m
    load = tl.load(x + indices_0 * x_stride_0, mask_0, other=0)
    sum_1 = tl.sum(load, 0)
    floordiv = triton_helpers.div_floor_integer(offset_0, _BLOCK_SIZE_0)
    tl.store(partial + floordiv * partial_stride_0, sum_1, None)

def fn(x: torch.Tensor):
    m = x.size(0)
    block_m = 64
    tiles_m = (m + block_m - 1) // block_m
    partial = torch.zeros(tiles_m, dtype=x.dtype, device=x.device)
    _BLOCK_SIZE_0 = 64
    _fn_kernel[triton.cdiv(m, _BLOCK_SIZE_0),](x, partial, partial.stride(0), x.stride(0), m, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return partial.sum()

def _fn_make_precompiler(x: torch.Tensor):
    m = x.size(0)
    block_m = 64
    tiles_m = (m + block_m - 1) // block_m
    partial = torch.zeros(tiles_m, dtype=x.dtype, device=x.device)
    _BLOCK_SIZE_0 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, partial, partial.stride(0), x.stride(0), m, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )
        torch.testing.assert_close(result, x.sum())

    def test_matmul_split_k(self):
        """Test matmul_split_k kernel with register_tunable"""

        @helion.kernel(
            config=helion.Config(
                block_sizes=[32, 64, 64],
                loop_orders=[[1, 2, 0]],
                num_warps=16,
                num_stages=8,
                indexing="block_ptr",
                split_k=64,
            )
        )
        def matmul_split_k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.zeros(
                [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
            )
            split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
            k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
            for tile_m, tile_n, outer_k in hl.tile(
                [m, n, k], block_size=[None, None, k_block]
            ):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for inner_k in hl.tile(outer_k.begin, outer_k.end):
                    acc = torch.addmm(acc, x[tile_m, inner_k], y[inner_k, tile_n])
                hl.atomic_add(out, [tile_m, tile_n], acc)
            return out

        m, k, n = 64, 4096, 64
        x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
        y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)

        code, result = code_and_output(matmul_split_k, (x, y))
        expected = x @ y
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1)
        self.assertIsInstance(
            matmul_split_k.bind((x, y)).config_spec.user_defined_tunables["split_k"],
            PowerOfTwoFragment,
        )
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import helion
import triton
import triton.language as tl

import test.test_register_tunable as _source_module

@triton.jit
def _matmul_split_k_kernel(x, y, out, out_stride_0, out_stride_1, x_stride_0, x_stride_1, y_stride_0, y_stride_1, n, k, m, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_2: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_3: tl.constexpr):
    num_blocks_0 = tl.cdiv(n, _BLOCK_SIZE_1)
    num_blocks_1 = tl.cdiv(k, _BLOCK_SIZE_2)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0 % num_blocks_1
    pid_2 = tl.program_id(0) // (num_blocks_0 * num_blocks_1)
    offset_1 = pid_0 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < n
    offset_2 = pid_1 * _BLOCK_SIZE_2
    offset_0 = pid_2 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < m
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    tile_end = tl.minimum(offset_2 + _BLOCK_SIZE_2, k)
    for offset_3 in range(offset_2.to(tl.int32), tile_end.to(tl.int32), _BLOCK_SIZE_3):
        indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
        mask_3 = indices_3 < tile_end
        acc_copy = acc
        acc_copy_0 = acc_copy
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_3[None, :] * x_stride_1), mask_0[:, None] & mask_3[None, :], other=0)
        load_1 = tl.load(y + (indices_3[:, None] * y_stride_0 + indices_1[None, :] * y_stride_1), mask_3[:, None] & mask_1[None, :], other=0)
        acc = tl.dot(load, load_1, acc=acc_copy_0, input_precision='tf32')
    tl.atomic_add(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), acc, mask=mask_0[:, None] & mask_1[None, :], sem='relaxed')

def matmul_split_k(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.zeros([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    split_k = 64
    k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = k_block
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_3 = 64
    _matmul_split_k_kernel[triton.cdiv(n, _BLOCK_SIZE_1) * triton.cdiv(k, _BLOCK_SIZE_2) * triton.cdiv(m, _BLOCK_SIZE_0),](x, y, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), n, k, m, _BLOCK_SIZE_1, _BLOCK_SIZE_2, _BLOCK_SIZE_0, _BLOCK_SIZE_3, num_warps=16, num_stages=8)
    return out

def _matmul_split_k_make_precompiler(x: torch.Tensor, y: torch.Tensor):
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.zeros([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    split_k = 64
    k_block = helion.next_power_of_2(helion.cdiv(k, split_k))
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = k_block
    _BLOCK_SIZE_0 = 32
    _BLOCK_SIZE_3 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_matmul_split_k_kernel)(x, y, out, out.stride(0), out.stride(1), x.stride(0), x.stride(1), y.stride(0), y.stride(1), n, k, m, _BLOCK_SIZE_1, _BLOCK_SIZE_2, _BLOCK_SIZE_0, _BLOCK_SIZE_3, num_warps=16, num_stages=8)""",
        )


if __name__ == "__main__":
    unittest.main()
