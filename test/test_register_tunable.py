from __future__ import annotations

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

        x = torch.randn(128, device="cuda", dtype=torch.float32)
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

        x = torch.randn(128, device="cuda", dtype=torch.float32)
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

        x = torch.randn(128, device="cuda", dtype=torch.float32)
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
    tl.store(partial + tl.full([1], floordiv, tl.int32) * partial_stride_0, sum_1, None)

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
