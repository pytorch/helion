from __future__ import annotations

import unittest

from expecttest import TestCase
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


class TestMisc(TestCase):
    def test_decorator(self):
        def mydec(func):
            return func

        @mydec
        @helion.kernel
        def add1(x, y):
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        @helion.kernel
        @mydec
        def add2(x, y):
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        @mydec
        @helion.kernel(config=helion.Config(block_size=[4]))
        def add3(x, y):
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(4, device=DEVICE)

        code_and_output(add1, (x, x))

        with pytest.raises(
            expected_exception=helion.exc.DecoratorAfterHelionKernelDecorator,
            match="Decorators after helion kernel decorator are not allowed",
        ):
            code_and_output(add2, (x, x))

        code_and_output(add3, (x, x))

    def test_inputs(self):
        @helion.kernel
        def kernel(a_list, b_dict, b_tuple):
            a0, a1 = a_list
            b0 = b_dict["b0"]
            (b1,) = b_tuple
            c0, c1 = torch.empty_like(a0), torch.empty_like(a1)
            for tile in hl.tile(a0.size()):
                c0[tile] = a0[tile] + b0[tile]
                c1[tile] = a1[tile] + b1[tile]
            return [c0, c1]

        x = torch.randn(4, device=DEVICE)
        code, result = code_and_output(kernel, ([x, x], {"b0": x}, (x,)))
        torch.testing.assert_close(result[0], 2 * x)
        torch.testing.assert_close(result[1], 2 * x)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_kernel(a0, c0, c1, a0_size_0, a0_stride_0, c0_stride_0, c1_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
    mask_0 = indices_0 < a0_size_0
    load = tl.load(a0 + indices_0 * a0_stride_0, mask_0, other=0)
    load_1 = tl.load(a0 + indices_0 * a0_stride_0, mask_0, other=0)
    v_0 = load + load_1
    tl.store(c0 + indices_0 * c0_stride_0, v_0, mask_0)
    load_2 = tl.load(a0 + indices_0 * a0_stride_0, mask_0, other=0)
    load_3 = tl.load(a0 + indices_0 * a0_stride_0, mask_0, other=0)
    v_1 = load_2 + load_3
    tl.store(c1 + indices_0 * c1_stride_0, v_1, mask_0)

def kernel(a_list, b_dict, b_tuple):
    a0, a1 = a_list
    b0 = b_dict['b0']
    b1, = b_tuple
    c0, c1 = (torch.empty_like(a0), torch.empty_like(a1))
    _BLOCK_SIZE_0 = 4
    _kernel_kernel[triton.cdiv(a0.size(0), _BLOCK_SIZE_0),](a0, c0, c1, a0.size(0), a0.stride(0), c0.stride(0), c1.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return [c0, c1]

def _kernel_make_precompiler(a_list, b_dict, b_tuple):
    a0, a1 = a_list
    b0 = b_dict['b0']
    b1, = b_tuple
    c0, c1 = (torch.empty_like(a0), torch.empty_like(a1))
    _BLOCK_SIZE_0 = 4
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_kernel_kernel)(a0, c0, c1, a0.size(0), a0.stride(0), c0.stride(0), c1.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )


if __name__ == "__main__":
    unittest.main()
