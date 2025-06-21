from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
import unittest

from expecttest import TestCase
from packaging import version
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


class TestMisc(TestCase):
    def test_torch_alloc(self):
        @helion.kernel(config={"block_sizes": [64, 64]})
        def fn(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = x.new_empty([m])
            block_size_n = hl.register_block_size(n)
            for tile_m in hl.tile(m):
                acc = x.new_zeros([tile_m, block_size_n])
                for tile_n in hl.tile(n, block_size=block_size_n):
                    acc += x[tile_m, tile_n]
                out[tile_m] = acc.sum(dim=-1)
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(fn, (x,))
        torch.testing.assert_close(result, x.sum(-1), atol=1e-2, rtol=1e-2)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _fn_kernel(x, out, out_stride_0, x_stride_0, x_stride_1, m, n, _BLOCK_SIZE_1: tl.constexpr, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_1 = pid_0 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < m
    acc = tl.full([_BLOCK_SIZE_1, _BLOCK_SIZE_0], 0, tl.float32)
    for offset_0 in range(0, n.to(tl.int32), _BLOCK_SIZE_0):
        indices_0 = offset_0 + tl.arange(0, _BLOCK_SIZE_0).to(tl.int32)
        mask_0 = indices_0 < n
        acc_copy = acc
        acc_copy_0 = acc_copy
        load = tl.load(x + (indices_1[:, None] * x_stride_0 + indices_0[None, :] * x_stride_1), mask_1[:, None] & mask_0[None, :], other=0)
        acc = acc_copy_0 + load
    sum_1 = tl.sum(acc, 1)
    tl.store(out + indices_1 * out_stride_0, sum_1, mask_1)

def fn(x: torch.Tensor):
    m, n = x.size()
    out = x.new_empty([m])
    block_size_n = 64
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_0 = 64
    _fn_kernel[triton.cdiv(m, _BLOCK_SIZE_1),](x, out, out.stride(0), x.stride(0), x.stride(1), m, n, _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return out

def _fn_make_precompiler(x: torch.Tensor):
    m, n = x.size()
    out = x.new_empty([m])
    block_size_n = 64
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_0 = 64
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_fn_kernel)(x, out, out.stride(0), x.stride(0), x.stride(1), m, n, _BLOCK_SIZE_1, _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

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

    def test_patch_inductor_lowerings(self):
        if version.parse(torch.__version__.split("+")[0]) < version.parse("2.8"):
            from helion._compiler.inductor_lowering_extra import (
                register_inductor_lowering,
            )
        else:
            from torch._inductor.lowering import (
                register_lowering as register_inductor_lowering,
            )

        from helion._compiler.inductor_lowering_extra import inductor_lowering_dispatch
        from helion._compiler.inductor_lowering_extra import patch_inductor_lowerings

        inductor_lowerings_orig = torch._inductor.lowering.lowerings.copy()

        @torch.library.custom_op("helion_test::foo", mutates_args={})
        def foo(x: torch.Tensor) -> torch.Tensor:
            return x

        # Case 1: Register new lowering for the custom op
        @register_inductor_lowering(
            torch.ops.helion_test.foo, lowering_dict=inductor_lowering_dispatch
        )
        def foo_lowering(x):
            return x

        # Case 2: Register a patched lowering for add.Tensor
        @register_inductor_lowering(
            torch.ops.aten.add.Tensor, lowering_dict=inductor_lowering_dispatch
        )
        def add_lowering(*args, **kwargs):
            pass

        # Check that within `patch_inductor_lowerings()` context manager, the patched lowerings are used.
        with patch_inductor_lowerings():
            assert torch.ops.helion_test.foo in torch._inductor.lowering.lowerings
            assert torch.ops.aten.add.Tensor in torch._inductor.lowering.lowerings
            assert (
                torch._inductor.lowering.lowerings[torch.ops.aten.add.Tensor]
                != inductor_lowerings_orig[torch.ops.aten.add.Tensor]
            )

        # Check that outside the context manager, the original lowerings are restored.
        assert len(torch._inductor.lowering.lowerings.keys()) == len(
            inductor_lowerings_orig.keys()
        )
        for op in torch._inductor.lowering.lowerings:
            assert torch._inductor.lowering.lowerings[op] == inductor_lowerings_orig[op]

    def test_inputs(self):
        @helion.kernel
        def kernel(a_list, b_dict, b_tuple, c_named_tuple, d_dataclass):
            a0, a1 = a_list
            b0 = b_dict["b0"]
            (b1,) = b_tuple
            c0, c1 = c_named_tuple.x, c_named_tuple.y
            d0, d1 = d_dataclass.x, d_dataclass.y

            o0, o1 = torch.empty_like(a0), torch.empty_like(a1)
            for tile in hl.tile(a0.size()):
                o0[tile] = a0[tile] + b0[tile] + c0[tile] + d0[tile]
                o1[tile] = a1[tile] + b1[tile] + c1[tile] + d1[tile]
            return [o0, o1]

        x = torch.ones(4, device=DEVICE)
        Point = namedtuple("Point", ["x", "y"])  # noqa: PYI024
        p = Point(x, x)

        @dataclass(frozen=True)
        class Point2:
            x: torch.Tensor
            y: torch.Tensor

        p2 = Point2(x, x)

        code, result = code_and_output(kernel, ([x, x], {"b0": x}, (x,), p, p2))
        torch.testing.assert_close(result[0], 4 * x)
        torch.testing.assert_close(result[1], 4 * x)
        self.assertExpectedInline(
            code,
            """\
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_kernel(a0, o0, o1, a0_size_0, a0_stride_0, o0_stride_0, o1_stride_0, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < a0_size_0
    load = tl.load(a0 + indices_0 * a0_stride_0, mask_0, other=0)
    load_1 = tl.load(a0 + indices_0 * a0_stride_0, mask_0, other=0)
    v_0 = load + load_1
    load_2 = tl.load(a0 + indices_0 * a0_stride_0, mask_0, other=0)
    v_1 = v_0 + load_2
    load_3 = tl.load(a0 + indices_0 * a0_stride_0, mask_0, other=0)
    v_2 = v_1 + load_3
    tl.store(o0 + indices_0 * o0_stride_0, v_2, mask_0)
    load_4 = tl.load(a0 + indices_0 * a0_stride_0, mask_0, other=0)
    load_5 = tl.load(a0 + indices_0 * a0_stride_0, mask_0, other=0)
    v_3 = load_4 + load_5
    load_6 = tl.load(a0 + indices_0 * a0_stride_0, mask_0, other=0)
    v_4 = v_3 + load_6
    load_7 = tl.load(a0 + indices_0 * a0_stride_0, mask_0, other=0)
    v_5 = v_4 + load_7
    tl.store(o1 + indices_0 * o1_stride_0, v_5, mask_0)

def kernel(a_list, b_dict, b_tuple, c_named_tuple, d_dataclass):
    a0, a1 = a_list
    b0 = b_dict['b0']
    b1, = b_tuple
    c0, c1 = (c_named_tuple.x, c_named_tuple.y)
    d0, d1 = (d_dataclass.x, d_dataclass.y)
    o0, o1 = (torch.empty_like(a0), torch.empty_like(a1))
    _BLOCK_SIZE_0 = 4
    _kernel_kernel[triton.cdiv(a0.size(0), _BLOCK_SIZE_0),](a0, o0, o1, a0.size(0), a0.stride(0), o0.stride(0), o1.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)
    return [o0, o1]

def _kernel_make_precompiler(a_list, b_dict, b_tuple, c_named_tuple, d_dataclass):
    a0, a1 = a_list
    b0 = b_dict['b0']
    b1, = b_tuple
    c0, c1 = (c_named_tuple.x, c_named_tuple.y)
    d0, d1 = (d_dataclass.x, d_dataclass.y)
    o0, o1 = (torch.empty_like(a0), torch.empty_like(a1))
    _BLOCK_SIZE_0 = 4
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_kernel_kernel)(a0, o0, o1, a0.size(0), a0.stride(0), o0.stride(0), o1.stride(0), _BLOCK_SIZE_0, num_warps=4, num_stages=3)""",
        )

    def test_config_flatten_issue(self):
        @helion.kernel(use_default_config=True)
        def test_tile_id_atomic_add(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x, dtype=torch.int32)
            for tile_m, tile_n in hl.tile(x.size()):
                out[tile_m.begin, tile_n.begin] = 1
            return out

        x = torch.randn(64, 64, device="cuda")
        config = helion.Config(block_sizes=[16, 16])
        test_tile_id_atomic_add.bind((x,)).to_triton_code(config)
        result = test_tile_id_atomic_add.bind((x,)).compile_config(config)(x)
        self.assertEqual(result.sum().item(), 16)


if __name__ == "__main__":
    unittest.main()
