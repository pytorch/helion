from __future__ import annotations

from pathlib import Path
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path
from helion._testing import skipIfRefEager
import helion.language as hl

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")


@helion.kernel
def cast_after_div(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(ref)
    for tile in helion.language.tile(out.size()):
        a = x[tile].to(torch.float32)
        p = a / (1 + torch.exp(-a))
        out[tile] = p.to(ref.dtype)
    return out


class TestGenerateAst(RefEagerTestBase, TestCase):
    def test_add1d(self):
        args = (torch.randn([4096], device=DEVICE), torch.randn([4096], device=DEVICE))
        code, result = code_and_output(basic_kernels.add, args, block_size=1024)
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedJournal(code)

    def test_add2d(self):
        args = (
            torch.randn([100, 500], device=DEVICE),
            torch.randn([100, 500], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_sizes=[1024, 1], flatten_loop=True
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedJournal(code)

    def test_add2d_loop_order(self):
        args = (
            torch.randn([100, 500], device=DEVICE),
            torch.randn([100, 500], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add,
            args,
            block_sizes=[1024, 1],
            flatten_loops=[True],
            loop_order=(1, 0),
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedJournal(code)

    def test_add3d(self):
        args = (
            torch.randn([100, 500, 10], device=DEVICE),
            torch.randn([100, 500, 10], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_sizes=[1024, 1, 1], flatten_loop=True
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedJournal(code)

    def test_add3d_xy_grid(self):
        args = (
            torch.randn([100, 500, 10], device=DEVICE),
            torch.randn([100, 500, 10], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_sizes=[16, 16, 16], pid_type="xyz"
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedJournal(code)

    def test_add3d_reorder(self):
        args = (
            torch.randn([100, 500, 10], device=DEVICE),
            torch.randn([100, 500, 10], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add,
            args,
            block_sizes=[1024, 1, 1],
            flatten_loop=True,
            loop_order=(2, 0, 1),
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedJournal(code)

    def test_add_tilend0(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_sizes=[8, 16, 32], loop_order=(0, 1, 2)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedJournal(code)

    def test_add_tilend1(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_sizes=[8, 16, 32], loop_order=(2, 1, 0)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedJournal(code)

    def test_add_tilend2(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add, args, block_sizes=[1, 32, 32], loop_order=(0, 1, 2)
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedJournal(code)

    def test_add_tilend3(self):
        args = (
            torch.randn([512, 512, 512], device=DEVICE),
            torch.randn([512, 512, 512], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.add,
            args,
            block_sizes=[1, 32, 1],
            loop_order=(0, 2, 1),
            num_warps=8,
            num_stages=1,
        )
        torch.testing.assert_close(result, args[0] + args[1])
        self.assertExpectedJournal(code)

    def test_torch_ops_pointwise(self):
        args = (
            torch.randn([1024], device=DEVICE),
            torch.randn([1024], device=DEVICE),
        )
        code, result = code_and_output(
            basic_kernels.torch_ops_pointwise,
            args,
            block_size=128,
        )
        torch.testing.assert_close(
            result, torch.sigmoid(torch.add(torch.sin(args[0]), torch.cos(args[1])))
        )
        self.assertExpectedJournal(code)

    def test_hl_zeros_usage(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.hl_zeros_usage,
            args,
            block_sizes=[32, 32],
        )
        torch.testing.assert_close(result, args[0] * 2)
        self.assertExpectedJournal(code)

    def test_hl_full_usage(self):
        args = (torch.randn([512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.hl_full_usage,
            args,
            block_size=128,
        )
        torch.testing.assert_close(result, args[0] * 2 + 1)
        self.assertExpectedJournal(code)

    def test_hl_zeros_flat(self):
        args = (torch.randn([512, 512], device=DEVICE),)
        code, result = code_and_output(
            basic_kernels.hl_zeros_usage,
            args,
            block_sizes=[128, 1],
            flatten_loops=[True],
        )
        torch.testing.assert_close(result, args[0] * 2)
        self.assertExpectedJournal(code)

    def test_inplace_mul(self):
        args = (torch.randn([512, 512], device=DEVICE), 4)
        eager_result = args[0] * args[1]
        code, result = code_and_output(
            basic_kernels.inplace_mul,
            args,
            block_size=[128, 1],
            flatten_loop=True,
        )
        torch.testing.assert_close(result, eager_result)
        self.assertExpectedJournal(code)

    @skipIfRefEager("Codegen inspection not applicable in ref eager mode")
    def test_final_cast_enforced_for_to_dtype(self):
        x = torch.randn([1024], device=DEVICE, dtype=torch.bfloat16)
        ref = torch.empty_like(x)
        code, result = code_and_output(cast_after_div, (x, ref), block_size=256)
        # Ensure codegen emits a final tl.cast(..., tl.bfloat16)
        assert "tl.cast" in code and "tl.bfloat16" in code

    def test_sigmoid_scalar_autocast(self):
        @helion.kernel(
            config=helion.Config(
                block_sizes=[32],
                indexing="block_ptr",
            ),
            static_shapes=True,
        )
        def se_block_fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m, n], dtype=x.dtype, device=x.device)

            for tile_m in hl.tile(m):
                x_tile = x[tile_m, :]
                sigmoid_result = torch.sigmoid(x_tile @ w[:, :])
                acc = 2.0 * x_tile * sigmoid_result
                out[tile_m, :] = acc.to(x.dtype)

            return out

        m, n = 4096, 128
        dtype = torch.bfloat16

        x = torch.randn(m, n, device=DEVICE, dtype=dtype)
        w = torch.randn(n, n, device=DEVICE, dtype=dtype)

        code, result = code_and_output(se_block_fwd, (x, w))

        x_fp32 = x.to(torch.float32)
        w_fp32 = w.to(torch.float32)
        expected = (2.0 * x_fp32 * torch.sigmoid(x_fp32 @ w_fp32)).to(dtype)

        torch.testing.assert_close(result, expected, atol=1e-1, rtol=1e-1)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
