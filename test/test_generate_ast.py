from __future__ import annotations

from pathlib import Path
import unittest

import torch

from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import import_path

datadir = Path(__file__).parent / "data"
basic_kernels = import_path(datadir / "basic_kernels.py")


class TestGenerateAst(TestCase):
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


if __name__ == "__main__":
    unittest.main()
