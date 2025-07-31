from __future__ import annotations

import math
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion.exc import ShapeSpecializingAllocation
import helion.language as hl


class TestSpecialize(RefEagerTestBase, TestCase):
    maxDiff = 163842

    def test_sqrt_does_not_specialize(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            scale = 1.0 / math.sqrt(x.size(-1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * scale
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_sizes=[32, 1], flatten_loop=True)
        torch.testing.assert_close(result, x / math.sqrt(x.size(-1)))
        self.assertExpectedJournal(code)

    def test_specialize_host(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(-1))
            scale = 1.0 / math.sqrt(x.size(-1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = x[tile] * scale
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_sizes=[32, 32])
        torch.testing.assert_close(result, x / math.sqrt(x.size(-1)))
        self.assertExpectedJournal(code)

    def test_dynamic_size_block_errors(self):
        if self._in_ref_eager_mode:
            # In ref mode, shape specialization errors don't occur
            self.skipTest("Shape specialization behaves differently in ref eager mode")
            
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.zeros([tile, x.size(1)])
                acc += x[tile, :] + 1
                out[tile, :] = acc
            return out

        x = torch.randn([512, 512], device=DEVICE)
        with self.assertRaises(ShapeSpecializingAllocation):
            code_and_output(fn, (x,), block_size=16)

    def test_dynamic_size_block_specialize(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.zeros([tile, x.size(1)])
                acc += x[tile, :] + 1
                out[tile, :] = acc
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x + 1)
        self.assertEqual(len(fn.bind((x,)).config_spec.reduction_loops), 0)
        self.assertExpectedJournal(code)

    def test_dynamic_size_block_non_power_of_two(self):
        if self._in_ref_eager_mode:
            # This test uses next_power_of_2 which creates shape mismatches in ref mode
            self.skipTest("next_power_of_2 creates shape issues in ref eager mode")
            
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                acc = hl.zeros([tile, helion.next_power_of_2(x.size(1))])
                acc += x[tile, :] + 1
                out[tile, :] = acc
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x + 1)
        self.assertEqual(len(fn.bind((x,)).config_spec.reduction_loops), 0)
        self.assertIs(
            fn.bind((x,)),
            fn.bind((torch.zeros_like(x),)),
        )
        self.assertIsNot(
            fn.bind((x,)),
            fn.bind((torch.zeros_like(x[:, 1:]),)),
        )
        self.assertExpectedJournal(code)

    def test_specialize_reduce(self):
        @helion.kernel()
        def fn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            hl.specialize(x.size(1))
            out = x.new_empty([x.size(0)])
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile, :].sum(-1)
            return out

        x = torch.randn([500, 500], device=DEVICE)
        code, result = code_and_output(fn, (x,), block_size=32)
        torch.testing.assert_close(result, x.sum(-1))
        if not self._in_ref_eager_mode:
            # In ref mode, reduction_loops is not tracked the same way
            self.assertEqual(len(fn.bind((x,)).config_spec.reduction_loops), 1)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
