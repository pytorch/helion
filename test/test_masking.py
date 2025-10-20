from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import helion
from helion import _compat
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfRefEager
import helion.language as hl


class TestMasking(RefEagerTestBase, TestCase):
    def test_mask_dot(self):
        @helion.kernel(config={"block_sizes": [[32, 32], 32]}, dot_precision="ieee")
        def add1mm(x, y):
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n])
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k] + 1, y[tile_k, tile_n] + 1)
                out[tile_m, tile_n] = acc
            return out

        args = (
            torch.randn([100, 100], device=DEVICE),
            torch.randn([100, 100], device=DEVICE),
        )
        code, result = code_and_output(
            add1mm,
            args,
        )
        self.assertExpectedJournal(code)
        torch.testing.assert_close(
            result, (args[0] + 1) @ (args[1] + 1), rtol=1e-2, atol=1e-1
        )

    def test_no_mask_views0(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m, 1], device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :][:, :, None].sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, args[0].sum(dim=1, keepdim=True))
        self.assertNotIn("tl.where", code)

    def test_no_mask_views1(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].transpose(0, 1).sum(dim=0)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, args[0].sum(dim=1))
        self.assertNotIn("tl.where", code)

    def test_no_mask_full0(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                v = torch.zeros_like(x[tile_m, :])
                out[tile_m] = v.sum(-1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, torch.zeros_like(args[0]).sum(dim=1))
        self.assertNotIn("tl.where", code)

    def test_no_mask_full1(self):
        @helion.kernel(config={"block_size": [32, 32]})
        def fn(x):
            m, n = x.size()
            hl.specialize(n)
            out = torch.empty([m], device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                v = hl.zeros([tile_m, tile_n])
                out[tile_m] = v.sum(-1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, torch.zeros_like(args[0]).sum(dim=1))
        self.assertNotIn("tl.where", code)

    def test_mask_offset(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = (x[tile_m, :] + 1).sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, (args[0] + 1).sum(dim=1))
        self.assertIn("tl.where", code)

    def test_no_mask_inductor_ops(self):
        @helion.kernel(config={"block_sizes": [32]})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                x0 = x[tile_m, :] + 1
                x1 = x0 - 1
                # +1-1 cancels out, so no masking needed
                out[tile_m] = x1.sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, args[0].sum(dim=1))
        self.assertNotIn("tl.where", code)

    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_loop_carry_masking(self):
        @helion.kernel(config={"block_sizes": [32, 32]})
        def fn(x):
            m, n = x.size()
            block_n = hl.register_block_size(n)
            out = torch.empty([m], device=x.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_n])
                for _ in hl.tile(n, block_size=block_n):
                    # The first iteration, this doesn't need a mask -- but the second does
                    acc += acc.sum(dim=1, keepdim=True)
                    acc += 1
                out[tile_m] = acc.sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        self.assertIn("tl.where", code)
        self.assertExpectedJournal(code)

    @patch.object(_compat, "_supports_tensor_descriptor", lambda: False)
    @skipIfRefEager(
        "Test is block size dependent which is not supported in ref eager mode"
    )
    def test_tile_index_does_not_mask(self):
        @helion.kernel(config={"block_sizes": [32, 32], "indexing": "block_ptr"})
        def fn(x):
            m, n = x.size()
            out = torch.empty([m], device=x.device)
            block_size_n = hl.register_block_size(n)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, block_size_n])
                for tile_n in hl.tile(0, n, block_size_n):
                    acc += x[tile_m.index, tile_n.index]
                out[tile_m.index] = acc.sum(dim=1)
            return out

        args = (torch.randn([100, 100], device=DEVICE),)
        code, result = code_and_output(
            fn,
            args,
        )
        torch.testing.assert_close(result, args[0].sum(dim=1))
        self.assertNotIn("tl.where", code)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
