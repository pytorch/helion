from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


class TestConstExpr(RefEagerTestBase, TestCase):
    def test_constexpr_float(self):
        @helion.kernel()
        def fn(x: torch.Tensor, v: hl.constexpr) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = torch.sigmoid(x[tile] + v)
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, 5.0),
        )
        torch.testing.assert_close(result, torch.sigmoid(x + 5.0))
        self.assertExpectedJournal(code)

    def test_constexpr_float_wrapped(self):
        @helion.kernel()
        def fn(x: torch.Tensor, v: float) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                out[tile] = torch.sigmoid(x[tile] + v)
            return out

        x = torch.randn([512, 512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, hl.constexpr(5.0)),
        )
        torch.testing.assert_close(result, torch.sigmoid(x + 5.0))
        self.assertExpectedJournal(code)

    def test_constexpr_size(self):
        @helion.kernel()
        def fn(x: torch.Tensor, s: hl.constexpr) -> torch.Tensor:
            (b,) = x.size()
            out = torch.empty([b, s], device=x.device, dtype=x.dtype)
            for tile_b, tile_s in hl.tile([b, s]):
                out[tile_b, tile_s] = x[tile_b].view(-1, 1).expand(tile_b, tile_s)
            return out

        x = torch.randn([512], device=DEVICE)
        code, result = code_and_output(
            fn,
            (x, 16),
        )
        torch.testing.assert_close(result, x.view(-1, 1).expand(512, 16))
        self.assertExpectedJournal(code)

    def test_string_literal_arg(self):
        @helion.kernel()
        def fn(x: torch.Tensor, mode: str) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size()):
                if mode == "add":
                    out[tile] = x[tile] + 1.0
                elif mode == "mul":
                    out[tile] = x[tile] * 2.0
                else:
                    out[tile] = x[tile]
            return out

        x = torch.randn([512, 512], device=DEVICE)

        # Test "add" mode
        code, result = code_and_output(fn, (x, "add"))
        torch.testing.assert_close(result, x + 1.0)
        self.assertExpectedJournal(code)

        # Test "mul" mode
        code, result = code_and_output(fn, (x, "mul"))
        torch.testing.assert_close(result, x * 2.0)
        self.assertExpectedJournal(code)

        # Test default mode
        code, result = code_and_output(fn, (x, "default"))
        torch.testing.assert_close(result, x)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
