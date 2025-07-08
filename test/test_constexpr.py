from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


class TestConstExpr(TestCase):
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


if __name__ == "__main__":
    unittest.main()
