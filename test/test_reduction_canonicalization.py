from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@onlyBackends(["triton"])
class TestReductionCanonicalization(TestCase):
    def test_reassociate_adjacent_trailing_sums(self) -> None:
        @helion.kernel(reassociate_reductions=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            rows, groups, group_size = x.shape
            out = torch.empty(rows, device=x.device, dtype=torch.float32)
            for tile_r, tile_g, tile_n in hl.tile([rows, groups, group_size]):
                values = x[tile_r, tile_g, tile_n].to(torch.float32)
                total = torch.sum(torch.sum(values * values, dim=-1), dim=-1)
                out[tile_r] = total
            return out

        x = torch.randn(2, 16, 8, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(fn, (x,), block_sizes=[1, 16, 8])

        torch.testing.assert_close(
            result,
            torch.sum(torch.sum(x.float() * x.float(), dim=-1), dim=-1),
        )
        self.assertIn("tl.reshape", code)
        self.assertEqual(code.count("tl.sum("), 1)

    def test_reassociation_is_opt_in(self) -> None:
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            rows, groups, group_size = x.shape
            out = torch.empty(rows, device=x.device, dtype=torch.float32)
            for tile_r, tile_g, tile_n in hl.tile([rows, groups, group_size]):
                values = x[tile_r, tile_g, tile_n].to(torch.float32)
                total = torch.sum(torch.sum(values * values, dim=-1), dim=-1)
                out[tile_r] = total
            return out

        x = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(fn, (x,), block_sizes=[1, 4, 8])

        torch.testing.assert_close(
            result,
            torch.sum(torch.sum(x.float() * x.float(), dim=-1), dim=-1),
        )
        self.assertNotIn("tl.reshape", code)
        self.assertEqual(code.count("tl.sum("), 2)
