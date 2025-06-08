from __future__ import annotations

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl


class TestErrors(TestCase):
    maxDiff = 16384

    def test_tile_unpacking(self):
        @helion.kernel()
        def sum_kernel(x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, hidden = x.size()
            out = x.new_empty(batch, hidden)
            for tile_batch, tile_hidden in hl.tile(batch, hidden):
                out[tile_batch, tile_hidden] = x[tile_batch, :, tile_hidden].sum(1)
            return out

        with self.assertRaises(helion.exc.FailedToUnpackTile):
            code_and_output(sum_kernel, (torch.randn(2, 3, 4, device=DEVICE),))

    def test_tile_overpacking(self):
        @helion.kernel()
        def fn(x: torch.Tensor) -> torch.Tensor:
            batch = x.size(0)
            out = x.new_empty(batch)
            for tile_wrapped_in_tuple in hl.tile([batch]):
                out[tile_wrapped_in_tuple] = x[tile_wrapped_in_tuple, :].sum(1)
            return out

        with self.assertRaises(helion.exc.OverpackedTile):
            code_and_output(fn, (torch.randn(100, 100, device=DEVICE),))
