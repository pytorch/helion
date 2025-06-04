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
