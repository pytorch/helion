from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel(static_shapes=True, autotune_effort="none")
def prefetch_rows(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r in hl.tile(x.size(0), block_size=1):
        hl.prefetch(x, [tile_r.begin])
        out[tile_r, :] = x[tile_r, :] * 2
    return out


@onlyBackends(["triton"])
class TestPrefetch(RefEagerTestBase, TestCase):
    @skipIfNotCUDA()
    def test_row_prefetch_keeps_values(self) -> None:
        x = torch.randn((8, 512), device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(prefetch_rows, (x,))

        torch.testing.assert_close(out, x * 2)
        if not self._in_ref_eager_mode:
            self.assertIn("helion_cache_hints.prefetch_l2(x + ", code)
            self.assertIn(", 0, 2048)", code)

    @skipIfRefEager("prefetch regions are checked at compile time")
    def test_strided_region_raises(self) -> None:
        x = torch.randn((512, 8), device=DEVICE, dtype=torch.float32).t()
        with self.assertRaises(helion.exc.InvalidPrefetchRegion):
            code_and_output(prefetch_rows, (x,))
