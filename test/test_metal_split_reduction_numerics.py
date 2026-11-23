"""The split-reduction seeds (``metal/split_reduction_heuristic.py``) run
and match a float64 reference on an Apple GPU: a few-row int8 matrix-vector
product at each seed."""

from __future__ import annotations

import sys
import unittest

import pytest
import torch

from helion._compiler.metal.split_reduction_heuristic import (
    MetalSplitReductionHeuristic,
)
from helion._testing import skipUnlessBackends
from helion._testing.metal import heuristic_seeds
from helion._testing.metal import run_with_msl
from helion._testing.metal_split import split_gemv
from helion._testing.metal_split import split_gemv_args
from helion._testing.metal_split import split_gemv_reference

pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests require macOS"),
    skipUnlessBackends(["metal"]),
]


@unittest.skipUnless(
    sys.platform == "darwin" and torch.backends.mps.is_available(),
    "requires a Metal GPU",
)
class TestSplitReductionNumerics(unittest.TestCase):
    def test_seeds_match_reference(self) -> None:
        """Every seed at 1 to 4 rows (3: a row block of 4 over 3 rows, its
        fourth row masked), over K = 2080 = 8 * 256 + 32: the K seed's last
        step, at 1 and 2 rows, runs 32 of its 256 threads
        (``masked_branch``)."""
        for m in (1, 2, 3, 4):
            args = split_gemv_args(m, 256, 2080, "mps")
            ref = split_gemv_reference(args)
            seeds = heuristic_seeds(MetalSplitReductionHeuristic, split_gemv.bind(args))
            self.assertEqual(len(seeds), 3 if m <= 2 else 2)
            for config in seeds:
                with self.subTest(m=m, config=config):
                    out, _, _ = run_with_msl(split_gemv, args, config)
                    assert isinstance(out, torch.Tensor)
                    torch.testing.assert_close(
                        out.cpu().double(), ref, atol=2e-2, rtol=1e-2
                    )


if __name__ == "__main__":
    unittest.main()
