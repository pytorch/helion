"""The autotuner's seeds for a few-row grid over a K pass its threads split
and reduce after the loop (``metal/split_reduction_heuristic.py``), from
CPU binds; ``test_metal_split_reduction_numerics.py`` runs them."""

from __future__ import annotations

from typing import TYPE_CHECKING
import unittest

import pytest
import torch

from helion._compiler.metal.split_reduction_heuristic import (
    MetalSplitReductionHeuristic,
)
from helion._testing import skipUnlessBackends
from helion._testing.metal import heuristic_seeds
from helion._testing.metal_split import reduced_in_loop
from helion._testing.metal_split import split_gemv
from helion._testing.metal_split import split_gemv_args

if TYPE_CHECKING:
    import helion

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

F32 = torch.float32


def _seeds(
    bound: helion.runtime.kernel.BoundKernel,
) -> list[tuple[list[int], list[int], int]]:
    """The heuristic's seeds' ``(block_sizes, num_threads, num_warps)``."""
    seeds = heuristic_seeds(MetalSplitReductionHeuristic, bound)
    return [
        (c.config["block_sizes"], c.config["num_threads"], c.config["num_warps"])
        for c in seeds
    ]


class TestSplitReductionSeeds(unittest.TestCase):
    def test_few_rows(self) -> None:
        """4 rows: the pass block on 32 threads, the rows whole on one thread,
        4 or 2 columns per thread; planted at bind through the registry, the
        default config unchanged."""
        bound = split_gemv.bind(split_gemv_args(4, 512, 1024))
        expected = [([32, 4, 4], [32, 1, 1], 1), ([32, 4, 2], [32, 1, 1], 1)]
        self.assertEqual(_seeds(bound), expected)
        spec = bound.config_spec
        self.assertIn(MetalSplitReductionHeuristic.name, spec.autotuner_heuristics)
        planted = [
            (c.config["block_sizes"], c.config["num_threads"], c.config["num_warps"])
            for c in spec.compiler_seed_configs
        ]
        self.assertEqual(planted, expected)
        self.assertEqual(spec.default_config().config["block_sizes"], [16, 4, 16])

    def test_rows_round_up(self) -> None:
        """3 rows take a row block of 4 (a power of two)."""
        bound = split_gemv.bind(split_gemv_args(3, 512, 1024))
        self.assertEqual([b for b, _, _ in _seeds(bound)], [[32, 4, 4], [32, 4, 2]])

    def test_k_seed(self) -> None:
        """At 1 row the K seed leads, the pass block on 256 threads (8 SIMD
        groups) for one column; at 2 rows it comes last; a K of 64 puts it
        on 64 threads (2 SIMD groups)."""
        self.assertEqual(
            _seeds(split_gemv.bind(split_gemv_args(1, 512, 1024))),
            [
                ([256, 1, 1], [256, 1, 1], 8),
                ([32, 1, 4], [32, 1, 1], 1),
                ([32, 1, 2], [32, 1, 1], 1),
            ],
        )
        self.assertEqual(
            _seeds(split_gemv.bind(split_gemv_args(2, 512, 1024)))[-1],
            ([256, 2, 1], [256, 1, 1], 8),
        )
        self.assertEqual(
            _seeds(split_gemv.bind(split_gemv_args(1, 512, 64)))[0],
            ([64, 1, 1], [64, 1, 1], 2),
        )

    def test_no_seed(self) -> None:
        """No seed past 32 rows, nor when the chunks are summed in the loop."""
        self.assertEqual(_seeds(split_gemv.bind(split_gemv_args(64, 512, 1024))), [])
        args = (torch.randn(4, 1024), torch.randn(512, 1024))
        self.assertEqual(_seeds(reduced_in_loop.bind(args)), [])


if __name__ == "__main__":
    unittest.main()
