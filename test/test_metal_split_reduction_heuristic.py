"""The autotuner's seeds for a few-row grid over a K pass its threads split
and reduce after the loop (``metal/split_reduction_heuristic.py``), from
CPU binds; ``test_metal_split_reduction_numerics.py`` runs them."""

from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._compiler.metal.split_reduction_heuristic import (
    MetalSplitReductionHeuristic,
)
from helion._testing import skipUnlessBackends
from helion._testing.metal import heuristic_seeds
from helion._testing.metal_split import split_gemv
from helion._testing.metal_split import split_gemv_args
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

F32 = torch.float32


@helion.kernel(backend="metal", static_shapes=True)
def reduced_in_loop(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The K chunks summed inside the loop: nothing is reduced after it."""
    m, k = x.size()
    n = w.size(0)
    bk = hl.register_block_size(1, k)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k, block_size=bk):
            xk = x[tile_m, tile_k].to(F32)
            wk = w[tile_n, tile_k].to(F32)
            acc = acc + (xk.unsqueeze(1) * wk.unsqueeze(0)).sum(2)
        out[tile_m, tile_n] = acc
    return out


def _seeds(
    bound: helion.runtime.kernel.BoundKernel,
) -> list[tuple[list[int], list[int]]]:
    """The heuristic's seeds' ``(block_sizes, num_threads)``, each on one SIMD
    group."""
    seeds = heuristic_seeds(MetalSplitReductionHeuristic, bound)
    assert all(c.config["num_warps"] == 1 for c in seeds)
    return [(c.config["block_sizes"], c.config["num_threads"]) for c in seeds]


class TestSplitReductionSeeds(unittest.TestCase):
    def test_few_rows(self) -> None:
        """4 rows: the pass block on 32 threads, the rows whole on one thread,
        4 or 2 columns per thread; planted at bind through the registry, the
        default config unchanged."""
        bound = split_gemv.bind(split_gemv_args(4, 512, 1024))
        expected = [([32, 4, 4], [32, 1, 1]), ([32, 4, 2], [32, 1, 1])]
        self.assertEqual(_seeds(bound), expected)
        spec = bound.config_spec
        self.assertIn(MetalSplitReductionHeuristic.name, spec.autotuner_heuristics)
        planted = [
            (c.config["block_sizes"], c.config["num_threads"])
            for c in spec.compiler_seed_configs
        ]
        self.assertEqual(planted, expected)
        self.assertEqual(spec.default_config().config["block_sizes"], [16, 4, 16])

    def test_rows_round_up(self) -> None:
        """3 rows take a row block of 4 (a power of two); 1 row, 1."""
        self.assertEqual(
            [b for b, _ in _seeds(split_gemv.bind(split_gemv_args(3, 512, 1024)))],
            [[32, 4, 4], [32, 4, 2]],
        )
        self.assertEqual(
            [b for b, _ in _seeds(split_gemv.bind(split_gemv_args(1, 512, 1024)))],
            [[32, 1, 4], [32, 1, 2]],
        )

    def test_no_seed(self) -> None:
        """No seed past 32 rows, nor when the chunks are summed in the loop."""
        self.assertEqual(_seeds(split_gemv.bind(split_gemv_args(64, 512, 1024))), [])
        args = (torch.randn(4, 1024), torch.randn(512, 1024))
        self.assertEqual(_seeds(reduced_in_loop.bind(args)), [])


if __name__ == "__main__":
    unittest.main()
