"""The autotuner's seeds for a few-row K-loop matmul on fragments
(``metal/fragment_matmul_heuristic.py``), from CPU binds; the GPU module
``test_metal_fragment_mma.py`` compiles and runs one."""

from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._compiler.metal.fragment_matmul_heuristic import (
    MetalFragmentMatmulHeuristic,
)
from helion._compiler.metal.search_surface import kernel_facts
from helion._testing import skipUnlessBackends
from helion._testing.metal import heuristic_seeds
from helion._testing.metal_fragment import affine_args
from helion._testing.metal_fragment import affine_mm
from helion._testing.metal_staged import dequant_args
from helion._testing.metal_staged import dequant_mm
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

F32 = torch.float32


@helion.kernel(backend="metal", static_shapes=True)
def two_loops(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two K loops over different K blocks, each storing its own product:
    no one tile serves both."""
    m, k = x.size()
    k2 = z.size(0)
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    out2 = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        a = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            a = torch.addmm(a, x[tile_m, tile_k], y[tile_k, tile_n])
        b = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k2):
            b = torch.addmm(b, x[tile_m, tile_k], z[tile_k, tile_n])
        out[tile_m, tile_n] = a
        out2[tile_m, tile_n] = b
    return out, out2


def _seeds(bound: helion.runtime.kernel.BoundKernel) -> list[tuple[list[int], int]]:
    """The heuristic's seeds' ``(block_sizes, num_warps)``, each a fragment
    config."""
    seeds = heuristic_seeds(MetalFragmentMatmulHeuristic, bound)
    assert all(c.config["metal_matmul"] == "fragment" for c in seeds)
    return [(c.config["block_sizes"], c.config["num_warps"]) for c in seeds]


#: At K 1024: 8 K tiles of 128, so 8 SIMD groups split the loop (16 capped
#: at 8).
FEW_ROWS = [([16, 16, 128], 8), ([16, 32, 128], 8)]


class TestFragmentMatmulSeeds(unittest.TestCase):
    def test_few_rows(self) -> None:
        """A dequantized rhs at 8 rows: the largest fragment K tile (128),
        16 or 32 columns, 8 SIMD groups splitting K's 8 tiles; 16 where K
        has 16 tiles or more; planted at bind through the registry, the
        default config unchanged."""
        bound = dequant_mm.bind(dequant_args(8, 4096, 1024))
        spec = bound.config_spec
        self.assertEqual(kernel_facts(spec).fragment_blocks, ((0, 1, 2),))
        self.assertEqual(_seeds(bound), FEW_ROWS)
        self.assertIn(MetalFragmentMatmulHeuristic.name, spec.autotuner_heuristics)
        # Among the search surface's fragment seeds, which name no blocks.
        planted = [
            (c.config["block_sizes"], c.config["num_warps"])
            for c in spec.compiler_seed_configs
            if c.config.get("metal_matmul") == "fragment" and "block_sizes" in c.config
        ]
        self.assertEqual(planted, FEW_ROWS)
        self.assertEqual(spec.default_config().config["block_sizes"], [16, 16, 16])
        self.assertEqual(
            _seeds(dequant_mm.bind(dequant_args(8, 4096, 2048))),
            [
                ([16, 16, 128], 8),
                ([16, 16, 128], 16),
                ([16, 32, 128], 8),
                ([16, 32, 128], 16),
            ],
        )

    def test_straddling_codes(self) -> None:
        """5-bit codes that straddle 16-bit words, the case the seeds are
        for (a 16-position run per lane holds whole words), at 4, 8 and 32
        rows."""
        for m in (4, 8, 32):
            with self.subTest(m=m):
                self.assertEqual(
                    _seeds(affine_mm.bind(affine_args(5, 128, m, 256, 1024))), FEW_ROWS
                )

    def test_k_extent(self) -> None:
        """A K shorter than the largest K tile takes its own extent, one
        tile, which no SIMD groups split; 4 tiles split over 4 SIMD
        groups, the most that each have a tile."""
        self.assertEqual(
            _seeds(dequant_mm.bind(dequant_args(8, 256, 64))),
            [([16, 16, 64], 1), ([16, 32, 64], 1)],
        )
        self.assertEqual(
            _seeds(dequant_mm.bind(dequant_args(8, 256, 512))),
            [([16, 16, 128], 4), ([16, 32, 128], 4)],
        )

    def test_no_seed(self) -> None:
        """No seed past 32 rows (the staged-matmul seeds and MPP's default
        serve those), nor for matmuls over different K loops."""
        self.assertEqual(_seeds(dequant_mm.bind(dequant_args(64, 4096, 1024))), [])
        bound = two_loops.bind(
            (torch.randn(8, 256), torch.randn(256, 64), torch.randn(128, 64))
        )
        self.assertEqual(len(set(kernel_facts(bound.config_spec).fragment_blocks)), 2)
        self.assertEqual(_seeds(bound), [])


if __name__ == "__main__":
    unittest.main()
