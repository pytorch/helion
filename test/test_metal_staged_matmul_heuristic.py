"""The autotuner's seeds for a K-loop MPP matmul that stages an operand per K
tile (``metal/staged_matmul_heuristic.py``), from CPU binds; the GPU module
``test_metal_mpp_matmul.py`` compiles and runs them."""

from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._compiler.metal.search_surface import kernel_facts
from helion._compiler.metal.staged_matmul_heuristic import MetalStagedMatmulHeuristic
from helion._testing import skipUnlessBackends
from helion._testing.metal_staged import dequant_args
from helion._testing.metal_staged import dequant_mm
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

S = {"backend": "metal", "static_shapes": True}
F32 = torch.float32


@helion.kernel(**S)
def dequant_lhs(wq: torch.Tensor, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """W @ y with the dequantized W on the left: the staged side is M."""
    m = wq.size(0)
    k, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=y.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            words = wq[tile_m, tile_k.index // 8]
            nib = (words >> ((tile_k.index % 8) * 4)[None, :]) & 0xF
            w = (nib.to(F32) - 8.0) * s[tile_m, tile_k.index // 32]
            acc = torch.addmm(acc, w.to(y.dtype), y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(**S)
def dequant_gate_up(
    x: torch.Tensor,
    wq: torch.Tensor,
    s: torch.Tensor,
    wq2: torch.Tensor,
    s2: torch.Tensor,
) -> torch.Tensor:
    """Two dequantized rhs over the same blocks in one K loop (a fused gate
    and up projection), their products multiplied."""
    m, k = x.size()
    n = wq.size(0)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        gate = hl.zeros([tile_m, tile_n], dtype=F32)
        up = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            shift = ((tile_k.index % 8) * 4)[None, :]
            nib = (wq[tile_n, tile_k.index // 8] >> shift) & 0xF
            w = (nib.to(F32) - 8.0) * s[tile_n, tile_k.index // 32]
            nib2 = (wq2[tile_n, tile_k.index // 8] >> shift) & 0xF
            w2 = (nib2.to(F32) - 8.0) * s2[tile_n, tile_k.index // 32]
            gate = torch.addmm(gate, x[tile_m, tile_k], w.T.to(x.dtype))
            up = torch.addmm(up, x[tile_m, tile_k], w2.T.to(x.dtype))
        out[tile_m, tile_n] = gate * up
    return out


@helion.kernel(**S)
def plain_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def _seeds(bound: helion.runtime.kernel.BoundKernel) -> list[tuple[list[int], int]]:
    """The heuristic's seeds' ``(block_sizes, num_warps)``."""
    assert bound.host_function is not None
    with bound.env:
        seeds = MetalStagedMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    return [(c.config["block_sizes"], c.config["num_warps"]) for c in seeds]


class TestStagedMatmulSeeds(unittest.TestCase):
    def test_tall_block_shares_the_staged_tile(self) -> None:
        """A dequantized rhs at 511 rows: 256 rows at 8 SIMD groups beside
        narrow blocks of 16 and 32 and K blocks of 256 and 128, planted at
        bind through the registry; the default config is unchanged."""
        bound = dequant_mm.bind(dequant_args(511, 4096, 1024))
        spec = bound.config_spec
        self.assertEqual(kernel_facts(spec).staged_matmuls, ((0, 1, 2),))
        expected = [
            ([256, 16, 256], 8),
            ([256, 32, 256], 8),
            ([256, 16, 128], 8),
            ([256, 32, 128], 8),
        ]
        self.assertEqual(_seeds(bound), expected)
        self.assertIn(MetalStagedMatmulHeuristic.name, spec.autotuner_heuristics)
        planted = [
            (c.config["block_sizes"], c.config["num_warps"])
            for c in spec.compiler_seed_configs
            if "metal_matmul" not in c.config
        ]
        self.assertEqual(planted, expected)
        self.assertEqual(spec.default_config().config["block_sizes"], [16, 16, 16])

    def test_tall_block_follows_the_extent(self) -> None:
        """40 rows take a 64-row block at 2 SIMD groups; a staged lhs makes
        N the tall side; K blocks past K are dropped."""
        self.assertEqual(
            _seeds(dequant_mm.bind(dequant_args(40, 256, 512))),
            [
                ([64, 16, 256], 2),
                ([64, 32, 256], 2),
                ([64, 16, 128], 2),
                ([64, 32, 128], 2),
            ],
        )
        x, wq, s = dequant_args(8, 300, 128)
        bound = dequant_lhs.bind((wq, s, torch.randn(128, 300).to(torch.bfloat16)))
        self.assertEqual(kernel_facts(bound.config_spec).staged_matmuls, ((1, 0, 2),))
        self.assertEqual(_seeds(bound), [([16, 256, 128], 8), ([32, 256, 128], 8)])

    def test_matmuls_sharing_blocks(self) -> None:
        """Two staged matmuls over the same blocks take the one-matmul
        seeds."""
        x, wq, s = dequant_args(511, 4096, 1024)
        _, wq2, s2 = dequant_args(511, 4096, 1024)
        bound = dequant_gate_up.bind((x, wq, s, wq2, s2))
        self.assertEqual(
            kernel_facts(bound.config_spec).staged_matmuls, ((0, 1, 2), (0, 1, 2))
        )
        self.assertEqual(
            _seeds(bound),
            [
                ([256, 16, 256], 8),
                ([256, 32, 256], 8),
                ([256, 16, 128], 8),
                ([256, 32, 128], 8),
            ],
        )

    def test_no_seed(self) -> None:
        """No seed for a few rows (8: fragments and row kernels serve them),
        nor for a matmul of two device views, which stages nothing."""
        self.assertEqual(_seeds(dequant_mm.bind(dequant_args(8, 4096, 1024))), [])
        bound = plain_mm.bind((torch.randn(512, 256), torch.randn(256, 512)))
        self.assertEqual(kernel_facts(bound.config_spec).staged_matmuls, ())
        self.assertEqual(_seeds(bound), [])


if __name__ == "__main__":
    unittest.main()
