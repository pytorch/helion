"""The autotuner's seeds for a large K-loop matmul on MPP
(``metal/mpp_matmul_heuristic.py``), from CPU binds."""

from __future__ import annotations

from typing import TYPE_CHECKING
import unittest

import pytest
import torch

import helion
from helion._compiler.metal.mpp_matmul_heuristic import MetalMppMatmulHeuristic
from helion._testing import skipUnlessBackends
from helion._testing.metal_staged import dequant_args
from helion._testing.metal_staged import dequant_mm
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

F32 = torch.float32


@helion.kernel(backend="metal", static_shapes=True)
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


@helion.kernel(backend="metal", static_shapes=True)
def abs_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``plain_mm`` of both operands' magnitudes, each computed."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k].abs(), y[tile_k, tile_n].abs())
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def half_out_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """An ``hl.dot`` into a half accumulator, which MPP doesn't plan: it
    runs on the scalar contraction."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float16)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                x[tile_m, tile_k], y[tile_k, tile_n], acc=acc, out_dtype=torch.float16
            )
        out[tile_m, tile_n] = acc
    return out


def _seeds(bound: BoundKernel) -> list[tuple[object, object]]:
    """The heuristic's seeds' ``(block_sizes, num_warps)``."""
    assert bound.host_function is not None
    with bound.env:
        seeds = MetalMppMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    for seed in seeds:
        assert seed.config["metal_matmul"] == "mpp"
    return [(c.config["block_sizes"], c.config["num_warps"]) for c in seeds]


def _mm_args(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(m, k, dtype=torch.bfloat16),
        torch.randn(k, n, dtype=torch.bfloat16),
    )


class TestMppMatmulSeeds(unittest.TestCase):
    def test_large_matmul(self) -> None:
        """Both tiles, planted at bind through the registry ahead of the
        search surface's seeds."""
        bound = plain_mm.bind(_mm_args(1024, 2048, 1024))
        expected = [([32, 64, 256], 2), ([32, 128, 256], 4)]
        self.assertEqual(_seeds(bound), expected)
        spec = bound.config_spec
        self.assertIn(MetalMppMatmulHeuristic.name, spec.autotuner_heuristics)
        planted = [
            (c.config["block_sizes"], c.config["num_warps"])
            for c in spec.compiler_seed_configs[: len(expected)]
        ]
        self.assertEqual(planted, expected)

    def test_short_k_caps_the_k_block(self) -> None:
        self.assertEqual(
            _seeds(plain_mm.bind(_mm_args(512, 512, 64))),
            [([32, 64, 64], 2), ([32, 128, 64], 4)],
        )

    def test_few_rows_get_none(self) -> None:
        """Fewer rows than MIN_EXTENT: few-row matmuls have their own seeds."""
        self.assertEqual(_seeds(plain_mm.bind(_mm_args(64, 4096, 1024))), [])

    def test_operands_staged_on_both_sides_get_none(self) -> None:
        """Operands MPP stages in threadgroup tiles (computed ones), which a
        large tile's would overflow."""
        self.assertEqual(_seeds(abs_mm.bind(_mm_args(512, 512, 512))), [])

    def test_matmul_mpp_does_not_plan_gets_none(self) -> None:
        """A large matmul on the scalar contraction: the tiles are MPP's."""
        x = torch.randn(512, 512, dtype=torch.float16)
        self.assertEqual(_seeds(half_out_dot.bind((x, x.clone()))), [])

    def test_staged_matmul_gets_none(self) -> None:
        """A dequantized operand is ``staged_matmul_heuristic``'s."""
        self.assertEqual(_seeds(dequant_mm.bind(dequant_args(511, 4096, 1024))), [])


if __name__ == "__main__":
    unittest.main()
