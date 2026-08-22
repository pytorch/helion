"""Unit tests for the flydsl backend's elementwise (non-reduction) path.

Covers the minimal flydsl backend: grid mapping, per-thread vector load/store,
elementwise math, and column-tail masking -- no whole-row reductions.

flydsl is AMD/ROCm-only and experimental, so the whole module is skipped when
it is not importable.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl

pytest.importorskip("flydsl")


@helion.kernel(backend="flydsl")
def elementwise_double(x: torch.Tensor) -> torch.Tensor:
    # Pure elementwise map: no reduction (grid mapping + load/store/mul only).
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = x[tile_m, :] * 2.0
    return out


@helion.kernel(backend="flydsl")
def elementwise_min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Elementwise torch.minimum (exercises the minimum op override, which has
    # no native flydsl Vector method and uses -max(-a,-b)).
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = torch.minimum(x[tile_m, :], y[tile_m, :])
    return out


class TestFlydslElementwise(TestCase):
    def test_elementwise_map_no_reduction(self) -> None:
        # Non-reduction path: grid mapping + elementwise + load/store, no fold.
        for bm, dt in ((1, torch.float16), (4, torch.float16), (1, torch.float32)):
            x = torch.randn(16, 512, device=DEVICE, dtype=dt)
            _, out = code_and_output(elementwise_double, (x,), block_sizes=[bm])
            torch.testing.assert_close(out, x * 2.0)

    def test_elementwise_map_column_tail(self) -> None:
        # Non-reduction path with N not a multiple of the vector width.
        x = torch.randn(8, 300, device=DEVICE, dtype=torch.float16)
        _, out = code_and_output(elementwise_double, (x,), block_sizes=[1])
        torch.testing.assert_close(out, x * 2.0)

    def test_elementwise_minimum(self) -> None:
        # torch.minimum override (no native flydsl min method; -max(-a,-b)).
        x = torch.randn(8, 512, device=DEVICE, dtype=torch.float16)
        y = torch.randn(8, 512, device=DEVICE, dtype=torch.float16)
        _, out = code_and_output(elementwise_min, (x, y), block_sizes=[1])
        torch.testing.assert_close(out, torch.minimum(x, y))

    def test_autotune_elementwise(self) -> None:
        # The flydsl autotuner enumerates the bm (rows/block) knob and
        # FiniteSearch-picks a valid, correct config in-process (no subprocess
        # benchmark workers, which the flydsl JIT cannot survive).
        x = torch.randn(64, 512, device=DEVICE, dtype=torch.float16)
        bk = elementwise_double.bind((x,))
        cfg = bk.autotune((x,), force=True)
        self.assertIn("block_sizes", cfg.config)
        out = bk.compile_config(cfg)(x)
        torch.testing.assert_close(out, x * 2.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
