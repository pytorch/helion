"""Which grids the Metal autotuner times (``metal/grid_budget.py``): grid
blocks stay unfloored, and candidates are screened on their work per
threadgroup.  Binds CPU tensors only."""

from __future__ import annotations

import unittest
from unittest import mock

import pytest
import torch

import helion
from helion._compiler.metal import grid_budget
from helion._compiler.metal.backend import MetalBackend
from helion._testing import skipUnlessBackends
from helion.autotuner.finite_search import FiniteSearch
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

# An M4 Pro's: an occupancy budget of 16 * 64 = 1024 threadgroups.
_CORES = 16


@helion.kernel(backend="metal", static_shapes=True)
def add_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(backend="metal", static_shapes=True)
def few_row_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile, :].sum(-1)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def tile_bounded_row_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        for tile_r in hl.tile(tile_m.begin, tile_m.end):
            out[tile_r] = x[tile_r, :].sum(-1)
    return out


class TestMetalGridBudget(unittest.TestCase):
    def setUp(self) -> None:
        patch = mock.patch.object(grid_budget, "gpu_cores", return_value=_CORES)
        patch.start()
        self.addCleanup(patch.stop)

    def _viable(
        self, kernel: helion.Kernel, args: tuple[torch.Tensor, ...], blocks: list[int]
    ) -> bool:
        bound = kernel.bind(args)
        spec = bound.config_spec
        return spec.backend.autotune_config_is_viable(
            spec, helion.Config(block_sizes=blocks)
        )

    def _roots(
        self, kernel: helion.Kernel, args: tuple[torch.Tensor, ...]
    ) -> tuple[grid_budget.RootWork, ...]:
        backend = kernel.bind(args).config_spec.backend
        assert isinstance(backend, MetalBackend)
        return backend._grid_roots

    def test_grid_blocks_are_not_floored(self) -> None:
        """A few-row grid keeps its long axis searchable down to block 1: the
        shared per-axis floor would size the 17408-wide axis as if the 4-row
        axis ran at block 1, and so require 8 columns per block."""
        x = torch.empty(4, 17408)
        spec = add_one.bind((x,)).config_spec
        self.assertEqual(
            [b.autotuner_min for b in spec.block_sizes],
            [b.min_size for b in spec.block_sizes],
        )
        spec.raise_grid_block_minimums()
        self.assertEqual(spec.block_sizes.block_id_lookup(1).autotuner_min, 8)

    def test_few_row_matmul_reaches_one_column(self) -> None:
        """A K loop makes a small tile real work: 4 rows x 17408 columns at one
        or two columns per threadgroup (17408 or 8704 threadgroups, 5120 K
        trips each) are timed."""
        args = (
            torch.empty(4, 5120, dtype=torch.float16),
            torch.empty(5120, 17408, dtype=torch.float16),
        )
        for blocks in ([4, 2, 32], [4, 1, 32], [1, 1, 32]):
            self.assertTrue(self._viable(few_row_matmul, args, blocks), blocks)

    def test_elementwise_needs_a_simd_group_of_work(self) -> None:
        """Past the occupancy budget, an elementwise threadgroup must cover
        one SIMD group's worth of elements: blocks below 32 over 2**24 are not
        timed; 32 (the default) is."""
        args = (torch.empty(1 << 24, dtype=torch.float16),)
        self.assertFalse(self._viable(add_one, args, [1]))
        self.assertFalse(self._viable(add_one, args, [16]))
        self.assertTrue(self._viable(add_one, args, [32]))
        self.assertTrue(self._viable(add_one, args, [1024]))
        # 2-D: the product of the blocks counts.
        args = (torch.empty(4096, 4096, dtype=torch.float16),)
        self.assertFalse(self._viable(add_one, args, [4, 4]))
        self.assertTrue(self._viable(add_one, args, [1, 32]))

    def test_small_grids_fit_the_budget(self) -> None:
        """A grid within 16 * 64 threadgroups is timed at any block: a
        latency-bound elementwise kernel over 4096 elements at 8 per
        threadgroup (512 threadgroups)."""
        args = (torch.empty(4096, dtype=torch.float16),)
        self.assertTrue(self._viable(add_one, args, [8]))
        self.assertFalse(self._viable(add_one, args, [1]))

    def test_reduced_extent_counts_as_work(self) -> None:
        """A row reduction's reduced extent is work: one 64-wide row per
        threadgroup over 65573 rows is timed."""
        args = (torch.empty(65573, 64, dtype=torch.float16),)
        self.assertTrue(self._viable(row_sum, args, [1]))
        args = (torch.empty(65573, 16, dtype=torch.float16),)
        self.assertFalse(self._viable(row_sum, args, [1]))
        self.assertTrue(self._viable(row_sum, args, [2]))

    def test_loop_bounded_by_a_tile_counts_one_trip(self) -> None:
        """A loop whose bounds come from another tile has no extent of its
        own (its trips depend on the candidate), so it counts one trip; the
        reduction inside it still counts its 8."""
        args = (torch.empty(1 << 16, 8, dtype=torch.float16),)
        roots = self._roots(tile_bounded_row_sum, args)
        self.assertEqual([root.trips for root in roots], [8])

    def test_empty_axes_launch_nothing(self) -> None:
        """An empty grid axis launches no threadgroups, so any block passes."""
        for shape in ((0,), (0, 16), (16, 0)):
            args = (torch.empty(shape, dtype=torch.float16),)
            spec = add_one.bind(args).config_spec
            blocks = [b._fragment(spec).default_val for b in spec.block_sizes]
            with self.subTest(shape=shape):
                self.assertTrue(self._viable(add_one, args, blocks))
                self.assertTrue(self._viable(add_one, args, [1] * len(blocks)))

    def test_user_configs_are_not_screened(self) -> None:
        """A config given to the kernel runs as given: the screen is the
        autotuner's (``autotune_config_is_viable``), not codegen's."""
        x = torch.empty(1 << 24, dtype=torch.float16)
        bound = add_one.bind((x,))
        code = bound.to_triton_code(helion.Config(block_sizes=[1]))
        self.assertIn("_block_dims=(1, 1, 1)", code)

    def test_given_configs_are_not_screened(self) -> None:
        """A kernel's ``configs=[...]`` are timed as given: blocks the screen
        rejects as candidates (below 32 elements over 2**24) pass the
        finite search's filter, which would otherwise leave it nothing."""
        x = torch.empty(1 << 24, dtype=torch.float16)
        bound = add_one.bind((x,))
        configs = [helion.Config(block_sizes=[8]), helion.Config(block_sizes=[16])]
        spec = bound.config_spec
        self.assertFalse(
            any(spec.backend.autotune_config_is_viable(spec, c) for c in configs)
        )
        search = FiniteSearch(bound, (x,), configs)
        self.assertEqual(search._apply_config_filter(configs)[1], [0, 1])


if __name__ == "__main__":
    unittest.main()
