from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.cute.backend import CuteBackend
from helion._testing import skipIfRefEager
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import KernelGridFact
from helion.autotuner.config_spec import ReductionLoopSpec
from helion.autotuner.config_spec import RootGridFact
import helion.language as hl


class TestCuteRegionGridFloors(unittest.TestCase):
    def make_spec(self, sizes: tuple[int, ...]) -> ConfigSpec:
        spec = ConfigSpec(
            backend=CuteBackend(),
            target_device_capability=(10, 0),
            device=torch.device("cpu"),
            num_sm=148,
        )
        for block_id, size in enumerate(sizes):
            spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=size))
        spec.grid_block_ids = list(range(len(sizes)))
        return spec

    def apply_floor(self, spec: ConfigSpec) -> list[int]:
        with patch("helion.autotuner.config_spec.num_compute_units", return_value=148):
            spec.raise_grid_block_minimums()
        return [item.autotuner_min for item in spec.block_sizes]

    def test_pointwise_launch_uses_its_own_rank(self) -> None:
        # The fissioned launch (0, 1) and the remaining shared grid (2, 3) are
        # separate two-dimensional grids; each owns the whole budget.
        for size, expected in (
            (128, [1, 1, 1, 1]),
            (512, [4, 4, 4, 4]),
        ):
            with self.subTest(size=size):
                spec = self.make_spec((size,) * 4)
                spec.cute_pointwise_region_grid_groups = ((0, 1),)
                self.assertEqual(self.apply_floor(spec), expected)
                self.assertEqual([item.min_size for item in spec.block_sizes], [1] * 4)

    def test_shared_grid_retains_whole_kernel_budget(self) -> None:
        for size, expected in ((128, 8), (512, 16)):
            with self.subTest(size=size):
                spec = self.make_spec((size,) * 4)
                self.assertEqual(self.apply_floor(spec), [expected] * 4)

    def test_independent_axes_search_below_soft_floor_without_changing_defaults(
        self,
    ) -> None:
        spec = self.make_spec((1024,) * 4)
        spec.cute_pointwise_region_grid_groups = ((0, 1),)
        self.apply_floor(spec)
        defaults = [item._fragment(spec).default() for item in spec.block_sizes]
        assert [item._fragment(spec).low for item in spec.block_sizes] == [1, 1, 8, 8]
        spec.cute_pointwise_region_grid_groups = ()
        assert [item._fragment(spec).low for item in spec.block_sizes] == [8, 8, 8, 8]
        assert defaults == [item._fragment(spec).default() for item in spec.block_sizes]

    def test_reduction_default_keeps_the_old_clamped_floor(self) -> None:
        spec = self.make_spec((1024,) * 4)
        spec.cute_pointwise_region_grid_groups = ((0, 1),)
        self.apply_floor(spec)
        # A later reduction fact makes the raw per-axis default smaller than
        # the already established occupancy floor.
        spec.block_sizes.append(BlockSizeSpec(block_id=4, size_hint=1024))
        spec.reduction_loops.append(ReductionLoopSpec(block_id=4, size_hint=1024))
        spec.cute_pointwise_region_grid_groups = ()
        old_fragments = [item._fragment(spec) for item in spec.block_sizes]
        assert old_fragments[0].default_val < old_fragments[0].low
        old_default = spec.default_config()
        spec.cute_pointwise_region_grid_groups = ((0, 1),)
        new_fragments = [item._fragment(spec) for item in spec.block_sizes]
        assert new_fragments[0].low == 1 < old_fragments[0].low
        assert [item.default() for item in new_fragments] == [
            item.default() for item in old_fragments
        ]
        assert spec.default_config() == old_default

    def test_separate_grids_can_have_different_ranks(self) -> None:
        spec = self.make_spec((1280,) * 4)
        spec.cute_pointwise_region_grid_groups = ((0,), (1, 2))
        # Axis 3 is a one-dimensional grid of 1280 programs: under budget.
        self.assertEqual(self.apply_floor(spec), [1, 8, 8, 1])

    def test_hard_minimum_and_maximum_remain_authoritative(self) -> None:
        spec = self.make_spec((512,) * 4)
        spec.cute_pointwise_region_grid_groups = ((0, 1),)
        spec.block_sizes[0].update_min(8)
        spec.block_sizes[0].autotuner_min = 8
        spec.block_sizes[1].update_max(2)
        self.assertEqual(self.apply_floor(spec), [8, 2, 4, 4])
        self.assertEqual(spec.block_sizes[0]._fragment(spec).low, 8)
        self.assertEqual(spec.block_sizes[1]._fragment(spec).high, 2)

    def test_small_axis_hands_its_share_to_the_large_axis(self) -> None:
        # grpo_loss forward at B=4, L=256: the whole grid is 1024 programs at
        # tile 1x1, far under the 148*64 budget, so L must not be floored.
        # The legacy per-axis rule gave each axis sqrt(budget) ~ 98 blocks
        # and forbade tile_l=1 (256 // 98 = 2).
        spec = self.make_spec((4, 256))
        self.assertEqual(self.apply_floor(spec), [1, 1])
        # A large second axis still receives the budget the small one left.
        spec = self.make_spec((4, 1 << 20))
        self.assertEqual(self.apply_floor(spec), [1, 32])
        # Both axes large: the even share, as before.
        spec = self.make_spec((4096, 4096))
        self.assertEqual(self.apply_floor(spec), [32, 32])

    def test_each_root_grid_owns_a_full_budget(self) -> None:
        spec = self.make_spec((65536, 65536, 4, 256))
        spec.kernel_grid_fact = KernelGridFact(
            (RootGridFact(0, (0,)), RootGridFact(1, (1,)), RootGridFact(2, (2, 3))),
            (),
        )
        # One-dimensional roots floor like a single one-dimensional grid
        # (65536 // 9472 = 6 -> 4); the small two-dimensional root is free.
        self.assertEqual(self.apply_floor(spec), [4, 4, 1, 1])
        # An axis promoted after the topology was recorded joins the
        # unassigned grid instead of being dropped from the floor.
        spec = self.make_spec((65536, 65536))
        spec.kernel_grid_fact = KernelGridFact((RootGridFact(0, (0,)),), ())
        self.assertEqual(self.apply_floor(spec), [4, 4])

    def test_tied_axes_share_the_budget_equally(self) -> None:
        # Identical extents take one even split at once, so the floors are
        # the legacy per-axis ceil(budget ** (1 / n)) for a uniform grid and
        # do not depend on the axes' source order.
        for sizes, expected in (
            ((3104, 3104), [16, 16]),  # 3104 // 98 = 31 -> 16 on both axes
            ((336, 336, 336), [8, 8, 8]),  # 336 // 22 = 15 -> 8 on all three
            ((4096, 4096), [32, 32]),
            ((3104, 3104, 4), [1 if size == 4 else 16 for size in (3104, 3104, 4)]),
        ):
            with self.subTest(sizes=sizes):
                self.assertEqual(self.apply_floor(self.make_spec(sizes)), expected)
                reversed_spec = self.make_spec(tuple(reversed(sizes)))
                self.assertEqual(
                    self.apply_floor(reversed_spec), list(reversed(expected))
                )

    def test_empty_grid_needs_no_device_query(self) -> None:
        spec = self.make_spec(())
        with patch(
            "helion.autotuner.config_spec.num_compute_units",
            side_effect=AssertionError("empty grid queried the device"),
        ):
            spec.raise_grid_block_minimums()


class TestGridFloorsThroughDeviceIR(unittest.TestCase):
    @skipIfRefEager("the device IR is not built in ref eager mode")
    def test_two_root_loops_each_own_a_budget(self) -> None:
        # The floors are raised after the kernel's grid topology is recorded,
        # so two independent root loops are two one-dimensional grids
        # (65536 // 9472 = 6 -> 4 each) and not one 65536 x 65536 grid
        # (65536 // 98 = 668, capped by the 2-D default 32 -> [32, 32]).
        @helion.kernel(backend="triton", static_shapes=True, autotune_effort="full")
        def two_roots(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            a = torch.empty_like(x)
            b = torch.empty_like(y)
            for tile in hl.tile(x.size(0)):
                a[tile] = x[tile] + 1
            for tile in hl.tile(y.size(0)):
                b[tile] = y[tile] + 1
            return a, b

        with (
            _mock_cuda_unavailable(),
            patch("helion.autotuner.config_spec.num_compute_units", return_value=148),
        ):
            bound = _cpu_bind(two_roots, (torch.empty(65536), torch.empty(65536)))
        spec = bound.config_spec
        assert spec.kernel_grid_fact is not None
        self.assertEqual(len(spec.kernel_grid_fact.grid_groups), 2)
        self.assertEqual([item.autotuner_min for item in spec.block_sizes], [4, 4])


if __name__ == "__main__":
    unittest.main()
