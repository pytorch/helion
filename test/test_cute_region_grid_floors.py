from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from helion._compiler.cute.backend import CuteBackend
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import ReductionLoopSpec


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
        for size, expected in (
            (128, [1, 1, 8, 8]),
            (512, [4, 4, 16, 16]),
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
        assert [item._fragment(spec).low for item in spec.block_sizes] == [1, 1, 16, 16]
        spec.cute_pointwise_region_grid_groups = ()
        assert [item._fragment(spec).low for item in spec.block_sizes] == [8, 8, 16, 16]
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
        self.assertEqual(self.apply_floor(spec), [1, 8, 8, 16])

    def test_hard_minimum_and_maximum_remain_authoritative(self) -> None:
        spec = self.make_spec((512,) * 4)
        spec.cute_pointwise_region_grid_groups = ((0, 1),)
        spec.block_sizes[0].update_min(8)
        spec.block_sizes[0].autotuner_min = 8
        spec.block_sizes[1].update_max(2)
        self.assertEqual(self.apply_floor(spec), [8, 2, 16, 16])
        self.assertEqual(spec.block_sizes[0]._fragment(spec).low, 8)
        self.assertEqual(spec.block_sizes[1]._fragment(spec).high, 2)

    def test_empty_grid_needs_no_device_query(self) -> None:
        spec = self.make_spec(())
        with patch(
            "helion.autotuner.config_spec.num_compute_units",
            side_effect=AssertionError("empty grid queried the device"),
        ):
            spec.raise_grid_block_minimums()


if __name__ == "__main__":
    unittest.main()
