from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from helion._compiler.cute.backend import CuteBackend
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec


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

    def test_shared_grid_retains_whole_kernel_budget(self) -> None:
        for size, expected in ((128, 8), (512, 16)):
            with self.subTest(size=size):
                spec = self.make_spec((size,) * 4)
                self.assertEqual(self.apply_floor(spec), [expected] * 4)

    def test_empty_grid_needs_no_device_query(self) -> None:
        spec = self.make_spec(())
        with patch(
            "helion.autotuner.config_spec.num_compute_units",
            side_effect=AssertionError("empty grid queried the device"),
        ):
            spec.raise_grid_block_minimums()


if __name__ == "__main__":
    unittest.main()
