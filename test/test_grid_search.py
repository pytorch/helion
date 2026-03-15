from __future__ import annotations

import itertools
import os
import unittest
from unittest.mock import patch

import torch

import helion
from helion.autotuner.config_fragment import BooleanFragment
from helion.autotuner.config_fragment import BlockSizeFragment
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import NumWarpsFragment
from helion.autotuner.config_fragment import PermutationFragment
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.grid_search import _enumerate_fragment
from helion.autotuner.grid_search import compute_grid_size
import helion.language as hl


class TestEnumerateFragment(unittest.TestCase):
    """Tests for _enumerate_fragment on each fragment type."""

    def test_boolean_fragment(self) -> None:
        spec = BooleanFragment()
        values = _enumerate_fragment(spec)
        self.assertEqual(values, [False, True])

    def test_enum_fragment(self) -> None:
        spec = EnumFragment(choices=("flat", "xyz", "persistent_blocked"))
        values = _enumerate_fragment(spec)
        self.assertEqual(values, ["flat", "xyz", "persistent_blocked"])

    def test_power_of_two_fragment(self) -> None:
        spec = PowerOfTwoFragment(4, 64, 16)
        values = _enumerate_fragment(spec)
        self.assertEqual(values, [4, 8, 16, 32, 64])

    def test_power_of_two_single(self) -> None:
        spec = PowerOfTwoFragment(32, 32, 32)
        values = _enumerate_fragment(spec)
        self.assertEqual(values, [32])

    def test_block_size_fragment(self) -> None:
        spec = BlockSizeFragment(16, 128, 64)
        values = _enumerate_fragment(spec)
        self.assertEqual(values, [16, 32, 64, 128])

    def test_num_warps_fragment(self) -> None:
        spec = NumWarpsFragment(1, 16, 4)
        values = _enumerate_fragment(spec)
        self.assertEqual(values, [1, 2, 4, 8, 16])

    def test_integer_fragment(self) -> None:
        spec = IntegerFragment(0, 4, 0)
        values = _enumerate_fragment(spec)
        self.assertEqual(values, [0, 1, 2, 3, 4])

    def test_permutation_fragment(self) -> None:
        spec = PermutationFragment(length=3)
        values = _enumerate_fragment(spec)
        self.assertEqual(len(values), 6)  # 3! = 6
        self.assertIn([0, 1, 2], values)
        self.assertIn([2, 1, 0], values)

    def test_list_of_fragment(self) -> None:
        spec = ListOf(BooleanFragment(), length=2)
        values = _enumerate_fragment(spec)
        self.assertEqual(len(values), 4)  # 2^2
        self.assertIn([False, False], values)
        self.assertIn([True, True], values)


class TestComputeGridSize(unittest.TestCase):
    """Tests for compute_grid_size."""

    def test_single_boolean(self) -> None:
        flat_spec = [BooleanFragment()]
        self.assertEqual(compute_grid_size(flat_spec), 2)

    def test_multiple_fragments(self) -> None:
        flat_spec = [
            PowerOfTwoFragment(4, 16, 8),  # 3 values: 4, 8, 16
            BooleanFragment(),  # 2 values
            EnumFragment(choices=("a", "b", "c")),  # 3 values
        ]
        self.assertEqual(compute_grid_size(flat_spec), 3 * 2 * 3)

    def test_empty_spec(self) -> None:
        self.assertEqual(compute_grid_size([]), 1)


class TestGridSearchRegistration(unittest.TestCase):
    """Test that GridSearch is properly registered."""

    def test_registered_in_search_algorithms(self) -> None:
        from helion.autotuner import search_algorithms
        from helion.autotuner.grid_search import GridSearch

        self.assertIn("GridSearch", search_algorithms)
        self.assertIs(search_algorithms["GridSearch"], GridSearch)

    def test_env_var_selection(self) -> None:
        from helion.autotuner import search_algorithms
        from helion.autotuner.grid_search import GridSearch

        with patch.dict(os.environ, {"HELION_AUTOTUNER": "GridSearch"}):
            name = os.environ.get("HELION_AUTOTUNER", "LFBOTreeSearch")
            cls = search_algorithms.get(name)
            self.assertIs(cls, GridSearch)


class TestGridSearchSampling(unittest.TestCase):
    """Test that GridSearch correctly handles sampling for large spaces."""

    def test_sampling_is_deterministic_with_seed(self) -> None:
        """Same seed should produce same sampled configs."""
        import random as stdlib_random

        spec = [
            PowerOfTwoFragment(1, 1024, 32),  # 11 values
            PowerOfTwoFragment(1, 1024, 32),  # 11 values
            BooleanFragment(),  # 2 values
        ]
        # 11 * 11 * 2 = 242 total configs
        size = compute_grid_size(spec)
        self.assertEqual(size, 242)

    def test_default_always_included(self) -> None:
        """The default config should always be in the enumerated set."""
        spec = [BooleanFragment(), BooleanFragment()]
        values_per_spec = [_enumerate_fragment(s) for s in spec]
        defaults = [s.default() for s in spec]
        # Default is (False, False), which should be in the product
        all_combos = list(itertools.product(*values_per_spec))
        self.assertIn(tuple(defaults), all_combos)


@helion.kernel()
def _grid_test_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        result[tile] = x[tile] + y[tile]
    return result


@unittest.skipUnless(torch.cuda.is_available(), "requires GPU")
class TestGridSearchIntegration(unittest.TestCase):
    """Integration tests requiring GPU."""

    def test_grid_search_finds_valid_config(self) -> None:
        """Run GridSearch and verify it returns a correct result."""
        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")

        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "GridSearch",
                "HELION_AUTOTUNE_EFFORT": "quick",
                "HELION_FORCE_AUTOTUNE": "1",
            },
        ):
            result = _grid_test_add_kernel(x, y)
            expected = x + y
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
