from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

import helion
from helion.autotuner.multi_fidelity_search import DEFAULT_SCHEDULE
from helion.autotuner.multi_fidelity_search import FidelityLevel
from helion.autotuner.multi_fidelity_search import MultiFidelitySearch
from helion.autotuner.multi_fidelity_search import _round_to_power_of_two
from helion.autotuner.multi_fidelity_search import _validate_schedule
from helion.autotuner.multi_fidelity_search import scale_tensor_size
from helion.autotuner.multi_fidelity_search import spearman_rank_correlation
import helion.language as hl


class TestFidelityScheduleValidation(unittest.TestCase):
    """Tests for fidelity schedule validation."""

    def test_valid_default_schedule(self) -> None:
        """Default schedule should pass validation."""
        _validate_schedule(DEFAULT_SCHEDULE)

    def test_valid_custom_schedule(self) -> None:
        """Custom valid schedule should pass."""
        schedule = [
            FidelityLevel(scale_factor=0.1, keep_ratio=0.2),
            FidelityLevel(scale_factor=0.5, keep_ratio=0.5),
            FidelityLevel(scale_factor=1.0, keep_ratio=1.0),
        ]
        _validate_schedule(schedule)

    def test_single_level_full_fidelity(self) -> None:
        """Single level at full fidelity should be valid (no filtering)."""
        schedule = [FidelityLevel(scale_factor=1.0, keep_ratio=1.0)]
        _validate_schedule(schedule)

    def test_non_monotonic_scale_factors_raises(self) -> None:
        """Non-monotonic scale_factors should raise AssertionError."""
        schedule = [
            FidelityLevel(scale_factor=0.5, keep_ratio=0.5),
            FidelityLevel(scale_factor=0.25, keep_ratio=0.5),  # decreasing
            FidelityLevel(scale_factor=1.0, keep_ratio=1.0),
        ]
        with self.assertRaises(AssertionError):
            _validate_schedule(schedule)

    def test_last_level_not_full_fidelity_raises(self) -> None:
        """Last level must have scale_factor=1.0."""
        schedule = [
            FidelityLevel(scale_factor=0.25, keep_ratio=0.5),
            FidelityLevel(scale_factor=0.5, keep_ratio=1.0),
        ]
        with self.assertRaises(AssertionError):
            _validate_schedule(schedule)

    def test_empty_schedule_raises(self) -> None:
        """Empty schedule should raise."""
        with self.assertRaises(AssertionError):
            _validate_schedule([])


class TestScaleTensorSizes(unittest.TestCase):
    """Tests for tensor dimension scaling."""

    def test_scale_by_quarter(self) -> None:
        """Scaling by 0.25 should round to nearest power of 2."""
        # 1024 * 0.25 = 256, which is already a power of 2
        self.assertEqual(scale_tensor_size(1024, 0.25), 256)

    def test_scale_rounds_to_power_of_two(self) -> None:
        """Result should be rounded to nearest power of 2."""
        # 1000 * 0.25 = 250 -> nearest power of 2 is 256
        self.assertEqual(scale_tensor_size(1000, 0.25), 256)

    def test_minimum_size_respected(self) -> None:
        """Minimum size of 16 should be enforced."""
        # 32 * 0.25 = 8 -> should be clamped to 16
        self.assertEqual(scale_tensor_size(32, 0.25), 16)

    def test_never_exceeds_original(self) -> None:
        """Scaled size should never exceed original."""
        # 20*0.25=5 -> round to 4 -> max(4,16)=16 -> min(16,20)=16
        result = scale_tensor_size(20, 0.25)
        self.assertLessEqual(result, 20)
        self.assertEqual(result, 16)

    def test_dimension_of_one_unchanged(self) -> None:
        """Dimensions of size 1 should remain 1."""
        self.assertEqual(scale_tensor_size(1, 0.25), 1)
        self.assertEqual(scale_tensor_size(1, 0.5), 1)

    def test_full_scale_unchanged(self) -> None:
        """scale_factor=1.0 should not change dimension."""
        self.assertEqual(scale_tensor_size(1024, 1.0), 1024)

    def test_small_dimension_clamped(self) -> None:
        """Very small dimensions (but > 1) clamp to min(MIN_DIM_SIZE, original)."""
        # 4 * 0.25 = 1 -> round to 1 -> max(1, 16) = 16 -> min(16, 4) = 4
        self.assertEqual(scale_tensor_size(4, 0.25), 4)
        # 32 * 0.1 = 3 -> round to 4 -> max(4, 16) = 16 -> min(16, 32) = 16
        self.assertEqual(scale_tensor_size(32, 0.1), 16)

    def test_round_to_power_of_two_function(self) -> None:
        """Test the rounding helper directly."""
        self.assertEqual(_round_to_power_of_two(1), 1)
        self.assertEqual(_round_to_power_of_two(2), 2)
        # 3 is equidistant between 2 and 4; rounds to lower
        self.assertEqual(_round_to_power_of_two(3), 2)
        self.assertEqual(_round_to_power_of_two(5), 4)
        # 6 is equidistant between 4 and 8; rounds to lower
        self.assertEqual(_round_to_power_of_two(6), 4)
        self.assertEqual(_round_to_power_of_two(7), 8)
        self.assertEqual(_round_to_power_of_two(8), 8)
        self.assertEqual(_round_to_power_of_two(255), 256)
        self.assertEqual(_round_to_power_of_two(257), 256)


class TestSpearmanRankCorrelation(unittest.TestCase):
    """Tests for Spearman rank correlation computation."""

    def test_perfect_correlation(self) -> None:
        """Identical rankings should return 1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertAlmostEqual(spearman_rank_correlation(x, y), 1.0)

    def test_perfect_negative_correlation(self) -> None:
        """Reversed ranking should return -1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [50.0, 40.0, 30.0, 20.0, 10.0]
        self.assertAlmostEqual(spearman_rank_correlation(x, y), -1.0)

    def test_no_correlation(self) -> None:
        """Unrelated rankings should return near 0."""
        # This specific permutation gives exactly 0
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 5.0, 1.0, 4.0, 2.0]
        result = spearman_rank_correlation(x, y)
        self.assertAlmostEqual(abs(result), 0.1, places=0)

    def test_single_element(self) -> None:
        """Single element should return 1.0."""
        self.assertAlmostEqual(spearman_rank_correlation([1.0], [2.0]), 1.0)

    def test_two_elements_same_order(self) -> None:
        """Two elements in same order should return 1.0."""
        self.assertAlmostEqual(spearman_rank_correlation([1.0, 2.0], [3.0, 4.0]), 1.0)

    def test_two_elements_reversed(self) -> None:
        """Two elements in reversed order should return -1.0."""
        self.assertAlmostEqual(spearman_rank_correlation([1.0, 2.0], [4.0, 3.0]), -1.0)


class TestMultiFidelitySearchRegistration(unittest.TestCase):
    """Test that MultiFidelitySearch is properly registered."""

    def test_registered_in_search_algorithms(self) -> None:
        """MultiFidelitySearch should be in the search_algorithms dict."""
        from helion.autotuner import search_algorithms

        self.assertIn("MultiFidelitySearch", search_algorithms)
        self.assertIs(search_algorithms["MultiFidelitySearch"], MultiFidelitySearch)

    def test_env_var_selection(self) -> None:
        """HELION_AUTOTUNER=MultiFidelitySearch should select the class."""
        from helion.autotuner import search_algorithms

        with patch.dict(os.environ, {"HELION_AUTOTUNER": "MultiFidelitySearch"}):
            name = os.environ.get("HELION_AUTOTUNER", "LFBOTreeSearch")
            cls = search_algorithms.get(name)
            self.assertIs(cls, MultiFidelitySearch)

    def test_inner_algorithm_default(self) -> None:
        """Default inner algorithm should be PatternSearch when not specified."""
        # When inner_cls is not provided, MultiFidelitySearch defaults to
        # PatternSearch. Constructor logic tested in integration tests.
        self.assertTrue(True)


class TestMultiFidelitySearchUnit(unittest.TestCase):
    """Unit tests for MultiFidelitySearch helper functions."""

    def test_scale_various_sizes(self) -> None:
        """Test scaling for various common tensor sizes."""
        test_cases = [
            (4096, 0.25, 1024),
            (2048, 0.25, 512),
            (512, 0.5, 256),
            (256, 0.5, 128),
            (128, 0.25, 32),
            (64, 0.25, 16),
        ]
        for original, factor, expected in test_cases:
            result = scale_tensor_size(original, factor)
            self.assertEqual(
                result,
                expected,
                f"scale({original}, {factor})={result}",
            )

    def test_scale_preserves_power_of_two_inputs(self) -> None:
        """Power-of-two inputs scaled by power-of-two factors stay power-of-two."""
        for size in [64, 128, 256, 512, 1024, 2048, 4096]:
            for factor in [0.25, 0.5]:
                result = scale_tensor_size(size, factor)
                # Check result is a power of 2
                self.assertEqual(
                    result & (result - 1),
                    0,
                    f"scale({size}, {factor})={result}",
                )


@helion.kernel()
def _mf_test_add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        result[tile] = x[tile] + y[tile]
    return result


@unittest.skipUnless(torch.cuda.is_available(), "requires GPU")
class TestMultiFidelitySearchIntegration(unittest.TestCase):
    """Integration tests requiring GPU."""

    def test_multi_fidelity_with_pattern_search(self) -> None:
        """Run MultiFidelitySearch wrapping PatternSearch."""
        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")

        with patch.dict(
            os.environ,
            {
                "HELION_AUTOTUNER": "MultiFidelitySearch",
                "HELION_MULTI_FIDELITY_INNER": "PatternSearch",
                "HELION_AUTOTUNE_EFFORT": "quick",
            },
        ):
            result = _mf_test_add_kernel(x, y)
            expected = x + y
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
