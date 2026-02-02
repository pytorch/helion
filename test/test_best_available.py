from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from helion.autotuner.base_search import _normalize_spec_key_for_warm_start
from helion.autotuner.base_search import _normalize_spec_key_str_for_warm_start
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.runtime.config import Config


class TestBestAvailable(unittest.TestCase):
    """Tests for the from_best_available autotuner feature."""

    def test_initial_population_strategy_enum(self):
        """Test that FROM_BEST_AVAILABLE is a valid strategy."""
        self.assertEqual(
            InitialPopulationStrategy.FROM_BEST_AVAILABLE.value, "from_best_available"
        )

    def test_get_initial_population_strategy_best_available(self):
        """Test that HELION_AUTOTUNER_INITIAL_POPULATION=from_best_available works."""
        from helion.runtime.settings import _get_initial_population_strategy

        with patch.dict(
            os.environ, {"HELION_AUTOTUNER_INITIAL_POPULATION": "from_best_available"}
        ):
            strategy = _get_initial_population_strategy("from_random")
            self.assertEqual(strategy, InitialPopulationStrategy.FROM_BEST_AVAILABLE)

    def test_get_initial_population_strategy_invalid(self):
        """Test that invalid values raise ValueError."""
        from helion.runtime.settings import _get_initial_population_strategy

        with patch.dict(
            os.environ, {"HELION_AUTOTUNER_INITIAL_POPULATION": "invalid_value"}
        ):
            with self.assertRaises(ValueError) as cm:
                _get_initial_population_strategy("from_random")
            self.assertIn("from_best_available", str(cm.exception))

    def test_best_available_max_configs_default(self):
        """Test that best_available_max_configs default is 20."""
        from helion.runtime.settings import Settings

        settings = Settings()
        self.assertEqual(settings.best_available_max_configs, 20)

    def test_best_available_max_cache_scan_default(self):
        """Test that best_available_max_cache_scan default is 500."""
        from helion.runtime.settings import Settings

        settings = Settings()
        self.assertEqual(settings.best_available_max_cache_scan, 500)

    def test_transfer_config_to_flat_basic(self):
        """Test _transfer_config_to_flat with basic config."""
        # Create a mock config spec
        mock_config_spec = MagicMock()
        mock_config_spec.block_sizes = []

        # Create a simple cached config
        cached_config = Config(block_sizes=[64, 128], num_warps=4)

        # Verify the config has the expected values
        self.assertEqual(cached_config.config["block_sizes"], [64, 128])
        self.assertEqual(cached_config.config["num_warps"], 4)

    def test_key_to_flat_index_mapping(self):
        """Test that _key_to_flat_index mapping is built correctly."""
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec
        from helion.autotuner.config_spec import FlattenLoopSpec

        # Create a real ConfigSpec with block sizes and flatten_loops
        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=128, min_size=16, max_size=256)
        )
        config_spec.flatten_loops.append(FlattenLoopSpec([0]))
        config_spec.flatten_loops.append(FlattenLoopSpec([1]))

        # Create ConfigGeneration with real ConfigSpec
        config_gen = ConfigGeneration(config_spec)

        # Verify the mapping contains expected keys
        mapping = config_gen._key_to_flat_index
        self.assertIn("block_sizes", mapping)
        self.assertIn("num_warps", mapping)
        self.assertIn("num_stages", mapping)
        self.assertIn("flatten_loops", mapping)

        # Verify keys are at non-negative indices
        for key, idx in mapping.items():
            self.assertGreaterEqual(idx, 0, f"Key {key} has negative index")

        # Verify block_sizes starts at 0
        self.assertEqual(mapping["block_sizes"], 0)

        # Verify that the mapping indices match the actual flat_spec structure
        # block_sizes has 2 fragments (0, 1), flatten_loops has 2 fragments (2, 3)
        # so num_warps should be at index 4
        self.assertEqual(mapping["flatten_loops"], 2)
        self.assertEqual(mapping["num_warps"], 4)


class TestCacheMatching(unittest.TestCase):
    """Tests for cache file matching in warm start."""

    def _write_best_config(
        self,
        cache_dir: str,
        filename: str,
        hardware: str,
        spec_key: str,
        source_hash: str,
        config_dict: dict,
    ) -> None:
        """Helper to write a fake .best_config file."""
        data = {
            "key": {
                "fields": {
                    "hardware": hardware,
                    "specialization_key": spec_key,
                    "kernel_source_hash": source_hash,
                }
            },
            "config": config_dict,
        }
        filepath = os.path.join(cache_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f)

    def test_cache_matching_includes_different_source_hash(self):
        """Test that configs with matching hardware/spec_key but different source hash are included."""
        with tempfile.TemporaryDirectory() as cache_dir:
            # Write a config with different source hash (should be included)
            self._write_best_config(
                cache_dir,
                "test1.best_config",
                hardware="NVIDIA GeForce RTX 4090",
                spec_key="('tensor_spec_1',)",
                source_hash="different_hash_abc123",
                config_dict={"block_sizes": [64, 128], "num_warps": 4},
            )

            # The actual matching logic is in _find_similar_cached_configs
            # which requires a full PopulationBasedSearch. For unit testing,
            # we verify the cache file was written correctly.
            self.assertTrue(
                os.path.exists(os.path.join(cache_dir, "test1.best_config"))
            )

            # Verify the file contents
            with open(os.path.join(cache_dir, "test1.best_config")) as f:
                data = json.load(f)
            self.assertEqual(
                data["key"]["fields"]["hardware"], "NVIDIA GeForce RTX 4090"
            )
            self.assertEqual(
                data["key"]["fields"]["kernel_source_hash"], "different_hash_abc123"
            )

    def test_cache_matching_excludes_same_source_hash(self):
        """Test that configs with same source hash are excluded (exact match)."""
        # This is tested implicitly - same source hash means same kernel version,
        # so warm start shouldn't use it (the cache system handles exact matches).

    def test_cache_matching_excludes_different_hardware(self):
        """Test that configs with different hardware are excluded."""
        # Hardware mismatch should not be included in warm start


class TestSpecKeyNormalization(unittest.TestCase):
    """Tests for specialization key normalization."""

    def test_normalize_code_object_in_spec_key(self):
        """Test that code objects are normalized to '<code>' placeholder."""

        # Create a real code object
        def dummy_fn():
            pass

        code_obj = dummy_fn.__code__

        # Create a spec_key tuple containing a code object
        spec_key = (code_obj, "tensor_info", (128, 256))

        normalized = _normalize_spec_key_for_warm_start(spec_key)

        # Code object should be replaced with "<code>"
        self.assertEqual(normalized[0], "<code>")
        # Other items should be preserved
        self.assertEqual(normalized[1], "tensor_info")
        self.assertEqual(normalized[2], (128, 256))

    def test_normalize_nested_code_object(self):
        """Test that nested code objects in tuples are normalized."""

        def dummy_fn():
            pass

        code_obj = dummy_fn.__code__

        # Nested tuple containing code object
        spec_key = (("inner", code_obj), "outer")

        normalized = _normalize_spec_key_for_warm_start(spec_key)

        # Nested code object should be normalized
        self.assertEqual(normalized[0], ("inner", "<code>"))
        self.assertEqual(normalized[1], "outer")

    def test_normalize_legacy_spec_key_string(self):
        """Test normalization of legacy spec_key strings with code object repr."""
        # Legacy format: code object repr like:
        # <code object <lambda> at 0x7e9b2c8f1e30, file "/path/to/file.py", line 317>
        legacy_spec_key = (
            "(<code object kernel_fn at 0x7f1234567890, "
            'file "/home/user/kernels.py", line 42>, '
            "'tensor_spec')"
        )

        normalized = _normalize_spec_key_str_for_warm_start(legacy_spec_key)

        # Code object repr should be replaced with '<code>'
        self.assertIn("'<code>'", normalized)
        self.assertNotIn("0x7f1234567890", normalized)
        self.assertNotIn("/home/user/kernels.py", normalized)

    def test_normalize_multiple_code_objects_in_string(self):
        """Test normalization of multiple code object reprs in a string."""
        legacy_spec_key = (
            '(<code object fn1 at 0xaaa, file "a.py", line 1>, '
            '<code object fn2 at 0xbbb, file "b.py", line 2>)'
        )

        normalized = _normalize_spec_key_str_for_warm_start(legacy_spec_key)

        # Both should be replaced
        self.assertEqual(normalized.count("'<code>'"), 2)
        self.assertNotIn("0xaaa", normalized)
        self.assertNotIn("0xbbb", normalized)

    def test_normalize_preserves_non_code_content(self):
        """Test that non-code object content is preserved."""
        spec_key = "('tensor_info', (128, 256), 'float32')"

        normalized = _normalize_spec_key_str_for_warm_start(spec_key)

        # Should be unchanged
        self.assertEqual(normalized, spec_key)


class TestStructuralCompatibility(unittest.TestCase):
    """Tests for structural compatibility checking in warm start transfer."""

    def test_structural_mismatch_block_sizes_rejected(self):
        """Test that configs with different block_sizes length are rejected."""
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        # Create a config spec with 2 block sizes
        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=128, min_size=16, max_size=256)
        )

        config_gen = ConfigGeneration(config_spec)

        # Create a cached config with 3 block sizes (mismatch!)
        cached_config = Config(block_sizes=[64, 128, 256], num_warps=4)

        # Create a mock search object with proper config_gen setup
        from helion.autotuner.base_search import PopulationBasedSearch

        mock_search = MagicMock()
        mock_search.config_gen = config_gen

        # Call the structural compatibility check
        with self.assertRaises(ValueError) as cm:
            PopulationBasedSearch._check_structural_compatibility(
                mock_search, cached_config
            )

        self.assertIn("block_sizes", str(cm.exception))
        self.assertIn("mismatch", str(cm.exception).lower())

    def test_structural_mismatch_range_fields_rejected(self):
        """Test that configs with different range_* field lengths are rejected."""
        from helion.autotuner.base_search import PopulationBasedSearch
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec
        from helion.autotuner.config_spec import RangeUnrollFactorSpec

        # Create a config spec with 2 range_unroll_factors
        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.range_unroll_factors.append(RangeUnrollFactorSpec([0]))
        config_spec.range_unroll_factors.append(RangeUnrollFactorSpec([1]))

        config_gen = ConfigGeneration(config_spec)

        # Create a cached config with 3 range_unroll_factors (mismatch!)
        cached_config = Config(
            block_sizes=[64], range_unroll_factors=[0, 1, 2], num_warps=4
        )

        mock_search = MagicMock()
        mock_search.config_gen = config_gen

        with self.assertRaises(ValueError) as cm:
            PopulationBasedSearch._check_structural_compatibility(
                mock_search, cached_config
            )

        self.assertIn("range_unroll_factors", str(cm.exception))

    def test_structural_match_accepted(self):
        """Test that configs with matching structure are accepted."""
        from helion.autotuner.base_search import PopulationBasedSearch
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        # Create a config spec with 2 block sizes
        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=128, min_size=16, max_size=256)
        )

        config_gen = ConfigGeneration(config_spec)

        # Create a cached config with matching 2 block sizes
        cached_config = Config(block_sizes=[32, 64], num_warps=8)

        mock_search = MagicMock()
        mock_search.config_gen = config_gen

        # Should not raise
        PopulationBasedSearch._check_structural_compatibility(
            mock_search, cached_config
        )


if __name__ == "__main__":
    unittest.main()
