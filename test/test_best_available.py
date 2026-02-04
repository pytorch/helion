from __future__ import annotations

import json
import operator
import os
import tempfile
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from helion._testing import DEVICE
from helion._testing import skipIfCpu
from helion.autotuner.base_search import _normalize_spec_key_for_best_available
from helion.autotuner.base_search import _normalize_spec_key_str_for_best_available
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
        """Test _transfer_config_to_flat actually transfers values correctly."""
        from helion.autotuner.base_search import PopulationBasedSearch
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=128, min_size=16, max_size=256)
        )

        config_gen = ConfigGeneration(config_spec)
        cached_config = Config(block_sizes=[32, 64], num_warps=8, num_stages=3)

        mock_search = MagicMock()
        mock_search.config_gen = config_gen

        flat = PopulationBasedSearch._transfer_config_to_flat(
            mock_search, cached_config
        )

        self.assertEqual(flat[0], 32)
        self.assertEqual(flat[1], 64)

        num_warps_idx = config_gen._key_to_flat_index["num_warps"]
        self.assertEqual(flat[num_warps_idx], 8)

        num_stages_idx = config_gen._key_to_flat_index["num_stages"]
        self.assertEqual(flat[num_stages_idx], 3)

    def test_key_to_flat_index_mapping(self):
        """Test that _key_to_flat_index mapping is built correctly."""
        from helion.autotuner.config_fragment import Category
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec
        from helion.autotuner.config_spec import FlattenLoopSpec

        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=128, min_size=16, max_size=256)
        )
        config_spec.flatten_loops.append(FlattenLoopSpec([0]))
        config_spec.flatten_loops.append(FlattenLoopSpec([1]))

        config_gen = ConfigGeneration(config_spec)

        mapping = config_gen._key_to_flat_index
        self.assertIn("block_sizes", mapping)
        self.assertIn("num_warps", mapping)
        self.assertIn("num_stages", mapping)
        self.assertIn("flatten_loops", mapping)

        for key, idx in mapping.items():
            self.assertGreaterEqual(idx, 0, f"Key {key} has negative index")
            self.assertLess(
                idx, len(config_gen.flat_spec), f"Key {key} index out of bounds"
            )

        first_block_size_idx = next(
            i
            for i, spec in enumerate(config_gen.flat_spec)
            if spec.category() == Category.BLOCK_SIZE
        )
        self.assertEqual(mapping["block_sizes"], first_block_size_idx)

        num_warps_idx = next(
            i
            for i, spec in enumerate(config_gen.flat_spec)
            if spec.category() == Category.NUM_WARPS
        )
        self.assertEqual(mapping["num_warps"], num_warps_idx)

    def test_key_to_flat_index_mapping_sync_with_flat_spec(self):
        """Test that _key_to_flat_index mapping stays in sync with flat_spec order."""
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec
        from helion.autotuner.config_spec import FlattenLoopSpec
        from helion.autotuner.config_spec import LoopOrderSpec
        from helion.autotuner.config_spec import RangeUnrollFactorSpec

        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=128, min_size=16, max_size=256)
        )
        config_spec.loop_orders.append(LoopOrderSpec([0, 1]))
        config_spec.flatten_loops.append(FlattenLoopSpec([0]))
        config_spec.range_unroll_factors.append(RangeUnrollFactorSpec([0]))

        config_gen = ConfigGeneration(config_spec)
        mapping = config_gen._key_to_flat_index

        for key, idx in mapping.items():
            self.assertLess(
                idx,
                len(config_gen.flat_spec),
                f"Key {key} has index {idx} but flat_spec has only {len(config_gen.flat_spec)} elements",
            )

        sorted_items = sorted(mapping.items(), key=operator.itemgetter(1))
        for i in range(len(sorted_items) - 1):
            key1, idx1 = sorted_items[i]
            key2, idx2 = sorted_items[i + 1]
            self.assertLess(
                idx1,
                idx2,
                f"Key {key1} (idx={idx1}) should come before {key2} (idx={idx2})",
            )

        default_config = config_spec.default_config()
        default_flat = config_gen.default_flat()

        if "block_sizes" in mapping:
            idx = mapping["block_sizes"]
            block_sizes = default_config.config.get("block_sizes", [])
            assert isinstance(block_sizes, list)
            for i, expected in enumerate(block_sizes):
                self.assertEqual(
                    default_flat[idx + i],
                    expected,
                    f"block_sizes[{i}] mismatch at flat index {idx + i}",
                )


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
        mtime_offset: float = 0,
    ) -> None:
        """Helper to write a fake .best_config file."""
        import time

        data = {
            "key": {
                "fields": {
                    "hardware": hardware,
                    "specialization_key": spec_key,
                    "kernel_source_hash": source_hash,
                }
            },
            "config": json.dumps(config_dict),
        }
        filepath = os.path.join(cache_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f)
        if mtime_offset != 0:
            current_time = time.time()
            os.utime(
                filepath, (current_time + mtime_offset, current_time + mtime_offset)
            )

    def test_find_similar_cached_configs_end_to_end(self):
        """End-to-end test for _find_similar_cached_configs."""
        from pathlib import Path

        from helion.autotuner.base_search import PopulationBasedSearch

        with tempfile.TemporaryDirectory() as cache_dir:
            self._write_best_config(
                cache_dir,
                "match1.best_config",
                hardware="NVIDIA GeForce RTX 4090",
                spec_key="('tensor_spec',)",
                source_hash="hash1",
                config_dict={"block_sizes": [64, 128], "num_warps": 4},
                mtime_offset=-10,
            )

            self._write_best_config(
                cache_dir,
                "match2.best_config",
                hardware="NVIDIA GeForce RTX 4090",
                spec_key="('tensor_spec',)",
                source_hash="hash2",
                config_dict={"block_sizes": [32, 64], "num_warps": 8},
                mtime_offset=0,
            )

            self._write_best_config(
                cache_dir,
                "diff_hw.best_config",
                hardware="NVIDIA A100",
                spec_key="('tensor_spec',)",
                source_hash="hash3",
                config_dict={"block_sizes": [128, 256], "num_warps": 4},
            )

            self._write_best_config(
                cache_dir,
                "diff_spec.best_config",
                hardware="NVIDIA GeForce RTX 4090",
                spec_key="('different_spec',)",
                source_hash="hash4",
                config_dict={"block_sizes": [16, 32], "num_warps": 2},
            )

            mock_search = MagicMock()
            mock_search.log = MagicMock()
            mock_search.log.debug = MagicMock()
            mock_search.log.warning = MagicMock()
            mock_search.settings = MagicMock()
            mock_search.settings.best_available_max_cache_scan = 500
            mock_search._get_cache_directory = MagicMock(return_value=Path(cache_dir))
            mock_search._get_current_hardware_and_specialization = MagicMock(
                return_value=("NVIDIA GeForce RTX 4090", "('tensor_spec',)")
            )

            configs = PopulationBasedSearch._find_similar_cached_configs(
                mock_search, max_configs=10
            )

            self.assertEqual(len(configs), 2)
            self.assertEqual(configs[0].config["block_sizes"], [32, 64])
            self.assertEqual(configs[1].config["block_sizes"], [64, 128])

    def test_find_similar_cached_configs_legacy_code_object(self):
        """Test that legacy cache files with code object repr are matched after normalization."""
        from pathlib import Path

        from helion.autotuner.base_search import PopulationBasedSearch

        with tempfile.TemporaryDirectory() as cache_dir:
            self._write_best_config(
                cache_dir,
                "legacy.best_config",
                hardware="NVIDIA GeForce RTX 4090",
                spec_key="(<code object fn at 0xabc, file \"test.py\", line 1>, 'tensor_spec')",
                source_hash="hash1",
                config_dict={"block_sizes": [48, 96], "num_warps": 4},
            )

            mock_search = MagicMock()
            mock_search.log = MagicMock()
            mock_search.log.debug = MagicMock()
            mock_search.log.warning = MagicMock()
            mock_search.settings = MagicMock()
            mock_search.settings.best_available_max_cache_scan = 500
            mock_search._get_cache_directory = MagicMock(return_value=Path(cache_dir))
            mock_search._get_current_hardware_and_specialization = MagicMock(
                return_value=("NVIDIA GeForce RTX 4090", "('<code>', 'tensor_spec')")
            )

            configs = PopulationBasedSearch._find_similar_cached_configs(
                mock_search, max_configs=10
            )

            self.assertEqual(len(configs), 1)
            self.assertEqual(configs[0].config["block_sizes"], [48, 96])

    def test_find_similar_cached_configs_respects_max_configs(self):
        """Test that _find_similar_cached_configs respects max_configs limit."""
        from pathlib import Path

        from helion.autotuner.base_search import PopulationBasedSearch

        with tempfile.TemporaryDirectory() as cache_dir:
            for i in range(5):
                self._write_best_config(
                    cache_dir,
                    f"match{i}.best_config",
                    hardware="NVIDIA GeForce RTX 4090",
                    spec_key="('tensor_spec',)",
                    source_hash=f"hash{i}",
                    config_dict={"block_sizes": [32 * (i + 1)], "num_warps": 4},
                    mtime_offset=-i,
                )

            mock_search = MagicMock()
            mock_search.log = MagicMock()
            mock_search.log.debug = MagicMock()
            mock_search.log.warning = MagicMock()
            mock_search.settings = MagicMock()
            mock_search.settings.best_available_max_cache_scan = 500
            mock_search._get_cache_directory = MagicMock(return_value=Path(cache_dir))
            mock_search._get_current_hardware_and_specialization = MagicMock(
                return_value=("NVIDIA GeForce RTX 4090", "('tensor_spec',)")
            )

            configs = PopulationBasedSearch._find_similar_cached_configs(
                mock_search, max_configs=2
            )

            self.assertEqual(len(configs), 2)

    def test_cache_matching_excludes_different_hardware(self):
        """Test that configs with different hardware are excluded."""
        from pathlib import Path

        from helion.autotuner.base_search import PopulationBasedSearch

        with tempfile.TemporaryDirectory() as cache_dir:
            self._write_best_config(
                cache_dir,
                "diff_hw.best_config",
                hardware="NVIDIA A100",
                spec_key="('tensor_spec',)",
                source_hash="hash1",
                config_dict={"block_sizes": [64], "num_warps": 4},
            )

            mock_search = MagicMock()
            mock_search.log = MagicMock()
            mock_search.log.debug = MagicMock()
            mock_search.log.warning = MagicMock()
            mock_search.settings = MagicMock()
            mock_search.settings.best_available_max_cache_scan = 500
            mock_search._get_cache_directory = MagicMock(return_value=Path(cache_dir))
            mock_search._get_current_hardware_and_specialization = MagicMock(
                return_value=("NVIDIA GeForce RTX 4090", "('tensor_spec',)")
            )

            configs = PopulationBasedSearch._find_similar_cached_configs(
                mock_search, max_configs=10
            )

            self.assertEqual(len(configs), 0)


class TestSpecKeyNormalization(unittest.TestCase):
    """Tests for specialization key normalization."""

    def test_normalize_code_object_in_spec_key(self):
        """Test that code objects are normalized to '<code>' placeholder."""

        def dummy_fn():
            pass

        code_obj = dummy_fn.__code__
        spec_key = (code_obj, "tensor_info", (128, 256))

        normalized = _normalize_spec_key_for_best_available(spec_key)

        self.assertEqual(normalized[0], "<code>")
        self.assertEqual(normalized[1], "tensor_info")
        self.assertEqual(normalized[2], (128, 256))

    def test_normalize_nested_code_object(self):
        """Test that nested code objects in tuples are normalized."""

        def dummy_fn():
            pass

        code_obj = dummy_fn.__code__
        spec_key = (("inner", code_obj), "outer")

        normalized = _normalize_spec_key_for_best_available(spec_key)

        self.assertEqual(normalized[0], ("inner", "<code>"))
        self.assertEqual(normalized[1], "outer")

    def test_normalize_legacy_spec_key_string(self):
        """Test normalization of legacy spec_key strings with code object repr."""
        legacy_spec_key = (
            "(<code object kernel_fn at 0x7f1234567890, "
            'file "/home/user/kernels.py", line 42>, '
            "'tensor_spec')"
        )

        normalized = _normalize_spec_key_str_for_best_available(legacy_spec_key)

        self.assertIn("'<code>'", normalized)
        self.assertNotIn("0x7f1234567890", normalized)
        self.assertNotIn("/home/user/kernels.py", normalized)

    def test_normalize_multiple_code_objects_in_string(self):
        """Test normalization of multiple code object reprs in a string."""
        legacy_spec_key = (
            '(<code object fn1 at 0xaaa, file "a.py", line 1>, '
            '<code object fn2 at 0xbbb, file "b.py", line 2>)'
        )

        normalized = _normalize_spec_key_str_for_best_available(legacy_spec_key)

        self.assertEqual(normalized.count("'<code>'"), 2)
        self.assertNotIn("0xaaa", normalized)
        self.assertNotIn("0xbbb", normalized)

    def test_normalize_preserves_non_code_content(self):
        """Test that non-code object content is preserved."""
        spec_key = "('tensor_info', (128, 256), 'float32')"

        normalized = _normalize_spec_key_str_for_best_available(spec_key)

        self.assertEqual(normalized, spec_key)


class TestStructuralCompatibility(unittest.TestCase):
    """Tests for structural compatibility checking in warm start transfer."""

    def test_structural_mismatch_block_sizes_rejected(self):
        """Test that configs with different block_sizes length are rejected."""
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=128, min_size=16, max_size=256)
        )

        config_gen = ConfigGeneration(config_spec)
        cached_config = Config(block_sizes=[64, 128, 256], num_warps=4)

        from helion.autotuner.base_search import PopulationBasedSearch

        mock_search = MagicMock()
        mock_search.config_gen = config_gen

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

        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.range_unroll_factors.append(RangeUnrollFactorSpec([0]))
        config_spec.range_unroll_factors.append(RangeUnrollFactorSpec([1]))

        config_gen = ConfigGeneration(config_spec)
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

        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=128, min_size=16, max_size=256)
        )

        config_gen = ConfigGeneration(config_spec)
        cached_config = Config(block_sizes=[32, 64], num_warps=8)

        mock_search = MagicMock()
        mock_search.config_gen = config_gen

        PopulationBasedSearch._check_structural_compatibility(
            mock_search, cached_config
        )


class TestHardwareDetection(unittest.TestCase):
    """Tests for hardware detection from kernel arguments."""

    @skipIfCpu("Requires GPU for hardware detection")
    def test_hardware_detection_direct_tensor(self):
        """Test hardware detection with a direct tensor argument."""
        import torch

        from helion.autotuner.base_search import PopulationBasedSearch

        tensor = torch.zeros(10, device=DEVICE)
        mock_search = MagicMock()
        mock_search.args = [tensor]
        mock_search.kernel = MagicMock()
        mock_search.kernel.kernel = MagicMock()
        mock_search.kernel.kernel.specialization_key = MagicMock(return_value=("spec",))

        hardware, _ = PopulationBasedSearch._get_current_hardware_and_specialization(
            mock_search
        )

        self.assertIsNotNone(hardware)
        self.assertIsInstance(hardware, str)
        self.assertGreater(len(hardware), 0)

    @skipIfCpu("Requires GPU for hardware detection")
    def test_hardware_detection_list_of_tensors(self):
        """Test hardware detection with list[0] tensor (matches cache behavior)."""
        import torch

        from helion.autotuner.base_search import PopulationBasedSearch

        tensor = torch.zeros(10, device=DEVICE)
        mock_search = MagicMock()
        mock_search.args = [[tensor, "other_data"], "scalar_arg"]
        mock_search.kernel = MagicMock()
        mock_search.kernel.kernel = MagicMock()
        mock_search.kernel.kernel.specialization_key = MagicMock(return_value=("spec",))

        hardware, _ = PopulationBasedSearch._get_current_hardware_and_specialization(
            mock_search
        )

        self.assertIsNotNone(hardware)
        self.assertIsInstance(hardware, str)
        self.assertGreater(len(hardware), 0)

    def test_hardware_detection_no_tensor(self):
        """Test hardware detection returns None when no tensor found."""
        from helion.autotuner.base_search import PopulationBasedSearch

        mock_search = MagicMock()
        mock_search.args = [1, 2, "string", [1, 2, 3]]
        mock_search.kernel = MagicMock()
        mock_search.kernel.kernel = MagicMock()
        mock_search.kernel.kernel.specialization_key = MagicMock(return_value=("spec",))

        hardware, _ = PopulationBasedSearch._get_current_hardware_and_specialization(
            mock_search
        )

        self.assertIsNone(hardware)


if __name__ == "__main__":
    unittest.main()
