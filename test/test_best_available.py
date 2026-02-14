from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from helion._testing import DEVICE
from helion._testing import skipIfCpu
from helion.autotuner.base_cache import LooseAutotuneCacheKey
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

        num_warps_idx = config_gen._key_to_flat_indices["num_warps"][0]
        self.assertEqual(flat[num_warps_idx], 8)

        num_stages_idx = config_gen._key_to_flat_indices["num_stages"][0]
        self.assertEqual(flat[num_stages_idx], 3)

    def test_key_to_flat_indices_mapping(self):
        """Test that _key_to_flat_indices mapping is built correctly."""
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

        mapping = config_gen._key_to_flat_indices
        self.assertIn("block_sizes", mapping)
        self.assertIn("num_warps", mapping)
        self.assertIn("num_stages", mapping)
        self.assertIn("flatten_loops", mapping)

        # block_sizes should have 2 indices, num_warps should have 1
        self.assertEqual(len(mapping["block_sizes"]), 2)
        self.assertEqual(len(mapping["num_warps"]), 1)
        self.assertEqual(len(mapping["flatten_loops"]), 2)

        for key, indices in mapping.items():
            for idx in indices:
                self.assertGreaterEqual(idx, 0, f"Key {key} has negative index")
                self.assertLess(
                    idx, len(config_gen.flat_spec), f"Key {key} index out of bounds"
                )

        first_block_size_idx = next(
            i
            for i, spec in enumerate(config_gen.flat_spec)
            if spec.category() == Category.BLOCK_SIZE
        )
        self.assertEqual(mapping["block_sizes"][0], first_block_size_idx)

        num_warps_idx = next(
            i
            for i, spec in enumerate(config_gen.flat_spec)
            if spec.category() == Category.NUM_WARPS
        )
        self.assertEqual(mapping["num_warps"][0], num_warps_idx)

    def test_key_to_flat_indices_mapping_sync_with_flat_spec(self):
        """Test that _key_to_flat_indices mapping stays in sync with flat_spec order."""
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
        mapping = config_gen._key_to_flat_indices

        for key, indices in mapping.items():
            for idx in indices:
                self.assertLess(
                    idx,
                    len(config_gen.flat_spec),
                    f"Key {key} has index {idx} but flat_spec has only {len(config_gen.flat_spec)} elements",
                )

        # Verify indices are in ascending order across keys
        all_indices = [idx for indices in mapping.values() for idx in indices]
        self.assertEqual(all_indices, sorted(all_indices))

        default_config = config_spec.default_config()
        default_flat = config_gen.default_flat()

        if "block_sizes" in mapping:
            indices = mapping["block_sizes"]
            block_sizes = default_config.config.get("block_sizes", [])
            assert isinstance(block_sizes, list)
            for i, (idx, expected) in enumerate(zip(indices, block_sizes, strict=True)):
                self.assertEqual(
                    default_flat[idx],
                    expected,
                    f"block_sizes[{i}] mismatch at flat index {idx}",
                )

    def test_flat_key_layout_total_matches_flat_spec(self):
        """Test that flat_key_layout() total count equals flat_spec length."""
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec
        from helion.autotuner.config_spec import FlattenLoopSpec
        from helion.autotuner.config_spec import LoopOrderSpec

        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=128, min_size=16, max_size=256)
        )
        config_spec.loop_orders.append(LoopOrderSpec([0, 1]))
        config_spec.flatten_loops.append(FlattenLoopSpec([0]))

        layout = config_spec.flat_key_layout()
        layout_total = sum(count for _, count in layout)

        config_gen = ConfigGeneration(config_spec)
        self.assertEqual(
            layout_total,
            len(config_gen.flat_spec),
            f"flat_key_layout total {layout_total} != flat_spec length {len(config_gen.flat_spec)}",
        )

    def test_flatten_unflatten_roundtrip(self):
        """Test that flatten(unflatten(flat)) == flat for default config."""
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec
        from helion.autotuner.config_spec import FlattenLoopSpec
        from helion.autotuner.config_spec import LoopOrderSpec

        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=128, min_size=16, max_size=256)
        )
        config_spec.loop_orders.append(LoopOrderSpec([0, 1]))
        config_spec.flatten_loops.append(FlattenLoopSpec([0]))

        config_gen = ConfigGeneration(config_spec)
        default_flat = config_gen.default_flat()
        config = config_gen.unflatten(default_flat)
        roundtripped = config_gen.flatten(config)

        self.assertEqual(
            roundtripped,
            default_flat,
            "flatten(unflatten(default_flat)) != default_flat",
        )

    def test_flatten_with_dropped_keys(self):
        """Regression: normalize() drops num_sm_multiplier for non-persistent pid_types.

        flat_spec still has an entry for it (flat_config calls fn() before
        normalize drops the key).  flatten() must not shift later indices
        when a key is present in flat_spec but absent from config.config.
        """
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )

        config_gen = ConfigGeneration(config_spec)

        # Build a config with pid_type="flat" — normalize drops num_sm_multiplier
        config = Config(
            block_sizes=[64],
            num_warps=4,
            num_stages=2,
            pid_type="flat",
        )
        config_spec.normalize(config.config)

        # num_sm_multiplier should NOT be in the config after normalize
        self.assertNotIn("num_sm_multiplier", config.config)

        # flatten must not crash or mis-align indices
        flat = config_gen.flatten(config)
        self.assertEqual(len(flat), len(config_gen.flat_spec))

        # Roundtrip: unflatten should produce a valid config
        restored = config_gen.unflatten(flat)
        self.assertIn("block_sizes", restored.config)
        self.assertEqual(restored.config["block_sizes"], [64])

    def test_flatten_with_empty_list_keys(self):
        """Regression: normalize() can re-add empty-list keys.

        config_spec.normalize() unconditionally writes back
        ``config["range_warp_specializes"] = range_warp_specializes``
        (see config_spec.py normalize(), near the end of the method)
        even when the value is ``[]``.  Because the BlockIdSequence for
        range_warp_specialize may be empty, flat_key_layout() won't
        include it, yet config.config will contain the key.
        flatten() must skip such keys without crashing.
        """
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )

        config_gen = ConfigGeneration(config_spec)
        default_config = config_spec.default_config()

        # Manually add a spurious empty-list key (simulates normalize re-adding it)
        default_config.config["range_warp_specializes"] = []

        flat = config_gen.flatten(default_config)
        self.assertEqual(len(flat), len(config_gen.flat_spec))

    @patch("helion.autotuner.config_spec.use_tileir_tunables", return_value=True)
    @patch("helion.autotuner.config_spec.supports_maxnreg", return_value=False)
    def test_flatten_unflatten_with_tileir_duplicate_keys(
        self, _mock_maxnreg, _mock_tileir
    ):
        """TileIR yields num_warps/num_stages twice in _scalar_flat_fragments().

        The second occurrence overwrites the first in _build_key_index_mapping(),
        matching the dict.update() semantics in flat_config().
        flatten/unflatten must roundtrip correctly despite the duplicate entries.
        """
        from helion.autotuner.config_fragment import PowerOfTwoFragment
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec

        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        # Provide num_ctas/occupancy fragments that tileir expects
        config_spec.num_ctas = PowerOfTwoFragment(1, 2, 1)
        config_spec.occupancy = PowerOfTwoFragment(1, 8, 1)

        config_gen = ConfigGeneration(config_spec)

        # flat_key_layout should contain num_warps and num_stages twice
        layout_keys = [key for key, _ in config_spec.flat_key_layout()]
        self.assertEqual(layout_keys.count("num_warps"), 2)
        self.assertEqual(layout_keys.count("num_stages"), 2)

        # Roundtrip: default_flat -> unflatten -> flatten should be stable
        default_flat = config_gen.default_flat()
        config = config_gen.unflatten(default_flat)
        roundtripped = config_gen.flatten(config)
        self.assertEqual(len(roundtripped), len(default_flat))

        # The tileir-specific keys should be present in the config
        self.assertIn("num_ctas", config.config)
        self.assertIn("occupancy", config.config)


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
            mock_search.settings = MagicMock()
            mock_search.settings.best_available_max_cache_scan = 500
            mock_search._get_current_hardware_and_specialization = MagicMock(
                return_value=("NVIDIA GeForce RTX 4090", "('tensor_spec',)")
            )

            with patch(
                "helion.autotuner.local_cache.get_helion_cache_dir",
                return_value=Path(cache_dir),
            ):
                configs = PopulationBasedSearch._find_similar_cached_configs(
                    mock_search, max_configs=10
                )

            self.assertEqual(len(configs), 2)
            self.assertEqual(configs[0].config["block_sizes"], [32, 64])
            self.assertEqual(configs[1].config["block_sizes"], [64, 128])

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
            mock_search.settings = MagicMock()
            mock_search.settings.best_available_max_cache_scan = 500
            mock_search._get_current_hardware_and_specialization = MagicMock(
                return_value=("NVIDIA GeForce RTX 4090", "('tensor_spec',)")
            )

            with patch(
                "helion.autotuner.local_cache.get_helion_cache_dir",
                return_value=Path(cache_dir),
            ):
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
            mock_search.settings = MagicMock()
            mock_search.settings.best_available_max_cache_scan = 500
            mock_search._get_current_hardware_and_specialization = MagicMock(
                return_value=("NVIDIA GeForce RTX 4090", "('tensor_spec',)")
            )

            with patch(
                "helion.autotuner.local_cache.get_helion_cache_dir",
                return_value=Path(cache_dir),
            ):
                configs = PopulationBasedSearch._find_similar_cached_configs(
                    mock_search, max_configs=10
                )

            self.assertEqual(len(configs), 0)

    def test_cache_matching_with_code_object_in_spec_key(self):
        """End-to-end: cached entry with raw code object repr matches current
        key that has a different memory address for the same function.

        This simulates the matmul-with-activation-lambda scenario where
        put() stores the raw str() of specialization_key (containing
        <code object ...at 0xADDR>) and _find_similar_cached_configs
        must normalize both sides to find the match.
        """
        from pathlib import Path

        from helion.autotuner.base_search import PopulationBasedSearch
        from helion.autotuner.base_search import _normalize_spec_key_str

        # What put() stored: raw str() with a specific memory address
        stored_spec_key = (
            "((torch.float16, 'cuda', (2, 2), False, frozenset()), "
            "(torch.float16, 'cuda', (2, 2), False, frozenset()), "
            "(<code object addmm_epilogue at 0x7e56e22f1a70, "
            'file "/home/user/matmul.py", line 244>, '
            "<class 'float'>, <class 'float'>, "
            "(torch.float16, 'cuda', (2, 2), False, frozenset())))"
        )

        # What the current process computes: same function, different address
        current_raw_spec_key = (
            "((torch.float16, 'cuda', (2, 2), False, frozenset()), "
            "(torch.float16, 'cuda', (2, 2), False, frozenset()), "
            "(<code object addmm_epilogue at 0x7fff98761234, "
            'file "/home/user/matmul.py", line 244>, '
            "<class 'float'>, <class 'float'>, "
            "(torch.float16, 'cuda', (2, 2), False, frozenset())))"
        )
        # _get_current_hardware_and_specialization applies normalization
        current_normalized = _normalize_spec_key_str(current_raw_spec_key)

        with tempfile.TemporaryDirectory() as cache_dir:
            self._write_best_config(
                cache_dir,
                "matmul_activation.best_config",
                hardware="NVIDIA GeForce RTX 5090",
                spec_key=stored_spec_key,
                source_hash="hash1",
                config_dict={"block_sizes": [64, 128], "num_warps": 4},
            )

            # Also write a cache entry with different closure values —
            # should NOT match even after stripping the code object
            stored_different_closure = (
                "((torch.float32, 'cuda', (4, 4), False, frozenset()), "
                "(torch.float32, 'cuda', (4, 4), False, frozenset()), "
                "(<code object addmm_epilogue at 0x7e56e22f1a70, "
                'file "/home/user/matmul.py", line 244>, '
                "<class 'int'>, <class 'int'>, "
                "(torch.float32, 'cuda', (4, 4), False, frozenset())))"
            )
            self._write_best_config(
                cache_dir,
                "matmul_different_closure.best_config",
                hardware="NVIDIA GeForce RTX 5090",
                spec_key=stored_different_closure,
                source_hash="hash2",
                config_dict={"block_sizes": [32, 64], "num_warps": 8},
            )

            mock_search = MagicMock()
            mock_search.settings = MagicMock()
            mock_search.settings.best_available_max_cache_scan = 500
            mock_search._get_current_hardware_and_specialization = MagicMock(
                return_value=("NVIDIA GeForce RTX 5090", current_normalized)
            )

            with patch(
                "helion.autotuner.local_cache.get_helion_cache_dir",
                return_value=Path(cache_dir),
            ):
                configs = PopulationBasedSearch._find_similar_cached_configs(
                    mock_search, max_configs=10
                )

            # Only the matching closure entry should be returned
            self.assertEqual(len(configs), 1)
            self.assertEqual(configs[0].config["block_sizes"], [64, 128])


class TestIterCacheEntries(unittest.TestCase):
    """Tests for the iter_cache_entries() module-level API in local_cache."""

    def _write_cache_file(
        self,
        cache_dir: str,
        filename: str,
        hardware: str,
        spec_key: str,
        config_dict: dict,
        mtime_offset: float = 0,
    ) -> None:
        import time

        data = {
            "key": {
                "fields": {
                    "hardware": hardware,
                    "specialization_key": spec_key,
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

    def test_newest_first_ordering(self):
        """Test that entries are yielded newest first."""
        from pathlib import Path

        from helion.autotuner.local_cache import iter_cache_entries

        with tempfile.TemporaryDirectory() as cache_dir:
            self._write_cache_file(
                cache_dir,
                "old.best_config",
                "HW",
                "spec",
                {"block_sizes": [32], "num_warps": 4},
                mtime_offset=-10,
            )
            self._write_cache_file(
                cache_dir,
                "new.best_config",
                "HW",
                "spec",
                {"block_sizes": [64], "num_warps": 4},
                mtime_offset=0,
            )

            entries = list(iter_cache_entries(Path(cache_dir)))
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].config.config["block_sizes"], [64])
            self.assertEqual(entries[1].config.config["block_sizes"], [32])

    def test_corrupt_json_skipped(self):
        """Test that corrupt files are silently skipped."""
        from pathlib import Path

        from helion.autotuner.local_cache import iter_cache_entries

        with tempfile.TemporaryDirectory() as cache_dir:
            # Write a valid file
            self._write_cache_file(
                cache_dir,
                "valid.best_config",
                "HW",
                "spec",
                {"block_sizes": [64], "num_warps": 4},
            )
            # Write a corrupt file
            corrupt_path = os.path.join(cache_dir, "corrupt.best_config")
            Path(corrupt_path).write_text("not valid json {{{")

            entries = list(iter_cache_entries(Path(cache_dir)))
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].hardware, "HW")

    def test_max_scan_limits_results(self):
        """Test that max_scan limits how many files are parsed."""
        from pathlib import Path

        from helion.autotuner.local_cache import iter_cache_entries

        with tempfile.TemporaryDirectory() as cache_dir:
            for i in range(5):
                self._write_cache_file(
                    cache_dir,
                    f"entry{i}.best_config",
                    "HW",
                    "spec",
                    {"block_sizes": [32 * (i + 1)], "num_warps": 4},
                    mtime_offset=-i,
                )

            entries = list(iter_cache_entries(Path(cache_dir), max_scan=2))
            self.assertEqual(len(entries), 2)

    def test_nonexistent_directory(self):
        """Test that a nonexistent directory yields nothing."""
        from pathlib import Path

        from helion.autotuner.local_cache import iter_cache_entries

        entries = list(iter_cache_entries(Path("/nonexistent/path")))
        self.assertEqual(len(entries), 0)

    def test_fields_parsed_correctly(self):
        """Test that hardware, specialization_key, and config are correctly parsed."""
        from pathlib import Path

        from helion.autotuner.local_cache import iter_cache_entries

        with tempfile.TemporaryDirectory() as cache_dir:
            self._write_cache_file(
                cache_dir,
                "entry.best_config",
                hardware="NVIDIA RTX 5090",
                spec_key="('my_spec',)",
                config_dict={"block_sizes": [128], "num_warps": 8},
            )

            entries = list(iter_cache_entries(Path(cache_dir)))
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].hardware, "NVIDIA RTX 5090")
            self.assertEqual(entries[0].specialization_key, "('my_spec',)")
            self.assertEqual(entries[0].config.config["block_sizes"], [128])
            self.assertEqual(entries[0].config.config["num_warps"], 8)


class TestSpecKeyNormalization(unittest.TestCase):
    """Tests for specialization key normalization via _normalize_spec_key_str()."""

    def test_code_object_repr_stripped(self):
        """Test that code object reprs are stripped from strings."""
        from helion.autotuner.base_search import _normalize_spec_key_str

        raw = "(<code object <lambda> at 0x7cdd123, file \"foo.py\", line 322>, (torch.float16, 'cuda'))"
        result = _normalize_spec_key_str(raw)

        self.assertNotIn("<code object", result)
        self.assertIn("torch.float16", result)
        self.assertIn("'cuda'", result)

    def test_nested_code_objects_stripped(self):
        """Test that nested code objects in tuples are stripped."""
        from helion.autotuner.base_search import _normalize_spec_key_str

        raw = "((<code object helper at 0xabc, file \"x.py\", line 10>, 'inner'), 'outer')"
        result = _normalize_spec_key_str(raw)

        self.assertNotIn("<code object", result)
        self.assertIn("'inner'", result)
        self.assertIn("'outer'", result)

    def test_tensor_closure_info_preserved(self):
        """Test that tensor/closure information is preserved."""
        from helion.autotuner.base_search import _normalize_spec_key_str

        raw = "((torch.float16, 'cuda', (1024,), (1,), frozenset()),)"
        result = _normalize_spec_key_str(raw)

        self.assertEqual(result, raw)

    def test_end_to_end_matching(self):
        """Test that a stored cache entry with raw code object repr matches
        a current key computed with a different address."""
        from helion.autotuner.base_search import _normalize_spec_key_str

        # Simulated stored cache entry (raw str() with address)
        stored = "(<code object <lambda> at 0x7cdd1234abcd, file \"matmul.py\", line 42>, (torch.float16, 'cuda', (1024,), (1,), frozenset()))"
        # Simulated current key (different address)
        current = "(<code object <lambda> at 0x7fff9876fedc, file \"matmul.py\", line 42>, (torch.float16, 'cuda', (1024,), (1,), frozenset()))"

        self.assertEqual(
            _normalize_spec_key_str(stored),
            _normalize_spec_key_str(current),
        )

    def test_put_stores_raw_spec_key(self):
        """Test that put() stores the raw specialization_key (with code object reprs)."""
        from pathlib import Path

        from helion.autotuner.local_cache import LocalAutotuneCache

        def dummy_fn():
            pass

        code_obj = dummy_fn.__code__

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test.best_config"

            key = LooseAutotuneCacheKey(
                specialization_key=(code_obj, "tensor_spec"),
                extra_results=(),
                kernel_source_hash="abc123",
                hardware="test_hw",
                runtime_name="1.0",
            )

            mock_cache = MagicMock()
            mock_cache.key = key
            mock_cache._get_local_cache_path.return_value = cache_path

            LocalAutotuneCache.put(mock_cache, Config(block_sizes=[64], num_warps=4))

            data = json.loads(cache_path.read_text())
            spec_key_str = data["key"]["fields"]["specialization_key"]

            # put() stores raw str(v), so code object reprs are present
            self.assertIn("<code object", spec_key_str)
            self.assertIn("tensor_spec", spec_key_str)


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

    def test_hardware_detection_generic_adapter_no_inner_kernel(self):
        """Test that generic adapters without a .kernel attribute return None spec_key."""
        from helion.autotuner.base_search import PopulationBasedSearch

        mock_search = MagicMock()
        mock_search.args = [1, 2, "string"]
        mock_search.kernel = MagicMock(spec=[])  # no .kernel attribute

        hardware, spec_key = (
            PopulationBasedSearch._get_current_hardware_and_specialization(mock_search)
        )

        self.assertIsNone(spec_key)


if __name__ == "__main__":
    unittest.main()
