from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

from helion._testing import DEVICE
from helion._testing import skipIfCpu
from helion.autotuner.base_cache import LooseAutotuneCacheKey
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import _normalize_spec_key_str
from helion.autotuner.config_fragment import Category
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import FlattenLoopSpec
from helion.autotuner.config_spec import LoopOrderSpec
from helion.autotuner.config_spec import RangeUnrollFactorSpec
from helion.autotuner.local_cache import CacheEntry
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.local_cache import iter_cache_entries
from helion.autotuner.pattern_search import InitialPopulationStrategy
from helion.runtime.config import Config
from helion.runtime.settings import Settings
from helion.runtime.settings import _get_initial_population_strategy


class TestBestAvailable(unittest.TestCase):
    """Tests for the from_best_available autotuner feature."""

    def test_initial_population_strategy_enum(self):
        """Test that FROM_BEST_AVAILABLE is a valid strategy."""
        self.assertEqual(
            InitialPopulationStrategy.FROM_BEST_AVAILABLE.value, "from_best_available"
        )

    def test_get_initial_population_strategy_best_available(self):
        """Test that HELION_AUTOTUNER_INITIAL_POPULATION=from_best_available works."""
        with patch.dict(
            os.environ, {"HELION_AUTOTUNER_INITIAL_POPULATION": "from_best_available"}
        ):
            strategy = _get_initial_population_strategy("from_random")
            self.assertEqual(strategy, InitialPopulationStrategy.FROM_BEST_AVAILABLE)

    def test_get_initial_population_strategy_invalid(self):
        """Test that invalid values raise ValueError."""
        with patch.dict(
            os.environ, {"HELION_AUTOTUNER_INITIAL_POPULATION": "invalid_value"}
        ):
            with self.assertRaises(ValueError) as cm:
                _get_initial_population_strategy("from_random")
            self.assertIn("from_best_available", str(cm.exception))

    def test_best_available_max_configs_default(self):
        """Test that best_available_max_configs default is 20."""
        settings = Settings()
        self.assertEqual(settings.best_available_max_configs, 20)

    def test_best_available_max_cache_scan_default(self):
        """Test that best_available_max_cache_scan default is 500."""
        settings = Settings()
        self.assertEqual(settings.best_available_max_cache_scan, 500)

    def test_transfer_config_to_flat_basic(self):
        """Test _transfer_config_to_flat actually transfers values correctly."""
        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=128, min_size=16, max_size=256)
        )

        config_gen = ConfigGeneration(config_spec)
        cached_config = Config(block_sizes=[32, 64], num_warps=8, num_stages=3)
        entry = CacheEntry(
            hardware="test",
            specialization_key="test",
            config=cached_config,
        )

        mock_search = MagicMock()
        mock_search.config_gen = config_gen

        flat = PopulationBasedSearch._transfer_config_to_flat(
            mock_search, entry
        )

        self.assertEqual(flat[0], 32)
        self.assertEqual(flat[1], 64)

        num_warps_idx = config_gen._key_to_flat_indices["num_warps"][0]
        self.assertEqual(flat[num_warps_idx], 8)

        num_stages_idx = config_gen._key_to_flat_indices["num_stages"][0]
        self.assertEqual(flat[num_stages_idx], 3)

    def test_transfer_config_to_flat_uses_stored_flat(self):
        """Test _transfer_config_to_flat uses stored flat_config when available."""
        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )

        config_gen = ConfigGeneration(config_spec)
        stored_flat = tuple(config_gen.default_flat())
        entry = CacheEntry(
            hardware="test",
            specialization_key="test",
            config=Config(block_sizes=[64], num_warps=4),
            flat_config=stored_flat,
        )

        mock_search = MagicMock()
        mock_search.config_gen = config_gen

        flat = PopulationBasedSearch._transfer_config_to_flat(
            mock_search, entry
        )

        self.assertEqual(flat, list(stored_flat))

    def test_key_to_flat_indices_mapping(self):
        """Test that _key_to_flat_indices mapping is built correctly."""
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

        num_warps_indices = [
            i
            for i, spec in enumerate(config_gen.flat_spec)
            if spec.category() == Category.NUM_WARPS
        ]
        self.assertEqual(mapping["num_warps"][0], num_warps_indices[-1])

    def test_key_to_flat_indices_mapping_sync_with_flat_spec(self):
        """Test that _key_to_flat_indices mapping stays in sync with flat_spec order."""
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

        for key, indices in mapping.items():
            self.assertEqual(
                indices,
                sorted(indices),
                f"Indices for key {key!r} are not in ascending order: {indices}",
            )

        # Verify all mapped indices are unique (no two keys share an index)
        all_indices = [idx for indices in mapping.values() for idx in indices]
        self.assertEqual(len(all_indices), len(set(all_indices)))

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

    @patch("helion.autotuner.config_spec._get_backend", return_value="tileir")
    @patch("helion.autotuner.config_spec.supports_maxnreg", return_value=False)
    def test_flatten_unflatten_with_tileir_duplicate_keys(
        self, _mock_maxnreg, _mock_tileir
    ):
        """TileIR yields num_warps/num_stages twice in _scalar_flat_fragments().

        The second occurrence overwrites the first in _key_to_flat_indices,
        matching the dict.update() semantics in flat_config().
        flatten/unflatten must roundtrip correctly despite the duplicate entries.
        """
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
            # structural_fingerprint returns a dummy — no config_spec_hash in
            # old cache files, so all entries pass the fingerprint filter.
            mock_search.config_spec = MagicMock()
            mock_search.config_spec.structural_fingerprint = MagicMock(
                return_value=(("block_sizes", 2, 1, 1),)
            )

            with patch(
                "helion.autotuner.local_cache.get_helion_cache_dir",
                return_value=Path(cache_dir),
            ):
                entries = PopulationBasedSearch._find_similar_cached_configs(
                    mock_search, max_configs=10
                )

            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].config.config["block_sizes"], [32, 64])
            self.assertEqual(entries[1].config.config["block_sizes"], [64, 128])

    def test_find_similar_cached_configs_respects_max_configs(self):
        """Test that _find_similar_cached_configs respects max_configs limit."""
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
            mock_search.config_spec = MagicMock()
            mock_search.config_spec.structural_fingerprint = MagicMock(
                return_value=(("block_sizes", 1, 1),)
            )

            with patch(
                "helion.autotuner.local_cache.get_helion_cache_dir",
                return_value=Path(cache_dir),
            ):
                entries = PopulationBasedSearch._find_similar_cached_configs(
                    mock_search, max_configs=2
                )

            self.assertEqual(len(entries), 2)

    def test_cache_matching_excludes_different_hardware(self):
        """Test that configs with different hardware are excluded."""
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
            mock_search.config_spec = MagicMock()
            mock_search.config_spec.structural_fingerprint = MagicMock(
                return_value=(("block_sizes", 1, 1),)
            )

            with patch(
                "helion.autotuner.local_cache.get_helion_cache_dir",
                return_value=Path(cache_dir),
            ):
                entries = PopulationBasedSearch._find_similar_cached_configs(
                    mock_search, max_configs=10
                )

            self.assertEqual(len(entries), 0)

    def test_cache_matching_with_code_object_in_spec_key(self):
        """End-to-end: cached entry with raw code object repr matches current
        key that has a different memory address for the same function.

        This simulates the matmul-with-activation-lambda scenario where
        put() stores the raw str() of specialization_key (containing
        <code object ...at 0xADDR>) and _find_similar_cached_configs
        must normalize both sides to find the match.
        """
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
            mock_search.config_spec = MagicMock()
            mock_search.config_spec.structural_fingerprint = MagicMock(
                return_value=(("block_sizes", 2, 1, 1),)
            )

            with patch(
                "helion.autotuner.local_cache.get_helion_cache_dir",
                return_value=Path(cache_dir),
            ):
                entries = PopulationBasedSearch._find_similar_cached_configs(
                    mock_search, max_configs=10
                )

            # Only the matching closure entry should be returned
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].config.config["block_sizes"], [64, 128])


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
        entries = list(iter_cache_entries(Path("/nonexistent/path")))
        self.assertEqual(len(entries), 0)

    def test_fields_parsed_correctly(self):
        """Test that hardware, specialization_key, and config are correctly parsed."""
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
        raw = "(<code object <lambda> at 0x7cdd123, file \"foo.py\", line 322>, (torch.float16, 'cuda'))"
        result = _normalize_spec_key_str(raw)

        self.assertNotIn("<code object", result)
        self.assertIn("torch.float16", result)
        self.assertIn("'cuda'", result)

    def test_nested_code_objects_stripped(self):
        """Test that nested code objects in tuples are stripped."""
        raw = "((<code object helper at 0xabc, file \"x.py\", line 10>, 'inner'), 'outer')"
        result = _normalize_spec_key_str(raw)

        self.assertNotIn("<code object", result)
        self.assertIn("'inner'", result)
        self.assertIn("'outer'", result)

    def test_tensor_closure_info_preserved(self):
        """Test that tensor/closure information is preserved."""
        raw = "((torch.float16, 'cuda', (1024,), (1,), frozenset()),)"
        result = _normalize_spec_key_str(raw)

        self.assertEqual(result, raw)

    def test_end_to_end_matching(self):
        """Test that a stored cache entry with raw code object repr matches
        a current key computed with a different address."""
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
            mock_cache.kernel.backend_cache_key.return_value = None
            # Make flatten() return a JSON-serializable list
            mock_cache.kernel.config_spec.create_config_generation.return_value.flatten.return_value = [64, 4]

            LocalAutotuneCache.put(mock_cache, Config(block_sizes=[64], num_warps=4))

            data = json.loads(cache_path.read_text())
            spec_key_str = data["key"]["fields"]["specialization_key"]

            # put() stores raw str(v), so code object reprs are present
            self.assertIn("<code object", spec_key_str)
            self.assertIn("tensor_spec", spec_key_str)


class TestStructuralFingerprint(unittest.TestCase):
    """Tests for ConfigSpec.structural_fingerprint()."""

    def test_different_block_sizes_count(self):
        """ConfigSpecs with different block_sizes counts have different fingerprints."""
        spec_2 = ConfigSpec()
        spec_2.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
        spec_2.block_sizes.append(BlockSizeSpec(block_id=1, size_hint=128))

        spec_3 = ConfigSpec()
        spec_3.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
        spec_3.block_sizes.append(BlockSizeSpec(block_id=1, size_hint=128))
        spec_3.block_sizes.append(BlockSizeSpec(block_id=2, size_hint=256))

        self.assertNotEqual(
            spec_2.structural_fingerprint(), spec_3.structural_fingerprint()
        )

    def test_same_structure_same_fingerprint(self):
        """ConfigSpecs with same structure have the same fingerprint."""
        spec_a = ConfigSpec()
        spec_a.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
        spec_a.block_sizes.append(BlockSizeSpec(block_id=1, size_hint=128))

        spec_b = ConfigSpec()
        spec_b.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=32, min_size=8, max_size=512)
        )
        spec_b.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=256, min_size=16, max_size=1024)
        )

        self.assertEqual(
            spec_a.structural_fingerprint(), spec_b.structural_fingerprint()
        )

    def test_different_range_fields_count(self):
        """ConfigSpecs with different range field counts have different fingerprints."""
        spec_a = ConfigSpec()
        spec_a.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
        spec_a.range_unroll_factors.append(RangeUnrollFactorSpec([0]))

        spec_b = ConfigSpec()
        spec_b.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
        spec_b.range_unroll_factors.append(RangeUnrollFactorSpec([0]))
        spec_b.range_unroll_factors.append(RangeUnrollFactorSpec([1]))

        self.assertNotEqual(
            spec_a.structural_fingerprint(), spec_b.structural_fingerprint()
        )

    def test_loop_orders_block_ids_length(self):
        """Loop orders with different block_ids lengths produce different fingerprints."""
        spec_a = ConfigSpec()
        spec_a.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
        spec_a.loop_orders.append(LoopOrderSpec([0, 1]))

        spec_b = ConfigSpec()
        spec_b.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
        spec_b.loop_orders.append(LoopOrderSpec([0, 1, 2]))

        self.assertNotEqual(
            spec_a.structural_fingerprint(), spec_b.structural_fingerprint()
        )

    def test_fingerprint_is_hashable(self):
        """structural_fingerprint returns a hashable value."""
        spec = ConfigSpec()
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=64))
        fp = spec.structural_fingerprint()
        # Should be usable as a dict key or set member
        {fp: True}
        {fp}


class TestHardwareDetection(unittest.TestCase):
    """Tests for hardware detection from kernel arguments."""

    @skipIfCpu("Requires GPU for hardware detection")
    def test_hardware_detection_direct_tensor(self):
        """Test hardware detection with a direct tensor argument."""
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
        mock_search = MagicMock()
        mock_search.args = [1, 2, "string"]
        mock_search.kernel = MagicMock(spec=[])  # no .kernel attribute

        hardware, spec_key = (
            PopulationBasedSearch._get_current_hardware_and_specialization(mock_search)
        )

        self.assertIsNone(spec_key)


class TestGenerateBestAvailablePopulation(unittest.TestCase):
    """Tests for _generate_best_available_population_flat orchestration."""

    def _make_config_gen(self):
        """Create a ConfigGeneration with a simple 2-block spec."""
        config_spec = ConfigSpec()
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=0, size_hint=64, min_size=16, max_size=256)
        )
        config_spec.block_sizes.append(
            BlockSizeSpec(block_id=1, size_hint=128, min_size=16, max_size=256)
        )
        return ConfigGeneration(config_spec)

    def _make_mock_search(self, config_gen, cached_configs):
        """Create a mock PopulationBasedSearch with the given cached configs.

        cached_configs can be a list of Config objects (auto-wrapped in CacheEntry)
        or a list of CacheEntry objects.
        """
        entries = []
        for cfg in cached_configs:
            if isinstance(cfg, CacheEntry):
                entries.append(cfg)
            else:
                entries.append(
                    CacheEntry(hardware="test", specialization_key="test", config=cfg)
                )
        mock_search = MagicMock()
        mock_search.config_gen = config_gen
        mock_search.settings = Settings()
        mock_search.log = MagicMock()
        mock_search.log.debug = MagicMock()
        mock_search._find_similar_cached_configs = MagicMock(return_value=entries)
        # _transfer_config_to_flat now takes a CacheEntry
        mock_search._transfer_config_to_flat = (
            lambda entry: list(entry.flat_config)
            if entry.flat_config is not None
            else config_gen.flatten(entry.config)
        )
        return mock_search

    def test_default_only_when_no_cached(self):
        """Population contains only default config when no cached configs found."""
        config_gen = self._make_config_gen()
        mock_search = self._make_mock_search(config_gen, cached_configs=[])

        result = PopulationBasedSearch._generate_best_available_population_flat(
            mock_search
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], config_gen.default_flat())

    def test_cached_configs_added(self):
        """Cached configs are added after default."""
        config_gen = self._make_config_gen()
        cached = [
            Config(block_sizes=[32, 64], num_warps=8, num_stages=2),
            Config(block_sizes=[128, 256], num_warps=2, num_stages=4),
        ]
        mock_search = self._make_mock_search(config_gen, cached)

        result = PopulationBasedSearch._generate_best_available_population_flat(
            mock_search
        )

        self.assertEqual(len(result), 3)  # 1 default + 2 cached
        self.assertEqual(result[0], config_gen.default_flat())
        # Verify cached values appear in the flat configs
        num_warps_idx = config_gen._key_to_flat_indices["num_warps"][0]
        self.assertEqual(result[1][num_warps_idx], 8)
        self.assertEqual(result[2][num_warps_idx], 2)

    def test_duplicate_configs_deduplicated(self):
        """Duplicate cached configs are discarded."""
        config_gen = self._make_config_gen()
        same_config = Config(block_sizes=[32, 64], num_warps=8, num_stages=2)
        cached = [same_config, Config(block_sizes=[32, 64], num_warps=8, num_stages=2)]
        mock_search = self._make_mock_search(config_gen, cached)

        result = PopulationBasedSearch._generate_best_available_population_flat(
            mock_search
        )

        self.assertEqual(len(result), 2)  # 1 default + 1 unique cached

    def test_default_duplicate_in_cache_deduplicated(self):
        """A cached config identical to default is deduplicated."""
        config_gen = self._make_config_gen()
        default_config = config_gen.config_spec.default_config()
        cached = [default_config]
        mock_search = self._make_mock_search(config_gen, cached)

        result = PopulationBasedSearch._generate_best_available_population_flat(
            mock_search
        )

        self.assertEqual(len(result), 1)  # only default

    def test_failed_transfer_skipped(self):
        """Entries that fail transfer are skipped without crashing."""
        config_gen = self._make_config_gen()
        good_config = Config(block_sizes=[32, 64], num_warps=8, num_stages=2)

        # Simulate an old cache entry whose flatten() raises (wrong structure)
        bad_entry = CacheEntry(
            hardware="test",
            specialization_key="test",
            config=Config(block_sizes=[32, 64, 128], num_warps=4),
        )
        good_entry = CacheEntry(
            hardware="test",
            specialization_key="test",
            config=good_config,
        )

        mock_search = self._make_mock_search(config_gen, [bad_entry, good_entry])

        result = PopulationBasedSearch._generate_best_available_population_flat(
            mock_search
        )

        self.assertEqual(len(result), 2)  # 1 default + 1 good cached


if __name__ == "__main__":
    unittest.main()
