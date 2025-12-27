"""
Tests for AOT Autotuning Framework
==================================

Tests for the collect/measure/evaluate workflow.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from helion.autotuner.aot_cache import AOTDataStore
from helion.autotuner.aot_cache import ShapeKey
from helion.autotuner.aot_cache import _deserialize_tuple
from helion.autotuner.aot_cache import _serialize_tuple
from helion.autotuner.aot_cache import get_aot_mode
from helion.autotuner.heuristic_generator import MeasurementData
from helion.autotuner.heuristic_generator import PerformanceTarget
from helion.autotuner.heuristic_generator import select_config_subset
from helion.runtime.config import Config


class TestShapeKey:
    """Tests for ShapeKey class."""

    def test_to_dict_and_back(self) -> None:
        key = ShapeKey(
            kernel_name="test_kernel",
            specialization_key=(1024, 2048, "float32"),
            hardware_id="cuda_RTX4090_12.4",
        )
        d = key.to_dict()
        restored = ShapeKey.from_dict(d)
        assert restored.kernel_name == key.kernel_name
        assert restored.hardware_id == key.hardware_id

    def test_stable_hash(self) -> None:
        key1 = ShapeKey("k", (1, 2, 3), "hw")
        key2 = ShapeKey("k", (1, 2, 3), "hw")
        assert key1.stable_hash() == key2.stable_hash()

        key3 = ShapeKey("k", (1, 2, 4), "hw")
        assert key1.stable_hash() != key3.stable_hash()


class TestSerializeTuple:
    """Tests for tuple serialization."""

    def test_simple_tuple(self) -> None:
        t = (1, 2, 3)
        serialized = _serialize_tuple(t)
        deserialized = _deserialize_tuple(serialized)
        assert deserialized == t

    def test_nested_tuple(self) -> None:
        t = (1, (2, 3), 4)
        serialized = _serialize_tuple(t)
        deserialized = _deserialize_tuple(serialized)
        assert deserialized == t


class TestAOTDataStore:
    """Tests for AOTDataStore."""

    def test_save_and_load_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AOTDataStore(Path(tmpdir), "test_hw")

            shape_key = ShapeKey("kernel1", (1024,), "test_hw")
            config = Config(block_sizes=[64], num_warps=4)

            store.add_tuned_config("kernel1", config, shape_key)
            store.flush()

            # Load in new store
            store2 = AOTDataStore(Path(tmpdir), "test_hw")
            loaded = store2.load_tuned_configs()

            assert "kernel1" in loaded
            assert len(loaded["kernel1"]) == 1
            assert dict(loaded["kernel1"][0].config) == dict(config)

    def test_get_all_configs_for_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AOTDataStore(Path(tmpdir), "test_hw")

            # Add multiple configs for same kernel
            shape_key1 = ShapeKey("kernel1", (1024,), "test_hw")
            shape_key2 = ShapeKey("kernel1", (2048,), "test_hw")

            config1 = Config(block_sizes=[64], num_warps=4)
            config2 = Config(block_sizes=[128], num_warps=8)

            store.add_tuned_config("kernel1", config1, shape_key1)
            store.add_tuned_config("kernel1", config2, shape_key2)

            configs = store.get_all_configs_for_kernel("kernel1")
            assert len(configs) == 2


class TestConfigSubsetSelection:
    """Tests for config subset selection algorithm."""

    def test_single_config_optimal(self) -> None:
        # Create data where one config is optimal for all shapes
        data = MeasurementData(
            kernel_name="test",
            shape_features=[{"dim": 1024}, {"dim": 2048}],
            configs=[Config(block_sizes=[64]), Config(block_sizes=[128])],
            timings=np.array(
                [
                    [1.0, 2.0],  # Config 0 is best for shape 0
                    [1.0, 2.0],  # Config 0 is best for shape 1
                ]
            ),
            shape_hashes=["s1", "s2"],
            config_hashes=["c1", "c2"],
        )

        target = PerformanceTarget(goal_type="max_slowdown", threshold=1.1)
        selected, stats = select_config_subset(data, target)

        assert len(selected) == 1
        assert selected[0] == 0  # Config 0 should be selected

    def test_multiple_configs_needed(self) -> None:
        # Create data where different configs are optimal for different shapes
        data = MeasurementData(
            kernel_name="test",
            shape_features=[{"dim": 1024}, {"dim": 2048}],
            configs=[Config(block_sizes=[64]), Config(block_sizes=[128])],
            timings=np.array(
                [
                    [1.0, 10.0],  # Config 0 is best for shape 0
                    [10.0, 1.0],  # Config 1 is best for shape 1
                ]
            ),
            shape_hashes=["s1", "s2"],
            config_hashes=["c1", "c2"],
        )

        target = PerformanceTarget(goal_type="max_slowdown", threshold=1.1)
        selected, stats = select_config_subset(data, target)

        # Both configs needed to meet performance goal
        assert len(selected) == 2


class TestGetAOTMode:
    """Tests for get_aot_mode."""

    def test_default_mode(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            if "HELION_AOT_MODE" in os.environ:
                del os.environ["HELION_AOT_MODE"]
            # Default mode is "evaluate" to enable heuristic-based config selection
            assert get_aot_mode() == "evaluate"

    def test_collect_mode(self) -> None:
        with patch.dict(os.environ, {"HELION_AOT_MODE": "collect"}):
            assert get_aot_mode() == "collect"

    def test_invalid_mode(self) -> None:
        with (
            patch.dict(os.environ, {"HELION_AOT_MODE": "invalid"}),
            pytest.raises(ValueError),
        ):
            get_aot_mode()


class TestMeasurementsIO:
    """Tests for measurement file I/O."""

    def test_save_and_load_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AOTDataStore(Path(tmpdir), "test_hw")

            shape_key = ShapeKey("kernel1", (1024,), "test_hw")
            config = Config(block_sizes=[64], num_warps=4)

            store.save_measurement(
                kernel_name="kernel1",
                shape_key=shape_key,
                config=config,
                timing_ms=1.5,
                shape_features={"dim0": 1024},
            )

            measurements = store.load_measurements()
            assert len(measurements) == 1
            assert measurements[0]["kernel_name"] == "kernel1"
            assert measurements[0]["timing_ms"] == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
