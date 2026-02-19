"""
Tests for AOT Autotuning Framework
==================================

Tests for the collect/measure/evaluate workflow.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from helion._testing import onlyBackends
from helion.autotuner.aot_cache import ShapeKey
from helion.autotuner.aot_cache import _deserialize_tuple
from helion.autotuner.aot_cache import _serialize_tuple
from helion.autotuner.aot_cache import get_aot_mode
from helion.autotuner.heuristic_generator import PerformanceTarget
from helion.autotuner.heuristic_generator import ShapeConfigData
from helion.autotuner.heuristic_generator import select_config_subset
from helion.experimental.aot_kernel import aot_key
from helion.experimental.aot_kernel import extract_shape_features
from helion.runtime.config import Config


@onlyBackends(["triton"])
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


@onlyBackends(["triton"])
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


@onlyBackends(["triton"])
class TestConfigSubsetSelection:
    """Tests for config subset selection algorithm."""

    def test_single_config_optimal(self) -> None:
        # Create data where one config is optimal for all shapes
        data = ShapeConfigData(
            kernel_name="test",
            shape_features=[{"dim": 1024}, {"dim": 2048}],
            timings=np.array(
                [
                    [1.0, 2.0],  # Config 0 is best for shape 0
                    [1.0, 2.0],  # Config 0 is best for shape 1
                ]
            ),
            configs=[Config(block_sizes=[64]), Config(block_sizes=[128])],
            shape_hashes=["s1", "s2"],
            config_hashes=["c1", "c2"],
        )

        target = PerformanceTarget(goal_type="max_slowdown", threshold=1.1)
        selected, stats = select_config_subset(data, target)

        assert len(selected) == 1
        assert selected[0] == 0  # Config 0 should be selected

    def test_multiple_configs_needed(self) -> None:
        # Create data where different configs are optimal for different shapes
        data = ShapeConfigData(
            kernel_name="test",
            shape_features=[{"dim": 1024}, {"dim": 2048}],
            timings=np.array(
                [
                    [1.0, 10.0],  # Config 0 is best for shape 0
                    [10.0, 1.0],  # Config 1 is best for shape 1
                ]
            ),
            configs=[Config(block_sizes=[64]), Config(block_sizes=[128])],
            shape_hashes=["s1", "s2"],
            config_hashes=["c1", "c2"],
        )

        target = PerformanceTarget(goal_type="max_slowdown", threshold=1.1)
        selected, stats = select_config_subset(data, target)

        # Both configs needed to meet performance goal
        assert len(selected) == 2


@onlyBackends(["triton"])
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


@onlyBackends(["triton"])
class TestBatchedParameter:
    """Tests for the batched parameter in aot_kernel."""

    def test_extract_features_without_batched(self) -> None:
        """Test that extract_shape_features includes all dimensions without batched."""
        x = torch.randn(32, 128)
        features = extract_shape_features([x])

        assert "arg0_ndim" in features
        assert "arg0_dim0" in features
        assert "arg0_dim1" in features
        assert "arg0_numel" in features
        assert features["arg0_dim0"] == 32
        assert features["arg0_dim1"] == 128

    def test_extract_features_with_batched(self) -> None:
        """Test that extract_shape_features excludes batched dimensions."""
        x = torch.randn(32, 128)
        # First dimension is batched
        features = extract_shape_features([x], batched=[[0, None]])

        assert "arg0_ndim" in features
        assert "arg0_dim0" not in features  # Batched dim excluded
        assert "arg0_dim1" in features  # Non-batched dim included
        assert "arg0_numel" not in features  # numel excluded when has batched dims
        assert features["arg0_dim1"] == 128

    def test_extract_features_multiple_args(self) -> None:
        """Test batched with multiple arguments (like rms_norm)."""
        weight = torch.randn(128)
        input_tensor = torch.randn(32, 128)
        eps = 1e-5

        # weight: not batched, input: first dim batched, eps: scalar
        batched = [[None], [0, None], None]
        features = extract_shape_features([weight, input_tensor, eps], batched=batched)

        # Weight features (not batched)
        assert "arg0_dim0" in features
        assert "arg0_numel" in features
        assert features["arg0_dim0"] == 128

        # Input features (first dim batched)
        assert "arg1_dim0" not in features  # Batched
        assert "arg1_dim1" in features  # Not batched
        assert "arg1_numel" not in features  # Excluded due to batched dim
        assert features["arg1_dim1"] == 128

        # Scalar feature
        assert "arg2_scalar" in features
        assert features["arg2_scalar"] == eps

    def test_aot_key_same_for_different_batch_sizes(self) -> None:
        """Test that different batch sizes produce the same key when batched is specified."""
        x1 = torch.randn(32, 128)
        x2 = torch.randn(64, 128)  # Different batch size, same hidden dim

        key1 = aot_key(x1, batched=[[0, None]])
        key2 = aot_key(x2, batched=[[0, None]])

        assert key1 == key2

    def test_aot_key_different_for_different_non_batch_dims(self) -> None:
        """Test that different non-batch dimensions produce different keys."""
        x1 = torch.randn(32, 128)
        x2 = torch.randn(32, 256)  # Same batch size, different hidden dim

        key1 = aot_key(x1, batched=[[0, None]])
        key2 = aot_key(x2, batched=[[0, None]])

        assert key1 != key2

    def test_aot_key_rms_norm_scenario(self) -> None:
        """Test the rms_norm scenario with weight, input, eps."""
        weight = torch.randn(128)
        input1 = torch.randn(32, 128)
        input2 = torch.randn(64, 128)  # Different batch size
        eps = 1e-5

        batched = [[None], [0, None], None]

        key1 = aot_key(weight, input1, eps, batched=batched)
        key2 = aot_key(weight, input2, eps, batched=batched)

        # Keys should be the same despite different batch sizes
        assert key1 == key2

    def test_batched_with_no_batched_dims(self) -> None:
        """Test that specifying all None in batched is equivalent to no batched."""
        x = torch.randn(32, 128)

        # All dimensions marked as not batched
        features_with_batched = extract_shape_features([x], batched=[[None, None]])
        features_without_batched = extract_shape_features([x])

        assert features_with_batched == features_without_batched


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
