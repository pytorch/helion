"""
Tests for AOT Autotuning Framework
==================================

Tests for the collect/measure/evaluate workflow.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
import inspect
import json
import logging
from operator import itemgetter
import os
from pathlib import Path
import sys
import threading
import time
from types import ModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from helion import exc
import helion._argument_device as argument_device_module
import helion._hardware as hardware_module
from helion._hardware import HardwareInfo
from helion._testing import onlyBackends
import helion.autotuner.aot_cache as aot_cache_module
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.aot_cache import ShapeKey
from helion.autotuner.aot_cache import _deserialize_tuple
from helion.autotuner.aot_cache import _deserialize_value
from helion.autotuner.aot_cache import _serialize_tuple
from helion.autotuner.aot_cache import _serialize_value
from helion.autotuner.aot_cache import get_aot_mode
from helion.autotuner.aot_compile import _standalone_call_key
from helion.autotuner.aot_compile import canonical_kernel_source_path
from helion.autotuner.aot_compile import generate_standalone_file
from helion.autotuner.aot_kernel import HeuristicKeyFunction
from helion.autotuner.aot_kernel import aot_kernel
from helion.autotuner.aot_kernel import aot_key
from helion.autotuner.aot_kernel import extract_shape_features
from helion.autotuner.aot_runner import list_previous_runs
from helion.autotuner.heuristic_generator import PerformanceTarget
from helion.autotuner.heuristic_generator import ShapeConfigData
from helion.autotuner.heuristic_generator import compute_validity_partitions
from helion.autotuner.heuristic_generator import select_config_subset
import helion.language as hl
from helion.runtime.config import Config
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import _find_device

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

aot_kernel_module = importlib.import_module("helion.autotuner.aot_kernel")

B200 = HardwareInfo("cuda", "NVIDIA B200", "13.0", "sm100")
GB300 = HardwareInfo("cuda", "NVIDIA GB300", "13.0", "sm100")


@pytest.fixture(autouse=True)
def _clear_aot_test_caches() -> Iterator[None]:
    AOTAutotuneCache.clear_caches()
    HeuristicKeyFunction.clear_cache()
    hardware_module.clear_hardware_info_cache()
    yield
    AOTAutotuneCache.clear_caches()
    HeuristicKeyFunction.clear_cache()
    hardware_module.clear_hardware_info_cache()


def _kernel_source(tmp_path: Path, source: str = "def demo():\n    pass\n") -> Path:
    path = tmp_path / "kernel.py"
    path.write_text(source)
    return path


def _mock_active_aot(
    monkeypatch: pytest.MonkeyPatch,
    *,
    data_dir: Path | None = None,
    hardware: HardwareInfo | None = None,
    mode: str = "evaluate",
) -> None:
    monkeypatch.setenv("HELION_AOT_MODE", mode)
    if data_dir is not None:
        monkeypatch.setattr(aot_cache_module, "get_aot_data_dir", lambda: data_dir)
    if hardware is not None:
        monkeypatch.setattr(
            aot_cache_module,
            "get_hardware_info",
            lambda device=None: hardware,
        )


def _install_fake_jax(
    monkeypatch: pytest.MonkeyPatch,
    devices: Callable[[str], list[object]],
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(__version__="0.8.0", devices=devices),
    )


def _heuristic_cache_stub(
    source_path: Path,
    heuristic_path: Path,
    hardware: HardwareInfo,
) -> AOTAutotuneCache:
    cache = object.__new__(AOTAutotuneCache)
    cache.hardware = hardware
    cache._kernel_source = str(source_path.resolve())
    cache.kernel = SimpleNamespace(
        kernel=SimpleNamespace(name="demo", _aot_user_key=None)
    )
    cache.args = (16,)
    cache._find_heuristic_file = lambda: heuristic_path.resolve()
    cache._extract_shape_features = lambda args: {"value": args[0]}
    return cache


def test_list_previous_runs_handles_unknown_hardware(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "20260815-pending"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        '{"hardware_id": null, "started_at": "2026-08-15T08:00:00"}'
    )

    list_previous_runs(tmp_path)

    output = capsys.readouterr().out
    assert "20260815-pending" in output
    assert "unknown" in output


@onlyBackends(["triton", "cute"])
class TestShapeKey:
    """Tests for ShapeKey class."""

    def test_to_dict_and_back(self) -> None:
        hardware = HardwareInfo(
            device_kind="cuda",
            hardware_name="RTX4090",
            runtime_version="12.4",
            compute_capability="sm89",
        )
        key = ShapeKey(
            kernel_name="test_kernel",
            specialization_key=(1024, 2048, "float32"),
            hardware_id=hardware.hardware_id,
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


@onlyBackends(["triton", "cute"])
class TestCodeSerialization:
    """Tests for specialization keys containing function code objects (e.g. callable kernel args)."""

    def test_code_round_trips_through_serialize_value(self) -> None:
        def fn(v):
            return v * 2

        serialized = _serialize_value(fn.__code__)
        deserialized = _deserialize_value(serialized)
        assert deserialized == (
            fn.__code__.co_code,
            fn.__code__.co_consts,
            fn.__code__.co_names,
        )

    def test_stable_hash_same_for_rename(self) -> None:
        def double(v):
            return v * 2

        def double_renamed(v):
            return v * 2

        h1 = ShapeKey("k", (double.__code__,), "hw").stable_hash()
        h2 = ShapeKey("k", (double_renamed.__code__,), "hw").stable_hash()
        assert h1 == h2

    def test_stable_hash_differs_for_behavior_change(self) -> None:
        def double(v):
            return v * 2

        def triple(v):
            return v * 3

        h1 = ShapeKey("k", (double.__code__,), "hw").stable_hash()
        h2 = ShapeKey("k", (triple.__code__,), "hw").stable_hash()
        assert h1 != h2

    def test_stable_hash_differs_for_nested_lambda_behavior_change(self) -> None:
        def with_nested_lambda_2():
            return lambda v: v * 2

        def with_nested_lambda_3():
            return lambda v: v * 3

        h1 = ShapeKey("k", (with_nested_lambda_2.__code__,), "hw").stable_hash()
        h2 = ShapeKey("k", (with_nested_lambda_3.__code__,), "hw").stable_hash()
        assert h1 != h2

    def test_stable_hash_for_conames(self) -> None:
        def with_sin(v):
            return v.sin()

        def with_cos(v):
            return v.cos()

        h1 = ShapeKey("k", (with_sin.__code__,), "hw").stable_hash()
        h2 = ShapeKey("k", (with_cos.__code__,), "hw").stable_hash()
        assert h1 != h2

    def test_code_serializes_complex_and_ellipsis_consts(self) -> None:
        def with_complex(v):
            return v * 1j

        def with_ellipsis(v):  # noqa: FURB118
            return v[...]

        for fn in (with_complex, with_ellipsis):
            serialized = _serialize_value(fn.__code__)
            json.dumps(serialized)  # must not raise
            assert _deserialize_value(serialized) == (
                fn.__code__.co_code,
                fn.__code__.co_consts,
                fn.__code__.co_names,
            )

    def test_stable_hash_survives_save_load_round_trip(self) -> None:
        def fn(v):
            return v * 2

        key = ShapeKey("k", (fn.__code__,), "hw")
        original_hash = key.stable_hash()

        # Simulate a JSON save/load cycle
        reloaded = ShapeKey.from_dict(json.loads(json.dumps(key.to_dict())))
        assert reloaded.stable_hash() == original_hash
        reloaded_again = ShapeKey.from_dict(json.loads(json.dumps(reloaded.to_dict())))
        assert reloaded_again.stable_hash() == original_hash


@onlyBackends(["triton", "cute"])
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


@onlyBackends(["triton", "cute"])
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

        assert stats["num_partitions"] == 1
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


@onlyBackends(["triton", "cute"])
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


@onlyBackends(["triton", "cute"])
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


@onlyBackends(["triton", "cute"])
class TestConfigValidityPartitioning:
    """Tests for config validity partitioning in select_config_subset."""

    def test_single_config_with_validity_partitioning(self) -> None:
        """With max_configs=1, partitioning selects one config per partition."""
        # Two independent groups of shapes with disjoint valid configs
        data = ShapeConfigData(
            kernel_name="test",
            shape_features=[{"dim": i} for i in range(4)],
            timings=np.array(
                [
                    [1.0, 2.0, np.inf, np.inf],  # Partition 1
                    [2.0, 1.0, np.inf, np.inf],  # Partition 1
                    [np.inf, np.inf, 1.0, 2.0],  # Partition 2
                    [np.inf, np.inf, 2.0, 1.0],  # Partition 2
                ]
            ),
            configs=[Config(block_sizes=[i]) for i in [64, 128, 256, 512]],
            shape_hashes=["s0", "s1", "s2", "s3"],
            config_hashes=["c0", "c1", "c2", "c3"],
        )

        target = PerformanceTarget(
            goal_type="max_slowdown", threshold=1.1, max_configs=1, verbose=False
        )
        selected, stats = select_config_subset(data, target)

        # Should select configs from both partitions despite max_configs=1
        assert len(selected) >= 2
        # All shapes should have a valid selected config
        for i in range(4):
            assert any(np.isfinite(data.timings[i, j]) for j in selected)
        assert stats["num_partitions"] == 2

    def test_partitioning_independent_optimization(self) -> None:
        """Each partition selects its own optimal config independently."""
        data = ShapeConfigData(
            kernel_name="test",
            shape_features=[{"dim": i} for i in range(4)],
            timings=np.array(
                [
                    [1.0, 5.0, np.inf, np.inf],  # Partition 1: config 0 best
                    [1.5, 5.0, np.inf, np.inf],
                    [np.inf, np.inf, 1.0, 5.0],  # Partition 2: config 2 best
                    [np.inf, np.inf, 1.5, 5.0],
                ]
            ),
            configs=[Config(block_sizes=[i]) for i in [64, 128, 256, 512]],
            shape_hashes=["s0", "s1", "s2", "s3"],
            config_hashes=["c0", "c1", "c2", "c3"],
        )

        target = PerformanceTarget(
            goal_type="max_slowdown", threshold=1.1, max_configs=1, verbose=False
        )
        selected, stats = select_config_subset(data, target)

        # Config 0 for partition 1, config 2 for partition 2
        assert 0 in selected  # Best for partition 1
        assert 2 in selected  # Best for partition 2
        assert stats["num_partitions"] == 2

    def test_uncoverable_shapes_skipped(self) -> None:
        """Shapes with no valid config are handled gracefully."""
        data = ShapeConfigData(
            kernel_name="test",
            shape_features=[{"dim": i} for i in range(3)],
            timings=np.array(
                [
                    [1.0, 2.0],
                    [2.0, 1.0],
                    [np.inf, np.inf],  # No valid config
                ]
            ),
            configs=[Config(block_sizes=[64]), Config(block_sizes=[128])],
            shape_hashes=["s0", "s1", "s2"],
            config_hashes=["c0", "c1"],
        )

        target = PerformanceTarget(
            goal_type="max_slowdown", threshold=1.1, verbose=False
        )
        selected, stats = select_config_subset(data, target)

        # Stats should not be inf or nan
        assert np.isfinite(stats["max_slowdown"])
        assert np.isfinite(stats["geomean_slowdown"])
        assert np.isfinite(stats["avg_slowdown"])
        # Coverable shapes should be covered
        assert len(selected) >= 1

    def test_compute_validity_partitions_basic(self) -> None:
        """Test union-find partitioning directly."""
        timings = np.array(
            [
                [1.0, np.inf],
                [2.0, np.inf],
                [np.inf, 1.0],
                [np.inf, 2.0],
            ]
        )
        partitions, uncoverable = compute_validity_partitions(timings)

        assert len(partitions) == 2
        assert len(uncoverable) == 0

        # Check that shapes are correctly grouped
        partition_sets = [set(p) for p in partitions]
        assert {0, 1} in partition_sets
        assert {2, 3} in partition_sets

    def test_compute_validity_partitions_shared_config(self) -> None:
        """Shapes sharing a valid config are in the same partition."""
        timings = np.array(
            [
                [1.0, np.inf, 2.0],  # Configs 0,2 valid
                [np.inf, 1.0, 3.0],  # Configs 1,2 valid — connected via config 2
                [np.inf, 2.0, np.inf],  # Config 1 valid — connected via config 1
            ]
        )
        partitions, uncoverable = compute_validity_partitions(timings)

        # All connected through shared configs → single partition
        assert len(partitions) == 1
        assert set(partitions[0]) == {0, 1, 2}
        assert len(uncoverable) == 0

    def test_compute_validity_partitions_uncoverable(self) -> None:
        """Shapes with all-inf timings are uncoverable."""
        timings = np.array(
            [
                [1.0, 2.0],
                [np.inf, np.inf],  # Uncoverable
            ]
        )
        partitions, uncoverable = compute_validity_partitions(timings)

        assert len(partitions) == 1
        assert partitions[0] == [0]
        assert uncoverable == [1]

    def test_mixed_dimensionality_end_to_end(self) -> None:
        """Partitioning + decision tree correctly routes 2D vs 3D inputs."""
        from helion.autotuner.decision_tree_backend import DecisionTreeBackend

        # 2D shapes: no arg0_dim2 in feature dict
        # 3D shapes: arg0_dim2 present
        shape_features = [
            {"arg0_ndim": 2, "arg0_dim0": 1024, "arg0_dim1": 512},
            {"arg0_ndim": 2, "arg0_dim0": 2048, "arg0_dim1": 256},
            {"arg0_ndim": 3, "arg0_dim0": 32, "arg0_dim1": 1024, "arg0_dim2": 512},
            {"arg0_ndim": 3, "arg0_dim0": 64, "arg0_dim1": 512, "arg0_dim2": 256},
        ]

        # Configs 0,1 only valid for 2D; configs 2,3 only valid for 3D
        timings = np.array(
            [
                [1.0, 5.0, np.inf, np.inf],
                [1.5, 5.0, np.inf, np.inf],
                [np.inf, np.inf, 1.0, 5.0],
                [np.inf, np.inf, 1.5, 5.0],
            ]
        )

        data = ShapeConfigData(
            kernel_name="test_mixed",
            shape_features=shape_features,
            timings=timings,
            configs=[
                Config(block_sizes=[64]),
                Config(block_sizes=[128]),
                Config(block_sizes=[256]),
                Config(block_sizes=[512]),
            ],
            shape_hashes=["s0", "s1", "s2", "s3"],
            config_hashes=["c0", "c1", "c2", "c3"],
        )

        # Step 1: Config selection — partitioning should pick one config per partition
        target = PerformanceTarget(
            goal_type="max_slowdown", threshold=1.5, max_configs=1, verbose=False
        )
        selected_indices, stats = select_config_subset(data, target)

        assert stats["num_partitions"] == 2
        assert 0 in selected_indices  # Best for 2D partition
        assert 2 in selected_indices  # Best for 3D partition

        # Step 2: Train decision tree on the partitioned selection
        data.selected_config_indices = selected_indices
        selected_configs = [data.configs[i] for i in selected_indices]

        # Gather all feature names across shapes (2D shapes lack arg0_dim2)
        feature_names = sorted(
            {
                k
                for f in shape_features
                for k, v in f.items()
                if isinstance(v, (int, float))
            }
        )

        backend = DecisionTreeBackend()
        result = backend.generate_heuristic(
            kernel_name="test_mixed",
            data=data,
            selected_configs=selected_configs,
            feature_names=feature_names,
        )

        # Tree should perfectly separate 2D from 3D shapes
        assert result.model_accuracy == 1.0

        # Step 3: Execute generated code and verify runtime predictions
        exec_globals: dict[str, object] = {"torch": torch}
        exec(result.generated_code, exec_globals)

        key_fn = exec_globals["key_test_mixed"]
        autotune_fn = exec_globals["autotune_test_mixed"]

        # 2D tensor → config index 0 (the 2D partition's config)
        assert key_fn(torch.randn(100, 200)) == 0
        # 3D tensor → config index 1 (the 3D partition's config)
        assert key_fn(torch.randn(10, 100, 200)) == 1

        # autotune returns the actual config dicts
        assert autotune_fn(torch.randn(100, 200)) == dict(selected_configs[0])
        assert autotune_fn(torch.randn(10, 100, 200)) == dict(selected_configs[1])


def _load_generated(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _mock_cuda_hardware(monkeypatch: pytest.MonkeyPatch) -> list[torch.device]:
    observed_devices: list[torch.device] = []

    def get_device_properties(device: torch.device) -> SimpleNamespace:
        observed_devices.append(device)
        return SimpleNamespace(name="NVIDIA H100", major=9, minor=0)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.version, "hip", None)
    return observed_devices


def _get_uncached_hardware(device: torch.device) -> HardwareInfo:
    return hardware_module.get_hardware_info(device)


def _expected_argument_device(args: tuple[object, ...]) -> torch.device | None:
    """Use the uncached ``Kernel.bind`` traversal as the resolver oracle."""
    try:
        device = _find_device(args)
    except exc.NoTensorArgs:
        return None
    return argument_device_module._canonicalize_argument_device(device)


def test_argument_device_uses_pinned_torch_accelerator_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch,
        "privateuseone",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

    def current_accelerator(*, check_available: bool = False) -> SimpleNamespace:
        assert check_available
        return SimpleNamespace(type="privateuseone")

    monkeypatch.setattr(torch.accelerator, "current_accelerator", current_accelerator)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)

    assert argument_device_module._current_device_index("privateuseone") == 3


def test_argument_device_preserves_unavailable_indexless_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.accelerator,
        "current_accelerator",
        lambda *, check_available=False: (
            None if check_available else torch.device("cuda")
        ),
    )
    monkeypatch.setattr(
        torch.accelerator,
        "current_device_index",
        lambda: pytest.fail("unavailable accelerators have no current device"),
    )

    indexless = torch.device("cuda")
    assert argument_device_module._canonicalize_argument_device(indexless) == indexless


@pytest.mark.parametrize(
    ("tpu_devices", "expected_kind", "expected_name", "expected_cuda_devices"),
    (
        ((), "cuda", "NVIDIA H100", (torch.device("cuda:0"),)),
        (
            (SimpleNamespace(platform="tpu", device_kind="TPU v5p"),),
            "tpu",
            "TPU v5p",
            (),
        ),
    ),
    ids=("cuda-fallback", "tpu-preferred"),
)
def test_cpu_bridge_prefers_tpu_else_falls_back_to_visible_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tpu_devices: tuple[object, ...],
    expected_kind: str,
    expected_name: str,
    expected_cuda_devices: tuple[torch.device, ...],
) -> None:
    observed_devices = _mock_cuda_hardware(monkeypatch)
    requested_backends: list[str] = []

    def devices(backend: str) -> list[object]:
        requested_backends.append(backend)
        return list(tpu_devices)

    _install_fake_jax(monkeypatch, devices)

    hardware = _get_uncached_hardware(torch.device("cpu"))

    assert hardware.device_kind == expected_kind
    assert hardware.hardware_name == expected_name
    assert requested_backends == ["tpu"]
    assert observed_devices == list(expected_cuda_devices)


def test_cpu_bridge_surfaces_tpu_hardware_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_hardware_info = hardware_module.HardwareInfo

    def hardware_info(
        *,
        device_kind: str,
        hardware_name: str,
        runtime_version: str,
        compute_capability: str,
    ) -> HardwareInfo:
        if device_kind == "tpu":
            raise RuntimeError("invalid TPU hardware metadata")
        return original_hardware_info(
            device_kind=device_kind,
            hardware_name=hardware_name,
            runtime_version=runtime_version,
            compute_capability=compute_capability,
        )

    _mock_cuda_hardware(monkeypatch)
    monkeypatch.setattr(hardware_module, "HardwareInfo", hardware_info)
    _install_fake_jax(
        monkeypatch,
        lambda backend: [SimpleNamespace(platform="tpu", device_kind="TPU v5p")],
    )

    with pytest.raises(RuntimeError, match="invalid TPU hardware metadata"):
        _get_uncached_hardware(torch.device("cpu"))


@pytest.mark.parametrize("error_type", (ImportError, OSError, RuntimeError))
def test_cpu_bridge_logs_and_caches_jax_backend_failure_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error_type: type[Exception],
) -> None:
    observed_devices = _mock_cuda_hardware(monkeypatch)
    probe_count = 0
    requested_backends: list[str] = []

    def devices(backend: str) -> list[object]:
        nonlocal probe_count
        probe_count += 1
        requested_backends.append(backend)
        raise error_type("no TPU backend")

    _install_fake_jax(monkeypatch, devices)

    with caplog.at_level(logging.DEBUG, logger=hardware_module.__name__):
        hardware = hardware_module.get_hardware_info(torch.device("cpu"))
        cached_hardware = hardware_module.get_hardware_info(torch.device("cpu"))

    assert hardware == HardwareInfo("cuda", "NVIDIA H100", "13.0", "sm90")
    assert cached_hardware is hardware
    assert observed_devices == [torch.device("cuda:0")]
    assert probe_count == 1
    assert requested_backends == ["tpu"]
    assert "JAX TPU discovery failed" in caplog.text


def test_hardware_cache_canonicalizes_indexless_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    properties = {
        devices[0]: SimpleNamespace(name="NVIDIA B200", major=10, minor=0),
        devices[1]: SimpleNamespace(name="NVIDIA GB300", major=10, minor=0),
    }
    property_lookups: list[torch.device] = []

    def get_device_properties(device: torch.device) -> SimpleNamespace:
        property_lookups.append(device)
        return properties[device]

    current_indices = iter((0, 1, 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: next(current_indices))
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.version, "hip", None)
    indexless = torch.device("cuda")
    b200 = hardware_module.get_hardware_info(indexless)
    gb300 = hardware_module.get_hardware_info(indexless)
    b200_again = hardware_module.get_hardware_info(indexless)

    assert b200.hardware_name == "NVIDIA B200"
    assert gb300.hardware_name == "NVIDIA GB300"
    assert b200_again is b200
    assert property_lookups == list(devices)


@pytest.mark.parametrize("device", (torch.device("xpu"), torch.device("xpu:0")))
def test_unavailable_xpu_falls_back_to_visible_cuda(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
) -> None:
    observed_devices = _mock_cuda_hardware(monkeypatch)
    monkeypatch.setattr(
        torch,
        "xpu",
        SimpleNamespace(
            is_available=lambda: False,
            current_device=lambda: pytest.fail(
                "unavailable XPU must not query its current device"
            ),
        ),
    )
    _install_fake_jax(
        monkeypatch,
        lambda backend: pytest.fail("XPU fallback must not probe JAX"),
    )

    hardware = _get_uncached_hardware(device)

    assert hardware == HardwareInfo("cuda", "NVIDIA H100", "13.0", "sm90")
    assert observed_devices == [torch.device("cuda:0")]


def test_explicit_cuda_device_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_devices = _mock_cuda_hardware(monkeypatch)
    _install_fake_jax(
        monkeypatch,
        lambda backend: pytest.fail("explicit CUDA must not probe JAX"),
    )
    device = torch.device("cuda:3")

    hardware = _get_uncached_hardware(device)

    assert hardware == HardwareInfo("cuda", "NVIDIA H100", "13.0", "sm90")
    assert observed_devices == [device]


def test_heuristic_key_isolates_hardware_path_and_user_specialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    sm100_path = tmp_path / "_helion_aot_kernel_cuda_sm100.py"
    sm90_path = tmp_path / "_helion_aot_kernel_cuda_sm90.py"
    sm100_path.write_text("def key_demo(value):\n    return value // 10\n")
    sm90_path.write_text("def key_demo(value):\n    return value % 3\n")
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    override_path = override_dir / sm90_path.name
    override_path.write_text("def key_demo(value):\n    return -value\n")
    sm100_device = torch.device("cuda:0")
    sm90_device = torch.device("cuda:1")
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: (
            HardwareInfo("cuda", "B200", "13.0", "sm100")
            if device == sm100_device
            else HardwareInfo("cuda", "H100", "13.0", "sm90")
        ),
    )
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    key_fn = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=lambda value, device: value,
    )
    assert key_fn(7, sm100_device) == (
        "helion_aot_heuristic",
        str(sm100_path.resolve()),
        0,
    )
    assert key_fn(11, sm100_device) == (
        "helion_aot_heuristic",
        str(sm100_path.resolve()),
        1,
    )
    assert key_fn(7, sm90_device) == (
        "helion_aot_heuristic",
        str(sm90_path.resolve()),
        1,
    )
    monkeypatch.setenv(aot_cache_module.HEURISTIC_DIR_ENV, str(override_dir))
    assert key_fn(7, sm90_device) == (
        "helion_aot_heuristic",
        str(override_path.resolve()),
        -7,
    )


def test_exact_hardware_artifact_is_skipped_on_compatible_sm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    heuristic_path = tmp_path / "_helion_aot_kernel_cuda_sm100.py"
    heuristic_path.write_text(
        'SUPPORTED_HARDWARE_NAMES = ("NVIDIA B200",)\n'
        "def key_demo(value):\n"
        "    return value // 4\n"
    )
    b200_device = torch.device("cuda:0")
    gb300_device = torch.device("cuda:1")
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: B200 if device == b200_device else GB300,
    )
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    assert (
        aot_cache_module.find_heuristic_file(source_path, device=b200_device)
        == heuristic_path.resolve()
    )
    assert (
        aot_cache_module.find_heuristic_file(source_path, device=gb300_device) is None
    )
    assert aot_cache_module.heuristic_artifact_identity(heuristic_path, GB300) is None

    key_fn = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=lambda value, device: value,
    )
    artifact_identity = aot_cache_module.heuristic_artifact_identity(
        heuristic_path.resolve(),
        B200,
    )
    assert artifact_identity is not None
    assert key_fn(7, b200_device) == (
        "helion_aot_heuristic",
        artifact_identity,
        1,
    )
    monkeypatch.setattr(
        aot_cache_module,
        "find_heuristic_file",
        lambda *args, **kwargs: heuristic_path.resolve(),
    )
    assert key_fn(11, gb300_device) == 11


def test_symlinked_heuristic_metadata_uses_canonical_cache_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    target_path = tmp_path / "reviewed_heuristic.py"
    target_path.write_text('SUPPORTED_HARDWARE_NAMES = ("NVIDIA B200",)\n')
    heuristic_path = tmp_path / "_helion_aot_kernel_cuda_sm100.py"
    heuristic_path.symlink_to(target_path)
    hardware = B200
    device = torch.device("cuda:0")
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: hardware,
    )
    original_parse = aot_cache_module.ast.parse
    parse_calls = 0

    def counting_parse(*args: Any, **kwargs: Any) -> Any:
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(aot_cache_module.ast, "parse", counting_parse)
    found = aot_cache_module.find_heuristic_file(source_path, device=device)
    assert found == target_path.resolve()
    assert found is not None
    assert aot_cache_module.heuristic_artifact_identity(found, hardware) is not None
    assert parse_calls == 1


def test_exact_hardware_metadata_cache_refreshes_after_clear(tmp_path: Path) -> None:
    heuristic_path = tmp_path / "heuristic.py"
    heuristic_path.write_text('SUPPORTED_HARDWARE_NAMES = ("NVIDIA B200",)\n')
    b200_identity = aot_cache_module.heuristic_artifact_identity(heuristic_path, B200)
    assert b200_identity is not None
    heuristic_path.write_text('SUPPORTED_HARDWARE_NAMES = ("NVIDIA GB300",)\n')
    assert (
        aot_cache_module.heuristic_artifact_identity(heuristic_path, B200)
        == b200_identity
    )

    aot_cache_module.clear_heuristic_cache()
    assert aot_cache_module.heuristic_artifact_identity(heuristic_path, B200) is None
    gb300_identity = aot_cache_module.heuristic_artifact_identity(heuristic_path, GB300)
    assert gb300_identity is not None
    assert "NVIDIA GB300" in gb300_identity


def test_exact_hardware_metadata_ignores_nested_local_binding(tmp_path: Path) -> None:
    heuristic_path = tmp_path / "heuristic.py"
    heuristic_path.write_text(
        'SUPPORTED_HARDWARE_NAMES = ("NVIDIA B200",)\n'
        "def local_policy():\n"
        '    SUPPORTED_HARDWARE_NAMES = ("NVIDIA GB300",)\n'
        "    return SUPPORTED_HARDWARE_NAMES\n"
    )
    b200_identity = aot_cache_module.heuristic_artifact_identity(heuristic_path, B200)
    assert b200_identity is not None
    assert "NVIDIA B200" in b200_identity
    assert aot_cache_module.heuristic_artifact_identity(heuristic_path, GB300) is None


def test_exact_hardware_identity_rejects_malformed_metadata(tmp_path: Path) -> None:
    heuristic_path = tmp_path / "heuristic.py"
    heuristic_path.write_text("SUPPORTED_HARDWARE_NAMES = hardware_names()\n")
    with pytest.raises(
        aot_cache_module.HeuristicArtifactMetadataError,
        match="must have a literal value",
    ):
        aot_cache_module.heuristic_artifact_identity(heuristic_path, B200)


def test_user_key_uses_generated_projection_with_exact_hardware_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    heuristic_path = tmp_path / "_helion_aot_kernel_cuda_sm100.py"
    heuristic_path.write_text(
        'SUPPORTED_HARDWARE_NAMES = ("NVIDIA B200",)\n'
        "def key_demo(rows, columns):\n"
        "    return rows // 128, columns // 64\n"
    )
    b200_device = torch.device("cuda:0")
    gb300_device = torch.device("cuda:1")
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: B200 if device == b200_device else GB300,
    )
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    key_fn = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=lambda shape, device: (
            shape["rows"],
            (shape["columns"], None),
        ),
    )
    artifact_identity = aot_cache_module.heuristic_artifact_identity(
        heuristic_path.resolve(),
        B200,
    )
    assert artifact_identity is not None
    expected = (
        "helion_aot_heuristic",
        artifact_identity,
        (2, 2),
    )

    assert key_fn({"rows": 257, "columns": 129}, b200_device) == expected
    assert key_fn({"rows": 300, "columns": 180}, b200_device) == expected
    assert key_fn({"rows": 257, "columns": 129}, gb300_device) == (
        257,
        (129, None),
    )


def test_loaded_exact_hardware_artifact_does_not_leak_to_other_sm100(
    tmp_path: Path,
) -> None:
    source_path = _kernel_source(tmp_path)
    heuristic_path = tmp_path / "_helion_aot_kernel_cuda_sm100.py"
    heuristic_path.write_text(
        'SUPPORTED_HARDWARE_NAMES = ("NVIDIA B200",)\n'
        "def autotune_demo(value):\n"
        "    return {'block_sizes': [value]}\n"
    )
    b200_config = _heuristic_cache_stub(
        source_path, heuristic_path, B200
    )._get_heuristic_config()
    assert b200_config is not None
    assert b200_config["block_sizes"] == [16]
    gb300_cache = _heuristic_cache_stub(source_path, heuristic_path, GB300)
    assert gb300_cache._get_heuristic_config() is None
    assert gb300_cache._get_heuristic_config() is None
    assert (
        AOTAutotuneCache._supported_heuristics[(heuristic_path.resolve(), GB300)]
        is None
    )


def test_aot_cache_logs_invalid_loaded_hardware_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_path = _kernel_source(tmp_path)
    heuristic_path = tmp_path / "_helion_aot_kernel_cuda_sm100.py"
    heuristic_path.write_text(
        'SUPPORTED_HARDWARE_NAMES = ("NVIDIA B200",)\n'
        'globals()["SUPPORTED_HARDWARE_NAMES"] = ("NVIDIA GB300",)\n'
        "def autotune_demo(value):\n"
        "    return {'block_sizes': [value]}\n"
    )
    cache = _heuristic_cache_stub(source_path, heuristic_path, B200)

    with caplog.at_level(logging.WARNING, logger=aot_cache_module.__name__):
        assert cache._get_heuristic_config() is None
    assert "Skipping heuristic with invalid artifact metadata" in caplog.text
    assert "Failed to load heuristic from" not in caplog.text


def test_aot_key_logs_invalid_loaded_hardware_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_path = _kernel_source(tmp_path)
    heuristic_path = tmp_path / "_helion_aot_kernel_cuda_sm100.py"
    heuristic_path.write_text(
        'SUPPORTED_HARDWARE_NAMES = ("NVIDIA B200",)\n'
        'globals()["SUPPORTED_HARDWARE_NAMES"] = ("NVIDIA GB300",)\n'
        "def key_demo(value):\n"
        "    return value // 4\n"
    )
    hardware = B200
    device = torch.device("cuda:0")
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: hardware,
    )
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path / "aot-data"))
    key_fn = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=lambda value, device: value,
    )
    with caplog.at_level(logging.WARNING, logger=aot_kernel_module.__name__):
        result = key_fn(7, device)
    assert result == 7
    assert "Skipping heuristic with invalid artifact metadata" in caplog.text
    assert "Failed to load AOT heuristic key from" not in caplog.text


@pytest.mark.parametrize(
    "invalid_metadata",
    [False, True],
    ids=("unsupported", "invalid-metadata"),
)
def test_rejected_loaded_aot_key_artifact_is_not_reloaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_metadata: bool,
) -> None:
    source_path = _kernel_source(tmp_path)
    load_marker = tmp_path / "loads"
    heuristic_path = tmp_path / "_helion_aot_kernel_cuda_sm100.py"
    heuristic_path.write_text(
        "from pathlib import Path\n"
        f"_marker = Path({str(load_marker)!r})\n"
        "_marker.write_text(_marker.read_text() + 'x' if _marker.exists() else 'x')\n"
        "def key_demo(value):\n"
        "    return -1\n"
    )
    hardware = B200
    validation_calls = 0

    def reject_loaded_artifact(
        module: ModuleType,
        artifact: Path,
        current_hardware: HardwareInfo,
    ) -> bool:
        nonlocal validation_calls
        validation_calls += 1
        assert module.key_demo(1) == -1
        assert artifact == heuristic_path.resolve()
        assert current_hardware == hardware
        if invalid_metadata:
            raise aot_cache_module.HeuristicArtifactMetadataError("invalid metadata")
        return False

    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: hardware,
    )
    monkeypatch.setattr(
        aot_cache_module,
        "heuristic_module_supports_hardware",
        reject_loaded_artifact,
    )
    device = torch.device("cuda:0")
    first = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=lambda value, device: value,
    )
    second = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=lambda value, device: value,
    )

    assert first(7, device) == 7
    assert second(11, device) == 11
    assert load_marker.read_text() == "x"
    assert validation_calls == 1


def test_heuristic_key_nested_device_path_hot_loop_and_device_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    tensors = (torch.empty(0), torch.empty(0, device=torch.device("meta")))
    devices = tuple(tensor.device for tensor in tensors)
    heuristic_paths = {
        device: tmp_path / f"heuristic_{device.type}.py" for device in devices
    }
    for heuristic_path in heuristic_paths.values():
        heuristic_path.write_text("")
    discovered_devices: list[torch.device | None] = []
    recursive_lookups = 0
    original_lookup = argument_device_module._find_argument_device_with_path

    def find_heuristic_file(
        _source_file: str,
        *,
        kernel_name: str,
        data_dir: Path,
        device: torch.device | None,
        resolved_heuristic_dir: Path | None = None,
    ) -> Path:
        assert kernel_name == "demo"
        assert data_dir == tmp_path
        discovered_devices.append(device)
        assert device is not None
        return heuristic_paths[device]

    def counted_lookup(
        args: tuple[object, ...],
    ) -> tuple[torch.device, object] | None:
        nonlocal recursive_lookups
        recursive_lookups += 1
        return original_lookup(args)

    _mock_active_aot(monkeypatch, data_dir=tmp_path, hardware=B200)
    monkeypatch.setattr(aot_cache_module, "find_heuristic_file", find_heuristic_file)
    monkeypatch.setattr(
        argument_device_module,
        "_find_argument_device_with_path",
        counted_lookup,
    )
    key_fn = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=itemgetter("shape"),
    )

    def payload(shape: int, tensor: torch.Tensor) -> dict[str, object]:
        return {
            "shape": shape,
            "metadata": [None, {"runtime": {"tensor": tensor}}],
        }

    assert key_fn(payload(7, tensors[0])) == (
        "helion_aot_heuristic",
        str(heuristic_paths[devices[0]]),
        7,
    )
    assert key_fn(payload(11, tensors[1])) == (
        "helion_aot_heuristic",
        str(heuristic_paths[devices[1]]),
        11,
    )
    for shape in range(10_000):
        key_fn(payload(shape, tensors[shape % 2]))

    # This is a deterministic hot-path microbenchmark: only the first call may
    # recursively discover the nested device. Every later call indexes the
    # learned path directly, including calls that switch devices.
    assert recursive_lookups == 1
    assert discovered_devices == list(devices)


def test_heuristic_key_canonicalizes_indexless_device_before_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    hardware_by_device = {
        devices[0]: B200,
        devices[1]: GB300,
    }
    heuristic_paths = {
        device: tmp_path / f"heuristic_{device.index}.py" for device in devices
    }
    for heuristic_path in heuristic_paths.values():
        heuristic_path.write_text("def key_demo(shape, device):\n    return shape\n")
    hardware_lookups: list[torch.device | None] = []
    artifact_lookups: list[torch.device | None] = []

    def get_hardware_info(device: torch.device | None = None) -> HardwareInfo:
        hardware_lookups.append(device)
        assert device is not None
        return hardware_by_device[device]

    def find_heuristic_file(
        _source_file: str,
        *,
        kernel_name: str,
        data_dir: Path,
        device: torch.device | None,
        resolved_heuristic_dir: Path | None = None,
    ) -> Path:
        assert kernel_name == "demo"
        assert data_dir == tmp_path
        artifact_lookups.append(device)
        assert device is not None
        return heuristic_paths[device]

    _mock_active_aot(monkeypatch, data_dir=tmp_path)
    monkeypatch.setattr(aot_cache_module, "get_hardware_info", get_hardware_info)
    monkeypatch.setattr(aot_cache_module, "find_heuristic_file", find_heuristic_file)
    current_index = [0, 1]
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.current_device", lambda: current_index.pop(0))
    key_fn = HeuristicKeyFunction(str(source_path), "demo")
    indexless = torch.device("cuda")

    assert key_fn(7, indexless) == (
        "helion_aot_heuristic",
        str(heuristic_paths[devices[0]]),
        7,
    )
    assert key_fn(11, indexless) == (
        "helion_aot_heuristic",
        str(heuristic_paths[devices[1]]),
        11,
    )
    assert hardware_lookups == list(devices)
    assert artifact_lookups == list(devices)


def test_heuristic_key_constexpr_tensor_matches_kernel_device_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    heuristic_path = tmp_path / "_helion_aot_kernel_cuda_sm100.py"
    heuristic_path.write_text("def key_demo(constant, tensor):\n    return 23\n")
    cpu_tensor = torch.empty(0)
    with FakeTensorMode():
        cuda_tensor = torch.empty(0, device=torch.device("cuda", 0))
    args = (hl.constexpr(cpu_tensor), cuda_tensor)
    cuda = torch.device("cuda:0")
    observed_devices: list[torch.device | None] = []

    def get_hardware_info(device: torch.device | None = None) -> HardwareInfo:
        observed_devices.append(device)
        assert device == cuda
        return B200

    _mock_active_aot(monkeypatch)
    monkeypatch.setattr(aot_cache_module, "get_hardware_info", get_hardware_info)
    assert _find_device(args) == cuda
    key_fn = HeuristicKeyFunction(str(source_path), "demo")
    assert key_fn(*args) == (
        "helion_aot_heuristic",
        str(heuristic_path.resolve()),
        23,
    )
    assert observed_devices == [cuda, cuda]


def test_learned_device_path_matches_kernel_traversal_and_invalidates() -> None:
    cpu_tensor = torch.empty(0)
    meta_tensor = torch.empty(0, device=torch.device("meta"))
    with FakeTensorMode():
        cuda_tensor = torch.empty(0, device=torch.device("cuda", 0))
    cases: tuple[tuple[object, ...], ...] = (
        (torch.device("cpu"), meta_tensor),
        (cpu_tensor, torch.device("meta")),
        (hl.constexpr(cpu_tensor), cuda_tensor),
        ([torch.device("cuda:1")], torch.device("cuda:0")),
        ({"outer": [None, torch.device("meta")]}, torch.device("cpu")),
        ([None, {"constant": hl.constexpr(cpu_tensor), "tensor": cuda_tensor}],),
        ({"empty": [None, 7]},),
    )

    for args in cases:
        expected = _expected_argument_device(args)
        discovered = argument_device_module._find_argument_device_with_path(args)
        assert (None if discovered is None else discovered[0]) == expected
        if discovered is not None:
            device, path = discovered
            assert argument_device_module._device_at_path(args, path) == device

    initial_args = (None, {"later": [torch.device("meta")]})
    initial = argument_device_module._find_argument_device_with_path(initial_args)
    assert initial is not None
    _, learned_path = initial

    earlier_device_args = (
        torch.device("cpu"),
        {"later": [torch.device("meta")]},
    )
    assert (
        argument_device_module._device_at_path(earlier_device_args, learned_path)
        is None
    )
    rediscovered = argument_device_module._find_argument_device_with_path(
        earlier_device_args
    )
    assert rediscovered is not None
    assert rediscovered[0] == _expected_argument_device(earlier_device_args)

    changed_structure_args = (None, {"later": {"device": torch.device("meta")}})
    assert (
        argument_device_module._device_at_path(changed_structure_args, learned_path)
        is None
    )
    rediscovered = argument_device_module._find_argument_device_with_path(
        changed_structure_args
    )
    assert rediscovered is not None
    assert rediscovered[0] == _expected_argument_device(changed_structure_args)


def test_heuristic_key_rejects_in_process_aot_mode_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    monkeypatch.setenv("HELION_AOT_MODE", "disabled")
    key_fn = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=lambda value: value,
    )

    assert key_fn(7) == 7
    HeuristicKeyFunction.clear_cache()
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    with pytest.raises(
        RuntimeError,
        match="HELION_AOT_MODE changed from 'disabled' to 'evaluate'.*fresh process",
    ):
        key_fn(7)


def _direct_bound_stub(kernel: object) -> Any:
    return SimpleNamespace(
        kernel=kernel,
        _cache_managed=False,
        _compiler_seed_specialization_extractors=(),
        _run=lambda value: value,
    )


def test_direct_aot_bound_call_rejects_aot_mode_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def demo(value: int) -> int:
        return value

    monkeypatch.setenv("HELION_AOT_MODE", "disabled")
    kernel = aot_kernel(demo)
    assert kernel._key_fn is not None
    kernel._key_fn(7)
    bound = _direct_bound_stub(kernel)

    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    with pytest.raises(
        RuntimeError,
        match="HELION_AOT_MODE changed from 'disabled' to 'evaluate'.*fresh process",
    ):
        BoundKernel.__call__(bound, 7)


def test_direct_aot_bound_call_rejects_active_data_dir_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def demo(value: int) -> int:
        return value

    first_data_dir = tmp_path / "first"
    second_data_dir = tmp_path / "second"
    monkeypatch.setenv("HELION_AOT_MODE", "collect")
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(first_data_dir))
    kernel = aot_kernel(demo)
    assert kernel._key_fn is not None
    kernel._key_fn(7)
    bound = _direct_bound_stub(kernel)

    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(second_data_dir))
    with pytest.raises(
        RuntimeError, match="HELION_AOT_DATA_DIR setting changed.*fresh process"
    ):
        BoundKernel.__call__(bound, 7)


def test_direct_disabled_aot_bound_call_ignores_data_dir_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_calls = 0

    def demo(value: int) -> int:
        return value

    def user_key(value: int) -> int:
        nonlocal key_calls
        key_calls += 1
        return value

    monkeypatch.setenv("HELION_AOT_MODE", "disabled")
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path / "first"))
    kernel = aot_kernel(demo, key=user_key)
    assert kernel._key_fn is not None
    kernel._key_fn(7)
    assert key_calls == 1
    bound = _direct_bound_stub(kernel)

    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path / "second"))
    assert BoundKernel.__call__(bound, 7) == 7
    assert key_calls == 1


def test_disabled_heuristic_key_ignores_aot_data_dir_and_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    first_cwd = tmp_path / "first-cwd"
    second_cwd = tmp_path / "second-cwd"
    first_cwd.mkdir()
    second_cwd.mkdir()
    monkeypatch.setenv("HELION_AOT_MODE", "disabled")
    monkeypatch.setenv("HELION_AOT_DATA_DIR", "~missing-user/aot")
    monkeypatch.chdir(first_cwd)
    monkeypatch.setattr(
        aot_cache_module,
        "get_aot_data_dir",
        lambda: pytest.fail("disabled mode must not resolve the AOT data directory"),
    )
    key_fn = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=lambda value: value,
    )

    assert key_fn(7) == 7
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path / "changed"))
    monkeypatch.chdir(second_cwd)
    assert key_fn(11) == 11


def test_heuristic_key_resolves_aot_data_dir_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    data_dir_calls = 0

    def get_data_dir() -> Path:
        nonlocal data_dir_calls
        data_dir_calls += 1
        return tmp_path

    _mock_active_aot(monkeypatch)
    monkeypatch.setattr(aot_cache_module, "get_aot_data_dir", get_data_dir)
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: B200,
    )
    monkeypatch.setattr(
        aot_cache_module,
        "find_heuristic_file",
        lambda *_args, **_kwargs: None,
    )
    key_fn = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=lambda value: value,
    )

    assert key_fn(7) == 7
    assert data_dir_calls == 1
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self, *args, **kwargs: pytest.fail(
            "steady-state dispatch must not re-resolve the AOT data directory"
        ),
    )
    assert key_fn(11) == 11
    assert data_dir_calls == 1


def test_heuristic_key_resolves_override_and_imports_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    heuristic_dir = tmp_path / "heuristics"
    heuristic_dir.mkdir()
    load_marker = tmp_path / "loaded.txt"
    heuristic_file = heuristic_dir / "heuristic_demo.py"
    heuristic_file.write_text(
        f"with open({str(load_marker)!r}, 'a') as marker:\n"
        "    marker.write('x')\n"
        "def key_demo(value):\n"
        "    return value // 2\n"
    )
    heuristic_identity = str(heuristic_file.resolve())
    override_resolves = 0
    original_resolve = Path.resolve

    def counted_resolve(path: Path, strict: bool = False) -> Path:
        nonlocal override_resolves
        if path == heuristic_dir:
            override_resolves += 1
        return original_resolve(path, strict=strict)

    _mock_active_aot(monkeypatch)
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path / "aot-data"))
    monkeypatch.setenv("HELION_HEURISTIC_DIR", str(heuristic_dir))
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: B200,
    )
    monkeypatch.setattr(Path, "resolve", counted_resolve)
    key_fn = HeuristicKeyFunction(str(source_path), "demo")
    assert key_fn(8) == (
        "helion_aot_heuristic",
        heuristic_identity,
        4,
    )
    assert key_fn(10) == (
        "helion_aot_heuristic",
        heuristic_identity,
        5,
    )
    assert override_resolves == 1
    assert load_marker.read_text() == "x"


def test_heuristic_key_relative_override_tracks_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    for cwd, result in ((first_cwd, 1), (second_cwd, 2)):
        heuristic_dir = cwd / "heuristics"
        heuristic_dir.mkdir(parents=True)
        (heuristic_dir / "heuristic_demo.py").write_text(
            f"def key_demo(value):\n    return {result}\n"
        )

    _mock_active_aot(monkeypatch)
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path / "aot-data"))
    monkeypatch.setenv("HELION_HEURISTIC_DIR", "heuristics")
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: B200,
    )
    key_fn = HeuristicKeyFunction(str(source_path), "demo")
    monkeypatch.chdir(first_cwd)
    first = key_fn(7)
    monkeypatch.chdir(second_cwd)
    second = key_fn(7)

    assert isinstance(first, tuple)
    assert isinstance(second, tuple)
    assert first[-1] == 1
    assert second[-1] == 2
    assert first[1] != second[1]


def test_heuristic_key_pins_default_data_dir_at_first_active_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    first_cwd = tmp_path / "first-cwd"
    second_cwd = tmp_path / "second-cwd"
    first_cwd.mkdir()
    second_cwd.mkdir()
    observed_data_dirs: list[Path] = []

    def find_heuristic_file(
        _source_file: str,
        *,
        kernel_name: str,
        data_dir: Path,
        device: torch.device | None,
        resolved_heuristic_dir: Path | None = None,
    ) -> None:
        assert kernel_name == "demo"
        assert device is not None
        observed_data_dirs.append(data_dir)
        return None

    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    monkeypatch.delenv("HELION_AOT_DATA_DIR", raising=False)
    monkeypatch.chdir(first_cwd)
    monkeypatch.setattr(
        aot_cache_module,
        "get_heuristic_hardware",
        lambda device: B200,
    )
    monkeypatch.setattr(aot_cache_module, "find_heuristic_file", find_heuristic_file)
    key_fn = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=lambda value, device: value,
    )

    assert key_fn(7, torch.device("cpu")) == 7
    monkeypatch.chdir(second_cwd)
    assert key_fn(11, torch.device("meta")) == 11
    assert observed_data_dirs == [first_cwd / ".helion_aot"] * 2


@pytest.mark.parametrize("aot_mode", ("collect", "measure", "evaluate", "compile"))
def test_heuristic_key_rejects_data_dir_change_in_other_active_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aot_mode: str,
) -> None:
    source_path = _kernel_source(tmp_path)
    first_data_dir = tmp_path / "first"
    second_data_dir = tmp_path / "second"
    if aot_mode == "evaluate":
        for data_dir, result in ((first_data_dir, 1), (second_data_dir, 2)):
            data_dir.mkdir()
            aot_cache_module.write_hardware_manifest(data_dir, B200)
            (data_dir / "heuristic_demo.py").write_text(
                f"def key_demo(value):\n    return {result}\n"
            )
    monkeypatch.setenv("HELION_AOT_MODE", aot_mode)
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(first_data_dir))
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: B200,
    )
    if aot_mode != "evaluate":
        monkeypatch.setattr(
            aot_cache_module,
            "find_heuristic_file",
            lambda *_args, **_kwargs: None,
        )
    key_fn = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=lambda value: value,
    )

    first_result = key_fn(7)
    if aot_mode == "evaluate":
        assert first_result == (
            "helion_aot_heuristic",
            str((first_data_dir / "heuristic_demo.py").resolve()),
            1,
        )
    else:
        assert first_result == 7
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(second_data_dir))
    with pytest.raises(
        RuntimeError,
        match=(
            "HELION_AOT_DATA_DIR setting changed from .*first.* to .*second.*"
            ".*fresh process"
        ),
    ):
        key_fn(7)


def test_heuristic_key_device_path_recovers_from_structure_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    device = torch.device("cuda:0")
    recursive_lookups = 0
    original_lookup = argument_device_module._find_argument_device_with_path

    def counted_lookup(
        args: tuple[object, ...],
    ) -> tuple[torch.device, object] | None:
        nonlocal recursive_lookups
        recursive_lookups += 1
        return original_lookup(args)

    _mock_active_aot(monkeypatch, data_dir=tmp_path, hardware=B200)
    monkeypatch.setattr(
        aot_cache_module,
        "find_heuristic_file",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        argument_device_module,
        "_find_argument_device_with_path",
        counted_lookup,
    )
    key_fn = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=itemgetter("shape"),
    )

    assert key_fn({"shape": 7, "nested": [None, device]}) == 7
    assert key_fn({"shape": 11, "nested": {"device": device}}) == 11
    assert key_fn({"shape": 13, "nested": {"device": device}}) == 13

    # Switching the cached list hop to a mapping invalidates the old path,
    # triggers one safe rediscovery, and then becomes direct again.
    assert recursive_lookups == 2


def test_heuristic_key_device_path_recovers_when_earlier_device_appears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    earlier_device = torch.device("cpu")
    later_device = torch.device("meta")
    for device in (earlier_device, later_device):
        (tmp_path / f"heuristic_{device.type}.py").write_text("")
    discovered_devices: list[torch.device | None] = []

    def find_heuristic_file(
        _source_file: str,
        *,
        kernel_name: str,
        data_dir: Path,
        device: torch.device | None,
        resolved_heuristic_dir: Path | None = None,
    ) -> Path:
        assert kernel_name == "demo"
        assert data_dir == tmp_path
        discovered_devices.append(device)
        assert device is not None
        return tmp_path / f"heuristic_{device.type}.py"

    _mock_active_aot(monkeypatch, data_dir=tmp_path, hardware=B200)
    monkeypatch.setattr(aot_cache_module, "find_heuristic_file", find_heuristic_file)
    key_fn = HeuristicKeyFunction(
        str(source_path),
        "demo",
        user_key=lambda optional, fallback: 7,
    )

    assert key_fn(None, later_device) == (
        "helion_aot_heuristic",
        str(tmp_path / "heuristic_meta.py"),
        7,
    )
    assert key_fn(earlier_device, later_device) == (
        "helion_aot_heuristic",
        str(tmp_path / "heuristic_cpu.py"),
        7,
    )
    assert discovered_devices == [later_device, earlier_device]


def test_heuristic_key_keeps_concurrent_device_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _kernel_source(tmp_path)
    sm100_path = tmp_path / "_helion_aot_kernel_cuda_sm100.py"
    sm90_path = tmp_path / "_helion_aot_kernel_cuda_sm90.py"

    class SupportModule(ModuleType):
        entered: threading.Event
        release: threading.Event

    support_name = "_helion_aot_concurrent_key_test"
    support = SupportModule(support_name)
    entered = threading.Event()
    release = threading.Event()
    support.entered = entered
    support.release = release
    monkeypatch.setitem(sys.modules, support_name, support)
    sm100_path.write_text(
        f"from {support_name} import entered, release\n"
        "def key_demo(value, device):\n"
        "    entered.set()\n"
        "    assert release.wait(5)\n"
        "    return value\n"
    )
    sm90_path.write_text("def key_demo(value, device):\n    return value\n")

    sm100_device = torch.device("cuda:0")
    sm90_device = torch.device("cuda:1")
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: (
            HardwareInfo("cuda", "B200", "13.0", "sm100")
            if device == sm100_device
            else HardwareInfo("cuda", "H100", "13.0", "sm90")
        ),
    )
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    try:
        key_fn = HeuristicKeyFunction(str(source_path), "demo")
        with ThreadPoolExecutor(max_workers=1) as pool:
            sm100_result = pool.submit(key_fn, 100, sm100_device)
            assert entered.wait(5)
            sm90_result = key_fn(90, sm90_device)
            release.set()

        assert sm100_result.result() == (
            "helion_aot_heuristic",
            str(sm100_path.resolve()),
            100,
        )
        assert sm90_result == (
            "helion_aot_heuristic",
            str(sm90_path.resolve()),
            90,
        )
    finally:
        release.set()


def test_static_standalone_call_key_covers_runtime_specialization() -> None:
    class Point(NamedTuple):
        x: int
        y: int

    tensor = torch.empty_strided((2, 3), (4, 1))
    base = _standalone_call_key((tensor, (1, 2), torch.float16, torch.device("cpu")))

    assert base != _standalone_call_key(
        (
            torch.empty_strided((3, 2), (2, 1)),
            (1, 2),
            torch.float16,
            torch.device("cpu"),
        )
    )
    assert base != _standalone_call_key(
        (tensor.as_strided((2, 3), (3, 1)), (1, 2), torch.float16, torch.device("cpu"))
    )
    assert base != _standalone_call_key(
        (tensor.to(torch.float64), (1, 2), torch.float16, torch.device("cpu"))
    )
    assert base != _standalone_call_key(
        (tensor, (1, 3), torch.float16, torch.device("cpu"))
    )
    assert _standalone_call_key(({"a": 1, "b": 2},)) == _standalone_call_key(
        ({"b": 2, "a": 1},)
    )
    with pytest.raises(TypeError, match="does not support object"):
        _standalone_call_key((object(),))
    with pytest.raises(TypeError, match="Point"):
        _standalone_call_key((Point(1, 2),))


def test_non_file_aot_kernels_use_distinct_code_identities() -> None:
    def load_kernel(source: str) -> Any:
        namespace: dict[str, object] = {}
        exec(compile(source, "<string>", "exec"), namespace)
        return namespace["demo"]

    first = aot_kernel(load_kernel("def demo(value):\n    return value\n"))
    second = aot_kernel(load_kernel("def demo(value):\n    return value + 1\n"))
    assert isinstance(first._key_fn, HeuristicKeyFunction)
    assert isinstance(second._key_fn, HeuristicKeyFunction)
    assert first._key_fn.kernel_source_file is None
    assert second._key_fn.kernel_source_file is None
    assert (
        first._key_fn._kernel_source_identity != second._key_fn._kernel_source_identity
    )


def test_aot_cache_canonicalizes_defaults_for_compile_get(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.empty(2)
    signature = inspect.signature(lambda x, metadata=(1, 2): x)

    def normalize_args(*args: object) -> tuple[object, ...]:
        bound = signature.bind(*args)
        bound.apply_defaults()
        return tuple(bound.args)

    def demo(x: torch.Tensor, metadata: tuple[int, int] = (1, 2)) -> torch.Tensor:
        return x

    kernel_api = SimpleNamespace(
        __code__=demo.__code__,
        name="demo",
        normalize_args=normalize_args,
        specialization_key=lambda args: tuple(args),
        _aot_collect_fn=None,
        _aot_measure_fn=None,
        _aot_user_key=None,
        _aot_workflow_done=False,
    )
    bound_kernel = SimpleNamespace(
        kernel=kernel_api,
        env=SimpleNamespace(device=torch.device("cuda:3")),
        is_cacheable=lambda: True,
    )
    autotuner = SimpleNamespace(kernel=bound_kernel, args=(tensor,))
    monkeypatch.setenv("HELION_AOT_MODE", "compile")
    monkeypatch.setattr(aot_cache_module, "get_aot_data_dir", lambda: tmp_path)
    observed_devices: list[torch.device | None] = []

    def hardware_for(device: torch.device | None = None) -> SimpleNamespace:
        observed_devices.append(device)
        return SimpleNamespace(hardware_id="test-hardware")

    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        hardware_for,
    )

    cache = AOTAutotuneCache(autotuner)
    assert cache.args == (tensor, (1, 2))
    assert observed_devices == [torch.device("cuda:3")]
    compiled: list[bool] = []
    selected_args: list[tuple[object, ...]] = []
    config = Config(block_sizes=[16])
    cache._maybe_run_compile = lambda: compiled.append(True)

    def get_config(args: tuple[object, ...]) -> Config:
        selected_args.append(args)
        return config

    cache._get_heuristic_config = get_config
    assert cache.get() is config
    assert compiled == [True]
    assert selected_args == [(tensor, (1, 2))]


def test_compile_mode_skips_non_standalone_kernel() -> None:
    def demo(x: torch.Tensor) -> torch.Tensor:
        return x

    assert aot_kernel(demo)._aot_standalone is True
    assert aot_kernel(demo, standalone=False)._aot_standalone is False

    cache = object.__new__(AOTAutotuneCache)
    cache.mode = "compile"
    cache.args = ()
    cache.kernel = SimpleNamespace(
        kernel=SimpleNamespace(name="demo", _aot_standalone=False)
    )
    expected = Config(block_sizes=[16])
    cache._get_heuristic_config = lambda args: expected

    assert cache.get() is expected


def test_standalone_preserves_cute_launcher_import(tmp_path: Path) -> None:
    output = generate_standalone_file(
        "demo",
        [
            (
                "from __future__ import annotations\n"
                "from helion.runtime import default_cute_launcher as "
                "_default_cute_launcher\n\n"
                "def demo(x, *, _launcher=_default_cute_launcher):\n"
                "    return x\n"
            )
        ],
        "",
        tmp_path,
    )

    source = output.read_text()
    assert (
        "from helion.runtime import default_cute_launcher as _default_cute_launcher"
        in source
    )


def test_dynamic_compile_cache_uses_source_hardware_and_heuristic_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_kernel(path: Path) -> Any:
        path.write_text("def demo(x):\n    return x\n")
        namespace: dict[str, Any] = {}
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        return namespace["demo"]

    first_kernel = load_kernel(tmp_path / "first_source.py")
    second_kernel = load_kernel(tmp_path / "second_source.py")
    first_heuristic_dir = tmp_path / "first_heuristic"
    second_heuristic_dir = tmp_path / "second_heuristic"
    first_heuristic_dir.mkdir()
    second_heuristic_dir.mkdir()
    for heuristic_dir, block_size in (
        (first_heuristic_dir, 16),
        (second_heuristic_dir, 32),
    ):
        (heuristic_dir / "heuristic_demo.py").write_text(
            f"CONFIGS = [{{'block_sizes': [{block_size}]}}]\n"
            "def autotune_demo(*args):\n"
            "    return CONFIGS[0]\n"
        )

    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: (
            HardwareInfo("cuda", "B200", "13.0", "sm100")
            if device == torch.device("cuda:0")
            else HardwareInfo("cuda", "H100", "13.0", "sm90")
        ),
    )
    monkeypatch.setenv("HELION_AOT_MODE", "compile")
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path / "aot_data"))
    compiled: list[str] = []

    def make_cache(
        kernel_function: Any,
        device: torch.device,
        heuristic_dir: Path,
        label: str,
    ) -> AOTAutotuneCache:
        def to_triton_code(_config: Config) -> str:
            compiled.append(label)
            return "def demo(x):\n    return x\n"

        kernel_api = SimpleNamespace(
            __code__=kernel_function.__code__,
            name="demo",
            normalize_args=lambda *args: args,
            specialization_key=lambda args: tuple(args),
            _aot_collect_fn=None,
            _aot_measure_fn=None,
            _aot_standalone=True,
            _aot_user_key=None,
            _aot_workflow_done=False,
        )
        bound_kernel = SimpleNamespace(
            kernel=kernel_api,
            env=SimpleNamespace(device=device),
            is_cacheable=lambda: True,
            settings=SimpleNamespace(static_shapes=False),
            to_triton_code=to_triton_code,
        )
        autotuner = SimpleNamespace(
            kernel=bound_kernel,
            args=(1,),
            config_spec=SimpleNamespace(
                default_config=lambda: Config(block_sizes=[1]),
            ),
            settings=SimpleNamespace(ignore_warnings=set()),
        )
        monkeypatch.setenv("HELION_HEURISTIC_DIR", str(heuristic_dir))
        return AOTAutotuneCache(autotuner)

    cases = (
        (first_kernel, torch.device("cuda:0"), first_heuristic_dir, "first"),
        (first_kernel, torch.device("cuda:0"), first_heuristic_dir, "duplicate"),
        (first_kernel, torch.device("cuda:0"), second_heuristic_dir, "heuristic"),
        (first_kernel, torch.device("cuda:1"), first_heuristic_dir, "hardware"),
        (second_kernel, torch.device("cuda:0"), first_heuristic_dir, "source"),
    )
    selected = [make_cache(*case).get() for case in cases]

    assert compiled == ["first", "heuristic", "hardware", "source"]
    assert [config["block_sizes"] for config in selected if config is not None] == [
        [16],
        [16],
        [32],
        [16],
        [16],
    ]


def test_static_aot_compile_accumulates_observed_shapes(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("def demo(x, metadata=(1, 2)):\n    return x\n")
    linked_source = tmp_path / "linked.py"
    linked_source.symlink_to(source_path)
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(linked_source), "exec"), namespace)
    kernel_function = namespace["demo"]
    signature = inspect.signature(kernel_function)

    def normalize_args(*args: object) -> tuple[object, ...]:
        bound = signature.bind(*args)
        bound.apply_defaults()
        return tuple(bound.args)

    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text("def autotune_demo(*args):\n    return {}\n")
    output_path = tmp_path / "source_demo_standalone.py"
    config = Config(block_sizes=[16])

    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = tmp_path
    cache.hardware_id = "test-hardware"
    cache._kernel_source_file = canonical_kernel_source_path(str(linked_source))
    cache.args = (torch.empty(2),)

    def get_heuristic_config(args: tuple[object, ...]) -> Config:
        assert args[1] == (1, 2)
        return config

    cache._get_heuristic_config = get_heuristic_config

    def to_triton_code(_config: Config) -> str:
        size = cache.args[0].size(0)
        return (
            "from __future__ import annotations\n\n"
            "def demo(x, metadata=(1, 2)):\n"
            f"    return {size}\n"
        )

    cache.kernel = SimpleNamespace(
        kernel=SimpleNamespace(
            __code__=kernel_function.__code__,
            name="demo",
            normalize_args=normalize_args,
        ),
        to_triton_code=to_triton_code,
    )

    def compile_shapes(sizes: tuple[int, ...]) -> str:
        AOTAutotuneCache.clear_caches()
        for size in sizes:
            cache.args = (torch.empty(size),)
            cache._compile_current_static_shape(heuristic_path, "demo")
            assert output_path.exists()
        return output_path.read_text()

    forward_source = compile_shapes((2, 3))
    reverse_source = compile_shapes((3, 2))
    assert forward_source == reverse_source
    module = _load_generated(output_path, "test_static_aot_compile")
    assert module.demo(torch.empty(2)) == 2
    assert module.demo(x=torch.empty(2)) == 2
    assert module.demo(torch.empty(3)) == 3
    with pytest.raises(ValueError, match="No standalone variant"):
        module.demo(torch.empty(4))

    cache.kernel.to_triton_code = lambda _config: (
        "from __future__ import annotations\n\ndef demo(x):\n    return 99\n"
    )
    with pytest.raises(RuntimeError, match="value-derived compile-time metadata"):
        cache._compile_current_static_shape(heuristic_path, "demo")

    prior_source = output_path.read_text()
    cache.args = (torch.empty(4),)

    def fail_compile(_config: Config) -> str:
        raise ValueError("compile failed")

    cache.kernel.to_triton_code = fail_compile
    with pytest.raises(RuntimeError, match="variant failed to compile"):
        cache._compile_current_static_shape(heuristic_path, "demo")
    assert output_path.read_text() == prior_source


def test_static_aot_compile_supports_non_file_kernel_source(tmp_path: Path) -> None:
    namespace: dict[str, object] = {}
    exec(compile("def demo(x):\n    return x\n", "<stdin>", "exec"), namespace)
    kernel_function = namespace["demo"]
    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text("def autotune_demo(*args):\n    return {}\n")
    config = Config(block_sizes=[16])

    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = tmp_path
    cache.hardware_id = "test-hardware"
    cache._kernel_source_file = canonical_kernel_source_path("<stdin>")
    cache.args = (torch.empty(2),)
    cache._get_heuristic_config = lambda _args: config
    cache.kernel = SimpleNamespace(
        kernel=SimpleNamespace(
            __code__=kernel_function.__code__,
            name="demo",
            normalize_args=lambda *args: tuple(args),
        ),
        to_triton_code=lambda _config: (
            "from __future__ import annotations\n\ndef demo(x):\n    return 2\n"
        ),
    )

    cache._compile_current_static_shape(heuristic_path, "demo")
    output_path = tmp_path / "demo_standalone.py"
    assert output_path.exists()
    module = _load_generated(output_path, "test_static_aot_non_file")
    assert module.demo(torch.empty(2)) == 2

    other_dir = tmp_path / "other"
    cache.data_dir = other_dir
    cache._compile_current_static_shape(heuristic_path, "demo")
    other_output = other_dir / "demo_standalone.py"
    assert other_output.exists()


def test_static_aot_compile_serializes_concurrent_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("def demo(x):\n    return x\n")
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    kernel_function = namespace["demo"]
    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text("def autotune_demo(*args):\n    return {}\n")
    config = Config(block_sizes=[16])
    codegen_barrier = threading.Barrier(2)

    def make_cache(size: int) -> AOTAutotuneCache:
        cache = object.__new__(AOTAutotuneCache)
        cache.data_dir = tmp_path
        cache.hardware_id = "test-hardware"
        cache._kernel_source_file = canonical_kernel_source_path(source_path)
        cache.args = (torch.empty(size),)
        cache._get_heuristic_config = lambda _args: config

        def to_triton_code(_config: Config) -> str:
            codegen_barrier.wait(timeout=5)
            return (
                "from __future__ import annotations\n\n"
                f"def demo(x):\n    return {size}\n"
            )

        cache.kernel = SimpleNamespace(
            kernel=SimpleNamespace(
                __code__=kernel_function.__code__,
                name="demo",
                normalize_args=lambda *args: tuple(args),
            ),
            to_triton_code=to_triton_code,
        )
        return cache

    original_generate = aot_cache_module.generate_standalone_file
    active_writers = 0
    max_active_writers = 0
    writers_lock = threading.Lock()

    def tracked_generate(
        kernel_name: str,
        triton_codes: list[str],
        heuristic_code: str,
        output_dir: Path,
        kernel_source_file: str | None = None,
        dispatch_keys: list[tuple[object, ...]] | None = None,
    ) -> Path:
        nonlocal active_writers, max_active_writers
        with writers_lock:
            active_writers += 1
            max_active_writers = max(max_active_writers, active_writers)
        try:
            time.sleep(0.05)
            return original_generate(
                kernel_name,
                triton_codes,
                heuristic_code,
                output_dir,
                kernel_source_file,
                dispatch_keys,
            )
        finally:
            with writers_lock:
                active_writers -= 1

    monkeypatch.setattr(aot_cache_module, "generate_standalone_file", tracked_generate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                make_cache(size)._compile_current_static_shape,
                heuristic_path,
                "demo",
            )
            for size in (2, 3)
        ]
        for future in futures:
            future.result()

    assert max_active_writers == 1
    module = _load_generated(
        tmp_path / "source_demo_standalone.py",
        "test_static_aot_concurrent",
    )
    assert module.demo(torch.empty(2)) == 2
    assert module.demo(torch.empty(3)) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
