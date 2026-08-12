"""
Tests for AOT Autotuning Framework
==================================

Tests for the collect/measure/evaluate workflow.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import inspect
import os
import sys
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from unittest.mock import patch

import numpy as np
import pytest
import torch

from helion._hardware import HardwareInfo
from helion._testing import onlyBackends
import helion.autotuner.aot_cache as aot_cache_module
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.aot_cache import ShapeKey
from helion.autotuner.aot_cache import _deserialize_tuple
from helion.autotuner.aot_cache import _serialize_tuple
from helion.autotuner.aot_cache import get_aot_mode
from helion.autotuner.aot_compile import _standalone_call_key
from helion.autotuner.aot_compile import generate_standalone_file
from helion.autotuner.aot_kernel import aot_key
from helion.autotuner.aot_kernel import extract_shape_features
from helion.autotuner.heuristic_generator import PerformanceTarget
from helion.autotuner.heuristic_generator import ShapeConfigData
from helion.autotuner.heuristic_generator import compute_validity_partitions
from helion.autotuner.heuristic_generator import select_config_subset
from helion.runtime.config import Config

if TYPE_CHECKING:
    from pathlib import Path


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

    def test_all_configs_valid_no_partitioning(self) -> None:
        """Single partition when all configs are valid for all shapes."""
        data = ShapeConfigData(
            kernel_name="test",
            shape_features=[{"dim": 1024}, {"dim": 2048}],
            timings=np.array(
                [
                    [1.0, 2.0],
                    [1.0, 2.0],
                ]
            ),
            configs=[Config(block_sizes=[64]), Config(block_sizes=[128])],
            shape_hashes=["s1", "s2"],
            config_hashes=["c1", "c2"],
        )

        target = PerformanceTarget(
            goal_type="max_slowdown", threshold=1.1, verbose=False
        )
        selected, stats = select_config_subset(data, target)

        assert stats["num_partitions"] == 1
        assert len(selected) == 1
        assert selected[0] == 0  # Config 0 is best

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

    kernel_api = SimpleNamespace(
        name="demo",
        normalize_args=normalize_args,
        specialization_key=lambda args: tuple(args),
    )
    bound_kernel = SimpleNamespace(kernel=kernel_api, is_cacheable=lambda: True)
    autotuner = SimpleNamespace(kernel=bound_kernel, args=(tensor,))
    monkeypatch.setenv("HELION_AOT_MODE", "compile")
    monkeypatch.setattr(aot_cache_module, "get_aot_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda: SimpleNamespace(hardware_id="test-hardware"),
    )

    cache = AOTAutotuneCache(autotuner)
    assert cache.args == (tensor, (1, 2))
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


def test_static_aot_compile_accumulates_observed_shapes(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("def demo(x, metadata=(1, 2)):\n    return x\n")
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
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
    AOTAutotuneCache.clear_caches()


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

    AOTAutotuneCache.clear_caches()
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
    AOTAutotuneCache.clear_caches()


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

    import helion.autotuner.aot_compile as aot_compile

    original_generate = aot_compile.generate_standalone_file
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

    monkeypatch.setattr(aot_compile, "generate_standalone_file", tracked_generate)
    AOTAutotuneCache.clear_caches()
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
    AOTAutotuneCache.clear_caches()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
