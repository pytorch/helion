"""
Tests for AOT Autotuning Framework
==================================

Tests for the collect/measure/evaluate workflow.
"""

from __future__ import annotations

import ast
import builtins
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import partial
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import struct
import sys
import threading
import time
from types import ModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
import torch

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.common import matches_hardware
from helion._hardware import HardwareInfo
from helion._hardware import get_hardware_info
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import load_generated_module
from helion._testing.triton_codegen import interpret_triton_kernel
from helion.autotuner import BooleanFragment
from helion.autotuner import EnumFragment
import helion.autotuner.aot_cache as aot_cache_module
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.aot_cache import ShapeKey
from helion.autotuner.aot_cache import _deserialize_tuple
from helion.autotuner.aot_cache import _deserialize_value
from helion.autotuner.aot_cache import _serialize_tuple
from helion.autotuner.aot_cache import _serialize_value
from helion.autotuner.aot_cache import get_aot_mode
from helion.autotuner.aot_compile import _standalone_call_key
from helion.autotuner.aot_compile import generate_standalone_file
from helion.autotuner.aot_kernel import aot_key
from helion.autotuner.aot_kernel import extract_shape_features
from helion.autotuner.base_cache import AutotuneCacheBase
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.finite_search import FiniteSearch
from helion.autotuner.heuristic_generator import PerformanceTarget
from helion.autotuner.heuristic_generator import ShapeConfigData
from helion.autotuner.heuristic_generator import compute_validity_partitions
from helion.autotuner.heuristic_generator import select_config_subset
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
import helion.language as hl
from helion.runtime.config import Config
from helion.runtime.kernel import _constexpr_key
from helion.runtime.settings import Settings
from helion.runtime.settings import default_autotuner_fn
from helion.runtime.structural_policy import StructuralPolicyError

pytestmark = pytest.mark.cpu_only

_STANDALONE_KEY_LIMIT = 3


def test_future_cuda_arch_does_not_load_sm103_aot_fallback(tmp_path: Path) -> None:
    source_path = tmp_path / "demo.py"
    source_path.touch()
    (tmp_path / "_helion_aot_demo_cuda_sm103.py").touch()
    sm100_heuristic = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    sm100_heuristic.touch()
    future_hardware = HardwareInfo(
        device_kind="cuda",
        hardware_name="NVIDIA future GPU",
        runtime_version="14.0",
        compute_capability="sm120",
    )

    aot_cache_module.clear_heuristic_cache()
    try:
        with patch.object(
            aot_cache_module,
            "get_hardware_info",
            return_value=future_hardware,
        ):
            assert aot_cache_module.find_heuristic_file(source_path) == sm100_heuristic
    finally:
        aot_cache_module.clear_heuristic_cache()


def test_rocm_hardware_info_uses_the_base_arch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "7.2.0")

    def rocm_hardware(name: str) -> HardwareInfo:
        # ROCm reports the arch with its target features.
        props = SimpleNamespace(gcnArchName=name)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: props)
        get_hardware_info.cache_clear()
        return get_hardware_info()

    source_path = tmp_path / "demo.py"
    source_path.touch()
    (tmp_path / "_helion_aot_demo_rocm_gfx950.py").touch()
    aot_cache_module.clear_heuristic_cache()
    try:
        mi300x = rocm_hardware("gfx942:sramecc+:xnack-")
        assert mi300x.compute_capability == "gfx942"
        # The full name, with features, still separates cache entries.
        assert mi300x.hardware_name == "gfx942:sramecc+:xnack-"
        # The fallback runs to older arches only, never to gfx950.
        compute_ids = mi300x.get_compatible_compute_ids()
        assert compute_ids[:3] == ["gfx942", "gfx942:sramecc+:xnack-", "gfx941"]
        assert "gfx950" not in compute_ids
        # A heuristic saved under the full arch name, as older versions named
        # them, still loads...
        saved = tmp_path / "_helion_aot_demo_rocm_gfx942:sramecc+:xnack-.py"
        saved.touch()
        assert aot_cache_module.find_heuristic_file(source_path) == saved
        # ...until one is regenerated under the base arch.
        gfx942_heuristic = tmp_path / "_helion_aot_demo_rocm_gfx942.py"
        gfx942_heuristic.touch()
        aot_cache_module.clear_heuristic_cache()
        assert aot_cache_module.find_heuristic_file(source_path) == gfx942_heuristic

        # Heuristic hardware targets name the base arch and match the report.
        env = SimpleNamespace(device=torch.device("cuda", 0))
        gfx950_only = (("rocm", "gfx950"),)
        rocm_hardware("gfx950:sramecc+:xnack-")
        assert matches_hardware(env, gfx950_only)  # pyrefly: ignore [bad-argument-type]
        rocm_hardware("gfx942:sramecc+:xnack-")
        assert not matches_hardware(env, gfx950_only)  # pyrefly: ignore [bad-argument-type]
    finally:
        get_hardware_info.cache_clear()
        aot_cache_module.clear_heuristic_cache()


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

    @pytest.mark.parametrize("kind", ["float", "fragment", "fragment_set"])
    def test_typed_constexpr_shape_key_round_trip(self, kind) -> None:
        @helion.kernel()
        def kernel(x, value: hl.constexpr):
            out = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                out[tile] = x[tile] + 1
            return out

        tensor = torch.empty(3)
        payloads = (
            "0000000000000000",
            "8000000000000000",
            "7ff8000000000001",
            "fff8000000000002",
        )
        hashes = []
        for index, payload in enumerate(payloads):
            scalar = struct.unpack("!d", bytes.fromhex(payload))[0]
            value = scalar if kind == "float" else EnumFragment((scalar,))
            if kind == "fragment_set":
                # Equal declarations are distinct set members under their
                # public identity equality; serialization must keep the count.
                value = frozenset(BooleanFragment() for _ in range(index + 1))
            args = (tensor, value)
            with cpu_render_target():
                if kind != "fragment_set":
                    kernel.bind(args)
                # Frozensets have specialization keys even though they are
                # not accepted frontend arguments; cover that key codec too.
                key = ShapeKey("typed", kernel.specialization_key(args), "cpu")
            original_hash = key.stable_hash()
            restored = ShapeKey.from_dict(json.loads(json.dumps(key.to_dict())))
            assert restored.specialization_key == key.specialization_key
            assert restored.stable_hash() == original_hash
            assert ShapeKey.from_dict(restored.to_dict()).stable_hash() == original_hash
            hashes.append(original_hash)
        assert len(set(hashes)) == len(payloads)

    def test_boolean_constexpr_shape_keys_remain_distinct_after_json(self) -> None:
        hashes = []
        for value in (True, 1, False, 0):
            # Exercise the same recursive key used by real constexpr bindings,
            # including frozensets (keyable but not frontend arguments).
            signature = (_constexpr_key(((value,), frozenset((value,)))),)
            key = ShapeKey("boolean", signature, "cpu")
            restored = ShapeKey.from_dict(json.loads(json.dumps(key.to_dict())))
            assert restored.specialization_key == signature
            assert restored.stable_hash() == key.stable_hash()
            hashes.append(restored.stable_hash())
        assert len(set(hashes)) == 4

    def test_nested_finite_float_shape_keys_remain_distinct_after_json(self) -> None:
        hashes = []
        keys = []
        for value in (1, 1.0, -2, -2.0):
            signature = (_constexpr_key(((value,), frozenset((value,)))),)
            key = ShapeKey("nested-finite", signature, "cpu")
            restored = ShapeKey.from_dict(json.loads(json.dumps(key.to_dict())))
            assert restored.specialization_key == signature
            assert restored.stable_hash() == key.stable_hash()
            hashes.append(restored.stable_hash())
            keys.append(restored.specialization_key)
        assert len(set(hashes)) == 4
        assert len(set(keys)) == 4

    def test_search_policy_is_structural_and_legacy_serialization_is_unchanged(
        self,
    ) -> None:
        legacy = ShapeKey("k", (1, 2, 3), "hw")
        policy = ShapeKey("k", (1, 2, 3), "hw", search_policy_hash="full-v1")

        assert "search_policy_hash" not in legacy.to_dict()
        assert legacy.stable_hash() == "7736dd40a5cb84bd"
        assert policy.to_dict()["search_policy_hash"] == "full-v1"
        assert legacy.stable_hash() != policy.stable_hash()
        assert ShapeKey.from_dict(legacy.to_dict()).search_policy_hash == ""

    def test_aot_shape_key_includes_cute_flash_search_policy(self) -> None:
        cache = object.__new__(AOTAutotuneCache)
        cache.autotuner = SimpleNamespace()
        cache.args = ()
        cache.hardware_id = "hw"
        specialization_key = Mock(return_value=("shape",))
        cache.kernel = SimpleNamespace(
            config_spec=SimpleNamespace(cute_flash_search_enabled=True),
            kernel=SimpleNamespace(
                name="kernel",
                specialization_key=specialization_key,
            ),
        )

        with patch(
            "helion.autotuner.local_cache._search_policy_hash",
            return_value="full-v1",
        ) as policy_hash:
            key = cache._create_shape_key()

        assert key.search_policy_hash == "full-v1"
        specialization_key.assert_called_once_with(())
        policy_hash.assert_called_once_with(
            cache.autotuner,
            cute_flash_search_enabled=True,
            explicit_aot=True,
        )

    def test_uncacheable_policy_still_uses_aot_evaluate_selection(self) -> None:
        selected = Config(block_sizes=[8])
        cache = object.__new__(AOTAutotuneCache)
        cache.mode = "evaluate"
        cache._verbose = False
        cache.autotuner = SimpleNamespace(
            _search_policy_cacheable=False,
            log=Mock(),
        )
        cache.args = ()
        cache.get = Mock(return_value=selected)  # type: ignore[method-assign]
        cache._maybe_run_input_fn_workflows = Mock()  # type: ignore[method-assign]
        cache._run_autotune_trials = Mock()  # type: ignore[method-assign]

        result = cache.autotune()

        assert result == selected
        cache.get.assert_called_once_with()
        cache._run_autotune_trials.assert_not_called()

    def test_uncacheable_policy_still_persists_aot_collection(self) -> None:
        selected = Config(block_sizes=[8])
        cache = object.__new__(AOTAutotuneCache)
        cache.mode = "collect"
        cache.autotuner = SimpleNamespace(
            _search_policy_cacheable=False,
            log=Mock(),
        )
        cache.args = ()
        cache.kernel = SimpleNamespace()
        cache.get = Mock(return_value=None)  # type: ignore[method-assign]
        cache.put = Mock()  # type: ignore[method-assign]
        cache._maybe_run_input_fn_workflows = Mock()  # type: ignore[method-assign]
        cache._run_autotune_trials = Mock(  # type: ignore[method-assign]
            return_value=selected
        )

        result = cache.autotune()

        assert result == selected
        cache.get.assert_called_once_with()
        cache.put.assert_called_once_with(selected)


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


def test_static_standalone_call_key_covers_descriptor_predicates() -> None:
    backing = torch.empty(32, dtype=torch.float32)
    aligned = backing.as_strided((2, 3), (4, 1), storage_offset=0)
    unaligned = backing.as_strided((2, 3), (4, 1), storage_offset=1)
    aligned_offset = backing.as_strided((2, 3), (4, 1), storage_offset=4)
    other_aligned = torch.empty_strided((2, 3), (4, 1))

    assert aligned.data_ptr() % 16 == 0
    assert unaligned.data_ptr() % 16 != 0
    assert aligned_offset.data_ptr() % 16 == 0
    assert aligned_offset.storage_offset() != 0
    assert _standalone_call_key((aligned,)) == _standalone_call_key((unaligned,))

    def descriptor_key(tensor: torch.Tensor) -> tuple[object, ...]:
        return _standalone_call_key((tensor,), tensor_descriptor_guards=True)

    assert descriptor_key(aligned) == descriptor_key(other_aligned)
    assert descriptor_key(aligned) != descriptor_key(unaligned)
    assert descriptor_key(aligned) != descriptor_key(aligned_offset)


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
    bound_kernel = SimpleNamespace(
        kernel=kernel_api,
        settings=Settings(),
        config_spec=SimpleNamespace(cute_flash_search_enabled=False),
        is_cacheable=lambda: True,
    )
    autotuner = SimpleNamespace(
        kernel=bound_kernel,
        settings=bound_kernel.settings,
        args=(tensor,),
        _benchmark_provider_cls=LocalBenchmarkProvider,
        _benchmark_prelude=None,
    )
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


def test_standalone_helper_name_in_literal_is_not_a_dependency(tmp_path: Path) -> None:
    source = 'def label(x):\n    return "helion_triton_helpers.device_error"\n'
    path = generate_standalone_file("label", [source], "", tmp_path)
    module = runpy.run_path(str(path))
    assert module["label"](None) == "helion_triton_helpers.device_error"
    assert "helion_triton_helpers" not in module


@onlyBackends(["triton"])
def test_standalone_resolves_device_helpers_in_their_own_scope(tmp_path: Path) -> None:
    source = (
        "def length(helion_triton_helpers):\n"
        "    return helion_triton_helpers.size(0)\n"
        "def nested(helion_triton_helpers):\n"
        "    return lambda: helion_triton_helpers.size(0)\n"
        "def check():\n"
        "    return helion_triton_helpers.device_error('overflow')\n"
    )
    path = generate_standalone_file("length", [source], "", tmp_path)
    module = runpy.run_path(str(path))
    x = torch.empty(7)
    assert module["length"](x) == module["nested"](x)() == 7
    assert isinstance(module["helion_triton_helpers"], ModuleType)
    with pytest.raises(ValueError, match="unsupported standalone Triton helpers.*size"):
        generate_standalone_file(
            "length", [source.replace(".device_error(", ".size(")], "", tmp_path
        )


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


@pytest.mark.parametrize(
    "descriptor_config",
    [
        {"block_sizes": [16], "indexing": "tensor_descriptor"},
        {"block_sizes": [16], "atomic_indexing": ["tensor_descriptor"]},
    ],
)
def test_dynamic_standalone_rejects_tensor_descriptors_before_codegen(
    tmp_path: Path,
    descriptor_config: dict[str, object],
) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("def demo(x):\n    return x\n")
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    kernel_function = namespace["demo"]
    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text(f"CONFIGS = [{descriptor_config!r}]\n")
    normalized: list[Config] = []
    codegen_calls: list[Config] = []

    def normalize(config: Config) -> Config:
        normalized.append(config)
        return config

    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = tmp_path
    cache.hardware_id = "test-hardware"
    cache.kernel = SimpleNamespace(
        settings=SimpleNamespace(static_shapes=False),
        kernel=SimpleNamespace(name="demo", __code__=kernel_function.__code__),
        _normalized_config_copy=normalize,
        to_triton_code=lambda config: codegen_calls.append(config),
    )
    cache._find_heuristic_file = lambda: heuristic_path

    AOTAutotuneCache.clear_caches()
    with pytest.raises(exc.InvalidAPIUsage, match="dynamic standalone AOT"):
        cache._maybe_run_compile()
    assert len(normalized) == 1
    assert codegen_calls == []
    assert not AOTAutotuneCache._compiled_kernels
    assert not (tmp_path / "source_demo_standalone.py").exists()
    AOTAutotuneCache.clear_caches()


@skipIfRefEager("generates code, which ref mode does not build")
@onlyBackends(["triton"])
@pytest.mark.parametrize(
    "source_kind",
    [
        "tensor-size",
        "aliased-tensor-size",
        "integer",
        "tuple-item",
        "attribute",
        "constexpr",
    ],
)
def test_dynamic_standalone_checks_baked_inputs(
    tmp_path: Path, source_kind: str
) -> None:
    class Pair(NamedTuple):
        first: int
        second: int

    scalar = source_kind not in ("tensor-size", "aliased-tensor-size")
    aliased = source_kind == "aliased-tensor-size"
    source = tmp_path / "source.py"
    expression = {
        "integer": "hl.specialize(x)",
        "tuple-item": "hl.specialize(x[0])",
        "attribute": "hl.specialize(x.first)",
        "constexpr": "x",
    }.get(source_kind, "hl.specialize(x.size(0))")
    source.write_text(
        "import helion.language as hl\n"
        f"def fixed(owner, x{': hl.constexpr' if source_kind == 'constexpr' else ''}, out):\n"
        f"    n = {expression}\n"
        "    for _ in hl.grid(1):\n"
        "        out[0] = n\n"
    )
    kernel = helion.kernel(load_generated_module(source).fixed, static_shapes=False)
    n = 7 if scalar or aliased else 2**31 + 7
    out = torch.full((1,), -1, dtype=torch.int64)

    def input_value(n):
        if source_kind == "tuple-item":
            return (n, 2)
        if source_kind == "attribute":
            return Pair(n, 2)
        return n if scalar else torch.empty(n, 0)

    value = input_value(n)
    owner = value if aliased else torch.empty(3)
    args = (owner, value, out)
    model = tmp_path / "model.py"
    model.write_text("CONFIGS = [{}]\n")
    with (
        cpu_render_target(),
        patch(
            "helion.autotuner.aot_cache.get_hardware_info",
            return_value=HardwareInfo(
                device_kind="cuda",
                hardware_name="CPU proof",
                runtime_version="13.0",
                compute_capability="sm100",
            ),
        ),
    ):
        cache = AOTAutotuneCache(BaseSearch(cpu_bind(kernel, args), args))
        with patch.object(cache, "_find_heuristic_file", return_value=model):
            cache._maybe_run_compile()
    run = load_generated_module(tmp_path / "source_fixed_standalone.py").fixed
    run(*args, _launcher=interpret_triton_kernel)
    assert out.item() == n
    if aliased:
        # The two positions may separate when the substituted dimension still agrees.
        run(owner, torch.empty(n, 0), out, _launcher=interpret_triton_kernel)
        assert out.item() == n
    hold = Mock()
    mismatches = (
        (input_value(15), input_value(7.0))
        if scalar
        else (torch.empty(15 if aliased else 7, 0),)
    )
    for mismatch in mismatches:
        with pytest.raises(ValueError, match="specialized input value"):
            run(owner=owner, x=mismatch, out=out, _launcher=hold)
    hold.assert_not_called()


@skipIfRefEager("generates code, which ref mode does not build")
@onlyBackends(["triton"])
@pytest.mark.parametrize(
    ("value", "different"),
    [
        (None, 17),
        (7, 17),
        (0.0, -0.0),
        tuple(
            struct.unpack("!d", bytes.fromhex(payload))[0]
            for payload in ("7ff8000000000001", "7ff8000000000002")
        ),
        (torch.Size((4, 8)), torch.Size((4, 9))),
        pytest.param(range(3), None, id="unsupported-range"),
    ],
)
def test_standalone_constexpr_observes_the_bound_literal(
    tmp_path: Path, value, different
):
    source = tmp_path / "source.py"
    source.write_text(
        "import helion.language as hl\n"
        "def fixed(out, value: hl.constexpr):\n"
        "    for i in hl.grid(1):\n"
        "        out[i] = 23\n"
    )
    kernel = helion.kernel(load_generated_module(source).fixed, static_shapes=False)
    out = torch.zeros(1, dtype=torch.int64)
    args = (out, hl.constexpr(value))
    model = tmp_path / "model.py"
    model.write_text("CONFIGS = [{}]\n")
    with (
        cpu_render_target(),
        patch(
            "helion.autotuner.aot_cache.get_hardware_info",
            return_value=HardwareInfo("cuda", "CPU proof", "13.0", "sm100"),
        ),
    ):
        bound = cpu_bind(kernel, args)
        cache = AOTAutotuneCache(BaseSearch(bound, args))
        with patch.object(cache, "_find_heuristic_file", return_value=model):
            if isinstance(value, range):
                artifact = tmp_path / "source_fixed_standalone.py"
                artifact.write_text("# previous export\n")
                with pytest.raises(
                    exc.InvalidAPIUsage, match="constexpr argument 'value'"
                ):
                    cache._maybe_run_compile()
                assert artifact.read_text() == "# previous export\n"
                return
            cache._maybe_run_compile()
    assert _constexpr_key(value) != _constexpr_key(different)
    run = load_generated_module(tmp_path / "source_fixed_standalone.py").fixed
    for equal in (value, hl.constexpr(value)):
        out.fill_(-1)
        run(out, equal, _launcher=interpret_triton_kernel)
        assert out.item() == 23
    hold = Mock()
    for other in (different, hl.constexpr(different)):
        with pytest.raises(ValueError, match="specialized input value"):
            run(out, other, _launcher=hold)
    hold.assert_not_called()


@skipIfRefEager("generates code, which ref mode does not build")
@onlyBackends(["triton"])
def test_standalone_shifted_tile_inlines_its_host_grid_length(tmp_path: Path) -> None:
    @helion.kernel(backend="triton", static_shapes=False)
    def max(x, out):  # noqa: A001 - test the exported name
        for t in hl.tile(1, x.size(0)):
            out[t] = x[t]

    x = torch.arange(7)
    out = torch.full_like(x, -1)
    config = Config(block_sizes=[4])
    with cpu_render_target():
        bound = cpu_bind(max, (x, out))
        direct = bound.to_code(
            config, options=helion.OutputCodeOptions(allow_helion_deps=False)
        )
        ordinary = bound.to_code(config)
    assert "helion.runtime._grid_range_length(" in ordinary
    direct_path = tmp_path / "direct.py"
    direct_path.write_text(direct)
    aot = generate_standalone_file("max", [ordinary], "", tmp_path)
    for path in (direct_path, aot):
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name.split(".")[0] == "helion" for alias in node.names)
            or isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "helion"
            for node in ast.walk(ast.parse(path.read_text()))
        )
        out.fill_(-1)
        load_generated_module(path).max(x, out, _launcher=interpret_triton_kernel)
        torch.testing.assert_close(out, torch.tensor([-1, 1, 2, 3, 4, 5, 6]))


@skipIfRefEager("generates code, which ref mode does not build")
@onlyBackends(["triton"])
def test_standalone_loaded_range_links_its_device_overflow_check(
    tmp_path: Path,
) -> None:
    @helion.kernel(backend="triton", static_shapes=False)
    def write(lo, hi, out):
        for i in hl.grid(lo, hi):
            out[i] = i + 1

    lo, hi = torch.tensor(1), torch.tensor(4)
    out = torch.zeros(5, dtype=torch.int64)
    config = Config(block_sizes=[], pid_type="persistent_blocked")
    with cpu_render_target():
        bound = cpu_bind(write, (lo, hi, out))
        direct = bound.to_code(
            config, options=helion.OutputCodeOptions(allow_helion_deps=False)
        )
        ordinary = bound.to_code(config)
    assert "helion_triton_helpers.device_error(" in ordinary
    direct_path = tmp_path / "direct.py"
    direct_path.write_text(direct)
    aot = generate_standalone_file("write", [ordinary], "", tmp_path)
    for path in (direct_path, aot):
        mod = load_generated_module(path)
        from triton.language.core import is_builtin

        # Triton permits module globals in JIT code; SimpleNamespace works
        # in the interpreter but is refused by the native compiler.
        assert isinstance(mod.helion_triton_helpers, ModuleType)
        assert is_builtin(mod.helion_triton_helpers.device_error)
        out.zero_()
        with patch.object(mod.helion.runtime, "get_num_sm", return_value=1):
            mod.write(lo, hi, out, _launcher=interpret_triton_kernel)
        torch.testing.assert_close(out, torch.tensor([0, 2, 3, 4, 0]))


@pytest.mark.parametrize("expected_width", [0, None])
def test_dynamic_width_guard_preserves_keyword_heuristics(
    tmp_path: Path, expected_width: int | None
) -> None:
    # A public name must not shadow the builtins used by the linked width guard.
    path = generate_standalone_file(
        "max",
        [
            "def max(*args, **kwargs):\n    raise RuntimeError('failed config')\n",
            "def max(x):\n    return 1\n",
            "def max(x):\n    return 2\n",
        ],
        "def key_max(x):\n    return 1 + int(x.size(0) > 8)\n",
        tmp_path,
        expected_width=expected_width,
    )
    pick = runpy.run_path(str(path))["max"]
    for size, expected in ((7, 1), (15, 2)):
        x = torch.empty(size, 0)
        assert pick(x) == pick(x=x) == expected


def test_ordinary_selector_keeps_its_imports_and_builtins(tmp_path: Path) -> None:
    # Neither child imports torch, and the public name shadows the model's int.
    path = generate_standalone_file(
        "int",
        ["def int(x):\n    return 16\n", "def int(x):\n    return 32\n"],
        "import torch\nfrom helion import Config\n"
        "def key_int(x):\n    return int(isinstance(x, torch.Tensor) and x.numel() > 4)\n",
        tmp_path,
    )
    original_import = __import__

    def forbid_helion(name, *args, **kwargs):
        if name.split(".")[0] == "helion":
            raise AssertionError("standalone selector imported Helion")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=forbid_helion):
        module = runpy.run_path(str(path))
        select = module["int"]
        assert select(torch.empty(3)) == 16
        assert select(x=torch.empty(8)) == 32
    assert "key_int" not in module and "torch" not in module


def test_standalone_uses_shipped_fp8_selector_helpers(tmp_path: Path) -> None:
    name = "dynamic_per_token_scaled_fp8_quant"
    model = (
        Path(__file__).resolve().parents[1]
        / "pretuned_kernels"
        / name
        / f"_helion_aot_{name}_cuda_sm100.py"
    )
    original = runpy.run_path(str(model))
    children = [
        f"def {name}(scale, x): return {i}" for i in range(len(original["_KEYS"]))
    ]
    path = generate_standalone_file(name, children, model.read_text(), tmp_path)
    module = runpy.run_path(str(path))
    for shape, expected in (((16, 4096), 13), ((3, 2100), 6)):
        x = torch.empty(1).expand(shape)
        assert module[name](None, x) == original[f"key_{name}"](None, x) == expected
    assert "_pick_index" not in module


def test_standalone_runs_shipped_scaled_mm_model_initialization(tmp_path: Path) -> None:
    model = (
        Path(__file__).resolve().parents[1]
        / "pretuned_kernels/scaled_mm/_helion_aot_scaled_mm_cuda_sm90.py"
    )
    original = runpy.run_path(str(model))
    children = [
        f"def scaled_mm(c, a, b): return {i}" for i in range(len(original["_KEYS"]))
    ]
    path = generate_standalone_file("scaled_mm", children, model.read_text(), tmp_path)
    select = runpy.run_path(str(path))["scaled_mm"]
    x = torch.empty(1)
    args = (None, x.expand(3, 2048), x.expand(2048, 4096))
    # The two initialization loops populate a sorted table. M=3 picks M=4.
    assert (
        select(*args)
        == original["key_scaled_mm"](*args)
        == original["_KEYS"].index((2048, 4096, 4))
    )


def test_standalone_uses_assigned_selector_and_table_writes(tmp_path: Path) -> None:
    model = (
        "import functools\nTABLE = {}\nTABLE[7] = 1\n"
        "def pick(x): return TABLE.get(x.numel(), 0)\n"
        "key_demo = functools.cache(pick)\n"
    )
    children = ["def demo(x): return 16", "def demo(x): return 32"]
    path = generate_standalone_file("demo", children, model, tmp_path)
    select = runpy.run_path(str(path))["demo"]
    assert select(torch.empty(6)) == 16
    assert select(torch.empty(7)) == 32
    # Namespace inspection could observe a seemingly unused Helion import.
    original = path.read_bytes()
    with pytest.raises(ValueError, match="requires a Helion import"):
        generate_standalone_file(
            "demo",
            children,
            "from helion import Config\n" + model + "vars()\n",
            tmp_path,
        )
    assert path.read_bytes() == original


def test_standalone_registers_private_selector_by_export_path(tmp_path: Path) -> None:
    # Dataclasses resolves annotations through sys.modules[cls.__module__].
    model = (
        "from __future__ import annotations\nfrom dataclasses import dataclass\n"
        "@dataclass\nclass Bucket:\n    n: int\n"
        "def key_demo(x): return int(x.numel() == Bucket(7).n)\n"
    )
    children = ["def demo(x): return 16", "def demo(x): return 32"]
    selectors = []
    for directory in (tmp_path / "a", tmp_path / "b"):
        path = generate_standalone_file("demo", children, model, directory)
        exported = runpy.run_path(str(path))
        assert exported["demo"](torch.empty(7)) == 32
        selectors.append(exported["_helion_aot_select"])
    assert selectors[0].__module__ != selectors[1].__module__
    for selector in selectors:
        assert sys.modules[selector.__module__].key_demo is selector


@skipIfRefEager("generates code, which ref mode does not build")
@onlyBackends(["triton"])
def test_compile_saved_model_with_training_only_helion_import(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text(
        "import helion.language as hl\n"
        "def demo(x):\n"
        "    for t in hl.tile(x.size(0)):\n"
        "        x[t] = x[t] + 1\n"
    )
    kernel = helion.kernel(load_generated_module(source).demo, static_shapes=False)
    model = tmp_path / "model.py"
    model_source = (
        "from __future__ import annotations\nfrom dataclasses import dataclass\n"
        "MODEL_FILE = __file__\n"
        "CONFIGS = [{'block_sizes': [16]}, {'block_sizes': [32]}]\n"
        "@dataclass\nclass Bucket:\n    n: int\n"
        "def key_demo(x): return int(Bucket(7).n == x.numel())\n"
        "def autotune_demo(x):\n    from helion import Config\n    return Config()\n"
    )
    model.write_text(model_source)
    args = (torch.arange(7),)
    with (
        cpu_render_target(),
        patch(
            "helion.autotuner.aot_cache.get_hardware_info",
            return_value=HardwareInfo("cuda", "CPU proof", "13.0", "sm100"),
        ),
    ):
        cache = AOTAutotuneCache(BaseSearch(cpu_bind(kernel, args), args))
        path = tmp_path / "source_demo_standalone.py"
        with patch.object(cache, "_find_heuristic_file", return_value=model):
            cache._maybe_run_compile()
            saved = AOTAutotuneCache._heuristic_modules[model]
            assert sys.modules[saved.Bucket.__module__] is saved
            assert Path(saved.MODEL_FILE) == model
            deployed = load_generated_module(path)
            selector = deployed._helion_aot_select
            assert Path(sys.modules[selector.__module__].MODEL_FILE) == path
            assert selector(*args) == 1 and selector(torch.empty(8)) == 0
            deployed.demo(*args, _launcher=interpret_triton_kernel)
            torch.testing.assert_close(args[0], torch.arange(1, 8))
            original = path.read_bytes()
            for nonportable in (
                model_source
                + "import importlib\nruntime = importlib.import_module('helion')\n",
                model_source.replace(
                    "return int(Bucket(7).n == x.numel())", "return autotune_demo(x)"
                ),
            ):
                AOTAutotuneCache.clear_caches()
                model.write_text(nonportable)
                with pytest.raises(ValueError, match="requires a Helion import"):
                    cache._maybe_run_compile()
                assert path.read_bytes() == original


def test_dynamic_standalone_selects_on_portable_user_key(tmp_path: Path) -> None:
    def size_key(x: torch.Tensor) -> tuple[int, torch.dtype]:
        return x.numel(), x.dtype

    children = ["def choose(x):\n    return 16", "def choose(x):\n    return 32"]
    selector = "def key_choose(n, width):\n    return int(n > 3 and width == 2)"
    path = generate_standalone_file(
        "choose", children, selector, tmp_path, user_key=size_key
    )
    choose = runpy.run_path(str(path))["choose"]
    assert choose(x=torch.empty(8, dtype=torch.int16)) == 32
    assert choose(torch.empty(3, dtype=torch.int16)) == 16
    assert choose(torch.empty(8, dtype=torch.int32)) == 16

    original = path.read_bytes()
    with pytest.raises(StructuralPolicyError, match="named function"):
        generate_standalone_file(
            "choose", children, selector, tmp_path, user_key=lambda x: x.numel()
        )
    assert path.read_bytes() == original


def test_standalone_key_captures_nested_globals_without_helion(tmp_path: Path) -> None:
    def count_large_dims(x):
        return sum(d > _STANDALONE_KEY_LIMIT for d in x.shape)

    def needs_helion(x):
        return helion.cdiv(x.numel(), 8)

    def runtime_import(x):
        return importlib.import_module("helion").cdiv(x.numel(), 8)

    def indirect(x):
        return x.numel() + globals().get("_STANDALONE_KEY_LIMIT", 0)

    def qualified(x):
        return x.numel() + builtins.globals().get("_STANDALONE_KEY_LIMIT", 0)

    def imported(x):
        from builtins import globals as namespace

        return x.numel() + namespace().get("_STANDALONE_KEY_LIMIT", 0)

    def typed_key(x):
        if TYPE_CHECKING:
            from helion import Config

            assert Config is not None
        return builtins.len(x)

    children = ["def choose(x): return 16", "def choose(x): return 32"]
    selector = "def key_choose(count): return int(count > 1)"
    path = generate_standalone_file(
        "choose", children, selector, tmp_path, user_key=count_large_dims
    )
    choose = runpy.run_path(str(path))["choose"]
    assert choose(torch.empty(8, 4)) == 32
    assert choose(torch.empty(8, 3)) == 16
    path = generate_standalone_file(
        "choose", children, selector, tmp_path, user_key=typed_key
    )
    assert runpy.run_path(str(path))["choose"](torch.empty(4)) == 32
    original = path.read_bytes()
    for key, reason in (
        (needs_helion, "requires a Helion import"),
        (runtime_import, "requires a Helion import"),
        (indirect, "inspects its namespace"),
        (qualified, "inspects its namespace"),
        (imported, "inspects its namespace"),
    ):
        with pytest.raises(StructuralPolicyError, match=reason):
            generate_standalone_file(
                "choose", children, selector, tmp_path, user_key=key
            )
        assert path.read_bytes() == original


def test_dynamic_standalone_still_accepts_pointer_config(tmp_path: Path) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("def demo(x):\n    return x\n")
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    kernel_function = namespace["demo"]
    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text(
        "import functools\n"
        "@functools.cache\n"
        "def autotune_demo(x):\n"
        "    _C = [{'block_sizes': [16], 'indexing': 'pointer'}]\n"
        "    return _C[0]\n"
    )
    codegen_calls: list[Config] = []

    def to_triton_code(config: Config) -> str:
        codegen_calls.append(config)
        return "from __future__ import annotations\n\ndef demo(x):\n    return x\n"

    class BoundKernelStub(SimpleNamespace):
        """Support weak references like the bound kernel the cache stores."""

    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = tmp_path
    cache.hardware_id = "test-hardware"
    cache.kernel = BoundKernelStub(
        settings=SimpleNamespace(static_shapes=False, index_dtype=None),
        host_function=SimpleNamespace(constexpr_args={}),
        env=SimpleNamespace(
            index_dtype=torch.int32, wide_empty_grid=False, specialized_vars=set()
        ),
        kernel=SimpleNamespace(
            name="demo",
            __code__=kernel_function.__code__,
            normalize_args=lambda *args: args,
        ),
        _normalized_config_copy=lambda config: config,
        to_triton_code=to_triton_code,
    )
    cache._find_heuristic_file = lambda: heuristic_path

    AOTAutotuneCache.clear_caches()
    cache.args = (torch.empty(7, 0),)
    with (
        patch.object(
            cache.kernel, "to_triton_code", side_effect=RuntimeError("bad config")
        ),
        pytest.raises(RuntimeError, match="bad config"),
    ):
        cache._maybe_run_compile()
    assert not AOTAutotuneCache._compiled_kernels
    assert not (tmp_path / "source_demo_standalone.py").exists()
    cache._maybe_run_compile()
    assert len(codegen_calls) == 1
    path = tmp_path / "source_demo_standalone.py"
    demo = runpy.run_path(str(path))["demo"]
    assert demo(x=cache.args[0]) is cache.args[0]
    with pytest.raises(ValueError, match="No standalone variant"):
        demo(x=torch.empty(2**31 + 7, 0))
    # A partially compiled file may already serve a successful child, but
    # requesting the export again must still retry its previously failed child.
    AOTAutotuneCache.clear_caches()
    heuristic_path.write_text(
        "CONFIGS = [{'block_sizes': [16]}, {'block_sizes': [32]}]\n"
        "def key_demo(x): return int(x.shape[0] > 7)\n"
    )

    def first_attempt(config):
        if config.block_sizes[0] == 32:
            raise RuntimeError("transient compile failure")
        return to_triton_code(config)

    with patch.object(cache.kernel, "to_triton_code", side_effect=first_attempt):
        cache._maybe_run_compile()
    partial = runpy.run_path(str(path))["demo"]
    assert partial(cache.args[0]) is cache.args[0]
    other = torch.empty(8, 0)
    with pytest.raises(RuntimeError, match="failed to compile"):
        partial(other)

    def second_attempt(config):
        if config.block_sizes[0] == 16:
            raise RuntimeError("previously compiled child is temporarily unavailable")
        return to_triton_code(config)

    with patch.object(cache.kernel, "to_triton_code", side_effect=second_attempt):
        cache._maybe_run_compile()
    recovered = runpy.run_path(str(path))["demo"]
    assert recovered(other) is other and recovered(cache.args[0]) is cache.args[0]

    def bind_mode(static: bool) -> None:
        cache.kernel.settings.static_shapes = static
        cache.kernel.kernel = helion.kernel(kernel_function, static_shapes=static)
        if static:
            _bind_static_export_inputs(cache, *cache.args)

    # Alternate full publications at the same path, including hits on both
    # in-memory caches. The file must always dispatch the last requested mode.
    with patch.object(cache, "_get_heuristic_config", return_value=Config()):
        for static in (True, False, True, False):
            bind_mode(static)
            if static:
                cache._maybe_run_compile()
            else:
                with patch.object(
                    cache.kernel, "to_triton_code", side_effect=second_attempt
                ):
                    cache._maybe_run_compile()
            exported = runpy.run_path(str(path))["demo"]
            if static:
                with pytest.raises(ValueError, match="No standalone variant"):
                    exported(other)
            else:
                assert (
                    exported(other) is other
                    and exported(cache.args[0]) is cache.args[0]
                )

    # An incomplete dispatcher displaced by a static one is no longer active.
    # A different dynamic binding may publish its own partial, without serving
    # the first binding's output or inheriting its working children.
    AOTAutotuneCache.clear_caches()
    with patch.object(cache.kernel, "to_triton_code", side_effect=first_attempt):
        cache._maybe_run_compile()
    bind_mode(True)
    with patch.object(cache, "_get_heuristic_config", return_value=Config()):
        cache._maybe_run_compile()
    bind_mode(False)
    cache.kernel = BoundKernelStub(**vars(cache.kernel))
    with patch.object(cache.kernel, "to_triton_code", side_effect=second_attempt):
        cache._maybe_run_compile()
    displaced = runpy.run_path(str(path))["demo"]
    assert displaced(other) is other
    with pytest.raises(RuntimeError, match="failed to compile"):
        displaced(cache.args[0])
    cache._maybe_run_compile()

    # A completed previous file cannot admit an unrelated binding's cache hit.
    cache.kernel = BoundKernelStub(**vars(cache.kernel))
    replacement = "def demo(x): return 123"
    with patch.object(cache.kernel, "to_triton_code", return_value=replacement):
        cache._maybe_run_compile()
    assert runpy.run_path(str(path))["demo"](other) == 123

    def alternative_key(x):
        return x.numel()

    previous = path.read_bytes()
    with (
        patch.object(
            cache.kernel.kernel, "_aot_user_key", alternative_key, create=True
        ),
        pytest.raises(StructuralPolicyError, match="same custom key"),
    ):
        cache._maybe_run_compile()
    assert path.read_bytes() == previous
    cache.kernel.settings.index_dtype = torch.int32
    with pytest.raises(StructuralPolicyError, match="same index dtype"):
        cache._maybe_run_compile()
    AOTAutotuneCache.clear_caches()
    cache.kernel.settings.index_dtype = None
    cache.kernel.env.index_dtype = torch.int64
    cache.args = (torch.empty(2**31 + 7, 0),)
    cache._maybe_run_compile()
    small = torch.empty(7, 0)
    assert runpy.run_path(str(path))["demo"](small) is small
    AOTAutotuneCache.clear_caches()


def _bind_static_export_inputs(cache: AOTAutotuneCache, *args: object) -> None:
    # Only the generated child is stubbed; use the kernel's real binding signature.
    cache.args = args
    if not hasattr(cache.kernel, "env"):
        cache.kernel.env = SimpleNamespace(specialized_vars=set())
        cache.kernel.host_function = None
    kernel = cache.kernel.kernel
    cache.kernel._base_spec_key = kernel._base_specialization_key(
        kernel.normalize_args(*args)
    )


def test_static_aot_compile_accumulates_observed_shapes(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("def demo(x, metadata=(1, 2)):\n    return x\n")
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    kernel_function = namespace["demo"]
    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text("def autotune_demo(*args):\n    return {}\n")
    output_path = tmp_path / "source_demo_standalone.py"
    config = Config(block_sizes=[16])

    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = tmp_path
    cache.hardware_id = "test-hardware"

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
        kernel=helion.kernel(kernel_function, static_shapes=True),
        to_triton_code=to_triton_code,
    )

    def compile_shapes(sizes: tuple[int, ...]) -> str:
        AOTAutotuneCache.clear_caches()
        for size in sizes:
            _bind_static_export_inputs(cache, torch.empty(size))
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
    _bind_static_export_inputs(cache, torch.empty(4))

    def fail_compile(_config: Config) -> str:
        raise ValueError("compile failed")

    cache.kernel.to_triton_code = fail_compile
    with pytest.raises(RuntimeError, match="variant failed to compile"):
        cache._compile_current_static_shape(heuristic_path, "demo")
    assert output_path.read_text() == prior_source
    AOTAutotuneCache.clear_caches()


def test_static_aot_compile_dispatches_descriptor_alignment(tmp_path: Path) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("def demo(x):\n    return x\n")
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    kernel_function = namespace["demo"]
    heuristic_path = tmp_path / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text("def autotune_demo(*args):\n    return {}\n")
    config = Config(block_sizes=[16])

    backing = torch.empty(32, dtype=torch.float32)
    aligned = backing.as_strided((2, 3), (4, 1), storage_offset=0)
    unaligned = backing.as_strided((2, 3), (4, 1), storage_offset=1)
    aligned_offset = backing.as_strided((2, 3), (4, 1), storage_offset=4)

    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = tmp_path
    cache.hardware_id = "test-hardware"
    cache._get_heuristic_config = lambda _args: config
    result = 0

    def to_triton_code(_config: Config) -> str:
        return (
            f"from __future__ import annotations\n\ndef demo(x):\n    return {result}\n"
        )

    cache.kernel = SimpleNamespace(
        kernel=helion.kernel(kernel_function, static_shapes=True),
        to_triton_code=to_triton_code,
    )

    AOTAutotuneCache.clear_caches()
    for tensor, expected in ((aligned, 1), (unaligned, 2)):
        _bind_static_export_inputs(cache, tensor)
        result = expected
        cache._compile_current_static_shape(
            heuristic_path,
            "demo",
            tensor_descriptor_guards=True,
        )

    output_path = tmp_path / "source_demo_standalone.py"
    module = _load_generated(output_path, "test_static_aot_descriptor_alignment")
    assert module.demo(torch.empty_strided((2, 3), (4, 1))) == 1
    assert module.demo(unaligned) == 2
    with pytest.raises(ValueError, match="No standalone variant"):
        module.demo(aligned_offset)
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
    cache._get_heuristic_config = lambda _args: config
    cache.kernel = SimpleNamespace(
        kernel=helion.kernel(kernel_function, static_shapes=True),
        to_triton_code=lambda _config: (
            "from __future__ import annotations\n\ndef demo(x):\n    return 2\n"
        ),
    )

    _bind_static_export_inputs(cache, torch.empty(2))
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
        cache._get_heuristic_config = lambda _args: config

        def to_triton_code(_config: Config) -> str:
            codegen_barrier.wait(timeout=5)
            return (
                "from __future__ import annotations\n\n"
                f"def demo(x):\n    return {size}\n"
            )

        cache.kernel = SimpleNamespace(
            kernel=helion.kernel(kernel_function, static_shapes=True),
            to_triton_code=to_triton_code,
        )
        _bind_static_export_inputs(cache, torch.empty(size))
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
        tensor_descriptor_guards: bool = False,
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
                tensor_descriptor_guards,
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


@helion.aot_kernel()
def _explicit_aot_pointwise(x):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + 1
    return out


class _ExplicitAOTProvider(LocalBenchmarkProvider):
    pass


@onlyBackends(["triton", "cute"])
def test_custom_provider_aot_artifacts_are_deterministic(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HELION_AOT_MODE", "collect")
    monkeypatch.delenv("HELION_SKIP_CACHE", raising=False)
    monkeypatch.delenv("HELION_ASSERT_CACHE_HIT", raising=False)
    args = (torch.arange(8, dtype=torch.float32),)
    with (
        cpu_render_target(),
        patch.object(
            aot_cache_module,
            "get_hardware_info",
            return_value=HardwareInfo(
                device_kind="cuda",
                hardware_name="CPU bookkeeping proof",
                runtime_version="0",
                compute_capability="sm100",
            ),
        ),
    ):
        bound = cpu_bind(_explicit_aot_pointwise, args)
        config = bound.config_spec.default_config()

        def make_cache():
            search = FiniteSearch(
                bound,
                args,
                [config, Config(block_sizes=[16])],
                benchmark_provider_cls=partial(_ExplicitAOTProvider),
            )
            return AOTAutotuneCache(search)

        first = make_cache()
        # Real collection bookkeeping, with the device launch replaced by the
        # equivalent CPU result. No benchmark provider or timer executes.
        with (
            patch.object(first, "_run_autotune_trials", return_value=config) as tune,
            patch.object(bound, "compile_config", return_value=lambda x: x + 1),
        ):
            assert first.autotune() == config
        tune.assert_called_once()
        second = make_cache()
        assert first.shape_key == second.shape_key
        # Explicit collection owns this artifact even though automatic winner
        # caching is disabled for its custom benchmark provider.
        monkeypatch.setenv("HELION_ASSERT_CACHE_HIT", "1")
        with patch.object(second, "_run_autotune_trials") as tune:
            assert second.autotune() == config
        tune.assert_not_called()
        assert len(second._tuned_configs[bound.kernel.name]) == 1
        for cache in (first, second):
            cache._save_measurement(bound.kernel.name, cache.shape_key, config, 1.0, {})
        rows = list(csv.DictReader(first._measurements_file.open()))
        assert rows[0]["shape_hash"] == rows[1]["shape_hash"]
        monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
        assert make_cache().shape_key == first.shape_key
        assert "cache reuse is disabled" not in caplog.text


_AOT_ANCHOR = (2, 3)


def _aot_anchor():
    return _AOT_ANCHOR


@helion.aot_kernel()
def _fresh_aot_binding(x, value: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        offset = 2 if value is _aot_anchor() else 5
        out[tile] = x[tile] + offset
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("mode", ["evaluate", "compile"])
@pytest.mark.parametrize("construction", ["default", "indexed", "direct"])
@skipIfRefEager("reads compiler state, which ref mode does not build")
def test_fresh_aot_binding_preserves_explicit_selection_mode(
    monkeypatch, tmp_path, mode, construction
):
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HELION_AOT_MODE", mode)
    args = (torch.arange(8, dtype=torch.float32), _AOT_ANCHOR)
    with (
        cpu_render_target(),
        patch.object(
            aot_cache_module,
            "get_hardware_info",
            return_value=HardwareInfo(
                device_kind="cuda",
                hardware_name="CPU bookkeeping proof",
                runtime_version="0",
                compute_capability="sm100",
            ),
        ),
    ):
        bound = _fresh_aot_binding.bind(args)
        assert bound.env.requires_fresh_binding
        if construction == "direct":
            cache = AOTAutotuneCache(LFBOTreeSearch(bound, args))
        else:
            factory = (
                AOTAutotuneCache[LFBOTreeSearch]
                if construction == "indexed"
                else default_autotuner_fn
            )
            cache = factory(bound, args)
        assert isinstance(cache, AOTAutotuneCache)
        assert cache.mode == mode
        selected = bound.config_spec.default_config()
        with (
            patch.object(cache, "_get_heuristic_config", return_value=selected),
            patch.object(cache, "_maybe_run_compile") as compile_entry,
            patch.object(
                AutotuneCacheBase,
                "_run_autotune_trials",
                side_effect=AssertionError("AOT started tuning"),
            ),
        ):
            assert cache.autotune() == selected
        assert compile_entry.call_count == int(mode == "compile")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
