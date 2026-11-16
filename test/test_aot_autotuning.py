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
import py_compile
import runpy
import stat
import struct
import sys
import threading
import time
from types import ModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
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
from helion._testing.aot_export_scenarios import HELD_A
from helion._testing.aot_export_scenarios import HELD_B
from helion._testing.aot_export_scenarios import INELIGIBLE
from helion._testing.aot_export_scenarios import KERNEL
from helion._testing.aot_export_scenarios import OTHER_DIR
from helion._testing.aot_export_scenarios import OTHER_SOURCE
from helion._testing.aot_export_scenarios import PLAIN
from helion._testing.aot_export_scenarios import SHAPES
from helion._testing.aot_export_scenarios import TWIN
from helion._testing.aot_export_scenarios import import_after_rewrite
from helion._testing.aot_export_scenarios import run_export_scenario
from helion._testing.aot_export_scenarios import scenario_names
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import load_generated_module
from helion._testing.held_overrides import hold_block_sizes
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
from helion.autotuner.aot_kernel import HeuristicKeyFunction
from helion.autotuner.aot_kernel import aot_key
from helion.autotuner.aot_kernel import extract_shape_features
from helion.autotuner.aot_structural_policy import policy_path
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
from helion.runtime.cute.structural_policy import CuteStructuralPolicy
from helion.runtime.kernel import _constexpr_key
from helion.runtime.settings import Settings
from helion.runtime.settings import default_autotuner_fn
from helion.runtime.structural_policy import StructuralPolicyError

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from collections.abc import Callable

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
            config_spec=SimpleNamespace(
                unsearched_config=lambda config, overrides: config
            ),
            settings=SimpleNamespace(autotune_config_overrides={}),
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


class _BoundKernelStub(SimpleNamespace):
    """A stand-in for a bound kernel that supports weak references, like the
    bound kernel the cache keeps for an ordinary dynamic export."""


def _stub_export_cache(
    source_dir: Path | None,
    *,
    data_dir: Path | None = None,
    static: bool = False,
    heuristic: str = "CONFIGS = [{'block_sizes': [16]}]\n",
    source: str = "def demo(x):\n    return x\n",
) -> AOTAutotuneCache:
    """A compile-mode ``AOTAutotuneCache`` over a stub binding of the kernel
    ``demo`` whose ``source`` is ``source_dir / "source.py"``, so that it is
    exported to ``source_dir / "source_demo_standalone.py"``; without a
    ``source_dir`` the kernel has no source file and is exported to
    ``data_dir``. Its tuning data, the ``heuristic`` text, is in
    ``data_dir`` (``source_dir`` by default), where
    ``cache._find_heuristic_file()`` finds it. The kernel's backend holds no
    override. A test sets ``to_triton_code`` on ``cache.kernel``, replaces
    what else it varies and binds other arguments than a tensor of two
    elements with ``_bind_static_export_inputs``."""
    filename = "<stdin>"
    if source_dir is not None:
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "source.py").write_text(source)
        filename = str(source_dir / "source.py")
    if data_dir is None:
        assert source_dir is not None
        data_dir = source_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    namespace: dict[str, Any] = {}
    exec(compile(source, filename, "exec"), namespace)
    heuristic_path = data_dir / "_helion_aot_demo_cuda_sm100.py"
    heuristic_path.write_text(heuristic)
    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = data_dir
    cache.hardware_id = "test-hardware"
    cache._get_heuristic_config = lambda _args, snapshot=None: Config(block_sizes=[16])
    cache._find_heuristic_file = lambda: heuristic_path
    cache.kernel = _BoundKernelStub(
        settings=SimpleNamespace(static_shapes=static, index_dtype=None),
        host_function=SimpleNamespace(constexpr_args={}),
        env=SimpleNamespace(
            index_dtype=torch.int32, wide_empty_grid=False, specialized_vars=set()
        ),
        kernel=helion.kernel(namespace["demo"], static_shapes=static),
        _normalized_config_copy=lambda config: config,
        _unsearched_config=lambda config, report=False: config,
        _unsearched_overrides_identity=lambda: None,
    )
    _bind_static_export_inputs(cache, torch.empty(2))
    return cache


def _stub_code(value: object) -> str:
    """The code a stub kernel generates: ``demo`` returns ``value``."""
    return f"from __future__ import annotations\n\ndef demo(x):\n    return {value}\n"


def _call_exported(path: Path, size: int) -> int:
    """Call the ``demo`` of the standalone file at ``path``, executed from
    its text: the file as it is on disk."""
    return runpy.run_path(str(path))["demo"](torch.empty(size))


def _export_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """What the AOT cache logged, of what ``caplog`` holds."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "helion.autotuner.aot_cache"
    ]


def _rewrite_under_bytecode(path: Path, content: bytes) -> None:
    """Replace the heuristic at ``path`` by ``content``, of the same size,
    within the second of the file replaced: Python's import of the path
    would then run the bytecode of that file, which is written here."""
    written = path.stat()
    py_compile.compile(
        str(path),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    path.write_bytes(content)
    os.utime(path, ns=(written.st_atime_ns, written.st_mtime_ns))
    assert path.stat().st_size == written.st_size


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
        _unsearched_config=lambda config, report=False: config,
        _unsearched_overrides_identity=lambda: None,
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


_HELD_DESCRIPTORS = '{"indexing": "tensor_descriptor"}'


def test_dynamic_standalone_descriptor_check_reads_the_config_the_kernel_compiles(
    tmp_path: Path,
) -> None:
    # The tuning data stores a pointer config, and the backend holds the
    # kernel's unsearched configs to an override that selects descriptor
    # indexing (stubbed). The export compiles the changed config, so the
    # descriptor check reads that one: the dynamic export is rejected before
    # code generation.
    changed = Config(block_sizes=[16], indexing="tensor_descriptor")
    compiled: list[Config] = []
    cache = _stub_export_cache(
        tmp_path,
        heuristic="CONFIGS = [{'block_sizes': [16], 'indexing': 'pointer'}]\n",
    )
    cache.kernel.to_triton_code = compiled.append
    cache.kernel._unsearched_config = lambda config, report=False: changed
    cache.kernel._unsearched_overrides_identity = lambda: _HELD_DESCRIPTORS

    AOTAutotuneCache.clear_caches()
    try:
        with pytest.raises(exc.InvalidAPIUsage, match="dynamic standalone AOT"):
            cache._maybe_run_compile()
        assert compiled == []
        assert not (tmp_path / "source_demo_standalone.py").exists()
    finally:
        AOTAutotuneCache.clear_caches()


def test_a_static_export_under_held_overrides_keeps_its_calls_in_one_entry(
    tmp_path: Path,
) -> None:
    # The tuning data stores a pointer config, and the backend's hook, which
    # answers by the binding, selects descriptor indexing for one call of
    # the kernel and leaves the config alone for another (stubbed: by the
    # size of the argument). The descriptor guards are one flag for the
    # file and part of its entry, so a static export under held overrides
    # has them for every call: both calls are variants of one entry, and
    # the file dispatches each to its own. Guards read off the config of
    # the call at hand would keep the two calls in two entries, each of
    # which writes the file without the other's variant.
    output_path = tmp_path / "source_demo_standalone.py"
    held: list[str | None] = [None]
    cache = _stub_export_cache(
        tmp_path,
        static=True,
        heuristic="CONFIGS = [{'block_sizes': [16], 'indexing': 'pointer'}]\n",
    )
    cache.kernel.to_triton_code = lambda config: _stub_code(
        1000 * (config.indexing == "tensor_descriptor") + cache.args[0].size(0)
    )
    cache.kernel._unsearched_config = lambda config, report=False: (
        Config(block_sizes=[16], indexing="tensor_descriptor")
        if held[0] is not None and cache.args[0].size(0) == 3
        else config
    )
    cache.kernel._unsearched_overrides_identity = lambda: held[0]

    def exported(size: int) -> int:
        _bind_static_export_inputs(cache, torch.empty(size))
        cache._maybe_run_compile()
        return _call_exported(output_path, size)

    def entries() -> list[tuple[str | bool, str | bool, int]]:
        # Of each entry: its descriptor guards, what its kernel holds, and
        # the number of its variants.
        return [
            (key[-2], key[-1], len(variants))
            for key, variants in AOTAutotuneCache._compiled_kernel_variants.items()
        ]

    AOTAutotuneCache.clear_caches()
    try:
        # Without held overrides the hook is not asked: the guards are read
        # off the stored configs, which need none.
        assert [exported(2), exported(3)] == [2, 3]
        assert entries() == [(False, "", 2)]
        held[0] = _HELD_DESCRIPTORS
        assert [exported(2), exported(3)] == [2, 1003]
        assert entries()[1:] == [(True, _HELD_DESCRIPTORS, 2)]
        # The call the hook left alone is still dispatched to its variant,
        # and a tensor the descriptor predicates set apart has none.
        assert _call_exported(output_path, 2) == 2
        module = runpy.run_path(str(output_path))
        with pytest.raises(ValueError, match="No standalone variant"):
            module["demo"](torch.empty(4)[1:])
    finally:
        AOTAutotuneCache.clear_caches()


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
        _unsearched_config=lambda config, report=False: config,
        _unsearched_overrides_identity=lambda: None,
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
        _unsearched_config=lambda config, report=False: config,
        _unsearched_overrides_identity=lambda: None,
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


def test_a_second_plain_dynamic_call_does_not_publish_the_export_again(
    tmp_path: Path,
) -> None:
    # The plain dynamic export holds every config. A call that arrives while
    # another call of its binding is still generating the export generates
    # the code as well and publishes the file; the first call, which then
    # finds the file written from what it generated, writes nothing, and
    # neither does a call once the file is written.
    exported = tmp_path / "source_demo_standalone.py"
    generating = threading.Event()
    release = threading.Event()
    codegen_calls: list[Config] = []

    def to_triton_code(config: Config) -> str:
        codegen_calls.append(config)
        if len(codegen_calls) == 1:
            # The first call's code generation, which the second call meets.
            generating.set()
            assert release.wait(60)
        return _stub_code("x")

    cache = _stub_export_cache(tmp_path)
    cache.kernel.to_triton_code = to_triton_code

    AOTAutotuneCache.clear_caches()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(cache._maybe_run_compile)
            try:
                assert generating.wait(60)
                cache._maybe_run_compile()
                assert len(codegen_calls) == 2
                written = exported.stat().st_mtime_ns
            finally:
                release.set()
            first.result(timeout=60)
        assert exported.stat().st_mtime_ns == written
        cache._maybe_run_compile()
        assert len(codegen_calls) == 2
        assert exported.stat().st_mtime_ns == written
    finally:
        AOTAutotuneCache.clear_caches()


def test_a_failed_plain_dynamic_export_is_made_again(tmp_path: Path) -> None:
    # An export that ends without its file leaves nothing behind that would
    # skip the next call.
    exported = tmp_path / "source_demo_standalone.py"
    codegen_calls: list[Config] = []

    def to_triton_code(config: Config) -> str:
        codegen_calls.append(config)
        return _stub_code("x")

    cache = _stub_export_cache(tmp_path)
    cache.kernel.to_triton_code = to_triton_code

    AOTAutotuneCache.clear_caches()
    try:
        with (
            patch(
                "helion.autotuner.aot_compile.generate_standalone_file",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(OSError, match="disk full"),
        ):
            cache._maybe_run_compile()
        assert not exported.exists()
        # Nor an owner: a file has one once it is written.
        assert not AOTAutotuneCache._standalone_export_owners
        cache._maybe_run_compile()
        assert len(codegen_calls) == 2
        assert exported.exists()
        assert len(AOTAutotuneCache._standalone_export_owners) == 1
    finally:
        AOTAutotuneCache.clear_caches()


def test_standalone_export_is_written_from_the_entry_of_its_held_overrides(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A backend may hold the configs a kernel selects without a search to
    # some of its ``autotune_config_overrides``
    # (``BackendSearchSurface.overrides_without_search``). The same call then
    # exports other code once those overrides change: another kernel over
    # the same function, or the same one after ``Kernel.reset()``. What a
    # kernel holds is part of the entry its variants are kept in, and the
    # file is written from the entry of the kernel that exports: the
    # variants of other overrides neither collide with it nor show in it,
    # and they are written again when their overrides export next.
    output_path = tmp_path / "source_demo_standalone.py"
    held: list[str | None] = [None]
    codegen_calls: list[Config] = []
    cache = _stub_export_cache(tmp_path, static=True)

    def to_triton_code(config: Config) -> str:
        codegen_calls.append(config)
        return _stub_code(100 * config.block_sizes[0] + cache.args[0].size(0))

    cache.kernel.to_triton_code = to_triton_code
    # The stored config as the kernel compiles it under the held overrides.
    cache.kernel._unsearched_config = lambda config, report=False: (
        config if held[0] is None else Config(block_sizes=[32])
    )
    cache.kernel._unsearched_overrides_identity = lambda: held[0]

    def exported(size: int) -> int:
        _bind_static_export_inputs(cache, torch.empty(size))
        cache._maybe_run_compile()
        return _call_exported(output_path, size)

    caplog.set_level("WARNING", logger="helion.autotuner.aot_cache")
    AOTAutotuneCache.clear_caches()
    # Static shapes: one variant per observed call.
    assert [exported(2), exported(3)] == [1602, 1603]
    held[0] = '{"opt_out": true}'
    assert exported(2) == 3202
    with pytest.raises(ValueError, match="No standalone variant"):
        _call_exported(output_path, 3)
    # Under one set of overrides the calls accumulate, a repeated call is
    # not exported again, and other code for a known call still collides.
    assert [exported(3), exported(2)] == [3203, 3202]
    assert len(codegen_calls) == 5
    written = output_path.read_text()
    assert exported(3) == 3203 and output_path.read_text() == written
    cache.kernel._unsearched_config = lambda config, report=False: Config(
        block_sizes=[64]
    )
    with pytest.raises(RuntimeError, match="call-key collision"):
        exported(3)
    # The overrides the export began under write the file from their own
    # entry again, which kept both of its calls: the file on disk was
    # generated under the others. Two kernels that hold different overrides
    # and export in turn therefore each find a file without the other's
    # variants.
    held[0] = None
    cache.kernel._unsearched_config = lambda config, report=False: config
    assert exported(3) == 1603
    assert _call_exported(output_path, 2) == 1602

    # Dynamic shapes: every heuristic config, compiled once while the
    # binding and what it holds stay the same. One table is kept per file,
    # so the return to the first overrides generates theirs again.
    AOTAutotuneCache.clear_caches()
    cache.kernel.settings.static_shapes = False
    codegen_calls.clear()
    assert exported(2) == 1602 and exported(2) == 1602
    assert len(codegen_calls) == 1
    for identity, value in (('{"opt_out": true}', 3202), (None, 1602)):
        held[0] = identity
        cache.kernel._unsearched_config = (
            lambda config, report=False, identity=identity: (
                config if identity is None else Config(block_sizes=[32])
            )
        )
        assert exported(2) == value and exported(2) == value
    assert len(codegen_calls) == 3
    # No export warns: each writes every variant of its own entry.
    assert not _export_warnings(caplog)
    AOTAutotuneCache.clear_caches()


@pytest.mark.parametrize("static", [False, True])
def test_exports_racing_under_other_overrides_leave_the_last_publishers_file(
    static: bool, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Two kernels over one function hold different overrides and export from
    # two threads. The first is still generating its code when the second
    # writes the file. Code is generated outside the lock and a variant is
    # published into the entry of the overrides its kernel holds, so the
    # first then writes the file from its own entry, as it would had its
    # turn come later: the file holds its variant alone, and the second
    # kernel's call does not count as exported.
    output_path = tmp_path / "source_demo_standalone.py"
    first_generating = threading.Event()
    second_published = threading.Event()

    def make_cache(held: str | None, value: int) -> AOTAutotuneCache:
        def to_triton_code(_config: Config) -> str:
            if held is None and not second_published.is_set():
                first_generating.set()
                assert second_published.wait(timeout=30)
            return _stub_code(value)

        cache = _stub_export_cache(tmp_path, static=static)
        cache.kernel.to_triton_code = to_triton_code
        cache.kernel._unsearched_overrides_identity = lambda: held
        return cache

    def published() -> int:
        return _call_exported(output_path, 2)

    first = make_cache(None, 1)
    second = make_cache('{"opt_out": true}', 2)

    def export_second() -> None:
        assert first_generating.wait(timeout=30)
        try:
            second._maybe_run_compile()
        finally:
            second_published.set()

    caplog.set_level("WARNING", logger="helion.autotuner.aot_cache")
    AOTAutotuneCache.clear_caches()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(first._maybe_run_compile),
            pool.submit(export_second),
        ]
        for future in futures:
            future.result()
    assert published() == 1
    # The first kernel's call is exported, so it is not written again ...
    written = output_path.read_text()
    first._maybe_run_compile()
    assert output_path.read_text() == written
    # ... and the second kernel's next call writes the file from its own
    # entry, as it does when the two take turns.
    second._maybe_run_compile()
    assert published() == 2
    # No export warns: no variant is dropped on the way.
    assert not _export_warnings(caplog)
    AOTAutotuneCache.clear_caches()


class _StubExports:
    """Standalone exports of stub kernels through the static or the plain
    dynamic path, for ``helion._testing.aot_export_scenarios``. The stub
    backend holds a kernel's unsearched configs to ``block_sizes``."""

    kernels = frozenset({KERNEL, TWIN, OTHER_DIR, OTHER_SOURCE})
    # A file is written from the entry of the kernel that exports.
    owned = False
    # The block size the stored config is compiled with under what a kernel
    # holds; the generated code returns it.
    _BLOCK: ClassVar[dict[str | None, int]] = {PLAIN: 16, HELD_A: 32, HELD_B: 64}

    def __init__(self, tmp_path: Path, *, static: bool) -> None:
        self.static = static
        # A static export raises the error a scenario expects of failing
        # code generation; the plain dynamic path raises the compiler's own
        # where its every config fails.
        self.codegen_failure_raises = static
        self._root = tmp_path
        # Every kernel's tuning data holds the same heuristic.
        self._caches = {
            name: _stub_export_cache(
                self._source(name).parent,
                data_dir=tmp_path / ("other_aot" if name == OTHER_DIR else "aot"),
                static=static,
            )
            for name in sorted(self.kernels)
        }

    def _source(self, kernel: str) -> Path:
        directory = "other_source" if kernel == OTHER_SOURCE else "source"
        return self._root / directory / "source.py"

    def export(
        self,
        kernel: str,
        held: str | None,
        shape: int,
        *,
        during_codegen: Callable[[], None] | None = None,
        fail: bool = False,
    ) -> None:
        cache = self._caches[kernel]
        _bind_static_export_inputs(cache, torch.empty(shape))
        # A static export compiles the config of the call at hand, which the
        # hook leaves alone for ``INELIGIBLE``; a dynamic export is generated
        # once, for every call.
        applies = held is not PLAIN and not (self.static and shape == INELIGIBLE)
        block = self._BLOCK[held if applies else PLAIN]

        def to_triton_code(config: Config) -> str:
            if fail:
                raise ValueError("stub code generation failed")
            value = 1000 * config.block_sizes[0] + (shape if self.static else 0)
            if during_codegen is not None:
                during_codegen()
            return _stub_code(value)

        cache.kernel._unsearched_config = lambda config, report=False: (
            Config(block_sizes=[block]) if applies else config
        )
        cache.kernel._unsearched_overrides_identity = lambda: (
            None if held is PLAIN else json.dumps({"held": held})
        )
        cache.kernel.to_triton_code = to_triton_code
        cache._maybe_run_compile()

    def file(self, kernel: str) -> Path:
        return self._source(kernel).with_name("source_demo_standalone.py")

    def files(self) -> set[Path]:
        return set(self._root.rglob("*_standalone*.py"))

    def dispatched(self, path: Path) -> dict[int, str | None]:
        # Executed from its text: the file as it is on disk.
        module = runpy.run_path(str(path))
        held = {block: name for name, block in self._BLOCK.items()}
        variants: dict[int, str | None] = {}
        for shape in SHAPES:
            try:
                value = module["demo"](torch.empty(shape))
            except ValueError as error:
                assert "No standalone variant" in str(error)
                continue
            block, call = divmod(value, 1000)
            assert call == (shape if self.static else 0)
            variants[shape] = held[block]
        return variants


# Each scenario is another sequence of exports, and the static and the
# dynamic path keep their entries apart, so every scenario runs on both. A
# scenario with failing code generation runs on the static path, which
# raises the error it expects.
@pytest.mark.parametrize(
    ("scenario", "static"),
    [
        (scenario, static)
        for static in (False, True)
        for scenario in scenario_names(_StubExports.kernels, failures=static)
    ],
)
def test_standalone_export_scenarios(
    scenario: str, static: bool, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # What every standalone file holds after each export of kernels whose
    # held overrides change, differ or agree: see the scenario table.
    AOTAutotuneCache.clear_caches()
    try:
        run_export_scenario(scenario, _StubExports(tmp_path, static=static), caplog)
    finally:
        AOTAutotuneCache.clear_caches()


def test_a_static_call_is_exported_again_once_another_entry_wrote_its_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # One file, two entries: the calls of one kernel on two hardware ids.
    # Each export writes the file from its own entry, so a call whose entry
    # did not write the file last is on disk no longer, whatever its entry
    # remembers. It is written again, with every variant of its entry.
    AOTAutotuneCache.clear_caches()
    caplog.set_level("WARNING", logger="helion.autotuner.aot_cache")
    driver = _StubExports(tmp_path, static=True)
    cache = driver._caches[KERNEL]
    small, large = SHAPES[:2]

    def export(hardware: str, shape: int) -> dict[int, str | None]:
        cache.hardware_id = hardware
        driver.export(KERNEL, PLAIN, shape)
        return driver.dispatched(driver.file(KERNEL))

    try:
        assert export("first", small) == {small: PLAIN}
        assert export("second", large) == {large: PLAIN}
        # Not skipped as exported: the other entry wrote the file last.
        assert export("first", small) == {small: PLAIN}
        assert export("first", large) == {small: PLAIN, large: PLAIN}
        assert export("second", large) == {large: PLAIN}
        # The file is its entry's: a repeated call writes nothing.
        written = driver.file(KERNEL).stat().st_mtime_ns
        assert export("second", large) == {large: PLAIN}
        assert driver.file(KERNEL).stat().st_mtime_ns == written
        # A switch back to an entry that holds two variants writes both, not
        # the call at hand alone.
        assert export("first", small) == {small: PLAIN, large: PLAIN}
        assert not _export_warnings(caplog)
    finally:
        AOTAutotuneCache.clear_caches()


def test_a_plain_dynamic_call_is_exported_again_once_a_static_kernel_wrote_its_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Two kernel objects over one function, one of them with static shapes,
    # write the same file. The dynamic kernel's export dispatches every
    # shape; the static kernel then writes the file from its own entry,
    # which holds the one shape it was called with. The dynamic kernel's
    # next call is not skipped as exported: the file is written from its
    # table again.
    AOTAutotuneCache.clear_caches()
    caplog.set_level("WARNING", logger="helion.autotuner.aot_cache")
    dynamic = _StubExports(tmp_path, static=False)
    static = _StubExports(tmp_path, static=True)
    path = dynamic.file(KERNEL)
    assert static.file(KERNEL) == path
    small, large = SHAPES[:2]
    every_shape = dict.fromkeys(SHAPES, PLAIN)
    try:
        dynamic.export(KERNEL, PLAIN, small)
        assert dynamic.dispatched(path) == every_shape
        static.export(KERNEL, PLAIN, large)
        assert static.dispatched(path) == {large: PLAIN}
        dynamic.export(KERNEL, PLAIN, small)
        assert dynamic.dispatched(path) == every_shape
        # The file is its entry's: a repeated call writes nothing.
        written = path.stat().st_mtime_ns
        dynamic.export(KERNEL, PLAIN, large)
        assert path.stat().st_mtime_ns == written
        # The static kernel's turn again: its entry kept its variant.
        static.export(KERNEL, PLAIN, small)
        assert static.dispatched(path) == {small: PLAIN, large: PLAIN}
        assert not _export_warnings(caplog)
    finally:
        AOTAutotuneCache.clear_caches()


def test_a_plain_dynamic_call_is_exported_again_once_other_tuning_data_wrote_its_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Two kernels over one function, with tuning data in two directories,
    # write the same file, each from the configs of its own data: the plain
    # dynamic path knows an export by its binding, the hardware id and the
    # bytes of the heuristic and the source. A call does not count as
    # exported on what another kernel of its name wrote.
    caplog.set_level("WARNING", logger="helion.autotuner.aot_cache")
    path = tmp_path / "source_demo_standalone.py"
    generated: list[int] = []
    during: list[int] = []

    def to_triton_code(config: Config) -> str:
        (block,) = config.block_sizes
        if block == 16 and not generated:
            # The other kernel's call arrives while this one's code is
            # being generated.
            during.append(exported(64))
        generated.append(block)
        return _stub_code(block)

    caches = {
        block: _stub_export_cache(
            tmp_path,
            data_dir=tmp_path / f"aot_{block}",
            heuristic=f"CONFIGS = [{{'block_sizes': [{block}]}}]\n",
        )
        for block in (16, 64)
    }
    for cache in caches.values():
        cache.kernel.to_triton_code = to_triton_code

    def exported(block: int) -> int:
        caches[block]._maybe_run_compile()
        return _call_exported(path, 2)

    AOTAutotuneCache.clear_caches()
    try:
        # The call of another kernel that arrives while this one's code is
        # generated writes the file, which this kernel's export then
        # replaces.
        assert exported(16) == 16
        assert (during, generated) == ([64], [64, 16])
        assert [exported(64), exported(16)] == [64, 16]
        # The file is its kernel's: a repeated call generates nothing.
        assert exported(16) == 16
        assert generated == [64, 16, 64, 16]
        # The same data on another hardware id is exported again.
        caches[16].hardware_id = "other-hardware"
        assert exported(16) == 16
        assert generated == [64, 16, 64, 16, 16]
        assert not _export_warnings(caplog)
    finally:
        AOTAutotuneCache.clear_caches()


def test_a_static_export_selects_from_its_own_heuristic_file(tmp_path: Path) -> None:
    # Two static kernels over one function, each handed the heuristic file
    # of its own tuning data, write the same file from two entries. The
    # variant of each entry is generated from the config its own heuristic
    # selects, through the selector itself (``_get_heuristic_config``). The
    # selections used to be kept by source file, kernel name and shape, so
    # the second kernel selected what the first one's heuristic had
    # selected.
    #
    # An export reads its heuristic file anew, for the selection, the
    # entry and the embedded text alike: a file rewritten to the bytes of
    # the other kernel's shares the entry of those bytes.
    path = tmp_path / "source_demo_standalone.py"
    caches = {
        block: _stub_export_cache(
            tmp_path,
            data_dir=tmp_path / f"aot_{block}",
            static=True,
            heuristic=(
                f"CONFIGS = [{{'block_sizes': [{block}]}}]\n"
                "def autotune_demo(*args):\n    return CONFIGS[0]\n"
            ),
        )
        for block in (16, 64)
    }
    for cache in caches.values():
        # The stub of the selector goes; the class's own selects.
        del cache._get_heuristic_config
        cache.kernel.to_triton_code = lambda config: _stub_code(config.block_sizes[0])

    def exported(block: int) -> tuple[list[int], int]:
        cache = caches[block]
        config = cache._get_heuristic_config(cache.args)
        assert config is not None
        cache._maybe_run_compile()
        return config.block_sizes, _call_exported(path, 2)

    AOTAutotuneCache.clear_caches()
    try:
        # Each export asks the selector again, for the selection of its own
        # heuristic file.
        assert [exported(16), exported(64)] == [([16], 16), ([64], 64)]
        files = {block: cache._find_heuristic_file() for block, cache in caches.items()}
        _rewrite_under_bytecode(files[16], files[64].read_bytes())
        assert [exported(16)[1], exported(64)[1]] == [64, 64]
        # Once the caches are cleared, as in a new process, the module the
        # selector keeps for the path is executed from the file's bytes,
        # not from the bytecode of the file replaced: both kernels select
        # alike.
        AOTAutotuneCache.clear_caches()
        assert [exported(16), exported(64)] == [([64], 64), ([64], 64)]
    finally:
        AOTAutotuneCache.clear_caches()


def test_a_heuristic_is_executed_as_an_import_executes_its_file(tmp_path: Path) -> None:
    # The module of a heuristic is executed from the file's bytes, and
    # names its file as Python's import of the file does: the module names
    # its file, its functions name the path they were compiled under, and
    # the bytes are compiled without the future statements of the module
    # that executes them, under which an annotation is kept as text.
    path = tmp_path / "heuristic_demo.py"
    path.write_text("def autotune_demo(*args) -> dict:\n    return {}\n")
    try:
        module = AOTAutotuneCache._load_heuristic_module(path, None)
        function = module.autotune_demo
        assert (module.__file__, function.__code__.co_filename) == (str(path),) * 2
        assert function.__annotations__ == {"return": dict}
        assert inspect.getsource(function) == path.read_text()
    finally:
        AOTAutotuneCache.clear_caches()


def test_a_dynamic_standalone_selector_executes_its_heuristic_as_the_cache_does(
    tmp_path: Path,
) -> None:
    # The selector of a dynamic standalone file executes the heuristic's
    # text, in a file that begins with the ``annotations`` future import.
    # The text is compiled without that import, as the cache compiles the
    # heuristic (above): a key that reads one of the heuristic's annotations
    # selects in the file what it selects for the cache. Compiled under the
    # file's future statements, the annotation was text and the call failed.
    model = (
        "def index_type() -> int:\n    return 0\n"
        "def key_demo(x):\n"
        "    return index_type.__annotations__['return'](x.numel() > 4)\n"
    )
    heuristic = tmp_path / "heuristic_demo.py"
    heuristic.write_text(model)
    children = ["def demo(x): return 16", "def demo(x): return 32"]
    path = generate_standalone_file("demo", children, model, tmp_path)
    try:
        key = AOTAutotuneCache._load_heuristic_module(heuristic, None).key_demo
        select = runpy.run_path(str(path))["demo"]
        for size in (3, 8):
            x = torch.empty(size)
            assert select(x) == (16, 32)[key(x)]
    finally:
        AOTAutotuneCache.clear_caches()


def test_a_plain_heuristic_is_executed_once_for_two_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two threads that ask for a plain heuristic's module at once get one
    # module, executed once, whatever the file holds when the second thread
    # gets its turn.
    path = tmp_path / "heuristic_demo.py"
    path.write_text("VALUE = 16\n")
    executed: list[bytes] = []
    execute = aot_cache_module.load_policy_module

    def slow_execute(path: Path, source: bytes) -> Any:
        executed.append(source)
        time.sleep(0.05)
        return execute(path, source)

    monkeypatch.setattr(aot_cache_module, "load_policy_module", slow_execute)
    AOTAutotuneCache.clear_caches()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(AOTAutotuneCache._load_heuristic_module, path, None)
                for _ in range(2)
            ]
            modules = [future.result() for future in futures]
        assert modules[0] is modules[1]
        assert executed == [b"VALUE = 16\n"]
    finally:
        AOTAutotuneCache.clear_caches()


def test_the_key_function_of_an_aot_kernel_executes_the_bytes_of_its_heuristic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The key function of an ``aot_kernel`` executes the heuristic for
    # itself, beside the module the cache selects from. Both execute the
    # file's bytes. Python's import would run the bytecode of the file
    # replaced, and the kernel's key would be the old heuristic's beside a
    # config of the new one.
    _lookup_hardware(monkeypatch)
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    heuristic = tmp_path / "_helion_aot_source_cuda_sm100.py"
    heuristic.write_text("def key_demo(*args):\n    return 16\n")
    _rewrite_under_bytecode(heuristic, b"def key_demo(*args):\n    return 64\n")
    HeuristicKeyFunction.clear_cache()
    aot_cache_module.clear_heuristic_cache()
    try:
        key = HeuristicKeyFunction(str(tmp_path / "source.py"), "demo")
        assert key(torch.empty(2)) == 64
    finally:
        HeuristicKeyFunction.clear_cache()
        aot_cache_module.clear_heuristic_cache()


def _lookup_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hardware of ``find_heuristic_file``, which a process without a
    GPU has none of, and no ``HELION_HEURISTIC_DIR``."""
    monkeypatch.delenv("HELION_HEURISTIC_DIR", raising=False)
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda: HardwareInfo(
            device_kind="cuda",
            hardware_name="stub",
            runtime_version="13.0",
            compute_capability="sm100",
        ),
    )


@pytest.mark.parametrize("apart", ["data directory", "working directory", "kernel"])
@pytest.mark.parametrize("order", [(16, 64, 16), (64, 16, 64)])
def test_a_cache_selects_from_the_heuristic_of_its_kernel_and_data_directory(
    apart: str, order: tuple[int, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two caches over one source file, as above, but each finds its
    # heuristic, a ``heuristic_<kernel>.py``, through the lookup itself
    # (``find_heuristic_file``): one function with tuning data in two
    # directories, or two kernels with tuning data in one. The two
    # directories are named apart, or by one relative
    # ``HELION_AOT_DATA_DIR`` under two working directories. The lookup
    # keeps its answer by the kernel and the data directory, resolved, and
    # answers with the file in that directory, which the selections and the
    # module are kept by. It used to keep it by the source file alone: the
    # second cache got the first one's heuristic, and selected from it or,
    # for another kernel, found no ``autotune_<kernel>`` in it.
    _lookup_hardware(monkeypatch)
    caches = {}
    files = {}
    working = {}
    for block in (16, 64):
        name = "other" if apart == "kernel" and block == 64 else "demo"
        working[block] = tmp_path
        data_dir = tmp_path / ("aot" if apart == "kernel" else f"aot_{block}")
        if apart == "working directory":
            working[block] = tmp_path / f"cwd_{block}"
            data_dir = working[block] / "aot"
        cache = _stub_export_cache(
            tmp_path,
            data_dir=data_dir,
            heuristic=(
                f"def autotune_{name}(*args):\n"
                f"    return {{'block_sizes': [{block}]}}\n"
            ),
        )
        cache.kernel.kernel.name = name
        # The heuristic goes where the lookup searches for the tuning data
        # of a kernel, and the stubs of the lookup and of the selector go.
        files[block] = data_dir / f"heuristic_{name}.py"
        cache._find_heuristic_file().rename(files[block])
        del cache._find_heuristic_file
        del cache._get_heuristic_config
        if apart == "working directory":
            # The data directory as the cache takes it from the variable.
            monkeypatch.setenv("HELION_AOT_DATA_DIR", "aot")
            cache.data_dir = aot_cache_module.get_aot_data_dir()
            assert not cache.data_dir.is_absolute()
        caches[block] = cache

    def selected(block: int) -> tuple[Path | None, list[int] | None]:
        cache = caches[block]
        monkeypatch.chdir(working[block])
        config = cache._get_heuristic_config(cache.args)
        return cache._find_heuristic_file(), config and config.block_sizes

    AOTAutotuneCache.clear_caches()
    try:
        assert [selected(block) for block in order] == [
            (files[block], [block]) for block in order
        ]
    finally:
        AOTAutotuneCache.clear_caches()


def test_a_lookup_without_a_data_directory_answers_for_its_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A lookup without a data directory, the key function's of an
    # ``aot_kernel``, takes the first answer its kernel got and then answers
    # for all the kernel's lookups: the key function and the cache of a
    # kernel select from one file. It used to answer for every kernel of
    # the source file.
    _lookup_hardware(monkeypatch)
    find = aot_cache_module.find_heuristic_file
    source = tmp_path / "source.py"
    data = {name: tmp_path / name for name in ("first", "second")}
    for directory in data.values():
        directory.mkdir()
        for kernel in ("demo", "other"):
            (directory / f"heuristic_{kernel}.py").touch()
    found = data["first"] / "heuristic_demo.py"

    aot_cache_module.clear_heuristic_cache()
    try:
        # Holds what the lookup does, not a wanted result: whether an
        # ``aot_kernel`` selects from the ``heuristic_<kernel>.py`` of its
        # data directory, which the key function's lookup does not search,
        # is a decision on public behavior that is still open. A key
        # function after a cache gets the file the cache found, which its
        # own lookup does not search for, and so does every later lookup
        # of the kernel, over another data directory as well.
        assert find(source, "demo", data["first"]) == found
        assert find(source, "demo") == found
        assert find(source, "demo", data["second"]) == found
        # The same again: a key function that looked up first found
        # nothing, for every data directory of its kernel.
        aot_cache_module.clear_heuristic_cache()
        assert find(source, "demo") is None
        assert find(source, "demo", data["first"]) is None
        assert find(source, "demo", data["second"]) is None
        # Another kernel of the source file finds its own, whatever the
        # first kernel's key function found.
        assert find(source, "other", data["second"]) == (
            data["second"] / "heuristic_other.py"
        )
        # Two key functions and a ``heuristic_<kernel>.py`` for each in
        # ``HELION_HEURISTIC_DIR``, a relative one here: the answer is the
        # file found, in its resolved directory.
        aot_cache_module.clear_heuristic_cache()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HELION_HEURISTIC_DIR", "first")
        assert find(source, "demo") == found
        assert find(source, "other") == data["first"] / "heuristic_other.py"
    finally:
        aot_cache_module.clear_heuristic_cache()


@pytest.mark.parametrize(
    "policy", [None, CuteStructuralPolicy()], ids=["plain", "policy"]
)
def test_a_lookup_names_a_linked_heuristic_by_its_link(
    policy: CuteStructuralPolicy | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The answer is the file's name in its resolved directory, here a data
    # directory named by itself and by a link to it. The name is not
    # resolved: a heuristic that is a link is read through the link, so a
    # link that is swapped is followed as a file rewritten in place is, by
    # the next export and, under a structural policy, the next selection.
    # Resolved, the answer named the link's first target until the lookups
    # were cleared.
    _lookup_hardware(monkeypatch)
    data = tmp_path / "aot"
    data.mkdir()
    (data / "tuned.py").touch()
    link = policy_path(data / "heuristic_demo.py", policy)
    link.symlink_to("tuned.py")
    (tmp_path / "linked").symlink_to(data)
    aot_cache_module.clear_heuristic_cache()
    try:
        for directory in (data, tmp_path / "linked"):
            found = aot_cache_module.find_heuristic_file(
                tmp_path / "source.py", "demo", directory, policy=policy
            )
            assert found == link
    finally:
        aot_cache_module.clear_heuristic_cache()


@pytest.mark.parametrize("named", ["relative", "linked"])
@pytest.mark.parametrize("directory", ["source", "HELION_HEURISTIC_DIR", "data"])
def test_a_lookup_under_a_policy_follows_its_directories(
    directory: str, named: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under a structural policy a model is read at each selection, and the
    # key of the lookup holds every directory a candidate is searched in,
    # resolved as the answer's is: the source file's,
    # ``HELION_HEURISTIC_DIR`` and the data directory. A relative one under
    # another working directory, or a link to a directory that is
    # retargeted, is another directory, whose model the lookup finds. With
    # one of them kept as it is spelled, the key answered with the model of
    # the first directory, which the answer names now that it holds a
    # resolved directory.
    _lookup_hardware(monkeypatch)
    policy = CuteStructuralPolicy()
    link = tmp_path / "current"
    file = "heuristic_demo.py"
    if directory == "source":
        file = "_helion_aot_source_cuda_sm100.py"
    for name in ("first", "second"):
        model = policy_path(tmp_path / name / "models" / file, policy)
        model.parent.mkdir(parents=True)
        model.write_text(name)

    def found(name: str) -> str:
        if named == "relative":
            monkeypatch.chdir(tmp_path / name)
            models = "models"
        else:
            link.unlink(missing_ok=True)
            link.symlink_to(tmp_path / name / "models")
            models = str(link)
        source = str(tmp_path / "source.py")
        data_dir = None
        if directory == "source":
            # The name the kernel's code has for its file.
            source = f"{models}/source.py"
        elif directory == "data":
            monkeypatch.setenv("HELION_AOT_DATA_DIR", models)
            data_dir = aot_cache_module.get_aot_data_dir()
        else:
            monkeypatch.setenv("HELION_HEURISTIC_DIR", models)
        model = aot_cache_module.find_heuristic_file(
            source, "demo", data_dir, policy=policy
        )
        assert model is not None
        return model.read_text()

    order = ["first", "second", "first"]
    aot_cache_module.clear_heuristic_cache()
    try:
        assert [found(name) for name in order] == order
    finally:
        aot_cache_module.clear_heuristic_cache()


def test_a_rewritten_standalone_file_is_imported_as_written(tmp_path: Path) -> None:
    # One kernel on two hardware ids, whose exports have the same size. The
    # second export replaces a file
    # that has been imported, within the second it was written in. The
    # writer removes the bytecode of the file it replaces, so the import
    # that follows runs the file as written.
    output_path = tmp_path / "source_demo_standalone.py"
    cache = _stub_export_cache(tmp_path)

    def export(hardware: str, value: int) -> None:
        cache.hardware_id = hardware
        cache.kernel.to_triton_code = lambda _config: _stub_code(value)
        cache._maybe_run_compile()

    AOTAutotuneCache.clear_caches()
    try:
        export("first", 16)
        before, after = import_after_rewrite(output_path, lambda: export("second", 32))
        assert (before.demo(torch.empty(2)), after.demo(torch.empty(2))) == (16, 32)
    finally:
        AOTAutotuneCache.clear_caches()


def test_an_export_goes_on_where_bytecode_cannot_be_removed(tmp_path: Path) -> None:
    # The removal of the bytecode is best effort. Here the bytecode of the
    # file is in a read-only directory, from which this process cannot
    # remove it, as it cannot from one that another account wrote. The
    # export on another hardware id replaces the file all the same, and the
    # first one's next call writes the file again.
    output_path = tmp_path / "source_demo_standalone.py"
    bytecode = importlib.util.cache_from_source(output_path)
    cache = _stub_export_cache(tmp_path)

    def exported(hardware: str, value: int) -> int:
        cache.hardware_id = hardware
        cache.kernel.to_triton_code = lambda _config: _stub_code(value)
        cache._maybe_run_compile()
        return _call_exported(output_path, 2)

    AOTAutotuneCache.clear_caches()
    try:
        assert exported("first", 16) == 16
        py_compile.compile(str(output_path), bytecode, doraise=True)
        directory = os.path.dirname(bytecode)
        mode = stat.S_IMODE(os.stat(directory).st_mode)
        os.chmod(directory, 0o500)
        try:
            if os.access(directory, os.W_OK):
                pytest.skip("this process can remove files from a read-only directory")
            assert exported("second", 32) == 32
        finally:
            os.chmod(directory, mode)
        assert os.path.exists(bytecode)
        assert exported("first", 16) == 16
    finally:
        AOTAutotuneCache.clear_caches()


def _indexed_heuristic(blocks: tuple[int, ...], selected: int, style: str) -> str:
    """A heuristic for the stub kernel ``demo`` with a config per block size
    of ``blocks``, which selects the one at index ``selected``. ``style`` is
    where it keeps the configs: in a module-level ``CONFIGS``, or in the
    ``_C`` of ``autotune_demo`` as the generated heuristics do."""
    configs = [f"{{'block_sizes': [{block}]}}" for block in blocks]
    key = f"def key_demo(*args):\n    return {selected}\n"
    if style == "CONFIGS":
        return f"CONFIGS = [{', '.join(configs)}]\n{key}"
    listed = "".join(f"        {config},\n" for config in configs)
    return (
        f"{key}\n\ndef autotune_demo(*args):\n    _C = [\n{listed}    ]\n"
        "    return _C[key_demo(*args)]\n"
    )


@pytest.mark.parametrize("style", ["CONFIGS", "_C"])
def test_a_plain_dynamic_export_is_of_one_reading_of_its_heuristic(
    tmp_path: Path, style: str
) -> None:
    # A plain dynamic export embeds the heuristic, which selects among the
    # exported configs by index, so the configs and the embedded text are of
    # one reading of the file: also when the file is rewritten between two
    # exports of a process, which the bytes of the heuristic turn into a
    # new export, or while an export generates its code.
    path = tmp_path / "source_demo_standalone.py"
    generated: list[int] = []
    rewrites: list[str] = []

    def to_triton_code(config: Config) -> str:
        (block,) = config.block_sizes
        if rewrites:
            heuristic_path.write_text(rewrites.pop())
        generated.append(block)
        return _stub_code(block)

    cache = _stub_export_cache(
        tmp_path, heuristic=_indexed_heuristic((16, 32), 1, style)
    )
    cache.kernel.to_triton_code = to_triton_code
    heuristic_path = cache._find_heuristic_file()
    assert heuristic_path is not None

    def exported() -> int:
        cache._maybe_run_compile()
        return _call_exported(path, 2)

    AOTAutotuneCache.clear_caches()
    try:
        assert exported() == 32
        heuristic_path.write_text(_indexed_heuristic((64, 128, 256), 2, style))
        assert exported() == 256
        assert generated == [16, 32, 64, 128, 256]
        # The file is of these bytes: a repeated call generates nothing.
        assert exported() == 256
        assert generated == [16, 32, 64, 128, 256]
        # Rewritten again, and once more while that export generates its
        # code: the export is of the bytes it read, and the next call
        # exports the bytes the file holds then.
        heuristic_path.write_text(_indexed_heuristic((1, 2, 3, 4), 3, style))
        rewrites.append(_indexed_heuristic((16, 32), 1, style))
        assert exported() == 4
        assert exported() == 32
        assert generated == [16, 32, 64, 128, 256, 1, 2, 3, 4, 16, 32]
    finally:
        AOTAutotuneCache.clear_caches()


_C_FUNCTION = (
    "def autotune_demo(*args):\n"
    "    _C = [\n"
    "        {'block_sizes': [16]},\n"
    "        {'block_sizes': [64]},\n"
    "    ]\n"
    "    return _C[1]\n"
)


@pytest.mark.parametrize("held", [None, "held"])
def test_a_plain_dynamic_export_reads_the_configs_of_an_imported_function(
    held: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A heuristic written by hand may import ``autotune_<kernel>``: the
    # ``_C`` list is then in the file of the function, not in the bytes of
    # the heuristic the export read, where the function's first line is
    # some other text. Both configs are exported: sought in the heuristic's
    # own lines, the list was not found and the export was skipped.
    #
    # No digest an export is known by covers that file. A call that reads
    # other configs from it is skipped while its binding wrote the export
    # last, as any call of an unchanged entry is, also for a kernel that
    # holds overrides: the rejection of a binding the hook answers
    # otherwise for compares what the hook made of the same configs.
    module = "helion_test_imported_autotune"
    imported = tmp_path / f"{module}.py"
    imported.write_text("# The configs of demo.\n\n" + _C_FUNCTION)
    monkeypatch.syspath_prepend(tmp_path)
    path = tmp_path / "source_demo_standalone.py"
    generated: list[int] = []

    def to_triton_code(config: Config) -> str:
        generated.append(config.block_sizes[0])
        return _stub_code(config.block_sizes[0])

    cache = _stub_export_cache(
        tmp_path,
        heuristic=(
            f"from {module} import autotune_demo\n\n"
            "def key_demo(*args):\n    return 1\n"
        ),
    )
    cache.kernel.to_triton_code = to_triton_code
    cache.kernel._unsearched_overrides_identity = lambda: held
    AOTAutotuneCache.clear_caches()
    try:
        cache._maybe_run_compile()
        assert generated == [16, 64]
        assert _call_exported(path, 2) == 64
        imported.write_text(
            "# The configs of demo, edited.\n\n" + _C_FUNCTION.replace("[64]", "[128]")
        )
        cache._maybe_run_compile()
        assert generated == [16, 64]
        assert _call_exported(path, 2) == 64
    finally:
        AOTAutotuneCache.clear_caches()
        sys.modules.pop(module, None)


def test_a_static_export_reads_the_configs_of_the_bytes_it_selects_from(
    tmp_path: Path,
) -> None:
    # A static export reads the heuristic's configs for the descriptor
    # guards of its file. Where they are the ``_C`` of ``autotune_<kernel>``
    # it reads them from the bytes it read the file as, which its selector
    # is handed as well: once the file holds other configs, the next export
    # reads those, for the guards and for the selection alike.
    guards: list[tuple[bool, bytes]] = []

    def compile_static(
        heuristic_file: Path,
        kernel_name: str,
        *,
        tensor_descriptor_guards: bool,
        model_source: bytes,
    ) -> None:
        guards.append((tensor_descriptor_guards, model_source))

    descriptor = _C_FUNCTION.replace("[64]}", "[64], 'indexing': 'tensor_descriptor'}")
    cache = _stub_export_cache(tmp_path, static=True, heuristic=descriptor)
    cache._compile_current_static_shape = compile_static
    AOTAutotuneCache.clear_caches()
    try:
        cache._maybe_run_compile()
        cache._find_heuristic_file().write_text(_C_FUNCTION)
        cache._maybe_run_compile()
        assert guards == [(True, descriptor.encode()), (False, _C_FUNCTION.encode())]
    finally:
        AOTAutotuneCache.clear_caches()


_C_CONFIGS = [{"block_sizes": [16]}, {"block_sizes": [64]}]


@pytest.mark.parametrize(
    ("heuristic", "configs"),
    [
        ("import functools\n\n@functools.lru_cache\n" + _C_FUNCTION, _C_CONFIGS),
        (
            "import functools\n\n"
            "def logged(fn):\n"
            "    @functools.wraps(fn)\n"
            "    def wrapper(*args):\n"
            "        return fn(*args)\n"
            "    return wrapper\n\n"
            "@logged\n" + _C_FUNCTION,
            _C_CONFIGS,
        ),
        (
            _C_FUNCTION.replace("autotune_demo", "configs")
            + "\nimport functools\n\nautotune_demo = functools.partial(configs)\n",
            None,
        ),
    ],
    ids=["lru_cache", "wraps", "partial"],
)
def test_the_configs_of_a_wrapped_function_are_read_from_the_bytes(
    tmp_path: Path, heuristic: str, configs: list[dict[str, object]] | None
) -> None:
    # What ``autotune_<kernel>`` names may be a wrapper, one without code of
    # its own (``functools.lru_cache``) or with code that is not the
    # function's. The ``_C`` list is that of the function under the
    # decorators, as ``inspect.getsource`` reads it, and of the bytes the
    # module was executed from: the file holds another list by now. A
    # callable ``inspect.getsource`` has no text for (a ``functools.partial``)
    # has no list to read, which is no error.
    path = tmp_path / "heuristic_demo.py"
    source = heuristic.encode()
    module = AOTAutotuneCache._load_heuristic_module(path, None, source)
    path.write_text(heuristic.replace("16", "128"))
    try:
        assert (
            AOTAutotuneCache._parse_configs_from_autotune(module, "demo", path, source)
            == configs
        )
    finally:
        AOTAutotuneCache.clear_caches()


_KEY_FUNCTION = "def key_demo(*args):\n    return 1\n\n"


@pytest.mark.parametrize(
    "heuristic",
    [
        "import functools\n\n" + _KEY_FUNCTION + "@functools.lru_cache\n" + _C_FUNCTION,
        "import functools\n\n"
        + _KEY_FUNCTION
        + "@functools.lru_cache(\n    maxsize=None,\n)\n"
        + _C_FUNCTION,
        # A decorator of the heuristic's own, above ``key_<kernel>``.
        "import functools\n\n"
        "def logged(fn):\n"
        "    @functools.wraps(fn)\n"
        "    def wrapper(*args):\n"
        "        return fn(*args)\n"
        "    return wrapper\n\n"
        "@logged\n" + _C_FUNCTION + "\n" + _KEY_FUNCTION,
    ],
    ids=["lru_cache", "several_lines", "wraps_before_key"],
)
def test_a_plain_dynamic_export_of_a_decorated_autotune_selects_by_its_key(
    tmp_path: Path, heuristic: str
) -> None:
    # The configs of a decorated ``autotune_<kernel>`` are read from under
    # its decorators, whose lines begin the function's text, and the
    # exported file, which selects by ``key_<kernel>``, is executed with
    # what the decorators name: both configs are exported and the file
    # selects the second.
    path = tmp_path / "source_demo_standalone.py"
    generated: list[int] = []

    def to_triton_code(config: Config) -> str:
        generated.append(config.block_sizes[0])
        return _stub_code(config.block_sizes[0])

    cache = _stub_export_cache(tmp_path, heuristic=heuristic)
    cache.kernel.to_triton_code = to_triton_code
    AOTAutotuneCache.clear_caches()
    try:
        cache._maybe_run_compile()
        assert generated == [16, 64]
        assert _call_exported(path, 2) == 64
    finally:
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
        _unsearched_config=lambda config, report=False: config,
        _unsearched_overrides_identity=lambda: None,
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
        _unsearched_config=lambda config, report=False: config,
        _unsearched_overrides_identity=lambda: None,
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
            _unsearched_config=lambda config, report=False: config,
            _unsearched_overrides_identity=lambda: None,
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


@onlyBackends(["triton", "cute"])
def test_overrides_a_backend_does_not_hold_leave_an_unsearched_config_alone() -> None:
    # A standalone export is identified by what its kernel's backend holds
    # unsearched configs to (``ConfigSpec.unsearched_overrides_identity``):
    # a kernel that identity names nothing for shares the export of a kernel
    # without overrides, so it has to compile what that one compiles.
    # ``ConfigSpec.unsearched_config`` does not ask the backend's hook then,
    # whatever the hook would return.
    with cpu_render_target():
        bound = cpu_bind(_explicit_aot_pointwise, (torch.arange(8.0),))
    spec = bound.config_spec
    config = spec.default_config()
    overrides = {"block_sizes": [64]}
    surface = type(spec.search_surface)

    def applied(
        self: object, spec: object, config: dict[str, object], overrides: dict[str, Any]
    ) -> dict[str, object]:
        return {**config, **overrides}

    def nothing(self: object, overrides: dict[str, object]) -> dict[str, object]:
        return {}

    def everything(self: object, overrides: dict[str, object]) -> dict[str, object]:
        return dict(overrides)

    with patch.object(surface, "overrides_without_search", applied):
        with patch.object(surface, "overrides_held_without_search", nothing):
            assert spec.unsearched_overrides_identity(overrides) is None
            assert spec.unsearched_config(config, overrides) is config
        with patch.object(surface, "overrides_held_without_search", everything):
            assert spec.unsearched_overrides_identity(overrides) == json.dumps(
                overrides
            )
            selected = spec.unsearched_config(config, overrides)
            assert selected is not config and selected.block_sizes == [64]
            # This hook answers for its own result with a copy, where its
            # contract says ``None``. Nothing changes, so the config itself
            # comes back, in each form an autotuner hands one over in.
            assert spec.unsearched_config(selected, overrides) is selected
            options = dict(selected)
            assert spec.unsearched_config(options, overrides) is options


@onlyBackends(["triton", "cute"])
def test_measure_mode_times_a_stored_config_as_the_kernel_compiles_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Measure mode times the configs of the tuning data, which no search
    # under the kernel's overrides need have written. A backend that holds
    # such configs to an override (stubbed as ``block_sizes``) compiles
    # another config for them, and that is the one timed: on the shape at
    # hand and on each shape of ``measure_fn``, whose own binding the hook
    # is asked for. Every timing is saved under the stored config, which a
    # heuristic selects by.
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HELION_AOT_MODE", "measure")
    args = (torch.arange(8, dtype=torch.float32),)
    inputs = [(torch.empty(size),) for size in (16, 24)]
    stored = [Config(block_sizes=[16]), Config(block_sizes=[32])]
    timed: list[Config] = []
    compiled: list[Config] = []

    def benchmark(config: Config) -> SimpleNamespace:
        timed.append(config)
        return SimpleNamespace(perf=float(len(timed)))

    def compile_config(
        bound: object, config: Config, *, allow_print: bool = True
    ) -> Callable[..., None]:
        compiled.append(config)
        return lambda *args: None

    def measured() -> list[dict[str, list[int]]]:
        with cache._measurements_file.open() as rows:
            return [json.loads(row["config"]) for row in csv.DictReader(rows)]

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
        kernel = helion.aot_kernel(
            _explicit_aot_pointwise.fn,
            autotune_config_overrides={"block_sizes": [64]},
            measure_fn=lambda: iter(inputs),
        )
        bound = cpu_bind(kernel, args)
        surface = type(bound.config_spec.search_surface)
        search = LFBOTreeSearch(bound, args)
        search.benchmark_provider = Mock()
        cache = AOTAutotuneCache(search)
        for config in stored:
            cache._add_tuned_config(kernel.name, config, cache.shape_key)
        with (
            hold_block_sizes(surface),
            patch.object(search, "_prepare"),
            patch.object(search, "benchmark", benchmark),
        ):
            assert cache.measure_all_configs() == [(stored[0], 1.0), (stored[1], 2.0)]
        assert [config.block_sizes for config in timed] == [[64], [64]]
        assert measured() == [{"block_sizes": [16]}, {"block_sizes": [32]}]

        with (
            # The hook leaves the cache's own binding alone here: a workflow
            # that asked it for the shapes of ``measure_fn`` would time the
            # stored configs.
            hold_block_sizes(surface, ineligible={id(bound.config_spec)}),
            patch("helion.runtime.kernel.BoundKernel.compile_config", compile_config),
            patch("triton.testing.do_bench", return_value=1.25),
        ):
            cache._run_measure_fn_workflow()
        assert [config.block_sizes for config in compiled] == [[64]] * 4
        assert measured()[2:] == [{"block_sizes": [16]}, {"block_sizes": [32]}] * 2


@onlyBackends(["triton", "cute"])
def test_a_plain_dynamic_export_rejects_a_binding_the_hook_answers_otherwise_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The export of a kernel without static shapes holds the code of one
    # binding, the one that exported last, and serves every call. A
    # backend's hook answers for a binding (stubbed: it holds unsearched
    # configs to ``block_sizes`` except on chosen bindings), so two bindings
    # of one kernel under the same overrides can compile a stored config
    # differently, and the file has no variant per binding. The export of
    # such a binding would replace the other binding's config for its calls
    # without a word. It is rejected, before any code is generated for it.
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HELION_AOT_MODE", "compile")
    heuristic = tmp_path / "heuristic.py"
    heuristic.write_text("CONFIGS = [{'block_sizes': [16]}]\n")
    generated: list[int] = []
    exported: list[list[str]] = []

    def export(**written: Any) -> Path:
        exported.append(written["triton_codes"])
        return tmp_path / "exported.py"

    def to_triton_code(config: Config) -> str:
        generated.append(config.block_sizes[0])
        return _stub_code(config.block_sizes[0])

    def cache_of(dtype: torch.dtype) -> AOTAutotuneCache:
        args = (torch.zeros(8, dtype=dtype),)
        bound = cpu_bind(kernel, args)
        monkeypatch.setattr(bound, "to_triton_code", to_triton_code)
        cache = AOTAutotuneCache(BaseSearch(bound, args))
        monkeypatch.setattr(cache, "_find_heuristic_file", lambda: heuristic)
        return cache

    def rejected(cache: AOTAutotuneCache) -> None:
        with pytest.raises(exc.InvalidAPIUsage, match="one binding"):
            cache._maybe_run_compile()

    def left_alone(*caches: AOTAutotuneCache) -> set[int]:
        return {id(cache.kernel.config_spec) for cache in caches}

    AOTAutotuneCache.clear_caches()
    try:
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
            patch("helion.autotuner.aot_compile.generate_standalone_file", export),
        ):
            kernel = helion.aot_kernel(
                _explicit_aot_pointwise.fn,
                static_shapes=False,
                autotune_config_overrides={"block_sizes": [64]},
            )
            first, alike, other = map(
                cache_of, (torch.float32, torch.float64, torch.int32)
            )
            surface = type(first.kernel.config_spec.search_surface)
            with hold_block_sizes(surface, left_alone(first, alike)):
                first._maybe_run_compile()
                assert exported == [[_stub_code(16)]]
                # A binding the hook answers alike for exports its own
                # code, as every other binding does.
                alike._maybe_run_compile()
                rejected(other)
                # The file stays the export of a binding answered alike.
                first._maybe_run_compile()
                assert generated == [16] * 3
                assert exported == [[_stub_code(16)]] * 3
            # The other way around: the held binding begins the export.
            AOTAutotuneCache.clear_caches()
            with hold_block_sizes(surface, left_alone(first)):
                other._maybe_run_compile()
                assert exported[3:] == [[_stub_code(64)]]
                alike._maybe_run_compile()
                rejected(first)
                assert generated[3:] == [64] * 2
                assert exported[3:] == [[_stub_code(64)]] * 2
            # A binding that exports while another generates its code is
            # answered by what is published. The binding answered otherwise
            # that arrives after one answered alike has published is
            # rejected before its code is generated ...
            AOTAutotuneCache.clear_caches()
            arrived: list[str] = []

            def meanwhile(config: Config) -> str:
                alike._maybe_run_compile()
                rejected(other)
                arrived.append("exported, rejected")
                return to_triton_code(config)

            monkeypatch.setattr(first.kernel, "to_triton_code", meanwhile)
            with hold_block_sizes(surface, left_alone(first, alike)):
                first._maybe_run_compile()
            assert arrived == ["exported, rejected"]
            assert generated[5:] == [16] * 2
            assert exported[5:] == [[_stub_code(16)]] * 2
            # ... and where it published first, the binding still generating
            # is rejected when its own turn to publish comes: the file
            # stays the other binding's.
            AOTAutotuneCache.clear_caches()

            def overtaken(config: Config) -> str:
                other._maybe_run_compile()
                return to_triton_code(config)

            monkeypatch.setattr(first.kernel, "to_triton_code", overtaken)
            with hold_block_sizes(surface, left_alone(first, alike)):
                rejected(first)
            assert generated[7:] == [64, 16]
            assert exported[7:] == [[_stub_code(64)]]
    finally:
        AOTAutotuneCache.clear_caches()


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
