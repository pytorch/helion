from __future__ import annotations

import ast
import copy
import csv
import importlib.util
import inspect
import json
import operator
import os
import py_compile
from types import ModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from test.test_cute_structural_cache import _cpu_only  # noqa: F401
from test.test_cute_structural_cache import _search
from test.test_cute_structural_compatibility import _pointwise

import helion
from helion._testing import skipUnlessBackends
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.aot_cache import ShapeKey
from helion.autotuner.aot_cache import find_heuristic_file
from helion.autotuner.aot_kernel import HeuristicKeyFunction
from helion.autotuner.aot_kernel import _flatten_key_value
from helion.autotuner.aot_kernel import extract_key_features
from helion.autotuner.aot_kernel import extract_shape_features
from helion.autotuner.aot_runner import RunConfig
from helion.autotuner.aot_runner import run_build_heuristic_phase
from helion.autotuner.aot_runner import run_evaluate_phase
from helion.autotuner.aot_runner import run_measure_phase
from helion.autotuner.aot_structural_policy import MODEL_MANIFEST
from helion.autotuner.aot_structural_policy import attach_model_policy
from helion.autotuner.aot_structural_policy import measurement_files
from helion.autotuner.aot_structural_policy import model_configs
from helion.autotuner.aot_structural_policy import policy_path
from helion.autotuner.base_search import BaseSearch
from helion.autotuner.heuristic_generator import PerformanceTarget
from helion.autotuner.heuristic_generator import evaluate_heuristic
from helion.autotuner.heuristic_generator import generate_heuristic
from helion.autotuner.heuristic_generator import get_backend
from helion.autotuner.heuristic_generator import load_measurements
from helion.runtime.cute_structural_config import CuteStructuralConfig
from helion.runtime.cute_structural_config import StructuralPolicyError
from helion.runtime.cute_structural_policy import CUTE_STRUCTURAL_SETTINGS
from helion.runtime.cute_structural_policy import CuteStructuralPolicy

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any


@pytest.fixture(autouse=True)
def _key_cache() -> Iterator[None]:
    HeuristicKeyFunction.clear_cache()
    yield
    HeuristicKeyFunction.clear_cache()


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _function(tmp_path: Path) -> Callable[..., Any]:
    source = tmp_path / "original.py"
    source.write_text(
        "import torch\nimport helion.language as hl\n" + inspect.getsource(_pointwise)
    )
    return _module(source)._pointwise


def _measurements(
    policy: CuteStructuralPolicy | None, names: tuple[str, ...] = ("_pointwise",)
) -> AOTAutotuneCache:
    cache = AOTAutotuneCache(_search(policy))
    configs = [
        helion.Config.from_dict({"block_sizes": [n], "num_warps": None})
        for n in (32, 64)
    ]
    for name in names:
        for size, timings in ((37, (1.0, 4.0)), (91, (4.0, 1.0))):
            for config, timing in zip(configs, timings, strict=True):
                cache._save_measurement(
                    name,
                    ShapeKey(name, (size,), cache.hardware_id),
                    config,
                    timing,
                    {
                        "arg0_dim0": size,
                        "arg0_numel": size,
                        "arg0_dtype": "torch.float32",
                    },
                )
    return cache


def _target_options(backend: str) -> PerformanceTarget:
    return PerformanceTarget(
        backend=backend,
        min_configs=2,
        max_configs=2,
        verbose=False,
        print_score_matrix=False,
        feature_selection=False,
    )


@skipUnlessBackends(["cute"])
def test_real_measurement_partitions_preserve_legacy_and_equal_knob_identity() -> None:
    policies = (
        None,
        CuteStructuralPolicy(),
        CuteStructuralPolicy(cute_region_fission=True),
    )
    caches = [_measurements(policy) for policy in policies]
    files = measurement_files(caches[0].data_dir, caches[0].hardware_id)
    assert set(files) == {cache._measurements_file for cache in caches}
    hashes = []
    for cache, policy in zip(caches, policies, strict=True):
        with cache._measurements_file.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4
        hashes.append(rows[0]["config_hash"])
        if policy is None:
            assert list(rows[0]) == [
                "kernel_name",
                "shape_hash",
                "config_hash",
                "config",
                "shape_features",
                "timing_ms",
            ]
        else:
            assert rows[0]["cute_structural_policy_id"] == policy.identity()
        data = load_measurements(cache._measurements_file)["_pointwise"]
        assert data.cute_structural_policy == policy
        if policy is not None:
            assert all(config.config["num_warps"] is None for config in data.configs)
        assert sorted(data.shape_features, key=operator.itemgetter("arg0_dim0")) == [
            {"arg0_dim0": size, "arg0_numel": size, "arg0_dtype": "torch.float32"}
            for size in (37, 91)
        ]
    assert len(set(hashes)) == 3


@pytest.mark.parametrize("change", ["version", "id", "config", "mixed", "filename"])
@skipUnlessBackends(["cute"])
def test_measurement_validation_precedes_config_interpretation(
    change: str, tmp_path: Path
) -> None:
    cache = _measurements(CuteStructuralPolicy())
    path = cache._measurements_file
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if change == "version":
        rows[0]["aot_policy_version"] = "2"
    elif change == "id":
        rows[0]["cute_structural_policy_id"] = "bad"
    elif change == "config":
        rows[0]["config_hash"] = "bad"
    elif change == "mixed":
        other = CuteStructuralPolicy(cute_region_fission=True)
        rows[1]["cute_structural_policy"] = other.to_json()
        rows[1]["cute_structural_policy_id"] = other.identity()
    else:
        path = tmp_path / "measurements_hw__policy_bad.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with (
        patch.object(
            helion.Config,
            "from_dict",
            side_effect=AssertionError("knobs read too soon"),
        ),
        pytest.raises(StructuralPolicyError),
    ):
        load_measurements(path)


@pytest.mark.parametrize("backend", ["decision_tree", "nearest_neighbor"])
@pytest.mark.parametrize("combined", [False, True])
@skipUnlessBackends(["cute"])
def test_installed_training_backends_preserve_numeric_keys_and_config_order(
    backend: str,
    combined: bool,
    tmp_path: Path,
) -> None:
    policy = CuteStructuralPolicy(cute_flatten_nested_reductions=True)
    names = ("_pointwise", "other") if combined else ("_pointwise",)
    cache = _measurements(policy, names)
    source = tmp_path / "original.py"
    source.touch()
    with patch(
        "helion._hardware.get_hardware_info",
        return_value=SimpleNamespace(device_kind="cuda", compute_capability="sm100"),
    ):
        results = generate_heuristic(
            cache._measurements_file,
            tmp_path,
            target=_target_options(backend),
            kernel_source_files=dict.fromkeys(names, str(source)),
        )
    combined_module = _module(
        policy_path(tmp_path / "_helion_aot_original_cuda_sm100.py", policy)
    )
    for name in names:
        single = _module(policy_path(tmp_path / f"heuristic_{name}.py", policy))
        data = load_measurements(cache._measurements_file)[name]
        result = results[name]
        data.selected_config_indices = [
            data.configs.index(config) for config in result.selected_configs
        ]
        raw = (
            get_backend(backend)
            .generate_heuristic(
                name, data, result.selected_configs, ["arg0_dim0", "arg0_numel"]
            )
            .generated_code
        )
        raw_key = next(
            node
            for node in ast.parse(raw).body
            if isinstance(node, ast.FunctionDef) and node.name == f"key_{name}"
        )
        generated_key = next(
            node
            for node in ast.parse(result.generated_code).body
            if isinstance(node, ast.FunctionDef) and node.name == f"key_{name}"
        )
        assert ast.dump(raw_key) == ast.dump(generated_key)
        for module in (single, combined_module):
            records = model_configs(module, name, policy)
            assert records is not None
            assert [record.config for record in records] == result.selected_configs
            for n in (1, 37, 60, 91, 512):
                args = (torch.empty(n),)
                index = getattr(module, f"key_{name}")(*args)
                record = getattr(module, f"autotune_{name}")(*args)
                assert isinstance(record, CuteStructuralConfig)
                assert record.config == result.selected_configs[index]
                assert record.config.config["num_warps"] is None
                detached = record.config
                detached.block_sizes[0] += 1
                assert record.config == result.selected_configs[index]
                selected_from_features = getattr(module, f"select_config_{name}")(
                    extract_shape_features(args)
                )
                assert selected_from_features.config == record.config
    evaluation = evaluate_heuristic(cache._measurements_file, tmp_path)
    assert set(evaluation) == set(names)
    assert all(stats["coverage"] == 1.0 for stats in evaluation.values())
    assert all(stats["max_slowdown"] == 1.0 for stats in evaluation.values())


@pytest.mark.parametrize("failure", [False, True])
@skipUnlessBackends(["cute"])
def test_measure_all_configs_uses_selected_partition_and_restores_provider(
    failure: bool,
) -> None:
    off = CuteStructuralPolicy()
    on = CuteStructuralPolicy(cute_region_fission=True)
    search = _search(off)
    cache = AOTAutotuneCache(search)
    config = search.config_spec.default_config()
    cache._add_tuned_config(search.kernel.kernel.name, config, cache.shape_key)
    cache._save_tuned_configs()
    foreign = AOTAutotuneCache(_search(on))
    foreign._add_tuned_config(
        search.kernel.kernel.name, helion.Config(block_sizes=[1024]), cache.shape_key
    )
    foreign._save_tuned_configs()
    cache = AOTAutotuneCache(search)
    provider = Mock()
    search.benchmark_provider = provider
    original_precompile = search.settings.autotune_precompile
    if failure:
        provider.setup.side_effect = RuntimeError("setup failure")
    with (
        patch.object(search, "_prepare"),
        patch.object(
            search, "benchmark", return_value=SimpleNamespace(perf=1.25)
        ) as benchmark,
    ):
        if failure:
            with pytest.raises(RuntimeError, match="setup failure"):
                cache.measure_all_configs()
            benchmark.assert_not_called()
        else:
            assert cache.measure_all_configs() == [(config, 1.25)]
            benchmark.assert_called_once_with(config)
    provider.setup.assert_called_once_with()
    provider.cleanup.assert_called_once_with()
    assert search.settings.autotune_precompile == original_precompile


@skipUnlessBackends(["cute"])
def test_real_measure_fn_rebinds_and_times_complete_allocating_callable(
    tmp_path: Path,
) -> None:
    function = _function(tmp_path)
    policy = CuteStructuralPolicy()
    inputs = [(torch.empty(size),) for size in (37, 91)]
    kernel = helion.aot_kernel(
        function,
        backend="cute",
        cute_structural_policy=policy,
        measure_fn=lambda: iter(inputs),
    )
    bound = kernel._bind_isolated(inputs[0])
    cache = AOTAutotuneCache(BaseSearch(bound, inputs[0]))
    config = bound.config_spec.default_config()
    cache._add_tuned_config(kernel.name, config, cache.shape_key)
    seen = []
    outputs = []

    def compile_config(
        actual_bound: Any, actual_config: helion.Config
    ) -> Callable[..., torch.Tensor]:
        assert actual_bound.kernel is kernel
        assert actual_config == config
        code = actual_bound.to_code(actual_config)
        assert "torch.empty" in code
        seen.append((actual_bound, code))

        def call(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            outputs.append(out)
            return out

        return call

    with (
        patch("helion.runtime.kernel.BoundKernel.compile_config", compile_config),
        patch(
            "triton.testing.do_bench", side_effect=lambda fn: (fn(), fn(), 1.25)[2]
        ) as timer,
    ):
        cache._run_measure_fn_workflow()
    assert timer.call_count == 2 and len(seen) == 2 and len(outputs) == 4
    assert len({out.data_ptr() for out in outputs}) == 4
    assert [tuple(out.shape) for out in outputs] == [(37,), (37,), (91,), (91,)]
    assert (
        len(load_measurements(cache._measurements_file)[kernel.name].shape_features)
        == 2
    )


def _write_model(
    path: Path, name: str, policy: CuteStructuralPolicy, selected: int = 0
) -> None:
    code = f"calls = []\ndef key_{name}(*args):\n    calls.append(args)\n    return {selected}\ndef autotune_{name}(*args):\n    return {{}}\n"
    path.write_text(
        attach_model_policy(
            code,
            {name: [helion.Config(block_sizes=[32]), helion.Config(block_sizes=[64])]},
            policy,
        )
    )


@pytest.mark.parametrize("user_key", [False, True])
def test_public_prebinding_key_uses_resolved_policy_and_policy_scoped_cache(
    user_key: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = _function(tmp_path)
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    monkeypatch.setenv("HELION_HEURISTIC_DIR", str(tmp_path))
    off = CuteStructuralPolicy()
    on = CuteStructuralPolicy(cute_region_fission=True)
    for policy, index in ((off, 0), (on, 1)):
        _write_model(
            policy_path(tmp_path / "heuristic__pointwise.py", policy),
            function.__name__,
            policy,
            index,
        )
    keys = []
    for policy, index in ((off, 0), (off, 0), (on, 1)):
        kwargs = {"key": lambda x: (x.numel(),)} if user_key else {"batched": [[0]]}
        kernel = helion.aot_kernel(
            function,
            backend="cute",
            config=CuteStructuralConfig(helion.Config(block_sizes=[32]), policy),
            **kwargs,
        )
        for flag in CUTE_STRUCTURAL_SETTINGS:
            monkeypatch.setenv(
                f"HELION_{flag.upper()}", str(int(not getattr(policy, flag)))
            )
        key_fn = kernel._key_fn
        assert isinstance(key_fn, HeuristicKeyFunction)
        assert key_fn(torch.empty(37)) == index
        keys.append(key_fn._key_fn)
        for flag in CUTE_STRUCTURAL_SETTINGS:
            monkeypatch.delenv(f"HELION_{flag.upper()}")
    assert keys[0] is keys[1] and keys[0] is not keys[2]


@pytest.mark.parametrize(
    "change", ["policy", "version", "id", "config_policy", "missing"]
)
def test_bad_model_rejected_before_key_or_config_callback(
    change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = _function(tmp_path)
    policy = CuteStructuralPolicy()
    path = policy_path(tmp_path / "heuristic__pointwise.py", policy)
    _write_model(path, function.__name__, policy)
    module = _module(path)
    payload = copy.deepcopy(vars(module)[MODEL_MANIFEST])
    if change == "policy":
        payload["policy"] = CuteStructuralPolicy(cute_region_fission=True).to_dict()
    elif change == "version":
        payload["version"] = 2
    elif change == "id":
        payload["policy_id"] = "wrong"
    elif change == "config_policy":
        payload["configs"][function.__name__][0]["policy"] = CuteStructuralPolicy(
            cute_region_fission=True
        ).to_dict()
    else:
        payload = None
    path.write_text(
        f"{MODEL_MANIFEST} = {payload!r}\ndef key__pointwise(*args):\n    raise AssertionError('selector entered')\ndef autotune__pointwise(*args):\n    raise AssertionError('selector entered')\n"
    )
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    monkeypatch.setenv("HELION_HEURISTIC_DIR", str(tmp_path))
    kernel = helion.aot_kernel(function, backend="cute", cute_structural_policy=policy)
    with (
        patch(
            "helion.runtime.kernel.CompileEnvironment",
            side_effect=AssertionError("tracing entered"),
        ),
        pytest.raises(StructuralPolicyError),
    ):
        kernel.bind((torch.empty(37),))


def test_policy_lookup_never_falls_back_to_unversioned_model(tmp_path: Path) -> None:
    source = tmp_path / "original.py"
    source.touch()
    (tmp_path / "_helion_aot_original_cuda_sm100.py").write_text(
        "raise AssertionError('legacy imported')"
    )
    assert (
        find_heuristic_file(
            source, "_pointwise", tmp_path, policy=CuteStructuralPolicy()
        )
        is None
    )


@pytest.mark.parametrize("backend", ["decision_tree", "nearest_neighbor"])
@skipUnlessBackends(["cute"])
def test_actual_runner_partitions_train_and_evaluate_separately(
    backend: str, tmp_path: Path
) -> None:
    policies = (
        None,
        CuteStructuralPolicy(),
        CuteStructuralPolicy(cute_region_fission=True),
    )
    source = tmp_path / "original.py"
    source.touch()
    caches = [_measurements(policy) for policy in policies]
    for measured in caches:
        cache = AOTAutotuneCache(measured.autotuner)
        cache._add_tuned_config(
            "_pointwise",
            helion.Config(block_sizes=[32]),
            cache.shape_key,
            kernel_source_file=str(source),
        )
        cache._save_tuned_configs()
    config = RunConfig(
        benchmark_cmd=["original-benchmark", "--unchanged"],
        output_dir=caches[0].data_dir.parent,
        run_id=caches[0].data_dir.name,
        hardware_id=caches[0].hardware_id,
        min_configs=2,
        max_configs=2,
        backend=backend,
        feature_selection=False,
        print_score_matrix=False,
    )
    with (
        patch(
            "helion._hardware.get_hardware_info",
            return_value=SimpleNamespace(
                device_kind="cuda", compute_capability="sm100"
            ),
        ),
        patch(
            "helion.autotuner.aot_runner.run_benchmark", return_value=(0, "", "")
        ) as benchmark,
    ):
        assert run_measure_phase(config)
        assert run_build_heuristic_phase(config)
        assert run_evaluate_phase(config)
    assert benchmark.call_count == 2
    for call, mode in zip(
        benchmark.call_args_list, ("measure", "evaluate"), strict=True
    ):
        assert call.args[0] == config.benchmark_cmd
        assert call.args[1] == {
            "HELION_AOT_MODE": mode,
            "HELION_AOT_DATA_DIR": str(config.run_dir),
            "HELION_AUTOTUNE_CACHE": "AOTAutotuneCache",
        }
    for policy in policies:
        model = policy_path(tmp_path / "_helion_aot_original_cuda_sm100.py", policy)
        assert model.is_file()
        summary = json.loads(
            policy_path(
                config.run_dir / f"heuristic_summary_{config.hardware_id}.json", policy
            ).read_text()
        )
        assert summary["_pointwise"]["num_configs"] == 2
        evaluation = json.loads(
            policy_path(
                config.run_dir / f"evaluation_{config.hardware_id}.json", policy
            ).read_text()
        )
        if policy is not None:
            assert evaluation["_pointwise"]["coverage"] == 1.0
            assert evaluation["_pointwise"]["max_slowdown"] == 1.0


@skipUnlessBackends(["cute"])
def test_content_scoped_model_cache_ignores_stale_same_size_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = _function(tmp_path)
    policy = CuteStructuralPolicy()
    path = policy_path(tmp_path / "heuristic__pointwise.py", policy)
    _write_model(path, function.__name__, policy, selected=0)
    py_compile.compile(str(path), doraise=True)
    before_stat = path.stat()
    old_source = path.read_bytes()
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    monkeypatch.setenv("HELION_HEURISTIC_DIR", str(tmp_path))
    kernels = []
    for _index in range(2):
        kernel = helion.aot_kernel(
            function, backend="cute", cute_structural_policy=policy
        )
        assert kernel._key_fn(torch.empty(37)) == 0
        kernels.append(kernel)
    assert kernels[0]._key_fn._key_fn is kernels[1]._key_fn._key_fn
    cache = AOTAutotuneCache(
        BaseSearch(kernels[0]._bind_isolated((torch.empty(37),)), (torch.empty(37),))
    )
    with patch.object(cache, "_find_heuristic_file", return_value=path):
        assert cache._get_heuristic_config().block_sizes == [32]
    _write_model(path, function.__name__, policy, selected=1)
    assert len(path.read_bytes()) == len(old_source)
    os.utime(path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
    next_kernel = helion.aot_kernel(
        function, backend="cute", cute_structural_policy=policy
    )
    assert next_kernel._key_fn(torch.empty(37)) == 1
    assert next_kernel._key_fn._key_fn is not kernels[0]._key_fn._key_fn
    with patch.object(cache, "_find_heuristic_file", return_value=path):
        assert cache._get_heuristic_config().block_sizes == [64]
    # Existing bound policy/key remains immutable; a fresh kernel observes the
    # edited model. Both source versions retain separate in-memory modules.
    assert kernels[0]._key_fn(torch.empty(37)) == 0


@pytest.mark.parametrize("backend", ["decision_tree", "nearest_neighbor"])
@pytest.mark.parametrize("user_key", [False, True])
@skipUnlessBackends(["cute"])
def test_feature_evaluation_reuses_real_dtype_and_user_key_features(
    backend: str, user_key: bool, tmp_path: Path
) -> None:
    policy = CuteStructuralPolicy(cute_region_fission=True)
    cache = AOTAutotuneCache(_search(policy))
    configs = [helion.Config(block_sizes=[32]), helion.Config(block_sizes=[64])]
    samples = []
    for index, dtype in enumerate((torch.float16, torch.float32)):
        args = (torch.empty(37, dtype=dtype),)
        key = (37, dtype)
        features = (
            extract_key_features(key) if user_key else extract_shape_features(args)
        )
        samples.append((_flatten_key_value(key) if user_key else args, features))
        for config_index, config in enumerate(configs):
            cache._save_measurement(
                "sample",
                ShapeKey("sample", (index,), cache.hardware_id),
                config,
                1.0 if config_index == index else 4.0,
                features,
            )
    with patch(
        "helion._hardware.get_hardware_info",
        return_value=SimpleNamespace(device_kind="cuda", compute_capability="sm100"),
    ):
        result = generate_heuristic(
            cache._measurements_file, tmp_path, target=_target_options(backend)
        )["sample"]
    module = _module(policy_path(tmp_path / "heuristic_sample.py", policy))
    for args, features in samples:
        index = module.key_sample(*args)
        assert (
            module.select_config_sample(features).config
            == result.selected_configs[index]
        )
    evaluation = evaluate_heuristic(cache._measurements_file, tmp_path)["sample"]
    assert evaluation["coverage"] == 1.0
    assert evaluation["max_slowdown"] == 1.0
