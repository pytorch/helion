from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import inspect
from itertools import starmap
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast

from benchmarks.cute import compare_grouped_gemm_backends
from benchmarks.cute import cublaslt_grouped_gemm
from benchmarks.cute import cudnn_grouped_gemm
from benchmarks.cute import cutlass_contiguous_grouped_gemm
from benchmarks.cute import grouped_gemm_benchmark as common
from benchmarks.cute import grouped_gemm_deepgemm_support
from benchmarks.cute import grouped_gemm_provider_campaign as campaign
from benchmarks.cute import quack_grouped_gemm
import pytest
import torch

SOURCE_IDENTITY = "commit"
TELEMETRY_SAMPLE = campaign._TelemetrySample(
    pstate="P0",
    sm_clock_mhz=1000.0,
    memory_clock_mhz=2000.0,
    power_draw_watts=500.0,
    power_limit_watts=1000.0,
    active_clock_event_reasons=0,
)
PROVIDER_ADAPTERS = (
    (
        "deepgemm",
        grouped_gemm_deepgemm_support,
        "prepare_deepgemm_default",
        "deepgemm_root",
    ),
    ("quack", quack_grouped_gemm, "prepare_quack_default", "quack_root"),
    ("cudnn", cudnn_grouped_gemm, "prepare_cudnn_default", None),
    ("cublaslt", cublaslt_grouped_gemm, "prepare_cublaslt_default", None),
    (
        "cutlass",
        cutlass_contiguous_grouped_gemm,
        "prepare_cutlass_default",
        "cutlass_root",
    ),
)
OFFICIAL_CASES = common.official_cases()


@pytest.fixture
def cuda_stack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, dict[str, Path]]:
    cuda_home = tmp_path / "cuda"
    artifacts = {
        name: cuda_home / "lib" / f"{name}.so"
        for name in common.CUDA_STACK_PRELOAD_LIBRARY_PREFIXES
    }
    for artifact in artifacts.values():
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()
    monkeypatch.setattr(
        campaign, "_installed_cuda_stack", lambda: (cuda_home, artifacts)
    )
    return cuda_home, artifacts


def _accuracy_evidence(
    *, replay: bool = False, group_count: int = 1
) -> dict[str, object]:
    result: dict[str, object] = {
        "ok": True,
        "group_count": group_count,
        "rtol": common.CORRECTNESS_RTOL,
        "atol": common.CORRECTNESS_ATOL,
        "max_normalized_diff": 0.0,
        "max_abs": 0.0,
        "mismatch_count": 0,
    }
    if replay:
        result.update(
            poisoned_replay_rewrote_output=True,
            repeat_replay_exact=True,
            post_timing=_accuracy_evidence(group_count=group_count),
        )
    return result


def _prepared_provider(provider: str) -> common.PreparedImplementation:
    return common.PreparedImplementation(
        name=provider,
        call=lambda: None,
        output_tensors=lambda _result: (),
        logical_outputs=lambda _result: (),
        config=_provider_config(provider, 0),
    )


def _versions(*, driver: str = "590.00") -> dict[str, object]:
    return {
        "python": "3.12.0",
        "torch": "2.10.0.dev",
        "torch_cuda": "13.0",
        "triton": "3.6.0",
        "cutlass_dsl": "4.7.0",
        "cuda_driver": driver,
        "cuda_stack": {
            "distribution_versions": dict(common.CUDA_STACK_DISTRIBUTION_VERSIONS),
            "release": common.CUDA_TOOLKIT_RELEASE,
            "compiler_version": common.CUDA_COMPILER_VERSION,
        },
    }


def _provider_config(provider: str, row_index: int) -> dict[str, object]:
    case = OFFICIAL_CASES[row_index]
    if provider == "deepgemm":
        details = {
            "api": {
                "function": "m_grouped_bf16_gemm_nt_contiguous",
                "compiled_dims": "nk",
                "use_psum_layout": False,
                "ensure_zero_padding": False,
            },
            "provenance": {
                "git_head": grouped_gemm_deepgemm_support.DEEPGEMM_COMMIT,
                "cutlass_head": grouped_gemm_deepgemm_support.DEEPGEMM_CUTLASS_COMMIT,
                "fmt_head": grouped_gemm_deepgemm_support.DEEPGEMM_FMT_COMMIT,
                "version": grouped_gemm_deepgemm_support.DEEPGEMM_VERSION,
                "native_extension_sha256": "0" * 64,
            },
        }
    elif provider == "quack":
        details = {
            "benchmark_label": quack_grouped_gemm.QUACK_BENCHMARK_LABEL,
            "selection_api": "gemm(default tuned=True)",
            "requested_config": None,
            "selected_config": {"b_layout": campaign.BENCHMARK_B_LAYOUT},
            "requested_dynamic_scheduler": False,
            "resolved_dynamic_scheduler": False,
            "tuned": True,
            "resolved_split_k": 1,
            "dispatch_plan": {"type": "quack.gemm._GemmPlan"},
            "package": {
                "source_provenance": {
                    "kind": "upstream_main_snapshot",
                    "repository": quack_grouped_gemm.QUACK_REPOSITORY,
                    "commit": quack_grouped_gemm.QUACK_COMMIT,
                    "base_release_tag": quack_grouped_gemm.QUACK_BASE_RELEASE_TAG,
                    "is_formal_release": False,
                    "benchmark_label": quack_grouped_gemm.QUACK_BENCHMARK_LABEL,
                },
                "upstream_commit": quack_grouped_gemm.QUACK_COMMIT,
                "distribution_version": quack_grouped_gemm.QUACK_PACKAGE_METADATA_VERSION,
                "dependency_versions": quack_grouped_gemm.QUACK_DEPENDENCY_VERSIONS,
                "module_version": quack_grouped_gemm.QUACK_PACKAGE_METADATA_VERSION,
                "installation": "editable",
            },
        }
    elif provider == "cudnn":
        cudart = {
            "distribution": common.CUDA_RUNTIME_DISTRIBUTION,
            "package_version": common.CUDA_RUNTIME_VERSION,
        }
        details = {
            "baseline": cudnn_grouped_gemm.CUDNN_GROUPED_BASELINE,
            "frontend_version": cudnn_grouped_gemm.CUDNN_FRONTEND_VERSION,
            "backend_version": cudnn_grouped_gemm.CUDNN_BACKEND_VERSION,
            "runtime": {
                "frontend": {
                    "distribution": cudnn_grouped_gemm.CUDNN_FRONTEND_DISTRIBUTION,
                    "package_version": cudnn_grouped_gemm.CUDNN_FRONTEND_VERSION,
                },
                "requested_cuda_runtime": cudart,
                "backend_libraries": {
                    "distribution": cudnn_grouped_gemm.CUDNN_BACKEND_DISTRIBUTION,
                    "package_version": (
                        cudnn_grouped_gemm.CUDNN_BACKEND_DISTRIBUTION_VERSION
                    ),
                },
                "loaded_cuda_runtime": cudart,
            },
            "plan": {"selection": "graph_build_default"},
        }
    elif provider == "cublaslt":
        problems = tuple((m, case.n, case.k, 1) for m in case.actual_ms)
        details = {
            "library": {
                "distribution": cublaslt_grouped_gemm.CUBLASLT_DISTRIBUTION,
                "package_version": cublaslt_grouped_gemm.CUBLASLT_DISTRIBUTION_VERSION,
                "library_version": cublaslt_grouped_gemm.CUBLASLT_LIBRARY_VERSION,
            },
            "group_count": case.groups,
            "grouped_average_preferences": (
                cublaslt_grouped_gemm.cublaslt_grouped_preference_values(problems)
            ),
            "heuristic_query_capacity": (
                cublaslt_grouped_gemm.CUBLASLT_HEURISTIC_QUERY_CAPACITY
            ),
            "selected_algorithm": {"serialized_hex": "00", "heuristic_rank": 0},
        }
    else:
        details = {
            "repository": cutlass_contiguous_grouped_gemm.CUTLASS_REPOSITORY,
            "release_tag": cutlass_contiguous_grouped_gemm.CUTLASS_RELEASE_TAG,
            "commit": cutlass_contiguous_grouped_gemm.CUTLASS_COMMIT,
            "operator_api_version": (
                cutlass_contiguous_grouped_gemm.CUTLASS_OPERATOR_API_VERSION
            ),
            "target_sm": "100a",
            "registry_tuning": {
                "method": "all_supported_operators_cold_l2",
                "timer": "bench_pre_captured_cudagraphs",
                "capture_warmups": 2,
                "repetitions": 32,
                "selected_candidate_index": 0,
                "candidates": [
                    {
                        "registry_index": index,
                        "operator_name": f"cutlass.candidate.{index}",
                        "config": {
                            "use_2cta_mma": False,
                            "tile_shape": [128 * (index + 1), 128, 64],
                            "cluster_shape": [1, 1, 1],
                        },
                        "compiled_for": "sm100",
                        "selection_median_ms": 1.0 + index,
                        "correctness_checked": True,
                    }
                    for index in range(2)
                ],
            },
        }
    return common.provider_config(provider, details)


def _helion_config(row_index: int) -> dict[str, object]:
    return {
        "selection_mode": campaign.HELION_SELECTION,
        "autotuned": False,
        "b_layout": campaign.BENCHMARK_B_LAYOUT,
        "selection_api": "BoundKernel.set_config(ConfigSpec.default_config())",
        "config": {
            "row": row_index,
            "block": 128,
            "tcgen05_grouped_mode": "worklist_nm",
            "tcgen05_grouped_worklist_source_m_tile": campaign.SOURCE_M_TILE,
        },
    }


def _result(provider: str, replicate: int) -> dict[str, Any]:
    return {
        "schema": campaign.RESULT_SCHEMA,
        "provider": provider,
        "replicate": replicate,
        "device": {
            "visible": "GPU-test",
            "name": "NVIDIA B200",
            "capability": [10, 0],
            "uuid": "GPU-test",
            "multi_processor_count": 148,
            "total_memory": 192_000_000_000,
        },
        "source": SOURCE_IDENTITY,
        "versions": _versions(),
        "rows": [
            {
                "case": case.as_dict(),
                "configs": {
                    "helion": _helion_config(case.row_index),
                    "provider": _provider_config(provider, case.row_index),
                },
                "correctness": {
                    "helion": _accuracy_evidence(replay=True, group_count=case.groups),
                    "provider": _accuracy_evidence(
                        replay=True, group_count=case.groups
                    ),
                },
                "timings": {"helion_ms": 1.0, "provider_ms": 2.0},
            }
            for case in OFFICIAL_CASES
        ],
    }


def _campaign_results() -> list[dict[str, Any]]:
    return list(starmap(_result, campaign.publication_runs()))


def _campaign_result(
    results: list[dict[str, Any]], provider: str, replicate: int
) -> dict[str, Any]:
    return next(
        result
        for result in results
        if result["provider"] == provider and result["replicate"] == replicate
    )


PROVIDER_PIN_PATHS = (
    ("deepgemm", ("provenance", "git_head")),
    ("quack", ("package", "upstream_commit")),
    ("cudnn", ("plan", "selection")),
    ("cublaslt", ("selected_algorithm", "heuristic_rank")),
    ("cutlass", ("commit",)),
)


def test_cli_smokes_fixed_plan_and_legacy_dispatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert compare_grouped_gemm_backends.main(["--provider-defaults-plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["providers"] == list(campaign.PROVIDERS)
    assert plan["replicates"] == campaign.PUBLICATION_REPLICATES
    assert plan["worker_count"] == len(campaign.publication_runs())
    assert plan["run_order"] == [
        {"provider": provider, "replicate": replicate}
        for provider, replicate in campaign.publication_runs()
    ]
    assert plan["cases"] == [case.as_dict() for case in OFFICIAL_CASES]
    assert plan["helion"]["selection"] == campaign.HELION_SELECTION
    assert plan["helion"]["required_heuristic"] == campaign.GROUPED_WORKLIST_HEURISTIC
    assert plan["provider_selections"] == dict(campaign.PROVIDER_SELECTION_MODES)
    assert plan["protocol"]["balanced_rotated_reversed_order"] is True
    assert plan["protocol"]["fail_closed_source_gpu_stack_and_telemetry"] is True

    assert compare_grouped_gemm_backends.main(["--list-cases", "--json"]) == 0
    legacy_cases = json.loads(capsys.readouterr().out)
    assert legacy_cases
    assert {"name", "shape_label", "problems"} <= legacy_cases[0].keys()


def test_helion_requires_grouped_compiler_primary_and_effective_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler_default = {
        "tcgen05_grouped_mode": "worklist_nm",
        "tcgen05_grouped_worklist_source_m_tile": campaign.SOURCE_M_TILE,
        "optional": None,
    }
    config_spec = SimpleNamespace(
        compiler_default_config=SimpleNamespace(config=compiler_default),
        autotuner_heuristics=[campaign.GROUPED_WORKLIST_HEURISTIC],
        default_config=lambda: SimpleNamespace(config={}),
    )
    kernel = SimpleNamespace(
        bind=lambda _args: SimpleNamespace(config_spec=config_spec)
    )
    inputs = SimpleNamespace(b=object())
    packed = SimpleNamespace(a=object(), worklist=object())
    monkeypatch.setattr(campaign, "_make_helion_kernel", lambda: kernel)
    monkeypatch.setattr(common, "pack_compact_rows", lambda *_args: packed)

    config_spec.autotuner_heuristics = []
    with pytest.raises(RuntimeError, match="did not fire exactly once"):
        campaign.prepare_helion(cast("Any", inputs))
    config_spec.autotuner_heuristics = [campaign.GROUPED_WORKLIST_HEURISTIC]
    for invalid_compiler_default in (
        {},
        {"tcgen05_grouped_mode": "worklist_nm"},
        {
            "tcgen05_grouped_mode": "worklist_nm",
            "tcgen05_grouped_worklist_source_m_tile": 128,
        },
    ):
        config_spec.compiler_default_config = SimpleNamespace(
            config=invalid_compiler_default
        )
        config_spec.default_config = lambda: SimpleNamespace(config=compiler_default)
        with pytest.raises(RuntimeError, match="primary config"):
            campaign.prepare_helion(cast("Any", inputs))
    config_spec.compiler_default_config = SimpleNamespace(config=compiler_default)
    for effective in (
        {},
        {"tcgen05_grouped_mode": "worklist_nm"},
        {
            "tcgen05_grouped_mode": "worklist_nm",
            "tcgen05_grouped_worklist_source_m_tile": campaign.SOURCE_M_TILE,
        },
    ):
        config_spec.default_config = lambda effective=effective: SimpleNamespace(
            config=effective
        )
        with pytest.raises(RuntimeError, match="not the grouped-worklist default"):
            campaign.prepare_helion(cast("Any", inputs))
    config_spec.autotuner_heuristics *= 2
    with pytest.raises(RuntimeError, match="did not fire exactly once"):
        campaign.prepare_helion(cast("Any", inputs))


@pytest.mark.parametrize(
    ("provider", "module", "preparer", "root_parameter"), PROVIDER_ADAPTERS
)
def test_provider_dispatches_all_public_defaults(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    module: object,
    preparer: str,
    root_parameter: str | None,
) -> None:
    assert tuple(case[0] for case in PROVIDER_ADAPTERS) == campaign.PROVIDERS
    calls: list[tuple[object, dict[str, object]]] = []
    inputs = cast("Any", object())
    roots = {name: Path(f"/{name}") for name in ("deepgemm", "quack", "cutlass")}
    prepared = _prepared_provider(provider)

    def prepare(
        actual_inputs: object, **kwargs: object
    ) -> common.PreparedImplementation:
        calls.append((actual_inputs, kwargs))
        return prepared

    monkeypatch.setattr(module, preparer, prepare)

    def dispatch() -> common.PreparedImplementation:
        return campaign.prepare_provider_default(
            provider,
            inputs,
            deepgemm_root=roots["deepgemm"],
            quack_root=roots["quack"],
            cutlass_root=roots["cutlass"],
        )

    result = dispatch()
    expected_kwargs = (
        {} if root_parameter is None else {root_parameter: roots[provider]}
    )
    assert calls == [(inputs, expected_kwargs)]
    assert result is prepared
    prepared.config["b_layout"] = "n_major"
    with pytest.raises(RuntimeError, match="invalid selection contract"):
        dispatch()


def test_source_identity_rejects_dirty_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        campaign,
        "_git_value",
        lambda *arguments: "dirty.py" if arguments[0] == "status" else "identity",
    )

    with pytest.raises(RuntimeError, match="clean Helion checkout"):
        campaign._source_identity()


@pytest.mark.parametrize(
    "provider_output_ok", (True, False), ids=("success", "timing-mutation")
)
def test_run_case_preserves_validation_and_rejects_timing_mutation(
    monkeypatch: pytest.MonkeyPatch,
    provider_output_ok: bool,
) -> None:
    from pretuned_kernels import _bench

    events: list[tuple[object, ...]] = []
    case = OFFICIAL_CASES[0]
    device = torch.device("cuda", 0)
    inputs = SimpleNamespace(oracle=(object(),))
    helion = SimpleNamespace(name="helion", config={"config": "helion"})
    provider = SimpleNamespace(name="provider", config={})

    def helion_replay() -> None:
        raise AssertionError("the test timer must not execute GPU work")

    def provider_replay() -> None:
        raise AssertionError("the test timer must not execute GPU work")

    captures = {
        "helion": SimpleNamespace(replay=helion_replay, name="helion"),
        "provider": SimpleNamespace(replay=provider_replay, name="provider"),
    }

    def make_inputs(
        actual_case: common.GroupedGemmCase,
        actual_device: torch.device,
        *,
        seed: int,
    ) -> object:
        events.append(("inputs", actual_case.id, actual_device, seed))
        return inputs

    def validated_capture(prepared: Any, oracle: object) -> tuple[object, object]:
        assert oracle is inputs.oracle
        events.append(("validate", prepared.name))
        return captures[prepared.name], _accuracy_evidence(replay=True)

    def benchmark(calls: Any, *, rep: int) -> list[float]:
        callbacks = list(calls)
        events.append(
            (
                "timer",
                rep,
                callbacks[0] is helion_replay,
                callbacks[1] is provider_replay,
            )
        )
        return [1.0, 2.0]

    failed_check: dict[str, object] = {
        "ok": False,
        "groups": [],
        "max_abs_diff": 1.0,
    }
    checks = iter(
        (
            _accuracy_evidence(),
            _accuracy_evidence() if provider_output_ok else failed_check,
        )
    )

    def check(captured: Any, oracle: object) -> dict[str, object]:
        assert oracle is inputs.oracle
        events.append(("post", captured.name))
        return next(checks)

    monkeypatch.setattr(common, "make_inputs", make_inputs)
    monkeypatch.setattr(
        campaign,
        "prepare_helion",
        lambda _inputs: events.append(("prepare", "helion")) or helion,
    )
    monkeypatch.setattr(
        campaign,
        "prepare_provider_default",
        lambda name, _inputs, **_kwargs: events.append(("prepare", name)) or provider,
    )
    monkeypatch.setattr(campaign, "_validated_capture", validated_capture)
    monkeypatch.setattr(torch.random, "fork_rng", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(
        _bench,
        "thermal_warmup",
        lambda duration: events.append(("thermal_warmup", duration)),
    )
    monkeypatch.setattr(_bench, "bench_pre_captured_cudagraphs", benchmark)
    monkeypatch.setattr(common, "check_correctness", check)

    def run() -> dict[str, object]:
        return campaign.run_case(
            "cudnn",
            case,
            device,
            cutlass_root=None,
            deepgemm_root=None,
        )

    if not provider_output_ok:
        with pytest.raises(RuntimeError, match="provider output changed during"):
            run()
        return

    row = run()

    assert events == [
        ("inputs", case.id, device, case.row_index),
        ("prepare", "helion"),
        ("prepare", "cudnn"),
        ("validate", "helion"),
        ("validate", "provider"),
        ("thermal_warmup", 10_000),
        ("timer", 102, True, True),
        ("post", "helion"),
        ("post", "provider"),
    ]
    assert row["timings"] == {"helion_ms": 1.0, "provider_ms": 2.0}


def test_worker_environment_scrubs_controls_and_uses_fresh_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cuda_stack: tuple[Path, dict[str, Path]],
) -> None:
    cuda_home, artifacts = cuda_stack
    monkeypatch.setenv("HELION_BACKEND", "triton")
    monkeypatch.setenv("PYTHONPATH", "/stale")
    monkeypatch.setenv("LD_PRELOAD", "/stale.so")
    first = campaign._worker_environment(
        tmp_path / "first",
        cuda_visible_devices="GPU-test",
    )
    second = campaign._worker_environment(
        tmp_path / "second",
        cuda_visible_devices="GPU-test",
    )

    assert first["HELION_BACKEND"] == "cute"
    assert first["PYTHONPATH"] == str(campaign.REPO_ROOT)
    assert first["PYTHONNOUSERSITE"] == "1"
    assert first["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert first["CUDA_HOME"] == first["CUDA_PATH"] == str(cuda_home)
    assert first["CUDNN_FRONTEND_CUDART_LIB_NAME"] == str(artifacts["cudart"])
    assert first["PATH"].split(os.pathsep)[0] == str(cuda_home / "bin")
    assert first["LD_PRELOAD"].split(os.pathsep) == [
        str(artifacts[name]) for name in common.CUDA_STACK_PRELOAD_LIBRARY_PREFIXES
    ]
    first_caches = {first[name] for name in campaign.WORKER_CACHE_NAMES}
    second_caches = {second[name] for name in campaign.WORKER_CACHE_NAMES}
    assert first_caches.isdisjoint(second_caches)
    assert all(Path(path).is_dir() for path in first_caches | second_caches)


def test_cuda_toolchain_identity_validates_nvcc_once(
    monkeypatch: pytest.MonkeyPatch,
    cuda_stack: tuple[Path, dict[str, Path]],
) -> None:
    cuda_home, artifacts = cuda_stack
    toolkit_calls: list[Path] = []

    def toolkit_identity(root: Path) -> dict[str, str]:
        toolkit_calls.append(root)
        return {
            "release": common.CUDA_TOOLKIT_RELEASE,
            "compiler_version": common.CUDA_COMPILER_VERSION,
        }

    monkeypatch.setattr(
        campaign,
        "_cuda_toolkit_identity",
        toolkit_identity,
    )
    artifacts_by_prefix = {
        prefix: artifacts[name]
        for name, prefix in common.CUDA_STACK_PRELOAD_LIBRARY_PREFIXES.items()
    }
    monkeypatch.setattr(
        common,
        "mapped_library_paths",
        lambda prefix: (artifacts_by_prefix[prefix],),
    )
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.setenv("CUDNN_FRONTEND_CUDART_LIB_NAME", str(artifacts["cudart"]))
    assert "artifact" not in json.dumps(campaign._cuda_toolchain_identity())
    assert toolkit_calls == [cuda_home]
    monkeypatch.setattr(common, "mapped_library_paths", lambda _prefix: ())
    with pytest.raises(RuntimeError, match="worker loaded"):
        campaign._cuda_toolchain_identity()


def test_worker_revalidates_cuda_libraries_after_all_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events: list[object] = []
    artifacts = {"cudart": tmp_path / "libcudart.so"}
    device_info = {
        "visible": "GPU-test",
        "name": "NVIDIA B200",
        "capability": [10, 0],
        "uuid": "GPU-test",
        "multi_processor_count": 148,
        "total_memory": 192_000_000_000,
    }
    startup_environment = campaign._selection_environment()
    monkeypatch.setattr(
        campaign,
        "HELION_SELECTION_STARTUP_ENVIRONMENT",
        startup_environment,
    )
    monkeypatch.delenv("HELION_HEURISTIC_DIR", raising=False)
    monkeypatch.setattr(common, "require_single_visible_device", lambda: "GPU-test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(campaign, "_device_info", lambda _device: device_info)
    monkeypatch.setattr(
        campaign,
        "_cuda_toolchain_identity",
        lambda: events.append("startup") or _versions()["cuda_stack"],
    )
    monkeypatch.setattr(common, "configure_oracle_precision", lambda: "highest")
    monkeypatch.setattr(
        campaign, "_installed_cuda_stack", lambda: (tmp_path, artifacts)
    )
    monkeypatch.setattr(
        campaign,
        "_validate_mapped_cuda_libraries",
        lambda actual: events.append(("post", actual)),
    )
    monkeypatch.setattr(
        campaign,
        "run_case",
        lambda _provider, case, _device, **_kwargs: (
            events.append(("row", case.row_index)) or {}
        ),
    )
    monkeypatch.setattr(campaign, "_source_identity", lambda: SOURCE_IDENTITY)
    monkeypatch.setattr(campaign, "_cuda_driver_version", lambda _uuid: "590.00")
    monkeypatch.setattr(campaign.importlib.metadata, "version", lambda _name: "1.0")

    assert (
        campaign._run_worker(
            argparse.Namespace(
                provider="cudnn",
                replicate=0,
                run_dir=run_dir,
                deepgemm_root=None,
                quack_root=None,
                cutlass_root=None,
            )
        )
        == 0
    )
    assert events == [
        "startup",
        *(("row", case.row_index) for case in OFFICIAL_CASES),
        ("post", artifacts),
    ]


def test_cublaslt_layout_maps_k_major_storage() -> None:
    transpose, dimensions = cublaslt_grouped_gemm.cublaslt_layout_values(
        ((2, 3, 4, 1), (5, 6, 7, 1))
    )

    assert transpose == cublaslt_grouped_gemm._CUBLAS_OP_T
    assert dimensions == {
        "a_rows": [4, 7],
        "a_columns": [3, 6],
        "a_leading_dimensions": [4, 7],
        "b_rows": [4, 7],
        "b_columns": [2, 5],
        "b_leading_dimensions": [4, 7],
        "output_rows": [3, 6],
        "output_columns": [2, 5],
        "output_leading_dimensions": [3, 6],
    }
    assert cublaslt_grouped_gemm.cublaslt_grouped_preference_values(
        ((2, 3, 4, 1), (5, 6, 7, 1))
    ) == {
        "average_reduction_dim": 5,
        "average_output_rows": 4,
        "average_output_columns": 3,
    }


def test_cutlass_tunes_every_candidate_with_balanced_cold_l2_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pretuned_kernels import _bench

    candidates = tuple(
        cutlass_contiguous_grouped_gemm._CutlassCandidate(
            registry_index=10 + index,
            operator_name=name,
            compiled_for="sm100",
            prepared=common.PreparedImplementation(
                name=name,
                call=lambda: None,
                output_tensors=lambda _result: (),
                logical_outputs=lambda _result: (),
                config={"tile": index},
            ),
        )
        for index, name in enumerate(("alpha", "beta", "gamma"))
    )
    captures = {
        candidate.operator_name: SimpleNamespace(replay=object())
        for candidate in candidates
    }
    captured_names: list[str] = []

    def capture(prepared: common.PreparedImplementation, *, warmups: int) -> object:
        assert warmups == 2
        captured_names.append(prepared.name)
        return captures[prepared.name]

    def benchmark(calls: object, *, rep: int) -> list[float]:
        assert list(cast("Any", calls)) == [
            captures[name].replay for name in ("alpha", "beta", "gamma")
        ]
        assert rep == 36
        return [2.0, 1.0, 1.0]

    monkeypatch.setattr(common, "capture_implementation", capture)
    monkeypatch.setattr(
        common,
        "validate_capture",
        lambda *_args: _accuracy_evidence(replay=True),
    )
    monkeypatch.setattr(_bench, "bench_pre_captured_cudagraphs", benchmark)

    evidence = cutlass_contiguous_grouped_gemm._tune_candidates(
        candidates,
        cast("tuple[torch.Tensor, ...]", (object(),)),
    )

    assert evidence["repetitions"] == 36
    assert evidence["selected_candidate_index"] == 1
    candidate_evidence = cast("list[dict[str, object]]", evidence["candidates"])
    assert [candidate["selection_median_ms"] for candidate in candidate_evidence] == [
        2.0,
        1.0,
        1.0,
    ]
    assert captured_names == ["alpha", "beta", "gamma"]


def test_campaign_has_no_reviewed_profile_or_aot_dependency() -> None:
    source = inspect.getsource(campaign) + inspect.getsource(campaign.common)

    assert "reviewed_profiles" not in source
    assert "aot_kernel" not in source
    assert "deepgemm_selected_path" not in inspect.getsource(campaign.common)


def test_summary_happy_path_and_documented_tuning_variation() -> None:
    results = _campaign_results()
    for row in _campaign_result(results, "quack", 1)["rows"]:
        row["configs"]["provider"]["selected_config"]["block"] = 256
    cutlass_tuning = _campaign_result(results, "cutlass", 1)["rows"][0]["configs"][
        "provider"
    ]["registry_tuning"]
    cutlass_tuning["candidates"][0]["selection_median_ms"] = 1.5

    summary = campaign.summarize_results(results)

    assert summary["cases"] == [case.as_dict() for case in OFFICIAL_CASES]
    assert set(summary["provider_results"]) == set(campaign.PROVIDERS)
    quack = summary["provider_results"]["quack"]
    assert quack["varying_config_rows"] == list(range(len(OFFICIAL_CASES)))
    assert quack["cross_replicate_geomean"] == pytest.approx(2.0)
    assert quack["row_wins"] == len(OFFICIAL_CASES)
    assert summary["provider_results"]["cutlass"]["varying_config_rows"] == []


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("schema", "result identity is inconsistent"),
        ("provider", "changed provider selection"),
        ("timing", "invalid timings"),
        ("correctness", "missing correctness evidence"),
        ("helion", "fixed Helion config changed"),
    ),
)
def test_summary_fails_closed_on_material_evidence(
    mutation: str,
    error: str,
) -> None:
    results = _campaign_results()
    row = results[0]["rows"][0]
    if mutation == "schema":
        results[0]["schema"] = "wrong"
    elif mutation == "provider":
        row["configs"]["provider"]["provenance"]["git_head"] = "wrong"
    elif mutation == "timing":
        row["timings"]["helion_ms"] = float("nan")
    elif mutation == "correctness":
        row["correctness"]["provider"]["post_timing"]["ok"] = False
    else:
        _campaign_result(results, "deepgemm", 1)["rows"][0]["configs"]["helion"][
            "config"
        ]["block"] = 256

    with pytest.raises(RuntimeError, match=error):
        campaign.summarize_results(results)


@pytest.mark.parametrize(("provider", "path"), PROVIDER_PIN_PATHS)
def test_provider_contract_rejects_changed_pin(
    provider: str,
    path: tuple[str, ...],
) -> None:
    case = OFFICIAL_CASES[0]
    config = _provider_config(provider, 0)
    device = _result(provider, 0)["device"]
    assert campaign._valid_provider_contract(
        config,
        provider,
        expected_case=case.as_dict(),
        device=device,
        complete=True,
    )
    parent: dict[str, Any] = config
    for field in path[:-1]:
        parent = cast("dict[str, Any]", parent[field])
    parent[path[-1]] = 1 if parent[path[-1]] == 0 else "wrong"
    assert not campaign._valid_provider_contract(
        config,
        provider,
        expected_case=case.as_dict(),
        device=device,
        complete=True,
    )


def test_summary_rejects_selection_identity_drift() -> None:
    results = _campaign_results()
    quack = _campaign_result(results, "quack", 1)["rows"][0]["configs"]["provider"]
    quack["package"]["build_identity"] = "different"
    with pytest.raises(RuntimeError, match="provenance changed across replicates"):
        campaign.summarize_results(results)

    results = _campaign_results()
    tuning = _campaign_result(results, "cutlass", 1)["rows"][0]["configs"]["provider"][
        "registry_tuning"
    ]
    tuning["candidates"][0]["selection_median_ms"] = 2.0
    tuning["candidates"][1]["selection_median_ms"] = 1.0
    tuning["selected_candidate_index"] = 1
    with pytest.raises(RuntimeError, match="cutlass selected config changed"):
        campaign.summarize_results(results)


def test_monitored_worker_rejects_transient_foreign_application(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = SimpleNamespace(pid=1234, poll=lambda: None)
    clean = (campaign._ComputeApplication(1234, "worker"),)
    foreign = (campaign._ComputeApplication(9999, "foreign"),)
    samples = iter((clean, foreign))
    terminated: list[object] = []
    monkeypatch.setattr(campaign, "COMPUTE_APPLICATION_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(
        campaign.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(
        campaign,
        "_query_telemetry",
        lambda _uuid: TELEMETRY_SAMPLE,
    )
    monkeypatch.setattr(
        campaign,
        "_query_compute_applications",
        lambda _uuid: next(samples),
    )
    monkeypatch.setattr(
        campaign.os,
        "getpgid",
        lambda pid: 1234 if pid == 1234 else 9999,
    )
    monkeypatch.setattr(campaign, "_terminate_process", terminated.append)
    monkeypatch.setattr(campaign, "_require_target_gpu_idle", lambda _uuid: None)
    monkeypatch.setattr(campaign, "_wait_for_target_gpu_idle", lambda _uuid: None)
    monkeypatch.setattr(campaign.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(campaign.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="outside worker process group 1234"):
        campaign._run_monitored_worker(
            ("worker",),
            environment={},
            log_path=tmp_path / "worker.log",
            target_gpu_uuid="GPU-test",
        )

    assert terminated == [process]


def test_gpu_idle_gate_rejects_existing_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        campaign,
        "_query_compute_applications",
        lambda _uuid: (campaign._ComputeApplication(9999, "foreign"),),
    )

    with pytest.raises(RuntimeError, match="target GPU is not idle"):
        campaign._require_target_gpu_idle("GPU-test")


def test_live_telemetry_is_compact_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        campaign,
        "_nvidia_smi",
        lambda *_args: "GPU-test, P0, 1000, 2000, 500, 1000, 0x1\n",
    )
    sample = campaign._query_telemetry("GPU-test")
    second = replace(
        sample,
        pstate="P1",
        sm_clock_mhz=800.0,
        memory_clock_mhz=1800.0,
        power_draw_watts=400.0,
        active_clock_event_reasons=0,
    )
    summary = campaign._summarize_telemetry((sample, second))
    assert cast("dict[str, float]", summary["power_draw_watts"])["mean"] == 450.0
    assert summary["active_clock_event_reasons"] == {"0x1": 1}
    for changed, message in (
        (replace(sample, power_draw_watts=-1.0), "invalid power or clock value"),
        (replace(sample, power_limit_watts=900.0), "power limit changed"),
        (replace(sample, active_clock_event_reasons=0x9), "disallowed"),
    ):
        with pytest.raises(RuntimeError, match=message):
            campaign._summarize_telemetry((sample, changed))
    with pytest.raises(RuntimeError, match="target GPU UUID"):
        campaign._query_telemetry("GPU-other")


@pytest.mark.parametrize(
    ("mutation", "error", "expected_launch_count"),
    (
        ("provider", "changed provider selection", 1),
        ("source", "source identity changed", 2),
        ("device", "GPU identity changed", 2),
        ("versions", "software versions changed", 2),
    ),
)
def test_campaign_validates_each_worker_before_launching_the_next(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    error: str,
    expected_launch_count: int,
) -> None:
    launches: list[tuple[str, int]] = []

    def run_worker(
        command: list[str], **_kwargs: object
    ) -> tuple[int, list[campaign._TelemetrySample], int]:
        provider = command[command.index("--provider") + 1]
        replicate = int(command[command.index("--provider-replicate") + 1])
        launches.append((provider, replicate))
        result = _result(provider, replicate)
        if mutation == "provider":
            result["rows"][0]["configs"]["provider"]["provenance"]["git_head"] = "wrong"
        elif len(launches) == 2:
            if mutation == "source":
                result["source"] = "other-commit"
            elif mutation == "device":
                result["device"]["multi_processor_count"] += 1
            else:
                result["versions"]["cuda_driver"] = "591.00"
        run_dir = Path(command[command.index("--provider-run-dir") + 1])
        common.write_result(run_dir / "result.json", result)
        return 0, [TELEMETRY_SAMPLE], 1

    monkeypatch.setattr(campaign, "_provider_roots", lambda _args: {})
    monkeypatch.setattr(campaign, "_resolve_target_gpu", lambda _selector: "GPU-test")
    monkeypatch.setattr(campaign, "_source_identity", lambda: SOURCE_IDENTITY)
    monkeypatch.setattr(campaign, "_worker_environment", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(campaign, "_run_monitored_worker", run_worker)

    with pytest.raises(RuntimeError, match=error):
        campaign._run_campaign(
            argparse.Namespace(
                cuda_visible_devices="0", output_dir=tmp_path / "campaign"
            )
        )

    assert launches == list(campaign.publication_runs()[:expected_launch_count])


def test_campaign_records_compact_per_run_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "campaign"

    def run_worker(
        command: list[str], **_kwargs: object
    ) -> tuple[int, list[campaign._TelemetrySample], int]:
        provider = command[command.index("--provider") + 1]
        replicate = int(command[command.index("--provider-replicate") + 1])
        run_dir = Path(command[command.index("--provider-run-dir") + 1])
        common.write_result(run_dir / "result.json", _result(provider, replicate))
        return 0, [TELEMETRY_SAMPLE], replicate + 1

    monkeypatch.setattr(campaign, "_provider_roots", lambda _args: {})
    monkeypatch.setattr(campaign, "_resolve_target_gpu", lambda _selector: "GPU-test")
    monkeypatch.setattr(campaign, "_source_identity", lambda: SOURCE_IDENTITY)
    monkeypatch.setattr(campaign, "_worker_environment", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(campaign, "_run_monitored_worker", run_worker)
    monkeypatch.setattr(campaign, "_print_summary", lambda _summary: None)

    assert (
        campaign._run_campaign(
            argparse.Namespace(cuda_visible_devices="0", output_dir=output_dir)
        )
        == 0
    )
    monitoring = json.loads((output_dir / "summary.json").read_text())["monitoring"]
    per_run = monitoring["by_provider_replicate"]
    assert {
        run_id: (entry["sample_count"], entry["compute_application_sample_count"])
        for run_id, entry in per_run.items()
    } == {
        f"{provider}-r{replicate}": (1, replicate + 1)
        for provider, replicate in campaign.publication_runs()
    }
    assert monitoring["compute_application_sample_count"] == sum(
        replicate + 1 for _provider, replicate in campaign.publication_runs()
    )
    assert all("samples" not in entry for entry in per_run.values())
