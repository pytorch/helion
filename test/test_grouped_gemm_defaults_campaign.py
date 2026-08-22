from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
from itertools import starmap
import json
import math
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
from typing import Any
from typing import cast

from benchmarks.cute import compare_grouped_gemm_defaults as benchmark
from benchmarks.cute import grouped_gemm_benchmark as common
from pretuned_kernels.grouped_gemm_deepgemm.reviewed_profiles import OfficialShape
import pytest
import torch

SOURCE = {"commit": "source", "tree": "tree"}


def _versions(
    *,
    cuda_driver: str = "590.00",
    cuda_home: str = "/cuda-a",
) -> dict[str, object]:
    return {
        "python": "3.12.0",
        "torch": "2.10.0.dev",
        "torch_cuda": "13.0",
        "triton": "3.6.0",
        "cutlass_dsl": "4.7.0",
        "cuda_driver": cuda_driver,
        "cuda_stack": {
            "distribution_versions": dict(common.CUDA_STACK_DISTRIBUTION_VERSIONS),
            "release": common.CUDA_TOOLKIT_RELEASE,
            "compiler_version": common.CUDA_COMPILER_VERSION,
            "artifact_sha256": {
                label: f"sha256-{label}"
                for label in common.CUDA_STACK_REQUIRED_ARTIFACTS
            },
            "loaded_preload_sha256": {
                label: f"sha256-{label}"
                for label in common.CUDA_STACK_PRELOAD_LIBRARY_PREFIXES
            },
            "cuda_home": cuda_home,
            "nvcc": f"{cuda_home}/bin/nvcc",
            "cudart": f"{cuda_home}/lib/libcudart.so.13",
        },
    }


def _row(
    row_index: int,
    speedup: float,
    config_hash: str,
    provider_config_hash: str,
) -> dict[str, object]:
    return {
        "row_index": row_index,
        "configs": {
            "helion": {"config_sha256": config_hash},
            "provider": {"config_sha256": provider_config_hash},
        },
        "timings": {
            "helion_ms": 1.0,
            "provider_ms": speedup,
            "helion_speedup": speedup,
        },
    }


def _result(
    provider: str,
    replicate: int,
    *,
    speedup: float = 2.0,
    row_speedups: tuple[float, ...] | None = None,
    config_prefix: str = "fixed",
    provider_config_prefix: str | None = None,
    helion_selection: str = "compiler_heuristic",
    versions: dict[str, object] | None = None,
) -> dict[str, object]:
    speedups = row_speedups or tuple(speedup + index / 100 for index in range(8))
    assert len(speedups) == 8
    return {
        "schema": benchmark.RESULT_SCHEMA,
        "provider": provider,
        "replicate": replicate,
        "helion_selection": helion_selection,
        "settings": {"layout_policy": benchmark.BENCHMARK_LAYOUT_POLICY},
        "device": {"uuid": "GPU-test", "visible": "GPU-test"},
        "source": SOURCE,
        "versions": versions or _versions(),
        "rows": [
            _row(
                row_index,
                row_speedup,
                f"{config_prefix}-{row_index}",
                f"{provider_config_prefix or provider}-{row_index}",
            )
            for row_index, row_speedup in enumerate(speedups)
        ],
    }


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "providers": ("quack", "cudnn"),
        "replicates": 2,
        "helion_selection": "compiler_heuristic",
        "output_dir": tmp_path / "campaign",
        "cuda_visible_devices": "GPU-test",
        "deepgemm_root": None,
        "cutlass_root": None,
        "quack_root": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _mock_cuda_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cuda_home = tmp_path / "cuda-runtime"
    artifacts = {
        "cudart": cuda_home / "lib" / "libcudart.so.13",
        "cublas": cuda_home / "lib" / "libcublas.so.13",
        "cublaslt": cuda_home / "lib" / "libcublasLt.so.13",
        "nvrtc": cuda_home / "lib" / "libnvrtc.so.13",
    }
    monkeypatch.setattr(
        benchmark,
        "_installed_cuda_stack",
        lambda: (cuda_home, artifacts),
    )


def test_parse_provider_subset_preserves_order_and_rejects_invalid() -> None:
    assert benchmark.parse_providers("quack,cudnn,cublaslt") == (
        "quack",
        "cudnn",
        "cublaslt",
    )
    for invalid in ("", "quack,quack", "unknown"):
        with pytest.raises(argparse.ArgumentTypeError):
            benchmark.parse_providers(invalid)


def test_cuda_stack_uses_latest_pinned_releases() -> None:
    assert common.CUDA_STACK_DISTRIBUTION_VERSIONS == {
        "nvidia-cuda-runtime": "13.3.29",
        "nvidia-cuda-nvcc": "13.3.73",
        "nvidia-nvvm": "13.3.73",
        "nvidia-cuda-crt": "13.3.73",
        "nvidia-cuda-nvrtc": "13.3.33",
        "nvidia-cublas": "13.6.1.10",
    }


def test_public_cli_requires_external_output_directory() -> None:
    parser = benchmark.build_arg_parser()
    assert parser.parse_args(("--cuda-visible-devices", "GPU-test")).output_dir is None
    with pytest.raises(ValueError, match="--output-dir is required"):
        benchmark.main(("--cuda-visible-devices", "GPU-test"))
    args = parser.parse_args(
        (
            "--cuda-visible-devices",
            "GPU-test",
            "--output-dir",
            "/tmp/grouped-gemm-results",
        )
    )
    assert args.output_dir == Path("/tmp/grouped-gemm-results")
    assert args.providers == benchmark.PROVIDERS
    assert args.replicates == 3
    assert args.helion_selection == "compiler_heuristic"
    assert not args.worker
    assert args.provider is args.replicate is args.run_dir is None


@pytest.mark.parametrize("relative_pythonpath", (False, True))
def test_direct_script_uses_helpers_from_its_checkout(
    tmp_path: Path,
    relative_pythonpath: bool,
) -> None:
    stale_root = tmp_path / "stale"
    stale_module = stale_root / "benchmarks" / "cute" / "grouped_gemm_benchmark.py"
    stale_module.parent.mkdir(parents=True)
    (stale_root / "benchmarks" / "__init__.py").write_text("")
    (stale_root / "benchmarks" / "cute" / "__init__.py").write_text("")
    stale_module.write_text("raise RuntimeError('imported stale benchmark helper')\n")
    completed = subprocess.run(
        (benchmark.sys.executable, str(Path(benchmark.__file__)), "--help"),
        check=False,
        capture_output=True,
        cwd=stale_root if relative_pythonpath else tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": "." if relative_pythonpath else str(stale_root),
        },
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_provider_roots_are_required_only_when_selected(tmp_path: Path) -> None:
    deepgemm = tmp_path / "deepgemm"
    cutlass = tmp_path / "cutlass"
    quack = tmp_path / "quack"
    deepgemm.mkdir()
    cutlass.mkdir()
    quack.mkdir()
    args = _args(
        tmp_path,
        providers=("deepgemm", "cutlass", "quack"),
        deepgemm_root=deepgemm,
        cutlass_root=cutlass,
        quack_root=quack,
    )
    assert benchmark._provider_roots(args) == {
        "deepgemm": deepgemm,
        "cutlass": cutlass,
        "quack": quack,
    }
    args.deepgemm_root = None
    with pytest.raises(ValueError, match="deepgemm-root"):
        benchmark._provider_roots(args)

    args.providers = ("cutlass",)
    args.quack_root = quack
    with pytest.raises(ValueError, match="quack-root"):
        benchmark._provider_roots(args)


def test_source_identity_requires_clean_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "_git_value",
        lambda *args: "dirty.py" if args[0] == "status" else args[-1],
    )
    with pytest.raises(RuntimeError, match="clean Helion checkout"):
        benchmark._source_identity()


def test_make_inputs_is_seed_repeatable_and_changes_both_operands() -> None:
    case = common.GroupedGemmCase(
        id="seed-test",
        row_index=5,
        groups=2,
        expected_m_per_group=2,
        n=3,
        k=4,
        actual_ms=(2, 1),
    )
    first = common.make_inputs(case, torch.device("cpu"), seed=17)
    repeated = common.make_inputs(case, torch.device("cpu"), seed=17)
    changed = common.make_inputs(case, torch.device("cpu"), seed=18)

    assert torch.equal(first.compact_a, repeated.compact_a)
    assert torch.equal(first.b, repeated.b)
    assert all(starmap(torch.equal, zip(first.oracle, repeated.oracle, strict=True)))
    assert not torch.equal(first.compact_a, changed.compact_a)
    assert not torch.equal(first.b, changed.b)


def test_run_case_uses_row_seed_and_bridges_provider_config_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pretuned_kernels import _bench

    case = common.GroupedGemmCase(
        id="run-case-test",
        row_index=6,
        groups=1,
        expected_m_per_group=1,
        n=1,
        k=1,
        actual_ms=(1,),
    )
    shape = OfficialShape(6, 1, 1, 1, 1)
    inputs = SimpleNamespace(oracle=())
    profile = object()
    seeds: list[int] = []
    provider_config = {"provider": "test", "nested": {"value": 7}}
    helion = SimpleNamespace(config={"config_sha256": "helion"}, name="helion")
    provider_impl = SimpleNamespace(config=provider_config, name="provider")
    capture = SimpleNamespace(replay=lambda: None)

    def make_inputs(
        _case: object,
        _shape: object,
        _device: object,
        *,
        seed: int,
    ) -> tuple[object, object]:
        seeds.append(seed)
        return inputs, profile

    monkeypatch.setattr(benchmark, "make_exact_common_inputs", make_inputs)
    monkeypatch.setattr(benchmark, "prepare_helion", lambda *_args: helion)
    monkeypatch.setattr(
        benchmark,
        "prepare_provider_default",
        lambda *_args, **_kwargs: provider_impl,
    )
    monkeypatch.setattr(
        benchmark,
        "_validated_capture",
        lambda *_args: (capture, {"ok": True}),
    )
    monkeypatch.setattr(torch.random, "fork_rng", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(_bench, "thermal_warmup", lambda _duration: None)
    monkeypatch.setattr(
        _bench,
        "bench_pre_captured_cudagraphs",
        lambda _calls, *, rep: [1.0, 2.0],
    )

    result = benchmark.run_case(
        "quack",
        case,
        shape,
        torch.device("cuda", 0),
        helion_selection="compiler_heuristic",
        cutlass_root=None,
        deepgemm_root=None,
    )

    assert seeds == [6]
    configs = result["configs"]
    assert isinstance(configs, dict)
    recorded_provider_config = configs["provider"]
    assert isinstance(recorded_provider_config, dict)
    assert recorded_provider_config["config_sha256"] == common.config_sha256(
        provider_config
    )


def _mock_pinned_cuda_distributions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    version_overrides: dict[str, str] | None = None,
) -> tuple[Path, dict[str, Path]]:
    package_root = tmp_path / "site"
    artifact_paths = {
        label: package_root / relative_path
        for label, (_name, relative_path) in (
            common.CUDA_STACK_REQUIRED_ARTIFACTS.items()
        )
    }
    distribution_files: dict[str, list[Path]] = {
        name: [] for name in common.CUDA_STACK_DISTRIBUTION_VERSIONS
    }
    for label, (name, relative_path) in common.CUDA_STACK_REQUIRED_ARTIFACTS.items():
        path = artifact_paths[label]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        distribution_files[name].append(relative_path)
    distributions = {}
    for name, relative_paths in distribution_files.items():
        distributions[name] = SimpleNamespace(
            version=(version_overrides or {}).get(
                name, common.CUDA_STACK_DISTRIBUTION_VERSIONS[name]
            ),
            files=tuple(relative_paths),
            locate_file=lambda path, root=package_root: root / path,
        )
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        distributions.__getitem__,
    )
    cuda_home = package_root / "nvidia" / "cu13"
    monkeypatch.setattr(
        benchmark,
        "_cuda_toolkit_identity",
        lambda path: {
            "cuda_home": str(path.resolve()),
            "nvcc": str(path.resolve() / "bin" / "nvcc"),
            "release": common.CUDA_TOOLKIT_RELEASE,
            "compiler_version": common.CUDA_COMPILER_VERSION,
        },
    )
    return cuda_home.resolve(), {
        label: path.resolve() for label, path in artifact_paths.items()
    }


def test_installed_cuda_stack_requires_exact_pinned_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cuda_home, artifacts = _mock_pinned_cuda_distributions(monkeypatch, tmp_path)
    external_toolkit = tmp_path / "external-cuda"
    (external_toolkit / "bin").mkdir(parents=True)
    (external_toolkit / "bin" / "nvcc").touch()
    monkeypatch.setenv("CUDA_HOME", str(external_toolkit))
    assert benchmark._installed_cuda_stack() == (cuda_home, artifacts)


def test_cuda_toolchain_identity_records_actual_pinned_distributions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cuda_home, artifacts = _mock_pinned_cuda_distributions(monkeypatch, tmp_path)
    cudart = artifacts["cudart"]
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.setenv("CUDNN_FRONTEND_CUDART_LIB_NAME", str(cudart))
    mapped = {
        "libcudart.so": artifacts["cudart"],
        "libcublas.so": artifacts["cublas"],
        "libcublasLt.so": artifacts["cublaslt"],
        "libnvrtc.so": artifacts["nvrtc"],
    }
    monkeypatch.setattr(
        common,
        "mapped_library_paths",
        lambda prefix: (mapped[prefix],),
    )

    identity = benchmark._cuda_toolchain_identity()

    assert identity["distribution_versions"] == (
        common.CUDA_STACK_DISTRIBUTION_VERSIONS
    )
    assert identity["release"] == common.CUDA_TOOLKIT_RELEASE
    assert identity["compiler_version"] == common.CUDA_COMPILER_VERSION
    assert identity["loaded_preload_sha256"] == {
        label: common.file_sha256(artifacts[label])
        for label in common.CUDA_STACK_PRELOAD_LIBRARY_PREFIXES
    }
    artifact_sha256 = cast("dict[str, str]", identity["artifact_sha256"])
    assert set(artifact_sha256) == set(common.CUDA_STACK_REQUIRED_ARTIFACTS)


def test_cuda_toolchain_identity_rejects_mapped_runtime_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cuda_home, artifacts = _mock_pinned_cuda_distributions(monkeypatch, tmp_path)
    cudart = artifacts["cudart"]
    foreign = tmp_path / "foreign" / "libcudart.so.13"
    foreign.parent.mkdir()
    foreign.touch()
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.setenv("CUDNN_FRONTEND_CUDART_LIB_NAME", str(cudart))
    mapped = {
        "libcublas.so": artifacts["cublas"],
        "libcublasLt.so": artifacts["cublaslt"],
        "libnvrtc.so": artifacts["nvrtc"],
    }
    monkeypatch.setattr(
        common,
        "mapped_library_paths",
        lambda prefix: (foreign,) if prefix == "libcudart.so" else (mapped[prefix],),
    )

    with pytest.raises(RuntimeError, match="loaded cudart libraries"):
        benchmark._cuda_toolchain_identity()


@pytest.mark.parametrize("distribution_name", common.CUDA_STACK_DISTRIBUTION_VERSIONS)
def test_installed_cuda_stack_rejects_any_distribution_version_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    distribution_name: str,
) -> None:
    _mock_pinned_cuda_distributions(
        monkeypatch,
        tmp_path,
        version_overrides={distribution_name: "0.0.0"},
    )
    with pytest.raises(RuntimeError, match=f"{distribution_name} is 0.0.0"):
        benchmark._installed_cuda_stack()


def test_installed_cuda_stack_rejects_nvcc_binary_version_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_pinned_cuda_distributions(monkeypatch, tmp_path)
    monkeypatch.setattr(
        benchmark,
        "_cuda_toolkit_identity",
        lambda path: {
            "cuda_home": str(path.resolve()),
            "nvcc": str(path.resolve() / "bin" / "nvcc"),
            "release": "13.3",
            "compiler_version": "13.3.72",
        },
    )
    with pytest.raises(RuntimeError, match="V13.3.72, expected release"):
        benchmark._installed_cuda_stack()


def test_cuda_toolkit_identity_records_nvcc_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    completed = SimpleNamespace(
        returncode=0,
        stdout="Cuda compilation tools, release 13.3, V13.3.73\n",
        stderr="",
    )
    monkeypatch.setattr(
        benchmark.subprocess, "run", lambda *_args, **_kwargs: completed
    )

    assert benchmark._cuda_toolkit_identity(tmp_path) == {
        "cuda_home": str(tmp_path.resolve()),
        "nvcc": str(tmp_path.resolve() / "bin" / "nvcc"),
        "release": "13.3",
        "compiler_version": "13.3.73",
    }


@pytest.mark.parametrize(
    ("selection", "aot_mode", "skip_cache"),
    (
        ("final_reviewed_aot", "evaluate", False),
        ("compiler_heuristic", "disabled", False),
        ("live_autotune", "disabled", True),
    ),
)
def test_worker_environment_sets_mode_and_fresh_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selection: str,
    aot_mode: str,
    skip_cache: bool,
) -> None:
    _mock_cuda_runtime(monkeypatch, tmp_path)
    stale = {
        "CC": "/stale",
        "CUBLAS_WORKSPACE_CONFIG": "stale",
        "CUDA_HOME": "/stale",
        "CUDNN_LOGLEVEL_DBG": "stale",
        "DG_JIT_CACHE_DIR": "/stale",
        "HELION_AOT_MODE": "stale",
        "HELION_AUTOTUNE_BEST_OF_K": "99",
        "HELION_BACKEND": "triton",
        "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "7",
        "HELION_CUTE_MMA_IMPL": "stale",
        "HELION_HEURISTIC_DIR": "/stale",
        "HELION_SKIP_CACHE": "stale",
        "LD_PRELOAD": "/stale",
        "NVIDIA_TF32_OVERRIDE": "stale",
        "PYTHONHOME": "/stale",
        "PYTHONPATH": "/stale",
        "PYTORCH_CUDA_ALLOC_CONF": "stale",
        "QUACK_CACHE_DIR": "/stale",
        "CUTLASS_LOG_LEVEL": "stale",
        "TORCH_LOGS": "stale",
        "TRITON_CACHE_DIR": "/stale",
    }
    for name, value in stale.items():
        monkeypatch.setenv(name, value)
    run_dir = tmp_path / selection
    environment = benchmark._worker_environment(
        run_dir,
        cuda_visible_devices="GPU-test",
        helion_selection=selection,
    )
    other = benchmark._worker_environment(
        tmp_path / f"{selection}-other",
        cuda_visible_devices="GPU-test",
        helion_selection=selection,
    )
    assert environment["HELION_AOT_MODE"] == aot_mode
    assert ("HELION_SKIP_CACHE" in environment) is skip_cache
    assert environment["HELION_BACKEND"] == "cute"
    assert environment["HELION_CUTE_MMA_IMPL"] == "tcgen05"
    assert environment["HELION_AUTOTUNER"] == "LFBOTreeSearch"
    assert all(environment.get(name) != value for name, value in stale.items())
    for name in (
        "HELION_AUTOTUNE_BEST_OF_K",
        "HELION_HEURISTIC_DIR",
    ):
        assert name not in environment
    assert environment["HELION_CAP_AUTOTUNE_NUM_NEIGHBORS"] == "-1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert environment["LD_PRELOAD"].split(os.pathsep) == [
        str(tmp_path / "cuda-runtime" / "lib" / "libcudart.so.13"),
        str(tmp_path / "cuda-runtime" / "lib" / "libcublas.so.13"),
        str(tmp_path / "cuda-runtime" / "lib" / "libcublasLt.so.13"),
        str(tmp_path / "cuda-runtime" / "lib" / "libnvrtc.so.13"),
    ]
    assert environment["PYTHONPATH"] == str(benchmark.REPO_ROOT)
    assert environment["PYTHONNOUSERSITE"] == "1"
    cache_paths = {environment[name] for name in benchmark.WORKER_CACHE_NAMES}
    other_cache_paths = {other[name] for name in benchmark.WORKER_CACHE_NAMES}
    assert cache_paths.isdisjoint(other_cache_paths)
    assert all(Path(path).is_dir() for path in cache_paths | other_cache_paths)


def test_worker_command_reexecutes_same_script_with_fresh_pycache(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, helion_selection="compiler_heuristic")
    run_dir = tmp_path / "run"
    quack_root = tmp_path / "quack"
    roots = {"quack": quack_root}
    command = benchmark._worker_command(args, "quack", 2, run_dir, roots)
    other_command = benchmark._worker_command(
        args,
        "quack",
        2,
        tmp_path / "other",
        roots,
    )
    assert command[:4] == [
        benchmark.sys.executable,
        "-u",
        "-X",
        f"pycache_prefix={run_dir / 'cache' / 'pycache'}",
    ]
    assert "--worker" in command
    assert command[command.index("--provider") + 1] == "quack"
    assert command[command.index("--replicate") + 1] == "2"
    assert "--output-dir" not in command
    assert command[command.index("--helion-selection") + 1] == "compiler_heuristic"
    assert Path(command[command.index("--quack-root") + 1]) == quack_root
    assert other_command[3] != command[3]


def test_worker_environment_exposes_quack_source_to_compile_children(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_cuda_runtime(monkeypatch, tmp_path)
    quack_root = tmp_path / "quack"
    quack_root.mkdir()

    environment = benchmark._worker_environment(
        tmp_path / "run",
        cuda_visible_devices="GPU-test",
        helion_selection="compiler_heuristic",
        quack_root=quack_root,
    )

    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(benchmark.REPO_ROOT),
        str(quack_root),
    ]


def test_worker_restores_quack_source_after_direct_script_path_scrub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    quack_root = (tmp_path / "quack").resolve()
    quack_root.mkdir()
    monkeypatch.setattr(benchmark.sys, "path", [str(benchmark.REPO_ROOT)])

    benchmark._restore_provider_import_path("quack", quack_root)
    benchmark._restore_provider_import_path("quack", quack_root)

    assert benchmark.sys.path == [str(benchmark.REPO_ROOT), str(quack_root)]


def test_monitored_worker_records_periodic_and_final_samples_then_cleans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    polls = iter((None, 0))
    events: list[Any] = []
    process: Any = SimpleNamespace(
        pid=1234,
        poll=lambda: next(polls),
        wait=lambda: events.append("wait") or 7,
    )

    def popen(command: object, **kwargs: object) -> object:
        events.append((command, kwargs))
        return process

    samples = iter(("periodic", "final"))
    idle_checks: list[tuple[str, str]] = []
    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    monkeypatch.setattr(benchmark, "_query_telemetry", lambda _device: next(samples))
    monkeypatch.setattr(benchmark, "_terminate_process", events.append)
    monkeypatch.setattr(
        benchmark,
        "_require_target_gpu_idle",
        lambda uuid: idle_checks.append(("before", uuid)),
    )
    monkeypatch.setattr(
        benchmark,
        "_wait_for_target_gpu_idle",
        lambda uuid: idle_checks.append(("after", uuid)),
    )
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _seconds: None)
    telemetry_path = tmp_path / "telemetry.csv"

    assert benchmark._run_monitored_worker(
        ("worker",),
        environment={"TEST": "1"},
        log_path=tmp_path / "worker.log",
        telemetry_path=telemetry_path,
        target_gpu_uuid="GPU-test",
    ) == (7, 2)
    command, kwargs = events[0]
    assert command == ("worker",)
    assert kwargs == {
        "cwd": benchmark.REPO_ROOT,
        "env": {"TEST": "1"},
        "stdin": subprocess.DEVNULL,
        "stdout": kwargs["stdout"],
        "stderr": subprocess.STDOUT,
        "start_new_session": True,
    }
    assert events[1:] == ["wait", process]
    assert idle_checks == [("before", "GPU-test"), ("after", "GPU-test")]
    assert telemetry_path.read_text().splitlines() == [
        ",".join(benchmark.TELEMETRY_FIELDS),
        "periodic",
        "final",
    ]


def test_telemetry_allows_idle_and_sw_power_cap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.csv"
    rows = [
        ",".join(benchmark.TELEMETRY_FIELDS),
        "2026-01-01,GPU-test,P0,1000,100,1000,40,100,1,0x1",
        "2026-01-01,GPU-test,P0,1000,100,1000,40,100,1,0x4",
        "2026-01-01,GPU-test,P0,1000,100,1000,40,100,1,0x5",
    ]
    path.write_text("\n".join(rows) + "\n")

    summary = benchmark._summarize_telemetry(path, "GPU-test")

    assert summary["active_clock_event_reason_sample_count"] == 3
    assert summary["disallowed_clock_event_reason_sample_count"] == 0
    assert summary["disallowed_clock_event_reasons"] == {}
    assert summary["power_limit_watts"] == 1000.0


def test_telemetry_flags_hw_slowdown(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.csv"
    path.write_text(
        "\n".join(
            (
                ",".join(benchmark.TELEMETRY_FIELDS),
                "2026-01-01,GPU-test,P0,1000,100,1000,40,100,1,0x9",
            )
        )
        + "\n"
    )

    summary = benchmark._summarize_telemetry(path, "GPU-test")

    assert summary["disallowed_clock_event_reason_sample_count"] == 1
    assert summary["disallowed_clock_event_reasons"] == {"0x8": 1}


def test_telemetry_rejects_power_limit_changes(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.csv"
    path.write_text(
        "\n".join(
            (
                ",".join(benchmark.TELEMETRY_FIELDS),
                "2026-01-01,GPU-test,P0,1000,100,1000,40,100,1,0x0",
                "2026-01-01,GPU-test,P0,1000,100,900,40,100,1,0x0",
            )
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="power limit changed"):
        benchmark._summarize_telemetry(path, "GPU-test")


def test_terminate_process_escalates_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, signal.Signals]] = []
    waits: list[bool] = []
    process: Any = SimpleNamespace(pid=1234, wait=lambda: waits.append(True))
    monkeypatch.setattr(benchmark, "WORKER_TERMINATION_GRACE_SECONDS", 0)
    monkeypatch.setattr(benchmark, "_process_group_exists", lambda _group: True)
    monkeypatch.setattr(
        benchmark.os,
        "killpg",
        lambda group, signum: signals.append((group, signum)),
    )

    benchmark._terminate_process(process)

    assert signals == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]
    assert waits == [True]


def test_post_worker_idle_check_retries_context_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def require_idle(_uuid: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("CUDA context is still visible")

    monkeypatch.setattr(benchmark, "_require_target_gpu_idle", require_idle)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _seconds: None)
    benchmark._wait_for_target_gpu_idle("GPU-test")
    assert attempts == 2


def test_summary_keeps_provider_geomeans_worst_rows_and_configs_separate() -> None:
    results = [
        _result("quack", 0, row_speedups=(0.25, *(1.0,) * 7)),
        _result("cudnn", 0, row_speedups=(9.0,) * 8),
        _result("quack", 1, row_speedups=(1.0, *(4.0,) * 7)),
        _result("cudnn", 1, row_speedups=(9.0,) * 8),
    ]
    summary = benchmark.summarize_results(
        results,
        providers=("quack", "cudnn"),
        replicates=2,
        helion_selection="compiler_heuristic",
    )
    assert summary["providers"] == ["quack", "cudnn"]
    quack = summary["provider_results"]["quack"]
    cudnn = summary["provider_results"]["cudnn"]
    assert quack["cross_replicate_geomean"] == pytest.approx(
        math.prod((0.25, *(1.0,) * 8, *(4.0,) * 7)) ** (1 / 16)
    )
    assert quack["worst_row"]["row_index"] == 0
    assert quack["worst_row"]["geomean_speedup"] == pytest.approx(0.5)
    assert quack["worst_row"]["replicate_speedups"] == [0.25, 1.0]
    assert quack["worst_row"]["helion_ms"] == {
        "median_across_replicates": 1.0,
        "replicate_medians": [1.0, 1.0],
    }
    assert quack["worst_row"]["provider_ms"] == {
        "median_across_replicates": 0.625,
        "replicate_medians": [0.25, 1.0],
    }
    assert quack["row_wins"] == 7
    assert cudnn["cross_replicate_geomean"] == pytest.approx(9.0)
    assert cudnn["worst_row"]["geomean_speedup"] == pytest.approx(9.0)
    assert all(item["invariant"] for item in quack["config_distributions"])
    assert all(item["invariant"] for item in cudnn["config_distributions"])
    assert all(item["invariant"] for item in summary["helion_config_distributions"])
    assert summary["publication_eligible"] is False
    assert summary["minimum_publication_replicates"] == 3
    assert summary["protocol"] == {
        "fresh_process_and_caches_per_provider_replicate": True,
        "rows_per_replicate": 8,
        "thermal_warmup_ms": 10_000,
        "paired_cold_l2_samples": 102,
        "balanced_rotated_reversed_order": True,
        "row_timing_statistic": "median_ms",
        "raw_paired_samples_retained": False,
        "raw_paired_samples_note": (
            "the shared timer returns one median per implementation"
        ),
        "layout_policy": benchmark.BENCHMARK_LAYOUT_POLICY,
    }


@pytest.mark.parametrize(
    "fixed_selection", ("compiler_heuristic", "final_reviewed_aot")
)
def test_fixed_summary_requires_same_config_across_providers(
    fixed_selection: str,
) -> None:
    results = [
        _result(
            "quack",
            0,
            config_prefix="first",
            helion_selection="live_autotune",
        ),
        _result(
            "cudnn",
            0,
            config_prefix="second",
            helion_selection="live_autotune",
        ),
    ]
    fixed_results = [
        {**result, "helion_selection": fixed_selection} for result in results
    ]
    with pytest.raises(RuntimeError, match="fixed Helion config changed"):
        benchmark.summarize_results(
            fixed_results,
            providers=("quack", "cudnn"),
            replicates=1,
            helion_selection=fixed_selection,
        )
    summary = benchmark.summarize_results(
        results,
        providers=("quack", "cudnn"),
        replicates=1,
        helion_selection="live_autotune",
    )
    distribution = summary["helion_config_distributions"][0]
    assert distribution["invariant"] is False
    assert distribution["config_sha256_counts"] == {"first-0": 1, "second-0": 1}
    assert summary["publication_eligible"] is False


def test_summary_rejects_provider_config_drift() -> None:
    results = [
        _result("quack", 0),
        _result("quack", 1, provider_config_prefix="changed"),
    ]

    with pytest.raises(RuntimeError, match="quack selected config changed"):
        benchmark.summarize_results(
            results,
            providers=("quack",),
            replicates=2,
            helion_selection="compiler_heuristic",
        )


def test_summary_publication_boundary_requires_three_fixed_replicates() -> None:
    summary = benchmark.summarize_results(
        [_result("quack", replicate) for replicate in range(3)],
        providers=("quack",),
        replicates=3,
        helion_selection="compiler_heuristic",
    )

    assert summary["publication_eligible"] is True


def test_live_autotune_is_not_publishable_with_three_replicates() -> None:
    summary = benchmark.summarize_results(
        [
            _result(
                "quack",
                replicate,
                config_prefix=f"search-{replicate}",
                helion_selection="live_autotune",
            )
            for replicate in range(3)
        ],
        providers=("quack",),
        replicates=3,
        helion_selection="live_autotune",
    )

    assert summary["publication_eligible"] is False


def test_summary_compares_semantic_stack_values_not_install_paths() -> None:
    summary = benchmark.summarize_results(
        [
            _result("quack", 0, versions=_versions(cuda_home="/worker-a/cuda")),
            _result("cudnn", 0, versions=_versions(cuda_home="/worker-b/cuda")),
        ],
        providers=("quack", "cudnn"),
        replicates=1,
        helion_selection="compiler_heuristic",
    )

    assert summary["software_stack"]["cuda_driver"] == "590.00"
    assert "cuda_home" not in summary["software_stack"]


def test_summary_rejects_semantic_stack_drift_across_providers() -> None:
    results = [
        _result("quack", 0),
        _result("cudnn", 0, versions=_versions(cuda_driver="591.00")),
    ]

    with pytest.raises(RuntimeError, match="semantic software stack changed"):
        benchmark.summarize_results(
            results,
            providers=("quack", "cudnn"),
            replicates=1,
            helion_selection="compiler_heuristic",
        )


def test_campaign_runs_sequential_provider_replicates_and_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_cuda_runtime(monkeypatch, tmp_path)
    args = _args(tmp_path)
    calls: list[tuple[str, int]] = []

    def run_worker(
        command: list[str],
        *,
        environment: dict[str, str],
        log_path: Path,
        telemetry_path: Path,
        target_gpu_uuid: str,
    ) -> tuple[int, int]:
        provider = command[command.index("--provider") + 1]
        replicate = int(command[command.index("--replicate") + 1])
        run_dir = Path(command[command.index("--run-dir") + 1])
        calls.append((provider, replicate))
        assert environment["CUDA_VISIBLE_DEVICES"] == target_gpu_uuid
        log_path.write_text("")
        telemetry_path.write_text("timestamp,uuid\n")
        (run_dir / "result.json").write_text(
            json.dumps(
                _result(
                    provider,
                    replicate,
                    helion_selection=args.helion_selection,
                )
            )
        )
        return 0, 2

    monkeypatch.setattr(benchmark, "_run_monitored_worker", run_worker)
    monkeypatch.setattr(benchmark, "_resolve_target_gpu", lambda _selector: "GPU-test")
    monkeypatch.setattr(
        benchmark,
        "_summarize_telemetry",
        lambda _path, _uuid: {
            "sample_count": 2,
            "power_limit_watts": 1000.0,
            "active_clock_event_reason_sample_count": 2,
            "active_clock_event_reasons": {"0x1": 1, "0x4": 1},
            "disallowed_clock_event_reason_sample_count": 0,
            "disallowed_clock_event_reasons": {},
        },
    )
    monkeypatch.setattr(benchmark, "_source_identity", lambda: SOURCE)
    assert benchmark._run_campaign(args) == 0
    assert calls == [
        ("quack", 0),
        ("cudnn", 0),
        ("quack", 1),
        ("cudnn", 1),
    ]
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["schema"] == benchmark.SUMMARY_SCHEMA
    assert len(summary["runs"]) == 4
    assert summary["monitoring"]["power_limit_watts"] == 1000.0
    assert summary["monitoring"]["active_clock_event_reasons"] == {
        "0x1": 4,
        "0x4": 4,
    }


def test_campaign_rejects_disallowed_clock_event_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_cuda_runtime(monkeypatch, tmp_path)
    args = _args(tmp_path, providers=("quack",), replicates=1)

    def run_worker(
        command: list[str],
        **_kwargs: object,
    ) -> tuple[int, int]:
        run_dir = Path(command[command.index("--run-dir") + 1])
        (run_dir / "result.json").write_text(json.dumps(_result("quack", 0)))
        return 0, 1

    monkeypatch.setattr(benchmark, "_run_monitored_worker", run_worker)
    monkeypatch.setattr(benchmark, "_resolve_target_gpu", lambda _selector: "GPU-test")
    monkeypatch.setattr(
        benchmark,
        "_summarize_telemetry",
        lambda _path, _uuid: {
            "sample_count": 1,
            "power_limit_watts": 1000.0,
            "active_clock_event_reason_sample_count": 1,
            "active_clock_event_reasons": {"0x8": 1},
            "disallowed_clock_event_reason_sample_count": 1,
            "disallowed_clock_event_reasons": {"0x8": 1},
        },
    )
    monkeypatch.setattr(benchmark, "_source_identity", lambda: SOURCE)

    with pytest.raises(RuntimeError, match="disallowed clock-event reasons"):
        benchmark._run_campaign(args)


def test_campaign_rejects_power_limit_drift_across_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_cuda_runtime(monkeypatch, tmp_path)
    args = _args(tmp_path, providers=("quack",), replicates=2)

    def run_worker(command: list[str], **_kwargs: object) -> tuple[int, int]:
        replicate = int(command[command.index("--replicate") + 1])
        run_dir = Path(command[command.index("--run-dir") + 1])
        (run_dir / "result.json").write_text(json.dumps(_result("quack", replicate)))
        return 0, 1

    power_limits = iter((1000.0, 900.0))
    monkeypatch.setattr(benchmark, "_run_monitored_worker", run_worker)
    monkeypatch.setattr(benchmark, "_resolve_target_gpu", lambda _selector: "GPU-test")
    monkeypatch.setattr(
        benchmark,
        "_summarize_telemetry",
        lambda _path, _uuid: {
            "sample_count": 1,
            "power_limit_watts": next(power_limits),
            "active_clock_event_reason_sample_count": 0,
            "active_clock_event_reasons": {},
            "disallowed_clock_event_reason_sample_count": 0,
            "disallowed_clock_event_reasons": {},
        },
    )
    monkeypatch.setattr(benchmark, "_source_identity", lambda: SOURCE)

    with pytest.raises(RuntimeError, match="power limit changed across"):
        benchmark._run_campaign(args)
