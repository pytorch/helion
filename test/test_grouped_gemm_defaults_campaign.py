from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

from benchmarks.cute import compare_grouped_gemm_defaults as benchmark
import pytest

SOURCE = {"commit": "source", "tree": "tree"}


def _row(row_index: int, speedup: float, config_hash: str) -> dict[str, object]:
    return {
        "row_index": row_index,
        "configs": {"helion": {"config_sha256": config_hash}},
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
    helion_selection: str = "compiler_heuristic",
) -> dict[str, object]:
    speedups = row_speedups or tuple(speedup + index / 100 for index in range(8))
    assert len(speedups) == 8
    return {
        "schema": benchmark.RESULT_SCHEMA,
        "provider": provider,
        "replicate": replicate,
        "helion_selection": helion_selection,
        "device": {"uuid": "GPU-test", "visible": "GPU-test"},
        "source": SOURCE,
        "rows": [
            _row(row_index, row_speedup, f"{config_prefix}-{row_index}")
            for row_index, row_speedup in enumerate(speedups)
        ],
    }


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "providers": ("quack", "cudnn"),
        "replicates": 2,
        "helion_selection": "final_reviewed_aot",
        "output_dir": tmp_path / "campaign",
        "cuda_visible_devices": "GPU-test",
        "deepgemm_root": None,
        "cutlass_root": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _mock_cuda_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cuda_home = tmp_path / "cuda-runtime"
    monkeypatch.setattr(
        benchmark,
        "_installed_cuda_runtime",
        lambda: (cuda_home, cuda_home / "lib" / "libcudart.so.13"),
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


def test_public_cli_requires_external_output_directory() -> None:
    parser = benchmark.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(("--cuda-visible-devices", "GPU-test"))
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
    assert args.helion_selection == "final_reviewed_aot"
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
    deepgemm.mkdir()
    cutlass.mkdir()
    args = _args(
        tmp_path,
        providers=("deepgemm", "cutlass"),
        deepgemm_root=deepgemm,
        cutlass_root=cutlass,
    )
    assert benchmark._provider_roots(args) == {
        "deepgemm": deepgemm,
        "cutlass": cutlass,
    }
    args.deepgemm_root = None
    with pytest.raises(ValueError, match="deepgemm-root"):
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
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_HOME": "/stale",
        "DG_JIT_CACHE_DIR": "/stale",
        "HELION_AOT_MODE": "stale",
        "HELION_AUTOTUNE_BEST_OF_K": "99",
        "HELION_BACKEND": "triton",
        "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "7",
        "HELION_CUTE_MMA_IMPL": "stale",
        "HELION_HEURISTIC_DIR": "/stale",
        "HELION_SKIP_CACHE": "stale",
        "LD_PRELOAD": "/stale",
        "NVIDIA_TF32_OVERRIDE": "1",
        "PYTHONHOME": "/stale",
        "PYTHONPATH": "/stale",
        "QUACK_CACHE_DIR": "/stale",
        "TRITON_CACHE_DIR": "/stale",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
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
        "CUBLAS_WORKSPACE_CONFIG",
        "HELION_AUTOTUNE_BEST_OF_K",
        "HELION_HEURISTIC_DIR",
        "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    ):
        assert name not in environment
    assert environment["HELION_CAP_AUTOTUNE_NUM_NEIGHBORS"] == "-1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-test"
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
    command = benchmark._worker_command(args, "quack", 2, run_dir, {})
    other_command = benchmark._worker_command(args, "quack", 2, tmp_path / "other", {})
    assert command[:4] == [
        benchmark.sys.executable,
        "-u",
        "-X",
        f"pycache_prefix={run_dir / 'cache' / 'pycache'}",
    ]
    assert "--worker" in command
    assert command[command.index("--provider") + 1] == "quack"
    assert command[command.index("--replicate") + 1] == "2"
    assert Path(command[command.index("--output-dir") + 1]) == args.output_dir
    assert command[command.index("--helion-selection") + 1] == "compiler_heuristic"
    assert other_command[3] != command[3]


def test_monitored_worker_records_periodic_and_final_samples_then_cleans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    polls = iter((None, 0))
    events: list[object] = []
    process = SimpleNamespace(
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


def test_terminate_process_escalates_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, signal.Signals]] = []
    waits: list[bool] = []
    process = SimpleNamespace(pid=1234, wait=lambda: waits.append(True))
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


def test_provider_order_reverses_pairs_and_rotates_later_pairs() -> None:
    providers = ("a", "b", "c")
    assert [
        benchmark._provider_order(providers, replicate) for replicate in range(6)
    ] == [
        ("a", "b", "c"),
        ("b", "c", "a"),
        ("c", "a", "b"),
        ("c", "b", "a"),
        ("a", "c", "b"),
        ("b", "a", "c"),
    ]


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
    assert quack["worst_row"] == {
        "row_index": 0,
        "geomean_speedup": 0.5,
        "replicate_speedups": [0.25, 1.0],
    }
    assert quack["row_wins"] == 7
    assert cudnn["cross_replicate_geomean"] == pytest.approx(9.0)
    assert cudnn["worst_row"]["geomean_speedup"] == pytest.approx(9.0)
    assert all(item["invariant"] for item in summary["helion_config_distributions"])
    assert summary["protocol"] == {
        "fresh_process_and_caches_per_provider_replicate": True,
        "provider_worker_order": "rotated_then_reversed",
        "rows_per_replicate": 8,
        "thermal_warmup_ms": 10_000,
        "paired_cold_l2_samples": 102,
        "balanced_rotated_reversed_order": True,
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
        return 0, 1

    monkeypatch.setattr(benchmark, "_run_monitored_worker", run_worker)
    monkeypatch.setattr(benchmark, "_resolve_target_gpu", lambda _selector: "GPU-test")
    monkeypatch.setattr(
        benchmark,
        "_summarize_telemetry",
        lambda _path, _uuid: {
            "sample_count": 1,
            "active_clock_event_reason_sample_count": 0,
            "active_clock_event_reasons": {},
        },
    )
    monkeypatch.setattr(benchmark, "_source_identity", lambda: SOURCE)
    assert benchmark._run_campaign(args) == 0
    assert calls == [
        ("quack", 0),
        ("cudnn", 0),
        ("cudnn", 1),
        ("quack", 1),
    ]
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["schema"] == benchmark.SUMMARY_SCHEMA
    assert len(summary["runs"]) == 4
