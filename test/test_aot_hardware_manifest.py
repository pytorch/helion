from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from helion import exc
from helion._hardware import HardwareInfo
import helion.autotuner.aot_cache as aot_cache_module
from helion.autotuner.aot_cache import AOTAutotuneCache
from helion.autotuner.aot_cache import load_hardware_manifest
from helion.autotuner.aot_kernel import HeuristicKeyFunction
import helion.autotuner.aot_runner as aot_runner_module
from helion.autotuner.aot_runner import RunConfig
from helion.runtime.config import Config

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    import os
    from types import FunctionType

    from helion.autotuner.base_search import BaseSearch

CUDA = HardwareInfo("cuda", "NVIDIA B200", "13.0", "sm100")
CUDA_OTHER = HardwareInfo("cuda", "NVIDIA GB200", "13.0", "sm100")
TPU = HardwareInfo("tpu", "TPU v5p", "0.8.0", "tpu-v5p")


@pytest.fixture(autouse=True)
def _clear_aot_test_caches() -> Iterator[None]:
    AOTAutotuneCache.clear_caches()
    HeuristicKeyFunction.clear_cache()
    yield
    AOTAutotuneCache.clear_caches()
    HeuristicKeyFunction.clear_cache()


def _run_config(output_dir: Path, hardware: HardwareInfo | None = None) -> RunConfig:
    return RunConfig(
        benchmark_cmd=["python", "benchmark.py"],
        output_dir=output_dir,
        hardware=hardware,
        run_id="run",
        print_score_matrix=False,
    )


def _prepared_run(
    output_dir: Path,
    hardware: HardwareInfo | None = CUDA,
    *,
    manifest: bool = True,
    logs: bool = True,
) -> RunConfig:
    config = _run_config(output_dir, hardware)
    config.run_dir.mkdir()
    if logs:
        config.run_log_dir.mkdir()
    if manifest:
        assert hardware is not None
        aot_cache_module.write_hardware_manifest(config.run_dir, hardware)
    return config


def _mock_cache_environment(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
    *,
    mode: str,
    hardware: HardwareInfo = CUDA,
) -> None:
    monkeypatch.setenv("HELION_AOT_MODE", mode)
    monkeypatch.setattr(aot_cache_module, "get_aot_data_dir", lambda: data_dir)
    monkeypatch.setattr(
        aot_cache_module,
        "get_hardware_info",
        lambda device=None: hardware,
    )


def _symlink_loop(directory: Path) -> Path:
    first = directory / "first.py"
    second = directory / "second.py"
    first.symlink_to(second)
    second.symlink_to(first)
    return first


def _write_tuned_configs(
    config: RunConfig,
    data: dict[str, list[dict[str, object]]],
) -> Path:
    assert config.hardware is not None
    path = config.run_dir / f"tuned_configs_{config.hardware.hardware_id}.json"
    path.write_text(json.dumps(data))
    return path


def _successful_benchmark(*args: object, **kwargs: object) -> tuple[int, str, str]:
    return 0, "", ""


def _assert_phase_requires_refresh(
    monkeypatch: pytest.MonkeyPatch,
    config: RunConfig,
    run_phase: Callable[[RunConfig], bool],
    refresh: Callable[[], None],
) -> None:
    monkeypatch.setattr(aot_runner_module, "run_benchmark", _successful_benchmark)
    assert not run_phase(config)

    def refreshing_benchmark(*args: object, **kwargs: object) -> tuple[int, str, str]:
        refresh()
        return 0, "", ""

    monkeypatch.setattr(aot_runner_module, "run_benchmark", refreshing_benchmark)
    assert run_phase(config)


def _pending_logs(tmp_path: Path) -> tuple[Path, Path]:
    pending = tmp_path / "runner_pending.log"
    pending.write_text("pending\n")
    return pending, tmp_path / "runner_cuda.log"


def _autotuner(device: torch.device) -> SimpleNamespace:
    def demo(value: int) -> int:
        return value

    kernel_api = SimpleNamespace(
        __code__=demo.__code__,
        name="demo",
        normalize_args=lambda *args: args,
        specialization_key=lambda args: tuple(args),
        _aot_collect_fn=None,
        _aot_measure_fn=None,
        _aot_user_key=None,
        _aot_workflow_done=False,
    )
    bound_kernel = SimpleNamespace(
        kernel=kernel_api,
        env=SimpleNamespace(device=device),
        is_cacheable=lambda: True,
    )
    return SimpleNamespace(
        kernel=bound_kernel,
        args=(1,),
        config_spec=SimpleNamespace(
            default_config=lambda: Config(block_sizes=[16]),
        ),
    )


def test_explicit_hardware_device_rejects_pseudo_tpu_type() -> None:
    pseudo_tpu = cast("torch.device", SimpleNamespace(type="tpu"))

    assert aot_cache_module._explicit_hardware_device(pseudo_tpu) is None
    assert aot_cache_module._explicit_hardware_device(torch.device("cpu")) == (
        torch.device("cpu")
    )


def test_collect_cache_records_explicit_hardware_and_rejects_mixing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_devices: list[torch.device | None] = []

    def hardware_for(device: torch.device | None = None) -> HardwareInfo:
        observed_devices.append(device)
        return TPU if device == torch.device("cpu") else CUDA

    monkeypatch.setenv("HELION_AOT_MODE", "collect")
    monkeypatch.setattr(aot_cache_module, "get_aot_data_dir", lambda: tmp_path)
    monkeypatch.setattr(aot_cache_module, "get_hardware_info", hardware_for)
    assert AOTAutotuneCache(_autotuner(torch.device("cpu"))).hardware == TPU
    assert load_hardware_manifest(tmp_path) == TPU
    with pytest.raises(exc.AOTHardwareManifestError, match="Ambiguous AOT hardware"):
        AOTAutotuneCache(_autotuner(torch.device("cuda:0")))

    assert observed_devices == [torch.device("cpu"), torch.device("cuda:0")]


@pytest.mark.parametrize("source_kind", ("loop", "non_file"))
def test_heuristic_key_and_cache_share_non_file_source_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
) -> None:
    if source_kind == "loop":
        source_file = str(_symlink_loop(tmp_path))
    else:
        source_file = "<stdin>"

    override_dir = tmp_path / "override"
    override_dir.mkdir()
    source_specific = (
        override_dir
        / f"_helion_aot_{Path(source_file).stem}_{CUDA.device_kind}_{CUDA.compute_capability}.py"
    )
    source_specific.write_text(
        "def key_demo(value):\n    return 999\n"
        "def autotune_demo(value):\n    return {'block_sizes': [999]}\n"
    )
    aot_cache_module.write_hardware_manifest(tmp_path, CUDA)
    generic_heuristic = tmp_path / "heuristic_demo.py"
    generic_heuristic.write_text(
        "def key_demo(value):\n    return 17\n"
        "def autotune_demo(value):\n    return {'block_sizes': [17]}\n"
    )

    namespace: dict[str, object] = {}
    exec(
        compile("def demo(value):\n    return value\n", source_file, "exec"), namespace
    )
    kernel_function = cast("FunctionType", namespace["demo"])
    autotuner = _autotuner(torch.device("cuda:0"))
    autotuner.kernel.kernel.__code__ = kernel_function.__code__

    _mock_cache_environment(monkeypatch, tmp_path, mode="evaluate")
    monkeypatch.setenv("HELION_HEURISTIC_DIR", str(override_dir))
    key_fn = HeuristicKeyFunction(
        source_file,
        "demo",
        code_object=kernel_function.__code__,
    )
    cache = AOTAutotuneCache(cast("BaseSearch", autotuner))

    assert key_fn.kernel_source_file is None
    assert cache._kernel_source_file is None
    assert key_fn._kernel_source_identity == cache._kernel_source
    key = key_fn(7)
    assert isinstance(key, tuple)
    assert key[-1] == 17
    assert cache._find_heuristic_file() == generic_heuristic.resolve()
    if source_kind == "loop":
        other_namespace: dict[str, object] = {}
        exec(
            compile("def demo(value):\n    return value + 1\n", source_file, "exec"),
            other_namespace,
        )
        other_autotuner = _autotuner(torch.device("cuda:0"))
        other_autotuner.kernel.kernel.__code__ = cast(
            "FunctionType", other_namespace["demo"]
        ).__code__
        other_cache = AOTAutotuneCache(cast("BaseSearch", other_autotuner))
        assert other_cache._kernel_source_file is None
        assert cache._kernel_source != other_cache._kernel_source


def test_manifest_is_idempotent_and_self_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        aot_cache_module.os,
        "link",
        lambda *_args: pytest.fail("manifest publication must not use hard links"),
    )
    aot_cache_module.write_hardware_manifest(tmp_path, TPU)
    aot_cache_module.write_hardware_manifest(tmp_path, TPU)
    assert load_hardware_manifest(tmp_path) == TPU

    data = json.loads((tmp_path / "hardware.json").read_text())
    data["future_optional_field"] = "accepted"
    (tmp_path / "hardware.json").write_text(json.dumps(data))
    assert load_hardware_manifest(tmp_path) == TPU

    data["hardware_id"] = CUDA.hardware_id
    (tmp_path / "hardware.json").write_text(json.dumps(data))
    with pytest.raises(exc.AOTHardwareManifestError, match="Inconsistent AOT hardware"):
        load_hardware_manifest(tmp_path)


def test_identical_manifest_write_skips_temp_file_and_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aot_cache_module.write_hardware_manifest(tmp_path, CUDA)

    def unexpected_temporary_file(*args: object, **kwargs: object) -> object:
        raise AssertionError("identical manifest must not create a temporary file")

    def unexpected_fsync(file_descriptor: int) -> None:
        raise AssertionError("identical manifest must not fsync")

    monkeypatch.setattr(
        aot_cache_module.tempfile,
        "NamedTemporaryFile",
        unexpected_temporary_file,
    )
    monkeypatch.setattr(aot_cache_module.os, "fsync", unexpected_fsync)

    aot_cache_module.write_hardware_manifest(tmp_path, CUDA)
    assert load_hardware_manifest(tmp_path) == CUDA


def test_manifest_atomic_replace_failure_has_actionable_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_file = tmp_path / "hardware.json"
    original_replace = aot_cache_module.os.replace

    def fail_manifest_replace(
        source: os.PathLike[str], destination: os.PathLike[str]
    ) -> None:
        raise OSError("atomic replacement unavailable")

    monkeypatch.setattr(aot_cache_module.os, "replace", fail_manifest_replace)

    with pytest.raises(exc.AOTHardwareManifestError) as error:
        aot_cache_module.write_hardware_manifest(tmp_path, CUDA)

    message = str(error.value)
    for token in (
        "atomically create AOT hardware manifest",
        "HELION_AOT_DATA_DIR",
        "fresh, writable directory",
        "recollect",
    ):
        assert token in message
    assert not manifest_file.exists()
    assert not list(tmp_path.glob(".hardware.json.*.tmp"))

    monkeypatch.setattr(aot_cache_module.os, "replace", original_replace)
    aot_cache_module.write_hardware_manifest(tmp_path, CUDA)
    assert load_hardware_manifest(tmp_path) == CUDA


def test_manifest_reader_never_observes_partial_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_file = tmp_path / "hardware.json"
    replace_entered = threading.Event()
    release_replace = threading.Event()
    original_replace = aot_cache_module.os.replace

    def blocking_replace(
        source: os.PathLike[str], destination: os.PathLike[str]
    ) -> None:
        assert Path(destination) == manifest_file
        json.loads(Path(source).read_text())
        replace_entered.set()
        assert release_replace.wait(5)
        original_replace(source, destination)

    monkeypatch.setattr(aot_cache_module.os, "replace", blocking_replace)
    with ThreadPoolExecutor(max_workers=1) as pool:
        publication = pool.submit(
            aot_cache_module.write_hardware_manifest,
            tmp_path,
            CUDA,
        )
        try:
            assert replace_entered.wait(5)
            assert not manifest_file.exists()
            with pytest.raises(
                exc.AOTHardwareManifestError, match="missing required hardware.json"
            ):
                load_hardware_manifest(tmp_path)
        finally:
            release_replace.set()
        publication.result(timeout=5)

    assert load_hardware_manifest(tmp_path) == CUDA


def test_manifest_concurrent_writers_publish_exactly_one_identity(
    tmp_path: Path,
) -> None:
    start = threading.Barrier(2)

    def publish(hardware: HardwareInfo) -> tuple[HardwareInfo, Exception | None]:
        start.wait(timeout=5)
        try:
            aot_cache_module.write_hardware_manifest(tmp_path, hardware)
        except exc.AOTHardwareManifestError as error:
            return hardware, error
        return hardware, None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, (CUDA, TPU)))

    successes = [hardware for hardware, error in results if error is None]
    failures = [error for _, error in results if error is not None]
    assert len(successes) == 1
    assert len(failures) == 1
    assert "Ambiguous AOT hardware identity" in str(failures[0])
    assert load_hardware_manifest(tmp_path) == successes[0]


def test_stale_crashed_writer_temp_does_not_poison_manifest(tmp_path: Path) -> None:
    (tmp_path / ".hardware.json.lock").touch()
    stale_temp = tmp_path / ".hardware.json.crashed.tmp"
    stale_temp.write_text('{"hardware_id":')

    aot_cache_module.write_hardware_manifest(tmp_path, CUDA)

    assert load_hardware_manifest(tmp_path) == CUDA
    assert stale_temp.read_text() == '{"hardware_id":'


@pytest.mark.parametrize(
    "artifact_name",
    (
        "heuristic_demo.py",
        "_helion_aot_kernel_cuda_sm100.py",
        "tuned_configs_cuda_legacy.json",
        "measurements_cuda_legacy.csv",
        "heuristic_summary_cuda_legacy.json",
        "evaluation_cuda_legacy.json",
        "demo_standalone.py",
        "__pycache__/heuristic_demo.cpython-310.pyc",
        "__pycache__/demo_standalone.cpython-310.pyc",
    ),
)
def test_collect_rejects_pre_manifest_aot_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    artifact = tmp_path / artifact_name
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("legacy\n")
    _mock_cache_environment(monkeypatch, tmp_path, mode="collect")

    with pytest.raises(exc.AOTHardwareManifestError) as error:
        AOTAutotuneCache(_autotuner(torch.device("cuda:0")))

    message = str(error.value)
    assert "fresh AOT data directory" in message
    assert "Conflicting legacy artifacts" in message
    assert artifact_name in message
    assert not (tmp_path / "hardware.json").exists()


@pytest.mark.parametrize("path", ("collect", "evaluate_key", "evaluate_config"))
def test_legacy_generic_heuristic_requires_fresh_recollect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    executed = tmp_path / "executed"
    (tmp_path / "kernel.py").write_text("def demo():\n    pass\n")
    (tmp_path / "heuristic_demo.py").write_text(
        f"from pathlib import Path\nPath({str(executed)!r}).write_text('executed')\n"
    )
    _mock_cache_environment(
        monkeypatch,
        tmp_path,
        mode="collect" if path == "collect" else "evaluate",
    )
    with pytest.raises(exc.AOTHardwareManifestError) as error:
        if path == "collect":
            AOTAutotuneCache(_autotuner(torch.device("cuda:0")))
        elif path == "evaluate_key":
            HeuristicKeyFunction(str(tmp_path / "kernel.py"), "demo")(1)
        else:
            AOTAutotuneCache(_autotuner(torch.device("cuda:0"))).get()
    message = str(error.value)
    for token in (
        "missing required hardware.json",
        "pre-manifest AOT run",
        "fresh AOT data directory",
        "HELION_AOT_DATA_DIR",
        "new aot_runner --run-id",
        "intentionally not adopted in place",
    ):
        assert token in message
    assert error.value.report().startswith("ERROR[AOTHardwareManifestError]")
    assert not executed.exists()


@pytest.mark.parametrize(
    ("mode", "heuristic_source", "expected_block_size"),
    (
        (
            "disabled",
            "raise AssertionError('disabled mode loaded an AOT artifact')\n",
            16,
        ),
        (
            "evaluate",
            "def autotune_demo(*args):\n    return {'block_sizes': [32]}\n",
            32,
        ),
    ),
)
def test_aot_artifacts_are_scoped_to_the_selected_heuristic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    heuristic_source: str,
    expected_block_size: int,
) -> None:
    aot_cache_module.write_hardware_manifest(tmp_path, TPU)
    heuristic_dir = tmp_path / "override"
    heuristic_dir.mkdir()
    (heuristic_dir / "heuristic_demo.py").write_text(heuristic_source)
    _mock_cache_environment(monkeypatch, tmp_path, mode=mode)
    monkeypatch.setenv("HELION_HEURISTIC_DIR", str(heuristic_dir))

    cache = AOTAutotuneCache(_autotuner(torch.device("cuda:0")))
    assert cache.get() == Config(block_sizes=[expected_block_size])


@pytest.mark.parametrize(
    ("recorded_hardware", "error"),
    (
        (None, "fresh AOT data directory"),
        (TPU, "AOT hardware identity mismatch"),
        (CUDA, None),
    ),
)
def test_generic_heuristic_requires_matching_manifest_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_hardware: HardwareInfo | None,
    error: str | None,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = source_dir / "kernel.py"
    source_file.write_text("def demo():\n    pass\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    executed = tmp_path / "executed"
    if recorded_hardware is not None:
        aot_cache_module.write_hardware_manifest(data_dir, recorded_hardware)
    (data_dir / "heuristic_demo.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(executed)!r}).write_text('executed')\n"
        "def key_demo(*args):\n"
        "    return 7\n"
    )

    _mock_cache_environment(monkeypatch, data_dir, mode="evaluate")
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(data_dir))
    key_fn = HeuristicKeyFunction(str(source_file), "demo")
    if error is None:
        result = key_fn(1)
        assert result == (
            "helion_aot_heuristic",
            str((data_dir / "heuristic_demo.py").resolve()),
            7,
        )
        assert executed.is_file()
    else:
        with pytest.raises(exc.AOTHardwareManifestError, match=error):
            key_fn(1)
        assert not executed.exists()


@pytest.mark.parametrize(
    ("recorded_hardware", "error"),
    (
        (None, "missing required hardware.json"),
        (TPU, "AOT hardware identity mismatch"),
        (CUDA, None),
    ),
)
def test_direct_measure_requires_matching_run_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_hardware: HardwareInfo | None,
    error: str | None,
) -> None:
    if recorded_hardware is not None:
        aot_cache_module.write_hardware_manifest(tmp_path, recorded_hardware)
    (tmp_path / f"tuned_configs_{CUDA.hardware_id}.json").write_text(
        json.dumps({"demo": []})
    )
    _mock_cache_environment(monkeypatch, tmp_path, mode="measure")

    if error is not None:
        with pytest.raises(exc.AOTHardwareManifestError, match=error):
            AOTAutotuneCache(cast("BaseSearch", _autotuner(torch.device("cuda:0"))))
    else:
        cache = AOTAutotuneCache(cast("BaseSearch", _autotuner(torch.device("cuda:0"))))
        assert cache.hardware == CUDA


def test_tuned_config_roundtrip_preserves_standalone_policy(tmp_path: Path) -> None:
    cache = object.__new__(AOTAutotuneCache)
    cache.data_dir = tmp_path
    cache.hardware_id = CUDA.hardware_id
    cache._tuned_configs = {
        "demo": [
            aot_cache_module.TunedConfig(
                config=Config(block_sizes=[16]),
                shape_key=aot_cache_module.ShapeKey(
                    kernel_name="demo",
                    specialization_key=(),
                    hardware_id=CUDA.hardware_id,
                ),
                standalone=False,
            )
        ]
    }

    cache._save_tuned_configs()

    data = json.loads(cache._configs_file.read_text())
    assert data["demo"][0]["standalone"] is False
    assert cache._load_tuned_configs()["demo"][0].standalone is False

    del data["demo"][0]["standalone"]
    cache._configs_file.write_text(json.dumps(data))
    assert cache._load_tuned_configs()["demo"][0].standalone is True


def test_collect_runner_adopts_subprocess_hardware_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _prepared_run(tmp_path, None, manifest=False)
    (config.run_log_dir / "runner_pending.log").write_text("runner log\n")
    observed_logs: list[Path] = []

    def run_benchmark(
        cmd: list[str],
        env: dict[str, str],
        log_file: Path,
        phase: str,
        kernels: list[str] | None = None,
    ) -> tuple[int, str, str]:
        observed_logs.append(log_file)
        log_file.write_text("collect log\n")
        aot_cache_module.write_hardware_manifest(config.run_dir, TPU)
        (config.run_dir / f"tuned_configs_{TPU.hardware_id}.json").write_text(
            json.dumps({"demo": [{}]})
        )
        return 0, "", ""

    monkeypatch.setattr(aot_runner_module, "run_benchmark", run_benchmark)

    assert aot_runner_module.run_collect_phase(config)
    assert config.hardware == TPU
    assert [path.name for path in observed_logs] == ["collect_pending.log"]
    assert (config.run_log_dir / f"collect_{TPU.hardware_id}.log").is_file()
    assert (config.run_log_dir / f"runner_{TPU.hardware_id}.log").is_file()
    assert not list(config.run_log_dir.glob("*pending*"))


def test_collect_runner_reports_child_failure_before_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _prepared_run(tmp_path, None, manifest=False)
    monkeypatch.setattr(
        aot_runner_module,
        "run_benchmark",
        lambda *args, **kwargs: (23, "", ""),
    )

    assert not aot_runner_module.run_collect_phase(config)
    assert "Collect phase failed with return code 23" in caplog.text
    assert "missing required hardware.json" not in caplog.text


def test_resume_requires_and_uses_recorded_hardware(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    aot_cache_module.write_hardware_manifest(run_dir, TPU)
    assert aot_runner_module.resolve_run_hardware(tmp_path, "run", "measure") == TPU

    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    assert (
        aot_runner_module.resolve_run_hardware(tmp_path, "missing", "collect") is None
    )
    with pytest.raises(exc.AOTHardwareManifestError, match="Cannot resume AOT run"):
        aot_runner_module.resolve_run_hardware(tmp_path, "missing", "measure")

    (missing_dir / "tuned_configs_unknown.json").write_text("{}")
    with pytest.raises(exc.AOTHardwareManifestError, match="Cannot resume AOT run"):
        aot_runner_module.resolve_run_hardware(tmp_path, "missing", "collect")


@pytest.mark.parametrize("phase", ("collect", "measure", "evaluate", "compile"))
def test_device_phases_reject_configured_manifest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    config = _prepared_run(tmp_path, TPU, manifest=False, logs=False)
    aot_cache_module.write_hardware_manifest(config.run_dir, CUDA)

    def unexpected_benchmark(*args: object, **kwargs: object) -> tuple[int, str, str]:
        raise AssertionError("manifest mismatch must fail before launching the child")

    monkeypatch.setattr(aot_runner_module, "run_benchmark", unexpected_benchmark)

    run_phase = getattr(aot_runner_module, f"run_{phase}_phase")
    assert not run_phase(config)


@pytest.mark.parametrize("mode", ("collect", "measure", "evaluate", "compile"))
def test_benchmark_env_requests_child_hardware_validation(
    tmp_path: Path,
    mode: str,
) -> None:
    config = _run_config(tmp_path, CUDA)

    env = aot_runner_module._benchmark_env(config, mode)

    assert env == {
        "HELION_AOT_MODE": mode,
        "HELION_AOT_DATA_DIR": str(config.run_dir),
        "HELION_AUTOTUNE_CACHE": "AOTAutotuneCache",
        aot_cache_module.AOT_RUNNER_HARDWARE_VALIDATION_ENV: "1",
    }

    config.hardware = None
    assert aot_cache_module.AOT_RUNNER_HARDWARE_VALIDATION_ENV not in (
        aot_runner_module._benchmark_env(config, mode)
    )


def test_resumed_collect_launches_child_selected_nondefault_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _prepared_run(tmp_path)
    config.benchmark_cmd = ["python", "benchmark.py", "--device", "cuda:1"]
    configs_file = _write_tuned_configs(config, {"demo": [{"config": {"old": True}}]})
    launches: list[list[str]] = []

    # This seam existed in the buggy parent-device probe. Keeping the fake with
    # raising=False makes the regression test fail if that probe is restored.
    monkeypatch.setattr(
        aot_runner_module,
        "get_hardware_info",
        lambda device=None: CUDA_OTHER,
        raising=False,
    )

    def refresh_configs(
        cmd: list[str],
        env: dict[str, str],
        *args: object,
        **kwargs: object,
    ) -> tuple[int, str, str]:
        launches.append(cmd)
        assert env[aot_cache_module.AOT_RUNNER_HARDWARE_VALIDATION_ENV] == "1"
        configs_file.write_text(json.dumps({"demo": [{"config": {"new": True}}]}))
        return 0, "", ""

    monkeypatch.setattr(aot_runner_module, "run_benchmark", refresh_configs)

    assert aot_runner_module.run_collect_phase(config)
    assert launches == [config.benchmark_cmd]


@pytest.mark.parametrize("mode", ("evaluate", "compile"))
def test_runner_guard_rejects_actual_child_hardware_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    aot_cache_module.write_hardware_manifest(tmp_path, CUDA)
    child_device = torch.device("cuda:1")
    observed_devices: list[torch.device | None] = []

    def child_hardware(device: torch.device | None = None) -> HardwareInfo:
        observed_devices.append(device)
        return CUDA_OTHER

    monkeypatch.setenv("HELION_AOT_MODE", mode)
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(aot_cache_module.AOT_RUNNER_HARDWARE_VALIDATION_ENV, "1")
    monkeypatch.setattr(aot_cache_module, "get_hardware_info", child_hardware)

    with pytest.raises(
        exc.AOTHardwareManifestError, match="hardware identity mismatch"
    ):
        AOTAutotuneCache(_autotuner(child_device))

    assert observed_devices == [child_device]


def test_runner_guard_rejects_source_adjacent_heuristic_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    aot_cache_module.write_hardware_manifest(run_dir, CUDA)
    source_file = tmp_path / "kernel.py"
    source_file.write_text("def demo(device):\n    return device\n")
    executed = tmp_path / "executed"
    (tmp_path / "_helion_aot_kernel_cuda_sm100.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(executed)!r}).write_text('executed')\n"
        "def key_demo(device):\n    return 1\n"
    )
    child_device = torch.device("cuda:1")
    observed_devices: list[torch.device | None] = []

    def child_hardware(device: torch.device | None = None) -> HardwareInfo:
        observed_devices.append(device)
        return CUDA_OTHER

    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    monkeypatch.setenv("HELION_AOT_DATA_DIR", str(run_dir))
    monkeypatch.setenv(aot_cache_module.AOT_RUNNER_HARDWARE_VALIDATION_ENV, "1")
    monkeypatch.setattr(aot_cache_module, "get_hardware_info", child_hardware)

    key_fn = HeuristicKeyFunction(str(source_file), "demo")
    with pytest.raises(
        exc.AOTHardwareManifestError, match="hardware identity mismatch"
    ):
        key_fn(child_device)

    assert observed_devices == [child_device]
    assert not executed.exists()


def test_evaluate_benchmark_failure_fails_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _prepared_run(tmp_path)
    results = {
        "demo": {
            "max_slowdown": 1.0,
            "geomean_slowdown": 1.0,
            "avg_slowdown": 1.0,
        }
    }
    monkeypatch.setattr(
        aot_runner_module,
        "evaluate_heuristic",
        lambda **kwargs: results,
    )

    def fail_benchmark(
        cmd: list[str],
        env: dict[str, str],
        *args: object,
        **kwargs: object,
    ) -> tuple[int, str, str]:
        assert env[aot_cache_module.AOT_RUNNER_HARDWARE_VALIDATION_ENV] == "1"
        return 23, "", ""

    monkeypatch.setattr(aot_runner_module, "run_benchmark", fail_benchmark)

    assert not aot_runner_module.run_evaluate_phase(config)
    assert (
        json.loads((config.run_dir / f"evaluation_{CUDA.hardware_id}.json").read_text())
        == results
    )


def test_resumed_collect_requires_fresh_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _prepared_run(tmp_path)
    configs_file = _write_tuned_configs(config, {"demo": [{"config": {"old": True}}]})
    _assert_phase_requires_refresh(
        monkeypatch,
        config,
        aot_runner_module.run_collect_phase,
        lambda: configs_file.write_text(
            json.dumps({"demo": [{"config": {"fresh": True}}, {"config": {}}]})
        ),
    )


def test_measure_requires_fresh_nonempty_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _prepared_run(tmp_path)
    measurements_file = config.run_dir / f"measurements_{CUDA.hardware_id}.csv"
    measurements_file.write_text("header\nstale-row\n")

    def refresh_measurements() -> None:
        with measurements_file.open("a") as output:
            output.write("fresh-row\n")

    _assert_phase_requires_refresh(
        monkeypatch,
        config,
        aot_runner_module.run_measure_phase,
        refresh_measurements,
    )


def test_compile_requires_fresh_standalone_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _prepared_run(tmp_path)
    source_file = tmp_path / "kernel.py"
    source_file.write_text("def demo():\n    pass\n")
    _write_tuned_configs(config, {"demo": [{"kernel_source_file": str(source_file)}]})
    standalone_file = tmp_path / "kernel_demo_standalone.py"
    standalone_file.write_text("stale\n")
    _assert_phase_requires_refresh(
        monkeypatch,
        config,
        aot_runner_module.run_compile_phase,
        lambda: standalone_file.write_text("fresh standalone\n"),
    )


def test_compile_accepts_resolved_symlink_source_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _prepared_run(tmp_path)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = source_dir / "kernel.py"
    source_file.write_text("def demo():\n    pass\n")
    linked_source = tmp_path / "linked.py"
    linked_source.symlink_to(source_file)
    _write_tuned_configs(config, {"demo": [{"kernel_source_file": str(linked_source)}]})
    unresolved_output = tmp_path / "linked_demo_standalone.py"
    unresolved_output.write_text("stale unresolved\n")
    resolved_output = source_dir / "kernel_demo_standalone.py"
    _assert_phase_requires_refresh(
        monkeypatch,
        config,
        aot_runner_module.run_compile_phase,
        lambda: resolved_output.write_text("fresh resolved\n"),
    )


def test_compile_symlink_loop_uses_run_directory_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _prepared_run(tmp_path)
    first_link = _symlink_loop(tmp_path)
    _write_tuned_configs(config, {"demo": [{"kernel_source_file": str(first_link)}]})
    fallback_output = config.run_dir / "demo_standalone.py"
    _assert_phase_requires_refresh(
        monkeypatch,
        config,
        aot_runner_module.run_compile_phase,
        lambda: fallback_output.write_text("fresh fallback\n"),
    )


def test_compile_rejects_conflicting_eligible_source_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _prepared_run(tmp_path)
    first_source = tmp_path / "first.py"
    second_source = tmp_path / "second.py"
    first_source.write_text("def demo():\n    pass\n")
    second_source.write_text("def demo():\n    pass\n")
    _write_tuned_configs(
        config,
        {
            "demo": [
                {"kernel_source_file": str(first_source)},
                {"kernel_source_file": str(second_source)},
            ]
        },
    )
    assert aot_cache_module.load_kernel_source_files(
        config.run_dir, CUDA.hardware_id
    ) == {"demo": str(first_source)}

    def unexpected_benchmark(*args: object, **kwargs: object) -> tuple[int, str, str]:
        raise AssertionError("ambiguous compile metadata must fail before launch")

    monkeypatch.setattr(aot_runner_module, "run_benchmark", unexpected_benchmark)

    assert not aot_runner_module.run_compile_phase(config)
    assert "kernel 'demo' has conflicting canonical source identities" in caplog.text


@pytest.mark.parametrize(
    "metadata",
    (
        b"\xff",
        json.dumps({"demo": [{"kernel_source_file": None}]}).encode("utf-16"),
        json.dumps({"demo": [{"kernel_source_file": None}]}).encode("utf-32"),
    ),
    ids=("invalid-bytes", "utf-16-json", "utf-32-json"),
)
def test_compile_rejects_non_utf8_metadata_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    metadata: bytes,
) -> None:
    config = _prepared_run(tmp_path)
    (config.run_dir / f"tuned_configs_{CUDA.hardware_id}.json").write_bytes(metadata)

    def unexpected_benchmark(*args: object, **kwargs: object) -> tuple[int, str, str]:
        raise AssertionError("invalid UTF-8 metadata must fail before launch")

    monkeypatch.setattr(aot_runner_module, "run_benchmark", unexpected_benchmark)

    assert not aot_runner_module.run_compile_phase(config)
    assert "is not valid UTF-8" in caplog.text


def test_compile_requires_every_filtered_kernel_to_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _prepared_run(tmp_path)
    source_file = tmp_path / "kernels.py"
    source_file.write_text("def first():\n    pass\n\ndef second():\n    pass\n")
    _write_tuned_configs(
        config,
        {
            kernel_name: [{"kernel_source_file": str(source_file)}]
            for kernel_name in ("first", "second")
        },
    )
    first_output = tmp_path / "kernels_first_standalone.py"
    first_output.write_text("stale first\n")
    launches: list[list[str] | None] = []

    def refresh_first_only(
        cmd: list[str],
        env: dict[str, str],
        log_file: Path,
        phase: str,
        kernels: list[str] | None = None,
    ) -> tuple[int, str, str]:
        launches.append(kernels)
        first_output.write_text(f"fresh first {len(launches)}\n")
        return 0, "", ""

    monkeypatch.setattr(aot_runner_module, "run_benchmark", refresh_first_only)

    assert not aot_runner_module.run_compile_phase(config)
    assert "did not write fresh artifacts for: second" in caplog.text
    config.kernels = ["first"]
    assert aot_runner_module.run_compile_phase(config)
    assert launches == [None, ["first"]]


def test_compile_does_not_require_non_standalone_kernel_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _prepared_run(tmp_path)
    source_file = tmp_path / "kernels.py"
    source_file.write_text("def enabled():\n    pass\n\ndef skipped():\n    pass\n")
    alternate_source = tmp_path / "alternate.py"
    alternate_source.write_text("def skipped():\n    pass\n")
    _write_tuned_configs(
        config,
        {
            "enabled": [
                {
                    "kernel_source_file": str(source_file),
                    "standalone": True,
                }
            ],
            "skipped": [
                {
                    "kernel_source_file": str(source_file),
                    "standalone": False,
                },
                {
                    "kernel_source_file": str(alternate_source),
                    "standalone": False,
                },
            ],
        },
    )
    enabled_output = tmp_path / "kernels_enabled_standalone.py"

    def refresh_enabled(
        cmd: list[str],
        env: dict[str, str],
        log_file: Path,
        phase: str,
        kernels: list[str] | None = None,
    ) -> tuple[int, str, str]:
        enabled_output.write_text("fresh enabled\n")
        return 0, "", ""

    monkeypatch.setattr(aot_runner_module, "run_benchmark", refresh_enabled)
    assert aot_runner_module.run_compile_phase(config)

    config.kernels = ["skipped"]
    monkeypatch.setattr(aot_runner_module, "run_benchmark", _successful_benchmark)
    assert aot_runner_module.run_compile_phase(config)


def test_collect_retry_tolerates_interrupted_manifest_publication(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_name = aot_cache_module.AOT_HARDWARE_MANIFEST
    (run_dir / f".{manifest_name}.lock").write_text("")
    (run_dir / f".{manifest_name}.stale.tmp").write_text("partial\n")

    assert aot_runner_module.resolve_run_hardware(tmp_path, "run", "collect") is None

    (run_dir / f".{manifest_name}.tmp").write_text("not a writer temp\n")
    with pytest.raises(exc.AOTHardwareManifestError, match="Cannot resume AOT run"):
        aot_runner_module.resolve_run_hardware(tmp_path, "run", "collect")


def test_finalize_pending_log_bounds_name_collisions(
    tmp_path: Path,
) -> None:
    pending, authoritative = _pending_logs(tmp_path)
    for attempt in range(aot_runner_module._MAX_PENDING_LOG_FINALIZE_ATTEMPTS):
        destination = (
            authoritative
            if attempt == 0
            else authoritative.with_name(f"{authoritative.name}.{attempt}")
        )
        destination.write_text("existing\n")

    with pytest.raises(RuntimeError, match="log-name collisions"):
        aot_runner_module._finalize_pending_log(
            pending,
            authoritative,
        )

    assert pending.is_file()


@pytest.mark.parametrize("collision", (False, True))
def test_finalize_pending_log_preserves_open_inode_without_hard_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision: bool,
) -> None:
    pending, authoritative = _pending_logs(tmp_path)
    selected = authoritative
    if collision:
        authoritative.write_text("earlier attempt\n")
        selected = tmp_path / "runner_cuda.log.1"
    monkeypatch.setattr(
        aot_runner_module.os,
        "link",
        lambda *_args: pytest.fail("log publication must not use hard links"),
    )

    with pending.open("a") as open_log:
        aot_runner_module._finalize_pending_log(pending, authoritative)
        open_log.write("continued\n")

    if collision:
        assert authoritative.read_text() == "earlier attempt\n"
    assert selected.read_text() == "pending\ncontinued\n"
    assert not pending.exists()


def test_build_uses_recorded_hardware_for_adjacent_heuristic_name(
    tmp_path: Path,
) -> None:
    config = _prepared_run(tmp_path, TPU, manifest=False, logs=False)
    source_file = tmp_path / "kernel.py"
    source_file.write_text("def demo():\n    pass\n")
    measurements_file = config.run_dir / f"measurements_{TPU.hardware_id}.csv"
    with open(measurements_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "kernel_name",
                "shape_hash",
                "config_hash",
                "config",
                "shape_features",
                "timing_ms",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "kernel_name": "demo",
                "shape_hash": "shape",
                "config_hash": "config",
                "config": json.dumps({"block_sizes": [16]}),
                "shape_features": json.dumps({"arg0_dim0": 16}),
                "timing_ms": 1.0,
            }
        )
    (config.run_dir / f"tuned_configs_{TPU.hardware_id}.json").write_text(
        json.dumps({"demo": [{"kernel_source_file": str(source_file)}]})
    )

    assert aot_runner_module.run_build_heuristic_phase(config)
    assert (tmp_path / "_helion_aot_kernel_tpu_tpu-v5p.py").is_file()
    assert not (tmp_path / "_helion_aot_kernel_cuda_sm100.py").exists()
