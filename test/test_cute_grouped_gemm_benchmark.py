from __future__ import annotations

import argparse
import builtins
from contextlib import AbstractContextManager
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any

from benchmarks.cute import grouped_gemm_deepgemm_support as deepgemm_support
from pretuned_kernels import _bench as pretuned_bench
from pretuned_kernels import run as pretuned_runner
from pretuned_kernels.grouped_gemm import (
    _helion_aot_grouped_gemm_cuda_sm100 as grouped_heuristic,
)
from pretuned_kernels.grouped_gemm_deepgemm import (
    _deepgemm_public_api as deepgemm_public_api,
)
import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "cute"
PRETUNED_DIR = Path(__file__).resolve().parents[1] / "pretuned_kernels"
PRETUNED_WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "benchmark_pretuned.yml"
)


def _load_path(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _assert_aot_training_does_not_load_reference(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    main: Callable[[], object],
    reference_path_name: str,
    expected_training_args: tuple[object, ...],
    modes: tuple[str, ...] = ("collect", "compile", "measure"),
) -> None:
    result = {
        "helion_wins": 0,
        "total": 0,
        "geomean": 0.0,
        "best_speedup": 0.0,
        "baselines": {},
    }

    def fake_run_aot_training(*args: object) -> dict[str, object]:
        assert args == expected_training_args
        return result

    monkeypatch.setattr(module, "_run_aot_training", fake_run_aot_training)
    monkeypatch.setattr(
        module,
        reference_path_name,
        lambda: pytest.fail("AOT training loaded an external reference"),
    )
    for mode in modes:
        monkeypatch.setenv("HELION_AOT_MODE", mode)
        assert main() is result


def _assert_dashboard_uses_shared_timer(
    monkeypatch: pytest.MonkeyPatch,
    bench: Any,
    main: Callable[[], object],
    *,
    repetitions: int,
    invoke_make_calls: bool = False,
) -> dict[str, Any]:
    expected: dict[str, Any] = {"ok": True}

    def fake_run_sweep(*args: object, **kwargs: object) -> dict[str, Any]:
        assert kwargs["use_cudagraph"] is False
        assert kwargs["pre_captured_cudagraph"] is True
        assert kwargs["rep"] == repetitions
        assert kwargs["thermal_warmup_ms"] == 10_000
        if invoke_make_calls:
            cases: Any = args[0]
            make_calls: Any = args[1]
            for case in cases:
                make_calls(case)
        return expected

    monkeypatch.setattr(bench, "run_sweep", fake_run_sweep)
    assert main() is expected
    return expected


@pytest.fixture(scope="module")
def cutlass_benchmark() -> Iterator[Any]:
    module_name = "helion_test_compare_grouped_gemm_backends"
    yield _load_path(
        BENCHMARK_DIR / "compare_grouped_gemm_backends.py",
        module_name,
    )
    sys.modules.pop(module_name, None)


@pytest.fixture(scope="module")
def cublas_adapter() -> Iterator[Any]:
    module_name = "helion_test_cublas_grouped_gemm"
    yield _load_path(BENCHMARK_DIR / "cublas_grouped_gemm.py", module_name)
    sys.modules.pop(module_name, None)


def test_published_cutlass_cases_manifest_is_fixed(cutlass_benchmark: Any) -> None:
    manifest = tuple(
        (
            case.name,
            case.problems,
            case.ab_stages,
            case.acc_stages,
            case.c_stages,
        )
        for case in cutlass_benchmark.CASES
    )

    assert hashlib.sha256(repr(manifest).encode()).hexdigest() == (
        "a53f21c946fa3dba89bea7d28c4b3888da05e797bfa8bf4950b94ee3e5ffe5c5"
    )


@pytest.fixture(scope="module")
def pretuned_grouped_gemm() -> Iterator[Any]:
    module_name = "helion_test_pretuned_grouped_gemm"
    yield _load_path(
        PRETUNED_DIR / "grouped_gemm" / "grouped_gemm.py",
        module_name,
    )
    sys.modules.pop(module_name, None)


@pytest.fixture(scope="module")
def pretuned_deepgemm() -> Iterator[Any]:
    module_name = "helion_test_pretuned_grouped_gemm_deepgemm"
    yield _load_path(
        PRETUNED_DIR / "grouped_gemm_deepgemm" / "grouped_gemm_deepgemm.py",
        module_name,
    )
    sys.modules.pop(module_name, None)


def test_bench_module_import_does_not_require_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_module = builtins.__import__
    import_module("helion")

    def import_without_triton(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "triton" or name.startswith("triton."):
            raise ModuleNotFoundError(name)
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_triton)
    module_name = "helion_test_pretuned_bench_without_triton"
    try:
        module = _load_path(PRETUNED_DIR / "_bench.py", module_name)
        assert module.geomean((1.0, 4.0)) == 2.0
    finally:
        sys.modules.pop(module_name, None)


def test_cutlass_timings_use_shared_timer(
    cutlass_benchmark: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    thermal_warmups: list[int] = []
    args = cutlass_benchmark._parser().parse_args(
        ["--repetitions", "12", "--thermal-warmup-ms", "0"]
    )

    def fake_bench(calls: list[Any], rep: int) -> list[float]:
        assert rep == 12
        order.extend(call() for call in calls)
        return [1.0, 2.0]

    monkeypatch.setattr(cutlass_benchmark, "bench_pre_captured_cudagraphs", fake_bench)
    monkeypatch.setattr(cutlass_benchmark, "thermal_warmup", thermal_warmups.append)
    timings = cutlass_benchmark._bench_pair(
        {
            "helion_retained": lambda: "H",
            cutlass_benchmark.CUTLASS_KERNEL_BASELINE: lambda: "K",
        },
        args,
    )

    assert order == ["H", "K"]
    assert thermal_warmups == [0]
    assert timings["helion_retained"]["median_ms"] == 1.0
    assert timings[cutlass_benchmark.CUTLASS_KERNEL_BASELINE]["median_ms"] == 2.0


def test_cutlass_loader_verifies_and_retains_bytes(
    cutlass_benchmark: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "grouped_gemm.py"
    source.write_text("raise AssertionError('must not execute')\n")
    with pytest.raises(ValueError, match="CUTLASS source SHA256 mismatch"):
        cutlass_benchmark.load_cutlass_source(source)

    source.write_text(
        "class GroupedGemmKernel:\n"
        "    marker = 'verified'\n"
        "    num_tensormaps = bytes_per_tensormap = 1\n"
        "def create_tensor_and_stride(*args):\n"
        "    return args\n"
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(cutlass_benchmark, "CUTLASS_SHA256", digest)
    monkeypatch.setattr(
        cutlass_benchmark.importlib.metadata,
        "version",
        lambda _name: "test",
    )

    module, provenance = cutlass_benchmark.load_cutlass_source(source)
    source.write_text("raise AssertionError('changed after verification')\n")

    assert module.GroupedGemmKernel.marker == "verified"
    assert module.__file__ == f"<helion-cutlass-grouped-gemm-{digest}.py>"
    assert provenance["source_sha256"] == digest
    assert provenance["expected_cutlass_commit"] == cutlass_benchmark.CUTLASS_COMMIT


def test_cutlass_summary_reports_kernel_baseline(
    cutlass_benchmark: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = cutlass_benchmark.CASES[0]
    case_dir = tmp_path / "cutlass" / case.name
    case_dir.mkdir(parents=True)
    (case_dir / "result.json").write_text(
        '{"problem_sizes": [], "timings": {}, "helion_over_cutlass_kernel": 0.8}'
    )
    monkeypatch.setattr(cutlass_benchmark, "_git", lambda *_args: "test")
    summary = cutlass_benchmark._build_summary(
        argparse.Namespace(
            out_dir=tmp_path,
            cutlass_source=tmp_path / "grouped_gemm.py",
        ),
        [case],
        [],
        [],
    )

    assert summary["wins_vs_cutlass_kernel"] == 1
    assert summary["geomean_helion_over_cutlass_kernel"] == 0.8


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16),
    ids=("fp16", "bf16"),
)
def test_cublas_grouped_adapter_matches_torch(
    cublas_adapter: Any,
    dtype: torch.dtype,
) -> None:
    if torch.version.cuda is None or not torch.cuda.is_available():
        pytest.skip("cuBLAS grouped adapter requires an NVIDIA CUDA device")
    device = torch.device("cuda", torch.cuda.current_device())
    problems = ((3, 16, 8, 1), (5, 24, 16, 1))

    group_a = tuple(
        torch.randn((m, k), device=device, dtype=dtype) for m, _n, k, _batch in problems
    )
    group_b = tuple(
        torch.randn((n, k), device=device, dtype=dtype) for _m, n, k, _batch in problems
    )
    outputs = tuple(
        torch.empty((m, n), device=device, dtype=dtype) for m, n, _k, _batch in problems
    )
    expected = tuple(a @ b.T for a, b in zip(group_a, group_b, strict=True))

    launch, _provenance = cublas_adapter.prepare_cublas(
        problems,
        group_a,
        group_b,
        outputs,
    )
    launch()
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for output in outputs:
        output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize(device)

    for actual, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("selected", "expected"),
    (
        ("", {"cutlass": True, "deepgemm": True}),
        ("grouped_gemm", {"cutlass": True, "deepgemm": False}),
        ("grouped_gemm_deepgemm", {"cutlass": False, "deepgemm": True}),
        (
            "grouped_gemm, grouped_gemm_deepgemm",
            {"cutlass": True, "deepgemm": True},
        ),
    ),
)
def test_grouped_reference_selection(
    selected: str,
    expected: dict[str, bool],
) -> None:
    assert pretuned_runner.grouped_reference_requirements(selected) == expected


def test_grouped_reference_workflow_pins_match_benchmarks(
    cutlass_benchmark: Any,
) -> None:
    workflow = PRETUNED_WORKFLOW.read_text()
    for pin in (
        cutlass_benchmark.CUTLASS_COMMIT,
        cutlass_benchmark.CUTLASS_SHA256,
        deepgemm_support.DEEPGEMM_COMMIT,
    ):
        assert pin in workflow
    assert 'KERNEL_ARGS=(--kernels "$SELECTED_KERNELS")' in workflow
    assert '"${KERNEL_ARGS[@]}"' in workflow


def test_generated_grouped_heuristic_selects_measured_and_fallback_pipelines(
    pretuned_grouped_gemm: Any,
) -> None:
    static_signature_key = "tcgen05_grouped_static_problem_signature"
    expected_measured_stages = [(2, 1, 2), *((8, 2, 4),) * 6]
    measured_signatures = tuple(
        pretuned_grouped_gemm._problem_signature(case.problems)
        for case in pretuned_grouped_gemm.CASES
    )

    assert len(set(measured_signatures)) == len(measured_signatures)
    assert len(grouped_heuristic.CONFIGS) == len(measured_signatures)
    for signature, published, expected_stages in zip(
        measured_signatures,
        grouped_heuristic.CONFIGS,
        expected_measured_stages,
        strict=True,
    ):
        selected = grouped_heuristic.autotune_grouped_gemm(*signature)
        assert selected == published
        assert (
            selected["tcgen05_ab_stages"],
            selected["tcgen05_acc_stages"],
            selected["tcgen05_c_stages"],
        ) == expected_stages
        assert selected[static_signature_key] == list(signature[: 1 + 3 * signature[0]])

    for signature, expected_stages in (
        ((1, 256, 64, 64), (2, 1, 2)),
        ((1, 8192, 8192, 64), (8, 2, 4)),
    ):
        selected = grouped_heuristic.autotune_grouped_gemm(*signature)
        assert (
            selected["tcgen05_ab_stages"],
            selected["tcgen05_acc_stages"],
            selected["tcgen05_c_stages"],
        ) == expected_stages
        assert selected[static_signature_key] == list(signature)


def test_pretuned_grouped_kernels_register_for_b200_and_ship_only_sm100() -> None:
    for name in ("grouped_gemm", "grouped_gemm_deepgemm"):
        assert name in pretuned_runner.KERNELS
        assert pretuned_runner._supported_hardware(name) == {"b200"}
        heuristic_files = sorted(
            path.name
            for path in (pretuned_runner.PRETUNED_KERNELS_DIR / name).glob(
                f"_helion_aot_{name}_cuda_sm*.py"
            )
        )
        assert heuristic_files == [f"_helion_aot_{name}_cuda_sm100.py"]


def test_pretuned_runner_preserves_benchmark_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {"provider": {"commit": "abc"}}
    details = {"rows": [{"config": "reviewed"}]}
    module = SimpleNamespace(
        main=lambda verbose: {
            "helion_wins": 1,
            "total": 1,
            "geomean": 1.1,
            "best_speedup": 1.1,
            "baselines": {},
            "benchmark_metadata": metadata,
            "benchmark_details": details,
        },
        use_cudagraph=lambda: True,
    )
    monkeypatch.setattr(pretuned_runner, "_supported_hardware", lambda _name: {"b200"})
    monkeypatch.setattr(pretuned_runner, "_import_kernel_module", lambda _name: module)

    record = pretuned_runner.run_kernel("grouped_gemm_deepgemm", "b200")

    assert record["benchmark_metadata"] is metadata
    assert record["benchmark_details"] is details


def test_pretuned_deepgemm_reference_computes_valid_rows_and_zero_padding(
    pretuned_deepgemm: Any,
) -> None:
    a = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    b = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    worklist = torch.tensor(((0, 0, 1, 2), (1, 2, 1, 2)), dtype=torch.int32)

    output = pretuned_deepgemm._reference(a, b, worklist)

    torch.testing.assert_close(output[0], a[0] @ b[0].T)
    torch.testing.assert_close(output[2], a[2] @ b[1].T)
    torch.testing.assert_close(output[[1, 3]], torch.zeros_like(output[[1, 3]]))


@pytest.mark.parametrize("replay_writes_output", (True, False))
def test_grouped_gemm_tuner_validates_pointer_targets(
    pretuned_grouped_gemm: Any,
    monkeypatch: pytest.MonkeyPatch,
    replay_writes_output: bool,
) -> None:
    output = torch.zeros(2)
    expected = torch.tensor([1.0, 2.0])
    failures: list[object] = []
    cleared: list[object] = []

    def replay() -> None:
        if replay_writes_output:
            output.copy_(expected)

    class FakeCapture(AbstractContextManager[SimpleNamespace]):
        def __enter__(self) -> SimpleNamespace:
            return SimpleNamespace(replay=replay)

        def __exit__(self, *args: object) -> None:
            return None

    provider = pretuned_grouped_gemm._ColdCudagraphBenchmarkProvider.__new__(
        pretuned_grouped_gemm._ColdCudagraphBenchmarkProvider
    )
    provider._validation = pretuned_grouped_gemm._GroupedValidation(
        (output,), (expected,)
    )
    provider._repetitions = 6
    provider.args = ()
    provider.settings = SimpleNamespace(
        autotune_accuracy_check=True,
        autotune_ignore_errors=True,
    )
    provider._autotune_metrics = SimpleNamespace(num_configs_tested=0)
    provider._record_accuracy_failure = failures.append
    provider._clear_jit_fast_path_caches = cleared.append
    provider.log = SimpleNamespace(warning=lambda _message: None)

    monkeypatch.setattr(
        pretuned_grouped_gemm.helion_runtime, "cute_cuda_graph", FakeCapture
    )
    monkeypatch.setattr(pretuned_grouped_gemm.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        pretuned_grouped_gemm._BENCH,
        "bench_pre_captured_cudagraph",
        lambda _call, rep: 0.5,
    )

    def candidate() -> torch.Tensor:
        output.copy_(expected)
        return output

    config = object()
    perf = provider._benchmark_function(config, candidate)

    assert perf == (0.5 if replay_writes_output else float("inf"))
    assert failures == ([] if replay_writes_output else [config])
    assert provider._autotune_metrics.num_configs_tested == 1
    assert cleared == [candidate]


def test_grouped_gemm_aot_training_does_not_load_cutlass(
    pretuned_grouped_gemm: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_aot_training_does_not_load_reference(
        monkeypatch,
        pretuned_grouped_gemm,
        lambda: pretuned_grouped_gemm.main(verbose=False),
        "_cutlass_source_path",
        (False,),
    )


def test_deepgemm_aot_training_does_not_load_reference(
    pretuned_deepgemm: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_aot_training_does_not_load_reference(
        monkeypatch,
        deepgemm_public_api,
        lambda: deepgemm_public_api.main(
            pretuned_deepgemm.create_grouped_gemm_deepgemm_kernel,
            verbose=False,
        ),
        "_deepgemm_root",
        (pretuned_deepgemm.create_grouped_gemm_deepgemm_kernel, False),
        modes=("collect", "measure"),
    )


def test_pre_captured_graph_sweep_uses_shared_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], int]] = []

    def timer(functions: list[Any], rep: int) -> list[float]:
        calls.append(([function() for function in functions], rep))
        return [1.0, 2.0]

    monkeypatch.setattr(pretuned_bench, "bench_pre_captured_cudagraphs", timer)
    metrics = pretuned_bench.run_sweep(
        ["shape"],
        lambda _shape: (
            lambda: "helion",
            [("cutlass", lambda: "cutlass")],
            "shape",
        ),
        use_cudagraph=False,
        pre_captured_cudagraph=True,
        shape_header="case",
        rep=7,
        verbose=False,
    )

    assert calls == [(["helion", "cutlass"], 7)]
    assert metrics["geomean"] == 2.0


def test_grouped_gemm_dashboard_selects_shared_timer(
    pretuned_grouped_gemm: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    monkeypatch.setattr(pretuned_grouped_gemm, "_cutlass_source_path", lambda: tmp_path)
    monkeypatch.setattr(
        pretuned_grouped_gemm._COMPARE,
        "load_cutlass_source",
        lambda _path: (object(), {}),
    )

    _assert_dashboard_uses_shared_timer(
        monkeypatch,
        pretuned_grouped_gemm._BENCH,
        lambda: pretuned_grouped_gemm.main(verbose=False),
        repetitions=204,
    )


def test_deepgemm_dashboard_selects_shared_timer(
    pretuned_deepgemm: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    monkeypatch.setattr(deepgemm_public_api, "_deepgemm_root", lambda: tmp_path)
    monkeypatch.setattr(
        deepgemm_public_api._SUPPORT,
        "import_deepgemm",
        lambda _root, _alignment: (object(), {}),
    )

    def fake_captured_calls(
        case: tuple[Any, ...],
        _deep_gemm: object,
        _kernel_factory: object,
        selected_configs: dict[int, dict[str, object]],
    ) -> tuple[object, list[object], str]:
        shape = case[0]
        selected_configs[shape.row_index] = {"config_name": f"row-{shape.row_index}"}
        return object(), [], "case"

    monkeypatch.setattr(deepgemm_public_api, "_captured_calls", fake_captured_calls)
    expected = _assert_dashboard_uses_shared_timer(
        monkeypatch,
        deepgemm_public_api._BENCH,
        lambda: deepgemm_public_api.main(
            pretuned_deepgemm.create_grouped_gemm_deepgemm_kernel,
            verbose=False,
        ),
        repetitions=102,
        invoke_make_calls=True,
    )
    metadata = expected["benchmark_metadata"]
    assert metadata["deepgemm_api"] == {
        "function": "m_grouped_bf16_gemm_nt_contiguous",
        "b_major": "k",
        "compiled_dims": "nk",
        "use_psum_layout": False,
        "ensure_zero_padding": False,
        "m_alignment": deepgemm_support.M_ALIGNMENT,
    }
    assert metadata["reviewed_profile_manifest_sha256"]
    assert len(expected["benchmark_details"]["reviewed_helion_configs"]) == 8


@pytest.mark.parametrize(
    ("environment", "message"),
    (
        ({"HELION_AOT_MODE": "disabled"}, "requires HELION_AOT_MODE=evaluate"),
        ({"HELION_AOT_MODE": "compile"}, "requires HELION_AOT_MODE=evaluate"),
        (
            {"HELION_AOT_MODE": "evaluate", "HELION_HEURISTIC_DIR": "/tmp/other"},
            "does not permit HELION_HEURISTIC_DIR",
        ),
    ),
)
def test_deepgemm_dashboard_rejects_aot_overrides(
    pretuned_deepgemm: Any,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    message: str,
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        deepgemm_public_api.main(
            pretuned_deepgemm.create_grouped_gemm_deepgemm_kernel,
            verbose=False,
        )


def test_pre_captured_graph_timer_balances_and_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[str] = []
    event_count = 36
    next_event = 0

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            nonlocal next_event
            assert enable_timing
            self.index = next_event
            next_event += 1

        def record(self) -> None:
            operations.append("start" if self.index < event_count // 2 else "end")

        def elapsed_time(self, _end: FakeEvent) -> float:
            return float(self.index // 6 + 1)

    class FakeDeviceInterface:
        Event = FakeEvent

        @staticmethod
        def synchronize() -> None:
            operations.append("sync")

    driver = SimpleNamespace(
        get_device_interface=lambda: FakeDeviceInterface,
        get_empty_cache_for_benchmark=object,
        clear_cache=lambda _cache: operations.append("clear"),
    )
    monkeypatch.setitem(
        sys.modules,
        "triton",
        SimpleNamespace(runtime=SimpleNamespace(driver=SimpleNamespace(active=driver))),
    )

    names = ("A", "B", "C")
    timings = pretuned_bench.bench_pre_captured_cudagraphs(
        [lambda name=name: operations.append(name) for name in names], rep=6
    )

    order = ["A", "B", "C", "B", "C", "A", "C", "A", "B"]
    order += ["C", "B", "A", "A", "C", "B", "B", "A", "C"]
    assert operations[:18] == order
    assert operations[18] == "sync"
    measured: list[str] = []
    for name in order:
        measured.extend(("clear", "start", name, "end"))
    assert operations[19:-1] == measured
    assert operations[-1] == "sync"
    assert timings == [1.0, 2.0, 3.0]
