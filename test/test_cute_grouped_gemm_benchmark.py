from __future__ import annotations

import argparse
import builtins
from contextlib import AbstractContextManager
import hashlib
import importlib.util
from itertools import starmap
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "cute"
PRETUNED_DIR = Path(__file__).resolve().parents[1] / "pretuned_kernels"


def _load_path(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
    module = _load_path(PRETUNED_DIR / "_bench.py", module_name)
    assert module.geomean((1.0, 4.0)) == 2.0
    sys.modules.pop(module_name)


@pytest.fixture(scope="module")
def cutlass_benchmark() -> Any:
    return _load_path(
        BENCHMARK_DIR / "compare_grouped_gemm_backends.py",
        "helion_test_compare_grouped_gemm_backends",
    )


@pytest.fixture(scope="module")
def cublas_adapter() -> Any:
    return _load_path(
        BENCHMARK_DIR / "cublas_grouped_gemm.py",
        "helion_test_cublas_grouped_gemm",
    )


@pytest.fixture(scope="module")
def deepgemm_benchmark() -> Any:
    return _load_path(
        BENCHMARK_DIR / "deepgemm_selected_path.py",
        "helion_test_deepgemm_selected_path",
    )


def test_published_manifests_are_fixed(
    cutlass_benchmark: Any, deepgemm_benchmark: Any
) -> None:
    manifest = (
        tuple(
            (
                case.name,
                case.problems,
                case.ab_stages,
                case.acc_stages,
                case.c_stages,
            )
            for case in cutlass_benchmark.CASES
        ),
        tuple(tuple(shape) for shape in deepgemm_benchmark.OFFICIAL_SHAPES),
        (
            deepgemm_benchmark.DEEPGEMM_SELECTED_TILE_M,
            deepgemm_benchmark.DEEPGEMM_SELECTED_TILE_N,
            deepgemm_benchmark.DEEPGEMM_SELECTED_TILE_K,
            deepgemm_benchmark.M_ALIGNMENT,
        ),
        deepgemm_benchmark.selected_config().config,
    )
    assert hashlib.sha256(repr(manifest).encode()).hexdigest() == (
        "5995339a3112d28f958a8a6bbe2f36fddae01e7c914feaeccd4ee3d394283efc"
    )


@pytest.fixture(scope="module")
def pretuned_grouped_gemm() -> Any:
    return _load_path(
        PRETUNED_DIR / "grouped_gemm" / "grouped_gemm.py",
        "helion_test_pretuned_grouped_gemm",
    )


@pytest.fixture(scope="module")
def grouped_gemm_heuristic() -> Any:
    return _load_path(
        PRETUNED_DIR / "grouped_gemm" / "_helion_aot_grouped_gemm_cuda_sm100.py",
        "helion_test_pretuned_grouped_gemm_heuristic",
    )


@pytest.fixture(scope="module")
def pretuned_deepgemm() -> Any:
    return _load_path(
        PRETUNED_DIR / "grouped_gemm_deepgemm" / "grouped_gemm_deepgemm.py",
        "helion_test_pretuned_grouped_gemm_deepgemm",
    )


@pytest.fixture(scope="module")
def deepgemm_heuristic() -> Any:
    return _load_path(
        PRETUNED_DIR
        / "grouped_gemm_deepgemm"
        / "_helion_aot_grouped_gemm_deepgemm_cuda_sm100.py",
        "helion_test_pretuned_grouped_gemm_deepgemm_heuristic",
    )


@pytest.fixture(scope="module")
def pretuned_bench() -> Any:
    return _load_path(PRETUNED_DIR / "_bench.py", "helion_test_pretuned_bench")


def test_cutlass_timings_use_shared_timer(
    cutlass_benchmark: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    thermal_warmups: list[int] = []
    args = cutlass_benchmark._parser().parse_args([])
    args.repetitions = 12
    args.thermal_warmup_ms = 0

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
    cutlass_benchmark: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
        cutlass_benchmark.importlib.metadata, "version", lambda _name: "test"
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


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_cublas_grouped_adapter_matches_torch(
    cublas_adapter: Any, dtype: torch.dtype
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
        problems, group_a, group_b, outputs
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


def test_pretuned_grouped_gemm_configs(
    pretuned_grouped_gemm: Any,
    grouped_gemm_heuristic: Any,
) -> None:
    candidates = pretuned_grouped_gemm._AOT_CONFIGS
    assert [
        (
            config["tcgen05_ab_stages"],
            config["tcgen05_acc_stages"],
            config["tcgen05_c_stages"],
        )
        for config in candidates
    ] == [(2, 1, 2), (8, 2, 4)]

    signatures = tuple(
        pretuned_grouped_gemm._problem_signature(case.problems)
        for case in pretuned_grouped_gemm.CASES
    )
    assert len(set(signatures)) == len(signatures)
    static_key = pretuned_grouped_gemm.STATIC_PROBLEM_SIGNATURE_CONFIG_KEY
    selected = tuple(starmap(grouped_gemm_heuristic.autotune_grouped_gemm, signatures))
    assert [
        (config["tcgen05_ab_stages"], config["tcgen05_acc_stages"])
        for config in selected
    ] == [(2, 1), *((8, 2),) * 6]
    for signature, config in zip(signatures, selected, strict=True):
        assert config[static_key] == list(signature[: 1 + 3 * signature[0]])

    unseen = (1, 256, 64, 64)
    fallback = grouped_gemm_heuristic.autotune_grouped_gemm(*unseen)
    assert fallback[static_key] == list(unseen)
    assert fallback["tcgen05_ab_stages"] == 2


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


def test_pretuned_deepgemm_contract(
    deepgemm_benchmark: Any,
    pretuned_deepgemm: Any,
    deepgemm_heuristic: Any,
) -> None:
    assert [
        config["tcgen05_ab_stages"] for config in pretuned_deepgemm._AOT_CONFIGS
    ] == [4, 5, 6, 7]
    assert deepgemm_heuristic.autotune_grouped_gemm_deepgemm(1, 2, 3) == (
        deepgemm_benchmark.selected_config().config
    )

    a = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    b = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    worklist = torch.tensor(((0, 0, 1, 2), (1, 2, 1, 2)), dtype=torch.int32)
    output = pretuned_deepgemm._reference(a, b, worklist)
    torch.testing.assert_close(output[0], a[0] @ b[0].T)
    torch.testing.assert_close(output[2], a[2] @ b[1].T)
    torch.testing.assert_close(output[[1, 3]], torch.zeros_like(output[[1, 3]]))


@pytest.mark.parametrize("replay_writes_output", (True, False))
def test_deepgemm_tuner_validates_captured_replay(
    pretuned_deepgemm: Any,
    monkeypatch: pytest.MonkeyPatch,
    replay_writes_output: bool,
) -> None:
    output = torch.zeros(2)
    expected = torch.tensor([1.0, 2.0])
    failures: list[object] = []

    def replay() -> None:
        if replay_writes_output:
            output.copy_(expected)

    class FakeCapture(AbstractContextManager[SimpleNamespace]):
        def __enter__(self) -> SimpleNamespace:
            return SimpleNamespace(replay=replay)

        def __exit__(self, *args: object) -> None:
            return None

    provider = pretuned_deepgemm._ColdCudagraphBenchmarkProvider.__new__(
        pretuned_deepgemm._ColdCudagraphBenchmarkProvider
    )
    provider.args = ()
    provider.settings = SimpleNamespace(autotune_accuracy_check=True)
    provider._record_accuracy_failure = failures.append
    provider._validate_against_baseline = lambda _config, actual, _args: torch.equal(
        actual, expected
    )

    monkeypatch.setattr(
        pretuned_deepgemm.helion_runtime, "cute_cuda_graph", FakeCapture
    )
    monkeypatch.setattr(pretuned_deepgemm.torch.cuda, "synchronize", lambda: None)

    def candidate() -> torch.Tensor:
        output.copy_(expected)
        return output

    config = object()
    captured = provider._capture_validated_replay(config, candidate)
    assert (captured is not None) is replay_writes_output
    assert failures == ([] if replay_writes_output else [config])


def test_grouped_gemm_aot_training_does_not_load_cutlass(
    pretuned_grouped_gemm: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        "helion_wins": 0,
        "total": 0,
        "geomean": 0.0,
        "best_speedup": 0.0,
        "baselines": {},
    }
    monkeypatch.setattr(
        pretuned_grouped_gemm, "_run_aot_training", lambda _verbose: result
    )
    monkeypatch.setattr(
        pretuned_grouped_gemm,
        "_cutlass_source_path",
        lambda: pytest.fail("AOT training loaded an external reference"),
    )
    for mode in ("collect", "compile", "measure"):
        monkeypatch.setenv("HELION_AOT_MODE", mode)
        assert pretuned_grouped_gemm.main(verbose=False) is result


def test_deepgemm_aot_training_does_not_load_reference(
    pretuned_deepgemm: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        "helion_wins": 0,
        "total": 0,
        "geomean": 0.0,
        "best_speedup": 0.0,
        "baselines": {},
    }
    monkeypatch.setattr(pretuned_deepgemm, "_run_aot_training", lambda _verbose: result)
    monkeypatch.setattr(
        pretuned_deepgemm,
        "_deepgemm_root",
        lambda: pytest.fail("AOT training loaded an external reference"),
    )
    for mode in ("collect", "compile", "measure"):
        monkeypatch.setenv("HELION_AOT_MODE", mode)
        assert pretuned_deepgemm.main(verbose=False) is result


def test_pre_captured_graph_sweep_uses_shared_timer(
    pretuned_bench: Any, monkeypatch: pytest.MonkeyPatch
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

    recorded: dict[str, object] = {}
    expected = {"ok": True}

    def fake_run_sweep(*args: object, **kwargs: object) -> dict[str, bool]:
        recorded["args"] = args
        recorded.update(kwargs)
        return expected

    monkeypatch.setattr(pretuned_grouped_gemm._BENCH, "run_sweep", fake_run_sweep)
    assert pretuned_grouped_gemm.main(verbose=False) is expected
    assert recorded["use_cudagraph"] is False
    assert recorded["pre_captured_cudagraph"] is True
    assert recorded["rep"] == 204
    assert recorded["thermal_warmup_ms"] == 10_000


def test_deepgemm_dashboard_selects_shared_timer(
    pretuned_deepgemm: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HELION_AOT_MODE", "evaluate")
    monkeypatch.setattr(pretuned_deepgemm, "_deepgemm_root", lambda: tmp_path)
    monkeypatch.setattr(
        pretuned_deepgemm._HARNESS,
        "import_deepgemm",
        lambda _root, _alignment: (object(), {}),
    )

    recorded: dict[str, object] = {}
    expected = {"ok": True}

    def fake_run_sweep(*args: object, **kwargs: object) -> dict[str, bool]:
        recorded["args"] = args
        recorded.update(kwargs)
        return expected

    monkeypatch.setattr(pretuned_deepgemm._BENCH, "run_sweep", fake_run_sweep)
    assert pretuned_deepgemm.main(verbose=False) is expected
    assert recorded["use_cudagraph"] is False
    assert recorded["pre_captured_cudagraph"] is True
    assert recorded["rep"] == 102
    assert recorded["thermal_warmup_ms"] == 10_000


def test_pre_captured_graph_timer_balances_and_clears(
    pretuned_bench: Any, monkeypatch: pytest.MonkeyPatch
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


def test_deepgemm_rows_and_single_rng_stream(deepgemm_benchmark: Any) -> None:
    actual = deepgemm_benchmark.official_actual_ms(seed=0)
    expected = (
        (9884, 9459, 7801, 7007),
        (8247, 7724, 9586, 7225),
        (8076, 8601, 10197, 8215),
        (7119, 9449, 8773, 6965),
        (5102, 5282, 4858, 5084, 3629, 4660, 5076, 4548),
        (4027, 3114, 3934, 4368, 5111, 5242, 4039, 4993),
        (3507, 4845, 4215, 2901, 4635, 3847, 4894, 4509),
        (2870, 4080, 4999, 3466, 3666, 5006, 3336, 4261),
    )
    selected = deepgemm_benchmark.parse_rows("7,2-4,2")

    assert actual == expected
    assert selected == [2, 3, 4, 7]
    assert tuple(actual[row] for row in selected) == tuple(
        expected[row] for row in selected
    )
    assert deepgemm_benchmark.parse_rows("all") == list(range(8))
    with pytest.raises(argparse.ArgumentTypeError, match="invalid row range"):
        deepgemm_benchmark.parse_rows("4-2")
    with pytest.raises(argparse.ArgumentTypeError, match="out of range"):
        deepgemm_benchmark.parse_rows("8")


def test_deepgemm_timings_are_paired_and_alternate_first_graph(
    deepgemm_benchmark: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay_order: list[str] = []
    synchronizations = 0

    class FakeGraph:
        def __init__(self, name: str) -> None:
            self.name = name

        def replay(self) -> None:
            replay_order.append(self.name)

    class FakeEvent:
        def record(self) -> None:
            pass

        def elapsed_time(self, end: FakeEvent) -> float:
            return 2.0

    class FakeL2Flush:
        zero_calls = 0

        def zero_(self) -> None:
            self.zero_calls += 1

    def synchronize() -> None:
        nonlocal synchronizations
        synchronizations += 1

    monkeypatch.setattr(
        deepgemm_benchmark.torch.cuda, "Event", lambda **_kwargs: FakeEvent()
    )
    monkeypatch.setattr(deepgemm_benchmark.torch.cuda, "synchronize", synchronize)
    l2_flush = FakeL2Flush()
    monkeypatch.setattr(
        deepgemm_benchmark.torch,
        "empty",
        lambda size, **_kwargs: l2_flush if size == 1 else None,
    )
    thermal_warmups: list[int] = []
    monkeypatch.setattr(deepgemm_benchmark, "thermal_warmup", thermal_warmups.append)
    args = argparse.Namespace(
        samples=4,
        iters=2,
        warmups=2,
        thermal_warmup_ms=0,
        l2_flush_bytes=4,
    )

    first, second = deepgemm_benchmark.graph_timings(
        (FakeGraph("H"), FakeGraph("D")), args, flops=4_000_000_000
    )

    assert replay_order == list("HDDHHHDDDDHHHHDDDDHH")
    assert synchronizations == 1 + args.samples
    assert thermal_warmups == [0]
    assert l2_flush.zero_calls == 2 * args.samples * args.iters
    assert first["samples_us"] == second["samples_us"] == [2000.0] * args.samples
    defaults = deepgemm_benchmark.build_arg_parser().parse_args([])
    assert (
        defaults.samples,
        defaults.iters,
        defaults.warmups,
        defaults.thermal_warmup_ms,
        defaults.l2_flush_bytes,
    ) == (10, 50, 5, 10000, 8_000_000_000)
