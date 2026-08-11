from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "cute"


def _load_script(filename: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, BENCHMARK_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cutlass_benchmark() -> Any:
    return _load_script(
        "compare_grouped_gemm_backends.py",
        "helion_test_compare_grouped_gemm_backends",
    )


@pytest.fixture(scope="module")
def deepgemm_benchmark() -> Any:
    return _load_script(
        "deepgemm_selected_path.py",
        "helion_test_deepgemm_selected_path",
    )


def test_published_manifests_are_fixed(
    cutlass_benchmark: Any, deepgemm_benchmark: Any
) -> None:
    manifest = (
        tuple(
            (case.name, case.problems, case.reserved_sms, case.ab_stages)
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
        "608089dc016171e07938f1a67c9c42ed56e292a2d99038b8a2410393d7aa8c74"
    )


def test_cutlass_timings_are_paired_and_balanced(
    cutlass_benchmark: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    thermal_warmups: list[int] = []
    args = cutlass_benchmark._parser().parse_args([])
    assert (args.num_runs, args.cache_warmup_calls, args.thermal_warmup_ms) == (
        6,
        5,
        10000,
    )
    args.num_runs = 2
    args.cache_warmup_calls = args.thermal_warmup_ms = 0

    def fake_bench(fn: Any, _args: argparse.Namespace) -> float:
        fn()
        return 1.0

    monkeypatch.setattr(cutlass_benchmark, "_do_bench", fake_bench)
    monkeypatch.setattr(cutlass_benchmark, "_thermal_warmup", thermal_warmups.append)
    timings = cutlass_benchmark._bench_pair(
        {
            "helion_retained": lambda: order.append("H"),
            "cutlass": lambda: order.append("C"),
        },
        args,
    )

    assert order == list("HCCH")
    assert thermal_warmups == [0]
    assert timings["helion_retained"]["runs_ms"] == [1.0, 1.0]
    with pytest.raises(argparse.ArgumentTypeError, match="positive even"):
        cutlass_benchmark._positive_even("5")


def test_cutlass_loader_verifies_and_retains_bytes(
    cutlass_benchmark: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "grouped_gemm.py"
    source.write_text("raise AssertionError('must not execute')\n")
    with pytest.raises(ValueError, match="CUTLASS source SHA256 mismatch"):
        cutlass_benchmark._load_cutlass_source(source)

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

    module, provenance = cutlass_benchmark._load_cutlass_source(source)
    source.write_text("raise AssertionError('changed after verification')\n")

    assert module.GroupedGemmKernel.marker == "verified"
    assert module.__file__ == f"<helion-cutlass-grouped-gemm-{digest}.py>"
    assert provenance["source_sha256"] == digest
    assert provenance["expected_cutlass_commit"] == cutlass_benchmark.CUTLASS_COMMIT


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
