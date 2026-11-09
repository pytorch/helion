from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.cute import compare_example_backends as compare
from benchmarks.cute import example_baselines as baselines
from benchmarks.cute import example_cases as cases
from benchmarks.cute.example_report import aggregate
from benchmarks.cute.example_report import report_results
import pytest
import torch
import torch._dynamo.exc

pytestmark = pytest.mark.cpu_only


def test_catalog_covers_example_kernel_modules():
    root = Path(__file__).resolve().parents[1] / "examples"
    kernel_modules = set()
    for path in root.rglob("*.py"):
        if "distributed" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        if any(
            isinstance(node, ast.FunctionDef)
            and any("kernel" in ast.unparse(d) for d in node.decorator_list)
            for node in ast.walk(tree)
        ):
            kernel_modules.add(".".join(path.relative_to(root).with_suffix("").parts))
    assert kernel_modules - set(cases.MODULES) == set()
    assert len(cases.MODULES) == len(set(cases.MODULES))
    assert all(len(shapes) == 3 for _, shapes in cases.CHECKS.values())


def test_collection_restores_example_recorder():
    import helion._testing as testing

    original = testing.run_example

    def check(rows, cols):
        return testing.run_example(torch.add, torch.add, (torch.ones(2), torch.ones(2)))

    with patch(
        "helion._testing.import_example", return_value=SimpleNamespace(check=check)
    ):
        result = cases.collect("add", 0)
    assert testing.run_example is original
    assert result[0].name == "00_forward"
    assert result[0].kernels == {"helion": torch.add}

    def broken(rows, cols):
        raise ValueError("fixture failed")

    with (
        patch(
            "helion._testing.import_example", return_value=SimpleNamespace(check=broken)
        ),
        pytest.raises(ValueError, match="fixture failed"),
    ):
        cases.collect("add", 0)
    assert testing.run_example is original


def test_complete_int4_and_splitk_catalogues_without_optional_imports():
    for module, name, expected in (
        ("int4_gemm", "00_forward", {"quack", "quack.compile"}),
        (
            "matmul_split_k",
            "01_forward",
            {"quack", "quack.compile", "quack.split_k", "quack.split_k.compile"},
        ),
    ):
        case = cases.Case(module, name, (), {}, {"aten": torch.add})
        assert expected <= baselines.baseline_factories(case).keys()


def test_quack_jagged_bmm_baseline_is_offered_at_every_d():
    # quack's varlen-M gemm read back 95.7% mismatched at the D=32 workload
    # and passed at D=64 and D=128: the candidate stays in every inventory, so
    # the D=32 mismatch is a recorded baseline failure, not a hidden one.
    offsets = torch.tensor([0, 3, 5])
    for features in (32, 64, 128):
        case = cases.Case(
            "jagged_dense_bmm",
            "00_forward",
            (
                offsets,
                torch.ones(5, features),
                torch.ones(2, features, 16),
                torch.ones(2, 16),
            ),
            {},
            {"aten": torch.add},
        )
        factories = baselines.baseline_factories(case)
        assert {
            "quack.grouped_mm",
            "aten.grouped_mm",
            "aten.padded_bmm",
            "compile.padded_bmm",
        } <= factories.keys(), features


@pytest.mark.parametrize(
    ("error", "invalidated"),
    [
        (
            RuntimeError(
                "CUDA error: operation failed due to a previous error during capture"
            ),
            True,
        ),
        (
            RuntimeError(
                "Cannot copy between CPU and CUDA tensors during CUDA graph capture"
            ),
            True,
        ),
        (
            RuntimeError(
                "CUDA error: operation not permitted when stream is capturing"
            ),
            True,
        ),
        (torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB"), False),
        (ValueError("expected a contiguous tensor"), False),
        # Naming the capture is not one of CUDA's three messages.
        (RuntimeError("captured output has the wrong shape"), False),
    ],
)
def test_only_a_host_synchronization_makes_a_call_uncapturable(error, invalidated):
    # ``measure`` downgrades a baseline to eager-only timings only when CUDA
    # reports the capture itself invalidated; a failure that would fail
    # eagerly too stays a graph error the worker fails on.
    assert baselines.capture_invalidated(error) is invalidated


def test_a_previous_capture_error_is_classified_by_its_chained_cause():
    # CUDA reports "a previous error during capture" after any failure inside
    # the capture; the failure itself is chained to it and is what the record
    # names and ``measure`` classifies.
    previous = "CUDA error: operation failed due to a previous error during capture"
    oom = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
    error = RuntimeError(previous)
    error.__context__ = oom
    assert baselines.capture_cause(error) is oom
    assert baselines.capture_invalidated(error) is False
    error.__cause__ = RuntimeError("operation not permitted when stream is capturing")
    assert baselines.capture_cause(error) is error.__cause__
    assert baselines.capture_invalidated(error) is True
    # Unchained, the report itself is the host synchronization's symptom.
    bare = RuntimeError(previous)
    assert baselines.capture_cause(bare) is bare
    assert baselines.capture_invalidated(bare) is True


def test_backward_callable_uses_identical_upstream_gradients():
    x = torch.randn(3, 4, requires_grad=True)
    case = cases.Case("test", "01_backward", (x,), {}, {}, mode="backward")
    direct = baselines.make_callable(lambda value: value.square(), case)
    different_layout = baselines.make_callable(
        lambda value: value.square().t().contiguous().t(), case
    )
    torch.testing.assert_close(direct()[0], different_layout()[0])


def test_collection_mirrors_run_example_comparison_policy():
    import helion._testing as testing

    def check(rows, cols):
        x = torch.ones(2, requires_grad=True)
        testing.run_example(
            torch.square, torch.square, (x,), bwd=True, bwd_relative_l2=0.02
        )
        testing.run_example(
            torch.add,
            torch.add,
            (torch.ones(2), torch.ones(2)),
            atol=0.01,
            max_mismatch_pct=0.002,
            max_mismatched_abs_diff=1.5,
        )

    with patch(
        "helion._testing.import_example", return_value=SimpleNamespace(check=check)
    ):
        bwd, fwd = cases.collect("add", 0)
    assert (bwd.mode, bwd.check, bwd.rtol, bwd.atol) == (
        "backward",
        "relative_l2",
        0.02,
        0.0,
    )
    assert (fwd.check, fwd.rtol, fwd.atol) == ("mismatch_budget", 1e-2, 0.01)
    assert (fwd.max_mismatch_pct, fwd.max_mismatched_abs_diff) == (0.002, 1.5)


def test_relative_l2_check_uses_the_run_example_helper():
    expected = torch.ones(4, 4)
    case = cases.Case(
        "test", "00_forward", (expected,), {}, {}, check="relative_l2", rtol=0.3
    )
    # One element off by 1.0 is a 25% relative L2 error, within budget even
    # though an elementwise check at this rtol would reject it ...
    actual = expected.clone()
    actual[0, 0] = 2.0
    baselines.assert_correct(actual, expected, case)
    # ... and a larger miss is reported by helion._testing's shared helper, so
    # the catalog and run_example judge gradients identically.
    with pytest.raises(AssertionError, match="relative L2 error .* exceeds"):
        baselines.assert_correct(actual * 2, expected, case)


@pytest.mark.parametrize(
    "check", ["close", "relative_l2", "mismatch_budget", "dropout"]
)
def test_output_shape_and_dtype_are_checked_before_numeric_comparison(check):
    expected = torch.ones(2, 3, dtype=torch.bfloat16)
    case = cases.Case(
        "test",
        "00_forward",
        (0.0, expected, 123),
        {},
        {},
        check=check,
        max_mismatch_pct=0.01,
    )
    baselines.assert_correct(expected.clone(), expected, case)
    with pytest.raises(AssertionError):
        baselines.assert_correct(expected.float(), expected, case)
    with pytest.raises(AssertionError):
        baselines.assert_correct(expected.flatten(), expected, case)


def cpu_worker(monkeypatch, tmp_path, case, provider, impl, *, measure, batched, cold):
    """Run ``compare.worker`` on a CPU case with its CUDA timers replaced."""
    import helion.autotuner.metrics as metrics

    monkeypatch.setattr(cases, "collect", lambda module, shape: [case])
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU fixture")
    monkeypatch.setattr(metrics, "register_post_autotune_hook", lambda callback: None)
    monkeypatch.setattr(baselines, "active_kernel_configs", list)
    monkeypatch.setattr(baselines, "measure", measure)
    monkeypatch.setattr(baselines, "measure_batched", batched)
    monkeypatch.setattr(baselines, "measure_cold", cold)
    args = SimpleNamespace(
        module="test",
        shape=0,
        case="00_forward",
        provider=provider,
        impl=impl,
        seed=7,
        repetitions=1,
        output=tmp_path / f"{provider}-{impl}.json",
    )
    compare.worker(args)
    return json.loads(args.output.read_text())


def captured(fn, repetitions, expected, case, *, output_dtype):
    return {"graph_capturable": True, "graph_correctness": "pass", "graph_ms": 1.0}


def batched(fn, repetitions, calls, expected, case, *, output_dtype):
    return {"graph_calls_ms": 1.0, "graph_long_ms": 1.0, "graph_long_calls": calls}


def cold(fn, repetitions):
    return {"bench_ms": 1.0}


@torch.no_grad()
def test_worker_records_and_enforces_the_selected_output_contract(
    monkeypatch, tmp_path
):
    case = cases.Case(
        "test",
        "00_forward",
        (torch.ones(2, 3, dtype=torch.bfloat16),),
        {"helion": lambda value: value.clone()},
        {"aten": lambda value: value.float()},
        output_dtypes={"cute:helion": torch.bfloat16},
        reference_note="Test fixture: FP32 reference with BF16 kernel output.",
    )

    def measure(fn, repetitions, expected, case, *, output_dtype):
        # Exercise the propagated contract; no CUDA timer runs in this CPU test.
        assert output_dtype == torch.bfloat16
        baselines.assert_correct(fn(), expected, case, output_dtype=output_dtype)
        return captured(fn, repetitions, expected, case, output_dtype=output_dtype)

    def measure_batched(fn, repetitions, calls, expected, case, *, output_dtype):
        assert output_dtype == torch.bfloat16
        return batched(
            fn, repetitions, calls, expected, case, output_dtype=output_dtype
        )

    result = cpu_worker(
        monkeypatch,
        tmp_path,
        case,
        "cute",
        "helion",
        measure=measure,
        batched=measure_batched,
        cold=cold,
    )
    assert result["status"] == "ok", result.get("error")
    assert result["output_contract"] == {
        "shape": [2, 3],
        "dtype": "torch.bfloat16",
    }
    assert result["reference_output"]["dtype"] == "torch.float32"
    assert result["output"]["dtype"] == "torch.bfloat16"
    assert result["post_measurement_output"] == result["output"]
    assert result["reference_note"] == case.reference_note

    # The declaration belongs only to cute:helion, not every implementation.
    rejected = cpu_worker(
        monkeypatch,
        tmp_path,
        case,
        "triton",
        "helion",
        measure=measure,
        batched=measure_batched,
        cold=cold,
    )
    assert rejected["status"] == "error"
    assert "eager_correctness" not in rejected
    assert rejected["output"]["dtype"] == "torch.bfloat16"
    assert rejected["output_contract"]["dtype"] == "torch.float32"


@torch.no_grad()
def test_worker_judges_an_implementation_by_its_own_tolerance(monkeypatch, tmp_path):
    # A baseline the catalogue allows a looser bound (``tolerances`` keyed
    # like ``output_dtypes``) is checked at that bound and records it; the
    # Helion kernel off by the same amount is held to the case's bound.
    case = cases.Case(
        "test",
        "00_forward",
        (torch.zeros(4),),
        {"helion": lambda value: value + 0.25},
        {"aten": lambda value: value.clone(), "loose": lambda value: value + 0.25},
        rtol=0.0,
        atol=0.1,
        tolerances={"baselines:loose": (0.0, 0.5)},
        auto_compile=False,
    )

    def measure(fn, repetitions, expected, case, *, output_dtype):
        baselines.assert_correct(fn(), expected, case, output_dtype=output_dtype)
        return captured(fn, repetitions, expected, case, output_dtype=output_dtype)

    loose = cpu_worker(
        monkeypatch,
        tmp_path,
        case,
        "baselines",
        "loose",
        measure=measure,
        batched=batched,
        cold=cold,
    )
    assert loose["status"] == "ok", loose.get("error")
    assert (loose["rtol"], loose["atol"]) == (0.0, 0.5)
    strict = cpu_worker(
        monkeypatch,
        tmp_path,
        case,
        "cute",
        "helion",
        measure=measure,
        batched=batched,
        cold=cold,
    )
    assert strict["status"] == "error" and "eager_correctness" not in strict
    assert (strict["rtol"], strict["atol"]) == (0.0, 0.1)


@torch.no_grad()
def test_worker_keeps_the_eager_timing_of_a_host_syncing_baseline(
    monkeypatch, tmp_path
):
    case = cases.Case(
        "test",
        "00_forward",
        (torch.ones(2, 3),),
        {"helion": lambda value: value.clone()},
        {"aten": lambda value: value.clone()},
        auto_compile=False,
    )

    def uncaptured(fn, repetitions, expected, case, *, output_dtype):
        return {
            "event_ms": 0.7,
            "graph_capturable": False,
            "graph_ms": None,
            "graph_error": "RuntimeError: operation failed during capture",
        }

    def never(*args, **kwargs):
        raise AssertionError("an uncapturable call must not be captured again")

    result = cpu_worker(
        monkeypatch,
        tmp_path,
        case,
        "baselines",
        "aten",
        measure=uncaptured,
        batched=never,
        cold=never,
    )
    assert result["status"] == "ok", result.get("error")
    assert result["graph_capturable"] is False
    assert result["graph_ms"] is None and result["event_ms"] == 0.7
    assert "graph_calls_ms" not in result and "bench_ms" not in result
    assert result["eager_correctness"] == "pass"
    assert result["post_measurement_correctness"] == "pass"

    # A Helion kernel that cannot be captured is a failure, not an eager-only
    # record.
    def failed(*args, **kwargs):
        return {"graph_calls_error": "RuntimeError: capture", "bench_error": "same"}

    result = cpu_worker(
        monkeypatch,
        tmp_path,
        case,
        "cute",
        "helion",
        measure=uncaptured,
        batched=failed,
        cold=failed,
    )
    assert result["status"] == "error"
    assert "CUDA graph measurement or correctness failed" in result["error"]
    assert result["graph_capturable"] is False


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ImportError("No module named 'quack'"), "baseline_unavailable"),
        (
            torch._dynamo.exc.Unsupported("Dynamic control flow is not supported"),
            "baseline_unavailable",
        ),
        (
            torch._dynamo.exc.UserError(
                torch._dynamo.exc.UserErrorType.DYNAMIC_CONTROL_FLOW,
                "Could not guard on data-dependent expression u1 > u0",
            ),
            "baseline_unavailable",
        ),
        (
            RuntimeError("No available kernel. Aborting execution."),
            "baseline_unavailable",
        ),
        (RuntimeError("CUDA error: an illegal memory access"), "error"),
        (AssertionError("Tensor-likes are not close!"), "error"),
    ],
)
@torch.no_grad()
def test_worker_marks_baselines_this_machine_cannot_provide(
    monkeypatch, tmp_path, error, status
):
    def failing(value):
        raise error

    case = cases.Case(
        "test",
        "00_forward",
        (torch.ones(2),),
        {"helion": failing},
        {"aten": lambda value: value.clone(), "quack": failing},
        auto_compile=False,
    )
    result = cpu_worker(
        monkeypatch,
        tmp_path,
        case,
        "baselines",
        "quack",
        measure=captured,
        batched=batched,
        cold=cold,
    )
    assert result["status"] == status
    assert type(error).__name__ in result["error"]
    # The same exception from a Helion kernel, or from the reference every
    # implementation is checked against, is a failure of the cell.
    result = cpu_worker(
        monkeypatch,
        tmp_path,
        case,
        "cute",
        "helion",
        measure=captured,
        batched=batched,
        cold=cold,
    )
    assert result["status"] == "error"
    case.baselines = {"aten": failing}
    result = cpu_worker(
        monkeypatch,
        tmp_path,
        case,
        "baselines",
        "aten",
        measure=captured,
        batched=batched,
        cold=cold,
    )
    assert result["status"] == "error"


@torch.enable_grad()
def test_grpo_reference_promotion_does_not_waive_output_dtypes(monkeypatch):
    # Place the original catalogue inputs on CPU; do not replace its reference
    # or invoke either Helion kernel in this test.
    monkeypatch.setattr(
        cases,
        "rand",
        lambda *shape, dtype=torch.float32, grad=False: torch.randn(
            *shape, dtype=dtype, requires_grad=grad
        ),
    )
    for name in ("ones", "randint"):
        original = getattr(torch, name)
        monkeypatch.setattr(
            torch,
            name,
            lambda *args, original=original, **kwargs: original(
                *args, **{**kwargs, "device": torch.device("cpu")}
            ),
        )
    forward, backward = cases.collect("grpo_loss", 0)
    expected = baselines.make_callable(
        next(iter(forward.baselines.values())), forward
    )()
    assert all(value.dtype == torch.float32 for value in expected)
    baselines.assert_correct(expected, expected, forward)
    with pytest.raises(AssertionError):
        baselines.assert_correct(
            tuple(value.bfloat16() for value in expected), expected, forward
        )
    (gradient,) = baselines.make_callable(
        next(iter(backward.baselines.values())), backward
    )()
    assert gradient.dtype == torch.bfloat16
    assert "Only reference logits" in forward.reference_note
    assert backward.reference_note == forward.reference_note
    assert forward.output_dtypes == backward.output_dtypes == {}


def test_worker_environment_removes_inherited_budget_and_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HELION_AUTOTUNE_BUDGET_SECONDS", "1")
    monkeypatch.setenv("HELION_AUTOTUNE_CONFIG_OVERRIDES", '{"block_sizes": [1]}')
    monkeypatch.setenv("HELION_AUTOTUNE_EFFORT", "none")
    monkeypatch.delenv("HELION_BENCHMARK_GRAPH_WARMUP", raising=False)
    env = compare.worker_environment("2", "cute", tmp_path, 1234)
    assert env["HELION_AUTOTUNE_EFFORT"] == "full"
    assert env["HELION_FORCE_AUTOTUNE"] == "1"
    assert env["HELION_AUTOTUNE_RANDOM_SEED"] == "1234"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert "HELION_AUTOTUNE_BUDGET_SECONDS" not in env
    assert "HELION_AUTOTUNE_CONFIG_OVERRIDES" not in env
    assert "HELION_BENCHMARK_GRAPH_WARMUP" not in env
    assert env["HELION_CACHE_DIR"] == str(tmp_path / "helion")
    assert compare.graph_state_regime().startswith("steady:")
    # The documented opt-out of the graph dispatch warm-up is the one HELION_*
    # variable the workers inherit, and the run record names the regime.
    monkeypatch.setenv("HELION_BENCHMARK_GRAPH_WARMUP", "0")
    env = compare.worker_environment("2", "cute", tmp_path, 1234)
    assert env["HELION_BENCHMARK_GRAPH_WARMUP"] == "0"
    assert compare.graph_state_regime().startswith("fast:")
    monkeypatch.setenv("HELION_BENCHMARK_GRAPH_WARMUP", "maybe")
    with pytest.raises(ValueError, match="HELION_BENCHMARK_GRAPH_WARMUP"):
        compare.graph_state_regime()  # the driver fails fast, with the name


def test_graph_state_warmup_reports_the_first_timed_readings_state(monkeypatch):
    # The record field describes the state of the process's first timed
    # reading; a warm-up that failed first and succeeded before a later
    # measurement is reported as such rather than overwritten.
    from helion.autotuner import benchmarking

    notes = iter(["failed: RuntimeError: busy", benchmarking._GRAPH_WARMUP_NOTE])
    monkeypatch.setattr(benchmarking, "steady_graph_dispatch", lambda: next(notes))
    monkeypatch.setattr(baselines, "_graph_state_note", None)
    assert baselines.graph_state_warmup() == "failed: RuntimeError: busy"
    assert baselines.graph_state_warmup() == (
        f"failed: RuntimeError: busy; then {benchmarking._GRAPH_WARMUP_NOTE}"
    )
    monkeypatch.setattr(
        benchmarking, "steady_graph_dispatch", lambda: benchmarking._GRAPH_WARMUP_NOTE
    )
    assert baselines.graph_state_warmup() == (
        f"failed: RuntimeError: busy; then {benchmarking._GRAPH_WARMUP_NOTE}"
    )
    monkeypatch.setattr(baselines, "_graph_state_note", None)
    assert baselines.graph_state_warmup() == benchmarking._GRAPH_WARMUP_NOTE
    assert baselines.graph_state_warmup() == benchmarking._GRAPH_WARMUP_NOTE


def test_pins_match_a_kernel_recorded_with_a_duplicate_suffix(tmp_path):
    # ``name@2`` marks a second kernel built from one function; which of the
    # two gets the suffix depends on the collector's order, so a pin read
    # from a record that named ``name@2`` must still apply.
    from benchmarks.cute import example_case_bench as case_bench

    import helion

    record = tmp_path / "record.json"
    record.write_text(
        json.dumps(
            {
                "bound_kernels": [
                    {
                        "kernel": "examples.attention.attention@2",
                        "configs": ['{"block_sizes": [16]}', None],
                    }
                ]
            }
        )
    )
    kernels = {
        name: SimpleNamespace(configs=None)
        for name in (
            "examples.attention.attention",
            "examples.attention.attention@2",
            "examples.attention.other",
        )
    }
    case_bench.pin_configs(SimpleNamespace(pin_from=record, pin=[]), kernels)
    for name in ("examples.attention.attention", "examples.attention.attention@2"):
        assert kernels[name].configs == [helion.Config(block_sizes=[16])]
    assert kernels["examples.attention.other"].configs is None
    with pytest.raises(ValueError, match="matches nothing"):
        case_bench.pin_configs(
            SimpleNamespace(pin_from=None, pin=['missing@2={"block_sizes": [8]}']),
            kernels,
        )


def fixture_records(providers=compare.PROVIDERS):
    """A complete run of ``providers`` over three shapes: cute 1.0 ms, triton
    2.0 ms, baselines 1.5 ms on every shape."""
    run = {
        "schema": compare.SCHEMA,
        "status": "complete",
        "modules": ["add"],
        "providers": list(providers),
        "shapes": [0, 1, 2],
    }
    records = []
    for shape in run["shapes"]:
        records.append(
            {
                "module": "add",
                "shape": shape,
                "provider": "inventory",
                "status": "ok",
                "cases": [{"name": "00_forward"}],
            }
        )
        for provider, latency in (("cute", 1.0), ("triton", 2.0), ("baselines", 1.5)):
            if provider not in providers:
                continue
            records.append(
                {
                    "module": "add",
                    "case": "00_forward",
                    "shape": shape,
                    "provider": provider,
                    "impl": provider,
                    "status": "ok",
                    "graph_ms": latency,
                    "graph_state_warmup": "16 eager launches on side streams",
                    "eager_correctness": "pass",
                    "graph_correctness": "pass",
                    "post_measurement_correctness": "pass",
                }
            )
    return run, records


def test_report_names_the_graph_dispatch_state_and_marks_disagreement():
    from benchmarks.cute.example_report import graph_state_lines

    run, records = fixture_records()
    data = aggregate(run, records)
    assert data["graph_states"] == {"16 eager launches on side streams": 9}
    (line,) = graph_state_lines(data["graph_states"])
    assert line.startswith("Graph dispatch state of all 9 measured records: 16 eager")
    # A record from before the field existed, and one whose warm-up failed:
    # the run's readings are no longer all comparable, and the report says so
    # instead of averaging across regimes.
    del records[1]["graph_state_warmup"]
    records[2]["graph_state_warmup"] = "failed: RuntimeError: busy"
    data = aggregate(run, records)
    assert data["graph_states"] == {
        "16 eager launches on side streams": 7,
        "failed: RuntimeError: busy": 1,
        "unrecorded": 1,
    }
    (line,) = graph_state_lines(data["graph_states"])
    assert line.startswith("WARNING: the measured records disagree")
    assert "unrecorded (1)" in line and "failed: RuntimeError: busy (1)" in line


def test_higher_is_better_and_incomplete_shapes_are_not_dropped():
    run, records = fixture_records()
    variant = aggregate(run, records)["variants"][0]
    assert variant["cute_vs_triton"] == pytest.approx(2)
    assert variant["cute_vs_best"] == pytest.approx(1.5)
    records[-1].update(status="error", error="missing optional package")
    result = aggregate(run, records)
    assert result["variants"][0]["status"] == "incomplete"
    assert result["variants"][0]["valid_shapes"] == 2
    assert result["failures"] == [records[-1]]


def test_eager_only_baselines_are_listed_not_ranked():
    run, records = fixture_records()
    eager_only = {
        "module": "add",
        "case": "00_forward",
        "shape": 0,
        "provider": "baselines",
        "impl": "aten.loop",
        "status": "ok",
        "graph_capturable": False,
        "graph_ms": None,
        "event_ms": 0.9,
        "eager_correctness": "pass",
        "post_measurement_correctness": "pass",
    }
    data = aggregate(run, [*records, eager_only])
    assert data["uncapturable"] == [eager_only] and data["failures"] == []
    shape0 = data["shapes"][0]
    assert (shape0["baselines_ms"], shape0["baselines_impl"]) == (1.5, "baselines")
    assert data["variants"][0]["status"] == "ok"
    # Helion kernels are never eager-only, and an eager-only record still has
    # to have passed its eager checks.
    with pytest.raises(ValueError, match="not captured"):
        aggregate(run, [*records, {**eager_only, "provider": "cute", "impl": "h2"}])
    with pytest.raises(ValueError, match="missing correctness"):
        aggregate(run, [*records, {**eager_only, "eager_correctness": "failed"}])


def test_completeness_and_speedups_follow_the_runs_providers():
    # A run restricted to ``--providers cute,baselines`` has no Triton
    # records by design: its shape rows are complete and ranked against the
    # baselines, and the Triton speedup is not reported rather than N/A.
    run, records = fixture_records(["cute", "baselines"])
    data = aggregate(run, records)
    assert data["ratios"] == ["cute_vs_best"]
    for row in data["shapes"]:
        assert row["status"] == "ok" and "triton_ms" not in row
        assert row["cute_vs_best"] == pytest.approx(1.5)
        assert "cute_vs_triton" not in row
    (variant,) = data["variants"]
    assert variant["status"] == "ok" and variant["valid_shapes"] == 3
    assert variant["cute_vs_best"] == pytest.approx(1.5)
    assert "cute_vs_triton" not in variant
    # A record from a provider the run did not request is a corrupt run.
    _, (_, triton, *_) = fixture_records(["triton"])
    with pytest.raises(ValueError, match="did not request"):
        aggregate(run, [*records, triton])
    # Without baselines the best compared implementation is Triton's.
    run, records = fixture_records(["cute", "triton"])
    (variant,) = aggregate(run, records)["variants"]
    assert (variant["cute_vs_triton"], variant["cute_vs_best"]) == (2.0, 2.0)
    # A run without CuTe compares nothing against CuTe: rows are complete
    # and carry their latencies, no speedup is defined.
    run, records = fixture_records(["triton", "baselines"])
    data = aggregate(run, records)
    assert data["ratios"] == []
    assert all(row["status"] == "ok" for row in data["shapes"])
    assert data["variants"] == [
        {
            "variant": "add/00_forward",
            "valid_shapes": 3,
            "requested_shapes": 3,
            "status": "ok",
        }
    ]
    # A results.json from before the field compared exactly PROVIDERS, so it
    # reports as a run of them.
    run, records = fixture_records()
    expected = aggregate(run, records)
    del run["providers"]
    assert aggregate(run, records) == expected
    assert expected["ratios"] == ["cute_vs_triton", "cute_vs_best"]


@pytest.mark.parametrize("corruption", ["nan", "zero", "correctness", "duplicate"])
def test_invalid_results_are_rejected(corruption):
    run, records = fixture_records()
    if corruption == "nan":
        records[-1]["graph_ms"] = float("nan")
    elif corruption == "zero":
        records[-1]["graph_ms"] = 0
    elif corruption == "correctness":
        records[-1]["graph_correctness"] = "failed"
    else:
        records.append(copy.deepcopy(records[-1]))
    with pytest.raises(ValueError):
        aggregate(run, records)


@pytest.mark.parametrize(
    "providers",
    [compare.PROVIDERS, ("cute", "baselines"), ("baselines",)],
    ids=["all", "cute-baselines", "baselines-only"],
)
def test_report_reads_pinned_results_and_generates_matplotlib_chart(
    providers, tmp_path
):
    pytest.importorskip("matplotlib")
    run, records = fixture_records(providers)
    run["jobs"] = []
    for index, record in enumerate(records):
        path = tmp_path / f"{index}.json"
        content = json.dumps(record).encode()
        path.write_bytes(content)
        run["jobs"].append(
            {"path": path.name, "sha256": hashlib.sha256(content).hexdigest()}
        )
    source = tmp_path / "results.json"
    source.write_text(json.dumps(run))
    report_results(source, tmp_path / "report")
    report = (tmp_path / "report/report.md").read_text()
    measured = 3 * len(providers)
    assert (
        f"Graph dispatch state of all {measured} measured records: 16 eager" in report
    )
    assert "0 baseline jobs synchronize with the host" in report
    table = [line for line in report.splitlines() if line.startswith("| add/")]
    if "cute" in providers:
        # The chart and the table carry one column per reported speedup.
        png = tmp_path / "report/speedups.png"
        assert png.read_bytes().startswith(b"\x89PNG")
        assert "higher is better" in report
        assert ("vs Triton" in report) is ("triton" in providers)
        assert table == [
            "| add/00_forward | 3/3 | 2.000× | 1.500× |"
            if "triton" in providers
            else "| add/00_forward | 3/3 | 1.500× |"
        ]
    else:
        assert not (tmp_path / "report/speedups.png").exists()
        assert "no CuTe speedups are reported" in report
        assert table == ["| add/00_forward | 3/3 |"]
    assert json.loads((tmp_path / "report/uncapturable.json").read_text()) == []
    (tmp_path / "0.json").write_text("{}")
    with pytest.raises(ValueError, match="hash changed"):
        report_results(source, tmp_path / "invalid")


def test_failed_module_inventories_remain_visible():
    run, records = fixture_records()
    run["modules"].append("exp")
    records.extend(
        {
            "module": "exp",
            "shape": shape,
            "provider": "inventory",
            "status": "error",
            "error": "fixture unavailable",
        }
        for shape in run["shapes"]
    )
    result = aggregate(run, records)
    missing = next(
        row for row in result["variants"] if row["variant"].startswith("exp/")
    )
    assert missing["status"] == "incomplete" and missing["valid_shapes"] == 0
    assert len(result["inventory"]) == 6
    assert sum(row["status"] == "ok" for row in result["inventory"]) == 3
    assert len(result["failures"]) == 3


def test_provenance_records_power_limit_and_software(monkeypatch):
    monkeypatch.setattr(
        compare.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="GPU-123, NVIDIA B200, 750.00, 580.0\n"
        ),
    )
    monkeypatch.setattr(
        compare.importlib.metadata, "version", lambda package: f"{package}-test"
    )
    result = compare.provenance("2")
    assert result["gpu_properties"] == {
        "uuid": "GPU-123",
        "name": "NVIDIA B200",
        "power_limit_w": 750.0,
        "driver_version": "580.0",
    }
    assert result["software"]["torch"] == "torch-test"
    assert result["software"]["quack-kernels"] == "quack-kernels-test"


def test_run_restricts_jobs_to_requested_cases(monkeypatch, tmp_path):
    from benchmarks.cute import compare_attention_backends

    launched = []

    def fake_run(command, **kwargs):
        assert "worker" not in command
        return SimpleNamespace(returncode=1, stdout="")

    def fake_popen(command, **kwargs):
        option = dict(zip(command[3::2], command[4::2], strict=False))
        result = {
            "module": option["--module"],
            "shape": int(option["--shape"]),
            "case": option["--case"],
            "provider": option["--provider"],
            "impl": option["--impl"],
            "status": "ok",
        }
        if option["--provider"] == "inventory":
            result["cases"] = [
                {"name": name, "kernels": ["helion"], "baselines": ["aten"]}
                for name in ("00_forward", "01_backward")
            ]
        else:
            launched.append((option["--case"], option["--provider"]))
        Path(option["--output"]).write_text(json.dumps(result))
        assert kwargs["start_new_session"]
        return SimpleNamespace(returncode=0, pid=-1, wait=lambda: 0)

    reaped = []
    monkeypatch.setattr(compare.subprocess, "run", fake_run)
    monkeypatch.setattr(compare.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(compare.os, "killpg", lambda pid, sig: reaped.append(pid))
    monkeypatch.setattr(compare, "check_gpu_idle", lambda gpu, **kwargs: "GPU-0")
    monkeypatch.setattr(compare, "provenance", lambda gpu: {})
    monkeypatch.setattr(
        compare_attention_backends, "_git_source_snapshot", lambda *args: None
    )
    args = SimpleNamespace(
        modules="add",
        shapes="0",
        cases="01_backward",
        providers="",
        impls="",
        seed=1,
        gpu="0",
        repetitions=1,
        output=tmp_path / "run",
    )
    monkeypatch.delenv("HELION_BENCHMARK_GRAPH_WARMUP", raising=False)
    compare.run(args)
    record = json.loads((tmp_path / "run" / "results.json").read_text())
    assert record["status"] == "complete"
    assert record["cases"] == ["01_backward"]
    assert record["providers"] == ["cute", "triton", "baselines"]
    assert record["impls"] is None
    assert record["graph_state"].startswith("steady:")
    assert "graph_state" in record["timer"]
    assert sorted(launched) == [
        ("01_backward", "baselines"),
        ("01_backward", "cute"),
        ("01_backward", "triton"),
    ]
    # Every worker's session (inventory included) is reaped.
    assert len(reaped) == 4


def test_run_filters_jobs_and_marks_a_crashed_baseline_unavailable(
    monkeypatch, tmp_path
):
    from benchmarks.cute import compare_attention_backends

    launched = []
    inventory = {"status": "ok"}

    def fake_popen(command, **kwargs):
        option = dict(zip(command[3::2], command[4::2], strict=False))
        result = {
            "module": option["--module"],
            "shape": int(option["--shape"]),
            "case": option["--case"],
            "provider": option["--provider"],
            "impl": option["--impl"],
            "status": "ok",
        }
        returncode = 0
        if option["--provider"] == "inventory":
            result["status"] = inventory["status"]
            result["cases"] = [
                {
                    "name": "01_backward",
                    "kernels": ["helion", "helion_fast"],
                    "baselines": ["aten", "torch.compile"],
                }
            ]
        else:
            launched.append((option["--provider"], option["--impl"]))
            # A worker that dies writes no record.
            returncode = -11 if option["--impl"] in {"aten", "helion"} else 0
        if returncode == 0:
            Path(option["--output"]).write_text(json.dumps(result))
        return SimpleNamespace(returncode=returncode, pid=-1, wait=lambda: 0)

    monkeypatch.setattr(
        compare.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1)
    )
    monkeypatch.setattr(compare.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(compare.os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(compare, "check_gpu_idle", lambda gpu, **kwargs: "GPU-0")
    monkeypatch.setattr(compare, "provenance", lambda gpu: {})
    monkeypatch.setattr(
        compare_attention_backends, "_git_source_snapshot", lambda *args: None
    )
    monkeypatch.delenv("HELION_BENCHMARK_GRAPH_WARMUP", raising=False)
    args = SimpleNamespace(
        modules="add",
        shapes="0",
        cases="",
        providers="baselines,cute",
        impls="aten,helion,torch.compile",
        seed=1,
        gpu="0",
        repetitions=1,
        output=tmp_path / "run",
    )
    compare.run(args)
    record = json.loads((tmp_path / "run" / "results.json").read_text())
    assert record["providers"] == ["baselines", "cute"]
    assert record["impls"] == ["aten", "helion", "torch.compile"]
    assert sorted(launched) == [
        ("baselines", "aten"),
        ("baselines", "torch.compile"),
        ("cute", "helion"),
    ]
    statuses = {}
    for job in record["jobs"][1:]:
        row = json.loads((tmp_path / "run" / job["path"]).read_text())
        statuses[(row["provider"], row["impl"])] = (
            job["status"],
            row.get("returncode"),
        )
    # A dead baseline worker is a baseline this machine cannot provide; a
    # dead Helion worker is a failure.
    assert statuses == {
        ("baselines", "aten"): ("baseline_unavailable", -11),
        ("baselines", "torch.compile"): ("ok", None),
        ("cute", "helion"): ("process_error", -11),
    }
    args.providers, args.output = "quack", tmp_path / "rejected"
    with pytest.raises(ValueError, match="--providers"):
        compare.run(args)
    # An implementation name no requested case offers is refused after the
    # inventories, before any job runs.
    args.providers, args.impls = "baselines,cute", "aten,helion_fastest"
    args.output = tmp_path / "rejected_impl"
    launched.clear()
    with pytest.raises(ValueError, match=r"--impls .*\['helion_fastest'\]"):
        compare.run(args)
    assert launched == []
    # So is a name the requested providers never draw: a baseline under
    # `--providers cute` would otherwise pass validation and launch nothing.
    args.providers, args.impls = "cute", "aten"
    args.output = tmp_path / "rejected_provider_impl"
    with pytest.raises(ValueError, match=r"--impls .*\['aten'\]"):
        compare.run(args)
    assert launched == []
    # With every inventory failed there is nothing to check --impls against;
    # the run completes with the inventory failures recorded.
    inventory["status"] = "error"
    args.output = tmp_path / "no_inventory"
    compare.run(args)
    record = json.loads((args.output / "results.json").read_text())
    assert record["status"] == "complete"
    assert [job["status"] for job in record["jobs"]] == ["error"]
    assert launched == []


def test_gpu_idle_check_waits_out_a_departing_process(monkeypatch):
    apps = iter(["GPU-0, 123\n", ""])

    def fake_run(command, **kwargs):
        if "--query-gpu=uuid" in command:
            return SimpleNamespace(stdout="GPU-0\n")
        return SimpleNamespace(stdout=next(apps))

    monkeypatch.setattr(compare.subprocess, "run", fake_run)
    monkeypatch.setattr(compare.time, "sleep", lambda seconds: None)
    assert compare.check_gpu_idle("0", grace_seconds=30.0) == "GPU-0"
    apps = iter(["GPU-0, 123\n"] * 2)
    with pytest.raises(RuntimeError, match="busy"):
        compare.check_gpu_idle("0", grace_seconds=5.0)
