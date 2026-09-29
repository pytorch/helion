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
    # This shared engine is exercised by its eight public linear examples.
    assert kernel_modules - set(cases.MODULES) == {"linear.linear_attention_engine"}
    assert len(cases.MODULES) == len(set(cases.MODULES))
    assert all(len(shapes) == 3 for _, shapes in cases.CHECKS.values())


def test_collection_restores_example_recorder():
    import helion._testing as testing

    original = testing.run_example

    def check(rows, cols):
        return testing.run_example(torch.add, torch.add, (torch.ones(2), torch.ones(2)))

    with patch.object(
        cases.importlib, "import_module", return_value=SimpleNamespace(check=check)
    ):
        result = cases.collect("add", 0)
    assert testing.run_example is original
    assert result[0].name == "00_forward"
    assert result[0].kernels == {"helion": torch.add}

    def broken(rows, cols):
        raise ValueError("fixture failed")

    with (
        patch.object(
            cases.importlib, "import_module", return_value=SimpleNamespace(check=broken)
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


def test_backward_callable_uses_identical_upstream_gradients():
    x = torch.randn(3, 4, requires_grad=True)
    case = cases.Case("test", "01_backward", (x,), {}, {}, mode="backward")
    direct = baselines.make_callable(lambda value: value.square(), case)
    different_layout = baselines.make_callable(
        lambda value: value.square().t().contiguous().t(), case
    )
    torch.testing.assert_close(direct()[0], different_layout()[0])


@pytest.mark.parametrize("check", ["close", "relative_l2", "dropout"])
def test_output_shape_and_dtype_are_checked_before_numeric_comparison(check):
    expected = torch.ones(2, 3, dtype=torch.bfloat16)
    case = cases.Case("test", "00_forward", (0.0, expected, 123), {}, {}, check=check)
    baselines.assert_correct(expected.clone(), expected, case)
    with pytest.raises(AssertionError):
        baselines.assert_correct(expected.float(), expected, case)
    with pytest.raises(AssertionError):
        baselines.assert_correct(expected.flatten(), expected, case)


@torch.no_grad()
def test_worker_records_and_enforces_the_selected_output_contract(
    monkeypatch, tmp_path
):
    import helion.autotuner.metrics as metrics

    case = cases.Case(
        "test",
        "00_forward",
        (torch.ones(2, 3, dtype=torch.bfloat16),),
        {"helion": lambda value: value.clone()},
        {"aten": lambda value: value.float()},
        output_dtypes={"cute:helion": torch.bfloat16},
        reference_note="Test fixture: FP32 reference with BF16 kernel output.",
    )
    monkeypatch.setattr(cases, "collect", lambda module, shape: [case])
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU fixture")
    monkeypatch.setattr(metrics, "register_post_autotune_hook", lambda callback: None)
    monkeypatch.setattr(baselines, "active_kernel_configs", list)

    def measure(fn, repetitions, expected, case, *, output_dtype):
        # Exercise the propagated contract; no CUDA timer runs in this CPU test.
        assert output_dtype == torch.bfloat16
        baselines.assert_correct(fn(), expected, case, output_dtype=output_dtype)
        return {"graph_correctness": "pass", "graph_ms": 1.0}

    monkeypatch.setattr(baselines, "measure", measure)
    args = SimpleNamespace(
        module="test",
        shape=0,
        case="00_forward",
        provider="cute",
        impl="helion",
        seed=7,
        repetitions=1,
        output=tmp_path / "matched.json",
    )
    compare.worker(args)
    result = json.loads(args.output.read_text())
    assert result["status"] == "ok"
    assert result["output_contract"] == {
        "shape": [2, 3],
        "dtype": "torch.bfloat16",
    }
    assert result["reference_output"]["dtype"] == "torch.float32"
    assert result["output"]["dtype"] == "torch.bfloat16"
    assert result["post_measurement_output"] == result["output"]
    assert result["reference_note"] == case.reference_note

    # The declaration belongs only to cute:helion, not every implementation.
    args.provider = "triton"
    args.output = tmp_path / "undeclared.json"
    compare.worker(args)
    rejected = json.loads(args.output.read_text())
    assert rejected["status"] == "error"
    assert "eager_correctness" not in rejected
    assert rejected["output"]["dtype"] == "torch.bfloat16"
    assert rejected["output_contract"]["dtype"] == "torch.float32"


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
    env = compare.worker_environment("2", "cute", tmp_path, 1234)
    assert env["HELION_AUTOTUNE_EFFORT"] == "full"
    assert env["HELION_FORCE_AUTOTUNE"] == "1"
    assert env["HELION_AUTOTUNE_RANDOM_SEED"] == "1234"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert "HELION_AUTOTUNE_BUDGET_SECONDS" not in env
    assert "HELION_AUTOTUNE_CONFIG_OVERRIDES" not in env
    assert env["HELION_CACHE_DIR"] == str(tmp_path / "helion")


def fixture_records():
    run = {
        "schema": compare.SCHEMA,
        "status": "complete",
        "modules": ["add"],
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
            records.append(
                {
                    "module": "add",
                    "case": "00_forward",
                    "shape": shape,
                    "provider": provider,
                    "impl": provider,
                    "status": "ok",
                    "graph_ms": latency,
                    "eager_correctness": "pass",
                    "graph_correctness": "pass",
                    "post_measurement_correctness": "pass",
                }
            )
    return run, records


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


def test_report_reads_pinned_results_and_generates_matplotlib_chart(tmp_path):
    pytest.importorskip("matplotlib")
    run, records = fixture_records()
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
    assert (tmp_path / "report/speedups.png").read_bytes().startswith(b"\x89PNG")
    assert "higher is better" in (tmp_path / "report/report.md").read_text()
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
