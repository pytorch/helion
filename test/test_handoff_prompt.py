from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import pytest

from helion.autotuner.handoff_prompt import _evaluation_summary
from helion.autotuner.handoff_prompt import build_handoff_prompt


def _evaluation(perf: float = 1.15) -> dict[str, Any]:
    return {
        "ok": True,
        "perf": perf,
        "unit": "ms",
        "directory": "/records/EVALUATION_PROVENANCE",
        "created_at": "RECORDED_TIMESTAMP",
        "cases": [
            {
                "index": 0,
                "status": "ok",
                "perf": perf,
                "noise": 0.05,
                "error": None,
                "samples": [perf - 0.05, perf, perf + 0.05],
                "source_hash": "SOURCE_PROVENANCE",
                "source_path": "/records/SOURCE_PROVENANCE.py",
                "progress_path": "/records/PROGRESS_PROVENANCE.jsonl",
                "phase": "complete",
                "timing": "cuda_graph",
            }
        ],
    }


@pytest.fixture
def manifest() -> dict[str, Any]:
    # No autotune payload is needed to describe the native task.
    return {
        "schema_version": 1,
        "created_at": "MANIFEST_TIMESTAMP",
        "objective": {
            "aggregation": "geomean",
            "unit": "ms",
            "reference_latencies": None,
        },
        "cases": [
            {
                "source": "case_0/kernel.py",
                "entrypoint": "attention_output",
                "backend": "cute",
                "hardware": {"device": "test accelerator", "compute_units": 100},
                "input_metadata": [
                    {
                        "type": "Tensor",
                        "shape": [1, 2, 1024, 128],
                        "dtype": "float16",
                        "stride": [262144, 131072, 128, 1],
                        "storage_offset": 0,
                        "storage_group": 0,
                    }
                ],
                "dependencies": ["torch", "cutlass"],
                "static_shapes": True,
                "reference_kind": "REFERENCE_PROVENANCE",
                "timing": "cuda_graph",
                "atol": 0.01,
                "rtol": 0.02,
                "scale_atol": True,
                "inputs": "case_0/INPUT_PROVENANCE.pt",
                "reference": "case_0/REFERENCE_PROVENANCE.pt",
            }
        ],
        "baseline_evaluation": _evaluation(),
    }


def _json_after(prompt: str, heading: str) -> Any:
    # The case list has a prose suffix; decode just the first JSON value.
    return json.JSONDecoder().raw_decode(prompt.split(heading + "\n", 1)[1])[0]


@pytest.mark.parametrize(
    ("backend", "label"),
    (("cute", "CuTe DSL (CUTLASS)"), ("triton", "Triton"), ("custom", "custom")),
)
def test_native_case_contracts_and_backend(
    manifest: dict[str, Any], backend: str, label: str
) -> None:
    case = manifest["cases"][0]
    case["backend"] = backend
    prompt = build_handoff_prompt(manifest)
    (contract,) = _json_after(prompt, "Cases:")
    assert contract["backend"] == label
    for key in (
        "source",
        "entrypoint",
        "hardware",
        "input_metadata",
        "dependencies",
        "static_shapes",
        "timing",
        "atol",
        "rtol",
        "scale_atol",
    ):
        assert contract[key] == case[key]
    assert {"inputs", "reference", "reference_kind"}.isdisjoint(contract)
    for instruction in ("output structure", "input mutations", "aliases"):
        assert instruction in prompt
    assert "caller checks correctness" in prompt
    assert "manifest.json" not in prompt
    assert "evaluate.py" not in prompt


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize(
    ("aggregation", "label"), [("geomean", "geometric mean"), ("max", "maximum")]
)
def test_latency_objective_omits_unused_reference_data(
    manifest: dict[str, Any], count: int, aggregation: str, label: str
) -> None:
    manifest["objective"] = {
        "aggregation": aggregation,
        "unit": "ms",
        "reference_latencies": ["UNUSED_REFERENCE_DATA"],
    }
    if count == 2:
        second = deepcopy(manifest["cases"][0])
        second["source"] = "case_1/kernel.py"
        manifest["cases"].append(second)
    prompt = build_handoff_prompt(manifest)
    assert "(ms)" in prompt
    assert "reference-latency" not in prompt
    assert "UNUSED_REFERENCE_DATA" not in prompt
    assert len(_json_after(prompt, "Cases:")) == count
    if count == 1:
        assert "kernel latency" in prompt
        assert label not in prompt
    else:
        assert label in prompt
        assert "per-case latencies" in prompt
    manifest["objective"].pop("reference_latencies")
    assert build_handoff_prompt(manifest) == prompt


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize(
    ("aggregation", "label"), [("geomean", "geometric mean"), ("max", "maximum")]
)
def test_ratio_objective_keeps_case_order_and_latency_units(
    manifest: dict[str, Any], count: int, aggregation: str, label: str
) -> None:
    references = [2.0, 4.0][:count]
    manifest["objective"] = {
        "aggregation": aggregation,
        "unit": "ratio",
        "reference_latencies": references,
    }
    if count == 2:
        second = deepcopy(manifest["cases"][0])
        second.update(source="case_1/kernel.py", timing="wall_clock")
        manifest["cases"].append(second)
    evaluation = manifest["baseline_evaluation"]
    evaluation.update(unit="ratio", perf=0.5)
    evaluation["cases"][0].update(perf=1.0, noise=0.02)
    if count == 2:
        other = deepcopy(evaluation["cases"][0])
        other.update(index=1, perf=2.0, noise=0.03)
        evaluation["cases"].append(other)
    prompt = build_handoff_prompt(manifest)
    assert label in prompt
    assert "latency/reference-latency ratios" in prompt
    assert "Reference latencies (ms, in case order): " in prompt
    assert (
        json.loads(prompt.split("in case order): ", 1)[1].split(".\n\n", 1)[0])
        == references
    )
    assert "Native baseline" not in prompt
    baseline = _evaluation_summary(evaluation)
    assert baseline["unit"] == "ratio"
    assert baseline["perf"] == 0.5
    assert [
        (case["index"], case["perf"], case["noise"]) for case in baseline["cases"]
    ] == [(0, 1.0, 0.02), (1, 2.0, 0.03)][:count]
    assert [case["timing"] for case in _json_after(prompt, "Cases:")] == [
        "cuda_graph",
        "wall_clock",
    ][:count]


@pytest.mark.parametrize("aliases", [[0, 1, 2, 3], [0, 1, 0], [], None])
def test_alias_contract_has_explicit_storage_semantics(
    manifest: dict[str, Any], aliases: list[int] | None
) -> None:
    manifest["cases"][0]["alias_signature"] = aliases
    prompt = build_handoff_prompt(manifest)
    (case,) = _json_after(prompt, "Cases:")
    if aliases is None:
        assert "alias_signature" not in prompt
    else:
        assert case["alias_signature"] == aliases
        assert "output tensors followed by input tensors" in prompt
        assert "equal IDs share storage" in prompt
        assert "different IDs do not" in prompt
        assert "local to each signature" in prompt


def test_prompt_ignores_autotune_and_baseline_records_without_mutation(
    manifest: dict[str, Any],
) -> None:
    manifest.pop("baseline_evaluation")
    expected = build_handoff_prompt(manifest)
    assert "autotune" not in manifest
    manifest["autotune"] = {
        "config": {"block_sizes": "SEARCH_CONFIG_PAYLOAD"},
        "measurements": ["SEARCH_HISTORY_PAYLOAD" * 1000] * 507,
        # Deliberately not JSON serializable; irrelevant data must not be read.
        "finalists": {frozenset({"SEARCH_FINALIST_PAYLOAD"})},
        "progress": {"trials": 987654321},
    }
    manifest["baseline_evaluation"] = _evaluation(98765.4321)
    manifest["unrelated_metadata"] = "UNRELATED_PAYLOAD"
    before = deepcopy(manifest)
    prompt = build_handoff_prompt(manifest)
    assert prompt == expected
    assert manifest == before
    for omitted in (
        "autotune",
        "config",
        "finalist",
        "SEARCH_",
        "UNRELATED_",
        "987654321",
        "98765.4321",
        "Native baseline",
        "Round feedback",
        '"perf"',
        '"noise"',
        '"samples"',
    ):
        assert omitted not in prompt


def test_submission_evaluation_summary_keeps_results_without_record_payloads(
    manifest: dict[str, Any],
) -> None:
    before = deepcopy(manifest)
    baseline = _evaluation_summary(manifest["baseline_evaluation"])
    assert baseline["ok"] is True
    assert baseline["unit"] == "ms"
    assert baseline["perf"] == 1.15
    assert baseline["cases"][0] == {
        "index": 0,
        "status": "ok",
        "perf": 1.15,
        "noise": 0.05,
    }
    serialized = json.dumps(baseline)
    for omitted in (
        "samples",
        "PROVENANCE",
        "TIMESTAMP",
        "source_hash",
        "progress_path",
    ):
        assert omitted not in serialized
    assert manifest == before


def test_failed_submission_evaluation_preserves_actual_error(
    manifest: dict[str, Any],
) -> None:
    error = "RuntimeError: /tmp/kernel.py:17: invalid reduction"
    evaluation = manifest["baseline_evaluation"]
    evaluation.update(ok=False, perf=None)
    evaluation["cases"][0].update(status="error", perf=None, noise=None, error=error)
    baseline = _evaluation_summary(evaluation)
    assert baseline["ok"] is False
    assert baseline["perf"] is None
    assert baseline["unit"] == "ms"
    assert baseline["cases"][0] == {"index": 0, "status": "error", "error": error}
