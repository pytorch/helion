from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import TYPE_CHECKING

from benchmarks.cute.paper_attention_repro import protocol
from benchmarks.cute.paper_attention_repro import run
from benchmarks.cute.paper_attention_repro import worker
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def profile() -> dict:
    return protocol.read_json(run.HERE / "reference_cases.json")


def test_full_profile_and_gpu_assignment(profile: dict) -> None:
    cases = protocol.select_cases(profile, "all", "0,1,2")
    assert len(cases) == 87
    assert sum(case["batch"] == "fused" for case in cases) == 33
    assert sum(case["batch"] == "sparse" for case in cases) == 54
    assert {case["shape"]["dtype"] for case in cases} == {"bfloat16"}
    gpu_by_variant: dict[str, set[int]] = {}
    for gpu, gpu_cases in protocol.shard_cases(cases, [0, 1, 3]).items():
        for case in gpu_cases:
            gpu_by_variant.setdefault(case["variant"], set()).add(gpu)
    assert len(gpu_by_variant) == 29
    assert all(len(gpus) == 1 for gpus in gpu_by_variant.values())


def test_forward_contracts_and_sdpa_preparation(profile: dict) -> None:
    fused = protocol.select_cases(profile, "sliding-window", "0")[0]
    sparse = protocol.select_cases(profile, "sta2d", "0")[0]
    assert (
        protocol.contract(fused, "forward", "helion-cute")["output"]
        == "attention output"
    )
    assert "LSE" in protocol.contract(sparse, "forward", "helion-cute")["output"]
    assert (
        protocol.contract(fused, "forward", "sdpa")["sdpa_mask_or_bias"]
        == "outside timing"
    )
    assert (
        protocol.contract(sparse, "forward", "sdpa")["sdpa_mask_or_bias"]
        == "inside timing"
    )
    assert protocol.contract(sparse, "forward", "flex-cute")["block_mask_size"] == [
        256,
        128,
    ]


def test_backward_contract_uses_measured_source(profile: dict, tmp_path: Path) -> None:
    case = protocol.select_cases(profile, "alibi", "0")[0]
    metadata = protocol.contract(case, "backward", "helion-cute")
    assert metadata["output"] == "dq, dk, dv"
    assert "delta" in metadata["timed_region"]
    assert "outside timing" in metadata["forward_state"]
    assert protocol.runtime_target(case, "backward") == "backward"
    args = argparse.Namespace(
        gpus=[0],
        impls=["helion-cute"],
        mode="backward",
        retune=False,
        config_dir=tmp_path,
        out=tmp_path / "out",
    )
    jobs = run.make_jobs(
        [case], args, {"targets": {"backward": {"helion_commit": "61c840f39"}}}
    )
    assert jobs[0]["runtime"]["helion_commit"] == "61c840f39"
    assert jobs[0]["record_metadata"]["measurement"]["autotune_effort"] == "pinned"
    assert jobs[0]["record_metadata"]["measurement"]["autotune_budget_seconds"] == 0


def test_latest_mla_is_explicitly_unavailable(profile: dict, tmp_path: Path) -> None:
    cases = protocol.select_cases(profile, "mla-causal", "0,1,2")
    args = argparse.Namespace(
        gpus=[1],
        impls=["helion-cute", "sdpa"],
        mode="backward",
        retune=True,
        config_dir=None,
        out=tmp_path,
    )
    jobs = run.make_jobs(cases, args, {"targets": {}})
    assert len(jobs) == 6
    assert all(job["skip_reason"] for job in jobs if job["impl"] == "helion-cute")
    assert all(job["skip_reason"] is None for job in jobs if job["impl"] == "sdpa")
    assert all(job["case"]["shape"]["hkv"] == 16 for job in jobs)


@pytest.mark.parametrize("mode", ["forward", "backward"])
def test_original_vsa_flash_is_unsupported(
    profile: dict, tmp_path: Path, mode: str
) -> None:
    cases = protocol.select_cases(profile, "vsa", "0,1,2")
    args = argparse.Namespace(
        gpus=[0],
        impls=["flex-cute", "flex-triton"],
        mode=mode,
        retune=False,
        config_dir=None,
        out=tmp_path,
    )
    jobs = run.make_jobs(cases, args, {"targets": {}})
    assert all(
        "128-token" in job["skip_reason"] for job in jobs if job["impl"] == "flex-cute"
    )
    assert all(
        job["skip_reason"] is None for job in jobs if job["impl"] == "flex-triton"
    )


def test_forward_cute_cannot_silently_use_a_generic_path() -> None:
    worker.check_execution_path({"accuracy": "PASS", "flash_fired": True}, True)
    worker.check_execution_path({"accuracy": "PASS", "flash_fired": False}, False)
    with pytest.raises(RuntimeError, match="requires the fused CuTe flash path"):
        worker.check_execution_path({"accuracy": "PASS", "flash_fired": False}, True)


def test_summary_excludes_wrong_and_unmeasured_backends() -> None:
    records = [
        {"case_id": "a", "impl": "helion-cute", "accuracy": "PASS", "median_ms": 2},
        {"case_id": "a", "impl": "sdpa", "accuracy": "PASS", "median_ms": 4},
        {"case_id": "a", "impl": "flex-cute", "accuracy": "FAIL", "median_ms": 0.1},
        {"case_id": "a", "impl": "unmeasured", "accuracy": "PASS", "median_ms": 0.5},
        {"case_id": "b", "impl": "helion-cute", "accuracy": "FAIL", "median_ms": 1},
        {"case_id": "b", "impl": "sdpa", "accuracy": "PASS", "median_ms": 4},
    ]
    summary = run.summarize(records, [{}] * len(records))
    assert summary["complete"]
    assert summary["compared_shapes"] == 1
    assert summary["geomean_vs_best_nonhelion"] == 2


def test_preflight_detects_unrecorded_source_change(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    target = {
        "python": sys.executable,
        "helion_root": str(root),
        "harness_root": str(root),
        "helion_commit": "a16ffc60",
        "harness_commit": "a16ffc60",
        "software_validated": True,
        "source_status": "compatible_archived_cleanup",
        "exact_code": False,
        "code_deviation": "Measured refs unavailable",
        "helion_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "harness_diff_sha256": hashlib.sha256(b"").hexdigest(),
    }
    monkeypatch.setattr(run, "command_output", lambda command, **kwargs: "a16ffc60abcd")
    monkeypatch.setattr(
        run.subprocess, "check_output", lambda *args, **kwargs: b"changed"
    )
    result = run.preflight(
        {"status": "ready", "targets": {"forward_fused": target}},
        {"forward_fused"},
        probe_versions=False,
    )
    assert not result["ready"]
    assert any("unrecorded tracked changes" in error for error in result["errors"])
    assert not any("source status/deviation" in error for error in result["errors"])


def test_output_does_not_overwrite_inputs_or_existing_results(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    with pytest.raises(ValueError, match="protected input/source"):
        protocol.protected_output(configs / "new", [configs])
    output = tmp_path / "old"
    output.mkdir()
    (output / "results.jsonl").write_text("preserved\n")
    with pytest.raises(ValueError, match="new or empty"):
        protocol.protected_output(output, [])
    assert (output / "results.jsonl").read_text() == "preserved\n"


@pytest.mark.parametrize("torch_version,ready", [("pinned", True), ("changed", False)])
def test_preflight_checks_live_package_version(
    tmp_path: Path, monkeypatch, torch_version: str, ready: bool
) -> None:
    target = {
        "python": sys.executable,
        "helion_root": str(tmp_path),
        "harness_root": str(tmp_path),
        "helion_commit": "a16ffc60",
        "harness_commit": "a16ffc60",
        "software_validated": True,
        "software_targets": {"torch": "pinned"},
        "source_status": "compatible_archived_cleanup",
        "exact_code": False,
        "code_deviation": "Measured revisions unavailable",
    }

    def output(command, **kwargs):
        if command[0] == "git":
            return "a16ffc60abcd"
        return json.dumps({"torch": torch_version})

    monkeypatch.setattr(run, "command_output", output)
    monkeypatch.setattr(run.subprocess, "check_output", lambda *args, **kwargs: b"")
    result = run.preflight(
        {"status": "ready", "targets": {"forward_fused": target}},
        {"forward_fused"},
        probe_versions=True,
    )
    assert result["ready"] is ready
    assert result["live_software_probe"]
