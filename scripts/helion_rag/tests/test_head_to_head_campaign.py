"""Unit tests for the resumable four-arm campaign driver (no GPU/subprocess)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from helion_rag.experiment.head_to_head import build_schedule
from helion_rag.experiment.head_to_head import build_study_manifest
from helion_rag.experiment.scheduler import ScheduledRun

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "helion_rag" / "run_head_to_head_campaign.py"
_SPEC = importlib.util.spec_from_file_location("run_head_to_head_campaign", MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def _payload(*, correct: bool = True, attempts: int = 12) -> dict[str, object]:
    return {
        "readiness_seconds": 1.0,
        "selected_latency_ms": 0.5 if correct else None,
        "correct": correct,
        "tier": None,
        "decision": "BaselineSearch",
        "provider": {"requests": 0},
        "attempt_accounting": {"frozen_limit": 80, "attempted": attempts},
        "hybrid_stage_breakdown": None,
    }


def _fake_ok(cmd, *, env, correct: bool = True, **kwargs):
    """Simulate a worker that emits one event and prints a completed payload."""
    log = Path(env["HELION_RAG_EVENT_LOG"])
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text('{"event":1}\n', encoding="utf-8")
    return subprocess.CompletedProcess(
        cmd, 0, stdout=json.dumps(_payload(correct=correct)) + "\n", stderr=""
    )


def _counter():
    state = {"n": 0}

    def monotonic() -> float:
        state["n"] += 1
        return float(state["n"])

    return monotonic


def _run_unit(tmp_path: Path, run: ScheduledRun, run_process):
    return driver._run_one(
        run,
        worker_command=["python", "-c", "pass"],
        output_dir=tmp_path,
        base_seed=1000,
        effort="quick",
        candidate_attempt_limit=80,
        provider="vertex",
        model="claude-opus-4-8",
        timeout_seconds=1800.0,
        audit_path=tmp_path / "audit_ledger.jsonl",
        run_process=run_process,
        monotonic=_counter(),
    )


_RUN = ScheduledRun(workload="wl0", arm="lfbo", repetition=1, order_index=0)


def test_run_one_records_completed_payload(tmp_path: Path) -> None:
    result = _run_unit(tmp_path, _RUN, _fake_ok)
    assert result["status"] == "completed"
    assert result["correct"] is True
    assert result["attempt"] == 1
    assert (tmp_path / "audit_ledger.jsonl").is_file()


def test_run_one_retries_once_before_any_event(tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky(cmd, *, env, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="early death")
        return _fake_ok(cmd, env=env)

    result = _run_unit(tmp_path, _RUN, flaky)
    assert calls["n"] == 2  # one infrastructure retry
    assert result["status"] == "completed"
    assert result["attempt"] == 2
    # Both launch attempts are preserved in the audit ledger.
    audit_lines = (tmp_path / "audit_ledger.jsonl").read_text().splitlines()
    assert len(audit_lines) == 2


def test_run_one_does_not_retry_after_an_event_was_emitted(tmp_path: Path) -> None:
    calls = {"n": 0}

    def fail_after_event(cmd, *, env, **kwargs):
        calls["n"] += 1
        Path(env["HELION_RAG_EVENT_LOG"]).parent.mkdir(parents=True, exist_ok=True)
        Path(env["HELION_RAG_EVENT_LOG"]).write_text('{"event":1}\n', encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="crash after event"
        )

    result = _run_unit(tmp_path, _RUN, fail_after_event)
    assert calls["n"] == 1  # emitted a terminal event => never retried
    assert result["status"] == "failed"


def test_run_one_timeout_is_censored_not_retried(tmp_path: Path) -> None:
    calls = {"n": 0}

    def timeout(cmd, *, env, timeout=None, **kwargs):
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd, timeout or 1)

    result = _run_unit(tmp_path, _RUN, timeout)
    assert calls["n"] == 1
    assert result["status"] == "censored"
    assert result["right_censoring_boundary_seconds"] == 1800.0


def test_reconstruct_uses_schedule_order_not_append_order(tmp_path: Path) -> None:
    schedule = build_schedule(["wl0"], schedule_seed=1, repetitions=2)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # Write result files in reverse order to prove identity is not append order.
    for run in reversed(schedule):
        driver._atomic_write_json(
            driver._result_path(results_dir, run),
            {
                "workload": run.workload,
                "arm": run.arm,
                "repetition": run.repetition,
                "status": "completed",
                "correct": True,
            },
        )
    records, summary = driver._reconstruct(results_dir, schedule)
    assert [(r["arm"], r["repetition"]) for r in records] == [
        (run.arm, run.repetition) for run in schedule
    ]
    assert summary["recorded"] == len(schedule)


def test_refresh_api_key_mints_a_fresh_token_when_helper_is_set(tmp_path: Path) -> None:
    helper = tmp_path / "key_helper.sh"
    helper.write_text("#!/bin/sh\necho fresh-token-123\n", encoding="utf-8")
    helper.chmod(0o755)
    env = {"HELION_LLM_API_KEY_HELPER": str(helper), "HELION_LLM_API_KEY": "stale"}
    driver._refresh_api_key(env)
    assert env["HELION_LLM_API_KEY"] == "fresh-token-123"


def test_refresh_api_key_is_a_noop_without_a_helper() -> None:
    env = {"HELION_LLM_API_KEY": "unchanged"}
    driver._refresh_api_key(env)
    assert env["HELION_LLM_API_KEY"] == "unchanged"


def test_lock_prevents_a_second_campaign(tmp_path: Path) -> None:
    lock = driver._acquire_lock(tmp_path)
    try:
        with pytest.raises(SystemExit, match="already running"):
            driver._acquire_lock(tmp_path)
    finally:
        lock.unlink(missing_ok=True)


def _patch_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(args, workloads):
        import os

        return build_study_manifest(
            workloads=workloads,
            repetitions=args.repetitions,
            base_seed=args.base_seed,
            schedule_seed=args.schedule_seed,
            candidate_attempt_limit=args.candidate_attempt_limit,
            effort=args.effort,
            provider=args.provider,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            generation_id=os.environ["HELION_RAG_GENERATION_ID"],
            code_id="testcode",
        )

    monkeypatch.setattr(driver, "_build_manifest", fake)


def test_resume_skips_completed_units_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_manifest(monkeypatch)
    monkeypatch.setenv("HELION_RAG_GENERATION_ID", "000000")
    argv = [
        "--output-dir",
        str(tmp_path),
        "--workload",
        "add-1024x1024",
        "--repetitions",
        "1",
        "--min-valid-reps",
        "1",
    ]
    rc1 = driver.main(argv, run_process=_fake_ok)
    assert rc1 == 0
    assert (
        len(list((tmp_path / "results").glob("*.json"))) == 4
    )  # 1 wl x 4 arms x 1 rep

    calls = {"n": 0}

    def should_not_run(cmd, *, env, **kwargs):
        calls["n"] += 1
        return _fake_ok(cmd, env=env)

    rc2 = driver.main([*argv, "--resume"], run_process=should_not_run)
    assert rc2 == 0
    assert calls["n"] == 0  # every unit already terminal => nothing re-run


def test_resume_rejects_a_manifest_from_a_different_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_manifest(monkeypatch)
    monkeypatch.setenv("HELION_RAG_GENERATION_ID", "000000")
    base = [
        "--output-dir",
        str(tmp_path),
        "--workload",
        "add-1024x1024",
        "--repetitions",
        "1",
        "--min-valid-reps",
        "1",
    ]
    assert (
        driver.main([*base, "--candidate-attempt-limit", "80"], run_process=_fake_ok)
        == 0
    )
    rc = driver.main(
        [*base, "--candidate-attempt-limit", "40", "--resume"], run_process=_fake_ok
    )
    assert rc == 2  # different budget => refuse to mix


def test_unknown_workload_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_manifest(monkeypatch)
    monkeypatch.setenv("HELION_RAG_GENERATION_ID", "000000")
    rc = driver.main(
        [
            "--output-dir",
            str(tmp_path),
            "--workloads",
            "does-not-exist",
            "--repetitions",
            "1",
        ],
        run_process=_fake_ok,
    )
    assert rc == 2


def test_workloads_subset_runs_only_the_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_manifest(monkeypatch)
    monkeypatch.setenv("HELION_RAG_GENERATION_ID", "000000")
    rc = driver.main(
        [
            "--output-dir",
            str(tmp_path),
            "--workloads",
            "add-1024x1024,softmax-2048x2048",
            "--repetitions",
            "1",
            "--min-valid-reps",
            "1",
        ],
        run_process=_fake_ok,
    )
    assert rc == 0
    # 2 workloads x 4 arms x 1 rep = 8 terminal records, and nothing else ran.
    assert len(list((tmp_path / "results").glob("*.json"))) == 8


def test_exit_code_flags_insufficient_valid_reps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_manifest(monkeypatch)
    monkeypatch.setenv("HELION_RAG_GENERATION_ID", "000000")

    def incorrect(cmd, *, env, **kwargs):
        return _fake_ok(cmd, env=env, correct=False)

    rc = driver.main(
        [
            "--output-dir",
            str(tmp_path),
            "--workload",
            "add-1024x1024",
            "--repetitions",
            "1",
            "--min-valid-reps",
            "1",
        ],
        run_process=incorrect,
    )
    # All runs complete-but-incorrect => 0 valid reps < 1 => nonzero exit.
    assert rc == 1
