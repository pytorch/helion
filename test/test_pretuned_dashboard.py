"""Tests for the pretuned benchmark dashboard data builder."""

from __future__ import annotations

import datetime
import importlib.util
import json
from pathlib import Path

_BUILDER_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "dashboard"
    / "build_pretuned_dashboard_data.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_pretuned_dashboard_builder", _BUILDER_PATH
)
assert _SPEC is not None
assert _SPEC.loader is not None
_BUILDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BUILDER)


def _run(run_id: str, date: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "sha": f"sha{run_id}",
        "full_sha": f"full-sha-{run_id}",
        "date": date,
        "branch": "main",
        "is_nightly": True,
    }


def _write_artifact(cache_dir: Path, run_id: str, records: list[dict]) -> None:
    artifact_dir = cache_dir / run_id / "pretuned-results-h100"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "pretuned-bench.json").write_text(json.dumps(records))


def test_latest_kernel_failure_replaces_previous_speedup(tmp_path: Path) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    first_run = _run("1", (now - datetime.timedelta(days=1)).isoformat())
    second_run = _run("2", now.isoformat())
    runs = [first_run, second_run]
    common = {
        "device": "NVIDIA H100",
        "compute_capability": "sm90",
        "cudagraph": False,
    }
    _write_artifact(
        tmp_path / "first-build",
        "1",
        [
            {
                **common,
                "kernel": "failed_kernel",
                "geomean": 2.0,
                "helion_wins": 4,
                "total": 4,
                "best_speedup": 2.5,
            },
            {
                **common,
                "kernel": "healthy_kernel",
                "geomean": 1.5,
                "helion_wins": 3,
                "total": 4,
                "best_speedup": 2.0,
            },
        ],
    )
    existing = _BUILDER.build_dashboard_data(tmp_path / "first-build", [first_run])

    _write_artifact(
        tmp_path / "incremental-build",
        "2",
        [
            {
                **common,
                "kernel": "failed_kernel",
                "error": "RuntimeError: compile failed",
            },
            {
                **common,
                "kernel": "healthy_kernel",
                "geomean": 3.0,
                "helion_wins": 4,
                "total": 4,
                "best_speedup": 3.5,
            },
            {
                **common,
                "kernel": "unsupported_kernel",
                "skipped": "pretuned for ['b200']; current hardware is h100",
            },
        ],
    )

    data = _BUILDER.build_dashboard_data(
        tmp_path / "incremental-build", runs, existing_data=existing
    )
    summary = {entry["kernel"]: entry for entry in data["summary"]}

    failed = summary["failed_kernel"]
    assert failed["failed"] is True
    assert failed["status"] == "failed"
    assert failed["error"] == "RuntimeError: compile failed"
    assert failed["geomean"] == 0
    assert failed["geomean_delta_pct"] is None
    assert failed["history"][-1]["error"] == "RuntimeError: compile failed"

    assert "unsupported_kernel" not in summary
    assert data["stats"]["failed_count"] == 1
    assert data["stats"]["unchanged_count"] == 0
    assert data["stats"]["geomean"] == 3.0
