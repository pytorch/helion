"""Resumable overnight driver for the four-arm head-to-head campaign.

Runs every selected workload x 4 arms x N repetitions through one balanced
schedule, each in an isolated subprocess, and is safe to interrupt and resume.
The published campaign used 33 of the registered workloads with 5 repetitions
(660 run units); ``--workloads`` selects a subset and defaults to the whole
registry. Every run's terminal outcome is stored atomically in a unique file
keyed by (workload, arm, repetition); ``runs.jsonl`` and the summary are
reconstructed from those files, never from append order.

Overnight (detached), from the repo root:

    source .helion-rag/env.sh
    export HELION_RAG_GENERATION_ID=000000     # required; see the note below
    export HELION_LLM_API_KEY=...              # from your own credential source
    export HF_HUB_OFFLINE=1

    PYTHONPATH=scripts/helion_rag \\
      nohup .venv/bin/python scripts/helion_rag/run_head_to_head_campaign.py \\
      --output-dir .helion-rag/head_to_head_4arm \\
      --repetitions 5 --candidate-attempt-limit 80 --timeout-seconds 1800 \\
      --resume > .helion-rag/head_to_head_4arm/campaign.log 2>&1 &

Or under tmux:  ``tmux new -s h2h`` then run the command above (without nohup).
Inspect status:   ``tail -f .helion-rag/head_to_head_4arm/campaign.log`` and
``python scripts/helion_rag/run_head_to_head_campaign.py --output-dir ... --status``.
Safe resume:      re-run the same command with ``--resume`` (skips completed units;
refuses to mix results from a different budget/model/generation/code/schedule).
Analyze:          ``python scripts/helion_rag/analyze_head_to_head.py --campaign ...``.

Pinning ``HELION_RAG_GENERATION_ID`` is mandatory: without it the RAG arm hits a
GenerationPinError, silently falls back to BaselineSearch, and the retrieval
contrast this campaign measures disappears. See
``docs/rag_autotuning_experiment.md`` for the full setup.

The published run took ~14 h on a single H100, subject to provider and
compilation latency.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from helion_rag.experiment.head_to_head import ARMS
from helion_rag.experiment.head_to_head import build_run_environment
from helion_rag.experiment.head_to_head import build_schedule
from helion_rag.experiment.head_to_head import build_study_manifest
from helion_rag.experiment.head_to_head import manifest_matches
from helion_rag.experiment.head_to_head import registered_workloads
from helion_rag.experiment.scheduler import ScheduledRun

_TERMINAL_STATUSES = ("completed", "failed", "censored")


# ── Atomic IO ───────────────────────────────────────────────────────────
def _atomic_write_json(path: Path, obj: object) -> None:
    """Write JSON to a temp sibling then rename, so readers never see partials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _result_key(run: ScheduledRun) -> str:
    return f"{run.workload}__{run.arm}__r{run.repetition}"


def _result_path(results_dir: Path, run: ScheduledRun) -> Path:
    return results_dir / f"{_result_key(run)}.json"


def _event_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _parse_worker_output(stdout: str) -> dict[str, object]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("worker produced no JSON result")
    parsed = json.loads(lines[-1])
    if not isinstance(parsed, dict):
        raise ValueError("worker result must be a JSON object")
    return parsed


# ── Lock / PID file ─────────────────────────────────────────────────────
def _acquire_lock(output_dir: Path) -> Path:
    """Create an exclusive PID lock so two campaigns cannot target one directory."""
    lock = output_dir / "campaign.lock"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        holder = lock.read_text(encoding="utf-8").strip() or "?"
        if _pid_alive(holder):
            raise SystemExit(
                f"campaign already running (lock {lock} held by PID {holder})"
            )
        # Stale lock from a dead process; reclaim it.
        lock.unlink(missing_ok=True)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        handle.write(str(os.getpid()))
    return lock


def _pid_alive(pid_text: str) -> bool:
    try:
        pid = int(pid_text)
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
    except OSError as error:
        return error.errno == errno.EPERM  # exists but not ours
    return True


# ── One run unit (with retry-once-before-any-event) ─────────────────────
def _run_one(
    run: ScheduledRun,
    *,
    worker_command: Sequence[str],
    output_dir: Path,
    base_seed: int,
    effort: str,
    candidate_attempt_limit: int,
    provider: str,
    model: str,
    timeout_seconds: float,
    audit_path: Path,
    run_process: Callable[..., subprocess.CompletedProcess[str]],
    monotonic: Callable[[], float],
) -> dict[str, object]:
    """Launch one run unit, retrying once only on a pre-event infrastructure death."""
    base_record: dict[str, object] = {
        "workload": run.workload,
        "arm": run.arm,
        "repetition": run.repetition,
        "order_index": run.order_index,
        "random_seed": base_seed + run.repetition - 1,
    }
    max_attempts = 2
    last_failure: dict[str, object] | None = None
    for attempt in range(1, max_attempts + 1):
        env = build_run_environment(
            dict(os.environ),
            run,
            output_dir=output_dir,
            base_seed=base_seed,
            effort=effort,
            candidate_attempt_limit=candidate_attempt_limit,
            provider=provider,
            model=model,
        )
        if attempt > 1:
            # Retry uses a fresh cache/event/scratch directory but the same seed.
            _isolate_retry(env, attempt)
        _refresh_api_key(env)
        event_log = Path(env["HELION_RAG_EVENT_LOG"])
        # Start each attempt from an empty event log so a re-run (e.g. resume of a
        # unit whose result was cleared) emits exactly one event rather than
        # appending a second to a stale log.
        event_log.unlink(missing_ok=True)
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

        started = monotonic()
        timed_out = False
        oserror: str | None = None
        completed: subprocess.CompletedProcess[str] | None = None
        try:
            completed = run_process(
                list(worker_command),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
        except OSError as error:
            oserror = str(error)
        elapsed = monotonic() - started
        event_count = _event_count(event_log)

        # Timeout censoring is terminal (never dropped, never retried).
        if timed_out:
            record = {
                **base_record,
                "status": "censored",
                "elapsed_seconds": elapsed,
                "right_censoring_boundary_seconds": timeout_seconds,
                "returncode": None,
                "attempt": attempt,
                "event_count": event_count,
                "event_log": _relative(output_dir, event_log),
                "error": "worker deadline exceeded",
            }
            _append_audit(audit_path, record)
            return record

        returncode = completed.returncode if completed is not None else None
        if completed is not None and returncode == 0:
            try:
                payload = _parse_worker_output(completed.stdout)
                record = {
                    **base_record,
                    "status": "completed",
                    "elapsed_seconds": elapsed,
                    "returncode": 0,
                    "attempt": attempt,
                    "event_count": event_count,
                    "event_log": _relative(output_dir, event_log),
                    **payload,
                }
                _append_audit(audit_path, record)
                return record
            except (ValueError, json.JSONDecodeError) as error:
                failure_error = f"unparsable worker output: {error}"
        elif oserror is not None:
            failure_error = oserror
        else:
            failure_error = (
                (completed.stderr or "")[-4000:] if completed else "no output"
            )

        record = {
            **base_record,
            "status": "failed",
            "elapsed_seconds": elapsed,
            "returncode": returncode,
            "attempt": attempt,
            "event_count": event_count,
            "event_log": _relative(output_dir, event_log),
            "error": failure_error,
        }
        _append_audit(audit_path, record)
        # Retry once only when the subprocess died before emitting any canonical
        # event; a run that emitted a terminal event is never retried.
        if event_count == 0 and attempt < max_attempts:
            last_failure = record
            continue
        return record
    assert last_failure is not None
    return last_failure


def _refresh_api_key(env: dict[str, str]) -> None:
    """Refresh ``HELION_LLM_API_KEY`` from a helper before a run, if configured.

    Vertex/gateway tokens are short-lived, so a multi-hour campaign captured with
    a single token would start failing the LLM arms mid-run. When
    ``HELION_LLM_API_KEY_HELPER`` names an executable, invoke it per run unit to
    mint a fresh token. Opt-in: absent the variable, nothing changes. A helper
    failure leaves any existing key in place (the run then surfaces an auth error
    rather than a silent stale token).
    """
    helper = env.get("HELION_LLM_API_KEY_HELPER")
    if not helper:
        return
    try:
        result = subprocess.run(
            [helper], capture_output=True, text=True, timeout=30, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return
    key = result.stdout.strip()
    if key:
        env["HELION_LLM_API_KEY"] = key


def _isolate_retry(env: dict[str, str], attempt: int) -> None:
    suffix = f"-retry{attempt}"
    cache = env["HELION_CACHE_DIR"] + suffix
    env["HELION_CACHE_DIR"] = cache
    env["TORCHINDUCTOR_CACHE_DIR"] = str(Path(cache) / "torchinductor")
    env["TRITON_CACHE_DIR"] = str(Path(cache) / "triton")
    env["TMPDIR"] = env["TMPDIR"] + suffix
    log = Path(env["HELION_RAG_EVENT_LOG"])
    env["HELION_RAG_EVENT_LOG"] = str(log.with_name(f"{log.stem}{suffix}{log.suffix}"))


def _relative(output_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def _append_audit(audit_path: Path, record: dict[str, object]) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "workload": record["workload"],
        "arm": record["arm"],
        "repetition": record["repetition"],
        "attempt": record.get("attempt"),
        "status": record["status"],
        "returncode": record.get("returncode"),
        "event_count": record.get("event_count"),
        "elapsed_seconds": record.get("elapsed_seconds"),
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


# ── Reconstruction from per-run files (never append order) ───────────────
def _reconstruct(
    results_dir: Path, schedule: Sequence[ScheduledRun]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    for run in schedule:
        path = _result_path(results_dir, run)
        if path.is_file():
            records.append(json.loads(path.read_text(encoding="utf-8")))

    per_kernel_arm: dict[str, dict[str, int]] = {}
    for record in records:
        key = f"{record['workload']}::{record['arm']}"
        bucket = per_kernel_arm.setdefault(
            key, {"completed": 0, "correct": 0, "censored": 0, "failed": 0}
        )
        status = str(record.get("status"))
        if status in bucket:
            bucket[status] += 1
        if status == "completed" and record.get("correct") is True:
            bucket["correct"] += 1

    summary = {
        "total_run_units": len(schedule),
        "recorded": len(records),
        "by_status": dict(Counter(str(r.get("status")) for r in records)),
        "per_kernel_arm": per_kernel_arm,
    }
    return records, summary


def _write_runs_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


# ── Driver ───────────────────────────────────────────────────────────────
def _build_manifest(
    args: argparse.Namespace, workloads: Sequence[str]
) -> dict[str, object]:
    from helion.autotuner.base_cache import helion_key

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
        code_id=helion_key(),
    )


def _status_report(output_dir: Path) -> int:
    manifest_path = output_dir / "study_manifest.json"
    if not manifest_path.is_file():
        print(f"no campaign at {output_dir}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schedule = build_schedule(
        list(manifest["workloads"]),
        schedule_seed=int(manifest["schedule_seed"]),
        repetitions=int(manifest["repetitions"]),
    )
    _records, summary = _reconstruct(output_dir / "results", schedule)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    run_process: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--candidate-attempt-limit", type=int, default=80)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--schedule-seed", type=int, default=20260722)
    parser.add_argument("--effort", default="quick")
    parser.add_argument("--provider", default="vertex")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--min-valid-reps", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--workload", default=None, help="restrict to one workload (pilot)"
    )
    parser.add_argument(
        "--workloads",
        default=None,
        help="comma-separated subset of workload ids to run (overrides --workload)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true", help="print progress and exit")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    if args.status:
        return _status_report(output_dir)

    if not os.environ.get("HELION_RAG_GENERATION_ID"):
        print(
            "REFUSING: HELION_RAG_GENERATION_ID is unset; the retrieval arm needs "
            "the pinned generation. Export it first (e.g. 000000).",
            file=sys.stderr,
        )
        return 2

    registry = set(registered_workloads())
    if args.workloads:
        workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    elif args.workload:
        workloads = [args.workload]
    else:
        workloads = registered_workloads()
    unknown = [w for w in workloads if w not in registry]
    if unknown:
        print(
            f"REFUSING: unknown workload id(s) {unknown}; not in the registry.",
            file=sys.stderr,
        )
        return 2
    intended = _build_manifest(args, workloads)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "study_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            print(
                f"REFUSING: {manifest_path} already exists; pass --resume to continue.",
                file=sys.stderr,
            )
            return 2
        if (
            not manifest_matches(existing)
            or existing.get("config_hash") != intended["config_hash"]
        ):
            print(
                "REFUSING to resume: the existing manifest was produced with a "
                "different budget / model / generation / code / schedule.",
                file=sys.stderr,
            )
            return 2
    else:
        _atomic_write_json(manifest_path, intended)

    schedule = build_schedule(
        workloads, schedule_seed=args.schedule_seed, repetitions=args.repetitions
    )
    schedule_path = output_dir / "schedule.json"
    if not schedule_path.is_file():
        _atomic_write_json(schedule_path, [dataclasses.asdict(run) for run in schedule])

    lock = _acquire_lock(output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit_ledger.jsonl"
    worker_command = [
        sys.executable,
        "-m",
        "helion_rag.experiment.head_to_head_worker",
    ]
    total = len(schedule)
    arm_failures: Counter[str] = Counter()
    ran = 0
    campaign_start = time.monotonic()
    try:
        for index, run in enumerate(schedule):
            result_path = _result_path(results_dir, run)
            if result_path.is_file():
                continue  # terminal unit already recorded; skip on resume
            result = _run_one(
                run,
                worker_command=worker_command,
                output_dir=output_dir,
                base_seed=args.base_seed,
                effort=args.effort,
                candidate_attempt_limit=args.candidate_attempt_limit,
                provider=args.provider,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                audit_path=audit_path,
                run_process=run_process,
                monotonic=time.monotonic,
            )
            _atomic_write_json(result_path, result)
            ran += 1
            if not (
                result.get("status") == "completed" and result.get("correct") is True
            ):
                arm_failures[run.arm] += 1
            if ran % max(1, args.progress_every) == 0 or index + 1 == total:
                _print_progress(
                    schedule, results_dir, arm_failures, campaign_start, ran
                )
    finally:
        lock.unlink(missing_ok=True)

    records, summary = _reconstruct(results_dir, schedule)
    _write_runs_jsonl(output_dir / "runs.jsonl", records)
    _atomic_write_json(output_dir / "summary.json", summary)

    return _exit_code(schedule, results_dir, summary, args.min_valid_reps)


def _print_progress(
    schedule: Sequence[ScheduledRun],
    results_dir: Path,
    arm_failures: Counter[str],
    campaign_start: float,
    ran: int,
) -> None:
    done = sum(1 for run in schedule if _result_path(results_dir, run).is_file())
    total = len(schedule)
    elapsed = time.monotonic() - campaign_start
    rate = elapsed / ran if ran else 0.0
    remaining = total - done
    eta_min = (rate * remaining) / 60.0 if rate else 0.0
    failures = (
        ", ".join(f"{arm}={n}" for arm, n in sorted(arm_failures.items())) or "none"
    )
    print(
        f"[{done}/{total}] remaining={remaining} elapsed={elapsed / 60:.0f}m "
        f"eta~{eta_min:.0f}m failures/censored: {failures}",
        flush=True,
    )


def _exit_code(
    schedule: Sequence[ScheduledRun],
    results_dir: Path,
    summary: dict[str, object],
    min_valid_reps: int,
) -> int:
    missing = [run for run in schedule if not _result_path(results_dir, run).is_file()]
    if missing:
        print(
            f"INCOMPLETE: {len(missing)} run units have no terminal record",
            file=sys.stderr,
        )
        return 1
    per_kernel_arm = summary["per_kernel_arm"]
    assert isinstance(per_kernel_arm, dict)
    workloads = sorted({run.workload for run in schedule})
    under = [
        f"{workload}::{arm}"
        for workload in workloads
        for arm in ARMS
        if per_kernel_arm.get(f"{workload}::{arm}", {}).get("correct", 0)
        < min_valid_reps
    ]
    if under:
        print(
            f"INSUFFICIENT: {len(under)} kernel-arm cells have < {min_valid_reps} "
            f"valid reps (e.g. {under[:3]})",
            file=sys.stderr,
        )
        return 1
    print("campaign complete; all run units recorded and inferentially eligible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
