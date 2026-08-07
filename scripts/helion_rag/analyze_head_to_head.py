"""Analysis, trajectories, and figures for the four-arm head-to-head campaign.

Consumes a campaign directory produced by ``run_head_to_head_campaign.py``
(``results/*.json`` terminal records plus per-run ``events/*.jsonl`` logs) and
writes final-outcome tables, an aggregate statistical report, per-candidate
trajectories, and poster-ready figures. LFBO is the principal display baseline
while the full pairwise matrix is retained.

    PYTHONPATH=scripts/helion_rag .venv/bin/python \\
      scripts/helion_rag/analyze_head_to_head.py --campaign .helion-rag/head_to_head_4arm

Outputs (under ``<campaign>/analysis`` and ``<campaign>/figures``):
  per_run.csv, per_kernel_arm.csv, all_arm_table.{csv,md}, aggregate_statistics.{json,csv},
  reliability.csv, cost.csv, trajectory_long.csv, and SVG/PDF/PNG figures.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path
import shutil
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Sequence

from helion_rag.experiment.head_to_head import ARM_LFBO
from helion_rag.experiment.head_to_head import ARMS
from helion_rag.stats.gates import holm_adjust
from helion_rag.stats.paired import bootstrap_geometric_mean_ci
from helion_rag.stats.paired import rank_biserial
from helion_rag.stats.paired import wilcoxon_pvalue
from helion_rag.stats.paired import wins_ties_losses

OUTCOMES = ("selected_latency_ms", "readiness_seconds")
ARM_COLORS = {
    ARM_LFBO: "#777777",
    "llm": "#0072B2",
    "hybrid_lfbo_llm": "#009E73",
    "contextual_rag_llm": "#D55E00",
}
POSTER_LFBO_COLOR = "#E76F51"
POSTER_LLM_COLOR = "#4C78A8"
POSTER_HYBRID_COLOR = "#8F63B8"
POSTER_RAG_COLOR = "#169C95"
POSTER_TIE_COLOR = "#8A8A8A"
POSTER_RATIO_NOISE = 0.025
POSTER_ARM_COLORS = {
    ARM_LFBO: POSTER_LFBO_COLOR,
    "llm": POSTER_LLM_COLOR,
    "hybrid_lfbo_llm": POSTER_HYBRID_COLOR,
    "contextual_rag_llm": POSTER_RAG_COLOR,
}
POSTER_ARM_LABELS = {
    ARM_LFBO: "LFBO",
    "llm": "LLM",
    "hybrid_lfbo_llm": "Hybrid",
    "contextual_rag_llm": "RAG-LLM",
}
POSTER_WORKLOAD_CANDIDATES = (
    "matmul-4096x4096x4096",
    "matmul-8192x8192x8192",
    "matmul_split_k-64x16384x64",
    "attention-2x8x4096x64",
    "attention-2x8x8192x64",
    "fp8_attention-2x4x2048x64",
    "fp8_attention-2x4x8192x64",
    "grouped_gemm-g8m512",
    "swiglu-8192x8192",
    "softmax-4096x32768",
    "gdn_fwd_h-b1h4s8192ds128",
    "rms_norm-4096x32768",
    "rope-1x4x2x8192x128",
    "mamba2_chunk_scan-b1h4s4096ds256",
    "mamba2_chunk_scan-b1h4s8192ds256",
)


# ── Loading ──────────────────────────────────────────────────────────────
def load_runs(campaign: Path) -> list[dict[str, object]]:
    """Load every terminal per-run record from ``<campaign>/results``."""
    results_dir = campaign / "results"
    if not results_dir.is_dir():
        raise FileNotFoundError(f"no results directory at {results_dir}")
    runs: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("*.json")):
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    if not runs:
        raise ValueError(f"{results_dir} contains no run records")
    return runs


def _is_valid(record: dict[str, object]) -> bool:
    return record.get("status") == "completed" and record.get("correct") is True


def _number(record: dict[str, object], field: str) -> float | None:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _provider_field(record: dict[str, object], field: str) -> object:
    provider = record.get("provider")
    return provider.get(field) if isinstance(provider, dict) else None


def _attempts(record: dict[str, object]) -> object:
    accounting = record.get("attempt_accounting")
    return accounting.get("attempted") if isinstance(accounting, dict) else None


# ── Tidy per-run CSV ─────────────────────────────────────────────────────
def tidy_rows(runs: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in runs:
        breakdown = record.get("hybrid_stage_breakdown")
        breakdown = breakdown if isinstance(breakdown, dict) else {}
        rows.append(
            {
                "workload": record.get("workload"),
                "arm": record.get("arm"),
                "repetition": record.get("repetition"),
                "random_seed": record.get("random_seed"),
                "status": record.get("status"),
                "correct": record.get("correct"),
                "selected_latency_ms": record.get("selected_latency_ms"),
                "readiness_seconds": record.get("readiness_seconds"),
                "internal_selected_performance": record.get(
                    "internal_selected_performance"
                ),
                "incumbent_best_perf": record.get("incumbent_best_perf"),
                "attempted": _attempts(record),
                "provider_requests": _provider_field(record, "requests"),
                "input_tokens": _provider_field(record, "input_tokens"),
                "output_tokens": _provider_field(record, "output_tokens"),
                "tier": record.get("tier"),
                "decision": record.get("decision"),
                "evaluation_count": record.get("evaluation_count"),
                "llm_attempts": breakdown.get("llm_attempts"),
                "lfbo_attempts": breakdown.get("lfbo_attempts"),
            }
        )
    return rows


def _write_csv(
    path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


# ── Per-kernel arm summaries + all-arm table ─────────────────────────────
def _index(
    runs: Sequence[dict[str, object]],
) -> dict[tuple[str, str, int], dict[str, object]]:
    return {(str(r["workload"]), str(r["arm"]), int(r["repetition"])): r for r in runs}


def _bounded_campaign_oracle(runs: Sequence[dict[str, object]]) -> dict[str, float]:
    """Per-kernel oracle = min stabilized selected latency across all arms/reps.

    Distinct from the trajectory's internal incumbent oracle: this uses the
    stabilized ``run_example`` latency, matching the final-outcome/regret table.
    """
    oracle: dict[str, float] = {}
    for record in runs:
        if not _is_valid(record):
            continue
        value = _number(record, "selected_latency_ms")
        if value is None:
            continue
        workload = str(record["workload"])
        oracle[workload] = min(oracle.get(workload, math.inf), value)
    return oracle


def _lfbo_baseline(runs: Sequence[dict[str, object]]) -> dict[str, float]:
    """Per-kernel median LFBO selected latency, the display-baseline reference."""
    by_kernel: dict[str, list[float]] = {}
    for record in runs:
        if record.get("arm") != ARM_LFBO or not _is_valid(record):
            continue
        value = _number(record, "selected_latency_ms")
        if value is None:
            continue
        by_kernel.setdefault(str(record["workload"]), []).append(value)
    return {workload: statistics.median(vs) for workload, vs in by_kernel.items()}


def per_kernel_arm(runs: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    workloads = sorted({str(r["workload"]) for r in runs})
    oracle = _bounded_campaign_oracle(runs)
    lfbo_base = _lfbo_baseline(runs)
    rows: list[dict[str, object]] = []
    for workload in workloads:
        for arm in ARMS:
            cells = [r for r in runs if r["workload"] == workload and r["arm"] == arm]
            valid = [r for r in cells if _is_valid(r)]
            latencies = [
                v for r in valid if (v := _number(r, "selected_latency_ms")) is not None
            ]
            readiness = [
                v for r in valid if (v := _number(r, "readiness_seconds")) is not None
            ]
            attempts = [
                v for r in valid if isinstance((v := _attempts(r)), (int, float))
            ]
            in_tokens = [
                v
                for r in cells
                if isinstance((v := _provider_field(r, "input_tokens")), int)
            ]
            out_tokens = [
                v
                for r in cells
                if isinstance((v := _provider_field(r, "output_tokens")), int)
            ]
            median_latency = statistics.median(latencies) if latencies else None
            base = oracle.get(workload)
            regret_pct = (
                (median_latency / base - 1.0) * 100.0
                if median_latency is not None and base
                else None
            )
            lbase = lfbo_base.get(workload)
            regret_vs_lfbo_pct = (
                (median_latency / lbase - 1.0) * 100.0
                if median_latency is not None and lbase
                else None
            )
            rows.append(
                {
                    "workload": workload,
                    "arm": arm,
                    "n_runs": len(cells),
                    "n_correct": len(valid),
                    "n_censored": sum(
                        1 for r in cells if r.get("status") == "censored"
                    ),
                    "n_failed": sum(1 for r in cells if r.get("status") == "failed"),
                    "median_selected_latency_ms": median_latency,
                    "bounded_oracle_ms": base,
                    "regret_pct": regret_pct,
                    "regret_vs_lfbo_pct": regret_vs_lfbo_pct,
                    "median_readiness_seconds": statistics.median(readiness)
                    if readiness
                    else None,
                    "median_attempts": statistics.median(attempts)
                    if attempts
                    else None,
                    "input_tokens_total": sum(in_tokens),
                    "output_tokens_total": sum(out_tokens),
                    "provider_requests": next(
                        (_provider_field(r, "requests") for r in cells), None
                    ),
                }
            )
    return rows


def per_arm_summary(runs: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """One row per arm rolling up every headline metric across kernels."""
    kernel_rows = per_kernel_arm(runs)
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        arm_kernels = [r for r in kernel_rows if r["arm"] == arm and r["n_correct"]]
        latencies = [r["median_selected_latency_ms"] for r in arm_kernels]
        readiness = [r["median_readiness_seconds"] for r in arm_kernels]
        attempts = [
            r["median_attempts"]
            for r in arm_kernels
            if r["median_attempts"] is not None
        ]
        regrets = [r["regret_pct"] for r in arm_kernels if r["regret_pct"] is not None]
        regrets_lfbo = [
            r["regret_vs_lfbo_pct"]
            for r in arm_kernels
            if r["regret_vs_lfbo_pct"] is not None
        ]
        cells = [r for r in runs if r["arm"] == arm]
        rows.append(
            {
                "arm": arm,
                "kernels_valid": len(arm_kernels),
                "geomean_selected_latency_ms": _geomean(latencies),
                "geomean_readiness_seconds": _geomean(readiness),
                "geomean_regret_pct": (
                    (_geomean([1.0 + x / 100.0 for x in regrets]) - 1.0) * 100.0
                    if regrets and not any(x is None for x in regrets)
                    else None
                ),
                "geomean_regret_vs_lfbo_pct": (
                    (_geomean([1.0 + x / 100.0 for x in regrets_lfbo]) - 1.0) * 100.0
                    if regrets_lfbo and not any(x is None for x in regrets_lfbo)
                    else None
                ),
                "median_attempts": statistics.median(attempts) if attempts else None,
                "provider_requests_total": sum(
                    v
                    for r in cells
                    if isinstance((v := _provider_field(r, "requests")), int)
                ),
                "input_tokens_total": sum(
                    v
                    for r in cells
                    if isinstance((v := _provider_field(r, "input_tokens")), int)
                ),
                "output_tokens_total": sum(
                    v
                    for r in cells
                    if isinstance((v := _provider_field(r, "output_tokens")), int)
                ),
            }
        )
    return rows


def _geomean(values: Sequence[object]) -> float | None:
    finite = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and math.isfinite(v) and v > 0
    ]
    if not finite:
        return None
    return math.exp(sum(math.log(v) for v in finite) / len(finite))


# ── Matched-ratio aggregate statistics (12 Holm-corrected tests) ─────────
def _matched_ratios(
    index: dict[tuple[str, str, int], dict[str, object]],
    *,
    numerator: str,
    denominator: str,
    outcome: str,
    workloads: Sequence[str],
    repetitions: Sequence[int],
) -> tuple[list[float], int, int]:
    """Return per-kernel median ratios (num/den) plus matched and total blocks."""
    per_kernel: list[float] = []
    matched = 0
    total = 0
    for workload in workloads:
        rep_ratios: list[float] = []
        for rep in repetitions:
            total += 1
            num = index.get((workload, numerator, rep))
            den = index.get((workload, denominator, rep))
            if num is None or den is None or not _is_valid(num) or not _is_valid(den):
                continue
            num_v = _number(num, outcome)
            den_v = _number(den, outcome)
            if num_v is None or den_v is None or den_v == 0.0:
                continue
            matched += 1
            rep_ratios.append(num_v / den_v)
        if rep_ratios:
            per_kernel.append(statistics.median(rep_ratios))
    return per_kernel, matched, total


def aggregate_statistics(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    index = _index(runs)
    workloads = sorted({str(r["workload"]) for r in runs})
    repetitions = sorted({int(r["repetition"]) for r in runs})
    contrasts = list(itertools.combinations(ARMS, 2))

    entries: list[dict[str, object]] = []
    pvalues: list[float] = []
    for outcome in OUTCOMES:
        for arm_a, arm_b in contrasts:
            # Orient as arm_b / arm_a so LFBO-baseline pairs read as "other / LFBO".
            numerator, denominator = arm_b, arm_a
            ratios, matched, total = _matched_ratios(
                index,
                numerator=numerator,
                denominator=denominator,
                outcome=outcome,
                workloads=workloads,
                repetitions=repetitions,
            )
            logs = [math.log(r) for r in ratios if r > 0 and math.isfinite(r)]
            interval = bootstrap_geometric_mean_ci(ratios)
            wins, ties, losses = wins_ties_losses(ratios)
            p = wilcoxon_pvalue(logs)
            pvalues.append(p if math.isfinite(p) else 1.0)
            entries.append(
                {
                    "outcome": outcome,
                    "numerator_arm": numerator,
                    "denominator_arm": denominator,
                    "baseline_display": denominator == ARM_LFBO,
                    "geometric_mean_ratio": interval.estimate,
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                    "kernels": len(ratios),
                    "wilcoxon_p": p,
                    "rank_biserial": rank_biserial(logs),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "matched_blocks": matched,
                    "total_blocks": total,
                    "joint_success_coverage": (matched / total) if total else 0.0,
                }
            )
    holm = holm_adjust(pvalues)
    for entry, adjusted in zip(entries, holm, strict=True):
        entry["holm_p"] = adjusted
    return {
        "arms": list(ARMS),
        "baseline": ARM_LFBO,
        "outcomes": list(OUTCOMES),
        "family_size": len(entries),
        "contrasts": entries,
    }


# ── Reliability and cost tables ──────────────────────────────────────────
def reliability_rows(runs: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    workloads = sorted({str(r["workload"]) for r in runs})
    for arm in ARMS:
        cells = [r for r in runs if r["arm"] == arm]
        rows.append(
            {
                "arm": arm,
                "runs": len(cells),
                "correct": sum(1 for r in cells if _is_valid(r)),
                "censored": sum(1 for r in cells if r.get("status") == "censored"),
                "failed": sum(1 for r in cells if r.get("status") == "failed"),
                "kernels_3of5": sum(
                    1
                    for w in workloads
                    if sum(1 for r in cells if r["workload"] == w and _is_valid(r)) >= 3
                ),
            }
        )
    return rows


def cost_rows(runs: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        cells = [r for r in runs if r["arm"] == arm]
        requests = [
            v for r in cells if isinstance((v := _provider_field(r, "requests")), int)
        ]
        inp = [
            v
            for r in cells
            if isinstance((v := _provider_field(r, "input_tokens")), int)
        ]
        out = [
            v
            for r in cells
            if isinstance((v := _provider_field(r, "output_tokens")), int)
        ]
        rows.append(
            {
                "arm": arm,
                "provider_requests_total": sum(requests),
                "input_tokens_total": sum(inp),
                "output_tokens_total": sum(out),
            }
        )
    return rows


# ── Trajectories ─────────────────────────────────────────────────────────
def _load_event(campaign: Path, record: dict[str, object]) -> dict[str, object] | None:
    rel = record.get("event_log")
    if not isinstance(rel, str):
        return None
    path = campaign / rel
    if not path.is_file():
        return None
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        return None
    event = json.loads(lines[0])
    return event if isinstance(event, dict) else None


def trajectory_rows(
    campaign: Path, runs: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    """Flatten each run's evaluations into one row per candidate evaluation."""
    raw: list[dict[str, object]] = []
    # First pass: gather incumbents to define the per-workload internal oracle.
    oracle: dict[str, float] = {}
    per_run_evals: list[tuple[dict[str, object], list[dict[str, object]]]] = []
    for record in runs:
        event = _load_event(campaign, record)
        evals = event.get("evaluations") if isinstance(event, dict) else None
        evals = evals if isinstance(evals, list) else []
        per_run_evals.append((record, evals))
        workload = str(record.get("workload"))
        for ev in evals:
            inc = ev.get("incumbent_best_perf")
            if isinstance(inc, (int, float)) and math.isfinite(float(inc)):
                oracle[workload] = min(oracle.get(workload, math.inf), float(inc))

    for record, evals in per_run_evals:
        workload = str(record.get("workload"))
        arm = str(record.get("arm"))
        rep = record.get("repetition")
        base = oracle.get(workload)
        benchmarked = 0
        for ev in evals:
            if ev.get("benchmark_status") == "ok":
                benchmarked += 1
            inc = ev.get("incumbent_best_perf")
            inc_f = (
                float(inc)
                if isinstance(inc, (int, float)) and math.isfinite(float(inc))
                else None
            )
            ratio = (inc_f / base) if (inc_f is not None and base) else None
            raw.append(
                {
                    "workload": workload,
                    "repetition": rep,
                    "arm": arm,
                    "stage": ev.get("candidate_category"),
                    "candidate_source": ev.get("candidate_source"),
                    "elapsed_seconds": ev.get("elapsed_seconds"),
                    "benchmarked_index": benchmarked,
                    "internal_performance": ev.get("performance"),
                    "incumbent_best_perf": inc_f,
                    "normalized_incumbent_ratio": ratio,
                }
            )
    return raw


# ── Poster-chart estimands ───────────────────────────────────────────────
def poster_workloads(
    kernel_rows: Sequence[dict[str, object]], *, limit: int = 15
) -> list[str]:
    """Choose a diverse poster subset and order it by LFBO readiness.

    ``POSTER_WORKLOAD_CANDIDATES`` names one representative shape per kernel
    family in the published campaign. Any workload it does not cover -- a
    synthetic fixture, or a campaign with a different selection -- falls back to
    that campaign's hardest LFBO workloads.
    """
    lfbo_readiness = {
        str(row["workload"]): float(row["median_readiness_seconds"])
        for row in kernel_rows
        if row.get("arm") == ARM_LFBO
        and isinstance(row.get("median_readiness_seconds"), (int, float))
    }
    preferred = [w for w in POSTER_WORKLOAD_CANDIDATES if w in lfbo_readiness]
    remaining = sorted(
        (w for w in lfbo_readiness if w not in preferred),
        key=lfbo_readiness.get,
        reverse=True,
    )
    selected = (preferred + remaining)[: max(0, limit)]
    return sorted(selected, key=lfbo_readiness.get, reverse=True)


def poster_performance_ratios(
    kernel_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Every non-baseline arm's per-kernel ratio to LFBO with a tie band."""
    indexed = {(str(row["workload"]), str(row["arm"])): row for row in kernel_rows}
    workloads = sorted({str(row["workload"]) for row in kernel_rows})
    result: list[dict[str, object]] = []
    for workload in workloads:
        lfbo = indexed.get((workload, ARM_LFBO), {}).get("median_selected_latency_ms")
        if not isinstance(lfbo, (int, float)):
            continue
        for arm in ARMS:
            if arm == ARM_LFBO:
                continue
            latency = indexed.get((workload, arm), {}).get("median_selected_latency_ms")
            if not isinstance(latency, (int, float)):
                continue
            ratio = float(latency) / float(lfbo)
            classification = (
                "win"
                if ratio < 1.0 - POSTER_RATIO_NOISE
                else "loss"
                if ratio > 1.0 + POSTER_RATIO_NOISE
                else "tie"
            )
            result.append(
                {
                    "workload": workload,
                    "arm": arm,
                    "ratio": ratio,
                    "classification": classification,
                }
            )
    workload_order = {
        workload: index
        for index, workload in enumerate(
            sorted(
                workloads,
                key=lambda w: (
                    float(
                        indexed.get((w, "contextual_rag_llm"), {}).get(
                            "median_selected_latency_ms", math.inf
                        )
                    )
                    / float(
                        indexed.get((w, ARM_LFBO), {}).get(
                            "median_selected_latency_ms", 1.0
                        )
                    )
                ),
            )
        )
    }
    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    return sorted(
        result,
        key=lambda row: (
            workload_order[str(row["workload"])],
            arm_order[str(row["arm"])],
        ),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _trajectory_by_run(
    trajectory: Sequence[dict[str, object]],
    *,
    arms: Sequence[str],
    workloads: set[str] | None = None,
    value_field: str = "normalized_incumbent_ratio",
) -> dict[tuple[str, str, int], list[tuple[float, float]]]:
    series: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    for row in trajectory:
        arm = str(row.get("arm"))
        workload = str(row.get("workload"))
        elapsed = row.get("elapsed_seconds")
        value = row.get(value_field)
        repetition = row.get("repetition")
        if arm not in arms or (workloads is not None and workload not in workloads):
            continue
        if not isinstance(repetition, int):
            continue
        if not isinstance(elapsed, (int, float)) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(elapsed)) or not math.isfinite(float(value)):
            continue
        series[(workload, arm, repetition)].append((float(elapsed), float(value)))
    for points in series.values():
        points.sort()
    return series


def _latest_value(
    points: Sequence[tuple[float, float]], elapsed: float
) -> float | None:
    value: float | None = None
    for timestamp, candidate in points:
        if timestamp > elapsed:
            break
        value = candidate
    return value


def _time_grid(trajectory: Sequence[dict[str, object]], points: int) -> list[float]:
    elapsed = [
        float(row["elapsed_seconds"])
        for row in trajectory
        if isinstance(row.get("elapsed_seconds"), (int, float))
        and math.isfinite(float(row["elapsed_seconds"]))
    ]
    if not elapsed:
        return []
    maximum = max(elapsed)
    return [maximum * i / max(1, points - 1) for i in range(max(2, points))]


def suite_regret_curve(
    trajectory: Sequence[dict[str, object]], *, points: int = 121
) -> list[dict[str, object]]:
    """Suite median and IQR of trajectory regret, aggregated by workload."""
    arms = tuple(ARMS)
    run_series = _trajectory_by_run(trajectory, arms=arms)
    workloads = sorted({key[0] for key in run_series})
    grid = _time_grid(trajectory, points)
    rows: list[dict[str, object]] = []
    for arm in arms:
        for elapsed in grid:
            workload_regrets: list[float] = []
            for workload in workloads:
                candidates = [
                    _latest_value(series, elapsed)
                    for (w, a, _), series in run_series.items()
                    if w == workload and a == arm
                ]
                available = [value for value in candidates if value is not None]
                required = max(1, math.ceil(len(candidates) / 2))
                if len(available) < required:
                    continue
                workload_regrets.append(
                    max(0.0, (statistics.median(available) - 1.0) * 100.0)
                )
            if not workload_regrets:
                continue
            rows.append(
                {
                    "arm": arm,
                    "elapsed_seconds": elapsed,
                    "q25_pct": _quantile(workload_regrets, 0.25),
                    "median_pct": statistics.median(workload_regrets),
                    "q75_pct": _quantile(workload_regrets, 0.75),
                    "workloads": len(workload_regrets),
                }
            )
    return rows


def time_to_hit_curve(
    trajectory: Sequence[dict[str, object]],
    *,
    threshold_pct: float = 5.0,
    points: int = 121,
) -> list[dict[str, object]]:
    """CDF of workload-level median time to reach a trajectory-regret target."""
    arms = tuple(ARMS)
    run_series = _trajectory_by_run(trajectory, arms=arms)
    workloads = sorted({key[0] for key in run_series})
    grid = _time_grid(trajectory, points)
    target = 1.0 + threshold_pct / 100.0
    rows: list[dict[str, object]] = []
    for arm in arms:
        workload_hits: dict[str, float] = {}
        for workload in workloads:
            runs = [
                series
                for (w, a, _), series in run_series.items()
                if w == workload and a == arm
            ]
            hit_times = [
                next(
                    (elapsed for elapsed, ratio in series if ratio <= target), math.inf
                )
                for series in runs
            ]
            finite = [elapsed for elapsed in hit_times if math.isfinite(elapsed)]
            required = max(1, math.ceil(len(runs) / 2))
            workload_hits[workload] = (
                statistics.median(finite) if len(finite) >= required else math.inf
            )
        for elapsed in grid:
            reached = sum(hit <= elapsed for hit in workload_hits.values())
            rows.append(
                {
                    "arm": arm,
                    "elapsed_seconds": elapsed,
                    "coverage_pct": 100.0 * reached / len(workloads)
                    if workloads
                    else 0.0,
                    "threshold_pct": threshold_pct,
                }
            )
    return rows


def _workload_step_curve(
    trajectory: Sequence[dict[str, object]],
    workload: str,
    *,
    points: int = 101,
) -> list[dict[str, object]]:
    """Median best-latency step curve across repetitions for LFBO and RAG."""
    arms = tuple(ARMS)
    filtered = [row for row in trajectory if row.get("workload") == workload]
    run_series = _trajectory_by_run(
        filtered,
        arms=arms,
        workloads={workload},
        value_field="incumbent_best_perf",
    )
    grid = _time_grid(filtered, points)
    rows: list[dict[str, object]] = []
    for arm in arms:
        arm_series = [
            series
            for (w, a, _), series in run_series.items()
            if w == workload and a == arm
        ]
        for elapsed in grid:
            available = [
                value
                for series in arm_series
                if (value := _latest_value(series, elapsed)) is not None
            ]
            if len(available) < max(1, math.ceil(len(arm_series) / 2)):
                continue
            rows.append(
                {
                    "workload": workload,
                    "arm": arm,
                    "elapsed_seconds": elapsed,
                    "median_latency_ms": statistics.median(available),
                }
            )
    return rows


# ── Performance/readiness frontier ───────────────────────────────────────
_ELLIPSE_SEGMENTS = 72


def frontier_points(
    kernel_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Every (workload, arm) as one point in (latency, readiness) space.

    Both outcomes are strictly positive, so the frontier is summarized in log
    space; rows missing either coordinate are dropped rather than imputed.
    """
    points: list[dict[str, object]] = []
    for row in kernel_rows:
        latency = row.get("median_selected_latency_ms")
        readiness = row.get("median_readiness_seconds")
        if not isinstance(latency, (int, float)) or not isinstance(
            readiness, (int, float)
        ):
            continue
        if not (float(latency) > 0.0 and float(readiness) > 0.0):
            continue
        points.append(
            {
                "workload": str(row["workload"]),
                "arm": str(row["arm"]),
                "latency_ms": float(latency),
                "readiness_seconds": float(readiness),
            }
        )
    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    return sorted(
        points,
        key=lambda row: (str(row["workload"]), arm_order[str(row["arm"])]),
    )


def _covariance_axes(
    xs: Sequence[float], ys: Sequence[float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Scaled principal axes of the sample covariance of a 2-D point cloud.

    Returns the two axis vectors already scaled by one standard deviation, so an
    ellipse is ``centre + axis0 * cos(t) + axis1 * sin(t)``. A degenerate cloud
    (fewer than two points, or zero spread on an axis) yields zero-length axes,
    which collapses the ellipse onto its centre instead of producing NaNs.
    """
    count = len(xs)
    if count < 2:
        return (0.0, 0.0), (0.0, 0.0)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    var_x = sum((x - mean_x) ** 2 for x in xs) / (count - 1)
    var_y = sum((y - mean_y) ** 2 for y in ys) / (count - 1)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (count - 1)
    trace = var_x + var_y
    gap = math.sqrt(max(0.0, (var_x - var_y) ** 2 + 4.0 * cov**2))
    first = max(0.0, (trace + gap) / 2.0)
    second = max(0.0, (trace - gap) / 2.0)
    if abs(cov) < 1e-18:
        # Already axis-aligned; the eigenvectors are the coordinate axes.
        return (math.sqrt(var_x), 0.0), (0.0, math.sqrt(var_y))
    length = math.hypot(first - var_y, cov)
    unit = ((first - var_y) / length, cov / length)
    return (
        (unit[0] * math.sqrt(first), unit[1] * math.sqrt(first)),
        (-unit[1] * math.sqrt(second), unit[0] * math.sqrt(second)),
    )


def frontier_strategy_summaries(
    points: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Per-arm centroid, one-sigma dispersion ellipse, and Pareto membership.

    The centroid is the geometric mean of each outcome (the ellipse is fitted in
    log space, so exponentiating keeps every vertex positive and matches the
    log-scaled axes the figure uses). ``pareto`` marks the arm centroids that no
    other arm beats on both latency and readiness at once.
    """
    by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in points:
        by_arm[str(row["arm"])].append(row)

    summaries: list[dict[str, object]] = []
    for arm in ARMS:
        arm_points = by_arm.get(arm)
        if not arm_points:
            continue
        log_latency = [math.log(float(row["latency_ms"])) for row in arm_points]
        log_readiness = [
            math.log(float(row["readiness_seconds"])) for row in arm_points
        ]
        centre = (
            sum(log_latency) / len(log_latency),
            sum(log_readiness) / len(log_readiness),
        )
        axis0, axis1 = _covariance_axes(log_latency, log_readiness)
        ellipse = []
        for step in range(_ELLIPSE_SEGMENTS + 1):
            angle = 2.0 * math.pi * step / _ELLIPSE_SEGMENTS
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            ellipse.append(
                (
                    math.exp(centre[0] + axis0[0] * cos_a + axis1[0] * sin_a),
                    math.exp(centre[1] + axis0[1] * cos_a + axis1[1] * sin_a),
                )
            )
        summaries.append(
            {
                "arm": arm,
                "latency_ms": math.exp(centre[0]),
                "readiness_seconds": math.exp(centre[1]),
                "kernels": len(arm_points),
                "ellipse": ellipse,
            }
        )

    for row in summaries:
        row["pareto"] = not any(
            other is not row
            and other["latency_ms"] <= row["latency_ms"]
            and other["readiness_seconds"] <= row["readiness_seconds"]
            and (
                other["latency_ms"] < row["latency_ms"]
                or other["readiness_seconds"] < row["readiness_seconds"]
            )
            for other in summaries
        )
    return summaries


# ── Figures (gnuplot; skipped gracefully when data is too sparse) ─────────
def _gnuplot_quote(path: Path) -> str:
    return str(path).replace("'", "''")


def _render(program: str, output: Path, width: float, height: float) -> None:
    term = {
        "svg": f"set terminal svg size {round(width * 240)},{round(height * 240)} font 'DejaVu Sans,20' noenhanced background rgb 'white'",
        "pdf": f"set terminal pdfcairo size {width:.2f}in,{height:.2f}in font 'DejaVu Sans,9' noenhanced color",
        "png": f"set terminal pngcairo size {round(width * 300)},{round(height * 300)} font 'DejaVu Sans,24' noenhanced background rgb 'white'",
    }[output.suffix.removeprefix(".")]
    full = f"{term}\nset output '{_gnuplot_quote(output)}'\n{program}\n"
    completed = subprocess.run(
        ["gnuplot"], input=full, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"gnuplot failed for {output}: {completed.stderr.strip()}")


def _render_all(
    program: str, figure_dir: Path, stem: str, width: float, height: float
) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    produced: list[str] = []
    for suffix in ("svg", "pdf", "png"):
        output = figure_dir / f"{stem}.{suffix}"
        _render(program, output, width, height)
        produced.append(output.name)
    return produced


def _aggregate_effects_figure(
    aggregate: dict[str, object], figure_dir: Path
) -> list[str]:
    """Bar chart of geometric-mean ratios vs LFBO for both outcomes."""
    contrasts = [c for c in aggregate["contrasts"] if c["baseline_display"]]
    if not contrasts:
        return []
    lines = ["# arm outcome ratio low high"]
    labels: list[str] = []
    idx = 0
    tics: list[str] = []
    for outcome in OUTCOMES:
        for c in contrasts:
            if c["outcome"] != outcome:
                continue
            estimate = c["geometric_mean_ratio"]
            if not isinstance(estimate, (int, float)) or not math.isfinite(estimate):
                continue
            low = c["ci_low"] if math.isfinite(c["ci_low"]) else estimate
            high = c["ci_high"] if math.isfinite(c["ci_high"]) else estimate
            lines.append(f"{idx} {estimate} {low} {high}")
            tics.append(f"'{c['numerator_arm']}\\n{outcome.split('_')[0]}' {idx}")
            labels.append(str(c["numerator_arm"]))
            idx += 1
    if idx == 0:
        return []
    data = figure_dir / "01_aggregate_effects.dat"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data.write_text("\n".join(lines) + "\n", encoding="utf-8")
    program = (
        "set style data histograms\nset style fill solid 0.7 border -1\n"
        "set ylabel 'geometric-mean ratio vs LFBO (lower is better)'\n"
        "set yrange [0:*]\nset grid ytics\n"
        f"set xtics ({', '.join(tics)}) rotate by -30\n"
        "set arrow from graph 0, first 1 to graph 1, first 1 nohead lc rgb '#000000' dt 2\n"
        f"plot '{_gnuplot_quote(data)}' using 1:3:4:xtic(1) with yerrorbars pt 7 lc rgb '#0072B2' notitle, "
        f"'{_gnuplot_quote(data)}' using 1:2 with points pt 7 ps 1.5 lc rgb '#D55E00' notitle"
    )
    return _render_all(program, figure_dir, "01_aggregate_effects", 6.5, 4.2)


def _trajectory_trend_figure(
    trajectory: Sequence[dict[str, object]],
    figure_dir: Path,
    *,
    x_field: str,
    stem: str,
    xlabel: str,
) -> list[str]:
    """Per-arm median normalized-incumbent curve vs elapsed time or eval index."""
    # Aggregate: within (arm, x) take the median normalized ratio across runs.
    by_arm: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in trajectory:
        x = row.get(x_field)
        y = row.get("normalized_incumbent_ratio")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        by_arm[str(row["arm"])][round(float(x), 3)].append(float(y))
    series: dict[str, list[tuple[float, float]]] = {}
    for arm, buckets in by_arm.items():
        points = sorted((x, statistics.median(ys)) for x, ys in buckets.items())
        if points:
            series[arm] = points
    if not series:
        return []
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_parts: list[str] = []
    for arm in ARMS:
        if arm not in series:
            continue
        data = figure_dir / f"{stem}_{arm}.dat"
        data.write_text(
            "\n".join(f"{x} {y}" for x, y in series[arm]) + "\n", encoding="utf-8"
        )
        color = ARM_COLORS.get(arm, "#333333")
        plot_parts.append(
            f"'{_gnuplot_quote(data)}' using 1:2 with steps lw 2 lc rgb '{color}' title '{arm}'"
        )
    program = (
        f"set xlabel '{xlabel}'\nset ylabel 'normalized incumbent (>=1, lower better)'\n"
        "set logscale y\nset grid\nset key top right\n"
        "plot " + ", ".join(plot_parts)
    )
    return _render_all(program, figure_dir, stem, 6.5, 4.2)


def _short_workload(workload: str) -> str:
    suffix = workload.rsplit("-", 1)[-1]
    if workload.startswith("matmul_split_k-"):
        return f"Split-K K{suffix.split('x')[1]}"
    if workload.startswith("matmul-"):
        return f"MatMul {suffix.split('x')[0]}"
    if workload.startswith("fp8_attention-"):
        return f"FP8 attn S{suffix.split('x')[2]}"
    if workload.startswith("attention-"):
        return f"Attn S{suffix.split('x')[2]}"
    if workload.startswith("grouped_gemm-"):
        return f"Grouped {suffix.split('m')[0].upper()}"
    if workload.startswith("swiglu-"):
        return f"SwiGLU {suffix.split('x')[0]}"
    if workload.startswith("softmax-"):
        return f"Softmax N{suffix.split('x')[1]}"
    if workload.startswith("rms_norm-"):
        return f"RMS N{suffix.split('x')[1]}"
    if workload.startswith("rope-"):
        return f"RoPE S{suffix.split('x')[3]}"
    sequence = re.search(r"s(\d+)ds", suffix)
    if workload.startswith("gdn_fwd_h-"):
        return f"GDN S{sequence.group(1) if sequence else suffix}"
    if workload.startswith("mamba2_chunk_scan-"):
        return f"Mamba S{sequence.group(1) if sequence else suffix}"
    return workload


def _search_overhead_figure(
    runs: Sequence[dict[str, object]],
    kernel_rows: Sequence[dict[str, object]],
    figure_dir: Path,
) -> list[str]:
    selected = poster_workloads(kernel_rows)
    if not selected:
        return []
    lines = ["# index lfbo llm hybrid rag workload"]
    evaluation_gms: dict[str, list[float]] = defaultdict(list)
    readiness_gms: dict[str, list[float]] = defaultdict(list)
    for index, workload in enumerate(selected):
        values: dict[str, tuple[float, float]] = {}
        for arm in ARMS:
            valid = [
                row
                for row in runs
                if row.get("workload") == workload
                and row.get("arm") == arm
                and _is_valid(row)
            ]
            evaluations = statistics.median(
                float(row["evaluation_count"])
                for row in valid
                if isinstance(row.get("evaluation_count"), (int, float))
            )
            readiness = statistics.median(
                float(row["readiness_seconds"])
                for row in valid
                if isinstance(row.get("readiness_seconds"), (int, float))
            )
            values[arm] = (evaluations, readiness)
            evaluation_gms[arm].append(evaluations)
            readiness_gms[arm].append(readiness)
        lines.append(
            f"{index} {values[ARM_LFBO][0]} {values['llm'][0]} "
            f"{values['hybrid_lfbo_llm'][0]} {values['contextual_rag_llm'][0]} "
            f'"{_short_workload(workload)}"'
        )
    data = figure_dir / "04_search_overhead.dat"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config_summary = "/".join(f"{_geomean(evaluation_gms[arm]):.1f}" for arm in ARMS)
    time_summary = "/".join(f"{_geomean(readiness_gms[arm]):.1f}" for arm in ARMS)
    title = (
        "Search overhead (LFBO/LLM/Hybrid/RAG): "
        f"config GMs {config_summary}; readiness GMs {time_summary}s"
    )
    program = (
        "set style data histograms\nset style histogram clustered gap 1\n"
        "set style fill solid 0.86 border -1\nset boxwidth 0.85\n"
        f"set title '{title}'\nset ylabel 'median evaluated configurations'\n"
        "set yrange [0:*]\nset grid ytics\nset key top right horizontal\n"
        "set xtics rotate by -45 right font ',10'\nset bmargin 9\n"
        f"plot '{_gnuplot_quote(data)}' using 2:xtic(6) lc rgb '{POSTER_LFBO_COLOR}' title 'LFBO', "
        f"'' using 3 lc rgb '{POSTER_LLM_COLOR}' title 'LLM', "
        f"'' using 4 lc rgb '{POSTER_HYBRID_COLOR}' title 'Hybrid', "
        f"'' using 5 lc rgb '{POSTER_RAG_COLOR}' title 'RAG-LLM'"
    )
    return _render_all(program, figure_dir, "04_search_overhead", 9.0, 5.5)


def _write_step_series(
    figure_dir: Path,
    stem: str,
    rows: Sequence[dict[str, object]],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for arm in ARMS:
        path = figure_dir / f"{stem}_{arm}.dat"
        arm_rows = [row for row in rows if row.get("arm") == arm]
        path.write_text(
            "\n".join(
                f"{row['elapsed_seconds']} {row['median_latency_ms']}"
                for row in arm_rows
            )
            + "\n",
            encoding="utf-8",
        )
        paths[arm] = path
    return paths


def _cold_start_figure(
    trajectory: Sequence[dict[str, object]],
    kernel_rows: Sequence[dict[str, object]],
    figure_dir: Path,
) -> list[str]:
    available = {str(row["workload"]) for row in kernel_rows}
    workload = (
        "matmul-8192x8192x8192"
        if "matmul-8192x8192x8192" in available
        else poster_workloads(kernel_rows, limit=1)[0]
    )
    rows = _workload_step_curve(trajectory, workload)
    if not rows:
        return []
    paths = _write_step_series(figure_dir, "05_cold_start_step", rows)
    plot_parts = [
        f"'{_gnuplot_quote(paths[arm])}' using 1:2 with steps lw 3 "
        f"lc rgb '{POSTER_ARM_COLORS[arm]}' title '{POSTER_ARM_LABELS[arm]}'"
        for arm in ARMS
    ]
    program = (
        f"set title 'Cold-start search trajectory: {_short_workload(workload)}'\n"
        "set xlabel 'wall-clock search time (s)'\n"
        "set ylabel 'median best latency found (ms)'\nset grid\nset key top right\n"
        "plot " + ", ".join(plot_parts)
    )
    return _render_all(program, figure_dir, "05_cold_start_step", 7.0, 4.5)


def _per_kernel_ratio_figure(
    kernel_rows: Sequence[dict[str, object]], figure_dir: Path
) -> list[str]:
    ratios = poster_performance_ratios(kernel_rows)
    if not ratios:
        return []
    workloads = list(dict.fromkeys(str(row["workload"]) for row in ratios))
    workload_index = {workload: index for index, workload in enumerate(workloads)}
    offsets = {"llm": -0.22, "hybrid_lfbo_llm": 0.0, "contextual_rag_llm": 0.22}
    color_ids = {"llm": 1, "hybrid_lfbo_llm": 2, "contextual_rag_llm": 3}
    lines = ["# y ratio color"]
    for row in ratios:
        arm = str(row["arm"])
        y = workload_index[str(row["workload"])] + offsets[arm]
        lines.append(f"{y} {row['ratio']} {color_ids[arm]}")
    data = figure_dir / "06_per_kernel_ratio.dat"
    data.write_text("\n".join(lines) + "\n", encoding="utf-8")
    low = min(float(row["ratio"]) for row in ratios) * 0.85
    high = max(float(row["ratio"]) for row in ratios) * 1.15
    y_tics = ", ".join(
        f"'{_short_workload(workload)}' {index}"
        for index, workload in enumerate(workloads)
    )
    program = (
        "set title 'Per-kernel selected-latency ratios versus LFBO'\n"
        "set xlabel 'arm / LFBO latency ratio (log scale; <1 beats LFBO)'\n"
        f"set logscale x 2\nset xrange [{low}:{high}]\n"
        f"set yrange [{len(workloads) - 0.5}:-0.5]\nset grid xtics\n"
        f"set ytics ({y_tics}) font ',9'\nset key top right horizontal\n"
        f"set linetype 1 lc rgb '{POSTER_LLM_COLOR}'\n"
        f"set linetype 2 lc rgb '{POSTER_HYBRID_COLOR}'\n"
        f"set linetype 3 lc rgb '{POSTER_RAG_COLOR}'\n"
        "set arrow 1 from 1,graph 0 to 1,graph 1 nohead lw 2 lc rgb '#222222'\n"
        "set arrow 2 from 0.975,graph 0 to 0.975,graph 1 nohead dt 2 lc rgb '#666666'\n"
        "set arrow 3 from 1.025,graph 0 to 1.025,graph 1 nohead dt 2 lc rgb '#666666'\n"
        "set lmargin 20\n"
        f"plot '{_gnuplot_quote(data)}' using (1):1:($2-1):(0):3 with vectors nohead lw 3 lc variable notitle, "
        "'' using 2:1:3 with points pt 7 ps 0.55 lc variable notitle, "
        f"1/0 with points pt 7 lc rgb '{POSTER_LLM_COLOR}' title 'LLM', "
        f"1/0 with points pt 7 lc rgb '{POSTER_HYBRID_COLOR}' title 'Hybrid', "
        f"1/0 with points pt 7 lc rgb '{POSTER_RAG_COLOR}' title 'RAG-LLM'"
    )
    return _render_all(program, figure_dir, "06_per_kernel_ratio", 8.0, 9.0)


def _suite_regret_figure(
    trajectory: Sequence[dict[str, object]], figure_dir: Path
) -> list[str]:
    rows = suite_regret_curve(trajectory)
    if not rows:
        return []
    paths: dict[str, Path] = {}
    for arm in ARMS:
        path = figure_dir / f"07_suite_regret_{arm}.dat"
        selected = [row for row in rows if row["arm"] == arm]
        path.write_text(
            "\n".join(
                f"{row['elapsed_seconds']} {row['q25_pct']} {row['median_pct']} "
                f"{row['q75_pct']} {row['workloads']}"
                for row in selected
            )
            + "\n",
            encoding="utf-8",
        )
        paths[arm] = path
    total_workloads = max(int(row["workloads"]) for row in rows)
    full_suite_times = [
        next(
            float(row["elapsed_seconds"])
            for row in rows
            if row["arm"] == arm and int(row["workloads"]) == total_workloads
        )
        for arm in ARMS
    ]
    x_max = max(full_suite_times) * 1.1 if full_suite_times else 0.0
    ribbons = [
        f"'{_gnuplot_quote(paths[arm])}' using 1:2:4 with filledcurves "
        f"lc rgb '{POSTER_ARM_COLORS[arm]}' fs transparent solid 0.10 notitle"
        for arm in ARMS
    ]
    medians = [
        f"'{_gnuplot_quote(paths[arm])}' using 1:3 with steps lw 3 "
        f"lc rgb '{POSTER_ARM_COLORS[arm]}' title '{POSTER_ARM_LABELS[arm]} median (IQR)'"
        for arm in ARMS
    ]
    program = (
        "set title 'Suite-wide normalized search regret'\n"
        "set xlabel 'wall-clock search time (s)'\nset ylabel 'oracle regret (%)'\n"
        f"set xrange [0:{x_max}]\nset yrange [0:*]\nset grid\nset key top right\n"
        "set arrow from graph 0,first 5 to graph 1,first 5 nohead dt 2 lc rgb '#555555'\n"
        "plot " + ", ".join(ribbons + medians)
    )
    return _render_all(program, figure_dir, "07_suite_regret", 7.0, 4.5)


def _time_to_hit_figure(
    trajectory: Sequence[dict[str, object]], figure_dir: Path
) -> list[str]:
    rows = time_to_hit_curve(trajectory)
    if not rows:
        return []
    paths: dict[str, Path] = {}
    for arm in ARMS:
        path = figure_dir / f"08_time_to_hit_{arm}.dat"
        selected = [row for row in rows if row["arm"] == arm]
        path.write_text(
            "\n".join(
                f"{row['elapsed_seconds']} {row['coverage_pct']}" for row in selected
            )
            + "\n",
            encoding="utf-8",
        )
        paths[arm] = path
    final_coverage = {
        arm: float([row for row in rows if row["arm"] == arm][-1]["coverage_pct"])
        for arm in ARMS
    }
    change_times: list[float] = []
    for arm in ARMS:
        previous: float | None = None
        for row in (row for row in rows if row["arm"] == arm):
            current = float(row["coverage_pct"])
            if previous is not None and current != previous:
                change_times.append(float(row["elapsed_seconds"]))
            previous = current
    x_max = max(change_times) * 1.15 if change_times else 1.0
    title = "Final <=5% coverage (LFBO/LLM/Hybrid/RAG): " + "/".join(
        f"{final_coverage[arm]:.1f}%" for arm in ARMS
    )
    plot_parts = [
        f"'{_gnuplot_quote(paths[arm])}' using 1:2 with steps lw 3 "
        f"lc rgb '{POSTER_ARM_COLORS[arm]}' title '{POSTER_ARM_LABELS[arm]}'"
        for arm in ARMS
    ]
    program = (
        f"set title '{title}'\n"
        "set xlabel 'wall-clock search time (s)'\n"
        "set ylabel 'workloads reaching <=5% regret (%)'\n"
        f"set xrange [0:{x_max}]\nset yrange [0:100]\nset grid\nset key bottom right\n"
        "plot " + ", ".join(plot_parts)
    )
    return _render_all(program, figure_dir, "08_time_to_hit", 7.0, 4.5)


def _small_multiples_figure(
    trajectory: Sequence[dict[str, object]],
    kernel_rows: Sequence[dict[str, object]],
    figure_dir: Path,
) -> list[str]:
    selected = poster_workloads(kernel_rows)
    if not selected:
        return []
    paths: dict[tuple[str, str], Path] = {}
    max_elapsed = 0.0
    for index, workload in enumerate(selected):
        rows = _workload_step_curve(trajectory, workload, points=81)
        for row in rows:
            max_elapsed = max(max_elapsed, float(row["elapsed_seconds"]))
        written = _write_step_series(figure_dir, f"09_small_{index:02d}", rows)
        for arm, path in written.items():
            paths[workload, arm] = path
    columns = min(5, len(selected))
    rows_count = math.ceil(len(selected) / columns)
    commands = [
        f"set multiplot layout {rows_count},{columns} rowsfirst title 'Per-workload cold-start trajectories'",
        f"set xrange [0:{max_elapsed}]",
        "set grid",
        "set xlabel 'time (s)' font ',7'",
        "set ylabel 'best ms' font ',7'",
        "set xtics font ',6'",
        "set ytics font ',6'",
    ]
    for index, workload in enumerate(selected):
        commands.append(f"set title '{_short_workload(workload)}' font ',7'")
        commands.append("set key top right font ',6'" if index == 0 else "unset key")
        commands.append(
            "plot "
            + ", ".join(
                f"'{_gnuplot_quote(paths[workload, arm])}' using 1:2 with steps lw 2 "
                f"lc rgb '{POSTER_ARM_COLORS[arm]}' title '{POSTER_ARM_LABELS[arm]}'"
                for arm in ARMS
            )
        )
    commands.append("unset multiplot")
    return _render_all("\n".join(commands), figure_dir, "09_small_multiples", 12.0, 8.0)


def _frontier_scatter_figure(
    kernel_rows: Sequence[dict[str, object]], figure_dir: Path
) -> list[str]:
    """Readiness-vs-latency frontier: every kernel, per-arm centroids, Pareto path.

    Supersedes the hand-written pareto gnuplot: ranges, ellipses, and the frontier
    polyline are all derived from the data rather than pinned to one campaign.
    """
    points = frontier_points(kernel_rows)
    summaries = frontier_strategy_summaries(points)
    if not summaries:
        return []

    parts: list[str] = []
    for arm in ARMS:
        arm_points = [row for row in points if row["arm"] == arm]
        if not arm_points:
            continue
        scatter = figure_dir / f"10_frontier_points_{arm}.dat"
        scatter.write_text(
            "\n".join(
                f"{row['readiness_seconds']} {row['latency_ms']}" for row in arm_points
            )
            + "\n",
            encoding="utf-8",
        )
        parts.append(
            f"'{_gnuplot_quote(scatter)}' using 1:2 with points pt 6 ps 0.5 "
            f"lc rgb '{POSTER_ARM_COLORS[arm]}' notitle"
        )

    for row in summaries:
        arm = str(row["arm"])
        ellipse = figure_dir / f"10_frontier_ellipse_{arm}.dat"
        ellipse.write_text(
            "\n".join(f"{x} {y}" for x, y in row["ellipse"]) + "\n", encoding="utf-8"
        )
        parts.append(
            f"'{_gnuplot_quote(ellipse)}' using 2:1 with lines lw 2 dt 3 "
            f"lc rgb '{POSTER_ARM_COLORS[arm]}' notitle"
        )

    frontier = sorted(
        (row for row in summaries if row["pareto"]),
        key=lambda row: float(row["readiness_seconds"]),
    )
    if len(frontier) > 1:
        path = figure_dir / "10_frontier_path.dat"
        path.write_text(
            "\n".join(
                f"{row['readiness_seconds']} {row['latency_ms']}" for row in frontier
            )
            + "\n",
            encoding="utf-8",
        )
        parts.append(
            f"'{_gnuplot_quote(path)}' using 1:2 with lines lw 3 dt 2 "
            f"lc rgb '{POSTER_TIE_COLOR}' title 'Pareto frontier'"
        )

    for row in summaries:
        arm = str(row["arm"])
        centroid = figure_dir / f"10_frontier_centroid_{arm}.dat"
        centroid.write_text(
            f"{row['readiness_seconds']} {row['latency_ms']}\n", encoding="utf-8"
        )
        marker = 7 if row["pareto"] else 6
        parts.append(
            f"'{_gnuplot_quote(centroid)}' using 1:2 with points pt {marker} ps 2.4 "
            f"lc rgb '{POSTER_ARM_COLORS[arm]}' title '{POSTER_ARM_LABELS[arm]}'"
        )

    program = (
        "set title 'Performance-readiness frontier "
        "(geometric-mean centroids; lower-left is better)'\n"
        "set xlabel 'readiness time (s)'\nset ylabel 'selected latency (ms)'\n"
        "set logscale xy\nset grid\nset key outside right top\n"
        "plot " + ", ".join(parts)
    )
    return _render_all(program, figure_dir, "10_frontier_scatter", 8.0, 5.5)


def generate_figures(
    aggregate: dict[str, object],
    trajectory: Sequence[dict[str, object]],
    figure_dir: Path,
    *,
    runs: Sequence[dict[str, object]] = (),
    kernel_rows: Sequence[dict[str, object]] = (),
) -> dict[str, list[str]]:
    """Render the available figures; skip (with a note) when gnuplot is absent."""
    produced: dict[str, list[str]] = {}
    if shutil.which("gnuplot") is None:
        return {"_skipped": ["gnuplot not found; figures not rendered"]}
    for name, builder in (
        ("aggregate_effects", lambda: _aggregate_effects_figure(aggregate, figure_dir)),
        (
            "trajectory_walltime",
            lambda: _trajectory_trend_figure(
                trajectory,
                figure_dir,
                x_field="elapsed_seconds",
                stem="02_trajectory_walltime",
                xlabel="elapsed seconds",
            ),
        ),
        (
            "trajectory_candidates",
            lambda: _trajectory_trend_figure(
                trajectory,
                figure_dir,
                x_field="benchmarked_index",
                stem="03_trajectory_candidates",
                xlabel="benchmarked configurations",
            ),
        ),
        (
            "search_overhead",
            lambda: _search_overhead_figure(runs, kernel_rows, figure_dir),
        ),
        (
            "cold_start_step",
            lambda: _cold_start_figure(trajectory, kernel_rows, figure_dir),
        ),
        (
            "per_kernel_ratio",
            lambda: _per_kernel_ratio_figure(kernel_rows, figure_dir),
        ),
        (
            "suite_regret",
            lambda: _suite_regret_figure(trajectory, figure_dir),
        ),
        (
            "time_to_hit",
            lambda: _time_to_hit_figure(trajectory, figure_dir),
        ),
        (
            "small_multiples",
            lambda: _small_multiples_figure(trajectory, kernel_rows, figure_dir),
        ),
        (
            "frontier_scatter",
            lambda: _frontier_scatter_figure(kernel_rows, figure_dir),
        ),
    ):
        try:
            files = builder()
        except RuntimeError as error:
            produced[name] = [f"error: {error}"]
            continue
        produced[name] = files or ["skipped: insufficient data"]
    return produced


# ── Markdown all-arm table ───────────────────────────────────────────────
def _all_arm_markdown(kernel_rows: Sequence[dict[str, object]]) -> str:
    header = (
        "| kernel | arm | correct | latency (ms) | regret % | regret/LFBO % "
        "| wall (s) | configs | in tok | out tok |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for row in kernel_rows:
        lines.append(
            f"| {row['workload']} | {row['arm']} | {row['n_correct']}/{row['n_runs']} "
            f"| {_fmt(row['median_selected_latency_ms'])} "
            f"| {_fmt(row['regret_pct'])} "
            f"| {_fmt(row['regret_vs_lfbo_pct'])} "
            f"| {_fmt(row['median_readiness_seconds'])} "
            f"| {_fmt(row['median_attempts'])} "
            f"| {_fmt(row['input_tokens_total'])} "
            f"| {_fmt(row['output_tokens_total'])} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.4g}"
    return "-"


# ── Driver ───────────────────────────────────────────────────────────────
def analyze(campaign: Path) -> dict[str, object]:
    runs = load_runs(campaign)
    analysis_dir = campaign / "analysis"
    figure_dir = campaign / "figures"

    tidy = tidy_rows(runs)
    _write_csv(analysis_dir / "per_run.csv", tidy, list(tidy[0].keys()))

    kernel_rows = per_kernel_arm(runs)
    _write_csv(
        analysis_dir / "per_kernel_arm.csv", kernel_rows, list(kernel_rows[0].keys())
    )
    _write_csv(
        analysis_dir / "all_arm_table.csv", kernel_rows, list(kernel_rows[0].keys())
    )
    (analysis_dir / "all_arm_table.md").write_text(
        _all_arm_markdown(kernel_rows), encoding="utf-8"
    )

    arm_summary = per_arm_summary(runs)
    _write_csv(
        analysis_dir / "per_arm_summary.csv", arm_summary, list(arm_summary[0].keys())
    )

    aggregate = aggregate_statistics(runs)
    (analysis_dir / "aggregate_statistics.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        analysis_dir / "aggregate_statistics.csv",
        aggregate["contrasts"],
        list(aggregate["contrasts"][0].keys()) if aggregate["contrasts"] else [],
    )

    reliability = reliability_rows(runs)
    _write_csv(
        analysis_dir / "reliability.csv", reliability, list(reliability[0].keys())
    )
    cost = cost_rows(runs)
    _write_csv(analysis_dir / "cost.csv", cost, list(cost[0].keys()))

    trajectory = trajectory_rows(campaign, runs)
    traj_fields = [
        "workload",
        "repetition",
        "arm",
        "stage",
        "candidate_source",
        "elapsed_seconds",
        "benchmarked_index",
        "internal_performance",
        "incumbent_best_perf",
        "normalized_incumbent_ratio",
    ]
    _write_csv(analysis_dir / "trajectory_long.csv", trajectory, traj_fields)

    figures = generate_figures(
        aggregate, trajectory, figure_dir, runs=runs, kernel_rows=kernel_rows
    )

    return {
        "runs": len(runs),
        "kernels": len({r["workload"] for r in runs}),
        "trajectory_rows": len(trajectory),
        "figures": figures,
        "outputs": sorted(str(p.relative_to(campaign)) for p in analysis_dir.glob("*")),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args(argv)
    summary = analyze(Path(args.campaign))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
