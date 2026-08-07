"""Aggregate leave-one-workload-out results and apply the preregistered criteria."""

from __future__ import annotations

import argparse
import json
import math
from math import comb
from pathlib import Path
import random
import statistics

# Preregistered default non-inferiority margins (design manifest overrides these).
_DEFAULT_MARGINS = {"perf_lfbo": 1.02, "time": 1.05}
_TIPPING_POINTS = (1.25, 1.5, 2.0)


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _paired_ratios(
    results: list[dict], numerator: str, denominator: str, field: str
) -> list[dict]:
    by_key = {
        (row["kernel"], row["workload_key"], row["rep"], row["arm"]): row
        for row in results
        if row.get("ok") and row.get(field) is not None
    }
    observations = []
    base_keys = {
        (kernel, workload, rep)
        for kernel, workload, rep, arm in by_key
        if arm == denominator
    }
    for kernel, workload, rep in sorted(base_keys):
        left = by_key.get((kernel, workload, rep, numerator))
        right = by_key.get((kernel, workload, rep, denominator))
        if not left or not right or not right[field] or left[field] <= 0:
            continue
        observations.append(
            {
                "kernel": kernel,
                "workload_key": workload,
                "rep": rep,
                "ratio": left[field] / right[field],
            }
        )
    return observations


def _workload_mean_logs(observations: list[dict]) -> dict[tuple[str, str], float]:
    """Average per-rep log-ratios within each (kernel, workload) — equal weight.

    Reps are collapsed *before* resampling so a workload with more surviving
    reps does not gain weight.
    """
    groups: dict[tuple[str, str], list[float]] = {}
    for row in observations:
        groups.setdefault((row["kernel"], row["workload_key"]), []).append(
            math.log(row["ratio"])
        )
    return {key: sum(logs) / len(logs) for key, logs in groups.items()}


def _cluster_interval(
    workload_logs: dict[tuple[str, str], float], *, samples: int, seed: int = 0
) -> dict:
    """Cluster bootstrap: resample kernels, then workloads within each kernel."""
    if not workload_logs:
        return {"estimate": None, "ci95": [None, None], "n": 0}
    estimate = math.exp(sum(workload_logs.values()) / len(workload_logs))
    by_kernel: dict[str, list[float]] = {}
    for (kernel, _workload), value in workload_logs.items():
        by_kernel.setdefault(kernel, []).append(value)
    rng = random.Random(seed)
    kernels = list(by_kernel)
    draws = []
    for _ in range(samples):
        values: list[float] = []
        for kernel in rng.choices(kernels, k=len(kernels)):
            workloads = by_kernel[kernel]
            values.extend(rng.choices(workloads, k=len(workloads)))
        if values:
            draws.append(math.exp(sum(values) / len(values)))
    return {
        "estimate": estimate,
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "n": len(workload_logs),
    }


def _axis_interval(
    rows: list[dict],
    candidate: str,
    baseline: str,
    field: str,
    *,
    samples: int,
    seed: int,
) -> dict:
    observations = _paired_ratios(rows, candidate, baseline, field)
    return _cluster_interval(
        _workload_mean_logs(observations), samples=samples, seed=seed
    )


def _completion_table(
    rows: list[dict], candidate: str, baseline: str
) -> tuple[dict[str, int], dict[tuple[str, str], str]]:
    """Workload-level four-outcome completion for a matched pair.

    A workload completes for an arm iff *every* required rep produced a valid
    ``perf_ms``; required reps are those seen for either arm (a wholly missing
    or timed-out candidate rep counts as a failure, never a silent drop).
    """

    def valid(row: dict) -> bool:
        return bool(row.get("ok")) and (row.get("perf_ms") or 0) > 0

    reps_by_workload: dict[tuple[str, str], set] = {}
    cells: dict[tuple, bool] = {}
    for row in rows:
        if row.get("arm") not in (candidate, baseline):
            continue
        workload = (row["kernel"], row["workload_key"])
        reps_by_workload.setdefault(workload, set()).add(row.get("rep"))
        cells[(workload, row["arm"], row.get("rep"))] = valid(row)

    def complete(workload: tuple[str, str], arm: str) -> bool:
        return all(
            cells.get((workload, arm, rep), False) for rep in reps_by_workload[workload]
        )

    table = {
        "both_complete": 0,
        "baseline_complete_candidate_failed": 0,
        "baseline_failed_candidate_complete": 0,
        "both_failed": 0,
    }
    per_workload: dict[tuple[str, str], str] = {}
    for workload in reps_by_workload:
        base_ok = complete(workload, baseline)
        cand_ok = complete(workload, candidate)
        if base_ok and cand_ok:
            outcome = "both_complete"
        elif base_ok and not cand_ok:
            outcome = "baseline_complete_candidate_failed"
        elif cand_ok and not base_ok:
            outcome = "baseline_failed_candidate_complete"
        else:
            outcome = "both_failed"
        table[outcome] += 1
        per_workload[workload] = outcome
    return table, per_workload


def _mcnemar_one_sided_p(b: int, c: int) -> float:
    """One-sided exact McNemar P(X>=b | Binomial(b+c, 0.5)) — excess cand failures."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(b, n + 1)) / (2**n)


def _completion_gate(table: dict[str, int]) -> dict:
    total = sum(table.values())
    candidate_complete = (
        table["both_complete"] + table["baseline_failed_candidate_complete"]
    )
    completion = candidate_complete / total if total else 0.0
    b = table["baseline_complete_candidate_failed"]
    c = table["baseline_failed_candidate_complete"]
    p_value = _mcnemar_one_sided_p(b, c)
    return {
        "candidate_completion": completion,
        "n": total,
        "mcnemar_p": p_value,
        "passes": completion >= 0.95 and p_value > 0.05,
        "table": table,
    }


def _tipping_points(
    perf_observations: list[dict],
    per_workload: dict[tuple[str, str], str],
    *,
    penalties: tuple[float, ...] = _TIPPING_POINTS,
) -> dict[str, float | None]:
    """Perf point estimate with candidate-only failures imputed at each penalty.

    Only ``baseline_complete_candidate_failed`` workloads are imputed — a failed
    baseline leaves the ratio undefined, so those cells never receive a
    reciprocal penalty.
    """
    base_logs = _workload_mean_logs(perf_observations)
    failed = [
        workload
        for workload, outcome in per_workload.items()
        if outcome == "baseline_complete_candidate_failed"
    ]
    out: dict[str, float | None] = {}
    for penalty in penalties:
        logs = dict(base_logs)
        for workload in failed:
            logs[workload] = math.log(penalty)
        out[str(penalty)] = math.exp(sum(logs.values()) / len(logs)) if logs else None
    return out


def _analyze_pair(
    rows: list[dict],
    candidate: str,
    baseline: str,
    *,
    margins: dict,
    samples: int,
    seed: int,
) -> dict:
    """Asymmetric verdict: perf must be non-inferior, autotune time superior.

    The asymmetry is deliberate. Warm-starting LFBO is a *cost* lever, so the
    claim it has to earn is a faster search; on kernel quality it only has to
    prove it does no harm, at the preregistered ``perf_lfbo`` margin.
    """
    perf_observations = _paired_ratios(rows, candidate, baseline, "perf_ms")
    perf = _cluster_interval(
        _workload_mean_logs(perf_observations), samples=samples, seed=seed
    )
    autotune = _axis_interval(
        rows, candidate, baseline, "autotune_time_s", samples=samples, seed=seed + 1
    )
    end_to_end = _axis_interval(
        rows, candidate, baseline, "end_to_end_s", samples=samples, seed=seed + 2
    )
    table, per_workload = _completion_table(rows, candidate, baseline)
    gate = _completion_gate(table)
    tipping = _tipping_points(perf_observations, per_workload)

    perf_upper = perf["ci95"][1]
    autotune_upper = autotune["ci95"][1]
    delta_perf = margins["perf_lfbo"]

    perf_non_inferior = perf_upper is not None and perf_upper <= delta_perf
    time_superior = autotune_upper is not None and autotune_upper < 1.0
    if gate["passes"] and perf_non_inferior:
        verdict = "effective" if time_superior else "non_inferior_only"
    else:
        verdict = "not_demonstrated"
    return {
        "candidate": candidate,
        "baseline": baseline,
        "perf_ratio": perf,
        "autotune_ratio": autotune,
        "end_to_end_ratio": end_to_end,
        "delta_perf": delta_perf,
        "completion": gate,
        "perf_tipping_points": tipping,
        "verdict": verdict,
    }


def analyze_workload_results(
    results: list[dict],
    oracles: list[dict],
    *,
    margins: dict | None = None,
    bootstrap_samples: int = 2000,
    pairs: dict[str, str] | None = None,
) -> dict:
    """Analyse the matched pair with the preregistered criteria."""
    margins = {**_DEFAULT_MARGINS, **(margins or {})}
    pairs = pairs or {"rag_lfbo": "lfbo"}
    # Evaluation cells only: preflight baselines never enter treatment estimates.
    eval_rows = [row for row in results if row.get("phase", "eval") == "eval"]
    deduplicated: dict = {}
    for row in eval_rows:
        key = row.get("resume_key") or (
            row.get("kernel"),
            row.get("workload_key"),
            row.get("arm"),
            row.get("rep"),
        )
        if key not in deduplicated or row.get("ok"):
            deduplicated[key] = row
    rows = list(deduplicated.values())
    oracle_by_workload = {
        row["workload_key"]: row["oracle_perf_ms"]
        for row in oracles
        if row.get("ok") and row.get("oracle_perf_ms")
    }
    # A RAG cell that Tier-0 matches (exact replay) means the held-out workload
    # was not truly held out -> leakage, regardless of the per-cell shape proxy.
    leakage_rows = [
        row
        for row in rows
        if row.get("heldout_shape_leaked")
        or (row.get("arm") in pairs and row.get("tier") == 0)
    ]
    tier0_rag_cells = sum(
        1 for row in rows if row.get("arm") in pairs and row.get("tier") == 0
    )
    same_kernel_rates = [
        row["same_kernel_neighbor_rate"]
        for row in rows
        if row.get("same_kernel_neighbor_rate") is not None
    ]
    tier1_rows = [
        row for row in rows if row.get("arm") in pairs and row.get("tier") == 1
    ]
    rag_rows = [row for row in rows if row.get("arm") in pairs]
    seed = 0
    analyses = {}
    for candidate, baseline in sorted(pairs.items()):
        analyses[candidate] = _analyze_pair(
            rows,
            candidate,
            baseline,
            margins=margins,
            samples=bootstrap_samples,
            seed=seed,
        )
        seed += 10
    return {
        "pairs": analyses,
        "margins": margins,
        "diagnostics": {
            "heldout_shape_leakage_count": len(leakage_rows),
            "tier0_rag_cells": tier0_rag_cells,
            "median_same_kernel_neighbor_rate": _median(same_kernel_rates),
            "tier1_coverage": (len(tier1_rows) / len(rag_rows)) if rag_rows else None,
            "oracle_workloads": len(oracle_by_workload),
        },
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _fmt_ci(interval: dict) -> str:
    estimate = interval.get("estimate")
    low, high = interval.get("ci95", [None, None])
    if estimate is None or low is None or high is None:
        return "n/a"
    return f"{estimate:.4f} [{low:.4f}, {high:.4f}] (n={interval.get('n')})"


def _markdown_workload(report: dict) -> str:
    diag = report["diagnostics"]
    lines = [
        "# Leave-one-workload-out results",
        "",
        "## Diagnostics",
        "",
        f"- Held-out shape leakage (per-cell proxy): "
        f"{diag['heldout_shape_leakage_count']}",
        f"- Median same-kernel neighbour rate: "
        f"{diag['median_same_kernel_neighbor_rate']}",
        f"- Tier-1 coverage: {diag['tier1_coverage']}",
        f"- Margins: {report['margins']}",
        "",
    ]
    for candidate, pair in report["pairs"].items():
        gate = pair["completion"]
        lines += [
            f"## {candidate} vs {pair['baseline']}",
            "",
            f"- Verdict: **{pair['verdict']}**",
            f"- perf ratio: {_fmt_ci(pair['perf_ratio'])} "
            f"(delta_perf={pair['delta_perf']})",
            f"- autotune-time ratio: {_fmt_ci(pair['autotune_ratio'])}",
            f"- end-to-end ratio: {_fmt_ci(pair['end_to_end_ratio'])}",
        ]
        lines += [
            f"- completion: {gate['candidate_completion']:.1%} "
            f"(McNemar p={gate['mcnemar_p']:.3f}, passes={gate['passes']})",
            f"- completion table: {gate['table']}",
            f"- perf tipping points: {pair['perf_tipping_points']}",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--oracles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--margins-json", type=Path, help="JSON margins from the design manifest"
    )
    args = parser.parse_args(argv)
    results = _load_jsonl(args.results)
    oracles = _load_jsonl(args.oracles)
    margins = (
        json.loads(args.margins_json.read_text(encoding="utf-8"))
        if args.margins_json
        else None
    )
    report = analyze_workload_results(results, oracles, margins=margins)
    markdown = _markdown_workload(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(report, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["pairs"], sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
