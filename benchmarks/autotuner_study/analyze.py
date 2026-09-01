"""Analyze autotuner study campaign output.

Reads run directories produced by campaign.py and reports:
  * per-run: unique candidates evaluated, best-so-far curve, failure mix
  * per case x algorithm: final quality (independent measurement), candidate
    counts, wall time, variance across seeds
  * evals-to-quality: unique candidates needed to get within X% of the best
    perf any run found for the case
  * per-config-key impact tiers from matched pairs (configs differing in
    exactly one key)

Usage:
    python benchmarks/autotuner_study/analyze.py --root /tmp/autotuner_study/audit [--json out.json]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

TERMINAL_OK = "ok"
NON_CANDIDATE_STATUSES = {"started"}


def load_run(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    spec_path = run_dir / "spec.json"
    csv_path = run_dir / "autotune.csv"
    if not summary_path.exists() or not csv_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}

    with csv_path.open() as f:
        rows: list[dict[str, str]] = list(csv.DictReader(f))

    # Terminal rows in order; unique candidate index assigned at first
    # terminal appearance of each config_id.
    seen: dict[str, int] = {}
    events: list[dict[str, Any]] = []  # per terminal row
    for row in rows:
        status = row["status"]
        if status in NON_CANDIDATE_STATUSES:
            continue
        config_id = row["config_id"]
        if config_id not in seen:
            seen[config_id] = len(seen)
        perf = float(row["perf_ms"]) if row["perf_ms"] else math.inf
        events.append(
            {
                "config_id": config_id,
                "unique_idx": seen[config_id],
                "generation": int(row["generation"] or 0),
                "status": status,
                "perf_ms": perf,
                "timestamp_s": float(row["timestamp_s"] or 0.0),
            }
        )

    # Best-so-far curve indexed by number of unique candidates evaluated.
    best_curve: list[tuple[int, float]] = []  # (n_unique_evaluated, best_perf)
    best = math.inf
    n_unique = 0
    for ev in events:
        n_unique = max(n_unique, ev["unique_idx"] + 1)
        if ev["status"] == TERMINAL_OK and ev["perf_ms"] < best:
            best = ev["perf_ms"]
            best_curve.append((n_unique, best))

    status_counts = collections.Counter(ev["status"] for ev in events)
    gen_counts = collections.Counter(ev["generation"] for ev in events)

    # Measurement-noise floor: spread among repeated ok-measurements of the
    # same config within this run.
    perfs_by_config: dict[str, list[float]] = collections.defaultdict(list)
    for ev in events:
        if ev["status"] == TERMINAL_OK and math.isfinite(ev["perf_ms"]):
            perfs_by_config[ev["config_id"]].append(ev["perf_ms"])
    noise_ratios = sorted(
        max(perfs) / min(perfs)
        for perfs in perfs_by_config.values()
        if len(perfs) >= 2 and min(perfs) > 0
    )

    # Phase attribution: best ok perf within the initial population (gen 0).
    gen0_best = min(
        (
            ev["perf_ms"]
            for ev in events
            if ev["generation"] == 0 and ev["status"] == TERMINAL_OK
        ),
        default=math.inf,
    )
    gen0_unique = len({ev["config_id"] for ev in events if ev["generation"] == 0})

    metrics = summary.get("metrics") or [{}]
    return {
        "run_id": run_dir.name,
        "case": summary["case"],
        "algorithm": (spec.get("algorithm") or "default")
        + (f"-{spec['tag']}" if spec.get("tag") else ""),
        "seed": int(summary.get("seed") or 0),
        "selected_perf_ms": summary.get("selected_perf_ms"),
        "default_perf_ms": summary.get("default_perf_ms"),
        "wall_time_s": summary.get("wall_time_s"),
        "n_unique": n_unique,
        "n_attempts": len(events),
        "status_counts": dict(status_counts),
        "gen_counts": dict(sorted(gen_counts.items())),
        "best_curve": best_curve,
        "final_search_best": best,
        "gen0_best": gen0_best,
        "gen0_unique": gen0_unique,
        "noise_ratio_p50": (
            noise_ratios[len(noise_ratios) // 2] if noise_ratios else None
        ),
        "noise_ratio_p90": (
            noise_ratios[min(len(noise_ratios) - 1, int(0.9 * len(noise_ratios)))]
            if noise_ratios
            else None
        ),
        "metrics": metrics[0] if metrics else {},
    }


def evals_to_reach(best_curve: list[tuple[int, float]], target: float) -> int | None:
    for n_unique, best in best_curve:
        if best <= target:
            return n_unique
    return None


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for run in runs:
        by_case[run["case"]].append(run)

    report: dict[str, Any] = {"cases": {}}
    for case, case_runs in sorted(by_case.items()):
        # Best independently-measured perf across all runs of this case.
        best_known = min(
            (r["selected_perf_ms"] for r in case_runs if r["selected_perf_ms"]),
            default=None,
        )
        case_report: dict[str, Any] = {"best_known_ms": best_known, "algorithms": {}}
        by_alg: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for run in case_runs:
            by_alg[run["algorithm"]].append(run)
        for alg, alg_runs in sorted(by_alg.items()):
            perfs = [r["selected_perf_ms"] for r in alg_runs if r["selected_perf_ms"]]
            uniques = [r["n_unique"] for r in alg_runs]
            walls = [r["wall_time_s"] for r in alg_runs if r["wall_time_s"]]
            # candidates needed to get search-best within 5% of case best-known
            reach5 = []
            if best_known:
                for r in alg_runs:
                    reach5.append(evals_to_reach(r["best_curve"], best_known * 1.05))
            # Fraction of the unique-candidate budget spent after the search
            # was already within 1% of its own final best (wasted tail).
            tails = []
            gen0_fracs = []
            for r in alg_runs:
                if r["best_curve"] and r["n_unique"]:
                    n_at = evals_to_reach(
                        r["best_curve"], r["final_search_best"] * 1.01
                    )
                    if n_at is not None:
                        tails.append(1.0 - n_at / r["n_unique"])
                if (
                    r["default_perf_ms"]
                    and math.isfinite(r["gen0_best"])
                    and math.isfinite(r["final_search_best"])
                    and r["default_perf_ms"] > r["final_search_best"]
                ):
                    total_gain = r["default_perf_ms"] - r["final_search_best"]
                    gen0_gain = max(0.0, r["default_perf_ms"] - r["gen0_best"])
                    gen0_fracs.append(min(1.0, gen0_gain / total_gain))
            case_report["algorithms"][alg] = {
                "n_runs": len(alg_runs),
                "tail_waste_mean": statistics.mean(tails) if tails else None,
                "gen0_gain_frac_mean": (
                    statistics.mean(gen0_fracs) if gen0_fracs else None
                ),
                "perf_ms_mean": statistics.mean(perfs) if perfs else None,
                "perf_ms_max": max(perfs) if perfs else None,
                "perf_ms_min": min(perfs) if perfs else None,
                "vs_best_known_mean": (
                    statistics.mean(p / best_known for p in perfs)
                    if perfs and best_known
                    else None
                ),
                "unique_mean": statistics.mean(uniques) if uniques else None,
                "wall_s_mean": statistics.mean(walls) if walls else None,
                "reach_5pct": reach5,
                "seeds": sorted(r["seed"] for r in alg_runs),
            }
        report["cases"][case] = case_report
    return report


def print_report(report: dict[str, Any]) -> None:
    for case, case_report in report["cases"].items():
        best_known = case_report["best_known_ms"]
        print(f"\n=== {case} (best known {fmt(best_known, 4)} ms) ===")
        header = (
            f"{'algorithm':<32}{'runs':>5}{'perf(ms) mean':>15}{'worst':>9}"
            f"{'vs best':>9}{'#unique':>9}{'wall(s)':>9}{'tail':>7}{'gen0':>7}  reach5%"
        )
        print(header)
        for alg, stats in case_report["algorithms"].items():
            reach = ",".join(str(n) if n else "-" for n in stats["reach_5pct"])
            print(
                f"{alg:<32}{stats['n_runs']:>5}"
                f"{fmt(stats['perf_ms_mean'], 4):>15}"
                f"{fmt(stats['perf_ms_max'], 4):>9}"
                f"{fmt(stats['vs_best_known_mean'], 3):>9}"
                f"{fmt(stats['unique_mean'], 0):>9}"
                f"{fmt(stats['wall_s_mean'], 0):>9}"
                f"{fmt(stats['tail_waste_mean'], 2):>7}"
                f"{fmt(stats['gen0_gain_frac_mean'], 2):>7}  {reach}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    runs = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        run = load_run(run_dir)
        if run is not None:
            runs.append(run)
    if not runs:
        print("no completed runs found", file=sys.stderr)
        return
    print(f"loaded {len(runs)} completed runs")

    report = summarize(runs)
    print_report(report)
    if args.json:
        report["runs"] = runs
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
