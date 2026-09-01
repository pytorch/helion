"""Build the final old-vs-new comparison table for the autotuner study.

Quality comes from the head-to-head winners measurement (measure_winners.py
output): each run's selected config was re-timed in one interleaved batch per
case, removing cross-process measurement bias. Cost comes from each run's
per-candidate CSV (unique config_ids). Wall time comes from run summaries
(only comparable for sequentially executed campaigns).

Usage:
    python benchmarks/autotuner_study/final_table.py \
        --winners /tmp/autotuner_study/winners.json \
        --roots /tmp/autotuner_study/audit /tmp/autotuner_study/v2 ...
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


def load_runs(roots: list[Path]) -> list[dict[str, Any]]:
    runs = []
    for root in roots:
        for run_dir in sorted(root.iterdir()):
            summary_path = run_dir / "summary.json"
            csv_path = run_dir / "autotune.csv"
            spec_path = run_dir / "spec.json"
            if not summary_path.exists() or not csv_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
            unique = set()
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    if row["status"] != "started":
                        unique.add(row["config_id"])
            label = (spec.get("algorithm") or "default") + (
                f"-{spec['tag']}" if spec.get("tag") else ""
            )
            runs.append(
                {
                    "root": root.name,
                    "run_id": f"{root.name}/{run_dir.name}",
                    "case": summary["case"],
                    "label": label,
                    "config_key": json.dumps(
                        summary["selected_config"], sort_keys=True, default=str
                    ),
                    "n_unique": len(unique),
                    "wall_time_s": summary.get("wall_time_s"),
                }
            )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winners", required=True)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--labels", default=None, help="comma-separated label filter")
    args = parser.parse_args()

    winners = json.loads(Path(args.winners).read_text())
    # case -> config_key -> head-to-head measured perf
    measured: dict[str, dict[str, float]] = {}
    case_best: dict[str, float] = {}
    for case, entries in winners.items():
        table = {}
        for entry in entries:
            key = json.dumps(entry["config"], sort_keys=True, default=str)
            table[key] = entry["perf_ms"]
        measured[case] = table
        case_best[case] = min(table.values())

    runs = load_runs([Path(r) for r in args.roots])
    wanted = set(args.labels.split(",")) if args.labels else None

    # per label per case: quality ratios (head-to-head), evals
    by_label: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    missing = 0
    for run in runs:
        if wanted is not None and run["label"] not in wanted:
            continue
        table = measured.get(run["case"])
        if table is None:
            continue
        perf = table.get(run["config_key"])
        if perf is None:
            missing += 1
            continue
        run["h2h_perf"] = perf
        run["quality"] = perf / case_best[run["case"]]
        by_label[run["label"]][run["case"]].append(run)
    if missing:
        print(f"note: {missing} runs' configs missing from winners measurement")

    labels = sorted(by_label)
    cases = sorted({c for label in labels for c in by_label[label]})
    print("\nPer-case quality (head-to-head perf / case best; mean over seeds):")
    header = f"{'case':<20}" + "".join(f"{label:>22}" for label in labels)
    print(header)
    for case in cases:
        row = f"{case:<20}"
        for label in labels:
            runs_here = by_label[label].get(case, [])
            if runs_here:
                q = statistics.mean(r["quality"] for r in runs_here)
                e = statistics.mean(r["n_unique"] for r in runs_here)
                row += f"{q:>14.3f} ({e:>4.0f})"
            else:
                row += f"{'-':>22}"
        print(row)

    print("\nAggregate (geomean quality, mean/median evals, worst-case quality):")
    for label in labels:
        quals = []
        evals = []
        worst = []
        for case in cases:
            runs_here = by_label[label].get(case, [])
            if not runs_here:
                continue
            quals.append(statistics.mean(r["quality"] for r in runs_here))
            worst.append(max(r["quality"] for r in runs_here))
            evals.extend(r["n_unique"] for r in runs_here)
        if not quals:
            continue
        geo = math.exp(sum(math.log(q) for q in quals) / len(quals))
        geo_worst = math.exp(sum(math.log(q) for q in worst) / len(worst))
        print(
            f"  {label:<28} quality {geo:.3f}  worst-seed {geo_worst:.3f}  "
            f"evals mean {statistics.mean(evals):.0f} median {statistics.median(evals):.0f}"
            f"  ({len(quals)} cases)"
        )


if __name__ == "__main__":
    main()
