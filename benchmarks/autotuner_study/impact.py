"""Per-config-key performance impact analysis from campaign logs.

For each kernel case, pools every (config, perf) observation across all runs
(joining autotune.csv rows to the .meta.jsonl config map), then finds matched
pairs: configs identical except for exactly one config key. The distribution
of perf ratios within those pairs measures how much that key matters for that
kernel, which drives the impact-tier classification of config knobs.

Usage:
    python benchmarks/autotuner_study/impact.py --root /tmp/autotuner_study/audit
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


def canon(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def frac_gt(ratios: list[float], threshold: float) -> float:
    return sum(r > threshold for r in ratios) / len(ratios)


def load_case_observations(
    root: Path,
) -> dict[str, dict[str, tuple[dict[str, Any], float]]]:
    """case -> config_id -> (config dict, best finite perf observed)."""
    cases: dict[str, dict[str, tuple[dict[str, Any], float]]] = collections.defaultdict(
        dict
    )
    for run_dir in sorted(root.iterdir()):
        summary_path = run_dir / "summary.json"
        meta_path = run_dir / "autotune.meta.jsonl"
        csv_path = run_dir / "autotune.csv"
        if not (summary_path.exists() and meta_path.exists() and csv_path.exists()):
            continue
        case = json.loads(summary_path.read_text())["case"]
        configs: dict[str, dict[str, Any]] = {}
        with meta_path.open() as f:
            for line in f:
                record = json.loads(line)
                configs.update(record.get("configs", {}))
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if row["status"] != "ok" or not row["perf_ms"]:
                    continue
                config_id = row["config_id"]
                config = configs.get(config_id)
                if config is None:
                    continue
                perf = float(row["perf_ms"])
                prev = cases[case].get(config_id)
                if prev is None or perf < prev[1]:
                    cases[case][config_id] = (config, perf)
    return cases


def matched_pair_impact(
    observations: dict[str, tuple[dict[str, Any], float]],
) -> dict[str, list[float]]:
    """key -> list of pairwise max/min perf ratios among configs differing only in key."""
    all_keys: set[str] = set()
    for config, _perf in observations.values():
        all_keys.update(config)
    impact: dict[str, list[float]] = collections.defaultdict(list)
    for key in sorted(all_keys):
        buckets: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
        for config, perf in observations.values():
            if not math.isfinite(perf):
                continue
            rest = canon({k: v for k, v in config.items() if k != key})
            buckets[rest].append((canon(config.get(key)), perf))
        for group in buckets.values():
            by_value: dict[str, float] = {}
            for value, perf in group:
                if value not in by_value or perf < by_value[value]:
                    by_value[value] = perf
            if len(by_value) < 2:
                continue
            perfs = sorted(by_value.values())
            impact[key].append(perfs[-1] / perfs[0])
    return impact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    cases = load_case_observations(Path(args.root))
    result: dict[str, Any] = {}
    agg: dict[str, list[float]] = collections.defaultdict(list)
    for case, observations in sorted(cases.items()):
        impact = matched_pair_impact(observations)
        print(f"\n=== {case} ({len(observations)} unique configs) ===")
        print(
            f"{'key':<28}{'pairs':>7}{'median':>9}{'p90':>9}{'max':>9}{'>1%':>7}{'>5%':>7}{'>20%':>7}"
        )
        rows = []
        for key, ratios in sorted(
            impact.items(), key=lambda kv: -statistics.median(kv[1])
        ):
            ratios.sort()
            n = len(ratios)
            median = statistics.median(ratios)
            p90 = ratios[min(n - 1, int(0.9 * n))]
            print(
                f"{key:<28}{n:>7}{median:>9.3f}{p90:>9.3f}{ratios[-1]:>9.2f}"
                f"{frac_gt(ratios, 1.01):>7.0%}{frac_gt(ratios, 1.05):>7.0%}{frac_gt(ratios, 1.20):>7.0%}"
            )
            rows.append(
                {
                    "key": key,
                    "pairs": n,
                    "median": median,
                    "p90": p90,
                    "max": ratios[-1],
                    "frac_gt_1pct": frac_gt(ratios, 1.01),
                    "frac_gt_5pct": frac_gt(ratios, 1.05),
                    "frac_gt_20pct": frac_gt(ratios, 1.20),
                }
            )
            agg[key].extend(ratios)
        result[case] = rows

    print("\n=== aggregate across cases ===")
    print(
        f"{'key':<28}{'pairs':>7}{'median':>9}{'p90':>9}{'>1%':>7}{'>5%':>7}{'>20%':>7}"
    )
    for key, ratios in sorted(agg.items(), key=lambda kv: -statistics.median(kv[1])):
        ratios.sort()
        n = len(ratios)
        p90 = ratios[min(n - 1, int(0.9 * n))]
        print(
            f"{key:<28}{n:>7}{statistics.median(ratios):>9.3f}{p90:>9.3f}"
            f"{frac_gt(ratios, 1.01):>7.0%}{frac_gt(ratios, 1.05):>7.0%}{frac_gt(ratios, 1.20):>7.0%}"
        )

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
