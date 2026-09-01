"""Decompose per-run final-quality gaps into selection vs exploration error.

Joins campaign run directories with the head-to-head shortlist measurements
from measure_shortlists.py. For every run:

  * selected_ratio   = h2h(selected config) / h2h(case best config)
  * oracle_ratio     = min over the run's finalist shortlist of h2h / case best
  * selection_regret = selected_ratio / oracle_ratio
        (>1 means the final shootout picked the wrong finalist under noise)
  * exploration gap  = oracle_ratio - 1.0
        (>0 means nothing near the case best ever reached the shortlist)
  * saw_best         = whether the case-best config was *evaluated at all*
        during the run (exact config match over the full per-candidate CSV)

Each case must appear in exactly one shortlists file (the per-GPU split
produced by measure_shortlists.py); duplicate cases across files are not
merged - the last file wins.

Usage:
    python benchmarks/autotuner_study/diagnose.py \
        --roots /tmp/autotuner_study/round2/baseline \
        --shortlists shortlists_gpu1.json shortlists_gpu2.json ... [--json out.json]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def canonical(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def load_shortlists(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """case -> {perf_by_key, best_ms, best_key, refs_by_run}."""
    cases: dict[str, dict[str, Any]] = {}
    for path in paths:
        for case, entries in json.loads(path.read_text()).items():
            perf_by_key: dict[str, float] = {}
            refs_by_run: dict[str, list[tuple[str, dict[str, Any]]]] = (
                collections.defaultdict(list)
            )
            spreads: list[float] = []
            for entry in entries:
                key = canonical(entry["config"])
                perf_by_key[key] = entry["perf_ms"]
                if "perf_ms_a" in entry:
                    spreads.append(
                        abs(entry["perf_ms_a"] - entry["perf_ms_b"]) / entry["perf_ms"]
                    )
                for run_id, ref in entry["refs"].items():
                    refs_by_run[run_id].append((key, ref))
            best_key = min(perf_by_key, key=perf_by_key.get)  # type: ignore[arg-type]
            spreads.sort()
            noise_floor = (
                spreads[min(len(spreads) - 1, int(0.9 * len(spreads)))]
                if spreads
                else 0.0
            )
            cases[case] = {
                "perf_by_key": perf_by_key,
                "best_ms": perf_by_key[best_key],
                "best_key": best_key,
                "noise_floor": noise_floor,
                "refs_by_run": dict(refs_by_run),
            }
    return cases


def run_evaluated_keys(run_dir: Path) -> set[str]:
    """Canonical keys of every config that got a terminal ok/fail row."""
    import helion

    csv_path = run_dir / "autotune.csv"
    reprs: set[str] = set()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["status"] != "started":
                reprs.add(row["config"])
    keys: set[str] = set()
    for config_repr in reprs:
        config = eval(config_repr, {"Config": helion.Config})
        keys.add(canonical(config.config))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--shortlists", nargs="+", required=True)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    cases = load_shortlists([Path(p) for p in args.shortlists])

    rows: list[dict[str, Any]] = []
    for root in [Path(r) for r in args.roots]:
        for run_dir in sorted(root.iterdir()):
            summary_path = run_dir / "summary.json"
            spec_path = run_dir / "spec.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
            case = summary["case"]
            if case not in cases:
                continue
            info = cases[case]
            run_id = f"{root.name}/{run_dir.name}"
            refs = info["refs_by_run"].get(run_id, [])
            selected_keys = [k for k, ref in refs if ref.get("selected")]
            shortlist_keys = [k for k, ref in refs if "rank" in ref]
            if not selected_keys:
                continue
            best_ms = info["best_ms"]
            selected_ms = info["perf_by_key"][selected_keys[0]]
            oracle_candidates = [
                info["perf_by_key"][k] for k in {*selected_keys, *shortlist_keys}
            ]
            oracle_ms = min(oracle_candidates)
            saw_best = info["best_key"] in run_evaluated_keys(run_dir)
            rows.append(
                {
                    "run_id": run_id,
                    "case": case,
                    "algorithm": (spec.get("algorithm") or "default")
                    + (f"-{spec['tag']}" if spec.get("tag") else ""),
                    "seed": spec.get("seed"),
                    "selected_ratio": selected_ms / best_ms,
                    "oracle_ratio": oracle_ms / best_ms,
                    "selection_regret": selected_ms / oracle_ms,
                    "noise_floor": info["noise_floor"],
                    "real_regret": (selected_ms / oracle_ms - 1.0)
                    > max(0.01, info["noise_floor"]),
                    "saw_best": saw_best,
                }
            )

    by_group: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(
        list
    )
    for row in rows:
        by_group[(row["case"], row["algorithm"])].append(row)

    header = (
        f"{'case':<20}{'algorithm':<24}{'n':>3}{'sel mean':>9}{'sel max':>9}"
        f"{'orc mean':>9}{'orc max':>9}{'noise':>7}{'regret':>7}{'saw_best':>9}"
    )
    print(header)
    agg: dict[str, list[float]] = collections.defaultdict(list)
    for (case, algorithm), group in sorted(by_group.items()):
        sel = [g["selected_ratio"] for g in group]
        orc = [g["oracle_ratio"] for g in group]
        regret = sum(g["real_regret"] for g in group)
        saw = sum(g["saw_best"] for g in group)
        print(
            f"{case:<20}{algorithm:<24}{len(group):>3}"
            f"{statistics.mean(sel):>9.3f}{max(sel):>9.3f}"
            f"{statistics.mean(orc):>9.3f}{max(orc):>9.3f}"
            f"{group[0]['noise_floor']:>7.3f}"
            f"{regret:>7}{saw:>6}/{len(group)}"
        )
        agg[algorithm + "|sel"].extend(sel)
        agg[algorithm + "|orc"].extend(orc)
        agg[algorithm + "|regret"].extend(g["selection_regret"] for g in group)

    print("\noverall by algorithm:")
    algorithms = sorted({key.split("|")[0] for key in agg})
    for algorithm in algorithms:
        sel = agg[algorithm + "|sel"]
        orc = agg[algorithm + "|orc"]
        regret = agg[algorithm + "|regret"]
        print(
            f"  {algorithm:<24} n={len(sel):<4} selected mean {statistics.mean(sel):.3f}"
            f" max {max(sel):.3f} | oracle mean {statistics.mean(orc):.3f}"
            f" max {max(orc):.3f} | mean regret {statistics.mean(regret):.3f}"
        )

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
