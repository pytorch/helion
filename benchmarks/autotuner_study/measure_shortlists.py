"""Head-to-head measurement of every run's finalist shortlist.

For each run under the given campaign roots, reconstruct the configs the
final-verification shootout chose between (the top-N configs by best observed
in-search timing, plus the selected config), dedupe them per kernel case, and
time all of them together with interleaved_bench on the current GPU. Combined
with the per-candidate CSVs this lets diagnose.py decompose each run's final
quality gap into selection error (the shootout picked the wrong finalist) and
exploration error (no finalist was near the case best).

Run one instance per GPU with a --cases filter matching the campaign's
case->GPU pinning so timings stay apples-to-apples with the runs.

The reconstruction approximates the runtime shootout: --top defaults to the
Triton finalist count (8; CuTe uses 32), ranks by best observed in-search
perf rather than the runtime's deduped finalist-history perf, and does not
model pinned finalists, so diagnose.py's oracle is an approximation of the
set the shootout actually timed.

Usage:
    CUDA_VISIBLE_DEVICES=3 python benchmarks/autotuner_study/measure_shortlists.py \
        --roots /tmp/autotuner_study/round2/baseline --out shortlists_gpu3.json \
        [--cases a,b] [--top 10] [--repeat 150]
"""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import json
import math
import operator
from pathlib import Path
import sys
from typing import Any
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_shortlist(run_dir: Path, top: int) -> tuple[str, dict[str, Any]] | None:
    """Return (case, {selected_config, ranked: [(config_repr, best_ms)]})."""
    summary_path = run_dir / "summary.json"
    csv_path = run_dir / "autotune.csv"
    if not summary_path.exists() or not csv_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    best_by_repr: dict[str, float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok" or not row["perf_ms"]:
                continue
            perf = float(row["perf_ms"])
            if not math.isfinite(perf):
                continue
            config_repr = row["config"]
            if perf < best_by_repr.get(config_repr, math.inf):
                best_by_repr[config_repr] = perf
    ranked = sorted(best_by_repr.items(), key=operator.itemgetter(1))[:top]
    return summary["case"], {
        "selected_config": summary["selected_config"],
        "ranked": ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default=None)
    # Default matches _FINAL_REBENCHMARK_TOP_K_DEFAULT (Triton).
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    from kernels import KERNEL_CASES  # pyrefly: ignore [missing-import]
    from kernels import load_kernel  # pyrefly: ignore [missing-import]
    import torch

    import helion
    from helion.autotuner.benchmarking import interleaved_bench

    wanted = set(args.cases.split(",")) if args.cases else None

    # case -> canonical config json -> {"config": dict, "refs": {run_id: ref}}
    pool: dict[str, dict[str, dict[str, Any]]] = {}

    def add_ref(
        case: str, config: dict[str, Any], run_id: str, ref: dict[str, Any]
    ) -> None:
        key = json.dumps(config, sort_keys=True, default=str)
        entry = pool.setdefault(case, {}).setdefault(
            key, {"config": config, "refs": {}}
        )
        entry["refs"].setdefault(run_id, {}).update(ref)

    for root in [Path(r) for r in args.roots]:
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            loaded = run_shortlist(run_dir, args.top)
            if loaded is None:
                continue
            case, data = loaded
            if wanted is not None and case not in wanted:
                continue
            run_id = f"{root.name}/{run_dir.name}"
            add_ref(case, data["selected_config"], run_id, {"selected": True})
            for rank, (config_repr, best_ms) in enumerate(data["ranked"]):
                config = eval(config_repr, {"Config": helion.Config})
                add_ref(
                    case,
                    config.config,
                    run_id,
                    {"rank": rank, "observed_ms": best_ms},
                )

    device = torch.device("cuda")
    results: dict[str, Any] = {}
    for case_name, entries in sorted(pool.items()):
        case = KERNEL_CASES[case_name]
        kernel = load_kernel(case)
        kernel_args = case.make_args(device)
        bound = kernel.bind(kernel_args)  # pyrefly: ignore [missing-attribute]
        default_config = bound.config_spec.default_config().config
        add_ref(case_name, default_config, "<default>", {})
        keys = list(entries.keys())
        fns: list[Callable[[], object]] = []
        keep: list[str] = []
        for key in keys:
            try:
                compiled = bound.compile_config(helion.Config(**entries[key]["config"]))
                fns.append(functools.partial(compiled, *kernel_args))
                keep.append(key)
            except Exception as e:  # compile failures are data, not fatal
                print(f"{case_name}: compile failed ({e}) for {entries[key]['refs']}")
        for fn in fns:
            fn()
        torch.cuda.synchronize()
        print(f"{case_name}: timing {len(fns)} unique configs")
        # Two independent interleaved passes: their per-config disagreement is
        # the measurement noise floor diagnose.py uses to separate real
        # selection regret from phantom regret on near-tied finalists.
        timings_a = interleaved_bench(fns, repeat=args.repeat)
        timings_b = interleaved_bench(fns, repeat=args.repeat)
        case_result = []
        for key, perf_a, perf_b in zip(keep, timings_a, timings_b, strict=True):
            case_result.append(
                {
                    "perf_ms": (perf_a + perf_b) / 2,
                    "perf_ms_a": perf_a,
                    "perf_ms_b": perf_b,
                    "config": entries[key]["config"],
                    "refs": entries[key]["refs"],
                }
            )
        case_result.sort(key=operator.itemgetter("perf_ms"))
        results[case_name] = case_result
        best = case_result[0]["perf_ms"] if case_result else float("nan")
        print(f"{case_name}: best {best:.5f} ms over {len(case_result)} configs")
        del kernel_args, fns, bound
        gc.collect()
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
