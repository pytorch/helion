"""Head-to-head re-measurement of autotuning winners for the final report.

Collects the selected config from every completed run under one or more
campaign roots, dedups them per kernel case, compiles each, and times them
together with interleaved_bench on the current GPU (run with
CUDA_VISIBLE_DEVICES=<idle gpu>, one instance at a time). This removes
run-to-run measurement bias so final-quality comparisons across algorithms
and campaigns are apples-to-apples.

Usage:
    CUDA_VISIBLE_DEVICES=3 python benchmarks/autotuner_study/measure_winners.py \
        --roots /tmp/autotuner_study/audit /tmp/autotuner_study/proto1 \
        --out /tmp/autotuner_study/winners.json [--cases a,b] [--repeat 200]
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import operator
from pathlib import Path
import sys
from typing import Any
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def collect_configs(roots: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    """case -> canonical config json -> {config, runs: [run ids]}."""
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for root in roots:
        for run_dir in sorted(root.iterdir()):
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            case = summary["case"]
            config = summary["selected_config"]
            key = json.dumps(config, sort_keys=True, default=str)
            entry = by_case.setdefault(case, {}).setdefault(
                key, {"config": config, "runs": []}
            )
            entry["runs"].append(f"{root.name}/{run_dir.name}")
    return by_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default=None)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    from kernels import KERNEL_CASES  # pyrefly: ignore [missing-import]
    from kernels import load_kernel  # pyrefly: ignore [missing-import]
    import torch

    import helion
    from helion.autotuner.benchmarking import interleaved_bench

    by_case = collect_configs([Path(r) for r in args.roots])
    wanted = set(args.cases.split(",")) if args.cases else None

    device = torch.device("cuda")
    results: dict[str, Any] = {}
    for case_name, entries in sorted(by_case.items()):
        if wanted is not None and case_name not in wanted:
            continue
        case = KERNEL_CASES[case_name]
        kernel = load_kernel(case)
        kernel_args = case.make_args(device)
        bound = kernel.bind(kernel_args)  # pyrefly: ignore [missing-attribute]
        configs = [entry["config"] for entry in entries.values()]
        labels = [entry["runs"] for entry in entries.values()]
        default_config = bound.config_spec.default_config().config
        default_key = json.dumps(default_config, sort_keys=True, default=str)
        if default_key in entries:
            entries[default_key]["runs"].append("<default>")
        else:
            configs.append(default_config)
            labels.append(["<default>"])
        fns: list[Callable[[], object]] = []
        keep: list[int] = []
        for i, config in enumerate(configs):
            try:
                compiled = bound.compile_config(helion.Config(**config))
                fns.append(functools.partial(compiled, *kernel_args))
                keep.append(i)
            except Exception as e:
                print(f"{case_name}: compile failed for {labels[i]}: {e}")
        # Warm up each once so compilation/caching is out of the timing loop.
        for fn in fns:
            fn()
        torch.cuda.synchronize()
        timings = interleaved_bench(fns, repeat=args.repeat)
        case_result = []
        for idx, timing in zip(keep, timings, strict=True):
            case_result.append(
                {
                    "runs": labels[idx],
                    "config": configs[idx],
                    "perf_ms": timing,
                }
            )
            print(f"{case_name}: {timing:.5f} ms  {labels[idx]}")
        results[case_name] = sorted(case_result, key=operator.itemgetter("perf_ms"))
        del kernel_args, fns, bound
        gc.collect()
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
