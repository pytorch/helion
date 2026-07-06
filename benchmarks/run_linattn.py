"""Linear-attention benchmark runner: Helion vs FLA.

Collects the per-variant Helion-vs-FLA timings the example suite already
measures (each examples/linear/example_*.py exposes a benchmark() that returns
its rows) and writes them in the dashboard's helionbench.json schema. These
kernels have no TritonBench operator, so FLA is the baseline; forward and
forward+backward each become a dashboard row (the latter as "<variant>-bwd",
matching the -bwd convention in benchmarks/run.py).

Usage:
    python -m benchmarks.run_linattn --output helionbench.json
    python -m benchmarks.run_linattn --kernel simple_gla,full_gla
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from typing import Any

import torch

# (name, B, T, H, D); D is both the query/key and the value dim (D = DV).
# The six production shapes from flash-linear-attention/benchmarks/ops/registry.
SHAPES: list[tuple[str, int, int, int, int]] = [
    ("B1_T8192_H96_D128", 1, 8192, 96, 128),
    ("B2_T16384_H16_D128", 2, 16384, 16, 128),
    ("B4_T2048_H16_D128", 4, 2048, 16, 128),
    ("B4_T4096_H64_D128", 4, 4096, 64, 128),
    ("B8_T2048_H32_D256", 8, 2048, 32, 256),
    ("B8_T1024_H8_D64", 8, 1024, 8, 64),
]

# variant name -> example module under examples.linear. Each module's
# benchmark(configs) returns (cfg, helion_fwd_ms, fla_fwd_ms, helion_fb_ms,
# fla_fb_ms) rows.
VARIANTS = [
    "vanilla_linear_attn",
    "simple_gla",
    "retention",
    "full_gla",
    "delta_rule",
    "gated_delta_rule",
    "kda",
]

# benchmark() takes (B, H, T, D, DV); our shapes use D == DV.
_CONFIGS = [(name, b, h, t, d, d) for (name, b, t, h, d) in SHAPES]


def write_results_json(
    output: str,
    results: dict[str, list[tuple[Any, ...]]],
    verdicts: dict[str, list[tuple[bool, bool]]],
) -> None:
    """Emit the dashboard's helionbench.json schema: one record per
    (model, metric) with parallel shape and benchmark_values arrays. Forward and
    forward+backward are separate models ("<variant>" and "<variant>-bwd").

    helion_accuracy carries each shape's pass/fail (forward vs the recurrent
    reference, backward vs the chunked reference); latency/speedup are emitted
    unconditionally. The dashboard reads accuracy == 0 with speedup > 0 as "ran
    but wrong" and accuracy == 0 with no speedup as "did not run", so perf must
    stay raw even when a shape fails."""
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    records: list[dict[str, Any]] = []

    def add_metric(
        model: str, metric: str, shapes: list[str], values: list[float]
    ) -> None:
        if not shapes or not values:
            return
        records.append(
            {
                "benchmark": {
                    "name": "Helion Linear-Attention Benchmark",
                    "extra_info": {"device": device, "backend": "triton"},
                },
                "model": {"name": model},
                "metric": {"name": metric, "benchmark_values": values},
                "shape": shapes,
            }
        )

    for variant, rows in results.items():
        if not rows:
            continue
        labels = [s[0] for s in SHAPES[: len(rows)]]
        vrd = verdicts.get(variant, [])
        fwd_ok = [1.0 if i < len(vrd) and vrd[i][0] else 0.0 for i in range(len(rows))]
        bwd_ok = [1.0 if i < len(vrd) and vrd[i][1] else 0.0 for i in range(len(rows))]
        # Forward: Helion accuracy + latency + speedup vs FLA; FLA reported at
        # 1.0 so the dashboard's Triton column shows the comparison baseline.
        add_metric(variant, "helion_accuracy", labels, fwd_ok)
        add_metric(variant, "helion_latency_ms", labels, [r[1] for r in rows])
        add_metric(
            variant,
            "helion_speedup",
            labels,
            [r[2] / r[1] if r[1] else 0.0 for r in rows],
        )
        add_metric(variant, "triton_speedup", labels, [1.0 for _ in rows])
        # Forward+backward as a separate "<variant>-bwd" row.
        bwd = f"{variant}-bwd"
        add_metric(bwd, "helion_accuracy", labels, bwd_ok)
        add_metric(bwd, "helion_latency_ms", labels, [r[3] for r in rows])
        add_metric(
            bwd, "helion_speedup", labels, [r[4] / r[3] if r[3] else 0.0 for r in rows]
        )
        add_metric(bwd, "triton_speedup", labels, [1.0 for _ in rows])

    if os.path.exists(output):
        try:
            with open(output) as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    records = existing + records
        except (OSError, json.JSONDecodeError):
            pass

    with open(output, "w") as f:
        json.dump(records, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel",
        default=",".join(VARIANTS),
        help="comma-separated variant names (default: all)",
    )
    parser.add_argument(
        "--num-shapes", type=int, default=None, help="cap number of shapes"
    )
    parser.add_argument(
        "--output", default=None, help="write helionbench.json to this path"
    )
    args = parser.parse_args()

    names = [
        n.strip().removesuffix("-bwd") for n in args.kernel.split(",") if n.strip()
    ]
    names = list(dict.fromkeys(names))
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown kernels: {unknown}; choose from {VARIANTS}")

    configs = _CONFIGS if args.num_shapes is None else _CONFIGS[: args.num_shapes]
    bench_configs = [c[1:] for c in configs]  # drop the label for benchmark()

    results: dict[str, list[tuple[Any, ...]]] = {}
    verdicts: dict[str, list[tuple[bool, bool]]] = {}
    for name in names:
        print(f"=== {name} ===")
        mod = importlib.import_module(f"examples.linear.example_{name}")
        verdicts[name] = mod.accuracy(bench_configs)
        results[name] = mod.benchmark(bench_configs)

    if args.output:
        write_results_json(args.output, results, verdicts)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
