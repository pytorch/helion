"""Focused head-to-head: LFBO vs LLMSeededLFBO.

Reads results.jsonl produced by run_compare.py and writes a comparison table
of compile wall-clock and the resulting helion kernel ms — no torch / flex /
ref baselines, no speedup-vs-baseline numbers.

Usage:
    python benchmarks/llm_compare/summarize.py <results.jsonl> [--out report.md]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

LFBO_LABEL = "LFBO"
LLM_LABEL_PREFIX = "LLMSeededLFBO"


def load(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in jsonl_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def best_helion_ms(record: dict) -> float | None:
    timings = record.get("helion_timings") or {}
    if not timings:
        return None
    return min(timings.values())


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def geomean(values: list[float]) -> float | None:
    finite = [v for v in values if v is not None and v > 0 and math.isfinite(v)]
    if not finite:
        return None
    return math.exp(statistics.fmean(math.log(v) for v in finite))


def fmt_s(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "-"


def fmt_ms(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "-"


def fmt_x(value: float | None) -> str:
    return f"{value:.2f}×" if value is not None else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", type=Path, help="path to results.jsonl from run_compare.py"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="optional path to write the .md report"
    )
    args = parser.parse_args()

    rows = load(args.results)
    by_kernel: dict[str, dict[str, dict]] = {}
    for r in rows:
        label = r["autotuner_label"]
        bucket = LFBO_LABEL if label == LFBO_LABEL else LLM_LABEL_PREFIX
        by_kernel.setdefault(r["kernel"], {})[bucket] = r

    lines: list[str] = [
        "# LFBO vs LLM-seeded LFBO\n",
        (
            "Compile wall-clock = end-to-end subprocess time (Python startup + module "
            "load + autotune + correctness/benchmark). Helion ms = min helion impl time "
            "from the example's `run_example` table.\n"
        ),
        (
            "Ratios are LLM / LFBO: <1.0× means LLM-seeded is **faster** "
            "(less compile time, or faster tuned kernel).\n"
        ),
        "## Per-kernel comparison\n",
        (
            "| kernel | wall LFBO (s) | wall LLM (s) | wall ratio | "
            "helion LFBO (ms) | helion LLM (ms) | kernel ratio | LFBO exit | LLM exit |"
        ),
        "|" + "|".join(["---"] * 9) + "|",
    ]

    wall_ratios: list[float] = []
    kernel_ratios: list[float] = []
    for kernel in sorted(by_kernel):
        cells = by_kernel[kernel]
        lfbo = cells.get(LFBO_LABEL)
        llm = cells.get(LLM_LABEL_PREFIX)
        wall_lfbo = lfbo.get("wall_clock_s") if lfbo else None
        wall_llm = llm.get("wall_clock_s") if llm else None
        wall_r = ratio(wall_llm, wall_lfbo)
        ms_lfbo = best_helion_ms(lfbo) if lfbo else None
        ms_llm = best_helion_ms(llm) if llm else None
        ms_r = ratio(ms_llm, ms_lfbo)
        if wall_r is not None:
            wall_ratios.append(wall_r)
        if ms_r is not None:
            kernel_ratios.append(ms_r)
        lines.append(
            f"| {kernel} | {fmt_s(wall_lfbo)} | {fmt_s(wall_llm)} | "
            f"{fmt_x(wall_r)} | {fmt_ms(ms_lfbo)} | {fmt_ms(ms_llm)} | "
            f"{fmt_x(ms_r)} | "
            f"{lfbo['exit_code'] if lfbo else '-'} | "
            f"{llm['exit_code'] if llm else '-'} |"
        )

    lines.extend(
        [
            "\n## Aggregates (geomean of available ratios)\n",
            (
                f"- **Compile wall-clock ratio (LLM/LFBO)**: "
                f"{fmt_x(geomean(wall_ratios))}  ({len(wall_ratios)} kernels)"
            ),
            (
                f"- **Helion kernel ms ratio (LLM/LFBO)**: "
                f"{fmt_x(geomean(kernel_ratios))}  ({len(kernel_ratios)} kernels)"
            ),
        ]
    )

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"Wrote: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
