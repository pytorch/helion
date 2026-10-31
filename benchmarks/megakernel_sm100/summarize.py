"""Merge every system's result files into results/summary.{md,json}.

    python summarize.py [--results results]

A result file is JSON with ``system``, ``gpu`` and ``rows``; each row has a
``workload``, a ``case``, the implementation's name (``impl``) and its cold-L2
time ``median_us`` (per layer for the whole-model systems), and optionally the
profiler-free cross-check ``loop_us`` (see coldl2).  Repeated rounds of a system
(several files) are combined by their minimum per case: each system is credited
with each case's best round, so a baseline that varies between processes
(SGLang's startup autotune, its per-process PDL overlap) gets its best process.
The headline number of a workload with several routing cases is the mean over
its cases.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
SYSTEMS = ("helion", "vllm", "sglang", "mirage", "thunderkittens")
LABELS = {
    "helion": "Helion",
    "vllm": "vLLM",
    "sglang": "SGLang",
    "mirage": "Mirage",
    "thunderkittens": "ThunderKittens",
}
# Headline case per workload; None averages every case (routing cases).
HEADLINE = {
    "qwen3_8b_layer": "ctx1024",
    "llama32_1b_layer": "ctx1024",
    "qwen3_30b_a3b_moe": None,
    "deepseek_v3_mla": "ctx1024",
    "deepseek_v3_moe_tp4": None,
}
TITLES = {
    "qwen3_8b_layer": "Qwen3-8B decoder layer (BF16, ctx 1024)",
    "llama32_1b_layer": "Llama-3.2-1B decoder layer (BF16, ctx 1024)",
    "qwen3_30b_a3b_moe": "Qwen3-30B-A3B MoE block (BF16)",
    "deepseek_v3_mla": "DeepSeek-V3 MLA decode (16 heads, ctx 1024)",
    "deepseek_v3_moe_tp4": "DeepSeek-V3 NVFP4 MoE + all-reduce + RMSNorm (TP4)",
}


def _nested() -> dict:
    return defaultdict(lambda: defaultdict(lambda: defaultdict(list)))


def load(results: Path) -> tuple[dict, dict, dict, list[dict]]:
    """(times and loop times [workload][case][system] -> per-round values, impl
    names, run metadata)."""
    times, loops = _nested(), _nested()
    impls: dict = defaultdict(set)
    runs = []
    for path in sorted(results.glob("*.json")):
        if path.name.startswith("summary"):
            continue
        data = json.loads(path.read_text())
        if "rows" not in data:
            continue
        system = data["system"]
        runs.append(
            {
                "file": path.name,
                "system": system,
                "gpu": data.get("gpu"),
                "versions": data.get("versions"),
            }
        )
        for row in data["rows"]:
            times[row["workload"]][row["case"]][system].append(row["median_us"])
            if "loop_us" in row:
                loops[row["workload"]][row["case"]][system].append(row["loop_us"])
            impls[system].add(row["impl"])
    return times, loops, impls, runs


def combine(times: dict) -> tuple[dict, dict]:
    """(detail[workload][case][system], headline[workload][system]) in us."""
    detail = {
        workload: {
            case: {system: min(values) for system, values in by_system.items()}
            for case, by_system in cases.items()
        }
        for workload, cases in times.items()
    }
    headline = {}
    for workload, cases in detail.items():
        case = HEADLINE.get(workload)
        if case is not None:
            headline[workload] = dict(cases.get(case, {}))
            continue
        per_system = defaultdict(list)
        for by_system in cases.values():
            for system, value in by_system.items():
                per_system[system].append(value)
        headline[workload] = {
            system: statistics.fmean(values) for system, values in per_system.items()
        }
    return detail, headline


def _row(name: str, by_system: dict[str, float]) -> str:
    cells = [f"{by_system[s]:.2f}" if s in by_system else "-" for s in SYSTEMS]
    baselines = {s: v for s, v in by_system.items() if s != "helion"}
    if "helion" in by_system and baselines:
        best = min(baselines, key=lambda system: baselines[system])
        speedup = f"{baselines[best] / by_system['helion']:.2f}x vs {LABELS[best]}"
    else:
        speedup = "-"
    return f"| {name} | " + " | ".join(cells) + f" | {speedup} |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    cli = parser.parse_args()
    times, loops, impls, runs = load(cli.results)
    detail, headline = combine(times)
    _loop_detail, loop_headline = combine(loops)
    header = (
        "| Workload | " + " | ".join(LABELS[s] for s in SYSTEMS) + " | Helion speedup |"
    )
    rule = "|" + "---|" * (len(SYSTEMS) + 2)
    lines = [
        "# Cold-L2 latency on B200 (us, lower is better, best round)",
        "",
        "## Profiler-free (loop_us)",
        "",
        (
            "Each call's increment over a one-kernel graph in a back-to-back (reset, "
            "flush, call) loop timed with cudaEvents."
        ),
        "",
        header,
        rule,
    ]
    lines += [
        _row(TITLES.get(w, w), loop_headline[w]) for w in HEADLINE if w in loop_headline
    ]
    lines += [
        "",
        "## CUPTI span (first kernel start to last kernel end)",
        "",
        header,
        rule,
    ]
    lines += [_row(TITLES.get(w, w), headline[w]) for w in HEADLINE if w in headline]
    lines += [
        "",
        "## Every case (CUPTI span)",
        "",
        header.replace("Workload", "Workload / case"),
        rule,
    ]
    for workload in sorted(detail):
        for case in sorted(detail[workload], key=lambda c: (len(c), c)):
            lines.append(_row(f"{workload} / {case}", detail[workload][case]))
    lines += ["", "## Implementations", ""]
    lines += [
        f"- {LABELS.get(s, s)}: " + "; ".join(sorted(impls[s]))
        for s in SYSTEMS
        if s in impls
    ]
    lines += ["", f"{len(runs)} result files: " + ", ".join(r["file"] for r in runs)]
    (cli.results / "summary.md").write_text("\n".join(lines) + "\n")
    (cli.results / "summary.json").write_text(
        json.dumps(
            {
                "headline": headline,
                "loop_headline": loop_headline,
                "detail": detail,
                "runs": runs,
            },
            indent=1,
        )
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
