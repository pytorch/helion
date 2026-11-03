"""Merge the per-process run JSONs of mirage_layers.py (and ../thunderkittens/tk_llama_layers.py)
into one results JSON per workload:

  mirage_qwen3_8b_layer.json  mirage_llama32_1b_layer.json  mirage_qwen3_30b_moe.json
  mirage_dsv3_mla.json        tk_llama32_1b_layer.json

The per-layer cost is the least-squares slope of T(L) over L through every (L, instance median)
point of every process of the headline variant (the variant with the lowest per-layer cost among
those run in >= 2 processes whose output is right, i.e. the smallest-L models are within
CORRECT_REL_L2 of the PyTorch reference: the fairest choice for the baseline).  The file also carries the
"rows" list of the benchmark-wide schema (benchmarks/megakernel_sm100/summarize.py).
"""

from __future__ import annotations

import argparse
import collections
import json
import operator
from pathlib import Path
import statistics
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Sequence

WORKLOADS = {  # run workload -> (results file, schema workload, schema case, description)
    "qwen3_8b": (
        "qwen3_8b_layer",
        "qwen3_8b_layer",
        "ctx1024",
        "Qwen3-8B decoder layer, batch 1, bf16, decode at 1024 KV rows (position 1023)",
    ),
    "llama32_1b": (
        "llama32_1b_layer",
        "llama32_1b_layer",
        "ctx1024",
        "Llama-3.2-1B decoder layer, batch 1, bf16, decode at 1024 KV rows (position 1023)",
    ),
    "qwen3_30b_moe": (
        "qwen3_30b_moe",
        "qwen3_30b_a3b_moe",
        "b1",
        "Qwen3-30B-A3B MoE block (RMSNorm + router top-8 + experts + residual), batch 1, bf16",
    ),
    "dsv3_mla": (
        "dsv3_mla",
        "deepseek_v3_mla",
        "ctx1024",
        "DeepSeek-V3 absorbed MLA decode, 16 heads (TP8), 1 query token, 1024 KV rows, decode + reduce",
    ),
}
PREFIX = {"mirage_mpk": "mirage", "thunderkittens": "tk"}
NOTES = {}  # (system, workload) -> list of notes; filled from notes.json next to this file if present
# A config can be the headline only if its smallest-L models match the PyTorch reference: relative
# L2 error of the output below this (bf16 configs that compute the right thing are at 0.2-1.2%;
# attention over 1023 of the 1024 KV rows gives 4-5%).
CORRECT_REL_L2 = 0.025


def lstsq_slope(points: Sequence[tuple[float, float]]) -> float:
    xs, ys = [x for x, _ in points], [y for _, y in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)


def slope_se(points: Sequence[tuple[float, float]]) -> float:
    """Standard error of the least-squares slope (from the residuals)."""
    n = len(points)
    xs, ys = [x for x, _ in points], [y for _, y in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = lstsq_slope(points)
    a = my - b * mx
    sxx = sum((x - mx) ** 2 for x in xs)
    rss = sum((y - a - b * x) ** 2 for x, y in points)
    return (rss / (n - 2) / sxx) ** 0.5 if n > 2 else float("nan")


def summarize(system: str, workload: str, runs: list[dict], alternatives: dict) -> dict:
    Ls = sorted({int(L) for r in runs for L in r["T_by_layers"]})
    points = [
        (int(k.split("i")[0][1:]), v["median_us"])
        for r in runs
        for k, v in r["instances"].items()
    ]
    mean_points = [
        (int(k.split("i")[0][1:]), v["mean_us"])
        for r in runs
        for k, v in r["instances"].items()
    ]
    loop_points = [
        (int(k.split("i")[0][1:]), v["loop_us"])
        for r in runs
        for k, v in r["instances"].items()
    ]
    T: dict[int, dict[str, Any]] = {}
    for L in Ls:
        spans = sorted(
            x
            for r in runs
            for k, v in r["spans_us"].items()
            if int(k.split("i")[0][1:]) == L
            for x in v
        )
        T[L] = {
            "median_us": statistics.median(spans),
            "p10_us": spans[len(spans) // 10],
            "p90_us": spans[(len(spans) * 9) // 10],
            "mean_us": statistics.fmean(spans),
            "n": len(spans),
            "process_medians_us": [r["T_by_layers"][str(L)]["median_us"] for r in runs],
            "instance_medians_us": [
                m for r in runs for m in r["T_by_layers"][str(L)]["instance_medians_us"]
            ],
            "loop_us": statistics.median(y for x, y in loop_points if x == L),
            "instance_loop_us": [y for x, y in loop_points if x == L],
        }
    per_layer = lstsq_slope(points)
    corr = [c for r in runs for c in r["correctness"].values()]
    corr1 = [
        c
        for r in runs
        for k, c in r["correctness"].items()
        if k.startswith(f"L{min(Ls)}i")
    ]
    commit = runs[0]["commit"]
    file_stem, s_workload, s_case, desc = WORKLOADS[workload]
    sha = commit["commit"][:8]
    impl = (
        f"Mirage MPK (mpk@{sha})"
        if system == "mirage_mpk"
        else f"ThunderKittens ({sha})"
    )
    if system == "mirage_mpk":
        where = (
            f"{len(runs)} processes x {runs[0]['n_instances']} builds per L; coldl2.time_cold_many: one session "
            f"per process, every model interleaved"
        )
    else:
        where = (
            f"{len(runs)} rounds of one process per L (a process loads one build) x {runs[0]['n_instances']} "
            f"models per L; coldl2.time_cold_many per process"
        )
    method = (
        f"slope of cold-L2 T(L) over L = {Ls} layers (one decode step per launch, every layer with its own "
        f"weights and KV; least squares over {len(points)} (L, instance median) points from {where}; "
        f"CUDA graph, L2 flush before every launch)"
    )
    out: dict[str, Any] = {
        "system": "mirage" if system == "mirage_mpk" else "thunderkittens",
        "workload": workload,
        "description": desc,
        "variant": runs[0].get("variant", "latency"),
        "commit": commit["commit"],
        "versions": {
            "commit": commit["commit"],
            "describe": commit["describe"],
            "path": commit["path"],
            "dirty_files": commit["dirty_files"],
        },
        "per_layer_us": per_layer,
        "per_layer_method": method,
        "per_layer_us_from_means": lstsq_slope(mean_points),
        "per_layer_se_us": slope_se(points),
        "per_layer_loop_us": lstsq_slope(loop_points),
        "per_layer_loop_se_us": slope_se(loop_points),
        "per_layer_loop_method": "the same least-squares fit on coldl2's profiler-free loop_us of every "
        "model (cudaEvent loop over 200 x (reset, flush, launch) minus the same loop with a one-kernel "
        "graph, median of 7 trials; measured before the profiler session)",
        "per_layer_us_without_L1": lstsq_slope([pt for pt in points if pt[0] > 1]),
        "per_layer_us_L_ge_4": (
            lstsq_slope([pt for pt in points if pt[0] >= 4])
            if len({pt[0] for pt in points if pt[0] >= 4}) >= 2
            else None
        ),
        "T_by_layers": T,
        "pairwise_us": (T[Ls[-1]]["median_us"] - T[Ls[0]]["median_us"])
        / (Ls[-1] - Ls[0]),
        "processes": [
            {
                "file": r["_file"],
                "per_layer_us": r["per_layer_us"],
                "per_layer_us_from_means": r["per_layer_us_from_means"],
                "pairwise_us": r["pairwise_us"],
                "half_slopes_us": r.get("half_slopes_us"),
                "per_layer_loop_us": r["per_layer_loop_us"],
                "T_median_by_layers": {
                    L: v["median_us"] for L, v in r["T_by_layers"].items()
                },
                "pid": r["pid"],
                "time": r["time"],
                "gpu_wait_log": r["gpu_wait_log"],
            }
            for r in runs
        ],
        "correctness": {
            "max_abs": max(c["max_abs"] for c in corr),
            "max_rel": max(c["max_rel"] for c in corr),
            "rel_l2": max(c["rel_l2"] for c in corr),
            f"L{min(Ls)}_models": {
                "max_abs": max(c["max_abs"] for c in corr1),
                "max_rel": max(c["max_rel"] for c in corr1),
                "rel_l2": max(c["rel_l2"] for c in corr1),
            },
            "what": "worst over every model of every process; output vs a PyTorch reference of the "
            "same L-layer stack for the one decode step",
            "per_process": [r["correctness"] for r in runs],
            "after_timing": [r["correctness_after_timing"] for r in runs],
            "verify": [r.get("verify", {}) for r in runs],
        },
        "ctx_kv_rows": runs[0].get("ctx_kv_rows"),
        "decode_position": runs[0].get("decode_position"),
        "gpu": runs[0]["gpu"],
        "gpus_all_processes": [r["gpu"] for r in runs],
        "build": runs[0].get("build", runs[0].get("builds")),
        "alternative_variants": alternatives,
        "notes": NOTES.get(f"{system}/{workload}", []),
    }
    out["rows"] = [
        {
            "workload": s_workload,
            "case": s_case,
            "impl": impl,
            "median_us": per_layer,
            "loop_us": out["per_layer_loop_us"],
            "method": method,
            "ctx_kv_rows": out["ctx_kv_rows"],
            "decode_position": out["decode_position"],
            "batch": 1,
            "T_by_layers_us": {
                L: {
                    k: T[L][k]
                    for k in ("median_us", "p10_us", "p90_us", "mean_us", "loop_us")
                }
                for L in Ls
            },
            "per_process_per_layer_us": [r["per_layer_us"] for r in runs],
            "pairwise_us": out["pairwise_us"],
            "per_layer_se_us": out["per_layer_se_us"],
            "variant": out["variant"],
            "correctness_rel_l2": out["correctness"]["rel_l2"],
        }
    ]
    return out


def merge_round(parts: list[dict]) -> dict:
    """One run from the one-L-per-process JSONs of a round (thunderkittens: one build per process)."""
    parts = sorted(parts, key=operator.itemgetter("layers"))
    run = {
        k: parts[0][k]
        for k in (
            "system",
            "workload",
            "variant",
            "commit",
            "ctx_kv_rows",
            "decode_position",
            "n_instances",
            "gpu",
        )
    }
    for key in (
        "instances",
        "spans_us",
        "T_by_layers",
        "correctness",
        "correctness_after_timing",
        "build",
    ):
        run[key] = {k: v for r in parts for k, v in r[key].items()}
    points = [
        (r["layers"], v["median_us"]) for r in parts for v in r["instances"].values()
    ]
    run["per_layer_us"] = lstsq_slope(points)
    run["per_layer_us_from_means"] = lstsq_slope(
        [(r["layers"], v["mean_us"]) for r in parts for v in r["instances"].values()]
    )
    run["per_layer_loop_us"] = lstsq_slope(
        [(r["layers"], v["loop_us"]) for r in parts for v in r["instances"].values()]
    )
    lo, hi = parts[0], parts[-1]
    run["pairwise_us"] = (
        hi["T_by_layers"][str(hi["layers"])]["median_us"]
        - lo["T_by_layers"][str(lo["layers"])]["median_us"]
    ) / (hi["layers"] - lo["layers"])
    run["half_slopes_us"] = None
    run["pid"] = [r["pid"] for r in parts]
    run["time"] = [r["time"] for r in parts]
    run["gpu_wait_log"] = [x for r in parts for x in r["gpu_wait_log"]]
    run["_file"] = f"round {parts[0]['round']}: " + ", ".join(
        Path(r["_file"]).name for r in parts
    )
    run["gpus_by_process"] = [r["gpu"] for r in parts]
    return run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs", nargs="+", required=True, help="directories of per-process JSONs"
    )
    p.add_argument("--results", required=True, help="where the merged JSONs go")
    args = p.parse_args()
    notes_file = Path(__file__).with_name("notes.json")
    if notes_file.exists():
        NOTES.update(json.loads(notes_file.read_text()))
    groups = collections.defaultdict(list)
    rounds = collections.defaultdict(list)
    for d in args.runs:
        for f in sorted(Path(d).glob("*.json")):
            r = json.loads(f.read_text())
            r["_file"] = str(f)
            key = (r["system"], r["workload"], r.get("variant", "latency"))
            if "round" in r:  # one layer count per process: merged per round below
                rounds[(*key, r["round"])].append(r)
            else:
                groups[key].append(r)
    full = collections.defaultdict(set)  # every layer count any round of the group has
    for key, parts in rounds.items():
        full[key[:3]] |= {r["layers"] for r in parts}
    for key, parts in sorted(rounds.items()):
        have = {r["layers"] for r in parts}
        if have != full[key[:3]]:
            print(
                f"skipping incomplete round {key}: has L {sorted(have)} of {sorted(full[key[:3]])}"
            )
            continue
        groups[key[:3]].append(merge_round(parts))
    by_workload = collections.defaultdict(dict)
    for (system, workload, variant), runs in groups.items():
        by_workload[(system, workload)][variant] = runs
    for (system, workload), variants in by_workload.items():
        slopes = {
            v: lstsq_slope(
                [
                    (int(k.split("i")[0][1:]), x["median_us"])
                    for r in runs
                    for k, x in r["instances"].items()
                ]
            )
            for v, runs in variants.items()
        }
        small_l_rel_l2 = {
            v: max(
                c["rel_l2"]
                for r in runs
                for k, c in r["correctness"].items()
                if int(k.split("i")[0][1:])
                == min(int(L) for r2 in runs for L in r2["T_by_layers"])
            )
            for v, runs in variants.items()
        }
        correct = [v for v in variants if small_l_rel_l2[v] < CORRECT_REL_L2]
        if not correct:
            raise SystemExit(
                f"{system}/{workload}: no config passes the correctness check"
            )
        eligible = [v for v in correct if len(variants[v]) >= 2] or correct
        head = min(eligible, key=lambda v: slopes[v])
        alternatives = {
            v: {
                "per_layer_us": slopes[v],
                "processes": len(runs),
                "smallest_L_max_rel_l2": small_l_rel_l2[v],
                "correct": v in correct,
                "per_process_per_layer_us": [r["per_layer_us"] for r in runs],
                "per_process_per_layer_loop_us": [r["per_layer_loop_us"] for r in runs],
                "files": [r["_file"] for r in runs],
            }
            for v, runs in variants.items()
        }
        res = summarize(system, workload, variants[head], alternatives)
        Path(args.results).mkdir(parents=True, exist_ok=True)
        path = Path(args.results) / f"{PREFIX[system]}_{WORKLOADS[workload][0]}.json"
        path.write_text(json.dumps(res, indent=1))
        print(
            f"{path}: {res['per_layer_us']:.2f} +- {res['per_layer_se_us']:.2f} us/layer (loop_us "
            f"{res['per_layer_loop_us']:.2f}), without L=1 "
            f"{res['per_layer_us_without_L1']:.2f}, L>=4 {res['per_layer_us_L_ge_4']}, pairwise {res['pairwise_us']:.2f} "
            f"({head}; processes "
            f"{', '.join(f'{x:.2f}' for x in res['rows'][0]['per_process_per_layer_us'])}; "
            f"alternatives {', '.join(f'{v} {s:.2f}' + ('' if v in correct else ' (WRONG OUTPUT)') for v, s in slopes.items())})"
        )


if __name__ == "__main__":
    main()
