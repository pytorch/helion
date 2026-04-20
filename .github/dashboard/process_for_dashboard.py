#!/usr/bin/env python3
"""Process Helion benchmark artifacts into dashboard-ready JSON.

Reads helionbench.json files from benchmark CI artifacts, aggregates
across runs, and outputs a single JSON file that the dashboard renders
with no client-side computation needed. Supports incremental updates
via --existing-data to avoid reprocessing old runs.

Usage:
    python process_for_dashboard.py \
        --cache-dir /path/to/benchmark-cache \
        --runs-meta /path/to/runs-meta.json \
        --output /path/to/dashboard-data.json
"""

import argparse
import datetime
import glob
import json
import math
import os
import sys


def geo_mean(values):
    pos = [v for v in values if v and v > 0]
    if not pos:
        return 0.0
    return math.exp(sum(math.log(v) for v in pos) / len(pos))


def avg(values):
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def fmt_delta(curr, prev):
    if not prev or prev == 0 or not curr:
        return None
    return round(((curr - prev) / prev) * 100, 2)


def parse_run(run_dir):
    """Parse all helionbench.json files in a run directory."""
    kernels = []
    for bench_dir in sorted(glob.glob(os.path.join(run_dir, "benchmark-results-*"))):
        dirname = os.path.basename(bench_dir)
        parts = dirname.replace("benchmark-results-", "").split("-", 1)
        platform_short = parts[0] if parts else "unknown"
        kernel_name = parts[1] if len(parts) > 1 else "unknown"

        bench_file = os.path.join(bench_dir, "helionbench.json")
        if not os.path.exists(bench_file):
            continue

        try:
            with open(bench_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if not data:
            continue

        platform = data[0].get("benchmark", {}).get("extra_info", {}).get("device", platform_short)
        if platform == platform_short and platform in ("b200", "h100", "mi325x", "mi350x"):
            continue

        # Use model name from JSON instead of directory name
        actual_kernel = data[0].get("model", {}).get("name", kernel_name)

        # Group records by model name (a single file may contain multiple kernels)
        by_model = {}
        for item in data:
            model = item.get("model", {}).get("name", actual_kernel)
            if model not in by_model:
                by_model[model] = {"shapes": item.get("shape", []), "metrics": {}}
            by_model[model]["metrics"][item["metric"]["name"]] = item["metric"]["benchmark_values"]

        for model_name, model_data in by_model.items():
            if not model_data["metrics"].get("helion_speedup"):
                continue

            kernels.append({
                "kernel": model_name,
                "platform": platform,
                "platform_short": platform_short,
                "shapes": model_data["shapes"],
                "metrics": model_data["metrics"],
            })

    return kernels


def build_dashboard_data(cache_dir, runs_meta, existing_data=None):
    """Build the complete dashboard data structure.

    For runs that have local artifacts (in cache_dir), parse them fresh.
    For runs that only exist in existing_data, reuse the cached history.
    """
    RETENTION_DAYS = 90
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=RETENTION_DAYS)
    runs_meta_sorted = sorted(
        [r for r in runs_meta if datetime.datetime.fromisoformat(r["date"].replace("Z", "+00:00")) >= cutoff],
        key=lambda r: r["date"],
    )
    run_ids = [r["run_id"] for r in runs_meta_sorted]

    # Build index of existing history keyed by kernel|platform -> run_id -> history entry
    existing_history = {}
    if existing_data and "summary" in existing_data:
        for entry in existing_data["summary"]:
            key = f"{entry['kernel']}|{entry['platform']}"
            if key not in existing_history:
                existing_history[key] = {}
            for h in entry.get("history", []):
                existing_history[key][h["run_id"]] = h

    # Parse each run — use local artifacts if available, skip if not (will use existing history later)
    runs = []
    for meta in runs_meta_sorted:
        run_dir = os.path.join(cache_dir, meta["run_id"])
        if os.path.isdir(run_dir) and os.listdir(run_dir):
            kernels = parse_run(run_dir)
            runs.append({**meta, "kernel_count": len(kernels), "kernels": kernels})
        else:
            runs.append({**meta, "kernel_count": 0, "kernels": []})

    # Build per-kernel history across runs
    # For runs with local artifacts, compute fresh. For others, reuse existing history.
    kernel_index = {}

    # First pass: add history from freshly parsed runs
    for run in runs:
        for k in run["kernels"]:
            key = f"{k['kernel']}|{k['platform']}"
            if key not in kernel_index:
                kernel_index[key] = {
                    "kernel": k["kernel"],
                    "platform": k["platform"],
                    "platform_short": k["platform_short"],
                    "shapes": k["shapes"],
                    "history": [],
                }
            helion_speedup_gm = geo_mean(k["metrics"].get("helion_speedup", []))
            triton_speedup_gm = geo_mean(k["metrics"].get("triton_speedup", []))
            tc_speedup_gm = geo_mean(k["metrics"].get("torch_compile_speedup", []))
            compile_time_avg = avg(k["metrics"].get("helion_compile_time_s", []))
            helion_latency = k["metrics"].get("helion_latency_ms", [])
            helion_latency_avg = avg([v for v in helion_latency if v]) if helion_latency else 0
            kernel_index[key]["history"].append({
                "run_id": run["run_id"],
                "sha": run["sha"],
                "full_sha": run["full_sha"],
                "date": run["date"],
                "branch": run.get("branch", "main"),
                "helion_speedup_geomean": round(helion_speedup_gm, 4),
                "triton_speedup_geomean": round(triton_speedup_gm, 4),
                "torch_compile_speedup_geomean": round(tc_speedup_gm, 4),
                "compile_time_avg_s": round(compile_time_avg, 2),
                "helion_latency_avg_ms": round(helion_latency_avg, 4),
                "per_shape": {
                    "shapes": k["shapes"],
                    "helion_speedup": k["metrics"].get("helion_speedup", []),
                    "triton_speedup": k["metrics"].get("triton_speedup", []),
                    "torch_compile_speedup": k["metrics"].get("torch_compile_speedup", []),
                    "helion_accuracy": k["metrics"].get("helion_accuracy", []),
                    "helion_compile_time_s": k["metrics"].get("helion_compile_time_s", []),
                    "helion_latency_ms": k["metrics"].get("helion_latency_ms", []),
                },
            })

    # Second pass: for runs without local artifacts, fill in from existing history
    fresh_run_ids = {run["run_id"] for run in runs if run["kernels"]}
    for run in runs:
        if run["run_id"] in fresh_run_ids:
            continue
        for key, cached_runs in existing_history.items():
            if run["run_id"] in cached_runs:
                if key not in kernel_index:
                    parts = key.split("|", 1)
                    platform_short = ""
                    for entry in (existing_data or {}).get("summary", []):
                        if entry["kernel"] == parts[0] and entry["platform"] == (parts[1] if len(parts) > 1 else ""):
                            platform_short = entry["platform_short"]
                            break
                    kernel_index[key] = {
                        "kernel": parts[0],
                        "platform": parts[1] if len(parts) > 1 else "unknown",
                        "platform_short": platform_short,
                        "shapes": [],
                        "history": [],
                    }
                kernel_index[key]["history"].append(cached_runs[run["run_id"]])

    # Sort each kernel's history by date
    for entry in kernel_index.values():
        entry["history"].sort(key=lambda h: h["date"])

    # Build the final summary with deltas pre-computed
    summary = []
    for key, entry in sorted(kernel_index.items()):
        history = entry["history"]
        latest = None
        prev = None
        for h in reversed(history):
            if h["helion_speedup_geomean"] > 0:
                if latest is None:
                    latest = h
                elif prev is None:
                    prev = h
                    break

        speedup_delta = fmt_delta(
            latest["helion_speedup_geomean"],
            prev["helion_speedup_geomean"]
        ) if latest and prev else None

        compile_delta = fmt_delta(
            prev["compile_time_avg_s"],  # inverted: lower is better
            latest["compile_time_avg_s"]
        ) if latest and prev and latest["compile_time_avg_s"] > 0 else None

        latency_delta = fmt_delta(
            prev["helion_latency_avg_ms"],  # inverted: lower is better
            latest["helion_latency_avg_ms"]
        ) if latest and prev and latest["helion_latency_avg_ms"] > 0 else None

        # Classification — prefer latency (lower is better), fall back to speedup
        status = "unchanged"
        if latency_delta is not None:
            if latency_delta > 10:
                status = "improved"
            elif latency_delta < -10:
                status = "regressed"
        elif speedup_delta is not None:
            if speedup_delta > 10:
                status = "improved"
            elif speedup_delta < -10:
                status = "regressed"

        # Accuracy check
        acc_failures = []
        if latest and latest["per_shape"]["helion_accuracy"]:
            for i, a in enumerate(latest["per_shape"]["helion_accuracy"]):
                if a < 1.0:
                    shapes = latest["per_shape"]["shapes"]
                    shape_name = shapes[i] if i < len(shapes) else f"shape #{i}"
                    acc_failures.append(shape_name)

        summary.append({
            "kernel": entry["kernel"],
            "platform": entry["platform"],
            "platform_short": entry["platform_short"],
            "status": status,
            "accuracy_failures": acc_failures,
            # Latest values
            "helion_speedup_geomean": latest["helion_speedup_geomean"] if latest else 0,
            "triton_speedup_geomean": latest["triton_speedup_geomean"] if latest else 0,
            "torch_compile_speedup_geomean": latest["torch_compile_speedup_geomean"] if latest else 0,
            "compile_time_avg_s": latest["compile_time_avg_s"] if latest else 0,
            "helion_latency_avg_ms": latest["helion_latency_avg_ms"] if latest else 0,
            # Deltas vs previous run
            "speedup_delta_pct": speedup_delta,
            "compile_delta_pct": compile_delta,
            "latency_delta_pct": latency_delta,
            # Full history for sparklines and detail views
            "history": history,
        })

    # Compute overall stats
    platforms = sorted(set(s["platform"] for s in summary))
    platform_shorts = sorted(set(s["platform_short"] for s in summary))
    unique_kernels = sorted(set(s["kernel"] for s in summary))

    improved_count = sum(1 for s in summary if s["status"] == "improved")
    regressed_count = sum(1 for s in summary if s["status"] == "regressed")
    total_acc_failures = sum(len(s["accuracy_failures"]) for s in summary)

    overall_helion_gm = geo_mean([s["helion_speedup_geomean"] for s in summary if s["helion_speedup_geomean"] > 0])
    overall_triton_gm = geo_mean([s["triton_speedup_geomean"] for s in summary if s["triton_speedup_geomean"] > 0])
    overall_tc_gm = geo_mean([s["torch_compile_speedup_geomean"] for s in summary if s["torch_compile_speedup_geomean"] > 0])
    helion_wins = sum(1 for s in summary if s["helion_speedup_geomean"] > max(s["triton_speedup_geomean"], s["torch_compile_speedup_geomean"]))

    latest_run = runs_meta_sorted[-1] if runs_meta_sorted else {}

    return {
        "generated_at": runs[-1]["date"] if runs else "",
        "latest_run": latest_run,
        "runs": [{k: v for k, v in r.items() if k != "kernels"} for r in runs],
        "platforms": platforms,
        "platform_shorts": platform_shorts,
        "unique_kernels": unique_kernels,
        "stats": {
            "total_kernels": len(unique_kernels),
            "total_combos": len(summary),
            "num_platforms": len(platforms),
            "improved_count": improved_count,
            "regressed_count": regressed_count,
            "unchanged_count": len(summary) - improved_count - regressed_count,
            "accuracy_failures": total_acc_failures,
            "helion_geomean": round(overall_helion_gm, 4),
            "triton_geomean": round(overall_triton_gm, 4),
            "torch_compile_geomean": round(overall_tc_gm, 4),
            "helion_wins": helion_wins,
        },
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Process Helion benchmarks for dashboard")
    parser.add_argument("--cache-dir", required=True, help="Directory containing run subdirectories")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--runs-meta", required=True, help="JSON file with run metadata (run_id, sha, date, branch)")
    parser.add_argument("--existing-data", default=None, help="Previous dashboard-data.json to merge with (for incremental updates)")
    args = parser.parse_args()

    with open(args.runs_meta) as f:
        runs_meta = json.load(f)

    existing_data = None
    if args.existing_data:
        try:
            with open(args.existing_data) as f:
                existing_data = json.load(f)
            print(f"Loaded existing data: {len(existing_data.get('runs', []))} runs")
        except (json.JSONDecodeError, OSError, KeyError):
            print("No valid existing data found, building from scratch")

    data = build_dashboard_data(args.cache_dir, runs_meta, existing_data)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    stats = data["stats"]
    print(f"Dashboard data written to {args.output}")
    print(f"  {stats['total_combos']} kernel/platform combos across {stats['num_platforms']} platforms")
    print(f"  {len(data['runs'])} runs")
    print(f"  {stats['improved_count']} improved, {stats['regressed_count']} regressed, {stats['unchanged_count']} unchanged")
    print(f"  Helion geomean: {stats['helion_geomean']}x, wins: {stats['helion_wins']}/{stats['total_combos']}")


if __name__ == "__main__":
    main()
