#!/usr/bin/env python3
"""Build dashboard-data.json from Helion benchmark artifacts.

Fetches benchmark runs via the GitHub API, downloads new artifacts, and
aggregates them into a single JSON file the dashboard renders with no
client-side computation. Supports incremental updates via --existing-url.

Designed to run in GitHub Actions (uses `gh` CLI for API access).
"""

import argparse
import datetime
import glob
import json
import math
import os
import subprocess
import urllib.error
import urllib.request
import zipfile

RETENTION_DAYS = 90


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


def gh_api(endpoint):
    r = subprocess.run(f'gh api "{endpoint}"', shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)


def fetch_runs(repo, workflow_name, days):
    workflows = gh_api(f"repos/{repo}/actions/workflows")
    if not workflows:
        return []
    wf_id = next((w["id"] for w in workflows.get("workflows", []) if w["name"] == workflow_name), None)
    if not wf_id:
        return []
    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs = []
    for page in range(1, 11):
        data = gh_api(f"repos/{repo}/actions/workflows/{wf_id}/runs?per_page=100&page={page}&status=completed&created=>{cutoff}")
        if not data or not data.get("workflow_runs"):
            break
        for r in data["workflow_runs"]:
            if r.get("conclusion") in ("success", "failure"):
                runs.append({
                    "run_id": str(r["id"]),
                    "sha": r["head_sha"][:8],
                    "full_sha": r["head_sha"],
                    "date": r["created_at"],
                    "branch": r.get("head_branch", "main"),
                })
    return runs


def download_artifacts(repo, run_id, dest):
    data = gh_api(f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100")
    if not data:
        return
    for art in data.get("artifacts", []):
        if not art["name"].startswith("benchmark-results-"):
            continue
        art_dir = os.path.join(dest, art["name"])
        os.makedirs(art_dir, exist_ok=True)
        zip_path = os.path.join(art_dir, "a.zip")
        with open(zip_path, "wb") as zf:
            r = subprocess.run(
                ["gh", "api", f"repos/{repo}/actions/artifacts/{art['id']}/zip"],
                stdout=zf, stderr=subprocess.DEVNULL,
            )
        if r.returncode == 0 and os.path.getsize(zip_path) > 0:
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(art_dir)
            except zipfile.BadZipFile:
                pass
            os.remove(zip_path)


def parse_run(run_dir):
    """Parse all helionbench.json files in a run directory into kernel entries."""
    kernels = []
    for bench_dir in sorted(glob.glob(os.path.join(run_dir, "benchmark-results-*"))):
        dirname = os.path.basename(bench_dir)
        parts = dirname.replace("benchmark-results-", "").split("-", 1)
        platform_short = parts[0] if parts else "unknown"

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

        # Group records by model name (a single file may contain multiple kernels)
        by_model = {}
        for item in data:
            model = item.get("model", {}).get("name")
            if not model:
                continue
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


def build_history_entry(run, metrics, shapes):
    helion_lat = [v for v in metrics.get("helion_latency_ms", []) if v]
    return {
        "run_id": run["run_id"],
        "sha": run["sha"],
        "full_sha": run["full_sha"],
        "date": run["date"],
        "branch": run.get("branch", "main"),
        "helion_speedup_geomean": round(geo_mean(metrics.get("helion_speedup", [])), 4),
        "triton_speedup_geomean": round(geo_mean(metrics.get("triton_speedup", [])), 4),
        "torch_compile_speedup_geomean": round(geo_mean(metrics.get("torch_compile_speedup", [])), 4),
        "compile_time_avg_s": round(avg(metrics.get("helion_compile_time_s", [])), 2),
        "helion_latency_avg_ms": round(avg(helion_lat), 4) if helion_lat else 0,
        "per_shape": {
            "shapes": shapes,
            "helion_speedup": metrics.get("helion_speedup", []),
            "triton_speedup": metrics.get("triton_speedup", []),
            "torch_compile_speedup": metrics.get("torch_compile_speedup", []),
            "helion_accuracy": metrics.get("helion_accuracy", []),
            "helion_compile_time_s": metrics.get("helion_compile_time_s", []),
            "helion_latency_ms": metrics.get("helion_latency_ms", []),
        },
    }


def build_dashboard_data(cache_dir, runs_meta, existing_data=None):
    """Aggregate benchmark data across runs into the dashboard JSON structure."""
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=RETENTION_DAYS)
    runs_meta_sorted = sorted(
        [r for r in runs_meta if datetime.datetime.fromisoformat(r["date"].replace("Z", "+00:00")) >= cutoff],
        key=lambda r: r["date"],
    )

    # Index existing history by kernel|platform -> run_id
    existing_history = {}
    existing_summary = {e["kernel"] + "|" + e["platform"]: e for e in (existing_data or {}).get("summary", [])}
    for key, entry in existing_summary.items():
        existing_history[key] = {h["run_id"]: h for h in entry.get("history", [])}

    # Parse each run's artifacts if present
    runs = []
    for meta in runs_meta_sorted:
        run_dir = os.path.join(cache_dir, meta["run_id"])
        kernels = parse_run(run_dir) if os.path.isdir(run_dir) and os.listdir(run_dir) else []
        runs.append({**meta, "kernel_count": len(kernels), "kernels": kernels})

    # Build kernel history: fresh artifacts override, then fill from existing history
    kernel_index = {}
    fresh_run_ids = set()

    for run in runs:
        if run["kernels"]:
            fresh_run_ids.add(run["run_id"])
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
            kernel_index[key]["history"].append(build_history_entry(run, k["metrics"], k["shapes"]))

    for run in runs:
        if run["run_id"] in fresh_run_ids:
            continue
        for key, cached_runs in existing_history.items():
            if run["run_id"] not in cached_runs:
                continue
            if key not in kernel_index:
                prev = existing_summary.get(key, {})
                kernel_index[key] = {
                    "kernel": prev.get("kernel", key.split("|")[0]),
                    "platform": prev.get("platform", key.split("|")[1] if "|" in key else "unknown"),
                    "platform_short": prev.get("platform_short", ""),
                    "shapes": [],
                    "history": [],
                }
            kernel_index[key]["history"].append(cached_runs[run["run_id"]])

    for entry in kernel_index.values():
        entry["history"].sort(key=lambda h: h["date"])

    # Build summary (Overview/Speedup tabs) based on main branch only
    summary = []
    for key, entry in sorted(kernel_index.items()):
        latest = prev = None
        for h in reversed(entry["history"]):
            if h.get("branch") != "main" or h["helion_speedup_geomean"] <= 0:
                continue
            if latest is None:
                latest = h
            elif prev is None:
                prev = h
                break

        speedup_delta = fmt_delta(latest["helion_speedup_geomean"], prev["helion_speedup_geomean"]) if latest and prev else None
        compile_delta = fmt_delta(prev["compile_time_avg_s"], latest["compile_time_avg_s"]) if latest and prev and latest["compile_time_avg_s"] > 0 else None
        latency_delta = fmt_delta(prev["helion_latency_avg_ms"], latest["helion_latency_avg_ms"]) if latest and prev and latest["helion_latency_avg_ms"] > 0 else None

        # Classification — prefer latency, fall back to speedup
        classify_delta = latency_delta if latency_delta is not None else speedup_delta
        status = "improved" if classify_delta and classify_delta > 10 else "regressed" if classify_delta and classify_delta < -10 else "unchanged"

        acc_failures = []
        if latest and latest["per_shape"]["helion_accuracy"]:
            shapes = latest["per_shape"]["shapes"]
            for i, a in enumerate(latest["per_shape"]["helion_accuracy"]):
                if a < 1.0:
                    acc_failures.append(shapes[i] if i < len(shapes) else f"shape #{i}")

        summary.append({
            "kernel": entry["kernel"],
            "platform": entry["platform"],
            "platform_short": entry["platform_short"],
            "status": status,
            "accuracy_failures": acc_failures,
            "helion_speedup_geomean": latest["helion_speedup_geomean"] if latest else 0,
            "triton_speedup_geomean": latest["triton_speedup_geomean"] if latest else 0,
            "torch_compile_speedup_geomean": latest["torch_compile_speedup_geomean"] if latest else 0,
            "compile_time_avg_s": latest["compile_time_avg_s"] if latest else 0,
            "helion_latency_avg_ms": latest["helion_latency_avg_ms"] if latest else 0,
            "speedup_delta_pct": speedup_delta,
            "compile_delta_pct": compile_delta,
            "latency_delta_pct": latency_delta,
            "history": entry["history"],
        })

    platforms = sorted({s["platform"] for s in summary})
    platform_shorts = sorted({s["platform_short"] for s in summary})
    unique_kernels = sorted({s["kernel"] for s in summary})
    improved = sum(1 for s in summary if s["status"] == "improved")
    regressed = sum(1 for s in summary if s["status"] == "regressed")

    return {
        "generated_at": runs[-1]["date"] if runs else "",
        "latest_run": runs_meta_sorted[-1] if runs_meta_sorted else {},
        "runs": [{k: v for k, v in r.items() if k != "kernels"} for r in runs],
        "platforms": platforms,
        "platform_shorts": platform_shorts,
        "unique_kernels": unique_kernels,
        "stats": {
            "total_kernels": len(unique_kernels),
            "total_combos": len(summary),
            "num_platforms": len(platforms),
            "improved_count": improved,
            "regressed_count": regressed,
            "unchanged_count": len(summary) - improved - regressed,
            "accuracy_failures": sum(len(s["accuracy_failures"]) for s in summary),
            "helion_geomean": round(geo_mean([s["helion_speedup_geomean"] for s in summary]), 4),
            "triton_geomean": round(geo_mean([s["triton_speedup_geomean"] for s in summary]), 4),
            "torch_compile_geomean": round(geo_mean([s["torch_compile_speedup_geomean"] for s in summary]), 4),
            "helion_wins": sum(1 for s in summary if s["helion_speedup_geomean"] > max(s["triton_speedup_geomean"], s["torch_compile_speedup_geomean"])),
        },
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--workflow-name", default="Benchmark Dispatch")
    parser.add_argument("--existing-url", default=None, help="URL to fetch previous dashboard-data.json for incremental updates")
    parser.add_argument("--output", default="dashboard-data.json")
    args = parser.parse_args()

    existing = {}
    if args.existing_url:
        try:
            with urllib.request.urlopen(args.existing_url, timeout=30) as resp:
                existing = json.load(resp)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            print(f"Could not fetch {args.existing_url} (probably first run)")
    print(f"Existing data: {len(existing.get('runs', []))} runs")

    runs = fetch_runs(args.repo, args.workflow_name, RETENTION_DAYS)
    print(f"Found {len(runs)} runs in last {RETENTION_DAYS} days")

    cache_dir = "./benchmark-cache"
    os.makedirs(cache_dir, exist_ok=True)
    existing_ids = {r["run_id"] for r in existing.get("runs", [])}
    for r in runs:
        if r["run_id"] in existing_ids:
            continue
        print(f"Downloading run {r['run_id']} ({r['sha']})...")
        download_artifacts(args.repo, r["run_id"], os.path.join(cache_dir, r["run_id"]))

    data = build_dashboard_data(cache_dir, runs, existing)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    s = data["stats"]
    print(f"Output: {s['total_combos']} combos, {len(data['runs'])} runs, {s['improved_count']} improved, {s['regressed_count']} regressed")


if __name__ == "__main__":
    main()
