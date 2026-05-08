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
import re
import subprocess
import urllib.error
import urllib.request
import zipfile

RETENTION_DAYS = 365


def get_active_platforms(workflow_path):
    """Extract active platform aliases from the benchmark dispatch workflow."""
    try:
        with open(workflow_path) as f:
            return set(re.findall(r"alias:\s*(\S+)", f.read()))
    except OSError:
        return None


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
        print(f"Warning: gh api failed for {endpoint}: {r.stderr.strip()}")
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
        # Avoid status= and created= query filters — they are unreliable
        # on the GitHub API (intermittently return 0 results). Filter
        # client-side instead.
        data = gh_api(f"repos/{repo}/actions/workflows/{wf_id}/runs?per_page=100&page={page}")
        if not data or not data.get("workflow_runs"):
            break
        past_cutoff = False
        for r in data["workflow_runs"]:
            if r["created_at"] < cutoff:
                past_cutoff = True
                break
            if r.get("conclusion") in ("success", "failure", "cancelled"):
                # Nightly runs are dispatched by benchmark_nightly.yml's cron via
                # GITHUB_TOKEN, so actor is "github-actions[bot]". Manual user
                # dispatches have the user's login. Use actor (not triggering_actor):
                # triggering_actor occasionally returns "pytorch-bot[bot]" on cron
                # runs, presumably a re-run side effect, which mis-tags them.
                actor_login = (r.get("actor") or {}).get("login", "")
                runs.append({
                    "run_id": str(r["id"]),
                    "sha": r["head_sha"][:8],
                    "full_sha": r["head_sha"],
                    "date": r["created_at"],
                    "branch": r.get("head_branch", "main"),
                    "is_nightly": actor_login == "github-actions[bot]",
                })
        if past_cutoff:
            break
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


def parse_run(run_dir, active_platforms=None):
    """Parse all helionbench.json files in a run directory into kernel entries."""
    kernels = []
    for bench_dir in sorted(glob.glob(os.path.join(run_dir, "benchmark-results-*"))):
        dirname = os.path.basename(bench_dir)
        parts = dirname.replace("benchmark-results-", "").split("-", 1)
        platform_short = parts[0] if parts else "unknown"

        if active_platforms and platform_short not in active_platforms:
            continue

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
    # `compile_time_geomean_s = None` (rather than 0.0) when no compile-time
    # data was emitted, so index.html's `formatCompileTime(null)` renders as
    # "--" instead of "0.0s". 0.0 is reserved for runs that genuinely had
    # data but it averaged to zero.
    compile_times = metrics.get("helion_compile_time_s", [])
    compile_geomean = round(geo_mean(compile_times), 2) if compile_times else None
    return {
        "run_id": run["run_id"],
        "sha": run["sha"],
        "full_sha": run["full_sha"],
        "date": run["date"],
        "branch": run.get("branch", "main"),
        "is_nightly": bool(run.get("is_nightly")),
        "helion_speedup_geomean": round(geo_mean(metrics.get("helion_speedup", [])), 4),
        "triton_speedup_geomean": round(geo_mean(metrics.get("triton_speedup", [])), 4),
        "torch_compile_speedup_geomean": round(geo_mean(metrics.get("torch_compile_speedup", [])), 4),
        "compile_time_geomean_s": compile_geomean,
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


def build_dashboard_data(cache_dir, runs_meta, existing_data=None, active_platforms=None):
    """Aggregate benchmark data across runs into the dashboard JSON structure."""
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=RETENTION_DAYS)
    runs_meta_sorted = sorted(
        [r for r in runs_meta if datetime.datetime.fromisoformat(r["date"].replace("Z", "+00:00")) >= cutoff],
        key=lambda r: r["date"],
    )

    # Index existing history by kernel|platform_short -> run_id.
    # Older caches may contain duplicate (kernel, platform_short) rows because
    # AMD runners occasionally report platform as "AMD Radeon Graphics" vs
    # "AMD Instinct MI325X". Merge those duplicates here.
    existing_summary = {}
    existing_history = {}
    for e in (existing_data or {}).get("summary", []):
        if active_platforms and e.get("platform_short", "") not in active_platforms:
            continue
        key = e["kernel"] + "|" + e.get("platform_short", "")
        existing_summary[key] = e
        # Back-fill is_nightly on legacy cached history (pre-nightly-tagging):
        # treat any main-branch run as nightly so historical Overview data
        # survives the schema bump.
        for h in e.get("history", []):
            h.setdefault("is_nightly", h.get("branch") == "main")
        existing_history.setdefault(key, {}).update({h["run_id"]: h for h in e.get("history", [])})

    # Parse each run's artifacts if present
    runs = []
    for meta in runs_meta_sorted:
        run_dir = os.path.join(cache_dir, meta["run_id"])
        kernels = parse_run(run_dir, active_platforms) if os.path.isdir(run_dir) and os.listdir(run_dir) else []
        runs.append({**meta, "kernel_count": len(kernels), "kernels": kernels})

    # Build kernel history: fresh artifacts override, then fill from existing history
    kernel_index = {}
    fresh_run_ids = set()
    run_ids_with_data = set()

    for run in runs:
        if run["kernels"]:
            fresh_run_ids.add(run["run_id"])
            run_ids_with_data.add(run["run_id"])
        for k in run["kernels"]:
            key = f"{k['kernel']}|{k['platform_short']}"
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
            run_ids_with_data.add(run["run_id"])
            if key not in kernel_index:
                prev = existing_summary.get(key, {})
                kernel_index[key] = {
                    "kernel": prev.get("kernel", key.split("|")[0]),
                    "platform": prev.get("platform", "unknown"),
                    "platform_short": prev.get("platform_short", key.split("|")[1] if "|" in key else ""),
                    "shapes": [],
                    "history": [],
                }
            kernel_index[key]["history"].append(cached_runs[run["run_id"]])

    for entry in kernel_index.values():
        entry["history"].sort(key=lambda h: h["date"])

    # Overview only reflects the main-branch nightly cron. Side-branch nightly
    # workflows (e.g., Benchmark TPU Nightly on yifeixu/tpu-nightly-benchmark)
    # also pass is_nightly but shouldn't drive the Overview's "Latest Commit"
    # card or the per-kernel infra-missing classification.
    is_overview_run = lambda r: r.get("is_nightly") and r.get("branch") == "main"

    # Global latest nightly main run_id — used to detect kernels missing from
    # the latest benchmark (likely CI infra issue vs. actual kernel crash).
    latest_nightly_meta = next((r for r in reversed(runs_meta_sorted) if is_overview_run(r)), None)
    latest_nightly_run_id = latest_nightly_meta["run_id"] if latest_nightly_meta else None

    # Build summary (Overview/Speedup tabs) from main-branch nightly runs only.
    # Manual user dispatches and side-branch nightly workflows still appear in
    # entry["history"] for the Compare tab but never overwrite Overview perf.
    summary = []
    for key, entry in sorted(kernel_index.items()):
        # Single pass: latest_main is the most recent nightly main entry (used
        # for failure classification); latest_data/prev_data are the most recent
        # with non-zero data (used for perf display and deltas).
        latest_main = latest_data = prev_data = None
        for h in reversed(entry["history"]):
            if not is_overview_run(h):
                continue
            if latest_main is None:
                latest_main = h
            if h["helion_speedup_geomean"] > 0:
                if latest_data is None:
                    latest_data = h
                elif prev_data is None:
                    prev_data = h
            if latest_main is not None and prev_data is not None:
                break

        # All deltas use natural sign: positive = metric value increased.
        # Higher speedup = better; higher latency / compile time = worse.
        speedup_delta = fmt_delta(latest_data["helion_speedup_geomean"], prev_data["helion_speedup_geomean"]) if latest_data and prev_data else None
        compile_delta = fmt_delta(latest_data["compile_time_geomean_s"], prev_data["compile_time_geomean_s"]) if latest_data and prev_data and latest_data["compile_time_geomean_s"] and latest_data["compile_time_geomean_s"] > 0 else None
        latency_delta = fmt_delta(latest_data["helion_latency_avg_ms"], prev_data["helion_latency_avg_ms"]) if latest_data and prev_data and latest_data["helion_latency_avg_ms"] > 0 else None

        # Negate latency so >0 means improvement, matching speedup's direction.
        classify_delta = -latency_delta if latency_delta is not None else speedup_delta
        status = "improved" if classify_delta and classify_delta > 10 else "regressed" if classify_delta and classify_delta < -10 else "unchanged"

        acc_failures = []
        run_failures = []
        # Kernel absent from the global latest nightly run → likely CI infra issue.
        infra_missing = bool(latest_nightly_run_id) and (not latest_main or latest_main["run_id"] != latest_nightly_run_id)
        if not infra_missing and latest_main:
            latest_ps = latest_main.get("per_shape") or {}
            shapes = latest_ps.get("shapes", [])
            speedups = latest_ps.get("helion_speedup", [])
            for i, a in enumerate(latest_ps.get("helion_accuracy", [])):
                if a >= 1.0:
                    continue
                label = shapes[i] if i < len(shapes) else f"shape #{i}"
                sp = speedups[i] if i < len(speedups) else 0
                (acc_failures if sp and sp > 0 else run_failures).append(label)

        # Strip per_shape from all history entries except latest_main; it's only
        # read from latest_main (failure classification + detail modal's charts).
        keep_run_id = latest_main["run_id"] if latest_main else None
        for h in entry["history"]:
            if h["run_id"] != keep_run_id:
                h.pop("per_shape", None)

        # has_nightly_data flags entries for Overview/Speedup filtering.
        # Manual-only kernels still appear in summary so Compare tab can use them.
        summary.append({
            "kernel": entry["kernel"],
            "platform": entry["platform"],
            "platform_short": entry["platform_short"],
            "has_nightly_data": latest_data is not None,
            "status": status,
            "accuracy_failures": acc_failures,
            "run_failures": run_failures,
            "last_seen_date": latest_main["date"] if latest_main else None,
            "infra_missing": infra_missing,
            "helion_speedup_geomean": latest_data["helion_speedup_geomean"] if latest_data else 0,
            "triton_speedup_geomean": latest_data["triton_speedup_geomean"] if latest_data else 0,
            "torch_compile_speedup_geomean": latest_data["torch_compile_speedup_geomean"] if latest_data else 0,
            "compile_time_geomean_s": latest_data["compile_time_geomean_s"] if latest_data else 0,
            "helion_latency_avg_ms": latest_data["helion_latency_avg_ms"] if latest_data else 0,
            "speedup_delta_pct": speedup_delta,
            "compile_delta_pct": compile_delta,
            "latency_delta_pct": latency_delta,
            "history": entry["history"],
        })

    # Stats and platform/kernel lists reflect only entries with nightly data
    nightly_summary = [s for s in summary if s["has_nightly_data"]]
    platforms = sorted({s["platform"] for s in nightly_summary})
    platform_shorts = sorted({s["platform_short"] for s in nightly_summary})
    unique_kernels = sorted({s["kernel"] for s in nightly_summary})
    improved = sum(1 for s in nightly_summary if s["status"] == "improved")
    regressed = sum(1 for s in nightly_summary if s["status"] == "regressed")

    overview_runs = [r for r in runs_meta_sorted if is_overview_run(r)]
    latest_nightly_run = overview_runs[-1] if overview_runs else {}

    # Drop runs whose artifacts expired and weren't previously cached; otherwise
    # they pad charts with no-data gaps.
    output_runs = [r for r in runs if r["run_id"] in run_ids_with_data]

    return {
        "generated_at": runs[-1]["date"] if runs else "",
        "latest_run": latest_nightly_run,
        "runs": [{k: v for k, v in r.items() if k != "kernels"} for r in output_runs],
        "platforms": platforms,
        "platform_shorts": platform_shorts,
        "unique_kernels": unique_kernels,
        "stats": {
            "total_kernels": len(unique_kernels),
            "total_combos": len(nightly_summary),
            "num_platforms": len(platforms),
            "improved_count": improved,
            "regressed_count": regressed,
            "unchanged_count": len(nightly_summary) - improved - regressed,
            "accuracy_failures": sum(len(s["accuracy_failures"]) for s in nightly_summary),
            "run_failures": sum(len(s["run_failures"]) for s in nightly_summary),
            "infra_missing": sum(1 for s in nightly_summary if s["infra_missing"]),
            "helion_geomean": round(geo_mean([s["helion_speedup_geomean"] for s in nightly_summary]), 4),
            "triton_geomean": round(geo_mean([s["triton_speedup_geomean"] for s in nightly_summary]), 4),
            "torch_compile_geomean": round(geo_mean([s["torch_compile_speedup_geomean"] for s in nightly_summary]), 4),
            "helion_wins": sum(1 for s in nightly_summary if s["helion_speedup_geomean"] > max(s["triton_speedup_geomean"], s["torch_compile_speedup_geomean"])),
        },
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--workflow-name", default="Benchmark Dispatch")
    parser.add_argument("--existing-url", default=None, help="URL to fetch previous dashboard-data.json for incremental updates")
    parser.add_argument("--dispatch-workflow", default=".github/workflows/benchmark_dispatch.yml",
                        help="Path to benchmark dispatch workflow to derive active platforms")
    parser.add_argument("--output", default="dashboard-data.json")
    args = parser.parse_args()

    active_platforms = get_active_platforms(args.dispatch_workflow)
    if active_platforms:
        print(f"Active platforms: {', '.join(sorted(active_platforms))}")
    else:
        print("Warning: could not read dispatch workflow, showing all platforms")

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

    if not runs and existing.get("runs"):
        print(f"ERROR: fetch_runs returned 0 but existing data has {len(existing['runs'])} runs.")
        print("This is likely a transient API failure. Using existing data as-is.")
        with open(args.output, "w") as f:
            json.dump(existing, f, indent=2)
        s = existing.get("stats", {})
        print(f"Output (existing): {s.get('total_combos', 0)} combos, {len(existing['runs'])} runs")
        return

    cache_dir = "./benchmark-cache"
    os.makedirs(cache_dir, exist_ok=True)
    existing_ids = {r["run_id"] for r in existing.get("runs", [])}
    for r in runs:
        if r["run_id"] in existing_ids:
            continue
        print(f"Downloading run {r['run_id']} ({r['sha']})...")
        download_artifacts(args.repo, r["run_id"], os.path.join(cache_dir, r["run_id"]))

    data = build_dashboard_data(cache_dir, runs, existing, active_platforms)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    s = data["stats"]
    print(f"Output: {s['total_combos']} combos, {len(data['runs'])} runs, {s['improved_count']} improved, {s['regressed_count']} regressed")


if __name__ == "__main__":
    main()
