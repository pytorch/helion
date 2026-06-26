#!/usr/bin/env python3
"""Build pretuned-dashboard-data.json from pretuned-kernel nightly artifacts.

Companion to build_dashboard_data.py for the pretuned-kernel suite
(pretuned_kernels/run.py output). Fetches runs of the "Benchmark Pretuned
Dispatch" workflow, downloads each run's pretuned-results-* artifacts, and
aggregates the per-kernel SUMMARY metrics (geomean speedup vs eager, win count,
best speedup) into a single JSON the dashboard's Pretuned tab renders with no
client-side computation. Supports incremental updates via --existing-url.

Scheduled runs (GitHub event == "schedule") are treated as the nightly trend;
manual workflow_dispatch runs stay in per-kernel history for context but never
drive the headline numbers.
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
    """Extract platform aliases from the pretuned dispatch workflow."""
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


def fmt_delta(curr, prev):
    if not prev or prev == 0 or not curr:
        return None
    return round(((curr - prev) / prev) * 100, 2)


def gh_api(endpoint):
    r = subprocess.run(
        f'gh api "{endpoint}"', shell=True, capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"Warning: gh api failed for {endpoint}: {r.stderr.strip()}")
        return None
    return json.loads(r.stdout)


def fetch_runs(repo, workflow_name, days):
    workflows = gh_api(f"repos/{repo}/actions/workflows")
    if not workflows:
        return []
    wf_id = next(
        (w["id"] for w in workflows.get("workflows", []) if w["name"] == workflow_name),
        None,
    )
    if not wf_id:
        return []
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs = []
    for page in range(1, 11):
        # Filter client-side: status=/created= query filters are unreliable on
        # the GitHub API (intermittently return 0 results).
        data = gh_api(
            f"repos/{repo}/actions/workflows/{wf_id}/runs?per_page=100&page={page}"
        )
        if not data or not data.get("workflow_runs"):
            break
        past_cutoff = False
        for r in data["workflow_runs"]:
            if r["created_at"] < cutoff:
                past_cutoff = True
                break
            if r.get("conclusion") in ("success", "failure", "cancelled"):
                # Cron-scheduled runs are the nightly trend. Manual
                # workflow_dispatch runs stay in history but don't drive the
                # headline numbers. Accept both a direct `schedule` event and a
                # bot-driven dispatch (as benchmark_nightly.yml does for the main
                # pipeline) so either wiring tags nightly correctly.
                actor_login = (r.get("actor") or {}).get("login", "")
                is_nightly = (
                    r.get("event") == "schedule"
                    or actor_login == "github-actions[bot]"
                )
                runs.append(
                    {
                        "run_id": str(r["id"]),
                        "sha": r["head_sha"][:8],
                        "full_sha": r["head_sha"],
                        "date": r["created_at"],
                        "branch": r.get("head_branch", "main"),
                        "is_nightly": is_nightly,
                    }
                )
        if past_cutoff:
            break
    return runs


def download_artifacts(repo, run_id, dest):
    data = gh_api(f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100")
    if not data:
        return
    for art in data.get("artifacts", []):
        if not art["name"].startswith("pretuned-results-"):
            continue
        art_dir = os.path.join(dest, art["name"])
        os.makedirs(art_dir, exist_ok=True)
        zip_path = os.path.join(art_dir, "a.zip")
        with open(zip_path, "wb") as zf:
            r = subprocess.run(
                ["gh", "api", f"repos/{repo}/actions/artifacts/{art['id']}/zip"],
                stdout=zf,
                stderr=subprocess.DEVNULL,
            )
        if r.returncode == 0 and os.path.getsize(zip_path) > 0:
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(art_dir)
            except zipfile.BadZipFile:
                pass
            os.remove(zip_path)


def parse_run(run_dir, active_platforms=None):
    """Parse pretuned-bench.json files in a run directory into kernel entries."""
    kernels = []
    for bench_dir in sorted(glob.glob(os.path.join(run_dir, "pretuned-results-*"))):
        platform_short = os.path.basename(bench_dir).replace("pretuned-results-", "")
        if active_platforms and platform_short not in active_platforms:
            continue
        bench_file = os.path.join(bench_dir, "pretuned-bench.json")
        if not os.path.exists(bench_file):
            continue
        try:
            with open(bench_file) as f:
                records = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for rec in records or []:
            if "geomean" not in rec:
                # Kernel crashed on this run; skip (build_history records gaps).
                continue
            kernels.append(
                {
                    "kernel": rec["kernel"],
                    "platform": rec.get("device", platform_short),
                    "platform_short": platform_short,
                    "compute": rec.get("compute_capability", ""),
                    "cudagraph": bool(rec.get("cudagraph", False)),
                    "geomean": rec["geomean"],
                    "helion_wins": rec.get("helion_wins", 0),
                    "total": rec.get("total", 0),
                    "best_speedup": rec.get("best_speedup", 0.0),
                }
            )
    return kernels


def build_history_entry(run, k):
    return {
        "run_id": run["run_id"],
        "sha": run["sha"],
        "full_sha": run["full_sha"],
        "date": run["date"],
        "branch": run.get("branch", "main"),
        "is_nightly": bool(run.get("is_nightly")),
        "cudagraph": bool(k.get("cudagraph", False)),
        "geomean": round(k["geomean"], 4),
        "helion_wins": k["helion_wins"],
        "total": k["total"],
        "best_speedup": round(k["best_speedup"], 4),
    }


def build_dashboard_data(cache_dir, runs_meta, existing_data=None, active_platforms=None):
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=RETENTION_DAYS
    )
    runs_meta_sorted = sorted(
        [
            r
            for r in runs_meta
            if datetime.datetime.fromisoformat(r["date"].replace("Z", "+00:00"))
            >= cutoff
        ],
        key=lambda r: r["date"],
    )

    # Index existing history by kernel|platform_short -> {run_id: entry}.
    existing_summary = {}
    existing_history = {}
    for e in (existing_data or {}).get("summary", []):
        if active_platforms and e.get("platform_short", "") not in active_platforms:
            continue
        key = e["kernel"] + "|" + e.get("platform_short", "")
        existing_summary[key] = e
        existing_history.setdefault(key, {}).update(
            {h["run_id"]: h for h in e.get("history", [])}
        )

    runs = []
    for meta in runs_meta_sorted:
        run_dir = os.path.join(cache_dir, meta["run_id"])
        kernels = (
            parse_run(run_dir, active_platforms)
            if os.path.isdir(run_dir) and os.listdir(run_dir)
            else []
        )
        runs.append({**meta, "kernel_count": len(kernels), "kernels": kernels})

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
                    "compute": k["compute"],
                    "cudagraph": k.get("cudagraph", False),
                    "history": [],
                }
            kernel_index[key]["cudagraph"] = k.get("cudagraph", False)
            kernel_index[key]["history"].append(build_history_entry(run, k))

    # Backfill from cached history for runs whose artifacts have expired.
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
                    "platform_short": prev.get(
                        "platform_short", key.split("|")[1] if "|" in key else ""
                    ),
                    "compute": prev.get("compute", ""),
                    "cudagraph": prev.get("cudagraph", False),
                    "history": [],
                }
            kernel_index[key]["history"].append(cached_runs[run["run_id"]])

    for entry in kernel_index.values():
        entry["history"].sort(key=lambda h: h["date"])

    is_nightly_main = lambda r: r.get("is_nightly") and r.get("branch", "main") == "main"

    summary = []
    for key, entry in sorted(kernel_index.items()):
        latest = prev = None
        for h in reversed(entry["history"]):
            if not is_nightly_main(h):
                continue
            if latest is None:
                latest = h
            elif prev is None:
                prev = h
                break
        geomean_delta = (
            fmt_delta(latest["geomean"], prev["geomean"]) if latest and prev else None
        )
        status = (
            "improved"
            if geomean_delta and geomean_delta > 10
            else "regressed"
            if geomean_delta and geomean_delta < -10
            else "unchanged"
        )
        summary.append(
            {
                "kernel": entry["kernel"],
                "platform": entry["platform"],
                "platform_short": entry["platform_short"],
                "compute": entry["compute"],
                "cudagraph": bool(entry.get("cudagraph", False)),
                "has_nightly_data": latest is not None,
                "status": status,
                "geomean_delta_pct": geomean_delta,
                "geomean": latest["geomean"] if latest else 0,
                "helion_wins": latest["helion_wins"] if latest else 0,
                "total": latest["total"] if latest else 0,
                "best_speedup": latest["best_speedup"] if latest else 0,
                "last_seen_date": latest["date"] if latest else None,
                "history": entry["history"],
            }
        )

    nightly_summary = [s for s in summary if s["has_nightly_data"]]
    platforms = sorted({s["platform"] for s in nightly_summary})
    platform_shorts = sorted({s["platform_short"] for s in nightly_summary})
    unique_kernels = sorted({s["kernel"] for s in nightly_summary})
    improved = sum(1 for s in nightly_summary if s["status"] == "improved")
    regressed = sum(1 for s in nightly_summary if s["status"] == "regressed")

    overview_runs = [r for r in runs_meta_sorted if is_nightly_main(r)]
    latest_nightly_run = overview_runs[-1] if overview_runs else {}
    output_runs = [r for r in runs if r["run_id"] in run_ids_with_data]

    return {
        "generated_at": runs[-1]["date"] if runs else "",
        "latest_run": latest_nightly_run,
        "runs": [{k: v for k, v in r.items() if k != "kernels"} for r in output_runs],
        "platforms": platforms,
        "platform_shorts": platform_shorts,
        "kernels": unique_kernels,
        "stats": {
            "total_kernels": len(unique_kernels),
            "total_combos": len(nightly_summary),
            "num_platforms": len(platforms),
            "improved_count": improved,
            "regressed_count": regressed,
            "unchanged_count": len(nightly_summary) - improved - regressed,
            "geomean": round(geo_mean([s["geomean"] for s in nightly_summary]), 4),
            "helion_wins": sum(1 for s in nightly_summary if s["geomean"] > 1.0),
        },
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--workflow-name", default="Benchmark Pretuned Dispatch")
    parser.add_argument(
        "--existing-url",
        default=None,
        help="URL to fetch previous pretuned-dashboard-data.json for incremental updates",
    )
    parser.add_argument(
        "--dispatch-workflow",
        default=".github/workflows/benchmark_pretuned_dispatch.yml",
        help="Path to dispatch workflow to derive active platforms",
    )
    parser.add_argument("--output", default="pretuned-dashboard-data.json")
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
        print(
            f"ERROR: fetch_runs returned 0 but existing data has "
            f"{len(existing['runs'])} runs. Likely a transient API failure; "
            f"using existing data as-is."
        )
        with open(args.output, "w") as f:
            json.dump(existing, f, indent=2)
        return

    cache_dir = "./pretuned-benchmark-cache"
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
    print(
        f"Output: {s['total_combos']} combos, {len(data['runs'])} runs, "
        f"{s['improved_count']} improved, {s['regressed_count']} regressed"
    )


if __name__ == "__main__":
    main()
