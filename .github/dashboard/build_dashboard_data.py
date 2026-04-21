#!/usr/bin/env python3
"""Fetch benchmark artifacts and build dashboard-data.json.

Designed to run in GitHub Actions. Uses `gh` CLI for API access.
Only downloads artifacts for runs not already in existing data.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import zipfile


def gh_api(endpoint):
    r = subprocess.run(
        f'gh api "{endpoint}"', shell=True, capture_output=True, text=True
    )
    if r.returncode != 0:
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
        data = gh_api(
            f"repos/{repo}/actions/workflows/{wf_id}/runs"
            f"?per_page=100&page={page}&status=completed&created=>{cutoff}"
        )
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--workflow-name", default="Benchmark Dispatch")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--existing-data", default=None)
    parser.add_argument("--output", default="dashboard-data.json")
    args = parser.parse_args()

    existing = {}
    if args.existing_data and os.path.exists(args.existing_data):
        try:
            with open(args.existing_data) as f:
                existing = json.load(f)
            print(f"Loaded existing data: {len(existing.get('runs', []))} runs")
        except (json.JSONDecodeError, OSError):
            pass
    existing_ids = {r["run_id"] for r in existing.get("runs", [])}

    all_runs = fetch_runs(args.repo, args.workflow_name, args.days)
    print(f"Found {len(all_runs)} runs in last {args.days} days")

    cache_dir = "./benchmark-cache"
    os.makedirs(cache_dir, exist_ok=True)
    new_count = 0
    for r in all_runs:
        if r["run_id"] in existing_ids:
            continue
        print(f"Downloading run {r['run_id']} ({r['sha']})...")
        download_artifacts(args.repo, r["run_id"], os.path.join(cache_dir, r["run_id"]))
        new_count += 1
    print(f"Downloaded {new_count} new runs")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from process_for_dashboard import build_dashboard_data

    data = build_dashboard_data(cache_dir, all_runs, existing)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    s = data["stats"]
    print(f"Output: {s['total_combos']} combos, {len(data['runs'])} runs, "
          f"{s['improved_count']} improved, {s['regressed_count']} regressed")


if __name__ == "__main__":
    main()
