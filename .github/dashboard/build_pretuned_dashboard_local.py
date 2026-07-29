#!/usr/bin/env python3
"""Build pretuned-dashboard-data.json from LOCAL artifacts (no GitHub fetch).

Dev/test helper for the Pretuned dashboard tab. Point it at a directory that
contains one or more ``pretuned-results-<alias>/pretuned-bench.json`` files
(exactly what ``pretuned_kernels/run.py`` writes and what CI uploads as
artifacts), and it produces a ``pretuned-dashboard-data.json`` the dashboard can
render -- without touching the GitHub API.

Typical use on an H100 box::

    # 1. produce a local artifact (dir name must be pretuned-results-<alias>)
    python pretuned_kernels/run.py \\
        --output /tmp/reports/pretuned-results-h100/pretuned-bench.json

    # 2. build the dashboard JSON next to index.html
    python .github/dashboard/build_pretuned_dashboard_local.py \\
        --results-dir /tmp/reports \\
        --output .github/dashboard/pretuned-dashboard-data.json

    # 3. serve and open the Pretuned tab
    cd .github/dashboard && python -m http.server 8899

Pass ``--existing <file>`` (e.g. the file from a previous run of this script) to
append the current artifacts as a new dated run, so you can preview the
over-time trend chart with more than one data point.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Empty-but-valid main dashboard-data.json. index.html's init() renders all three
# core tabs (Overview/Speedup/Compare) on load; they tolerate empty arrays, but
# renderOverview() pipes latest_run.sha/full_sha through escHtml(), which throws on
# undefined -- so latest_run must carry empty strings, not be {}.
_EMPTY_MAIN_DASHBOARD = {
    "generated_at": "",
    "latest_run": {
        "run_id": "",
        "sha": "",
        "full_sha": "",
        "date": "",
        "branch": "main",
        "is_nightly": False,
    },
    "runs": [],
    "platforms": [],
    "platform_shorts": [],
    "unique_kernels": [],
    "stats": {
        "total_kernels": 0,
        "total_combos": 0,
        "num_platforms": 0,
        "improved_count": 0,
        "regressed_count": 0,
        "unchanged_count": 0,
        "accuracy_failures": 0,
        "run_failures": 0,
        "infra_missing": 0,
        "helion_geomean": 0,
        "triton_geomean": 0,
        "torch_compile_geomean": 0,
        "helion_wins": 0,
    },
    "summary": [],
}


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "_pretuned_builder", _HERE / "build_pretuned_dashboard_data.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing pretuned-results-<alias>/pretuned-bench.json subdirs.",
    )
    parser.add_argument(
        "--output",
        default=str(_HERE / "pretuned-dashboard-data.json"),
        help="Where to write the dashboard JSON.",
    )
    parser.add_argument(
        "--existing",
        default=None,
        help="Prior pretuned-dashboard-data.json to extend (appends this as a new run).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Synthetic run id (default: a timestamp, so repeated runs accumulate history).",
    )
    parser.add_argument("--sha", default="local", help="Commit sha label for the run.")
    parser.add_argument(
        "--date",
        default=None,
        help="ISO8601 run date (default: now, UTC).",
    )
    parser.add_argument(
        "--no-dummy-main",
        action="store_true",
        help="Skip writing a placeholder dashboard-data.json (the main tabs' data).",
    )
    args = parser.parse_args()

    builder = _load_builder()

    results_dir = Path(args.results_dir).resolve()
    result_subdirs = sorted(results_dir.glob("pretuned-results-*"))
    if not result_subdirs:
        raise SystemExit(
            f"No pretuned-results-* subdirs under {results_dir}. "
            f"Run pretuned_kernels/run.py --output "
            f"{results_dir}/pretuned-results-h100/pretuned-bench.json first."
        )
    print(f"Found {len(result_subdirs)} platform artifact(s): "
          f"{', '.join(d.name for d in result_subdirs)}")

    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = args.run_id or now.strftime("local-%Y%m%d%H%M%S")
    date = args.date or now.strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = args.sha

    existing = {}
    if args.existing and Path(args.existing).exists():
        with open(args.existing) as f:
            existing = json.load(f)
        print(f"Extending existing data: {len(existing.get('runs', []))} prior run(s)")

    # build_dashboard_data expects a cache laid out as <cache>/<run_id>/pretuned-results-*/.
    with tempfile.TemporaryDirectory() as cache_dir:
        run_cache = Path(cache_dir) / run_id
        run_cache.mkdir(parents=True)
        for sub in result_subdirs:
            shutil.copytree(sub, run_cache / sub.name)

        # Existing runs come from the prior output so their cached history is reused.
        runs_meta = list(existing.get("runs", []))
        runs_meta = [r for r in runs_meta if r.get("run_id") != run_id]
        runs_meta.append(
            {
                "run_id": run_id,
                "sha": sha[:8],
                "full_sha": sha,
                "date": date,
                "branch": "main",
                "is_nightly": True,
            }
        )

        data = builder.build_dashboard_data(
            cache_dir, runs_meta, existing, active_platforms=None
        )

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    s = data["stats"]
    print(
        f"Wrote {args.output}: {s['total_combos']} kernel/platform combos, "
        f"{len(data['runs'])} run(s), geomean {s['geomean']}x"
    )

    # Serving .github/dashboard/ locally loads dashboard-data.json (the main
    # tritonbench tabs) too; it's CI-generated and absent locally. Drop an
    # empty-but-valid placeholder next to the output so Overview/Speedup/Compare
    # render cleanly instead of "Data Not Loaded". Never clobber a real one.
    if not args.no_dummy_main:
        main_path = Path(args.output).resolve().parent / "dashboard-data.json"
        if main_path.exists():
            print(f"Left existing {main_path} untouched.")
        else:
            with main_path.open("w") as f:
                json.dump(_EMPTY_MAIN_DASHBOARD, f, indent=2)
            print(f"Wrote placeholder {main_path} (empty main dashboard data).")


if __name__ == "__main__":
    main()
