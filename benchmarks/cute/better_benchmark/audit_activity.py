"""Flag the records of a run whose foreign-activity samples do not cover
their jobs, or that foreign processes contaminated.

A job samples ``nvidia-smi pmon`` on its GPU for as long as it runs
(``common.watch_foreign_gpu_activity``), and a record whose samples do not
cover the job cannot tell whether another process computed during it: the
campaign keeps such a record, and the report leaves it out of the
aggregates only when the record shows the gap. Records made before the
sampling recorded its coverage (``watched_s``, ``max_gap_s``) can read
"sampled and clean" though their sampling thread died early or nvidia-smi
failed for most of the job. This script reads every autotune and bench
record of a run (``<run>/configs``, ``<run>/bench``, and the attempts the
campaign set aside as ``<record>.attempt<k>``) and flags those that

- have no activity samples, or none that read the GPU;
- lost more samples than ``common.activity_coverage_issue`` allows;
- went longer than ``common.MAX_ACTIVITY_GAP_S`` without a sample, or,
  recorded before the gaps were, took fewer samples than one per
  ``common.MAX_SAMPLE_PERIOD_S`` over their ``wall_s``;
- or whose job log shows the sampling thread dying (``Exception in thread
  ... (poll)``);

and those that processes outside the job contaminated by the campaign's rule
(``common.activity_issue``: a bench record at one sample of
``common.CONTAMINATION_SM_PCT``, an autotune record only past
``common.AUTOTUNE_CONTAMINATION_SHARE`` of its job and
``common.AUTOTUNE_CONTAMINATION_MIN_S``, and a record the campaign kept
flagged ``contaminated``).

usage: audit_activity.py RUN_DIR [--all] [--json]

It prints one line per flagged standing record, with the reason and the
numbers it went by (with ``--all``, every record, the set-aside attempts
included), and exits with status 1 when it flagged a standing record: the
attempts were set aside because they were contaminated or uncovered.
``--json`` prints the rows as JSON instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.cute.better_benchmark import common


def sampler_died(log: Path) -> bool:
    """Whether the job log shows the sampling thread's traceback."""
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return False
    return any(
        line.startswith("Exception in thread") and "(poll)" in line
        for line in text.splitlines()
    )


def audit(run: Path) -> list[dict[str, Any]]:
    """A row per record of ``run``: its kind, path, numbers, whether it is
    an attempt the campaign set aside, and why its samples do not cover its
    job or which processes contaminated it (None when neither)."""
    rows: list[dict[str, Any]] = []
    for kind, directory, logs in (
        ("autotune", run / "configs", run / "logs" / "autotune"),
        ("bench", run / "bench", run / "logs" / "bench"),
    ):
        for path in sorted(directory.glob("*.json*")):
            if not (path.suffix == ".json" or ".json.attempt" in path.name):
                continue
            attempt = path.suffix != ".json"
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                rows.append(
                    {
                        "kind": kind,
                        "record": str(path),
                        "attempt": attempt,
                        "issue": str(error),
                    }
                )
                continue
            activity = record.get("foreign_gpu_activity") or {}
            issue = common.activity_issue(record)
            # The campaign logs a job under its record's name, and sets an
            # attempt's log aside with its record.
            if sampler_died(logs / path.name.replace(".json", ".log", 1)):
                issue = "the sampling thread died" + (f"; {issue}" if issue else "")
            rows.append(
                {
                    "kind": kind,
                    "record": str(path),
                    "attempt": attempt,
                    "wall_s": record.get("wall_s"),
                    "samples": activity.get("samples"),
                    "unknown_samples": activity.get("unknown_samples"),
                    "max_gap_s": activity.get("max_gap_s"),
                    "reused_from": record.get("reused_from"),
                    "issue": issue,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--all", action="store_true", help="list every record")
    parser.add_argument("--json", action="store_true", help="print JSON rows")
    args = parser.parse_args()
    rows = audit(args.run_dir)
    standing = [row for row in rows if not row.get("attempt")]
    flagged = [row for row in standing if row["issue"]]
    shown = rows if args.all else flagged
    if args.json:
        print(json.dumps(shown, indent=1))
    else:
        for row in shown:
            numbers = ", ".join(
                f"{key}={row[key]}"
                for key in ("wall_s", "samples", "unknown_samples", "max_gap_s")
                if row.get(key) is not None
            )
            reused = (
                f", reused from {row['reused_from']}" if row.get("reused_from") else ""
            )
            print(
                f"{row['kind']:8} {Path(row['record']).name}: "
                f"{row['issue'] or 'covered and clean'} ({numbers}{reused})"
            )
        print(f"{len(flagged)} of {len(standing)} standing records flagged")
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
