#!/usr/bin/env python3
"""Record per-module test durations in test/shard_durations.json.

CI splits the CuTe jobs across shards by module (``HELION_TEST_SHARD=i/n`` in
``test/conftest.py``), and ``helion._testing.sharding`` assigns the modules to
shards from the seconds recorded here so the shards finish together. Refresh
the file from junit reports of the CuTe GPU lane with the pull-request filter
(``not slow and (not cpu_only or cuda)``) and ``junit_duration_report=total``:

    scripts/run_tests.py --backend cute --lane gpu
    python scripts/shard_durations.py .logs/tests/<run>/cute-gpu*.xml

Do not import CPU-only or slow-tier reports: modules omitted from the CuTe GPU
PR lane do not receive a shard assignment.

Modules in the reports replace their recorded seconds, other entries stay,
and entries whose module no longer exists are dropped; run it without reports
after deleting or renaming a test module to drop its entry. ``--shards N``
prints the split the recorded durations give N shards (of the reports'
modules, or of every recorded module without reports) instead of writing.
Positive measurements have a minimum recorded weight of 0.1 seconds so
rounding cannot create a zero placeholder. The original JUnit reports retain
the unrounded measurements; unreported weights are not changed. This recording
floor does not change shard assignments: the scheduler separately applies its
1.0-second minimum weight to these small values.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from helion._testing.sharding import assign_shards
from helion._testing.sharding import load_durations
from helion._testing.sharding import shard_loads

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_of(classname: str, root: Path) -> str | None:
    """``test.test_foo.TestBar`` -> ``test/test_foo.py`` (the longest prefix
    that names a file under ``root``)."""
    parts = classname.split(".")
    for end in range(len(parts), 0, -1):
        candidate = "/".join(parts[:end]) + ".py"
        if (root / candidate).is_file():
            return candidate
    return None


def read_junit(path: Path, root: Path = REPO_ROOT) -> dict[str, float]:
    """Seconds per test module in one junit report."""
    seconds: dict[str, float] = defaultdict(float)
    for case in ET.parse(path).getroot().iter("testcase"):
        classname = case.get("classname", "")
        if not classname:  # a collection error or a session-level entry
            continue
        module = _module_of(classname, root)
        if module is None:
            raise SystemExit(
                f"{path}: cannot map classname {case.get('classname')!r} to a "
                f"module under {root}"
            )
        seconds[module] += float(case.get("time", 0.0))
    return dict(seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "junitxml",
        nargs="*",
        type=Path,
        help="junit reports to record (none: only drop deleted modules)",
    )
    parser.add_argument(
        "--shards",
        type=int,
        metavar="N",
        help="print the split of the recorded durations into N shards",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "test" / "shard_durations.json",
        help="durations file (default: this checkout's test/shard_durations.json)",
    )
    args = parser.parse_args(argv)

    recorded = load_durations(args.output)
    durations = dict(recorded)
    reported: set[str] = set()
    for report in args.junitxml:
        seconds = read_junit(report)
        durations.update(
            {
                module: max(0.1, round(elapsed, 1)) if elapsed > 0 else 0.0
                for module, elapsed in seconds.items()
            }
        )
        reported.update(seconds)
    durations = {
        module: seconds
        for module, seconds in durations.items()
        if (REPO_ROOT / module).is_file()
    }

    if args.shards:
        # The split of the reported modules (one lane's junit reports), or of
        # every recorded module without reports.
        modules = reported or set(durations)
        assignment = assign_shards(modules, args.shards, durations)
        loads = shard_loads(assignment, args.shards, durations)
        for shard, load in enumerate(loads):
            heaviest = sorted(
                (m for m, s in assignment.items() if s == shard),
                key=lambda m: -durations.get(m, 0.0),
            )[:3]
            print(
                f"shard {shard}: {load / 60:.1f} min over "
                f"{sum(s == shard for s in assignment.values())} modules; "
                f"heaviest {', '.join(heaviest)}"
            )
        return 0

    args.output.write_text(json.dumps(dict(sorted(durations.items())), indent=1) + "\n")
    dropped = sorted(set(recorded) - set(durations))
    print(
        f"{args.output}: {len(durations)} modules, {len(reported)} re-recorded, "
        f"{len(dropped)} dropped{': ' + ', '.join(dropped) if dropped else ''}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
