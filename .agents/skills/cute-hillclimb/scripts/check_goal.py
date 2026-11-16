#!/usr/bin/env python3
"""Goal gate for the cute-hillclimb skill.

Reads `artifacts/goal.json` (created in Step 1 of the skill) and decides
whether the hillclimb goal is met: EVERY variant's helion/baseline ratio is
at least 1.0 minus a fixed 1% noise margin (bar 0.99), each ratio is backed
by an artifact file that exists on disk, and the geomean of the ratios meets
`criteria.geomean_min`.

Modes:
  check_goal.py            Human-readable PASS/FAIL table. Exit 0 iff met.
  check_goal.py --gate     Stop-hook mode (the same Stop-hook machinery the
                           built-in /goal command uses). Exit 0 to allow the
                           turn to end; exit 2 with a reason on stderr to
                           block it. A missing goal file allows the stop, so
                           the hook is inert until Step 1 activates it.

Goal file schema:
  {
    "task": "softmax",
    "status": "running",            # or "paused" / "blocked", see below
    "paused_reason": null,          # required when status == "paused"
    "question_for_user": null,      # required when status == "blocked"
    "criteria": {"geomean_min": 1.0},
    "shapes": [
      {"name": "32768x4096",
       "ratio": 1.044, "artifact": "artifacts/<run-name>/x.jsonl"}
    ]
  }

`ratio` is helion throughput / best-baseline throughput (equivalently
baseline time / helion time; >1 means helion is faster) from the newest
cold-full-autotune measurement (interleaved ABAB verify runs when they
exist). `status` is only an escape hatch: "paused" when the user explicitly
said to stop (quote their instruction in paused_reason), "blocked" when
hand-edit falsification evidence says the goal is unreachable (put the
question in question_for_user). "Met" is never declared — it is computed
here. A goal file left over from a PREVIOUS run's task should be archived
into its old `artifacts/<run-name>/` directory, not evaluated.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

# Run-to-run spread on a power-capped B200 is about 1%.
NOISE_MARGIN_PCT = 1.0


def repo_root() -> str:
    root = os.environ.get("CLAUDE_PROJECT_DIR")
    if root:
        return root
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() or os.getcwd()


def evaluate(goal: dict, root: str) -> tuple[bool, list[str], list[str]]:
    """Returns (met, table_lines, failure_summaries)."""
    lines: list[str] = []
    fails: list[str] = []
    ratios: list[float] = []
    shapes = goal.get("shapes") or []
    if not shapes:
        return False, ["goal file lists no variants"], ["no variants listed"]
    bar = 1.0 - NOISE_MARGIN_PCT / 100.0
    for shape in shapes:
        name = shape.get("name", "?")
        ratio = shape.get("ratio")
        artifact = shape.get("artifact")
        problems = []
        if not isinstance(ratio, (int, float)):
            problems.append("no measured ratio")
        elif ratio <= 0:
            problems.append(f"ratio {ratio!r} is not a positive number")
        else:
            ratios.append(float(ratio))
            if ratio < bar:
                problems.append(f"ratio {ratio:.3f} < bar {bar:.3f}")
        if not artifact or not os.path.exists(os.path.join(root, artifact)):
            problems.append("no artifact on disk")
        if problems:
            fails.append(f"{name} ({'; '.join(problems)})")
            lines.append(f"FAIL {name}: {'; '.join(problems)}")
        else:
            lines.append(
                f"PASS {name}: ratio {ratio:.3f} >= bar {bar:.3f}  [{artifact}]"
            )
    geomean_min = float((goal.get("criteria") or {}).get("geomean_min", 1.0))
    if len(ratios) == len(shapes):
        geomean = math.prod(ratios) ** (1.0 / len(ratios))
        ok = geomean >= geomean_min
        lines.append(
            f"{'PASS' if ok else 'FAIL'} geomean: {geomean:.4f} (min {geomean_min})"
        )
        if not ok:
            fails.append(f"geomean {geomean:.4f} < {geomean_min}")
    else:
        lines.append("geomean: not computable (unmeasured variants)")
    return not fails, lines, fails


def gate(goal_path: str, root: str) -> int:
    if not sys.stdin.isatty():
        sys.stdin.read()  # consume the hook's JSON payload
    if not os.path.exists(goal_path):
        return 0  # no active hillclimb in this repo
    try:
        return _gate_verdict(goal_path, root)
    except Exception as e:  # fail closed: a malformed goal file must block
        print(
            f"cute-hillclimb stop gate: cannot evaluate {goal_path} ({e}); "
            "fix the goal file.",
            file=sys.stderr,
        )
        return 2


def _gate_verdict(goal_path: str, root: str) -> int:
    goal = json.loads(open(goal_path).read())
    status = goal.get("status", "running")
    if status == "paused":
        if (goal.get("paused_reason") or "").strip():
            return 0
        print(
            "cute-hillclimb stop gate: status is 'paused' but paused_reason "
            "is empty. Quote the user's instruction to pause there, or set "
            "status back to 'running'.",
            file=sys.stderr,
        )
        return 2
    if status == "blocked":
        if (goal.get("question_for_user") or "").strip():
            return 0
        print(
            "cute-hillclimb stop gate: status is 'blocked' but "
            "question_for_user is empty. State the question (with your "
            "hand-edit falsification evidence), or set status back to "
            "'running'.",
            file=sys.stderr,
        )
        return 2
    met, _lines, fails = evaluate(goal, root)
    if met:
        return 0
    n = len(goal.get("shapes") or [])
    print(
        f"cute-hillclimb stop gate: goal NOT met — {len(fails)} failing: "
        + "; ".join(fails)
        + ". Return to Step 2 of the skill and keep optimizing the worst "
        "failing variant; do not write a final report or describe the work "
        "as done. Escape hatches (only with cause): if the user explicitly "
        "told you to stop, set status 'paused' + paused_reason in "
        "artifacts/goal.json; if a hand-edit of the generated kernel to "
        "the baseline's structure still cannot reach parity, set status "
        "'blocked' + question_for_user. If this goal file belongs to a "
        "PREVIOUS run's task, archive it into its old artifacts/<run-name>/ "
        "directory and write a fresh one for the current task."
        + (f" ({n} variants total.)" if n else ""),
        file=sys.stderr,
    )
    return 2


def main() -> int:
    root = repo_root()
    goal_path = os.path.join(root, "artifacts", "goal.json")
    if "--gate" in sys.argv[1:]:
        return gate(goal_path, root)
    if not os.path.exists(goal_path):
        print(f"no goal file at {goal_path} (Step 1 of the skill creates it)")
        return 1
    goal = json.loads(open(goal_path).read())
    status = goal.get("status", "running")
    if status != "running":
        print(f"status: {status}")
    met, lines, _fails = evaluate(goal, root)
    print("\n".join(lines))
    if met:
        print("GOAL MET")
        return 0
    if status == "blocked" and (goal.get("question_for_user") or "").strip():
        print(
            "GOAL NOT MET — BLOCKED, awaiting the user's decision "
            "(question_for_user is set; the stop gate allows ending the turn)"
        )
    elif status == "paused" and (goal.get("paused_reason") or "").strip():
        print(
            "GOAL NOT MET — PAUSED at the user's instruction "
            "(the stop gate allows ending the turn)"
        )
    else:
        print("GOAL NOT MET")
    return 1


if __name__ == "__main__":
    sys.exit(main())
