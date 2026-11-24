#!/usr/bin/env python3
"""Goal gate for the inductor-hillclimb skill.

Reads `inductor_artifacts/goal.json` (created in Step 1 of the skill) and
decides whether the run's goal is met: EVERY target's ratio is at least 1.0
minus a fixed 1% noise margin (bar 0.99), and every ratio is a finite positive
number backed by an artifact file under `inductor_artifacts/` (not the goal
file itself). The regression corpus is one more target: its worst
before / after time ratio.

Modes:
  check_goal.py            Human-readable PASS/FAIL table. Exit 0 iff met.
  check_goal.py --gate     Stop-hook mode. Exit 0 to allow the turn to end;
                           exit 2 with a reason on stderr to block it. A
                           missing goal file allows the stop, so the hook is
                           inert until Step 1 creates it.

Goal file:
  {
    "task": "llama MLP block, bf16",
    "backend": "cute",
    "baseline": "inductor max_autotune + coordinate_descent_tuning",
    "status": "running",            # or "paused" / "blocked", see below
    "paused_reason": null,          # required when status == "paused"
    "question_for_user": null,      # required when status == "blocked"
    "targets": [
      {"name": "mlp fwd 8x4096",
       "ratio": 1.044, "artifact": "inductor_artifacts/<run-name>/x.json"},
      {"name": "regression corpus (worst of 51)",
       "ratio": 0.997, "artifact": "inductor_artifacts/<run-name>/corpus.json"}
    ]
  }

A target's `ratio` is baseline time / helion.inductor time from one process on
one GPU (above 1 means helion.inductor is faster). `status` is only an escape
hatch: "paused" when the user explicitly said to stop (quote them in
paused_reason), "blocked" when the goal is unreachable within scope (the
question, with the evidence, in question_for_user). A goal file left from a
previous run's task is archived into its `inductor_artifacts/<run-name>/`.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

BAR = 0.99  # 1.0 minus the 1% run-to-run noise of a power-capped GPU
ARTIFACT_DIR = "inductor_artifacts"
GOAL_FILE = os.path.join(ARTIFACT_DIR, "goal.json")


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


def text(goal: dict, key: str) -> str:
    value = goal.get(key)
    return value.strip() if isinstance(value, str) else ""


def artifact_problem(root: str, artifact: object) -> str | None:
    """Why `artifact` cannot back a ratio, or None: it must name a file under
    the repo's inductor_artifacts/ directory."""
    if not isinstance(artifact, str) or not artifact:
        return "no artifact"
    base = os.path.realpath(os.path.join(root, ARTIFACT_DIR))
    path = os.path.realpath(os.path.join(root, artifact))
    if os.path.commonpath([base, path]) != base:
        return f"artifact outside {ARTIFACT_DIR}/"
    if path == os.path.realpath(os.path.join(root, GOAL_FILE)):
        return "the goal file is not an artifact"
    if not os.path.isfile(path):
        return "no artifact file on disk"
    return None


def target_problems(root: str, target: object) -> list[str]:
    """Why `target` fails: no finite positive ratio (JSON true, Infinity and
    NaN are not), a ratio under the bar, or no artifact."""
    if not isinstance(target, dict):
        return ["not an object with a ratio and an artifact"]
    ratio = target.get("ratio")
    problems = []
    if not (
        isinstance(ratio, (int, float))
        and not isinstance(ratio, bool)
        and math.isfinite(ratio)
        and ratio > 0
    ):
        problems.append(f"no measured ratio ({ratio!r})")
    elif ratio < BAR:
        problems.append(f"ratio {ratio:.3f} < bar {BAR:.2f}")
    problem = artifact_problem(root, target.get("artifact"))
    if problem:
        problems.append(problem)
    return problems


def load_goal(goal_path: str) -> dict:
    """The goal file's object; ValueError for a file that is not JSON, not an
    object, or whose targets are not a non-empty list."""
    goal = json.loads(open(goal_path).read())
    if not isinstance(goal, dict):
        raise ValueError("the goal file is not a JSON object")
    if not isinstance(goal.get("targets"), list) or not goal["targets"]:
        raise ValueError("targets is not a non-empty list")
    return goal


def evaluate(goal: dict, root: str) -> tuple[list[str], list[str]]:
    """The table lines and the failing targets."""
    lines, fails = [], []
    for target in goal["targets"]:
        name = target.get("name", "?") if isinstance(target, dict) else "?"
        problems = target_problems(root, target)
        if problems:
            fails.append(f"{name} ({'; '.join(problems)})")
            lines.append(f"FAIL {name}: {'; '.join(problems)}")
        else:
            lines.append(f"PASS {name}: {target['ratio']:.3f}  [{target['artifact']}]")
    return lines, fails


def escape(goal: dict) -> str | None:
    """Why the turn may end with the goal unmet, or None; raises ValueError
    for an escape hatch without its explanation."""
    status = goal.get("status", "running")
    if status == "paused":
        if not text(goal, "paused_reason"):
            raise ValueError("status is 'paused' but paused_reason is empty")
        return "PAUSED at the user's instruction"
    if status == "blocked":
        if not text(goal, "question_for_user"):
            raise ValueError("status is 'blocked' but question_for_user is empty")
        return "BLOCKED, awaiting the user's decision"
    return None


def gate(goal_path: str, root: str) -> int:
    if not sys.stdin.isatty():
        sys.stdin.read()  # consume the hook's JSON payload
    if not os.path.exists(goal_path):
        return 0  # no active hillclimb in this repo
    try:
        goal = load_goal(goal_path)
        if escape(goal) is not None:
            return 0
        _lines, fails = evaluate(goal, root)
    except Exception as e:  # fail closed: a malformed goal file blocks
        print(
            f"inductor-hillclimb stop gate: cannot evaluate {goal_path} ({e}); "
            "fix the goal file.",
            file=sys.stderr,
        )
        return 2
    if not fails:
        return 0
    print(
        f"inductor-hillclimb stop gate: goal NOT met, {len(fails)} failing: "
        + "; ".join(fails)
        + ". Go back to Step 2 and keep optimizing the worst failing target (or "
        "fix the regression); do not write a final report. Only with cause: set "
        f"status 'paused' + paused_reason in {GOAL_FILE} if the user told you to "
        "stop, or 'blocked' + question_for_user if the goal contract's evidence "
        "says the goal is unreachable within scope. A goal file of a previous "
        f"run's task goes into its {ARTIFACT_DIR}/<run-name>/ directory.",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    root = repo_root()
    goal_path = os.path.join(root, GOAL_FILE)
    if "--gate" in sys.argv[1:]:
        return gate(goal_path, root)
    if not os.path.exists(goal_path):
        print(f"no goal file at {goal_path} (Step 1 of the skill creates it)")
        return 1
    try:
        goal = load_goal(goal_path)
        reason = escape(goal)
    except ValueError as e:
        print(f"invalid goal file: {e}")
        return 1
    lines, fails = evaluate(goal, root)
    print(f"baseline: {goal.get('baseline')!r}, backend: {goal.get('backend')!r}")
    print("\n".join(lines))
    if not fails:
        print("GOAL MET")
        return 0
    print(f"GOAL NOT MET — {reason}" if reason else "GOAL NOT MET")
    return 1


if __name__ == "__main__":
    sys.exit(main())
