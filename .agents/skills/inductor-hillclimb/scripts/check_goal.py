#!/usr/bin/env python3
"""Goal gate for the inductor-hillclimb skill.

Reads `inductor_artifacts/goal.json` (created in Step 1 of the skill) and
decides whether the run's goal is met:

- EVERY target's baseline / helion.inductor time ratio is at least 1.0 minus
  a fixed 1% noise margin (bar 0.99), and the geomean of the ratios meets
  `criteria.geomean_min`, which may be raised above 1.0 but not lowered;
- the regression corpus's worst before / after time ratio (`corpus`) is at
  least the bar;
- every ratio is a finite positive number backed by an artifact file under
  `inductor_artifacts/` (not the goal file itself);
- and the stage is the last one: a stage 1 that passes still blocks, until
  the goal file is rewritten for stage 2 and that passes too, unless the
  prompt limited the run to stage 1 (`final_stage`: 1, with the prompt's
  words quoted in `final_stage_reason`).

The skill keeps its goal file and scratch outputs in `inductor_artifacts/`,
apart from cute-hillclimb's `artifacts/`, so a cute-hillclimb run started from
this skill keeps its own goal and Stop gate.

Modes:
  check_goal.py            Human-readable PASS/FAIL table. Exit 0 iff met.
  check_goal.py --gate     Stop-hook mode (the same Stop-hook machinery the
                           built-in /goal command uses). Exit 0 to allow the
                           turn to end; exit 2 with a reason on stderr to
                           block it. A missing goal file allows the stop, so
                           the hook is inert until Step 1 activates it.

Goal file schema:
  {
    "task": "llama MLP block, bf16",
    "stage": 1,                     # 1 or 2
    "backend": "cute",              # helion.backend, recorded for the labels
    "final_stage": 2,               # optional; 1 when the prompt says so,
    "final_stage_reason": null,     # ... quoting it here
    "baseline": "inductor max_autotune + coordinate_descent_tuning",
    "status": "running",            # or "paused" / "blocked", see below
    "paused_reason": null,          # required when status == "paused"
    "question_for_user": null,      # required when status == "blocked"
    "criteria": {"geomean_min": 1.0},
    "targets": [
      {"name": "mlp fwd 8x4096",
       "ratio": 1.044, "artifact": "inductor_artifacts/<run-name>/x.json"}
    ],
    "corpus": {"ratio": 0.997, "artifact": "inductor_artifacts/<run-name>/corpus.json"}
  }

`baseline` names the stage's baseline: stock Inductor's best configuration in
stage 1, and in stage 2, per target, the faster of that and the best known
implementation. A target's `ratio` is
baseline time / helion.inductor time from one process on one GPU (>1 means
helion.inductor is faster), from the newest cold full-effort measurement
(interleaved ABAB verify runs when they exist). The corpus's `ratio` is the
worst before / after time ratio among the regression-corpus graphs re-measured
in Step 3, and its artifact lists them. `status` is only an escape hatch:
"paused" when the user explicitly said to stop (quote their instruction in
paused_reason), "blocked" when the goal contract's evidence says the goal is
unreachable within scope (put the question, with the evidence, in
question_for_user). "Met" is never declared — it is computed here. A goal file
left over from a PREVIOUS run's task should be archived into its old
`inductor_artifacts/<run-name>/` directory, not evaluated.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

# Run-to-run spread on a power-capped B200 is about 1%.
NOISE_MARGIN_PCT = 1.0
BAR = 1.0 - NOISE_MARGIN_PCT / 100.0
ARTIFACT_DIR = "inductor_artifacts"
GOAL_FILE = os.path.join(ARTIFACT_DIR, "goal.json")
STAGES = (1, 2)


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


def measured(ratio: object) -> bool:
    """Whether `ratio` is a finite number: JSON true, Infinity and NaN are not."""
    return (
        isinstance(ratio, (int, float))
        and not isinstance(ratio, bool)
        and math.isfinite(ratio)
    )


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


def check_entry(root: str, entry: object) -> tuple[float | None, list[str]]:
    """The entry's ratio (None unless measured and positive) and its problems."""
    if not isinstance(entry, dict):
        return None, ["not an object with a ratio and an artifact"]
    ratio = entry.get("ratio")
    problems = []
    value = None
    if not measured(ratio):
        problems.append("no measured ratio")
    elif ratio <= 0:
        problems.append(f"ratio {ratio!r} is not a positive number")
    else:
        value = float(ratio)
        if value < BAR:
            problems.append(f"ratio {value:.3f} < bar {BAR:.3f}")
    problem = artifact_problem(root, entry.get("artifact"))
    if problem:
        problems.append(problem)
    return value, problems


def evaluate(goal: dict, root: str) -> tuple[bool, list[str], list[str]]:
    """Whether the current stage's criteria are met, with the table lines and
    the failure summaries."""
    lines: list[str] = []
    fails: list[str] = []
    ratios: list[float] = []
    targets = goal.get("targets") or []
    if not targets:
        return False, ["goal file lists no targets"], ["no targets listed"]
    for target in targets:
        name = target.get("name", "?") if isinstance(target, dict) else "?"
        value, problems = check_entry(root, target)
        if value is not None:
            ratios.append(value)
        if problems:
            fails.append(f"{name} ({'; '.join(problems)})")
            lines.append(f"FAIL {name}: {'; '.join(problems)}")
        else:
            lines.append(
                f"PASS {name}: ratio {value:.3f} >= bar {BAR:.3f}  "
                f"[{target['artifact']}]"
            )
    geomean_min = geomean_bar(goal)
    if len(ratios) == len(targets):
        # In logs: a product of large ratios would overflow to inf.
        geomean = math.exp(math.fsum(map(math.log, ratios)) / len(ratios))
        ok = geomean >= geomean_min
        lines.append(
            f"{'PASS' if ok else 'FAIL'} geomean: {geomean:.4f} (min {geomean_min})"
        )
        if not ok:
            fails.append(f"geomean {geomean:.4f} < {geomean_min}")
    else:
        lines.append("geomean: not computable (unmeasured targets)")
    corpus = goal.get("corpus")
    if corpus is None:
        problems = ["not checked (no corpus entry)"]
    else:
        value, problems = check_entry(root, corpus)
    if problems:
        fails.append(f"regression corpus ({'; '.join(problems)})")
        lines.append(f"FAIL regression corpus: {'; '.join(problems)}")
    else:
        lines.append(
            f"PASS regression corpus: worst ratio {value:.3f} >= bar {BAR:.3f}  "
            f"[{corpus['artifact']}]"
        )
    return not fails, lines, fails


def geomean_bar(goal: dict) -> float:
    """The goal's geomean minimum: a finite number of at least 1.0 (the goal
    contract's bar, which a goal file may raise but not lower); raises
    ValueError for others."""
    criteria = goal.get("criteria") or {}
    value = criteria.get("geomean_min", 1.0) if isinstance(criteria, dict) else None
    if not measured(value) or value < 1.0:
        raise ValueError(f"criteria.geomean_min {value!r} is not a number >= 1.0")
    return float(value)


def stage_number(value: object) -> bool:
    """Whether `value` is 1 or 2: JSON true, which Python equals to 1, is not."""
    return isinstance(value, int) and not isinstance(value, bool) and value in STAGES


def stages(goal: dict) -> tuple[int, int]:
    """The goal's stage and final stage; raises ValueError for others."""
    stage = goal.get("stage", 1)
    final = goal.get("final_stage", 2)
    if not (stage_number(stage) and stage_number(final)) or stage > final:
        raise ValueError(f"stage {stage!r} / final_stage {final!r}")
    if final == 1 and not text(goal, "final_stage_reason"):
        raise ValueError(
            "final_stage is 1 but final_stage_reason is empty: quote the "
            "prompt's limit there, or set final_stage back to 2"
        )
    return stage, final


def next_stage_instruction() -> str:
    return (
        f"Copy the goal file to {ARTIFACT_DIR}/<run-name>/goal-stage1.json, then "
        f'rewrite {GOAL_FILE} in place for stage 2: "stage": 2, a baseline that '
        "is, per target, the faster of the stage-1 baseline and the best known "
        "implementation, and every target's ratio against it. "
        "Then return to Step 2."
    )


def gate(goal_path: str, root: str) -> int:
    if not sys.stdin.isatty():
        sys.stdin.read()  # consume the hook's JSON payload
    if not os.path.exists(goal_path):
        return 0  # no active hillclimb in this repo
    try:
        return _gate_verdict(goal_path, root)
    except Exception as e:  # fail closed: a malformed goal file must block
        print(
            f"inductor-hillclimb stop gate: cannot evaluate {goal_path} ({e}); "
            "fix the goal file.",
            file=sys.stderr,
        )
        return 2


def _gate_verdict(goal_path: str, root: str) -> int:
    goal = json.loads(open(goal_path).read())
    status = goal.get("status", "running")
    if status == "paused":
        if text(goal, "paused_reason"):
            return 0
        print(
            "inductor-hillclimb stop gate: status is 'paused' but paused_reason "
            "is empty. Quote the user's instruction to pause there, or set "
            "status back to 'running'.",
            file=sys.stderr,
        )
        return 2
    if status == "blocked":
        if text(goal, "question_for_user"):
            return 0
        print(
            "inductor-hillclimb stop gate: status is 'blocked' but "
            "question_for_user is empty. State the question, with the goal "
            "contract's evidence, or set status back to 'running'.",
            file=sys.stderr,
        )
        return 2
    stage, final = stages(goal)
    met, _lines, fails = evaluate(goal, root)
    if met and stage == final:
        return 0
    if met:
        print(
            f"inductor-hillclimb stop gate: stage {stage} is met, but the run "
            f"ends at stage {final}. {next_stage_instruction()}",
            file=sys.stderr,
        )
        return 2
    n = len(goal.get("targets") or [])
    print(
        f"inductor-hillclimb stop gate: stage {stage} NOT met — {len(fails)} "
        "failing: "
        + "; ".join(fails)
        + ". Return to Step 2 of the skill and keep optimizing the worst "
        "failing target (or fix the regression); do not write a final report "
        "or describe the work as done. Escape hatches (only with cause): if "
        "the user explicitly told you to stop, set status 'paused' + "
        f"paused_reason in {GOAL_FILE}; if the goal contract's evidence says "
        "the goal is unreachable within scope (the hand-written Helion version "
        "still loses after the Helion compiler work, or the change needs a hook "
        "PyTorch does not have), set status 'blocked' + question_for_user. If this "
        "goal file belongs to a PREVIOUS run's task, archive it into its old "
        f"{ARTIFACT_DIR}/<run-name>/ directory and write a fresh one for the "
        "current task." + (f" ({n} targets total.)" if n else ""),
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
    goal = json.loads(open(goal_path).read())
    status = goal.get("status", "running")
    if status != "running":
        print(f"status: {status}")
    try:
        stage, final = stages(goal)
        geomean_bar(goal)
    except ValueError as e:
        print(f"invalid goal file: {e}")
        return 1
    print(f"stage {stage} of {final}: baseline {goal.get('baseline')!r}")
    if final == 1:
        print(f"final stage 1: {text(goal, 'final_stage_reason')}")
    met, lines, _fails = evaluate(goal, root)
    print("\n".join(lines))
    if met and stage == final:
        print("GOAL MET")
        return 0
    if met:
        print(f"STAGE {stage} MET — NOT DONE. {next_stage_instruction()}")
        return 1
    if status == "blocked" and text(goal, "question_for_user"):
        print(
            "GOAL NOT MET — BLOCKED, awaiting the user's decision "
            "(question_for_user is set; the stop gate allows ending the turn)"
        )
    elif status == "paused" and text(goal, "paused_reason"):
        print(
            "GOAL NOT MET — PAUSED at the user's instruction "
            "(the stop gate allows ending the turn)"
        )
    else:
        print("GOAL NOT MET")
    return 1


if __name__ == "__main__":
    sys.exit(main())
