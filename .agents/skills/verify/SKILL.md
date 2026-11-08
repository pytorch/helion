---
name: verify
description: Verify a change before committing it or handing it back — lint, choose the test lanes that match what changed, run them with scripts/run_tests.py on GPUs from the board, review with autoreview when the change warrants it, then fix and repeat. Use when a change is ready to commit, after a rebase, or before a stack is handed back for submission.
---

# Verify a change

The policy is in `AGENTS.md` (How Much to Test, GPU Usage, Code Review); this
skill is the procedure. Run the checks that fit the change, fix what they
find at the root (never by suppressing a test), and rerun only what is
affected.

## 1. Lint

`./lint.sh` runs the pre-commit hooks the Lint CI job runs, with their pinned
tool versions, and applies their fixes; `./lint.sh check <paths>` is CI's
single pass: it fails when a hook reports a problem or changes a file and
shows the change. A clean tree before you start makes the `git status`
afterwards meaningful.

## 2. Decide the lanes

Write the decision down in one line before running anything, for example
"core change (compile_environment.py): Triton and CuTe lanes, PR tier" or
"test-only change: that module on cute". The table in `AGENTS.md` maps changed
paths to lanes; `scripts/run_tests.py --changed` lists the modules that name
your changed sources and runs them, and tells you when a change is too broad
for a selection.

Include the slow tier (`--slow`) for slow tests you touched and for autotune
search or interpreter-spawning changes. New test files need a
`pytest --durations=20` look, and heavy ones a `scripts/shard_durations.py`
re-record.

## 3. GPUs

`scripts/gpu_board.py status` shows what is free. `run_tests.py` picks and
claims GPUs itself (`-n K`, or `--gpus 2,5` for specific ones) and releases
them when it finishes; it refuses to start when nothing is free, or when a GPU
you named is claimed by another account or busy, rather than piling onto
someone's GPU (`--force` overrides). The CPU lane needs no GPU. Never run two
suites on one GPU.

## 4. Run

```
scripts/run_tests.py --changed                              # modules naming the sources changed since origin/main
scripts/run_tests.py --changed --since HEAD~1               # ... changed by the last commit
scripts/run_tests.py --backend cute --lane cpu               # CuTe codegen check, no GPU
scripts/run_tests.py --backend cute --gpus 2,5               # CuTe lanes, GPU lane over two GPUs
scripts/run_tests.py --backend triton,cute -n 2              # core change: both backends
scripts/run_tests.py --slow test/test_autotuner.py           # one module including its slow tier
scripts/run_tests.py --backend cute test/test_cute_x.py -- -k softmax   # pytest options after --
```

Each run writes logs, junit files, the exact commands and `summary.txt` under
`.logs/tests/<timestamp>-<tag>/`, and prints a `VERDICT` line (if the
script itself fails rather than reporting a verdict, stop and report it, per
`AGENTS.md`):

- `PASS`: everything passed.
- `FAIL`: a test still fails alone, or a job crashed (read its log; an
  out-of-memory failure includes a GPU memory report).
- `FLAKY`: every failure passed in isolation. That is order dependence, usually
  state hung on a cached compiled kernel that an earlier test left behind; make
  the test independent of earlier launches instead of retrying.

Long runs go in a worktree if you want to keep editing meanwhile; editing the
checkout a run reads corrupts its results.

## 5. Review

`scripts/autoreview.py` (uncommitted changes) or `scripts/autoreview.py
--head` (the last commit) when the change is a complete logical unit; skip it
for trivial fixes and run it once over the finished stack
(`scripts/autoreview.py origin/main..HEAD`) before handing a larger project
back. Address every finding or explain why it does not apply. If the review
itself fails, stop and report it; do not substitute your own review.

## 6. Report

A short table of what ran: backend, lane, GPUs, result, and the review
outcome, plus anything deliberately not run and why.

Notes:

- Don't pass `HELION_AUTOTUNE_EFFORT=none` to a lane run; it changes execution
  paths and breaks tests. Use it only for targeted local iteration.
- `pytest -ra` prints skip reasons when a skip count looks suspicious.
