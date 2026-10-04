# Rebase onto main, 2026-10-02

The nine-commit stack on `cute-sequence-skill-completion-20260930` has been rebased onto `d85472b02010381a89cb023288fdecd2161020bc`, the fetched `origin/main`. Main advanced by 29 commits from the former base `256f2c8c510312a444c56b03590aee8e3fc574c0`.

The original tip, `815437bdd06bf59e3db1c578b322d91ce4cd0874`, remains at `backup/cute-sequence-before-main-rebase-20261002T185946Z`. The unrelated primary worktree and its branch were not changed. Nothing was pushed or installed.

## Conflict resolution

- Retained main's per-tensor actual-pointer alignment, schema and disk-cache keys, alignment guards for fast relaunch, vector-alignment specialization, and new wrapper paths. Integrated the stack's kernel metadata and helper fingerprints with them. Main supersedes the old scalar-pointer AST proof, so commit 8 now contains alignment regression tests only. Its tests cover cache separation, stronger/weaker views, numerical results and graph replay; the GPU cases still require execution after this rebase. See [alignment rationale](alignment-rebase-plan.md).
- Combined main's structural, RNG, prefetch and materialization settings with the stack's chained-graph and loop-state settings. Kept native ownership priority, main's final AST handling, and metadata validation before shape baking. Preserved main's broader synthetic-lane contraction guard, including dynamic reduction extents.
- Preserved typed marker ownership when replaying affine producers into shared plans, with failures leaving the original IR unchanged. Added seven ownership/rebinding cases. See [affine review](affine-merge-review.md) and [compiler review](compiler-merge-review.md).
- Moved the shared packed store into `packed_store.py` so CPU code generation does not eagerly depend on `cutlass.experimental`. Existing runtime callers re-export the identical function; its decorator and body are unchanged. The new subprocess regression reproduces the import failure before the repair and skips when the optional SDK is absent. See [import-isolation repair](cpu-import-isolation.md).
- Preserved private timing inputs while adopting main's `pre_warmed`, long-probe and worker-cleanup protocol. Profiler timing still collects attributed evidence when the ordinary timer reuses a probe. Fresh-worker finalist timing, fixed repetitions and main's incumbent-selection guard remain intact. The timer fixture adjustment is folded into commit 2. See [profiler resolution](profiler-conflicts.md) and [independent Opus review](profiler-merge-independent-review.md).

## Validation of the rebased source

| Check | Result |
|---|---|
| `./lint.sh fix` | Ruff and codespell pass; only the 41 missing JAX/Pallas imports explicitly waived by the user remain |
| Combined targeted CPU coverage | 731 passed, 72 CUDA skips, 19 subtests passed; no unresolved CPU failure |
| Optional SDK absent | Fresh subprocess import-blocker probe confirms the regression test skips when `cutlass` is absent |
| Independent reviews | Compiler, affine, positional and full history review LGTM; independent Opus profiler and final runtime reviews LGTM |
| Full default / CuTe GPU suites | Not run after the rebase |

The CPU count is deduplicated across three targeted runs; it is not a full-suite run. The first selection had 658 passes, 70 skips and ten failures. Eight failures were GPU-only prefill cases inadvertently selected with CUDA hidden; they failed at allocation and remain outside the passing CPU scope. Two were formatting-dependent bounds assertions, repaired in commit 7 without changing compiler behavior. The follow-up passed 82 tests and 19 subtests with 16 CUDA skips, including the repaired file and additional multi-shape/structural-cache checks. The final optional-SDK guard regression passed separately and is folded into commit 3.

All 1,725 files in the final frozen source match the working source. The only changes between the first CPU snapshot and the final snapshot are the two test corrections above; production source is identical. See the [deduplicated CPU receipt](cpu-validation-summary.json), [initial run](cpu-integration-v1/run.log), [follow-up](cpu-followup-v2/run.log), [optional-SDK test](cpu-optional-dependency-v3/run.log), [missing-SDK probe](missing-sdk-skip-probe.log), and [final source inventory](validation-v3/source-inventory.json). The [first runtime review](runtime-integration-independent-review.md) identified the optional-SDK guard; the [final Opus review](runtime-integration-independent-review-v2.md) confirms it is resolved. [History integration review](final-history-integration-review.md) found no unexplained lost changes.

The CPU import regression exercises static absolute imports through `builtins.__import__`; it does not claim coverage of dynamic `importlib.import_module` calls. The scalar-view GPU tests also contain shared-kernel and alignment-cache assertions that still require GPU verification before landing.

Full default and CuTe GPU suites were not run after the rebase. The preflight detected another job on GPU 2 before any pytest process launched. GPUs 0, 1 and 3 had already been excluded earlier in this session. The [cute-hillclimb skill](../../.claude/skills/cute-hillclimb/SKILL.md) requires: “if a GPU is in use by someone else, avoid that GPU for the rest of the session”. No eligible local GPU remained. See the [preflight](integration-v1/gpu-preflight.txt) and [not-run receipt](gpu-validation-not-run.json). Both full suites must be run sequentially on an eligible idle GPU for full validation of this rebase.

The previous full-suite passes and performance measurements remain historical evidence for the previous source. No new timing, cold autotune, generated-source applicability audit, or GPU performance claim was made for the rebased commits. The unchanged 46-row ledger still prints `GOAL MET` with geometric mean 1.0367 against the configured minimum 1.01; this is a check of preserved measurements, not qualification of the new source. See [historical checker output](historical-goal-checker.log) and the [historical hillclimb report](../skill-completion-20260930/REPORT.md).

## Rebased history

| Commit | Change |
|---|---|
| `658f37845752` | [benchmark] Add fused KDA kernels and reproducible backend comparisons |
| `c956b69e3bfc` | [autotuner] Isolate candidate inputs throughout compilation and timing |
| `4a71cfd8ef95` | [cutedsl] Lower recurrent contraction graphs with shared storage plans |
| `808744a969c6` | [cutedsl] Bind native prefill schedules to shared graph lowering |
| `21369b6aaf2c` | [cutedsl] Schedule prepared warp tiles and tune external state transport |
| `24164f761c5f` | [autotuner] Add opt-in profiler timing with checked search evidence |
| `d0f2a6466f39` | [cutedsl] Lower colliding axes and split reductions with positional phases |
| `a5f69cbd68f7` | [cutedsl] Cover scalar-view alignment in cached launches |
| This final commit | [noland] Record shared CuTe hillclimb evidence and verification |

The benchmark commit remains first, followed by logical backend and autotuner commits. The only `[noland]` commit remains last and holds planning material and reports. Raw measurements, configs, validation logs and `artifacts/goal.json` stay uncommitted. The [range diff](range-diff-final.txt) records the completed replay; the [final audit](final-history-audit.json) records the completed history and source inventory.
