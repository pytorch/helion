# Cross-Loop Static Pipeline Plan

This is the living plan for preparing the static cross-loop scheduler for
upstream review. Update the checklist and decisions as the implementation
changes.

## Submission goals

- Preserve the correctness and supported behavior of the implementation on
  `helion-cross-kernel`.
- Keep `helion-cross-kernel-static-pipeline` as the source-of-truth branch for
  this PR.
- Reduce review size through deduplication and coherent stacked prerequisites,
  not by deleting conservative fallbacks, semantic cases, or the independent
  schedule oracle.
- Keep `cross_loop_schedule="barrier"` as the safe default.
- Keep the existing public `num_sm_multiplier` choices: `1`, `2`, `4`, and `8`.

## Current baseline

The initial static-pipeline commit is the scheduler implementation extracted
from `helion-cross-kernel`. At the start of this cleanup, the corresponding
scheduler, codegen, and test files were byte-for-byte identical.

The diff against `helion-cross-kernel-dependencies` contained approximately:

- 4,376 lines of scheduler and codegen churn.
- 5,996 lines of test churn.
- 1,273 lines of integration churn in existing compiler/runtime files.

The main size problem is therefore test construction and the combination of
several independently reviewable layers, not duplicated alternative scheduler
implementations.

After the mechanical cleanup, the implementation and tests are approximately
10,714 additions against `helion-cross-kernel-dependencies`, 806 fewer than the
original commit (excluding this living-plan document). The remaining reduction
should come from stacking coherent implementation layers, not deleting
scheduler behavior.

## Non-negotiable correctness constraints

- Keep symbolic dependency relations compact; production scheduling must not
  materialize task-count-sized relations.
- Preserve exact readiness when it is provable and conservative root/family
  fallback when it is not.
- Preserve multi-producer joins, fanout, nested-loop readiness, partial and
  strided accesses, noncanonical origins, and independent graph components.
- Preserve CUDA Graph replay and persistent state epoch behavior.
- Preserve the independent materialized test oracle.
- Do not remove a test scenario solely to reduce line count.
- Do not merge a static polling path whose progress guarantee can deadlock in a
  supported execution environment.

## Immediate cleanup

- [x] Rebase the original static-pipeline change onto the latest
  `helion-cross-kernel-dependencies` commit.
- [x] Remove the `CROSS_LOOP_SCHEDULE_CONFIG` key-name constant and use the
  normal `"cross_loop_schedule"` config plumbing.
- [x] Separate backend capability from per-kernel eligibility: expose
  `cross_loop_schedule` through an optional `EnumFragment`, like other
  conditional config fields, without special-casing `supports_config_key()`.
- [x] Read the selected policy through `Config.cross_loop_schedule` in codegen.
- [x] Restore the public `num_sm_multiplier` annotation and documentation to
  `1 | 2 | 4 | 8`; the base branch's internal maximum remains unchanged.
- [x] Remove unused `_maximum_axis_task_count()` and `task_id_expr()` helpers.
- [x] Restore `_prepare_persistent_body()`'s existing `list[ast.AST]` interface;
  keep the scheduler-specific narrowing local to cross-loop codegen.
- [x] Remove the unrelated `KernelCompiler` documentation-only edit.
- [x] Reject unresolved allocation identities instead of silently dropping
  their memory operations from multi-root dependency analysis. Keep this fix
  and its regression in the `helion-cross-kernel-dependencies` prerequisite.
- [x] Derive task families and noncanonical task origins from `device_ir` at
  the production dependency-builder call site; keep source phases explicit
  because installed dependency phases later replace them.
- [x] Run formatting, lint, production-file type checking, and focused tests
  after the structural cleanup.

## Reduce `test/test_cross_loop_scheduler.py`

Keep all named semantic scenarios. Reduce construction boilerplate first; do
not replace the suite with one opaque mega-parameterized test.

Introduce compact helpers directly in `test/test_cross_loop_scheduler.py` for:

- `site_domain(*axis_specs, identity=None)` for ordinary configured domains.
- `event_domain(*counts, identity=None)` for readiness-key domains.
- `access(root, kind, axes=..., ...)` plus
  `dependency_graph(root_axes, *accesses)`, assigning access IDs by order.
- `full_point_map(source, target, *expressions)` only for full-domain,
  single-valued relations. Keep unusual partial, strided, multi-piece, and
  deliberately non-lowerable relations explicit.
- `task_segment(...)`, while keeping worker span and dispatch offset explicit
  because those values are scheduler behavior under test.
- Clearly named configured adapters such as `configured_readiness_events()` and
  `configured_static_pipeline_plan()` rather than shadowing production
  `build_*` functions.

Do not add cross-test utility modules. Keep each test file self-contained, even
when a small fixture must be repeated. Table-drive only tests that express two
sides of one invariant without hiding their legality argument; the first
candidate is equivalent fanout coalescing versus swapped-axis non-coalescing.

Result: the scheduler tests and their former 364-line oracle now occupy 3,203
lines in one self-contained file, down from 3,935 lines combined. All 47 test
methods and all 205 assertions remain.

After the mechanical rewrite:

- [x] Compare the collected test names before and after.
- [x] Verify every former scenario appears either as a named test or named
  `subTest` case.
- [x] Run the independent schedule oracle over the same cases.
- [x] Review the compact helpers for hidden defaults that could weaken tests.

## Reduce unrelated blast radius

The following changes affect ordinary kernels and should not remain buried in
the scheduler feature diff. Preserve them as stacked prerequisites if required
for performance or correctness; do not silently discard them.

- Per-loop alias-aware `range_num_stages` disabling in
  `compile_environment.py`, with its host alias analysis and loop tests.
- The one-warp `range_num_stages` behavior change in `tile_strategy.py` and its
  runtime test.
- Generic outlined Triton helper support in `device_function.py`.
- Persistent Triton state allocation and exact occupancy validation in the
  launcher.

For every changed shared interface, verify both the feature path and the
unchanged default path. In particular:

- `DeviceFunction.body` and persistent-body preparation types.
- AST cloning and preservation of Helion metadata.
- helper-function emission when no cross-loop helpers exist.
- ordinary launcher behavior when no persistent-state or residency arguments
  are supplied.
- non-cross-loop range-pipelining behavior.

## Correctness issue requiring an explicit decision

The current occupancy check proves only theoretical whole-grid residency on an
otherwise idle device. Another stream can consume CTA slots after the check,
leaving polling consumers resident while a required producer is queued.

Before enabling static polling as supported behavior, choose and test one of:

1. a launch mechanism that guarantees progress for the complete worker cohort;
2. serialization or an exclusivity guard for static-pipeline launches; or
3. a narrow, explicit execution contract that rejects unsupported concurrent
   use.

Add a bounded two-stream contention regression. Keep the existing CUDA Graph
and repeated-epoch coverage.

## Adopted review stack

Split the implementation into two stacked PRs on top of
`helion-cross-kernel-dependencies` without changing the final feature set.

1. **Pure scheduler/model** (approximately 5.58K additions): add
   `cross_loop_scheduler.py`, `CrossLoopSchedulingError`, and the 46 CPU/pure
   scheduler tests with the independent materialized oracle. Do not expose a
   config option or change compiler, codegen, launcher, or runtime behavior in
   this PR. Omit the two bound Helion kernels and the CUDA DeviceIR-site
   integration test from this layer.
2. **Codegen and compiler/runtime integration** (approximately 5.14K additions
   before this plan document): add `cross_loop_codegen.py`, DeviceIR access and
   phase wiring, AST site propagation and outlined helpers, normal config
   integration, persistent runtime state and occupancy validation, and all
   generated-code/end-to-end tests. Restore the two kernels and DeviceIR-site
   integration test here. Keep the safe `barrier` default and expose
   `static_pipeline` only together with its lowering and runtime support.

The scheduler PR is intentionally dormant outside direct imports. Freeze its
dataclasses and `build_static_pipeline_plan()` boundary before review so
integration changes do not repeatedly invalidate the first review. Verify 46
test methods and 187 assertions in the scheduler PR, then all 47 methods and
205 assertions in the combined stack. Both PRs must be green independently,
and the combined stack must retain every current correctness scenario.

Keep this living plan in the top integration branch rather than presenting it
as product documentation; its contents should become the stacked PR
descriptions before submission.

## Final validation

- [x] Ruff formatting and lint, plus repository-wide codespell.
- [x] Pyrefly on every changed production file: zero errors. The full-project
  wrapper remains environment-blocked because this checkout's configured
  `/home/eche/local/pytorch` search path is absent; its remaining diagnostics
  are missing third-party imports and existing test-file diagnostics.
- [ ] Complete tile-dependency, scheduler, codegen, config, loop, and runtime
  tests.
- [ ] Affine-chain compile-scaling check.
- [ ] Structural lowering comparison for the canonical kernels.
- [ ] Cold-L2 Qwen3, Gemma4 A4B/E4B, GPT-OSS, and DeepSeek-V3 measurements.
- [ ] Two-stream progress test or enforced rejection contract.
- [x] Final independent reviewer pass over correctness and blast radius.

Current focused results: all 32 cross-loop codegen tests and 16 subtests pass;
all 150 selected config, scheduler, and tile-dependency tests and 39 subtests
pass (two CUDA-dependent integration tests were deselected). `git diff --check`
and bytecode compilation also pass.
