---
name: inductor-hillclimb
description: Make torch.compile with helion.inductor (cuda_backend="helion") generate the fastest code for a given model or FX graph — first beat the prompt's baseline (by default the best stock Inductor configuration, max-autotune + coordinate descent), then match or beat the best known implementation (hand-written Helion kernels, or the fastest library code for the pattern) — by general changes to the out-of-tree backend (lowering choices, fusion, scheduling, HelionKernel codegen, the wrapper; FX passes last) or, where Helion itself is the limit, the Helion compiler; then shape the pull requests and report before/after numbers.
hooks:
  Stop:
    - hooks:
        - type: command
          command: python3 "$CLAUDE_PROJECT_DIR/.claude/skills/inductor-hillclimb/scripts/check_goal.py" --gate
---

Your task is to make helion.inductor — `torch.compile` with `cuda_backend="helion"`, code in `helion/inductor/` — generate the fastest possible code for a given model or FX graph. Ask for the graphs and the GPU if the prompt leaves them out; the baseline, the Helion backend (`helion.backend`) and the shapes have the defaults below.

**The goal:** first beat the prompt's baseline on every target, by default the best code stock Inductor generates (at least `max_autotune=True` with `coordinate_descent_tuning=True`). Then match or beat the best known implementation of the same computation: Helion kernels you write by hand for the graph, a better-benchmark port, or the fastest library code for the pattern.

**The method:** write (or find) the Helion program that reaches the target, then change the backend so it generates that program. `pretuned_kernels/`, `examples/`, `benchmarks/` and better-benchmark's `kernels/` hold fast Helion kernels to start from. Where the generated Helion code is already what you would write and still loses, the Helion compiler is the limit: optimize it with the cute-hillclimb skill on CuTe, or change the Triton backend or the autotuner on Triton, under the same rules. Inside this run, use cute-hillclimb's optimize-and-measure loop only: write no `artifacts/goal.json` (its Stop gate would then block this run's stops; archive one left over), and skip its history cleanup and report.

# Rules for every change

helion.inductor is a general compiler: every change runs on every graph that reaches it. Never special-case a model, shape or kernel name; an optimization that fires only on the target is a special case in disguise. Everything lives out of tree, in `helion/inductor/` and Helion itself: use Inductor's hooks and subclass what they accept, add no hooks to PyTorch and monkeypatch nothing. If no hook can express a change, ask the user.

**Name dims by their tiles.** helion.inductor works on N-D tensors, where rows and columns mean nothing past 2-D. In code, comments, tests and pull request descriptions, call the pointwise dims z, y and x (tile_z, tile_y, tile_x) and the reduction dim r, its extent the rdim: "an rdim of 4096", "a sum over y", "tile_x's slice of w", never "a row of 4096" or "a column sum". Inductor's own names (`_native_matmul_row_epilogue_can_fuse`) and Helion's CuTe options (`cute_unpadded_rows`) keep theirs.

**Reuse before you build.** The bar for a completely new mechanism is high.
- First helion.inductor itself: when a feature already does part of the job, reuse it, extend it with a heuristic, or generalize it. Where several features overlap, merge them into one general mechanism during the hillclimb.
- Then Inductor: search `torch/_inductor` and its config (`torch/_inductor/config.py`) for the job, including flags that are off by default and features gated to Triton (`triton.cooperative_reductions` and `combo_kernels` are two examples). If one exists, do not reimplement it: turn it on for helion.inductor and decide when it applies in `HelionChoices`. For example, mix-order reductions use Inductor's node and check, with Helion's conditions in `HelionChoices` (https://github.com/pytorch/helion/pull/4582).
- Never change Inductor's process-global defaults; stock baselines compile in the same process. "On by default" means on under `cuda_backend="helion"` only, for example a helion.inductor option that defaults to on.
- When a change would help stock Inductor too (a Triton-only gate that should ask the backend, a split heuristic), implement it out of tree first, then prepare the pytorch/pytorch patch with tests and a description, and hand it to the user to open. The helion.inductor workaround says it goes once the upstream change reaches a torch nightly.

**Pick the most general layer.** Fix a divergence in lowering choices, fusion, scheduling, codegen or the wrapper (Deciding where an optimization belongs); then the Helion compiler; an FX pass last.
- An FX pass is for a computation whose form must change and that no other layer can express. Its pull request says why the other layers cannot.
- Graph rewrites are where reviews find correctness bugs (for example, the reviews of the cat pass in https://github.com/pytorch/helion/pull/4585 found most of the cases below). Check every rewrite against views and aliases (`as_strided`, `_unsafe_view`, graph outputs, ops that return their input such as `aten.positive`), layouts (channels-last, transposed operands, slices), in-place writes, and dtype promotion.
- When a general mechanism later covers a pass's cases, delete the pass. For example, an FX pass that split BatchNorm's NCHW var_mean gave way to a lowering choice, `HelionChoices.layout_split()` in `reduction_split_factor` (https://github.com/pytorch/helion/pull/4582).

**Watch compile time.** Every pass runs on every graph's compile. Avoid work quadratic or worse in a graph's nodes, a kernel's statements or its accesses: rescanning a body for each candidate, comparing all pairs. Count once and look up. For example, a review found the CuTe lane-jam pass counting a loop body's bindings once per candidate copy (https://github.com/pytorch/helion/pull/4579).

**Generality: correct everywhere, fires on the general pattern, declines elsewhere.** A change must give correct results on every graph. It should fire wherever its pattern appears in general form, and fall back to the stock path everywhere else. Draw the decline boundary from a principle ("the layout is not static"), never from a list of shapes. Check:
- Indexing changes: multi-dim tiles (2-D, 3-D, which dims are tiled, sizes that do not divide the block); indirect access (gathers, scatters, embedding lookups, indices read from another tensor); index expressions (affine with any coefficients, `//` and `%`, Min/Max, negative offsets, stride-0 broadcasts, masked boundaries of cat and pad, reshaped views).
- Lowering, fusion, scheduling and codegen changes: sizes (1, odd and prime, not a power of 2, not dividing the block, 0); strides (non-contiguous, transposed or permuted, channels-last, slice offsets, expanded inputs); rank (1-D to 4-D and up, lower-rank operands that broadcast); dtypes (fp32, bf16, fp16, fp64, int, bool, mixed promotion); ops (the whole family, e.g. sum/mean/amax/amin/prod/argmax/argmin/any/var_mean/online softmax; in-place ops; several outputs); dynamic shapes (at least a safe decline).

Each commit message gets a `Generality:` paragraph saying which of these the change covers and where it declines. Each change's tests cover at least one non-contiguous input, one size that does not divide the block, one other rank or broadcast, and one other dtype; an indexing change adds an indirect access and a `//` or `%` index. Codegen checks bind on CPU and numerics take a few small GPU tests (AGENTS.md, Test Budget Rules). Then run `<skill>/scripts/sweep_variants.py` over the change's examples: it compiles random variants of them (sizes from a pool without 0 unless an example lists it, layouts, the float dtype; rank, ops and static shapes stay as the example writes them), compares what a deterministic function returns (a graph that draws random numbers or writes into its arguments needs a test of its own) from stock Inductor, helion.inductor and eager against a float64 reference, and reports where the change fired, declined, failed to compile or lost accuracy. For example, its random layouts found that tiling by index dims had stopped compiling stride-0 graph inputs (https://github.com/pytorch/helion/pull/4580).

**Numerics: on average at least as accurate as eager.** The reference is the graph in float64 eager. Judge each input distribution on its own: normal, uniform, a large mean with a small variance, outlier channels, a wide log-uniform range, values constant along the last dim, values near the subnormal range, mixed signs that sum to about zero, signed zeros, NaN and inf, and values near overflow.
- helion.inductor's mean error, in units of the output dtype's spacing, is at most eager's plus 1. The +1 allows one op's rounding: Helion's fp32 division is within 2 ulp. Reduction order, decompositions and cancellation get no allowance beyond three standard errors of eager's mean, which matter only where the errors run to many ulp.
- Float outputs off by a quarter of their value (of the output's median magnitude, for a smaller value) are no more common than eager's, and absent where eager has none: a mean over many right elements hides a few wrong ones.
- Where the reference is no more precise than eager (a float64 graph, or one that casts to float32 itself, which its example declares), eager's error is 0 by construction: these two bounds are then the larger of eager's and stock Inductor's.
- Its share of inf or NaN outputs where the reference is finite is at most stock Inductor's. Compare shares rather than single inputs, because which input overflows depends on rounding.
- Integer and bool outputs, and inf or NaN of the reference, are missed no more often than eager misses them.

`sweep_variants.py` checks these. Put the numbers in a `Numerics:` paragraph of the commit message.

**Study a heuristic before you encode it.** An optimization that wins on the target rarely wins everywhere.
- Sweep it, forced on and off, across the features it plausibly depends on: shapes, dtypes, layouts, reduction sizes, GPUs.
- Write the rule over the features that separate winning from losing.
- Where the boundary is fuzzy, or a choice helps some graphs and hurts others, give it to the autotuner: a knob, or an emitted form whose config space covers both choices (for example, a persistent rdim that `reduction_loops` can roll, https://github.com/pytorch/helion/pull/4580). Heuristics are fine as autotuner seeds.
- Keep the sweep under `inductor_artifacts/` and summarize its evidence in the commit message.

# Goal contract — do not stop early

- **Goal file:** in Step 1 you create `inductor_artifacts/goal.json`, with one target per graph × shape and one for the regression corpus. After every measurement you report, update that target's `ratio` and `artifact`.
  - `ratio` = baseline time / helion.inductor time, from one process on one GPU. For the corpus it is the worst before / after time of the graphs Step 3 re-measured, in both phases below.
  - `artifact` = the raw result file under `inductor_artifacts/` that backs the ratio.
  - The schema is in `<skill>/scripts/check_goal.py`.
- **Done** means every target's ratio is at least 0.99 (1.0 within the 1% noise margin), from cold compiles at full autotune effort; interleaved ABAB runs are the ground truth where they exist. The targets start against the baseline. When they all pass, rewrite every graph target's ratio against the best known implementation (the corpus target keeps its before / after ratio), unless the prompt limits the run to the baseline: the gate then blocks again until those pass too.
- **Stop gate:** a Stop hook runs `<skill>/scripts/check_goal.py --gate` whenever you try to end your turn. While a target fails, it blocks the stop: go back to Step 2 instead of summarizing. Run `python3 .claude/skills/inductor-hillclimb/scripts/check_goal.py` yourself to see the verdict.
- **The only ways to end with the goal unmet:** the user tells you to stop (`"status": "paused"`, their words in `"paused_reason"`), or you have evidence the goal is unreachable within scope (`"status": "blocked"`, the question and the evidence in `"question_for_user"`). There are two kinds of evidence:
  - The hand-written Helion version of the worst target loses to what the target is measured against (the baseline, then the best known implementation) after a full autotune, and you could not close the gap in the Helion compiler.
  - The change needs a hook PyTorch does not have.

  A hand-written version that reaches it is not such evidence; it is the specification of your next change.
- **Context loss:** at the start of every iteration, re-read the goal file and your plan doc's log. After each iteration, append one line to the log: the worst target, the layer and idea tried, the result. Trust the checker over a summary.

# The pipeline and its hooks

| Stage | What happens | Out-of-tree hook |
|---|---|---|
| Dynamo, pre-grad | FX graph of torch ops | none for a backend; `config.pre_grad_custom_pass` belongs to users |
| AOTAutograd | functionalization, decompositions, forward/backward partitioning | the decomposition table (`torch._inductor.decomposition`), gated as below; the joint-graph custom passes belong to users |
| Post-grad | FX passes on the aten graph | the CUDA device custom pass, held by `HelionPostGradPass` (`helion/inductor/fx_passes/post_grad.py`), which chains the previous one |
| Lowering | aten ops become Inductor IR | `torch._inductor.lowering.user_lowerings` (returning None falls back); a `PatternMatcherPass` of Helion's own; the scheduling's `get_backend_features()`; the `InductorChoices` hooks asked while lowering, which `HelionChoices` can override: `use_native_matmul`, `reduction_split_factor`, `should_use_cooperative_reduction` |
| Scheduling | fusion, loop order, tiling, reduction strategy | `HelionChoices` (`helion/inductor/choices.py`: `can_fuse*`, `score_fusion`); the scheduling's `can_fuse_*`, `group_fn`, `select_tiling` |
| Kernel codegen | each fused group becomes one kernel | `HelionScheduling` (`codegen/scheduling.py`); `HelionKernel` (`codegen/kernel.py`) emits the Helion source; `HelionOverrides`, `HelionPrinter` (`codegen/overrides.py`); `HelionChoices.should_use_persistent_reduction` |
| Wrapper | Python that allocates buffers and launches the kernels | how `HelionScheduling` defines and calls the kernels; the wrapper codegen given to `register_backend_for_device` |
| Helion | each kernel compiles to Triton or CuTe and autotunes at its first call | the Helion compiler, backends and autotuner |

New code goes where its stage lives: `fx_passes/` (one module per pattern family), `lowering/` (create it with the first one), `choices.py`, `codegen/`, `support.py` (what Helion kernels support), `config.py` (the `helion.*` options). Two rules for every hook, because helion.inductor shares the `cuda` device with stock Inductor in one process:

- **Inert unless `cuda_backend == "helion"`,** so stock compiles, the baselines among them, are unaffected.
  - `lowerings` and Inductor's own pattern passes are process-global: leave them alone, and use `user_lowerings` and a pass of Helion's own.
  - A decomposition override is the one change to a shared table. Register it at import, before any compile (`select_decomp_table()` caches); keep the callable it replaces and call that unless `cuda_backend == "helion"` on a CUDA device, so stock entries behave as before.
  - Gate user lowerings the same way, falling back to the `user_lowerings` entry you replaced, which may be a user's.
- **Keep the cache key honest.** `config.py`'s `_source_hash` hashes `helion/inductor/`. Anything outside it that changes the generated code reaches Inductor's FX graph cache key another way: an option, or a custom pass's `uuid()`.

# Deciding where an optimization belongs

1. **Write the target program:** the Helion kernels you would write by hand. Make it correct against eager and faster than the baseline; it is the specification.
2. **Compare the generated program to it, outside in:**
   - kernel boundaries;
   - per kernel, loop structure, tiling, persistence, accumulators, indexing and memory access;
   - the host code between kernels;
   - the config.

   `TORCH_LOGS=post_grad_graphs,ir_post_fusion,fusion,output_code` and `run_and_get_code` show each step.
3. **Fix the first divergence at the most general layer that can express it:**

| Divergence | Layer |
|---|---|
| Same ops, different kernel boundaries | fusion policy in `HelionChoices`, or the split it decides while lowering (`reduction_split_factor`); `HelionScheduling` when Inductor's scheduler cannot express the grouping (mix-order reductions, matmul prologues and epilogues) |
| Same boundaries, different kernel structure or indexing | `HelionKernel` codegen, or the `HelionChoices` decision that feeds it (persistence, tiling, cooperative reduction) |
| Same kernels, different host code: extra views, copies or allocations, one storage passed as several arguments, launch and argument counts, CUDA-graph compatibility | the wrapper: how `HelionScheduling` defines and calls the kernels |
| Same Helion source as the target, but slower | the Helion compiler or backend (cute-hillclimb on CuTe) |
| Same source, and only a hand-picked config wins | the autotuner's search space or seeds |
| The target computes it differently (an op a decomposition broke up, an algebraic rewrite), and no layer above can express it | a gated decomposition, a user lowering or, last, an FX pass |

4. **Hand-edit before you implement.** Edit the emitted Helion source toward the target, measuring each edit; small differences matter. Implement only the winning edits, at their layer. For example, the order of an index expression can decide whether Helion proves it non-negative, and with it whether loads vectorize.

# Measuring

- **The corpus tool:** better-benchmark's harness in `benchmarks/cute/better_benchmark/` (its README).
  - `bench.py` times stock Inductor with coordinate descent (`inductor_cd`), the hand-written ports (`helion_<backend>`), helion.inductor cold (`inductor_helion`) and its pinned output code (`inductor_helion_pretuned`), all in one process under one timer. The helion.inductor implementations and `inductor_pretuned.py` come with https://github.com/pytorch/helion/pull/4596.
  - A baseline the harness lacks (the default `max_autotune`, a library kernel) gets a `bench.py` implementation of its own, so that it runs under the same timer in the same process.
  - `campaign.py` runs the whole selection, and `report.py` summarizes it.
- **Iterate cheaply.** For a change to code generation only, compare before and after with each kernel's config pinned: `inductor_pretuned.py --configs-from` reuses the configs of an earlier cold run for the kernels whose source did not change. Re-autotune only the kernels the change rewrote. Develop with quick autotune; run full effort once, at the end.
- **Same timer, same process, same GPU.** Time the whole compiled graph (CUDA events, L2 flushed between calls) for every implementation, and give every compiled one the same CUDA-graph setting. Absolute times differ between runs (for example, up to 1.7x on the H100s of https://github.com/pytorch/helion/pull/4596), so compare within a process. For close calls, use interleaved ABAB runs.
- **Kernel time and graph time.** Record both. A large gap is host time, which points at the wrapper; CuTe launches cost more than Triton's on graphs of a few microseconds.
- **Cold means cold.** A reported number comes from fresh `TORCHINDUCTOR_CACHE_DIR` and `HELION_CACHE_DIR` with `HELION_FORCE_AUTOTUNE=1`.
- **Stock stays stock.** Baselines compile with `cuda_backend` other than "helion". Importing helion.inductor must leave stock Inductor's code unchanged.
- **Numerics are part of every result:** a faster graph that changes the results is a bug. Autotuning is randomized: re-run a target near the bar before you act on it.
- The cute-hillclimb pitfalls apply too: thermal state, one config per process, a frozen tree while a run reads it.

# Hillclimbing task

## Step 1: Targets, baselines and the regression corpus

1. **Targets:** each graph × input shape from the prompt, reproduced standalone (an FX graph module plus inputs; better-benchmark repros, Dynamo graph dumps and the minifier all work). Pick 4-8 realistic shapes if the prompt does not say. Record the backend in the goal file's `"backend"` and keep it for every number.
2. **Baselines:** eager, stock Inductor at its defaults, and its best configuration (`max_autotune` with `coordinate_descent_tuning`, plus the configurations that can win for the pattern). Also measure the best known implementation: a better-benchmark port when one exists, else a hand-written Helion version or composed library kernels (FlashAttention, FlashInfer, CUTLASS, Quack, Liger and the like; run them on your GPU and shapes). Record each one's version, config and source.
3. **Regression corpus:** graphs that run through the same backend code without being targets, at least the better-benchmark selection. Save their output code (the module Inductor writes: the Helion kernels and the wrapper that calls them) and cold times under `inductor_artifacts/<run-name>/before/`.
4. **Measure** helion.inductor cold at full effort on every target, then create the goal file. Archive one left by an earlier run into that run's directory first.

## Step 2: Optimize the worst target

1. Pick the target with the lowest ratio, and refresh its target program.
2. Find the first divergence and its layer (Deciding where an optimization belongs).
3. Hand-edit until the generated program reaches the target.
4. Implement the edit as generally as the Rules require, with its tests, a `sweep_variants.py` run and a numerics check.
5. Commit it, run `scripts/autoreview.py --head` on it, and re-measure.

Keep a ranked idea backlog in your plan doc. Running out of knob and seed ideas is not a reason to stop: escalate to a new schedule, a codegen change or a compiler change. An optimization counts only once a cold full-effort autotune finds it; a config that wins only when hand-picked means the search space or seeding needs fixing.

## Step 3: Completion check, or loop again

1. Regenerate the corpus's output code and diff it against `before/`. Re-measure every graph whose code changed, kernels or wrapper, or the whole corpus after a compiler, backend or autotuner change.
2. Every re-measured graph keeps `before / after >= 0.99`. Time near-misses ABAB against the base commit. Fix a regression, or turn the choice behind it into an autotuner knob.
3. Update the goal file from your newest cold numbers and run the checker. Targets within about twice the noise margin of the bar get an ABAB run, and you record that ratio.

## Step 4: Shape the pull requests

Rewrite the history into reviewable pull requests (the stack-history skill has the recipes):
- One per compiler stage or concern, about 400-800 lines of code each. Prefix helion.inductor commits with `[inductor]`; compiler, backend and autotuner commits use their own areas.
- Benchmark and harness code goes first. Files generated from the stack's own code, such as pinned helion.inductor output code, go after the commits that generate them, with harness code that needs them.
- Planning files go in a final `[noland]` commit. Raw results, reference kernels and the goal file stay uncommitted under `inductor_artifacts/`.
- A Helion bug unrelated to the stack gets its own commit at the bottom of the stack (AGENTS.md, Fix What You Find).
- A submitted stack keeps its pull requests (AGENTS.md). If the user has you squash one anyway, list the pull requests left orphaned for them to close.

Each pull request description leads with examples, one per change, after a short paragraph on why:
- **A helion.inductor change:** a minimal PyTorch program that hits only the change's core logic (not a copy of a benchmark repro), then the generated Helion kernels before and after, in full, at autotune effort none, with a comment on each changed line.
- **An FX pass:** the PyTorch program before and after the pass.
- **A Helion compiler change:** a minimal Helion kernel, then the generated CuTe or Triton code before and after, abridged to the changed lines and a few lines around them.

`examples.md` in this skill shows one of each, abridged from https://github.com/pytorch/helion/pull/4584, https://github.com/pytorch/helion/pull/4585 and https://github.com/pytorch/helion/pull/4579. `<skill>/scripts/pr_examples.py` compiles an examples file in the trees before and after the change. It writes the markdown with a `# <-` on every changed line, for you to fill in. Below the examples go the before/after times of the graphs the change affects, the details, `Generality:`, `Numerics:` and `Testing:`.

Comments and docstrings in the code center on the code it generates.
- Code needed for correctness shows what is generated without it and what then breaks. Verify that by removing the code.
- A design choice shows the alternative's generated code and why this one wins.
- Delete code whose removal changes nothing.

Before submitting, re-check every number and claim in each message against the current code: late review rounds mostly find messages that fell behind their code. Write for the reviewer, not as a session log.

## Step 5: Final report

Write it once `check_goal.py` prints GOAL MET against the best known implementation (against the baseline when the prompt limits the run to it), or the user accepted a paused or blocked outcome. Include:
- A target table: eager, stock Inductor (default and best), helion.inductor before and after, the best known implementation, geomeans, with the artifact behind each row.
- The regression corpus: which graphs changed and how their times moved.
- Labels: torch version and overlays, Helion commit, backend, GPU and power cap, autotune effort, CUDA-graph setting.
- The changes: each one's layer and why it belongs there. Flag FX passes, pending upstream patches, and anything that changes the generated code for every helion.inductor user or reaches beyond `helion/inductor/`.

# Notes

- **Paths:** `<skill>` is `.claude/skills/inductor-hillclimb`, this skill's directory; a bare `scripts/...` is the repository's (`autoreview.py`, `gpu_board.py`, `run_tests.py`).
- **Scratch:** pick a run name (e.g. `ih-llama-mlp-2026-11-19`) and keep every uncommitted output under `inductor_artifacts/<run-name>/` (gitignored). Never delete the raw files behind a reported number.
- **GPUs:** take them through `scripts/gpu_board.py` (AGENTS.md, GPU Usage); one GPU per A/B comparison.
- **Subagents** suit self-contained side tasks: babysitting a long autotune, porting a reference kernel, regenerating and diffing the corpus. Keep idea selection, reported measurements and the goal file in the main session.
