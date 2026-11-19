---
name: inductor-hillclimb
description: Make torch.compile with helion.inductor (cuda_backend="helion") generate the fastest code for a given model or FX graph — first beat the best stock Inductor configuration (max-autotune + coordinate descent), then match or beat the best known implementation (hand-written Helion kernels, or the fastest library code for the pattern) — by changing the out-of-tree backend (decompositions, FX passes, lowerings, fusion choices, HelionScheduling, HelionKernel) or, where Helion itself is the limit, the Helion compiler; then clean up history and report before/after numbers.
hooks:
  Stop:
    - hooks:
        - type: command
          command: python3 "$CLAUDE_PROJECT_DIR/.claude/skills/inductor-hillclimb/scripts/check_goal.py" --gate
---

Your task is to make helion.inductor — `torch.compile` with the option `cuda_backend="helion"`, code in `helion/inductor/` — generate the fastest possible code for a given model or FX graph. The prompt names the model or graph; if it does not, ask.

**Two stages:**
1. Beat the best code stock Inductor can generate for that graph. A good starting point is `max_autotune=True` together with `coordinate_descent_tuning=True`.
2. Match or beat the best known implementation of the same computation. That is either Helion kernels you write by hand for the graph, or the fastest code you can find for the pattern (Finding the best known implementation). The backend must generate Helion code that is that good.

The method: write (or find) the Helion code that reaches the target, then change the backend so it generates that code. `pretuned_kernels/`, `examples/` and `benchmarks/` hold many fast Helion kernels to start from. Where the generated Helion code is already what you would write and Helion still loses, the Helion compiler is the limit. On the CuTe backend, optimize it with the cute-hillclimb skill, using the generated kernel as the target kernel. On the Triton backend, change the Triton backend or the autotuner directly, under the same rules.

**Scope rules:**
- Everything lives out of tree, in `helion/inductor/` and in Helion itself. Use Inductor's existing hooks, and subclass and override the classes they accept. Do not add hooks to PyTorch, and do not monkeypatch PyTorch internals. If no hook can express a change, ask the user (see the goal contract).
- Read `torch/_inductor` freely as a reference, but implement the idea in the way you think is best and most general.
- **This is a general compiler.** Every change runs on every model and FX graph that reaches helion.inductor, so keep changes general: they should help every graph with a similar pattern. Never special-case a model, shape or kernel name. Before you implement a change, answer these questions, and put the answers in its commit message:
  - **Reuse.** Is there a simpler, more robust solution, or one that reuses more of the existing codebase (Inductor, Helion, helion.inductor)? Write clean code. The bar for a new mechanism that replicates an existing analysis is extremely high; clearing it may mean subsuming the existing mechanism rather than adding a second one beside it.
  - **Mechanism.** If the change adds extensive logic, is there a generic mechanism we could offer that enables the same functionality, and that later optimizations can build on?
  - **The general pass.** What is the general compiler pass behind the optimization? If you found it on a matmul, does it also apply to general reductions? Where else would it help: other patterns, other models?
  - **Blast radius.** Which other models and FX graphs does it change, and does it help or hurt them? Measure it on the regression corpus (Step 3). A helion.inductor change reaches both Helion backends, so also check that the other backend still compiles the emitted code. An optimization that fires only on the target is a special case in disguise.
- When a choice helps some graphs and hurts others, expose it to the autotuner instead of hardcoding it: add a knob, or emit a form whose config space covers both choices (for example a persistent row that `reduction_loops` can roll). Heuristics are fine as autotuner seeds.
- **Study a heuristic before you encode it.** An optimization that wins on the target rarely wins everywhere. Before a new optimization or choice goes into `HelionChoices`, a fusion rule or a codegen decision, find out when it pays off.
  - Sweep it, forced on and off, across the features it plausibly depends on: shapes and sizes, dtypes, layouts and strides, reduction sizes, and GPUs if you have more than one kind.
  - Write the decision as a rule over the features that separate the winning region from the losing one. Where the boundary is fuzzy, make it a knob or an autotuner seed instead.
  - Never enable an optimization for every shape and dtype just because it won on the target.
  - Keep the sweep under `inductor_artifacts/`, and summarize the rule's evidence in the commit message.

# Goal contract — do not stop early

Completion is defined operationally and enforced mechanically: the checker below evaluates every criterion of this section, the stage included.

- **Definition of done (stage 1):** all three must hold.
  - For EVERY target (graph × input shape), helion.inductor matches or beats the best stock Inductor configuration within the checker's 1% noise margin.
  - The geomean of the targets' ratios is >= 1.0.
  - Nothing in the regression corpus got slower (Step 3).

  The numbers come from cold compiles at full autotune effort (`helion.autotune_effort="full"`). Iterate with quick or LLM-guided tuning, but report full. Where interleaved ABAB verify runs exist, they are the ground truth. "Geomean >= 1 but some targets behind" is NOT done. Stock Inductor's own kernel on the same GPU is a constructive proof that the gap is closable.
- **Stage 2:** once stage 1 passes, rewrite the goal file for stage 2 and keep going; the same rules apply. A target's stage-2 baseline is the faster of its stage-1 baseline and the best known implementation (Step 3), so stage 2 never gives back the win over stock Inductor. The run is done when stage 2 passes, or stage 1 if the prompt limits the run to it (`"final_stage": 1`, with the prompt's words quoted in `"final_stage_reason"`). Until then the Stop gate keeps blocking, also after stage 1 passes.
- **Goal file:** in Step 1 you create `inductor_artifacts/goal.json` with one entry per target. After every measurement you report, update that target's `ratio` and `artifact`. This skill keeps its goal file and scratch outputs in `inductor_artifacts/`, apart from cute-hillclimb's `artifacts/`, so a cute-hillclimb run you start from Step 2 keeps its own goal and Stop gate. That nested run ends only through cute-hillclimb's own exits: GOAL MET from its checker (its Step 3), `"blocked"` with its falsification evidence, or `"paused"` on the user's explicit instruction. Its history cleanup and report are this skill's Steps 4 and 5.
  - `ratio` = baseline time / helion.inductor time, from one process on one GPU. Above 1 means helion.inductor is faster.
  - `artifact` = the raw result file under `inductor_artifacts/` that backs the ratio. Never record a ratio without an artifact.
  - `stage` = 1 or 2, and `baseline` names the stage's baseline.
  - `corpus` = the regression corpus check of Step 3: the worst `before time / after time` among the graphs you re-measured, and the artifact that lists them.
- **Stop gate:** this skill registers a Stop hook — the same machinery the built-in `/goal` command uses — that runs `scripts/check_goal.py --gate` whenever you try to end your turn. While any target or the corpus fails, or a stage is left, it blocks the stop and says why: go back to Step 2 instead of summarizing. Run `python3 .claude/skills/inductor-hillclimb/scripts/check_goal.py` yourself at any time to see the current verdict.
- **Only two legitimate ways to end with the goal unmet:**
  1. The user explicitly tells you to stop or pause. Set `"status": "paused"` and quote their instruction in `"paused_reason"`.
  2. You have evidence that the goal is unreachable within scope. Set `"status": "blocked"` and put a concrete question, with the evidence, in `"question_for_user"`. There are two kinds of evidence:
     - The hand-written Helion version of the worst target — the code the backend would have to generate — loses to the baseline on the same GPU after a full autotune, and you could not close the gap in the Helion compiler (cute-hillclimb on CuTe; the Triton backend and autotuner on Triton).
     - The change needs a hook that PyTorch does not have.

     If the hand-written version DOES reach the baseline, that is not impossibility evidence. It is the specification of your next backend change; keep going.

  Never silently downgrade the goal, and never present a final report while the checker says a target fails.
- **Context-loss protection:** long hillclimbs get compacted. At the start of every optimization iteration, re-read `inductor_artifacts/goal.json` and the iteration log in your plan doc. After every iteration, append one log line: the worst target, the layer and idea tried, the cold result and the checker verdict. If a summary in your context claims success, trust the checker output, not the summary.

# The pipeline and its hooks

`torch.compile` takes a graph through these stages. The hook is where helion.inductor can change each one without touching PyTorch.

| Stage | What happens | Out-of-tree hook |
|---|---|---|
| Dynamo, pre-grad | FX graph of torch ops; pre-grad passes | none for a backend: `config.pre_grad_custom_pass` belongs to users; rewrite in the post-grad pass instead |
| AOTAutograd | functionalization, decompositions, joint forward/backward graph, partitioning into forward and backward | the decomposition table (`torch._inductor.decomposition`), gated as below; the joint-graph passes (`config.joint_custom_pre_pass`/`joint_custom_post_pass`) belong to users |
| Post-grad | FX passes on the aten graph: pattern rewrites, reinplacing | the CUDA device custom pass (`register_backend_for_device(..., device_custom_pass=)`); helion.inductor's `HelionPostGradPass` (`helion/inductor/fx_passes/post_grad.py`) already holds it and chains the previous pass |
| Lowering | aten ops become Inductor IR (Pointwise, Reduction, templates, fallbacks) | `torch._inductor.lowering.user_lowerings` (consulted before the built-in lowerings; returning None falls back to them); `register_lowering_pattern` in a backend-owned `PatternMatcherPass`; the scheduling's `get_backend_features()`, which decides some lowerings; the `HelionChoices` hooks Inductor asks while lowering: `use_native_matmul`, `reduction_split_factor`, `should_use_cooperative_reduction` |
| Scheduling | fusion of IR nodes into kernels, loop order, tiling, reduction strategy | `InductorChoices` (`can_fuse`, `can_fuse_vertical`, `can_fuse_horizontal`, `score_fusion`), registered with `register_inductor_choices` as `HelionChoices` (`helion/inductor/choices.py`); the scheduling class's own `can_fuse_*`, `group_fn`, `select_tiling` |
| Kernel codegen | each fused node group becomes one kernel | `HelionScheduling` (`helion/inductor/codegen/scheduling.py`): `codegen_node`, `codegen_node_schedule`, `codegen_template`, `codegen_combo_kernel`, `codegen_mix_order_reduction`, …; `HelionKernel` (`helion/inductor/codegen/kernel.py`) emits the Helion source; `HelionOverrides` and `HelionPrinter` (`helion/inductor/codegen/overrides.py`) emit ops and index expressions; `HelionChoices.should_use_persistent_reduction`, asked when a kernel is created |
| Wrapper | Python code that allocates buffers and launches the kernels (CUDA graphs under `mode="reduce-overhead"`) | the wrapper codegen given to `register_backend_for_device` |
| Helion | each emitted kernel is compiled to Triton or CuTe (the `helion.backend` option) and autotuned at its first call | the Helion compiler, backends and autotuner (the cute-hillclimb skill on CuTe) |

`helion/inductor/` follows these stages. Put new code where its stage lives:

| Package | What goes there |
|---|---|
| `fx_passes/` | graph rewrites: one module per pattern family, applied by `HelionPostGradPass` in `post_grad.py` |
| `lowering/` (create it when the first one comes) | decompositions and lowerings |
| `choices.py` | `HelionChoices`, the heuristics Inductor asks while lowering, fusing and generating code |
| `codegen/` | `scheduling.py` (`HelionScheduling`), `kernel.py` (`HelionKernel`), `overrides.py` (`HelionOverrides`, `HelionPrinter`); a wrapper codegen would go here too |
| `support.py` | what Helion kernels support (matmul dtypes, wrappers, numerics modes), asked by both `HelionChoices` and `HelionScheduling`; new support checks go here |
| `config.py` | the `helion.*` options; `_source_hash` puts every module of the package into Inductor's FX graph cache key |

Two rules for every hook, because helion.inductor shares the `cuda` device with stock Inductor in the same process:

- **Scope it to the Helion backend.** Every override must be inert unless the current compile uses `cuda_backend == "helion"`, so stock Inductor compiles, the stage-1 baseline among them, are unaffected.
  - The decomposition table, the built-in `lowerings` dict and Inductor's own pattern passes (`pass_patterns`) are process-global. A backend that owns its device type can change them freely; helion.inductor, which shares `cuda` with stock Inductor, must not.
  - Inductor caches the decomposition table it uses (`select_decomp_table()` returns the `functools.cache`d `fast_random_decomps()`), so register a decomposition override when `helion.inductor` is imported, before any compile in the process, or clear that cache when you register it.
  - Gate a decomposition or a lowering on a CUDA device as well as on `cuda_backend == "helion"`: the same graph can hold CPU ops. Where the gate declines, fall back to what was there before: the upstream decomposition, or the `user_lowerings` entry you replaced, which may be a user's. Chain it, as `HelionPostGradPass` chains `inner`. Prefer `user_lowerings` over `register_lowering`, and register patterns in Helion's own `PatternMatcherPass` inside `HelionPostGradPass`.
  - Leave the `config.*_custom_pass` settings to users. If helion.inductor takes them, a user's own pass replaces it, or it replaces theirs.
- **Keep the cache key honest.** Anything that changes generated code must reach Inductor's FX graph cache key; otherwise a cached graph from before the change gets reused.
  - helion.inductor's config module is in the key, its options and its private `_source_hash` alike, and `_source_hash` hashes every module under `helion/inductor/`. A code change there misses the cache on its own.
  - Something outside `helion/inductor/` that changes the generated code must reach the key another way: an option in `config.py`, or a custom pass's `uuid()`.

# Deciding where an optimization belongs

This is the central decision. Work from a target program toward the generated one, and change the earliest layer where they diverge.

1. **Write the target program:** the Helion kernels you would write by hand for this graph. Start from `pretuned_kernels/`, `examples/` and `benchmarks/`, or port the fastest library implementation of the pattern to Helion. Make it correct against eager and measure it. It must beat the stage's baseline; if it does not, improve it first, because it is the specification.
2. **Compare the generated program to it, from the outside in:**
   - kernel boundaries: which graph ops land in which kernel;
   - per kernel: the loop structure, tiling, persistence, accumulator form, indexing and memory access;
   - the config.

   Inductor's debug output gives the post-grad graph, the IR, the fusion decisions and the code (`TORCH_COMPILE_DEBUG=1`; `TORCH_LOGS=post_grad_graphs,ir_post_fusion,fusion,schedule,output_code`; `run_and_get_code`).
3. **Match the earliest divergence to a layer:**

| Divergence | Layer |
|---|---|
| The target computes it differently: online instead of two-pass softmax, an op that a decomposition broke up, an algebraic rewrite | a post-grad pattern in `HelionPostGradPass`; a gated decomposition or user lowering when the change must happen before or during lowering |
| Same ops, different kernel boundaries, and Inductor's scheduler can express the grouping | fusion policy in `HelionChoices` (`can_fuse`, `score_fusion`), or the split decision it makes while lowering (`reduction_split_factor`) |
| Same ops, different kernel boundaries, and the scheduler cannot express the grouping: mix-order reductions (a row reduction and a column reduction of the same input in one kernel), matmul prologues and epilogues, combo kernels | `HelionScheduling` |
| Same boundaries, different kernel structure | `HelionKernel` codegen, or the `HelionChoices` decision that feeds it (persistent reduction; cooperative reduction, decided while lowering; tiling) |
| Same Helion source as the target, but slower | the Helion compiler or backend: the cute-hillclimb skill for CuTe, the Triton backend otherwise |
| Same source, and only a hand-picked config wins | the autotuner's search space or seeds |

4. **Hand-edit before you implement.** Edit the emitted Helion source until it reaches the target, measure each edit, and only then implement the winning edits as a general change at that layer. Small differences in the source can matter a lot, so check them one at a time.
   - *For example* (an illustration only, not a checklist): the order in which an index expression is computed can decide whether Helion can prove it non-negative, and with it whether loads vectorize. Likewise, how a reduction loop is written can decide whether the reduction runs once or on every iteration.

## Finding the best known implementation

Start where the comparisons are already wired up, then go to the pattern's specialist libraries. "Fastest" depends on the GPU generation and the shapes, so read each candidate's own benchmarks and then run it on your GPU and shapes.

- **Already wired up:**
  - `pretuned_kernels/`: each kernel's `main()` benchmarks it against named baselines.
  - `benchmarks/run.py`: compares Helion with torch.compile, Triton and eager on TritonBench operators.
  - TritonBench itself, which wraps several implementations per operator.
- **By pattern:**

| Pattern | Where the fastest code usually is |
|---|---|
| Attention: prefill, training | FlashAttention 2/3/4 (FA4: Blackwell, CuTe DSL), cuDNN SDPA, ThunderKittens; FlexAttention as the PyTorch baseline |
| Attention: decode, paged KV, MLA | FlashInfer, FlashMLA, vLLM, SGLang, TensorRT-LLM |
| GEMM, grouped GEMM, FP8/FP4 | cuBLAS/cuBLASLt, CUTLASS and its CuTe DSL examples, DeepGEMM, FBGEMM-GPU, Marlin/Machete (vLLM), TransformerEngine |
| Norms, softmax, cross-entropy, memory-bound fusions | Quack, Liger Kernel, TransformerEngine, Apex, FlashInfer (norm, RoPE, activations), Unsloth |
| MoE: routing, permutation, grouped experts | vLLM and SGLang fused MoE, DeepGEMM, TensorRT-LLM, MegaBlocks |
| Quantization | TorchAO, FBGEMM-GPU, TransformerEngine, vLLM/SGLang quantization kernels |
| Linear attention, SSMs, causal conv | flash-linear-attention, mamba-ssm, causal-conv1d |
| Sampling, top-k, sort, scan | FlashInfer sampling, CUB |
| Embeddings, jagged tensors, recommendation | FBGEMM-GPU, TorchRec, generative-recommenders (HSTU) |

A library kernel rarely matches a whole graph. Compose the library kernels into a reference implementation of the same graph, with the same numerics, before you use it as the stage-2 baseline.

# General notes

## Subagents

Use subagents for self-contained side tasks and have them report conclusions, not raw logs. Good side tasks:

- babysitting a long compile or autotune;
- writing or porting a reference kernel;
- searching the web for the fastest implementation of a pattern;
- regenerating the emitted Helion source of the regression corpus and diffing it;
- running a test suite.

Keep the optimization loop itself in the main session: idea selection, measurements you will report, and goal-file updates.

## Scratch artifacts

Pick a run name at the start (e.g. `ih-llama-mlp-2026-11-19`) and write ALL non-committed outputs under `inductor_artifacts/<run-name>/` in the repo root: raw results, emitted-source snapshots, reference kernels, debug dumps and probe logs. The goal file sits at the fixed path `inductor_artifacts/goal.json`. The `inductor_artifacts/` directory is gitignored; never delete or overwrite the raw files behind reported numbers.

## Shared machine

Follow the GPU Usage rules in `AGENTS.md`: take GPUs through `scripts/gpu_board.py` (`pick` for a session, `run -- <command>` around one compile or benchmark), and keep one GPU for one A/B comparison.

## Measurement pitfalls

The cute-hillclimb list applies here too: the same timer for every implementation, controlled thermal state, one config per process, a frozen tree while runs read it, and bimodal small shapes. In addition:

- **Time the whole compiled graph the same way for every implementation.** Use CUDA-event timing of the full callable with L2 flushed between calls, or the per-kernel GPU time from the profiler, and use the same method everywhere.
  - Host launch overhead is real for end-to-end latency (CuTe launches cost more than Triton's). Measure with and without CUDA graphs, and report both when they disagree.
  - `mode="max-autotune"` also turns on CUDA graphs; give every compiled implementation the same CUDA-graph setting.
- **Cold means cold.** For a number you report, compile with a fresh `TORCHINDUCTOR_CACHE_DIR`, a fresh `HELION_CACHE_DIR` and `HELION_FORCE_AUTOTUNE=1`. A warm cache silently reports an old kernel.
- **Stock Inductor must stay stock.** Compile the baselines in a process where helion.inductor's overrides are inert (cuda_backend is not "helion"), or before `import helion.inductor`. Check once that importing it leaves stock Inductor's generated code unchanged.
- **Numerics are part of the result.** Compare every implementation to eager (an fp32 or fp64 reference for low-precision graphs) with stated tolerances. A faster graph that changes the results is a bug, not a win.
- **Autotuning is randomized.** Re-run targets whose ratio sits near the bar before you act on them. Iterate with `helion.autotune_effort="quick"` or LLM-guided search (`HELION_AUTOTUNER=LLMGuidedSearch`), but report only full-effort cold numbers.

## Commit your work as you go

Group work into logical, complete chunks where everything is working, and commit each chunk (see AGENTS.md, Commits and the Stack). Give each backend change tests in `test/test_inductor_*.py` that show the emitted source. Use the verify skill to test and review each chunk before committing it.

# Hillclimbing task

## Step 1: Targets, baselines and the regression corpus

1. **Backend and targets:** the Helion backend the generated kernels use, `helion.backend` ("triton" by default, "cute" with `HELION_BACKEND=cute`), as the prompt says; record it in the goal file's `"backend"` field, and keep it for every number you compare. Then the model or FX graph from the prompt, with realistic input shapes; pick 4-8 if the prompt does not say. A model may give several graphs (graph breaks; forward and backward for training), and each graph × shape is a target.
   - Reproduce each graph standalone: an FX graph module plus inputs. Dynamo's graph dumps, the minifier and better-benchmark-style repros all work.
2. **Baselines:** eager; stock Inductor at its defaults; and stock Inductor's best configuration, at least `max_autotune=True` with `coordinate_descent_tuning=True`. Also try the other configurations that can win for the pattern, such as `max_autotune_gemm_backends` and `epilogue_fusion`. The best of these is the stage-1 baseline.
   - Also measure the best known implementation you can find or write for each target. The faster of it and the stage-1 baseline is the target's stage-2 baseline. Record every implementation's version, config and source.
3. **Regression corpus:** graphs that are not targets but run through the same backend code, used to check that a change did not make other graphs slower. Use the model's other graphs, `test/test_inductor_*.py`, and the better-benchmark repros. Those are FX graphs with inputs from a better-benchmark checkout, which the harness in `benchmarks/cute/better_benchmark/` reads through `BETTER_BENCHMARK_ROOT`; that directory's `kernels/` holds hand-written Helion ports of the selected repros, useful as stage-2 references.
   - Save the emitted Helion source of all of them under `inductor_artifacts/<run-name>/emitted-before/`.
   - Measure every corpus graph cold, as you measure the targets, and save the raw times as `inductor_artifacts/<run-name>/before/corpus.json`. Step 3 compares against them.
4. **Measure** helion.inductor cold at full effort on every target, and save the raw results.
5. **Create the goal file.** If `inductor_artifacts/goal.json` exists from a previous run, archive it into that run's directory first. The Stop gate is active from this point on. The schema is in `scripts/check_goal.py`; a minimal example:

```json
{
  "task": "llama MLP block, bf16",
  "stage": 1,
  "backend": "cute",
  "baseline": "inductor max_autotune + coordinate_descent_tuning",
  "status": "running",
  "criteria": {"geomean_min": 1.0},
  "targets": [
    {"name": "mlp fwd 8x4096",
     "ratio": 0.82, "artifact": "inductor_artifacts/ih-llama-mlp-2026-11-19/before/mlp_fwd_8x4096.json"}
  ],
  "corpus": {"ratio": 1.0, "artifact": "inductor_artifacts/ih-llama-mlp-2026-11-19/before/corpus.json"}
}
```

## Step 2: Optimize the worst target

1. Pick the target with the lowest ratio.
2. Write or refresh its target program, and find the layer where the generated program first diverges (Deciding where an optimization belongs).
3. Hand-edit until the generated program reaches the target.
4. Implement the edit as generally as possible at that layer, with tests.
5. Re-measure.

Maintain a ranked idea backlog in your plan doc. Exhausting knob- and seed-level ideas is not a stopping condition: escalate to structural changes — a new schedule, a new FX rewrite, a compiler change (through cute-hillclimb on CuTe).

An optimization only counts once a cold full-effort autotune finds it. A config that wins only when hand-picked is a hardcoded heuristic in disguise; fix the search space or the seeding until the search finds it on its own.

## Step 3: Completion check, or loop again

Refresh `inductor_artifacts/goal.json` from your newest cold numbers, then run the checker and believe it:

```
python3 .claude/skills/inductor-hillclimb/scripts/check_goal.py
```

Targets sitting within about twice the noise margin of the bar get an interleaved ABAB verify run, and you record THAT ratio.

Then check the regression corpus:

1. Regenerate its emitted Helion source and diff it against `emitted-before/`.
2. Re-measure every graph whose emitted source changed. After a change to the Helion compiler, a backend or the autotuner, which can change a graph's speed without changing its emitted source, re-measure the whole corpus.
3. Each re-measured graph must keep `before time / after time >= 0.99` against `before/corpus.json`: at most the 1% noise margin slower. For a graph near that bar, time the base commit (from a worktree) and the current one interleaved, ABAB, and use that ratio. If one regressed, fix it, or turn the choice behind it into a knob the autotuner can make.
4. Record the worst of those ratios as `corpus.ratio` in the goal file, with the file that lists every re-measured graph as `corpus.artifact`.

When stage 1 passes and nothing regressed, move to stage 2:

1. Copy the goal file to `inductor_artifacts/<run-name>/goal-stage1.json`.
2. Rewrite `inductor_artifacts/goal.json` in place: `"stage": 2`, a `baseline` that names, per target, the faster of the stage-1 baseline and the best known implementation, and every target's ratio against it. Never leave the goal file missing, which turns the gate off.
3. Go back to Step 2.

If stage 2 cannot be met within scope, end it only through the goal contract's exits. Move to Step 4 once both stages pass or have ended that way, and the good leads are explored.

## Step 4: Clean up git history

You should now have a git history of all the optimization work, relative to the base you built on. Rewrite this history into clear logical chunks with good commit messages:

- All benchmark and harness changes go in a single commit at the start of the history.
- Planning files, plots and anything else that should not be submitted upstream go in a single `[noland]` commit at the end of the history. These files must not be referenced by other commits.
- Raw results, reference kernels, emitted-source snapshots and the goal file stay uncommitted under `inductor_artifacts/`.
- Group the remaining work into logical pieces that are easy to review. Avoid numerous small commits, but use your best judgment.
- Prefix helion.inductor commits with `[inductor]`. Compiler, backend and autotuner commits use their own areas (AGENTS.md).

The stack-history skill has the rewriting recipes; use the verify skill to review and test each commit.

## Step 5: Final report

Write the report only once `check_goal.py` prints GOAL MET (which it does only for the final stage), or once the user has explicitly accepted a paused or blocked outcome. Include the checker output for each stage. The report needs:

- **Target table:** one row per target, with eager, stock Inductor (default and best configuration), helion.inductor before and after, and the best known implementation, ending with geomeans. Name the backing artifact for each row.
- **Regression corpus summary:** which graphs changed and how their times moved.
- **Labels:** the torch version and anything overlaid on it, the Helion commit, the Helion backend, the GPU model and power cap, the autotune effort and the CUDA-graph setting.
- **Changes:** for each change, which layer it is in, why it belongs there, its answers to the general-compiler questions (Scope rules), and the sweep evidence for any heuristic. Flag anything that alters the generated code for every helion.inductor user, or that reaches beyond `helion/inductor/`.
