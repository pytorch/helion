---
name: cute-hillclimb
description: Iteratively optimize a Helion kernel under HELION_BACKEND=cute until it matches or beats the best available baseline — via general backend/autotuner changes (knobs and seeds, not kernel-specific hacks) — then clean up history and report before/after numbers.
hooks:
  Stop:
    - hooks:
        - type: command
          command: python3 "$CLAUDE_PROJECT_DIR/.claude/skills/cute-hillclimb/scripts/check_goal.py" --gate
---

Your task is to improve the performance of a specific Helion kernel (with `HELION_BACKEND=cute`) until it matches the state of the art. Limit your changes to the Helion CuTe backend and autotuner; do not make changes to the Helion frontend language. Make your changes as general as possible, so optimizations are likely to apply to many different kernels — don't over-specialize to one specific kernel. When you hit a meaningful choice (e.g. A is better for one kernel and B is better for another), add an autotuner knob so we can find the answer automatically. Avoid hardcoding shape- or kernel-specific heuristics. The one place heuristics are OK is autotuner seeds: a seed gives the autotuner a head start without hardcoding the choice. If the CuTe backend's design is a limiting factor, feel free to change the design and refactor.

# Goal contract — do not stop early

Completion is defined operationally and enforced mechanically:

- **Definition of done:** for EVERY variant, helion-cute matches or beats the best baseline within a 1% noise margin (the checker's built-in tolerance), AND the geomean speedup is >= 1.0. Both come from cold-full-autotune numbers; where interleaved ABAB verify runs exist, they are the ground truth. "Geomean >= 1 but some variants behind" is NOT done. There is no "close enough", and a persistent gap is never "inherent overhead" — the baseline's own number on the same GPU is a constructive proof the deficit is closable.
- **Goal file:** in Step 1 you create `artifacts/goal.json` with one entry per variant. After every reported measurement, update that variant's `ratio` (helion throughput / best-baseline throughput, equivalently baseline time / helion time — >1 means helion is faster) and `artifact` (path to the raw result file that backs it). Never record a ratio without an artifact.
- **Stop gate:** this skill registers a Stop hook — the same machinery the built-in `/goal` command uses — that runs `scripts/check_goal.py --gate` whenever you try to end your turn. While any variant fails, it blocks the stop and lists the failing variants: go back to Step 2 instead of summarizing. Run `python3 .claude/skills/cute-hillclimb/scripts/check_goal.py` yourself at any time to see the current verdict.
- **Only two legitimate ways to end with the goal unmet:**
  1. The user explicitly tells you to stop or pause → set `"status": "paused"` in the goal file, quoting their instruction in `"paused_reason"`.
  2. You have falsification evidence the goal is unreachable within scope → set `"status": "blocked"` and put a concrete question in `"question_for_user"`, together with the evidence. Falsification evidence means: you hand-edited the generated kernel for the worst failing variant to adopt the baseline's exact structure and it still cannot reach parity on the same GPU. When the best baseline is another backend (no CuTe structure to copy literally), first ablate the baseline's own winning config to isolate which features carry its edge, transplant those mechanisms by hand, and include the ablation in the evidence. If the hand-edit DOES reach parity, that is not impossibility evidence — it is the specification of your next compiler pass; keep going.
  Never silently downgrade the goal, and never present a final report while the checker says a variant fails.
- **Context-loss protection:** long hillclimbs get compacted. At the start of every optimization iteration re-read `artifacts/goal.json` and the iteration log in your plan doc; after every iteration append one log line (worst variant, idea tried, cold-autotune result, checker verdict). If a summary in your context claims success, trust the checker output, not the summary.

# General notes

## Subagents

Use subagents for self-contained side tasks — babysitting a long autotune, sweeping shapes across GPUs, comparing generated code against the baseline, running a test suite — and have them report conclusions, not raw logs. Keep the optimization loop itself (idea selection, measurements you will report, goal-file updates) in the main session.

## Scratch artifacts

Pick a run name at the start (e.g. `softmax-2026-08-30`) and write ALL non-committed outputs — raw benchmark output (JSON), winning autotuner configs, probe logs — under `artifacts/<run-name>/` in the repo root; the goal file sits at the fixed path `artifacts/goal.json`. The `artifacts/` directory is gitignored; never delete or overwrite the raw files backing reported numbers.

## Shared machine

Unless the prompt says otherwise, assume you are on a shared server with multiple GPUs and play nice with other users. Autotuning, benchmarking, and tests are sensitive to other load on the GPU, so run on an idle GPU by explicitly setting `CUDA_VISIBLE_DEVICES`. Before launching GPU jobs, check `nvidia-smi` for other processes; if a GPU is in use by someone else, avoid that GPU for the rest of the session — the other user is likely to reuse it.

Parallel experiments and tests on different GPUs are fine, but performance varies between GPUs due to thermals and power limits, so keep comparisons apples-to-apples: stick to the same GPU for the same benchmark+shape, and never run a baseline on one GPU and compare its absolute number against one collected on a different GPU. Good ways to parallelize: run `HELION_BACKEND=triton` tests on one GPU and `HELION_BACKEND=cute` tests on another, shard different shapes or kernels across GPUs, or use a spare GPU to try out ideas or re-run autotuning to check robustness (autotuning is randomized).

## Measurement pitfalls

Each of these has produced a convincing-but-fake regression on this hardware; guard against all of them:

- **Use the same timer for every backend.** Time all implementations with the same CUDA-event-based `do_bench`. A wall-clock timer charges per-launch host overhead to the kernel (~0.2 ms/launch for CuTe), which fabricates a deficit of roughly 0.5% on a tens-of-ms kernel while every other backend is event-timed.
- **Control thermal state.** Under a power cap the GPU heat-soaks during long runs (roughly 1% slowdown per minute until it plateaus, on a B200 at 750 W), so a multi-hour autotune followed immediately by measurement makes the "after" number look worse than it is. Let the GPU idle back to near its process-startup temperature before any measurement you intend to compare or report, and interleave A/B runs (ABAB) rather than running all of A then all of B, so drift hits both sides equally.
- **One config per process.** `kernel.bind(args)` memoizes by args: binding the same kernel+args twice returns the same object, so a second `set_config` clobbers the first and an in-process A/B silently measures one config twice. Compare configs in separate processes.
- **Background runs re-read the working tree.** Benchmark and autotune subprocesses import the repo at spawn time, so editing compiler sources or running `git checkout`/`stash` while a background run is live corrupts or crashes it. Freeze the tree for the duration; do bisections and side experiments in a separate `git worktree`.
- **Small-shape readings can be bimodal.** Kernels in the tens-of-µs range can flip between two stable do_bench readings (up to ~2× apart) across processes, even when profilers show identical kernels. Anchor every probe batch with a re-run of a known-good config and discard the batch if the anchor moved; never act on a single reading.

## Commit your work as you go

Group work into logical, complete chunks where everything is working, and commit each chunk to save progress. Use the cute-verify skill to test and review each chunk before committing it.

# Hillclimbing task

## Step 1: Establish a target kernel and baseline

Decide what to optimize and get a good baseline. The prompt should say what the target kernel is; if not, ask the user. Pick a small number of variants of that kernel to test and optimize (use 8 if not otherwise specified). Variants can be different input shapes or minor variations of the kernel.

Find or write a Helion implementation of the target kernel. There may already be one in `examples/` or `benchmarks/cute/` — reuse it where possible. Verify the kernel is correct under both `HELION_BACKEND=triton` and `HELION_BACKEND=cute`; fix bugs if necessary.

Next, find a good baseline to compare performance against. We will optimize the Helion version, but to know whether it is good we need a fair baseline. There may already be one in `benchmarks/` or `examples/`. At minimum you should have a PyTorch baseline and a CuTe baseline; more is fine too. A baseline written in CuTe is extremely useful because you can compare Helion's generated CuTe code against it for optimization ideas. A good source of CuTe kernels is Quack (`./quack` or https://github.com/Dao-AILab/quack.git); if you don't find one there, you may need to search the web or write one yourself. Spend a little time optimizing the baseline to make sure we are comparing against something solid.

Next, select the variants or shapes to test on. Pick realistic shapes — you may need to do some research here. Existing benchmarks, or the shapes used in our baselines' own benchmarking setups, are good sources. Don't pick shapes that are too small: they will be overhead-bound (mainly waiting for kernel launches), and you are looking for memory- or compute-bound shapes. Some variety is good, since you want your optimizations to generalize.

You can create a benchmark harness in `benchmarks/cute/`; reuse code where possible. Report performance in TFLOPS or GB/s, depending on whether we expect the kernel to be compute- or memory-bound. Build in a way to pin each implementation's config independently from the start — the interleaved ABAB verify runs in Step 3 alternate implementations with their own autotune winners.

Measure and record the starting performance. At the end you will need to report the original numbers, so measure and save them now: write the raw benchmark output (JSON) and the winning autotuner configs under `artifacts/<run-name>/`. Record every implementation's winning config, baselines included — you will need the baseline's config later for ablations and ABAB verify runs. These artifacts back the "before" column of the final report — do not lose them. Allow a full autotuning run from a cold start — fresh random seed, autotune cache bypassed — which might take an hour or more; never cut autotuning short or reuse a warm cache for any number reported to the user. These runs re-read the working tree, so let them finish before making Step 2 source changes (or explore in a separate `git worktree` meanwhile).

Now create the goal file (see the goal contract above); if `artifacts/goal.json` already exists from a previous run, archive it into that run's directory first. Write `artifacts/goal.json` with one entry per variant, and fill in the starting ratios and artifact paths from the run you just did. The Stop gate is active from this point on. Schema documented in `scripts/check_goal.py`; minimal example:

```json
{
  "task": "softmax",
  "status": "running",
  "criteria": {"geomean_min": 1.0},
  "shapes": [
    {"name": "32768x4096",
     "ratio": 0.87, "artifact": "artifacts/softmax-2026-08-30/before_gpu5.jsonl"}
  ]
}
```

If needed, make a commit to save your progress before moving on to the next step.

## Step 2: Optimize the performance of one shape+variant

Pick the shape/variant with the largest performance gap to the best baseline, and optimize it until we are beating that baseline. You will need to come up with optimization ideas. One good way is to experiment with manual modifications to Helion's output code, especially comparing it against the handwritten CuTe baseline — this lets you iterate on ideas quickly without first implementing a general compiler pass. Then implement the winning ideas as generally as possible, adding autotuner knobs to expose choices where needed. If the autotuner is unable to find the best configuration, you may need to improve the autotuner seeding algorithm.

Maintain a ranked idea backlog in your plan doc. Exhausting the backlog at the knob/seed level is not a stopping condition — escalate to structural changes: diff the generated kernel against the handwritten baseline down to the instruction level, transplant the baseline's structure into the generated code by hand, and turn whatever reaches parity into a general compiler pass or a redesigned lowering.

An optimization only counts once a *cold* full autotune finds it: run a fresh full-effort search (new random seed, autotune cache bypassed) and confirm it lands on the improvement. A config that only wins when hand-picked is a hardcoded heuristic in disguise — fix the search space or the seeding until the autotuner finds it on its own.

If needed, make a commit to save your progress before moving on to the next step.

## Step 3: Completion check, or loop again

Refresh `artifacts/goal.json` from your newest cold-full-autotune numbers, then run the checker and believe it:

```
python3 .claude/skills/cute-hillclimb/scripts/check_goal.py
```

It applies the goal contract mechanically: every variant's ratio must be >= 0.99 (a fixed 1% noise margin, about the run-to-run spread on a power-capped B200) with an artifact on disk behind it, and the geomean must meet the minimum. When a variant sits within about twice the noise margin of the bar, settle it with an interleaved ABAB verify run (alternate helion and baseline on the same GPU) and record THAT ratio — one-sided thermal drift of ~1% is enough to flip a verdict in either direction.

GOAL MET is a floor, not a ceiling: once the checker passes, go above and beyond — spend some time pushing the speedups higher still (remaining backlog ideas, structural tricks the baseline has that you haven't adopted), and move on to Step 4 only after all good performance leads are explored. Otherwise go back to Step 2 and attack the worst failing variant. A gap that persists on a subset of variants (often the largest shapes, which should be the easiest to saturate) means the backend is missing a capability the baseline has; the fix is usually structural, not another knob sweep. Keep looping until the checker passes (the goal contract lists the only legitimate exits).

## Step 4: Clean up git history

You should now have a git history (relative to `origin/main`) of all the optimization work. Rewrite this history into clear logical chunks with good commit messages:

- All benchmarking/baseline changes go in a single commit at the start of the history.
- Planning files, plots, and anything else that should not be submitted upstream go in a single `[noland]` commit at the end of the history; these files must not be referenced by other commits. Raw benchmark artifacts, autotuner configs, and the goal file stay uncommitted under `artifacts/` (save the final checker output there too).
- Group the remaining work into logical pieces that are easy to review. Avoid numerous small commits, but use your best judgment.
- Prefix CuTe backend commits with `[cutedsl]`.

Use cute-verify as needed to review and test each commit.

## Step 5: Final report

Prepare a final report for the user. Write it only once `check_goal.py` prints GOAL MET (or the user has explicitly accepted a paused/blocked outcome), and include the checker output in the report. It should include a table listing the performance of each variant and each baseline, with Helion+CuTe numbers from both before this hillclimbing process and after, ending with a summary row: geomean for speedup/ratio columns, arithmetic mean for absolute-value columns (GB/s, TFLOPS). Every number in the table must be traceable to a raw artifact under `artifacts/<run-name>/` (name the backing file for each row) — the "before" column comes from the Step 1 artifacts. Label the numbers with version numbers and the GPU model and power cap.

The report should also summarize the changes made, and flag anything risky or with possible impact on parts of Helion beyond the CuTe backend.
