# Autotuner study working notes

Campaign artifacts live in /tmp/autotuner_study/ (not committed).
This file accumulates evidence and design decisions as the study progresses.

## Setup

- 14 kernel cases (benchmarks/autotuner_study/kernels.py), B200, triton backend.
- Audit: per case, default LFBOTreeSearch x4 seeds, PatternSearch x3, DE x2
  (126 runs), HELION_SKIP_CACHE=1, fixed seeds, per-candidate CSV logging,
  case pinned to one GPU. Independent post-search measurement of selected and
  default config via interleaved_bench(repeat=50).
- Metrics: quality = independently measured selected-config latency;
  cost = unique configs evaluated (distinct config_id in autotune.csv).

## Mid-campaign observations (default = LFBOTreeSearch, full effort)

1. Seed-to-seed final-quality variance is the dominant quality problem on
   compute-bound kernels: attention-2k64 selected 0.279 / 0.303 / 0.362 ms
   across 3 seeds (worst +29%); gathergemv 0.083-0.101 (+21%). The worst
   attention run also stopped earliest (308 unique evals) - premature
   convergence into a bad structural basin (pointer indexing +
   persistent_interleaved), not budget exhaustion.
2. Memory-bound kernels (layernorm, rmsnorm, bmm, softmax): the default
   config or a compiler seed is already within a few % of the final answer;
   400-1300 unique candidates buy 0-7% over the default config. bmm reached
   within 5% of best-known at candidate #1 in all 4 runs.
3. Tail waste: 30-76% of unique candidates are evaluated after the run is
   already within 1% of its own final result.
4. Initial population composition (attention): default config 2.6 ms,
   compiler heuristic seed 0.386 ms, best random-of-97 ~0.37-0.53 ms. The
   ~100 uniform-random configs contribute almost nothing over the compiler
   seed; the local search phase (0.386 -> 0.279) is where good runs win.
5. Structural valley: `indexing` etc. are ListOf fragments whose
   pattern_neighbors change ONE list slot at a time; moving all 5 memory ops
   from pointer to tensor_descriptor requires stepping through mixed configs.
   Winning attention config differs from the compiler seed in block_sizes
   (2 dims), indexing (3+ slots), num_warps, num_stages simultaneously.

## Code-level issues found by reading (to validate with data)

- LFBO `visited` poisoning: in the non-flash path every *generated* neighbor
  is added to `visited`, even the ~90% never selected/benchmarked
  (surrogate_pattern_search.py:4801-4804). The surrogate can permanently
  blacklist candidates it mis-scored early.
- Copy death by quantization: n_sorted = int(len(candidates) * 0.10) is 0
  when <10 fresh candidates remain, ending the copy even when improvement
  was still happening (surrogate_pattern_search.py:4807-4812).
- `current` can be dropped by _surrogate_select, letting a copy move to a
  worse `current` (min over selected only).
- Stale surrogate labels: train_y keeps the first one-shot perf; rebenchmark
  refinements never propagate back (except inf-repairs).
- PatternSearch block-size pair cross-product adds ~4*C(B,2) candidates per
  generation regardless of observed sensitivity.
- Initial random population has no dedup (non-flash) and randomizes ALL
  knobs including <1% ones, adding aliasing + noise.
- DE re-benchmarks duplicate trial vectors (no config-level memoization).

## Prototype directions (to be driven by full audit data)

- P1 better seeding: replace most of the 100 uniform-random initial configs
  with a small deterministic structural coverage design over high-impact
  knobs (pid_type x indexing-uniform x canonical block shapes x warps),
  anchored at default + compiler seeds; dedup by canonical config; keep loow
  -impact knobs at defaults in seeds.
- P2 tiered search: classify fragments into impact tiers (data:
  impact.py matched pairs); main search mutates only high/medium tier
  coordinates; a final coordinate-descent sweep handles low-tier knobs
  (cardinality 2-4 each) on the incumbent.
- P3 tail/stuck fixes: global early-stop on stagnation; min selection >= 1
  to avoid quantization death; allow re-proposal (no visited-poisoning);
  uniform-change neighbors for ListOf keys (set all slots at once).
- P4 surrogate fixes: re-sync train_y before fit; global selection across
  copies instead of per-copy quotas.

## Plateau-breakout analysis (91 audit runs)

Simulating a hard stagnation stop on per-generation best-so-far curves
(threshold 0.5%/gen): patience=3 would end 28/91 runs >1% worse than the
full run (p95 +14%, max +39%); patience=5 still hurts 16/91 (p95 +7%).
Late breakouts after long plateaus are real (some are 10-40% gains,
mostly attention/matmul-class). A pure generation-count stop rule trades
quality for evaluations; escape mechanisms (multi-slot uniform moves,
diverse seeds/restarts) should come first, with any stop rule kept
conservative. The v2 ablation tests patience=3 explicitly (v2 vs v2-nostop).

## Full audit results (99 runs, 11 cases; matmul-class rerun pending)

Quality = mean independently measured selected-config latency / case best
known; evals = mean unique configs. Summary (default LFBOTreeSearch vs
PatternSearch vs DE):

- default: quality 1.00-1.38 (attention-2k64 1.10, gathergemv 1.38, bmm
  1.08, crossentropy 1.09, fp8gemm 1.10), evals 400-1200.
- PatternSearch: usually the best final quality (attention-2k64 1.001,
  fp8gemm 1.05, welford 1.01) but 2-3x the evals (1200-2100).
- DE: fixed 1600 evals; excellent on attention (1.001-1.01), catastrophic
  on fp8gemm (1.56), gathergemv (1.44), welford (1.19).
- gathergemv: no algorithm reliably finds the best basin (block [32,64],
  1 warp, 7 stages); 7/9 runs land 30-60% off.
- Tail waste 70-94% on memory-bound cases for all algorithms.

Conclusion: the surrogate saves evaluations but gives up real quality vs
plain pattern search; the v2 goal is PatternSearch-level quality at
LFBO-level (or lower) evaluation counts.

## Complete audit aggregate (126 runs, 14 cases)

Geomean over cases of (mean selected latency / case best-known); evals =
mean unique configs per run:

| algorithm      | quality | evals | wall(s) |
|----------------|---------|-------|---------|
| default (LFBO) | 1.075   | 744   | 352     |
| PatternSearch  | 1.054   | 1419  | 426     |
| DE             | 1.113   | 1598  | 640     |

v2 target: quality <= 1.03 at evals <= 600.

## Ablation + validation results (238 + 84 runs; head-to-head re-measured)

Aggregate over the 11 measurement-stable cases (splitk/gathergemv/
crossentropy excluded for context-sensitive timing; full-14 tables in
REPORT.md). quality = geomean over cases of mean head-to-head perf ratio
vs case best; worst = geomean of per-case worst seed.

| variant             | quality | worst | evals |
|---------------------|---------|-------|-------|
| default (old)       | 1.057   | 1.106 | 759   |
| v2 bundle (p3)      | 1.045   | 1.065 | 512   |
| v2 no stop          | 1.041   | 1.057 | 591   |
| v2 fixes only       | 1.043   | 1.075 | 663   |
| v2 no freeze        | 1.043   | 1.055 | 529   |
| v2 no seed          | 1.053   | 1.090 | 489   |
| PatternSearch (old) | 1.030   | 1.049 | 1623  |
| PatternSearch + v2  | 1.046   | 1.067 | 952   |

Attribution: fixes+uniform moves carry most of the quality gain; structured
population is protective on attention-4k128/matmul-wide; probe+freeze
rescues crossentropy/welford; patience-3 stop saves ~15% evals at ~0.4%
mean quality (cost concentrated in occasional stuck attention runs -
patience 5 does not fix those; restart-on-stagnation is the right future
work). Session-to-session head-to-head instability on splitk/gathergemv/
crossentropy (10-40% context-sensitive swings) is itself a finding: the
autotuner's objective (isolated do_bench with L2 clears) and deployment-like
interleaved timing can disagree substantially on those kernels.

## Simplified variant + cute validation (follow-up)

- Branch autotuner-v2-simple (merged to dev): probe/freeze/sweep removed
  (~284 net lines vs ~535). LFBO fixes + uniform ListOf neighbors are now
  unconditional (pattern_version/lfbo_version bumped); structured population
  + stagnation stop stay behind HELION_AUTOTUNER_V2=1.
- Triton re-validation, stable-11, fresh head-to-head session: old default
  1.056/1.106/759; simplified default (no flag) 1.039/1.065/621; +flag
  1.048/1.059/517; old PatternSearch 1.029/1.044/1623.
- Cute: two pre-existing crashes in the OLD autotuner fixed (EnumFragment
  encode + pattern_neighbors raised on normalization-off-surface values,
  aborting every cute matmul/bmm autotune). Post-fix: v2 equal-or-better
  (crossentropy 1.060 vs 1.227; matmul-class surfaces nearly fully pinned,
  4-8 unique configs; welford is a rare-basin coin flip). Cute
  attention-2k64 autotuning fails for all algorithms: hung candidates ->
  unkillable benchmark worker (D-state) -> run aborts; worker-layer
  robustness bug, filed as future work.

## Final state correction

This file is a chronological log; sections above describe intermediate
states and cite pre-fix line numbers that no longer correspond to the final
tree. What shipped (see REPORT.md section 8): a flag-free improved default
- the HELION_AUTOTUNER_V2 flag and both features behind it (structured
population, stagnation stop) were removed after review. The final change
set is the EnumFragment robustness fixes, uniform-list ListOf neighbors
(pattern_version 2), and the four LFBO search-loop fixes (lfbo_version 2).

# Round 2 (2026-08-31): robustness — consistently closer to 1.0x

Round 1 shipped as 4 PRs (dev 062bd4a1 = new baseline). Round-2 goal per
Jason: recompute the baseline at HEAD and make final quality consistently
closer to 1.0x. All round-2 campaign data lives under
/tmp/autotuner_study/round2/; prototypes live on branch autotuner-r2.

## Rebaseline

- 154 runs: 8 seeds (201-208) default + 3 seeds PatternSearch per case,
  14 triton cases, GPUs 1-7 (one case pinned per GPU as before).

## Mechanism findings before the arm campaigns

- Selection vs exploration: joining round-1 winners pools with each run's
  full evaluated-config set shows runs almost never evaluate-and-reject a
  config that head-to-head beats their selection — different seeds end in
  disjoint local optima, so the residual gap is dominated by exploration,
  not final-pick noise. New tools measure_shortlists.py (times every run's
  finalist shortlist head-to-head, two A/B passes for a per-case noise
  floor) and diagnose.py (per-run selected vs shortlist-oracle vs case-best
  decomposition) make this measurable per run.
- Metric noise floor: two repeat=200 interleaved passes still disagree by
  1-2% per config on fast kernels; single-pass regret numbers at repeat=50
  produced a phantom 3.9% selection regret on rmsnorm whose finalists are
  actually tied within noise. All regret claims must clear the A/B floor.
- Timing methodology: the CUDA final shootout (final_rebenchmark_best)
  times the top-8 shortlist with *unpaired sequential* steady windows;
  paired interleaved timing and steady timing genuinely disagree on
  tiering for some kernels (rmsnorm steady shows two tiers 2% apart where
  interleaved says tied).
- Early collective stall: most runs stop improving by generation ~3 while
  searching to 8-17 (confirmatory tail); splitk-32768 collapses at
  generation 2 with all five copies stalling under patience=1.
- Accuracy-gate false rejections (the big one): the default accuracy check
  compares candidates against the kernel's own default-config output with
  fixed atol=rtol=1e-2. On matmul_split_k (K=32768, fp16) ~85% of
  candidates — including every split_k>1 config — fail at ~30 of 4096
  elements, always the same near-zero elements at the same magnitudes:
  legitimate accumulation-order noise proportional to the global output
  scale (~N(0,181) here), not element magnitude. fp64 ground-truth check:
  default config is maxabs 0.24 from truth, split_k=4..32 are 0.4-0.8 —
  the same numerical class; the gate rejects configs as accurate as its
  own baseline. The search was structurally confined to split_k=1 in every
  campaign to date.

## Round-2 prototype arms (env-gated on autotuner-r2)

- HELION_SCALED_ATOL=1 ("acc"): scale the accuracy atol floor by
  max(1, rms(expected)) per tensor, never tightening. Smoke: splitk search
  reaches split_k=8 at 0.0215ms vs 0.0380ms baseline-search best (~43%
  faster), selected config verified against fp64.
- HELION_AUTOTUNE_POLISH=10 ("polish"): full-neighborhood pattern descent
  from the incumbent after the main loop (no surrogate pruning), until a
  round fails to improve. Cheap: most of the neighborhood is already
  visited (rmsnorm smoke: 26 new evals).
- HELION_FINAL_INTERLEAVED=1 ("fsel"): paired event-based interleaved
  timing for the final shortlist shootout instead of unpaired steady
  windows.
- HELION_AUTOTUNE_RESTARTS=3 ("hop"): after all copies stall with
  generation budget left, benchmark 20 radius-2 multi-key kicks of the
  incumbent and descend from the best kick (basin hopping). Splitk smoke:
  ran but all kicks failed the (unfixed) accuracy gate — restarts need the
  acc fix on gate-limited cases.
- Arm campaign: 5 arms (acc, polish, fsel, hop, all) x 4 seeds x 14 cases,
  every arm carries the acc fix so non-splitk cases isolate the search
  changes.

## Round-2 results and ship decision

- Single arms (4 seeds x 14): all four beat baseline on full-14 mean
  (acc 1.109, polish 1.100, fsel 1.107, hop 1.098 vs baseline 1.194);
  fsel halves selection regret (1.035 -> 1.014) as predicted by the
  timing-methodology probe.
- Bundles (8 seeds x 14): pfa (acc+polish+fsel) 1.077/1.046
  (full-14/stable-12 means) at 781 evals beats all (=pfa+hop)
  1.083/1.051 at 888 evals. hop rejected. PatternSearch reference
  1.228/1.058 at 1538 evals.
- pfa vs baseline per case (8 seeds): improves or ties 12/14; splitk
  2.58 -> 1.33, matmul-wide 1.24 -> 1.11, welford 1.14 -> 1.06,
  crossentropy 1.28 -> 1.19; regressions within noise (attention-4k128
  +0.7% vs 1.1% floor) or single-seed (softmax one 1.084).
- Wall time: baseline 295s mean/run, pfa 327s (+11% for +29% evals;
  parallel compile absorbs most of it), PatternSearch 466s.
- Ship shape (branch autotuner-r2-ship): scaled atol applies whenever the
  user didn't pin autotune_baseline_atol (explicit tolerances stay exact),
  finalist shootout interleaved unless isolated opt-in, polish_rounds=10
  constructor arg, hop code removed, lfbo_version 3, +2 accuracy unit
  tests; 343 tests pass.
