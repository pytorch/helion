# Helion autotuner study: audit, redesign, and results

*2026-08-30, B200 (8x), triton backend. All artifacts under `/tmp/autotuner_study/`;
tooling in `benchmarks/autotuner_study/`; prototype code on branch
`autotuner-v2-fixes` (worktree `/data/users/jansel/ws7/helion-proto2`).*

## 1. Method

- 14 kernel cases (memory- and compute-bound, no overhead-bound shapes):
  matmul 4096^3 + wide, attention 2k/d64 + 4k/d128, softmax two-pass, rms/layer
  norm, cross-entropy 131k vocab, bmm, split-k, long-sum, fp8 gemm,
  gather-gemv, welford.
- Every run: fresh subprocess, pinned GPU (each case always on the same GPU),
  fixed seed, `HELION_SKIP_CACHE=1`, per-candidate CSV via
  `HELION_AUTOTUNE_LOG`, plus an independent post-search measurement of the
  selected and default configs.
- Cost metric: unique configs evaluated (distinct `config_id` per run).
- Quality metric: the selected config of *every* run of a case re-timed
  head-to-head in one process/one interleaved batch on the case's GPU
  (`measure_winners.py`), normalized to the best config that any run of any
  algorithm found for that case.
- 5 campaigns, 468 autotuning runs total: audit (126 runs: default
  LFBOTreeSearch x4 seeds, PatternSearch x3, DifferentialEvolution x2 per
  case), v2 ablation (238 runs: 5 LFBO variants + PatternSearch-v2, 3/2 seeds),
  v3/v4 validation (84 runs), sequential wall-time (20 runs).

## 2. What the audit of the current autotuner showed

1. **The default (LFBOTreeSearch) trades real quality for fewer evals.**
   Geomean final quality 1.085x of best-known at ~744 unique evals/run; plain
   PatternSearch reaches 1.070 but needs ~1419 evals; DE 1.108 at ~1600.
2. **Seed variance is the dominant quality failure.** attention-2k64 default
   runs landed at 0.279/0.303/0.322/0.362 ms (worst +29%); the worst run also
   *stopped earliest* — premature convergence into a bad structural basin
   (pointer indexing + persistent_interleaved), not budget exhaustion.
3. **The 100-config uniform-random initial population is nearly useless.**
   In attention runs, best random config ~0.37-0.53 ms vs the compiler
   heuristic seed at 0.386 ms; in layernorm/rmsnorm runs *nothing* random beat
   the default config. All the differentiating work happens in local search.
4. **30-94% of evaluations are tail waste** (spent after the run is already
   within 1% of its own final answer), yet hard early stopping is dangerous:
   simulating a patience-3 generation stop on the audit curves would end
   28/91 runs >1% worse (p95 +14%) because late breakouts are real.
5. **Structural valleys stall the search.** ListOf keys (e.g. `indexing` with
   5 memory ops) only had element-at-a-time neighbors; crossing from
   all-`pointer` to all-`tensor_descriptor` required stepping through worse
   mixed configs. Winning attention configs differ from the compiler seed in
   block_sizes, indexing, num_warps, and num_stages simultaneously.
6. **Measurement noise is comparable to tail gains.** Identical configs
   re-measured across runs vary 1-3.5% (median) and up to ~10% (p90, short
   kernels); split-k timings are context-sensitive by up to 60% between
   benchmark harnesses/processes, so late search decisions are noise-driven.
7. **LFBO implementation issues** (found by reading, confirmed by data):
   rejected proposals permanently poisoned the `visited` set; per-copy
   selection quota `int(n*0.10)` truncates to 0 and kills copies when <10
   fresh neighbors remain; the incumbent can be dropped by surrogate
   selection so a copy can drift to worse configs; surrogate labels never
   updated with rebenchmarked timings.

## 3. The v2 feature bundle (all opt-in flags; `HELION_AUTOTUNER_V2=1`)

| Flag | What it does |
|------|--------------|
| `search_fixes` (LFBO) | only benchmarked configs enter `visited`; per-copy selection floor (incumbent + >=1 real neighbor); incumbent always retained; surrogate labels re-synced from rebenchmarked members before each fit |
| `HELION_LISTOF_UNIFORM_NEIGHBORS` | ListOf keys also propose uniform lists (all slots set to one value) so multi-slot valleys are crossed in one move |
| `structured_population` | FROM_RANDOM initial population = default + seeds + perturbations of the *default* that randomize only high-impact keys (block sizes, reduction loops, warps, stages, pid_type, indexing, ...), low-impact keys stay at defaults; ~25% fully-random tail; deduplicated by canonical config |
| `probe_first_generation` + `adaptive_freeze_threshold=0.02` | copy 0's first generation is a deterministic single-coordinate probe (batch-median offset-normalized); low-impact coordinates whose measured solo effect stays <2% are frozen for the rest of the main search |
| `finishing_sweep_rounds=2` | coordinate-descent passes over frozen/low-impact keys around the winner at the end |
| `stagnation_patience` | stop the main loop after N generations without >max(0.1%, 0.5%) improvement |

## 4. Results (head-to-head re-measured quality)

Every run's selected config from all 448 campaign runs was re-timed in one
interleaved batch per case. Aggregate = geomean over cases of mean-over-seeds
quality (1.000 = best config any run found for that case); worst-seed =
geomean of per-case worst seed; evals = mean unique configs per run.

Over the 11 measurement-stable cases (splitk, gathergemv, and crossentropy
excluded: their timings swing 10-40% with measurement context, equally for
all algorithms — see section 5.1):

| algorithm | quality | worst-seed | evals |
|-----------|---------|------------|-------|
| default LFBOTreeSearch (old) | 1.057 | 1.106 | 759 |
| PatternSearch (old) | 1.030 | 1.049 | 1623 |
| DifferentialEvolution (old) | 1.092 | 1.111 | 1598 |
| **v2 bundle, patience 3 (new)** | **1.045** | **1.065** | **512** |
| v2 bundle, no stop | 1.041 | 1.057 | 591 |
| v2 bundle, patience 5 | 1.060 | 1.087 | 536 |
| v2 fixes only | 1.043 | 1.075 | 663 |
| v2 without freeze/probe | 1.043 | 1.055 | 529 |
| v2 without structured population | 1.053 | 1.090 | 489 |
| PatternSearch + v2 | 1.046 | 1.067 | 952 |

Headline: **the v2 bundle beats the old default on every axis — ~1.2% faster
final configs on average, 4pp lower worst-seed regret, and 33% fewer unique
candidates evaluated.** It reaches within ~1.5% of old PatternSearch's
quality at less than a third of its evaluations. (Full-14-case aggregates
show the same ordering: old default 1.093 @ 744 evals vs v2 1.076 @ 498.)
Raising stop patience to 5 did NOT help — the runs the softer stop was meant
to save are stuck in bad basins that more grinding does not escape.

Ablation attribution (LFBO base, per-case numbers in the campaign JSONs):
the search fixes + uniform-list moves carry most of the quality gain
(attention-2k64 old 1.121 -> fixonly 1.077 -> v2 1.007; fp8gemm 1.096 ->
fixonly 1.026); the probe/freeze rescues crossentropy (nofreeze 1.359 vs v2
1.268) and welford (nofreeze-with-probe 1.019 vs fixonly 1.033); structured
population is protective on attention-4k128 (noseed 1.157 vs 1.052) and
matmul-wide (1.216 vs 1.106); the patience-3 stop converts tail waste into a
~14% eval saving at ~0.4% mean quality cost.

## 5. Wall-time (sequential, one run at a time, GPU 1)

5 cases x 2 seeds, strictly one autotuning run at a time, alternating
baseline/v2 to balance thermal drift, full host available for compilation:

| pair geomean (v2 / baseline) | ratio |
|------------------------------|-------|
| wall time | **0.80** |
| unique candidates | 0.87 |
| selected-config latency | 1.01 (within noise) |

v2 autotuned 20% faster by wall-clock on this subset (up to 44% faster on
matmul-4096); one v2 gathergemv run found the rare fast basin (0.0654 ms vs
baseline's 0.082-0.086). The larger parallel campaigns (all 14 cases) show
the bigger 33% eval reduction; this sequential subset skews toward cases
where the baseline is already lean.

### 5.1 Measurement findings that matter beyond this study

- In-search one-shot timings carry a systematic offset vs rebenchmarked
  values (the coordinate probe needed batch-median normalization), and
  identical configs vary 1-3.5% median run-to-run.
- split-k / gather-gemv / cross-entropy timings are context-sensitive by up
  to 60% between processes and batch compositions (L2/cache state), so the
  search objective (isolated do_bench with L2 clears) and deployment-style
  interleaved timing can disagree about which config is best. Future quality
  evaluations should re-measure candidate winners head-to-head in a single
  process, as done here.

## 6. Recommendations

1. **Adopt the v2 bundle** (branch `autotuner-v2-fixes`) as the default
   search behavior after a soak period behind `HELION_AUTOTUNER_V2=1`:
   better quality, variance, evals, and wall time than the current default.
   All flags stay independently controllable, and default behavior + cache
   keys are unchanged when disabled.
2. **Keep the stop rule at patience 3** for the eval/wall-time win; quality-
   priority users can set `stagnation_patience=0` (~0.4% better at ~15% more
   evals) or use PatternSearch+v2 (1.046 at 952 evals) / old PatternSearch
   (1.030 at 1623 evals).
3. **Future work, in expected-value order:**
   - Restart-on-stagnation: instead of stopping a stagnant search, reseed a
     copy from a structurally different basin (e.g. the best member of the
     2nd-best structural family). The remaining quality gap is concentrated
     in occasional stuck runs (attention-2k64: ~1 in 3 seeds lands 15-30%
     off for every algorithm, old and new).
   - Cross-run warm starting: seed the surrogate training set from prior
     runs' per-candidate logs (`seed_training_data` already exists).
   - Objective fidelity: for cache-context-sensitive kernels, interleave the
     incumbent into candidate timing batches so the search optimizes
     deployment-like latency.

## 7. Simplified variant and cute-backend validation (follow-up)

The probe/freeze/sweep machinery was removed after the ablation showed it
marginal, producing branch `autotuner-v2-simple` (~284 net lines vs ~535):

- **Now unconditional** (they only remove search pathologies and never
  measured worse): the LFBO fixes (benchmarked-only visited set, per-copy
  selection floor, incumbent retention, rebenchmark-synced surrogate labels)
  and uniform ListOf neighbors. `pattern_version`/`lfbo_version` bumped.
- **Behind `HELION_AUTOTUNER_V2=1`**: structured_population and
  stagnation_patience=3.

Triton re-validation (84 runs, fresh head-to-head session, 11 stable cases):

| arm | quality | worst-seed | evals |
|-----|---------|------------|-------|
| old default | 1.056 | 1.106 | 759 |
| simplified default (fixes only, no flag) | **1.039** | 1.065 | 621 |
| simplified + flag (seed+stop) | 1.048 | **1.059** | **517** |
| old PatternSearch | 1.029 | 1.044 | 1623 |

The simplified branch's *default* behavior is already the quality leader
among LFBO variants at 18% fewer evals than the old default; the flag adds
the remaining eval savings (-32% total) and the best worst-seed variance.

Cute-backend validation (7 non-flash cases x 3 seeds x 2 arms, 45 runs)
surfaced two pre-existing crashes in the *old* autotuner before any
comparison could run: config normalization writes values outside the
searched enum surface on tcgen05 matmuls (e.g. `tcgen05_tvm_ffi_launch`
pinned to `(True,)` while opted-out configs hold `False`), crashing both
`EnumFragment.encode` and `EnumFragment.pattern_neighbors` — every cute
matmul/bmm autotune aborted. Both are fixed (out-of-domain values encode as
zero one-hot / neighbor onto the full surface). With the fixes, head-to-head
results: cute matmul/bmm search surfaces are nearly fully pinned (4-8 unique
configs; both arms find the same optimum), crossentropy is a clear v2 win
(1.060 vs 1.227 of case best), gathergemv/rmsnorm tie, and welford is a
rare-basin coin flip (1 of 6 runs across both arms found a 2.5x faster
config). Aggregate 1.17 vs 1.16 — v2 generalizes safely (equal or better,
no regressions, same eval counts); the mechanisms are backend-agnostic but
matter less where cute pins most knobs. Separately, cute attention-2k64
autotuning is broken for *all* algorithms in this environment: hung
candidate kernels leave benchmark workers unkillable (D-state) and the run
aborts (3/3 baseline failures) — a robustness bug worth fixing in the
benchmark-worker layer.

## 8. Final state: flag removed (follow-up 2)

The `HELION_AUTOTUNER_V2` flag and both features behind it (structured
initial population, stagnation stop) were removed: with a quality-first
posture the stop rule's ~1% mean quality cost buys nothing we want, and the
flag-off configuration was the measured quality leader. The shipped result
is a **flag-free improved default** — net 74 insertions / 31 deletions vs
the pre-study baseline across 3 files:

- `config_fragment.py`: uniform-list ListOf neighbors; EnumFragment
  encode/pattern_neighbors tolerate normalization-off-surface values
  (fixes cute matmul autotune crashes).
- `surrogate_pattern_search.py`: benchmarked-only visited set, per-copy
  selection floor, incumbent retention, rebenchmark-synced surrogate labels.
- `pattern_search.py`: version bump for cute-flash cache-policy keys.

Measured (head-to-head, 11 stable kernels): quality 1.039 vs old 1.056,
worst-seed 1.065 vs 1.106, evals 621 vs 759. The eval-count-focused knobs
(smaller budgets) remain available through the existing effort profiles.

## 9. Artifacts

- Tooling: `benchmarks/autotuner_study/` (registry, campaign runner,
  analyzers, head-to-head measurement, wall-time driver); findings log in
  `NOTES.md`.
- Campaigns (per-run CSVs + summaries):
  `/tmp/autotuner_study/{audit,v2,v2b,v3,v4,walltime}`; head-to-head
  measurements `winners2.json`; reports `*_report.json`.
- Code history: branches `autotuner-v2` -> `autotuner-v2-fixes` (full
  bundle) -> `autotuner-v2-simple` -> final flag-free default on dev; full
  autotuner test suite passes at every step.
- Additional campaign roots: `/tmp/autotuner_study/{v2s,cute-base,cute-v2s}`,
  head-to-head sessions `winners3.json` (triton) and `cutewinners.json` (cute).

## 10. Round 2: robustness — consistently closer to 1.0x (2026-08-31/09-01)

Round 1 shipped as four PRs; round 2 rebaselined at that HEAD (062bd4a1) and
asked where the remaining gap to 1.0x lives. All quality numbers below are
head-to-head: every run's selected config (plus each run's finalist
shortlist) re-measured per case in one interleaved batch on the case's
pinned GPU, two independent passes to establish a per-case noise floor
(`measure_shortlists.py` + `diagnose.py`). Lower is better; 1.000 = best
config any run found. Baseline campaign: 8 seeds default + 3 seeds
PatternSearch x 14 cases; arm campaigns: 4 seeds per single arm, 8 seeds
per bundle; ~750 autotune runs total in round 2.

### 10.1 Where the round-1 gap lives (baseline diagnosis)

* Exploration, not selection: decomposing each run's gap into selection
  error (final shootout picked the wrong finalist) vs exploration error
  (nothing near case-best ever reached the shortlist) shows the gap is
  dominated by exploration everywhere except the context-sensitive kernels
  (splitk, crossentropy, welford), where the unpaired steady final shootout
  also mis-ranks. Different seeds end in disjoint local optima; e.g.
  matmul-wide's good basin (block [128-256, 256, 32] + persistent_interleaved)
  was found by 3/3 PatternSearch runs and 0/8 LFBO runs.
* Early collective stall: most runs stop improving by generation ~3 while
  searching to 8-17; the worst attention seed (1.371) stalled at
  generation 5 of 20 with the rest of its budget unused.
* Accuracy-gate false rejection (the largest single find): the default
  check compares candidates to the kernel's own default-config output with
  fixed elementwise atol=rtol=1e-2. Near-zero elements of large reductions
  legitimately differ across accumulation orders in proportion to the
  *global* output scale, so on matmul_split_k (K=32768, fp16) ~85% of
  candidates — including every split_k>1 config — were falsely rejected
  (all failures at the same ~30/4096 near-zero elements; fp64 ground truth
  shows the rejected configs are exactly as accurate as the baseline
  itself). The search was structurally confined to split_k=1 in every
  campaign ever run on this kernel.

### 10.2 What was tried (single arms, 4 seeds x 14 cases each)

| arm | change | full-14 mean | stable-12 mean | evals |
|---|---|---|---|---|
| baseline | round-1 default | 1.194 | 1.072 | 593 |
| acc | scale accuracy atol floor by max(1, rms(expected)) | 1.109 | 1.059 | 642 |
| polish | full-neighborhood descent after the main loop stalls | 1.100 | 1.050 | 777 |
| fsel | paired interleaved timing for the finalist shootout | 1.107 | 1.075 | 621 |
| hop | basin-hop restarts reinvesting unused generation budget | 1.098 | 1.052 | 771 |

(4-seed single-arm numbers carry ~±0.01-0.02 of seed luck; the bundles
below are the 8-seed decision data. fsel cannot change the search
trajectory — it only re-times the final shortlist — and it halves measured
selection regret, mean 1.035 -> 1.014.)

### 10.3 Bundles (8 seeds x 14 cases each)

| bundle | full-14 mean / worst-seed | stable-12 mean / worst-seed | evals | wall |
|---|---|---|---|---|
| baseline | 1.194 / 1.309 | 1.072 / 1.138 | 593 | 295s |
| pfa = acc+polish+fsel | **1.077 / 1.192** | **1.046 / 1.111** | 781 | 327s |
| all = pfa+hop | 1.083 / 1.215 | 1.051 / 1.102 | 888 | 379s |
| PatternSearch (ref) | 1.228 / 1.249 | 1.058 / 1.072 | 1538 | 466s |

pfa improves or ties 12 of 14 cases at 8 seeds (splitk 2.58 -> 1.33,
matmul-wide 1.24 -> 1.11, welford 1.14 -> 1.06, crossentropy 1.28 -> 1.19,
matmul-4096 1.055 -> 1.031, layernorm 1.088 -> 1.060); the two regressions
are within the case noise floor (attention-4k128 +0.7% vs 1.1% floor) or a
single-seed miss (softmax 1.084 once, mean 1.013). It beats PatternSearch's
mean at half the evals and 70% of the wall time. hop was evaluated and
rejected: inside the bundle it costs ~12% more evals without improving the
mean.

### 10.4 What ships (flag-free, on top of the round-1 stack)

1. Scale-aware accuracy tolerance: when the user has not pinned an explicit
   `autotune_baseline_atol`, each tensor leaf's absolute-tolerance floor
   becomes `atol * max(1, rms(expected))` — never tighter than before,
   correctly looser for large-magnitude accumulations, still rejects
   corruption at the output's own scale. User-specified tolerances remain
   exact.
2. Polish descent (`polish_rounds=10`): after the surrogate-guided main
   loop stalls, plain pattern-search descent benchmarks the entire
   deterministic radius-1 neighborhood (no surrogate pruning) and moves to
   the best neighbor until a round fails to improve.
3. Paired finalist timing: the final-verification shortlist is timed with
   the event-based interleaved bench instead of unpaired sequential steady
   windows (which drifted 1-3% between candidates and mis-ranked near-tied
   finalists).

lfbo_version bumped to 3. Cute-flash searches are unaffected by polish
(they run their own terminal refinement) and pick up the scaled tolerance
and paired finalist timing.

### 10.5 Remaining gaps / future work

* splitk residual variance: even in the unlocked space, seeds split between
  the split_k=8 family (~0.021ms) and lesser basins (worst 2.05). The
  initial population never seeds structured split_k variants.
* attention-2k64 worst seed is still 1.27: early collective stall in a bad
  basin; hop attacked this but did not pay for itself overall. A smarter
  restart trigger (only when generations remain AND the incumbent is far
  from the surrogate's frontier) may do better.
* crossentropy/gathergemv timing context-sensitivity (up to 60% across
  process/batch composition) still limits both the search's own signal and
  any single-number quality claim.
