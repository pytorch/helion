# cross_entropy cute-hillclimb — run `cross-entropy-2026-09-01`

Goal: helion-cute >= best baseline on every variant (1% noise margin),
geomean >= 1.0, from cold-full-autotune numbers. Goal file:
`artifacts/goal.json`; raw artifacts under `artifacts/cross-entropy-2026-09-01/`.

## Setup

- Kernel: per-row CE fwd, `loss[i] = logsumexp(x[i,:]) - x[i,target[i]]`,
  fp32 loss out, int64 targets. Defined in
  `benchmarks/cute/compare_cross_entropy_backends.py` (`_helion_kernel`).
  Correctness verified under both backends incl. odd N=50257 (predication)
  and the indirect target gather (`/tmp/ce_probe.py`, 2026-09-01).
- Baselines: aten, torch.compile, quack (heuristic), quack-tuned
  (sweep threads_per_row × num_threads × cluster_n × online_softmax ×
  reload_from — strongest), helion-triton.
- 16 variants `MxNxdtype` (DEFAULT_SHAPES in the harness): N 512..262144,
  fp16/bf16/fp32, incl. GPT-2 odd vocab 50257, Llama3 131072, Gemma 262144.
- Metric: GB/s = (M*N*elem + M*8 + M*4) / time (quack's accounting).
- Machine: shared B200 x8 (850 W cap). GPU 0 busy w/ other users — do not use.
  GPUs 1-7 idle at session start.

## Measurement rules (from rmsnorm run, 2026-09)

- Interleaved ABAB is ground truth for near-bar variants; heat-soaked GPU
  reads ~5% low. Cooldown to 55C before measuring.
- One config per process (bind memoization); subprocess isolation in harness.
- Same GPU for A and B of any comparison. Background runs re-read the tree:
  freeze tree during runs, side-experiments in a separate worktree.

## Status log

- 2026-09-01: Step 1 DONE. before_gpu*.jsonl complete (96 records), goal.json
  written. Geomean 0.9323; 14/16 variants below bar. quack ERR on 50257xfp16
  (its cp.async can't do 16b copies), helion-cute already wins there (1.122).

## Iteration log (append one line per optimization iteration)

- it1 (2026-09-01, GPU1, 32768x8192xfp16 worst=0.856): hand edits on winning
  config codegen: online-pair single reduce = NO WIN (4065 vs 4130); full
  unroll+static cache idx = NO WIN (4196=4196); exp2 fastmath=True on the one
  exp callsite = 4812 vs 4130 base, quack=4816 on same GPU → ENTIRE gap is the
  non-FTZ exp2 denormal fixup. Plan: FX-marked provably-FTZ-safe exp
  (sum(exp(x-amax(x))) pattern) emitting fastmath=True by default; quack and
  triton (libdevice) already flush denormals here.
- it1 landed: commit dd1d17bc (mark_ftz_safe_exp_nodes + op override +
  tests). after_ftz cold campaign: 512bf16 1.006, 1024fp16 1.054,
  4096bf16 1.037, 16384bf16 1.017, 32768fp16 1.055, 262144bf16 1.014,
  65536bf16 1.000, 8192fp32 0.999. Still failing: 2048fp32 0.859,
  8192fp16 0.964, 32768fp32 0.81 (autotune landed on bad config 4003 vs
  4714 before — rerun needed).
- it2 (2026-09-01, 32768x2048xfp32 worst): ncu showed helion-cute == triton
  at locked AND boost clocks (42us, 84% DRAM); do_bench gap (58.4 vs 48.2us)
  is dirty-L2 interference from do_bench's cache flush (clean-read flush:
  both 43.0us). Cause: cute loads have no L2 policy (only L1 hints); triton
  evict_first hints L2. Fix: 'streaming' load_eviction_policies choice →
  cop='cs' (ld.global.cs). Hand: 57.3→47.1us; harness pinned: 4606→5578
  (triton 5575 same GPU). Also fixed fuser to match hinted-scalar vs plain
  scalar loads; found+dodged nvvm.load.ext bf16 cache-modifier ICE (scalar
  16-bit sites stay unhinted). Worktree commit 47831350 (ce-streaming).
  KNOWN BUG found on the way: cute_vector_widths V=4 on fp32 rolled reduce
  (vec 'vec' mode) emits vec-INDEXED scalar loads reading 1/V of the row —
  silently wrong when pinned; autotuner accuracy check filters it. TODO fix.
- it3 (2026-09-02): fp32 corruption ROOT-CAUSED: 'vec' mode's feeds_reduction
  gate never matches aten reduction targets → scalar loads under V-lattice;
  even upstream test kernel cute_normalize_by_sum V=4 fp32 was 82% wrong
  (its test asserts strings only). Fix (commit 378d2573): retire 'vec' mode
  (always 'unroll'), extend unroll hoist to fp32 via Uint32 carrier vector
  (+ byte-based LDG.128 caps). fp32 V=4 CE: 4606 scalar → 4766 vec, correct.
- Cold-findability: 2048fp32 cold autotune → 5582 (>= triton 5575) via
  nt512/chunk64/evictions combo; 8192fp16 cold → 4811 WITH 'streaming' in
  the winning config. Streaming knob merged (47831350). Final full cold
  campaign (final_gpu*.jsonl) launched 2026-09-02 with ftz+streaming+fp32vec.

## Findings

- Quack-tuned winners (all shapes measured so far) use online_softmax=True:
  single pass over x maintaining (running max, rescaled sum). 128-256 threads,
  tpr 32..256, cluster 1 (cluster 4 first appears at N=32768 fp16).
  The 2-pass non-online quack variants lose everywhere → single-pass
  structure is the baseline's main edge. helion-cute generates 2 sweeps
  (max, then sum(exp)) with a register fuse-cache; when the row doesn't fit
  in registers the 2nd sweep re-reads gmem (2x DRAM traffic on big N).

## Lowering facts (from code-study agents, 2026-09-01)

- LoopedReductionStrategy: helion/_compiler/reduction_strategy.py:1190.
  reduction_loops=R roll chunk; lane loop strided hard-coded (cute_lane_layouts
  inert for rolled); vec loads via cute_vector_widths (lane_extent%V==0);
  ONE _cute_grouped_reduce_shared_two_stage per reduction in outer_suffix.
- Fuse cache (fuse_two_pass_loads.py): needs STATIC trip count; "auto" only
  when slots*V <= 64 elems/thread; HELION_FUSER_MODE=smem exists (env only);
  under cute_cluster_n>1 rolled the roll range starts at block_idx()[1]*slice
  -> trip unresolvable -> NO fuse cache (consume sweep re-reads L2 slice).
- Cluster rolled gate (_maybe_apply_cute_rolled_cluster :1306): needs no mask
  (N % (cl*R) == 0), rows-per-CTA==1, fp32 acc, sum/max/min only. Each
  reduction emits its OWN DSM exchange -> max+sum = 2 sequential round-trips.
- fuse_cluster_online_pair (cluster_online_pair.py): single-exchange packed
  (max,sum) rescale fold — tile-form only (single-trip top-level Fors + one
  fuse-cache slot); fails closed on rolled form.
- online_to_3pass: cute AST pre-pass converting textual softmax_two_pass
  online form INTO 3-pass. No general online-softmax lowering exists.
- Indirect gather: scalar pointer math, every thread executes redundantly;
  under cluster all CTAs execute gather + epilogue store (identical values).
- Config space: reduction_loops pow2 [8..N] (forced rolled when N>1024),
  num_threads rdim pow2<=1024 (only shrinks), vec {1,2,4,8},
  cluster {1,2,4,8,16}, reloads {auto,register,gmem}, min_blocks_per_mp
  {0,1,2,3,4,6}. max_reduction_threads=1024.
- Seeds firing for CE (canonical row reduction): ReductionTile, WideChunk,
  RolledRowLadder (tpr 32/128/256/512 by N), RolledClusterLadder (N>=64k:
  cl=N/16384 cap 16, 128 tpr, chunk 4096).

## Idea backlog (ranked)

1. DONE-ISH #2: fuse cache under cluster row-split landed (7de56cb7,
   +9% on cluster forms) — but cluster forms still lose to the gmem-reload
   form on CE 262144 (3489 vs ~3650): the TWO sequential DSM exchanges
   (max, then sum stalls on it) dominate at 128-thread CTAs. Remaining
   lead: rolled-form online pair single exchange (quack's
   online_softmax_reduce edge) — would make cluster the winning family at
   huge N. Not needed for the goal (both 262144 variants pass via
   gmem-reload + ABAB).
2. it1 showed the CTA-local online pair (no cluster) is NOT a win at mid-N
   (4065 vs 4130) — reduce cost is not the bottleneck there.
3. Target-logit gather redundancy: minor, unmeasured.
4. Odd-N (50257): helion-cute already 1.075 (quack cannot compile it).

## GOAL MET (2026-09-02)

check_goal.py: PASS all 16, geomean 1.0379
(artifacts/cross-entropy-2026-09-01/final_checker_output.txt). ABAB
verify runs (ground truth where present): 8192fp16 1.018, 32768fp16
1.014, 32768fp32 1.198, 131072bf16 1.032, 131072fp16 0.9945,
262144fp16 1.0059, 262144bf16 0.992 (all >= 0.99 bar). Lessons: the
default 55C cooldown never waits on these B200s (idle ~40C) — pass
--cooldown-temp-c 42 for ABAB; 262144-class kernels are bimodal via
sweep-2 L2-hit luck (readings 0.88ms vs 1.19ms for the same config).
