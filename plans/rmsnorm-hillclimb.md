# RMSNorm CuTe hillclimb — run `rmsnorm-2026-09-01`

Goal: helion-cute matches/beats best baseline on every variant (ratio >= 0.99,
geomean >= 1.0), per `.claude/skills/cute-hillclimb/scripts/check_goal.py`.

## Setup

- Kernel: `examples/rms_norm.py::rms_norm_fwd` — training-mode fwd, returns
  (y in input dtype, rstd fp32). bf16 inputs, weight bf16, eps=1e-6.
- Harness: `benchmarks/cute/compare_rmsnorm_backends.py` (adapted from the
  layernorm compare). Memory-bound; GB/s with bytes = 2*M*N*2 + N*2 + M*4.
- Impls: aten (`_fused_rms_norm`), compile, quack (heuristic cfg),
  quack-tuned (full cfg sweep = strongest baseline), helion-triton, helion-cute.
- Variants (8, bf16): 32768x{1024,4096,8192,16384,32768,65536},
  16384x131072, 8192x262144 — same as the layernorm compare.
- GPUs: B200 850W. GPU 0 in use by another user — do not touch.
  Shapes 1-4 measured on GPU 6; shapes 5-8 on GPU 5. Keep per-shape
  comparisons on their home GPU. GPU 7 for smoke/experiments.
- Artifacts: `artifacts/rmsnorm-2026-09-01/`; goal file `artifacts/goal.json`.

## Status: COMPLETE (2026-09-02) — GOAL MET, geomean 1.0320, all 8 pass

Final checker output: artifacts/rmsnorm-2026-09-01/final_checker_output.txt
History: 42bbfb1e (benchmark), c68bfb41 (feature), + this [noland] commit.
262144 ABAB verify: helion 1.7695 ms vs quack-tuned 1.8677 ms (ratio 1.056).
Remaining unexplored leads (for a future pass): smem staging option for
cute_reduction_reloads (quack's reload_from="smem") for mid-size rows;
seeding the evict-hint gmem-reload config family explicitly; N=4096-class
shapes sit at parity 1.00-1.02 with autotune variance dominating.
Note: three background GPU jobs were externally killed on 2026-09-02
(final_gpu5 sweep after shape 6, a 131072 re-run, first suite re-run);
remaining verification was completed in foreground. 16384x131072's goal
entry uses after_cluster_gpu5.jsonl (cold full autotune on the
cluster-enabled code; the later seed tweak only adds a seed).

## Context from prior hillclimbs (softmax, layernorm)

The cute backend already has row-reduction machinery from those runs:
SIMT register-resident row reductions (strided/blocked lane layouts, vec
loads/stores), rolled-reduction chunks (`reduction_loops`), reload knob
(`cute_reduction_reloads`), thread-block clusters (`cute_cluster_n`),
eviction-policy hints (`load_eviction_policies`), occupancy knob
(`cute_min_blocks_per_mp`), serial-smem block reduce. RMSNorm is structurally
layernorm-minus-mean, so expect helion-cute to start reasonably close; gaps
likely at extreme-N shapes.

## Quack rmsnorm fwd anatomy (from code-search agent, 2026-09-01)

Single-pass, non-persistent. grid=[ceil(M/rows_per_CTA), cluster_n], one row
split across cluster_n CTAs along N. cp.async gmem->smem staging of x
(128-bit atoms, 16B aligned), weight loaded gmem->reg between cp.async commit
and wait (latency overlap). Reduction: intra-thread fp32 -> warp butterfly
shuffle -> cross-warp via tiny smem -> cross-CTA via st.async DSM stores +
one mbarrier (each CTA ends with full row sum); cluster arrive/wait deferred
past local reduce. rstd via fastmath rsqrt. Epilogue reload_from
{None|smem|gmem}; heuristic: smem for N>8k. vecsize=gcd(N,64)=8 elems bf16.
Zero predication when N == tiler_n*cluster_n (power-of-2 N). Config space:
num_threads {128,256} x threads_per_row {64,128,256} x cluster_n {1,2,4,8,16}
x reload_from {None,smem,gmem} = 75 cfgs on B200. Heuristic large-N ladder:
N<=16k cl1, 32k cl2, 64k cl4, 128k cl8, else cl16; tiler_n fixed at 16384
elems (32 KiB smem), never rolls a loop over N. Fwd has NO TMA, NO eviction
hints, NO persistent grid (those are bwd-only).

## Helion-cute lowering facts (from code-search agent, 2026-09-01)

rms_norm_fwd = 1 tile block (rows) + 1 rolled rdim -> LoopedReductionStrategy
whenever N>1024 (persistent normalized away above 1024 threads). Rolled lanes
are inherently strided; `cute_lane_layouts` is a NO-OP for the rdim. Reduce
after roll: warp_reduction (NT<=32) else `_cute_grouped_reduce_shared_two_stage`.
x reuse across sweeps = `fuse_two_pass_loads` register cache: auto <=64
elems/thread, "register" forces <=1024, "gmem" disables (reload from L2, use
load_eviction_policies). `cute_cluster_n` applies ONLY to the tile-loop
(PerThreadNDTileStrategy) form — never to rolled rms_norm. Seeds firing here:
CuteReductionTileHeuristic, CuteReductionWideChunkHeuristic (chunk=max(1024,
N/2)), CuteRolledRowLadderHeuristic (tpr ladder 32/128/256/512/1024, chunk=
min(N/2,65536), V=8). Generated code: HELION_PRINT_OUTPUT_CODE=1;
bound.to_triton_code(config).

## Idea backlog (ranked; refresh each iteration)

1. (measure first) Compare winning cute config + generated code vs quack's
   rmsnorm kernel per failing shape.
2. STRUCTURAL: cluster DSM row-split for rolled reductions — quack's large-N
   edge (cl 4/8/16 at 64k/128k/256k). Today cute_cluster_n never applies to
   the rolled form. Would need: split roll range across cluster CTAs +
   cluster reduce at finalize (reduce_helpers has DSM machinery already).
3. STRUCTURAL: smem-staged reload option for rolled reductions
   (cute_reduction_reloads="smem"): stage x chunk in smem during accumulate
   sweep (cp.async), consume sweep reloads from smem — quack's reload_from=
   "smem". Avoids 2nd gmem/L2 read at 64<elems/thread<=? and frees registers.
4. Check autotuner seeding covers large-N reload/eviction combos (gmem reload
   + evict_last on first read may keep row in L2 for consume sweep).

## Review findings to fix (2026-09-02, Opus review of HEAD~2; fixes queued
until final sweeps release the tree — none affect rmsnorm numbers)

1. MUST: non-fp32 acc dtypes silently lose precision through the fp32
   cluster DSM buffer (int64 sum reproduced wrong at cl=4). Raise
   BackendUnsupported when acc dtype != fp32 with cluster_n>1 (rolled
   path; check tile path too).
2. MUST: outer-graph RMW (hl.atomic_add) executes once per cluster CTA
   (reproduced 4x overcount at cl=4 pinned). Gate rolled cluster off when
   device IR contains atomics (check tile path too).
3. Seed heuristic: round cluster_n down to PoT (6/7 fall off the
   EnumFragment surface for LFBO); drop dead chunk<2 check.
4. Dedupe buf/mbar preamble (rolled vs BlockReduction copies).
5. Tests for the raise paths.
(Declined: merging the compare harnesses — repo convention is parallel
per-kernel compare scripts.)

## Verification status (2026-09-02)

- cute-verify on feature commit: lint clean; default suite 3603 passed;
  cute suite 3904 passed / 3 xfailed; Opus review -> 2 correctness bugs.
- Review fixes applied + amended into feature commit c68bfb41:
  fp32-accumulator hard-gate (rolled raise at finalize; same raise added to
  the tile-form BlockReduction cluster branch), atomic-ops gate via new
  DeviceIR.has_atomic_ops() (rolled + tile-form gates), PoT rounding in
  cluster seed, dead chunk<2 check dropped, buf/mbar preamble deduped into
  _cute_cluster_reduce_smem_vars. 2 new tests (int64 acc raises; atomics
  keep knob inert; verified live on B200 first). None of the fixes change
  rmsnorm codegen (fp32 acc, no atomics, PoT shapes) so sweep numbers on
  the pre-fix commit remain valid.
- Final cold sweep on final code: shapes 1-4 GPU6 (4871/6026/6208/5899),
  shapes 5-6 GPU5 (5337/5017); GPU5 sweep externally killed after shape 6 —
  131072 re-running (final2), 262144 uses after_seed2 (already final code)
  + ABAB verify vs quack-tuned running.

## Iteration log

(append one line per iteration: worst variant, idea tried, cold-autotune
result, checker verdict)

- 2026-09-01: Step 1 done. Cold-autotune baseline: FAIL 32768x1024 (0.970),
  16384x131072 (0.869), 8192x262144 (0.826); other 5 pass (1.02-1.13).
  goal.json created.
- 2026-09-01: 32768x1024 — probe batch found no better config (all <= anchor);
  anchor itself drifted 5034->4882 within batch => noise regime. ABAB verify
  (4x interleaved compile vs cute winner, GPU 6): 26.755us vs 26.735us,
  ratio 1.001 -> PASS. Recorded in abab_n1024.jsonl.
- 2026-09-01: 8192x262144 — hand-edit cluster row-split transplant
  (hand_cluster.py): cl=16 nt=128 -> 4544 GB/s (~parity w/ quack-tuned 4584,
  vs 3786 before). 16384x131072: cl=8 nt=128 -> 4882 (BEATS quack 4671).
  => implemented general rolled-cluster support: LoopedReductionStrategy
  gate + rank-sliced roll range + _cute_grouped_reduce_cluster finalize
  (reduction_strategy.py), CuteRolledClusterLadderHeuristic seed, tests in
  test/test_cute_rolled_cluster.py. Compiler-path smoke @262144 cl16: 4395;
  quick-effort autotune found cluster cfg on its own: 4503. Full cold
  autotune for both large shapes running on GPU 5.
- 2026-09-01: Cold autotune results: 16384x131072 -> 4746 GB/s (ratio 1.016
  PASS; winner is a NON-cluster evict-hinted gmem-reload cfg — M=16384 keeps
  ~37MB working set in L2, re-measured 5047 when cool). 8192x262144 -> 4110
  (0.897 FAIL; search converged to cl=2). Pinned re-measure: cl16+chunk4096
  +nt128 = 4616 (>= quack-tuned 4584) vs cl2 ~4360, seed(chunk=slice) 4350.
  => seed chunk policy updated to min(4096, slice) (measured 4616 vs 4350);
  re-running cold autotune @262144. NOTE GPU5 noise ±5% when heat-soaked;
  cool readings higher.
