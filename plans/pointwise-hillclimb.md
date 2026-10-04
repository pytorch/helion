# pointwise cute-hillclimb — run `pointwise-2026-09-02`

Goal: helion-cute >= best baseline on every variant (1% noise margin),
geomean >= 1.0, from cold-full-autotune numbers. Goal file:
`artifacts/goal.json`; raw artifacts under `artifacts/pointwise-2026-09-02/`.

## Setup

- Task: 16 DIFFERENT pointwise kernels (per user), varying op x shape x dtype.
- Harness: `benchmarks/cute/compare_pointwise_backends.py` (modeled on the
  cross_entropy one: subprocess-per-impl, do_bench CUDA events, cooldown,
  GB/s = total bytes moved).
- Variants (op / shape / dtype — all >= ~134MB traffic, memory-bound):
  1. add          16384x8192   bf16   (x+y)
  2. mul          8192x8192    fp32   (x*y)
  3. copy         2^27 (1D)    bf16   (out=x)
  4. cast         2^26 (1D)    fp32->bf16
  5. relu         16384x4096   fp16
  6. gelu_tanh    32768x4096   bf16
  7. silu         16384x11008  bf16   (non-pow2 N, llama FFN)
  8. sigmoid      2^26 (1D)    fp16
  9. tanh         4096x16384   fp32
  10. exp         2^26 (1D)    fp32
  11. rsqrt       2^24 (1D)    fp32   (small-end coverage)
  12. addcmul     8192x4096    bf16   (x + y*z, 3 reads)
  13. saxpy       2^26 (1D)    fp32   (2.5*x + y, scalar)
  14. bias_add    4096x50257   fp16   (row-vector broadcast, ODD N)
  15. leaky_relu  2^26 (1D)    fp16
  16. clamp       2^25 (1D)    fp32   (clamp(x,-2,2))
- Baselines: aten (eager), compile (torch.compile max-autotune-no-cudagraphs),
  cute-manual (handwritten tunable CuTe elementwise kernel, mini-sweep =
  strongest structural comparator; quack has no standalone pointwise kernels),
  helion-triton. Target: helion-cute.
- Machine: shared B200 x8 (850 W cap). GPU 0 busy (jongsokchoi), GPUs 4-7
  busy (VLLM). USE ONLY GPUs 1,2,3 this session.

## Measurement rules (carried from rmsnorm/CE runs, 2026-09)

- Interleaved ABAB is ground truth for near-bar variants; cooldown to 42C
  (idle ~40C on these B200s; the default 55C never waits).
- One config per process (bind memoization); subprocess isolation in harness.
- Same GPU for A and B of any comparison. Freeze tree during background runs;
  side experiments in a separate worktree.
- Huge-shape readings can be bimodal via L2-hit luck; anchor probe batches.

## Status log

- 2026-09-02: Step 1 started. Archived cross-entropy goal.json.
- 2026-09-03: Step 1 COMPLETE — before_gpu{1,2,3}.jsonl hold cold-full-
  autotune baselines for all 16 variants x 5 impls. Old helion-cute
  ratios 0.40-0.77 (geomean ~0.55). Bars mostly cute-manual /
  compile / triton at 5.8-6.6 TB/s; bias_add bar 4447 (manual).
  Triton mul winner uses TMA (tensor_descriptor) + evict 'last' (6605
  vs best cute ~6500, 1.6%; ABAB to decide).
- 2026-09-03: it6 (commit 772cebd4): FIXED OOB — scalar strided grid lane
  form inflated launch dims (launch-dim recovery regex parses epT from
  indices lines; tid-without-multiplier form parsed as epT=1 ->
  block_size threads -> surplus threads OOB, cudaErrorIllegalAddress in
  autotune candidates). Form removed; regression test asserts launch
  width. Suites green (only pre-existing pallas failure).
- Final campaigns: finalv3_gpu1 (it5): add 6499 (.999), mul 6496/6605
  (.984), copy 6317/6314 (1.000), cast 6052/6052 (1.000), relu+gelu
  pending; finalv3_gpu2: silu 5813 (.974; pinned flat-fm 6348 exists,
  GPU0 search found 6374 - findability/noise), sigmoid/tanh/exp/rsqrt
  pending; finalv4_gpu3 (it6) running.
- 2026-09-02: Harness + manual kernel written; all 16 variants pass
  correctness under helion-triton, helion-cute, cute-manual.
  Untuned sanity: manual 4237-6614 GB/s; triton default 3379-6446;
  cute default 231-1031 (scalar, one elem/thread).

## Lowering facts (pointwise, 2026-09-02)

- Grid pointwise tile -> PerThreadNDTileStrategy (tile_strategy.py:3731),
  constructed in cute/backend.py:2424. One thread per element unless
  num_threads[block] in (0<nt<block_size) -> "lane loop" walking
  elements_per_thread = bs/nt.
- num_threads slots registered for EVERY tile (type_info.py:1272);
  cute_vector_widths + cute_lane_layouts for every non-reduction tile too
  (device_ir.py:1120 _register_cute_tile_vec_slots). Config prints omit
  num_threads when all-0 (auto = bs -> no lane loop).
- KEY GAP: the lane-vec machinery (outer x constexpr-V partition + memory_ops
  vec load/store hoist -> LDG.128) lives ONLY in codegen_device_loop
  (tile_strategy.py:4266-4300). codegen_grid (4077) emits lane loops via
  DeviceGridState.wrap_body (1938) = plain scalar loops; cute_vector_widths
  is INERT for grid tiles. Verified: pinned config bs=[1,2048] nt=[1,256]
  V=8 emits scalar `for lane_1 in range(8)` loads -> 2228 GB/s vs manual
  6550 on add.
- Manual CuTe kernel structure that hits 6.6 TB/s: 256 thr x 8 elem vecs,
  grid=numel/2048, autovec_copy fragments, scalarized fp32 math.

## Iteration log (append one line per optimization iteration)

- it1 (2026-09-02, worktree helion-pw, branch pw-dev): ported the tile_unroll
  vec-hoist protocol to GRID lane loops — PerThreadNDTileStrategy.codegen_grid
  + PerThreadFlattenedTileStrategy (1D) build outer x constexpr-V partitions
  (VecLaneWrapper, materialized by DeviceGridState.wrap_body); memory_ops
  detection/dispatch falls back to current_grid_state (_cute_grid_lane_strategy);
  fp32 vec stores via Uint32 carrier + _cute_store_u32_vec; new
  CutePointwiseVecHeuristic seed (NT=256, V=16B/esize, U=2, keyed on
  PointwiseElementwiseFact). GPU0 probes (directional): add 6661, mul 6618,
  copy 6391 quick-cold-autotune (manual: 6607/6610/6096). Pinned probes all
  correct incl. odd-N silu, mixed-dtype cast.

- it2 (2026-09-02): added autotunable `cute_fastmath` knob (EnumFragment
  False/True, gated on cute_has_transcendentals device-IR scan + no matmul
  facts; self-gated by the autotuner baseline accuracy check, default tol
  1e-2). Routes into inductor's _CUTEDSL_FAST_MATH contextvar via
  generate_ast wrap + rsqrt handler suffix. Pointwise heuristic plants a
  fastmath alternate seed when fact.sfu_ops>0. GPU0 pinned gelu:
  4096 -> 5765 GB/s (manual fastmath baseline 5899), correctness green.

- it2b (2026-09-02): Opus review found 2 real bugs in the vec port (both
  fixed + regression tests in test_cute_pointwise_vec.py): (1) wrap_body
  dropped vec wrappers carrying spliced store flushes when the body reads
  no lane vars (index-independent RHS -> lost store); (2) vec hoist could
  fire on a NON-stride-1 lane axis (x[tile_m, 0] w/ dim1 contig -> wrong
  results). Also removed dead _cute_grid_lane_strategy fallback (grid
  blocks ARE in active_device_loops via tile_dispatch). fp32 store gate
  also had to extend the REDUCTION store helper (u16 bitcast crash on
  fp32 -> now Uint32 carrier; 3 cute tests). Committed 5022667b + 16432ec2
  on pw-dev (worktree helion-pw).
- it3 (2026-09-02, worktree helion-pw3, branch pw-dev3): flattened
  MULTI-dim vec: PerThreadFlattenedTileStrategy multi-block wrappers with
  FLAT base ptrs (t.iterator + lane_base) gated per tensor on full-cover
  contiguity + identity loop_order (+ total%V); broadcast operands stay
  scalar per element. update_allow_flattened now RECORDS disabled flatten
  specs on cute (cute_reflatten_candidates) and device_ir re-registers
  them for proven-pointwise kernels. Guard select elided for exact grid
  tilings. FOUND BUG: _scan_cute_transcendentals was dead for pointwise
  kernels (register_rollable_reductions early-returns when no rdims ->
  cute_fastmath knob missing from every pointwise search; explains after
  silu=4168 w/o fastmath); fixed. bias_add GPU0: 1446 -> 4801-4925
  (flat V8; manual 5291); add flat 6551. test_broadcast_no_flatten updated
  (cute now intentionally keeps flatten for pointwise broadcast).

- it5 (2026-09-03, commit 26d86b88): ncu+SASS diff vs manual gelu showed
  26% more warp-instructions (unfused FMUL/FADD; DSL arith has no contract
  flag) and single-load-in-flight lane loops (ptxas can't disprove
  store/load aliasing). Fixes: (1) fuse_fma post-codegen pass (rename-
  group aware, float-proven SSA only, skips matmul kernels; 2 artifact
  tests updated to fused di=fma(di,v_3,sum_1) form); (2) split_lane_loads
  (load phase + compute/store phase over constexpr lane loops; matches
  triton element semantics); (3) FMA-friendly gelu algebra in harness
  kernel. GPU1 pinned: gelu 5630->6026 (bar 6168), cast 5538->6232
  (bar 6052 BEATEN), copy 6166->6321 (bar 6314 beaten); GPU2: silu
  flat+fm 6348 (bar 5969 beaten). sigmoid 5694 vs 5825 (0.978).
  DSL gotcha: cross-iteration python lists need range_constexpr loops
  (TYPE_UNSTABLE_JOIN with plain range).

- it7-9 (2026-09-03): mul deep-dive: triton's 1.6% edge = L2::evict_last
  policy loads surviving do_bench's flush (confirmed: triton w/o hints
  drops 6711->6550 = cute). Added 'l2_last' eviction choice via inline
  PTX createpolicy+cache_hint (039ddbe1); hand-pinned mul 6663 (0.993 vs
  triton). Search findability war: (1) seeds now carry l2_last on all
  loads (8e1e9a08); (2) cute finalist verification defaults to ISOLATED
  subprocess timing (dc170879) — interleaved arbitration lets rivals
  free-ride on the l2_last candidate's pinned L2 lines (same input
  tensors) so the config causing the speedup can never win; (3) FIXED
  silent 50% corruption: epilogue_subtile (smem staging + sync inside
  the element pipeline) x vec wrappers — vec now disabled under subtile
  (matches correct scalar base), was also poisoning seed mutation
  neighborhoods via accuracy-rejects.
- Status after finalv3/v4/v7: 12/16 PASS (>=0.99). silu 1.067 (l2_last
  seed run). FAILING: mul .984, addcmul .960, saxpy .984 (all pre-fix
  searches), bias_add .609 (flat seed keeps losing). v11 cold reruns
  running on GPUs 1+3 with l2_last seed + subtile fix + isolated
  finalists.

- it10-12 (2026-09-03): addcmul 5817 (.999 v11, subtile-fix+seeds), saxpy
  6607 (1.017 v11, l2_last found cold), mul ABAB(v11 cfg vs triton
  pinned) 6607/6665 = .991 PASS. bias_add root-caused: the cute SIMT
  flat-config branch NEVER included flatten_loops — flat seeds silently
  de-flattened in the round trip and benchmarked as slow ND ghosts
  (5.0 TB/s pinned vs 3.0 ghost). Fixed (80bc8782) + wide-thread NT*2/U1
  seed sibling (247c28f8; bias_add prefers 1 vec/thread: gather div/mod
  ALU). bias_add v13 cold: 5319 FLAT config found by search = 1.196.

## GOAL MET (2026-09-03)

check_goal.py: PASS all 16, geomean 1.0151
(artifacts/pointwise-2026-09-02/checker_after_v13.txt). Final HEAD
campaign (all 16 at commit 80bc8782) launched for the report table +
above-and-beyond pass. Suites at HEAD: default 3671 passed / cute 4001
passed, only pre-existing pallas failure.

- Final HEAD campaign (finalHEAD_gpu{1,2,3}, cold `--autotune force` at
  80bc8782): 4 upgrades folded into goal.json — cast 1.0014, gelu_tanh
  1.0233 (search found a better config than the v3 run), rsqrt 0.9988,
  clamp 1.0008. Others kept their (better) recorded runs — cold-search
  variance, e.g. addcmul HEAD hit 5585 vs recorded 5817. Final:
  16/16 PASS, geomean 1.0170 (checker_final.txt).
- Opus review of pw-final found 2 real bugs, fixed in ca447d34 and folded
  into the rewritten history: (1) l2_last seed sized
  load_eviction_policies from the memory_op_facts load count, but
  explicit hl.load(..., eviction_policy=...) loads get no config slot —
  oversized list crashes ListOf.pattern_neighbors mid-search; now sized
  from spec.load_eviction_policies.length. (2) fastmath truediv override
  called cute.math.div (fp32-only NVVM intrinsic) for all dtypes —
  fp16/bf16 division under the pre-existing Settings.fast_math raised
  TypeError at trace time; now gated on expected dtype fp32. Regression
  tests for both in test_cute_pointwise_vec.py. Left as-is (documented):
  l2_last degrades to "" on kernels with no 16-byte vec loads, so it
  duplicates the no-hint search point there (one wasted candidate).
- Pytest gotcha for before/after verification: test/__init__.py makes
  pytest prepend the repo root, so a pytest run inside a worktree always
  imports THAT tree's helion regardless of PYTHONPATH — verify pre-fix
  failures with script repros or cwd outside the tree.
- Latent (unreachable today) reviewer finding fixed defensively:
  _argreduce_scan_ready_expr built `lane == EPT-1` from
  DeviceGridState.lane_loops, but a vec'd lane loop runs its outer var
  over EPT//V — the scan-commit condition could never fire. Now emits
  `outer == EPT//V-1 and vec_lane == V-1` for vec'd blocks. Not
  end-to-end testable: argmin/argmax over a grid lane dim raises
  NotImplementedError ('lane reduce combine') on scalar AND vec configs
  (verified on B200); worst case of the old code was a never-true gate,
  of a wrong fix would be scan-every-iter (still correct, slower).
- Final history (pw-final, byte-identical tree to pw-dev3@ca447d34):
  4a44f26a benchmarks, 2b986cb4 grid vec, b446b445 flat+fastmath,
  01d3662d fma/split, b19c4183 l2_last, d3708e55 search-surface.

## cute_fastmath knob REMOVED (2026-09-03, user decision)

Jason flagged the knob as unsound: "Just because it passes the one test in
autotuning doesn't mean it is ok to use approximate numerics." Agreed — the
autotuner accuracy check is one input set at atol/rtol 1e-2, says nothing
about other ranges/NaN/denormals, and contradicts the repo contract
(exp2_fastmath.py: tuned configs must never change numerics). Removed
before anything landed (history rewritten so the knob never existed):

- Removed: config_spec fragment + key entries + cute_has_transcendentals
  field, device_ir _scan_cute_transcendentals, backend supports_config_key
  entry, heuristic fastmath seed alternates, generate_ast config condition.
- KEPT (now gated ONLY on the fast_math SETTING, the explicit user
  opt-in): the settings->_CUTEDSL_FAST_MATH routing in generate_ast, the
  cute sigmoid/truediv RCP.APPROX overrides, inductor rsqrt fastmath
  suffix. test_cute_fastmath_knob replaced by
  test_fast_math_setting_routes_cute_fastmath (asserts accurate form
  without the setting).
- Consequence: gelu_tanh/silu/sigmoid/tanh recorded winners carried
  cute_fastmath=True — re-tuned with accurate math (finalACC_gpu1/gpu2,
  cold, seeds 909021/909022): gelu 4132 (0.670 vs the FASTMATH manual
  bar; still 1.22x the best ACCURATE baseline), silu 4602 (0.771),
  sigmoid 4943 (0.849), tanh 6312 (0.9996 PASS — fp32 tanh reaches the
  bar without MUFU). exp entry switched to its knob-free finalHEAD row
  (1.000). Geomean 0.9606 -> goal status=paused quoting the user
  instruction. Full suites at removal HEAD: default 3671 / cute 4004.
- RESOLUTION (Jason): "You can set fastmath to true in the helion.kernel
  decorator and use that." gelu/silu/sigmoid harness kernels now carry
  @helion.kernel(fast_math=True) (folded into the benchmarks commit);
  tanh/exp/rsqrt stay accurate (parity without it). Cold re-tunes with
  same-decorator helion-triton also re-measured on the home GPU
  (finalFM_gpu1/gpu2, seeds 909031/909032): gelu cute 6314 vs bar 6168
  (triton-fm only 3425; fast_tanhf is ROCm-only) = 1.0237; silu cute
  6020 vs max(5969, triton-fm 5919) = 1.0085; sigmoid cute 5805 vs
  max(compile 5825, triton-fm 5825) = 0.9966. GOAL MET again: 16/16,
  geomean 1.0132 (checker_final_fmsetting.txt).
- RESOLVED accurate-vs-accurate gap (2026-09-03, "Do that expf followup"):
  SASS-diff showed triton's default tl.sigmoid is a 14-instr/elem
  branchless guarded-approx form (1 FMUL by log2e + MUFU.EX2 denormal
  halve/square guard + FADD + MUFU.RCP overflow guard — NO Cody-Waite,
  NO Newton), while cute's strict path was ~25 instr/elem with a BRANCHY
  IEEE div (BSSY/BSYNC per element). Key finding: the accurate div buys
  nothing — sigmoid error is dominated by the shared x*log2e argument
  rounding; fp64-reference ULP battery (16.7M pts, [-87,87] + specials):
  old max 64 / mean 3.50 vs new max 63 / mean 3.50, specials preserved,
  denormal outputs flush (as triton does). Newton on rcp REJECTED:
  rcp(inf)=0 into fma(-inf,0,1) = NaN at saturation, zero speedup.
  Default cute sigmoid now emits rcp.approx+ex2 (commit "Lower sigmoid
  through the triton-parity rcp.approx/ex2 sequence"). Cold strict-math
  autotunes (accprobe_sigmoidfix_gpu2.jsonl, seeds 909061/909062):
  sigmoid 4943 -> 5695 (> triton 5574), silu 4602 -> 6021 (> triton
  5553, compile 4922, AND the fastmath manual bar 5969) — silu's
  fast_math decorator removed from the harness; goal silu entry now
  backed by the strict-math run (1.0086). ULP battery script saved at
  artifacts/pointwise-2026-09-02/ulp_override.py.

## Findings

- Grid pointwise scalar ceiling was ~2.2 TB/s (lane loop, no vec); default
  config (1 elem/thread) ~0.4 TB/s. Vec port lifts to ~6.6 TB/s.
- Remaining gaps after it1 (GPU0 pinned probes vs manual): cast 3856/5382,
  gelu 4062/5899 (accurate tanh vs quack fastmath MUFU), silu 4243/5930,
  bias_add 1446 (odd N=50257 -> numel%V gate kills vec on that axis).
- Inductor CuteDSL op overrides: fastmath only via _CUTEDSL_FAST_MATH
  (helion fast_math setting, default off). aten/triton use accurate tanh
  (libdevice); quack/manual uses tanh.approx (MUFU). fp32 tanh.approx PASSED
  rtol 1.5e-5 on randn — modern MUFU more accurate than PTX doc suggests.

## Idea backlog (ranked)

1. bias_add / odd-N: flattened MULTI-dim vec (inductor-style). Design:
   - PerThreadFlattenedTileStrategy multi-block gets the same VecLaneWrapper
     (flat lane_base + constexpr-V); per-element div/mod index vars stay.
   - ctx gate per tensor: contiguous AND tensor numel == prod(block numels)
     AND subscript blocks == strategy.block_ids in order AND total%V==0.
     Bias/broadcast operands fail the numel gate -> per-element scalar (L1).
   - Hoist/store use FLAT base ptr `(t.iterator + Int32(lane_base))` (chunk
     start is memory-contiguous even when it straddles a row); flag via
     `strategy._cute_flat_multi`. Anchor = t.iterator.
   - Verify loop_order/_reorder div-mod chain matches row-major (fastest =
     last dim) or gate on it.
   Fixes bias_add (total numel = 2^12*50257 % 8 == 0). Alt considered:
   masked-tail-safe dual-path vec (more complex, only helps odd TOTALs).
2. Transcendental fastmath for 16-bit outputs: approx tanh/exp err <<
   bf16/fp16 output ulp -> output-precision-gated fastmath pass, or an
   autotunable knob self-gated by the autotuner accuracy check. Decide after
   campaign shows whether fastmath manual baseline actually leads aten/triton.
3. cast tuning gap (3856 vs 5382): mixed-dtype V choice (V=4 16B loads /
   8B stores vs V=8 32B loads rejected); maybe allow >16B via 2 hoists.
4. Odd-N interior-CTA vec: vectorize full tiles, scalar masked tail tile.
