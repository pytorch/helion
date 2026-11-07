# Inductor native-matmul -> Helion (CuTe) epilogue corpus: hillclimb plan  [noland]

Run name: `inductor-epilogue-2026-11-06`. Raw artifacts: `artifacts/inductor-epilogue-2026-11-06/`.
Goal file: `artifacts/goal.json`.

## Setup

- Corpus: `~/better-benchmark/mm_repro/canonical/epilogue` (345 repros, 650 shape points).
- Generated Helion code, one directory per shape point:
  - `<repro>/helion/<shape>/`    — original pytorch-nullplay (gh/nullplay/13) codegen ("v0").
  - `<repro>/helion_v1/<shape>/` — codegen after the Inductor fixes on branch
    `helion-epilogue-hillclimb` ("v1").
  - `kernel_<i>.py` (standalone Helion kernel + call layouts), `region_<i>.py` (the post-grad
    FX region the kernel fuses, runnable), `output_code.py`, `meta.json`.
- Tooling: `~/better-benchmark/helion_native_mm/`
  - `gen_helion_corpus.py`  generate the corpus (hooks HelionScheduling.codegen_node_schedule).
  - `analyze_corpus.py`     per-kernel features -> `corpus_epilogue.json`.
  - `chunk_corpus.py`       chunk assignment -> `chunks_epilogue.json`.
  - `bench_region.py`       time one Helion kernel (pinned config / autotune) or its region under
                            `torch.compile(max_autotune_gemm_backends=ATEN)`; both CUDA-graph replays
                            under triton `do_bench` (L2 flushed), median reported.
  - `abab.py`               interleaved Helion-vs-ATen rounds in one process.
  - `run_jobs.py`           GPU job runner with a per-GPU timing lock.

## Baseline definition

ATen side = `torch.compile(region, max_autotune=True, max_autotune_gemm_backends="ATEN",
coordinate_descent_tuning=True)`: cuBLAS(Lt) for the GEMM (bias folded by addmm when possible),
Triton for any epilogue left over. Helion side = the Inductor-generated Helion kernel with its
best (cold, full-effort) autotuned config. FP32 regions run TF32 on both sides.

## Chunks (`chunks_epilogue.json`, 638 kernels)

| chunk | kernels | content |
|---|---|---|
| c01_mm_core | 120 | bf16 pure 2D GEMM, >= 1 wave, K >= 512, 8.6-275 GFLOP |
| c02_mm_huge | 46 | bf16 pure 2D GEMM >= 275 GFLOP |
| c03_mm_small | 26 | bf16 pure 2D GEMM, >= 1 wave, < 8.6 GFLOP |
| c04_mm_splitk | 116 | bf16 pure 2D GEMM, < 1 wave, K >= 1024 (weight grads) |
| c05_mm_mem | 81 | bf16 pure 2D GEMM, K <= 256 or tiny |
| c06_bmm_attn | 99 | bf16 bmm, K <= 64 or min(M,N) <= 64 |
| c07_bmm_other | 10 | bf16 bmm, rest |
| c08_epi_simple | 55 | bf16 GEMM + bias / residual / scalar |
| c09_epi_complex | 71 | bf16 GEMM + transcendental / select / multi-output |
| c10_fp32 | 14 | fp32 (tf32) |

## Findings before the hillclimb (default configs, quickmap)

1. [Inductor] every bf16 kernel loads MMA operands as `x.to(float32).to(bfloat16)` (codegen upcast
   + the native-mm lowering's cast back). Helion's dense tcgen05 GEMM analysis rejects that operand
   chain, so all 624 bf16 kernels fall to `cute_mma_dag` (1-stage, non-persistent default) or SIMT.
   Fixed in Inductor: MMA operands load in storage dtype; HelionValue tracks dtype to elide no-op
   casts.
2. [Inductor] addmm lowers to `1.0 * bias + 1.0 * acc`; Helion's tcgen05 epilogue splice only takes
   `acc + bias`. Fixed in the native-mm addmm lowering (unit alpha/beta skip the mul).
3. [Helion] batched kernels with a non-standard batch stride on B (`[B, K, N]` strides
   `[64, 1, 16384]`, attention heads interleaved) get no tensor-core path at all (SIMT, 280x slower).
4. [Helion] dense tcgen05 default config cliffs: partial 256-wide tiles (M or N not a multiple of
   256, e.g. 6272, 25216, 384, 1000), MN-major A (TN layout, ~0.2x of cuBLAS), fp32 output stores.
5. [Helion] complex epilogues: once the operands are clean, the dense tcgen05 path claims the kernel
   (mma_dag declines dots the tcgen05 analyzer accepts) but its epilogue splice rejects
   GELU-backward-style DAGs -> BackendUnsupported for every tcgen05 config.
6. [Inductor] `helion_autotune_effort` defaults to "none", so the default config IS the shipped
   performance; default-config cliffs matter.
7. [Inductor/test] CuTeDSL `@cute.kernel` keeps `inspect.currentframe()` in a reference cycle that
   pins the first compile's caller frames (graph tensors) -> cudagraph-tree pool check failure.
   HelionKernelWrapper now gc.collect()s after the first compile of a specialization.

## Chunk 1 (c01_mm_core) variants

| id | kernel | shape (MxNxK), layout | note |
|---|---|---|---|
| V1 | mm_7bd5ecef3a22/efb531d1/0 | 2560x2560x4096 TN, fp32 out | weight grad, fp32 accumulation |
| V2 | mm_7bd5ecef3a22/1cb0d111/0 | 10240x2560x4096 TN, fp32 out | same family, larger |
| V3 | mm_48f03df82d42/119ae3d1/0 | 25216x768x3072 NN | M not a multiple of 256 |
| V4 | addmm_aa57c10ff773/9f73a2d1/0 | 1000x4096x2880 NT | odd M, K % 128 != 0 |
| V5 | mm_df39dd2c56c3/78462fcb/0 | 4096x8008x2560 NT | N not a multiple of 16 |
| V6 | mm_361d214de6ab/3de88781/0 | 16384x1024x4096 NN | aligned |
| V7 | addmm_6c3e3c7544c9/5c678663/0 | 4096x4096x4096 NT | aligned square |
| V8 | addmm_c2a7d70fc859/94eaacf3/0 | 16384x384x768 NT | small, N = 1.5 x 256 |

cuBLAS (nvjet) picks: 2-CTA everywhere, 128x192 / 128x256 / 192x128 per-CTA tiles, BK 64, 6-7 stages,
clusters 2x1 or 2x2.

## Idea backlog (ranked)

1. TN (MN-major A) on the dense tcgen05 path.
2. Partial-tile handling: tile-size seeds that divide M/N; masked TMA epilogue cost.
3. fp32 output stores.

## Iteration log

(one line per iteration: worst variant, idea, cold-autotune result, checker verdict)

- it0 (baseline, `before/results.jsonl`): v0 = original inductor codegen, v1 = inductor operand fix.
  v0 full autotune: V1 0.83, V2 0.76, V3 0.79, V4 0.37, V5 0.03 (SIMT), V6 0.79, V7 0.76, V8 0.82.
  v1 full autotune: V1 1.07, V2 1.10, V3 1.00, V4 0.87, V5 0.86, V6/V7/V8 pending. GOAL NOT MET.
- measurement fix: event-timed do_bench of a one-kernel CUDA graph is dominated by launch/PDL
  overhead that depends on the GPU's thermal state (V4: do_bench ratio 1.08 cool / 0.87 hot while
  the CUPTI kernel durations gave 0.944 / 0.957). Primary metric is now `abab_dev.py`: interleaved
  rounds of [L2 flush; graph replay], CUPTI device time summed over every kernel of a call, median.
  Rounds agree within 0.5%. Also fixed: abab graphs did not keep their tensors alive.
- region fix: addmm regions whose bias the Helion kernel does not read (the bias add was fused into
  a different kernel) now compute mm only (94 regions patched; ATen times barely move).
- it1: output-edge admission (M/N ragged, K whole -> persistent + CtaGroup.TWO search; K tile capped
  to the largest power of two dividing K when no edge family applies) + FFI direct entry requires
  whole 256x256 tiles. Correctness sweep (9 shapes, 108 sampled configs): 49 ran, 0 wrong.
  Device-time status (autotune-found configs): V1 1.059, V2 1.101, V3 0.992, V4 0.944, V5 0.961,
  V6 0.974, V7 1.006, V8 0.900; geomean 0.990. GOAL NOT MET.
- it2 (hand-picked, before any autotune): V6's needle is the 2x2 cluster (A multicast) capped at the
  resident clusters with a grouped raster: 0.997 (2x1 winner 0.974; 2x2 with l2_groupings=1 0.91;
  2x2 under the default sm_pairs cap launches 2x the co-resident clusters). The paired-pipeline
  2x2 twin seed existed with l2_groupings=1 only -> added the grouped (8) twin seed.
  it2 cold full autotunes (fresh seeds): V6 finds the 2x2/resident/grouped family -> 0.993
  (3 processes); V5 0.981 (2x1, l2_groupings=16). Status: V1 1.059, V2 1.101, V3 0.992, V4 0.944,
  V5 0.981, V6 0.993, V7 1.006, V8 0.900; geomean 0.995. GOAL NOT MET (V4, V5, V8).
- Quack (CuTe DSL hand kernels, device time) on the failing shapes: V4 best 0.962, V8 best 0.889 —
  the same gaps as Helion. V4 at locked clocks: same per-K cycles as nvjet, +2.4k cycles fixed
  (prologue/epilogue) per tile; V8: cuBLAS runs a 192-wide N tile (N=384 in two tiles, 1.94 waves
  on 66 clustered pairs); power-of-two tiles give 2.6 waves (86%).
- it3 (autotune-off default, separate worktree): Inductor runs Helion with autotune_effort="none";
  outside the FFI envelope the default was the fragment-minimum 128x16x16 tile: 0.08-0.17 of cuBLAS
  (V1 0.15, V2 0.17, V3 0.085, V4 0.096, V5 0.083). New CuteTcgen05DefaultTileHeuristic promotes
  the proved seed minimizing wave-quantized MMA time (+ one-CTA 128x256 on whole tiles, grouped
  raster, runtime-validated persistent schedules only): V1 1.064, V2 1.049, V3 0.981, V4 0.930,
  V5 0.950, V8 0.895 (V6 0.951 / V7 0.973 keep the FFI default). 50-kernel cross-chunk sample:
  44 correct, 6 pre-existing BackendUnsupported (same as before), 0 new failures.
- it4 (hand-edited generated kernels, power-of-two tiles only; same-process interleaved device
  time, `handedit_multi.py`, artifacts `handedit/`): edits applied to the generated V4/V8 code.
  - V4 (1000x4096x2880, 2x1 256x256x64, one tile per pair): generated 0.946. Freeing TMEM before
    the final TMA-store drain 0.957; plus ab=6 with the TMA-store epilogue kept 0.985 (cuBLAS runs
    64x6 stages). Helion's output-edge admission judged ab=6 + (128, 64) C ring with
    `c_stages_fits`, whose budget already holds back 28 KiB for the C ring, so it counted the ring
    twice and fell back to the SIMT epilogue (229 548 B arena of 232 448 B; the autotune-off
    default was ab=6 and silently SIMT: 0.934). No gain: TMA-first prologue reorder, L2 line
    prefetch of the first K tiles, B multicast via cluster (4,1,1), 4 AB stages, register-direct
    stores, 128x32/128x16 subtiles, software-pipelined TMEM loads.
  - V8 (16384x384x768, 2x1 256x128, 192 pair tiles on 74 pairs = 3/3/2): generated 0.898; drain
    after dealloc 0.906; bk=64 x 8 stages + 4 C stages 0.918. Per-tile trace: tile 0 runs near MMA
    peak, tiles 1-2 are 1.6-1.9x slower while the previous tile's epilogue runs; removing only the
    TMA stores (diagnostic, wrong output) brings tile 1 from 2.64 to 1.76 us warm, so the bulk
    stores stall the next tile's loads in the TMA unit. cuBLAS (`nvjet_sm100_tst_192x128_64x6_2x2_2cta`,
    SASS via ncu): one 128x192 tile per CTA, 64 clusters of 4 in two hardware waves (no persistent
    loop, so stores never overlap the CTA's own loads), A multicast across the two N pairs,
    per-warp epilogue (each warp: LDTM.16dp256bit -> STSM -> its own UTMASTG of a 32-row strip,
    no 128-thread barrier). Helion one-tile-per-CTA launch (full grid) is much worse (0.72: ~3.4 us
    fixed setup/teardown per CTA). Next-tile L2 prefetch (per line or bulk) is worse (0.84 / 0.75):
    the dirty-L2 benchmark is DRAM-bound and prefetch adds traffic. 2x3 cluster (A multicast to all
    three N pairs) faults (multicast slice of 128 rows by 3).
  - Compiler changes: output-edge TMA-store admission also accepts the measured arena model
    (`output_edge_tma_store_smem_fits`, C ring at the subtile the edge plan renders); two-CTA
    teardowns drain the bulk stores after the TMEM dealloc (`drain_after_dealloc` for every
    two-CTA kernel, previously row-union only). The compiler now emits the hand-edited V4/V8
    kernels byte for byte. Same process (GPU 3): V4 default 0.932 -> 0.986, old best autotuned
    0.954 -> new full autotune 0.985 (finds 2x1 256x256x64 ab=6, l2_groupings=2: the needle);
    V5 (N edge) it2 config at ab=6 0.977 -> 0.989, default 0.944 -> 0.961, new full autotune
    (ab=6, l2_groupings=32, CLC persistent) 0.990 / 0.999 on GPUs 3 / 2 (it2 best 0.971 / 0.982 in
    the same processes); V8 0.90 -> 0.91-0.92 for every hand-picked config (default, c=4,
    interleaved, bk=64 x 8); its full autotune returns the default config (0.911).
    Chunk 1 final autotuned geomean 1.003: GOAL MET (criterion geomean >= 1.0); V8 stays at 0.91
    (structural: persistent tiles overlap stores with the next tile's loads; cuBLAS's 192-wide
    tile covers N=384 in one cluster with A multicast).
  - Per-warp TMA stores (nvjet's epilogue, hand edit: 32-row TMA box per warp, warp-local syncs,
    per-warp bulk groups): V4 epilogue 1.14 -> 0.88 us, kernel +1%; V8 -1% (4x the TMA store
    commands contend with the next tile's loads). Worth it only for one-tile-per-CTA kernels; not
    implemented.

## Category c01, chunk 2 (19 kernels: NT bf16, short K, N <= 1024; BERT-family inference + timm swin/beit)

Columns: default config with the original inductor codegen + Helion base (`orig`), the inductor fix
+ Helion base (`v1base`), + the chunk-1 Helion stack (`pushed`, #4322-#4324), + the deep direct-entry
default (`deep`), and the full-autotune config (`autotuned`, fresh seeded search per distinct shape).
CUPTI device time vs ATen/cuBLAS, one process per kernel, GPUs 0-4.

| id | M x N x K | model | orig | v1base | pushed | deep | autotuned |
|---|---|---|---|---|---|---|---|
| C2_01 | 8192x768x768 | infer/AllenaiLongformerBase | 0.741 | 0.792 | 0.886 | 0.887 | 0.921 |
| C2_02 | 16384x768x768 | infer/BertForMaskedLM | 0.663 | 0.917 | 0.920 | 0.953 | 0.961 |
| C2_03 | 16384x768x768 | infer/YituTechConvBert | 0.657 | 0.914 | 0.920 | 0.962 | 0.967 |
| C2_04 | 16384x768x768 | infer/BertForMaskedLM | 0.654 | 0.905 | 0.907 | 0.944 | 0.964 |
| C2_05 | 16384x768x768 | infer/LayoutLMForMaskedLM | 0.652 | 0.900 | 0.917 | 0.953 | 0.973 |
| C2_06 | 16384x768x768 | infer/PLBartForCausalLM | 0.664 | 0.911 | 0.916 | 0.959 | 0.971 |
| C2_07 | 32768x768x768 | infer/DistilBertForMaskedLM | 0.575 | 0.908 | 0.906 | 0.934 | 0.952 |
| C2_08 | 25216x768x768 | timm infer/beit_base_patch16_224 | 0.631 | 0.102 | 0.940 | 0.937 | 0.944 |
| C2_09 | 100352x256x512 | timm infer/swin_base_patch4_window7_224 | 0.586 | 0.931 | 0.933 | 0.937 | 0.942 |
| C2_10 | 25088x512x1024 | timm infer/swin_base_patch4_window7_224 | 0.660 | 0.987 | 0.986 | 1.030 | 1.040 |
| C2_11 | 8192x1024x1024 | infer/M2M100ForConditionalGeneration | 0.535 | 0.903 | 0.932 | 0.973 | 0.956 |
| C2_12 | 8192x1024x1024 | infer/MBartForCausalLM | 0.528 | 0.910 | 0.913 | 0.952 | 0.984 |
| C2_13 | 8192x1024x1024 | infer/MegatronBertForCausalLM | 0.534 | 0.915 | 0.926 | 0.960 | 0.982 |
| C2_14 | 8192x1024x1024 | infer/M2M100ForConditionalGeneration | 0.596 | 0.926 | 0.934 | 0.974 | 0.985 |
| C2_15 | 16384x1024x1024 | infer/PegasusForCausalLM | 0.584 | 0.921 | 0.925 | 0.944 | 0.992 |
| C2_16 | 16384x1024x1024 | infer/TrOCRForCausalLM | 0.589 | 0.924 | 0.926 | 0.956 | 0.995 |
| C2_17 | 6272x3072x1024 | timm infer/swin_base_patch4_window7_224 | 0.639 | 0.088 | 0.924 | 0.927 | 0.909 |
| C2_18 | 32768x768x768 | infer/DistilBertForMaskedLM | 0.562 | 0.905 | 0.889 | 0.935 | 0.954 |
| C2_19 | 8192x1024x1024 | infer/BartForCausalLM | 0.534 | 0.926 | 0.918 | 0.964 | 0.982 |
| geomean | | | 0.607 | 0.718 | 0.922 | 0.951 | 0.967 |

- Needle (hand-picked first, then confirmed by the full autotune on every whole-tile shape): the
  same 192 KiB AB ring as six 64-deep stages instead of three 128-deep ones (cuBLAS runs 64x6).
  Same-process: 16384x768x768 0.925 -> 0.957, 8192x1024x1024 0.915 -> 0.958, 16384x1024x1024
  0.940 -> 0.980, 4096^3 0.969 -> 0.983; c=2 beats the admitted (6, 4) tuple by 0.4-1.7%. The
  direct-entry (FFI) seed, which is the autotune-off default for whole 256x256 tiles, now takes
  bk=64/ab=6/c=2 where K allows (`deep` column).
- No gain from the epilogue knobs on these shapes (c=4, acc_stages=1, epilogue warp groups,
  64-wide subtile); the store-warp split is not valid for this family.
- Remaining gaps are wave quantization with power-of-two tiles: 6272x3072 is 4.05 waves of
  256x256 pairs, N=768 is three 256-wide tiles (cuBLAS: 192-wide or 256x128 tiles, 2x2 clusters).

### c01 chunk 2 hill-climb log (goal: every kernel >= 0.99 of cuBLAS, full autotune)

- it1: per-K-step timeline trace (data-ready timestamps per stage) of 16384x768x768: the mainloop is
  MMA-bound at the same rate as cuBLAS; the gap is outside the MMA. Epilogue diagnostics
  (no TMA store / no smem staging / no fence) each recover 0.25-0.6 us; nvjet's fence has no
  MEMBAR.ALL.CTA (ptxas always emits one for fence.proxy.async.shared::cta -- not reachable from
  PTX); 16dp256b TMEM loads + stmatrix, one barrier per subtile, per-warp stores, 4-stage C ring,
  64-wide subtiles: no gain or worse.
- it2 (retracted): EVICT_LAST cache policy on the A/B TMA loads looked like +4-14% (pushed as
  helion #4328, then withdrawn). It is a benchmark artifact: normal-priority flush writes evict
  their own lines before EVICT_LAST ones, so tagged operands survive the 256 MB L2 flush between
  timed calls. With the ATen region reading the same input tensors, ATen speeds up by the same
  8% (C2_01: 12.10 -> 11.21 us) and the Helion/ATen ratio does not move (0.919 vs 0.924).
- Measurement fixes: (1) buffer placement alone moves an 11 us GEMM by up to 12% (identical code
  and data, fresh buffers), so separate buffers per variant (and for ATen) bias same-process
  comparisons; handedit_multi now runs every Helion variant and the ATen region on one buffer
  set per placement and takes the median over placements. (2) Any change that leaves operands
  resident across the flush must be checked with ATen on the same inputs.
- it3: with the fair harness, chunk 2 is ~0.94-0.95 of cuBLAS (C2_01 0.942-0.951, C2_02 0.943);
  the earlier hand-edit epilogue ablations (differences of 1-3%) were within placement noise and
  are being redone.
- it4 (fair harness, 8 placements, median of per-placement ratios): chunk geomean 0.961; C2_10 1.030,
  C2_15 0.995, C2_16 1.001 pass; 16384x768x768 family 0.962-0.972, 8192x1024x1024 family 0.956-0.963,
  32768x768x768 0.935-0.940, 25216x768x768 0.944, 100352x256x512 0.950, 8192x768x768 0.924,
  6272x3072x1024 0.911.
  - Epilogue ablations (C2_02, two GPUs x 10 placements): no async-proxy fence +1.1% (ptxas always emits
    MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S for fence.proxy.async.shared::cta / ::cluster / the
    sync_restrict form; nvjet's SASS has FENCE.VIEW.ASYNC.S alone), no STS +1.2%, no TMEM load +1.2%,
    no TMA store +0.4%, none of the three +4.1% (1.016). Worst shapes: dropping the output TMA store
    alone gives +5.6% (8192x768), +16% (6272x3072), +7% (32768x768), +8% (25216x768); EVICT_FIRST on
    the store +2.3% on 6272x3072. Neutral or worse: 16dp256b TMEM loads + STSM, x16/x8 TMEM loads,
    store issued by warp 2/3, 64-wide subtile, C ring 3/4 (8192x768 +1.4%), one barrier per subtile,
    paced epilogue (nanosleep between subtiles), staggered prefill, register-direct store (0.74).
  - nvjet structure (ncu + SASS): 4-CTA clusters along M with B multicast on the N=768/1024 shapes
    (16384x768x768: 151 MB L2->SM vs Helion 2x1 201 MB), non-power-of-two tiles (192x128, 128x192) on
    8192x768 / 25216x768 / 32768x768, swapped "v" kernels with M-paired 512x128 pair tiles on
    6272x3072 (385 MB vs Helion's best 462 MB), CLC tile scheduling, 5-7 AB stages, ~100 KB of C
    staging (ACQBULK twice per kernel), NANOSLEEP.SYNCS barrier waits.
  - Per-K-step trace (2x2 8192x1024x1024): steady-state period 310 ns per 64-deep k-block (MMA ideal
    ~150 ns), TMA issue->consumer 720 ns, ~2 us from entry to the first MMA (cold-L2 fetch), 66 of 74
    pairs active (4-CTA clusters fit 33 at a time). Helion needs ab=8 where nvjet runs 5 stages:
    swapped 6272x3072 at ab=5 is 0.80 vs ab=8 0.92.
  - Backlog (ranked): (1) M-paired / B-multicast pair tiles for the swapped and N=768 families
    (cluster along M with B multicast, M-paired at bn=128 with a column-major output); (2) dense N,M
    (swap-AB) orientation in the compiler (today grouped-only); (3) full-tile C staging with rare
    store waits; (4) CLC scheduling for 2-CTA tiles; (5) per-stage latency of the AB ring (why
    Helion needs 8 stages).
- Remaining (to re-check with the fair harness): wave quantization -- 32768x768x768 (5.19 rounds of 256x256 pair tiles),
  25216x768x768 (4.01 rounds) and 6272x3072x1024 (4.05 rounds). Swapping the operand roles (pair
  splits N, 128-row M tiles: 588/591 tiles = 7.95/7.99 rounds) moves 6272x3072x1024 0.915 -> 0.932.

## Chunk 1 status (2026-11-06, B200; CUPTI device time vs ATen/cuBLAS interleaved on one GPU per row; GPUs 0-3 capped at 750 W, GPUs 4 and 7 at 850 W (V5 rows on GPU 7); >1 = Helion faster)

| id | shape | v0 autotuned | v1 autotuned (inductor fix) | final autotuned | default before | default after |
|---|---|---|---|---|---|---|
| V1 | 2560x2560x4096 TN fp32-out | 0.786 | 1.059 | 1.059 | 0.150 | 1.064 |
| V2 | 10240x2560x4096 TN fp32-out | 0.727 | 1.101 | 1.101 | 0.167 | 1.049 |
| V3 | 25216x768x3072 NN | 0.781 | 0.993 | 0.992 | 0.085 | 0.981 |
| V4 | 1000x4096x2880 NT | 0.302 | 0.841 | 0.985 | 0.096 | 0.986 |
| V5 | 4096x8008x2560 NT | 0.071 | 0.815 | 0.990 | 0.083 | 0.961 |
| V6 | 16384x1024x4096 NN | 0.737 | 0.974 | 0.993 | 0.958 | 0.951 |
| V7 | 4096x4096x4096 NT | 0.662 | 1.006 | 1.006 | 0.968 | 0.973 |
| V8 | 16384x384x768 NT | 0.726 | 0.893 | 0.911 | 0.142 | 0.911 |
| geomean | | 0.491 | 0.955 | 1.003 | 0.20 | 0.98 |

V4/V5/V8 rows updated after it4 (same-process device time: V4/V5 on GPU 3, V5 also 0.999 on GPU 2; V8 on GPU 2).
Open (chunk 1): V8 (0.911, structural, see it4). Quack shows the same V8 gap.
Non-power-of-two tiles: BlockSizeSpec (_PowerOfTwoBlockIdItem) rejects them for every backend and
BlockSizeFragment only searches powers of two. With that check relaxed in a probe, the CuTe
codegen ran V8 correctly on 256x192 (CtaGroup.TWO) and 128x192 tiles, but the best 192 config was
0.919 vs 0.899 for the best power-of-two config: the tile buys ~2 of V8's ~10 points. V8's K=768
(12 K-steps per tile) makes it the same fixed per-tile overhead problem as V4.

## Chunks 2-3 iteration log (2026-11-07, B200 750 W, fair harness: shared buffers, median of 8 placements)

- C-ring pricing (d13744ac0): the plain pair's (128, 32) C subtile was priced at the with-source
  (128, 64) width, capping the 256x256 pair at ab=5/6; pricing it right lets ab=7 and c=4 configs in.
  Chunk 2 geomean 0.961 -> 0.973.
- Aligned-N for 16-bit pairs (fb3eb40cf): 192-wide physical tiles behind the 256 carrier (bf16/fp16,
  M256 pair, (128, 32) C ring, ab=7), argument outputs, MN-major operands, partial last M tiles
  (TMA store clips), L2-grouping seeds for tall problems. Cold autotunes now pick tile_n=192 / ab=7
  for C1_08, C2_01, C2_07, C2_08, C2_18, C3_01, C3_02.
  C2_01 0.92 -> 0.99, C2_08 0.95 -> 1.00 (two GPUs; ab 7 + L2 grouping 16 -- grouping 2 gives 0.93),
  C2_07/C2_18 0.94 -> 0.98, C3_01/C3_02 0.95 -> 0.98, C1_08 0.91 -> 0.98.
- C3_05 (32768x768x768, B N-major): 192 tiles lose (96-column N-major B halves take SW64). Swapped
  orientation (out^T = B^T A^T, column-major output; nvjet's "v" kernel) needed three prototype
  relaxations (aligned-N for column-major outputs, N tails, narrow column-major C subtile) and reached
  0.96 = the normal orientation, not beyond. SASS diff vs nvjet: same structure (static persistent,
  7 stages, 4 UTCHMMA + 1 commit per k-block, 6 store subtiles); Helion adds MEMBAR.ALL.CTA before
  each store fence and issues two 2D TMA stores per subtile (nvjet one 3D); per-k-block trace shows
  MMA waits in the last 5 of 12 k-blocks of each tile.
- 16384x768 family (C2_02..06): wave-quantization sweep (M 9472..18944): Helion steps a full round at
  each multiple of 74 pairs, cuBLAS scales smoothly. Split-K tail emulation (no reduction cost) 0.963
  vs 0.970: at K=768 each split unit writes a full tile. nvjet's exact structure transplanted by hand:
  4-CTA cluster with B multicast (33 resident clusters) 0.958; non-persistent launch 0.72-0.74.
  None beat the persistent 2x1 pair (0.970-0.973).
- C2_17 (6272x3072x1024): swapped "v" 256x128 ab8 0.93, 2CTA bm=128 pair 0.935 (ab8), one-CTA
  128x256 0.92; nvjet runs "v" 256x128 at 5 stages.
