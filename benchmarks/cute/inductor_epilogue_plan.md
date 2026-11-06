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

Category order (owner, 2026-11-07): finish c01_mm_core, then c08_epi_simple -> c09_epi_complex ->
c06_bmm_attn; the remaining categories after those.

## Process (owner directive, 2026-11-08)

- Hill-climb one chunk at a time, strictly in order: E1 -> E2 -> E3 -> E4 (c08_epi_simple), then
  F1 -> F2 -> F3 -> F4 -> F5 (c09_epi_complex), then G1 -> ... -> G5 (c06_bmm_attn).
- A chunk is done when every kernel in it reaches >= 0.99 of the ATen baseline (fair, interleaved;
  cold full autotunes), or is documented as blocked with falsification evidence (cute-hillclimb
  contract). Do not start, and do not measure, the next chunk before the current one is done.
- Always use /cute-hillclimb for every chunk; open a PR after each finished chunk.
- After each finished chunk, report a table: every kernel's speedup over ATen before and after the
  chunk's hill climb ("before" = the default config at the start of c08 work, run in the same fair
  harness; "after" = best of the new default and the cold-autotuned configs).

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
- C3_13 (8192x1024x1024, head-split store out[y//512, (x+1024y)//64 % 16, y%512, (x+1024y)%64]):
  the tcgen05 path accepted only out[tile_m, tile_n], so the kernel fell back to SIMT (4.46 ms vs
  23.9 us for cuBLAS + permute). Radix stores (512c6a1a4): the store's index arithmetic is
  interpreted on concrete coordinates and admitted when the address separates into mixed-radix
  row and column layouts ((512,16):(64,524288), (64,16):(1,32768)); the TMA store goes through
  one hierarchical [M, N] tensor (d_hier_layout). Cold autotune 19.3 us = 1.26x ATen; the C3_11
  config 18.4 us = 1.31x. Review rounds added: int32 overflow check, CTA tiles must partition the
  runs (seq 384 with 256-row pair tiles was silently wrong), identity epilogue only.
- Final-selection timer. The cold autotunes of the 8192x1024 family picked configs 1-3.5% worse
  than a plain 2x1 256x256x64 ab6 config in the fair harness (C2_11 0.958 vs 0.989, C3_11 0.964
  vs 0.996, C3_12 0.948 vs 0.983). The final selection's 9 finalists all read 23.52-23.55 us:
  per-call-flush event timing (flush+call minus flush) is DRAM-bound by the 256 MB flush's dirty
  lines, and one flushed call between two events reads 20.45 us for both configs (B200 events are
  quantized to 1.953 us), while the trace kernel durations are 16.47 vs 16.05 us. A ``kernel``
  regime (per-call flush, samples = trace kernel durations) as CuTe's final-selection default picks
  the 2x1 configs: C2_11 0.958 -> 0.995 (cold autotune), C3_11 -> the 0.996 config.
  Measured over the 31 chunk 2-3 kernels below 0.99 (cold autotunes before/after, both GPUs):
  geomean 0.970 -> 0.977; C2_08, C2_15, C2_16, C3_16, C3_18, C3_20 reach 0.99. Review rounds made
  each sample the span of the call's kernels (PDL overlap is not counted twice), sized the windows
  with the flush, kept the long-kernel probe, and rewarm the device inside the profiling session
  (the first session takes ~1 s to start): C2_11's two configs then read 16.8-17.8 vs 16.0-16.3 us
  in every round, where they had overlapped.

## Chunk 4 (c01 C4_01..C4_20, 2026-11-07)

- Defaults: 18/20 at 1.03-1.16 of ATen (ATen pays an extra Triton cast: these are bf16 GEMMs into
  fp32 outputs, cuBLAS stores bf16). C4_19 0.987, C4_15 0.89.
- C4_15 (3072x1024x6272, MN-major operands, fp32 out): 256x256 pairs make 48 pairs for 74 SM pairs;
  nvjet runs 256x192 pairs (2x2, 6 stages): 72 pairs with a 64-wide last N tile. The aligned-N
  pair admitted 16-bit outputs and whole N tiles only. With FP32 outputs (C ring priced at the
  stored width, with-source subtile) and partial last N tiles (loads zero-fill, TMA store clips):
  cold autotune picks tile_n=192 ab=6 -> 1.07 (36.0 vs 37.7 us).

## c08_epi_simple pre-survey (CPU codegen under the GB300 target, 55 kernels, 2026-11-07)

- 31 bias epilogues (`out = (in0[tile_x] + acc).to(bf16)`, rowvec bias; some `* 0.125`) bind with the tcgen05
  search but without paired-pipeline SMEM facts: `analyze_pipeline_smem_facts` models broadcast-row staging only
  for FP8, so 16-bit + bias gets 5-7 legacy seeds and no deep-ring / paired search. Largest kernels of the
  category (8192x50272x768, 32768x30528x768, 175360x2304x768). First target: model the 16-bit rowvec stage.
- 15 have pipeline facts (fp32 outputs with aux adds, MN/MN weight-gradient + residual).
- 5 refused by the chain analyzer (BackendUnsupported at the default config): `(bias + acc) * 0.125` on
  8192x768x768, `acc * (mask.bool().float() * 2.0)` (dropout scale) on 128x9216x4096, 128x16x16 `mul, sub`,
  128x1000x768 two-aux add.
- 1 CPU-bind error (30522x128x32768 MN/MN fp32: a device query in the split-K path), 3 tiny (no tcgen05).

## Chunk 5 (c01 K/K remainder, 18 kernels; 2026-11-07)

- Cold full autotunes at 416779e05 on GPUs 1 and 3 (shared with another user's processes for part of
  the run). Fair harness on GPU 1, two passes: geomean 0.975, 5/18 >= 0.99.
- Below the bar and why (nvjet kernel names from the CUDA trace):
  - C5_01 1000x2560x2048 0.93, C5_06 2048x2560x2560 0.93, C5_07 4096x1536x1536 0.94: nvjet swaps the
    operands ("h" kernels, 144x128 / 176x128 per CTA): the flexible MMA-N tiles M (1000 = 7 x 144),
    the 2-CTA M tiles N. Helion's best is the aligned-N 256x160 pair (10% more work per SM).
  - C5_12 4096x2560x2560 0.90: nvjet 256x192 (2x1, 7 stages); Helion's 192 clipped pair reads 0.80
    (224 tiles = 3.03 waves), 224 0.88, 160 0.91 -- nvjet balances its partial wave.
  - C5_13 16384x768x2304 0.95, C5_16 25088x1536x512 0.95: the default config is better (0.982,
    0.975); the final selection's rounds swung 19% within one config under GPU contention.
  - C5_02-05 (M=1000) 0.976-0.984, C5_14/15/17 0.983-0.988: same tiles as nvjet, per-tile gaps.
- Default-config finding: C5_01's default was the flat 128x16 kernel (0.11): the cluster_m=2 work
  estimate counted whole tiles only (30 < 37 pairs). Fixed with the aligned-N default candidates
  (commit "Autotune-off default from the aligned-N pairs"): c01 defaults change on 19 kernels,
  0.860 -> 1.020 geomean over them; MN-major B halves narrower than 64-byte swizzle atoms (96, 160,
  224) are no longer seeded (0.74 / 0.66 vs 1.09 / 0.88).

## c01 chunks 6-7 pre-survey (K/MN, 35 kernels)

- All have pipeline and dense facts. C7_15/C7_17 (K=8008 fp32 out) get no seeds and the flat
  default (partial K tile of 8 elements).

## Chunk 6 (c01 K/MN, 18 kernels; 2026-11-07)

- Cold full autotunes at a41275f92 (frozen /tmp/helion_at6) on GPUs 1 and 3. Fair harness on GPU 1:
  old default 0.986, new default (aligned-N default commit) 0.999, autotuned 1.013 geomean; 14/18
  >= 0.99 (C6_07 0.90 -> 1.05, C6_05 0.93 -> 1.03).
- Below the bar: C6_18 20008x768x2048 0.904 (partial last wave: 237 256x256 pairs = 3.2 waves),
  C6_13 6272x2048x1024 0.975, C6_08 32768x256x1024 0.981, C6_11 8192x1024x4096 0.982 (its default
  reads 1.019: identical configs read up to 0.7% apart within one run; the final selection times one
  argument placement, the fair harness eight).

## c08 prep (bias epilogues; 2026-11-07)

- Default survey (current stack, B200): 30/55 c08 kernels default to the flat 128x16 kernel. 23 of them
  are addmm epilogues with a 16-bit bias row: `analyze_pipeline_smem_facts` refused any broadcast aux
  for 16-bit inputs, so no pipeline facts -> no paired-pipeline seeds and no default tile.
- `[cutedsl] Allocation proof for 16-bit bias-row epilogues` records the bias rows (the SMEM model
  already prices row stages under pre_acc_wait). 19 defaults change; all verified against the FX region
  (`/tmp/verify_kernel.py`). First reads (1 placement): 16384x30528x768+bias 0.088 -> 0.736,
  8192x768x768+bias+residual 0.187 -> 0.815.
- Harness fix (better-benchmark `handedit_multi.py`): the correctness check compared against a bare
  `a @ b` (every bias epilogue read WRONG and was dropped) and the shared ATen inputs were taken from the
  kernel's reinterpreted views (a region input of shape (4, 2048, 768) arrived as (8192, 768) and the
  region failed to compile). Both now use the ARGS buffers by name; the reference is the eager region.
- Still flat after the change (12): M=128 decode shapes (128x16x16, 128x768x768, 128x1000x768,
  128x9216x4096 bool mask, 128x50400x4096), 1000x32x2880, 25216x576x192, 30000x128x4096 and
  30522x128x32768 (MN-major A, FP32 out), 1000x4096x4096 + residual (only role_local_with_scheduler
  seeds survive the partial M tile with an exact aux).
- Default placement: under `pre_acc_wait` an output-edge bias GEMM keeps every tile on the SIMT store
  (`output_edge_tma_store_smem_fits` refuses the hybrid TMA store beside rows); `post_acc_wait` takes
  the TMA store for full tiles. 19 changed defaults, SIMT vs TMA store: TMA wins from ~16 tiles per CTA
  pair (16384x30528x768 0.63 -> 0.92), SIMT below (32768x576x192 0.50 vs 0.29, 2048x8008x2560 1.06 vs
  0.86). Lifting the refusal for unstaged rows makes both placements TMA (few-tile kernels lose), so the
  default keys on tiles per CTA instead (`[cutedsl] Output-edge bias defaults take the TMA store on many
  tiles`). 19-kernel default geomean: flat 0.11 -> facts 0.65 -> placement rule 0.85.
- GPU contention: jansel's autotunes now hop across GPUs 0-3 (GPU 1 at 00:26-00:40 during the at9
  re-runs: rounds 2-4 of every finalist read 30% slow, 907 MHz under the 750 W cap). The e1_03/e1_07
  at9 autotunes were re-run on GPU 2.

## Chunk 7 (c01, 17 kernels) and the K-tail family (2026-11-08)

- Cold full autotunes at 6cdb5137e; fair pass 1 (GPU 3, 8 placements): autotuned 0.977, 11/17.
  Below the bar: C7_06 25344x768x2304 0.947, C7_07 32768x768x2304 0.978, C7_08/10 100352x256xK
  0.974/0.982, C7_15 2048x2560x8008 0.774, C7_17 4096x2560x8008 0.748.
- C7_15/17 (K=8008, whole M/N): the K tail confined them to the flat one-CTA pipeline.
  `[cutedsl] Edge + K-tail pairs for K tails on whole tiles, and as the default`: the edge + K-tail
  family takes 16-bit K tails on whole tiles (single ragged edge + K tail fails the runtime guard),
  and the default promotes its plain CLC seed. Re-autotuned (326e04b82): C7_15 0.75 -> 0.93,
  C7_17 0.70 -> 0.91 (defaults 0.08 -> 0.92 / 0.07 -> 0.89).

## c08 chunk 1 (14 kernels; 2026-11-08)

- Autotunes at 8eb0fc84b/4aef4155f (bias facts, placement rule); fair (GPU 3, 8 placements), best of
  default and autotuned: geomean 1.117 (the 1x384x384, 5x5x64 and 1000x32x2880 kernels run ~2x
  ATen), 5/14 >= 0.99. The big-N bias GEMMs top out at 0.85-0.93 (32768x30528x768 0.846,
  8192x50272x768 0.917, 16384x30528x768 0.930). The corpus bias leaves carry `.to(torch.float32)`,
  so their rows are unpromoted (no SMEM stage, SIMT loads per subtile).
- Not the bias: on 32768x30528x768 (GPU 2, interleaved, L2 flushed) Helion's plain GEMM with the same
  [256, 256, 64] ab6 default runs 1728 us vs cuBLAS 1600 us (0.926), and the bias adds 2.4% (1770 us);
  `torch.addmm` costs the same as `torch.mm`. The short-K large-MN mainloop under the 750 W cap is the
  gap. Promoting the cast-wrapped rows (an `({aux}).to(cutlass.Float32)` leaf counted as promoted,
  staged under pre_acc_wait with a (128, 32) subtile) moved 8192x2304x768 by ~2% (0.954 default ->
  0.983 pre ab6, overlapping ranges) and exposed a hybrid-edge NameError at ab5 (the row stage is
  created in the full-tile branch only): not pursued.
- GPU contention made several autotunes pick worse configs than the default (rounds 2-4 of a final
  selection 30% slow when jansel's jobs land on the GPU); the fair harness measures default and all
  autotuned configs and takes the best.

## Epilogue-chain coverage (c08/c09; 2026-11-08)

- CPU codegen of every c08/c09 kernel's default config (B200 target): c08 50/55 and c09 18/71 compiled;
  the rest raised `BackendUnsupported` on every config (the flat default included). The c08 E2_10 autotune
  failed for that reason.
- Causes: (1) Inductor's 0-d constants (`hl.full([], 0.5, dtype=torch.float32)`; 63/71 c09 kernels and
  6/55 c08 carry them) were not literals to the chain; (2) the lowering's `strip_unused_inputs` turns
  `y * y` into `mul(y, None)`; (3) `mask.to(torch.bool).to(torch.float32)` had no cast step; (4) no
  `where` step (ReLU / threshold backward).
- `[cutedsl] Epilogue chains take repeated operands, 0-d constants and boolean masks`: c08 55/55, c09
  44/71. `[cutedsl] Epilogue chains select between a literal and the carrier`: c09 50/71. All 37 newly
  compiling kernels match their FX region on B200 (`/tmp/verify_kernel.py`; it now pairs each kernel
  output with the region output of its shape: regions may also return constants like `full_default`).
- Still unsupported in c09 (21): selects whose branches both depend on the carrier (SELU backward),
  tanh-approximate GELU backward chains, two-output epilogues (element-type mismatch), and 2 plain
  FP32-cast epilogues refused by `tcgen05_flat_role_coordinates`. Boolean masks load as their Int8
  storage on every element-wise path (partial tiles, the SIMT edge of a TMA-store epilogue, broadcast
  masks) and stay off the staged rings (C-input warp, aux-TMA row stage).
- Seeds: the allocation proof for bias rows also enables the paired-pipeline seed family for bias
  GEMMs (both placements, shallow and deepest fitting ring per tile), as for plain GEMMs; several c08
  autotune winners are its configs (E2_01 256x64x64 ab9 pre, E2_02 128x256x128 ab4 post, E2_06/E2_08,
  E1_03). Three test_cute_backend seed-list tests now exclude that family when pinning the curated ones.

## c08 chunks 2-3 and the staged bias row (2026-11-08)

- Fair (GPU 3, 8 placements, best of default and autotuned): chunk 2 geomean 0.870, 3/14 >= 0.99;
  chunk 3 geomean 1.021, 7/14 >= 0.99. Gaps: K=128 bias GEMMs (4096x30000x128 0.57, 32768x30528x128
  0.89 / 0.885), the big-N short-K bias GEMMs (0.85-0.95), M=128 decode shapes (128x50400x4096 0.91,
  128x1000x768 0.33 untuned), and the MN-major N=128 FP32-output GEMMs (30522x128x32768 0.53,
  30000x128x4096 0.80).
- K=128 bias: 4096x30000x128 runs the plain GEMM in 51 us and the bias GEMM in 110 us (torch 60): with
  two K tiles per output tile the epilogue bounds the pair, and the `post_acc_wait` row load from
  global memory leaves its latency on every subtile. Staging the row (`pre_acc_wait`) needed (a) the
  hybrid-store edge branch to read the row from global memory (it built its view over the stage only
  the full-tile branch allocates: `NameError`, so every such config failed) and (b) Inductor's
  `bias.to(torch.float32) + acc` leaf to count as promoted. Survey (GPU 1, default post ab6 -> pre
  ab5, us): 4096x30000x128 136 -> 98, 32768x30528x128 745 -> 596 and 752 -> 583; K=192..768 shapes
  lose 0-8% (32768x576x192 38.9 -> 53.3 on the SIMT store), hence the default rule: at most two K tiles
  per output tile and at least 16 tiles per pair, the deepest ring that keeps the hybrid TMA store.
- Fair (at14 = edge/promo/default stack, GPU 3, 8 placements): 4096x30000x128 0.53 (old default) ->
  0.76 (new default; 0.78 autotuned pre ab3), 32768x30528x128 0.76 -> 0.98 and 0.77 -> 0.97.
- Big-N short-K: cuBLAS runs `nvjet_sm100_tst_128x256_64x6_2x1_2cta` on 32768x30528x768,
  8192x29056x1024 and 16384x50272x768 (a 128-row pair tile, 64 rows per CTA: twice the accumulator
  stages of the 256x256 pair) and `192x128_64x6` on 25216x2304x768. Helion has no 16-bit 128-row
  pair; next: epilogue warp groups (`tcgen05_epi_warp_groups=2`) on the 256x256 pair.

## c09 coverage: steps that read the carrier more than once (2026-11-08)

- After the chain/where commits, 20 c09 kernels still failed every config: 15 read the matmul result
  more than once (GELU-tanh backward `acc * f(x) + acc * g(x)`, SELU backward `where(c, acc*k1*exp(x),
  acc*k2)`, Hardswish backward `where(c, g*(x/3+0.5), g)`, `where(m, s[()], acc*2)`), 3 are two-output
  epilogues with different output types (`tcgen05 epilogue element-type mismatch`), 2 store or load
  through div/mod index maps (`out[y % 8, y // 8, x]`; `tcgen05_flat_role_coordinates` refusal).
- `[cutedsl] Epilogue chains take steps that read the carrier more than once`: the expression from
  the op down to the nearest node every carrier path passes through is one step (`_carrier_join`);
  selects inside expressions (`_SelectTensorExpr`); one leaf per input across steps (`_chain_of`).
  c09: 51 -> 65/71 compile (c08 55/55). The 14 new ones match their FX region within the bf16
  region's own error (5 differ by 1-2 bf16 ulps on large values and are closer to the FP32-operand
  region than the bf16 region is; the K=197952 SELU kernel on the flat default differs by
  accumulation order).
- c09 chunk files: `c09_chunk{1..5}_variants.json` (65 compiling kernels, F1_01..F5_09). Chunk 1 is
  13 bmm kernels whose defaults are the [1, 16, 16, 16] flat tile: their stores write through
  permute/reshape index maps (`out[z // 64, (x + 512y + 32768z) % 512, ...]`) or read a batch-broadcast
  aux, and `cute_tcgen05_search_enabled` is off for them.
- Next on the store side: 32-bit and 16-bit two-output epilogues; on the perf side, `pre_acc_wait` ab6
  + the hybrid TMA store for many-tile bias GEMMs (a reviewer measured 208 vs 211 us at K=768) needs
  the SIMT-store choice for few-tile shapes to move off `output_edge_tma_store_smem_fits`'s refusal
  (e.g. an explicit `tcgen05_c_store_mode="direct"`).

## E1 hill climb, round 2 (2026-11-08)

- Before (c08 start, flat SIMT defaults) -> round-1 best: see goal.json; 9/14 below 0.99 (big-N K=768 bias
  GEMMs 0.85-0.97, 8192x2304x768 0.95, 25216x576x192 0.95 autotuned).
- quack's 256x256 2-CTA GEMM (the same tile; cuBLAS's nvjet_sm100_tst_128x256_64x6_2x1_2cta is also a 256x256
  pair, 128x256 per CTA) runs 32768x30528x768 + bias at 0.96-1.03 of cuBLAS where Helion ran 0.79-0.85: not the
  tile. ncu: Helion's per-SM active cycles spread 1.29-1.49M (quack 1.28-1.34M) with equal tensor work per SM.
- Per-CTA clock64 stamps (hand-instrumented generated code, `c08_e1_hc/trace/`): the 128 partial-N tiles (N=30528
  ends 64 columns into tile 120) stall the MMA ~27K cycles each (SIMT edge store: one 16-bit element per thread
  and instruction; a full tile's epilogue takes 3.3K), and the static schedule gave 37 of 74 pairs 3-4 of them:
  those pairs end at 1.43-1.46M cycles, the rest at 1.34M.
- `[cutedsl] Output-edge epilogues with register auxiliary loads store every tile through TMA`: the
  partial-output TMA store (D descriptor clamps) now takes SIMT aux loads (rows staged per tile through a
  predicated copy, other operands behind the full-tile guard). Defaults: 32768x30528 0.866 -> 0.920,
  16384x30528 0.946 -> 0.968, 8192x50272 0.932 -> 0.968 / 0.968 -> 1.012, 4096x128104 0.975 -> 1.039 / 0.970 -> 1.061.
- `[cutedsl] Output-edge auxiliary GEMMs search the tile order and default to the swizzle`: the aux-edge search
  projection reset l2_groupings and the swizzle to 1; both stay searchable, and the family's default takes
  swizzle 8 on the plain raster (32768x30528: 0.92 -> 0.99; the grouped raster (4) costs 5-13% on the others).
- `[cutedsl] Direct-entry bias defaults stage the promoted row`: whole-tile bias GEMMs (FFI direct entry) read
  the bias row from GMEM per (128, 32) subtile; per-CTA globaltimer stamps on 8192x2304x768 show the last commit at
  18.6 us and the end at 22.7 us (the last tile's epilogue tail). pre_acc_wait at ab=6: 0.945 -> 0.983,
  16384x2304x768 0.979 -> 1.000, 32768x1536x768 0.980 -> 0.982. The FFI projection used to strip the placement
  from every FFI search candidate.
- `[cutedsl] One-CTA N-edge GEMMs store every tile through TMA and run persistent`: 25216x576x192 + bias (cuBLAS:
  nvjet 192x128 one-CTA tile) could only run whole 128x64 one-CTA tiles (A re-read 9x per tile row; 0.95); one CTA
  now takes the partial-output TMA store for N-edge tiles and the search keeps such persistent samples: 128x256 ab3
  0.99 (default), 128x128 ab6 1.03 (cold autotune). Side effects on other chunks' defaults (regression check,
  old -> new default): E3_01 128x50400x4096 0.16 -> 0.82, E3_02 0.84 -> 0.94, c01 H12 512x30528x32768 0.18 -> 0.88,
  E2_03/04/05/08 +0.8-2% (direct-entry staged row).
- Tile order (clean sweep, GPU 2, 6 placements): groups of 32 M tiles win or tie everywhere (32768x30528 0.96, +sw8
  0.98; 16384x30528 0.98; plain raster 0.86/0.96); default changed to l2_groupings=[32] (+ swizzle 8 for A > 32 MB).
- Measurement: GPUs are power-capped at 750 W; under sustained back-to-back runs the big GEMMs throttle to ~800 MHz
  and Helion ties cuBLAS (16384x30528x768: 841-857 vs 851 us; 8192x2304x768: 28.5 vs 29.2 us). In the fair harness
  cuBLAS's time moves up to 8% with buffer placement (582-631 us on 16384x30528x768) while Helion's is flat (638-640),
  and GPU 3 reads 2-7% lower than GPUs 1/2. Hand-edits that did not move 16384x30528x768: a counted MMA-warp tile loop
  (no scheduler calls: 0.977 -> 0.979), the scheduler-warp strategy (+0.4%), groups of 16/64, c_stages=4 (no room).
  ncu: Helion reads 44% less DRAM than cuBLAS (147 vs 263 MB) but 11% more L2 sectors (170.7M vs 153.7M) for the
  same XBAR-delivered volume (188.9M sectors).

## E1 result (closed 2026-11-08; owner: move on to E2)

Fair harness, geomean of GPUs 2 and 3, 8 placements; after = best of the new default and cold autotunes
(`artifacts/.../c08_e1_hc/final/final3_g{2,3}.jsonl`). 7/14 >= 0.99: E1_01 1.99, E1_04 2.11, E1_05 2.02,
E1_07 0.996, E1_11 1.000, E1_12 1.015, E1_14 1.045. Below the bar (blocked, see goal.json blocked_note):
E1_02 0.964, E1_03 0.939, E1_06 0.982, E1_08 0.988, E1_09 0.948, E1_10 0.982, E1_13 0.984.
Before (c08-start default) -> after: 0.07-0.24 -> 0.94-2.1 for the 11 kernels whose c08-start default was flat.

## E2 (started 2026-11-08)

- before configs: `c08_e2_hc/cfg/e2_XX_before.json` (c08_default dNN_old for the kernels in changed_c08.json, else
  c08_e2_fair def12); current default def23 from tree 08676e226; round-9 autotunes at9.
- E1 review round 3 (folded into the E1 commits): the edge swizzle repair is general (a flat walk, and
  `tcgen05_cluster_n=2` on edge/K-tail shapes, whose split scheduler refuses a swizzle); the grouping is 32 only over
  more than 32 M tiles (fewer: groups of 32 decode the plain raster); the swizzle-8 default is gone (four 8-placement
  runs on 32768x30528x768: 0.95/0.97, 0.92/0.92, 0.91/0.97, 0.98/0.94 without/with it); one-CTA N-edge persistence
  only on 128-row tiles and the normal store (64 rows: aux fragment layout mismatch; `direct`: no TMA epilogue).
- E2 rules (WIP commit): output-edge register-aux defaults take `post_acc_wait` (the partial-output TMA store) at any
  tile count; aux-edge groups of 32 M tiles on long N sweeps (>= 32 N tiles), 4 on short ones (plain raster where
  the group is as tall as the M tiles).
- Staged-row lookahead (commit "Persistent partial-output epilogues load their staged rows one tile ahead"):
  E2_07's epilogue spent 0.96 us of each 2.3 us tile staging the bias row (synchronous copy 0.54, fence 0.39,
  barrier); the role now copies the next tile's row with cp.async into the other of two slots (a lookahead
  scheduler), waits with `cp.async.wait_group 1`, no fence. Defaults, old -> new: E2_07 0.92 -> 1.06, E2_12
  1.04 -> 1.12, E2_13 1.04 -> 1.13, E2_01 1.06 -> 1.10, E1_14 0.99 -> 1.01, E3_03 0.97 -> 0.99; K=768 big grids
  unchanged within noise.
- E2_10 (128x1000x768, `(bias + acc + res) * 0.5`): the cold autotune (tree e2b) finds 128x8 persistent one-CTA
  tiles, 4.58 us against ATen's 6.03 (nvjet 64x16 + a Triton pointwise): 1.31 on the CUDA-graph harness (the
  autotuner's own 8.2 us includes the 4 us launch).
- Remaining below 0.99 (defaults / round-9 autotunes): E2_03 0.98/0.99, E2_06 0.98/0.98, E2_09 0.94/0.98,
  E2_14 0.97/1.00 -- big-grid K=768-1024 bias GEMMs like E1's blocked set; cold autotunes on tree e2d running.

## E2 result (closed 2026-11-08)

Fair harness, geomean of GPUs 0 and 1, 8 placements, tree e2d (`c08_e2_hc/m3_g{0,1}.jsonl`); after = best of the
new default (def25) and the cold autotunes (at9, at10, at11). 10/14 >= 0.99: E2_01 1.163, E2_02 1.105,
E2_04 1.000, E2_05 1.139, E2_07 1.060, E2_08 0.994, E2_10 1.301, E2_11 1.054, E2_12 1.061, E2_13 1.063.
Below the bar (blocked, goal.json blocked_note): E2_03 0.975, E2_06 0.978, E2_09 0.978, E2_14 0.958 -- big
K=768-1024 bias GEMMs of E1's blocked class; E2_09 and E2_14 are byte-identical kernels 2% apart in the same runs.
Review rounds 4-5 on the E1 commits (folded): column-major outputs with auxiliary operands keep the split store
(the partial-output body's aux reads do not take the accumulator layout); one-CTA N-edge persistence needs TMA
operands and known store facts; a one-CTA sample shortens its AB ring instead of going flat; an exact-shape
residual takes the one-CTA N-edge default only from 8 K tiles; the partial-output row stages go through the
default-layout arena (`extra_stage_bytes`). PRs are deferred (owner): push the whole stack after the F chunks.

## E3 (c08 chunk 3, 2026-11-08)

- Start (2-GPU, m1): 7/14 >= 0.99 with the best of default and earlier autotunes. Gaps: E3_07 30522x128x32768
  0.54, E3_14 30000x128x4096 0.79, E3_01 128x50400x4096 + bias 0.91, E3_08 32128x512x8192 (two FP32 residuals)
  0.91, E3_11 8192x1024x1024 (bf16 residual, FP32 out) 0.93, E3_03 0.955, E3_02 0.989, E3_13 0.991.
- E3_07: A is x.T of a 30522-wide activation (K stride 61044 B, not a 16-byte multiple), so the TMA pipeline gets
  A through `stage_aligned_operands`' host copy: 0.5 ms fill + 1.4 ms transposing copy (profiled on a shared GPU)
  ahead of a 0.37 ms GEMM; with the copy cached the GEMM alone runs 1.21 of ATen (cutlass_80 align2 kernel,
  485 us). Tried: per-residue-class TensorMaps (K = 4q + r, each class a 16-byte-aligned base rounded down, MN
  shifted by the rounding, both operands' stages holding the same K rows per K block; /tmp/kres_attempt). The
  multi-copy machinery is right (forced 2 classes on aligned strides: exact), but a TMA box start whose innermost
  byte offset is not a 16-byte multiple is an illegal instruction (8-byte coordinate offset: CUDA error 715 with
  SW128 and SW0 SMEM; 16-byte offset runs). Remaining route: a cp.async (4-byte) producer for the misaligned
  operand. Blocked.
- E3_14 (N=128, MN-major A, memory-bound on A): 235 128x128 tiles = 1.6 waves; cuBLAS runs one wave of 146 CTAs
  with swapped 416-wide tiles (nvjet 64x416 2cta). Larger stages (bk64/128) and M-paired tiles lose; the pair
  config fails codegen. Needs a swapped orientation with wide MMA N, or stream-K.
- E3_01: the default took 128x256 (197 tiles, 1.33 waves); decode GEMMs (one row of M tiles) now also get the
  one-CTA 128x128 candidate (394 tiles, 2.66 waves): 0.83 -> 0.99 (commit "Decode GEMMs default to the one-CTA
  128x128 tile"; no other corpus default changes).
- E3_08: today's cold autotune finds the M128 pair at 128x256x128 cluster 2 (64 rows per CTA, cuBLAS's tile):
  0.91 -> 0.96-0.98. One-CTA epi_tma aux rings do not fit beside ab>=4; the 256x256 pair runs L1-bound on the two
  FP32 residual reads (78% L1TEX stalls).
- E3_11: one wave of 64 pair tiles; the C-input-warp strategy is best (0.91-0.93). The register prefetch ring and
  hoist are gated on aux dtype == output dtype (the aux-TMA detector); relaxed in an experiment they ran slower
  than the C-input warp here.
- Default-heuristic fix (folded into the one-CTA commit): the one-CTA N-edge residual gate reads a dtype-agnostic
  detector (`host_function_has_tcgen05_exact_shape_aux_operand`): a bf16 residual of an FP32 output was not gated.

## E3 result (closed 2026-11-08)

Fair harness, ratio of ATen time, 2-GPU geomean; before = the pre-chunk default; after = best of the
current default (def27) and cold-autotuned configs.

| kernel | shape | before | after | config | >= 0.99 |
|---|---|---|---|---|---|
| E3_01 | 128x50400x4096 K/K | 0.159 | 0.990 | def27 | yes |
| E3_02 | 32768x30528x768 K/MN | 0.097 | 0.975 | at20 | no |
| E3_03 | 16384x30528x768 K/MN | 0.095 | 0.971 | at20 | no |
| E3_04 | 1x384x384 K/K | 0.089 | 1.964 | at11 | yes |
| E3_05 | 8192x768x768 K/K | 1.196 | 1.192 | def26 | yes |
| E3_06 | 128x16x16 K/MN | 1.252 | 2.204 | at13 | yes |
| E3_07 | 30522x128x32768 MN/MN | 0.235 | 0.998 | def29 (cp.async A, paired 256x128) | yes |
| E3_08 | 32128x512x8192 MN/MN | 0.919 | 0.967 | at20 | no |
| E3_09 | 128x768x768 K/K | 0.170 | 1.614 | at11 | yes |
| E3_10 | 128x16x16 K/MN | 1.240 | 2.186 | at13 | yes |
| E3_11 | 8192x1024x1024 K/MN | 0.841 | 1.084 | at22 (epi_tma pair ab5) | yes |
| E3_12 | 2048x2560x2560 K/MN | 0.953 | 1.022 | at11 | yes |
| E3_13 | 29056x1024x8192 MN/MN | 0.951 | 0.989 | at11 | no |
| E3_14 | 30000x128x4096 MN/MN | 0.194 | 0.857 | def30 (pair 256x128 ab8) | no |

9/14 at 0.99 (E3_01 at 0.990 on the new decode default; E3_13 0.989-0.991 across rounds). Blocked notes in goal.json.

Reopened E3_07 / E3_11 / E3_14 (2026-11-08, owner's targets 0.99 / 0.95 / 0.95):

- E3_07: the epilogue warps copy the unaligned M-major A (x.T, 61044-byte K rows) with cp.async
  in row classes of 16/8/4 bytes; a 256-row tile pairs two 128-row CtaGroup.ONE subtiles on one
  B stage (B is 8 MB, re-read by every M tile). Prototype v4 (hand) 1.018, Helion default 0.998
  (128-row tile 0.931). Pitfalls hit: separate edge-tile copies kept their partitions live and
  spilled 255 registers (2.26 ms); two UMMA statements on one B fragment got a CTA barrier from
  the barrier pass that deadlocked the cp.async producer.
- E3_11: the cold autotune on the mixed-dtype aux tree finds epi_tma 256x256 pair ab5: 1.084.
- E3_14: N=128 over M=30000; every power-of-two M tile quantizes to ~0.80 of 148 SMs (cuBLAS:
  swapped 64x416 2-CTA tiles, 146 CTAs in one wave). The owner kept power-of-two tiles; the
  narrow-N pair is now the default (0.19 -> 0.857), M-paired one-CTA with TMA A measured 0.795.

## E4 result (c08 chunk 4, closed 2026-11-09)

Fair harness (6 rounds x 8 placements, CUDA graphs, interleaved with the ATen region), GPUs 0+1,
2-GPU geomean; "before" = the chunk's start-of-E4 default, "default" = the default on the E4 commits,
"at13" = the earlier cold autotune. Chunk geomean (default) 0.461 -> 0.995; 7/13 >= 0.99.

| Kernel | Shape | before | default | at13 | notes |
|---|---|---|---|---|---|
| E4_01 | 128x9216x4096 + bias | 0.177 | 1.137 | 1.140 | small-grid 128-row pair |
| E4_02 | 4096^3 + 3 operands | 0.969 | 1.047 | 1.034 | epi_tma, grouped by 8 |
| E4_03 | 32768x256x256 + 3 | 0.498 | 0.953 | 0.933 | epi_tma 256x128, ring 4 / AB 4 |
| E4_04 | 4096^3 + 11, FP32 out | 0.665 | 0.869 | 0.871 | epilogue bound (see below) |
| E4_05 | 16384x4096x4096 + 11 | 0.748 | 0.842 | 0.834 | " |
| E4_06 | 4096x16384x4096 + 11 | 0.761 | 0.839 | 0.793 | " |
| E4_07 | 1000x4096x4096 + residual | 0.092 | 0.934 | 0.771 | M-edge pair, producer warp, bk=32 ab=8 |
| E4_08 | 8192x768x768 | 1.039 | 1.172 | 1.226 | |
| E4_09 | 2048x8008x2560 | 0.100 | 1.133 | 1.103 | |
| E4_10 | 32000x1024x8192 + 1 | 1.069 | 1.089 | 1.098 | |
| E4_11 | 8192x1024x1024 + 3, FP32 | 0.552 | 0.998 | 0.769 | epi_tma |
| E4_12 | 16384x1024x1024 + 3 | 0.542 | 0.986 | 0.808 | epi_tma |
| E4_13 | 16384x768x768 + 3 | 0.429 | 1.019 | 0.655 | epi_tma |

Commits: small-grid 128-row pair default; whole-tile residual pairs on the epilogue TMA ring
(transcendental and mask gates, ring depth for short K, grouping 8 on staged long sweeps); residual
GEMMs over an M edge on the 256x256 pair (+ the persistence-model merge fix); aux producer warps keep
any AB ring within two 128-deep stages' bytes (E4_07 0.79 -> 0.93).

Open: E4_04/05/06 read eleven bf16 residuals per FP32 output element. ncu on E4_04 (default):
L1/TEX throughput 78%, DRAM 41%, long-scoreboard stalls dominate; the staged operands share the SMEM
bandwidth with the MMA operand reads. epi_warp_groups=2, c_stages=4 and deeper aux rings did not help;
needs an epilogue redesign (e.g. operands read once per tile through fewer, wider stages).

## F (c09 epi_complex, 2026-11-09)

F1 (batched attention bmms with fused head merges), screens on one GPU, of the ATen bmm, SIMT
fallback -> default: F1_03 1.06, F1_04 1.09, F1_05 1.35, F1_06 1.07, F1_07 0.97, F1_10 1.03, F1_11 1.30,
F1_12 1.44, F1_13 1.05, F1_14 1.34 (all 0.004-0.03 before); F1_09 (two outputs, harness cannot check)
0.97 by do_bench, its cold autotune 1.21 ([1,64,64,128] persistent ab6, not generalized yet);
F1_01 0.21 (96x65 output, unaligned rows), F1_02/F1_08 0.01 (1000x1000 per batch, 1001-element
rows): unaligned destinations, deferred with the 21 such c06 kernels (needs a coalesced staged
SIMT store). Commits: batched column-major operands through TMA (fp32 and 8-column K-major B keep
the generic lowering; a row-major fp32 bmm returned wrong results on the base); batched mixed-radix
stores (L mode, column-major D, chains, (64, 32) boxes, one-CTA defaults from the layout).

Transcendental epilogues (erf/exp/tanh/sigmoid on an operand; GELU, SELU, SiLU backward, 27 corpus
kernels): one epilogue warp group issued the exact polynomials one dependent step at a time and the
epilogue outlasted the MMA (GELU backward 16384x4096x1024: 0.33). Two groups: 1.02 (three 0.75, four
0.64; fast-math 1.38 is the user's opt-in). Screen, one -> two groups: F3_06 0.32 -> 1.01, F3_11
0.33 -> 1.02, F3_12 0.39 -> 1.17, F4_02 0.25 -> 0.99, F2_06 0.79 -> 1.19, F3_01 0.60 -> 1.02, F4_01
0.19 -> 0.80, F3_04 0.23 -> 0.76, F5_01 0.54 -> 0.69 (all 27 improve).

Review rounds 4-5 (folded): the E2 commit got its message (it was WIP); the aux producer commit is
squashed into the M-edge one (the M-edge default overflowed SMEM with two residuals without it); the
FFI direct-entry default no longer takes atomic kernels; small grids no pair fills in one wave take
the fewest waves (128x9600x4096 137 -> 29 us); the mask-only TMA ring is refused; the inductor
test pins its hybrid-store config (the mask-only default moved off the pair at the any-dtype aux
commit, back at the M-edge commit).

Small grids no 128-row pair serves (64-row M, K tails, very long K) now split K on a one-CTA
64x64x64 tile (slices of >= 4 k-blocks, <= 3 CTAs per SM), screen of the ATen GEMM: F5_06 0.13 ->
1.56, F5_04 0.31 -> 1.70, F5_09 0.27 -> 1.65, F4_12 0.31 -> 1.52, F2_08 0.31 -> 1.48, F2_07 0.27 ->
1.29, F3_02 0.31 -> 1.48, E1_01 0.14 -> 1.79 (64x32 tiles), E2_10 0.35 -> 1.33, F2_11 0.008 -> 0.57
(the largest split, 16, caps it). Caveat: a split launch has no fast relaunch (its workspace is a
launcher-owned tensor): ~100 us host time outside CUDA graphs. Long-N sweeps group by 32 from every
pair seed (plain 16384x30528x768 0.92 -> 0.96).

Open in c09: F1_01/F1_02/F1_08/F2_04 (unaligned outputs), F4_06/F3_04/F2_14/F5_01 (0.69-0.80),
F5_02 (8192x480x240 + sigmoid, 0.50), F2_11 (needs splits past 16 or a pair split).

## G (c06 bmm_attn, 2026-11-09; closed for now)

99 batched kernels. Fair run (6 rounds x 8 placements, two GPUs), of the ATen bmm: geomean 0.908, 27
at >= 0.99 (``artifacts/inductor-epilogue-2026-11-06/c06_fair_final``). Commits: batched pair defaults
walk their batches slowest (256 x 512x1024x64 0.78 -> 0.90); batched GEMMs with K not whole in 64-deep
k-blocks take the one-CTA tile (1024 x 128x128x32 0.17 -> 0.98); many small matrices or an operand /
output TMA cannot address take the one-CTA 64-row tile (window attention 32768 x 49x49x32 0.023 ->
0.93); the 37-batch pair case. Open: the B x 512x1024x64 store-bound family (0.80-0.90; DRAM 63%
against nvjet's 76%).

## F revisit (2026-11-09 afternoon): sub-0.99 kernels (stopped here)

Harness fix first: the corpus harness's flush detector (``abab_dev.profile_side``) splits calls at
any kernel named memset/fill/zero/reduce, so a split-K launch's workspace memset (replayed with every
graph launch) and ATen-side reduce/fill kernels cut calls in two, and the median under-counted them
(split-K Helion kernels ~2x: F5_04 split 5.6 us "1.67" -> 13.0 us, 0.77). F re-measured with a
detector that only takes the >= 15 us cache flush (``hm_fix.py``, 6 rounds x 8 placements, GPU 0,
``c09_fair_fixed``): **50/65 >= 0.99, geomean 1.061**. GPU 0 hits its 750 W cap during these runs
(SM clock 1965 -> ~1.5 GHz), so epilogue-bound kernels vary between sessions (F4_07 1.21 at 14:00,
0.96 at 18:00 with identical code; both detectors agree within a session). The c01/c08/c06 numbers
predate the fix.

Commits this afternoon:
- Narrow batched persistent defaults walk their batches slowest: F1_07 0.96 -> 1.17, F2_03 0.96 ->
  1.10 (F1_03/06/10/13, F2_02 and five G kernels up 5-15%).
- A mask beside the ring's operand reads one subtile ahead (two groups, deep ring): F2_09 0.88 ->
  0.96.
- Two register masks over a short K loop take the capped three-group ring: F5_07 0.91 -> 1.00.
- Outputs with unaligned rows store through a staged SMEM tile (each warp instruction one row's 32
  consecutive elements instead of 32 rows' one; flat tiles over one k-block at one AB stage): F1_08
  0.34 -> 0.71, F1_02/F2_04 0.34 -> 0.63-0.65; G5_03 0.79 -> 1.66, ten window-attention G kernels
  0.93 -> 1.21-1.27 (old detector; no memset in either). ncu on F1_08: L1/TEX 72%, DRAM 11%, five
  CTAs per SM, one 64x128 tile each.
- Split K only over K loops of >= 32 k-blocks (the memset and reduction lose on 16): F5_04 0.92,
  F2_08 0.90, F4_12 0.90, F5_09 0.85, F3_02 0.82, F2_07 0.66 unsplit; F5_06 0.68, F2_11 0.54 split.
- Small grids into outputs no TensorMap addresses take the one-CTA tile: F1_01 0.29 -> 1.46 (its
  ATen region's second kernel was also cut by the old detector: 3.1 -> 9.3 us).

Open in c09 (15 below 0.99): F5_02 0.50 (N edge + K tail: pairs and persistent tiles refuse partial
tiles), F2_11 0.54 (serial last-arriver reduction), F2_04/F1_02/F1_08 0.63-0.71 (one small tile per
CTA), F2_07/F5_06/F3_02/F5_09/F2_08/F4_12/F5_04 0.66-0.92 (small grids on the one-CTA 64x64 tile),
F4_07/F2_09/F2_14 0.96-0.97 (power-cap sensitive).
