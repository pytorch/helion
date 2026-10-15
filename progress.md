# K18 MXFP8 quantization progress

## Run

- Worktree: `/home/shangdiy/helion-k18-cute-hillclimb-20261014`.
- Base: Helion main `a03e0f28406db93df8d6e9a5fd03bf995dfcb16c`.
- Target: all six published BF16 TorchAO RCEIL shapes; exact E4M3 payload and E8M0 scale bytes.
- GPU constraint: **devices 2–3 only**, per latest user instruction. All six final comparisons and verification use GPU 2, at 1400 W. Earlier GPU 1 receipts remain archived; GPU 3 is occupied.
- Completion: every shape at least 0.99× best baseline and geometric mean at least 1.0×; short configuration sweeps and pinned interleaved verification.
- Functional changes are committed in five logical benchmark/backend/test commits; progress and report are in the final `[noland]` commit. Raw results and configurations: `artifacts/k18-mxfp8-2026-10-14/`.

## Current measurements by shape

Historical starting timings are trusted at the user’s instruction. Optimization uses short sweeps and pinned configurations; no cold autotunes. A fresh same-GPU anchor replaces the historical target for a measured ratio when available.

| Case | Shape | Historical CuTe (µs) | Paired best baseline (µs) | Final CuTe (µs) | Baseline / CuTe | Correctness | Status | Artifacts |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | 1x128 | 1.760 | 1.760 (live helion-triton) | 1.728 | 1.019× | PASS exact / graph | faster | [before](artifacts/k18-mxfp8-2026-10-14/historical/K18_00_cute.json) · [latest](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case00/helion-cute/result.json) · [baseline](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case00/helion-triton/result.json) |
| 1 | 137x64 | 1.920 | 1.808 (live helion-triton) | 1.792 | 1.009× | PASS exact / graph | faster | [before](artifacts/k18-mxfp8-2026-10-14/historical/K18_01_cute.json) · [latest](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case01/helion-cute/result.json) · [baseline](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case01/helion-triton/result.json) |
| 2 | 128x1024 | 2.176 | 1.856 (live transformer-engine-cuda) | 1.873 | 0.991× | PASS exact / graph | within 1% | [before](artifacts/k18-mxfp8-2026-10-14/historical/K18_02_cute.json) · [latest](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case02/helion-cute/result.json) · [baseline](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case02/transformer-engine-cuda/result.json) |
| 3 | 4096x4096 | 19.840 | 8.576 (live transformer-engine-cuda) | 8.480 | 1.011× | PASS exact / graph | faster | [before](artifacts/k18-mxfp8-2026-10-14/historical/K18_03_cute.json) · [latest](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case03/helion-cute/result.json) · [baseline](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case03/transformer-engine-cuda/result.json) |
| 4 | 8192x8192 | 72.000 | 31.760 (live helion-triton) | 31.793 | 0.999× | PASS exact / graph | within 1% | [before](artifacts/k18-mxfp8-2026-10-14/historical/K18_04_cute.json) · [latest](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case04/helion-cute/result.json) · [baseline](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case04/helion-triton/result.json) |
| 5 | 8192x16384 | 140.609 | 60.448 (live transformer-engine-cuda) | 60.337 | 1.002× | PASS exact / graph | faster | [before](artifacts/k18-mxfp8-2026-10-14/historical/K18_05_cute.json) · [latest](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case05/helion-cute/result.json) · [baseline](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case05/transformer-engine-cuda/result.json) |

Geometric mean vs matched baseline anchors: **1.0051×**. All six performance targets pass the 1% match criterion. Verification is complete: both full test suites and lint are clean; final review is LGTM.

Historical receipts use GPU 1 and CUPTI. All subsequent GPU work uses devices 2–3; final verification uses GPU 2. Earlier small-shape GPU 1 probes remain archived. Each ratio’s GPU and timer provenance is recorded in `artifacts/goal.json`.

## Supplied report: historical context

Measured on GPU 1 with Helion `063d8f58`; CUPTI cold L2. These numbers are not used for this run’s completion ratios.

| Shape | Helion CuTe (µs) | Helion Triton (µs) | FlashInfer (µs) | Compiled Torch (µs) | Transformer Engine (µs) | CuTe / FlashInfer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x128 | 1.760 | 1.728 | 1.856 | 1.728 | 4.928 | 1.055× |
| 137x64 | 1.920 | 1.824 | 1.920 | 2.528 | 9.312 | 1.000× |
| 128x1024 | 2.176 | 2.016 | 2.080 | 2.016 | 1.953 | 0.956× |
| 4096x4096 | 19.840 | 10.208 | 10.656 | 10.208 | 8.640 | 0.537× |
| 8192x8192 | 72.000 | 32.129 | 35.168 | 35.233 | 31.969 | 0.488× |
| 8192x16384 | 140.609 | 60.608 | 68.193 | 60.224 | 60.320 | 0.485× |

Raw historical receipts are copied under `artifacts/k18-mxfp8-2026-10-14/historical/`.

## Ranked ideas

1. Compare generated loads, block32 reductions, FP8 conversion and stores with FlashInfer.
2. Improve contiguous vector packets and packed FP8 conversion/stores through general CuTe lowering.
3. Simplify the kernel’s indexing/reduction shape where that exposes the intended layout.
4. Test warp-local group layouts and expose successful structures through general search knobs/seeds.

## Iteration log

| Time (UTC) | Worst shape | Idea / action | Result | Checker |
| --- | --- | --- | --- | --- |
| 2026-10-14T22:32:07.036325+00:00 | pending | Created isolated worktree; archived source report and receipts; prepared six-shape goal. | pending | measurements pending |

## Measurement plan update

User instruction: **“don’t do cold autotune, trust the historical context numbers and start optimize.”**

The cold baseline campaign has been stopped. Its completed largest-shape receipts and partial logs are preserved under `artifacts/k18-mxfp8-2026-10-14/before/` as diagnostics. The starting columns will use the historical receipts. Short pinned probes now validate the indexed K18 kernel and general CuTe FP8 packet store path.

### Optimization iterations

| Time (UTC) | Shape | Trial | Result | Ratio | Artifact |
| --- | --- | --- | ---: | ---: | --- |
| 2026-10-14T23:07:21.211504+00:00 | 8192x16384 | Indexed block32 + packed FP8; V16, two threads/group, block256, flat grid. | 66.464 µs | 0.909× vs same-GPU TE | `artifacts/k18-mxfp8-2026-10-14/probe_indexed_v16/case05/helion-cute/result.json` |
| 2026-10-14T23:12:10.965971+00:00 | 8192x16384 | probe_indexed_v8 | 78.368 µs | 0.771× vs same-GPU TE | `artifacts/k18-mxfp8-2026-10-14/probe_indexed_v8/case05/helion-cute/result.json` |
| 2026-10-14T23:12:24.306476+00:00 | 8192x16384 | probe_unclamped_v16 | 66.016 µs | 0.915× vs same-GPU TE | `artifacts/k18-mxfp8-2026-10-14/probe_unclamped_v16/case05/helion-cute/result.json` |
| 2026-10-14T23:12:38.046610+00:00 | 8192x16384 | probe_unclamped_v16_persistent4 | 70.592 µs | 0.856× vs same-GPU TE | `artifacts/k18-mxfp8-2026-10-14/probe_unclamped_v16_persistent4/case05/helion-cute/result.json` |
| 2026-10-14T23:12:52.287004+00:00 | 8192x16384 | probe_indexed_v8_mpb4 | 78.656 µs | 0.768× vs same-GPU TE | `artifacts/k18-mxfp8-2026-10-14/probe_indexed_v8_mpb4/case05/helion-cute/result.json` |
| 2026-10-14T23:13:10.330538+00:00 | 8192x16384 | probe_indexed_v16_mpb4 | 66.128 µs | 0.914× vs same-GPU TE | `artifacts/k18-mxfp8-2026-10-14/probe_indexed_v16_mpb4/case05/helion-cute/result.json` |
| 2026-10-14T23:13:29.465392+00:00 | 8192x16384 | probe_indexed_v16_persistent4 | 72.704 µs | 0.831× vs same-GPU TE | `artifacts/k18-mxfp8-2026-10-14/probe_indexed_v16_persistent4/case05/helion-cute/result.json` |
| 2026-10-14T23:13:43.811712+00:00 | 8192x16384 | probe_bf16_max_v16 | 72.192 µs | 0.837× vs same-GPU TE | `artifacts/k18-mxfp8-2026-10-14/probe_bf16_max_v16/case05/helion-cute/result.json` |
| 2026-10-14T23:13:57.359328+00:00 | 8192x16384 | probe_bf16_max_v8 | 78.336 µs | 0.771× vs same-GPU TE | `artifacts/k18-mxfp8-2026-10-14/probe_bf16_max_v8/case05/helion-cute/result.json` |
| 2026-10-14T23:24:50.089724+00:00 | 8192x16384 | Two groups per thread; V16,256groups,128x2threads; CUPTI. | 62.017 µs | 0.973× vs same-GPU TE60.385µs | `artifacts/k18-mxfp8-2026-10-14/probe_two_groups_cupti/case05/helion-cute/result.json` |
| 2026-10-14T23:25:24.396698+00:00 | 8192x16384 | probe_bf16_scale_two_groups; CUPTI | 61.407 µs | 0.983× vs TE | `artifacts/k18-mxfp8-2026-10-14/probe_bf16_scale_two_groups/case05/helion-cute/result.json` |
| 2026-10-14T23:25:56.852988+00:00 | 8192x16384 | probe_two_groups_auto; CUPTI | 62.047 µs | 0.973× vs TE | `artifacts/k18-mxfp8-2026-10-14/probe_two_groups_auto/case05/helion-cute/result.json` |
| 2026-10-14T23:26:13.911284+00:00 | 8192x16384 | probe_two_groups_first; CUPTI | 62.016 µs | 0.974× vs TE | `artifacts/k18-mxfp8-2026-10-14/probe_two_groups_first/case05/helion-cute/result.json` |
| 2026-10-14T23:26:32.340962+00:00 | 8192x16384 | probe_indexed_first; CUPTI | 64.576 µs | 0.935× vs TE | `artifacts/k18-mxfp8-2026-10-14/probe_indexed_first/case05/helion-cute/result.json` |
| 2026-10-14T23:26:50.281979+00:00 | 8192x16384 | probe_indexed_cluster8; CUPTI | 64.576 µs | 0.935× vs TE | `artifacts/k18-mxfp8-2026-10-14/probe_indexed_cluster8/case05/helion-cute/result.json` |
| 2026-10-14T23:41:47.207041+00:00 | case5 | native_scale_pure_v2; CUPTI | 61.505 µs | see goal | `artifacts/k18-mxfp8-2026-10-14/native_scale_pure_v2/case05/helion-cute/result.json` |
| 2026-10-14T23:41:47.207602+00:00 | case3 | native_scale_pure_v2; CUPTI | 10.816 µs | see goal | `artifacts/k18-mxfp8-2026-10-14/native_scale_pure_v2/case03/helion-cute/result.json` |
| 2026-10-14T23:41:47.207989+00:00 | case4 | native_scale_pure_v2; CUPTI | 32.768 µs | see goal | `artifacts/k18-mxfp8-2026-10-14/native_scale_pure_v2/case04/helion-cute/result.json` |
| 2026-10-14T23:42:25.586826+00:00 | case5 | bf16_native_pure_v2; CUPTI | 61.377 µs | see goal | `artifacts/k18-mxfp8-2026-10-14/bf16_native_pure_v2/case05/helion-cute/result.json` |
| 2026-10-14T23:42:25.587334+00:00 | case3 | bf16_native_pure_v2; CUPTI | 11.009 µs | see goal | `artifacts/k18-mxfp8-2026-10-14/bf16_native_pure_v2/case03/helion-cute/result.json` |
| 2026-10-14T23:42:25.587706+00:00 | case4 | bf16_native_pure_v2; CUPTI | 32.928 µs | see goal | `artifacts/k18-mxfp8-2026-10-14/bf16_native_pure_v2/case04/helion-cute/result.json` |
| 2026-10-14T23:43:06.239093+00:00 | case5 | bf16_scale_pure_v2; CUPTI | 61.377 µs | see goal | `artifacts/k18-mxfp8-2026-10-14/bf16_scale_pure_v2/case05/helion-cute/result.json` |
| 2026-10-14T23:43:06.239583+00:00 | case3 | bf16_scale_pure_v2; CUPTI | 11.072 µs | see goal | `artifacts/k18-mxfp8-2026-10-14/bf16_scale_pure_v2/case03/helion-cute/result.json` |
| 2026-10-14T23:43:06.239948+00:00 | case4 | bf16_scale_pure_v2; CUPTI | 32.961 µs | see goal | `artifacts/k18-mxfp8-2026-10-14/bf16_scale_pure_v2/case04/helion-cute/result.json` |
| 2026-10-15T00:12:33.043418+00:00 | case3 | Packed BF16 words, scalar Int16 output stores (packing missing). | 31.936 µs | rejected | `artifacts/k18-mxfp8-2026-10-14/packed_word_initial/case03/helion-cute/result.json` |
| 2026-10-15T00:12:33.043681+00:00 | case4 | Packed BF16 words, scalar Int16 output stores (packing missing). | 112.161 µs | rejected | `artifacts/k18-mxfp8-2026-10-14/packed_word_initial/case04/helion-cute/result.json` |
| 2026-10-15T00:12:33.043856+00:00 | case5 | Packed BF16 words, scalar Int16 output stores (packing missing). | 218.208 µs | rejected | `artifacts/k18-mxfp8-2026-10-14/packed_word_initial/case05/helion-cute/result.json` |

| 2026-10-14T23:53:04.661718+00:00 | case0 | small_native: case00_bf16_native_winner | 1.920 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/small_native/case00_bf16_native_winner/case00/helion-cute/result.json` |
| 2026-10-14T23:52:30.872029+00:00 | case0 | small_native: case00_native_winner | 1.888 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/small_native/case00_native_winner/case00/helion-cute/result.json` |
| 2026-10-14T23:53:51.841468+00:00 | case1 | small_native: case01_bf16_native_winner | 1.888 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/small_native/case01_bf16_native_winner/case01/helion-cute/result.json` |
| 2026-10-14T23:53:36.250337+00:00 | case1 | small_native: case01_native_historical_geometry | 1.856 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/small_native/case01_native_historical_geometry/case01/helion-cute/result.json` |
| 2026-10-14T23:54:38.890252+00:00 | case2 | small_native: case02_bf16_native_winner | 1.952 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/small_native/case02_bf16_native_winner/case02/helion-cute/result.json` |
| 2026-10-14T23:54:23.274520+00:00 | case2 | small_native: case02_native_historical_geometry | 2.048 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/small_native/case02_native_historical_geometry/case02/helion-cute/result.json` |
| 2026-10-14T23:56:59.945339+00:00 | case3 | mid_geometry: integer_max_v16_two_groups | 10.881 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/mid_geometry/integer_max_v16_two_groups/case03/helion-cute/result.json` |
| 2026-10-14T23:59:10.867019+00:00 | case3 | mid_geometry: native_scale_g1024_t128 | 12.512 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/mid_geometry/native_scale_g1024_t128/case03/helion-cute/result.json` |
| 2026-10-14T23:59:42.203493+00:00 | case3 | mid_geometry: native_scale_g128_r4_v8 | 11.680 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/mid_geometry/native_scale_g128_r4_v8/case03/helion-cute/result.json` |
| 2026-10-14T23:58:06.431997+00:00 | case3 | mid_geometry: native_scale_g128_t128 | 10.112 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/mid_geometry/native_scale_g128_t128/case03/helion-cute/result.json` |
| 2026-10-14T23:57:50.459928+00:00 | case3 | mid_geometry: native_scale_g128_t64 | 10.720 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/mid_geometry/native_scale_g128_t64/case03/helion-cute/result.json` |
| 2026-10-14T23:58:23.047058+00:00 | case3 | mid_geometry: native_scale_g256_t128_rowv2 | 10.816 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/mid_geometry/native_scale_g256_t128_rowv2/case03/helion-cute/result.json` |
| 2026-10-14T23:57:16.746470+00:00 | case3 | mid_geometry: native_scale_g32_t32 | 10.432 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/mid_geometry/native_scale_g32_t32/case03/helion-cute/result.json` |
| 2026-10-14T23:58:39.273944+00:00 | case3 | mid_geometry: native_scale_g512_t128 | 12.128 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/mid_geometry/native_scale_g512_t128/case03/helion-cute/result.json` |
| 2026-10-14T23:58:55.113800+00:00 | case3 | mid_geometry: native_scale_g512_t256 | 10.688 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/mid_geometry/native_scale_g512_t256/case03/helion-cute/result.json` |
| 2026-10-14T23:59:26.316597+00:00 | case3 | mid_geometry: native_scale_g64_r8_v4 | 17.056 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/mid_geometry/native_scale_g64_r8_v4/case03/helion-cute/result.json` |
| 2026-10-15T00:19:25.757360+00:00 | case3 | packed_word_vector: case03 | 10.880 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/packed_word_vector/case03/helion-cute/result.json` |
| 2026-10-15T00:19:36.153844+00:00 | case4 | packed_word_vector: case04 | 32.800 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/packed_word_vector/case04/helion-cute/result.json` |
| 2026-10-15T00:19:46.294688+00:00 | case5 | packed_word_vector: case05 | 61.440 µs | see shape table | `artifacts/k18-mxfp8-2026-10-14/packed_word_vector/case05/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | g64_t64_persistent_blocked_8 | 11.488 µs | rejected; flat 10.080 µs retained | `artifacts/k18-mxfp8-2026-10-14/mid_persistent/g64_t64_persistent_blocked_8/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | g64_t64_persistent_interleaved_16 | 10.592 µs | rejected; flat 10.080 µs retained | `artifacts/k18-mxfp8-2026-10-14/mid_persistent/g64_t64_persistent_interleaved_16/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | g64_t64_persistent_interleaved_4 | 14.176 µs | rejected; flat 10.080 µs retained | `artifacts/k18-mxfp8-2026-10-14/mid_persistent/g64_t64_persistent_interleaved_4/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | g64_t64_persistent_interleaved_8 | 11.392 µs | rejected; flat 10.080 µs retained | `artifacts/k18-mxfp8-2026-10-14/mid_persistent/g64_t64_persistent_interleaved_8/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | v16_two_groups_persistent_blocked_8 | 11.552 µs | rejected; flat 10.080 µs retained | `artifacts/k18-mxfp8-2026-10-14/mid_persistent/v16_two_groups_persistent_blocked_8/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | v16_two_groups_persistent_interleaved_16 | 10.816 µs | rejected; flat 10.080 µs retained | `artifacts/k18-mxfp8-2026-10-14/mid_persistent/v16_two_groups_persistent_interleaved_16/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | v16_two_groups_persistent_interleaved_4 | 11.712 µs | rejected; flat 10.080 µs retained | `artifacts/k18-mxfp8-2026-10-14/mid_persistent/v16_two_groups_persistent_interleaved_4/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | v16_two_groups_persistent_interleaved_8 | 11.040 µs | rejected; flat 10.080 µs retained | `artifacts/k18-mxfp8-2026-10-14/mid_persistent/v16_two_groups_persistent_interleaved_8/case03/helion-cute/result.json` |

### Profiling

NCU full and source reports for CuTe and Transformer Engine are in `artifacts/k18-mxfp8-2026-10-14/profile/k18-mid-20261015/`. Analysis compares launch geometry, cache traffic, and stall PCs; profiler timings use base clocks and are diagnostic.

### Independent packet loading

| Time (UTC) | Shape | Trial | Result | Decision | Artifact |
| --- | --- | --- | ---: | --- | --- |
| 2026-10-15 | 4096x4096 | native_scale_g256_blocked_unroll2 | 10.432 µs | best 9.921 µs; test packed arithmetic next | `artifacts/k18-mxfp8-2026-10-14/mid_unroll/native_scale_g256_blocked_unroll2/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | native_scale_g256_strided_unroll2 | 10.016 µs | best 9.921 µs; test packed arithmetic next | `artifacts/k18-mxfp8-2026-10-14/mid_unroll/native_scale_g256_strided_unroll2/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | packed_bf16_g256_blocked_unroll2 | 10.464 µs | best 9.921 µs; test packed arithmetic next | `artifacts/k18-mxfp8-2026-10-14/mid_unroll/packed_bf16_g256_blocked_unroll2/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | packed_bf16_g256_strided_unroll2 | 9.984 µs | best 9.921 µs; test packed arithmetic next | `artifacts/k18-mxfp8-2026-10-14/mid_unroll/packed_bf16_g256_strided_unroll2/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | packed_bf16_g512_strided_unroll4 | 9.952 µs | best 9.921 µs; test packed arithmetic next | `artifacts/k18-mxfp8-2026-10-14/mid_unroll/packed_bf16_g512_strided_unroll4/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | packed_word_two_groups_strided | 10.528 µs | rejected | `artifacts/k18-mxfp8-2026-10-14/warp_groups/results/packed_word_two_groups_strided/case03/helion-cute/result.json` |
| 2026-10-15 | 4096x4096 | v16_two_groups_strided | 10.464 µs | rejected | `artifacts/k18-mxfp8-2026-10-14/warp_groups/results/v16_two_groups_strided/case03/helion-cute/result.json` |

### GPU constraint update

User instruction: **“use gpu 2-3 only.”** Both earlier campaigns have completed; all subsequent GPU work is restricted to devices 2–3. Small-shape final measurements will use fresh GPU 2 anchors.

| Shape | Trial | Result | Decision | Artifact |
| --- | --- | ---: | --- | --- |
| [8192, 8192] | native_scale_g512_strided_unroll4 | 32.064 µs | within 1% of TE; ABAB pending | `artifacts/k18-mxfp8-2026-10-14/large_unroll/native_scale_g512_strided_unroll4/case04/helion-cute/result.json` |
| [8192, 16384] | native_scale_g512_strided_unroll4 | 60.737 µs | within 1% of TE; ABAB pending | `artifacts/k18-mxfp8-2026-10-14/large_unroll/native_scale_g512_strided_unroll4/case05/helion-cute/result.json` |
| [8192, 8192] | packed_bf16_g512_strided_unroll4 | 32.063 µs | within 1% of TE; ABAB pending | `artifacts/k18-mxfp8-2026-10-14/large_unroll/packed_bf16_g512_strided_unroll4/case04/helion-cute/result.json` |
| [8192, 16384] | packed_bf16_g512_strided_unroll4 | 60.833 µs | within 1% of TE; ABAB pending | `artifacts/k18-mxfp8-2026-10-14/large_unroll/packed_bf16_g512_strided_unroll4/case05/helion-cute/result.json` |

The latest GPU 1 small-shape geometry batch is diagnostic: its unchanged generated-code anchor moved from 1.888 to 1.984 µs while TE held 1.856 µs. No winner was selected from that batch. Raw receipts: `artifacts/k18-mxfp8-2026-10-14/small_native_geometry/`.

### Packed arithmetic and proof experiments

- A hand-edited packed-max-only candidate passed exact eager/graph validation and measured **10.464 µs** on 4096×4096, equal to its original word-kernel anchor. Receipt: `artifacts/k18-mxfp8-2026-10-14/handedit/packed_absmax_v1/case03/helion-cute/result.json`.
- The bounds knob is not currently exposed for these reductions. The ordinary probe rejected it before measurement. CPU generation with admission enabled removes three guard conditions; the bounds proof still controls every simplification. Logs: `artifacts/k18-mxfp8-2026-10-14/proven_bounds/`.
- The combined packed-max/native-BF16-payload candidate is frozen under `artifacts/k18-mxfp8-2026-10-14/handedit/combined_v1/`, now measured below.
- Full profile diagnosis: [NCU report](artifacts/k18-mxfp8-2026-10-14/profile/k18-mid-20261015/REPORT.md).

### Native BF16 arithmetic and compute phases

All 65,536 BF16 encodings passed native E4M3FN/E5M2 conversion at V4/V8/V16 on GPU 2. Trace confirms 12 native helper calls; six clamp tests also pass after fixing `cuda.kernel` argument provenance. Evidence: `artifacts/k18-mxfp8-2026-10-14/compiler_probe/artifacts/native_bf16_probe/native_gpu2_verified2_trace.json`.

| Shape | Trial | Result | Artifact |
| --- | --- | ---: | --- |
| 4096x4096 | combined_v1/g1024_strided | 11.136 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/combined_v1/results/g1024_strided/case03/helion-cute/result.json` |
| 4096x4096 | combined_v1/g256_blocked | 10.432 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/combined_v1/results/g256_blocked/case03/helion-cute/result.json` |
| 4096x4096 | combined_v1/g256_strided | 9.920 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/combined_v1/results/g256_strided/case03/helion-cute/result.json` |
| 4096x4096 | combined_v1/g512_strided | 9.824 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/combined_v1/results/g512_strided/case03/helion-cute/result.json` |
| 4096x4096 | phase_split_v1/g256_bounds | 9.888 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/phase_split_v1/results/g256_bounds/case03/helion-cute/result.json` |
| 4096x4096 | phase_split_v1/g256_phases | 9.920 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/phase_split_v1/results/g256_phases/case03/helion-cute/result.json` |
| 4096x4096 | phase_split_v1/g512_phases | 9.792 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/phase_split_v1/results/g512_phases/case03/helion-cute/result.json` |
| 4096x4096 | phase_split_v1/g512_phases_bounds | 9.728 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/phase_split_v1/results/g512_phases_bounds/case03/helion-cute/result.json` |

This batch reached 9.728 µs. Later mapping probes and the fresh baseline appear below.

### Exact thread mapping and finite scale math

| Shape | Trial | Result | Artifact |
| --- | --- | ---: | --- |
| 4096x4096 | te_mapping_v1/g256_warp | 9.856 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/te_mapping_v1/results/g256_warp/case03/helion-cute/result.json` |
| 4096x4096 | te_mapping_v1/g256_warp_phases | 9.856 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/te_mapping_v1/results/g256_warp_phases/case03/helion-cute/result.json` |
| 4096x4096 | te_mapping_v1/g512_warp_phases | 9.632 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/te_mapping_v1/results/g512_warp_phases/case03/helion-cute/result.json` |
| 4096x4096 | finite_sf_v1/g256_warp | 9.856 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/finite_sf_v1/results/g256_warp/case03/helion-cute/result.json` |
| 4096x4096 | finite_sf_v1/g256_warp_phases | 9.856 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/finite_sf_v1/results/g256_warp_phases/case03/helion-cute/result.json` |
| 4096x4096 | finite_sf_v1/g512_warp_phases | 9.632 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/finite_sf_v1/results/g512_warp_phases/case03/helion-cute/result.json` |

The exact mapping improves the medium shape to **9.632 µs**. Finite BF16 scale simplification has no measurable gain. A fresh five-run TE anchor is **8.608 µs**. The next trial keeps the maximum in packed BF16 through the warp shuffle, matching the remaining TE reduction structure.

### Packed maximum through the warp

The packed BF16 shuffle/fold variants pass exact eager and graph validation. Their timings did not close the remaining medium-shape gap. CPU proof includes 85 cases and matching group32 cardinalities.

| Shape | Trial | Result | Artifact |
| --- | --- | ---: | --- |
| 4096x4096 | g256_te_phases_direct | 9.857 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/packed_warp_v1/results/g256_te_phases_direct/case03/helion-cute/result.json` |
| 4096x4096 | g256_te_phases_staged | 9.888 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/packed_warp_v1/results/g256_te_phases_staged/case03/helion-cute/result.json` |
| 4096x4096 | g512_phases_bounds_direct | 9.824 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/packed_warp_v1/results/g512_phases_bounds_direct/case03/helion-cute/result.json` |
| 4096x4096 | g512_phases_bounds_staged | 9.760 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/packed_warp_v1/results/g512_phases_bounds_staged/case03/helion-cute/result.json` |

Next: compare output allocation and cache behavior under the benchmark's cache flush. The shader's logical output bytes are unchanged; the dirty-cache profile includes differing writeback traffic.

### Output allocation and cache policy

Using identical preallocated buffers in an interleaved ABAB probe gives CuTe9.232µs versusTE8.640µs. This identifies an allocation/cache-placement contribution, with a remaining kernel gap. Raw evidence: `artifacts/k18-mxfp8-2026-10-14/same_buffers/summary.json`.

| Shape | Trial | Result | Artifact |
| --- | --- | ---: | --- |
| 4096x4096 | allocation_v1/anchor | 9.632 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/allocation_v1/results/anchor/case03/helion-cute/result.json` |
| 4096x4096 | allocation_v1/joint_buffer | 9.632 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/allocation_v1/results/joint_buffer/case03/helion-cute/result.json` |
| 4096x4096 | allocation_v1/scales_first | 9.696 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/allocation_v1/results/scales_first/case03/helion-cute/result.json` |
| 4096x4096 | allocation_v1/uint8_view | 9.632 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/allocation_v1/results/uint8_view/case03/helion-cute/result.json` |
| 4096x4096 | l2_retention_v1/g256_warp_phases | 9.216 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/l2_retention_v1/results/g256_warp_phases/case03/helion-cute/result.json` |
| 4096x4096 | l2_retention_v1/g512_warp_phases | 9.184 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/l2_retention_v1/results/g512_warp_phases/case03/helion-cute/result.json` |
| 4096x4096 | cache_pairs_v1/g256_cs_last | 9.216 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/cache_pairs_v1/results/g256_cs_last/case03/helion-cute/result.json` |
| 4096x4096 | cache_pairs_v1/g512_cg_last | 9.184 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/cache_pairs_v1/results/g512_cg_last/case03/helion-cute/result.json` |
| 4096x4096 | cache_pairs_v1/g512_cs_last | 9.184 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/cache_pairs_v1/results/g512_cs_last/case03/helion-cute/result.json` |
| 4096x4096 | cache_pairs_v1/g512_last_anchor | 9.184 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/cache_pairs_v1/results/g512_last_anchor/case03/helion-cute/result.json` |

The exact TE payload L2retention descriptor is `0x14F0000000000000`, already implemented in Helion's gated L2 store helper. It improves the medium shape to9.184µs. The next probe also uses TE's32-byte read-only load with L2evict-first priority.

### Exact cache policies close the medium-shape gap

| Shape | Kernel | Result | Artifact |
| --- | --- | ---: | --- |
| 4096x4096 | te_cache_v1/g256_warp_phases | 8.640 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/te_cache_v1/results/g256_warp_phases/case03/helion-cute/result.json` |
| 4096x4096 | te_cache_v1/g512_warp_phases | 8.640 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/te_cache_v1/results/g512_warp_phases/case03/helion-cute/result.json` |
| 4096x4096 | fp32_cache_v1/g512 | 8.800 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/fp32_cache_v1/results/g512/case03/helion-cute/result.json` |
| 4096x4096 | fp32_cache_v1/g64 | 8.416 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/fp32_cache_v1/results/g64/case03/helion-cute/result.json` |

The indexed FP32 native-scale kernel with64 groups/CTA,64 row threads,2 reduction threads andV16 input packets reaches **8.416 µs**, versus the fresh **8.608 µs** TE anchor (**1.023×**). This requires the32-byte readonly `ld.global.nc.L2::evict_first` input policy and16-byte L2evict-last FP8 output packets. Packed BF16 arithmetic and physicalthread rewrites are unnecessary for this winning configuration. General cache-policy options are being integrated.

### FP32 cache-policy checks on the largest shapes

| Shape | Trial | Result | Artifact |
| --- | --- | ---: | --- |
| 8192x8192 | g512 | 31.840 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/fp32_large_v1/results/g512/case04/helion-cute/result.json` |
| 8192x16384 | g512 | 60.992 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/fp32_large_v1/results/g512/case05/helion-cute/result.json` |
| 8192x8192 | g64 | 31.808 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/fp32_large_v1/results/g64/case04/helion-cute/result.json` |
| 8192x16384 | g64 | 60.352 µs | `artifacts/k18-mxfp8-2026-10-14/handedit/fp32_large_v1/results/g64/case05/helion-cute/result.json` |

The same64-group FP32 configuration reaches **31.808µs** on8192×8192 and **60.352µs** on8192×16384. Allthree large shapes now match or beat their current same-GPU TE anchors within1%. Small-shape anchors and candidate pins are being repeated onGPU2.

### Small shapes moved to GPU 2

| Shape | Triton | FlashInfer | TE | Compiled Torch | Native CuTe pin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1x128 | 1.760 µs | 1.888 µs | 4.480 µs | 1.760 µs | 1.728 µs |
| 137x64 | 1.792 µs | 1.920 µs | 7.776 µs | 1.888 µs | 1.792 µs |
| 128x1024 | 1.920 µs | 1.984 µs | 1.856 µs | 2.688 µs | 1.952 µs |

Raw receipts: `artifacts/k18-mxfp8-2026-10-14/small_gpu2/`. The two smallest cases now pass on GPU2;128×1024 remains below the1% margin. Short geometry/cache probes continue for that case. The16-byte input policy uses `createpolicy` plus the nc cache-hint load; static `L2::evict_first` requires32-byte load width.

### All shapes have a matching pinned prototype

The FP32 128-group configuration for128×1024 reaches **1.856µs** after removing scale cases unreachable for finite RCEIL input. For finite input, even FP32max/448 is below2**120, so the E8M0 scale byte is0..247; native conversion handles zero directly and the reciprocal never needs the byte254 branch. The change is in the isolated K18 kernel source. Raw evidence: `artifacts/k18-mxfp8-2026-10-14/handedit/small_finite_v1/results/case02/helion-cute/result.json`.

The hand-prototype goal checker passes allsix shapes at **1.0069×** geometric mean. Final verification still requires the integrated compiler options, ordinary K18 invocation, and interleaved checks. Checker receipt: `artifacts/k18-mxfp8-2026-10-14/goal_hand_prototypes_01.txt`.

### Public K18 integration and interleaved check

The ordinary AOT invocation uses six saved CuTe configs and forbids search. All six shapes pass exact eager and changed-input CUDA graph checks. First public receipts: `artifacts/k18-mxfp8-2026-10-14/public_all_first/`. The complete case 2 ABAB check is **1.872 µs CuTe vs 1.840 µs TE**, ratio **0.982906×**, so the case remains open. Receipts: `artifacts/k18-mxfp8-2026-10-14/public_case2_abab_v1/summary.json`. Four pinned geometry probes are running for this case.

All finite BF16/FP16 encodings and 103,687 FP32 boundary/random samples passed exact payload and scale bytes, input preservation and changed-input graph replay. Evidence: `artifacts/k18-mxfp8-2026-10-14/final_correctness/summary.json`.

### Resolving the remaining 128×1024 margin

The public kernel uses the edited source and saved config; `public_launch_audit_v1/actual_run.py` records the code actually launched. The CUPTI timer measures the span from the first to last GPU activity in each invocation, including provider adapters.

| Probe | CuTe | Paired TE | Decision | Artifact directory |
| --- | ---: | ---: | --- | --- |
| Four layouts (N1/N2) | 1.920–2.208 µs | 1.824–1.856 µs | rejected | `public_small_geometry_v2` |
| Scale store after payload | 1.888 µs | 1.856 µs | rejected | `public_store_order_v1` |
| BF16 max before FP32 widening | 1.984 µs | 1.824 µs | rejected | `public_bf16_max_v1` |
| Native BF16 pair scaling/conversion | 1.888 µs | 1.856 µs | rejected | `public_native_bf16_payload_v1` |
| Proven bounds (96 → 94 instructions) | 1.888 µs | 1.856 µs | no timing win | `public_fp32_bounds_v1` |
| Balanced FP32 maximum tree | 1.888 µs | 1.856 µs | rejected | `handedit/balanced_max_v1` |
| Linear physical thread layout | 1.888 µs | 1.856 µs | rejected | `handedit/linear_threads_v1` |
| Full finite BF16 transplant | 1.856 µs | 1.856 µs | prototype parity | `handedit/full_bf16_finite_v1` |
| Allocate scales before payload | **1.856 µs** | **1.856 µs** | public candidate; ABAB pending | `public_scales_first_v1` |

All directories are under `artifacts/k18-mxfp8-2026-10-14/`. The final candidate keeps indexed FP32 arithmetic, native finite RCEIL and the general cache policies. Kernel/source prototypes and clamp experiments remain archived. New bounds admission and adjacent tests pass **111 CPU cases**; packet policies/conversion pass **164 CPU cases**, with 18 GPU cases pending. Full lint has the same eight optional-dependency type errors as a clean base; the changed compiler/benchmark files pass Pyrefly, Ruff and format checks.

## Final interleaved result

**GOAL MET:** all six public AOT shapes are at least 0.99× their best matched GPU 2 baseline; geometric mean **1.0051×**. Every call used its saved config, with search forbidden, and passed exact payload/scales, input-preservation and changed-input graph replay. Raw ABAB receipts: `artifacts/k18-mxfp8-2026-10-14/final_abab_v1/`; checker: `artifacts/k18-mxfp8-2026-10-14/goal_final_abab_v1.txt`.

The benchmarked source is frozen under `artifacts/k18-mxfp8-2026-10-14/final_source_snapshot/`; the final worktree snapshot is `final_source_snapshot_v4/`. All benchmarked kernel/compiler files are unchanged. The independent final review is **LGTM**, including bounds admission, version bump, actual AOT/source verification and test reference corrections. Both full suites finished cleanly on GPU 2.

### Default suite verification

Full default backend suite on GPU 2: **17,841 passed, 10,753 skipped, 3,148 subtests passed, 1 failure**, in 48m50s. The one failure is `test_cute_topk.py::test_topk_composition_normalizes_before_selection[64-replicated-softmax]`: one BF16 value had relative error 0.004089 versus tolerance 0.004. The focused full-trace retry **passed** in 9.60s. A separate CPU source comparison found all eight normalization/selection variants byte-for-byte identical to the clean base. This supports an existing unseeded one-ULP tolerance flake; the full run's failed exit is preserved. Evidence: `artifacts/k18-mxfp8-2026-10-14/verification_debug/topk_codegen_current_01/` and `topk_codegen_base_01/`. Logs: `pytest_default_full_01.log` and `pytest_default_topk_retry_01.log`.

### Final lint and CuTe verification

Full `./lint.sh` now **passes**, including Ruff, format, codespell and Pyrefly (0 errors). A separately reviewed five-file optional benchmark compatibility patch resolves the eight errors also present on the clean base. Evidence: `artifacts/k18-mxfp8-2026-10-14/verification_lint/`; root lint log: `lint_final_04.log`.

The first full CuTe suite finished on **GPU 2**: **27,261 passed, 1,325 skipped, 4,648 subtests passed, 2 failures**, in 1h25m25s. Both failures were stale expectations for the new `cute_store_policy` field: the autotuner's allowed key sets and the CuTe reproduction journal. Those expectations are updated and **both focused reruns pass** (5.92s). Final review of the updates is LGTM. Logs: `artifacts/k18-mxfp8-2026-10-14/pytest_cute_full_01.log` and `pytest_cute_expectations_retry_01.log`.

The second default check reproduced the same BF16 rounding failure and captured its original input. It was stopped at 90% before changing source; the CuTe second check did not start. The actual output is identical on current and clean-base compilers on GPU 2. The test now compares `normalized.float()` with the unrounded FP32 reference, retaining `rtol=0.004`, `atol=1e-7`, an explicit BF16 dtype check, exact selection and input preservation. The captured-input check and all eight affected variants **pass**; independent review is LGTM. Evidence: `verification_debug/captured_topk_gpu_comparison_01.json`, `pytest_topk_fp32_reference_01.log`, and `pytest_default_full_02.interrupted.json`.

The `cute-verify` skill requires clean full checks after fixes. Warm full default and CuTe checks finished sequentially on GPU 2. Logs are `pytest_default_full_03.log` and `pytest_cute_full_03.log`. No K18 search was run.

**Both full checks are clean.** Default: 17,842 passed, 10,753 skipped, 3,148 subtests passed, in 45m41s. CuTe: 27,263 passed, 1,325 skipped, 4,648 subtests passed, in 1h15m07s. Neither run had failures.

Full lint after these changes **passes** (`lint_final_06.log`). Final source snapshot `final_source_snapshot_v4/` adds the reference test correction; all 35 earlier sources, including the benchmarked kernel/compiler, are unchanged.

## Completion

| Check | Result |
| --- | --- |
| Performance | All six shapes ≥0.99× best matched GPU 2 baseline; geometric mean 1.0051× |
| Exact correctness | All six public calls, graph replay and exhaustive finite-value checks pass |
| Full default suite | 17,842 passed / 10,753 skipped; no failures |
| Full CuTe suite | 27,263 passed / 1,325 skipped; no failures |
| Lint | Ruff, format, codespell and Pyrefly clean |
| Independent review | LGTM |
| Source integrity | All 36 final snapshot files match the worktree |

Final checker: `artifacts/k18-mxfp8-2026-10-14/goal_final_completion.txt` (**GOAL MET**). Detailed report: [FINAL_REPORT.md](artifacts/k18-mxfp8-2026-10-14/FINAL_REPORT.md). Functional changes are committed locally; raw artifacts remain uncommitted.

## Git cleanup and library integration

The history is organized on the tested Helion base `a03e0f28`:

| Order | Commit | Purpose |
| --- | --- | --- |
| 1 | `171363ad` | Benchmark harness and optional baseline API fixes |
| 2 | `e25ee72f` | FP8/integer packet stores |
| 3 | `91f23191` | Proven load/store cache policies |
| 4 | `b1197a47` | BF16 normalization reference correction |
| 5 | `d5969acd` | Short-row seeds and static bounds admission |
| 6 | Final `[noland]` commit | This progress file and the detailed report |

The packet and policy commit prefixes pass independent CPU checks: **103 passed / 18 GPU skips** and **127 passed**. All backend/source bytes still match the fully verified snapshot. Raw measurements, configs, profiles and the goal file remain ignored under `artifacts/`.

The hill-climbed kernel and configs are committed in `/home/shangdiy/helion-kernel-library` as **`c60b2e5`**. All 27 config keys remain; all nine exact entries (including the six measured presets) match the candidate. Eighteen generic fallback layouts were corrected for small clipped blocks, and other CUDA recipes now use supported operations. GPU 2 integration checks: **242 default tests passed; 242 CuTe tests passed**. Final review is LGTM. Existing unrelated library edits were preserved. No commits were pushed.

The detailed report is [k18_hillclimb_report.md](k18_hillclimb_report.md).
