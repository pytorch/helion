# CuTe top-k hillclimb final report

Report date: 2026-10-12. Rebase update: 2026-10-13.
This document is intentionally in a `[no land]`
commit and is separate from the production changes.

**Retained pinned revalidation:** the measured generic lowering is
**0.986317×** overall against the retained handwritten CuTe configurations.
Five selected-softmax cases miss the 0.99 parity margin; an anchored repeat
confirms **18.69% higher latency** for BF16 65536×64 K16 with softmax.
All 31 shipping cases pass native output checks. This full sweep uses
`fast_math=False`. A subsequent `fast_math=True` comparison reaches within 1%
of the handwritten baseline on N64/K16 and N64/K32, as documented below.
The accepted V18 result remains historical; the fast-math ablation covers two
cases rather than a new complete sweep.

**Historical accepted result (V18):** frozen V18 matches the old handwritten
Helion/CuTe path closely across 31 cases, with geometric-mean speedup
**1.001396×**. The largest remaining regression is **2.12% higher latency**
for BF16 65,536×128, K=8 with selected softmax (12.832 to 13.104 μs).
The user explicitly accepted this result:

> it's ok to be 2.1% slower for now. consider the goal met.

**Version scope:** the V18 table measures frozen V18. Retained pinned
measurements and validation are recorded separately below. The latest rebase
and narrowed finalist-history policy postdate these measurements. Optimization
remains paused at the earlier user-accepted outcome; this verification did
not collect a new cold-full-autotune winner or update the historical goal file.

## Historical versions and measurement protocol

- Old implementation: Helion/CuTe revision `6d8f34d0787033530b5d30d24dd33cd63d212060`,
  source digest `9d407052d4b821078ed27bf4ce27e1a4570365fb6582e74afa36989e76650312`. It calls handwritten
  local/distributed or coarse-selection runtime helpers.
- Measured optimized implementation: V18 source digest `e0fb820fd62e012fd7e62efd2dbe33123f6cce27ab155bc41eb22ac56f8b140f`.
- Published production code: `5d2c8e261f9a977592a3c780ba6b69d8a94ccd02`, production tree `992979d9bf31318a66027dcb81c792e232cb508b`;
  based on remote main `04f25d2a76ccee0d5f993e961ced7a1770473df2`.
- Measured folded production snapshot, before the later rebase and finalist
  scope change: `ebdfd28a88154843d74d3cb0e89c2d750c80360a`, tree `47c82b235c60a543415f942e886ac4f49b3f63b2`;
  verified production source `203dda5836db9cd1bdf9177d3801b06c589582f033ec34a790b202cb2762685c`.
- Hardware/environment: NVIDIA GB300, **1400 W** power cap; frozen
  Torch 2.14.0, Triton 3.8.0 and CUTLASS DSL 4.7.0 environment.

In the historical V18 table, the before column is the old implementation
remeasured at its retained
configuration in the final paired protocol. It is not a quotation of the
original Step 1 cold-search observation. The after column uses the matching
V18 worker. V18 winners were obtained by cold full autotuning; configurations
and source snapshots were frozen before final comparisons.

Each reference/candidate pair shares a physical GPU, allocation, input seed
and timer. Separate processes avoid binding-cache interference. Two rounds
produce ten report medians per implementation; each report median covers
100 CUDA-event samples of pre-captured graph replay, with an L2 clear before
the event pair. The displayed latency is the median of those ten medians.
Per-case speedup is the minimum paired baseline/V18 ratio across all retained
references; all 31 limiting references are the old CuTe implementation.

Evidence comprises 121 reference pairs (55 old CuTe, 35 Helion/Triton and
31 Torch), 484 worker results and 2,420 report medians. CUDA-event intervals
exclude host launch, compilation and L2-clear cost. Remote workers used a
40°C start target and qualified local workers 48°C. These controls do not
establish a statistical confidence interval or guarantee constant clocks.

## Retained pinned performance revalidation

These measurements cover the named pre-rebase snapshots below. No new
performance result was collected for the rebased stack or the narrowed
finalist-history policy; pinned execution bypasses that search policy.

The retained measurements cover all 31 shipping cases against all 55 retained old
CuTe configurations: 220 qualified workers and 1,100 stored report medians.
Every worker passed direct execution, poisoned graph replay, unchanged-input
and tie-valid selection checks. The geometric mean is **0.986317×**;
five selected-softmax cases fall below the 0.99 parity margin. The worst
is BF16 65536×64 K16 with softmax, **10.760 → 12.800 μs**, or
**18.96% higher latency**. These current results do not meet the original
strict performance criterion or establish the accepted V18 outcome for the
current generic epilogue.

These are pinned shipping-config measurements, with no new cold autotuning.
They use the retained event/graph/cold-L2 timer, separate ABAB processes,
2 rounds × 5 medians × 100 events, seed 20261008, a 48°C cooldown ceiling
and 1400 W on GB300 `GPU-f07565f4-646c-24a4-8ecd-513648cf1ffe`.
Torch 2.14.0, Triton 3.8.0 and CUTLASS DSL 4.7.0 match the historical environment.
Individual event samples and a continuous clock/exclusivity trace were not
stored; seven worker spread warnings are retained in the raw audit.

The measured compiler is production `5d2c8e26` plus the explicit sum-dtype
correction, source `e1a72a0579eaed99e98c4cfa190130b52bec594b6bdc729a68ba3104b96219ae`.
The final review fixes additionally change five unrelated sampling AOT modules
(source `203dda5836db9cd1bdf9177d3801b06c589582f033ec34a790b202cb2762685c`).
All 600 compiler/runtime/top-k/MoE Python files are byte-identical between
the measured snapshot and that revised tree; shipping top-k/MoE pins are
unchanged. This equivalence covers the top-k sweep, not the corrected sampling
configurations. See [source equivalence](../artifacts/topk-final-verification-20261012-v1/performance/v1-v2-topk-source-equivalence.json).

Each row uses the minimum paired ratio across its retained references and
displays the baseline and candidate latencies from that same pair. No timings
are joined across pairs or GPUs. The C identifier resolves to the matching
`rows` entry in the [current evidence map](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json), including pins and all
four raw worker paths and hashes.

| Case | Old CuTe μs | Current μs | Paired speedup | Input GB/s old | Input GB/s current | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MoE FP32 32768×256 K8 | 25.536 | 25.160 | 1.014944× | 1314.0 | 1333.6 | [C01](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-000) |
| BF16 65536×64 K8 | 10.784 | 10.792 | 0.999259× | 777.9 | 777.3 | [C02](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-002) |
| BF16 65536×64 K8 + softmax | 10.784 | 10.784 | 1.000000× | 777.9 | 777.9 | [C03](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-004) |
| BF16 65536×64 K16 | 10.784 | 10.784 | 1.000000× | 777.9 | 777.9 | [C04](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-007) |
| BF16 65536×64 K16 + softmax | 10.760 | 12.800 | 0.840625× | 779.6 | 655.4 | [C05](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-011) |
| BF16 65536×64 K32 | 12.832 | 12.832 | 1.000000× | 653.7 | 653.7 | [C06](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-012) |
| BF16 65536×64 K32 + softmax | 12.832 | 14.880 | 0.862366× | 653.7 | 563.8 | [C07](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-015) |
| BF16 65536×128 K8 | 12.848 | 12.832 | 1.001247× | 1305.8 | 1307.5 | [C08](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-019) |
| BF16 65536×128 K8 + softmax | 12.832 | 13.216 | 0.970944× | 1307.5 | 1269.5 | [C09](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-020) |
| BF16 65536×128 K16 | 14.872 | 14.872 | 1.000000× | 1128.1 | 1128.1 | [C10](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-023) |
| BF16 65536×128 K16 + softmax | 14.920 | 15.360 | 0.971354× | 1124.5 | 1092.3 | [C11](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-025) |
| BF16 65536×128 K32 | 18.976 | 18.976 | 1.000000× | 884.1 | 884.1 | [C12](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-026) |
| BF16 65536×128 K32 + softmax | 18.944 | 19.000 | 0.997053× | 885.6 | 883.0 | [C13](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-028) |
| BF16 65536×256 K8 | 19.008 | 18.976 | 1.001686× | 1765.3 | 1768.3 | [C14](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-029) |
| BF16 65536×256 K8 + softmax | 19.392 | 19.128 | 1.013802× | 1730.3 | 1754.2 | [C15](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-033) |
| BF16 65536×256 K16 | 23.072 | 23.104 | 0.998615× | 1454.3 | 1452.3 | [C16](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-034) |
| BF16 65536×256 K16 + softmax | 23.056 | 23.072 | 0.999307× | 1455.3 | 1454.3 | [C17](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-035) |
| BF16 65536×256 K32 | 27.584 | 27.440 | 1.005248× | 1216.4 | 1222.8 | [C18](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-036) |
| BF16 65536×256 K32 + softmax | 27.584 | 29.232 | 0.943623× | 1216.4 | 1147.9 | [C19](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-038) |
| BF16 65536×512 K8 | 31.264 | 31.264 | 1.000000× | 2146.5 | 2146.5 | [C20](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-039) |
| BF16 65536×512 K8 + softmax | 31.264 | 31.328 | 0.997957× | 2146.5 | 2142.1 | [C21](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-042) |
| BF16 65536×512 K16 | 39.456 | 39.448 | 1.000203× | 1700.9 | 1701.2 | [C22](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-043) |
| BF16 65536×512 K16 + softmax | 39.456 | 39.440 | 1.000406× | 1700.9 | 1701.5 | [C23](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-044) |
| BF16 65536×512 K32 | 47.632 | 47.664 | 0.999329× | 1408.9 | 1408.0 | [C24](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-045) |
| BF16 65536×512 K32 + softmax | 47.648 | 48.032 | 0.992005× | 1408.4 | 1397.2 | [C25](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-046) |
| BF16 65536×1024 K8 | 53.784 | 53.800 | 0.999703× | 2495.5 | 2494.8 | [C26](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-048) |
| BF16 65536×1024 K8 + softmax | 53.792 | 54.176 | 0.992912× | 2495.1 | 2477.4 | [C27](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-049) |
| BF16 65536×1024 K16 | 68.160 | 68.168 | 0.999883× | 1969.2 | 1968.9 | [C28](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-050) |
| BF16 65536×1024 K16 + softmax | 68.144 | 68.136 | 1.000117× | 1969.6 | 1969.9 | [C29](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-052) |
| BF16 65536×1024 K32 | 86.944 | 86.736 | 1.002398× | 1543.7 | 1547.4 | [C30](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-053) |
| BF16 65536×1024 K32 + softmax | 86.608 | 87.008 | 0.995403× | 1549.7 | 1542.6 | [C31](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json) (retained-054) |
| **Summary** | **31.663** | **31.885** | **0.986317×** | **1411.1** | **1400.0** | 31 cases |

Summary latencies and input bandwidths use arithmetic means; ratios use the
geometric mean. Group ratios are **1.000503×** for plain top-k, **0.970480×**
for selected softmax and **1.014944×** for full-row MoE.

A separate anchored repetition confirmed the slower epilogue. Plain N64/K16
anchors changed −0.074% for the baseline and −0.223% for current, both within
1%. All 32 repeated workers qualified. Each softmax repetition has four
ABAB rounds and 20 stored medians per implementation; anchors use two rounds.
The full 31-case table above remains the original complete pinned sweep.

| Softmax case | Initial paired ratio | Anchored repeat ratio | Repeat latency change |
| --- | ---: | ---: | ---: |
| BF16 65536×64 K16 | 0.840625× | 0.842500× | +18.69% |
| BF16 65536×64 K32 | 0.862366× | 0.862366× | +15.96% |
| BF16 65536×128 K16 | 0.971354× | 0.972353× | +2.84% |

Broad median ranges remain for N64/K16 and N128/K16 (about 10%), while
all four round ratios show slower current kernels. Samples were not dropped
or merged into the 31-case geometric mean. See [repeat audit](../artifacts/topk-final-verification-20261012-v1/performance/evidence/anchored-repeat-audit-v1-1791836051557635591.json), SHA-256
`9c8071b2642d7f642b255f29f741aa967f346cf3777a8a3ff83ab70e9796c2a9`;
[independent verification](../artifacts/topk-final-verification-20261012-v1/performance/evidence/anchored-repeat-audit-v1-1791836051557635591-verified-1791836078131984124.json)
checks all 2,844 raw/source inputs.

Cached machine code for N64/K16 softmax contains eight reciprocals and 40
FMA instructions in the accurate-division sequences of the current epilogue, versus one reciprocal and
no refinement FMAs in the old baseline. There are no spills, and current
stores remain vectorized. The baseline uses unconditional approximate
reciprocal and fast-math exp; current defaults retain accurate division and
exp denormal handling. Reusing a replicated denominator under the existing
`fast_math` policy is confirmed by the ablation below: the existing generic
lowering shares the reciprocal when that policy permits it. Production
kernel defaults were not changed by this verification. See [cached-code diagnosis](../artifacts/topk-final-verification-20261012-v1/review/perf-n64k16-source-v1/diagnosis.md).

## Fast-math selected-softmax comparison

The old handwritten selected-softmax helper unconditionally uses approximate
reciprocal and fast-math exponentiation. Current generic lowering uses accurate
FP32 division and exponentiation handling by default. Enabling the existing
`fast_math=True` kernel setting permits approximate/FTZ FP32 division and
fast exponentiation, while keeping the generic Helion kernel and existing
configuration pins. No production kernel default was changed by this ablation.

The two N64 selected-softmax cases show lower GPU latency with fast math.
Each default/fast pair uses identical current pins. A separate pair compares
the retained handwritten pin against current fast math; its absolute current
latency can differ from the first pair, so timings are not joined across pairs.

| BF16 case | Current default → fast (μs) | Default / fast | Old handwritten → current fast (μs) | Old / current fast |
| --- | ---: | ---: | ---: | ---: |
| 65536×64 K16 + softmax | 12.768 → 10.848 | 1.176991× | 10.752 → 10.816 | 0.994083× |
| 65536×64 K32 + softmax | 14.888 → 12.832 | 1.160224× | 12.800 → 12.832 | 0.997506× |

Fast math reduces default-mode latency by **15.04% / 13.81%**. Against the
handwritten baseline, current fast mode has **0.595% / 0.250%** higher latency,
within the 1% noise margin on these two cases, rather than a measured win.

All 16 workers in four separate-process ABAB pairs qualify, with direct
execution, poisoned replay, input preservation, tie-valid selection and
shape/dtype checks unchanged. The experiment uses GPU 2
`GPU-d178336e-85b3-e1bf-bbfc-d8f2e2054b65`, 1400 W, a 48°C cooldown ceiling,
seed 20261008, two rounds × five medians × 100 captured-graph cold-L2 events.
Package versions and sources match the main verification. Requested/effective
math settings are recorded; the copied harness only adds math-setting plumbing.

Cached machine code confirms the mechanism. The selection plans are identical
between modes. On the main kernel path, fast math changes K16/K32 reciprocal
counts **8/16 → 1**, accurate-division sequence FMA counts **40/80 → 0**, and
exponentiation denormal guards **8/16 → 0**. The EX2 counts remain 8/16.
The FMA counts include 32/64 strict refinement operations and 8/16 product
operations; an additional 12 FMA instructions in the default slow-division
helper are outside that main-path count and need not execute. This is static
cached assembly inspection, rather than a partition of measured cycles.
See [cached assembly evidence](../artifacts/topk-fastmath-ablation-20261012-v1/sass-inspection/summary.json).

This is a pinned math-mode comparison, with no new cold autotuning or full
31-case fast-mode sweep. Fast math relaxes floating-point policy and can change
BF16 rounding/subnormal results; passing the retained finite-input tolerance
checks does not prove bitwise probability identity. The full default-mode
results above and historical V18 acceptance are preserved. See [fast-math report](../artifacts/topk-fastmath-ablation-20261012-v1/report.md),
[raw summary](../artifacts/topk-fastmath-ablation-20261012-v1/summary.json),
[qualification](../artifacts/topk-fastmath-ablation-20261012-v1/qualification.json), and
[effective settings](../artifacts/topk-fastmath-ablation-20261012-v1/math-settings-proof.json).

## Controlled key-encoding comparisons

The only changed config field is `cute_topk_key_dtype`, with int32 as control.
Each comparison uses separate current-compiler processes on GB300
`GPU-a58c8148-8e58-d9ae-d97f-3cd7322d9e02` with the same paired protocol.
All 64 workers pass native output checks and match independent generated-AST
proofs. Twenty-four requested configurations produce 22 distinct kernels:
numeric `float32` aliases int32 for N1024/K8, with and without softmax.
Only the 16 distinct alternatives were timed.

Ratios below are int32 time / alternative time; greater than one is faster.

| BF16 carrier | Numeric float32 | float32_bits | float32_native |
| --- | ---: | ---: | ---: |
| 65536×64 K32 | 0.995729× | 1.002451× | 0.825253× |
| 65536×64 K32 + softmax | 0.999472× | 0.998944× | 0.883699× |
| 65536×256 K16 | 0.944830× | 0.999334× | 0.761929× |
| 65536×256 K16 + softmax | 0.946738× | 1.001333× | 0.761228× |
| 65536×1024 K8 | int32 alias; not timed | 0.999556× | 0.653962× |
| 65536×1024 K8 + softmax | int32 alias; not timed | 0.999287× | 0.661254× |

Float-bit keys are within about 0.25% of int32 on these carriers. Numeric
float32 is about 5.6–5.8% slower at N256/K16; native float keys are slower on
all six carriers. This audit finds no independent benefit for those two
alternatives here and does not establish a universally best key mode.
Absolute timings from the two GPUs are not compared. Per-pair times, raw
workers and proof hashes are in the current evidence map and
[encoding proof](../artifacts/topk-final-verification-20261012-v1/performance/key-dtype-proof.json).

## Measured V18 results

Latencies are μs. Input GB/s is `rows × width × dtype_bytes / latency_us /
1000`; it is an input-size normalization, not measured physical memory
traffic. Softmax on BF16 top-k normalizes the selected K values; the FP32
MoE case normalizes the full expert row before selecting K weights.

The Helion/Triton column is its fastest qualified retained pin; Torch uses
its retained reference. These auxiliary baseline times belong to their own
matched V18 workers, which can differ from the displayed CuTe pair. Their
paired candidate times and ratios are recorded in the evidence map.

| Case | Old Helion/CuTe before | V18 after | Paired speedup | Input GB/s before | Input GB/s after | Helion/Triton baseline | Torch baseline | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MoE FP32 32768×256 K8 | 27.360 | 26.056 | 1.050046× | 1226.4 | 1287.8 | 14027.392 | 346.776 | [E01](#e01) |
| BF16 65536×64 K8 | 10.936 | 11.032 | 0.991298× | 767.1 | 760.4 | 357.472 | 220.648 | [E02](#e02) |
| BF16 65536×64 K8 + softmax | 10.784 | 10.816 | 0.997041× | 777.9 | 775.6 | 115.264 | 228.288 | [E03](#e03) |
| BF16 65536×64 K16 | 10.784 | 10.816 | 0.997041× | 777.9 | 775.6 | 353.360 | 223.920 | [E04](#e04) |
| BF16 65536×64 K16 + softmax | 10.816 | 10.832 | 0.998523× | 775.6 | 774.4 | 111.152 | 232.688 | [E05](#e05) |
| BF16 65536×64 K32 | 12.800 | 12.848 | 0.996264× | 655.4 | 652.9 | 385.536 | 219.616 | [E06](#e06) |
| BF16 65536×64 K32 + softmax | 12.832 | 12.832 | 1.000000× | 653.7 | 653.7 | 111.120 | 233.984 | [E07](#e07) |
| BF16 65536×128 K8 | 13.120 | 13.064 | 1.004287× | 1278.8 | 1284.2 | 3261.936 | 363.440 | [E08](#e08) |
| BF16 65536×128 K8 + softmax | 12.832 | 13.104 | 0.979243× | 1307.5 | 1280.3 | 647.648 | 371.232 | [E09](#e09) |
| BF16 65536×128 K16 | 14.920 | 14.920 | 1.000000× | 1124.5 | 1124.5 | 3677.464 | 375.296 | [E10](#e10) |
| BF16 65536×128 K16 + softmax | 15.016 | 14.888 | 1.008598× | 1117.3 | 1126.9 | 647.888 | 383.928 | [E11](#e11) |
| BF16 65536×128 K32 | 18.976 | 18.976 | 1.000000× | 884.1 | 884.1 | 3597.792 | 383.472 | [E12](#e12) |
| BF16 65536×128 K32 + softmax | 19.240 | 19.232 | 1.000416× | 872.0 | 872.4 | 582.152 | 398.088 | [E13](#e13) |
| BF16 65536×256 K8 | 19.072 | 19.008 | 1.003367× | 1759.4 | 1765.3 | 41991.000 | 686.592 | [E14](#e14) |
| BF16 65536×256 K8 + softmax | 21.232 | 21.112 | 1.005684× | 1580.4 | 1589.4 | 7006.576 | 695.016 | [E15](#e15) |
| BF16 65536×256 K16 | 23.072 | 23.128 | 0.997579× | 1454.3 | 1450.8 | 55290.888 | 695.648 | [E16](#e16) |
| BF16 65536×256 K16 + softmax | 23.080 | 23.040 | 1.001736× | 1453.8 | 1456.4 | 7815.632 | 705.056 | [E17](#e17) |
| BF16 65536×256 K32 | 28.032 | 27.584 | 1.016241× | 1197.0 | 1216.4 | 45289.312 | 715.288 | [E18](#e18) |
| BF16 65536×256 K32 + softmax | 29.424 | 29.472 | 0.998371× | 1140.4 | 1138.5 | 10894.880 | 730.264 | [E19](#e19) |
| BF16 65536×512 K8 | 31.312 | 31.280 | 1.001023× | 2143.2 | 2145.4 | 367026.657 | 932.832 | [E20](#e20) |
| BF16 65536×512 K8 + softmax | 31.304 | 31.296 | 1.000256× | 2143.8 | 2144.3 | 342130.379 | 952.592 | [E21](#e21) |
| BF16 65536×512 K16 | 39.456 | 39.464 | 0.999797× | 1700.9 | 1700.5 | 344411.964 | 942.560 | [E22](#e22) |
| BF16 65536×512 K16 + softmax | 39.456 | 39.456 | 1.000000× | 1700.9 | 1700.9 | 377892.204 | 940.800 | [E23](#e23) |
| BF16 65536×512 K32 | 47.648 | 47.664 | 0.999664× | 1408.4 | 1408.0 | N/A | 943.584 | [E24](#e24) |
| BF16 65536×512 K32 + softmax | 47.656 | 47.664 | 0.999832× | 1408.2 | 1408.0 | N/A | 960.000 | [E25](#e25) |
| BF16 65536×1024 K8 | 54.048 | 54.064 | 0.999704× | 2483.3 | 2482.6 | N/A | 1135.040 | [E26](#e26) |
| BF16 65536×1024 K8 + softmax | 53.792 | 53.800 | 0.999851× | 2495.1 | 2494.8 | N/A | 1158.656 | [E27](#e27) |
| BF16 65536×1024 K16 | 68.176 | 68.160 | 1.000235× | 1968.7 | 1969.2 | N/A | 1144.568 | [E28](#e28) |
| BF16 65536×1024 K16 + softmax | 68.136 | 68.096 | 1.000587× | 1969.9 | 1971.0 | N/A | 1152.960 | [E29](#e29) |
| BF16 65536×1024 K32 | 87.064 | 86.944 | 1.001380× | 1541.6 | 1543.7 | N/A | 1154.240 | [E30](#e30) |
| BF16 65536×1024 K32 + softmax | 86.656 | 86.928 | 0.996871× | 1548.9 | 1544.0 | N/A | 1168.640 | [E31](#e31) |
| **Summary** | **31.904** | **31.857** | **1.001396×** | **1397.3** | **1399.4** | **70766.333 (n=23)** | **670.829** | 31 cases |

Summary latency/throughput values are arithmetic means; speedup uses the
geometric mean. Eight Triton entries have no qualified retained result:
width 512/K32 with and without softmax, and all six width 1024 variants.
N/A does not imply universal unsupported behavior or infinite speedup.
The largest measured gain is FP32 MoE, **1.050046×**.

## Goal checker and acceptance

The original skill criterion remains a paired ratio of at least 0.99 for
every case and a geometric mean of at least 1.0. **30 of 31 cases pass the
per-case bar; the geometric mean passes. The strict goal was not met.**
The user accepted the remaining 2.1% regression and ended optimization.
The goal file records that acceptance as a paused outcome.

Checker output excerpt, evaluated against the retained artifacts:

```text
status: paused
FAIL topk-bfloat16-65536x128-k8-softmax: ratio 0.979 < bar 0.990
PASS geomean: 1.0014 (min 1.0)
GOAL NOT MET — PAUSED at the user's instruction (the stop gate allows ending the turn)
```

The human-readable checker returned **1**. The stop gate returned **0**
because the user-accepted paused reason is present, with no output. This
is an accepted exception, not a strict `GOAL MET` result.
Complete output: [checker-human.txt](../artifacts/topk-final-report-20261012-v1/checker-human.txt); gate receipt: [checker.json](../artifacts/topk-final-report-20261012-v1/checker.json).

## Backend and history changes

- Sorting, merging and guarded coarse selection are tensor programs lowered
  through shared register planning, permutation routing, scalar lowering
  and elimination of unused values. Complete handwritten top-k runtime
  implementations were removed.
- Ordinary row inputs, reductions and output expressions use ownership,
  bounds and disjoint-I/O proofs. Selected softmax now uses generic
  pointwise/reduction lowering and the known maximum at a sorted endpoint.
- Scalar lowering reuses scoped Inductor graph/kernel handlers while retaining
  dtype rounding and position-preserving input checks. Unsafe partial loops
  and effectful programs retain fallback or rejection behavior.
- General configuration/seed choices expose layout, encoding, vectorization
  and register-budget tradeoffs. Compiler-qualified CuTe top-k/sort searches
  refresh finalist history after rebenchmarking. Existing flash-search behavior
  is preserved; other searches retain their previous historical-minimum ranking.
- Runtime code is limited to reusable device/IR bridges; nine internal DSL
  decorators were removed. Sorting tables and their license live in the
  compiler. The benchmark driver stays in ignored local artifacts.
- Pretuned recipes include 30 BF16 top-k modes plus FP32 MoE, as well as
  sampling recipes. `topk(x, k, softmax=False)` keeps softmax optional.
  Policy-qualified AOT filenames use upstream discovery.
- Backend lowering and its CI corrections are now merged in remote main as
  PR #3915. The remaining local production history has two logical commits,
  with every changed file owned by one commit. Duplicate #3816 and #3915
  changes are inherited from main; the remaining CI corrections stay folded
  into their owners.

| Production commit | Scope |
| --- | --- |
| `c0dc004d` | CuTe top-k/sort finalist-history refresh |
| `5699e5f0` | Pretuned recipes and related CI corrections |

The stack was rebased from `04f25d2a76ccee0d5f993e961ced7a1770473df2` onto freshly fetched remote
main `279e851e124850c5cb402007a60f18e127911844` on 2026-10-13. The earlier statement that it was
already based on `c3fe06e3` was inaccurate: that update had only added stack
metadata. This actual rebase completed without conflicts and dropped the
backend commit `65e74854`, whose patch is already merged as `279e851e`
(PR #3915). The scoped autotuner and recipe patches and their stack-info
trailers are unchanged. The final report remains the last `[no land]` commit.
The finalist policy extension now requires `backend_name == "cute"` and the
compiler's existing `cute_topk_search_enabled` capability. This includes
plain top-k, selected softmax, FP32 MoE and compatible full-row sorts using
the same selection lowering. Generic CuTe programs and alias fallback retain
the original policy; the preexisting flash policy remains unchanged. No new
kernel-name detector, configuration knob, benchmark phase or timing budget
was added. This restriction limits the rollout; it does not prove that the
latest-timing policy improves every eligible selection kernel.

Before this rebase, the focused CPU history checks passed **7 tests and 18 subtests**, covering both
retention policies, reinsertion, pruning, backend guards, pinned configurations
and snapshot isolation. The longest complete test case is 0.003 seconds.
See [focused test summary](../artifacts/topk-finalist-scope-20261012-v1/focused-tests-summary.json)
and [scope review](../artifacts/topk-finalist-scope-20261012-v1/review.md).
Eight additional real CPU bindings verify the feature and policy together:
plain top-k, selected softmax, FP32 full-row softmax followed by top-k,
full-row sort, ordinary softmax, ordinary row sum, alias fallback, and top-k
with seed heuristics disabled. All pass; the longest binding case is 0.409
seconds. Native-compilation and CUDA-initialization call counts are zero.
See [CPU binding proof](../artifacts/topk-finalist-scope-20261012-v1/cpu-binding-proof/results.json)
and [updated fold receipt](../artifacts/topk-finalist-scope-20261012-v1/folded.json).
No full test suite or new performance sweep was run for this restriction.

The final report is the only addition in the last `[no land]` commit.
Verification dtype and test-fixture fixes are included in the merged backend;
the five sampling configuration fixes remain folded into the pretuned-kernel
commit. This rewritten stack has not been pushed. No production code
references this document or requires its artifacts.

After the 2026-10-13 rebase, the focused checks pass **30 tests and 18
subtests**: seven finalist-history tests, four dead-lane-loop AST tests, six
top-k source-generation cases, one MoE source-generation case, five top-k
registration/input-key cases and seven policy-discovery cases. The longest
complete test case is 3.973 seconds. The eight real CPU binding checks also
pass, with a maximum case time of 0.413 seconds. Native compilation and CUDA
initialization guards record zero calls. Ruff checks and formatting pass.
These checks cover CPU integration; no full suite or new kernel-performance
measurement was collected after the rebase. See [rebase checks](../artifacts/topk-rebase-remote-main-20261013-v1/checks.json),
[upstream review](../artifacts/topk-rebase-remote-main-20261013-v1/upstream-review.md)
and [rebase receipt](../artifacts/topk-rebase-remote-main-20261013-v1/rebased.json).

## Retained validation

The retained CI audit covers previously published PR head
`fff8b4eec415b6f68b620ddea68528f397692aa4`, whose only difference from
production `5d2c8e26` was the report at that revision. GitHub run
[38290334900](https://github.com/pytorch/helion/actions/runs/38290334900)
passed every non-TPU matrix and notebook job. That head's separate
lint, documentation and CodeQL checks also passed. The CuTe B200 suite reports **25,507 passed / 1,301 skipped**;
Triton B200 reports **11,063 / 12,033**; AMD reports **10,727 / 12,369**;
XPU reports **10,384 / 12,712**. The old AMD worker exits and XPU shutdown did
not recur. Attempt 2 retried only TPU and carried forward successful job logs.

TPU failed before pytest in that run: the logs report GCP project
`ml-velocity-actions-testing` suspended (`CONSUMER_SUSPENDED`). This was
provisioning infrastructure, not a test assertion. Those CI results cover the
folded published fixes; they predate the local verification fixes and this
rebase. The remaining stack requires fresh CI after publication. See [CI audit](../artifacts/topk-final-verification-20261012-v1/ci/ci-audit.json).

Local full suites run sequentially on the frozen row-corrected snapshot
`e1a72a05…`, on GB300 GPU 2, with no `HELION_AUTOTUNE_EFFORT` override:
`pytest test/ -n4 -q --tb=line --timeout=180 --timeout-method=thread` plus
JUnit output; the second run adds `HELION_BACKEND=cute`. The environment is
Python 3.12.15, Torch 2.14.0+cu130, Triton 3.8.0 and CUTLASS DSL 4.7.0.

| Local full suite | Parent passed | Terminal failed | Terminal errors | Skipped | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 15,843 | 11 | 27 | 10,208 | 3,358.14 |
| CuTe | 24,751 | 17 | 27 | 1,287 | 5,228.35 |

These are raw suite outcomes, not counts after substituting focused reruns.
The default suite additionally reports 2,743 passed subtests; its JUnit
parent testcase count is 26,089, while its `tests` attribute includes those
subtests. The raw nonzero exit codes and all failure evidence are retained.
The CuTe suite additionally reports 4,195 passed subtests. Its 26,082 JUnit
parent testcase elements contain 15 failures and 29 errors: the two worklist
worker exits are setup errors in XML but terminal failures in the table.
The JUnit `tests` attribute is 30,277 including passed subtests. This explains
the category difference without changing the raw adverse-outcome count.

The ten sampling failures and 27 import/setup errors have the focused
corrections below. Seven additional CuTe cases required final reconciliation:

- One recurrence benchmark-body test had the same namespace import collision;
  its exact corrected-environment node passes in 7.07 seconds.
- Four paired-CLC cases required SM100, while the test admitted all major-10
  GPUs. The existing backend gate is exactly `(10, 0)` on both main and this
  stack. The test guard now matches it; eight CPU controls preserve SM100
  carriers and SM103 rejection, and the four exact GB300 nodes skip specifically
  for SM100 with zero binding/compilation calls. The numerical assertions and
  B200 coverage remain unchanged. See [paired-CLC ledger](../artifacts/topk-final-verification-20261012-v1/tests/failure-audit/paired-clc-final/final-ledger.json).
- Two worklist tests ended as crashed workers. Private current/main probes
  reproduce unexpected full autotuning after the B200 hardware mock ends:
  the named promotion key changes, the call rebinds, and the pinned config is
  lost. All four original probes exit under a 50-second pytest timeout.
  Keeping the existing mock through pinned execution and unchanged numerical
  assertions fixes this fixture defect. The final two native nodes pass in
  4.715/5.112 seconds, and the side-store rejection control passes in 2.437
  seconds, with no autotuning logged. The exact original full-run worker-exit
  cause remains unproven because no assertion traceback was retained. See
  [worklist recovery](../artifacts/topk-final-verification-20261012-v1/tests/failure-audit/full-suite-new-v1/native-gpu2-v5/results.json).

No additional stack implementation defect was identified in these seven
cases. The source/test changes were checked with focused reruns; the original
full-suite failures remain in [full-suite audit](../artifacts/topk-final-verification-20261012-v1/tests/full-suite-final-audit.json).

The new corrections and focused recovery are:

- Explicit reduced-precision sums now cast each input to the requested dtype
  before widening for accumulation. Six CPU semantic cases and six native
  top-k/sum compositions pass; the slowest native call is 2.51 seconds.
  Ordinary FP32 selected-softmax source is byte-identical across this fix.
- Eleven sampling phase configs now explicitly select `pid_type="flat"`;
  varlen top-k changes one reduction-thread coordinate from 8 to 32. This
  preserves ordered-phase/layout guards. The varlen source exactly matches
  the intended original layout with the diagnostic early guard deferred.
  All ten previously failing GB300 AOT cases pass natively (12.18–27.57 s).
- The benchmark fixture now explicitly targets SM100, matching its declared
  B200 corpus. The prior implicit SM103 target produces 103 configurations
  on both main and this stack; SM100 produces the expected 100 and preserves
  RNG state. The original GPU-visible main/current failure was reproduced
  before this one-line test correction.
- The installed kernel-library editable path shadows Helion's namespace
  `benchmarks` package. A task-local runner shim restricts imports and removes
  only that external path in child processes. Production packaging and the
  shared venv are unchanged. All 17 grouped GEMM fixture checks pass on both
  main and the row-corrected snapshot under CPU/native guards.
- The original GDN failure is in Torch CUDA SVD setup, before the Helion
  kernel. The exact fresh test passes in 3.96 seconds without a loader
  workaround; the original worker-specific lookup cause remains unproven.

The ten formerly uncollected modules were run on revised v2: default
**613 passed / 3 failed / 140 skipped**, CuTe **753 / 3 / 0**. Their three
remaining fixture/child-import failures were fixed and rerun on v3:
**501 passed per backend**, covering all 499 benchmark tests and both child
CLI cases. The exact node-ID union gives **616 passed / 140 skipped** for
default and **756 passed** for CuTe. These combined v2/v3 outcomes are not a
new 756-case or full-suite run. See [recovery ledger](../artifacts/topk-final-verification-20261012-v1/tests/failure-audit/final-ledger.json)
and [diagnosis](../artifacts/topk-final-verification-20261012-v1/tests/failure-audit/diagnosis.md).

The final production/test/report review is **LGTM**, covering all 79 changed
code/test paths, the latest fixture fixes, report arithmetic and retained
evidence. See [final review](../artifacts/topk-final-verification-20261012-v1/review/final-report-review.md).
The [earlier v3 review](../artifacts/topk-final-verification-20261012-v1/review/final-production-review-v3.md)
independently checked 316 raw performance workers.
Ruff, formatting, device lint and codespell pass. Configured full Pyrefly
retains **49 diagnostics identical to main**, with none in changed included
code; compiler/new row-reduction tests have zero targeted errors. Tests and
AOT modules are excluded by the repository's Pyrefly configuration. An
explicit forced check of the edited benchmark test returns the same 180
existing diagnostics as main, none on the edited fixture line. See
[full type attribution](../artifacts/topk-final-verification-20261012-v1/lint/pyrefly-main-attribution.json)
and [forced test attribution](../artifacts/topk-final-verification-20261012-v1/lint/final-v3-forced-benchmark-attribution.json).

## Broader correctness and compilation coverage

The CPU audit covers 16 ordinary FP32/BF16 row cases, eight additional
complete-row programs and four guards. Fourteen ordinary sources are
byte-identical to main; the two BF16 argextrema sources add packed keys.
The three-round median bind-plus-sourcegen geometric-mean ratio is
**1.007662×**, with a worst **0.135930 → 0.152821 s**. This audit times CPU
tracing/source generation rather than native compilation. New row-program
median totals remain at or below **0.496855 s**. See [CPU coverage](../artifacts/topk-final-verification-20261012-v1/compile_coverage/report.md).

The separate native audit uses 56 fresh-process/private-cache workers,
44 cold native builds and two correctness input sets per supported case.
All 16 ordinary cases pass on both main and current; all eight newly admitted
complete-row sum/conditional cases pass on current and retain main's source
rejection. Alias fallback passes; partial-warp/effectful guards reject.
The outcomes are **42 passes, 12 expected source rejections and two failures**.

Both failures are the same inherited unbounded-gather axis bug: main and
current generate byte-identical source and each mismatch 248/256 elements
(maximum absolute error 61) against the independent dim-2 Torch reference.
This control remains a known fallback defect; it was not changed to match
incorrect generated code. No newly admitted row program failed this matrix.

Across the 16 ordinary native pairs, compiler-call geometric-mean
current/main time is **1.024559×**; worst **1.330308×**, BF16 conditional 4×64,
**0.148293 → 0.197275 s**. The maximum current native compiler call is
**0.279662 s**. The original, bytecode-cache-unmatched fresh-process binding/sourcegen
ratios are **1.406320× /
1.224030×**, with a Python-preparation outlier **0.046497 → 0.803042 s**.
These single observations include startup variation and do not establish
causal or persistent regressions. Timers separate binding, source generation,
Python preparation, the actual `cute.compile` call and first invocation.
CUDA module loading is lazy for both repositories. The initial eager-loading
attempt (two completed workers and one partial) is retained separately and
excluded from this matrix. See [native coverage](../artifacts/topk-final-verification-20261012-v1/compile_coverage/native-v2/report.md)
and [raw native summary](../artifacts/topk-final-verification-20261012-v1/compile_coverage/native-v2/summary.json).

A bounded three-round follow-up used the same kernels, configurations,
private caches, lazy loading and timers for three selected ordinary cases:
BF16 conditionals at widths 64/1024 and an FP32 softmax anchor at width 1024.
All **18 workers pass** both input sets with one native compilation each;
generated source is byte-identical across repositories and rounds.

Under the original unequal Python bytecode-cache states, binding/source-generation
increases recur in all nine pairs.
Ratios of selected-case medians have geometric means **1.411731× /
1.250082×**; this is a selected subset, not a replacement for the 16-case
matrix. Binding medians add **0.315–0.352 seconds**. Native compiler median
ratios have geometric mean **1.025643×**.

| Original selected case (unequal Python caches) | Bind main → current (s) | Source generation main → current (s) | Native compiler main → current (s) |
| --- | ---: | ---: | ---: |
| BF16 conditional 4×64 | 0.810185 → 1.161894 | 0.232208 → 0.294471 | 0.130172 → 0.140878 |
| BF16 conditional 4×1024 | 0.821840 → 1.153917 | 0.229653 → 0.296576 | 0.122072 → 0.124955 |
| FP32 softmax 4×1024 | 0.792032 → 1.106699 | 0.249752 → 0.297918 | 0.176171 → 0.171577 |

The original BF16 1024 preparation spike did not recur: medians are
0.038043 → 0.036725 seconds. Another BF16 64 worker shows a 1.217015-second
preparation spike and 3.207608-second bind, so occasional startup outliers
remain. The native 33% observation did not recur; BF16 64 native medians are
**1.082244×**, with individual ratios 1.065411×, 1.123986× and 1.056502×.
No samples were dropped. Subsequent profiling found unequal Python bytecode
caches in this comparison. These observations do not establish a persistent
40% increase in backend analysis per kernel; the cause audit follows below.
See [repeated compilation stages](../artifacts/topk-final-verification-20261012-v1/compile_coverage/native-v3-repeat/report.md)
and [full distributions](../artifacts/topk-final-verification-20261012-v1/compile_coverage/native-v3-repeat/summary.json).


## Startup cause and matched bytecode-cache controls

The earlier startup comparison had a Python cache confound. Main had 98
valid bytecode caches among the timed Helion imports, while the verification
tree had none. `PYTHONDONTWRITEBYTECODE=1` prevented its repeated processes
from creating caches. In the first-bind cProfile pair, `builtins.compile`
self time increased **0.001739 → 0.342533 seconds**; that **0.340794-second**
delta explains almost all of the **0.346175-second** bind difference.
`source_to_code` ran zero times on main and 95 times on current.

Eight source-only diagnostic workers compare the original cache states and
matched private cold/prewarmed caches. With both bytecode caches cold,
first bind is **1.636831 → 1.653519 seconds**, a **0.016688-second** difference.
The imports/setup/bind/sourcegen stage sum is **17.351204 → 17.492663 seconds**
in that one pair. Second new-kernel isolated binds are **0.091179 → 0.092337 s**;
second source generation is **0.056641 → 0.055922 s**. Matched prewarmed caches
also give nearly equal second binds/sourcegen; a main first-bind/setup outlier
prevents interpreting its apparent current advantage as a parity result.

The added first-use paths are lazy imports: `CuteTopkHeuristic.register_facts`
loads top-k/row modules, and `CuteBackend.pre_codegen` loads the register
region/program/tensor modules. Actual register-region planning takes only
**29 μs first / 16 μs warm** for this anchor. Codegen registrations reload 15
modules once on both trees; under the original unequal caches their profiled
cost is **0.015152 → 0.078817 s**, with no second-call reload. Matched-cold
first sourcegen still has a **0.099746-second** gap in one pair, including
**0.036705 seconds** compiling new register modules, while prewarmed and
second-call source generation remove that large difference.

This evidence substantially removes the earlier binding gap and identifies
bytecode compilation/first-use imports, rather than recurring row planning,
as its cause. These are one-anchor diagnostic cache controls with profiling
and import-recording overhead, not a confidence interval or a new end-to-end
latency guarantee. All emitted sources and frozen code hashes match; no native
CuTe compilation was allowed. The GPU selected-softmax event measurements
exclude compiler startup and remain unchanged. See [startup cause](../artifacts/topk-startup-cause-20261012-v1/report.md),
[profile summary](../artifacts/topk-startup-cause-20261012-v1/summary.json), and
[bytecode cache proof](../artifacts/topk-startup-cause-20261012-v1/bytecode-cache-proof.json).

## Historical validation


Historical full-suite results measure V18 and retain their failures:

| Historical suite | JUnit passed | Failed | Errors | Skipped |
| --- | ---: | ---: | ---: | ---: |
| CuTe | 25,889 | 5 | 0 | 1,912 |
| Default | 18,424 | 6 | 0 | 8,591 |

JUnit counts include subtests. Ten structural/source failures match retained
baseline evidence; the remaining default-suite register-cache timing failure
has recurrence evidence but no matched timing attribution. The suite exits
remain nonzero. Historical lint passed Ruff/codespell and retained 49
environment/upstream Pyrefly diagnostics, with zero in changed files.
Full-suite source/evidence: [local-tests-v18-full-terminal-preservation.json](../artifacts/topk-helion-network-20261008/local-tests-v18-full-terminal-preservation.json).

Focused checks for later cleanup and CI work are separate evidence:

| Focused check | Result |
| --- | --- |
| Generic softmax codegen, row-local reductions and value recovery | 15 passed |
| All saved top-k/MoE configurations | 31 single fused launches generated |
| Trimmed pretuned unit compilation cases | 7 passed |
| Runtime staging boundaries | 12 SDK checks passed |
| Original native fragment cases and index-remapping check on GB300 | 5 passed, 3 subtests passed |
| Native conditional coarse selection, direct/packed recovery | 2 passed |
| BF16 simulator payload cases | 4 passed |
| Scalar cache/source checks after final extent guard | 6 passed |
| Axis scaffolding and live-coordinate validation | 13 passed |
| Final focused Ruff, formatting, device lint and codespell | Passed |
| Focused Pyrefly for edited runtime/register/scalar files | 0 errors |
| Latest top-k alias-fallback cases and controls | 28 passed; 8 singleton cases rechecked |
| Native cast chains, singleton stores and unsafe-consumer guards | 6 passed |
| Cold/warm CPU cache preservation with CuTe DSL 4.7 | 4 passed |
| PyTorch 2.13/Triton 3.7.1 descriptor cases after cold CPU codegen | 7 passed |
| Compiler type checks after the latest CI fixes | 0 errors |

Focused evidence: [generic softmax report](../artifacts/topk-softmax-generic-20261009-v1/report.md),
[runtime SDK checks](../artifacts/topk-runtime-boundaries-20261009-v1/sdk-tests.log), and
[earlier CI fix report](../artifacts/topk-pr3819-ci-20261011-v1/report.md).
The [CI validation before this fold](../artifacts/topk-pr3819-ci-20261012-v2/fix-summary.md)
records the cast/store and CPU capability-cache fixes. Final test/probe calls
in this CI validation finished within 2.4 seconds.

Current limits for review:

- Retained default-math selected-softmax measurements have five gaps below 0.99.
  Explicit fast math removes most of the deficit in the two measured N64 cases;
  a full fast-mode sweep was not collected. The historical V18 outcome and
  retained default-mode timing table remain separately qualified.
- Retained full-suite runs and reviews are recorded above. Subsequent
  recipe/test fixes were checked with focused reruns, rather than another full
  run on final v5. No full post-rebase suite was run.
- The retained #3819 CI audit covers head `fff8b4ee` and the folded fixes.
  TPU provisioning failed in that run because its GCP project was suspended.
  The rewritten local stack still requires fresh CI after publication.
- Shared register/reduction/scalar lowering has retained native and CPU
  coverage. The original CPU startup comparison had unequal bytecode caches;
  matched controls substantially remove the binding gap. The retained matrix
  includes an inherited gather fallback defect. Key-mode comparisons cover six pinned BF16 carriers
  and do not
  establish a universally best mode or independent benefit for every option.

## Evidence index

Raw measurements, configs, goal file and canonical archives remain under
ignored `artifacts/`; they are not included in this commit. Links are local
repository evidence paths and may not be available in a bare GitHub checkout.
Worker locators can refer to members of preserved archives rather than
standalone files. The canonical manifests and metric map resolve them.

- Current pinned table and per-row raw workers: [current-report-table-v1.json](../artifacts/topk-final-verification-20261012-v1/performance/current-report-table-v1.json), SHA-256
  `1cd212943bee556b7aa2d44edbee0e3cb71c805a91a905d9075e5d1c0b4c38dc`.
- Complete 55-reference/current source audit: [current-pinned-audit.json](../artifacts/topk-final-verification-20261012-v1/performance/evidence/current-pinned-audit-v1-complete-1791835260564110135.json), SHA-256
  `a28da6537a7318a4cbf1594affb345bc61b86cee7f9d5fe4ed46ab220a00824f`.
- Current source/test freeze: [snapshot-v5.json](../artifacts/topk-final-verification-20261012-v1/snapshot-v5.json).
- Current recovered/full-suite evidence: [full-suite-final-audit.json](../artifacts/topk-final-verification-20261012-v1/tests/full-suite-final-audit.json), SHA-256
  `cbc1c798f091ddc6a0538cac6dd6216c7ea2e0103fd24249247fbeb3fcab4a39`;
  [recovery ledger](../artifacts/topk-final-verification-20261012-v1/tests/failure-audit/final-ledger.json).
- Authoritative historical V18 table/config/reference data: [final-report-data.json](../artifacts/topk-helion-network-20261008/final-report-data.json), SHA-256
  `6cd0b9b92f63a62e600613a1d88e9d254a076dd2dc652c4b7d4dcb2c64cb7349`.
- Per-metric paired worker paths/hashes: [evidence.json](../artifacts/topk-final-report-20261012-v1/evidence.json), SHA-256
  `29c12db703ae96f658e712302a51fb10457496e05828ab960b6b477033122ba7`.
- Original accepted report: [final-report.md](../artifacts/topk-helion-network-20261008/final-report.md).
- Measured source provenance: [provenance.json](../artifacts/topk-helion-network-20261008/candidate-v18-snapshot-v1/provenance.json).
- Folded verified local tree: [folded.json](../artifacts/topk-stack-fold-20261012T224107Z/folded.json).
- Earlier published folded tree: [folded.json](../artifacts/topk-pr3819-ci-20261012-v2/folded.json).
- Earlier folded tree (2026-10-11): [folded.json](../artifacts/topk-pr3819-ci-20261011-v1/folded.json).

### E01

`moe-float32-32768x256-k8`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/local-v18-moe-gpu2-final-tools/comparison.json); SHA-256
`76cf21c0eb2d6d3873b7242d0215759384349ccd91b3fa2c17b576e581d60b03`.
Full row record: `final-report-data.json#/rows/0`. Preserved canonical:
[local-v18-moe-gpu2-final-terminal-preservation.json](../artifacts/topk-helion-network-20261008/local-v18-moe-gpu2-final-terminal-preservation.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/0/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/0/references`.

### E02

`topk-bfloat16-65536x64-k8`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/local-v18-slot87-gpu2-final-tools/comparison.json); SHA-256
`e981e86b336a927b5c91ce1f1fbfd434e980b92824644d55857e53c662677a60`.
Full row record: `final-report-data.json#/rows/1`. Preserved canonical:
[local-v18-slot87-gpu2-final-terminal-preservation.json](../artifacts/topk-helion-network-20261008/local-v18-slot87-gpu2-final-terminal-preservation.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/1/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/1/references`.

### E03

`topk-bfloat16-65536x64-k8-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-reassignment-v3-slot97-terminal-preserved/prewarm-retained-v18-runs-v1/reassignment-v3/controls/97/collect/comparison.json); SHA-256
`f3585bbce67b9d51dc1dd969b97566ab0b67d8648e83f3151fe731362bfee887`.
Full row record: `final-report-data.json#/rows/2`. Preserved canonical:
[retained-v18-reassignment-v3-slot97-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-reassignment-v3-slot97-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/2/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/2/references`.

### E04

`topk-bfloat16-65536x64-k16`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-empty-reassignment118-v1-slot118-terminal-preserved/prewarm-retained-v18-runs-v1/empty-reassignment118-v1/controls/118/collect/comparison.json); SHA-256
`55844d3d93e5a63624bee8493f1b420d8faec8ac839fc2f89414ab18af944c8a`.
Full row record: `final-report-data.json#/rows/3`. Preserved canonical:
[retained-v18-empty-reassignment118-v1-slot118-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-empty-reassignment118-v1-slot118-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/3/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/3/references`.

### E05

`topk-bfloat16-65536x64-k16-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot109-terminal-preserved/prewarm-retained-v18-runs-v1/replacements-v1/controls/109/collect/comparison.json); SHA-256
`fdec76722ad0b184938f0d84371af0f25fb7bd105e936c546cb132176b36b083`.
Full row record: `final-report-data.json#/rows/4`. Preserved canonical:
[retained-v18-replacements-v1-slot109-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot109-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/4/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/4/references`.

### E06

`topk-bfloat16-65536x64-k32`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-tail-final-recovery-slot97-terminal-preserved/prewarm-retained-v18-tail-final-recovery-v1-data/completion/97/comparison.json); SHA-256
`36f8630891a56f1cfa488f6e2de37f62b330d51fa84f40e9596a79fbc45104e4`.
Full row record: `final-report-data.json#/rows/5`. Preserved canonical:
[retained-tail-final-recovery-slot97-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-tail-final-recovery-slot97-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/5/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/5/references`.

### E07

`topk-bfloat16-65536x64-k32-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-final-only104-terminal-preserved/prewarm-retained-v18-final-only-reuse-v1-data/completion/104/comparison.json); SHA-256
`66bba48ae050b62ec269d07253f5f0feebb1c4f528bee3c5b62cc7fe9c44c230`.
Full row record: `final-report-data.json#/rows/6`. Preserved canonical:
[retained-final-only104-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-final-only104-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/6/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/6/references`.

### E08

`topk-bfloat16-65536x128-k8`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-reassignment-v3-slot105-terminal-preserved/prewarm-retained-v18-runs-v1/reassignment-v3/controls/105/collect/comparison.json); SHA-256
`2e27538fa54ad8616be89d90ca8991bf02249fc71df5cd5151c08f00df4fd4d9`.
Full row record: `final-report-data.json#/rows/7`. Preserved canonical:
[retained-v18-reassignment-v3-slot105-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-reassignment-v3-slot105-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/7/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/7/references`.

### E09

`topk-bfloat16-65536x128-k8-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-empty-reassignment-v1-slot117-terminal-preserved/prewarm-retained-v18-runs-v1/empty-reassignment-v1/controls/117/collect/comparison.json); SHA-256
`da5bd7c0cb5d6ef18d485e768d88c9645a993693a8320fa21a47d2b177600e93`.
Full row record: `final-report-data.json#/rows/8`. Preserved canonical:
[retained-v18-empty-reassignment-v1-slot117-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-empty-reassignment-v1-slot117-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/8/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/8/references`.

### E10

`topk-bfloat16-65536x128-k16`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-reassignment116-v1-slot116-terminal-preserved/prewarm-retained-v18-runs-v1/reassignment116-v1/controls/116/collect/comparison.json); SHA-256
`47ab57d81aa6d603fe3fc56ab87d61d8b2b1da410f9be8ce4a18bfe81611bd13`.
Full row record: `final-report-data.json#/rows/9`. Preserved canonical:
[retained-v18-reassignment116-v1-slot116-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-reassignment116-v1-slot116-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/9/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/9/references`.

### E11

`topk-bfloat16-65536x128-k16-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-reuse-v1-slot70-terminal-preserved/prewarm-retained-v18-runs-v1/reuse-v1/controls/70/collect/comparison.json); SHA-256
`d2e79068a8aadae4ff8ec2a8d24a658b200a8a759d2ada629d15325d9a278acf`.
Full row record: `final-report-data.json#/rows/10`. Preserved canonical:
[retained-v18-reuse-v1-slot70-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-reuse-v1-slot70-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/10/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/10/references`.

### E12

`topk-bfloat16-65536x128-k32`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot111-terminal-preserved/prewarm-retained-v18-runs-v1/replacements-v1/controls/111/collect/comparison.json); SHA-256
`0424f6e85dd0b310e5c38139243c2f9e56ab8b179742e3562e11ed27e118b1ab`.
Full row record: `final-report-data.json#/rows/11`. Preserved canonical:
[retained-v18-replacements-v1-slot111-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot111-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/11/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/11/references`.

### E13

`topk-bfloat16-65536x128-k32-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/local-v18-slot63-gpu2-final-tools/comparison.json); SHA-256
`5dc5622bfe977488511c4e9a6fe1d94075a02085a80e208e1b9ed7823c69274b`.
Full row record: `final-report-data.json#/rows/12`. Preserved canonical:
[local-v18-slot63-gpu2-final-terminal-preservation.json](../artifacts/topk-helion-network-20261008/local-v18-slot63-gpu2-final-terminal-preservation.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/12/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/12/references`.

### E14

`topk-bfloat16-65536x256-k8`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot112-terminal-preserved/prewarm-retained-v18-runs-v1/replacements-v1/controls/112/collect/comparison.json); SHA-256
`1fb6170672bc479d52e64b30be3330042474896c1c5d4f764c00b3ffbd2ff633`.
Full row record: `final-report-data.json#/rows/13`. Preserved canonical:
[retained-v18-replacements-v1-slot112-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot112-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/13/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/13/references`.

### E15

`topk-bfloat16-65536x256-k8-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot113-terminal-preserved/prewarm-retained-v18-runs-v1/replacements-v1/controls/113/collect/comparison.json); SHA-256
`a6ad45bde115495d750765988d2a165ee573d797c19087625fbbb317b1114d47`.
Full row record: `final-report-data.json#/rows/14`. Preserved canonical:
[retained-v18-replacements-v1-slot113-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot113-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/14/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/14/references`.

### E16

`topk-bfloat16-65536x256-k16`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot114-terminal-preserved/prewarm-retained-v18-runs-v1/replacements-v1/controls/114/collect/comparison.json); SHA-256
`ca7587b0ed7edbf6c1a6fa2fceb7569f754d1114b4d90ebac103cacc09ae6e92`.
Full row record: `final-report-data.json#/rows/15`. Preserved canonical:
[retained-v18-replacements-v1-slot114-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot114-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/15/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/15/references`.

### E17

`topk-bfloat16-65536x256-k16-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-reuse-v1-slot104-terminal-preserved/prewarm-retained-v18-runs-v1/reuse-v1/controls/104/collect/comparison.json); SHA-256
`fda2f8480686d84fed2e61c49ace080c2105d8e8e911cdbd01411ed993d52c0d`.
Full row record: `final-report-data.json#/rows/16`. Preserved canonical:
[retained-v18-reuse-v1-slot104-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-reuse-v1-slot104-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/16/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/16/references`.

### E18

`topk-bfloat16-65536x256-k32`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-tail-final-recovery-slot70-terminal-preserved/prewarm-retained-v18-tail-final-recovery-v1-data/completion/70/comparison.json); SHA-256
`4b042727fc9f0e2b8fa92736f0a72ca527ee85e22cb716ef96c45a91a3e97c55`.
Full row record: `final-report-data.json#/rows/17`. Preserved canonical:
[retained-tail-final-recovery-slot70-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-tail-final-recovery-slot70-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/17/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/17/references`.

### E19

`topk-bfloat16-65536x256-k32-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/local-v18-w256k32sm-gpu2-final-tools/comparison.json); SHA-256
`fd6af1073d6ce72f4aebcc168e707593a3a9d8eca3c0c39d5d548ba40ba663ec`.
Full row record: `final-report-data.json#/rows/18`. Preserved canonical:
[local-v18-w256k32sm-gpu2-final-terminal-preservation.json](../artifacts/topk-helion-network-20261008/local-v18-w256k32sm-gpu2-final-terminal-preservation.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/18/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/18/references`.

### E20

`topk-bfloat16-65536x512-k8`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-reuse-v1-slot97-terminal-preserved/prewarm-retained-v18-runs-v1/reuse-v1/controls/97/collect/comparison.json); SHA-256
`ecb20b8245f4d7489b875dbc4d0ea1734d2b3dcac88a03b88e8e9f53756895c1`.
Full row record: `final-report-data.json#/rows/19`. Preserved canonical:
[retained-v18-reuse-v1-slot97-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-reuse-v1-slot97-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/19/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/19/references`.

### E21

`topk-bfloat16-65536x512-k8-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot115-terminal-preserved/prewarm-retained-v18-runs-v1/replacements-v1/controls/115/collect/comparison.json); SHA-256
`964893c0cfe105e5c8a746adf30966447a008333e979ecd3ab8896cc107ca971`.
Full row record: `final-report-data.json#/rows/20`. Preserved canonical:
[retained-v18-replacements-v1-slot115-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot115-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/20/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/20/references`.

### E22

`topk-bfloat16-65536x512-k16`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot116-terminal-preserved/prewarm-retained-v18-runs-v1/replacements-v1/controls/116/collect/comparison.json); SHA-256
`aabbebd8d5a8feb3e410da0c8ea381fea464b69cb60ccb5a7532b4bafc11a24e`.
Full row record: `final-report-data.json#/rows/21`. Preserved canonical:
[retained-v18-replacements-v1-slot116-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-replacements-v1-slot116-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/21/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/21/references`.

### E23

`topk-bfloat16-65536x512-k16-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-reassignment-v3-slot70-terminal-preserved/prewarm-retained-v18-runs-v1/reassignment-v3/controls/70/collect/comparison.json); SHA-256
`9e52994e1874349466bb93384abed12fbaff7b5dc4164d07443c3a4e9caeec95`.
Full row record: `final-report-data.json#/rows/22`. Preserved canonical:
[retained-v18-reassignment-v3-slot70-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-reassignment-v3-slot70-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/22/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/22/references`.

### E24

`topk-bfloat16-65536x512-k32`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-reuse-v1-slot107-terminal-preserved/prewarm-retained-v18-runs-v1/reuse-v1/controls/107/collect/comparison.json); SHA-256
`a52591e1ab7d0a5900d998506e2394220320069747159205ed4d9d37c37a7e33`.
Full row record: `final-report-data.json#/rows/23`. Preserved canonical:
[retained-v18-reuse-v1-slot107-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-reuse-v1-slot107-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/23/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/23/references`.

### E25

`topk-bfloat16-65536x512-k32-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-reuse-v1-slot93-terminal-preserved/prewarm-retained-v18-runs-v1/reuse-v1/controls/93/collect/comparison.json); SHA-256
`2a91bb9460053f0d350f51241f0ef12ff67d9cbdc770dcf71a3b529f751b565d`.
Full row record: `final-report-data.json#/rows/24`. Preserved canonical:
[retained-v18-reuse-v1-slot93-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-reuse-v1-slot93-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/24/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/24/references`.

### E26

`topk-bfloat16-65536x1024-k8`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/local-v18-slot108-gpu3-final-tools/comparison.json); SHA-256
`19a718a12611d5da2e5c8e133df42175466831ea1374a1fdf6bcaf83863357c9`.
Full row record: `final-report-data.json#/rows/25`. Preserved canonical:
[local-v18-slot108-gpu3-final-terminal-preservation.json](../artifacts/topk-helion-network-20261008/local-v18-slot108-gpu3-final-terminal-preservation.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/25/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/25/references`.

### E27

`topk-bfloat16-65536x1024-k8-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-reassignment-v3-slot104-terminal-preserved/prewarm-retained-v18-runs-v1/reassignment-v3/controls/104/collect/comparison.json); SHA-256
`bb11f1f019bc6f095abf246f7561355846f028772a738ec75f5c4d10fce885df`.
Full row record: `final-report-data.json#/rows/26`. Preserved canonical:
[retained-v18-reassignment-v3-slot104-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-reassignment-v3-slot104-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/26/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/26/references`.

### E28

`topk-bfloat16-65536x1024-k16`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-slot100-budget128-v1-slot100-terminal-preserved/prewarm-retained-v18-runs-v1/slot100-budget128-v1/controls/100/collect/comparison.json); SHA-256
`6396bd0e5330df7210490b1f0ddc03dc822a05c70504d6f75394ee394f8f5430`.
Full row record: `final-report-data.json#/rows/27`. Preserved canonical:
[retained-v18-slot100-budget128-v1-slot100-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-slot100-budget128-v1-slot100-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/27/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/27/references`.

### E29

`topk-bfloat16-65536x1024-k16-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-reassignment-v3-slot107-terminal-preserved/prewarm-retained-v18-runs-v1/reassignment-v3/controls/107/collect/comparison.json); SHA-256
`7b9e4c41e0056815b707c4023751f1019bf871061363b792fe70e05b0a91032a`.
Full row record: `final-report-data.json#/rows/28`. Preserved canonical:
[retained-v18-reassignment-v3-slot107-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-reassignment-v3-slot107-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/28/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/28/references`.

### E30

`topk-bfloat16-65536x1024-k32`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-v18-reuse-v1-slot105-terminal-preserved/prewarm-retained-v18-runs-v1/reuse-v1/controls/105/collect/comparison.json); SHA-256
`9a695718ee4cad3886949e4ab54cb81ca8d6f89ddb43596144a6e6bd7937ac20`.
Full row record: `final-report-data.json#/rows/29`. Preserved canonical:
[retained-v18-reuse-v1-slot105-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-v18-reuse-v1-slot105-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/29/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/29/references`.

### E31

`topk-bfloat16-65536x1024-k32-softmax`. Primary paired comparison: [comparison.json](../artifacts/topk-helion-network-20261008/retained-tail-final-recovery-slot105-terminal-preserved/prewarm-retained-v18-tail-final-recovery-v1-data/completion/105/comparison.json); SHA-256
`26130203f85275bfe967a619d578f40d8fc97550e9e6a2acebc5d0529cdb3bac`.
Full row record: `final-report-data.json#/rows/30`. Preserved canonical:
[retained-tail-final-recovery-slot105-terminal-verification.json](../artifacts/topk-helion-network-20261008/retained-tail-final-recovery-slot105-terminal-verification.json). Worker paths/hashes and auxiliary paired timings
are in `evidence.json#/rows/30/selected_metric_sources`. Full pin
configurations are in `final-report-data.json#/rows/30/references`.
