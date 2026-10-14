**[no land] GB300 top-k config discovery**

This report is layered above compiler PRs [#3982](https://github.com/pytorch/helion/pull/3982), [#3983](https://github.com/pytorch/helion/pull/3983), and [#3984](https://github.com/pytorch/helion/pull/3984), rebased on [`cedaa67`](https://github.com/pytorch/helion/commit/cedaa67b59e8bb687bb9234cad9d4fc7e7056bbd). The corresponding kernel-library commit is [`a22cf4c`](https://github.com/yushangdi/helion-kernel-library/commit/a22cf4c1a2947cedccc1c452f6fb250aa049ddd5). Performance on the current rebased compiler/library head has not been revalidated across all 40 cases.

The report and companion [provenance index](gb300-topk-config-discovery-20261014.json) form the report commit. Raw logs, source/config snapshots and receipts remain uncommitted under the shared local root `/data/users/shangdiy/helion-topk-paper-hillclimb-20261013/artifacts/topk-paper-20261013/`. The code-form paths below identify those artifacts; they are not GitHub links. The index records paths, hashes, source IDs and metrics without embedding raw configuration dictionaries.

The eight committed SM103 tables contain key/config records. The selectors match accepted K4/K5 configurations and the latest V29 K6 receipt configurations; this is distinct from the earlier timing admissions retained for K6/00 and K6/03. Config/source copying and static equivalence audits do not substitute for a complete performance run on the committed heads. K6/00 and K6/03 retain the earlier accepted timing banks as qualified below.

The user explicitly deferred cold-search verification:

> for each kernel and shape, let's first focus on getting a config that matches/exceed the goal. delay the verification for whether a cold full search can find the config. record the config you found for each shape

**Config discovery phase: 40 pinned configurations**

All 40 named-baseline config targets meet the agreed per-shape baseline/Helion ratio bar of 0.99. K5/03 is 0.997245× (11.616 versus 11.584 µs), about 0.28% slower and within that tolerance. The geometric mean of the accepted baseline/Helion ratios is **1.1863×** (bar 1.00×). The authoritative inputs are the current `compiler:artifacts/goal.json`, its exact config-artifact pointers, and the `checkers/config-discovery-40-current-v2.txt`; the [machine-readable index](gb300-topk-config-discovery-20261014.json) records full values, stages, hashes, and source references.

This is the accepted pinned-config discovery phase. Cold-full autotuner rediscovery is deferred by the user's explicit instruction. The before-CuTe cold baseline is incomplete, so no before-CuTe timing column is reported. Manual full test runs and full production review remain deferred. Repository CI is tracked separately on the linked PRs; those runs do not reproduce the frozen performance banks. Final landing validation is not claimed. The 40 admissions use their exact versioned source/config banks, so this report makes no claim that one current tree reproduces every row.

Measurements used local **NVIDIA GB300 GPUs at 1400 W**, physical **GPU2/GPU3**, with per-row GPU UUIDs in the JSON. Candidate CuTe DSL is **4.7**; the FlashInfer radix baseline uses CuTe DSL **4.8**; SGLang is **0.5.21** with SGLang-kernel **0.4.7**. Candidate fast math is **False**. Workers use the same validated CUDA-event graph and cold-L2 timing policy; the V3/V4 timer source is byte-identical (`dc78cd2506363251b3b602dd51ade9785c4adafee399f3feb564de3dcd04fb98`). Per-row package/config/source details remain in the accepted records.

K4 keeps `torch.topk` in ordinary Helion and lowers the canonical Int32/F32-rank expression to cooperative radix selection, with typed epilogues and scoped geometry, cache, scan and payload-recovery choices. K5 uses ordinary Helion softmax and top-k expressions with saved configuration families. K6 uses the general Int64 sorting network, preserves unconditional in-call Torch RNG, and uses a guarded per-chunk Int64 mass computation whose 53-bit proof preserves the FP64 sum, with FP64 fallback. Replay, reseeding and weighted-distribution checks preserve each implementation's public sampling contract; cross-backend fixed-uniform token identity is not a required contract.

The `k6/current-v29-final-audit.md` and `k6/current-v29-configs-v2/ledger.json` record **36 passing workers** for source, eager/graph, changed/restored-input, correctness and distribution checks. RNG output-variation checks report 30 `PASS` and six `NOT_TESTABLE` results for the unique-support K=1 case; the separate generator-state proof is retained. The new K6/00 timing window is excluded for **1.2126% FlashInfer baseline-window variation** (endpoint anchor drift was 0%); its admitted v1 conservative **1.104148×** ratio remains. K6/03 retains its v1 source bank and unique-support K1 generator-state proof; the current 124.144/234.672 µs result (1.890321×) is separate supporting evidence and does not replace that admission. Qualified current K6/01, /02, /04 and /05 records are the exact v2 goal pointers shown below.

Cleanup removed or reverted obsolete experiments across **28 paths**, recorded in `cleanup-obsolete-experiments-v1/final-summary.json`. Its `cleanup-obsolete-experiments-v1/source-identity.json` preserves raw generated-code, normalized-config, and host/device AST bytes exactly. Cleanup used no GPU, native compilation, or full test suite; the immutable performance banks remain unchanged.

All timings below are microseconds from the accepted config records. `flashinfer-radix-filter`, `sglang-cuda`, and `flashinfer-joint` are the named baseline identifiers from the goal. The last row uses arithmetic means for absolute latencies and a geometric mean for ratios. Source versions can differ across rows; each row identifies the frozen evidence used for admission.

| Case | Exact shape | Named baseline | Baseline µs | Helion µs | Accepted ratio | Config | Raw summary / receipt | Source version |
|---|---|---|---:|---:|---:|---|---|---|
| K04/00 | `[4, 4096]; k=512; float32; uniform` | flashinfer-radix-filter | 13.760 | 13.680 | 1.005848× | `configs/K04-00-v1.json` | `k4/public-c1-confirmation-prep-v1/K04-00/confirmation-window-v1/summary.json` | `k4/public-c1-confirmation-prep-v1/K04-00/candidate-manifest.json` |
| K04/01 | `[8, 8192]; k=1024; float32; uniform` | flashinfer-radix-filter | 15.776 | 15.744 | 1.002033× | `configs/K04-01-v1.json` | `k4/public-fourflag-near-confirm-execution-v1/K04-01/summary.json` | `k4/fourflag-small-config-prep-v1/T1024_C1_B4096_all/source-manifest.json` |
| K04/02 | `[16, 32768]; k=2048; float32; uniform` | flashinfer-radix-filter | 23.968 | 19.808 | 1.210016× | `configs/K04-02-v1.json` | `k4/public-guided-guards-screen-v1/K04-02/far-summary-v1.json` | `k4/radix-control-breadth-v1/T1024_C4_B4096_P256_U4/source-manifest-v1.json` |
| K04/03 | `[24, 128000]; k=2048; float32; uniform` | flashinfer-radix-filter | 30.096 | 28.000 | 1.074857× | `configs/K04-03-v1.json` | `k4/public-guided-guards-screen-v1/K04-03/far-summary-v1.json` | `k4/radix-control-breadth-v1/T1024_C4_B4096_P256_U4/source-manifest-v1.json` |
| K04/04 | `[8, 8192]; k=512; float32; ragged` | flashinfer-radix-filter | 15.776 | 15.712 | 1.004073× | `configs/K04-04-v1.json` | `k4/public-fourflag-near-confirm-execution-v1/K04-04/summary.json` | `k4/fourflag-small-config-prep-v1/T1024_C1_B4096_all/source-manifest.json` |
| K04/05 | `[4, 4096]; k=512; bfloat16; uniform` | flashinfer-radix-filter | 15.776 | 13.600 | 1.160000× | `configs/K04-05-v1.json` | `k4/public-c1-config-discovery-v2/K04-05/01-T512_C1_B4096_direct/result.json` | `k4/production-radix-public-v1/fixed-config-breadth-v2/T512_C1_B4096_direct/source-manifest.json` |
| K04/06 | `[8, 8192]; k=1024; bfloat16; uniform` | flashinfer-radix-filter | 15.792 | 15.728 | 1.004069× | `configs/K04-06-v1.json` | `k4/public-c1-confirmation-prep-v1/K04-06/confirmation-window-v1/summary.json` | `k4/public-c1-confirmation-prep-v1/K04-06/candidate-manifest.json` |
| K04/07 | `[16, 32768]; k=2048; bfloat16; uniform` | flashinfer-radix-filter | 19.872 | 17.760 | 1.118919× | `configs/K04-07-v1.json` | `k4/public-guided-guards-screen-v1/K04-07/far-summary-v1.json` | `k4/radix-control-breadth-v1/T1024_C4_B4096_P128_U1/source-manifest-v1.json` |
| K04/08 | `[24, 128000]; k=2048; bfloat16; uniform` | flashinfer-radix-filter | 23.984 | 23.904 | 1.003347× | `configs/K04-08-v1.json` | `k4/public-production-inverse-K04-08-abab-v1/window/summary.json` | `k4/production-radix-public-inverse-v1/source-manifest-v1.json` |
| K04/09 | `[8, 8192]; k=512; bfloat16; ragged` | flashinfer-radix-filter | 15.776 | 13.696 | 1.151869× | `configs/K04-09-v1.json` | `k4/public-fourflag-small-screen-v1/K04-09/far-summary-v1.json` | `k4/fourflag-small-config-prep-v1/T1024_C1_B4096_all/source-manifest.json` |
| K05/00 | `[1, 8]; k=2; float32; renormalize=false` | sglang-cuda | 7.520 | 5.600 | 1.342857× | `configs/K05-00-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-00/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/01 | `[32, 64]; k=8; float32; renormalize=false` | sglang-cuda | 7.584 | 7.536 | 1.006369× | `configs/K05-01-v1.json` | `k5/config-discovery-v1/runs/confirm-retry-01-v1/K05-01/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/02 | `[512, 128]; k=8; float32; renormalize=false` | sglang-cuda | 7.536 | 7.536 | 1.000000× | `configs/K05-02-v1.json` | `k5/config-discovery-v1/runs/confirm-v1/K05-02/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/03 | `[4096, 256]; k=8; float32; renormalize=false` | sglang-cuda | 11.584 | 11.616 | 0.997245× | `configs/K05-03-v1.json` | `k5/config-discovery-v1/runs/confirm-v1/K05-03/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/04 | `[4, 512]; k=10; float32; renormalize=false` | sglang-cuda | 9.568 | 7.552 | 1.266949× | `configs/K05-04-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-04/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/05 | `[128, 512]; k=10; float32; renormalize=false` | sglang-cuda | 9.600 | 7.552 | 1.271186× | `configs/K05-05-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-05/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/06 | `[1, 8]; k=2; float32; renormalize=true` | sglang-cuda | 7.536 | 7.520 | 1.002128× | `configs/K05-06-v1.json` | `k5/config-discovery-v1/runs/confirm-retry-03-v1/K05-06/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/07 | `[32, 64]; k=8; float32; renormalize=true` | sglang-cuda | 7.712 | 7.488 | 1.029915× | `configs/K05-07-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-07/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/08 | `[512, 128]; k=8; float32; renormalize=true` | sglang-cuda | 7.552 | 7.536 | 1.002123× | `configs/K05-08-v1.json` | `k5/config-discovery-v1/runs/confirm-v1/K05-08/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/09 | `[4096, 256]; k=8; float32; renormalize=true` | sglang-cuda | 11.616 | 9.568 | 1.214047× | `configs/K05-09-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-09/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/10 | `[4, 512]; k=10; float32; renormalize=true` | sglang-cuda | 9.568 | 7.552 | 1.266949× | `configs/K05-10-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-10/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/11 | `[128, 512]; k=10; float32; renormalize=true` | sglang-cuda | 9.600 | 8.192 | 1.171875× | `configs/K05-11-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-11/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/12 | `[1, 8]; k=2; bfloat16; renormalize=false` | sglang-cuda | 7.520 | 5.568 | 1.350575× | `configs/K05-12-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-12/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/13 | `[32, 64]; k=8; bfloat16; renormalize=false` | sglang-cuda | 7.552 | 7.536 | 1.002123× | `configs/K05-13-v1.json` | `k5/config-discovery-v1/runs/confirm-v1/K05-13/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/14 | `[512, 128]; k=8; bfloat16; renormalize=false` | sglang-cuda | 7.536 | 7.520 | 1.002128× | `configs/K05-14-v1.json` | `k5/config-discovery-v1/runs/confirm-v1/K05-14/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/15 | `[4096, 256]; k=8; bfloat16; renormalize=false` | sglang-cuda | 11.616 | 7.520 | 1.544681× | `configs/K05-15-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-15/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/16 | `[4, 512]; k=10; bfloat16; renormalize=false` | sglang-cuda | 9.536 | 7.520 | 1.268085× | `configs/K05-16-v2.json` | `k5/config-discovery-v1/runs/initial-v1/K05-16/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/17 | `[128, 512]; k=10; bfloat16; renormalize=false` | sglang-cuda | 9.648 | 7.536 | 1.280255× | `configs/K05-17-v1.json` | `k5/config-discovery-v1/runs/confirm-v1/K05-17/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/18 | `[1, 8]; k=2; bfloat16; renormalize=true` | sglang-cuda | 7.520 | 5.728 | 1.312849× | `configs/K05-18-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-18/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/19 | `[32, 64]; k=8; bfloat16; renormalize=true` | sglang-cuda | 7.536 | 7.536 | 1.000000× | `configs/K05-19-v1.json` | `k5/config-discovery-v1/runs/confirm-v1/K05-19/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/20 | `[512, 128]; k=8; bfloat16; renormalize=true` | sglang-cuda | 7.520 | 7.456 | 1.008584× | `configs/K05-20-v1.json` | `k5/config-discovery-v1/runs/confirm-v1/K05-20/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/21 | `[4096, 256]; k=8; bfloat16; renormalize=true` | sglang-cuda | 11.616 | 7.520 | 1.544681× | `configs/K05-21-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-21/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/22 | `[4, 512]; k=10; bfloat16; renormalize=true` | sglang-cuda | 9.568 | 7.520 | 1.272340× | `configs/K05-22-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-22/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K05/23 | `[128, 512]; k=10; bfloat16; renormalize=true` | sglang-cuda | 9.600 | 7.520 | 1.276596× | `configs/K05-23-v1.json` | `k5/config-discovery-v1/runs/initial-v1/K05-23/summary.json` | `k5/snapshots/generic-seeded-v1/manifest.json` |
| K06/00 | `[1, 32000]; k=50; p=0.9; float32` | flashinfer-joint | 40.064 | 35.504 | 1.104148×† | `configs/K06-00-v1.json` | `k6/public-v23-anchored-v1/K06-00/summary.json` | `k6/snapshots/public-v23-k4-pipeline-v1/manifest.json` |
| K06/01 | `[19, 128256]; k=50; p=0.9; float32` | flashinfer-joint | 93.344 | 56.480 | 1.652691× | `configs/K06-01-v2.json` | `k6/current-v29-anchored-v1/K06-01/summary.json` | `k4/production-radix-public-inverse-v1/source-manifest-v1.json` |
| K06/02 | `[99, 128256]; k=50; p=0.1; float32` | flashinfer-joint | 244.912 | 136.320 | 1.796596× | `configs/K06-02-v2.json` | `k6/current-v29-anchored-v1/K06-02/summary.json` | `k4/production-radix-public-inverse-v1/source-manifest-v1.json` |
| K06/03 | `[989, 32000]; k=1; p=0.9; float32` | flashinfer-joint | 234.672 | 124.320 | 1.887645× | `configs/K06-03-v1.json` | `k6/public-v18-anchored-v1/K06-03/summary.json` | `k6/snapshots/public-v18-v1/manifest.json` |
| K06/04 | `[16, 151936]; k=100; p=0.9; float32` | flashinfer-joint | 105.184 | 68.768 | 1.529549× | `configs/K06-04-v2.json` | `k6/current-v29-anchored-v1/K06-04/summary.json` | `k4/production-radix-public-inverse-v1/source-manifest-v1.json` |
| K06/05 | `[128, 151936]; k=1000; p=0.5; float32` | flashinfer-joint | 264.944 | 244.896 | 1.081863× | `configs/K06-05-v2.json` | `k6/current-v29-anchored-v1/K06-05/summary.json` | `k4/production-radix-public-inverse-v1/source-manifest-v1.json` |
| **All 40** | **40 shapes; arithmetic latency means** | Mixed named baselines | **34.686** | **25.641** | **1.186292× GM** | [index](gb300-topk-config-discovery-20261014.json) | `checkers/config-discovery-40-current-v2.txt` | Per row |

† K6/00 lists the stored Helion and baseline medians (35.504 and 40.064 µs). Their median ratio is 1.128436×. Acceptance and every geometric mean instead use the conservative `min(FlashInfer)/max(Helion)` ratio **1.104148×**. K5 initial records use `baseline_us_conservative_min`; confirmation records that only store `baseline_us` retain that field. K4 uses `candidate_us`/`baseline_us`; K6 uses `candidate_median_us`/`named_baseline_median_us` for the displayed absolute timings.

| Group | Shapes | Baseline arithmetic mean µs | Helion arithmetic mean µs | Accepted-ratio GM |
|---|---:|---:|---:|---:|
| K04 | 10 | 19.057600 | 17.763200 | 1.070828× |
| K05 | 24 | 8.906000 | 7.572000 | 1.172662× |
| K06 | 6 | 163.853332 | 111.048000 | 1.473627× |
| All 40 | 40 | 34.686000 | 25.641200 | 1.186292× |

The arithmetic means weight each listed shape once and retain its full public-operation latency; they do not normalize by batch, vocabulary, or elements. Raw per-run values and package/config/source provenance remain in the referenced admission records.


**Published compiler stack and recorded validation**

The three compiler commits are `5790d7127183d707afa3d8768ab3eff49f5e5e5a` (#3982), `af4eff1b321d1acdf49a6d7d36ac810f92e76d0c` (#3983), and `456bbf63374c585a025dcc37b66e2798e22770f8` (#3984). Their final tree is `97de6ce556e7e6b3af747a5520f63ebb356dbf6a`. Temporary-index replay of all three layers onto `cedaa67b` matched the independently ported full delta. The saved runs below precede publication and use earlier compiler/library worktrees and frozen source banks. Their results are distinct from the later repository CI runs shown on the linked PRs.

| Change | Recorded checks | Local evidence |
|---|---|---|
| AST-epoch load-fusion cache | 36 CPU tests; all 18 captured fuser invocations produced identical output with and without caching | `startup-profile-v1/fuser-focused-tests-v2.log`; `startup-profile-v1/fixture-byte-comparison.json` |
| Stable sort and bounded Int64 networks | 48 CPU/model cases; six unchanged legacy-source controls; both sort directions staged through SDK IR without native compilation | `int64-value-network-v1/pytest-v2.log`; `int64-value-network-v1/legacy-source-identity.json`; `int64-value-network-v1/sdk-result.json` |
| Small-row top-k seeds | Four corrected seed nodes and 20 existing selection checks passed in separate runs | `k5/small_row_seed_cpu_tests.log`; `k5/selection_existing_cpu_tests.log` |
| Cooperative radix and payload recovery | Ten focused inverse/legacy CPU checks; ten K4 shapes through offline-AOT source generation; four native exceptional-bit cases with graph replay | `k4/cached-key-payload-promotion-v1/focused-cpu-v3.log`; `k4/production-radix-public-inverse-v1/public-aot-cpu-receipt-v1.json`; `k4/cached-key-payload-promotion-v1/native-exceptional-stress-v2/result.json` |

The K6 V29 audit covers 12 Helion and 24 named-baseline executions on the final joint `d23c0142…` source bank. Both native integration tests, including maximum ties and graph RNG, passed before cleanup (`k6/current-v29-exact-tests-v1/result.json`). Cleanup then proved 13 live configurations, generated source files and host/device ASTs identical. The later library preparation recorded 21 portable CPU AOT checks with compiler `test.*` imports blocked and 29 destination CPU reference/metadata checks. These are not a full native revalidation of library commit `a22cf4c`.

**Remaining work before landing**

- Revalidate the complete K4/K5/K6 performance and correctness sweep on the published compiler and library heads. The timing table contains accepted source-bank admissions, including the retained V23 and V18 K6 cases. No new performance result was collected for this report.
- Verify cold full-effort autotuner rediscovery. Saved configurations and appended seeds do not establish that a fresh search reaches the recorded winners. The complete pre-hillclimb cold-CuTe baseline is also unavailable.
- Complete production review and assess the full repository CI results. Shared analysis caching, sort/scan/load lowering and the common reduction roller can affect other eligible programs; the roller's stable-sort handling also needs coverage on other backends. Broader correctness and compilation/startup-time coverage is still required.
- Compare key-encoding and schedule alternatives under controlled conditions. Accepted end-to-end configurations do not prove each mode or optional knob's independent benefit.

**Fixed-config checker output**

The checker validates the stored ratios and the existence of their artifact files. Its `GOAL MET` verdict is the numeric config-discovery verdict under the user's explicit deferral; it does not independently prove cold-search reachability or current-tree execution coverage.

<details>
<summary>40-case checker output, re-evaluated for this report</summary>

```text
PASS K04/00: ratio 1.006 >= bar 0.990  [artifacts/topk-paper-20261013/k4/public-c1-confirmation-prep-v1/K04-00/confirmation-window-v1/summary.json]
PASS K04/01: ratio 1.002 >= bar 0.990  [artifacts/topk-paper-20261013/k4/public-fourflag-near-confirm-execution-v1/K04-01/summary.json]
PASS K04/02: ratio 1.210 >= bar 0.990  [artifacts/topk-paper-20261013/k4/public-guided-guards-screen-v1/K04-02/far-summary-v1.json]
PASS K04/03: ratio 1.075 >= bar 0.990  [artifacts/topk-paper-20261013/k4/public-guided-guards-screen-v1/K04-03/far-summary-v1.json]
PASS K04/04: ratio 1.004 >= bar 0.990  [artifacts/topk-paper-20261013/k4/public-fourflag-near-confirm-execution-v1/K04-04/summary.json]
PASS K04/05: ratio 1.160 >= bar 0.990  [artifacts/topk-paper-20261013/configs/K04-05-v1.json]
PASS K04/06: ratio 1.004 >= bar 0.990  [artifacts/topk-paper-20261013/k4/public-c1-confirmation-prep-v1/K04-06/confirmation-window-v1/summary.json]
PASS K04/07: ratio 1.119 >= bar 0.990  [artifacts/topk-paper-20261013/k4/public-guided-guards-screen-v1/K04-07/far-summary-v1.json]
PASS K04/08: ratio 1.003 >= bar 0.990  [artifacts/topk-paper-20261013/k4/public-production-inverse-K04-08-abab-v1/window/summary.json]
PASS K04/09: ratio 1.152 >= bar 0.990  [artifacts/topk-paper-20261013/k4/public-fourflag-small-screen-v1/K04-09/far-summary-v1.json]
PASS K05/00: ratio 1.343 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-00/summary.json]
PASS K05/01: ratio 1.006 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/confirm-retry-01-v1/K05-01/summary.json]
PASS K05/02: ratio 1.000 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/confirm-v1/K05-02/summary.json]
PASS K05/03: ratio 0.997 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/confirm-v1/K05-03/summary.json]
PASS K05/04: ratio 1.267 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-04/summary.json]
PASS K05/05: ratio 1.271 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-05/summary.json]
PASS K05/06: ratio 1.002 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/confirm-retry-03-v1/K05-06/summary.json]
PASS K05/07: ratio 1.030 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-07/summary.json]
PASS K05/08: ratio 1.002 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/confirm-v1/K05-08/summary.json]
PASS K05/09: ratio 1.214 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-09/summary.json]
PASS K05/10: ratio 1.267 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-10/summary.json]
PASS K05/11: ratio 1.172 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-11/summary.json]
PASS K05/12: ratio 1.351 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-12/summary.json]
PASS K05/13: ratio 1.002 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/confirm-v1/K05-13/summary.json]
PASS K05/14: ratio 1.002 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/confirm-v1/K05-14/summary.json]
PASS K05/15: ratio 1.545 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-15/summary.json]
PASS K05/16: ratio 1.268 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-16/summary.json]
PASS K05/17: ratio 1.280 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/confirm-v1/K05-17/summary.json]
PASS K05/18: ratio 1.313 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-18/summary.json]
PASS K05/19: ratio 1.000 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/confirm-v1/K05-19/summary.json]
PASS K05/20: ratio 1.009 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/confirm-v1/K05-20/summary.json]
PASS K05/21: ratio 1.545 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-21/summary.json]
PASS K05/22: ratio 1.272 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-22/summary.json]
PASS K05/23: ratio 1.277 >= bar 0.990  [artifacts/topk-paper-20261013/k5/config-discovery-v1/runs/initial-v1/K05-23/summary.json]
PASS K06/00: ratio 1.104 >= bar 0.990  [artifacts/topk-paper-20261013/k6/public-v23-anchored-v1/K06-00/summary.json]
PASS K06/01: ratio 1.653 >= bar 0.990  [artifacts/topk-paper-20261013/k6/current-v29-anchored-v1/K06-01/summary.json]
PASS K06/02: ratio 1.797 >= bar 0.990  [artifacts/topk-paper-20261013/k6/current-v29-anchored-v1/K06-02/summary.json]
PASS K06/03: ratio 1.888 >= bar 0.990  [artifacts/topk-paper-20261013/k6/public-v18-anchored-v1/K06-03/summary.json]
PASS K06/04: ratio 1.530 >= bar 0.990  [artifacts/topk-paper-20261013/k6/current-v29-anchored-v1/K06-04/summary.json]
PASS K06/05: ratio 1.082 >= bar 0.990  [artifacts/topk-paper-20261013/k6/current-v29-anchored-v1/K06-05/summary.json]
PASS geomean: 1.1863 (min 1.0)
GOAL MET
```

</details>
