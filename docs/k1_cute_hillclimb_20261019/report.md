# K1 grouped BF16 GEMM: CuTe before/after report

Three fresh full cold searches cover six variants. All six cold ratios exceed 1.0; geometric mean is **1.018844×**. The six unchanged public `benchmark.sh` AOT checks meet the 0.99 matching threshold, with geometric mean **1.032790×**: five beat QuACK, and wide skew is 0.47% slower (**0.995289×**), within the skill’s 1% tolerance.

**Hardware and versions:** NVIDIA GB300, SM103, 152 SMs, **1400 W**. Cases 0/1 used GPU 0, 2/3 GPU 1, and 4/5 GPU 2; GPU 3 belonged to another user. Torch 2.14.1+cu130, Triton 3.8.0, QuACK 0.6.5, CUTLASS DSL 4.8.0, FlashInfer 0.7.2, CUPTI 13.0.1. Original Helion: `bfc25902f558b036e2c602be72d31d1c04fe4363` (`1.5.2.dev35+gbfc25902f`). Frozen V8: that same pin plus patch `a9538714676e19a72174bc5d6ad7461873b33ae7f65035be38264d7f411f89d3`.

## Useful throughput: before and final cold search

Values are **TFLOPS**, computed from useful `2*M*N*K` operations and the saved latency; `M` already includes every expert’s rows. Padding and redundant issued work do not add to this operation count. `Q` is the handwritten CuTe QuACK baseline; `Torch` is the timed PyTorch baseline. QuACK is the faster available baseline in every row.

Cases 0–3 “before” values are the original completed cold-search artifacts. Those searches selected small-K fallback configurations (`[256,32,16]` and `[128,16,16]`); the large narrow gains document automatic native-kernel discovery as well as the backend changes. The original wide cold search failed: the preserved failure log and scalar-kernel diagnosis are linked below. Cases 4–5 marked **†** therefore use the original **saved-config public AOT** receipts, not a completed original cold winner. Those before configurations use the pre-existing logical BK128; final cold searches keep the frontend’s registered K64 ceiling. Starting wide Torch throughput was not measured and is shown as unavailable.

| Case; M/E/N/K; partition | Before CuTe | Before Q | Before Torch | Before best/CuTe | V8 cold CuTe | V8 cold Q | V8 cold Torch | V8 best/CuTe | Backing raw files |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0; 2048/16/1536/2048; uniform | 0.243 | 481.355 | 161.061 | 0.0005044× | 483.667 | 481.922 | 161.092 | 1.003622× | [B0](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/before/family0.json) · [A0](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/after/v8-family0.json) |
| 1; 2048/16/1536/2048; skewed | 0.240 | 485.115 | 167.877 | 0.0004958× | 485.115 | 484.541 | 167.666 | 1.001186× | [B0](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/before/family0.json) · [A0](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/after/v8-family0.json) |
| 2; 8192/16/1536/2048; uniform | 0.241 | 1279.279 | 581.971 | 0.0001884× | 1290.038 | 1196.592 | 583.979 | 1.078094× | [B1](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/before/family1.json) · [A1](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/after/v8-family1.json) |
| 3; 8192/16/1536/2048; skewed | 0.239 | 988.703 | 558.271 | 0.0002420× | 1042.784 | 1014.704 | 558.852 | 1.027673× | [B1](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/before/family1.json) · [A1](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/after/v8-family1.json) |
| 4†; 8192/8/28672/4096; uniform | 1688.524 | 1688.381 | — | 1.0000842× | 1714.624 | 1708.192 | 1685.155 | 1.003765× | [BW†](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/before/aot-wide/gb300/forward/cute/K1/results.json) · [A2](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/after/v8-family2.json) |
| 5†; 8192/8/28672/4096; skewed | 1625.419 | 1606.614 | — | 1.0117048× | 1624.094 | 1622.538 | 1546.944 | 1.000959× | [BW†](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/before/aot-wide/gb300/forward/cute/K1/results.json) · [A2](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/after/v8-family2.json) |
| Means 0–3; ratio geomeans | 0.241 | 808.613 | 367.295 | 0.0003268× | 825.401 | 794.440 | 367.897 | 1.027186× | Derived from linked rows |
| Means 4–5†; ratio geomeans | 1656.971 | 1647.498 | — | 1.0058778× | 1669.359 | 1665.365 | 1616.049 | 1.002361× | Derived from linked rows |
| All-six means†; ratio geomeans | 552.484 | 1088.241 | 367.295 (4 rows) | 0.0047534× | 1106.720 | 1084.748 | 783.948 | 1.018844× | Derived from linked rows |

The final all-six row uses arithmetic means for TFLOPS and geometric means for ratios. Its before CuTe/QuACK columns combine four cold-search rows with two saved-AOT rows; the before Torch mean uses only four available rows, while final cold Torch uses six. The subgroup means keep those sample sets explicit. Before-to-after ratio changes also include each run’s own independently tuned QuACK winner.

**Protocol difference:** the original narrow harness used separate operand copies. V8 passes the original Case tuple into parent QuACK plans/search and shares those exact originals across CuTe/QuACK/PyTorch publication, with independent CPU byte/pointer checks outside timing. Isolated autotuner workers reconstruct argument snapshots; their pointer identity is not recorded. The narrow before receipts record four CUPTI/cold-L2/graph measurements per implementation, but omit per-record warmup/rep/heat settings and any thermal-cooldown target or outcome. V8 records four mirrored H/Q/T rounds with 100 ms warmup, 500 ms samples, 10 s thermal warmup and startup-temperature +3°C cooling targets. The historical before measurements therefore do not attest the same thermal protocol as V8. The wide before fallback uses the stock five-run public protocol. These mixed starting measurements document the original implementation/search outcome; acceptance comes from final cold and final public runs.

Preserved wide failure: [original cold log](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/logs/before-family2.log) · [scalar-kernel diagnosis](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/probes/before-family2-timeout-analysis.md). Complete derived values, raw hashes and formulas: [throughput metrics](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/probes/step45-report-review/throughput-metrics.v1.json).

## Final public AOT performance

Each case used the original public operation and an independent fresh QuACK cache/tuning directory. Helion AOT selected the saved family cold winner with runtime autotuning disabled. All numerical, exact source/version, config and cache-winner checks passed. The original stock protocol has five measurements for each implementation: 100 ms warmup, 500 ms samples, 10 s heat, cold L2, CUDA graphs, CUPTI and startup +3°C cooling. These are stock ordered measurements, not an interleaved ABAB run. Public receipts time QuACK only; the PyTorch numbers above come from the cold harness.

| Case | CuTe µs | Q µs | CuTe TFLOPS | Q TFLOPS | Q/CuTe | Result | Backing raw receipt |
|---|---:|---:|---:|---:|---:|---|---|
| 0 | 26.5920 | 26.7520 | 484.541 | 481.643 | 1.006017× | beats | [P0](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/aot/v8-case0/results/gb300/forward/cute/K1/results.json) |
| 1 | 26.8800 | 27.0080 | 479.349 | 477.077 | 1.004762× | beats | [P1](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/aot/v8-case1/results/gb300/forward/cute/K1/results.json) |
| 2 | 39.8400 | 45.2480 | 1293.665 | 1139.047 | 1.135743× | beats | [P2](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/aot/v8-case2/results/gb300/forward/cute/K1/results.json) |
| 3 | 49.2160 | 51.2960 | 1047.212 | 1004.749 | 1.042263× | beats | [P3](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/aot/v8-case3/results/gb300/forward/cute/K1/results.json) |
| 4 | 1096.7890 | 1117.6860 | 1754.344 | 1721.544 | 1.019053× | beats | [P4](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/aot/v8-case4/results/gb300/forward/cute/K1/results.json) |
| 5 | 1181.5410 | 1175.9750 | 1628.505 | 1636.213 | 0.995289× | matches within 1% | [P5](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/aot/v8-case5/results/gb300/forward/cute/K1/results.json) |
| Arithmetic means; ratio geomean | 403.4763 | 407.3275 | 1114.603 | 1076.712 | 1.032790× | all six ≥0.99; geomean ≥1 | Derived from six linked receipts |

The cold harness attests original input bytes after timing; ordinary public receipts save equal caller contracts and numerical checks, but do not store a post-timing byte snapshot. The independent final audit preserves that evidence boundary.

## General changes and verification

- Added typed device-worklist MN geometry, immutable A/B descriptors and expert-bounded rank3 B for feature tails, plus a proved bounded D pointer shift. Ownership, signed indexing, fresh-output, dtype and TMA alignment/resource proofs gate admission.
- Added optional worker balancing and device panel choices. The compact role prefix scheduler selects resource-derived M128 N192/N256 or M256 N224/N256 cores from live clipped row counts in one CUDA launch. Default-off and old M128 source bytes/seed/RNG behavior are preserved; no K1 name or shape hint authorizes lowering.
- Improved CuTe matmul graph timing and long-kernel probes, effective-source dedup/fresh retiming, coherent MultiShape steady selection, and compiler seed overflow/pinning. Optional reference latencies use the supported autotune_multi API with explicit cache identity.
- Scoped automatic Python GC suspension during graph capture prevents the observed CUDA-resource cleanup capture invalidation and restores the caller’s GC state.
- Published exactly three existing SM103 JSON entries, SHA 0e0b62810287b451cd37d6e8566604d2e4f5bb5d53a2069c7fdbbcfaf8254ffd. Kernel frontend/body, registered logical K64 limit, public operation/config pickers/AOT shims and SM100 configs remain unchanged.
- 143 distinct frozen V8 CPU config/source tests and two native CUDA live-graph tests passed; the shared harness passed 23 CPU tests. Coverage includes int32/int64 live offsets, empty/tail/skew routes, input/weight mutation, preserved default sources, and shared-memory/TMEM limits. Detailed proofs and earlier regression gates remain in the plan.

**Impact beyond CuTe:** source-cache retiming, compiler seed overflow/pinning, MultiShape final selection and optional reference-latency API touch shared autotuner/runtime code. Graph capture temporarily changes automatic GC state. CPU regressions cover the shared behavior, and the existing CC/MTIA changes were preserved in the main port. GPU performance evidence here is specific to GB300 K1 with this software stack; no other architecture or kernel speedup is claimed.

## Review history and replay

Step 4 is complete: the local Helion stack has four commits and the kernel-library stack has three, with the report, plan and progress docs in the final `[noland]` commit of each. The [Local history stacks and verification](#local-history-stacks-and-verification) section lists the code commits and validated code pin. The authoritative [stack manifest](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/probes/step45-history/stack-integration-v4/stack-manifest-v3.json) records the final tips and patch hashes, and the [verification summary](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/probes/step45-history/stack-integration-v4/check-summary-v3.json) records the passing checks. Original dirty checkouts and raw artifacts are preserved; nothing has been pushed.

Replay on a free GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash /home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/replay_v8.sh
```

Set `REPRO_CLEAN_CASES="0 1"` for a subset. [Replay](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/replay_v8.sh) · [Plan](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/plan.md) · [Progress](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/progress.md) · [Cold audit](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/probes/prefix-adaptive-v8-search-audit/audit-summary.v1.json) · [Public AOT audit](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/probes/prefix-adaptive-v8-search-audit/final-target-audit.v1.json) · [Source/replay audit](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/probes/final-v8-actual-main-audit/attestation.json).

## Archived final goal checker

[Exact checker receipt](/home/shangdiy/helion-kernel-library/artifacts/k1-cute-hillclimb-20261019/final-v8-checker.txt)

```text
PASS K1 case 0: M2048/E16/N1536/K2048 uniform: ratio 1.004 >= bar 0.990  [artifacts/k1-cute-hillclimb-20261019/after/v8-family0.json]
PASS K1 case 1: M2048/E16/N1536/K2048 skewed: ratio 1.001 >= bar 0.990  [artifacts/k1-cute-hillclimb-20261019/after/v8-family0.json]
PASS K1 case 2: M8192/E16/N1536/K2048 uniform: ratio 1.078 >= bar 0.990  [artifacts/k1-cute-hillclimb-20261019/after/v8-family1.json]
PASS K1 case 3: M8192/E16/N1536/K2048 skewed: ratio 1.028 >= bar 0.990  [artifacts/k1-cute-hillclimb-20261019/after/v8-family1.json]
PASS K1 case 4: M8192/E8/N28672/K4096 uniform: ratio 1.004 >= bar 0.990  [artifacts/k1-cute-hillclimb-20261019/after/v8-family2.json]
PASS K1 case 5: M8192/E8/N28672/K4096 skewed: ratio 1.001 >= bar 0.990  [artifacts/k1-cute-hillclimb-20261019/after/v8-family2.json]
PASS geomean: 1.0188 (min 1.0)
GOAL MET
```

## Local history stacks and verification

The local review branches preserve both original dirty mains and the measured frozen BFC V8 source. Nothing has been pushed. Raw measurements, autotuner output configs and the goal remain under the ignored run directory.

| Repository | Branch | Base | Logical code commits |
|---|---|---|---|
| Helion | `yushangdi/k1-cute-hillclimb-20261020` | `origin/main` / `cc1458c31` | benchmark `5ebf0d473`; native grouped worklists `3bf308c40`; atomic prefix recipes `1d1c592ad` |
| Kernel library | `k1-cute-hillclimb-20261020` | `origin/benchmark_cleanup` / `453b27a8` | benchmark `b64a9fbb`; three SM103 winners + code dependency `50a5b1c1` |

The library dependency pin is the last Helion code commit, `1d1c592ad40411eb6c4fe4176e0bc44c4efb0a6c`, before either final `[noland]` report commit. The clean detached tree at `artifacts/k1-cute-hillclimb-20261019/helion-history-production` passes the ordinary source-pin check with `REPRO_CLEAN_HELION_PATCH` unset. The private library environment changes only `helion.revision`; its SM103 JSON remains exactly `0e0b62810287b451cd37d6e8566604d2e4f5bb5d53a2069c7fdbbcfaf8254ffd`, with three exact cold winner rows and 18 unchanged rows. All SM100 rows, frontend code and shape data are preserved.

Native K1 compiler and launcher files match the measured frozen V8 source byte-for-byte. The complete 647-file package proof has 644 exact files, two generic benchmark fixes and one AST-identical formatter wrap. The fixes preserve environment-only legacy graph timing and reject failed/unavailable readings before source-cache refresh. Recorded K1 timers requested the same graph policy, and public AOT timed calls do not use the source-retiming path. The performance tables remain attributed to the original frozen BFC V8 receipts; no new rewritten-stack performance measurement is claimed.

### Verification record

Verification uses the clean CI environment: Torch 2.14.1+cu130, Triton 3.8.0, project-pinned CUTLASS DSL 4.7.0, CUDA bindings 13.0.3 and JAX 0.11.2. The original performance environment remains unchanged at CUTLASS DSL 4.8.0.

| Check | Result |
|---|---|
| Initial full default suite v3 | 15 failed / 18,531 passed / 12,030 skipped / 3,247 subtests passed; historical failure retained |
| Initial full CuTe suite v3 | 37 failed / 29,159 passed / 1,376 skipped / 4,675 subtests passed; historical failure retained |
| Repaired benchmark failures and regressions | 35 default + 35 CuTe passed on separate GPUs; source held |
| Repaired native failures and GPU fixture | 19 default passed + 54 existing CuTe-only skips; 73 CuTe passed + 3 subtests, no skips |
| Atomic prefix source/config/CUDA gate | 159 passed, no skips; source held |
| Final `./lint.sh check` v4 | passed, 0 type errors; existing project warnings/suppressions unchanged |
| Final full default v4 | 18,553 passed / 12,031 skipped / 3,247 subtests passed; zero failures, source held |
| Final full CuTe v4 | 29,209 passed / 1,368 skipped / 4,682 subtests passed; zero failures, source held |
| Normal committed-code pin and K1 Config transfer | passed; 28 existing checks and exact Config objects/roundtrips for all six cases, K64, no tuning or CUDA work |
| Extended non-K1 library picker module | 316 passed / 12 failures on byte-identical baseline expectations; suffix `_triton`/`_packed` and existing mxfp8 SM100 selection expectations documented separately |
| Independent final review | LGTM; exact package/seam, guards, transfer and repaired coverage sealed |

Detailed patches, inventories, source-pin proof, repair logs and check receipts are under `artifacts/k1-cute-hillclimb-20261019/probes/step45-history/stack-integration-v4/`. Final stack manifests and mail patch series record the code commits and final `[noland]` heads separately. No submitted source commit references those planning/report files.
