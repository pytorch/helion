# K3 CuTe hillclimb results

**Result:** CuTe is fastest for all six requested K3 shapes on GPU 3. All six cold full searches, six actual public CUDA tests and twelve installed public benchmark workers pass. Cold geometric mean speedup is **1.0704×**; final public verification is **1.0679×** against the fastest observed competitor.

Scope: K3 only on GPU 3. Case 1 is **M=8, N=8192, K=1280**.

## Cold full performance

All after timings below use GPU 3, GB300/SM103, 152 SMs and 1400 W. Each search is fresh full effort with a new seed, cache bypass and no time budget. Ratios use paired canonical competitor/CuTe latency.

| Case / M×N×K | CuTe before (µs) | Before GPU | CuTe cold after (µs) | Native after (µs) | Triton after (µs) | Native/CuTe | Triton/CuTe | Raw evidence |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0 / 1×1792×5120 | 13.280 | 0 | 7.712 | 8.544 | 13.520 | 1.1079× | 1.7531× | [before](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case0/result.json) · [after](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case0-cold-r2/case0/result.json) |
| 1 / 8×8192×1280 | 18.977 | 3 | 5.824 | 6.272 | 18.689 | 1.0769× | 3.2089× | [before](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/case1-m8-starting-aot-r1/results/gb300/forward/cute/K3/K3_01.json) · [after](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case1-m8-cold-r1/case1/result.json) |
| 2 / 16×2048×4096 | 12.832 | 0 | 7.104 | 7.680 | 12.512 | 1.0811× | 1.7613× | [before](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case2/result.json) · [after](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case2-cold-r1/case2/result.json) |
| 3 / 32×5120×2048 | 13.312 | 0 | 6.400 | 6.752 | 33.328 | 1.0550× | 5.2076× | [before](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case3/result.json) · [after](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case3-cold-r1/case3/result.json) |
| 4 / 128×1792×5120 | 16.992 | 0 | 8.128 | 8.672 | 47.968 | 1.0669× | 5.9016× | [before](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case4/result.json) · [after](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case4-cold-r1/case4/result.json) |
| 5 / 1024×8192×1280 | 19.104 | 0 | 11.552 | 11.968 | 25.504 | 1.0360× | 2.2078× | [before](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case5/result.json) · [after](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case5-coverage-cold-r2/case5/result.json) |
| Mean latency / geometric mean ratio | 15.749† | mixed† | 7.787 | 8.315 | 25.253 | 1.0704× | 2.9598× | [formulas and hashes](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/final-cold-facts-r1.json) |

† Before values are historical public CuTe requests: cases 0/2/3/4/5 were recorded on GPU 0, and replacement M8 case 1 on GPU 3. Before routes 0/1/2 used Triton fallback;3/4/5 used scale unpack plus CuTe GEMM. The before mean describes those historical rows. Performance ratios use the same-GPU 3 after pairs.

## Historical baseline rows

| Case | CuTe request (µs) | Native (µs) | Triton (µs) | Actual requested-CuTe route | Raw |
| ---: | ---: | ---: | ---: | :--- | :--- |
| 0 | 13.280 | 8.768 | 13.312 | Triton fallback | [JSON](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case0/result.json) |
| 1 | 18.977 | 6.272 | 18.976 | Triton fallback | [JSON](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/case1-m8-starting-aot-r1/results/gb300/forward/cute/K3/K3_01.json) |
| 2 | 12.832 | 7.904 | 12.832 | Triton fallback | [JSON](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case2/result.json) |
| 3 | 13.312 | 6.977 | 34.112 | CuTe scale unpack + CuTe GEMM | [JSON](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case3/result.json) |
| 4 | 16.992 | 8.896 | 31.232 | CuTe scale unpack + CuTe GEMM | [JSON](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case4/result.json) |
| 5 | 19.104 | 12.192 | 25.632 | CuTe scale unpack + CuTe GEMM | [JSON](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case5/result.json) |
| Mean historical latency | 15.749 | 8.502 | 22.683 | mixed historical routes/GPUs | — |

## Implementation and verification

- Original packed kernel executable body, masks, allocations, partition and numerical contract are preserved. A private CuTe AOT binding owns its JSON configs and picker.
- General backend lowering supports the registered task grid, packed scale transfer, unit-row tensor maps and in-place scale buffers. Larger outputs use register prefetch with two TMA output stages.
- Compiler coverage adds the joint scale/output-mode profiles after the normal initial-population cap. The last cold search finds the improvement from the original starting table.
- All cold checks cover eager output, poisoned graph output, four changed live inputs, restored inputs and complete input storage. Six public CUDA cases independently pass with no skipped cases or runtime search.

CUDA receipt: [six actual passed cases](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-cuda-correctness-r1/receipt.json)

## Final installed public benchmark

These results come from the unchanged `benchmark.sh gb300 both K3` with guarded canonical timing and exact saved AOT configurations. The native column uses the faster native observation across the two backend workers.

| Case | CuTe (µs) | Native best (µs) | Original Triton (µs) | Best baseline/CuTe | Raw public results |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 0 | 7.712 | 8.544 | 13.280 | 1.1079× | [CuTe](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/results/gb300/forward/cute/K3/K3_00.json) · [Triton](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/results/gb300/forward/triton/K3/K3_00.json) |
| 1 | 5.824 | 6.272 | 18.976 | 1.0769× | [CuTe](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/results/gb300/forward/cute/K3/K3_01.json) · [Triton](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/results/gb300/forward/triton/K3/K3_01.json) |
| 2 | 7.104 | 7.680 | 12.576 | 1.0811× | [CuTe](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/results/gb300/forward/cute/K3/K3_02.json) · [Triton](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/results/gb300/forward/triton/K3/K3_02.json) |
| 3 | 6.432 | 6.720 | 33.920 | 1.0448× | [CuTe](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/results/gb300/forward/cute/K3/K3_03.json) · [Triton](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/results/gb300/forward/triton/K3/K3_03.json) |
| 4 | 8.128 | 8.704 | 31.041 | 1.0709× | [CuTe](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/results/gb300/forward/cute/K3/K3_04.json) · [Triton](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/results/gb300/forward/triton/K3/K3_04.json) |
| 5 | 11.552 | 11.872 | 25.248 | 1.0277× | [CuTe](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/results/gb300/forward/cute/K3/K3_05.json) · [Triton](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/results/gb300/forward/triton/K3/K3_05.json) |
| Mean latency / geometric mean ratio | 7.792 | 8.299 | 22.507 | 1.0679× | [verified formulas and hashes](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/root-final-review.json) |

[Six actual CUDA passes](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-cuda-correctness-r1/receipt.json) · [Twelve public workers](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/final-main-k3-all6-aot-r1/receipt.json) · [GPU 3 release](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/read_only_pipeline_review_k6/final_main_k3_cuda_aot_release.json)

Every worker verifies actual backend/source/shim, exact saved configuration, no runtime search, native recipe, original CUPTI timer, canonical settings, compiler/library HEAD and frozen source files. All 24 timing calls were accepted without overlap.


## Versions and method

Python 3.12.13; Torch 2.14.1+cu130; Triton 3.8.0; CUTLASS DSL 4.8.0; DeepGEMM 2.8.1 (linked source 057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7); FlashInfer 0.7.2; CUPTI 13.0.1.

Library base 453b27a8cb1a4a98bf068303d0348d6c0037eecb; Helion base bfc25902f558b036e2c602be72d31d1c04fe4363. Cases0–4 use compiler f9c19ca6; case5 uses coverage successor e9495caa. All nine math/runtime files are unchanged between them; installed backend patch 18a14109 adds two coverage files to the eleven-file declaration.

Canonical verification uses five runs per window, 100 ms warmup, 500 ms repetitions, 10 s thermal warmup, CUDA graphs and cold L2. Each competitor has two interleaved candidate/competitor windows. Native recipe is (1, 1, 32); packed-scale preparation is performed before the timed calls.

[Final checker: GOAL MET](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/goal-check-final-all-verified.txt) · [Main application](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/integration-final-r4/root-main-apply.json) · [Installed source provenance](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/integration-final-r4/root-main-backend-install.json)

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=3 REPRO_CLEAN_PYTHON=repro_clean/.venv/gb300-main/bin/python \
  bash repro_clean/benchmark.sh gb300 both K3
```

Setup for a clean environment is documented in [K3_PACKED_CUTE.md](/home/shangdiy/helion-kernel-library-benchmark_cleanup/repro_clean/K3_PACKED_CUTE.md).

## Final checker

```text
PASS K3/0: ratio 1.108 >= bar 0.990  [artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case0-cold-r2/case0/result.json]
PASS K3/1: ratio 1.077 >= bar 0.990  [artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case1-m8-cold-r1/case1/result.json]
PASS K3/2: ratio 1.081 >= bar 0.990  [artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case2-cold-r1/case2/result.json]
PASS K3/3: ratio 1.055 >= bar 0.990  [artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case3-cold-r1/case3/result.json]
PASS K3/4: ratio 1.067 >= bar 0.990  [artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case4-cold-r1/case4/result.json]
PASS K3/5: ratio 1.036 >= bar 0.990  [artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case5-coverage-cold-r2/case5/result.json]
PASS geomean: 1.0704 (min 1.0)
GOAL MET
```


## Logical GEMM throughput

TFLOP/s = 2 × M × N × K / (latency in µs × 1,000,000). This counts the logical GEMM work. Before measurements use the saved historical GPUs/backend routes; after values use canonical cold pairs on GPU 3.

| Case / M×N×K | Before CuTe request | Before native | Before Triton | After CuTe | After native | After Triton | Raw before / after |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0 / 1×1792×5120 | 1.382 | 2.093 | 1.378 | 2.379 | 2.148 | 1.357 | [before](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case0/result.json) · [after](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case0-cold-r2/case0/result.json) |
| 1 / 8×8192×1280 | 8.841 | 26.749 | 8.841 | 28.807 | 26.749 | 8.977 | [before](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/production/case1-m8-starting-aot-r1/results/gb300/forward/cute/K3/K3_01.json) · [after](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case1-m8-cold-r1/case1/result.json) |
| 2 / 16×2048×4096 | 20.919 | 33.962 | 20.919 | 37.787 | 34.951 | 21.454 | [before](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case2/result.json) · [after](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case2-cold-r1/case2/result.json) |
| 3 / 32×5120×2048 | 50.412 | 96.186 | 19.673 | 104.858 | 99.391 | 20.136 | [before](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case3/result.json) · [after](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case3-cold-r1/case3/result.json) |
| 4 / 128×1792×5120 | 138.230 | 264.030 | 75.205 | 288.978 | 270.850 | 48.966 | [before](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case4/result.json) · [after](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case4-cold-r1/case4/result.json) |
| 5 / 1024×8192×1280 | 1124.102 | 1761.388 | 837.814 | 1858.971 | 1794.355 | 842.018 | [before](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/before-aot/case5/result.json) · [after](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/compiled-original/gpu3/original-final-case5-coverage-cold-r2/case5/result.json) |
| Arithmetic mean TFLOP/s | 223.981 | 364.068 | 160.638 | 386.963 | 371.407 | 157.151 | logical2MNK / saved latency |

## Updated combined chart

[PNG](/home/shangdiy/helion-kernel-library-benchmark_cleanup/repro_clean/results/benchmark_20261020T025310Z_2669525_K3_updated/speedups.png) · [PDF](/home/shangdiy/helion-kernel-library-benchmark_cleanup/repro_clean/results/benchmark_20261020T025310Z_2669525_K3_updated/speedups.pdf) · [SVG](/home/shangdiy/helion-kernel-library-benchmark_cleanup/repro_clean/results/benchmark_20261020T025310Z_2669525_K3_updated/speedups.svg) · [provenance](/home/shangdiy/helion-kernel-library-benchmark_cleanup/repro_clean/results/benchmark_20261020T025310Z_2669525_K3_updated/chart_update_receipt.json)

K3 CuTe chart label is **1.069×**. The chart follows the existing per-worker baseline/CuTe formula (geometric mean **1.068797×**); the conservative final report uses the faster native observation across both worker pairs (**1.067891×**). All 156 non-K3 records are preserved.

## Shared-code and dependency considerations

- Shared `helion/autotuner/config_spec.py` and `helion/runtime/cute/launcher.py` participate in config admission and argument preparation. The new optimized path is guarded by packed-MXFP8 structure, metadata, resource and complete-reduction proofs. The full default test suite checks effects on existing users.
- Scale transfers, shared-buffer lifetimes and TMA output stores depend on the CuTe/CUTLASS DSL API. GB300 correctness and performance use the benchmark environment’s DSL 4.8.0. Upstream CI declares DSL 4.7.0; static checking uses that declared dependency separately, while full GPU suite compatibility is reported explicitly.
- CUDA evidence covers GPU 3 GB300/SM103, 152 SMs, 1400 W. Other GPU architectures and framework versions have separate configuration and validation requirements.
- An explicitly forced oversized S12/N224/K128/C2 profile is rejected before native compilation/launch. The supported measured S6 profile uses 211,968 B, within 232,448 B.
- The original packed kernel executable body and public signature remain intact; source-filename metadata supplies an independent CuTe AOT shim and keyed JSON picker.


## Step 4 local history

Branch: `k3-cute-hillclimb-clean-20261019`. Base: `origin/main` at `ceafab07aab63876d87af36b5a7286cfb060ea0f`.

| Order | Commit | Subject |
| --- | --- | --- |
| 1 | `8bc29759848f880301c2bba744b2ee8af4629c6a` | [cutedsl] extend native MXFP8 for packed scales and registered grids |
| 2 | `ad8b5695884a2d6f24ff81d0a1fd3fe7499b50b9` | [cutedsl] add TMA and in-place staging for packed MXFP8 scales |
| 3 | `14a4f0d74080ce606640614e49dd45afd3cafc86` | [cutedsl] add optional MXFP8 accumulator and TMA output modes |
| 4 | `fa7bf8115920e242c56f064ef1384bffe59317b6` | [cutedsl] measure joint MXFP8 output modes after the initial population cap |
| 5 | `b8c35144d63ee7d46fbe9ce0d20e1f93876465c8` | [cutedsl] Validate MXFP8 metadata and configuration choices |

The backend source is frozen at `b8c35144d63ee7d46fbe9ce0d20e1f93876465c8`. The amendment changes only an existing test’s exact expected MXFP8 tile vocabulary; all 11 backend files and 16 previous file hashes are unchanged. The full default run tested `1246698e`; the corrected CuTe run tests `b8c35144`. The changed test module is excluded from the default backend by its existing marker. A single final `[noland]` commit contains the plan, report and plots; the backend commits have no dependency on those files. Raw results, saved autotuner configs and the goal file remain uncommitted. No remote PRs have been submitted.

## Full backend verification

All four `cute-verify` checks pass.

| Check | Result | Evidence |
| :--- | :--- | :--- |
| `./lint.sh check` | PASS: Ruff format/check, codespell, Pyrefly 0 errors | [Final committed-source lint](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/step4-verify/lint-r4-config-coverage-frozen-receipt.json) |
| `pytest test/ -q --tb=line` | 18,071 passed / 12,053 skipped; 3,245 subtests passed | [Full default receipt](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/review-stack-verification/final-clean-stack-ci47-default-full-r1/receipt.json) |
| `HELION_BACKEND=cute pytest test/ -q --tb=line` | 28,724 passed / 1,393 skipped / 0 xfail; 4,680 subtests passed | [Full CuTe receipt](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/review-stack-verification/final-clean-stack-ci47-cute-full-r2/receipt.json) |
| Code review | LGTM, including the configuration-vocabulary update | [Original review](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/step4-verify/code-review-final-1246698e-r1.md) · [Amendment review](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/step4-verify/code-review-addendum-b8c35144-r1.json) |

Both full suites finish with exit 0, zero JUnit failures/errors, complete source/dependency/main-runtime audits, and GPU 3 released. Default has 156 guard rows and CuTe has 216; both have zero foreign-process detections. The full commands retain the complete `test/` scope, with no global autotuning-effort or cache override. GPU 3 ran the suites sequentially.

The default suite tested `1246698e69196357879da855cf132f7be5fd609c`; the corrected CuTe suite tested `b8c35144d63ee7d46fbe9ce0d20e1f93876465c8`. The sole change between them is an existing test's exact expected MXFP8 N-tile tuple. All 11 backend files and 16 prior file hashes are unchanged. The module's existing marker excludes it from the default backend. The update retains exact ordering, NVFP4 rejection, N64 normalization, generated-code and configuration roundtrip checks. The final CuTe run executes and passes the updated test.

Default JUnit reconciles 30,124 testcase rows plus 3,245 subtests. CuTe reconciles 30,117 rows plus 4,680 subtests. Default skips are mostly CuTe-only backend exclusions (11,368); CuTe skips are mostly default-backend exclusions (836), macOS Metal tests (153) and Triton-specific pipelines. Remaining skips include unavailable optional JAX/Pallas, FlyDSL and multiple-accelerator coverage. Full `-ra` skip reasons are preserved in the raw logs.

### Dependency setup and coverage

Full suite validation uses the repository's declared CUTLASS DSL **4.7.0** in an isolated test environment, with pytest 9.1.1 installed inside its site packages. Torch 2.14.1+cu130, Triton 3.8.0 and NumPy 2.5.3 reuse the installed benchmark package files. SDK 4.7's protobuf 6.33.6 and nvdisasm 13.4.92 dependencies are isolated. The main benchmark environment retains SDK **4.8.0** and all 172 package versions.

The initial full SDK 4.8 diagnostic found eight inherited SDK compatibility failures. CPU API, IR and model probes reproduced their underlying causes on `origin/main` and the K3 stack. Two child-process tests also lost pytest when replacing `PYTHONPATH`. The supported environment resolves all ten cases, with 10 passes and no skips. The first CuTe run then found the old exact N-tile expectation; the test-only update reflects the new legal MXFP8 domain while retaining the NVFP4 domain. No test was removed or disabled.

K3's cold searches, six public CUDA cases, twelve public benchmark workers and native CUBIN equivalence proofs remain labeled with their actual SDK **4.8.0**. The complete supported SDK 4.7 suites verify the cleanup branch. These results do not claim complete SDK 4.8 compatibility for unrelated existing Helion operations or coverage on other accelerator platforms.

[Targeted ten-case confirmation](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/review-stack-verification/final-clean-stack-ci47-target10-r1/receipt.json) · [SDK attribution](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/step4-verify/r5-sdk-attribution-k4-r1/REVIEW-r2.md) · [Atomic SDK attribution](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/step4-verify/r5-atomic-sdk-attribution-k4-r1/REVIEW.md).

The initial diagnostic monitoring gap is preserved in its failure/release audit. Final full suites use validated ancestry tracking for detached autotuner workers, durable per-test failure records and complete JSON-line monitoring; their uninterrupted guard receipts pass.

[Four-check independent audit](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/step4-verify/root-four-checks-final-r1.json) · [Isolated test runtime manifest](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/review-stack-verification/test-runtime-v5-manifest.json) · [Step 5 table formulas and raw hashes](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/K3/step5-performance-tables-r1.json) · [Final checker: GOAL MET](/home/shangdiy/helion-kernel-library-benchmark_cleanup/artifacts/cute-fastest-20261019/goal-check-step5-complete-r1.txt).
