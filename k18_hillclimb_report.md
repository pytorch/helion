# K18 MXFP8 hillclimb report

All six published BF16/TorchAO RCEIL shapes meet the 1% match criterion against the best measured baseline. The same-GPU geometric mean is **1.0051×**; four shapes are faster and two are within the match margin.

## Measurement setup

| | Trusted before context | Final comparison |
| --- | --- | --- |
| Host | `devgpu042.lco3.facebook.com` | `devgpu042.lco3.facebook.com` |
| GPU | GB300, physical GPU 1, SM103 | GB300, physical GPU 2, SM103, 152 SMs |
| GPU UUID | `GPU-7910a120-6368-4db1-a571-d4ce1b1e02a4` | `GPU-e4ee5998-db2d-edd0-3a15-956b96a5fa93` |
| Power cap | Not recorded in historical receipts | 1400 W, recorded in every receipt |
| Helion | Clean `063d8f58f4a3898d9de60e23620a2f18e44825a5` | `a03e0f28406db93df8d6e9a5fd03bf995dfcb16c` plus the accepted changes |
| Timing | CUPTI, captured CUDA graph, cold L2 | Same CUPTI function, captured CUDA graph, cold L2 |

Both environments use PyTorch **2.14.0+cu130**, Triton **3.8.0**, CUTLASS DSL **4.7.0**, FlashInfer **0.7.0**, and Transformer Engine **2.20.2**. Historical receipts also record driver **580.126.20** and CUPTI **13.0.1**; the raw version snapshots are linked by the [receipt ledger and calculations](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/report_audit.json).

The user requested historical starting numbers and no cold autotuning. The initial cold campaign was stopped; optimization used short pinned probes. Final public AOT calls forbid search, verify the bound configuration, and save the actually launched source. GPU work after the device restriction used GPU 2.

For each final shape, A1/B1/A2/B2 ran in separate processes on GPU 2, with five runs of 200 ms per campaign and the same input hash and timer. The reported statistic is the median of the two campaign medians for each implementation. The paired winner and CuTe use this ABAB result; other providers use the archived same-GPU anchors. Provider timings include the complete timed call and its output adaptations. The before column is historical GPU 1 context; no cross-GPU speedup is inferred.

## Per-shape latency

Times are microseconds. Each numeric cell links to its backing receipt. `Best / CuTe` is baseline time divided by CuTe time on GPU 2; values above one favor CuTe. All absolute-value summary columns use unweighted arithmetic means across the six shapes; the ratio column uses a geometric mean.

| Shape | Before CuTe (GPU 1) | After CuTe (GPU 2) | Triton (GPU 2) | FlashInfer (GPU 2) | TE (GPU 2) | Compiled Torch (GPU 2) | Paired best | Best / CuTe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1x128 | [1.7600](artifacts/k18-mxfp8-2026-10-14/historical/K18_00_cute.json) | [1.7280](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case00/helion-cute/result.json) | [1.7600](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case00/helion-triton/result.json) | [1.8880](artifacts/k18-mxfp8-2026-10-14/small_gpu2/baselines/case00/flashinfer-cute/result.json) | [4.4800](artifacts/k18-mxfp8-2026-10-14/small_gpu2/baselines/case00/transformer-engine-cuda/result.json) | [1.7600](artifacts/k18-mxfp8-2026-10-14/small_gpu2/baselines/case00/torch-reference-compile/result.json) | Triton | [1.018519×](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case00/helion-cute/comparison_4b3359c46b78.json) |
| 137x64 | [1.9200](artifacts/k18-mxfp8-2026-10-14/historical/K18_01_cute.json) | [1.7920](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case01/helion-cute/result.json) | [1.8080](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case01/helion-triton/result.json) | [1.9200](artifacts/k18-mxfp8-2026-10-14/small_gpu2/baselines/case01/flashinfer-cute/result.json) | [7.7760](artifacts/k18-mxfp8-2026-10-14/small_gpu2/baselines/case01/transformer-engine-cuda/result.json) | [1.8880](artifacts/k18-mxfp8-2026-10-14/small_gpu2/baselines/case01/torch-reference-compile/result.json) | Triton | [1.008929×](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case01/helion-cute/comparison_fe4a00cda375.json) |
| 128x1024 | [2.1760](artifacts/k18-mxfp8-2026-10-14/historical/K18_02_cute.json) | [1.8725](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case02/helion-cute/result.json) | [1.9200](artifacts/k18-mxfp8-2026-10-14/small_gpu2/baselines/case02/helion-triton/result.json) | [1.9840](artifacts/k18-mxfp8-2026-10-14/small_gpu2/baselines/case02/flashinfer-cute/result.json) | [1.8560](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case02/transformer-engine-cuda/result.json) | [2.6880](artifacts/k18-mxfp8-2026-10-14/small_gpu2/baselines/case02/torch-reference-compile/result.json) | TE | [0.991188×](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case02/helion-cute/comparison_565d0dee0edd.json) |
| 4096x4096 | [19.8400](artifacts/k18-mxfp8-2026-10-14/historical/K18_03_cute.json) | [8.4800](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case03/helion-cute/result.json) | [10.2410](artifacts/k18-mxfp8-2026-10-14/large_gpu2/baselines/case03/helion-triton/result.json) | [10.6240](artifacts/k18-mxfp8-2026-10-14/large_gpu2/baselines/case03/flashinfer-cute/result.json) | [8.5760](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case03/transformer-engine-cuda/result.json) | [10.2400](artifacts/k18-mxfp8-2026-10-14/large_gpu2/baselines/case03/torch-reference-compile/result.json) | TE | [1.011321×](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case03/helion-cute/comparison_b22b34688d91.json) |
| 8192x8192 | [72.0000](artifacts/k18-mxfp8-2026-10-14/historical/K18_04_cute.json) | [31.7925](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case04/helion-cute/result.json) | [31.7605](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case04/helion-triton/result.json) | [35.2960](artifacts/k18-mxfp8-2026-10-14/large_gpu2/baselines/case04/flashinfer-cute/result.json) | [31.8080](artifacts/k18-mxfp8-2026-10-14/large_gpu2/baselines/case04/transformer-engine-cuda/result.json) | [31.9040](artifacts/k18-mxfp8-2026-10-14/large_gpu2/baselines/case04/torch-reference-compile/result.json) | Triton | [0.998993×](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case04/helion-cute/comparison_2b19780bfbcf.json) |
| 8192x16384 | [140.6090](artifacts/k18-mxfp8-2026-10-14/historical/K18_05_cute.json) | [60.3365](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case05/helion-cute/result.json) | [60.4800](artifacts/k18-mxfp8-2026-10-14/large_gpu2/baselines/case05/helion-triton/result.json) | [67.7435](artifacts/k18-mxfp8-2026-10-14/large_gpu2/baselines/case05/flashinfer-cute/result.json) | [60.4480](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case05/transformer-engine-cuda/result.json) | [60.5440](artifacts/k18-mxfp8-2026-10-14/large_gpu2/baselines/case05/torch-reference-compile/result.json) | TE | [1.001848×](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case05/helion-cute/comparison_bb9e0b6f15fe.json) |
| [Mean time / GM ratio](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/report_audit.json) | 39.7175 | 17.6669 | 17.9949 | 19.9093 | 19.1573 | 18.1707 | per shape | 1.005094× |

The 128×1024 and 8192×8192 ratios are below one and above the checker’s 0.99 threshold. The completion claim uses that declared noise margin.

### Historical baseline context

These four providers were measured with the historical CuTe column on GPU 1. They are retained for the complete starting comparison; final completion uses GPU 2. Historical `Best / CuTe` uses the fastest of these four historical providers.

| Shape | Triton (GPU 1) | FlashInfer (GPU 1) | TE (GPU 1) | Compiled Torch (GPU 1) | Historical best / CuTe |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1x128 | [1.7280](artifacts/k18-mxfp8-2026-10-14/historical/K18_00_triton.json) | [1.8560](artifacts/k18-mxfp8-2026-10-14/historical/K18_00_baseline.json) | [4.9280](artifacts/k18-mxfp8-2026-10-14/historical/transformer-engine-cuda/K18_00_baseline.json) | [1.7280](artifacts/k18-mxfp8-2026-10-14/historical/torch-reference-compile/K18_00_baseline.json) | [0.981818×](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/report_audit.json) |
| 137x64 | [1.8240](artifacts/k18-mxfp8-2026-10-14/historical/K18_01_triton.json) | [1.9200](artifacts/k18-mxfp8-2026-10-14/historical/K18_01_baseline.json) | [9.3120](artifacts/k18-mxfp8-2026-10-14/historical/transformer-engine-cuda/K18_01_baseline.json) | [2.5280](artifacts/k18-mxfp8-2026-10-14/historical/torch-reference-compile/K18_01_baseline.json) | [0.950000×](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/report_audit.json) |
| 128x1024 | [2.0160](artifacts/k18-mxfp8-2026-10-14/historical/K18_02_triton.json) | [2.0800](artifacts/k18-mxfp8-2026-10-14/historical/K18_02_baseline.json) | [1.9530](artifacts/k18-mxfp8-2026-10-14/historical/transformer-engine-cuda/K18_02_baseline.json) | [2.0160](artifacts/k18-mxfp8-2026-10-14/historical/torch-reference-compile/K18_02_baseline.json) | [0.897518×](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/report_audit.json) |
| 4096x4096 | [10.2080](artifacts/k18-mxfp8-2026-10-14/historical/K18_03_triton.json) | [10.6560](artifacts/k18-mxfp8-2026-10-14/historical/K18_03_baseline.json) | [8.6400](artifacts/k18-mxfp8-2026-10-14/historical/transformer-engine-cuda/K18_03_baseline.json) | [10.2080](artifacts/k18-mxfp8-2026-10-14/historical/torch-reference-compile/K18_03_baseline.json) | [0.435484×](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/report_audit.json) |
| 8192x8192 | [32.1290](artifacts/k18-mxfp8-2026-10-14/historical/K18_04_triton.json) | [35.1680](artifacts/k18-mxfp8-2026-10-14/historical/K18_04_baseline.json) | [31.9690](artifacts/k18-mxfp8-2026-10-14/historical/transformer-engine-cuda/K18_04_baseline.json) | [35.2330](artifacts/k18-mxfp8-2026-10-14/historical/torch-reference-compile/K18_04_baseline.json) | [0.444014×](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/report_audit.json) |
| 8192x16384 | [60.6080](artifacts/k18-mxfp8-2026-10-14/historical/K18_05_triton.json) | [68.1930](artifacts/k18-mxfp8-2026-10-14/historical/K18_05_baseline.json) | [60.3200](artifacts/k18-mxfp8-2026-10-14/historical/transformer-engine-cuda/K18_05_baseline.json) | [60.2240](artifacts/k18-mxfp8-2026-10-14/historical/torch-reference-compile/K18_05_baseline.json) | [0.428308×](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/report_audit.json) |
| [Mean time / GM ratio](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/report_audit.json) | 18.0855 | 19.9788 | 19.5203 | 18.6562 | 0.640945× |

### Nominal bandwidth

Traffic is BF16 input + FP8 payload + one uint8 scale per 32 elements: **3.03125 bytes per element**. GB/s = bytes / elapsed seconds / 10⁹. This is contract traffic; it is not a measured DRAM counter and does not count extra provider padding or conversion traffic. The same timing receipts back these calculated values; [receipt ledger and calculations](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/report_audit.json) records the bytes and formula for every row.

| Shape | Before CuTe GB/s (GPU 1) | After CuTe GB/s (GPU 2) | Paired best GB/s (GPU 2) |
| --- | ---: | ---: | ---: |
| 1x128 | [0.220](artifacts/k18-mxfp8-2026-10-14/historical/K18_00_cute.json) | [0.225](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case00/helion-cute/result.json) | [0.220](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case00/helion-triton/result.json) |
| 137x64 | [13.843](artifacts/k18-mxfp8-2026-10-14/historical/K18_01_cute.json) | [14.831](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case01/helion-cute/result.json) | [14.700](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case01/helion-triton/result.json) |
| 128x1024 | [182.588](artifacts/k18-mxfp8-2026-10-14/historical/K18_02_cute.json) | [212.183](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case02/helion-cute/result.json) | [214.069](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case02/transformer-engine-cuda/result.json) |
| 4096x4096 | [2563.303](artifacts/k18-mxfp8-2026-10-14/historical/K18_03_cute.json) | [5997.162](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case03/helion-cute/result.json) | [5930.030](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case03/transformer-engine-cuda/result.json) |
| 8192x8192 | [2825.330](artifacts/k18-mxfp8-2026-10-14/historical/K18_04_cute.json) | [6398.482](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case04/helion-cute/result.json) | [6404.929](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case04/helion-triton/result.json) |
| 8192x16384 | [2893.467](artifacts/k18-mxfp8-2026-10-14/historical/K18_05_cute.json) | [6742.975](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case05/helion-cute/result.json) | [6730.537](artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case05/transformer-engine-cuda/result.json) |
| [Arithmetic mean](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/report_audit.json) | 1413.125 | 3227.643 | 3215.748 |

## Goal checker

[Complete saved checker output](artifacts/k18-mxfp8-2026-10-14/goal_final_completion.txt):

```text
PASS 1x128: ratio 1.019 >= bar 0.990  [artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case00/helion-cute/comparison_4b3359c46b78.json]
PASS 137x64: ratio 1.009 >= bar 0.990  [artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case01/helion-cute/comparison_fe4a00cda375.json]
PASS 128x1024: ratio 0.991 >= bar 0.990  [artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case02/helion-cute/comparison_565d0dee0edd.json]
PASS 4096x4096: ratio 1.011 >= bar 0.990  [artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case03/helion-cute/comparison_b22b34688d91.json]
PASS 8192x8192: ratio 0.999 >= bar 0.990  [artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case04/helion-cute/comparison_2b19780bfbcf.json]
PASS 8192x16384: ratio 1.002 >= bar 0.990  [artifacts/k18-mxfp8-2026-10-14/final_abab_v1/aggregate/case05/helion-cute/comparison_bb9e0b6f15fe.json]
PASS geomean: 1.0051 (min 1.0)
GOAL MET
```

## Accepted changes

- **Kernel:** index a `(groups, 32)` view to expose contiguous BF16 packets, retain FP32 maxima, use native upward E8M0 conversion for finite TorchAO RCEIL input on SM100+, and allocate scales before the FP8 payload. Other recipes keep their existing rounding, zero-scale and clamping paths.
- **Packet I/O:** general CuTe FP8/integer vector stores and late FP8 conversion/store fusion reduce scalar instructions. Declared-pure scalar inline assembly can move through persistent row passes only after the existing opcode and effect proof succeeds.
- **Cache choices:** `load_eviction_policies="nc_l2_first"` and `cute_store_policy="l2_evict_last"` expose separately selectable input/output policies for proven packets. Read-only loads require a whole-kernel storage/alias proof; tails and unsupported sites retain ordinary accesses.
- **Search and bounds:** general seeds now cover short persistent reductions, wider fragments and supported cache pairs. Static persistent reductions can request the existing `cute_proven_bounds` proof without changing its safety requirements or defaults. Only the 128×1024 K18 pin enables that proof.
- **Verification fixes:** update the config-key and repro expectations for `cute_store_policy`; keep the BF16 softmax top-k reference in FP32 with unchanged tolerances and explicit dtype checking. A separate benchmark compatibility patch updates current Quack APIs and callable types and FlashAttention’s dynamic arithmetic attributes.

Hand prototypes and rejected clamp folding, BF16 payload conversion, balanced-max and thread-projection experiments remain archived under this run. The measured executable sources are preserved in [performance snapshot](artifacts/k18-mxfp8-2026-10-14/final_source_snapshot/manifest.json); the fully verified source including the independent compatibility/reference fixes is in [snapshot v4](artifacts/k18-mxfp8-2026-10-14/final_source_snapshot_v4/manifest.json). The [integrity receipt](artifacts/k18-mxfp8-2026-10-14/final_source_integrity.json) verifies all 36 saved files and unchanged benchmarked compiler/kernel sources.

## Library integration

The accepted kernel and six exact CuTe pins have been copied to `/home/shangdiy/helion-kernel-library`:

- [K18 kernel](../helion-kernel-library/helion_kernel_library/_kernels/mxfp8_quantize/mxfp8_quantize.py). The measured TorchAO/native path is unchanged. Additional integration fixes use supported CUDA operations for other recipes and distinguish NVIDIA from ROCm.
- [SM103 CuTe configuration](../helion-kernel-library/helion_kernel_library/configs/mxfp8_quantize/nvidia_sm103_cute.json). All nine exact entries, including the six measured BF16/TorchAO presets, match the candidate. All 27 keys remain. The 18 generic fallback layouts now use one row thread so clipped blocks stay legal.
- [K18 README](../helion-kernel-library/helion_kernel_library/_kernels/mxfp8_quantize/README.md) documents the contract and required companion Helion changes. The existing AOT shim loads the JSON picker.

The [integration manifest](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/library_integration/manifest.json) records source/config hashes, and [CPU library validation](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/library_integration/pytest_cpu.log) records **109 passed, 133 deselected**. Kernel Ruff checks and `git diff --check` pass. Existing unrelated library changes are preserved.

<!-- ROOT_LIBRARY_GPU_VERIFICATION_START -->
Final library integration is committed as `c60b2e5` on `benchmark_all`. Complete operation tests passed on GPU 2 with both backends: **242 default tests** in 36.67 s and **242 CuTe tests** in 53.46 s. These include all six target shapes, exact bytes, finite extrema, recipe switching, and changed-input CUDA graph replay with search forbidden. [Default GPU check](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/library_gpu_default_03.log) · [CuTe GPU check](artifacts/k18-mxfp8-2026-10-14/step4_cleanup/library_gpu_cute_03.log).

Library CPU checks also pass: 109 operation tests, 42 source-generation cases for full/clipped layouts, and 172,964 vLLM scale inputs. Final library review is LGTM. The earlier integration failures exposed existing unsupported recipe operations and invalid clipped fallback layouts; those raw logs remain archived. No measured config was retuned or changed.
<!-- ROOT_LIBRARY_GPU_VERIFICATION_END -->

## Verification

| Check | Result | Evidence |
| --- | --- | --- |
| Public AOT, all six shapes | Exact payload/scale bytes; input preserved; original and changed-input graph replay; actual config verified | Per-shape ABAB receipts above |
| Finite boundary coverage | 65,280 BF16 encodings, 63,488 FP16 encodings, 103,687 FP32 boundary/random samples pass | [Correctness summary](artifacts/k18-mxfp8-2026-10-14/final_correctness/summary.json) |
| New CPU tests | 183 passed, 50 skipped; bounds/adjacent proof tests 111 passed | [New tests](artifacts/k18-mxfp8-2026-10-14/final_cpu_all_new_01.log) · [Bounds tests](artifacts/k18-mxfp8-2026-10-14/final_bounds_cpu_01.log) |
| Full default suite, GPU 2 | 17,842 passed, 10,753 skipped, 3,148 subtests passed; no failures | [Clean full default log](artifacts/k18-mxfp8-2026-10-14/pytest_default_full_03.log) |
| Full CuTe suite, GPU 2 | 27,263 passed, 1,325 skipped, 4,648 subtests passed; no failures | [Clean full CuTe log](artifacts/k18-mxfp8-2026-10-14/pytest_cute_full_03.log) |
| `./lint.sh check` | Ruff, formatting, codespell and Pyrefly pass; 0 type errors | [Full lint log](artifacts/k18-mxfp8-2026-10-14/lint_final_06.log) |
| Independent code review | LGTM, including final bounds and reference dtype invariant | [Verification summary](artifacts/k18-mxfp8-2026-10-14/verification_final_summary.json) |

The original failing default and CuTe runs are retained. The top-k discrepancy reproduced with identical current/base GPU output and came from the rounded BF16 reference; the fixed reference keeps the existing tolerance. The two CuTe expectation failures were the new config field. Evidence: [original default run](artifacts/k18-mxfp8-2026-10-14/pytest_default_full_01.log), [captured current/base comparison](artifacts/k18-mxfp8-2026-10-14/verification_debug/captured_topk_gpu_comparison_01.json), [eight reference variants](artifacts/k18-mxfp8-2026-10-14/pytest_topk_fp32_reference_01.log), [original CuTe run](artifacts/k18-mxfp8-2026-10-14/pytest_cute_full_01.log), and [expectation retries](artifacts/k18-mxfp8-2026-10-14/pytest_cute_expectations_retry_01.log). The full clean reruns above include these fixes. The finite-value enumeration uses the accepted arithmetic before the allocation-order change; final ABAB checks the actual public source on all six shapes. Full-suite test kernels can exercise their normal tuning paths; no cold/full K18 search was resumed.

## Scope and compatibility

- The library kernel is defined for finite contiguous 2D BF16/FP16/FP32 input with a positive width divisible by 32. Native TorchAO RCEIL simplifications rely on finiteness: scale codes are 0…247, and the rounded scale bounds payload magnitude by 448. Nonfinite inputs are outside this contract.
- `nc_l2_first` lowering requires **SM100+ and CUDA major ≥13**, initialized input storage, a full aligned 16/32-byte packet, and proof that the kernel never writes the source or an alias. Unknown writes or incomplete alias facts use ordinary loads. The opaque output descriptor is validated only for **SM103 and CUDA major 13**; other targets normalize `cute_store_policy` to `default`.
- The changes extend shared runtime `Config` types/serialization, config-spec normalization/search and CuTe lowering in `helion/language/memory_ops.py`, in addition to backend passes. They affect the exposed configuration API beyond compiler internals. `nc_l2_first` is CuTe-specific; Triton’s valid load-policy values are unchanged. The CuTe prior version increases from 3 to 4.
- The copied SM103 CuTe pins require this companion Helion stack. An older installed Helion will reject the new config fields/values. The kernel source changes across architectures and both default/CuTe paths; the saved fallback entries remain, while the measured performance claim covers the six BF16/RCEIL shapes on GB300 GPU 2.
- Cache choice and short-row seed performance can differ by target and workload. Default access/tail behavior is preserved by proof and target gates, but no performance guarantee is claimed for other GPUs or shapes. The separate Quack compatibility changes affect other optional benchmark harnesses and were included in lint and final review.
- Reproducing the original Triton anchors after library integration requires the preserved original kernel source, rather than importing the newly optimized library kernel. The [original source](artifacts/k18-mxfp8-2026-10-14/sources/original_library_kernel.py) and baseline receipts remain archived. The K18 benchmark harness requires an external library checkout on `PYTHONPATH`, including its benchmark support helpers.

## Git history

<!-- ROOT_GIT_HISTORY_START -->
The stack preserves the tested Helion base `a03e0f28`; it is an ancestor of the current `origin/main` (`e73fda10`). Splitting the changes into commits preserves the verified source bytes.

| Order | Commit | Change |
| --- | --- | --- |
| 1 | `171363ad` | Benchmark harness and optional baseline API compatibility |
| 2 | `e25ee72f` | FP8/integer packet stores and proven inline-assembly relocation |
| 3 | `91f23191` | Proven readonly packet loads and packet-store cache choices |
| 4 | `b1197a47` | BF16 normalization reference precision correction |
| 5 | `d5969acd` | Short-row seeds and optional static bounds admission |
| 6 | Final `[noland]` commit | Per-shape progress and this report |

Packet and cache-policy prefixes were independently checked from fixed commit archives: 103 passed / 18 GPU skips and 127 passed, respectively. The full-stack checks above remain applicable because compiler and test source bytes are unchanged. Library integration is committed separately as `c60b2e5`. No commits were pushed.
<!-- ROOT_GIT_HISTORY_END -->

Raw timing JSON, winning-config receipts, generated kernels, profiles, snapshots and the goal file remain ignored and uncommitted under `artifacts/`. The copied kernel-owned JSON is the library’s public configuration source. Planning and this portable report belong in the final `[noland]` commit. [progress.md](progress.md) retains every shape’s optimization history.
