# K3 CuTe hillclimb plan and outcomes

## Scope and goal

The active task is K3 packed MXFP8 GEMM on six requested shapes. Case 1 was
changed by the user from M7 to M8; the earlier M7 evidence remains historical.
K4, K6 and K19 were stopped at the user's request and are excluded from this
goal. The final authorized device is physical GPU 3 only.

Use the original public operation, packed Int32 scale recipe, masks, host
allocations, partition setting and numerical policy. The target is the faster
of the native implementation and Helion Triton. Every active Helion stage must
use CuTe. Callers select no compiler configuration and trigger no runtime search.

The performance contract requires every case to reach at least 0.99 for
baseline latency divided by CuTe latency, and the geometric mean to reach 1.0.
Only completed fresh full searches and canonical same-GPU verification qualify.
Each run uses a new seed, cache bypass and no time budget. The current checker
prints **GOAL MET**: cold geometric mean 1.0704, public geometric mean 1.0679.

All final comparisons use GB300/SM103, 152 SMs, a 1400 W power limit and GPU 3.
The benchmark runtime is Python 3.12.13, Torch 2.14.1+cu130, Triton 3.8.0,
CUTLASS DSL 4.8.0 and the linked DeepGEMM revision
`057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`.

## Ranked leads and outcomes

| Lead | Outcome |
| --- | --- |
| Admit the original packed contraction through native CuTe lowering | Canonical packed-scale provenance and registered complete-K grid proofs retain the original executable body and three block owners. Rounded Int32 coordinate bounds are checked. |
| Reduce padded operand transfers and expose independent geometry | Logical-M, swap-AB, compact TMA, smaller N tiles and pipeline depth choices become general configuration options and seeds. Cold searches select different legal geometries across cases 0–4. |
| Match the native scale pipeline | SF-TMA, in-place scale transposition, shared operand/scale readiness and safe stage lifetimes remove unnecessary traffic. Independent producer tails fix a demonstrated phase mutation. Unit coordinate support retains the same packed contract. |
| Overlap accumulation and output transfer | Optional accumulator banks, full register prefetch and two TMA output stages improve the large case. Eligibility checks output metadata, alignment, TMEM and shared-memory capacity. |
| Ensure full search measures coupled output modes | The first case 5 search evaluated only default output modes because 134 default hints preceded 81 output hints and the initial population stopped at 100. A general coverage group appends and pins four legal joint profiles without changing the old seed order, population prefix or RNG state. |
| Probe remaining structural ideas | Column output layout, early TMEM permit release and logical M16 were measured on the same GPU with anchors. They did not improve the tested gap and were not adopted. Packed K128 layout repair and N8 admission were retained because they restored legal general choices. |

Manual probes established which structures to generalize; their timings do not
replace cold search evidence. The original case 5 gap of 13.984 us versus native
11.904 us is archived. The fresh coverage search instead finds 11.552 us versus
native 11.968 us from the original starting table.

## Final cold results and configuration references

Paths below are relative to the campaign root
`artifacts/cute-fastest-20261019/`. Raw results and JSON configurations remain
uncommitted; the notes commit contains only this plan, the report and plots.

| Case | M, N, K | CuTe / native, us | Native/CuTe | Cold result | Exported configuration |
| --- | --- | --- | --- | --- | --- |
| 0 | 1, 1792, 5120 | 7.712 / 8.544 | 1.1079 | `K3/compiled-original/gpu3/original-final-case0-cold-r2/case0/result.json` | `K3/cold-winning-configs/case0-original-final-r2.json` |
| 1 | 8, 8192, 1280 | 5.824 / 6.272 | 1.0769 | `K3/compiled-original/gpu3/original-final-case1-m8-cold-r1/case1/result.json` | `K3/cold-winning-configs/case1-original-final-r1.json` |
| 2 | 16, 2048, 4096 | 7.104 / 7.680 | 1.0811 | `K3/compiled-original/gpu3/original-final-case2-cold-r1/case2/result.json` | `K3/cold-winning-configs/case2-original-final-r1.json` |
| 3 | 32, 5120, 2048 | 6.400 / 6.752 | 1.0550 | `K3/compiled-original/gpu3/original-final-case3-cold-r1/case3/result.json` | `K3/cold-winning-configs/case3-original-final-r1.json` |
| 4 | 128, 1792, 5120 | 8.128 / 8.672 | 1.0669 | `K3/compiled-original/gpu3/original-final-case4-cold-r1/case4/result.json` | `K3/cold-winning-configs/case4-original-final-r1.json` |
| 5 | 1024, 8192, 1280 | 11.552 / 11.968 | 1.0360 | `K3/compiled-original/gpu3/original-final-case5-coverage-cold-r2/case5/result.json` | `K3/cold-winning-configs/case5-original-coverage-final-r2.json` |

The report records Triton results, historical before values, public verification,
logical throughput, arithmetic means, geometric means and all raw hashes.
Historical before values from GPU 0 are labeled separately and do not enter
the paired GPU 3 speedups.

## General capabilities and public integration

- Extend the existing native MXFP8 backend with packed-scale and registered-grid
  proofs; no frontend language or workload rewrite is needed.
- Keep global, separate SF and in-place SF choices available where legal.
- Retain existing byte/E8M0 and Triton configuration banks and public signatures.
- Use a kernel-owned private AOT bridge, independent picker and keyed JSON table
  for the original packed frontend. The six saved selections match cold exports.
- Expose output transfer as an optional enum. The default behavior remains
  available; the measured joint profile fits the established resource limits.
- Use compiler coverage after the population cap. Unsupported metadata,
  unavailable carriers and explicit default choices retain their legal behavior.
- Declare and verify the backend patch against the pinned upstream revision.
  Ordinary setup and `benchmark.sh` use installed source records.

## Measured and cleanup source provenance

The measured Helion base is
`bfc25902f558b036e2c602be72d31d1c04fe4363`. Cases 0–4 use the frozen full patch
`f9c19ca6663360852c315cfa7b21d988b2058bd828b46792c2283fe87079c0ed`.
Case 5 uses the coverage successor
`e9495caad98602cff38aec6298852a59680159939a9b58bf4e259dbe4f49ee04`.
Their nine existing backend/runtime files are byte-identical. The installed
backend-only patch has eleven source files and SHA256
`18a141096993efd10fac7403ae9ced42af26275a044ada412395fd57600ab2a2`.

The separate clean local branch is `k3-cute-hillclimb-clean-20261019`, based on
`origin/main` at `ceafab07aab63876d87af36b5a7286cfb060ea0f`. It retains the two
newer upstream commits and has five logical `[cutedsl]` commits, ending at
`b8c35144d63ee7d46fbe9ce0d20e1f93876465c8`. The final test-only amendment updates the exact MXFP8 vocabulary expectation and retains all NVFP4/N64 checks. Ruff and typing cleanup preserve all
recorded configurations, host code, native plans and compared CUBINs.

The clean stack is distinct from the measured and installed source. Those
sources and all archived evidence remain unchanged. No push or remote PR has
been created. The final `[noland]` commit contains exactly the plan, report
and six plot files; backend commits do not reference these notes.

## Verification checklist

- **PASS:** all six completed cold full searches and canonical verification.
- **PASS:** checker output at `goal-check-final-all-verified.txt`.
- **PASS:** six actual public CUDA cases, no skips or runtime search, all four
  live inputs, poisoned output, replay and input storage preservation. Receipt:
  `K3/production/final-main-k3-cuda-correctness-r1/receipt.json`.
- **PASS:** twelve public benchmark workers and 24 clean canonical timing calls.
  Receipt: `K3/production/final-main-k3-all6-aot-r1/receipt.json`.
- **PASS:** committed clean-stack Ruff, codespell and CI-aligned Pyrefly checks.
  Receipt: `K3/step4-verify/lint-r4-config-coverage-frozen-receipt.json`.
- **PASS:** final code review and source/config/native equivalence receipts.
  Review: `K3/step4-verify/code-review-final-1246698e-r1.md`; amendment: `K3/step4-verify/code-review-addendum-b8c35144-r1.json`.
- **PASS:** full default suite: 18,071 passed, 12,053 skipped and 3,245 passed subtests, tested at `1246698e`. Receipt: `K3/review-stack-verification/final-clean-stack-ci47-default-full-r1/receipt.json`.
- **PASS:** full CuTe suite: 28,724 passed, 1,393 skipped and 4,680 passed subtests, tested at `b8c35144`. Receipt: `K3/review-stack-verification/final-clean-stack-ci47-cute-full-r2/receipt.json`.
- **PASS:** both full gates finish with exit 0, zero failed cases, clean source/dependency/main-runtime audits and no foreign GPU processes. The sole source change between them is the CuTe-only vocabulary test; all backend bytes are identical. The suites run sequentially on GPU 3 with declared SDK 4.7. The benchmark/native K3 evidence uses SDK 4.8.
- **FINAL DELIVERY:** one eight-file `[noland]` plan/report/plots commit and six-commit local patch-stack export follow the verified source release. Raw JSON, configs and goal remain uncommitted.

Static checking uses isolated CI dependencies, including CUTLASS DSL 4.7.0.
The full GPU suites use the supported isolated test environment with DSL 4.7.0 and the installed Torch/Triton files. K3 benchmark and native proofs use DSL 4.8.0. Neither
verification environment changes the installed framework or baseline packages.

## Delivery and archival rules

The final notes layout is `notes/cute-hillclimb/k3-20261019/` with `PLAN.md`,
`REPORT.md` and PNG/PDF/SVG versions of the combined speedup chart and case
heatmap. The chart preserves all 156 non-K3 records. Raw JSON, configs, goal,
receipts and caches remain under campaign artifacts, outside committed notes.

The original campaign `plan.md`, minified historical iteration logs and failed
or superseded measurements remain intact. This polished plan is the delivery
summary; it does not replace the original record.
