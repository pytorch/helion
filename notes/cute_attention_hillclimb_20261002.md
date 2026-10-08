These local review notes belong only to the final `[noland]` commit. Omit that
commit from an upstream submission. Raw measurements, configurations, plans and
goal records remain outside git history.

The attention-only stack is based on fetched remote main
`132c3147ce664bb5811341be2f81c18f753b2c7f`. It contains resident attention lowering
and legal search choices, joint pipeline/register seeds, the twelve SM103
pretuned recipes, then these notes. The earlier stack is preserved at
`8711daa30e03a9577bc77eb4d7d517c7072c3107` on
`codex/attention-rebased-final-20261007`.

Process logging, worker isolation, source-retirement protocols and benchmark
provenance changes were removed from this stack. Their existing upstream
implementations remain unchanged. The retained benchmark changes validate the
attention-specific conditional parent search and its implementation-owned
coordinates; they keep the existing strict attention CLI consistent with the
new search policy without accepting incomplete qualification or malformed
proposal budgets.

The attention compiler, resource accounting, capability constraints and final
joint seeds match the reviewed rebased implementation. Kernel bodies in
`pretuned_kernels/attention/attention.py` match the output-only dense and causal
frontends in `examples/attention.py`. The exact twelve returned configuration
dictionaries are unchanged. They require contiguous BHND inputs, matching
Q/K/V shapes and dtypes, and GB300/SM103. They are not general shape dispatch or
SM100 presets. The experimental launcher cache change is not included.

Original optimization evidence remains immutable. Its twelve final
configurations came from strict cold FULL searches, with independent discovery
on both original targets. The original goal checker reports GOAL MET: every
baseline/Helion ratio is at least 0.99 and the geometric mean is 1.036810.
Reserved shapes informed later development and are regression cases rather
than unseen holdouts. Those measurements used the earlier source and
`atol=0.05, rtol=0.02`; they do not establish the tighter qualification below.

The new prospective precision requirements are:

| Output dtype | atol | rtol | Allowed failing elements |
| --- | ---: | ---: | ---: |
| FP16 | 0.001 | 0.001 | 0 |
| BF16 | 0.005 | 0.005 | 0 |

Both actual and reference outputs must be finite. These are engineering
thresholds chosen before native results, accounting for different output
rounding and cancellation near zero. They are not claimed as a universal FA4
tolerance. Independent qualification uses FP64 QK, softmax and PV on the same
already-quantized inputs, with query chunks and a global causal mask. It covers
all twelve presets with multiple seeds and compares the available baselines
under the same requirements. Native results are pending at this snapshot.

The packaged recipe registers one dtype-aware accuracy callback and uses it
from `main()` before timing. Its ordinary reference is output-only cuDNN SDPA;
the independent FP64 oracle is an external validation artifact. As in upstream
Helion, a custom accuracy callback disables ordinary search-result caching and
uses in-process autotune accuracy checking. Stored AOT preset evaluation does
not autotune.

Current CPU validation comprises 44 attention search checks, 354 strict
attention-validator checks and 70 preset/joint-seed checks. These passed;
scoped Ruff and independent reviews passed. Full default/CuTe suites and native
precision/performance validation of this narrower stack remain pending. A
native-only test was mistakenly selected in an earlier CUDA-hidden CPU run;
that command-selection failure remains recorded and is not a native result.

The first rebase comparison of the earlier W/F snapshots completed all twelve
cases with valid source, configuration, accuracy and ownership records. Nine
passed its fixed statistical criteria, cases05/09 were inconclusive, and
case08 had a resolved public-call slowdown. Subsequent unscored controls
distinguished device execution from public launch overhead. Those results are
preserved; no narrower-stack performance admission is inferred from them.

Original supported boundary controls included legal D64/D128 configurations at
nonaligned sequence lengths and generic D256 paths. Initial default N257
failures remain documented. Neither arbitrary-shape support nor general
default-N257 or optimized resident-D256 support is claimed. Tighter boundary
qualification is pending with the native precision panel.

Evidence:

- [Original final report](/home/shangdiy/helion-cute-attention-matrix-20261002/artifacts/attention-matrix-20261002T204942Z/FINAL_REPORT.md), [per-row raw evidence and configs](/home/shangdiy/helion-cute-attention-matrix-20261002/artifacts/attention-matrix-20261002T204942Z/FINAL_MATRIX.csv), and [original goal checker](/home/shangdiy/helion-cute-attention-matrix-20261002/artifacts/attention-matrix-20261002T204942Z/FINAL_GOAL_CHECK_COMPLETION_20261007T025742Z.txt).
- [Narrowed scope request](/home/shangdiy/helion-cute-attention-matrix-20261002/artifacts/attention-matrix-20261002T204942Z/pretuned_rebase_20261007T202157Z/attention_only_scope/REQUEST.json), [search validation](/home/shangdiy/helion-cute-attention-matrix-20261002/artifacts/attention-matrix-20261002T204942Z/pretuned_rebase_20261007T202157Z/attention_only_search_preparation/FINAL.json), and [strict-validator validation](/home/shangdiy/helion-cute-attention-matrix-20261002/artifacts/attention-matrix-20261002T204942Z/pretuned_rebase_20261007T202157Z/attention_only_audit/benchmark_review/FINAL.json).
- [Frozen precision policy](/home/shangdiy/helion-cute-attention-matrix-20261002/artifacts/attention-matrix-20261002T204942Z/pretuned_rebase_20261007T202157Z/attention_only_scope/PRECISION_POLICY.json), [precision design](/home/shangdiy/helion-cute-attention-matrix-20261002/artifacts/attention-matrix-20261002T204942Z/pretuned_rebase_20261007T202157Z/attention_only_scope/precision_design/DESIGN.md), and [preset CPU checks](/home/shangdiy/helion-cute-attention-matrix-20261002/artifacts/attention-matrix-20261002T204942Z/pretuned_rebase_20261007T202157Z/attention_only_scope/preset_precision_cpu.xml).
- [Preserved initial rebase comparison](/home/shangdiy/helion-cute-attention-matrix-20261002/artifacts/attention-matrix-20261002T204942Z/pretuned_rebase_20261007T202157Z/final_review/fixed_v1_complete/REPORT.md) and [current task state](/home/shangdiy/helion-cute-attention-matrix-20261002/artifacts/attention-matrix-20261002T204942Z/pretuned_rebase_20261007T202157Z/STATE.json).
