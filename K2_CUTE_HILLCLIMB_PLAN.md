# Paused K2 CuTe hillclimb: follow-up plan

The user stopped tuning on 2026-10-15 and requested history cleanup, preset export, and a report. The performance objective remains unmet: 10 of 16 targets meet the 0.99 ratio threshold; the current geometric mean is 1.0100877096396688.

## Pending cases

Ratio is fastest measured baseline time / Helion time. Larger is faster.

| Shape (M, K, N) | Scale mode | Current ratio |
| --- | --- | ---: |
| (128, 2880, 5120) | tensor-tensor | 0.841428551 |
| (128, 2880, 5120) | token-channel | 0.887591856 |
| (128, 2880, 5120) | tensor-channel | 0.890666743 |
| (256, 4096, 2048) | tensor-tensor | 0.934006677 |
| (128, 2880, 5120) | token-tensor | 0.940558680 |
| (64, 4096, 4096) | tensor-tensor | 0.982117080 |

## Saved measured state

- Branch: `cute-hillclimb-k2-gb300-paused-20261015`. The measured V8 state originated on `6a1bddbe8eb61aa0b64bd102ab35fa97996c1322`; the stack was subsequently rebased onto `7b8d220bb0e9d51aadb7312e5b2428d081ad1dff`.
- The historical V8 backend reproduced all sixteen measured kernel sources and wrapper plans exactly under CPU binding. The separately requested frontend refactor changes the implementation of the four M256 cases; their saved ratios remain historical until re-measured.
- `pretuned_kernels/scaled_mm_cute/` combines the original B200 rowwise preset and the GB300 scalar/vector-scale preset under the corrected name. Its `scaled_mm()` wrapper dispatches by input architecture and shape. The SM100 AOT file preserves the B200 configurations; the SM103 AOT file covers all twenty original GB300 shape/scale cases, including the four unchanged M1 skinny cases.
- Raw results, goal, audits, configs, and rejected attempts remain under `artifacts/k2-original-gb300-20261013/`.
- The existing Helion `handoff` branch and its separate work are preserved.

## Resume work

1. Confirm an exclusive GPU using the existing all-GPU traffic guard. Stop owned GPU work if unrelated traffic appears; require all four GPUs to be free for 30 seconds before retrying.
   Complete the consolidated `scaled_mm_cute` hardware correctness check, including legacy row-broadcast scale views; its latest attempt was interrupted by another GPU job after two passing subtests. The kernel bodies/configs are preserved and the merged dispatch/AOT paths pass CPU verification.
2. Resume from the saved pending ratios and immutable raw evidence. Re-run each required comparison on the same GPU/runtime with full cold autotuning and complete provider calls. The K2 graph timer now warms its clear-only batch before timing, so records made with the earlier timer are not timing-identical; re-run both sides of a comparison with the same harness.
3. Validate the reviewed Native64 v3 package on hardware before measuring the scalar input preload prototype. V1 stopped on a PTX address-space checker omission; V2 stopped on the package authorization scope. Neither attempt produced a complete native gate.
4. Finish independent native-package/cold-adapter review and actual Native37 qualification for `worktrees/coverage-normal-c1-v1` before benchmarking its C1 and initial-coverage changes. The prototype has CPU/source checks; its performance gain remains unmeasured.
5. Test the coupled N64/cluster-N2/BK128 candidates that the previous initial population omitted. Keep the 100-row initial budget and normal LFBO search limits.
6. Preserve older measurements as history and use the latest qualified result. Settle ratios near the threshold with interleaved verification.

## Remaining verification

The saved full test-suite runs retain failures and an incomplete CuTe run; they are not a clean verification gate. The measured V8 chunk has 423 CPU tests plus 58 subtests and 202 native cases passed. The later frontend refactor passes CPU code-generation/layout checks and eight scoped GB300 correctness cases, including all four transposed scale modes. It has no new timing measurements. Full default/CuTe suites and final performance verification remain required before an upstream submission.

## Prototype locations

- Scalar preload: `artifacts/k2-original-gb300-20261013/worktrees/dense-scalar-input-preload-v1/`.
- Coherent Native64 v3: `artifacts/k2-original-gb300-20261013/local_native_scalar_20261015_v3/`.
- Combined coverage/C1 compiler: `artifacts/k2-original-gb300-20261013/worktrees/coverage-normal-c1-v1/`.
- Native37 package review snapshot: `artifacts/k2-original-gb300-20261013/local_native_coverage_c1_20261015/`.
- Original iteration and recovery history: `artifacts/k2-original-gb300-20261013/PLAN.md`.
