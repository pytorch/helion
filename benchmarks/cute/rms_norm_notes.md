# RMSNorm follow-up optimization

On physical GPU 3 (NVIDIA GB300, 1400 W), full-width rolled chunks improve
`32768x256` from **7.232 us to 7.104 us (1.0180x)** in process-isolated
old/new/old/new verification. Each process uses five CUPTI measurements with
CUDA graphs and cold L2. GPUs 1 and 2 were never used.

A fresh-cache full LFBO search (seed 2026092501, 398 configurations, 312 s)
selected the same layout automatically: 16 rows per CTA, 16 reduction threads
per row, a 256-element rolled chunk, vector width 16, and register reloads.
The kernel body remains identical to the original benchmark.

The change preserves explicit full-width power-of-two chunks separately from
persistent reductions, adds a general SM100 reduction seed, and recognizes
symbolic input widths during seed dtype discovery. A row-stride alignment
guard safely falls back to scalar accesses for views such as stride 257.
Persistent defaults and non-CuTe normalization remain unchanged.

Implementation commit: `013c65a8` (`[cutedsl] Tune full-width rolled reduction chunks`).

## Final replay

All times below are microseconds. Initial CuTe times come from the original
`artifacts/rms-norm-2026-09-24/before_cold_full.json`; previous-stack times
come from `artifacts/rms-norm-2026-09-24/review_fixed_replay.json`.
Final times and each row's ratio are backed by
`artifacts/rms-norm-next-2026-09-25/final_replay.json` and `final_replay_<shape>.json` in that directory.
Effective GB/s counts one input read, one weight read, and one output write.
Differences on unchanged shapes are replay variation; only the 256-wide gain
was established with alternating old/new configurations.

| Shape | Best saved baseline | Initial CuTe | Previous stack | Final CuTe | Final GB/s | Best/final |
|---|---:|---:|---:|---:|---:|---:|
| 2048x1024 | cutlass 3.552 | 6.624 | 3.392 | 3.424 | 2450.5 | 1.0374 |
| 2048x4096 | flashinfer 7.616 | 14.721 | 6.464 | 6.496 | 5166.7 | 1.1724 |
| 2048x8192 | cudnn 12.032 | 25.920 | 11.616 | 11.807 | 5685.2 | 1.0191 |
| 2048x16384 | cudnn 24.544 | 46.145 | 22.240 | 22.561 | 5950.7 | 1.0879 |
| 2048x32768 | liger-cute 43.200 | 86.433 | 42.945 | 42.785 | 6275.6 | 1.0097 |
| 4096x3584 | cudnn 10.977 | 22.784 | 9.600 | 9.504 | 6179.2 | 1.1550 |
| 4096x7168 | flashinfer 20.064 | 40.320 | 20.160 | 20.096 | 5844.7 | 0.9984 |
| 16384x8192 | flashinfer 79.425 | 140.033 | 76.769 | 77.600 | 6918.7 | 1.0235 |
| 32768x256 | quack-cute 7.520 | 11.488 | 7.264 | 7.104 | 4723.4 | 1.0586 |
| 32768x4096 | flashinfer 79.425 | 132.545 | 78.624 | 78.528 | 6836.8 | 1.0114 |
| 32768x65536 | cutlass 1202.278 | 2237.768 | 1199.397 | 1198.467 | 7167.5 | 1.0032 |
| 16384x131072 | cutlass 1250.149 | 2821.838 | 1222.183 | 1223.078 | 7023.4 | 1.0221 |
| 8192x262144 | cudnn 1264.707 | 2980.465 | 1243.528 | 1243.399 | 6908.9 | 1.0171 |
| Mean / ratio geomean | 308.115 | 659.006 | 303.399 | 303.450 | 5933.2 | 1.0460 |

Versions: PyTorch 2.14.0+cu130, CUDA 13.0, Triton 3.8.0, CUPTI 13.0.1;
profiling used Nsight Compute 2025.3.1. Saved baselines are the requested
historical results; they were not rerun during this follow-up. Their exact
versions and provenance are in `artifacts/rms-norm-next-2026-09-25/saved_baselines.json`.

## Other opportunities tested

| Probe | Result |
|---|---|
| Cast-wrapped FMA recognition | No gain; SASS already uses FHFMA.BF16. |
| Defer initial cluster wait | Wide case 1.2434 -> 1.3163 ms; rejected. |
| Pipeline the inner lane loop | Wide case 1.2434 -> 1.4972 ms; rejected. |
| Unroll the wide lane loop | No measurable gain. |
| Packed FP32 arithmetic | Under 1% isolated improvements, with regressions elsewhere; rejected. |
| Stronger pointer alignment | No measurable gain. |
| Explicit 256-bit PTX loads/stores | Under 1% isolated improvement; rejected. |

Raw probes, generated kernels, rejected patch, and profiler reports remain
under `artifacts/rms-norm-next-2026-09-25/`. The initial `current_8192x262144.json` and
`profile/rms-norm-winners-20260925/reports/full_32768x256.ncu-rep` briefly
overlapped and are excluded. NCU replay durations are diagnostic only.
For generated-code probes, the generated `.py` file defines the implementation;
the early probe JSON `config` field alone does not describe manual edits.

## Validation

| Check | Result |
|---|---|
| CuTe focused suites | 369 passed, 29 skipped, 290 subtests passed |
| Default focused suites | 94 passed, 33 skipped, 52 subtests passed |
| Ruff, formatting, codespell | Pass |
| Changed production files and new test module: Pyrefly | 0 errors |
| Independent code review | LGTM |
| Full suites | Blocked: pytest-xdist unavailable; fallback collection has 21 dependency/import errors |
| Repository-wide Pyrefly | 49 existing errors outside changed production files |

Regression coverage includes full/persistent round-trips and search neighbors,
preserved defaults, tiny/oversized chunks, symbolic-shape seed discovery, and
both centered and var_mean LayerNorm with register/global reloads, partial row
tiles, strided inputs, and unaligned row strides in FP32 and BF16.

## Goal checker

```text
PASS 2048x1024: ratio 1.037 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS 2048x4096: ratio 1.172 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS 2048x8192: ratio 1.019 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS 2048x16384: ratio 1.088 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS 2048x32768: ratio 1.010 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS 4096x3584: ratio 1.155 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS 4096x7168: ratio 0.998 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS 16384x8192: ratio 1.024 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS 32768x256: ratio 1.059 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS 32768x4096: ratio 1.011 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS 32768x65536: ratio 1.003 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS 16384x131072: ratio 1.022 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS 8192x262144: ratio 1.017 >= bar 0.990  [artifacts/rms-norm-next-2026-09-25/final_replay.json]
PASS geomean: 1.0460 (min 1.0)
GOAL MET
```

## Saved baseline detail (microseconds)

Every cell is backed by `artifacts/rms-norm-next-2026-09-25/saved_baselines.json`.
Unavailable implementations are marked `-`.

| Shape | cudnn | cutlass | flashinfer | helion-cute | helion-tileir | helion-triton | liger-cute | liger-triton | quack-cute | torch-eager |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048x1024 | 3.904 | 3.552 | 3.584 | 6.720 | 6.944 | 3.648 | 3.712 | 3.776 | 3.616 | 5.120 |
| 2048x4096 | 8.640 | 8.000 | 7.616 | 13.760 | 9.600 | 6.080 | 8.736 | 8.481 | 8.480 | 9.632 |
| 2048x8192 | 12.032 | 14.112 | 13.473 | 25.888 | 13.504 | 12.640 | 14.208 | 15.617 | 14.112 | 17.025 |
| 2048x16384 | 24.544 | 25.312 | 25.248 | 46.496 | 27.168 | 25.600 | 24.544 | 27.968 | 25.313 | 36.608 |
| 2048x32768 | 45.408 | 43.680 | 62.048 | 86.464 | 45.824 | 47.457 | 43.200 | 63.072 | 43.297 | 70.368 |
| 4096x3584 | 10.977 | 13.056 | 12.256 | 23.264 | 12.128 | 12.000 | 13.568 | 13.216 | 12.672 | 16.608 |
| 4096x7168 | 20.672 | 20.896 | 20.064 | 40.128 | 20.928 | 20.800 | 21.088 | 26.337 | 21.056 | 28.448 |
| 16384x8192 | 87.968 | 79.712 | 79.425 | 139.937 | 78.912 | 78.112 | 83.072 | 96.640 | 82.945 | 92.609 |
| 32768x256 | 11.424 | 7.584 | 10.464 | 11.776 | 9.344 | 5.536 | 7.648 | 7.552 | 7.520 | 27.968 |
| 32768x4096 | 85.504 | 79.777 | 79.425 | 132.544 | 78.048 | 77.537 | 82.976 | 85.377 | 82.432 | 90.016 |
| 32768x65536 | 1386.212 | 1202.278 | 22978.815 | 2252.294 | 1288.005 | 1281.062 | - | 2464.332 | 1226.694 | 1993.156 |
| 16384x131072 | 1302.724 | 1250.149 | 1950.694 | 2821.575 | 1721.991 | 1539.320 | - | - | 1279.655 | 1964.932 |
| 8192x262144 | 1264.707 | 1404.454 | 13518.390 | 2979.720 | 1802.148 | 1570.040 | - | - | 1467.847 | 2054.212 |
