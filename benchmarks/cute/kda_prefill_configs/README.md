# KDA prefill winning configurations

These configurations were selected by fresh full autotuning of
[`kda_prefill_native_math`](../kda_prefill_fused.py) on NVIDIA GB300
(SM103, 152 SMs, 1400 W power limit), then checked in a separate pinned process
on the same GPU. Each JSON preserves the exact winning `stage_configs` from
the final v9 campaign. The JSON uses the existing result format so the loader
rejects a mismatched shape; machine paths, timings, and build metadata are omitted.

| `--shape` | Workload | Configuration |
| --- | --- | --- |
| 0 | B4, T512, H16, D128; uniform | [b4_t512_h16_d128.json](b4_t512_h16_d128.json) |
| 1 | B16, mean T2048, H16, D128; ragged | [b16_mean_t2048_h16_d128_ragged.json](b16_mean_t2048_h16_d128_ragged.json) |
| 2 | B16, T8192, H32, D128; uniform | [b16_t8192_h32_d128.json](b16_t8192_h32_d128.json) |

Inputs are BF16 Q/K/V, raw gates and beta logits, with FP32 gate parameters and
initial state. The kernel includes normalization and gate activation, preserves
the initial state, and returns BF16 output plus separate FP32 final state.
Authoritative recurrent state stays FP32; matrix operations use BF16 copies.
These are measured choices for the listed shapes and GPU, not universal defaults.

From the repository root, use an existing CUDA/CuTe environment and the assigned
GB300. Keep the scheduler's `CUDA_VISIBLE_DEVICES` assignment. Substitute the
library and FlashInfer checkout paths, and choose fresh output/cache directories:

```bash
python benchmarks/cute/kda_prefill_hillclimb.py \
  --library-root /path/to/helion-kernel-library \
  --flashinfer-src /path/to/flashinfer-v0.7.0 \
  --shape 0 --mode pinned \
  --configs benchmarks/cute/kda_prefill_configs/b4_t512_h16_d128.json \
  --helion-schedule fused --partition all --state-abi fp32 \
  --implementations cake,helion --input-seed 0 --cycles 60 \
  --cache-root /tmp/kda-prefill-small-pinned-unique \
  --output artifacts/kda-prefill-small-pinned-unique
```

FlashInfer must be a clean v0.7.0 checkout at
`4d75a33f19aaf48b44d5b1c5dbca33bc1eca5c58`; the harness checks this source identity.
The library supplies the breadth inputs, FP64 reference, and CAKE adapter.
Output directories must be new and include an `artifacts` path component.
Pinned mode loads the saved winner without searching, and checks reference
accuracy, input immutability, and repeated outputs before and after timing.
Timing uses complete CUDA graphs with cold-L2 treatment and 60 interleaved
CAKE/Helion ABAB/BABA cycles.

For a fresh search, remove `--configs`, change to `--mode autotune`, use
`--implementations cake,cute,helion --cycles 30`, and choose new cache/output
directories. The harness runs full effort over all nine schedule/order choices
with a fresh random tuning seed and cache bypass. Do not add a tuning budget or
generation cap. To verify that run's winner, use its emitted `configs.json` in a
new pinned process on the same GPU.

Source receipts: final v9 cold runs `shape0-1565690`, `shape1-1565691`, and
`shape2-1565687`; compiler snapshot archive SHA256
`3447702bbce187f584a09ca3a42bb3622f39647542b30aa148ab5d0651b022f4`.
Raw timing and source archives remain outside this portable configuration bundle.
