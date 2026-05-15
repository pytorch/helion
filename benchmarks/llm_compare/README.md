# LFBO vs LLM-seeded LFBO comparison

Head-to-head benchmark of `LFBOTreeSearch` against `LLMSeededLFBOTreeSearch`
(Claude Opus 4.7 seed, `effort_level=max`, `fast_mode=1`) across a fixed set
of example kernels at full autotune effort.

## Running

```bash
HELION_LLM_API_KEY="$(your-key-source)" \
python benchmarks/llm_compare/run_compare.py --gpu 0 --timeout 2400
python benchmarks/llm_compare/summarize.py \
    benchmarks/llm_compare/runs/<tag>/results.jsonl \
    --out benchmarks/llm_compare/runs/<tag>/headtohead.md
```

The harness sets `HELION_SKIP_CACHE=1` per cell so neither autotuner can reuse
the other's cached `best_config`. Other caches (inductor, Triton) are left
alone so the run doesn't disturb concurrent helion jobs on the same host.

## Results — 6 kernels on NVIDIA B200, full effort, single shot per cell

| kernel     | wall LFBO (s) | wall LLM (s) | wall ratio  | helion LFBO (ms) | helion LLM (ms) | kernel ratio |
| ---------- | ------------- | ------------ | ----------- | ---------------- | --------------- | ------------ |
| attention  |        251.9 |        114.6 |  **0.46×**  |          0.0594  |         0.0548  |       0.92×  |
| gdn_fwd_h  |        286.7 |        118.1 |  **0.41×**  |          0.5910  |         0.6165  |       1.04×  |
| layer_norm |       1358.3 |        723.7 |  **0.53×**  |          0.0349  |         0.0389  |       1.11×  |
| matmul     |        117.9 |         97.8 |  **0.83×**  |          0.0154  |         0.0143  |       0.93×  |
| rope       |         96.1 |         56.7 |  **0.59×**  |          0.0497  |         0.0458  |       0.92×  |
| softmax    |        573.3 |        162.8 |  **0.28×**  |          0.0143  |         0.0165  |       1.15×  |

**Geomean compile wall-clock ratio (LLM / LFBO): 0.49×** — LLM-seeded is ~2×
faster to autotune, on every kernel.

**Geomean helion kernel-ms ratio (LLM / LFBO): 1.01×** — tuned-kernel quality
is essentially tied (3 small wins, 3 small losses, within typical autotune
noise).

Ratios are LLM / LFBO; `<1.0×` means LLM-seeded wins.

## Notes

- `matmul` exits non-zero on both autotuners because of a pre-existing
  `addmm`-lambda pickle bug in `examples/matmul.py`; the primary `matmul`
  `run_example` runs and times correctly before the crash.
- Single shot per cell, so ±10% on individual kernel-ms cells is plausible.
  The wall-clock signal (geomean 0.49× across 6/6 kernels) is far above noise.
