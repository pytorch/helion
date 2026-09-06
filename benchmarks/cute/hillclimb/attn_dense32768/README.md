# Hillclimb record: dense attention `2x32x32768x64` fp16, `HELION_BACKEND=cute`

One JSON file per optimization step. Each file pins the exact `helion.Config`
that produced the recorded number, so any step can be re-benchmarked later
without re-running the search that found it.

Shape: `z=2 h=32 seq_len=32768 head_dim=64 dtype=float16 causal=0 biased=0`
(`num_kv = 32768 / 128 = 256` KV tiles).

Hardware for the recorded numbers: NVIDIA GB300 (sm_103), 1400 W power limit.
Baseline for the `ratio` column: FlashAttention-4 (`fa4-v4.0.0.beta23`,
CuTe 4.7.0) measured interleaved with Helion on the same GPU in the same batch.

## Step file schema

| field | meaning |
| --- | --- |
| `step` | monotonically increasing step index |
| `name` | short name of the optimization |
| `timestamp_utc` | when the step's measurement completed |
| `elapsed_from_t0` | wall-clock time from the start of the hillclimb |
| `measurement.helion_cute_tflops` | median-of-medians TFLOP/s for Helion+CuTe |
| `measurement.fa4_tflops` | the interleaved FA4 median in the same batch |
| `measurement.ratio` | `helion_cute_tflops / fa4_tflops` |
| `artifacts` | raw per-run JSON files (uncommitted, under `artifacts/`) |
| `config` | the full `helion.Config` kwargs to reproduce the kernel |
| `changes` | source changes this step depends on (commit subjects) |

## Re-benchmarking a step

```bash
python benchmarks/cute/hillclimb/attn_dense32768/rebench.py \
    benchmarks/cute/hillclimb/attn_dense32768/step00_baseline.json --gpu 3
```

`rebench.py` reads the pinned config out of a step file and drives
`benchmarks/cute/compare_attention_backends.py` with it, interleaved against
FA4, exactly the way the number was originally taken.

## Measurement hazard on this box

GB300 GPUs on one board share a power/cooling envelope. A heavy job on a
*sibling* GPU roughly halves the clocks of the GPU under test: the same kernel
and config measured 1294 TFLOP/s on a quiet board and 564 TFLOP/s while a
neighbour was drawing 826 W. Always interleave Helion and the baseline in the
same batch on the same GPU, and record the sibling GPUs' power draw.
