# Hillclimb record: 8 attention shapes vs tuned FA4, GB300 (sm_103)

One JSON file per milestone. Each pins the measurement, the commits it depends
on, and (for the final step) the exact per-shape `helion.Config` the compiler
promotes, so any step can be re-measured later.

Shapes: `z=2 h=32 head_dim=64 dtype=float16`, dense at S = 32K/64K/128K/256K and
causal at S = 64K/128K/256K/512K.

Baseline `fa4-tuned`: FlashAttention-4 driven at the per-shape winning
configurations from `helion_paper/data/attention_fa4_tuned_gb300.csv`. All eight
recorded plans use `tile_mn=[128, 160]`. This is ~10% faster than
`flash_attn_func`'s built-in heuristic, so comparing against the default would
understate the baseline.

Every number is the pooled median of all `do_bench` samples from interleaved
Helion/FA4 rounds on one idle GPU. Interleaving matters: GB300 GPUs on a board
share a power envelope, and Helion's samples are bimodal against the 1400 W cap,
so single-batch readings on these shapes move by several percent.

| step | geomean | what |
| --- | ---: | --- |
| 10 | 0.950 | `origin/main` seeds |
| 11 | 0.975 | wide KV tile for dense (width dimension, masked tail, dense-256K resident seed) |
| 12 | 1.000 | causal resident lowering gated on its schedule; two stale `kv_stage` seeds retuned |
| 13 | 1.006 | KV-width legality fixes; dense seeds on the one-CTA pipeline |
| 14 | **1.008** | causal 64K on the resident lowering -- all eight shapes beat tuned FA4 |

`step14` is the final state. Its numbers pool every interleaved batch whose
emitted kernel is identical, giving 54-63 samples per shape, because Helion is
bimodal *per process* against the 1400 W cap: on dense 64K one process reads
50.57 ms and the next 54.0 ms, a 6.7% spread, while FA4 holds 51.2-51.3 ms.
Three-round batches on these shapes are not enough to rank anything.
