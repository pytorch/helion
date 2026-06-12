# Helion FP8 RowWise `scaled_mm` vs vLLM (skinny M)

A Helion FP8 (e4m3) RowWise `scaled_mm` kernel that **matches or beats** vLLM's
hand-tuned cutlass kernel (`vllm._custom_ops.cutlass_scaled_mm`) on skinny-M
decode shapes on B200.

- Kernel: [`examples/scaled_mm.py`](../examples/scaled_mm.py)
- Benchmark: [`scaled_mm_vllm_benchmark.py`](scaled_mm_vllm_benchmark.py)

```
out[m, n] = scale_a[m] * scale_b[n] * sum_k a[m, k] * b[k, n]    (a,b FP8 e4m3; out BF16)
```

## Benchmark results (B200)

Interleaved timing (vLLM and Helion measured alternately each rep, 25-rep median)
inside amortized cudagraphs (64 calls/graph). All outputs correct (rel-err ~1e-2).

| Shape (M×N×K)   |   vLLM | Helion `scaled_mm_into` + ping-pong | ratio | |
|-----------------|-------:|------------------------------------:|------:|--------|
| 1 × 4096 × 4096   | 4.37µs | **3.94µs** | 0.90× | **WIN**  |
| 16 × 4096 × 4096  | 4.38µs | **4.19µs** | 0.96× | **WIN**  |
| 1 × 14336 × 4096  | 9.26µs | **8.90µs** | 0.96× | **WIN**  |
| 16 × 14336 × 4096 | 8.42µs | **8.73µs** | 1.04× | MATCH    |

**3 WINs, 1 MATCH.** Best configs: `split_k ∈ {8,16}`, `bn ∈ {64..256}`,
`bk ∈ {128,256}`, `num_stages=2`, `num_warps=4`, **pointer** loads.

The self-contained `scaled_mm` (Helion native autotuner) lands around 1.27×
(full effort) — see "Why autotuning needs help" below.

## What makes the kernel fast

This is a **memory-bandwidth-bound** problem: reading all of B (e.g. 4096×4096
FP8 = 16 MB) against ~8 TB/s HBM is a ~2µs floor; vLLM at 4.4µs ≈ 3.5 TB/s. The
contest is HBM-bandwidth utilization, not compute. Three things close the gap to
cutlass:

1. **Split-K for occupancy.** At skinny M a plain GEMM launches only `N/bn`
   CTAs (e.g. 32–112), leaving most of the 148 SMs idle, so it can't saturate
   HBM. Splitting the K reduction across CTAs (`split_k` is an `hl.register_tunable`)
   yields `N/bn × split_k` CTAs — enough parallelism to hide B-load latency and
   reach ~3 TB/s. The rowwise scale is **linear across K**, so each K-split folds
   `scale_a[m]·scale_b[n]` into its own partial and the partials simply sum.

2. **Single-kernel atomic accumulation** (no separate reduce kernel). Split-K
   partials are combined with `hl.atomic_add` directly into the output, folding
   the rowwise scale per-partial. This avoids the ~2µs tax of a second reduction
   kernel that a workspace-based split-K would pay.

3. **Overlapping the memset that atomics require** (the whole remaining gap).
   `atomic_add` needs a zeroed output. That `torch.zeros` costs ~0.74µs at M=16
   / ~1.2µs at M=1 — measured directly as the empty-vs-zeros delta, and it is
   *exactly* the gap to vLLM (without it, M=16 hits 4.33µs ≈ vLLM's 4.38µs).
   `scaled_mm_into` therefore takes a **caller-provided pre-zeroed buffer** and
   does no internal allocation/memset, so the zeroing can be hoisted out of the
   timed region and **double-buffered** ("ping-pong"): two output buffers + a
   dedicated zeroing stream re-zero buffer *j* for its next use while the current
   kernel runs. The tiny memset (128 KB) overlaps the 16 MB B read for free, and
   the kernel reaches its memset-free compute floor. This is legitimate in a real
   pipeline (the consumer reads buffer A while B is prepared).

What did **not** matter:
- **Codegen changes were unnecessary.** Helion's generated Triton for the split-K
  atomic kernel is essentially hand-written quality — coalesced bk-contiguous B
  loads, an FP8 `tl.dot`, and `tl.atomic_add` — so the win came from kernel
  *structure* and *deployment*, not from modifying the compiler.
- **TMA / `tensor_descriptor` loads were 2×+ worse** here than plain pointer
  loads, because the tiles are small and the B view is column-major.

## Why autotuning needs help

Helion's native autotuner times each config as an **isolated single call**, so it
(a) cannot observe the ping-pong deployment benefit and (b) tends to pick
`split_k=1`. The benchmark therefore tunes `scaled_mm_into` with a
**deployment-aware search**: it ranks a small candidate set by each config's
actual ping-pong cudagraph time — autotuning against the metric that matters.

## Reproduce

```bash
python benchmarks/scaled_mm_vllm_benchmark.py     # requires vLLM + a CUDA GPU
python examples/scaled_mm.py                        # correctness only (no vLLM)
```
