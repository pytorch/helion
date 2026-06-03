# Helion scaled_mm vs vLLM cutlass on Qwen3-1.7B FP8 layer GEMMs (B200)

**23 WIN, 2 MATCH, 7 MISS** of 32 shapes (interleaved cudagraph timing; ratio = helion / vLLM, lower is better).

Two regimes, split by arithmetic intensity (M*N*K). **Memory-bound** (low FLOP -- small M, or small N/K): split-K + ping-pong (`into`) wins decisively (down to 0.64x) -- split-K fills the machine and the atomic output memset is overlapped on a side stream. **Compute-bound** (high FLOP -- M=512 prefill, or large N=12288 / K=6144 at M>=32): Helion's Triton GEMM trails vLLM's cutlass Blackwell FP8 kernel by ~1.1-1.8x (the misses). This is MMA-pipeline efficiency (cutlass's tcgen05 Blackwell GEMM), not algorithm, and is the known Triton-vs-cutlass gap at compute-bound FP8 -- it does not close with config tuning (verified across tile shapes, persistent kernels, TMA, warp specialization, and full native autotune). All numbers use interleaved timing (vLLM and Helion alternated each rep) so GPU-clock drift cancels.

| M | K | N | vLLM (us) | Helion (us) | ratio | result | kernel |
|--:|--:|--:|----------:|------------:|------:|:------:|:-------|
| 1 | 2048 | 4096 | 3.878 | 3.072 | 0.792x | WIN | `into sk=16 bs=[16, 256, 128]` |
| 1 | 2048 | 2048 | 3.876 | 2.502 | 0.645x | WIN | `into sk=16 bs=[16, 256, 128]` |
| 1 | 2048 | 12288 | 6.260 | 5.122 | 0.818x | WIN | `into sk=8 bs=[16, 256, 128]` |
| 1 | 6144 | 2048 | 5.345 | 4.899 | 0.917x | WIN | `into sk=16 bs=[16, 256, 128]` |
| 2 | 2048 | 4096 | 3.887 | 3.072 | 0.790x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 2 | 2048 | 2048 | 3.876 | 2.470 | 0.637x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 2 | 2048 | 12288 | 6.119 | 5.128 | 0.838x | WIN | `into sk=8 bs=[16, 128, 128]` |
| 2 | 6144 | 2048 | 5.344 | 4.353 | 0.815x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 4 | 2048 | 4096 | 3.895 | 3.098 | 0.795x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 4 | 2048 | 2048 | 3.876 | 2.472 | 0.638x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 4 | 2048 | 12288 | 6.078 | 5.140 | 0.846x | WIN | `into sk=8 bs=[16, 128, 128]` |
| 4 | 6144 | 2048 | 5.345 | 4.356 | 0.815x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 8 | 2048 | 4096 | 3.894 | 3.108 | 0.798x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 8 | 2048 | 2048 | 3.875 | 2.505 | 0.646x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 8 | 2048 | 12288 | 5.814 | 5.171 | 0.890x | WIN | `into sk=4 bs=[16, 128, 128]` |
| 8 | 6144 | 2048 | 5.344 | 4.356 | 0.815x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 16 | 2048 | 4096 | 3.904 | 3.263 | 0.836x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 16 | 2048 | 2048 | 3.877 | 2.627 | 0.677x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 16 | 2048 | 12288 | 5.704 | 5.225 | 0.916x | WIN | `into sk=4 bs=[16, 128, 128]` |
| 16 | 6144 | 2048 | 5.344 | 4.423 | 0.828x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 32 | 2048 | 4096 | 4.024 | 4.163 | 1.034x | MATCH | `into sk=8 bs=[16, 128, 128]` |
| 32 | 2048 | 2048 | 3.998 | 3.140 | 0.785x | WIN | `into sk=8 bs=[16, 128, 128]` |
| 32 | 2048 | 12288 | 6.488 | 7.291 | 1.124x | MISS | `into sk=4 bs=[32, 128, 128]` |
| 32 | 6144 | 2048 | 5.830 | 5.265 | 0.903x | WIN | `into sk=16 bs=[16, 128, 128]` |
| 64 | 2048 | 4096 | 4.174 | 4.545 | 1.089x | MISS | `into sk=4 bs=[64, 128, 256]` |
| 64 | 2048 | 2048 | 4.119 | 4.130 | 1.003x | MATCH | `into sk=8 bs=[16, 128, 128]` |
| 64 | 2048 | 12288 | 5.947 | 6.529 | 1.098x | MISS | `plain sk=1 bs=[64, 128, 128]` |
| 64 | 6144 | 2048 | 5.861 | 5.692 | 0.971x | WIN | `into sk=16 bs=[64, 128, 256]` |
| 512 | 2048 | 4096 | 6.518 | 9.049 | 1.388x | MISS | `plain sk=1 bs=[128, 128, 128]` |
| 512 | 2048 | 2048 | 5.826 | 6.838 | 1.174x | MISS | `plain sk=1 bs=[128, 64, 128]` |
| 512 | 2048 | 12288 | 14.983 | 26.514 | 1.770x | MISS | `plain sk=1 bs=[128, 256, 128]` |
| 512 | 6144 | 2048 | 10.942 | 15.086 | 1.379x | MISS | `plain sk=1 bs=[128, 64, 128]` |
