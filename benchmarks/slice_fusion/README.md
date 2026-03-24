# Slice Fusion Benchmark: Helion vs Inductor

## Context

**Issue**: [pytorch/helion#1679 — Enable Partial Tile Store/Loading](https://github.com/pytorch/helion/issues/1679)
**Internal diff**: D96227354 — `[Helion][LCE] support slice fusion for wukong rocs`

### Problem

In LCE (Linear Combination Embedding) models, a common pattern is:

```python
output = A^T @ B^T + bias      # matmul
sliced = output[:, :slice_size, :]  # followed by a slice
```

Fusing the slice into the matmul kernel avoids materializing the full output. However,
Helion's current approach uses `if tile_n.end <= slice_size` to guard the slice store,
which **silently drops boundary tiles** when `slice_size` is not a multiple of the
tile size.

### Two Approaches Compared

1. **Helion-Original** (from issue/diff): Conditional `tile_n.end <= slice_size` store
   inside the kernel. Fast when it works, but has the partial tile correctness bug.

2. **Helion-View** (inductor-style): Compute the full output in the kernel, return the
   slice as a zero-cost `output[:, :slice_size, :]` view outside the kernel. Always
   correct, no conditional branches in the hot loop.

### Shape Under Test

From D96227354 benchmark — the shape where Inductor has its largest backward advantage:

```
B=4096, M=128, N=192, K=48, slice_size=96
transpose_A=True, transpose_C=True, 1D bias
```

D96227354 reported: **Inductor=0.699ms, Helion=1.108ms** (backward, Inductor 1.6x faster)

## How to Run

### On H100 with denoising (recommended)

```bash
# Forward benchmark
CUDA_VISIBLE_DEVICES=7 triton/scripts/denoise.sh \
  python benchmarks/slice_fusion/bench_fwd.py

# Backward benchmark
CUDA_VISIBLE_DEVICES=7 triton/scripts/denoise.sh \
  python benchmarks/slice_fusion/bench_bwd.py
```

### Quick run (skip autotuning)

```bash
CUDA_VISIBLE_DEVICES=7 HELION_USE_DEFAULT_CONFIG=1 \
  python benchmarks/slice_fusion/bench_fwd.py

CUDA_VISIBLE_DEVICES=7 HELION_USE_DEFAULT_CONFIG=1 \
  python benchmarks/slice_fusion/bench_bwd.py
```

### Run both

```bash
CUDA_VISIBLE_DEVICES=7 triton/scripts/denoise.sh bash -c '
  python benchmarks/slice_fusion/bench_fwd.py && \
  python benchmarks/slice_fusion/bench_bwd.py
'
```

## Preliminary Results (devgpu, not denoised)

### Backward — shape (4096, 128, 192, 48, 96)

| Approach | Latency (ms) | vs Inductor |
|----------|-------------|-------------|
| Inductor (torch.compile) | 0.220 | 1.00x |
| Helion-View (pre-accumulate) | 0.484 | 0.46x |
| Helion-Orig (inline slice) | 0.834 | 0.26x |

### Key Findings

1. **Helion-View is 1.7x faster than Helion-Original** — removing the conditional
   `tile_n.end <= slice_size` branch and duplicate d_sliced loads helps significantly.

2. **Inductor still wins ~2.2x** over the best Helion approach, likely because it uses
   cuBLAS for the batched matmuls which is highly optimized for small K=48.

3. **Helion-View is always correct** — no partial tile bug since the slice is a view.

### Generated Code Comparison

**torch.compile (Inductor)** decomposes into 4 kernel launches:
- 2 pointwise dtype-cast kernels (A fp32→bf16, B fp32→bf16)
- 1 cuBLAS `bmm` call (the actual matmul)
- 1 fused transpose + bias-add + contiguous kernel
- Slice is a zero-cost `reinterpret_tensor` view

**Helion** generates 1 single fused TTIR kernel:
- All operations (cast, matmul via `tt.dot`, bias-add, transpose, store) in one kernel
- No intermediate buffers — all computation in registers
- Uses `scf.if` for the conditional slice store (Original) or no branching (View)

## Files

- `bench_fwd.py` — Forward kernel benchmark (matmul + bias + slice)
- `bench_bwd.py` — Backward kernel benchmark (d_A, d_B, d_bias with slice gradient)
- `README.md` — This file
