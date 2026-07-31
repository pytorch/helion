# [CuTe] Beat CUTLASS across the scaled FP8 matmul sweep

## Summary

Improve Blackwell CuTe scaled-FP8 matmul code generation and extend the
production benchmark from 17 to 45 shapes. The added `M=2/8/16/32` shapes now
use tensor cores instead of the scalar skinny-M fallback. The latest 12 shapes
use FP8 `(K, N)` pairs from vLLM's H100 `scaled_mm` configuration in
vllm-project/vllm#46522.

## Codegen optimizations

### Small-N tensor-core mainloop

- Admit FP8 problems whose logical N is smaller than the minimum searched MMA
  tile.
- Pad the tcgen05 N tile to 16 columns and let TMA zero-fill columns outside
  the source tensor instead of staging the narrow operand with thousands of
  scalar loads.
- Keep the optimized path narrowly guarded to persistent, single-CTA-cluster,
  transposed-RHS problems with full M and K tiles.
- Use ceiling tile counts when validating one-shot role scheduling so a single
  partial N tile is counted correctly.
- Keep `BN < 32` outputs on the direct SIMT epilogue; the TMA-store staging and
  barriers cost more than they save for these narrow stores.

### Narrow SIMT epilogue

- Detect grids where one output dimension is smaller than its CTA tile, making
  every output tile statically an edge tile.
- Remove the unused full-tile branch for that case and share one coordinate
  partition across both scale loads and the output store.
- Fuse the predicates for the row and column scale fragments into one edge
  traversal.
- Issue scale loads before the accumulator consumer wait, overlapping global
  memory latency with completion of the tensor-core mainloop. The accumulator
  wait remains first-subtile-only for multi-subtile epilogues.
- Multiply prefetched row and column scales into the accumulator sequentially,
  avoiding an extra scale-product live range on the smallest four-CTA grid.

### Existing wide-shape improvements in this branch

- Add swapped A/B tcgen05 lowering with column-major output support.
- Support role-local persistent scheduling for two-CTA and four-CTA clusters.
- Add one-shot scheduling for device-filling static FP8 grids and bounded
  persistent scheduling for larger grids.
- Remove redundant scheduler and ownership work from validated static paths.
- Tune broadcast row/column scale handling and TMA/SIMT epilogues.

## Production tuning

- Keep `M=1` on the scalar skinny-M decode kernel.
- Dispatch all added `M=2/8/16/32` shapes through the swapped tensor-core
  kernel.
- Use `64x16x256`, one accumulator stage, and a nine-stage A/B pipeline for
  `M=2/8/16`; reduce the A/B depth to seven for `K=2048`.
- Use `64x32x128`, one accumulator stage, and a twelve-stage A/B pipeline for
  `M=32`.
- Use persistent-interleaved scheduling for `(2, 4096, 256)` and
  persistent-blocked scheduling for the other new shapes.
- Add exact AOT table entries for all 28 new shapes while preserving the
  existing M64, M512, and 4096-square configurations.
- Add `(2048, 12288)`, `(5120, 5120)`, and `(6144, 2048)` vLLM weight shapes
  at each of `M=2/8/16/32`.
- Use `64x32x256` with a seven-stage A/B pipeline for the new M=32 rows; this
  improves that subset from 0.96-1.00x to 1.02-1.04x versus CUTLASS.

## Performance

Tested on NVIDIA GB200 with cold-L2 CUDA-graph timing:

```bash
python pretuned_kernels/scale_mm_cute/scale_mm_cute.py
```

Latest complete 45-shape sweep:

- Helion vs CUTLASS: **44/45 wins**, 1.040x geomean
- Helion vs torch: **45/45 wins**, 1.434x geomean
- Helion vs best baseline: **44/45 wins**, 1.039x geomean

The 12 vLLM-derived additions are **12/12 wins** versus CUTLASS with a 1.060x
geomean. The only full-sweep miss is the pre-existing `(2, 4096, 256)` row at
0.99x in that run; five isolated reruns ranged from 0.999x to 1.015x.

Representative new shapes:

| M | K | N | Helion (us) | CUTLASS (us) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 2 | 2048 | 12288 | 8.16 | 8.40 | 1.03x |
| 8 | 6144 | 2048 | 7.24 | 7.86 | 1.09x |
| 16 | 5120 | 5120 | 8.02 | 8.45 | 1.05x |
| 32 | 2048 | 12288 | 8.21 | 9.10 | 1.11x |

## Verification

- Explicit correctness checks passed against `torch._scaled_mm` for all 28 new
  shapes (`rtol=0.03`, `atol=0.03`).
- The complete 45-shape performance sweep completed successfully.
- Added regression coverage for FP8 small-N persistent search and narrow SIMT
  epilogue load/wait ordering.
- Six focused planner, codegen-ordering, and GPU edge-correctness tests pass.
- Ruff format/check passed for every changed Python file.
- Python compile checks and `git diff --check` passed.

`pytest` is absent from the active conda environment, so the focused tests were
run by loading the existing pytest package from the `pyrefly-check` environment.
The full lint wrapper also scans untracked profiling artifacts and reports
unrelated errors there; targeted Ruff checks on all changed files pass.
