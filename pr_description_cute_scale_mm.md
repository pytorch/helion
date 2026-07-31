# [CuTe] Beat CUTLASS across the scaled FP8 matmul sweep

## Summary

Improve Blackwell CuTe scaled-FP8 matmul code generation and extend the
production benchmark from 17 to 33 shapes. The added `M=2/8/16/32` shapes now
use tensor cores instead of the scalar skinny-M fallback, and Helion is faster
than CUTLASS on every shape in the complete GB200 sweep.

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
- Add exact AOT table entries for all 16 new shapes while preserving the
  existing M64, M512, and 4096-square configurations.

## Performance

Tested on NVIDIA GB200 with cold-L2 CUDA-graph timing:

```bash
python pretuned_kernels/scale_mm_cute/scale_mm_cute.py
```

Final complete sweep:

- Helion vs CUTLASS: **33/33 wins**, 1.036x geomean
- Helion vs torch: **33/33 wins**, 1.446x geomean
- Helion vs best baseline: **33/33 wins**, 1.034x geomean

Representative new shapes:

| M | K | N | Helion (us) | CUTLASS (us) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 2 | 4096 | 256 | 4.84 | 4.87 | 1.01x |
| 8 | 4096 | 4096 | 6.78 | 6.99 | 1.03x |
| 16 | 2048 | 4096 | 5.05 | 5.30 | 1.05x |
| 32 | 4096 | 256 | 5.15 | 5.43 | 1.05x |

## Verification

- Explicit correctness checks passed against `torch._scaled_mm` for all 16 new
  shapes (`rtol=0.03`, `atol=0.03`).
- The complete 33-shape performance sweep completed successfully.
- Added regression coverage for FP8 small-N persistent search and narrow SIMT
  epilogue load/wait ordering.
- Six focused planner, codegen-ordering, and GPU edge-correctness tests pass.
- Ruff format/check passed for every changed Python file.
- Python compile checks and `git diff --check` passed.

`pytest` is absent from the active conda environment, so the focused tests were
run by loading the existing pytest package from the `pyrefly-check` environment.
The full lint wrapper also scans untracked profiling artifacts and reports
unrelated errors there; targeted Ruff checks on all changed files pass.
