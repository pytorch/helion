# jagged_sum Pallas lowering experiment

Probes whether Helion's existing Pallas backend can lower a slab-style
`jagged_sum` source into Mosaic-friendly code (BlockSpec + BoundedSlice +
`pl.ds`), without any compiler changes.

## What this answers

The question on the table: can we delay the
`helion/_compiler/pallas/templates/jagged_reduce.py` plan by rewriting
just the Helion source for `jagged_sum`, and let the existing Pallas
backend handle the rest?

To answer it we need to know **what the Pallas backend emits today** for
four progressively-more-slab-friendly source forms:

| Variant | Source pattern | What it tests |
|---|---|---|
| `v1_gather` | Existing example: `hl.load(x_flat, [flat])` over `[tile_b, tile_k, tile_m]` | Baseline — does the current source even compile on Pallas? |
| `v2_2d_index` | Same shape, but `x_data[base, :]` instead of flat `hl.load` | Does dropping the explicit flatten change the lowering? |
| `v3_slab_grid` | `for i in hl.grid(num_rows): x_data[start:end, :].sum(0)` | Cleanest slab form — does Helion accept dynamic slicing? |
| `v4_grid_jagged_tile` | `hl.grid(num_rows)` + inner `hl.jagged_tile(nnz)` with scalar-start fancy index | Hybrid: per-batch program, tile-based inner load with a known scalar base |

Each variant runs on randomized inputs (`B=8, M=128, max_seqlen=64`) and
asserts numerical correctness against a CPU reference.

## How to run on a TPU VM

```bash
# from this directory on the TPU VM
bash run.sh
```

The script:

1. Dumps environment info (Python/JAX/Pallas/Helion versions, TPU
   detection, `BoundedSlice`/`ds` availability) to `outputs/00_env.log`.
2. Runs each variant with:
   - `HELION_BACKEND=pallas`
   - `HELION_PRINT_OUTPUT_CODE=1` — emits the lowered Pallas code
   - `HELION_DEBUG_DTYPE_ASSERTS=1`
   - `HELION_LOGS=+pallas`
3. Captures stdout+stderr of each run into `outputs/<variant>.log`.
4. Writes a one-line-per-variant summary to `outputs/_summary.txt`.
5. Tars `outputs/` into `jagged_sum_pallas_outputs.tar.gz` for push-back.

## Pushing outputs back

After `run.sh` finishes, copy the tarball back to the host:

```bash
# from the TPU VM, replace HOST and PATH appropriately
scp jagged_sum_pallas_outputs.tar.gz HOST:PATH/helion/tmp/jagged_sum_pallas_experiment/
```

Or just `cat outputs/*.log` into a paste — the per-variant logs are the
artifact we need to see.

## Expected outcomes (priors)

- **v1**: likely produces gather-style Pallas that Mosaic either refuses
  or runs slowly. We mostly want to see *what* it emits.
- **v3**: most likely to fail at compile time — Helion may not support
  dynamic slicing on loaded scalar bounds. If it *does* compile and emits
  `BlockSpec(BoundedSlice(...))`, we're done — that's the cheapest path.
- **v4**: most likely to actually run if v3 doesn't, because the inner
  load is shape-known per program; whether it emits a slab DMA or a row
  loop is the interesting question.

What we're looking for in the printed code:

- **Best case**: `pl.BlockSpec((pl.BoundedSlice(...), 128), ...)` —
  Helion already lifts the slab access into a BlockSpec.
- **Acceptable**: `pltpu.make_async_copy` + `pl.ds(start, size)` inside
  the kernel — Helion does the orchestration itself.
- **Bad**: a per-element gather loop or a `tl.load` flavored call — that
  means we still need the template plan.

## Files

- `variants/_common.py` — shared input gen + reference
- `variants/v1_gather.py` …  `v4_grid_jagged_tile.py` — the four source forms
- `run.sh` — TPU-VM-side harness
- `outputs/` — written by `run.sh`; what to push back
