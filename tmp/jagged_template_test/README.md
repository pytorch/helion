# jagged_sum_pallas template — standalone correctness test

First-push validation of `helion/runtime/pallas_templates/jagged_reduce.py`
on a TPU VM. Bypasses the Helion compiler entirely — calls the template
function directly with JAX arrays — so the failure mode (if any) is
narrowly attributable to the kernel itself, not to the compiler
integration we layer on later.

## What it tests

`test_jagged_sum.py` runs `jagged_sum_pallas` at five input sizes and two
`block_L` values (32, 128) — the static L-block height the kernel uses
for its DMA stride:

| Size | B | M_actual | max_seq | Notes |
|---|---|---|---|---|
| `html_example` | 4 | 8 | — | `seq_lengths=[3,7,5,9]` — matches the HTML walkthrough, exercises M zero-padding (8 → 128) |
| `tiny` | 4 | 128 | 16 | M already lane-aligned |
| `small` | 8 | 128 | 64 | |
| `medium` | 32 | 128 | 512 | scratch / buffering pressure |
| `large` | 128 | 128 | 2048 | full single-tile pressure |

For each, it checks numerical equivalence to a NumPy reference and
verifies that the M-padded lanes (cols `[M_actual, M_padded)` for
`html_example`) hold zero in the output.

## Run on TPU VM

```bash
cd helion/tmp/jagged_template_test
bash run.sh
```

`run.sh` captures everything under `outputs/`:

- `00_env.log` — versions, JAX devices, BoundedSlice/ds availability,
  template-import sanity
- `test.log` — full test stdout+stderr (including Python tracebacks)
- `_summary.txt` — one-line-per-size status + last 30 lines of `test.log`
- `jagged_template_outputs.tar.gz` — same payload bundled

## Push back

Either commit `outputs/` to the branch (so the host side can `git pull`),
or scp the tarball back. Same workflow as the prior experiment dir.

## What the host will look for

The signal we care about, in order of severity:

1. **Template import fails** (`IMPORT FAILED` in test.log) — the template
   file has a Python-level error. Should be caught locally before push,
   but the env-dump's import sanity will surface it.
2. **`KERNEL FAILED` at `html_example`** — Mosaic/Pallas rejected the
   BlockSpec/BoundedSlice/ds shape. The traceback will tell us which
   piece (probably BlockSpec signature, index_map signature, or a Pallas
   call shape mismatch).
3. **`CORRECTNESS FAILED`** — masking is wrong (most likely on the
   sublane-aligned tail), or output write is wrong.
4. **Only `large` fails** — sizing/perf rather than correctness. Bracket
   the VMEM ceiling and tune `k_sz`.
5. **All pass** — proceed to Phase 2 (Helion compiler integration).
