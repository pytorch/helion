"""Diagnose the 2.5% inaccuracy in examples/jagged_sum_tpu.py.

Calls the user's pure-Helion DSL kernel at multiple seeds, compares output
to an fp64 reference computed on CPU, and reports:

  - max|out - ref| per seed
  - count and percentage of cells exceeding (rtol=1e-3, atol=1e-3)
  - first ~10 bad cell coordinates with (ref, out, diff)
  - per-row bad-cell counts (clusters by item index if it's a race)
  - whether the set of bad cells is the same across seeds (deterministic
    codegen bug) or shifts (race / fp32 non-determinism)

Two diagnostic axes the output answers:

  (a) Race condition on `out[item_idx, tile_m] += partial` across L-tiles
      → bad cells SHIFT across seeds; per-row counts are concentrated on
        the *longest* items (most L-tiles touching them).

  (b) Deterministic codegen bug (e.g. wrong masking)
      → bad cells stay the SAME across seeds; pattern is structural
        (e.g. only rows in certain ranges).

The HELION_PRINT_OUTPUT_CODE=1 in run.sh also dumps Helion's lowered
Pallas code into the same log so we can read the out_spec / index_map.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

# Allow importing examples.jagged_sum_tpu and the sibling kernel_variants
# module when running from this dir.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from helion._testing import DEVICE  # noqa: E402
from kernel_variants import VARIANTS  # noqa: E402


def reference_fp64(x_data: torch.Tensor, x_offsets: torch.Tensor) -> torch.Tensor:
    """fp64 reference on CPU. Cast back to input dtype/device at the end."""
    x64 = x_data.detach().cpu().to(torch.float64)
    off_cpu = x_offsets.detach().cpu()
    nrows = int(off_cpu.numel() - 1)
    M = int(x_data.size(1))
    out64 = torch.zeros((nrows, M), dtype=torch.float64)
    for i in range(nrows):
        s = int(off_cpu[i])
        e = int(off_cpu[i + 1])
        if e > s:
            out64[i] = x64[s:e].sum(dim=0)
    return out64.to(x_data.dtype).to(x_data.device)


def make_inputs(seed: int, B: int = 8, M: int = 128, max_seqlen: int = 64):
    """Mirror the user's `examples/jagged_sum_tpu.main()` setup, parameterized."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    seq_lengths = torch.randint(1, max_seqlen + 1, (B,), generator=g)
    x_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            torch.cumsum(seq_lengths, dim=0),
        ]
    ).to(torch.int32).to(DEVICE)

    nnz = int(x_offsets[-1])
    x_data = torch.randn(nnz, M, generator=g).to(DEVICE)

    # Mirror the user's MAX_BLOCK_SIZE=1024 padding.
    MAX_BLOCK_SIZE = 1024
    padded_L = ((nnz + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE) * MAX_BLOCK_SIZE
    x_padded = torch.zeros((padded_L, M), dtype=x_data.dtype, device=x_data.device)
    x_padded[:nnz, :] = x_data
    return x_padded, x_offsets, seq_lengths.tolist()


def analyze(
    variant_name: str,
    kernel_fn,
    seed: int,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> dict:
    print(
        f"\n{'='*78}\n=== variant={variant_name!r} seed={seed}\n{'='*78}",
        flush=True,
    )
    x_padded, offsets, seq_lengths = make_inputs(seed)
    print(
        f"INPUT  x_padded.shape={tuple(x_padded.shape)}  L_nnz={int(offsets[-1])}",
        flush=True,
    )
    print(f"INPUT  offsets={offsets.tolist()}", flush=True)
    print(f"INPUT  seq_lengths={seq_lengths}", flush=True)

    try:
        out = kernel_fn(x_padded, offsets)
    except Exception:
        print("KERNEL FAILED — full traceback:", flush=True)
        traceback.print_exc()
        return {"variant": variant_name, "seed": seed, "n_bad": None, "bad_idx": None}

    ref = reference_fp64(x_padded, offsets)
    diff = (out - ref).abs()
    max_diff = float(diff.max().item())
    print(f"RESULT max|out - ref(fp64)| = {max_diff:.6e}", flush=True)

    # Cell-level threshold (matches our template-test tolerance).
    threshold = atol + rtol * ref.abs()
    bad = (diff > threshold)
    n_bad = int(bad.sum().item())
    total = int(bad.numel())
    pct = 100.0 * n_bad / total if total else 0.0
    print(f"RESULT bad cells: {n_bad} / {total}  ({pct:.2f}%)  "
          f"@ rtol={rtol} atol={atol}", flush=True)

    # Sample first ~10 bad cells.
    bad_idx_list = []
    if n_bad > 0:
        bad_idx = torch.nonzero(bad.cpu()).tolist()
        print(f"RESULT first {min(10, len(bad_idx))} bad cells:", flush=True)
        for r, c in bad_idx[:10]:
            rv = float(ref[r, c].item())
            ov = float(out[r, c].item())
            dv = float(diff[r, c].item())
            print(
                f"   [{r:3d}, {c:3d}]  ref={rv:+.6e}  out={ov:+.6e}  diff={dv:.6e}",
                flush=True,
            )
        bad_idx_list = bad_idx

        # Per-row counts: a race shows up as elevated counts on rows
        # corresponding to the LONGEST items (most L-tiles touching them).
        bad_per_row = bad.sum(dim=1).cpu().tolist()
        print(f"RESULT bad cells per row: {bad_per_row}", flush=True)
        print(f"RESULT seq_lengths        : {seq_lengths}", flush=True)
    else:
        print("RESULT PASS (no bad cells)", flush=True)

    return {"variant": variant_name, "seed": seed, "n_bad": n_bad, "bad_idx": bad_idx_list}


def main() -> int:
    import jax

    print("jax", jax.__version__, flush=True)
    print("jax devices:", jax.devices(), flush=True)
    print("torch device (DEVICE):", DEVICE, flush=True)

    # Sweep all three pallas_loop_type variants on the same data, same seed.
    # Single seed is enough — the bug is deterministic, and using only one
    # seed keeps the log tractable.
    seed = 0
    all_results = []
    for variant_name, kernel_fn in VARIANTS.items():
        all_results.append(analyze(variant_name, kernel_fn, seed))

    print(f"\n{'='*78}\n=== variant summary (seed={seed})\n{'='*78}", flush=True)
    for r in all_results:
        print(
            f"variant={r['variant']:>14}  n_bad={r['n_bad']}",
            flush=True,
        )

    # Cross-variant bad-cell set comparison. If two variants have identical
    # bad-cell sets, the bug is in the upstream Helion source-to-Pallas
    # lowering (independent of loop_type). If a variant has zero bad cells,
    # the bug is loop_type-specific (and that variant is the workaround).
    results = all_results
    if all(r["n_bad"] is not None for r in results):
        sets = [{tuple(c) for c in (r["bad_idx"] or [])} for r in results]
        for i in range(1, len(sets)):
            same = sets[i] == sets[0]
            inter = len(sets[i] & sets[0])
            print(
                f"  variant[0]={results[0]['variant']!r} vs "
                f"variant[{i}]={results[i]['variant']!r}: "
                f"|same|={same}  |intersection|={inter}  "
                f"|sym_diff|={len(sets[i] ^ sets[0])}",
                flush=True,
            )

    any_bad = any(r["n_bad"] for r in results if r["n_bad"] is not None)
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
