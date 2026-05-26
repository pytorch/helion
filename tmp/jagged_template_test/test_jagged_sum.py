"""Standalone correctness test for helion.runtime.pallas_templates.jagged_sum_pallas.

Runs the template directly with JAX arrays (no Helion compiler or torch in the
loop) so we get a clean signal on whether the BlockSpec + BoundedSlice + pl.ds
shape actually works on TPU. Once this passes we layer the Helion compiler
integration on top.
"""
from __future__ import annotations

import sys
import traceback
from typing import Any

import numpy as np


def make_inputs(
    B: int,
    M_actual: int,
    max_seq: int = 32,
    seq_lengths: list[int] | None = None,
    seed: int = 0,
) -> tuple[Any, Any, list[int], int, int]:
    """Build (x_padded[L, M_padded], offsets[B+1], seq_lengths, M_actual, M_padded).

    Pads M_actual up to a multiple of 128 (lane alignment) using zero — zero
    is the identity for sum, so padded lanes contribute nothing.
    """
    import jax.numpy as jnp

    rng = np.random.RandomState(seed)
    if seq_lengths is None:
        seq_lengths = rng.randint(1, max_seq + 1, size=B).astype(np.int32).tolist()
    seq_lengths_np = np.asarray(seq_lengths, dtype=np.int32)
    offsets_np = np.concatenate([[0], np.cumsum(seq_lengths_np)]).astype(np.int32)
    L = int(offsets_np[-1])

    M_padded = ((M_actual + 127) // 128) * 128
    x = rng.randn(L, M_actual).astype(np.float32)
    if M_padded != M_actual:
        x_padded_np = np.zeros((L, M_padded), dtype=np.float32)
        x_padded_np[:, :M_actual] = x
    else:
        x_padded_np = x

    x_padded = jnp.asarray(x_padded_np)
    offsets = jnp.asarray(offsets_np, dtype=jnp.int32)
    return x_padded, offsets, seq_lengths, M_actual, M_padded


def reference_numpy(x_np: np.ndarray, offsets_np: np.ndarray) -> np.ndarray:
    B = len(offsets_np) - 1
    M = x_np.shape[1]
    out = np.zeros((B, M), dtype=x_np.dtype)
    for i in range(B):
        s, e = int(offsets_np[i]), int(offsets_np[i + 1])
        if e > s:
            out[i] = x_np[s:e].sum(axis=0)
    return out


SIZES: list[tuple[str, dict]] = [
    # Matches the HTML walkthrough: B=4, ni=[3,7,5,9], M_actual=8 → M_padded=128.
    # Exercises the M-padding path and an irregular seq distribution.
    ("html_example", dict(B=4, M_actual=8, seq_lengths=[3, 7, 5, 9])),
    # Tiny: a few items, M is already lane-aligned.
    ("tiny",   dict(B=4,  M_actual=128, max_seq=16)),
    # Small: realistic-looking single-batch sizes.
    ("small",  dict(B=8,  M_actual=128, max_seq=64)),
    # Medium: stresses scratch + buffer count.
    ("medium", dict(B=32, M_actual=128, max_seq=512)),
    # Large: full single-tile pressure (~32 KiB per VMEM block at k_sz=64).
    # Not "huge" — that bracket is for once we get past the basics.
    ("large",  dict(B=128, M_actual=128, max_seq=2048)),
]

K_SIZES = [16, 64]


def run_one(name: str, params: dict, *, k_sz: int) -> int:
    print("=" * 78, flush=True)
    print(f"=== size '{name}' params={params} k_sz={k_sz}", flush=True)
    print("=" * 78, flush=True)
    try:
        x_padded, offsets, seq_lengths, M_actual, M_padded = make_inputs(**params)
    except Exception:
        print("INPUT-GEN FAILED — full traceback:", flush=True)
        traceback.print_exc()
        return 3

    print(f"x_padded.shape={tuple(x_padded.shape)} dtype={x_padded.dtype}", flush=True)
    print(f"offsets={offsets.tolist()}  L={int(offsets[-1])}", flush=True)
    print(f"seq_lengths={seq_lengths}", flush=True)
    print(f"M_actual={M_actual} M_padded={M_padded}", flush=True)

    # Import inside the function so that import-time errors land in the per-size
    # log section (helps when the template itself fails to import).
    try:
        from helion.runtime.pallas_templates import jagged_sum_pallas
    except Exception:
        print("IMPORT FAILED — full traceback:", flush=True)
        traceback.print_exc()
        return 4

    try:
        out = jagged_sum_pallas(x_padded, offsets, k_sz=k_sz)
        out.block_until_ready()
    except Exception:
        print("KERNEL FAILED — full traceback:", flush=True)
        traceback.print_exc()
        return 1

    print(f"out.shape={tuple(out.shape)} dtype={out.dtype}", flush=True)

    try:
        x_np = np.asarray(x_padded)
        offsets_np = np.asarray(offsets)
        out_np = np.asarray(out)
        ref_np = reference_numpy(x_np, offsets_np)
        diff = float(np.max(np.abs(out_np - ref_np)))
        print(f"max|out - ref| = {diff:.6e}", flush=True)
        if M_actual < M_padded:
            tail_max = float(np.max(np.abs(out_np[:, M_actual:])))
            print(f"max|out[:, M_actual:M_padded]| = {tail_max:.6e}", flush=True)
        if diff > 1e-4:
            print(f"CORRECTNESS FAILED (max diff {diff:.6e} > 1e-4)", flush=True)
            return 2
        print("PASS", flush=True)
        return 0
    except Exception:
        print("REF CHECK FAILED — full traceback:", flush=True)
        traceback.print_exc()
        return 2


def main() -> int:
    import jax

    print(f"jax {jax.__version__}", flush=True)
    print(f"jax devices: {jax.devices()}", flush=True)

    overall_rc = 0
    for name, params in SIZES:
        for k_sz in K_SIZES:
            rc = run_one(name, dict(params), k_sz=k_sz)
            overall_rc = max(overall_rc, rc)
            print(f"SUMMARY size='{name}' k_sz={k_sz} rc={rc}", flush=True)
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
