# SPDX-License-Identifier: Apache-2.0
"""Mosaic-friendly approximate top-k for the Pallas backend's aten.topk lowering.

`jax.lax.top_k` is not implemented in the Mosaic (Pallas/TPU) lowering, so the
Pallas backend cannot lower `aten.topk` to it. Instead we lower to this
divide-and-filter algorithm, borrowed from tallax
(https://github.com/oliverdutton/tallax, `tax/divide_and_filter_topk`): it uses
only ops Mosaic supports (strided slices, iota, where, min/max reductions), so it
compiles on TPU and, because it is plain jnp emitted inline into the kernel body,
FUSES with the surrounding kernel ops.

Algorithm (single-pass approximate path, == tallax `approx_max_k`):
  1. Divide V into `num_bins` interleaved bins (bin b = columns b, b+num_bins, ...);
     carry a per-bin running max and its GLOBAL vocab index (via where, no argmax).
  2. Select the top-k of the `num_bins` per-bin maxima by iterative masked-argmax
     (values come out descending; indices are int32 into V).
Recall ~= `recall_target` (approximate, like tallax); the top-1 is exact.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from jax import lax
import jax.numpy as jnp

if TYPE_CHECKING:
    import jax

_NUM_LANES = 128


def num_bins_for(k: int, vocab: int, recall_target: float = 0.95) -> int:
    """tallax's TPU-KNN recall formula (approx_max_k.py): ceil((k-1)/(1-recall)),
    rounded up to a lane multiple and capped at the padded vocab."""
    if k <= 1:
        nb = 1
    else:
        nb = math.ceil((k - 1) / (1.0 - recall_target))
    nb = ((nb + _NUM_LANES - 1) // _NUM_LANES) * _NUM_LANES
    cap = ((vocab + _NUM_LANES - 1) // _NUM_LANES) * _NUM_LANES
    return min(nb, cap)


def divide_filter_topk(
    x: jax.Array, k: int, recall_target: float = 0.95
) -> tuple[jax.Array, jax.Array]:
    """Approximate top-k over the last axis. x: (rows, V) -> (values, indices),
    each (rows, k); values descending, indices int32 into V. k must be static."""
    rows, vocab = x.shape
    num_bins = num_bins_for(k, vocab, recall_target)
    neg = jnp.finfo(x.dtype).min
    col = lax.broadcasted_iota(jnp.int32, (rows, num_bins), 1)  # (rows, num_bins)

    # Pad V up to a whole number of strided passes of width num_bins.
    num_slices = -(-vocab // num_bins)  # ceil
    pad = num_slices * num_bins - vocab
    if pad:
        x = jnp.pad(x, ((0, 0), (0, pad)), constant_values=neg)

    # Step 1: per-bin running max + carried global index (strided bins).
    best_val = jnp.full((rows, num_bins), neg, x.dtype)
    best_idx = jnp.zeros((rows, num_bins), jnp.int32)
    for i in range(num_slices):
        seg = x[:, i * num_bins : (i + 1) * num_bins]  # (rows, num_bins)
        gidx = i * num_bins + col  # global vocab index
        take = seg > best_val
        best_val = jnp.where(take, seg, best_val)
        best_idx = jnp.where(take, gidx, best_idx)

    # Step 2: top-k of the num_bins representatives via iterative masked select.
    out_vals = []
    out_idxs = []
    work = best_val
    for _ in range(k):
        m = jnp.max(work, axis=1, keepdims=True)  # (rows, 1)
        hit = work == m
        # tie-break: lowest column among the maxima -> exactly one winner/row
        pick = jnp.min(jnp.where(hit, col, num_bins), axis=1, keepdims=True)
        chosen = col == pick  # (rows, num_bins)
        out_vals.append(jnp.max(jnp.where(chosen, work, neg), axis=1))
        out_idxs.append(jnp.max(jnp.where(chosen, best_idx, -1), axis=1))
        work = jnp.where(chosen, neg, work)
    values = jnp.stack(out_vals, axis=1)  # (rows, k)
    indices = jnp.stack(out_idxs, axis=1).astype(jnp.int32)
    return values, indices
