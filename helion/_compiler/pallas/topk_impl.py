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
  2. Select the top-k of the `num_bins` per-bin maxima. Two implementations:
     - "bitonic" (DEFAULT, for rows <= 128): a log^2(N)-depth parallel bitonic
       top-k ported from tallax's compressed-transpose kernel (see below). ~2.6x
       faster stage-2 on TPU than the iterative path (e.g. 381us -> 146us at
       rows=128, V=202048, k=64), self-contained (no tallax dependency).
     - "iterative" (fallback, for rows > 128): O(k) rounds of masked-argmax. The
       sequential reductions are the whole perf gap vs bitonic.
     HELION_TOPK_STAGE2 = "bitonic" | "iterative" forces one path (default "auto").
Both select EXACTLY from the same per-bin maxima, so recall/values are identical;
only the stage-2 speed differs. Recall ~= `recall_target` (approximate, like
tallax; the top-1 value is exact).
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

from jax import lax
import jax.numpy as jnp

if TYPE_CHECKING:
    import jax

_NUM_LANES = 128
NUM_LANES = 128
NUM_SUBLANES = 8


def num_bins_for(k: int, vocab: int, recall_target: float = 0.99) -> int:
    """tallax's TPU-KNN recall formula (approx_max_k.py): ceil((k-1)/(1-recall)),
    rounded up to a lane multiple and capped at the padded vocab. Default recall
    is 0.99 -- callers that feed the top-k into an exact threshold (e.g. a top-p
    nucleus) need high recall; raising it costs more bins (i.e. more work)."""
    if k <= 1:
        nb = 1
    else:
        nb = math.ceil((k - 1) / (1.0 - recall_target))
    nb = ((nb + _NUM_LANES - 1) // _NUM_LANES) * _NUM_LANES
    cap = ((vocab + _NUM_LANES - 1) // _NUM_LANES) * _NUM_LANES
    return min(nb, cap)


# ===========================================================================
# Bitonic top-k (stage 2) -- ported from tallax's compressed-transpose kernel
# (tallax/tax/bitonic/{sort,topk}.py + utils.py), keeping tallax's exact index
# math and comparison ops, specialized to: 2D (rows, N), rows <= 128, N a
# multiple of 128; num_keys=1 (sort by vals, carry idx); axis=1; DESCENDING; k a
# power of two.
#
# Why compressed transpose: a naive (rows, N) reshape/roll compare-swap puts the
# sort dim on the LANE axis, so sub-lane (stride < 128) compare-swaps need
# sub-lane movement inside a 128-lane register -- Mosaic cannot lower that
# ("Invalid input layout"). tallax moves the SORT dim onto sublanes+tiles and the
# BATCH dim onto lanes, so compare-swaps are cross-sublane / cross-tile (which
# Mosaic lowers). The only cross-lane permute is the coarse "lane merges", reached
# only when rows < 128 (batch padded below a full lane).
# ===========================================================================


def _log2_ceil(x: int) -> int:
    """Ceiling of log2(x) (int -> int)."""
    if x == 0:
        return 0
    return math.ceil(math.log2(x))


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _ceil_multiple(i: int, n: int) -> int:
    return _cdiv(i, n) * n


def _dtype_info(dtype):  # type: ignore[no-untyped-def]
    if jnp.issubdtype(dtype, jnp.floating):
        return jnp.finfo(dtype)
    return jnp.iinfo(dtype)


def _to_32bit_dtype(dtype):  # type: ignore[no-untyped-def]
    if jnp.issubdtype(dtype, jnp.floating):
        return jnp.float32
    if jnp.issubdtype(dtype, jnp.integer):
        return jnp.int32
    if jnp.issubdtype(dtype, jnp.bool_):
        return jnp.int32
    raise ValueError(f"dtype not recognized: {dtype}")


def _pad_min(arr, target_shape):  # type: ignore[no-untyped-def]
    """Pad a 2D array up to target_shape with the dtype MIN (DESC top-k => padded
    entries must be smallest so they never enter the top-k)."""
    pad_widths = [(0, target_shape[d] - arr.shape[d]) for d in range(arr.ndim)]
    if all(w == (0, 0) for w in pad_widths):
        return arr
    min_val = _dtype_info(arr.dtype).min
    return jnp.pad(arr, pad_widths, mode="constant", constant_values=min_val)


def _iota_tile(dim, tile_shape=(NUM_SUBLANES, NUM_LANES)):  # type: ignore[no-untyped-def]
    if dim == 0:
        col = jnp.arange(tile_shape[0], dtype=jnp.int32).reshape(tile_shape[0], 1)
    else:
        col = jnp.arange(tile_shape[1], dtype=jnp.int32).reshape(1, tile_shape[1])
    return jnp.broadcast_to(col, tile_shape)


def _bit_indicator(bit_position: int, index):  # type: ignore[no-untyped-def]
    return (index & (1 << bit_position)) > 0


def _split_array_to_tiles(arr, tile_shape=(NUM_SUBLANES, NUM_LANES)):  # type: ignore[no-untyped-def]
    tile_dim0, tile_dim1 = tile_shape
    tile_rows = arr.shape[0] // tile_dim0
    tile_cols = arr.shape[1] // tile_dim1
    assert arr.shape[0] % tile_dim0 == 0 and arr.shape[1] % tile_dim1 == 0
    out = []
    for row_strip in jnp.split(arr, tile_rows, axis=0):
        out.extend(jnp.split(row_strip, tile_cols, axis=1))
    return out


def _to_compressed_transpose(arr):  # type: ignore[no-untyped-def]
    """(batch_size, dim1) -> (full_size, 128) compressed transpose."""
    dim0, dim1 = arr.shape
    assert NUM_LANES % dim0 == 0 and dim0 <= NUM_LANES
    assert dim1 >= NUM_LANES, "simplified: N (padded) is always >= 128"
    arrs = jnp.split(arr, NUM_LANES // dim0, axis=1)
    return jnp.concatenate(arrs, axis=0).T  # transpose: lane<->sublane


def _from_compressed_transpose(tiles, dim0):  # type: ignore[no-untyped-def]
    """Inverse of _to_compressed_transpose for the final (top-k) tiles."""
    assert NUM_LANES % dim0 == 0 and dim0 <= NUM_LANES
    arr = jnp.concatenate(tiles, axis=0)
    original_dim1 = arr.shape[0]
    if original_dim1 < NUM_LANES:
        arr = jnp.pad(arr, [(0, NUM_LANES - original_dim1), (0, 0)])
    arr = arr.T  # transpose back
    assert arr.shape[0] == NUM_LANES
    arrs = jnp.split(arr, arr.shape[0] // dim0, axis=0)
    arr = jnp.concatenate(arrs, axis=1)
    if original_dim1 < NUM_LANES:
        arr = arr[:, :original_dim1]
    return arr


def _resplit(list_of_tiles, target_tile_dim0: int):  # type: ignore[no-untyped-def]
    """Re-tile a list of (d0,128) tiles so each is (target_tile_dim0,128). For
    this port tiles are always (8,128) and target is 8 -> a no-op; kept for
    faithfulness to tallax."""
    tiles = list(list_of_tiles)
    dim0 = tiles[0].shape[0]
    if dim0 == target_tile_dim0:
        return tiles
    if dim0 > target_tile_dim0:
        out = []
        for tile in tiles:
            out.extend(jnp.split(tile, dim0 // target_tile_dim0, axis=0))
        return out
    ll = target_tile_dim0 // dim0
    return [
        jnp.concatenate(tiles[i * ll : (i + 1) * ll], axis=0)
        for i in range(len(tiles) // ll)
    ]


def _compare_and_swap(
    v_left, i_left, v_right, i_right, *, is_descending, is_right_half=None
):  # type: ignore[no-untyped-def]
    """Compare (v_left vs v_right), conditionally swap, carrying idx (num_keys=1).
    `is_descending` is a python bool (static, per-tile) or a bool array
    (per-element). `is_right_half` (intra-tile CASE-1) marks the upper element of
    each compare pair -> returns one merged array per operand; when None
    (cross-tile CASE-2) returns a (left, right) pair per operand."""
    if is_right_half is not None:
        vl = jnp.where(is_right_half, v_right, v_left)
        vr = jnp.where(is_right_half, v_left, v_right)
    else:
        vl, vr = v_left, v_right

    if isinstance(is_descending, bool) and is_descending:
        mask = vl > vr
    else:
        mask = vr > vl
    mask = mask.astype(jnp.int32)

    if is_descending is not None and not isinstance(is_descending, bool):
        mask = mask.astype(bool) ^ is_descending.astype(bool)

    if is_right_half is None:
        return (
            jnp.where(mask, v_left, v_right),
            jnp.where(mask, i_left, i_right),
            jnp.where(mask, v_right, v_left),
            jnp.where(mask, i_right, i_left),
        )
    return jnp.where(mask, v_left, v_right), jnp.where(mask, i_left, i_right)


def _pair_slice_start(i, separation, slice_length=1):  # type: ignore[no-untyped-def]
    if slice_length > separation:
        raise ValueError(f"{separation=} must be >= {slice_length=}")
    slices_per_pair = separation // slice_length
    pair_idx = i // slices_per_pair
    slice_idx = i % slices_per_pair
    return pair_idx * 2 * separation + slice_idx * slice_length


def _is_descending(
    stage, tile_start_offset, tile_local_offset, sort_dim_offset, full_size
):  # type: ignore[no-untyped-def]
    """Bitonic sort direction for `stage` (tallax verbatim; `stage` always int).
    Returns a python bool (constant within a tile) OR a bool array (varies within
    a tile); _compare_and_swap handles both."""
    if (stage < _log2_ceil(NUM_SUBLANES)) or (stage >= _log2_ceil(full_size)):
        return _bit_indicator(stage, tile_local_offset + sort_dim_offset)
    if (stage >= _log2_ceil(NUM_SUBLANES)) and (stage < _log2_ceil(full_size)):
        return _bit_indicator(stage, tile_start_offset + sort_dim_offset)
    return _bit_indicator(
        stage, tile_start_offset + tile_local_offset + sort_dim_offset
    )


def _bitonic_substage(
    vals_tiles,
    idx_tiles,
    *,
    substage,
    batch_size,
    stage=None,
    sort_dim_offset=0,
    full_size=None,
    max_reduce=False,
):  # type: ignore[no-untyped-def]
    """One bitonic compare-swap substage over the tile lists. separation =
    2**substage. CASE-1 axis0 (sep<8): intra-tile sublane permute. CASE-1 axis1
    (sep>=full_size): cross-lane permute (only when batch_size<128). CASE-2
    (8<=sep<full_size): cross-tile elementwise. max_reduce keeps only the max
    side."""
    assert max_reduce or stage is not None
    separation = 2**substage
    if full_size is None:
        full_size = len(vals_tiles) * vals_tiles[0].shape[0]

    if separation < NUM_SUBLANES or separation >= full_size:
        axis = int(separation >= full_size)  # 0 = sublane ; 1 = cross-lane
        intra_tile_separation = (
            separation if axis == 0 else (separation * batch_size) // full_size
        )
        vals_tiles = _resplit(vals_tiles, NUM_SUBLANES)
        idx_tiles = _resplit(idx_tiles, NUM_SUBLANES)
        tile_local_offset = _iota_tile(0) + (_iota_tile(1) // batch_size) * full_size
        is_right_half = _bit_indicator(
            _log2_ceil(intra_tile_separation), _iota_tile(axis)
        )
        permutation = jnp.bitwise_xor(_iota_tile(axis), intra_tile_separation)
        out_vals = [None] * len(vals_tiles)
        out_idx = [None] * len(idx_tiles)
        for t in range(len(vals_tiles)):
            v, i = vals_tiles[t], idx_tiles[t]
            v_perm = jnp.take_along_axis(v, permutation, axis=axis)
            i_perm = jnp.take_along_axis(i, permutation, axis=axis)
            is_desc = (
                True
                if max_reduce
                else _is_descending(
                    stage=stage,
                    tile_start_offset=t * NUM_SUBLANES,
                    tile_local_offset=tile_local_offset,
                    sort_dim_offset=sort_dim_offset,
                    full_size=full_size,
                )
            )
            out_vals[t], out_idx[t] = _compare_and_swap(
                v,
                i,
                v_perm,
                i_perm,
                is_descending=is_desc,
                is_right_half=is_right_half,
            )
    else:
        vals_tiles = _resplit(vals_tiles, NUM_SUBLANES)
        idx_tiles = _resplit(idx_tiles, NUM_SUBLANES)
        tile_shape = vals_tiles[0].shape
        num_tiles = len(vals_tiles)
        tile_separation = separation // tile_shape[0]
        tile_local_offset = (
            _iota_tile(0, tile_shape)
            + (_iota_tile(1, tile_shape) // batch_size) * full_size
        )
        out_vals = [None] * num_tiles
        out_idx = [None] * num_tiles
        for i_pair in range(num_tiles // 2):
            j = _pair_slice_start(i_pair, separation=tile_separation)
            jr = j + tile_separation
            is_desc = (
                True
                if max_reduce
                else _is_descending(
                    stage=stage,
                    tile_start_offset=j * tile_shape[0],
                    tile_local_offset=tile_local_offset,
                    sort_dim_offset=sort_dim_offset,
                    full_size=full_size,
                )
            )
            nvl, nil, nvr, nir = _compare_and_swap(
                vals_tiles[j],
                idx_tiles[j],
                vals_tiles[jr],
                idx_tiles[jr],
                is_descending=is_desc,
                is_right_half=None,
            )
            out_vals[j], out_idx[j] = nvl, nil
            if not max_reduce:
                out_vals[jr], out_idx[jr] = nvr, nir

    if max_reduce:
        out_vals = [v for v in out_vals if v is not None]
        out_idx = [v for v in out_idx if v is not None]
    assert all(v is not None for v in out_vals)
    return out_vals, out_idx


def _compute_padded_shape(unpadded_dim0, unpadded_dim1, k):  # type: ignore[no-untyped-def]
    """Minimal (dim0, dim1) compatible with compressed-transpose (tallax verbatim,
    sort_axis=1 branch)."""
    if unpadded_dim0 >= NUM_LANES:
        return (
            _ceil_multiple(unpadded_dim0, NUM_LANES),
            _ceil_multiple(unpadded_dim1, max(k, NUM_SUBLANES)),
        )
    dim0s = [
        2**i
        for i in range(_log2_ceil(NUM_SUBLANES), _log2_ceil(NUM_LANES) + 1)
        if 2**i >= unpadded_dim0
    ]
    shapes = [
        (
            dim0,
            _ceil_multiple(
                _ceil_multiple(unpadded_dim1, NUM_LANES * NUM_LANES // dim0),
                max(k, NUM_SUBLANES) * NUM_LANES // dim0,
            ),
        )
        for dim0 in dim0s
    ]
    return min(shapes, key=lambda x: (x[0] * x[1], -x[0]))


def _bitonic_topk_desc(
    vals: jax.Array, idx: jax.Array, k: int
) -> tuple[jax.Array, jax.Array]:
    """Top-k of `vals` along axis=1, DESCENDING, carrying `idx`, via tallax's
    compressed-transpose bitonic. vals/idx: (rows, N) -> (rows, k). rows <= 128,
    N a multiple of 128; k a power of two, k <= N."""
    assert vals.shape == idx.shape and len(vals.shape) == 2
    rows, n = vals.shape
    unpadded_k = k
    k = 2 ** _log2_ceil(k)  # snap to power of two (idempotent if already pow2)
    unpadded_sort_dim = n
    if unpadded_k > unpadded_sort_dim:
        raise ValueError("k must be <= N")

    padded_shape = _compute_padded_shape(rows, n, k)
    vals_p = _pad_min(vals, padded_shape).astype(_to_32bit_dtype(vals.dtype))
    idx_p = _pad_min(idx, padded_shape).astype(_to_32bit_dtype(idx.dtype))
    batch_size = vals_p.shape[0]
    assert batch_size <= NUM_LANES

    vals_tiles = _split_array_to_tiles(_to_compressed_transpose(vals_p))
    idx_tiles = _split_array_to_tiles(_to_compressed_transpose(idx_p))
    num_tiles = len(vals_tiles)
    unsplit_dim0 = num_tiles * vals_tiles[0].shape[0]
    assert unsplit_dim0 % k == 0

    num_merges = _log2_ceil(unpadded_sort_dim) - _log2_ceil(k)
    num_sublane_merges = _log2_ceil(_cdiv(NUM_SUBLANES, k))
    num_lane_merges = _log2_ceil(_cdiv(unpadded_sort_dim, num_tiles * NUM_SUBLANES))
    num_tile_merges = num_merges - num_sublane_merges - num_lane_merges

    def _ss(vt, it, **kw):  # type: ignore[no-untyped-def]
        return _bitonic_substage(vt, it, batch_size=batch_size, **kw)

    def _max_reduce_stage(vt, it, reduce_stage):  # type: ignore[no-untyped-def]
        for substage in range(_log2_ceil(k))[::-1]:
            vt, it = _ss(vt, it, substage=substage, stage=reduce_stage)
        return _ss(vt, it, substage=reduce_stage, max_reduce=True)

    # build bitonic sequences up to length k/2
    for stage in range(1, _log2_ceil(k)):
        for substage in range(stage)[::-1]:
            vals_tiles, idx_tiles = _ss(
                vals_tiles, idx_tiles, substage=substage, stage=stage
            )

    # progressive cross-TILE merges first (special remainder handling)
    for _ in range(num_tile_merges):
        remainder_length = len(vals_tiles) % (2 * _cdiv(k, NUM_SUBLANES))
        rem_v = rem_i = None
        if remainder_length:
            rem_v = vals_tiles[-remainder_length:]
            rem_i = idx_tiles[-remainder_length:]
            vals_tiles = vals_tiles[:-remainder_length]
            idx_tiles = idx_tiles[:-remainder_length]
        vals_tiles, idx_tiles = _max_reduce_stage(
            vals_tiles,
            idx_tiles,
            reduce_stage=_log2_ceil(_ceil_multiple(k, NUM_SUBLANES)),
        )
        if remainder_length:
            vals_tiles = vals_tiles + rem_v
            idx_tiles = idx_tiles + rem_i

    # cross-LANE merges (only when batch_size < 128)
    for i in range(num_lane_merges)[::-1]:
        vals_tiles, idx_tiles = _max_reduce_stage(
            vals_tiles,
            idx_tiles,
            reduce_stage=_log2_ceil(_ceil_multiple(k, NUM_SUBLANES)) + i,
        )
    # intra-tile SUBLANE merges (none when k >= 8)
    for i in range(num_sublane_merges)[::-1]:
        vals_tiles, idx_tiles = _max_reduce_stage(
            vals_tiles, idx_tiles, reduce_stage=_log2_ceil(k) + i
        )

    # final sort: turn the length-k bitonic seq fully DESC (sort_dim_offset=k)
    for substage in range(_log2_ceil(k))[::-1]:
        vals_tiles, idx_tiles = _ss(
            vals_tiles,
            idx_tiles,
            substage=substage,
            stage=_log2_ceil(k),
            sort_dim_offset=k,
        )

    top_vals = _from_compressed_transpose(vals_tiles, dim0=batch_size)
    top_idx = _from_compressed_transpose(idx_tiles, dim0=batch_size)
    return top_vals[:rows, :unpadded_k], top_idx[:rows, :unpadded_k]


def divide_filter_topk(
    x: jax.Array, k: int, recall_target: float = 0.99
) -> tuple[jax.Array, jax.Array]:
    """Approximate top-k over the last axis. x: (rows, V) -> (values, indices),
    each (rows, k); values descending, indices int32 into V. k must be static."""
    rows, vocab = x.shape
    orig_dtype = x.dtype
    # Mosaic can't relayout the i1 masks produced by sub-32-bit (e.g. bf16)
    # compares in the strided loop ("Invalid relayout ... vector<8x128xi1>"), so
    # run the reduction in f32 and cast the returned values back. (tallax instead
    # packs a bf16 value + its index into a single i32.)
    if jnp.issubdtype(orig_dtype, jnp.floating) and orig_dtype != jnp.float32:
        x = x.astype(jnp.float32)
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

    # Step 2: top-k of the num_bins representatives. Default is the bitonic sort
    # (tallax-style compressed-transpose, ~2.6x faster stage-2 on TPU) whenever
    # the tile fits the compressed-transpose layout (rows <= 128 lanes); otherwise
    # fall back to the iterative masked-select. HELION_TOPK_STAGE2 = "bitonic" |
    # "iterative" forces one path (default "auto").
    _stage2 = os.environ.get("HELION_TOPK_STAGE2", "auto")
    if _stage2 == "bitonic" or (_stage2 == "auto" and rows <= NUM_LANES):
        values, indices = _bitonic_topk_desc(best_val, best_idx, k)
        return values.astype(orig_dtype), indices.astype(jnp.int32)

    # Step 2 (iterative fallback): O(k) rounds of masked argmax.
    out_vals = []
    out_idxs = []
    work = best_val
    for _ in range(k):
        m = jnp.max(work, axis=1, keepdims=True)  # (rows, 1)  R1
        hit = work == m
        # tie-break: lowest column among the maxima -> exactly one winner/row
        pick = jnp.min(jnp.where(hit, col, num_bins), axis=1, keepdims=True)  # R2
        chosen = col == pick  # (rows, num_bins)
        out_vals.append(m[:, 0])  # reuse R1 (the max IS the extracted value)
        out_idxs.append(jnp.max(jnp.where(chosen, best_idx, -1), axis=1))  # R3
        work = jnp.where(chosen, neg, work)
    values = jnp.stack(out_vals, axis=1).astype(orig_dtype)  # (rows, k)
    indices = jnp.stack(out_idxs, axis=1).astype(jnp.int32)
    return values, indices
