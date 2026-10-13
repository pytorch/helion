"""Host-side support for the fused flash-attention document-mask path."""

from __future__ import annotations

import torch
from torch.utils.weak import WeakIdKeyDictionary

# The KV-range prelude and the per-tile predicates index the table by 128-token
# tile regardless of the configured KV tile width.
FLASH_DOCUMENT_TILE = 128

# Per document tensor: ``(num_tiles, data_ptr, _version) -> table``. Keyed by
# tensor identity (weakly, so a dead tensor drops its entry) and guarded by the
# tensor's version counter, the same contract the grouped scheduler metadata
# uses (``_tcgen05_grouped_tensor_mutation_key``): ordinary in-place torch
# writes bump ``_version`` and rebuild the table. The table decides which KV
# tiles the kernel visits and which it masks, so a stale entry silently drops
# or admits keys; two write paths bypass the guard:
#
# * CUDA-graph capture: a replay re-runs the captured launch with whatever
#   table pointer was bound at capture time, so a table reused from a warm-up
#   call would never follow ``doc.copy_()`` between replays. The table is
#   therefore always rebuilt (and never cached) while the current stream is
#   capturing: the build becomes part of the graph and every replay recomputes
#   it from the ids it reads.
# * Version-bypassing writes (raw-pointer kernels writing the ids) remain
#   unsupported, as for the grouped scheduler metadata; rewarm with the final
#   ids or write them through torch.
#
# Inference tensors carry no version counter and are never cached.
_DOCUMENT_TILE_BOUNDS_CACHE: WeakIdKeyDictionary = WeakIdKeyDictionary()


def _build_document_tile_bounds(
    document_ids: torch.Tensor, num_tiles: int
) -> torch.Tensor:
    # The flash plan requires ``seq % 128 == 0`` and sizes the table with
    # ``num_tiles = seq // 128``, so the ids reshape exactly.
    batch, seq = document_ids.shape
    assert seq == num_tiles * FLASH_DOCUMENT_TILE
    lo, hi = torch.aminmax(
        document_ids.reshape(batch, num_tiles, FLASH_DOCUMENT_TILE), dim=-1
    )
    return torch.stack((lo, hi)).to(torch.int32)


def document_tile_bounds(document_ids: torch.Tensor, num_tiles: int) -> torch.Tensor:
    """Per-tile ``[min, max]`` document-id table for the flash KV-range prelude.

    ``document_ids`` is the ``(batch, seq)`` id tensor the kernel masks with;
    the result is an ``int32`` ``(2, batch, num_tiles)`` tensor whose ``[0]`` /
    ``[1]`` planes hold the smallest / largest id of each 128-token tile. The
    device prelude keeps a KV tile whenever its id interval overlaps the query
    tile's interval. That is a superset of "the two tiles share an id" for ANY
    id assignment (a shared id lies inside both intervals), so no monotone or
    packed layout is assumed; the exact per-score document mask still runs on
    every retained tile.

    The table is cached per document tensor and rebuilt when the tensor's
    version counter (any in-place torch write) or storage changes, so the
    steady-state launch pays no extra kernels and keeps a stable table pointer.
    Under CUDA-graph capture it is rebuilt inside the captured region on every
    call (see ``_DOCUMENT_TILE_BOUNDS_CACHE``), so graph replays recompute it
    from the current ids; raw-pointer writes to the ids are not tracked.
    """
    if document_ids.is_inference() or (
        document_ids.is_cuda and torch.cuda.is_current_stream_capturing()
    ):
        return _build_document_tile_bounds(document_ids, num_tiles)
    key = (num_tiles, document_ids.data_ptr(), document_ids._version)
    cached = _DOCUMENT_TILE_BOUNDS_CACHE.get(document_ids)
    if cached is not None and cached[0] == key:
        return cached[1]
    table = _build_document_tile_bounds(document_ids, num_tiles)
    _DOCUMENT_TILE_BOUNDS_CACHE[document_ids] = (key, table)
    return table


# Segment-aligned tiling of the per-row KV lower bound (ROW_KV_BEGIN plans on
# the fa4 pair body, ``cute_flash_row_bound_tiling == "segment"``): the pair
# body walks a host-built work table instead of the global 256-row pair grid.
ROW_BOUND_PAIR_ROWS = 256
ROW_BOUND_KV_TILE = 128
ROW_BOUND_WORK_COLUMNS = 4

# Per lower-bound tensor: ``(seq, data_ptr, _version) -> table``; the same
# identity / version contract and graph-capture rule as the document table.
_ROW_BOUND_WORK_CACHE: WeakIdKeyDictionary = WeakIdKeyDictionary()


def row_bound_work_rows(batch: int, seq: int) -> int:
    """Row count of the work table: one count row plus the exact worst-case
    item count (every row its own one-pair segment)."""
    return batch * seq + 1


def row_bound_segmentation(lower_bound: torch.Tensor) -> torch.Tensor:
    """Per batch row, whether the ``[B, S]`` lower-bound table is a
    segmentation: ``lb[0] == 0`` and ``lb[q] in {lb[q - 1], q}`` for ``q >= 1``
    (so ``lb[q]`` is the start of the segment containing ``q``: non-decreasing,
    ``lb[q] <= q`` and ``lb[lb[q]] == lb[q]`` follow).  Compared in int64, so
    sentinels, negative entries and int64 tables are judged exactly."""
    bound = lower_bound.to(torch.int64)
    seq = bound.shape[1]
    rows = torch.arange(seq, device=bound.device, dtype=torch.int64)
    is_start = bound == rows
    continues = bound[:, 1:] == bound[:, :-1]
    return is_start[:, 0] & (is_start[:, 1:] | continues).all(dim=1)


def _build_row_bound_work_table(lower_bound: torch.Tensor, seq: int) -> torch.Tensor:
    # Every op below has a static shape (no ``nonzero`` / ``.item()``), so the
    # build is legal inside a CUDA-graph capture.
    batch, lb_seq = lower_bound.shape
    assert lb_seq == seq and seq % ROW_BOUND_PAIR_ROWS == 0
    device = lower_bound.device
    bound = lower_bound.to(torch.int64)
    rows = torch.arange(seq, device=device, dtype=torch.int64)
    valid = row_bound_segmentation(lower_bound)
    # Segment starts: the table's own starts for a valid row, the whole
    # sequence (one pseudo segment at row 0) for a row that is not a
    # segmentation; its items run the per-row masked body.
    starts = (bound == rows) & valid[:, None]
    starts[:, 0] = True
    seg_start = torch.cummax(torch.where(starts, rows, 0), dim=1).values
    next_start = torch.where(starts, rows, seq)
    next_start = torch.cat(
        (
            next_start[:, 1:],
            torch.full((batch, 1), seq, device=device, dtype=torch.int64),
        ),
        dim=1,
    )
    seg_end = torch.flip(torch.cummin(torch.flip(next_start, (1,)), dim=1).values, (1,))
    seg_len = seg_end - seg_start
    offset = rows - seg_start
    pair = offset // ROW_BOUND_PAIR_ROWS
    is_item = offset % ROW_BOUND_PAIR_ROWS == 0
    # Longest-first order: the pair's KV tiles (its own rows' causal extent,
    # or the segment's end for a ragged last pair).
    kv_tiles = (
        torch.minimum((pair + 1) * ROW_BOUND_PAIR_ROWS, seg_len) + ROW_BOUND_KV_TILE - 1
    ) // ROW_BOUND_KV_TILE
    work = torch.where(is_item, kv_tiles, -1).reshape(-1)
    order = torch.sort(work, descending=True, stable=True).indices
    masked = (~valid[:, None]).expand(batch, seq).to(torch.int64)
    batch_index = torch.arange(batch, device=device, dtype=torch.int64)[:, None].expand(
        batch, seq
    )
    fields = torch.stack(
        (batch_index, seg_start, seg_len, pair * 2 + masked), dim=-1
    ).reshape(-1, ROW_BOUND_WORK_COLUMNS)
    table = torch.empty(
        (row_bound_work_rows(batch, seq), ROW_BOUND_WORK_COLUMNS),
        dtype=torch.int32,
        device=device,
    )
    table[1:] = fields[order].to(torch.int32)
    table[0, 0] = is_item.sum().to(torch.int32)
    table[0, 1:] = 0
    return table


def row_bound_work_table(lower_bound: torch.Tensor, seq: int) -> torch.Tensor:
    """Work table of the segment-aligned per-row-KV-lower-bound flash plans.

    ``lower_bound`` is the ``(batch, seq)`` table the kernel masks with
    (``key >= lb[b, query]`` with the causal bound).  The result is an
    ``int32`` ``(batch * seq + 1, 4)`` tensor: row 0 holds the item count in
    column 0; rows ``1 .. count`` are the work items ``(batch row, segment
    start, segment length, 2 * pair + masked)`` in longest-first order.  A
    batch row whose table is a segmentation (``row_bound_segmentation``) is
    tiled per segment: one item per 256-row pair of each segment, ``masked``
    0, the lower bound trivially true on every key of the segment.  Any other
    batch row (arbitrary lower bounds, rows without keys, sentinels) keeps the
    global grid in the same table: the whole sequence as one segment, one item
    per 256-row pair, ``masked`` 1, and the kernel runs its per-row range mask
    on those items.  Each row contributes at least ``seq / 256`` items, so the
    count never falls below the global pair count.

    Cached per lower-bound tensor behind its version counter and rebuilt
    inside a CUDA-graph capture on every call, exactly like
    ``document_tile_bounds`` (see ``_DOCUMENT_TILE_BOUNDS_CACHE``).
    """
    if lower_bound.is_inference() or (
        lower_bound.is_cuda and torch.cuda.is_current_stream_capturing()
    ):
        return _build_row_bound_work_table(lower_bound, seq)
    key = (seq, lower_bound.data_ptr(), lower_bound._version)
    cached = _ROW_BOUND_WORK_CACHE.get(lower_bound)
    if cached is not None and cached[0] == key:
        return cached[1]
    table = _build_row_bound_work_table(lower_bound, seq)
    _ROW_BOUND_WORK_CACHE[lower_bound] = (key, table)
    return table
