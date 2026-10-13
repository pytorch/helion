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
