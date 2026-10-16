"""Host-side helpers of the FlexAttention envelope's CuTe flash plan.

``flex_item_table`` builds the work-item order of the persistent list walk:
the pair items sorted longest-first by the size of their KV tile union, so
the static strided scheduler hands the longest walks to the CTAs dispatched
first and the short ones fill the tail (report 02 §6.9; the per-item trip
count of a BlockMask varies with the mask, and a plain strided order over
``(bh, pair)`` was measured 6-8% behind the non-persistent grid at 15-40%
density).

``flex_item_words`` packs every pair item's BlockMask rows into tile
bitmask words (``cute_flash_flex_item_words``) so the device roles load a
few words per item instead of rebuilding the masks from the index lists.

``flex_split_workspace`` allocates the split-KV walk's partial-output
workspace and arrival counters (``cute_flash_kv_split``).
"""

from __future__ import annotations

import torch
from torch.utils.weak import WeakIdKeyDictionary

# Cached per ``kv_num_blocks`` tensor behind the four tables' storage pointers
# and version counters; rebuilt (never cached) while the current stream is
# capturing a CUDA graph so a replay recomputes it from the tables it reads,
# exactly like ``document_tile_bounds`` / ``row_bound_work_table``. Inference
# tensors carry no version counter and are never cached.
_FLEX_ITEM_TABLE_CACHE: WeakIdKeyDictionary = WeakIdKeyDictionary()


# Split-KV workspaces per (device, stream, items, splits, head dim): the
# partial buffers are never cleared (every split writes its rows before any
# merge reads them) and the counters reset themselves (the last arriver
# zeroes its counter), so one allocation serves every launch on a stream.
_FLEX_SPLIT_WORKSPACE_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


def flex_split_workspace_elems(
    items: int, splits: int, head_dim_v: int
) -> tuple[int, int, int]:
    """Element counts of the three regions of the split-KV workspace: the
    fp32 partial output tiles, the fp32 row-statistics tiles and the int32
    arrival counters (in that order, all 4-byte elements)."""
    return (
        items * splits * 2 * 128 * head_dim_v,
        items * splits * 2 * 128 * 2,
        items * 2 * 4,
    )


def flex_split_workspace(
    like: torch.Tensor, *, items: int, splits: int, head_dim_v: int
) -> torch.Tensor:
    """Workspace of the split-KV list walk (``cute_flash_kv_split``).

    One flat fp32 buffer on ``like``'s device: ``items * splits * 2``
    unnormalized (128, head_dim_v) output tiles, then as many (128, 2)
    row-statistics tiles (row max, row sum), then ``items * 2 * 4`` arrival
    counters (per item, 128-row stage and 32-row warp group; int32 bits in
    fp32 slots, the device recasts them), zero between launches. One tensor,
    so a launch pays one host lookup and one kernel argument.

    Re-entrancy: the buffer is cached per device and CUDA stream, so
    launches on one stream (ordered) share it and launches on different
    streams get their own; two concurrent launches of the kernel on the
    SAME stream are impossible, and a launch on another stream never sees
    another stream's buffer. Under CUDA-graph capture the buffer is
    allocated inside the capture (the graph's private pool; the counters'
    zeroing becomes a memset node) and is not cached. The partial region
    holds no state across launches (uninitialized memory is only ever read
    after the writing split's release), the counters do (each returns to
    zero before its launch ends).
    """
    partial_elems, stat_elems, counter_elems = flex_split_workspace_elems(
        items, splits, head_dim_v
    )

    def build() -> torch.Tensor:
        workspace = torch.empty(
            partial_elems + stat_elems + counter_elems,
            dtype=torch.float32,
            device=like.device,
        )
        # Only the counters need a value (an fp32 zero is an int32 zero).
        workspace[partial_elems + stat_elems :].zero_()
        return workspace

    if like.is_cuda:
        if torch.cuda.is_current_stream_capturing():
            return build()
        # The raw stream handle: the Stream object constructor costs 2.5 us
        # per launch; this is the accessor inductor's wrappers use.
        stream = torch._C._cuda_getCurrentRawStream(like.device.index)
    else:
        stream = None
    key = (like.device, stream, items, splits, head_dim_v)
    cached = _FLEX_SPLIT_WORKSPACE_CACHE.get(key)
    if cached is None:
        cached = build()
        _FLEX_SPLIT_WORKSPACE_CACHE[key] = cached
    return cached


def _dense_tiles(count: torch.Tensor, index: torch.Tensor, num_kv: int) -> torch.Tensor:
    """``(Bm, Hm, R, num_kv)`` int8 membership of one BlockMask list."""
    length = index.shape[-1]
    valid = (
        torch.arange(length, device=index.device, dtype=torch.int64)
        < count.to(torch.int64)[..., None]
    )
    # Out-of-range entries past the count (and any stray id) land in a
    # scratch column that is dropped.
    target = torch.where(valid, index.to(torch.int64).clamp_(0, num_kv), num_kv)
    dense = torch.zeros(
        (*index.shape[:-1], num_kv + 1), dtype=torch.int8, device=index.device
    )
    dense.scatter_(-1, target, 1)
    return dense[..., :num_kv]


def _pack_tile_words(dense: torch.Tensor, num_kv: int) -> torch.Tensor:
    """``(..., words)`` int32 bitmask words of a ``(..., num_kv)`` 0/1
    membership: tile ``t`` is bit ``t & 31`` of word ``t >> 5`` (the device
    walk's ``_flash_flex_mask_build`` layout)."""
    words = -(-num_kv // 32)
    padded = torch.nn.functional.pad(dense.to(torch.int64), (0, words * 32 - num_kv))
    shifts = torch.arange(32, device=dense.device, dtype=torch.int64)
    packed = (padded.reshape(*dense.shape[:-1], words, 32) << shifts).sum(dim=-1)
    # Two's complement: bit 31 set is a negative int32.
    packed = torch.where(packed >= 1 << 31, packed - (1 << 32), packed)
    return packed.to(torch.int32)


def _build_flex_item_tables(
    part_cnt: torch.Tensor,
    part_idx: torch.Tensor,
    full_cnt: torch.Tensor,
    full_idx: torch.Tensor,
    *,
    q_block: int,
    batch: int,
    heads: int,
    num_pairs: int,
    num_kv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The item order and the per-item bitmask words (see ``flex_item_table``
    and ``flex_item_words``), built together from one pass over the tables."""
    # Every op below has a static shape (no ``nonzero`` / ``.item()``), so
    # the build is legal inside a CUDA-graph capture.
    assert q_block in (128, 256)
    table_batch, table_heads, rows = part_cnt.shape
    assert batch % table_batch == 0 and heads % table_heads == 0
    part = _dense_tiles(part_cnt, part_idx, num_kv)
    full = _dense_tiles(full_cnt, full_idx, num_kv)
    if q_block == 256:
        assert rows >= num_pairs
        part = part[:, :, :num_pairs]
        full = full[:, :, :num_pairs]
        pair_union = torch.maximum(part, full)
        # One row per pair: [partial words, full words].
        per_pair = torch.stack([part, full], dim=3)
    else:
        # Rows ``2p`` and ``2p + 1`` form pair ``p``; a ragged query length
        # leaves the last pair with a single row (its second stage: no tiles).
        assert num_pairs * 2 >= rows >= num_pairs * 2 - 1
        if rows < num_pairs * 2:
            pad = (0, 0, 0, num_pairs * 2 - rows)
            part = torch.nn.functional.pad(part, pad)
            full = torch.nn.functional.pad(full, pad)
        part = part.reshape(table_batch, table_heads, num_pairs, 2, num_kv)
        full = full.reshape(table_batch, table_heads, num_pairs, 2, num_kv)
        pair_union = torch.maximum(part, full).amax(dim=3)
        # Per pair: [stage-0 partial, stage-0 full, stage-1 partial, stage-1 full].
        per_pair = torch.stack(
            [part[:, :, :, 0], full[:, :, :, 0], part[:, :, :, 1], full[:, :, :, 1]],
            dim=3,
        )
    count = pair_union.to(torch.int32).sum(dim=-1)
    words = _pack_tile_words(per_pair, num_kv)  # (Bm, Hm, pairs, lists, W)
    # Broadcast tables (``Bm`` / ``Hm`` of 1, or any divisor) index with
    # ``b % Bm`` / ``h % Hm`` like the device body.
    b_index = torch.arange(batch, device=count.device) % table_batch
    h_index = torch.arange(heads, device=count.device) % table_heads
    count = count[b_index][:, h_index].reshape(-1)
    # Flat: the device indexes it linearly (``item * stride + word``); a 2-D
    # CuTe tensor would resolve a linear index through its column-major
    # layout instead.
    words = words[b_index][:, h_index].reshape(-1)
    # Longest-first, stable: equal counts keep the ``(bh, pair)`` order.
    order = torch.sort(count, descending=True, stable=True).indices
    return order.to(torch.int32), words.contiguous()


def _flex_item_tables(
    part_cnt: torch.Tensor,
    part_idx: torch.Tensor,
    full_cnt: torch.Tensor,
    full_idx: torch.Tensor,
    **kwargs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``_build_flex_item_tables`` cached per ``part_cnt`` tensor behind the
    four tables' pointers and version counters (rebuilt on every call under
    CUDA-graph capture and for inference tensors)."""
    tables = (part_cnt, part_idx, full_cnt, full_idx)
    if any(table.is_inference() for table in tables) or (
        part_cnt.is_cuda and torch.cuda.is_current_stream_capturing()
    ):
        return _build_flex_item_tables(*tables, **kwargs)
    key = (
        tuple(sorted(kwargs.items())),
        tuple((table.data_ptr(), table._version) for table in tables),
    )
    cached = _FLEX_ITEM_TABLE_CACHE.get(part_cnt)
    if cached is not None and cached[0] == key:
        return cached[1]
    built = _build_flex_item_tables(*tables, **kwargs)
    _FLEX_ITEM_TABLE_CACHE[part_cnt] = (key, built)
    return built


def flex_item_words(
    part_cnt: torch.Tensor,
    part_idx: torch.Tensor,
    full_cnt: torch.Tensor,
    full_idx: torch.Tensor,
    *,
    q_block: int,
    batch: int,
    heads: int,
    num_pairs: int,
    num_kv: int,
) -> torch.Tensor:
    """Per-item KV tile bitmasks of the flex list walk
    (``cute_flash_flex_item_words``).

    A flat ``int32`` table of ``batch * heads * num_pairs`` rows of ``lists *
    words`` entries (row ``bh * num_pairs + pair`` at ``row * lists * words``);
    ``words = ceil(num_kv / 32)``; the lists are
    ``[partial, full]`` of the pair's row for a 256-row BlockMask and
    ``[partial(2p), full(2p), partial(2p+1), full(2p+1)]`` for a 128-row one
    (a ragged last pair's missing row: zero words).  Tile ``t`` is bit
    ``t & 31`` of word ``t >> 5``, the layout the device walk builds from the
    index lists; every walking role reads its words instead of walking the
    lists (one load per word, no dependent chain).  Broadcast tables are
    expanded per ``(b, h)``.  Cached and rebuilt like ``flex_item_table``.
    """
    return _flex_item_tables(
        part_cnt,
        part_idx,
        full_cnt,
        full_idx,
        q_block=q_block,
        batch=batch,
        heads=heads,
        num_pairs=num_pairs,
        num_kv=num_kv,
    )[1]


def flex_item_table(
    part_cnt: torch.Tensor,
    part_idx: torch.Tensor,
    full_cnt: torch.Tensor,
    full_idx: torch.Tensor,
    *,
    q_block: int,
    batch: int,
    heads: int,
    num_pairs: int,
    num_kv: int,
) -> torch.Tensor:
    """Work-item order of the persistent flex list walk.

    The tables are the BlockMask's ``(Bm, Hm, R)`` counts and ``(Bm, Hm, R,
    L)`` tile-id lists (partial and full).  The result is an ``int32``
    ``(batch * heads * num_pairs,)`` tensor of item ids ``bh * num_pairs +
    pair`` sorted by the pair's KV tile union longest-first (stable); the
    device body's strided loop reads ``item = table[tile_id]`` and decodes
    ``(bh, pair)`` from it.  The order changes nothing numerically: every item
    computes exactly the rows it owns.

    Cached per ``part_cnt`` tensor behind the four tables' pointers and
    version counters; rebuilt on every call under CUDA-graph capture.
    """
    return _flex_item_tables(
        part_cnt,
        part_idx,
        full_cnt,
        full_idx,
        q_block=q_block,
        batch=batch,
        heads=heads,
        num_pairs=num_pairs,
        num_kv=num_kv,
    )[0]
