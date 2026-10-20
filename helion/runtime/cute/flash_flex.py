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

# The 2-CTA flex backward's walk scratch per (device, stream, CTAs, stride):
# every CTA writes its own merged walk before it reads it back, so the buffer
# is never cleared and one allocation serves every launch on a stream.
_FLEX_BWD_WALK_SCRATCH_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


def flex_bwd_walk_scratch(
    like: torch.Tensor, *, ctas: int, stride: int
) -> torch.Tensor:
    """Scratch of the 2-CTA flex backward's merged pair walk
    (``cute_flash_bwd_flex_2cta``): ``ctas`` regions of ``stride`` int32 step
    entries on ``like``'s device, one per CTA of the launch (the CTA's
    linear block index picks its region). The prologue writes the CTA's
    whole walk there and every chunk of the step table is a copy of a window
    of it, so a long walk never rebuilds its membership bitmaps.

    Never cleared: a region is read only by the CTA that wrote it, after
    the CTA-wide barrier that follows the writes. Re-entrancy as for
    :func:`flex_split_workspace`: cached per device and CUDA stream, so
    ordered launches on one stream share it and other streams get their
    own; allocated afresh (uncached) under CUDA-graph capture.
    """

    def build() -> torch.Tensor:
        return torch.empty(ctas * stride, dtype=torch.int32, device=like.device)

    if like.is_cuda:
        if torch.cuda.is_current_stream_capturing():
            return build()
        stream = torch._C._cuda_getCurrentRawStream(like.device.index)
    else:
        stream = None
    key = (like.device, stream, ctas, stride)
    cached = _FLEX_BWD_WALK_SCRATCH_CACHE.get(key)
    if cached is None:
        cached = build()
        _FLEX_BWD_WALK_SCRATCH_CACHE[key] = cached
    return cached


def flex_tail_base(
    total_tiles: int, num_sm: int, tail_split: int, tail_items_ws: int
) -> int:
    """The first tail unit of the tail split (``cute_flash_flex_tail_split``)
    for a grid of ``total_tiles`` one-item units on a device of ``num_sm``
    SMs: the full waves' ``waves * num_sm`` units run unsplit, the last
    partial wave's items as ``tail_split`` units each, when there is at
    least one full wave, the tail is not empty, the workspace holds the tail
    (``tail_items_ws`` items) and the split tail fits one wave of SMs.
    Otherwise ``total_tiles``: no unit is split.  Launch-time arithmetic
    (the launcher passes the result as a runtime argument), so the same
    compiled kernel follows the device it runs on."""
    if num_sm <= 0 or tail_split <= 1:
        return total_tiles
    waves_full, tail_items = divmod(total_tiles, num_sm)
    if (
        waves_full >= 1
        and 0 < tail_items <= tail_items_ws
        and tail_items * tail_split <= num_sm
    ):
        return waves_full * num_sm
    return total_tiles


def flex_split_workspace_elems(
    items: int, splits: int, head_dim_v: int, banks: int = 1
) -> tuple[int, int, int]:
    """Element counts of the three regions of the split-KV workspace: the
    fp32 partial output tiles, the fp32 row-statistics tiles and the int32
    arrival counters (in that order, all 4-byte elements); ``banks`` counter
    banks (the tail split keeps its generation words in a second one)."""
    return (
        items * splits * 2 * 128 * head_dim_v,
        items * splits * 2 * 128 * 2,
        items * 2 * 4 * banks,
    )


def flex_split_workspace(
    like: torch.Tensor, *, items: int, splits: int, head_dim_v: int, banks: int = 1
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
    zero before its launch ends; the tail split's generation words, a
    second bank, only ever count up and are compared for a change).
    """
    partial_elems, stat_elems, counter_elems = flex_split_workspace_elems(
        items, splits, head_dim_v, banks
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
    key = (like.device, stream, items, splits, head_dim_v, banks)
    cached = _FLEX_SPLIT_WORKSPACE_CACHE.get(key)
    if cached is None:
        cached = build()
        _FLEX_SPLIT_WORKSPACE_CACHE[key] = cached
    return cached


def flex_mask_words(num_kv: int) -> int:
    """32-bit words of a KV tile bitmask over ``num_kv`` tiles."""
    return -(-num_kv // 32)


# Per-item header of the words table when the device walks the words with a
# cursor (two words or more): ``[union popcount, highest non-empty union
# word, lowest non-empty union word, stage presence flags]``. A one-word
# mask keeps the header-less rows (the words are registers there).
FLEX_WORDS_HEADER = 4

# The host flags of the list walk's tables (``flex_item_flags``), a runtime
# kernel scalar: bit 0, every list is full (no partial tile, every KV tile in
# every row's full list: torch's noop / full BlockMask); bit 1, the item
# order is the identity.
FLEX_FLAG_ALL_FULL = 1
FLEX_FLAG_IDENTITY_ORDER = 2


def flex_words_header(num_kv: int) -> int:
    """Length of the per-item header of ``flex_item_words`` rows (0 or
    ``FLEX_WORDS_HEADER``); the device body applies the same rule."""
    return FLEX_WORDS_HEADER if flex_mask_words(num_kv) >= 2 else 0


# Transient budget of the row-chunked table builds (the int64 scatter
# targets of ``_dense_tiles``): a long sequence with many heads would
# otherwise materialise gigabytes at once.
_FLEX_BUILD_CHUNK_BYTES = 64 << 20


def _dense_tiles(count: torch.Tensor, index: torch.Tensor, num_kv: int) -> torch.Tensor:
    """``(Bm, Hm, R, num_kv)`` int8 membership of one BlockMask list."""
    length = index.shape[-1]
    flat_index = index.reshape(-1, length)
    flat_count = count.reshape(-1)
    dense = torch.zeros(
        (flat_index.shape[0], num_kv + 1), dtype=torch.int8, device=index.device
    )
    rows_per_chunk = max(1, _FLEX_BUILD_CHUNK_BYTES // (8 * max(length, 1)))
    for start in range(0, flat_index.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, flat_index.shape[0])
        chunk = flat_index[start:stop]
        valid = (
            torch.arange(length, device=index.device, dtype=torch.int64)
            < flat_count[start:stop].to(torch.int64)[:, None]
        )
        # Out-of-range entries past the count (and any stray id) land in a
        # scratch column that is dropped.
        target = torch.where(valid, chunk.to(torch.int64).clamp_(0, num_kv), num_kv)
        dense[start:stop].scatter_(-1, target, 1)
    return dense[:, :num_kv].reshape(*index.shape[:-1], num_kv)


def _pack_tile_words(dense: torch.Tensor, num_kv: int) -> torch.Tensor:
    """``(..., words)`` int32 bitmask words of a ``(..., num_kv)`` 0/1
    membership: tile ``t`` is bit ``t & 31`` of word ``t >> 5`` (the device
    walk's ``_flash_flex_mask_build`` layout). Packed a byte at a time (one
    transient byte per tile) and viewed as little-endian int32 words."""
    words = flex_mask_words(num_kv)
    padded = torch.nn.functional.pad(dense.to(torch.uint8), (0, words * 32 - num_kv))
    # The bit weights are built on the device (``arange`` is a kernel): a
    # host-side constant would be a pageable copy with a stream sync, which
    # is not permitted while a CUDA graph is being captured (the tables are
    # rebuilt on every call under capture).
    shifts = torch.arange(8, device=dense.device, dtype=torch.uint8)
    bits = torch.ones((), device=dense.device, dtype=torch.uint8) << shifts
    packed = (padded.reshape(*dense.shape[:-1], words * 4, 8) * bits).sum(
        dim=-1, dtype=torch.uint8
    )
    return packed.contiguous().view(torch.int32)


def _words_header(
    words: torch.Tensor, count: torch.Tensor, stage_lists: int
) -> torch.Tensor:
    """``(..., FLEX_WORDS_HEADER)`` int32 header of ``(..., lists, W)`` words:
    the union popcount (``count``), the highest and lowest non-empty union
    word (``-1`` / ``0`` for an empty item) and a bit per stage that has
    any tile (``stage_lists`` lists per stage)."""
    union = words[..., 0, :]
    for index in range(1, words.shape[-2]):
        union = union | words[..., index, :]
    nonzero = union != 0
    any_word = nonzero.any(dim=-1)
    num_words = nonzero.shape[-1]
    lowest = nonzero.to(torch.int32).argmax(dim=-1).to(torch.int32)
    highest = (
        num_words - 1 - nonzero.flip(-1).to(torch.int32).argmax(dim=-1).to(torch.int32)
    )
    stages = words.shape[-2] // stage_lists
    flags = torch.zeros(count.shape, dtype=torch.int32, device=count.device)
    for stage in range(stages):
        lists = words[..., stage * stage_lists : (stage + 1) * stage_lists, :]
        present = (lists != 0).any(dim=-1).any(dim=-1)
        flags = flags | (present.to(torch.int32) << stage)
    return torch.stack(
        [
            count.to(torch.int32),
            torch.where(any_word, highest, torch.full_like(highest, -1)),
            torch.where(any_word, lowest, torch.zeros_like(lowest)),
            flags,
        ],
        dim=-1,
    ).to(torch.int32)


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
    group: int = 0,
    flag_bits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The item order, the per-item bitmask words and the flag bits (see
    ``flex_item_table``, ``flex_item_words`` and ``flex_item_flags``), built
    together from one pass over the tables.
    ``group`` > 0 orders the items longest-first within groups of that many
    consecutive batch-heads (the launch group whose K/V stay in L2), the
    groups in order; 0 is one group over every batch-head.  ``flag_bits``
    adds the two-element device tensor the host flags are read from (no
    sync here: ``flex_item_flags`` reads it back lazily); False leaves it
    None (the table path).  Every op below allows a CUDA-graph capture."""
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
        # The rows the walk reads (the flags are theirs).
        real_part, real_full = part, full
        pair_union = torch.maximum(part, full)
        # One row per pair: [partial words, full words].
        per_pair = torch.stack([part, full], dim=3)
    else:
        # Rows ``2p`` and ``2p + 1`` form pair ``p``; a ragged query length
        # leaves the last pair with a single row (its second stage: no tiles).
        assert num_pairs * 2 >= rows >= num_pairs * 2 - 1
        real_part, real_full = part, full
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
    lists = words.shape[-2]
    rows = words.reshape(*words.shape[:-2], lists * words.shape[-1])
    if flex_words_header(num_kv):
        # The cursor walk's per-item header ahead of the words (two lists,
        # partial and full, per stage).
        header = _words_header(words, count, 2)
        rows = torch.cat([header, rows], dim=-1)
    # Broadcast tables (``Bm`` / ``Hm`` of 1, or any divisor) index with
    # ``b % Bm`` / ``h % Hm`` like the device body.
    b_index = torch.arange(batch, device=count.device) % table_batch
    h_index = torch.arange(heads, device=count.device) % table_heads
    count = count[b_index][:, h_index].reshape(-1)
    # Flat: the device indexes it linearly (``item * stride + word``); a 2-D
    # CuTe tensor would resolve a linear index through its column-major
    # layout instead.
    words = rows[b_index][:, h_index].reshape(-1)
    # Longest-first, stable: equal counts keep the ``(bh, pair)`` order.
    order = torch.sort(count, descending=True, stable=True).indices
    if group > 0:
        # Then by launch group of ``group`` batch-heads (stable: the
        # longest-first order holds within each group).
        groups = torch.div(order, num_pairs * group, rounding_mode="floor")
        order = order[torch.sort(groups, stable=True).indices]
    order = order.to(torch.int32)
    bits = _flex_flag_bits(real_part, real_full, order) if flag_bits else None
    return order, words.contiguous(), bits


def _flex_flag_bits(
    part: torch.Tensor, full: torch.Tensor, order: torch.Tensor
) -> torch.Tensor:
    """The two device booleans behind the host flags: no row has a partial
    tile and every row's full list holds every KV tile (``part`` / ``full``
    the dense membership of the rows the walk reads), and ``order`` is the
    identity.  Built with the tables (two tiny reductions, no sync)."""
    all_full = torch.logical_and(torch.all(part == 0), torch.all(full != 0))
    identity = torch.all(
        order == torch.arange(order.numel(), device=order.device, dtype=order.dtype)
    )
    return torch.stack([all_full, identity])


def _flex_flags_from_bits(bits: torch.Tensor) -> int:
    """``FLEX_FLAG_ALL_FULL`` / ``FLEX_FLAG_IDENTITY_ORDER`` from the flag
    bits: the one device-to-host read of the host flags."""
    all_full, identity = bits.tolist()
    return (FLEX_FLAG_ALL_FULL if all_full else 0) | (
        FLEX_FLAG_IDENTITY_ORDER if identity else 0
    )


class _FlexItemTables:
    """One cache entry of the list walk's host tables: the item order, the
    words and the flag bits, whose host read (``flags``) happens at most
    once, when ``flex_item_flags`` first asks for it (the ws body, the
    cooperative pair and the cursor walk bind no flags and never sync)."""

    __slots__ = ("flag_bits", "flags", "order", "words")

    def __init__(
        self, order: torch.Tensor, words: torch.Tensor, flag_bits: torch.Tensor | None
    ) -> None:
        self.order = order
        self.words = words
        self.flag_bits = flag_bits
        self.flags: int | None = None


def _flex_tables_uncached(tables: tuple[torch.Tensor, ...]) -> bool:
    """Whether the tables are rebuilt on every call: inference tensors
    (no version counter) and a CUDA-graph capture (the replay must recompute
    them from the tables it reads)."""
    return any(table.is_inference() for table in tables) or (
        tables[0].is_cuda and torch.cuda.is_current_stream_capturing()
    )


def _flex_item_tables(
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
    group: int = 0,
) -> _FlexItemTables:
    """``_build_flex_item_tables`` cached per ``part_cnt`` tensor behind the
    four tables' pointers and version counters (rebuilt on every call under
    CUDA-graph capture and for inference tensors, without the flag bits)."""
    tables = (part_cnt, part_idx, full_cnt, full_idx)
    kwargs = {
        "q_block": q_block,
        "batch": batch,
        "heads": heads,
        "num_pairs": num_pairs,
        "num_kv": num_kv,
        "group": group,
    }
    if _flex_tables_uncached(tables):
        return _FlexItemTables(*_build_flex_item_tables(*tables, **kwargs))
    key = (
        tuple(sorted(kwargs.items())),
        tuple((table.data_ptr(), table._version) for table in tables),
    )
    # One entry per geometry under the tables' version (the item order's
    # launch group differs from the words table's call): a single slot
    # thrashed between the two calls of one launch and rebuilt both tables
    # every call.
    cached: tuple[object, dict[object, _FlexItemTables]] | None = (
        _FLEX_ITEM_TABLE_CACHE.get(part_cnt)
    )
    if cached is None or cached[0] != key[1]:
        cached = (key[1], {})
        _FLEX_ITEM_TABLE_CACHE[part_cnt] = cached
    built = cached[1].get(key[0])
    if built is None:
        built = _FlexItemTables(
            *_build_flex_item_tables(*tables, flag_bits=True, **kwargs)
        )
        cached[1][key[0]] = built
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

    A flat ``int32`` table of ``batch * heads * num_pairs`` rows of
    ``flex_words_header(num_kv) + lists * words`` entries (row ``bh *
    num_pairs + pair``; the header, present from two words on, is the
    cursor walk's ``[union popcount, highest word, lowest word, stage
    flags]``); ``words = ceil(num_kv / 32)``; the lists are
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
    ).words


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
    group: int = 0,
) -> torch.Tensor:
    """Work-item order of the persistent flex list walk (and of the
    single-tile body's cooperative pair: ``group`` batch-heads per launch
    group, longest-first inside each).

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
        group=group,
    ).order


# The backward's derived KV-side lists, cached per ``part_cnt`` tensor like the
# forward's item tables.
_FLEX_BWD_KV_TABLE_CACHE: WeakIdKeyDictionary = WeakIdKeyDictionary()


def _build_flex_bwd_kv_tables(
    part_cnt: torch.Tensor,
    part_idx: torch.Tensor,
    full_cnt: torch.Tensor,
    full_idx: torch.Tensor,
    *,
    q_blocks: int,
    kv_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The transposed lists of the backward BlockMask tables: from the Q-side
    ``(Bm, Hm, >= kv_blocks)`` counts / ``(Bm, Hm, >= kv_blocks, L)`` Q block
    lists (partial, full; the matcher admits tables with more KV columns than
    N needs, only the first ``kv_blocks`` are KV blocks of the sequence) to
    per-Q-block KV block lists ``(Bm, Hm, q_blocks)`` / ``(Bm, Hm, q_blocks,
    kv_blocks)`` in ascending KV order. Every op has a static shape
    (``argsort`` over the dense membership, no ``nonzero``), so the build is
    legal inside a CUDA-graph capture."""
    tables = []
    for count, index in ((part_cnt, part_idx), (full_cnt, full_idx)):
        count = count[..., :kv_blocks]
        index = index[..., :kv_blocks, :]
        # (Bm, Hm, kv_blocks, q_blocks) membership -> (Bm, Hm, q_blocks, kv_blocks)
        dense = _dense_tiles(count, index, q_blocks).transpose(-1, -2).contiguous()
        kv_count = dense.sum(dim=-1, dtype=torch.int32)
        # Members first, stable: ascending KV block order among the members.
        order = torch.argsort(dense, dim=-1, descending=True, stable=True)
        tables.extend((kv_count.contiguous(), order.to(torch.int32).contiguous()))
    return (tables[0], tables[1], tables[2], tables[3])


def flex_bwd_kv_tables(
    part_cnt: torch.Tensor,
    part_idx: torch.Tensor,
    full_cnt: torch.Tensor,
    full_idx: torch.Tensor,
    *,
    q_blocks: int,
    kv_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(kv_part_cnt, kv_part_idx, kv_full_cnt, kv_full_idx)``: the KV block
    lists per Q block the fused backward's dQ clusters walk, derived on the
    device from the Q-side tables the kernel receives (the BlockMask's
    ``q_num_blocks`` / ``q_indices`` and full variants). Cached the way the
    forward's item tables are (``_FLEX_ITEM_TABLE_CACHE``): a weak entry per
    ``part_cnt`` tensor identity whose key is every table's ``data_ptr`` and
    ``_version`` plus the block counts, so a BlockMask rebuilt in place or a
    new mask reusing the storage misses; rebuilt on every call under
    CUDA-graph capture (the lists must live in the graph's pool) and for
    inference tensors (no version counter)."""
    tables = (part_cnt, part_idx, full_cnt, full_idx)
    kwargs = {"q_blocks": q_blocks, "kv_blocks": kv_blocks}
    if any(table.is_inference() for table in tables) or (
        part_cnt.is_cuda and torch.cuda.is_current_stream_capturing()
    ):
        return _build_flex_bwd_kv_tables(*tables, **kwargs)
    key = (
        tuple(sorted(kwargs.items())),
        tuple((table.data_ptr(), table._version) for table in tables),
    )
    cached = _FLEX_BWD_KV_TABLE_CACHE.get(part_cnt)
    if cached is not None and cached[0] == key:
        return cached[1]
    built = _build_flex_bwd_kv_tables(*tables, **kwargs)
    _FLEX_BWD_KV_TABLE_CACHE[part_cnt] = (key, built)
    return built


def flex_item_flags(
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
    group: int = 0,
) -> int:
    """Host flags of the flex list walk's tables, a runtime kernel scalar
    (a plain ``int`` argument: no recompile, the compiled kernel branches
    on it).

    ``FLEX_FLAG_ALL_FULL`` (bit 0): no row has a partial tile and every row's
    full list holds every KV tile (torch's noop / full BlockMask).  The
    device then takes the words of ``flex_item_words`` as constants without
    loading them, so every step runs the full path, exactly as the table
    would have made it (a full tile never runs the mask hook).  A ragged
    query length's dead second stage (no row) is excluded on both sides.

    ``FLEX_FLAG_IDENTITY_ORDER`` (bit 1): the item order of
    ``flex_item_table`` (with this ``group``) is the identity (every pair has
    the same tile count, e.g. every non-causal full walk), so the device
    decodes the pair item from its work-item id without the table load.

    Built with the tables and cached like them; the flags are read back to
    the host once per cached entry, here (one stream sync, only for the
    renders that bind them).  0 under CUDA-graph capture and for inference
    tensors, without touching the tables (they are rebuilt on every call
    there, by the words and order arguments): the device loads the tables.
    """
    tables = (part_cnt, part_idx, full_cnt, full_idx)
    if _flex_tables_uncached(tables):
        return 0
    entry = _flex_item_tables(
        *tables,
        q_block=q_block,
        batch=batch,
        heads=heads,
        num_pairs=num_pairs,
        num_kv=num_kv,
        group=group,
    )
    if entry.flags is None:
        assert entry.flag_bits is not None
        entry.flags = _flex_flags_from_bits(entry.flag_bits)
    return entry.flags
