"""First exact online-softmax block for a keys-first paged attention schedule."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims


@cute.jit
def output_normalization_factor(
    safe_denominator: cutlass.Float32,
    reciprocal: cutlass.Constexpr,
    output_scale_log2: cutlass.Constexpr = 0,
) -> cutlass.Float32:
    """Publish the source-selected divisor or one rounded row reciprocal."""
    assert cutlass.const_expr(reciprocal or output_scale_log2 == 0)
    if cutlass.const_expr(reciprocal):
        factor = cutlass.Float32(1) / safe_denominator
        if cutlass.const_expr(output_scale_log2 != 0):
            factor = factor * (2.0**output_scale_log2)
        return factor
    return safe_denominator


@cute.jit
def normalize_output(
    value: cutlass.Float32, factor: cutlass.Float32, reciprocal: cutlass.Constexpr
) -> cutlass.Float32:
    if cutlass.const_expr(reciprocal):
        return value * factor
    return value / factor


@cute.jit
def logical_attention_mask(
    logical_key_base: cutlass.Int64,
    key_offset: cutlass.Int32,
    length: cutlass.Int32,
    query: cutlass.Constexpr,
    query_length: cutlass.Constexpr,
    group_size: cutlass.Constexpr,
    window: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
) -> cutlass.Boolean:
    """Retain scalar INT32 subtraction before index-width promotion.

    The semantic plan proves this exact operation order. Converting length
    before subtraction changes wrapped tensor arithmetic at INT32 boundaries.
    The grouped row is nonnegative and bounded by the physical query tile.
    """
    key = coordinate_dtype(logical_key_base) + coordinate_dtype(key_offset)
    base = cutlass.Int32(length - query_length)
    position = coordinate_dtype(base) + coordinate_dtype(query // group_size)
    visible = key <= position
    if cutlass.const_expr(window is not None):
        visible = visible & (key >= position - window)
    return visible


@cute.jit
def _ordered_tile_ranges(
    logical_key_base: cutlass.Int64,
    key_count: cutlass.Constexpr,
    length: cutlass.Int32,
    query_columns: cutlass.Constexpr,
    query_length: cutlass.Constexpr,
    group_size: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
) -> tuple[
    cutlass.Integer, cutlass.Integer, cutlass.Integer, cutlass.Integer, cutlass.Boolean
]:
    """First and last key of a tile, the positions of its first and last query
    rows, and whether both ranges are ordered.

    Ordered ranges exclude signed wrap, so their ends bound every key and
    every row of the tile. Length subtraction retains INT32 semantics before
    coordinate promotion, just as in logical_attention_mask. The interior
    test and the key clip both start here, so they cannot disagree about
    which tiles are ordered or where their keys and rows lie.
    """
    key_begin = coordinate_dtype(logical_key_base)
    key_end = key_begin + coordinate_dtype(key_count - 1)
    query_begin = coordinate_dtype(cutlass.Int32(length - query_length))
    query_end = query_begin + coordinate_dtype((query_columns - 1) // group_size)
    ordered = (key_end >= key_begin) & (query_end >= query_begin)
    return key_begin, key_end, query_begin, query_end, ordered


@cute.jit
def _ordered_window_bounds(
    query_begin: cutlass.Integer, query_end: cutlass.Integer, window: cutlass.Constexpr
) -> tuple[cutlass.Integer, cutlass.Integer, cutlass.Boolean]:
    """Lower window bounds of a tile's first and last query rows, and whether
    they are ordered.

    A window can wrap for only the earlier query rows. Their lower bound must
    remain ordered before the first row's is used as the loosest and the last
    row's as the strictest.
    """
    lower_begin = query_begin - window
    lower_end = query_end - window
    return lower_begin, lower_end, lower_end >= lower_begin


@cute.jit
def logical_attention_tile_is_interior(
    logical_key_base: cutlass.Int64,
    key_count: cutlass.Constexpr,
    length: cutlass.Int32,
    query_columns: cutlass.Constexpr,
    query_length: cutlass.Constexpr,
    group_size: cutlass.Constexpr,
    window: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
) -> cutlass.Boolean:
    """Prove that every key is visible to every row of a nonempty tile.

    Ordered key and query ranges exclude signed wrap before using their extrema
    (_ordered_tile_ranges, _ordered_window_bounds). Other tiles keep the
    per-element mask.
    """
    assert cutlass.const_expr(key_count > 0 and query_columns > 0 and group_size > 0)
    key_begin, key_end, query_begin, query_end, ordered = _ordered_tile_ranges(
        logical_key_base,
        key_count,
        length,
        query_columns,
        query_length,
        group_size,
        coordinate_dtype,
    )
    interior = ordered & (key_end <= query_begin)
    if cutlass.const_expr(window is not None):
        _lower_begin, lower_end, lower_ordered = _ordered_window_bounds(
            query_begin, query_end, window
        )
        interior = interior & lower_ordered & (key_begin >= lower_end)
    return interior


@cute.jit
def logical_attention_key_shift(
    logical_key_base: cutlass.Int64,
    key_count: cutlass.Constexpr,
    length: cutlass.Int32,
    query_columns: cutlass.Constexpr,
    query_length: cutlass.Constexpr,
    group_size: cutlass.Constexpr,
    window: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    key_clip: cutlass.Constexpr,
) -> cutlass.Int32:
    """Offset of a tile's key box that leaves out keys no row can see.

    Row r of a shifted tile holds key ``shift + r`` of its page. Rows whose
    key lies outside [0, key_count) are outside the descriptor's key extent:
    the TMA unit fills them with zeros instead of fetching them from their
    own addresses, and shifted_attention_key masks them. ``leading`` clips
    drop keys before every row's window with a positive shift, ``trailing``
    clips drop keys after every row's causal bound with a negative shift,
    and ``both`` keeps the larger saving of a tile that has the two. ``off``
    is a whole tile: shift zero, without an operation.

    The loader issues a tile's boxes at this shift and the softmax helpers
    map their lanes with it (shifted_attention_key), all from this function
    and the same arguments, so the two sides cannot disagree about a key.

    The ranges and ordering checks are those of
    logical_attention_tile_is_interior, from the same two helpers: with
    ordered keys, positions and window bounds, a key below the first row's
    lower bound or above the last row's position fails logical_attention_mask
    for every row in the mask's own wrapping arithmetic. Any other tile keeps
    shift zero, as does every interior tile. Differences are taken only between
    values inside one ordered tile, so they fit the key count.

    Shifts are multiples of four keys, which keeps the packed scale box (one
    INT32 word per key, key innermost) 16-byte aligned, and at most
    key_count - 4, so some row always remains inside the descriptor.
    """
    assert cutlass.const_expr(key_clip in ("off", "leading", "trailing", "both"))
    if cutlass.const_expr(key_clip == "off"):
        return cutlass.Int32(0)
    assert cutlass.const_expr(key_clip == "trailing" or window is not None)
    assert cutlass.const_expr(key_count >= 8 and key_count % 4 == 0)
    assert cutlass.const_expr(query_columns > 0 and group_size > 0)
    key_begin, key_end, query_begin, query_end, ordered = _ordered_tile_ranges(
        logical_key_base,
        key_count,
        length,
        query_columns,
        query_length,
        group_size,
        coordinate_dtype,
    )
    trailing = cutlass.Int32(0)
    if cutlass.const_expr(key_clip != "leading"):
        if query_end < key_end:
            trailing = cutlass.Int32(key_count)
            if query_end >= key_begin:
                trailing = cutlass.Int32(key_end - query_end)
        trailing = cutlass.Int32(cutlass.min(trailing & -4, key_count - 4))
    leading = cutlass.Int32(0)
    if cutlass.const_expr(key_clip != "trailing"):
        lower_begin, _lower_end, lower_ordered = _ordered_window_bounds(
            query_begin, query_end, window
        )
        ordered = ordered & lower_ordered
        if lower_begin > key_begin:
            leading = cutlass.Int32(key_count)
            if lower_begin <= key_end:
                leading = cutlass.Int32(lower_begin - key_begin)
        leading = cutlass.Int32(cutlass.min(leading & -4, key_count - 4))
    shift = cutlass.Int32(0)
    if ordered:
        shift = leading
        if trailing > leading:
            shift = cutlass.Int32(-trailing)
    return shift


@cute.jit
def shifted_attention_key(
    thread: cutlass.Int32,
    logical_key_base: cutlass.Int64,
    key_count: cutlass.Constexpr,
    length: cutlass.Int32,
    query_columns: cutlass.Constexpr,
    query_length: cutlass.Constexpr,
    group_size: cutlass.Constexpr,
    window: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    key_clip: cutlass.Constexpr,
) -> tuple[cutlass.Int32, cutlass.Boolean]:
    """The page key that softmax lane ``thread`` holds in its row of a tile,
    and whether that row is a key of the page rather than zero fill.

    The loader issued the tile at logical_attention_key_shift of the same
    arguments, so the lane's row holds key ``thread + shift``; a key outside
    [0, key_count) is zero fill, which no row may see. A whole tile
    (``off``) keeps the lane's own key and every row, without an operation.
    """
    if cutlass.const_expr(key_clip == "off"):
        return thread, cutlass.Boolean(True)
    key_offset = thread + logical_attention_key_shift(
        logical_key_base,
        key_count,
        length,
        query_columns,
        query_length,
        group_size,
        window,
        coordinate_dtype,
        key_clip,
    )
    return key_offset, cutlass.Uint32(key_offset) < cutlass.Uint32(key_count)


@cute.jit
def online_softmax_128x64(
    score_base: cutlass.Int32,
    probability: cute.Tensor,
    state: cute.Tensor,
    scratch: cute.Tensor,
    thread: cutlass.Int32,
    logical_key_base: cutlass.Int64,
    length: cutlass.Int32,
    query_length: cutlass.Constexpr,
    group_size: cutlass.Constexpr,
    window: cutlass.Constexpr,
    score_scale: cutlass.Float32,
    barrier_id: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
) -> None:
    """Four full warps consume S[key,query] and produce P[query,key].

    State rows hold the running maximum, FP32 denominator, accumulator
    rescale, and safe denominator. The caller initializes the first two.
    Probability has the physical K-major SW128 layout expected by BF16
    tensor-core B descriptors. Scratch is [4,64] FP32 and is reusable after
    the final named barrier. The surrounding owner handles S/P lifetime
    and the producer/consumer barriers for other participating warps.
    """
    warp = thread // 32
    lane = thread % 32
    address = (warp * 32 << 16) | score_base
    scores = prims.tcgen05_ld(
        "32x32b", cutlass.inttoptr(address, 6, cutlass.Float32), num=64
    )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    values = cute.make_rmem_tensor((64,), cutlass.Float32)
    for query in cutlass.range_constexpr(64):
        visible = logical_attention_mask(
            logical_key_base,
            thread,
            length,
            query,
            query_length,
            group_size,
            window,
            coordinate_dtype,
        )
        score = scores[query] * score_scale
        if not visible:
            score = cutlass.Float32(float("-inf"))
        values[query] = score
        maximum = cute.arch.warp_redux_sync(score, "fmax", nan=True)
        if lane == 0:
            scratch[warp, query] = maximum
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
    if thread < 64:
        maximum = cutlass.Float32(state[0, thread])
        for w in cutlass.range_constexpr(4):
            maximum = cute.math.max(
                maximum, cutlass.Float32(scratch[w, thread]), propagate_nan=True
            )
        safe_maximum = maximum
        if maximum == float("-inf"):
            safe_maximum = cutlass.Float32(0.0)
        state[2, thread] = cute.math.exp(
            cutlass.Float32(state[0, thread]) - safe_maximum, fastmath=fast_math
        )
        state[0, thread] = maximum
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
    for query in cutlass.range_constexpr(64):
        safe_maximum = cutlass.Float32(state[0, query])
        if safe_maximum == float("-inf"):
            safe_maximum = cutlass.Float32(0.0)
        value = cute.math.exp(values[query] - safe_maximum, fastmath=fast_math)
        offset = (
            query * 64 + ((thread % 64) ^ ((query % 8) * 8)) + (thread // 64) * 4096
        )
        probability[offset] = cutlass.BFloat16(value)
        total = value
        for shift in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync_bfly(total, 1 << shift)
        if lane == 0:
            scratch[warp, query] = total
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
    if thread < 64:
        total = cutlass.Float32(scratch[0, thread]) + cutlass.Float32(
            scratch[1, thread]
        )
        total = total + cutlass.Float32(scratch[2, thread])
        total = total + cutlass.Float32(scratch[3, thread])
        denominator = (
            cutlass.Float32(state[1, thread]) * cutlass.Float32(state[2, thread])
            + total
        )
        state[1, thread] = denominator
        safe_denominator = denominator
        if denominator == 0:
            safe_denominator = cutlass.Float32(1.0)
        state[3, thread] = safe_denominator
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)


@cute.jit
def mma_128x64x128(
    a_base: cutlass.Int32,
    d_base: cutlass.Int32,
    b_pointer: cute.Pointer,
    done: cute.Pointer,
    accumulate: cutlass.Constexpr,
) -> None:
    desc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        n_dim=64,
        m_dim=128,
        a_major=0,
        b_major=0,
    )
    bd = prims.Tcgen05SmemDesc.build(
        cutlass.Int32(b_pointer.toint()),
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    acc = cutlass.inttoptr(d_base, 6, cutlass.Float32)
    for step in cutlass.range_constexpr(8):
        if prims.elect_sync():
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                acc,
                cutlass.inttoptr(a_base + step * 8, 6, cutlass.Int32),
                bd.advance_start_address((step % 4) * 32 + (step // 4) * 8192),
                desc,
                accumulate or step != 0,
            )
    if prims.elect_sync():
        prims.tcgen05_commit(done, group=prims.CTAGroup.CTA_1)
