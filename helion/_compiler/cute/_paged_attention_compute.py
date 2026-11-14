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

    Ordered key and query ranges exclude signed wrap before using their extrema.
    Length subtraction retains INT32 semantics before coordinate promotion,
    just as in logical_attention_mask. Other tiles keep the per-element mask.
    """
    assert cutlass.const_expr(key_count > 0 and query_columns > 0 and group_size > 0)
    key_begin = coordinate_dtype(logical_key_base)
    key_end = key_begin + coordinate_dtype(key_count - 1)
    query_begin = coordinate_dtype(cutlass.Int32(length - query_length))
    query_end = query_begin + coordinate_dtype((query_columns - 1) // group_size)
    interior = (
        (key_end >= key_begin) & (query_end >= query_begin) & (key_end <= query_begin)
    )
    if cutlass.const_expr(window is not None):
        lower_begin = query_begin - window
        lower_end = query_end - window
        # A window can wrap for only the earlier query rows. Their lower bound
        # must remain ordered before using the last row as the strictest one.
        interior = interior & (lower_end >= lower_begin) & (key_begin >= lower_end)
    return interior


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
