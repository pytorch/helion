"""Arithmetic for explicitly centered, per-key online attention sources.

Keep this recurrence separate from the ordinary per-key implementation so its
scalar instruction selection and rounding remain unchanged.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from ._flash_row_mma_runtime import pack_bf16x2
from ._paged_attention_compute import logical_attention_mask
from ._paged_attention_compute import logical_attention_tile_is_interior
from ._paged_attention_partial_compute import load_query_panel
from .device.ptx import sts_v4_b32


@cute.jit
def _store_reference_probability_and_denominator(
    values: cute.Tensor,
    probability: cute.Tensor,
    state: cute.Tensor,
    partials: cute.Tensor,
    thread: cutlass.Int32,
    query_columns: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
    rescale_denominator: cutlass.Constexpr,
    base2_scores: cutlass.Constexpr,
) -> None:
    # Each key lane writes eight adjacent queries with one aligned 16-byte
    # store. The complete eight-query groups follow the selected panel width.
    for group in cutlass.range_constexpr(query_columns // 8):
        packed = cute.make_rmem_tensor((4,), cutlass.Uint32)
        for pair in cutlass.range_constexpr(4):
            query = group * 8 + pair * 2
            x0 = values[query]
            x1 = values[query + 1]
            if cutlass.const_expr(rescale_denominator):
                x0, x1 = cute.arch.add_packed_f32x2(
                    (x0, x1),
                    (
                        cutlass.Float32(state[3, query]),
                        cutlass.Float32(state[3, query + 1]),
                    ),
                )
            if cutlass.const_expr(base2_scores):
                p0 = cute.math.exp2(x0, fastmath=fast_math)
                p1 = cute.math.exp2(x1, fastmath=fast_math)
            else:
                p0 = cute.math.exp(x0, fastmath=fast_math)
                p1 = cute.math.exp(x1, fastmath=fast_math)
            old = (
                cutlass.Float32(partials[query]),
                cutlass.Float32(partials[query + 1]),
            )
            if cutlass.const_expr(rescale_denominator):
                alpha = (
                    cutlass.Float32(state[2, query]),
                    cutlass.Float32(state[2, query + 1]),
                )
                if cutlass.const_expr(fast_math):
                    l0, l1 = cute.arch.fma_packed_f32x2(old, alpha, (p0, p1))
                else:
                    scaled = cute.arch.mul_packed_f32x2(old, alpha)
                    l0, l1 = cute.arch.add_packed_f32x2(scaled, (p0, p1))
            else:
                l0, l1 = cute.arch.add_packed_f32x2(old, (p0, p1))
            partials[query] = l0
            partials[query + 1] = l1
            packed[pair] = pack_bf16x2(p0, p1)
        offset = group * 1024 + thread * 8
        address = cutlass.Int32((probability.iterator + offset).toint())
        sts_v4_b32(address, packed[0], packed[1], packed[2], packed[3])


@cute.jit
def _reference_values(
    scores: cute.Tensor,
    state: cute.Tensor,
    values: cute.Tensor,
    thread: cutlass.Int32,
    logical_key_base: cutlass.Int64,
    length: cutlass.Int32,
    query_length: cutlass.Constexpr,
    group_size: cutlass.Constexpr,
    window: cutlass.Constexpr,
    score_scale: cutlass.Float32,
    maximum_invalid: cutlass.Boolean,
    fast_math: cutlass.Constexpr,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    apply_mask: cutlass.Constexpr,
) -> cutlass.Boolean:
    """Center scores and vote without per-element masking on proved interior tiles."""
    local_maxima = cute.make_rmem_tensor((4,), cutlass.Float32)
    for chain in cutlass.range_constexpr(4):
        local_maxima[chain] = cutlass.Float32(float("-inf"))
    for pair in cutlass.range_constexpr(query_columns // 2):
        query = pair * 2
        ref0 = cutlass.Float32(state[1, query])
        ref1 = cutlass.Float32(state[1, query + 1])
        if cutlass.const_expr(fast_math):
            x0, x1 = cute.arch.fma_packed_f32x2(
                (scores[query], scores[query + 1]),
                (score_scale, score_scale),
                (-ref0, -ref1),
            )
        else:
            scaled = cute.arch.mul_packed_f32x2(
                (scores[query], scores[query + 1]), (score_scale, score_scale)
            )
            x0, x1 = cute.arch.add_packed_f32x2(scaled, (-ref0, -ref1))
        relative = (x0, x1)
        for element in cutlass.range_constexpr(2):
            row = query + element
            x = relative[element]
            if cutlass.const_expr(apply_mask):
                visible = logical_attention_mask(
                    logical_key_base,
                    thread,
                    length,
                    row,
                    query_length,
                    group_size,
                    window,
                    coordinate_dtype,
                )
                if not visible:
                    x = cutlass.Float32(float("-inf"))
            values[row] = x
            local_maxima[row % 4] = cute.arch.fmax(local_maxima[row % 4], x, nan=True)
    maximum01 = cute.arch.fmax(local_maxima[0], local_maxima[1], nan=True)
    maximum23 = cute.arch.fmax(local_maxima[2], local_maxima[3], nan=True)
    local_maximum = cute.arch.fmax(maximum01, maximum23, nan=True)
    # A false vote proves reference == old maximum >= 0 and every
    # visible x <= 0. Therefore shift == 0 and alpha == 1 exactly.
    return maximum_invalid | (local_maximum > 0) | (local_maximum != local_maximum)


@cute.jit
def online_softmax_reference(
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
    partials: cute.Tensor,
    maximum_invalid: cutlass.Boolean,
    fast_math: cutlass.Constexpr,
    score_free: cute.Pointer,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr = 40,
    base2_scores: cutlass.Constexpr = False,
) -> tuple[cutlass.Boolean, cutlass.Boolean]:
    """Consume the separately proved reference-score recurrence.

    State row0 is the running maximum; row1 caches its nonnegative finite
    reference, row2 carries alpha, and row3 carries this tile's reference shift.
    The owner overwrites rows1/3 with the final denominator after the loop.
    The returned flags are uniform over the 128 participating compute threads.
    The invalid flag replaces repeated reference guards in every key lane;
    it is refreshed at the existing maximum-publication barrier.
    """
    assert cutlass.const_expr(query_columns % 8 == 0)
    warp = thread // 32
    lane = thread % 32
    address = (warp * 32 << 16) | score_base
    scores = load_query_panel(address, query_columns)
    # All score loads retire before the issuer may reuse their TMEM panel.
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.sync_warp()
    if lane == 0:
        prims.mbarrier_arrive(score_free)
    values = cute.make_rmem_tensor((query_columns,), cutlass.Float32)
    interior = logical_attention_tile_is_interior(
        logical_key_base,
        128,  # One key per participating compute thread, as in the barriers below.
        length,
        query_columns,
        query_length,
        group_size,
        window,
        coordinate_dtype,
    )
    needs_update = cutlass.Boolean(False)
    if interior:
        needs_update = _reference_values(
            scores,
            state,
            values,
            thread,
            logical_key_base,
            length,
            query_length,
            group_size,
            window,
            score_scale,
            maximum_invalid,
            fast_math,
            coordinate_dtype,
            query_columns,
            apply_mask=False,
        )
    else:
        needs_update = _reference_values(
            scores,
            state,
            values,
            thread,
            logical_key_base,
            length,
            query_length,
            group_size,
            window,
            score_scale,
            maximum_invalid,
            fast_math,
            coordinate_dtype,
            query_columns,
            apply_mask=True,
        )
    maximum_changed = prims.barrier_cta_red(
        needs_update, barrier_id, "or", thread_count=128
    )
    if maximum_changed:
        for query in cutlass.range_constexpr(query_columns):
            maximum = cute.arch.warp_redux_sync(values[query], "fmax", nan=True)
            if lane == 0:
                scratch[warp, query] = maximum
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
        next_invalid = cutlass.Boolean(False)
        if thread < query_columns:
            previous = cutlass.Float32(state[0, thread])
            reference = cutlass.Float32(state[1, thread])
            delta = cutlass.Float32(float("-inf"))
            for w in cutlass.range_constexpr(4):
                delta = cute.arch.fmax(
                    delta, cutlass.Float32(scratch[w, thread]), nan=True
                )
            maximum = cute.arch.fmax(previous, delta + reference, nan=True)
            safe_maximum = maximum
            if maximum == float("-inf"):
                safe_maximum = cutlass.Float32(0.0)
            alpha_input = cutlass.Float32(state[0, thread]) - safe_maximum
            if cutlass.const_expr(base2_scores):
                state[2, thread] = cute.math.exp2(alpha_input, fastmath=fast_math)
            else:
                state[2, thread] = cute.math.exp(alpha_input, fastmath=fast_math)
            state[0, thread] = maximum
            # Row 3 is not read by the owner until the final denominator
            # epilogue overwrites it, so it can hold the XREF shift in-loop.
            state[3, thread] = reference - safe_maximum
            next_reference = cutlass.Float32(0)
            valid_reference = (maximum >= 0) & (maximum < float("inf"))
            if valid_reference:
                next_reference = maximum
            state[1, thread] = next_reference
            next_invalid = not valid_reference
        maximum_invalid = prims.barrier_cta_red(
            next_invalid, barrier_id, "or", thread_count=128
        )
    # The uniform vote proves every maximum is finite and unchanged, so
    # alpha=exp(old_max-old_max)=1 and no state writes or alpha reads are needed.
    if maximum_changed:
        _store_reference_probability_and_denominator(
            values,
            probability,
            state,
            partials,
            thread,
            query_columns,
            fast_math,
            rescale_denominator=True,
            base2_scores=base2_scores,
        )
    else:
        _store_reference_probability_and_denominator(
            values,
            probability,
            state,
            partials,
            thread,
            query_columns,
            fast_math,
            rescale_denominator=False,
            base2_scores=base2_scores,
        )
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
    return maximum_changed, maximum_invalid
