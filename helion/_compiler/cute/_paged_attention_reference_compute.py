"""Arithmetic for explicitly centered, per-key online attention sources.

Keep this recurrence separate from the ordinary per-key implementation so its
scalar instruction selection and rounding remain unchanged.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from ._flash_row_mma_runtime import pack_bf16x2
from ._mma_dag_tcgen05_runtime import mbar_wait
from ._mma_dag_tcgen05_runtime import mbar_wait_sleep
from ._paged_attention_compute import logical_attention_mask
from ._paged_attention_compute import logical_attention_tile_is_interior
from ._paged_attention_partial_compute import load_query_panel
from ._paged_attention_partial_compute import store_query_panel
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
    probability_schedule: cutlass.Constexpr = "serial",
    denominator_address: cutlass.Int32 | None = None,
    previous_pv: cute.Pointer | None = None,
    previous_phase: cutlass.Int32 | cutlass.Int64 = cutlass.Int32(0),  # noqa: B008
    have_previous: cutlass.Boolean = cutlass.Boolean(False),  # noqa: B008
    previous_sleep: cutlass.Constexpr = False,
) -> None:
    # Each key lane writes eight adjacent queries with one aligned 16-byte
    # store. The complete eight-query groups follow the selected panel width.
    if cutlass.const_expr(probability_schedule == "early_tmem"):
        loaded_partials = load_query_panel(denominator_address, query_columns)
        if cutlass.const_expr(query_columns == 40):
            partials = loaded_partials
        else:
            # Power-of-two loads return an immutable Vector. The recurrence
            # needs mutable registers while updating this page's denominator.
            partials = cute.make_rmem_tensor((query_columns,), cutlass.Float32)
            for query in cutlass.range_constexpr(query_columns):
                partials[query] = loaded_partials[query]
    # Initialize outside the constexpr branch for CuTe's variable analysis.
    # The serial path never reads this tensor and eliminates the allocation.
    packed_values = cute.make_rmem_tensor((query_columns // 2,), cutlass.Uint32)
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
            if cutlass.const_expr(probability_schedule == "serial"):
                packed[pair] = pack_bf16x2(p0, p1)
            else:
                packed_values[group * 4 + pair] = pack_bf16x2(p0, p1)
        if cutlass.const_expr(probability_schedule == "serial"):
            offset = group * 1024 + thread * 8
            address = cutlass.Int32((probability.iterator + offset).toint())
            sts_v4_b32(address, packed[0], packed[1], packed[2], packed[3])
    if cutlass.const_expr(probability_schedule == "early_tmem"):
        store_query_panel(denominator_address, partials, query_columns)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    if cutlass.const_expr(probability_schedule != "serial"):
        # Previous PV owns shared P and TMEM O. Computing the next P in
        # registers and updating its disjoint denominator may overlap it;
        # every shared P overwrite and subsequent O rescale must wait here.
        if have_previous:
            if cutlass.const_expr(previous_sleep):
                mbar_wait_sleep(previous_pv, previous_phase)
            else:
                mbar_wait(previous_pv, previous_phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        for group in cutlass.range_constexpr(query_columns // 8):
            offset = group * 1024 + thread * 8
            address = cutlass.Int32((probability.iterator + offset).toint())
            sts_v4_b32(
                address,
                packed_values[group * 4],
                packed_values[group * 4 + 1],
                packed_values[group * 4 + 2],
                packed_values[group * 4 + 3],
            )


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
    rowwise_rescale_threshold: cutlass.Constexpr = None,
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
    if cutlass.const_expr(rowwise_rescale_threshold is not None):
        # The source retains each old maximum when all row predicates are false.
        # Its reference then equals old, so shift=0 and alpha=1 even for 0 < x <= T.
        return (
            maximum_invalid
            | (local_maximum > cutlass.Float32(rowwise_rescale_threshold))
            | (local_maximum != local_maximum)
        )
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
    # Disjoint 128-thread groups may supply different warp-uniform IDs.
    barrier_id: cutlass.Int32,
    partials: cute.Tensor,
    maximum_invalid: cutlass.Boolean,
    fast_math: cutlass.Constexpr,
    score_free: cute.Pointer,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr = 40,
    base2_scores: cutlass.Constexpr = False,
    convergent_branches: cutlass.Constexpr = False,
    group_barrier: cutlass.Constexpr = True,
    probability_schedule: cutlass.Constexpr = "serial",
    denominator_address: cutlass.Int32 | None = None,
    previous_pv: cute.Pointer | None = None,
    previous_phase: cutlass.Int32 | cutlass.Int64 = cutlass.Int32(0),  # noqa: B008
    have_previous: cutlass.Boolean = cutlass.Boolean(False),  # noqa: B008
    previous_sleep: cutlass.Constexpr = False,
    rowwise_rescale_threshold: cutlass.Constexpr = None,
    headwide_rescale: cutlass.Constexpr = False,
) -> tuple[cutlass.Boolean, cutlass.Boolean]:
    """Consume the separately proved reference-score recurrence.

    State row0 is the running maximum; row1 caches its nonnegative finite
    reference, row2 carries alpha, and row3 carries this tile's reference shift.
    The owner overwrites rows1/3 with the final denominator after the loop.
    The returned flags are uniform over the 128 participating compute threads.
    The invalid flag replaces repeated reference guards in every key lane;
    it is refreshed at the existing maximum-publication barrier.
    convergent_branches keeps long uniform arms separate in the paired caller.
    When group_barrier is False, the caller must publish every writer warp
    before a consumer reads P. Serial callers wait before the next call;
    early schedules pass previous_pv so this helper waits before rewriting P.
    The next call's maximum vote barrier orders successive probability-ready
    phases, and the maximum-publication barrier orders state/scratch reuse.
    Caller writes to state after return require their own group synchronization.
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
    # The paired arena caller opts into a convergence point in each uniform
    # arm so NVVM preserves the branch instead of predicating both long score
    # passes. A warp-uniform annotation alone did not prevent this if-conversion.
    # Other callers keep their existing control flow.
    needs_update = cutlass.Boolean(False)
    if interior:
        if cutlass.const_expr(convergent_branches):
            cute.arch.sync_warp()
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
            rowwise_rescale_threshold=rowwise_rescale_threshold,
        )
    else:
        if cutlass.const_expr(convergent_branches):
            cute.arch.sync_warp()
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
            rowwise_rescale_threshold=rowwise_rescale_threshold,
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
            if cutlass.const_expr(
                rowwise_rescale_threshold is not None and not headwide_rescale
            ):
                # A true head vote does not imply that this row requested rescaling.
                row_rescale = (
                    (delta > cutlass.Float32(rowwise_rescale_threshold))
                    | (not ((previous >= 0) & (previous < float("inf"))))
                    | (delta != delta)
                )
                if not row_rescale:
                    maximum = previous
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
            probability_schedule=probability_schedule,
            denominator_address=denominator_address,
            previous_pv=previous_pv,
            previous_phase=previous_phase,
            have_previous=have_previous,
            previous_sleep=previous_sleep,
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
            probability_schedule=probability_schedule,
            denominator_address=denominator_address,
            previous_pv=previous_pv,
            previous_phase=previous_phase,
            have_previous=have_previous,
            previous_sleep=previous_sleep,
        )
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    if cutlass.const_expr(group_barrier):
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
    return maximum_changed, maximum_invalid
