"""Collective arithmetic for a proved conditional online-attention page.

This helper does not own a root or load an operand. The caller supplies the
original visibility, score factor, source iteration predicate and complete
score tile, after proving the dedicated region and its memory lifetimes.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from .device.ptx import ptx_op

_center_fma = ptx_op(
    "_center_fma",
    "fma.rn.f32 $0, $1, $2, $3;",
    (cutlass.Float32, cutlass.Float32, cutlass.Float32),
    (cutlass.Float32,),
    side_effects=False,
    memory=False,
    doc="The exact explicit scalar centering FMA from the source recurrence.",
)


@cute.jit
def conditional_probability(
    scores: cute.Tensor,
    visible: cute.Tensor,
    probability: cute.Tensor,
    maximum: cute.Tensor,
    alpha: cute.Tensor,
    scratch: cute.Tensor,
    partials: cute.Tensor,
    thread: cutlass.Int32,
    rescale_previous: cutlass.Boolean,
    score_factor: cutlass.Float32,
    threshold: cutlass.Float32,
    query_columns: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
) -> cutlass.Boolean:
    """128 key owners produce P and update their FP32 per-key L carries.

    Every caller thread owns one key, and every key evaluates all query rows.
    Shared maximum/alpha have query_columns entries; scratch is [4,columns].
    P is the existing MN_INTER layout: eight queries per key per 1024-element
    group. All arrays are distinct and retained through the final CTA join.
    The caller retires prior P readers before entry. Scores/visibility/partials
    are lane-local registers. The returned uniform flag says O must rescale;
    this helper does not read or modify O.
    """
    assert cutlass.const_expr(0 < query_columns <= 64 and query_columns % 8 == 0)
    warp = thread // 32
    lane = thread % 32
    values = cute.make_rmem_tensor((query_columns,), cutlass.Float32)
    changed = cutlass.Boolean(False)
    for query in cutlass.range_constexpr(query_columns):
        additive = cutlass.Float32(float("-inf"))
        if visible[query]:
            additive = -cutlass.Float32(maximum[query])
        # Masking remains inside the FMA additive operand. A masked NaN score
        # must not become a finite/masked probability through a post-FMA mask.
        centered = cutlass.Float32(_center_fma(scores[query], score_factor, additive))
        values[query] = cute.math.exp2(centered, fastmath=fast_math)
        changed = changed | (centered > threshold)
    changed = prims.barrier_cta_red(changed, barrier_id, "or", thread_count=128)
    if changed:
        for query in cutlass.range_constexpr(query_columns):
            # This multiplication is a separate source node from centering.
            candidate = cutlass.Float32(scores[query]) * score_factor
            if not visible[query]:
                candidate = cutlass.Float32(float("-inf"))
            if candidate != candidate:
                candidate = cutlass.Float32(float("-inf"))
            reduced = cute.arch.warp_redux_sync(candidate, "fmax", nan=True)
            if lane == 0:
                scratch[warp, query] = reduced
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
        if thread < query_columns:
            reduced = cutlass.Float32(scratch[0, thread])
            for group in cutlass.range_constexpr(1, 4):
                reduced = cute.arch.fmax(
                    reduced, cutlass.Float32(scratch[group, thread]), nan=True
                )
            previous = cutlass.Float32(maximum[thread])
            current = cute.arch.fmax(previous, reduced, nan=False)
            factor = cutlass.Float32(1)
            if current != float("-inf"):
                factor = cute.math.exp2(previous - current, fastmath=fast_math)
            alpha[thread] = factor
            maximum[thread] = current
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
        for query in cutlass.range_constexpr(query_columns):
            additive = cutlass.Float32(float("-inf"))
            if visible[query]:
                additive = -cutlass.Float32(maximum[query])
            centered = cutlass.Float32(
                _center_fma(scores[query], score_factor, additive)
            )
            values[query] = cute.math.exp2(centered, fastmath=fast_math)
            if rescale_previous:
                partials[query] = partials[query] * cutlass.Float32(alpha[query])
    for query in cutlass.range_constexpr(query_columns):
        value = cutlass.Float32(values[query])
        partials[query] = partials[query] + value
        offset = (query // 8) * 1024 + thread * 8 + query % 8
        probability[offset] = cutlass.BFloat16(value)
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
    return changed & rescale_previous


@cute.jit
def conditional_denominator(
    partials: cute.Tensor,
    scratch: cute.Tensor,
    denominator: cute.Tensor,
    thread: cutlass.Int32,
    query_columns: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
) -> None:
    """Retain the source's 16/8/4/2/1 tree then left-fold its four groups."""
    warp = thread // 32
    lane = thread % 32
    for query in cutlass.range_constexpr(query_columns):
        value = cutlass.Float32(partials[query])
        for shift in cutlass.range_constexpr(5):
            value = value + cute.arch.shuffle_sync_bfly(value, 16 >> shift)
        if lane == 0:
            scratch[warp, query] = value
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
    if thread < query_columns:
        total = cutlass.Float32(scratch[0, thread])
        for group in cutlass.range_constexpr(1, 4):
            total = total + cutlass.Float32(scratch[group, thread])
        denominator[thread] = total
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
