"""Compile-visible observer of the actual conditional probability helper.

No attention, MMA, or input-gather schedule is implemented by this observer.
Its outputs retain every mutated carry, probability and collective decision.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from helion._compiler.cute._conditional_attention_compute import conditional_denominator
from helion._compiler.cute._conditional_attention_compute import conditional_probability


@cute.kernel
def observe_conditional_page(
    source_scores: cute.Tensor,
    source_visible: cute.Tensor,
    source_maximum: cute.Tensor,
    source_partials: cute.Tensor,
    output_state: cute.Tensor,
    output_partials: cute.Tensor,
    output_probability: cute.Tensor,
    output_flags: cute.Tensor,
    rescale_previous: cutlass.Boolean,
    score_factor: cutlass.Float32,
    threshold: cutlass.Float32,
    query_columns: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
) -> None:
    thread = cutlass.Int32(cute.arch.thread_idx()[0])
    allocator = cutlass.utils.SmemAllocator()
    maximum = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((query_columns,)), byte_alignment=16
    )
    alpha = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((query_columns,)), byte_alignment=16
    )
    denominator = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((query_columns,)), byte_alignment=16
    )
    scratch = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((4, query_columns), stride=(query_columns, 1)),
        byte_alignment=16,
    )
    probability = allocator.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((128 * query_columns,)),
        byte_alignment=128,
    )
    scores = cute.make_rmem_tensor((query_columns,), cutlass.Float32)
    visible = cute.make_rmem_tensor((query_columns,), cutlass.Boolean)
    partials = cute.make_rmem_tensor((query_columns,), cutlass.Float32)
    for query in cutlass.range_constexpr(query_columns):
        scores[query] = source_scores[thread, query]
        visible[query] = source_visible[thread, query] != 0
        partials[query] = source_partials[thread, query]
    if thread < query_columns:
        maximum[thread] = source_maximum[thread]
        # Alpha is not meaningful on a false vote. Keep its untouched state
        # observable without reading uninitialized shared storage.
        alpha[thread] = cutlass.Float32(7)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    rescale = conditional_probability(
        scores,
        visible,
        probability,
        maximum,
        alpha,
        scratch,
        partials,
        thread,
        rescale_previous,
        score_factor,
        threshold,
        query_columns,
        0,
        fast_math,
    )
    for query in cutlass.range_constexpr(query_columns):
        output_partials[thread, query] = partials[query]
        offset = (query // 8) * 1024 + thread * 8 + query % 8
        output_probability[query, thread] = probability[offset]
    output_flags[thread] = cutlass.Int32(rescale)
    conditional_denominator(partials, scratch, denominator, thread, query_columns, 0)
    if thread < query_columns:
        output_state[thread, 0] = maximum[thread]
        output_state[thread, 1] = alpha[thread]
        output_state[thread, 2] = denominator[thread]


@cute.jit
def launch_conditional_page(
    source_scores: cute.Tensor,
    source_visible: cute.Tensor,
    source_maximum: cute.Tensor,
    source_partials: cute.Tensor,
    output_state: cute.Tensor,
    output_partials: cute.Tensor,
    output_probability: cute.Tensor,
    output_flags: cute.Tensor,
    stream: object,
    rescale_previous: cutlass.Boolean,
    score_factor: cutlass.Float32,
    threshold: cutlass.Float32,
    query_columns: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
) -> None:
    observe_conditional_page(
        source_scores,
        source_visible,
        source_maximum,
        source_partials,
        output_state,
        output_partials,
        output_probability,
        output_flags,
        rescale_previous,
        score_factor,
        threshold,
        query_columns,
        fast_math,
    ).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)
