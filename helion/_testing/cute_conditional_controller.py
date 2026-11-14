"""Offline observer of the composed controller; inputs are already BF16."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from helion._compiler.cute._conditional_attention_page import (
    conditional_attention_begin,
)
from helion._compiler.cute._conditional_attention_page import conditional_attention_end
from helion._compiler.cute._conditional_attention_page import conditional_attention_page


@cute.kernel
def observe_conditional_controller(
    keys: cute.Tensor,
    values: cute.Tensor,
    queries: cute.Tensor,
    old_maximum: cute.Tensor,
    old_partials: cute.Tensor,
    old_output: cute.Tensor,
    source_visible: cute.Tensor,
    new_maximum: cute.Tensor,
    new_partials: cute.Tensor,
    new_output: cute.Tensor,
    new_probability: cute.Tensor,
    rescale_previous: cutlass.Boolean,
    score_factor: cutlass.Float32,
    threshold: cutlass.Float32,
    columns: cutlass.Constexpr,
    steps: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
) -> None:
    thread = cutlass.Int32(cute.arch.thread_idx()[0])
    allocator = cutlass.utils.SmemAllocator()
    key = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((16384,)), byte_alignment=1024
    )
    value = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((16384,)), byte_alignment=1024
    )
    query = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((128 * columns,)), byte_alignment=1024
    )
    probability = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((128 * columns,)), byte_alignment=128
    )
    maximum = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((columns,)), byte_alignment=16
    )
    alpha = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((columns,)), byte_alignment=16
    )
    scratch = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((4, columns), stride=(columns, 1)),
        byte_alignment=16,
    )
    partials = cute.make_rmem_tensor((columns,), cutlass.Float32)
    accumulator = cute.make_rmem_tensor((columns,), cutlass.Float32)
    visible = cute.make_rmem_tensor((columns,), cutlass.Boolean)
    for channel in cutlass.range_constexpr(128):
        offset = thread * 64 + ((channel % 64) ^ ((thread % 8) * 8))
        offset = offset + 8192 * (channel // 64)
        key[offset] = keys[thread, channel]
        value[offset] = values[thread, channel]
    for row in cutlass.range_constexpr(columns):
        offset = row * 64 + ((thread % 64) ^ ((row % 8) * 8))
        offset = offset + 64 * columns * (thread // 64)
        query[offset] = queries[row, thread]
        partials[row] = old_partials[thread, row]
        accumulator[row] = old_output[thread, row]
        visible[row] = source_visible[thread, row] != 0
    if thread < columns:
        maximum[thread] = old_maximum[thread]
        alpha[thread] = cutlass.Float32(7)
    base, done = conditional_attention_begin(accumulator, thread, columns)
    conditional_attention_page(
        key,
        value,
        query,
        probability,
        maximum,
        alpha,
        scratch,
        partials,
        accumulator,
        visible,
        thread,
        rescale_previous,
        score_factor,
        threshold,
        columns,
        *steps,
        fast_math,
        base,
        done,
    )
    conditional_attention_end(base, done, accumulator, thread, columns)
    for row in cutlass.range_constexpr(columns):
        new_partials[thread, row] = partials[row]
        new_output[thread, row] = accumulator[row]
        new_probability[row, thread] = probability[
            (row // 8) * 1024 + thread * 8 + row % 8
        ]
    if thread < columns:
        new_maximum[thread] = maximum[thread]


@cute.jit
def launch_conditional_controller(
    keys: cute.Tensor,
    values: cute.Tensor,
    queries: cute.Tensor,
    old_maximum: cute.Tensor,
    old_partials: cute.Tensor,
    old_output: cute.Tensor,
    source_visible: cute.Tensor,
    new_maximum: cute.Tensor,
    new_partials: cute.Tensor,
    new_output: cute.Tensor,
    new_probability: cute.Tensor,
    stream: object,
    rescale_previous: cutlass.Boolean,
    score_factor: cutlass.Float32,
    threshold: cutlass.Float32,
    columns: cutlass.Constexpr,
    steps: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
) -> None:
    observe_conditional_controller(
        keys,
        values,
        queries,
        old_maximum,
        old_partials,
        old_output,
        source_visible,
        new_maximum,
        new_partials,
        new_output,
        new_probability,
        rescale_previous,
        score_factor,
        threshold,
        columns,
        steps,
        fast_math,
    ).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)
