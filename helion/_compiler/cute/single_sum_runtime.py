"""One independent output of the audited paired host FP32 sum/cast tree."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from ...runtime.cute.source_dependencies import set_helper_source_hash
from .resident_reduction_runtime import _cute_resident_load_vector
from .resident_reduction_runtime import _cute_resident_store_vector


@cute.kernel
def single_sum_cast_kernel(
    left: cute.Tensor,
    left_out: cute.Tensor,
    rows: cutlass.Constexpr[int],
    columns: cutlass.Constexpr[int],
    row_lanes: cutlass.Constexpr[int],
    column_threads: cutlass.Constexpr[int],
    vector: cutlass.Constexpr[int],
) -> None:
    tx = cutlass.Int32(cute.arch.thread_idx()[0])
    ty = cutlass.Int32(cute.arch.thread_idx()[1])
    column = (cutlass.Int32(cute.arch.block_idx()[0]) * column_threads + tx) * vector
    accum = cute.make_rmem_tensor((4, vector), cutlass.Float32)
    accum.fill(cutlass.Float32(0))
    # Preserve all four independent FP32 chains, including their +0 identity.
    for chunk in range(cute.ceil_div(rows, 4 * row_lanes)):
        for slot in cutlass.range_constexpr(4):
            row = (chunk * 4 + slot) * row_lanes + ty
            if row < rows and column < columns:
                offset_left = row * cutlass.Int32(left.layout.stride[0]) + column  # pyrefly: ignore[missing-attribute]
                values_left = _cute_resident_load_vector(left, offset_left, vector)
                for lane in cutlass.range_constexpr(vector):
                    accum[slot, lane] = accum[slot, lane] + values_left[lane]
    for slot in cutlass.range_constexpr(1, 4):
        for lane in cutlass.range_constexpr(vector):
            accum[0, lane] = accum[0, lane] + accum[slot, lane]
    if cutlass.const_expr(row_lanes > 1):
        shared = cute.make_tensor(
            cute.arch.alloc_smem(
                cutlass.Float32,
                row_lanes * column_threads * vector,
                alignment=16,
            ),
            cute.make_layout(row_lanes * column_threads * vector),
        )
        temp_left = cute.make_rmem_tensor(vector, cutlass.Float32)
        for lane in cutlass.range_constexpr(vector):
            temp_left[lane] = accum[0, lane]
        shared_left = (ty * column_threads + tx) * vector
        _cute_resident_store_vector(shared, shared_left, temp_left, vector)
        for level in cutlass.range_constexpr(row_lanes.bit_length() - 1):
            offset: cutlass.Constexpr = row_lanes >> (level + 1)
            # Inactive columns still participate in every CTA barrier.
            cute.arch.sync_threads()
            if ty < offset:
                other_left = _cute_resident_load_vector(
                    shared, shared_left + offset * column_threads * vector, vector
                )
                for lane in cutlass.range_constexpr(vector):
                    accum[0, lane] = accum[0, lane] + other_left[lane]
                    temp_left[lane] = accum[0, lane]
                _cute_resident_store_vector(shared, shared_left, temp_left, vector)
    if ty == 0 and column < columns:
        final_left = cute.make_rmem_tensor(vector, left_out.element_type)
        for lane in cutlass.range_constexpr(vector):
            final_left[lane] = left_out.element_type(accum[0, lane])
        _cute_resident_store_vector(left_out, column, final_left, vector)


set_helper_source_hash(single_sum_cast_kernel, "single_host_sum")
