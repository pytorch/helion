"""Complete native input fragments for structurally proved row reductions."""

from __future__ import annotations

from typing import cast

import cutlass
from cutlass import cute


@cute.jit
def _cute_complete_row_tensor_copy(
    source: cute.Tensor,
    row: cutlass.Int32,
    column_tile: cutlass.Int32,
    *,
    columns: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    vector_width: cutlass.Constexpr,
) -> cute.TensorSSA:
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        source.element_type,
        num_bits_per_copy=vector_width * source.element_type.width,
    )
    thread_layout = cute.make_ordered_layout((1, threads), order=(1, 0))
    value_layout = cute.make_layout((1, vector_width))
    tiled_copy = cute.make_tiled_copy_tv(atom, thread_layout, value_layout)
    tile = (1, columns)
    data = cute.local_tile(source, tile, (row, column_tile))
    identity = cute.make_identity_tensor(source.shape)
    coordinates = cute.local_tile(identity, tile, (row, column_tile))
    thread_copy = tiled_copy.get_slice(cute.arch.thread_idx()[0])
    partition = thread_copy.partition_S(data)
    coordinate_partition = thread_copy.partition_S(coordinates)
    predicates = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(coordinate_partition, mode=[0, 1]),
                cute.size(coordinate_partition, mode=[1]),
                cute.size(coordinate_partition, mode=[2]),
            ),
            stride=(cute.size(coordinate_partition, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for vector_rest in cutlass.range_constexpr(predicates.shape[0]):
        for column_rest in cutlass.range_constexpr(predicates.shape[2]):
            predicates[vector_rest, 0, column_rest] = cute.elem_less(
                coordinate_partition[(0, vector_rest), 0, column_rest][1],
                cast("tuple[int, int]", source.shape)[1],
            )
    raw = cute.make_rmem_tensor_like(partition)
    cute.copy(atom, partition, raw, pred=predicates)
    fill = cute.make_rmem_tensor_like(raw[(None, 0), None, 0])
    fill.fill(-raw.element_type.inf)
    for vector_rest in cutlass.range_constexpr(raw.shape[0][1]):
        for column_rest in cutlass.range_constexpr(raw.shape[2]):
            if not predicates[vector_rest, 0, column_rest]:
                cute.autovec_copy(fill, raw[(None, vector_rest), None, column_rest])
    return raw.load().to(cutlass.Float32)
