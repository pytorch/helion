"""Copies and sums for compiler-proven resident feature fragments."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import CopyUniversalOp


@cute.jit
def _cute_resident_copy_async(
    source: cute.Tensor,
    destination: cute.Tensor,
    source_offset: cutlass.Int32,
    destination_offset: cutlass.Int32,
    width: cutlass.Constexpr[int],
) -> None:
    byte_count: cutlass.Constexpr = width * source.element_type.width // 8
    if cutlass.const_expr(byte_count in (4, 8, 16)):
        if source_offset % width == 0:
            cute.arch.cp_async_shared_global(
                destination.iterator + cute.assume(destination_offset, divby=width),
                source.iterator + cute.assume(source_offset, divby=width),
                byte_count,
                "cg" if cutlass.const_expr(byte_count == 16) else "ca",
            )
        else:
            for lane in cutlass.range_constexpr(width):
                (destination.iterator + destination_offset + lane).store(
                    (source.iterator + source_offset + cutlass.Int32(lane)).load()
                )
    else:
        for lane in cutlass.range_constexpr(width):
            (destination.iterator + destination_offset + lane).store(
                (source.iterator + source_offset + cutlass.Int32(lane)).load()
            )


@cute.jit
def _cute_resident_sums(
    values: cute.Tensor, threads: cutlass.Constexpr[int]
) -> cute.Tensor:
    result = cute.make_rmem_tensor(values.shape, cutlass.Float32)
    if cutlass.const_expr(threads <= 32):
        for index in cutlass.range_constexpr(cute.size(values)):
            if cutlass.const_expr(threads == 1):
                result[index] = values[index]
            else:
                result[index] = cute.arch.warp_reduction_sum(
                    values[index], threads_in_group=threads
                )
    else:
        warps: cutlass.Constexpr = threads // 32
        tid = cutlass.Int32(cute.arch.thread_idx()[0])
        shared = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Float32, cute.make_layout(cute.size(values) * warps)
        )
        for index in cutlass.range_constexpr(cute.size(values)):
            value = cute.arch.warp_reduction_sum(values[index])
            if tid % 32 == 0:
                shared[index * warps + tid // 32] = value
        cute.arch.sync_threads()
        for index in cutlass.range_constexpr(cute.size(values)):
            # Every warp independently combines the published warp partials.
            # Repeating the same contiguous subgroup across a warp broadcasts
            # the result to every feature lane with one shared load per lane.
            value = shared[index * warps + tid % warps]
            value = cute.arch.warp_reduction_sum(value, threads_in_group=warps)
            # Keep the original positive-zero identity of the shared combine.
            result[index] = cutlass.Float32(0) + value
        # Every reader finishes before the next row group reuses this array.
        cute.arch.sync_threads()
    return result


@cute.jit
def _cute_resident_sums_disjoint(
    values: cute.Tensor,
    threads: cutlass.Constexpr[int],
    scratch: cute.Tensor,
    slot: cutlass.Constexpr[int],
) -> cute.Tensor:
    # The caller supplies disjoint row slices and the retirement barrier.
    # A serial deferred caller proves its next-group ready barrier dominates
    # same-slice reuse; this helper retains the full-CTA publication barrier.
    result = cute.make_rmem_tensor(values.shape, cutlass.Float32)
    if cutlass.const_expr(threads <= 32):
        for index in cutlass.range_constexpr(cute.size(values)):
            if cutlass.const_expr(threads == 1):
                result[index] = values[index]
            else:
                result[index] = cute.arch.warp_reduction_sum(
                    values[index], threads_in_group=threads
                )
    else:
        warps: cutlass.Constexpr = threads // 32
        tid = cutlass.Int32(cute.arch.thread_idx()[0])
        shared = cute.make_tensor(
            scratch.iterator + slot * cute.size(values) * warps,
            cute.make_layout(cute.size(values) * warps),
        )
        for index in cutlass.range_constexpr(cute.size(values)):
            value = cute.arch.warp_reduction_sum(values[index])
            if tid % 32 == 0:
                shared[index * warps + tid // 32] = value
        cute.arch.sync_threads()
        for index in cutlass.range_constexpr(cute.size(values)):
            # Every warp independently combines the published warp partials.
            # Repeating the same contiguous subgroup across a warp broadcasts
            # the result to every feature lane with one shared load per lane.
            value = shared[index * warps + tid % warps]
            value = cute.arch.warp_reduction_sum(value, threads_in_group=warps)
            # Keep the original positive-zero identity of the shared combine.
            result[index] = cutlass.Float32(0) + value
    return result


@cute.jit
def _cute_resident_load_vector(
    tensor: cute.Tensor,
    offset: cutlass.Int32,
    width: cutlass.Constexpr[int],
) -> cute.Tensor:
    result = cute.make_rmem_tensor(width, tensor.element_type)
    # The caller proves the tensor base alignment and complete vector domain.
    # Checking the final Int32 offset also handles a misaligned row/view.
    if offset % width == 0:
        source = cute.make_tensor(
            tensor.iterator + cute.assume(offset, divby=width), cute.make_layout(width)
        )
        atom = cute.make_copy_atom(
            CopyUniversalOp(),
            tensor.element_type,
            num_bits_per_copy=min(128, width * tensor.element_type.width),
        )
        cute.copy(atom, source, result)
    else:
        for lane in cutlass.range_constexpr(width):
            result[lane] = (tensor.iterator + offset + cutlass.Int32(lane)).load()
    return result


@cute.jit
def _cute_resident_store_vector(
    tensor: cute.Tensor,
    offset: cutlass.Int32,
    fragment: cute.Tensor,
    width: cutlass.Constexpr[int],
) -> None:
    if offset % width == 0:
        destination = cute.make_tensor(
            tensor.iterator + cute.assume(offset, divby=width), cute.make_layout(width)
        )
        atom = cute.make_copy_atom(
            CopyUniversalOp(),
            tensor.element_type,
            num_bits_per_copy=min(128, width * tensor.element_type.width),
        )
        cute.copy(atom, fragment, destination)
    else:
        for lane in cutlass.range_constexpr(width):
            (tensor.iterator + offset + cutlass.Int32(lane)).store(fragment[lane])
