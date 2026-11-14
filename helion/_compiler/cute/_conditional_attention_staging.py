"""Coalesced native byte pairs in the existing conditional shared-memory layout."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from ._paged_attention_runtime import _native_s2f6x2_to_bf16x2


@cute.jit
def stage_native_ss_pairs(
    raw: cute.Tensor,
    scales: cute.Tensor,
    destination: cute.Tensor,
    physical_page: cutlass.Int64,
    head: cutlass.Int64,
    row_shift: cutlass.Int64,
    thread: cutlass.Int32,
    raw_shape: cutlass.Constexpr,
    raw_strides: cutlass.Constexpr,
    scale_strides: cutlass.Constexpr,
    index_type: cutlass.Constexpr,
) -> None:
    """Retain original clamped loads, bounds and +0 masking for all 128x128 values.

    The caller proves the actual raw pointer is two-byte aligned, channel
    stride is one, and all outer byte strides are even. Adjacent channels
    never cross a scale32 group. Both K and V use this same SS layout; this
    is not the distinct transposed TMEM converter. Publication remains owned
    by the unchanged page controller's converged CTA join.
    """
    # Keep live packets bounded; fully unrolling all rows can spill the caller.
    for iteration in cutlass.range(64, unroll=1):
        linear = 2 * thread + 256 * iteration
        row = linear // 128
        channel = linear % 128
        shifted = index_type(row) + index_type(row_shift)
        safe_row = cutlass.min(cutlass.max(shifted, 0), 127)
        word = cutlass.Int32(0)
        scale = cutlass.Int32(0)
        if (
            physical_page >= 0
            and physical_page < raw_shape[0]
            and head >= 0
            and head < raw_shape[2]
        ):
            raw_offset = (
                cutlass.Int64(physical_page) * raw_strides[0]
                + cutlass.Int64(safe_row) * raw_strides[1]
                + cutlass.Int64(head) * raw_strides[2]
                + cutlass.Int64(channel)
            )
            pair = cute.recast_ptr(
                (raw.iterator + raw_offset).align(2), dtype=cutlass.Uint16
            )
            word = cutlass.Int32(pair.load())
            scale_offset = (
                cutlass.Int64(physical_page) * scale_strides[0]
                + cutlass.Int64(safe_row) * scale_strides[1]
                + cutlass.Int64(head) * scale_strides[2]
                + cutlass.Int64(channel // 32) * scale_strides[3]
            )
            scale = cutlass.Int32(
                cutlass.Uint8((scales.iterator + scale_offset).load())
            )
        decoded = _native_s2f6x2_to_bf16x2(word, scale * 0x0101)
        if shifted < 0 or shifted >= 128:
            decoded = cutlass.Int32(0)
        offset = row * 64 + ((channel % 64) ^ ((row % 8) * 8)) + 8192 * (channel // 64)
        pair_output = cute.recast_ptr(
            (destination.iterator + offset).align(4), dtype=cutlass.Int32
        )
        pair_output.store(decoded)
