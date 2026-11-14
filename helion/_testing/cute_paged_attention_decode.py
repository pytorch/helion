"""Native row and transpose shared-byte-to-TMEM fixtures for decoder contracts."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from helion._compiler.cute._paged_attention_runtime import (
    _packed_scale_layout_supported,
)
from helion._compiler.cute._paged_attention_runtime import _scale_pairs_use_s2f6
from helion._compiler.cute._paged_attention_runtime import _scale_word_uses_s2f6
from helion._compiler.cute._paged_attention_runtime import stage_scaled_s8_rows_to_tmem
from helion._compiler.cute._paged_attention_runtime import (
    stage_scaled_s8_transpose_to_tmem,
)


@cute.kernel
def packet_kernel(
    raw: cute.Tensor,
    scales: cute.Tensor,
    output: cute.Tensor,
    scale_row_stride: cutlass.Constexpr[int],
    scale_swizzle: cutlass.Constexpr[bool],
    transpose: cutlass.Constexpr[bool],
    columns: cutlass.Constexpr[int],
    scale_group: cutlass.Constexpr[int],
    post_scale_log2: cutlass.Constexpr[int],
    positive_zero: cutlass.Constexpr[bool] = False,
) -> None:
    assert not transpose or (columns == 128 and scale_group == 32)
    assert not positive_zero or transpose
    row = cute.arch.thread_idx()[0]
    tile = cute.arch.block_idx()[0]
    raw_storage = cute.arch.alloc_smem(cutlass.Int8, 2 * 128 * 128, alignment=128)
    # Exercise the dynamic raw-slot offset used by the production rings.
    raw_ptr = raw_storage + cute.assume((tile % 2) * (128 * 128), divby=128 * 128)
    scale_storage = cute.arch.alloc_smem(
        cutlass.Uint8, 2 * 128 * scale_row_stride, alignment=16
    )
    # Mirror the production ring: every possible dynamic offset is 16-byte aligned.
    scale_ptr = scale_storage + cute.assume(
        (tile % 2) * (128 * scale_row_stride), divby=16
    )
    raw_layout = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3), 0, cute.make_layout((128, 128), stride=(128, 1))
    )
    shared_raw = cute.make_tensor(raw_ptr, raw_layout)
    scale_layout = cute.make_layout(
        (128, columns // scale_group), stride=(scale_row_stride, 1)
    )
    if cutlass.const_expr(scale_swizzle):
        scale_layout = cute.make_composed_layout(
            cute.make_swizzle(1, 0, 2), 0, scale_layout
        )
    shared_scales = cute.make_tensor(scale_ptr, scale_layout)
    assert _packed_scale_layout_supported(shared_scales) == (
        columns // scale_group == 4 and scale_row_stride == 4 and not scale_swizzle
    )
    slot = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=4)
    for column in cutlass.range_constexpr(columns):
        shared_raw[row, column] = raw[tile, row, column]
    for group in cutlass.range_constexpr(columns // scale_group):
        shared_scales[row, group] = scales[tile, row, group]
    if row < 32:
        prims.tcgen05_alloc(slot, 128, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
    cute.arch.barrier()
    base = cute.make_tensor(slot, cute.make_layout((1,)))[0]
    if cutlass.const_expr(transpose):
        stage_scaled_s8_transpose_to_tmem(
            shared_raw,
            shared_scales,
            row // 32 * 32,
            row % 32,
            base,
            128,
            post_scale_log2,
            positive_zero=positive_zero,
        )
    else:
        stage_scaled_s8_rows_to_tmem(
            shared_raw, shared_scales, row, base, columns, scale_group, post_scale_log2
        )
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier()
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    for group in cutlass.range_constexpr(columns // 32):
        address = (row // 32 * 32 << 16) | (base + group * 16)
        result = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(address, 6, cutlass.Int32), num=16
        )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        for j in cutlass.range_constexpr(16):
            output[tile, row, group * 16 + j] = result[j]
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier()
    if row < 32:
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        prims.tcgen05_dealloc(
            cutlass.inttoptr(base, 6, cutlass.Int32), 128, group=prims.CTAGroup.CTA_1
        )


@cute.jit
def launch(
    raw: cute.Tensor,
    scales: cute.Tensor,
    output: cute.Tensor,
    stream: object,
    scale_row_stride: cutlass.Constexpr[int] = 4,
    scale_swizzle: cutlass.Constexpr[bool] = False,
    transpose: cutlass.Constexpr[bool] = True,
    columns: cutlass.Constexpr[int] = 128,
    scale_group: cutlass.Constexpr[int] = 32,
    post_scale_log2: cutlass.Constexpr[int] = 0,
    positive_zero: cutlass.Constexpr[bool] = False,
) -> None:
    packet_kernel(
        raw,
        scales,
        output,
        scale_row_stride,
        scale_swizzle,
        transpose,
        columns,
        scale_group,
        post_scale_log2,
        positive_zero,
    ).launch(grid=(cute.size(raw, mode=[0]), 1, 1), block=(128, 1, 1), stream=stream)


@cute.kernel
def packed_guard_kernel(pairs: cute.Tensor, output: cute.Tensor) -> None:
    index = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
    pair = cutlass.Int32(pairs[index])
    safe_pair = cutlass.Int32(0x7654F7F7)
    output[index, 0] = cutlass.Int32(_scale_pairs_use_s2f6(pair, safe_pair))
    output[index, 1] = cutlass.Int32(_scale_pairs_use_s2f6(safe_pair, pair))
    for rotation in cutlass.range_constexpr(4):
        word = cutlass.Uint32(pair)
        if cutlass.const_expr(rotation != 0):
            low = cutlass.Uint32(word << (8 * rotation))
            high = cutlass.Uint32(word >> (32 - 8 * rotation))
            word = cutlass.Uint32(low | high)
        output[index, 2 + rotation] = cutlass.Int32(
            _scale_word_uses_s2f6(cutlass.Int32(word))
        )


@cute.jit
def launch_packed_guard(
    pairs: cute.Tensor, output: cute.Tensor, stream: object
) -> None:
    packed_guard_kernel(pairs, output).launch(
        grid=(512, 1, 1), block=(128, 1, 1), stream=stream
    )
