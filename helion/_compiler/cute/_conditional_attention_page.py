"""Serial SS page controller for a separately proved conditional root.

The caller supplies complete BF16 panels, original visibility and FP32 carries.
The caller owns one nonempty-loop lifetime; each page owns panel publication
and both MMA completions. No helper derives input addresses or enables a route.
"""

from __future__ import annotations

from typing import cast

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from ._conditional_attention_compute import conditional_probability
from ._conditional_attention_mma import conditional_mma_ss
from ._paged_attention_partial_compute import load_query_panel
from ._paged_attention_partial_compute import store_query_panel
from .device.sync import mbar_wait


@cute.jit
def conditional_attention_begin(
    accumulator: cute.Tensor,
    thread: cutlass.Int32,
    columns: cutlass.Constexpr,
) -> tuple[cutlass.Int32, cute.Pointer]:
    """Acquire a nonempty loop's TMEM and publish its original FP32 O once."""
    assert cutlass.const_expr(columns in (8, 16, 32, 40, 64))
    warp = thread // 32
    slot = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=4)
    done = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    if thread == 0:
        prims.mbarrier_init(done, 1)
        prims.fence_mbarrier_init()
    if warp == 0:
        prims.tcgen05_alloc(slot, 128, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    base = cute.make_tensor(slot, cute.make_layout((1,)))[0]
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    store_query_panel((warp * 32 << 16) | (base + 64), accumulator, columns)
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    return base, done


@cute.jit
def conditional_attention_end(
    base: cutlass.Int32,
    done: cute.Pointer,
    accumulator: cute.Tensor,
    thread: cutlass.Int32,
    columns: cutlass.Constexpr,
) -> None:
    """Snapshot final O, then retire the completed nonempty loop's resources."""
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    output = load_query_panel(((thread // 32) * 32 << 16) | (base + 64), columns)
    for row in cutlass.range_constexpr(columns):
        accumulator[row] = output[row]
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    if thread // 32 == 0:
        prims.tcgen05_dealloc(
            cutlass.inttoptr(base, 6, cutlass.Float32),
            128,
            group=prims.CTAGroup.CTA_1,
        )
    if thread == 0:
        prims.mbarrier_inval(done)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)


@cute.jit
def conditional_attention_page(
    key: cute.Tensor,
    value: cute.Tensor,
    query: cute.Tensor,
    probability: cute.Tensor,
    maximum: cute.Tensor,
    alpha: cute.Tensor,
    scratch: cute.Tensor,
    partials: cute.Tensor,
    accumulator: cute.Tensor,
    visible: cute.Tensor,
    thread: cutlass.Int32,
    rescale_previous: cutlass.Boolean,
    score_factor: cutlass.Float32,
    threshold: cutlass.Float32,
    columns: cutlass.Constexpr,
    key_steps: cutlass.Constexpr,
    value_steps: cutlass.Constexpr,
    query_steps: cutlass.Constexpr,
    probability_steps: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
    base: cutlass.Int32,
    done: cute.Pointer,
) -> None:
    # These are the widths supported by the existing exact TMEM panel reader.
    assert cutlass.const_expr(columns in (8, 16, 32, 40, 64))
    warp = thread // 32
    # Publish this page's panels even when the lifetime began on an earlier page.
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    # Pair the preceding page's BEFORE/join with this page's TMEM accesses.
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    output_address = (warp * 32 << 16) | (base + 64)
    # O remains in columns [64,64+columns); QK only overwrites [0,columns).
    # Keep the existing page rendezvous while removing the redundant O store.
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if warp == 0:
        a = prims.Tcgen05SmemDesc.build(
            cutlass.Int32(cast("cute.Pointer", key.iterator).toint()),
            leading_byte_offset=16,
            stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        b = prims.Tcgen05SmemDesc.build(
            cutlass.Int32(cast("cute.Pointer", query.iterator).toint()),
            leading_byte_offset=16,
            stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        conditional_mma_ss(
            a, b, base, done, key_steps, query_steps, columns, False, False, False
        )
    mbar_wait(done, cutlass.Int32(0))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    # Retire every phase0 waiter before the issuer can advance phase1.
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    scores = load_query_panel((warp * 32 << 16) | base, columns)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
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
        columns,
        0,
        fast_math,
    )
    if rescale:
        output = load_query_panel(output_address, columns)
        for row in cutlass.range_constexpr(columns):
            accumulator[row] = cutlass.Float32(output[row]) * cutlass.Float32(
                alpha[row]
            )
        store_query_panel(output_address, accumulator, columns)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if warp == 0:
        a = prims.Tcgen05SmemDesc.build(
            cutlass.Int32(cast("cute.Pointer", value.iterator).toint()),
            leading_byte_offset=16384,
            stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        b = prims.Tcgen05SmemDesc.build(
            cutlass.Int32(cast("cute.Pointer", probability.iterator).toint()),
            leading_byte_offset=128,
            stride_byte_offset=2048,
            layout=prims.Tcgen05SmemSwizzle.NONE,
        )
        conditional_mma_ss(
            a,
            b,
            base + 64,
            done,
            value_steps,
            probability_steps,
            columns,
            True,
            True,
            True,
        )
    mbar_wait(done, cutlass.Int32(1))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    # Both completions have advanced the barrier back to phase zero. The join
    # above retires all phase-one waiters before the next
    # page can issue QK, or the caller can invalidate/deallocate the resources.
