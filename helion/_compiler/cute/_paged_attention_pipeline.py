"""The 13-warp SM100 paged-attention schedule.

Four compute warps own the online recurrence. Two groups of four conversion
warps consume raw K/V TMA rings and write disjoint TMEM panels; the last warp
owns tensor-map loads. Local barriers protect raw and converted slot reuse.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as experimental_cuda
import cutlass.experimental.primitives as prims

from ._mma_dag_tcgen05_runtime import mbar_wait
from ._paged_attention_compute import mma_128x64x128
from ._paged_attention_compute import online_softmax_128x64
from ._paged_attention_runtime import _packed_scale_layout_supported
from ._paged_attention_runtime import enqueue_paged_tma_operand
from ._paged_attention_runtime import stage_scaled_s8_rows_to_tmem
from ._paged_attention_runtime import stage_scaled_s8_transpose_to_tmem


@cute.jit
def paged_attention_body(
    q: cute.Tensor,
    pages: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor | None,
    tile: cutlass.Int32,
    first_page: cutlass.Int64,
    last_page: cutlass.Int64,
    length: cutlass.Int32,
    query_len: cutlass.Constexpr,
    group: cutlass.Constexpr,
    query_rows: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    window: cutlass.Constexpr,
    score_scale: cutlass.Float32,
    fast_math: cutlass.Constexpr,
    kmap: experimental_cuda.TensorMap,
    ksmap: experimental_cuda.TensorMap,
    vmap: experimental_cuda.TensorMap,
    vsmap: experimental_cuda.TensorMap,
    coordinate_dtype: cutlass.Constexpr,
) -> None:
    thread = cute.arch.thread_idx()[0]
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = thread % 32
    head = tile % heads
    request = tile // heads
    # Source endpoints were evaluated at their original integer widths before
    # this call. Widening before subtraction preserves range(begin, end).
    count = cutlass.Int64(0)
    if last_page > first_page:
        count = last_page - first_page
    raw_k = cute.arch.alloc_smem(cutlass.Int8, 2 * 16384, alignment=1024)
    raw_v = cute.arch.alloc_smem(cutlass.Int8, 2 * 16384, alignment=1024)
    scale_k = cute.arch.alloc_smem(cutlass.Uint8, 2 * 512, alignment=128)
    scale_v = cute.arch.alloc_smem(cutlass.Uint8, 2 * 512, alignment=128)
    raw_layout = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3), 0, cute.make_layout((128, 128), stride=(128, 1))
    )
    scale_layout = cute.make_layout((128, 4), stride=(4, 1))
    shared_q = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.BFloat16, 8192, alignment=1024),
        cute.make_layout((8192,)),
    )
    probability = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.BFloat16, 8192, alignment=1024),
        cute.make_layout((8192,)),
    )
    state = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, 256, alignment=16),
        cute.make_layout((4, 64), stride=(64, 1)),
    )
    scratch = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, 256, alignment=16),
        cute.make_layout((4, 64), stride=(64, 1)),
    )
    slot = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=4)
    raw_ready = cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)
    raw_free = cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)
    key_ready = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    key_free = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    value_ready = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    value_free = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    qk_done = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    pv_done = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    if thread == 0:
        for ring in cutlass.range_constexpr(2):
            prims.mbarrier_init(raw_ready + ring, 2)
            prims.mbarrier_init(raw_free + ring, 8)
        prims.mbarrier_init(key_ready, 4)
        prims.mbarrier_init(value_ready, 4)
        prims.mbarrier_init(key_free, 1)
        prims.mbarrier_init(value_free, 1)
        prims.mbarrier_init(qk_done, 1)
        prims.mbarrier_init(pv_done, 1)
        prims.fence_mbarrier_init()
    if thread < 64:
        state[0, thread] = cutlass.Float32(float("-inf"))
        state[1, thread] = cutlass.Float32(0)
        state[2, thread] = cutlass.Float32(0)
        state[3, thread] = cutlass.Float32(1)
        for k in cutlass.range_constexpr(128):
            offset = thread * 64 + ((k % 64) ^ ((thread % 8) * 8)) + (k // 64) * 4096
            value = cutlass.BFloat16(0)
            if thread < query_rows:
                value = q[tile, thread, k]
            shared_q[offset] = value
    if warp == 0:
        prims.tcgen05_alloc(slot, 256, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    cute.arch.barrier()
    base = cute.make_tensor(slot, cute.make_layout((1,)))[0]
    if warp < 4:
        output_address = (warp * 32 << 16) | (base + 192)
        zero = cutlass.Array(cutlass.Float32, 64, alignment=16)
        for query in cutlass.range_constexpr(64):
            zero[query] = cutlass.Float32(0)
        prims.tcgen05_st(
            "32x32b", cutlass.inttoptr(output_address, 6, cutlass.Float32), zero[0:64]
        )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.barrier(barrier_id=2, number_of_threads=128)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        for step in cutlass.range(count):
            mbar_wait(key_ready, step & 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if warp == 0:
                mma_128x64x128(base, base + 128, shared_q.iterator, qk_done, False)
                if prims.elect_sync():
                    prims.tcgen05_commit(key_free, group=prims.CTAGroup.CTA_1)
            mbar_wait(qk_done, step & 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            online_softmax_128x64(
                base + 128,
                probability,
                state,
                scratch,
                thread,
                cutlass.Int64((first_page + step) * 128),
                length,
                query_len,
                group,
                window,
                score_scale,
                1,
                fast_math,
                coordinate_dtype,
            )
            previous = prims.tcgen05_ld(
                "32x32b", cutlass.inttoptr(output_address, 6, cutlass.Float32), num=64
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            rescaled = cutlass.Array(cutlass.Float32, 64, alignment=16)
            for query in cutlass.range_constexpr(64):
                rescaled[query] = previous[query] * state[2, query]
            prims.tcgen05_st(
                "32x32b",
                cutlass.inttoptr(output_address, 6, cutlass.Float32),
                rescaled[0:64],
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.barrier(barrier_id=2, number_of_threads=128)
            mbar_wait(value_ready, step & 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if warp == 0:
                mma_128x64x128(
                    base + 64, base + 192, probability.iterator, pv_done, True
                )
                if prims.elect_sync():
                    prims.tcgen05_commit(value_free, group=prims.CTAGroup.CTA_1)
            mbar_wait(pv_done, step & 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        result = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(output_address, 6, cutlass.Float32), num=64
        )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        for query in cutlass.range_constexpr(64):
            if cutlass.const_expr(query < query_rows):
                output[tile, query, thread] = cutlass.BFloat16(
                    result[query] / state[3, query]
                )
        if cutlass.const_expr(lse is not None):
            assert lse is not None
            if thread < query_rows:
                lse[tile, thread] = state[0, thread] + cute.math.log(
                    state[1, thread], fastmath=fast_math
                )
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    elif warp < 8:
        row = (warp - 4) * 32 + lane
        for step in cutlass.range(count):
            ring = step & 1
            mbar_wait(raw_ready + ring, step // 2 & 1)
            mbar_wait(key_free, (step & 1) ^ 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            # Every 512-byte slot preserves the allocation's 128-byte alignment.
            staged_scales = cute.make_tensor(
                scale_k + cute.assume(ring * 512, divby=512), scale_layout
            )
            assert _packed_scale_layout_supported(staged_scales)
            stage_scaled_s8_rows_to_tmem(
                cute.make_tensor(
                    raw_k + cute.assume(ring * 16384, divby=16384), raw_layout
                ),
                staged_scales,
                row,
                base,
                128,
                32,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.sync_warp()
            if lane == 0:
                prims.mbarrier_arrive(key_ready)
                prims.mbarrier_arrive(raw_free + ring)
    elif warp < 12:
        dimension = (warp - 8) * 32
        for step in cutlass.range(count):
            ring = step & 1
            mbar_wait(raw_ready + ring, step // 2 & 1)
            mbar_wait(value_free, (step & 1) ^ 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            # Re-establish alignment after the dynamic offset: the 128-byte
            # allocation and 512-byte slots make align's round-up a no-op.
            staged_scales = cute.make_tensor(
                scale_v + cute.assume(ring * 512, divby=512), scale_layout
            )
            assert _packed_scale_layout_supported(staged_scales)
            stage_scaled_s8_transpose_to_tmem(
                cute.make_tensor(
                    raw_v + cute.assume(ring * 16384, divby=16384), raw_layout
                ),
                staged_scales,
                dimension,
                lane,
                base + 64,
                128,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.sync_warp()
            if lane == 0:
                prims.mbarrier_arrive(value_ready)
                prims.mbarrier_arrive(raw_free + ring)
    else:
        if lane == 0:
            for step in cutlass.range(count):
                ring = step & 1
                mbar_wait(raw_free + ring, (step // 2 & 1) ^ 1)
                physical = cutlass.Int32(pages[request, first_page + step])
                enqueue_paged_tma_operand(
                    kmap,
                    ksmap,
                    raw_k + ring * 16384,
                    scale_k + ring * 512,
                    raw_ready + ring,
                    physical,
                    cutlass.Int32(0),
                    head,
                    128,
                )
                enqueue_paged_tma_operand(
                    vmap,
                    vsmap,
                    raw_v + ring * 16384,
                    scale_v + ring * 512,
                    raw_ready + ring,
                    physical,
                    cutlass.Int32(0),
                    head,
                    128,
                )
    cute.arch.barrier()
    if warp == 0:
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        prims.tcgen05_dealloc(
            cutlass.inttoptr(base, 6, cutlass.Int32), 256, group=prims.CTAGroup.CTA_1
        )
