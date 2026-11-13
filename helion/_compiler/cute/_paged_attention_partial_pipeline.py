"""Paged attention for an explicit per-key denominator recurrence.

Four compute warps own online state. Four conversion warps stage K then V;
all four are required by the physical TMEM row mapping. The ninth warp owns
raw TMA and the tenth issues QK ahead of the current softmax. Separate key/value
release barriers protect the converted panels. The optional twelve-warp schedule
adds two idle control warps to complete the register-donating warpgroup. It gives
compute/conversion/control threads 128/64/32 registers. Keep each quota call
inside its role branch: the vendor compiler rejected a join placed directly
after divergent quota changes. The scoped schedule requires two resident CTAs
so the entry budget is below the compute quota and above the donor quotas.
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as experimental_cuda
import cutlass.experimental.primitives as prims

from ._mma_dag_tcgen05_runtime import mbar_wait
from ._paged_attention_io import load_query_element
from ._paged_attention_io import store_lse_element
from ._paged_attention_io import store_output_element
from ._paged_attention_partial_compute import load_query_panel
from ._paged_attention_partial_compute import mma_128xnx128
from ._paged_attention_partial_compute import online_softmax_per_key
from ._paged_attention_partial_compute import store_query_panel
from ._paged_attention_reference_compute import online_softmax_reference
from ._paged_attention_runtime import _packed_scale_layout_supported
from ._paged_attention_runtime import enqueue_paged_tma_operand
from ._paged_attention_runtime import stage_scaled_s8_rows_to_tmem
from ._paged_attention_runtime import stage_scaled_s8_transpose_to_tmem


@cute.jit
def attention_page_coordinates(
    first: cutlass.Int64,
    last: cutlass.Int64,
    step: cutlass.Int64,
    reflected_dtype: cutlass.Constexpr = None,
) -> tuple[cutlass.Int64, cutlass.Int64]:
    """Keep typed source reflection and key-width multiplication before widening."""
    iteration = first + step
    page = iteration
    key_base = iteration * 128
    if cutlass.const_expr(reflected_dtype is not None):
        total = reflected_dtype(reflected_dtype(first) + reflected_dtype(last))
        prefix = reflected_dtype(total - reflected_dtype(1))
        token = reflected_dtype(cutlass.Int64(prefix) - iteration)
        page = cutlass.Int64(token)
        key_base = cutlass.Int64(reflected_dtype(token * reflected_dtype(128)))
    return page, key_base


@cute.jit
def paged_attention_partial_body(
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
    query_columns: cutlass.Constexpr = 40,
    io_layouts: cutlass.Constexpr = None,
    scoped_registers: cutlass.Constexpr = False,
    reference_scores: cutlass.Constexpr = False,
    reflected_dtype: cutlass.Constexpr = None,
    base2_scores: cutlass.Constexpr = False,
) -> None:
    assert cutlass.const_expr(not base2_scores or reference_scores)
    thread = cute.arch.thread_idx()[0]
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = thread % 32
    head = tile % heads
    request = tile // heads
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
        cute.arch.alloc_smem(cutlass.BFloat16, query_columns * 128, alignment=1024),
        cute.make_layout((query_columns * 128,)),
    )
    probability = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.BFloat16, query_columns * 128, alignment=1024),
        cute.make_layout((query_columns * 128,)),
    )
    state = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, 4 * query_columns, alignment=16),
        cute.make_layout((4, query_columns), stride=(query_columns, 1)),
    )
    scratch = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, 4 * query_columns, alignment=16),
        cute.make_layout((4, query_columns), stride=(query_columns, 1)),
    )
    slot = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=4)
    raw_ready = cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)
    raw_free = cute.arch.alloc_smem(cutlass.Int64, 2, alignment=8)
    key_ready = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    key_free = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    value_ready = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    value_free = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    score_free = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    qk_done = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    pv_done = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    if thread == 0:
        for ring in cutlass.range_constexpr(2):
            prims.mbarrier_init(raw_ready + ring, 2)
            prims.mbarrier_init(raw_free + ring, 4)
        prims.mbarrier_init(key_ready, 4)
        prims.mbarrier_init(value_ready, 4)
        prims.mbarrier_init(key_free, 1)
        prims.mbarrier_init(value_free, 1)
        prims.mbarrier_init(score_free, 4)
        prims.mbarrier_init(qk_done, 1)
        prims.mbarrier_init(pv_done, 1)
        prims.fence_mbarrier_init()
    if thread < query_columns:
        state[0, thread] = cutlass.Float32(float("-inf"))
        # The reference-score recurrence reads this cached zero on its first tile.
        state[1, thread] = cutlass.Float32(0)
        state[2, thread] = cutlass.Float32(0)
        state[3, thread] = cutlass.Float32(1)
    # Conversion and TMA warps help stage Q too. Only the QK warp reads it,
    # after the common fence and CTA barrier below. A uniform trip count keeps
    # the subsequent tensor-memory instructions out of divergent loop control.
    # Request full unrolling for scheduling freedom; the backend still decides
    # whether it can resolve the CTA width and fully expand the loop.
    for start in cutlass.range(
        0, query_columns * 128, cute.arch.block_dim()[0], unroll_full=True
    ):
        element = start + thread
        if element < query_columns * 128:
            query_row = element // 128
            k = element % 128
            offset = (
                query_row * 64
                + ((k % 64) ^ ((query_row % 8) * 8))
                + (k // 64) * (query_columns * 64)
            )
            value = cutlass.BFloat16(0)
            if query_row < query_rows:
                value = load_query_element(
                    q, tile, query_row, k, heads, group, io_layouts
                )
            shared_q[offset] = value
    if warp == 0:
        prims.tcgen05_alloc(slot, 256, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    cute.arch.barrier()
    base = cute.make_tensor(slot, cute.make_layout((1,)))[0]
    if warp < 4:
        if cutlass.const_expr(scoped_registers):
            cute.arch.setmaxregister_increase(128)
        output_address = (warp * 32 << 16) | (base + 192)
        zero = cutlass.Array(cutlass.Float32, query_columns, alignment=16)
        for query in cutlass.range_constexpr(query_columns):
            zero[query] = cutlass.Float32(0)
        store_query_panel(output_address, zero, query_columns)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.barrier(barrier_id=2, number_of_threads=128)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        partials = cute.make_rmem_tensor((query_columns,), cutlass.Float32)
        for query in cutlass.range_constexpr(query_columns):
            partials[query] = cutlass.Float32(0)
        reference_invalid = cutlass.Boolean(True)
        for step in cutlass.range(count):
            _logical_page, logical_key_base = attention_page_coordinates(
                first_page, last_page, step, reflected_dtype
            )
            mbar_wait(qk_done, step & 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if cutlass.const_expr(reference_scores):
                maximum_changed, reference_invalid = online_softmax_reference(
                    base + 128,
                    probability,
                    state,
                    scratch,
                    thread,
                    logical_key_base,
                    length,
                    query_len,
                    group,
                    window,
                    score_scale,
                    1,
                    partials,
                    reference_invalid,
                    fast_math,
                    score_free,
                    coordinate_dtype,
                    query_columns,
                    base2_scores,
                )
            else:
                maximum_changed = online_softmax_per_key(
                    base + 128,
                    probability,
                    state,
                    scratch,
                    thread,
                    logical_key_base,
                    length,
                    query_len,
                    group,
                    window,
                    score_scale,
                    1,
                    partials,
                    fast_math,
                    score_free,
                    coordinate_dtype,
                    query_columns,
                )
            if maximum_changed:
                previous = load_query_panel(output_address, query_columns)
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                rescaled = cutlass.Array(cutlass.Float32, query_columns, alignment=16)
                for query in cutlass.range_constexpr(query_columns):
                    rescaled[query] = previous[query] * state[2, query]
                store_query_panel(output_address, rescaled, query_columns)
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
            mbar_wait(value_ready, step & 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if warp == 0:
                mma_128xnx128(
                    base + 64,
                    base + 192,
                    probability.iterator,
                    pv_done,
                    True,
                    query_columns,
                    b_mn_major=True,
                )
                if prims.elect_sync():
                    prims.tcgen05_commit(value_free, group=prims.CTAGroup.CTA_1)
            mbar_wait(pv_done, step & 1)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        for query in cutlass.range_constexpr(query_columns):
            total = partials[query]
            for shift in cutlass.range_constexpr(5):
                total = total + cute.arch.shuffle_sync_bfly(total, 1 << shift)
            if lane == 0:
                scratch[warp, query] = total
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        if thread < query_columns:
            total = scratch[0, thread] + scratch[1, thread]
            total = total + scratch[2, thread]
            total = total + scratch[3, thread]
            state[1, thread] = total
            safe_total = total
            if total == 0:
                safe_total = cutlass.Float32(1)
            state[3, thread] = safe_total
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        result = load_query_panel(output_address, query_columns)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        for query in cutlass.range_constexpr(query_columns):
            if cutlass.const_expr(query < query_rows):
                store_output_element(
                    output,
                    tile,
                    query,
                    thread,
                    cutlass.BFloat16(result[query] / state[3, query]),
                    heads,
                    group,
                    io_layouts,
                )
        if cutlass.const_expr(lse is not None):
            assert lse is not None
            if thread < query_rows:
                lse_maximum = state[0, thread]
                if cutlass.const_expr(base2_scores):
                    lse_maximum = lse_maximum * cutlass.Float32(math.log(2))
                store_lse_element(
                    lse,
                    tile,
                    thread,
                    lse_maximum + cute.math.log(state[1, thread], fastmath=fast_math),
                    heads,
                    group,
                    io_layouts,
                )
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    elif warp < 8:
        if cutlass.const_expr(scoped_registers):
            cute.arch.setmaxregister_decrease(64)
        row = (warp - 4) * 32 + lane
        dimension = (warp - 4) * 32
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
        if cutlass.const_expr(scoped_registers):
            # Every warp in the third warpgroup must donate before its
            # individual control role. Warps 10 and 11 do no further work.
            cute.arch.setmaxregister_decrease(32)
        if warp == 8:
            if lane == 0:
                for step in cutlass.range(count):
                    ring = step & 1
                    mbar_wait(raw_free + ring, (step // 2 & 1) ^ 1)
                    logical_page, _logical_key_base = attention_page_coordinates(
                        first_page, last_page, step, reflected_dtype
                    )
                    physical = cutlass.Int32(pages[request, logical_page])
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
        elif warp == 9:
            for step in cutlass.range(count):
                mbar_wait(score_free, (step & 1) ^ 1)
                mbar_wait(key_ready, step & 1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                mma_128xnx128(
                    base, base + 128, shared_q.iterator, qk_done, False, query_columns
                )
                if prims.elect_sync():
                    prims.tcgen05_commit(key_free, group=prims.CTAGroup.CTA_1)
    cute.arch.barrier()
    if warp == 0:
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        prims.tcgen05_dealloc(
            cutlass.inttoptr(base, 6, cutlass.Int32), 256, group=prims.CTAGroup.CTA_1
        )
