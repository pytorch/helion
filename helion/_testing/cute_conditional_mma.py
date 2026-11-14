"""Offline-visible shared-A descriptor and serial completion observer.

Inputs are already BF16. This tests neither a decoder nor full attention.
The original QK and PV logical orientations both remain observable in outputs.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from helion._compiler.cute._conditional_attention_mma import conditional_mma_ss
from helion._compiler.cute._paged_attention_partial_compute import load_query_panel
from helion._compiler.cute._paged_attention_partial_compute import store_query_panel
from helion._compiler.cute.device.sync import mbar_wait


@cute.kernel
def observe_conditional_mma(
    keys: cute.Tensor,
    values: cute.Tensor,
    queries: cute.Tensor,
    probabilities: cute.Tensor,
    scores_out: cute.Tensor,
    output_out: cute.Tensor,
    columns: cutlass.Constexpr,
    key_steps: cutlass.Constexpr,
    value_steps: cutlass.Constexpr,
    query_steps: cutlass.Constexpr,
    probability_steps: cutlass.Constexpr,
) -> None:
    thread = cutlass.Int32(cute.arch.thread_idx()[0])
    warp = thread // 32
    key = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.BFloat16, 16384, alignment=1024),
        cute.make_layout((16384,)),
    )
    value = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.BFloat16, 16384, alignment=1024),
        cute.make_layout((16384,)),
    )
    query = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.BFloat16, 128 * columns, alignment=1024),
        cute.make_layout((128 * columns,)),
    )
    probability = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.BFloat16, 128 * columns, alignment=16),
        cute.make_layout((128 * columns,)),
    )
    slot = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=4)
    done = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=8)
    if thread == 0:
        prims.mbarrier_init(done, 1)
        prims.fence_mbarrier_init()
    for channel in cutlass.range_constexpr(128):
        offset = thread * 64 + ((channel % 64) ^ ((thread % 8) * 8))
        offset = offset + 8192 * (channel // 64)
        key[offset] = keys[thread, channel]
        value[offset] = values[thread, channel]
    for row in cutlass.range_constexpr(columns):
        offset = row * 64 + ((thread % 64) ^ ((row % 8) * 8))
        offset = offset + 64 * columns * (thread // 64)
        query[offset] = queries[row, thread]
        probability[(row // 8) * 1024 + thread * 8 + row % 8] = probabilities[
            row, thread
        ]
    if warp == 0:
        prims.tcgen05_alloc(slot, 128, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    base = cute.make_tensor(slot, cute.make_layout((1,)))[0]
    output_address = (warp * 32 << 16) | (base + 64)
    zeros = cute.make_rmem_tensor((columns,), cutlass.Float32)
    zeros.fill(cutlass.Float32(0))
    store_query_panel(output_address, zeros, columns)
    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if warp == 0:
        a = prims.Tcgen05SmemDesc.build(
            cutlass.Int32(key.iterator.toint()),
            leading_byte_offset=16,
            stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        b = prims.Tcgen05SmemDesc.build(
            cutlass.Int32(query.iterator.toint()),
            leading_byte_offset=16,
            stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        conditional_mma_ss(
            a, b, base, done, key_steps, query_steps, columns, False, False, False
        )
    mbar_wait(done, cutlass.Int32(0))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    # Every waiter retires phase0 before the issuer can advance phase1.
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    scores = load_query_panel((warp * 32 << 16) | base, columns)
    for row in cutlass.range_constexpr(columns):
        scores_out[thread, row] = scores[row]
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if warp == 0:
        a = prims.Tcgen05SmemDesc.build(
            cutlass.Int32(value.iterator.toint()),
            leading_byte_offset=16384,
            stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        b = prims.Tcgen05SmemDesc.build(
            cutlass.Int32(probability.iterator.toint()),
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
    output = load_query_panel(output_address, columns)
    for row in cutlass.range_constexpr(columns):
        output_out[thread, row] = output[row]
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier(barrier_id=0, number_of_threads=128)
    if warp == 0:
        prims.tcgen05_dealloc(
            cutlass.inttoptr(base, 6, cutlass.Float32),
            128,
            group=prims.CTAGroup.CTA_1,
        )
    if thread == 0:
        prims.mbarrier_inval(done)


@cute.jit
def launch_conditional_mma(
    keys: cute.Tensor,
    values: cute.Tensor,
    queries: cute.Tensor,
    probabilities: cute.Tensor,
    scores_out: cute.Tensor,
    output_out: cute.Tensor,
    stream: object,
    columns: cutlass.Constexpr,
    key_steps: cutlass.Constexpr,
    value_steps: cutlass.Constexpr,
    query_steps: cutlass.Constexpr,
    probability_steps: cutlass.Constexpr,
) -> None:
    observe_conditional_mma(
        keys,
        values,
        queries,
        probabilities,
        scores_out,
        output_out,
        columns,
        key_steps,
        value_steps,
        query_steps,
        probability_steps,
    ).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)
