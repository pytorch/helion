"""Arithmetic for attention with an explicit per-key denominator carry.

Each key lane keeps one FP32 partial per query. The source graph contains
this two-dimensional state and reduces it once after the loop.

PTX permits M128/N8-multiple dense BF16 MMA for CTA-group1 (including TMEM A):
CUDA 13.0 PTX ISA, section 9.7.16.2.1, Table 39; section 9.7.16.10 permits A
in tensor or shared memory. This is wider than the CUTLASS C++ TS atom gate.
https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tcgen05-matrix-shape
The 8-bit transposed-B restriction does not apply to the decoded BF16 inputs.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as prims

from ._flash_row_mma_runtime import pack_bf16x2
from ._paged_attention_compute import logical_attention_mask
from .device.ptx import sts_v4_b32


@cute.jit
def load_query_panel(
    address: cutlass.Int32, query_columns: cutlass.Constexpr
) -> cute.Tensor | cutlass.Array:
    """Read a native 8/32-column panel or the split 32+8-column panel."""
    if cutlass.const_expr(query_columns == 40):
        lo = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(address, 6, cutlass.Float32), num=32
        )
        hi = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(address + 32, 6, cutlass.Float32), num=8
        )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        value = cute.make_rmem_tensor((40,), cutlass.Float32)
        for index in cutlass.range_constexpr(32):
            value[index] = lo[index]
        for index in cutlass.range_constexpr(8):
            value[index + 32] = hi[index]
    else:
        value = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(address, 6, cutlass.Float32), num=query_columns
        )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
    return value


@cute.jit
def store_query_panel(
    address: cutlass.Int32,
    value: cute.Tensor | cutlass.Array,
    query_columns: cutlass.Constexpr,
) -> None:
    """Use separate physical arrays for the non-power-of-two 40-column panel."""
    if cutlass.const_expr(query_columns == 40):
        lo = cutlass.Array(cutlass.Float32, 32, alignment=16)
        hi = cutlass.Array(cutlass.Float32, 8, alignment=16)
        for index in cutlass.range_constexpr(32):
            lo[index] = value[index]
        for index in cutlass.range_constexpr(8):
            hi[index] = value[index + 32]
        prims.tcgen05_st(
            "32x32b", cutlass.inttoptr(address, 6, cutlass.Float32), lo[0:32]
        )
        prims.tcgen05_st(
            "32x32b", cutlass.inttoptr(address + 32, 6, cutlass.Float32), hi[0:8]
        )
    else:
        packed = cutlass.Array(cutlass.Float32, query_columns, alignment=16)
        for index in cutlass.range_constexpr(query_columns):
            packed[index] = value[index]
        prims.tcgen05_st(
            "32x32b",
            cutlass.inttoptr(address, 6, cutlass.Float32),
            packed[0:query_columns],
        )


@cute.jit
def mma_128xnx128(
    a_base: cutlass.Int32,
    d_base: cutlass.Int32,
    b_pointer: cute.Pointer,
    done: cute.Pointer,
    accumulate: cutlass.Constexpr,
    query_columns: cutlass.Constexpr,
    b_mn_major: cutlass.Constexpr = False,
) -> None:
    desc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        n_dim=query_columns,
        m_dim=128,
        a_major=0,
        b_major=1 if b_mn_major else 0,
    )
    if cutlass.const_expr(b_mn_major):
        # MN-major without swizzling stores 16-byte N chunks per K row.
        # LBO spans eight K rows; SBO spans eight N columns. A K16 step is
        # therefore 2*LBO (256 bytes), not the swizzled layout's 2*SBO.
        bd = prims.Tcgen05SmemDesc.build(
            cutlass.Int32(b_pointer.toint()),
            leading_byte_offset=128,
            stride_byte_offset=2048,
            layout=prims.Tcgen05SmemSwizzle.NONE,
        )
    else:
        bd = prims.Tcgen05SmemDesc.build(
            cutlass.Int32(b_pointer.toint()),
            leading_byte_offset=16,
            stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
    acc = cutlass.inttoptr(d_base, 6, cutlass.Float32)
    for step in cutlass.range_constexpr(8):
        offset = (
            step * 256
            if b_mn_major
            else (step % 4) * 32 + (step // 4) * (query_columns * 128)
        )
        if prims.elect_sync():
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                acc,
                cutlass.inttoptr(a_base + step * 8, 6, cutlass.Int32),
                bd.advance_start_address(offset),
                desc,
                accumulate or step != 0,
            )
    if prims.elect_sync():
        prims.tcgen05_commit(done, group=prims.CTAGroup.CTA_1)


@cute.jit
def _store_probability_and_denominator(
    values: cute.Tensor,
    probability: cute.Tensor,
    state: cute.Tensor,
    partials: cute.Tensor,
    thread: cutlass.Int32,
    query_columns: cutlass.Constexpr,
    fast_math: cutlass.Constexpr,
    rescale_denominator: cutlass.Constexpr,
) -> None:
    # Each key lane writes eight adjacent queries with one aligned 16-byte
    # store. The complete eight-query groups follow the selected panel width.
    for group in cutlass.range_constexpr(query_columns // 8):
        packed = cute.make_rmem_tensor((4,), cutlass.Uint32)
        for pair in cutlass.range_constexpr(4):
            values_pair = cute.make_rmem_tensor((2,), cutlass.Float32)
            for element in cutlass.range_constexpr(2):
                query = group * 8 + pair * 2 + element
                safe_maximum = cutlass.Float32(state[0, query])
                if cutlass.const_expr(rescale_denominator):
                    if safe_maximum == float("-inf"):
                        safe_maximum = cutlass.Float32(0.0)
                value = cute.math.exp(values[query] - safe_maximum, fastmath=fast_math)
                values_pair[element] = value
                if cutlass.const_expr(rescale_denominator):
                    partials[query] = (
                        cutlass.Float32(partials[query])
                        * cutlass.Float32(state[2, query])
                        + value
                    )
                else:
                    partials[query] = cutlass.Float32(partials[query]) + value
            packed[pair] = pack_bf16x2(values_pair[0], values_pair[1])
        offset = group * 1024 + thread * 8
        address = cutlass.Int32((probability.iterator + offset).toint())
        sts_v4_b32(address, packed[0], packed[1], packed[2], packed[3])


@cute.jit
def online_softmax_per_key(
    score_base: cutlass.Int32,
    probability: cute.Tensor,
    state: cute.Tensor,
    scratch: cute.Tensor,
    thread: cutlass.Int32,
    logical_key_base: cutlass.Int64,
    length: cutlass.Int32,
    query_length: cutlass.Constexpr,
    group_size: cutlass.Constexpr,
    window: cutlass.Constexpr,
    score_scale: cutlass.Float32,
    barrier_id: cutlass.Constexpr,
    partials: cute.Tensor,
    fast_math: cutlass.Constexpr,
    score_free: cute.Pointer,
    coordinate_dtype: cutlass.Constexpr,
    query_columns: cutlass.Constexpr = 40,
) -> cutlass.Boolean:
    """Four full warps consume S[key,query] and produce P[query,key].

    State rows 0 and 2 hold the running maximum and accumulator rescale.
    Row 2 is refreshed only when the returned vote is true; callers must not
    read its stale alpha on the unchanged-maximum path.
    Each key lane retains its FP32 denominator in partials; the pipeline
    reduces those partials only after the loop, then fills state rows 1 and 3.
    Probability has the MN_INTER layout expected by the PV tensor-core B
    descriptor. Scratch is [4,query_columns] FP32 and is reusable after the
    final named barrier. After retiring the score loads, each compute warp
    arrives once on score_free (whose arrival count is four), allowing the
    issuer to overwrite S. The returned vote is uniform across all 128 compute
    threads: when false, alpha is one and the owner must skip its accumulator
    rescale and associated barrier. The owner retains P's lifetime.
    """
    assert cutlass.const_expr(query_columns % 8 == 0)
    warp = thread // 32
    lane = thread % 32
    address = (warp * 32 << 16) | score_base
    scores = load_query_panel(address, query_columns)
    # All score loads retire before the issuer may reuse their TMEM panel.
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.sync_warp()
    if lane == 0:
        prims.mbarrier_arrive(score_free)
    values = cute.make_rmem_tensor((query_columns,), cutlass.Float32)
    needs_update = cutlass.Boolean(False)
    for query in cutlass.range_constexpr(query_columns):
        visible = logical_attention_mask(
            logical_key_base,
            thread,
            length,
            query,
            query_length,
            group_size,
            window,
            coordinate_dtype,
        )
        score = scores[query] * score_scale
        if not visible:
            score = cutlass.Float32(float("-inf"))
        values[query] = score
        previous = cutlass.Float32(state[0, query])
        # With a finite unchanged maximum alpha is exactly exp(0) == 1.
        # NaN scores and any nonfinite previous maximum retain the full path.
        needs_update = (
            needs_update
            | (score > previous)
            | (score != score)
            | (previous != previous)
            | (previous == float("-inf"))
            | (previous == float("inf"))
        )
    maximum_changed = prims.barrier_cta_red(
        needs_update, barrier_id, "or", thread_count=128
    )
    if maximum_changed:
        for query in cutlass.range_constexpr(query_columns):
            maximum = cute.arch.warp_redux_sync(values[query], "fmax", nan=True)
            if lane == 0:
                scratch[warp, query] = maximum
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
        if thread < query_columns:
            maximum = cutlass.Float32(state[0, thread])
            for w in cutlass.range_constexpr(4):
                maximum = cute.math.max(
                    maximum, cutlass.Float32(scratch[w, thread]), propagate_nan=True
                )
            safe_maximum = maximum
            if maximum == float("-inf"):
                safe_maximum = cutlass.Float32(0.0)
            state[2, thread] = cute.math.exp(
                cutlass.Float32(state[0, thread]) - safe_maximum, fastmath=fast_math
            )
            state[0, thread] = maximum
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
    # The uniform vote proves every maximum is finite and unchanged, so
    # alpha=exp(old_max-old_max)=1 and no state writes or alpha reads are needed.
    if maximum_changed:
        _store_probability_and_denominator(
            values,
            probability,
            state,
            partials,
            thread,
            query_columns,
            fast_math,
            rescale_denominator=True,
        )
    else:
        _store_probability_and_denominator(
            values,
            probability,
            state,
            partials,
            thread,
            query_columns,
            fast_math,
            rescale_denominator=False,
        )
    cute.arch.fence_proxy(kind="async.shared", space="cta")
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
    return maximum_changed
