from __future__ import annotations

import cutlass
import cutlass.cute as cute

from .cluster_helpers import store_shared_remote_x4


@cute.jit
def _warp_sum4(
    v0: cutlass.Float32,
    v1: cutlass.Float32,
    v2: cutlass.Float32,
    v3: cutlass.Float32,
) -> tuple[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32]:
    for offset in (16, 8, 4, 2, 1):
        v0 += cute.arch.shuffle_sync_bfly(v0, offset=offset)
        v1 += cute.arch.shuffle_sync_bfly(v1, offset=offset)
        v2 += cute.arch.shuffle_sync_bfly(v2, offset=offset)
        v3 += cute.arch.shuffle_sync_bfly(v3, offset=offset)
    return v0, v1, v2, v3


@cute.jit
def _cute_grouped_reduce_cluster_sum4(
    v0: cutlass.Float32,
    v1: cutlass.Float32,
    v2: cutlass.Float32,
    v3: cutlass.Float32,
    tid: cutlass.Int32,
    buf_ptr: cute.Pointer,
    mbar: cute.Pointer,
    group_span: cutlass.Constexpr,
    cluster_n: cutlass.Constexpr,
) -> tuple[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32]:
    """Sum four independent FP32 values with one cluster transaction.

    The caller provides a 16-byte-aligned buffer of ``4 * slots`` FP32 values
    and an initialized, single-use mbarrier, and synchronizes the cluster before
    entry. One full CTA of ``group_span`` threads participates in each row.
    """
    warps = group_span // 32
    slots = warps * cluster_n
    lane = tid % 32
    warp = tid // 32
    rank = cutlass.Int32(cute.arch.block_idx_in_cluster())
    w0, w1, w2, w3 = _warp_sum4(v0, v1, v2, v3)
    if tid == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(mbar, slots * 16)
    if lane < cluster_n:
        store_shared_remote_x4(
            w0,
            w1,
            w2,
            w3,
            smem_ptr=buf_ptr + (rank * warps + warp) * 4,
            mbar_ptr=mbar,
            peer_cta_rank_in_cluster=lane,
        )
    cute.arch.mbarrier_wait(mbar, 0)
    buf = cute.make_tensor(buf_ptr, cute.make_layout((4, slots), stride=(1, 4)))
    x0 = cutlass.Float32(0)
    x1 = cutlass.Float32(0)
    x2 = cutlass.Float32(0)
    x3 = cutlass.Float32(0)
    for offset in cutlass.range_constexpr((slots + 31) // 32):
        slot = lane + offset * 32
        if slot < slots:
            fragment = cute.make_fragment_like(buf[(None, slot)])
            cute.autovec_copy(buf[(None, slot)], fragment)
            values = fragment.load()
            x0 += values[0]
            x1 += values[1]
            x2 += values[2]
            x3 += values[3]
    return _warp_sum4(x0, x1, x2, x3)
