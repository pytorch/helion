# Runtime support for the alternating-warpgroup flash-attention family (fa4_alt).
#
# NOTE: deliberately NO ``from __future__ import annotations`` -- it stringifies
# annotations, which breaks ``@cute.struct`` field-type resolution. See
# ``_flash_runtime.py`` for the same constraint; pyproject.toml exempts both
# modules from the forced future import (I002).
#
# The family processes one 128-row query tile per persistent work item. Two
# softmax warpgroups own alternating KV steps (A: even, B: odd) on their own
# score and probability buffers, a QK-issue warp and a PV-issue warp run the
# two MMA streams independently, and the finished item's dead Q stage stages
# the output tile for the TMA store. The helpers here are the pieces that
# differ from the fa4 body: the shared-storage struct, the whole-row score read
# that releases the score buffer after a single read, and the exp/convert/store
# pass that runs from the held registers.

from typing import Any
from typing import cast

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32

from . import _flash_runtime as _rt


def flash_fa4_alt_shared_storage(
    head_dim: int,
    k_stage: int,
    v_stage: int,
    dtype: object = cutlass.BFloat16,
) -> type:
    """SharedStorage of the alternating-warpgroup family.

    Q is double buffered across work items (the next item's tile is prefetched
    mid-item), K and V have separate rings (the QK stream runs two tiles ahead
    of the PV stream, so K needs the deeper ring), and there is no separate
    output stage: the correction warps stage O into the finished item's dead Q
    stage and the store warp releases that stage to the loader afterwards.
    ``sScale`` carries one alpha slot per softmax warpgroup (the handoff after
    the P-buffer wait proves the previous slot was consumed), ``sM`` the running
    max handed between the warpgroups, ``sL`` the even warpgroup's partial row
    sum handed to the odd one once per item (rewritten only after the odd
    warpgroup's ``l_consumed`` arrival for the previous item: with one KV pair
    per item nothing else orders the even warpgroup's next item behind the odd
    warpgroup's end of the current one). The odd warpgroup's alpha slot also
    carries the item's row sum to the correction warps; the next item's first
    alpha is written only after their ``rowsum_consumed`` arrival, since the
    PV chain that orders the in-item alpha handoffs does not cover that write.
    """

    @cute.struct
    class AltSharedStorage:
        q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
        k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * k_stage]
        v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * v_stage]
        # tcgen05.commit (cnt 1): S_x full after QK, PV done per P buffer, O full.
        s_full_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        pv_done_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        o_full_mbar: cute.struct.MemRange[cutlass.Int64, 1]
        # One elected arrival per softmax warp (cnt 4): S_x read into registers.
        s_empty_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        # Staged-P publication per P buffer: pfor = softmax(P[0:96]) + correction
        # (cnt 2*128); pfor2 = softmax(P[96:128]) (cnt 128).
        pfor_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        pfor2_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        # correction -> store warp: the staged output tile is in the dead Q stage.
        corr_epi_full_mbar: cute.struct.MemRange[cutlass.Int64, 1]
        # odd warpgroup -> even warpgroup (cnt 128): the partial row sum of the
        # previous work item was read, so ``sL`` may be rewritten and the named
        # barrier of the handoff cannot see two even-side arrivals per sync.
        l_consumed_mbar: cute.struct.MemRange[cutlass.Int64, 1]
        rowsum_consumed_mbar: cute.struct.MemRange[cutlass.Int64, 1]
        tmem_dealloc_mbar: cute.struct.MemRange[cutlass.Int64, 1]
        tmem_holding_buf: cutlass.Int32
        sScale: cute.struct.MemRange[cutlass.Float32, 2 * 128]
        sM: cute.struct.MemRange[cutlass.Float32, 2 * 128]
        sL: cute.struct.MemRange[cutlass.Float32, 128]
        sQ: cute.struct.Align[cute.struct.MemRange[dtype, 2 * 128 * head_dim], 1024]
        sK: cute.struct.Align[
            cute.struct.MemRange[dtype, k_stage * 128 * head_dim], 1024
        ]
        sV: cute.struct.Align[
            cute.struct.MemRange[dtype, v_stage * 128 * head_dim], 1024
        ]

    return AltSharedStorage


def fa4_alt_rowmax_hold(
    tiled_ld: object,
    tLDtS: cute.Tensor,
    tLDcS: cute.Tensor,
    ld_chunks: int,
    s_empty_ptr: object,
) -> tuple[Float32, list[cute.Tensor]]:
    """Read the whole score row once, release the score buffer, fold the max.

    All ``ld_chunks`` t2r loads are issued back to back and pinned so ptxas
    cannot sink them below the folds; one fence drains them, then one elected
    thread per warp arrives on ``s_empty_ptr`` (the QK warp may overwrite the
    buffer). The fragments stay live for ``fa4_alt_exp_pass_held``. The fold
    uses the fa4 balanced tree from ``-inf``; folding the previous running max
    in afterwards is exact (fmax is exact and associative), so the result equals
    the fa4 body's ``fa4_disc_rowmax_balanced`` seeded with that max.
    """
    ld_shape = cast("Any", tLDcS[None, 0, None, None]).shape
    frgs = [cute.make_rmem_tensor(ld_shape, cutlass.Float32) for _ in range(ld_chunks)]
    for ci in range(ld_chunks):
        cute.copy(tiled_ld, tLDtS[None, ci, None, None], frgs[ci])
        _rt._disc_pin_frag(frgs[ci])
    cute.arch.fence_view_async_tmem_load()
    cute.arch.sync_warp()
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(s_empty_ptr)
    row_max = cutlass.Float32(-cutlass.Float32.inf)
    for ci in range(ld_chunks):
        row_max = _rt._fmax_reduce_chunk_balanced(frgs[ci], row_max)
    return row_max, frgs


def fa4_alt_exp_pass_held(
    frgs: list[cute.Tensor],
    tiled_st: object,
    tSTtP: cute.Tensor,
    tSTcS: cute.Tensor,
    scale: Float32,
    minus_max_scale: Float32,
    e2e_freq: int,
    e2e_res: int,
    e2e_offset: int,
    pfor_ptr: object,
    pfor2_ptr: object,
    p_store_split: int,
    p_store_chunks: int,
    io_dtype: object,
    p_empty_ptr: object,
    p_empty_phase: object,
    handoff_enabled: object,
    scale_t: cute.Tensor,
    wg: int,
    tidx: object,
    alpha: Float32,
    bar_id: object,
    pair_batch: int = 1,
    emu_batch: int = 1,
    degree2: bool = False,
    degree1: bool = False,
    wait_hint: int = 10_000_000,
) -> Float32:
    """The fa4 chunked exp/convert/store pass from the held score fragments.

    Same per-chunk numerics as ``fa4_disc_exp_convert_store_pipe`` (scale and
    subtract, exp2 with the pipe split, fp16 convert, r2t store into the P
    buffer, row sum, the 3/4 + 1/4 staged-P arrivals on ``pfor``/``pfor2``),
    without TMEM loads. Before the first store the pass waits on ``p_empty_ptr``
    (this warpgroup's previous PV completed, so the P buffer is free). That PV
    needed the correction warps' arrival for the previous step of this
    warpgroup, which implies they consumed the previous alpha: the handoff of
    this step's alpha (``scale_t[wg, tidx]`` then ``bar.arrive`` on
    ``bar_id``) happens right after that wait, so the single slot is free and
    the named barrier cannot see two arrivals before one sync.
    """
    p_sum = cutlass.Float32(0.0)
    _rt.mbar_spin_wait(p_empty_ptr, p_empty_phase, wait_hint)

    def _handoff() -> None:
        scale_t[wg, tidx] = alpha
        _rt.named_barrier_arrive_unaligned(bar_id, 64)

    def _no_handoff() -> None:
        pass

    cutlass.cutlass_dsl.if_generate(handoff_enabled, _handoff, _no_handoff, [], [])
    for ci in range(p_store_chunks):
        cur = frgs[ci]
        _rt._disc_chunk_exp(
            cur,
            scale,
            minus_max_scale,
            e2e_freq,
            e2e_res,
            e2e_offset,
            ci >= p_store_chunks - 1,
            pair_batch,
            emu_batch,
            degree2,
            degree1,
        )
        _rt._disc_chunk_convert_store(cur, tiled_st, tSTtP, tSTcS, ci, io_dtype)
        p_sum = p_sum + _rt._disc_chunk_rowsum(cur)
        _rt._disc_chunk_release(
            ci, p_store_split, p_store_chunks, pfor_ptr, pfor2_ptr, None, None, None
        )
    cute.arch.fence_view_async_tmem_store()
    _rt.mbarrier_arrive(pfor2_ptr)
    return p_sum
