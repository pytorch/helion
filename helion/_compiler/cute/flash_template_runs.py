"""Template runs shared by the fused flash-attention FORWARD emitters.

Role: special-purpose emitter.  Leaf helper module for ``cute_flash`` (the FA4, warp-specialized and
legacy bodies) and ``cute_flash_gated``: string renderers for the runs of
emitted CuTe DSL text that two or more forward emitter sites repeated
verbatim.  Every renderer returns exactly the text its sites emitted, at the
sites' indentation (an ``indent`` parameter only where the sites differ), so
the renders are byte-identical to the inline forms they replace.  It imports
nothing from the emitters, so every forward module can import it at module
level.
"""

from __future__ import annotations

import textwrap

# --------------------------------------------------------------------------
# Warp-specialized producer (warp 0): QK and PV steps
# --------------------------------------------------------------------------


def _ws_qk_mma(s_buf: str, indent: str) -> str:
    """The QK MMA of the current K tile into the S buffer ``s_buf``: the
    ``flash_nk`` K-chunk gemms of the warp-specialized producer, accumulating
    from the second chunk on."""
    return textwrap.indent(
        f"""for flash_kp in cutlass.range(flash_nk, unroll_full=True):
    _flash_qk_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_kp != 0)
    cute.gemm(_flash_qk_mma, {s_buf}, tSrQ[None, None, flash_kp, flash_q_full.index],
              tSrK[None, None, flash_kp, flash_k_full.index], {s_buf})""",
        indent,
    )


def flash_ws_qk_tile(s_buf: str) -> str:
    """One QK tile into a fixed S buffer (8-space indented, under
    ``if warp_idx == 0:``): wait for the K tile, acquire the S stage, the QK
    MMA, commit the stage and release K.  The warp-specialized prologues
    (QK(0) -> S0) and the one-pass producer (QK(0) -> S0, QK(1) -> S1)."""
    return f"""        flash_k_full = flash_k_cons.wait_and_advance()
        flash_s_handle = flash_mma_s_prod.acquire_and_advance()
{_ws_qk_mma(s_buf, "        ")}
        flash_s_handle.commit()
        flash_k_full.release()"""


def flash_ws_qk_step(indent: str) -> str:
    """One QK tile into S[flash_qk_idx % 2] and the parity toggle: the
    steady-state ahead-QK of the warp-specialized producer and the gated
    producer's prologue QK, which differ only in indentation."""
    return textwrap.indent(
        f"""flash_k_full = flash_k_cons.wait_and_advance()
flash_s_handle = flash_mma_s_prod.acquire_and_advance()
if (flash_qk_idx % 2) == 0:
{_ws_qk_mma("tStS0", "    ")}
else:
{_ws_qk_mma("tStS1", "    ")}
flash_s_handle.commit()
flash_k_full.release()
flash_qk_idx = (flash_qk_idx + 1) % 2""",
        indent,
    )


def _ws_pv_mma(p_buf: str, indent: str) -> str:
    """The PV MMA of the current V tile reading the P fragment ``p_buf``: the
    ``flash_nk2`` K-chunk gemms into the single O accumulator, accumulating
    from the second chunk on and, from the second KV tile on, over the O the
    previous tiles left."""
    return textwrap.indent(
        f"""for flash_kp in cutlass.range(flash_nk2, unroll_full=True):
    _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_first_acc | (flash_kp != 0))
    cute.gemm(_flash_pv_mma, tOtO, {p_buf}[None, None, flash_kp, 0],
              tOrV[None, None, flash_kp, flash_v_full.index], tOtO)""",
        indent,
    )


def flash_ws_pv_step(indent: str) -> str:
    """One PV tile reading P[flash_p_idx % 2] (the parity the softmax
    consumer published) into O, accumulating once O holds a tile: the
    steady-state PV of the warp-specialized producer and the gated
    producer's, which differ only in indentation."""
    return textwrap.indent(
        f"""flash_first_acc = flash_o_started
if (flash_p_idx % 2) == 0:
{_ws_pv_mma("tOrP0", "    ")}
else:
{_ws_pv_mma("tOrP1", "    ")}""",
        indent,
    )


# --------------------------------------------------------------------------
# Softmax consumers: row reductions and the online-softmax step
# --------------------------------------------------------------------------


def flash_row_reduce_max(packed_reduce: bool, tensor: str, init: str) -> str:
    """The row max of the score fragment ``tensor`` seeded with ``init``: the
    packed f32x2 reduction or the DSL's ``reduce``."""
    if packed_reduce:
        return f"_helion_flash_rt.fmax_reduce_packed({tensor}, {init})"
    return f"{tensor}.load().reduce(cute.ReductionOp.MAX, {init}, 0)"


def flash_row_reduce_sum(packed_reduce: bool, tensor: str) -> str:
    """The row sum of the probability fragment ``tensor`` (packed or plain)."""
    if packed_reduce:
        return f"_helion_flash_rt.fadd_reduce_packed({tensor})"
    return f"{tensor}.load().reduce(cute.ReductionOp.ADD, 0.0, 0)"


def flash_online_softmax_step(
    indent: str,
    *,
    packed_reduce: bool,
    alpha_pre: str,
    exp2_block: str,
    io_dtype: str,
    alpha_post: str,
) -> str:
    """The online-softmax step over the score tile in ``tLDrS``: the running
    row max (the previous one kept in ``flash_old_row_max`` for the alpha)
    and its ``-inf``-safe copy, the caller's ``alpha_pre`` lines (the
    rescale pin or the minus-max scale), the ``exp2_block`` sweep, the row
    sum, the P cast into ``tSTrS_e``, the caller's ``alpha_post`` (empty when
    the pin computed the alpha) and the row-sum update.  The three blocks are
    pre-indented by the caller.  The legacy body (4-space) and the
    warp-specialized consumer step (12-space)."""
    return f"""{indent}flash_old_row_max = flash_row_max
{indent}flash_row_max = {flash_row_reduce_max(packed_reduce, "tLDrS", "flash_row_max")}
{indent}flash_row_max_safe = flash_row_max
{indent}if flash_row_max == -cutlass.Float32.inf:
{indent}    flash_row_max_safe = cutlass.Float32(0.0)
{alpha_pre}

{exp2_block}
{indent}flash_p_sum = {flash_row_reduce_sum(packed_reduce, "tLDrS")}

{indent}tSTrS = cute.make_rmem_tensor(tSTcS.shape, cutlass.Float32)
{indent}tSTrS_e = cute.make_tensor(
{indent}    cute.recast_ptr(tSTrS.iterator, dtype={io_dtype}), tLDrS.layout)
{indent}tSTrS_e.store(tLDrS.load().to({io_dtype}))

{alpha_post}
{indent}flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum"""


# --------------------------------------------------------------------------
# Warp-specialized softmax consumer: the step frame
# --------------------------------------------------------------------------


def flash_ws_step_head(score_transform: str) -> str:
    """The head of one warp-specialized consumer step (12-space): wait for
    the S stage, load the score tile from the TMEM buffer of the stage's
    parity into ``tLDrS``, fence, and the caller's score transform.  The
    single-row and the two-row consumer steps."""
    return f"""            flash_s_full = flash_mma_s_cons.wait_and_advance()
            tLDrS = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
            if (flash_s_idx % 2) == 0:
                cute.copy(flash_tiled_ld0, tLDtS0, tLDrS)
            else:
                cute.copy(flash_tiled_ld1, tLDtS1, tLDrS)
            cute.arch.fence_view_async_tmem_load(){score_transform}"""


def flash_ws_step_tail(rescale_block: str) -> str:
    """The tail of one warp-specialized consumer step (12-space): store the
    P tile into the TMEM buffer of the stage's parity, fence, the caller's O
    rescale block, publish P to the MMA warp, release S and toggle the
    parity.  The single-row and the two-row consumer steps."""
    return f"""            if (flash_s_idx % 2) == 0:
                cute.copy(flash_tiled_st0, tSTrS, tSTtS0)
            else:
                cute.copy(flash_tiled_st1, tSTrS, tSTtS1)
            cute.arch.fence_view_async_tmem_store()

{rescale_block}
            flash_p_handle = flash_p_ready_prod.acquire_and_advance()
            flash_p_handle.commit()
            flash_s_full.release()
            flash_o_started = cutlass.Boolean(True)
            flash_s_idx = (flash_s_idx + 1) % 2"""


# --------------------------------------------------------------------------
# FA4 MMA warp: the staged-P PV of one stage
# --------------------------------------------------------------------------


def fa4_staged_pv(stage: str, indent: str, wait_hint: int) -> str:
    """The PV of ``stage`` over a staged P tile in the ``cute.gemm`` form of
    the FA4 MMA warp: the first ``flash_pv_split`` K-chunks once the
    softmax's first P-arrive barrier (``flash_pfor``) completes, the rest
    once the second (``flash_pfor2``) does; ``flash_O_acc{stage}`` is False
    for the first KV tile.  The steady loop (12-space) and the epilogue
    (8-space) of both stages."""
    return textwrap.indent(
        f"""_helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + {stage}, flash_pfor_phase, {wait_hint})
_helion_flash_rt.tcgen05_fence_after_thread_sync()
for flash_kp in cutlass.range_constexpr(flash_pv_split):
    _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, flash_O_acc{stage} | (flash_kp != 0))
    cute.gemm(_flash_pv_mma, tOtO{stage}, tOrP{stage}[None, None, flash_kp, 0],
              tOrV[None, None, flash_kp, flash_v_full.index], tOtO{stage})
_helion_flash_rt.mbar_spin_wait(flash_pfor2_ptr + {stage}, flash_pfor_phase, {wait_hint})
_helion_flash_rt.tcgen05_fence_after_thread_sync()
for flash_kp in cutlass.range_constexpr(flash_pv_split, flash_nk2):
    _flash_pv_mma.set(cute_tcgen05_flash.Field.ACCUMULATE, True)
    cute.gemm(_flash_pv_mma, tOtO{stage}, tOrP{stage}[None, None, flash_kp, 0],
              tOrV[None, None, flash_kp, flash_v_full.index], tOtO{stage})""",
        indent,
    )


# --------------------------------------------------------------------------
# CTA-wide setup: TMA and TMEM partitions
# --------------------------------------------------------------------------

# The per-CTA TMA partitions of Q, K and V over their tiled gmem views (``gQ``
# / ``gK`` / ``gV``) and smem rings, grouping the three spatial modes so the
# ring stage stays the trailing index.  The FA4 body setup and the gated
# setup.
FLASH_QKV_TMA_PARTITIONS = """tSgQ = flash_qkt.partition_A(gQ)
tSgK = flash_qkt.partition_B(gK)
tOgV = flash_pvt.partition_B(gV)
tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(
    _flash_tma_q, 0, cute.make_layout(1),
    cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(
    _flash_tma_k, 0, cute.make_layout(1),
    cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
tVsV, tVgV_dkl = cute_cpasync_flash.tma_partition(
    _flash_tma_v, 0, cute.make_layout(1),
    cute.group_modes(sV, 0, 3), cute.group_modes(tOgV, 0, 3))"""


def flash_ws_s_load_partitions(ld_atom: str) -> str:
    """The per-stage t2r copies of the double-buffered S (``tStS0`` /
    ``tStS1``) and this thread's load partitions, with the coordinate
    partition ``tLDcS`` over stage 0.  The warp-specialized body setup and
    the gated setup."""
    return f"""flash_ld_atom = cute.make_copy_atom({ld_atom}, cutlass.Float32)
flash_tiled_ld0 = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS0)
flash_tiled_ld1 = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS1)
flash_thr_ld0 = flash_tiled_ld0.get_slice(flash_local_tidx)
flash_thr_ld1 = flash_tiled_ld1.get_slice(flash_local_tidx)
tLDtS0 = flash_thr_ld0.partition_S(tStS0)
tLDtS1 = flash_thr_ld1.partition_S(tStS1)
tLDcS = flash_thr_ld0.partition_D(tScS)"""


def flash_ws_p_store_partitions(
    *,
    kv_cols: int,
    io_dtype: str,
    q_rows: int,
    p_base: str,
    p1_offset: int,
    st_atom: str,
) -> str:
    """The two P buffers (``kv_cols`` of ``io_dtype`` re-viewed as fp32
    columns at ``p_base`` and ``p_base + p1_offset``), their r2t copies and
    this thread's store partitions, with the coordinate partition ``tSTcS``
    over buffer 0.  The warp-specialized body setup and the gated setup."""
    return f"""flash_tilePlikeFP32 = {kv_cols} // cutlass.Float32.width * {io_dtype}.width
flash_P_layout = cute.composition(flash_s_layout, cute.make_layout(({q_rows}, flash_tilePlikeFP32)))
tStS_P0 = cute.make_tensor({p_base}, flash_P_layout)
tStS_P1 = cute.make_tensor({p_base} + {p1_offset}, flash_P_layout)
flash_tScS_P_layout = cute.composition(tScS.layout, cute.make_layout(({q_rows}, flash_tilePlikeFP32)))
tScS_P = cute.make_tensor(tScS.iterator, flash_tScS_P_layout)
flash_st_atom = cute.make_copy_atom({st_atom}, cutlass.Float32)
flash_tiled_st0 = cute_tcgen05_flash.make_tmem_copy(flash_st_atom, tStS_P0)
flash_tiled_st1 = cute_tcgen05_flash.make_tmem_copy(flash_st_atom, tStS_P1)
flash_thr_st0 = flash_tiled_st0.get_slice(flash_local_tidx)
flash_thr_st1 = flash_tiled_st1.get_slice(flash_local_tidx)
tSTtS0 = flash_thr_st0.partition_D(tStS_P0)
tSTtS1 = flash_thr_st1.partition_D(tStS_P1)
tSTcS = flash_thr_st0.partition_S(tScS_P)"""
