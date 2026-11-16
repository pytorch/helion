"""Shared pieces of the fused flash-attention BACKWARD emitters.

Role: special-purpose emitter.  Leaf helper module for ``cute_flash_bwd`` (dense 1-CTA / 2-CTA),
``cute_flash_bwd_flex`` / ``_flex_2cta`` / ``_flex_pair`` (FlexAttention) and
``cute_flash_bwd_row_mma``: the autotuner key constants of the backward
surface, the text renderers the stage emitters share (hook chunk prefixes,
the LSE bias fold, chunked step regions, register splits, ``setmaxnreg``),
the dK/dV epilogue block and the CLC persistent-scheduler fragments.  It
reads no ``CompileEnvironment`` or settings and imports nothing from the
emitters, so every backward module can import it at module level.
"""

from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING
from typing import NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    from .flash_score_mod import FlexHookCode

# --------------------------------------------------------------------------
# Autotuner keys of the backward surface (``flash_surface.enable_cute_flash_bwd_search``)
# --------------------------------------------------------------------------
# The backward knob names and their ordered tuple are declared once, in the
# ``flash_knobs`` registry; the backward emitters read them from here.
from .flash_knobs import FLASH_BWD_AUX_SMEM_KEY as FLASH_BWD_AUX_SMEM_KEY
from .flash_knobs import FLASH_BWD_BIAS_FOLD_KEY as FLASH_BWD_BIAS_FOLD_KEY
from .flash_knobs import FLASH_BWD_CONFIG_KEYS as FLASH_BWD_CONFIG_KEYS
from .flash_knobs import FLASH_BWD_DS_TMEM_KEY as FLASH_BWD_DS_TMEM_KEY
from .flash_knobs import FLASH_BWD_EXP2_EMU_KEY as FLASH_BWD_EXP2_EMU_KEY
from .flash_knobs import FLASH_BWD_EXP2_F32_KEY as FLASH_BWD_EXP2_F32_KEY
from .flash_knobs import FLASH_BWD_FAMILY_KEY as FLASH_BWD_FAMILY_KEY
from .flash_knobs import FLASH_BWD_HOOK_COLS_KEY as FLASH_BWD_HOOK_COLS_KEY
from .flash_knobs import FLASH_BWD_HOOK_PIN_KEY as FLASH_BWD_HOOK_PIN_KEY
from .flash_knobs import FLASH_BWD_KV_STAGE_KEY as FLASH_BWD_KV_STAGE_KEY
from .flash_knobs import FLASH_BWD_MASK_FORM_KEY as FLASH_BWD_MASK_FORM_KEY
from .flash_knobs import FLASH_BWD_PERSISTENT_KEY as FLASH_BWD_PERSISTENT_KEY
from .flash_knobs import FLASH_BWD_REGS_KEY as FLASH_BWD_REGS_KEY
from .flash_knobs import FLASH_BWD_ROW_KV_KEY as FLASH_BWD_ROW_KV_KEY
from .flash_knobs import FLASH_BWD_ROW_Q_CHUNK_KEY as FLASH_BWD_ROW_Q_CHUNK_KEY
from .flash_knobs import FLASH_BWD_ROW_Q_SPLIT_KEY as FLASH_BWD_ROW_Q_SPLIT_KEY
from .flash_knobs import FLASH_BWD_S_FIRST_KEY as FLASH_BWD_S_FIRST_KEY
from .flash_knobs import FLASH_BWD_STATS_MMA_KEY as FLASH_BWD_STATS_MMA_KEY
from .flash_knobs import FLASH_BWD_TABLE_STEPS_KEY as FLASH_BWD_TABLE_STEPS_KEY
from .flash_knobs import FLASH_BWD_TILE_GROUP_KEY as FLASH_BWD_TILE_GROUP_KEY
from .flash_knobs import FLASH_BWD_TWO_CTA_KEY as FLASH_BWD_TWO_CTA_KEY

# --------------------------------------------------------------------------
# Text renderers shared by the stage emitters
# --------------------------------------------------------------------------


def indent_lines(text: str, indent: int) -> str:
    """Re-indent a hook block (complete lines) to ``indent`` spaces."""
    if not text.strip():
        return ""
    body = textwrap.indent(textwrap.dedent(text).rstrip("\n"), " " * indent)
    return body + "\n"


def merge_blocks(*texts: str) -> str:
    """The union of the hooks' prelude lines in first-seen order (the renderer
    emits identical lines for shared values, so exact-text dedup is exact)."""
    seen: list[str] = []
    for text in texts:
        for line in textwrap.dedent(text).splitlines():
            if line not in seen:
                seen.append(line)
    return "\n".join(seen) + ("\n" if seen else "")


# Register budgets per warp role (dQ-reduce warps / compute warpgroups / MMA +
# load warps), ``cute_flash_bwd_regs``: 128 R + 256 C + 128 L registers must
# fit the SM's 65536, i.e. R + 2 C + L <= 512. FA4's 1-CTA split is the
# default; the others trade dQ-reduce or load/MMA registers for the compute
# warps, whose score_mod / mask_mod hooks are the first to spill.
FBWD_REG_SPLITS: tuple[str, ...] = (
    # First = the default config's: the hooks' per-lane work lives in the
    # compute warps, and 152 registers there measured best on the probed
    # variants (alibi 421 -> 376 us, causal 194 -> 177, noop 280 -> 270 at 2K).
    "120/152/88",
    "152/136/88",
    "136/144/88",
    "104/160/88",
    "88/168/88",
    # The 2-CTA kernel's splits: its compute warps spill below 152 registers
    # (the merged walk's per-step state next to the exchange staging) and
    # its MMA / load warps hold six rings and the cluster descriptors
    # (FA4's 2-CTA budget of 104); its reduce warps drain half a dQ tile.
    "104/152/104",
    "136/136/104",
)


def reg_split_counts(value: object) -> tuple[int, int, int]:
    text = str(value) if value else FBWD_REG_SPLITS[0]
    if text not in FBWD_REG_SPLITS:
        text = FBWD_REG_SPLITS[0]
    reduce_regs, compute_regs, other_regs = (int(part) for part in text.split("/"))
    assert reduce_regs + 2 * compute_regs + other_regs <= 512
    return reduce_regs, compute_regs, other_regs


def setmaxreg(count: int) -> str:
    """The ``setmaxnreg`` call moving a warp from the 128-register launch
    budget to ``count`` (an increase or a decrease, never both)."""
    if count > 128:
        return f"cute.arch.setmaxregister_increase({count})"
    if count < 128:
        return f"cute.arch.setmaxregister_decrease({count})"
    return "pass"


def hook_chunk_prefix(
    ind: str, *, h: int, hook_cols: int, pre_scaled: bool, has_grad: bool
) -> str:
    """The lines opening one ``hook_cols``-column sub-chunk of a compute
    warpgroup's 32-column score chunk ``fbwd_ch`` (the 1-CTA and 2-CTA flex
    kernels; the pair kernel's dQ clusters index the chunk differently):
    the thread's chunk index within the 128-column tile (``fbwd_ch2``), its
    register offset, the score / coordinate sub-chunk views, the pre-scale of
    the raw scores and, under a Jacobian hook, the copy of the pre-hook
    scores it reads."""
    subs = 32 // hook_cols
    lines = [
        f"{ind}fbwd_ch2 = fbwd_ch * {subs} + fbwd_sub + {2 * subs * h}",
        f"{ind}fbwd_off = fbwd_ch * 32 + fbwd_sub * {hook_cols}",
        f"{ind}fbwd_s_chunk = cute.make_tensor(tLDrS[None, None, fbwd_ch].iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))",
        f"{ind}fbwd_cchunk = cute.logical_divide(tLDcS[None, None, fbwd_ch], {hook_cols})[None, fbwd_sub]",
    ]
    if pre_scaled:
        lines.append(
            f"{ind}_helion_flash_rt._scale_fragment_packed_f32x2(fbwd_s_chunk, fbwd_scale)"
        )
    if has_grad:
        lines.extend(
            (
                f"{ind}fbwd_spre_chunk = cute.make_tensor(fbwd_spre[None, None, fbwd_ch].iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))",
                f"{ind}cute.autovec_copy(fbwd_s_chunk, fbwd_spre_chunk)",
            )
        )
    return "\n".join(lines) + "\n"


def hook_jacobian_lines(*, h: int, hook_cols: int, jac_body: str | None) -> str:
    """The Jacobian hook's pass over one 32-column dS chunk ``fbwd_ch`` of a
    compute warpgroup (the 1-CTA and 2-CTA flex kernels): per
    ``hook_cols``-column sub-chunk, the chunk index and the pre-hook score /
    coordinate views ``jac_body`` reads, then dS *= the rendered Jacobian.
    Empty without a Jacobian region (``jac_body`` None)."""
    if jac_body is None:
        return ""
    subs = 32 // hook_cols
    return (
        f"                for fbwd_sub in cutlass.range_constexpr({subs}):\n"
        f"                    fbwd_ch2 = fbwd_ch * {subs} + fbwd_sub + {2 * subs * h}\n"
        f"                    fbwd_spre_chunk = cute.make_tensor(fbwd_spre[None, None, fbwd_ch].iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))\n"
        f"                    fbwd_cchunk = cute.logical_divide(tLDcS[None, None, fbwd_ch], {hook_cols})[None, fbwd_sub]\n"
        + indent_lines(jac_body, 20)
        + f"                    fbwd_ds_sub = cute.make_tensor(tLDrdP.iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))\n"
        f"                    _helion_flash_rt.fbwd_mul_pairs_packed(fbwd_ds_sub, fbwd_spre, fbwd_ch * 32 + fbwd_sub * {hook_cols}, {hook_cols})\n"
    )


def lse_fold_lines(ind: str, *, hook_cols: int, fold_body: str | None) -> str:
    """The folded additive bias: ``g`` rendered into ``fbwd_gfrag`` by the
    plain hook's chunk body ``fold_body``, then ``lse' = lse - c g`` on the
    sub-chunk's LSE values. Empty when the fold is off (``fold_body`` None)."""
    if fold_body is None:
        return ""
    return (
        f"{ind}fbwd_gfrag = cute.make_rmem_tensor(cute.make_layout({hook_cols}), cutlass.Float32)\n"
        + indent_lines(fold_body, len(ind))
        + f"{ind}_helion_flash_rt.fbwd_lse_fold(fbwd_lse_sub, fbwd_gfrag, fbwd_neg_log2c)\n"
    )


def chunked_region_renderer(
    chunk_loop: Callable[[int], str] | None,
) -> Callable[[str, int], str]:
    """``chunked_region(text, indent)``: a role's step region (over
    ``fbwd_steps``) under the chunk-loop head ``chunk_loop(indent)`` when the
    step table is chunked, its loop then running over the chunk's
    ``fbwd_csteps``; the text unchanged when ``chunk_loop`` is None."""

    def chunked_region(text: str, indent: int) -> str:
        if chunk_loop is None:
            return text
        return chunk_loop(indent) + textwrap.indent(
            re.sub(r"\bfbwd_steps\b", "fbwd_csteps", text), "    "
        )

    return chunked_region


def fbwd_tile_loop(region: str, *, head: str, tail: str) -> str:
    """A warp role's per-tile ``region`` under the persistent walk: indented
    one level between the loop ``head`` (the tile fetch and the per-tile
    derivations) and its ``tail`` (the response consume and the tile
    counter)."""
    return head + textwrap.indent(region, "    ") + tail


# --------------------------------------------------------------------------
# The dK/dV epilogue block, the CLC persistent-scheduler fragments and the
# 2-CTA dS exchange (dense and flex kernels)
# --------------------------------------------------------------------------


def fbwd_epilogue_block(
    *,
    h: int,
    rows: int,
    d: int,
    epi_off: str,
    epi_scale: str,
    kv_tile: str,
    io_dtype: str,
    dkv_wait: str,
    v_empty_expr: str,
    gm_expr: str | None = None,
    cols: int | None = None,
    thr_mma: str = "fbwd_tst",
    slots: int | None = None,
    stage_buf: str | None = None,
    kv_slot: str = "0",
    tail_extra: str = "",
) -> str:
    """dK/dV epilogue of compute warpgroup ``h``: TMEM -> registers -> a free
    smem buffer as (128, 64) SWIZZLE_128B boxes -> one TMA store per box.
    ``stage_buf`` overrides the staging buffer (the persistent flex alias_s
    walk stages dV through the dO slot, see the flex emitter); ``kv_slot``
    is the V ring slot dV stages through (the slot's boxes, under a
    multi-slot ring); ``tail_extra`` is appended after the issuing warp's
    final store wait (the flex persistent walk hands the dK staging back
    there).

    ``d`` is the accumulator's TMEM width (the MMA's N, a multiple of 64 in
    the padded flex plan); ``cols`` (default ``d``, a multiple of 16) the
    columns the tensor has: a partial last box is stored after its last
    chunk and the TMA unit clips the columns past the tensor extent.
    ``thr_mma`` is the thread MMA whose accumulator partition the TMEM tile
    follows (its N is ``d``); ``slots`` the (128, 64) boxes the staging
    buffer holds: a tile with more boxes than slots reuses them in order,
    waiting for the store that read the slot before overwriting it.

    dV (h=0) stages through this tile's V operand slot (free once dP(n-1)
    retired, i.e. at dv_done) and hands the slot back to the load warp with an
    arrive on the V ring's empty barrier (``v_empty_expr``) after the store has
    completed; dK (h=1) stages through sdS (free after the tail dQ, i.e. at
    dkv_done), so K/Kt can be released by the MMA warp at the tail and the next
    tile's K loads overlap the epilogue."""
    buf = stage_buf or ("sV" if h == 0 else "sdS")
    tma = "_fbwd_tma_dv" if h == 0 else "_fbwd_tma_dk"
    gm = ("_fbwd_mdVt" if h == 0 else "_fbwd_mdKt") if gm_expr is None else gm_expr
    warp_lo = 4 if h == 0 else 8
    cols = d if cols is None else cols
    assert cols % 16 == 0 and cols <= d, (cols, d)
    n_chunks = cols // 16
    n_boxes = -(-n_chunks // 4)
    slots = n_boxes if slots is None else slots

    def slot(c: int) -> int:
        return (c // 4) % slots

    def reuse_wait(c: int) -> str:
        # The slot's earlier box: its TMA store must have read the staging
        # before the warps overwrite it (the issuing warp waits, then the
        # group syncs).
        if c % 4 != 0 or c // 4 < slots:
            return ""
        return f"""
        if warp_idx == {warp_lo}:
            cute.arch.cp_async_bulk_wait_group({slots - 1}, read=True)
        _helion_flash_rt.named_barrier_wait_unaligned({5 + h}, 128)"""

    chunks = "".join(
        reuse_wait(c)
        + (
            f"""
        fbwd_row64 = cute.make_rmem_tensor(cute.make_layout(64), {io_dtype})"""
            if c % 4 == 0
            else ""
        )
        + f"""
        fbwd_reg = cute.make_rmem_tensor(tEPIcC[None, 0, None].shape, cutlass.Float32)
        cute.copy(fbwd_tiled_epi_ld, tEPItT[None, {c}, None], fbwd_reg)
        {epi_scale}
        fbwd_rego = cute.make_tensor(fbwd_row64.iterator + {16 * (c % 4)}, cute.make_layout(tEPIcC[None, 0, None].shape))
        fbwd_rego.store(fbwd_reg.load().to({io_dtype}))"""
        + (
            f"""
        cute.autovec_copy(fbwd_row64, fbwd_sEpi[fbwd_local_tidx, None, fbwd_epi_slot0 + {slot(c)}])
        cute.arch.fence_view_async_shared()
        _helion_flash_rt.named_barrier_wait_unaligned({5 + h}, 128)
        if warp_idx == {warp_lo}:
            tEPIsS{c // 4}, tEPIgS{c // 4} = cute_cpasync_flash.tma_partition({tma}, 0, cute.make_layout(1), cute.group_modes(fbwd_sEpi[None, None, fbwd_epi_slot0 + {slot(c)}], 0, 2), cute.group_modes(fbwd_gEpi[None, None, {kv_tile}, {c // 4}], 0, 2))
            cute.copy({tma}, tEPIsS{c // 4}, tEPIgS{c // 4})
            cute.arch.cp_async_bulk_commit_group()"""
            if c % 4 == 3 or c == n_chunks - 1
            else ""
        )
        for c in range(n_chunks)
    )
    hs = (
        """
            with cute.arch.elect_one():
                _helion_flash_rt.mbarrier_arrive(fbwd_kv_empty_ptr)"""
        if h == 0
        else ""
    )
    kv_empty = (
        f"""
        fbwd_kv_empty_ptr = {v_empty_expr}
"""
        if h == 0
        else ""
    )
    done_ptr = "fbwd_dv_done_ptr" if h == 0 else "fbwd_dkv_done_ptr"
    return f"""        _helion_flash_rt.mbar_spin_wait({done_ptr}, {dkv_wait}, 10000000)
        fbwd_dkv_shape = {thr_mma}.partition_shape_C(({rows}, {d}))
        tDKV_frag = {thr_mma}.make_fragment_C(fbwd_dkv_shape)
        tEPI_t = cute.make_tensor(fbwd_tmem_ptr + {epi_off}, tDKV_frag.layout)
        fbwd_epi_tiler = ((cute.size(tEPI_t, mode=[0, 0]), cute.size(tEPI_t, mode=[0, 1])),)
        tEPI_epi = cute.zipped_divide(tEPI_t, fbwd_epi_tiler)
        fbwd_epi_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(16)), cutlass.Float32)
        fbwd_tiled_epi_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_epi_ld_atom, tEPI_epi[None, 0])
        fbwd_thr_epi_ld = fbwd_tiled_epi_ld.get_slice(fbwd_local_tidx)
        tEPItT = fbwd_thr_epi_ld.partition_S(tEPI_epi)
        fbwd_cepi = cute.make_identity_tensor(({rows}, {d}))
        fbwd_c_epi = cute.zipped_divide({thr_mma}.partition_C(fbwd_cepi), fbwd_epi_tiler)
        tEPIcC = fbwd_thr_epi_ld.partition_D(fbwd_c_epi)
        # Epilogue: the tile is staged through the (now free) {buf} operand buffer
        # as (128, 64) SWIZZLE_128B boxes -- 16 columns per t2r chunk, one row
        # per lane -- and each complete box is TMA-stored by the group's first
        # warp (warp-collective issue).
        fbwd_sEpi = storage.{buf}.get_tensor(_fbwd_epil.outer, swizzle=_fbwd_epil.inner)
        fbwd_epi_slot0 = {f"{d // 64} * ({kv_slot})" if h == 0 else "0"}
        fbwd_gEpi = cute.local_tile({gm}, (128, 64), (None, None)){kv_empty}{chunks}
        if warp_idx == {warp_lo}:
            cute.arch.cp_async_bulk_wait_group(0, read=True){hs}
{tail_extra}"""


FBWD_CLC_CONSUME = (
    "{ind}fbwd_clc_pipe.consumer_wait(fbwd_clc_cstate)\n"
    "{ind}fbwd_clc_rptr = (fbwd_clc_base + fbwd_clc_cstate.index * cutlass.Int32(4)).align(16)\n"
    "{ind}fbwd_work = fbwd_clc_hw.work_tile_info_from_clc_response(fbwd_clc_rptr)\n"
    "{ind}fbwd_clc_pipe.consumer_release(fbwd_clc_cstate)\n"
    "{ind}fbwd_clc_cstate.advance()\n"
)


def fbwd_clc_setup(total_ctas: int, cluster: int, cta_layout: str) -> str:
    """The CLC scheduler's state at kernel entry (body indentation)."""
    return f"""
    # CLC dynamic tile scheduler: warp 15 steals not-yet-launched tiles and the
    # 16-byte responses flow through a 2-stage ring every warp consumes. All
    # 512 threads of every CTA in the cluster release into the leader's empty
    # barrier, so its arrive count scales with the cluster size.
    fbwd_clc_params = cutlass_utils_flash.ClcDynamicPersistentTileSchedulerParams(problem_shape_ntile_mnl=({total_ctas}, 1, 1), cluster_shape_mnk=({cluster}, 1, 1))
    fbwd_clc_hw = cutlass_utils_flash.ClcDynamicPersistentTileScheduler.create(fbwd_clc_params, cute.arch.block_idx(), cute.arch.grid_dim(), storage.clc_response.data_ptr())
    fbwd_clc_base = storage.clc_response.data_ptr()
    fbwd_clc_pipe = cutlass_pipeline_flash.PipelineClcFetchAsync.create(barrier_storage=storage.clc_mbar_ptr.data_ptr(), num_stages=2, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread, {512 * cluster}), tx_count=16, cta_layout_vmnk={cta_layout})
    fbwd_clc_cstate = cutlass_pipeline_flash.make_pipeline_state(cutlass_pipeline_flash.PipelineUserType.Consumer, 2)
"""


def fbwd_clc_consume_only(ind: str) -> str:
    """A warp with no per-tile work still consumes every CLC response."""
    return (
        f"{ind}fbwd_work = fbwd_clc_hw.initial_work_tile_info()\n"
        f"{ind}while fbwd_work.is_valid_tile:\n"
        + FBWD_CLC_CONSUME.format(ind=ind + "    ")
    )


def fbwd_clc_scheduler(ind: str, leader_guard: str | None) -> str:
    """Scheduler warp: steal the next tile (leader CTA only in a cluster; the
    multicast response reaches every CTA) and consume like everyone else."""
    issue = (
        f"{ind}    with cute.arch.elect_one():\n"
        f"{ind}        cute.arch.issue_clc_query(fbwd_clc_mbar, fbwd_clc_rptr)\n"
    )
    if leader_guard is not None:
        issue = f"{ind}    if {leader_guard}:\n" + "".join(
            "    " + line + "\n" for line in issue.splitlines()
        )
    # The query is issued only once the current tile's dV is complete (dv_done):
    # reserving the next tile at tile start would commit it a whole tile early
    # and, with causal tiles of very different lengths, leave the tail badly
    # balanced (gqa-causal lost 12% that way); the epilogue still hides the
    # response latency.
    return (
        f"{ind}fbwd_clc_pstate = cutlass_pipeline_flash.make_pipeline_state(cutlass_pipeline_flash.PipelineUserType.Producer, 2)\n"
        f"{ind}fbwd_clc_tile_it = cutlass.Int32(0)\n"
        f"{ind}fbwd_work = fbwd_clc_hw.initial_work_tile_info()\n"
        f"{ind}while fbwd_work.is_valid_tile:\n"
        f"{ind}    _helion_flash_rt.mbar_spin_wait(fbwd_dv_done_ptr, fbwd_clc_tile_it % 2, 10000000)\n"
        f"{ind}    fbwd_clc_tile_it = fbwd_clc_tile_it + 1\n"
        f"{ind}    fbwd_clc_pipe.producer_acquire(fbwd_clc_pstate)\n"
        f"{ind}    fbwd_clc_mbar = fbwd_clc_pipe.producer_get_barrier(fbwd_clc_pstate)\n"
        f"{ind}    fbwd_clc_rptr = (fbwd_clc_base + fbwd_clc_pstate.index * cutlass.Int32(4)).align(16)\n"
        + issue
        + f"{ind}    fbwd_clc_pstate.advance()\n"
        + FBWD_CLC_CONSUME.format(ind=ind + "    ")
        + f"{ind}fbwd_clc_pipe.producer_tail(fbwd_clc_pstate)\n"
    )


FBWD_2CTA_SEND = """                if fbwd_local_tidx == 0:
                    fbwd_peer = fbwd_rank ^ 1
                    cute.arch.mbarrier_arrive_and_expect_tx(fbwd_ds_cl_full_ptr, 16384, peer_cta_rank_in_cluster=fbwd_peer)
                    _helion_flash_rt.cpasync_bulk_s2cluster(fbwd_sdSx_ptr, fbwd_sdS_mn.iterator + fbwd_rank * 8192, fbwd_ds_cl_full_ptr, 16384, fbwd_peer)"""


# dS (bf16-packed, 64 words per compute warpgroup) sits in its OWN dP column
# half (words 0-31 of 256 + 64*h): a warpgroup's dS stores then only overwrite
# dP columns it has already loaded, so the dP/dS pass runs in two 32-column
# chunks without cross-warpgroup barriers. The dK gemm's A operand is thus
# gapped: k-tiles 0-3 at words 0-31, k-tiles 4-7 at words 64-95.
FBWD_2CTA_DS_OFFSETS = tuple((k // 4) * 64 + (k % 4) * 8 for k in range(8))


# --------------------------------------------------------------------------
# Template runs shared by the backward bodies: the text below is emitted
# verbatim by two or more of the dense 1-CTA / 2-CTA and flex 1-CTA / 2-CTA /
# pair kernels (the parameters are the holes that differ between them).
# --------------------------------------------------------------------------


def fbwd_compute_tmem_views(*, rows: int, s_col: int, dp_col: int) -> str:
    """A compute warpgroup's view of its 64-column halves of the S and dP
    accumulators (``rows`` x 128 tiles starting at TMEM columns ``s_col`` /
    ``dp_col``), after the allocation handshake with the TMEM warp."""
    return f"""        fbwd_s_shape = fbwd_sst.partition_shape_C(({rows}, 128))
        tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
        fbwd_half_layout = cute.composition(tStS_frag.layout, cute.make_layout((128, 64)))
        tS_h = cute.make_tensor(fbwd_tmem_ptr + {s_col}, fbwd_half_layout)
        tdP_h = cute.make_tensor(fbwd_tmem_ptr + {dp_col}, fbwd_half_layout)
"""


def fbwd_compute_step_state(*, scale_expr: str, log2_factor: float) -> str:
    """A flex compute warpgroup's per-tile state: the S / dQ-free barrier
    phases, the step counter, the softmax scale, its log2 conversion factor
    and the thread's stats column."""
    return f"""        fbwd_phase = cutlass.Int32(0)
        fbwd_dqf_phase = cutlass.Int32(0)
        fbwd_gstep = cutlass.Int32(0)
        fbwd_scale = cutlass.Float32({scale_expr})
        fbwd_log2c = cutlass.Float32({log2_factor!r})
        fbwd_neg_log2c = cutlass.Float32({-log2_factor!r})
        fbwd_scale2 = cutlass.Float32(({scale_expr}) * {log2_factor!r})
        fbwd_stat_j = fbwd_local_tidx % 64
"""


# The compute warpgroup's t2r partitions of its S / dP halves (``cS`` is the
# tile's coordinate tensor) and the layout of the 32-column P half it writes.
FBWD_COMPUTE_LOAD_VIEWS = """        tScS = fbwd_sst.partition_C(cS)
        fbwd_cs_half_layout = cute.composition(tScS.layout, cute.make_layout((128, 64)))
        tScS_h = cute.make_tensor(tScS.iterator, fbwd_cs_half_layout)
        fbwd_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
        fbwd_tiled_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_ld_atom, tS_h)
        fbwd_thr_ld = fbwd_tiled_ld.get_slice(fbwd_local_tidx)
        tLDtS = fbwd_thr_ld.partition_S(tS_h)
        tLDtdP = fbwd_thr_ld.partition_S(tdP_h)
        tLDcS = fbwd_thr_ld.partition_D(tScS_h)
        fbwd_p_half_layout = cute.composition(tStS_frag.layout, cute.make_layout((128, 32)))
"""


def fbwd_p_store_views(*, p_col: int, ds_col: int | None = None) -> str:
    """The compute warpgroup's r2t partitions for its bf16-packed P half at
    TMEM column ``p_col`` and, in the 2-CTA kernels (``ds_col``), its dS
    half."""
    ds_view = (
        f"        tDS_h = cute.make_tensor(fbwd_tmem_ptr + {ds_col}, fbwd_p_half_layout)\n"
        if ds_col is not None
        else ""
    )
    ds_part = (
        "        tSTtDS = fbwd_thr_st.partition_D(tDS_h)\n"
        if ds_col is not None
        else ""
    )
    return (
        f"        tP_h = cute.make_tensor(fbwd_tmem_ptr + {p_col}, fbwd_p_half_layout)\n"
        + ds_view
        + """        fbwd_cp_half_layout = cute.composition(tScS.layout, cute.make_layout((128, 32)))
        tPcS_h = cute.make_tensor(tScS.iterator, fbwd_cp_half_layout)
        fbwd_st_atom = cute.make_copy_atom(cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(16)), cutlass.Float32)
        fbwd_tiled_st = cute_tcgen05_flash.make_tmem_copy(fbwd_st_atom, tP_h)
        fbwd_thr_st = fbwd_tiled_st.get_slice(fbwd_local_tidx)
        tSTtP = fbwd_thr_st.partition_D(tP_h)
"""
        + ds_part
        + "        tSTcP = fbwd_thr_st.partition_S(tPcS_h)\n"
    )


# The 1-CTA kernels' operand buffers: Q / dO / K / V as loaded, their
# transposed re-views for the dK / dV / dQ MMAs and dS in both major-nesses.
FBWD_SMEM_VIEWS = """    sQ = storage.sQ.get_tensor(_fbwd_qsl.outer, swizzle=_fbwd_qsl.inner)
    sdO = storage.sdO.get_tensor(_fbwd_dosl.outer, swizzle=_fbwd_dosl.inner)
    sK = storage.sK.get_tensor(_fbwd_ksl.outer, swizzle=_fbwd_ksl.inner)
    sV = storage.sV.get_tensor(_fbwd_vsl.outer, swizzle=_fbwd_vsl.inner)
    sQt = cute.make_tensor(cute.recast_ptr(sQ.iterator, _fbwd_qtl.inner), _fbwd_qtl.outer)
    sdOt = cute.make_tensor(cute.recast_ptr(sdO.iterator, _fbwd_dotl.inner), _fbwd_dotl.outer)
    sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, _fbwd_ktl.inner), _fbwd_ktl.outer)
    fbwd_sdS_mn = storage.sdS.get_tensor(_fbwd_dssl.outer, swizzle=_fbwd_dssl.inner)
    fbwd_sdS_nk = cute.make_tensor(cute.recast_ptr(fbwd_sdS_mn.iterator, _fbwd_dsnk.inner), _fbwd_dsnk.outer)
"""

# The 1-CTA kernels' barrier arrive counts: the MMA warp's single-thread
# commits, the 256 compute threads' P / dS handoffs and the 128 reduce
# threads' dQ release.
FBWD_MBAR_INITS = """    if tidx == 0:
        cute.arch.mbarrier_init(fbwd_s_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dp_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_p_full_ptr, 256)
        cute.arch.mbarrier_init(fbwd_ds_full_ptr, 256)
        cute.arch.mbarrier_init(fbwd_dq_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dq_empty_ptr, 128)
        cute.arch.mbarrier_init(fbwd_dkv_done_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dv_done_ptr, 1)
"""


def fbwd_tmem_warp(body: str, *, dims: tuple[int, int, int] | None = None) -> str:
    """Warp 12, the TMEM owner, around its per-kernel ``body``: the
    512-column allocation, the 1-CTA kernels' accumulator fragments
    (``dims`` = the dV / dK / dQ widths; the 2-CTA kernels shape theirs in
    the body), the allocation handshake with the other twelve warps and the
    release once the body has run."""
    shapes = ""
    if dims is not None:
        dv, dk, dq = dims
        shapes = f"""        fbwd_s_shape = fbwd_sst.partition_shape_C((128, 128))
        tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
        fbwd_dkv_shape = fbwd_tst.partition_shape_C((128, {dv}))
        tDKV_frag = fbwd_tst.make_fragment_C(fbwd_dkv_shape)
        fbwd_dk_shape = fbwd_dskt.partition_shape_C((128, {dk}))
        tDK_frag = fbwd_dskt.make_fragment_C(fbwd_dk_shape)
        fbwd_dq_shape = fbwd_dqt.partition_shape_C((128, {dq}))
        tDQ_frag = fbwd_dqt.make_fragment_C(fbwd_dq_shape)
"""
    return f"""    if warp_idx == 12:
        fbwd_tmem.allocate(512)
{shapes}        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
{body}        fbwd_tmem.relinquish_alloc_permit()
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem.free(fbwd_tmem_ptr)
"""


def fbwd_mma_fragments(
    ind: str, *, dp_off: int, dv_off: int, dk_off: int, dq_off: int
) -> str:
    """The 1-CTA MMA warp's accumulator views (S at column 0, the others at
    the given TMEM columns) and operand fragments for its four MMAs."""
    return f"""{ind}tS_t = cute.make_tensor(fbwd_tmem_ptr, tStS_frag.layout)
{ind}tdP_t = cute.make_tensor(fbwd_tmem_ptr + {dp_off}, tStS_frag.layout)
{ind}tDV_t = cute.make_tensor(fbwd_tmem_ptr + {dv_off}, tDKV_frag.layout)
{ind}tDK_t = cute.make_tensor(fbwd_tmem_ptr + {dk_off}, tDK_frag.layout)
{ind}tDQ_t = cute.make_tensor(fbwd_tmem_ptr + {dq_off}, tDQ_frag.layout)
{ind}tP = cute.make_tensor(tS_t.iterator, _fbwd_ptl.outer)
{ind}tSrK = fbwd_sst.make_fragment_A(sK)
{ind}tSrQ = fbwd_sst.make_fragment_B(sQ)
{ind}tSrV = fbwd_sst.make_fragment_A(sV)
{ind}tSrDO = fbwd_sst.make_fragment_B(sdO)
{ind}tTSrP = fbwd_tst.make_fragment_A(tP)
{ind}tTSrDOt = fbwd_tst.make_fragment_B(sdOt)
{ind}tKSrDS = fbwd_dskt.make_fragment_A(fbwd_sdS_nk)
{ind}tKSrQt = fbwd_dskt.make_fragment_B(sQt)
{ind}tDQrDS = fbwd_dqt.make_fragment_A(fbwd_sdS_mn)
{ind}tDQrKt = fbwd_dqt.make_fragment_B(sKt)
"""


def fbwd_mma_descriptor_setup(ind: str, *, ts_op: str, dsk_op: str) -> str:
    """The 1-CTA MMA warp's accumulator addresses, operand descriptor bases
    and instruction descriptors for the S / dP, dV (``ts_op``) and dK
    (``dsk_op``) MMAs."""
    return f"""{ind}fbwd_s_addr = tS_t.iterator.toint()
{ind}fbwd_dp_addr = tdP_t.iterator.toint()
{ind}fbwd_dv_addr = tDV_t.iterator.toint()
{ind}fbwd_dk_addr = tDK_t.iterator.toint()
{ind}fbwd_dqa_addr = tDQ_t.iterator.toint()
{ind}fbwd_k_base = _helion_flash_ptx.smem_desc_base_from_tensor(sK, _helion_flash_ptx.Major.K)
{ind}fbwd_v_base = _helion_flash_ptx.smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.K)
{ind}_helion_flash_ptx.declare_ptx_idesc(_fbwd_ss_mma.op, "helion_fbwd_ss_idesc")
{ind}fbwd_q_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQ, _helion_flash_ptx.Major.K)
{ind}fbwd_do_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdO, _helion_flash_ptx.Major.K)
{ind}fbwd_dot_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdOt, _helion_flash_ptx.Major.MN)
{ind}_helion_flash_ptx.declare_ptx_idesc({ts_op}.op, "helion_fbwd_ts_idesc")
{ind}fbwd_dsnk_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sdS_nk, _helion_flash_ptx.Major.K)
{ind}fbwd_qt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQt, _helion_flash_ptx.Major.MN)
{ind}_helion_flash_ptx.declare_ptx_idesc({dsk_op}.op, "helion_fbwd_dsk_idesc")
"""


def fbwd_mma_tile_entry(ind: str) -> str:
    """The 1-CTA MMA warp's entry into a KV tile: the K / K^T slot wait, the
    parked operand start addresses it reads back and the first-step flags."""
    return f"""{ind}fbwd_ke = fbwd_k_cons.wait_and_advance()
{ind}fbwd_mb[0] = _helion_flash_ptx.make_smem_desc_start_addr(sK[None, None, None, fbwd_ke.index].iterator)
{ind}fbwd_mb[4] = _helion_flash_ptx.make_smem_desc_start_addr(sKt[None, None, None, fbwd_ke.index].iterator)
{ind}fbwd_st_k = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 0)
{ind}fbwd_st_dsnk = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 2)
{ind}fbwd_st_dsmn = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 3)
{ind}fbwd_st_kt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 4)
{ind}fbwd_dv_zero = cutlass.Boolean(True)
{ind}fbwd_dk_zero = cutlass.Boolean(True)
"""


def fbwd_mma_step_loop_head(ind: str) -> str:
    """After the 1-CTA MMA warp's peeled first step: the dO slot release and
    the head of the software-pipelined step loop (the volatile re-reads of
    the operand addresses and the next Q slot wait)."""
    return f"""{ind}fbwd_dv_zero = cutlass.Boolean(False)
{ind}fbwd_doe.release()
{ind}for fbwd_i in cutlass.range(fbwd_steps - 1, unroll=1):
{ind}    fbwd_st_k = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 0)
{ind}    fbwd_st_v = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 1)
{ind}    fbwd_st_dsnk = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 2)
{ind}    fbwd_st_dsmn = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 3)
{ind}    fbwd_st_kt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 4)
{ind}    fbwd_qe2 = fbwd_q_cons.wait_and_advance()
"""


def fbwd_dq_reduce_views(*, d: int, dq_off: int) -> str:
    """The reduce warps' view of the (128, ``d``) dQ accumulator at TMEM
    column ``dq_off``, after the allocation handshake."""
    return f"""        fbwd_dq_shape = fbwd_dqt.partition_shape_C((128, {d}))
        tDQ_frag = fbwd_dqt.make_fragment_C(fbwd_dq_shape)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
        tDQ_t = cute.make_tensor(fbwd_tmem_ptr + {dq_off}, tDQ_frag.layout)
"""


def fbwd_dq_reduce_partitions(
    ind: str, *, d: int, thr_mma: str = "fbwd_dqt", slot_b: str = ""
) -> str:
    """The reduce warps' t2r partitions of the (128, ``d``) dQ tile
    ``tDQ_t`` (``thr_mma`` is the thread MMA whose accumulator it follows);
    ``slot_b`` is the second-slot partition of the 64-column chunked flex
    dQ, placed ahead of the coordinate partition."""
    return f"""{ind}cDQ = cute.make_identity_tensor((128, {d}))
{ind}tDQcDQ = {thr_mma}.partition_C(cDQ)
{ind}fbwd_dq_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
{ind}fbwd_tiled_dq_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_dq_ld_atom, tDQ_t)
{ind}fbwd_thr_dq_ld = fbwd_tiled_dq_ld.get_slice(fbwd_local_tidx)
{ind}tRDtDQ = fbwd_thr_dq_ld.partition_S(tDQ_t)
{slot_b}{ind}tRDcDQ = fbwd_thr_dq_ld.partition_D(tDQcDQ)
"""


# The 2-CTA kernels' operand buffers: Q / dO / K in both major-nesses (each
# CTA holds a different half of the operand per orientation, so neither half
# re-views as the other; flash_bwd_2cta_shared_storage lists the halves), the
# dS exchange buffer, the stats rows and the dQ staging.
FBWD_2CTA_SMEM_VIEWS = """    sQ = storage.sQ.get_tensor(_fbwd_qsl.outer, swizzle=_fbwd_qsl.inner)
    sdOt = storage.sdOt.get_tensor(_fbwd_dosl.outer, swizzle=_fbwd_dosl.inner)
    sdO = storage.sdO.get_tensor(_fbwd_dotl.outer, swizzle=_fbwd_dotl.inner)
    sQt = storage.sQt.get_tensor(_fbwd_qtl.outer, swizzle=_fbwd_qtl.inner)
    sK = storage.sK.get_tensor(_fbwd_ksl.outer, swizzle=_fbwd_ksl.inner)
    sV = storage.sV.get_tensor(_fbwd_vsl.outer, swizzle=_fbwd_vsl.inner)
    sKt = storage.sKt.get_tensor(_fbwd_ktl.outer, swizzle=_fbwd_ktl.inner)
    fbwd_sdS_mn = storage.sdS.get_tensor(_fbwd_dssl.outer, swizzle=_fbwd_dssl.inner)
    fbwd_sdSx_plain = storage.sdSx.get_tensor(cute.make_layout(8192))
    fbwd_sdSx_ptr = cute.recast_ptr(fbwd_sdSx_plain.iterator, _fbwd_dssl.inner)
    fbwd_sLSE = storage.sLSE.get_tensor(cute.make_layout(2 * 128))
    fbwd_sDelta = storage.sDelta.get_tensor(cute.make_layout(2 * 128))
    fbwd_sdq = storage.sdQaccum.get_tensor(_fbwd_dqsl.outer, swizzle=_fbwd_dqsl.inner)
"""


def fbwd_2cta_operand_partitions(d: int) -> str:
    """The 2-CTA kernels' per-rank thread MMAs, the global operand tiles of
    head dim ``d`` (256-row K / V / K^T, 128-row Q / dO in both major-nesses)
    and their TMA partitions."""
    return f"""    fbwd_sst = _fbwd_ss_mma.get_slice(fbwd_rank)
    fbwd_tst = _fbwd_ts_mma.get_slice(fbwd_rank)
    fbwd_dqt = _fbwd_dq_mma.get_slice(fbwd_rank)
    fbwd_mcast = cutlass_pipeline_flash.PipelineUmmaAsync._compute_tmem_sync_mask(_fbwd_cluster_vmnk)
    gK = cute.flat_divide(_fbwd_mKt, cute.select((256, 128, {d}), mode=[0, 2]))
    gV = cute.flat_divide(_fbwd_mVt, cute.select((256, 128, {d}), mode=[0, 2]))
    gQ = cute.flat_divide(_fbwd_mQt, cute.select((256, 128, {d}), mode=[1, 2]))
    gdOt = cute.flat_divide(_fbwd_mdOtN, cute.select((256, 128, {d}), mode=[1, 2]))
    gdO = cute.flat_divide(_fbwd_mdOtT, cute.select((256, {d}, 128), mode=[1, 2]))
    gQt = cute.flat_divide(_fbwd_mQtT, cute.select((256, {d}, 128), mode=[1, 2]))
    gKt = cute.flat_divide(_fbwd_mKtT, cute.select((128, {d}, 256), mode=[1, 2]))
    tSgK = fbwd_sst.partition_A(gK)
    tSgV = fbwd_sst.partition_A(gV)
    tSgQ = fbwd_sst.partition_B(gQ)
    tSgDOt = fbwd_sst.partition_B(gdOt)
    tTSgDO = fbwd_tst.partition_B(gdO)
    tTSgQt = fbwd_tst.partition_B(gQt)
    tDQgKt = fbwd_dqt.partition_B(gKt)
    tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
    tVsV, tVgV_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tSgV, 0, 3))
    tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
    tDOTsDOT, tDOTgDOT_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_dot, 0, cute.make_layout(1), cute.group_modes(sdOt, 0, 3), cute.group_modes(tSgDOt, 0, 3))
    tDOsDO, tDOgDO_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_do2, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 3), cute.group_modes(tTSgDO, 0, 3))
    tQTsQT, tQTgQT_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_qt, 0, cute.make_layout(1), cute.group_modes(sQt, 0, 3), cute.group_modes(tTSgQt, 0, 3))
    tKTsKT, tKTgKT_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_kt, 0, cute.make_layout(1), cute.group_modes(sKt, 0, 3), cute.group_modes(tDQgKt, 0, 3))
"""


def fbwd_2cta_mma_setup(
    *, d: int, s_base: str, dp_off: int, dv_off: int, dk_off: int, dq_off: int
) -> str:
    """The 2-CTA leader's MMA warp setup (12-space indentation, under its
    leader guard): the accumulator fragments of head dim ``d`` (S at
    ``s_base``, dP / dV / dK / dQ at the TMEM columns ``dp_off`` / ``dv_off``
    / ``dk_off`` / ``dq_off``), operand fragments, descriptor bases and
    instruction descriptors, the parked operand start addresses and the
    barrier phases."""
    return f"""            fbwd_s_shape = fbwd_sst.partition_shape_C((256, 128))
            tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
            fbwd_dkv_shape = fbwd_tst.partition_shape_C((256, {d}))
            tDKV_frag = fbwd_tst.make_fragment_C(fbwd_dkv_shape)
            fbwd_dq_shape = fbwd_dqt.partition_shape_C((128, {d}))
            tDQ_frag = fbwd_dqt.make_fragment_C(fbwd_dq_shape)
            tS_t = cute.make_tensor({s_base}, tStS_frag.layout)
            tdP_t = cute.make_tensor(fbwd_tmem_ptr + {dp_off}, tStS_frag.layout)
            tDV_t = cute.make_tensor(fbwd_tmem_ptr + {dv_off}, tDKV_frag.layout)
            tDK_t = cute.make_tensor(fbwd_tmem_ptr + {dk_off}, tDKV_frag.layout)
            tDQ_t = cute.make_tensor(fbwd_tmem_ptr + {dq_off}, tDQ_frag.layout)
            tP = cute.make_tensor(tS_t.iterator, _fbwd_ptl.outer)
            tDS = cute.make_tensor(tdP_t.iterator, _fbwd_ptl.outer)
            tSrK = fbwd_sst.make_fragment_A(sK)
            tSrQ = fbwd_sst.make_fragment_B(sQ)
            tSrV = fbwd_sst.make_fragment_A(sV)
            tSrDOt = fbwd_sst.make_fragment_B(sdOt)
            tTSrP = fbwd_tst.make_fragment_A(tP)
            tTSrDS = fbwd_tst.make_fragment_A(tDS)
            tTSrDO = fbwd_tst.make_fragment_B(sdO)
            tTSrQt = fbwd_tst.make_fragment_B(sQt)
            tDQrDS = fbwd_dqt.make_fragment_A(fbwd_sdS_mn)
            tDQrKt = fbwd_dqt.make_fragment_B(sKt)
            fbwd_s_addr = tS_t.iterator.toint()
            fbwd_dp_addr = tdP_t.iterator.toint()
            fbwd_dv_addr = tDV_t.iterator.toint()
            fbwd_dk_addr = tDK_t.iterator.toint()
            fbwd_dqa_addr = tDQ_t.iterator.toint()
            fbwd_k_base = _helion_flash_ptx.smem_desc_base_from_tensor(sK, _helion_flash_ptx.Major.K)
            fbwd_v_base = _helion_flash_ptx.smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.K)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_ss_mma.op, "helion_fbwd_ss_idesc")
            fbwd_q_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQ, _helion_flash_ptx.Major.K)
            fbwd_dot_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdOt, _helion_flash_ptx.Major.K)
            fbwd_do2_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdO, _helion_flash_ptx.Major.MN)
            fbwd_qt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQt, _helion_flash_ptx.Major.MN)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_ts_mma.op, "helion_fbwd_ts_idesc")
            fbwd_dsmn_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sdS_mn, _helion_flash_ptx.Major.MN)
            fbwd_kt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sKt, _helion_flash_ptx.Major.MN)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_dq_mma.op, "helion_fbwd_dq_idesc")
            # Operand start addresses parked in smem; re-read with a volatile
            # load per iteration so the per-k descriptors are never hoisted.
            fbwd_mb = storage.mma_base.get_tensor(cute.make_layout(8))
            fbwd_mb[0] = _helion_flash_ptx.make_smem_desc_start_addr(sK[None, None, None, 0].iterator)
            fbwd_mb[1] = _helion_flash_ptx.make_smem_desc_start_addr(sV[None, None, None, 0].iterator)
            fbwd_mb[2] = _helion_flash_ptx.make_smem_desc_start_addr(fbwd_sdS_mn[None, None, None, 0].iterator)
            fbwd_mb[3] = _helion_flash_ptx.make_smem_desc_start_addr(sQ[None, None, None, 0].iterator)
            fbwd_mb[4] = _helion_flash_ptx.make_smem_desc_start_addr(sdOt[None, None, None, 0].iterator)
            fbwd_mb[5] = _helion_flash_ptx.make_smem_desc_start_addr(sdO[None, None, None, 0].iterator)
            fbwd_mb[6] = _helion_flash_ptx.make_smem_desc_start_addr(sQt[None, None, None, 0].iterator)
            fbwd_mb[7] = _helion_flash_ptx.make_smem_desc_start_addr(sKt[None, None, None, 0].iterator)
            fbwd_pf_phase = cutlass.Int32(0)
            fbwd_dst_phase = cutlass.Int32(0)
            fbwd_dss_phase = cutlass.Int32(0)
            fbwd_dcl_phase = cutlass.Int32(0)
            fbwd_dqe_phase = cutlass.Int32(0)
"""


# --------------------------------------------------------------------------
# The flex kernels' P pass: the hooks' chunk bodies around the exp2 recompute
# --------------------------------------------------------------------------


class FbwdPSpec(NamedTuple):
    """What a flex kernel's P pass needs besides the hook: the sub-chunk
    width, the exp2 scale symbol, the exp2 flavour knobs, the sub-chunk
    opener (the kernel's chunk index, register offset and views) and the
    plain hook's body under the LSE fold (None when the fold is off)."""

    hook_cols: int
    p_scale: str
    exp2_f32: bool
    exp2_emu: int
    chunk_prefix: Callable[[str], str]
    fold_body: str | None


def fbwd_p_sub_loop(ind: str, hook: FlexHookCode | None, spec: FbwdPSpec) -> str:
    """The P recompute of one 32-column score chunk in ``hook_cols``-lane
    sub-chunks: the folded bias and ``hook``'s chunk body (the mask hook on
    masked steps; None on the fold's unmasked steps) transform a sub-chunk's
    scores and the exp2 recompute consumes them before the next sub-chunk's
    hook runs, so ptxas has no long run of precomputable hook values to hoist
    above the S wait (where they spill)."""
    cols = spec.hook_cols
    return (
        f"{ind}for fbwd_sub in cutlass.range_constexpr({32 // cols}):\n"
        + spec.chunk_prefix(ind + "    ")
        + f"{ind}    fbwd_lse_sub = cute.make_tensor(fbwd_lse_frg.iterator + fbwd_sub * {cols}, cute.make_layout({cols}))\n"
        + lse_fold_lines(ind + "    ", hook_cols=cols, fold_body=spec.fold_body)
        + (indent_lines(hook.chunk_body, len(ind) + 4) if hook is not None else "")
        + f"{ind}    _helion_flash_rt.fbwd_p_pairs_packed(tLDrS, fbwd_lse_sub, fbwd_off, {cols}, {spec.p_scale}, None, fbwd_off, exp2_f32={spec.exp2_f32}, emu_res={spec.exp2_emu})\n"
    )


def fbwd_p_pass(
    hook: FlexHookCode | None,
    ind: str,
    spec: FbwdPSpec,
    *,
    h: int,
    lse_prescaled: bool,
    lse_in_mma: bool = False,
) -> str:
    """The P pass of compute warpgroup ``h`` over its two 32-column chunks
    (the 1-CTA and 2-CTA flex kernels): the chunk's LSE values -- the staged
    row, or zeros when the S MMA already subtracted the LSE (``lse_in_mma``)
    -- then the sub-chunk loop. A natural-log LSE (the in-body form) goes to
    the exp2 base with the no-visible-key rows (lse == -inf) mapped to 0; the
    preprocess kernel's ``lse2`` arrives converted (``lse_prescaled``)."""
    if lse_in_mma:
        lse = f"{ind}    fbwd_lse_frg.fill(cutlass.Float32(0.0))\n"
    else:
        lse = (
            f"{ind}    fbwd_lse_v = cute.make_tensor(fbwd_sLSE.iterator + fbwd_par + {64 * h} + fbwd_ch * 32, cute.make_layout(32))\n"
            f"{ind}    cute.autovec_copy(fbwd_lse_v, fbwd_lse_frg)\n"
        )
        if not lse_prescaled:
            lse += f"{ind}    _helion_flash_rt.fbwd_lse_to_log2(fbwd_lse_frg, fbwd_log2c)\n"
    return (
        f"{ind}for fbwd_ch in cutlass.range_constexpr(2):\n"
        f"{ind}    fbwd_lse_frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)\n"
        + lse
        + fbwd_p_sub_loop(ind + "    ", hook, spec)
    )


def fbwd_p_passes(
    p_pass: Callable[[FlexHookCode | None, str], str],
    *,
    masked: FlexHookCode | None,
    unmasked: FlexHookCode | None,
) -> str:
    """A step's P pass(es) at the step-loop body's indentation: with a mask
    hook, its pass on masked steps and the plain pass otherwise; else the
    plain pass alone."""
    if masked is None:
        return p_pass(unmasked, " " * 12)
    return (
        "            if fbwd_masked != 0:\n"
        + p_pass(masked, " " * 16)
        + "            else:\n"
        + p_pass(unmasked, " " * 16)
    )
