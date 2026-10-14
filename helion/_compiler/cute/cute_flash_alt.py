"""Device body of the alternating-warpgroup flash-attention family (``fa4_alt``).

One 128-row query tile per persistent work item. Warp roles (16 warps):

* warps 0-3 / 4-7: softmax warpgroups A / B. A owns the even KV steps on the
  score buffer S_a and the probability buffer P_a, B the odd steps on S_b/P_b.
  Each step: wait S full, read the whole row into registers (one TMEM read,
  then release the buffer to the QK warp), fold the local max, receive the
  previous step's effective max from the other warpgroup (smem + named barrier,
  strictly alternating), publish this step's effective max, run the fa4
  chunked exp/convert/store pass from the registers into P_x (3/4 + 1/4 staged
  arrivals), hand alpha to the correction warps after the P-buffer wait, and
  fold the chunk sums into a partial row sum rescaled by every alpha (the other
  warpgroup's alpha is recomputed exactly from the two effective maxes). A
  hands its partial sum to B once per item; B owns the final row sum and LSE.
  Named-barrier handoffs (bar.arrive by the producer, bar.sync by the consumer)
  are only safe when the producer cannot arrive twice before one sync. The max
  handoffs alternate by construction and the alpha handoffs are ordered by the
  P-buffer wait, but with one KV pair per item nothing orders A's next item
  behind B's end of the current one, so A waits on the ``l_consumed`` mbarrier
  (B's read of the previous item's partial sum) before rewriting the slot and
  arriving.
* warps 8-11: correction. Rescale the single O by alpha(i) before PV(i) (after
  PV(i-1) completed), and stage the normalized output tile into the finished
  item's dead Q stage.
* warp 12: TMEM allocator and the QK stream: QK(i) into S_x as soon as K(i)
  landed and S_x was read.
* warp 13: the PV stream: PV(i) from P_x into O once P(i) is published and O
  rescaled; commits pv_done_x and o_full.
* warp 14: loader. Q (two stages, the next item's tile prefetched mid-item),
  separate K (``kv_stage`` deep) and V (two deep) rings; K runs two tiles ahead
  of V, and the next item's first two K tiles precede this item's last two V
  tiles.
* warp 15: TMA store of the staged output tile; releases the Q stage.

TMEM (512 columns at head_dim 128): S_a [0,128) S_b [128,256) O [256,384)
P_a [384,448) P_b [448,512). The numerics are the fa4 body's per element; the
row sum is accumulated per warpgroup (a reordering within the online-softmax
recurrence), so results differ from the fa4 body at the rounding level only.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .flash_schedule import FlashScheduleSpec
from .flash_schedule import build_fa4_alt_schedule
from .flash_schedule import verify_flash_schedule

if TYPE_CHECKING:
    from .cute_flash import FlashAttentionConfig

FA4_ALT_V_STAGE = 2
FA4_ALT_HEAD_DIMS = (128,)
# The next item's Q is prefetched after this many K/V pairs (its stage frees
# once the previous item's output store read it; prefetching earlier blocks
# the K/V stream behind that store).
_Q_PREFETCH_PAIRS = 4


def fa4_alt_supported(
    *,
    head_dim: int,
    num_kv: int,
    is_causal: bool,
    plain_row_body: bool,
    has_row_epilogue: bool,
    kv_tile_n: int,
) -> bool:
    """Legality of the alternating family for a semantic class."""
    return (
        head_dim in FA4_ALT_HEAD_DIMS
        and num_kv >= 2
        and num_kv % 2 == 0
        and not is_causal
        and plain_row_body
        and not has_row_epilogue
        and kv_tile_n == 128
    )


def _power2_decode(value: str, divisor: int) -> tuple[str, str]:
    if divisor <= 1:
        return "cutlass.Int32(0)", value
    if divisor & (divisor - 1) == 0:
        shift = divisor.bit_length() - 1
        return f"({value} & cutlass.Int32({divisor - 1}))", f"({value} >> {shift})"
    return f"({value} % {divisor})", f"({value} // {divisor})"


class _Src:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, indent: int, text: str = "") -> None:
        self.lines.append(" " * indent + text if text else "")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _item_loop_head(s: _Src, indent: int, total_tiles: int, num_m_tiles: int) -> None:
    m_expr, bh_expr = _power2_decode("flash_tile_id", num_m_tiles)
    s.line(indent, "flash_tile_id = cutlass.Int32(cute.arch.block_idx()[0])")
    s.line(indent, "flash_grid_dim = cutlass.Int32(cute.arch.grid_dim()[0])")
    s.line(indent, f"while flash_tile_id < {total_tiles}:")
    s.line(indent + 4, f"flash_bh = {bh_expr}")
    s.line(indent + 4, f"flash_m_tile = {m_expr}")


def _item_loop_tail(s: _Src, indent: int) -> None:
    s.line(indent + 4, "flash_tile_id = flash_tile_id + flash_grid_dim")


def _tmem_views(s: _Src, indent: int, hd: int) -> None:
    s.line(indent, "flash_qk_acc_shape = flash_qkt.partition_shape_C((128, 128))")
    s.line(indent, "tStS = flash_qkt.make_fragment_C(flash_qk_acc_shape)")
    s.line(indent, f"flash_pv_acc_shape = flash_pvt.partition_shape_C((128, {hd}))")
    s.line(indent, "tOtO = flash_pvt.make_fragment_C(flash_pv_acc_shape)")
    s.line(indent, "_helion_flash_rt.named_barrier_wait_unaligned(2, 14 * 32)")
    s.line(indent, "flash_tmem_ptr = flash_tmem.retrieve_ptr(cutlass.Float32)")
    s.line(indent, "tStSa = cute.make_tensor(flash_tmem_ptr, tStS.layout)")
    s.line(indent, "tStSb = cute.make_tensor(flash_tmem_ptr + 128, tStS.layout)")
    s.line(indent, "tOtO = cute.make_tensor(flash_tmem_ptr + 256, tOtO.layout)")
    s.line(
        indent, f"tStPa = cute.make_tensor(flash_tmem_ptr + {256 + hd}, tStS.layout)"
    )
    s.line(
        indent,
        f"tStPb = cute.make_tensor(flash_tmem_ptr + {256 + hd + 64}, tStS.layout)",
    )


def emit_flash_fa4_alt_device_body(
    *,
    head_dim: int,
    num_kv: int,
    sequence_extent: int,
    total_tiles: int,
    cfg: FlashAttentionConfig,
    has_lse: bool,
    io_dtype: str,
    lse_scale: float,
    exp2_pair_batch: int,
    exp2_emu_batch: int,
    exp2_degree2: bool,
    exp2_degree1: bool,
    e2e_freq: int,
    e2e_res: int,
    kv_group: int = 1,
) -> list[ast.stmt]:
    """Emit the alternating-warpgroup device body (see the module docstring).

    ``kv_group`` is the grouped-query divisor: K/V carry ``batch // kv_group``
    entries and are read at batch head ``flash_bh // kv_group``.
    """
    hd = head_dim
    assert hd in FA4_ALT_HEAD_DIMS
    assert num_kv >= 2 and num_kv % 2 == 0
    assert sequence_extent % 128 == 0
    num_m_tiles = sequence_extent // 128
    k_stage = cfg.kv_stage
    v_stage = FA4_ALT_V_STAGE
    assert 2 <= k_stage <= 3
    verify_flash_schedule(
        build_fa4_alt_schedule(
            FlashScheduleSpec(
                head_dim=hd,
                kv_depth=k_stage,
                v_depth=v_stage,
                query_slots_per_cta=1,
                alternating_warpgroups=True,
                persistent=True,
                kv_iterations=num_kv,
                stage_output=True,
                split_p_arrive=True,
                dtype_bytes=2,
            )
        )
    )
    wait_hint = int(cfg.wait_hint)
    threshold = float(cfg.rescale_threshold)
    pairs = num_kv // 2 - 1
    q_prefetch_pairs = min(_Q_PREFETCH_PAIRS, pairs)
    e2e_offset = int(cfg.e2e_offset)
    pass_kwargs = ""
    if exp2_pair_batch != 1:
        pass_kwargs += f", pair_batch={exp2_pair_batch}, emu_batch={exp2_emu_batch}"
    if exp2_degree1:
        pass_kwargs += ", degree1=True"
    elif exp2_degree2:
        pass_kwargs += ", degree2=True"
    pass_kwargs += f", wait_hint={wait_hint}"
    s = _Src()

    # ------------------------------------------------------------------ setup
    s.line(0, "tidx, _, _ = cute.arch.thread_idx()")
    s.line(0, "warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())")
    s.line(0, "flash_mma_tile_coord_v = cutlass.Int32(0)")
    s.line(0, "flash_local_tidx = tidx % 128")
    s.line(0, "if warp_idx == 0:")
    for name in ("q", "k", "v", "o"):
        s.line(4, f"cute_cpasync_flash.prefetch_descriptor(_flash_tma_{name})")
    s.line(
        0,
        f"_flash_storage_cls = _helion_flash_alt_rt.flash_fa4_alt_shared_storage({hd}, {k_stage}, {v_stage}, {io_dtype})",
    )
    s.line(0, "smem = cutlass_utils_flash.SmemAllocator()")
    s.line(0, "storage = smem.allocate(_flash_storage_cls)")
    s.line(0, "sQ = storage.sQ.get_tensor(_flash_qsl.outer, swizzle=_flash_qsl.inner)")
    s.line(0, "sK = storage.sK.get_tensor(_flash_ksl.outer, swizzle=_flash_ksl.inner)")
    s.line(0, "sV = storage.sV.get_tensor(_flash_vsl.outer, swizzle=_flash_vsl.inner)")
    s.line(0, "flash_scale_t = storage.sScale.get_tensor(cute.make_layout((2, 128)))")
    s.line(0, "flash_m_t = storage.sM.get_tensor(cute.make_layout((2, 128)))")
    s.line(0, "flash_l_t = storage.sL.get_tensor(cute.make_layout((128,)))")
    for name in (
        "s_full",
        "s_empty",
        "pfor",
        "pfor2",
        "pv_done",
        "o_full",
        "corr_epi_full",
        "l_consumed",
        "rowsum_consumed",
        "tmem_dealloc",
    ):
        s.line(0, f"flash_{name}_ptr = storage.{name}_mbar.data_ptr()")
    s.line(0, "if tidx == 0:")
    s.line(4, "for flash_st in cutlass.range_constexpr(2):")
    s.line(8, "cute.arch.mbarrier_init(flash_s_full_ptr + flash_st, 1)")
    s.line(8, "cute.arch.mbarrier_init(flash_s_empty_ptr + flash_st, 4)")
    s.line(8, "cute.arch.mbarrier_init(flash_pfor_ptr + flash_st, 256)")
    s.line(8, "cute.arch.mbarrier_init(flash_pfor2_ptr + flash_st, 128)")
    s.line(8, "cute.arch.mbarrier_init(flash_pv_done_ptr + flash_st, 1)")
    s.line(4, "cute.arch.mbarrier_init(flash_o_full_ptr, 1)")
    s.line(4, "cute.arch.mbarrier_init(flash_corr_epi_full_ptr, 128)")
    s.line(4, "cute.arch.mbarrier_init(flash_l_consumed_ptr, 128)")
    s.line(4, "cute.arch.mbarrier_init(flash_rowsum_consumed_ptr, 128)")
    s.line(0, "cute.arch.mbarrier_init_fence()")
    s.line(0, "cute.arch.sync_threads()")
    s.line(
        0,
        "flash_tmem_user_bar = cutlass_pipeline_flash.NamedBarrier(barrier_id=2, num_threads=14 * 32)",
    )
    s.line(
        0,
        "flash_tmem = cutlass_utils_flash.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=flash_tmem_user_bar, allocator_warp_id=12, is_two_cta=False, two_cta_tmem_dealloc_mbar_ptr=flash_tmem_dealloc_ptr)",
    )
    s.line(
        0,
        f"flash_q_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_qsl, mode=[0, 1, 2]))",
    )
    s.line(
        0,
        f"flash_k_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_ksl, mode=[0, 1, 2]))",
    )
    s.line(
        0,
        f"flash_v_bytes = cute.size_in_bytes({io_dtype}, cute.select(_flash_vsl, mode=[0, 1, 2]))",
    )
    for name, nbytes, bar, stages in (
        ("q", "flash_q_bytes", "q_mbar_ptr", 2),
        ("k", "flash_k_bytes", "k_mbar_ptr", k_stage),
        ("v", "flash_v_bytes", "v_mbar_ptr", v_stage),
    ):
        s.line(
            0,
            f"flash_{name}_prod, flash_{name}_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={stages}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count={nbytes}, barrier_storage=storage.{bar}.data_ptr()).make_participants()",
        )
    s.line(0, "flash_qkt = _flash_qk_mma.get_slice(flash_mma_tile_coord_v)")
    s.line(0, "flash_pvt = _flash_pv_mma.get_slice(flash_mma_tile_coord_v)")
    s.line(
        0,
        f"gQ = cute.flat_divide(_flash_mQt, cute.select((128, 128, {hd}), mode=[0, 2]))",
    )
    s.line(
        0,
        f"gK = cute.flat_divide(_flash_mKt, cute.select((128, 128, {hd}), mode=[1, 2]))",
    )
    s.line(
        0,
        f"gV = cute.flat_divide(_flash_mVt, cute.select((128, {hd}, 128), mode=[1, 2]))",
    )
    s.line(0, "tSgQ = flash_qkt.partition_A(gQ)")
    s.line(0, "tSgK = flash_qkt.partition_B(gK)")
    s.line(0, "tOgV = flash_pvt.partition_B(gV)")
    s.line(
        0,
        "tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(_flash_tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))",
    )
    s.line(
        0,
        "tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(_flash_tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))",
    )
    s.line(
        0,
        "tVsV, tVgV_dkl = cute_cpasync_flash.tma_partition(_flash_tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tOgV, 0, 3))",
    )
    s.line(
        0,
        "# the finished item's dead Q stage stages its output tile for the TMA store (same bytes, epilogue layout)",
    )
    s.line(
        0,
        "sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, _flash_osl.inner), _flash_osl.outer)",
    )
    s.line(
        0,
        f"gO_tma = cute.flat_divide(_flash_mOt, cute.select((128, {hd}, 128), mode=[0, 1]))",
    )
    s.line(0, "tOgO_tma_mma = flash_pvt.partition_C(gO_tma)")
    s.line(
        0,
        "tOsO_tma, tOgO_tma = cute_cpasync_flash.tma_partition(_flash_tma_o, 0, cute.make_layout(1), cute.group_modes(sO, 0, 2), cute.group_modes(tOgO_tma_mma, 0, 3))",
    )

    # ------------------------------------------------- warp 15: output store
    s.line(0, "if warp_idx == 15:")
    s.line(4, f"cute.arch.setmaxregister_decrease({cfg.other_regs})")
    s.line(4, "flash_corr_epi_full_phase = cutlass.Int32(0)")
    s.line(4, "flash_q_stage = cutlass.Int32(0)")
    _item_loop_head(s, 4, total_tiles, num_m_tiles)
    s.line(8, "flash_q_full = flash_q_cons.wait_and_advance()")
    s.line(
        8,
        f"_helion_flash_rt.mbar_spin_wait(flash_corr_epi_full_ptr, flash_corr_epi_full_phase, {wait_hint})",
    )
    s.line(8, "flash_corr_epi_full_phase ^= 1")
    s.line(8, "with cute.arch.elect_one():")
    s.line(
        12,
        "cute.copy(_flash_tma_o, tOsO_tma[None, flash_q_stage], tOgO_tma[None, flash_m_tile, 0, flash_bh])",
    )
    s.line(12, "cute.arch.cp_async_bulk_commit_group()")
    s.line(12, "cute.arch.cp_async_bulk_wait_group(0, read=True)")
    s.line(8, "flash_q_full.release()")
    s.line(8, "flash_q_stage ^= 1")
    _item_loop_tail(s, 4)

    # --------------------------------------------------------- warp 14: loads
    def q_load(ind: int, tq: str, m: str) -> None:
        s.line(ind, "flash_qe = flash_q_prod.acquire_and_advance()")
        s.line(
            ind,
            f"cute.copy(_flash_tma_q, {tq}[None, {m}], tQsQ[None, flash_qe.index], tma_bar_ptr=flash_qe.barrier)",
        )

    def k_load(ind: int, tk: str, kv: str) -> None:
        s.line(ind, "flash_ke = flash_k_prod.acquire_and_advance()")
        s.line(
            ind,
            f"cute.copy(_flash_tma_k, {tk}[None, {kv}], tKsK[None, flash_ke.index], tma_bar_ptr=flash_ke.barrier)",
        )

    def v_load(ind: int, kv: str) -> None:
        s.line(ind, "flash_ve = flash_v_prod.acquire_and_advance()")
        s.line(
            ind,
            f"cute.copy(_flash_tma_v, tVgV[None, {kv}], tVsV[None, flash_ve.index], tma_bar_ptr=flash_ve.barrier)",
        )

    def kv_pairs(ind: int, start: int, stop: int) -> None:
        if start >= stop:
            return
        s.line(ind, f"for flash_j in cutlass.range({start}, {stop}, unroll=1):")
        s.line(ind + 4, "flash_kv0 = 2 * flash_j")
        k_load(ind + 4, "tKgK", "flash_kv0 + 2")
        v_load(ind + 4, "flash_kv0")
        k_load(ind + 4, "tKgK", "flash_kv0 + 3")
        v_load(ind + 4, "flash_kv0 + 1")

    def kv_bh(bh: str) -> str:
        return bh if kv_group == 1 else f"{bh} // {kv_group}"

    def load_item(ind: int, with_next: bool) -> None:
        m_expr, bh_expr = _power2_decode("flash_tile_id", num_m_tiles)
        s.line(ind, f"flash_bh = {bh_expr}")
        s.line(ind, f"tKgK = tKgK_kdl[None, None, 0, {kv_bh('flash_bh')}]")
        s.line(ind, f"tVgV = tVgV_dkl[None, 0, None, {kv_bh('flash_bh')}]")
        if with_next:
            m_next, bh_next = _power2_decode("flash_tile_next", num_m_tiles)
            s.line(ind, "flash_tile_next = flash_tile_id + flash_grid_dim")
            s.line(ind, f"flash_bh_next = {bh_next}")
            s.line(ind, f"flash_m_next = {m_next}")
            s.line(ind, "tQgQn = tQgQ_qdl[None, None, 0, flash_bh_next]")
            s.line(ind, f"tKgKn = tKgK_kdl[None, None, 0, {kv_bh('flash_bh_next')}]")
        if pairs == 0:
            # a two-step item: its two V tiles first (nothing else to overlap
            # them with), then the next item's Q and first two K tiles, whose
            # stages free behind this item's output store and QKs
            v_load(ind, "0")
            v_load(ind, "1")
            if with_next:
                q_load(ind, "tQgQn", "flash_m_next")
                k_load(ind, "tKgKn", "0")
                k_load(ind, "tKgKn", "1")
            return
        kv_pairs(ind, 0, q_prefetch_pairs)
        if with_next:
            s.line(
                ind,
                "# the next item's Q: its stage frees once the previous item's output store read it",
            )
            q_load(ind, "tQgQn", "flash_m_next")
        kv_pairs(ind, q_prefetch_pairs, pairs)
        if with_next:
            s.line(
                ind,
                "# the next item's first two K tiles follow this item's last K tile in the K ring",
            )
            k_load(ind, "tKgKn", "0")
            k_load(ind, "tKgKn", "1")
        v_load(ind, f"{num_kv - 2}")
        v_load(ind, f"{num_kv - 1}")

    m_expr, bh_expr = _power2_decode("flash_tile_id", num_m_tiles)
    s.line(0, "if warp_idx == 14:")
    s.line(4, f"cute.arch.setmaxregister_decrease({cfg.other_regs})")
    s.line(4, "flash_tile_id = cutlass.Int32(cute.arch.block_idx()[0])")
    s.line(4, "flash_grid_dim = cutlass.Int32(cute.arch.grid_dim()[0])")
    s.line(
        4,
        f"flash_tile_count = ({total_tiles} - flash_tile_id + flash_grid_dim - cutlass.Int32(1)) // flash_grid_dim",
    )
    s.line(4, "# first item: Q, K0, K1")
    s.line(4, f"flash_bh = {bh_expr}")
    s.line(4, f"flash_m_tile = {m_expr}")
    s.line(4, "tQgQ = tQgQ_qdl[None, None, 0, flash_bh]")
    s.line(4, f"tKgK = tKgK_kdl[None, None, 0, {kv_bh('flash_bh')}]")
    q_load(4, "tQgQ", "flash_m_tile")
    k_load(4, "tKgK", "0")
    k_load(4, "tKgK", "1")
    s.line(4, "for flash_item in cutlass.range(flash_tile_count - 1, unroll=1):")
    load_item(8, True)
    s.line(8, "flash_tile_id = flash_tile_next")
    s.line(4, "# last item (no successor)")
    load_item(4, False)
    s.line(4, "flash_k_prod.tail()")
    s.line(4, "flash_v_prod.tail()")
    s.line(4, "flash_q_prod.tail()")

    # ------------------------------------------------------ warp 12: QK stream
    s.line(0, "if warp_idx == 12:")
    s.line(4, f"cute.arch.setmaxregister_decrease({cfg.other_regs})")
    s.line(4, "flash_tmem.allocate(512)")
    _tmem_views(s, 4, hd)
    s.line(4, "tSrQ = flash_qkt.make_fragment_A(sQ)")
    s.line(4, "tSrK = flash_qkt.make_fragment_B(sK)")
    s.line(4, "flash_sa_addr = tStSa.iterator.toint()")
    s.line(4, "flash_sb_addr = tStSb.iterator.toint()")
    s.line(
        4,
        "flash_q_smem_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQ, _helion_flash_ptx.Major.K)",
    )
    s.line(
        4,
        "flash_k_smem_base = _helion_flash_ptx.smem_desc_base_from_tensor(sK, _helion_flash_ptx.Major.K)",
    )
    s.line(
        4,
        "_helion_flash_ptx.declare_ptx_idesc(_flash_qk_mma.op, 'helion_flash_qk_mma_idesc')",
    )
    s.line(4, "flash_se_phase = cutlass.Int32(1)")
    _item_loop_head(s, 4, total_tiles, num_m_tiles)
    s.line(8, "flash_q_full = flash_q_cons.wait_and_advance()")
    s.line(
        8,
        "flash_q_start = _helion_flash_ptx.make_smem_desc_start_addr(sQ[None, None, None, flash_q_full.index].iterator)",
    )
    s.line(8, "for flash_j in cutlass.range(_flash_num_kv_tiles // 2, unroll=1):")
    for buf, s_addr in ((0, "flash_sa_addr"), (1, "flash_sb_addr")):
        s.line(12, "flash_k_full = flash_k_cons.wait_and_advance()")
        s.line(
            12,
            f"_helion_flash_rt.mbar_spin_wait(flash_s_empty_ptr + {buf}, flash_se_phase, {wait_hint})",
        )
        s.line(
            12,
            f"_helion_flash_ptx.gemm_ptx_dyn_qk({s_addr}, flash_q_start, flash_q_smem_base, tSrQ[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(sK[None, None, None, flash_k_full.index].iterator), flash_k_smem_base, tSrK[None, None, None, 0].layout, 'helion_flash_qk_mma_idesc', zero_init=True)",
        )
        s.line(12, "with cute.arch.elect_one():")
        s.line(16, f"cute_tcgen05_flash.commit(flash_s_full_ptr + {buf})")
        s.line(12, "flash_k_full.release()")
    s.line(12, "flash_se_phase ^= 1")
    _item_loop_tail(s, 4)
    s.line(4, "flash_tmem.relinquish_alloc_permit()")
    s.line(4, "_helion_flash_rt.named_barrier_wait_unaligned(2, 14 * 32)")
    s.line(4, "flash_tmem.free(flash_tmem_ptr)")

    # ------------------------------------------------------ warp 13: PV stream
    s.line(0, "if warp_idx == 13:")
    s.line(4, f"cute.arch.setmaxregister_decrease({cfg.other_regs})")
    _tmem_views(s, 4, hd)
    s.line(4, "tOrV = flash_pvt.make_fragment_B(sV)")
    s.line(4, "tP = cute.make_tensor(tStSa.iterator, _flash_ptl.outer)")
    s.line(4, "tOrP = flash_pvt.make_fragment_A(tP)")
    s.line(4, "flash_o_addr = tOtO.iterator.toint()")
    s.line(4, "flash_pa_addr = tStPa.iterator.toint()")
    s.line(4, "flash_pb_addr = tStPb.iterator.toint()")
    s.line(
        4,
        "flash_v_smem_base = _helion_flash_ptx.smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.MN)",
    )
    s.line(
        4,
        "_helion_flash_ptx.declare_ptx_idesc(_flash_pv_mma.op, 'helion_flash_pv_mma_idesc')",
    )
    s.line(4, "flash_pfor_phase = cutlass.Int32(0)")
    _item_loop_head(s, 4, total_tiles, num_m_tiles)
    s.line(8, "flash_o_zero = cutlass.Boolean(True)")
    s.line(8, "for flash_j in cutlass.range(_flash_num_kv_tiles // 2, unroll=1):")
    for buf, p_addr in ((0, "flash_pa_addr"), (1, "flash_pb_addr")):
        s.line(12, "flash_v_full = flash_v_cons.wait_and_advance()")
        s.line(
            12,
            f"_helion_flash_rt.mbar_spin_wait(flash_pfor_ptr + {buf}, flash_pfor_phase, {wait_hint})",
        )
        s.line(
            12,
            f"_helion_flash_ptx.gemm_ptx_precomputed_pv_ts(flash_o_addr, {p_addr}, _helion_flash_ptx.make_smem_desc_start_addr(sV[None, None, None, flash_v_full.index].iterator), flash_v_smem_base, tOrP[None, None, None, 0].layout, tOrV[None, None, None, 0].layout, 'helion_flash_pv_mma_idesc', mbar_ptr=flash_pfor2_ptr + {buf}, mbar_phase=flash_pfor_phase, zero_init=flash_o_zero, wait_hint={wait_hint})",
        )
        s.line(12, "flash_o_zero = cutlass.Boolean(False)")
        s.line(12, "with cute.arch.elect_one():")
        s.line(16, f"cute_tcgen05_flash.commit(flash_pv_done_ptr + {buf})")
        s.line(12, "flash_v_full.release()")
    s.line(12, "flash_pfor_phase ^= 1")
    s.line(8, "with cute.arch.elect_one():")
    s.line(12, "cute_tcgen05_flash.commit(flash_o_full_ptr)")
    _item_loop_tail(s, 4)
    s.line(4, "_helion_flash_rt.named_barrier_arrive_unaligned(2, 14 * 32)")

    # ------------------------------------------- warps 0-3 / 4-7: softmax A/B
    for x in (0, 1):
        y = 1 - x
        tag = "ab"[x]
        bar_pub, bar_rcv = 11 + x, 11 + y
        s.line(
            0, "if warp_idx < 4:" if x == 0 else "if (warp_idx >= 4) & (warp_idx < 8):"
        )
        s.line(4, f"cute.arch.setmaxregister_increase({cfg.softmax_regs})")
        _tmem_views(s, 4, hd)
        s.line(4, "cS = cute.make_identity_tensor((128, 128))")
        s.line(4, "tScS = flash_qkt.partition_C(cS)")
        s.line(
            4,
            "flash_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)",
        )
        s.line(
            4,
            f"flash_tiled_ld = cute_tcgen05_flash.make_tmem_copy(flash_ld_atom, tStS{tag})",
        )
        s.line(4, "flash_thr_ld = flash_tiled_ld.get_slice(flash_local_tidx)")
        s.line(4, f"tLDtS = flash_thr_ld.partition_S(tStS{tag})")
        s.line(4, "tLDcS = flash_thr_ld.partition_D(tScS)")
        s.line(
            4, f"flash_tilePlikeFP32 = 128 // cutlass.Float32.width * {io_dtype}.width"
        )
        s.line(
            4,
            "flash_P_layout = cute.composition(tStS.layout, cute.make_layout((128, flash_tilePlikeFP32)))",
        )
        s.line(4, f"tStP_P = cute.make_tensor(tStP{tag}.iterator, flash_P_layout)")
        s.line(
            4,
            "flash_tScS_P_layout = cute.composition(tScS.layout, cute.make_layout((128, flash_tilePlikeFP32)))",
        )
        s.line(4, "tScS_P = cute.make_tensor(tScS.iterator, flash_tScS_P_layout)")
        s.line(
            4,
            "flash_st_atom = cute.make_copy_atom(cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(16)), cutlass.Float32)",
        )
        s.line(
            4,
            "flash_tiled_st = cute_tcgen05_flash.make_tmem_copy(flash_st_atom, tStP_P)",
        )
        s.line(4, "flash_thr_st = flash_tiled_st.get_slice(flash_local_tidx)")
        s.line(4, "tSTtP = flash_thr_st.partition_D(tStP_P)")
        s.line(4, "tSTcS = flash_thr_st.partition_S(tScS_P)")
        s.line(4, "flash_P_STORE_CHUNKS = cute.size(tSTtP, mode=[2])")
        s.line(4, "flash_LD_CHUNKS = cute.size(tLDtS, mode=[1])")
        s.line(
            4, "flash_LD_CHUNKS_PER_P_STORE = flash_LD_CHUNKS // flash_P_STORE_CHUNKS"
        )
        s.line(4, "flash_PV_SPLIT_LD_CHUNKS = flash_LD_CHUNKS * 3 // 4")
        s.line(
            4,
            "flash_P_STORE_SPLIT = (flash_PV_SPLIT_LD_CHUNKS + flash_LD_CHUNKS_PER_P_STORE - 1) // flash_LD_CHUNKS_PER_P_STORE",
        )
        s.line(4, "flash_s_full_phase = cutlass.Int32(0)")
        s.line(4, "flash_pvd_phase = cutlass.Int32(1)")
        if x == 0:
            s.line(4, "flash_first_item = cutlass.Boolean(True)")
            # waits for the odd warpgroup's read of the previous item's partial
            # sum; parity 1 passes a fresh barrier for the first item
            s.line(4, "flash_lc_phase = cutlass.Int32(1)")
        else:
            # waits for the correction warps' read of the previous item's row
            # sum out of this warpgroup's alpha slot; parity 1 passes a fresh
            # barrier for the first item
            s.line(4, "flash_rsc_phase = cutlass.Int32(1)")
        _item_loop_head(s, 4, total_tiles, num_m_tiles)
        if x == 1:
            s.line(
                8,
                "# the alpha slot still holds the previous item's row sum until the correction warps read it; the first alpha of this item (written after the P-buffer wait, which the row-sum publish already passed) must not overtake that read or arrive on the handoff barrier twice before one correction sync",
            )
            s.line(
                8,
                f"_helion_flash_rt.mbar_spin_wait(flash_rowsum_consumed_ptr, flash_rsc_phase, {wait_hint})",
            )
            s.line(8, "flash_rsc_phase ^= 1")
        s.line(8, "flash_row_max = cutlass.Float32(-cutlass.Float32.inf)")
        s.line(8, "flash_part_sum = cutlass.Float32(0.0)")
        s.line(8, "flash_alpha = cutlass.Float32(1.0)")
        s.line(8, "for flash_j in cutlass.range(_flash_num_kv_tiles // 2, unroll=1):")
        s.line(12, f"flash_kv = 2 * flash_j + {x}")
        s.line(
            12,
            f"_helion_flash_rt.mbar_spin_wait(flash_s_full_ptr + {x}, flash_s_full_phase, {wait_hint})",
        )
        s.line(12, "flash_s_full_phase ^= 1")
        s.line(
            12,
            f"flash_local_max, flash_frgs = _helion_flash_alt_rt.fa4_alt_rowmax_hold(flash_tiled_ld, tLDtS, tLDcS, flash_LD_CHUNKS, flash_s_empty_ptr + {x})",
        )
        s.line(12, "flash_prev2_max = flash_row_max")
        if x == 0:
            s.line(12, "flash_old_row_max = cutlass.Float32(-cutlass.Float32.inf)")
            s.line(12, "if (flash_j != 0) | (flash_first_item == False):")
            s.line(16, f"_helion_flash_rt.named_barrier_wait_unaligned({bar_rcv}, 256)")
            s.line(12, "if flash_j != 0:")
            s.line(16, f"flash_old_row_max = flash_m_t[{y}, flash_local_tidx]")
        else:
            s.line(12, f"_helion_flash_rt.named_barrier_wait_unaligned({bar_rcv}, 256)")
            s.line(12, f"flash_old_row_max = flash_m_t[{y}, flash_local_tidx]")
        # this step's effective max and its publication first (a multiply and a compare on the path)
        s.line(12, "flash_row_max = cute.arch.fmax(flash_old_row_max, flash_local_max)")
        s.line(12, "flash_row_max_safe = flash_row_max")
        s.line(12, "if flash_row_max == -cutlass.Float32.inf:")
        s.line(16, "flash_row_max_safe = cutlass.Float32(0.0)")
        s.line(
            12,
            "flash_acc_log = _flash_scale_log2 * (flash_old_row_max - flash_row_max_safe)",
        )
        s.line(
            12, f"flash_keep_max = (flash_kv != 0) & (flash_acc_log >= -{threshold!r})"
        )
        s.line(12, "if flash_keep_max:")
        s.line(16, "flash_row_max = flash_old_row_max")
        s.line(16, "flash_row_max_safe = flash_old_row_max")
        s.line(12, f"flash_m_t[{x}, flash_local_tidx] = flash_row_max")
        s.line(12, f"_helion_flash_rt.named_barrier_arrive_unaligned({bar_pub}, 256)")
        # this step's alpha (fa4 formula) and the other warpgroup's previous alpha (same formula)
        s.line(12, "flash_alpha = cute.math.exp2(flash_acc_log, fastmath=True)")
        s.line(12, "if flash_keep_max:")
        s.line(16, "flash_alpha = cutlass.Float32(1.0)")
        s.line(
            12, "flash_minus_max_scale = (0.0 - flash_row_max_safe) * _flash_scale_log2"
        )
        s.line(12, "flash_prev_safe = flash_old_row_max")
        s.line(12, "if flash_old_row_max == -cutlass.Float32.inf:")
        s.line(16, "flash_prev_safe = cutlass.Float32(0.0)")
        s.line(
            12,
            "flash_prev_log = _flash_scale_log2 * (flash_prev2_max - flash_prev_safe)",
        )
        s.line(12, "flash_alpha_prev = cute.math.exp2(flash_prev_log, fastmath=True)")
        s.line(12, f"if (flash_kv > 1) & (flash_prev_log >= -{threshold!r}):")
        s.line(16, "flash_alpha_prev = cutlass.Float32(1.0)")
        if x == 0:
            s.line(12, "if flash_j == 0:")
            s.line(16, "flash_alpha_prev = cutlass.Float32(1.0)")
        s.line(12, "flash_do_handoff = flash_kv != 0")
        s.line(
            12,
            f"flash_p_sum = _helion_flash_alt_rt.fa4_alt_exp_pass_held(flash_frgs, flash_tiled_st, tSTtP, tSTcS, _flash_scale_log2, flash_minus_max_scale, {e2e_freq}, {e2e_res}, {e2e_offset}, flash_pfor_ptr + {x}, flash_pfor2_ptr + {x}, flash_P_STORE_SPLIT, flash_P_STORE_CHUNKS, {io_dtype}, flash_pv_done_ptr + {x}, flash_pvd_phase, flash_do_handoff, flash_scale_t, {x}, flash_local_tidx, flash_alpha, {3 + 4 * x} + warp_idx % 4{pass_kwargs})",
        )
        s.line(12, "flash_pvd_phase ^= 1")
        s.line(
            12,
            "flash_part_sum = flash_part_sum * flash_alpha_prev * flash_alpha + flash_p_sum",
        )
        if x == 0:
            s.line(
                8,
                "# the slot is rewritten (and the handoff barrier arrived on) only after the odd warpgroup read the previous item's partial sum: one even arrival per odd sync",
            )
            s.line(
                8,
                f"_helion_flash_rt.mbar_spin_wait(flash_l_consumed_ptr, flash_lc_phase, {wait_hint})",
            )
            s.line(8, "flash_lc_phase ^= 1")
            s.line(8, "flash_l_t[flash_local_tidx] = flash_part_sum")
            s.line(8, "_helion_flash_rt.named_barrier_arrive_unaligned(13, 256)")
            s.line(8, "flash_first_item = cutlass.Boolean(False)")
        else:
            s.line(8, "_helion_flash_rt.named_barrier_wait_unaligned(13, 256)")
            s.line(
                8,
                "flash_row_sum = flash_l_t[flash_local_tidx] * flash_alpha + flash_part_sum",
            )
            s.line(8, "_helion_flash_rt.mbarrier_arrive(flash_l_consumed_ptr)")
            s.line(
                8,
                "# the last PV's completion implies the correction consumed the last alpha: slot free, barrier in step",
            )
            s.line(
                8,
                f"_helion_flash_rt.mbar_spin_wait(flash_pv_done_ptr + {x}, flash_pvd_phase, {wait_hint})",
            )
            s.line(8, f"flash_scale_t[{x}, flash_local_tidx] = flash_row_sum")
            s.line(
                8,
                f"_helion_flash_rt.named_barrier_arrive_unaligned({3 + 4 * x} + warp_idx % 4, 64)",
            )
            if has_lse:
                value = (
                    "flash_row_max * _flash_scale_log2 + cute.math.log2(flash_row_sum)"
                )
                if abs(lse_scale - 1.0) > 1e-7:
                    value = f"({value}) * cutlass.Float32({lse_scale!r})"
                s.line(
                    8,
                    f"_flash_mLSE[flash_m_tile * 128 + flash_local_tidx, flash_bh] = {value}",
                )
        _item_loop_tail(s, 4)
        s.line(4, "_helion_flash_rt.named_barrier_arrive_unaligned(2, 14 * 32)")

    # --------------------------------------------------- warps 8-11: correction
    s.line(0, "if (warp_idx >= 8) & (warp_idx < 12):")
    s.line(4, f"cute.arch.setmaxregister_decrease({cfg.corr_regs})")
    _tmem_views(s, 4, hd)
    s.line(4, "flash_o_full_phase = cutlass.Int32(0)")
    s.line(4, "flash_pvd_phase0 = cutlass.Int32(0)")
    s.line(4, "flash_pvd_phase1 = cutlass.Int32(0)")
    s.line(4, "flash_q_stage = cutlass.Int32(0)")
    s.line(4, "_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 0)")
    _item_loop_head(s, 4, total_tiles, num_m_tiles)
    s.line(8, "for flash_j in cutlass.range(_flash_num_kv_tiles // 2, unroll=1):")

    def corr_step(ind: int, x: int) -> None:
        other = 1 - x
        s.line(
            ind,
            f"_helion_flash_rt.named_barrier_wait_unaligned({3 + 4 * x} + warp_idx % 4, 64)",
        )
        s.line(ind, f"flash_a = flash_scale_t[{x}, flash_local_tidx]")
        s.line(
            ind, "flash_need_rescale = cute.arch.vote_ballot_sync(flash_a < 1.0) != 0"
        )
        s.line(ind, "if flash_need_rescale:")
        s.line(
            ind + 4,
            "# O = alpha * O only after the previous step's PV finished writing it",
        )
        s.line(
            ind + 4,
            f"_helion_flash_rt.mbar_spin_wait(flash_pv_done_ptr + {other}, flash_pvd_phase{other}, {wait_hint})",
        )
        s.line(
            ind + 4,
            f"_helion_flash_rt.rescale_o_tmem(tOtO, flash_a, flash_local_tidx, {hd}, {cfg.rescale_chunk_cols})",
        )
        s.line(ind + 4, "cute.arch.fence_view_async_tmem_store()")
        s.line(ind, f"flash_pvd_phase{other} ^= 1")
        s.line(ind, f"_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + {x})")

    s.line(12, "if flash_j != 0:")
    corr_step(16, 0)
    corr_step(12, 1)
    s.line(8, "_helion_flash_rt.named_barrier_wait_unaligned(7 + warp_idx % 4, 64)")
    s.line(
        8,
        "flash_inv_sum = _helion_flash_rt.rcp_approx_ftz(flash_scale_t[1, flash_local_tidx])",
    )
    # the odd warpgroup may now write its next item's first alpha into the slot
    s.line(8, "_helion_flash_rt.mbarrier_arrive(flash_rowsum_consumed_ptr)")
    s.line(
        8,
        f"_helion_flash_rt.mbar_spin_wait(flash_o_full_ptr, flash_o_full_phase, {wait_hint})",
    )
    s.line(8, "flash_o_full_phase ^= 1")
    s.line(
        8,
        f"_helion_flash_rt.fa4_correction_epilogue_to_smem_scoped(flash_pvt, tOtO, sO[None, None, flash_q_stage], flash_local_tidx, flash_inv_sum, {hd}, {cfg.corr_tile_size}, {io_dtype})",
    )
    s.line(8, "cute.arch.fence_view_async_shared()")
    s.line(8, "cute.arch.mbarrier_arrive(flash_corr_epi_full_ptr)")
    s.line(8, "flash_q_stage ^= 1")
    s.line(8, "flash_pvd_phase1 ^= 1")
    s.line(8, "_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 0)")
    _item_loop_tail(s, 4)
    s.line(4, "_helion_flash_rt.named_barrier_arrive_unaligned(2, 14 * 32)")
    return list(ast.parse(s.text()).body)
