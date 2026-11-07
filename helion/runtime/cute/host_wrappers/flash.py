"""Host wrapper emitters of the flash-attention families.

The fused tcgen05 flash forward (``helion_flash``: fa4, ws_overlap, the
2-CTA pairs, the flex envelope), the gated attention over jagged rows, the
four backward layouts of ``helion_flash_bwd`` (plain, 2-CTA, flex 2-CTA,
flex pair) and the register-MMA row programs.  Each emitter builds the
tiled MMAs and TMA atoms the device body consumes over the wrapper's tensor
views and appends the kernel arguments in the device body's parameter order.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import cast

from .... import exc
from ..arg_signature import schema_pointer_alignment
from ..occupancy import persistent_cluster_ctas
from .context import HostWrapperContext
from .context import HostWrapperEmitter

if TYPE_CHECKING:
    from collections.abc import Sequence


def flex_rank4_layout(
    rows: int,
    cols: int,
    heads: int,
    batch: int,
    strides: Sequence[int],
    *,
    mn_major: bool = False,
) -> str:
    """A rank-4 TMA map over a ``[B, H, S, D]`` flex operand's strides
    (``flex_match.flex_operand_strides`` form): ``(S, D, H, B)``, or
    ``(D, S, H, B)`` for the MN-major V. The device body indexes the
    trailing ``(head, batch)`` modes."""
    batch_stride, head_stride, row_stride, _ = strides
    if mn_major:
        return (
            f"cute.make_layout(({cols}, {rows}, {heads}, {batch}), "
            f"stride=(1, {row_stride}, {head_stride}, {batch_stride}))"
        )
    return (
        f"cute.make_layout(({rows}, {cols}, {heads}, {batch}), "
        f"stride=({row_stride}, 1, {head_stride}, {batch_stride}))"
    )


def emit_small_biased_attention(
    ctx: HostWrapperContext, plan: dict[str, object]
) -> None:
    """``helion_small_biased_attention``: one CTA per (sequence row, batch)."""
    batch = ctx.plan_int("batch")
    seq = ctx.plan_int("seq")
    ctx.body.extend(
        [
            f"    grid_x = cutlass.Int32({seq})",
            f"    grid_y = cutlass.Int32({batch})",
            "    grid_z = cutlass.Int32(1)",
        ]
    )


def emit_flash_row_mma(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``helion_flash_row_mma``: register-MMA row programs."""
    # Register-MMA row programs: one CTA per (batch*head, row tile), plain
    # tensor arguments, no host-side descriptors. The body moves 16-byte
    # cp.async and st.global.v4 packets from the tensor bases; codegen
    # resolves bindings with an under-aligned base to the tcgen05
    # families, and a wrapper schema that still proves one only 8-byte
    # aligned is refused here rather than faulting on the device.
    if ctx.schema_key is not None:
        aux_keys = tuple(
            f"epi_aux{index}_idx"
            for index in range(ctx.plan_int("epi_aux_count", default=0))
        )
        for key in (
            "q_idx",
            "k_idx",
            "v_idx",
            "o_idx",
            "lse_idx",
            "bias_idx",
            *aux_keys,
        ):
            index = ctx.plan_optional_int(key)
            if index is not None and (
                schema_pointer_alignment(ctx.schema_key[index]) % 16
            ):
                raise exc.BackendUnsupported(
                    "cute",
                    "row_mma flash requires 16-byte-aligned q/k/v/o/lse/bias "
                    "and row-epilogue aux bases",
                )
    ctx.body.extend(
        [
            f"    grid_x = cutlass.Int32({ctx.plan_int('total_tiles')})",
            "    grid_y = cutlass.Int32(1)",
            "    grid_z = cutlass.Int32(1)",
        ]
    )


def emit_flash_gated_row_mma(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``helion_flash_gated_row_mma``: register-MMA gated attention row programs."""
    # Register-MMA gated attention row programs (cute_flash_gated_row_mma):
    # the host grid, plain tensor arguments, no descriptors. The body
    # moves 16-byte cp.async packets of q/k/v rows and 4-byte output
    # packets. An under-aligned q/k/v/o binding is staged through a
    # 16-byte-aligned copy at launch (``STAGED_PLAN_OPERANDS``); a
    # schema that still proves a base under-aligned for its access width
    # is refused here rather than faulting on the device.
    if ctx.schema_key is not None:
        for key, width in (
            ("q_idx", 16),
            ("k_idx", 16),
            ("v_idx", 16),
            ("o_idx", 4),
        ):
            index = ctx.plan_optional_int(key)
            if index is not None and (
                schema_pointer_alignment(ctx.schema_key[index]) % width
            ):
                raise exc.BackendUnsupported(
                    "cute",
                    "the gated row_mma body requires 16-byte-aligned q/k/v "
                    "and a 4-byte-aligned o base",
                )


def emit_flash_bwd_row_mma(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``helion_flash_bwd_row_mma``: register-MMA backward row programs."""
    # Register-MMA backward row programs (cute_flash_bwd_row_mma): one
    # CTA per (KV group, kv_rows) tile, plain tensor arguments, no
    # host-side descriptors. The body moves 16-byte cp.async packets of
    # q/k/v/do and of the fp32 lse/delta rows, reduce-adds fp32 pairs
    # into dq and stores 16-byte dk/dv packets, so a schema that proves
    # any base under-aligned is refused here rather than faulting.
    if ctx.schema_key is not None:
        for key in (
            "q_idx",
            "k_idx",
            "v_idx",
            "do_idx",
            "lse_idx",
            "delta_idx",
            "dq_idx",
            "dk_idx",
            "dv_idx",
        ):
            index = ctx.plan_optional_int(key)
            if index is not None and (
                schema_pointer_alignment(ctx.schema_key[index]) % 16
            ):
                raise exc.BackendUnsupported(
                    "cute",
                    "the row_mma flash backward requires 16-byte-aligned "
                    "q/k/v/do/lse/delta/dq/dk/dv bases",
                )
    ctx.body.extend(
        [
            f"    grid_x = cutlass.Int32({ctx.plan_int('total_tiles')})",
            "    grid_y = cutlass.Int32(1)",
            "    grid_z = cutlass.Int32(1)",
        ]
    )


def emit_flash(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``helion_flash``: the fused tcgen05 flash-attention forward."""
    # Fused tcgen05 flash-attention host setup: reorder Helion's (B, S, D)
    # tensors to the reference (S, D, B) / (D, S, B) layouts, build the two
    # tiled_mma (QK from SMEM, PV with OperandSource.TMEM) and the three TMA
    # atoms, then append all kernel args. This mirrors the standalone
    # 3D-batched host setup validated for the specialized flash path.
    q_idx = ctx.plan_int("q_idx")
    k_idx = ctx.plan_int("k_idx")
    v_idx = ctx.plan_int("v_idx")
    o_idx = ctx.plan_int("o_idx")
    lse_idx = ctx.plan_optional_int("lse_idx")
    # The flex row-max store beside the LSE (the same (S, B) view).
    row_max_idx = ctx.plan_optional_int("row_max_idx")
    bias_idx = ctx.plan_optional_int("bias_idx")
    # The staged bias ring's depth (``cute_flash_bias_stage='tma'``; None
    # or 0: the softmax warps read the bias from global memory).
    bias_stage = ctx.plan_optional_int("bias_stage")
    alibi_idx = ctx.plan_optional_int("alibi_idx")
    document_idx = ctx.plan_optional_int("document_idx")
    row_kv_begin_idx = ctx.plan_optional_int("row_kv_begin_idx")
    row_kv_begin_work_idx = ctx.plan_optional_int("row_kv_begin_work_idx")
    seq = ctx.plan_int("seq")
    # The flex envelope's K/V length (the Q/O/LSE length is ``seq``).
    kv_seq = ctx.plan_int("kv_seq", default=seq)
    # The runtime softmax scale (the flex envelope's scale argument or a
    # dense plan's traced host scale) and the factor that turns it into the
    # exp2-domain scale.
    scale_idx = ctx.plan_optional_int("scale_idx")
    scale_peeled = bool(plan.get("scale_peeled", True))
    scale_log2_multiplier = plan.get("scale_log2_multiplier", math.log2(math.e))
    assert isinstance(scale_log2_multiplier, float)
    head_dim = ctx.plan_int("head_dim")
    # V/O head dim: equals ``head_dim`` except for the asymmetric MLA pair
    # (192-wide Q/K against 128-wide V/O).
    head_dim_v = ctx.plan_int("head_dim_v", default=head_dim)
    uneven_kv_ring = bool(plan.get("uneven_kv_ring"))
    batch = ctx.plan_int("batch")
    # Grouped-query attention: K/V hold ``batch // kv_group`` batch heads
    # (the device body reads them at ``bh // kv_group``); Q/O keep ``batch``.
    kv_group = ctx.plan_int("kv_group", default=1)
    kv_batch = batch // kv_group
    scale_log2 = plan["scale_log2"]
    assert isinstance(scale_log2, float)
    score_bias_scale = plan.get("score_bias_scale", 0.0)
    assert isinstance(score_bias_scale, float)
    alibi_count = ctx.plan_int("alibi_count", default=batch)
    document_batch = ctx.plan_int("document_batch", default=batch)
    document_heads_per_batch = ctx.plan_int("document_heads_per_batch", default=1)
    row_kv_begin_batch = ctx.plan_int("row_kv_begin_batch", default=batch)
    row_kv_begin_heads_per_batch = ctx.plan_int(
        "row_kv_begin_heads_per_batch", default=1
    )
    kv_stage = ctx.plan_int("kv_stage")
    # The K ring may be one slot deeper than V's (``k_stage``).
    k_stage = ctx.plan_int("k_stage", default=kv_stage)
    q_stage = ctx.plan_int("q_stage", default=1)
    use_2cta_instrs = bool(plan.get("use_2cta_instrs"))
    use_cga2_local_cta = bool(plan.get("use_cga2_local_cta"))
    use_clc_scheduler = bool(plan.get("use_clc_scheduler"))
    # The cooperative pair's cluster holds ``pair_cluster`` MMA pairs
    # (two pairs share every K/V tile by TMA multicast across the pairs).
    pair_cluster = ctx.plan_int("pair_cluster", default=1) if use_2cta_instrs else 1
    cluster_m = 2 * pair_cluster if use_2cta_instrs else 2 if use_cga2_local_cta else 1
    # KV tile width. 128 is the historical fixed value; a wider tile
    # amortizes the per-tile softmax correction over more columns, which is
    # where FA4's tuned plans get their edge on sm_103.
    kv_n = ctx.plan_int("kv_tile_n", default=128)
    num_kv = (kv_seq + kv_n - 1) // kv_n
    # Static-persistent scheduler: total_tiles = num_bh * num_m_tiles (the
    # flat tile-id space the device-body strided while loop walks). When
    # persistent, the host clamps grid_x down to min(total_tiles, num_SMs)
    # so each SM gets one CTA that strides over many work tiles.
    persistent = bool(plan.get("persistent"))
    total_tiles = ctx.plan_int("total_tiles", default=batch * (seq // 128))
    # The split-KV list walk (``cute_flash_kv_split``): that many CTAs
    # per pair item (the device body decodes the split id).
    kv_split = ctx.plan_int("kv_split", default=1)
    # The tail split: the last partial wave's items as split units.
    tail_split = ctx.plan_int("flex_tail_split", default=0)
    tail_items_ws = ctx.plan_int("flex_tail_items_ws", default=0)
    pass_dynamic_tile_counts = plan.get("topology") != "fa4"
    hd = head_dim
    hd_v = head_dim_v
    dtype = str(plan.get("dtype", "cutlass.Float16"))
    assert dtype in ("cutlass.Float16", "cutlass.BFloat16", "cutlass.Float8E4M3FN")
    # V's storage: a dense ``[kv_batch, head_dim_v, seq]`` buffer is the
    # K-major (D, S, B) B operand of the PV MMA (the canonical buffer is
    # MN-major); the plan read it from the host tensor's strides.
    v_k_major = bool(plan.get("v_k_major"))
    tensor_4d_batch = ctx.plan_int("tensor_4d_batch", default=0)
    tensor_4d_heads = ctx.plan_int("tensor_4d_heads", default=0)
    use_tensor_4d_tma = (
        tensor_4d_batch > 0
        and tensor_4d_heads > 0
        and tensor_4d_batch * tensor_4d_heads == batch
    )
    # The 4-D maps (and the strided flex maps, which imply them) describe
    # the canonical V alone.
    assert not (v_k_major and use_tensor_4d_tma)
    # (S, D, B) views over the existing (B, S, D) row-major buffers. The
    # dense FA4 4D-TMA knob instead treats the same flat storage as
    # (S, D, H, Z), matching FA4's tensor-map rank for contiguous q[z,h,s,d].
    bw = "cutlass.utils.blackwell_helpers"
    # Query-tile height: the two-warpgroup ws_overlap body can run 64-row
    # (tcgen05 M=64) tiles; fa4 and the 2-CTA families stay at 128/256.
    q_tile_m = ctx.plan_int("q_tile_m", default=128)
    mma_m = 256 if use_2cta_instrs else q_tile_m
    qkd = f"({mma_m}, {kv_n}, {hd})"
    pvd = f"({mma_m}, {hd_v}, {kv_n})"
    operand_strides = plan.get("operand_strides")
    if operand_strides is not None:
        # Strided flex operands: rank-4 (S, D, H, B) maps over the real
        # [B, H, S, D] strides of each operand (unit inner stride, the
        # rest 16-byte aligned; ``flex_match.flex_operand_strides``). The
        # device body indexes them ``[.., head, batch]`` like the dense
        # 4D family; K/V span their own head count and length.
        assert use_tensor_4d_tma
        assert isinstance(operand_strides, (list, tuple))
        q_st, k_st, v_st, o_st = cast(
            "tuple[Sequence[int], ...]", tuple(operand_strides)
        )
        kv_heads = tensor_4d_heads // kv_group
        sdb = flex_rank4_layout(seq, hd, tensor_4d_heads, tensor_4d_batch, q_st)
        sdb_o = flex_rank4_layout(seq, hd_v, tensor_4d_heads, tensor_4d_batch, o_st)
        sdb_kv = flex_rank4_layout(kv_seq, hd, kv_heads, tensor_4d_batch, k_st)
        dsb_kv = flex_rank4_layout(
            kv_seq, hd_v, kv_heads, tensor_4d_batch, v_st, mn_major=True
        )
    else:
        if use_tensor_4d_tma:
            # The 4D descriptors describe Q/K/V/O with one head dim and
            # one sequence length.
            assert hd_v == hd
            assert kv_seq == seq
            bh_stride = seq * hd
            batch_stride = tensor_4d_heads * bh_stride
            sdb = (
                f"cute.make_layout(({seq}, {hd}, {tensor_4d_heads}, "
                f"{tensor_4d_batch}), stride=({hd}, 1, {bh_stride}, "
                f"{batch_stride}))"
            )
            dsb = (
                f"cute.make_layout(({hd}, {seq}, {tensor_4d_heads}, "
                f"{tensor_4d_batch}), stride=(1, {hd}, {bh_stride}, "
                f"{batch_stride}))"
            )
            sdb_o = sdb
        else:
            # Q (and K) are ``hd`` wide; O (and the row-epilogue aux rows
            # that share its view) and the MN-major V are ``hd_v`` wide.
            sdb = (
                f"cute.make_layout(({seq}, {hd}, {batch}), "
                f"stride=({hd}, 1, {seq * hd}))"
            )
            sdb_o = (
                f"cute.make_layout(({seq}, {hd_v}, {batch}), "
                f"stride=({hd_v}, 1, {seq * hd_v}))"
            )
            # V as the (D, S, B) B operand: the canonical ``[B, S, D]``
            # buffer is MN-major (D contiguous), the pre-transposed
            # ``[B, D, S]`` buffer K-major (S contiguous).
            dsb = (
                f"cute.make_layout(({hd_v}, {seq}, {batch}), "
                f"stride=({seq}, 1, {seq * hd_v}))"
                if v_k_major
                else f"cute.make_layout(({hd_v}, {seq}, {batch}), "
                f"stride=(1, {hd_v}, {seq * hd_v}))"
            )
        # K/V layouts span their own (grouped) batch-head count and length.
        if kv_group == 1 and kv_seq == seq:
            sdb_kv = sdb
            dsb_kv = dsb
        else:
            sdb_kv = (
                f"cute.make_layout(({kv_seq}, {hd}, {kv_batch}), "
                f"stride=({hd}, 1, {kv_seq * hd}))"
            )
            dsb_kv = (
                f"cute.make_layout(({hd_v}, {kv_seq}, {kv_batch}), "
                f"stride=({kv_seq}, 1, {kv_seq * hd_v}))"
                if v_k_major
                else f"cute.make_layout(({hd_v}, {kv_seq}, {kv_batch}), "
                f"stride=(1, {hd_v}, {kv_seq * hd_v}))"
            )
    ssb = f"cute.make_layout(({seq}, {seq}, {batch}), stride=({seq}, 1, {seq * seq}))"
    sb = f"cute.make_layout(({seq}, {batch}), stride=(1, {seq}))"
    majk = "cute.nvgpu.OperandMajorMode.K"
    cg1 = "cute.nvgpu.tcgen05.CtaGroup.ONE"
    cg = "cute.nvgpu.tcgen05.CtaGroup.TWO" if use_2cta_instrs else cg1
    # K/V multicast across the two pairs of a four-CTA cluster.
    kv_op = (
        f"cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp({cg})"
        if pair_cluster > 1
        else "_flash_op"
    )
    sel = "cute.select"
    flash_lines = [
        f"_flash_mQ = cute.make_tensor(arg{q_idx}.iterator, {sdb})",
        f"_flash_mK = cute.make_tensor(arg{k_idx}.iterator, {sdb_kv})",
        # V is the (D, S, B) B operand (MN-major unless ``v_k_major``).
        f"_flash_mV = cute.make_tensor(arg{v_idx}.iterator, {dsb_kv})",
        f"_flash_mO = cute.make_tensor(arg{o_idx}.iterator, {sdb_o})",
        f"_flash_qk_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majk}, cutlass.Float32, {cg}, ({mma_m}, {kv_n}))",
        f"_flash_pv_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majk if v_k_major else 'cute.nvgpu.OperandMajorMode.MN'}, cutlass.Float32, {cg}, ({mma_m}, {hd_v}), cute.nvgpu.tcgen05.OperandSource.TMEM)",
        f"_flash_cluster_layout_vmnk = cute.tiled_divide(cute.make_layout(({cluster_m}, 1, 1)), (_flash_qk_mma.thr_id.shape,))",
        f"_flash_qsl = {bw}.make_smem_layout_a(_flash_qk_mma, {qkd}, {dtype}, {q_stage})",
        # K/V are multi-stage TMA rings (Stage 3); the stage count must match
        # the device-body ring depths + the SharedStorage MemRange depths.
        f"_flash_ksl = {bw}.make_smem_layout_b(_flash_qk_mma, {qkd}, {dtype}, {k_stage})",
        f"_flash_vsl = {bw}.make_smem_layout_b(_flash_pv_mma, {pvd}, {dtype}, {kv_stage})",
    ]
    if hd_v != hd:
        # The aliased K/V ring alternates K and V through one physical
        # ring whose slots are K's (wider) tile, so V's stage stride has
        # to be K's: recompose V's swizzled layout over the shared stage
        # stride (FA4's ``same_hdim_kv_padded`` layout fix-up). One V stage
        # still covers only its own ``kv_n x hd_v`` box. FA4's uneven
        # [wide, narrow, wide] ring strides both operands by half a K
        # plus half a V tile instead (the device body shifts the odd slot).
        if uneven_kv_ring:
            stage_stride = (
                "(max(_flash_ksl.outer.stride[-1], 0) "
                "+ max(_flash_vsl.outer.stride[-1], 0)) // 2"
            )
        else:
            stage_stride = (
                "max(max(_flash_ksl.outer.stride[-1], 0), "
                "max(_flash_vsl.outer.stride[-1], 0))"
            )
        flash_lines.append(f"_flash_kv_stage_stride = {stage_stride}")
        if uneven_kv_ring:
            flash_lines.append(
                "_flash_ksl = cute.make_composed_layout(_flash_ksl.inner, 0, "
                f"cute.make_layout((*_flash_ksl.outer.shape[:-1], {k_stage}), "
                "stride=(*_flash_ksl.outer.stride[:-1], _flash_kv_stage_stride)))"
            )
        flash_lines.append(
            "_flash_vsl = cute.make_composed_layout(_flash_vsl.inner, 0, "
            f"cute.make_layout((*_flash_vsl.outer.shape[:-1], {kv_stage}), "
            "stride=(*_flash_vsl.outer.stride[:-1], _flash_kv_stage_stride)))"
        )
    flash_lines += [
        f"_flash_ptl = {bw}.make_smem_layout_a(_flash_pv_mma, {pvd}, {dtype}, 1)",
        f"_flash_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp({cg})",
        f"_flash_tma_q, _flash_mQt = cute.nvgpu.make_tiled_tma_atom_A(_flash_op, _flash_mQ, {sel}(_flash_qsl, mode=[0, 1, 2]), {qkd}, _flash_qk_mma, _flash_cluster_layout_vmnk.shape)",
        f"_flash_tma_k, _flash_mKt = cute.nvgpu.make_tiled_tma_atom_B({kv_op}, _flash_mK, {sel}(_flash_ksl, mode=[0, 1, 2]), {qkd}, _flash_qk_mma, _flash_cluster_layout_vmnk.shape)",
        f"_flash_tma_v, _flash_mVt = cute.nvgpu.make_tiled_tma_atom_B({kv_op}, _flash_mV, {sel}(_flash_vsl, mode=[0, 1, 2]), {pvd}, _flash_pv_mma, _flash_cluster_layout_vmnk.shape)",
        f"_flash_scale_log2 = cutlass.Float32({scale_log2!r})",
        f"_flash_num_kv_tiles = cutlass.Int32({num_kv})",
    ]
    if scale_idx is not None:
        # The flex envelope's runtime scale: the hooks read it as
        # ``_flash_scale``; a mask-only plan (raw QK units, the scale
        # peeled into the PASS2 multiplier) derives the log2 scale from
        # it and needs it positive (a non-positive scale would flip the
        # row max), while a score region owns the multiply itself.
        flash_lines.append(f"_flash_scale = cutlass.Float32(arg{scale_idx})")
        if scale_peeled:
            # A non-positive scale would flip the row max; the example's
            # host wrapper asserts ``scale > 0`` (a staged value cannot be
            # asserted inside the DSL wrapper).
            flash_lines.append(
                "_flash_scale_log2 = _flash_scale"
                f" * cutlass.Float32({scale_log2_multiplier!r})"
            )
    if bias_idx is not None:
        flash_lines.extend(
            [
                f"_flash_mBias = cute.make_tensor(arg{bias_idx}.iterator, {ssb})",
                f"_flash_score_bias_scale = cutlass.Float32({score_bias_scale!r})",
            ]
        )
        if bias_stage:
            # The staged bias: (128, 64) boxes of the (m, n, bh) view in
            # the epilogue tile's swizzled smem layout, ``bias_stage``
            # stages of both query tiles' two halves of a KV tile.
            flash_lines.extend(
                [
                    (
                        f"_flash_bsl = {bw}.make_smem_layout_epi("
                        f"{dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, "
                        f"(128, 64), {bias_stage * q_stage * 2})"
                    ),
                    (
                        "_flash_bias_cta_v = cute.composition("
                        "cute.make_identity_layout(_flash_mBias.shape), (128, 64))"
                    ),
                    (
                        "_flash_tma_bias, _flash_mBiasT = "
                        "cute.nvgpu.cpasync.make_tiled_tma_atom("
                        "cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), _flash_mBias, "
                        "cute.select(_flash_bsl, mode=[0, 1]), _flash_bias_cta_v)"
                    ),
                ]
            )
    if alibi_idx is not None:
        flash_lines.extend(
            [
                (
                    f"_flash_mAlibi = cute.make_tensor(arg{alibi_idx}.iterator, "
                    f"cute.make_layout(({alibi_count},), stride=(1,)))"
                ),
                f"_flash_num_alibi = cutlass.Int32({alibi_count})",
            ]
        )
    if document_idx is not None:
        sdoc = f"cute.make_layout(({seq}, {document_batch}), stride=(1, {seq}))"
        # Host-built ``(2, doc_batch, num_tiles)`` int32 [min, max] id table,
        # viewed as ``(num_tiles, doc_batch, 2)`` so the device indexes it
        # ``[tile, doc_bh, 0 | 1]``.
        document_range_idx = ctx.plan_int("document_range_idx")
        document_tiles = ctx.plan_int("document_tiles")
        srange = (
            f"cute.make_layout(({document_tiles}, {document_batch}, 2), "
            f"stride=(1, {document_tiles}, {document_tiles * document_batch}))"
        )
        flash_lines.extend(
            [
                f"_flash_mDoc = cute.make_tensor(arg{document_idx}.iterator, {sdoc})",
                (
                    "_flash_doc_heads_per_batch = "
                    f"cutlass.Int32({document_heads_per_batch})"
                ),
                (
                    "_flash_mDocRange = cute.make_tensor("
                    f"arg{document_range_idx}.iterator, {srange})"
                ),
            ]
        )
    if row_kv_begin_idx is not None:
        # Per-row KV lower-bound table, viewed like the document table.
        srkb = f"cute.make_layout(({seq}, {row_kv_begin_batch}), stride=(1, {seq}))"
        flash_lines.extend(
            [
                (
                    "_flash_mRowKvBegin = cute.make_tensor("
                    f"arg{row_kv_begin_idx}.iterator, {srkb})"
                ),
                (
                    "_flash_rkb_heads_per_batch = "
                    f"cutlass.Int32({row_kv_begin_heads_per_batch})"
                ),
            ]
        )
        if row_kv_begin_work_idx is not None:
            # Host-built ``(rows, 4)`` int32 work table of the segment
            # tiling: row 0 holds the item count, the rest the
            # (batch row, segment start, segment length, pair code) items.
            work_rows = ctx.plan_int("row_kv_begin_work_rows")
            flash_lines.append(
                "_flash_mRkbWork = cute.make_tensor("
                f"arg{row_kv_begin_work_idx}.iterator, "
                f"cute.make_layout(({work_rows}, 4), stride=(4, 1)))"
            )
    # Fused row-epilogue inputs share O's (S, D, B) view so the device body
    # can partition them exactly like the O store.
    epi_aux_count = ctx.plan_int("epi_aux_count", default=0)
    for index in range(epi_aux_count):
        aux_idx = ctx.plan_int(f"epi_aux{index}_idx")
        flash_lines.append(
            f"_flash_mEpiAux{index} = cute.make_tensor(arg{aux_idx}.iterator, {sdb_o})"
        )
    if pass_dynamic_tile_counts:
        flash_lines.extend(
            [
                f"_flash_num_bh = cutlass.Int32({batch})",
                f"_flash_total_tiles = cutlass.Int32({total_tiles})",
            ]
        )
    if lse_idx is not None:
        flash_lines.append(
            f"_flash_mLSE = cute.make_tensor(arg{lse_idx}.iterator, {sb})"
        )
    if row_max_idx is not None:
        flash_lines.append(
            f"_flash_mRowMax = cute.make_tensor(arg{row_max_idx}.iterator, {sb})"
        )
    epi_tma = bool(plan.get("epi_tma"))
    epi_stg = bool(plan.get("epi_stg"))
    if epi_tma or epi_stg:
        # Build the O smem layout for epilogue-warp store paths. The TMA
        # variant also builds the O TMA STORE atom; the STG variant reuses
        # the layout but stores with a universal-copy tiled copy in device code.
        # One sO stage per resident Q tile: the fa4 topology drains both
        # Q tiles of a work item, ws_overlap stages its single tile.
        # Per-CTA output tile: 128 rows for fa4 (also under CtaGroup.TWO,
        # where each CTA stores its own half) or the ws_overlap tile height.
        otile = f"({q_tile_m}, {hd_v})"
        # One sO tile shared by both query stages (``cute_flash_o_stage``
        # = 1) keeps the layout at one stage; the device indexes stage 0.
        o_stages = 1 if plan.get("o_stage") == 1 else q_stage
        flash_lines.extend(
            [
                (
                    f"_flash_osl = {bw}.make_smem_layout_epi("
                    f"{dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, {otile}, "
                    f"{o_stages})"
                ),
            ]
        )
        if epi_tma:
            flash_lines.extend(
                [
                    (
                        f"_flash_o_cta_v = cute.composition("
                        f"cute.make_identity_layout(_flash_mO.shape), {otile})"
                    ),
                    (
                        "_flash_tma_o, _flash_mOt = "
                        "cute.nvgpu.cpasync.make_tiled_tma_atom("
                        "cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), _flash_mO, "
                        "cute.select(_flash_osl, mode=[0, 1]), _flash_o_cta_v)"
                    ),
                ]
            )
        else:
            flash_lines.append("_flash_mOt = _flash_mO")
        if plan.get("epi_aux_tma"):
            # The fused row epilogue's aux tile is TMA-loaded into the free
            # sO stage with the output tile's smem layout.
            flash_lines.extend(
                [
                    (
                        "_flash_aux_cta_v = cute.composition("
                        f"cute.make_identity_layout(_flash_mEpiAux0.shape), {otile})"
                    ),
                    (
                        "_flash_tma_aux0, _flash_mEpiAux0t = "
                        "cute.nvgpu.cpasync.make_tiled_tma_atom("
                        "cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), _flash_mEpiAux0, "
                        "cute.select(_flash_osl, mode=[0, 1]), _flash_aux_cta_v)"
                    ),
                ]
            )
    else:
        # mO stays the (S, D, B) view (no TMA atom; the epilogue uses
        # autovec_copy straight to gmem).
        flash_lines.append("_flash_mOt = _flash_mO")
    ctx.body.extend(f"    {line}" for line in flash_lines)
    if use_clc_scheduler:
        # CLC launches the full problem grid; the device starts from blockIdx
        # and uses cluster launch control to dynamically steal remaining work.
        clc_heads = ctx.plan_int("clc_heads_per_batch", batch)
        if clc_heads <= 0 or batch % clc_heads != 0:
            clc_heads = batch
        ctx.body.extend(
            [
                f"    grid_x = cutlass.Int32({total_tiles // batch})",
                f"    grid_y = cutlass.Int32({clc_heads})",
                f"    grid_z = cutlass.Int32({batch // clc_heads})",
            ]
        )
    elif persistent:
        # Cap the flat grid at num_SMs (computed host-side from the q tensor's
        # device at wrapper-build time and baked as a literal). grid_y/grid_z
        # stay 1 (already true for the flat flash grid). The device-body
        # strided while loop then covers all total_tiles work items.
        assert ctx.num_sm is not None and ctx.num_sm > 0
        ctas_per_sm = max(1, ctx.plan_int("persistent_ctas_per_sm", 1))
        max_ctas = ((ctx.num_sm * ctas_per_sm) // cluster_m) * cluster_m
        if cluster_m > 2:
            # ``num_sm // cluster_m`` is exact for the CTA pair (every
            # TPC holds one) but over-counts the two-pair cluster: a
            # cluster lives inside one GPC and the GPCs hold uneven TPC
            # counts (a B200's 148 SMs hold 33 four-CTA clusters, not
            # 37), and the surplus persistent clusters form a second
            # hardware wave. The driver's count sizes the grid where the
            # device is known (the SM-count model stays the fallback).
            max_ctas = min(
                max_ctas,
                persistent_cluster_ctas(ctx.device, cluster_m, ctx.num_sm)
                * ctas_per_sm,
            )
        grid_cap = min(total_tiles * kv_split * cluster_m, max_ctas)
        ctx.body.append(f"    grid_x = cutlass.Int32({grid_cap})")
    elif plan.get("topology") == "fa4" and tail_split > 1:
        # The tail split (``cute_flash_flex_tail_split``): the full waves
        # of the device run unsplit, the last partial wave's items as
        # ``tail_split`` split units each (one wave of CTAs); a tail the
        # idle SMs cannot hold, a grid of one wave or less or an unknown
        # device keeps the whole grid unsplit (the kernel reads the first
        # tail unit as an argument).
        from ..flash_flex import flex_tail_base

        tail_base = flex_tail_base(
            total_tiles, ctx.num_sm or 0, tail_split, tail_items_ws
        )
        ctx.body.extend(
            [
                f"    _flash_tail_base = cutlass.Int32({tail_base})",
                f"    grid_x = cutlass.Int32({tail_base + (total_tiles - tail_base) * tail_split})",
            ]
        )
    elif plan.get("topology") == "fa4":
        # The fa4 topology processes a PAIR of adjacent 128-row Q-tiles per
        # CTA, so it needs exactly total_tiles (= batch * seq // 256) CTAs
        # (times the split-KV factor). The default root grid would launch
        # batch * seq // 128; override it to the halved fa4 tile count.
        ctx.body.append(
            f"    grid_x = cutlass.Int32({total_tiles * kv_split * cluster_m})"
        )
    elif use_2cta_instrs:
        # The single-tile body's cooperative pair: ``cluster_m`` CTAs per
        # work item (the flex walk's ragged last pair has a dead second
        # tile the default 128-row root grid does not count).
        ctx.body.append(f"    grid_x = cutlass.Int32({total_tiles * cluster_m})")
    elif q_tile_m != 128:
        # 64-row ws_overlap tiles: one CTA per (bh, 64-row tile), twice the
        # default 128-row root grid.
        ctx.body.append(f"    grid_x = cutlass.Int32({total_tiles})")
    ctx.call_args.extend(
        [
            "_flash_qk_mma",
            "_flash_pv_mma",
            "_flash_tma_q",
            "_flash_mQt",
            "_flash_tma_k",
            "_flash_mKt",
            "_flash_tma_v",
            "_flash_mVt",
            "_flash_mOt",
            "_flash_qsl",
            "_flash_ksl",
            "_flash_vsl",
            "_flash_ptl",
            "_flash_scale_log2",
            "_flash_num_kv_tiles",
        ]
    )
    if pass_dynamic_tile_counts:
        ctx.call_args.extend(["_flash_num_bh", "_flash_total_tiles"])
    if tail_split > 1:
        ctx.call_args.append("_flash_tail_base")
    if lse_idx is not None:
        ctx.call_args.append("_flash_mLSE")
    if row_max_idx is not None:
        ctx.call_args.append("_flash_mRowMax")
    if bias_idx is not None:
        ctx.call_args.extend(["_flash_mBias", "_flash_score_bias_scale"])
        if bias_stage:
            ctx.call_args.extend(["_flash_tma_bias", "_flash_mBiasT", "_flash_bsl"])
    if alibi_idx is not None:
        ctx.call_args.extend(["_flash_mAlibi", "_flash_num_alibi"])
    if document_idx is not None:
        ctx.call_args.extend(
            ["_flash_mDoc", "_flash_doc_heads_per_batch", "_flash_mDocRange"]
        )
    if row_kv_begin_idx is not None:
        ctx.call_args.extend(["_flash_mRowKvBegin", "_flash_rkb_heads_per_batch"])
        if row_kv_begin_work_idx is not None:
            ctx.call_args.append("_flash_mRkbWork")
            if epi_tma:
                # The ragged tiles' vector store needs the plain O view.
                ctx.call_args.append("_flash_mO")
    if scale_idx is not None:
        ctx.call_args.append("_flash_scale")
    if epi_tma:
        ctx.call_args.extend(["_flash_tma_o", "_flash_osl"])
        if plan.get("epi_tma_ragged"):
            # A ragged query length's partial tiles take the vector
            # store over the plain O view.
            ctx.call_args.append("_flash_mO")
    elif epi_stg:
        ctx.call_args.append("_flash_osl")
    ctx.call_args.extend(f"_flash_mEpiAux{index}" for index in range(epi_aux_count))
    if plan.get("epi_aux_tma"):
        ctx.call_args.extend(["_flash_tma_aux0", "_flash_mEpiAux0t"])


def emit_flash_gated(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``helion_flash_gated``: fused tcgen05 gated attention over jagged rows."""
    # Imported here: the gated module pulls in the compiler's device IR,
    # which imports the runtime package this module belongs to.
    from ...._compiler.cute.cute_flash_gated import GATED_KERNEL_PARAMS

    # Fused tcgen05 gated attention over jagged rows: Q/K/V/O are rank-3
    # ``[rows, D, lanes]`` (or ``[lanes, rows, D]``) host tensors. The TMA
    # descriptors cover the whole tensor with its runtime row extent so the
    # device body can ``domain_offset`` to any row base and partial tiles
    # past the end are zero-filled. Layout literals come from the
    # ``arg{i}_shape{d}`` / ``arg{i}_stride{d}`` wrapper bindings.
    q_idx = ctx.plan_int("q_idx")
    k_idx = ctx.plan_int("k_idx")
    v_idx = ctx.plan_int("v_idx")
    o_idx = ctx.plan_int("o_idx")
    hd = ctx.plan_int("head_dim")
    bn = ctx.plan_int("kv_tile")
    bm = ctx.plan_int("q_tile", default=128)
    kv_stage = ctx.plan_int("kv_stage")
    dtype = str(plan.get("dtype", "cutlass.Float16"))
    assert dtype in ("cutlass.Float16", "cutlass.BFloat16", "cutlass.Float32")
    # fp32 tensors run the MMA as tf32 (Helion's default dot precision):
    # the gmem tensors keep Float32, the smem layouts / MMA use TFloat32
    # and the TMA descriptors recast via ``internal_type`` (same width).
    mma_dtype = str(plan.get("mma_dtype", dtype))
    tma_internal = (
        ", internal_type=cutlass.TFloat32" if mma_dtype == "cutlass.TFloat32" else ""
    )

    def rows_layout(idx: int, key: str) -> str:
        row_dim = ctx.plan_int(f"{key}_row_dim")
        lane_dim = ctx.plan_int(f"{key}_lane_dim")
        return (
            f"cute.make_layout((arg{idx}_shape{row_dim}, {hd}, "
            f"arg{idx}_shape{lane_dim}), stride=(arg{idx}_stride{row_dim}, 1, "
            f"arg{idx}_stride{lane_dim}))"
        )

    def cols_layout(idx: int, key: str) -> str:
        row_dim = ctx.plan_int(f"{key}_row_dim")
        lane_dim = ctx.plan_int(f"{key}_lane_dim")
        return (
            f"cute.make_layout(({hd}, arg{idx}_shape{row_dim}, "
            f"arg{idx}_shape{lane_dim}), stride=(1, arg{idx}_stride{row_dim}, "
            f"arg{idx}_stride{lane_dim}))"
        )

    bw = "cutlass.utils.blackwell_helpers"
    qkd = f"({bm}, {bn}, {hd})"
    pvd = f"({bm}, {hd}, {bn})"
    majk = "cute.nvgpu.OperandMajorMode.K"
    cg1 = "cute.nvgpu.tcgen05.CtaGroup.ONE"
    sel = "cute.select"
    gated_lines = [
        f"_flash_mQ = cute.make_tensor(arg{q_idx}.iterator, {rows_layout(q_idx, 'q')})",
        f"_flash_mK = cute.make_tensor(arg{k_idx}.iterator, {rows_layout(k_idx, 'k')})",
        f"_flash_mV = cute.make_tensor(arg{v_idx}.iterator, {cols_layout(v_idx, 'v')})",
        f"_flash_mOt = cute.make_tensor(arg{o_idx}.iterator, {rows_layout(o_idx, 'o')})",
        f"_flash_qk_mma = {bw}.make_trivial_tiled_mma({mma_dtype}, {mma_dtype}, {majk}, {majk}, cutlass.Float32, {cg1}, ({bm}, {bn}))",
        f"_flash_pv_mma = {bw}.make_trivial_tiled_mma({mma_dtype}, {mma_dtype}, {majk}, cute.nvgpu.OperandMajorMode.MN, cutlass.Float32, {cg1}, ({bm}, {hd}), cute.nvgpu.tcgen05.OperandSource.TMEM)",
        "_flash_cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (_flash_qk_mma.thr_id.shape,))",
        f"_flash_qsl = {bw}.make_smem_layout_a(_flash_qk_mma, {qkd}, {mma_dtype}, 1)",
        f"_flash_ksl = {bw}.make_smem_layout_b(_flash_qk_mma, {qkd}, {mma_dtype}, {kv_stage})",
        f"_flash_vsl = {bw}.make_smem_layout_b(_flash_pv_mma, {pvd}, {mma_dtype}, {kv_stage})",
        f"_flash_ptl = {bw}.make_smem_layout_a(_flash_pv_mma, {pvd}, {mma_dtype}, 1)",
        f"_flash_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp({cg1})",
        f"_flash_tma_q, _flash_mQt = cute.nvgpu.make_tiled_tma_atom_A(_flash_op, _flash_mQ, {sel}(_flash_qsl, mode=[0, 1, 2]), {qkd}, _flash_qk_mma, _flash_cluster_layout_vmnk.shape{tma_internal})",
        f"_flash_tma_k, _flash_mKt = cute.nvgpu.make_tiled_tma_atom_B(_flash_op, _flash_mK, {sel}(_flash_ksl, mode=[0, 1, 2]), {qkd}, _flash_qk_mma, _flash_cluster_layout_vmnk.shape{tma_internal})",
        f"_flash_tma_v, _flash_mVt = cute.nvgpu.make_tiled_tma_atom_B(_flash_op, _flash_mV, {sel}(_flash_vsl, mode=[0, 1, 2]), {pvd}, _flash_pv_mma, _flash_cluster_layout_vmnk.shape{tma_internal})",
    ]
    ctx.body.extend(f"    {line}" for line in gated_lines)
    ctx.call_args.extend(GATED_KERNEL_PARAMS)


def _emit_flash_bwd_flex_two_cta(
    ctx: HostWrapperContext, plan: dict[str, object]
) -> None:
    """The 2-CTA flex backward at head dim 128 (``cute_flash_bwd_flex_2cta``)."""
    # The 2-CTA flex backward at head dim 128 (``cute_flash_bwd_flex_2cta``):
    # the dense 2-CTA operand set (CtaGroup.TWO MMAs, natural- and
    # transposed-orientation Q / dO halves, the 256-row K^T half, the
    # exchanged dS) over the flex (seq, D, B*H) maps; dQ reduce-added in
    # (32 x 32) fp32 boxes, dK / dV stored as (128, 64) boxes per
    # (batch, kv head).
    q_idx = ctx.plan_int("q_idx")
    k_idx = ctx.plan_int("k_idx")
    v_idx = ctx.plan_int("v_idx")
    do_idx = ctx.plan_int("do_idx")
    lse_idx = ctx.plan_int("lse_idx")
    delta_idx = ctx.plan_int("delta_idx")
    dq_idx = ctx.plan_int("dq_idx")
    dk_idx = ctx.plan_int("dk_idx")
    dv_idx = ctx.plan_int("dv_idx")
    hd = ctx.plan_int("head_dim")
    assert hd == 128
    tq = ctx.plan_int("total_q_rows")
    seq_q = ctx.plan_int("flex_seq_q")
    seq_k = ctx.plan_int("flex_seq_k")
    bh_q = ctx.plan_int("flex_bh_q")
    bh_k = ctx.plan_int("flex_bh_k")
    dtype = str(plan.get("dtype", "cutlass.Float16"))
    assert dtype in ("cutlass.Float16", "cutlass.BFloat16")
    bw = "cutlass.utils.blackwell_helpers"
    ssd = f"(256, 128, {hd})"
    tsd = f"(256, {hd}, 128)"
    dqd = f"(128, {hd}, 256)"
    majk = "cute.nvgpu.OperandMajorMode.K"
    majmn = "cute.nvgpu.OperandMajorMode.MN"
    cg2 = "cute.nvgpu.tcgen05.CtaGroup.TWO"
    sel = "cute.select"
    q_sdb = f"cute.make_layout(({seq_q}, {hd}, {bh_q}), stride=({hd}, 1, {seq_q * hd}))"
    k_sdb = f"cute.make_layout(({seq_k}, {hd}, {bh_k}), stride=({hd}, 1, {seq_k * hd}))"
    q_dsb = f"cute.make_layout(({hd}, {seq_q}, {bh_q}), stride=(1, {hd}, {seq_q * hd}))"
    k_dsb = f"cute.make_layout(({hd}, {seq_k}, {bh_k}), stride=(1, {hd}, {seq_k * hd}))"
    fbwd_lines = [
        f"_fbwd_mQ = cute.make_tensor(arg{q_idx}.iterator, {q_sdb})",
        f"_fbwd_mK = cute.make_tensor(arg{k_idx}.iterator, {k_sdb})",
        f"_fbwd_mV = cute.make_tensor(arg{v_idx}.iterator, {k_sdb})",
        f"_fbwd_mdO = cute.make_tensor(arg{do_idx}.iterator, {q_sdb})",
        f"_fbwd_mQT = cute.make_tensor(arg{q_idx}.iterator, {q_dsb})",
        f"_fbwd_mdOT = cute.make_tensor(arg{do_idx}.iterator, {q_dsb})",
        f"_fbwd_mKT = cute.make_tensor(arg{k_idx}.iterator, {k_dsb})",
        f"_fbwd_ss_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majk}, cutlass.Float32, {cg2}, (256, 128))",
        f"_fbwd_ts_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majmn}, cutlass.Float32, {cg2}, (256, {hd}), cute.nvgpu.tcgen05.OperandSource.TMEM)",
        f"_fbwd_dq_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majmn}, {majmn}, cutlass.Float32, {cg2}, (128, {hd}))",
        "_fbwd_cluster_vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (_fbwd_ss_mma.thr_id.shape,))",
        f"_fbwd_ksl = {bw}.make_smem_layout_a(_fbwd_ss_mma, {ssd}, {dtype}, 1)",
        f"_fbwd_vsl = {bw}.make_smem_layout_a(_fbwd_ss_mma, {ssd}, {dtype}, 1)",
        f"_fbwd_qsl = {bw}.make_smem_layout_b(_fbwd_ss_mma, {ssd}, {dtype}, 1)",
        f"_fbwd_dosl = {bw}.make_smem_layout_b(_fbwd_ss_mma, {ssd}, {dtype}, 1)",
        f"_fbwd_ptl = {bw}.make_smem_layout_a(_fbwd_ts_mma, {tsd}, {dtype}, 1)",
        f"_fbwd_dotl = {bw}.make_smem_layout_b(_fbwd_ts_mma, {tsd}, {dtype}, 1)",
        f"_fbwd_qtl = {bw}.make_smem_layout_b(_fbwd_ts_mma, {tsd}, {dtype}, 1)",
        f"_fbwd_ktl = {bw}.make_smem_layout_b(_fbwd_dq_mma, {dqd}, {dtype}, 1)",
        f"_fbwd_dssl = {bw}.make_smem_layout_a(_fbwd_dq_mma, {dqd}, {dtype}, 1)",
        f"_fbwd_op2 = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp({cg2})",
        f"_fbwd_tma_k, _fbwd_mKt = cute.nvgpu.make_tiled_tma_atom_A(_fbwd_op2, _fbwd_mK, {sel}(_fbwd_ksl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_v, _fbwd_mVt = cute.nvgpu.make_tiled_tma_atom_A(_fbwd_op2, _fbwd_mV, {sel}(_fbwd_vsl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_q, _fbwd_mQt = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mQ, {sel}(_fbwd_qsl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_dot, _fbwd_mdOtN = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mdO, {sel}(_fbwd_dosl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_do2, _fbwd_mdOtT = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mdOT, {sel}(_fbwd_dotl, mode=[0, 1, 2]), {tsd}, _fbwd_ts_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_qt, _fbwd_mQtT = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mQT, {sel}(_fbwd_qtl, mode=[0, 1, 2]), {tsd}, _fbwd_ts_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_kt, _fbwd_mKtT = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mKT, {sel}(_fbwd_ktl, mode=[0, 1, 2]), {dqd}, _fbwd_dq_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_mLSE = cute.make_tensor(arg{lse_idx}.iterator, cute.make_layout(({tq},)))",
        f"_fbwd_mDelta = cute.make_tensor(arg{delta_idx}.iterator, cute.make_layout(({tq},)))",
        f"_fbwd_mDQ = cute.make_tensor(arg{dq_idx}.iterator, cute.make_layout(({tq * hd},)))",
        f"_fbwd_mDQ2 = cute.make_tensor(arg{dq_idx}.iterator, {q_sdb})",
        f"_fbwd_dqsl = {bw}.make_smem_layout_epi(cutlass.Float32, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, (32, 32), 4)",
        "_fbwd_tma_dq, _fbwd_mDQt = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(), _fbwd_mDQ2, cute.select(_fbwd_dqsl, mode=[0, 1]), (32, 32))",
        f"_fbwd_mdK = cute.make_tensor(arg{dk_idx}.iterator, {k_sdb})",
        f"_fbwd_mdV = cute.make_tensor(arg{dv_idx}.iterator, {k_sdb})",
        f"_fbwd_epil = {bw}.make_smem_layout_epi({dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, (128, 64), {hd // 64})",
        "_fbwd_tma_dv, _fbwd_mdVt = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), _fbwd_mdV, cute.select(_fbwd_epil, mode=[0, 1]), (128, 64))",
        "_fbwd_tma_dk, _fbwd_mdKt = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), _fbwd_mdK, cute.select(_fbwd_epil, mode=[0, 1]), (128, 64))",
    ]
    ctx.body.extend(f"    {line}" for line in fbwd_lines)
    ctx.call_args.extend(
        [
            "_fbwd_ss_mma",
            "_fbwd_ts_mma",
            "_fbwd_dq_mma",
            "_fbwd_cluster_vmnk",
            "_fbwd_tma_q",
            "_fbwd_mQt",
            "_fbwd_tma_k",
            "_fbwd_mKt",
            "_fbwd_tma_v",
            "_fbwd_mVt",
            "_fbwd_tma_dot",
            "_fbwd_mdOtN",
            "_fbwd_tma_do2",
            "_fbwd_mdOtT",
            "_fbwd_tma_qt",
            "_fbwd_mQtT",
            "_fbwd_tma_kt",
            "_fbwd_mKtT",
            "_fbwd_qsl",
            "_fbwd_ksl",
            "_fbwd_vsl",
            "_fbwd_dosl",
            "_fbwd_ptl",
            "_fbwd_dotl",
            "_fbwd_qtl",
            "_fbwd_ktl",
            "_fbwd_dssl",
            "_fbwd_mLSE",
            "_fbwd_mDelta",
            "_fbwd_mDQ",
            "_fbwd_tma_dq",
            "_fbwd_mDQt",
            "_fbwd_dqsl",
            "_fbwd_mdK",
            "_fbwd_mdV",
            "_fbwd_epil",
            "_fbwd_tma_dv",
            "_fbwd_mdVt",
            "_fbwd_tma_dk",
            "_fbwd_mdKt",
        ]
    )


def _emit_flash_bwd_two_cta(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """The 2-CTA cluster backward (FA4 SM100 layout)."""
    # 2-CTA cluster variant (FA4 SM100 backward layout): all M-widened
    # tiled_mmas are CtaGroup.TWO; Q/dO get separate natural- and
    # transposed-orientation half loads; Kt spans the 256-row cluster KV
    # tile; sdS is the exchanged dQ A operand.
    q_idx = ctx.plan_int("q_idx")
    k_idx = ctx.plan_int("k_idx")
    v_idx = ctx.plan_int("v_idx")
    do_idx = ctx.plan_int("do_idx")
    lse_idx = ctx.plan_int("lse_idx")
    delta_idx = ctx.plan_int("delta_idx")
    dq_idx = ctx.plan_int("dq_idx")
    dk_idx = ctx.plan_int("dk_idx")
    dv_idx = ctx.plan_int("dv_idx")
    hd = ctx.plan_int("head_dim")
    tq = ctx.plan_int("total_q_rows")
    tk = ctx.plan_int("total_kv_rows")
    dtype = str(plan.get("dtype", "cutlass.Float16"))
    assert dtype in ("cutlass.Float16", "cutlass.BFloat16")
    bw = "cutlass.utils.blackwell_helpers"
    ssd = f"(256, 128, {hd})"
    tsd = f"(256, {hd}, 128)"
    dqd = f"(128, {hd}, 256)"
    majk = "cute.nvgpu.OperandMajorMode.K"
    majmn = "cute.nvgpu.OperandMajorMode.MN"
    cg2 = "cute.nvgpu.tcgen05.CtaGroup.TWO"
    sel = "cute.select"
    q_sdb = f"cute.make_layout(({tq}, {hd}, 1), stride=({hd}, 1, {tq * hd}))"
    k_sdb = f"cute.make_layout(({tk}, {hd}, 1), stride=({hd}, 1, {tk * hd}))"
    q_dsb = f"cute.make_layout(({hd}, {tq}, 1), stride=(1, {hd}, {tq * hd}))"
    k_dsb = f"cute.make_layout(({hd}, {tk}, 1), stride=(1, {hd}, {tk * hd}))"
    fbwd_lines = [
        f"_fbwd_mQ = cute.make_tensor(arg{q_idx}.iterator, {q_sdb})",
        f"_fbwd_mK = cute.make_tensor(arg{k_idx}.iterator, {k_sdb})",
        f"_fbwd_mV = cute.make_tensor(arg{v_idx}.iterator, {k_sdb})",
        f"_fbwd_mdO = cute.make_tensor(arg{do_idx}.iterator, {q_sdb})",
        f"_fbwd_mQT = cute.make_tensor(arg{q_idx}.iterator, {q_dsb})",
        f"_fbwd_mdOT = cute.make_tensor(arg{do_idx}.iterator, {q_dsb})",
        f"_fbwd_mKT = cute.make_tensor(arg{k_idx}.iterator, {k_dsb})",
        f"_fbwd_ss_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majk}, cutlass.Float32, {cg2}, (256, 128))",
        f"_fbwd_ts_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majmn}, cutlass.Float32, {cg2}, (256, {hd}), cute.nvgpu.tcgen05.OperandSource.TMEM)",
        f"_fbwd_dq_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majmn}, {majmn}, cutlass.Float32, {cg2}, (128, {hd}))",
        "_fbwd_cluster_vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (_fbwd_ss_mma.thr_id.shape,))",
        f"_fbwd_ksl = {bw}.make_smem_layout_a(_fbwd_ss_mma, {ssd}, {dtype}, 1)",
        f"_fbwd_vsl = {bw}.make_smem_layout_a(_fbwd_ss_mma, {ssd}, {dtype}, 1)",
        f"_fbwd_qsl = {bw}.make_smem_layout_b(_fbwd_ss_mma, {ssd}, {dtype}, 1)",
        f"_fbwd_dosl = {bw}.make_smem_layout_b(_fbwd_ss_mma, {ssd}, {dtype}, 1)",
        f"_fbwd_ptl = {bw}.make_smem_layout_a(_fbwd_ts_mma, {tsd}, {dtype}, 1)",
        f"_fbwd_dotl = {bw}.make_smem_layout_b(_fbwd_ts_mma, {tsd}, {dtype}, 1)",
        f"_fbwd_qtl = {bw}.make_smem_layout_b(_fbwd_ts_mma, {tsd}, {dtype}, 1)",
        f"_fbwd_ktl = {bw}.make_smem_layout_b(_fbwd_dq_mma, {dqd}, {dtype}, 1)",
        f"_fbwd_dssl = {bw}.make_smem_layout_a(_fbwd_dq_mma, {dqd}, {dtype}, 1)",
        f"_fbwd_op2 = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp({cg2})",
        f"_fbwd_tma_k, _fbwd_mKt = cute.nvgpu.make_tiled_tma_atom_A(_fbwd_op2, _fbwd_mK, {sel}(_fbwd_ksl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_v, _fbwd_mVt = cute.nvgpu.make_tiled_tma_atom_A(_fbwd_op2, _fbwd_mV, {sel}(_fbwd_vsl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_q, _fbwd_mQt = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mQ, {sel}(_fbwd_qsl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_dot, _fbwd_mdOtN = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mdO, {sel}(_fbwd_dosl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_do2, _fbwd_mdOtT = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mdOT, {sel}(_fbwd_dotl, mode=[0, 1, 2]), {tsd}, _fbwd_ts_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_qt, _fbwd_mQtT = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mQT, {sel}(_fbwd_qtl, mode=[0, 1, 2]), {tsd}, _fbwd_ts_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_kt, _fbwd_mKtT = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mKT, {sel}(_fbwd_ktl, mode=[0, 1, 2]), {dqd}, _fbwd_dq_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_mLSE = cute.make_tensor(arg{lse_idx}.iterator, cute.make_layout(({tq},)))",
        f"_fbwd_mDelta = cute.make_tensor(arg{delta_idx}.iterator, cute.make_layout(({tq},)))",
        f"_fbwd_mDQ = cute.make_tensor(arg{dq_idx}.iterator, cute.make_layout(({tq * hd},)))",
        f"_fbwd_mDQ2 = cute.make_tensor(arg{dq_idx}.iterator, cute.make_layout(({tq}, {hd}), stride=({hd}, 1)))",
        f"_fbwd_dqsl = {bw}.make_smem_layout_epi(cutlass.Float32, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, (32, 32), 4)",
        "_fbwd_tma_dq, _fbwd_mDQt = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(), _fbwd_mDQ2, cute.select(_fbwd_dqsl, mode=[0, 1]), (32, 32))",
        f"_fbwd_mdK = cute.make_tensor(arg{dk_idx}.iterator, cute.make_layout(({tk}, {hd}), stride=({hd}, 1)))",
        f"_fbwd_mdV = cute.make_tensor(arg{dv_idx}.iterator, cute.make_layout(({tk}, {hd}), stride=({hd}, 1)))",
        f"_fbwd_epil = {bw}.make_smem_layout_epi({dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, (128, 64), {hd // 64})",
        "_fbwd_tma_dv, _fbwd_mdVt = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), _fbwd_mdV, cute.select(_fbwd_epil, mode=[0, 1]), (128, 64))",
        "_fbwd_tma_dk, _fbwd_mdKt = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), _fbwd_mdK, cute.select(_fbwd_epil, mode=[0, 1]), (128, 64))",
    ]
    ctx.body.extend(f"    {line}" for line in fbwd_lines)
    ctx.call_args.extend(
        [
            "_fbwd_ss_mma",
            "_fbwd_ts_mma",
            "_fbwd_dq_mma",
            "_fbwd_cluster_vmnk",
            "_fbwd_tma_q",
            "_fbwd_mQt",
            "_fbwd_tma_k",
            "_fbwd_mKt",
            "_fbwd_tma_v",
            "_fbwd_mVt",
            "_fbwd_tma_dot",
            "_fbwd_mdOtN",
            "_fbwd_tma_do2",
            "_fbwd_mdOtT",
            "_fbwd_tma_qt",
            "_fbwd_mQtT",
            "_fbwd_tma_kt",
            "_fbwd_mKtT",
            "_fbwd_qsl",
            "_fbwd_ksl",
            "_fbwd_vsl",
            "_fbwd_dosl",
            "_fbwd_ptl",
            "_fbwd_dotl",
            "_fbwd_qtl",
            "_fbwd_ktl",
            "_fbwd_dssl",
            "_fbwd_mLSE",
            "_fbwd_mDelta",
            "_fbwd_mDQ",
            "_fbwd_tma_dq",
            "_fbwd_mDQt",
            "_fbwd_dqsl",
            "_fbwd_mdK",
            "_fbwd_mdV",
            "_fbwd_epil",
            "_fbwd_tma_dv",
            "_fbwd_mdVt",
            "_fbwd_tma_dk",
            "_fbwd_mdKt",
        ]
    )


def _emit_flash_bwd_flex_pair(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """The 2-CTA flex backward at head dim 256 (``cute_flash_bwd_flex_pair``)."""
    # The 2-CTA flex backward at head dim 256 (``cute_flash_bwd_flex_pair``):
    # two cluster roles per 128-row KV block. The dK/dV clusters (64 KV
    # rows per CTA) run S / dP as the pair's M=128 MMAs over K-major K / V
    # (A) and Q / dO (B, 64 Q rows per CTA, 256 wide) and dV / dK as
    # M=128 N=256 MMAs over the staged P^T / dS^T (A, K-major 64 x 128)
    # and dO^T / Q^T halves (B, MN-major: each CTA its 128 head-dim
    # columns x 128 Q), storing (64, 64) boxes. The dQ clusters (64 Q rows
    # of the step per CTA) run S^T / dP^T with Q / dO as the A operands
    # and K / V as the B operands (the same tiles and layouts, the roles
    # swapped) and dQ = dS K as an M=128 N=256 MMA over the staged dS (A)
    # and K^T (B, MN-major: each CTA its 128 head-dim columns x 128 KV
    # rows), reduce-added in (32 x 16) fp32 boxes. Maps are the flex
    # (seq, D, B*H) ones (contiguous operands).
    q_idx = ctx.plan_int("q_idx")
    k_idx = ctx.plan_int("k_idx")
    v_idx = ctx.plan_int("v_idx")
    do_idx = ctx.plan_int("do_idx")
    lse_idx = ctx.plan_int("lse_idx")
    delta_idx = ctx.plan_int("delta_idx")
    dq_idx = ctx.plan_int("dq_idx")
    dk_idx = ctx.plan_int("dk_idx")
    dv_idx = ctx.plan_int("dv_idx")
    hd = ctx.plan_int("head_dim")
    assert hd == 256
    tq = ctx.plan_int("total_q_rows")
    seq_q = ctx.plan_int("flex_seq_q")
    seq_k = ctx.plan_int("flex_seq_k")
    bh_q = ctx.plan_int("flex_bh_q")
    bh_k = ctx.plan_int("flex_bh_k")
    dtype = str(plan.get("dtype", "cutlass.Float16"))
    assert dtype in ("cutlass.Float16", "cutlass.BFloat16")
    bw = "cutlass.utils.blackwell_helpers"
    majk = "cute.nvgpu.OperandMajorMode.K"
    majmn = "cute.nvgpu.OperandMajorMode.MN"
    cg2 = "cute.nvgpu.tcgen05.CtaGroup.TWO"
    sel = "cute.select"
    ssd = f"(128, 128, {hd})"
    pvd = f"(128, {hd}, 128)"
    q_sdb = f"cute.make_layout(({seq_q}, {hd}, {bh_q}), stride=({hd}, 1, {seq_q * hd}))"
    k_sdb = f"cute.make_layout(({seq_k}, {hd}, {bh_k}), stride=({hd}, 1, {seq_k * hd}))"
    q_dsb = f"cute.make_layout(({hd}, {seq_q}, {bh_q}), stride=(1, {hd}, {seq_q * hd}))"
    fbwd_lines = [
        f"_fbwd_mQ = cute.make_tensor(arg{q_idx}.iterator, {q_sdb})",
        f"_fbwd_mK = cute.make_tensor(arg{k_idx}.iterator, {k_sdb})",
        f"_fbwd_mV = cute.make_tensor(arg{v_idx}.iterator, {k_sdb})",
        f"_fbwd_mdO = cute.make_tensor(arg{do_idx}.iterator, {q_sdb})",
        f"_fbwd_mQT = cute.make_tensor(arg{q_idx}.iterator, {q_dsb})",
        f"_fbwd_mdOT = cute.make_tensor(arg{do_idx}.iterator, {q_dsb})",
        f"_fbwd_ss_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majk}, cutlass.Float32, {cg2}, (128, 128))",
        f"_fbwd_pv_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majmn}, cutlass.Float32, {cg2}, (128, {hd}))",
        "_fbwd_cluster_vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (_fbwd_ss_mma.thr_id.shape,))",
        f"_fbwd_ksl = {bw}.make_smem_layout_a(_fbwd_ss_mma, {ssd}, {dtype}, 1)",
        f"_fbwd_vsl = {bw}.make_smem_layout_a(_fbwd_ss_mma, {ssd}, {dtype}, 1)",
        f"_fbwd_qsl = {bw}.make_smem_layout_b(_fbwd_ss_mma, {ssd}, {dtype}, 1)",
        f"_fbwd_dosl = {bw}.make_smem_layout_b(_fbwd_ss_mma, {ssd}, {dtype}, 1)",
        f"_fbwd_pal = {bw}.make_smem_layout_a(_fbwd_pv_mma, {pvd}, {dtype}, 1)",
        f"_fbwd_dotl = {bw}.make_smem_layout_b(_fbwd_pv_mma, {pvd}, {dtype}, 1)",
        f"_fbwd_dotl2 = {bw}.make_smem_layout_b(_fbwd_pv_mma, {pvd}, {dtype}, 2)",
        f"_fbwd_qtl = {bw}.make_smem_layout_b(_fbwd_pv_mma, {pvd}, {dtype}, 1)",
        f"_fbwd_op2 = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp({cg2})",
        f"_fbwd_tma_k, _fbwd_mKt = cute.nvgpu.make_tiled_tma_atom_A(_fbwd_op2, _fbwd_mK, {sel}(_fbwd_ksl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_v, _fbwd_mVt = cute.nvgpu.make_tiled_tma_atom_A(_fbwd_op2, _fbwd_mV, {sel}(_fbwd_vsl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_q, _fbwd_mQt = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mQ, {sel}(_fbwd_qsl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_dot, _fbwd_mdOtN = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mdO, {sel}(_fbwd_dosl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_do2, _fbwd_mdOtT = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mdOT, {sel}(_fbwd_dotl, mode=[0, 1, 2]), {pvd}, _fbwd_pv_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_qt, _fbwd_mQtT = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mQT, {sel}(_fbwd_qtl, mode=[0, 1, 2]), {pvd}, _fbwd_pv_mma, _fbwd_cluster_vmnk.shape)",
        # The dQ clusters' operands (the Q-split orientation): Q / dO as
        # the A operands of S^T / dP^T, K / V as their B operands, and
        # K^T (this CTA's 128 head-dim columns x 128 KV rows) as the dQ
        # MMA's B operand.
        f"_fbwd_mKT = cute.make_tensor(arg{k_idx}.iterator, cute.make_layout(({hd}, {seq_k}, {bh_k}), stride=(1, {hd}, {seq_k * hd})))",
        f"_fbwd_tma_qa, _fbwd_mQa = cute.nvgpu.make_tiled_tma_atom_A(_fbwd_op2, _fbwd_mQ, {sel}(_fbwd_ksl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_doa, _fbwd_mdOa = cute.nvgpu.make_tiled_tma_atom_A(_fbwd_op2, _fbwd_mdO, {sel}(_fbwd_vsl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_kb, _fbwd_mKb = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mK, {sel}(_fbwd_qsl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_vb, _fbwd_mVb = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mV, {sel}(_fbwd_dosl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_kt2, _fbwd_mKt2 = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op2, _fbwd_mKT, {sel}(_fbwd_dotl, mode=[0, 1, 2]), {pvd}, _fbwd_pv_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_mLSE = cute.make_tensor(arg{lse_idx}.iterator, cute.make_layout(({tq},)))",
        f"_fbwd_mDelta = cute.make_tensor(arg{delta_idx}.iterator, cute.make_layout(({tq},)))",
        f"_fbwd_mDQ2 = cute.make_tensor(arg{dq_idx}.iterator, cute.make_layout(({seq_q}, {hd}, {bh_q}), stride=({hd}, 1, {seq_q * hd})))",
        f"_fbwd_dqsl = {bw}.make_smem_layout_epi(cutlass.Float32, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, (32, 16), 4)",
        # The dQ clusters' once-per-tile drain: 16 boxes per 32 KB operand buffer.
        f"_fbwd_dqsl16 = {bw}.make_smem_layout_epi(cutlass.Float32, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, (32, 16), 16)",
        "_fbwd_tma_dq, _fbwd_mDQt = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(), _fbwd_mDQ2, cute.select(_fbwd_dqsl, mode=[0, 1]), (32, 16))",
        f"_fbwd_mdK = cute.make_tensor(arg{dk_idx}.iterator, {k_sdb})",
        f"_fbwd_mdV = cute.make_tensor(arg{dv_idx}.iterator, {k_sdb})",
        f"_fbwd_epil = {bw}.make_smem_layout_epi({dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, (64, 64), 4)",
        "_fbwd_tma_dv, _fbwd_mdVt = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), _fbwd_mdV, cute.select(_fbwd_epil, mode=[0, 1]), (64, 64))",
        "_fbwd_tma_dk, _fbwd_mdKt = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), _fbwd_mdK, cute.select(_fbwd_epil, mode=[0, 1]), (64, 64))",
    ]
    ctx.body.extend(f"    {line}" for line in fbwd_lines)
    from ...._compiler.cute.cute_flash_bwd_flex_pair import (
        _FLASH_BWD_FLEX_PAIR_KERNEL_PARAMS,
    )

    # The generated kernel's parameter order (one list, owned by the emitter).
    ctx.call_args.extend(_FLASH_BWD_FLEX_PAIR_KERNEL_PARAMS)


def _emit_flash_bwd(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """The one-CTA fused tcgen05 attention backward (dense and flex walks)."""
    # Fused tcgen05 attention-backward host setup: four tiled_mmas (S/dP
    # smem-smem, dV tmem-A, dK smem K-major, dQ smem MN-major), TMA atoms
    # for K/V (per-CTA single tiles) and Q/dO (inner-loop rings), plus the
    # raw fp32 LSE/delta/dq views and the dK/dV output views.
    q_idx = ctx.plan_int("q_idx")
    k_idx = ctx.plan_int("k_idx")
    v_idx = ctx.plan_int("v_idx")
    do_idx = ctx.plan_int("do_idx")
    lse_idx = ctx.plan_int("lse_idx")
    delta_idx = ctx.plan_int("delta_idx")
    dq_idx = ctx.plan_int("dq_idx")
    dk_idx = ctx.plan_int("dk_idx")
    dv_idx = ctx.plan_int("dv_idx")
    hd = ctx.plan_int("head_dim")
    # The value head dim: dO / V / dV tiles, the dP MMA's K extent and the
    # dV MMA's N extent (Q / K / dK / dQ use ``hd``).
    hd_v = ctx.plan_int("head_dim_v", hd)
    # Shared-memory tiles and TMA boxes are whole SWIZZLE_128B atoms: the
    # head dims padded to multiples of 64 columns (the TMA unit zero-fills
    # the columns past the tensor extent and clips the stores); the MMAs
    # that take N from the head dim get exact-width instruction
    # descriptors from the ``_x`` tiled MMAs.
    hd_p = -(-hd // 64) * 64
    hd_v_p = -(-hd_v // 64) * 64
    # The dQ MMA's N: the whole (padded) head dim, or one 64-column chunk
    # when the flex walk drains dQ chunk by chunk (its K^T operand layout
    # and fragments then carry a 3-wide N mode; the body offsets them).
    dq_n = ctx.plan_optional_int("flex_dq_chunk") or hd_p
    tq = ctx.plan_int("total_q_rows")
    tk = ctx.plan_int("total_kv_rows")
    q_stage = ctx.plan_int("q_stage")
    do_stage = ctx.plan_int("do_stage")
    kv_stage = ctx.plan_int("kv_stage", 1)
    epi_stages = max(hd_p, hd_v_p) // 64 * kv_stage
    dtype = str(plan.get("dtype", "cutlass.Float16"))
    assert dtype in ("cutlass.Float16", "cutlass.BFloat16")
    bw = "cutlass.utils.blackwell_helpers"
    ssd = f"(128, 128, {hd_p})"
    ssd_v = f"(128, 128, {hd_v_p})"
    tsd = f"(128, {hd_p}, 128)"
    tsd_v = f"(128, {hd_v_p}, 128)"
    majk = "cute.nvgpu.OperandMajorMode.K"
    majmn = "cute.nvgpu.OperandMajorMode.MN"
    cg1 = "cute.nvgpu.tcgen05.CtaGroup.ONE"
    sel = "cute.select"
    # The FlexAttention backward walks per-(batch, head) sequences: the
    # maps are (seq, D, B*H) so a ragged M / N is zero-filled on load and
    # clipped on store by the TMA unit; the dense path keeps the flat
    # (rows, D) view (its sequences are multiples of 128).
    flex_seq_q = ctx.plan_optional_int("flex_seq_q")
    flex_strides = plan.get("flex_strides")
    seq_k = ctx.plan_optional_int("flex_seq_k")
    if flex_seq_q is not None:
        assert seq_k is not None
        bh_q = ctx.plan_int("flex_bh_q")
        bh_k = ctx.plan_int("flex_bh_k")
        q_sdb = f"cute.make_layout(({flex_seq_q}, {hd}, {bh_q}), stride=({hd}, 1, {flex_seq_q * hd}))"
        do_sdb = f"cute.make_layout(({flex_seq_q}, {hd_v}, {bh_q}), stride=({hd_v}, 1, {flex_seq_q * hd_v}))"
        k_sdb = (
            f"cute.make_layout(({seq_k}, {hd}, {bh_k}), stride=({hd}, 1, {seq_k * hd}))"
        )
        v_sdb = f"cute.make_layout(({seq_k}, {hd_v}, {bh_k}), stride=({hd_v}, 1, {seq_k * hd_v}))"
        dq2_layout = f"cute.make_layout(({flex_seq_q}, {hd}, {bh_q}), stride=({hd}, 1, {flex_seq_q * hd}))"
        dk_sdb = k_sdb
        dv_sdb = v_sdb
    else:
        q_sdb = f"cute.make_layout(({tq}, {hd}, 1), stride=({hd}, 1, {tq * hd}))"
        do_sdb = f"cute.make_layout(({tq}, {hd_v}, 1), stride=({hd_v}, 1, {tq * hd_v}))"
        k_sdb = f"cute.make_layout(({tk}, {hd}, 1), stride=({hd}, 1, {tk * hd}))"
        v_sdb = f"cute.make_layout(({tk}, {hd_v}, 1), stride=({hd_v}, 1, {tk * hd_v}))"
        dq2_layout = f"cute.make_layout(({tq}, {hd}), stride=({hd}, 1))"
        dk_sdb = f"cute.make_layout(({tk}, {hd}), stride=({hd}, 1))"
        dv_sdb = f"cute.make_layout(({tk}, {hd_v}), stride=({hd_v}, 1))"
    # One map per operand: the flex walk's strided operands (a
    # ``[B, S, H, D]`` transpose, a packed-QKV slice) take rank-4
    # (seq, D, H, B) maps over their real strides (unit inner stride, the
    # rest 16-byte aligned); the device body indexes them by head and
    # batch. dQ, LSE and delta stay contiguous.
    q_layout, do_layout = q_sdb, do_sdb
    k_layout, v_layout = k_sdb, v_sdb
    dk_layout, dv_layout = dk_sdb, dv_sdb
    if flex_strides is not None:
        assert flex_seq_q is not None and seq_k is not None
        assert isinstance(flex_strides, (list, tuple))
        flex_batch = ctx.plan_int("flex_batch")
        flex_heads = ctx.plan_int("flex_heads")
        flex_kv_heads = ctx.plan_int("flex_kv_heads")
        q_st, do_st, k_st, v_st, dk_st, dv_st = cast(
            "tuple[Sequence[int], ...]", tuple(flex_strides)
        )
        q_layout = flex_rank4_layout(flex_seq_q, hd, flex_heads, flex_batch, q_st)
        do_layout = flex_rank4_layout(flex_seq_q, hd_v, flex_heads, flex_batch, do_st)
        k_layout = flex_rank4_layout(seq_k, hd, flex_kv_heads, flex_batch, k_st)
        v_layout = flex_rank4_layout(seq_k, hd_v, flex_kv_heads, flex_batch, v_st)
        dk_layout = flex_rank4_layout(seq_k, hd, flex_kv_heads, flex_batch, dk_st)
        dv_layout = flex_rank4_layout(seq_k, hd_v, flex_kv_heads, flex_batch, dv_st)
    fbwd_lines = [
        f"_fbwd_mQ = cute.make_tensor(arg{q_idx}.iterator, {q_layout})",
        f"_fbwd_mK = cute.make_tensor(arg{k_idx}.iterator, {k_layout})",
        f"_fbwd_mV = cute.make_tensor(arg{v_idx}.iterator, {v_layout})",
        f"_fbwd_mdO = cute.make_tensor(arg{do_idx}.iterator, {do_layout})",
        f"_fbwd_ss_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majk}, cutlass.Float32, {cg1}, (128, 128))",
        f"_fbwd_ts_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majmn}, cutlass.Float32, {cg1}, (128, {hd_v_p}), cute.nvgpu.tcgen05.OperandSource.TMEM)",
        f"_fbwd_dsk_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majmn}, cutlass.Float32, {cg1}, (128, {hd_p}))",
        f"_fbwd_dq_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majmn}, {majmn}, cutlass.Float32, {cg1}, (128, {dq_n}))",
        "_fbwd_cluster_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (_fbwd_ss_mma.thr_id.shape,))",
        f"_fbwd_ksl = {bw}.make_smem_layout_a(_fbwd_ss_mma, {ssd}, {dtype}, {kv_stage})",
        f"_fbwd_vsl = {bw}.make_smem_layout_a(_fbwd_ss_mma, {ssd_v}, {dtype}, {kv_stage})",
        f"_fbwd_qsl = {bw}.make_smem_layout_b(_fbwd_ss_mma, {ssd}, {dtype}, {q_stage})",
        f"_fbwd_dosl = {bw}.make_smem_layout_b(_fbwd_ss_mma, {ssd_v}, {dtype}, {do_stage})",
        f"_fbwd_ptl = {bw}.make_smem_layout_a(_fbwd_ts_mma, {tsd_v}, {dtype}, 1)",
        f"_fbwd_dotl = {bw}.make_smem_layout_b(_fbwd_ts_mma, {tsd_v}, {dtype}, {do_stage})",
        f"_fbwd_qtl = {bw}.make_smem_layout_b(_fbwd_dsk_mma, {tsd}, {dtype}, {q_stage})",
        f"_fbwd_dsnk = {bw}.make_smem_layout_a(_fbwd_dsk_mma, {tsd}, {dtype}, 1)",
        f"_fbwd_dssl = {bw}.make_smem_layout_a(_fbwd_dq_mma, {tsd}, {dtype}, 1)",
        f"_fbwd_ktl = {bw}.make_smem_layout_b(_fbwd_dq_mma, {tsd}, {dtype}, {kv_stage})",
        "_fbwd_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cute.nvgpu.tcgen05.CtaGroup.ONE)",
        f"_fbwd_tma_k, _fbwd_mKt = cute.nvgpu.make_tiled_tma_atom_A(_fbwd_op, _fbwd_mK, {sel}(_fbwd_ksl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_v, _fbwd_mVt = cute.nvgpu.make_tiled_tma_atom_A(_fbwd_op, _fbwd_mV, {sel}(_fbwd_vsl, mode=[0, 1, 2]), {ssd_v}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_q, _fbwd_mQt = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op, _fbwd_mQ, {sel}(_fbwd_qsl, mode=[0, 1, 2]), {ssd}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_tma_do, _fbwd_mdOt = cute.nvgpu.make_tiled_tma_atom_B(_fbwd_op, _fbwd_mdO, {sel}(_fbwd_dosl, mode=[0, 1, 2]), {ssd_v}, _fbwd_ss_mma, _fbwd_cluster_vmnk.shape)",
        f"_fbwd_mLSE = cute.make_tensor(arg{lse_idx}.iterator, cute.make_layout(({tq},)))",
        f"_fbwd_mDelta = cute.make_tensor(arg{delta_idx}.iterator, cute.make_layout(({tq},)))",
        f"_fbwd_mDQ = cute.make_tensor(arg{dq_idx}.iterator, cute.make_layout(({tq * hd},)))",
        f"_fbwd_mDQ2 = cute.make_tensor(arg{dq_idx}.iterator, {dq2_layout})",
        f"_fbwd_dqsl = {bw}.make_smem_layout_epi(cutlass.Float32, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, (32, 32), {ctx.plan_optional_int('flex_dq_boxes') or 8})",
        "_fbwd_tma_dq, _fbwd_mDQt = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(), _fbwd_mDQ2, cute.select(_fbwd_dqsl, mode=[0, 1]), (32, 32))",
        f"_fbwd_mdK = cute.make_tensor(arg{dk_idx}.iterator, {dk_layout})",
        f"_fbwd_mdV = cute.make_tensor(arg{dv_idx}.iterator, {dv_layout})",
        f"_fbwd_epil = {bw}.make_smem_layout_epi({dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, (128, 64), {epi_stages})",
        "_fbwd_tma_dv, _fbwd_mdVt = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), _fbwd_mdV, cute.select(_fbwd_epil, mode=[0, 1]), (128, 64))",
        "_fbwd_tma_dk, _fbwd_mdKt = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), _fbwd_mdK, cute.select(_fbwd_epil, mode=[0, 1]), (128, 64))",
    ]
    exact_mmas: list[str] = []
    if hd_v != hd_v_p:
        exact_mmas.append("_fbwd_ts_mma_x")
        fbwd_lines.append(
            f"_fbwd_ts_mma_x = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majmn}, cutlass.Float32, {cg1}, (128, {hd_v}), cute.nvgpu.tcgen05.OperandSource.TMEM)"
        )
    if hd != hd_p:
        exact_mmas += ["_fbwd_dsk_mma_x", "_fbwd_dq_mma_x"]
        fbwd_lines += [
            f"_fbwd_dsk_mma_x = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majmn}, cutlass.Float32, {cg1}, (128, {hd}))",
            f"_fbwd_dq_mma_x = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majmn}, {majmn}, cutlass.Float32, {cg1}, (128, {hd}))",
        ]
    if plan.get("flex_ds_tmem_dk"):
        # dS^T through TMEM feeds the dK MMA (A from TMEM, B = Q^T MN-major)
        # with its own N = D atom; the exact-N descriptor twin at padded D.
        exact_mmas.append("_fbwd_dskts_mma")
        fbwd_lines.append(
            f"_fbwd_dskts_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majmn}, cutlass.Float32, {cg1}, (128, {hd_p}), cute.nvgpu.tcgen05.OperandSource.TMEM)"
        )
        if hd != hd_p:
            exact_mmas.append("_fbwd_dskts_mma_x")
            fbwd_lines.append(
                f"_fbwd_dskts_mma_x = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majmn}, cutlass.Float32, {cg1}, (128, {hd}), cute.nvgpu.tcgen05.OperandSource.TMEM)"
            )
    aug_stages = ctx.plan_optional_int("flex_aug_stages") or 0
    if aug_stages:
        # The flex walk's statistics slices: one (128 x 16) B-operand
        # slice per Q stage and the (128 x 16) A-operand constant slice.
        fbwd_lines += [
            f"_fbwd_augl = {bw}.make_smem_layout_b(_fbwd_ss_mma, (128, 128, 16), {dtype}, {aug_stages})",
            f"_fbwd_kaugl = {bw}.make_smem_layout_a(_fbwd_ss_mma, (128, 128, 16), {dtype}, 1)",
        ]
    ctx.body.extend(f"    {line}" for line in fbwd_lines)
    ctx.call_args.extend(
        [
            "_fbwd_ss_mma",
            "_fbwd_ts_mma",
            "_fbwd_dsk_mma",
            "_fbwd_dq_mma",
            "_fbwd_tma_q",
            "_fbwd_mQt",
            "_fbwd_tma_k",
            "_fbwd_mKt",
            "_fbwd_tma_v",
            "_fbwd_mVt",
            "_fbwd_tma_do",
            "_fbwd_mdOt",
            "_fbwd_qsl",
            "_fbwd_ksl",
            "_fbwd_vsl",
            "_fbwd_dosl",
            "_fbwd_ptl",
            "_fbwd_qtl",
            "_fbwd_dotl",
            "_fbwd_ktl",
            "_fbwd_dssl",
            "_fbwd_dsnk",
            "_fbwd_mLSE",
            "_fbwd_mDelta",
            "_fbwd_mDQ",
            "_fbwd_tma_dq",
            "_fbwd_mDQt",
            "_fbwd_dqsl",
            "_fbwd_mdK",
            "_fbwd_mdV",
            "_fbwd_epil",
            "_fbwd_tma_dv",
            "_fbwd_mdVt",
            "_fbwd_tma_dk",
            "_fbwd_mdKt",
        ]
    )
    ctx.call_args.extend(exact_mmas)
    if aug_stages:
        ctx.call_args.extend(["_fbwd_augl", "_fbwd_kaugl"])


def emit_flash_bwd(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``helion_flash_bwd``: the backward layout the plan's flags select."""
    if plan.get("flex_two_cta"):
        _emit_flash_bwd_flex_two_cta(ctx, plan)
    elif plan.get("two_cta"):
        _emit_flash_bwd_two_cta(ctx, plan)
    elif plan.get("flex_pair"):
        _emit_flash_bwd_flex_pair(ctx, plan)
    else:
        _emit_flash_bwd(ctx, plan)


# The flash families (dense, gated, backward) share one generated preamble that
# imports these runtime modules; the alternating-warpgroup body adds its own.
FLASH_DEPENDENCIES = (
    "_compiler/cute/_flash_runtime.py",
    "_compiler/cute/_flash_alt_runtime.py",
    "_compiler/cute/_flash_gemm_ptx.py",
    "_compiler/cute/_mlir_compat.py",
    "_compiler/cute/epilogue_helpers.py",
)

# The register-MMA families (row_mma flash, warp_mma GEMM, register subblocks,
# mma_dag) share the row-program runtime, which takes its PTX primitives from
# the device-helper table.
ROW_MMA_DEPENDENCIES = (
    "_compiler/cute/_flash_row_mma_runtime.py",
    "_compiler/cute/device/ptx.py",
)

EMITTERS = (
    HostWrapperEmitter("helion_small_biased_attention", emit_small_biased_attention),
    HostWrapperEmitter(
        "helion_flash_row_mma",
        emit_flash_row_mma,
        source_dependencies=ROW_MMA_DEPENDENCIES,
    ),
    HostWrapperEmitter(
        "helion_flash_bwd_row_mma",
        emit_flash_bwd_row_mma,
        source_dependencies=ROW_MMA_DEPENDENCIES,
    ),
    HostWrapperEmitter(
        "helion_flash", emit_flash, source_dependencies=FLASH_DEPENDENCIES
    ),
    HostWrapperEmitter(
        "helion_flash_gated", emit_flash_gated, source_dependencies=FLASH_DEPENDENCIES
    ),
    HostWrapperEmitter(
        "helion_flash_gated_row_mma",
        emit_flash_gated_row_mma,
        source_dependencies=(
            *FLASH_DEPENDENCIES,
            "_compiler/cute/_flash_row_mma_runtime.py",
        ),
    ),
    HostWrapperEmitter(
        "helion_flash_bwd", emit_flash_bwd, source_dependencies=FLASH_DEPENDENCIES
    ),
)
