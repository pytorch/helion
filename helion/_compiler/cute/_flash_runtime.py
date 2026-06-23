# Runtime support for the fused tcgen05 flash-attention codegen path.
#
# NOTE: deliberately NO ``from __future__ import annotations`` -- it stringifies
# annotations, which breaks ``@cute.struct`` field-type resolution
# (MemRange/Align). See ``.notes/spikes/STAGE0_FINDINGS.md`` Gotcha G. The
# generated Helion module DOES carry ``from __future__ import annotations`` at
# its top, so the struct + the inline-traced rescale helper must live here, in a
# real module compiled without that flag, and be imported by the generated code.
import functools
from functools import partial
from typing import Any
from typing import cast

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import nvvm
from cutlass.base_dsl.typing import Numeric
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Float32
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op


@functools.cache
def flash_shared_storage(
    head_dim: int,
    kv_stage: int = 1,
    s_stage: int = 1,
    dtype: object = cutlass.Float16,
) -> type:
    """Build the SharedStorage struct for a given head_dim / kv_stage / s_stage.

    K and V live in SEPARATE smem (a single buffer can't guarantee QK finishes
    reading K before V overwrites it). With ``kv_stage > 1`` each is a
    multi-buffered TMA ring so warp 0 can prefetch the next KV tiles' K/V while
    the current tile's MMA/softmax run (Stage 3). The mbarrier MemRanges are
    ``2 * stages`` deep (full + empty per stage).

    With ``s_stage == 2`` (Stage 4 warp-spec + double-buffered S) the ``mma_s``
    (QK->softmax) and ``p_ready`` (consumer->warp0 "PV safe") pipelines are 2 deep
    so the producer warpgroup runs QK(k+1) into the OTHER S buffer while the
    consumer warpgroup runs softmax(k) -- the QK MMA overlaps the softmax.
    """

    @cute.struct
    class SharedStorage:
        q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
        k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * kv_stage]
        v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * kv_stage]
        mma_s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * s_stage]
        mma_o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
        # p_ready: consumer warpgroup -> warp 0, "P written + O rescaled -> PV
        # safe" (Stage 4). Unused at s_stage==1 (single-warpgroup path) but always
        # allocated to keep the struct layout uniform.
        p_ready_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * s_stage]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
        tmem_holding_buf: cutlass.Int32
        sQ: cute.struct.Align[cute.struct.MemRange[dtype, 128 * head_dim], 1024]
        sK: cute.struct.Align[
            cute.struct.MemRange[dtype, 128 * head_dim * kv_stage], 1024
        ]
        sV: cute.struct.Align[
            cute.struct.MemRange[dtype, 128 * head_dim * kv_stage], 1024
        ]

    return SharedStorage


@functools.cache
def flash_fa4_shared_storage(
    head_dim: int,
    kv_stage: int,
    q_stage: int = 2,
    s_corr_stage: int = 2,
    dtype: object = cutlass.Float16,
    epi_tma: bool = False,
) -> type:
    """FA4-topology SharedStorage (faithful port of the spike struct).

    16-warp / 512-thread layout: 2 softmax warpgroups (128 threads each), a
    correction warpgroup, and single MMA/load/epilogue/empty warps. The raw
    mbarriers (s_full / pfor / pfor2 / o_full / s*_corr) are FA4-style raw
    handshakes. K and V share one FA4-style KV pipeline and one shared-memory
    ring; the emitter aliases the V tensor over ``sK`` with the V swizzle.
    s0_corr / s1_corr use the first ``s_corr_stage`` slots as full barriers and
    the second half as empty barriers. ``sScale`` is the FA4-style softmax to
    correction handoff: steady slots carry alpha, and the final slot carries
    row_sum. softmax_threads is fixed at 128 (one warpgroup). The optional
    epi-TMA path gets a dedicated 2-stage ``sO`` buffer so persistent work-items
    can load the next Q tile while the epilogue drains the previous O tile.
    """
    softmax_threads = 128

    if epi_tma:

        @cute.struct
        class SharedStorage:
            q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, q_stage * 2]
            kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, kv_stage * 2]
            # Raw mbarriers (FA4-style). s_full / o_full are tcgen05.commit (cnt 1);
            # pfor = softmax(P[0:96]) + correction (cnt 2*128); pfor2 = softmax(P[96:128]) (cnt 128).
            s_full_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            pfor_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            pfor2_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            o_full_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            # correction -> epilogue TMA-store full/empty handshakes.
            corr_epi_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
            # softmax -> correction raw full/empty handshakes.
            s0_corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, s_corr_stage * 2]
            s1_corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, s_corr_stage * 2]
            tmem_dealloc_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
            sScale: cute.struct.MemRange[
                cutlass.Float32, s_corr_stage * q_stage * softmax_threads
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[dtype, 128 * head_dim * q_stage], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[dtype, 128 * head_dim * kv_stage], 1024
            ]
            sO: cute.struct.Align[cute.struct.MemRange[dtype, 128 * head_dim * 2], 1024]

        return SharedStorage

    @cute.struct
    class SharedStorage:
        q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, q_stage * 2]
        kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, kv_stage * 2]
        # Raw mbarriers (FA4-style). s_full / o_full are tcgen05.commit (cnt 1);
        # pfor = softmax(P[0:96]) + correction (cnt 2*128); pfor2 = softmax(P[96:128]) (cnt 128).
        s_full_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        pfor_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        pfor2_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        o_full_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        corr_epi_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
        # softmax -> correction raw full/empty handshakes.
        s0_corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, s_corr_stage * 2]
        s1_corr_mbar_ptr: cute.struct.MemRange[cutlass.Int64, s_corr_stage * 2]
        tmem_dealloc_mbar: cute.struct.MemRange[cutlass.Int64, 1]
        tmem_holding_buf: cutlass.Int32
        sScale: cute.struct.MemRange[
            cutlass.Float32, s_corr_stage * q_stage * softmax_threads
        ]
        sQ: cute.struct.Align[
            cute.struct.MemRange[dtype, 128 * head_dim * q_stage], 1024
        ]
        sK: cute.struct.Align[
            cute.struct.MemRange[dtype, 128 * head_dim * kv_stage], 1024
        ]

    return SharedStorage


def mbar_spin_wait(mbar_ptr: object, phase: object) -> None:
    """FA4-style TIGHT busy-spin mbarrier wait (NO nanosleep backoff).

    ``cute.arch.mbarrier_wait`` lowers to a try_wait with a NANOSLEEP backoff
    loop; on the shallow attention pipeline every handoff (s_full / pfor / pfor2)
    is frequent and short, so a just-missed warp sleeps and resumes late. This
    instead re-checks immediately (busy-spin), resuming the instant the barrier
    flips. ``phase`` is i32 0/1 matching mbarrier_wait's parity. Verbatim port of
    the spike helper.
    """
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(mbar_ptr.toint()).ir_value(),  # pyrefly: ignore[missing-attribute]
            cutlass.Int32(phase).ir_value(),  # pyrefly: ignore[bad-argument-type]
        ],
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1, 10000000;\n\t"
        "@P1 bra DONE;\n\t"
        "bra LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def rcp_approx_ftz(x: object, *, loc: object = None, ip: object = None) -> Float32:
    """FA4-style approximate reciprocal that lowers to ``rcp.approx.ftz.f32``."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],  # pyrefly: ignore[bad-argument-type]
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# ===========================================================================
# FA4 ex2_emulation_2: degree-3 minimax poly software-exp2 on the FMA/ALU
# pipe (packed-f32x2).  Ported from flash_attn.cute.utils.py via the spike
# at .scratch/fa4_step21a_spike.py.  STRING rounding modes ('rn'/'rm') are
# used here -- the installed CuTe DSL requires strings, NOT
# nvvm.RoundingModeKind enums (those TypeError at trace time).
# ===========================================================================

_fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd="rn")
_sub_packed_f32x2 = partial(
    cute.arch.calc_packed_f32x2_op,
    src_c=None,
    calc_func=nvvm.sub_packed_f32x2,
    rnd="rn",
)

# Degree-3 minimax coeffs and round-to-int magic (2^23 + 2^22 = 12582912.0).
_POLY_EX2_DEG3 = (
    1.0,
    0.695146143436431884765625,
    0.227564394474029541015625,
    0.077119089663028717041015625,
)
_FP32_ROUND_INT = float(2**23 + 2**22)


@dsl_user_op
def _combine_int_frac_ex2(
    x_rounded: Float32, frac_ex2: Float32, *, loc: object = None, ip: object = None
) -> Float32:
    """FA4 combine_int_frac_ex2: shift integer part into fp32 exponent field."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x_rounded).ir_value(loc=loc, ip=ip),  # pyrefly: ignore[bad-argument-type]
                Float32(frac_ex2).ir_value(loc=loc, ip=ip),  # pyrefly: ignore[bad-argument-type]
            ],
            "{\n\t"
            ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
            "mov.b32 x_rounded_i, $1;\n\t"
            "mov.b32 frac_ex_i, $2;\n\t"
            "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
            "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
            "mov.b32 $0, out_i;\n\t"
            "}\n",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _evaluate_polynomial_2(x: Float32, y: Float32, poly: tuple) -> tuple:
    """Horner evaluation of poly on (x, y) via packed-f32x2 FMA."""
    deg = len(poly) - 1
    out = (poly[deg], poly[deg])
    for i in range(deg - 1, -1, -1):
        out = _fma_packed_f32x2(out, (x, y), (poly[i], poly[i]))
    return out


def ex2_emulation_2(x: Float32, y: Float32) -> tuple:
    """FA4 ex2_emulation_2: software exp2 for a pair (x, y) via FMA/ALU pipe.

    Fully on the FMA/ALU pipe (packed-f32x2 add/sub/mul + shl/IMAD), NO XU.
    Assumes x, y <= 127.0.  Splits each input into integer + fractional parts
    using the round-to-int magic constant, evaluates the degree-3 poly on the
    frac, and reconstructs 2^x = 2^int * 2^frac.
    """
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    xy_rounded = cute.arch.add_packed_f32x2(
        xy_clamped,
        (_FP32_ROUND_INT, _FP32_ROUND_INT),
        rnd="rm",
    )
    xy_rounded_back = _sub_packed_f32x2(xy_rounded, (_FP32_ROUND_INT, _FP32_ROUND_INT))
    xy_frac = _sub_packed_f32x2(xy_clamped, xy_rounded_back)
    xy_frac_ex2 = _evaluate_polynomial_2(xy_frac[0], xy_frac[1], _POLY_EX2_DEG3)
    x_out = _combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0])
    y_out = _combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1])
    return x_out, y_out


def exp2_split_inplace(
    tLDrS: cute.Tensor,
    scale: Float32,
    minus_max_scale: Float32,
    e2e_freq: int,
    e2e_res: int,
) -> None:
    """FA4-style per-pair exp2 pipe-split applied in place on the S fragment.

    All pairs are scaled first (scale * x + minus_max_scale), then:
      * most pairs  (pair_idx % e2e_freq < e2e_freq - e2e_res)  -> hardware XU
        (cute.arch.exp2 = ex2.approx.ftz.f32)
      * residue pairs (the remaining e2e_res per period)         -> software
        FMA-pipe poly (ex2_emulation_2)
      * the last pair is always routed to the XU (FA4 e2e_frg_limit=1 guard)

    ``e2e_freq`` and ``e2e_res`` are Python-level constants resolved at codegen
    time, so the per-pair conditional becomes a static branch during DSL trace
    (no runtime overhead for the gate itself).
    """
    n = cute.size(tLDrS)
    last_pair_idx = n // 2 - 1
    for i in range(0, n, 2):
        r0, r1 = cute.arch.fma_packed_f32x2(
            (tLDrS[i], tLDrS[i + 1]),
            (scale, scale),
            (minus_max_scale, minus_max_scale),
        )
        tLDrS[i] = r0
        tLDrS[i + 1] = r1
    for pair_idx, i in enumerate(range(0, n, 2)):
        is_last = pair_idx >= last_pair_idx
        use_xu = (pair_idx % e2e_freq) < (e2e_freq - e2e_res) or is_last
        if use_xu:
            tLDrS[i] = cute.arch.exp2(tLDrS[i])
            tLDrS[i + 1] = cute.arch.exp2(tLDrS[i + 1])
        else:
            tLDrS[i], tLDrS[i + 1] = ex2_emulation_2(tLDrS[i], tLDrS[i + 1])  # pyrefly: ignore[bad-argument-type]


def mask_r2p_sm100_rank1(x: cute.Tensor, col_limit: cutlass.Int32) -> None:
    """Mask a rank-1 SM100 score fragment using FA4's R2P-friendly bit pattern."""
    ncol = cute.size(x.shape)
    x_any = cast("Any", x)
    for s in range(cute.ceil_div(ncol, 24)):
        col_limit_s = cutlass.max(col_limit - cutlass.Int32(s * 24), cutlass.Int32(0))
        col_limit_s = cutlass.min(col_limit_s, cutlass.Int32(24))
        mask = (cutlass.Int32(1) << col_limit_s) - cutlass.Int32(1)
        for i in range(min(24, ncol - s * 24)):
            in_bound = cutlass.Boolean(mask & (cutlass.Int32(1) << cutlass.Int32(i)))
            c = s * 24 + i
            x_any[c] = cutlass.Float32(
                cutlass.select_(
                    in_bound, cutlass.Float32(x_any[c]), -cutlass.Float32.inf
                )
            )


def mask_r2p_sm100_range(
    x: cute.Tensor,
    col_start: cutlass.Int32,
    col_limit: cutlass.Int32,
) -> None:
    """Mask a rank-1 score fragment to ``col_start <= col < col_limit``."""
    ncol = cute.size(x.shape)
    x_any = cast("Any", x)
    for s in range(cute.ceil_div(ncol, 24)):
        chunk_start = cutlass.Int32(s * 24)
        col_start_s = cutlass.max(col_start - chunk_start, cutlass.Int32(0))
        col_start_s = cutlass.min(col_start_s, cutlass.Int32(24))
        col_limit_s = cutlass.max(col_limit - chunk_start, cutlass.Int32(0))
        col_limit_s = cutlass.min(col_limit_s, cutlass.Int32(24))
        before_limit = (cutlass.Int32(1) << col_limit_s) - cutlass.Int32(1)
        before_start = (cutlass.Int32(1) << col_start_s) - cutlass.Int32(1)
        mask = before_limit & ~before_start
        for i in range(min(24, ncol - s * 24)):
            in_bound = cutlass.Boolean(mask & (cutlass.Int32(1) << cutlass.Int32(i)))
            c = s * 24 + i
            x_any[c] = cutlass.Float32(
                cutlass.select_(
                    in_bound, cutlass.Float32(x_any[c]), -cutlass.Float32.inf
                )
            )


def causal_mask_t2r(
    tLDrS: cute.Tensor,
    tLDcS: cute.Tensor,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
) -> None:
    """Apply square causal masking to a loaded 128x128 score tile.

    Helion's validated flash path currently only handles square self-attention
    with exact 128-row/column tiles, so the causal predicate is simply
    ``global_k <= global_q``. This mirrors FA4's SM100 mask placement: after the
    T2R score load and before row-max/softmax.
    """
    row_idx = cast("Any", tLDcS)[0][0] + m_block * 128
    col_limit = row_idx + 1 - n_block * 128
    mask_r2p_sm100_rank1(tLDrS, col_limit)


def add_score_bias_t2r(
    tLDrS: cute.Tensor,
    tLDcS: cute.Tensor,
    mBias: cute.Tensor,
    bh: cutlass.Int32,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
    bias_scale: Float32,
) -> None:
    """Add an exact-shape score bias to a loaded 128x128 score row.

    ``tLDrS`` is kept in raw-QK units in the flash kernels; ``bias_scale`` converts
    the source bias into those same units so the existing row-max, alpha, and LSE
    math can continue to apply one shared ``_flash_scale_log2`` later.
    """
    row_idx = cast("Any", tLDcS)[0][0] + m_block * 128
    score = cast("Any", tLDrS)
    coord = cast("Any", tLDcS)
    bias_row = cast("Any", mBias[row_idx, None, bh])
    for i in range(cute.size(tLDrS)):
        col_idx = coord[i][1] + n_block * 128
        score[i] = (
            cutlass.Float32(score[i]) + cutlass.Float32(bias_row[col_idx]) * bias_scale
        )


def add_score_bias_t2r_contiguous(
    tLDrS: cute.Tensor,
    tLDcS: cute.Tensor,
    mBias: cute.Tensor,
    bh: cutlass.Int32,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
    bias_scale: Float32,
    bias_dtype: type[Numeric] = cutlass.Float16,
) -> None:
    """Add score bias when the score fragment covers contiguous columns."""
    row_idx = cast("Any", tLDcS)[0][0] + m_block * 128
    col_start = cast("Any", tLDcS)[0][1] + n_block * 128
    bias_row = cast("Any", mBias[row_idx, None, bh])
    score = cast("Any", tLDrS)
    vec_width = 4
    for i in range(0, cute.size(tLDrS), vec_width):
        bias_vec = cute.arch.load(
            bias_row.iterator + col_start + i,
            ir.VectorType.get([vec_width], cutlass.Uint16.mlir_type),
        )
        for j in range(vec_width):
            bias_val = cast("Any", cutlass.Uint16(bias_vec[j]).bitcast(bias_dtype))
            score[i + j] = (
                cutlass.Float32(score[i + j]) + cutlass.Float32(bias_val) * bias_scale
            )


def add_relative_bias_t2r(
    tLDrS: cute.Tensor,
    tLDcS: cute.Tensor,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
    bias_scale: Float32,
) -> None:
    """Add a relative-position linear bias to a loaded score row."""
    row_idx = cast("Any", tLDcS)[0][0] + m_block * 128
    score = cast("Any", tLDrS)
    coord = cast("Any", tLDcS)
    for i in range(cute.size(tLDrS)):
        col_idx = coord[i][1] + n_block * 128
        score[i] = (
            cutlass.Float32(score[i]) + cutlass.Float32(row_idx - col_idx) * bias_scale
        )


def add_alibi_bias_t2r(
    tLDrS: cute.Tensor,
    tLDcS: cute.Tensor,
    mAlibi: cute.Tensor,
    bh: cutlass.Int32,
    alibi_count: cutlass.Int32,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
    bias_scale: Float32,
) -> None:
    """Add ALiBi-style ``(key - query) * slope`` to a loaded score row."""
    row_idx = cast("Any", tLDcS)[0][0] + m_block * 128
    score = cast("Any", tLDrS)
    coord = cast("Any", tLDcS)
    alibi = cast("Any", mAlibi)
    slope = cutlass.Float32(alibi[bh % alibi_count])
    for i in range(cute.size(tLDrS)):
        col_idx = coord[i][1] + n_block * 128
        score[i] = (
            cutlass.Float32(score[i])
            + cutlass.Float32(col_idx - row_idx) * slope * bias_scale
        )


def sliding_window_mask_t2r(
    tLDrS: cute.Tensor,
    tLDcS: cute.Tensor,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
    window_size: cutlass.Int32,
) -> None:
    """Apply a causal local-attention window: ``0 <= query-key <= window``."""
    row_idx = cast("Any", tLDcS)[0][0] + m_block * 128
    block_start = n_block * 128
    col_start = row_idx - window_size - block_start
    col_limit = row_idx + 1 - block_start
    mask_r2p_sm100_range(tLDrS, col_start, col_limit)


def prefix_lm_mask_t2r(
    tLDrS: cute.Tensor,
    tLDcS: cute.Tensor,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
    prefix_length: cutlass.Int32,
) -> None:
    """Apply a prefix-LM mask: prefix keys are visible, suffix is causal."""
    row_idx = cast("Any", tLDcS)[0][0] + m_block * 128
    col_limit = cutlass.max(row_idx + 1, prefix_length) - n_block * 128
    mask_r2p_sm100_rank1(tLDrS, col_limit)


def document_mask_t2r(
    tLDrS: cute.Tensor,
    tLDcS: cute.Tensor,
    mDoc: cute.Tensor,
    bh: cutlass.Int32,
    doc_heads_per_batch: cutlass.Int32,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
) -> None:
    """Apply a causal same-document mask using doc ids shaped as ``(batch, seq)``."""
    row_idx = cast("Any", tLDcS)[0][0] + m_block * 128
    score = cast("Any", tLDrS)
    coord = cast("Any", tLDcS)
    doc = cast("Any", mDoc)
    doc_bh = bh // doc_heads_per_batch
    row_doc = doc[row_idx, doc_bh]
    for i in range(cute.size(tLDrS)):
        col_idx = coord[i][1] + n_block * 128
        keep = (row_idx >= col_idx) & (row_doc == doc[col_idx, doc_bh])
        score[i] = cutlass.Float32(
            cutlass.select_(keep, cutlass.Float32(score[i]), -cutlass.Float32.inf)
        )


def document_tile_maybe_active(
    mDoc: cute.Tensor,
    bh: cutlass.Int32,
    doc_heads_per_batch: cutlass.Int32,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
) -> cutlass.Boolean:
    """Conservative whole-tile predicate for causal same-document masking."""
    doc = cast("Any", mDoc)
    doc_bh = bh // doc_heads_per_batch
    q_start = m_block * 128
    k_start = n_block * 128
    q_bits = cutlass.Int32(0)
    k_bits = cutlass.Int32(0)
    for i in range(128):
        q_doc = cast("Any", cutlass.Int32(doc[q_start + i, doc_bh]))
        k_doc = cast("Any", cutlass.Int32(doc[k_start + i, doc_bh]))
        q_bits = q_bits | (cutlass.Int32(1) << (q_doc & cutlass.Int32(31)))
        k_bits = k_bits | (cutlass.Int32(1) << (k_doc & cutlass.Int32(31)))
    return (n_block <= m_block) & ((q_bits & k_bits) != 0)


def _document_tile_bits_warp(
    doc_tensor: object,
    doc_bh: cutlass.Int32,
    start: cutlass.Int32,
) -> cutlass.Int32:
    """Modulo-32 doc-id fingerprint used only for conservative tile pruning.

    Collisions are allowed: the exact per-score document mask still runs for every
    retained tile, so false positives only cost extra work and never affect output.
    """
    doc = cast("Any", doc_tensor)
    lane = cute.arch.lane_idx()
    bits = cutlass.Int32(0)
    for i in range(4):
        doc_id = cast(
            "Any",
            cutlass.Int32(doc[start + lane + cutlass.Int32(i * 32), doc_bh]),
        )
        bits = bits | (cutlass.Int32(1) << (doc_id & cutlass.Int32(31)))
    for offset in (16, 8, 4, 2, 1):
        bits = bits | cute.arch.shuffle_sync_bfly(
            bits,
            offset=offset,
            mask=-1,
            mask_and_clamp=31,
        )
    return bits


def softcap_t2r(
    tLDrS: cute.Tensor,
    score_scale_log2: Float32,
    softcap_log2: Float32,
) -> None:
    """Apply ``softcap * tanh(score / softcap)`` in raw-QK score units."""
    score = cast("Any", tLDrS)
    raw_softcap = softcap_log2 / score_scale_log2
    for i in range(cute.size(tLDrS)):
        score[i] = raw_softcap * cute.math.tanh(
            cutlass.Float32(score[i]) * score_scale_log2 / softcap_log2
        )


def causal_mask_t2r_chunk(
    tLDrS: cute.Tensor,
    tLDcS: cute.Tensor,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
    chunk_idx: int,
) -> None:
    """Causal-mask one FA4 32-column score chunk after a T2R load."""
    row_idx = cast("Any", tLDcS)[0][0] + m_block * 128
    chunk_cols = cute.size(tLDrS.shape)
    col_limit = row_idx + 1 - n_block * 128 - cutlass.Int32(chunk_idx * chunk_cols)
    # Keep the FA4-style straight-line mask. A dynamic chunk bounds branch needs
    # a JIT helper here and regresses B200 causal hd64 codegen.
    mask_r2p_sm100_rank1(tLDrS, col_limit)


def fa4_exp2_convert_rowsum(
    tLDrS: cute.Tensor,
    tSTrS_e: cute.Tensor,
    scale: Float32,
    minus_max_scale: Float32,
    e2e_freq: int,
    e2e_res: int,
    io_dtype: object = cutlass.Float16,
) -> Float32:
    """FUSED FA4 softmax: scale-subtract + exp2(pipe-split) + f32->f16 convert +
    deferred packed row-sum, returning the row-sum.

    Faithful port of the spike's PROVEN zero-spill structure
    (.scratch/fa4_faithful_spike.py ``_lean_exp_convert`` FA4_EXP=="sp"/"fa4ip" ->
    ``_sp_exp2_convert`` + ``_row_sum_packed``). Replaces the prior 3-pass
    [exp2_split_inplace (full-row scale-subtract THEN a full-row exp2 split) +
    fadd_reduce_packed (a 3rd full-row sum pass) + separate frag-by-frag convert]
    sequence whose simultaneous live set (full f32 row + the exp2-emulation temps +
    the reduction accumulators + the fp16 buffer) overran the 200-reg grant and
    spilled (~109 LDL/STL in the softmax region).

    The fused convert pass holds only ONE 32-elem fragment's transients at a time:
    per 32-element fragment, in order:
      (1) packed-f32x2 scale-subtract (scale * x + minus_max_scale);
      (2) exp2 with the FA4 per-pair pipe-split -- most pairs to the hardware XU
          (cute.arch.exp2 = ex2.approx.ftz.f32), the e2e_res residue pairs per period
          to the FMA-pipe poly (ex2_emulation_2); the last pair of the row is always
          routed to the XU (FA4 e2e_frg_limit=1 guard);
      (3) ONE 32-wide .store(.load().to(fp16)) of the fragment into tSTrS_e (frees
          the fragment's f32 temps before the next fragment).
    The row-sum is then a SEPARATE deferred pass (``_row_sum_packed``) over the now
    in-place exp'd ``tLDrS`` -- exactly the spike ordering. Deferring it keeps the 4
    packed accumulators OFF the peak during the exp2/convert burst (folding them in
    inline re-introduced ~45 spills/softmax, the convert temps + 8 accum regs
    coexisting); the spike measured this deferred form as STACK:0.

    The pair gate matches ``exp2_split_inplace`` EXACTLY (global pair index across the
    full row), so the numerics are byte-identical to the prior split path -- only the
    op grouping / register lifetimes change. ``e2e_freq`` / ``e2e_res`` are codegen
    constants so the per-pair branch is resolved statically during trace.
    """
    n = cute.size(tLDrS)
    assert n % 8 == 0, "fa4_exp2_convert_rowsum requires a multiple-of-8 row"
    frg_tile = 32
    frg_cnt = n // frg_tile
    last_pair_idx = n // 2 - 1
    tLDrS_frg = cute.logical_divide(tLDrS, cute.make_layout(frg_tile))
    tSTrS_e_frg = cute.logical_divide(tSTrS_e, cute.make_layout(frg_tile))
    for j in range(frg_cnt):
        base_pair = (j * frg_tile) // 2
        for k in range(0, frg_tile, 2):
            r0, r1 = cute.arch.fma_packed_f32x2(
                (tLDrS_frg[k, j], tLDrS_frg[k + 1, j]),
                (scale, scale),
                (minus_max_scale, minus_max_scale),
            )
            pair_idx = base_pair + k // 2
            is_last = pair_idx >= last_pair_idx
            use_xu = (pair_idx % e2e_freq) < (e2e_freq - e2e_res) or is_last
            if use_xu:
                r0 = cute.arch.exp2(r0)
                r1 = cute.arch.exp2(r1)
            else:
                r0, r1 = ex2_emulation_2(r0, r1)  # pyrefly: ignore[bad-argument-type]
            tLDrS_frg[k, j] = r0
            tLDrS_frg[k + 1, j] = r1
        # ONE 32-wide convert -> staged-P fp16 chunk; frees the fragment's temps.
        tSTrS_e_frg[None, j].store(  # pyrefly: ignore[missing-attribute]
            tLDrS_frg[None, j].load().to(io_dtype)  # pyrefly: ignore[missing-attribute]
        )
    # DEFERRED packed row-sum (spike _row_sum_packed): 4 packed-f32x2 accumulators
    # over the now-exp'd row, AFTER the convert burst so they never coexist with
    # the convert temps -> off the spill peak.
    s0 = (tLDrS[0], tLDrS[1])
    s1 = (tLDrS[2], tLDrS[3])
    s2 = (tLDrS[4], tLDrS[5])
    s3 = (tLDrS[6], tLDrS[7])
    for i in range(8, n, 8):
        s0 = _add_packed_f32x2(s0, (tLDrS[i + 0], tLDrS[i + 1]))
        s1 = _add_packed_f32x2(s1, (tLDrS[i + 2], tLDrS[i + 3]))
        s2 = _add_packed_f32x2(s2, (tLDrS[i + 4], tLDrS[i + 5]))
        s3 = _add_packed_f32x2(s3, (tLDrS[i + 6], tLDrS[i + 7]))
    s0 = _add_packed_f32x2(s0, s1)
    s2 = _add_packed_f32x2(s2, s3)
    s0 = _add_packed_f32x2(s0, s2)
    return cutlass.Float32(s0[0]) + cutlass.Float32(s0[1])  # pyrefly: ignore[bad-argument-type]


def _fmax_reduce_chunk(frg: cute.Tensor, init_val: Float32) -> Float32:
    """fmax over a small fragment ``frg`` (one 32-elem t2r chunk) with 2 scalar
    accumulators, folding the running ``init_val`` row-max in. Used by the
    chunked-t2r row-max pass so the full 128-elem row is NEVER held -- verbatim
    port of the spike ``_fmax_reduce_chunk`` (.scratch/fa4_faithful_spike.py)."""
    n = cute.size(frg)
    lm0 = cute.arch.fmax(init_val, frg[0])
    lm0 = cute.arch.fmax(lm0, frg[1])
    lm1 = cute.arch.fmax(frg[2], frg[3])
    for i in range(4, n, 4):
        lm0 = cute.arch.fmax(lm0, cute.arch.fmax(frg[i + 0], frg[i + 1]))
        lm1 = cute.arch.fmax(lm1, cute.arch.fmax(frg[i + 2], frg[i + 3]))
    return cute.arch.fmax(lm0, lm1)


def fa4_disc_rowmax(
    tiled_ld: object,
    tLDtS: cute.Tensor,
    tLDcS: cute.Tensor,
    row_max: Float32,
    ld_chunks: int,
) -> Float32:
    """CHUNKED-t2r PASS 1 (row-max). For each of ``ld_chunks`` 32-elem column
    chunks, t2r ONE chunk into a small fragment, fold into the running max via
    scalar fmax, then FREE the chunk -- so the full fp32 row is NEVER simultaneously
    resident (peak live = ONE 32-elem fragment). This is the spike's PROVEN
    zero-spill ``_disc_pass1_max`` structure (.scratch/fa4_faithful_spike.py): the
    KEYSTONE finding is that the chunked-t2r ("disc") softmax body is the ONLY one
    that closes the FA4 200/64/48/24 setmaxnreg split with ZERO spill, whereas the
    whole-row ("sp") body keeps a 128-f32 row resident and spills past the grant.

    A SINGLE trailing fence (no per-chunk fence) lets the ``ld_chunks`` t2r issues
    pipeline back-to-back while the fmax folds interleave with the in-flight reads.
    ``tLDtS`` has the chunked partition shape ((32,1), ld_chunks, 1, 1) (chunk =
    mode[1]); ``tLDcS`` is the matching coord partition for the per-chunk shape.

    L2: the per-chunk fragment SHAPE is loop-invariant (same 32-elem chunk every
    iteration), so it is read ONCE before the loop rather than re-sliced per chunk.
    """
    ld_shape = tLDcS[None, 0, None, None].shape  # pyrefly: ignore[missing-attribute]
    for ci in range(ld_chunks):
        frg = cute.make_rmem_tensor(ld_shape, cutlass.Float32)
        cute.copy(tiled_ld, tLDtS[None, ci, None, None], frg)
        row_max = _fmax_reduce_chunk(frg, row_max)
    cute.arch.fence_view_async_tmem_load()
    return row_max


def fa4_disc_rowmax_causal(
    tiled_ld: object,
    tLDtS: cute.Tensor,
    tLDcS: cute.Tensor,
    row_max: Float32,
    ld_chunks: int,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
) -> Float32:
    """Causal variant of ``fa4_disc_rowmax`` for the FA4 topology."""
    ld_shape = tLDcS[None, 0, None, None].shape  # pyrefly: ignore[missing-attribute]
    for ci in range(ld_chunks):
        frg = cute.make_rmem_tensor(ld_shape, cutlass.Float32)
        cute.copy(tiled_ld, tLDtS[None, ci, None, None], frg)
        causal_mask_t2r_chunk(
            frg,
            cast("cute.Tensor", tLDcS[None, ci, None, None]),
            m_block,
            n_block,
            ci,
        )
        row_max = _fmax_reduce_chunk(frg, row_max)
    cute.arch.fence_view_async_tmem_load()
    return row_max


def fa4_disc_exp_convert_store(
    tiled_ld: object,
    tLDtS: cute.Tensor,
    tLDcS: cute.Tensor,
    tiled_st: object,
    tSTtS: cute.Tensor,
    tSTcS: cute.Tensor,
    scale: Float32,
    minus_max_scale: Float32,
    e2e_freq: int,
    e2e_res: int,
    e2e_offset: int,
    pfor_ptr_stage: object,
    pfor2_ptr_stage: object,
    p_store_split: int,
    p_store_chunks: int,
    io_dtype: object = cutlass.Float16,
) -> Float32:
    """CHUNKED-t2r PASS 2 (exp + convert + r2t-store, FUSED with row-sum + the
    staged-P MMA handshake). For each P-store chunk ci: t2r 32 S cols into a small
    f32 fragment, packed-f32x2 scale-subtract, exp2 with the FA4 per-pair pipe-split,
    convert that 32-elem fragment to a 16-reg fp16 chunk and r2t-STORE it in place
    over the same TMEM cols, fold the post-exp f32 chunk into the running row-sum,
    then FREE the chunk. Neither the full fp32 S NOR the full fp16 P is ever fully
    materialized -- peak live = ONE 32-elem fragment + its 16-reg fp16 temp. Verbatim
    port of the spike's ``_disc_pass2_exp`` serial form (.scratch/fa4_faithful_spike.py).

    STAGED-P handshake preserved EXACTLY: after the chunk at ``p_store_split - 1``
    (the 3/4 boundary = first 96 kv) a fence + ``mbarrier_arrive(pfor)`` releases the
    MMA's first PV K-chunk group; after the last chunk a fence + ``mbarrier_arrive(
    pfor2)`` releases the final group. Load chunk ci and store chunk ci alias the
    SAME TMEM cols (read-before-write in place), so this is safe.

    The exp2 pipe-split gate is CHUNK-LOCAL (pair index ``i`` within the 32-elem
    chunk == FA4's ``k``; ``ci`` == FA4's fragment index ``j``; last chunk forced to
    XU via the e2e_frg_limit=1 guard) -- the FA4 ``apply_exp2_convert`` gate. This
    differs from the whole-row helper's global-pair gate, so the XU/poly routing
    differs slightly, but both stay within the fp16 rounding floor (~2.4e-4).
    """
    p_sum = cutlass.Float32(0.0)
    # L2: the per-chunk fragment SHAPE is loop-invariant -> read once, not per chunk.
    ld_shape = tLDcS[None, 0, None, None].shape  # pyrefly: ignore[missing-attribute]
    for ci in range(p_store_chunks):
        frg = cute.make_rmem_tensor(ld_shape, cutlass.Float32)
        cute.copy(tiled_ld, tLDtS[None, ci, None, None], frg)
        last_frag = ci >= p_store_chunks - 1
        _disc_chunk_exp(
            frg, scale, minus_max_scale, e2e_freq, e2e_res, e2e_offset, last_frag
        )
        _disc_chunk_convert_store(frg, tiled_st, tSTtS, tSTcS, ci, io_dtype)
        p_sum = p_sum + _disc_chunk_rowsum(frg)
        if ci == p_store_split - 1:
            cute.arch.fence_view_async_tmem_store()
            cute.arch.mbarrier_arrive(pfor_ptr_stage)
    cute.arch.fence_view_async_tmem_store()
    cute.arch.mbarrier_arrive(pfor2_ptr_stage)
    return p_sum


def fa4_disc_exp_convert_store_causal(
    tiled_ld: object,
    tLDtS: cute.Tensor,
    tLDcS: cute.Tensor,
    tiled_st: object,
    tSTtS: cute.Tensor,
    tSTcS: cute.Tensor,
    scale: Float32,
    minus_max_scale: Float32,
    e2e_freq: int,
    e2e_res: int,
    e2e_offset: int,
    pfor_ptr_stage: object,
    pfor2_ptr_stage: object,
    p_store_split: int,
    p_store_chunks: int,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
    io_dtype: object = cutlass.Float16,
) -> Float32:
    """Causal variant of ``fa4_disc_exp_convert_store``."""
    p_sum = cutlass.Float32(0.0)
    ld_shape = tLDcS[None, 0, None, None].shape  # pyrefly: ignore[missing-attribute]
    for ci in range(p_store_chunks):
        frg = cute.make_rmem_tensor(ld_shape, cutlass.Float32)
        cute.copy(tiled_ld, tLDtS[None, ci, None, None], frg)
        causal_mask_t2r_chunk(
            frg,
            cast("cute.Tensor", tLDcS[None, ci, None, None]),
            m_block,
            n_block,
            ci,
        )
        last_frag = ci >= p_store_chunks - 1
        _disc_chunk_exp(
            frg, scale, minus_max_scale, e2e_freq, e2e_res, e2e_offset, last_frag
        )
        _disc_chunk_convert_store(frg, tiled_st, tSTtS, tSTcS, ci, io_dtype)
        p_sum = p_sum + _disc_chunk_rowsum(frg)
        if ci == p_store_split - 1:
            cute.arch.fence_view_async_tmem_store()
            cute.arch.mbarrier_arrive(pfor_ptr_stage)
    cute.arch.fence_view_async_tmem_store()
    cute.arch.mbarrier_arrive(pfor2_ptr_stage)
    return p_sum


def _disc_chunk_exp(
    frg: cute.Tensor,
    scale: Float32,
    minus_max_scale: Float32,
    e2e_freq: int,
    e2e_res: int,
    e2e_offset: int,
    last_frag: bool,
) -> None:
    """In-place packed scale-subtract then exp2(pipe-split) over ONE 32-elem chunk.

    The pair index ``i`` within the chunk == FA4's ``k``; ``last_frag`` is FA4's
    e2e_frg_limit=1 guard (the row's last chunk is forced to the hardware XU).
    ``e2e_offset`` shifts the residue phase for the two softmax stages.
    Verbatim from the serial ``fa4_disc_exp_convert_store`` body; factored out so the
    serial and software-pipelined pass2 share the SAME per-chunk numerics."""
    nf = cute.size(frg)
    for i in range(0, nf, 2):
        r0, r1 = cute.arch.fma_packed_f32x2(
            (frg[i], frg[i + 1]),
            (scale, scale),
            (minus_max_scale, minus_max_scale),
        )
        use_xu = ((i + e2e_offset) % e2e_freq) < (e2e_freq - e2e_res) or last_frag
        if use_xu:
            frg[i] = cute.arch.exp2(r0)
            frg[i + 1] = cute.arch.exp2(r1)
        else:
            frg[i], frg[i + 1] = ex2_emulation_2(r0, r1)  # pyrefly: ignore[bad-argument-type]


def _disc_chunk_convert_store(
    frg: cute.Tensor,
    tiled_st: object,
    tSTtS: cute.Tensor,
    tSTcS: cute.Tensor,
    ci: int,
    io_dtype: object = cutlass.Float16,
) -> None:
    """Convert one ALREADY-exp'd 32-elem f32 chunk to a 16-reg fp16 chunk and
    r2t-STORE it in place over the same TMEM cols. No row-sum (the caller folds the
    sum off the post-exp f32 values via ``_disc_chunk_rowsum``)."""
    st_shape = tSTcS[None, None, 0].shape  # pyrefly: ignore[missing-attribute]
    pchunk = cute.make_rmem_tensor(st_shape, cutlass.Float32)
    pchunk_e = cute.make_tensor(
        cute.recast_ptr(pchunk.iterator, dtype=io_dtype), frg.layout
    )
    pchunk_e.store(frg.load().to(io_dtype))  # pyrefly: ignore[missing-attribute]
    cute.copy(tiled_st, pchunk, tSTtS[None, None, ci])


def _disc_chunk_rowsum(frg: cute.Tensor) -> Float32:
    """Fold one post-exp chunk into a row-sum via packed-f32x2 accumulators."""
    return fadd_reduce_packed(frg)


@dsl_user_op
def _disc_pin_frag(frg: cute.Tensor, *, loc: object = None, ip: object = None) -> None:
    """Inline-asm scheduling barrier that PINS a just-prefetched t2r fragment.

    The native ``cute.copy`` t2r lowers to a no-side-effect tcgen05.ld that ptxas
    freely re-schedules ADJACENT to its FFMA2 consumer (re-derivation), defeating the
    software-pipeline prefetch (campaign night-8: prior intra-iteration chunk-prefetch
    was sometimes ptxas-re-scheduled to neutral). This op gives the SASS scheduler a
    genuine data dependency on every element of the prefetched fragment via "f"
    (read-only) operand constraints with an EMPTY (no-op) side-effecting asm body:
    ``has_side_effects=True`` + a live "f" use means NVVM/ptxas cannot fold/delete it
    NOR sink the producing LDTM past it, so the prefetched load stays live ABOVE the
    barrier and its TMEM-read latency overlaps the current chunk's exp2 burst. The body
    emits NOTHING (the values pass through), so it costs zero SASS instructions; it only
    constrains scheduling. Verbatim port of the spike ``_sched_barrier`` "operand" form
    (.scratch/fa4_faithful_spike.py), which was the DECISIVE pin mechanism (the empty
    "bar"/"memfence" forms were re-derived to byte-identical SASS)."""
    n = cute.size(frg)
    ops = [
        cutlass.Float32(frg[i]).ir_value(loc=loc, ip=ip)  # pyrefly: ignore[bad-argument-type]
        for i in range(n)
    ]
    cons = ",".join(["f"] * n)
    llvm.inline_asm(
        None,
        ops,
        "// disc t2r pin\n",
        cons,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def fa4_disc_exp_convert_store_pipe(
    tiled_ld: object,
    tLDtS: cute.Tensor,
    tLDcS: cute.Tensor,
    tiled_st: object,
    tSTtS: cute.Tensor,
    tSTcS: cute.Tensor,
    scale: Float32,
    minus_max_scale: Float32,
    e2e_freq: int,
    e2e_res: int,
    e2e_offset: int,
    pfor_ptr_stage: object,
    pfor2_ptr_stage: object,
    p_store_split: int,
    p_store_chunks: int,
    pipe_depth: int,
    io_dtype: object = cutlass.Float16,
) -> Float32:
    """SOFTWARE-PIPELINED chunked-t2r PASS 2 (the L1 lever). Same numerics + staged-P
    handshake + zero-spill peak (ONE chunk + a bounded pipeline window) as the serial
    ``fa4_disc_exp_convert_store``, but the t2r of chunk ``ci+pipe_depth`` is prefetched
    (async tcgen05.ld) BEFORE chunk ``ci``'s scale/exp2/convert/store, so the next
    chunk's TMEM-read latency overlaps the current chunk's XU(exp2) burst instead of
    stalling on the bulk ``tcgen05.wait::ld`` (attacks long_scoreboard).

    Ordering per the spike's PROVEN FA4_T2R_PIN form (.scratch/fa4_faithful_spike.py
    ``_disc_pass2_exp``): issue the next prefetch, then ``_disc_pin_frag`` (a
    side-effecting inline-asm scheduling barrier with a live data dependency on the
    prefetched fragment) so ptxas cannot sink the freshly-issued LDTM down to be
    adjacent to its later FFMA2 consumer, then ``fence_view_async_tmem_load`` to drain
    chunk ``ci``'s OWN load (issued ``pipe_depth`` iters ago -> already complete), then
    consume chunk ``ci``. ``n_buf = pipe_depth + 1`` distinct fragment buffers so the
    slot being consumed (``ci % n_buf``) is never the slot being prefetched
    (``(ci+pipe_depth) % n_buf``). Load chunk ci and store chunk ci alias the SAME TMEM
    cols (read-before-write in place), so this is safe.

    The pin is the decisive bit: campaign night-8 found a PLAIN prefetch was often
    ptxas-re-scheduled to neutral, and the spike confirmed only the operand-pin form
    actually held the overlap. ``pipe_depth=1`` is NOT this path (the caller routes that
    to the serial helper); this is only entered for ``pipe_depth >= 2``. When
    ``pipe_depth >= p_store_chunks`` (the hd64 default: 4 chunks, depth 4), all
    chunks are prefetched and pinned in the prologue, then consumed without steady
    prefetches. That full-prologue mode intentionally trades a larger fragment
    window for fewer loop-carried t2r scheduling points."""
    p_sum = cutlass.Float32(0.0)
    ld_shape = tLDcS[None, 0, None, None].shape  # pyrefly: ignore[missing-attribute]
    n_buf = pipe_depth + 1
    frgs = [cute.make_rmem_tensor(ld_shape, cutlass.Float32) for _ in range(n_buf)]

    def _prefetch(slot: int, idx: int) -> None:
        cute.copy(tiled_ld, tLDtS[None, idx, None, None], frgs[slot])

    # Prologue: issue the first ``pipe_depth`` chunk loads so the steady-state body
    # always has a prefetch in flight ``pipe_depth`` chunks ahead. If depth covers
    # all chunks, this becomes the intentional full-prologue mode described above.
    for j in range(min(pipe_depth, p_store_chunks)):
        _prefetch(j % n_buf, j)
        _disc_pin_frag(frgs[j % n_buf])
    for ci in range(p_store_chunks):
        nxt = ci + pipe_depth
        cur = frgs[ci % n_buf]
        if nxt < p_store_chunks:
            _prefetch(nxt % n_buf, nxt)
            _disc_pin_frag(frgs[nxt % n_buf])
        # Drain cur's OWN load (issued pipe_depth iters ago = complete). This bulk
        # wait does NOT re-drain the just-issued prefetch onto the critical path
        # because the pin keeps it live above this fence in the IR.
        cute.arch.fence_view_async_tmem_load()
        last_frag = ci >= p_store_chunks - 1
        _disc_chunk_exp(
            cur, scale, minus_max_scale, e2e_freq, e2e_res, e2e_offset, last_frag
        )
        _disc_chunk_convert_store(cur, tiled_st, tSTtS, tSTcS, ci, io_dtype)
        p_sum = p_sum + _disc_chunk_rowsum(cur)
        if ci == p_store_split - 1:
            cute.arch.fence_view_async_tmem_store()
            cute.arch.mbarrier_arrive(pfor_ptr_stage)
    cute.arch.fence_view_async_tmem_store()
    cute.arch.mbarrier_arrive(pfor2_ptr_stage)
    return p_sum


def fa4_disc_exp_convert_store_pipe_causal(
    tiled_ld: object,
    tLDtS: cute.Tensor,
    tLDcS: cute.Tensor,
    tiled_st: object,
    tSTtS: cute.Tensor,
    tSTcS: cute.Tensor,
    scale: Float32,
    minus_max_scale: Float32,
    e2e_freq: int,
    e2e_res: int,
    e2e_offset: int,
    pfor_ptr_stage: object,
    pfor2_ptr_stage: object,
    p_store_split: int,
    p_store_chunks: int,
    pipe_depth: int,
    m_block: cutlass.Int32,
    n_block: cutlass.Int32,
    io_dtype: object = cutlass.Float16,
) -> Float32:
    """Causal variant of ``fa4_disc_exp_convert_store_pipe``."""
    p_sum = cutlass.Float32(0.0)
    ld_shape = tLDcS[None, 0, None, None].shape  # pyrefly: ignore[missing-attribute]
    n_buf = pipe_depth + 1
    frgs = [cute.make_rmem_tensor(ld_shape, cutlass.Float32) for _ in range(n_buf)]

    def _prefetch(slot: int, idx: int) -> None:
        cute.copy(tiled_ld, tLDtS[None, idx, None, None], frgs[slot])

    for j in range(min(pipe_depth, p_store_chunks)):
        _prefetch(j % n_buf, j)
        _disc_pin_frag(frgs[j % n_buf])
    for ci in range(p_store_chunks):
        nxt = ci + pipe_depth
        cur = frgs[ci % n_buf]
        if nxt < p_store_chunks:
            _prefetch(nxt % n_buf, nxt)
            _disc_pin_frag(frgs[nxt % n_buf])
        cute.arch.fence_view_async_tmem_load()
        causal_mask_t2r_chunk(
            cur,
            cast("cute.Tensor", tLDcS[None, ci, None, None]),
            m_block,
            n_block,
            ci,
        )
        last_frag = ci >= p_store_chunks - 1
        _disc_chunk_exp(
            cur, scale, minus_max_scale, e2e_freq, e2e_res, e2e_offset, last_frag
        )
        _disc_chunk_convert_store(cur, tiled_st, tSTtS, tSTcS, ci, io_dtype)
        p_sum = p_sum + _disc_chunk_rowsum(cur)
        if ci == p_store_split - 1:
            cute.arch.fence_view_async_tmem_store()
            cute.arch.mbarrier_arrive(pfor_ptr_stage)
    cute.arch.fence_view_async_tmem_store()
    cute.arch.mbarrier_arrive(pfor2_ptr_stage)
    return p_sum


def rescale_o_tmem(
    tOtO: cute.Tensor,
    alpha: object,
    tidx: object,
    head_dim: int,
    rescale_chunk_cols: int = 0,
) -> None:
    """O = alpha * O in place in TMEM (t2r -> scale -> r2t).

    hd64 defaults to 32-col chunks to stay within the tightest register budgets.
    Manual FA4 experiments can opt into 64-col chunks to reduce loop/address
    overhead where that larger fragment still compiles.
    """
    corr_tile_size = 32 if head_dim == 64 else 16
    if rescale_chunk_cols in (16, 32, 64) and head_dim % rescale_chunk_cols == 0:
        corr_tile_size = rescale_chunk_cols
    cO = cute.make_identity_tensor((128, head_dim))
    tOcO = cute.make_tensor(cO.iterator, cO.layout)
    ld_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition(corr_tile_size)), cutlass.Float32
    )
    st_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition(corr_tile_size)), cutlass.Float32
    )
    tOtO_i_layout = cute.composition(
        tOtO.layout, cute.make_layout((128, corr_tile_size))
    )
    tOcO_i_layout = cute.composition(
        tOcO.layout, cute.make_layout((128, corr_tile_size))
    )
    tOtO_i = cute.make_tensor(tOtO.iterator, tOtO_i_layout)
    tOcO_i = cute.make_tensor(tOcO.iterator, tOcO_i_layout)
    tiled_ld = tcgen05.make_tmem_copy(ld_atom, tOtO_i)
    tiled_st = tcgen05.make_tmem_copy(st_atom, tOtO_i)
    thr_ld = tiled_ld.get_slice(tidx)
    thr_st = tiled_st.get_slice(tidx)
    tLD = thr_ld.partition_S(tOtO_i)
    tLDc = thr_ld.partition_D(tOcO_i)
    tST = thr_st.partition_D(tOtO_i)
    for i in range(head_dim // corr_tile_size):
        tLD_i = cute.make_tensor(tLD.iterator + i * corr_tile_size, tLD.layout)
        tST_i = cute.make_tensor(tST.iterator + i * corr_tile_size, tST.layout)
        reg = cute.make_rmem_tensor(tLDc.shape, cutlass.Float32)
        cute.copy(tiled_ld, tLD_i, reg)
        _scale_fragment_packed_f32x2(reg, alpha)
        cute.copy(tiled_st, reg, tST_i)


def _scale_fragment_packed_f32x2(frg: cute.Tensor, scale: object) -> None:
    """Scale an even-sized fp32 register fragment with packed f32x2 muls."""
    n = cute.size(frg)
    assert n % 2 == 0, "_scale_fragment_packed_f32x2 requires even fragment size"
    for i in range(0, n, 2):
        r0, r1 = cute.arch.mul_packed_f32x2(
            (frg[i], frg[i + 1]),
            (scale, scale),
        )
        frg[i] = r0
        frg[i + 1] = r1


# ===========================================================================
# FA4 multi-accumulator packed reductions (flash_attn.cute.utils fadd_reduce /
# fmax_reduce, arch>=100 path).  The naive cute TensorSSA ``.reduce(ADD/MAX)``
# over the per-thread 128-wide softmax row lowers to a SINGLE-accumulator
# serial FADD/FMNMX chain (~128 deep -> a ~512-cycle serial critical path per
# KV tile).  SASS diff vs FA4 (.scratch/sass_diff/SASS_DIFF_REPORT.md) pinned
# this as the dominant softmax-consumer stall: the scheduler blocks on the
# running accumulator even though independent exp2 work exists, dropping issue
# rate to ~1/14 cyc and starving the XU (exp2) pipe to ~42% (FA4 ~75%).  These
# helpers break the chain into 4 INDEPENDENT packed-f32x2 accumulators (8
# partial sums in flight), hiding the ~4-cycle op latency so the scheduler can
# interleave the exp2 stream.  Faithful port of FA4's reductions; require
# cute.size(frg) % 8 == 0 (true for the 128-wide row).
# ===========================================================================

_add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd="rn")


@dsl_user_op
def _fmax3(
    a: object, b: object, c: object = None, *, loc: object = None, ip: object = None
) -> Float32:
    """2- or 3-input f32 max via nvvm.fmax (lowers to FMNMX / FMNMX3).

    The installed CuTe-DSL ``nvvm.fmax`` infers its result type (no leading
    result-type positional, unlike the FA4 reference); passing ``T.f32()`` as a
    3rd positional is rejected. Validated in .scratch/red_validate_spike.py.
    """
    return cutlass.Float32(
        nvvm.fmax(
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),  # pyrefly: ignore[bad-argument-type]
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),  # pyrefly: ignore[bad-argument-type]
            c=cutlass.Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,  # pyrefly: ignore[bad-argument-type]
            loc=loc,
            ip=ip,
        )
    )


def fadd_reduce_packed(frg: cute.Tensor, init_val: object = None) -> Float32:
    """Sum-reduce a register fragment with 4 packed-f32x2 accumulators.

    Mirrors flash_attn.cute.utils.fadd_reduce (arch>=100). ``frg`` is a flat
    fp32 rmem fragment with a multiple-of-8 element count.
    """
    n = cute.size(frg)
    assert n % 8 == 0, "fadd_reduce_packed requires a multiple-of-8 fragment"
    local_sum = [
        (frg[0], frg[1]),
        (frg[2], frg[3]),
        (frg[4], frg[5]),
        (frg[6], frg[7]),
    ]
    if init_val is not None:
        local_sum[0] = _add_packed_f32x2((init_val, 0.0), local_sum[0])
    for i in range(8, n, 8):
        local_sum[0] = _add_packed_f32x2(local_sum[0], (frg[i + 0], frg[i + 1]))
        local_sum[1] = _add_packed_f32x2(local_sum[1], (frg[i + 2], frg[i + 3]))
        local_sum[2] = _add_packed_f32x2(local_sum[2], (frg[i + 4], frg[i + 5]))
        local_sum[3] = _add_packed_f32x2(local_sum[3], (frg[i + 6], frg[i + 7]))
    local_sum[0] = _add_packed_f32x2(local_sum[0], local_sum[1])
    local_sum[2] = _add_packed_f32x2(local_sum[2], local_sum[3])
    local_sum[0] = _add_packed_f32x2(local_sum[0], local_sum[2])
    s_lo, s_hi = local_sum[0]
    return cutlass.Float32(s_lo) + cutlass.Float32(s_hi)  # pyrefly: ignore[bad-argument-type]


def fmax_reduce_packed(frg: cute.Tensor, init_val: object = None) -> Float32:
    """Max-reduce a register fragment with 4 accumulators + 3-input fmax.

    Mirrors flash_attn.cute.utils.fmax_reduce (arch>=100). Forces FMNMX3 (the
    cute ``.reduce`` lowers to ~50% 2-input / 50% 3-input + a serial fold).
    """
    n = cute.size(frg)
    assert n % 8 == 0, "fmax_reduce_packed requires a multiple-of-8 fragment"
    local_max_0 = (
        _fmax3(init_val, frg[0], frg[1])
        if init_val is not None
        else _fmax3(frg[0], frg[1])
    )
    local_max = [
        local_max_0,
        _fmax3(frg[2], frg[3]),
        _fmax3(frg[4], frg[5]),
        _fmax3(frg[6], frg[7]),
    ]
    for i in range(8, n, 8):
        local_max[0] = _fmax3(local_max[0], frg[i + 0], frg[i + 1])
        local_max[1] = _fmax3(local_max[1], frg[i + 2], frg[i + 3])
        local_max[2] = _fmax3(local_max[2], frg[i + 4], frg[i + 5])
        local_max[3] = _fmax3(local_max[3], frg[i + 6], frg[i + 7])
    local_max[0] = _fmax3(local_max[0], local_max[1])
    return _fmax3(local_max[0], local_max[2], local_max[3])
