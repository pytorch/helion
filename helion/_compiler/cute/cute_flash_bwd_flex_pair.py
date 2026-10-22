"""The 2-CTA FlexAttention backward at head dim 256 (the ``pair`` TMEM plan).

No 1-CTA plan fits head dim 256: dK and dV alone are 512 TMEM columns and
the 128-row K / V tiles with Q / dO are 256 KB of shared memory. One kernel
cannot mix ``cta_group::1`` and ``cta_group::2`` tcgen05 instructions, and a
pair MMA never splits its K dimension, so a cluster that splits the KV rows
(for dK / dV) cannot also form the dQ partial whose K dimension is those
rows. The kernel therefore runs two cluster roles per 128-row KV block --
one BlockMask KV block, so both roles share the flex step table of
:mod:`cute_flash_bwd_flex` verbatim -- the way FA4's hd256 backward splits
into a dK/dV and a dQ kernel:

* the dK/dV clusters (role 0, 64 KV rows per CTA): the leader issues
  S = K Q^T and dP = V dO^T (M=128 pair MMAs, K = 256; each CTA holds its 64
  Q rows of Q / dO as the split B operand), both CTAs' compute warpgroups
  run the softmax passes on their 64 rows and stage P (into the consumed Q
  slot) and dS as the K-major A operands of the pair's dV = P^T dO and
  dK = dS^T Q MMAs (M=128, N=256, K=128; B: each CTA's 128 head-dim columns
  of dO^T / Q^T). TMEM per CTA: dK 128 + dV 128 + S 64 + dP 64. Epilogue:
  64-row dK / dV tiles stored as (64, 64) boxes.
* the dQ clusters (role 1, 64 Q rows of the step per CTA): the leader
  issues S^T = Q K^T and dP^T = dO V^T (Q / dO as the A operands, the
  resident K / V tiles as B), the compute warpgroups recompute P and dS
  with one Q row per thread (the LSE / delta terms are per-thread scalars)
  and stage dS (64 q x 128 kv, K-major) as the A operand of dQ = dS K
  (M=128, N=256, K=128; B: each CTA's 128 head-dim columns of K^T, resident
  per tile); the reduce warps TMA-reduce-add the (2,2)-layout accumulator
  as (32 x 16) fp32 boxes. TMEM per CTA: S^T 64 + dP^T 64 + dQ 128.

The pair's M=128 MMAs pack a CTA's 64 rows as 128 lanes x N/2 columns; the
partition's row coordinates are the pair's 128 rows. Single-stage rings.
Not supported here (the config space hides them): the persistent walk,
statistics MMAs, staged aux tables, chunked step tables, the additive-bias
fold on the dQ clusters, and strided operand maps.
"""

from __future__ import annotations

import textwrap
import typing
from typing import TYPE_CHECKING

import torch

from ..device_function import TensorArg
from .cute_flash_bwd_flex import _lines
from .cute_flash_bwd_flex import _merge_blocks
from .cute_flash_bwd_flex import _reg_split
from .cute_flash_bwd_flex import _render_hooks
from .cute_flash_bwd_flex import _setmaxreg
from .cute_flash_bwd_flex import _step_table_fill
from .cute_flash_bwd_flex import _step_table_loop
from .cute_flash_bwd_flex import _tile_decode
from .cute_flash_bwd_flex import additive_bias_region
from .cute_flash_bwd_flex import flex_bwd_hook_plan
from .cute_flash_bwd_flex import flex_bwd_pair_split_p
from .cute_flash_bwd_flex import flex_bwd_pair_table_steps
from .flash_score_mod import render_flex_hook
from .flex_bwd_match import flex_bwd_tmem_plan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_function import DeviceFunction
    from ..generate_ast import GenerateAST
    from .flash_score_mod import FlexHookCode
    from .flex_bwd_match import FlexBwdMatch
    from .flex_bwd_match import FlexBwdPlan

__all__ = ["codegen_flex_bwd_pair", "emit_flash_bwd_flex_pair_device_body"]

# The wrapper-only kernel parameters, in the launcher's ``flex_pair`` order.
_FLASH_BWD_FLEX_PAIR_KERNEL_PARAMS = (
    "_fbwd_ss_mma",
    "_fbwd_pv_mma",
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
    "_fbwd_qsl",
    "_fbwd_ksl",
    "_fbwd_vsl",
    "_fbwd_dosl",
    "_fbwd_pal",
    "_fbwd_dotl",
    "_fbwd_qtl",
    "_fbwd_mLSE",
    "_fbwd_mDelta",
    "_fbwd_tma_dq",
    "_fbwd_mDQt",
    "_fbwd_dqsl",
    "_fbwd_epil",
    "_fbwd_tma_dv",
    "_fbwd_mdVt",
    "_fbwd_tma_dk",
    "_fbwd_mdKt",
    "_fbwd_mdK",
    "_fbwd_mdV",
    "_fbwd_tma_qa",
    "_fbwd_mQa",
    "_fbwd_tma_doa",
    "_fbwd_mdOa",
    "_fbwd_tma_kb",
    "_fbwd_mKb",
    "_fbwd_tma_vb",
    "_fbwd_mVb",
    "_fbwd_tma_kt2",
    "_fbwd_mKt2",
    "_fbwd_dqsl16",
    "_fbwd_dotl2",
)

# The dQ clusters' KV-side lists, derived on the host from the Q-side tables
# the kernel receives (``flex_bwd_kv_tables``): the import is emitted once per
# module, one host statement ahead of the launch binds the builder's tuple
# (cached per BlockMask) and the four tables enter as extra kernel arguments
# indexing it.
_FLEX_BWD_KV_TABLE_IMPORT = """\
from helion.runtime.cute.flash_flex import flex_bwd_kv_tables as _helion_flash_flex_bwd_kv_tables
"""
_KV_TABLE_NAMES = ("kv_part_cnt", "kv_part_idx", "kv_full_cnt", "kv_full_idx")


def _kv_table_args(
    cg: GenerateAST,
    df: DeviceFunction,
    walk_args: Sequence[TensorArg],
    *,
    q_blocks: int,
    kv_blocks: int,
) -> list[TensorArg]:
    """Bind the four derived KV-side tables (``(Bm, Hm, q_blocks)`` counts and
    ``(Bm, Hm, q_blocks, kv_blocks)`` KV block lists, partial then full).

    The builder runs once per launch: a host statement emitted here (the
    lowering runs while the grid loop is visited, so it lands ahead of the
    launch call) binds its tuple and the four arguments index it; four host
    expressions calling it would rebuild the lists four times whenever the
    cache is bypassed (CUDA-graph capture, inference tensors). The builder
    slices the Q-side tables to ``kv_blocks`` columns, so these shapes hold
    for tables wider than N needs."""
    import ast as ast_module

    from ..ast_extension import statement_from_string
    from ..compile_environment import CompileEnvironment

    if not getattr(cg, "_helion_flash_flex_bwd_kv_module_emitted", False):
        cg._helion_flash_flex_bwd_kv_module_emitted = True  # type: ignore[attr-defined]
        for stmt in ast_module.parse(_FLEX_BWD_KV_TABLE_IMPORT).body:
            cg.module_statements.append(stmt)
    assert len(walk_args) == 4
    env = CompileEnvironment.current()
    table_batch, table_heads = (int(size) for size in walk_args[0].fake_value.shape[:2])
    tables = ", ".join(arg.host_str() for arg in walk_args)
    bound = df.new_var("fbwd_kv_tables")
    cg.host_statements.append(
        statement_from_string(
            f"{bound} = _helion_flash_flex_bwd_kv_tables({tables}, "
            f"q_blocks={q_blocks}, kv_blocks={kv_blocks})"
        )
    )
    args: list[TensorArg] = []
    for index, name in enumerate(_KV_TABLE_NAMES):
        shape: tuple[int, ...] = (table_batch, table_heads, q_blocks)
        if index % 2:
            shape = (*shape, kv_blocks)
        with env.fake_mode:
            fake = torch.empty(shape, dtype=torch.int32, device=env.device)
        arg = TensorArg(df.new_var(f"fbwd_{name}"), fake, f"{bound}[{index}]")
        df.arguments.append(arg)
        df._tensor_args[fake] = arg
        df.placeholder_args.add(arg.name)
        args.append(arg)
    return args


_PIPELINE = (
    "cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={stages}, "
    "producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), "
    "consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), "
    "tx_count=fbwd_{name}_bytes, barrier_storage=storage.{name}_mbar_ptr.data_ptr(), "
    "cta_layout_vmnk=_fbwd_cluster_vmnk).make_participants()"
)
_MCAST_COMMIT = (
    "cute_tcgen05_flash.commit({ptr}, fbwd_mcast, cute_tcgen05_flash.CtaGroup.TWO)"
)
_WAIT = "_helion_flash_rt.mbar_spin_wait({ptr}, {phase}, 10000000)"


def _gemm(
    acc: str,
    a: str,
    a_frag: str,
    b: str,
    b_frag: str,
    idesc: str,
    zero: str,
) -> str:
    """One pair SS gemm over the parked operand starts (``fbwd_st_<name>``)."""
    return (
        f"_helion_flash_ptx.gemm_ptx_dyn_qk({acc}, fbwd_st_{a}, fbwd_{a}_base, {a_frag}[None, None, None, 0].layout, "
        f"fbwd_st_{b}, fbwd_{b}_base, {b_frag}[None, None, None, 0].layout, {idesc!r}, zero_init={zero}, cta_group=2)"
    )


def _render_hooks_q(
    plan: FlexBwdPlan,
    *,
    device_function: DeviceFunction,
    mask_form: str,
    hook_cols: int,
) -> tuple[FlexHookCode | None, FlexHookCode | None, FlexHookCode | None]:
    """The dQ clusters' hooks: :func:`cute_flash_bwd_flex._render_hooks` in
    the forward frame (thread row = Q row, lanes = KV columns): the Q tile
    is the renderer's ``m`` tile and the KV tile its ``kv`` tile."""
    has_score = plan.score_region_q is not None
    has_mask = plan.mask_region_q is not None
    has_grad = plan.grad_region_q is not None
    hook_plan = flex_bwd_hook_plan(plan, q_frame=True)
    common: dict[str, object] = {
        "stage": 0,
        "coord_tensor": "fbwd_cchunk",
        "m_tile_expr": "fbwd_m_tile_h",
        "kv_tile_expr": "fbwd_n_tile_h",
        "chunk_expr": "fbwd_ch2",
        "chunk_cols": hook_cols,
        "indent": 20,
        "item_indent": 8,
        "tile_indent": 12,
        "row_coord_expr": "fbwd_thr_row",
        "device_function": device_function,
        "coord_bounds": (
            plan.batch,
            plan.heads,
            plan.num_q_tiles_padded * 128,
            plan.num_kv_tiles * 128,
        ),
        "mask_form": mask_form,
        "aux_smem_budget": 0,
        "aux_smem_staged": frozenset(),
    }
    plain = masked = jac = None
    if has_score:
        plain = render_flex_hook(
            hook_plan,
            score_tensor="fbwd_s_chunk",
            with_score=True,
            with_mask=False,
            tail_mask=False,
            **common,  # pyrefly: ignore[bad-argument-type]
        )
    if has_mask:
        masked = render_flex_hook(
            hook_plan,
            score_tensor="fbwd_s_chunk",
            with_score=has_score,
            with_mask=True,
            tail_mask=False,
            **common,  # pyrefly: ignore[bad-argument-type]
        )
    if has_grad:
        jac = render_flex_hook(
            flex_bwd_hook_plan(plan, grad=True, q_frame=True),
            score_tensor="fbwd_spre_chunk",
            with_score=True,
            with_mask=False,
            tail_mask=False,
            **common,  # pyrefly: ignore[bad-argument-type]
        )
    return plain, masked, jac


def _pair_compute_block(
    *,
    h: int,
    dq_role: bool,
    plan: FlexBwdPlan,
    io_dtype: str,
    scale_expr: str,
    exp2_f32: bool,
    plain: FlexHookCode | None,
    masked: FlexHookCode | None,
    jac: FlexHookCode | None,
    hook_cols: int,
    hook_pin: bool,
    bias_fold: bool,
    exp2_emu: int,
    s_off: int,
    dp_off: int,
    dv_off: int,
    dk_off: int,
    split_p: bool,
) -> str:
    """One compute warpgroup's body (indent 8, after the TMEM retrieve): the
    CTA's 64 rows x TMEM column chunk ``h`` of S / dP (32 columns per
    thread; the row and first column a lane holds
    come from the partition's coordinate tensor), the exp2 and dS passes
    with the flex hooks, and the staging of P / dS (dK/dV clusters) or dS
    (dQ clusters). The dK/dV clusters end with the dV (h=0) / dK (h=1)
    epilogue of the CTA's 64 rows."""
    warp_lo, warp_hi = (4, 8) if h == 0 else (8, 12)
    has_score = (plan.score_region_q if dq_role else plan.score_region) is not None
    has_score = has_score and not bias_fold
    has_grad = (plan.grad_region_q if dq_role else plan.grad_region) is not None
    pre_scaled = has_score or has_grad
    p_scale = "fbwd_log2c" if pre_scaled else "fbwd_scale2"
    hooks = [hook for hook in (plain, masked, jac) if hook is not None]
    item_block = _merge_blocks(*(hook.item_prelude for hook in hooks))
    tile_block = _merge_blocks(*(hook.tile_prologue for hook in hooks))
    # The hooks' item-level values run per step when the group has several
    # Q heads (the dK/dV clusters' steps change the Q head).
    per_step_item = plan.group > 1 and not dq_role
    subs = 32 // hook_cols

    def chunk_prefix(ind: str) -> str:
        # The thread's hook chunk index within the 128-column tile, its
        # register offset and its score / coordinate sub-chunks.
        lines = [
            f"{ind}fbwd_ch2 = fbwd_c0 // {hook_cols} + fbwd_sub",
            f"{ind}fbwd_off = fbwd_sub * {hook_cols}",
            f"{ind}fbwd_s_chunk = cute.make_tensor(tLDrS.iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))",
            f"{ind}fbwd_cchunk = cute.logical_divide(tLDcS_h, {hook_cols})[None, fbwd_sub]",
        ]
        if pre_scaled:
            lines.append(
                f"{ind}_helion_flash_rt._scale_fragment_packed_f32x2(fbwd_s_chunk, fbwd_scale)"
            )
        if has_grad:
            lines.extend(
                (
                    f"{ind}fbwd_spre_chunk = cute.make_tensor(fbwd_spre.iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))",
                    f"{ind}cute.autovec_copy(fbwd_s_chunk, fbwd_spre_chunk)",
                )
            )
        return "\n".join(lines) + "\n"

    def bias_lines(ind: str) -> str:
        if not bias_fold or plain is None:
            return ""
        return (
            f"{ind}fbwd_gfrag = cute.make_rmem_tensor(cute.make_layout({hook_cols}), cutlass.Float32)\n"
            + _lines(plain.chunk_body, len(ind))
            + f"{ind}_helion_flash_rt.fbwd_lse_fold(fbwd_lse_sub, fbwd_gfrag, fbwd_neg_log2c)\n"
        )

    # The LSE term: per Q column (staged per step, dK/dV clusters) or the
    # thread's row (dQ clusters: the resident fragment of the tile setup).
    lse_load = (
        ""
        if dq_role
        else "{ind}fbwd_lse_frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)\n"
        "{ind}fbwd_lse_v = cute.make_tensor(fbwd_sLSE.iterator + fbwd_par + fbwd_c0, cute.make_layout(32))\n"
        "{ind}cute.autovec_copy(fbwd_lse_v, fbwd_lse_frg)\n"
        + (
            ""
            if plan.lse_prescaled
            else "{ind}_helion_flash_rt.fbwd_lse_to_log2(fbwd_lse_frg, fbwd_log2c)\n"
        )
    )

    def p_pass(hook: FlexHookCode | None, ind: str) -> str:
        return (
            lse_load.format(ind=ind)
            + f"{ind}for fbwd_sub in cutlass.range_constexpr({subs}):\n"
            + chunk_prefix(ind + "    ")
            + f"{ind}    fbwd_lse_sub = cute.make_tensor(fbwd_lse_frg.iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))\n"
            + bias_lines(ind + "    ")
            + (_lines(hook.chunk_body, len(ind) + 4) if hook is not None else "")
            + f"{ind}    _helion_flash_rt.fbwd_p_pairs_packed(tLDrS, fbwd_lse_sub, fbwd_off, {hook_cols}, {p_scale}, None, fbwd_off, exp2_f32={exp2_f32}, emu_res={exp2_emu})\n"
        )

    unmasked_hook = None if bias_fold else plain
    if masked is not None:
        p_passes = (
            "            if fbwd_masked != 0:\n"
            + p_pass(masked, " " * 16)
            + "            else:\n"
            + p_pass(unmasked_hook, " " * 16)
        )
    else:
        p_passes = p_pass(unmasked_hook, " " * 12)
    jac_lines = ""
    if jac is not None:
        jac_lines = (
            f"            for fbwd_sub in cutlass.range_constexpr({subs}):\n"
            f"                fbwd_ch2 = fbwd_c0 // {hook_cols} + fbwd_sub\n"
            f"                fbwd_spre_chunk = cute.make_tensor(fbwd_spre.iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))\n"
            f"                fbwd_cchunk = cute.logical_divide(tLDcS_h, {hook_cols})[None, fbwd_sub]\n"
            + _lines(jac.chunk_body, 16)
            + f"                fbwd_ds_sub = cute.make_tensor(tLDrdP.iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))\n"
            f"                _helion_flash_rt.fbwd_mul_pairs_packed(fbwd_ds_sub, fbwd_spre, fbwd_sub * {hook_cols}, {hook_cols})\n"
        )
    spre_alloc = (
        "            fbwd_spre = cute.make_rmem_tensor(tLDcS_h.shape, cutlass.Float32)\n"
        if has_grad
        else ""
    )
    pin_line = (
        "fbwd_m_tile_h = _helion_flash_rt.fbwd_pin_i32(fbwd_m_tile)"
        if hook_pin
        else "fbwd_m_tile_h = fbwd_m_tile"
    )
    item_lines = _lines(item_block, 12 if per_step_item else 8)
    tile_lines = _lines(tile_block, 12)
    item_once = "" if per_step_item else item_lines
    item_each = item_lines if per_step_item else ""
    # The dQ clusters' item-level hook lines read the tile's Q tile (pinned
    # once in the setup); the dK/dV clusters pin the step's Q tile after the
    # S wait.
    pin_late = "" if dq_role else f"            {pin_line}\n"
    # dK/dV clusters: the first warpgroup stages the step's 128 LSE / delta
    # columns (threads 0-63 the LSE, 64-127 the delta; two columns each),
    # one step ahead, in the free parity slot; both warpgroups read them
    # behind a 256-thread barrier (which also orders the other group's reads
    # of a slot before its refill two steps later).
    stat_slot = "0"

    def stage_lines(ind: str, slot: str, entry: str) -> str:
        return (
            f"{ind}fbwd_bhq{entry} = fbwd_bhk * {plan.group} + ((fbwd_e{entry} >> 16) & 255)\n"
            f"{ind}fbwd_scol0 = fbwd_bhq{entry} * {plan.q_seq} + cutlass.min((fbwd_e{entry} & 65535) * 128 + fbwd_stat_j, cutlass.Int32({plan.q_seq - 1}))\n"
            f"{ind}fbwd_scol1 = fbwd_bhq{entry} * {plan.q_seq} + cutlass.min((fbwd_e{entry} & 65535) * 128 + 64 + fbwd_stat_j, cutlass.Int32({plan.q_seq - 1}))\n"
            f"{ind}if fbwd_local_tidx < 64:\n"
            f"{ind}    _helion_flash_rt.cp_async_ca_4(fbwd_sLSE.iterator + ({slot} + fbwd_stat_j), _fbwd_mLSE.iterator + fbwd_scol0)\n"
            f"{ind}    _helion_flash_rt.cp_async_ca_4(fbwd_sLSE.iterator + ({slot} + 64 + fbwd_stat_j), _fbwd_mLSE.iterator + fbwd_scol1)\n"
            f"{ind}else:\n"
            f"{ind}    _helion_flash_rt.cp_async_ca_4(fbwd_sDelta.iterator + ({slot} + fbwd_stat_j), _fbwd_mDelta.iterator + fbwd_scol0)\n"
            f"{ind}    _helion_flash_rt.cp_async_ca_4(fbwd_sDelta.iterator + ({slot} + 64 + fbwd_stat_j), _fbwd_mDelta.iterator + fbwd_scol1)\n"
            f"{ind}cute.arch.cp_async_commit_group()\n"
        )

    if dq_role:
        stage_first = ""
        stage_next = ""
        stat_load = ""
        stat_wait = ""
        delta_lines = ""
    elif h == 0:
        stage_first = (
            "        if fbwd_steps > 0:\n"
            "            fbwd_e0 = cutlass.Int32(fbwd_sSteps[0])\n"
            + stage_lines(" " * 12, stat_slot, "0")
        )
        stage_next = (
            "            if fbwd_i + 1 < fbwd_steps:\n"
            "                fbwd_e1 = cutlass.Int32(fbwd_sSteps[fbwd_i + 1])\n"
            + stage_lines(" " * 16, f"128 - fbwd_par + {stat_slot}", "1")
        )
        stat_load = ""
        stat_wait = (
            "            cute.arch.cp_async_wait_group(0)\n"
            "            _helion_flash_rt.named_barrier_wait_unaligned(3, 256)\n"
        )
    else:
        stage_first = ""
        stage_next = ""
        stat_load = ""
        stat_wait = (
            "            _helion_flash_rt.named_barrier_wait_unaligned(3, 256)\n"
        )
    delta_lines = ""
    if not dq_role:
        delta_lines = (
            "            fbwd_dlt_frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)\n"
            "            fbwd_dlt_v = cute.make_tensor(fbwd_sDelta.iterator + fbwd_par + fbwd_c0, cute.make_layout(32))\n"
            "            cute.autovec_copy(fbwd_dlt_v, fbwd_dlt_frg)\n"
        )
    # S / dP come in two TMEM buffers by step parity (``fbwd_b``), each with
    # its own barriers (one phase per two steps): the leader issues the next
    # step's S / dP while the passes of this one run, and the warpgroups
    # hand a buffer back once dP left TMEM (``stq_empty``). The staging
    # buffer the dS pass overwrites: the dK/dV clusters' dS (read by the
    # pair's dK MMA: ``sds_free``), the dQ clusters' dS[b] (read by dQ(i-2):
    # ``dqq_full[b]``).
    buffers = (
        "            fbwd_b = fbwd_i % 2\n"
        "            fbwd_pb = (fbwd_i >> 1) & 1\n"
        f"            tSb_t = cute.make_tensor(fbwd_tmem_ptr + {s_off} + 128 * fbwd_b, tStS_frag.layout)\n"
        f"            tdPb_t = cute.make_tensor(fbwd_tmem_ptr + {dp_off} + 128 * fbwd_b, tStS_frag.layout)\n"
        f"            tLDtS_b = fbwd_thr_ld.partition_S(tSb_t)[None, {h}, 0, 0]\n"
        f"            tLDtdP_b = fbwd_thr_ld.partition_S(tdPb_t)[None, {h}, 0, 0]\n"
    )
    consumed = (
        "            cute.arch.sync_warp()\n"
        "            with cute.arch.elect_one():\n"
        "                _helion_flash_rt.mbarrier_arrive(fbwd_stq_empty_ptr + fbwd_b, cutlass.Int32(0))\n"
    )
    if dq_role:
        ds_store = (
            "            if fbwd_i >= 1:\n"
            f"                {_WAIT.format(ptr='fbwd_dqq_full_ptr', phase='(fbwd_i - 1) & 1')}\n"
            "            cute.autovec_copy(fbwd_ds_flat, fbwd_sdS_v[fbwd_crow, None, fbwd_cc])\n"
        )
        ds_full = "fbwd_dsq_full_ptr"
    elif split_p:
        # dS has its own buffer: its write waits only for the previous
        # step's dK MMA (which read dS(i-1) from it).
        ds_store = (
            "            if fbwd_i > 0:\n"
            f"                {_WAIT.format(ptr='fbwd_sdsfree_ptr', phase='fbwd_sdsf_phase')}\n"
            "                fbwd_sdsf_phase ^= 1\n"
            "            cute.autovec_copy(fbwd_ds_flat, fbwd_sdS_v[fbwd_crow, None, fbwd_cc])\n"
        )
        ds_full = "fbwd_ds_full_ptr"
    else:
        ds_store = (
            f"            {_WAIT.format(ptr='fbwd_pv_done_ptr', phase='fbwd_i & 1')}\n"
            "            cute.autovec_copy(fbwd_ds_flat, fbwd_sdS_v[fbwd_crow, None, fbwd_cc])\n"
        )
        ds_full = "fbwd_ds_full_ptr"
    # The dK/dV clusters stage P (bf16) as the dV MMA's A operand right after
    # computing it, before waiting for dP (the dV MMA then runs while dS is
    # computed). With ``split_p`` the P buffer is free once the previous
    # step's dV MMA read it; otherwise P shares the dS buffer, free once the
    # previous step's dK MMA read dS, and the dS write of this step waits
    # for the dV MMA.
    if dq_role:
        p_stage = ""
    else:
        p_wait = (
            f"                {_WAIT.format(ptr='fbwd_pv_done_ptr', phase='(fbwd_i - 1) & 1')}\n"
            if split_p
            else f"                {_WAIT.format(ptr='fbwd_sdsfree_ptr', phase='fbwd_sdsf_phase')}\n"
            "                fbwd_sdsf_phase ^= 1\n"
        )
        p_stage = (
            f"            fbwd_p_bf = cute.make_rmem_tensor(tLDrS.layout, {io_dtype})\n"
            f"            fbwd_p_bf.store(tLDrS.load().to({io_dtype}))\n"
            "            fbwd_p_flat = cute.make_tensor(fbwd_p_bf.iterator, cute.make_layout(32))\n"
            "            if fbwd_i > 0:\n"
            + p_wait
            + "            cute.autovec_copy(fbwd_p_flat, fbwd_sP_v[fbwd_crow, None, fbwd_cc])\n"
            "            cute.arch.fence_view_async_shared()\n"
            "            cute.arch.sync_warp()\n"
            "            with cute.arch.elect_one():\n"
            "                _helion_flash_rt.mbarrier_arrive(fbwd_p_full_ptr, cutlass.Int32(0))\n"
        )
    epilogue = ""
    zero_store = "        pass\n"
    if not dq_role:
        epi_off = dv_off if h == 0 else dk_off
        epi_scale = (
            "_helion_flash_rt._scale_fragment_packed_f32x2(fbwd_reg, cutlass.Float32("
            + scale_expr
            + "))"
            if h == 1
            else "pass"
        )
        epi_buf = "sV" if h == 0 else "sK"
        epi_tma = "_fbwd_tma_dv" if h == 0 else "_fbwd_tma_dk"
        epi_gm = "_fbwd_mdVt" if h == 0 else "_fbwd_mdKt"
        # dV waits for the pair's last dV MMA (the leader's multicast
        # commit); dK for the dS staging's last release, committed after
        # the pair's last dK MMA.
        epi_wait = (
            "        _helion_flash_rt.mbar_spin_wait(fbwd_dv_done_ptr, 0, 10000000)\n"
            if h == 0
            else "        _helion_flash_rt.mbar_spin_wait(fbwd_sdsfree_ptr, fbwd_sdsf_phase, 10000000)\n"
        )
        chunks = "".join(
            (
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
        cute.autovec_copy(fbwd_row64, fbwd_sEpi[fbwd_erow, None, fbwd_eslot0 + {c // 4}])"""
                if c % 4 == 3
                else ""
            )
            for c in range(8)
        )
        epilogue = f"""{epi_wait}        fbwd_dkv_shape = fbwd_pvt.partition_shape_C((128, 256))
        tDKV_frag = fbwd_pvt.make_fragment_C(fbwd_dkv_shape)
        tEPI_t = cute.make_tensor(fbwd_tmem_ptr + {epi_off}, tDKV_frag.layout)
        fbwd_epi_tiler = ((cute.size(tEPI_t, mode=[0, 0]), cute.size(tEPI_t, mode=[0, 1])),)
        tEPI_epi = cute.zipped_divide(tEPI_t, fbwd_epi_tiler)
        fbwd_epi_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(16)), cutlass.Float32)
        fbwd_tiled_epi_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_epi_ld_atom, tEPI_epi[None, 0])
        fbwd_thr_epi_ld = fbwd_tiled_epi_ld.get_slice(fbwd_local_tidx)
        tEPItT = fbwd_thr_epi_ld.partition_S(tEPI_epi)
        fbwd_cepi = cute.make_identity_tensor((128, 256))
        fbwd_c_epi = cute.zipped_divide(fbwd_pvt.partition_C(fbwd_cepi), fbwd_epi_tiler)
        tEPIcC = fbwd_thr_epi_ld.partition_D(fbwd_c_epi)
        # This thread's row (within the CTA's 64) and first head-dim column
        # (its lane half's 128 columns = two (64, 64) boxes); the tile is
        # staged through the free {epi_buf} buffer as four boxes and stored
        # by the group's first warp.
        fbwd_erow = cutlass.Int32(tEPIcC[0][0]) % 64
        fbwd_eslot0 = cutlass.Int32(tEPIcC[0][1]) // 64
        fbwd_sEpi = storage.{epi_buf}.get_tensor(_fbwd_epil.outer, swizzle=_fbwd_epil.inner)
        fbwd_gEpi = cute.local_tile({epi_gm}[None, None, fbwd_bhk], (64, 64), (None, None)){chunks}
        cute.arch.fence_view_async_shared()
        _helion_flash_rt.named_barrier_wait_unaligned({5 + h}, 128)
        if warp_idx == {warp_lo}:
            for fbwd_j in cutlass.range_constexpr(4):
                tEPIsS, tEPIgS = cute_cpasync_flash.tma_partition({epi_tma}, 0, cute.make_layout(1), cute.group_modes(fbwd_sEpi[None, None, fbwd_j], 0, 2), cute.group_modes(fbwd_gEpi[None, None, fbwd_n_tile * 2 + fbwd_rank, fbwd_j], 0, 2))
                cute.copy({epi_tma}, tEPIsS, tEPIgS)
                cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
"""
        zero_target = "_fbwd_mdV" if h == 0 else "_fbwd_mdK"
        zero_store = f"""        if fbwd_local_tidx < 64:
            fbwd_zrow = fbwd_kv_base + fbwd_local_tidx
            if fbwd_zrow < {plan.kv_seq}:
                fbwd_zeros = cute.make_rmem_tensor(cute.make_layout(256), {io_dtype})
                fbwd_zeros.fill({io_dtype}(0.0))
                fbwd_zdst = cute.make_tensor({zero_target}[None, None, fbwd_bhk][fbwd_zrow, None].iterator, cute.make_layout(256))
                cute.autovec_copy(fbwd_zeros, fbwd_zdst)
"""
    p_view = (
        ""
        if dq_role
        else "        fbwd_sP_v = cute.make_tensor(fbwd_sP.iterator, cute.composition(fbwd_sP.layout, cute.make_layout((64, 32, 4))))\n"
    )
    # The dQ clusters' resident per-row terms: the thread's Q row of the
    # tile (clamped into the sequence: the TMA zero-filled the rows past
    # it) selects its LSE (converted once) and delta.
    ds1_view = (
        f"""        fbwd_qrow_t = cutlass.min(fbwd_m_tile_t * 128 + fbwd_thr_row, cutlass.Int32({plan.q_seq - 1}))
        fbwd_lse_frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
        fbwd_lse_frg.fill(_fbwd_mLSE[fbwd_bhq_t * {plan.q_seq} + fbwd_qrow_t])
        fbwd_dlt_frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
        fbwd_dlt_frg.fill(_fbwd_mDelta[fbwd_bhq_t * {plan.q_seq} + fbwd_qrow_t])
"""
        + (
            ""
            if plan.lse_prescaled
            else "        _helion_flash_rt.fbwd_lse_to_log2(fbwd_lse_frg, fbwd_log2c)\n"
        )
        if dq_role
        else ""
    )
    setup = f"""        fbwd_s_shape = fbwd_sst.partition_shape_C((128, 128))
        tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
        # This CTA's S / dP (64 rows x 128 columns in the pair's 128-lane x
        # 64-column layout); the group takes column chunk {h} of the
        # partition: one 32-column chunk per thread.
        tS_t = cute.make_tensor(fbwd_tmem_ptr + {s_off}, tStS_frag.layout)
        tdP_t = cute.make_tensor(fbwd_tmem_ptr + {dp_off}, tStS_frag.layout)
        cS = cute.make_identity_tensor((128, 128))
        tScS = fbwd_sst.partition_C(cS)
        fbwd_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
        fbwd_tiled_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_ld_atom, tS_t)
        fbwd_thr_ld = fbwd_tiled_ld.get_slice(fbwd_local_tidx)
        tLDtS = fbwd_thr_ld.partition_S(tS_t)
        tLDtdP = fbwd_thr_ld.partition_S(tdP_t)
        tLDcS = fbwd_thr_ld.partition_D(tScS)
        tLDtS_h = tLDtS[None, {h}, 0, 0]
        tLDtdP_h = tLDtdP[None, {h}, 0, 0]
        tLDcS_h = tLDcS[None, {h}, 0, 0]
        # The thread's row (the partition's rows are the pair's 128; the
        # CTA's 64 start at its rank) and first column of the 128-wide tile.
        fbwd_crow = cutlass.Int32(tLDcS_h[0][0]) % 64
        fbwd_c0 = cutlass.Int32(tLDcS_h[0][1])
        fbwd_cc = fbwd_c0 // 32
        fbwd_thr_row = fbwd_rank * 64 + fbwd_crow
        # The staged tiles are (64 rows x 128 columns) K-major: this thread's
        # 32 columns of its row.
{p_view}        fbwd_sdS_v = cute.make_tensor(fbwd_sdS_nk.iterator, cute.composition(fbwd_sdS_nk.layout, cute.make_layout((64, 32, 4))))
{ds1_view}        fbwd_scale = cutlass.Float32({scale_expr})
        fbwd_log2c = cutlass.Float32({plan.log2_factor!r})
        fbwd_neg_log2c = cutlass.Float32({-plan.log2_factor!r})
        fbwd_scale2 = cutlass.Float32(({scale_expr}) * {plan.log2_factor!r})
""" + (
        ""
        if dq_role
        # dK/dV clusters: the dS staging's sds_free phase and, for the
        # staging warpgroup, this thread's LSE / delta staging column.
        else "        fbwd_sdsf_phase = cutlass.Int32(0)\n"
        + ("        fbwd_stat_j = fbwd_local_tidx % 64\n" if h == 0 else "")
    )
    if dq_role:
        step_head = f"""            fbwd_n_tile_s = fbwd_e & 65535
            fbwd_masked = fbwd_e >> 24
            fbwd_n_tile_h = {"_helion_flash_rt.fbwd_pin_i32(fbwd_n_tile_s)" if hook_pin else "fbwd_n_tile_s"}
"""
    else:
        step_head = f"""            fbwd_m_tile = fbwd_e & 65535
            fbwd_gi = (fbwd_e >> 16) & 255
            fbwd_masked = fbwd_e >> 24
            fbwd_bhq = fbwd_bhk * {plan.group} + fbwd_gi
            flash_bh = fbwd_bhq
"""
    steps_region = f"""{stage_first}        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
            fbwd_e = cutlass.Int32(fbwd_sSteps[fbwd_i])
{step_head}{item_each}            fbwd_par = (fbwd_i % 2) * 128
{stat_load}{buffers}            {_WAIT.format(ptr="fbwd_stq_full_ptr + fbwd_b", phase="fbwd_pb")}
            tLDrS = cute.make_rmem_tensor(tLDcS_h.shape, cutlass.Float32)
{spre_alloc}            cute.copy(fbwd_tiled_ld, tLDtS_b, tLDrS)
            cute.arch.fence_view_async_tmem_load()
{stat_wait}{pin_late}{tile_lines}{stage_next}{p_passes}{p_stage}            {_WAIT.format(ptr="fbwd_dptq_full_ptr + fbwd_b", phase="fbwd_pb")}
            tLDrdP = cute.make_rmem_tensor(tLDcS_h.shape, cutlass.Float32)
            cute.copy(fbwd_tiled_ld, tLDtdP_b, tLDrdP)
{delta_lines}            cute.arch.fence_view_async_tmem_load()
{consumed}            _helion_flash_rt.fbwd_ds_pairs_packed(tLDrdP, tLDrS, 0, fbwd_dlt_frg, 32)
{jac_lines}            fbwd_ds_bf = cute.make_rmem_tensor(tLDrdP.layout, {io_dtype})
            fbwd_ds_bf.store(tLDrdP.load().to({io_dtype}))
            fbwd_ds_flat = cute.make_tensor(fbwd_ds_bf.iterator, cute.make_layout(32))
{ds_store}            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                _helion_flash_rt.mbarrier_arrive({ds_full}, cutlass.Int32(0))
"""
    if dq_role:
        tail = ""
    else:
        tail = f"""        if fbwd_steps > 0:
{textwrap.indent(epilogue.rstrip(), "    ")}
        else:
{textwrap.indent(zero_store, "    ")}"""
    tile_head = (
        f"        flash_bh = fbwd_bhq_t\n        fbwd_m_tile_h = {'_helion_flash_rt.fbwd_pin_i32(fbwd_m_tile_t)' if hook_pin else 'fbwd_m_tile_t'}\n"
        if dq_role
        else f"        flash_bh = fbwd_bhk * {plan.group}\n"
    )
    tile_region = f"""{tile_head}{item_once}{steps_region}{tail}"""
    return setup + tile_region


class _RolePieces(typing.NamedTuple):
    """One role's warp bodies at indent 8: the TMA warp's step loop, the MMA
    warp's region, the reduce warps' region and the two compute
    warpgroups' bodies."""

    tma: str
    mma: str
    reduce: str
    compute: tuple[str, str]


def _reloads(names: tuple[str, ...], ind: str, slots: tuple[int, ...] = ()) -> str:
    """Re-read the parked operand starts (``mma_base`` slots ``slots``, by
    default the names' positions) with volatile loads."""
    slots = slots or tuple(range(len(names)))
    return "".join(
        f"{ind}fbwd_st_{name} = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + {slot})\n"
        for slot, name in zip(slots, names, strict=True)
    )


def _dkdv_role(
    *,
    plan: FlexBwdPlan,
    io_dtype: str,
    tmem_s: int,
    tmem_dp: int,
    tmem_dv: int,
    tmem_dk: int,
    compute_blocks: tuple[str, str],
    step_decode: str,
    split_p: bool,
) -> _RolePieces:
    """Role 0: the dK/dV clusters' warp pieces (after the common prologue).

    The TMA warp loads a step's Q^T one iteration late (after the next
    step's Q and dO^T): its single slot frees only when dK of the step
    before completes, at the end of that step, and acquiring it in order
    held back the next step's Q so S(i+1) waited on the load every step."""
    qt_load = (
        "cute.copy(_fbwd_tma_qt, tQTgQT_qdl[None, 0, None, fbwd_bhq_qt][None, fbwd_m_tile_qt], "
        "tQTsQT[None, fbwd_qte.index], tma_bar_ptr=fbwd_qte.barrier)"
    )
    tma_steps = f"""        if fbwd_steps > 0:
            fbwd_m_tile_qt = cutlass.Int32(0)
            fbwd_bhq_qt = cutlass.Int32(0)
            for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
{textwrap.indent(step_decode, "    ")}
                tQgQ = tQgQ_qdl[None, None, 0, fbwd_bhq]
                tDOTgDOT = tDOTgDOT_qdl[None, None, 0, fbwd_bhq]
                tDOgDO = tDOgDO_qdl[None, 0, None, fbwd_bhq]
                fbwd_qe = fbwd_q_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_q, tQgQ[None, fbwd_m_tile], tQsQ[None, fbwd_qe.index], tma_bar_ptr=fbwd_qe.barrier)
                fbwd_dote = fbwd_dot_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_dot, tDOTgDOT[None, fbwd_m_tile], tDOTsDOT[None, fbwd_dote.index], tma_bar_ptr=fbwd_dote.barrier)
                if fbwd_i > 0:
                    fbwd_qte = fbwd_qt_prod.acquire_and_advance()
                    {qt_load}
                fbwd_m_tile_qt = fbwd_m_tile
                fbwd_bhq_qt = fbwd_bhq
                fbwd_dhe = fbwd_do_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_do2, tDOgDO[None, fbwd_m_tile], tDOsDO[None, fbwd_dhe.index], tma_bar_ptr=fbwd_dhe.barrier)
            fbwd_qte = fbwd_qt_prod.acquire_and_advance()
            {qt_load}
            fbwd_q_prod.tail()
            fbwd_dot_prod.tail()
            fbwd_do_prod.tail()
            fbwd_qt_prod.tail()
            fbwd_k_prod.tail()
            fbwd_v_prod.tail()
"""
    s_gemm0 = _gemm_dyn(
        "fbwd_s_addr",
        "fbwd_st_k",
        "fbwd_k_base",
        "tSrK",
        "fbwd_st_q",
        "fbwd_q_base",
        "tSrQ",
        "helion_fbwd_ss_idesc",
    )
    s_gemm = _gemm_dyn(
        "fbwd_s_addr + 128 * fbwd_bn",
        "fbwd_st_k",
        "fbwd_k_base",
        "tSrK",
        "fbwd_st_q",
        "fbwd_q_base",
        "tSrQ",
        "helion_fbwd_ss_idesc",
    )
    dp_gemm = _gemm_dyn(
        "fbwd_dp_addr + 128 * fbwd_b",
        "fbwd_st_v",
        "fbwd_v_base",
        "tSrV",
        "fbwd_st_dot",
        "fbwd_dot_base",
        "tSrDOt",
        "helion_fbwd_ss_idesc",
    )
    dv_gemm = _gemm(
        "fbwd_dv_addr",
        "p",
        "tPVrP",
        "do",
        "tPVrDO",
        "helion_fbwd_pv_idesc",
        "fbwd_dv_zero",
    )
    dk_gemm = _gemm(
        "fbwd_dk_addr",
        "ds",
        "tPVrDS",
        "qt",
        "tPVrQt",
        "helion_fbwd_pv_idesc",
        "fbwd_dk_zero",
    )
    # Step i: dP(i) into buffer b (dO released at once), S(i+1) into the
    # other buffer as soon as step i-1 read it and Q(i+1) landed (Q is
    # released right after its S), then dV(i) once P(i) sits in the dS buffer
    # (``pv_done`` lets the warpgroups overwrite it with dS) and dK(i) once
    # dS(i) is staged.
    leader_step = f"""                fbwd_b = fbwd_i % 2
                fbwd_bn = fbwd_b ^ 1
                fbwd_dote = fbwd_dot_cons.wait_and_advance()
                {dp_gemm}
                with cute.arch.elect_one():
                    {_MCAST_COMMIT.format(ptr="fbwd_dptq_full_ptr + fbwd_b")}
                fbwd_dote.release()
                if fbwd_i + 1 < fbwd_steps:
                    if fbwd_i >= 1:
                        fbwd_pn = ((fbwd_i - 1) >> 1) & 1
                        {_WAIT.format(ptr="fbwd_stq_empty_ptr + fbwd_bn", phase="fbwd_pn")}
                    fbwd_qe = fbwd_q_cons.wait_and_advance()
                    {s_gemm}
                    with cute.arch.elect_one():
                        {_MCAST_COMMIT.format(ptr="fbwd_stq_full_ptr + fbwd_bn")}
                    fbwd_qe.release()
                {_WAIT.format(ptr="fbwd_p_full_ptr", phase="fbwd_pf_phase")}
                fbwd_pf_phase ^= 1
                fbwd_dhe = fbwd_do_cons.wait_and_advance()
                {dv_gemm}
                fbwd_dv_zero = cutlass.Boolean(False)
                with cute.arch.elect_one():
                    {_MCAST_COMMIT.format(ptr="fbwd_pv_done_ptr")}
                fbwd_dhe.release()
                {_WAIT.format(ptr="fbwd_ds_full_ptr", phase="fbwd_dsf_phase")}
                fbwd_dsf_phase ^= 1
                fbwd_qte = fbwd_qt_cons.wait_and_advance()
                {dk_gemm}
                fbwd_dk_zero = cutlass.Boolean(False)
                with cute.arch.elect_one():
                    {_MCAST_COMMIT.format(ptr="fbwd_sdsfree_ptr")}
                fbwd_qte.release()
"""
    bases = ("sK", "sV", "sQ", "sdOt", "sdO", "sQt", "fbwd_sdS_nk") + (
        ("fbwd_sP",) if split_p else ()
    )
    # The dV A operand: the staged P buffer (slot 7) or the dS buffer.
    p_start = (
        "_helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 7)"
        if split_p
        else "fbwd_st_ds"
    )
    base_lines = "".join(
        f"            fbwd_mb[{i}] = _helion_flash_ptx.make_smem_desc_start_addr({tensor}[None, None, None, 0].iterator)\n"
        for i, tensor in enumerate(bases)
    )
    mma_region = f"""        if fbwd_steps > 0:
            tS_t = cute.make_tensor(fbwd_tmem_ptr + {tmem_s}, tStS_frag.layout)
            tdP_t = cute.make_tensor(fbwd_tmem_ptr + {tmem_dp}, tStS_frag.layout)
            tDV_t = cute.make_tensor(fbwd_tmem_ptr + {tmem_dv}, tDKV_frag.layout)
            tDK_t = cute.make_tensor(fbwd_tmem_ptr + {tmem_dk}, tDKV_frag.layout)
            tSrK = fbwd_sst.make_fragment_A(sK)
            tSrQ = fbwd_sst.make_fragment_B(sQ)
            tSrV = fbwd_sst.make_fragment_A(sV)
            tSrDOt = fbwd_sst.make_fragment_B(sdOt)
            tPVrP = fbwd_pvt.make_fragment_A(fbwd_sP)
            tPVrDO = fbwd_pvt.make_fragment_B(sdO)
            tPVrDS = fbwd_pvt.make_fragment_A(fbwd_sdS_nk)
            tPVrQt = fbwd_pvt.make_fragment_B(sQt)
            fbwd_s_addr = tS_t.iterator.toint()
            fbwd_dp_addr = tdP_t.iterator.toint()
            fbwd_dv_addr = tDV_t.iterator.toint()
            fbwd_dk_addr = tDK_t.iterator.toint()
            fbwd_k_base = _helion_flash_ptx.smem_desc_base_from_tensor(sK, _helion_flash_ptx.Major.K)
            fbwd_v_base = _helion_flash_ptx.smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.K)
            fbwd_q_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQ, _helion_flash_ptx.Major.K)
            fbwd_dot_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdOt, _helion_flash_ptx.Major.K)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_ss_mma.op, "helion_fbwd_ss_idesc")
            fbwd_p_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sP, _helion_flash_ptx.Major.K)
            fbwd_do_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdO, _helion_flash_ptx.Major.MN)
            fbwd_ds_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sdS_nk, _helion_flash_ptx.Major.K)
            fbwd_qt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQt, _helion_flash_ptx.Major.MN)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_pv_mma.op, "helion_fbwd_pv_idesc")
            fbwd_mb = storage.mma_base.get_tensor(cute.make_layout(8))
{base_lines}            fbwd_dv_zero = cutlass.Boolean(True)
            fbwd_dk_zero = cutlass.Boolean(True)
            fbwd_pf_phase = cutlass.Int32(0)
            fbwd_dsf_phase = cutlass.Int32(0)
            if fbwd_is_leader:
                fbwd_ke = fbwd_k_cons.wait_and_advance()
                fbwd_ve = fbwd_v_cons.wait_and_advance()
{_reloads(("k", "q"), " " * 16, (0, 2))}                fbwd_qe = fbwd_q_cons.wait_and_advance()
                {s_gemm0}
                with cute.arch.elect_one():
                    {_MCAST_COMMIT.format(ptr="fbwd_stq_full_ptr")}
                fbwd_qe.release()
                for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
{_reloads(("k", "v", "q", "dot", "do", "qt", "ds"), " " * 20)}                    fbwd_st_p = {p_start}
{textwrap.indent(leader_step, "    ")}                with cute.arch.elect_one():
                    {_MCAST_COMMIT.format(ptr="fbwd_dv_done_ptr")}
                fbwd_ke.release()
                fbwd_ve.release()
"""
    mma = f"""        fbwd_s_shape = fbwd_sst.partition_shape_C((128, 128))
        tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
        fbwd_dkv_shape = fbwd_pvt.partition_shape_C((128, 256))
        tDKV_frag = fbwd_pvt.make_fragment_C(fbwd_dkv_shape)
{mma_region}"""
    return _RolePieces(tma_steps, mma, "        pass\n", compute_blocks)


def _gemm_dyn(
    acc: str,
    a_start: str,
    a_base: str,
    a_frag: str,
    b_start: str,
    b_base: str,
    b_frag: str,
    idesc: str,
    zero: str = "True",
) -> str:
    """One pair SS gemm with explicit accumulator / operand start expressions."""
    return (
        f"_helion_flash_ptx.gemm_ptx_dyn_qk({acc}, {a_start}, {a_base}, {a_frag}[None, None, None, 0].layout, "
        f"{b_start}, {b_base}, {b_frag}[None, None, None, 0].layout, {idesc!r}, zero_init={zero}, cta_group=2)"
    )


def _dq_role(
    *,
    plan: FlexBwdPlan,
    io_dtype: str,
    scale_expr: str,
    tmem_s: int,
    tmem_dp: int,
    tmem_dq: int,
    compute_blocks: tuple[str, str],
) -> _RolePieces:
    """Role 1: the dQ clusters' warp pieces (after the common prologue).

    Q-major: the cluster's Q block (Q / dO resident, 64 rows per CTA) walks
    its KV blocks; per step the leader issues S^T(i) / dP^T(i) into the
    step-parity TMEM buffers (``tmem_s + 128 b`` / ``tmem_dp + 128 b``, one
    mbarrier each plus a consumed barrier so no producer runs two phases
    ahead of a waiter) as soon as the step's K / V landed, then dQ(i-1) =
    dS(i-1) K^T(i-1) accumulating into the single ``tmem_dq`` accumulator
    once dS(i-1) is staged (``dsq_full``) and its K^T stage arrived; the
    reduce warps drain the accumulator once per tile (``dv_done`` reused as
    the tile's dQ-complete barrier) through the then idle K / V tiles as
    (32 x 16) fp32 reduce-add boxes."""
    tma_steps = """        if fbwd_steps > 0:
            for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
                fbwd_e = cutlass.Int32(fbwd_sSteps[fbwd_i])
                fbwd_n_tile_s = fbwd_e & 65535
                fbwd_ke = fbwd_k_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_kb, tKBgKB_kdl[None, None, 0, fbwd_bhk_q][None, fbwd_n_tile_s], tKBsKB[None, fbwd_ke.index], tma_bar_ptr=fbwd_ke.barrier)
                fbwd_ve = fbwd_v_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_vb, tVBgVB_kdl[None, None, 0, fbwd_bhk_q][None, fbwd_n_tile_s], tVBsVB[None, fbwd_ve.index], tma_bar_ptr=fbwd_ve.barrier)
                fbwd_kte = fbwd_kt2_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_kt2, tKT2gKT2_kdl[None, 0, None, fbwd_bhk_q][None, fbwd_n_tile_s], tKT2sKT2[None, fbwd_kte.index], tma_bar_ptr=fbwd_kte.barrier)
            fbwd_k_prod.tail()
            fbwd_v_prod.tail()
            fbwd_kt2_prod.tail()
            fbwd_q_prod.tail()
            fbwd_dot_prod.tail()
"""
    st_gemm = _gemm_dyn(
        "fbwd_s_addr + 128 * fbwd_b",
        "fbwd_st_qa",
        "fbwd_qa_base",
        "tSrQa",
        "fbwd_st_kb",
        "fbwd_kb_base",
        "tSrKb",
        "helion_fbwdq_ss_idesc",
    )
    dpt_gemm = _gemm_dyn(
        "fbwd_dp_addr + 128 * fbwd_b",
        "fbwd_st_doa",
        "fbwd_doa_base",
        "tSrdOa",
        "fbwd_st_vb",
        "fbwd_vb_base",
        "tSrVb",
        "helion_fbwdq_ss_idesc",
    )
    dq_gemm = _gemm_dyn(
        "fbwd_dq_addr",
        "fbwd_st_ds",
        "fbwd_ds_base",
        "tPVrDS",
        "fbwd_st_kt2 + 2048 * fbwd_kte.index",
        "fbwd_kt2_base",
        "tPVrKt2",
        "helion_fbwdq_pv_idesc",
        zero="fbwd_dq_zero",
    )
    # dQ(j): dS(j) staged by both CTAs, K^T(j) from its ring stage; the
    # commit releases the dS staging to the compute warps.
    dq_block = f"""                {_WAIT.format(ptr="fbwd_dsq_full_ptr", phase="fbwd_j & 1")}
                fbwd_kte = fbwd_kt2_cons.wait_and_advance()
                {dq_gemm}
                fbwd_dq_zero = cutlass.Boolean(False)
                with cute.arch.elect_one():
                    {_MCAST_COMMIT.format(ptr="fbwd_dqq_full_ptr")}
                fbwd_kte.release()
"""
    leader_step = f"""                fbwd_b = fbwd_i % 2
                fbwd_pb = (fbwd_i >> 1) & 1
                if fbwd_i >= 2:
                    {_WAIT.format(ptr="fbwd_stq_empty_ptr + fbwd_b", phase="fbwd_pb ^ 1")}
                fbwd_ke = fbwd_k_cons.wait_and_advance()
                {st_gemm}
                with cute.arch.elect_one():
                    {_MCAST_COMMIT.format(ptr="fbwd_stq_full_ptr + fbwd_b")}
                fbwd_ke.release()
                fbwd_ve = fbwd_v_cons.wait_and_advance()
                {dpt_gemm}
                with cute.arch.elect_one():
                    {_MCAST_COMMIT.format(ptr="fbwd_dptq_full_ptr + fbwd_b")}
                fbwd_ve.release()
                if fbwd_i >= 1:
                    fbwd_j = fbwd_i - 1
{textwrap.indent(dq_block, "    ")}"""
    bases = ("sQa", "sKb", "sdOa", "sVb", "fbwd_sdS_nk", "sKt2")
    base_lines = "".join(
        f"            fbwd_mb[{i}] = _helion_flash_ptx.make_smem_desc_start_addr({tensor}[None, None, None, 0].iterator)\n"
        for i, tensor in enumerate(bases)
    )
    mma_region = f"""        if fbwd_steps > 0:
            tS_t = cute.make_tensor(fbwd_tmem_ptr + {tmem_s}, tStS_frag.layout)
            tdP_t = cute.make_tensor(fbwd_tmem_ptr + {tmem_dp}, tStS_frag.layout)
            tDQ_t = cute.make_tensor(fbwd_tmem_ptr + {tmem_dq}, tDKV_frag.layout)
            tSrQa = fbwd_sst.make_fragment_A(sQa)
            tSrKb = fbwd_sst.make_fragment_B(sKb)
            tSrdOa = fbwd_sst.make_fragment_A(sdOa)
            tSrVb = fbwd_sst.make_fragment_B(sVb)
            tPVrDS = fbwd_pvt.make_fragment_A(fbwd_sdS_nk)
            tPVrKt2 = fbwd_pvt.make_fragment_B(sKt2)
            fbwd_s_addr = tS_t.iterator.toint()
            fbwd_dp_addr = tdP_t.iterator.toint()
            fbwd_dq_addr = tDQ_t.iterator.toint()
            fbwd_qa_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQa, _helion_flash_ptx.Major.K)
            fbwd_kb_base = _helion_flash_ptx.smem_desc_base_from_tensor(sKb, _helion_flash_ptx.Major.K)
            fbwd_doa_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdOa, _helion_flash_ptx.Major.K)
            fbwd_vb_base = _helion_flash_ptx.smem_desc_base_from_tensor(sVb, _helion_flash_ptx.Major.K)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_ss_mma.op, "helion_fbwdq_ss_idesc")
            fbwd_ds_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sdS_nk, _helion_flash_ptx.Major.K)
            fbwd_kt2_base = _helion_flash_ptx.smem_desc_base_from_tensor(sKt2, _helion_flash_ptx.Major.MN)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_pv_mma.op, "helion_fbwdq_pv_idesc")
            fbwd_mb = storage.mma_base.get_tensor(cute.make_layout(8))
{base_lines}            if fbwd_is_leader:
                fbwd_qe = fbwd_q_cons.wait_and_advance()
                fbwd_dote = fbwd_dot_cons.wait_and_advance()
                fbwd_dq_zero = cutlass.Boolean(True)
                for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
{_reloads(("qa", "kb", "doa", "vb", "ds", "kt2"), " " * 20)}{textwrap.indent(leader_step, "    ")}                fbwd_j = fbwd_steps - 1
{_reloads(("ds", "kt2"), " " * 16, (4, 5))}{dq_block}                with cute.arch.elect_one():
                    {_MCAST_COMMIT.format(ptr="fbwd_dv_done_ptr")}
                fbwd_qe.release()
                fbwd_dote.release()
"""

    def drain(buffer: str) -> str:
        boxes = "".join(
            f"""
                fbwd_dq_flat = cute.make_tensor(tRDrDQ[None, {c // 2}, 0, 0].iterator + {16 * (c % 2)}, cute.make_layout(16))
                cute.autovec_copy(fbwd_dq_flat, {buffer}[None, None, fbwd_slot0 + {c}][fbwd_my_row, None])"""
            for c in range(8)
        )
        stores = "".join(
            f"""
                tDQsDQ, tDQgDQ = cute_cpasync_flash.tma_partition(_fbwd_tma_dq, 0, cute.make_layout(1), cute.group_modes({buffer}[None, None, fbwd_slot0 + {c}], 0, 2), cute.group_modes(fbwd_gdq[None, None, fbwd_gdq_row, fbwd_my_c0 + {c}], 0, 2))
                cute.copy(_fbwd_tma_dq, tDQsDQ, tDQgDQ)
                cute.arch.cp_async_bulk_commit_group()"""
            for c in range(8)
        )
        return (
            boxes
            + """
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()"""
            + stores
            + "\n"
        )

    # The (2,2) accumulator layout: this thread's row (of the pair's 128 Q
    # rows) and first head-dim column from the partition's coordinates; the
    # warp's eight boxes go to eight slots of its K or V tile.
    reduce = f"""        if fbwd_steps > 0:
            fbwd_dkv_shape = fbwd_pvt.partition_shape_C((128, 256))
            tDKV_frag = fbwd_pvt.make_fragment_C(fbwd_dkv_shape)
            tDQ_t = cute.make_tensor(fbwd_tmem_ptr + {tmem_dq}, tDKV_frag.layout)
            cDQ = cute.make_identity_tensor((128, 256))
            tDQcDQ = fbwd_pvt.partition_C(cDQ)
            fbwd_dq_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
            fbwd_tiled_dq_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_dq_ld_atom, tDQ_t)
            fbwd_thr_dq_ld = fbwd_tiled_dq_ld.get_slice(fbwd_local_tidx)
            tRDtDQ = fbwd_thr_dq_ld.partition_S(tDQ_t)
            tRDcDQ = fbwd_thr_dq_ld.partition_D(tDQcDQ)
            fbwd_my_m = cutlass.Int32(tRDcDQ[0][0])
            fbwd_my_row = fbwd_my_m % 32
            fbwd_my_c0 = cutlass.Int32(tRDcDQ[0][1]) // 16
            fbwd_slot0 = (warp_idx % 2) * 8
            fbwd_gdq_row = fbwd_m_tile_t * 4 + fbwd_my_m // 32
            fbwd_gdq = cute.local_tile(_fbwd_mDQt[None, None, fbwd_bhq_t], (32, 16), (None, None))
            {_WAIT.format(ptr="fbwd_dv_done_ptr", phase="0")}
            tRDrDQ = cute.make_rmem_tensor(tRDcDQ.shape, cutlass.Float32)
            cute.copy(fbwd_tiled_dq_ld, tRDtDQ, tRDrDQ)
            cute.arch.fence_view_async_tmem_load()
            _helion_flash_rt._scale_fragment_packed_f32x2(tRDrDQ, cutlass.Float32({scale_expr}))
            if warp_idx < 2:{drain("fbwd_sdq16a")}            else:{drain("fbwd_sdq16b")}            cute.arch.cp_async_bulk_wait_group(0, read=True)
"""
    mma = f"""        fbwd_s_shape = fbwd_sst.partition_shape_C((128, 128))
        tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
        fbwd_dkv_shape = fbwd_pvt.partition_shape_C((128, 256))
        tDKV_frag = fbwd_pvt.make_fragment_C(fbwd_dkv_shape)
{mma_region}"""
    return _RolePieces(tma_steps, mma, reduce, compute_blocks)


def emit_flash_bwd_flex_pair_device_body(
    *,
    plan: FlexBwdPlan,
    io_dtype: str,
    scale_expr: str,
    exp2_f32: bool,
    mask_form: str,
    bindings: list[str],
    device_function: DeviceFunction,
    reg_split: str | None,
    hook_cols: int,
    hook_pin: bool,
    bias_fold: bool,
    tile_group: int,
    exp2_emu: int,
    n_dkdv: int,
    n_dq: int,
    q_tiles: int,
    q_blocks: int,
    kv_blocks: int,
) -> str:
    """The pair kernel's device body (see the module docstring).

    ``n_dkdv`` / ``n_dq`` are the two roles' cluster counts (clusters
    ``[0, n_dkdv)`` take a KV block, ``[n_dkdv, n_dkdv + n_dq)`` a Q block of
    the ``q_tiles`` per batch-head, the rest run an empty tile);
    ``q_blocks`` / ``kv_blocks`` the BlockMask's block counts.
    """
    tmem = flex_bwd_tmem_plan(plan.head_dim, plan.head_dim_v)
    assert tmem is not None and tmem.mode == "pair"
    assert plan.operand_strides is None, "the pair kernel takes contiguous maps"
    assert plan.regions_q_captured
    bias = additive_bias_region(plan) if bias_fold else None
    _hooks, plain, masked, jac = _render_hooks(
        plan,
        device_function=device_function,
        mask_form=mask_form,
        hook_cols=hook_cols,
        bias=bias,
        aux_smem_budget=0,
    )
    plain_q, masked_q, jac_q = _render_hooks_q(
        plan,
        device_function=device_function,
        mask_form=mask_form,
        hook_cols=hook_cols,
    )
    regs_reduce, regs_compute, regs_other = _reg_split(reg_split)
    binding_lines = "".join(f"    {line}\n" for line in bindings)
    max_steps = flex_bwd_pair_table_steps(plan)
    # P gets its own staging buffer when the storage still fits (the step
    # table decides); else it shares the dS buffer (see the storage).
    split_p = flex_bwd_pair_split_p(plan)
    p_buffer = (
        "    fbwd_sP = storage.sP.get_tensor(_fbwd_pal.outer, swizzle=_fbwd_pal.inner)\n"
        if split_p
        else "    # The dK/dV clusters stage P into the dS buffer too: it is free between\n"
        "    # the dK MMA of the previous step and this step's dS write, and the dV\n"
        "    # MMA that reads it comes before the dS pass ends.\n"
        "    fbwd_sP = fbwd_sdS_nk\n"
    )
    common = {
        "plan": plan,
        "io_dtype": io_dtype,
        "scale_expr": scale_expr,
        "exp2_f32": exp2_f32,
        "hook_cols": hook_cols,
        "hook_pin": hook_pin,
        "exp2_emu": exp2_emu,
        "dv_off": tmem.dv_off,
        "dk_off": tmem.dk_off,
        "split_p": split_p,
    }
    # The fold only when the score transform is an additive bias (the knob
    # alone does not make one; see additive_bias_region).
    dkdv_compute = tuple(
        _pair_compute_block(
            h=h,
            dq_role=False,
            plain=plain,
            masked=masked,
            jac=jac,
            bias_fold=bias is not None,
            s_off=tmem.s_off,
            dp_off=tmem.dp_off,
            **common,  # pyrefly: ignore[bad-argument-type]
        )
        for h in (0, 1)
    )
    # The dQ clusters: S^T at 0 / 128, dP^T at 64 / 192 (two buffers by step
    # parity), the dQ accumulator at the plan's ``dq_off``.
    dq_compute = tuple(
        _pair_compute_block(
            h=h,
            dq_role=True,
            plain=plain_q,
            masked=masked_q,
            jac=jac_q,
            bias_fold=False,
            s_off=0,
            dp_off=64,
            **common,  # pyrefly: ignore[bad-argument-type]
        )
        for h in (0, 1)
    )
    step_decode = f"""            fbwd_e = cutlass.Int32(fbwd_sSteps[fbwd_i])
            fbwd_m_tile = fbwd_e & 65535
            fbwd_bhq = fbwd_bhk * {plan.group} + ((fbwd_e >> 16) & 255)"""
    dkdv = _dkdv_role(
        plan=plan,
        io_dtype=io_dtype,
        split_p=split_p,
        tmem_s=tmem.s_off,
        tmem_dp=tmem.dp_off,
        tmem_dv=tmem.dv_off,
        tmem_dk=tmem.dk_off,
        compute_blocks=dkdv_compute,  # pyrefly: ignore[bad-argument-type]
        step_decode=step_decode,
    )
    dq = _dq_role(
        plan=plan,
        io_dtype=io_dtype,
        scale_expr=scale_expr,
        tmem_s=0,
        tmem_dp=64,
        tmem_dq=tmem.dq_off,
        compute_blocks=dq_compute,  # pyrefly: ignore[bad-argument-type]
    )

    def role_branch(dq_text: str, dkdv_text: str) -> str:
        return (
            "        if fbwd_is_dq:\n"
            + textwrap.indent(dq_text, "    ")
            + "        else:\n"
            + textwrap.indent(dkdv_text, "    ")
        )

    compute_blocks = "".join(
        f"""    if (warp_idx >= {lo}) & (warp_idx < {lo + 4}):
        {_setmaxreg(regs_compute)}
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
"""
        + role_branch(dq.compute[h], dkdv.compute[h])
        + "        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)\n"
        for h, lo in ((0, 4), (1, 8))
    )
    walk = plan.walk
    qsub, ksub = walk.q_block // 128, walk.kv_block // 128
    table_batch_q = (
        f"fbwd_bq % {walk.table_batch}" if walk.table_batch > 1 else "cutlass.Int32(0)"
    )
    table_head_q = (
        f"fbwd_hq_t % {walk.table_heads}"
        if walk.table_heads > 1
        else "cutlass.Int32(0)"
    )
    # The dQ clusters' step table: the Q block's KV lists (full list first,
    # then the partial one), each KV block expanded to its 128-row tiles;
    # clusters past the last Q tile walk nothing.
    q_table = f"""        fbwd_tb_q = {table_batch_q}
        fbwd_th_q = {table_head_q}
        fbwd_qrow = fbwd_m_tile_t // {qsub}
        fbwd_cfq = cutlass.Int32(_fbwd_kv_full_cnt[fbwd_tb_q, fbwd_th_q, fbwd_qrow])
        fbwd_cpq = cutlass.Int32(_fbwd_kv_part_cnt[fbwd_tb_q, fbwd_th_q, fbwd_qrow])
        fbwd_steps = (fbwd_cfq + fbwd_cpq) * {ksub}
        if fbwd_qtile >= {n_dq}:
            fbwd_steps = cutlass.Int32(0)
        for fbwd_fe in cutlass.range(tidx, fbwd_steps, 512, unroll=1):
            fbwd_rem = fbwd_fe // {ksub}
            fbwd_sub = fbwd_fe % {ksub}
            fbwd_entry = cutlass.Int32(0)
            if (fbwd_rem >= 0) & (fbwd_rem < fbwd_cfq):
                fbwd_kvblk = cutlass.Int32(_fbwd_kv_full_idx[fbwd_tb_q, fbwd_th_q, fbwd_qrow, fbwd_rem])
                fbwd_entry = fbwd_kvblk * {ksub} + fbwd_sub
            fbwd_rem = fbwd_rem - fbwd_cfq
            if (fbwd_rem >= 0) & (fbwd_rem < fbwd_cpq):
                fbwd_kvblk = cutlass.Int32(_fbwd_kv_part_idx[fbwd_tb_q, fbwd_th_q, fbwd_qrow, fbwd_rem])
                fbwd_entry = (fbwd_kvblk * {ksub} + fbwd_sub) | (1 << 24)
            fbwd_sSteps[fbwd_fe] = fbwd_entry
"""
    kv_table = textwrap.indent(_step_table_fill(plan) + _step_table_loop(plan), "    ")
    table_region = (
        "    fbwd_steps = cutlass.Int32(0)\n    if fbwd_is_dq:\n"
        + q_table
        + "    else:\n"
        + kv_table
    )
    pipelines = "".join(
        f"    fbwd_{name}_prod, fbwd_{name}_cons = {_PIPELINE.format(name=name, stages=1)}\n"
        for name in ("q", "dot", "do", "qt", "k", "v")
    ) + (
        "    fbwd_kt2_prod, fbwd_kt2_cons = "
        + _PIPELINE.format(name="kt2", stages=2).replace(
            "tx_count=fbwd_kt2_bytes", "tx_count=fbwd_do_bytes"
        )
        + "\n"
    )
    # Every full barrier of the pair counts both CTAs' tile bytes (the loads
    # complete on the leader's barrier; the follower's never completes and is
    # never waited on). The dQ clusters reuse the rings for their operands
    # of the same size: q for Q (A), dot for dO (A), k / v for K / V (B),
    # do for the K^T tile.
    ring_bytes = "".join(
        f"    fbwd_{name}_bytes = cute.size_in_bytes({io_dtype}, cute.select({layout}, mode=[0, 1, 2])) * 2\n"
        for name, layout in (
            ("q", "_fbwd_qsl"),
            ("dot", "_fbwd_dosl"),
            ("do", "_fbwd_dotl"),
            ("qt", "_fbwd_qtl"),
            ("k", "_fbwd_ksl"),
            ("v", "_fbwd_vsl"),
        )
    )
    prefetch = "".join(
        f"        cute_cpasync_flash.prefetch_descriptor({atom})\n"
        for atom in (
            "_fbwd_tma_q",
            "_fbwd_tma_k",
            "_fbwd_tma_v",
            "_fbwd_tma_dot",
            "_fbwd_tma_do2",
            "_fbwd_tma_qt",
            "_fbwd_tma_dq",
            "_fbwd_tma_qa",
            "_fbwd_tma_doa",
            "_fbwd_tma_kb",
            "_fbwd_tma_vb",
            "_fbwd_tma_kt2",
        )
    )
    if n_dq == n_dkdv:
        # As many dQ clusters as dK/dV clusters: alternate the roles in launch
        # order so a batch-head's two walks share the L2 at the same time.
        role_decode = """    # Even clusters own a KV block (the dK/dV role), odd clusters a Q block
    # of one batch-head (the dQ role), in the same order.
    fbwd_block = cutlass.Int32(cute.arch.block_idx()[0])
    fbwd_cluster = fbwd_block // 2
    fbwd_is_dq = cute.arch.make_warp_uniform(fbwd_cluster % 2) >= cutlass.Int32(1)
    fbwd_tile_id = fbwd_cluster // 2
    fbwd_qtile = fbwd_cluster // 2
"""
    else:
        role_decode = f"""    # Clusters [0, {n_dkdv}) own a KV block (the dK/dV role), the next
    # {n_dq} a Q block of one batch-head (the dQ role); any further
    # cluster runs an empty tile.
    fbwd_block = cutlass.Int32(cute.arch.block_idx()[0])
    fbwd_cluster = fbwd_block // 2
    fbwd_is_dq = cute.arch.make_warp_uniform(fbwd_cluster // {n_dkdv}) >= cutlass.Int32(1)
    fbwd_tile_id = cutlass.min(fbwd_cluster, cutlass.Int32({n_dkdv - 1}))
    fbwd_qtile = fbwd_cluster - {n_dkdv}
"""
    body = f"""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    fbwd_local_tidx = tidx % 128
    fbwd_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    fbwd_is_leader = fbwd_rank == cutlass.Int32(0)
{role_decode}{_tile_decode(plan, tile_group)}    fbwd_b = fbwd_bhk // {plan.kv_heads}
    fbwd_hk = fbwd_bhk % {plan.kv_heads}
    fbwd_kv_base = fbwd_n_tile * 128 + fbwd_rank * 64
    fbwd_qt_id = cutlass.max(cutlass.min(fbwd_qtile, cutlass.Int32({n_dq - 1})), cutlass.Int32(0))
    fbwd_bhq_t = fbwd_qt_id // {q_tiles}
    fbwd_m_tile_t = fbwd_qt_id % {q_tiles}
    fbwd_bq = fbwd_bhq_t // {plan.heads}
    fbwd_hq_t = fbwd_bhq_t % {plan.heads}
    fbwd_bhk_q = fbwd_bq * {plan.kv_heads} + fbwd_hq_t // {plan.group}
{binding_lines}    if warp_idx == 0:
{prefetch}    _fbwd_storage_cls = _helion_flash_rt.flash_bwd_flex_pair_shared_storage({io_dtype}, {max_steps}, {split_p})
    smem = cutlass_utils_flash.SmemAllocator()
    storage = smem.allocate(_fbwd_storage_cls)
    fbwd_sSteps = storage.sSteps.get_tensor(cute.make_layout({max_steps}))
{table_region}    sQ = storage.sQ.get_tensor(_fbwd_qsl.outer, swizzle=_fbwd_qsl.inner)
    sdOt = storage.sdOt.get_tensor(_fbwd_dosl.outer, swizzle=_fbwd_dosl.inner)
    sdO = storage.sdOqt.get_tensor(_fbwd_dotl.outer, swizzle=_fbwd_dotl.inner)
    sQt = cute.make_tensor(cute.recast_ptr(storage.sdOqt.data_ptr() + 16384, _fbwd_qtl.inner), _fbwd_qtl.outer)
    sK = storage.sK.get_tensor(_fbwd_ksl.outer, swizzle=_fbwd_ksl.inner)
    sV = storage.sV.get_tensor(_fbwd_vsl.outer, swizzle=_fbwd_vsl.inner)
    fbwd_sdS_nk = storage.sdS.get_tensor(_fbwd_pal.outer, swizzle=_fbwd_pal.inner)
{p_buffer}
    # The dQ clusters' views of the same buffers: the resident Q / dO as A
    # operands, the step's K / V as B operands (the K-major (64, 256) layouts
    # coincide), the two-stage K^T ring over the dO^T / Q^T tiles, and the
    # once-per-tile dQ drain staged through the K / V tiles.
    sQa = storage.sQ.get_tensor(_fbwd_ksl.outer, swizzle=_fbwd_ksl.inner)
    sdOa = storage.sdOt.get_tensor(_fbwd_vsl.outer, swizzle=_fbwd_vsl.inner)
    sKt2 = storage.sdOqt.get_tensor(_fbwd_dotl2.outer, swizzle=_fbwd_dotl2.inner)
    sKb = storage.sK.get_tensor(_fbwd_qsl.outer, swizzle=_fbwd_qsl.inner)
    sVb = storage.sV.get_tensor(_fbwd_dosl.outer, swizzle=_fbwd_dosl.inner)
    fbwd_sdq16a = cute.make_tensor(cute.recast_ptr(storage.sK.data_ptr(), _fbwd_dqsl16.inner, dtype=cutlass.Float32), _fbwd_dqsl16.outer)
    fbwd_sdq16b = cute.make_tensor(cute.recast_ptr(storage.sV.data_ptr(), _fbwd_dqsl16.inner, dtype=cutlass.Float32), _fbwd_dqsl16.outer)
    fbwd_sLSE = storage.sLSE.get_tensor(cute.make_layout(2 * 128))
    fbwd_sDelta = storage.sDelta.get_tensor(cute.make_layout(2 * 128))
    fbwd_p_full_ptr = storage.p_full_mbar.data_ptr()
    fbwd_ds_full_ptr = storage.ds_full_mbar.data_ptr()
    fbwd_sdsfree_ptr = storage.sds_free_mbar.data_ptr()
    fbwd_pv_done_ptr = storage.pv_done_mbar.data_ptr()
    fbwd_dv_done_ptr = storage.dv_done_mbar.data_ptr()
    fbwd_stq_full_ptr = storage.stq_full_mbar.data_ptr()
    fbwd_dptq_full_ptr = storage.dptq_full_mbar.data_ptr()
    fbwd_stq_empty_ptr = storage.stq_empty_mbar.data_ptr()
    fbwd_dsq_full_ptr = storage.dsq_full_mbar.data_ptr()
    fbwd_dqq_full_ptr = storage.dqq_full_mbar.data_ptr()
    fbwd_tmem_dealloc_ptr = storage.tmem_dealloc_mbar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(fbwd_p_full_ptr, 16)
        cute.arch.mbarrier_init(fbwd_ds_full_ptr, 16)
        cute.arch.mbarrier_init(fbwd_sdsfree_ptr, 1)
        cute.arch.mbarrier_init(fbwd_pv_done_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dv_done_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dsq_full_ptr, 16)
        cute.arch.mbarrier_init(fbwd_dqq_full_ptr, 1)
        for fbwd_bi in cutlass.range_constexpr(2):
            cute.arch.mbarrier_init(fbwd_stq_full_ptr + fbwd_bi, 1)
            cute.arch.mbarrier_init(fbwd_dptq_full_ptr + fbwd_bi, 1)
            cute.arch.mbarrier_init(fbwd_stq_empty_ptr + fbwd_bi, 16)
    cute.arch.mbarrier_init_fence()
    fbwd_tmem_user_bar = cutlass_pipeline_flash.NamedBarrier(barrier_id=2, num_threads=13 * 32)
    fbwd_tmem = cutlass_utils_flash.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=fbwd_tmem_user_bar, allocator_warp_id=12, is_two_cta=True, two_cta_tmem_dealloc_mbar_ptr=fbwd_tmem_dealloc_ptr)
{ring_bytes}{pipelines}    cutlass_pipeline_flash.pipeline_init_arrive(cluster_shape_mn=_fbwd_cluster_vmnk, is_relaxed=True)
    cutlass_pipeline_flash.pipeline_init_wait(cluster_shape_mn=_fbwd_cluster_vmnk)
    fbwd_sst = _fbwd_ss_mma.get_slice(fbwd_rank)
    fbwd_pvt = _fbwd_pv_mma.get_slice(fbwd_rank)
    fbwd_mcast = cutlass_pipeline_flash.PipelineUmmaAsync._compute_tmem_sync_mask(_fbwd_cluster_vmnk)
    gK = cute.flat_divide(_fbwd_mKt, cute.select((128, 128, 256), mode=[0, 2]))
    gV = cute.flat_divide(_fbwd_mVt, cute.select((128, 128, 256), mode=[0, 2]))
    gQ = cute.flat_divide(_fbwd_mQt, cute.select((128, 128, 256), mode=[1, 2]))
    gdOt = cute.flat_divide(_fbwd_mdOtN, cute.select((128, 128, 256), mode=[1, 2]))
    gdO = cute.flat_divide(_fbwd_mdOtT, cute.select((128, 256, 128), mode=[1, 2]))
    gQt = cute.flat_divide(_fbwd_mQtT, cute.select((128, 256, 128), mode=[1, 2]))
    gQa = cute.flat_divide(_fbwd_mQa, cute.select((128, 128, 256), mode=[0, 2]))
    gdOa = cute.flat_divide(_fbwd_mdOa, cute.select((128, 128, 256), mode=[0, 2]))
    gKb = cute.flat_divide(_fbwd_mKb, cute.select((128, 128, 256), mode=[1, 2]))
    gVb = cute.flat_divide(_fbwd_mVb, cute.select((128, 128, 256), mode=[1, 2]))
    gKt2 = cute.flat_divide(_fbwd_mKt2, cute.select((128, 256, 128), mode=[1, 2]))
    tSgK = fbwd_sst.partition_A(gK)
    tSgV = fbwd_sst.partition_A(gV)
    tSgQ = fbwd_sst.partition_B(gQ)
    tSgDOt = fbwd_sst.partition_B(gdOt)
    tPVgDO = fbwd_pvt.partition_B(gdO)
    tPVgQt = fbwd_pvt.partition_B(gQt)
    tSgQa = fbwd_sst.partition_A(gQa)
    tSgdOa = fbwd_sst.partition_A(gdOa)
    tSgKb = fbwd_sst.partition_B(gKb)
    tSgVb = fbwd_sst.partition_B(gVb)
    tPVgKt2 = fbwd_pvt.partition_B(gKt2)
    tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
    tVsV, tVgV_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tSgV, 0, 3))
    tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
    tDOTsDOT, tDOTgDOT_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_dot, 0, cute.make_layout(1), cute.group_modes(sdOt, 0, 3), cute.group_modes(tSgDOt, 0, 3))
    tDOsDO, tDOgDO_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_do2, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 3), cute.group_modes(tPVgDO, 0, 3))
    tQTsQT, tQTgQT_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_qt, 0, cute.make_layout(1), cute.group_modes(sQt, 0, 3), cute.group_modes(tPVgQt, 0, 3))
    tQAsQA, tQAgQA_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_qa, 0, cute.make_layout(1), cute.group_modes(sQa, 0, 3), cute.group_modes(tSgQa, 0, 3))
    tDOAsDOA, tDOAgDOA_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_doa, 0, cute.make_layout(1), cute.group_modes(sdOa, 0, 3), cute.group_modes(tSgdOa, 0, 3))
    tKBsKB, tKBgKB_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_kb, 0, cute.make_layout(1), cute.group_modes(sKb, 0, 3), cute.group_modes(tSgKb, 0, 3))
    tVBsVB, tVBgVB_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_vb, 0, cute.make_layout(1), cute.group_modes(sVb, 0, 3), cute.group_modes(tSgVb, 0, 3))
    tKT2sKT2, tKT2gKT2_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_kt2, 0, cute.make_layout(1), cute.group_modes(sKt2, 0, 3), cute.group_modes(tPVgKt2, 0, 3))
    if warp_idx == 13:
        # The tile's resident operands: K / V of the KV block (dK/dV role),
        # Q / dO of the Q block (dQ role).
        if fbwd_steps > 0:
            if fbwd_is_dq:
                fbwd_qe = fbwd_q_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_qa, tQAgQA_qdl[None, None, 0, fbwd_bhq_t][None, fbwd_m_tile_t], tQAsQA[None, fbwd_qe.index], tma_bar_ptr=fbwd_qe.barrier)
                fbwd_dote = fbwd_dot_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_doa, tDOAgDOA_qdl[None, None, 0, fbwd_bhq_t][None, fbwd_m_tile_t], tDOAsDOA[None, fbwd_dote.index], tma_bar_ptr=fbwd_dote.barrier)
            else:
                fbwd_ke = fbwd_k_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_k, tKgK_kdl[None, None, 0, fbwd_bhk][None, fbwd_n_tile], tKsK[None, fbwd_ke.index], tma_bar_ptr=fbwd_ke.barrier)
                fbwd_ve = fbwd_v_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_v, tVgV_kdl[None, None, 0, fbwd_bhk][None, fbwd_n_tile], tVsV[None, fbwd_ve.index], tma_bar_ptr=fbwd_ve.barrier)
    cute.arch.sync_threads()
    if (warp_idx >= 12) & (warp_idx < 16):
        {_setmaxreg(regs_other)}
    if warp_idx == 13:
{role_branch(dq.tma, dkdv.tma)}    if warp_idx == 12:
        fbwd_tmem.allocate(512)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
{role_branch(dq.mma, dkdv.mma)}        fbwd_tmem.relinquish_alloc_permit()
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem.free(fbwd_tmem_ptr)
    if warp_idx < 4:
        {_setmaxreg(regs_reduce)}
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
{role_branch(dq.reduce, dkdv.reduce)}        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)
{compute_blocks}"""
    return body  # noqa: RET504


def codegen_flex_bwd_pair(
    cg: GenerateAST,
    match: FlexBwdMatch,
    *,
    args: dict[str, TensorArg],
    scalar_exprs: list[str],
    scale_expr: str,
    io_dtype_str: str,
    exp2_f32: bool,
    mask_form: str,
    reg_split: object,
    hook_cols: int,
    hook_pin: bool,
    bias_fold: bool,
    tile_group: int,
) -> bool:
    """Emit the pair kernel (the arguments were validated by
    :func:`cute_flash_bwd_flex.codegen_attention_flash_bwd_flex`)."""
    import ast as ast_module

    from ..compile_environment import CompileEnvironment
    from .cute_flash import emit_flash_module_statements

    df = cg.device_function
    plan = match.plan
    env = CompileEnvironment.current()
    batch, heads, kv_heads = plan.batch, plan.heads, plan.kv_heads
    exp2_emu_cfg = df.config.config.get("cute_flash_bwd_exp2_emu", 0)
    exp2_emu = (
        exp2_emu_cfg if isinstance(exp2_emu_cfg, int) and env.settings.fast_math else 0
    )
    walk = plan.walk
    emit_flash_module_statements(cg)
    bindings = [
        f"_flash_heads = {heads}",
        f"_fbwd_part_cnt = {args[walk.part_cnt_name].name}",
        f"_fbwd_part_idx = {args[walk.part_idx_name].name}",
        f"_fbwd_full_cnt = {args[walk.full_cnt_name].name}",
        f"_fbwd_full_idx = {args[walk.full_idx_name].name}",
    ]
    for index, buffer in enumerate(plan.aux_buffers):
        bindings.append(f"_flash_aux{index} = {args[buffer.name].name}")
    for index, expr in enumerate(scalar_exprs):
        bindings.append(f"_flash_hs{index} = {expr}")
    # The dQ clusters' KV-side lists, derived on the host from the Q-side
    # tables (``flex_bwd_kv_tables``).
    q_blocks = -(-plan.q_seq // walk.q_block)
    kv_blocks = -(-plan.kv_seq // walk.kv_block)
    kv_args = _kv_table_args(
        cg,
        df,
        [
            args[walk.part_cnt_name],
            args[walk.part_idx_name],
            args[walk.full_cnt_name],
            args[walk.full_idx_name],
        ],
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
    )
    bindings.extend(
        f"_fbwd_{name} = {arg.name}"
        for name, arg in zip(_KV_TABLE_NAMES, kv_args, strict=True)
    )
    # Two (2, 1, 1) clusters' worth of CTAs per KV block's share: the dK/dV
    # clusters (one per KV block), then the dQ clusters (one per Q block of
    # every batch-head); the grid is the KV-block count times this factor.
    q_tiles = -(-plan.q_seq // 128)
    n_dkdv = batch * kv_heads * plan.num_kv_tiles
    n_dq = batch * heads * q_tiles
    grid_multiplier = 2 * (-(-(n_dkdv + n_dq) // n_dkdv))
    wrapper_plan: dict[str, object] = {
        "kind": "helion_flash_bwd",
        "flex_pair": True,
        "q_name": args[match.q_name].name,
        "k_name": args[match.k_name].name,
        "v_name": args[match.v_name].name,
        "do_name": args[match.do_name].name,
        "lse_name": args[match.lse_name].name,
        "delta_name": args[match.delta_name].name,
        "dq_name": args[match.dq_name].name,
        "dk_name": args[match.dk_name].name,
        "dv_name": args[match.dv_name].name,
        "head_dim": plan.head_dim,
        "dtype": io_dtype_str,
        "total_q_rows": plan.num_bh * plan.q_seq,
        "total_kv_rows": batch * kv_heads * plan.kv_seq,
        "q_stage": 1,
        "do_stage": 1,
        "kv_stage": 1,
        "two_cta": False,
        "persistent": False,
        "total_tiles": batch * kv_heads * plan.num_kv_tiles,
        "topology": "fa4",
        "flex_seq_q": plan.q_seq,
        "flex_bh_q": plan.num_bh,
        "flex_seq_k": plan.kv_seq,
        "flex_bh_k": batch * kv_heads,
    }
    cg.cute_wrapper_plans.append(wrapper_plan)  # type: ignore[attr-defined]
    df.wrapper_only_params.extend(_FLASH_BWD_FLEX_PAIR_KERNEL_PARAMS)
    df.placeholder_args.update(a.name for a in args.values())
    cg.cute_uses_matmul = True  # type: ignore[attr-defined]
    df.cute_state.attention_flash_threads = 512
    df.cute_state.cluster_shape = (2, 1, 1)
    df.cute_state.launch_grid_multiplier = grid_multiplier
    body_src = emit_flash_bwd_flex_pair_device_body(
        plan=plan,
        io_dtype=io_dtype_str,
        scale_expr=scale_expr,
        exp2_f32=exp2_f32,
        mask_form=mask_form,
        bindings=bindings,
        device_function=df,
        reg_split=None if reg_split is None else str(reg_split),
        hook_cols=hook_cols,
        hook_pin=hook_pin,
        bias_fold=bias_fold,
        tile_group=tile_group,
        exp2_emu=exp2_emu,
        n_dkdv=n_dkdv,
        n_dq=n_dq,
        q_tiles=q_tiles,
        q_blocks=q_blocks,
        kv_blocks=kv_blocks,
    )
    wrapped = ast_module.parse("if True:\n" + body_src)
    assert isinstance(wrapped.body[0], ast_module.If)
    df.body = list(wrapped.body[0].body)
    df.preamble = []
    return True
