"""Fused tcgen05 attention-BACKWARD emitter for the FlexAttention envelope.

The matched kernel (``flex_bwd_match.py``) is lowered to the same 512-thread
SM100 structure as the dense backward (``cute_flash_bwd.py``: 4 dQ-reduce
warps, two compute warpgroups owning the column halves of the 128x128 S^T /
dP^T tiles, one MMA warp, one TMA warp; TMEM-resident dK/dV; dQ by TMA
reduce-add) with three generalizations:

* **The Q walk is a list.** Each CTA owns one 128-row KV tile of one (batch,
  kv head). At setup every thread fills a shared-memory *step table* from the
  BlockMask's Q-side lists of the group's q heads (full list first, then the
  partial list; one entry per 128-row Q tile, ``q tile | q head << 16 |
  masked << 24``), and every warp role reads its step from the table instead
  of recomputing an interval walk. An empty table (a KV tile no query block
  attends) stores zero dK/dV and skips the pipeline.
* **Per-(batch, head) TMA maps.** Q/dO/dQ are ``(M, D, B*H)`` and K/V/dK/dV
  ``(N, D, B*Hk)`` tensors, so a ragged sequence length (``M % 128`` or
  ``N % 128`` nonzero) is zero-filled on load and clipped on store by the
  TMA unit; only the LSE/delta staging clamps its column index.
* **score_mod / mask_mod / score_mod_bwd hooks.** The regions are rendered
  by the forward's per-element renderer (``flash_score_mod.render_flex_hook``)
  in the transposed frame (its row coordinate is the KV index, its lane the
  Q index; see ``flex_bwd_match``): the score hook transforms the scaled S^T
  chunk before the exp2 recompute, the mask hook selects ``-inf`` on the
  partial-list steps, and the Jacobian hook ``J = score_mod_bwd(1, S)``
  (rendered on a register copy of the pre-modification scores) multiplies dS
  before its bf16 convert.
"""

from __future__ import annotations

import dataclasses
import textwrap
from typing import TYPE_CHECKING

import torch

from .attention_plan import AttentionScorePlan
from .attention_plan import FlexRegion
from .flash_score_mod import FlexAuxSmem
from .flash_score_mod import FlexHookCode
from .flash_score_mod import flex_aux_smem_candidates
from .flash_score_mod import flex_aux_smem_layout
from .flash_score_mod import render_flex_hook

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from ..device_function import TensorArg
    from ..generate_ast import GenerateAST
    from .flex_bwd_match import FlexBwdMatch
    from .flex_bwd_match import FlexBwdPlan

_aten = torch.ops.aten

__all__ = [
    "codegen_attention_flash_bwd_flex",
    "emit_flash_bwd_flex_device_body",
    "flex_bwd_hook_plan",
]


def flex_bwd_hook_plan(
    plan: FlexBwdPlan,
    *,
    grad: bool = False,
    score_region: FlexRegion | None = None,
    with_mask: bool = True,
) -> AttentionScorePlan:
    """The forward-style plan the hook renderer consumes: the backward's
    regions (already in the transposed frame) as ``score_region`` /
    ``mask_region``; with ``grad`` the Jacobian region as the score region;
    ``score_region`` overrides the score region (the additive bias ``g``)."""
    if score_region is None:
        score_region = plan.grad_region if grad else plan.score_region
    return AttentionScorePlan(
        head_dim=plan.head_dim,
        qk_scale_log2=1.0,
        lse_scale=1.0,
        modifiers=(),
        kv_group=plan.group,
        head_dim_v=None,
        # Swapped: the renderer's ``q`` axis is the KV sequence here.
        q_seq=plan.kv_seq,
        kv_seq=plan.q_seq,
        kv_walk=None,
        score_region=score_region,
        mask_region=plan.mask_region if (with_mask and not grad) else None,
        aux_buffers=plan.aux_buffers,
        host_scalars=plan.host_scalars,
        scale_arg=plan.scale_arg,
        scale_peeled=True,
    )


def fbwd_aux_smem_candidates(plan: FlexBwdPlan) -> tuple[int, ...]:
    """Padded byte sizes of the aux tables worth staging in shared memory
    (``flex_aux_smem_candidates`` over the backward's hook plan): the tables
    some load indexes with a data-dependent value (the backward's per-element
    gathers are latency chains behind the table loads)."""
    return flex_aux_smem_candidates(flex_bwd_hook_plan(plan))


def fbwd_aux_smem_layout(plan: FlexBwdPlan, budget: int) -> list[FlexAuxSmem]:
    """The staged tables of ``cute_flash_bwd_aux_smem`` with their offsets."""
    return flex_aux_smem_layout(flex_bwd_hook_plan(plan), budget)


def additive_bias_region(plan: FlexBwdPlan) -> FlexRegion | None:
    """``g`` of a score region ``score + g`` whose ``g`` does not read the score
    (an additive bias: ALiBi, relative / table biases), as a region of its own;
    None for any other score transform or when a Jacobian region exists."""
    region = plan.score_region
    if region is None or plan.grad_region is not None:
        return None
    out = region.output
    if out.target is not _aten.add.Tensor or len(out.args) != 2:
        return None
    score_ph = next(n for n in region.graph.nodes if n.op == "placeholder")
    lhs, rhs = out.args
    if lhs is score_ph and isinstance(rhs, torch.fx.Node):
        g = rhs
    elif rhs is score_ph and isinstance(lhs, torch.fx.Node):
        g = lhs
    else:
        return None
    # ``g`` must not depend on the score placeholder.
    stack, seen = [g], set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node is score_ph:
            return None
        stack.extend(a for a in node.all_input_nodes)
    graph = torch.fx.Graph()
    mapping: dict[torch.fx.Node, torch.fx.Node] = {}
    graph.graph_copy(region.graph, mapping, return_output_node=False)
    graph.output(mapping[g])
    graph.eliminate_dead_code()
    graph.lint()
    return FlexRegion(graph, mapping[g])


def _lines(text: str, indent: int) -> str:
    """Re-indent a hook block (complete lines) to ``indent`` spaces."""
    if not text.strip():
        return ""
    body = textwrap.indent(textwrap.dedent(text).rstrip("\n"), " " * indent)
    return body + "\n"


def _merge_blocks(*texts: str) -> str:
    """The union of the hooks' prelude lines in first-seen order (the renderer
    emits identical lines for shared values, so exact-text dedup is exact)."""
    seen: list[str] = []
    for text in texts:
        for line in textwrap.dedent(text).splitlines():
            if line not in seen:
                seen.append(line)
    return "\n".join(seen) + ("\n" if seen else "")


def _render_hooks(
    plan: FlexBwdPlan,
    *,
    device_function: DeviceFunction,
    mask_form: str,
    hook_cols: int = 32,
    bias: FlexRegion | None = None,
    aux_smem_budget: int = 0,
) -> tuple[
    list[FlexHookCode], FlexHookCode | None, FlexHookCode | None, FlexHookCode | None
]:
    """The unmasked hook, the masked hook and the Jacobian hook (each None when
    the plan has no region for it). With ``bias`` (the folded additive score
    bias) the unmasked hook renders ``g`` into ``fbwd_gfrag`` and the masked
    hook carries the mask only. ``aux_smem_budget`` (bytes) selects the aux
    tables the hooks read from their shared-memory copy: the fill's set
    (``fbwd_aux_smem_layout`` over the whole plan), passed to every hook so
    a sub-plan's own candidate list never names a pointer the fill did not
    bind."""
    has_score = plan.score_region is not None and bias is None
    has_mask = plan.mask_region is not None
    has_grad = plan.grad_region is not None
    hook_plan = flex_bwd_hook_plan(plan)
    if bias is not None:
        # The score transform is folded into the LSE: the masked hook carries
        # the mask only.
        hook_plan = dataclasses.replace(hook_plan, score_region=None)
    common: dict[str, object] = {
        "stage": 0,
        "coord_tensor": "fbwd_cchunk",
        "m_tile_expr": "fbwd_n_tile",
        "kv_tile_expr": "fbwd_m_tile_h",
        "chunk_expr": "fbwd_ch2",
        "chunk_cols": hook_cols,
        "indent": 20,
        "item_indent": 8,
        "tile_indent": 12,
        "row_coord_expr": "fbwd_thr_row",
        "device_function": device_function,
        # Bounds of every row / column the hook can evaluate: the KV tiles of
        # the renderer's row axis, the Q tiles of its lane axis.
        "coord_bounds": (
            plan.batch,
            plan.heads,
            plan.num_kv_tiles * 128,
            plan.num_q_tiles_padded * 128,
        ),
        "mask_form": mask_form,
        "aux_smem_budget": aux_smem_budget,
        "aux_smem_staged": frozenset(
            entry.aux_index for entry in fbwd_aux_smem_layout(plan, aux_smem_budget)
        ),
    }
    hooks: list[FlexHookCode] = []
    plain = masked = jac = None
    if bias is not None:
        plain = render_flex_hook(
            flex_bwd_hook_plan(plan, score_region=bias, with_mask=False),
            score_tensor="fbwd_gfrag",
            with_score=True,
            with_mask=False,
            tail_mask=False,
            **common,  # pyrefly: ignore[bad-argument-type]
        )
        hooks.append(plain)
    elif has_score:
        plain = render_flex_hook(
            hook_plan,
            score_tensor="fbwd_s_chunk",
            with_score=True,
            with_mask=False,
            tail_mask=False,
            **common,  # pyrefly: ignore[bad-argument-type]
        )
        hooks.append(plain)
    if has_mask:
        masked = render_flex_hook(
            hook_plan,
            score_tensor="fbwd_s_chunk",
            with_score=has_score,
            with_mask=True,
            tail_mask=False,
            **common,  # pyrefly: ignore[bad-argument-type]
        )
        hooks.append(masked)
    if has_grad:
        jac = render_flex_hook(
            flex_bwd_hook_plan(plan, grad=True),
            score_tensor="fbwd_spre_chunk",
            with_score=True,
            with_mask=False,
            tail_mask=False,
            **common,  # pyrefly: ignore[bad-argument-type]
        )
        hooks.append(jac)
    return hooks, plain, masked, jac


def _compute_block(
    *,
    h: int,
    plan: FlexBwdPlan,
    io_dtype: str,
    scale_expr: str,
    out_scale_expr: str,
    exp2_f32: bool,
    plain: FlexHookCode | None,
    masked: FlexHookCode | None,
    jac: FlexHookCode | None,
    regs_compute: int = 136,
    hook_cols: int = 32,
    hook_pin: bool = True,
    bias_fold: bool = False,
    lse_in_mma: bool = False,
    delta_in_mma: bool = False,
    exp2_emu: int = 0,
    persistent: bool = False,
    p_head: str = "",
    p_tail: str = "",
    ds_tmem: bool = False,
) -> str:
    """One compute warpgroup's body (h = column half 0/1, warps 4-7 / 8-11).

    ``lse_in_mma`` / ``delta_in_mma``: the S / dP MMAs already subtracted the
    per-column LSE / delta terms (statistics slices, see
    ``_fit_smem_budget``), so the P pass runs on a zero LSE fragment and the
    dS pass is the plain ``P * dP'`` product; no per-step staging of them.
    ``ds_tmem`` (knob ``cute_flash_bwd_ds_tmem``): the dS pass also packs its
    bf16 dS^T into the (consumed) dP columns of TMEM, the way P lands in the
    S columns, so the dK MMA reads its A operand from TMEM (TS mode) instead
    of the smem dS staging (one 32 KB smem read per step less; the staging
    stays for the dQ MMA, whose M axis is Q).
    """
    from .cute_flash_bwd import _fbwd_epilogue_block

    d = plan.head_dim
    warp_lo, warp_hi = (4, 8) if h == 0 else (8, 12)
    has_score = plan.score_region is not None and not bias_fold
    has_grad = plan.grad_region is not None
    # With the additive bias folded into the LSE no hook reads the score, so
    # the scale stays fused in the exp2 pass.
    pre_scaled = has_score or has_grad
    # ``c`` of exp2(score * c - lse * c): the natural-unit score when the
    # hooks need it (then the scale is applied per element first), else the
    # scale folded in.
    p_scale = "fbwd_log2c" if pre_scaled else "fbwd_scale2"
    hooks = [hook for hook in (plain, masked, jac) if hook is not None]
    item_block = _merge_blocks(*(hook.item_prelude for hook in hooks))
    tile_block = _merge_blocks(*(hook.tile_prologue for hook in hooks))
    per_step_item = plan.group > 1

    # The P pass of one 32-column chunk runs in ``hook_cols``-lane groups: the
    # hook transforms a group's scores and the exp2 recompute consumes them
    # before the next group's hook runs, so ptxas has no long run of
    # precomputable hook values to hoist above the S wait (where they spill).
    subs = 32 // hook_cols

    def chunk_prefix(ind: str) -> str:
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

    def bias_lines(ind: str) -> str:
        """The folded bias: ``g`` rendered into ``fbwd_gfrag``, then
        ``lse' = lse - c g`` on the sub-chunk's LSE values."""
        if not bias_fold or plain is None:
            return ""
        return (
            f"{ind}fbwd_gfrag = cute.make_rmem_tensor(cute.make_layout({hook_cols}), cutlass.Float32)\n"
            + _lines(plain.chunk_body, len(ind))
            + f"{ind}_helion_flash_rt.fbwd_lse_fold(fbwd_lse_sub, fbwd_gfrag, fbwd_neg_log2c)\n"
        )

    def p_pass(hook: FlexHookCode | None, ind: str) -> str:
        # Under the fold ``hook`` is the mask hook (masked steps) or None.
        return (
            f"{ind}for fbwd_ch in cutlass.range_constexpr(2):\n"
            f"{ind}    fbwd_lse_frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)\n"
            + (
                f"{ind}    fbwd_lse_frg.fill(cutlass.Float32(0.0))\n"
                if lse_in_mma
                else f"{ind}    fbwd_lse_v = cute.make_tensor(fbwd_sLSE.iterator + fbwd_par + {64 * h} + fbwd_ch * 32, cute.make_layout(32))\n"
                f"{ind}    cute.autovec_copy(fbwd_lse_v, fbwd_lse_frg)\n"
            )
            # A natural-log LSE (the in-body form) goes to the exp2 base with the
            # no-visible-key rows (lse == -inf) mapped to 0; the preprocess
            # kernel's ``lse2`` arrives converted.
            + (
                ""
                if plan.lse_prescaled or lse_in_mma
                else f"{ind}    _helion_flash_rt.fbwd_lse_to_log2(fbwd_lse_frg, fbwd_log2c)\n"
            )
            + f"{ind}    for fbwd_sub in cutlass.range_constexpr({subs}):\n"
            + chunk_prefix(ind + "        ")
            + f"{ind}        fbwd_lse_sub = cute.make_tensor(fbwd_lse_frg.iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))\n"
            + bias_lines(ind + "        ")
            + (_lines(hook.chunk_body, len(ind) + 8) if hook is not None else "")
            + f"{ind}        _helion_flash_rt.fbwd_p_pairs_packed(tLDrS, fbwd_lse_sub, fbwd_off, {hook_cols}, {p_scale}, None, fbwd_off, exp2_f32={exp2_f32}, emu_res={exp2_emu})\n"
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
            f"                for fbwd_sub in cutlass.range_constexpr({subs}):\n"
            f"                    fbwd_ch2 = fbwd_ch * {subs} + fbwd_sub + {2 * subs * h}\n"
            f"                    fbwd_spre_chunk = cute.make_tensor(fbwd_spre[None, None, fbwd_ch].iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))\n"
            f"                    fbwd_cchunk = cute.logical_divide(tLDcS[None, None, fbwd_ch], {hook_cols})[None, fbwd_sub]\n"
            + _lines(jac.chunk_body, 20)
            + f"                    fbwd_ds_sub = cute.make_tensor(tLDrdP.iterator + fbwd_sub * {hook_cols}, cute.make_layout({hook_cols}))\n"
            f"                    _helion_flash_rt.fbwd_mul_pairs_packed(fbwd_ds_sub, fbwd_spre, fbwd_ch * 32 + fbwd_sub * {hook_cols}, {hook_cols})\n"
        )
    spre_alloc = (
        "            fbwd_spre = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)\n"
        if has_grad
        else ""
    )
    ds_lines = (
        "                cute.arch.fence_view_async_tmem_load()\n"
        "                _helion_flash_rt.fbwd_mul_pairs_packed(tLDrdP, tLDrS, fbwd_ch * 32, 32)\n"
        if delta_in_mma
        else "                fbwd_dlt_frg = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)\n"
        f"                fbwd_dlt_v = cute.make_tensor(fbwd_sDelta.iterator + fbwd_par + {64 * h} + fbwd_ch * 32, cute.make_layout(32))\n"
        "                cute.autovec_copy(fbwd_dlt_v, fbwd_dlt_frg)\n"
        "                cute.arch.fence_view_async_tmem_load()\n"
        "                _helion_flash_rt.fbwd_ds_pairs_packed(tLDrdP, tLDrS, fbwd_ch * 32, fbwd_dlt_frg, 32)\n"
    )
    epilogue = _fbwd_epilogue_block(
        h=h,
        rows=128,
        d=d,
        epi_off=str(256 if h == 0 else "FBWD_DK_OFF"),
        epi_scale=(
            "_helion_flash_rt._scale_fragment_packed_f32x2(fbwd_reg, cutlass.Float32("
            + out_scale_expr
            + "))"
            if h == 1
            else "pass"
        ),
        kv_tile="fbwd_n_tile",
        io_dtype=io_dtype,
        dkv_wait="fbwd_nz_it % 2" if persistent else "0",
        v_empty_expr="storage.v_mbar_ptr.data_ptr() + 1",
        gm_expr=("_fbwd_mdVt" if h == 0 else "_fbwd_mdKt") + "[None, None, fbwd_bhk]",
    ).replace("FBWD_KV_SLOT", "0")
    if persistent and h == 1:
        # The dK staging (sdS) is free for the next tile's first dS once the
        # store has read it.
        tail_old = "        if warp_idx == 8:\n            cute.arch.cp_async_bulk_wait_group(0, read=True)\n"
        assert epilogue.count(tail_old) == 1, epilogue[-400:]
        epilogue = epilogue.replace(
            tail_old,
            tail_old
            + "            with cute.arch.elect_one():\n"
            + "                _helion_flash_rt.mbarrier_arrive(fbwd_sdsfree_ptr)\n",
        )
    zero_target = "_fbwd_mdV" if h == 0 else "_fbwd_mdK"
    zero_store = f"""        fbwd_zrow = fbwd_kv_base + fbwd_local_tidx
        if fbwd_zrow < {plan.kv_seq}:
            fbwd_zeros = cute.make_rmem_tensor(cute.make_layout({d}), {io_dtype})
            fbwd_zeros.fill({io_dtype}(0.0))
            fbwd_zdst = cute.make_tensor({zero_target}[None, None, fbwd_bhk][fbwd_zrow, None].iterator, cute.make_layout({d}))
            cute.autovec_copy(fbwd_zeros, fbwd_zdst)
"""
    pin_line = (
        "fbwd_m_tile_h = _helion_flash_rt.fbwd_pin_i32(fbwd_m_tile)"
        if hook_pin
        else "fbwd_m_tile_h = fbwd_m_tile"
    )
    # dS(i-1) in smem is read by the dQ and dK MMAs (dK is the later of the
    # two in the pipe); in the persistent walk the previous tile's dK
    # epilogue stages dK through the same buffer (sds_free from WG1).
    sds_guard = (
        """            if (fbwd_i == 0) & (fbwd_nz_it > 0):
                _helion_flash_rt.mbar_spin_wait(fbwd_sdsfree_ptr, (fbwd_nz_it - 1) & 1, 10000000)
            if (fbwd_i > 0) | (fbwd_nz_it > 0):
                _helion_flash_rt.mbar_spin_wait(fbwd_dk_done_ptr, fbwd_dqf_phase, 10000000)
                fbwd_dqf_phase ^= 1
"""
        if persistent
        else """            if fbwd_i > 0:
                _helion_flash_rt.mbar_spin_wait(fbwd_dk_done_ptr, fbwd_dqf_phase, 10000000)
                fbwd_dqf_phase ^= 1
"""
    )
    item_lines = _lines(item_block, 12 if per_step_item else 8)
    tile_lines = _lines(tile_block, 12)
    item_once = "" if per_step_item else item_lines
    item_each = item_lines if per_step_item else ""
    stage_lse, stage_delta = not lse_in_mma, not delta_in_mma

    def stage_lines(ind: str, dst_off: str, col: str) -> str:
        """cp.async staging of the next step's LSE / delta columns (threads
        0-63 the LSE, 64-127 the delta; a term carried by the MMA is skipped)."""
        lse_line = f"{ind}    _helion_flash_rt.cp_async_ca_4(fbwd_sLSE.iterator + ({dst_off} + fbwd_stat_j), _fbwd_mLSE.iterator + {col})\n"
        dlt_line = f"{ind}    _helion_flash_rt.cp_async_ca_4(fbwd_sDelta.iterator + ({dst_off} + fbwd_stat_j), _fbwd_mDelta.iterator + {col})\n"
        if stage_lse and stage_delta:
            body = f"{ind}if fbwd_local_tidx < 64:\n{lse_line}{ind}else:\n{dlt_line}"
        elif stage_lse:
            body = f"{ind}if fbwd_local_tidx < 64:\n{lse_line}"
        elif stage_delta:
            body = f"{ind}if fbwd_local_tidx >= 64:\n{dlt_line}"
        else:
            return ""
        return body + f"{ind}cute.arch.cp_async_commit_group()\n"

    stage_any = stage_lse or stage_delta
    stage_first = (
        (
            "        if fbwd_steps > 0:\n"
            "            fbwd_e0 = cutlass.Int32(fbwd_sSteps[fbwd_tbase])\n"
            "            fbwd_bhq0 = fbwd_bhk * "
            + str(plan.group)
            + " + ((fbwd_e0 >> 16) & 255)\n"
            f"            fbwd_stage_col0 = fbwd_bhq0 * {plan.q_seq} + cutlass.min((fbwd_e0 & 65535) * 128 + {64 * h} + fbwd_stat_j, cutlass.Int32({plan.q_seq - 1}))\n"
            + stage_lines(
                " " * 12, f"(fbwd_gstep % 2) * 128 + {64 * h}", "fbwd_stage_col0"
            )
        )
        if stage_any
        else ""
    )
    stage_next = (
        (
            "            if fbwd_i + 1 < fbwd_steps:\n"
            "                fbwd_e1 = cutlass.Int32(fbwd_sSteps[fbwd_tbase + fbwd_i + 1])\n"
            f"                fbwd_bhq1 = fbwd_bhk * {plan.group} + ((fbwd_e1 >> 16) & 255)\n"
            f"                fbwd_stage_col = fbwd_bhq1 * {plan.q_seq} + cutlass.min((fbwd_e1 & 65535) * 128 + {64 * h} + fbwd_stat_j, cutlass.Int32({plan.q_seq - 1}))\n"
            + stage_lines(" " * 16, f"128 - fbwd_par + {64 * h}", "fbwd_stage_col")
        )
        if stage_any
        else ""
    )
    stage_wait = "            cute.arch.cp_async_wait_group(0)\n" if stage_any else ""
    # dS^T packed into the dP columns of TMEM for the dK MMA (TS mode): the
    # 32 words of this half's bf16 pairs land at the P store's offsets, +128.
    ds_tmem_setup = (
        (
            f"        tDS_h = cute.make_tensor(fbwd_tmem_ptr + {128 + 64 * h}, fbwd_p_half_layout)\n"
            "        tSTtDS = fbwd_thr_st.partition_D(tDS_h)\n"
        )
        if ds_tmem
        else ""
    )
    ds_tmem_alloc = (
        "            tSTrDS = cute.make_rmem_tensor(tSTcP.shape, cutlass.Float32)\n"
        if ds_tmem
        else ""
    )
    ds_bf_alloc = (
        f"                fbwd_ds_bf = cute.make_tensor(cute.recast_ptr(tSTrDS.iterator, dtype={io_dtype}) + fbwd_ch * 32, tLDrdP.layout)\n"
        if ds_tmem
        else f"                fbwd_ds_bf = cute.make_rmem_tensor(tLDrdP.layout, {io_dtype})\n"
    )
    ds_tmem_store = (
        "            cute.copy(fbwd_tiled_st, tSTrDS, tSTtDS)\n"
        "            cute.arch.fence_view_async_tmem_store()\n"
        if ds_tmem
        else ""
    )
    setup = f"""
    if (warp_idx >= {warp_lo}) & (warp_idx < {warp_hi}):
        {_setmaxreg(regs_compute)}
        fbwd_s_shape = fbwd_sst.partition_shape_C((128, 128))
        tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
        fbwd_half_layout = cute.composition(tStS_frag.layout, cute.make_layout((128, 64)))
        tS_h = cute.make_tensor(fbwd_tmem_ptr + {64 * h}, fbwd_half_layout)
        tdP_h = cute.make_tensor(fbwd_tmem_ptr + {128 + 64 * h}, fbwd_half_layout)
        # Coordinates of this half's columns within the 128-wide tile (the
        # hooks read the Q position off them).
        cS = cute.domain_offset((0, {64 * h}), cute.make_identity_tensor((128, 128)))
        tScS = fbwd_sst.partition_C(cS)
        fbwd_cs_half_layout = cute.composition(tScS.layout, cute.make_layout((128, 64)))
        tScS_h = cute.make_tensor(tScS.iterator, fbwd_cs_half_layout)
        fbwd_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
        fbwd_tiled_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_ld_atom, tS_h)
        fbwd_thr_ld = fbwd_tiled_ld.get_slice(fbwd_local_tidx)
        tLDtS = fbwd_thr_ld.partition_S(tS_h)
        tLDtdP = fbwd_thr_ld.partition_S(tdP_h)
        tLDcS = fbwd_thr_ld.partition_D(tScS_h)
        fbwd_p_half_layout = cute.composition(tStS_frag.layout, cute.make_layout((128, 32)))
        tP_h = cute.make_tensor(fbwd_tmem_ptr + {64 * h}, fbwd_p_half_layout)
        fbwd_cp_half_layout = cute.composition(tScS.layout, cute.make_layout((128, 32)))
        tPcS_h = cute.make_tensor(tScS.iterator, fbwd_cp_half_layout)
        fbwd_st_atom = cute.make_copy_atom(cute_tcgen05_flash.St32x32bOp(cute_tcgen05_flash.Repetition(16)), cutlass.Float32)
        fbwd_tiled_st = cute_tcgen05_flash.make_tmem_copy(fbwd_st_atom, tP_h)
        fbwd_thr_st = fbwd_tiled_st.get_slice(fbwd_local_tidx)
        tSTtP = fbwd_thr_st.partition_D(tP_h)
        tSTcP = fbwd_thr_st.partition_S(tPcS_h)
{ds_tmem_setup}        fbwd_sdS_mnc = cute.make_tensor(fbwd_sdS_mn.iterator, cute.composition(fbwd_sdS_mn.layout, cute.make_layout((32, 2, 2, 128))))
        fbwd_crow = cutlass.Int32(tLDcS[0][0])
        fbwd_sdS_c0 = fbwd_sdS_mnc[None, 0, {h}, fbwd_crow]
        fbwd_sdS_c1 = fbwd_sdS_mnc[None, 1, {h}, fbwd_crow]
        fbwd_thr_row = cutlass.Int32(tLDcS[0][0])
        fbwd_phase = cutlass.Int32(0)
        fbwd_dqf_phase = cutlass.Int32(0)
        fbwd_gstep = cutlass.Int32(0)
        fbwd_scale = cutlass.Float32({scale_expr})
        fbwd_log2c = cutlass.Float32({plan.log2_factor!r})
        fbwd_neg_log2c = cutlass.Float32({-plan.log2_factor!r})
        fbwd_scale2 = cutlass.Float32(({scale_expr}) * {plan.log2_factor!r})
        fbwd_stat_j = fbwd_local_tidx % 64
"""
    tile_region = f"""        flash_bh = fbwd_bhk * {plan.group}
{item_once}{stage_first}        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
            fbwd_e = cutlass.Int32(fbwd_sSteps[fbwd_tbase + fbwd_i])
            fbwd_m_tile = fbwd_e & 65535
            fbwd_gi = (fbwd_e >> 16) & 255
            fbwd_masked = fbwd_e >> 24
            fbwd_bhq = fbwd_bhk * {plan.group} + fbwd_gi
            flash_bh = fbwd_bhq
{item_each}            fbwd_par = (fbwd_gstep % 2) * 128
            _helion_flash_rt.mbar_spin_wait(fbwd_s_full_ptr, fbwd_phase, 10000000)
            tLDrS = cute.make_rmem_tensor(tLDcS.shape, cutlass.Float32)
{spre_alloc}            cute.copy(fbwd_tiled_ld, tLDtS, tLDrS)
            cute.arch.fence_view_async_tmem_load()
{stage_wait}            _helion_flash_rt.named_barrier_wait_unaligned({3 + h}, 128)
            {pin_line}
{tile_lines}{stage_next}{p_passes}            tSTrP = cute.make_rmem_tensor(tSTcP.shape, cutlass.Float32)
            tSTrP_e = cute.make_tensor(cute.recast_ptr(tSTrP.iterator, dtype={io_dtype}), tLDrS.layout)
            tSTrP_e.store(tLDrS.load().to({io_dtype}))
            cute.copy(fbwd_tiled_st, tSTrP, tSTtP)
            cute.arch.fence_view_async_tmem_store()
            _helion_flash_rt.mbarrier_arrive(fbwd_p_full_ptr)
            _helion_flash_rt.mbar_spin_wait(fbwd_dp_full_ptr, fbwd_phase, 10000000)
{sds_guard}{ds_tmem_alloc}            for fbwd_ch in cutlass.range_constexpr(2):
                tLDrdP = cute.make_rmem_tensor(tLDcS[None, None, 0].shape, cutlass.Float32)
                cute.copy(fbwd_tiled_ld, tLDtdP[None, None, fbwd_ch], tLDrdP)
{ds_lines}{jac_lines}{ds_bf_alloc}                fbwd_ds_bf.store(tLDrdP.load().to({io_dtype}))
                fbwd_ds_flat = cute.make_tensor(fbwd_ds_bf.iterator, cute.make_layout(32))
                if fbwd_ch == 0:
                    cute.autovec_copy(fbwd_ds_flat, fbwd_sdS_c0)
                else:
                    cute.autovec_copy(fbwd_ds_flat, fbwd_sdS_c1)
{ds_tmem_store}            cute.arch.fence_view_async_shared()
            _helion_flash_rt.mbarrier_arrive(fbwd_ds_full_ptr)
            fbwd_phase ^= 1
            fbwd_gstep = fbwd_gstep + 1
        if fbwd_steps > 0:
{textwrap.indent(epilogue.rstrip(), "    ")}
        else:
{textwrap.indent(zero_store, "    ")}"""
    if persistent:
        tile_region = p_head + textwrap.indent(tile_region, "    ") + p_tail
    return (
        setup
        + tile_region
        + "        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)\n"
    )


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
)


def _reg_split(value: object) -> tuple[int, int, int]:
    text = str(value) if value else FBWD_REG_SPLITS[0]
    if text not in FBWD_REG_SPLITS:
        text = FBWD_REG_SPLITS[0]
    reduce_regs, compute_regs, other_regs = (int(part) for part in text.split("/"))
    assert reduce_regs + 2 * compute_regs + other_regs <= 512
    return reduce_regs, compute_regs, other_regs


def _setmaxreg(count: int) -> str:
    """The ``setmaxnreg`` call moving a warp from the 128-register launch
    budget to ``count`` (an increase or a decrease, never both)."""
    if count > 128:
        return f"cute.arch.setmaxregister_increase({count})"
    if count < 128:
        return f"cute.arch.setmaxregister_decrease({count})"
    return "pass"


_SM100_SMEM_CAPACITY = 232448
# Wrapper-built smem layouts of the statistics slices (``flex_aug_stages``).
_FLASH_BWD_FLEX_AUG_PARAMS = ("_fbwd_augl", "_fbwd_kaugl")


@dataclasses.dataclass(frozen=True)
class _SmemPlan:
    """The shared-memory plan of one flex backward kernel: the Q / dO ring
    depths, the dQ drain ring, and which per-column statistics terms ride the
    MMA (``aug_stages`` > 0 adds the statistics slices; see
    :func:`_fit_smem_budget`)."""

    q_stage: int
    do_stage: int
    dq_boxes: int
    aug_stages: int
    lse_in_mma: bool
    delta_in_mma: bool
    # Bytes of the flex hook's staged aux tables (``fbwd_aux_smem_layout``;
    # 0 = the gathers read global memory).
    aux_bytes: int = 0

    @property
    def stage_lse(self) -> bool:
        return not self.lse_in_mma

    @property
    def stage_delta(self) -> bool:
        return not self.delta_in_mma


def _fit_smem_budget(
    plan: FlexBwdPlan,
    *,
    stats_mma: bool,
    raw_score: bool,
    device: torch.device,
    persistent: bool = False,
    aux_smem: int = 0,
) -> _SmemPlan | None:
    """Fit the storage into the per-CTA SMEM capacity (None: nothing fits,
    the caller declines the fused path).

    Q / dO rings: 2 / 1 stages at head dim 128, 4 / 4 at 64 (FA4's). The dQ
    drain ring is 8 boxes (2 per reduce warp) unless the step table overflows
    the capacity, then 4 (one per warp; the drain is off the critical path).
    ``stats_mma`` (knob ``cute_flash_bwd_stats_mma``) moves the per-column
    LSE / delta subtractions into the S / dP MMAs through K-slice tiles
    (``flash_bwd_shared_storage(aug_stages=...)``): the LSE term only when no
    hook needs the raw score (``raw_score`` False), delta always. The slices
    cost one Q-ring stage of 4 KB each plus 8 KB of constants; when they do
    not fit, the dO ring gives up stages down to 2, then the mode is dropped.
    ``persistent`` sizes the step table for its double buffer (two tables
    plus the two length words). ``aux_smem`` (knob ``cute_flash_bwd_aux_smem``,
    bytes) stages the hook's gathered aux tables in shared memory
    (``fbwd_aux_smem_layout``); the region is dropped first when the storage
    does not fit.
    """
    import cutlass

    from ._flash_runtime import flash_bwd_shared_storage
    from .tcgen05_config import CuteTcgen05Config

    capacity = (
        CuteTcgen05Config.per_cta_smem_capacity_bytes(device) or _SM100_SMEM_CAPACITY
    )
    dtype = cutlass.BFloat16 if plan.io_dtype == torch.bfloat16 else cutlass.Float16
    d = plan.head_dim
    q_stage = 2 if d == 128 else 4
    do_stage_max = 1 if d == 128 else 4

    steps = max(plan.max_steps, 1)
    table_entries = 2 * steps + 2 if persistent else steps
    aux_total = sum(
        entry.padded_bytes for entry in fbwd_aux_smem_layout(plan, aux_smem)
    )

    def size(
        do_stage: int, dq_boxes: int, aug: int, lse: bool, delta: bool, aux: int
    ) -> int:
        return flash_bwd_shared_storage(
            d,
            q_stage,
            do_stage,
            1,
            dtype,
            table_entries,
            dq_boxes,
            aug,
            lse,
            delta,
            flex_aux_bytes=aux,
        ).size_in_bytes()

    for aux in sorted({aux_total, 0}, reverse=True):
        if stats_mma:
            lse_in_mma = not raw_score
            for do_stage in range(do_stage_max, max(min(do_stage_max, 2) - 1, 0), -1):
                for dq_boxes in (8, 4):
                    if (
                        size(do_stage, dq_boxes, q_stage, not lse_in_mma, False, aux)
                        <= capacity
                    ):
                        return _SmemPlan(
                            q_stage, do_stage, dq_boxes, q_stage, lse_in_mma, True, aux
                        )
        for dq_boxes in (8, 4):
            if size(do_stage_max, dq_boxes, 0, True, True, aux) <= capacity:
                return _SmemPlan(q_stage, do_stage_max, dq_boxes, 0, False, False, aux)
    return None


# dQ fp32 working set (bytes) one batch-head group of CTAs may keep hot in L2:
# the TMA bulk reduce-adds of the dQ accumulator stay L2 hits while the running
# CTAs cover few enough (batch, head) rows (B200: 126 MB L2 shared with the
# Q / dO tiles of the same heads).
_FBWD_TILE_GROUP_L2_BYTES = 64 << 20


def fbwd_tile_group_choices(plan: FlexBwdPlan) -> tuple[int, ...]:
    """Legal ``cute_flash_bwd_tile_group`` values, the L2-sized group first.

    ``0``: KV tile fastest (the ``num_kv_tiles`` CTAs of one batch / kv head
    are consecutive). ``G >= 2``: the grid is walked in groups of ``G`` batch
    / kv heads; inside a group the KV tile is the slow index (all tile-0 CTAs
    of the group first, then tile 1, ...): heaviest first for causal masks
    (the hardware dispatches blocks in index order, so this is list
    scheduling by non-increasing weight) while the dQ rows the group's CTAs
    reduce into stay inside the L2 budget. ``G`` runs over the powers of two
    dividing the batch-head count; the default is the widest group whose dQ
    rows fit ``_FBWD_TILE_GROUP_L2_BYTES``.
    """
    bhk = plan.batch * plan.kv_heads
    legal = [g for g in (1 << k for k in range(1, bhk.bit_length())) if bhk % g == 0]
    dq_bytes_per_bhk = plan.group * plan.q_seq * plan.head_dim * 4
    fitting = [g for g in legal if g * dq_bytes_per_bhk <= _FBWD_TILE_GROUP_L2_BYTES]
    default = max(fitting) if fitting else 0
    return (default, *(g for g in (0, *legal) if g != default))


def _tile_decode(plan: FlexBwdPlan, tile_group: int) -> str:
    """Map the launch's block index to ``(fbwd_bhk, fbwd_n_tile)``; see
    :func:`fbwd_tile_group_choices` (an illegal group falls back to ``0``)."""
    n, bhk = plan.num_kv_tiles, plan.batch * plan.kv_heads
    if tile_group < 2 or bhk % tile_group != 0:
        return (
            f"    fbwd_bhk = fbwd_tile_id // {n}\n"
            f"    fbwd_n_tile = fbwd_tile_id % {n}\n"
        )
    per_group = tile_group * n
    return (
        f"    fbwd_tgrp = fbwd_tile_id // {per_group}\n"
        f"    fbwd_trem = fbwd_tile_id % {per_group}\n"
        f"    fbwd_n_tile = fbwd_trem // {tile_group}\n"
        f"    fbwd_bhk = fbwd_tgrp * {tile_group} + fbwd_trem % {tile_group}\n"
    )


def _step_table_fill(plan: FlexBwdPlan) -> str:
    """The step-table count loads (unindented lines, 4-space base).

    Every thread loads the full / partial list lengths of the group's q heads
    for this KV tile and derives ``fbwd_steps``; :func:`_step_table_loop`
    then fills the table: the lists of the group's q heads in order (full
    list, then partial list of head 0, then head 1, ...), each list position
    expanded to ``QB // 128`` Q tiles.
    """
    walk = plan.walk
    sub = walk.q_block // 128
    lines = [
        f"fbwd_tb = fbwd_b % {walk.table_batch}"
        if walk.table_batch > 1
        else "fbwd_tb = cutlass.Int32(0)",
        "fbwd_total = cutlass.Int32(0)",
    ]
    for gi in range(plan.group):
        th = (
            f"(fbwd_hk * {plan.group} + {gi}) % {walk.table_heads}"
            if walk.table_heads > 1
            else "cutlass.Int32(0)"
        )
        lines += [
            f"fbwd_th{gi} = {th}",
            f"fbwd_cf{gi} = cutlass.Int32(_fbwd_full_cnt[fbwd_tb, fbwd_th{gi}, fbwd_n_tile])",
            f"fbwd_cp{gi} = cutlass.Int32(_fbwd_part_cnt[fbwd_tb, fbwd_th{gi}, fbwd_n_tile])",
            f"fbwd_total = fbwd_total + fbwd_cf{gi} + fbwd_cp{gi}",
        ]
    lines.append(f"fbwd_steps = fbwd_total * {sub}")
    return "\n".join("    " + line for line in lines) + "\n"


def _step_table_loop(
    plan: FlexBwdPlan,
    *,
    thread: str = "tidx",
    stride: int = 512,
    table: str = "fbwd_sSteps",
    index_base: str = "",
) -> str:
    """The fill loop of :func:`_step_table_fill` (every thread owns the
    entries ``thread, thread + stride, ...``); runs after the count loads so
    the TMA warp can issue the K/V loads in between. The persistent walk's
    TMA warp fills the double-buffered table alone (``thread`` = lane,
    ``stride`` 32, ``table`` the tile's buffer)."""
    sub = plan.walk.q_block // 128
    dst = table  # the loop below rebinds ``table`` to the list tensors
    lines = [
        f"for fbwd_fe in cutlass.range({thread}, fbwd_steps, {stride}, unroll=1):",
        f"    fbwd_rem = fbwd_fe // {sub}",
        f"    fbwd_sub = fbwd_fe % {sub}",
        "    fbwd_entry = cutlass.Int32(0)",
    ]
    for gi in range(plan.group):
        for masked, cnt, table in (
            (0, f"fbwd_cf{gi}", "_fbwd_full_idx"),
            (1, f"fbwd_cp{gi}", "_fbwd_part_idx"),
        ):
            lines += [
                f"    if (fbwd_rem >= 0) & (fbwd_rem < {cnt}):",
                f"        fbwd_qblk = cutlass.Int32({table}[fbwd_tb, fbwd_th{gi}, fbwd_n_tile, fbwd_rem])",
                f"        fbwd_entry = (fbwd_qblk * {sub} + fbwd_sub) | ({gi} << 16) | ({masked} << 24)",
                f"    fbwd_rem = fbwd_rem - {cnt}",
            ]
    lines.append(f"    {dst}[{index_base}fbwd_fe] = fbwd_entry")
    return "\n".join("    " + line for line in lines) + "\n"


def emit_flash_bwd_flex_device_body(
    *,
    plan: FlexBwdPlan,
    io_dtype: str,
    scale_expr: str,
    q_stage: int,
    do_stage: int,
    exp2_f32: bool,
    mask_form: str,
    bindings: list[str],
    device_function: DeviceFunction,
    reg_split: str | None = None,
    hook_cols: int = 32,
    hook_pin: bool = True,
    bias_fold: bool = True,
    tile_group: int = 0,
    smem: _SmemPlan | None = None,
    exp2_emu: int = 0,
    persistent: bool = False,
    total_tiles: int = 0,
    ds_tmem: bool = False,
) -> str:
    """The flex backward device body (1-CTA, 512 threads); see the module doc.

    ``persistent``: ``min(total_tiles, SMs)`` CTAs walk the tiles through the
    CLC dynamic scheduler (warp 15 steals not-yet-launched tiles in launch
    order, so the ``cute_flash_bwd_tile_group`` order is kept). The TMA warp
    builds each tile's step table into a double buffer (``table_full`` /
    ``table_empty`` barriers), every role loops over tiles, the per-tile
    barriers (dv_done / dkv_done) run on the non-empty-tile parity, the dK
    staging is handed back to the dS pass through ``sds_free``, and K / Q of
    the next tile load while the current epilogue drains.
    """
    from .cute_flash_bwd import _FBWD_2CTA_DS_OFFSETS

    d = plan.head_dim
    dk_off = 256 + (64 if d == 64 else 128)
    dq_off = 384 if d == 64 else 128
    kv_stage = 1
    n_chunks = d // 32
    assert hook_cols in (8, 16, 32)
    if smem is None:
        smem = _SmemPlan(q_stage, do_stage, 8, 0, False, False)
    assert (smem.q_stage, smem.do_stage) == (q_stage, do_stage)
    dq_boxes = smem.dq_boxes
    bias = additive_bias_region(plan) if bias_fold else None
    # The staged aux tables: the budget the fitter kept (its layout is the
    # renderer's) and the CTA-wide fill before the setup sync.
    aux_layout = fbwd_aux_smem_layout(plan, smem.aux_bytes) if smem.aux_bytes else []
    assert sum(entry.padded_bytes for entry in aux_layout) == smem.aux_bytes
    aux_fill = ""
    if aux_layout:
        from .cute_flash import _flash_flex_aux_smem_fill

        aux_fill = "".join(
            f"    {line}\n"
            for line in _flash_flex_aux_smem_fill(flex_bwd_hook_plan(plan), aux_layout)
        )
    _hooks, plain, masked, jac = _render_hooks(
        plan,
        device_function=device_function,
        mask_form=mask_form,
        hook_cols=hook_cols,
        bias=bias,
        aux_smem_budget=smem.aux_bytes,
    )
    regs_reduce, regs_compute, regs_other = _reg_split(reg_split)
    binding_lines = "".join(f"    {line}\n" for line in bindings)
    max_steps = max(plan.max_steps, 1)
    table_entries = 2 * max_steps + 2 if persistent else max_steps

    def decode_lines(ind: str) -> str:
        text = _tile_decode(plan, tile_group) + (
            f"    fbwd_b = fbwd_bhk // {plan.kv_heads}\n"
            f"    fbwd_hk = fbwd_bhk % {plan.kv_heads}\n"
            "    fbwd_kv_base = fbwd_n_tile * 128\n"
        )
        return "".join(ind + line[4:] for line in text.splitlines(keepends=True))

    def p_head(kind: str, indent: int) -> str:
        """Persistent tile loop head of one warp role (``kind``: ``tma`` builds
        the tile's step table, ``cons`` reads it, ``mma`` reads its length)."""
        if not persistent:
            return ""
        ind = " " * indent
        inner = ind + "    "
        lines = [
            f"{ind}fbwd_work = fbwd_clc_hw.initial_work_tile_info()\n",
            f"{ind}fbwd_tile_it = cutlass.Int32(0)\n",
            f"{ind}fbwd_nz_it = cutlass.Int32(0)\n",
            f"{ind}while fbwd_work.is_valid_tile:\n",
            f"{inner}fbwd_tile_id = cutlass.Int32(fbwd_work.tile_idx[0])\n",
            decode_lines(inner),
            f"{inner}fbwd_tbuf = fbwd_tile_it % 2\n",
            f"{inner}fbwd_tbase = fbwd_tbuf * {max_steps}\n",
        ]
        if kind == "tma":
            counts = _step_table_fill(plan)
            fill = _step_table_loop(
                plan, thread="fbwd_lane", stride=32, index_base="fbwd_tbase + "
            )
            lines += [
                "".join(inner + line[4:] for line in counts.splitlines(keepends=True)),
                f"{inner}_helion_flash_rt.mbar_spin_wait(fbwd_tempty_ptr + fbwd_tbuf, ((fbwd_tile_it // 2) & 1) ^ 1, 10000000)\n",
                "".join(inner + line[4:] for line in fill.splitlines(keepends=True)),
                f"{inner}fbwd_sSteps[{2 * max_steps} + fbwd_tbuf] = fbwd_steps\n",
                f"{inner}cute.arch.sync_warp()\n",
                f"{inner}with cute.arch.elect_one():\n",
                f"{inner}    _helion_flash_rt.mbarrier_arrive(fbwd_tfull_ptr + fbwd_tbuf)\n",
            ]
        else:
            lines += [
                f"{inner}_helion_flash_rt.mbar_spin_wait(fbwd_tfull_ptr + fbwd_tbuf, (fbwd_tile_it // 2) & 1, 10000000)\n",
                f"{inner}fbwd_steps = cutlass.Int32(fbwd_sSteps[{2 * max_steps} + fbwd_tbuf])\n",
            ]
        return "".join(lines)

    def p_tail(kind: str, indent: int) -> str:
        if not persistent:
            return ""
        from .cute_flash_bwd import _FBWD_CLC_CONSUME

        ind = " " * indent
        inner = ind + "    "
        lines = []
        if kind in ("cons", "mma"):
            # Every reader of the tile's table buffer (the 12 consumer warps
            # and the MMA warp, which reads its length word) releases it.
            lines += [
                f"{inner}cute.arch.sync_warp()\n",
                f"{inner}with cute.arch.elect_one():\n",
                f"{inner}    _helion_flash_rt.mbarrier_arrive(fbwd_tempty_ptr + fbwd_tbuf)\n",
            ]
        lines += [
            _FBWD_CLC_CONSUME.format(ind=inner),
            f"{inner}fbwd_tile_it = fbwd_tile_it + 1\n",
            f"{inner}if fbwd_steps > 0:\n",
            f"{inner}    fbwd_nz_it = fbwd_nz_it + 1\n",
        ]
        return "".join(lines)

    compute_blocks = "".join(
        _compute_block(
            h=h,
            plan=plan,
            io_dtype=io_dtype,
            scale_expr=scale_expr,
            out_scale_expr=scale_expr,
            exp2_f32=exp2_f32,
            plain=plain,
            masked=masked,
            jac=jac,
            regs_compute=regs_compute,
            hook_cols=hook_cols,
            hook_pin=hook_pin,
            bias_fold=bias is not None,
            lse_in_mma=smem.lse_in_mma,
            delta_in_mma=smem.delta_in_mma,
            exp2_emu=exp2_emu,
            persistent=persistent,
            p_head=p_head("cons", 8),
            p_tail=p_tail("cons", 8),
            ds_tmem=ds_tmem,
        ).replace("FBWD_DK_OFF", str(dk_off))
        for h in (0, 1)
    )
    if d == 128:
        dq_empty_wait_g2 = """
            _helion_flash_rt.mbar_spin_wait(fbwd_dq_empty_ptr, fbwd_dqe_phase, 10000000)
            fbwd_dqe_phase ^= 1"""
        dq_empty_wait_g5 = ""
        dq_empty_wait_tail = ""
    else:
        dq_empty_wait_g2 = ""
        g5_cond = "(fbwd_i > 0) | (fbwd_nz_it > 0)" if persistent else "fbwd_i > 0"
        tail_cond = (
            "(fbwd_steps > 1) | (fbwd_nz_it > 0)" if persistent else "fbwd_steps > 1"
        )
        dq_empty_wait_g5 = f"""
            if {g5_cond}:
                _helion_flash_rt.mbar_spin_wait(fbwd_dq_empty_ptr, fbwd_dqe_phase, 10000000)
                fbwd_dqe_phase ^= 1"""
        dq_empty_wait_tail = f"""
        if {tail_cond}:
            _helion_flash_rt.mbar_spin_wait(fbwd_dq_empty_ptr, fbwd_dqe_phase, 10000000)
            fbwd_dqe_phase ^= 1"""
    reduce_body = f"""            _helion_flash_rt.mbar_spin_wait(fbwd_dq_full_ptr, fbwd_phase, 10000000)
            fbwd_phase ^= 1
            fbwd_my_row = cutlass.Int32(tRDcDQ[0][0]) % 32
            fbwd_gdq_row = fbwd_m_tile * 4 + warp_idx
            fbwd_gdq = cute.local_tile(_fbwd_mDQt[None, None, fbwd_bhq], (32, 32), (None, None))
            tRDrDQ = cute.make_rmem_tensor(tRDcDQ.shape, cutlass.Float32)
            cute.copy(fbwd_tiled_dq_ld, tRDtDQ, tRDrDQ)
            cute.arch.fence_view_async_tmem_load()
            _helion_flash_rt.mbarrier_arrive(fbwd_dq_empty_ptr)
            _helion_flash_rt._scale_fragment_packed_f32x2(tRDrDQ, fbwd_out_scale)
            for fbwd_c in cutlass.range_constexpr({n_chunks}):
                fbwd_dq_flat = cute.make_tensor(tRDrDQ[None, fbwd_c, 0, 0].iterator, cute.make_layout(32))
                fbwd_sdq_box = fbwd_sdq[None, None, {"warp_idx * 2 + fbwd_c % 2" if dq_boxes == 8 else "warp_idx"}]
                cute.arch.cp_async_bulk_wait_group({1 if dq_boxes == 8 else 0}, read=True)
                cute.arch.sync_warp()
                cute.autovec_copy(fbwd_dq_flat, fbwd_sdq_box[fbwd_my_row, None])
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                fbwd_gdq_box = fbwd_gdq[None, None, fbwd_gdq_row, fbwd_c]
                tDQsDQ, tDQgDQ = cute_cpasync_flash.tma_partition(_fbwd_tma_dq, 0, cute.make_layout(1), cute.group_modes(fbwd_sdq_box, 0, 2), cute.group_modes(fbwd_gdq_box, 0, 2))
                cute.copy(_fbwd_tma_dq, tDQsDQ, tDQgDQ)
                cute.arch.cp_async_bulk_commit_group()"""
    step_decode = f"""            fbwd_e = cutlass.Int32(fbwd_sSteps[fbwd_tbase + fbwd_i])
            fbwd_m_tile = fbwd_e & 65535
            fbwd_bhq = fbwd_bhk * {plan.group} + ((fbwd_e >> 16) & 255)"""
    aug = smem.aug_stages > 0
    stat_views = "".join(
        f"    {name} = storage.{field}.get_tensor(cute.make_layout(2 * 128))\n"
        for name, field, on in (
            ("fbwd_sLSE", "sLSE", smem.stage_lse),
            ("fbwd_sDelta", "sDelta", smem.stage_delta),
        )
        if on
    )
    aug_views = (
        f"""    fbwd_sAug = storage.sAug.get_tensor(_fbwd_augl.outer, swizzle=_fbwd_augl.inner)
    fbwd_sKaug = storage.sKaug.get_tensor(_fbwd_kaugl.outer, swizzle=_fbwd_kaugl.inner)
    fbwd_sVaug = storage.sVaug.get_tensor(_fbwd_kaugl.outer, swizzle=_fbwd_kaugl.inner)
    fbwd_sAug_nks = cute.make_tensor(fbwd_sAug.iterator, cute.composition(fbwd_sAug.layout, cute.make_layout((128, 16, {smem.aug_stages}))))
    fbwd_sKaug_mk = cute.make_tensor(fbwd_sKaug.iterator, cute.composition(fbwd_sKaug.layout, cute.make_layout((128, 16))))
    fbwd_sVaug_mk = cute.make_tensor(fbwd_sVaug.iterator, cute.composition(fbwd_sVaug.layout, cute.make_layout((128, 16))))
    fbwd_aug_full_ptr = storage.aug_full_mbar.data_ptr()
"""
        if aug
        else ""
    )
    aug_mbar_init = (
        "".join(
            f"        cute.arch.mbarrier_init(fbwd_aug_full_ptr + {stage}, 1)\n"
            for stage in range(smem.aug_stages)
        )
        if aug
        else ""
    )
    # The constant K / V slices: -1 in the LSE columns (0-2) of the K slice
    # and the delta columns (8-10) of the V slice, zero elsewhere; threads
    # 0-127 / 128-255 each write one row, made visible to the MMA (async
    # proxy) by the fence before the table sync.
    aug_const_fill = (
        f"""    if tidx < 256:
        fbwd_crow16 = cute.make_rmem_tensor(cute.make_layout(16), {io_dtype})
        fbwd_crow16.fill({io_dtype}(0.0))
        if tidx < 128:
            fbwd_crow16[0] = {io_dtype}(-1.0)
            fbwd_crow16[1] = {io_dtype}(-1.0)
            fbwd_crow16[2] = {io_dtype}(-1.0)
            cute.autovec_copy(fbwd_crow16, fbwd_sKaug_mk[tidx, None])
        else:
            fbwd_crow16[8] = {io_dtype}(-1.0)
            fbwd_crow16[9] = {io_dtype}(-1.0)
            fbwd_crow16[10] = {io_dtype}(-1.0)
            cute.autovec_copy(fbwd_crow16, fbwd_sVaug_mk[tidx - 128, None])
        cute.arch.fence_view_async_shared()
"""
        if aug
        else ""
    )

    def aug_fill(ind: int) -> str:
        """The TMA warp writes the statistics slice of the Q stage it just
        issued (lane l owns rows 4l..4l+3 of the Q tile), then signals the
        stage's aug_full barrier (one elected arrival after a warp sync)."""
        if not aug:
            return ""
        pad = " " * ind
        lse_src = (
            "cutlass.Float32(_fbwd_mLSE[fbwd_augcol])"
            if plan.lse_prescaled
            else "_helion_flash_rt.fbwd_lse_scalar_to_log2(cutlass.Float32(_fbwd_mLSE[fbwd_augcol]), fbwd_log2c_tma)"
        )
        return (
            f"{pad}fbwd_augrow16 = cute.make_rmem_tensor(cute.make_layout(16), {io_dtype})\n"
            f"{pad}for fbwd_ar in cutlass.range_constexpr(4):\n"
            f"{pad}    fbwd_augq = fbwd_lane * 4 + fbwd_ar\n"
            f"{pad}    fbwd_augcol = fbwd_bhq * {plan.q_seq} + cutlass.min(fbwd_m_tile * 128 + fbwd_augq, cutlass.Int32({plan.q_seq - 1}))\n"
            f"{pad}    fbwd_auglse = {lse_src} * fbwd_inv_scale2\n"
            f"{pad}    fbwd_augdlt = cutlass.Float32(_fbwd_mDelta[fbwd_augcol])\n"
            f"{pad}    _helion_flash_rt.fbwd_aug_row_fill(fbwd_augrow16, fbwd_auglse, fbwd_augdlt, {smem.lse_in_mma})\n"
            f"{pad}    cute.autovec_copy(fbwd_augrow16, fbwd_sAug_nks[fbwd_augq, None, fbwd_qe.index])\n"
            f"{pad}cute.arch.fence_view_async_shared()\n"
            f"{pad}cute.arch.sync_warp()\n"
            f"{pad}with cute.arch.elect_one():\n"
            f"{pad}    _helion_flash_rt.mbarrier_arrive(fbwd_aug_full_ptr + fbwd_qe.index)\n"
        )

    aug_fill_first = aug_fill(12)
    aug_fill_loop = aug_fill(16)
    aug_mma_setup = (
        """            tSrKaug = fbwd_sst.make_fragment_A(fbwd_sKaug)
            tSrVaug = fbwd_sst.make_fragment_A(fbwd_sVaug)
            tSrAug = fbwd_sst.make_fragment_B(fbwd_sAug)
            fbwd_kaug_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sKaug, _helion_flash_ptx.Major.K)
            fbwd_aug_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sAug, _helion_flash_ptx.Major.K)
            fbwd_st_kaug = _helion_flash_ptx.make_smem_desc_start_addr(fbwd_sKaug[None, None, None, 0].iterator)
            fbwd_st_vaug = _helion_flash_ptx.make_smem_desc_start_addr(fbwd_sVaug[None, None, None, 0].iterator)
"""
        if aug
        else ""
    )

    def aug_s_mma(handle: str, ind: int) -> str:
        """After the S MMA of a step: wait for the step's statistics slice,
        then the K-slice MMA accumulating ``-lse2 / c`` into S (when the LSE
        rides the MMA; the wait alone when only delta does)."""
        if not aug:
            return ""
        pad = " " * ind
        lines = [
            f"{pad}_helion_flash_rt.mbar_spin_wait(fbwd_aug_full_ptr + {handle}.index, fbwd_aug_phase, 10000000)",
            f"{pad}if {handle}.index == {smem.aug_stages - 1}:",
            f"{pad}    fbwd_aug_phase ^= 1",
        ]
        if smem.lse_in_mma:
            lines.append(
                f'{pad}_helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_s_addr, fbwd_st_kaug, fbwd_kaug_base, tSrKaug[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(fbwd_sAug[None, None, None, {handle}.index].iterator), fbwd_aug_base, tSrAug[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=False)'
            )
        return "\n".join(lines) + "\n"

    def aug_dp_mma(handle: str, ind: int) -> str:
        """After the dP MMA of a step: the K-slice MMA accumulating ``-delta``
        into dP (the slice was awaited at the S site)."""
        if not aug or not smem.delta_in_mma:
            return ""
        pad = " " * ind
        return f'{pad}_helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dp_addr, fbwd_st_vaug, fbwd_kaug_base, tSrVaug[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(fbwd_sAug[None, None, None, {handle}.index].iterator), fbwd_aug_base, tSrAug[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=False)\n'

    tma_scalars = (
        f"""        fbwd_inv_scale2 = cutlass.Float32(1.0) / cutlass.Float32(({scale_expr}) * {plan.log2_factor!r})
        fbwd_log2c_tma = cutlass.Float32({plan.log2_factor!r})
"""
        if aug
        else ""
    )
    if persistent:
        from .cute_flash_bwd import _FBWD_CLC_CONSUME
        from .cute_flash_bwd import _fbwd_clc_consume_only
        from .cute_flash_bwd import _fbwd_clc_setup

        clc_setup = _fbwd_clc_setup(total_tiles, 1, "cute.make_layout((1, 1, 1, 1))")
        prologue_k_fill_aug = (
            f"{aug_const_fill}    cute.arch.sync_threads()\n" if aug else ""
        )
        # Scheduler warp: the next tile is requested once this tile's loads
        # are all issued (tile_issued, every tile), so the response latency
        # hides behind the last steps and the epilogue.
        clc_warps = (
            "    if warp_idx == 14:\n"
            + _fbwd_clc_consume_only(" " * 8)
            + "    if warp_idx == 15:\n"
            "        fbwd_clc_pstate = cutlass_pipeline_flash.make_pipeline_state(cutlass_pipeline_flash.PipelineUserType.Producer, 2)\n"
            "        fbwd_clc_tile_it = cutlass.Int32(0)\n"
            "        fbwd_work = fbwd_clc_hw.initial_work_tile_info()\n"
            "        while fbwd_work.is_valid_tile:\n"
            "            _helion_flash_rt.mbar_spin_wait(fbwd_tissued_ptr, fbwd_clc_tile_it & 1, 10000000)\n"
            "            fbwd_clc_tile_it = fbwd_clc_tile_it + 1\n"
            "            fbwd_clc_pipe.producer_acquire(fbwd_clc_pstate)\n"
            "            fbwd_clc_mbar = fbwd_clc_pipe.producer_get_barrier(fbwd_clc_pstate)\n"
            "            fbwd_clc_rptr = (fbwd_clc_base + fbwd_clc_pstate.index * cutlass.Int32(4)).align(16)\n"
            "            with cute.arch.elect_one():\n"
            "                cute.arch.issue_clc_query(fbwd_clc_mbar, fbwd_clc_rptr)\n"
            "            fbwd_clc_pstate.advance()\n"
            + _FBWD_CLC_CONSUME.format(ind=" " * 12)
            + "        fbwd_clc_pipe.producer_tail(fbwd_clc_pstate)\n"
        )
        flex_mbar_init = (
            "        cute.arch.mbarrier_init(fbwd_tfull_ptr, 1)\n"
            "        cute.arch.mbarrier_init(fbwd_tfull_ptr + 1, 1)\n"
            "        cute.arch.mbarrier_init(fbwd_tempty_ptr, 13)\n"
            "        cute.arch.mbarrier_init(fbwd_tempty_ptr + 1, 13)\n"
            "        cute.arch.mbarrier_init(fbwd_sdsfree_ptr, 1)\n"
            "        cute.arch.mbarrier_init(fbwd_tissued_ptr, 1)\n"
        )
        prologue_table = ""
        prologue_k_fill = ""
        tab_alias = ""
        tma_k_issue = (
            "            fbwd_ke = fbwd_k_prod.acquire_and_advance()\n"
            "            cute.copy(_fbwd_tma_k, tKgK_kdl[None, None, 0, fbwd_bhk][None, fbwd_n_tile], tKsK[None, fbwd_ke.index], tma_bar_ptr=fbwd_ke.barrier)\n"
        )
        tma_issued = (
            "        cute.arch.sync_warp()\n"
            "        with cute.arch.elect_one():\n"
            "            _helion_flash_rt.mbarrier_arrive(fbwd_tissued_ptr)\n"
        )
        mma_dq_prologue = (
            """
            if fbwd_nz_it > 0:
                _helion_flash_rt.mbar_spin_wait(fbwd_dq_empty_ptr, fbwd_dqe_phase, 10000000)
                fbwd_dqe_phase ^= 1"""
            if d == 128
            else ""
        )
    else:
        clc_setup = ""
        clc_warps = ""
        flex_mbar_init = ""
        prologue_table = _step_table_fill(plan)
        prologue_k_fill = (
            f"{aug_const_fill}    if warp_idx == 13:\n"
            "        # The K load needs only the tile coordinates: issue it before the\n"
            "        # step-table fill so it lands while the table is built. The single\n"
            "        # K stage is never recycled, so its producer state ends here (no tail).\n"
            "        if fbwd_steps > 0:\n"
            "            fbwd_ke = fbwd_k_prod.acquire_and_advance()\n"
            "            cute.copy(_fbwd_tma_k, tKgK_kdl[None, None, 0, fbwd_bhk][None, fbwd_n_tile], tKsK[None, fbwd_ke.index], tma_bar_ptr=fbwd_ke.barrier)\n"
            + _step_table_loop(plan)
            + "    cute.arch.sync_threads()\n"
        )
        tab_alias = "    fbwd_tbase = cutlass.Int32(0)\n"
        prologue_k_fill_aug = ""
        tma_k_issue = ""
        tma_issued = ""
        mma_dq_prologue = ""
    tma_region = f"""        if fbwd_steps > 0:
{tma_k_issue}            # First step's Q ahead of V (S = K Q^T is the first MMA; dP = V dO^T follows).
            fbwd_e = cutlass.Int32(fbwd_sSteps[fbwd_tbase])
            fbwd_m_tile = fbwd_e & 65535
            fbwd_bhq = fbwd_bhk * {plan.group} + (fbwd_e >> 16 & 255)
            tQgQ = tQgQ_qdl[None, None, 0, fbwd_bhq]
            tDOgDO = tDOgDO_qdl[None, None, 0, fbwd_bhq]
            fbwd_qe = fbwd_q_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_q, tQgQ[None, fbwd_m_tile], tQsQ[None, fbwd_qe.index], tma_bar_ptr=fbwd_qe.barrier)
{aug_fill_first}            tVgV = tVgV_kdl[None, None, 0, fbwd_bhk]
            fbwd_ve = fbwd_v_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_v, tVgV[None, fbwd_n_tile], tVsV[None, fbwd_ve.index], tma_bar_ptr=fbwd_ve.barrier)
            fbwd_doe = fbwd_do_prod.acquire_and_advance()
            cute.copy(_fbwd_tma_do, tDOgDO[None, fbwd_m_tile], tDOsDO[None, fbwd_doe.index], tma_bar_ptr=fbwd_doe.barrier)
            for fbwd_i in cutlass.range(1, fbwd_steps, unroll=1):
{textwrap.indent(step_decode, "    ")}
                tQgQ = tQgQ_qdl[None, None, 0, fbwd_bhq]
                tDOgDO = tDOgDO_qdl[None, None, 0, fbwd_bhq]
                fbwd_qe = fbwd_q_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_q, tQgQ[None, fbwd_m_tile], tQsQ[None, fbwd_qe.index], tma_bar_ptr=fbwd_qe.barrier)
{aug_fill_loop}                fbwd_doe = fbwd_do_prod.acquire_and_advance()
                cute.copy(_fbwd_tma_do, tDOgDO[None, fbwd_m_tile], tDOsDO[None, fbwd_doe.index], tma_bar_ptr=fbwd_doe.barrier)
"""
    dk_gemm_ss = (
        "_helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dk_addr, fbwd_st_dsnk, fbwd_dsnk_base, "
        "tKSrDS[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr("
        "sQt[None, None, None, fbwd_qe.index].iterator), fbwd_qt_base, "
        'tKSrQt[None, None, None, 0].layout, "helion_fbwd_dsk_idesc", zero_init=fbwd_dk_zero)'
    )
    # TS mode: A = dS^T from the dP columns of TMEM (packed by the dS pass),
    # B = Q^T from smem, with the dV MMA's atom and instruction descriptor
    # (same shape, dtypes and B major).
    dk_gemm_ts = (
        "_helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dk_addr, fbwd_dp_addr, "
        "_helion_flash_ptx.make_smem_desc_start_addr(sQt[None, None, None, fbwd_qe.index].iterator), "
        "fbwd_qt_base, tTSrDS[None, None, None, 0].layout, tTSrQt[None, None, None, 0].layout, "
        f'"helion_fbwd_ts_idesc", zero_init=fbwd_dk_zero, a_offsets={_FBWD_2CTA_DS_OFFSETS})'
    )
    dk_gemm = dk_gemm_ts if ds_tmem else dk_gemm_ss
    ds_tmem_mma_setup = (
        (
            "            tDS = cute.make_tensor(tdP_t.iterator, _fbwd_ptl.outer)\n"
            "            tTSrDS = fbwd_tst.make_fragment_A(tDS)\n"
            "            tTSrQt = fbwd_tst.make_fragment_B(sQt)\n"
        )
        if ds_tmem
        else ""
    )
    mma_region = f"""        if fbwd_steps > 0:
            tS_t = cute.make_tensor(fbwd_tmem_ptr, tStS_frag.layout)
            tdP_t = cute.make_tensor(fbwd_tmem_ptr + 128, tStS_frag.layout)
            tDV_t = cute.make_tensor(fbwd_tmem_ptr + 256, tDKV_frag.layout)
            tDK_t = cute.make_tensor(fbwd_tmem_ptr + {dk_off}, tDK_frag.layout)
            tDQ_t = cute.make_tensor(fbwd_tmem_ptr + {dq_off}, tDQ_frag.layout)
            tP = cute.make_tensor(tS_t.iterator, _fbwd_ptl.outer)
            tSrK = fbwd_sst.make_fragment_A(sK)
            tSrQ = fbwd_sst.make_fragment_B(sQ)
            tSrV = fbwd_sst.make_fragment_A(sV)
            tSrDO = fbwd_sst.make_fragment_B(sdO)
            tTSrP = fbwd_tst.make_fragment_A(tP)
            tTSrDOt = fbwd_tst.make_fragment_B(sdOt)
            tKSrDS = fbwd_dskt.make_fragment_A(fbwd_sdS_nk)
            tKSrQt = fbwd_dskt.make_fragment_B(sQt)
            tDQrDS = fbwd_dqt.make_fragment_A(fbwd_sdS_mn)
            tDQrKt = fbwd_dqt.make_fragment_B(sKt)
{ds_tmem_mma_setup}            fbwd_s_addr = tS_t.iterator.toint()
            fbwd_dp_addr = tdP_t.iterator.toint()
            fbwd_dv_addr = tDV_t.iterator.toint()
            fbwd_dk_addr = tDK_t.iterator.toint()
            fbwd_dqa_addr = tDQ_t.iterator.toint()
            fbwd_k_base = _helion_flash_ptx.smem_desc_base_from_tensor(sK, _helion_flash_ptx.Major.K)
            fbwd_v_base = _helion_flash_ptx.smem_desc_base_from_tensor(sV, _helion_flash_ptx.Major.K)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_ss_mma.op, "helion_fbwd_ss_idesc")
            fbwd_q_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQ, _helion_flash_ptx.Major.K)
            fbwd_do_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdO, _helion_flash_ptx.Major.K)
            fbwd_dot_base = _helion_flash_ptx.smem_desc_base_from_tensor(sdOt, _helion_flash_ptx.Major.MN)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_ts_mma.op, "helion_fbwd_ts_idesc")
            fbwd_dsnk_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sdS_nk, _helion_flash_ptx.Major.K)
            fbwd_qt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sQt, _helion_flash_ptx.Major.MN)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_dsk_mma.op, "helion_fbwd_dsk_idesc")
            fbwd_dsmn_base = _helion_flash_ptx.smem_desc_base_from_tensor(fbwd_sdS_mn, _helion_flash_ptx.Major.MN)
            fbwd_kt_base = _helion_flash_ptx.smem_desc_base_from_tensor(sKt, _helion_flash_ptx.Major.MN)
            _helion_flash_ptx.declare_ptx_idesc(_fbwd_dq_mma.op, "helion_fbwd_dq_idesc")
{aug_mma_setup}            fbwd_mb = storage.mma_base.get_tensor(cute.make_layout(8))
            fbwd_mb[2] = _helion_flash_ptx.make_smem_desc_start_addr(fbwd_sdS_nk[None, None, None, 0].iterator)
            fbwd_mb[3] = _helion_flash_ptx.make_smem_desc_start_addr(fbwd_sdS_mn[None, None, None, 0].iterator)
            fbwd_ke = fbwd_k_cons.wait_and_advance()
            fbwd_mb[0] = _helion_flash_ptx.make_smem_desc_start_addr(sK[None, None, None, fbwd_ke.index].iterator)
            fbwd_mb[4] = _helion_flash_ptx.make_smem_desc_start_addr(sKt[None, None, None, fbwd_ke.index].iterator)
            fbwd_st_k = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 0)
            fbwd_st_dsnk = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 2)
            fbwd_st_dsmn = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 3)
            fbwd_st_kt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 4)
            fbwd_dv_zero = cutlass.Boolean(True)
            fbwd_dk_zero = cutlass.Boolean(True)
            fbwd_qe = fbwd_q_cons.wait_and_advance()
            fbwd_doe = fbwd_do_cons.wait_and_advance()
            fbwd_ve = fbwd_v_cons.wait_and_advance()
            fbwd_mb[1] = _helion_flash_ptx.make_smem_desc_start_addr(sV[None, None, None, fbwd_ve.index].iterator)
            fbwd_st_v = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 1)
            _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_s_addr, fbwd_st_k, fbwd_k_base, tSrK[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(sQ[None, None, None, fbwd_qe.index].iterator), fbwd_q_base, tSrQ[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True)
{aug_s_mma("fbwd_qe", 12)}            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_s_full_ptr)
{mma_dq_prologue}
            _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dp_addr, fbwd_st_v, fbwd_v_base, tSrV[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(sdO[None, None, None, fbwd_doe.index].iterator), fbwd_do_base, tSrDO[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True)
{aug_dp_mma("fbwd_qe", 12)}            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_dp_full_ptr)
            _helion_flash_rt.mbar_spin_wait(fbwd_p_full_ptr, fbwd_pf_phase, 10000000)
            fbwd_pf_phase ^= 1
            _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dv_addr, fbwd_s_addr, _helion_flash_ptx.make_smem_desc_start_addr(sdOt[None, None, None, fbwd_doe.index].iterator), fbwd_dot_base, tTSrP[None, None, None, 0].layout, tTSrDOt[None, None, None, 0].layout, "helion_fbwd_ts_idesc", zero_init=fbwd_dv_zero, a_offsets={_FBWD_2CTA_DS_OFFSETS})
            fbwd_dv_zero = cutlass.Boolean(False)
            fbwd_doe.release()
            for fbwd_i in cutlass.range(fbwd_steps - 1, unroll=1):
                fbwd_st_k = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 0)
                fbwd_st_v = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 1)
                fbwd_st_dsnk = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 2)
                fbwd_st_dsmn = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 3)
                fbwd_st_kt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 4)
                fbwd_qe2 = fbwd_q_cons.wait_and_advance()
                _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_s_addr, fbwd_st_k, fbwd_k_base, tSrK[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(sQ[None, None, None, fbwd_qe2.index].iterator), fbwd_q_base, tSrQ[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True)
{aug_s_mma("fbwd_qe2", 16)}                with cute.arch.elect_one():
                    cute_tcgen05_flash.commit(fbwd_s_full_ptr)
                _helion_flash_rt.mbar_spin_wait(fbwd_ds_full_ptr, fbwd_dsf_phase, 10000000)
                fbwd_dsf_phase ^= 1{textwrap.indent(dq_empty_wait_g5, "    ")}
                _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dqa_addr, fbwd_st_dsmn, fbwd_dsmn_base, tDQrDS[None, None, None, 0].layout, fbwd_st_kt, fbwd_kt_base, tDQrKt[None, None, None, 0].layout, "helion_fbwd_dq_idesc", zero_init=True)
                with cute.arch.elect_one():
                    cute_tcgen05_flash.commit(fbwd_dq_full_ptr)
                {dk_gemm}
                fbwd_dk_zero = cutlass.Boolean(False)
                with cute.arch.elect_one():
                    cute_tcgen05_flash.commit(fbwd_dk_done_ptr)
                fbwd_qe.release()
                fbwd_qe = fbwd_qe2{textwrap.indent(dq_empty_wait_g2, "    ")}
                fbwd_doe = fbwd_do_cons.wait_and_advance()
                _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dp_addr, fbwd_st_v, fbwd_v_base, tSrV[None, None, None, 0].layout, _helion_flash_ptx.make_smem_desc_start_addr(sdO[None, None, None, fbwd_doe.index].iterator), fbwd_do_base, tSrDO[None, None, None, 0].layout, "helion_fbwd_ss_idesc", zero_init=True)
{aug_dp_mma("fbwd_qe", 16)}                with cute.arch.elect_one():
                    cute_tcgen05_flash.commit(fbwd_dp_full_ptr)
                _helion_flash_rt.mbar_spin_wait(fbwd_p_full_ptr, fbwd_pf_phase, 10000000)
                fbwd_pf_phase ^= 1
                _helion_flash_ptx.gemm_ptx_precomputed_pv_ts(fbwd_dv_addr, fbwd_s_addr, _helion_flash_ptx.make_smem_desc_start_addr(sdOt[None, None, None, fbwd_doe.index].iterator), fbwd_dot_base, tTSrP[None, None, None, 0].layout, tTSrDOt[None, None, None, 0].layout, "helion_fbwd_ts_idesc", zero_init=False, a_offsets={_FBWD_2CTA_DS_OFFSETS})
                fbwd_doe.release()
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_dv_done_ptr)
            fbwd_st_k = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 0)
            fbwd_st_v = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 1)
            fbwd_st_dsnk = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 2)
            fbwd_st_dsmn = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 3)
            fbwd_st_kt = _helion_flash_rt.ld_volatile_shared_u32(fbwd_mb.iterator + 4)
            _helion_flash_rt.mbar_spin_wait(fbwd_ds_full_ptr, fbwd_dsf_phase, 10000000)
            fbwd_dsf_phase ^= 1{textwrap.indent(dq_empty_wait_tail, "    ")}
            _helion_flash_ptx.gemm_ptx_dyn_qk(fbwd_dqa_addr, fbwd_st_dsmn, fbwd_dsmn_base, tDQrDS[None, None, None, 0].layout, fbwd_st_kt, fbwd_kt_base, tDQrKt[None, None, None, 0].layout, "helion_fbwd_dq_idesc", zero_init=True)
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_dq_full_ptr)
            {dk_gemm}
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_dk_done_ptr)
            fbwd_qe.release()
            with cute.arch.elect_one():
                cute_tcgen05_flash.commit(fbwd_dkv_done_ptr)
            fbwd_ke.release()
"""
    reduce_region = f"""        for fbwd_i in cutlass.range(fbwd_steps, unroll=1):
{step_decode}
{reduce_body}
"""
    if persistent:
        tma_region = (
            p_head("tma", 8)
            + textwrap.indent(tma_region + tma_issued, "    ")
            + p_tail("tma", 8)
        )
        mma_region = (
            p_head("mma", 8) + textwrap.indent(mma_region, "    ") + p_tail("mma", 8)
        )
        reduce_region = (
            p_head("cons", 8)
            + textwrap.indent(reduce_region, "    ")
            + p_tail("cons", 8)
        )
    tma_block = (
        "    if warp_idx == 13:\n        fbwd_lane = tidx % 32\n"
        + tma_scalars
        + tma_region
        + (
            "        fbwd_q_prod.tail()\n        fbwd_do_prod.tail()\n"
            if persistent
            else "            fbwd_q_prod.tail()\n            fbwd_do_prod.tail()\n"
        )
    )
    # Barrier parities live outside the tile loop (they continue across the
    # persistent walk's tiles); the aux-slice parity follows the Q ring.
    mma_block = (
        "        fbwd_pf_phase = cutlass.Int32(0)\n"
        "        fbwd_dsf_phase = cutlass.Int32(0)\n"
        "        fbwd_dqe_phase = cutlass.Int32(0)\n"
        + ("        fbwd_aug_phase = cutlass.Int32(0)\n" if aug else "")
        + mma_region
    )
    reduce_block = reduce_region
    body = f"""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    fbwd_local_tidx = tidx % 128
    fbwd_tile_id = cutlass.Int32(cute.arch.block_idx()[0])
{_tile_decode(plan, tile_group)}    fbwd_b = fbwd_bhk // {plan.kv_heads}
    fbwd_hk = fbwd_bhk % {plan.kv_heads}
    fbwd_kv_base = fbwd_n_tile * 128
{binding_lines}    if warp_idx == 0:
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_q)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_k)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_v)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_do)
        cute_cpasync_flash.prefetch_descriptor(_fbwd_tma_dq)
    _fbwd_storage_cls = _helion_flash_rt.flash_bwd_shared_storage({d}, {q_stage}, {do_stage}, {kv_stage}, {io_dtype}, {table_entries}, {dq_boxes}, {smem.aug_stages}, {smem.stage_lse}, {smem.stage_delta}, {smem.aux_bytes})
    smem = cutlass_utils_flash.SmemAllocator()
    storage = smem.allocate(_fbwd_storage_cls)
    fbwd_sSteps = storage.sSteps.get_tensor(cute.make_layout({table_entries}))
{tab_alias}{prologue_table}    sQ = storage.sQ.get_tensor(_fbwd_qsl.outer, swizzle=_fbwd_qsl.inner)
    sdO = storage.sdO.get_tensor(_fbwd_dosl.outer, swizzle=_fbwd_dosl.inner)
    sK = storage.sK.get_tensor(_fbwd_ksl.outer, swizzle=_fbwd_ksl.inner)
    sV = storage.sV.get_tensor(_fbwd_vsl.outer, swizzle=_fbwd_vsl.inner)
    sQt = cute.make_tensor(cute.recast_ptr(sQ.iterator, _fbwd_qtl.inner), _fbwd_qtl.outer)
    sdOt = cute.make_tensor(cute.recast_ptr(sdO.iterator, _fbwd_dotl.inner), _fbwd_dotl.outer)
    sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, _fbwd_ktl.inner), _fbwd_ktl.outer)
    fbwd_sdS_mn = storage.sdS.get_tensor(_fbwd_dssl.outer, swizzle=_fbwd_dssl.inner)
    fbwd_sdS_nk = cute.make_tensor(cute.recast_ptr(fbwd_sdS_mn.iterator, _fbwd_dsnk.inner), _fbwd_dsnk.outer)
{stat_views}    fbwd_sdq = storage.sdQaccum.get_tensor(_fbwd_dqsl.outer, swizzle=_fbwd_dqsl.inner)
{aug_views}    fbwd_s_full_ptr = storage.s_full_mbar.data_ptr()
    fbwd_dp_full_ptr = storage.dp_full_mbar.data_ptr()
    fbwd_p_full_ptr = storage.p_full_mbar.data_ptr()
    fbwd_ds_full_ptr = storage.ds_full_mbar.data_ptr()
    fbwd_dq_full_ptr = storage.dq_full_mbar.data_ptr()
    fbwd_dq_empty_ptr = storage.dq_empty_mbar.data_ptr()
    fbwd_dkv_done_ptr = storage.dkv_done_mbar.data_ptr()
    fbwd_dk_done_ptr = storage.dk_done_mbar.data_ptr()
    fbwd_dv_done_ptr = storage.dv_done_mbar.data_ptr()
    fbwd_tfull_ptr = storage.flex_mbar.data_ptr()
    fbwd_tempty_ptr = storage.flex_mbar.data_ptr() + 2
    fbwd_sdsfree_ptr = storage.flex_mbar.data_ptr() + 4
    fbwd_tissued_ptr = storage.flex_mbar.data_ptr() + 5
    fbwd_tmem_dealloc_ptr = storage.tmem_dealloc_mbar.data_ptr()
    if tidx == 0:
        cute.arch.mbarrier_init(fbwd_s_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dp_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_p_full_ptr, 256)
        cute.arch.mbarrier_init(fbwd_ds_full_ptr, 256)
        cute.arch.mbarrier_init(fbwd_dq_full_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dq_empty_ptr, 128)
        cute.arch.mbarrier_init(fbwd_dkv_done_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dv_done_ptr, 1)
        cute.arch.mbarrier_init(fbwd_dk_done_ptr, 1)
{flex_mbar_init}{aug_mbar_init}    fbwd_tmem_user_bar = cutlass_pipeline_flash.NamedBarrier(barrier_id=2, num_threads=13 * 32)
    fbwd_tmem = cutlass_utils_flash.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=fbwd_tmem_user_bar, allocator_warp_id=12, is_two_cta=False, two_cta_tmem_dealloc_mbar_ptr=fbwd_tmem_dealloc_ptr)
    fbwd_q_bytes = cute.size_in_bytes({io_dtype}, cute.select(_fbwd_qsl, mode=[0, 1, 2]))
    fbwd_k_bytes = cute.size_in_bytes({io_dtype}, cute.select(_fbwd_ksl, mode=[0, 1, 2]))
    fbwd_q_prod, fbwd_q_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={q_stage}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_q_bytes, barrier_storage=storage.q_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
    fbwd_do_prod, fbwd_do_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={do_stage}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_q_bytes, barrier_storage=storage.do_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
    fbwd_k_prod, fbwd_k_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={kv_stage}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_k_bytes, barrier_storage=storage.k_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
    fbwd_v_prod, fbwd_v_cons = cutlass_pipeline_flash.PipelineTmaUmma.create(num_stages={kv_stage}, producer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), consumer_group=cutlass_pipeline_flash.CooperativeGroup(cutlass_pipeline_flash.Agent.Thread), tx_count=fbwd_k_bytes, barrier_storage=storage.v_mbar_ptr.data_ptr(), defer_sync=True).make_participants()
    # One fence + sync for every mbarrier above (the pipelines are created
    # ``defer_sync``); the step-table fill below ends in the second sync.
{clc_setup}{aux_fill}    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    fbwd_sst = _fbwd_ss_mma.get_slice(0)
    gK = cute.flat_divide(_fbwd_mKt, cute.select((128, 128, {d}), mode=[0, 2]))
    tSgK = fbwd_sst.partition_A(gK)
    tKsK, tKgK_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(tSgK, 0, 3))
{prologue_k_fill}{prologue_k_fill_aug}    fbwd_tst = _fbwd_ts_mma.get_slice(0)
    fbwd_dskt = _fbwd_dsk_mma.get_slice(0)
    fbwd_dqt = _fbwd_dq_mma.get_slice(0)
    gQ = cute.flat_divide(_fbwd_mQt, cute.select((128, 128, {d}), mode=[1, 2]))
    gV = cute.flat_divide(_fbwd_mVt, cute.select((128, 128, {d}), mode=[0, 2]))
    gdO = cute.flat_divide(_fbwd_mdOt, cute.select((128, 128, {d}), mode=[1, 2]))
    tSgQ = fbwd_sst.partition_B(gQ)
    tSgV = fbwd_sst.partition_A(gV)
    tSgdO = fbwd_sst.partition_B(gdO)
    tQsQ, tQgQ_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(tSgQ, 0, 3))
    tVsV, tVgV_kdl = cute_cpasync_flash.tma_partition(_fbwd_tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(tSgV, 0, 3))
    tDOsDO, tDOgDO_qdl = cute_cpasync_flash.tma_partition(_fbwd_tma_do, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 3), cute.group_modes(tSgdO, 0, 3))
    if (warp_idx >= 12) & (warp_idx < 16):
        {_setmaxreg(regs_other)}
{tma_block}    if warp_idx == 12:
        fbwd_tmem.allocate(512)
        fbwd_s_shape = fbwd_sst.partition_shape_C((128, 128))
        tStS_frag = fbwd_sst.make_fragment_C(fbwd_s_shape)
        fbwd_dkv_shape = fbwd_tst.partition_shape_C((128, {d}))
        tDKV_frag = fbwd_tst.make_fragment_C(fbwd_dkv_shape)
        fbwd_dk_shape = fbwd_dskt.partition_shape_C((128, {d}))
        tDK_frag = fbwd_dskt.make_fragment_C(fbwd_dk_shape)
        fbwd_dq_shape = fbwd_dqt.partition_shape_C((128, {d}))
        tDQ_frag = fbwd_dqt.make_fragment_C(fbwd_dq_shape)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
{mma_block}        fbwd_tmem.relinquish_alloc_permit()
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem.free(fbwd_tmem_ptr)
    if warp_idx < 4:
        {_setmaxreg(regs_reduce)}
        fbwd_dq_shape = fbwd_dqt.partition_shape_C((128, {d}))
        tDQ_frag = fbwd_dqt.make_fragment_C(fbwd_dq_shape)
        _helion_flash_rt.named_barrier_wait_unaligned(2, 13 * 32)
        fbwd_tmem_ptr = fbwd_tmem.retrieve_ptr(cutlass.Float32)
        tDQ_t = cute.make_tensor(fbwd_tmem_ptr + {dq_off}, tDQ_frag.layout)
        cDQ = cute.make_identity_tensor((128, {d}))
        tDQcDQ = fbwd_dqt.partition_C(cDQ)
        fbwd_dq_ld_atom = cute.make_copy_atom(cute_tcgen05_flash.Ld32x32bOp(cute_tcgen05_flash.Repetition(32)), cutlass.Float32)
        fbwd_tiled_dq_ld = cute_tcgen05_flash.make_tmem_copy(fbwd_dq_ld_atom, tDQ_t)
        fbwd_thr_dq_ld = fbwd_tiled_dq_ld.get_slice(fbwd_local_tidx)
        tRDtDQ = fbwd_thr_dq_ld.partition_S(tDQ_t)
        tRDcDQ = fbwd_thr_dq_ld.partition_D(tDQcDQ)
        fbwd_out_scale = cutlass.Float32({scale_expr})
        fbwd_phase = cutlass.Int32(0)
{reduce_block}        cute.arch.cp_async_bulk_wait_group(0, read=True)
        _helion_flash_rt.named_barrier_arrive_unaligned(2, 13 * 32)
{clc_warps}{compute_blocks}"""
    return body  # noqa: RET504


def codegen_attention_flash_bwd_flex(cg: GenerateAST, match: FlexBwdMatch) -> bool:
    """Replace the device body with the fused flex backward kernel (True when
    emitted; False to fall back to the generic lowering)."""
    import ast as ast_module

    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction
    from .cute_flash import _flash_graph_host_tensors
    from .cute_flash import emit_flash_module_statements
    from .cute_flash_bwd import _FLASH_BWD_KERNEL_PARAMS
    from .flex_match import flex_host_scalar_device_name
    from .flex_match import flex_host_scalar_symbol

    df = cg.device_function
    plan = match.plan
    host_tensors = _flash_graph_host_tensors(df.codegen.codegen_graphs)
    role_names = {
        "q": match.q_name,
        "k": match.k_name,
        "v": match.v_name,
        "do": match.do_name,
        "lse": match.lse_name,
        "delta": match.delta_name,
        "dq": match.dq_name,
        "dk": match.dk_name,
        "dv": match.dv_name,
    }
    required = [*role_names.values(), *match.table_names, *match.aux_names]
    if any(name not in host_tensors for name in required):
        return False
    args: dict[str, TensorArg] = {}
    for name in dict.fromkeys(required):
        prefer_name = name if name.isidentifier() else None
        args[name] = df.tensor_arg(host_tensors[name], prefer_name=prefer_name)
    d = plan.head_dim
    batch, heads, kv_heads = plan.batch, plan.heads, plan.kv_heads
    checks = (
        (match.q_name, (batch, heads, plan.q_seq, d), plan.io_dtype),
        (match.do_name, (batch, heads, plan.q_seq, plan.head_dim_v), plan.io_dtype),
        (match.k_name, (batch, kv_heads, plan.kv_seq, d), plan.io_dtype),
        (match.v_name, (batch, kv_heads, plan.kv_seq, plan.head_dim_v), plan.io_dtype),
        (match.dk_name, (batch, kv_heads, plan.kv_seq, d), plan.io_dtype),
        (match.dv_name, (batch, kv_heads, plan.kv_seq, plan.head_dim_v), plan.io_dtype),
        (match.dq_name, (batch, heads, plan.q_seq, d), torch.float32),
        (match.lse_name, (batch, heads, plan.q_seq), torch.float32),
        (match.delta_name, (batch, heads, plan.q_seq), torch.float32),
    )
    for name, shape, dtype in checks:
        fake = args[name].fake_value
        if (
            tuple(int(s) for s in fake.shape) != shape
            or fake.dtype != dtype
            or not fake.is_contiguous()
        ):
            return False
    walk = plan.walk
    for index, name in enumerate(match.table_names):
        fake = args[name].fake_value
        expected_ndim = 3 if index % 2 == 0 else 4
        if (
            fake.ndim != expected_ndim
            or fake.dtype != torch.int32
            or tuple(int(s) for s in fake.shape[:3])
            != (walk.table_batch, walk.table_heads, walk.table_cols)
            or not fake.is_contiguous()
        ):
            return False
    for buffer in plan.aux_buffers:
        fake = args[buffer.name].fake_value
        if (
            fake.dtype != buffer.dtype
            or tuple(int(s) for s in fake.shape) != buffer.shape
            or tuple(int(s) for s in fake.stride()) != buffer.strides
        ):
            return False
    host_function = HostFunction.current()
    scalar_exprs: list[str] = []
    for name in plan.host_scalars:
        symbol = flex_host_scalar_symbol(host_function, name)
        arg_name = flex_host_scalar_device_name(df, host_function, name)
        if symbol is None or arg_name is None:
            return False
        kind = "Int32" if symbol.is_integer else "Float32"  # pyrefly: ignore[missing-attribute]
        scalar_exprs.append(f"cutlass.{kind}({arg_name})")
    if plan.scale_arg is not None:
        scale_name = flex_host_scalar_device_name(df, host_function, plan.scale_arg)
        if scale_name is None:
            return False
        scale_expr = f"cutlass.Float32({scale_name})"
    else:
        assert plan.scale_value is not None
        scale_expr = repr(float(plan.scale_value))

    env = CompileEnvironment.current()
    io_dtype_str = env.backend.dtype_str(plan.io_dtype)
    exp2_f32 = bool(df.config.config.get("cute_flash_bwd_exp2_f32", 0))
    mask_form = str(df.config.config.get("cute_flash_bwd_mask_form", "bitmask"))
    reg_split = df.config.config.get("cute_flash_bwd_regs")
    hook_cols_cfg = df.config.config.get("cute_flash_bwd_hook_cols", 32)
    hook_cols = hook_cols_cfg if hook_cols_cfg in (8, 16, 32) else 32
    hook_pin = bool(df.config.config.get("cute_flash_bwd_hook_pin", 1))
    bias_fold = bool(df.config.config.get("cute_flash_bwd_bias_fold", 1))
    tile_group_cfg = df.config.config.get("cute_flash_bwd_tile_group", 0)
    tile_group = tile_group_cfg if isinstance(tile_group_cfg, int) else 0
    stats_mma = bool(df.config.config.get("cute_flash_bwd_stats_mma", 0))
    persistent = bool(df.config.config.get("cute_flash_bwd_persistent", 0))
    # dS^T through TMEM for the dK MMA: head dim 64 only (at 128 the dQ
    # accumulator aliases the dP columns the packed dS^T would occupy).
    ds_tmem = bool(df.config.config.get("cute_flash_bwd_ds_tmem", 0)) and d == 64
    total_tiles = batch * kv_heads * plan.num_kv_tiles
    exp2_emu_cfg = df.config.config.get("cute_flash_bwd_exp2_emu", 0)
    # The polynomial exp2 is an approximation: fast_math kernels only.
    exp2_emu = (
        exp2_emu_cfg if isinstance(exp2_emu_cfg, int) and env.settings.fast_math else 0
    )
    # Hooks that read the natural-unit score (a non-additive score_mod or a
    # Jacobian) need S before the LSE term is folded in.
    bias_region = additive_bias_region(plan) if bias_fold else None
    raw_score = (
        plan.score_region is not None and bias_region is None
    ) or plan.grad_region is not None
    from .cute_flash import flash_flex_aux_smem_canonical

    aux_smem_cfg = df.config.config.get("cute_flash_bwd_aux_smem", 0)
    aux_smem = flash_flex_aux_smem_canonical(
        aux_smem_cfg if isinstance(aux_smem_cfg, int) else 0,
        fbwd_aux_smem_candidates(plan),
    )
    smem = _fit_smem_budget(
        plan,
        stats_mma=stats_mma,
        raw_score=raw_score,
        device=env.device,
        persistent=persistent,
        aux_smem=aux_smem,
    )
    if smem is None:
        # Even the smallest storage (a 128K-token walk's step table, say)
        # exceeds the SM's shared memory: leave the kernel to the generic path.
        return False
    q_stage, do_stage, dq_boxes = smem.q_stage, smem.do_stage, smem.dq_boxes

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
    wrapper_plan: dict[str, object] = {
        "kind": "helion_flash_bwd",
        "q_name": args[match.q_name].name,
        "k_name": args[match.k_name].name,
        "v_name": args[match.v_name].name,
        "do_name": args[match.do_name].name,
        "lse_name": args[match.lse_name].name,
        "delta_name": args[match.delta_name].name,
        "dq_name": args[match.dq_name].name,
        "dk_name": args[match.dk_name].name,
        "dv_name": args[match.dv_name].name,
        "head_dim": d,
        "dtype": io_dtype_str,
        "total_q_rows": plan.num_bh * plan.q_seq,
        "total_kv_rows": batch * kv_heads * plan.kv_seq,
        "q_stage": q_stage,
        "do_stage": do_stage,
        "kv_stage": 1,
        "two_cta": False,
        "persistent": persistent,
        "total_tiles": total_tiles,
        "topology": "fa4",
        # Per-(batch, head) geometry: the launcher builds (seq, D, B*H) maps.
        "flex_seq_q": plan.q_seq,
        "flex_bh_q": plan.num_bh,
        "flex_seq_k": plan.kv_seq,
        "flex_bh_k": batch * kv_heads,
        "flex_dq_boxes": dq_boxes,
        "flex_aug_stages": smem.aug_stages,
    }
    cg.cute_wrapper_plans.append(wrapper_plan)  # type: ignore[attr-defined]
    df.wrapper_only_params.extend(_FLASH_BWD_KERNEL_PARAMS)
    if smem.aug_stages:
        df.wrapper_only_params.extend(_FLASH_BWD_FLEX_AUG_PARAMS)
    df.placeholder_args.update(a.name for a in args.values())
    cg.cute_uses_matmul = True  # type: ignore[attr-defined]
    df.cute_state.attention_flash_threads = 512
    # The table / aux bindings are CTA-wide names: emit them before the warp
    # roles (the compute blocks re-bind nothing).
    body_src = emit_flash_bwd_flex_device_body(
        plan=plan,
        io_dtype=io_dtype_str,
        scale_expr=scale_expr,
        q_stage=q_stage,
        do_stage=do_stage,
        exp2_f32=exp2_f32,
        mask_form=mask_form,
        bindings=bindings,
        device_function=df,
        reg_split=None if reg_split is None else str(reg_split),
        hook_cols=hook_cols,
        hook_pin=hook_pin,
        bias_fold=bias_fold,
        tile_group=tile_group,
        smem=smem,
        exp2_emu=exp2_emu,
        persistent=persistent,
        total_tiles=total_tiles,
        ds_tmem=ds_tmem,
    )
    wrapped = ast_module.parse("if True:\n" + body_src)
    assert isinstance(wrapped.body[0], ast_module.If)
    df.body = list(wrapped.body[0].body)
    df.preamble = []
    return True
