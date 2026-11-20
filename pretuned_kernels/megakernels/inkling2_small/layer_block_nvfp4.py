"""Inkling-2-Small block-boundary NVFP4 decoder layer of one rank as one Helion megakernel (TP4, batch-1 verify
window of T = 8 rows): the attention half's roots (``a_`` arguments: attn_block.py's, which also start a new
attn_res block and its statistics, with PQR reading an fp32 rel_proj copy and A / AN reading PQR-masked
relative-position bias tiles), then moe_nvfp4.py's router + routed-expert roots. Generated from the two kernels'
sources."""

from __future__ import annotations

from pretuned_kernels.megakernels.inkling2_small._common import E4M3
from pretuned_kernels.megakernels.inkling2_small._common import LN2
from pretuned_kernels.megakernels.inkling2_small._common import LOG2E
from pretuned_kernels.megakernels.inkling2_small._common import _distinct_table
from pretuned_kernels.megakernels.inkling2_small._common import _expert_of
from pretuned_kernels.megakernels.inkling2_small._common import _l2_clip
from pretuned_kernels.megakernels.inkling2_small._common import _local_ids
from pretuned_kernels.megakernels.inkling2_small._common import _renorm
from pretuned_kernels.megakernels.inkling2_small._common import _rsqrt_approx
from pretuned_kernels.megakernels.inkling2_small._common import _scale_tile
from pretuned_kernels.megakernels.inkling2_small._common import _select
from pretuned_kernels.megakernels.inkling2_small._common import _sigmoid
from pretuned_kernels.megakernels.inkling2_small._common import _verify_conv
from pretuned_kernels.megakernels.inkling2_small._fp4 import capped_swiglu
from pretuned_kernels.megakernels.inkling2_small._fp4 import quant46
from pretuned_kernels.megakernels.inkling2_small._fp4 import quant46s
from pretuned_kernels.megakernels.inkling2_small._fp4 import rcp_rn
import torch

import helion.language as hl


def _ink_layer(
    a_x: torch.Tensor,  # [R, H] bf16: the attention input (attn_norm output); a_rows past T zero
    a_w_qkvr: torch.Tensor,  # [N, S, H/S] bf16 view of this rank's qkvr weight [N, H] (N = 3792: q|r|a_gate|sinks|k|v|pad)
    a_part: torch.Tensor,  # [S, R, N] fp32 workspace: the qkvr split-K partials
    a_k_conv_w: torch.Tensor,  # [Dkv, 4] bf16: k_sconv weight (Dkv = 2 a_heads a_x 128)
    a_v_conv_w: torch.Tensor,  # [Dkv, 4] bf16
    a_k_conv_cache: torch.Tensor,  # [NB * CB * 2, 128] bf16: the K stream of the layer's paged conv cache, row slot * 2 + head
    a_v_conv_cache: torch.Tensor,  # [NB * CB * 2, 128] bf16: its V stream
    a_conv_pages: torch.Tensor,  # [MB] int32: the request's conv-cache block table (position p in block p // CB)
    a_conv_loc: torch.Tensor,  # [T] int64: the verify a_rows' conv-cache a_slots (block * CB + offset; < 0: none)
    a_conv_block: hl.constexpr,  # CB: the conv cache's positions a block
    a_q_scale: torch.Tensor,  # [12, 128] bf16: q_l2_clip scale
    a_r_scale: torch.Tensor,  # [12, 16] bf16: rel_l2_clip scale
    a_rel_proj32: torch.Tensor,  # [16, E] fp32: rel_logits_proj widened from bf16 (E = a_rel_extent = 1024)
    a_k_buf: torch.Tensor,  # [NB * BS, 2, 128] bf16: this layer's K cache by slot (vLLM's kv_cache, slot-major, [..., :128])
    a_v_buf: torch.Tensor,  # [NB * BS, 2, 128] bf16: its V cache ([..., 128:])
    a_pages: torch.Tensor,  # [MB] int32: the request's block table (position p in block a_pages[p // BS], row p % BS)
    a_seq_len: torch.Tensor,  # [1] int32: the request's a_length including the T new a_rows
    a_kv_loc: torch.Tensor,  # [T] int64: the new a_rows' KV cache a_slots (block * BS + row; < 0: none)
    a_page_size: hl.constexpr,  # BS: the KV cache's positions a block
    a_q_ws: torch.Tensor,  # [T, 2, 8, 128] bf16 workspace (a_padded a_heads zero)
    a_rel_ws: torch.Tensor,  # [T, 12, E] bf16 workspace (unused: PQR writes the biases below directly)
    a_bias_ws: torch.Tensor,  # [2, 64, LIVE_PAGES x BS] bf16 workspace: A's bias by (row, window key); -inf where a
    #                           row may not see the key (the host fills it with -inf: padded-head rows stay so)
    a_nbias_ws: torch.Tensor,  # [2, 64, 16] bf16 workspace: AN's bias by (row, new key), -inf likewise
    a_att_ws: torch.Tensor,  # [2, PAGES, 64, 128] fp32 workspace: per-a_page normalized partial outputs
    a_lse_ws: torch.Tensor,  # [2, PAGES, 64] fp32 workspace: per-a_page log2-sum-exp2 (-inf: no visible key)
    a_kv_new: torch.Tensor,  # [2, 2, 16, 128] bf16 workspace: (K | V, KV head, new row (T used), dim)
    a_att_new: torch.Tensor,  # [2, 64, 128] fp32 workspace: AN's normalized partial outputs
    a_lse_new: torch.Tensor,  # [2, 64] fp32 workspace: AN's log2-sum-exp2
    a_o_in: torch.Tensor,  # [R, 4, 384] bf16 workspace: a_gate-a_sink output by K quarter (pad a_rows zero)
    a_wo: torch.Tensor,  # [H, 4, 384] bf16 view of this rank's wo_ud weight [H, 1536]
    a_part_o: torch.Tensor,  # [4, R, H] fp32 workspace: a_wo partials by K quarter
    a_opart: torch.Tensor,  # [T, H/C, C] bf16 symmetric: this rank's a_wo a_rows (the all-reduce input), C columns a task
    a_a_conv_w: torch.Tensor,  # [H, 4] bf16: attn_sconv weight
    a_a_conv_cache: torch.Tensor,  # [NB * CB * 2, H/2] bf16: the attn_sconv stream of the paged conv cache (as K's)
    a_res_block: torch.Tensor,  # [T, H] bf16: this block-boundary layer's partial (a new block)
    a_res_queries: torch.Tensor,  # [2 Qp, H/D, D] bf16: every attn_res query (H^-0.5-scaled), bf16 high parts then lows
    a_res_dpart: torch.Tensor,  # [H/D, 3, Qp, R] fp32 workspace: by K a_chunk, the block's dots with the query highs, lows,
    a_res_hist: torch.Tensor,  # [NB, T, H/D, D] bf16 attention: the blocks (slot a_res_slot[0] gets a_res_block's a_rows)
    a_res_rstd: torch.Tensor,  # [NB, T] fp32 attention: their a_rows' rstd (RMS over H, eps 1e-6)
    a_res_dots: torch.Tensor,  # [NB, NQ, T] fp32 attention: their a_rows . every query
    a_res_slot: torch.Tensor,  # [1] int32: the new block's index
    a_res_q: torch.Tensor,  # [H/V, V] fp32: the MLP input's attn_res query, scaled by H^-0.5
    a_res_w: torch.Tensor,  # [H/V, V]: mlp_norm's weight
    a_res_acc: torch.Tensor,  # [T, H/V, V] fp32: the blocks before the new one, softmax-weighted (the MoE kernel's F)
    a_res_stat: torch.Tensor,  # [T, H/C, 4] fp32: (their logit max, exp a_total, a_res_acc's square sum by columns, 0)
    a_res_part: torch.Tensor,  # [T, H/C, 8] fp32 workspace: J's sums over the new block b and the attention a_rows a_x
    x: torch.Tensor,  # [R, H] bf16: the MoE input (mlp_norm output), replicated; rows past T zero
    gate_w: torch.Tensor,  # [N/128, 128, H/128, 128] bf16 view of the router weight [N, H]
    gate_b: torch.Tensor,  # [N] fp32 router bias
    logit_part: torch.Tensor,  # [S_G, P, R, 128] fp32 workspace: router logit K-part partials, P = next_pow2(N/128)
    ids: torch.Tensor,  # [T * K] int32 out: global expert ids, token-major
    gammas: torch.Tensor,  # [T * K] fp32 out: routing weights (route_scale applied)
    w13: torch.Tensor,  # [E, 2I/128, 128, H/512, 256] uint8 view (moe_views): e2m1x2, rows gate (2i) / up (2i+1)
    w13_s: torch.Tensor,  # [E, 2I/128, H/512, 16, 256] uint8 view of the swizzled E4M3 block scales
    w13_amax: torch.Tensor,  # [E] fp32
    w13_den: torch.Tensor,  # [1] fp32
    w2: torch.Tensor,  # [E, H/128, 128, I/256, 128] uint8 view
    w2_s: torch.Tensor,  # [E, H/128, I/256, 8, 256] uint8 view of the swizzled scales
    w2_amax: torch.Tensor,  # [E] fp32
    w2_den: torch.Tensor,  # [1] fp32
    xq: torch.Tensor,  # [R, H/512, 256] uint8 workspace (R = 16 token rows)
    xs: torch.Tensor,  # [R, H/512, 32] uint8
    x_amax: torch.Tensor,  # [R] fp32
    h: torch.Tensor,  # [S, 2I/128, T, 64] bf16: live token rows of the SwiGLU columns, a block per R1 task
    h_max: torch.Tensor,  # [S, 2I/128, T] fp32: per-tile row |max| of h
    zq: torch.Tensor,  # [S, R, I/256, 128] uint8
    zs: torch.Tensor,  # [S, R, I/256, 16] uint8
    z_amax: torch.Tensor,  # [S, R] fp32
    rows_out: torch.Tensor,  # [T * K, H] bf16: the rows of the routes to this rank's experts (others untouched)
    ppart: torch.Tensor,  # [T, H/256, 256] fp32 symmetric: each token's sum of its rows on this rank
    attention: torch.Tensor,  # [T, H] bf16: the layer's attention output (after attn_sconv)
    partial: torch.Tensor,  # [T, H] bf16: the attn_res partial (residual stream) before this layer
    conv_w: torch.Tensor,  # [H, 4] bf16: mlp_sconv weight
    conv_cache: torch.Tensor,  # [NB * CB * KVH, H / KVH] bf16: the mlp_sconv stream of the layer's paged conv cache,
    #                            row slot * KVH + head
    conv_pages: torch.Tensor,  # [MB] int32: the request's conv-cache block table (position p in block p // CB)
    conv_loc: torch.Tensor,  # [T] int64: the verify rows' conv-cache slots (block * CB + offset; < 0: none)
    seq_len: torch.Tensor,  # [1] int32: the request's length including the T verify rows
    conv_block: hl.constexpr,  # CB: the conv cache's positions a block
    out: torch.Tensor,  # [T, H/V, V] bf16 out: partial + attention + mlp_sconv(combined routed rows) (the next partial)
    res_qidx: torch.Tensor,  # [8] int32: (live blocks, A, query of slot 0 .. QP - 1, 0 ..) (indices into res_dots)
    res_q: torch.Tensor,  # [QP, H/V, V] fp32: the slots' queries, scaled by H^-0.5 (slot 0: the next MLP input's;
    # 1 .. A: the next attention inputs' (A = 1 local: attn_norm, 3 global: q / k / v_input_norm); padding 0)
    res_w: torch.Tensor,  # [QP, H/V, V]: the slots' norm weights (slot 0 unused)
    res_acc: torch.Tensor,  # [QP, T, H/V, V] fp32 out: the blocks' softmax-weighted normalized sum per slot (slot 0:
    # the next attention kernel's B)
    res_stat: torch.Tensor,  # [QP, T, H/F, 4] fp32 out: (their logit max, exp total, res_acc's square sums, 0)
    res_part: torch.Tensor,  # [QP, T, H/C, 4] fp32 workspace: J's (q . x, x . x, res_acc . x, 0) of the new partial x
    x_next: torch.Tensor,  # [A, R, H/V, V] bf16 out: the next attention kernel's input rows (norm(attn_res(...)))
    group_name: hl.ProcessGroupName,
    world: hl.constexpr,
    rank: hl.constexpr,
    gate_cap: hl.constexpr,
    up_cap: hl.constexpr,
    route_scale: hl.constexpr,
) -> torch.Tensor:
    hl.specialize(a_x.size())
    hl.specialize(a_x.stride())
    hl.specialize(a_w_qkvr.size())
    hl.specialize(a_w_qkvr.stride())
    hl.specialize(a_part.size())
    hl.specialize(a_part.stride())
    hl.specialize(a_k_conv_w.size())
    hl.specialize(a_k_conv_w.stride())
    hl.specialize(a_v_conv_w.size())
    hl.specialize(a_v_conv_w.stride())
    hl.specialize(a_k_conv_cache.size())
    hl.specialize(a_k_conv_cache.stride())
    hl.specialize(a_v_conv_cache.size())
    hl.specialize(a_v_conv_cache.stride())
    hl.specialize(a_conv_pages.size())
    hl.specialize(a_conv_pages.stride())
    hl.specialize(a_conv_loc.size())
    hl.specialize(a_conv_loc.stride())
    hl.specialize(a_q_scale.size())
    hl.specialize(a_q_scale.stride())
    hl.specialize(a_r_scale.size())
    hl.specialize(a_r_scale.stride())
    hl.specialize(a_rel_proj32.size())
    hl.specialize(a_rel_proj32.stride())
    hl.specialize(a_k_buf.size())
    hl.specialize(a_k_buf.stride())
    hl.specialize(a_v_buf.size())
    hl.specialize(a_v_buf.stride())
    hl.specialize(a_pages.size())
    hl.specialize(a_pages.stride())
    hl.specialize(a_seq_len.size())
    hl.specialize(a_seq_len.stride())
    hl.specialize(a_kv_loc.size())
    hl.specialize(a_kv_loc.stride())
    hl.specialize(a_q_ws.size())
    hl.specialize(a_q_ws.stride())
    hl.specialize(a_rel_ws.size())
    hl.specialize(a_rel_ws.stride())
    hl.specialize(a_bias_ws.size())
    hl.specialize(a_bias_ws.stride())
    hl.specialize(a_nbias_ws.size())
    hl.specialize(a_nbias_ws.stride())
    hl.specialize(a_att_ws.size())
    hl.specialize(a_att_ws.stride())
    hl.specialize(a_lse_ws.size())
    hl.specialize(a_lse_ws.stride())
    hl.specialize(a_kv_new.size())
    hl.specialize(a_kv_new.stride())
    hl.specialize(a_att_new.size())
    hl.specialize(a_att_new.stride())
    hl.specialize(a_lse_new.size())
    hl.specialize(a_lse_new.stride())
    hl.specialize(a_o_in.size())
    hl.specialize(a_o_in.stride())
    hl.specialize(a_wo.size())
    hl.specialize(a_wo.stride())
    hl.specialize(a_part_o.size())
    hl.specialize(a_part_o.stride())
    hl.specialize(a_opart.size())
    hl.specialize(a_opart.stride())
    hl.specialize(a_a_conv_w.size())
    hl.specialize(a_a_conv_w.stride())
    hl.specialize(a_a_conv_cache.size())
    hl.specialize(a_a_conv_cache.stride())
    hl.specialize(a_res_block.size())
    hl.specialize(a_res_block.stride())
    hl.specialize(a_res_queries.size())
    hl.specialize(a_res_queries.stride())
    hl.specialize(a_res_dpart.size())
    hl.specialize(a_res_dpart.stride())
    hl.specialize(a_res_hist.size())
    hl.specialize(a_res_hist.stride())
    hl.specialize(a_res_rstd.size())
    hl.specialize(a_res_rstd.stride())
    hl.specialize(a_res_dots.size())
    hl.specialize(a_res_dots.stride())
    hl.specialize(a_res_slot.size())
    hl.specialize(a_res_slot.stride())
    hl.specialize(a_res_q.size())
    hl.specialize(a_res_q.stride())
    hl.specialize(a_res_w.size())
    hl.specialize(a_res_w.stride())
    hl.specialize(a_res_acc.size())
    hl.specialize(a_res_acc.stride())
    hl.specialize(a_res_stat.size())
    hl.specialize(a_res_stat.stride())
    hl.specialize(a_res_part.size())
    hl.specialize(a_res_part.stride())
    a_hidden = a_x.size(1)
    a_n_qkvr, a_k_splits, a_k_per = a_w_qkvr.size()
    a_tokens = a_q_ws.size(0)
    a_kv_heads, a_padded, a_head_dim = a_q_ws.size(1), a_q_ws.size(2), a_q_ws.size(3)
    a_heads = a_q_scale.size(0)
    a_group = a_heads // a_kv_heads
    a_d_rel = a_r_scale.size(1)
    a_rel_extent = a_rel_proj32.size(1)
    a_n_pages = a_att_ws.size(1)
    a_rows = (
        a_padded * a_tokens
    )  # attention a_rows of a KV a_group: (token, a_padded head)
    a_q_width = a_heads * a_head_dim
    a_kv_width = a_kv_heads * a_head_dim
    a_r_col = a_q_width
    a_gate_col = a_r_col + a_heads * a_d_rel
    a_sink_col = a_gate_col + a_q_width
    a_k_col = a_sink_col + a_heads
    a_v_col = a_k_col + a_kv_width
    a_o_parts, a_o_split = a_wo.size(1), a_wo.size(2)
    a_taps = a_k_conv_w.size(1)
    a_conv_scale = (a_taps + 1) ** -0.5
    a_x3 = a_x.view(a_x.size(0), a_k_splits, a_k_per)
    a_window_left = a_rel_extent - 1
    a_scale_log2 = a_head_dim**-0.5 * LOG2E
    a_head_cols = a_a_conv_cache.size(1)
    a_opart_peers = torch.ops.symm_mem.get_remote_tensors(a_opart, group_name)
    a_new_keys = a_kv_new.size(2)
    a_live_pages = (
        a_window_left + a_page_size - 1
    ) // a_page_size + 1  # a_pages a 1023-key cached window can touch
    a_chunk = a_opart.size(2)  # all-reduce columns a task
    a_d_k = a_res_hist.size(3)  # D1's K columns a task (a_res_hist's last dim)
    a_d_qp = a_res_dpart.size(2)  # query a_rows a half: highs [0, Qp), lows [Qp, 2 Qp)
    a_n_queries = a_res_dots.size(1)
    a_res_cols = a_res_q.size(1)
    a_stat_rows = (
        1 << (a_res_stat.size(1) - 1).bit_length()
    )  # F2 tasks a row, a_padded to a power of two

    a_att_flat = a_att_ws.view(
        a_att_ws.size(0), a_att_ws.size(1), a_att_ws.size(2) * a_att_ws.size(3)
    )  # M's [a_group, a_page, row * dim]
    res_hist = a_res_hist.view(
        a_res_hist.size(0),
        a_res_hist.size(1),
        a_res_hist.size(2) * a_res_hist.size(3) // res_q.size(2),
        res_q.size(2),
    )
    res_rstd = a_res_rstd
    res_dots = a_res_dots
    hl.specialize(x.size())
    hl.specialize(x.stride())
    hl.specialize(gate_w.size())
    hl.specialize(gate_w.stride())
    hl.specialize(gate_b.size())
    hl.specialize(gate_b.stride())
    hl.specialize(logit_part.size())
    hl.specialize(logit_part.stride())
    hl.specialize(ids.size())
    hl.specialize(ids.stride())
    hl.specialize(gammas.size())
    hl.specialize(gammas.stride())
    hl.specialize(w13.size())
    hl.specialize(w13.stride())
    hl.specialize(w13_s.size())
    hl.specialize(w13_s.stride())
    hl.specialize(w13_amax.size())
    hl.specialize(w13_amax.stride())
    hl.specialize(w13_den.size())
    hl.specialize(w13_den.stride())
    hl.specialize(w2.size())
    hl.specialize(w2.stride())
    hl.specialize(w2_s.size())
    hl.specialize(w2_s.stride())
    hl.specialize(w2_amax.size())
    hl.specialize(w2_amax.stride())
    hl.specialize(w2_den.size())
    hl.specialize(w2_den.stride())
    hl.specialize(xq.size())
    hl.specialize(xq.stride())
    hl.specialize(xs.size())
    hl.specialize(xs.stride())
    hl.specialize(x_amax.size())
    hl.specialize(x_amax.stride())
    hl.specialize(h.size())
    hl.specialize(h.stride())
    hl.specialize(h_max.size())
    hl.specialize(h_max.stride())
    hl.specialize(zq.size())
    hl.specialize(zq.stride())
    hl.specialize(zs.size())
    hl.specialize(zs.stride())
    hl.specialize(z_amax.size())
    hl.specialize(z_amax.stride())
    hl.specialize(rows_out.size())
    hl.specialize(rows_out.stride())
    hl.specialize(ppart.size())
    hl.specialize(ppart.stride())
    hl.specialize(attention.size())
    hl.specialize(attention.stride())
    hl.specialize(partial.size())
    hl.specialize(partial.stride())
    hl.specialize(conv_w.size())
    hl.specialize(conv_w.stride())
    hl.specialize(conv_cache.size())
    hl.specialize(conv_cache.stride())
    hl.specialize(conv_pages.size())
    hl.specialize(conv_pages.stride())
    hl.specialize(conv_loc.size())
    hl.specialize(conv_loc.stride())
    hl.specialize(seq_len.size())
    hl.specialize(seq_len.stride())
    hl.specialize(out.size())
    hl.specialize(out.stride())
    hl.specialize(res_qidx.size())
    hl.specialize(res_qidx.stride())
    hl.specialize(res_q.size())
    hl.specialize(res_q.stride())
    hl.specialize(res_w.size())
    hl.specialize(res_w.stride())
    hl.specialize(res_acc.size())
    hl.specialize(res_acc.stride())
    hl.specialize(res_stat.size())
    hl.specialize(res_stat.stride())
    hl.specialize(res_part.size())
    hl.specialize(res_part.stride())
    hl.specialize(x_next.size())
    hl.specialize(x_next.stride())
    hidden = x.size(1)
    experts = w13.size(0)
    t13, k13 = w13.size(1), w13.size(3)  # 128-row W13 tiles, 256-byte K steps
    t2, k2 = w2.size(1), w2.size(3)  # 128-row W2 tiles, 128-byte K steps
    pairs = ids.size(0)
    topk = 8
    tokens = pairs // topk
    n_tiles, g_steps = gate_w.size(0), gate_w.size(2)
    g_parts = logit_part.size(0)
    g_per = g_steps // g_parts
    router_n = n_tiles * 128
    router_p = logit_part.size(1) * 128  # padded to a power of two: arange sizes
    rows = xq.size(0)
    assert rows == 2 * tokens, "the MMA pads T = 8 token rows to 16"
    slots = h.size(0)
    base = rank * experts
    chunks_p = 1 << (k13 - 1).bit_length()
    taps = conv_w.size(1)
    conv_scale = (taps + 1) ** -0.5
    head_cols = conv_cache.size(1)
    conv_heads = hidden // head_cols
    ppart_peers = torch.ops.symm_mem.get_remote_tensors(ppart, group_name)
    res_slots, res_cols = res_q.size(0), res_q.size(2)
    res_chunk = 256  # J's columns a task
    f_chunk = res_cols  # F's columns a task
    stat_rows = (
        1 << (res_stat.size(2) - 1).bit_length()
    )  # F tasks a row, padded to a power of two

    # The code outgrows the launch prefetch and weights streamed between launches evict it from L2.
    # R2's flat views: W2 rows / zq rows as bytes, W2 scale chunks as [K step x 8, 256], zs rows as bytes.
    w2f = w2.view(w2.size(0), w2.size(1), w2.size(2), w2.size(3) * w2.size(4))
    w2sf = w2_s.view(
        w2_s.size(0), w2_s.size(1), w2_s.size(2) * w2_s.size(3), w2_s.size(4)
    )
    zqf = zq.view(zq.size(0), zq.size(1), zq.size(2) * zq.size(3))
    zsf = zs.view(zs.size(0), zs.size(1), zs.size(2) * zs.size(3))
    hl.prefetch_code()

    # P. Stock: qkvr (bf16 a_x bf16 -> bf16; here fp32 partials by K split, summed in split order by the readers).
    for a_p_n, a_p_s in hl.tile([a_n_qkvr, a_k_splits], block_size=[128, 1]):
        a_p_acc = hl.zeros([a_p_n, a_x.size(0)], dtype=torch.float32)
        for a_p_k in hl.tile(a_k_per, block_size=128):
            a_p_acc = hl.dot(
                a_w_qkvr[a_p_n, a_p_s.begin, a_p_k],
                a_x3[:, a_p_s.begin, a_p_k].permute(1, 0),
                acc=a_p_acc,
            )
        a_part[a_p_s.begin, :, a_p_n] = a_p_acc.permute(1, 0)

    # D. The new block's per-row stats for every later attn_res (the MoE kernels' F read them): its a_rows . every
    # query (bf16 high + low query parts on the tensor cores, fp32 accumulation: ~fp32 dots) and their rstd, by K
    # a_chunk (D1, whose query-high tasks also copy the a_rows into the block history), summed in a_chunk order (D2).
    for a_d_m, a_d_c in hl.tile([2 * a_d_qp, a_hidden // a_d_k], block_size=[128, 1]):
        a_d_r = hl.arange(a_res_dpart.size(3))  # the MMA's a_rows: T, a_padded
        a_d_acc = hl.zeros([a_d_m, a_res_dpart.size(3)], dtype=torch.float32)
        for a_d_kk in hl.tile(a_d_k, block_size=128):
            a_d_xk = hl.load(
                a_res_block,
                [
                    a_d_r[:, None],
                    (a_d_c.begin * a_d_k + a_d_kk.begin + hl.arange(128))[None, :],
                ],
                extra_mask=(a_d_r < a_tokens)[:, None],
            )
            a_d_acc = hl.dot(
                a_res_queries[a_d_m, a_d_c.begin, a_d_kk],
                a_d_xk.permute(1, 0),
                acc=a_d_acc,
            )
        a_res_dpart[
            a_d_c.begin, a_d_m.begin // a_d_qp, a_d_m.begin % a_d_qp + hl.arange(128), :
        ] = a_d_acc
        if a_d_m.begin == 0:
            a_d_x = hl.load(
                a_res_block,
                [a_d_r[:, None], (a_d_c.begin * a_d_k + hl.arange(a_d_k))[None, :]],
                extra_mask=(a_d_r < a_tokens)[:, None],
            ).to(torch.float32)
            a_res_dpart[a_d_c.begin, 2, 0, :] = torch.sum(a_d_x * a_d_x, dim=1)
            a_d_rows = hl.arange(a_tokens)
            a_res_hist[a_res_slot[0], a_d_rows, a_d_c.begin, :] = a_res_block[
                a_d_rows, a_d_c.begin * a_d_k + hl.arange(a_d_k)
            ]

    for a_e_t in hl.tile(a_tokens, block_size=1):
        a_e_dots = torch.sum(a_res_dpart[:, 0, :, a_e_t.begin], dim=0) + torch.sum(
            a_res_dpart[:, 1, :, a_e_t.begin], dim=0
        )
        a_e_i = hl.arange(a_d_qp)
        hl.store(
            a_res_dots,
            [a_res_slot[0], a_e_i, a_e_t.begin],
            a_e_dots,
            extra_mask=a_e_i < a_n_queries,
        )
        a_e_ss = torch.sum(a_res_dpart[:, 2, 0, a_e_t.begin], dim=0)
        a_res_rstd[a_res_slot[0], a_e_t.begin] = torch.rsqrt(
            a_e_ss * (1.0 / a_hidden) + 1e-6
        )

    # PQR. Stock: q_l2_clip (learnable per-(head, dim) scale), bf16, written a_padded (head h -> a_group h // 6, h % 6);
    # rel_l2_clip (scale, threshold 8), bf16, rel_logits_proj (K = 16), bf16.
    for a_q_t, a_q_h in hl.tile([a_tokens, a_heads], block_size=[1, 1]):
        # Every load first (a load after a store does not move above it).
        a_q_cols = a_q_h.begin * a_head_dim + hl.arange(a_head_dim)
        a_r_cols = a_r_col + a_q_h.begin * a_d_rel + hl.arange(a_d_rel)
        a_q_sc = a_q_scale[a_q_h.begin, :].to(torch.float32)
        a_r_sc = a_r_scale[a_q_h.begin, :].to(torch.float32)
        a_r_proj = a_rel_proj32[:, :]
        a_q_raw = (
            torch.sum(a_part[:, a_q_t.begin, a_q_cols], dim=0)
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        a_r_raw = (
            torch.sum(a_part[:, a_q_t.begin, a_r_cols], dim=0)
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        a_q_in = a_q_sc * a_q_raw
        a_q_ws[a_q_t.begin, a_q_h.begin // a_group, a_q_h.begin % a_group, :] = (
            _l2_clip(a_q_in[None, :], 22.627416997969522, 1e-6)
            .to(torch.bfloat16)
            .reshape(a_head_dim)
        )
        a_r_in = a_r_sc * a_r_raw
        a_r_bf = (
            _l2_clip(a_r_in[None, :], 8.0, 1e-6).to(torch.bfloat16).to(torch.float32)
        )
        a_rel = torch.sum(a_r_bf.reshape(a_d_rel)[:, None] * a_r_proj, dim=0).to(
            torch.bfloat16
        )
        # The biases as A / AN read them: rel distance e biases window key x = cache_end + t - window_base - e (A,
        # cached keys: e > t) and new key t - e (AN: e <= t); every other key of the row gets -inf (disjoint masks).
        a_q_e = hl.arange(a_rel_extent)
        a_q_row = a_q_t.begin * a_padded + a_q_h.begin % a_group
        a_q_g = a_q_h.begin // a_group
        a_q_cend = a_seq_len[0] - a_tokens
        a_q_base = (
            torch.clamp_min(a_q_cend - a_window_left, 0) // a_page_size * a_page_size
        )
        a_q_x = a_q_cend + a_q_t.begin - a_q_base - a_q_e
        hl.store(
            a_bias_ws,
            [a_q_g, a_q_row, a_q_x],
            a_rel,
            extra_mask=(a_q_e > a_q_t.begin) & (a_q_x >= 0),
        )
        a_q_keys = hl.arange(a_live_pages * a_page_size)
        hl.store(
            a_bias_ws,
            [a_q_g, a_q_row, a_q_keys],
            hl.full([a_live_pages * a_page_size], float("-inf"), dtype=torch.bfloat16),
            extra_mask=(a_q_keys >= a_q_cend - a_q_base)
            | (a_q_keys < a_q_cend + a_q_t.begin - a_q_base - a_window_left),
        )
        hl.store(
            a_nbias_ws,
            [a_q_g, a_q_row, a_q_t.begin - a_q_e],
            a_rel,
            extra_mask=a_q_e <= a_q_t.begin,
        )
        a_q_new = hl.arange(a_new_keys)
        hl.store(
            a_nbias_ws,
            [a_q_g, a_q_row, a_q_new],
            hl.full([a_new_keys], float("-inf"), dtype=torch.bfloat16),
            extra_mask=a_q_new > a_q_t.begin,
        )

    # PK. Stock: attention_prologue's K / V verify sconv (static weight, residual, (W + 1)^-0.5, window save),
    # K L2 clip, V RMSNorm (no gamma), bf16 at every step. The a_rows go to a_kv_new (M stores them into the cache).
    for a_k_g, a_k_t in hl.tile([a_kv_heads, a_tokens], block_size=[1, 1]):
        a_k_end = a_seq_len[0] - a_tokens
        a_kv_cols = a_k_g.begin * a_head_dim + hl.arange(a_head_dim)
        a_k_conv = _verify_conv(
            a_part,
            a_k_col,
            a_kv_cols,
            a_k_t.begin,
            a_k_conv_w,
            a_k_conv_cache,
            a_k_g.begin,
            a_kv_heads,
            a_conv_pages,
            a_conv_loc,
            a_conv_block,
            a_k_end,
            a_taps,
        )
        a_k_new = _l2_clip(a_k_conv[None, :], 22.627416997969522, 1e-6).to(
            torch.bfloat16
        )
        a_kv_new[0, a_k_g.begin, a_k_t.begin, :] = a_k_new.reshape(a_head_dim)

    # PV. The V half of PK (its own tasks: the K and V convs run side by side).
    for a_v_g, a_v_t in hl.tile([a_kv_heads, a_tokens], block_size=[1, 1]):
        a_v_end = a_seq_len[0] - a_tokens
        a_vv_cols = a_v_g.begin * a_head_dim + hl.arange(a_head_dim)
        a_v_conv = _verify_conv(
            a_part,
            a_v_col,
            a_vv_cols,
            a_v_t.begin,
            a_v_conv_w,
            a_v_conv_cache,
            a_v_g.begin,
            a_kv_heads,
            a_conv_pages,
            a_conv_loc,
            a_conv_block,
            a_v_end,
            a_taps,
        )
        a_v_rstd = _rsqrt_approx(
            torch.sum(a_v_conv * a_v_conv, dim=0) * (1.0 / a_head_dim) + 1e-6
        )
        a_kv_new[1, a_v_g.begin, a_v_t.begin, :] = (a_v_conv * a_v_rstd).to(
            torch.bfloat16
        )

    # A. Stock: FA4 over the window (a_keys [pos - 1023, pos] of each query row), the a_rel a_bias added unscaled
    # (rel_bias_scale_after_add: q.k * scale + a_rel);
    # one task per (KV a_group, 128-key a_page): the (token, a_padded head) a_rows' softmax over the a_page's cached a_keys
    # (positions before the new a_rows, which AN covers).
    for a_a_g, a_a_p in hl.tile([a_kv_heads, a_live_pages], block_size=[1, 1]):
        a_length = a_seq_len[0]
        a_cache_end = a_length - a_tokens
        a_first_page = torch.clamp_min(a_cache_end - a_window_left, 0) // a_page_size
        a_page_idx = a_first_page + a_a_p.begin
        # The a_page's cached a_keys first: nothing in this launch writes them before A (M stores the new a_rows after),
        # so they load ahead of the wait for PQR's q / bias.
        a_a_live = a_page_idx * a_page_size < a_cache_end
        a_page = hl.load(a_pages, [a_page_idx], extra_mask=a_a_live).to(torch.int64)
        a_keys = hl.arange(a_page_size)
        a_a_on = (
            hl.zeros([a_page_size], dtype=torch.int32) + a_a_live.to(torch.int32)
        ) != 0
        a_slots = a_page * a_page_size + a_keys
        a_kk = hl.load(
            a_k_buf, [a_slots, a_a_g.begin, slice(None)], extra_mask=a_a_on[:, None]
        )
        a_vv = hl.load(
            a_v_buf, [a_slots, a_a_g.begin, slice(None)], extra_mask=a_a_on[:, None]
        )
        if a_a_live:
            a_qq = a_q_ws[:, a_a_g.begin, :, :].reshape(a_rows, a_head_dim)
            # PQR's masked bias (-inf where the row may not see the key): the page's keys are window keys
            # p * BS + key (the window starts at a_first_page).
            a_bias = a_bias_ws[a_a_g.begin, :, a_a_p.begin * a_page_size + a_keys]
            a_scores = hl.dot(a_qq, a_kk.permute(1, 0), out_dtype=torch.float32)
            a_logits = a_scores * a_scale_log2 + a_bias.to(torch.float32) * LOG2E
            a_run_max = torch.amax(a_logits, dim=1)
            a_safe_max = torch.where(a_run_max > float("-inf"), a_run_max, 0.0)
            a_prob = torch.exp2(a_logits - a_safe_max[:, None])
            a_run_sum = torch.sum(a_prob, dim=1)
            a_pv = hl.dot(a_prob.to(torch.bfloat16), a_vv, out_dtype=torch.float32)
            a_seen = a_run_sum > 0.0
            a_att_ws[a_a_g.begin, a_a_p.begin, :, :] = (
                a_pv / torch.where(a_seen, a_run_sum, 1.0)[:, None]
            )
            a_lse_ws[a_a_g.begin, a_a_p.begin, :] = torch.where(
                a_seen, a_safe_max + torch.log2(a_run_sum), float("-inf")
            )
        else:  # a a_page past the cached window: weight 0 in M
            a_lse_ws[a_a_g.begin, a_a_p.begin, :] = hl.full(
                [a_rows], float("-inf"), dtype=torch.float32
            )

    # AN. The same over the T new a_keys of each KV a_group (a_kv_new; every one is inside every row's window).
    for a_n_g in hl.tile(a_kv_heads, block_size=1):
        # q and the masked bias (PQR's) before K / V (PK / PV's): their loads overlap the later wait.
        a_nq = a_q_ws[:, a_n_g.begin, :, :].reshape(a_rows, a_head_dim)
        a_n_bias = a_nbias_ws[a_n_g.begin, :, :]
        a_nk = a_kv_new[0, a_n_g.begin, :, :]
        a_nv = a_kv_new[1, a_n_g.begin, :, :]
        a_n_scores = hl.dot(a_nq, a_nk.permute(1, 0), out_dtype=torch.float32)
        a_n_logits = a_n_scores * a_scale_log2 + a_n_bias.to(torch.float32) * LOG2E
        a_n_max = torch.amax(a_n_logits, dim=1)
        a_n_safe = torch.where(a_n_max > float("-inf"), a_n_max, 0.0)
        a_n_prob = torch.exp2(a_n_logits - a_n_safe[:, None])
        a_n_sum = torch.sum(a_n_prob, dim=1)
        a_n_pv = hl.dot(a_n_prob.to(torch.bfloat16), a_nv, out_dtype=torch.float32)
        a_n_seen = a_n_sum > 0.0
        a_att_new[a_n_g.begin, :, :] = (
            a_n_pv / torch.where(a_n_seen, a_n_sum, 1.0)[:, None]
        )
        a_lse_new[a_n_g.begin, :] = torch.where(
            a_n_seen, a_n_safe + torch.log2(a_n_sum), float("-inf")
        )

    # M. Stock: FA4's split merge (natural-log a_lse), then gate_sink: attention * sigmoid(a_gate) * sigmoid(a_lse - a_sink), bf16.
    # One task per (token, head) (vLLM's combine grid): every a_page's partial in one load (no serial a_page chain).
    # A partial of weight 0 is still read (finite: A only writes normalized partials into the zeroed workspace).
    for a_m_t, a_m_h in hl.tile([a_tokens, a_heads], block_size=[1, 1]):
        a_m_g = a_m_h.begin // a_group
        a_m_r = (
            a_m_t.begin * a_padded + a_m_h.begin % a_group
        )  # the head's attention row in its a_group
        a_m_dims = hl.arange(a_head_dim)
        a_m_pages = hl.arange(a_n_pages)
        a_m_loc = a_kv_loc[a_m_t.begin]
        a_m_kn = a_kv_new[0, a_m_g, a_m_t.begin, :]
        a_m_vn = a_kv_new[1, a_m_g, a_m_t.begin, :]
        a_gate = torch.sum(
            hl.load(
                a_part,
                [
                    slice(None),
                    a_m_t.begin,
                    a_gate_col + a_m_h.begin * a_head_dim + a_m_dims,
                ],
            ),
            dim=0,
        )
        a_gate = a_gate.to(torch.bfloat16).to(torch.float32)
        a_sink = torch.sum(
            hl.load(
                a_part,
                [slice(None), a_m_t.begin, a_sink_col + a_m_h.begin + hl.arange(1)],
            ),
            dim=0,
        )
        a_sink = a_sink.to(torch.bfloat16).to(torch.float32)
        a_m_lse = torch.where(
            a_m_pages < a_live_pages,
            hl.load(
                a_lse_ws, [a_m_g, a_m_pages, a_m_r], extra_mask=a_m_pages < a_live_pages
            ),
            float("-inf"),
        )  # [a_pages] (A writes the a_live_pages a window can touch)
        a_n_lse = hl.load(a_lse_new, [a_m_g, a_m_r + hl.arange(1)])  # [1]
        a_top = torch.maximum(torch.amax(a_m_lse, dim=0), a_n_lse)
        a_top = torch.where(a_top > float("-inf"), a_top, 0.0)
        a_n_weight = torch.exp2(a_n_lse - a_top)
        a_p_weight = torch.exp2(a_m_lse - a_top)
        a_total = torch.sum(a_p_weight, dim=0) + a_n_weight
        a_att = hl.load(
            a_att_flat,
            [a_m_g, a_m_pages[:, None], a_m_r * a_head_dim + a_m_dims[None, :]],
            extra_mask=(a_m_pages < a_live_pages)[:, None],
        )
        a_merged = (
            torch.sum(a_att * a_p_weight[:, None], dim=0)
            + hl.load(a_att_new, [a_m_g, a_m_r, a_m_dims]) * a_n_weight
        )
        a_merged = a_merged / torch.where(a_total > 0.0, a_total, 1.0)
        a_lse = (a_top + torch.log2(a_total)) * LN2
        a_gated = a_merged * _sigmoid(a_gate)
        a_gated = a_gated * _sigmoid(a_lse - a_sink)
        a_m_col0 = (
            a_m_h.begin * a_head_dim
        )  # (a_head_dim divides a_o_split: a head's columns lie in one K quarter)
        hl.store(
            a_o_in,
            [a_m_t.begin, a_m_col0 // a_o_split, a_m_col0 % a_o_split + a_m_dims],
            a_gated.to(torch.bfloat16),
        )
        # The new a_rows into the cache (the a_group's first head), after every A task read it (M waits for them).
        a_m_ok = ((hl.zeros([a_head_dim], dtype=torch.int64) + a_m_loc) >= 0) & (
            (hl.zeros([a_head_dim], dtype=torch.int64) + a_m_h.begin % a_group) == 0
        )
        hl.store(a_k_buf, [a_m_loc, a_m_g, a_m_dims], a_m_kn, extra_mask=a_m_ok)
        hl.store(a_v_buf, [a_m_loc, a_m_g, a_m_dims], a_m_vn, extra_mask=a_m_ok)

    # O. Stock: wo_ud (bf16; here fp32 partials by K quarter, summed in order by C).
    for a_o_n, a_o_q in hl.tile([a_hidden, a_o_parts], block_size=[128, 1]):
        a_o_acc = hl.zeros([a_o_n, a_x.size(0)], dtype=torch.float32)
        for a_o_k in hl.tile(a_o_split, block_size=128):
            a_o_acc = hl.dot(
                a_wo[a_o_n, a_o_q.begin, a_o_k],
                a_o_in[:, a_o_q.begin, a_o_k].permute(1, 0),
                acc=a_o_acc,
            )
        a_part_o[a_o_q.begin, :, a_o_n] = a_o_acc.permute(1, 0)

    # C. The bf16 a_wo row of each (token, 128 columns) into the symmetric buffer: one dense tile per task, so the
    # peers poll it in-band (no system-scope release).
    for a_c_t, a_c_n in hl.tile([a_tokens, a_hidden // a_chunk], block_size=[1, 1]):
        a_c_cols = a_c_n.begin * a_chunk + hl.arange(a_chunk)
        a_opart[a_c_t.begin, a_c_n.begin, :] = (
            (
                (
                    a_part_o[0, a_c_t.begin, a_c_cols]
                    + a_part_o[1, a_c_t.begin, a_c_cols]
                )
                + a_part_o[2, a_c_t.begin, a_c_cols]
            )
            + a_part_o[3, a_c_t.begin, a_c_cols]
        ).to(torch.bfloat16)

    # J. Stock: the push all-reduce (rank-order fp32 sum of the bf16 a_rows, bf16) fused with attn_sconv (residual,
    # (W + 1)^-0.5, bf16; a_taps before the verify a_rows from the paged conv cache, zero before position 0; each row's
    # conv input stored in the cache). Each peer's a_rows t - 3 .. t are polled as one tile (a_rows before the window
    # read row 0, polled anyway, and take the cache).
    for a_j_t, a_j_n in hl.tile([a_tokens, a_hidden // a_chunk], block_size=[1, 1]):
        a_j_cols = a_j_n.begin * a_chunk + hl.arange(a_chunk)
        # (a task's columns lie in one row of attention's view: a_chunk divides a_res_cols)
        a_j_col0 = a_j_n.begin * a_chunk
        a_j_vcol = a_j_col0 % a_res_cols + hl.arange(a_chunk)
        # B's inputs first (not written by the peers: their loads overlap the poll).
        a_j_q = a_res_q[a_j_col0 // a_res_cols, a_j_vcol]
        a_j_a = a_res_acc[a_j_t.begin, a_j_col0 // a_res_cols, a_j_vcol]
        a_j_b = a_res_block[a_j_t.begin, a_j_cols].to(torch.float32)
        a_j_head = (
            a_j_col0 // a_head_cols
        )  # (a_chunk divides the stream's columns a head)
        a_j_hcol = a_j_col0 % a_head_cols + hl.arange(a_chunk)
        a_j_tap = hl.arange(a_taps)
        a_j_src = a_j_t.begin - (a_taps - 1) + a_j_tap
        a_j_in = a_j_src >= 0
        a_j_rows = torch.clamp_min(a_j_src, 0)
        # The conv a_taps' history and weights (no peer writes them) also load ahead of the poll.
        a_j_pos = a_seq_len[0] - a_tokens + a_j_src  # the a_taps' absolute positions
        a_j_hist = (~a_j_in) & (a_j_pos >= 0)
        a_j_safe = torch.clamp_min(a_j_pos, 0)
        a_j_blk = hl.load(a_conv_pages, [a_j_safe // a_conv_block], extra_mask=a_j_hist)
        a_j_crow = (
            a_j_blk * a_conv_block + a_j_safe % a_conv_block
        ) * a_kv_heads + a_j_head
        a_j_prior = hl.load(
            a_a_conv_cache,
            [a_j_crow[:, None], a_j_hcol[None, :]],
            extra_mask=a_j_hist[:, None],
        ).to(torch.float32)
        a_j_w = a_a_conv_w[a_j_cols, :].to(torch.float32).permute(1, 0)
        a_j_loc = a_conv_loc[a_j_t.begin]
        a_reduced = hl.zeros([a_taps, a_chunk], dtype=torch.float32)
        for a_peer_part in a_opart_peers:
            a_reduced = a_reduced + a_peer_part[a_j_rows, a_j_n.begin, :].to(
                torch.float32
            )
        a_reduced = a_reduced.to(torch.bfloat16).to(torch.float32)
        a_j_value = torch.where(a_j_in[:, None], a_reduced, a_j_prior)
        a_j_acc = torch.sum(a_j_value * a_j_w, dim=0)
        a_j_current = torch.sum(
            torch.where((a_j_tap == a_taps - 1)[:, None], a_j_value, 0.0), dim=0
        )
        hl.store(
            a_a_conv_cache,
            [a_j_loc * a_kv_heads + a_j_head, a_j_hcol],
            a_j_current.to(torch.bfloat16),
            extra_mask=(hl.zeros([a_chunk], dtype=torch.int64) + a_j_loc) >= 0,
        )
        a_j_out = ((a_j_acc + a_j_current) * a_conv_scale).to(torch.bfloat16)
        attention[a_j_t.begin, a_j_cols] = a_j_out
        # B's sums over the two new attn_res sources (the block b and the attention a_rows a_x; its share of the columns)
        # as one [8, a_chunk] reduction: (q . a_x, a_x . a_x, a_res_acc . a_x, q . b, b . b, a_res_acc . b, b . a_x, 0).
        a_j_x = a_j_out.to(torch.float32)
        a_j_i = hl.arange(8)[:, None]
        a_j_left = torch.where(a_j_i < 3, a_j_x[None, :], a_j_b[None, :])
        a_j_right = torch.where(
            (a_j_i == 0) | (a_j_i == 3),
            a_j_q[None, :],
            torch.where(
                (a_j_i == 1) | (a_j_i == 6),
                a_j_x[None, :],
                torch.where(
                    (a_j_i == 2) | (a_j_i == 5),
                    a_j_a[None, :],
                    torch.where(a_j_i == 4, a_j_b[None, :], 0.0),
                ),
            ),
        )
        a_res_part[a_j_t.begin, a_j_n.begin, :] = torch.sum(a_j_left * a_j_right, dim=1)

    # B. Stock: block_attn_res over the blocks and this layer's attention a_rows (normalize_values: softmax over the
    # sources of (q . x_n) * rstd(x_n), the rstd-normalized sources summed, fp32) + mlp_norm (RMSNorm, eps 1e-6,
    # bf16) of the MoE input, per (token, J's columns): the new block and the attention a_rows join the older blocks'
    # softmax (a_res_acc, a_res_stat: the previous MoE kernel's F); the sum's square from J's and F's sums.
    for a_b_t, a_b_n in hl.tile([a_tokens, a_hidden // a_chunk], block_size=[1, 1]):
        # Inputs J does not write first (their loads overlap the wait for J).
        a_b_col0 = a_b_n.begin * a_chunk
        a_b_v = a_b_col0 // a_res_cols
        a_b_cols = a_b_col0 % a_res_cols + hl.arange(a_chunk)
        a_b_acc = a_res_acc[a_b_t.begin, a_b_v, a_b_cols]
        a_b_w = a_res_w[a_b_v, a_b_cols].to(torch.float32)
        a_b_b = a_res_block[a_b_t.begin, a_b_col0 + hl.arange(a_chunk)].to(
            torch.float32
        )
        a_b_sums = torch.sum(
            a_res_part[a_b_t.begin, :, :], dim=0
        )  # in J's column order
        a_b_i = hl.arange(8)
        a_b_dx = torch.sum(torch.where(a_b_i == 0, a_b_sums, 0.0), dim=0)
        a_b_sx = torch.sum(torch.where(a_b_i == 1, a_b_sums, 0.0), dim=0)
        a_b_ax = torch.sum(torch.where(a_b_i == 2, a_b_sums, 0.0), dim=0)
        a_b_db = torch.sum(torch.where(a_b_i == 3, a_b_sums, 0.0), dim=0)
        a_b_sb = torch.sum(torch.where(a_b_i == 4, a_b_sums, 0.0), dim=0)
        a_b_ab = torch.sum(torch.where(a_b_i == 5, a_b_sums, 0.0), dim=0)
        a_b_bx = torch.sum(torch.where(a_b_i == 6, a_b_sums, 0.0), dim=0)
        # F2's tasks agree on the max and a_total; the square sums add in column order.
        a_b_rows = hl.arange(a_stat_rows)[:, None]
        a_b_cols4 = hl.arange(4)[None, :]
        a_b_stat = hl.load(
            a_res_stat,
            [a_b_t.begin, a_b_rows, a_b_cols4],
            extra_mask=a_b_rows < a_res_stat.size(1),
        )
        a_b_live = a_b_rows < a_res_stat.size(1)
        a_b_max = torch.amax(
            torch.amax(
                torch.where(a_b_live & (a_b_cols4 == 0), a_b_stat, float("-inf")), dim=1
            ),
            dim=0,
        )
        a_b_total = torch.amax(
            torch.amax(
                torch.where(a_b_live & (a_b_cols4 == 1), a_b_stat, float("-inf")), dim=1
            ),
            dim=0,
        )
        a_b_sq = torch.sum(
            torch.sum(torch.where(a_b_cols4 == 2, a_b_stat, 0.0), dim=1), dim=0
        )
        a_b_rx = torch.rsqrt(a_b_sx * (1.0 / a_hidden) + 1e-6)
        a_b_rb = torch.rsqrt(a_b_sb * (1.0 / a_hidden) + 1e-6)
        a_b_lx = a_b_dx * a_b_rx
        a_b_lb = a_b_db * a_b_rb
        a_b_top = torch.maximum(torch.maximum(a_b_max, a_b_lb), a_b_lx)
        a_b_old = torch.exp(a_b_max - a_b_top)
        a_b_eb = torch.exp(a_b_lb - a_b_top)
        a_b_ex = torch.exp(a_b_lx - a_b_top)
        a_b_inv = 1.0 / (a_b_total * a_b_old + a_b_eb + a_b_ex)
        a_b_ca = a_b_old * a_b_inv
        a_b_cb = a_b_eb * a_b_rb * a_b_inv
        a_b_cx = a_b_ex * a_b_rx * a_b_inv
        a_b_hh = (
            a_b_ca * a_b_ca * a_b_sq
            + a_b_cb * a_b_cb * a_b_sb
            + a_b_cx * a_b_cx * a_b_sx
            + 2.0
            * (
                a_b_ca * a_b_cb * a_b_ab
                + a_b_ca * a_b_cx * a_b_ax
                + a_b_cb * a_b_cx * a_b_bx
            )
        )
        a_b_norm = torch.rsqrt(a_b_hh * (1.0 / a_hidden) + 1e-6)
        a_b_x = attention[a_b_t.begin, a_b_col0 + hl.arange(a_chunk)].to(torch.float32)
        x[a_b_t.begin, a_b_col0 + hl.arange(a_chunk)] = (
            (a_b_acc * a_b_ca + a_b_b * a_b_cb + a_b_x * a_b_cx) * a_b_norm * a_b_w
        ).to(torch.bfloat16)

    # G1. Stock: Inkling2Gate's linear_k_group_partials (bf16 x bf16 -> fp32 router logits by K part).
    for g_e, g_s in hl.tile([n_tiles, g_parts], block_size=[1, 1]):
        g_acc = hl.zeros([128, rows], dtype=torch.float32)
        for g_k in hl.tile(g_per, block_size=1):
            g_lhs = gate_w[g_e.begin, :, g_s.begin * g_per + g_k.begin, :]
            g_rhs = x[:, (g_s.begin * g_per + g_k.begin) * 128 + hl.arange(128)]
            g_acc = hl.dot(g_lhs, g_rhs.permute(1, 0), acc=g_acc)
        logit_part[g_s.begin, g_e.begin, :, :] = g_acc.permute(1, 0)

    # F. Stock: block_attn_res (normalize_values: softmax over the sources of (q . x_n) * rstd(x_n), the
    # rstd-normalized sources summed, fp32) of the next layer's attention input (B adds the newest source, this
    # kernel's output) and MLP input (the next attention kernel's B adds its newest), here over the blocks, from
    # their rows' dots and rstd (res_dots, res_rstd): each query's softmax max-shifted online in block order (the
    # same in every task), the weighted sum over the task's columns. Block 0 is always live.
    for g_t, g_c in hl.tile([tokens, hidden // f_chunk], block_size=[1, 1]):
        g_col0 = g_c.begin * f_chunk
        g_v = g_col0 // res_cols
        g_cols = g_col0 % res_cols + hl.arange(f_chunk)
        g_blocks = res_qidx[0]
        g_q = hl.load(res_qidx, [2 + hl.arange(res_slots)])
        # A finite floor for the online max (an exact -inf start makes the masked-value analysis compute -inf - -inf);
        # block 0 is always live, so the floor never survives.
        g_max = hl.full([res_slots], -1e30, dtype=torch.float32)
        g_total = hl.zeros([res_slots], dtype=torch.float32)
        g_acc = hl.zeros([res_slots, f_chunk], dtype=torch.float32)
        for g_b in hl.static_range(res_hist.size(0)):
            g_live = (hl.zeros([f_chunk], dtype=torch.int32) + g_b) < g_blocks
            g_x = hl.load(
                res_hist, [g_b, g_t.begin, g_v, g_cols], extra_mask=g_live
            ).to(torch.float32)
            g_rstd = res_rstd[g_b, g_t.begin]
            g_on = (hl.zeros([res_slots], dtype=torch.int32) + g_b) < g_blocks
            g_logit = torch.where(
                g_on, hl.load(res_dots, [g_b, g_q, g_t.begin]) * g_rstd, -1e30
            )
            g_top = torch.maximum(g_max, g_logit)
            g_scale = torch.exp(g_max - g_top)
            g_ex = torch.exp(g_logit - g_top)
            g_total = g_total * g_scale + g_ex
            g_acc = g_acc * g_scale[:, None] + g_x[None, :] * (g_ex * g_rstd)[:, None]
            g_max = g_top
        g_sl = hl.arange(res_slots)
        res_acc[g_sl, g_t.begin, g_v, g_cols] = g_acc
        g_j = hl.arange(4)[None, :]
        res_stat[g_sl, g_t.begin, g_c.begin, hl.arange(4)] = torch.where(
            g_j == 0,
            g_max[:, None],
            torch.where(
                g_j == 1,
                g_total[:, None],
                torch.where(g_j == 2, torch.sum(g_acc * g_acc, dim=1)[:, None], 0.0),
            ),
        )

    # G2. Stock: route_topk's selection (K parts summed in order, biased sigmoid, stable top-8 by key desc / id asc).
    for g2_t in hl.tile(tokens, block_size=1):
        logits = torch.sum(logit_part[:, :, g2_t.begin, :], dim=0).reshape(router_p)
        eids = hl.arange(router_p)
        bias = hl.load(gate_b, [eids], extra_mask=eids < router_n)
        ids[g2_t.begin * topk + hl.arange(topk)] = _select(
            logits, bias, router_n, router_p
        )

    # GW. route_topk's logsigmoid-renormalized weights x route_scale, apart from the selection: the expert weights
    # stream from the ids while it runs (only R1's epilogue reads the weights). The logits as G2 sums them.
    for gw_t in hl.tile(tokens, block_size=1):
        w_logits = torch.sum(logit_part[:, :, gw_t.begin, :], dim=0).reshape(router_p)
        w_ids = hl.load(ids, [gw_t.begin * topk + hl.arange(topk)])
        gammas[gw_t.begin * topk + hl.arange(topk)] = _renorm(
            w_logits, w_ids, router_p, route_scale
        )

    # Q. Stock: routed_quant quantize (four-over-six NVFP4 of each routed token row); one task per (token row,
    # 512-column chunk = one R1 K step), each taking the row amax over the full row.
    for q_t, q_c in hl.tile([tokens, k13], block_size=[1, 1]):
        chunk = hl.arange(chunks_p)[:, None]
        xrow = hl.load(
            x,
            [q_t.begin, chunk * 512 + hl.arange(512)[None, :]],
            extra_mask=chunk < k13,
        )
        amax = torch.amax(
            torch.amax(torch.abs(xrow.to(torch.float32)), dim=-1), dim=-1
        ) + hl.zeros([1], dtype=torch.float32)
        xc = x[q_t.begin, q_c.begin * 512 + hl.arange(512)].to(torch.float32)
        packed, scale = quant46(xc.reshape(1, 32, 16), amax)
        xq[q_t.begin, q_c.begin, hl.arange(256)] = packed.reshape(256).to(torch.uint8)
        xs[q_t.begin, q_c.begin, hl.arange(32)] = scale.reshape(32).to(torch.uint8)
        if q_c.begin == 0:
            x_amax[q_t.begin + hl.arange(1)] = amax

    # R1. Stock: routed_swap_matmul GEMM1 + swiglu (scale, bf16 gate/up, capped SwiGLU x gamma, bf16).
    for r1_q, r1_j in hl.tile([t13, slots], block_size=[1, 1]):
        lids = _local_ids(hl.load(ids, [hl.arange(pairs)]), base, experts)
        words, counts, before, distinct = _distinct_table(lids)
        if r1_j.begin < distinct:
            e = _expert_of(words, counts, before, r1_j.begin)
            # The epilogue's operands load ahead of the K loop, under its weight stream (a load issued after the
            # accumulator is ready waits a whole loaded-HBM round trip).
            scale = (x_amax[hl.arange(rows)] * w13_amax[e]) * rcp_rn(
                1536.0 * w13_den[0]
            )
            # gamma of (token row, expert): the routing weight where the token routes to it, else 0
            slot = hl.arange(rows)[:, None] * topk + hl.arange(topk)[None, :]
            live = slot < pairs
            rid = hl.load(ids, [slot], extra_mask=live)
            rg = hl.load(gammas, [slot], extra_mask=live)
            gamma = torch.sum(torch.where((rid == e + base) & live, rg, 0.0), dim=-1)
            acc = hl.zeros([128, rows], dtype=torch.float32)
            for r1_k in hl.tile(k13, block_size=1):
                lhs = w13[e, r1_q.begin, :, r1_k.begin, :]
                lhs_s = _scale_tile(w13_s[e, r1_q.begin, r1_k.begin, :, :], 8)
                rhs = xq[:, r1_k.begin, :]
                rhs_s = xs[:, r1_k.begin, :]
                acc = hl.dot_scaled(
                    lhs,
                    lhs_s.view(E4M3),
                    "e2m1",
                    rhs.permute(1, 0),
                    rhs_s.view(E4M3),
                    "e2m1",
                    acc=acc,
                    out_dtype=torch.float32,
                )
            # The tile's rows in checkpoint order (stored row 8 b + a of each 32 is row 4 a + b), then (gate, up) pairs.
            y = (
                (acc * scale[None, :])
                .reshape(4, 4, 8, rows)
                .permute(0, 2, 1, 3)
                .reshape(64, 2, rows)
                .permute(0, 2, 1)
            )
            gate, up = hl.split(y)
            gate = gate.to(torch.bfloat16).to(torch.float32)
            up = up.to(torch.bfloat16).to(torch.float32)
            v = capped_swiglu(gate, up, gamma[None, :], gate_cap, up_cap).to(
                torch.bfloat16
            )
            # The live token rows as one contiguous block per task, so the Z tasks reading them count its arrivals.
            # (T = 8: the first half of the MMA's 16 rows; the host checks rows == 2 * T)
            v_t, v_pad = hl.split(
                v.permute(1, 0).reshape(2, tokens, 64).permute(1, 2, 0)
            )
            h[r1_j.begin, r1_q.begin, :, :] = v_t
            h_max[r1_j.begin, r1_q.begin, :] = torch.amax(
                torch.abs(v_t.to(torch.float32)), dim=1
            )

    # Z. Stock: swiglu_quantize_routed's four-over-six quantization of the bf16 SwiGLU row; one task per
    # (distinct local expert, 256-column chunk = one R2 K step) over all token rows.
    for z_c, z_j in hl.tile([k2, slots], block_size=[1, 1]):
        lids2 = _local_ids(hl.load(ids, [hl.arange(pairs)]), base, experts)
        words2, counts2, before2, distinct2 = _distinct_table(lids2)
        if z_j.begin < distinct2:
            # Only the live token rows: the MMA pad rows of zq / zs stay zero.
            # The h tile loads before the h_max reduction (its barrier would otherwise order the two loads).
            hv = (
                h[z_j.begin, z_c.begin * 4 + hl.arange(4), :, :]
                .permute(1, 0, 2)
                .reshape(tokens, 256)
            )
            amax2 = torch.amax(h_max[z_j.begin, :, :], dim=0)
            packed2, scale2 = quant46s(
                hv.to(torch.float32).reshape(tokens, 16, 16), amax2
            )
            hl.store(
                zq,
                [z_j.begin, hl.arange(tokens), z_c.begin, hl.arange(128)],
                packed2.reshape(tokens, 128).to(torch.uint8),
            )
            hl.store(
                zs,
                [z_j.begin, hl.arange(tokens), z_c.begin, hl.arange(16)],
                scale2.to(torch.uint8),
            )
            if z_c.begin == 0:
                hl.store(z_amax, [z_j.begin, hl.arange(tokens)], amax2)

    # R2. Stock: GEMM2 (W2) + routed_rows (scale, bf16 rows of the routed tokens).
    for r2_o, r2_j in hl.tile([t2, slots], block_size=[1, 1]):
        lids3 = _local_ids(hl.load(ids, [hl.arange(pairs)]), base, experts)
        words3, counts3, before3, distinct3 = _distinct_table(lids3)
        if r2_j.begin < distinct3:
            e2 = _expert_of(words3, counts3, before3, r2_j.begin)
            # The epilogue's operands load ahead of the K loop (as R1's); each token routes an expert at most once.
            scale2 = (z_amax[r2_j.begin, hl.arange(rows)] * w2_amax[e2]) * rcp_rn(
                1536.0 * w2_den[0]
            )
            slot2 = hl.arange(rows)[:, None] * topk + hl.arange(topk)[None, :]
            live2 = slot2 < pairs
            rid2 = hl.load(ids, [slot2], extra_mask=live2)
            match = ((rid2 == e2 + base) & live2).to(torch.int32)
            route = torch.sum(match * slot2, dim=-1)
            routed = torch.sum(match, dim=-1) > 0
            acc2 = hl.zeros([128, rows], dtype=torch.float32)
            # 256-byte K steps over the flat 1152-byte rows (5 steps: the last box's second half lies past the
            # end and the TMA fills it with zeros: exact zero products), so a ring slot carries R1's bytes.
            for r2_k in hl.tile(w2f.size(3), block_size=256):
                lhs2 = w2f[e2, r2_o.begin, :, r2_k]
                lhs2_s = _scale_tile(
                    w2sf[e2, r2_o.begin, r2_k.begin // 16 + hl.arange(16), :], 8
                )
                rhs2 = zqf[r2_j.begin, :, r2_k]
                rhs2_s = zsf[r2_j.begin, :, r2_k.begin // 8 + hl.arange(32)]
                acc2 = hl.dot_scaled(
                    lhs2,
                    lhs2_s.view(E4M3),
                    "e2m1",
                    rhs2.permute(1, 0),
                    rhs2_s.view(E4M3),
                    "e2m1",
                    acc=acc2,
                    out_dtype=torch.float32,
                )
            val = (
                (acc2 * scale2[None, :])
                .reshape(4, 4, 8, rows)
                .permute(0, 2, 1, 3)
                .reshape(128, rows)
                .to(torch.bfloat16)
                .permute(1, 0)
            )
            hl.store(
                rows_out,
                [route[:, None], r2_o.begin * 128 + hl.arange(128)[None, :]],
                val,
                extra_mask=routed[:, None],
            )

    # C. Stock: RoutedExchange.combine's sum of a token's rows, this rank's share (its local routes in top-k order).
    # One dense tile per task: the peers poll the partials in-band, with no system-scope release behind them.
    for c_t, c_c in hl.tile([tokens, hidden // 256], block_size=[1, 1]):
        c_routes = c_t.begin * topk + hl.arange(topk)
        c_local = _local_ids(hl.load(ids, [c_routes]), base, experts) >= 0
        c_rows = hl.load(
            rows_out,
            [c_routes[:, None], c_c.begin * 256 + hl.arange(256)[None, :]],
            extra_mask=c_local[:, None],
        )
        ppart[c_t.begin, c_c.begin, :] = torch.sum(c_rows.to(torch.float32), dim=0)

    # J. Stock: RoutedExchange.combine (here every rank's partial summed in rank order) fused with
    # the mlp_sconv (residual: partial + attention + conv; taps before the verify rows from the paged conv cache, zero
    # before position 0), storing each row's conv input in the cache. One task per (token, 256 columns); each peer's
    # rows t - 3 .. t are polled as one tile (rows before the window read row 0, polled anyway, and take the cache).
    for j_t, j_c in hl.tile([tokens, hidden // 256], block_size=[1, 1]):
        cols = j_c.begin * 256 + hl.arange(256)
        # B's inputs first (not written by the peers: their loads overlap the poll).
        j_col0 = j_c.begin * res_chunk
        j_vcol = j_col0 % res_cols + hl.arange(res_chunk)
        j_s = hl.arange(res_slots)
        j_q = res_q[j_s, j_col0 // res_cols, j_vcol]
        j_a = res_acc[j_s, j_t.begin, j_col0 // res_cols, j_vcol]
        j_head = (
            j_c.begin * 256 // head_cols
        )  # (256 divides the stream's columns a head)
        j_hcol = j_c.begin * 256 % head_cols + hl.arange(256)
        j_tap = hl.arange(taps)
        j_src = j_t.begin - (taps - 1) + j_tap
        j_in = j_src >= 0
        j_rows = torch.clamp_min(j_src, 0)
        # The conv taps' history and weights (no peer writes them) also load ahead of the poll.
        j_pos = seq_len[0] - tokens + j_src  # the taps' absolute positions
        j_hist = (~j_in) & (j_pos >= 0)
        j_safe = torch.clamp_min(j_pos, 0)
        j_blk = hl.load(conv_pages, [j_safe // conv_block], extra_mask=j_hist)
        j_crow = (j_blk * conv_block + j_safe % conv_block) * conv_heads + j_head
        prior = hl.load(
            conv_cache, [j_crow[:, None], j_hcol[None, :]], extra_mask=j_hist[:, None]
        ).to(torch.float32)
        weight = (
            conv_w[cols, :].to(torch.bfloat16).to(torch.float32).permute(1, 0)
        )  # stock rounds it to bf16
        j_loc = conv_loc[j_t.begin]
        j_part = partial[j_t.begin, cols].to(torch.float32)
        j_attn = attention[j_t.begin, cols].to(torch.float32)
        combined = hl.zeros([taps, 256], dtype=torch.float32)
        for peer_part in ppart_peers:
            combined = combined + peer_part[j_rows, j_c.begin, :]
        combined = combined.to(torch.bfloat16).to(torch.float32)
        value = torch.where(j_in[:, None], combined, prior)
        conv_acc = torch.sum(value * weight, dim=0)
        current = torch.sum(
            torch.where((j_tap == taps - 1)[:, None], value, 0.0), dim=0
        )
        hl.store(
            conv_cache,
            [j_loc * conv_heads + j_head, j_hcol],
            current.to(torch.bfloat16),
            extra_mask=(hl.zeros([256], dtype=torch.int64) + j_loc) >= 0,
        )
        y = ((conv_acc + current) * conv_scale).to(torch.bfloat16).to(torch.float32)
        total = j_part + j_attn + y
        j_out = total.to(torch.bfloat16)
        out[j_t.begin, j_col0 // res_cols, j_vcol] = j_out
        # B's sums over the newest attn_res source (its share of the columns), per slot as one [QP, 4, chunk]
        # reduction: (q . x, x . x, res_acc . x, 0).
        j_x = j_out.to(torch.float32)
        j_i = hl.arange(4)[None, :, None]
        j_terms = j_x[None, None, :] * torch.where(
            j_i == 0,
            j_q[:, None, :],
            torch.where(
                j_i == 1,
                j_x[None, None, :],
                torch.where(j_i == 2, j_a[:, None, :], 0.0),
            ),
        )
        res_part[j_s, j_t.begin, j_c.begin, hl.arange(4)] = torch.sum(j_terms, dim=2)

    # B. Stock: block_attn_res + the norms (RMSNorm, eps 1e-6, bf16) of the next layer's attention inputs (slots 1 ..
    # A), per (token, J's columns): the new partial (this kernel's output) joins F's softmax; the sum's square from
    # J's and F's sums (|h|^2 = cA^2 |A|^2 + 2 cA cx A.x + cx^2 |x|^2).
    for b_t, b_n in hl.tile([tokens, hidden // res_chunk], block_size=[1, 1]):
        # Inputs J does not write first (their loads overlap the wait for J).
        b_col0 = b_n.begin * res_chunk
        b_v = b_col0 // res_cols
        b_cols = b_col0 % res_cols + hl.arange(res_chunk)
        b_s = hl.arange(res_slots)
        b_a = res_acc[b_s, b_t.begin, b_v, b_cols]
        b_w = res_w[b_s, b_v, b_cols].to(torch.float32)
        # The older sources' max and total (the same in every chunk) and square sums (added in column order).
        b_rows = hl.arange(stat_rows)
        b_live = (b_rows < res_stat.size(2))[None, :, None]
        b_c4 = hl.arange(4)[None, None, :]
        b_stat = hl.load(
            res_stat, [b_s, b_t.begin, b_rows, hl.arange(4)], extra_mask=b_live
        )
        b_max = torch.amax(
            torch.amax(torch.where(b_live & (b_c4 == 0), b_stat, float("-inf")), dim=2),
            dim=1,
        )
        b_total = torch.amax(
            torch.amax(torch.where(b_live & (b_c4 == 1), b_stat, float("-inf")), dim=2),
            dim=1,
        )
        b_sq = torch.sum(torch.sum(torch.where(b_c4 == 2, b_stat, 0.0), dim=2), dim=1)
        b_sums = torch.sum(
            res_part[b_s, b_t.begin, :, :], dim=1
        )  # [QP, 4] in J's column order
        b_i = hl.arange(4)[None, :]
        b_dot = torch.sum(torch.where(b_i == 0, b_sums, 0.0), dim=1)
        b_ss = torch.sum(torch.where(b_i == 1, b_sums, 0.0), dim=1)
        b_ax = torch.sum(torch.where(b_i == 2, b_sums, 0.0), dim=1)
        b_rstd = torch.rsqrt(b_ss * (1.0 / hidden) + 1e-6)
        b_logit = b_dot * b_rstd
        b_top = torch.maximum(b_max, b_logit)
        b_old = torch.exp(b_max - b_top)
        b_new = torch.exp(b_logit - b_top)
        b_inv = 1.0 / (b_total * b_old + b_new)
        b_ca = b_old * b_inv
        b_cx = b_new * b_rstd * b_inv
        b_hh = b_ca * b_ca * b_sq + 2.0 * b_ca * b_cx * b_ax + b_cx * b_cx * b_ss
        b_norm = torch.rsqrt(b_hh * (1.0 / hidden) + 1e-6)
        b_x = out[b_t.begin, b_v, b_cols].to(torch.float32)
        b_y = (
            (b_a * b_ca[:, None] + b_x[None, :] * b_cx[:, None]) * b_norm[:, None] * b_w
        )
        b_in = (b_s >= 1) & (b_s <= res_qidx[1])
        hl.store(
            x_next,
            [b_s - 1, b_t.begin, b_v, b_cols],
            b_y.to(torch.bfloat16),
            extra_mask=b_in[:, None],
        )
    return out
