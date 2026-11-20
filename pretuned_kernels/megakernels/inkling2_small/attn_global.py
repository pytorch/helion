"""Inkling-2-Small global (full-attention) layer's attention half of one rank as one Helion kernel (TP4,
batch-1 verify window of T = 8 rows), plus the MLP input of the layer (attn_res + mlp_norm, like ink_attn_v8.py), on
vLLM's layouts (ink_attn_global_s16.py on SGLang's): the paged KV cache ([blocks, KV heads, block size, K | V]) and
the paged conv-state cache (each stream's conv input at every absolute position). The qkvr weight is a copy in
SGLang's row order (q | k | v | r | gate | sinks: 128-aligned inputs for P's tiles).

Differences from the local layers (ink_attn_v8.py): three attention inputs (q / k / v_input_norm of a 3-query
attn_res: q, r, gate and sinks project the q input, k the k input, v the v input); K/V convs with token-dependent
weights (static_w[lag] + bf16(x_k|v . weight_proj[head, lag]), residual, (2W + 1)^-0.5); the log-scaling tau
(1 + 0.1 ln max(1, (pos + 1) / 128000)) on the clipped q and r; full attention over the whole cache (the rel bias
only within rel_extent = 1024 keys: 0 past it); the full KV pool.

Roots (rank-local: 12 query heads in 2 KV groups of 6, padded to 8; head_dim 128):
    P   (128 qkvr columns, K split) -> fp32 partials (each column tile's own input)         [30 x S]
    PD  (K | V, K part, token) -> the conv weights' token-dependent part (x . weight_proj)   [2 x 9 x T]
    PQR (token, head) -> clipped, tau-scaled q (padded heads stay 0), r and its rel bias    [T x 12]
    PK  (KV head, token) -> K / V dynamic convs, K clip, V norm -> the new rows (kv_new)    [2 x T]
    A   (KV group, split) -> partial attention over the split's cached pages (split,        [2 x 64]
        split + 64, ..: online softmax; positions before the T new rows)
    AN  (KV group) -> partial attention over the T new keys (kv_new)                        [2]
    M   (token, KV group) -> the new rows into the KV cache; the splits + AN merged,        [T x 2]
        gate-sink -> o_in
    O, C, J, B as ink_attn_v8.py
"""

from __future__ import annotations

from pretuned_kernels.megakernels.inkling2_small._common import LN2
from pretuned_kernels.megakernels.inkling2_small._common import LOG2E
from pretuned_kernels.megakernels.inkling2_small._common import _l2_clip
from pretuned_kernels.megakernels.inkling2_small._common import _rsqrt_approx
from pretuned_kernels.megakernels.inkling2_small._common import _sigmoid
import torch

import helion.language as hl

TAU_FLOOR, TAU_ALPHA = 128000, 0.1  # log_scaling_n_floor, log_scaling_alpha


def _verify_conv(
    part: torch.Tensor,
    col0: int,
    cols: torch.Tensor,
    row: int,
    static_w: torch.Tensor,
    dyn: torch.Tensor,
    conv_cache: torch.Tensor,
    head: int,
    n_heads: int,
    conv_pages: torch.Tensor,
    conv_loc: torch.Tensor,
    conv_block: int,
    cache_end: torch.Tensor,
    taps: int,
) -> torch.Tensor:
    """One verify row of attention_prologue's dynamic-weight sconv over qkvr columns col0 + cols (weight
    static_w[lag] + bf16(dyn[lag]), lag 0 first, residual, (2W + 1)^-0.5; the taps before the verify rows from the
    paged conv cache, zero before position 0), storing the row's conv input in the cache (row slot * n_heads + head of the stream);
    bf16 like the stock conv output."""
    width = cols.size(0)
    dims = hl.arange(width)
    acc = hl.zeros([width], dtype=torch.float32)
    current = hl.zeros([width], dtype=torch.float32)
    for tap in hl.static_range(taps):
        lag = tap
        src = row - lag
        in_window = (hl.zeros([width], dtype=torch.int32) + src) >= 0
        raw = torch.sum(
            hl.load(
                part, [slice(None), src, col0 + cols], extra_mask=in_window[None, :]
            ),
            dim=0,
        )
        raw = raw.to(torch.bfloat16).to(torch.float32)
        if lag == 0:
            value = raw
            current = raw
            loc = hl.load(conv_loc, [row])
            hl.store(
                conv_cache,
                [loc * n_heads + head, dims],
                raw.to(torch.bfloat16),
                extra_mask=(hl.zeros([width], dtype=torch.int64) + loc) >= 0,
            )
        else:
            pos = cache_end + src
            hist = (~in_window) & ((hl.zeros([width], dtype=torch.int32) + pos) >= 0)
            safe = torch.clamp_min(pos, 0)
            blk = hl.load(
                conv_pages,
                [torch.clamp_max(safe // conv_block, conv_pages.size(0) - 1)],
            )
            prior = hl.load(
                conv_cache,
                [(blk * conv_block + safe % conv_block) * n_heads + head, dims],
                extra_mask=hist,
            ).to(torch.float32)
            value = torch.where(in_window, raw, prior)
        weight = hl.load(static_w, [lag, cols]).to(torch.float32) + torch.sum(
            torch.where(hl.arange(taps) == lag, dyn, 0.0), dim=0
        ).to(torch.bfloat16).to(torch.float32)
        acc = acc + value * weight
    return (
        ((acc + current) * (2 * taps + 1) ** -0.5).to(torch.bfloat16).to(torch.float32)
    )


def _ink_attn_global(
    xs: torch.Tensor,  # [3, R, H] bf16: the q / k / v attention inputs (q/k/v_input_norm outputs); rows past T zero
    src_tab: torch.Tensor,  # [N / 128] int32: the input (0 q, 1 k, 2 v) each 128-column qkvr tile projects
    w_qkvr: torch.Tensor,  # [N, S, H/S] bf16 view of this rank's qkvr weight [N, H] (N = 3792: q|k|v|r|gate|sinks|pad)
    part: torch.Tensor,  # [S, R, N] fp32 workspace: the qkvr split-K partials
    dyn_proj: torch.Tensor,  # [2, 2 * 4, H] bf16: (k | v)_sconv weight_proj ((head, lag) rows), stacked
    dyn: torch.Tensor,  # [2, KD, T, 2 * 4] fp32 workspace: (K | V, K part, token, (head, lag)) x . weight_proj
    k_conv_w: torch.Tensor,  # [4, Dkv] bf16: k_sconv static_w (Dkv = 2 heads x 128)
    v_conv_w: torch.Tensor,  # [4, Dkv] bf16
    k_conv_cache: torch.Tensor,  # [NB * CB * 2, 128] bf16: the K stream of the layer's paged conv cache, row slot * 2 + head
    v_conv_cache: torch.Tensor,  # [NB * CB * 2, 128] bf16: its V stream
    conv_pages: torch.Tensor,  # [MB] int32: the request's conv-cache block table (position p in block p // CB)
    conv_loc: torch.Tensor,  # [T] int64: the verify rows' conv-cache slots (block * CB + offset; < 0: none)
    conv_block: hl.constexpr,  # CB: the conv cache's positions a block
    q_scale: torch.Tensor,  # [12, 128] bf16: q_l2_clip scale
    r_scale: torch.Tensor,  # [12, 16] bf16: rel_l2_clip scale
    rel_proj: torch.Tensor,  # [16, E] bf16: rel_logits_proj (E = rel_extent = 1024)
    k_buf: torch.Tensor,  # [NB * BS, 2, 128] bf16: this layer's K cache by slot (vLLM's kv_cache, slot-major, [..., :128])
    v_buf: torch.Tensor,  # [NB * BS, 2, 128] bf16: its V cache ([..., 128:])
    pages: torch.Tensor,  # [MB] int32: the request's block table (position p in block pages[p // BS], row p % BS)
    seq_len: torch.Tensor,  # [1] int32: the request's length including the T new rows
    kv_loc: torch.Tensor,  # [T] int64: the new rows' KV cache slots (block * BS + row; < 0: none)
    page_size: hl.constexpr,  # BS: the KV cache's positions a block
    q_ws: torch.Tensor,  # [T, 2, 8, 128] bf16 workspace (padded heads zero)
    rel_ws: torch.Tensor,  # [T, 12, E] bf16 workspace
    att_ws: torch.Tensor,  # [2, SPLITS, 64, 128] fp32 workspace: per-split normalized partial outputs
    lse_ws: torch.Tensor,  # [2, SPLITS, 64] fp32 workspace: per-split log2-sum-exp2 (-inf: no visible key)
    kv_new: torch.Tensor,  # [2, 2, 16, 128] bf16 workspace: (K | V, KV head, new row (T used), dim)
    att_new: torch.Tensor,  # [2, 64, 128] fp32 workspace: AN's normalized partial outputs
    lse_new: torch.Tensor,  # [2, 64] fp32 workspace: AN's log2-sum-exp2
    o_in: torch.Tensor,  # [R, 4, 384] bf16 workspace: gate-sink output by K quarter (pad rows zero)
    wo: torch.Tensor,  # [H, 4, 384] bf16 view of this rank's wo_ud weight [H, 1536]
    part_o: torch.Tensor,  # [4, R, H] fp32 workspace: wo partials by K quarter
    opart: torch.Tensor,  # [T, H/C, C] bf16 symmetric: this rank's wo rows (the all-reduce input), C columns a task
    a_conv_w: torch.Tensor,  # [H, 4] bf16: attn_sconv weight
    a_conv_cache: torch.Tensor,  # [NB * CB * 2, H/2] bf16: the attn_sconv stream of the paged conv cache (as K's)
    out: torch.Tensor,  # [T, H/V, V] bf16 out: attn_sconv(all-reduced wo rows) (V = res_q's columns)
    res_partial: torch.Tensor,  # [T, H/V, V] bf16: the residual stream's partial block (the newest source's addend)
    res_q: torch.Tensor,  # [H/V, V] fp32: the MLP input's attn_res query, scaled by H^-0.5
    res_w: torch.Tensor,  # [H/V, V]: mlp_norm's weight
    res_acc: torch.Tensor,  # [T, H/V, V] fp32: the older sources' softmax-weighted normalized sum (the MoE kernel's F)
    res_stat: torch.Tensor,  # [T, H/C, 4] fp32: (their logit max, exp total, res_acc's square sum by columns, 0)
    res_part: torch.Tensor,  # [T, H/C, 4] fp32 workspace: J's (q . x, x . x, res_acc . x, 0) of the newest source x
    x_next: torch.Tensor,  # [R, H/V, V] bf16 out: the MoE kernel's input rows (mlp_norm(attn_res(...)))
    res_addend: hl.constexpr,  # the newest source is res_partial + attn_sconv (else attn_sconv alone)
    group_name: hl.ProcessGroupName,
    world: hl.constexpr,
    rank: hl.constexpr,
) -> torch.Tensor:
    hl.specialize(xs.size())
    hl.specialize(xs.stride())
    hl.specialize(src_tab.size())
    hl.specialize(src_tab.stride())
    hl.specialize(dyn_proj.size())
    hl.specialize(dyn_proj.stride())
    hl.specialize(dyn.size())
    hl.specialize(dyn.stride())
    hl.specialize(w_qkvr.size())
    hl.specialize(w_qkvr.stride())
    hl.specialize(part.size())
    hl.specialize(part.stride())
    hl.specialize(k_conv_w.size())
    hl.specialize(k_conv_w.stride())
    hl.specialize(v_conv_w.size())
    hl.specialize(v_conv_w.stride())
    hl.specialize(k_conv_cache.size())
    hl.specialize(k_conv_cache.stride())
    hl.specialize(v_conv_cache.size())
    hl.specialize(v_conv_cache.stride())
    hl.specialize(conv_pages.size())
    hl.specialize(conv_pages.stride())
    hl.specialize(conv_loc.size())
    hl.specialize(conv_loc.stride())
    hl.specialize(q_scale.size())
    hl.specialize(q_scale.stride())
    hl.specialize(r_scale.size())
    hl.specialize(r_scale.stride())
    hl.specialize(rel_proj.size())
    hl.specialize(rel_proj.stride())
    hl.specialize(k_buf.size())
    hl.specialize(k_buf.stride())
    hl.specialize(v_buf.size())
    hl.specialize(v_buf.stride())
    hl.specialize(pages.size())
    hl.specialize(pages.stride())
    hl.specialize(seq_len.size())
    hl.specialize(seq_len.stride())
    hl.specialize(kv_loc.size())
    hl.specialize(kv_loc.stride())
    hl.specialize(q_ws.size())
    hl.specialize(q_ws.stride())
    hl.specialize(rel_ws.size())
    hl.specialize(rel_ws.stride())
    hl.specialize(att_ws.size())
    hl.specialize(att_ws.stride())
    hl.specialize(lse_ws.size())
    hl.specialize(lse_ws.stride())
    hl.specialize(kv_new.size())
    hl.specialize(kv_new.stride())
    hl.specialize(att_new.size())
    hl.specialize(att_new.stride())
    hl.specialize(lse_new.size())
    hl.specialize(lse_new.stride())
    hl.specialize(o_in.size())
    hl.specialize(o_in.stride())
    hl.specialize(wo.size())
    hl.specialize(wo.stride())
    hl.specialize(part_o.size())
    hl.specialize(part_o.stride())
    hl.specialize(opart.size())
    hl.specialize(opart.stride())
    hl.specialize(a_conv_w.size())
    hl.specialize(a_conv_w.stride())
    hl.specialize(a_conv_cache.size())
    hl.specialize(a_conv_cache.stride())
    hl.specialize(out.size())
    hl.specialize(out.stride())
    hl.specialize(res_partial.size())
    hl.specialize(res_partial.stride())
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
    hidden = xs.size(2)
    mma_rows = xs.size(1)
    n_qkvr, k_splits, k_per = w_qkvr.size()
    tokens = q_ws.size(0)
    kv_heads, padded, head_dim = q_ws.size(1), q_ws.size(2), q_ws.size(3)
    heads = q_scale.size(0)
    group = heads // kv_heads
    d_rel = r_scale.size(1)
    rel_extent = rel_proj.size(1)
    n_splits = att_ws.size(1)
    rows = padded * tokens  # attention rows of a KV group: (token, padded head)
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    r_col = q_width + 2 * kv_width
    gate_col = r_col + heads * d_rel
    sink_col = gate_col + q_width
    o_parts, o_split = wo.size(1), wo.size(2)
    taps = k_conv_w.size(0)
    conv_scale = (taps + 1) ** -0.5  # attn_sconv (static weight)
    x4 = xs.view(3, mma_rows, k_splits, k_per)
    dyn_parts = dyn.size(1)
    window_left = rel_extent - 1
    scale_log2 = head_dim**-0.5 * LOG2E
    head_cols = a_conv_cache.size(1)
    opart_peers = torch.ops.symm_mem.get_remote_tensors(opart, group_name)
    new_keys = kv_new.size(2)
    chunk = opart.size(2)  # all-reduce columns a task
    res_cols = res_q.size(1)
    stat_rows = (
        1 << (res_stat.size(1) - 1).bit_length()
    )  # F2 tasks a row, padded to a power of two

    hl.prefetch_code()

    # P. Stock: qkvr (bf16 x bf16 -> bf16; here fp32 partials by K split, summed in split order by the readers).
    for p_n, p_s in hl.tile([n_qkvr, k_splits], block_size=[128, 1]):
        p_src = src_tab[p_n.begin // 128]
        p_acc = hl.zeros([p_n, mma_rows], dtype=torch.float32)
        for p_k in hl.tile(k_per, block_size=128):
            p_acc = hl.dot(
                w_qkvr[p_n, p_s.begin, p_k],
                x4[p_src, :, p_s.begin, p_k].permute(1, 0),
                acc=p_acc,
            )
        part[p_s.begin, :, p_n] = p_acc.permute(1, 0)

    # PD. Stock: the dynamic convs' weight projections (linear_k_group_partials_pair: x_k|v . weight_proj, summed
    # and rounded to bf16 by the conv; here fp32 partials by K part, summed in part order and rounded by PK).
    for d_kv, d_p, d_t in hl.tile([2, dyn_parts, tokens], block_size=[1, 1, 1]):
        d_cols = d_p.begin * (hidden // dyn_parts) + hl.arange(hidden // dyn_parts)
        d_x = xs[1 + d_kv.begin, d_t.begin, d_cols].to(torch.float32)
        dyn[d_kv.begin, d_p.begin, d_t.begin, :] = torch.sum(
            dyn_proj[d_kv.begin, :, d_cols].to(torch.float32) * d_x[None, :], dim=1
        )

    # PQR. Stock: q_l2_clip (learnable per-(head, dim) scale), bf16, times tau, bf16, written padded (head h ->
    # group h // 6, h % 6); rel_l2_clip (scale, threshold 8), bf16, times tau, bf16, rel_logits_proj (K = 16), bf16.
    for q_t, q_h in hl.tile([tokens, heads], block_size=[1, 1]):
        q_pos = (
            hl.zeros([1], dtype=torch.int32) + (seq_len[0] - tokens + q_t.begin + 1)
        ).to(torch.float32)
        q_tau = 1.0 + TAU_ALPHA * torch.log(
            torch.clamp_min(q_pos * (1.0 / TAU_FLOOR), 1.0)
        )
        q_cols = q_h.begin * head_dim + hl.arange(head_dim)
        q_raw = (
            torch.sum(part[:, q_t.begin, q_cols], dim=0)
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        q_in = q_scale[q_h.begin, :].to(torch.float32) * q_raw
        q_clip = (
            _l2_clip(q_in[None, :], 22.627416997969522, 1e-6)
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        q_ws[q_t.begin, q_h.begin // group, q_h.begin % group, :] = (
            (q_clip * q_tau[:, None]).to(torch.bfloat16).reshape(head_dim)
        )
        r_cols = r_col + q_h.begin * d_rel + hl.arange(d_rel)
        r_raw = (
            torch.sum(part[:, q_t.begin, r_cols], dim=0)
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        r_in = r_scale[q_h.begin, :].to(torch.float32) * r_raw
        r_clip = _l2_clip(r_in[None, :], 8.0, 1e-6).to(torch.bfloat16).to(torch.float32)
        r_bf = (r_clip * q_tau[:, None]).to(torch.bfloat16).to(torch.float32)
        rel = torch.sum(
            r_bf.reshape(d_rel)[:, None] * rel_proj[:, :].to(torch.float32), dim=0
        )
        rel_ws[q_t.begin, q_h.begin, :] = rel.to(torch.bfloat16)

    # PK. Stock: attention_prologue's K / V verify sconv (dynamic weight, residual, (2W + 1)^-0.5, window save),
    # K L2 clip, V RMSNorm (no gamma), bf16 at every step. The rows go to kv_new (M stores them into the cache).
    for k_g, k_t in hl.tile([kv_heads, tokens], block_size=[1, 1]):
        k_end = seq_len[0] - tokens
        kv_cols = k_g.begin * head_dim + hl.arange(head_dim)
        k_lags = k_g.begin * taps + hl.arange(taps)
        k_dyn = torch.sum(hl.load(dyn, [0, slice(None), k_t.begin, k_lags]), dim=0)
        v_dyn = torch.sum(hl.load(dyn, [1, slice(None), k_t.begin, k_lags]), dim=0)
        k_conv = _verify_conv(
            part,
            q_width,
            kv_cols,
            k_t.begin,
            k_conv_w,
            k_dyn,
            k_conv_cache,
            k_g.begin,
            kv_heads,
            conv_pages,
            conv_loc,
            conv_block,
            k_end,
            taps,
        )
        v_conv = _verify_conv(
            part,
            q_width + kv_width,
            kv_cols,
            k_t.begin,
            v_conv_w,
            v_dyn,
            v_conv_cache,
            k_g.begin,
            kv_heads,
            conv_pages,
            conv_loc,
            conv_block,
            k_end,
            taps,
        )
        k_new = _l2_clip(k_conv[None, :], 22.627416997969522, 1e-6).to(torch.bfloat16)
        kv_new[0, k_g.begin, k_t.begin, :] = k_new.reshape(head_dim)
        v_rstd = _rsqrt_approx(
            torch.sum(v_conv * v_conv, dim=0) * (1.0 / head_dim) + 1e-6
        )
        kv_new[1, k_g.begin, k_t.begin, :] = (v_conv * v_rstd).to(torch.bfloat16)

    # A. Stock: FA4 over the whole cache (split KV), the rel bias added unscaled (rel_bias_scale_after_add:
    # q.k * scale + rel) for keys within rel_extent of the row (0 past it); one task per (KV group, split): the
    # (token, padded head) rows' online softmax over the split's cached pages (positions before the new rows,
    # which AN covers).
    for a_g, a_s in hl.tile([kv_heads, n_splits], block_size=[1, 1]):
        cache_end = seq_len[0] - tokens
        a_iters = (
            (cache_end + page_size - 1) // page_size - a_s.begin + n_splits - 1
        ) // n_splits
        # (a split past the cached pages writes nothing: M reads the live splits only)
        if a_iters > 0:
            a_rows = hl.arange(rows)
            r_tok = a_rows // padded
            r_head = a_rows % padded
            qq = q_ws[:, a_g.begin, :, :].reshape(rows, head_dim)
            q_pos = cache_end + r_tok
            m_i = hl.full([rows], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([rows], dtype=torch.float32)
            acc = hl.zeros([rows, head_dim], dtype=torch.float32)
            for a_i in hl.tile(a_iters, block_size=1):
                page_idx = a_s.begin + a_i.begin * n_splits
                page = pages[page_idx].to(torch.int64)
                keys = hl.arange(page_size)
                slots = page * page_size + keys
                kv_pos = page_idx * page_size + keys
                kk = hl.load(k_buf, [slots, a_g.begin, slice(None)])
                vv = hl.load(v_buf, [slots, a_g.begin, slice(None)])
                dist = q_pos[:, None] - kv_pos[None, :]
                valid = (kv_pos < cache_end)[None, :] & (r_head < group)[:, None]
                bias = hl.load(
                    rel_ws,
                    [r_tok[:, None], (a_g.begin * group + r_head)[:, None], dist],
                    extra_mask=valid & (dist <= window_left),
                )
                scores = hl.dot(qq, kk.permute(1, 0), out_dtype=torch.float32)
                logits = torch.where(
                    valid,
                    scores * scale_log2 + bias.to(torch.float32) * LOG2E,
                    float("-inf"),
                )
                m_new = torch.maximum(m_i, torch.amax(logits, dim=1))
                m_safe = torch.where(m_new > float("-inf"), m_new, 0.0)
                prob = torch.exp2(logits - m_safe[:, None])
                rescale = torch.exp2(m_i - m_safe)
                l_i = l_i * rescale + torch.sum(prob, dim=1)
                acc = acc * rescale[:, None] + hl.dot(
                    prob.to(torch.bfloat16), vv, out_dtype=torch.float32
                )
                m_i = m_new
            seen = l_i > 0.0
            att_ws[a_g.begin, a_s.begin, :, :] = (
                acc / torch.where(seen, l_i, 1.0)[:, None]
            )
            lse_ws[a_g.begin, a_s.begin, :] = torch.where(
                seen, m_i + torch.log2(l_i), float("-inf")
            )

    # AN. The same over the T new keys of each KV group (kv_new; every one is inside every row's window).
    for n_g in hl.tile(kv_heads, block_size=1):
        n_rows = hl.arange(rows)
        n_tok = n_rows // padded
        n_head = n_rows % padded
        n_keys = hl.arange(new_keys)
        nk = kv_new[0, n_g.begin, :, :]
        nv = kv_new[1, n_g.begin, :, :]
        nq = q_ws[:, n_g.begin, :, :].reshape(rows, head_dim)
        n_dist = n_tok[:, None] - n_keys[None, :]
        n_valid = (n_dist >= 0) & (n_head < group)[:, None]
        n_bias = hl.load(
            rel_ws,
            [n_tok[:, None], (n_g.begin * group + n_head)[:, None], n_dist],
            extra_mask=n_valid,
        )
        n_scores = hl.dot(nq, nk.permute(1, 0), out_dtype=torch.float32)
        n_logits = torch.where(
            n_valid,
            n_scores * scale_log2 + n_bias.to(torch.float32) * LOG2E,
            float("-inf"),
        )
        n_max = torch.amax(n_logits, dim=1)
        n_safe = torch.where(n_max > float("-inf"), n_max, 0.0)
        n_prob = torch.exp2(n_logits - n_safe[:, None])
        n_sum = torch.sum(n_prob, dim=1)
        n_pv = hl.dot(n_prob.to(torch.bfloat16), nv, out_dtype=torch.float32)
        n_seen = n_sum > 0.0
        att_new[n_g.begin, :, :] = n_pv / torch.where(n_seen, n_sum, 1.0)[:, None]
        lse_new[n_g.begin, :] = torch.where(
            n_seen, n_safe + torch.log2(n_sum), float("-inf")
        )

    # M. Stock: FA4's split merge (natural-log lse), then gate_sink: out * sigmoid(gate) * sigmoid(lse - sink), bf16.
    # A partial of weight 0 is still read (finite: A only writes normalized partials into the zeroed workspace).
    for m_t, m_g in hl.tile([tokens, kv_heads], block_size=[1, 1]):
        # The new rows into the cache, after every A task of this group read it (M waits for them).
        m_loc = kv_loc[m_t.begin]
        m_ok = (hl.zeros([head_dim], dtype=torch.int64) + m_loc) >= 0
        m_dims = hl.arange(head_dim)
        hl.store(
            k_buf,
            [m_loc, m_g.begin, m_dims],
            kv_new[0, m_g.begin, m_t.begin, :],
            extra_mask=m_ok,
        )
        hl.store(
            v_buf,
            [m_loc, m_g.begin, m_dims],
            kv_new[1, m_g.begin, m_t.begin, :],
            extra_mask=m_ok,
        )
        m_rows = m_t.begin * padded + hl.arange(padded)
        m_splits = hl.arange(n_splits)
        # Splits past the cached pages hold no keys (their partials are not read).
        m_live = torch.clamp_max(
            (seq_len[0] - tokens + page_size - 1) // page_size, n_splits
        )
        m_on = (m_splits < m_live)[:, None]
        m_lse = torch.where(
            m_on,
            hl.load(
                lse_ws, [m_g.begin, m_splits[:, None], m_rows[None, :]], extra_mask=m_on
            ),
            float("-inf"),
        )  # [splits, padded]
        n_lse = hl.load(lse_new, [m_g.begin, m_rows])
        top = torch.maximum(torch.amax(m_lse, dim=0), n_lse)
        top = torch.where(top > float("-inf"), top, 0.0)
        n_weight = torch.exp2(n_lse - top)
        total = torch.sum(torch.exp2(m_lse - top[None, :]), dim=0) + n_weight
        # One live split at a time (a [splits, 8, 128] fp32 tile spills).
        merged = (
            hl.load(att_new, [m_g.begin, m_rows[:, None], m_dims[None, :]])
            * n_weight[:, None]
        )
        for m_p in hl.tile(m_live, block_size=1):
            p_weight = torch.exp2(hl.load(lse_ws, [m_g.begin, m_p.begin, m_rows]) - top)
            merged = (
                merged
                + hl.load(
                    att_ws, [m_g.begin, m_p.begin, m_rows[:, None], m_dims[None, :]]
                )
                * p_weight[:, None]
            )
        merged = merged / torch.where(total > 0.0, total, 1.0)[:, None]
        lse = (top + torch.log2(total)) * LN2
        m_heads = m_g.begin * group + hl.arange(padded)
        live_h = hl.arange(padded) < group
        g_flat = hl.arange(padded * head_dim)
        gate = torch.sum(
            hl.load(
                part,
                [
                    slice(None),
                    m_t.begin,
                    gate_col + m_g.begin * group * head_dim + g_flat,
                ],
                extra_mask=(g_flat < group * head_dim)[None, :],
            ),
            dim=0,
        )
        gate = gate.reshape(padded, head_dim).to(torch.bfloat16).to(torch.float32)
        sink = (
            torch.sum(
                hl.load(
                    part,
                    [slice(None), m_t.begin, sink_col + m_heads],
                    extra_mask=live_h[None, :],
                ),
                dim=0,
            )
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        gated = merged * _sigmoid(gate)
        gated = gated * _sigmoid(lse - sink)[:, None]
        # The group's (head, dim) row split into K quarters of o_split columns.
        m_col = (
            m_g.begin * group * head_dim
            + (hl.arange(padded) * head_dim)[:, None]
            + hl.arange(head_dim)[None, :]
        )
        hl.store(
            o_in,
            [m_t.begin, m_col // o_split, m_col % o_split],
            gated.to(torch.bfloat16),
            extra_mask=live_h[:, None],
        )

    # O. Stock: wo_ud (bf16; here fp32 partials by K quarter, summed in order by C).
    for o_n, o_q in hl.tile([hidden, o_parts], block_size=[128, 1]):
        o_acc = hl.zeros([o_n, mma_rows], dtype=torch.float32)
        for o_k in hl.tile(o_split, block_size=128):
            o_acc = hl.dot(
                wo[o_n, o_q.begin, o_k],
                o_in[:, o_q.begin, o_k].permute(1, 0),
                acc=o_acc,
            )
        part_o[o_q.begin, :, o_n] = o_acc.permute(1, 0)

    # C. The bf16 wo row of each (token, 128 columns) into the symmetric buffer: one dense tile per task, so the
    # peers poll it in-band (no system-scope release).
    for c_t, c_n in hl.tile([tokens, hidden // chunk], block_size=[1, 1]):
        c_cols = c_n.begin * chunk + hl.arange(chunk)
        opart[c_t.begin, c_n.begin, :] = (
            (
                (part_o[0, c_t.begin, c_cols] + part_o[1, c_t.begin, c_cols])
                + part_o[2, c_t.begin, c_cols]
            )
            + part_o[3, c_t.begin, c_cols]
        ).to(torch.bfloat16)

    # J. Stock: the push all-reduce (rank-order fp32 sum of the bf16 rows, bf16) fused with attn_sconv (verify
    # window: the conv-state prefix, window save, residual, (W + 1)^-0.5, bf16). Each peer's rows t - 3 .. t are
    # polled as one tile (rows before the window read row 0, polled anyway, and take the conv-state prefix).
    for j_t, j_n in hl.tile([tokens, hidden // chunk], block_size=[1, 1]):
        j_cols = j_n.begin * chunk + hl.arange(chunk)
        # (a task's columns lie in one row of out's view: chunk divides res_cols)
        j_col0 = j_n.begin * chunk
        j_vcol = j_col0 % res_cols + hl.arange(chunk)
        # B's inputs first (not written by the peers: their loads overlap the poll).
        j_q = res_q[j_col0 // res_cols, j_vcol]
        j_a = res_acc[j_t.begin, j_col0 // res_cols, j_vcol]
        j_p = res_partial[j_t.begin, j_col0 // res_cols, j_vcol].to(torch.float32)
        j_head = j_col0 // head_cols  # (chunk divides the stream's columns a head)
        j_hcol = j_col0 % head_cols + hl.arange(chunk)
        j_tap = hl.arange(taps)
        j_src = j_t.begin - (taps - 1) + j_tap
        j_in = j_src >= 0
        j_rows = torch.clamp_min(j_src, 0)
        reduced = hl.zeros([taps, chunk], dtype=torch.float32)
        for peer_part in opart_peers:
            reduced = reduced + peer_part[j_rows, j_n.begin, :].to(torch.float32)
        reduced = reduced.to(torch.bfloat16).to(torch.float32)
        j_pos = seq_len[0] - tokens + j_src  # the taps' absolute positions
        j_hist = (~j_in) & (j_pos >= 0)
        j_safe = torch.clamp_min(j_pos, 0)
        j_blk = hl.load(conv_pages, [j_safe // conv_block], extra_mask=j_hist)
        j_crow = (j_blk * conv_block + j_safe % conv_block) * kv_heads + j_head
        j_prior = hl.load(
            a_conv_cache, [j_crow[:, None], j_hcol[None, :]], extra_mask=j_hist[:, None]
        ).to(torch.float32)
        j_value = torch.where(j_in[:, None], reduced, j_prior)
        j_w = a_conv_w[j_cols, :].to(torch.float32).permute(1, 0)
        j_acc = torch.sum(j_value * j_w, dim=0)
        j_current = torch.sum(
            torch.where((j_tap == taps - 1)[:, None], j_value, 0.0), dim=0
        )
        j_loc = conv_loc[j_t.begin]
        hl.store(
            a_conv_cache,
            [j_loc * kv_heads + j_head, j_hcol],
            j_current.to(torch.bfloat16),
            extra_mask=(hl.zeros([chunk], dtype=torch.int64) + j_loc) >= 0,
        )
        j_out = ((j_acc + j_current) * conv_scale).to(torch.bfloat16)
        out[j_t.begin, j_col0 // res_cols, j_vcol] = j_out
        # B's sums over the newest attn_res source (its share of the columns), as one [4, chunk] reduction:
        # (q . x, x . x, res_acc . x, 0).
        j_x = j_out.to(torch.float32)
        if res_addend:
            j_x = j_x + j_p
        j_i = hl.arange(4)[:, None]
        j_terms = j_x[None, :] * torch.where(
            j_i == 0,
            j_q[None, :],
            torch.where(
                j_i == 1, j_x[None, :], torch.where(j_i == 2, j_a[None, :], 0.0)
            ),
        )
        res_part[j_t.begin, j_n.begin, :] = torch.sum(j_terms, dim=1)

    # B. Stock: block_attn_res (normalize_values: softmax over the sources of (q . x_n) * rstd(x_n), the
    # rstd-normalized sources summed, fp32) + mlp_norm (RMSNorm, eps 1e-6, bf16) of the MoE input, per (token, J's
    # columns): the newest source (attn_sconv's rows plus the partial block) joins the older sources' softmax
    # (res_acc, res_stat: the previous MoE kernel's F); the sum's square from J's and F's sums (|h|^2 = cA^2 |A|^2 + 2 cA cx A.x + cx^2 |x|^2).
    for b_t, b_n in hl.tile([tokens, hidden // chunk], block_size=[1, 1]):
        b_sums = torch.sum(res_part[b_t.begin, :, :], dim=0)  # in J's column order
        b_i = hl.arange(4)
        b_dot = torch.sum(torch.where(b_i == 0, b_sums, 0.0), dim=0)
        b_ss = torch.sum(torch.where(b_i == 1, b_sums, 0.0), dim=0)
        b_ax = torch.sum(torch.where(b_i == 2, b_sums, 0.0), dim=0)
        # F2's tasks agree on the max and total; the square sums add in column order.
        b_rows = hl.arange(stat_rows)[:, None]
        b_cols4 = hl.arange(4)[None, :]
        b_stat = hl.load(
            res_stat, [b_t.begin, b_rows, b_cols4], extra_mask=b_rows < res_stat.size(1)
        )
        b_live = b_rows < res_stat.size(1)
        b_max = torch.amax(
            torch.amax(
                torch.where(b_live & (b_cols4 == 0), b_stat, float("-inf")), dim=1
            ),
            dim=0,
        )
        b_total = torch.amax(
            torch.amax(
                torch.where(b_live & (b_cols4 == 1), b_stat, float("-inf")), dim=1
            ),
            dim=0,
        )
        b_sq = torch.sum(
            torch.sum(torch.where(b_cols4 == 2, b_stat, 0.0), dim=1), dim=0
        )
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
        b_col0 = b_n.begin * chunk
        b_v = b_col0 // res_cols
        b_cols = b_col0 % res_cols + hl.arange(chunk)
        b_x = out[b_t.begin, b_v, b_cols].to(torch.float32)
        if res_addend:
            b_x = b_x + res_partial[b_t.begin, b_v, b_cols].to(torch.float32)
        x_next[b_t.begin, b_v, b_cols] = (
            (res_acc[b_t.begin, b_v, b_cols] * b_ca + b_x * b_cx)
            * b_norm
            * res_w[b_v, b_cols].to(torch.float32)
        ).to(torch.bfloat16)
    return out


def source_table(
    n_qkvr: int, q_width: int, kv_width: int, device: torch.device
) -> torch.Tensor:
    """The attention input each 128-column qkvr tile projects: k columns the k input, v columns the v input."""
    cols = torch.arange(0, n_qkvr, 128)
    src = ((cols >= q_width).int() + (cols >= q_width + kv_width).int()) * (
        cols < q_width + 2 * kv_width
    ).int()
    return src.to(torch.int32).to(device)


# KV splits of A (a split takes every SPLITS-th cached page).
SPLITS = 16
DYN_PARTS = 9  # PD's K parts (512 columns each)
