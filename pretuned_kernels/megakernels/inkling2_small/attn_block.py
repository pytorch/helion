"""Inkling-2-Small local (sliding-window) attention half of one rank as one Helion kernel (TP4, batch-1 verify
window of T = 8 rows): QKVR GEMM .. K/V short convs + clips / norm .. rel-bias window attention .. gate-sink ..
wo_ud .. cross-rank all-reduce (in-band) + attn_sconv, on vLLM's layouts (ink_attn_v13.py on SGLang's): qkvr
columns q | r | gate | sinks | k | v, the paged KV cache ([blocks, KV heads, block size, K | V]) and the paged
conv-state cache (each stream's conv input at every absolute position).

Stock (SGLang inkling2 verify, padded FA4 path): qkvr (bf16 GEMM), attention_prologue (K/V verify sconv with the
conv-state prefix + window save, K L2 clip, V RMSNorm, Q / rel L2 clips with learnable scales, rel = r @ proj; all
bf16), FA4 paged attention (window_left 1023, rel bias added before the softmax scale; the new K/V rows stored
into the paged cache first), gate_sink (out * sigmoid(gate) * sigmoid(lse - sink)), wo_ud (bf16), push all-reduce
(rank-order fp32 sum of the bf16 partials), attn_sconv (verify window).

Roots (rank-local: 12 query heads in 2 KV groups of 6, padded to 8; head_dim 128):
    P   (128 qkvr columns, K split) -> fp32 partials                                        [30 x S]
    PQR (token, head) -> clipped q (padded heads stay 0), clipped r and its rel bias row    [T x 12]
    PK  (KV head, token) -> K / V convs, K clip, V norm -> the new rows (kv_new)            [2 x T]
    A   (KV group, 128-key page of the window) -> partial attention of the 8 x 8 rows over  [2 x 16]
        the cached keys (positions before the T new rows: A never waits for PK)
    AN  (KV group) -> partial attention over the T new keys (kv_new)                        [2]
    M   (token, KV group) -> the new rows into the KV cache (after every A read it); the    [T x 2]
        pages + AN merged, gate-sink -> o_in
    O   (128 hidden columns, K quarter = half a KV group) -> fp32 wo partials               [36 x 4]
    C   (token, 128 columns) -> the bf16 wo row into the symmetric buffer (polled in-band)  [T x 36]
    J   (token, 128 columns) -> every rank's rows summed (the all-reduce), attn_sconv       [T x 36]
"""

from __future__ import annotations

from pretuned_kernels.megakernels.inkling2_small._common import LN2
from pretuned_kernels.megakernels.inkling2_small._common import LOG2E
from pretuned_kernels.megakernels.inkling2_small._common import _l2_clip
from pretuned_kernels.megakernels.inkling2_small._common import _rsqrt_approx
from pretuned_kernels.megakernels.inkling2_small._common import _sigmoid
from pretuned_kernels.megakernels.inkling2_small._common import _verify_conv
import torch

import helion.language as hl


def _ink_attn_local(
    x: torch.Tensor,  # [R, H] bf16: the attention input (attn_norm output); rows past T zero
    w_qkvr: torch.Tensor,  # [N, S, H/S] bf16 view of this rank's qkvr weight [N, H] (N = 3792: q|r|gate|sinks|k|v|pad)
    part: torch.Tensor,  # [S, R, N] fp32 workspace: the qkvr split-K partials
    k_conv_w: torch.Tensor,  # [Dkv, 4] bf16: k_sconv weight (Dkv = 2 heads x 128)
    v_conv_w: torch.Tensor,  # [Dkv, 4] bf16
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
    att_ws: torch.Tensor,  # [2, PAGES, 64, 128] fp32 workspace: per-page normalized partial outputs
    lse_ws: torch.Tensor,  # [2, PAGES, 64] fp32 workspace: per-page log2-sum-exp2 (-inf: no visible key)
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
    res_block: torch.Tensor,  # [T, H] bf16: this block-boundary layer's partial (a new block)
    res_queries: torch.Tensor,  # [2 Qp, H/D, D] bf16: every attn_res query (H^-0.5-scaled), bf16 high parts then lows
    res_dpart: torch.Tensor,  # [H/D, 3, Qp, R] fp32 workspace: by K chunk, the block's dots with the query highs, lows,
    #                           and (at query 0) its square sums
    res_hist: torch.Tensor,  # [NB, T, H/D, D] bf16 out: the blocks (slot res_slot[0] gets res_block's rows)
    res_rstd: torch.Tensor,  # [NB, T] fp32 out: their rows' rstd (RMS over H, eps 1e-6)
    res_dots: torch.Tensor,  # [NB, NQ, T] fp32 out: their rows . every query
    res_slot: torch.Tensor,  # [1] int32: the new block's index
    res_q: torch.Tensor,  # [H/V, V] fp32: the MLP input's attn_res query, scaled by H^-0.5
    res_w: torch.Tensor,  # [H/V, V]: mlp_norm's weight
    res_acc: torch.Tensor,  # [T, H/V, V] fp32: the blocks before the new one, softmax-weighted (the MoE kernel's F)
    res_stat: torch.Tensor,  # [T, H/C, 4] fp32: (their logit max, exp total, res_acc's square sum by columns, 0)
    res_part: torch.Tensor,  # [T, H/C, 8] fp32 workspace: J's sums over the new block b and the attention rows x
    x_next: torch.Tensor,  # [R, H/V, V] bf16 out: the MoE kernel's input rows (mlp_norm(attn_res(...)))
    group_name: hl.ProcessGroupName,
    world: hl.constexpr,
    rank: hl.constexpr,
) -> torch.Tensor:
    hl.specialize(x.size())
    hl.specialize(x.stride())
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
    hl.specialize(res_block.size())
    hl.specialize(res_block.stride())
    hl.specialize(res_queries.size())
    hl.specialize(res_queries.stride())
    hl.specialize(res_dpart.size())
    hl.specialize(res_dpart.stride())
    hl.specialize(res_hist.size())
    hl.specialize(res_hist.stride())
    hl.specialize(res_rstd.size())
    hl.specialize(res_rstd.stride())
    hl.specialize(res_dots.size())
    hl.specialize(res_dots.stride())
    hl.specialize(res_slot.size())
    hl.specialize(res_slot.stride())
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
    n_qkvr, k_splits, k_per = w_qkvr.size()
    tokens = q_ws.size(0)
    kv_heads, padded, head_dim = q_ws.size(1), q_ws.size(2), q_ws.size(3)
    heads = q_scale.size(0)
    group = heads // kv_heads
    d_rel = r_scale.size(1)
    rel_extent = rel_proj.size(1)
    n_pages = att_ws.size(1)
    rows = padded * tokens  # attention rows of a KV group: (token, padded head)
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    r_col = q_width
    gate_col = r_col + heads * d_rel
    sink_col = gate_col + q_width
    k_col = sink_col + heads
    v_col = k_col + kv_width
    o_parts, o_split = wo.size(1), wo.size(2)
    taps = k_conv_w.size(1)
    conv_scale = (taps + 1) ** -0.5
    x3 = x.view(x.size(0), k_splits, k_per)
    window_left = rel_extent - 1
    scale_log2 = head_dim**-0.5 * LOG2E
    head_cols = a_conv_cache.size(1)
    opart_peers = torch.ops.symm_mem.get_remote_tensors(opart, group_name)
    new_keys = kv_new.size(2)
    live_pages = (
        window_left + page_size - 1
    ) // page_size + 1  # pages a 1023-key cached window can touch
    chunk = opart.size(2)  # all-reduce columns a task
    d_k = res_hist.size(3)  # D1's K columns a task (res_hist's last dim)
    d_qp = res_dpart.size(2)  # query rows a half: highs [0, Qp), lows [Qp, 2 Qp)
    n_queries = res_dots.size(1)
    res_cols = res_q.size(1)
    stat_rows = (
        1 << (res_stat.size(1) - 1).bit_length()
    )  # F2 tasks a row, padded to a power of two

    att_flat = att_ws.view(
        att_ws.size(0), att_ws.size(1), att_ws.size(2) * att_ws.size(3)
    )  # M's [group, page, row * dim]
    hl.prefetch_code()

    # P. Stock: qkvr (bf16 x bf16 -> bf16; here fp32 partials by K split, summed in split order by the readers).
    for p_n, p_s in hl.tile([n_qkvr, k_splits], block_size=[128, 1]):
        p_acc = hl.zeros([p_n, x.size(0)], dtype=torch.float32)
        for p_k in hl.tile(k_per, block_size=128):
            p_acc = hl.dot(
                w_qkvr[p_n, p_s.begin, p_k],
                x3[:, p_s.begin, p_k].permute(1, 0),
                acc=p_acc,
            )
        part[p_s.begin, :, p_n] = p_acc.permute(1, 0)

    # D. The new block's per-row stats for every later attn_res (the MoE kernels' F read them): its rows . every
    # query (bf16 high + low query parts on the tensor cores, fp32 accumulation: ~fp32 dots) and their rstd, by K
    # chunk (D1, whose query-high tasks also copy the rows into the block history), summed in chunk order (D2).
    for d_m, d_c in hl.tile([2 * d_qp, hidden // d_k], block_size=[128, 1]):
        d_r = hl.arange(res_dpart.size(3))  # the MMA's rows: T, padded
        d_acc = hl.zeros([d_m, res_dpart.size(3)], dtype=torch.float32)
        for d_kk in hl.tile(d_k, block_size=128):
            d_xk = hl.load(
                res_block,
                [
                    d_r[:, None],
                    (d_c.begin * d_k + d_kk.begin + hl.arange(128))[None, :],
                ],
                extra_mask=(d_r < tokens)[:, None],
            )
            d_acc = hl.dot(
                res_queries[d_m, d_c.begin, d_kk], d_xk.permute(1, 0), acc=d_acc
            )
        res_dpart[
            d_c.begin, d_m.begin // d_qp, d_m.begin % d_qp + hl.arange(128), :
        ] = d_acc
        if d_m.begin == 0:
            d_x = hl.load(
                res_block,
                [d_r[:, None], (d_c.begin * d_k + hl.arange(d_k))[None, :]],
                extra_mask=(d_r < tokens)[:, None],
            ).to(torch.float32)
            res_dpart[d_c.begin, 2, 0, :] = torch.sum(d_x * d_x, dim=1)
            d_rows = hl.arange(tokens)
            res_hist[res_slot[0], d_rows, d_c.begin, :] = res_block[
                d_rows, d_c.begin * d_k + hl.arange(d_k)
            ]

    for e_t in hl.tile(tokens, block_size=1):
        e_dots = torch.sum(res_dpart[:, 0, :, e_t.begin], dim=0) + torch.sum(
            res_dpart[:, 1, :, e_t.begin], dim=0
        )
        e_i = hl.arange(d_qp)
        hl.store(
            res_dots, [res_slot[0], e_i, e_t.begin], e_dots, extra_mask=e_i < n_queries
        )
        e_ss = torch.sum(res_dpart[:, 2, 0, e_t.begin], dim=0)
        res_rstd[res_slot[0], e_t.begin] = torch.rsqrt(e_ss * (1.0 / hidden) + 1e-6)

    # PQR. Stock: q_l2_clip (learnable per-(head, dim) scale), bf16, written padded (head h -> group h // 6, h % 6);
    # rel_l2_clip (scale, threshold 8), bf16, rel_logits_proj (K = 16), bf16.
    for q_t, q_h in hl.tile([tokens, heads], block_size=[1, 1]):
        q_cols = q_h.begin * head_dim + hl.arange(head_dim)
        q_raw = (
            torch.sum(part[:, q_t.begin, q_cols], dim=0)
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        q_in = q_scale[q_h.begin, :].to(torch.float32) * q_raw
        q_ws[q_t.begin, q_h.begin // group, q_h.begin % group, :] = (
            _l2_clip(q_in[None, :], 22.627416997969522, 1e-6)
            .to(torch.bfloat16)
            .reshape(head_dim)
        )
        r_cols = r_col + q_h.begin * d_rel + hl.arange(d_rel)
        r_raw = (
            torch.sum(part[:, q_t.begin, r_cols], dim=0)
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        r_in = r_scale[q_h.begin, :].to(torch.float32) * r_raw
        r_bf = _l2_clip(r_in[None, :], 8.0, 1e-6).to(torch.bfloat16).to(torch.float32)
        rel = torch.sum(
            r_bf.reshape(d_rel)[:, None] * rel_proj[:, :].to(torch.float32), dim=0
        )
        rel_ws[q_t.begin, q_h.begin, :] = rel.to(torch.bfloat16)

    # PK. Stock: attention_prologue's K / V verify sconv (static weight, residual, (W + 1)^-0.5, window save),
    # K L2 clip, V RMSNorm (no gamma), bf16 at every step. The rows go to kv_new (M stores them into the cache).
    for k_g, k_t in hl.tile([kv_heads, tokens], block_size=[1, 1]):
        k_end = seq_len[0] - tokens
        kv_cols = k_g.begin * head_dim + hl.arange(head_dim)
        k_conv = _verify_conv(
            part,
            k_col,
            kv_cols,
            k_t.begin,
            k_conv_w,
            k_conv_cache,
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

    # PV. The V half of PK (its own tasks: the K and V convs run side by side).
    for v_g, v_t in hl.tile([kv_heads, tokens], block_size=[1, 1]):
        v_end = seq_len[0] - tokens
        vv_cols = v_g.begin * head_dim + hl.arange(head_dim)
        v_conv = _verify_conv(
            part,
            v_col,
            vv_cols,
            v_t.begin,
            v_conv_w,
            v_conv_cache,
            v_g.begin,
            kv_heads,
            conv_pages,
            conv_loc,
            conv_block,
            v_end,
            taps,
        )
        v_rstd = _rsqrt_approx(
            torch.sum(v_conv * v_conv, dim=0) * (1.0 / head_dim) + 1e-6
        )
        kv_new[1, v_g.begin, v_t.begin, :] = (v_conv * v_rstd).to(torch.bfloat16)

    # A. Stock: FA4 over the window (keys [pos - 1023, pos] of each query row), the rel bias added unscaled
    # (rel_bias_scale_after_add: q.k * scale + rel);
    # one task per (KV group, 128-key page): the (token, padded head) rows' softmax over the page's cached keys
    # (positions before the new rows, which AN covers).
    for a_g, a_p in hl.tile([kv_heads, n_pages], block_size=[1, 1]):
        length = seq_len[0]
        cache_end = length - tokens
        first_page = torch.clamp_min(cache_end - window_left, 0) // page_size
        page_idx = first_page + a_p.begin
        a_rows = hl.arange(rows)
        r_tok = a_rows // padded
        r_head = a_rows % padded
        # The page's cached keys first: nothing in this launch writes them before A (M stores the new rows after),
        # so they load ahead of the wait for PQR's q / rel rows.
        a_live = page_idx * page_size < cache_end
        page = hl.load(pages, [page_idx], extra_mask=a_live).to(torch.int64)
        keys = hl.arange(page_size)
        a_on = (hl.zeros([page_size], dtype=torch.int32) + a_live.to(torch.int32)) != 0
        slots = page * page_size + keys
        kk = hl.load(k_buf, [slots, a_g.begin, slice(None)], extra_mask=a_on[:, None])
        vv = hl.load(v_buf, [slots, a_g.begin, slice(None)], extra_mask=a_on[:, None])
        if a_live:
            kv_pos = page_idx * page_size + keys
            in_cache = kv_pos < cache_end
            qq = q_ws[:, a_g.begin, :, :].reshape(rows, head_dim)
            q_pos = cache_end + r_tok
            dist = q_pos[:, None] - kv_pos[None, :]
            valid = (
                in_cache[None, :] & (dist <= window_left) & (r_head < group)[:, None]
            )
            bias = hl.load(
                rel_ws,
                [r_tok[:, None], (a_g.begin * group + r_head)[:, None], dist],
                extra_mask=valid,
            )
            scores = hl.dot(qq, kk.permute(1, 0), out_dtype=torch.float32)
            logits = torch.where(
                valid,
                scores * scale_log2 + bias.to(torch.float32) * LOG2E,
                float("-inf"),
            )
            run_max = torch.amax(logits, dim=1)
            safe_max = torch.where(run_max > float("-inf"), run_max, 0.0)
            prob = torch.exp2(logits - safe_max[:, None])
            run_sum = torch.sum(prob, dim=1)
            pv = hl.dot(prob.to(torch.bfloat16), vv, out_dtype=torch.float32)
            seen = run_sum > 0.0
            att_ws[a_g.begin, a_p.begin, :, :] = (
                pv / torch.where(seen, run_sum, 1.0)[:, None]
            )
            lse_ws[a_g.begin, a_p.begin, :] = torch.where(
                seen, safe_max + torch.log2(run_sum), float("-inf")
            )
        else:  # a page past the cached window: weight 0 in M
            lse_ws[a_g.begin, a_p.begin, :] = hl.full(
                [rows], float("-inf"), dtype=torch.float32
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
    # One task per (token, head) (vLLM's combine grid): every page's partial in one load (no serial page chain).
    # A partial of weight 0 is still read (finite: A only writes normalized partials into the zeroed workspace).
    for m_t, m_h in hl.tile([tokens, heads], block_size=[1, 1]):
        m_g = m_h.begin // group
        m_r = (
            m_t.begin * padded + m_h.begin % group
        )  # the head's attention row in its group
        m_dims = hl.arange(head_dim)
        m_pages = hl.arange(n_pages)
        gate = torch.sum(
            hl.load(
                part, [slice(None), m_t.begin, gate_col + m_h.begin * head_dim + m_dims]
            ),
            dim=0,
        )
        gate = gate.to(torch.bfloat16).to(torch.float32)
        sink = torch.sum(
            hl.load(
                part, [slice(None), m_t.begin, sink_col + m_h.begin + hl.arange(1)]
            ),
            dim=0,
        )
        sink = sink.to(torch.bfloat16).to(torch.float32)
        m_lse = hl.load(lse_ws, [m_g, m_pages, m_r])  # [pages]
        n_lse = hl.load(lse_new, [m_g, m_r + hl.arange(1)])  # [1]
        top = torch.maximum(torch.amax(m_lse, dim=0), n_lse)
        top = torch.where(top > float("-inf"), top, 0.0)
        n_weight = torch.exp2(n_lse - top)
        p_weight = torch.exp2(m_lse - top)
        total = torch.sum(p_weight, dim=0) + n_weight
        att = hl.load(
            att_flat,
            [m_g, m_pages[:, None], m_r * head_dim + m_dims[None, :]],
            extra_mask=(m_pages < live_pages)[:, None],
        )
        merged = (
            torch.sum(att * p_weight[:, None], dim=0)
            + hl.load(att_new, [m_g, m_r, m_dims]) * n_weight
        )
        merged = merged / torch.where(total > 0.0, total, 1.0)
        lse = (top + torch.log2(total)) * LN2
        gated = merged * _sigmoid(gate)
        gated = gated * _sigmoid(lse - sink)
        m_col0 = (
            m_h.begin * head_dim
        )  # (head_dim divides o_split: a head's columns lie in one K quarter)
        hl.store(
            o_in,
            [m_t.begin, m_col0 // o_split, m_col0 % o_split + m_dims],
            gated.to(torch.bfloat16),
        )
        # The new rows into the cache (the group's first head), after every A task read it (M waits for them).
        m_loc = kv_loc[m_t.begin]
        m_ok = ((hl.zeros([head_dim], dtype=torch.int64) + m_loc) >= 0) & (
            (hl.zeros([head_dim], dtype=torch.int64) + m_h.begin % group) == 0
        )
        hl.store(
            k_buf, [m_loc, m_g, m_dims], kv_new[0, m_g, m_t.begin, :], extra_mask=m_ok
        )
        hl.store(
            v_buf, [m_loc, m_g, m_dims], kv_new[1, m_g, m_t.begin, :], extra_mask=m_ok
        )

    # O. Stock: wo_ud (bf16; here fp32 partials by K quarter, summed in order by C).
    for o_n, o_q in hl.tile([hidden, o_parts], block_size=[128, 1]):
        o_acc = hl.zeros([o_n, x.size(0)], dtype=torch.float32)
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

    # J. Stock: the push all-reduce (rank-order fp32 sum of the bf16 rows, bf16) fused with attn_sconv (residual,
    # (W + 1)^-0.5, bf16; taps before the verify rows from the paged conv cache, zero before position 0; each row's
    # conv input stored in the cache). Each peer's rows t - 3 .. t are polled as one tile (rows before the window
    # read row 0, polled anyway, and take the cache).
    for j_t, j_n in hl.tile([tokens, hidden // chunk], block_size=[1, 1]):
        j_cols = j_n.begin * chunk + hl.arange(chunk)
        # (a task's columns lie in one row of out's view: chunk divides res_cols)
        j_col0 = j_n.begin * chunk
        j_vcol = j_col0 % res_cols + hl.arange(chunk)
        # B's inputs first (not written by the peers: their loads overlap the poll).
        j_q = res_q[j_col0 // res_cols, j_vcol]
        j_a = res_acc[j_t.begin, j_col0 // res_cols, j_vcol]
        j_b = res_block[j_t.begin, j_cols].to(torch.float32)
        j_head = j_col0 // head_cols  # (chunk divides the stream's columns a head)
        j_hcol = j_col0 % head_cols + hl.arange(chunk)
        j_tap = hl.arange(taps)
        j_src = j_t.begin - (taps - 1) + j_tap
        j_in = j_src >= 0
        j_rows = torch.clamp_min(j_src, 0)
        # The conv taps' history and weights (no peer writes them) also load ahead of the poll.
        j_pos = seq_len[0] - tokens + j_src  # the taps' absolute positions
        j_hist = (~j_in) & (j_pos >= 0)
        j_safe = torch.clamp_min(j_pos, 0)
        j_blk = hl.load(conv_pages, [j_safe // conv_block], extra_mask=j_hist)
        j_crow = (j_blk * conv_block + j_safe % conv_block) * kv_heads + j_head
        j_prior = hl.load(
            a_conv_cache, [j_crow[:, None], j_hcol[None, :]], extra_mask=j_hist[:, None]
        ).to(torch.float32)
        j_w = a_conv_w[j_cols, :].to(torch.float32).permute(1, 0)
        j_loc = conv_loc[j_t.begin]
        reduced = hl.zeros([taps, chunk], dtype=torch.float32)
        for peer_part in opart_peers:
            reduced = reduced + peer_part[j_rows, j_n.begin, :].to(torch.float32)
        reduced = reduced.to(torch.bfloat16).to(torch.float32)
        j_value = torch.where(j_in[:, None], reduced, j_prior)
        j_acc = torch.sum(j_value * j_w, dim=0)
        j_current = torch.sum(
            torch.where((j_tap == taps - 1)[:, None], j_value, 0.0), dim=0
        )
        hl.store(
            a_conv_cache,
            [j_loc * kv_heads + j_head, j_hcol],
            j_current.to(torch.bfloat16),
            extra_mask=(hl.zeros([chunk], dtype=torch.int64) + j_loc) >= 0,
        )
        j_out = ((j_acc + j_current) * conv_scale).to(torch.bfloat16)
        out[j_t.begin, j_col0 // res_cols, j_vcol] = j_out
        # B's sums over the two new attn_res sources (the block b and the attention rows x; its share of the columns)
        # as one [8, chunk] reduction: (q . x, x . x, res_acc . x, q . b, b . b, res_acc . b, b . x, 0).
        j_x = j_out.to(torch.float32)
        j_i = hl.arange(8)[:, None]
        j_left = torch.where(j_i < 3, j_x[None, :], j_b[None, :])
        j_right = torch.where(
            (j_i == 0) | (j_i == 3),
            j_q[None, :],
            torch.where(
                (j_i == 1) | (j_i == 6),
                j_x[None, :],
                torch.where(
                    (j_i == 2) | (j_i == 5),
                    j_a[None, :],
                    torch.where(j_i == 4, j_b[None, :], 0.0),
                ),
            ),
        )
        res_part[j_t.begin, j_n.begin, :] = torch.sum(j_left * j_right, dim=1)

    # B. Stock: block_attn_res over the blocks and this layer's attention rows (normalize_values: softmax over the
    # sources of (q . x_n) * rstd(x_n), the rstd-normalized sources summed, fp32) + mlp_norm (RMSNorm, eps 1e-6,
    # bf16) of the MoE input, per (token, J's columns): the new block and the attention rows join the older blocks'
    # softmax (res_acc, res_stat: the previous MoE kernel's F); the sum's square from J's and F's sums.
    for b_t, b_n in hl.tile([tokens, hidden // chunk], block_size=[1, 1]):
        b_sums = torch.sum(res_part[b_t.begin, :, :], dim=0)  # in J's column order
        b_i = hl.arange(8)
        b_dx = torch.sum(torch.where(b_i == 0, b_sums, 0.0), dim=0)
        b_sx = torch.sum(torch.where(b_i == 1, b_sums, 0.0), dim=0)
        b_ax = torch.sum(torch.where(b_i == 2, b_sums, 0.0), dim=0)
        b_db = torch.sum(torch.where(b_i == 3, b_sums, 0.0), dim=0)
        b_sb = torch.sum(torch.where(b_i == 4, b_sums, 0.0), dim=0)
        b_ab = torch.sum(torch.where(b_i == 5, b_sums, 0.0), dim=0)
        b_bx = torch.sum(torch.where(b_i == 6, b_sums, 0.0), dim=0)
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
        b_rx = torch.rsqrt(b_sx * (1.0 / hidden) + 1e-6)
        b_rb = torch.rsqrt(b_sb * (1.0 / hidden) + 1e-6)
        b_lx = b_dx * b_rx
        b_lb = b_db * b_rb
        b_top = torch.maximum(torch.maximum(b_max, b_lb), b_lx)
        b_old = torch.exp(b_max - b_top)
        b_eb = torch.exp(b_lb - b_top)
        b_ex = torch.exp(b_lx - b_top)
        b_inv = 1.0 / (b_total * b_old + b_eb + b_ex)
        b_ca = b_old * b_inv
        b_cb = b_eb * b_rb * b_inv
        b_cx = b_ex * b_rx * b_inv
        b_hh = (
            b_ca * b_ca * b_sq
            + b_cb * b_cb * b_sb
            + b_cx * b_cx * b_sx
            + 2.0 * (b_ca * b_cb * b_ab + b_ca * b_cx * b_ax + b_cb * b_cx * b_bx)
        )
        b_norm = torch.rsqrt(b_hh * (1.0 / hidden) + 1e-6)
        b_col0 = b_n.begin * chunk
        b_v = b_col0 // res_cols
        b_cols = b_col0 % res_cols + hl.arange(chunk)
        b_x = out[b_t.begin, b_v, b_cols].to(torch.float32)
        b_b = res_block[b_t.begin, b_col0 + hl.arange(chunk)].to(torch.float32)
        x_next[b_t.begin, b_v, b_cols] = (
            (res_acc[b_t.begin, b_v, b_cols] * b_ca + b_b * b_cb + b_x * b_cx)
            * b_norm
            * res_w[b_v, b_cols].to(torch.float32)
        ).to(torch.bfloat16)
    return out
