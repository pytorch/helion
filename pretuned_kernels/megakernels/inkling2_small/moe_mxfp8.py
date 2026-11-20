"""Inkling-2-Small router + EP routed experts of one rank (MXFP8 layers 40-47) as one Helion kernel, on vLLM's layouts
(ink_moe_mx6.py on SGLang's): TRT-LLM expert weights (rows shuffled in blocks of 32) with their R128c4 block
scales as is (a K step's two 512-byte atoms per 128-row tile: tensor-descriptor loads, tcgen05.cp) and the paged
conv-state cache (each stream's conv input at every absolute position).

Stock (SGLang inkling2 MXFP8, T <= 32 verify): Inkling2Gate (as NVFP4), mxfp8_quantize of the token rows (e8m0
exponent per 32 values, e4m3), routed_swap_ab_matmul GEMM1 (W13, interleaved gate/up rows, bf16 out),
mxfp8_swiglu_quantize_routed (capped SwiGLU x gamma, bf16, MXFP8), GEMM2 (W2, bf16 out), routed_rows.

Roots (rank r, local experts [r*E, r*E + E)):
    G1  (128-expert router tile, K part) -> fp32 logit partials (the router is replicated: every rank)  [3 x 9]
    G2  token -> biased-sigmoid stable top-8 ids and their renormalized weights                      [T]
    Q   (token, 256 columns) -> MXFP8 x (the same for every expert; pad rows stay 0)                 [T x 18]
    R1  (128-row W13 tile, distinct local expert) -> the tile's 64 SwiGLU columns as two MXFP8       [36 x S]
        blocks of every token row (its own blocks: no |max| hop; block-scaled MMAs take 128 rows)
    R2  (128-row W2 tile, distinct local expert) -> bf16 routed rows of the tokens routing to it     [36 x S]
    C   (token, 256 columns) -> this rank's sum of the token's rows (polled in-band)                 [T x 18]
    J   (token, 256 columns) -> the routed rows' sum (pushed by their owners), mlp_sconv, residual  [T x 18]
Token rows pad to 16 (dot_scaled needs N >= 16); slot j is the j-th distinct local expert (ascending).
"""

from __future__ import annotations

from pretuned_kernels.megakernels.inkling2_small._common import _distinct_table
from pretuned_kernels.megakernels.inkling2_small._common import _expert_of
from pretuned_kernels.megakernels.inkling2_small._common import _local_ids
from pretuned_kernels.megakernels.inkling2_small._common import _mxfp8_block
from pretuned_kernels.megakernels.inkling2_small._common import _renorm
from pretuned_kernels.megakernels.inkling2_small._common import _scale_tile
from pretuned_kernels.megakernels.inkling2_small._common import _select
from pretuned_kernels.megakernels.inkling2_small._fp4 import capped_swiglu
import torch

import helion.language as hl


def _ink_moe_mx(
    x: torch.Tensor,  # [R, H] bf16: the MoE input (mlp_norm output), replicated; rows past T zero
    gate_w: torch.Tensor,  # [N/128, 128, H/128, 128] bf16 view of the router weight [N, H]
    gate_b: torch.Tensor,  # [N] fp32 router bias
    logit_part: torch.Tensor,  # [S_G, P, R, 128] fp32 workspace: router logit K-part partials, P = next_pow2(N/128)
    ids: torch.Tensor,  # [T * K] int32 out: global expert ids, token-major
    gammas: torch.Tensor,  # [T * K] fp32 out: routing weights (route_scale applied)
    w13: torch.Tensor,  # [E, 2I/128, 128, H/256, 256] e4m3 view (mx_views): rows gate (2i) / up (2i+1)
    w13_s: torch.Tensor,  # [E, 2I/128, H/256, 8, 128] uint8 view of vLLM's R128c4 e8m0 scales (one per 32 values)
    w2: torch.Tensor,  # [E, H/128, 128, I/256, 256] e4m3 view
    w2_s: torch.Tensor,  # [E, H/128, I/256, 8, 128] uint8 view of the R128c4 scales
    xq: torch.Tensor,  # [R, H/256, 256] e4m3 workspace (R = 16 token rows)
    xs: torch.Tensor,  # [R, H/256, 8] uint8 workspace, used as [H/256, R, 8] (xs4): a K step's 16 rows contiguous
    zq: torch.Tensor,  # [S, R, I/256, 256] e4m3: the SwiGLU rows of each distinct local expert (pad rows 0)
    zs: torch.Tensor,  # [S, R, I/256, 8] uint8 workspace, used as [S, I/256, R, 8] (zs4)
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
    res_hist: torch.Tensor,  # [NB, T, H/V, V] bf16: the attn_res blocks (written by the boundary attention kernels)
    res_rstd: torch.Tensor,  # [NB, T] fp32: each block's rows' rstd (RMS over H, eps 1e-6)
    res_dots: torch.Tensor,  # [NB, NQ, T] fp32: each block's rows . every (H^-0.5-scaled) attn_res query
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
    hl.specialize(w2.size())
    hl.specialize(w2.stride())
    hl.specialize(w2_s.size())
    hl.specialize(w2_s.stride())
    hl.specialize(xq.size())
    hl.specialize(xq.stride())
    hl.specialize(xs.size())
    hl.specialize(xs.stride())
    hl.specialize(zq.size())
    hl.specialize(zq.stride())
    hl.specialize(zs.size())
    hl.specialize(zs.stride())
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
    hl.specialize(res_hist.size())
    hl.specialize(res_hist.stride())
    hl.specialize(res_rstd.size())
    hl.specialize(res_rstd.stride())
    hl.specialize(res_dots.size())
    hl.specialize(res_dots.stride())
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
    t13, k13 = w13.size(1), w13.size(3)  # 128-row W13 tiles, 256-value K steps
    t2, k2 = w2.size(1), w2.size(3)  # 128-row W2 tiles, 256-value K steps
    z_cols = (
        w13.size(2) // 2
    )  # each W13 tile's SwiGLU columns (two 32-value MXFP8 blocks)
    x_step = xq.size(2)
    z_step = zq.size(3)
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
    slots = zq.size(0)
    base = rank * experts
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
    # The activation scales by K step: a step's [16 rows, 8] tile is 128 contiguous bytes, a tensor-descriptor box
    # (xsf / zsf), so every operand of the scaled MMAs streams through the ring.
    xs4 = xs.view(xs.size(1), xs.size(0), xs.size(2))
    xsf = xs.view(xs.size(1), xs.size(0) * xs.size(2))
    zs4 = zs.view(zs.size(0), zs.size(2), zs.size(1), zs.size(3))
    zsf = zs.view(zs.size(0), zs.size(2), zs.size(1) * zs.size(3))

    # The code outgrows the launch prefetch and weights streamed between launches evict it from L2.
    hl.prefetch_code()

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

    # G2. Stock: route_topk (K parts summed in order, biased sigmoid, stable top-8 by key desc / id asc,
    # logsigmoid-renormalized weights x route_scale).
    for g2_t in hl.tile(tokens, block_size=1):
        logits = torch.sum(logit_part[:, :, g2_t.begin, :], dim=0).reshape(router_p)
        eids = hl.arange(router_p)
        bias = hl.load(gate_b, [eids], extra_mask=eids < router_n)
        sel_ids = _select(logits, bias, router_n, router_p)
        ids[g2_t.begin * topk + hl.arange(topk)] = sel_ids
        gammas[g2_t.begin * topk + hl.arange(topk)] = _renorm(
            logits, sel_ids, router_p, route_scale
        )

    # Q. Stock: mxfp8_quantize of each token row; one task per (token row, 256-column chunk = one R1 K step).
    for q_t, q_c in hl.tile([tokens, k13], block_size=[1, 1]):
        xc = x[q_t.begin, q_c.begin * x_step + hl.arange(x_step)].to(torch.float32)
        q_values, q_exps = _mxfp8_block(xc.reshape(x_step // 32, 32))
        xq[q_t.begin, q_c.begin, :] = q_values.reshape(x_step)
        xs4[q_c.begin, q_t.begin, :] = q_exps

    # R1. Stock: routed_swap_ab_matmul GEMM1 (bf16 gate/up) + mxfp8_swiglu_quantize_routed (capped SwiGLU x gamma,
    # bf16, MXFP8 per 32 columns: this tile's 64 columns are two blocks of every row).
    for r1_q, r1_j in hl.tile([t13, slots], block_size=[1, 1]):
        lids = _local_ids(hl.load(ids, [hl.arange(pairs)]), base, experts)
        words, counts, before, distinct = _distinct_table(lids)
        if r1_j.begin < distinct:
            e = _expert_of(words, counts, before, r1_j.begin)
            # gamma of (token row, expert): the routing weight where the token routes to it, else 0 (pad rows too)
            slot = hl.arange(rows)[:, None] * topk + hl.arange(topk)[None, :]
            live = slot < pairs
            rid = hl.load(ids, [slot], extra_mask=live)
            rg = hl.load(gammas, [slot], extra_mask=live)
            gamma = torch.sum(torch.where((rid == e + base) & live, rg, 0.0), dim=-1)
            acc = hl.zeros([128, rows], dtype=torch.float32)
            for r1_k in hl.tile(k13, block_size=1):
                lhs = w13[e, r1_q.begin, :, r1_k.begin, :]
                lhs_s = _scale_tile(w13_s[e, r1_q.begin, r1_k.begin, :, :], 2)
                rhs = xq[:, r1_k.begin, :]
                rhs_s = xsf[r1_k.begin, :].reshape(rows, 8)
                acc = hl.dot_scaled(
                    lhs,
                    lhs_s,
                    "e4m3",
                    rhs.permute(1, 0),
                    rhs_s,
                    "e4m3",
                    acc=acc,
                    out_dtype=torch.float32,
                )
            # The tile's rows in checkpoint order (stored row 8 b + a of each 32 is row 4 a + b), then (gate, up) pairs.
            y = (
                acc.reshape(4, 4, 8, rows)
                .permute(0, 2, 1, 3)
                .reshape(z_cols, 2, rows)
                .permute(0, 2, 1)
            )
            gate, up = hl.split(y)
            gate = gate.to(torch.bfloat16).to(torch.float32)
            up = up.to(torch.bfloat16).to(torch.float32)
            v = (
                capped_swiglu(gate, up, gamma[None, :], gate_cap, up_cap)
                .to(torch.bfloat16)
                .to(torch.float32)
            )
            z_values, z_exps = _mxfp8_block(v.permute(1, 0).reshape(rows * 2, 32))
            z_col = r1_q.begin * z_cols
            hl.store(
                zq,
                [
                    r1_j.begin,
                    hl.arange(rows),
                    z_col // z_step,
                    z_col % z_step + hl.arange(z_cols),
                ],
                z_values.reshape(rows, z_cols),
            )
            hl.store(
                zs4,
                [
                    r1_j.begin,
                    z_col // z_step,
                    hl.arange(rows),
                    z_col % z_step // 32 + hl.arange(2),
                ],
                z_exps.reshape(rows, 2),
            )

    # R2. Stock: GEMM2 (W2, bf16 out) + routed_rows (bf16 rows of the routed tokens).
    for r2_o, r2_j in hl.tile([t2, slots], block_size=[1, 1]):
        lids3 = _local_ids(hl.load(ids, [hl.arange(pairs)]), base, experts)
        words3, counts3, before3, distinct3 = _distinct_table(lids3)
        if r2_j.begin < distinct3:
            e2 = _expert_of(words3, counts3, before3, r2_j.begin)
            slot2 = hl.arange(rows)[:, None] * topk + hl.arange(topk)[None, :]
            live2 = slot2 < pairs
            rid2 = hl.load(ids, [slot2], extra_mask=live2)
            match = ((rid2 == e2 + base) & live2).to(torch.int32)
            route = torch.sum(match * slot2, dim=-1)
            routed = torch.sum(match, dim=-1) > 0
            acc2 = hl.zeros([128, rows], dtype=torch.float32)
            for r2_k in hl.tile(k2, block_size=1):
                lhs2 = w2[e2, r2_o.begin, :, r2_k.begin, :]
                lhs2_s = _scale_tile(w2_s[e2, r2_o.begin, r2_k.begin, :, :], 2)
                rhs2 = zq[r2_j.begin, :, r2_k.begin, :]
                rhs2_s = zsf[r2_j.begin, r2_k.begin, :].reshape(rows, 8)
                acc2 = hl.dot_scaled(
                    lhs2,
                    lhs2_s,
                    "e4m3",
                    rhs2.permute(1, 0),
                    rhs2_s,
                    "e4m3",
                    acc=acc2,
                    out_dtype=torch.float32,
                )
            val = (
                acc2.reshape(4, 4, 8, rows)
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
        combined = hl.zeros([taps, 256], dtype=torch.float32)
        for peer_part in ppart_peers:
            combined = combined + peer_part[j_rows, j_c.begin, :]
        combined = combined.to(torch.bfloat16).to(torch.float32)
        j_pos = seq_len[0] - tokens + j_src  # the taps' absolute positions
        j_hist = (~j_in) & (j_pos >= 0)
        j_safe = torch.clamp_min(j_pos, 0)
        j_blk = hl.load(conv_pages, [j_safe // conv_block], extra_mask=j_hist)
        j_crow = (j_blk * conv_block + j_safe % conv_block) * conv_heads + j_head
        prior = hl.load(
            conv_cache, [j_crow[:, None], j_hcol[None, :]], extra_mask=j_hist[:, None]
        ).to(torch.float32)
        value = torch.where(j_in[:, None], combined, prior)
        weight = (
            conv_w[cols, :].to(torch.bfloat16).to(torch.float32).permute(1, 0)
        )  # stock rounds it to bf16
        conv_acc = torch.sum(value * weight, dim=0)
        current = torch.sum(
            torch.where((j_tap == taps - 1)[:, None], value, 0.0), dim=0
        )
        j_loc = conv_loc[j_t.begin]
        hl.store(
            conv_cache,
            [j_loc * conv_heads + j_head, j_hcol],
            current.to(torch.bfloat16),
            extra_mask=(hl.zeros([256], dtype=torch.int64) + j_loc) >= 0,
        )
        y = ((conv_acc + current) * conv_scale).to(torch.bfloat16).to(torch.float32)
        total = (
            partial[j_t.begin, cols].to(torch.float32)
            + attention[j_t.begin, cols].to(torch.float32)
            + y
        )
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
