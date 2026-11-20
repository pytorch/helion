"""Inkling-2-Small router + EP routed experts of one rank (NVFP4 layers 0-39) as one Helion kernel, on vLLM's layouts
(ink_moe_v6.py on SGLang's): TRT-LLM expert weights (rows shuffled in blocks of 32, R128c4 block scales) and the
paged conv-state cache (each stream's conv input at every absolute position).

Stock (SGLang inkling2, T <= 32 verify): Inkling2Gate (linear_k_group_partials + route_topk: sigmoid + bias stable
top-8, logsigmoid-renormalized weights x route_scale), routed_quant quantize (four-over-six NVFP4 of each routed
token row), routed_swap_matmul GEMM1 (W13, interleaved gate/up rows), swiglu_quantize_routed (scale, bf16 gate/up,
capped SwiGLU x gamma, bf16, four-over-six), GEMM2 (W2), routed_rows (scale, bf16 rows into the exchange buffer).

Roots (rank r, local experts [r*E, r*E + E)):
    G1  (128-expert router tile, K part) -> fp32 logit partials (the router is replicated: every rank)  [3 x 9]
    G2  token -> biased-sigmoid stable top-8 ids and their renormalized weights                      [T]
    Q   (token, 512 columns) -> four-over-six NVFP4 x (the same for every expert; pad rows stay 0) [T x 9]
    R1  (128-row W13 tile, distinct local expert) -> bf16 SwiGLU columns h and their |max|           [36 x S]
    Z   (256 columns, distinct local expert) -> four-over-six NVFP4 of h's live token rows          [9 x S]
    R2  (128-row W2 tile, distinct local expert) -> bf16 routed rows of the tokens routing to it     [36 x S]
    J   (token, 256 columns) -> the routed rows' sum (pushed by their owners), mlp_sconv, residual  [T x 18]
Token rows pad to 16 (dot_scaled needs N >= 16); slot j is the j-th distinct local expert (ascending).
"""

from __future__ import annotations

from pretuned_kernels.megakernels.inkling2_small._fp4 import capped_swiglu
from pretuned_kernels.megakernels.inkling2_small._fp4 import quant46
from pretuned_kernels.megakernels.inkling2_small._fp4 import quant46s
from pretuned_kernels.megakernels.inkling2_small._fp4 import rcp_rn
import torch

import helion.language as hl

E4M3 = torch.float8_e4m3fn


def _or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a | b


def _add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def _popcount(v: torch.Tensor) -> torch.Tensor:
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    return (((v + (v >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24


def _distinct_table(ids: torch.Tensor):
    """Local expert ids [n] (< 128; -1 = not local) -> bitmap words [4], popcounts, exclusive prefix, distinct count."""
    lanes = hl.arange(4)
    row = ids.unsqueeze(0)
    bits = torch.where((row >> 5) == lanes.unsqueeze(1), 1 << (row & 31), 0)
    words = hl.reduce(_or, bits, dim=1)
    counts = _popcount(words)
    before = hl.cumsum(counts, dim=0) - counts
    return words, counts, before, hl.reduce(_add, counts, dim=0)


def _expert_of(
    words: torch.Tensor, counts: torch.Tensor, before: torch.Tensor, j: object
) -> torch.Tensor:
    """The j-th distinct local expert (ascending)."""
    hit = (before <= j) & (before + counts > j)
    word = hl.reduce(_or, torch.where(hit, words, 0), dim=0)
    key = hl.reduce(_or, torch.where(hit, before * 8 + hl.arange(4), 0), dim=0)
    upto = _popcount(word & ((2 << hl.arange(32)) - 1))
    bit = hl.reduce(_add, (upto <= j - (key >> 3)).to(torch.int32), dim=0)
    return (key & 7) * 32 + bit


def _select(logits: torch.Tensor, bias: torch.Tensor, n: int, n_pad: int):
    """route_topk at top-8: the biased sigmoid's stable top-8 as eight max rounds over distinct packed keys (key
    desc, id asc: the sort's order)."""
    experts = hl.arange(n_pad)
    selection = rcp_rn(1.0 + torch.exp(0.0 - logits)) + bias
    kbits = (selection + 0.0).view(dtype=torch.int32)
    ordered = torch.where(kbits >= 0, kbits, kbits ^ 0x7FFFFFFF).to(torch.int64)
    packed = (ordered << 32) | (n_pad - 1 - experts).to(torch.int64)
    packed = torch.where(experts < n, packed, -(1 << 62))
    rank8 = hl.arange(8)
    best = hl.zeros([8], dtype=torch.int64)
    for kk in hl.static_range(8):
        top = torch.amax(packed, dim=0)
        best = torch.where(rank8 == kk, top, best)
        packed = torch.where(packed == top, -(1 << 62), packed)
    return (n_pad - 1) - (best & 0xFFFF).to(torch.int32)


def _renorm(
    logits: torch.Tensor, sel_ids: torch.Tensor, n_pad: int, route_scale: float
):
    """renorm_topk_logits_scaled's logsigmoid weights of the selected ids (its summation order)."""
    experts = hl.arange(n_pad)
    picked = torch.sum(
        torch.where(experts[None, :] == sel_ids[:, None], logits[None, :], 0.0), dim=-1
    )
    log_probs = torch.minimum(picked, hl.zeros([8], dtype=torch.float32)) - torch.log1p(
        torch.exp(0.0 - torch.abs(picked))
    )
    top_lp = torch.amax(log_probs, dim=0)
    shifted = torch.exp(log_probs - top_lp)
    total = (
        (_pick(shifted, 0) + _pick(shifted, 2))
        + (_pick(shifted, 1) + _pick(shifted, 3))
    ) + (
        (_pick(shifted, 4) + _pick(shifted, 6))
        + (_pick(shifted, 5) + _pick(shifted, 7))
    )
    return torch.exp(log_probs - (top_lp + torch.log(total))) * route_scale


def _pick(values: torch.Tensor, i: int) -> torch.Tensor:
    return torch.sum(torch.where(hl.arange(8) == i, values, 0.0), dim=0)


def _scale_tile(chunks: torch.Tensor, col_blocks: int) -> torch.Tensor:
    """Swizzled scale bytes of one 128-row tile over one K step ([col_blocks * 2, 256]: CUTLASS's 512-byte
    (32 row, 4 row group, 4 column) atoms, K-major) -> [128 rows, col_blocks * 4]."""
    return (
        chunks.reshape(1, col_blocks, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(128, col_blocks * 4)
    )


def _local_ids(ids: torch.Tensor, base: int, experts: int) -> torch.Tensor:
    local = ids - base
    return torch.where((local >= 0) & (local < experts), local, -1)


def _ink_moe_local(
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


def moe_views(
    w13: torch.Tensor, w13_s: torch.Tensor, w2: torch.Tensor, w2_s: torch.Tensor
):
    """vLLM's expert weights (rows shuffled in blocks of 32) and R128c4 block scales ([E, N, K / 16] E4M3, as
    uint8) as the kernel's (expert, 128-row tile, row, K step, byte) / (expert, 128-row tile, K step, chunk) views."""
    e, n13, kb13 = w13.shape
    _, h, kb2 = w2.shape
    return (
        w13.view(e, n13 // 128, 128, kb13 // 256, 256),
        w13_s.view(e, n13 // 128, kb13 // 256, 16, 256),
        w2.view(e, h // 128, 128, kb2 // 128, 128),
        w2_s.view(e, h // 128, kb2 // 128, 8, 256),
    )


# F tasks: (token, 512 columns)
F_CHUNK = 512
