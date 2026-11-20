# ruff: noqa: ANN001, ANN002, ANN201, ANN202
# pyrefly: ignore-errors
"""One GPT-OSS-120B decoder layer for an EAGLE3 speculative-decoding verify
step as a single gluon megakernel (NVIDIA SM100): batch one, T = 4 tokens
(the last accepted token and three draft tokens, a causal chain).

Same roots as ``gpt_oss_decode_layer``, widened to 4 tokens:

  1. QKV projection: the operand has 8 rows per token (its 4 weighted expert
     outputs, its residual, 3 zero rows); the MMA's N reduction sums each
     token's group.  Writes ``x_res``, the layer's input residual, which is
     also EAGLE3's captured hidden state on the capture layers.
  2. attention: (kv head, chunk, query token) tasks; each query sees the
     cached tokens in its window (split in 4 chunks) and the new tokens up to
     itself (chunk 0, with the sink); chunk 0 also writes the token's K/V.
  3. O projection: N = (head, token, chunk) columns, merged in the epilogue.
  4. post-norm: (channel slice, token pair) tasks; the 4 tokens' 4 MXFP4 terms
     are the experts' 16 operand rows.  The last task to finish (an acq_rel
     counter) routes: the tokens' distinct experts (at most 16) spread evenly
     over 16 expert slots, with each slot's per-token weight and rank.
  5. expert gate/up and 6. down over the 16 slots (slot-fastest order).  A
     slot computes all 4 tokens (its weight bytes are the cost; the tokens are
     free) and keeps those routed to its expert; each token's output lands in
     its (token, rank) rows, written once.  An empty slot reads expert 128,
     past the weights' end: the TMA zero-fills it without memory traffic.

The expert weights are passed as views with their true extents (w13 K 1440 of
1536 bytes, w2 rows 2880 of 3072 and K 1440 of 1472): TMA zero-fills the
padding instead of streaming it.

``correctness_check()`` compares one layer against an FP32 reference on
synthetic weights (no checkpoint, SGLang or FlashInfer needed); ``main()``
also reports the cold-L2 latency.  End to end in SGLang on one GB200
(GPT-OSS-120B, batch 1, EAGLE3 with 3 steps and 4 draft tokens), this kernel
serves 979 tok/s against SGLang's 803 (1.22x) at the same acceptance length.

    PYTHONPATH=. python pretuned_kernels/megakernels/gpt_oss_120b/gpt_oss_verify_layer.py
"""

from __future__ import annotations

from typing import Any

from pretuned_kernels.megakernels.gpt_oss_120b._common import HIDDEN
from pretuned_kernels.megakernels.gpt_oss_120b._common import SLIDING_WINDOW
from pretuned_kernels.megakernels.gpt_oss_120b._common import check
from pretuned_kernels.megakernels.gpt_oss_120b._common import e2m1_exponent
from pretuned_kernels.megakernels.gpt_oss_120b._common import e2m1_term
from pretuned_kernels.megakernels.gpt_oss_120b._common import make_context
from pretuned_kernels.megakernels.gpt_oss_120b._common import make_layer
from pretuned_kernels.megakernels.gpt_oss_120b._common import pack_nibbles
from pretuned_kernels.megakernels.gpt_oss_120b._common import reference_attention
from pretuned_kernels.megakernels.gpt_oss_120b._common import reference_moe
from pretuned_kernels.megakernels.gpt_oss_120b._common import rope
from pretuned_kernels.megakernels.gpt_oss_120b._common import rope_cos_sin
import torch

import helion
import helion.language as hl

TOKENS = 4


def _inv_rms4(ssq):
    """[4] per-token inverse RMS from ``ssq`` [4, 64] partial sums of squares."""
    tt = hl.arange(4)
    c = hl.arange(64)
    return torch.rsqrt(
        torch.sum(hl.load(ssq, [tt[:, None], c[None, :]]), dim=1) * (1.0 / 2880.0)
        + 1e-5
    )


def _route_table(router_acc, ssq, router_b):
    """``_route4`` for all 16 slots at once: expert [16] (128: none), per-token
    weight [16, 4] and rank [16, 4] (-1: not routed)."""
    tt = hl.arange(4)
    idx = hl.arange(128)
    inv_rms = _inv_rms4(ssq)
    logits = (
        hl.load(router_acc, [tt[:, None], idx[None, :]]) * inv_rms[:, None]
        + hl.load(router_b, [idx]).to(torch.float32)[None, :]
    )
    r = hl.arange(4)
    ids4 = hl.zeros([4, 4], dtype=torch.int32)  # [token, rank]
    v4 = hl.zeros([4, 4], dtype=torch.float32)
    lg = logits
    for k in hl.static_range(4):
        mk = torch.amax(lg, dim=1)
        ik = torch.amin(torch.where(lg == mk[:, None], idx[None, :], 128), dim=1)
        ids4 = torch.where(r[None, :] == k, ik[:, None], ids4)
        v4 = torch.where(r[None, :] == k, mk[:, None], v4)
        lg = torch.where(idx[None, :] == ik[:, None], -3.0e38, lg)
    p4 = torch.exp(v4 - torch.amax(v4, dim=1)[:, None])
    p4 = p4 / torch.sum(p4, dim=1)[:, None]
    picks = ids4.reshape(16)
    ii = hl.arange(16)
    earlier = ii[None, :] < ii[:, None]
    seen = torch.sum(
        ((picks[:, None] == picks[None, :]) & earlier).to(torch.int32), dim=1
    )
    first = seen == 0
    distinct = torch.sum(first.to(torch.int32), dim=0)
    order = torch.sum((first[None, :] & earlier).to(torch.int32), dim=1)
    hit = first[None, :] & (
        ((order * 16) // distinct)[None, :] == ii[:, None]
    )  # [slot, pick]
    active = torch.sum(hit.to(torch.int32), dim=1)
    expert = torch.where(
        active > 0, torch.sum(torch.where(hit, picks[None, :], 0), dim=1), 128
    )
    match = ids4[None, :, :] == expert[:, None, None]  # [slot, token, rank]
    weight = torch.sum(torch.where(match, p4[None, :, :], 0.0), dim=2)
    rank = torch.sum(torch.where(match, r[None, None, :] + 1, 0), dim=2) - 1
    return expert, weight, rank


def _rope4(v, cs, heads: int):
    """Neox RoPE of ``v`` [4, heads * 64] with ``cs`` [4, 64] (cos | sin per
    token) -> [4, heads, 64]."""
    x1, x2 = hl.split(v.reshape(4, heads, 2, 32).permute(0, 1, 3, 2))
    cos, sin = hl.split(cs.reshape(4, 2, 32).permute(0, 2, 1))
    o1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
    o2 = x2 * cos[:, None, :] + x1 * sin[:, None, :]
    return hl.join(o1, o2).permute(0, 1, 3, 2).reshape(4, heads, 64)


# The projection and expert operands, their scales and biases stream through
# the gluon TMA ring (their indexing slots below); every other access is a
# pointer access.  Each projection/expert K loop gets an MMA partition, so a
# task's epilogue overlaps the next task's MMAs.  Derived from the source by
# ``_common.ring_config_fields`` (rerun it after editing the kernel).
_RING_INDEXING = (2, 3, 33, 34, 79, 80, 81, 82, 83, 89, 90, 91, 92, 93)
CONFIG = {
    "indexing": [
        "tensor_descriptor" if i in _RING_INDEXING else "pointer" for i in range(96)
    ],
    "range_warp_specializes": [
        None,
        True,
        None,
        True,
        None,
        True,
        None,
        None,
        True,
        None,
        True,
    ],
    "pid_type": "persistent_blocked",
    "num_sm_multiplier": 1,
    "num_stages": 8,
    "num_warps": 4,
    "cross_loop_pipeline": "static",
}


@helion.kernel(
    config=helion.Config(**CONFIG),
    static_shapes=True,
    backend="gluon",
    ignore_warnings=[helion.exc.FastMathUnused],
)
def gpt_oss_verify_layer(
    x_h,  # [4, 2880] fp32 previous post-attention residual (layer 0: embedding)
    part_in,  # [4, 4, 2880] fp32 previous weighted expert outputs (token, rank)
    qkv_w,  # [600 * 3 * 128, 64] bf16: QKV weight times ln1, tile-contiguous
    bq_in,  # [32, 15, 192] bf16: row 8t + r: token t's expert output r (r < 4), residual (r = 4)
    qkv_b,  # [5120] bf16
    cos_sin,  # [P, 64] fp32 (cos | sin)
    positions,  # [4] int64 (consecutive)
    out_loc,  # [4] int64: the new tokens' slots in this layer's pool
    page_table,  # [R, W] int32 (row 0)
    window,  # [1] int32: 127 sliding / -1 full
    k_cache,  # [slots * 512] bf16 (flat view of [slots, 8, 64])
    v_cache,  # [slots * 512] bf16
    sinks,  # [64] fp32
    o_w,  # [2880, 16, 256] bf16 (view of [2880, 4096])
    o_b,  # [2880] bf16
    ln2_w,  # [2880] bf16
    router_w,  # [128, 2880] bf16
    router_b,  # [128] bf16
    w13,  # [E*5888, 1440] u8 (view of [E*5888, 1536]: TMA zero-fills the K padding)
    w13_sp,  # [E*46, 24, 32, 16] u8 packed scales
    w13_b,  # [E, 5888] fp32
    w2,  # [E, 2880, 1440] u8 (view of [E*3072, 1472]: TMA zero-fills the padding rows and K)
    w2_sp,  # [E*24, 23, 32, 16] u8 packed scales
    w2_b,  # [E, 3072] fp32
    x_res,  # [4, 2880] fp32: this layer's input residual (the EAGLE3 capture)
    ssq_a,  # [4, 64] fp32 scratch (columns >= 15 stay zero)
    qkv_acc,  # [5120, 4] fp32 scratch (split-K atomics; zero on entry, re-zeroed by the post-norm)
    m_part,  # [64, 4, 4] fp32 scratch: per (q head, token, chunk) running max
    l_part,  # [64, 4, 4] fp32 scratch: per (q head, token, chunk) softmax sum
    attn_b,  # [16, 64, 256] bf16 scratch: block-diagonal O operand (zeros off-diagonal)
    op_part,  # [4, 16, 2880] fp32 scratch: (token, K split) partials
    y_q,  # [16, 1536] u8 scratch: row 4 * term + token
    y_sp,  # [24, 32, 16] u8 scratch (unused rows hold 127)
    a_q,  # [16, 16, 1536] u8 scratch: per slot, row 4 * term + token
    a_s,  # [16, 24, 32, 16] u8 scratch: packed intermediate scales (unused bytes hold 127)
    ssq_m,  # [4, 64] fp32 scratch (columns >= 45 stay zero)
    router_acc,  # [4, 128] fp32 scratch: router dots (atomics)
    slot_ex,  # [16, 4] int32 scratch: each slot's expert (128: none), repeated
    slot_w,  # [16, 4] fp32 scratch: each slot's per-token routing weight (0: not routed)
    slot_rk,  # [16, 4] int32 scratch: each slot's expert's rank per token (-1: not routed)
    fw_ex,  # [16, 4] int32 scratch: slot_ex as forwarded by the gate/up tasks
    fw_w,  # [16, 4] fp32 scratch
    fw_rk,  # [16, 4] int32 scratch
    route_cnt,  # [1] int32 scratch: finished post-norm tasks (zero between launches)
    bq_out,  # [32, 2880] bf16: this layer's bq_in for the next layer
    h_out,  # [4, 2880] fp32: this layer's post-attention residual
    part_out,  # [4, 4, 2880] fp32: this layer's weighted expert outputs (token, rank)
):
    # (no hl.prefetch_code(): all 36 layers run this one kernel, its text stays in L2;
    # the prefetching CTAs only delayed their first MMAs)
    hl.pdl_launch_dependents()
    hl.pdl_wait()
    # 1. QKV projection, 15 K splits of 192 (split-K atomics).  Row tile 0
    # also forms the FP32 residual slice and its sums of squares, and zeroes
    # the router accumulator.
    for tile_m, tile_n, tile_s in hl.tile([5120, 32, 15], block_size=[128, 32, 1]):
        lead = tile_m.begin + tile_s.begin * 0
        tt = hl.arange(4)
        col = tile_s.begin * 192 + hl.arange(256)
        cm = (hl.arange(256) < 192) & ((col * 0 + lead) == 0)
        x = hl.load(
            x_h, [tt[:, None], col[None, :]], extra_mask=cm[None, :]
        ) + torch.sum(
            hl.load(
                part_in,
                [tt[:, None, None], hl.arange(4)[None, :, None], col[None, None, :]],
                extra_mask=cm[None, None, :],
            ),
            dim=1,
        )
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(192, block_size=64):
            blk = (tile_m.begin // 128 * 15 + tile_s.begin) * 3 + tile_k.begin // 64
            acc = hl.dot(
                qkv_w[blk * 128 + hl.arange(128), :],
                bq_in[tile_n, tile_s.begin, tile_k].T,
                acc=acc,
            )
        hl.atomic_add(
            qkv_acc,
            [tile_m.begin + hl.arange(128), tt],
            torch.sum(acc.reshape(128, 4, 8), dim=2),
        )
        hl.store(x_res, [tt[:, None], col[None, :]], x, extra_mask=cm[None, :])
        sq = torch.sum(torch.where(cm[None, :], x * x, 0.0), dim=1)
        hl.store(ssq_a, [tt, tile_s.begin], sq, extra_mask=(tt * 0 + lead) == 0)
        zi = hl.arange(128)
        hl.store(
            router_acc,
            [tt[:, None], zi[None, :]],
            hl.zeros([4, 128], dtype=torch.float32),
            extra_mask=(zi[None, :] * 0 + tt[:, None] * 0 + lead + tile_s.begin) == 0,
        )
    # 2. attention: (kv head, chunk, query token) tasks.  Query t (position
    # pos + t) sees the cached tokens [lo_t, pos) (split in 4 chunks) and the
    # new tokens 0..t (chunk 0, with the sink); chunk 0 also writes token t's
    # K/V to the cache.
    for tile_g, tile_c, tile_q in hl.tile([8, 4, 4], block_size=[1, 1, 1]):
        g = tile_g.begin
        tq = tile_q.begin
        tt = hl.arange(4)
        pos = hl.load(positions, [0]).to(torch.int32)
        win = hl.load(window, [0])
        d = hl.arange(64)
        hj = hl.arange(8)
        jq = hl.arange(512)
        lo = torch.where(win >= 0, torch.clamp(pos + tq - win, min=0), 0)
        t0 = lo // 64
        nt = torch.where(pos > lo, (pos - 1) // 64 - t0 + 1, 0)
        per = (nt + 3) // 4
        tb = t0 + torch.minimum(tile_c.begin * per, nt)
        te = t0 + torch.minimum((tile_c.begin + 1) * per, nt)
        page0 = hl.load(page_table, [0, torch.where(tb < te, tb, 0)])
        idx0 = (page0.to(torch.int64) * 64 + d)[:, None] * 512 + g * 64 + d[None, :]
        kt0 = hl.load(k_cache, [idx0])  # [tok, d]
        vt0 = hl.load(v_cache, [idx0])  # [tok, d]
        q_raw = hl.load(qkv_acc, [g * 512 + jq, tq])
        k_raw = hl.load(qkv_acc, [(4096 + g * 64 + d)[None, :], tt[:, None]])  # [4, 64]
        v_raw = hl.load(qkv_acc, [(4608 + g * 64 + d)[None, :], tt[:, None]])
        q_bias = hl.load(qkv_b, [g * 512 + jq]).to(torch.float32)
        k_bias = hl.load(qkv_b, [4096 + g * 64 + d]).to(torch.float32)
        v_bias = hl.load(qkv_b, [4608 + g * 64 + d]).to(torch.float32)
        sink_raw = hl.load(sinks, [g * 8 + hj])
        cs = hl.load(cos_sin, [pos + tq, d])
        cs4 = hl.load(cos_sin, [(pos + tt)[:, None], d[None, :]])  # [4, 64]
        inv = _inv_rms4(ssq_a)
        inv_q = torch.sum(torch.where(tt == tq, inv, 0.0), dim=0)
        qb = rope(q_raw * inv_q + q_bias, cs, 8).to(torch.bfloat16)  # [8, 64]
        kb = (
            _rope4(k_raw * inv[:, None] + k_bias[None, :], cs4, 1)
            .reshape(4, 64)
            .to(torch.bfloat16)
        )
        vb = (v_raw * inv[:, None] + v_bias[None, :]).to(torch.bfloat16)
        hl.prefetch(router_w, [(g * 16 + tile_c.begin * 4 + tq) % 128, slice(None)])
        # the new tokens i <= t (chunk 0), one key at a time, and the sink
        first = (hj + tile_c.begin * 8) < 8  # [8] all true on chunk 0
        qf = qb.to(torch.float32)
        kf = kb.to(torch.float32)
        vf = vb.to(torch.float32)
        m_i = torch.where(first, sink_raw, -1.0e30)
        l_i = torch.where(first, 1.0, 0.0)
        acc = hl.zeros([64, 8], dtype=torch.float32)  # o^T [d, head]
        for i in hl.static_range(4):
            k_i = torch.sum(torch.where(tt[:, None] == i, kf, 0.0), dim=0)
            v_i = torch.sum(torch.where(tt[:, None] == i, vf, 0.0), dim=0)
            ok = first & ((hj * 0 + tq) >= i)
            s_i = torch.where(ok, torch.sum(qf * k_i[None, :], dim=1) * 0.125, -1.0e30)
            m_new = torch.maximum(m_i, s_i)
            alpha = torch.exp(m_i - m_new)
            e_i = torch.where(ok, torch.exp(s_i - m_new), 0.0)
            l_i = l_i * alpha + e_i
            acc = acc * alpha[None, :] + v_i[:, None] * e_i[None, :]
            m_i = m_new
        q_t = qb.T  # [64, 8]
        # the cached tiles: the next tile's K/V load before this one's math
        kt = kt0
        vt = vt0
        for tile_t in hl.tile(tb, te, block_size=1):
            nxt = tile_t.begin + 1
            pagen = hl.load(page_table, [0, torch.where(nxt < te, nxt, tb)])
            idxn = (pagen.to(torch.int64) * 64 + d)[:, None] * 512 + g * 64 + d[None, :]
            ktn = hl.load(k_cache, [idxn])
            vtn = hl.load(v_cache, [idxn])
            tok = tile_t.begin * 64 + d
            s = hl.dot(kt, q_t) * 0.125  # [tok, head]
            valid = (tok >= lo) & (tok < pos)
            s = torch.where(valid[:, None], s, -1.0e30)
            m_new = torch.maximum(m_i, torch.amax(s, dim=0))
            p = torch.where(valid[:, None], torch.exp(s - m_new[None, :]), 0.0)
            alpha = torch.exp(m_i - m_new)
            l_i = l_i * alpha + torch.sum(p, dim=0)
            acc = acc * alpha[None, :] + hl.dot(vt.T, p.to(torch.bfloat16))
            m_i = m_new
            kt = ktn
            vt = vtn
        o = acc / torch.clamp(l_i, min=1.0e-30)[None, :]  # [d, head]
        hl.store(m_part, [g * 8 + hj, tq, tile_c.begin], m_i)
        hl.store(l_part, [g * 8 + hj, tq, tile_c.begin], l_i)
        # attn_b[s, n, k]: s = 2g + h // 4, n = 16 (h % 4) + 4 t + c, k = 64 (h % 4) + d
        jj = hj[:, None]
        hl.store(
            attn_b,
            [
                g * 2 + jj // 4,
                (jj % 4) * 16 + tq * 4 + tile_c.begin,
                (jj % 4) * 64 + d[None, :],
            ],
            o.T.to(torch.bfloat16),
        )
        loc = hl.load(out_loc, [tq])
        kv_mask = (d + tile_c.begin * 64) < 64
        k_t = torch.sum(torch.where(tt[:, None] == tq, kf, 0.0), dim=0)
        v_t = torch.sum(torch.where(tt[:, None] == tq, vf, 0.0), dim=0)
        hl.store(
            k_cache,
            [loc * 512 + g * 64 + d],
            k_t.to(torch.bfloat16),
            extra_mask=kv_mask,
        )
        hl.store(
            v_cache,
            [loc * 512 + g * 64 + d],
            v_t.to(torch.bfloat16),
            extra_mask=kv_mask,
        )
    # 4. O projection, 16 K splits of 4 heads; N = (head, token, chunk)
    # columns, merged in the epilogue with the chunks' softmax weights
    for tile_m, tile_n, tile_s in hl.tile([2880, 64, 16], block_size=[128, 64, 1]):
        jj = hl.arange(4)
        tt = hl.arange(4)
        cc = hl.arange(4)
        mi = [
            tile_s.begin * 4 + jj[:, None, None],
            tt[None, :, None],
            cc[None, None, :],
        ]
        mc = hl.load(m_part, mi)  # [head, token, chunk]
        lc = hl.load(l_part, mi)
        mx = torch.amax(mc, dim=2)
        wc = lc * torch.exp(mc - mx[:, :, None])
        wc = wc / torch.sum(wc, dim=2)[:, :, None]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(256, block_size=64):
            acc = hl.dot(
                o_w[tile_m, tile_s.begin, tile_k],
                attn_b[tile_s.begin, tile_n, tile_k].T,
                acc=acc,
            )
        v = (acc * wc.reshape(64)[None, :]).reshape(128, 4, 4, 4)
        om = tile_m.begin + hl.arange(128)
        hl.store(
            op_part,
            [hl.arange(4)[None, :], tile_s.begin + om[:, None] * 0, om[:, None]],
            torch.sum(torch.sum(v, dim=3), dim=1),
            extra_mask=(om < 2880)[:, None],
        )
    # 5. post-norm statistics, router partials, 4-term MXFP4 activation
    for tile_c, tile_q in hl.tile([2880, 2], block_size=[64, 1]):
        # (channel slice, token pair) tasks: every load first, then each
        # token's math; the router partials and the activation (what the
        # experts wait for) go out before this layer's outputs for the next one
        cl = tile_c.begin + hl.arange(64)
        q2 = tile_q.begin * 2 + hl.arange(2)
        xr = hl.load(x_res, [q2[:, None], cl[None, :]])  # [2, 64]
        op = hl.load(
            op_part,
            [q2[:, None, None], hl.arange(16)[None, :, None], cl[None, None, :]],
        )  # [2, 16, 64]
        ob = hl.load(o_b, [cl]).to(torch.float32)
        ln2 = hl.load(ln2_w, [cl]).to(torch.float32)
        rw = hl.load(router_w, [hl.arange(128)[:, None], cl[None, :]]).to(
            torch.float32
        )  # [128, 64]
        xa = xr + torch.sum(op, dim=1) + ob[None, :]
        for j in hl.static_range(2):
            qt = tile_q.begin * 2 + j
            x = torch.sum(torch.where(hl.arange(2)[:, None] == j, xa, 0.0), dim=0)
            yu = x * ln2
            hl.atomic_add(
                router_acc, [qt, hl.arange(128)], torch.sum(rw * yu[None, :], dim=1)
            )
            hl.store(ssq_m, [qt, tile_c.id], torch.sum(x * x, dim=0))
            r = yu.reshape(2, 32)
            e0 = e2m1_exponent(r)
            for jj in hl.static_range(4):
                e = e0 - 2.0 * jj
                code, r = e2m1_term(r, e)
                hl.store(
                    y_q,
                    [jj * 4 + qt, tile_c.begin // 2 + hl.arange(32)],
                    pack_nibbles(code, 2).reshape(32),
                )
                hl.store(
                    y_sp,
                    [
                        tile_c.begin // 128,
                        jj * 4 + qt,
                        (tile_c.begin // 32) % 4 + hl.arange(2),
                    ],
                    (e + 127.0).to(torch.uint8),
                )
            hl.store(h_out, [qt, cl], x)
            hl.store(bq_out, [qt * 8 + 4, cl], x.to(torch.bfloat16))
        zi = tile_c.id * 128 + hl.arange(128)
        hl.store(
            qkv_acc,
            [zi[:, None], hl.arange(4)[None, :]],
            hl.zeros([128, 4], dtype=torch.float32),
            extra_mask=(zi < 5120)[:, None] & ((zi * 0 + tile_q.begin) == 0)[:, None],
        )
        # the last task to finish routes: every slot's expert, weights, ranks
        done = hl.atomic_add(route_cnt, [0], 1, sem="acq_rel")
        if done == 89:
            expert, weight, rank = _route_table(router_acc, ssq_m, router_b)
            tt = hl.arange(4)
            si = hl.arange(16)
            hl.store(
                slot_ex,
                [si[:, None], tt[None, :]],
                (expert[:, None] + tt[None, :] * 0).to(torch.int32),
            )
            hl.store(slot_w, [si[:, None], tt[None, :]], weight)
            hl.store(slot_rk, [si[:, None], tt[None, :]], rank.to(torch.int32))
            hl.store(route_cnt, [0], done * 0)
    # 6. expert gate/up + SwiGLU, 4-term MXFP4 intermediate (slot-fastest
    # task order: a worker's consecutive tasks are consecutive slots)
    # (the 46th row tile is the intermediate's padding: its a_q columns stay zero)
    for tile_s, tile_m, tile_n in hl.tile([16, 5760, 16], block_size=[1, 128, 16]):
        ex = hl.load(slot_ex, [tile_s.begin, 0])
        inv_rms = _inv_rms4(ssq_m)
        # forward the slot's routing to the down tasks (their only producer
        # is then this root, so they keep per-slot waits)
        t4 = hl.arange(4)
        hl.store(fw_ex, [tile_s.begin, t4], hl.load(slot_ex, [tile_s.begin, t4]))
        hl.store(fw_w, [tile_s.begin, t4], hl.load(slot_w, [tile_s.begin, t4]))
        hl.store(fw_rk, [tile_s.begin, t4], hl.load(slot_rk, [tile_s.begin, t4]))
        rows = ex * 5888 + tile_m.begin
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(1536, block_size=128):
            acc = hl.dot_scaled(
                w13[rows + hl.arange(128), tile_k],
                w13_sp[
                    ex * 46 + tile_m.begin // 128,
                    tile_k.begin // 64 + hl.arange(2),
                    :,
                    :,
                ]
                .reshape(2, 32, 4, 4)
                .permute(2, 1, 0, 3)
                .reshape(128, 8),
                "e2m1",
                y_q[tile_n, tile_k].T,
                y_sp[tile_k.begin // 64 + hl.arange(2), :, :]
                .reshape(2, 32, 4, 4)
                .permute(2, 1, 0, 3)
                .reshape(128, 8),
                "e2m1",
                acc=acc,
            )
        # columns 4 * term + token: sum each token's terms
        pre = (
            torch.sum(acc.reshape(128, 4, 4), dim=1) * inv_rms[None, :]
            + hl.load(w13_b, [ex, tile_m.begin + hl.arange(128)])[:, None]
        ).T  # [token, row]
        up, gate = hl.split(pre.reshape(4, 4, 2, 2, 8).permute(0, 1, 2, 4, 3))
        gate = torch.clamp(gate, max=7.0)
        up = torch.clamp(up, min=-7.0, max=7.0)
        act = (up + 1.0) * gate * torch.sigmoid(1.702 * gate)  # [4, 4, 2, 8]
        r = act.permute(0, 1, 3, 2).reshape(4, 2, 32)
        term = hl.arange(4)
        q16 = hl.zeros([4, 4, 32], dtype=torch.int32)
        s16 = hl.full([4, 4, 2], 127.0, dtype=torch.float32)
        e0 = e2m1_exponent(r)  # [4, 2]
        for t in hl.static_range(4):
            e = e0 - 2.0 * t
            code, r = e2m1_term(r, e)
            packed = pack_nibbles(code.reshape(8, 32), 8).reshape(4, 32).to(torch.int32)
            q16 = torch.where(term[:, None, None] == t, packed[None, :, :], q16)
            s16 = torch.where(term[:, None, None] == t, (e + 127.0)[None, :, :], s16)
        a_q[tile_s.begin, tile_n, tile_m.begin // 4 + hl.arange(32)] = q16.reshape(
            16, 32
        ).to(torch.uint8)
        a_s[
            tile_s.begin,
            tile_m.begin // 256,
            tile_n,
            (tile_m.begin // 64) % 4 + hl.arange(2),
        ] = s16.reshape(16, 2).to(torch.uint8)
    # 7. expert down projection: each routed token's output, weighted by its
    # routing probability, into its (token, rank) rows
    for tile_s, tile_m, tile_n in hl.tile([16, 2944, 16], block_size=[1, 128, 16]):
        ex = hl.load(fw_ex, [tile_s.begin, 0])
        weight = hl.load(fw_w, [tile_s.begin, hl.arange(4)])
        rank = hl.load(fw_rk, [tile_s.begin, hl.arange(4)])
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        # 12 K steps of 128 B; the last one's upper half reads past the 1472
        # columns (TMA zero fill; the intermediate's padding there is zero)
        for tile_k in hl.tile(1536, block_size=128):
            acc = hl.dot_scaled(
                w2[ex, tile_m.begin + hl.arange(128), tile_k],
                w2_sp[
                    ex * 24 + tile_m.begin // 128,
                    tile_k.begin // 64 + hl.arange(2),
                    :,
                    :,
                ]
                .reshape(2, 32, 4, 4)
                .permute(2, 1, 0, 3)
                .reshape(128, 8),
                "e2m1",
                a_q[tile_s.begin, tile_n, tile_k].T,
                a_s[tile_s.begin, tile_k.begin // 64 + hl.arange(2), :, :]
                .reshape(2, 32, 4, 4)
                .permute(2, 1, 0, 3)
                .reshape(128, 8),
                "e2m1",
                acc=acc,
            )
        tt = hl.arange(4)
        o = (
            torch.sum(acc.reshape(128, 4, 4), dim=1)
            + hl.load(w2_b, [ex, tile_m.begin + hl.arange(128)])[:, None]
        ) * weight[None, :]
        p = tile_m.begin + hl.arange(128)
        logical = (p // 32) * 32 + (p % 8) * 4 + (p % 32) // 8
        keep = (logical < 2880)[:, None] & (rank >= 0)[None, :]
        rk = torch.clamp(rank, min=0)
        hl.store(
            part_out, [tt[None, :], rk[None, :], logical[:, None]], o, extra_mask=keep
        )
        hl.store(
            bq_out,
            [(tt * 8 + rk)[None, :], logical[:, None]],
            o.to(torch.bfloat16),
            extra_mask=keep,
        )
    return h_out, part_out


# (position, window, token similarity): similarity 1 gives the 4 tokens one
# input (they then share most experts, so most expert slots are empty)
CHECK_CASES = (
    (1000, SLIDING_WINDOW, 0.0),
    (1000, -1, 0.0),
    (37, -1, 0.0),
    (400, -1, 1.0),
)
BENCHMARK_CASES = ((400, SLIDING_WINDOW), (400, -1), (4000, -1))


def make_scratch(device: torch.device) -> tuple[torch.Tensor, ...]:
    """The kernel's scratch operands, in argument order (after ``x_res``).
    Zero (or 127) on first use; the kernel leaves them ready for its next
    launch."""
    f32, bf16, u8, i32 = torch.float32, torch.bfloat16, torch.uint8, torch.int32

    def z(*shape, dtype=f32):
        return torch.zeros(*shape, dtype=dtype, device=device)

    return (
        z(TOKENS, 64),  # ssq_a
        z(5120, TOKENS),  # qkv_acc
        z(64, TOKENS, 4),  # m_part
        z(64, TOKENS, 4),  # l_part
        z(16, 64, 256, dtype=bf16),  # attn_b
        z(TOKENS, 16, HIDDEN),  # op_part
        z(16, 1536, dtype=u8),  # y_q
        # y_sp, a_s: each 16-byte scale row on its own 128-byte line (every
        # expert task reads the same rows each K step)
        torch.full((24, 32, 128), 127, dtype=u8, device=device)[:, :, :16],
        z(16, 16, 1536, dtype=u8),  # a_q
        torch.full((16, 24, 32, 128), 127, dtype=u8, device=device)[:, :, :, :16],
        z(TOKENS, 64),  # ssq_m
        z(TOKENS, 128),  # router_acc
        z(16, 4, dtype=i32),  # slot_ex
        z(16, 4),  # slot_w
        z(16, 4, dtype=i32),  # slot_rk
        z(16, 4, dtype=i32),  # fw_ex
        z(16, 4),  # fw_w
        z(16, 4, dtype=i32),  # fw_rk
        z(1, dtype=i32),  # route_cnt
    )


def kernel_args(served, ctx, cos_sin, x_h, part_in, x_res, scratch, outputs) -> tuple:
    """Operands for one layer: ``x_h`` [4, 2880] and ``part_in`` [4, 4, 2880]
    are the previous layer's FP32 outputs, ``x_res`` [4, 2880] receives this
    layer's input residual; ``outputs`` = (bq_out, h_out, part_out)."""
    bq_in = torch.zeros(TOKENS, 8, HIDDEN, dtype=torch.bfloat16, device=x_h.device)
    bq_in[:, :4] = part_in.to(torch.bfloat16)
    bq_in[:, 4] = x_h.to(torch.bfloat16)
    experts = served["w13_b"].size(0)
    return (
        x_h, part_in, served["qkv_w"], bq_in.view(8 * TOKENS, 15, 192), served["qkv_b"],
        cos_sin, ctx.positions, ctx.out_loc, ctx.page_table, ctx.window,
        ctx.k_cache.view(-1), ctx.v_cache.view(-1), served["sinks"],
        served["o_w"], served["o_b"], served["ln2"], served["router_w"], served["router_b"],
        # the expert weights at their true extents: TMA zero-fills the padding
        served["w13"][:, :1440], served["w13_sp"], served["w13_b"],
        served["w2"].view(experts, 3072, 1472)[:, :HIDDEN, :1440], served["w2_sp"], served["w2_b"],
        x_res, *scratch, *outputs,
    )  # fmt: skip


def _make_case(served, cos_sin, pos: int, win: int, similarity: float, device):
    gen = torch.Generator(device=device).manual_seed(pos)
    ctx = make_context(pos, win, tokens=TOKENS, seed=pos, device=device)
    x_h = torch.randn(TOKENS, HIDDEN, generator=gen, device=device) * 2.0
    part_in = torch.randn(TOKENS, 4, HIDDEN, generator=gen, device=device) * 0.5
    x_h = similarity * x_h[:1] + (1 - similarity) * x_h
    part_in = similarity * part_in[:1] + (1 - similarity) * part_in
    x_res = torch.zeros(TOKENS, HIDDEN, device=device)
    outputs = (
        torch.zeros(8 * TOKENS, HIDDEN, dtype=torch.bfloat16, device=device),
        torch.zeros(TOKENS, HIDDEN, device=device),
        torch.zeros(TOKENS, 4, HIDDEN, device=device),
    )
    scratch = make_scratch(device)
    args = kernel_args(served, ctx, cos_sin, x_h, part_in, x_res, scratch, outputs)
    return ctx, x_h, part_in, x_res, scratch, outputs, args


def correctness_check(verbose: bool = False) -> None:
    """One layer against the FP32 reference (the captured residual,
    attention, the K/V writes, the MoE, and the next layer's operand rows)."""
    device = torch.device("cuda")
    raw, served = make_layer(seed=0, device=device)
    cos_sin = rope_cos_sin(device)
    for pos, win, similarity in CHECK_CASES:
        ctx, x_h, part_in, x_res, scratch, outputs, args = _make_case(
            served, cos_sin, pos, win, similarity, device
        )
        bq_out, h_out, part_out = outputs
        gpt_oss_verify_layer(*args)
        x = x_h + part_in.sum(1)
        h_ref, k_ref, v_ref = reference_attention(raw, ctx, cos_sin, x)
        moe_ref, ids = reference_moe(raw, h_out)
        bq = bq_out.view(TOKENS, 8, HIDDEN)
        errors = {
            "x_res": check("x_res", x_res, x, 1e-6),
            "attention": check("attention", h_out - x, h_ref - x, 2e-2),
            "k write": check("k write", ctx.k_cache[ctx.out_loc], k_ref, 2e-2),
            "v write": check("v write", ctx.v_cache[ctx.out_loc], v_ref, 2e-2),
            "moe": max(
                check(f"moe token {t}", part_out[t].sum(0), moe_ref[t], 1e-2)
                for t in range(TOKENS)
            ),
            "bq experts": check("bq experts", bq[:, :4], part_out, 1e-2),
            "bq residual": check("bq residual", bq[:, 4], h_out, 1e-2),
        }
        if int(scratch[-1].item()) != 0:
            raise AssertionError("route_cnt was not reset for the next launch")
        if verbose:
            cells = "  ".join(f"{k} {v:.2e}" for k, v in errors.items())
            print(
                f"pos {pos:5d} window {win:4d} distinct experts "
                f"{ids.unique().numel():2d}: {cells}"
            )


def use_cudagraph() -> bool:
    return True


@torch.inference_mode()
def main(verbose: bool = True) -> dict[str, Any]:
    """Check against the FP32 reference, then report cold-L2 CUDA-graph
    latencies of one layer."""
    from pretuned_kernels._bench import bench_pre_captured_cudagraph
    from pretuned_kernels._bench import capture_cuda_graph

    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("gpt_oss_verify_layer is pretuned for SM100")
    correctness_check(verbose=verbose)
    device = torch.device("cuda")
    _, served = make_layer(seed=0, device=device)
    cos_sin = rope_cos_sin(device)
    latency = {}
    for pos, win in BENCHMARK_CASES:
        args = _make_case(served, cos_sin, pos, win, 0.0, device)[-1]
        graph, _ = capture_cuda_graph(lambda args=args: gpt_oss_verify_layer(*args))
        us = bench_pre_captured_cudagraph(graph.replay) * 1000
        latency[f"pos{pos}_window{win}"] = us
        if verbose:
            print(f"pos {pos:5d} window {win:4d}: {us:.2f} us")
    return {"latency_us": latency}


if __name__ == "__main__":
    main()
