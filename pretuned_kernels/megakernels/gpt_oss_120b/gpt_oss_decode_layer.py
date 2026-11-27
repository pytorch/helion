# ruff: noqa: ANN001, ANN002, ANN201, ANN202
# pyrefly: ignore-errors
"""One GPT-OSS-120B decoder layer for batch-one decode as a single gluon
megakernel (NVIDIA SM100).

Roots (top-level ``hl.tile`` loops, chained by the static cross-loop pipeline;
the projection and expert operands stream through the gluon TMA ring, and each
of their K loops gets an MMA partition):

  1. QKV projection (bf16 tcgen05, weight rows as M, 15 split-K partials
     added with FP32 atomics).  The weight carries ln1 and the RMS is applied
     after the projection, so the operand is the previous layer's raw
     outputs.
  2. attention: (kv head, chunk) tasks over the paged cache; RoPE, the
     attention sink, the new token's K/V write; online-softmax partials.
  3. O projection (bf16 tcgen05, 16 split-K partials), the chunks merged with
     their softmax weights in the epilogue.
  4. post-norm: the post-attention residual, router partials (FP32 atomics)
     and the 4-term MXFP4 activation (near-exact: the experts see the
     activation to ~2**-8 of its block maximum).
  5. expert gate/up + clamped SwiGLU (MXFP4 x MXFP4 tcgen05 ``dot_scaled``),
     4-term MXFP4 intermediate; each task routes (top-4 of the router dots).
  6. expert down projection, weighted by the routing probability.

The atomics' order varies from run to run, so the outputs are not bitwise
reproducible across runs (they stay within the reference tolerances).

The residual stream between layers stays FP32: a layer reads the previous
layer's post-attention residual and its four weighted expert outputs, and
writes its own.  All 36 layers of the model run this kernel (sliding-window
and full-attention layers differ only in ``window``).  Weights are SGLang's
served tensors (``moe_runner_backend=flashinfer_mxfp4``) but for QKV, which
takes ln1 folded in and is re-tiled per task (``_common.make_layer``): with its
QKV weight prepared that way, the kernel serves a running SGLang instance in
place of its per-layer launches.

``correctness_check()`` compares one layer against an FP32 reference on
synthetic weights (no checkpoint, SGLang or FlashInfer needed); ``main()``
also reports the cold-L2 latency.  End to end in SGLang on one GB200
(GPT-OSS-120B, batch 1), this kernel together with ``gpt_oss_verify_layer``
(for EAGLE3 verify steps) serves 979 tok/s against SGLang's 803 (1.22x).

    PYTHONPATH=. python pretuned_kernels/megakernels/gpt_oss_120b/gpt_oss_decode_layer.py
"""

from __future__ import annotations

from typing import Any

from pretuned_kernels.megakernels.gpt_oss_120b._common import EPS
from pretuned_kernels.megakernels.gpt_oss_120b._common import HIDDEN
from pretuned_kernels.megakernels.gpt_oss_120b._common import INV_HIDDEN
from pretuned_kernels.megakernels.gpt_oss_120b._common import SLIDING_WINDOW
from pretuned_kernels.megakernels.gpt_oss_120b._common import check
from pretuned_kernels.megakernels.gpt_oss_120b._common import clamped_swiglu
from pretuned_kernels.megakernels.gpt_oss_120b._common import e2m1_exponent
from pretuned_kernels.megakernels.gpt_oss_120b._common import e2m1_term
from pretuned_kernels.megakernels.gpt_oss_120b._common import make_context
from pretuned_kernels.megakernels.gpt_oss_120b._common import make_layer
from pretuned_kernels.megakernels.gpt_oss_120b._common import pack_nibbles
from pretuned_kernels.megakernels.gpt_oss_120b._common import reference_attention
from pretuned_kernels.megakernels.gpt_oss_120b._common import reference_moe
from pretuned_kernels.megakernels.gpt_oss_120b._common import rope
from pretuned_kernels.megakernels.gpt_oss_120b._common import rope_cos_sin
from pretuned_kernels.megakernels.gpt_oss_120b._common import scale_tiles
import torch

import helion
import helion.language as hl


def _route(router_acc, ssq_part, router_b, slot):
    """Top-4 of the router logits (renormalized softmax): the expert and weight
    of ``slot``.  Each logit's FP32 bits, as an order-preserving int, pack
    with its index into one int64 key (ties to the lower index, the verify
    layer's order), so a round of the top-4 is one max reduction.
    ``router_acc`` [128] holds the dots of the un-normalized hidden with the
    router rows; ``ssq_part`` per-chunk sums of squares."""
    idx = hl.arange(128)
    inv_rms = _inv_rms(ssq_part)
    logits = hl.load(router_acc, [idx]) * inv_rms + hl.load(router_b, [idx]).to(
        torch.float32
    )
    # (a negative float's magnitude bits reversed; picked keys drop below all)
    bits = logits.view(torch.int32)
    bits = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    key = bits.to(torch.int64) * 256 + (255 - idx).to(torch.int64)
    k0 = torch.amax(key, dim=0)
    key = torch.where(key == k0, -(1 << 40), key)
    k1 = torch.amax(key, dim=0)
    key = torch.where(key == k1, -(1 << 40), key)
    k2 = torch.amax(key, dim=0)
    key = torch.where(key == k2, -(1 << 40), key)
    k3 = torch.amax(key, dim=0)
    k = hl.arange(4)
    keys4 = torch.where(
        k == 0, k0, torch.where(k == 1, k1, torch.where(k == 2, k2, k3))
    )
    ids4 = 255 - (keys4 & 255)
    bits4 = (keys4 >> 8).to(torch.int32)
    v4 = torch.where(bits4 < 0, bits4 ^ 0x7FFFFFFF, bits4).view(torch.float32)
    p4 = torch.exp(v4 - torch.amax(v4, dim=0))
    p4 = p4 / torch.sum(p4, dim=0)
    expert = torch.sum(torch.where(k == slot, ids4, 0), dim=0)
    weight = torch.sum(torch.where(k == slot, p4, 0.0), dim=0)
    return expert, weight, inv_rms


def _inv_rms(ssq):
    c = hl.arange(64)
    return torch.rsqrt(torch.sum(hl.load(ssq, [c]), dim=0) * INV_HIDDEN + EPS)


# The projection and expert operands, their scales and biases stream through
# the gluon TMA ring (their indexing slots below); every other access is a
# pointer access.  Each projection/expert K loop gets an MMA partition, so a
# task's epilogue overlaps the next task's MMAs.  These are the fields
# ``_ring_config.ring_config_fields(bound, RING_LOADS, warp_specialize="k_loops")``
# derives from the bound kernel (test_verify_megakernels_codegen checks it).
_RING_INDEXING = (2, 3, 32, 33, 55, 56, 57, 58, 59, 65, 66, 67, 68, 69)
CONFIG = {
    "indexing": [
        "tensor_descriptor" if i in _RING_INDEXING else "pointer" for i in range(72)
    ],
    "range_warp_specializes": [
        None,
        True,
        None,
        None,
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
def gpt_oss_decode_layer(
    x_h,  # [2880] fp32 previous post-attention residual (layer 0: embedding)
    part_in,  # [4, 2880] fp32 previous weighted expert outputs (layer 0: zeros)
    qkv_w,  # [600 * 3 * 128, 64] bf16: QKV weight times ln1, tile-contiguous (each
    # (row tile, K split) task's 3 K steps of [128, 64] one 48 KB block)
    bq_in,  # [16, 15, 192] bf16: previous expert outputs (rows 0-3), residual (row 4)
    qkv_b,  # [5120] bf16
    cos_sin,  # [P, 64] fp32 (cos | sin)
    positions,  # [1] int64
    out_loc,  # [1] int64: the new token's slot in this layer's pool
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
    w13,  # [E*5888, 1536] u8
    w13_sp,  # [E*46, 24, 32, 16] u8 packed scales
    w13_b,  # [E, 5888] fp32
    w2,  # [E*3072, 1472] u8
    w2_sp,  # [E*24, 23, 32, 16] u8 packed scales
    w2_b,  # [E, 3072] fp32
    x_res,  # [2880] fp32 scratch
    ssq_a,  # [64] fp32 scratch (entries >= 15 stay zero)
    qkv_acc,  # [5120] fp32 scratch: QKV projection (split-K atomics; zero on entry, re-zeroed by the post-norm)
    m_part,  # [64, 16] fp32 scratch: per (q head, chunk) running max
    l_part,  # [64, 16] fp32 scratch: per (q head, chunk) softmax sum
    attn_b,  # [16, 64, 256] bf16 scratch: block-diagonal O operand (zeros off-diagonal)
    op_part,  # [16, 2880] fp32 scratch
    y_q,  # [16, 1536] u8 scratch (rows >= 4 stay zero)
    y_sp,  # [24, 32, 16] u8 scratch (unused rows hold 127)
    a_q,  # [4, 16, 1536] u8 scratch
    a_s,  # [4, 24, 32, 16] u8 scratch: packed intermediate scales (unused bytes hold 127)
    ssq_m,  # [64] fp32 scratch (rows >= 45 stay zero)
    router_acc,  # [128] fp32 scratch: router dots (atomics)
    bq_out,  # [16, 2880] bf16: this layer's bq_in for the next layer (rows 0-4
    # written; the others must be zero: the next layer's QKV sums all 16 rows)
    h_out,  # [2880] fp32: this layer's post-attention residual
    part_out,  # [4, 2880] fp32: this layer's weighted expert outputs
):
    # Code prefetch: 0.7-1.9 us faster on GB300 with cold L2 (at most about 1 us
    # in the verify layer, which skips it).
    hl.prefetch_code()
    # PDL: the next layer's kernel launches as this one's CTAs retire; this
    # one waits for the previous layer at entry
    hl.pdl_launch_dependents()
    hl.pdl_wait()
    # 1. QKV projection, 15 K splits of 192 (split-K atomics).  The weight
    # carries ln1 and the RMS applies after the projection, so the operand is
    # the previous layer's outputs as they are: the four weighted expert
    # outputs and the post-attention residual (bf16 rows, summed by the MMA's
    # N reduction).  Row tile 0 also forms the FP32 residual and the slice's
    # sum of squares; the router accumulator is zeroed here (the previous
    # layer's readers are done; this layer's post-norm adds into it later).
    for tile_m, tile_n, tile_s in hl.tile([5120, 16, 15], block_size=[128, 16, 1]):
        # row tile 0's residual slice: its loads go out before the K loop
        lead = tile_m.begin + tile_s.begin * 0
        col = tile_s.begin * 192 + hl.arange(256)
        cm = (hl.arange(256) < 192) & ((col * 0 + lead) == 0)
        x = hl.load(x_h, [col], extra_mask=cm) + torch.sum(
            hl.load(
                part_in, [hl.arange(4)[:, None], col[None, :]], extra_mask=cm[None, :]
            ),
            dim=0,
        )
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(192, block_size=64):
            blk = (tile_m.begin // 128 * 15 + tile_s.begin) * 3 + tile_k.begin // 64
            acc = hl.dot(
                qkv_w[blk * 128 + hl.arange(128), :],
                bq_in[tile_n, tile_s.begin, tile_k].T,
                acc=acc,
            )
        hl.atomic_add(qkv_acc, [tile_m], torch.sum(acc, dim=1))
        hl.store(x_res, [col], x, extra_mask=cm)
        sq = torch.sum(torch.where(cm, x * x, 0.0), dim=0)
        k4 = hl.arange(4)
        hl.store(
            ssq_a,
            [tile_s.begin + k4 * 0],
            sq + hl.zeros([4], dtype=torch.float32),
            extra_mask=(k4 == 0) & ((k4 * 0 + lead) == 0),
        )
        zi = hl.arange(128)
        hl.store(
            router_acc,
            [zi],
            hl.zeros([128], dtype=torch.float32),
            extra_mask=(zi * 0 + lead + tile_s.begin) == 0,
        )
    # 2. attention: (kv head, chunk) tasks over the cached tokens; chunk 0 also
    # takes the sink and the new token (and writes its K/V to the cache).  Each
    # task writes its chunk-normalized output into the block-diagonal O
    # operand (column (head, chunk)) and its (max, sum) for the merge.
    for tile_g, tile_c in hl.tile([8, 16], block_size=[1, 1]):
        g = tile_g.begin
        # every load the prologue needs first, then the math
        pos = hl.load(positions, [0]).to(torch.int32)
        win = hl.load(window, [0])
        d = hl.arange(64)
        hj = hl.arange(8)
        jq = hl.arange(512)
        lo = torch.where(win >= 0, torch.clamp(pos - win, min=0), 0)
        t0 = lo // 64
        nt = torch.where(pos > lo, (pos - 1) // 64 - t0 + 1, 0)
        per = (nt + 15) // 16
        t_begin = t0 + torch.minimum(tile_c.begin * per, nt)
        t_end = t0 + torch.minimum((tile_c.begin + 1) * per, nt)
        # the chunk's first KV tile (its page, then K and V) ahead of the rest:
        # its loads overlap the projection's
        has0 = t_begin < t_end
        page0 = hl.load(page_table, [0, torch.where(has0, t_begin, 0)])
        idx0 = (page0.to(torch.int64) * 64 + d)[:, None] * 512 + g * 64 + d[None, :]
        kt0 = hl.load(k_cache, [idx0])  # [tok, d]
        vt0 = hl.load(v_cache, [idx0])  # [tok, d]
        q_raw = hl.load(qkv_acc, [g * 512 + jq])
        k_raw = hl.load(qkv_acc, [4096 + g * 64 + d])
        v_raw = hl.load(qkv_acc, [4608 + g * 64 + d])
        q_bias = hl.load(qkv_b, [g * 512 + jq]).to(torch.float32)
        k_bias = hl.load(qkv_b, [4096 + g * 64 + d]).to(torch.float32)
        v_bias = hl.load(qkv_b, [4608 + g * 64 + d]).to(torch.float32)
        sink_raw = hl.load(sinks, [g * 8 + hj])
        inv = _inv_rms(ssq_a)
        cs = hl.load(cos_sin, [pos, d])
        q = rope(q_raw * inv + q_bias, cs, 8)
        kn = rope(k_raw * inv + k_bias, cs, 1)
        vn = v_raw * inv + v_bias
        qb = q.to(torch.bfloat16)
        kb = kn.to(torch.bfloat16)
        vb = vn.to(torch.bfloat16)
        # warm L2 with the router rows the post-norm reads (one row per task)
        hl.prefetch(router_w, [g * 16 + tile_c.begin, slice(None)])
        first = (hj + tile_c.begin * 8) < 8  # [8] all true on chunk 0
        s_new = torch.sum(qb.to(torch.float32) * kb.to(torch.float32), dim=1) * 0.125
        s_new = torch.where(first, s_new, -1.0e30)
        sink = torch.where(first, sink_raw, -1.0e30)
        m_i = torch.maximum(sink, s_new)
        e_new = torch.where(first, torch.exp(s_new - m_i), 0.0)
        l_i = torch.where(first, torch.exp(sink - m_i), 0.0) + e_new
        acc = vb.to(torch.float32)[:, None] * e_new[None, :]  # o^T [d, head]
        q_t = qb.T  # [64, 8]
        tok = t_begin * 64 + d
        valid = (tok >= lo) & (tok < pos) & has0
        s = torch.where(
            valid[:, None], hl.dot(kt0, q_t) * 0.125, -1.0e30
        )  # [tok, head]
        m_new = torch.maximum(m_i, torch.amax(s, dim=0))
        p = torch.where(valid[:, None], torch.exp(s - m_new[None, :]), 0.0)
        alpha = torch.exp(m_i - m_new)
        l_i = l_i * alpha + torch.sum(p, dim=0)
        acc = acc * alpha[None, :] + hl.dot(vt0.T, p.to(torch.bfloat16))
        m_i = m_new
        for tile_t in hl.tile(t_begin + 1, t_end, block_size=1):
            page = hl.load(page_table, [0, tile_t.begin])
            tok = tile_t.begin * 64 + d
            idx = (page.to(torch.int64) * 64 + d)[:, None] * 512 + g * 64 + d[None, :]
            kt = hl.load(k_cache, [idx])  # [tok, d]
            vt = hl.load(v_cache, [idx])  # [tok, d]
            s = hl.dot(kt, q_t) * 0.125  # [tok, head]
            valid = (tok >= lo) & (tok < pos)
            s = torch.where(valid[:, None], s, -1.0e30)
            m_new = torch.maximum(m_i, torch.amax(s, dim=0))
            p = torch.where(valid[:, None], torch.exp(s - m_new[None, :]), 0.0)
            alpha = torch.exp(m_i - m_new)
            l_i = l_i * alpha + torch.sum(p, dim=0)
            acc = acc * alpha[None, :] + hl.dot(vt.T, p.to(torch.bfloat16))
            m_i = m_new
        o = acc / torch.clamp(l_i, min=1.0e-30)[None, :]  # [d, head]
        hl.store(m_part, [g * 8 + hj, tile_c.begin], m_i)
        hl.store(l_part, [g * 8 + hj, tile_c.begin], l_i)
        # attn_b[s, n, k]: s = 2g + j // 4, n = (j % 4) * 16 + c, k = (j % 4) * 64 + d
        jj = hj[:, None]
        hl.store(
            attn_b,
            [g * 2 + jj // 4, (jj % 4) * 16 + tile_c.begin, (jj % 4) * 64 + d[None, :]],
            o.T.to(torch.bfloat16),
        )
        loc = hl.load(out_loc, [0])
        kv_mask = (d + tile_c.begin * 64) < 64
        hl.store(k_cache, [loc * 512 + g * 64 + d], kb.reshape(64), extra_mask=kv_mask)
        hl.store(v_cache, [loc * 512 + g * 64 + d], vb, extra_mask=kv_mask)
    # 3. O projection, 16 K splits of 4 heads; N = (head, chunk) columns, merged
    # in the epilogue with the chunks' softmax weights
    for tile_m, tile_n, tile_s in hl.tile([2880, 64, 16], block_size=[128, 64, 1]):
        # the merge weights first: their loads overlap the MMAs
        hh = hl.arange(4)
        cc = hl.arange(16)
        mc = hl.load(m_part, [tile_s.begin * 4 + hh[:, None], cc[None, :]])  # [4, 16]
        lc = hl.load(l_part, [tile_s.begin * 4 + hh[:, None], cc[None, :]])
        mx = torch.amax(mc, dim=1)
        wc = lc * torch.exp(mc - mx[:, None])
        wc = wc / torch.sum(wc, dim=1)[:, None]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(256, block_size=64):
            acc = hl.dot(
                o_w[tile_m, tile_s.begin, tile_k],
                attn_b[tile_s.begin, tile_n, tile_k].T,
                acc=acc,
            )
        op_part[tile_s.begin, tile_m] = torch.sum(acc * wc.reshape(64)[None, :], dim=1)
    # 4. post-norm statistics, router partials, 4-term MXFP4 activation
    for tile_c in hl.tile(2880, block_size=64):
        # every load first (their latencies overlap), then the math; the
        # router partials and the activation (what the experts wait for) go
        # out before this layer's outputs for the next one
        xr = x_res[tile_c]
        op = op_part[:, tile_c]
        ob = o_b[tile_c].to(torch.float32)
        ln2 = ln2_w[tile_c].to(torch.float32)
        rw = router_w[:, tile_c].to(torch.float32)
        x = xr + torch.sum(op, dim=0) + ob
        yu = x * ln2
        hl.atomic_add(router_acc, [hl.arange(128)], torch.sum(rw * yu[None, :], dim=1))
        ssq_m[tile_c.id] = torch.sum(x * x, dim=0)
        r = yu.reshape(2, 32)
        e0 = e2m1_exponent(r)
        for t in hl.static_range(4):
            e = e0 - 2.0 * t
            code, r = e2m1_term(r, e)
            y_q[t, tile_c.begin // 2 + hl.arange(32)] = pack_nibbles(code, 2).reshape(
                32
            )
            y_sp[tile_c.begin // 128, t, (tile_c.begin // 32) % 4 + hl.arange(2)] = (
                e + 127.0
            ).to(torch.uint8)
        h_out[tile_c] = x
        bq_out[4, tile_c] = x.to(torch.bfloat16)
        zi = tile_c.id * 128 + hl.arange(128)
        hl.store(
            qkv_acc, [zi], hl.zeros([128], dtype=torch.float32), extra_mask=zi < 5120
        )
    # 5. expert gate/up + SwiGLU, 4-term MXFP4 intermediate
    for tile_m, tile_n, tile_s in hl.tile([5888, 16, 4], block_size=[128, 16, 1]):
        expert, _, inv_rms = _route(router_acc, ssq_m, router_b, tile_s.begin)
        rows = expert * 5888 + tile_m.begin
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(1536, block_size=128):
            acc = hl.dot_scaled(
                w13[rows + hl.arange(128), tile_k],
                scale_tiles(
                    w13_sp[
                        expert * 46 + tile_m.begin // 128,
                        tile_k.begin // 64 + hl.arange(2),
                        :,
                        :,
                    ]
                ),
                "e2m1",
                y_q[tile_n, tile_k].T,
                scale_tiles(y_sp[tile_k.begin // 64 + hl.arange(2), :, :]),
                "e2m1",
                acc=acc,
            )
        pre = torch.sum(acc, dim=1) * inv_rms + w13_b[expert, tile_m]
        up, gate = hl.split(pre.reshape(4, 2, 2, 8).permute(0, 1, 3, 2))
        act = clamped_swiglu(gate, up)  # [4, 2, 8]
        r = act.permute(0, 2, 1).reshape(2, 32)
        term = hl.arange(16)
        q16 = hl.zeros([16, 32], dtype=torch.int32)
        s16 = hl.full([16, 2], 127.0, dtype=torch.float32)
        e0 = e2m1_exponent(r)
        for t in hl.static_range(4):
            e = e0 - 2.0 * t
            code, r = e2m1_term(r, e)
            packed = pack_nibbles(code, 2).reshape(32).to(torch.int32)
            q16 = torch.where(term[:, None] == t, packed[None, :], q16)
            s16 = torch.where(term[:, None] == t, (e + 127.0)[None, :], s16)
        a_q[tile_s.begin, tile_n, tile_m.begin // 4 + hl.arange(32)] = q16.to(
            torch.uint8
        )
        # packed (tcgen05.cp) layout: scale column c of row n at [c // 4, n, c % 4]
        a_s[
            tile_s.begin,
            tile_m.begin // 256,
            tile_n,
            (tile_m.begin // 64) % 4 + hl.arange(2),
        ] = s16.to(torch.uint8)
    # 6. expert down projection, weighted by the routing probability
    for tile_m, tile_n, tile_s in hl.tile([2944, 16, 4], block_size=[128, 16, 1]):
        expert, weight, _ = _route(router_acc, ssq_m, router_b, tile_s.begin)
        rows = expert * 3072 + tile_m.begin
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        # 12 K steps of 128 B; the last one's upper half reads past the 1472
        # columns (TMA zero fill; the intermediate's padding there is zero)
        for tile_k in hl.tile(1536, block_size=128):
            acc = hl.dot_scaled(
                w2[rows + hl.arange(128), tile_k],
                scale_tiles(
                    w2_sp[
                        expert * 24 + tile_m.begin // 128,
                        tile_k.begin // 64 + hl.arange(2),
                        :,
                        :,
                    ]
                ),
                "e2m1",
                a_q[tile_s.begin, tile_n, tile_k].T,
                scale_tiles(a_s[tile_s.begin, tile_k.begin // 64 + hl.arange(2), :, :]),
                "e2m1",
                acc=acc,
            )
        o = (torch.sum(acc, dim=1) + w2_b[expert, tile_m]) * weight
        p = tile_m.index
        logical = (p // 32) * 32 + (p % 8) * 4 + (p % 32) // 8
        hl.store(part_out, [tile_s.begin, logical], o, extra_mask=logical < 2880)
        hl.store(
            bq_out,
            [tile_s.begin, logical],
            o.to(torch.bfloat16),
            extra_mask=logical < 2880,
        )
    return h_out, part_out


# (position, window): a sliding-window layer, full layers of one and of several
# KV tiles per attention chunk, a context shorter than one KV tile
CHECK_CASES = ((1000, SLIDING_WINDOW), (1000, -1), (4000, -1), (37, -1))
BENCHMARK_CASES = ((1000, SLIDING_WINDOW), (1000, -1), (4000, -1))


def make_scratch(device: torch.device) -> tuple[torch.Tensor, ...]:
    """The kernel's scratch operands, in argument order.  Zero (or 127) on
    first use; the kernel leaves them ready for its next launch."""
    f32, bf16, u8 = torch.float32, torch.bfloat16, torch.uint8

    def z(*shape, dtype=f32):
        return torch.zeros(*shape, dtype=dtype, device=device)

    return (
        z(HIDDEN),  # x_res
        z(64),  # ssq_a
        z(5120),  # qkv_acc
        z(64, 16),  # m_part
        z(64, 16),  # l_part
        z(16, 64, 256, dtype=bf16),  # attn_b
        z(16, HIDDEN),  # op_part
        z(16, 1536, dtype=u8),  # y_q
        # y_sp, a_s: each 16-byte scale row on its own 128-byte line (every
        # expert task reads the same rows each K step)
        torch.full((24, 32, 128), 127, dtype=u8, device=device)[:, :, :16],
        z(4, 16, 1536, dtype=u8),  # a_q
        torch.full((4, 24, 32, 128), 127, dtype=u8, device=device)[:, :, :, :16],
        z(64),  # ssq_m
        z(128),  # router_acc
    )


def kernel_args(served, ctx, cos_sin, x_h, part_in, scratch, outputs) -> tuple:
    """Operands for one layer: ``x_h`` [2880] and ``part_in`` [4, 2880] are the
    previous layer's FP32 outputs; ``outputs`` = (bq_out, h_out, part_out)."""
    bq_in = torch.zeros(16, HIDDEN, dtype=torch.bfloat16, device=x_h.device)
    bq_in[:4] = part_in.to(torch.bfloat16)
    bq_in[4] = x_h.to(torch.bfloat16)
    return (
        x_h, part_in, served["qkv_w"], bq_in.view(16, 15, 192), served["qkv_b"],
        cos_sin, ctx.positions, ctx.out_loc, ctx.page_table, ctx.window,
        ctx.k_cache.view(-1), ctx.v_cache.view(-1), served["sinks"],
        served["o_w"], served["o_b"], served["ln2"], served["router_w"], served["router_b"],
        served["w13"], served["w13_sp"], served["w13_b"],
        served["w2"], served["w2_sp"], served["w2_b"],
        *scratch, *outputs,
    )  # fmt: skip


def _make_case(served, cos_sin, pos: int, win: int, device, scratch=None):
    gen = torch.Generator(device=device).manual_seed(pos)
    ctx = make_context(pos, win, tokens=1, seed=pos, device=device)
    x_h = torch.randn(HIDDEN, generator=gen, device=device) * 2.0
    part_in = torch.randn(4, HIDDEN, generator=gen, device=device) * 0.5
    outputs = (
        torch.zeros(16, HIDDEN, dtype=torch.bfloat16, device=device),
        torch.zeros(HIDDEN, device=device),
        torch.zeros(4, HIDDEN, device=device),
    )
    if scratch is None:
        scratch = make_scratch(device)
    args = kernel_args(served, ctx, cos_sin, x_h, part_in, scratch, outputs)
    return ctx, x_h, part_in, outputs, args


def correctness_check(verbose: bool = False) -> None:
    """One layer against the FP32 reference (attention, the K/V write, the
    MoE, and the next layer's operand rows).  The cases share one scratch, so
    each launch must leave it ready for the next (``qkv_acc`` back to zero)."""
    device = torch.device("cuda")
    raw, served = make_layer(seed=0, device=device)
    cos_sin = rope_cos_sin(device)
    scratch = make_scratch(device)
    for pos, win in CHECK_CASES:
        ctx, x_h, part_in, (bq_out, h_out, part_out), args = _make_case(
            served, cos_sin, pos, win, device, scratch
        )
        gpt_oss_decode_layer(*args)
        if scratch[2].any():  # qkv_acc
            raise AssertionError("qkv_acc was not re-zeroed for the next launch")
        x = (x_h + part_in.sum(0))[None]
        h_ref, k_ref, v_ref = reference_attention(raw, ctx, cos_sin, x)
        moe_ref, ids = reference_moe(raw, h_out[None])
        errors = {
            "attention": check("attention", h_out - x[0], h_ref[0] - x[0], 2e-2),
            "k write": check("k write", ctx.k_cache[ctx.out_loc], k_ref, 2e-2),
            "v write": check("v write", ctx.v_cache[ctx.out_loc], v_ref, 2e-2),
            "moe": check("moe", part_out.sum(0), moe_ref[0], 1e-2),
            "bq experts": check("bq experts", bq_out[:4], part_out, 1e-2),
            "bq residual": check("bq residual", bq_out[4], h_out, 1e-2),
        }
        if verbose:
            cells = "  ".join(f"{k} {v:.2e}" for k, v in errors.items())
            print(f"pos {pos:5d} window {win:4d} experts {ids[0].tolist()}: {cells}")


def use_cudagraph() -> bool:
    return True


@torch.inference_mode()
def main(verbose: bool = True) -> dict[str, Any]:
    """Check against the FP32 reference, then report cold-L2 CUDA-graph
    latencies of one layer."""
    from pretuned_kernels._bench import bench_pre_captured_cudagraph
    from pretuned_kernels._bench import capture_cuda_graph

    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("gpt_oss_decode_layer is pretuned for SM100")
    correctness_check(verbose=verbose)
    device = torch.device("cuda")
    _, served = make_layer(seed=0, device=device)
    cos_sin = rope_cos_sin(device)
    latency = {}
    for pos, win in BENCHMARK_CASES:
        args = _make_case(served, cos_sin, pos, win, device)[-1]
        graph, _ = capture_cuda_graph(lambda args=args: gpt_oss_decode_layer(*args))
        us = bench_pre_captured_cudagraph(graph.replay) * 1000
        latency[f"pos{pos}_window{win}"] = us
        if verbose:
            print(f"pos {pos:5d} window {win:4d}: {us:.2f} us")
    return {"latency_us": latency}


if __name__ == "__main__":
    main()
