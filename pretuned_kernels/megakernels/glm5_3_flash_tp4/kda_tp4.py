# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""GLM-5.3 Flash KDA speculative verification as one Gluon megakernel: a KDA
layer's TP4 shard (16 heads of 128) for one request of T <= 8 draft tokens,
from the layer's mHC pre output to the FFN's input (the attention -> FFN mHC
boundary folded into its tail), writing the per-token recurrent states
(``intermediate_ssm``) and conv windows that SGLang's commit reads
(``_setup.kda_reference`` has the PyTorch reference).

Roots, in launch order (one persistent CTA per SM, TMA ring):
  1. f_a projection in 8 K splits of 512 (8 tasks); each task also computes its
     split of the beta projection by a register dot from a plain-load alias of
     the hidden states (off the ring); partials into ``pfa`` and ``pbeta``.
  2. g_a projection (the low-rank output gate's first factor), 8 K splits.
  3. q/k/v projection: [T, 4096] x [6144, 4096]^T per head (its q, k, v rows)
     in 4 K splits of 1024 (12 tasks a head, 192 tasks, head-major), each
     split's partial into ``pq`` at the task's head.
  4. recurrence per (head, 16 value rows) (128 tasks): the decay (the F_B
     projection of f_a), the causal conv of q, k (all 128 channels) and v (the
     task's 16 rows), the q/k L2 norm and the delta rule over the T tokens in
     order; each token's conv window and recurrent state.
  5. gated RMSNorm per head (16 tasks) with the gate G_B(g_a), the BF16 operand
     of
  6. the output projection [T, 2048] x [4096, 2048]^T in 2 K splits (128 tasks),
     BF16 partials into this rank's symmetric buffer; then the boundary's comb
     (raw: the previous boundary's logits, their Sinkhorn here);
  7-9. the mHC boundary (``_mhc``): the ranks' partials summed split-major in
     rank order (FP32, BF16), hc_post, the FFN's hc_pre and
     post_attention_layernorm -> the FFN input, R', the FFN's post / comb.

Split-K partials are summed by their readers in split order and the all-reduce
in rank order (no atomics): the kernel is deterministic run to run. Lanes past
the window (T < 8) are masked off every partial load and cache store; their
register lanes never mix into a window token's values (every cross-lane
reduction is over channels, the delta rule runs the T tokens only).

A task waits for a producer root at its first read of that root's output, in
source order: the recurrence reads f_a's partials (and computes the decay)
before the q/k/v projection's, and the norm reads the gate before the
recurrence output. A recurrence task waits for its head's 12 projection tasks
only (a readiness counter per head: ``pq`` keeps the head an axis of its own),
so the first heads' recurrences run beside the projection's second wave.

``correctness_check()`` compares every rank with the torch reference (``_setup``) on
synthetic weights; ``main()`` runs it, then times the kernel with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_flash_tp4/kda_tp4.py [--check]
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_buffer
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels._ring_config import ring_config_fields
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _mhc
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _setup
from pretuned_kernels.megakernels.kda_mtp._common import conv_tap
from pretuned_kernels.megakernels.kda_mtp._common import conv_weights
from pretuned_kernels.megakernels.kda_mtp._common import conv_window
from pretuned_kernels.megakernels.kda_mtp._common import delta_step
from pretuned_kernels.megakernels.kda_mtp._common import silu_conv
from pretuned_kernels.megakernels.kda_mtp._common import token_operands
import torch
import torch.distributed as dist

import helion
import helion.language as hl

LANES = 8  # window token lanes (T <= 8)
V_TILE = 16  # value rows per recurrence task
QKV_SPLITS = 4
GATE_SPLITS = 8


def _psum(p, live):
    """The q/k/v projection's output [8, C] from its K-split partials ``p``
    [splits, 8, C]: summed in order, rounded to bf16 (the projection's dtype);
    0 past the window."""
    p = torch.where(live[None, :, None], p, 0.0)
    return torch.sum(p, dim=0).to(torch.bfloat16).to(torch.float32)


def _win3(iconv, row, okt, col, x0, x1, x2, cmask):
    """Token t's conv window (inputs at t-2, t-1, t) -> iconv[row, t], channels
    where ``cmask``, tokens where ``okt`` (in the window, a real request)."""
    tt = hl.arange(8)[:, None]
    zt = tt * 0
    m = okt & cmask[None, :]
    lane = torch.where(okt, tt, 0)
    hl.store(
        iconv, [row + zt, lane, zt, col[None, :]], x2.to(torch.bfloat16), extra_mask=m
    )
    hl.store(
        iconv,
        [row + zt, lane, zt + 1, col[None, :]],
        x1.to(torch.bfloat16),
        extra_mask=m,
    )
    hl.store(
        iconv,
        [row + zt, lane, zt + 2, col[None, :]],
        x0.to(torch.bfloat16),
        extra_mask=m,
    )


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def glmf_kda_tp4(
    x4,  # [T, 4, 1024] bf16: the layer's mHC pre output (4 K splits)
    w_qkv,  # [48, 128, 4, 1024] bf16: fused_qkvbfg_a_proj rows q k v (head tiles of 128)
    x8,  # [T, 8, 512] bf16: the same hidden states (8 K splits)
    w_fa,  # [128, 8, 512] bf16: rows f_a
    w_ga,  # [128, 8, 512] bf16: rows g_a
    xp,  # [T, 8, 512] bf16: the hidden states, plain loads (off the ring)
    w_beta,  # [16, 8, 512] bf16: rows b
    f_b,  # [2048, 128] bf16: fused_fg_b[0]
    g_b,  # [2048, 128] bf16: fused_fg_b[1]
    conv_w,  # [3, 4, 2048] fp32: conv weights taps-major (prepared once)
    a_log,  # [16] fp32
    dt_bias,  # [2048] fp32
    conv_state,  # [P, 3, 6144] bf16 (read only)
    rec_state,  # [P, 16, 128, 128] fp32 (read only)
    state_idx,  # [1] int32: the request's mamba slot (-1: padding)
    issm,  # [R, T, 16, 128, 128] fp32: per-token recurrent states (written)
    iconv,  # [R, T, 3, 6144] bf16: per-token conv windows (written)
    iconv_idx,  # [1] int32: the request's intermediate row
    norm_w,  # [128] fp32: the gated norm's weight
    w_out,  # [4096, 2, 1024] bf16: o_proj (2 K splits)
    # scratch (fp32 but ob)
    pq,  # [3, 4, 8, 16, 128] q/k/v projection's split-K partials (q/k/v, split, lane, head, row)
    pfa,  # [8, 8, 128] f_a projection's split-K partials
    pga,  # [8, 8, 128] g_a projection's split-K partials
    pbeta,  # [8, 8, 16] beta projection's split-K partials
    stg,  # [128, 3, 8, 128] per recurrence task q, k (normalized), v
    sdec,  # [128, 8, 128] per recurrence task decay
    sbeta,  # [128, 8] per recurrence task beta
    opre,  # [8, 2048] recurrence output before the gated norm
    ob,  # [T, 2, 1024] bf16: the gated norm's output (o_proj's 2 K splits)
    symmetric_partial,  # [2, T, 4096] bf16 symmetric memory: this rank's o_proj K-split partials
    # the mHC boundary after the attention (``_mhc``): its residual streams and
    # mixes in, the FFN's hc fn / base / scale and norm, the FFN input and next
    # mixes out
    r,  # [T, 4, 4096] bf16
    post,  # [T, 4] fp32
    comb,  # [T, 4, 4] fp32
    fn_pad,  # [32, 4, 4096] bf16
    hc_scale,  # [3] fp32
    base_pre,  # [4] fp32
    base_post,  # [4] fp32
    base_comb,  # [4, 4] fp32
    ffn_norm_w,  # [4096] bf16: post_attention_layernorm
    mix_part,  # [S, 8, 32] fp32 scratch (the Sinkhorn's mixes)
    mix_t,  # [8, 32, S] fp32 scratch: the same, token-major (finalize's warp-local sums)
    gram_part,  # [8, 4, 4, S] fp32 scratch, token-major
    sq_part,  # [8, S] fp32 scratch, token-major
    li,  # [T, 4096] bf16 out: the FFN's input
    r_new,  # [T, 4, 4096] bf16 out
    post_out,  # [T, 4] fp32 out
    comb_out,  # [T, 4, 4] fp32 out (``comb_out_raw``: logits)
    comb_s,  # [T, 4, 4] fp32 scratch: the final comb (``comb_in_raw``: the head root's Sinkhorn)
    group_name: hl.ProcessGroupName,
    comb_in_raw: hl.constexpr,  # comb holds the previous boundary's logits: Sinkhorn them here
    comb_out_raw: hl.constexpr,  # comb_out gets this boundary's logits (the next MK takes them raw)
) -> None:
    hl.specialize(x4.size())
    hl.specialize(x4.stride())
    hl.specialize(w_qkv.size())
    hl.specialize(w_qkv.stride())
    hl.specialize(x8.size())
    hl.specialize(x8.stride())
    hl.specialize(w_fa.size())
    hl.specialize(w_fa.stride())
    hl.specialize(w_ga.size())
    hl.specialize(w_ga.stride())
    hl.specialize(xp.size())
    hl.specialize(xp.stride())
    hl.specialize(w_beta.size())
    hl.specialize(w_beta.stride())
    hl.specialize(f_b.size())
    hl.specialize(f_b.stride())
    hl.specialize(g_b.size())
    hl.specialize(g_b.stride())
    hl.specialize(conv_w.size())
    hl.specialize(conv_w.stride())
    hl.specialize(a_log.size())
    hl.specialize(a_log.stride())
    hl.specialize(dt_bias.size())
    hl.specialize(dt_bias.stride())
    hl.specialize(conv_state.size())
    hl.specialize(conv_state.stride())
    hl.specialize(rec_state.size())
    hl.specialize(rec_state.stride())
    hl.specialize(state_idx.size())
    hl.specialize(state_idx.stride())
    hl.specialize(issm.size())
    hl.specialize(issm.stride())
    hl.specialize(iconv.size())
    hl.specialize(iconv.stride())
    hl.specialize(iconv_idx.size())
    hl.specialize(iconv_idx.stride())
    hl.specialize(norm_w.size())
    hl.specialize(norm_w.stride())
    hl.specialize(w_out.size())
    hl.specialize(w_out.stride())
    hl.specialize(pq.size())
    hl.specialize(pq.stride())
    hl.specialize(pfa.size())
    hl.specialize(pfa.stride())
    hl.specialize(pga.size())
    hl.specialize(pga.stride())
    hl.specialize(pbeta.size())
    hl.specialize(pbeta.stride())
    hl.specialize(stg.size())
    hl.specialize(stg.stride())
    hl.specialize(sdec.size())
    hl.specialize(sdec.stride())
    hl.specialize(sbeta.size())
    hl.specialize(sbeta.stride())
    hl.specialize(opre.size())
    hl.specialize(opre.stride())
    hl.specialize(ob.size())
    hl.specialize(ob.stride())
    hl.specialize(symmetric_partial.size())
    hl.specialize(symmetric_partial.stride())
    hl.specialize(r.size())
    hl.specialize(r.stride())
    hl.specialize(post.size())
    hl.specialize(post.stride())
    hl.specialize(comb.size())
    hl.specialize(comb.stride())
    hl.specialize(fn_pad.size())
    hl.specialize(fn_pad.stride())
    hl.specialize(hc_scale.size())
    hl.specialize(hc_scale.stride())
    hl.specialize(base_pre.size())
    hl.specialize(base_pre.stride())
    hl.specialize(base_post.size())
    hl.specialize(base_post.stride())
    hl.specialize(base_comb.size())
    hl.specialize(base_comb.stride())
    hl.specialize(ffn_norm_w.size())
    hl.specialize(ffn_norm_w.stride())
    hl.specialize(mix_part.size())
    hl.specialize(mix_part.stride())
    hl.specialize(mix_t.size())
    hl.specialize(mix_t.stride())
    hl.specialize(gram_part.size())
    hl.specialize(gram_part.stride())
    hl.specialize(sq_part.size())
    hl.specialize(sq_part.stride())
    hl.specialize(li.size())
    hl.specialize(li.stride())
    hl.specialize(r_new.size())
    hl.specialize(r_new.stride())
    hl.specialize(post_out.size())
    hl.specialize(post_out.stride())
    hl.specialize(comb_out.size())
    hl.specialize(comb_out.stride())
    hl.specialize(comb_s.size())
    hl.specialize(comb_s.stride())
    tokens = x4.size(0)
    heads = a_log.size(0)
    head = f_b.size(1)
    seg = heads * head
    hidden = w_out.size(0)
    qkv_k = w_qkv.size(3)
    gate_k = w_fa.size(2)
    qkv_splits = w_qkv.size(2)
    out_k = w_out.size(2)  # (host code: a size read in a device loop stays symbolic)
    assert (
        w_qkv.size(0) == 3 * heads
        and w_qkv.size(1) == head
        and w_fa.size(1) == 8
        and pq.size(1) == qkv_splits
    )
    assert tokens <= 8 and head == 128 and heads * 8 == stg.size(0)
    assert li.size() == (tokens, hidden) and symmetric_partial.size() == (
        2,
        tokens,
        hidden,
    )
    assert mix_part.size(0) * 128 == hidden
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    scale = head**-0.5
    eps = 1e-5
    lower_bound = -5.0
    remote_partials = torch.ops.symm_mem.get_remote_tensors(
        symmetric_partial, group_name
    )

    hl.pdl_wait()
    # The code toward L2 first: in a decoder layer the MoE's weight stream
    # evicted it.
    hl.prefetch_code()
    # 1. f_a (128 rows) and beta (16 rows), 8 K splits of 512 (8 tasks)
    for tile_s, tile_m, tile_n in hl.tile([8, 128, tokens], block_size=[1, 128, 16]):
        t8 = hl.arange(8)
        j16 = hl.arange(16)
        kk = hl.arange(gate_k)
        hx = hl.load(
            xp,
            [
                torch.where(t8 < tokens, t8, 0)[:, None],
                tile_s.begin + t8[:, None] * 0,
                kk[None, :],
            ],
            extra_mask=(t8 < tokens)[:, None],
        )
        wb = hl.load(
            w_beta, [j16[:, None], tile_s.begin + j16[:, None] * 0, kk[None, :]]
        )
        # beta: this split's [8, 512] x [16, 512]^T in registers (off the ring)
        pb = hl.dot(hx, wb.T)  # [8, 16] fp32
        hl.store(pbeta, [tile_s.begin + t8[:, None] * 0, t8[:, None], j16[None, :]], pb)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(gate_k, block_size=64):
            acc = hl.dot(
                w_fa[tile_m, tile_s.begin, tile_k],
                x8[tile_n, tile_s.begin, tile_k].T,
                acc=acc,
            )
        pfa[tile_s.begin, tile_n, tile_m] = acc.T
    # 2. g_a (128 rows), 8 K splits of 512 (8 tasks)
    for tile_s, tile_m, tile_n in hl.tile([8, 128, tokens], block_size=[1, 128, 16]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(gate_k, block_size=64):
            acc = hl.dot(
                w_ga[tile_m, tile_s.begin, tile_k],
                x8[tile_n, tile_s.begin, tile_k].T,
                acc=acc,
            )
        pga[tile_s.begin, tile_n, tile_m] = acc.T
    # 3. q/k/v projection (6144 rows): per head, its q, k and v rows (128 each)
    # in K splits (4 of 1024: 12 tasks a head, 192 tasks), head-major so the
    # first heads finish in the first wave. The partials keep the head an axis
    # of its own (written at the task's head): a recurrence task waits for its
    # head's 12 tasks, not the whole projection.
    for tile_ks, tile_n, tile_h in hl.tile(
        [3 * qkv_splits, tokens, heads], block_size=[1, 16, 1]
    ):
        kind = tile_ks.begin // qkv_splits
        sp = tile_ks.begin % qkv_splits
        acc = hl.zeros([head, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(qkv_k, block_size=64):
            acc = hl.dot(
                w_qkv[kind * heads + tile_h.begin, :, sp, tile_k],
                x4[tile_n, sp, tile_k].T,
                acc=acc,
            )
        pq[kind, sp, tile_n, tile_h.begin, :] = acc.T

    # 4. per (head, 16 value rows): the F_B decay and beta, the conv (q, k all
    # channels; v this task's 16), q/k L2 norm, then the delta rule over the
    # window's tokens in order. The operands go through task-private scratch
    # (the compiler places the CTA barriers between) so the chain runs in the
    # state's layout: key vectors [1, 128], value columns [16, 1].
    for tile_h, tile_v in hl.tile([heads, head], block_size=[1, 16]):
        h = tile_h.begin
        v0 = tile_v.begin
        tt = hl.arange(8)
        d = hl.arange(128)
        vr = v0 + hl.arange(16)
        zt = tt[:, None] * 0
        z2 = vr[:, None] * 0
        live = tt < tokens
        slot = hl.load(state_idx, [0])
        row = hl.load(iconv_idx, [0])
        ok = slot >= 0
        sl = torch.where(ok, slot, 0)
        # int64 through the loaded slot: the scratch addresses need no widening
        task = h * 8 + tile_v.id + slot * 0
        mine = (d >= v0) & (d < v0 + 16)
        okt = ((zt + ok) != 0) & live[:, None]
        # every global load first; the f_a and beta sums are this task's first
        # reads of a projection: it waits for f_a here, for q/k/v at _psum
        s8 = hl.arange(8)[:, None, None]
        fa = torch.sum(
            hl.load(
                pfa,
                [s8, tt[None, :, None], d[None, None, :]],
                extra_mask=live[None, :, None],
            ),
            dim=0,
        )
        braw = torch.sum(
            hl.load(
                pbeta,
                [s8, tt[None, :, None], h + zt[None, :, :]],
                extra_mask=live[None, :, None],
            ),
            dim=0,
        )  # [8, 1]
        fb = hl.load(f_b, [h * 128 + d[:, None], d[None, :]])  # [d, j]
        dtb = hl.load(dt_bias, [h * 128 + zt + d[None, :]])
        al = hl.load(a_log, [h])
        wq0, wq1, wq2, wq3 = conv_weights(conv_w, 0, h * 128 + d)
        wk0, wk1, wk2, wk3 = conv_weights(conv_w, 1, h * 128 + d)
        wv0, wv1, wv2, wv3 = conv_weights(conv_w, 2, h * 128 + vr)
        st = hl.load(
            rec_state,
            [sl + z2, h + z2, vr[:, None], d[None, :]],
            extra_mask=(z2 + ok) != 0,
        )  # [16 v, 128 k]
        beta = torch.sigmoid(braw.to(torch.bfloat16).to(torch.float32))  # [8, 1]
        gate = hl.dot(fa.to(torch.bfloat16), fb.T)  # [8, 128] fp32
        graw = gate.to(torch.bfloat16).to(torch.float32) + dtb
        logd = lower_bound * torch.sigmoid(torch.exp(al) * graw)  # [8, 128]
        hl.store(sdec, [task + zt, tt[:, None], d[None, :]], torch.exp(logd))
        hl.store(sbeta, [task + zt, tt[:, None]], beta)
        cq = h * 128 + d
        ck = seg + cq
        cv = 2 * seg + h * 128 + vr
        # the committed windows (inputs only), then the projection outputs once;
        # the shifted taps read them back from this task's stg slots, which the
        # normalized q, k, v overwrite below
        hq1 = conv_window(conv_state, sl, ok, cq, 1)
        hq2 = conv_window(conv_state, sl, ok, cq, 2)
        hq3 = conv_window(conv_state, sl, ok, cq, 3)
        hk1 = conv_window(conv_state, sl, ok, ck, 1)
        hk2 = conv_window(conv_state, sl, ok, ck, 2)
        hk3 = conv_window(conv_state, sl, ok, ck, 3)
        hv1 = conv_window(conv_state, sl, ok, cv, 1)
        hv2 = conv_window(conv_state, sl, ok, cv, 2)
        hv3 = conv_window(conv_state, sl, ok, cv, 3)
        q_0 = _psum(pq[0, :, :, h, :], live)
        k_0 = _psum(pq[1, :, :, h, :], live)
        v_0 = _psum(pq[2, :, :, h, tile_v], live)
        hl.store(stg, [task + zt, zt, tt[:, None], d[None, :]], q_0)
        hl.store(stg, [task + zt, zt + 1, tt[:, None], d[None, :]], k_0)
        hl.store(stg, [task + zt, zt + 2, tt[:, None], (vr - v0)[None, :]], v_0)
        q_1 = conv_tap(stg, task, 0, d, hq1, 1)
        q_2 = conv_tap(stg, task, 0, d, hq2, 2)
        q_3 = conv_tap(stg, task, 0, d, hq3, 3)
        k_1 = conv_tap(stg, task, 1, d, hk1, 1)
        k_2 = conv_tap(stg, task, 1, d, hk2, 2)
        k_3 = conv_tap(stg, task, 1, d, hk3, 3)
        v_1 = conv_tap(stg, task, 2, vr - v0, hv1, 1)
        v_2 = conv_tap(stg, task, 2, vr - v0, hv2, 2)
        v_3 = conv_tap(stg, task, 2, vr - v0, hv3, 3)
        _win3(iconv, row, okt, cq, q_0, q_1, q_2, mine)
        _win3(iconv, row, okt, ck, k_0, k_1, k_2, mine)
        _win3(iconv, row, okt, cv, v_0, v_1, v_2, vr >= 0)
        yq = silu_conv(q_3, q_2, q_1, q_0, wq0, wq1, wq2, wq3)
        yk = silu_conv(k_3, k_2, k_1, k_0, wk0, wk1, wk2, wk3)
        yv = silu_conv(v_3, v_2, v_1, v_0, wv0, wv1, wv2, wv3)  # [8, 16]
        hl.store(
            stg,
            [task + zt, zt, tt[:, None], d[None, :]],
            yq * (torch.rsqrt(torch.sum(yq * yq, dim=1) + 1e-6) * scale)[:, None],
        )
        hl.store(
            stg,
            [task + zt, zt + 1, tt[:, None], d[None, :]],
            yk * torch.rsqrt(torch.sum(yk * yk, dim=1) + 1e-6)[:, None],
        )
        hl.store(stg, [task + zt, zt + 2, tt[:, None], (vr - v0)[None, :]], yv)
        okv = (z2 + ok) != 0
        for t in hl.static_range(tokens):
            q_t, k_t, e_t, v_t, b_t = token_operands(stg, sdec, sbeta, task, d, t)
            st, o_t = delta_step(st, q_t, k_t, e_t, v_t, b_t)
            hl.store(
                issm,
                [row + z2, t + z2, h + z2, vr[:, None], d[None, :]],
                st,
                extra_mask=okv,
            )
            hl.store(opre, [t + z2, h * 128 + vr[:, None]], o_t)
    # 5. per head: gated RMSNorm (gate = sigmoid(G_B(g_a))) -> bf16 operand of
    # the output projection
    for tile_h in hl.tile(heads, block_size=1):
        h = tile_h.begin
        tt = hl.arange(8)
        d = hl.arange(128)
        live = tt < tokens
        lane = torch.where(live, tt, 0)
        slot = hl.load(state_idx, [0])
        s8 = hl.arange(8)[:, None, None]
        # the gate first: its projection finishes long before the recurrence
        ga = torch.sum(
            hl.load(
                pga,
                [s8, tt[None, :, None], d[None, None, :]],
                extra_mask=live[None, :, None],
            ),
            dim=0,
        )
        gb = hl.load(g_b, [h * 128 + d[:, None], d[None, :]])  # [d, j]
        nw = hl.load(norm_w, [tt[:, None] * 0 + d[None, :]])  # [8, 128]
        g = torch.sigmoid(
            hl.dot(ga.to(torch.bfloat16), gb.T).to(torch.bfloat16).to(torch.float32)
        )
        # SGLang's verify rounds the recurrence output to BF16 before the norm
        o = (
            hl.load(opre, [lane[:, None], h * 128 + d[None, :]])
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        rinv = torch.rsqrt(torch.sum(o * o, dim=1) * (1.0 / 128) + eps)
        on = o * rinv[:, None] * nw * g
        hl.store(
            ob,
            [lane[:, None], tt[:, None] * 0 + h // 8, (h % 8) * 128 + d[None, :]],
            torch.where((tt[:, None] * 0 + slot) >= 0, on, 0.0).to(torch.bfloat16),
            extra_mask=live[:, None],
        )
    # 6. output projection in 2 K splits (128 tasks of 128 KB: the ring holds
    # most of a task's weights before its input is ready), BF16 into this rank's
    # symmetric partial (stock's AR dtype: the all-reduce's in-band bytes, not
    # its sum, set its latency)
    for tile_m, tile_n, tile_s in hl.tile([hidden, tokens, 2], block_size=[64, 16, 1]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(out_k, block_size=128):
            acc = hl.dot(
                w_out[tile_m, tile_s.begin, tile_k],
                ob[tile_n, tile_s.begin, tile_k].T,
                acc=acc,
            )
        symmetric_partial[tile_s.begin, tile_n, tile_m] = acc.T.to(torch.bfloat16)

    # 6b. the boundary's comb (raw: the previous boundary's logits, their
    # Sinkhorn here, before the chunk root reads it; final: copied). Last of the
    # body's roots: its 3 tasks take workers the others leave idle (as the first
    # root they shift the static placement of every later root: +4 us)
    for cs_t in hl.tile(tokens, block_size=2):
        cs_c = comb[cs_t, :, :]
        if comb_in_raw:
            cs_c = _mhc.sinkhorn_init(cs_c)
            for _cs_it in hl.tile(19, block_size=1):  # ITERS - 1
                cs_c = _mhc.sinkhorn_step(cs_c)
        comb_s[cs_t, :, :] = cs_c
    hl.pdl_launch_dependents()
    # 7-9. the mHC boundary (``_mhc``): the all-reduce of the ranks' o_proj
    # partials, hc_post, the FFN's hc_pre -> the FFN input, R', the FFN's post /
    # comb
    for tt, tc in hl.tile([tokens, hidden], block_size=[8, 128]):
        _mhc.chunk(
            remote_partials,
            tt,
            tc,
            r,
            post,
            comb_s,
            fn_pad,
            r_new,
            mix_part,
            gram_part,
            sq_part,
            ksplit=2,
            mix_t=mix_t,
        )
    for tb, tn in hl.tile([tokens, hidden], block_size=[8, 512]):
        _mhc.finalize(
            tb,
            tn,
            hidden,
            r_new,
            ffn_norm_w,
            mix_part,
            gram_part,
            sq_part,
            hc_scale,
            base_pre,
            li,
            mix_t=mix_t,
        )
    for ts in hl.tile(tokens, block_size=2):
        c = _mhc.comb_logits(
            ts,
            hidden,
            mix_part,
            sq_part,
            hc_scale,
            base_post,
            base_comb,
            post_out,
        )
        if not comb_out_raw:
            c = _mhc.sinkhorn_init(c)
            # ITERS - 1 (a literal: a module int reaches the kernel as a runtime scalar)
            for _it in hl.tile(19, block_size=1):
                c = _mhc.sinkhorn_step(c)
        comb_out[ts, :, :] = c


RING_LOADS = frozenset(["w_qkv", "x4", "w_fa", "w_ga", "x8", "w_out", "ob"])


def mk_config(bound) -> helion.Config:
    """The loads of ``RING_LOADS`` through the TMA ring (8 deep), every other
    access by pointer; one persistent CTA per SM claiming the tasks root by root;
    every projection's K loop warp-specialized (``k_loops``: each projection is a
    3-D root, the output projection with a unit split dimension)."""
    fields = ring_config_fields(bound, RING_LOADS, warp_specialize="k_loops")
    return helion.Config(
        indexing=fields["indexing"],
        pid_type="persistent_blocked",
        num_sm_multiplier=1,
        num_stages=8,
        num_warps=8,
        cross_loop_pipeline="dynamic",
        range_warp_specializes=fields["range_warp_specializes"],
    )


@functools.cache
def scratch(device: torch.device, tokens: int) -> dict[str, torch.Tensor]:
    """The kernel's scratch buffers on ``device``, one set for every call."""
    f32 = torch.float32
    S, D, H = _setup.SEGMENT, _setup.KDA_HEAD_DIM, _setup.KDA_HEADS

    def z(*shape: int, dtype=f32):
        return torch.zeros(*shape, device=device, dtype=dtype)

    return {
        "pq": z(3, QKV_SPLITS, LANES, H, D),
        "pfa": z(GATE_SPLITS, LANES, D),
        "pga": z(GATE_SPLITS, LANES, D),
        "pbeta": z(GATE_SPLITS, LANES, H),
        "stg": z(H * D // V_TILE, 3, LANES, D),
        "sdec": z(H * D // V_TILE, LANES, D),
        "sbeta": z(H * D // V_TILE, LANES),
        "opre": z(LANES, S),
        "ob": z(tokens, 2, S // 2, dtype=torch.bfloat16),
        "comb_s": z(tokens, 4, 4),
    }


def prepared_weights(
    fused_weight, fg_b, conv_weight, norm_weight, output_weight
) -> dict:
    """The layer's per-rank weights as the kernel reads them: views of SGLang's
    fused a-proj rows (no copies), F_B / G_B, the conv weights taps-major and
    the norm weight in FP32 (prepared once per layer)."""
    S, D, H = _setup.SEGMENT, _setup.KDA_HEAD_DIM, _setup.KDA_HEADS
    hidden = fused_weight.size(1)
    w = fused_weight
    fa, ga, b = _setup.FA_ROW, _setup.GA_ROW, _setup.B_ROW
    return {
        "w_qkv": w[: 3 * S].view(3 * H, D, QKV_SPLITS, hidden // QKV_SPLITS),
        "w_fa": w[fa : fa + D].view(D, GATE_SPLITS, hidden // GATE_SPLITS),
        "w_ga": w[ga : ga + D].view(D, GATE_SPLITS, hidden // GATE_SPLITS),
        "w_beta": w[b : b + H].view(H, GATE_SPLITS, hidden // GATE_SPLITS),
        "f_b": fg_b[0],
        "g_b": fg_b[1],
        # taps-major rows; SGLang's weight is channel-major [3S, 4]
        "conv_w": conv_weight.view(3, S, 4).transpose(1, 2).contiguous(),
        "norm_w": norm_weight.float().contiguous(),
        "w_out": output_weight.view(output_weight.size(0), 2, S // 2),
    }


def kernel_args(
    hidden_states,
    pw,
    a_log,
    dt_bias,
    conv_state,
    rec_state,
    state_idx,
    issm,
    iconv,
    iconv_idx,
    symmetric_partial,
    mhc,
    out,
    group_name,
    comb_in_raw=False,
    comb_out_raw=False,
) -> tuple:
    """``glmf_kda_tp4``'s arguments: K-split views of the hidden states, the
    prepared weights ``pw``, the state, the module's scratch, the mHC boundary's
    ``mhc`` (``_mhc.boundary_operands``: the FFN side's) and ``out`` = (li,
    r_new, post_out, comb_out). ``comb_in_raw``: mhc's comb holds the previous
    boundary's logits (``_mhc.comb_logits``), else the final comb; ``comb_out_raw``:
    comb_out gets this boundary's logits, else the final comb (each mode pair
    compiles a kernel of its own)."""
    tokens, hidden = hidden_states.shape
    sc = scratch(hidden_states.device, tokens)
    x4 = hidden_states.view(tokens, QKV_SPLITS, hidden // QKV_SPLITS)
    x8 = hidden_states.view(tokens, GATE_SPLITS, hidden // GATE_SPLITS)
    # a distinct view object for the plain loads: the memory ops of one tensor
    # object share its name, which would take x8's dots off the ring
    xp = hidden_states.view(tokens, GATE_SPLITS, hidden // GATE_SPLITS)
    return (
        x4,
        pw["w_qkv"],
        x8,
        pw["w_fa"],
        pw["w_ga"],
        xp,
        pw["w_beta"],
        pw["f_b"],
        pw["g_b"],
        pw["conv_w"],
        a_log,
        dt_bias,
        conv_state,
        rec_state,
        state_idx,
        issm,
        iconv,
        iconv_idx,
        pw["norm_w"],
        pw["w_out"],
        sc["pq"],
        sc["pfa"],
        sc["pga"],
        sc["pbeta"],
        sc["stg"],
        sc["sdec"],
        sc["sbeta"],
        sc["opre"],
        sc["ob"],
        symmetric_partial,
        *_mhc.tail_args(mhc, hidden_states.device, hidden),
        *out,
        sc["comb_s"],
        group_name,
        bool(comb_in_raw),
        bool(comb_out_raw),
    )


def symmetric_partial_spec(tokens: int) -> tuple[tuple[int, ...], torch.dtype]:
    """The (shape, dtype) of the symmetric buffer the kernel's o_proj writes: BF16
    partials of its two K splits."""
    return (2, tokens, _setup.HIDDEN), torch.bfloat16


def compile_kernel(args: tuple):
    """``glmf_kda_tp4`` compiled for ``args`` (once per binding)."""
    return _setup.compiled(glmf_kda_tp4, mk_config, args)


def _args(t: dict, partial, mhc, out, group_name, raw=False) -> tuple:
    """``kernel_args`` of ``_setup.kda_inputs``' tensors ``t``."""
    pw = prepared_weights(
        t["fused_weight"],
        t["fg_b"],
        t["conv_weight"],
        t["norm_weight"],
        t["output_weight"],
    )
    return kernel_args(
        t["hidden_states"],
        pw,
        t["a_log"],
        t["dt_bias"],
        t["conv_state"],
        t["recurrent_state"],
        t["state_indices"],
        t["intermediate_ssm"],
        t["intermediate_conv"],
        t["intermediate_indices"],
        partial,
        mhc,
        out,
        group_name,
        raw,
        raw,
    )


# ---------------------------- check and timing (torch.distributed.run, 4 ranks)


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Every rank its own layer shard and window (``_setup.kda_inputs``), for a
    request, a padding slot and a request with raw combs in and out: li, R',
    post', comb' against the boundary of the ranks' ``_setup.kda_reference``
    partials (``_setup.report_boundary``) and the cache writes against the
    reference's (``_setup.kda_writes_correct``). Then every case again back to
    back (no host synchronization) and once more with NaN scratch and outputs:
    the same bits, cache writes included."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    T, H = _setup.T, _setup.HIDDEN
    partial = symmetric_buffer(*symmetric_partial_spec(T), group.group_name, device)
    t = _setup.boundary_inputs(seed=11, device=device)
    launches, ok = [], True
    for name, padded, raw in (
        ("request", False, False),
        ("padding", True, False),
        ("request, raw combs", False, True),
    ):
        if raw:  # the comb as logits
            t = dict(t, comb=_setup.comb_logits(seed=5, device=device))
        inputs = _setup.kda_inputs(seed=3 + rank, padded=padded, device=device)
        reference = {**inputs, **{n: inputs[n].clone() for n in _setup.KDA_WRITES}}
        out = _mhc.new_outputs(t["r"])
        args = _args(
            inputs, partial, _mhc.boundary_operands(t), out, group.group_name, raw
        )
        kernel = compile_kernel(args)
        kernel(*args)
        torch.cuda.synchronize()
        rows = []
        ok &= _setup.report_boundary(
            rows, out, _setup.kda_reference(reference), t, group, raw
        )
        writes = _setup.kda_writes_correct(inputs, reference)
        rows.append(f"  {'cache writes':<22} {'ok' if writes else 'DIFFER  <-- FAIL'}")
        ok &= writes
        written = [inputs[n] for n in _setup.KDA_WRITES]
        launches.append((functools.partial(kernel, *args), [*out, *written]))
        if verbose or not ok:
            say(f"rank{rank} {name}\n" + "\n".join(rows))
    poisoned = [
        *launches[-1][1][:4],
        *scratch(device, T).values(),
        *_mhc.scratch(device, H).values(),
    ]
    back_to_back, nan = _setup.repeat_bitwise(launches, poisoned)
    ok &= back_to_back and nan
    if verbose or not ok:
        say(
            f"rank{rank} repeats: back to back {'bitwise' if back_to_back else 'DIFFER  <-- FAIL'}, "
            f"NaN scratch {'bitwise' if nan else 'DIFFERS  <-- FAIL'}"
        )
    verdict(ok, group, "glmf_kda_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing: ``LAYERS`` distinct layers (default 8: weights and
    states) back to back, ``REPS`` times (default 4) in one CUDA graph."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "8"))
    reps = int(os.environ.get("REPS", "4"))
    T = _setup.T
    t = _setup.boundary_inputs(device=device)
    mhc, out = _mhc.boundary_operands(t), _mhc.new_outputs(t["r"])
    partial = symmetric_buffer(*symmetric_partial_spec(T), group.group_name, device)
    args = [
        _args(
            _setup.kda_inputs(seed=1000 * i + rank, device=device),
            partial,
            mhc,
            out,
            group.group_name,
        )
        for i in range(layers)
    ]
    kernel = compile_kernel(args[0])

    def step() -> None:
        for a in args:
            kernel(*a)

    median, minimum = graph_time(step, layers, reps)
    if verbose:
        say(f"rank{rank} T={T} us/launch med {median:.2f} min {minimum:.2f}")
    return {T: {"latency_us": median, "min_latency_us": minimum}}


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main("glmf_kda_tp4", correctness_check, benchmark, verbose, check_only)


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
