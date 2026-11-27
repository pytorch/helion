# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""GLM-5.3 Flash MLA attention output side as one Gluon megakernel: a TP4 rank's
16 heads for one request of T <= 8 verify tokens, from the FP8 absorbed queries and
the DSA top-k to the FFN's input (SGLang: trtllm-gen sparse MLA, W_UV bmm, o_proj,
all-reduce, mHC boundary; ``glm5_3_tp4._setup.out_reference`` and
``_setup.boundary`` the PyTorch references).

Roots, in launch order (one persistent CTA per SM, TMA ring for the weights):
  1. splits: per (token, split of 128 top-k picks; 64 up to 4 tokens), the sparse
     MLA over the picked slots of the raw FP8 cache (512 B a token, NoPE) with
     swapped operands (slots as the MMA rows, heads as its columns): S^T = K q^T, a
     one-block softmax (log2 domain), O^T = K^T P^T with P in E4M3 x 448 (x 256 for
     64 picks); BF16 [512, 16] partials and the per-head log2-sum-exp2 (FP32), as
     stock's trtllm-gen sparse MLA stores its multi-CTA KV-split partials. The top-k
     list is packed (valid slots first): a split past a row's count loads nothing.
  2. merge: per (token, 32 latent columns), the splits in split order, BF16 (the
     trtllm kernel's output).
  3. W_UV per (head, 128 value rows), BF16 (SGLang's bmm).
  4. o_proj [4096, 4096] per (128 rows, K chunk of 4 heads) (128 tasks): BF16 partials
     into this rank's symmetric buffer [4, T, 4096] (no publish hop: the all-reduce sums
     every rank's chunks).
  5-7. the attention -> FFN mHC boundary (``_mhc``, token-major mode): the all-reduce
     of the ranks' partials (split-major over ranks, FP32), hc_post with the
     attention's post / comb, the FFN's hc_pre (+ post_attention_layernorm) -> the FFN
     input, R', the FFN's post / comb (Sinkhorn in 2-token tiles).

Every split sum runs in a fixed order (no atomics): deterministic run to run.

``correctness_check()`` compares every rank with the torch reference (``_setup``) on
synthetic weights; ``main()`` runs it, then times the kernel with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_flash_tp4/mla_out_tp4.py [--check]
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import symmetric_buffer
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels._ring_config import ring_config_fields
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _mhc
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _setup
from pretuned_kernels.megakernels.glm5_3_tp4._setup import make_cache
from pretuned_kernels.megakernels.glm5_3_tp4._setup import out_reference
from pretuned_kernels.megakernels.glm5_3_tp4._setup import out_weights
import torch
import torch.distributed as dist

import helion
import helion.language as hl

# A row's top-k list: 512 closed pools of 4 tokens, then its open pool's <= 3.
TOPK_WIDTH = _setup.ATTN_SHAPE.top_k + _setup.ATTN_SHAPE.kpool - 1


def splits_for(tokens: int, width: int = TOPK_WIDTH) -> tuple[int, int]:
    """(picks a split, splits a token): 64-pick splits up to 4 tokens, 128 past 4 (the
    window's split tasks stay within one wave of workers)."""
    n = 64 * (1 + tokens // 5)
    return n, (width + n - 1) // n


# The device helpers below share the sparse MLA's split and merge roots with
# ``mtp_attn_out_tp4``. Inside a helper a subscript without a tile is torch's,
# which rejects a symbolic row or axis: they index by hl.load / hl.store (ALL: a
# whole axis) and expand with unsqueeze.
ALL = slice(None)


def attention_split(
    q, kv_cache, topk, att_ws, lse_ws, token, split, split_n, sizes, scale_log2, p_scale
) -> None:
    """Row ``token``'s sparse MLA over its picks ``split`` * ``split_n`` onward
    (one block, no online rescaling; swapped operands): the split's output
    (latent-major, BF16) into ``att_ws`` and its log2-sum-exp2 into ``lse_ws``.
    P goes to E4M3 scaled by ``p_scale`` against the FP8 latent (divided out).
    ``sizes``: the kernel's specialized (top-k width, latent width)."""
    top_k, kv_lora = sizes
    sp_q = hl.load(q, [token, ALL, ALL])
    cols = split * split_n + hl.arange(split_n)
    slot = hl.load(topk, [token, cols], extra_mask=cols < top_k)
    valid = (slot >= 0) & (cols < top_k)
    safe = torch.where(valid, slot, 0)
    key = hl.load(
        kv_cache,
        [safe.unsqueeze(1), hl.arange(kv_lora).unsqueeze(0)],
        extra_mask=valid.unsqueeze(1),
    )
    scores = hl.dot(key, sp_q.T, out_dtype=torch.float32)
    scores = torch.where(valid.unsqueeze(1), scores * scale_log2, float("-inf"))
    sp_max = torch.amax(scores, dim=0)
    safe_max = torch.where(sp_max == float("-inf"), 0.0, sp_max)
    prob = torch.exp2(scores - safe_max.unsqueeze(0))
    sp_sum = torch.sum(prob, dim=0)
    acc = hl.dot(
        key.T, (prob * p_scale).to(torch.float8_e4m3fn), out_dtype=torch.float32
    )
    seen = sp_sum > 0.0
    safe_sum = torch.where(seen, sp_sum, 1.0)
    hl.store(
        att_ws,
        [split, token, ALL, ALL],
        (acc / (p_scale * safe_sum.unsqueeze(0))).to(torch.bfloat16),
    )
    hl.store(
        lse_ws,
        [split, token, ALL],
        torch.where(seen, safe_max + torch.log2(safe_sum), float("-inf")),
    )


def merge_splits(lse_ws, att_ws, attn, token, cols) -> None:
    """Row ``token``'s split outputs at latent columns ``cols`` merged in split
    order -> ``attn`` (BF16)."""
    mg_lse = hl.load(lse_ws, [ALL, token, ALL])
    mg_vals = att_ws[:, token, cols, :].to(torch.float32)
    mg_best = torch.amax(mg_lse, dim=0)
    mg_best = torch.where(mg_best == float("-inf"), 0.0, mg_best)
    mg_w = torch.exp2(mg_lse - mg_best.unsqueeze(0))
    mg_out = torch.sum(mg_w.unsqueeze(1) * mg_vals, dim=0)
    mg_den = torch.sum(mg_w, dim=0)
    attn[token, :, cols] = (
        mg_out / torch.where(mg_den > 0, mg_den, 1.0).unsqueeze(0)
    ).T.to(torch.bfloat16)


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def glmf_mla_out_tp4(
    q,  # [T, 16, 512] e4m3: the absorbed queries
    kv_cache,  # [slots, 512] e4m3: one layer's raw MLA cache
    topk,  # [T, W] int32: each row's picked slots, packed (-1 past them)
    w_uv,  # [16, 256, 512] bf16: W_UV per head (K contiguous)
    scale_log2: float,  # softmax scale * log2(e)
    att_ws,  # [S, T, 512, 16] bf16 scratch: split outputs (latent-major)
    lse_ws,  # [S, T, 16] fp32 scratch: split log2-sum-exp2
    attn,  # [T, 16, 512] bf16 out: the sparse MLA
    attn_v,  # [T, 16, 256] bf16 out: the value rows
    w_o,  # [4096, K, 4096 / K] bf16: o_proj in K chunks (a view)
    symmetric_partial,  # [K, T, 4096] bf16 symmetric memory: this rank's o_proj K-chunk partials
    # the attention -> FFN mHC boundary (``_mhc``)
    r,  # [T, 4, 4096] bf16
    post,  # [T, 4] fp32
    comb,  # [T, 4, 4] fp32
    fn_pad,  # [32, 4, 4096] bf16
    hc_scale,  # [3] fp32
    base_pre,  # [4] fp32
    base_post,  # [4] fp32
    base_comb,  # [4, 4] fp32
    ffn_norm_w,  # [4096] bf16: post_attention_layernorm
    mix_part,  # [S2, 8, 32] fp32 scratch
    gram_part,  # [8, 4, 4, S2] fp32 scratch, token-major
    sq_part,  # [8, S2] fp32 scratch, token-major
    mix_t,  # [8, 32, S2] fp32 scratch
    li,  # [T, 4096] bf16 out: the FFN's input
    r_new,  # [T, 4, 4096] bf16 out
    post_out,  # [T, 4] fp32 out
    comb_out,  # [T, 4, 4] fp32 out (``comb_out_raw``: logits)
    comb_s,  # [T, 4, 4] fp32 scratch: the final comb (``comb_in_raw``: the head root's Sinkhorn)
    group_name: hl.ProcessGroupName,
    comb_in_raw: hl.constexpr,  # comb holds the previous boundary's logits: Sinkhorn them here
    comb_out_raw: hl.constexpr,  # comb_out gets this boundary's logits (the next MK takes them raw)
) -> None:
    hl.specialize(q.size())
    hl.specialize(q.stride())
    hl.specialize(kv_cache.stride())
    hl.specialize(topk.size())
    hl.specialize(topk.stride())
    hl.specialize(w_uv.size())
    hl.specialize(w_uv.stride())
    hl.specialize(att_ws.size())
    hl.specialize(att_ws.stride())
    hl.specialize(lse_ws.size())
    hl.specialize(lse_ws.stride())
    hl.specialize(attn.size())
    hl.specialize(attn.stride())
    hl.specialize(attn_v.size())
    hl.specialize(attn_v.stride())
    hl.specialize(w_o.size())
    hl.specialize(w_o.stride())
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
    hl.specialize(gram_part.size())
    hl.specialize(gram_part.stride())
    hl.specialize(sq_part.size())
    hl.specialize(sq_part.stride())
    hl.specialize(mix_t.size())
    hl.specialize(mix_t.stride())
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

    rows, heads, kv_lora = q.shape
    top_k = topk.size(1)
    v_head = w_uv.size(1)
    hidden, o_chunks, o_k = w_o.shape
    q_width = o_chunks * o_k
    assert rows <= 8 and heads == 16 and kv_lora == 512 and q_width == heads * v_head
    assert mix_part.size(0) * 128 == hidden
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    # (Arithmetic on the specialized row count: a branch merges into a symbol.)
    longer = rows // 5
    split_n = 64 * (1 + longer)
    splits = (top_k + split_n - 1) // split_n
    # P's E4M3 scale: the format's maximum for the longer splits (their smaller
    # probabilities keep more mantissa bits).
    p_scale = 256.0 + 192.0 * longer
    assert att_ws.size(0) == splits
    attn_v_chunks = attn_v.view(rows, o_chunks, o_k)
    remote_partials = torch.ops.symm_mem.get_remote_tensors(
        symmetric_partial, group_name
    )

    hl.pdl_wait()
    # The code toward L2 first: in a decoder layer the previous weight stream
    # evicted it.
    hl.prefetch_code()

    # 0. the boundary's comb (raw: the previous boundary's logits, their
    # Sinkhorn here, long before the chunk root reads it; final: copied)
    for cs_t in hl.tile(rows, block_size=2):
        cs_c = comb[cs_t, :, :]
        if comb_in_raw:
            cs_c = _mhc.sinkhorn_init(cs_c)
            for _cs_it in hl.tile(19, block_size=1):  # ITERS - 1
                cs_c = _mhc.sinkhorn_step(cs_c)
        comb_s[cs_t, :, :] = cs_c

    # 1. sparse MLA splits per (token, split).
    for sp_t, sp_s in hl.tile([rows, splits], block_size=[1, 1]):
        attention_split(
            q,
            kv_cache,
            topk,
            att_ws,
            lse_ws,
            sp_t.begin,
            sp_s.begin,
            split_n,
            (top_k, kv_lora),
            scale_log2,
            p_scale,
        )

    # 2. merge per (token, 32 latent columns): the splits in split order.
    for mg_t, mg_c in hl.tile([rows, kv_lora], block_size=[1, 32]):
        merge_splits(lse_ws, att_ws, attn, mg_t.begin, mg_c)

    # 3. value up-projection per (head, 128 value rows).
    for uv_h, uv_n, uv_t in hl.tile([heads, v_head, rows], block_size=[1, 128, 16]):
        uv_acc = hl.zeros([uv_n, uv_t], dtype=torch.float32)
        for uv_k in hl.tile(kv_lora, block_size=64):
            uv_acc = hl.dot(
                w_uv[uv_h.begin, uv_n, uv_k], attn[uv_t, uv_h.begin, uv_k].T, acc=uv_acc
            )
        attn_v[uv_t, uv_h.begin, uv_n] = uv_acc.T.to(torch.bfloat16)

    # 4. o_proj per (128 rows, K chunk): BF16 partials (stock's AR dtype) into
    # this rank's symmetric buffer (the boundary's all-reduce sums every rank's
    # chunks: no publish hop).
    for o_n, o_c, o_t in hl.tile([hidden, o_chunks, rows], block_size=[128, 1, 16]):
        o_acc = hl.zeros([o_n, o_t], dtype=torch.float32)
        for o_kk in hl.tile(o_k, block_size=64):
            o_acc = hl.dot(
                w_o[o_n, o_c.begin, o_kk],
                attn_v_chunks[o_t, o_c.begin, o_kk].T,
                acc=o_acc,
            )
        symmetric_partial[o_c.begin, o_t, o_n] = o_acc.T.to(torch.bfloat16)

    hl.pdl_launch_dependents()
    # 5-7. the mHC boundary: AR over the ranks' partials, hc_post, the FFN's
    # hc_pre.
    for tt, tc in hl.tile([rows, hidden], block_size=[8, 128]):
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
            ksplit=o_chunks,
            mix_t=mix_t,
        )
    for tb, tn in hl.tile([rows, hidden], block_size=[8, 512]):
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
    for ts in hl.tile(rows, block_size=2):
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


RING_LOADS = frozenset(["w_uv", "attn", "w_o", "attn_v_chunks"])
O_CHUNKS = 4  # o_proj's K chunks (BF16 partials each: the all-reduce sums them)


def mk_config(bound) -> helion.Config:
    """The weight streams (W_UV, o_proj) and their activation operands through the TMA
    ring, every other access by pointer (the key gathers among them); the dot roots
    warp-specialized; the all-reduce in band (Lamport). No tensor-memory prefetch:
    at T=6 with raw combs (as served), cold, W_UV's tail gains nothing and W_UV's
    and o_proj's together cost 6%."""
    fields = ring_config_fields(bound, RING_LOADS)
    default = bound.config_spec.default_config().config
    return helion.Config(
        indexing=fields["indexing"],
        pid_type="persistent_blocked",
        num_sm_multiplier=1,
        num_stages=6,
        num_warps=4,
        cross_loop_pipeline="dynamic",
        range_warp_specializes=fields["range_warp_specializes"],
        cross_loop_inband_protocol=["lamport"]
        * len(default.get("cross_loop_inband_protocol", [])),
    )


@functools.cache
def scratch(device: torch.device, tokens: int) -> dict[str, torch.Tensor]:
    """The kernel's scratch and per-call outputs (attn, attn_v) on ``device``."""
    _, splits = splits_for(tokens)
    heads, latent, v_head = (
        _setup.ATTN_SHAPE.heads,
        _setup.ATTN_SHAPE.kv_lora,
        _setup.ATTN_SHAPE.v_head,
    )
    bf16 = torch.bfloat16
    return {
        "att_ws": torch.zeros(splits, tokens, latent, heads, device=device, dtype=bf16),
        "lse_ws": torch.zeros(
            splits, tokens, heads, device=device, dtype=torch.float32
        ),
        "attn": torch.zeros(tokens, heads, latent, device=device, dtype=bf16),
        "attn_v": torch.zeros(tokens, heads, v_head, device=device, dtype=bf16),
        "comb_s": torch.zeros(tokens, 4, 4, device=device, dtype=torch.float32),
    }


def kernel_args(
    q8,
    kv_cache,
    topk,
    w_uv,
    o_proj,
    symmetric_partial,
    mhc,
    out,
    group_name,
    comb_in_raw=False,
    comb_out_raw=False,
) -> tuple:
    """``glmf_mla_out_tp4``'s arguments: q8 [T, 16, 512] E4M3 (the front's), kv_cache
    [slots, 512] E4M3, topk [T, 2051] int32 cache slots (-1 padded: SGLang's pooled
    top-k with the token -> slot table), W_UV [16, 256, 512], o_proj [4096, 16 x
    256], ``symmetric_partial`` per ``symmetric_partial_spec`` (rendezvoused on
    ``group_name``) and the attention -> FFN boundary's ``mhc``
    (``_mhc.boundary_operands``). The kernel writes ``out`` = (li [T, H] BF16:
    the FFN input, r_new [T, 4, H] BF16: R', post_out [T, 4] FP32, comb_out [T, 4,
    4] FP32) (``_mhc.new_outputs``). ``comb_in_raw``: mhc's comb holds the
    previous boundary's logits (``_mhc.comb_logits``), else the final comb;
    ``comb_out_raw``: comb_out gets this boundary's logits, else the final comb."""
    tokens = q8.shape[0]
    sc = scratch(q8.device, tokens)
    r, post, comb, fpad, hc_scale, base, norm_w = mhc
    ms = _mhc.scratch(q8.device, o_proj.shape[0])
    return (
        q8,
        kv_cache,
        topk,
        w_uv,
        _setup.SCALE_LOG2,
        sc["att_ws"],
        sc["lse_ws"],
        sc["attn"],
        sc["attn_v"],
        o_proj.view(o_proj.shape[0], O_CHUNKS, o_proj.shape[1] // O_CHUNKS),
        symmetric_partial,
        r,
        post,
        comb,
        fpad,
        hc_scale,
        base[0:4],
        base[4:8],
        base[8:24].view(4, 4),
        norm_w,
        ms["mix_part"],
        ms["gram_t"],
        ms["sq_t"],
        ms["mix_t"],
        *out,
        sc["comb_s"],
        group_name,
        bool(comb_in_raw),
        bool(comb_out_raw),
    )


def symmetric_partial_spec(tokens: int) -> tuple[tuple[int, ...], torch.dtype]:
    """The (shape, dtype) of the symmetric buffer the kernel's o_proj writes: BF16
    partials of its O_CHUNKS K chunks (the boundary's all-reduce sums every rank's
    chunks)."""
    return (O_CHUNKS, tokens, _setup.HIDDEN), torch.bfloat16


def compile_kernel(args: tuple):
    """``glmf_mla_out_tp4`` compiled for ``args`` (once per binding)."""
    return _setup.compiled(glmf_mla_out_tp4, mk_config, args)


def _args(w, c, partial, mhc, out, group_name, raw=False) -> tuple:
    """``kernel_args`` of out weights ``w`` and a cache ``c`` (``glm5_3_tp4._setup``'s
    ``out_weights`` / ``make_cache`` at ``_setup.ATTN_SHAPE``)."""
    return kernel_args(
        c["q"],
        c["kv_cache"],
        c["topk"],
        w["w_uv"],
        w["o_proj"],
        partial,
        mhc,
        out,
        group_name,
        raw,
        raw,
    )


# ---------------------------- check and timing (torch.distributed.run, 4 ranks)


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Every rank its own weights' slice and cache (``glm5_3_tp4._setup``'s at
    ``_setup.ATTN_SHAPE``): a short context (every pool picked, the list shorter
    than its width), a long one (512 random pools and the open pool) and the short
    one with raw combs in and out: the sparse MLA and the value rows against
    ``out_reference``, li, R', post', comb' against the boundary of the ranks'
    reference partials (``_setup.report_boundary``). Then every case again back
    to back (no host synchronization) and once more with NaN scratch and
    outputs: the same bits."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    T, shape = _setup.T, _setup.ATTN_SHAPE
    t = _setup.boundary_inputs(seed=11, device=device)
    partial = symmetric_buffer(*symmetric_partial_spec(T), group.group_name, device)
    w = out_weights(shape, seed=100 + rank, device=device)
    launches, ok = [], True
    for context, raw in ((700, False), (9000, False), (700, True)):
        if raw:  # the comb as logits
            t = dict(t, comb=_setup.comb_logits(seed=5, device=device))
        c = make_cache(T, context, shape, seed=11 + rank, device=device)
        out = _mhc.new_outputs(t["r"])
        args = _args(
            w, c, partial, _mhc.boundary_operands(t), out, group.group_name, raw
        )
        kernel = compile_kernel(args)
        kernel(*args)
        torch.cuda.synchronize()
        attn, v, reference = out_reference(c["q"], c["kv_cache"], c["topk"], w, shape)
        sc = scratch(device, T)
        rows = []
        # P in E4M3 (as trtllm-gen's sparse MLA)
        ok &= report(rows, "sparse MLA", sc["attn"], attn, 2e-2)
        ok &= report(rows, "value rows", sc["attn_v"], v, 2e-2)
        ok &= _setup.report_boundary(rows, out, reference, t, group, raw)
        launches.append((functools.partial(kernel, *args), out))
        if verbose or not ok:
            say(f"rank{rank} context {context} raw combs {raw}\n" + "\n".join(rows))
    poisoned = [
        *launches[-1][1],
        *scratch(device, T).values(),
        *_mhc.scratch(device, _setup.HIDDEN).values(),
    ]
    back_to_back, nan = _setup.repeat_bitwise(launches, poisoned)
    ok &= back_to_back and nan
    if verbose or not ok:
        say(
            f"rank{rank} repeats: back to back {'bitwise' if back_to_back else 'DIFFER  <-- FAIL'}, "
            f"NaN scratch {'bitwise' if nan else 'DIFFERS  <-- FAIL'}"
        )
    verdict(ok, group, "glmf_mla_out_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing: ``LAYERS`` distinct weight copies and caches (default 8)
    back to back, ``REPS`` times (default 4) in one CUDA graph; ``CONTEXT``
    committed tokens (default 8192)."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "8"))
    reps = int(os.environ.get("REPS", "4"))
    context = int(os.environ.get("CONTEXT", "8192"))
    T, shape = _setup.T, _setup.ATTN_SHAPE
    t = _setup.boundary_inputs(device=device)
    mhc, out = _mhc.boundary_operands(t), _mhc.new_outputs(t["r"])
    partial = symmetric_buffer(*symmetric_partial_spec(T), group.group_name, device)
    args = [
        _args(
            out_weights(shape, seed=1000 * i + rank, device=device),
            make_cache(T, context, shape, seed=i, device=device),
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
        say(
            f"rank{rank} T={T} context={context} us/launch med {median:.2f} min {minimum:.2f}"
        )
    return {T: {"latency_us": median, "min_latency_us": minimum}}


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The four-rank check, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main(
        "glmf_mla_out_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
