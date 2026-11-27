# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""GLM-5.3 Flash MLA prologue + DSA indexer as one Gluon megakernel (``mla_front_tp4``'s
roots and the pooled indexer's in one launch): both read the layer input, the
indexer's query reads q_lora, so the two chains overlap instead of the indexer
waiting for the whole front. For one request of T <= 8 verify rows: the FP8
absorbed queries and the window's MLA cache rows, the tail-ring / pooled-cache writes and
every row's logits over its pools (SGLang's top-k v2 reads them).

Roots, in launch order (each a single wave on the 152 workers, the small ones packed on
the workers the big ones leave idle):
  1. a | kw: q_a | kv_a | [wk | weights_proj | compress_gate | 0] (BF16 [2432, 4096]) per
     (128 rows, K chunk of 512): 152 tasks of FP32 partials.
  2. q norm per token -> q_lora (BF16); 3. kv per token -> RMSNorm, E4M3 into the cache.
  4. key per token: LayerNorm (BF16) and gate score into the tail ring.
  5. compress: the window's closed pools (<= 2) -> the pooled index-K cache (one task).
  6. wq_b | q_b (BF16 [32 x 128 + 16 x 256, 1536]) per (128 rows, K chunk of 768): 128
     tasks of FP32 partials (the indexer's rows first: its chain is the longer).
  7. indexer query per (token, 16 heads): Hadamard (dot), pow2 FP8, head weights.
  8. absorb per (head, 128 latent rows): W_UK over q's nope dims -> E4M3 queries.
  9. logits per pooled page (DeepGEMM's epilogue).

A padding request (its first MLA cache slot 0: SGLang's dummy slot, which CUDA-graph
padding writes) skips the tail-ring and pooled-cache writes: roots 4-5 store only
when cache_loc[0] != 0. The split-K partials are summed in a fixed order and the
logits in DeepGEMM's (no atomics): the outputs are run-to-run deterministic.

``correctness_check()`` compares every rank with the torch reference (``_setup``) on
synthetic weights; ``main()`` runs it, then times the kernel with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_flash_tp4/mla_frontidx_tp4.py [--check]
"""

from __future__ import annotations

import functools
from itertools import starmap
import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import same
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels._ring_config import ring_config_fields
from pretuned_kernels.megakernels.glm5_3_flash_tp4 import _setup
from pretuned_kernels.megakernels.glm5_3_tp4._setup import cache_views
from pretuned_kernels.megakernels.glm5_3_tp4._setup import front_reference
from pretuned_kernels.megakernels.glm5_3_tp4._setup import front_weights
from pretuned_kernels.megakernels.glm5_3_tp4.glm5_3_indexer_tp4 import page_logits
import torch
import torch.distributed as dist

import helion
import helion.language as hl

LANES = 8
A_CHUNKS = 8  # K chunks of the layer input (512): the a and kw projections
B_CHUNKS = 2  # K chunks of q_lora (768): wq_b | q_b, 128 tasks (one wave beside the small roots)


def _logits_page(start, keys, kscale, row_q, row_w, logits):
    """One pooled page of 64 pools from pool ``start`` (its keys / scales loaded):
    their logits as GLM-5.3's keys' (``page_logits``)."""
    page_logits(start + hl.arange(64), keys, kscale, row_q, row_w, logits)


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def glmf_mla_frontidx_tp4(
    x_a,  # [T, A, 4096 / A] bf16: the layer input in K chunks (a view)
    w_ak,  # [2432, A, 4096 / A] bf16: q_a | kv_a | wk | weights_proj | compress_gate | 0
    q_a_norm,  # [1536] bf16
    kv_a_norm,  # [512] bf16
    w_q,  # [32 x 128 + 16 x 256, B, 1536 / B] bf16: wq_b | q_b
    w_uk,  # [16, 512, 256] bf16
    k_norm_w,  # [128] fp32
    k_norm_b,  # [128] fp32
    ape,  # [4, 128] fp32
    hmat,  # [128, 128] bf16
    cache_loc,  # [T] int64: the window's MLA cache slots
    kv_cache,  # [slots, 512] e4m3
    tail_k,  # [reqs, 10, 128] bf16
    tail_s,  # [reqs, 10, 128] bf16
    req,  # [1] int64
    write_start,  # [1] int32
    tail_start,  # [1] int32
    write_loc,  # [2] int64
    cache_keys,  # [pages, 66, 128] e4m3: pooled index-K cache keys
    cache_scales,  # [pages, 2112] fp32: its scales
    page_table,  # [W] int32: the request's pooled pages
    pool_lens,  # [T] int32
    ak_parts,  # [A, 8, 2432] fp32 scratch
    q_lora,  # [T, 1536] bf16 out
    q_parts,  # [B, 8, 8192] fp32 scratch
    q8,  # [T, 16, 512] e4m3 out: the MLA queries
    iq8,  # [8, 32, 128] e4m3 out: the indexer queries
    weights,  # [8, 32] fp32 out
    logits,  # [8, W * 64] fp32 out
) -> None:
    hl.specialize(x_a.size())
    hl.specialize(x_a.stride())
    hl.specialize(w_ak.size())
    hl.specialize(w_ak.stride())
    hl.specialize(q_a_norm.size())
    hl.specialize(q_a_norm.stride())
    hl.specialize(kv_a_norm.size())
    hl.specialize(kv_a_norm.stride())
    hl.specialize(w_q.size())
    hl.specialize(w_q.stride())
    hl.specialize(w_uk.size())
    hl.specialize(w_uk.stride())
    hl.specialize(k_norm_w.size())
    hl.specialize(k_norm_b.size())
    hl.specialize(ape.size())
    hl.specialize(ape.stride())
    hl.specialize(hmat.size())
    hl.specialize(hmat.stride())
    hl.specialize(cache_loc.size())
    hl.specialize(cache_loc.stride())
    hl.specialize(kv_cache.stride())
    hl.specialize(tail_k.size())
    hl.specialize(tail_k.stride())
    hl.specialize(tail_s.size())
    hl.specialize(tail_s.stride())
    hl.specialize(req.size())
    hl.specialize(write_start.size())
    hl.specialize(tail_start.size())
    hl.specialize(write_loc.size())
    hl.specialize(write_loc.stride())
    hl.specialize(cache_keys.stride())
    hl.specialize(cache_scales.stride())
    hl.specialize(page_table.size())
    hl.specialize(pool_lens.size())
    hl.specialize(ak_parts.size())
    hl.specialize(ak_parts.stride())
    hl.specialize(q_lora.size())
    hl.specialize(q_lora.stride())
    hl.specialize(q_parts.size())
    hl.specialize(q_parts.stride())
    hl.specialize(q8.size())
    hl.specialize(q8.stride())
    hl.specialize(iq8.size())
    hl.specialize(iq8.stride())
    hl.specialize(weights.size())
    hl.specialize(weights.stride())
    hl.specialize(logits.size())
    hl.specialize(logits.stride())

    rows, a_chunks, a_k = x_a.shape
    ak_rows = w_ak.size(0)
    q_lora_n = q_a_norm.size(0)
    kv_lora = kv_a_norm.size(0)
    a_rows = q_lora_n + kv_lora  # the kw rows follow
    heads, _, nope = w_uk.shape
    q_rows, b_chunks, b_k = w_q.shape
    head_dim = k_norm_w.size(0)
    iheads = 32
    iq_rows = iheads * head_dim  # the q_b rows follow
    ring = tail_k.size(1)
    pages_w = page_table.size(0)
    assert rows <= 8 and ak_rows == a_rows + 384 and q_rows == iq_rows + heads * nope
    assert head_dim == 128 and nope == 2 * head_dim
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    eps = 1e-5
    ln_eps = 1e-6
    page = 64
    kpool = 4
    q_lora_c = q_lora.view(rows, b_chunks, b_k)
    row_groups = (rows + 1) // 2
    iq8_rows = iq8.view(8 * iheads, head_dim)
    q_parts_heads = q_parts.view(b_chunks, 8, q_rows // head_dim, head_dim)
    qb_parts = q_parts.view(
        b_chunks, 8, q_rows // nope, nope
    )  # q_b's heads from iheads // 2
    # 148: at most one logits task per worker of a B200 (148 SMs; a GB300 runs
    # 152 workers, 4 of them without one). GLM-5.3's choice (measured against
    # 128 tasks, with other changes), not measured against 152.
    logit_tasks = min(pages_w, 148)

    hl.pdl_wait()
    hl.prefetch_code()

    # 1. the fused a | kw projection: q_a | kv_a | the indexer's key / head gate
    # / compress gate
    for a_n, a_c, a_t in hl.tile([ak_rows, a_chunks, rows], block_size=[128, 1, 16]):
        a_acc = hl.zeros([a_n, a_t], dtype=torch.float32)
        for a_kk in hl.tile(a_k, block_size=128):
            a_acc = hl.dot(
                w_ak[a_n, a_c.begin, a_kk], x_a[a_t, a_c.begin, a_kk].T, acc=a_acc
            )
        ak_parts[a_c.begin, a_t, a_n] = a_acc.T

    # 2. q norm per token -> q_lora
    for qn_t in hl.tile(rows, block_size=1):
        qn_tok = qn_t.begin
        qn_w = q_a_norm[:].to(torch.float32)
        qn_x = torch.sum(ak_parts[:, qn_tok, :q_lora_n], dim=0)
        qn_x = qn_x.to(torch.bfloat16).to(torch.float32)
        qn_inv = torch.rsqrt(torch.sum(qn_x * qn_x) / q_lora_n + eps)
        q_lora[qn_tok, :] = (qn_x * qn_inv * qn_w).to(torch.bfloat16)

    # 3. kv per token -> the MLA cache slot
    for kn_t in hl.tile(rows, block_size=1):
        kn_tok = kn_t.begin
        kn_w = kv_a_norm[:].to(torch.float32)
        kn_slot = cache_loc[kn_tok]
        kn_x = torch.sum(ak_parts[:, kn_tok, q_lora_n:a_rows], dim=0)
        kn_x = kn_x.to(torch.bfloat16).to(torch.float32)
        kn_inv = torch.rsqrt(torch.sum(kn_x * kn_x) / kv_lora + eps)
        kv_cache[kn_slot, :] = (
            (kn_x * kn_inv * kn_w).to(torch.bfloat16).to(torch.float8_e4m3fn)
        )

    # 4. the window's index keys and gate scores into the tail ring
    for kp_t in hl.tile(rows, block_size=1):
        kp_tok = kp_t.begin
        kp_req = req[0]
        kp_slot = (write_start[0] + kp_tok) % ring
        kp_w = k_norm_w[:]
        kp_b = k_norm_b[:]
        kp_k = torch.sum(ak_parts[:, kp_tok, a_rows : a_rows + head_dim], dim=0)
        kp_s = torch.sum(
            ak_parts[
                :, kp_tok, a_rows + head_dim + iheads : a_rows + 2 * head_dim + iheads
            ],
            dim=0,
        )
        kp_k = kp_k.to(torch.bfloat16).to(torch.float32)
        kp_mean = torch.sum(kp_k) / head_dim
        kp_d = kp_k - kp_mean
        kp_inv = torch.rsqrt(torch.sum(kp_d * kp_d) / head_dim + ln_eps)
        kp_dim = hl.arange(head_dim)
        kp_m = (kp_dim * 0 + cache_loc[0]) != 0
        hl.store(
            tail_k,
            [kp_req, kp_slot, kp_dim],
            (kp_d * kp_inv * kp_w + kp_b).to(torch.bfloat16),
            extra_mask=kp_m,
        )
        hl.store(
            tail_s, [kp_req, kp_slot, kp_dim], kp_s.to(torch.bfloat16), extra_mask=kp_m
        )

    # 5. the window's closed pools (16-pool tile, the first n live), one task
    for cp_i in hl.tile(1, block_size=1):  # noqa: B007
        cp_hm = hmat[:, :]  # first: it waits for nothing
        cp_req = req[0]
        cp_ws = write_start[0]
        cp_base = tail_start[0]
        cp_live0 = cache_loc[0] != 0
        cp_n = (cp_ws + rows) // kpool - cp_ws // kpool
        cp_j = hl.arange(16)
        cp_live = (cp_j < cp_n) & cp_live0
        cp_dim = hl.arange(head_dim)
        cp_sc = [
            hl.load(
                tail_s,
                [
                    cp_req,
                    torch.where(cp_live, (cp_base + cp_j * kpool + s) % ring, 0)[
                        :, None
                    ],
                    cp_dim[None, :],
                ],
            ).to(torch.float32)
            + hl.load(ape, [s + cp_j[:, None] * 0, cp_dim[None, :]])
            for s in hl.static_range(4)
        ]
        cp_kv = [
            hl.load(
                tail_k,
                [
                    cp_req,
                    torch.where(cp_live, (cp_base + cp_j * kpool + s) % ring, 0)[
                        :, None
                    ],
                    cp_dim[None, :],
                ],
            ).to(torch.float32)
            for s in hl.static_range(4)
        ]
        # (two 1-D index tensors would form an outer product: one index tensor)
        cp_loc = hl.load(write_loc, [torch.where(cp_live, cp_j, 0)])
        cp_m = torch.maximum(
            torch.maximum(cp_sc[0], cp_sc[1]), torch.maximum(cp_sc[2], cp_sc[3])
        )
        cp_p = [torch.exp(v - cp_m) for v in cp_sc]
        cp_den = (cp_p[0] + cp_p[1]) + (cp_p[2] + cp_p[3])
        cp_acc = (cp_kv[0] * cp_p[0] + cp_kv[1] * cp_p[1]) + (
            cp_kv[2] * cp_p[2] + cp_kv[3] * cp_p[3]
        )
        cp_x = (cp_acc / cp_den).to(torch.bfloat16)
        # H is symmetric: x H = (H x^T)^T  [16 pools, 128 dims]
        cp_y = hl.dot(cp_x, cp_hm, out_dtype=torch.float32)
        # 0.0883883...: 128^-0.5, the Hadamard transform's normalization
        cp_y = (cp_y * 0.08838834764831845).to(torch.bfloat16).to(torch.float32)
        cp_amax = torch.clamp(torch.amax(torch.abs(cp_y), dim=1), min=1e-4)
        cp_scale = torch.exp2(torch.ceil(torch.log2(cp_amax * (1.0 / 448.0))))
        cp_q = torch.clamp(cp_y / cp_scale[:, None], -448.0, 448.0).to(
            torch.float8_e4m3fn
        )
        cp_page = cp_loc // page
        cp_off = cp_loc % page
        hl.store(
            cache_keys,
            [cp_page[:, None], cp_off[:, None], cp_dim[None, :]],
            cp_q,
            extra_mask=cp_live[:, None],
        )
        hl.store(
            cache_scales,
            [cp_page[:, None], (32 * page + cp_off)[:, None]],
            cp_scale[:, None],
            extra_mask=cp_live[:, None],
        )

    # 6. wq_b | q_b: FP32 partials of its K chunks of q_lora
    for q_n, q_c, q_t in hl.tile([q_rows, b_chunks, rows], block_size=[128, 1, 16]):
        q_acc = hl.zeros([q_n, q_t], dtype=torch.float32)
        for q_k in hl.tile(b_k, block_size=128):
            q_acc = hl.dot(
                w_q[q_n, q_c.begin, q_k], q_lora_c[q_t, q_c.begin, q_k].T, acc=q_acc
            )
        q_parts[q_c.begin, q_t, q_n] = q_acc.T

    # 7. the indexer's queries per (token, 16 heads) and head weights
    for qp_t, qp_h in hl.tile([rows, iheads], block_size=[1, 16]):
        qp_tok = qp_t.begin
        qp_hm = hmat[:, :]  # first: it waits for nothing
        qp_q = torch.sum(q_parts_heads[:, qp_tok, qp_h, :], dim=0).to(torch.bfloat16)
        qp_g = torch.sum(ak_parts[:, qp_tok, a_rows + head_dim + qp_h.index], dim=0)
        qp_y = hl.dot(
            qp_q, qp_hm, out_dtype=torch.float32
        )  # [16 heads, 128] (H symmetric)
        # 0.0883883...: 128^-0.5, the Hadamard transform's normalization
        qp_y = (qp_y * 0.08838834764831845).to(torch.bfloat16).to(torch.float32)
        qp_amax = torch.clamp(torch.amax(torch.abs(qp_y), dim=1), min=1e-4)
        qp_scale = torch.exp2(torch.ceil(torch.log2(qp_amax * (1.0 / 448.0))))
        iq8[qp_tok, qp_h, :] = torch.clamp(qp_y / qp_scale[:, None], -448.0, 448.0).to(
            torch.float8_e4m3fn
        )
        # gate * heads^-0.5 (32^-0.5 = 0.1767766...) * q_scale * the softmax
        # scale (head_dim^-0.5 = 128^-0.5 = 0.0883883...)
        weights[qp_tok, qp_h] = (
            qp_g * 0.17677669529663687 * qp_scale * 0.08838834764831845
        )

    hl.pdl_launch_dependents()
    # 8. absorb per (head, 128 latent rows)
    for ab_h, ab_l, ab_t in hl.tile([heads, kv_lora, rows], block_size=[1, 128, 16]):
        ab_acc = hl.zeros([ab_l, ab_t], dtype=torch.float32)
        for ab_k in hl.tile(nope, block_size=64):
            ab_q = qb_parts[0, ab_t, iheads // 2 + ab_h.begin, ab_k]
            for ab_c in hl.static_range(1, b_chunks):
                ab_q = ab_q + qb_parts[ab_c, ab_t, iheads // 2 + ab_h.begin, ab_k]
            ab_acc = hl.dot(
                w_uk[ab_h.begin, ab_l, ab_k], ab_q.to(torch.bfloat16).T, acc=ab_acc
            )
        q8[ab_t, ab_h.begin, ab_l] = ab_acc.T.to(torch.bfloat16).to(torch.float8_e4m3fn)

    # 9. logits per pooled page
    for lg_w in hl.tile(logit_tasks, block_size=1):
        lg_phys = page_table[lg_w.begin]
        lg_keys = cache_keys[lg_phys, :page, :]
        lg_ks = cache_scales[lg_phys, 32 * page + hl.arange(page)]
        lg_pages = (pool_lens[rows - 1] + page - 1) // page
        lg_steps = (lg_pages - lg_w.begin + logit_tasks - 1) // logit_tasks
        lg_q = [
            iq8_rows[lg_g * 2 * iheads + hl.arange(2 * iheads), :]
            for lg_g in hl.static_range(row_groups)
        ]
        lg_wts = [
            weights[lg_g * 2 + hl.arange(2), :].reshape(2, 8, 4)
            for lg_g in hl.static_range(row_groups)
        ]
        _logits_page(lg_w.begin * page, lg_keys, lg_ks, lg_q, lg_wts, logits)
        for lg_i in hl.tile(1, lg_steps, block_size=1):
            lg_page = lg_w.begin + lg_i.begin * logit_tasks
            lg_phys_i = page_table[lg_page]
            _logits_page(
                lg_page * page,
                cache_keys[lg_phys_i, :page, :],
                cache_scales[lg_phys_i, 32 * page + hl.arange(page)],
                lg_q,
                lg_wts,
                logits,
            )


RING_LOADS = frozenset(["w_ak", "x_a", "w_q", "q_lora_c", "w_uk"])


def mk_config(bound) -> helion.Config:
    """The weights and their activation operands through the TMA ring; the dot roots
    warp-specialized. No tensor-memory prefetch: neither K loop takes one (q's
    [128, 1, 128] BF16 weight tiles have 256-byte rows, a tail 128-byte ones; the
    absorb's dot issues from the body)."""
    fields = ring_config_fields(bound, RING_LOADS)
    return helion.Config(
        indexing=fields["indexing"],
        pid_type="persistent_blocked",
        num_sm_multiplier=1,
        num_stages=3,
        num_warps=4,
        cross_loop_pipeline="dynamic",
        range_warp_specializes=fields["range_warp_specializes"],
    )


@functools.cache
def scratch(device: torch.device, tokens: int, width: int) -> dict[str, torch.Tensor]:
    """The kernel's scratch and outputs (q_lora, q8, iq8, weights, logits) on
    ``device``; ``width``: the page table's entries."""
    s, f32 = _setup.ATTN_SHAPE, torch.float32
    heads, dim, e4m3 = _setup.IDX_HEADS, _setup.IDX_DIM, torch.float8_e4m3fn

    def z(*shape: int, dtype=f32):
        return torch.zeros(*shape, device=device, dtype=dtype)

    return {
        "ak_parts": z(A_CHUNKS, LANES, s.q_lora + s.kv_lora + 384),
        "q_lora": z(tokens, s.q_lora, dtype=torch.bfloat16),
        "q_parts": z(B_CHUNKS, LANES, heads * dim + s.heads * s.nope),
        "q8": z(tokens, s.heads, s.kv_lora, dtype=e4m3),
        "iq8": z(LANES, heads, dim, dtype=e4m3),
        "weights": z(LANES, heads),
        "logits": z(LANES, width * _setup.POOL_PAGE),
    }


def prepared_weights(w: dict) -> dict:
    """The layer's weights as the kernel reads them (prepared once): SGLang's fused
    q_a | kv_a rows with the indexer's [wk | weights_proj | compress_gate | 0]
    (BF16; weights_proj's FP32 values are BF16-exact, as the checkpoint stores
    them) below them, 2432 rows (19 tiles of 128: one wave of 152 tasks), and
    wq_b | q_b (the indexer's first: its chain is the longer), in K chunks; W_UK,
    the norms, the pool compression's ape and the Hadamard matrix."""
    weights_proj = w["weights_proj"].to(torch.bfloat16)
    assert torch.equal(weights_proj.float(), w["weights_proj"]), (
        "weights_proj not BF16-exact"
    )
    w_ak = torch.cat([w["qkv_a"], w["wk"], weights_proj, w["compress_gate"]])
    w_ak = torch.cat([w_ak, w_ak.new_zeros(2432 - w_ak.size(0), w_ak.size(1))])
    w_q = torch.cat([w["wq_b"], w["q_b"]])
    return {
        "w_ak": w_ak.view(w_ak.size(0), A_CHUNKS, -1),
        "q_a_norm": w["q_a_norm"],
        "kv_a_norm": w["kv_a_norm"],
        "w_q": w_q.view(w_q.size(0), B_CHUNKS, -1),
        "w_uk": w["w_uk"],
        "k_norm_w": w["k_norm_w"].contiguous(),
        "k_norm_b": w["k_norm_b"].contiguous(),
        "ape": w["ape"].contiguous(),
        "hmat": _setup.hadamard_matrix(_setup.IDX_DIM, w_ak.device).to(torch.bfloat16),
    }


def kernel_args(x, pw, cache_loc, kv_cache, st) -> tuple:
    """The kernel's arguments: ``x`` the attention input [T, 4096] BF16 (contiguous),
    ``pw`` (``prepared_weights``), ``cache_loc`` the window's MLA cache slots [T]
    int64, ``kv_cache`` [slots, 512] E4M3, ``st`` the request's indexer state
    (``_setup.index_state``' names: the pooled index-K cache, its page table, the
    tail rings, the write plan, the rows' pool counts). Outputs in ``scratch``:
    q8 [T, 16, 512] E4M3 (the out MK's query) and logits [8, W * 64] FP32 (the
    pooled top-k's)."""
    tokens = x.shape[0]
    sc = scratch(x.device, tokens, st["page_table"].numel())
    keys, scales = cache_views(st["cache"])
    return (
        x.view(tokens, A_CHUNKS, -1),
        pw["w_ak"],
        pw["q_a_norm"],
        pw["kv_a_norm"],
        pw["w_q"],
        pw["w_uk"],
        pw["k_norm_w"],
        pw["k_norm_b"],
        pw["ape"],
        pw["hmat"],
        cache_loc,
        kv_cache,
        st["tail_k"],
        st["tail_s"],
        st["req"],
        st["write_start"],
        st["tail_start"],
        st["write_loc"].view(-1),
        keys,
        scales,
        st["page_table"],
        st["pool_lens"],
        sc["ak_parts"],
        sc["q_lora"],
        sc["q_parts"],
        sc["q8"],
        sc["iq8"],
        sc["weights"],
        sc["logits"],
    )


def compile_kernel(args: tuple):
    """``glmf_mla_frontidx_tp4`` compiled for ``args`` (once per binding)."""
    return _setup.compiled(glmf_mla_frontidx_tp4, mk_config, args)


# ---------- check and timing (torch.distributed.run, 4 ranks; no communication)

SLOTS = 16384  # the checks' MLA cache slots
# A served CUDA graph's padded page table (the longest context's pooled pages).
WIDE_PAGES = 793


def _inputs(layer: int, context: int, device, wide: int = 0) -> tuple:
    """Layer ``layer``'s weights (``glm5_3_tp4._setup.front_weights`` at
    ``_setup.ATTN_SHAPE`` and ``_setup.indexer_layer_weights``), an input window, a
    zero MLA cache with the window's slots, the indexer state after ``context``
    tokens (its page table padded to ``wide`` entries)."""
    w = front_weights(_setup.ATTN_SHAPE, seed=1000 * layer, device=device)
    w.update(_setup.indexer_layer_weights(seed=1000 * layer + 1, device=device))
    g = torch.Generator(device=device).manual_seed(50 + layer)
    x = torch.randn(_setup.T, _setup.HIDDEN, generator=g, device=device).to(
        torch.bfloat16
    )
    kv_cache = torch.zeros(
        SLOTS, _setup.ATTN_SHAPE.kv_lora, dtype=torch.float8_e4m3fn, device=device
    )
    cache_loc = torch.randperm(SLOTS, generator=g, device=device)[: _setup.T]
    state = _setup.index_state(context, seed=layer, device=device, wide_pages=wide)
    return w, x, kv_cache, cache_loc, state


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Every rank its own layer, after a short context and a long one (with a
    served graph's padded page table): against ``front_reference`` q_lora, the FP8
    queries and the window's cache rows; against ``_setup.indexer_reference`` the
    tail rings, the compressed pools' keys and scales, the indexer queries, the
    head weights and every row's logits over its pools (DeepGEMM's epilogue on
    the same FP8 inputs); then the step again from the restored state with NaN
    scratch: the same bits."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    T = _setup.T
    ok = True
    for context, wide in ((1002, 0), (9001, WIDE_PAGES)):
        w, x, kv_cache, cache_loc, st0 = _inputs(rank, context, device, wide)
        st = _setup.clone_state(st0)
        args = kernel_args(x, prepared_weights(w), cache_loc, kv_cache, st)
        kernel = compile_kernel(args)
        kernel(*args)
        torch.cuda.synchronize()
        q_lora, q8, kv8 = front_reference(x, w, None, None, _setup.ATTN_SHAPE)
        rs = _setup.clone_state(st0)
        iq8, weights, logits = _setup.indexer_reference(x, q_lora, w, rs)
        sc = scratch(device, T, st["page_table"].numel())
        keys, scales = cache_views(st["cache"])
        ref_keys, ref_scales = cache_views(rs["cache"])
        lens = [int(n) for n in st0["pool_lens"]]
        rows = []
        for label, got, expected, tol in (
            ("q_lora", sc["q_lora"], q_lora, 1e-2),
            ("q (FP8)", sc["q8"].float(), q8.float(), 3e-2),
            ("cache rows (FP8)", kv_cache[cache_loc].float(), kv8.float(), 3e-2),
            ("tail keys", st["tail_k"], rs["tail_k"], 4e-3),
            ("tail scores", st["tail_s"], rs["tail_s"], 4e-3),
            ("pool keys (FP8)", keys.float(), ref_keys.float(), 3e-2),
            ("pool scales", scales, ref_scales, 1e-2),
            ("indexer q (FP8)", sc["iq8"][:T].float(), iq8.float(), 3e-2),
            ("head weights", sc["weights"][:T], weights, 2e-2),
            (
                "logits",
                torch.cat([sc["logits"][i, :n] for i, n in enumerate(lens)]),
                torch.cat([logits[i, :n] for i, n in enumerate(lens)]),
                3e-2,
            ),
        ):
            ok &= report(rows, label, got, expected, tol)
        outputs = [
            sc["q_lora"],
            sc["q8"],
            kv_cache,
            st["tail_k"],
            st["tail_s"],
            st["cache"],
            sc["iq8"][:T],
            sc["weights"][:T],
            sc["logits"],
        ]
        first = [o.clone() for o in outputs]
        for k in ("tail_k", "tail_s", "cache"):
            st[k].copy_(st0[k])
        kv_cache.zero_()
        _setup.poison(sc.values())
        kernel(*args)
        torch.cuda.synchronize()
        # (logits past a row's pools are not written)
        first[-1], outputs[-1] = (
            [o[i, :n] for i, n in enumerate(lens)] for o in (first[-1], outputs[-1])
        )
        repeat = all(starmap(same, zip(outputs[:-1], first[:-1], strict=True)))
        repeat &= all(starmap(same, zip(outputs[-1], first[-1], strict=True)))
        rows.append(
            f"  {'NaN scratch repeat':<22} {'bitwise' if repeat else 'DIFFERS  <-- FAIL'}"
        )
        ok &= repeat
        if verbose or not ok:
            say(f"rank{rank} context {context}\n" + "\n".join(rows))
    verdict(ok, group, "glmf_mla_frontidx_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing: ``LAYERS`` distinct layers (default 8) back to back,
    ``REPS`` times (default 4) in one CUDA graph, ``CONTEXT`` committed tokens
    (default 8192) with a served graph's padded page table."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    layers = int(os.environ.get("LAYERS", "8"))
    reps = int(os.environ.get("REPS", "4"))
    context = int(os.environ.get("CONTEXT", "8192"))
    args = []
    for i in range(layers):
        w, x, kv_cache, cache_loc, st = _inputs(i, context, device, WIDE_PAGES)
        args.append(kernel_args(x, prepared_weights(w), cache_loc, kv_cache, st))
    kernel = compile_kernel(args[0])

    def step() -> None:
        for a in args:
            kernel(*a)

    median, minimum = graph_time(step, layers, reps)
    if verbose:
        say(
            f"rank{rank} T={_setup.T} context={context} us/launch med {median:.2f} "
            f"min {minimum:.2f}"
        )
    return {_setup.T: {"latency_us": median, "min_latency_us": minimum}}


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The check on every rank, then the cold-weight timing (``check_only``: the
    check alone)."""
    return run_main(
        "glmf_mla_frontidx_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
