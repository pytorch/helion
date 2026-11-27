# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""GLM-5.3 (GlmMoeDsa) DSA indexer for a speculative-verify window (one
request, T <= 8 rows), from the layer input and q_lora to every row's logits
over its keys, as one persistent Gluon megakernel; the top-2048 selection stays
SGLang's (``topk_transform_paged_v2`` reads the logits). Replicated on every
rank (SGLang's indexer weights are not tensor-parallel).

The roots (SGLang's ``Indexer._fused_q_prepare_and_store`` + DeepGEMM's
``fp8_paged_mqa_logits``):

- kw: [wk | weights_proj] (BF16 [160 (+32 zero), 6144]) as FP32 partials of 8 K
  chunks; summed in chunk order and rounded to BF16 (the GEMM output).
- q: wq_b (BF16 [32 * 128, 2048]) as FP32 partials of 4 K chunks.
- key, per token: the key's LayerNorm (FP32), interleaved RoPE of its leading
  64 dims, rounded to BF16, one FP8 scale (max(1e-4, absmax) / 448), stored in
  the paged index-K cache at the window's slot.
- query, per token: RoPE of each head's leading 64 dims, an FP8 scale per head;
  the head weights gate * softmax_scale * heads^-0.5 * q_scale.
- logits, per page of 64 keys (the window's own keys from the key root), a
  fixed number of tasks each looping over every logit_tasks-th page up to the
  last row's key count (the page table may be a CUDA graph's: the max
  context's pages): the FP8 scores of the page's keys against each 2-row
  group's heads (one MMA a group), then
  DeepGEMM's FP32 epilogue in its order: relu as (s + |s|), four FMA chains over
  the heads (h = 4 k + m, k ascending), (c0 + c2) + (c1 + c3), halved, times the
  key's scale.

Fed the kernel's FP8 queries, head weights and cache, DeepGEMM's
``fp8_paged_mqa_logits`` returns the same logits bit for bit. The split-K
partials are summed in a fixed order (no atomics): the outputs are run-to-run
deterministic.

``correctness_check()`` compares with the torch reference (``_setup``) at the
served window, each rank on its own copy; ``main()`` runs it, then times the
kernel with cold weights::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \\
      python -m torch.distributed.run --standalone --nproc-per-node=4 \\
      pretuned_kernels/megakernels/glm5_3_tp4/glm5_3_indexer_tp4.py [--check]
"""

from __future__ import annotations

import os
import sys
from typing import Any

from pretuned_kernels.megakernels._qwen3_5_tp4 import graph_time
from pretuned_kernels.megakernels._qwen3_5_tp4 import report
from pretuned_kernels.megakernels._qwen3_5_tp4 import run_main
from pretuned_kernels.megakernels._qwen3_5_tp4 import say
from pretuned_kernels.megakernels._qwen3_5_tp4 import verdict
from pretuned_kernels.megakernels.glm5_3_tp4 import _setup
from pretuned_kernels.megakernels.glm5_3_tp4.glm5_3_attn_front_tp4 import (
    rope_interleaved,
)
import torch
import torch.distributed as dist

import helion
import helion.language as hl

RING_LOADS = ("kw_w", "x_chunked", "qb_w", "q_lora_chunked")

CONFIG = {
    "block_sizes": [16, 128, 16, 128],
    "num_warps": 4,
    "pid_type": "persistent_blocked",
    "cross_loop_pipeline": "dynamic",
    "num_sm_multiplier": 1,
    "num_stages": 4,
    # The GEMMs' operands on the TMA ring (ring_config_fields(bound,
    # RING_LOADS)); 55 memory ops at 6 rows (three 2-row logits groups).
    "indexing": [
        "tensor_descriptor" if op in (0, 1, 3, 4) else "pointer" for op in range(55)
    ],
    "range_warp_specializes": [True, None, True, None, None, None, None, None],
}


def page_logits(pos, keys, kscale, row_q, row_w, logits):
    """The logits of a page of 64 keys at positions ``pos`` (``keys`` FP8, scales
    ``kscale``) into ``logits``: each 2-row group's FP8 scores and DeepGEMM's
    epilogue (relu as s + |s|, four FMA chains over h = 4 k + m, (c0 + c2) +
    (c1 + c3), halved, times the key's scale). Flash's pooled indexer
    (``glm5_3_flash_tp4.mla_frontidx_tp4``) shares it."""
    page = 64
    kk = hl.arange(8)
    for g in hl.static_range(len(row_q)):
        group_rows = g * 2 + hl.arange(2)
        s = hl.dot(keys, row_q[g].T, out_dtype=torch.float32)
        r = (s + torch.abs(s)).reshape(page, 2, 8, 4)
        c = hl.zeros([page, 2, 4], dtype=torch.float32)
        for k in hl.static_range(8):
            rk = torch.sum(torch.where(kk[None, None, :, None] == k, r, 0.0), dim=2)
            wk = torch.sum(torch.where(kk[None, :, None] == k, row_w[g], 0.0), dim=1)
            c = c + rk * wk[None, :, :]
        pair = torch.sum(c.reshape(page, 2, 2, 2), dim=2)
        out = torch.sum(pair, dim=2) * 0.5 * kscale[:, None]
        hl.store(logits, [group_rows[:, None], pos[None, :]], out.T)


def _logits_page(
    start,
    phys,
    keys,
    kscale,
    first,
    rows,
    k_new,
    k_new_scale,
    cache_keys,
    cache_scales,
    row_q,
    row_w,
    logits,
):
    """One page of 64 keys from key ``start`` (cache page ``phys``, its keys and
    scales loaded): the window's keys merged in (and stored into the cache),
    then their logits (``page_logits``)."""
    page = 64
    pos = start + hl.arange(page)
    new = (pos >= first) & (pos < first + rows)
    idx = torch.clamp(pos - first, 0, rows - 1)
    # (A helper's accesses go through hl.load / hl.store: only the kernel body's
    # subscripts become device accesses.)
    keys = torch.where(
        new[:, None], hl.load(k_new, [idx[:, None], hl.arange(128)[None, :]]), keys
    )
    kscale = torch.where(new, hl.load(k_new_scale, [idx]), kscale)
    # The window's keys into the cache (a masked store: rows of the window
    # only).
    phys_rows = phys + hl.zeros([page], dtype=torch.int32)
    hl.store(
        cache_keys,
        [phys_rows[:, None], hl.arange(page)[:, None], hl.arange(128)[None, :]],
        keys,
        extra_mask=new[:, None],
    )
    hl.store(
        cache_scales, [phys_rows, 32 * page + hl.arange(page)], kscale, extra_mask=new
    )
    page_logits(pos, keys, kscale, row_q, row_w, logits)


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    config=helion.Config(**CONFIG),
    ignore_warnings=[helion.exc.TensorOperationInWrapper, helion.exc.FastMathUnused],
)
def glm5_3_indexer_tp4(
    x,  # [T, 6144] bf16: the layer's input RMSNorm output
    q_lora,  # [T, 2048] bf16
    wkw,  # [192, 6144] bf16: wk (128) | weights_proj (32) | zero rows (32)
    wq_b,  # [4096, 2048] bf16: per head 128 rows (rope 64 | nope 64)
    k_norm_weight,  # [128] f32
    k_norm_bias,  # [128] f32
    cos_sin,  # [max_position, 64] f32: cos (32) | sin (32)
    positions,  # [T] int64
    cache_keys,  # [pages, 66, 128] e4m3: the index-K cache's keys (rows 0-63)
    cache_scales,  # [pages, 2112] f32: the same cache's key scales (2048 + j)
    page_table,  # [W] int32: the request's pages
    lengths,  # [T] int32: row i's key count (context + window up to row i)
    weight_scale: float,  # softmax_scale * heads ** -0.5
):
    hl.specialize(x.size())
    hl.specialize(x.stride())
    hl.specialize(q_lora.size())
    hl.specialize(q_lora.stride())
    hl.specialize(wkw.size())
    hl.specialize(wkw.stride())
    hl.specialize(wq_b.size())
    hl.specialize(wq_b.stride())
    hl.specialize(k_norm_weight.size())
    hl.specialize(k_norm_bias.size())
    hl.specialize(cos_sin.stride())
    hl.specialize(positions.size())
    hl.specialize(cache_keys.stride())
    hl.specialize(cache_scales.stride())
    hl.specialize(page_table.size())
    hl.specialize(lengths.size())

    rows, hidden = x.shape
    q_lora_n = q_lora.size(1)
    kw_rows = wkw.size(0)
    head_dim = k_norm_weight.size(0)
    heads = wq_b.size(0) // head_dim
    pages_w = page_table.size(0)
    assert rows <= 8 and heads == 32 and head_dim == 128 and kw_rows == 192
    dev = x.device
    # Literals: a module-level int reaches the kernel as a runtime scalar.
    page = 64
    eps = 1e-6
    a_chunks = 8
    a_chunk_k = hidden // a_chunks
    x_chunked = x.view(rows, a_chunks, a_chunk_k)
    kw_w = wkw.view(kw_rows, a_chunks, a_chunk_k)
    kw_parts = torch.empty((a_chunks, rows, kw_rows), dtype=torch.float32, device=dev)
    b_chunks = 4
    b_chunk_k = q_lora_n // b_chunks
    q_lora_chunked = q_lora.view(rows, b_chunks, b_chunk_k)
    qb_w = wq_b.view(heads * head_dim, b_chunks, b_chunk_k)
    q_parts = torch.empty(
        (b_chunks, rows, heads * head_dim), dtype=torch.float32, device=dev
    )
    q_parts_heads = q_parts.view(b_chunks, rows, heads, head_dim)
    k_new = torch.empty((rows, head_dim), dtype=torch.float8_e4m3fn, device=dev)
    k_new_scale = torch.empty((rows,), dtype=torch.float32, device=dev)
    # The queries, head weights and logits in 8 rows, the logits in groups of 2
    # rows (a [keys, 2 * heads] MMA and its epilogue each: power-of-two shapes
    # whatever the row count); rows past ``rows`` are never written nor read
    # back (callers take the first rows). (Arithmetic on the specialized rows.)
    row_groups = (rows + 1) // 2
    rows_pad = 8
    q8 = torch.empty((rows_pad, heads, head_dim), dtype=torch.float8_e4m3fn, device=dev)
    q8_rows = q8.view(rows_pad * heads, head_dim)
    weights = torch.empty((rows_pad, heads), dtype=torch.float32, device=dev)
    logits = torch.empty((rows_pad, pages_w * page), dtype=torch.float32, device=dev)
    # Logits tasks: each takes the pages s, s + logit_tasks, ... of the rows'
    # keys (the page table may be far wider, e.g. a CUDA graph's max context).
    # 148: at most one per worker of a B200 (148 SMs; a GB300 runs 152 workers,
    # 4 of them without one). Measured against 128 tasks (an earlier variant,
    # with other changes), not against 152.
    logit_tasks = min(pages_w, 148)

    hl.pdl_wait()
    hl.prefetch_code()

    # kw: FP32 partials of the K chunks.
    for kw_n, kw_c, kw_t in hl.tile(
        [kw_rows, a_chunks, rows], block_size=[64, 1, None]
    ):
        kw_acc = hl.zeros([kw_n, kw_t], dtype=torch.float32)
        for kw_k in hl.tile(a_chunk_k):
            kw_acc = hl.dot(
                kw_w[kw_n, kw_c.begin, kw_k],
                x_chunked[kw_t, kw_c.begin, kw_k].T,
                acc=kw_acc,
            )
        kw_parts[kw_c.begin, kw_t, kw_n] = kw_acc.T

    # q: FP32 partials of the K chunks.
    for qb_n, qb_c, qb_t in hl.tile(
        [heads * head_dim, b_chunks, rows], block_size=[128, 1, None]
    ):
        qb_acc = hl.zeros([qb_n, qb_t], dtype=torch.float32)
        for qb_k in hl.tile(b_chunk_k):
            qb_acc = hl.dot(
                qb_w[qb_n, qb_c.begin, qb_k],
                q_lora_chunked[qb_t, qb_c.begin, qb_k].T,
                acc=qb_acc,
            )
        q_parts[qb_c.begin, qb_t, qb_n] = qb_acc.T

    # The window's keys: LayerNorm, RoPE, BF16, FP8 (into the cache with the
    # logits of their page).
    for kp_t in hl.tile(rows, block_size=1):
        kp_token = kp_t.begin
        kp_pos = positions[kp_token]
        kp_cos = cos_sin[kp_pos, :32]
        kp_sin = cos_sin[kp_pos, 32:]
        kp_lo = torch.sum(kw_parts[:, kp_token, :64], dim=0)
        kp_hi = torch.sum(kw_parts[:, kp_token, 64:128], dim=0)
        kp_lo = kp_lo.to(torch.bfloat16).to(torch.float32)
        kp_hi = kp_hi.to(torch.bfloat16).to(torch.float32)
        kp_mean = (torch.sum(kp_lo) + torch.sum(kp_hi)) / head_dim
        kp_dlo = kp_lo - kp_mean
        kp_dhi = kp_hi - kp_mean
        kp_inv = torch.rsqrt(
            (torch.sum(kp_dlo * kp_dlo) + torch.sum(kp_dhi * kp_dhi)) / head_dim + eps
        )
        kp_lo = kp_dlo * kp_inv * k_norm_weight[:64] + k_norm_bias[:64]
        kp_hi = kp_dhi * kp_inv * k_norm_weight[64:] + k_norm_bias[64:]
        kp_lo = rope_interleaved(
            kp_lo[None, :], kp_cos[None, :], kp_sin[None, :], sin_first=True
        ).reshape(64)
        kp_lo = kp_lo.to(torch.bfloat16).to(torch.float32)
        kp_hi = kp_hi.to(torch.bfloat16).to(torch.float32)
        kp_max = torch.maximum(
            torch.amax(torch.abs(kp_lo)), torch.amax(torch.abs(kp_hi))
        )
        kp_scale = torch.clamp(kp_max, min=1e-4) / 448.0
        kp_inv_scale = 1.0 / kp_scale
        k_new[kp_token, :64] = torch.clamp(kp_lo * kp_inv_scale, -448.0, 448.0).to(
            torch.float8_e4m3fn
        )
        k_new[kp_token, 64:] = torch.clamp(kp_hi * kp_inv_scale, -448.0, 448.0).to(
            torch.float8_e4m3fn
        )
        k_new_scale[kp_t] = kp_scale + hl.zeros([kp_t], dtype=torch.float32)

    # The queries per (token, 8 heads): RoPE, FP8 per head; the head weights.
    for qp_t, qp_h in hl.tile([rows, heads], block_size=[1, 8]):
        qp_token = qp_t.begin
        qp_pos = positions[qp_token]
        qp_cos = cos_sin[qp_pos, :32]
        qp_sin = cos_sin[qp_pos, 32:]
        qp_lo = torch.sum(q_parts_heads[:, qp_token, qp_h, :64], dim=0)
        qp_hi = torch.sum(q_parts_heads[:, qp_token, qp_h, 64:], dim=0)
        qp_gate = torch.sum(kw_parts[:, qp_token, head_dim + qp_h.index], dim=0)
        qp_lo = rope_interleaved(
            qp_lo.to(torch.bfloat16).to(torch.float32),
            qp_cos[None, :],
            qp_sin[None, :],
            sin_first=True,
        )
        qp_hi = qp_hi.to(torch.bfloat16).to(torch.float32)
        qp_max = torch.maximum(
            torch.amax(torch.abs(qp_lo), dim=1), torch.amax(torch.abs(qp_hi), dim=1)
        )
        qp_scale = torch.clamp(qp_max, min=1e-4) / 448.0
        qp_inv_scale = 1.0 / qp_scale
        q8[qp_token, qp_h, :64] = torch.clamp(
            qp_lo * qp_inv_scale[:, None], -448.0, 448.0
        ).to(torch.float8_e4m3fn)
        q8[qp_token, qp_h, 64:] = torch.clamp(
            qp_hi * qp_inv_scale[:, None], -448.0, 448.0
        ).to(torch.float8_e4m3fn)
        weights[qp_token, qp_h] = (
            qp_gate.to(torch.bfloat16).to(torch.float32) * weight_scale * qp_scale
        )

    hl.pdl_launch_dependents()
    # Logits per page: the page's keys (the window's from the key root, stored
    # into the cache) against every row's heads, DeepGEMM's epilogue. Rows see
    # increasing key counts: the last row's bounds the pages, the window's keys
    # start at the first row's count - 1. A task takes the pages w, w +
    # logit_tasks, ...
    for lg_w in hl.tile(logit_tasks, block_size=1):
        # The task's first page (w < the table's width) ahead of its waits; a
        # task past the rows' pages computes it anyway: logits past a row's
        # key count are garbage the top-k never reads, and none of its keys is
        # the window's, so it stores nothing into the cache.
        lg_phys = page_table[lg_w.begin]
        lg_keys = cache_keys[lg_phys, :page, :]
        lg_kscale = cache_scales[lg_phys, 32 * page + hl.arange(page)]
        lg_len = lengths[rows - 1]
        lg_first = lengths[0] - 1
        lg_pages = (lg_len + page - 1) // page
        lg_steps = (lg_pages - lg_w.begin + logit_tasks - 1) // logit_tasks
        # Every row group's queries and head weights, once a task (after the
        # waits).
        lg_q = [
            q8_rows[lg_g * 2 * heads + hl.arange(2 * heads), :]
            for lg_g in hl.static_range(row_groups)
        ]
        lg_wts = [
            weights[lg_g * 2 + hl.arange(2), :].reshape(2, 8, 4)
            for lg_g in hl.static_range(row_groups)
        ]
        _logits_page(
            lg_w.begin * page,
            lg_phys,
            lg_keys,
            lg_kscale,
            lg_first,
            rows,
            k_new,
            k_new_scale,
            cache_keys,
            cache_scales,
            lg_q,
            lg_wts,
            logits,
        )
        for lg_i in hl.tile(1, lg_steps, block_size=1):
            lg_page = lg_w.begin + lg_i.begin * logit_tasks
            lg_phys_i = page_table[lg_page]
            _logits_page(
                lg_page * page,
                lg_phys_i,
                cache_keys[lg_phys_i, :page, :],
                cache_scales[lg_phys_i, 32 * page + hl.arange(page)],
                lg_first,
                rows,
                k_new,
                k_new_scale,
                cache_keys,
                cache_scales,
                lg_q,
                lg_wts,
                logits,
            )
    return logits, q8, weights, k_new, k_new_scale


# The served verify window (EAGLE 5/1/6; the MTP draft extend's is 6 too).
TOKENS = (6,)


def _args(inp: dict, w: dict):
    keys, scales = _setup.cache_views(inp["cache"])
    return (
        inp["x"],
        inp["q_lora"],
        w["wkw"],
        w["wq_b"],
        w["k_norm_weight"],
        w["k_norm_bias"],
        inp["cos_sin"],
        inp["positions"],
        keys,
        scales,
        inp["page_table"],
        inp["lengths"],
        _setup.INDEXER_WEIGHT_SCALE,
    )


# A served CUDA graph's page-table width (SGLang's DSA real_page_table: the max
# context's 64-key pages).
WIDE_PAGES = 3170


def _wide(inp: dict, width: int) -> dict:
    """``inp`` with its page table padded to ``width`` entries (page 0 past the
    request's pages, as a graph's stale rows)."""
    if not width:
        return inp
    pt = inp["page_table"]
    pad = torch.zeros(width - pt.numel(), dtype=pt.dtype, device=pt.device)
    return dict(inp, page_table=torch.cat([pt, pad]))


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def correctness_check(group: dist.ProcessGroup, verbose: bool = True) -> None:
    """Against ``_setup.indexer_reference`` at the served window, for a short
    context (one page short of the top-k) and a long one: the window's FP8 keys
    (and their cache slots), the FP8 queries and head weights within 2e-2
    relative RMS, each row's logits over its keys within 1e-2, and a second
    launch bitwise equal; the long context also with the page table padded to a
    CUDA graph's width (the served one's: the max context's pages)."""
    rank = group.rank()
    w = _setup.indexer_weights(seed=5)
    ok = True
    for tokens in TOKENS:
        for context, width in ((1980, 0), (8000, 0), (8000, WIDE_PAGES)):
            inp = _wide(
                _setup.indexer_inputs(tokens, context, seed=11 + context), width
            )
            ref = _setup.indexer_reference(inp, w)
            out = glm5_3_indexer_tp4(*_args(inp, w))
            torch.cuda.synchronize()
            rows = []
            ok &= report(
                rows,
                "keys (FP8)",
                out[3].float() * out[4][:, None],
                ref["k8"].float() * ref["k_scale"][:, None],
                2e-2,
            )
            keys, _ = _setup.cache_views(inp["cache"])
            slot = inp["cache_slot"]
            ok &= report(
                rows,
                "cache rows",
                keys[slot // 64, slot % 64].float(),
                out[3].float(),
                1e-6,
            )
            ok &= report(
                rows, "queries (FP8)", out[1][:tokens].float(), ref["q8"].float(), 2e-2
            )
            ok &= report(rows, "head weights", out[2][:tokens], ref["weights"], 2e-2)
            # Each row's logits over its own keys (the reference covers the
            # request's pages only).
            valid = [int(n) for n in inp["lengths"]]
            got = torch.cat([out[0][i, :n] for i, n in enumerate(valid)])
            exp = torch.cat([ref["logits"][i, :n] for i, n in enumerate(valid)])
            ok &= report(rows, "logits", got, exp, 1e-2)
            again = glm5_3_indexer_tp4(*_args(inp, w))
            same = all(
                torch.equal(_bits(a), _bits(b))
                for a, b in zip(
                    (out[1][:tokens], out[2][:tokens], out[3], out[4]),
                    (again[1][:tokens], again[2][:tokens], again[3], again[4]),
                    strict=True,
                )
            ) and all(
                torch.equal(again[0][i, :n], out[0][i, :n]) for i, n in enumerate(valid)
            )
            rows.append(
                f"  {'determinism':<22} {'bitwise' if same else 'DIFFERS  <-- FAIL'}"
            )
            ok &= same
            if verbose or not ok:
                say(
                    f"rank{rank} T={tokens} context {context} page table "
                    f"{inp['page_table'].numel()}\n" + "\n".join(rows)
                )
    verdict(ok, group, "glm5_3_indexer_tp4")


def benchmark(group: dist.ProcessGroup, verbose: bool = True) -> dict[str, Any]:
    """Cold-weight timing at the served window: ``LAYERS`` weight copies (default
    8) back to back, ``REPS`` times (default 4) in one CUDA graph; ``CONTEXT``
    keys (8192), the page table a served graph's width."""
    rank = group.rank()
    layers = int(os.environ.get("LAYERS", "8"))
    reps = int(os.environ.get("REPS", "4"))
    context = int(os.environ.get("CONTEXT", "8192"))
    results = {}
    for tokens in TOKENS:
        args = [
            _args(
                _wide(_setup.indexer_inputs(tokens, context, seed=i), WIDE_PAGES),
                _setup.indexer_weights(seed=100 + i),
            )
            for i in range(layers)
        ]

        def step(args=args) -> None:
            for a in args:
                glm5_3_indexer_tp4(*a)

        median, minimum = graph_time(step, layers, reps)
        if verbose:
            say(
                f"rank{rank} T={tokens} context={context} us/launch med {median:.2f} "
                f"min {minimum:.2f}"
            )
        results[tokens] = {"latency_us": median, "min_latency_us": minimum}
    return results


def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """The check on every rank, then the cold-weight timing (``check_only``:
    the check alone)."""
    return run_main(
        "glm5_3_indexer_tp4", correctness_check, benchmark, verbose, check_only
    )


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
