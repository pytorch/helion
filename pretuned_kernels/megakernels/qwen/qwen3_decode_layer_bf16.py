# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""Qwen3-8B batch-one BF16 decode-layer megakernel, pretuned for NVIDIA B200.

The BF16 sibling of ``qwen3_decode_layer``: the same single Helion function
(residual RMSNorm, QKV projection, Q/K norm and RoPE, KV-cache update, split
paged attention and merge, output projection, and the complete gated FFN) with
BF16 weights for all four projections and no activation quantization.  This is
the precision Mirage's Qwen3-8B persistent megakernel uses.  The benchmark
checks the result against vLLM's compiled BF16 ``Qwen3DecoderLayer`` with its
default backend selection.

Split attention runs ahead of the QKV projection where it can: splits are
enumerated slowest and in reverse, so the (few) splits inside the context are
each program's last task and the tasks past the context run while QKV still
streams; an active split loads its block-table entries, K rows, position and
cos/sin and L2-prefetches its V rows before it first touches QKV's output.
Three prefetch-only loops (as in the H100 layer) pull weights into L2 across
hand-offs where HBM would idle: the first o_proj columns while attention
finishes, each program's first gate_up tile, and the first down_proj columns
across the SiLU hand-off.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

import torch

import helion
import helion.language as hl

BATCH = 1
HIDDEN = 4096
INTERMEDIATE = 12288
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
CONTEXT = 8192
# Mirage's OSDI'26 setup decodes 1024 tokens after a 64-token prompt, so its
# per-step contexts span 64..1088; 576 and 1088 are the middle and the end.
# 1001/4093/8191 are the FP8 sibling's contexts (batch one only).
CONTEXT_LENGTHS = (576, 1088)
EXTRA_CONTEXT_LENGTHS = (1001, 4093, 8191)
# Benchmarked at batch 1; the correctness check covers every shipped batch.
BATCHES = (1, 2, 4)
BENCHMARK_BATCHES = (1,)
CACHE_BLOCK = 16
ATTENTION_SPLITS = 256
GROUP = 128
EPS = 1e-6
# L2 prefetch extents (hints only): the first columns of o_proj and down_proj.
PREFETCH_O_COLS = 2048
PREFETCH_W2_COLS = 1024


@helion.aot_kernel(
    static_shapes=False,
    backend="triton",
    triton_do_not_specialize=False,
)
def qwen3_decode_layer_bf16(
    hidden_states,
    residual,
    pre_weight,
    pre_normed,
    qkv_weight,
    q_weight,
    k_weight,
    cos_sin,
    position,
    kv_cache,
    block_table,
    slot_mapping,
    o_weight,
    post_weight,
    ffn_normed,
    w13,
    w2,
    context_lens,
    hidden,
    intermediate,
    q_heads,
    kv_heads,
    head_dim,
    context,
    cache_block,
    attention_splits,
    group,
    eps,
):
    # Same physical-capacity contract as the FP8 Qwen3 layer: one B1/Q1
    # bucket with runtime context lengths, positions, block tables, slot
    # mappings, and KV data.  All four projections read BF16 weights.
    batch_capacity = hidden_states.size(0)
    hl.specialize(batch_capacity)
    hl.specialize(hidden_states.size(0))
    torch._check(hidden_states.size(0) == batch_capacity)
    hl.specialize(residual.size(0))
    torch._check(residual.size(0) == batch_capacity)
    hl.specialize(pre_normed.size(0))
    torch._check(pre_normed.size(0) == batch_capacity)
    hl.specialize(position.size(0))
    torch._check(position.size(0) == batch_capacity)
    hl.specialize(block_table.size(0))
    torch._check(block_table.size(0) == batch_capacity)
    hl.specialize(slot_mapping.size(0))
    torch._check(slot_mapping.size(0) == batch_capacity)
    hl.specialize(ffn_normed.size(0))
    torch._check(ffn_normed.size(0) == batch_capacity)
    hl.specialize(context_lens.size(0))
    torch._check(context_lens.size(0) == batch_capacity)
    hl.specialize(hidden_states.size(1))
    hl.specialize(residual.size(1))
    hl.specialize(pre_weight.size(0))
    hl.specialize(pre_normed.size(1))
    hl.specialize(qkv_weight.size(0))
    hl.specialize(qkv_weight.size(1))
    hl.specialize(q_weight.size(0))
    hl.specialize(k_weight.size(0))
    hl.specialize(cos_sin.size(0))
    hl.specialize(cos_sin.size(1))
    hl.specialize(kv_cache.size(0))
    hl.specialize(kv_cache.size(1))
    hl.specialize(kv_cache.size(2))
    hl.specialize(kv_cache.size(3))
    hl.specialize(block_table.size(1))
    hl.specialize(o_weight.size(0))
    hl.specialize(o_weight.size(1))
    hl.specialize(post_weight.size(0))
    hl.specialize(ffn_normed.size(1))
    hl.specialize(w13.size(0))
    hl.specialize(w13.size(1))
    hl.specialize(w2.size(0))
    hl.specialize(w2.size(1))
    hl.specialize(hidden_states.stride(0))
    hl.specialize(hidden_states.stride(1))
    hl.specialize(residual.stride(0))
    hl.specialize(residual.stride(1))
    hl.specialize(pre_weight.stride(0))
    hl.specialize(pre_normed.stride(0))
    hl.specialize(pre_normed.stride(1))
    hl.specialize(qkv_weight.stride(0))
    hl.specialize(qkv_weight.stride(1))
    hl.specialize(q_weight.stride(0))
    hl.specialize(k_weight.stride(0))
    hl.specialize(cos_sin.stride(0))
    hl.specialize(cos_sin.stride(1))
    hl.specialize(position.stride(0))
    hl.specialize(kv_cache.stride(0))
    hl.specialize(kv_cache.stride(1))
    hl.specialize(kv_cache.stride(2))
    hl.specialize(kv_cache.stride(3))
    hl.specialize(block_table.stride(0))
    hl.specialize(block_table.stride(1))
    hl.specialize(slot_mapping.stride(0))
    hl.specialize(o_weight.stride(0))
    hl.specialize(o_weight.stride(1))
    hl.specialize(post_weight.stride(0))
    hl.specialize(ffn_normed.stride(0))
    hl.specialize(ffn_normed.stride(1))
    hl.specialize(w13.stride(0))
    hl.specialize(w13.stride(1))
    hl.specialize(w2.stride(0))
    hl.specialize(w2.stride(1))
    hl.specialize(context_lens.stride(0))
    torch._check(batch_capacity >= 1)

    pre_input = hidden_states
    pre_residual = residual
    pre_norm_weight = pre_weight
    pre_epsilon = eps
    pre_group_size = group
    _, pre_hidden_size = pre_input.shape
    pre_num_tokens = batch_capacity
    hl.specialize(pre_hidden_size)
    hl.specialize(pre_group_size)
    pre_groups_per_row = pre_hidden_size // pre_group_size
    hl.specialize(pre_groups_per_row)
    pre_rms_partials = torch.empty(
        (pre_num_tokens, pre_groups_per_row),
        dtype=torch.float32,
        device=pre_input.device,
    )
    qkv_mm_k = pre_hidden_size
    qkv_mm_m = batch_capacity
    qkv_mm_n = qkv_weight.size(0)
    qkv = torch.empty(
        (qkv_mm_m, qkv_mm_n),
        dtype=torch.bfloat16,
        device=pre_input.device,
    )
    batch = batch_capacity
    query = qkv[:, : q_heads * head_dim].view(batch, q_heads, head_dim)
    key_begin = q_heads * head_dim
    key = qkv[:, key_begin : key_begin + kv_heads * head_dim].view(
        batch, kv_heads, head_dim
    )
    value = qkv[
        :, key_begin + kv_heads * head_dim : (q_heads + 2 * kv_heads) * head_dim
    ].view(batch, kv_heads, head_dim)
    # Q/K norm and neox RoPE run inside the attention root, whose split holding
    # the new token also writes its KV-cache row.
    qk_head_dim = head_dim
    qk_eps = eps
    qk_q_weight = q_weight
    qk_k_weight = k_weight
    qk_cos_sin_cache = cos_sin
    qk_position_ids = position
    hl.specialize(qkv.shape[1])
    qk_rotary_dim = qk_cos_sin_cache.shape[1]
    hl.specialize(qk_rotary_dim)
    qk_embed_dim = qk_rotary_dim // 2
    hl.specialize(qk_head_dim)
    cache_key = key
    cache_value = value
    cache_kv_cache = kv_cache
    cache_slot_mapping = slot_mapping
    cache_block_size = cache_block
    _, cache_num_kv_heads, cache_head_dim = cache_key.shape
    hl.specialize(cache_num_kv_heads)
    hl.specialize(cache_head_dim)
    hl.specialize(cache_block_size)
    attention_split_query = query
    attention_split_kv_cache = kv_cache
    attention_split_block_table = block_table
    attention_split_context = context
    attention_split_block_size = cache_block
    attention_split_q_per_kv = q_heads // kv_heads
    attention_split_splits = attention_splits
    (
        _,
        attention_split_num_q_heads,
        attention_split_head_dim,
    ) = attention_split_query.shape
    attention_split_num_tokens = batch_capacity
    attention_split_num_kv_heads = attention_split_kv_cache.shape[2]
    assert (
        attention_split_num_q_heads
        == attention_split_num_kv_heads * attention_split_q_per_kv
    )
    assert attention_split_context % attention_split_splits == 0
    hl.specialize(attention_split_head_dim)
    hl.specialize(attention_split_num_kv_heads)
    hl.specialize(attention_split_q_per_kv)
    hl.specialize(attention_split_context)
    hl.specialize(attention_split_block_size)
    hl.specialize(attention_split_splits)
    assert attention_split_context // attention_split_splits <= 64
    attention_split_split_context = attention_split_context // attention_split_splits
    attention_split_token_kv_heads = (
        attention_split_num_tokens * attention_split_num_kv_heads
    )
    partial_out = torch.empty(
        (
            attention_split_splits,
            attention_split_token_kv_heads,
            attention_split_q_per_kv,
            attention_split_head_dim,
        ),
        device=attention_split_query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (
            attention_split_splits,
            attention_split_token_kv_heads,
            attention_split_q_per_kv,
        ),
        device=attention_split_query.device,
        dtype=torch.float32,
    )
    attention_split_qk_scale = 1.0 / math.sqrt(attention_split_head_dim) * 1.44269504
    attention_merge_partial_out = partial_out
    attention_merge_partial_lse = partial_lse
    (
        attention_merge_splits,
        attention_merge_num_kv_heads,
        attention_merge_q_per_kv,
        attention_merge_head_dim,
    ) = attention_merge_partial_out.shape
    hl.specialize(attention_merge_splits)
    hl.specialize(attention_merge_num_kv_heads)
    hl.specialize(attention_merge_q_per_kv)
    hl.specialize(attention_merge_head_dim)
    attention_merge_query_heads = (
        attention_merge_num_kv_heads * attention_merge_q_per_kv
    )
    hl.specialize(attention_merge_query_heads)
    attention_merge_partial_out_flat = attention_merge_partial_out.view(
        attention_merge_splits,
        attention_merge_query_heads,
        attention_merge_head_dim,
    )
    attention_merge_partial_lse_flat = attention_merge_partial_lse.view(
        attention_merge_splits, attention_merge_query_heads
    )
    attention = torch.empty(
        (attention_merge_query_heads, attention_merge_head_dim),
        dtype=torch.bfloat16,
        device=attention_merge_partial_out.device,
    )
    attention_flat = attention.view(batch, hidden)
    o_mm_m = batch_capacity
    o_mm_n, o_mm_k = o_weight.size()
    # o_proj is split over K so each head group's partial can start as soon
    # as its heads are merged.
    o_chunks = 4
    o_chunk_k = o_mm_k // o_chunks
    o_partials = torch.empty(
        (o_mm_m, o_chunks, o_mm_n), dtype=torch.float32, device=pre_input.device
    )
    attention_chunked = attention_flat.view(o_mm_m, o_chunks, o_chunk_k)
    attention_out = torch.empty(
        (o_mm_m, o_mm_n), dtype=torch.bfloat16, device=pre_input.device
    )
    post_residual = residual
    post_norm_weight = post_weight
    post_epsilon = eps
    post_group_size = group
    post_hidden_size = o_mm_n
    post_num_tokens = batch_capacity
    post_groups_per_row = post_hidden_size // post_group_size
    hl.specialize(post_groups_per_row)
    post_rms_partials = torch.empty(
        (post_num_tokens, post_groups_per_row),
        dtype=torch.float32,
        device=pre_input.device,
    )
    w13_m = batch_capacity
    w13_n, w13_k = w13.size()
    # RMSNorm's per-token scale is applied after the projection
    # (W (x r g) = r W (x g)), so each K chunk of gate_up only waits for its
    # own residual-sum groups rather than for the full-row reduction.
    w13_chunks = 4
    w13_chunk_k = w13_k // w13_chunks
    w13_partials = torch.empty(
        (w13_m, w13_chunks, w13_n), dtype=torch.float32, device=pre_input.device
    )
    ffn_chunked = ffn_normed.view(w13_m, w13_chunks, w13_chunk_k)
    gate_up = torch.empty((w13_m, w13_n), dtype=torch.bfloat16, device=pre_input.device)
    gate_up_halves = gate_up.view(w13_m, 2, w13_n // 2)
    w13_partial_halves = w13_partials.view(w13_m, w13_chunks, 2, w13_n // 2)
    activation_m = batch_capacity
    activation_intermediate = w13_n // 2
    activation_group_size = group
    hl.specialize(activation_group_size)
    # Two W2 K partitions; the chunk dimension stays visible so dependency
    # analysis can release each half as soon as its activation groups are ready.
    w2_chunks = 2
    w2_chunk_k = activation_intermediate // w2_chunks
    activation_chunked = torch.empty(
        (activation_m, w2_chunks, w2_chunk_k),
        dtype=torch.bfloat16,
        device=pre_input.device,
    )
    activation = activation_chunked.view(activation_m, activation_intermediate)
    w2_m = batch_capacity
    w2_n, w2_k = w2.size()
    w2_partials = torch.empty(
        (w2_m, w2_chunks, w2_n), dtype=torch.float32, device=pre_input.device
    )
    output = torch.empty((w2_m, w2_n), dtype=torch.bfloat16, device=pre_input.device)
    # L2 prefetch views; each ends in whole row segments, which hl.prefetch
    # covers with one bulk prefetch each.
    o_prefetch = o_weight.view(o_mm_n, -1, PREFETCH_O_COLS)[:, 0, :]
    w2_prefetch = w2.view(w2_n, -1, PREFETCH_W2_COLS)[:, 0, :]
    w13_prefetch = w13.view(w13_n, w13_chunks, w13_chunk_k)
    # input_layernorm: residual add and RMS partials.
    for pre_partial_m, pre_partial_n in hl.tile(
        [pre_num_tokens, pre_hidden_size], block_size=[1, pre_group_size]
    ):
        # Loads first: a load after a store to a possibly aliasing tensor waits
        # for it, which would serialize a second memory round trip.
        pre_scale = pre_norm_weight[pre_partial_n][None, :].to(torch.float32)
        pre_partial_values = pre_input[pre_partial_m, pre_partial_n].to(
            torch.float32
        ) + pre_residual[pre_partial_m, pre_partial_n].to(torch.float32)
        pre_residual[pre_partial_m, pre_partial_n] = pre_partial_values.to(
            pre_residual.dtype
        )
        pre_rms_partials[pre_partial_m, pre_partial_n.id] = torch.sum(
            pre_partial_values * pre_partial_values, dim=-1
        )
        pre_normed[pre_partial_m, pre_partial_n] = (pre_partial_values * pre_scale).to(
            torch.bfloat16
        )
    # self_attn.qkv_proj.
    for qkv_mm_tile_m, qkv_mm_tile_n in hl.tile([qkv_mm_m, qkv_mm_n]):
        qkv_mm_acc = hl.zeros([qkv_mm_tile_m, qkv_mm_tile_n], dtype=torch.float32)
        for qkv_mm_tile_k in hl.tile(qkv_mm_k):
            qkv_mm_acc = hl.dot(
                pre_normed[qkv_mm_tile_m, qkv_mm_tile_k],
                qkv_weight[qkv_mm_tile_n, qkv_mm_tile_k].T,
                acc=qkv_mm_acc,
            )
        qkv_square_sum = hl.zeros([qkv_mm_tile_m], dtype=torch.float32)
        for qkv_reduce_g in hl.tile(pre_groups_per_row, block_size=pre_groups_per_row):
            qkv_square_sum = qkv_square_sum + torch.sum(
                pre_rms_partials[qkv_mm_tile_m, qkv_reduce_g], dim=-1
            )
        qkv_inv_rms = torch.rsqrt(
            qkv_square_sum * (1.0 / pre_hidden_size) + pre_epsilon
        )
        qkv[qkv_mm_tile_m, qkv_mm_tile_n] = (qkv_mm_acc * qkv_inv_rms[:, None]).to(
            qkv.dtype
        )
    # vLLM: self_attn.attn split paged-attention accumulation.
    for (
        attention_split_tile_q,
        attention_split_tile_bg,
        attention_split_tile_rsplit,
    ) in hl.tile(
        [
            attention_split_q_per_kv,
            attention_split_token_kv_heads,
            attention_split_splits,
        ],
        block_size=[None, 1, 1],
    ):
        attention_split_m_i = hl.full(
            [attention_split_tile_bg, attention_split_tile_q],
            -3.4028234663852886e38,
            dtype=torch.float32,
        )
        attention_split_l_i = hl.full(
            [attention_split_tile_bg, attention_split_tile_q],
            1.0,
            dtype=torch.float32,
        )
        attention_split_acc = hl.zeros(
            [
                attention_split_tile_bg,
                attention_split_tile_q,
                attention_split_head_dim,
            ],
            dtype=torch.float32,
        )
        attention_split_split_idx = (
            attention_split_splits - 1 - attention_split_tile_rsplit.begin
        )
        attention_split_token = (
            attention_split_tile_bg.index // attention_split_num_kv_heads
        )
        attention_split_kv_head = (
            attention_split_tile_bg.index % attention_split_num_kv_heads
        )
        attention_split_context_len = hl.load(context_lens, [attention_split_token])
        if (
            attention_split_split_idx * attention_split_split_context
            < attention_split_context_len
        ):
            attention_split_query_head = (
                attention_split_kv_head[:, None] * attention_split_q_per_kv
                + attention_split_tile_q.index[None, :]
            )
            # q_norm/k_norm and rotary_emb for this kv group, in registers.
            attention_split_lo = hl.arange(qk_embed_dim)
            attention_split_hi = attention_split_lo + qk_embed_dim
            # This split's KV rows: nothing in the kernel writes them before the
            # attention reads them (the new row comes from registers), so these
            # loads are issued before the wait for qkv.
            attention_split_n = (
                attention_split_split_idx * attention_split_split_context
                + hl.arange(attention_split_split_context)
            )
            attention_split_valid_n = (
                attention_split_n[None, :] < attention_split_context_len[:, None]
            )
            attention_split_physical_block = hl.load(
                attention_split_block_table,
                [
                    attention_split_token[:, None],
                    (attention_split_n // attention_split_block_size)[None, :],
                ],
                extra_mask=attention_split_valid_n,
            )
            attention_split_block_offset = (
                attention_split_n % attention_split_block_size
            )
            attention_split_d = hl.arange(attention_split_head_dim)
            attention_split_k1 = hl.load(
                attention_split_kv_cache,
                [
                    attention_split_physical_block[:, :, None],
                    attention_split_block_offset[None, :, None],
                    attention_split_kv_head[:, None, None],
                    attention_split_lo[None, None, :],
                ],
                extra_mask=attention_split_valid_n[:, :, None],
            )
            attention_split_k2 = hl.load(
                attention_split_kv_cache,
                [
                    attention_split_physical_block[:, :, None],
                    attention_split_block_offset[None, :, None],
                    attention_split_kv_head[:, None, None],
                    attention_split_hi[None, None, :],
                ],
                extra_mask=attention_split_valid_n[:, :, None],
            )
            # Warm L2 with this split's V rows (two 128-byte lines each).
            hl.prefetch(
                attention_split_kv_cache,
                [
                    attention_split_physical_block[:, :, None],
                    attention_split_block_offset[None, :, None],
                    attention_split_kv_head[:, None, None],
                    (attention_split_head_dim + hl.arange(2) * 64)[None, None, :],
                ],
                extra_mask=attention_split_valid_n[:, :, None],
            )
            attention_split_pos = hl.load(qk_position_ids, [attention_split_token])
            attention_split_slot = hl.load(cache_slot_mapping, [attention_split_token])
            attention_split_cos = hl.load(
                qk_cos_sin_cache,
                [attention_split_pos[:, None], attention_split_lo[None, :]],
            )
            attention_split_sin = hl.load(
                qk_cos_sin_cache,
                [attention_split_pos[:, None], attention_split_hi[None, :]],
            )
            attention_split_q1 = hl.load(
                attention_split_query,
                [
                    attention_split_token[:, None, None],
                    attention_split_query_head[:, :, None],
                    attention_split_lo[None, None, :],
                ],
            ).to(torch.float32)
            attention_split_q2 = hl.load(
                attention_split_query,
                [
                    attention_split_token[:, None, None],
                    attention_split_query_head[:, :, None],
                    attention_split_hi[None, None, :],
                ],
            ).to(torch.float32)
            attention_split_q_rms = torch.rsqrt(
                (attention_split_q1.pow(2).sum(-1) + attention_split_q2.pow(2).sum(-1))
                * (1.0 / qk_head_dim)
                + qk_eps
            )
            attention_split_q1 = (
                (attention_split_q1 * attention_split_q_rms[:, :, None]).to(
                    torch.bfloat16
                )
                * qk_q_weight[attention_split_lo][None, None, :]
            ).to(torch.bfloat16)
            attention_split_q2 = (
                (attention_split_q2 * attention_split_q_rms[:, :, None]).to(
                    torch.bfloat16
                )
                * qk_q_weight[attention_split_hi][None, None, :]
            ).to(torch.bfloat16)
            attention_split_k1_new = hl.load(
                cache_key,
                [
                    attention_split_token[:, None],
                    attention_split_kv_head[:, None],
                    attention_split_lo[None, :],
                ],
            ).to(torch.float32)
            attention_split_k2_new = hl.load(
                cache_key,
                [
                    attention_split_token[:, None],
                    attention_split_kv_head[:, None],
                    attention_split_hi[None, :],
                ],
            ).to(torch.float32)
            attention_split_k_rms = torch.rsqrt(
                (
                    attention_split_k1_new.pow(2).sum(-1)
                    + attention_split_k2_new.pow(2).sum(-1)
                )
                * (1.0 / qk_head_dim)
                + qk_eps
            )
            attention_split_k1_new = (
                (attention_split_k1_new * attention_split_k_rms[:, None]).to(
                    torch.bfloat16
                )
                * qk_k_weight[attention_split_lo][None, :]
            ).to(torch.bfloat16)
            attention_split_k2_new = (
                (attention_split_k2_new * attention_split_k_rms[:, None]).to(
                    torch.bfloat16
                )
                * qk_k_weight[attention_split_hi][None, :]
            ).to(torch.bfloat16)
            attention_split_q_r1 = (
                (
                    attention_split_q1 * attention_split_cos[:, None, :]
                    - attention_split_q2 * attention_split_sin[:, None, :]
                ).to(torch.bfloat16)
                * attention_split_qk_scale
            ).to(attention_split_query.dtype)
            attention_split_q_r2 = (
                (
                    attention_split_q2 * attention_split_cos[:, None, :]
                    + attention_split_q1 * attention_split_sin[:, None, :]
                ).to(torch.bfloat16)
                * attention_split_qk_scale
            ).to(attention_split_query.dtype)
            attention_split_k_r1 = (
                attention_split_k1_new * attention_split_cos
                - attention_split_k2_new * attention_split_sin
            ).to(torch.bfloat16)
            attention_split_k_r2 = (
                attention_split_k2_new * attention_split_cos
                + attention_split_k1_new * attention_split_sin
            ).to(torch.bfloat16)
            attention_split_v_new = hl.load(
                cache_value,
                [
                    attention_split_token[:, None],
                    attention_split_kv_head[:, None],
                    hl.arange(attention_split_head_dim)[None, :],
                ],
            )
            attention_split_is_new = (
                attention_split_n[None, :] == attention_split_pos[:, None]
            )[:, :, None]
            attention_split_k1 = torch.where(
                attention_split_is_new,
                attention_split_k_r1[:, None, :],
                attention_split_k1,
            )
            attention_split_k2 = torch.where(
                attention_split_is_new,
                attention_split_k_r2[:, None, :],
                attention_split_k2,
            )
            attention_split_scores = torch.bmm(
                attention_split_q_r1, attention_split_k1.transpose(1, 2), torch.float32
            ) + torch.bmm(
                attention_split_q_r2, attention_split_k2.transpose(1, 2), torch.float32
            )
            attention_split_scores = torch.where(
                attention_split_valid_n[:, None, :],
                attention_split_scores,
                -3.4028234663852886e38,
            )
            attention_split_m_ij = torch.maximum(
                attention_split_m_i, torch.amax(attention_split_scores, -1)
            )
            attention_split_p = torch.exp2(
                attention_split_scores - attention_split_m_ij[:, :, None]
            )
            attention_split_alpha = torch.exp2(
                attention_split_m_i - attention_split_m_ij
            )
            attention_split_l_i = (
                attention_split_l_i * attention_split_alpha
                + torch.sum(attention_split_p, -1)
            )
            attention_split_acc = (
                attention_split_acc * attention_split_alpha[:, :, None]
            )
            attention_split_v = hl.load(
                attention_split_kv_cache,
                [
                    attention_split_physical_block[:, :, None],
                    attention_split_block_offset[None, :, None],
                    attention_split_kv_head[:, None, None],
                    (attention_split_d + attention_split_head_dim)[None, None, :],
                ],
                extra_mask=attention_split_valid_n[:, :, None],
            )
            attention_split_v = torch.where(
                attention_split_is_new,
                attention_split_v_new[:, None, :],
                attention_split_v,
            )
            attention_split_acc = torch.baddbmm(
                attention_split_acc,
                attention_split_p.to(attention_split_v.dtype),
                attention_split_v,
            )
            attention_split_m_i = attention_split_m_ij
            partial_out[
                attention_split_split_idx,
                attention_split_tile_bg,
                attention_split_tile_q,
                :,
            ] = attention_split_acc / attention_split_l_i[:, :, None]
            partial_lse[
                attention_split_split_idx,
                attention_split_tile_bg,
                attention_split_tile_q,
            ] = attention_split_m_i + torch.log2(attention_split_l_i)
            # KV-cache update: the split holding the new token stores its row.
            # Only that split reads the slot, and it takes the row from registers.
            attention_split_writes = (
                attention_split_pos // attention_split_split_context
                == attention_split_split_idx
            ) & (attention_split_tile_q.begin == 0)
            attention_split_slot_block = (attention_split_slot // cache_block_size)[
                :, None
            ]
            attention_split_slot_offset = (attention_split_slot % cache_block_size)[
                :, None
            ]
            attention_split_slot_head = attention_split_kv_head[:, None]
            hl.store(
                cache_kv_cache,
                [
                    attention_split_slot_block,
                    attention_split_slot_offset,
                    attention_split_slot_head,
                    attention_split_lo[None, :],
                ],
                attention_split_k_r1,
                extra_mask=attention_split_writes[:, None],
            )
            hl.store(
                cache_kv_cache,
                [
                    attention_split_slot_block,
                    attention_split_slot_offset,
                    attention_split_slot_head,
                    attention_split_hi[None, :],
                ],
                attention_split_k_r2,
                extra_mask=attention_split_writes[:, None],
            )
            hl.store(
                cache_kv_cache,
                [
                    attention_split_slot_block,
                    attention_split_slot_offset,
                    attention_split_slot_head,
                    (attention_split_d + attention_split_head_dim)[None, :],
                ],
                attention_split_v_new,
                extra_mask=attention_split_writes[:, None],
            )
    # L2 prefetch of the first o_proj columns while attention finishes.
    for prefetch_o_tile in hl.tile(o_mm_n, block_size=4):
        hl.prefetch(o_prefetch, [prefetch_o_tile, slice(None)])
    # vLLM: self_attn.attn split-output merge (single stage, valid splits only).
    for attention_merge_final_head in hl.tile(
        attention_merge_query_heads, block_size=1
    ):
        attention_merge_final_token = attention_merge_final_head.begin // (
            attention_split_num_kv_heads * attention_merge_q_per_kv
        )
        attention_merge_final_context_len = context_lens[attention_merge_final_token]
        attention_merge_final_valid = (
            attention_merge_final_context_len + attention_split_split_context - 1
        ) // attention_split_split_context
        attention_merge_final_m = hl.full(
            [attention_merge_final_head], -3.4028234663852886e38, dtype=torch.float32
        )
        attention_merge_final_l = hl.zeros(
            [attention_merge_final_head], dtype=torch.float32
        )
        attention_merge_final_acc = hl.zeros(
            [attention_merge_final_head, attention_merge_head_dim],
            dtype=torch.float32,
        )
        for attention_merge_final_tile_s in hl.tile(
            0, attention_merge_final_valid, block_size=32
        ):
            attention_merge_final_lse = attention_merge_partial_lse_flat[
                attention_merge_final_tile_s, attention_merge_final_head
            ]
            attention_merge_final_m_new = torch.maximum(
                attention_merge_final_m, torch.amax(attention_merge_final_lse, 0)
            )
            attention_merge_final_w = torch.exp2(
                attention_merge_final_lse - attention_merge_final_m_new[None, :]
            )
            attention_merge_final_alpha = torch.exp2(
                attention_merge_final_m - attention_merge_final_m_new
            )
            attention_merge_final_l = attention_merge_final_l * (
                attention_merge_final_alpha
            ) + torch.sum(attention_merge_final_w, 0)
            attention_merge_final_values = attention_merge_partial_out_flat[
                attention_merge_final_tile_s, attention_merge_final_head, :
            ]
            attention_merge_final_acc = attention_merge_final_acc * (
                attention_merge_final_alpha[:, None]
            ) + torch.sum(
                attention_merge_final_values
                * attention_merge_final_w[:, :, None].to(torch.float32),
                0,
            )
            attention_merge_final_m = attention_merge_final_m_new
        attention[attention_merge_final_head, :] = (
            attention_merge_final_acc / attention_merge_final_l[:, None]
        ).to(attention.dtype)
    # self_attn.o_proj.
    for o_mm_tile_m, o_mm_tile_chunk, o_mm_tile_n in hl.tile(
        [o_mm_m, o_chunks, o_mm_n], block_size=[None, 1, None]
    ):
        o_mm_acc = hl.zeros([o_mm_tile_m, o_mm_tile_n], dtype=torch.float32)
        for o_mm_tile_local_k in hl.tile(o_chunk_k):
            o_mm_tile_k = o_mm_tile_chunk.begin * o_chunk_k + o_mm_tile_local_k.index
            o_mm_acc = hl.dot(
                attention_chunked[
                    o_mm_tile_m, o_mm_tile_chunk.begin, o_mm_tile_local_k
                ],
                o_weight[o_mm_tile_n, o_mm_tile_k].T,
                acc=o_mm_acc,
            )
        o_partials[o_mm_tile_m, o_mm_tile_chunk, o_mm_tile_n] = o_mm_acc[:, None, :]
    # post_attention_layernorm: residual add and RMS partials.
    for post_partial_m, post_partial_n in hl.tile(
        [post_num_tokens, post_hidden_size],
        block_size=[1, post_group_size],
    ):
        post_scale = post_norm_weight[post_partial_n][None, :].to(torch.float32)
        post_residual_values = post_residual[post_partial_m, post_partial_n].to(
            torch.float32
        )
        post_attention = torch.sum(o_partials[post_partial_m, :, post_partial_n], dim=1)
        attention_out[post_partial_m, post_partial_n] = post_attention.to(
            attention_out.dtype
        )
        post_partial_values = (
            post_attention.to(attention_out.dtype).to(torch.float32)
            + post_residual_values
        )
        post_residual[post_partial_m, post_partial_n] = post_partial_values.to(
            post_residual.dtype
        )
        post_rms_partials[post_partial_m, post_partial_n.id] = torch.sum(
            post_partial_values * post_partial_values, dim=-1
        )
        ffn_normed[post_partial_m, post_partial_n] = (
            post_partial_values * post_scale
        ).to(torch.bfloat16)
    # L2 prefetch of each program's first gate_up tile: gate_up's tasks run
    # chunk-fastest (tile id -> 8-row group, K chunk), and tile i of a root
    # goes to program i (148 SMs x 7 programs in the batch-1 config).  A
    # module global here would be a runtime argument.
    for prefetch_w13_tile in hl.tile(148 * 7, block_size=1):
        hl.prefetch(
            w13_prefetch,
            [
                prefetch_w13_tile.id // w13_chunks * 8 + hl.arange(8),
                prefetch_w13_tile.id % w13_chunks,
                slice(None),
            ],
        )
    # mlp.gate_up_proj.
    for w13_tile_m, w13_tile_chunk, w13_tile_n in hl.tile(
        [w13_m, w13_chunks, w13_n], block_size=[None, 1, None]
    ):
        w13_acc = hl.zeros([w13_tile_m, w13_tile_n], dtype=torch.float32)
        for w13_tile_local_k in hl.tile(w13_chunk_k):
            w13_tile_k = w13_tile_chunk.begin * w13_chunk_k + w13_tile_local_k.index
            w13_acc = hl.dot(
                ffn_chunked[w13_tile_m, w13_tile_chunk.begin, w13_tile_local_k],
                w13[w13_tile_n, w13_tile_k].T,
                acc=w13_acc,
            )
        w13_partials[w13_tile_m, w13_tile_chunk, w13_tile_n] = w13_acc[:, None, :]
    # mlp.act_fn (SiLU and multiply, BF16 down_proj input).
    for activation_tile_m, activation_tile_i in hl.tile(
        [activation_m, activation_intermediate],
        block_size=[1, activation_group_size],
    ):
        post_square_sum = hl.zeros([activation_tile_m], dtype=torch.float32)
        for post_reduce_g in hl.tile(
            post_groups_per_row, block_size=post_groups_per_row
        ):
            post_square_sum = post_square_sum + torch.sum(
                post_rms_partials[activation_tile_m, post_reduce_g], dim=-1
            )
        post_inv_rms = torch.rsqrt(
            post_square_sum * (1.0 / post_hidden_size) + post_epsilon
        )
        activation_gate = (
            torch.sum(
                w13_partial_halves[activation_tile_m, :, 0, activation_tile_i],
                dim=1,
            )
            * post_inv_rms[:, None]
        ).to(torch.bfloat16)
        activation_up = (
            torch.sum(
                w13_partial_halves[activation_tile_m, :, 1, activation_tile_i],
                dim=1,
            )
            * post_inv_rms[:, None]
        ).to(torch.bfloat16)
        gate_up_halves[activation_tile_m, 0, activation_tile_i] = activation_gate
        gate_up_halves[activation_tile_m, 1, activation_tile_i] = activation_up
        activation_gate = activation_gate.to(torch.float32)
        activation[activation_tile_m, activation_tile_i] = (
            activation_gate * torch.sigmoid(activation_gate)
        ).to(torch.bfloat16) * activation_up.to(torch.bfloat16)
    # L2 prefetch of the first down_proj columns across the SiLU hand-off.
    for prefetch_down_tile in hl.tile(w2_n, block_size=4):
        hl.prefetch(w2_prefetch, [prefetch_down_tile, slice(None)])
    # mlp.down_proj (two K partitions).
    for w2_tile_m, w2_tile_chunk, w2_tile_n in hl.tile(
        [w2_m, w2_chunks, w2_n], block_size=[None, 1, None]
    ):
        w2_acc = hl.zeros([w2_tile_m, w2_tile_n], dtype=torch.float32)
        for w2_tile_local_k in hl.tile(w2_chunk_k):
            w2_tile_k = w2_tile_chunk.begin * w2_chunk_k + w2_tile_local_k.index
            w2_acc = hl.dot(
                activation_chunked[w2_tile_m, w2_tile_chunk.begin, w2_tile_local_k],
                w2[w2_tile_n, w2_tile_k].T,
                acc=w2_acc,
            )
        w2_partials[w2_tile_m, w2_tile_chunk, w2_tile_n] = w2_acc[:, None, :]
    for w2_final_m, w2_final_n in hl.tile([w2_m, w2_n], block_size=[1, 8]):
        w2_final_values = w2_partials[w2_final_m, :, w2_final_n]
        output[w2_final_m, w2_final_n] = torch.sum(w2_final_values, dim=1).to(
            output.dtype
        )
    return (
        output,
        pre_normed,
        qkv,
        partial_out,
        partial_lse,
        attention,
        attention_out,
        ffn_normed,
        gate_up,
        activation,
        residual,
    )


def use_cudagraph() -> bool:
    """main() captures and replays its own CUDA graphs."""
    return True


def _fp8_module():
    """The FP8 sibling module, whose vLLM harness helpers are reused."""
    name = "_helion_pretuned_qwen3_decode_layer_fp8_helpers"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent / "qwen3_decode_layer.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def has_vllm() -> bool:
    return _fp8_module().has_vllm()


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("qwen3_decode_layer_bf16 is pretuned only for NVIDIA SM100")


def _bf16_config() -> dict:
    config = dict(_fp8_module().QWEN3_8B_FP8_CONFIG)
    del config["quantization_config"]
    return config


def _make_inputs(
    seed: int = 0, context_len: int = CONTEXT, batch: int = BATCH
) -> dict[str, torch.Tensor]:
    """One decode token per sequence; every sequence has ``context_len`` rows."""
    if not 1 <= context_len <= CONTEXT:
        raise ValueError(f"context_len must be in [1, {CONTEXT}]")
    torch.manual_seed(seed)
    qkv_width = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
    logical_blocks = math.ceil(CONTEXT / CACHE_BLOCK)
    physical_blocks = math.ceil(logical_blocks * batch * 1.25)
    block_table = (
        torch.randperm(physical_blocks, device="cuda", dtype=torch.int64)[
            : logical_blocks * batch
        ]
        .to(torch.int32)
        .view(batch, logical_blocks)
    )
    final_logical_block = (context_len - 1) // CACHE_BLOCK
    final_block_offset = (context_len - 1) % CACHE_BLOCK
    final_physical_block = block_table[:, final_logical_block].to(torch.int64)

    def weight(rows: int, cols: int) -> torch.Tensor:
        return torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16) * (
            1.0 / math.sqrt(cols)
        )

    return {
        "hidden_states": torch.randn(
            (batch, HIDDEN), device="cuda", dtype=torch.bfloat16
        ),
        "residual": torch.randn((batch, HIDDEN), device="cuda", dtype=torch.bfloat16),
        "pre_weight": torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
        + 1.0,
        "pre_normed": torch.empty((batch, HIDDEN), device="cuda", dtype=torch.bfloat16),
        "qkv_weight": weight(qkv_width, HIDDEN),
        "q_weight": torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
        + 1.0,
        "k_weight": torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
        + 1.0,
        # Filled from the vLLM layer's rotary cache before either implementation runs.
        "cos_sin": torch.empty(
            (max(CONTEXT, 4096), HEAD_DIM), device="cuda", dtype=torch.bfloat16
        ),
        "position": torch.full(
            (batch,), context_len - 1, device="cuda", dtype=torch.int64
        ),
        "kv_cache": torch.randn(
            (physical_blocks, CACHE_BLOCK, KV_HEADS, 2 * HEAD_DIM),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        "block_table": block_table,
        "slot_mapping": final_physical_block * CACHE_BLOCK + final_block_offset,
        "o_weight": weight(HIDDEN, HIDDEN),
        "post_weight": torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
        + 1.0,
        "ffn_normed": torch.empty((batch, HIDDEN), device="cuda", dtype=torch.bfloat16),
        "w13": weight(2 * INTERMEDIATE, HIDDEN),
        "w2": weight(HIDDEN, INTERMEDIATE),
        "context_lens": torch.full(
            (batch,), context_len, device="cuda", dtype=torch.int64
        ),
    }


def _cache_rows(tensors: dict[str, torch.Tensor], *, vllm_layout: bool = False):
    """The KV rows this step writes, one [KV_HEADS, 2 * HEAD_DIM] view per sequence."""
    rows = []
    for slot in tensors["slot_mapping"].tolist():
        block, offset = divmod(int(slot), CACHE_BLOCK)
        if vllm_layout:
            rows.append(tensors["kv_cache"][block, :, offset])
        else:
            rows.append(tensors["kv_cache"][block, offset])
    return rows


def _make_reset(tensors: dict[str, torch.Tensor], *, vllm_layout: bool = False):
    initial_residual = tensors["residual"].clone()
    rows = _cache_rows(tensors, vllm_layout=vllm_layout)
    initial_rows = [row.clone() for row in rows]

    def reset() -> None:
        tensors["residual"].copy_(initial_residual)
        for row, initial in zip(rows, initial_rows, strict=True):
            row.copy_(initial)

    return reset


def _clone_writable(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned = dict(tensors)
    for name in ("residual", "pre_normed", "kv_cache", "ffn_normed"):
        cloned[name] = tensors[name].clone()
    return cloned


def _kernel_args(tensors: dict[str, torch.Tensor]) -> tuple[object, ...]:
    return (
        tensors["hidden_states"],
        tensors["residual"],
        tensors["pre_weight"],
        tensors["pre_normed"],
        tensors["qkv_weight"],
        tensors["q_weight"],
        tensors["k_weight"],
        tensors["cos_sin"],
        tensors["position"],
        tensors["kv_cache"],
        tensors["block_table"],
        tensors["slot_mapping"],
        tensors["o_weight"],
        tensors["post_weight"],
        tensors["ffn_normed"],
        tensors["w13"],
        tensors["w2"],
        tensors["context_lens"],
        HIDDEN,
        INTERMEDIATE,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        CONTEXT,
        CACHE_BLOCK,
        ATTENTION_SPLITS,
        GROUP,
        EPS,
    )


def _initialize_vllm(model_path: Path) -> tuple[object, object, object]:
    from vllm.config import CacheConfig
    from vllm.config import ModelConfig
    from vllm.config import VllmConfig
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment
    from vllm.distributed import initialize_model_parallel
    from vllm.utils.network_utils import get_open_port

    fp8 = _fp8_module()
    try:
        model_config = ModelConfig(
            model=str(model_path),
            tokenizer=str(model_path),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=CONTEXT,
            config_format="hf",
        )
        cache_config = CacheConfig(block_size=CACHE_BLOCK, cache_dtype="auto")
        vllm_config = VllmConfig(model_config=model_config, cache_config=cache_config)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=torch.cuda.current_device(),
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(1, 1)
    except Exception:
        fp8._destroy_vllm()
        raise
    return vllm_config, model_config, cache_config


def _make_attention_metadata(vllm_config, attention, tensors, layer_name, context_len):
    """vLLM decode metadata: one query token per sequence, equal context lengths."""
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backend import CommonAttentionMetadata

    batch = tensors["hidden_states"].size(0)
    common = CommonAttentionMetadata(
        query_start_loc=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
        query_start_loc_cpu=torch.arange(batch + 1, dtype=torch.int32),
        seq_lens=torch.full((batch,), context_len, device="cuda", dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.full((batch,), CONTEXT, dtype=torch.int32),
        num_reqs=batch,
        num_actual_tokens=batch,
        max_query_len=1,
        max_seq_len=context_len,
        block_table_tensor=tensors["block_table"],
        slot_mapping=tensors["slot_mapping"],
        causal=True,
        positions=tensors["position"],
    )
    spec = attention.get_kv_cache_spec(vllm_config)
    with set_current_vllm_config(vllm_config):
        metadata = (
            attention.get_attn_backend()
            .get_builder_cls()(spec, [layer_name], vllm_config, torch.device("cuda"))
            .build(0, common)
        )
    return {layer_name: metadata}, {layer_name: tensors["slot_mapping"]}


def _make_vllm_call(tensors: dict[str, torch.Tensor], context_len: int):
    """vLLM's compiled BF16 Qwen3DecoderLayer with default backend selection."""
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    fp8 = _fp8_module()
    fp8._clear_vllm_attention_cache()
    model_directory = tempfile.TemporaryDirectory(prefix="qwen3-8b-bf16-")
    model_path = Path(model_directory.name)
    (model_path / "config.json").write_text(json.dumps(_bf16_config()))
    vllm_config, model_config, cache_config = _initialize_vllm(model_path)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(vllm_config):
            wrapper = (
                fp8._make_compiled_layer_class()(
                    config=model_config.hf_text_config,
                    cache_config=cache_config,
                    quant_config=None,
                    prefix="model.layers.0",
                )
                .eval()
                .cuda()
            )
    finally:
        torch.set_default_dtype(old_dtype)
    layer = wrapper.layer
    tensors["cos_sin"].copy_(layer.self_attn.rotary_emb.cos_sin_cache[:CONTEXT])
    with torch.no_grad():
        layer.self_attn.qkv_proj.weight.copy_(tensors["qkv_weight"])
        layer.self_attn.o_proj.weight.copy_(tensors["o_weight"])
        layer.mlp.gate_up_proj.weight.copy_(tensors["w13"])
        layer.mlp.down_proj.weight.copy_(tensors["w2"])
        layer.self_attn.q_norm.weight.copy_(tensors["q_weight"])
        layer.self_attn.k_norm.weight.copy_(tensors["k_weight"])
        layer.input_layernorm.weight.copy_(tensors["pre_weight"])
        layer.post_attention_layernorm.weight.copy_(tensors["post_weight"])
    with set_current_vllm_config(vllm_config):
        process_weights_after_loading(wrapper, model_config, torch.device("cuda"))
    attention = layer.self_attn.attn
    layer_name = "model.layers.0.self_attn.attn"
    with set_current_vllm_config(vllm_config):
        supported_layouts = get_supported_kv_cache_layouts(
            [attention.get_attn_backend()]
        )
        cache_layout = resolve_kv_cache_layout(
            vllm_config,
            [[layout.name for layout in supported_layouts]],
            [attention.get_kv_cache_spec(vllm_config)],
        )
    vllm_tensors = {
        "hidden_states": tensors["hidden_states"],
        "residual": tensors["residual"].clone(),
        "position": tensors["position"],
        "kv_cache": fp8._make_vllm_cache(
            tensors["kv_cache"].clone(), cache_layout.layer_view_order
        ),
        "block_table": tensors["block_table"],
        "slot_mapping": tensors["slot_mapping"],
    }
    attention.kv_cache = vllm_tensors["kv_cache"]
    attention_metadata, slot_mapping = _make_attention_metadata(
        vllm_config, attention, vllm_tensors, layer_name, context_len
    )
    backend = f"{attention.get_attn_backend().get_name().lower()} + unquantized linear"

    def launch():
        with set_forward_context(
            attention_metadata,
            vllm_config=vllm_config,
            num_tokens=tensors["hidden_states"].size(0),
            slot_mapping=slot_mapping,
        ):
            return wrapper(
                vllm_tensors["position"],
                vllm_tensors["hidden_states"],
                vllm_tensors["residual"],
            )

    def close() -> None:
        try:
            fp8._destroy_vllm()
        finally:
            try:
                fp8._clear_vllm_attention_cache()
            finally:
                model_directory.cleanup()

    return launch, vllm_tensors, backend, close


def _assert_vllm_close(
    helion_outputs, vllm_outputs, helion_tensors, vllm_tensors
) -> None:
    vllm_output, vllm_residual = vllm_outputs
    torch.testing.assert_close(
        helion_outputs[0].float(), vllm_output.float(), atol=0.125, rtol=0.05
    )
    torch.testing.assert_close(
        helion_outputs[-1].float(), vllm_residual.float(), atol=0.125, rtol=0.03
    )
    for helion_row, vllm_row in zip(
        _cache_rows(helion_tensors),
        _cache_rows(vllm_tensors, vllm_layout=True),
        strict=True,
    ):
        torch.testing.assert_close(
            helion_row.float(), vllm_row.float(), atol=0.125, rtol=0.03
        )


@torch.inference_mode()
def correctness_check() -> None:
    """Check outputs and cache writes against compiled BF16 vLLM for every case."""
    _require_sm100()
    for batch in BATCHES:
        for context_len in CONTEXT_LENGTHS:
            base = _make_inputs(context_len=context_len, batch=batch)
            vllm_call, vllm_tensors, _backend, close_vllm = _make_vllm_call(
                base, context_len
            )
            try:
                helion_tensors = _clone_writable(base)
                helion_outputs = qwen3_decode_layer_bf16(*_kernel_args(helion_tensors))
                vllm_outputs = vllm_call()
                torch.cuda.synchronize()
                _assert_vllm_close(
                    helion_outputs, vllm_outputs, helion_tensors, vllm_tensors
                )
            finally:
                close_vllm()


@torch.inference_mode()
def main(
    verbose: bool = True, output: str | None = None, extra_contexts: bool = False
) -> dict:
    """Benchmark the BF16 megakernel against compiled BF16 vLLM with cold L2."""
    _require_sm100()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _bench import bench_pre_captured_cudagraphs
    from _bench import capture_cuda_graph
    from _bench import geomean
    from _bench import thermal_warmup

    cases = [
        (batch, context) for batch in BENCHMARK_BATCHES for context in CONTEXT_LENGTHS
    ]
    if extra_contexts:
        cases += [(1, context) for context in EXTRA_CONTEXT_LENGTHS]
    rows = []
    for batch, context_len in cases:
        base = _make_inputs(context_len=context_len, batch=batch)
        vllm_call, vllm_tensors, backend, close_vllm = _make_vllm_call(
            base, context_len
        )
        try:
            helion_tensors = _clone_writable(base)
            helion_reset = _make_reset(helion_tensors)
            vllm_reset = _make_reset(vllm_tensors, vllm_layout=True)
            helion_outputs = qwen3_decode_layer_bf16(*_kernel_args(helion_tensors))
            vllm_outputs = vllm_call()
            torch.cuda.synchronize()
            _assert_vllm_close(
                helion_outputs, vllm_outputs, helion_tensors, vllm_tensors
            )
            helion_graph, _ = capture_cuda_graph(
                lambda helion_tensors=helion_tensors: qwen3_decode_layer_bf16(
                    *_kernel_args(helion_tensors)
                ),
                helion_reset,
            )
            vllm_graph, _ = capture_cuda_graph(vllm_call, vllm_reset)
            thermal_warmup(10_000)
            helion_ms, vllm_ms = bench_pre_captured_cudagraphs(
                [helion_graph.replay, vllm_graph.replay],
                rep=200,
                resets=[helion_reset, vllm_reset],
            )
        finally:
            close_vllm()
        rows.append(
            {
                "batch": batch,
                "context": context_len,
                "helion_us": helion_ms * 1000,
                "vllm_us": vllm_ms * 1000,
                "vllm_backend": backend,
            }
        )
        if verbose:
            print(
                f"batch {batch} context {context_len:>5d}: helion {helion_ms * 1000:8.2f} us  "
                f"vllm ({backend}) {vllm_ms * 1000:8.2f} us  "
                f"speedup {vllm_ms / helion_ms:.2f}x",
                flush=True,
            )
    if output:
        Path(output).write_text(json.dumps(rows, indent=1))
    speedups = [row["vllm_us"] / row["helion_us"] for row in rows]
    return {
        "helion_wins": sum(1 for s in speedups if s > 1.0),
        "total": len(speedups),
        "geomean": geomean(speedups),
        "best_speedup": max(speedups),
        "rows": rows,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--extra-contexts", action="store_true")
    cli = parser.parse_args()
    main(output=cli.output, extra_contexts=cli.extra_contexts)
