# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""Qwen3-8B BF16 verify-layer megakernel: one sequence, T new tokens.

``llama32_1b_verify_layer_bf16``'s structure (KV write first, attention tasks of
all T x 4 query rows of a KV head over a split of the sequence, causal) with
Qwen3's per-head q/k RMSNorm and sizes, and gate_up split over K with a separate
activation phase.
"""

from __future__ import annotations

import math

from pretuned_kernels.megakernels._verify_layer import Layer
from pretuned_kernels.megakernels._verify_layer import correctness_check as check
from pretuned_kernels.megakernels._verify_layer import kernel_args
from pretuned_kernels.megakernels._verify_layer import make_inputs
import torch

import helion
import helion.language as hl

TOKENS = 8
HIDDEN = 4096
INTERMEDIATE = 12288
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
CONTEXT = 8192
CACHE_BLOCK = 16
ATTENTION_SPLITS = 16
GROUP = 128
EPS = 1e-6
ROPE_THETA = 1000000.0


@helion.kernel(
    static_shapes=False,
    backend="gluon",
    triton_do_not_specialize=False,
    # One persistent CTA per SM, weights and activations streamed through an
    # 8-deep TMA ring (blocks: QKV m/n/k, merge heads, o_proj m/n/k, gate_up
    # m/n/k, down_proj m/n/k, final reduce).
    config=helion.Config(
        block_sizes=[16, 64, 128, 4, 16, 64, 128, 16, 64, 128, 16, 64, 128, 128],
        num_warps=4,
        pid_type="persistent_blocked",
        cross_loop_pipeline="static",
        num_sm_multiplier=1,
        indexing="tensor_descriptor",
        num_stages=8,
    ),
)
def qwen3_verify_layer_bf16(
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
    # Shapes are specialized (one kernel per batch size, a fixed-capacity KV
    # cache); context lengths, positions, block tables, slot mappings and KV
    # data are runtime values.  All four projections read BF16 weights.
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
    torch._check(block_table.size(0) == 1)
    hl.specialize(slot_mapping.size(0))
    torch._check(slot_mapping.size(0) == batch_capacity)
    hl.specialize(ffn_normed.size(0))
    torch._check(ffn_normed.size(0) == batch_capacity)
    hl.specialize(context_lens.size(0))
    torch._check(context_lens.size(0) == 1)
    hl.specialize(hidden_states.size(1))
    hl.specialize(residual.size(1))
    hl.specialize(pre_weight.size(0))
    hl.specialize(pre_normed.size(1))
    hl.specialize(qkv_weight.size(0))
    hl.specialize(qkv_weight.size(1))
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
    hl.specialize(q_weight.size(0))
    hl.specialize(k_weight.size(0))
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
    qk_qkv = qkv
    qk_num_heads_q = q_heads
    qk_num_heads_k = kv_heads
    qk_num_heads_v = kv_heads
    qk_head_dim = head_dim
    # Qwen3: per-head RMSNorm of q and k (weights over head_dim) before RoPE.
    qk_eps = eps
    qk_q_weight = q_weight
    qk_k_weight = k_weight
    qk_cos_sin_cache = cos_sin
    qk_position_ids = position
    hl.specialize(qk_qkv.shape[1])
    qk_rotary_dim = qk_cos_sin_cache.shape[1]
    hl.specialize(qk_rotary_dim)
    qk_embed_dim = qk_rotary_dim // 2
    hl.specialize(qk_num_heads_q)
    hl.specialize(qk_num_heads_k)
    hl.specialize(qk_num_heads_v)
    hl.specialize(qk_head_dim)
    cache_key = key
    cache_value = value
    cache_kv_cache = kv_cache
    cache_slot_mapping = slot_mapping
    cache_block_size = cache_block
    _, cache_num_kv_heads, cache_head_dim = cache_key.shape
    cache_num_tokens = batch_capacity
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
    # Cache rows as (key | value) halves: a page's operand is a box at a
    # scalar block coordinate.
    attention_split_kv_halves = kv_cache.view(
        kv_cache.size(0), cache_block, kv_heads, 2, head_dim
    )
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
    # A split's most pages (the context's pages split evenly).
    attention_split_max_pages = (
        (attention_split_context + attention_split_block_size - 1)
        // attention_split_block_size
        + attention_split_splits
        - 1
    ) // attention_split_splits
    partial_out = torch.empty(
        (
            attention_split_splits,
            attention_split_num_tokens,
            attention_split_num_kv_heads,
            attention_split_q_per_kv,
            attention_split_head_dim,
        ),
        device=attention_split_query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (
            attention_split_splits,
            attention_split_num_tokens,
            attention_split_num_kv_heads,
            attention_split_q_per_kv,
        ),
        device=attention_split_query.device,
        dtype=torch.float32,
    )
    attention_split_qk_scale = 1.0 / math.sqrt(attention_split_head_dim) * 1.44269504
    # The query rows (token, KV head, head in group, rotary half), RMSNorm'd,
    # RoPE'd and scaled by the KV-cache update, once for every split.
    query_halves = qkv[:, : q_heads * head_dim].view(
        batch, kv_heads, q_heads // kv_heads, 2, head_dim // 2
    )
    query_roped = torch.empty_like(query_halves)
    query_roped_rows = query_roped.view(batch, kv_heads, q_heads // kv_heads, head_dim)
    attention_split_rows = attention_split_num_tokens * attention_split_q_per_kv
    attention_merge_head_dim = attention_split_head_dim
    attention = torch.empty(
        (
            attention_split_num_tokens,
            attention_split_num_kv_heads,
            attention_split_q_per_kv,
            attention_merge_head_dim,
        ),
        dtype=torch.bfloat16,
        device=partial_out.device,
    )
    attention_flat = attention.view(batch, hidden)
    o_mm_m = batch_capacity
    o_mm_n, o_mm_k = o_weight.size()
    # o_proj is split over K so each head group's partial can start as soon
    # as its heads are merged.
    o_chunks = 2
    o_chunk_k = o_mm_k // o_chunks
    o_partials = torch.empty(
        (o_mm_m, o_chunks, o_mm_n), dtype=torch.float32, device=pre_input.device
    )
    attention_chunked = attention_flat.view(o_mm_m, o_chunks, o_chunk_k)
    # Weights viewed by K partition: each tile is a box with a scalar partition
    # coordinate, so it can load by TMA.
    o_weight_chunked = o_weight.view(o_mm_n, o_chunks, o_chunk_k)
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
    # RMSNorm's per-token scale is applied after the projection:
    # W (x r g) = r W (x g).
    gate_up = torch.empty((w13_m, w13_n), dtype=torch.bfloat16, device=pre_input.device)
    activation_m = batch_capacity
    activation_intermediate = w13_n // 2
    activation_group_size = group
    hl.specialize(activation_group_size)
    # 2 W2 K partitions: one wave of long tasks (each task's fixed costs, its
    # dependency wait and epilogue, outweigh the part-empty wave); the chunk
    # dimension stays visible so dependency analysis releases each partition
    # once its activation groups land.
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
    w2_chunked = w2.view(w2_n, w2_chunks, w2_chunk_k)
    # gate_up is split over K into 2 partitions for the same reason (each task
    # one 64-row x 2048 box stream); the activation phase sums the partials.
    w13_chunks = 2
    w13_chunk_k = w13_k // w13_chunks
    w13_kchunked = w13.view(w13_n, w13_chunks, w13_chunk_k)
    ffn_chunked = ffn_normed.view(w13_m, w13_chunks, w13_chunk_k)
    w13_partials = torch.empty(
        (w13_m, w13_chunks, w13_n), dtype=torch.float32, device=pre_input.device
    )
    # Partials and gate_up as (gate | up, down_proj partition, column): the
    # activation phase reads and writes the layout down_proj reads.
    w13_partials_act = w13_partials.view(w13_m, w13_chunks, 2, w2_chunks, w2_chunk_k)
    gate_up_act = gate_up.view(w13_m, 2, w2_chunks, w2_chunk_k)
    w2_partials = torch.empty(
        (w2_m, w2_chunks, w2_n), dtype=torch.float32, device=pre_input.device
    )
    output = torch.empty((w2_m, w2_n), dtype=torch.bfloat16, device=pre_input.device)
    # input_layernorm: residual add and RMS partials.
    for pre_partial_m, pre_partial_n in hl.tile(
        [pre_num_tokens, pre_hidden_size], block_size=[pre_num_tokens, pre_group_size]
    ):
        # Loads first: a load after a store to a possibly aliasing tensor waits
        # for it, which would serialize a second memory round trip.
        pre_scale = pre_norm_weight[pre_partial_n][None, :].to(torch.float32)
        # The residual first: its load blocks (this phase writes it), the
        # input's streams meanwhile.
        pre_partial_values = pre_residual[pre_partial_m, pre_partial_n].to(
            torch.float32
        ) + pre_input[pre_partial_m, pre_partial_n].to(torch.float32)
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
        # Swap-AB: weight rows are the MMA's M, the tokens its N.
        qkv_mm_acc = hl.zeros([qkv_mm_tile_n, qkv_mm_tile_m], dtype=torch.float32)
        for qkv_mm_tile_k in hl.tile(qkv_mm_k):
            qkv_mm_acc = hl.dot(
                qkv_weight[qkv_mm_tile_n, qkv_mm_tile_k],
                pre_normed[qkv_mm_tile_m, qkv_mm_tile_k].T,
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
        qkv[qkv_mm_tile_m, qkv_mm_tile_n] = (qkv_mm_acc.T * qkv_inv_rms[:, None]).to(
            qkv.dtype
        )
    # self_attn.attn: KV-cache update (RoPE'd key), before attention, which
    # reads the new tokens' rows with the cached ones.
    for cache_tile_t, cache_tile_h in hl.tile(
        [cache_num_tokens, cache_num_kv_heads], block_size=[1, 1]
    ):
        cache_token = cache_tile_t.index
        cache_cache_head = cache_tile_h.index
        cache_lo = hl.arange(qk_embed_dim)
        cache_pos = hl.load(qk_position_ids, [cache_token])
        cache_cos = hl.load(
            qk_cos_sin_cache, [cache_pos[:, None], cache_lo[None, :]]
        ).to(torch.float32)[:, None, :]
        cache_sin = hl.load(
            qk_cos_sin_cache, [cache_pos[:, None], (cache_lo + qk_embed_dim)[None, :]]
        ).to(torch.float32)[:, None, :]
        cache_k_lo = cache_key[cache_tile_t, cache_tile_h, cache_lo].to(torch.float32)
        cache_k_hi = cache_key[cache_tile_t, cache_tile_h, cache_lo + qk_embed_dim].to(
            torch.float32
        )
        cache_k_rms = torch.rsqrt(
            (cache_k_lo.pow(2).sum(-1) + cache_k_hi.pow(2).sum(-1))
            * (1.0 / qk_head_dim)
            + qk_eps
        )[:, :, None]
        cache_k_lo = (
            (
                (cache_k_lo * cache_k_rms).to(torch.bfloat16)
                * qk_k_weight[cache_lo][None, None, :]
            )
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        cache_k_hi = (
            (
                (cache_k_hi * cache_k_rms).to(torch.bfloat16)
                * qk_k_weight[cache_lo + qk_embed_dim][None, None, :]
            )
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        cache_q_lo = query_halves[cache_tile_t, cache_tile_h, :, 0, :].to(torch.float32)
        cache_q_hi = query_halves[cache_tile_t, cache_tile_h, :, 1, :].to(torch.float32)
        cache_q_rms = torch.rsqrt(
            (cache_q_lo.pow(2).sum(-1) + cache_q_hi.pow(2).sum(-1))
            * (1.0 / qk_head_dim)
            + qk_eps
        )[:, :, :, None]
        cache_q_lo = (
            (
                (cache_q_lo * cache_q_rms).to(torch.bfloat16)
                * qk_q_weight[cache_lo][None, None, None, :]
            )
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        cache_q_hi = (
            (
                (cache_q_hi * cache_q_rms).to(torch.bfloat16)
                * qk_q_weight[cache_lo + qk_embed_dim][None, None, None, :]
            )
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        cache_q_cos = cache_cos[:, :, None, :]
        cache_q_sin = cache_sin[:, :, None, :]
        query_roped[cache_tile_t, cache_tile_h, :, 0, :] = (
            (cache_q_lo * cache_q_cos - cache_q_hi * cache_q_sin).to(torch.bfloat16)
            * attention_split_qk_scale
        ).to(torch.bfloat16)
        query_roped[cache_tile_t, cache_tile_h, :, 1, :] = (
            (cache_q_hi * cache_q_cos + cache_q_lo * cache_q_sin).to(torch.bfloat16)
            * attention_split_qk_scale
        ).to(torch.bfloat16)
        cache_value_value = cache_value[cache_tile_t, cache_tile_h, :]
        cache_slot = cache_slot_mapping[cache_token]
        cache_block_index = (cache_slot // cache_block_size)[:, None, None]
        cache_offset = (cache_slot % cache_block_size)[:, None, None]
        hl.store(
            cache_kv_cache,
            [
                cache_block_index,
                cache_offset,
                cache_cache_head[None, :, None],
                cache_lo[None, None, :],
            ],
            (cache_k_lo * cache_cos - cache_k_hi * cache_sin).to(torch.bfloat16),
        )
        hl.store(
            cache_kv_cache,
            [
                cache_block_index,
                cache_offset,
                cache_cache_head[None, :, None],
                (cache_lo + qk_embed_dim)[None, None, :],
            ],
            (cache_k_hi * cache_cos + cache_k_lo * cache_sin).to(torch.bfloat16),
        )
        hl.store(
            cache_kv_cache,
            [
                cache_block_index,
                cache_offset,
                cache_cache_head[None, :, None],
                (hl.arange(cache_head_dim) + cache_head_dim)[None, None, :],
            ],
            cache_value_value,
        )
    # self_attn.attn: split paged attention, all T query tokens of a KV head per
    # task (T * q_per_kv rows), over the cached rows and the new tokens'
    # (causal).
    for attention_split_tile_split, attention_split_tile_bg in hl.tile(
        [attention_split_splits, attention_split_num_kv_heads], block_size=[1, 1]
    ):
        attention_split_split_idx = attention_split_tile_split.begin
        attention_split_g = attention_split_tile_bg.begin
        attention_split_m_i = hl.full(
            [attention_split_rows], -3.4028234663852886e38, dtype=torch.float32
        )
        attention_split_l_i = hl.full([attention_split_rows], 1.0, dtype=torch.float32)
        attention_split_acc = hl.zeros(
            [attention_split_rows, attention_split_head_dim], dtype=torch.float32
        )
        attention_split_seq = context_lens[0]
        # The splits divide the sequence's cache pages evenly; a task streams
        # its pages whole (one box per page and operand, at the page's
        # physical block).
        attention_split_pages = (
            attention_split_seq + attention_split_block_size - 1
        ) // attention_split_block_size
        attention_split_split_pages = (
            attention_split_pages + attention_split_splits - 1
        ) // attention_split_splits
        attention_split_lo_p = attention_split_split_idx * attention_split_split_pages
        attention_split_hi_p = torch.minimum(
            attention_split_lo_p + attention_split_split_pages, attention_split_pages
        )
        # Row (token, head) r sees the keys up to its token's position.
        attention_split_last_n = (
            attention_split_seq
            - attention_split_num_tokens
            + hl.arange(attention_split_rows) // attention_split_q_per_kv
        )
        attention_split_q = query_roped_rows[:, attention_split_g, :, :].reshape(
            attention_split_rows, attention_split_head_dim
        )
        # The split's block-table entries (row 0, the sequence) in one load,
        # so the pages' loads do not each wait for their entry; ids past the
        # split read its last page's entry.
        attention_split_page_ids = attention_split_lo_p + hl.arange(
            attention_split_max_pages
        )
        attention_split_blocks = attention_split_block_table[
            0, torch.minimum(attention_split_page_ids, attention_split_pages - 1)
        ]
        if attention_split_lo_p < attention_split_pages:
            for attention_split_tile_p in hl.tile(
                attention_split_lo_p, attention_split_hi_p, block_size=1
            ):
                attention_split_physical_block = torch.sum(
                    torch.where(
                        attention_split_page_ids == attention_split_tile_p.begin,
                        attention_split_blocks,
                        0,
                    )
                )
                attention_split_n = attention_split_tile_p.begin * (
                    attention_split_block_size
                ) + hl.arange(attention_split_block_size)
                # Keys and queries in the same (neox halves) order.
                attention_split_k = attention_split_kv_halves[
                    attention_split_physical_block, :, attention_split_g, 0, :
                ]
                attention_split_scores = hl.dot(
                    attention_split_q, attention_split_k.T, out_dtype=torch.float32
                )
                # Causal; the page's rows past the sequence are past every
                # row's token.
                attention_split_scores = torch.where(
                    attention_split_n[None, :] <= attention_split_last_n[:, None],
                    attention_split_scores,
                    -3.4028234663852886e38,
                )
                attention_split_m_ij = torch.maximum(
                    attention_split_m_i, torch.amax(attention_split_scores, -1)
                )
                attention_split_p = torch.exp2(
                    attention_split_scores - attention_split_m_ij[:, None]
                )
                attention_split_alpha = torch.exp2(
                    attention_split_m_i - attention_split_m_ij
                )
                attention_split_l_i = attention_split_l_i * attention_split_alpha + (
                    torch.sum(attention_split_p, -1)
                )
                attention_split_v = attention_split_kv_halves[
                    attention_split_physical_block, :, attention_split_g, 1, :
                ]
                # P is 0 there, but a stale non-finite value would still leak.
                attention_split_v = torch.where(
                    (attention_split_n < attention_split_seq)[:, None],
                    attention_split_v,
                    0,
                )
                attention_split_acc = hl.dot(
                    attention_split_p.to(attention_split_v.dtype),
                    attention_split_v,
                    acc=attention_split_acc * attention_split_alpha[:, None],
                )
                attention_split_m_i = attention_split_m_ij
        partial_out[attention_split_tile_split, :, attention_split_tile_bg, :, :] = (
            attention_split_acc / attention_split_l_i[:, None]
        ).reshape(
            attention_split_num_tokens,
            attention_split_q_per_kv,
            attention_split_head_dim,
        )[None, :, None, :, :]
        partial_lse[attention_split_tile_split, :, attention_split_tile_bg, :] = (
            attention_split_m_i + torch.log2(attention_split_l_i)
        ).reshape(attention_split_num_tokens, attention_split_q_per_kv)[
            None, :, None, :
        ]
    # self_attn.attn: split-output merge; one task per (token, kv head).
    for merge_tile_t, merge_tile_kvh, merge_tile_q in hl.tile(
        [
            attention_split_num_tokens,
            attention_split_num_kv_heads,
            attention_split_q_per_kv,
        ],
        block_size=[1, 1, None],
    ):
        merge_m = hl.full(
            [merge_tile_kvh, merge_tile_q],
            -3.4028234663852886e38,
            dtype=torch.float32,
        )
        merge_l = hl.zeros([merge_tile_kvh, merge_tile_q], dtype=torch.float32)
        merge_acc = hl.zeros(
            [merge_tile_kvh, merge_tile_q, attention_merge_head_dim],
            dtype=torch.float32,
        )
        # Every split (an empty one holds lse = -3.4e38, out = 0).
        for merge_tile_s in hl.tile(attention_split_splits, block_size=4):
            merge_lse = partial_lse[
                merge_tile_s, merge_tile_t.begin, merge_tile_kvh, merge_tile_q
            ]
            merge_m_new = torch.maximum(merge_m, torch.amax(merge_lse, 0))
            merge_w = torch.exp2(merge_lse - merge_m_new[None, :, :])
            merge_alpha = torch.exp2(merge_m - merge_m_new)
            merge_l = merge_l * merge_alpha + torch.sum(merge_w, 0)
            merge_values = partial_out[
                merge_tile_s, merge_tile_t.begin, merge_tile_kvh, merge_tile_q, :
            ]
            merge_acc = merge_acc * merge_alpha[:, :, None] + torch.sum(
                merge_values * merge_w[:, :, :, None], 0
            )
            merge_m = merge_m_new
        attention[merge_tile_t, merge_tile_kvh, merge_tile_q, :] = (
            merge_acc / merge_l[:, :, None]
        ).to(attention.dtype)[None, :, :, :]
    # self_attn.o_proj.
    for o_mm_tile_m, o_mm_tile_chunk, o_mm_tile_n in hl.tile(
        [o_mm_m, o_chunks, o_mm_n], block_size=[None, 1, None]
    ):
        o_mm_acc = hl.zeros([o_mm_tile_n, o_mm_tile_m], dtype=torch.float32)
        for o_mm_tile_local_k in hl.tile(o_chunk_k):
            o_mm_acc = hl.dot(
                o_weight_chunked[o_mm_tile_n, o_mm_tile_chunk.begin, o_mm_tile_local_k],
                attention_chunked[
                    o_mm_tile_m, o_mm_tile_chunk.begin, o_mm_tile_local_k
                ].T,
                acc=o_mm_acc,
            )
        o_partials[o_mm_tile_m, o_mm_tile_chunk, o_mm_tile_n] = o_mm_acc.T[:, None, :]
    # post_attention_layernorm: residual add and RMS partials.
    for post_partial_m, post_partial_n in hl.tile(
        [post_num_tokens, post_hidden_size],
        block_size=[post_num_tokens, post_group_size],
    ):
        post_scale = post_norm_weight[post_partial_n][None, :].to(torch.float32)
        post_residual_values = post_residual[post_partial_m, post_partial_n].to(
            torch.float32
        )
        # The partials one K partition at a time: narrow boxes keep the ring's
        # slots small.
        post_attention = hl.zeros([post_partial_m, post_partial_n], dtype=torch.float32)
        for post_tile_chunk in hl.tile(o_chunks, block_size=1):
            post_attention = (
                post_attention
                + o_partials[post_partial_m, post_tile_chunk.begin, post_partial_n]
            )
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
    # mlp.gate_up_proj (K partitions): partial sums.
    for w13_tile_m, w13_tile_chunk, w13_tile_n in hl.tile(
        [w13_m, w13_chunks, w13_n], block_size=[None, 1, None]
    ):
        w13_acc = hl.zeros([w13_tile_n, w13_tile_m], dtype=torch.float32)
        for w13_tile_local_k in hl.tile(w13_chunk_k):
            w13_acc = hl.dot(
                w13_kchunked[w13_tile_n, w13_tile_chunk.begin, w13_tile_local_k],
                ffn_chunked[w13_tile_m, w13_tile_chunk.begin, w13_tile_local_k].T,
                acc=w13_acc,
            )
        w13_partials[w13_tile_m, w13_tile_chunk, w13_tile_n] = w13_acc.T[:, None, :]
    # mlp.act_fn: sums the partials, applies the RMSNorm scale
    # (W (x r g) = r W (x g)) and SiLU-and-multiply.
    for activation_tile_m, activation_tile_c, activation_tile_i in hl.tile(
        [activation_m, w2_chunks, w2_chunk_k],
        block_size=[activation_m, 1, activation_group_size],
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
        activation_gate = hl.zeros(
            [activation_tile_m, activation_tile_c, activation_tile_i],
            dtype=torch.float32,
        )
        activation_up = hl.zeros(
            [activation_tile_m, activation_tile_c, activation_tile_i],
            dtype=torch.float32,
        )
        # One K partition at a time (narrow ring slots).
        for activation_tile_k in hl.tile(w13_chunks, block_size=1):
            activation_gate = (
                activation_gate
                + w13_partials_act[
                    activation_tile_m,
                    activation_tile_k.begin,
                    0,
                    activation_tile_c,
                    activation_tile_i,
                ]
            )
            activation_up = (
                activation_up
                + w13_partials_act[
                    activation_tile_m,
                    activation_tile_k.begin,
                    1,
                    activation_tile_c,
                    activation_tile_i,
                ]
            )
        activation_gate = (activation_gate * post_inv_rms[:, None, None]).to(
            torch.bfloat16
        )
        activation_up = (activation_up * post_inv_rms[:, None, None]).to(torch.bfloat16)
        gate_up_act[activation_tile_m, 0, activation_tile_c, activation_tile_i] = (
            activation_gate
        )
        gate_up_act[activation_tile_m, 1, activation_tile_c, activation_tile_i] = (
            activation_up
        )
        activation_gate = activation_gate.to(torch.float32)
        activation_chunked[activation_tile_m, activation_tile_c, activation_tile_i] = (
            activation_gate * torch.sigmoid(activation_gate)
        ).to(torch.bfloat16) * activation_up.to(torch.bfloat16)
    # mlp.down_proj (K partitions).
    for w2_tile_m, w2_tile_chunk, w2_tile_n in hl.tile(
        [w2_m, w2_chunks, w2_n], block_size=[None, 1, None]
    ):
        w2_acc = hl.zeros([w2_tile_n, w2_tile_m], dtype=torch.float32)
        for w2_tile_local_k in hl.tile(w2_chunk_k):
            w2_acc = hl.dot(
                w2_chunked[w2_tile_n, w2_tile_chunk.begin, w2_tile_local_k],
                activation_chunked[w2_tile_m, w2_tile_chunk.begin, w2_tile_local_k].T,
                acc=w2_acc,
            )
        w2_partials[w2_tile_m, w2_tile_chunk, w2_tile_n] = w2_acc.T[:, None, :]
    for w2_final_m, w2_final_n in hl.tile([w2_m, w2_n], block_size=[1, None]):
        # A full slice over the partitions keeps the read affine: each output
        # tile waits only on its own down_proj tiles.
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


LAYER = Layer(
    hidden=HIDDEN,
    intermediate=INTERMEDIATE,
    q_heads=Q_HEADS,
    kv_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    context=CONTEXT,
    cache_block=CACHE_BLOCK,
    attention_splits=ATTENTION_SPLITS,
    group=GROUP,
    eps=EPS,
    rope_theta=ROPE_THETA,
    qk_norm=True,
)


def _make_inputs(
    seed: int = 0, context_len: int = 1024, tokens: int = TOKENS, init: bool = True
) -> dict[str, torch.Tensor]:
    """One sequence: ``context_len`` cached rows, then ``tokens`` new tokens
    (``init=False``: uninitialized, for code generation)."""
    return make_inputs(LAYER, seed, context_len, tokens, init)


def _kernel_args(tensors: dict[str, torch.Tensor]) -> tuple[object, ...]:
    return kernel_args(LAYER, tensors)


def correctness_check(
    tokens: int = TOKENS, context_lens: tuple[int, ...] = (5, 1001)
) -> None:
    """Check the outputs and the KV rows written to the cache against an FP32
    PyTorch reference.

    The default context lengths give a sequence of fewer cache blocks than
    attention splits (most splits empty) and one whose last block is partly filled.
    """
    check(LAYER, qwen3_verify_layer_bf16, tokens, context_lens)


if __name__ == "__main__":
    for window in (8, 16):
        correctness_check(window)
