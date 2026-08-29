# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""Qwen3-8B batch-one decode-layer megakernel, pretuned for NVIDIA B200.

The single Helion function's top-level tile loops implement residual RMSNorm
and FP8 quantization, QKV projection, Q/K norm and RoPE, KV-cache update, split
paged attention and merge, output projection, and the complete gated FFN.  The
benchmark fixes the production decode shape and checks it against the
corresponding compiled vLLM decoder layer with its default backend selection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


BATCH = 1
HIDDEN = 4096
INTERMEDIATE = 12288
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
CONTEXT = 8192
CACHE_BLOCK = 16
ATTENTION_SPLITS = 128
GROUP = 128
EPS = 1e-6
FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)

QWEN3_8B_FP8_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": HEAD_DIM,
    "hidden_act": "silu",
    "hidden_size": HIDDEN,
    "initializer_range": 0.02,
    "intermediate_size": INTERMEDIATE,
    "max_position_embeddings": 40960,
    "max_window_layers": 36,
    "model_type": "qwen3",
    "num_attention_heads": Q_HEADS,
    "num_hidden_layers": 36,
    "num_key_value_heads": KV_HEADS,
    "rms_norm_eps": EPS,
    "rope_scaling": None,
    "rope_theta": 1_000_000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [GROUP, GROUP],
    },
}


@helion.aot_kernel(static_shapes=True, backend="triton")
def qwen3_decode_layer(
    hidden_states,
    residual,
    pre_weight,
    pre_q,
    pre_scale,
    qkv_weight_q,
    qkv_weight_scale,
    q_weight,
    k_weight,
    cos_sin,
    position,
    kv_cache,
    block_table,
    slot_mapping,
    o_weight_q,
    o_weight_scale,
    attention_q,
    attention_scale,
    post_weight,
    ffn_q,
    ffn_scale,
    w13_q,
    w13_scale,
    w2_q,
    w2_scale,
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
    __td_pre_result = pre_q
    __td_pre_input = hidden_states
    __td_pre_weight = pre_weight
    __td_pre_scale = pre_scale
    __td_pre_epsilon = eps
    __td_pre_scale_ub = None
    __td_pre_residual = residual
    __td_pre_group_size = group
    __td_pre_is_scale_transposed = False
    assert __td_pre_input.ndim == 2
    __td_pre_num_tokens, __td_pre_hidden_size = __td_pre_input.shape
    hl.specialize(__td_pre_hidden_size)
    hl.specialize(__td_pre_group_size)
    __td_pre_groups_per_row = __td_pre_scale.shape[1]
    hl.specialize(__td_pre_groups_per_row)
    assert __td_pre_group_size == 128
    assert __td_pre_result.dtype == torch.float8_e4m3fn
    assert __td_pre_scale.dtype == torch.float32
    __td_pre_rms_partials = torch.empty(
        (__td_pre_num_tokens, __td_pre_groups_per_row),
        dtype=torch.float32,
        device=__td_pre_input.device,
    )
    __td_pre_unrounded_values = torch.empty_like(__td_pre_input, dtype=torch.float32)
    __td_qkv_mm_activation_q = pre_q
    __td_qkv_mm_activation_scale = pre_scale
    __td_qkv_mm_weight_q = qkv_weight_q
    __td_qkv_mm_weight_scale = qkv_weight_scale
    __td_qkv_mm_group_size = group
    __td_qkv_mm_m, __td_qkv_mm_k = __td_qkv_mm_activation_q.size()
    __td_qkv_mm_n, __td_qkv_mm_weight_k = __td_qkv_mm_weight_q.size()
    assert __td_qkv_mm_weight_k == __td_qkv_mm_k
    assert __td_qkv_mm_group_size == 128
    hl.specialize(__td_qkv_mm_group_size)
    qkv = torch.empty(
        (__td_qkv_mm_m, __td_qkv_mm_n),
        dtype=torch.bfloat16,
        device=__td_qkv_mm_activation_q.device,
    )
    batch = hidden_states.shape[0]
    query = qkv[:, : q_heads * head_dim].view(batch, q_heads, head_dim)
    key_begin = q_heads * head_dim
    key = qkv[:, key_begin : key_begin + kv_heads * head_dim].view(
        batch, kv_heads, head_dim
    )
    value = qkv[
        :, key_begin + kv_heads * head_dim : (q_heads + 2 * kv_heads) * head_dim
    ].view(batch, kv_heads, head_dim)
    __td_qk_qkv = qkv
    __td_qk_num_heads_q = q_heads
    __td_qk_num_heads_k = kv_heads
    __td_qk_num_heads_v = kv_heads
    __td_qk_head_dim = head_dim
    __td_qk_eps = eps
    __td_qk_q_weight = q_weight
    __td_qk_k_weight = k_weight
    __td_qk_cos_sin_cache = cos_sin
    __td_qk_is_neox = True
    __td_qk_position_ids = position
    __td_qk_forced_token_heads_per_warp = -1
    __td_qk_num_tokens = __td_qk_qkv.shape[0]
    __td_qk_total_heads = (
        __td_qk_num_heads_q + __td_qk_num_heads_k + __td_qk_num_heads_v
    )
    hl.specialize(__td_qk_qkv.shape[1])
    __td_qk__, __td_qk_rotary_dim = __td_qk_cos_sin_cache.shape
    hl.specialize(__td_qk_rotary_dim)
    __td_qk_embed_dim = __td_qk_rotary_dim // 2
    hl.specialize(__td_qk_num_heads_q)
    hl.specialize(__td_qk_num_heads_k)
    hl.specialize(__td_qk_num_heads_v)
    hl.specialize(__td_qk_head_dim)
    __td_qk_qk_heads = __td_qk_num_heads_q + __td_qk_num_heads_k
    __td_qk_qkv = __td_qk_qkv.view(
        __td_qk_num_tokens, __td_qk_total_heads, __td_qk_head_dim
    )
    __td_cache_key = key
    __td_cache_value = value
    __td_cache_kv_cache = kv_cache
    __td_cache_slot_mapping = slot_mapping
    __td_cache_block_size = cache_block
    __td_cache_num_tokens, __td_cache_num_kv_heads, __td_cache_head_dim = (
        __td_cache_key.shape
    )
    hl.specialize(__td_cache_num_kv_heads)
    hl.specialize(__td_cache_head_dim)
    hl.specialize(__td_cache_block_size)
    __td_attention_split_query = query
    __td_attention_split_kv_cache = kv_cache
    __td_attention_split_block_table = block_table
    __td_attention_split_context = context
    __td_attention_split_block_size = cache_block
    __td_attention_split_q_per_kv = q_heads // kv_heads
    __td_attention_split_splits = attention_splits
    (
        __td_attention_split_num_tokens,
        __td_attention_split_num_q_heads,
        __td_attention_split_head_dim,
    ) = __td_attention_split_query.shape
    __td_attention_split_num_kv_heads = __td_attention_split_kv_cache.shape[2]
    assert (
        __td_attention_split_num_q_heads
        == __td_attention_split_num_kv_heads * __td_attention_split_q_per_kv
    )
    assert __td_attention_split_context % __td_attention_split_splits == 0
    hl.specialize(__td_attention_split_head_dim)
    hl.specialize(__td_attention_split_num_kv_heads)
    hl.specialize(__td_attention_split_q_per_kv)
    hl.specialize(__td_attention_split_context)
    hl.specialize(__td_attention_split_block_size)
    hl.specialize(__td_attention_split_splits)
    __td_attention_split_split_context = (
        __td_attention_split_context // __td_attention_split_splits
    )
    __td_attention_split_token_kv_heads = (
        __td_attention_split_num_tokens * __td_attention_split_num_kv_heads
    )
    partial_out = torch.empty(
        (
            __td_attention_split_splits,
            __td_attention_split_token_kv_heads,
            __td_attention_split_q_per_kv,
            __td_attention_split_head_dim,
        ),
        device=__td_attention_split_query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (
            __td_attention_split_splits,
            __td_attention_split_token_kv_heads,
            __td_attention_split_q_per_kv,
        ),
        device=__td_attention_split_query.device,
        dtype=torch.float32,
    )
    __td_attention_split_qk_scale = (
        1.0 / math.sqrt(__td_attention_split_head_dim) * 1.44269504
    )
    __td_attention_merge_partial_out = partial_out
    __td_attention_merge_partial_lse = partial_lse
    (
        __td_attention_merge_splits,
        __td_attention_merge_num_kv_heads,
        __td_attention_merge_q_per_kv,
        __td_attention_merge_head_dim,
    ) = __td_attention_merge_partial_out.shape
    hl.specialize(__td_attention_merge_splits)
    hl.specialize(__td_attention_merge_num_kv_heads)
    hl.specialize(__td_attention_merge_q_per_kv)
    hl.specialize(__td_attention_merge_head_dim)
    __td_attention_merge_query_heads = (
        __td_attention_merge_num_kv_heads * __td_attention_merge_q_per_kv
    )
    hl.specialize(__td_attention_merge_query_heads)
    __td_attention_merge_merge_chunks = 16
    hl.specialize(__td_attention_merge_merge_chunks)
    assert __td_attention_merge_splits % __td_attention_merge_merge_chunks == 0
    __td_attention_merge_splits_per_chunk = (
        __td_attention_merge_splits // __td_attention_merge_merge_chunks
    )
    hl.specialize(__td_attention_merge_splits_per_chunk)
    __td_attention_merge_partial_out_flat = __td_attention_merge_partial_out.view(
        __td_attention_merge_splits,
        __td_attention_merge_query_heads,
        __td_attention_merge_head_dim,
    )
    __td_attention_merge_partial_lse_flat = __td_attention_merge_partial_lse.view(
        __td_attention_merge_splits, __td_attention_merge_query_heads
    )
    __td_attention_merge_partial_out_storage = (
        __td_attention_merge_partial_out_flat.view(-1)
    )
    __td_attention_merge_partial_lse_storage = (
        __td_attention_merge_partial_lse_flat.view(-1)
    )
    __td_attention_merge_chunk_out = torch.empty(
        (
            __td_attention_merge_merge_chunks,
            __td_attention_merge_query_heads,
            __td_attention_merge_head_dim,
        ),
        dtype=torch.float32,
        device=__td_attention_merge_partial_out.device,
    )
    __td_attention_merge_chunk_lse = torch.empty(
        (__td_attention_merge_merge_chunks, __td_attention_merge_query_heads),
        dtype=torch.float32,
        device=__td_attention_merge_partial_out.device,
    )
    __td_attention_merge_chunk_out_storage = __td_attention_merge_chunk_out.view(-1)
    __td_attention_merge_chunk_lse_storage = __td_attention_merge_chunk_lse.view(-1)
    attention = torch.empty(
        (__td_attention_merge_query_heads, __td_attention_merge_head_dim),
        dtype=torch.bfloat16,
        device=__td_attention_merge_partial_out.device,
    )
    attention_flat = attention.view(batch, hidden)
    __td_attention_quant_input = attention_flat
    __td_attention_quant_output_q = attention_q
    __td_attention_quant_output_s = attention_scale
    __td_attention_quant_group_size = group
    __td_attention_quant_eps = 1e-10
    __td_attention_quant_fp8_min = FP8_MIN
    __td_attention_quant_fp8_max = FP8_MAX
    __td_attention_quant_scale_ue8m0 = False
    __td_attention_quant_dummy_is_scale_transposed = False
    __td_attention_quant_dummy_is_tma_aligned = False
    __td_attention_quant_num_tokens, __td_attention_quant_hidden_size = (
        __td_attention_quant_input.shape
    )
    hl.specialize(__td_attention_quant_hidden_size)
    hl.specialize(__td_attention_quant_group_size)
    __td_attention_quant_groups_per_row = __td_attention_quant_output_s.shape[1]
    hl.specialize(__td_attention_quant_groups_per_row)
    __td_attention_quant_input = __td_attention_quant_input.view(
        __td_attention_quant_num_tokens,
        __td_attention_quant_groups_per_row,
        __td_attention_quant_group_size,
    )
    __td_attention_quant_output_q = __td_attention_quant_output_q.view(
        __td_attention_quant_num_tokens,
        __td_attention_quant_groups_per_row,
        __td_attention_quant_group_size,
    )
    __td_o_mm_activation_q = attention_q
    __td_o_mm_activation_scale = attention_scale
    __td_o_mm_weight_q = o_weight_q
    __td_o_mm_weight_scale = o_weight_scale
    __td_o_mm_group_size = group
    __td_o_mm_m, __td_o_mm_k = __td_o_mm_activation_q.size()
    __td_o_mm_n, __td_o_mm_weight_k = __td_o_mm_weight_q.size()
    assert __td_o_mm_weight_k == __td_o_mm_k
    assert __td_o_mm_group_size == 128
    hl.specialize(__td_o_mm_group_size)
    attention_out = torch.empty(
        (__td_o_mm_m, __td_o_mm_n),
        dtype=torch.bfloat16,
        device=__td_o_mm_activation_q.device,
    )
    __td_post_result = ffn_q
    __td_post_input = attention_out
    __td_post_weight = post_weight
    __td_post_scale = ffn_scale
    __td_post_epsilon = eps
    __td_post_scale_ub = None
    __td_post_residual = residual
    __td_post_group_size = group
    __td_post_is_scale_transposed = False
    assert __td_post_input.ndim == 2
    __td_post_num_tokens, __td_post_hidden_size = __td_post_input.shape
    hl.specialize(__td_post_hidden_size)
    hl.specialize(__td_post_group_size)
    __td_post_groups_per_row = __td_post_scale.shape[1]
    hl.specialize(__td_post_groups_per_row)
    assert __td_post_group_size == 128
    assert __td_post_result.dtype == torch.float8_e4m3fn
    assert __td_post_scale.dtype == torch.float32
    __td_post_rms_partials = torch.empty(
        (__td_post_num_tokens, __td_post_groups_per_row),
        dtype=torch.float32,
        device=__td_post_input.device,
    )
    __td_post_unrounded_values = torch.empty_like(__td_post_input, dtype=torch.float32)
    __td_w13_activation_q = ffn_q
    __td_w13_activation_scale = ffn_scale
    __td_w13_weight_q = w13_q
    __td_w13_weight_scale = w13_scale
    __td_w13_group_size = group
    __td_w13_m, __td_w13_k = __td_w13_activation_q.size()
    __td_w13_n, __td_w13_weight_k = __td_w13_weight_q.size()
    assert __td_w13_weight_k == __td_w13_k
    assert __td_w13_group_size == 128
    hl.specialize(__td_w13_group_size)
    gate_up = torch.empty(
        (__td_w13_m, __td_w13_n),
        dtype=torch.bfloat16,
        device=__td_w13_activation_q.device,
    )
    __td_activation_gate_up = gate_up
    __td_activation_group_size = group
    __td_activation_m, __td_activation_twice_intermediate = (
        __td_activation_gate_up.size()
    )
    __td_activation_intermediate = __td_activation_twice_intermediate // 2
    hl.specialize(__td_activation_group_size)
    __td_activation_groups = __td_activation_intermediate // __td_activation_group_size
    activation_q = torch.empty(
        (__td_activation_m, __td_activation_intermediate),
        dtype=torch.float8_e4m3fn,
        device=__td_activation_gate_up.device,
    )
    activation_scale = torch.empty(
        (__td_activation_m, __td_activation_groups),
        dtype=torch.float32,
        device=__td_activation_gate_up.device,
    )
    __td_w2_activation_q = activation_q
    __td_w2_activation_scale = activation_scale
    __td_w2_weight_q = w2_q
    __td_w2_weight_scale = w2_scale
    __td_w2_group_size = group
    __td_w2_m, __td_w2_k = __td_w2_activation_q.size()
    __td_w2_n, __td_w2_weight_k = __td_w2_weight_q.size()
    assert __td_w2_weight_k == __td_w2_k
    assert __td_w2_group_size == 128
    hl.specialize(__td_w2_group_size)
    output = torch.empty(
        (__td_w2_m, __td_w2_n), dtype=torch.bfloat16, device=__td_w2_activation_q.device
    )
    for __td_pre_partial_m, __td_pre_partial_n in hl.tile(
        [__td_pre_num_tokens, __td_pre_hidden_size], block_size=[1, __td_pre_group_size]
    ):
        __td_pre_partial_values = __td_pre_input[
            __td_pre_partial_m, __td_pre_partial_n
        ].to(torch.float32)
        if __td_pre_residual is not None:
            __td_pre_partial_values = (
                __td_pre_partial_values
                + __td_pre_residual[__td_pre_partial_m, __td_pre_partial_n]
            )
            __td_pre_residual[__td_pre_partial_m, __td_pre_partial_n] = (
                __td_pre_partial_values.to(__td_pre_residual.dtype)
            )
        __td_pre_unrounded_values[__td_pre_partial_m, __td_pre_partial_n] = (
            __td_pre_partial_values
        )
        __td_pre_rms_partials[__td_pre_partial_m, __td_pre_partial_n.id] = torch.sum(
            __td_pre_partial_values * __td_pre_partial_values, dim=-1
        )
    for __td_pre_quant_m, __td_pre_quant_g, __td_pre_quant_n in hl.tile(
        [__td_pre_num_tokens, __td_pre_groups_per_row, __td_pre_group_size],
        block_size=[1, 1, __td_pre_group_size],
    ):
        __td_pre_quant_m_idx = __td_pre_quant_m.begin + hl.arange(
            __td_pre_quant_m.block_size
        )
        __td_pre_quant_group_idx = __td_pre_quant_g.index
        __td_pre_quant_n_idx = (
            __td_pre_quant_group_idx[:, None] * __td_pre_group_size
            + __td_pre_quant_n.index[None, :]
        )
        __td_pre_quant_m_blk = __td_pre_quant_m_idx[:, None, None]
        __td_pre_quant_n_blk = __td_pre_quant_n_idx[None, :, :]
        __td_pre_square_sum = hl.zeros([__td_pre_quant_m], dtype=torch.float32)
        for __td_pre_reduce_g in hl.tile(__td_pre_groups_per_row, block_size=1):
            __td_pre_square_sum = __td_pre_square_sum + torch.sum(
                __td_pre_rms_partials[__td_pre_quant_m, __td_pre_reduce_g], dim=-1
            )
        __td_pre_inv_rms = torch.rsqrt(
            __td_pre_square_sum * (1.0 / __td_pre_hidden_size) + __td_pre_epsilon
        )
        __td_pre_quant_values = __td_pre_unrounded_values[
            __td_pre_quant_m_blk, __td_pre_quant_n_blk
        ]
        __td_pre_normalized = (
            __td_pre_quant_values * __td_pre_inv_rms[:, None, None]
        ).to(torch.bfloat16) * __td_pre_weight[__td_pre_quant_n_blk]
        __td_pre_quant_scale = torch.amax(torch.abs(__td_pre_normalized), dim=-1).to(
            torch.float32
        )
        if __td_pre_scale_ub is not None:
            __td_pre_quant_scale = __td_pre_quant_scale.clamp(
                max=hl.load(__td_pre_scale_ub, [])
            )
        __td_pre_quant_scale = (__td_pre_quant_scale / FP8_MAX).clamp(min=FP8_MIN_SCALE)
        __td_pre_scale[__td_pre_quant_m, __td_pre_quant_g] = __td_pre_quant_scale
        __td_pre_result[__td_pre_quant_m_blk, __td_pre_quant_n_blk] = (
            (__td_pre_normalized / __td_pre_quant_scale[:, :, None])
            .clamp(FP8_MIN, FP8_MAX)
            .to(__td_pre_result.dtype)
        )
    for __td_qkv_mm_tile_m, __td_qkv_mm_tile_n in hl.tile(
        [__td_qkv_mm_m, __td_qkv_mm_n], block_size=[1, None]
    ):
        __td_qkv_mm_acc = hl.zeros(
            [__td_qkv_mm_tile_m, __td_qkv_mm_tile_n], dtype=torch.float32
        )
        for __td_qkv_mm_tile_k in hl.tile(
            __td_qkv_mm_k, block_size=__td_qkv_mm_group_size
        ):
            __td_qkv_mm_partial = hl.dot(
                __td_qkv_mm_activation_q[__td_qkv_mm_tile_m, __td_qkv_mm_tile_k],
                __td_qkv_mm_weight_q[__td_qkv_mm_tile_n, __td_qkv_mm_tile_k].T,
            ).to(torch.float32)
            __td_qkv_mm_a_scale = __td_qkv_mm_activation_scale[
                __td_qkv_mm_tile_m, __td_qkv_mm_tile_k.id
            ].to(torch.float32)
            __td_qkv_mm_w_scale = __td_qkv_mm_weight_scale[
                __td_qkv_mm_tile_n.index // __td_qkv_mm_group_size,
                __td_qkv_mm_tile_k.id,
            ].to(torch.float32)
            __td_qkv_mm_acc = (
                __td_qkv_mm_acc
                + __td_qkv_mm_partial
                * __td_qkv_mm_a_scale[:, None]
                * __td_qkv_mm_w_scale[None, :]
            )
        qkv[__td_qkv_mm_tile_m, __td_qkv_mm_tile_n] = __td_qkv_mm_acc.to(qkv.dtype)
    for __td_qk_tile_m, __td_qk_tile_gn, __td_qk_tile_n in hl.tile(
        [__td_qk_num_tokens, __td_qk_qk_heads, __td_qk_head_dim],
        block_size=[1, None, __td_qk_head_dim],
    ):
        __td_qk_x = __td_qk_qkv[__td_qk_tile_m, __td_qk_tile_gn, __td_qk_tile_n].to(
            torch.float32
        )
        __td_qk_rms = torch.rsqrt(
            __td_qk_x.pow(2).sum(-1) * (1.0 / __td_qk_head_dim) + __td_qk_eps
        )
        __td_qk_use_q = (__td_qk_tile_gn.index < __td_qk_num_heads_q)[None, :, None]
        __td_qk_w = torch.where(
            __td_qk_use_q,
            __td_qk_q_weight[None, None, __td_qk_tile_n],
            __td_qk_k_weight[None, None, __td_qk_tile_n],
        )
        __td_qk_x = (__td_qk_x * __td_qk_rms[:, :, None]).to(
            __td_qk_qkv.dtype
        ) * __td_qk_w
        __td_qk_qkv[__td_qk_tile_m, __td_qk_tile_gn, __td_qk_tile_n] = __td_qk_x
        __td_qk_pos = __td_qk_position_ids[__td_qk_tile_m]
        __td_qk_cos = __td_qk_cos_sin_cache[__td_qk_pos, hl.arange(__td_qk_embed_dim)]
        __td_qk_sin = __td_qk_cos_sin_cache[
            __td_qk_pos, hl.arange(__td_qk_embed_dim) + __td_qk_embed_dim
        ]
        if __td_qk_is_neox:
            __td_qk_x1_offset = hl.arange(__td_qk_embed_dim)
            __td_qk_x2_offset = __td_qk_x1_offset + __td_qk_embed_dim
        else:
            __td_qk_x1_offset = hl.arange(__td_qk_embed_dim) * 2
            __td_qk_x2_offset = __td_qk_x1_offset + 1
        __td_qk_x1 = __td_qk_qkv[__td_qk_tile_m, __td_qk_tile_gn, __td_qk_x1_offset]
        __td_qk_x2 = __td_qk_qkv[__td_qk_tile_m, __td_qk_tile_gn, __td_qk_x2_offset]
        __td_qk_qkv[__td_qk_tile_m, __td_qk_tile_gn, __td_qk_x1_offset] = (
            __td_qk_x1 * __td_qk_cos[:, None, :] - __td_qk_x2 * __td_qk_sin[:, None, :]
        )
        __td_qk_qkv[__td_qk_tile_m, __td_qk_tile_gn, __td_qk_x2_offset] = (
            __td_qk_x2 * __td_qk_cos[:, None, :] + __td_qk_x1 * __td_qk_sin[:, None, :]
        )
    for __td_cache_tile_t, __td_cache_tile_h, __td_cache_tile_d in hl.tile(
        [__td_cache_num_tokens, __td_cache_num_kv_heads, __td_cache_head_dim],
        block_size=[1, 1, __td_cache_head_dim],
    ):
        __td_cache_token = __td_cache_tile_t.index
        __td_cache_cache_head = __td_cache_tile_h.index
        __td_cache_dimension = __td_cache_tile_d.index
        __td_cache_key_value = __td_cache_key[
            __td_cache_tile_t, __td_cache_tile_h, __td_cache_tile_d
        ]
        __td_cache_value_value = __td_cache_value[
            __td_cache_tile_t, __td_cache_tile_h, __td_cache_tile_d
        ]
        __td_cache_slot = __td_cache_slot_mapping[__td_cache_token]
        __td_cache_block = (__td_cache_slot // __td_cache_block_size)[:, None, None]
        __td_cache_offset = (__td_cache_slot % __td_cache_block_size)[:, None, None]
        hl.store(
            __td_cache_kv_cache,
            [
                __td_cache_block,
                __td_cache_offset,
                __td_cache_cache_head[None, :, None],
                __td_cache_dimension[None, None, :],
            ],
            __td_cache_key_value,
        )
        hl.store(
            __td_cache_kv_cache,
            [
                __td_cache_block,
                __td_cache_offset,
                __td_cache_cache_head[None, :, None],
                (__td_cache_dimension + __td_cache_head_dim)[None, None, :],
            ],
            __td_cache_value_value,
        )
    for (
        __td_attention_split_tile_split,
        __td_attention_split_tile_bg,
        __td_attention_split_tile_q,
    ) in hl.tile(
        [
            __td_attention_split_splits,
            __td_attention_split_token_kv_heads,
            __td_attention_split_q_per_kv,
        ],
        block_size=[1, 1, None],
    ):
        __td_attention_split_m_i = hl.full(
            [__td_attention_split_tile_bg, __td_attention_split_tile_q],
            float("-inf"),
            dtype=torch.float32,
        )
        __td_attention_split_l_i = hl.full(
            [__td_attention_split_tile_bg, __td_attention_split_tile_q],
            1.0,
            dtype=torch.float32,
        )
        __td_attention_split_acc = hl.zeros(
            [
                __td_attention_split_tile_bg,
                __td_attention_split_tile_q,
                __td_attention_split_head_dim,
            ],
            dtype=torch.float32,
        )
        __td_attention_split_split_idx = __td_attention_split_tile_split.begin
        __td_attention_split_token = (
            __td_attention_split_tile_bg.index // __td_attention_split_num_kv_heads
        )
        __td_attention_split_kv_head = (
            __td_attention_split_tile_bg.index % __td_attention_split_num_kv_heads
        )
        __td_attention_split_query_head = (
            __td_attention_split_kv_head[:, None] * __td_attention_split_q_per_kv
            + __td_attention_split_tile_q.index[None, :]
        )
        __td_attention_split_q_blk = __td_attention_split_query[
            __td_attention_split_token[:, None],
            __td_attention_split_query_head,
            :,
        ]
        __td_attention_split_q_blk = (
            __td_attention_split_q_blk * __td_attention_split_qk_scale
        ).to(__td_attention_split_query.dtype)
        for __td_attention_split_tile_local_n in hl.tile(
            __td_attention_split_split_context
        ):
            __td_attention_split_n = (
                __td_attention_split_split_idx * __td_attention_split_split_context
                + __td_attention_split_tile_local_n.index
            )
            __td_attention_split_physical_block = __td_attention_split_block_table[
                __td_attention_split_token[:, None],
                (__td_attention_split_n // __td_attention_split_block_size)[None, :],
            ]
            __td_attention_split_block_offset = (
                __td_attention_split_n % __td_attention_split_block_size
            )
            __td_attention_split_d = hl.arange(__td_attention_split_head_dim)
            __td_attention_split_k = hl.load(
                __td_attention_split_kv_cache,
                [
                    __td_attention_split_physical_block[:, :, None],
                    __td_attention_split_block_offset[None, :, None],
                    __td_attention_split_kv_head[:, None, None],
                    __td_attention_split_d[None, None, :],
                ],
            )
            __td_attention_split_scores = torch.bmm(
                __td_attention_split_q_blk,
                __td_attention_split_k.transpose(1, 2),
                torch.float32,
            )
            __td_attention_split_m_ij = torch.maximum(
                __td_attention_split_m_i, torch.amax(__td_attention_split_scores, -1)
            )
            __td_attention_split_p = torch.exp2(
                __td_attention_split_scores - __td_attention_split_m_ij[:, :, None]
            )
            __td_attention_split_alpha = torch.exp2(
                __td_attention_split_m_i - __td_attention_split_m_ij
            )
            __td_attention_split_l_i = (
                __td_attention_split_l_i * __td_attention_split_alpha
                + torch.sum(__td_attention_split_p, -1)
            )
            __td_attention_split_acc = (
                __td_attention_split_acc * __td_attention_split_alpha[:, :, None]
            )
            __td_attention_split_v = hl.load(
                __td_attention_split_kv_cache,
                [
                    __td_attention_split_physical_block[:, :, None],
                    __td_attention_split_block_offset[None, :, None],
                    __td_attention_split_kv_head[:, None, None],
                    (__td_attention_split_d + __td_attention_split_head_dim)[
                        None, None, :
                    ],
                ],
            )
            __td_attention_split_acc = torch.baddbmm(
                __td_attention_split_acc,
                __td_attention_split_p.to(__td_attention_split_v.dtype),
                __td_attention_split_v,
            )
            __td_attention_split_m_i = __td_attention_split_m_ij
        partial_out[
            __td_attention_split_tile_split,
            __td_attention_split_tile_bg,
            __td_attention_split_tile_q,
            :,
        ] = (__td_attention_split_acc / __td_attention_split_l_i[:, :, None])[
            None, :, :, :
        ]
        partial_lse[
            __td_attention_split_tile_split,
            __td_attention_split_tile_bg,
            __td_attention_split_tile_q,
        ] = (__td_attention_split_m_i + torch.log2(__td_attention_split_l_i))[
            None, :, :
        ]
    for __td_attention_merge_tile_chunk, __td_attention_merge_chunk_head in hl.tile(
        [__td_attention_merge_merge_chunks, __td_attention_merge_query_heads],
        block_size=[1, 1],
    ):
        __td_attention_merge_chunk_split_idx = (
            __td_attention_merge_tile_chunk.index[:, None]
            * __td_attention_merge_splits_per_chunk
            + hl.arange(__td_attention_merge_splits_per_chunk)[None, :]
        )
        __td_attention_merge_chunk_lse_offsets = (
            __td_attention_merge_chunk_split_idx[:, :, None]
            * __td_attention_merge_query_heads
            + __td_attention_merge_chunk_head.index[None, None, :]
        )
        __td_attention_merge_chunk_lse_values = (
            __td_attention_merge_partial_lse_storage[
                __td_attention_merge_chunk_lse_offsets
            ]
        )
        __td_attention_merge_chunk_max_lse = torch.amax(
            __td_attention_merge_chunk_lse_values, dim=1
        )
        __td_attention_merge_chunk_weights = torch.exp2(
            __td_attention_merge_chunk_lse_values
            - __td_attention_merge_chunk_max_lse[:, None, :]
        )
        __td_attention_merge_chunk_value_offsets = (
            __td_attention_merge_chunk_lse_offsets[:, :, :, None]
            * __td_attention_merge_head_dim
            + hl.arange(__td_attention_merge_head_dim)[None, None, None, :]
        )
        __td_attention_merge_chunk_values = __td_attention_merge_partial_out_storage[
            __td_attention_merge_chunk_value_offsets
        ]
        __td_attention_merge_chunk_denominator = torch.sum(
            __td_attention_merge_chunk_weights, dim=1
        )
        __td_attention_merge_chunk_merged = torch.sum(
            __td_attention_merge_chunk_values
            * __td_attention_merge_chunk_weights[:, :, :, None],
            dim=1,
        )
        __td_attention_merge_chunk_merged = (
            __td_attention_merge_chunk_merged
            / __td_attention_merge_chunk_denominator[:, :, None]
        )
        __td_attention_merge_chunk_out[
            __td_attention_merge_tile_chunk, __td_attention_merge_chunk_head, :
        ] = __td_attention_merge_chunk_merged
        __td_attention_merge_chunk_lse[
            __td_attention_merge_tile_chunk, __td_attention_merge_chunk_head
        ] = __td_attention_merge_chunk_max_lse + torch.log2(
            __td_attention_merge_chunk_denominator
        )
    for __td_attention_merge_final_head in hl.tile(
        __td_attention_merge_query_heads, block_size=1
    ):
        __td_attention_merge_final_chunk_idx = hl.arange(
            __td_attention_merge_merge_chunks
        )
        __td_attention_merge_final_lse_offsets = (
            __td_attention_merge_final_chunk_idx[:, None]
            * __td_attention_merge_query_heads
            + __td_attention_merge_final_head.index[None, :]
        )
        __td_attention_merge_final_lse_values = __td_attention_merge_chunk_lse_storage[
            __td_attention_merge_final_lse_offsets
        ]
        __td_attention_merge_final_max_lse = torch.amax(
            __td_attention_merge_final_lse_values, dim=0
        )
        __td_attention_merge_final_weights = torch.exp2(
            __td_attention_merge_final_lse_values
            - __td_attention_merge_final_max_lse[None, :]
        )
        __td_attention_merge_final_value_offsets = (
            __td_attention_merge_final_lse_offsets[:, :, None]
            * __td_attention_merge_head_dim
            + hl.arange(__td_attention_merge_head_dim)[None, None, :]
        )
        __td_attention_merge_final_values = __td_attention_merge_chunk_out_storage[
            __td_attention_merge_final_value_offsets
        ]
        __td_attention_merge_final_denominator = torch.sum(
            __td_attention_merge_final_weights, dim=0
        )
        __td_attention_merge_final_merged = torch.sum(
            __td_attention_merge_final_values
            * __td_attention_merge_final_weights[:, :, None],
            dim=0,
        )
        attention[__td_attention_merge_final_head, :] = (
            __td_attention_merge_final_merged
            / __td_attention_merge_final_denominator[:, None]
        ).to(attention.dtype)
    for (
        __td_attention_quant_tile_m,
        __td_attention_quant_tile_gn,
        __td_attention_quant_tile_n,
    ) in hl.tile(
        [
            __td_attention_quant_num_tokens,
            __td_attention_quant_groups_per_row,
            __td_attention_quant_group_size,
        ],
        block_size=[1, None, __td_attention_quant_group_size],
    ):
        __td_attention_quant_x = __td_attention_quant_input[
            __td_attention_quant_tile_m,
            __td_attention_quant_tile_gn,
            __td_attention_quant_tile_n,
        ]
        __td_attention_quant_s = (
            torch.amax(torch.abs(__td_attention_quant_x), dim=-1).clamp(
                min=__td_attention_quant_eps
            )
            / __td_attention_quant_fp8_max
        )
        if __td_attention_quant_scale_ue8m0:
            __td_attention_quant_s = torch.exp2(
                torch.ceil(torch.log2(__td_attention_quant_s))
            )
        __td_attention_quant_output_s[
            __td_attention_quant_tile_m, __td_attention_quant_tile_gn
        ] = __td_attention_quant_s
        __td_attention_quant_output_q[
            __td_attention_quant_tile_m,
            __td_attention_quant_tile_gn,
            __td_attention_quant_tile_n,
        ] = (
            (__td_attention_quant_x / __td_attention_quant_s[:, :, None])
            .clamp(__td_attention_quant_fp8_min, __td_attention_quant_fp8_max)
            .to(__td_attention_quant_output_q.dtype)
        )
    for __td_o_mm_tile_m, __td_o_mm_tile_n in hl.tile(
        [__td_o_mm_m, __td_o_mm_n], block_size=[1, None]
    ):
        __td_o_mm_acc = hl.zeros(
            [__td_o_mm_tile_m, __td_o_mm_tile_n], dtype=torch.float32
        )
        for __td_o_mm_tile_k in hl.tile(__td_o_mm_k, block_size=__td_o_mm_group_size):
            __td_o_mm_partial = hl.dot(
                __td_o_mm_activation_q[__td_o_mm_tile_m, __td_o_mm_tile_k],
                __td_o_mm_weight_q[__td_o_mm_tile_n, __td_o_mm_tile_k].T,
            ).to(torch.float32)
            __td_o_mm_a_scale = __td_o_mm_activation_scale[
                __td_o_mm_tile_m, __td_o_mm_tile_k.id
            ].to(torch.float32)
            __td_o_mm_w_scale = __td_o_mm_weight_scale[
                __td_o_mm_tile_n.index // __td_o_mm_group_size, __td_o_mm_tile_k.id
            ].to(torch.float32)
            __td_o_mm_acc = (
                __td_o_mm_acc
                + __td_o_mm_partial
                * __td_o_mm_a_scale[:, None]
                * __td_o_mm_w_scale[None, :]
            )
        attention_out[__td_o_mm_tile_m, __td_o_mm_tile_n] = __td_o_mm_acc.to(
            attention_out.dtype
        )
    for __td_post_partial_m, __td_post_partial_n in hl.tile(
        [__td_post_num_tokens, __td_post_hidden_size],
        block_size=[1, __td_post_group_size],
    ):
        __td_post_partial_values = __td_post_input[
            __td_post_partial_m, __td_post_partial_n
        ].to(torch.float32)
        if __td_post_residual is not None:
            __td_post_partial_values = (
                __td_post_partial_values
                + __td_post_residual[__td_post_partial_m, __td_post_partial_n]
            )
            __td_post_residual[__td_post_partial_m, __td_post_partial_n] = (
                __td_post_partial_values.to(__td_post_residual.dtype)
            )
        __td_post_unrounded_values[__td_post_partial_m, __td_post_partial_n] = (
            __td_post_partial_values
        )
        __td_post_rms_partials[__td_post_partial_m, __td_post_partial_n.id] = torch.sum(
            __td_post_partial_values * __td_post_partial_values, dim=-1
        )
    for __td_post_quant_m, __td_post_quant_g, __td_post_quant_n in hl.tile(
        [__td_post_num_tokens, __td_post_groups_per_row, __td_post_group_size],
        block_size=[1, 1, __td_post_group_size],
    ):
        __td_post_quant_m_idx = __td_post_quant_m.begin + hl.arange(
            __td_post_quant_m.block_size
        )
        __td_post_quant_group_idx = __td_post_quant_g.index
        __td_post_quant_n_idx = (
            __td_post_quant_group_idx[:, None] * __td_post_group_size
            + __td_post_quant_n.index[None, :]
        )
        __td_post_quant_m_blk = __td_post_quant_m_idx[:, None, None]
        __td_post_quant_n_blk = __td_post_quant_n_idx[None, :, :]
        __td_post_square_sum = hl.zeros([__td_post_quant_m], dtype=torch.float32)
        for __td_post_reduce_g in hl.tile(__td_post_groups_per_row, block_size=1):
            __td_post_square_sum = __td_post_square_sum + torch.sum(
                __td_post_rms_partials[__td_post_quant_m, __td_post_reduce_g], dim=-1
            )
        __td_post_inv_rms = torch.rsqrt(
            __td_post_square_sum * (1.0 / __td_post_hidden_size) + __td_post_epsilon
        )
        __td_post_quant_values = __td_post_unrounded_values[
            __td_post_quant_m_blk, __td_post_quant_n_blk
        ]
        __td_post_normalized = (
            __td_post_quant_values * __td_post_inv_rms[:, None, None]
        ).to(torch.bfloat16) * __td_post_weight[__td_post_quant_n_blk]
        __td_post_quant_scale = torch.amax(torch.abs(__td_post_normalized), dim=-1).to(
            torch.float32
        )
        if __td_post_scale_ub is not None:
            __td_post_quant_scale = __td_post_quant_scale.clamp(
                max=hl.load(__td_post_scale_ub, [])
            )
        __td_post_quant_scale = (__td_post_quant_scale / FP8_MAX).clamp(
            min=FP8_MIN_SCALE
        )
        __td_post_scale[__td_post_quant_m, __td_post_quant_g] = __td_post_quant_scale
        __td_post_result[__td_post_quant_m_blk, __td_post_quant_n_blk] = (
            (__td_post_normalized / __td_post_quant_scale[:, :, None])
            .clamp(FP8_MIN, FP8_MAX)
            .to(__td_post_result.dtype)
        )
    for __td_w13_tile_m, __td_w13_tile_n in hl.tile(
        [__td_w13_m, __td_w13_n], block_size=[1, None]
    ):
        __td_w13_acc = hl.zeros([__td_w13_tile_m, __td_w13_tile_n], dtype=torch.float32)
        for __td_w13_tile_k in hl.tile(__td_w13_k, block_size=__td_w13_group_size):
            __td_w13_partial = hl.dot(
                __td_w13_activation_q[__td_w13_tile_m, __td_w13_tile_k],
                __td_w13_weight_q[__td_w13_tile_n, __td_w13_tile_k].T,
            ).to(torch.float32)
            __td_w13_a_scale = __td_w13_activation_scale[
                __td_w13_tile_m, __td_w13_tile_k.id
            ].to(torch.float32)
            __td_w13_w_scale = __td_w13_weight_scale[
                __td_w13_tile_n.index // __td_w13_group_size, __td_w13_tile_k.id
            ].to(torch.float32)
            __td_w13_acc = (
                __td_w13_acc
                + __td_w13_partial
                * __td_w13_a_scale[:, None]
                * __td_w13_w_scale[None, :]
            )
        gate_up[__td_w13_tile_m, __td_w13_tile_n] = __td_w13_acc.to(gate_up.dtype)
    for __td_activation_tile_m, __td_activation_tile_i in hl.tile(
        [__td_activation_m, __td_activation_intermediate],
        block_size=[1, __td_activation_group_size],
    ):
        __td_activation_gate = __td_activation_gate_up[
            __td_activation_tile_m, __td_activation_tile_i
        ].to(torch.float32)
        __td_activation_up = __td_activation_gate_up[
            __td_activation_tile_m,
            __td_activation_tile_i + __td_activation_intermediate,
        ].to(torch.float32)
        __td_activation_activated = (
            __td_activation_gate
            * torch.sigmoid(__td_activation_gate)
            * __td_activation_up
        )
        __td_activation_scale = (
            torch.amax(torch.abs(__td_activation_activated), dim=-1) / FP8_MAX
        ).clamp(min=FP8_MIN_SCALE)
        activation_scale[__td_activation_tile_m, __td_activation_tile_i.id] = (
            __td_activation_scale
        )
        activation_q[__td_activation_tile_m, __td_activation_tile_i] = (
            (__td_activation_activated / __td_activation_scale[:, None])
            .clamp(FP8_MIN, FP8_MAX)
            .to(activation_q.dtype)
        )
    for __td_w2_tile_m, __td_w2_tile_n in hl.tile(
        [__td_w2_m, __td_w2_n], block_size=[1, None]
    ):
        __td_w2_acc = hl.zeros([__td_w2_tile_m, __td_w2_tile_n], dtype=torch.float32)
        for __td_w2_tile_k in hl.tile(__td_w2_k, block_size=__td_w2_group_size):
            __td_w2_partial = hl.dot(
                __td_w2_activation_q[__td_w2_tile_m, __td_w2_tile_k],
                __td_w2_weight_q[__td_w2_tile_n, __td_w2_tile_k].T,
            ).to(torch.float32)
            __td_w2_a_scale = __td_w2_activation_scale[
                __td_w2_tile_m, __td_w2_tile_k.id
            ].to(torch.float32)
            __td_w2_w_scale = __td_w2_weight_scale[
                __td_w2_tile_n.index // __td_w2_group_size, __td_w2_tile_k.id
            ].to(torch.float32)
            __td_w2_acc = (
                __td_w2_acc
                + __td_w2_partial * __td_w2_a_scale[:, None] * __td_w2_w_scale[None, :]
            )
        output[__td_w2_tile_m, __td_w2_tile_n] = __td_w2_acc.to(output.dtype)
    return (
        output,
        pre_q,
        pre_scale,
        qkv,
        partial_out,
        partial_lse,
        attention,
        attention_q,
        attention_scale,
        attention_out,
        ffn_q,
        ffn_scale,
        gate_up,
        activation_q,
        activation_scale,
        residual,
    )


def use_cudagraph() -> bool:
    """The timed closures replay pre-captured CUDA graphs."""
    return True


def has_vllm() -> bool:
    """Whether the optional production vLLM layer is importable."""
    try:
        from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer  # noqa: F401
    except ImportError:
        return False
    return True


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("qwen3_decode_layer is pretuned only for NVIDIA SM100")


def _make_fp8_random(shape: tuple[int, ...], scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * scale).to(
        torch.float8_e4m3fn
    )


def _make_inputs(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    hidden_groups = HIDDEN // GROUP
    intermediate_groups = INTERMEDIATE // GROUP
    qkv_width = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
    logical_blocks = math.ceil(CONTEXT / CACHE_BLOCK)
    physical_blocks = math.ceil(logical_blocks * 1.25)
    block_table = torch.randperm(physical_blocks, device="cuda", dtype=torch.int64)[
        :logical_blocks
    ].to(torch.int32)[None, :]
    final_logical_block = (CONTEXT - 1) // CACHE_BLOCK
    final_block_offset = (CONTEXT - 1) % CACHE_BLOCK
    final_physical_block = block_table[:, final_logical_block].to(torch.int64)
    return {
        "hidden_states": torch.randn(
            (BATCH, HIDDEN), device="cuda", dtype=torch.bfloat16
        ),
        "residual": torch.randn((BATCH, HIDDEN), device="cuda", dtype=torch.bfloat16),
        "pre_weight": (
            torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
        ),
        "pre_q": torch.empty((BATCH, HIDDEN), device="cuda", dtype=torch.float8_e4m3fn),
        "pre_scale": torch.empty(
            (BATCH, hidden_groups), device="cuda", dtype=torch.float32
        ),
        "qkv_weight_q": _make_fp8_random((qkv_width, HIDDEN)),
        "qkv_weight_scale": (
            torch.rand(
                (qkv_width // GROUP, hidden_groups),
                device="cuda",
                dtype=torch.float32,
            )
            * 0.01
            + 0.01
        ),
        "q_weight": (
            torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
        ),
        "k_weight": (
            torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
        ),
        # Filled from the production vLLM layer's rotary cache before either
        # implementation runs.
        "cos_sin": torch.empty(
            (max(CONTEXT, 4096), HEAD_DIM),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        "position": torch.full((BATCH,), CONTEXT - 1, device="cuda", dtype=torch.int64),
        "kv_cache": torch.randn(
            (
                physical_blocks,
                CACHE_BLOCK,
                KV_HEADS,
                2 * HEAD_DIM,
            ),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        "block_table": block_table,
        "slot_mapping": final_physical_block * CACHE_BLOCK + final_block_offset,
        "o_weight_q": _make_fp8_random((HIDDEN, HIDDEN)),
        "o_weight_scale": (
            torch.rand(
                (hidden_groups, hidden_groups),
                device="cuda",
                dtype=torch.float32,
            )
            * 0.01
            + 0.01
        ),
        "attention_q": torch.empty(
            (BATCH, HIDDEN), device="cuda", dtype=torch.float8_e4m3fn
        ),
        "attention_scale": torch.empty(
            (BATCH, hidden_groups), device="cuda", dtype=torch.float32
        ),
        "post_weight": (
            torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
        ),
        "ffn_q": torch.empty((BATCH, HIDDEN), device="cuda", dtype=torch.float8_e4m3fn),
        "ffn_scale": torch.empty(
            (BATCH, hidden_groups), device="cuda", dtype=torch.float32
        ),
        "w13_q": _make_fp8_random((2 * INTERMEDIATE, HIDDEN)),
        "w13_scale": (
            torch.rand(
                (2 * intermediate_groups, hidden_groups),
                device="cuda",
                dtype=torch.float32,
            )
            * (0.5 / math.sqrt(HIDDEN))
            + (0.75 / math.sqrt(HIDDEN))
        ),
        "w2_q": _make_fp8_random((HIDDEN, INTERMEDIATE)),
        "w2_scale": (
            torch.rand(
                (hidden_groups, intermediate_groups),
                device="cuda",
                dtype=torch.float32,
            )
            * (0.5 / math.sqrt(INTERMEDIATE))
            + (0.75 / math.sqrt(INTERMEDIATE))
        ),
    }


_LINEAR_WEIGHT_NAMES = (
    ("qkv_weight_q", "qkv_weight_scale"),
    ("o_weight_q", "o_weight_scale"),
    ("w13_q", "w13_scale"),
    ("w2_q", "w2_scale"),
)


def _make_helion_inputs(
    tensors: dict[str, torch.Tensor], use_ue8m0: bool
) -> dict[str, torch.Tensor]:
    cloned = dict(tensors)
    cloned["residual"] = tensors["residual"].clone()
    cloned["kv_cache"] = tensors["kv_cache"].clone()
    if use_ue8m0:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            requant_weight_ue8m0_inplace,
        )

        for weight_name, scale_name in _LINEAR_WEIGHT_NAMES:
            cloned[weight_name] = tensors[weight_name].clone()
            cloned[scale_name] = tensors[scale_name].clone()
            requant_weight_ue8m0_inplace(
                cloned[weight_name], cloned[scale_name], block_size=(GROUP, GROUP)
            )
    return cloned


def _kernel_args(tensors: dict[str, torch.Tensor]) -> tuple[object, ...]:
    return (
        tensors["hidden_states"],
        tensors["residual"],
        tensors["pre_weight"],
        tensors["pre_q"],
        tensors["pre_scale"],
        tensors["qkv_weight_q"],
        tensors["qkv_weight_scale"],
        tensors["q_weight"],
        tensors["k_weight"],
        tensors["cos_sin"],
        tensors["position"],
        tensors["kv_cache"],
        tensors["block_table"],
        tensors["slot_mapping"],
        tensors["o_weight_q"],
        tensors["o_weight_scale"],
        tensors["attention_q"],
        tensors["attention_scale"],
        tensors["post_weight"],
        tensors["ffn_q"],
        tensors["ffn_scale"],
        tensors["w13_q"],
        tensors["w13_scale"],
        tensors["w2_q"],
        tensors["w2_scale"],
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


def _make_compiled_layer_class() -> type[torch.nn.Module]:
    from torch import nn
    from vllm.compilation.decorators import support_torch_compile
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer

    @support_torch_compile(
        dynamic_arg_dims={"positions": 0, "hidden_states": 0, "residual": 0}
    )
    class CompiledQwen3Layer(nn.Module):
        def __init__(self, *, config, cache_config, quant_config, prefix) -> None:
            super().__init__()
            self.layer = Qwen3DecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        def forward(
            self, positions, hidden_states, residual
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self.layer(positions, hidden_states, residual)

    return CompiledQwen3Layer


def _linear_modules(layer) -> dict[str, tuple[object, str, str]]:
    return {
        "qkv": (layer.self_attn.qkv_proj, "qkv_weight_q", "qkv_weight_scale"),
        "o": (layer.self_attn.o_proj, "o_weight_q", "o_weight_scale"),
        "w13": (layer.mlp.gate_up_proj, "w13_q", "w13_scale"),
        "w2": (layer.mlp.down_proj, "w2_q", "w2_scale"),
    }


def _destroy_vllm() -> None:
    from vllm.distributed import destroy_distributed_environment
    from vllm.distributed import destroy_model_parallel

    destroy_model_parallel()
    destroy_distributed_environment()


def _prepare_vllm_attention_state() -> str | None:
    """Start backend selection from a clean cache and return its old override."""
    from vllm.v1.attention import selector
    from vllm.v1.attention.backends import utils

    previous_layout = utils._KV_CACHE_LAYOUT_OVERRIDE
    selector._cached_get_attn_backend.cache_clear()
    return previous_layout


def _restore_vllm_attention_state(previous_layout: str | None) -> None:
    """Undo the process-global state changed by vLLM backend selection."""
    from vllm.v1.attention import selector
    from vllm.v1.attention.backends import utils

    selector._cached_get_attn_backend.cache_clear()
    utils.set_kv_cache_layout(previous_layout)


def _initialize_vllm(model_path: Path) -> tuple[object, object, object, object]:
    from vllm.config import CacheConfig
    from vllm.config import ModelConfig
    from vllm.config import VllmConfig
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment
    from vllm.distributed import initialize_model_parallel
    from vllm.distributed import model_parallel_is_initialized
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.utils.network_utils import get_open_port

    if torch.distributed.is_initialized() or model_parallel_is_initialized():
        raise RuntimeError(
            "qwen3_decode_layer owns its temporary vLLM distributed state; "
            "run it outside an initialized distributed context"
        )
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
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[GROUP, GROUP],
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        device_index = torch.cuda.current_device()
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=device_index,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(1, 1)
    except Exception:
        _destroy_vllm()
        raise
    return vllm_config, model_config, cache_config, quant_config


def _copy_and_process_vllm_weights(
    wrapper,
    tensors: dict[str, torch.Tensor],
    model_config,
    vllm_config,
) -> None:
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    layer = wrapper.layer
    with torch.no_grad():
        for module, weight_name, scale_name in _linear_modules(layer).values():
            module.weight.copy_(tensors[weight_name])
            module.weight_scale_inv.copy_(tensors[scale_name])
        layer.self_attn.q_norm.weight.copy_(tensors["q_weight"])
        layer.self_attn.k_norm.weight.copy_(tensors["k_weight"])
        layer.input_layernorm.weight.copy_(tensors["pre_weight"])
        layer.post_attention_layernorm.weight.copy_(tensors["post_weight"])
    with set_current_vllm_config(vllm_config):
        process_weights_after_loading(wrapper, model_config, torch.device("cuda"))


def _make_vllm_cache(canonical_cache: torch.Tensor, cache_layout: str) -> torch.Tensor:
    logical_cache = canonical_cache.permute(0, 2, 1, 3)
    if cache_layout == "NHD":
        return logical_cache
    if cache_layout == "HND":
        return logical_cache.contiguous()
    raise ValueError(f"unsupported vLLM KV-cache layout: {cache_layout}")


def _make_attention_metadata(
    vllm_config, attention, tensors, layer_name: str
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backend import CommonAttentionMetadata

    query_start_loc = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([CONTEXT], device="cuda", dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=torch.tensor([CONTEXT], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([CONTEXT], dtype=torch.int32),
        _num_computed_tokens_cpu=torch.tensor([CONTEXT - 1], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=CONTEXT,
        block_table_tensor=tensors["block_table"],
        slot_mapping=tensors["slot_mapping"],
        causal=True,
        positions=tensors["position"],
    )
    spec = attention.get_kv_cache_spec(vllm_config)
    with set_current_vllm_config(vllm_config):
        try:
            metadata = (
                attention.get_attn_backend()
                .get_builder_cls()(
                    spec, [layer_name], vllm_config, torch.device("cuda")
                )
                .build(0, common)
            )
        except FileNotFoundError as error:
            if attention.get_attn_backend().get_name() == "FLASHINFER":
                raise RuntimeError(
                    "vLLM selected FlashInfer, but its kernel artifact is "
                    "unavailable. Install its build-time dependencies or "
                    "prebuild the production FlashInfer kernels."
                ) from error
            raise
    return {layer_name: metadata}, {layer_name: tensors["slot_mapping"]}


def _make_vllm_call(
    tensors: dict[str, torch.Tensor],
) -> tuple[
    Callable[[], tuple[torch.Tensor, torch.Tensor]],
    dict[str, torch.Tensor],
    bool,
    str,
    Callable[[], None],
]:
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backends.utils import get_kv_cache_layout

    previous_layout = _prepare_vllm_attention_state()
    model_directory = tempfile.TemporaryDirectory(prefix="qwen3-8b-fp8-")
    model_path = Path(model_directory.name)
    (model_path / "config.json").write_text(json.dumps(QWEN3_8B_FP8_CONFIG))
    initialized = False
    try:
        vllm_config, model_config, cache_config, quant_config = _initialize_vllm(
            model_path
        )
        initialized = True
        compiled_layer_class = _make_compiled_layer_class()
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with set_current_vllm_config(vllm_config):
                wrapper = (
                    compiled_layer_class(
                        config=model_config.hf_text_config,
                        cache_config=cache_config,
                        quant_config=quant_config,
                        prefix="model.layers.0",
                    )
                    .eval()
                    .cuda()
                )
        finally:
            torch.set_default_dtype(old_dtype)

        linear_kernels = {
            type(module.quant_method.fp8_linear).__name__
            for module, _weight, _scale in _linear_modules(wrapper.layer).values()
        }
        if len(linear_kernels) != 1:
            raise RuntimeError(f"vLLM selected mixed linear kernels: {linear_kernels}")
        linear_kernel = linear_kernels.pop()
        qkv_kernel = wrapper.layer.self_attn.qkv_proj.quant_method.fp8_linear
        quantizer = getattr(qkv_kernel, "quant_fp8", None)
        use_ue8m0 = bool(getattr(quantizer, "use_ue8m0", False))
        tensors["cos_sin"].copy_(
            wrapper.layer.self_attn.rotary_emb.cos_sin_cache[:CONTEXT]
        )
        _copy_and_process_vllm_weights(wrapper, tensors, model_config, vllm_config)

        attention = wrapper.layer.self_attn.attn
        layer_name = "model.layers.0.self_attn.attn"
        with set_current_vllm_config(vllm_config):
            cache_layout = get_kv_cache_layout()
        vllm_tensors = {
            "hidden_states": tensors["hidden_states"],
            "residual": tensors["residual"].clone(),
            "position": tensors["position"],
            "kv_cache": _make_vllm_cache(tensors["kv_cache"].clone(), cache_layout),
            "block_table": tensors["block_table"],
            "slot_mapping": tensors["slot_mapping"],
        }
        attention.kv_cache = vllm_tensors["kv_cache"]
        attention_metadata, slot_mapping = _make_attention_metadata(
            vllm_config, attention, vllm_tensors, layer_name
        )
        attention_backend = attention.get_attn_backend().get_name()
    except Exception:
        try:
            if initialized:
                _destroy_vllm()
        finally:
            try:
                _restore_vllm_attention_state(previous_layout)
            finally:
                model_directory.cleanup()
        raise

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        with set_forward_context(
            attention_metadata,
            vllm_config=vllm_config,
            num_tokens=BATCH,
            slot_mapping=slot_mapping,
        ):
            return wrapper(
                vllm_tensors["position"],
                vllm_tensors["hidden_states"],
                vllm_tensors["residual"],
            )

    def close() -> None:
        try:
            _destroy_vllm()
        finally:
            try:
                _restore_vllm_attention_state(previous_layout)
            finally:
                model_directory.cleanup()

    backend = f"{attention_backend.lower()} + {linear_kernel}"
    return launch, vllm_tensors, use_ue8m0, backend, close


def _cache_slot(
    tensors: dict[str, torch.Tensor], *, vllm_layout: bool = False
) -> torch.Tensor:
    slot = int(tensors["slot_mapping"][0].item())
    block = slot // CACHE_BLOCK
    offset = slot % CACHE_BLOCK
    if vllm_layout:
        return tensors["kv_cache"][block, :, offset]
    return tensors["kv_cache"][block, offset]


def _make_reset(
    tensors: dict[str, torch.Tensor], *, vllm_layout: bool = False
) -> Callable[[], None]:
    initial_residual = tensors["residual"].clone()
    cache_slot = _cache_slot(tensors, vllm_layout=vllm_layout)
    initial_cache_slot = cache_slot.clone()

    def reset() -> None:
        tensors["residual"].copy_(initial_residual)
        cache_slot.copy_(initial_cache_slot)

    return reset


def _assert_vllm_close(
    helion_outputs: tuple[torch.Tensor, ...],
    vllm_outputs: tuple[torch.Tensor, torch.Tensor],
    helion_tensors: dict[str, torch.Tensor],
    vllm_tensors: dict[str, torch.Tensor],
) -> None:
    helion_output = helion_outputs[0]
    helion_residual = helion_outputs[-1]
    vllm_output, vllm_residual = vllm_outputs
    torch.testing.assert_close(
        helion_output.float(), vllm_output.float(), atol=0.25, rtol=0.05
    )
    torch.testing.assert_close(
        helion_residual.float(), vllm_residual.float(), atol=0.125, rtol=0.03
    )
    torch.testing.assert_close(
        _cache_slot(helion_tensors).float(),
        _cache_slot(vllm_tensors, vllm_layout=True).float(),
        atol=0.125,
        rtol=0.03,
    )


@torch.inference_mode()
def correctness_check() -> None:
    """Check the one pretuned shape against vLLM's production decoder layer."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError("vLLM is required for the Qwen3 comparison")
    base = _make_inputs()
    vllm_call, vllm_tensors, use_ue8m0, _backend, close_vllm = _make_vllm_call(base)
    try:
        helion_tensors = _make_helion_inputs(base, use_ue8m0)
        helion_outputs = qwen3_decode_layer(*_kernel_args(helion_tensors))
        vllm_outputs = vllm_call()
        torch.cuda.synchronize()
        _assert_vllm_close(
            helion_outputs,
            vllm_outputs,
            helion_tensors,
            vllm_tensors,
        )
    finally:
        close_vllm()


@torch.inference_mode()
def main(verbose: bool = True) -> dict:
    """Benchmark one production decode shape against vLLM with cold L2."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError("vLLM is required for the Qwen3 comparison")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _bench import capture_cuda_graph
    from _bench import run_sweep

    base = _make_inputs()
    vllm_call, vllm_tensors, use_ue8m0, backend, close_vllm = _make_vllm_call(base)
    try:
        helion_tensors = _make_helion_inputs(base, use_ue8m0)
        helion_reset = _make_reset(helion_tensors)
        vllm_reset = _make_reset(vllm_tensors, vllm_layout=True)

        helion_outputs = qwen3_decode_layer(*_kernel_args(helion_tensors))
        vllm_outputs = vllm_call()
        torch.cuda.synchronize()
        _assert_vllm_close(
            helion_outputs,
            vllm_outputs,
            helion_tensors,
            vllm_tensors,
        )

        helion_graph, _ = capture_cuda_graph(
            lambda: qwen3_decode_layer(*_kernel_args(helion_tensors)),
            helion_reset,
        )
        vllm_graph, _ = capture_cuda_graph(vllm_call, vllm_reset)

        def make_calls(_shape: None) -> tuple:
            return (
                helion_graph.replay,
                [(f"vllm_auto ({backend})", vllm_graph.replay)],
                (f"{BATCH:>5d}  {HIDDEN:>6d}  {CONTEXT:>7d}  {ATTENTION_SPLITS:>6d}"),
            )

        return run_sweep(
            [None],
            make_calls,
            use_cudagraph=False,
            pre_captured_cudagraph=True,
            interleave_pre_captured=False,
            make_resets=lambda _shape: (helion_reset, vllm_reset),
            thermal_warmup_ms=10_000,
            verbose=verbose,
            shape_header=(
                f"{'batch':>5s}  {'hidden':>6s}  {'context':>7s}  {'splits':>6s}"
            ),
        )
    finally:
        close_vllm()


if __name__ == "__main__":
    main()
