# ruff: noqa: ANN001, ANN003, ANN202, A002
"""Qwen3 TileDependency probe with source-visible prefix and merge tiles.

This file deliberately does not rewrite a compiled tile body.  The same
Helion functions below are used as the source of the separate kernels and are
inlined, statement-for-statement, into the whole-layer graph.  Their finer
top-level loops expose the task decomposition used by the successful Triton
SM-overlap probe without teaching the compiler about RMSNorm or attention.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import math
from pathlib import Path
import sys
import types

import torch

import helion
from helion._compiler.cross_loop_scheduler import CROSS_LOOP_NUM_WORKERS_CONFIG
from helion._compiler.program_id import ForEachProgramID
import helion.language as hl

FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)
_USE_CANONICAL_ATTENTION_VIEWS = False
_USE_TASK_ALIGNED_ATTENTION = False


@helion.kernel(static_shapes=True, autotune_effort="none")
def tiled_rms_norm_per_block_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    """RMS/quant expressed as source-visible producer, finalize, consumer tiles."""
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)
    groups_per_row = scale.shape[1]
    hl.specialize(groups_per_row)
    assert group_size == 128
    assert result.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32

    rms_partials = torch.empty(
        (num_tokens, groups_per_row), dtype=torch.float32, device=input.device
    )
    unrounded_values = torch.empty_like(input, dtype=torch.float32)

    for partial_m, partial_n in hl.tile(
        [num_tokens, hidden_size], block_size=[1, group_size]
    ):
        partial_values = input[partial_m, partial_n].to(torch.float32)
        if residual is not None:
            partial_values = partial_values + residual[partial_m, partial_n]
            residual[partial_m, partial_n] = partial_values.to(residual.dtype)
        unrounded_values[partial_m, partial_n] = partial_values
        rms_partials[partial_m, partial_n.id] = torch.sum(
            partial_values * partial_values, dim=-1
        )

    for quant_m, quant_g, quant_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, 1, group_size]
    ):
        quant_m_idx = quant_m.begin + hl.arange(quant_m.block_size)
        quant_group_idx = quant_g.index
        quant_n_idx = quant_group_idx[:, None] * group_size + quant_n.index[None, :]
        quant_m_blk = quant_m_idx[:, None, None]
        quant_n_blk = quant_n_idx[None, :, :]
        square_sum = hl.zeros([quant_m], dtype=torch.float32)
        for reduce_g in hl.tile(groups_per_row, block_size=1):
            square_sum = square_sum + torch.sum(rms_partials[quant_m, reduce_g], dim=-1)
        inv_rms = torch.rsqrt(square_sum * (1.0 / hidden_size) + epsilon)
        quant_values = unrounded_values[quant_m_blk, quant_n_blk]
        normalized = (quant_values * inv_rms[:, None, None]).to(
            torch.bfloat16
        ) * weight[quant_n_blk]
        quant_scale = torch.amax(torch.abs(normalized), dim=-1).to(torch.float32)
        if scale_ub is not None:
            quant_scale = quant_scale.clamp(max=hl.load(scale_ub, []))
        quant_scale = (quant_scale / FP8_MAX).clamp(min=FP8_MIN_SCALE)
        scale[quant_m, quant_g] = quant_scale
        result[quant_m_blk, quant_n_blk] = (
            (normalized / quant_scale[:, :, None])
            .clamp(FP8_MIN, FP8_MAX)
            .to(result.dtype)
        )


@helion.kernel(static_shapes=True, autotune_effort="none")
def tiled_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Store one complete K/V head from each logical cache task."""
    num_tokens, num_kv_heads, head_dim = key.shape
    hl.specialize(num_kv_heads)
    hl.specialize(head_dim)
    hl.specialize(block_size)
    for tile_t, tile_h, tile_d in hl.tile(
        [num_tokens, num_kv_heads, head_dim],
        block_size=[1, 1, head_dim],
    ):
        token = tile_t.index
        cache_head = tile_h.index
        dimension = tile_d.index
        key_value = key[tile_t, tile_h, tile_d]
        value_value = value[tile_t, tile_h, tile_d]
        slot = slot_mapping[token]
        block = (slot // block_size)[:, None, None]
        offset = (slot % block_size)[:, None, None]
        hl.store(
            kv_cache,
            [
                block,
                offset,
                cache_head[None, :, None],
                dimension[None, None, :],
            ],
            key_value,
        )
        hl.store(
            kv_cache,
            [
                block,
                offset,
                cache_head[None, :, None],
                (dimension + head_dim)[None, None, :],
            ],
            value_value,
        )


@helion.kernel(static_shapes=True, autotune_effort="none")
def flat_fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int = -1,
) -> None:
    """Apply Q/K normalization and RoPE over contiguous flat head tiles."""
    num_tokens, qkv_width = qkv.shape
    total_heads = num_heads_q + num_heads_k + num_heads_v
    assert qkv_width == total_heads * head_dim
    _, rotary_dim = cos_sin_cache.shape
    hl.specialize(qkv_width)
    hl.specialize(rotary_dim)
    embed_dim = rotary_dim // 2
    hl.specialize(num_heads_q)
    hl.specialize(num_heads_k)
    hl.specialize(num_heads_v)
    hl.specialize(head_dim)
    assert is_neox
    assert rotary_dim == head_dim
    qk_width = (num_heads_q + num_heads_k) * head_dim

    for tile_m, tile_n in hl.tile([num_tokens, qk_width], block_size=[1, head_dim]):
        x = qkv[tile_m, tile_n].to(torch.float32)
        rms = torch.rsqrt(x.pow(2).sum(-1) * (1.0 / head_dim) + eps)
        dimension = tile_n.index - tile_n.begin
        use_q = tile_n.index < num_heads_q * head_dim
        weight = torch.where(use_q, q_weight[dimension], k_weight[dimension])
        x = (x * rms[:, None]).to(qkv.dtype) * weight[None, :]
        position = position_ids[tile_m]
        first_half = dimension < embed_dim
        partner_dimension = torch.where(
            first_half, dimension + embed_dim, dimension - embed_dim
        )
        partner = torch.gather(x, 1, partner_dimension[None, :])
        cos = cos_sin_cache[position, dimension % embed_dim]
        sin = cos_sin_cache[position, dimension % embed_dim + embed_dim]
        qkv[tile_m, tile_n] = x * cos[:, :] + torch.where(
            first_half[None, :], -partner * sin[:, :], partner * sin[:, :]
        )


@helion.kernel(static_shapes=True, autotune_effort="none")
def canonical_paged_gqa_decode_attention_split(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    block_size: int,
    q_per_kv: int,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split attention with partials stored in canonical query-head coordinates."""
    _, num_q_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[2]
    assert num_q_heads == num_kv_heads * q_per_kv
    assert context % splits == 0
    hl.specialize(head_dim)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(context)
    hl.specialize(block_size)
    hl.specialize(splits)
    split_context = context // splits
    q = query.view(num_kv_heads, q_per_kv, head_dim)
    partial_out = torch.empty(
        (splits, num_q_heads, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (splits, num_q_heads),
        device=query.device,
        dtype=torch.float32,
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_split, tile_g, tile_q in hl.tile(
        [splits, num_kv_heads, q_per_kv], block_size=[1, 1, None]
    ):
        m_i = hl.full([tile_g, tile_q], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_g, tile_q], 1.0, dtype=torch.float32)
        acc = hl.zeros([tile_g, tile_q, head_dim], dtype=torch.float32)
        split_idx = tile_split.begin
        q_blk = (q[tile_g, tile_q, :] * qk_scale).to(q.dtype)
        for tile_local_n in hl.tile(split_context):
            n = split_idx * split_context + tile_local_n.index
            physical_block = block_table[0, n // block_size]
            block_offset = n % block_size
            d = hl.arange(head_dim)
            k = hl.load(
                kv_cache,
                [
                    physical_block[None, :, None],
                    block_offset[None, :, None],
                    tile_g.index[:, None, None],
                    d[None, None, :],
                ],
            )
            scores = torch.bmm(q_blk, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(scores, -1))
            p = torch.exp2(scores - m_ij[:, :, None])
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            acc = acc * alpha[:, :, None]
            v = hl.load(
                kv_cache,
                [
                    physical_block[None, :, None],
                    block_offset[None, :, None],
                    tile_g.index[:, None, None],
                    (d + head_dim)[None, None, :],
                ],
            )
            acc = torch.baddbmm(acc, p.to(v.dtype), v)
            m_i = m_ij
        query_head = tile_g.index[:, None] * q_per_kv + tile_q.index[None, :]
        partial_out[tile_split, query_head, :] = (acc / l_i[:, :, None])[None, :, :, :]
        partial_lse[tile_split, query_head] = (m_i + torch.log2(l_i))[None, :, :]
    return partial_out, partial_lse


@helion.kernel(static_shapes=True, autotune_effort="none")
def task_aligned_paged_gqa_decode_attention_split(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    block_size: int,
    q_per_kv: int,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain the baseline layout while exposing the KV-head task coordinate."""
    _, num_q_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[2]
    assert num_q_heads == num_kv_heads * q_per_kv
    assert context % splits == 0
    hl.specialize(head_dim)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(context)
    hl.specialize(block_size)
    hl.specialize(splits)
    split_context = context // splits
    q = query.view(num_kv_heads, q_per_kv, head_dim)
    partial_out = torch.empty(
        (splits, num_kv_heads, q_per_kv, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (splits, num_kv_heads, q_per_kv),
        device=query.device,
        dtype=torch.float32,
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_split, tile_g, tile_q in hl.tile(
        [splits, num_kv_heads, q_per_kv], block_size=[1, 1, None]
    ):
        m_i = hl.full([tile_g, tile_q], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_g, tile_q], 1.0, dtype=torch.float32)
        acc = hl.zeros([tile_g, tile_q, head_dim], dtype=torch.float32)
        split_idx = tile_split.begin
        q_blk = (q[tile_g, tile_q, :] * qk_scale).to(q.dtype)
        for tile_local_n in hl.tile(split_context):
            n = split_idx * split_context + tile_local_n.index
            physical_block = block_table[0, n // block_size]
            block_offset = n % block_size
            d = hl.arange(head_dim)
            k = hl.load(
                kv_cache,
                [
                    physical_block[None, :, None],
                    block_offset[None, :, None],
                    tile_g.index[:, None, None],
                    d[None, None, :],
                ],
            )
            scores = torch.bmm(q_blk, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(scores, -1))
            p = torch.exp2(scores - m_ij[:, :, None])
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            acc = acc * alpha[:, :, None]
            v = hl.load(
                kv_cache,
                [
                    physical_block[None, :, None],
                    block_offset[None, :, None],
                    tile_g.index[:, None, None],
                    (d + head_dim)[None, None, :],
                ],
            )
            acc = torch.baddbmm(acc, p.to(v.dtype), v)
            m_i = m_ij
        partial_out[tile_split, tile_g, tile_q, :] = (acc / l_i[:, :, None])[
            None, :, :, :
        ]
        partial_lse[tile_split, tile_g, tile_q] = (m_i + torch.log2(l_i))[None, :, :]
    return partial_out, partial_lse


@helion.kernel(static_shapes=True, autotune_effort="none")
def tiled_merge_attention_splits(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    """Two source-visible merge levels matching the Triton overlap probe."""
    splits, num_kv_heads, q_per_kv, head_dim = partial_out.shape
    hl.specialize(splits)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(head_dim)
    query_heads = num_kv_heads * q_per_kv
    hl.specialize(query_heads)
    merge_chunks = 16
    hl.specialize(merge_chunks)
    assert splits % merge_chunks == 0
    splits_per_chunk = splits // merge_chunks
    hl.specialize(splits_per_chunk)

    partial_out_flat = partial_out.view(splits, query_heads, head_dim)
    partial_lse_flat = partial_lse.view(splits, query_heads)
    partial_out_storage = partial_out_flat.view(-1)
    partial_lse_storage = partial_lse_flat.view(-1)
    chunk_out = torch.empty(
        (merge_chunks, query_heads, head_dim),
        dtype=torch.float32,
        device=partial_out.device,
    )
    chunk_lse = torch.empty(
        (merge_chunks, query_heads),
        dtype=torch.float32,
        device=partial_out.device,
    )
    chunk_out_storage = chunk_out.view(-1)
    chunk_lse_storage = chunk_lse.view(-1)
    output = torch.empty(
        (query_heads, head_dim),
        dtype=torch.bfloat16,
        device=partial_out.device,
    )

    for tile_chunk, chunk_head in hl.tile(
        [merge_chunks, query_heads], block_size=[1, 1]
    ):
        chunk_split_idx = (
            tile_chunk.index[:, None] * splits_per_chunk
            + hl.arange(splits_per_chunk)[None, :]
        )
        chunk_lse_offsets = (
            chunk_split_idx[:, :, None] * query_heads + chunk_head.index[None, None, :]
        )
        chunk_lse_values = partial_lse_storage[chunk_lse_offsets]
        chunk_max_lse = torch.amax(chunk_lse_values, dim=1)
        chunk_weights = torch.exp2(chunk_lse_values - chunk_max_lse[:, None, :])
        chunk_value_offsets = (
            chunk_lse_offsets[:, :, :, None] * head_dim
            + hl.arange(head_dim)[None, None, None, :]
        )
        chunk_values = partial_out_storage[chunk_value_offsets]
        chunk_denominator = torch.sum(chunk_weights, dim=1)
        chunk_merged = torch.sum(chunk_values * chunk_weights[:, :, :, None], dim=1)
        chunk_merged = chunk_merged / chunk_denominator[:, :, None]
        chunk_out[tile_chunk, chunk_head, :] = chunk_merged
        chunk_lse[tile_chunk, chunk_head] = chunk_max_lse + torch.log2(
            chunk_denominator
        )

    for final_head in hl.tile(query_heads, block_size=1):
        final_chunk_idx = hl.arange(merge_chunks)
        final_lse_offsets = (
            final_chunk_idx[:, None] * query_heads + final_head.index[None, :]
        )
        final_lse_values = chunk_lse_storage[final_lse_offsets]
        final_max_lse = torch.amax(final_lse_values, dim=0)
        final_weights = torch.exp2(final_lse_values - final_max_lse[None, :])
        final_value_offsets = (
            final_lse_offsets[:, :, None] * head_dim
            + hl.arange(head_dim)[None, None, :]
        )
        final_values = chunk_out_storage[final_value_offsets]
        final_denominator = torch.sum(final_weights, dim=0)
        final_merged = torch.sum(final_values * final_weights[:, :, None], dim=0)
        output[final_head, :] = (final_merged / final_denominator[:, None]).to(
            output.dtype
        )
    return output


@helion.kernel(static_shapes=True, autotune_effort="none")
def task_aligned_per_token_group_fp8_quant(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool,
    dummy_is_tma_aligned: bool,
) -> None:
    """Quantize head-shaped attention without flattening its logical axes."""
    num_kv_heads, q_per_kv, head_dim = input.shape
    assert group_size == head_dim
    assert not scale_ue8m0
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(head_dim)
    hl.specialize(group_size)
    for tile_g, tile_q, tile_n in hl.tile(
        [num_kv_heads, q_per_kv, head_dim],
        block_size=[1, None, group_size],
    ):
        value = input[tile_g, tile_q, tile_n]
        scale = torch.amax(torch.abs(value), dim=-1).clamp(min=eps) / fp8_max
        flat_group = tile_g.index[:, None] * q_per_kv + tile_q.index[None, :]
        output_s[0, flat_group] = scale
        flat_n = flat_group[:, :, None] * group_size + tile_n.index[None, None, :]
        output_q[0, flat_n] = (
            (value / scale[:, :, None]).clamp(fp8_min, fp8_max).to(output_q.dtype)
        )


@helion.kernel(static_shapes=True, autotune_effort="none")
def canonical_tiled_merge_attention_splits(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    """The same merge tiles indexed through canonical multidimensional views."""
    splits, query_heads, head_dim = partial_out.shape
    hl.specialize(splits)
    hl.specialize(query_heads)
    hl.specialize(head_dim)
    merge_chunks = 16
    hl.specialize(merge_chunks)
    assert splits % merge_chunks == 0
    splits_per_chunk = splits // merge_chunks
    hl.specialize(splits_per_chunk)

    chunk_out = torch.empty(
        (merge_chunks, query_heads, head_dim),
        dtype=torch.float32,
        device=partial_out.device,
    )
    chunk_lse = torch.empty(
        (merge_chunks, query_heads),
        dtype=torch.float32,
        device=partial_out.device,
    )
    output = torch.empty(
        (query_heads, head_dim),
        dtype=torch.bfloat16,
        device=partial_out.device,
    )

    for tile_chunk, chunk_head in hl.tile(
        [merge_chunks, query_heads], block_size=[1, 1]
    ):
        chunk_split_idx = (
            tile_chunk.index[:, None] * splits_per_chunk
            + hl.arange(splits_per_chunk)[None, :]
        )
        chunk_lse_values = partial_lse[
            chunk_split_idx[:, :, None],
            chunk_head.index[None, None, :],
        ]
        chunk_max_lse = torch.amax(chunk_lse_values, dim=1)
        chunk_weights = torch.exp2(chunk_lse_values - chunk_max_lse[:, None, :])
        chunk_values = partial_out[
            chunk_split_idx[:, :, None, None],
            chunk_head.index[None, None, :, None],
            hl.arange(head_dim)[None, None, None, :],
        ]
        chunk_denominator = torch.sum(chunk_weights, dim=1)
        chunk_merged = torch.sum(chunk_values * chunk_weights[:, :, :, None], dim=1)
        chunk_merged = chunk_merged / chunk_denominator[:, :, None]
        chunk_out[tile_chunk, chunk_head, :] = chunk_merged
        chunk_lse[tile_chunk, chunk_head] = chunk_max_lse + torch.log2(
            chunk_denominator
        )

    for final_head in hl.tile(query_heads, block_size=1):
        final_chunk_idx = hl.arange(merge_chunks)
        final_lse_values = chunk_lse[
            final_chunk_idx[:, None],
            final_head.index[None, :],
        ]
        final_max_lse = torch.amax(final_lse_values, dim=0)
        final_weights = torch.exp2(final_lse_values - final_max_lse[None, :])
        final_values = chunk_out[
            final_chunk_idx[:, None, None],
            final_head.index[None, :, None],
            hl.arange(head_dim)[None, None, :],
        ]
        final_denominator = torch.sum(final_weights, dim=0)
        final_merged = torch.sum(final_values * final_weights[:, :, None], dim=0)
        output[final_head, :] = (final_merged / final_denominator[:, None]).to(
            output.dtype
        )
    return output.view(1, query_heads, head_dim)


@helion.kernel(static_shapes=True, autotune_effort="none")
def task_aligned_tiled_merge_attention_splits(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    """Express both merge fan-ins as ordinary tile ranges."""
    splits, num_kv_heads, q_per_kv, head_dim = partial_out.shape
    hl.specialize(splits)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(head_dim)
    merge_chunks = 16
    hl.specialize(merge_chunks)
    assert splits % merge_chunks == 0
    splits_per_chunk = splits // merge_chunks
    hl.specialize(splits_per_chunk)

    chunk_out = torch.empty(
        (merge_chunks, num_kv_heads, q_per_kv, head_dim),
        dtype=torch.float32,
        device=partial_out.device,
    )
    chunk_lse = torch.empty(
        (merge_chunks, num_kv_heads, q_per_kv),
        dtype=torch.float32,
        device=partial_out.device,
    )
    output = torch.empty(
        (num_kv_heads, q_per_kv, head_dim),
        dtype=torch.bfloat16,
        device=partial_out.device,
    )

    for tile_split, tile_g, tile_q in hl.tile(
        [splits, num_kv_heads, q_per_kv],
        block_size=[splits_per_chunk, 1, None],
    ):
        chunk_lse_values = partial_lse[tile_split, tile_g, tile_q]
        chunk_max_lse = torch.amax(chunk_lse_values, dim=0)
        chunk_weights = torch.exp2(chunk_lse_values - chunk_max_lse[None, :, :])
        chunk_values = partial_out[tile_split, tile_g, tile_q, :]
        chunk_denominator = torch.sum(chunk_weights, dim=0)
        chunk_merged = torch.sum(chunk_values * chunk_weights[:, :, :, None], dim=0)
        chunk_out[tile_split.id, tile_g, tile_q, :] = (
            chunk_merged / chunk_denominator[:, :, None]
        )
        chunk_lse[tile_split.id, tile_g, tile_q] = chunk_max_lse + torch.log2(
            chunk_denominator
        )

    for tile_chunk, tile_g, tile_q in hl.tile(
        [merge_chunks, num_kv_heads, q_per_kv],
        block_size=[merge_chunks, 1, None],
    ):
        final_lse_values = chunk_lse[tile_chunk, tile_g, tile_q]
        final_max_lse = torch.amax(final_lse_values, dim=0)
        final_weights = torch.exp2(final_lse_values - final_max_lse[None, :, :])
        final_values = chunk_out[tile_chunk, tile_g, tile_q, :]
        final_denominator = torch.sum(final_weights, dim=0)
        final_merged = torch.sum(final_values * final_weights[:, :, :, None], dim=0)
        output[tile_g, tile_q, :] = (final_merged / final_denominator[:, :, None]).to(
            output.dtype
        )
    return output.view(1, num_kv_heads * q_per_kv, head_dim)


def _compile_granular_separate_kernel(kernel, kernel_args, args):
    """Compile an unchanged granular source body as its own Helion launch."""
    bound = kernel.bind(kernel_args)
    values = dict(bound.config_spec.default_config())
    values.update(
        {
            "num_warps": 1,
            "num_stages": args.kernel_stages,
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
        }
    )
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config, bound.compile_config(config)


def _build_helion_reference(args, tensors):
    """Build a separate-launch graph from the exact sources used above."""
    from helion_qwen3_layer_baseline import FFN_CONFIGS
    from helion_qwen3_layer_baseline import block_fp8_mm
    from helion_qwen3_layer_baseline import compile_config
    from helion_qwen3_layer_baseline import fused_qk_norm_rope
    from helion_qwen3_layer_baseline import paged_gqa_decode_attention_split
    from helion_qwen3_layer_baseline import per_token_group_fp8_quant
    from helion_qwen3_layer_baseline import silu_and_mul_per_block_quant

    configs = json.loads(Path(args.config_path).read_text())
    initial_residual = tensors["residual"].clone()
    initial_kv_cache = tensors["kv_cache"].clone()
    rms_args = (
        tensors["pre_q"],
        tensors["hidden_states"],
        tensors["pre_weight"],
        tensors["pre_scale"],
        args.eps,
        None,
        tensors["residual"],
        args.group,
        False,
    )
    _, rms = _compile_granular_separate_kernel(
        tiled_rms_norm_per_block_quant, rms_args, args
    )
    rms(*rms_args)
    qkv_args = (
        tensors["pre_q"],
        tensors["pre_scale"],
        tensors["qkv_weight_q"],
        tensors["qkv_weight_scale"],
        args.group,
    )
    _, qkv_mm = compile_config(block_fp8_mm, qkv_args, configs["qkv_mm"])
    qkv = qkv_mm(*qkv_args)
    qk_args = (
        qkv,
        args.q_heads,
        args.kv_heads,
        args.kv_heads,
        args.head_dim,
        args.eps,
        tensors["q_weight"],
        tensors["k_weight"],
        tensors["cos_sin"],
        True,
        tensors["position"],
        -1,
    )
    qk_source = (
        flat_fused_qk_norm_rope if _USE_TASK_ALIGNED_ATTENTION else fused_qk_norm_rope
    )
    if _USE_TASK_ALIGNED_ATTENTION:
        _, qk = _compile_granular_separate_kernel(qk_source, qk_args, args)
        qk(*qk_args)
    else:
        _, qk = compile_config(qk_source, qk_args, configs["qk_norm_rope"])
        qk(*qk_args)
    key_begin = args.q_heads * args.head_dim
    qkv_width = (args.q_heads + 2 * args.kv_heads) * args.head_dim
    query = qkv[:, :key_begin].view(1, args.q_heads, args.head_dim)
    key = qkv[:, key_begin : key_begin + args.kv_heads * args.head_dim].view(
        1, args.kv_heads, args.head_dim
    )
    value = qkv[:, key_begin + args.kv_heads * args.head_dim : qkv_width].view(
        1, args.kv_heads, args.head_dim
    )
    cache_args = (
        key,
        value,
        tensors["kv_cache"],
        tensors["slot_mapping"],
        args.block_size,
    )
    _, cache = _compile_granular_separate_kernel(
        tiled_reshape_and_cache_flash, cache_args, args
    )
    cache(*cache_args)
    split_args = (
        query,
        tensors["kv_cache"],
        tensors["block_table"],
        args.context,
        args.block_size,
        args.q_heads // args.kv_heads,
        args.attention_splits,
    )
    split_source = (
        canonical_paged_gqa_decode_attention_split
        if _USE_CANONICAL_ATTENTION_VIEWS
        else paged_gqa_decode_attention_split
    )
    _, split_kernel = compile_config(
        split_source,
        split_args,
        configs["decode_attention_split"],
    )
    partial_out, partial_lse = split_kernel(*split_args)
    merge_args = (partial_out, partial_lse)
    if _USE_CANONICAL_ATTENTION_VIEWS:
        merge_kernel = canonical_tiled_merge_attention_splits
    elif _USE_TASK_ALIGNED_ATTENTION:
        merge_kernel = task_aligned_tiled_merge_attention_splits
    else:
        merge_kernel = tiled_merge_attention_splits
    _, merge = _compile_granular_separate_kernel(merge_kernel, merge_args, args)
    attention = merge(*merge_args)
    quant_args = (
        attention.view(1, args.hidden),
        tensors["attention_q"],
        tensors["attention_scale"],
        args.group,
        1e-10,
        -448.0,
        448.0,
        False,
        False,
        False,
    )
    _, attention_quant = compile_config(
        per_token_group_fp8_quant,
        quant_args,
        configs["attention_quant"],
    )
    attention_quant(*quant_args)
    o_args = (
        tensors["attention_q"],
        tensors["attention_scale"],
        tensors["o_weight_q"],
        tensors["o_weight_scale"],
        args.group,
    )
    _, o_mm = compile_config(block_fp8_mm, o_args, configs["o_mm"])
    attention_out = o_mm(*o_args)
    post_args = (
        tensors["ffn_q"],
        attention_out,
        tensors["post_weight"],
        tensors["ffn_scale"],
        args.eps,
        None,
        tensors["residual"],
        args.group,
        False,
    )
    rms(*post_args)
    w13_args = (
        tensors["ffn_q"],
        tensors["ffn_scale"],
        tensors["w13_q"],
        tensors["w13_scale"],
        args.group,
    )
    _, w13 = compile_config(block_fp8_mm, w13_args, FFN_CONFIGS["w13"])
    gate_up = w13(*w13_args)
    silu_args = (gate_up, args.group)
    _, silu = compile_config(
        silu_and_mul_per_block_quant,
        silu_args,
        FFN_CONFIGS["silu_quant"],
    )
    activation_q, activation_scale = silu(*silu_args)
    w2_args = (
        activation_q,
        activation_scale,
        tensors["w2_q"],
        tensors["w2_scale"],
        args.group,
    )
    _, w2 = compile_config(block_fp8_mm, w2_args, FFN_CONFIGS["w2"])
    output = w2(*w2_args)

    # Compilation materializes each downstream tensor by executing its kernel.
    # Restore the two externally mutable inputs so the caller's first launch is
    # iteration one, exactly like the persistent path used for validation.
    tensors["residual"].copy_(initial_residual)
    tensors["kv_cache"].copy_(initial_kv_cache)

    def launch():
        rms(*rms_args)
        local_qkv = qkv_mm(*qkv_args)
        qk(local_qkv, *qk_args[1:])
        local_query = local_qkv[:, :key_begin].view(1, args.q_heads, args.head_dim)
        local_key = local_qkv[
            :, key_begin : key_begin + args.kv_heads * args.head_dim
        ].view(1, args.kv_heads, args.head_dim)
        local_value = local_qkv[
            :, key_begin + args.kv_heads * args.head_dim : qkv_width
        ].view(1, args.kv_heads, args.head_dim)
        cache(
            local_key,
            local_value,
            tensors["kv_cache"],
            tensors["slot_mapping"],
            args.block_size,
        )
        local_partials, local_lse = split_kernel(
            local_query,
            tensors["kv_cache"],
            tensors["block_table"],
            args.context,
            args.block_size,
            args.q_heads // args.kv_heads,
            args.attention_splits,
        )
        local_attention = merge(local_partials, local_lse)
        attention_quant(
            local_attention.view(1, args.hidden),
            tensors["attention_q"],
            tensors["attention_scale"],
            args.group,
            1e-10,
            -448.0,
            448.0,
            False,
            False,
            False,
        )
        local_attention_out = o_mm(*o_args)
        rms(
            tensors["ffn_q"],
            local_attention_out,
            tensors["post_weight"],
            tensors["ffn_scale"],
            args.eps,
            None,
            tensors["residual"],
            args.group,
            False,
        )
        local_gate = w13(*w13_args)
        local_activation_q, local_activation_scale = silu(local_gate, args.group)
        local_output = w2(
            local_activation_q,
            local_activation_scale,
            tensors["w2_q"],
            tensors["w2_scale"],
            args.group,
        )
        return (
            local_output,
            local_qkv,
            local_partials,
            local_lse,
            local_attention,
            local_attention_out,
            local_gate,
            local_activation_q,
            local_activation_scale,
        )

    return launch, {
        "output": output,
        "qkv": qkv,
        "partial_out": partial_out,
        "partial_lse": partial_lse,
        "attention": attention,
        "attention_out": attention_out,
        "gate_up": gate_up,
        "activation_q": activation_q,
        "activation_scale": activation_scale,
    }


def _build_original_helion_reference(*args: object, **kwargs: object):
    from triton_qwen3_whole_layer_persistent import build_helion_reference

    return build_helion_reference(*args, **kwargs)


def _probe_config(bound, args):
    """Map the retained one-warp probe geometry onto the granular source."""
    values = dict(bound.config_spec.default_config())
    values.pop(CROSS_LOOP_NUM_WORKERS_CONFIG, None)
    uses_flat_qk = _USE_TASK_ALIGNED_ATTENTION or _USE_CANONICAL_ATTENTION_VIEWS
    downstream_shift = (
        2
        if _USE_TASK_ALIGNED_ATTENTION
        else -1
        if _USE_CANONICAL_ATTENTION_VIEWS
        else 0
    )
    block_size_by_id = {
        7: 8,  # QKV output tile
        (16 if uses_flat_qk else 17): 4,
        (18 if uses_flat_qk else 19): args.attention_context_block,
        24 + downstream_shift: (
            args.merge_q_block if _USE_TASK_ALIGNED_ATTENTION else 1
        ),
        27 + downstream_shift: 8,  # O output tile
        36 + downstream_shift: 16,  # W13 output tile
        41 + downstream_shift: 8,  # W2 output tile
    }
    if not _USE_TASK_ALIGNED_ATTENTION:
        block_size_by_id[10] = args.qk_head_block
    if _USE_TASK_ALIGNED_ATTENTION:
        block_size_by_id[21] = args.merge_q_block
        block_size_by_id[24] = args.merge_q_block
    values["block_sizes"] = [
        block_size_by_id.get(spec.block_id, default)
        for spec, default in zip(
            bound.config_spec.block_sizes, values["block_sizes"], strict=True
        )
    ]
    loop_orders = [
        [0, 1],
        [0, 1, 2],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2],
        [2, 1, 0],
        [0, 1],
        [0, 1, 2],
        [1, 0],
        [0, 1],
        [0, 1, 2],
        [0, 1],
        [0, 1],
        [0, 1],
    ]
    if uses_flat_qk:
        loop_orders[3] = [0, 1]
    if _USE_TASK_ALIGNED_ATTENTION:
        loop_orders[6] = [0, 1, 2]
        loop_orders.insert(7, [0, 1, 2])
    values["loop_orders"] = loop_orders
    values["l2_groupings"] = [1] * len(bound.config_spec.l2_groupings)

    def by_block_id(specs, choices, default):
        return [
            next(
                (
                    choices[block_id]
                    for block_id in spec.block_ids
                    if block_id in choices
                ),
                default,
            )
            for spec in specs
        ]

    qkv_range = 8
    attention_range = 18 if uses_flat_qk else 19
    projection_ranges = {
        qkv_range: 4,
        28 + downstream_shift: 4,
        37 + downstream_shift: 4,
        42 + downstream_shift: 4,
    }
    values["range_num_stages"] = by_block_id(
        bound.config_spec.range_num_stages, projection_ranges, 0
    )
    values["range_unroll_factors"] = by_block_id(
        bound.config_spec.range_unroll_factors,
        {
            qkv_range: 2,
            28 + downstream_shift: 2,
            37 + downstream_shift: 2,
            42 + downstream_shift: 4,
        },
        0,
    )
    values["range_multi_buffers"] = by_block_id(
        bound.config_spec.range_multi_buffers,
        {
            qkv_range: True,
            attention_range: True,
            28 + downstream_shift: False,
            37 + downstream_shift: True,
            42 + downstream_shift: False,
        },
        None,
    )
    values["range_flattens"] = by_block_id(
        bound.config_spec.range_flattens,
        {
            qkv_range: False,
            attention_range: True,
            28 + downstream_shift: False,
            37 + downstream_shift: False,
            42 + downstream_shift: True,
        },
        None,
    )
    values.update(
        {
            "num_warps": 1,
            "num_stages": args.kernel_stages,
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
        }
    )
    if CROSS_LOOP_NUM_WORKERS_CONFIG in bound.config_spec.user_defined_tunables:
        values[CROSS_LOOP_NUM_WORKERS_CONFIG] = 1024
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


def main() -> None:
    global _USE_CANONICAL_ATTENTION_VIEWS, _USE_TASK_ALIGNED_ATTENTION

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strict-validation", action="store_true")
    parser.add_argument("--no-waits", action="store_true")
    parser.add_argument("--skip-wait-prefix", action="append", default=[])
    parser.add_argument("--dump-accesses", action="store_true")
    parser.add_argument("--canonical-attention-views", action="store_true")
    parser.add_argument("--task-aligned-attention", action="store_true")
    parser.add_argument(
        "--reference",
        choices=("same_source", "tuned"),
        default="same_source",
    )
    args, remaining = parser.parse_known_args()
    if args.canonical_attention_views and args.task_aligned_attention:
        parser.error(
            "--canonical-attention-views and --task-aligned-attention are "
            "mutually exclusive"
        )
    _USE_CANONICAL_ATTENTION_VIEWS = args.canonical_attention_views
    _USE_TASK_ALIGNED_ATTENTION = args.task_aligned_attention

    if args.no_waits:
        ForEachProgramID._wait_for_counter = staticmethod(
            lambda **_kwargs: [ast.Pass()]
        )
    elif args.skip_wait_prefix:
        original_wait_for_counter = ForEachProgramID._wait_for_counter

        def filtered_wait_for_counter(**kwargs):
            if any(
                kwargs["prefix"].startswith(prefix) for prefix in args.skip_wait_prefix
            ):
                return [ast.Pass()]
            return original_wait_for_counter(**kwargs)

        ForEachProgramID._wait_for_counter = staticmethod(filtered_wait_for_counter)
    compatibility = types.ModuleType("triton_qwen3_sm_overlap_probe")
    compatibility.build_helion_reference = (
        _build_helion_reference
        if args.reference == "same_source"
        else _build_original_helion_reference
    )
    sys.modules.setdefault("triton_qwen3_sm_overlap_probe", compatibility)

    import helion_qwen3_tile_dependency as probe

    probe.rms_norm_per_block_quant = tiled_rms_norm_per_block_quant
    probe.reshape_and_cache_flash = tiled_reshape_and_cache_flash
    if args.task_aligned_attention or args.canonical_attention_views:
        probe.fused_qk_norm_rope = flat_fused_qk_norm_rope
    if args.task_aligned_attention:
        probe.paged_gqa_decode_attention_split = (
            task_aligned_paged_gqa_decode_attention_split
        )
        probe.per_token_group_fp8_quant = task_aligned_per_token_group_fp8_quant
    if args.canonical_attention_views:
        probe.paged_gqa_decode_attention_split = (
            canonical_paged_gqa_decode_attention_split
        )
    if args.canonical_attention_views:
        probe.merge_attention_splits = canonical_tiled_merge_attention_splits
    elif args.task_aligned_attention:
        probe.merge_attention_splits = task_aligned_tiled_merge_attention_splits
    else:
        probe.merge_attention_splits = tiled_merge_attention_splits
    probe._probe_matched_config = _probe_config
    if args.task_aligned_attention:
        original_compose = probe._compose_qwen3_layer_source

        def compose_task_aligned_source() -> str:
            return original_compose().replace(
                "attention_flat = attention.view(1, hidden)",
                "attention_flat = attention",
            )

        probe._compose_qwen3_layer_source = compose_task_aligned_source
    kernel, source = probe._build_composite_kernel()
    probe.GENERATED_SOURCE = source
    probe.qwen3_layer_tile_dependency = helion.kernel(
        static_shapes=True,
        autotune_effort="none",
    )(kernel.fn)
    sys.argv = [sys.argv[0], *remaining]
    if args.dump_accesses:

        def dump_run(probe_args) -> None:
            tensors = probe.allocate_layer(probe_args)
            bound = probe.qwen3_layer_tile_dependency.bind(
                probe._composite_args(tensors, probe_args)
            )
            host_function = bound.host_function
            assert host_function is not None
            dependency_plan = host_function.device_ir.tile_dependency_graph
            assert dependency_plan is not None
            print(
                "TASK_FAMILIES",
                [
                    dataclasses.asdict(family)
                    for family in host_function.device_ir.task_families
                ],
                flush=True,
            )
            for root in range(len(host_function.device_ir.task_families)):
                accesses = tuple(
                    access for access in dependency_plan.accesses if access.root == root
                )
                print(
                    "CROSS_LOOP_ACCESSES",
                    root,
                    [dataclasses.asdict(access) for access in accesses],
                    flush=True,
                )
            for edge in dependency_plan.edges:
                print(
                    "CROSS_LOOP_EDGE",
                    edge.producer_root,
                    edge.consumer_root,
                    dataclasses.asdict(edge),
                    flush=True,
                )

        probe.run = dump_run
        probe.main()
        return
    if not args.strict_validation:
        probe.main()
        return

    allocations: list[dict[str, torch.Tensor]] = []
    original_allocate_layer = probe.allocate_layer
    original_assert_close = torch.testing.assert_close
    validation_names = iter(
        (
            "output",
            "qkv",
            "partial_out",
            "partial_lse",
            "attention",
            "attention_out",
            "gate_up",
            "activation_q",
            "activation_scale",
        )
    )

    def tracked_allocate_layer(namespace):
        tensors = original_allocate_layer(namespace)
        allocations.append(tensors)
        return tensors

    def report(name: str, actual: object, expected: object) -> None:
        actual_tensor = torch.as_tensor(actual)
        expected_tensor = torch.as_tensor(expected)
        if actual_tensor.numel() == expected_tensor.numel():
            actual_tensor = actual_tensor.view_as(expected_tensor)
        difference = (actual_tensor.float() - expected_tensor.float()).abs()
        print(
            "STRICT_VALIDATION",
            name,
            {
                "exact": bool(torch.equal(actual_tensor, expected_tensor)),
                "mismatches": int((actual_tensor != expected_tensor).sum().item()),
                "elements": actual_tensor.numel(),
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
            },
            flush=True,
        )

    checked_bridge_state = False

    def diagnostic_assert_close(actual, expected, **kwargs: object) -> None:
        nonlocal checked_bridge_state
        if not checked_bridge_state and len(allocations) >= 2:
            checked_bridge_state = True
            persistent_tensors, reference_tensors = allocations[:2]
            for bridge_name in (
                "pre_q",
                "pre_scale",
                "kv_cache",
                "attention_q",
                "attention_scale",
                "ffn_q",
                "ffn_scale",
                "residual",
            ):
                report(
                    bridge_name,
                    persistent_tensors[bridge_name],
                    reference_tensors[bridge_name],
                )
        report(next(validation_names, "unexpected"), actual, expected)
        original_assert_close(actual, expected, **kwargs)

    probe.allocate_layer = tracked_allocate_layer
    torch.testing.assert_close = diagnostic_assert_close
    try:
        probe.main()
    finally:
        probe.allocate_layer = original_allocate_layer
        torch.testing.assert_close = original_assert_close


if __name__ == "__main__":
    main()
