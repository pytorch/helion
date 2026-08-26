# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""Helion/Triton implementation of a production Gemma 4 E4B decode layer.

The target is ``google/gemma-4-E4B-it`` at TP=1.  The benchmark covers the
four layer variants in the checkpoint: sliding/full attention crossed with
ordinary/YOCO KV-shared layers.  Every compute stage in vLLM's decoder-layer
path has a Helion equivalent, and the optimized graph fuses the small stages
whose intermediate tensors are not externally observable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from probes.common import benchmark_cache_mode
from probes.gemma4.common import E4B_LAYER_COUNTS
from probes.gemma4.common import E4B_REPRESENTATIVE_LAYERS
from probes.gemma4.common import Gemma4E4BShape
from probes.gemma4.common import allocate_layer
from probes.gemma4.common import benchmark_interleaved
from probes.gemma4.common import capture
from probes.gemma4.common import layer_reference
from probes.gemma4.common import paged_attention_reference
from probes.gemma4.common import require_idle_visible_gpu
from probes.gemma4.common import variant_name
from probes.gemma4.common import visible_gpu_pids

import helion
import helion.language as hl


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    coefficient = 0.7978845608028654
    return 0.5 * x * (1.0 + torch.tanh(coefficient * (x + 0.044715 * x * x * x)))


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """vLLM RMSNorm semantics, including the BF16 pre-weight rounding point."""
    m, n = x.size()
    hl.specialize(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m, block_size=1):
        values = x[tile_m, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(values * values, dim=-1) + eps)
        normalized = (values * inv_rms[:, None]).to(x.dtype)
        out[tile_m, :] = normalized * weight[None, :]
    return out


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def bf16_mm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """BF16 linear without bias, with weights stored as [N, K]."""
    m, k = x.size()
    n, weight_k = weight.size()
    assert k == weight_k
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                x[tile_m, tile_k],
                weight[tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def rms_qkv_mm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse input RMSNorm into the QKV projection for single-token decode."""
    m, k = x.size()
    n, weight_k = qkv_weight.size()
    assert weight_k == k
    assert m == 1
    hl.specialize(k)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        squared_sum = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            values = x[tile_m, tile_k].to(torch.float32)
            squared_sum = squared_sum + torch.sum(values * values, dim=-1)
        inv_rms = torch.rsqrt(squared_sum * (1.0 / k) + eps)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            values = x[tile_m, tile_k].to(torch.float32)
            normalized = (values * inv_rms[:, None]).to(x.dtype)
            normalized = normalized * norm_weight[tile_k]
            acc = torch.addmm(
                acc,
                normalized,
                qkv_weight[tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def qkv_norm_rope_baseline(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    position,
    num_q_heads,
    num_kv_heads,
    head_dim,
    eps,
    process_kv,
):
    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    q, k, v = qkv.split([q_width, kv_width, kv_width], dim=-1)

    def normalize(value, weight):
        value = value.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        value = value.to(qkv.dtype)
        return value if weight is None else value * weight

    q = normalize(q.view(-1, num_q_heads, head_dim), q_weight)
    if process_kv:
        k = normalize(k.view(-1, num_kv_heads, head_dim), k_weight)
        v = normalize(v.view(-1, num_kv_heads, head_dim), None)
    else:
        k = k.view(-1, num_kv_heads, head_dim)
        v = v.view(-1, num_kv_heads, head_dim)

    rotary_dim = cos_sin.shape[-1]
    half = rotary_dim // 2
    cache = cos_sin[position]
    cos = cache[..., :half][:, None, :]
    sin = cache[..., half:][:, None, :]

    def rotate(value):
        x1 = value[..., :half]
        x2 = value[..., half:rotary_dim]
        rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        return torch.cat((rotated, value[..., rotary_dim:]), dim=-1)

    q = rotate(q)
    k = rotate(k)
    qkv[:, :q_width].copy_(q.reshape_as(qkv[:, :q_width]))
    qkv[:, q_width : q_width + kv_width].copy_(k.reshape(-1, kv_width))
    if process_kv:
        qkv[:, q_width + kv_width :].copy_(v.reshape(-1, kv_width))


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=qkv_norm_rope_baseline,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=5e-2,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def qkv_norm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    position: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    process_kv: bool,
) -> None:
    """Fuse Q/K/V RMSNorm and partial NeoX RoPE over packed QKV."""
    num_tokens = qkv.shape[0]
    total_heads = num_q_heads + 2 * num_kv_heads
    qk_heads = num_q_heads + num_kv_heads
    processed_heads = total_heads if process_kv else qk_heads
    rotary_dim = cos_sin.shape[-1]
    half = rotary_dim // 2
    hl.specialize(qkv.shape[1])
    hl.specialize(num_q_heads)
    hl.specialize(num_kv_heads)
    hl.specialize(head_dim)
    hl.specialize(rotary_dim)
    packed = qkv.view(num_tokens, total_heads, head_dim)

    for tile_m, tile_h, tile_d in hl.tile(
        [num_tokens, processed_heads, head_dim],
        block_size=[1, None, head_dim],
    ):
        values = packed[tile_m, tile_h, tile_d].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(values * values, dim=-1) + eps)
        is_q = (tile_h.index < num_q_heads)[None, :, None]
        is_k = ((tile_h.index >= num_q_heads) & (tile_h.index < qk_heads))[
            None, :, None
        ]
        learned_weight = torch.where(
            is_q,
            q_weight[None, None, tile_d],
            k_weight[None, None, tile_d],
        )
        learned_weight = torch.where(is_q | is_k, learned_weight, 1.0)
        normalized = (values * inv_rms[:, :, None]).to(qkv.dtype) * learned_weight
        if process_kv:
            processed = normalized
        else:
            processed = torch.where(is_q, normalized, values.to(qkv.dtype))
        packed[tile_m, tile_h, tile_d] = processed

        rotary_head = (tile_h.index < qk_heads)[None, :, None]
        rotary_offset = hl.arange(half)
        x1 = packed[tile_m, tile_h, rotary_offset]
        x2 = packed[tile_m, tile_h, rotary_offset + half]
        pos = position[tile_m]
        cos = cos_sin[pos, rotary_offset]
        sin = cos_sin[pos, rotary_offset + half]
        o1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        o2 = x2 * cos[:, None, :] + x1 * sin[:, None, :]
        hl.store(
            packed,
            [
                tile_m.index[:, None, None],
                tile_h.index[None, :, None],
                rotary_offset[None, None, :],
            ],
            o1,
            extra_mask=rotary_head,
        )
        hl.store(
            packed,
            [
                tile_m.index[:, None, None],
                tile_h.index[None, :, None],
                (rotary_offset + half)[None, None, :],
            ],
            o2,
            extra_mask=rotary_head,
        )


def qkv_norm_rope_cache_baseline(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    position,
    kv_cache,
    slot_mapping,
    num_q_heads,
    num_kv_heads,
    head_dim,
    block_size,
    eps,
):
    qkv_norm_rope_baseline(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        position,
        num_q_heads,
        num_kv_heads,
        head_dim,
        eps,
        True,
    )
    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    slot = int(slot_mapping[0].item())
    block = slot // block_size
    offset = slot % block_size
    kv_cache[block, offset, :, :head_dim].copy_(
        qkv[:, q_width : q_width + kv_width].view(num_kv_heads, head_dim)
    )
    kv_cache[block, offset, :, head_dim:].copy_(
        qkv[:, q_width + kv_width :].view(num_kv_heads, head_dim)
    )


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=qkv_norm_rope_cache_baseline,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=5e-2,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def qkv_norm_rope_cache(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    position: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    eps: float,
) -> None:
    """Fuse Q/K/V normalization, RoPE, and the paged-cache update."""
    num_tokens = qkv.shape[0]
    total_heads = num_q_heads + 2 * num_kv_heads
    qk_heads = num_q_heads + num_kv_heads
    rotary_dim = cos_sin.shape[-1]
    half = rotary_dim // 2
    hl.specialize(qkv.shape[1])
    hl.specialize(num_q_heads)
    hl.specialize(num_kv_heads)
    hl.specialize(head_dim)
    hl.specialize(rotary_dim)
    hl.specialize(block_size)
    packed = qkv.view(num_tokens, total_heads, head_dim)

    for tile_m, tile_h, tile_d in hl.tile(
        [num_tokens, total_heads, head_dim],
        block_size=[1, None, head_dim],
    ):
        values = packed[tile_m, tile_h, tile_d].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(values * values, dim=-1) + eps)
        is_q = (tile_h.index < num_q_heads)[None, :, None]
        is_k = ((tile_h.index >= num_q_heads) & (tile_h.index < qk_heads))[
            None, :, None
        ]
        is_v = (tile_h.index >= qk_heads)[None, :, None]
        learned_weight = torch.where(
            is_q,
            q_weight[None, None, tile_d],
            k_weight[None, None, tile_d],
        )
        learned_weight = torch.where(is_q | is_k, learned_weight, 1.0)
        normalized = (values * inv_rms[:, :, None]).to(qkv.dtype) * learned_weight
        packed[tile_m, tile_h, tile_d] = normalized

        slot = slot_mapping[tile_m.index]
        cache_block = (slot // block_size)[:, None, None]
        cache_offset = (slot % block_size)[:, None, None]
        d = tile_d.index[None, None, :]
        hl.store(
            kv_cache,
            [
                cache_block,
                cache_offset,
                (tile_h.index - num_q_heads)[None, :, None],
                d,
            ],
            normalized,
            extra_mask=is_k,
        )
        hl.store(
            kv_cache,
            [
                cache_block,
                cache_offset,
                (tile_h.index - qk_heads)[None, :, None],
                d + head_dim,
            ],
            normalized,
            extra_mask=is_v,
        )

        rotary_head = (tile_h.index < qk_heads)[None, :, None]
        rotary_offset = hl.arange(half)
        x1 = packed[tile_m, tile_h, rotary_offset]
        x2 = packed[tile_m, tile_h, rotary_offset + half]
        pos = position[tile_m]
        cos = cos_sin[pos, rotary_offset]
        sin = cos_sin[pos, rotary_offset + half]
        o1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        o2 = x2 * cos[:, None, :] + x1 * sin[:, None, :]
        hl.store(
            packed,
            [
                tile_m.index[:, None, None],
                tile_h.index[None, :, None],
                rotary_offset[None, None, :],
            ],
            o1,
            extra_mask=rotary_head,
        )
        hl.store(
            packed,
            [
                tile_m.index[:, None, None],
                tile_h.index[None, :, None],
                (rotary_offset + half)[None, None, :],
            ],
            o2,
            extra_mask=rotary_head,
        )
        hl.store(
            kv_cache,
            [
                cache_block,
                cache_offset,
                (tile_h.index - num_q_heads)[None, :, None],
                rotary_offset[None, None, :],
            ],
            o1,
            extra_mask=is_k,
        )
        hl.store(
            kv_cache,
            [
                cache_block,
                cache_offset,
                (tile_h.index - num_q_heads)[None, :, None],
                (rotary_offset + half)[None, None, :],
            ],
            o2,
            extra_mask=is_k,
        )


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Store the current normalized K/V token in vLLM's paged-cache layout."""
    num_tokens, num_kv_heads, head_dim = key.shape
    hl.specialize(num_kv_heads)
    hl.specialize(head_dim)
    hl.specialize(block_size)
    for tile_t, tile_h, tile_d in hl.tile(
        [num_tokens, num_kv_heads, head_dim], block_size=[1, None, None]
    ):
        slot = slot_mapping[tile_t.index]
        block = (slot // block_size)[:, None, None]
        offset = (slot % block_size)[:, None, None]
        h = tile_h.index[None, :, None]
        d = tile_d.index[None, None, :]
        hl.store(kv_cache, [block, offset, h, d], key[tile_t, tile_h, tile_d])
        hl.store(
            kv_cache,
            [block, offset, h, d + head_dim],
            value[tile_t, tile_h, tile_d],
        )


def attention_split_baseline(
    query,
    kv_cache,
    block_table,
    context,
    attention_context,
    block_size,
    q_per_kv,
    splits,
):
    start = context - attention_context
    split_context = attention_context // splits
    num_kv_heads = kv_cache.shape[2]
    head_dim = query.shape[-1]
    q = query.view(num_kv_heads, q_per_kv, head_dim).float()
    output = torch.empty(
        (splits, num_kv_heads, q_per_kv, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    lse = torch.empty(
        (splits, num_kv_heads, q_per_kv),
        device=query.device,
        dtype=torch.float32,
    )
    for split in range(splits):
        logical = torch.arange(
            start + split * split_context,
            start + (split + 1) * split_context,
            device=query.device,
        )
        blocks = block_table[0, logical // block_size].long()
        offsets = logical % block_size
        values = kv_cache[blocks, offsets]
        key = values[..., :head_dim].permute(1, 0, 2).float()
        value = values[..., head_dim:].permute(1, 0, 2).float()
        scores = torch.einsum("gqd,gkd->gqk", q, key)
        lse[split] = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
        output[split] = torch.einsum(
            "gqk,gkd->gqd", torch.softmax(scores, dim=-1), value
        )
    return output, lse


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=attention_split_baseline,
    autotune_baseline_atol=1.5e-1,
    autotune_baseline_rtol=5e-2,
    backend="triton",
)
def paged_attention_split(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    attention_context: int,
    block_size: int,
    q_per_kv: int,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-KV Gemma 4 decode attention with scale=1.0."""
    _, num_q_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[2]
    assert num_q_heads == num_kv_heads * q_per_kv
    assert attention_context % splits == 0
    hl.specialize(head_dim)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(context)
    hl.specialize(attention_context)
    hl.specialize(block_size)
    hl.specialize(splits)
    split_context = attention_context // splits
    context_start = context - attention_context
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
    qk_scale = 1.4426950408889634
    for tile_split, tile_g, tile_q in hl.tile(
        [splits, num_kv_heads, q_per_kv], block_size=[1, 1, None]
    ):
        m_i = hl.full([tile_g, tile_q], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_g, tile_q], 1.0, dtype=torch.float32)
        acc = hl.zeros([tile_g, tile_q, head_dim], dtype=torch.float32)
        split_idx = tile_split.begin
        q_block = (q[tile_g, tile_q, :] * qk_scale).to(q.dtype)
        for tile_local_n in hl.tile(split_context):
            logical = context_start + split_idx * split_context + tile_local_n.index
            physical_block = block_table[0, logical // block_size]
            block_offset = logical % block_size
            d = hl.arange(head_dim)
            key = hl.load(
                kv_cache,
                [
                    physical_block[None, :, None],
                    block_offset[None, :, None],
                    tile_g.index[:, None, None],
                    d[None, None, :],
                ],
            )
            scores = torch.bmm(q_block, key.transpose(1, 2), torch.float32)
            next_max = torch.maximum(m_i, torch.amax(scores, dim=-1))
            probabilities = torch.exp2(scores - next_max[:, :, None])
            alpha = torch.exp2(m_i - next_max)
            l_i = l_i * alpha + torch.sum(probabilities, dim=-1)
            acc = acc * alpha[:, :, None]
            value = hl.load(
                kv_cache,
                [
                    physical_block[None, :, None],
                    block_offset[None, :, None],
                    tile_g.index[:, None, None],
                    (d + head_dim)[None, None, :],
                ],
            )
            acc = torch.baddbmm(acc, probabilities.to(value.dtype), value)
            m_i = next_max
        partial_out[tile_split, tile_g, tile_q, :] = (acc / l_i[:, :, None])[
            None, :, :, :
        ]
        partial_lse[tile_split, tile_g, tile_q] = (m_i + torch.log2(l_i))[None, :, :]
    return partial_out, partial_lse


def attention_baseline(
    query,
    kv_cache,
    block_table,
    context,
    attention_context,
    block_size,
    q_per_kv,
):
    return paged_attention_reference(
        query,
        kv_cache,
        block_table,
        context,
        attention_context,
        block_size,
        q_per_kv,
    )


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=attention_baseline,
    autotune_baseline_atol=1.5e-1,
    autotune_baseline_rtol=5e-2,
    backend="triton",
)
def paged_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    attention_context: int,
    block_size: int,
    q_per_kv: int,
) -> torch.Tensor:
    """One-pass paged decode attention for short sliding windows."""
    _, num_q_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[2]
    assert num_q_heads == num_kv_heads * q_per_kv
    hl.specialize(head_dim)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(context)
    hl.specialize(attention_context)
    hl.specialize(block_size)
    context_start = context - attention_context
    q = query.view(num_kv_heads, q_per_kv, head_dim)
    output = torch.empty_like(query)
    output = output.view(num_kv_heads, q_per_kv, head_dim)
    qk_scale = 1.4426950408889634
    for tile_g, tile_q in hl.tile([num_kv_heads, q_per_kv], block_size=[1, None]):
        m_i = hl.full([tile_g, tile_q], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_g, tile_q], 1.0, dtype=torch.float32)
        acc = hl.zeros([tile_g, tile_q, head_dim], dtype=torch.float32)
        q_block = (q[tile_g, tile_q, :] * qk_scale).to(q.dtype)
        for tile_local_n in hl.tile(attention_context):
            logical = context_start + tile_local_n.index
            physical_block = block_table[0, logical // block_size]
            block_offset = logical % block_size
            d = hl.arange(head_dim)
            key = hl.load(
                kv_cache,
                [
                    physical_block[None, :, None],
                    block_offset[None, :, None],
                    tile_g.index[:, None, None],
                    d[None, None, :],
                ],
            )
            scores = torch.bmm(q_block, key.transpose(1, 2), torch.float32)
            next_max = torch.maximum(m_i, torch.amax(scores, dim=-1))
            probabilities = torch.exp2(scores - next_max[:, :, None])
            alpha = torch.exp2(m_i - next_max)
            l_i = l_i * alpha + torch.sum(probabilities, dim=-1)
            acc = acc * alpha[:, :, None]
            value = hl.load(
                kv_cache,
                [
                    physical_block[None, :, None],
                    block_offset[None, :, None],
                    tile_g.index[:, None, None],
                    (d + head_dim)[None, None, :],
                ],
            )
            acc = torch.baddbmm(acc, probabilities.to(value.dtype), value)
            m_i = next_max
        output[tile_g, tile_q, :] = (acc / l_i[:, :, None]).to(output.dtype)
    return output.view(1, num_q_heads, head_dim)


def q_norm_rope_baseline(
    query,
    q_weight,
    cos_sin,
    position,
    num_q_heads,
    head_dim,
    eps,
):
    q = query.view(-1, num_q_heads, head_dim)
    values = q.float()
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    values = values.to(query.dtype) * q_weight
    rotary_dim = cos_sin.shape[-1]
    half = rotary_dim // 2
    cache = cos_sin[position]
    cos = cache[..., :half][:, None, :]
    sin = cache[..., half:][:, None, :]
    x1 = values[..., :half]
    x2 = values[..., half:rotary_dim]
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    values = torch.cat((rotated, values[..., rotary_dim:]), dim=-1)
    query.copy_(values.reshape_as(query))


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=q_norm_rope_baseline,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=5e-2,
    backend="triton",
)
def q_norm_rope(
    query: torch.Tensor,
    q_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    position: torch.Tensor,
    num_q_heads: int,
    head_dim: int,
    eps: float,
) -> None:
    """Q-only RMSNorm and partial NeoX RoPE for KV-shared layers."""
    num_tokens = query.shape[0]
    rotary_dim = cos_sin.shape[-1]
    half = rotary_dim // 2
    hl.specialize(query.shape[1])
    hl.specialize(num_q_heads)
    hl.specialize(head_dim)
    hl.specialize(rotary_dim)
    q = query.view(num_tokens, num_q_heads, head_dim)
    for tile_m, tile_h, tile_d in hl.tile(
        [num_tokens, num_q_heads, head_dim], block_size=[1, None, head_dim]
    ):
        values = q[tile_m, tile_h, tile_d].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(values * values, dim=-1) + eps)
        normalized = (values * inv_rms[:, :, None]).to(query.dtype) * q_weight[
            None, None, tile_d
        ]
        q[tile_m, tile_h, tile_d] = normalized
        rotary_offset = hl.arange(half)
        x1 = q[tile_m, tile_h, rotary_offset]
        x2 = q[tile_m, tile_h, rotary_offset + half]
        pos = position[tile_m]
        cos = cos_sin[pos, rotary_offset]
        sin = cos_sin[pos, rotary_offset + half]
        q[tile_m, tile_h, rotary_offset] = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        q[tile_m, tile_h, rotary_offset + half] = (
            x2 * cos[:, None, :] + x1 * sin[:, None, :]
        )


def merge_attention_baseline(partial_out, partial_lse):
    max_lse = partial_lse.amax(dim=0)
    weights = torch.exp2(partial_lse - max_lse[None])
    output = (partial_out * weights[..., None]).sum(dim=0)
    output = output / weights.sum(dim=0)[..., None]
    return output.to(torch.bfloat16).reshape(1, -1, partial_out.shape[-1])


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=merge_attention_baseline,
    autotune_baseline_atol=3e-2,
    autotune_baseline_rtol=3e-2,
    backend="triton",
)
def merge_attention(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    splits, num_kv_heads, q_per_kv, head_dim = partial_out.shape
    output = torch.empty(
        (num_kv_heads, q_per_kv, head_dim),
        device=partial_out.device,
        dtype=torch.bfloat16,
    )
    for tile_g, tile_q in hl.tile([num_kv_heads, q_per_kv], block_size=[1, None]):
        max_lse = hl.full([tile_g, tile_q], float("-inf"), dtype=torch.float32)
        denominator = hl.zeros([tile_g, tile_q], dtype=torch.float32)
        accumulator = hl.zeros([tile_g, tile_q, head_dim], dtype=torch.float32)
        for tile_split in hl.tile(splits):
            lse = partial_lse[tile_split, tile_g, tile_q]
            next_max = torch.maximum(max_lse, torch.amax(lse, dim=0))
            old_weight = torch.exp2(max_lse - next_max)
            weights = torch.exp2(lse - next_max[None, :, :])
            denominator = denominator * old_weight + torch.sum(weights, dim=0)
            values = partial_out[tile_split, tile_g, tile_q, :]
            accumulator = accumulator * old_weight[:, :, None] + torch.sum(
                values * weights[:, :, :, None], dim=0
            )
            max_lse = next_max
        output[tile_g, tile_q, :] = (accumulator / denominator[:, :, None]).to(
            output.dtype
        )
    return output.view(1, num_kv_heads * q_per_kv, head_dim)


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def post_attention_residual_pre_ff_norm(
    attention_out: torch.Tensor,
    residual: torch.Tensor,
    post_attention_weight: torch.Tensor,
    pre_ff_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse post-attention RMSNorm, residual add, and pre-FF RMSNorm."""
    m, hidden = attention_out.size()
    hl.specialize(hidden)
    updated_residual = torch.empty_like(residual)
    ff_input = torch.empty_like(attention_out)
    for tile_m in hl.tile(m, block_size=1):
        attention = attention_out[tile_m, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(attention * attention, dim=-1) + eps)
        normalized = (attention * inv_rms[:, None]).to(attention_out.dtype)
        updated = normalized * post_attention_weight[None, :] + residual[tile_m, :]
        updated_residual[tile_m, :] = updated
        values = updated.to(torch.float32)
        ff_inv_rms = torch.rsqrt(torch.mean(values * values, dim=-1) + eps)
        ff_input[tile_m, :] = (values * ff_inv_rms[:, None]).to(
            attention_out.dtype
        ) * pre_ff_weight[None, :]
    return updated_residual, ff_input


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def geglu(gate_up: torch.Tensor) -> torch.Tensor:
    """vLLM gelu_tanh_and_mul semantics."""
    m, twice_intermediate = gate_up.size()
    intermediate = twice_intermediate // 2
    output = torch.empty((m, intermediate), device=gate_up.device, dtype=gate_up.dtype)
    for tile_m, tile_i in hl.tile([m, intermediate], block_size=[1, None]):
        gate = gate_up[tile_m, tile_i].to(torch.float32)
        up = gate_up[tile_m, tile_i + intermediate]
        output[tile_m, tile_i] = _gelu_tanh(gate).to(up.dtype) * up
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def geglu_projection(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
) -> torch.Tensor:
    """Fuse the gate/up GEMM with its GeGLU epilogue."""
    m, hidden = x.size()
    twice_intermediate, weight_hidden = gate_up_weight.size()
    assert hidden == weight_hidden
    intermediate = twice_intermediate // 2
    output = torch.empty((m, intermediate), device=x.device, dtype=torch.bfloat16)
    for tile_m, tile_i in hl.tile([m, intermediate], block_size=[1, None]):
        gate_acc = hl.zeros([tile_m, tile_i], dtype=torch.float32)
        up_acc = hl.zeros([tile_m, tile_i], dtype=torch.float32)
        for tile_k in hl.tile(hidden):
            values = x[tile_m, tile_k]
            gate_acc = torch.addmm(
                gate_acc,
                values,
                gate_up_weight[tile_i, tile_k].T,
            )
            up_acc = torch.addmm(
                up_acc,
                values,
                gate_up_weight[tile_i + intermediate, tile_k].T,
            )
        gate = gate_acc.to(torch.bfloat16).to(torch.float32)
        up = up_acc.to(torch.bfloat16)
        output[tile_m, tile_i] = _gelu_tanh(gate).to(up.dtype) * up
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def post_ff_residual(
    down: torch.Tensor,
    residual: torch.Tensor,
    post_ff_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse post-FF RMSNorm with the second residual add."""
    m, hidden = down.size()
    hl.specialize(hidden)
    output = torch.empty_like(down)
    for tile_m in hl.tile(m, block_size=1):
        values = down[tile_m, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(values * values, dim=-1) + eps)
        normalized = (values * inv_rms[:, None]).to(down.dtype)
        output[tile_m, :] = normalized * post_ff_weight[None, :] + residual[tile_m, :]
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def ple_gate_gelu_mul(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    per_layer_input: torch.Tensor,
) -> torch.Tensor:
    """Fuse the PLE gate projection, GELU, and per-layer-input multiply."""
    m, hidden_size = hidden.size()
    ple, weight_hidden = gate_weight.size()
    assert hidden_size == weight_hidden
    output = torch.empty((m, ple), device=hidden.device, dtype=torch.bfloat16)
    for tile_m, tile_n in hl.tile([m, ple], block_size=[1, None]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(hidden_size):
            acc = torch.addmm(
                acc,
                hidden[tile_m, tile_k],
                gate_weight[tile_n, tile_k].T,
            )
        gate = acc.to(torch.bfloat16).to(torch.float32)
        output[tile_m, tile_n] = (
            _gelu_tanh(gate).to(torch.bfloat16) * per_layer_input[tile_m, tile_n]
        )
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def final_ple_norm_residual_scale(
    ple_projection: torch.Tensor,
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse PLE RMSNorm, residual addition, and Gemma 4's layer scalar."""
    m, hidden_size = ple_projection.size()
    hl.specialize(hidden_size)
    output = torch.empty_like(hidden)
    for tile_m in hl.tile(m, block_size=1):
        values = ple_projection[tile_m, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(values * values, dim=-1) + eps)
        normalized = (values * inv_rms[:, None]).to(ple_projection.dtype)
        scalar = hl.load(layer_scalar, [])
        output[tile_m, :] = (
            normalized * norm_weight[None, :] + hidden[tile_m, :]
        ) * scalar
    return output


def compile_default(kernel, kernel_args):
    bound = kernel.bind(kernel_args)
    config = bound.config_spec.default_config()
    return config, bound.compile_config(config)


def compile_config(kernel, kernel_args, config_dict):
    bound = kernel.bind(kernel_args)
    config = helion.Config.from_dict(config_dict)
    bound.config_spec.normalize(config.config)
    return config, bound.compile_config(config)


def tune_kernel(name, kernel, kernel_args, configs, config_path):
    print(f"autotune_start {name}", flush=True)
    started = time.perf_counter()
    bound = kernel.bind(kernel_args)
    config = bound.autotune(kernel_args, force=True)
    elapsed = time.perf_counter() - started
    configs[name] = dict(config)
    if config_path is not None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(configs, indent=2, sort_keys=True) + "\n")
    print(
        "autotune_result",
        json.dumps(
            {"name": name, "seconds": elapsed, "config": dict(config)},
            sort_keys=True,
        ),
        flush=True,
    )
    return config, bound.compile_config(config)


def _assert_close(name, actual, expected, *, atol=8e-2, rtol=4e-2):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    max_abs = float((actual.float() - expected.float()).abs().max().item())
    print(f"correctness {name} max_abs={max_abs:.6f}", flush=True)


def build_layer(args, tensors, shape, geometry, configs, config_path):
    tune_set = set(args.tune or [])
    tune_all = "all" in tune_set
    selected_configs = {}

    def build(name, kernel, kernel_args):
        if tune_all or name in tune_set:
            config, compiled = tune_kernel(
                name, kernel, kernel_args, configs, config_path
            )
        elif name in configs:
            config, compiled = compile_config(kernel, kernel_args, configs[name])
        else:
            config, compiled = compile_default(kernel, kernel_args)
        selected_configs[name] = dict(config)
        return compiled

    q_width = shape.q_heads * geometry.head_dim
    kv_width = shape.kv_heads * geometry.head_dim
    qkv_width = q_width + 2 * kv_width
    splits = args.full_splits if geometry.layer_type == "full" else args.sliding_splits

    input_norm_args = (
        tensors["hidden_states"],
        tensors["input_norm_weight"],
        shape.eps,
    )
    input_norm_kernel = build("rms_hidden", rms_norm, input_norm_args)
    input_norm = input_norm_kernel(*input_norm_args)

    qkv_mm_args = (input_norm, tensors["qkv_weight"])
    qkv_mm_name = f"qkv_mm_hd{geometry.head_dim}"
    qkv_mm_kernel = build(qkv_mm_name, bf16_mm, qkv_mm_args)
    qkv = qkv_mm_kernel(*qkv_mm_args)

    fused_qkv_args = (
        tensors["hidden_states"],
        tensors["input_norm_weight"],
        tensors["qkv_weight"],
        shape.eps,
    )
    fused_qkv_name = f"rms_qkv_mm_hd{geometry.head_dim}"
    fused_qkv_kernel = build(fused_qkv_name, rms_qkv_mm, fused_qkv_args)
    fused_qkv = fused_qkv_kernel(*fused_qkv_args)
    _assert_close("rms_qkv_mm", fused_qkv, qkv, atol=1.5e-1, rtol=5e-2)

    shared_q_kernel = None
    shared_q_norm_kernel = None
    shared_q_args = None
    shared_q_norm_args = None
    if geometry.kv_shared:
        shared_q_args = (
            input_norm,
            tensors["qkv_weight"][:q_width],
        )
        shared_q_kernel = build(f"q_mm_hd{geometry.head_dim}", bf16_mm, shared_q_args)
        optimized_q = shared_q_kernel(*shared_q_args)
        shared_q_norm_args = (
            optimized_q,
            tensors["q_norm_weight"],
            tensors["cos_sin"],
            tensors["position"],
            shape.q_heads,
            geometry.head_dim,
            shape.eps,
        )
        shared_q_norm_kernel = build(
            f"q_norm_rope_hd{geometry.head_dim}",
            q_norm_rope,
            shared_q_norm_args,
        )
        shared_q_norm_kernel(*shared_q_norm_args)

    norm_args = (
        fused_qkv,
        tensors["q_norm_weight"],
        tensors["k_norm_weight"],
        tensors["cos_sin"],
        tensors["position"],
        shape.q_heads,
        shape.kv_heads,
        geometry.head_dim,
        shape.eps,
        not geometry.kv_shared,
    )
    norm_name = (
        f"qkv_norm_rope_hd{geometry.head_dim}_"
        f"{'q_only' if geometry.kv_shared else 'qkv'}"
    )
    norm_kernel = build(norm_name, qkv_norm_rope, norm_args)
    norm_kernel(*norm_args)
    query = fused_qkv[:, :q_width].view(1, shape.q_heads, geometry.head_dim)
    key = fused_qkv[:, q_width : q_width + kv_width].view(
        1, shape.kv_heads, geometry.head_dim
    )
    value = fused_qkv[:, q_width + kv_width : qkv_width].view(
        1, shape.kv_heads, geometry.head_dim
    )
    if geometry.kv_shared:
        _assert_close(
            "shared_q_projection_norm_rope",
            optimized_q.view_as(query),
            query,
            atol=1.5e-1,
            rtol=5e-2,
        )

    cache_args = (
        key,
        value,
        tensors["kv_cache"],
        tensors["slot_mapping"],
        shape.block_size,
    )
    cache_kernel = None
    fused_norm_cache_kernel = None
    fused_norm_cache_args = None
    if not geometry.kv_shared:
        cache_name = f"kv_cache_hd{geometry.head_dim}"
        cache_kernel = build(cache_name, reshape_and_cache, cache_args)
        cache_kernel(*cache_args)
        fused_norm_cache_args = (
            qkv,
            tensors["q_norm_weight"],
            tensors["k_norm_weight"],
            tensors["cos_sin"],
            tensors["position"],
            tensors["kv_cache"],
            tensors["slot_mapping"],
            shape.q_heads,
            shape.kv_heads,
            geometry.head_dim,
            shape.block_size,
            shape.eps,
        )
        fused_norm_cache_kernel = build(
            f"qkv_norm_rope_cache_hd{geometry.head_dim}",
            qkv_norm_rope_cache,
            fused_norm_cache_args,
        )
        slot = int(tensors["slot_mapping"][0].item())
        cache_block = slot // shape.block_size
        cache_offset = slot % shape.block_size
        tensors["kv_cache"][cache_block, cache_offset].fill_(float("nan"))
        fused_norm_cache_kernel(*fused_norm_cache_args)
        _assert_close(
            "qkv_norm_rope_cache",
            qkv,
            fused_qkv,
            atol=5e-2,
            rtol=5e-2,
        )
        expected_cache_slot = torch.cat((key[0], value[0]), dim=-1)
        _assert_close(
            "qkv_norm_rope_cache_slot",
            tensors["kv_cache"][cache_block, cache_offset],
            expected_cache_slot,
            atol=5e-2,
            rtol=5e-2,
        )

    attention_args = (
        query,
        tensors["kv_cache"],
        tensors["block_table"],
        shape.context,
        geometry.attention_context,
        shape.block_size,
        shape.q_heads // shape.kv_heads,
        splits,
    )
    attention_name = f"attention_{geometry.layer_type}_hd{geometry.head_dim}_s{splits}"
    attention_kernel = build(attention_name, paged_attention_split, attention_args)
    partial_out, partial_lse = attention_kernel(*attention_args)
    merge_args = (partial_out, partial_lse)
    merge_name = f"attention_merge_hd{geometry.head_dim}_s{splits}"
    merge_kernel = build(merge_name, merge_attention, merge_args)
    attention = merge_kernel(*merge_args)
    reference_attention = paged_attention_reference(
        query,
        tensors["kv_cache"],
        tensors["block_table"],
        shape.context,
        geometry.attention_context,
        shape.block_size,
        shape.q_heads // shape.kv_heads,
    )
    _assert_close(
        "paged_attention",
        attention,
        reference_attention,
        atol=8e-2,
        rtol=4e-2,
    )
    direct_attention_kernel = None
    direct_attention_args = None
    if geometry.layer_type == "sliding":
        direct_attention_args = attention_args[:-1]
        direct_attention_kernel = build(
            f"attention_{geometry.layer_type}_hd{geometry.head_dim}_direct",
            paged_attention,
            direct_attention_args,
        )
        direct_attention = direct_attention_kernel(*direct_attention_args)
        _assert_close(
            "direct_attention",
            direct_attention,
            attention,
            atol=1.5e-1,
            rtol=5e-2,
        )

    o_args = (attention.view(1, q_width), tensors["o_weight"])
    o_name = f"o_mm_hd{geometry.head_dim}"
    o_kernel = build(o_name, bf16_mm, o_args)
    attention_out = o_kernel(*o_args)

    post_attention_args = (
        attention_out,
        tensors["hidden_states"],
        tensors["post_attention_norm_weight"],
        tensors["pre_ff_norm_weight"],
        shape.eps,
    )
    post_attention_kernel = build(
        "post_attention_residual_pre_ff_norm",
        post_attention_residual_pre_ff_norm,
        post_attention_args,
    )
    residual, ff_input = post_attention_kernel(*post_attention_args)

    gate_up_args = (ff_input, tensors["gate_up_weight"])
    gate_up_kernel = build("gate_up_mm", bf16_mm, gate_up_args)
    gate_up = gate_up_kernel(*gate_up_args)
    geglu_args = (gate_up,)
    geglu_kernel = build("geglu", geglu, geglu_args)
    matched_activation = geglu_kernel(*geglu_args)

    fused_geglu_args = (ff_input, tensors["gate_up_weight"])
    fused_geglu_kernel = build("geglu_projection", geglu_projection, fused_geglu_args)
    activation = fused_geglu_kernel(*fused_geglu_args)
    _assert_close(
        "geglu_projection", activation, matched_activation, atol=2e-1, rtol=6e-2
    )

    down_args = (activation, tensors["down_weight"])
    down_kernel = build("down_mm", bf16_mm, down_args)
    down = down_kernel(*down_args)

    post_ff_args = (
        down,
        residual,
        tensors["post_ff_norm_weight"],
        shape.eps,
    )
    post_ff_kernel = build("post_ff_residual", post_ff_residual, post_ff_args)
    hidden = post_ff_kernel(*post_ff_args)

    ple_args = (
        hidden,
        tensors["ple_gate_weight"],
        tensors["per_layer_input"],
    )
    ple_kernel = build("ple_gate_gelu_mul", ple_gate_gelu_mul, ple_args)
    ple_input = ple_kernel(*ple_args)

    ple_projection_args = (ple_input, tensors["ple_proj_weight"])
    ple_projection_kernel = build("ple_projection_mm", bf16_mm, ple_projection_args)
    ple_projection = ple_projection_kernel(*ple_projection_args)

    final_args = (
        ple_projection,
        hidden,
        tensors["post_ple_norm_weight"],
        tensors["layer_scalar"],
        shape.eps,
    )
    final_kernel = build(
        "final_ple_norm_residual_scale", final_ple_norm_residual_scale, final_args
    )
    final_kernel(*final_args)
    torch.cuda.synchronize()

    def launch_matched():
        local_input_norm = input_norm_kernel(*input_norm_args)
        local_qkv = qkv_mm_kernel(local_input_norm, tensors["qkv_weight"])
        norm_kernel(local_qkv, *norm_args[1:])
        local_query = local_qkv[:, :q_width].view(1, shape.q_heads, geometry.head_dim)
        local_key = local_qkv[:, q_width : q_width + kv_width].view(
            1, shape.kv_heads, geometry.head_dim
        )
        local_value = local_qkv[:, q_width + kv_width : qkv_width].view(
            1, shape.kv_heads, geometry.head_dim
        )
        if cache_kernel is not None:
            cache_kernel(
                local_key,
                local_value,
                tensors["kv_cache"],
                tensors["slot_mapping"],
                shape.block_size,
            )
        local_partials, local_lse = attention_kernel(local_query, *attention_args[1:])
        local_attention = merge_kernel(local_partials, local_lse)
        local_attention_out = o_kernel(
            local_attention.view(1, q_width), tensors["o_weight"]
        )
        local_residual, local_ff_input = post_attention_kernel(
            local_attention_out, *post_attention_args[1:]
        )
        local_gate_up = gate_up_kernel(local_ff_input, tensors["gate_up_weight"])
        local_activation = geglu_kernel(local_gate_up)
        local_down = down_kernel(local_activation, tensors["down_weight"])
        local_hidden = post_ff_kernel(
            local_down,
            local_residual,
            tensors["post_ff_norm_weight"],
            shape.eps,
        )
        local_ple = ple_kernel(
            local_hidden,
            tensors["ple_gate_weight"],
            tensors["per_layer_input"],
        )
        local_ple_projection = ple_projection_kernel(
            local_ple, tensors["ple_proj_weight"]
        )
        return final_kernel(
            local_ple_projection,
            local_hidden,
            tensors["post_ple_norm_weight"],
            tensors["layer_scalar"],
            shape.eps,
        )

    def launch_optimized():
        if geometry.kv_shared:
            assert shared_q_kernel is not None
            assert shared_q_norm_kernel is not None
            assert shared_q_args is not None
            local_input_norm = input_norm_kernel(*input_norm_args)
            local_q = shared_q_kernel(local_input_norm, tensors["qkv_weight"][:q_width])
            shared_q_norm_kernel(local_q, *shared_q_norm_args[1:])
            local_query = local_q.view(1, shape.q_heads, geometry.head_dim)
        else:
            local_input_norm = input_norm_kernel(*input_norm_args)
            local_qkv = qkv_mm_kernel(local_input_norm, tensors["qkv_weight"])
            assert fused_norm_cache_kernel is not None
            assert fused_norm_cache_args is not None
            fused_norm_cache_kernel(local_qkv, *fused_norm_cache_args[1:])
            local_query = local_qkv[:, :q_width].view(
                1, shape.q_heads, geometry.head_dim
            )
        local_partials, local_lse = attention_kernel(local_query, *attention_args[1:])
        local_attention = merge_kernel(local_partials, local_lse)
        local_attention_out = o_kernel(
            local_attention.view(1, q_width), tensors["o_weight"]
        )
        local_residual, local_ff_input = post_attention_kernel(
            local_attention_out, *post_attention_args[1:]
        )
        local_gate_up = gate_up_kernel(local_ff_input, tensors["gate_up_weight"])
        local_activation = geglu_kernel(local_gate_up)
        local_down = down_kernel(local_activation, tensors["down_weight"])
        local_hidden = post_ff_kernel(
            local_down,
            local_residual,
            tensors["post_ff_norm_weight"],
            shape.eps,
        )
        local_ple = ple_kernel(
            local_hidden,
            tensors["ple_gate_weight"],
            tensors["per_layer_input"],
        )
        local_ple_projection = ple_projection_kernel(
            local_ple, tensors["ple_proj_weight"]
        )
        return final_kernel(
            local_ple_projection,
            local_hidden,
            tensors["post_ple_norm_weight"],
            tensors["layer_scalar"],
            shape.eps,
        )

    return {
        "launch_matched": launch_matched,
        "launch_optimized": launch_optimized,
        "configs": selected_configs,
        "stage_outputs": {
            "query": query,
            "key": key,
            "value": value,
            "attention": attention,
        },
        "stage_calls": {
            "rms_hidden": lambda: input_norm_kernel(*input_norm_args),
            qkv_mm_name: lambda: qkv_mm_kernel(*qkv_mm_args),
            fused_qkv_name: lambda: fused_qkv_kernel(*fused_qkv_args),
            norm_name: lambda: norm_kernel(*norm_args),
            **(
                {f"kv_cache_hd{geometry.head_dim}": lambda: cache_kernel(*cache_args)}
                if cache_kernel is not None
                else {}
            ),
            **(
                {
                    f"qkv_norm_rope_cache_hd{geometry.head_dim}": (
                        lambda: fused_norm_cache_kernel(*fused_norm_cache_args)
                    )
                }
                if fused_norm_cache_kernel is not None
                else {}
            ),
            attention_name: lambda: attention_kernel(*attention_args),
            merge_name: lambda: merge_kernel(*merge_args),
            **(
                {
                    f"attention_{geometry.layer_type}_hd{geometry.head_dim}_direct": (
                        lambda: direct_attention_kernel(*direct_attention_args)
                    )
                }
                if direct_attention_kernel is not None
                else {}
            ),
            **(
                {
                    f"q_mm_hd{geometry.head_dim}": lambda: shared_q_kernel(
                        *shared_q_args
                    ),
                    f"q_norm_rope_hd{geometry.head_dim}": lambda: shared_q_norm_kernel(
                        *shared_q_norm_args
                    ),
                }
                if shared_q_kernel is not None
                else {}
            ),
            o_name: lambda: o_kernel(*o_args),
            "post_attention_residual_pre_ff_norm": lambda: post_attention_kernel(
                *post_attention_args
            ),
            "gate_up_mm": lambda: gate_up_kernel(*gate_up_args),
            "geglu": lambda: geglu_kernel(*geglu_args),
            "geglu_projection": lambda: fused_geglu_kernel(*fused_geglu_args),
            "down_mm": lambda: down_kernel(*down_args),
            "post_ff_residual": lambda: post_ff_kernel(*post_ff_args),
            "ple_gate_gelu_mul": lambda: ple_kernel(*ple_args),
            "ple_projection_mm": lambda: ple_projection_kernel(*ple_projection_args),
            "final_ple_norm_residual_scale": lambda: final_kernel(*final_args),
        },
    }


def run_layer(args, layer_idx, configs, config_path):
    configured_shape = Gemma4E4BShape(context=args.context, block_size=args.block_size)
    geometry = configured_shape.layer_geometry(layer_idx)
    effective_block_size = args.block_size
    if not args.disable_hybrid_page_promotion:
        effective_block_size *= 512 // geometry.head_dim
    shape = Gemma4E4BShape(context=args.context, block_size=effective_block_size)
    geometry = shape.layer_geometry(layer_idx)
    name = variant_name(geometry)
    tensors = allocate_layer(shape, geometry, args.seed)
    reference = layer_reference(tensors, shape, geometry)
    built = build_layer(args, tensors, shape, geometry, configs, config_path)
    stage_outputs = built["stage_outputs"]
    _assert_close(
        "query", stage_outputs["query"], reference["query"], atol=8e-2, rtol=4e-2
    )
    if not geometry.kv_shared:
        _assert_close(
            "key", stage_outputs["key"], reference["key"], atol=8e-2, rtol=4e-2
        )
        _assert_close(
            "value", stage_outputs["value"], reference["value"], atol=8e-2, rtol=4e-2
        )
    _assert_close(
        "attention",
        stage_outputs["attention"],
        reference["attention"],
        atol=1.5e-1,
        rtol=6e-2,
    )

    result = {
        "layer_idx": layer_idx,
        "variant": name,
        "head_dim": geometry.head_dim,
        "rotary_dim": geometry.rotary_dim,
        "kv_shared": geometry.kv_shared,
        "context": shape.context,
        "configured_block_size": args.block_size,
        "effective_kernel_block_size": shape.block_size,
        "attention_context": geometry.attention_context,
        "benchmark_mode": benchmark_cache_mode(),
        "splits": args.full_splits
        if geometry.layer_type == "full"
        else args.sliding_splits,
    }
    if args.include_configs:
        result["configs"] = built["configs"]
    matched_eager = built["launch_matched"]()
    torch.cuda.synchronize()
    _assert_close(
        "matched_eager_output",
        matched_eager,
        reference["output"],
        atol=2e-1,
        rtol=8e-2,
    )
    optimized_eager = built["launch_optimized"]()
    torch.cuda.synchronize()
    _assert_close(
        "optimized_eager_output",
        optimized_eager,
        reference["output"],
        atol=2e-1,
        rtol=8e-2,
    )
    if args.smoke and not args.benchmark:
        result["status"] = "smoke_ok"
        return result

    matched_graph, matched_output = capture(built["launch_matched"])
    matched_graph.replay()
    torch.cuda.synchronize()
    _assert_close(
        "matched_graph_output",
        matched_output,
        reference["output"],
        atol=2e-1,
        rtol=8e-2,
    )
    optimized_graph, optimized_output = capture(built["launch_optimized"])
    optimized_graph.replay()
    torch.cuda.synchronize()
    _assert_close(
        "optimized_graph_output",
        optimized_output,
        reference["output"],
        atol=2e-1,
        rtol=8e-2,
    )
    benchmark_pids = visible_gpu_pids()
    timings = benchmark_interleaved(
        {
            f"helion_{name}_matched": matched_graph.replay,
            f"helion_{name}_optimized": optimized_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
    if args.benchmark_stages:
        stage_graphs = {
            stage: capture(call)[0].replay
            for stage, call in built["stage_calls"].items()
        }
        timings.update(
            benchmark_interleaved(stage_graphs, args.repeats, args.batch_replays)
        )
    if visible_gpu_pids() != benchmark_pids:
        raise RuntimeError("GPU process set changed during benchmark")
    result["timings"] = timings
    return result


def run(args):
    require_idle_visible_gpu()
    config_path = Path(args.config_path) if args.config_path else None
    configs = (
        json.loads(config_path.read_text())
        if config_path is not None and config_path.exists()
        else {}
    )
    layers = E4B_REPRESENTATIVE_LAYERS if args.all_variants else (args.layer,)
    results = [run_layer(args, layer_idx, configs, config_path) for layer_idx in layers]
    if args.all_variants and all("timings" in result for result in results):
        weighted = {}
        for path in ("matched", "optimized"):
            total = 0.0
            for result in results:
                key = f"helion_{result['variant']}_{path}"
                total += (
                    result["timings"][key]["median_us"]
                    * E4B_LAYER_COUNTS[result["variant"]]
                )
            weighted[f"helion_e4b_42_layer_{path}_us"] = total
        summary = {"layers": results, "weighted_model_layer_sum": weighted}
    else:
        summary = {"layers": results}
    print(
        "RESULT_JSON",
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "helion_module": helion.__file__,
                **summary,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--disable-hybrid-page-promotion", action="store_true")
    parser.add_argument("--sliding-splits", type=int, default=16)
    parser.add_argument("--full-splits", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch-replays", type=int, default=20)
    parser.add_argument(
        "--config-path",
        default=str(Path(__file__).with_name("gemma4_e4b_b200_configs.json")),
    )
    parser.add_argument("--tune", nargs="*", default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-stages", action="store_true")
    parser.add_argument("--include-configs", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
