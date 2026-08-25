# ruff: noqa: A001, A002, ANN001, ANN201, ANN202, RET504
"""Fully autotuned separate-Helion baseline for one Qwen3-8B decode layer.

The graph mirrors the steady-state vLLM FP8 layer boundaries at TP=1, M=1:

    fused residual + RMSNorm + block-FP8 quant
      -> block-FP8 QKV projection
      -> fused Q/K RMSNorm + RoPE
      -> KV-cache update
      -> paged GQA decode attention
      -> block-FP8 quant
      -> block-FP8 O projection
      -> fused residual + RMSNorm + block-FP8 quant
      -> block-FP8 W13
      -> fused SiLU*up + block-FP8 quant
      -> block-FP8 W2

The three FFN kernels use the existing tuned Helion configurations from the
persistent FFN study. Every surrounding Helion kernel can be full-effort
autotuned independently and checkpointed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from probes.common import benchmark_interleaved
from probes.common import capture
from probes.common import make_fp8_random
from probes.common import require_idle_visible_gpu
from probes.common import visible_gpu_pids

import helion
import helion.language as hl

FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)

W13_CONFIG = {
    "atomic_indexing": [],
    "block_sizes": [16],
    "indexing": ["pointer"] * 5,
    "l2_groupings": [1],
    "load_eviction_policies": [""] * 4,
    "loop_orders": [[0, 1]],
    "num_stages": 4,
    "num_warps": 1,
    "pid_type": "flat",
    "range_flattens": [None, False],
    "range_multi_buffers": [None, True],
    "range_num_stages": [0, 0],
    "range_unroll_factors": [0, 2],
    "range_warp_specializes": [None, None],
}

ACTIVATION_CONFIG = {
    "atomic_indexing": [],
    "block_sizes": [],
    "indexing": ["pointer"] * 4,
    "l2_groupings": [1],
    "load_eviction_policies": ["", ""],
    "loop_orders": [[0, 1]],
    "num_stages": 1,
    "num_warps": 4,
    "pid_type": "flat",
    "range_flattens": [None],
    "range_multi_buffers": [None],
    "range_num_stages": [0],
    "range_unroll_factors": [0],
    "range_warp_specializes": [None],
}

W2_CONFIG = {
    "atomic_indexing": [],
    "block_sizes": [8],
    "indexing": ["pointer"] * 5,
    "l2_groupings": [1],
    "load_eviction_policies": [""] * 4,
    "loop_orders": [[0, 1]],
    "num_stages": 4,
    "num_warps": 1,
    "pid_type": "flat",
    "range_flattens": [None, True],
    "range_multi_buffers": [None, False],
    "range_num_stages": [0, 4],
    "range_unroll_factors": [0, 4],
    "range_warp_specializes": [None, None],
}

FFN_CONFIGS = {
    "w13": W13_CONFIG,
    "silu_quant": ACTIVATION_CONFIG,
    "w2": W2_CONFIG,
}


def rms_quant_baseline(
    result,
    input,
    weight,
    scale,
    epsilon,
    scale_ub,
    residual,
    group_size,
    is_scale_transposed,
):
    del is_scale_transposed
    num_tokens, hidden_size = input.shape
    x = input.float()
    if residual is not None:
        x = x + residual.float()
        residual.copy_(x.to(residual.dtype))
    rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)
    x_norm = (x * rms).to(input.dtype) * weight
    grouped = x_norm.view(num_tokens, hidden_size // group_size, group_size).float()
    s = grouped.abs().amax(-1)
    if scale_ub is not None:
        s = s.clamp(max=scale_ub)
    s = (s / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    scale.copy_(s)
    result.copy_((grouped / s[:, :, None]).clamp(FP8_MIN, FP8_MAX).view_as(result))


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=rms_quant_baseline,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def rms_norm_per_block_quant(
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
    """Target-shape copy of vLLM's Helion rms_norm_per_block_quant."""
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)
    groups_per_row = scale.shape[1]
    hl.specialize(groups_per_row)
    assert group_size == 128
    assert result.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32

    for tile_m in hl.tile(num_tokens, block_size=1):
        rms = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            rms = rms + x_blk.pow(2).sum(dim=-1)
        rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)

        m_idx = tile_m.begin + hl.arange(tile_m.block_size)
        m_blk = m_idx[:, None, None]
        for tile_gn, tile_n in hl.tile(
            [groups_per_row, group_size], block_size=[None, group_size]
        ):
            gn_idx = tile_gn.index
            n_idx = gn_idx[:, None] * group_size + tile_n.index[None, :]
            n_blk = n_idx[None, :, :]
            mask = (gn_idx < groups_per_row)[None, :, None]
            x_blk = hl.load(input, [m_blk, n_blk], extra_mask=mask).to(torch.float32)
            if residual is not None:
                x_blk = x_blk + hl.load(residual, [m_blk, n_blk], extra_mask=mask)
            w_blk = hl.load(weight, [n_blk], extra_mask=mask)
            x_norm = (x_blk * rms[:, None, None]).to(input.dtype) * w_blk
            s = torch.amax(torch.abs(x_norm), dim=-1).to(torch.float32)
            if scale_ub is not None:
                s = s.clamp(max=hl.load(scale_ub, []))
            s = (s / FP8_MAX).clamp(min=FP8_MIN_SCALE)
            scale[tile_m, tile_gn] = s
            y = (x_norm / s[:, :, None]).clamp(FP8_MIN, FP8_MAX).to(result.dtype)
            hl.store(result, [m_blk, n_blk], y, extra_mask=mask)
            if residual is not None:
                hl.store(
                    residual, [m_blk, n_blk], x_blk.to(residual.dtype), extra_mask=mask
                )


def qk_norm_rope_baseline(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    eps,
    q_weight,
    k_weight,
    cos_sin_cache,
    is_neox,
    position_ids,
    forced_token_heads_per_warp=-1,
):
    del num_heads_v, forced_token_heads_per_warp
    q_size = num_heads_q * head_dim
    kv_size = num_heads_k * head_dim
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
    qh = q.view(-1, num_heads_q, head_dim)
    kh = k.view(-1, num_heads_k, head_dim)
    qh = (
        qh.float() * torch.rsqrt(qh.float().square().mean(-1, keepdim=True) + eps)
    ).to(qkv.dtype) * q_weight
    kh = (
        kh.float() * torch.rsqrt(kh.float().square().mean(-1, keepdim=True) + eps)
    ).to(qkv.dtype) * k_weight
    cache = cos_sin_cache[position_ids]
    embed = cache.shape[-1] // 2
    cos, sin = cache[..., :embed], cache[..., embed:]
    if is_neox:

        def rotate(x):
            x1, x2 = x[..., :embed], x[..., embed : 2 * embed]
            return torch.cat(
                (
                    x1 * cos[:, None] - x2 * sin[:, None],
                    x2 * cos[:, None] + x1 * sin[:, None],
                ),
                dim=-1,
            )
    else:

        def rotate(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = x1 * cos[:, None] - x2 * sin[:, None]
            out[..., 1::2] = x2 * cos[:, None] + x1 * sin[:, None]
            return out

    qkv[:, :q_size].copy_(rotate(qh).reshape_as(q))
    qkv[:, q_size : q_size + kv_size].copy_(rotate(kh).reshape_as(k))


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=qk_norm_rope_baseline,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=5e-2,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def fused_qk_norm_rope(
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
    """Exact Helion body used by vLLM's fused_qk_norm_rope."""
    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v
    hl.specialize(qkv.shape[1])
    _, rotary_dim = cos_sin_cache.shape
    hl.specialize(rotary_dim)
    embed_dim = rotary_dim // 2
    hl.specialize(num_heads_q)
    hl.specialize(num_heads_k)
    hl.specialize(num_heads_v)
    hl.specialize(head_dim)
    qk_heads = num_heads_q + num_heads_k
    qkv = qkv.view(num_tokens, total_heads, head_dim)

    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, qk_heads, head_dim], block_size=[1, None, head_dim]
    ):
        x = qkv[tile_m, tile_gn, tile_n].to(torch.float32)
        rms = torch.rsqrt(x.pow(2).sum(-1) * (1.0 / head_dim) + eps)
        use_q = (tile_gn.index < num_heads_q)[None, :, None]
        w = torch.where(
            use_q,
            q_weight[None, None, tile_n],
            k_weight[None, None, tile_n],
        )
        x = (x * rms[:, :, None]).to(qkv.dtype) * w
        qkv[tile_m, tile_gn, tile_n] = x
        pos = position_ids[tile_m]
        cos = cos_sin_cache[pos, hl.arange(embed_dim)]
        sin = cos_sin_cache[pos, hl.arange(embed_dim) + embed_dim]
        if is_neox:
            x1_offset = hl.arange(embed_dim)
            x2_offset = x1_offset + embed_dim
        else:
            x1_offset = hl.arange(embed_dim) * 2
            x2_offset = x1_offset + 1
        x1 = qkv[tile_m, tile_gn, x1_offset]
        x2 = qkv[tile_m, tile_gn, x2_offset]
        qkv[tile_m, tile_gn, x1_offset] = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        qkv[tile_m, tile_gn, x2_offset] = x2 * cos[:, None, :] + x1 * sin[:, None, :]


def group_quant_baseline(
    input,
    output_q,
    output_s,
    group_size,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    dummy_is_scale_transposed=False,
    dummy_is_tma_aligned=False,
):
    del dummy_is_scale_transposed, dummy_is_tma_aligned
    grouped = input.view(input.shape[0], -1, group_size).float()
    s = grouped.abs().amax(-1).clamp(min=eps) / fp8_max
    if scale_ue8m0:
        s = torch.exp2(torch.ceil(torch.log2(s)))
    output_s.copy_(s)
    output_q.copy_((grouped / s[:, :, None]).clamp(fp8_min, fp8_max).view_as(output_q))


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=group_quant_baseline,
    # FP8 conversion can land on an adjacent representable value even when
    # the FP32 scale agrees. Keep this tolerance local to the FP8 payload.
    autotune_baseline_atol=1.0,
    autotune_baseline_rtol=2e-2,
)
def per_token_group_fp8_quant(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    """Exact Helion body used by vLLM's per_token_group_fp8_quant."""
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)
    groups_per_row = output_s.shape[1]
    hl.specialize(groups_per_row)
    input = input.view(num_tokens, groups_per_row, group_size)
    output_q = output_q.view(num_tokens, groups_per_row, group_size)
    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    ):
        x = input[tile_m, tile_gn, tile_n]
        s = torch.amax(torch.abs(x), dim=-1).clamp(min=eps) / fp8_max
        if scale_ue8m0:
            s = torch.exp2(torch.ceil(torch.log2(s)))
        output_s[tile_m, tile_gn] = s
        output_q[tile_m, tile_gn, tile_n] = (
            (x / s[:, :, None]).clamp(fp8_min, fp8_max).to(output_q.dtype)
        )


@helion.kernel(static_shapes=True, autotune_effort="full")
def block_fp8_mm(
    activation_q: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    m, k = activation_q.size()
    n, weight_k = weight_q.size()
    assert weight_k == k
    assert group_size == 128
    hl.specialize(group_size)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=activation_q.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=group_size):
            partial = hl.dot(
                activation_q[tile_m, tile_k],
                weight_q[tile_n, tile_k].T,
            ).to(torch.float32)
            a_scale = activation_scale[tile_m, tile_k.id].to(torch.float32)
            w_scale = weight_scale[tile_n.index // group_size, tile_k.id].to(
                torch.float32
            )
            acc = acc + partial * a_scale[:, None] * w_scale[None, :]
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(static_shapes=True, autotune_effort="full")
def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Helion equivalent of vLLM's separate FlashAttention cache-update op."""
    num_tokens, num_kv_heads, head_dim = key.shape
    hl.specialize(num_kv_heads)
    hl.specialize(head_dim)
    hl.specialize(block_size)
    for tile_t, tile_h, tile_d in hl.tile(
        [num_tokens, num_kv_heads, head_dim], block_size=[1, None, None]
    ):
        t = tile_t.index
        h = tile_h.index
        d = tile_d.index
        slot = slot_mapping[t]
        block = (slot // block_size)[:, None, None]
        offset = (slot % block_size)[:, None, None]
        h_idx = h[None, :, None]
        d_idx = d[None, None, :]
        key_value = key[tile_t, tile_h, tile_d]
        value_value = value[tile_t, tile_h, tile_d]
        hl.store(kv_cache, [block, offset, h_idx, d_idx], key_value)
        hl.store(
            kv_cache,
            [block, offset, h_idx, d_idx + head_dim],
            value_value,
        )


def paged_gqa_attention_baseline(
    query,
    kv_cache,
    block_table,
    context,
    block_size,
    q_per_kv,
):
    head_dim = query.shape[-1]
    outputs = []
    for token in range(query.shape[0]):
        blocks = block_table[token, : math.ceil(context / block_size)].long()
        logical = kv_cache[blocks].reshape(-1, kv_cache.shape[2], kv_cache.shape[3])[
            :context
        ]
        k = logical[..., :head_dim].permute(1, 0, 2).repeat_interleave(q_per_kv, dim=0)
        v = logical[..., head_dim:].permute(1, 0, 2).repeat_interleave(q_per_kv, dim=0)
        q = query[token].unsqueeze(1)
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(q, k, v).squeeze(1)
        )
    return torch.cat(outputs, dim=0).unsqueeze(0)


def paged_gqa_attention_split_baseline(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    block_size: int,
    q_per_kv: int,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = query.shape[-1]
    num_kv_heads = kv_cache.shape[2]
    num_tokens = query.shape[0]
    split_context = context // splits
    q = query.reshape(num_tokens * num_kv_heads, q_per_kv, head_dim).float()
    partial_out = torch.empty(
        (splits, num_tokens * num_kv_heads, q_per_kv, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (splits, num_tokens * num_kv_heads, q_per_kv),
        device=query.device,
        dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(head_dim)
    for token in range(num_tokens):
        blocks = block_table[token, : math.ceil(context / block_size)].long()
        logical = kv_cache[blocks].reshape(-1, kv_cache.shape[2], kv_cache.shape[3])[
            :context
        ]
        group_begin = token * num_kv_heads
        group_end = group_begin + num_kv_heads
        for split in range(splits):
            begin = split * split_context
            end = begin + split_context
            k = logical[begin:end, :, :head_dim].permute(1, 0, 2).float()
            v = logical[begin:end, :, head_dim:].permute(1, 0, 2).float()
            scores = torch.einsum("gqd,gnd->gqn", q[group_begin:group_end], k) * scale
            partial_lse[split, group_begin:group_end] = torch.logsumexp(
                scores, dim=-1
            ) * math.log2(math.e)
            partial_out[split, group_begin:group_end] = torch.einsum(
                "gqn,gnd->gqd", torch.softmax(scores, dim=-1), v
            )
    return partial_out, partial_lse


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=paged_gqa_attention_split_baseline,
    autotune_baseline_atol=8e-2,
    autotune_baseline_rtol=3e-2,
)
def paged_gqa_decode_attention_split(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    block_size: int,
    q_per_kv: int,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-KV partials for paged M=1 GQA decode attention."""
    num_tokens, num_q_heads, head_dim = query.shape
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
    token_kv_heads = num_tokens * num_kv_heads
    partial_out = torch.empty(
        (splits, token_kv_heads, q_per_kv, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (splits, token_kv_heads, q_per_kv),
        device=query.device,
        dtype=torch.float32,
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_split, tile_bg, tile_q in hl.tile(
        [splits, token_kv_heads, q_per_kv], block_size=[1, 1, None]
    ):
        m_i = hl.full([tile_bg, tile_q], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_bg, tile_q], 1.0, dtype=torch.float32)
        acc = hl.zeros([tile_bg, tile_q, head_dim], dtype=torch.float32)
        split_idx = tile_split.begin
        token = tile_bg.index // num_kv_heads
        kv_head = tile_bg.index % num_kv_heads
        query_head = kv_head[:, None] * q_per_kv + tile_q.index[None, :]
        q_blk = query[token[:, None], query_head, :]
        q_blk = (q_blk * qk_scale).to(query.dtype)
        for tile_local_n in hl.tile(split_context):
            n = split_idx * split_context + tile_local_n.index
            physical_block = block_table[token[:, None], (n // block_size)[None, :]]
            block_offset = n % block_size
            d = hl.arange(head_dim)
            k = hl.load(
                kv_cache,
                [
                    physical_block[:, :, None],
                    block_offset[None, :, None],
                    kv_head[:, None, None],
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
                    physical_block[:, :, None],
                    block_offset[None, :, None],
                    kv_head[:, None, None],
                    (d + head_dim)[None, None, :],
                ],
            )
            acc = torch.baddbmm(acc, p.to(v.dtype), v)
            m_i = m_ij
        partial_out[tile_split, tile_bg, tile_q, :] = (acc / l_i[:, :, None])[
            None, :, :, :
        ]
        partial_lse[tile_split, tile_bg, tile_q] = (m_i + torch.log2(l_i))[None, :, :]
    return partial_out, partial_lse


def merge_attention_baseline(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    max_lse = partial_lse.amax(dim=0)
    weights = torch.exp2(partial_lse - max_lse[None])
    output = (partial_out * weights[..., None]).sum(dim=0)
    output = output / weights.sum(dim=0)[..., None]
    return output.to(torch.bfloat16).view(1, -1, partial_out.shape[-1])


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=merge_attention_baseline,
    autotune_baseline_atol=2e-2,
    autotune_baseline_rtol=2e-2,
)
def merge_attention_splits(
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


@helion.kernel(static_shapes=True, autotune_effort="full")
def silu_and_mul_per_block_quant(
    gate_up: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, twice_intermediate = gate_up.size()
    intermediate = twice_intermediate // 2
    hl.specialize(group_size)
    groups = intermediate // group_size
    activation_q = torch.empty(
        (m, intermediate), dtype=torch.float8_e4m3fn, device=gate_up.device
    )
    activation_scale = torch.empty(
        (m, groups), dtype=torch.float32, device=gate_up.device
    )
    for tile_m, tile_i in hl.tile([m, intermediate], block_size=[1, group_size]):
        gate = gate_up[tile_m, tile_i].to(torch.float32)
        up = gate_up[tile_m, tile_i + intermediate].to(torch.float32)
        activated = gate * torch.sigmoid(gate) * up
        scale = (torch.amax(torch.abs(activated), dim=-1) / FP8_MAX).clamp(
            min=FP8_MIN_SCALE
        )
        activation_scale[tile_m, tile_i.id] = scale
        activation_q[tile_m, tile_i] = (
            (activated / scale[:, None]).clamp(FP8_MIN, FP8_MAX).to(activation_q.dtype)
        )
    return activation_q, activation_scale


def compile_config(kernel, kernel_args, config_dict):
    bound = kernel.bind(kernel_args)
    config = helion.Config.from_dict(config_dict)
    bound.config_spec.normalize(config.config)
    return config, bound.compile_config(config)


def compile_default(kernel, kernel_args):
    bound = kernel.bind(kernel_args)
    config = bound.config_spec.default_config()
    return config, bound.compile_config(config)


def tune(name, kernel, kernel_args, configs, config_path):
    print(f"autotune_start {name}", flush=True)
    started = time.perf_counter()
    bound = kernel.bind(kernel_args)
    config = bound.autotune(kernel_args, force=True)
    elapsed = time.perf_counter() - started
    configs[name] = dict(config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(configs, indent=2, sort_keys=True) + "\n")
    print(
        "autotune_result",
        json.dumps(
            {"name": name, "seconds": elapsed, "config": dict(config)}, sort_keys=True
        ),
        flush=True,
    )
    return config, bound.compile_config(config)


def make_cos_sin(max_position, head_dim, theta, device):
    inv = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(
        torch.arange(max_position, device=device, dtype=torch.float32), inv
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)


def allocate(args):
    torch.manual_seed(args.seed)
    device = "cuda"
    hidden_groups = args.hidden // args.group
    intermediate_groups = args.intermediate // args.group
    qkv_width = (args.q_heads + 2 * args.kv_heads) * args.head_dim
    logical_blocks = math.ceil(args.context / args.block_size)
    batch = args.batch
    physical_blocks = math.ceil(batch * logical_blocks * 1.25)
    block_table = (
        torch.randperm(physical_blocks, device=device, dtype=torch.int64)[
            : batch * logical_blocks
        ]
        .to(torch.int32)
        .view(batch, logical_blocks)
    )
    final_logical_block = (args.context - 1) // args.block_size
    final_block_offset = (args.context - 1) % args.block_size
    final_physical_blocks = block_table[:, final_logical_block].to(torch.int64)
    tensors = {
        "hidden_states": torch.randn(
            (batch, args.hidden), device=device, dtype=torch.bfloat16
        ),
        "residual": torch.randn(
            (batch, args.hidden), device=device, dtype=torch.bfloat16
        ),
        "pre_weight": torch.randn((args.hidden,), device=device, dtype=torch.bfloat16)
        * 0.1
        + 1.0,
        "post_weight": torch.randn((args.hidden,), device=device, dtype=torch.bfloat16)
        * 0.1
        + 1.0,
        "q_weight": torch.randn((args.head_dim,), device=device, dtype=torch.bfloat16)
        * 0.1
        + 1.0,
        "k_weight": torch.randn((args.head_dim,), device=device, dtype=torch.bfloat16)
        * 0.1
        + 1.0,
        "position": torch.full(
            (batch,), args.context - 1, device=device, dtype=torch.int64
        ),
        "cos_sin": make_cos_sin(
            max(args.context, 4096), args.head_dim, args.rope_theta, device
        ),
        "pre_q": torch.empty(
            (batch, args.hidden), device=device, dtype=torch.float8_e4m3fn
        ),
        "pre_scale": torch.empty(
            (batch, hidden_groups), device=device, dtype=torch.float32
        ),
        "qkv_weight_q": make_fp8_random((qkv_width, args.hidden)),
        "qkv_weight_scale": torch.rand(
            (qkv_width // args.group, hidden_groups), device=device
        )
        * 0.01
        + 0.01,
        "kv_cache": torch.randn(
            (physical_blocks, args.block_size, args.kv_heads, 2 * args.head_dim),
            device=device,
            dtype=torch.bfloat16,
        ),
        "block_table": block_table,
        "slot_mapping": final_physical_blocks * args.block_size + final_block_offset,
        "attention_q": torch.empty(
            (batch, args.hidden), device=device, dtype=torch.float8_e4m3fn
        ),
        "attention_scale": torch.empty(
            (batch, hidden_groups), device=device, dtype=torch.float32
        ),
        "o_weight_q": make_fp8_random((args.hidden, args.hidden)),
        "o_weight_scale": torch.rand((hidden_groups, hidden_groups), device=device)
        * 0.01
        + 0.01,
        "ffn_q": torch.empty(
            (batch, args.hidden), device=device, dtype=torch.float8_e4m3fn
        ),
        "ffn_scale": torch.empty(
            (batch, hidden_groups), device=device, dtype=torch.float32
        ),
        "w13_q": make_fp8_random((2 * args.intermediate, args.hidden)),
        "w13_scale": torch.rand((2 * intermediate_groups, hidden_groups), device=device)
        * (0.5 / args.hidden**0.5)
        + (0.75 / args.hidden**0.5),
        "w2_q": make_fp8_random((args.hidden, args.intermediate)),
        "w2_scale": torch.rand((hidden_groups, intermediate_groups), device=device)
        * (0.5 / args.intermediate**0.5)
        + (0.75 / args.intermediate**0.5),
    }
    return tensors


def run(args):
    require_idle_visible_gpu()
    if args.group != 128 or args.hidden != 4096 or args.intermediate != 12288:
        raise ValueError("this baseline is pinned to Qwen3-8B block-FP8 geometry")
    tensors = allocate(args)
    qkv_width = (args.q_heads + 2 * args.kv_heads) * args.head_dim
    config_path = Path(args.config_path)
    configs = json.loads(config_path.read_text()) if config_path.exists() else {}
    tune_set = set(args.tune or [])

    def build(name, kernel, kernel_args, known=None):
        if name in tune_set:
            return tune(name, kernel, kernel_args, configs, config_path)
        if known is not None:
            return compile_config(kernel, kernel_args, known)
        if name in configs:
            return compile_config(kernel, kernel_args, configs[name])
        if args.smoke:
            return compile_default(kernel, kernel_args)
        raise RuntimeError(f"missing tuned config for {name}; pass --tune {name}")

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
    rms_config, rms_compiled = build("rms_quant", rms_norm_per_block_quant, rms_args)
    rms_compiled(*rms_args)

    qkv_args = (
        tensors["pre_q"],
        tensors["pre_scale"],
        tensors["qkv_weight_q"],
        tensors["qkv_weight_scale"],
        args.group,
    )
    qkv_config, qkv_compiled = build("qkv_mm", block_fp8_mm, qkv_args)
    qkv = qkv_compiled(*qkv_args)

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
    qk_config, qk_compiled = build("qk_norm_rope", fused_qk_norm_rope, qk_args)
    qk_compiled(*qk_args)
    query = qkv[:, : args.q_heads * args.head_dim].view(
        args.batch, args.q_heads, args.head_dim
    )
    key_begin = args.q_heads * args.head_dim
    key = qkv[:, key_begin : key_begin + args.kv_heads * args.head_dim].view(
        args.batch, args.kv_heads, args.head_dim
    )
    value = qkv[:, key_begin + args.kv_heads * args.head_dim : qkv_width].view(
        args.batch, args.kv_heads, args.head_dim
    )

    cache_args = (
        key,
        value,
        tensors["kv_cache"],
        tensors["slot_mapping"],
        args.block_size,
    )
    cache_config, cache_compiled = build(
        "kv_cache_update", reshape_and_cache_flash, cache_args
    )
    cache_compiled(*cache_args)

    attention_split_args = (
        query,
        tensors["kv_cache"],
        tensors["block_table"],
        args.context,
        args.block_size,
        args.q_heads // args.kv_heads,
        args.attention_splits,
    )
    attention_split_config, attention_split_compiled = build(
        "decode_attention_split",
        paged_gqa_decode_attention_split,
        attention_split_args,
    )
    partial_attention, partial_lse = attention_split_compiled(*attention_split_args)
    attention_merge_args = (partial_attention, partial_lse)
    attention_merge_config, attention_merge_compiled = build(
        "decode_attention_merge",
        merge_attention_splits,
        attention_merge_args,
    )
    attention = attention_merge_compiled(*attention_merge_args)
    attention_flat = attention.view(args.batch, args.hidden)

    attention_quant_args = (
        attention_flat,
        tensors["attention_q"],
        tensors["attention_scale"],
        args.group,
        1e-10,
        FP8_MIN,
        FP8_MAX,
        False,
        False,
        False,
    )
    attention_quant_config, attention_quant_compiled = build(
        "attention_quant", per_token_group_fp8_quant, attention_quant_args
    )
    attention_quant_compiled(*attention_quant_args)

    o_args = (
        tensors["attention_q"],
        tensors["attention_scale"],
        tensors["o_weight_q"],
        tensors["o_weight_scale"],
        args.group,
    )
    o_config, o_compiled = build("o_mm", block_fp8_mm, o_args)
    attention_out = o_compiled(*o_args)

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
    rms_compiled(*post_args)

    w13_args = (
        tensors["ffn_q"],
        tensors["ffn_scale"],
        tensors["w13_q"],
        tensors["w13_scale"],
        args.group,
    )
    w13_config, w13_compiled = build("w13", block_fp8_mm, w13_args, FFN_CONFIGS["w13"])
    gate_up = w13_compiled(*w13_args)
    silu_args = (gate_up, args.group)
    silu_config, silu_compiled = build(
        "silu_quant",
        silu_and_mul_per_block_quant,
        silu_args,
        FFN_CONFIGS["silu_quant"],
    )
    activation_q, activation_scale = silu_compiled(*silu_args)
    w2_args = (
        activation_q,
        activation_scale,
        tensors["w2_q"],
        tensors["w2_scale"],
        args.group,
    )
    w2_config, w2_compiled = build("w2", block_fp8_mm, w2_args, FFN_CONFIGS["w2"])
    layer_out = w2_compiled(*w2_args)
    torch.cuda.synchronize()

    expected_attention = paged_gqa_attention_baseline(
        query,
        tensors["kv_cache"],
        tensors["block_table"],
        args.context,
        args.block_size,
        args.q_heads // args.kv_heads,
    )
    torch.testing.assert_close(
        attention.float(), expected_attention.float(), atol=8e-2, rtol=3e-2
    )

    def launch_layer():
        rms_compiled(*rms_args)
        local_qkv = qkv_compiled(*qkv_args)
        local_qk_args = (local_qkv, *qk_args[1:])
        qk_compiled(*local_qk_args)
        local_query = local_qkv[:, : args.q_heads * args.head_dim].view(
            args.batch, args.q_heads, args.head_dim
        )
        local_key = local_qkv[
            :, key_begin : key_begin + args.kv_heads * args.head_dim
        ].view(args.batch, args.kv_heads, args.head_dim)
        local_value = local_qkv[
            :, key_begin + args.kv_heads * args.head_dim : qkv_width
        ].view(args.batch, args.kv_heads, args.head_dim)
        cache_compiled(
            local_key,
            local_value,
            tensors["kv_cache"],
            tensors["slot_mapping"],
            args.block_size,
        )
        local_partials, local_lse = attention_split_compiled(
            local_query,
            tensors["kv_cache"],
            tensors["block_table"],
            args.context,
            args.block_size,
            args.q_heads // args.kv_heads,
            args.attention_splits,
        )
        local_attention = attention_merge_compiled(local_partials, local_lse)
        attention_quant_compiled(
            local_attention.view(args.batch, args.hidden),
            tensors["attention_q"],
            tensors["attention_scale"],
            args.group,
            1e-10,
            FP8_MIN,
            FP8_MAX,
            False,
            False,
            False,
        )
        local_attention_out = o_compiled(*o_args)
        rms_compiled(
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
        local_gate = w13_compiled(*w13_args)
        local_activation_q, local_activation_scale = silu_compiled(
            local_gate, args.group
        )
        return w2_compiled(
            local_activation_q,
            local_activation_scale,
            tensors["w2_q"],
            tensors["w2_scale"],
            args.group,
        )

    if args.smoke or tune_set:
        print(
            "CONFIG_JSON",
            json.dumps(
                {
                    "rms_quant": dict(rms_config),
                    "qkv_mm": dict(qkv_config),
                    "qk_norm_rope": dict(qk_config),
                    "kv_cache_update": dict(cache_config),
                    "decode_attention_split": dict(attention_split_config),
                    "decode_attention_merge": dict(attention_merge_config),
                    "attention_quant": dict(attention_quant_config),
                    "o_mm": dict(o_config),
                    "w13": dict(w13_config),
                    "silu_quant": dict(silu_config),
                    "w2": dict(w2_config),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.smoke:
            print("smoke_ok", flush=True)
        if not args.benchmark:
            return

    graph, graph_out = capture(launch_layer)
    graph.replay()
    torch.cuda.synchronize()
    assert graph_out.shape == layer_out.shape
    benchmark_pids = visible_gpu_pids()
    timings = benchmark_interleaved(
        {"helion_separate_whole_layer_graph": graph.replay},
        args.repeats,
        args.batch_replays,
    )
    if args.benchmark_stages:
        stage_graphs = {
            "rms_quant_pre": capture(lambda: rms_compiled(*rms_args))[0].replay,
            "qkv_mm": capture(lambda: qkv_compiled(*qkv_args))[0].replay,
            "qk_norm_rope": capture(lambda: qk_compiled(*qk_args))[0].replay,
            "kv_cache_update": capture(lambda: cache_compiled(*cache_args))[0].replay,
            "decode_attention_split": capture(
                lambda: attention_split_compiled(*attention_split_args)
            )[0].replay,
            "decode_attention_merge": capture(
                lambda: attention_merge_compiled(*attention_merge_args)
            )[0].replay,
            "attention_quant": capture(
                lambda: attention_quant_compiled(*attention_quant_args)
            )[0].replay,
            "o_mm": capture(lambda: o_compiled(*o_args))[0].replay,
            "rms_quant_post": capture(lambda: rms_compiled(*post_args))[0].replay,
            "w13": capture(lambda: w13_compiled(*w13_args))[0].replay,
            "silu_quant": capture(lambda: silu_compiled(*silu_args))[0].replay,
            "w2": capture(lambda: w2_compiled(*w2_args))[0].replay,
        }
        timings.update(
            benchmark_interleaved(
                stage_graphs,
                args.repeats,
                args.batch_replays,
            )
        )
    final_pids = visible_gpu_pids()
    if final_pids != benchmark_pids:
        raise RuntimeError(
            f"GPU process set changed during benchmark: before={sorted(benchmark_pids)}, after={sorted(final_pids)}"
        )
    print(
        "RESULT_JSON",
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "shape": {
                    "batch": args.batch,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "q_heads": args.q_heads,
                    "kv_heads": args.kv_heads,
                    "head_dim": args.head_dim,
                    "context": args.context,
                    "block_size": args.block_size,
                    "attention_splits": args.attention_splits,
                },
                "kernel_count": 12,
                "timings": timings,
                "configs": {
                    **{name: configs.get(name) for name in sorted(configs)},
                    "w13": FFN_CONFIGS["w13"],
                    "silu_quant": FFN_CONFIGS["silu_quant"],
                    "w2": FFN_CONFIGS["w2"],
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=12288)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--attention-splits", type=int, default=32)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch-replays", type=int, default=10)
    parser.add_argument(
        "--config-path",
        default=str(Path(__file__).with_name("qwen3_layer_helion_b200_configs.json")),
    )
    parser.add_argument(
        "--tune",
        nargs="*",
        choices=[
            "rms_quant",
            "qkv_mm",
            "qk_norm_rope",
            "kv_cache_update",
            "decode_attention_split",
            "decode_attention_merge",
            "attention_quant",
            "o_mm",
        ],
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-stages", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
