# ruff: noqa: A001, A002, ANN001, ANN202
"""Matched PDL-enabled Helion baseline for the Qwen3 decode-layer probe."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pretuned_kernels.megakernels._pdl import launch_dependent
from pretuned_kernels.megakernels._pdl import signal_dependents
from pretuned_kernels.megakernels._pdl import wait_and_launch_dependents
import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)

# These are the independently tuned B200 configurations from the two preserved
# production-boundary controls.  Fine-grained attention/cache kernels use the
# ordinary persistent launch envelope of the 106.224 us control; projections,
# normalization, and activation retain their operation-specific choices.
GRANULAR_CONFIG = {
    "num_warps": 1,
    "num_stages": 2,
    "pid_type": "persistent_blocked",
    "num_sm_multiplier": 8,
}
RMS_CONFIG = {
    "block_sizes": [4096, 32],
    "num_warps": 16,
    "num_stages": 1,
    "pid_type": "flat",
}
QKV_CONFIG = {
    "block_sizes": [16],
    "loop_orders": [[0, 1]],
    "l2_groupings": [1],
    "range_unroll_factors": [0, 2],
    "range_num_stages": [0, 0],
    "range_multi_buffers": [None, True],
    "range_flattens": [None, False],
    "num_warps": 1,
    "num_stages": 4,
    "pid_type": "flat",
}
ATTENTION_QUANT_CONFIG = {
    "block_sizes": [16],
    "loop_orders": [[0, 1, 2]],
    "l2_groupings": [1],
    "range_unroll_factors": [0],
    "range_multi_buffers": [None],
    "range_flattens": [None],
    "num_warps": 4,
    "num_stages": 1,
    "pid_type": "flat",
}
O_CONFIG = {
    "block_sizes": [8],
    "loop_orders": [[1, 0]],
    "l2_groupings": [4],
    "range_unroll_factors": [0, 2],
    "range_num_stages": [0, 0],
    "range_multi_buffers": [None, False],
    "range_flattens": [None, False],
    "num_warps": 1,
    "num_stages": 4,
    "pid_type": "flat",
}
W13_CONFIG = {
    "block_sizes": [16],
    "loop_orders": [[0, 1]],
    "l2_groupings": [1],
    "range_unroll_factors": [0, 2],
    "range_num_stages": [0, 0],
    "range_multi_buffers": [None, True],
    "range_flattens": [None, False],
    "num_warps": 1,
    "num_stages": 4,
    "pid_type": "flat",
}
ACTIVATION_CONFIG = {
    "loop_orders": [[0, 1]],
    "l2_groupings": [1],
    "range_unroll_factors": [0],
    "range_num_stages": [0],
    "range_multi_buffers": [None],
    "range_flattens": [None],
    "num_warps": 4,
    "num_stages": 1,
    "pid_type": "flat",
}
W2_CONFIG = {
    "block_sizes": [8],
    "loop_orders": [[0, 1]],
    "l2_groupings": [1],
    "range_unroll_factors": [0, 4],
    "range_num_stages": [0, 4],
    "range_multi_buffers": [None, False],
    "range_flattens": [None, True],
    "num_warps": 1,
    "num_stages": 4,
    "pid_type": "flat",
}


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def rms_norm_per_block_quant(
    result,
    input,
    weight,
    scale,
    epsilon,
    residual,
    group_size,
):
    """Residual RMSNorm and FP8 quantization at the production op boundary."""
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)
    groups_per_row = scale.shape[1]
    hl.specialize(groups_per_row)
    for tile_m in hl.tile(num_tokens, block_size=1):
        signal_dependents()
        rms = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            values = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                values = values + residual[tile_m, tile_n]
            rms = rms + values.pow(2).sum(dim=-1)
        rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)

        m_idx = tile_m.begin + hl.arange(tile_m.block_size)
        m_block = m_idx[:, None, None]
        for tile_group, tile_n in hl.tile(
            [groups_per_row, group_size], block_size=[None, group_size]
        ):
            group_idx = tile_group.index
            n_idx = group_idx[:, None] * group_size + tile_n.index[None, :]
            n_block = n_idx[None, :, :]
            values = input[m_block, n_block].to(torch.float32)
            if residual is not None:
                values = values + residual[m_block, n_block]
            normalized = (values * rms[:, None, None]).to(torch.bfloat16) * weight[
                n_block
            ]
            quant_scale = (
                torch.amax(torch.abs(normalized), dim=-1).to(torch.float32) / FP8_MAX
            ).clamp(min=FP8_MIN_SCALE)
            scale[tile_m, tile_group] = quant_scale
            result[m_block, n_block] = (
                (normalized / quant_scale[:, :, None])
                .clamp(FP8_MIN, FP8_MAX)
                .to(result.dtype)
            )
            if residual is not None:
                residual[m_block, n_block] = values.to(residual.dtype)


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def rms_norm_per_block_quant_pdl(
    result,
    input,
    weight,
    scale,
    epsilon,
    residual,
    group_size,
):
    """Downstream RMSNorm/quant launch with a programmatic dependency."""
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)
    groups_per_row = scale.shape[1]
    hl.specialize(groups_per_row)
    for tile_m in hl.tile(num_tokens, block_size=1):
        wait_and_launch_dependents()
        rms = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            values = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                values = values + residual[tile_m, tile_n]
            rms = rms + values.pow(2).sum(dim=-1)
        rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)

        m_idx = tile_m.begin + hl.arange(tile_m.block_size)
        m_block = m_idx[:, None, None]
        for tile_group, tile_n in hl.tile(
            [groups_per_row, group_size], block_size=[None, group_size]
        ):
            group_idx = tile_group.index
            n_idx = group_idx[:, None] * group_size + tile_n.index[None, :]
            n_block = n_idx[None, :, :]
            values = input[m_block, n_block].to(torch.float32)
            if residual is not None:
                values = values + residual[m_block, n_block]
            normalized = (values * rms[:, None, None]).to(torch.bfloat16) * weight[
                n_block
            ]
            quant_scale = (
                torch.amax(torch.abs(normalized), dim=-1).to(torch.float32) / FP8_MAX
            ).clamp(min=FP8_MIN_SCALE)
            scale[tile_m, tile_group] = quant_scale
            result[m_block, n_block] = (
                (normalized / quant_scale[:, :, None])
                .clamp(FP8_MIN, FP8_MAX)
                .to(result.dtype)
            )
            if residual is not None:
                residual[m_block, n_block] = values.to(residual.dtype)


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def block_fp8_mm(activation_q, activation_scale, weight_q, weight_scale, group_size):
    m, k = activation_q.size()
    n, weight_k = weight_q.size()
    assert weight_k == k
    hl.specialize(group_size)
    output = torch.empty((m, n), dtype=torch.bfloat16, device=activation_q.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        wait_and_launch_dependents()
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=group_size):
            partial = hl.dot(
                activation_q[tile_m, tile_k],
                weight_q[tile_n, tile_k].T,
            ).to(torch.float32)
            activation_s = activation_scale[tile_m, tile_k.id].to(torch.float32)
            weight_s = weight_scale[tile_n.index // group_size, tile_k.id].to(
                torch.float32
            )
            acc = acc + partial * activation_s[:, None] * weight_s[None, :]
        output[tile_m, tile_n] = acc.to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def fused_qk_norm_rope(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    eps,
    q_weight,
    k_weight,
    cos_sin_cache,
    position_ids,
):
    """Apply Q/K normalization and NeoX RoPE over flat head-sized tiles."""
    num_tokens, qkv_width = qkv.shape
    total_heads = num_heads_q + num_heads_k + num_heads_v
    assert qkv_width == total_heads * head_dim
    _, rotary_dim = cos_sin_cache.shape
    hl.specialize(qkv_width)
    hl.specialize(rotary_dim)
    hl.specialize(num_heads_q)
    hl.specialize(num_heads_k)
    hl.specialize(num_heads_v)
    hl.specialize(head_dim)
    embed_dim = rotary_dim // 2
    qk_width = (num_heads_q + num_heads_k) * head_dim
    for tile_m, tile_n in hl.tile([num_tokens, qk_width], block_size=[1, head_dim]):
        wait_and_launch_dependents()
        x = qkv[tile_m, tile_n].to(torch.float32)
        rms = torch.rsqrt(x.pow(2).sum(-1) * (1.0 / head_dim) + eps)
        dimension = tile_n.index - tile_n.begin
        use_q = tile_n.index < num_heads_q * head_dim
        norm_weight = torch.where(use_q, q_weight[dimension], k_weight[dimension])
        x = (x * rms[:, None]).to(qkv.dtype) * norm_weight[None, :]
        position = position_ids[tile_m]
        first_half = dimension < embed_dim
        partner_dimension = torch.where(
            first_half, dimension + embed_dim, dimension - embed_dim
        )
        partner = torch.gather(x, 1, partner_dimension[None, :])
        cos = cos_sin_cache[position, dimension % embed_dim]
        sin = cos_sin_cache[position, dimension % embed_dim + embed_dim]
        qkv[tile_m, tile_n] = x * cos + torch.where(
            first_half[None, :], -partner * sin, partner * sin
        )


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def reshape_and_cache_flash(key, value, kv_cache, slot_mapping, block_size):
    num_tokens, num_kv_heads, head_dim = key.shape
    hl.specialize(num_kv_heads)
    hl.specialize(head_dim)
    hl.specialize(block_size)
    for tile_t, tile_h, tile_d in hl.tile(
        [num_tokens, num_kv_heads, head_dim],
        block_size=[1, 1, head_dim],
    ):
        wait_and_launch_dependents()
        token = tile_t.index
        cache_head = tile_h.index
        dimension = tile_d.index
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
            key[tile_t, tile_h, tile_d],
        )
        hl.store(
            kv_cache,
            [
                block,
                offset,
                cache_head[None, :, None],
                (dimension + head_dim)[None, None, :],
            ],
            value[tile_t, tile_h, tile_d],
        )


@helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    backend="triton",
    triton_do_not_specialize=False,
)
def paged_gqa_decode_attention_split(
    query,
    kv_cache,
    block_table,
    context_lens,
    context_capacity,
    block_size,
    q_per_kv,
    splits,
):
    """Runtime-ragged split attention at a fixed physical capacity."""
    num_tokens, num_q_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[2]
    assert num_q_heads == num_kv_heads * q_per_kv
    assert context_capacity % splits == 0
    hl.specialize(num_tokens)
    hl.specialize(num_q_heads)
    hl.specialize(head_dim)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(context_capacity)
    hl.specialize(block_size)
    hl.specialize(splits)
    hl.specialize(query.stride(0))
    hl.specialize(query.stride(1))
    hl.specialize(query.stride(2))
    hl.specialize(kv_cache.stride(0))
    hl.specialize(kv_cache.stride(1))
    hl.specialize(kv_cache.stride(2))
    hl.specialize(kv_cache.stride(3))
    hl.specialize(block_table.size(0))
    hl.specialize(block_table.size(1))
    hl.specialize(block_table.stride(0))
    hl.specialize(block_table.stride(1))
    hl.specialize(context_lens.size(0))
    hl.specialize(context_lens.stride(0))

    split_context = context_capacity // splits
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
        wait_and_launch_dependents()
        m_i = hl.full([tile_bg, tile_q], -3.4028234663852886e38, dtype=torch.float32)
        l_i = hl.full([tile_bg, tile_q], 1.0, dtype=torch.float32)
        acc = hl.zeros([tile_bg, tile_q, head_dim], dtype=torch.float32)
        split_idx = tile_split.begin
        token = tile_bg.index // num_kv_heads
        kv_head = tile_bg.index % num_kv_heads
        context_len = hl.load(context_lens, [token])
        if split_idx * split_context < context_len:
            query_head = kv_head[:, None] * q_per_kv + tile_q.index[None, :]
            q_block = query[token[:, None], query_head, :]
            q_block = (q_block * qk_scale).to(query.dtype)
            for tile_local_n in hl.tile(split_context):
                n = split_idx * split_context + tile_local_n.index
                valid_n = n < context_len
                physical_block = hl.load(
                    block_table,
                    [token[:, None], (n // block_size)[None, :]],
                    extra_mask=valid_n[None, :],
                )
                block_offset = n % block_size
                dimension = hl.arange(head_dim)
                key = hl.load(
                    kv_cache,
                    [
                        physical_block[:, :, None],
                        block_offset[None, :, None],
                        kv_head[:, None, None],
                        dimension[None, None, :],
                    ],
                    extra_mask=valid_n[None, :, None],
                )
                scores = torch.bmm(q_block, key.transpose(1, 2), torch.float32)
                scores = torch.where(
                    valid_n[None, None, :],
                    scores,
                    -3.4028234663852886e38,
                )
                next_max = torch.maximum(m_i, torch.amax(scores, -1))
                probabilities = torch.exp2(scores - next_max[:, :, None])
                alpha = torch.exp2(m_i - next_max)
                l_i = l_i * alpha + torch.sum(probabilities, -1)
                acc = acc * alpha[:, :, None]
                value = hl.load(
                    kv_cache,
                    [
                        physical_block[:, :, None],
                        block_offset[None, :, None],
                        kv_head[:, None, None],
                        (dimension + head_dim)[None, None, :],
                    ],
                    extra_mask=valid_n[None, :, None],
                )
                acc = torch.baddbmm(acc, probabilities.to(value.dtype), value)
                m_i = next_max
        partial_out[tile_split, tile_bg, tile_q, :] = (acc / l_i[:, :, None])[
            None, :, :, :
        ]
        partial_lse[tile_split, tile_bg, tile_q] = (m_i + torch.log2(l_i))[None, :, :]
    return partial_out, partial_lse


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def merge_attention_splits(partial_out, partial_lse, merge_chunks):
    splits, num_kv_heads, q_per_kv, head_dim = partial_out.shape
    hl.specialize(splits)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(head_dim)
    hl.specialize(merge_chunks)
    query_heads = num_kv_heads * q_per_kv
    splits_per_chunk = splits // merge_chunks
    hl.specialize(query_heads)
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
        (query_heads, head_dim), dtype=torch.bfloat16, device=partial_out.device
    )
    for tile_chunk, tile_head in hl.tile(
        [merge_chunks, query_heads], block_size=[1, 1]
    ):
        wait_and_launch_dependents()
        split_idx = (
            tile_chunk.index[:, None] * splits_per_chunk
            + hl.arange(splits_per_chunk)[None, :]
        )
        lse_offsets = (
            split_idx[:, :, None] * query_heads + tile_head.index[None, None, :]
        )
        lse_values = partial_lse_storage[lse_offsets]
        max_lse = torch.amax(lse_values, dim=1)
        weights = torch.exp2(lse_values - max_lse[:, None, :])
        value_offsets = (
            lse_offsets[:, :, :, None] * head_dim
            + hl.arange(head_dim)[None, None, None, :]
        )
        values = partial_out_storage[value_offsets]
        denominator = torch.sum(weights, dim=1)
        merged = (
            torch.sum(values * weights[:, :, :, None], dim=1) / denominator[:, :, None]
        )
        chunk_out[tile_chunk, tile_head, :] = merged
        chunk_lse[tile_chunk, tile_head] = max_lse + torch.log2(denominator)
    for final_head in hl.tile(query_heads, block_size=1):
        wait_and_launch_dependents()
        chunk_idx = hl.arange(merge_chunks)
        lse_offsets = chunk_idx[:, None] * query_heads + final_head.index[None, :]
        lse_values = chunk_lse_storage[lse_offsets]
        max_lse = torch.amax(lse_values, dim=0)
        weights = torch.exp2(lse_values - max_lse[None, :])
        value_offsets = (
            lse_offsets[:, :, None] * head_dim + hl.arange(head_dim)[None, None, :]
        )
        values = chunk_out_storage[value_offsets]
        denominator = torch.sum(weights, dim=0)
        merged = torch.sum(values * weights[:, :, None], dim=0)
        output[final_head, :] = (merged / denominator[:, None]).to(output.dtype)
    return output.view(1, query_heads, head_dim)


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def per_token_group_fp8_quant(
    input,
    output_q,
    output_s,
    group_size,
    eps,
):
    num_tokens, hidden_size = input.shape
    groups_per_row = output_s.shape[1]
    hl.specialize(hidden_size)
    hl.specialize(group_size)
    hl.specialize(groups_per_row)
    input = input.view(num_tokens, groups_per_row, group_size)
    output_q = output_q.view(num_tokens, groups_per_row, group_size)
    for tile_m, tile_group, tile_n in hl.tile(
        [num_tokens, groups_per_row, group_size],
        block_size=[1, None, group_size],
    ):
        wait_and_launch_dependents()
        values = input[tile_m, tile_group, tile_n]
        scale = torch.amax(torch.abs(values), dim=-1).clamp(min=eps) / FP8_MAX
        output_s[tile_m, tile_group] = scale
        output_q[tile_m, tile_group, tile_n] = (
            (values / scale[:, :, None]).clamp(FP8_MIN, FP8_MAX).to(output_q.dtype)
        )


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def silu_and_mul_per_block_quant(gate_up, group_size):
    m, twice_intermediate = gate_up.size()
    intermediate = twice_intermediate // 2
    groups = intermediate // group_size
    hl.specialize(group_size)
    activation_q = torch.empty(
        (m, intermediate), dtype=torch.float8_e4m3fn, device=gate_up.device
    )
    activation_scale = torch.empty(
        (m, groups), dtype=torch.float32, device=gate_up.device
    )
    for tile_m, tile_i in hl.tile([m, intermediate], block_size=[1, group_size]):
        wait_and_launch_dependents()
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


def _compile(kernel, args, config_values):
    bound = kernel.bind(args)
    values = dict(bound.config_spec.default_config())
    values.update(config_values)
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return bound.compile_config(config)


def build(
    tensors: dict[str, torch.Tensor],
    *,
    hidden: int,
    intermediate: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    context: int,
    cache_block: int,
    attention_splits: int,
    group: int,
    eps: float,
) -> tuple[Callable[[], tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
    """Compile and materialize the matched twelve-launch PDL decoder graph."""
    rms_args = (
        tensors["pre_q"],
        tensors["hidden_states"],
        tensors["pre_weight"],
        tensors["pre_scale"],
        eps,
        tensors["residual"],
        group,
    )
    rms = _compile(rms_norm_per_block_quant, rms_args, RMS_CONFIG)
    rms(*rms_args)

    qkv_args = (
        tensors["pre_q"],
        tensors["pre_scale"],
        tensors["qkv_weight_q"],
        tensors["qkv_weight_scale"],
        group,
    )
    qkv_mm = _compile(block_fp8_mm, qkv_args, QKV_CONFIG)
    qkv = launch_dependent(qkv_mm, *qkv_args)

    qk_args = (
        qkv,
        q_heads,
        kv_heads,
        kv_heads,
        head_dim,
        eps,
        tensors["q_weight"],
        tensors["k_weight"],
        tensors["cos_sin"],
        tensors["position"],
    )
    qk = _compile(fused_qk_norm_rope, qk_args, GRANULAR_CONFIG)
    launch_dependent(qk, *qk_args)
    key_begin = q_heads * head_dim
    qkv_width = (q_heads + 2 * kv_heads) * head_dim
    query = qkv[:, :key_begin].view(-1, q_heads, head_dim)
    key = qkv[:, key_begin : key_begin + kv_heads * head_dim].view(
        -1, kv_heads, head_dim
    )
    value = qkv[:, key_begin + kv_heads * head_dim : qkv_width].view(
        -1, kv_heads, head_dim
    )

    cache_args = (key, value, tensors["kv_cache"], tensors["slot_mapping"], cache_block)
    cache = _compile(reshape_and_cache_flash, cache_args, GRANULAR_CONFIG)
    launch_dependent(cache, *cache_args)

    attention_args = (
        query,
        tensors["kv_cache"],
        tensors["block_table"],
        tensors["context_lens"],
        context,
        cache_block,
        q_heads // kv_heads,
        attention_splits,
    )
    attention_split = _compile(
        paged_gqa_decode_attention_split, attention_args, GRANULAR_CONFIG
    )
    partial_out, partial_lse = launch_dependent(attention_split, *attention_args)

    merge_args = (partial_out, partial_lse, 16)
    merge = _compile(merge_attention_splits, merge_args, GRANULAR_CONFIG)
    attention = launch_dependent(merge, *merge_args)

    attention_quant_args = (
        attention.view(-1, hidden),
        tensors["attention_q"],
        tensors["attention_scale"],
        group,
        1e-10,
    )
    attention_quant = _compile(
        per_token_group_fp8_quant,
        attention_quant_args,
        ATTENTION_QUANT_CONFIG,
    )
    launch_dependent(attention_quant, *attention_quant_args)

    o_args = (
        tensors["attention_q"],
        tensors["attention_scale"],
        tensors["o_weight_q"],
        tensors["o_weight_scale"],
        group,
    )
    o_mm = _compile(block_fp8_mm, o_args, O_CONFIG)
    attention_out = launch_dependent(o_mm, *o_args)

    post_args = (
        tensors["ffn_q"],
        attention_out,
        tensors["post_weight"],
        tensors["ffn_scale"],
        eps,
        tensors["residual"],
        group,
    )
    post_rms = _compile(rms_norm_per_block_quant_pdl, post_args, RMS_CONFIG)
    launch_dependent(post_rms, *post_args)

    w13_args = (
        tensors["ffn_q"],
        tensors["ffn_scale"],
        tensors["w13_q"],
        tensors["w13_scale"],
        group,
    )
    w13 = _compile(block_fp8_mm, w13_args, W13_CONFIG)
    gate_up = launch_dependent(w13, *w13_args)

    activation_args = (gate_up, group)
    activation = _compile(
        silu_and_mul_per_block_quant, activation_args, ACTIVATION_CONFIG
    )
    activation_q, activation_scale = launch_dependent(activation, *activation_args)

    w2_args = (
        activation_q,
        activation_scale,
        tensors["w2_q"],
        tensors["w2_scale"],
        group,
    )
    w2 = _compile(block_fp8_mm, w2_args, W2_CONFIG)
    output = launch_dependent(w2, *w2_args)

    def launch() -> tuple[torch.Tensor, ...]:
        rms(*rms_args)
        local_qkv = launch_dependent(qkv_mm, *qkv_args)
        launch_dependent(qk, local_qkv, *qk_args[1:])
        local_query = local_qkv[:, :key_begin].view(-1, q_heads, head_dim)
        local_key = local_qkv[:, key_begin : key_begin + kv_heads * head_dim].view(
            -1, kv_heads, head_dim
        )
        local_value = local_qkv[:, key_begin + kv_heads * head_dim : qkv_width].view(
            -1, kv_heads, head_dim
        )
        launch_dependent(
            cache,
            local_key,
            local_value,
            tensors["kv_cache"],
            tensors["slot_mapping"],
            cache_block,
        )
        local_partials, local_lse = launch_dependent(
            attention_split,
            local_query,
            tensors["kv_cache"],
            tensors["block_table"],
            tensors["context_lens"],
            context,
            cache_block,
            q_heads // kv_heads,
            attention_splits,
        )
        local_attention = launch_dependent(merge, local_partials, local_lse, 16)
        launch_dependent(
            attention_quant,
            local_attention.view(-1, hidden),
            tensors["attention_q"],
            tensors["attention_scale"],
            group,
            1e-10,
        )
        local_attention_out = launch_dependent(o_mm, *o_args)
        launch_dependent(
            post_rms,
            tensors["ffn_q"],
            local_attention_out,
            tensors["post_weight"],
            tensors["ffn_scale"],
            eps,
            tensors["residual"],
            group,
        )
        local_gate_up = launch_dependent(w13, *w13_args)
        local_activation_q, local_activation_scale = launch_dependent(
            activation, local_gate_up, group
        )
        local_output = launch_dependent(
            w2,
            local_activation_q,
            local_activation_scale,
            tensors["w2_q"],
            tensors["w2_scale"],
            group,
        )
        return (
            local_output,
            tensors["pre_q"],
            tensors["pre_scale"],
            local_qkv,
            local_partials,
            local_lse,
            local_attention.view(-1, head_dim),
            tensors["attention_q"],
            tensors["attention_scale"],
            local_attention_out,
            tensors["ffn_q"],
            tensors["ffn_scale"],
            local_gate_up,
            local_activation_q,
            local_activation_scale,
            tensors["residual"],
        )

    outputs = (
        output,
        tensors["pre_q"],
        tensors["pre_scale"],
        qkv,
        partial_out,
        partial_lse,
        attention.view(-1, head_dim),
        tensors["attention_q"],
        tensors["attention_scale"],
        attention_out,
        tensors["ffn_q"],
        tensors["ffn_scale"],
        gate_up,
        activation_q,
        activation_scale,
        tensors["residual"],
    )
    return launch, outputs
