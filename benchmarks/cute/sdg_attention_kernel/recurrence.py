"""Explicit BF16 post-decode /64 K/V representation with written score/output compensation."""

from __future__ import annotations

from typing import Annotated

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def paged_query_first_direct_attention(
    q_original: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    window_left: Annotated[int | None, hl.constexpr],
    key_block: Annotated[int, hl.constexpr],
    return_lse: Annotated[bool, hl.constexpr],
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch, query_len, heads, dim = q_original.shape
    pages, page_size, kv_heads, key_dim = k.shape
    group = heads // kv_heads
    rows = query_len * group
    items = batch * kv_heads
    assert heads % kv_heads == 0
    assert dim == key_dim
    assert key_block in (64, 128)
    assert page_size % key_block == 0
    assert dim % k_scale.shape[-1] == 0
    scale_group = dim // k_scale.shape[-1]
    subpages = page_size // key_block
    k_subpages = k.as_strided(
        (pages, subpages, key_block, kv_heads, dim),
        (k.stride(0), key_block * k.stride(1), k.stride(1), k.stride(2), k.stride(3)),
    )
    v_subpages = v.as_strided(
        (pages, subpages, key_block, kv_heads, dim),
        (v.stride(0), key_block * v.stride(1), v.stride(1), v.stride(2), v.stride(3)),
    )
    k_scale_subpages = k_scale.as_strided(
        (pages, subpages, key_block, kv_heads, k_scale.shape[3]),
        (
            k_scale.stride(0),
            key_block * k_scale.stride(1),
            k_scale.stride(1),
            k_scale.stride(2),
            k_scale.stride(3),
        ),
    )
    v_scale_subpages = v_scale.as_strided(
        (pages, subpages, key_block, kv_heads, v_scale.shape[3]),
        (
            v_scale.stride(0),
            key_block * v_scale.stride(1),
            v_scale.stride(1),
            v_scale.stride(2),
            v_scale.stride(3),
        ),
    )
    query_span = (
        1
        + (batch - 1) * q_original.stride(0)
        + (query_len - 1) * q_original.stride(1)
        + (heads - 1) * q_original.stride(2)
        + (dim - 1) * q_original.stride(3)
    )
    query_storage = q_original.as_strided((query_span,), (1,))
    out_span = (
        1
        + (batch - 1) * out.stride(0)
        + (query_len - 1) * out.stride(1)
        + (heads - 1) * out.stride(2)
        + (dim - 1) * out.stride(3)
    )
    output_storage = out.as_strided((out_span,), (1,))
    lse = torch.empty((batch, query_len, heads), device=out.device, dtype=torch.float32)
    lse_storage = lse.view(-1)
    scale = dim ** (-0.5) * 1.4426950408889634 * 64.0
    for item_tile, query_tile in hl.tile([items, rows], block_size=[1, None]):
        item = item_tile.id
        request = item // kv_heads
        kv_head = item % kv_heads
        query = query_tile.index // group
        head = kv_head * group + query_tile.index % group
        channel = hl.arange(dim)
        query_address = (
            request * q_original.stride(0)
            + query[:, None] * q_original.stride(1)
            + head[:, None] * q_original.stride(2)
            + channel[None, :] * q_original.stride(3)
        )
        q = query_storage[query_address]
        length = lengths[request]
        first_token = length * 0
        if window_left is not None:
            first_token = torch.clamp(length - query_len - window_left, min=0)
        first_page = first_token // page_size
        last_page = (length + page_size - 1) // page_size
        begin = first_page
        end = last_page
        query_position = length - query_len + query
        maximum = hl.full([query_tile], float("-inf"), dtype=torch.float32)
        denominator_lanes = hl.zeros([query_tile, key_block], dtype=torch.float32)
        accumulator = hl.zeros([query_tile, dim], dtype=torch.float32)
        for forward_tile in hl.grid(begin * subpages, end * subpages):
            token_tile = (begin + end) * subpages - 1 - forward_tile
            logical_page = token_tile // subpages
            subpage = token_tile % subpages
            physical_page = page_table[request, logical_page]
            key_integer = k_subpages[physical_page, subpage, :, kv_head, :]
            value_integer = v_subpages[physical_page, subpage, :, kv_head, :]
            key_scale = torch.exp2(
                k_scale_subpages[
                    physical_page, subpage, :, kv_head, hl.arange(dim) // scale_group
                ].float()
                - 127.0
            )
            value_scale = torch.exp2(
                v_scale_subpages[
                    physical_page, subpage, :, kv_head, hl.arange(dim) // scale_group
                ].float()
                - 127.0
            )
            key = (key_integer.float() * key_scale).to(q_original.dtype) * 0.015625
            value = (
                (value_integer.float() * value_scale).to(q_original.dtype) * 2.0**-22
            ) + 0.0
            reference = torch.where(
                (maximum >= 0.0) & (maximum < float("inf")), maximum, 0.0
            )
            relative_scores = hl.dot(q, key.T) * scale - reference[:, None]
            key_position = hl.arange(key_block) + token_tile * key_block
            visible = key_position[None, :] <= query_position[:, None]
            if window_left is not None:
                visible = visible & (
                    key_position[None, :] >= query_position[:, None] - window_left
                )
            relative_scores = torch.where(visible, relative_scores, float("-inf"))
            next_maximum = torch.maximum(
                maximum, torch.amax(relative_scores, dim=1) + reference
            )
            safe_maximum = torch.where(next_maximum == float("-inf"), 0.0, next_maximum)
            alpha = torch.exp2(maximum - safe_maximum)
            reference_shift = reference - safe_maximum
            probability = torch.exp2(relative_scores + reference_shift[:, None])
            denominator_lanes = denominator_lanes * alpha[:, None] + probability
            accumulator = hl.dot(
                probability.to(q_original.dtype),
                value,
                acc=accumulator * alpha[:, None],
            )
            maximum = next_maximum
        denominator = torch.sum(denominator_lanes, dim=1)
        safe_denominator = torch.where(denominator == 0, 1.0, denominator)
        output_address = (
            request * out.stride(0)
            + query[:, None] * out.stride(1)
            + head[:, None] * out.stride(2)
            + channel[None, :] * out.stride(3)
        )
        reciprocal = 1.0 / safe_denominator * 2.0**22
        output_storage[output_address] = accumulator * reciprocal[:, None]
        if return_lse:
            lse_address = (request * query_len + query) * heads + head
            lse_storage[lse_address] = maximum * 0.6931471805599453 + torch.log(
                denominator
            )
    return (out, lse if return_lse else None)
