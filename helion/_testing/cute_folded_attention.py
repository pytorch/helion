"""Small pure Helion attention fixture over original strided Q/O tensors.

The only device function performs QK, online softmax with a per-key denominator
carry, PV, and direct final output stores. No preparation or unpacking kernels.
Query row is query_index * group + group_head. Explicit flat storage views keep
paired row/head coordinates in one broadcasted index, preserving padded strides.
"""

from __future__ import annotations

import math
from typing import cast

import torch

import helion
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def folded_paged_attention(
    q_original: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    window_left: int | None,
    key_block: hl.constexpr,
    return_lse: hl.constexpr,
    out: torch.Tensor,
    reference_scores: hl.constexpr = False,  # pyrefly: ignore [bad-function-definition]
    reverse_pages: hl.constexpr = False,  # pyrefly: ignore [bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    window_left = hl.specialize(window_left)
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
    scale = dim**-0.5
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
        for iteration in hl.grid(begin * subpages, end * subpages):
            token_tile = (
                (begin + end) * subpages - 1 - iteration if reverse_pages else iteration
            )
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
            key = (key_integer.float() * key_scale).to(q_original.dtype)
            value = (value_integer.float() * value_scale).to(q_original.dtype)
            scores = hl.dot(q, key.T) * scale
            reference = maximum
            if reference_scores:
                reference = torch.where(
                    (maximum >= 0.0) & (maximum < float("inf")), maximum, 0.0
                )
                scores = scores - reference[:, None]
            key_position = hl.arange(key_block) + token_tile * key_block
            visible = key_position[None, :] <= query_position[:, None]
            if window_left is not None:
                visible = visible & (
                    key_position[None, :] >= query_position[:, None] - window_left
                )
            scores = torch.where(visible, scores, float("-inf"))
            if reference_scores:
                next_maximum = torch.maximum(
                    maximum, torch.amax(scores, dim=1) + reference
                )
            else:
                next_maximum = torch.maximum(maximum, torch.amax(scores, dim=1))
            safe_maximum = torch.where(next_maximum == float("-inf"), 0.0, next_maximum)
            alpha = torch.exp(maximum - safe_maximum)
            if reference_scores:
                probability = torch.exp(scores + (reference - safe_maximum)[:, None])
            else:
                probability = torch.exp(scores - safe_maximum[:, None])
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
        output_storage[output_address] = accumulator / safe_denominator[:, None]
        if return_lse:
            lse_address = (request * query_len + query) * heads + head
            lse_storage[lse_address] = maximum + torch.log(denominator)
    return out, lse if return_lse else None


def folded_attention_args(
    query_len: int, *, device: torch.device | str = "cpu"
) -> tuple[object, ...]:
    """Padded Q/O with pointer offsets, byte pages, and noncontiguous scales."""
    batch, heads, dim, pages = 2, 16, 128, 7
    q_strides = (query_len * 4608 + 256, 4608, 128, 1)
    out_strides = (query_len * 2176 + 256, 2176, 128, 1)
    q = torch.empty(batch * q_strides[0] + 8, dtype=torch.bfloat16, device=device)
    out = torch.full(
        (batch * out_strides[0] + 8,), -17, dtype=torch.bfloat16, device=device
    )
    storage = [
        torch.empty(pages * 33792, dtype=torch.int8, device=device) for _ in range(2)
    ]
    k, v = [x.as_strided((pages, 128, 2, 128), (33792, 256, 128, 1)) for x in storage]
    ks, vs = [
        x.view(torch.uint8).as_strided((pages, 128, 2, 4), (33792, 4, 512, 1), 32768)
        for x in storage
    ]
    return (
        q.as_strided((batch, query_len, heads, dim), q_strides, 8),
        k,
        v,
        ks,
        vs,
        torch.empty((batch, 3), dtype=torch.int32, device=device),
        torch.empty(batch, dtype=torch.int32, device=device),
        31,
        128,
        True,
        out.as_strided((batch, query_len, heads, dim), out_strides, 8),
    )


def folded_attention_reference(
    args: tuple[object, ...],
    *,
    reference_scores: bool = False,
    reverse_pages: bool = False,
    base2_scores: bool = False,
    output_reciprocal: bool = False,
    key_post_scale_log2: int = 0,
    value_post_scale_log2: int = 0,
    output_scale_log2: int = 0,
    value_positive_zero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager page recurrence with independent logical-to-original-layout mapping."""
    q, k, v, ks, vs, pages, lengths = (cast("torch.Tensor", arg) for arg in args[:7])
    window = cast("int | None", args[7])
    exponential = torch.exp2 if base2_scores else torch.exp
    batch, query_len, heads, dim = q.shape
    width, kv_heads = k.shape[1:3]
    group = heads // kv_heads
    output = torch.zeros_like(q)
    lse = torch.full(q.shape[:3], -float("inf"), device=q.device)
    for request in range(batch):
        length = int(lengths[request].item())
        begin = (
            max(length - query_len - window, 0) // width if window is not None else 0
        )
        end = (length + width - 1) // width
        for head in range(kv_heads):
            query = q[request, :, head * group : (head + 1) * group].reshape(-1, dim)
            maximum = torch.full((query_len * group,), -float("inf"), device=q.device)
            partials = torch.zeros((query_len * group, width), device=q.device)
            accumulator = torch.zeros_like(query, dtype=torch.float32)
            positions = (
                length
                - query_len
                + torch.arange(query_len, device=q.device).repeat_interleave(group)
            )
            tile_order = (
                range(end - 1, begin - 1, -1) if reverse_pages else range(begin, end)
            )
            for tile in tile_order:
                page = int(pages[request, tile].item())
                key = (
                    k[page, :, head].float()
                    * torch.exp2(ks[page, :, head].float() - 127).repeat_interleave(
                        32, -1
                    )
                ).to(q.dtype)
                value = (
                    v[page, :, head].float()
                    * torch.exp2(vs[page, :, head].float() - 127).repeat_interleave(
                        32, -1
                    )
                ).to(q.dtype)
                key = key * (2.0**key_post_scale_log2)
                value = value * (2.0**value_post_scale_log2)
                if value_positive_zero:
                    value = value + 0.0
                score_scale = dim**-0.5
                if base2_scores:
                    score_scale *= math.log2(math.e)
                score_scale *= 2.0**-key_post_scale_log2
                scores = query.float() @ key.float().T * score_scale
                reference = maximum
                if reference_scores:
                    reference = torch.where(
                        (maximum >= 0) & (maximum < float("inf")), maximum, 0.0
                    )
                    scores = scores - reference[:, None]
                keys = torch.arange(width, device=q.device) + tile * width
                visible = keys[None, :] <= positions[:, None]
                if window is not None:
                    visible &= keys[None, :] >= positions[:, None] - window
                scores = torch.where(visible, scores, -float("inf"))
                if reference_scores:
                    next_maximum = torch.maximum(maximum, scores.amax(1) + reference)
                else:
                    next_maximum = torch.maximum(maximum, scores.amax(1))
                safe_maximum = torch.where(
                    next_maximum == -float("inf"), 0, next_maximum
                )
                alpha = exponential(maximum - safe_maximum)
                if reference_scores:
                    probability = exponential(
                        scores + (reference - safe_maximum)[:, None]
                    )
                else:
                    probability = exponential(scores - safe_maximum[:, None])
                partials = partials * alpha[:, None] + probability
                accumulator = (
                    accumulator * alpha[:, None]
                    + probability.to(q.dtype).float() @ value.float()
                )
                maximum = next_maximum
            denominator = partials.sum(1)
            safe_denominator = torch.where(denominator == 0, 1, denominator)
            normalized = (
                accumulator
                * ((1.0 / safe_denominator) * (2.0**output_scale_log2))[:, None]
                if output_reciprocal
                else accumulator / safe_denominator[:, None]
            )
            output[request, :, head * group : (head + 1) * group] = normalized.reshape(
                query_len, group, dim
            )
            lse[request, :, head * group : (head + 1) * group] = (
                (maximum * math.log(2) if base2_scores else maximum)
                + torch.log(denominator)
            ).reshape(query_len, group)
    return output, lse
