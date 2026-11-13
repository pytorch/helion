"""Tunable partition of the authored two-stage conditional SPEC recurrence.

Only the source partition count becomes an integer config value. The threshold,
page ordering within each part, native decoder and FP32/BF16 boundaries are the
same authored contract. A changed partition can change floating-point results
and must pass the unchanged canonical oracle before performance evaluation.
"""

from __future__ import annotations

import math

import torch

from .cute_paged_conditional_semantics import native_s2f6_decode
from .cute_paged_conditional_semantics import paged_denominator_sum
from .cute_paged_conditional_semantics import paged_score_difference
import helion
from helion.autotuner import EnumFragment
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def tunable_conditional_attention(
    q_original: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    sm_count: int,
    page_slots: int,
    window_left: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sm_count = hl.specialize(sm_count)
    page_slots = hl.specialize(page_slots)
    window_left = hl.specialize(window_left)
    batch, query_len, heads, dim = q_original.shape
    work_buckets = hl.register_tunable(
        "partition_buckets",
        EnumFragment((min(batch, sm_count), min(batch, max(1, sm_count // 2)), batch)),
    )
    _, page_size, kv_heads, key_dim = k.shape
    group = heads // kv_heads
    rows = query_len * group
    assert query_len == 4 and rows == 32 and dim == key_dim == 128
    assert heads == 16 and kv_heads == 2 and page_size == 128
    assert q_original.dtype == output.dtype == torch.bfloat16
    assert lse.dtype == torch.float32 and lengths.dtype == torch.int32
    assert k.dtype == v.dtype == torch.int8
    assert k_scale.dtype == v_scale.dtype == torch.uint8
    assert page_table.dtype == torch.int32
    assert 1 <= work_buckets <= batch and page_slots > 0 and window_left >= 0
    assert page_slots * page_size >= window_left + query_len + page_size - 1
    assert output.shape == q_original.shape
    assert lse.shape == q_original.shape[:-1]
    # Lengths must be nonnegative and covered by the table. Every visited
    # physical page must be valid. These are input preconditions, not masks.
    partial_o = torch.empty(
        (batch * kv_heads * 2, rows // 4, dim, 4),
        dtype=torch.float32,
        device=q_original.device,
    )
    partial_ml = torch.empty(
        (batch * kv_heads * 2, rows, 2),
        dtype=torch.float32,
        device=q_original.device,
    )
    partial_flat = partial_o.reshape(-1)
    query_span = (
        1
        + (batch - 1) * q_original.stride(0)
        + (query_len - 1) * q_original.stride(1)
        + (heads - 1) * q_original.stride(2)
        + (dim - 1) * q_original.stride(3)
    )
    query_storage = q_original.as_strided((query_span,), (1,))
    lse_span = (
        1
        + (batch - 1) * lse.stride(0)
        + (query_len - 1) * lse.stride(1)
        + (heads - 1) * lse.stride(2)
    )
    lse_storage = lse.as_strided((lse_span,), (1,))
    total_slots = batch * page_slots
    factor = 64 * 1.4426950408889634 / math.sqrt(dim)

    # Publish only split requests' newer parts. Unused slots remain unread.
    for item in hl.tile(batch * kv_heads, block_size=1):
        request = item.id // kv_heads
        kv_head = item.id % kv_heads
        first_owner = ((request * page_slots + 1) * work_buckets - 1) // total_slots
        last_owner = ((request + 1) * page_slots * work_buckets - 1) // total_slots
        split = first_owner != last_owner
        cut = (first_owner + 1) * total_slots // work_buckets - request * page_slots
        row = hl.arange(rows)
        channel = hl.arange(dim)
        query = row // group
        head = kv_head * group + row % group
        slot = (request * kv_heads + kv_head) * 2 + 1
        partial_address = (
            slot * rows * dim
            + (row[:, None] // 4) * (dim * 4)
            + channel[None, :] * 4
            + row[:, None] % 4
        )
        if split:
            owner = last_owner
            relative_begin = cut
            relative_end = page_slots
            maximum = hl.full([rows], float("-inf"), dtype=torch.float32)
            denominator_lanes = hl.zeros([rows, page_size], dtype=torch.float32)
            accumulator = hl.zeros([rows, dim], dtype=torch.float32)
            seeded = False
            length = lengths[request]
            range_lo = torch.clamp_min(length - query_len - window_left, 0)
            first_page = range_lo // page_size
            pa = first_page + relative_begin
            pb = torch.minimum(
                first_page + relative_end, (length + page_size - 1) // page_size
            )
            count = torch.clamp_min(pb - pa, 0)
            key_lo = torch.maximum(range_lo, pa * page_size)
            key_hi = torch.minimum(length, pb * page_size)
            owner_end = (
                (owner + 1) * total_slots // work_buckets + page_slots - 1
            ) // page_slots
            move_oldest = (
                (request == owner_end - 1) & (key_lo > pa * page_size) & (count > 4)
            )
            oldest_position = torch.where(move_oldest, count - 4, count + 64)
            query_position = length - query_len + query
            query_address = (
                request * q_original.stride(0)
                + query[:, None] * q_original.stride(1)
                + head[:, None] * q_original.stride(2)
                + channel[None, :] * q_original.stride(3)
            )
            q = query_storage[query_address]
            for iteration in hl.grid(count):
                logical_page = torch.where(
                    oldest_position > iteration,
                    pa + count - 1 - iteration,
                    torch.where(
                        torch.eq(oldest_position, iteration), pa, pa + count - iteration
                    ),
                )
                token_base = logical_page * page_size
                shift = torch.where(
                    key_lo > token_base,
                    (key_lo - token_base) & -4,
                    torch.where(
                        key_hi < token_base + page_size,
                        -((token_base + page_size - key_hi) & -4),
                        0,
                    ),
                )
                token_row = hl.arange(page_size) + shift
                in_page = (token_row >= 0) & (token_row < page_size)
                safe_row = torch.clamp(token_row, min=0, max=page_size - 1)
                physical_page = page_table[request, logical_page]
                raw_key = k[physical_page, safe_row[:, None], kv_head, channel[None, :]]
                raw_value = v[
                    physical_page, safe_row[:, None], kv_head, channel[None, :]
                ]
                key_code = k_scale[
                    physical_page, safe_row[:, None], kv_head, channel[None, :] // 32
                ]
                value_code = v_scale[
                    physical_page, safe_row[:, None], kv_head, channel[None, :] // 32
                ]
                key = torch.where(
                    in_page[:, None], native_s2f6_decode(raw_key, key_code), 0.0
                )
                value = torch.where(
                    in_page[:, None], native_s2f6_decode(raw_value, value_code), 0.0
                )
                scores = hl.dot(q, key.T)
                key_position = token_base + token_row
                visible = (
                    in_page[None, :]
                    & (key_position[None, :] >= key_lo)
                    & (key_position[None, :] < key_hi)
                    & (key_position[None, :] <= query_position[:, None])
                    & (key_position[None, :] >= query_position[:, None] - window_left)
                )
                difference = paged_score_difference(scores, maximum, visible, factor)
                probability = torch.exp2(difference)
                # One vote spans all query columns and key lanes belonging to this head.
                if torch.any(difference > 2.0):
                    scaled_scores = torch.where(visible, scores * factor, float("-inf"))
                    # Native fmax reductions ignore NaNs; visible NaNs remain in P.
                    maximum_scores = torch.where(
                        torch.isnan(scaled_scores), float("-inf"), scaled_scores
                    )
                    next_maximum = torch.fmax(
                        maximum, torch.amax(maximum_scores, dim=1)
                    )
                    alpha = torch.where(
                        next_maximum == float("-inf"),
                        1.0,
                        torch.exp2(maximum - next_maximum),
                    )
                    if (iteration > 0) | seeded:
                        accumulator = accumulator * alpha[:, None]
                        denominator_lanes = denominator_lanes * alpha[:, None]
                    maximum = next_maximum
                    probability = torch.exp2(
                        paged_score_difference(scores, maximum, visible, factor)
                    )
                denominator_lanes = denominator_lanes + probability
                accumulator = hl.dot(
                    probability.to(torch.bfloat16), value, acc=accumulator
                )
            denominator = paged_denominator_sum(denominator_lanes)
            partial_flat[partial_address] = accumulator
            partial_ml[slot, row, 0] = maximum
            partial_ml[slot, row, 1] = denominator

    # Continue each imported newer state through its older pages.
    for item in hl.tile(batch * kv_heads, block_size=1):
        request = item.id // kv_heads
        kv_head = item.id % kv_heads
        first_owner = ((request * page_slots + 1) * work_buckets - 1) // total_slots
        last_owner = ((request + 1) * page_slots * work_buckets - 1) // total_slots
        split = first_owner != last_owner
        cut = (first_owner + 1) * total_slots // work_buckets - request * page_slots
        row = hl.arange(rows)
        channel = hl.arange(dim)
        query = row // group
        head = kv_head * group + row % group
        slot = (request * kv_heads + kv_head) * 2 + 1
        partial_address = (
            slot * rows * dim
            + (row[:, None] // 4) * (dim * 4)
            + channel[None, :] * 4
            + row[:, None] % 4
        )
        owner = first_owner
        relative_begin = 0
        relative_end = torch.clamp_max(lengths[request] * 0 + cut, page_slots)
        maximum = hl.full([rows], float("-inf"), dtype=torch.float32)
        denominator_lanes = hl.zeros([rows, page_size], dtype=torch.float32)
        accumulator = hl.zeros([rows, dim], dtype=torch.float32)
        seeded = split
        if split:
            maximum = partial_ml[slot, row, 0]
            imported_l = partial_ml[slot, row, 1]
            imported_o = partial_flat[partial_address]
            nonempty = maximum != float("-inf")
            accumulator = torch.where(nonempty[:, None], imported_o, 0.0)
            denominator_lanes = torch.where(
                (hl.arange(page_size)[None, :] == 0) & nonempty[:, None],
                imported_l[:, None],
                0.0,
            )
        length = lengths[request]
        range_lo = torch.clamp_min(length - query_len - window_left, 0)
        first_page = range_lo // page_size
        pa = first_page + relative_begin
        pb = torch.minimum(
            first_page + relative_end, (length + page_size - 1) // page_size
        )
        count = torch.clamp_min(pb - pa, 0)
        key_lo = torch.maximum(range_lo, pa * page_size)
        key_hi = torch.minimum(length, pb * page_size)
        owner_end = (
            (owner + 1) * total_slots // work_buckets + page_slots - 1
        ) // page_slots
        move_oldest = (
            (request == owner_end - 1) & (key_lo > pa * page_size) & (count > 4)
        )
        oldest_position = torch.where(move_oldest, count - 4, count + 64)
        query_position = length - query_len + query
        query_address = (
            request * q_original.stride(0)
            + query[:, None] * q_original.stride(1)
            + head[:, None] * q_original.stride(2)
            + channel[None, :] * q_original.stride(3)
        )
        q = query_storage[query_address]
        for iteration in hl.grid(count):
            logical_page = torch.where(
                oldest_position > iteration,
                pa + count - 1 - iteration,
                torch.where(
                    torch.eq(oldest_position, iteration), pa, pa + count - iteration
                ),
            )
            token_base = logical_page * page_size
            shift = torch.where(
                key_lo > token_base,
                (key_lo - token_base) & -4,
                torch.where(
                    key_hi < token_base + page_size,
                    -((token_base + page_size - key_hi) & -4),
                    0,
                ),
            )
            token_row = hl.arange(page_size) + shift
            in_page = (token_row >= 0) & (token_row < page_size)
            safe_row = torch.clamp(token_row, min=0, max=page_size - 1)
            physical_page = page_table[request, logical_page]
            raw_key = k[physical_page, safe_row[:, None], kv_head, channel[None, :]]
            raw_value = v[physical_page, safe_row[:, None], kv_head, channel[None, :]]
            key_code = k_scale[
                physical_page, safe_row[:, None], kv_head, channel[None, :] // 32
            ]
            value_code = v_scale[
                physical_page, safe_row[:, None], kv_head, channel[None, :] // 32
            ]
            key = torch.where(
                in_page[:, None], native_s2f6_decode(raw_key, key_code), 0.0
            )
            value = torch.where(
                in_page[:, None], native_s2f6_decode(raw_value, value_code), 0.0
            )
            scores = hl.dot(q, key.T)
            key_position = token_base + token_row
            visible = (
                in_page[None, :]
                & (key_position[None, :] >= key_lo)
                & (key_position[None, :] < key_hi)
                & (key_position[None, :] <= query_position[:, None])
                & (key_position[None, :] >= query_position[:, None] - window_left)
            )
            difference = paged_score_difference(scores, maximum, visible, factor)
            probability = torch.exp2(difference)
            # One vote spans all query columns and key lanes belonging to this head.
            if torch.any(difference > 2.0):
                scaled_scores = torch.where(visible, scores * factor, float("-inf"))
                # Native fmax reductions ignore NaNs; visible NaNs remain in P.
                maximum_scores = torch.where(
                    torch.isnan(scaled_scores), float("-inf"), scaled_scores
                )
                next_maximum = torch.fmax(maximum, torch.amax(maximum_scores, dim=1))
                alpha = torch.where(
                    next_maximum == float("-inf"),
                    1.0,
                    torch.exp2(maximum - next_maximum),
                )
                if (iteration > 0) | seeded:
                    accumulator = accumulator * alpha[:, None]
                    denominator_lanes = denominator_lanes * alpha[:, None]
                maximum = next_maximum
                probability = torch.exp2(
                    paged_score_difference(scores, maximum, visible, factor)
                )
            denominator_lanes = denominator_lanes + probability
            accumulator = hl.dot(probability.to(torch.bfloat16), value, acc=accumulator)
        denominator = paged_denominator_sum(denominator_lanes)
        # Explicit public empty-row extension; all other rows keep native arithmetic.
        # With complete page coverage, exactly the negative causal positions
        # have no visible keys. A nonempty row can retain an infinite maximum
        # when its visible scores are NaN; preserve that exceptional arithmetic.
        empty = query_position < 0
        normalized = accumulator * (64 * (1.0 / denominator))[:, None]
        output[request, query[:, None], head[:, None], channel[None, :]] = torch.where(
            empty[:, None], 0.0, normalized
        ).to(torch.bfloat16)
        lse_address = (
            request * lse.stride(0) + query * lse.stride(1) + head * lse.stride(2)
        )
        lse_storage[lse_address] = torch.where(
            empty, float("-inf"), (maximum + torch.log2(denominator)) * math.log(2)
        )
    return output, lse
