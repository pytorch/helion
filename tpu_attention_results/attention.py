"""Dense scaled-dot-product attention written in Helion."""

from __future__ import annotations

import math

import torch

import helion.language as hl


def attention_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Compute dense attention with an online softmax over streamed K/V tiles."""
    query_rows = query.size(-2)
    kv_rows = key.size(-2)
    assert kv_rows == value.size(-2)
    head_dim = hl.specialize(query.size(-1))
    assert head_dim == key.size(-1) == value.size(-1)

    query_view = query.reshape([-1, query_rows, head_dim])
    key_view = key.reshape([-1, kv_rows, head_dim])
    value_view = value.reshape([-1, kv_rows, head_dim])
    output = torch.empty_like(query_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504

    for tile_batch_head, tile_query in hl.tile([query_view.size(0), query_rows]):
        row_max = hl.full(
            [tile_batch_head, tile_query],
            float("-inf"),
            dtype=torch.float32,
        )
        row_sum = torch.full_like(row_max, 1.0)
        accumulator = hl.zeros(
            [tile_batch_head, tile_query, head_dim],
            dtype=torch.float32,
        )
        query_tile = query_view[tile_batch_head, tile_query, :]

        for tile_kv in hl.tile(kv_rows):
            scaled_query = query_tile * qk_scale
            key_tile = key_view[tile_batch_head, tile_kv, :]
            scores = torch.bmm(
                scaled_query,
                key_tile.transpose(1, 2),
                torch.float32,
            )
            next_max = torch.maximum(row_max, torch.amax(scores, dim=-1))
            probabilities = torch.exp2(scores - next_max[:, :, None])
            correction = torch.exp2(row_max - next_max)
            row_sum = row_sum * correction + torch.sum(probabilities, dim=-1)
            accumulator = accumulator * correction[:, :, None]
            value_tile = value_view[tile_batch_head, tile_kv, :]
            accumulator = torch.baddbmm(
                accumulator,
                probabilities.to(value_tile.dtype),
                value_tile,
            )
            row_max = next_max

        output[tile_batch_head, tile_query, :] = (accumulator / row_sum[:, :, None]).to(
            output.dtype
        )
    return output.view(query.size())
