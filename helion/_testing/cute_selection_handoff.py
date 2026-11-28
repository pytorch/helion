"""Two selections sharing buffers but with independently observable rankings."""

from __future__ import annotations

import torch

import helion.language as hl


def repeated_selections(
    x: torch.Tensor, other: torch.Tensor, first_largest: hl.constexpr
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    scores = torch.empty((1, 128), dtype=torch.float32, device=x.device)
    values = torch.empty((1, 4, 8), dtype=torch.float32, device=x.device)
    ids = torch.empty((1, 4, 8), dtype=torch.int32, device=x.device)
    flat = values.view(1, 32)
    flat_ids = ids.view(1, 32)
    first_values = torch.empty((1, 8), dtype=torch.float32, device=x.device)
    first_ids = torch.empty((1, 8), dtype=torch.int32, device=x.device)
    last_values = torch.empty_like(first_values)
    last_ids = torch.empty_like(first_ids)
    final = torch.empty_like(first_values)
    for r in hl.tile(128, block_size=128):
        scores[0, r] = x[0, r]
    for row, group in hl.tile([1, 4], block_size=[1, 1]):
        j = group.begin * 32 + hl.arange(32)
        if first_largest:
            q, ix = torch.topk(scores[row.begin, j], 8)
        else:
            q, ix = torch.topk(scores[row.begin, j], 8, largest=False)
        values[row.begin, group.begin, :] = q
        ids[row.begin, group.begin, :] = ix.to(torch.int32) + group.begin * 32
    for row in hl.tile(1, block_size=1):
        q, ix = torch.topk(flat[row.begin, :], 8)
        first_values[row.begin, :] = q
        first_ids[row.begin, :] = flat_ids[row.begin, ix]
    # This phase has either a concrete or a symbolic number of PIDs. Both
    # selection pairs above and below it use the very same global buffers.
    for r in hl.tile(other.size(1), block_size=1):
        scores[0, r.begin] = first_values[0, r.begin % 8] + other[0, r.begin]
    for row, group in hl.tile([1, 4], block_size=[1, 1]):
        j = group.begin * 32 + hl.arange(32)
        q, ix = torch.topk(scores[row.begin, j], 8, largest=not first_largest)
        values[row.begin, group.begin, :] = q
        ids[row.begin, group.begin, :] = ix.to(torch.int32) + group.begin * 32
    for row in hl.tile(1, block_size=1):
        q, ix = torch.topk(flat[row.begin, :], 8)
        last_values[row.begin, :] = q
        last_ids[row.begin, :] = flat_ids[row.begin, ix]
    for r in hl.tile(8, block_size=8):
        final[0, r] = last_values[0, r] + last_ids[0, r].to(torch.float32)
    return final, first_values, first_ids, last_values, last_ids, values, ids
