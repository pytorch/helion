"""Shared device-side building blocks for the DeepSeek NVFP4 kernels."""

from __future__ import annotations

import torch

import helion.language as hl

FP4_MAX = 6.0
MMA_N = 16


def fp4_nibble(value: torch.Tensor) -> torch.Tensor:
    """Round scaled values to E2M1 with FlashInfer's tie convention."""
    magnitude = torch.abs(value)
    code = (magnitude > 0.25).to(torch.int32)
    # Keep these additions functional: aten.add_ returns an aliased InputBuffer,
    # which the shared Inductor-to-Helion lowering cannot consume as pointwise IR.
    code = code + (magnitude >= 0.75).to(torch.int32)
    code = code + (magnitude > 1.25).to(torch.int32)
    code = code + (magnitude >= 1.75).to(torch.int32)
    code = code + (magnitude > 2.5).to(torch.int32)
    code = code + (magnitude >= 3.5).to(torch.int32)
    code = code + (magnitude > 5.0).to(torch.int32)
    return code | ((value < 0).to(torch.int32) << 3)


def first_mma_column(value: torch.Tensor) -> torch.Tensor:
    """Select the real token column from the padded native-MMA N tile."""
    column = hl.arange(MMA_N)
    return torch.sum(value * (column[None, :] == 0).to(torch.float32), dim=-1)


def stable_key(values: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
    """Pack float32 values and their indices into int64 keys ordered by value, then lower index."""
    bits = values.view(torch.int32).to(torch.int64)
    ordered = torch.where(bits < 0, -2147483648 - bits, bits)
    return ordered * count + (count - 1 - index.to(torch.int64))


def routing_keys(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    num_experts: int,
    num_groups: int,
    topk_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped DeepSeek-V3 routing of BF16 ``logits`` [experts]: (sigmoid scores,
    pick keys, expert indices).  The picks are the largest keys in order; experts
    outside the chosen groups key at -inf."""
    experts_per_group = num_experts // num_groups
    scores = torch.sigmoid(logits.to(torch.float32))
    grouped = (scores + correction_bias).view(num_groups, experts_per_group)
    within = hl.arange(experts_per_group)[None, :].to(torch.int32)
    best = torch.amax(grouped, dim=-1)
    best_id = torch.amin(
        torch.where(grouped == best[:, None], within, experts_per_group), dim=-1
    )
    second = torch.amax(
        torch.where(within == best_id[:, None], float("-inf"), grouped), dim=-1
    )
    group_scores = best + second
    group_indices = hl.arange(num_groups).to(torch.int32)
    # Stable group rank: a higher score first, ties to the lower group index.
    group_beats = (group_scores[None, :] > group_scores[:, None]) | (
        (group_scores[None, :] == group_scores[:, None])
        & (group_indices[None, :] < group_indices[:, None])
    )
    allowed = torch.sum(group_beats.to(torch.float32), dim=1) < topk_groups
    masked = torch.where(allowed[:, None], grouped, float("-inf")).view(num_experts)
    expert_indices = hl.arange(num_experts).to(torch.int32)
    # One max per pick: the key orders by value, then by the lower expert index.
    return scores, stable_key(masked, expert_indices, num_experts), expert_indices


def routed_slots(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    num_experts: int,
    num_groups: int,
    topk_groups: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(sigmoid scores, each expert's pick index or -1, expert indices)."""
    scores, keys, expert_indices = routing_keys(
        logits, correction_bias, num_experts, num_groups, topk_groups
    )
    slot = hl.full([num_experts], -1, dtype=torch.int32)
    for pick in hl.static_range(top_k):
        hit = keys == torch.amax(keys)
        slot = torch.where(hit, pick, slot)
        keys = torch.where(hit, -9223372036854775807, keys)
    return scores, slot, expert_indices
