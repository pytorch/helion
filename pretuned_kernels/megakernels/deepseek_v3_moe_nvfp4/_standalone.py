# ruff: noqa: ANN001, ANN202
"""Matched, independently tuned Helion kernels for DeepSeek-V3 NVFP4 MoE.

The nine launches implement exactly the persistent kernel's logical roots and
use PDL within each dependency chain. Independent router/input-quant work and
routed/shared expert branches use CUDA streams so the baseline is not
artificially serialized.
"""

from __future__ import annotations

from typing import Any

from pretuned_kernels.megakernels._pdl import launch_dependent
from pretuned_kernels.megakernels._pdl import wait_and_launch_dependents
import torch

import helion
import helion.language as hl

FP4_MAX = 6.0
MMA_N = 16


def _fp4_nibble(value: torch.Tensor) -> torch.Tensor:
    magnitude = torch.abs(value)
    code = (magnitude > 0.25).to(torch.int32)
    code += (magnitude >= 0.75).to(torch.int32)
    code += (magnitude > 1.25).to(torch.int32)
    code += (magnitude >= 1.75).to(torch.int32)
    code += (magnitude > 2.5).to(torch.int32)
    code += (magnitude >= 3.5).to(torch.int32)
    code += (magnitude > 5.0).to(torch.int32)
    return code | ((value < 0).to(torch.int32) << 3)


def _first_mma_column(value: torch.Tensor) -> torch.Tensor:
    column = hl.arange(MMA_N)
    return torch.sum(value * (column[None, :] == 0).to(torch.float32), dim=-1)


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def standalone_router(
    hidden: torch.Tensor, router_weight: torch.Tensor
) -> torch.Tensor:
    rows, hidden_size = hidden.shape
    experts, weight_hidden = router_weight.shape
    assert hidden_size == weight_hidden
    logits = torch.empty((rows, experts), dtype=torch.bfloat16, device=hidden.device)
    expert_block = hl.register_block_size(2, 32)
    reduction_block = hl.register_block_size(256, 1024)
    for tile_row, tile_expert in hl.tile([rows, experts], block_size=[1, expert_block]):
        accumulator = hl.zeros([tile_row, tile_expert], dtype=torch.float32)
        for tile_k in hl.tile(hidden_size, block_size=reduction_block):
            accumulator = torch.addmm(
                accumulator,
                hidden[tile_row, tile_k],
                router_weight[tile_expert, tile_k].T,
            )
        logits[tile_row, tile_expert] = accumulator.to(torch.bfloat16)
    return logits


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def standalone_input_quant(
    hidden: torch.Tensor, input_global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden_size = hidden.shape
    groups = hidden_size // 16
    hidden_q = torch.empty(
        (rows, hidden_size // 2), dtype=torch.uint8, device=hidden.device
    )
    hidden_q_groups = hidden_q.view(rows, groups, 8)
    hidden_scale = torch.empty(
        (rows, groups), dtype=torch.float8_e4m3fn, device=hidden.device
    )
    group_block = hl.register_block_size(1, 32)
    for tile_row, tile_group in hl.tile([rows, groups], block_size=[1, group_block]):
        packed_lane = hl.arange(8)[None, :]
        group_offsets = tile_group.index[:, None] * 16
        low_values = hidden[tile_row, group_offsets + packed_lane * 2].to(torch.float32)
        high_values = hidden[tile_row, group_offsets + packed_lane * 2 + 1].to(
            torch.float32
        )
        global_scale = torch.sum(input_global_scale[:])
        block_scale_f32 = (
            torch.maximum(
                torch.amax(torch.abs(low_values), dim=-1),
                torch.amax(torch.abs(high_values), dim=-1),
            )
            * global_scale
            / FP4_MAX
        )
        block_scale = block_scale_f32.to(torch.float8_e4m3fn)
        actual_scale = block_scale.to(torch.float32)
        divisor = torch.where(actual_scale > 0, actual_scale, 1.0)
        low = _fp4_nibble(low_values * global_scale / divisor[:, :, None])
        high = _fp4_nibble(high_values * global_scale / divisor[:, :, None])
        packed = (low | (high << 4)).to(torch.uint8)
        hidden_q_groups[tile_row, tile_group, :] = packed
        hidden_scale[tile_row, tile_group] = block_scale
    return hidden_q, hidden_scale


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def grouped_topk(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    num_groups: int,
    topk_groups: int,
    routed_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_experts = logits.size()
    top_k = hl.specialize(top_k)
    num_groups = hl.specialize(num_groups)
    topk_groups = hl.specialize(topk_groups)
    assert top_k == 8 and topk_groups == 4
    experts_per_group = num_experts // num_groups
    hl.specialize(batch)
    hl.specialize(num_experts)
    weights = torch.empty((batch, top_k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((batch, top_k), dtype=torch.int32, device=logits.device)
    for _program in hl.grid(1):
        wait_and_launch_dependents()
        # The production DeepSeek router promotes BF16 logits before applying
        # sigmoid and performs group selection in FP32.
        scores = torch.sigmoid(logits[:, :].to(torch.float32))
        biased = scores + correction_bias[None, :]
        grouped = biased.view(batch, num_groups, experts_per_group)
        negative_infinity = torch.full_like(grouped, float("-inf"))
        within_group_indices = hl.arange(experts_per_group)[None, None, :].to(
            torch.int32
        )
        group_best_1 = torch.amax(grouped, dim=-1, keepdim=True)
        group_best_id_1 = torch.amin(
            torch.where(
                grouped == group_best_1,
                within_group_indices,
                torch.full_like(within_group_indices, experts_per_group),
            ),
            dim=-1,
        )
        group_without_best = torch.where(
            within_group_indices == group_best_id_1[:, :, None],
            negative_infinity,
            grouped,
        )
        group_best_2 = torch.amax(group_without_best, dim=-1)
        group_scores = group_best_1.view(batch, num_groups) + group_best_2
        group_indices = hl.arange(num_groups)[None, :].to(torch.int32)
        group_max_1 = torch.amax(group_scores, dim=-1, keepdim=True)
        group_id_1 = torch.amin(
            torch.where(
                group_scores == group_max_1,
                group_indices,
                torch.full_like(group_indices, num_groups),
            ),
            dim=-1,
        )
        remaining_groups = torch.where(
            group_indices == group_id_1[:, None],
            torch.full_like(group_scores, float("-inf")),
            group_scores,
        )
        group_max_2 = torch.amax(remaining_groups, dim=-1, keepdim=True)
        group_id_2 = torch.amin(
            torch.where(
                remaining_groups == group_max_2,
                group_indices,
                torch.full_like(group_indices, num_groups),
            ),
            dim=-1,
        )
        remaining_groups = torch.where(
            group_indices == group_id_2[:, None],
            torch.full_like(group_scores, float("-inf")),
            remaining_groups,
        )
        group_max_3 = torch.amax(remaining_groups, dim=-1, keepdim=True)
        group_id_3 = torch.amin(
            torch.where(
                remaining_groups == group_max_3,
                group_indices,
                torch.full_like(group_indices, num_groups),
            ),
            dim=-1,
        )
        remaining_groups = torch.where(
            group_indices == group_id_3[:, None],
            torch.full_like(group_scores, float("-inf")),
            remaining_groups,
        )
        group_max_4 = torch.amax(remaining_groups, dim=-1, keepdim=True)
        group_id_4 = torch.amin(
            torch.where(
                remaining_groups == group_max_4,
                group_indices,
                torch.full_like(group_indices, num_groups),
            ),
            dim=-1,
        )
        allowed = (
            (group_indices == group_id_1[:, None])
            | (group_indices == group_id_2[:, None])
            | (group_indices == group_id_3[:, None])
            | (group_indices == group_id_4[:, None])
        ).view(batch, num_groups, 1)
        masked = torch.where(allowed, grouped, negative_infinity).view(
            batch, num_experts
        )
        expert_indices = hl.arange(num_experts)[None, :].to(torch.int32)
        masked_1 = masked
        value_0 = torch.amax(masked_1, dim=-1, keepdim=True)
        candidates_0 = masked_1 == value_0
        id_0 = torch.amin(
            torch.where(
                candidates_0,
                expert_indices,
                torch.full_like(expert_indices, num_experts),
            ),
            dim=-1,
        )
        selected_0 = expert_indices == id_0[:, None]
        masked_2 = torch.where(
            selected_0, torch.full_like(masked, float("-inf")), masked_1
        )
        value_1 = torch.amax(masked_2, dim=-1, keepdim=True)
        candidates_1 = masked_2 == value_1
        id_1 = torch.amin(
            torch.where(
                candidates_1,
                expert_indices,
                torch.full_like(expert_indices, num_experts),
            ),
            dim=-1,
        )
        selected_1 = expert_indices == id_1[:, None]
        masked_3 = torch.where(
            selected_1, torch.full_like(masked, float("-inf")), masked_2
        )
        value_2 = torch.amax(masked_3, dim=-1, keepdim=True)
        candidates_2 = masked_3 == value_2
        id_2 = torch.amin(
            torch.where(
                candidates_2,
                expert_indices,
                torch.full_like(expert_indices, num_experts),
            ),
            dim=-1,
        )
        selected_2 = expert_indices == id_2[:, None]
        masked_4 = torch.where(
            selected_2, torch.full_like(masked, float("-inf")), masked_3
        )
        value_3 = torch.amax(masked_4, dim=-1, keepdim=True)
        candidates_3 = masked_4 == value_3
        id_3 = torch.amin(
            torch.where(
                candidates_3,
                expert_indices,
                torch.full_like(expert_indices, num_experts),
            ),
            dim=-1,
        )
        selected_3 = expert_indices == id_3[:, None]
        masked_5 = torch.where(
            selected_3, torch.full_like(masked, float("-inf")), masked_4
        )
        value_4 = torch.amax(masked_5, dim=-1, keepdim=True)
        candidates_4 = masked_5 == value_4
        id_4 = torch.amin(
            torch.where(
                candidates_4,
                expert_indices,
                torch.full_like(expert_indices, num_experts),
            ),
            dim=-1,
        )
        selected_4 = expert_indices == id_4[:, None]
        masked_6 = torch.where(
            selected_4, torch.full_like(masked, float("-inf")), masked_5
        )
        value_5 = torch.amax(masked_6, dim=-1, keepdim=True)
        candidates_5 = masked_6 == value_5
        id_5 = torch.amin(
            torch.where(
                candidates_5,
                expert_indices,
                torch.full_like(expert_indices, num_experts),
            ),
            dim=-1,
        )
        selected_5 = expert_indices == id_5[:, None]
        masked_7 = torch.where(
            selected_5, torch.full_like(masked, float("-inf")), masked_6
        )
        value_6 = torch.amax(masked_7, dim=-1, keepdim=True)
        candidates_6 = masked_7 == value_6
        id_6 = torch.amin(
            torch.where(
                candidates_6,
                expert_indices,
                torch.full_like(expert_indices, num_experts),
            ),
            dim=-1,
        )
        selected_6 = expert_indices == id_6[:, None]
        masked_8 = torch.where(
            selected_6, torch.full_like(masked, float("-inf")), masked_7
        )
        value_7 = torch.amax(masked_8, dim=-1, keepdim=True)
        candidates_7 = masked_8 == value_7
        id_7 = torch.amin(
            torch.where(
                candidates_7,
                expert_indices,
                torch.full_like(expert_indices, num_experts),
            ),
            dim=-1,
        )
        # The selected IDs are unique, so load their scores directly instead of
        # running eight full-width one-hot reductions over all experts.
        weight_0 = torch.sigmoid(torch.sum(logits[:, id_0].float(), dim=-1))
        weight_1 = torch.sigmoid(torch.sum(logits[:, id_1].float(), dim=-1))
        weight_2 = torch.sigmoid(torch.sum(logits[:, id_2].float(), dim=-1))
        weight_3 = torch.sigmoid(torch.sum(logits[:, id_3].float(), dim=-1))
        weight_4 = torch.sigmoid(torch.sum(logits[:, id_4].float(), dim=-1))
        weight_5 = torch.sigmoid(torch.sum(logits[:, id_5].float(), dim=-1))
        weight_6 = torch.sigmoid(torch.sum(logits[:, id_6].float(), dim=-1))
        weight_7 = torch.sigmoid(torch.sum(logits[:, id_7].float(), dim=-1))
        denominator = (
            weight_0
            + weight_1
            + weight_2
            + weight_3
            + weight_4
            + weight_5
            + weight_6
            + weight_7
        )
        weights[:, 0] = weight_0 / denominator * routed_scale
        weights[:, 1] = weight_1 / denominator * routed_scale
        weights[:, 2] = weight_2 / denominator * routed_scale
        weights[:, 3] = weight_3 / denominator * routed_scale
        weights[:, 4] = weight_4 / denominator * routed_scale
        weights[:, 5] = weight_5 / denominator * routed_scale
        weights[:, 6] = weight_6 / denominator * routed_scale
        weights[:, 7] = weight_7 / denominator * routed_scale
        ids[:, 0] = id_0
        ids[:, 1] = id_1
        ids[:, 2] = id_2
        ids[:, 3] = id_3
        ids[:, 4] = id_4
        ids[:, 5] = id_5
        ids[:, 6] = id_6
        ids[:, 7] = id_7
    return weights, ids


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def selected_w13_swiglu_nvfp4(
    hidden_q: torch.Tensor,
    hidden_scale_bytes: torch.Tensor,
    w13: torch.Tensor,
    w13_scale_bytes: torch.Tensor,
    topk_ids: torch.Tensor,
    alpha1: torch.Tensor,
    activation_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selected-expert W4A4 GEMM1 with fused SwiGLU and NVFP4 requant."""
    tokens, packed_hidden = hidden_q.size()
    experts, twice_intermediate, weight_packed_hidden = w13.size()
    top_k = topk_ids.size(1)
    intermediate = twice_intermediate // 2
    activation_groups = intermediate // 16
    assert tokens == 1 and packed_hidden == weight_packed_hidden
    hl.specialize(experts)
    hl.specialize(top_k)
    hl.specialize(intermediate)
    output_q = torch.empty(
        (top_k, intermediate // 2), dtype=torch.uint8, device=hidden_q.device
    )
    output_scale = torch.empty(
        (top_k, activation_groups),
        dtype=torch.float8_e4m3fn,
        device=hidden_q.device,
    )
    output_q_groups = output_q.view(top_k, activation_groups, 8)
    hidden_groups = packed_hidden // 8
    weight_tma = w13.view(experts * (twice_intermediate // 128), 128, packed_hidden)
    flat_weight_scale = w13_scale_bytes.view(
        experts * twice_intermediate, hidden_groups
    )
    output_groups = hl.register_block_size(4, 4)
    block_groups = hl.register_block_size(32, 32)
    for tile_slot, tile_output_group in hl.tile(
        [top_k, activation_groups], block_size=[1, output_groups]
    ):
        wait_and_launch_dependents()
        slot = tile_slot.begin
        expert = topk_ids[0, slot]
        expert_address = expert.to(torch.int64)
        descriptor_row = (
            expert * (twice_intermediate // 128) + tile_output_group.begin // 4
        )
        physical_row_index = (tile_output_group.begin * 32 + hl.arange(128)).to(
            torch.int64
        )
        weight_row = expert_address * twice_intermediate + physical_row_index
        accumulator = hl.zeros([128, MMA_N], dtype=torch.float32)
        for tile_group in hl.tile(hidden_groups, block_size=block_groups):
            packed_index = (tile_group.begin * 8 + hl.arange(block_groups * 8)).to(
                torch.int64
            )
            lhs = weight_tma[descriptor_row, :, packed_index]
            lhs_scale = flat_weight_scale[
                weight_row[:, None], tile_group.index[None, :]
            ]
            hidden_bytes = hidden_q[0, packed_index]
            rhs = hidden_bytes[:, None].expand(hidden_bytes.size(0), MMA_N)
            hidden_scale = hidden_scale_bytes[0, tile_group]
            rhs_scale = hidden_scale[None, :].expand(MMA_N, block_groups)
            accumulator = hl.dot_scaled(
                lhs,
                lhs_scale,
                "e2m1",
                rhs,
                rhs_scale,
                "e2m1",
                acc=accumulator,
                out_dtype=torch.float32,
            )
        logical_rows = 64
        preactivation = _first_mma_column(accumulator).reshape(logical_rows, 2)
        pair = hl.arange(2)
        gate = torch.sum(preactivation * (pair[None, :] == 0), dim=-1)
        up = torch.sum(preactivation * (pair[None, :] == 1), dim=-1)
        first_scale = alpha1[expert]
        gate = gate * first_scale
        up = up * first_scale
        activated = gate * torch.sigmoid(gate) * up
        activated_groups = activated.reshape(logical_rows // 16, 16)
        global_scale = torch.sum(activation_global_scale[:])
        block_scale_f32 = (
            torch.amax(torch.abs(activated_groups), dim=-1) * global_scale / FP4_MAX
        )
        block_scale = block_scale_f32.to(torch.float8_e4m3fn)
        actual_scale = block_scale.to(torch.float32)
        divisor = torch.where(actual_scale > 0, actual_scale, 1.0)
        scaled = activated_groups * global_scale / divisor[:, None]
        nibbles = _fp4_nibble(scaled)
        low, high = hl.split(nibbles.reshape(logical_rows // 16, 8, 2))
        packed = low | (high << 4)
        output_q_groups[slot, tile_output_group, :] = packed.reshape(4, 8).to(
            torch.uint8
        )
        output_scale[slot, tile_output_group] = block_scale
    return output_q, output_scale


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def selected_w2_nvfp4(
    activation_q: torch.Tensor,
    activation_scale_bytes: torch.Tensor,
    w2: torch.Tensor,
    w2_scale_bytes: torch.Tensor,
    topk_ids: torch.Tensor,
    alpha2: torch.Tensor,
) -> torch.Tensor:
    """Selected-expert W4A4 GEMM2 with FP32 accumulation."""
    top_k, packed_intermediate = activation_q.size()
    experts, hidden, weight_packed_intermediate = w2.size()
    assert packed_intermediate == weight_packed_intermediate
    hl.specialize(experts)
    hl.specialize(top_k)
    hl.specialize(hidden)
    output = torch.empty((top_k, hidden), dtype=torch.bfloat16, device=w2.device)
    groups = packed_intermediate // 8
    flat_weight_scale = w2_scale_bytes.view(experts * hidden, groups)
    block_rows = hl.register_block_size(128, 256)
    block_groups = hl.register_block_size(32, 32)
    for tile_slot, tile_row in hl.tile([top_k, hidden], block_size=[1, block_rows]):
        wait_and_launch_dependents()
        slot = tile_slot.begin
        expert = topk_ids[0, slot]
        expert_address = expert.to(torch.int64)
        row_index = tile_row.index.to(torch.int32)
        weight_row = expert_address * hidden + row_index
        accumulator = hl.zeros([tile_row, MMA_N], dtype=torch.float32)
        for tile_group in hl.tile(groups, block_size=block_groups):
            packed_index = (tile_group.begin * 8 + hl.arange(block_groups * 8)).to(
                torch.int64
            )
            lhs = w2[expert, tile_row, packed_index]
            lhs_scale = flat_weight_scale[
                weight_row[:, None], tile_group.index[None, :]
            ]
            activation_bytes = activation_q[slot, packed_index]
            rhs = activation_bytes[:, None].expand(activation_bytes.size(0), MMA_N)
            activation_scale = activation_scale_bytes[slot, tile_group]
            rhs_scale = activation_scale[None, :].expand(MMA_N, block_groups)
            accumulator = hl.dot_scaled(
                lhs,
                lhs_scale,
                "e2m1",
                rhs,
                rhs_scale,
                "e2m1",
                acc=accumulator,
                out_dtype=torch.float32,
            )
        output[tile_slot, tile_row] = (
            (_first_mma_column(accumulator) * alpha2[expert])
            .reshape(1, tile_row.block_size)
            .to(torch.bfloat16)
        )
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def shared_w13_nvfp4(
    hidden_q: torch.Tensor,
    hidden_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha1: torch.Tensor,
) -> torch.Tensor:
    tokens, packed_hidden = hidden_q.shape
    twice_intermediate, weight_packed_hidden = weight.shape
    hidden_groups = packed_hidden // 8
    assert tokens == 1 and packed_hidden == weight_packed_hidden
    hl.specialize(twice_intermediate)
    output = torch.empty(
        (twice_intermediate,), dtype=torch.bfloat16, device=weight.device
    )
    flat_weight = weight.view(twice_intermediate, packed_hidden)
    flat_scale = weight_scale.view(twice_intermediate, hidden_groups)
    flat_hidden = hidden_q.view(packed_hidden)
    row_block = hl.register_block_size(16, 16)
    group_block = hl.register_block_size(64, 64)
    for tile_row in hl.tile(twice_intermediate, block_size=row_block):
        weight_row = tile_row.index.to(torch.int64)
        accumulator = hl.zeros([row_block], dtype=torch.float32)
        for tile_group in hl.tile(hidden_groups, block_size=group_block):
            group_mask = tile_group.index < hidden_groups
            offsets = weight_row[:, None] * hidden_groups + tile_group.index[None, :]
            mask = (tile_row.index[:, None] < twice_intermediate) & group_mask[None, :]
            values = hl.load_float4_e2m1fn_x16_to_float16(
                flat_weight, offsets, extra_mask=mask
            )
            hidden_values = hl.load_float4_e2m1fn_x16_to_float16(
                flat_hidden, tile_group.index, extra_mask=group_mask
            )
            contribution = hl.zeros([row_block, group_block], dtype=torch.float32)
            for lane in hl.static_range(16):
                contribution += values[lane] * hidden_values[lane][None, :]
            scale = flat_scale[weight_row[:, None], tile_group.index[None, :]].to(
                torch.float32
            )
            input_scale = hidden_scale[0, tile_group].to(torch.float32)
            accumulator += (
                contribution.to(torch.float32) * scale * input_scale[None, :]
            ).sum(dim=-1)
        output[tile_row] = (accumulator * torch.sum(alpha1[:])).to(torch.bfloat16)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def shared_swiglu_quant_nvfp4(
    preactivation: torch.Tensor,
    activation_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    twice_intermediate = preactivation.size(0)
    intermediate = twice_intermediate // 2
    groups = intermediate // 16
    output_q = torch.empty(
        (1, intermediate // 2), dtype=torch.uint8, device=preactivation.device
    )
    output_scale = torch.empty(
        (1, groups), dtype=torch.float8_e4m3fn, device=preactivation.device
    )
    output_q_groups = output_q.view(1, groups, 8)
    group_block = hl.register_block_size(32, 32)
    for tile_group in hl.tile(groups, block_size=group_block):
        wait_and_launch_dependents()
        index = tile_group.index[:, None] * 16 + hl.arange(16)[None, :]
        gate = preactivation[index * 2].to(torch.float32)
        up = preactivation[index * 2 + 1].to(torch.float32)
        # Preserve the BF16 W13-to-SwiGLU boundary used by production.
        activated = (
            (gate * torch.sigmoid(gate) * up).to(torch.bfloat16).to(torch.float32)
        )
        global_scale = torch.sum(activation_global_scale[:])
        scale_f32 = torch.amax(torch.abs(activated), dim=-1) * global_scale / FP4_MAX
        scale = scale_f32.to(torch.float8_e4m3fn)
        actual_scale = scale.to(torch.float32)
        divisor = torch.where(actual_scale > 0, actual_scale, 1.0)
        nibbles = _fp4_nibble(activated * global_scale / divisor[:, None])
        low, high = hl.split(nibbles.reshape(group_block, 8, 2))
        output_q_groups[0, tile_group, :] = (
            (low | (high << 4)).reshape(group_block, 8).to(torch.uint8)
        )
        output_scale[0, tile_group] = scale
    return output_q, output_scale


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def shared_w2_nvfp4(
    activation_q: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha2: torch.Tensor,
) -> torch.Tensor:
    tokens, packed_intermediate = activation_q.shape
    hidden, weight_packed_intermediate = weight.shape
    groups = packed_intermediate // 8
    assert tokens == 1 and packed_intermediate == weight_packed_intermediate
    hl.specialize(hidden)
    output = torch.empty((1, hidden), dtype=torch.bfloat16, device=weight.device)
    flat_weight = weight.view(hidden, packed_intermediate)
    flat_scale = weight_scale.view(hidden, groups)
    flat_activation = activation_q.view(packed_intermediate)
    row_block = hl.register_block_size(16, 16)
    group_block = hl.register_block_size(64, 64)
    for tile_row in hl.tile(hidden, block_size=row_block):
        wait_and_launch_dependents()
        weight_row = tile_row.index.to(torch.int64)
        accumulator = hl.zeros([row_block], dtype=torch.float32)
        for tile_group in hl.tile(groups, block_size=group_block):
            group_mask = tile_group.index < groups
            offsets = weight_row[:, None] * groups + tile_group.index[None, :]
            mask = (tile_row.index[:, None] < hidden) & group_mask[None, :]
            values = hl.load_float4_e2m1fn_x16_to_float16(
                flat_weight, offsets, extra_mask=mask
            )
            activation_values = hl.load_float4_e2m1fn_x16_to_float16(
                flat_activation, tile_group.index, extra_mask=group_mask
            )
            contribution = hl.zeros([row_block, group_block], dtype=torch.float32)
            for lane in hl.static_range(16):
                contribution += values[lane] * activation_values[lane][None, :]
            scale = flat_scale[weight_row[:, None], tile_group.index[None, :]].to(
                torch.float32
            )
            input_scale = activation_scale[0, tile_group].to(torch.float32)
            accumulator += (
                contribution.to(torch.float32) * scale * input_scale[None, :]
            ).sum(dim=-1)
        output[0, tile_row] = (accumulator * torch.sum(alpha2[:])).to(torch.bfloat16)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def weighted_add(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_output: torch.Tensor,
) -> torch.Tensor:
    top_k, hidden = expert_output.shape
    hl.specialize(top_k)
    output = torch.empty((1, hidden), dtype=torch.bfloat16, device=expert_output.device)
    for tile_n in hl.tile(hidden):
        wait_and_launch_dependents()
        values = expert_output[:, tile_n].to(torch.float32)
        weights = topk_weights[0, :].to(torch.bfloat16).to(torch.float32)
        routed = torch.sum(values * weights[:, None], dim=0, keepdim=True).to(
            torch.bfloat16
        )
        output[:, tile_n] = (routed + shared_output[:, tile_n]).to(torch.bfloat16)
    return output


CONFIGS: dict[str, dict[str, Any]] = {
    "router": {
        "block_sizes": [2, 256],
        "range_num_stages": [0, 0],
        "num_warps": 1,
        "num_stages": 6,
    },
    "input_quant": {
        "block_sizes": [4],
        "range_num_stages": [0],
        "num_warps": 4,
        "num_stages": 1,
    },
    "topk": {
        "block_sizes": [],
        "pid_type": "persistent_blocked",
        "num_sm_multiplier": 2,
        "maxnreg": 256,
        "num_stages": 1,
        "num_warps": 1,
        "range_unroll_factors": [1],
        "range_warp_specializes": [False],
        "range_num_stages": [1],
        "range_multi_buffers": [True],
    },
    "routed_w13": {
        "block_sizes": [4, 32],
        "loop_orders": [[0, 1]],
        "l2_groupings": [1],
        "range_unroll_factors": [0, 1],
        "range_num_stages": [0, 3],
        "range_multi_buffers": [None, True],
        "range_flattens": [None, False],
        "num_stages": 1,
        "num_warps": 4,
    },
    "routed_w2": {
        "block_sizes": [128, 32],
        "loop_orders": [[0, 1]],
        "l2_groupings": [1],
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
        "range_multi_buffers": [None, None],
        "range_flattens": [None, False],
        "num_stages": 2,
        "num_warps": 4,
    },
    "shared_w13": {
        "block_sizes": [16, 64],
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 3],
        "range_multi_buffers": [None, True],
        "num_stages": 1,
        "num_warps": 8,
    },
    "shared_activation": {
        "block_sizes": [32],
        "range_num_stages": [0],
        "num_stages": 1,
        "num_warps": 4,
    },
    "shared_w2": {
        "block_sizes": [16, 64],
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
        "num_stages": 1,
        "num_warps": 4,
    },
    "final": {
        "block_sizes": [32],
        "range_num_stages": [0],
        "num_stages": 1,
        "num_warps": 4,
    },
}


def _compile(name: str, kernel, args: tuple):
    bound = kernel.bind(args)
    values = dict(bound.config_spec.default_config())
    values.update(CONFIGS[name])
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    compiled = bound.compile_config(config)
    compiled(*args)
    torch.cuda.synchronize()
    return compiled


def build(tensors: dict[str, torch.Tensor], shape):
    """Build the matched nine-launch PDL graph with independently tuned roots."""
    hidden = tensors["hidden"]
    router = _compile("router", standalone_router, (hidden, tensors["router_weight"]))
    logits = router(hidden, tensors["router_weight"])
    input_quant = _compile(
        "input_quant",
        standalone_input_quant,
        (hidden, tensors["input_global_scale"]),
    )
    hidden_q, hidden_scale = input_quant(hidden, tensors["input_global_scale"])
    topk_args = (
        logits,
        tensors["correction_bias"],
        shape.top_k,
        shape.num_groups,
        shape.topk_groups,
        shape.routed_scale,
    )
    topk = _compile("topk", grouped_topk, topk_args)
    weights, ids = launch_dependent(topk, *topk_args)

    routed_w13_args = (
        hidden_q,
        hidden_scale,
        tensors["native_w13"],
        tensors["native_w13_scale"],
        ids,
        tensors["alpha1"],
        tensors["activation_global_scale"],
    )
    routed_w13 = _compile("routed_w13", selected_w13_swiglu_nvfp4, routed_w13_args)
    activation_q, activation_scale = launch_dependent(routed_w13, *routed_w13_args)
    routed_w2_args = (
        activation_q,
        activation_scale,
        tensors["w2"],
        tensors["w2_scale"],
        ids,
        tensors["alpha2"],
    )
    routed_w2 = _compile("routed_w2", selected_w2_nvfp4, routed_w2_args)
    expert_output = launch_dependent(routed_w2, *routed_w2_args)

    shared_w13_args = (
        hidden_q,
        hidden_scale,
        tensors["native_shared_w13"],
        tensors["native_shared_w13_scale"],
        tensors["shared_alpha1"],
    )
    shared_w13 = _compile("shared_w13", shared_w13_nvfp4, shared_w13_args)
    shared_preactivation = shared_w13(*shared_w13_args)
    shared_activation = _compile(
        "shared_activation",
        shared_swiglu_quant_nvfp4,
        (shared_preactivation, tensors["activation_global_scale"]),
    )
    shared_activation_q, shared_activation_scale = launch_dependent(
        shared_activation,
        shared_preactivation,
        tensors["activation_global_scale"],
    )
    shared_w2_args = (
        shared_activation_q,
        shared_activation_scale,
        tensors["shared_w2"],
        tensors["shared_w2_scale"],
        tensors["shared_alpha2"],
    )
    shared_w2 = _compile("shared_w2", shared_w2_nvfp4, shared_w2_args)
    shared_output = launch_dependent(shared_w2, *shared_w2_args)
    final = _compile("final", weighted_add, (expert_output, weights, shared_output))

    quant_stream = torch.cuda.Stream()
    shared_stream = torch.cuda.Stream()

    def launch():
        current_stream = torch.cuda.current_stream()
        quant_stream.wait_stream(current_stream)
        with torch.cuda.stream(quant_stream):
            local_hidden_q, local_hidden_scale = input_quant(
                hidden, tensors["input_global_scale"]
            )

        local_logits = router(hidden, tensors["router_weight"])
        local_weights, local_ids = launch_dependent(
            topk,
            local_logits,
            tensors["correction_bias"],
            shape.top_k,
            shape.num_groups,
            shape.topk_groups,
            shape.routed_scale,
        )

        shared_stream.wait_stream(quant_stream)
        with torch.cuda.stream(shared_stream):
            local_shared_preactivation = shared_w13(
                local_hidden_q,
                local_hidden_scale,
                tensors["native_shared_w13"],
                tensors["native_shared_w13_scale"],
                tensors["shared_alpha1"],
            )
            (
                local_shared_activation_q,
                local_shared_activation_scale,
            ) = launch_dependent(
                shared_activation,
                local_shared_preactivation,
                tensors["activation_global_scale"],
            )
            local_shared_output = launch_dependent(
                shared_w2,
                local_shared_activation_q,
                local_shared_activation_scale,
                tensors["shared_w2"],
                tensors["shared_w2_scale"],
                tensors["shared_alpha2"],
            )

        current_stream.wait_stream(quant_stream)
        local_activation_q, local_activation_scale = launch_dependent(
            routed_w13,
            local_hidden_q,
            local_hidden_scale,
            tensors["native_w13"],
            tensors["native_w13_scale"],
            local_ids,
            tensors["alpha1"],
            tensors["activation_global_scale"],
        )
        local_expert_output = launch_dependent(
            routed_w2,
            local_activation_q,
            local_activation_scale,
            tensors["w2"],
            tensors["w2_scale"],
            local_ids,
            tensors["alpha2"],
        )
        current_stream.wait_stream(shared_stream)
        local_output = launch_dependent(
            final, local_expert_output, local_weights, local_shared_output
        )
        return (
            local_output,
            local_logits,
            local_weights,
            local_ids,
            local_hidden_q,
            local_hidden_scale,
            local_activation_q,
            local_activation_scale,
            local_expert_output,
            local_shared_preactivation,
            local_shared_activation_q,
            local_shared_activation_scale,
            local_shared_output,
        )

    output = launch()
    torch.cuda.synchronize()
    return launch, output
