"""Matched local Helion control for the distributed TP4 MoE megakernel."""

from __future__ import annotations

from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import FP4_MAX
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import MMA_N
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import first_mma_column
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import fp4_nibble
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import (
    take_stable_argmax,
)
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4._common import take_stable_top8
import torch

import helion
import helion.language as hl


def _deepseek_v3_moe_nvfp4_tp_local(
    hidden_input: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    shared_w13: torch.Tensor,
    shared_w13_scale: torch.Tensor,
    shared_w2: torch.Tensor,
    shared_w2_scale: torch.Tensor,
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    shared_alpha1: torch.Tensor,
    shared_alpha2: torch.Tensor,
    input_global_scale: torch.Tensor,
    activation_global_scale: torch.Tensor,
    top_k: int,
    num_groups: int,
    topk_groups: int,
    routed_scale: float,
    w13_split_k: int,
    symmetric_output: torch.Tensor,
    group_name: hl.ProcessGroupName,
) -> tuple[torch.Tensor, ...]:
    # The model geometry and layouts are part of this pretuned kernel's fixed
    # contract. Tensor contents, including routing decisions, remain runtime
    # data even though their metadata is specialized explicitly.
    hl.specialize(hidden_input.size(0))
    hl.specialize(hidden_input.size(1))
    hl.specialize(router_weight.size(0))
    hl.specialize(router_weight.size(1))
    hl.specialize(correction_bias.size(0))
    hl.specialize(w13.size(0))
    hl.specialize(w13.size(1))
    hl.specialize(w13.size(2))
    hl.specialize(w13_scale.size(0))
    hl.specialize(w13_scale.size(1))
    hl.specialize(w13_scale.size(2))
    hl.specialize(w2.size(0))
    hl.specialize(w2.size(1))
    hl.specialize(w2.size(2))
    hl.specialize(w2_scale.size(0))
    hl.specialize(w2_scale.size(1))
    hl.specialize(w2_scale.size(2))
    hl.specialize(shared_w13.size(0))
    hl.specialize(shared_w13.size(1))
    hl.specialize(shared_w13_scale.size(0))
    hl.specialize(shared_w13_scale.size(1))
    hl.specialize(shared_w2.size(0))
    hl.specialize(shared_w2.size(1))
    hl.specialize(shared_w2_scale.size(0))
    hl.specialize(shared_w2_scale.size(1))
    hl.specialize(alpha1.size(0))
    hl.specialize(alpha2.size(0))
    hl.specialize(shared_alpha1.size(0))
    hl.specialize(shared_alpha2.size(0))
    hl.specialize(input_global_scale.size(0))
    hl.specialize(activation_global_scale.size(0))

    hl.specialize(hidden_input.stride(0))
    hl.specialize(hidden_input.stride(1))
    hl.specialize(router_weight.stride(0))
    hl.specialize(router_weight.stride(1))
    hl.specialize(correction_bias.stride(0))
    hl.specialize(w13.stride(0))
    hl.specialize(w13.stride(1))
    hl.specialize(w13.stride(2))
    hl.specialize(w13_scale.stride(0))
    hl.specialize(w13_scale.stride(1))
    hl.specialize(w13_scale.stride(2))
    hl.specialize(w2.stride(0))
    hl.specialize(w2.stride(1))
    hl.specialize(w2.stride(2))
    hl.specialize(w2_scale.stride(0))
    hl.specialize(w2_scale.stride(1))
    hl.specialize(w2_scale.stride(2))
    hl.specialize(shared_w13.stride(0))
    hl.specialize(shared_w13.stride(1))
    hl.specialize(shared_w13_scale.stride(0))
    hl.specialize(shared_w13_scale.stride(1))
    hl.specialize(shared_w2.stride(0))
    hl.specialize(shared_w2.stride(1))
    hl.specialize(shared_w2_scale.stride(0))
    hl.specialize(shared_w2_scale.stride(1))
    hl.specialize(alpha1.stride(0))
    hl.specialize(alpha2.stride(0))
    hl.specialize(shared_alpha1.stride(0))
    hl.specialize(shared_alpha2.stride(0))
    hl.specialize(input_global_scale.stride(0))
    hl.specialize(activation_global_scale.stride(0))

    rows, hidden_size = hidden_input.shape
    experts, router_hidden = router_weight.shape
    assert hidden_size == router_hidden
    router_logits = torch.empty(
        (rows, experts), dtype=torch.bfloat16, device=hidden_input.device
    )
    router_expert_block = hl.register_block_size(2, 32)
    router_reduction_block = hl.register_block_size(256, 1024)
    input_quant_groups = hidden_size // 16
    input_hidden_q = torch.empty(
        (rows, hidden_size // 2), dtype=torch.uint8, device=hidden_input.device
    )
    input_hidden_q_groups = input_hidden_q.view(rows, input_quant_groups, 8)
    input_hidden_scale = torch.empty(
        (rows, input_quant_groups),
        dtype=torch.float8_e4m3fn,
        device=hidden_input.device,
    )
    input_quant_group_block = hl.register_block_size(1, 32)
    hidden_q = input_hidden_q
    hidden_scale_bytes = input_hidden_scale
    logits = router_logits
    w13_scale_bytes = w13_scale
    w2_scale_bytes = w2_scale
    topk_batch, topk_num_experts = logits.size()
    top_k = hl.specialize(top_k)
    num_groups = hl.specialize(num_groups)
    topk_groups = hl.specialize(topk_groups)
    w13_split_k = hl.specialize(w13_split_k)
    assert top_k == 8 and topk_groups == 4
    topk_experts_per_group = topk_num_experts // num_groups
    hl.specialize(topk_batch)
    hl.specialize(topk_num_experts)
    topk_weights = torch.empty(
        (topk_batch, top_k), dtype=torch.float32, device=logits.device
    )
    topk_ids = torch.empty((topk_batch, top_k), dtype=torch.int32, device=logits.device)
    tokens, packed_hidden = hidden_q.size()
    experts, twice_intermediate, weight_packed_hidden = w13.size()
    intermediate = twice_intermediate // 2
    hidden_groups = packed_hidden // 8
    assert hidden_groups % w13_split_k == 0
    activation_groups = intermediate // 16
    assert tokens == 1 and packed_hidden == weight_packed_hidden
    hl.specialize(experts)
    hl.specialize(intermediate)
    activation_q = torch.empty(
        (top_k, intermediate // 2), dtype=torch.uint8, device=hidden_q.device
    )
    activation_scale = torch.empty(
        (top_k, activation_groups), dtype=torch.float8_e4m3fn, device=hidden_q.device
    )
    activation_q_groups = activation_q.view(top_k, activation_groups, 8)
    w13_tma = w13.view(experts * (twice_intermediate // 128), 128, packed_hidden)
    flat_w13_scale = w13_scale_bytes.view(experts * twice_intermediate, hidden_groups)
    w2_top_k, packed_intermediate = activation_q.size()
    w2_experts, hidden, weight_packed_intermediate = w2.size()
    w2_groups = packed_intermediate // 8
    assert packed_intermediate == weight_packed_intermediate
    hl.specialize(w2_experts)
    hl.specialize(w2_top_k)
    hl.specialize(hidden)
    expert_output = torch.empty(
        (w2_top_k, hidden), dtype=torch.bfloat16, device=w2.device
    )
    flat_w2_scale = w2_scale_bytes.view(w2_experts * hidden, w2_groups)
    shared_twice_intermediate, shared_weight_packed_hidden = shared_w13.shape
    shared_hidden, shared_weight_packed_intermediate = shared_w2.shape
    assert (
        shared_twice_intermediate == twice_intermediate
        and shared_weight_packed_hidden == packed_hidden
        and shared_hidden == hidden
        and shared_weight_packed_intermediate == packed_intermediate
    )
    flat_shared_w13 = shared_w13.view(twice_intermediate, packed_hidden)
    flat_shared_w13_scale = shared_w13_scale.view(twice_intermediate, hidden_groups)
    flat_shared_w2 = shared_w2.view(hidden, packed_intermediate)
    flat_shared_w2_scale = shared_w2_scale.view(hidden, w2_groups)
    w2_block_row = hl.register_block_size(128, 256)
    symmetric_rows, symmetric_hidden = symmetric_output.size()
    symmetric_rows = hl.specialize(symmetric_rows)
    symmetric_hidden = hl.specialize(symmetric_hidden)
    hl.specialize(symmetric_output.stride())
    assert symmetric_rows == 1 and symmetric_hidden == hidden
    assert symmetric_output.stride(0) == hidden
    assert symmetric_output.stride(1) == 1
    assert symmetric_output.storage_offset() == 0
    local_output = torch.as_strided(symmetric_output, (hidden,), (1,), storage_offset=0)
    output = symmetric_output
    shared_w13_preactivation = torch.empty(
        (twice_intermediate,), dtype=torch.bfloat16, device=w13.device
    )
    shared_activation_q = torch.empty(
        (1, intermediate // 2), dtype=torch.uint8, device=hidden_q.device
    )
    shared_activation_scale = torch.empty(
        (1, activation_groups), dtype=torch.float8_e4m3fn, device=hidden_q.device
    )
    shared_activation_q_groups = shared_activation_q.view(1, activation_groups, 8)
    shared_activation_q_flat = shared_activation_q.view(intermediate // 2)
    shared_hidden_q_flat = hidden_q.view(packed_hidden)
    shared_expert_output = torch.empty(
        (1, hidden), dtype=torch.bfloat16, device=w2.device
    )
    w13_partial = torch.empty(
        (top_k, activation_groups // 4, w13_split_k, 128),
        dtype=torch.float32,
        device=hidden_q.device,
    )
    w13_groups_per_split = hidden_groups // w13_split_k
    for tile_row, tile_expert in hl.tile(
        [rows, experts], block_size=[1, router_expert_block]
    ):
        accumulator = hl.zeros([tile_row, tile_expert], dtype=torch.float32)
        for tile_k in hl.tile(hidden_size, block_size=router_reduction_block):
            accumulator = torch.addmm(
                accumulator,
                hidden_input[tile_row, tile_k],
                router_weight[tile_expert, tile_k].T,
            )
        router_logits[tile_row, tile_expert] = accumulator.to(torch.bfloat16)
    for tile_row, tile_group in hl.tile(
        [rows, input_quant_groups], block_size=[1, input_quant_group_block]
    ):
        packed_lane = hl.arange(8)[None, :]
        group_offsets = tile_group.index[:, None] * 16
        low_values = hidden_input[tile_row, group_offsets + packed_lane * 2].to(
            torch.float32
        )
        high_values = hidden_input[tile_row, group_offsets + packed_lane * 2 + 1].to(
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
        low = fp4_nibble(low_values * global_scale / divisor[:, :, None])
        high = fp4_nibble(high_values * global_scale / divisor[:, :, None])
        packed = (low | high << 4).to(torch.uint8)
        input_hidden_q_groups[tile_row, tile_group, :] = packed
        input_hidden_scale[tile_row, tile_group] = block_scale
    for shared_w13_tile_row in hl.tile(twice_intermediate, block_size=16):
        shared_w13_weight_row = shared_w13_tile_row.index.to(torch.int64)
        shared_w13_accumulator = hl.zeros([shared_w13_tile_row], dtype=torch.float32)
        for shared_w13_tile_group in hl.tile(hidden_groups, block_size=64):
            shared_w13_group_mask = shared_w13_tile_group.index < hidden_groups
            shared_w13_group_offsets = (
                shared_w13_weight_row[:, None] * hidden_groups
                + shared_w13_tile_group.index[None, :]
            )
            shared_w13_weight_mask = (
                shared_w13_tile_row.index[:, None] < twice_intermediate
            ) & shared_w13_group_mask[None, :]
            shared_w13_values = hl.load_float4_e2m1fn_x16_to_float16(
                flat_shared_w13,
                shared_w13_group_offsets,
                extra_mask=shared_w13_weight_mask,
            )
            shared_w13_hidden_values = hl.load_float4_e2m1fn_x16_to_float16(
                shared_hidden_q_flat,
                shared_w13_tile_group.index,
                extra_mask=shared_w13_group_mask,
            )
            shared_w13_contribution = hl.zeros(
                [shared_w13_tile_row, shared_w13_tile_group], dtype=torch.float32
            )
            for shared_w13_lane in hl.static_range(16):
                shared_w13_contribution = (
                    shared_w13_contribution
                    + shared_w13_values[shared_w13_lane]
                    * shared_w13_hidden_values[shared_w13_lane][None, :]
                )
            shared_w13_scale_value = flat_shared_w13_scale[
                shared_w13_weight_row[:, None], shared_w13_tile_group.index[None, :]
            ].to(torch.float32)
            shared_w13_input_scale = hidden_scale_bytes[0, shared_w13_tile_group].to(
                torch.float32
            )
            shared_w13_accumulator = shared_w13_accumulator + (
                shared_w13_contribution.to(torch.float32)
                * shared_w13_scale_value
                * shared_w13_input_scale[None, :]
            ).sum(dim=-1)
        shared_w13_preactivation[shared_w13_tile_row] = (
            shared_w13_accumulator * torch.sum(shared_alpha1[:])
        ).to(torch.bfloat16)
    for _topk_program in hl.grid(1):
        topk_scores = torch.sigmoid(logits[:, :].to(torch.float32))
        topk_grouped = (topk_scores + correction_bias[None, :]).view(
            topk_batch, num_groups, topk_experts_per_group
        )
        topk_negative_infinity = torch.full_like(topk_grouped, float("-inf"))
        topk_within_group_indices = hl.arange(topk_experts_per_group)[None, None, :].to(
            torch.int32
        )
        topk_group_best = torch.amax(topk_grouped, dim=-1, keepdim=True)
        topk_group_best_id = torch.amin(
            torch.where(
                topk_grouped == topk_group_best,
                topk_within_group_indices,
                torch.full_like(topk_within_group_indices, topk_experts_per_group),
            ),
            dim=-1,
        )
        topk_group_without_best = torch.where(
            topk_within_group_indices == topk_group_best_id[:, :, None],
            topk_negative_infinity,
            topk_grouped,
        )
        topk_group_scores = topk_group_best.view(topk_batch, num_groups) + torch.amax(
            topk_group_without_best, dim=-1
        )
        topk_group_indices = hl.arange(num_groups)[None, :].to(torch.int32)
        topk_group_id, topk_remaining_groups = take_stable_argmax(
            topk_group_scores, topk_group_indices, num_groups
        )
        topk_allowed = topk_group_indices == topk_group_id[:, None]
        for _topk_group_rank in hl.static_range(1, topk_groups):
            topk_group_id, topk_remaining_groups = take_stable_argmax(
                topk_remaining_groups, topk_group_indices, num_groups
            )
            topk_allowed = topk_allowed | (topk_group_indices == topk_group_id[:, None])
        topk_masked = torch.where(
            topk_allowed.view(topk_batch, num_groups, 1),
            topk_grouped,
            topk_negative_infinity,
        ).view(topk_batch, topk_num_experts)
        topk_expert_indices = hl.arange(topk_num_experts)[None, :].to(torch.int32)
        (
            topk_id_0,
            topk_id_1,
            topk_id_2,
            topk_id_3,
            topk_id_4,
            topk_id_5,
            topk_id_6,
            topk_id_7,
        ) = take_stable_top8(topk_masked, topk_expert_indices, topk_num_experts)
        topk_weight_0 = torch.sigmoid(torch.sum(logits[:, topk_id_0].float(), dim=-1))
        topk_weight_1 = torch.sigmoid(torch.sum(logits[:, topk_id_1].float(), dim=-1))
        topk_weight_2 = torch.sigmoid(torch.sum(logits[:, topk_id_2].float(), dim=-1))
        topk_weight_3 = torch.sigmoid(torch.sum(logits[:, topk_id_3].float(), dim=-1))
        topk_weight_4 = torch.sigmoid(torch.sum(logits[:, topk_id_4].float(), dim=-1))
        topk_weight_5 = torch.sigmoid(torch.sum(logits[:, topk_id_5].float(), dim=-1))
        topk_weight_6 = torch.sigmoid(torch.sum(logits[:, topk_id_6].float(), dim=-1))
        topk_weight_7 = torch.sigmoid(torch.sum(logits[:, topk_id_7].float(), dim=-1))
        topk_denominator = (
            topk_weight_0
            + topk_weight_1
            + topk_weight_2
            + topk_weight_3
            + topk_weight_4
            + topk_weight_5
            + topk_weight_6
            + topk_weight_7
        )
        topk_ids[:, 0] = topk_id_0
        topk_ids[:, 1] = topk_id_1
        topk_ids[:, 2] = topk_id_2
        topk_ids[:, 3] = topk_id_3
        topk_ids[:, 4] = topk_id_4
        topk_ids[:, 5] = topk_id_5
        topk_ids[:, 6] = topk_id_6
        topk_ids[:, 7] = topk_id_7
        topk_weights[:, 0] = topk_weight_0 / topk_denominator * routed_scale
        topk_weights[:, 1] = topk_weight_1 / topk_denominator * routed_scale
        topk_weights[:, 2] = topk_weight_2 / topk_denominator * routed_scale
        topk_weights[:, 3] = topk_weight_3 / topk_denominator * routed_scale
        topk_weights[:, 4] = topk_weight_4 / topk_denominator * routed_scale
        topk_weights[:, 5] = topk_weight_5 / topk_denominator * routed_scale
        topk_weights[:, 6] = topk_weight_6 / topk_denominator * routed_scale
        topk_weights[:, 7] = topk_weight_7 / topk_denominator * routed_scale
    for w13_tile_slot, w13_tile_output_group, w13_tile_split in hl.tile(
        [top_k, activation_groups, w13_split_k], block_size=[1, 4, 1]
    ):
        w13_slot = w13_tile_slot.begin
        w13_split = w13_tile_split.begin
        w13_expert = topk_ids[0, w13_slot]
        w13_expert_address = w13_expert.to(torch.int64)
        w13_tma_row = (
            w13_expert * (twice_intermediate // 128) + w13_tile_output_group.begin // 4
        )
        w13_physical_row_index = (w13_tile_output_group.begin * 32 + hl.arange(128)).to(
            torch.int64
        )
        w13_weight_row = (
            w13_expert_address * twice_intermediate + w13_physical_row_index
        )
        w13_accumulator = hl.zeros([128, MMA_N], dtype=torch.float32)
        for w13_local_group in hl.tile(w13_groups_per_split, block_size=32):
            w13_group_begin = w13_split * w13_groups_per_split + w13_local_group.begin
            w13_group_index = w13_split * w13_groups_per_split + w13_local_group.index
            w13_packed_index = (
                w13_group_begin * 8 + hl.arange(w13_local_group.block_size * 8)
            ).to(torch.int64)
            w13_lhs = w13_tma[w13_tma_row, :, w13_packed_index]
            w13_lhs_scale = flat_w13_scale[
                w13_weight_row[:, None], w13_group_index[None, :]
            ]
            w13_hidden_bytes = hidden_q[0, w13_packed_index]
            w13_rhs = w13_hidden_bytes[:, None].expand(w13_hidden_bytes.size(0), MMA_N)
            w13_hidden_scale = hidden_scale_bytes[0, w13_group_index]
            w13_rhs_scale = w13_hidden_scale[None, :].expand(
                MMA_N, w13_local_group.block_size
            )
            w13_accumulator = hl.dot_scaled(
                w13_lhs,
                w13_lhs_scale,
                "e2m1",
                w13_rhs,
                w13_rhs_scale,
                "e2m1",
                acc=w13_accumulator,
                out_dtype=torch.float32,
            )
        w13_partial[
            w13_slot,
            w13_tile_output_group.begin // 4,
            w13_split,
            :,
        ] = first_mma_column(w13_accumulator)
    for w13_tile_slot, w13_tile_output_group in hl.tile(
        [top_k, activation_groups], block_size=[1, 4]
    ):
        w13_slot = w13_tile_slot.begin
        w13_expert = topk_ids[0, w13_slot]
        w13_preactivation = (
            w13_partial[
                w13_slot,
                w13_tile_output_group.begin // 4,
                :,
                hl.arange(128),
            ]
            .sum(dim=0)
            .reshape(64, 2)
        )
        w13_pair = hl.arange(2)
        w13_gate = torch.sum(w13_preactivation * (w13_pair[None, :] == 0), dim=-1)
        w13_up = torch.sum(w13_preactivation * (w13_pair[None, :] == 1), dim=-1)
        w13_first_scale = alpha1[w13_expert]
        w13_gate = w13_gate * w13_first_scale
        w13_up = w13_up * w13_first_scale
        w13_activated = w13_gate * torch.sigmoid(w13_gate) * w13_up
        w13_activated_groups = w13_activated.reshape(4, 16)
        w13_global_scale = torch.sum(activation_global_scale[:])
        w13_block_scale_f32 = (
            torch.amax(torch.abs(w13_activated_groups), dim=-1)
            * w13_global_scale
            / FP4_MAX
        )
        w13_block_scale = w13_block_scale_f32.to(torch.float8_e4m3fn)
        w13_actual_scale = w13_block_scale.to(torch.float32)
        w13_divisor = torch.where(w13_actual_scale > 0, w13_actual_scale, 1.0)
        w13_scaled = w13_activated_groups * w13_global_scale / w13_divisor[:, None]
        w13_nibbles = fp4_nibble(w13_scaled)
        w13_low, w13_high = hl.split(w13_nibbles.reshape(4, 8, 2))
        w13_packed = w13_low | w13_high << 4
        activation_q_groups[w13_slot, w13_tile_output_group, :] = w13_packed.reshape(
            4, 8
        ).to(torch.uint8)
        activation_scale[w13_slot, w13_tile_output_group] = w13_block_scale
    for shared_activation_tile_group in hl.tile(activation_groups, block_size=32):
        shared_activation_index = (
            shared_activation_tile_group.index[:, None] * 16 + hl.arange(16)[None, :]
        )
        shared_gate = shared_w13_preactivation[shared_activation_index * 2].to(
            torch.float32
        )
        shared_up = shared_w13_preactivation[shared_activation_index * 2 + 1].to(
            torch.float32
        )
        shared_activated = (
            (shared_gate * torch.sigmoid(shared_gate) * shared_up)
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        shared_global_scale = torch.sum(activation_global_scale[:])
        shared_block_scale_f32 = (
            torch.amax(torch.abs(shared_activated), dim=-1)
            * shared_global_scale
            / FP4_MAX
        )
        shared_block_scale = shared_block_scale_f32.to(torch.float8_e4m3fn)
        shared_actual_scale = shared_block_scale.to(torch.float32)
        shared_divisor = torch.where(shared_actual_scale > 0, shared_actual_scale, 1.0)
        shared_scaled = shared_activated * shared_global_scale / shared_divisor[:, None]
        shared_nibbles = fp4_nibble(shared_scaled)
        shared_low, shared_high = hl.split(
            shared_nibbles.reshape(shared_activation_tile_group.block_size, 8, 2)
        )
        shared_packed = shared_low | shared_high << 4
        shared_activation_q_groups[0, shared_activation_tile_group, :] = (
            shared_packed.reshape(shared_activation_tile_group.block_size, 8).to(
                torch.uint8
            )
        )
        shared_activation_scale[0, shared_activation_tile_group] = shared_block_scale
    for w2_tile_slot, w2_tile_row in hl.tile(
        [w2_top_k, hidden], block_size=[1, w2_block_row]
    ):
        w2_slot = w2_tile_slot.begin
        w2_expert = topk_ids[0, w2_slot]
        w2_expert_address = w2_expert.to(torch.int64)
        w2_row_index = w2_tile_row.index.to(torch.int32)
        w2_weight_row = w2_expert_address * hidden + w2_row_index
        w2_accumulator = hl.zeros([w2_block_row, MMA_N], dtype=torch.float32)
        for w2_tile_group in hl.tile(w2_groups, block_size=16):
            w2_packed_index = (
                w2_tile_group.begin * 8 + hl.arange(w2_tile_group.block_size * 8)
            ).to(torch.int64)
            w2_lhs = w2[w2_expert, w2_tile_row, w2_packed_index]
            w2_lhs_scale = flat_w2_scale[
                w2_weight_row[:, None], w2_tile_group.index[None, :]
            ]
            w2_activation_bytes = activation_q[w2_slot, w2_packed_index]
            w2_rhs = w2_activation_bytes[:, None].expand(
                w2_activation_bytes.size(0), MMA_N
            )
            w2_activation_scale = activation_scale[w2_slot, w2_tile_group]
            w2_rhs_scale = w2_activation_scale[None, :].expand(
                MMA_N, w2_tile_group.block_size
            )
            w2_accumulator = hl.dot_scaled(
                w2_lhs,
                w2_lhs_scale,
                "e2m1",
                w2_rhs,
                w2_rhs_scale,
                "e2m1",
                acc=w2_accumulator,
                out_dtype=torch.float32,
            )
        expert_output[w2_tile_slot, w2_tile_row] = (
            (first_mma_column(w2_accumulator) * alpha2[w2_expert])
            .reshape(1, w2_block_row)
            .to(torch.bfloat16)
        )
    for shared_w2_tile_row in hl.tile(hidden, block_size=16):
        shared_w2_weight_row = shared_w2_tile_row.index.to(torch.int64)
        shared_w2_accumulator = hl.zeros([shared_w2_tile_row], dtype=torch.float32)
        for shared_w2_tile_group in hl.tile(w2_groups, block_size=32):
            shared_w2_group_mask = shared_w2_tile_group.index < w2_groups
            shared_w2_group_offsets = (
                shared_w2_weight_row[:, None] * w2_groups
                + shared_w2_tile_group.index[None, :]
            )
            shared_w2_weight_mask = (
                shared_w2_tile_row.index[:, None] < hidden
            ) & shared_w2_group_mask[None, :]
            shared_w2_values = hl.load_float4_e2m1fn_x16_to_float16(
                flat_shared_w2,
                shared_w2_group_offsets,
                extra_mask=shared_w2_weight_mask,
            )
            shared_w2_activation_values = hl.load_float4_e2m1fn_x16_to_float16(
                shared_activation_q_flat,
                shared_w2_tile_group.index,
                extra_mask=shared_w2_group_mask,
            )
            shared_w2_contribution = hl.zeros(
                [shared_w2_tile_row, shared_w2_tile_group], dtype=torch.float32
            )
            for shared_w2_lane in hl.static_range(16):
                shared_w2_contribution = (
                    shared_w2_contribution
                    + shared_w2_values[shared_w2_lane]
                    * shared_w2_activation_values[shared_w2_lane][None, :]
                )
            shared_w2_scale_value = flat_shared_w2_scale[
                shared_w2_weight_row[:, None], shared_w2_tile_group.index[None, :]
            ].to(torch.float32)
            shared_w2_input_scale = shared_activation_scale[0, shared_w2_tile_group].to(
                torch.float32
            )
            shared_w2_accumulator = shared_w2_accumulator + (
                shared_w2_contribution.to(torch.float32)
                * shared_w2_scale_value
                * shared_w2_input_scale[None, :]
            ).sum(dim=-1)
        shared_expert_output[0, shared_w2_tile_row] = (
            shared_w2_accumulator * torch.sum(shared_alpha2[:])
        ).to(torch.bfloat16)
    for final_tile_n in hl.tile(hidden):
        final_routed_values = expert_output[:top_k, final_tile_n].to(torch.float32)
        final_routed_weights = (
            topk_weights[0, :top_k].to(torch.bfloat16).to(torch.float32)
        )
        final_routed_output = torch.sum(
            final_routed_values * final_routed_weights[:, None], dim=0, keepdim=True
        ).to(torch.bfloat16)
        final_shared_output = shared_expert_output[hl.arange(1), final_tile_n]
        local_output[final_tile_n] = (
            (final_routed_output + final_shared_output).reshape(-1).to(torch.bfloat16)
        )
    return (
        output,
        router_logits,
        topk_weights,
        topk_ids,
        input_hidden_q,
        input_hidden_scale,
        activation_q,
        activation_scale,
        expert_output,
        shared_w13_preactivation,
        shared_activation_q,
        shared_activation_scale,
        shared_expert_output,
    )


def _config() -> helion.Config:
    return helion.Config(
        block_sizes=[8, 512, 32, 256, 512],
        cross_loop_pipeline="dynamic",
        host_tensor_descriptors=True,
        indexing=[
            "tensor_descriptor" if index in (41, 43, 58, 60) else "pointer"
            for index in range(76)
        ],
        maxnreg=None,
        num_sm_multiplier=1,
        num_stages=1,
        num_warps=4,
        pid_type="persistent_blocked",
        range_flattens=[
            None,
            None,
            None,
            None,
            False,
            None,
            None,
            False,
            None,
            None,
            None,
            False,
            None,
            False,
            None,
        ],
        range_multi_buffers=[
            None,
            None,
            None,
            None,
            True,
            None,
            None,
            False,
            None,
            None,
            None,
            True,
            None,
            True,
            None,
        ],
        range_num_stages=[0, 4, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0],
    )


deepseek_v3_moe_nvfp4_tp_local = helion.kernel(
    _deepseek_v3_moe_nvfp4_tp_local,
    config=_config(),
    static_shapes=False,
    backend="triton",
)
