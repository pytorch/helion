# ruff: noqa: ANN201, ANN202, E402, I001
"""DeepSeek-V3 batch-one NVFP4 decode-MoE megakernel for NVIDIA B200.

The boundary includes router projection, DeepSeek grouped routing, NVFP4 input
quantization, routed and shared experts, and final accumulation. The matched
standalone uses the same nine PDL-chained logical roots and exact intermediate
numerics.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path
from typing import Any

VLLM_ROOT = os.environ.get("VLLM_ROOT")
if VLLM_ROOT:
    VLLM_ROOT = str(Path(VLLM_ROOT).resolve())

import torch

import helion
import helion.language as hl


FP4_MAX = 6.0
MMA_N = 16
ROUTED_SCALE = 2.5
OUTPUT_NAMES = (
    "output",
    "router_logits",
    "topk_weights",
    "topk_ids",
    "input_q",
    "input_scale",
    "routed_activation_q",
    "routed_activation_scale",
    "routed_expert_output",
    "shared_w13_preactivation",
    "shared_activation_q",
    "shared_activation_scale",
    "shared_expert_output",
)
ROUTING_CASES = (
    ("spread", (3, 17, 35, 50, 77, 95, 113, 127)),
    ("clustered", (0, 1, 2, 3, 4, 5, 6, 7)),
    ("edge", (128, 159, 160, 191, 192, 223, 224, 255)),
)
ROUTING_LOGITS = (8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)


@dataclasses.dataclass(frozen=True)
class Shape:
    batch: int = 1
    hidden: int = 7168
    intermediate: int = 2048
    num_experts: int = 256
    top_k: int = 8
    num_groups: int = 8
    topk_groups: int = 4
    routed_scale: float = ROUTED_SCALE


def _fp4_nibble(value: torch.Tensor) -> torch.Tensor:
    """Round scaled values to E2M1 with FlashInfer's tie convention."""
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
    """Select the real token column from the padded native-MMA N tile."""
    column = hl.arange(MMA_N)
    return torch.sum(value * (column[None, :] == 0).to(torch.float32), dim=-1)


@helion.aot_kernel(static_shapes=False, backend="triton")
def deepseek_v3_moe_nvfp4(
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
):
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
    w13_output_groups = hl.register_block_size(4, 4)
    w13_block_groups = hl.register_block_size(32, 32)
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
    w2_block_groups = hl.register_block_size(32, 32)
    output = torch.empty((1, hidden), dtype=torch.bfloat16, device=w2.device)
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
    shared_scalar_w13_row = hl.register_block_size(16, 16)
    shared_scalar_w13_group = hl.register_block_size(64, 64)
    shared_scalar_activation_group = hl.register_block_size(32, 32)
    shared_scalar_w2_row = hl.register_block_size(16, 16)
    shared_scalar_w2_group = hl.register_block_size(64, 64)
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
        low = _fp4_nibble(low_values * global_scale / divisor[:, :, None])
        high = _fp4_nibble(high_values * global_scale / divisor[:, :, None])
        packed = (low | high << 4).to(torch.uint8)
        input_hidden_q_groups[tile_row, tile_group, :] = packed
        input_hidden_scale[tile_row, tile_group] = block_scale
    for shared_w13_tile_row in hl.tile(
        twice_intermediate, block_size=shared_scalar_w13_row
    ):
        shared_w13_weight_row = shared_w13_tile_row.index.to(torch.int64)
        shared_w13_accumulator = hl.zeros([shared_scalar_w13_row], dtype=torch.float32)
        for shared_w13_tile_group in hl.tile(
            hidden_groups, block_size=shared_scalar_w13_group
        ):
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
                [shared_scalar_w13_row, shared_scalar_w13_group], dtype=torch.float32
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
        topk_group_best_1 = torch.amax(topk_grouped, dim=-1, keepdim=True)
        topk_group_best_id_1 = torch.amin(
            torch.where(
                topk_grouped == topk_group_best_1,
                topk_within_group_indices,
                torch.full_like(topk_within_group_indices, topk_experts_per_group),
            ),
            dim=-1,
        )
        topk_group_without_best = torch.where(
            topk_within_group_indices == topk_group_best_id_1[:, :, None],
            topk_negative_infinity,
            topk_grouped,
        )
        topk_group_scores = topk_group_best_1.view(topk_batch, num_groups) + torch.amax(
            topk_group_without_best, dim=-1
        )
        topk_group_indices = hl.arange(num_groups)[None, :].to(torch.int32)
        topk_group_max_1 = torch.amax(topk_group_scores, dim=-1, keepdim=True)
        topk_group_id_1 = torch.amin(
            torch.where(
                topk_group_scores == topk_group_max_1,
                topk_group_indices,
                torch.full_like(topk_group_indices, num_groups),
            ),
            dim=-1,
        )
        topk_remaining_groups = torch.where(
            topk_group_indices == topk_group_id_1[:, None],
            torch.full_like(topk_group_scores, float("-inf")),
            topk_group_scores,
        )
        topk_group_max_2 = torch.amax(topk_remaining_groups, dim=-1, keepdim=True)
        topk_group_id_2 = torch.amin(
            torch.where(
                topk_remaining_groups == topk_group_max_2,
                topk_group_indices,
                torch.full_like(topk_group_indices, num_groups),
            ),
            dim=-1,
        )
        topk_remaining_groups = torch.where(
            topk_group_indices == topk_group_id_2[:, None],
            torch.full_like(topk_group_scores, float("-inf")),
            topk_remaining_groups,
        )
        topk_group_max_3 = torch.amax(topk_remaining_groups, dim=-1, keepdim=True)
        topk_group_id_3 = torch.amin(
            torch.where(
                topk_remaining_groups == topk_group_max_3,
                topk_group_indices,
                torch.full_like(topk_group_indices, num_groups),
            ),
            dim=-1,
        )
        topk_remaining_groups = torch.where(
            topk_group_indices == topk_group_id_3[:, None],
            torch.full_like(topk_group_scores, float("-inf")),
            topk_remaining_groups,
        )
        topk_group_max_4 = torch.amax(topk_remaining_groups, dim=-1, keepdim=True)
        topk_group_id_4 = torch.amin(
            torch.where(
                topk_remaining_groups == topk_group_max_4,
                topk_group_indices,
                torch.full_like(topk_group_indices, num_groups),
            ),
            dim=-1,
        )
        topk_allowed = (
            (topk_group_indices == topk_group_id_1[:, None])
            | (topk_group_indices == topk_group_id_2[:, None])
            | (topk_group_indices == topk_group_id_3[:, None])
            | (topk_group_indices == topk_group_id_4[:, None])
        )
        topk_masked = torch.where(
            topk_allowed.view(topk_batch, num_groups, 1),
            topk_grouped,
            topk_negative_infinity,
        ).view(topk_batch, topk_num_experts)
        topk_expert_indices = hl.arange(topk_num_experts)[None, :].to(torch.int32)
        topk_value_0 = torch.amax(topk_masked, dim=-1, keepdim=True)
        topk_id_0 = torch.amin(
            torch.where(
                topk_masked == topk_value_0,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_masked_1 = torch.where(
            topk_expert_indices == topk_id_0[:, None],
            torch.full_like(topk_masked, float("-inf")),
            topk_masked,
        )
        topk_value_1 = torch.amax(topk_masked_1, dim=-1, keepdim=True)
        topk_id_1 = torch.amin(
            torch.where(
                topk_masked_1 == topk_value_1,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_masked_2 = torch.where(
            topk_expert_indices == topk_id_1[:, None],
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_1,
        )
        topk_value_2 = torch.amax(topk_masked_2, dim=-1, keepdim=True)
        topk_id_2 = torch.amin(
            torch.where(
                topk_masked_2 == topk_value_2,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_masked_3 = torch.where(
            topk_expert_indices == topk_id_2[:, None],
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_2,
        )
        topk_value_3 = torch.amax(topk_masked_3, dim=-1, keepdim=True)
        topk_id_3 = torch.amin(
            torch.where(
                topk_masked_3 == topk_value_3,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_masked_4 = torch.where(
            topk_expert_indices == topk_id_3[:, None],
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_3,
        )
        topk_value_4 = torch.amax(topk_masked_4, dim=-1, keepdim=True)
        topk_id_4 = torch.amin(
            torch.where(
                topk_masked_4 == topk_value_4,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_masked_5 = torch.where(
            topk_expert_indices == topk_id_4[:, None],
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_4,
        )
        topk_value_5 = torch.amax(topk_masked_5, dim=-1, keepdim=True)
        topk_id_5 = torch.amin(
            torch.where(
                topk_masked_5 == topk_value_5,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_masked_6 = torch.where(
            topk_expert_indices == topk_id_5[:, None],
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_5,
        )
        topk_value_6 = torch.amax(topk_masked_6, dim=-1, keepdim=True)
        topk_id_6 = torch.amin(
            torch.where(
                topk_masked_6 == topk_value_6,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_masked_7 = torch.where(
            topk_expert_indices == topk_id_6[:, None],
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_6,
        )
        topk_value_7 = torch.amax(topk_masked_7, dim=-1, keepdim=True)
        topk_id_7 = torch.amin(
            torch.where(
                topk_masked_7 == topk_value_7,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
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
        topk_weights[:, 0] = topk_weight_0 / topk_denominator * routed_scale
        topk_weights[:, 1] = topk_weight_1 / topk_denominator * routed_scale
        topk_weights[:, 2] = topk_weight_2 / topk_denominator * routed_scale
        topk_weights[:, 3] = topk_weight_3 / topk_denominator * routed_scale
        topk_weights[:, 4] = topk_weight_4 / topk_denominator * routed_scale
        topk_weights[:, 5] = topk_weight_5 / topk_denominator * routed_scale
        topk_weights[:, 6] = topk_weight_6 / topk_denominator * routed_scale
        topk_weights[:, 7] = topk_weight_7 / topk_denominator * routed_scale
        topk_ids[:, 0] = topk_id_0
        topk_ids[:, 1] = topk_id_1
        topk_ids[:, 2] = topk_id_2
        topk_ids[:, 3] = topk_id_3
        topk_ids[:, 4] = topk_id_4
        topk_ids[:, 5] = topk_id_5
        topk_ids[:, 6] = topk_id_6
        topk_ids[:, 7] = topk_id_7
    for w13_tile_slot, w13_tile_output_group in hl.tile(
        [top_k, activation_groups], block_size=[1, w13_output_groups]
    ):
        w13_slot = w13_tile_slot.begin
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
        for w13_tile_group in hl.tile(hidden_groups, block_size=w13_block_groups):
            w13_packed_index = (
                w13_tile_group.begin * 8 + hl.arange(w13_block_groups * 8)
            ).to(torch.int64)
            w13_lhs = w13_tma[w13_tma_row, :, w13_packed_index]
            w13_lhs_scale = flat_w13_scale[
                w13_weight_row[:, None], w13_tile_group.index[None, :]
            ]
            w13_hidden_bytes = hidden_q[0, w13_packed_index]
            w13_rhs = w13_hidden_bytes[:, None].expand(w13_hidden_bytes.size(0), MMA_N)
            w13_hidden_scale = hidden_scale_bytes[0, w13_tile_group]
            w13_rhs_scale = w13_hidden_scale[None, :].expand(MMA_N, w13_block_groups)
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
        w13_preactivation = _first_mma_column(w13_accumulator).reshape(64, 2)
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
        w13_nibbles = _fp4_nibble(w13_scaled)
        w13_low, w13_high = hl.split(w13_nibbles.reshape(4, 8, 2))
        w13_packed = w13_low | w13_high << 4
        activation_q_groups[w13_slot, w13_tile_output_group, :] = w13_packed.reshape(
            4, 8
        ).to(torch.uint8)
        activation_scale[w13_slot, w13_tile_output_group] = w13_block_scale
    for shared_activation_tile_group in hl.tile(
        activation_groups, block_size=shared_scalar_activation_group
    ):
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
        shared_nibbles = _fp4_nibble(shared_scaled)
        shared_low, shared_high = hl.split(
            shared_nibbles.reshape(shared_scalar_activation_group, 8, 2)
        )
        shared_packed = shared_low | shared_high << 4
        shared_activation_q_groups[0, shared_activation_tile_group, :] = (
            shared_packed.reshape(shared_scalar_activation_group, 8).to(torch.uint8)
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
        for w2_tile_group in hl.tile(w2_groups, block_size=w2_block_groups):
            w2_packed_index = (
                w2_tile_group.begin * 8 + hl.arange(w2_block_groups * 8)
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
            w2_rhs_scale = w2_activation_scale[None, :].expand(MMA_N, w2_block_groups)
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
            (_first_mma_column(w2_accumulator) * alpha2[w2_expert])
            .reshape(1, w2_block_row)
            .to(torch.bfloat16)
        )
    for shared_w2_tile_row in hl.tile(hidden, block_size=shared_scalar_w2_row):
        shared_w2_weight_row = shared_w2_tile_row.index.to(torch.int64)
        shared_w2_accumulator = hl.zeros([shared_scalar_w2_row], dtype=torch.float32)
        for shared_w2_tile_group in hl.tile(
            w2_groups, block_size=shared_scalar_w2_group
        ):
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
                [shared_scalar_w2_row, shared_scalar_w2_group], dtype=torch.float32
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
        output[:, final_tile_n] = (final_routed_output + final_shared_output).to(
            torch.bfloat16
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


def _ensure_vllm_importable() -> None:
    if VLLM_ROOT is not None and VLLM_ROOT not in sys.path:
        sys.path.insert(0, VLLM_ROOT)


def _router_weight(hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    hidden_f32 = hidden.float()
    denominator = torch.sum(hidden_f32 * hidden_f32)
    return (target.float().T * hidden_f32 / denominator).to(torch.bfloat16)


def _set_routing_case(
    tensors: dict[str, torch.Tensor], selected_ids: tuple[int, ...]
) -> None:
    target = tensors["routing_target"]
    target.fill_(-12.0)
    bias_index = torch.arange(
        tensors["correction_bias"].numel(),
        device=target.device,
        dtype=torch.int32,
    )
    tensors["correction_bias"].copy_((((bias_index * 7) % 17) - 8).float() * 0.001)
    for expert, value in zip(selected_ids, ROUTING_LOGITS, strict=True):
        target[0, expert] = value
    tensors["router_weight"].copy_(_router_weight(tensors["hidden"], target))


def _interleave_w13(
    weight: torch.Tensor,
    scale: torch.Tensor,
    intermediate: int,
    initialized_experts: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleave gate/up rows to match the native-MMA output geometry."""
    native_weight = torch.zeros_like(weight)
    native_scale = torch.zeros_like(scale)
    for expert in initialized_experts:
        native_weight_view = native_weight[expert].view(intermediate, 2, weight.size(2))
        native_weight_view[:, 0].copy_(weight[expert, :intermediate])
        native_weight_view[:, 1].copy_(weight[expert, intermediate:])
        native_scale_view = native_scale[expert].view(intermediate, 2, scale.size(2))
        native_scale_view[:, 0].copy_(scale[expert, :intermediate])
        native_scale_view[:, 1].copy_(scale[expert, intermediate:])
    return native_weight, native_scale


def _allocate(shape: Shape, seed: int = 17) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    hidden = torch.randn(shape.batch, shape.hidden, device=device, dtype=torch.bfloat16)
    routing_target = torch.empty(
        shape.batch, shape.num_experts, device=device, dtype=torch.bfloat16
    )
    router_weight = torch.empty(
        shape.num_experts, shape.hidden, device=device, dtype=torch.bfloat16
    )
    correction_bias = torch.zeros(shape.num_experts, device=device, dtype=torch.float32)
    input_global_scale = torch.ones(1, device=device, dtype=torch.float32)
    activation_global_scale = torch.full((1,), 16.0, device=device, dtype=torch.float32)

    weight_w13 = torch.zeros(
        shape.num_experts,
        2 * shape.intermediate,
        shape.hidden // 2,
        device=device,
        dtype=torch.uint8,
    )
    scale_w13 = torch.zeros(
        shape.num_experts,
        2 * shape.intermediate,
        shape.hidden // 16,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    weight_w2 = torch.zeros(
        shape.num_experts,
        shape.hidden,
        shape.intermediate // 2,
        device=device,
        dtype=torch.uint8,
    )
    scale_w2 = torch.zeros(
        shape.num_experts,
        shape.hidden,
        shape.intermediate // 16,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    shared_weight_w13 = torch.empty(
        2 * shape.intermediate,
        shape.hidden // 2,
        device=device,
        dtype=torch.uint8,
    )
    shared_scale_w13 = torch.empty(
        2 * shape.intermediate,
        shape.hidden // 16,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    shared_weight_w2 = torch.empty(
        shape.hidden,
        shape.intermediate // 2,
        device=device,
        dtype=torch.uint8,
    )
    shared_scale_w2 = torch.empty(
        shape.hidden,
        shape.intermediate // 16,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    initialized_experts = tuple(
        sorted(
            {
                expert
                for _label, selected_ids in ROUTING_CASES
                for expert in selected_ids
            }
            | {shape.num_experts - 1}
        )
    )
    w13_rows = torch.arange(2 * shape.intermediate, device=device, dtype=torch.int32)[
        :, None
    ]
    w13_groups = torch.arange(shape.hidden // 16, device=device, dtype=torch.int32)[
        None, :
    ]
    w13_scale_pattern = torch.exp2((-9 + (w13_rows + 3 * w13_groups) % 3).float()) * (
        1.0 + ((5 * w13_rows + 7 * w13_groups) % 8).float() / 8.0
    )
    w2_rows = torch.arange(shape.hidden, device=device, dtype=torch.int32)[:, None]
    w2_groups = torch.arange(
        shape.intermediate // 16, device=device, dtype=torch.int32
    )[None, :]
    w2_scale_pattern = torch.exp2((-7 + (w2_rows + 5 * w2_groups) % 3).float()) * (
        1.0 + ((3 * w2_rows + 7 * w2_groups) % 8).float() / 8.0
    )
    for expert in initialized_experts:
        weight_w13[expert].random_(0, 256)
        scale_w13[expert].copy_(
            torch.roll(w13_scale_pattern, shifts=expert % 3, dims=1)
        )
        weight_w2[expert].random_(0, 256)
        scale_w2[expert].copy_(torch.roll(w2_scale_pattern, shifts=expert % 3, dims=1))

    shared_weight_w13.random_(0, 256)
    shared_scale_w13.copy_(torch.roll(w13_scale_pattern, shifts=1, dims=1))
    shared_weight_w2.random_(0, 256)
    shared_scale_w2.copy_(torch.roll(w2_scale_pattern, shifts=1, dims=1))

    native_w13, native_w13_scale = _interleave_w13(
        weight_w13,
        scale_w13,
        shape.intermediate,
        initialized_experts,
    )
    native_shared_w13, native_shared_w13_scale = _interleave_w13(
        shared_weight_w13[None],
        shared_scale_w13[None],
        shape.intermediate,
        (0,),
    )
    alpha1 = torch.ones(shape.num_experts, device=device, dtype=torch.float32)
    alpha2 = torch.full(
        (shape.num_experts,),
        1.0 / float(activation_global_scale.item()),
        device=device,
        dtype=torch.float32,
    )
    shared_alpha1 = torch.ones(1, device=device, dtype=torch.float32)
    shared_alpha2 = torch.full(
        (1,),
        1.0 / float(activation_global_scale.item()),
        device=device,
        dtype=torch.float32,
    )
    tensors = {
        "hidden": hidden,
        "routing_target": routing_target,
        "router_weight": router_weight,
        "correction_bias": correction_bias,
        "w13": weight_w13,
        "w13_scale": scale_w13,
        "native_w13": native_w13,
        "native_w13_scale": native_w13_scale,
        "w2": weight_w2,
        "w2_scale": scale_w2,
        "shared_w13": shared_weight_w13,
        "shared_w13_scale": shared_scale_w13,
        "native_shared_w13": native_shared_w13[0],
        "native_shared_w13_scale": native_shared_w13_scale[0],
        "shared_w2": shared_weight_w2,
        "shared_w2_scale": shared_scale_w2,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "shared_alpha1": shared_alpha1,
        "shared_alpha2": shared_alpha2,
        "input_global_scale": input_global_scale,
        "activation_global_scale": activation_global_scale,
    }
    _set_routing_case(tensors, ROUTING_CASES[0][1])
    return tensors


def _kernel_args(tensors: dict[str, torch.Tensor], shape: Shape) -> tuple:
    return (
        tensors["hidden"],
        tensors["router_weight"],
        tensors["correction_bias"],
        tensors["native_w13"],
        tensors["native_w13_scale"],
        tensors["w2"],
        tensors["w2_scale"],
        tensors["native_shared_w13"],
        tensors["native_shared_w13_scale"],
        tensors["shared_w2"],
        tensors["shared_w2_scale"],
        tensors["alpha1"],
        tensors["alpha2"],
        tensors["shared_alpha1"],
        tensors["shared_alpha2"],
        tensors["input_global_scale"],
        tensors["activation_global_scale"],
        shape.top_k,
        shape.num_groups,
        shape.topk_groups,
        shape.routed_scale,
    )


def _standalone_module():
    repo_root = str(Path(__file__).resolve().parents[3])
    inserted = repo_root not in sys.path
    if inserted:
        sys.path.insert(0, repo_root)
    try:
        from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4 import _standalone
    finally:
        if inserted:
            sys.path.remove(repo_root)
    return _standalone


def _make_standalone_call(tensors: dict[str, torch.Tensor], shape: Shape) -> tuple:
    return _standalone_module().build(tensors, shape)


def _prepare_production_weights(
    tensors: dict[str, torch.Tensor], shape: Shape
) -> tuple[torch.Tensor, ...]:
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )

    # ModelOpt checkpoints store [gate; up]; TRT-LLM consumes [up; gate].
    weight_w13 = torch.cat(
        (
            tensors["w13"][:, shape.intermediate :],
            tensors["w13"][:, : shape.intermediate],
        ),
        dim=1,
    )
    scale_w13 = torch.cat(
        (
            tensors["w13_scale"][:, shape.intermediate :],
            tensors["w13_scale"][:, : shape.intermediate],
        ),
        dim=1,
    )
    return prepare_static_weights_for_trtllm_fp4_moe(
        weight_w13,
        tensors["w2"],
        scale_w13,
        tensors["w2_scale"],
        hidden_size=shape.hidden,
        intermediate_size=shape.intermediate,
        num_experts=shape.num_experts,
        is_gated_activation=True,
    )


def _make_vllm_linear(
    weight: torch.Tensor,
    scale: torch.Tensor,
    output_size: int,
    input_scale: float,
    alpha: float,
):
    from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale.clone(), requires_grad=False)
    layer.output_size_per_partition = output_size
    layer.input_global_scale_inv = torch.nn.Parameter(
        torch.tensor(input_scale, device="cuda", dtype=torch.float32),
        requires_grad=False,
    )
    layer.alpha = torch.nn.Parameter(
        torch.tensor(alpha, device="cuda", dtype=torch.float32),
        requires_grad=False,
    )
    kernel = init_nvfp4_linear_kernel()
    kernel.process_weights_after_loading(layer)
    return kernel, layer


def _make_vllm_call(
    tensors: dict[str, torch.Tensor], shape: Shape
) -> tuple[object, str, torch.Tensor]:
    _ensure_vllm_importable()
    import flashinfer
    from flashinfer import autotune as flashinfer_autotune
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )

    weight_w13, scale_w13, weight_w2, scale_w2 = _prepare_production_weights(
        tensors, shape
    )
    output = torch.empty(shape.batch, shape.hidden, device="cuda", dtype=torch.bfloat16)
    routing_replay = torch.empty(
        shape.batch, shape.top_k, device="cuda", dtype=torch.int16
    )
    output1_scale_scalar = tensors["alpha1"] * tensors["activation_global_scale"]

    def routed() -> torch.Tensor:
        logits = torch.nn.functional.linear(tensors["hidden"], tensors["router_weight"])
        hidden_q, hidden_scale = ops.scaled_fp4_quant(
            tensors["hidden"],
            tensors["input_global_scale"],
            is_sf_swizzled_layout=False,
        )
        result = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            routing_logits=logits,
            routing_bias=tensors["correction_bias"],
            hidden_states=hidden_q,
            hidden_states_scale=hidden_scale,
            gemm1_weights=weight_w13,
            gemm1_weights_scale=scale_w13,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=weight_w2,
            gemm2_weights_scale=scale_w2,
            gemm2_bias=None,
            output1_scale_scalar=output1_scale_scalar,
            output1_scale_gate_scalar=tensors["alpha1"],
            output2_scale_scalar=tensors["alpha2"],
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            n_group=shape.num_groups,
            topk_group=shape.topk_groups,
            intermediate_size=shape.intermediate,
            local_expert_offset=0,
            local_num_experts=shape.num_experts,
            routed_scaling_factor=shape.routed_scale,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            do_finalize=True,
            activation_type=activation_to_flashinfer_int(MoEActivation.SILU),
            enable_pdl=False,
            tune_max_num_tokens=8192,
            output=output,
            routing_replay_out=routing_replay,
        )
        return result[0]

    shared_w13_kernel, shared_w13_layer = _make_vllm_linear(
        tensors["shared_w13"],
        tensors["shared_w13_scale"],
        2 * shape.intermediate,
        1.0,
        float(tensors["shared_alpha1"].item()),
    )
    shared_w2_kernel, shared_w2_layer = _make_vllm_linear(
        tensors["shared_w2"],
        tensors["shared_w2_scale"],
        shape.hidden,
        16.0,
        float(tensors["shared_alpha2"].item()),
    )
    shared_activation = torch.empty(
        shape.batch, shape.intermediate, device="cuda", dtype=torch.bfloat16
    )

    def shared() -> torch.Tensor:
        gate_up = shared_w13_kernel.apply_weights(shared_w13_layer, tensors["hidden"])
        torch.ops._C.silu_and_mul(shared_activation, gate_up)
        return shared_w2_kernel.apply_weights(shared_w2_layer, shared_activation)

    shared_stream = torch.cuda.Stream()

    def launch() -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        shared_stream.wait_stream(current_stream)
        with torch.cuda.stream(shared_stream):
            shared_output = shared()
        routed_output = routed()
        current_stream.wait_stream(shared_stream)
        return routed_output + shared_output

    # Production loads a preselected FlashInfer tactic. Avoid online profiling
    # in the correctness and timing paths.
    with flashinfer_autotune(False):
        launch()
    torch.cuda.synchronize()

    def launch_tuned() -> torch.Tensor:
        with flashinfer_autotune(False):
            return launch()

    return launch_tuned, "flashinfer_trtllm", routing_replay


def _assert_standalone_exact(
    persistent: tuple[torch.Tensor, ...],
    standalone: tuple[torch.Tensor, ...],
) -> None:
    if len(persistent) != len(OUTPUT_NAMES) or len(standalone) != len(OUTPUT_NAMES):
        raise AssertionError("NVFP4 output manifest does not match the kernel")
    for name, persistent_value, standalone_value in zip(
        OUTPUT_NAMES, persistent, standalone, strict=True
    ):
        if not torch.equal(persistent_value, standalone_value):
            difference = float(
                (persistent_value.float() - standalone_value.float()).abs().max().item()
            )
            raise AssertionError(
                f"{name} differs between persistent and standalone "
                f"(max_abs={difference})"
            )


def _similarity_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f64 = actual.double()
    expected_f64 = expected.double()
    denominator = (actual_f64.square() + expected_f64.square()).sum().clamp(min=1e-30)
    return float(1.0 - 2.0 * (actual_f64 * expected_f64).sum() / denominator)


def _validate_vllm(
    persistent: tuple[torch.Tensor, ...],
    vllm_output: torch.Tensor,
    routing_replay: torch.Tensor,
) -> None:
    torch.testing.assert_close(
        persistent[3],
        routing_replay.to(torch.int32),
        rtol=0,
        atol=0,
    )
    error = _similarity_error(persistent[0], vllm_output)
    if not math.isfinite(error) or error > 1e-3:
        raise AssertionError(f"vLLM similarity error {error:.6g} exceeds 1e-3")


def _dispatch_cache_signature(call: object) -> tuple[tuple[object, int], ...]:
    cache = getattr(call, "_dispatch_cache", None)
    if not cache:
        raise RuntimeError("AOT dispatch cache was not populated after launch")
    return tuple((key, id(bound)) for key, bound in cache.items())


def use_cudagraph() -> bool:
    return True


def has_vllm() -> bool:
    _ensure_vllm_importable()
    try:
        import flashinfer  # noqa: F401
        from vllm.model_executor.kernels.linear import (  # noqa: F401
            init_nvfp4_linear_kernel,
        )
        from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (  # noqa: F401
            prepare_static_weights_for_trtllm_fp4_moe,
        )
    except ImportError:
        return False
    return True


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("deepseek_v3_moe_nvfp4 is pretuned only for NVIDIA SM100")


@torch.inference_mode()
def correctness_check() -> None:
    """Check every intermediate against the matched Helion graph and vLLM."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError(
            "vLLM with FlashInfer is required for the DeepSeek NVFP4 comparison"
        )
    shape = Shape()
    tensors = _allocate(shape)
    standalone_call, _standalone_output = _make_standalone_call(tensors, shape)
    vllm_call, _backend, routing_replay = _make_vllm_call(tensors, shape)
    kernel_args = _kernel_args(tensors, shape)
    dispatch_signature = None
    for _label, selected_ids in ROUTING_CASES:
        _set_routing_case(tensors, selected_ids)
        persistent = deepseek_v3_moe_nvfp4(*kernel_args)
        current_signature = _dispatch_cache_signature(deepseek_v3_moe_nvfp4)
        if dispatch_signature is None:
            dispatch_signature = current_signature
        elif current_signature != dispatch_signature:
            raise AssertionError("runtime routing triggered a recompilation")
        standalone = standalone_call()
        vllm_output = vllm_call()
        torch.cuda.synchronize()
        _assert_standalone_exact(persistent, standalone)
        _validate_vllm(persistent, vllm_output, routing_replay)
        expected_ids = torch.tensor([selected_ids], device="cuda", dtype=torch.int32)
        torch.testing.assert_close(
            torch.sort(persistent[3], dim=-1).values,
            torch.sort(expected_ids, dim=-1).values,
            rtol=0,
            atol=0,
        )


@torch.inference_mode()
def main(verbose: bool = True) -> dict[str, Any]:
    """Benchmark persistent, matched standalone, and production vLLM."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError(
            "vLLM with FlashInfer is required for the DeepSeek NVFP4 comparison"
        )

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _bench import capture_cuda_graph
    from _bench import run_sweep

    shape = Shape()
    tensors = _allocate(shape)
    standalone_call, _standalone_output = _make_standalone_call(tensors, shape)
    vllm_call, backend, routing_replay = _make_vllm_call(tensors, shape)
    kernel_args = _kernel_args(tensors, shape)
    dispatch_signature = None

    def make_calls(case: tuple[str, tuple[int, ...]]) -> tuple:
        nonlocal dispatch_signature
        label, selected_ids = case
        _set_routing_case(tensors, selected_ids)
        persistent = deepseek_v3_moe_nvfp4(*kernel_args)
        current_signature = _dispatch_cache_signature(deepseek_v3_moe_nvfp4)
        if dispatch_signature is None:
            dispatch_signature = current_signature
        elif current_signature != dispatch_signature:
            raise AssertionError("runtime routing triggered a recompilation")
        standalone = standalone_call()
        vllm_output = vllm_call()
        torch.cuda.synchronize()
        _assert_standalone_exact(persistent, standalone)
        _validate_vllm(persistent, vllm_output, routing_replay)

        persistent_graph, _ = capture_cuda_graph(
            lambda: deepseek_v3_moe_nvfp4(*kernel_args)
        )
        standalone_graph, _ = capture_cuda_graph(standalone_call)
        vllm_graph, _ = capture_cuda_graph(vllm_call)
        return (
            persistent_graph.replay,
            [
                ("standalone_helion_pdl", standalone_graph.replay),
                (f"vllm_auto ({backend})", vllm_graph.replay),
            ],
            f"{label:>10s}  {selected_ids!s:>36s}",
        )

    return run_sweep(
        ROUTING_CASES,
        make_calls,
        use_cudagraph=False,
        pre_captured_cudagraph=True,
        thermal_warmup_ms=10_000,
        verbose=verbose,
        shape_header=f"{'routing':>10s}  {'expert_ids':>36s}",
    )


if __name__ == "__main__":
    main()
