# ruff: noqa: ANN001, ANN201, ANN202, E402, I001
"""DeepSeek-V3 decode-MoE FP8 megakernel, pretuned for NVIDIA B200.

This compares one complete TP=1, B=1 MoE region against an independently
tuned multi-launch Helion graph and vLLM's automatically selected production
implementation. All paths share serialized E4M3FN weights, K128 activation
quantization, and the same routing-through-final-add logical boundary.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

VLLM_ROOT = os.environ.get("VLLM_ROOT")
if VLLM_ROOT:
    VLLM_ROOT = str(Path(VLLM_ROOT).resolve())

import torch

import helion
import helion.language as hl


FP8_DTYPE = torch.float8_e4m3fn
FP8_GROUP = 128
FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10


@dataclasses.dataclass(frozen=True)
class Shape:
    batch: int = 1
    hidden: int = 7168
    intermediate: int = 2048
    num_experts: int = 256
    top_k: int = 8
    num_groups: int = 8
    topk_groups: int = 4
    routed_scale: float = 2.5


def _ensure_vllm_importable() -> None:
    """Honor VLLM_ROOT only when the optional production baseline is used."""
    if VLLM_ROOT is not None and VLLM_ROOT not in sys.path:
        sys.path.insert(0, VLLM_ROOT)


@helion.aot_kernel(static_shapes=True, backend="triton")
def deepseek_v3_moe_fp8(
    hidden,
    router_weight,
    correction_bias,
    expert_w13_q,
    expert_w13_scale,
    expert_w2_q,
    expert_w2_scale,
    shared_w13_q,
    shared_w13_scale,
    shared_w2_q,
    shared_w2_scale,
    top_k,
    num_groups,
    topk_groups,
    routed_scale,
):
    router_hidden = hidden
    router_weight = router_weight
    router_m, router_k = router_hidden.size()
    router_n, router_weight_k = router_weight.size()
    assert router_k == router_weight_k
    router_logits = torch.empty(
        (router_m, router_n),
        dtype=torch.float32,
        device=router_hidden.device,
    )
    topk_logits = router_logits
    topk_correction_bias = correction_bias
    topk_top_k = top_k
    topk_num_groups = num_groups
    topk_topk_groups = topk_groups
    topk_routed_scale = routed_scale
    topk_batch, topk_num_experts = topk_logits.size()
    topk_top_k = hl.specialize(topk_top_k)
    topk_num_groups = hl.specialize(topk_num_groups)
    topk_topk_groups = hl.specialize(topk_topk_groups)
    assert topk_top_k == 8 and topk_topk_groups == 4
    topk_experts_per_group = topk_num_experts // topk_num_groups
    hl.specialize(topk_batch)
    hl.specialize(topk_num_experts)
    topk_weights = torch.empty(
        (topk_batch, topk_top_k),
        dtype=torch.float32,
        device=topk_logits.device,
    )
    topk_ids = torch.empty(
        (topk_batch, topk_top_k),
        dtype=torch.int32,
        device=topk_logits.device,
    )
    routed_input_quant_input = hidden
    routed_input_quant_rows, routed_input_quant_columns = routed_input_quant_input.shape
    routed_input_quant_groups = routed_input_quant_columns // 128
    hl.specialize(routed_input_quant_columns)
    assert routed_input_quant_columns == routed_input_quant_groups * 128
    routed_input_q = torch.empty_like(routed_input_quant_input, dtype=FP8_DTYPE)
    routed_input_scale = torch.empty(
        (routed_input_quant_rows, routed_input_quant_groups),
        dtype=torch.float32,
        device=routed_input_quant_input.device,
    )
    routed_input_quant_input_3d = routed_input_quant_input.view(
        routed_input_quant_rows, routed_input_quant_groups, 128
    )
    routed_input_quant_output_3d = routed_input_q.view(
        routed_input_quant_rows, routed_input_quant_groups, 128
    )
    shared_input_quant_input = hidden
    shared_input_quant_rows, shared_input_quant_columns = shared_input_quant_input.shape
    shared_input_quant_groups = shared_input_quant_columns // 128
    hl.specialize(shared_input_quant_columns)
    assert shared_input_quant_columns == shared_input_quant_groups * 128
    shared_input_q = torch.empty_like(shared_input_quant_input, dtype=FP8_DTYPE)
    shared_input_scale = torch.empty(
        (shared_input_quant_rows, shared_input_quant_groups),
        dtype=torch.float32,
        device=shared_input_quant_input.device,
    )
    shared_input_quant_input_3d = shared_input_quant_input.view(
        shared_input_quant_rows, shared_input_quant_groups, 128
    )
    shared_input_quant_output_3d = shared_input_q.view(
        shared_input_quant_rows, shared_input_quant_groups, 128
    )
    shared_w13_input_q = shared_input_q
    shared_w13_input_scale = shared_input_scale
    shared_w13_weight_q = shared_w13_q
    shared_w13_weight_scale = shared_w13_scale
    shared_w13_rows, shared_w13_reduction = shared_w13_input_q.shape
    shared_w13_output_size, shared_w13_weight_reduction = shared_w13_weight_q.shape
    assert shared_w13_reduction == shared_w13_weight_reduction
    hl.specialize(shared_w13_output_size)
    hl.specialize(shared_w13_reduction)
    shared_gate_up = torch.empty(
        (shared_w13_rows, shared_w13_output_size),
        dtype=torch.bfloat16,
        device=shared_w13_input_q.device,
    )
    expert_w13_input_q = routed_input_q
    expert_w13_input_scale = routed_input_scale
    expert_w13_weight_q = expert_w13_q
    expert_w13_weight_scale = expert_w13_scale
    expert_w13_topk_ids = topk_ids
    expert_w13_input_rows, expert_w13_reduction = expert_w13_input_q.shape
    (
        expert_w13_num_experts,
        expert_w13_output_size,
        expert_w13_weight_reduction,
    ) = expert_w13_weight_q.shape
    expert_w13_top_k = expert_w13_topk_ids.shape[1]
    assert expert_w13_reduction == expert_w13_weight_reduction
    assert expert_w13_input_rows == 1
    hl.specialize(expert_w13_num_experts)
    hl.specialize(expert_w13_output_size)
    hl.specialize(expert_w13_reduction)
    hl.specialize(expert_w13_top_k)
    expert_gate_up = torch.empty(
        (expert_w13_top_k, expert_w13_output_size),
        dtype=torch.bfloat16,
        device=expert_w13_input_q.device,
    )
    expert_w13_flat_ids = expert_w13_topk_ids.view(expert_w13_top_k)
    expert_w13_flat_weight = expert_w13_weight_q.view(
        expert_w13_num_experts * expert_w13_output_size,
        expert_w13_reduction,
    )
    expert_w13_flat_scale = expert_w13_weight_scale.view(
        expert_w13_num_experts * expert_w13_output_size // 128,
        expert_w13_reduction // 128,
    )
    expert_w13_flat_output = expert_gate_up.view(
        expert_w13_top_k * expert_w13_output_size, 1
    )
    expert_act_quant_gate_up = expert_gate_up
    expert_act_quant_rows, expert_act_quant_twice_intermediate = (
        expert_act_quant_gate_up.shape
    )
    expert_act_quant_intermediate = expert_act_quant_twice_intermediate // 2
    expert_act_quant_groups = expert_act_quant_intermediate // 128
    hl.specialize(expert_act_quant_intermediate)
    expert_act_quant_gate_up_grouped = expert_act_quant_gate_up.view(
        expert_act_quant_rows, 2, expert_act_quant_groups, 128
    ).permute(0, 2, 1, 3)
    expert_activation_q = torch.empty(
        (expert_act_quant_rows, expert_act_quant_intermediate),
        dtype=FP8_DTYPE,
        device=expert_act_quant_gate_up.device,
    )
    expert_activation_scale = torch.empty(
        (expert_act_quant_rows, expert_act_quant_groups),
        dtype=torch.float32,
        device=expert_act_quant_gate_up.device,
    )
    expert_act_quant_output_3d = expert_activation_q.view(
        expert_act_quant_rows, expert_act_quant_groups, 128
    )
    shared_act_quant_gate_up = shared_gate_up
    shared_act_quant_rows, shared_act_quant_twice_intermediate = (
        shared_act_quant_gate_up.shape
    )
    shared_act_quant_intermediate = shared_act_quant_twice_intermediate // 2
    shared_act_quant_groups = shared_act_quant_intermediate // 128
    hl.specialize(shared_act_quant_intermediate)
    shared_activation_q = torch.empty(
        (shared_act_quant_rows, shared_act_quant_intermediate),
        dtype=FP8_DTYPE,
        device=shared_act_quant_gate_up.device,
    )
    shared_activation_scale = torch.empty(
        (shared_act_quant_rows, shared_act_quant_groups),
        dtype=torch.float32,
        device=shared_act_quant_gate_up.device,
    )
    shared_act_quant_output_3d = shared_activation_q.view(
        shared_act_quant_rows, shared_act_quant_groups, 128
    )
    shared_w2_input_q = shared_activation_q
    shared_w2_input_scale = shared_activation_scale
    shared_w2_weight_q = shared_w2_q
    shared_w2_weight_scale = shared_w2_scale
    shared_w2_rows, shared_w2_reduction = shared_w2_input_q.shape
    shared_w2_output_size, shared_w2_weight_reduction = shared_w2_weight_q.shape
    assert shared_w2_reduction == shared_w2_weight_reduction
    hl.specialize(shared_w2_output_size)
    hl.specialize(shared_w2_reduction)
    shared_output = torch.empty(
        (shared_w2_rows, shared_w2_output_size),
        dtype=torch.bfloat16,
        device=shared_w2_input_q.device,
    )
    expert_w2_input_q = expert_activation_q
    expert_w2_input_scale = expert_activation_scale
    expert_w2_weight_q = expert_w2_q
    expert_w2_weight_scale = expert_w2_scale
    expert_w2_topk_ids = topk_ids
    expert_w2_top_k, expert_w2_reduction = expert_w2_input_q.shape
    (
        expert_w2_num_experts,
        expert_w2_output_size,
        expert_w2_weight_reduction,
    ) = expert_w2_weight_q.shape
    assert expert_w2_reduction == expert_w2_weight_reduction
    hl.specialize(expert_w2_num_experts)
    hl.specialize(expert_w2_output_size)
    hl.specialize(expert_w2_reduction)
    hl.specialize(expert_w2_top_k)
    expert_outputs = torch.empty(
        (expert_w2_top_k, expert_w2_output_size),
        dtype=torch.bfloat16,
        device=expert_w2_input_q.device,
    )
    expert_w2_flat_ids = expert_w2_topk_ids.view(expert_w2_top_k)
    expert_w2_flat_scale = expert_w2_weight_scale.view(
        expert_w2_num_experts * (expert_w2_output_size // 128),
        expert_w2_reduction // 128,
    )
    expert_reduce_expert_outputs = expert_outputs
    expert_reduce_topk_weights = topk_weights
    expert_reduce_top_k, expert_reduce_hidden = expert_reduce_expert_outputs.shape
    expert_reduce_top_k = hl.specialize(expert_reduce_top_k)
    routed_output = torch.empty(
        (1, expert_reduce_hidden),
        dtype=expert_reduce_expert_outputs.dtype,
        device=expert_reduce_expert_outputs.device,
    )
    final_add_routed = routed_output
    final_add_shared = shared_output
    final_add_rows, final_add_columns = final_add_routed.shape
    output = torch.empty_like(final_add_routed)
    for router_tile_m, router_tile_n in hl.tile(
        [router_m, router_n], block_size=[1, None]
    ):
        router_acc = hl.zeros([router_tile_m, router_tile_n], dtype=torch.float32)
        for router_tile_k in hl.tile(router_k):
            router_acc = torch.addmm(
                router_acc,
                router_hidden[router_tile_m, router_tile_k],
                router_weight[router_tile_n, router_tile_k].T,
            )
        router_logits[router_tile_m, router_tile_n] = router_acc
    for _topk_program in hl.grid(1):
        topk_scores = torch.sigmoid(topk_logits[:, :])
        topk_biased = topk_scores + topk_correction_bias[None, :]
        topk_grouped = topk_biased.view(
            topk_batch,
            topk_num_groups,
            topk_experts_per_group,
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
                torch.full_like(
                    topk_within_group_indices,
                    topk_experts_per_group,
                ),
            ),
            dim=-1,
        )
        topk_group_without_best = torch.where(
            topk_within_group_indices == topk_group_best_id_1[:, :, None],
            topk_negative_infinity,
            topk_grouped,
        )
        topk_group_best_2 = torch.amax(topk_group_without_best, dim=-1)
        topk_group_scores = (
            topk_group_best_1.view(topk_batch, topk_num_groups) + topk_group_best_2
        )
        topk_group_indices = hl.arange(topk_num_groups)[None, :].to(torch.int32)
        topk_group_max_1 = torch.amax(topk_group_scores, dim=-1, keepdim=True)
        topk_group_id_1 = torch.amin(
            torch.where(
                topk_group_scores == topk_group_max_1,
                topk_group_indices,
                torch.full_like(topk_group_indices, topk_num_groups),
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
                torch.full_like(topk_group_indices, topk_num_groups),
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
                torch.full_like(topk_group_indices, topk_num_groups),
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
                torch.full_like(topk_group_indices, topk_num_groups),
            ),
            dim=-1,
        )
        topk_allowed = (
            (topk_group_indices == topk_group_id_1[:, None])
            | (topk_group_indices == topk_group_id_2[:, None])
            | (topk_group_indices == topk_group_id_3[:, None])
            | (topk_group_indices == topk_group_id_4[:, None])
        ).view(topk_batch, topk_num_groups, 1)
        topk_masked = torch.where(
            topk_allowed,
            topk_grouped,
            topk_negative_infinity,
        ).view(topk_batch, topk_num_experts)
        topk_expert_indices = hl.arange(topk_num_experts)[None, :].to(torch.int32)
        topk_masked_1 = topk_masked
        topk_value_0 = torch.amax(topk_masked_1, dim=-1, keepdim=True)
        topk_candidates_0 = topk_masked_1 == topk_value_0
        topk_id_0 = torch.amin(
            torch.where(
                topk_candidates_0,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_selected_0 = topk_expert_indices == topk_id_0[:, None]
        topk_masked_2 = torch.where(
            topk_selected_0,
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_1,
        )
        topk_value_1 = torch.amax(topk_masked_2, dim=-1, keepdim=True)
        topk_candidates_1 = topk_masked_2 == topk_value_1
        topk_id_1 = torch.amin(
            torch.where(
                topk_candidates_1,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_selected_1 = topk_expert_indices == topk_id_1[:, None]
        topk_masked_3 = torch.where(
            topk_selected_1,
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_2,
        )
        topk_value_2 = torch.amax(topk_masked_3, dim=-1, keepdim=True)
        topk_candidates_2 = topk_masked_3 == topk_value_2
        topk_id_2 = torch.amin(
            torch.where(
                topk_candidates_2,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_selected_2 = topk_expert_indices == topk_id_2[:, None]
        topk_masked_4 = torch.where(
            topk_selected_2,
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_3,
        )
        topk_value_3 = torch.amax(topk_masked_4, dim=-1, keepdim=True)
        topk_candidates_3 = topk_masked_4 == topk_value_3
        topk_id_3 = torch.amin(
            torch.where(
                topk_candidates_3,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_selected_3 = topk_expert_indices == topk_id_3[:, None]
        topk_masked_5 = torch.where(
            topk_selected_3,
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_4,
        )
        topk_value_4 = torch.amax(topk_masked_5, dim=-1, keepdim=True)
        topk_candidates_4 = topk_masked_5 == topk_value_4
        topk_id_4 = torch.amin(
            torch.where(
                topk_candidates_4,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_selected_4 = topk_expert_indices == topk_id_4[:, None]
        topk_masked_6 = torch.where(
            topk_selected_4,
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_5,
        )
        topk_value_5 = torch.amax(topk_masked_6, dim=-1, keepdim=True)
        topk_candidates_5 = topk_masked_6 == topk_value_5
        topk_id_5 = torch.amin(
            torch.where(
                topk_candidates_5,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_selected_5 = topk_expert_indices == topk_id_5[:, None]
        topk_masked_7 = torch.where(
            topk_selected_5,
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_6,
        )
        topk_value_6 = torch.amax(topk_masked_7, dim=-1, keepdim=True)
        topk_candidates_6 = topk_masked_7 == topk_value_6
        topk_id_6 = torch.amin(
            torch.where(
                topk_candidates_6,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        topk_selected_6 = topk_expert_indices == topk_id_6[:, None]
        topk_masked_8 = torch.where(
            topk_selected_6,
            torch.full_like(topk_masked, float("-inf")),
            topk_masked_7,
        )
        topk_value_7 = torch.amax(topk_masked_8, dim=-1, keepdim=True)
        topk_candidates_7 = topk_masked_8 == topk_value_7
        topk_id_7 = torch.amin(
            torch.where(
                topk_candidates_7,
                topk_expert_indices,
                torch.full_like(topk_expert_indices, topk_num_experts),
            ),
            dim=-1,
        )
        # The selected IDs are unique, so load their scores directly instead of
        # running eight full-width one-hot reductions over all experts.
        topk_weight_0 = torch.sigmoid(torch.sum(topk_logits[:, topk_id_0], dim=-1))
        topk_weight_1 = torch.sigmoid(torch.sum(topk_logits[:, topk_id_1], dim=-1))
        topk_weight_2 = torch.sigmoid(torch.sum(topk_logits[:, topk_id_2], dim=-1))
        topk_weight_3 = torch.sigmoid(torch.sum(topk_logits[:, topk_id_3], dim=-1))
        topk_weight_4 = torch.sigmoid(torch.sum(topk_logits[:, topk_id_4], dim=-1))
        topk_weight_5 = torch.sigmoid(torch.sum(topk_logits[:, topk_id_5], dim=-1))
        topk_weight_6 = torch.sigmoid(torch.sum(topk_logits[:, topk_id_6], dim=-1))
        topk_weight_7 = torch.sigmoid(torch.sum(topk_logits[:, topk_id_7], dim=-1))
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
        topk_weights[:, 0] = topk_weight_0 / topk_denominator * topk_routed_scale
        topk_weights[:, 1] = topk_weight_1 / topk_denominator * topk_routed_scale
        topk_weights[:, 2] = topk_weight_2 / topk_denominator * topk_routed_scale
        topk_weights[:, 3] = topk_weight_3 / topk_denominator * topk_routed_scale
        topk_weights[:, 4] = topk_weight_4 / topk_denominator * topk_routed_scale
        topk_weights[:, 5] = topk_weight_5 / topk_denominator * topk_routed_scale
        topk_weights[:, 6] = topk_weight_6 / topk_denominator * topk_routed_scale
        topk_weights[:, 7] = topk_weight_7 / topk_denominator * topk_routed_scale
        topk_ids[:, 0] = topk_id_0
        topk_ids[:, 1] = topk_id_1
        topk_ids[:, 2] = topk_id_2
        topk_ids[:, 3] = topk_id_3
        topk_ids[:, 4] = topk_id_4
        topk_ids[:, 5] = topk_id_5
        topk_ids[:, 6] = topk_id_6
        topk_ids[:, 7] = topk_id_7
    for (
        routed_input_quant_tile_r,
        routed_input_quant_tile_g,
        routed_input_quant_tile_k,
    ) in hl.tile(
        [routed_input_quant_rows, routed_input_quant_groups, 128],
        block_size=[1, None, 128],
    ):
        routed_input_quant_values = routed_input_quant_input_3d[
            routed_input_quant_tile_r,
            routed_input_quant_tile_g,
            routed_input_quant_tile_k,
        ].to(torch.float32)
        routed_input_quant_scale = (
            torch.amax(torch.abs(routed_input_quant_values), dim=-1).clamp(min=FP8_EPS)
            / FP8_MAX
        )
        routed_input_quant_scale = torch.exp2(
            torch.ceil(torch.log2(routed_input_quant_scale))
        )
        routed_input_scale[routed_input_quant_tile_r, routed_input_quant_tile_g] = (
            routed_input_quant_scale
        )
        routed_input_quant_output_3d[
            routed_input_quant_tile_r,
            routed_input_quant_tile_g,
            routed_input_quant_tile_k,
        ] = (
            (routed_input_quant_values / routed_input_quant_scale[:, :, None])
            .clamp(FP8_MIN, FP8_MAX)
            .to(FP8_DTYPE)
        )
    for (
        shared_input_quant_tile_r,
        shared_input_quant_tile_g,
        shared_input_quant_tile_k,
    ) in hl.tile(
        [shared_input_quant_rows, shared_input_quant_groups, 128],
        block_size=[1, None, 128],
    ):
        shared_input_quant_values = shared_input_quant_input_3d[
            shared_input_quant_tile_r,
            shared_input_quant_tile_g,
            shared_input_quant_tile_k,
        ].to(torch.float32)
        shared_input_quant_scale = (
            torch.amax(torch.abs(shared_input_quant_values), dim=-1).clamp(min=FP8_EPS)
            / FP8_MAX
        )
        shared_input_quant_scale = torch.exp2(
            torch.ceil(torch.log2(shared_input_quant_scale))
        )
        shared_input_scale[shared_input_quant_tile_r, shared_input_quant_tile_g] = (
            shared_input_quant_scale
        )
        shared_input_quant_output_3d[
            shared_input_quant_tile_r,
            shared_input_quant_tile_g,
            shared_input_quant_tile_k,
        ] = (
            (shared_input_quant_values / shared_input_quant_scale[:, :, None])
            .clamp(FP8_MIN, FP8_MAX)
            .to(FP8_DTYPE)
        )
    for shared_w13_tile_r, shared_w13_tile_n in hl.tile(
        [shared_w13_rows, shared_w13_output_size],
        block_size=[1, None],
    ):
        shared_w13_accumulator = hl.zeros(
            [shared_w13_tile_r, shared_w13_tile_n],
            dtype=torch.float32,
        )
        for shared_w13_tile_k in hl.tile(shared_w13_reduction, block_size=128):
            shared_w13_partial = hl.dot(
                shared_w13_input_q[shared_w13_tile_r, shared_w13_tile_k],
                shared_w13_weight_q[shared_w13_tile_n, shared_w13_tile_k].T,
            ).to(torch.float32)
            shared_w13_activation_scale = shared_w13_input_scale[
                shared_w13_tile_r, shared_w13_tile_k.id
            ].to(torch.float32)
            shared_w13_block_scale = shared_w13_weight_scale[
                shared_w13_tile_n.index // 128, shared_w13_tile_k.id
            ].to(torch.float32)
            shared_w13_accumulator += (
                shared_w13_partial
                * shared_w13_activation_scale[:, None]
                * shared_w13_block_scale[None, :]
            )
        shared_gate_up[shared_w13_tile_r, shared_w13_tile_n] = (
            shared_w13_accumulator.to(shared_gate_up.dtype)
        )
    for expert_w13_tile_row in hl.tile(expert_w13_top_k * expert_w13_output_size):
        expert_w13_expert_slot = expert_w13_tile_row.index // expert_w13_output_size
        expert_w13_expert_row = expert_w13_tile_row.index % expert_w13_output_size
        expert_w13_expert = expert_w13_flat_ids[expert_w13_expert_slot].to(torch.int64)
        expert_w13_selected_row = (
            expert_w13_expert * expert_w13_output_size + expert_w13_expert_row
        )
        expert_w13_accumulator = hl.zeros([expert_w13_tile_row, 1], dtype=torch.float32)
        for expert_w13_tile_k in hl.tile(expert_w13_reduction, block_size=128):
            expert_w13_partial = hl.dot(
                expert_w13_flat_weight[expert_w13_selected_row, expert_w13_tile_k],
                expert_w13_input_q[:, expert_w13_tile_k].T,
            ).to(torch.float32)
            expert_w13_activation_scale = expert_w13_input_scale[
                :, expert_w13_tile_k.id
            ].to(torch.float32)
            expert_w13_scale_row = (
                expert_w13_expert * (expert_w13_output_size // 128)
                + expert_w13_expert_row // 128
            )
            expert_w13_block_scale = expert_w13_flat_scale[
                expert_w13_scale_row, expert_w13_tile_k.id
            ].to(torch.float32)
            expert_w13_accumulator += (
                expert_w13_partial
                * expert_w13_activation_scale
                * expert_w13_block_scale[:, None]
            )
        expert_w13_flat_output[expert_w13_tile_row, :] = expert_w13_accumulator.to(
            expert_gate_up.dtype
        )
    for (
        expert_act_quant_tile_r,
        expert_act_quant_tile_g,
        expert_act_quant_tile_k,
    ) in hl.tile(
        [expert_act_quant_rows, expert_act_quant_groups, 128],
        block_size=[1, None, 128],
    ):
        expert_act_quant_gate = expert_act_quant_gate_up_grouped[
            expert_act_quant_tile_r,
            expert_act_quant_tile_g,
            0,
            expert_act_quant_tile_k,
        ].to(torch.float32)
        expert_act_quant_up = expert_act_quant_gate_up_grouped[
            expert_act_quant_tile_r,
            expert_act_quant_tile_g,
            1,
            expert_act_quant_tile_k,
        ].to(torch.float32)
        expert_act_quant_activated = (
            expert_act_quant_gate
            * torch.sigmoid(expert_act_quant_gate)
            * expert_act_quant_up
        )
        expert_act_quant_scale = (
            torch.amax(torch.abs(expert_act_quant_activated), dim=-1).clamp(min=FP8_EPS)
            / FP8_MAX
        )
        expert_act_quant_scale = torch.exp2(
            torch.ceil(torch.log2(expert_act_quant_scale))
        )
        expert_activation_scale[expert_act_quant_tile_r, expert_act_quant_tile_g] = (
            expert_act_quant_scale
        )
        expert_act_quant_output_3d[
            expert_act_quant_tile_r,
            expert_act_quant_tile_g,
            expert_act_quant_tile_k,
        ] = (
            (expert_act_quant_activated / expert_act_quant_scale[:, :, None])
            .clamp(FP8_MIN, FP8_MAX)
            .to(FP8_DTYPE)
        )
    for (
        shared_act_quant_tile_r,
        shared_act_quant_tile_g,
        shared_act_quant_tile_k,
    ) in hl.tile(
        [shared_act_quant_rows, shared_act_quant_groups, 128],
        block_size=[1, None, 128],
    ):
        shared_act_quant_indices = (
            shared_act_quant_tile_g.index[:, None] * 128
            + shared_act_quant_tile_k.index[None, :]
        )
        shared_act_quant_gate = shared_act_quant_gate_up[
            shared_act_quant_tile_r, shared_act_quant_indices
        ].to(torch.float32)
        shared_act_quant_up = shared_act_quant_gate_up[
            shared_act_quant_tile_r,
            shared_act_quant_indices + shared_act_quant_intermediate,
        ].to(torch.float32)
        shared_act_quant_activated = (
            shared_act_quant_gate
            * torch.sigmoid(shared_act_quant_gate)
            * shared_act_quant_up
        )
        shared_act_quant_scale = (
            torch.amax(torch.abs(shared_act_quant_activated), dim=-1).clamp(min=FP8_EPS)
            / FP8_MAX
        )
        shared_act_quant_scale = torch.exp2(
            torch.ceil(torch.log2(shared_act_quant_scale))
        )
        shared_activation_scale[shared_act_quant_tile_r, shared_act_quant_tile_g] = (
            shared_act_quant_scale
        )
        shared_act_quant_output_3d[
            shared_act_quant_tile_r,
            shared_act_quant_tile_g,
            shared_act_quant_tile_k,
        ] = (
            (shared_act_quant_activated / shared_act_quant_scale[:, :, None])
            .clamp(FP8_MIN, FP8_MAX)
            .to(FP8_DTYPE)
        )
    for shared_w2_tile_r, shared_w2_tile_n in hl.tile(
        [shared_w2_rows, shared_w2_output_size],
        block_size=[1, None],
    ):
        shared_w2_accumulator = hl.zeros(
            [shared_w2_tile_r, shared_w2_tile_n],
            dtype=torch.float32,
        )
        for shared_w2_tile_k in hl.tile(shared_w2_reduction, block_size=128):
            shared_w2_partial = hl.dot(
                shared_w2_input_q[shared_w2_tile_r, shared_w2_tile_k],
                shared_w2_weight_q[shared_w2_tile_n, shared_w2_tile_k].T,
            ).to(torch.float32)
            shared_w2_activation_scale = shared_w2_input_scale[
                shared_w2_tile_r, shared_w2_tile_k.id
            ].to(torch.float32)
            shared_w2_block_scale = shared_w2_weight_scale[
                shared_w2_tile_n.index // 128, shared_w2_tile_k.id
            ].to(torch.float32)
            shared_w2_accumulator += (
                shared_w2_partial
                * shared_w2_activation_scale[:, None]
                * shared_w2_block_scale[None, :]
            )
        shared_output[shared_w2_tile_r, shared_w2_tile_n] = shared_w2_accumulator.to(
            shared_output.dtype
        )
    for expert_w2_tile_slot, expert_w2_tile_n in hl.tile(
        [expert_w2_top_k, expert_w2_output_size],
        block_size=[1, None],
    ):
        expert_w2_expert = expert_w2_flat_ids[expert_w2_tile_slot.begin].to(torch.int64)
        expert_w2_accumulator = hl.zeros(
            [expert_w2_tile_slot, expert_w2_tile_n],
            dtype=torch.float32,
        )
        for expert_w2_tile_k in hl.tile(expert_w2_reduction, block_size=128):
            expert_w2_partial = hl.dot(
                expert_w2_input_q[expert_w2_tile_slot, expert_w2_tile_k],
                expert_w2_weight_q[
                    expert_w2_expert,
                    expert_w2_tile_n,
                    expert_w2_tile_k,
                ].T,
            ).to(torch.float32)
            expert_w2_activation_scale = expert_w2_input_scale[
                expert_w2_tile_slot, expert_w2_tile_k.id
            ].to(torch.float32)
            expert_w2_scale_row = (
                expert_w2_expert * (expert_w2_output_size // 128)
                + expert_w2_tile_n.index // 128
            )
            expert_w2_block_scale = expert_w2_flat_scale[
                expert_w2_scale_row, expert_w2_tile_k.id
            ].to(torch.float32)
            expert_w2_accumulator += (
                expert_w2_partial
                * expert_w2_activation_scale[:, None]
                * expert_w2_block_scale[None, :]
            )
        expert_outputs[expert_w2_tile_slot, expert_w2_tile_n] = (
            expert_w2_accumulator.to(expert_outputs.dtype)
        )
    for expert_reduce_tile_n in hl.tile(expert_reduce_hidden):
        expert_reduce_values = expert_reduce_expert_outputs[:, expert_reduce_tile_n].to(
            torch.float32
        )
        expert_reduce_weights = expert_reduce_topk_weights[:, :].view(
            expert_reduce_top_k
        )
        routed_output[:, expert_reduce_tile_n] = torch.sum(
            expert_reduce_values * expert_reduce_weights[:, None],
            dim=0,
            keepdim=True,
        ).to(routed_output.dtype)
    for final_add_tile_r, final_add_tile_n in hl.tile(
        [final_add_rows, final_add_columns], block_size=[1, None]
    ):
        output[final_add_tile_r, final_add_tile_n] = (
            final_add_routed[final_add_tile_r, final_add_tile_n]
            + final_add_shared[final_add_tile_r, final_add_tile_n]
        )
    return (
        output,
        router_logits,
        topk_weights,
        topk_ids,
        routed_input_q,
        routed_input_scale,
        expert_gate_up,
        expert_activation_q,
        expert_activation_scale,
        expert_outputs,
        routed_output,
        shared_input_q,
        shared_input_scale,
        shared_gate_up,
        shared_activation_q,
        shared_activation_scale,
        shared_output,
    )


PERSISTENT_OUTPUTS = (
    "output",
    "router_logits",
    "topk_weights",
    "topk_ids",
    "routed_input_q",
    "routed_input_scale",
    "expert_gate_up",
    "expert_activation_q",
    "expert_activation_scale",
    "expert_outputs",
    "routed_output",
    "shared_input_q",
    "shared_input_scale",
    "shared_gate_up",
    "shared_activation_q",
    "shared_activation_scale",
    "shared_output",
)


def _ceil_pow2_scale(values: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    maximum = values.abs().float().amax(dim=dims).clamp(min=FP8_EPS)
    return torch.exp2(torch.ceil(torch.log2(maximum / FP8_MAX)))


def _quantize_weight(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = source.shape
    assert rows % FP8_GROUP == 0 and columns % FP8_GROUP == 0
    blocks = source.view(rows // FP8_GROUP, FP8_GROUP, columns // FP8_GROUP, FP8_GROUP)
    scale = _ceil_pow2_scale(blocks, (1, 3))
    quantized = (blocks / scale[:, None, :, None]).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    return quantized.view(rows, columns).contiguous(), scale.contiguous()


def _make_weight(
    rows: int, columns: int, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16) * scale
    return _quantize_weight(source)


def _reference_grouped_topk(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    num_groups: int,
    topk_groups: int,
    routed_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic noaux_tc oracle with lowest-index tie breaking."""
    scores = torch.sigmoid(logits.float())
    biased = scores + correction_bias.float()[None, :]
    batch, num_experts = biased.shape
    experts_per_group = num_experts // num_groups
    grouped = biased.view(batch, num_groups, experts_per_group)
    within_group_order = torch.argsort(grouped, dim=-1, descending=True, stable=True)
    group_scores = torch.gather(grouped, -1, within_group_order[..., :2]).sum(-1)
    group_ids = torch.argsort(group_scores, dim=-1, descending=True, stable=True)[
        ..., :topk_groups
    ]
    allowed_groups = torch.zeros_like(group_scores, dtype=torch.bool)
    allowed_groups.scatter_(1, group_ids, True)
    allowed = (
        allowed_groups[:, :, None]
        .expand(batch, num_groups, experts_per_group)
        .reshape(batch, num_experts)
    )
    masked = biased.masked_fill(~allowed, float("-inf"))
    ids = torch.argsort(masked, dim=-1, descending=True, stable=True)[..., :top_k]
    weights = torch.gather(scores, 1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True) * routed_scale
    return weights.float(), ids.to(torch.int32)


def _allocate(shape: Shape, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    hidden = torch.randn(shape.batch, shape.hidden, device="cuda", dtype=torch.bfloat16)
    router_weight = torch.randn(
        shape.num_experts, shape.hidden, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(shape.hidden)
    correction_bias = (
        torch.randn(shape.num_experts, device="cuda", dtype=torch.float32) * 0.1
    )
    reference_weights, reference_ids = _reference_grouped_topk(
        hidden.float() @ router_weight.float().T,
        correction_bias,
        shape.top_k,
        shape.num_groups,
        shape.topk_groups,
        shape.routed_scale,
    )
    selected = tuple(sorted(set(reference_ids.flatten().tolist())))
    if len(selected) != shape.top_k:
        raise AssertionError("batch-one routing must select top_k distinct experts")

    zero_scale = 2.0**-13
    expert_w13_q = torch.zeros(
        shape.num_experts,
        2 * shape.intermediate,
        shape.hidden,
        device="cuda",
        dtype=FP8_DTYPE,
    )
    expert_w13_scale = torch.full(
        (
            shape.num_experts,
            2 * shape.intermediate // FP8_GROUP,
            shape.hidden // FP8_GROUP,
        ),
        zero_scale,
        device="cuda",
        dtype=torch.float32,
    )
    expert_w2_q = torch.zeros(
        shape.num_experts,
        shape.hidden,
        shape.intermediate,
        device="cuda",
        dtype=FP8_DTYPE,
    )
    expert_w2_scale = torch.full(
        (shape.num_experts, shape.hidden // FP8_GROUP, shape.intermediate // FP8_GROUP),
        zero_scale,
        device="cuda",
        dtype=torch.float32,
    )
    for expert in selected:
        q, scale = _make_weight(
            2 * shape.intermediate, shape.hidden, 1.0 / math.sqrt(shape.hidden)
        )
        expert_w13_q[expert].copy_(q)
        expert_w13_scale[expert].copy_(scale)
        del q, scale
        q, scale = _make_weight(
            shape.hidden, shape.intermediate, 1.0 / math.sqrt(shape.intermediate)
        )
        expert_w2_q[expert].copy_(q)
        expert_w2_scale[expert].copy_(scale)
        del q, scale
    shared_w13_q, shared_w13_scale = _make_weight(
        2 * shape.intermediate, shape.hidden, 1.0 / math.sqrt(shape.hidden)
    )
    shared_w2_q, shared_w2_scale = _make_weight(
        shape.hidden, shape.intermediate, 1.0 / math.sqrt(shape.intermediate)
    )
    return {
        "hidden": hidden,
        "router_weight": router_weight,
        "correction_bias": correction_bias,
        "reference_topk_weights": reference_weights,
        "reference_topk_ids": reference_ids,
        "expert_w13_q": expert_w13_q,
        "expert_w13_scale": expert_w13_scale,
        "expert_w2_q": expert_w2_q,
        "expert_w2_scale": expert_w2_scale,
        "shared_w13_q": shared_w13_q,
        "shared_w13_scale": shared_w13_scale,
        "shared_w2_q": shared_w2_q,
        "shared_w2_scale": shared_w2_scale,
    }


def _initialize_vllm():
    _ensure_vllm_importable()
    from vllm.config import (
        CacheConfig,
        CUDAGraphMode,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.utils.network_utils import get_open_port
    from vllm.v1.worker.workspace import init_workspace_manager

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[FP8_GROUP, FP8_GROUP],
    )
    cache_config = CacheConfig(block_size=16, cache_dtype="auto")
    cache_config.num_gpu_blocks = 1
    vllm_config = VllmConfig(cache_config=cache_config, quant_config=quant_config)
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(model_type="deepseek_v3"),
        is_moe=True,
        is_mm_prefix_lm=False,
        is_diffusion=False,
        is_hybrid=False,
        is_attention_free=False,
        runner_type="generate",
        architectures=["DeepseekV3ForCausalLM"],
        max_model_len=4096,
        compute_hash=lambda: "deepseek-v3-moe-fp8-benchmark",
    )
    init_workspace_manager(torch.device("cuda"))
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=torch.cuda.current_device(),
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
    )
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(1, 1)
    return vllm_config, quant_config


def _share_parameter(parameter: torch.nn.Parameter, tensor: torch.Tensor) -> None:
    if parameter.shape != tensor.shape:
        raise ValueError(
            f"parameter shape {parameter.shape} != tensor shape {tensor.shape}"
        )
    parameter.data = tensor


def _build_vllm(
    tensors: dict[str, torch.Tensor], shape: Shape, vllm_config, quant_config
):
    from transformers import DeepseekV3Config
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE

    config = DeepseekV3Config(
        hidden_size=shape.hidden,
        intermediate_size=18432,
        moe_intermediate_size=shape.intermediate,
        n_routed_experts=shape.num_experts,
        n_shared_experts=1,
        num_experts_per_tok=shape.top_k,
        n_group=shape.num_groups,
        topk_group=shape.topk_groups,
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=shape.routed_scale,
        scoring_func="sigmoid",
        hidden_act="silu",
    )
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(vllm_config), torch.device("cuda"):
            layer = DeepseekV2MoE(
                config,
                vllm_config.parallel_config,
                quant_config=quant_config,
                prefix="model.layers.3.mlp",
            ).eval()
    finally:
        torch.set_default_dtype(old_dtype)

    _share_parameter(layer.gate.weight, tensors["router_weight"])
    assert layer.gate.e_score_correction_bias is not None
    _share_parameter(layer.gate.e_score_correction_bias, tensors["correction_bias"])
    routed = layer.experts.routed_experts
    _share_parameter(routed.w13_weight, tensors["expert_w13_q"])
    _share_parameter(routed.w13_weight_scale_inv, tensors["expert_w13_scale"])
    _share_parameter(routed.w2_weight, tensors["expert_w2_q"])
    _share_parameter(routed.w2_weight_scale_inv, tensors["expert_w2_scale"])
    assert layer.shared_experts is not None
    shared_w13 = layer.shared_experts.gate_up_proj
    shared_w2 = layer.shared_experts.down_proj
    _share_parameter(shared_w13.weight, tensors["shared_w13_q"])
    _share_parameter(shared_w13.weight_scale_inv, tensors["shared_w13_scale"])
    _share_parameter(shared_w2.weight, tensors["shared_w2_q"])
    _share_parameter(shared_w2.weight_scale_inv, tensors["shared_w2_scale"])
    with set_current_vllm_config(vllm_config):
        routed.quant_method.process_weights_after_loading(routed)
        shared_w13.quant_method.process_weights_after_loading(shared_w13)
        shared_w2.quant_method.process_weights_after_loading(shared_w2)
        shared_w13.update_param_tp_status()
        shared_w2.update_param_tp_status()

    def launch():
        with set_forward_context(
            None, vllm_config=vllm_config, num_tokens=shape.batch, slot_mapping=None
        ):
            return layer(tensors["hidden"])

    def routing():
        with set_forward_context(
            None, vllm_config=vllm_config, num_tokens=shape.batch, slot_mapping=None
        ):
            logits, _ = layer.gate(tensors["hidden"])
            weights, ids = layer.experts.router.select_experts(
                hidden_states=tensors["hidden"],
                router_logits=logits,
                topk_indices_dtype=layer.experts._quant_method.topk_indices_dtype,
            )
        return logits, weights, ids

    return layer, launch, routing


def _assert_vllm_contract(layer, vllm_config, quant_config, hidden) -> dict[str, Any]:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExpertsOrder,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        is_deep_gemm_e8m0_used,
    )

    routed_method = layer.experts.routed_experts.quant_method
    shared_w13_method = layer.shared_experts.gate_up_proj.quant_method
    shared_w2_method = layer.shared_experts.down_proj.quant_method
    shared_runner = layer.experts.shared_experts
    contract = {
        "routed_backend": getattr(
            routed_method.fp8_backend, "value", str(routed_method.fp8_backend)
        ),
        "routed_experts_class": routed_method.experts_cls.__name__,
        "shared_w13_backend": type(shared_w13_method.fp8_linear).__name__,
        "shared_w2_backend": type(shared_w2_method.fp8_linear).__name__,
        "ue8m0": bool(is_deep_gemm_e8m0_used()),
        "lora": vllm_config.lora_config is not None,
        "shared_expert_fused": bool(layer.is_fused_shared_expert_enabled),
        "shared_order": (
            shared_runner._determine_shared_experts_order(hidden).name
            if shared_runner is not None
            else None
        ),
        "serialized_checkpoint": bool(quant_config.is_checkpoint_fp8_serialized),
        "activation_scheme": quant_config.activation_scheme,
        "weight_block_size": quant_config.weight_block_size,
    }
    routed_experts_classes = {
        "DEEPGEMM": "TritonOrDeepGemmExperts",
        "FLASHINFER_TRTLLM": "TrtLlmFp8ExpertsMonolithic",
    }
    expected = {
        "shared_w13_backend": "DeepGemmFp8BlockScaledMMKernel",
        "shared_w2_backend": "DeepGemmFp8BlockScaledMMKernel",
        "ue8m0": True,
        "lora": False,
        "shared_expert_fused": False,
        "shared_order": SharedExpertsOrder.MULTI_STREAM_OVERLAPPED.name,
        "serialized_checkpoint": True,
        "activation_scheme": "dynamic",
        "weight_block_size": [FP8_GROUP, FP8_GROUP],
    }
    routed_backend = contract["routed_backend"]
    if (
        routed_backend not in routed_experts_classes
        or contract["routed_experts_class"] != routed_experts_classes[routed_backend]
        or any(contract[key] != value for key, value in expected.items())
    ):
        raise AssertionError(f"unexpected vLLM production contract: {contract}")
    return contract


def _standalone_module():
    repo_root = str(Path(__file__).resolve().parents[3])
    inserted = repo_root not in sys.path
    if inserted:
        sys.path.insert(0, repo_root)
    try:
        from pretuned_kernels.megakernels.deepseek_v3_moe_fp8 import _standalone
    finally:
        if inserted:
            sys.path.remove(repo_root)
    return _standalone


def _persistent_args(tensors: dict[str, torch.Tensor], shape: Shape) -> tuple:
    return (
        tensors["hidden"],
        tensors["router_weight"],
        tensors["correction_bias"],
        tensors["expert_w13_q"],
        tensors["expert_w13_scale"],
        tensors["expert_w2_q"],
        tensors["expert_w2_scale"],
        tensors["shared_w13_q"],
        tensors["shared_w13_scale"],
        tensors["shared_w2_q"],
        tensors["shared_w2_scale"],
        shape.top_k,
        shape.num_groups,
        shape.topk_groups,
        shape.routed_scale,
    )


def _assert_exact_outputs(
    persistent: tuple[torch.Tensor, ...],
    standalone: tuple[torch.Tensor, ...],
) -> None:
    if len(persistent) != len(PERSISTENT_OUTPUTS):
        raise AssertionError("persistent output arity does not match root manifest")
    if len(standalone) != len(PERSISTENT_OUTPUTS):
        raise AssertionError("standalone output arity does not match root manifest")
    for name, persistent_value, standalone_value in zip(
        PERSISTENT_OUTPUTS,
        persistent,
        standalone,
        strict=True,
    ):
        if not torch.equal(persistent_value, standalone_value):
            raise AssertionError(f"{name} differs between persistent and standalone")


def _similarity_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f64 = actual.double()
    expected_f64 = expected.double()
    denominator = (actual_f64.square() + expected_f64.square()).sum().clamp(min=1e-30)
    return float(1.0 - 2.0 * (actual_f64 * expected_f64).sum() / denominator)


def _validate(
    persistent: tuple[torch.Tensor, ...],
    standalone: tuple[torch.Tensor, ...],
    vllm_output: torch.Tensor,
    vllm_routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tensors: dict[str, torch.Tensor],
) -> None:
    _assert_exact_outputs(persistent, standalone)
    vllm_logits, vllm_topk_weights, vllm_topk_ids = vllm_routing
    torch.testing.assert_close(standalone[1], vllm_logits, rtol=2e-3, atol=2e-3)

    standalone_order = torch.argsort(standalone[3], dim=-1)
    vllm_order = torch.argsort(vllm_topk_ids, dim=-1)
    torch.testing.assert_close(
        torch.gather(standalone[3], 1, standalone_order),
        torch.gather(vllm_topk_ids, 1, vllm_order),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.gather(standalone[2], 1, standalone_order),
        torch.gather(vllm_topk_weights, 1, vllm_order),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        standalone[3],
        tensors["reference_topk_ids"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        standalone[2],
        tensors["reference_topk_weights"],
        rtol=1e-5,
        atol=1e-5,
    )
    for label, output in (
        ("persistent", persistent[0]),
        ("standalone", standalone[0]),
    ):
        error = _similarity_error(output, vllm_output)
        if not math.isfinite(error) or error > 0.02:
            raise AssertionError(
                f"{label} Helion differs excessively from production vLLM: "
                f"similarity error {error}"
            )


def _make_problem():
    shape = Shape()
    tensors = _allocate(shape, seed=0)
    vllm_config, quant_config = _initialize_vllm()
    vllm_layer, vllm_call, vllm_routing_call = _build_vllm(
        tensors,
        shape,
        vllm_config,
        quant_config,
    )
    contract = _assert_vllm_contract(
        vllm_layer,
        vllm_config,
        quant_config,
        tensors["hidden"],
    )
    standalone_call, standalone_output = _standalone_module().build(
        tensors,
        shape,
    )
    return (
        tensors,
        shape,
        standalone_call,
        standalone_output,
        vllm_call,
        vllm_routing_call,
        contract["routed_backend"],
    )


def _destroy_vllm() -> None:
    try:
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from vllm.v1.worker.workspace import reset_workspace_manager
    except ImportError:
        return

    try:
        destroy_model_parallel()
    finally:
        try:
            destroy_distributed_environment()
        finally:
            reset_workspace_manager()


def use_cudagraph() -> bool:
    """The timed closures replay pre-captured CUDA graphs."""
    return True


def has_vllm() -> bool:
    """Whether the optional production vLLM comparison is importable."""
    _ensure_vllm_importable()
    try:
        from vllm.config import VllmConfig  # noqa: F401
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE  # noqa: F401
    except ImportError:
        return False
    return True


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("deepseek_v3_moe_fp8 is pretuned only for NVIDIA SM100")


@torch.inference_mode()
def correctness_check() -> None:
    """Check persistent and separate Helion against production vLLM."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError("vLLM is required for the DeepSeek-V3 FP8 comparison")
    try:
        (
            tensors,
            shape,
            standalone_call,
            _standalone_output,
            vllm_call,
            vllm_routing_call,
            _backend,
        ) = _make_problem()
        persistent_output = deepseek_v3_moe_fp8(*_persistent_args(tensors, shape))
        standalone_output = standalone_call()
        vllm_output = vllm_call()
        vllm_routing = vllm_routing_call()
        torch.cuda.synchronize()
        _validate(
            persistent_output,
            standalone_output,
            vllm_output,
            vllm_routing,
            tensors,
        )
    finally:
        _destroy_vllm()


@torch.inference_mode()
def main(verbose: bool = True) -> dict[str, Any]:
    """Benchmark persistent, separate Helion, and production vLLM with cold L2."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError("vLLM is required for the DeepSeek-V3 FP8 comparison")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _bench import capture_cuda_graph
    from _bench import run_sweep

    try:
        (
            tensors,
            shape,
            standalone_call,
            _standalone_output,
            vllm_call,
            vllm_routing_call,
            backend,
        ) = _make_problem()
        kernel_args = _persistent_args(tensors, shape)
        persistent_output = deepseek_v3_moe_fp8(*kernel_args)
        standalone_output = standalone_call()
        vllm_output = vllm_call()
        vllm_routing = vllm_routing_call()
        torch.cuda.synchronize()
        _validate(
            persistent_output,
            standalone_output,
            vllm_output,
            vllm_routing,
            tensors,
        )

        # Keep production, matched standalone, then persistent capture order
        # stable: dynamic ticket assignment is predecessor-sensitive on B200.
        vllm_graph, captured_vllm = capture_cuda_graph(vllm_call)
        standalone_graph, captured_standalone = capture_cuda_graph(standalone_call)
        persistent_graph, captured_persistent = capture_cuda_graph(
            lambda: deepseek_v3_moe_fp8(*kernel_args)
        )
        vllm_graph.replay()
        standalone_graph.replay()
        persistent_graph.replay()
        torch.cuda.synchronize()
        _validate(
            captured_persistent,
            captured_standalone,
            captured_vllm,
            vllm_routing,
            tensors,
        )

        def make_calls(_shape: Shape) -> tuple:
            return (
                persistent_graph.replay,
                [
                    ("standalone_helion", standalone_graph.replay),
                    (f"vllm_auto ({backend})", vllm_graph.replay),
                ],
                (
                    f"{shape.batch:>5d}  {shape.hidden:>6d}  "
                    f"{shape.intermediate:>12d}  {shape.top_k:>5d}"
                ),
            )

        return run_sweep(
            (shape,),
            make_calls,
            use_cudagraph=False,
            pre_captured_cudagraph=True,
            thermal_warmup_ms=10_000,
            verbose=verbose,
            shape_header=(
                f"{'batch':>5s}  {'hidden':>6s}  {'intermediate':>12s}  {'top_k':>5s}"
            ),
        )
    finally:
        _destroy_vllm()


if __name__ == "__main__":
    main()
