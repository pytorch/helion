# ruff: noqa: ANN001, ANN201
# pyrefly: ignore-errors
"""Gemma 4 26B-A4B batch-one MoE megakernel, pretuned for NVIDIA B200.

The single Helion function's eight top-level tile loops implement router
projection, hierarchical top-k, gate/up projection, GeGLU, down projection,
expert reduction, and output RMSNorm.  The benchmark uses the production
batch-one geometry and checks the result against vLLM's Gemma 4 router and
fused-MoE implementation.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


BATCH = 1
HIDDEN = 2816
INTERMEDIATE = 704
NUM_EXPERTS = 128
TOP_K = 8
EPS = 1e-6


@helion.aot_kernel(static_shapes=True, backend="triton")
def gemma4_a4b_moe(
    residual,
    pre_ff_norm_weight,
    router_scale,
    root_size,
    router_weight,
    per_expert_scale,
    expert_gate_up_weight,
    expert_down_weight,
    post_ff_norm_weight,
    top_k,
    eps,
):
    __gemma4_router_project_hidden = residual
    __gemma4_router_project_scale = router_scale
    __gemma4_router_project_root_size = root_size
    __gemma4_router_project_weight = router_weight
    __gemma4_router_project_eps = eps
    __gemma4_router_project_m, __gemma4_router_project_k = (
        __gemma4_router_project_hidden.size()
    )
    __gemma4_router_project_num_experts, __gemma4_router_project_weight_k = (
        __gemma4_router_project_weight.size()
    )
    assert __gemma4_router_project_k == __gemma4_router_project_weight_k
    hl.specialize(__gemma4_router_project_k)
    hl.specialize(__gemma4_router_project_num_experts)
    router_logits = torch.empty(
        (__gemma4_router_project_m, __gemma4_router_project_num_experts),
        dtype=torch.float32,
        device=__gemma4_router_project_hidden.device,
    )
    __gemma4_route_candidates_logits = router_logits
    __gemma4_route_candidates_top_k = top_k
    __gemma4_route_candidates_m, __gemma4_route_candidates_num_experts = (
        __gemma4_route_candidates_logits.size()
    )
    __gemma4_route_candidates_top_k = hl.specialize(__gemma4_route_candidates_top_k)
    hl.specialize(__gemma4_route_candidates_num_experts)
    __gemma4_route_candidates_groups = 4
    __gemma4_route_candidates_group_size = (
        __gemma4_route_candidates_num_experts // __gemma4_route_candidates_groups
    )
    candidate_values = torch.empty(
        (
            __gemma4_route_candidates_m,
            __gemma4_route_candidates_groups,
            __gemma4_route_candidates_top_k,
        ),
        dtype=torch.float32,
        device=__gemma4_route_candidates_logits.device,
    )
    candidate_ids = torch.empty(
        (
            __gemma4_route_candidates_m,
            __gemma4_route_candidates_groups,
            __gemma4_route_candidates_top_k,
        ),
        dtype=torch.int32,
        device=__gemma4_route_candidates_logits.device,
    )
    __gemma4_route_merge_candidate_values = candidate_values
    __gemma4_route_merge_candidate_ids = candidate_ids
    __gemma4_route_merge_per_expert_scale = per_expert_scale
    __gemma4_route_merge_top_k = top_k
    (
        __gemma4_route_merge_m,
        __gemma4_route_merge_groups,
        __gemma4_route_merge_candidates_per_group,
    ) = __gemma4_route_merge_candidate_values.size()
    __gemma4_route_merge_top_k = hl.specialize(__gemma4_route_merge_top_k)
    __gemma4_route_merge_candidate_count = (
        __gemma4_route_merge_groups * __gemma4_route_merge_candidates_per_group
    )
    __gemma4_route_merge_values_flat = __gemma4_route_merge_candidate_values.view(
        __gemma4_route_merge_m, __gemma4_route_merge_candidate_count
    )
    __gemma4_route_merge_ids_flat = __gemma4_route_merge_candidate_ids.view(
        __gemma4_route_merge_m, __gemma4_route_merge_candidate_count
    )
    topk_weights = torch.empty(
        (__gemma4_route_merge_m, __gemma4_route_merge_top_k),
        dtype=torch.float32,
        device=__gemma4_route_merge_candidate_values.device,
    )
    topk_ids = torch.empty(
        (__gemma4_route_merge_m, __gemma4_route_merge_top_k),
        dtype=torch.int32,
        device=__gemma4_route_merge_candidate_values.device,
    )
    __gemma4_expert_gate_up_hidden = residual
    __gemma4_expert_gate_up_norm_weight = pre_ff_norm_weight
    __gemma4_expert_gate_up_expert_weight = expert_gate_up_weight
    __gemma4_expert_gate_up_topk_ids = topk_ids
    __gemma4_expert_gate_up_topk_weights = topk_weights
    __gemma4_expert_gate_up_eps = eps
    __gemma4_expert_gate_up_m, __gemma4_expert_gate_up_hidden_size = (
        __gemma4_expert_gate_up_hidden.size()
    )
    (
        __gemma4_expert_gate_up_num_experts,
        __gemma4_expert_gate_up_twice_intermediate,
        __gemma4_expert_gate_up_weight_hidden,
    ) = __gemma4_expert_gate_up_expert_weight.size()
    assert __gemma4_expert_gate_up_hidden_size == __gemma4_expert_gate_up_weight_hidden
    __gemma4_expert_gate_up_intermediate = (
        __gemma4_expert_gate_up_twice_intermediate // 2
    )
    __gemma4_expert_gate_up_top_k = __gemma4_expert_gate_up_topk_ids.size(1)
    hl.specialize(__gemma4_expert_gate_up_num_experts)
    hl.specialize(__gemma4_expert_gate_up_hidden_size)
    hl.specialize(__gemma4_expert_gate_up_intermediate)
    __gemma4_expert_gate_up_flattened_weight = (
        __gemma4_expert_gate_up_expert_weight.view(
            __gemma4_expert_gate_up_num_experts
            * __gemma4_expert_gate_up_twice_intermediate,
            __gemma4_expert_gate_up_hidden_size,
        )
    )
    gate_up = torch.empty(
        (
            __gemma4_expert_gate_up_m * __gemma4_expert_gate_up_top_k,
            __gemma4_expert_gate_up_twice_intermediate,
        ),
        dtype=__gemma4_expert_gate_up_hidden.dtype,
        device=__gemma4_expert_gate_up_hidden.device,
    )
    selected_ids = torch.empty(
        (__gemma4_expert_gate_up_m, __gemma4_expert_gate_up_top_k),
        dtype=torch.int32,
        device=__gemma4_expert_gate_up_hidden.device,
    )
    selected_weights = torch.empty(
        (__gemma4_expert_gate_up_m, __gemma4_expert_gate_up_top_k),
        dtype=torch.float32,
        device=__gemma4_expert_gate_up_hidden.device,
    )
    __gemma4_expert_geglu_gate_up = gate_up
    __gemma4_expert_geglu_m, __gemma4_expert_geglu_twice_intermediate = (
        __gemma4_expert_geglu_gate_up.size()
    )
    __gemma4_expert_geglu_intermediate = __gemma4_expert_geglu_twice_intermediate // 2
    activation_flat = torch.empty(
        (__gemma4_expert_geglu_m, __gemma4_expert_geglu_intermediate),
        device=__gemma4_expert_geglu_gate_up.device,
        dtype=__gemma4_expert_geglu_gate_up.dtype,
    )
    activation = activation_flat.view(
        residual.size(0), top_k, expert_down_weight.size(2)
    )
    __gemma4_expert_down_activation = activation
    __gemma4_expert_down_expert_weight = expert_down_weight
    __gemma4_expert_down_selected_ids = selected_ids
    __gemma4_expert_down_selected_weights = selected_weights
    (
        __gemma4_expert_down_m,
        __gemma4_expert_down_top_k,
        __gemma4_expert_down_intermediate,
    ) = __gemma4_expert_down_activation.size()
    (
        __gemma4_expert_down_num_experts,
        __gemma4_expert_down_hidden_size,
        __gemma4_expert_down_weight_intermediate,
    ) = __gemma4_expert_down_expert_weight.size()
    assert __gemma4_expert_down_intermediate == __gemma4_expert_down_weight_intermediate
    hl.specialize(__gemma4_expert_down_num_experts)
    hl.specialize(__gemma4_expert_down_hidden_size)
    __gemma4_expert_down_flattened_weight = __gemma4_expert_down_expert_weight.view(
        __gemma4_expert_down_num_experts * __gemma4_expert_down_hidden_size,
        __gemma4_expert_down_intermediate,
    )
    expert_outputs = torch.empty(
        (
            __gemma4_expert_down_m,
            __gemma4_expert_down_top_k,
            __gemma4_expert_down_hidden_size,
        ),
        dtype=__gemma4_expert_down_activation.dtype,
        device=__gemma4_expert_down_activation.device,
    )
    __gemma4_expert_reduce_expert_output = expert_outputs
    (
        __gemma4_expert_reduce_m,
        __gemma4_expert_reduce_top_k,
        __gemma4_expert_reduce_hidden_size,
    ) = __gemma4_expert_reduce_expert_output.size()
    moe_down = torch.empty(
        (__gemma4_expert_reduce_m, __gemma4_expert_reduce_hidden_size),
        dtype=__gemma4_expert_reduce_expert_output.dtype,
        device=__gemma4_expert_reduce_expert_output.device,
    )
    __gemma4_moe_post_norm_x = moe_down
    __gemma4_moe_post_norm_weight = post_ff_norm_weight
    __gemma4_moe_post_norm_eps = eps
    __gemma4_moe_post_norm_m, __gemma4_moe_post_norm_n = __gemma4_moe_post_norm_x.size()
    hl.specialize(__gemma4_moe_post_norm_n)
    moe_branch = torch.empty_like(__gemma4_moe_post_norm_x)
    for __gemma4_router_project_tile_m, __gemma4_router_project_tile_expert in hl.tile(
        [__gemma4_router_project_m, __gemma4_router_project_num_experts],
        block_size=[1, None],
    ):
        __gemma4_router_project_token = __gemma4_router_project_tile_m.begin
        __gemma4_router_project_row = __gemma4_router_project_hidden[
            __gemma4_router_project_token, :
        ].to(torch.float32)
        __gemma4_router_project_inv_rms = torch.rsqrt(
            torch.mean(
                __gemma4_router_project_row * __gemma4_router_project_row, dim=-1
            )
            + __gemma4_router_project_eps
        )
        __gemma4_router_project_root = hl.load(__gemma4_router_project_root_size, [])
        __gemma4_router_project_normalized = (
            __gemma4_router_project_row * __gemma4_router_project_inv_rms
        ).to(__gemma4_router_project_hidden.dtype)
        __gemma4_router_project_router_input = (
            __gemma4_router_project_normalized
            * __gemma4_router_project_root
            * __gemma4_router_project_scale[:]
        ).to(__gemma4_router_project_hidden.dtype)
        __gemma4_router_project_weights = __gemma4_router_project_weight[
            __gemma4_router_project_tile_expert, :
        ].to(torch.float32)
        __gemma4_router_project_acc = torch.sum(
            __gemma4_router_project_weights
            * __gemma4_router_project_router_input.to(torch.float32),
            dim=-1,
        )
        router_logits[
            __gemma4_router_project_token, __gemma4_router_project_tile_expert
        ] = __gemma4_router_project_acc
    for (
        __gemma4_route_candidates_tile_m,
        __gemma4_route_candidates_tile_group,
    ) in hl.tile(
        [__gemma4_route_candidates_m, __gemma4_route_candidates_groups],
        block_size=[1, 1],
    ):
        __gemma4_route_candidates_token = __gemma4_route_candidates_tile_m.begin
        __gemma4_route_candidates_group = __gemma4_route_candidates_tile_group.begin
        __gemma4_route_candidates_experts = (
            __gemma4_route_candidates_group * __gemma4_route_candidates_group_size
            + hl.arange(__gemma4_route_candidates_group_size)
        )
        __gemma4_route_candidates_values, __gemma4_route_candidates_ids = torch.topk(
            __gemma4_route_candidates_logits[
                __gemma4_route_candidates_token, __gemma4_route_candidates_experts
            ],
            __gemma4_route_candidates_top_k,
            dim=-1,
            largest=True,
        )
        candidate_values[
            __gemma4_route_candidates_token, __gemma4_route_candidates_group, :
        ] = __gemma4_route_candidates_values
        candidate_ids[
            __gemma4_route_candidates_token, __gemma4_route_candidates_group, :
        ] = (
            __gemma4_route_candidates_ids.to(torch.int32)
            + __gemma4_route_candidates_group * __gemma4_route_candidates_group_size
        )
    for __gemma4_route_merge_tile_m in hl.tile(__gemma4_route_merge_m, block_size=1):
        __gemma4_route_merge_token = __gemma4_route_merge_tile_m.begin
        __gemma4_route_merge_values, __gemma4_route_merge_positions = torch.topk(
            __gemma4_route_merge_values_flat[__gemma4_route_merge_token, :],
            __gemma4_route_merge_top_k,
            dim=-1,
            largest=True,
        )
        __gemma4_route_merge_ids = __gemma4_route_merge_ids_flat[
            __gemma4_route_merge_token, __gemma4_route_merge_positions
        ]
        __gemma4_route_merge_shifted = __gemma4_route_merge_values - torch.amax(
            __gemma4_route_merge_values, dim=-1, keepdim=True
        )
        __gemma4_route_merge_raw_weights = torch.exp(__gemma4_route_merge_shifted)
        __gemma4_route_merge_normalized = __gemma4_route_merge_raw_weights / torch.sum(
            __gemma4_route_merge_raw_weights, dim=-1, keepdim=True
        )
        topk_weights[__gemma4_route_merge_token, :] = (
            __gemma4_route_merge_normalized
            * __gemma4_route_merge_per_expert_scale[__gemma4_route_merge_ids].to(
                torch.float32
            )
        )
        topk_ids[__gemma4_route_merge_token, :] = __gemma4_route_merge_ids
    for (
        __gemma4_expert_gate_up_tile_m,
        __gemma4_expert_gate_up_tile_slot,
        __gemma4_expert_gate_up_tile_i,
    ) in hl.tile(
        [
            __gemma4_expert_gate_up_m,
            __gemma4_expert_gate_up_top_k,
            __gemma4_expert_gate_up_intermediate,
        ],
        block_size=[1, 1, None],
    ):
        __gemma4_expert_gate_up_token = __gemma4_expert_gate_up_tile_m.begin
        __gemma4_expert_gate_up_slot = __gemma4_expert_gate_up_tile_slot.begin
        __gemma4_expert_gate_up_selected_expert = __gemma4_expert_gate_up_topk_ids[
            __gemma4_expert_gate_up_token, __gemma4_expert_gate_up_slot
        ]
        __gemma4_expert_gate_up_gate_row = (
            __gemma4_expert_gate_up_selected_expert
            * __gemma4_expert_gate_up_twice_intermediate
            + __gemma4_expert_gate_up_tile_i.index
        )
        __gemma4_expert_gate_up_up_row = (
            __gemma4_expert_gate_up_gate_row + __gemma4_expert_gate_up_intermediate
        )
        __gemma4_expert_gate_up_row = __gemma4_expert_gate_up_hidden[
            __gemma4_expert_gate_up_token, :
        ].to(torch.float32)
        __gemma4_expert_gate_up_inv_rms = torch.rsqrt(
            torch.mean(
                __gemma4_expert_gate_up_row * __gemma4_expert_gate_up_row, dim=-1
            )
            + __gemma4_expert_gate_up_eps
        )
        __gemma4_expert_gate_up_gate_acc = hl.zeros(
            [__gemma4_expert_gate_up_tile_i], dtype=torch.float32
        )
        __gemma4_expert_gate_up_up_acc = hl.zeros(
            [__gemma4_expert_gate_up_tile_i], dtype=torch.float32
        )
        for __gemma4_expert_gate_up_tile_k in hl.tile(
            __gemma4_expert_gate_up_hidden_size
        ):
            __gemma4_expert_gate_up_values = __gemma4_expert_gate_up_hidden[
                __gemma4_expert_gate_up_token, __gemma4_expert_gate_up_tile_k
            ].to(torch.float32)
            __gemma4_expert_gate_up_normalized = (
                __gemma4_expert_gate_up_values * __gemma4_expert_gate_up_inv_rms
            ).to(__gemma4_expert_gate_up_hidden.dtype)
            __gemma4_expert_gate_up_expert_input = (
                __gemma4_expert_gate_up_normalized
                * __gemma4_expert_gate_up_norm_weight[__gemma4_expert_gate_up_tile_k]
            ).to(__gemma4_expert_gate_up_hidden.dtype)
            __gemma4_expert_gate_up_gate_weight = (
                __gemma4_expert_gate_up_flattened_weight[
                    __gemma4_expert_gate_up_gate_row, __gemma4_expert_gate_up_tile_k
                ].to(torch.float32)
            )
            __gemma4_expert_gate_up_up_weight = (
                __gemma4_expert_gate_up_flattened_weight[
                    __gemma4_expert_gate_up_up_row, __gemma4_expert_gate_up_tile_k
                ].to(torch.float32)
            )
            __gemma4_expert_gate_up_input_fp32 = (
                __gemma4_expert_gate_up_expert_input.to(torch.float32)
            )
            __gemma4_expert_gate_up_gate_acc = (
                __gemma4_expert_gate_up_gate_acc
                + torch.sum(
                    __gemma4_expert_gate_up_gate_weight
                    * __gemma4_expert_gate_up_input_fp32,
                    dim=-1,
                )
            )
            __gemma4_expert_gate_up_up_acc = __gemma4_expert_gate_up_up_acc + torch.sum(
                __gemma4_expert_gate_up_up_weight * __gemma4_expert_gate_up_input_fp32,
                dim=-1,
            )
        gate_up[
            __gemma4_expert_gate_up_token * __gemma4_expert_gate_up_top_k
            + __gemma4_expert_gate_up_slot,
            __gemma4_expert_gate_up_tile_i,
        ] = __gemma4_expert_gate_up_gate_acc.to(gate_up.dtype)
        gate_up[
            __gemma4_expert_gate_up_token * __gemma4_expert_gate_up_top_k
            + __gemma4_expert_gate_up_slot,
            __gemma4_expert_gate_up_tile_i.index + __gemma4_expert_gate_up_intermediate,
        ] = __gemma4_expert_gate_up_up_acc.to(gate_up.dtype)
        if __gemma4_expert_gate_up_tile_i.begin == 0:
            selected_ids[
                __gemma4_expert_gate_up_token, __gemma4_expert_gate_up_slot
            ] = __gemma4_expert_gate_up_selected_expert
            selected_weights[
                __gemma4_expert_gate_up_token, __gemma4_expert_gate_up_slot
            ] = __gemma4_expert_gate_up_topk_weights[
                __gemma4_expert_gate_up_token, __gemma4_expert_gate_up_slot
            ]
    for __gemma4_expert_geglu_tile_m, __gemma4_expert_geglu_tile_i in hl.tile(
        [__gemma4_expert_geglu_m, __gemma4_expert_geglu_intermediate],
        block_size=[1, None],
    ):
        __gemma4_expert_geglu_gate = __gemma4_expert_geglu_gate_up[
            __gemma4_expert_geglu_tile_m, __gemma4_expert_geglu_tile_i
        ].to(torch.float32)
        __gemma4_expert_geglu_up = __gemma4_expert_geglu_gate_up[
            __gemma4_expert_geglu_tile_m,
            __gemma4_expert_geglu_tile_i + __gemma4_expert_geglu_intermediate,
        ]
        activation_flat[__gemma4_expert_geglu_tile_m, __gemma4_expert_geglu_tile_i] = (
            0.5
            * __gemma4_expert_geglu_gate
            * (
                1.0
                + torch.tanh(
                    0.7978845608028654
                    * (
                        __gemma4_expert_geglu_gate
                        + 0.044715
                        * __gemma4_expert_geglu_gate
                        * __gemma4_expert_geglu_gate
                        * __gemma4_expert_geglu_gate
                    )
                )
            )
        ).to(__gemma4_expert_geglu_up.dtype) * __gemma4_expert_geglu_up
    for (
        __gemma4_expert_down_tile_m,
        __gemma4_expert_down_tile_slot,
        __gemma4_expert_down_tile_n,
    ) in hl.tile(
        [
            __gemma4_expert_down_m,
            __gemma4_expert_down_top_k,
            __gemma4_expert_down_hidden_size,
        ],
        block_size=[1, 1, None],
    ):
        __gemma4_expert_down_token = __gemma4_expert_down_tile_m.begin
        __gemma4_expert_down_slot = __gemma4_expert_down_tile_slot.begin
        __gemma4_expert_down_selected_expert = __gemma4_expert_down_selected_ids[
            __gemma4_expert_down_token, __gemma4_expert_down_slot
        ]
        __gemma4_expert_down_selected_row = (
            __gemma4_expert_down_selected_expert * __gemma4_expert_down_hidden_size
            + __gemma4_expert_down_tile_n.index
        )
        __gemma4_expert_down_acc = hl.zeros(
            [__gemma4_expert_down_tile_n], dtype=torch.float32
        )
        for __gemma4_expert_down_tile_k in hl.tile(__gemma4_expert_down_intermediate):
            __gemma4_expert_down_values = __gemma4_expert_down_activation[
                __gemma4_expert_down_token,
                __gemma4_expert_down_slot,
                __gemma4_expert_down_tile_k,
            ].to(torch.float32)
            __gemma4_expert_down_weights = __gemma4_expert_down_flattened_weight[
                __gemma4_expert_down_selected_row, __gemma4_expert_down_tile_k
            ].to(torch.float32)
            __gemma4_expert_down_acc = __gemma4_expert_down_acc + torch.sum(
                __gemma4_expert_down_weights * __gemma4_expert_down_values, dim=-1
            )
        __gemma4_expert_down_weight = __gemma4_expert_down_selected_weights[
            __gemma4_expert_down_token, __gemma4_expert_down_slot
        ].to(torch.float32)
        expert_outputs[
            __gemma4_expert_down_token,
            __gemma4_expert_down_slot,
            __gemma4_expert_down_tile_n,
        ] = (__gemma4_expert_down_acc * __gemma4_expert_down_weight).to(
            expert_outputs.dtype
        )
    for __gemma4_expert_reduce_tile_m, __gemma4_expert_reduce_tile_n in hl.tile(
        [__gemma4_expert_reduce_m, __gemma4_expert_reduce_hidden_size],
        block_size=[1, None],
    ):
        __gemma4_expert_reduce_values = __gemma4_expert_reduce_expert_output[
            __gemma4_expert_reduce_tile_m, :, __gemma4_expert_reduce_tile_n
        ].to(torch.float32)
        moe_down[__gemma4_expert_reduce_tile_m, __gemma4_expert_reduce_tile_n] = (
            torch.sum(__gemma4_expert_reduce_values, dim=1).to(moe_down.dtype)
        )
    for __gemma4_moe_post_norm_tile_m in hl.tile(
        __gemma4_moe_post_norm_m, block_size=1
    ):
        __gemma4_moe_post_norm_values = __gemma4_moe_post_norm_x[
            __gemma4_moe_post_norm_tile_m, :
        ].to(torch.float32)
        __gemma4_moe_post_norm_inv_rms = torch.rsqrt(
            torch.mean(
                __gemma4_moe_post_norm_values * __gemma4_moe_post_norm_values, dim=-1
            )
            + __gemma4_moe_post_norm_eps
        )
        __gemma4_moe_post_norm_normalized = (
            __gemma4_moe_post_norm_values * __gemma4_moe_post_norm_inv_rms[:, None]
        ).to(__gemma4_moe_post_norm_x.dtype)
        moe_branch[__gemma4_moe_post_norm_tile_m, :] = (
            __gemma4_moe_post_norm_normalized * __gemma4_moe_post_norm_weight[None, :]
        )
    return (
        moe_branch,
        router_logits,
        topk_weights,
        topk_ids,
        activation,
        expert_outputs,
        moe_down,
    )


def use_cudagraph() -> bool:
    """The timed closures replay pre-captured CUDA graphs."""
    return True


def has_vllm() -> bool:
    """Whether the optional production vLLM comparison is importable."""
    try:
        from vllm.config import VllmConfig  # noqa: F401
        from vllm.model_executor.models.gemma4 import Gemma4MoE  # noqa: F401
    except ImportError:
        return False
    return True


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("gemma4_a4b_moe is pretuned only for NVIDIA SM100")


def _make_inputs(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    return {
        "residual": torch.randn((BATCH, HIDDEN), device=device, dtype=torch.bfloat16)
        * 1.4,
        "pre_ff_norm_weight": (
            torch.randn(HIDDEN, device=device, dtype=torch.bfloat16) * 0.05 + 1.0
        ),
        "router_scale": (
            torch.randn(HIDDEN, device=device, dtype=torch.bfloat16) * 0.05 + 1.0
        ),
        "root_size": torch.tensor(HIDDEN**-0.5, device=device, dtype=torch.bfloat16),
        "router_weight": torch.randn(
            (NUM_EXPERTS, HIDDEN), device=device, dtype=torch.bfloat16
        )
        * 0.01,
        "per_expert_scale": (
            torch.randn(NUM_EXPERTS, device=device, dtype=torch.bfloat16) * 0.05 + 1.0
        ),
        "expert_gate_up_weight": torch.randn(
            (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.0094,
        "expert_down_weight": torch.randn(
            (NUM_EXPERTS, HIDDEN, INTERMEDIATE),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.0188,
        "post_ff_norm_weight": (
            torch.randn(HIDDEN, device=device, dtype=torch.bfloat16) * 0.05 + 1.0
        ),
    }


def _kernel_args(tensors: dict[str, torch.Tensor]) -> tuple[object, ...]:
    return (
        tensors["residual"],
        tensors["pre_ff_norm_weight"],
        tensors["router_scale"],
        tensors["root_size"],
        tensors["router_weight"],
        tensors["per_expert_scale"],
        tensors["expert_gate_up_weight"],
        tensors["expert_down_weight"],
        tensors["post_ff_norm_weight"],
        TOP_K,
        EPS,
    )


def _initialize_vllm() -> object:
    from vllm.config import CacheConfig
    from vllm.config import CUDAGraphMode
    from vllm.config import VllmConfig
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment
    from vllm.distributed import initialize_model_parallel
    from vllm.distributed import model_parallel_is_initialized
    from vllm.utils.network_utils import get_open_port
    from vllm.v1.worker.workspace import init_workspace_manager
    from vllm.v1.worker.workspace import is_workspace_manager_initialized

    if (
        torch.distributed.is_initialized()
        or model_parallel_is_initialized()
        or is_workspace_manager_initialized()
    ):
        raise RuntimeError(
            "gemma4_a4b_moe owns its temporary vLLM runtime state; "
            "run it outside an initialized vLLM or distributed context"
        )
    device_index = torch.cuda.current_device()
    init_workspace_manager(torch.device("cuda", device_index))
    try:
        cache_config = CacheConfig(block_size=16, cache_dtype="auto")
        vllm_config = VllmConfig(cache_config=cache_config)
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        vllm_config.model_config = SimpleNamespace(
            dtype=torch.bfloat16,
            is_mm_prefix_lm=False,
            is_moe=True,
            is_diffusion=False,
            is_hybrid=False,
            is_attention_free=True,
            runner_type="generate",
            architectures=["Gemma4ForConditionalGeneration"],
            max_model_len=8192,
            rswa_window=None,
            compute_hash=lambda: "gemma4-a4b-moe-pretuned",
        )
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=device_index,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(1, 1)
    except Exception:
        _destroy_vllm()
        raise
    return vllm_config


def _destroy_vllm() -> None:
    from vllm.distributed import destroy_distributed_environment
    from vllm.distributed import destroy_model_parallel
    from vllm.v1.worker.workspace import reset_workspace_manager

    destroy_model_parallel()
    destroy_distributed_environment()
    reset_workspace_manager()


def _make_vllm_call(
    tensors: dict[str, torch.Tensor],
) -> tuple[
    Callable[[], tuple[torch.Tensor, ...]],
    Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    str,
    Callable[[], None],
]:
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models.gemma4 import Gemma4MoE
    from vllm.model_executor.models.gemma4 import Gemma4Router
    from vllm.model_executor.models.gemma4 import gemma4_fused_routing_kernel_triton

    vllm_config = _initialize_vllm()
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        rms_norm_eps=EPS,
        num_experts=NUM_EXPERTS,
        top_k_experts=TOP_K,
        moe_intermediate_size=INTERMEDIATE,
    )
    try:
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with set_current_vllm_config(vllm_config):
                pre_norm = RMSNorm(HIDDEN, eps=EPS).eval().cuda()
                router = (
                    Gemma4Router(config, prefix="model.layers.0.router").eval().cuda()
                )
                moe = Gemma4MoE(config, prefix="model.layers.0.moe").eval().cuda()
                post_norm = RMSNorm(HIDDEN, eps=EPS).eval().cuda()
        finally:
            torch.set_default_dtype(old_dtype)

        routed_experts = moe.experts.routed_experts
        backend = routed_experts.quant_method.unquantized_backend.value
        with torch.no_grad():
            pre_norm.weight.copy_(tensors["pre_ff_norm_weight"])
            router.scale.copy_(tensors["router_scale"])
            router.proj.weight.copy_(tensors["router_weight"])
            moe.per_expert_scale.copy_(tensors["per_expert_scale"])
            routed_experts.w13_weight.copy_(tensors["expert_gate_up_weight"])
            routed_experts.w2_weight.copy_(tensors["expert_down_weight"])
            post_norm.weight.copy_(tensors["post_ff_norm_weight"])
            routed_experts.quant_method.process_weights_after_loading(routed_experts)
    except Exception:
        _destroy_vllm()
        raise

    def launch() -> tuple[torch.Tensor, ...]:
        with set_forward_context(None, vllm_config=vllm_config, num_tokens=BATCH):
            expert_input = pre_norm(tensors["residual"])
            router_logits = router(tensors["residual"])
            moe_down = moe(expert_input, router_logits)
            moe_branch = post_norm(moe_down)
        return moe_branch, router_logits, moe_down

    def routing(
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return gemma4_fused_routing_kernel_triton(
            router_logits, TOP_K, moe.per_expert_scale
        )

    def close() -> None:
        _destroy_vllm()

    return launch, routing, backend, close


def _assert_vllm_close(
    helion_outputs: tuple[torch.Tensor, ...],
    vllm_outputs: tuple[torch.Tensor, ...],
    routing: tuple[torch.Tensor, torch.Tensor],
) -> None:
    (
        helion_branch,
        helion_logits,
        helion_topk_weights,
        helion_topk_ids,
        _activation,
        _expert_outputs,
        helion_down,
    ) = helion_outputs
    vllm_branch, vllm_logits, vllm_down = vllm_outputs
    vllm_topk_weights, vllm_topk_ids = routing
    torch.testing.assert_close(
        helion_logits.float(), vllm_logits.float(), atol=0.05, rtol=0.02
    )
    torch.testing.assert_close(
        helion_topk_weights.float(),
        vllm_topk_weights.float(),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(helion_topk_ids, vllm_topk_ids)
    torch.testing.assert_close(
        helion_down.float(), vllm_down.float(), atol=0.15, rtol=0.06
    )
    torch.testing.assert_close(
        helion_branch.float(), vllm_branch.float(), atol=0.15, rtol=0.06
    )


@torch.inference_mode()
def correctness_check() -> None:
    """Check the one pretuned shape against vLLM's production MoE path."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError("vLLM is required for the Gemma 4 A4B comparison")
    tensors = _make_inputs()
    vllm_call, vllm_routing, _backend, close_vllm = _make_vllm_call(tensors)
    try:
        helion_outputs = gemma4_a4b_moe(*_kernel_args(tensors))
        vllm_outputs = vllm_call()
        routing = vllm_routing(vllm_outputs[1])
        torch.cuda.synchronize()
        _assert_vllm_close(helion_outputs, vllm_outputs, routing)
    finally:
        close_vllm()


@torch.inference_mode()
def main(verbose: bool = True) -> dict:
    """Benchmark the one production shape against vLLM with cold-L2 replay."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError("vLLM is required for the Gemma 4 A4B comparison")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _bench import capture_cuda_graph
    from _bench import run_sweep

    tensors = _make_inputs()
    vllm_call, vllm_routing, backend, close_vllm = _make_vllm_call(tensors)
    try:
        helion_outputs = gemma4_a4b_moe(*_kernel_args(tensors))
        vllm_outputs = vllm_call()
        routing = vllm_routing(vllm_outputs[1])
        torch.cuda.synchronize()
        _assert_vllm_close(helion_outputs, vllm_outputs, routing)

        helion_graph, _ = capture_cuda_graph(
            lambda: gemma4_a4b_moe(*_kernel_args(tensors))
        )
        vllm_graph, _ = capture_cuda_graph(vllm_call)

        def make_calls(_shape: None) -> tuple:
            return (
                helion_graph.replay,
                [(f"vllm_auto ({backend})", vllm_graph.replay)],
                f"{BATCH:>5d}  {HIDDEN:>6d}  {NUM_EXPERTS:>7d}",
            )

        return run_sweep(
            [None],
            make_calls,
            use_cudagraph=False,
            pre_captured_cudagraph=True,
            interleave_pre_captured=False,
            thermal_warmup_ms=10_000,
            verbose=verbose,
            shape_header=f"{'batch':>5s}  {'hidden':>6s}  {'experts':>7s}",
        )
    finally:
        close_vllm()


if __name__ == "__main__":
    main()
