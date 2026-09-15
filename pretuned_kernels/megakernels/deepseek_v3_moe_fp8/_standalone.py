# ruff: noqa: A002, ANN001, ANN202
"""Matched standalone Helion kernels for DeepSeek-V3 decode MoE FP8.

The twelve launches preserve the persistent kernel's complete logical boundary.
The shared-expert branch runs on an auxiliary stream, matching vLLM production.
"""

from __future__ import annotations

from typing import Any

import torch

import helion
import helion.language as hl

FP8_DTYPE = torch.float8_e4m3fn
FP8_GROUP = 128
FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10

# Cold-L2 B200 screen over output tiles 4..512, warps 1/2/4/8, and range
# stages 0..6.  These are ordinary Helion controls, not compiler special cases.
CONFIGS: dict[str, dict[str, Any]] = {
    "router": {
        "block_sizes": [2, 256],
        "num_stages": 6,
        "num_warps": 1,
        "range_num_stages": [0, 0],
    },
    "topk": {
        "block_sizes": [],
        "num_stages": 1,
        "num_warps": 1,
        "range_num_stages": [0],
    },
    "routed_input_quant": {
        "block_sizes": [2],
        "num_stages": 1,
        "num_warps": 4,
        "range_num_stages": [0],
    },
    "expert_w13": {
        "block_sizes": [32],
        "indexing": [
            "pointer",
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
        ],
        "load_eviction_policies": ["", "last", "", "", ""],
        "num_stages": 5,
        "num_warps": 1,
        "range_flattens": [None, True],
        "range_num_stages": [0, 0],
        "range_unroll_factors": [0, 3],
    },
    "expert_act_quant": {
        "block_sizes": [1],
        "num_stages": 1,
        "num_warps": 4,
        "range_num_stages": [0],
    },
    "expert_w2": {
        "block_sizes": [32],
        "indexing": [
            "pointer",
            "pointer",
            "tensor_descriptor",
            "pointer",
            "pointer",
            "pointer",
        ],
        "load_eviction_policies": ["", "first", "last", "first", ""],
        "loop_orders": [[1, 0]],
        "num_stages": 3,
        "num_warps": 1,
        "range_flattens": [None, False],
        "range_multi_buffers": [None, False],
        "range_num_stages": [0, 3],
        "range_unroll_factors": [0, 3],
        "range_warp_specializes": [None, False],
    },
    "expert_reduce": {
        "block_sizes": [64],
        "num_stages": 1,
        "num_warps": 1,
        "range_num_stages": [0],
    },
    "shared_input_quant": {
        "block_sizes": [2],
        "num_stages": 1,
        "num_warps": 4,
        "range_num_stages": [0],
    },
    "shared_w13": {
        "block_sizes": [8],
        "num_stages": 1,
        "num_warps": 1,
        "range_num_stages": [0, 6],
    },
    "shared_act_quant": {
        "block_sizes": [1],
        "num_stages": 1,
        "num_warps": 4,
        "range_num_stages": [0],
    },
    "shared_w2": {
        "block_sizes": [8],
        "num_stages": 1,
        "num_warps": 1,
        "range_num_stages": [0, 5],
    },
    "final_add": {
        "block_sizes": [256],
        "num_stages": 1,
        "num_warps": 4,
        "range_num_stages": [0],
    },
}


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def router_mm_fp32(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, k = hidden.size()
    n, weight_k = weight.size()
    assert k == weight_k
    output = torch.empty((m, n), dtype=torch.float32, device=hidden.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, hidden[tile_m, tile_k], weight[tile_n, tile_k].T)
        output[tile_m, tile_n] = acc
    return output


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
        scores = torch.sigmoid(logits[:, :])
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
        weight_0 = torch.sigmoid(torch.sum(logits[:, id_0], dim=-1))
        weight_1 = torch.sigmoid(torch.sum(logits[:, id_1], dim=-1))
        weight_2 = torch.sigmoid(torch.sum(logits[:, id_2], dim=-1))
        weight_3 = torch.sigmoid(torch.sum(logits[:, id_3], dim=-1))
        weight_4 = torch.sigmoid(torch.sum(logits[:, id_4], dim=-1))
        weight_5 = torch.sigmoid(torch.sum(logits[:, id_5], dim=-1))
        weight_6 = torch.sigmoid(torch.sum(logits[:, id_6], dim=-1))
        weight_7 = torch.sigmoid(torch.sum(logits[:, id_7], dim=-1))
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
def dynamic_k128_quant(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = input.shape
    groups = columns // 128
    hl.specialize(columns)
    assert columns == groups * 128
    output_q = torch.empty_like(input, dtype=FP8_DTYPE)
    output_scale = torch.empty((rows, groups), dtype=torch.float32, device=input.device)
    input_3d = input.view(rows, groups, 128)
    output_3d = output_q.view(rows, groups, 128)
    for tile_r, tile_g, tile_k in hl.tile(
        [rows, groups, 128], block_size=[1, None, 128]
    ):
        values = input_3d[tile_r, tile_g, tile_k].to(torch.float32)
        scale = torch.amax(torch.abs(values), dim=-1).clamp(min=FP8_EPS) / FP8_MAX
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
        output_scale[tile_r, tile_g] = scale
        output_3d[tile_r, tile_g, tile_k] = (
            (values / scale[:, :, None]).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
        )
    return output_q, output_scale


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def selected_expert_w13_fp8(
    input_q: torch.Tensor,
    input_scale: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    input_rows, reduction = input_q.shape
    num_experts, output_size, weight_reduction = weight_q.shape
    top_k = topk_ids.shape[1]
    assert reduction == weight_reduction
    assert input_rows == 1
    hl.specialize(num_experts)
    hl.specialize(output_size)
    hl.specialize(reduction)
    hl.specialize(top_k)
    output = torch.empty(
        (top_k, output_size), dtype=torch.bfloat16, device=input_q.device
    )
    flat_ids = topk_ids.view(top_k)
    flat_weight = weight_q.view(num_experts * output_size, reduction)
    flat_scale = weight_scale.view(num_experts * (output_size // 128), reduction // 128)
    flat_output = output.view(top_k * output_size, 1)
    for tile_row in hl.tile(top_k * output_size):
        expert_slot = tile_row.index // output_size
        expert_row = tile_row.index % output_size
        expert = flat_ids[expert_slot].to(torch.int64)
        selected_row = expert * output_size + expert_row
        accumulator = hl.zeros([tile_row, 1], dtype=torch.float32)
        for tile_k in hl.tile(reduction, block_size=128):
            partial = hl.dot(
                flat_weight[selected_row, tile_k],
                input_q[:, tile_k].T,
            ).to(torch.float32)
            activation_scale = input_scale[:, tile_k.id].to(torch.float32)
            scale_row = expert * (output_size // 128) + expert_row // 128
            block_scale = flat_scale[scale_row, tile_k.id].to(torch.float32)
            accumulator += partial * activation_scale * block_scale[:, None]
        flat_output[tile_row, :] = accumulator.to(output.dtype)
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def selected_expert_w2_fp8(
    input_q: torch.Tensor,
    input_scale: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    top_k, reduction = input_q.shape
    num_experts, output_size, weight_reduction = weight_q.shape
    assert reduction == weight_reduction
    hl.specialize(num_experts)
    hl.specialize(output_size)
    hl.specialize(reduction)
    hl.specialize(top_k)
    output = torch.empty(
        (top_k, output_size), dtype=torch.bfloat16, device=input_q.device
    )
    flat_ids = topk_ids.view(top_k)
    flat_scale = weight_scale.view(num_experts * (output_size // 128), reduction // 128)
    for tile_slot, tile_n in hl.tile([top_k, output_size], block_size=[1, None]):
        expert = flat_ids[tile_slot.begin].to(torch.int64)
        accumulator = hl.zeros([tile_slot, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(reduction, block_size=128):
            partial = hl.dot(
                input_q[tile_slot, tile_k],
                weight_q[expert, tile_n, tile_k].T,
            ).to(torch.float32)
            activation_scale = input_scale[tile_slot, tile_k.id].to(torch.float32)
            scale_row = expert * (output_size // 128) + tile_n.index // 128
            block_scale = flat_scale[scale_row, tile_k.id].to(torch.float32)
            accumulator += partial * activation_scale[:, None] * block_scale[None, :]
        output[tile_slot, tile_n] = accumulator.to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def swiglu_k128_quant(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, twice_intermediate = gate_up.shape
    intermediate = twice_intermediate // 2
    groups = intermediate // 128
    hl.specialize(intermediate)
    output_q = torch.empty((rows, intermediate), dtype=FP8_DTYPE, device=gate_up.device)
    output_scale = torch.empty(
        (rows, groups), dtype=torch.float32, device=gate_up.device
    )
    output_3d = output_q.view(rows, groups, 128)
    for tile_r, tile_g, tile_k in hl.tile(
        [rows, groups, 128], block_size=[1, None, 128]
    ):
        indices = tile_g.index[:, None] * 128 + tile_k.index[None, :]
        gate = gate_up[tile_r, indices].to(torch.float32)
        up = gate_up[tile_r, indices + intermediate].to(torch.float32)
        # Production consumes a BF16 W13 workspace before activation.
        activated = gate * torch.sigmoid(gate) * up
        scale = torch.amax(torch.abs(activated), dim=-1).clamp(min=FP8_EPS) / FP8_MAX
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
        output_scale[tile_r, tile_g] = scale
        output_3d[tile_r, tile_g, tile_k] = (
            (activated / scale[:, :, None]).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
        )
    return output_q, output_scale


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def block_scaled_fp8_mm(
    input_q: torch.Tensor,
    input_scale: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    rows, reduction = input_q.shape
    output_size, weight_reduction = weight_q.shape
    assert reduction == weight_reduction
    hl.specialize(output_size)
    hl.specialize(reduction)
    output = torch.empty(
        (rows, output_size), dtype=torch.bfloat16, device=input_q.device
    )
    for tile_r, tile_n in hl.tile([rows, output_size], block_size=[1, None]):
        accumulator = hl.zeros([tile_r, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(reduction, block_size=128):
            partial = hl.dot(input_q[tile_r, tile_k], weight_q[tile_n, tile_k].T).to(
                torch.float32
            )
            activation_scale = input_scale[tile_r, tile_k.id].to(torch.float32)
            block_scale = weight_scale[tile_n.index // 128, tile_k.id].to(torch.float32)
            accumulator += partial * activation_scale[:, None] * block_scale[None, :]
        output[tile_r, tile_n] = accumulator.to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def weighted_reduce(
    expert_outputs: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    top_k, hidden = expert_outputs.shape
    top_k = hl.specialize(top_k)
    output = torch.empty(
        (1, hidden), dtype=expert_outputs.dtype, device=expert_outputs.device
    )
    for tile_n in hl.tile(hidden):
        values = expert_outputs[:, tile_n].to(torch.float32)
        weights = topk_weights[:, :].view(top_k)
        output[:, tile_n] = torch.sum(
            values * weights[:, None], dim=0, keepdim=True
        ).to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def add_outputs(routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    rows, columns = routed.shape
    output = torch.empty_like(routed)
    for tile_r, tile_n in hl.tile([rows, columns], block_size=[1, None]):
        output[tile_r, tile_n] = routed[tile_r, tile_n] + shared[tile_r, tile_n]
    return output


def _compile(name: str, kernel, kernel_args):
    bound = kernel.bind(kernel_args)
    values = dict(bound.config_spec.default_config())
    values.update(CONFIGS[name])
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    compiled = bound.compile_config(config)
    compiled(*kernel_args)
    torch.cuda.synchronize()
    return compiled


def build(tensors: dict[str, torch.Tensor], shape):
    """Build the independently tuned, matched multi-launch Helion graph."""

    hidden = tensors["hidden"]
    router = _compile(
        "router",
        router_mm_fp32,
        (hidden, tensors["router_weight"]),
    )
    logits = router(hidden, tensors["router_weight"])
    topk_args = (
        logits,
        tensors["correction_bias"],
        shape.top_k,
        shape.num_groups,
        shape.topk_groups,
        shape.routed_scale,
    )
    topk = _compile("topk", grouped_topk, topk_args)
    weights, ids = topk(*topk_args)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        ids,
        tensors["reference_topk_ids"],
        rtol=0,
        atol=0,
    )

    routed_input_quant = _compile(
        "routed_input_quant",
        dynamic_k128_quant,
        (hidden,),
    )
    routed_q, routed_scale = routed_input_quant(hidden)
    expert_w13_args = (
        routed_q,
        routed_scale,
        tensors["expert_w13_q"],
        tensors["expert_w13_scale"],
        ids,
    )
    expert_w13 = _compile(
        "expert_w13",
        selected_expert_w13_fp8,
        expert_w13_args,
    )
    gate_up = expert_w13(*expert_w13_args)
    expert_activation_quant = _compile(
        "expert_act_quant",
        swiglu_k128_quant,
        (gate_up,),
    )
    activation_q, activation_scale = expert_activation_quant(gate_up)
    expert_w2_args = (
        activation_q,
        activation_scale,
        tensors["expert_w2_q"],
        tensors["expert_w2_scale"],
        ids,
    )
    expert_w2 = _compile(
        "expert_w2",
        selected_expert_w2_fp8,
        expert_w2_args,
    )
    expert_outputs = expert_w2(*expert_w2_args)
    expert_reduce = _compile(
        "expert_reduce",
        weighted_reduce,
        (expert_outputs, weights),
    )

    shared_input_quant = _compile(
        "shared_input_quant",
        dynamic_k128_quant,
        (hidden,),
    )
    shared_q, shared_scale = shared_input_quant(hidden)
    shared_w13_args = (
        shared_q,
        shared_scale,
        tensors["shared_w13_q"],
        tensors["shared_w13_scale"],
    )
    shared_w13 = _compile(
        "shared_w13",
        block_scaled_fp8_mm,
        shared_w13_args,
    )
    shared_gate_up = shared_w13(*shared_w13_args)
    shared_activation_quant = _compile(
        "shared_act_quant",
        swiglu_k128_quant,
        (shared_gate_up,),
    )
    shared_activation_q, shared_activation_scale = shared_activation_quant(
        shared_gate_up
    )
    shared_w2_args = (
        shared_activation_q,
        shared_activation_scale,
        tensors["shared_w2_q"],
        tensors["shared_w2_scale"],
    )
    shared_w2 = _compile(
        "shared_w2",
        block_scaled_fp8_mm,
        shared_w2_args,
    )
    shared_output = shared_w2(*shared_w2_args)
    routed_output = expert_reduce(expert_outputs, weights)
    final_add = _compile(
        "final_add",
        add_outputs,
        (routed_output, shared_output),
    )

    shared_stream = torch.cuda.Stream()

    def launch():
        current_stream = torch.cuda.current_stream()
        shared_stream.wait_stream(current_stream)
        local_logits = router(hidden, tensors["router_weight"])
        with torch.cuda.stream(shared_stream):
            local_shared_q, local_shared_scale = shared_input_quant(hidden)
            local_shared_gate_up = shared_w13(
                local_shared_q,
                local_shared_scale,
                tensors["shared_w13_q"],
                tensors["shared_w13_scale"],
            )
            local_shared_activation_q, local_shared_activation_scale = (
                shared_activation_quant(local_shared_gate_up)
            )
            local_shared_output = shared_w2(
                local_shared_activation_q,
                local_shared_activation_scale,
                tensors["shared_w2_q"],
                tensors["shared_w2_scale"],
            )

        local_weights, local_ids = topk(
            local_logits,
            tensors["correction_bias"],
            shape.top_k,
            shape.num_groups,
            shape.topk_groups,
            shape.routed_scale,
        )
        local_routed_q, local_routed_scale = routed_input_quant(hidden)
        local_gate_up = expert_w13(
            local_routed_q,
            local_routed_scale,
            tensors["expert_w13_q"],
            tensors["expert_w13_scale"],
            local_ids,
        )
        local_activation_q, local_activation_scale = expert_activation_quant(
            local_gate_up
        )
        local_expert_outputs = expert_w2(
            local_activation_q,
            local_activation_scale,
            tensors["expert_w2_q"],
            tensors["expert_w2_scale"],
            local_ids,
        )
        local_routed_output = expert_reduce(local_expert_outputs, local_weights)
        current_stream.wait_stream(shared_stream)
        local_output = final_add(local_routed_output, local_shared_output)
        return (
            local_output,
            local_logits,
            local_weights,
            local_ids,
            local_routed_q,
            local_routed_scale,
            local_gate_up,
            local_activation_q,
            local_activation_scale,
            local_expert_outputs,
            local_routed_output,
            local_shared_q,
            local_shared_scale,
            local_shared_gate_up,
            local_shared_activation_q,
            local_shared_activation_scale,
            local_shared_output,
        )

    output = launch()
    torch.cuda.synchronize()
    return launch, output
