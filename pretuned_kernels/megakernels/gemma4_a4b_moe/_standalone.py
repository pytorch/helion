# ruff: noqa: ANN001, ANN202
"""Matched separate-launch Helion baseline for the Gemma 4 A4B MoE probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


CONFIGS: dict[str, dict[str, object]] = {
    "router_project": {
        "block_sizes": [8],
        "num_warps": 4,
        "num_stages": 1,
    },
    "route_candidates": {"num_warps": 4, "num_stages": 1},
    "route_merge": {"num_warps": 4, "num_stages": 1},
    "expert_gate_up": {
        "block_sizes": [16, 256],
        "num_warps": 4,
        "num_stages": 1,
        "range_num_stages": [0, 3],
        "range_multi_buffers": [None, True],
        "range_unroll_factors": [0, 0],
    },
    "expert_geglu": {
        "block_sizes": [128],
        "num_warps": 2,
        "num_stages": 3,
    },
    "expert_down": {
        "block_sizes": [64, 64],
        "num_warps": 4,
        "num_stages": 1,
        "range_num_stages": [0, 5],
        "range_multi_buffers": [None, False],
    },
    "expert_reduce": {
        "block_sizes": [256],
        "num_warps": 2,
        "num_stages": 4,
    },
    "post_norm": {"num_warps": 8, "num_stages": 1},
}


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def router_project(residual, router_scale, root_size, router_weight, eps):
    batch, hidden = residual.size()
    num_experts, weight_hidden = router_weight.size()
    assert hidden == weight_hidden
    hl.specialize(hidden)
    hl.specialize(num_experts)
    logits = torch.empty(
        (batch, num_experts), dtype=torch.float32, device=residual.device
    )
    for tile_m, tile_expert in hl.tile([batch, num_experts], block_size=[1, None]):
        token = tile_m.begin
        row = residual[token, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(row * row, dim=-1) + eps)
        root = hl.load(root_size, [])
        normalized = (row * inv_rms).to(residual.dtype)
        router_input = (normalized * root * router_scale[:]).to(residual.dtype)
        weights = router_weight[tile_expert, :].to(torch.float32)
        logits[token, tile_expert] = torch.sum(
            weights * router_input.to(torch.float32), dim=-1
        )
    return logits


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def route_candidates(logits, top_k):
    batch, num_experts = logits.size()
    top_k = hl.specialize(top_k)
    hl.specialize(num_experts)
    groups = 4
    group_size = num_experts // groups
    values_out = torch.empty(
        (batch, groups, top_k), dtype=torch.float32, device=logits.device
    )
    ids_out = torch.empty(
        (batch, groups, top_k), dtype=torch.int32, device=logits.device
    )
    for tile_m, tile_group in hl.tile([batch, groups], block_size=[1, 1]):
        token = tile_m.begin
        group = tile_group.begin
        experts = group * group_size + hl.arange(group_size)
        values, ids = torch.topk(logits[token, experts], top_k, dim=-1, largest=True)
        values_out[token, group, :] = values
        ids_out[token, group, :] = ids.to(torch.int32) + group * group_size
    return values_out, ids_out


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def route_merge(candidate_values, candidate_ids, per_expert_scale, top_k):
    batch, groups, candidates_per_group = candidate_values.size()
    top_k = hl.specialize(top_k)
    candidate_count = groups * candidates_per_group
    values_flat = candidate_values.view(batch, candidate_count)
    ids_flat = candidate_ids.view(batch, candidate_count)
    weights_out = torch.empty(
        (batch, top_k), dtype=torch.float32, device=candidate_values.device
    )
    ids_out = torch.empty(
        (batch, top_k), dtype=torch.int32, device=candidate_values.device
    )
    for tile_m in hl.tile(batch, block_size=1):
        token = tile_m.begin
        values, positions = torch.topk(
            values_flat[token, :], top_k, dim=-1, largest=True
        )
        ids = ids_flat[token, positions]
        shifted = values - torch.amax(values, dim=-1, keepdim=True)
        raw_weights = torch.exp(shifted)
        normalized = raw_weights / torch.sum(raw_weights, dim=-1, keepdim=True)
        weights_out[token, :] = normalized * per_expert_scale[ids].to(torch.float32)
        ids_out[token, :] = ids
    return weights_out, ids_out


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def expert_gate_up(
    residual,
    norm_weight,
    expert_weight,
    topk_ids,
    topk_weights,
    eps,
):
    batch, hidden = residual.size()
    num_experts, twice_intermediate, weight_hidden = expert_weight.size()
    assert hidden == weight_hidden
    intermediate = twice_intermediate // 2
    top_k = topk_ids.size(1)
    hl.specialize(num_experts)
    hl.specialize(hidden)
    hl.specialize(intermediate)
    flat_weight = expert_weight.view(num_experts * twice_intermediate, hidden)
    gate_up = torch.empty(
        (batch * top_k, twice_intermediate),
        dtype=residual.dtype,
        device=residual.device,
    )
    selected_ids = torch.empty_like(topk_ids)
    selected_weights = torch.empty_like(topk_weights)
    for tile_m, tile_slot, tile_i in hl.tile(
        [batch, top_k, intermediate], block_size=[1, 1, None]
    ):
        token = tile_m.begin
        slot = tile_slot.begin
        selected_expert = topk_ids[token, slot]
        gate_row = selected_expert * twice_intermediate + tile_i.index
        up_row = gate_row + intermediate
        row = residual[token, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(row * row, dim=-1) + eps)
        gate_acc = hl.zeros([tile_i], dtype=torch.float32)
        up_acc = hl.zeros([tile_i], dtype=torch.float32)
        for tile_k in hl.tile(hidden):
            values = residual[token, tile_k].to(torch.float32)
            normalized = (values * inv_rms).to(residual.dtype)
            expert_input = (normalized * norm_weight[tile_k]).to(residual.dtype)
            gate_weights = flat_weight[gate_row, tile_k].to(torch.float32)
            up_weights = flat_weight[up_row, tile_k].to(torch.float32)
            input_fp32 = expert_input.to(torch.float32)
            gate_acc = gate_acc + torch.sum(gate_weights * input_fp32, dim=-1)
            up_acc = up_acc + torch.sum(up_weights * input_fp32, dim=-1)
        flat_row = token * top_k + slot
        gate_up[flat_row, tile_i] = gate_acc.to(gate_up.dtype)
        gate_up[flat_row, tile_i.index + intermediate] = up_acc.to(gate_up.dtype)
        if tile_i.begin == 0:
            selected_ids[token, slot] = selected_expert
            selected_weights[token, slot] = topk_weights[token, slot]
    return gate_up, selected_ids, selected_weights


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def expert_geglu(gate_up):
    assignments, twice_intermediate = gate_up.size()
    intermediate = twice_intermediate // 2
    output = torch.empty(
        (assignments, intermediate), dtype=gate_up.dtype, device=gate_up.device
    )
    for tile_m, tile_i in hl.tile([assignments, intermediate], block_size=[1, None]):
        gate = gate_up[tile_m, tile_i].to(torch.float32)
        up = gate_up[tile_m, tile_i + intermediate]
        output[tile_m, tile_i] = (
            0.5
            * gate
            * (
                1.0
                + torch.tanh(
                    0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
                )
            )
        ).to(up.dtype) * up
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def expert_down(activation, expert_weight, selected_ids, selected_weights):
    batch, top_k = selected_ids.size()
    assignments, intermediate = activation.size()
    num_experts, hidden, weight_intermediate = expert_weight.size()
    assert assignments == batch * top_k
    assert intermediate == weight_intermediate
    hl.specialize(num_experts)
    hl.specialize(hidden)
    flat_weight = expert_weight.view(num_experts * hidden, intermediate)
    output = torch.empty(
        (batch, top_k, hidden), dtype=activation.dtype, device=activation.device
    )
    for tile_m, tile_slot, tile_n in hl.tile(
        [batch, top_k, hidden], block_size=[1, 1, None]
    ):
        token = tile_m.begin
        slot = tile_slot.begin
        selected_expert = selected_ids[token, slot]
        selected_row = selected_expert * hidden + tile_n.index
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_k in hl.tile(intermediate):
            values = activation[token * top_k + slot, tile_k].to(torch.float32)
            weights = flat_weight[selected_row, tile_k].to(torch.float32)
            acc = acc + torch.sum(weights * values, dim=-1)
        route_weight = selected_weights[token, slot].to(torch.float32)
        output[token, slot, tile_n] = (acc * route_weight).to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def expert_reduce(expert_outputs):
    batch, _top_k, hidden = expert_outputs.size()
    output = torch.empty(
        (batch, hidden), dtype=expert_outputs.dtype, device=expert_outputs.device
    )
    for tile_m, tile_n in hl.tile([batch, hidden], block_size=[1, None]):
        values = expert_outputs[tile_m, :, tile_n].to(torch.float32)
        output[tile_m, tile_n] = torch.sum(values, dim=1).to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def post_norm(x, weight, eps):
    batch, hidden = x.size()
    hl.specialize(hidden)
    output = torch.empty_like(x)
    for tile_m in hl.tile(batch, block_size=1):
        values = x[tile_m, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(values * values, dim=-1) + eps)
        normalized = (values * inv_rms[:, None]).to(x.dtype)
        output[tile_m, :] = normalized * weight[None, :]
    return output


def _compile(kernel, args, config_name: str):
    bound = kernel.bind(args)
    config = helion.Config.from_dict(CONFIGS[config_name])
    bound.config_spec.normalize(config.config)
    return bound.compile_config(config)


def build(
    tensors: dict[str, torch.Tensor], top_k: int, eps: float
) -> tuple[Callable[[], tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
    """Compile the independently tuned eight-launch graph and return its closure."""
    router_args = (
        tensors["residual"],
        tensors["router_scale"],
        tensors["root_size"],
        tensors["router_weight"],
        eps,
    )
    router = _compile(router_project, router_args, "router_project")
    router_logits = router(*router_args)

    candidate_args = (router_logits, top_k)
    candidates = _compile(route_candidates, candidate_args, "route_candidates")
    candidate_values, candidate_ids = candidates(*candidate_args)

    merge_args = (candidate_values, candidate_ids, tensors["per_expert_scale"], top_k)
    merge = _compile(route_merge, merge_args, "route_merge")
    topk_weights, topk_ids = merge(*merge_args)

    gate_up_args = (
        tensors["residual"],
        tensors["pre_ff_norm_weight"],
        tensors["expert_gate_up_weight"],
        topk_ids,
        topk_weights,
        eps,
    )
    gate_up_kernel = _compile(expert_gate_up, gate_up_args, "expert_gate_up")
    gate_up, selected_ids, selected_weights = gate_up_kernel(*gate_up_args)

    geglu_args = (gate_up,)
    geglu = _compile(expert_geglu, geglu_args, "expert_geglu")
    activation = geglu(*geglu_args)

    down_args = (
        activation,
        tensors["expert_down_weight"],
        selected_ids,
        selected_weights,
    )
    down = _compile(expert_down, down_args, "expert_down")
    expert_outputs = down(*down_args)

    reduce_args = (expert_outputs,)
    reduce = _compile(expert_reduce, reduce_args, "expert_reduce")
    moe_down = reduce(*reduce_args)

    post_args = (moe_down, tensors["post_ff_norm_weight"], eps)
    post = _compile(post_norm, post_args, "post_norm")
    moe_branch = post(*post_args)

    def launch() -> tuple[torch.Tensor, ...]:
        local_logits = router(*router_args)
        local_candidates, local_candidate_ids = candidates(local_logits, top_k)
        local_weights, local_ids = merge(
            local_candidates,
            local_candidate_ids,
            tensors["per_expert_scale"],
            top_k,
        )
        local_gate_up, local_selected_ids, local_selected_weights = gate_up_kernel(
            tensors["residual"],
            tensors["pre_ff_norm_weight"],
            tensors["expert_gate_up_weight"],
            local_ids,
            local_weights,
            eps,
        )
        local_activation = geglu(local_gate_up)
        local_expert_outputs = down(
            local_activation,
            tensors["expert_down_weight"],
            local_selected_ids,
            local_selected_weights,
        )
        local_down = reduce(local_expert_outputs)
        local_branch = post(local_down, tensors["post_ff_norm_weight"], eps)
        return (
            local_branch,
            local_logits,
            local_weights,
            local_ids,
            local_activation.view(tensors["residual"].size(0), top_k, -1),
            local_expert_outputs,
            local_down,
        )

    outputs = (
        moe_branch,
        router_logits,
        topk_weights,
        topk_ids,
        activation.view(tensors["residual"].size(0), top_k, -1),
        expert_outputs,
        moe_down,
    )
    return launch, outputs
