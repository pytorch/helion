# ruff: noqa: ANN001, ANN202
# pyrefly: ignore-errors
"""Single-kernel Helion stress tests for the Gemma 4 26B-A4B MoE sub-layer.

The probe includes a verbatim production decomposition, assignment-local
source experiments, and a composition of the production grouped kernels. Its
Helion baselines and optional hand-written Triton comparison live beside it.
"""

from __future__ import annotations

import argparse
import ast
import json
import linecache
from pathlib import Path

import torch

from probes.gemma4.common import benchmark_interleaved
from probes.gemma4.common import capture
from probes.gemma4.common import visible_gpu_pids
from probes.gemma4.gemma4_a4b_moe_common import Gemma4A4BMoEShape
from probes.gemma4.gemma4_a4b_moe_common import allocate_moe
from probes.gemma4.gemma4_a4b_moe_common import max_aligned_tiles
from probes.gemma4.gemma4_a4b_moe_common import moe_reference
from probes.gemma4.gemma4_a4b_moe_common import tiles_per_expert as tiles_per_expert_of
import probes.gemma4.helion_gemma4_a4b_moe as separate
from probes.gemma4.helion_gemma4_e4b_megakernel import _Bridge
from probes.gemma4.helion_gemma4_e4b_megakernel import _helion_resources
from probes.gemma4.helion_gemma4_e4b_megakernel import _inline_invocation
from probes.gemma4.helion_gemma4_e4b_megakernel import _Invocation
import probes.gemma4.triton_gemma4_a4b_moe_megakernel as hand_triton

import helion
import helion.language as hl

_gelu_tanh = separate._gelu_tanh


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def router_project_by_expert(
    hidden: torch.Tensor,
    scale: torch.Tensor,
    root_size: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Project expert blocks while rematerializing the batch-1 router norm."""
    m, k = hidden.size()
    num_experts, weight_k = weight.size()
    assert k == weight_k
    hl.specialize(k)
    hl.specialize(num_experts)
    output = torch.empty((m, num_experts), dtype=torch.float32, device=hidden.device)
    for tile_m, tile_expert in hl.tile([m, num_experts], block_size=[1, None]):
        token = tile_m.begin
        row = hidden[token, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(row * row, dim=-1) + eps)
        root = hl.load(root_size, [])
        normalized = (row * inv_rms).to(hidden.dtype)
        router_input = (normalized * root * scale[:]).to(hidden.dtype)
        weights = weight[tile_expert, :].to(torch.float32)
        acc = torch.sum(weights * router_input.to(torch.float32), dim=-1)
        output[token, tile_expert] = acc
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def route_topk_iterative(
    logits: torch.Tensor,
    per_expert_scale: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k experts with repeated max/argmax instead of full ranking."""
    m, num_experts = logits.size()
    top_k = hl.specialize(top_k)
    hl.specialize(num_experts)
    weights_out = torch.empty((m, top_k), dtype=torch.float32, device=logits.device)
    ids_out = torch.empty((m, top_k), dtype=torch.int32, device=logits.device)
    for tile_m in hl.tile(m, block_size=1):
        token = tile_m.begin
        values = logits[token, :]
        experts = hl.arange(num_experts)
        slots = hl.arange(top_k)
        chosen = hl.zeros([top_k], dtype=torch.int32)
        raw_weights = hl.zeros([top_k], dtype=torch.float32)
        largest = torch.amax(values, dim=-1)
        for slot in hl.static_range(top_k):
            value = torch.amax(values, dim=-1)
            expert = torch.argmax(values, dim=-1).to(torch.int32)
            chosen = torch.where(slots == slot, expert, chosen)
            raw_weights = torch.where(
                slots == slot,
                torch.exp(value - largest),
                raw_weights,
            )
            values = torch.where(experts == expert, float("-inf"), values)
        normalized = raw_weights / torch.sum(raw_weights, dim=-1)
        weights_out[token, :] = normalized * per_expert_scale[chosen].to(torch.float32)
        ids_out[token, :] = chosen
    return weights_out, ids_out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def route_topk_candidates(
    logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k candidates independently from four expert groups."""
    m, num_experts = logits.size()
    top_k = hl.specialize(top_k)
    hl.specialize(num_experts)
    groups = 4
    group_size = num_experts // groups
    values_out = torch.empty(
        (m, groups, top_k), dtype=torch.float32, device=logits.device
    )
    ids_out = torch.empty((m, groups, top_k), dtype=torch.int32, device=logits.device)
    for tile_m, tile_group in hl.tile([m, groups], block_size=[1, 1]):
        token = tile_m.begin
        group = tile_group.begin
        experts = group * group_size + hl.arange(group_size)
        values, ids = torch.topk(logits[token, experts], top_k, dim=-1, largest=True)
        values_out[token, group, :] = values
        ids_out[token, group, :] = ids.to(torch.int32) + group * group_size
    return values_out, ids_out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def route_topk_merge(
    candidate_values: torch.Tensor,
    candidate_ids: torch.Tensor,
    per_expert_scale: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge grouped candidates and compute final normalized route weights."""
    m, groups, candidates_per_group = candidate_values.size()
    top_k = hl.specialize(top_k)
    candidate_count = groups * candidates_per_group
    values_flat = candidate_values.view(m, candidate_count)
    ids_flat = candidate_ids.view(m, candidate_count)
    weights_out = torch.empty(
        (m, top_k), dtype=torch.float32, device=candidate_values.device
    )
    ids_out = torch.empty((m, top_k), dtype=torch.int32, device=candidate_values.device)
    for tile_m in hl.tile(m, block_size=1):
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


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def pre_norm_expert_geglu_by_assignment(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    expert_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rematerialize pre-norm in assignment-local gate/up + GeGLU tasks."""
    m, hidden_size = hidden.size()
    num_experts, twice_intermediate, weight_hidden = expert_weight.size()
    assert hidden_size == weight_hidden
    intermediate = twice_intermediate // 2
    top_k = topk_ids.size(1)
    hl.specialize(num_experts)
    hl.specialize(hidden_size)
    hl.specialize(intermediate)
    flattened_weight = expert_weight.view(num_experts * twice_intermediate, hidden_size)
    output = torch.empty(
        (m, top_k, intermediate), dtype=hidden.dtype, device=hidden.device
    )
    selected_ids = torch.empty((m, top_k), dtype=torch.int32, device=hidden.device)
    selected_weights = torch.empty(
        (m, top_k), dtype=torch.float32, device=hidden.device
    )
    for tile_m, tile_slot, tile_i in hl.tile(
        [m, top_k, intermediate], block_size=[1, 1, None]
    ):
        token = tile_m.begin
        slot = tile_slot.begin
        selected_expert = topk_ids[token, slot]
        gate_row = selected_expert * twice_intermediate + tile_i.index
        up_row = gate_row + intermediate
        row = hidden[token, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(row * row, dim=-1) + eps)
        gate_acc = hl.zeros([tile_i], dtype=torch.float32)
        up_acc = hl.zeros([tile_i], dtype=torch.float32)
        for tile_k in hl.tile(hidden_size):
            values = hidden[token, tile_k].to(torch.float32)
            normalized = (values * inv_rms).to(hidden.dtype)
            expert_input = (normalized * norm_weight[tile_k]).to(hidden.dtype)
            gate_weight = flattened_weight[gate_row, tile_k].to(torch.float32)
            up_weight = flattened_weight[up_row, tile_k].to(torch.float32)
            input_fp32 = expert_input.to(torch.float32)
            gate_acc = gate_acc + torch.sum(
                gate_weight * input_fp32,
                dim=-1,
            )
            up_acc = up_acc + torch.sum(
                up_weight * input_fp32,
                dim=-1,
            )
        gate = gate_acc.to(torch.bfloat16).to(torch.float32)
        up = up_acc.to(torch.bfloat16)
        output[token, slot, tile_i] = _gelu_tanh(gate).to(up.dtype) * up
        if tile_i.begin == 0:
            selected_ids[token, slot] = selected_expert
            selected_weights[token, slot] = topk_weights[token, slot]
    return output, selected_ids, selected_weights


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def pre_norm_expert_gate_up_by_assignment(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    expert_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rematerialize pre-norm but preserve the gate/up materialization."""
    m, hidden_size = hidden.size()
    num_experts, twice_intermediate, weight_hidden = expert_weight.size()
    assert hidden_size == weight_hidden
    intermediate = twice_intermediate // 2
    top_k = topk_ids.size(1)
    hl.specialize(num_experts)
    hl.specialize(hidden_size)
    hl.specialize(intermediate)
    flattened_weight = expert_weight.view(num_experts * twice_intermediate, hidden_size)
    output = torch.empty(
        (m * top_k, twice_intermediate), dtype=hidden.dtype, device=hidden.device
    )
    selected_ids = torch.empty((m, top_k), dtype=torch.int32, device=hidden.device)
    selected_weights = torch.empty(
        (m, top_k), dtype=torch.float32, device=hidden.device
    )
    for tile_m, tile_slot, tile_i in hl.tile(
        [m, top_k, intermediate], block_size=[1, 1, None]
    ):
        token = tile_m.begin
        slot = tile_slot.begin
        selected_expert = topk_ids[token, slot]
        gate_row = selected_expert * twice_intermediate + tile_i.index
        up_row = gate_row + intermediate
        row = hidden[token, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(row * row, dim=-1) + eps)
        gate_acc = hl.zeros([tile_i], dtype=torch.float32)
        up_acc = hl.zeros([tile_i], dtype=torch.float32)
        for tile_k in hl.tile(hidden_size):
            values = hidden[token, tile_k].to(torch.float32)
            normalized = (values * inv_rms).to(hidden.dtype)
            expert_input = (normalized * norm_weight[tile_k]).to(hidden.dtype)
            gate_weight = flattened_weight[gate_row, tile_k].to(torch.float32)
            up_weight = flattened_weight[up_row, tile_k].to(torch.float32)
            input_fp32 = expert_input.to(torch.float32)
            gate_acc = gate_acc + torch.sum(gate_weight * input_fp32, dim=-1)
            up_acc = up_acc + torch.sum(up_weight * input_fp32, dim=-1)
        output[token * top_k + slot, tile_i] = gate_acc.to(output.dtype)
        output[token * top_k + slot, tile_i.index + intermediate] = up_acc.to(
            output.dtype
        )
        if tile_i.begin == 0:
            selected_ids[token, slot] = selected_expert
            selected_weights[token, slot] = topk_weights[token, slot]
    return output, selected_ids, selected_weights


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def expert_down_by_assignment(
    activation: torch.Tensor,
    expert_weight: torch.Tensor,
    selected_ids: torch.Tensor,
    selected_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute one assignment's output columns per task."""
    m, top_k, intermediate = activation.size()
    num_experts, hidden_size, weight_intermediate = expert_weight.size()
    assert intermediate == weight_intermediate
    hl.specialize(num_experts)
    hl.specialize(hidden_size)
    flattened_weight = expert_weight.view(num_experts * hidden_size, intermediate)
    output = torch.empty(
        (m, top_k, hidden_size),
        dtype=activation.dtype,
        device=activation.device,
    )
    for tile_m, tile_slot, tile_n in hl.tile(
        [m, top_k, hidden_size], block_size=[1, 1, None]
    ):
        token = tile_m.begin
        slot = tile_slot.begin
        selected_expert = selected_ids[token, slot]
        selected_row = selected_expert * hidden_size + tile_n.index
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_k in hl.tile(intermediate):
            values = activation[token, slot, tile_k].to(torch.float32)
            weights = flattened_weight[selected_row, tile_k].to(torch.float32)
            acc = acc + torch.sum(weights * values, dim=-1)
        weight = selected_weights[token, slot].to(torch.float32)
        output[token, slot, tile_n] = (acc * weight).to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def expert_sum(expert_output: torch.Tensor) -> torch.Tensor:
    m, top_k, hidden_size = expert_output.size()
    output = torch.empty(
        (m, hidden_size), dtype=expert_output.dtype, device=expert_output.device
    )
    for tile_m, tile_n in hl.tile([m, hidden_size], block_size=[1, None]):
        values = expert_output[tile_m, :, tile_n].to(torch.float32)
        output[tile_m, tile_n] = torch.sum(values, dim=1).to(output.dtype)
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def grouped_expert_geglu_task_major(
    expert_input: torch.Tensor,
    expert_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    active_tiles: torch.Tensor,
    tile_end: torch.Tensor,
    expert_counts: torch.Tensor,
    order: torch.Tensor,
    tile_tokens: int,
    tiles_per_expert: int,
    top_k: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Produce grouped activations and routing metadata in task coordinates."""
    batch, hidden = expert_input.size()
    num_experts, twice_intermediate, weight_hidden = expert_weight.size()
    assert hidden == weight_hidden
    intermediate = twice_intermediate // 2
    max_active = active_tiles.size(0)
    tile_tokens = hl.specialize(tile_tokens)
    tiles_per_expert = hl.specialize(tiles_per_expert)
    top_k = hl.specialize(top_k)
    hl.specialize(num_experts)
    hl.specialize(intermediate)
    assignments = batch * top_k
    expert_stride = tiles_per_expert * tile_tokens
    flat_weight = expert_weight.view(num_experts * twice_intermediate, hidden)
    flat_route_weights = topk_weights.view(assignments)
    output = torch.empty(
        (max_active, tile_tokens, intermediate),
        dtype=expert_input.dtype,
        device=expert_input.device,
    )
    active_out = torch.empty(
        (max_active,), dtype=torch.int32, device=expert_input.device
    )
    expert_out = torch.empty(
        (max_active,), dtype=torch.int32, device=expert_input.device
    )
    assignment_out = torch.empty(
        (max_active, tile_tokens), dtype=torch.int32, device=expert_input.device
    )
    route_weight_out = torch.empty(
        (max_active, tile_tokens), dtype=torch.float32, device=expert_input.device
    )
    for tile_t, tile_i in hl.tile([max_active, intermediate], block_size=[1, None]):
        active = tile_t.id < hl.load(tile_end, [num_experts - 1])
        group = hl.load(active_tiles, [tile_t.id], extra_mask=active)
        expert = group // tiles_per_expert
        local = group - expert * tiles_per_expert
        local_rows = local * tile_tokens + hl.arange(tile_tokens)
        count = hl.load(expert_counts, [expert], extra_mask=active)
        row_valid = active & (local_rows < count)
        assignment = hl.load(
            order,
            [expert * expert_stride + local_rows],
            extra_mask=row_valid,
        )
        route_weight = hl.load(
            flat_route_weights,
            [assignment],
            extra_mask=row_valid,
        )
        if tile_i.begin == 0:
            hl.store(active_out, [tile_t.id], active.to(torch.int32))
            hl.store(expert_out, [tile_t.id], expert)
            hl.store(
                assignment_out,
                [tile_t.id, hl.arange(tile_tokens)],
                torch.where(row_valid, assignment, -1),
            )
            hl.store(
                route_weight_out,
                [tile_t.id, hl.arange(tile_tokens)],
                torch.where(row_valid, route_weight, 0.0),
            )
        if active:
            token = assignment // top_k
            gate_row = expert * twice_intermediate + tile_i.index
            gate_acc = hl.zeros([tile_tokens, tile_i], dtype=torch.float32)
            up_acc = hl.zeros([tile_tokens, tile_i], dtype=torch.float32)
            for tile_k in hl.tile(hidden):
                values = hl.load(
                    expert_input,
                    [token, tile_k.index],
                    extra_mask=row_valid[:, None],
                )
                gate_acc = torch.addmm(
                    gate_acc, values, flat_weight[gate_row, tile_k].T
                )
                up_acc = torch.addmm(
                    up_acc,
                    values,
                    flat_weight[gate_row + intermediate, tile_k].T,
                )
            gate = gate_acc.to(torch.bfloat16).to(torch.float32)
            up = up_acc.to(torch.bfloat16)
            hl.store(
                output,
                [tile_t.id, hl.arange(tile_tokens), tile_i.index],
                _gelu_tanh(gate).to(up.dtype) * up,
                extra_mask=row_valid[:, None],
            )
    return output, active_out, expert_out, assignment_out, route_weight_out


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def grouped_expert_down_task_major(
    activation: torch.Tensor,
    expert_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    active_tiles: torch.Tensor,
    experts: torch.Tensor,
    assignments: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    """Consume one task-major activation group and scatter weighted outputs."""
    max_active, tile_tokens, intermediate = activation.size()
    num_experts, hidden, weight_intermediate = expert_weight.size()
    assert intermediate == weight_intermediate
    batch, top_k = topk_weights.size()
    hl.specialize(num_experts)
    hl.specialize(hidden)
    flat_weight = expert_weight.view(num_experts * hidden, intermediate)
    output = torch.empty(
        (batch * top_k, hidden), dtype=activation.dtype, device=activation.device
    )
    for tile_t, tile_n in hl.tile([max_active, hidden], block_size=[1, None]):
        active = hl.load(active_tiles, [tile_t.id]) != 0
        if active:
            expert = hl.load(experts, [tile_t.id])
            rows = hl.arange(tile_tokens)
            assignment = hl.load(assignments, [tile_t.id, rows])
            route_weight = hl.load(route_weights, [tile_t.id, rows])
            row_valid = assignment >= 0
            weight_row = expert * hidden + tile_n.index
            acc = hl.zeros([tile_tokens, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(intermediate):
                values = hl.load(
                    activation,
                    [tile_t.id, rows, tile_k.index],
                    extra_mask=row_valid[:, None],
                )
                acc = torch.addmm(
                    acc,
                    values,
                    flat_weight[weight_row, tile_k].T,
                )
            hl.store(
                output,
                [assignment, tile_n.index],
                (acc * route_weight[:, None]).to(output.dtype),
                extra_mask=row_valid[:, None],
            )
    return output


def _matched_events() -> tuple[_Invocation | _Bridge, ...]:
    return (
        _Invocation(
            "expert_pre_norm",
            separate.rms_norm,
            {
                "x": "residual",
                "weight": "pre_ff_norm_weight",
                "eps": "eps",
            },
            {"out": "expert_input"},
        ),
        _Invocation(
            "router_norm_scale",
            separate.router_norm_scale,
            {
                "hidden": "residual",
                "scale": "router_scale",
                "root_size": "root_size",
                "eps": "eps",
            },
            {"output": "router_hidden"},
        ),
        _Invocation(
            "router_mm_fp32",
            separate.router_mm_fp32,
            {"hidden": "router_hidden", "weight": "router_weight"},
            {"output": "router_logits"},
        ),
        _Invocation(
            "route_topk",
            separate.gemma4_route_topk,
            {
                "logits": "router_logits",
                "per_expert_scale": "per_expert_scale",
                "top_k": "top_k",
            },
            {"weights": "topk_weights", "ids": "topk_ids"},
        ),
        _Invocation(
            "expert_gate_up",
            separate.expert_gate_up,
            {
                "hidden": "expert_input",
                "expert_weight": "expert_gate_up_weight",
                "topk_ids": "topk_ids",
            },
            {"output": "gate_up_flat"},
        ),
        _Bridge("gate_up = gate_up_flat.view(top_k, expert_gate_up_weight.size(1))"),
        _Invocation(
            "expert_geglu",
            separate.geglu,
            {"gate_up": "gate_up"},
            {"output": "activation"},
        ),
        _Invocation(
            "expert_down",
            separate.expert_down,
            {
                "activation": "activation",
                "expert_weight": "expert_down_weight",
                "topk_ids": "topk_ids",
            },
            {"output": "expert_outputs"},
        ),
        _Invocation(
            "expert_reduce",
            separate.weighted_expert_reduce,
            {
                "expert_output": "expert_outputs",
                "topk_weights": "topk_weights",
            },
            {"output": "moe_down"},
        ),
        _Invocation(
            "moe_post_norm",
            separate.rms_norm,
            {"x": "moe_down", "weight": "post_ff_norm_weight", "eps": "eps"},
            {"out": "moe_branch"},
        ),
    )


def _assignment_events(
    *, topk_mode: str, fuse_geglu: bool = True
) -> tuple[_Invocation | _Bridge, ...]:
    events: list[_Invocation | _Bridge] = [
        _Invocation(
            "router_project",
            router_project_by_expert,
            {
                "hidden": "residual",
                "scale": "router_scale",
                "root_size": "root_size",
                "weight": "router_weight",
                "eps": "eps",
            },
            {"output": "router_logits"},
        ),
    ]
    if topk_mode == "hierarchical":
        events.extend(
            (
                _Invocation(
                    "route_candidates",
                    route_topk_candidates,
                    {"logits": "router_logits", "top_k": "top_k"},
                    {
                        "values_out": "candidate_values",
                        "ids_out": "candidate_ids",
                    },
                ),
                _Invocation(
                    "route_merge",
                    route_topk_merge,
                    {
                        "candidate_values": "candidate_values",
                        "candidate_ids": "candidate_ids",
                        "per_expert_scale": "per_expert_scale",
                        "top_k": "top_k",
                    },
                    {"weights_out": "topk_weights", "ids_out": "topk_ids"},
                ),
            )
        )
    else:
        topk_kernel = (
            route_topk_iterative
            if topk_mode == "iterative"
            else separate.gemma4_route_topk
        )
        events.append(
            _Invocation(
                "route_topk",
                topk_kernel,
                {
                    "logits": "router_logits",
                    "per_expert_scale": "per_expert_scale",
                    "top_k": "top_k",
                },
                (
                    {"weights_out": "topk_weights", "ids_out": "topk_ids"}
                    if topk_mode == "iterative"
                    else {"weights": "topk_weights", "ids": "topk_ids"}
                ),
            )
        )
    if fuse_geglu:
        events.append(
            _Invocation(
                "expert_geglu",
                pre_norm_expert_geglu_by_assignment,
                {
                    "hidden": "residual",
                    "norm_weight": "pre_ff_norm_weight",
                    "expert_weight": "expert_gate_up_weight",
                    "topk_ids": "topk_ids",
                    "topk_weights": "topk_weights",
                    "eps": "eps",
                },
                {
                    "output": "activation",
                    "selected_ids": "selected_ids",
                    "selected_weights": "selected_weights",
                },
            )
        )
    else:
        events.extend(
            (
                _Invocation(
                    "expert_gate_up",
                    pre_norm_expert_gate_up_by_assignment,
                    {
                        "hidden": "residual",
                        "norm_weight": "pre_ff_norm_weight",
                        "expert_weight": "expert_gate_up_weight",
                        "topk_ids": "topk_ids",
                        "topk_weights": "topk_weights",
                        "eps": "eps",
                    },
                    {
                        "output": "gate_up",
                        "selected_ids": "selected_ids",
                        "selected_weights": "selected_weights",
                    },
                ),
                _Invocation(
                    "expert_geglu",
                    separate.geglu,
                    {"gate_up": "gate_up"},
                    {"output": "activation_flat"},
                ),
                _Bridge(
                    "activation = activation_flat.view("
                    "residual.size(0), top_k, expert_down_weight.size(2))"
                ),
            )
        )
    events.extend(
        (
            _Invocation(
                "expert_down",
                expert_down_by_assignment,
                {
                    "activation": "activation",
                    "expert_weight": "expert_down_weight",
                    "selected_ids": "selected_ids",
                    "selected_weights": "selected_weights",
                },
                {"output": "expert_outputs"},
            ),
            _Invocation(
                "expert_reduce",
                expert_sum,
                {"expert_output": "expert_outputs"},
                {"output": "moe_down"},
            ),
            _Invocation(
                "moe_post_norm",
                separate.rms_norm,
                {"x": "moe_down", "weight": "post_ff_norm_weight", "eps": "eps"},
                {"out": "moe_branch"},
            ),
        )
    )
    return tuple(events)


def _grouped_events(
    *, task_major: bool, hierarchical_router: bool
) -> tuple[_Invocation | _Bridge, ...]:
    events: list[_Invocation | _Bridge] = [
        _Bridge("max_active_tiles = hl.specialize(max_active_tiles)"),
        _Invocation(
            "expert_pre_norm",
            separate.rms_norm,
            {
                "x": "residual",
                "weight": "pre_ff_norm_weight",
                "eps": "eps",
            },
            {"out": "expert_input"},
        ),
    ]
    if hierarchical_router:
        events.extend(
            (
                _Invocation(
                    "router_project",
                    router_project_by_expert,
                    {
                        "hidden": "residual",
                        "scale": "router_scale",
                        "root_size": "root_size",
                        "weight": "router_weight",
                        "eps": "eps",
                    },
                    {"output": "router_logits"},
                ),
                _Invocation(
                    "route_candidates",
                    route_topk_candidates,
                    {"logits": "router_logits", "top_k": "top_k"},
                    {
                        "values_out": "candidate_values",
                        "ids_out": "candidate_ids",
                    },
                ),
                _Invocation(
                    "route_merge",
                    route_topk_merge,
                    {
                        "candidate_values": "candidate_values",
                        "candidate_ids": "candidate_ids",
                        "per_expert_scale": "per_expert_scale",
                        "top_k": "top_k",
                    },
                    {"weights_out": "topk_weights", "ids_out": "topk_ids"},
                ),
            )
        )
    else:
        events.append(
            _Invocation(
                "router_projection_topk",
                separate.router_projection_topk,
                {
                    "hidden": "residual",
                    "scale": "router_scale",
                    "root_size": "root_size",
                    "router_weight": "router_weight",
                    "per_expert_scale": "per_expert_scale",
                    "top_k": "top_k",
                    "eps": "eps",
                },
                {"weights": "topk_weights", "ids": "topk_ids"},
            )
        )
    events.extend(
        (
            _Invocation(
                "expert_tiles",
                separate.moe_expert_tiles,
                {
                    "topk_ids": "topk_ids",
                    "tile_tokens": "tile_tokens",
                    "tiles_per_expert": "tiles_per_expert",
                    "num_experts": "num_experts",
                    "max_active_tiles": "max_active_tiles",
                },
                {
                    "counts_out": "expert_counts",
                    "tile_end_out": "tile_end",
                    "active_out": "active_tiles",
                },
            ),
            _Invocation(
                "assignment_order",
                separate.moe_assignment_order,
                {
                    "topk_ids": "topk_ids",
                    "tile_tokens": "tile_tokens",
                    "tiles_per_expert": "tiles_per_expert",
                    "num_experts": "num_experts",
                },
                {"order": "order"},
            ),
        )
    )
    if task_major:
        events.extend(
            (
                _Invocation(
                    "grouped_expert_geglu",
                    grouped_expert_geglu_task_major,
                    {
                        "expert_input": "expert_input",
                        "expert_weight": "expert_gate_up_weight",
                        "topk_weights": "topk_weights",
                        "active_tiles": "active_tiles",
                        "tile_end": "tile_end",
                        "expert_counts": "expert_counts",
                        "order": "order",
                        "tile_tokens": "tile_tokens",
                        "tiles_per_expert": "tiles_per_expert",
                        "top_k": "top_k",
                    },
                    {
                        "output": "activation",
                        "active_out": "scheduled_active_tiles",
                        "expert_out": "scheduled_experts",
                        "assignment_out": "scheduled_assignments",
                        "route_weight_out": "scheduled_route_weights",
                    },
                ),
                _Invocation(
                    "grouped_expert_down",
                    grouped_expert_down_task_major,
                    {
                        "activation": "activation",
                        "expert_weight": "expert_down_weight",
                        "topk_weights": "topk_weights",
                        "active_tiles": "scheduled_active_tiles",
                        "experts": "scheduled_experts",
                        "assignments": "scheduled_assignments",
                        "route_weights": "scheduled_route_weights",
                    },
                    {"output": "expert_outputs_flat"},
                ),
                _Bridge(
                    "expert_outputs = expert_outputs_flat.view("
                    "residual.size(0), top_k, expert_down_weight.size(1))"
                ),
                _Invocation(
                    "expert_reduce",
                    expert_sum,
                    {"expert_output": "expert_outputs"},
                    {"output": "moe_down"},
                ),
            )
        )
    else:
        events.extend(
            (
                _Invocation(
                    "grouped_expert_geglu",
                    separate.grouped_expert_geglu_projection,
                    {
                        "expert_input": "expert_input",
                        "expert_weight": "expert_gate_up_weight",
                        "active_tiles": "active_tiles",
                        "tile_end": "tile_end",
                        "expert_counts": "expert_counts",
                        "order": "order",
                        "tile_tokens": "tile_tokens",
                        "tiles_per_expert": "tiles_per_expert",
                        "top_k": "top_k",
                    },
                    {"output": "activation"},
                ),
                _Invocation(
                    "grouped_expert_down",
                    separate.grouped_expert_down,
                    {
                        "activation": "activation",
                        "expert_weight": "expert_down_weight",
                        "active_tiles": "active_tiles",
                        "tile_end": "tile_end",
                        "expert_counts": "expert_counts",
                        "order": "order",
                        "tile_tokens": "tile_tokens",
                        "tiles_per_expert": "tiles_per_expert",
                    },
                    {"output": "expert_outputs"},
                ),
                _Invocation(
                    "expert_reduce",
                    separate.batched_expert_reduce,
                    {
                        "expert_outputs": "expert_outputs",
                        "topk_weights": "topk_weights",
                    },
                    {"output": "moe_down"},
                ),
            )
        )
    events.append(
        _Invocation(
            "moe_post_norm",
            separate.rms_norm,
            {"x": "moe_down", "weight": "post_ff_norm_weight", "eps": "eps"},
            {"out": "moe_branch"},
        )
    )
    return tuple(events)


def _source_outputs(source_mode: str) -> tuple[str, ...]:
    if source_mode == "matched":
        return (
            "moe_branch",
            "expert_input",
            "router_logits",
            "topk_weights",
            "topk_ids",
            "gate_up",
            "activation",
            "expert_outputs",
            "moe_down",
        )
    if source_mode == "grouped":
        return (
            "moe_branch",
            "topk_weights",
            "topk_ids",
            "activation",
            "expert_outputs",
            "moe_down",
        )
    if source_mode in ("grouped_task_major", "grouped_task_major_fused_router"):
        return ("moe_branch", "topk_weights", "topk_ids", "moe_down")
    return (
        "moe_branch",
        "router_logits",
        "topk_weights",
        "topk_ids",
        "activation",
        "expert_outputs",
        "moe_down",
    )


def _compose_source(source_mode: str) -> str:
    preamble: list[ast.stmt] = []
    loops: list[ast.stmt] = []
    events = (
        _matched_events()
        if source_mode == "matched"
        else _grouped_events(
            task_major=source_mode != "grouped",
            hierarchical_router=source_mode == "grouped_task_major",
        )
        if source_mode
        in ("grouped", "grouped_task_major", "grouped_task_major_fused_router")
        else _assignment_events(
            topk_mode=(
                "iterative"
                if source_mode == "assignment"
                else "hierarchical"
                if source_mode.startswith("assignment_hierarchical_topk")
                else "torch"
            ),
            fuse_geglu=source_mode != "assignment_hierarchical_topk_unfused_geglu",
        )
    )
    for invocation in events:
        if isinstance(invocation, _Bridge):
            preamble.extend(ast.parse(invocation.source).body)
            continue
        invocation_preamble, invocation_loops = _inline_invocation(invocation)
        preamble.extend(invocation_preamble)
        loops.extend(invocation_loops)

    arguments = [
        "residual",
        "pre_ff_norm_weight",
        "router_scale",
        "root_size",
        "router_weight",
        "per_expert_scale",
        "expert_gate_up_weight",
        "expert_down_weight",
        "post_ff_norm_weight",
        "top_k",
        "eps",
        "tile_tokens",
        "tiles_per_expert",
        "num_experts",
        "max_active_tiles",
    ]
    outputs = _source_outputs(source_mode)
    function_name = f"gemma4_a4b_moe_{source_mode}_megakernel_source"
    function = ast.FunctionDef(
        name=function_name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in arguments],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            *preamble,
            *loops,
            ast.Return(
                value=ast.Tuple(
                    elts=[ast.Name(id=name, ctx=ast.Load()) for name in outputs],
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    return ast.unparse(module) + "\n"


def _build_megakernel(source_mode: str):
    source = _compose_source(source_mode)
    filename = str(
        Path(__file__).with_name(f"_generated_gemma4_a4b_moe_{source_mode}.py")
    )
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = globals()
    exec(compile(source, filename, "exec"), namespace)
    function = namespace[f"gemma4_a4b_moe_{source_mode}_megakernel_source"]
    return (
        helion.kernel(
            static_shapes=True,
            autotune_effort="none",
            backend="triton",
        )(function),
        source,
    )


MEGAKERNELS = {
    source_mode: _build_megakernel(source_mode)
    for source_mode in (
        "matched",
        "assignment",
        "assignment_torch_topk",
        "assignment_hierarchical_topk",
        "assignment_hierarchical_topk_unfused_geglu",
        "grouped",
        "grouped_task_major",
        "grouped_task_major_fused_router",
    )
}


def _megakernel_args(tensors, shape):
    tile_tokens = 16
    return (
        tensors["residual"],
        tensors["pre_ff_norm_weight_2"],
        tensors["router_scale"],
        tensors["root_size"],
        tensors["router_weight"],
        tensors["per_expert_scale"],
        tensors["expert_gate_up_weight"],
        tensors["expert_down_weight"],
        tensors["post_ff_norm_weight_2"],
        shape.top_k,
        shape.eps,
        tile_tokens,
        tiles_per_expert_of(shape, tile_tokens),
        shape.num_experts,
        max_aligned_tiles(shape, tile_tokens),
    )


def _config(bound, args):
    values = dict(bound.config_spec.default_config())
    if args.config_mode == "matched" and args.source_mode == "matched":
        configs = json.loads(Path(args.config_path).read_text())
        names = (
            "moe_b1_expert_pre_norm",
            "moe_b1_router_norm_scale",
            "moe_b1_router_mm_fp32",
            "moe_b1_route_topk",
            "moe_b1_expert_gate_up",
            "moe_b1_expert_geglu",
            "moe_b1_expert_down",
            "moe_b1_expert_reduce",
            "moe_b1_moe_post_norm",
        )
        root_configs = tuple(configs[name] for name in names)
        block_choices = {
            4: root_configs[2]["block_sizes"][0],
            5: root_configs[2]["block_sizes"][1],
            9: root_configs[4]["block_sizes"][0],
            10: root_configs[4]["block_sizes"][1],
            12: root_configs[5]["block_sizes"][0],
            13: root_configs[6]["block_sizes"][0],
            14: root_configs[6]["block_sizes"][1],
            15: root_configs[7]["block_sizes"][0],
        }
        values["block_sizes"] = [
            block_choices[spec.block_id] for spec in bound.config_spec.block_sizes
        ]
        range_choices = {
            (0,): (root_configs[0], 0),
            (2,): (root_configs[1], 0),
            (3, 4): (root_configs[2], 0),
            (5,): (root_configs[2], 1),
            (6,): (root_configs[3], 0),
            (9,): (root_configs[4], 0),
            (10,): (root_configs[4], 1),
            (11, 12): (root_configs[5], 0),
            (13,): (root_configs[6], 0),
            (14,): (root_configs[6], 1),
            (15,): (root_configs[7], 0),
            (16,): (root_configs[8], 0),
        }
        range_spec_names = {
            "range_unroll_factors": "range_unroll_factors",
            "range_warp_specializes": "range_warp_specialize",
            "range_num_stages": "range_num_stages",
            "range_multi_buffers": "range_multi_buffers",
            "range_flattens": "range_flattens",
        }
        for key, spec_name in range_spec_names.items():
            updated = []
            for spec, default in zip(
                getattr(bound.config_spec, spec_name), values[key], strict=True
            ):
                root_config, index = range_choices[tuple(spec.block_ids)]
                choices = root_config.get(key, ())
                updated.append(choices[index] if index < len(choices) else default)
            values[key] = updated
        values["loop_orders"] = [
            root_configs[2]["loop_orders"][0],
            root_configs[5]["loop_orders"][0],
        ]
        values["l2_groupings"] = [
            root_configs[2]["l2_groupings"][0],
            root_configs[5]["l2_groupings"][0],
        ]
        fact_counts = (3, 4, 3, 4, 4, 3, 5, 3, 3)
        load_counts = (2, 3, 2, 2, 3, 2, 4, 2, 2)
        values["indexing"] = [
            indexing
            for root_config, count in zip(root_configs, fact_counts, strict=True)
            for indexing in root_config["indexing"][:count]
        ]
        values["load_eviction_policies"] = [
            policy
            for root_config, count in zip(root_configs, load_counts, strict=True)
            for policy in root_config["load_eviction_policies"][:count]
        ]
    elif args.config_mode == "matched" and args.source_mode in (
        "grouped",
        "grouped_task_major",
        "grouped_task_major_fused_router",
    ):
        configs = json.loads(Path(args.config_path).read_text())
        prefix = f"moe_b{args.batch}"
        names = (
            "expert_pre_norm",
            "router_projection_topk",
            "expert_tiles_t16",
            "assignment_order_t16",
            "grouped_expert_geglu_projection_t16",
            "grouped_expert_down_t16",
            "batched_expert_reduce",
            "moe_post_norm",
        )
        root_configs = tuple(configs[f"{prefix}_{name}"] for name in names)
        if args.source_mode == "grouped_task_major":
            values["block_sizes"] = [
                args.router_block,
                *(
                    block_size
                    for root_config in root_configs[2:]
                    for block_size in root_config.get("block_sizes", ())
                ),
            ]
            values["loop_orders"][-3:] = [
                root_configs[4]["loop_orders"][0],
                root_configs[5]["loop_orders"][0],
                root_configs[6]["loop_orders"][0],
            ]
            values["l2_groupings"][-3:] = [
                root_configs[4]["l2_groupings"][0],
                root_configs[5]["l2_groupings"][0],
                root_configs[6]["l2_groupings"][0],
            ]
            range_choices = {
                (0,): (root_configs[0], 0),
                (2, 3): ({}, 0),
                (4, 5): ({}, 0),
                (7,): ({}, 0),
                (9,): (root_configs[2], 0),
                (10,): (root_configs[2], 1),
                (11,): (root_configs[2], 2),
                (12,): (root_configs[3], 0),
                (13,): (root_configs[3], 1),
                (14, 15): (root_configs[4], 0),
                (16,): (root_configs[4], 1),
                (17, 18): (root_configs[5], 0),
                (19,): (root_configs[5], 1),
                (20, 21): (root_configs[6], 0),
                (22,): (root_configs[7], 0),
            }
        else:
            values["block_sizes"] = [
                block_size
                for root_config in root_configs
                for block_size in root_config.get("block_sizes", ())
            ]
            values["loop_orders"] = [
                root_configs[4]["loop_orders"][0],
                root_configs[5]["loop_orders"][0],
                root_configs[6]["loop_orders"][0],
            ]
            values["l2_groupings"] = [
                root_configs[4]["l2_groupings"][0],
                root_configs[5]["l2_groupings"][0],
                root_configs[6]["l2_groupings"][0],
            ]
            range_choices = {
                (0,): (root_configs[0], 0),
                (2,): (root_configs[1], 0),
                (3,): (root_configs[1], 1),
                (4,): (root_configs[1], 2),
                (7,): (root_configs[2], 0),
                (8,): (root_configs[2], 1),
                (9,): (root_configs[2], 2),
                (10,): (root_configs[3], 0),
                (11,): (root_configs[3], 1),
                (12, 13): (root_configs[4], 0),
                (14,): (root_configs[4], 1),
                (15, 16): (root_configs[5], 0),
                (17,): (root_configs[5], 1),
                (18, 19): (root_configs[6], 0),
                (20,): (root_configs[7], 0),
            }
        grouped_block_start = 4 if args.source_mode == "grouped_task_major" else 5
        values["block_sizes"][grouped_block_start:] = [
            args.group_gate_block,
            args.group_gate_block_k,
            args.group_down_block,
            args.group_down_block_k,
            args.group_reduce_block,
        ]
        range_spec_names = {
            "range_unroll_factors": "range_unroll_factors",
            "range_warp_specializes": "range_warp_specialize",
            "range_num_stages": "range_num_stages",
            "range_multi_buffers": "range_multi_buffers",
            "range_flattens": "range_flattens",
        }
        for key, spec_name in range_spec_names.items():
            updated = []
            for spec, default in zip(
                getattr(bound.config_spec, spec_name), values[key], strict=True
            ):
                root_config, index = range_choices[tuple(spec.block_ids)]
                choices = root_config.get(key, ())
                updated.append(choices[index] if index < len(choices) else default)
            values[key] = updated
        gate_range = (16,) if args.source_mode == "grouped_task_major" else (14,)
        down_range = (19,) if args.source_mode == "grouped_task_major" else (17,)
        values["range_num_stages"] = [
            args.gate_stages
            if tuple(spec.block_ids) == gate_range
            else args.down_stages
            if tuple(spec.block_ids) == down_range
            else default
            for spec, default in zip(
                bound.config_spec.range_num_stages,
                values["range_num_stages"],
                strict=True,
            )
        ]
        if args.source_mode == "grouped_task_major" and args.group_use_tma:
            for index in (28, 30):
                values["indexing"][index] = "tensor_descriptor"
        if values.get("static_ranges"):
            values["static_ranges"] = root_configs[2]["static_ranges"]
    elif args.config_mode == "matched":
        if args.source_mode == "assignment_hierarchical_topk_unfused_geglu":
            gate_range = (11,)
            down_range = (17,)
            reduce_range = (18, 19)
            postnorm_range = (20,)
            block_sizes = [
                args.router_block,
                args.gate_block,
                args.gate_block_k,
                args.geglu_block,
                args.down_block,
                args.down_block_k,
                args.reduce_block,
            ]
        elif args.source_mode == "assignment_hierarchical_topk":
            gate_range = (11,)
            down_range = (15,)
            reduce_range = (16, 17)
            postnorm_range = (18,)
            block_sizes = [
                args.router_block,
                args.gate_block,
                args.gate_block_k,
                args.down_block,
                args.down_block_k,
                args.reduce_block,
            ]
        else:
            gate_range = (9,)
            down_range = (13,)
            reduce_range = (14, 15)
            postnorm_range = (16,)
            block_sizes = [
                args.router_block,
                args.gate_block,
                args.gate_block_k,
                args.down_block,
                args.down_block_k,
                args.reduce_block,
            ]
        values["block_sizes"] = block_sizes
        range_values = {
            "range_unroll_factors": {
                postnorm_range: 4,
            },
            "range_num_stages": {
                gate_range: args.gate_stages,
                down_range: args.down_stages,
                reduce_range: 1,
            },
            "range_multi_buffers": {
                gate_range: True,
                down_range: False,
            },
            "range_flattens": {
                gate_range: False,
                down_range: False,
            },
        }
        range_spec_names = {
            "range_unroll_factors": "range_unroll_factors",
            "range_num_stages": "range_num_stages",
            "range_multi_buffers": "range_multi_buffers",
            "range_flattens": "range_flattens",
        }
        for key, choices in range_values.items():
            values[key] = [
                choices.get(tuple(spec.block_ids), default)
                for spec, default in zip(
                    getattr(bound.config_spec, range_spec_names[key]),
                    values[key],
                    strict=True,
                )
            ]
        values["range_warp_specializes"] = [
            None for _ in values["range_warp_specializes"]
        ]
        values["l2_groupings"][-2:] = [
            args.gate_l2_grouping,
            args.down_l2_grouping,
        ]
    values.update(
        {
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
            "num_warps": args.num_warps,
            "num_stages": args.kernel_stages,
            "cross_loop_num_workers": args.workers,
        }
    )
    if args.maxnreg is not None:
        values["maxnreg"] = args.maxnreg
    if args.disable_warp_specialize:
        values["range_warp_specializes"] = [
            None for _ in values["range_warp_specializes"]
        ]
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


def _assert_close(name, actual, expected, *, atol=3e-1, rtol=1e-1):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    maximum = float((actual.float() - expected.float()).abs().max().item())
    print(f"megakernel_correctness {name} max_abs={maximum:.6f}", flush=True)


def _prepare_hand_triton(
    args: argparse.Namespace,
    tensors: dict[str, torch.Tensor],
    shape: Gemma4A4BMoEShape,
    reference: dict[str, torch.Tensor],
    *,
    fuse_tasks: bool,
    group_continuation: bool,
):
    hand_triton.LINE = args.hand_counter_stride
    hand_triton.D_LINE = hand_triton.tl.constexpr(args.hand_counter_stride)
    hand_args = argparse.Namespace(
        seed=args.seed,
        batch=args.batch,
        route_skew=args.route_skew,
        workers=args.hand_workers,
        num_warps=args.num_warps,
        kernel_stages=args.kernel_stages,
        tile_tokens=16,
        norm_block=256,
        router_block_k=args.router_block_k,
        router_expert_block=args.router_block,
        router_split=True,
        fuse_pre_norm=True,
        fuse_tasks=fuse_tasks,
        order_block=64,
        gate_block_n=args.gate_block,
        gate_block_k=args.gate_block_k,
        gate_stages=args.gate_stages,
        down_block_n=args.down_block,
        down_block_k=args.down_block_k,
        down_stages=args.down_stages,
        reduce_block=args.reduce_block,
        weight_major="kn",
        eviction="",
        schedule="static",
        formulation="gathered",
        expert_task="tile",
        group_continuation=group_continuation,
        poll_delay=0,
        no_waits=False,
        root_mask=-1,
    )
    prepared = hand_triton.prepare_variant(
        hand_args,
        shape,
        tensors,
        reference,
        "static",
        "tile",
        group_continuation,
    )
    assert prepared is not None
    _, call = prepared
    return call


def run(args) -> None:
    if not args.allow_busy:
        separate.require_idle_visible_gpu()
    if args.source_mode == "matched" and args.batch != 1:
        raise ValueError(
            "the matched source preserves batch-one standalone kernels; use "
            "assignment_hierarchical_topk_unfused_geglu for batched validation"
        )
    shape = Gemma4A4BMoEShape(batch=args.batch)
    tensors = allocate_moe(shape, args.seed, route_skew=args.route_skew)
    reference = moe_reference(tensors, shape)
    kernel_args = _megakernel_args(tensors, shape)
    kernel, source = MEGAKERNELS[args.source_mode]
    bound = kernel.bind(kernel_args)
    host_function = bound.host_function
    assert host_function is not None
    config = _config(bound, args)

    if args.print_source:
        print(source)
        return
    if args.dump_config:
        print("MEGAKERNEL_CONFIG", dict(config), flush=True)
        print("ROOT_BLOCK_IDS", host_function.device_ir.grid_block_ids, flush=True)
        graph = host_function.device_ir.tile_dependency_graph
        assert graph is not None
        print("DEPENDENCY_EDGES", graph.edges, flush=True)
        print("EXECUTION_SCOPES", graph.execution_scopes, flush=True)
        print(
            "BLOCK_SPECS",
            [
                {
                    "block_id": spec.block_id,
                    "size_hint": spec.size_hint,
                    "min_size": spec.min_size,
                    "max_size": spec.max_size,
                }
                for spec in bound.config_spec.block_sizes
            ],
            flush=True,
        )
        print(
            "RANGE_BLOCK_IDS",
            [spec.block_ids for spec in bound.config_spec.range_num_stages],
            flush=True,
        )
    if args.print_lowered:
        print(bound.to_triton_code(config, output_origin_lines=True))
        return
    if args.inspect_only:
        return

    compiled = bound.compile_config(config)
    outputs = compiled(*kernel_args)
    torch.cuda.synchronize()
    expected_expert_outputs = reference["expert_outputs"]
    if args.source_mode.startswith("assignment"):
        expected_expert_outputs = (
            expected_expert_outputs.float() * reference["topk_weights"][:, :, None]
        ).to(expected_expert_outputs.dtype)
    expected_by_name = {
        "moe_branch": reference["moe_branch"],
        "expert_input": reference["expert_input"],
        "router_logits": reference["router_logits"],
        "topk_weights": reference["topk_weights"],
        "topk_ids": reference["topk_ids"],
        "gate_up": reference["expert_gate_up"],
        "activation": reference["expert_activation"],
        "expert_outputs": expected_expert_outputs,
        "moe_down": reference["moe_down"],
    }
    for name, actual in zip(_source_outputs(args.source_mode), outputs, strict=True):
        wanted = expected_by_name[name].reshape(actual.shape)
        if name == "topk_ids":
            torch.testing.assert_close(actual, wanted)
        else:
            _assert_close(name, actual, wanted)
    print("MEGAKERNEL_RESOURCES", _helion_resources(compiled), flush=True)

    if args.smoke and not args.benchmark:
        return

    baseline_args = argparse.Namespace(tune=[], tile_tokens=16)
    config_path = Path(args.config_path)
    configs = json.loads(config_path.read_text())
    baseline_tensors = allocate_moe(shape, args.seed, route_skew=args.route_skew)
    built = separate.build_moe(
        baseline_args,
        baseline_tensors,
        shape,
        configs,
        config_path,
    )
    megakernel_graph, graph_outputs = capture(lambda: compiled(*kernel_args))
    megakernel_graph.replay()
    baseline_graphs = {}
    baseline_outputs = {}
    for launch_name in ("launch_matched", "launch_optimized", "launch_grouped"):
        if launch_name not in built:
            continue
        graph, output = capture(built[launch_name])
        graph.replay()
        short_name = launch_name.removeprefix("launch_")
        baseline_graphs[f"helion_a4b_moe_separate_{short_name}"] = graph
        baseline_outputs[short_name] = output
    if args.benchmark_fused_control:
        fused_kernel, _ = MEGAKERNELS["assignment_hierarchical_topk"]
        fused_bound = fused_kernel.bind(kernel_args)
        fused_args = argparse.Namespace(**vars(args))
        fused_args.source_mode = "assignment_hierarchical_topk"
        fused_compiled = fused_bound.compile_config(_config(fused_bound, fused_args))
        fused_outputs = fused_compiled(*kernel_args)
        torch.cuda.synchronize()
        _assert_close(
            "fused_control_moe_branch", fused_outputs[0], reference["moe_branch"]
        )
        print(
            "FUSED_CONTROL_RESOURCES",
            _helion_resources(fused_compiled),
            flush=True,
        )
        fused_graph, _ = capture(lambda: fused_compiled(*kernel_args))
        fused_graph.replay()
        baseline_graphs["helion_a4b_moe_fused_geglu"] = fused_graph
    if args.benchmark_hand:
        hand_modes = {
            "fused": ("triton_a4b_moe_fused_tasks", True, False),
            "keyed": ("triton_a4b_moe_keyed_roots", False, True),
            "barrier": ("triton_a4b_moe_root_barrier", False, False),
        }
        hand_variants = [hand_modes[args.hand_mode]]
        for name, fuse_tasks, group_continuation in hand_variants:
            hand_call = _prepare_hand_triton(
                args,
                tensors,
                shape,
                reference,
                fuse_tasks=fuse_tasks,
                group_continuation=group_continuation,
            )
            hand_graph, _ = capture(hand_call)
            hand_graph.replay()
            baseline_graphs[name] = hand_graph
    torch.cuda.synchronize()
    _assert_close("graph_moe_branch", graph_outputs[0], reference["moe_branch"])
    for short_name, output in baseline_outputs.items():
        _assert_close(
            f"baseline_{short_name}_moe_branch",
            output,
            reference["moe_branch"],
        )
    pids = visible_gpu_pids()
    timings = benchmark_interleaved(
        {
            "helion_a4b_moe_megakernel": megakernel_graph.replay,
            **{name: graph.replay for name, graph in baseline_graphs.items()},
        },
        args.repeats,
        args.batch_replays,
    )
    if visible_gpu_pids() != pids:
        raise RuntimeError("GPU process set changed during benchmark")
    print(
        "RESULT_JSON",
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "helion_module": helion.__file__,
                "resources": _helion_resources(compiled),
                "timings": timings,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--route-skew", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=444)
    parser.add_argument("--worker-multiplier", type=int, default=4)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--kernel-stages", type=int, default=1)
    parser.add_argument("--maxnreg", type=int, choices=(32, 64, 128, 256))
    parser.add_argument("--disable-warp-specialize", action="store_true")
    parser.add_argument(
        "--source-mode",
        choices=(
            "matched",
            "assignment",
            "assignment_torch_topk",
            "assignment_hierarchical_topk",
            "assignment_hierarchical_topk_unfused_geglu",
            "grouped",
            "grouped_task_major",
            "grouped_task_major_fused_router",
        ),
        default="matched",
    )
    parser.add_argument("--router-block", type=int, default=8)
    parser.add_argument("--router-block-k", type=int, default=256)
    parser.add_argument("--router-stages", type=int, default=3)
    parser.add_argument("--gate-block", type=int, default=16)
    parser.add_argument("--gate-block-k", type=int, default=256)
    parser.add_argument("--gate-stages", type=int, default=3)
    parser.add_argument("--gate-l2-grouping", type=int, default=1)
    parser.add_argument("--geglu-block", type=int, default=128)
    parser.add_argument("--down-block", type=int, default=64)
    parser.add_argument("--down-block-k", type=int, default=64)
    parser.add_argument("--down-stages", type=int, default=5)
    parser.add_argument("--down-l2-grouping", type=int, default=1)
    parser.add_argument("--reduce-block", type=int, default=32)
    parser.add_argument("--group-gate-block", type=int, default=64)
    parser.add_argument("--group-gate-block-k", type=int, default=128)
    parser.add_argument("--group-down-block", type=int, default=64)
    parser.add_argument("--group-down-block-k", type=int, default=64)
    parser.add_argument("--group-reduce-block", type=int, default=64)
    parser.add_argument("--group-use-tma", action="store_true")
    parser.add_argument(
        "--config-mode", choices=("default", "matched"), default="matched"
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-replays", type=int, default=50)
    parser.add_argument(
        "--config-path",
        default=str(Path(__file__).with_name("gemma4_a4b_moe_b200_configs.json")),
    )
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-fused-control", action="store_true")
    parser.add_argument("--benchmark-hand", action="store_true")
    parser.add_argument(
        "--hand-mode", choices=("fused", "keyed", "barrier"), default="fused"
    )
    parser.add_argument("--hand-workers", type=int, default=444)
    parser.add_argument("--hand-counter-stride", type=int, default=32)
    parser.add_argument("--print-source", action="store_true")
    parser.add_argument("--print-lowered", action="store_true")
    parser.add_argument("--dump-config", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
