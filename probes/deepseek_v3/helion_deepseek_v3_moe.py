# ruff: noqa: ANN001, ANN201, ANN202
"""Standalone Helion kernels for the DeepSeek-V3 MoE block."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from pathlib import Path


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
    assert top_k == 8
    assert topk_groups == 4
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

        # Helion's Triton backend does not currently lower aten.topk.  These
        # fixed-size reductions implement DeepSeek's exact noaux_tc policy:
        # sum the best two experts in each group, retain the best four groups,
        # then select the best eight experts.  Inputs are random, so equality
        # ties (where aten.topk's tie-breaking is unspecified) do not occur.
        group_best_1 = torch.amax(grouped, dim=-1, keepdim=True)
        group_without_best = torch.where(
            grouped == group_best_1, negative_infinity, grouped
        )
        group_best_2 = torch.amax(group_without_best, dim=-1)
        group_scores = group_best_1.view(batch, num_groups) + group_best_2

        group_max_1 = torch.amax(group_scores, dim=-1, keepdim=True)
        remaining_groups = torch.where(
            group_scores == group_max_1,
            torch.full_like(group_scores, float("-inf")),
            group_scores,
        )
        group_max_2 = torch.amax(remaining_groups, dim=-1, keepdim=True)
        remaining_groups = torch.where(
            group_scores == group_max_2,
            torch.full_like(group_scores, float("-inf")),
            remaining_groups,
        )
        group_max_3 = torch.amax(remaining_groups, dim=-1, keepdim=True)
        remaining_groups = torch.where(
            group_scores == group_max_3,
            torch.full_like(group_scores, float("-inf")),
            remaining_groups,
        )
        group_max_4 = torch.amax(remaining_groups, dim=-1, keepdim=True)
        allowed = (group_scores >= group_max_4).view(batch, num_groups, 1)
        masked = torch.where(
            allowed,
            grouped,
            negative_infinity,
        ).view(batch, num_experts)

        expert_indices = hl.arange(num_experts)[None, :].to(torch.int32)
        masked_1 = masked
        value_0 = torch.amax(masked_1, dim=-1, keepdim=True)
        selected_0 = masked_1 == value_0
        id_0 = torch.amin(
            torch.where(
                selected_0,
                expert_indices,
                torch.full_like(expert_indices, num_experts),
            ),
            dim=-1,
        )
        masked_2 = torch.where(
            selected_0, torch.full_like(masked, float("-inf")), masked_1
        )
        value_1 = torch.amax(masked_2, dim=-1, keepdim=True)
        selected_1 = masked_2 == value_1
        id_1 = torch.amin(
            torch.where(
                selected_1, expert_indices, torch.full_like(expert_indices, num_experts)
            ),
            dim=-1,
        )
        masked_3 = torch.where(
            selected_1, torch.full_like(masked, float("-inf")), masked_2
        )
        value_2 = torch.amax(masked_3, dim=-1, keepdim=True)
        selected_2 = masked_3 == value_2
        id_2 = torch.amin(
            torch.where(
                selected_2, expert_indices, torch.full_like(expert_indices, num_experts)
            ),
            dim=-1,
        )
        masked_4 = torch.where(
            selected_2, torch.full_like(masked, float("-inf")), masked_3
        )
        value_3 = torch.amax(masked_4, dim=-1, keepdim=True)
        selected_3 = masked_4 == value_3
        id_3 = torch.amin(
            torch.where(
                selected_3, expert_indices, torch.full_like(expert_indices, num_experts)
            ),
            dim=-1,
        )
        masked_5 = torch.where(
            selected_3, torch.full_like(masked, float("-inf")), masked_4
        )
        value_4 = torch.amax(masked_5, dim=-1, keepdim=True)
        selected_4 = masked_5 == value_4
        id_4 = torch.amin(
            torch.where(
                selected_4, expert_indices, torch.full_like(expert_indices, num_experts)
            ),
            dim=-1,
        )
        masked_6 = torch.where(
            selected_4, torch.full_like(masked, float("-inf")), masked_5
        )
        value_5 = torch.amax(masked_6, dim=-1, keepdim=True)
        selected_5 = masked_6 == value_5
        id_5 = torch.amin(
            torch.where(
                selected_5, expert_indices, torch.full_like(expert_indices, num_experts)
            ),
            dim=-1,
        )
        masked_7 = torch.where(
            selected_5, torch.full_like(masked, float("-inf")), masked_6
        )
        value_6 = torch.amax(masked_7, dim=-1, keepdim=True)
        selected_6 = masked_7 == value_6
        id_6 = torch.amin(
            torch.where(
                selected_6, expert_indices, torch.full_like(expert_indices, num_experts)
            ),
            dim=-1,
        )
        masked_8 = torch.where(
            selected_6, torch.full_like(masked, float("-inf")), masked_7
        )
        value_7 = torch.amax(masked_8, dim=-1, keepdim=True)
        selected_7 = masked_8 == value_7
        id_7 = torch.amin(
            torch.where(
                selected_7, expert_indices, torch.full_like(expert_indices, num_experts)
            ),
            dim=-1,
        )

        weight_0 = torch.sum(
            torch.where(selected_0, scores, torch.zeros_like(scores)), dim=-1
        )
        weight_1 = torch.sum(
            torch.where(selected_1, scores, torch.zeros_like(scores)), dim=-1
        )
        weight_2 = torch.sum(
            torch.where(selected_2, scores, torch.zeros_like(scores)), dim=-1
        )
        weight_3 = torch.sum(
            torch.where(selected_3, scores, torch.zeros_like(scores)), dim=-1
        )
        weight_4 = torch.sum(
            torch.where(selected_4, scores, torch.zeros_like(scores)), dim=-1
        )
        weight_5 = torch.sum(
            torch.where(selected_5, scores, torch.zeros_like(scores)), dim=-1
        )
        weight_6 = torch.sum(
            torch.where(selected_6, scores, torch.zeros_like(scores)), dim=-1
        )
        weight_7 = torch.sum(
            torch.where(selected_7, scores, torch.zeros_like(scores)), dim=-1
        )
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


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def selected_expert_w13(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    batch, hidden_size = hidden.size()
    num_experts, output_size, weight_hidden = weight.size()
    assert batch == 1 and hidden_size == weight_hidden
    top_k = topk_ids.size(1)
    hl.specialize(num_experts)
    hl.specialize(output_size)
    flat_weight = weight.view(num_experts * output_size, hidden_size)
    flat_ids = topk_ids.view(top_k)
    output = torch.empty((top_k, output_size), dtype=hidden.dtype, device=hidden.device)
    flat_output = output.view(top_k * output_size, 1)
    for tile_row in hl.tile(top_k * output_size):
        expert_slot = tile_row.index // output_size
        expert_row = tile_row.index % output_size
        selected_row = flat_ids[expert_slot] * output_size + expert_row
        acc = hl.zeros([tile_row, 1], dtype=torch.float32)
        for tile_k in hl.tile(hidden_size):
            acc = torch.addmm(
                acc,
                flat_weight[selected_row, tile_k],
                hidden[:, tile_k].T,
            )
        flat_output[tile_row, :] = acc.to(flat_output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    rows, twice_intermediate = gate_up.size()
    intermediate = twice_intermediate // 2
    output = torch.empty(
        (rows, intermediate), dtype=gate_up.dtype, device=gate_up.device
    )
    for tile_r, tile_i in hl.tile([rows, intermediate], block_size=[1, None]):
        gate = gate_up[tile_r, tile_i].to(torch.float32)
        up = gate_up[tile_r, tile_i + intermediate].to(torch.float32)
        output[tile_r, tile_i] = (gate * torch.sigmoid(gate) * up).to(output.dtype)
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def selected_expert_w2(
    activation: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    top_k, intermediate = activation.size()
    num_experts, hidden_size, weight_intermediate = weight.size()
    assert intermediate == weight_intermediate
    hl.specialize(num_experts)
    hl.specialize(hidden_size)
    flat_ids = topk_ids.view(top_k)
    output = torch.empty(
        (top_k, hidden_size), dtype=activation.dtype, device=activation.device
    )
    for tile_expert, tile_row in hl.tile([top_k, hidden_size], block_size=[1, None]):
        selected_expert = flat_ids[tile_expert.begin]
        acc = hl.zeros([tile_expert, tile_row], dtype=torch.float32)
        for tile_k in hl.tile(intermediate):
            acc = torch.addmm(
                acc,
                activation[tile_expert, tile_k],
                weight[selected_expert, tile_row, tile_k].T,
            )
        output[tile_expert, tile_row] = acc.to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def weighted_reduce(
    expert_outputs: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    top_k, hidden = expert_outputs.size()
    output = torch.empty(
        (1, hidden), dtype=expert_outputs.dtype, device=expert_outputs.device
    )
    top_k = hl.specialize(top_k)
    for tile_n in hl.tile(hidden):
        values = expert_outputs[:, tile_n].to(torch.float32)
        weights = topk_weights[:, :].view(top_k)
        output[:, tile_n] = torch.sum(
            values * weights[:, None], dim=0, keepdim=True
        ).to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def bf16_mm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    n, weight_k = weight.size()
    assert k == weight_k
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], weight[tile_n, tile_k].T)
        output[tile_m, tile_n] = acc.to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def add_outputs(routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    m, n = routed.size()
    output = torch.empty_like(routed)
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        output[tile_m, tile_n] = routed[tile_m, tile_n] + shared[tile_m, tile_n]
    return output


def _compile_default(kernel, kernel_args):
    bound = kernel.bind(kernel_args)
    config = bound.config_spec.default_config()
    return bound, config, bound.compile_config(config)


def _compile_config(kernel, kernel_args, values):
    bound = kernel.bind(kernel_args)
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return bound, config, bound.compile_config(config)


def _tune(name, kernel, kernel_args, configs, config_path):
    print(f"autotune_start {name}", flush=True)
    started = time.perf_counter()
    bound = kernel.bind(kernel_args)
    config = bound.autotune(kernel_args, force=True)
    configs[name] = dict(config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(configs, indent=2, sort_keys=True) + "\n")
    print(
        "autotune_result",
        json.dumps(
            {
                "name": name,
                "seconds": time.perf_counter() - started,
                "config": dict(config),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return bound, config, bound.compile_config(config)


def _resources(compiled) -> dict[str, int]:
    kernels = []
    for value in compiled.__globals__.values():
        caches = getattr(value, "device_caches", None)
        if not caches or torch.cuda.current_device() not in caches:
            continue
        kernels.extend(caches[torch.cuda.current_device()][0].values())
    if len(kernels) != 1:
        raise RuntimeError(f"expected one compiled Helion kernel, found {len(kernels)}")
    kernel = kernels[0]
    return {
        "registers": kernel.n_regs,
        "spills": kernel.n_spills,
        "shared": kernel.metadata.shared,
    }


def build_moe(args, tensors, shape, configs, config_path: Path):
    tune_set = set(args.tune or [])
    selected_configs = {}
    lowerings = {}
    resources = {}
    compiled_kernels = {}

    def build(name, kernel, kernel_args):
        if "all" in tune_set or name in tune_set:
            bound, config, compiled = _tune(
                name, kernel, kernel_args, configs, config_path
            )
        elif name in configs:
            bound, config, compiled = _compile_config(
                kernel, kernel_args, configs[name]
            )
        else:
            bound, config, compiled = _compile_default(kernel, kernel_args)
        selected_configs[name] = dict(config)
        lowerings[name] = bound.to_triton_code(config, output_origin_lines=True)
        # Triton compilation is lazy; launch once before querying the loaded
        # kernel's register/shared-memory metadata.  This is setup-only work.
        compiled(*kernel_args)
        torch.cuda.synchronize()
        resources[name] = _resources(compiled)
        compiled_kernels[name] = compiled
        return compiled

    hidden = tensors["hidden_states"]
    router_args = (hidden, tensors["router_weight"])
    router = build("router_mm_fp32", router_mm_fp32, router_args)
    logits = router(*router_args)
    route_args = (
        logits,
        tensors["correction_bias"],
        shape.top_k,
        shape.num_groups,
        shape.topk_groups,
        shape.routed_scale,
    )
    route = build("grouped_topk", grouped_topk, route_args)
    topk_weights, topk_ids = route(*route_args)

    expert_w13_args = (hidden, tensors["expert_w13"], topk_ids)
    expert_w13_kernel = build("expert_w13", selected_expert_w13, expert_w13_args)
    expert_gate_up = expert_w13_kernel(*expert_w13_args)
    expert_act_kernel = build("expert_swiglu", silu_and_mul, (expert_gate_up,))
    expert_activation = expert_act_kernel(expert_gate_up)
    expert_w2_args = (expert_activation, tensors["expert_w2"], topk_ids)
    expert_w2_kernel = build("expert_w2", selected_expert_w2, expert_w2_args)
    expert_outputs = expert_w2_kernel(*expert_w2_args)
    reduce_args = (expert_outputs, topk_weights)
    reduce_kernel = build("expert_reduce", weighted_reduce, reduce_args)
    routed_output = reduce_kernel(*reduce_args)

    shared_w13_args = (hidden, tensors["shared_w13"])
    shared_w13_kernel = build("shared_w13", bf16_mm, shared_w13_args)
    shared_gate_up = shared_w13_kernel(*shared_w13_args)
    shared_act_kernel = build("shared_swiglu", silu_and_mul, (shared_gate_up,))
    shared_activation = shared_act_kernel(shared_gate_up)
    shared_w2_args = (shared_activation, tensors["shared_w2"])
    shared_w2_kernel = build("shared_w2", bf16_mm, shared_w2_args)
    shared_output = shared_w2_kernel(*shared_w2_args)
    add_args = (routed_output, shared_output)
    add_kernel = build("final_add", add_outputs, add_args)
    output = add_kernel(*add_args)
    torch.cuda.synchronize()

    shared_stream = torch.cuda.Stream()

    def launch_serial():
        local_logits = router(hidden, tensors["router_weight"])
        local_weights, local_ids = route(
            local_logits,
            tensors["correction_bias"],
            shape.top_k,
            shape.num_groups,
            shape.topk_groups,
            shape.routed_scale,
        )
        local_gate_up = expert_w13_kernel(hidden, tensors["expert_w13"], local_ids)
        local_activation = expert_act_kernel(local_gate_up)
        local_expert_outputs = expert_w2_kernel(
            local_activation, tensors["expert_w2"], local_ids
        )
        local_routed = reduce_kernel(local_expert_outputs, local_weights)
        local_shared_gate_up = shared_w13_kernel(hidden, tensors["shared_w13"])
        local_shared_activation = shared_act_kernel(local_shared_gate_up)
        local_shared = shared_w2_kernel(local_shared_activation, tensors["shared_w2"])
        return add_kernel(local_routed, local_shared)

    def launch_overlap():
        current = torch.cuda.current_stream()
        shared_stream.wait_stream(current)
        local_logits = router(hidden, tensors["router_weight"])
        with torch.cuda.stream(shared_stream):
            local_shared_gate_up = shared_w13_kernel(hidden, tensors["shared_w13"])
            local_shared_activation = shared_act_kernel(local_shared_gate_up)
            local_shared = shared_w2_kernel(
                local_shared_activation, tensors["shared_w2"]
            )
        local_weights, local_ids = route(
            local_logits,
            tensors["correction_bias"],
            shape.top_k,
            shape.num_groups,
            shape.topk_groups,
            shape.routed_scale,
        )
        local_gate_up = expert_w13_kernel(hidden, tensors["expert_w13"], local_ids)
        local_activation = expert_act_kernel(local_gate_up)
        local_expert_outputs = expert_w2_kernel(
            local_activation, tensors["expert_w2"], local_ids
        )
        local_routed = reduce_kernel(local_expert_outputs, local_weights)
        current.wait_stream(shared_stream)
        return add_kernel(local_routed, local_shared)

    return {
        "launch_serial": launch_serial,
        "launch_overlap": launch_overlap,
        "output": output,
        "stage_outputs": {
            "router_logits": logits,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "expert_gate_up": expert_gate_up,
            "expert_activation": expert_activation,
            "expert_outputs": expert_outputs,
            "routed_output": routed_output,
            "shared_gate_up": shared_gate_up,
            "shared_activation": shared_activation,
            "shared_output": shared_output,
            "output": output,
        },
        "configs": selected_configs,
        "lowerings": lowerings,
        "resources": resources,
        "stage_launches": {
            "router_mm_fp32": lambda: router(*router_args),
            "grouped_topk": lambda: route(*route_args),
            "expert_w13": lambda: expert_w13_kernel(*expert_w13_args),
            "expert_swiglu": lambda: expert_act_kernel(expert_gate_up),
            "expert_w2": lambda: expert_w2_kernel(*expert_w2_args),
            "expert_reduce": lambda: reduce_kernel(*reduce_args),
            "shared_w13": lambda: shared_w13_kernel(*shared_w13_args),
            "shared_swiglu": lambda: shared_act_kernel(shared_gate_up),
            "shared_w2": lambda: shared_w2_kernel(*shared_w2_args),
            "final_add": lambda: add_kernel(*add_args),
        },
        "compiled_kernels": compiled_kernels,
    }
