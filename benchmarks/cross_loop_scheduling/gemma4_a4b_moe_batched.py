# ruff: noqa: ANN001, ANN201
"""Batched Gemma 4 A4B MoE: standalone Helion versus one megakernel.

This probe lives with the scheduler redesign and imports Helion only from this
worktree.  The standalone path is the eight top-level roots of the checked-in
pretuned megakernel emitted as eight ordinary Helion kernels.  Consequently
the two paths have the same inputs, math, intermediate dtypes, and outputs.
Their resource configurations remain independently selectable, as they must be
for a useful standalone control.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import itertools
import json
import linecache
from pathlib import Path
import sys
import textwrap
import types
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pretuned_kernels._bench import bench_pre_captured_cudagraphs
from pretuned_kernels._bench import capture_cuda_graph
from pretuned_kernels._bench import thermal_warmup
from pretuned_kernels.megakernels.gemma4_a4b_moe import gemma4_a4b_moe as gemma
from pretuned_kernels.megakernels.gemma4_a4b_moe._helion_aot_gemma4_a4b_moe_cuda_sm100 import (
    CONFIG,
)
import torch

import helion
from helion._compiler import cross_loop_scheduler
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import CompiledConfig
    from helion.runtime.kernel import Kernel


_STANDALONE_CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    # These are the independently tuned B200 tile/resource choices used by the
    # corresponding standalone roots.  Shape-dependent dimensions remain in
    # the ConfigSpec and are normalized for every batch below.
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
def router_project(
    residual,
    router_scale,
    root_size,
    router_weight,
    eps,
):
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
    batch, top_k, hidden = expert_outputs.size()
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


def _make_persistent_kernel(
    *,
    dynamic_batch: bool = False,
    triton_do_not_specialize: bool | None = None,
) -> Kernel:
    if triton_do_not_specialize is None:
        triton_do_not_specialize = dynamic_batch
    source = textwrap.dedent(inspect.getsource(gemma.gemma4_a4b_moe.fn))
    source = source.replace(
        '@helion.aot_kernel(static_shapes=True, backend="triton")',
        "@helion.kernel("
        f"static_shapes={not dynamic_batch!r}, autotune_effort=\"none\", "
        "backend=\"triton\", "
        f"triton_do_not_specialize={triton_do_not_specialize!r})",
        1,
    )
    if dynamic_batch:
        invariant_size_dimensions = {
            "residual": (1,),
            "pre_ff_norm_weight": (0,),
            "router_scale": (0,),
            "router_weight": (0, 1),
            "per_expert_scale": (0,),
            "expert_gate_up_weight": (0, 1, 2),
            "expert_down_weight": (0, 1, 2),
            "post_ff_norm_weight": (0,),
        }
        tensor_ranks = {
            "residual": 2,
            "pre_ff_norm_weight": 1,
            "router_scale": 1,
            "root_size": 0,
            "router_weight": 2,
            "per_expert_scale": 1,
            "expert_gate_up_weight": 3,
            "expert_down_weight": 3,
            "post_ff_norm_weight": 1,
        }
        invariant_specializations = "".join(
            f"    hl.specialize({name}.size({dimension}))\n"
            for name, dimensions in invariant_size_dimensions.items()
            for dimension in dimensions
        ) + "".join(
            f"    hl.specialize({name}.stride({dimension}))\n"
            for name, rank in tensor_ranks.items()
            for dimension in range(rank)
        )
        marker = "):\n    router_project_hidden = residual"
        if source.count(marker) != 1:
            raise RuntimeError("unexpected Gemma source signature/body boundary")
        source = source.replace(
            marker,
            "):\n" + invariant_specializations + "    router_project_hidden = residual",
            1,
        )
    module_name = "_helion_gemma4_batched_scheduler_probe"
    filename = f"<{module_name}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    module = types.ModuleType(module_name)
    module.__dict__.update(vars(gemma))
    module.__dict__["__name__"] = module_name
    module.__file__ = filename
    sys.modules[module_name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module.gemma4_a4b_moe


def _make_inputs(batch: int, seed: int) -> dict[str, torch.Tensor]:
    old_batch = gemma.BATCH
    gemma.BATCH = batch
    try:
        return gemma._make_inputs(seed)
    finally:
        gemma.BATCH = old_batch


def _make_batch_series_inputs(
    batches: tuple[int, ...], seed: int
) -> dict[int, dict[str, torch.Tensor]]:
    """Keep weights and each smaller batch's token prefix fixed across a sweep."""
    base = _make_inputs(1, seed)
    max_batch = max(batches)
    residual = base["residual"]
    if max_batch > 1:
        torch.manual_seed(seed + 1009)
        extra = (
            torch.randn(
                (max_batch - 1, gemma.HIDDEN),
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 1.4
        )
        residual = torch.cat((residual, extra), dim=0)
    return {
        batch: {
            **base,
            "residual": residual[:batch].contiguous(),
        }
        for batch in batches
    }


def _compile_persistent(
    kernel: Kernel,
    args: tuple[object, ...],
    *,
    global_list: bool,
    config_overrides: dict[str, object] | None = None,
) -> tuple[CompiledConfig, tuple[dict[str, object], ...]]:
    bound = kernel.bind(args)
    values = deepcopy(CONFIG)
    if config_overrides is not None:
        values.update(config_overrides)
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    original = cross_loop_scheduler._global_unit_list_schedule
    records: list[dict[str, object]] = []

    def traced(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original(*call_args, **call_kwargs)
        source_schedule = call_args[1]
        records.append(
            {
                "accepted": result is not None,
                "input_segments": len(source_schedule.segments),
                "output_segments": None if result is None else len(result.segments),
            }
        )
        return result

    try:
        cross_loop_scheduler._global_unit_list_schedule = (
            traced if global_list else lambda *args, **kwargs: None
        )
        bound._compile_cache.clear()
        compiled = bound.compile_config(config)
    finally:
        cross_loop_scheduler._global_unit_list_schedule = original
    return compiled, tuple(records)


def _standalone_pipeline(
    tensors: dict[str, torch.Tensor],
    *,
    config_mode: str,
) -> tuple[
    Callable[[], tuple[torch.Tensor, ...]],
    dict[str, object],
    dict[str, Callable[[], object]],
]:
    compiled: dict[str, CompiledConfig] = {}
    configs: dict[str, object] = {}

    def build(name: str, kernel: Kernel, args: tuple[object, ...]) -> object:
        bound = kernel.bind(args)
        values = dict(bound.config_spec.default_config())
        if config_mode == "matched":
            values.update(_STANDALONE_CONFIG_OVERRIDES[name])
        config = helion.Config.from_dict(values)
        bound.config_spec.normalize(config.config)
        compiled[name] = bound.compile_config(config)
        configs[name] = dict(config)
        return compiled[name](*args)

    router_args = (
        tensors["residual"],
        tensors["router_scale"],
        tensors["root_size"],
        tensors["router_weight"],
        gemma.EPS,
    )
    logits = build("router_project", router_project, router_args)
    candidate_args = (logits, gemma.TOP_K)
    candidates = build("route_candidates", route_candidates, candidate_args)
    merge_args = (
        candidates[0],
        candidates[1],
        tensors["per_expert_scale"],
        gemma.TOP_K,
    )
    topk_weights, topk_ids = build("route_merge", route_merge, merge_args)
    gate_args = (
        tensors["residual"],
        tensors["pre_ff_norm_weight"],
        tensors["expert_gate_up_weight"],
        topk_ids,
        topk_weights,
        gemma.EPS,
    )
    gate_up, selected_ids, selected_weights = build(
        "expert_gate_up", expert_gate_up, gate_args
    )
    geglu_args = (gate_up,)
    activation_flat = build("expert_geglu", expert_geglu, geglu_args)
    down_args = (
        activation_flat,
        tensors["expert_down_weight"],
        selected_ids,
        selected_weights,
    )
    expert_outputs = build("expert_down", expert_down, down_args)
    reduce_args = (expert_outputs,)
    moe_down = build("expert_reduce", expert_reduce, reduce_args)
    norm_args = (moe_down, tensors["post_ff_norm_weight"], gemma.EPS)
    build("post_norm", post_norm, norm_args)
    torch.cuda.synchronize()

    def launch() -> tuple[torch.Tensor, ...]:
        local_logits = compiled["router_project"](*router_args)
        local_candidates = compiled["route_candidates"](local_logits, gemma.TOP_K)
        local_weights, local_ids = compiled["route_merge"](
            local_candidates[0],
            local_candidates[1],
            tensors["per_expert_scale"],
            gemma.TOP_K,
        )
        local_gate, local_selected_ids, local_selected_weights = compiled[
            "expert_gate_up"
        ](
            tensors["residual"],
            tensors["pre_ff_norm_weight"],
            tensors["expert_gate_up_weight"],
            local_ids,
            local_weights,
            gemma.EPS,
        )
        local_activation = compiled["expert_geglu"](local_gate)
        local_expert_outputs = compiled["expert_down"](
            local_activation,
            tensors["expert_down_weight"],
            local_selected_ids,
            local_selected_weights,
        )
        local_down = compiled["expert_reduce"](local_expert_outputs)
        local_branch = compiled["post_norm"](
            local_down, tensors["post_ff_norm_weight"], gemma.EPS
        )
        batch = tensors["residual"].size(0)
        return (
            local_branch,
            local_logits,
            local_weights,
            local_ids,
            local_activation.view(batch, gemma.TOP_K, gemma.INTERMEDIATE),
            local_expert_outputs,
            local_down,
        )

    stage_calls = {
        "router_project": lambda: compiled["router_project"](*router_args),
        "route_candidates": lambda: compiled["route_candidates"](*candidate_args),
        "route_merge": lambda: compiled["route_merge"](*merge_args),
        "expert_gate_up": lambda: compiled["expert_gate_up"](*gate_args),
        "expert_geglu": lambda: compiled["expert_geglu"](*geglu_args),
        "expert_down": lambda: compiled["expert_down"](*down_args),
        "expert_reduce": lambda: compiled["expert_reduce"](*reduce_args),
        "post_norm": lambda: compiled["post_norm"](*norm_args),
    }
    return launch, configs, stage_calls


def _routing_summary(topk_ids: torch.Tensor) -> dict[str, object]:
    ids = topk_ids.detach().cpu()
    counts = torch.bincount(ids.flatten().to(torch.int64), minlength=gemma.NUM_EXPERTS)
    active = counts[counts != 0]
    pairwise_duplicate_sets = 0
    sorted_ids = torch.sort(ids, dim=1).values
    for left in range(ids.size(0)):
        for right in range(left + 1, ids.size(0)):
            pairwise_duplicate_sets += bool(
                torch.equal(sorted_ids[left], sorted_ids[right])
            )
    return {
        "ids_by_token": ids.tolist(),
        "assignments": ids.numel(),
        "distinct_experts": int(active.numel()),
        "max_tokens_per_expert": int(active.max().item()),
        # These are cache-locality diagnostics.  The source has a fixed eight
        # assignment tasks per token, irrespective of which experts are chosen.
        "pairwise_duplicate_expert_sets": pairwise_duplicate_sets,
    }


@torch.inference_mode()
def benchmark(
    batches: tuple[int, ...],
    *,
    repetitions: int,
    warmup_ms: int,
    standalone_config: str,
    scheduler_ablation: bool,
    benchmark_stages: bool,
    include_configs: bool,
    persistent_multipliers: tuple[int, ...],
    persistent_num_warps: int,
    persistent_maxnreg: int,
    persistent_block_sizes: tuple[int, ...] | None,
    persistent_gate_stages: tuple[int, ...],
    persistent_down_stages: tuple[int, ...],
    persistent_gate_unrolls: tuple[int, ...],
    persistent_down_unrolls: tuple[int, ...],
) -> dict[str, object]:
    gemma._require_sm100()
    results: list[dict[str, object]] = []
    inputs_by_batch = _make_batch_series_inputs(batches, seed=0)
    for batch in batches:
        tensors = inputs_by_batch[batch]
        args = gemma._kernel_args(tensors)
        persistent_kernel = _make_persistent_kernel()
        persistent_variants: list[
            tuple[str, CompiledConfig, tuple[dict[str, object], ...]]
        ] = []
        for (
            multiplier,
            gate_stages,
            down_stages,
            gate_unroll,
            down_unroll,
        ) in itertools.product(
            persistent_multipliers,
            persistent_gate_stages,
            persistent_down_stages,
            persistent_gate_unrolls,
            persistent_down_unrolls,
        ):
            range_num_stages = list(CONFIG["range_num_stages"])
            range_num_stages[4] = gate_stages
            range_num_stages[7] = down_stages
            range_unroll_factors = list(CONFIG["range_unroll_factors"])
            range_unroll_factors[4] = gate_unroll
            range_unroll_factors[7] = down_unroll
            overrides: dict[str, object] = {
                "num_sm_multiplier": multiplier,
                "num_warps": persistent_num_warps,
                "maxnreg": persistent_maxnreg,
                "range_num_stages": range_num_stages,
                "range_unroll_factors": range_unroll_factors,
            }
            if persistent_block_sizes is not None:
                overrides["block_sizes"] = list(persistent_block_sizes)
            persistent, plan = _compile_persistent(
                persistent_kernel,
                args,
                global_list=True,
                config_overrides=overrides,
            )
            name = "_".join(
                (
                    "persistent",
                    f"m{multiplier}",
                    f"w{persistent_num_warps}",
                    f"r{persistent_maxnreg}",
                    f"g{gate_stages}",
                    f"d{down_stages}",
                    f"gu{gate_unroll}",
                    f"du{down_unroll}",
                )
            )
            persistent_variants.append(
                (
                    name,
                    persistent,
                    plan,
                )
            )
            if scheduler_ablation:
                local_control, local_plan = _compile_persistent(
                    persistent_kernel,
                    args,
                    global_list=False,
                    config_overrides=overrides,
                )
                persistent_variants.append(
                    (
                        f"{name}_local_control",
                        local_control,
                        local_plan,
                    )
                )
        standalone, standalone_configs, stage_calls = _standalone_pipeline(
            tensors, config_mode=standalone_config
        )

        standalone_outputs = standalone()
        tolerances = (
            (0.15, 0.06),
            (0.05, 0.02),
            (2e-5, 2e-5),
            (0, 0),
            (0.2, 0.08),
            (0.25, 0.1),
            (0.25, 0.1),
        )
        for _name, persistent, _plan in persistent_variants:
            persistent_outputs = persistent(*args)
            torch.cuda.synchronize()
            for actual, expected, (atol, rtol) in zip(
                persistent_outputs, standalone_outputs, tolerances, strict=True
            ):
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

        standalone_graph, _ = capture_cuda_graph(standalone)
        persistent_graphs = [
            capture_cuda_graph(
                lambda persistent=persistent, args=args: persistent(*args)
            )[0]
            for _name, persistent, _plan in persistent_variants
        ]
        thermal_warmup(warmup_ms)
        timings = bench_pre_captured_cudagraphs(
            (standalone_graph.replay, *(graph.replay for graph in persistent_graphs)),
            rep=repetitions,
        )
        result: dict[str, object] = {
            "batch": batch,
            "standalone_config": standalone_config,
            "timings_us": {
                "standalone": timings[0] * 1000,
                **{
                    name: latency * 1000
                    for (name, _persistent, _plan), latency in zip(
                        persistent_variants, timings[1:], strict=True
                    )
                },
            },
            "routing": _routing_summary(standalone_outputs[3]),
            "persistent_plans": {
                name: plan for name, _persistent, plan in persistent_variants
            },
            "persistent_config": {
                "num_warps": persistent_num_warps,
                "maxnreg": persistent_maxnreg,
                "block_sizes": list(persistent_block_sizes)
                if persistent_block_sizes is not None
                else CONFIG["block_sizes"],
                "gate_stages": list(persistent_gate_stages),
                "down_stages": list(persistent_down_stages),
                "gate_unrolls": list(persistent_gate_unrolls),
                "down_unrolls": list(persistent_down_unrolls),
            },
        }
        if benchmark_stages:
            stage_names = tuple(stage_calls)
            stage_graphs = tuple(
                capture_cuda_graph(stage_calls[name])[0].replay for name in stage_names
            )
            stage_times = bench_pre_captured_cudagraphs(stage_graphs, rep=repetitions)
            result["standalone_stage_timings_us"] = {
                name: latency * 1000
                for name, latency in zip(stage_names, stage_times, strict=True)
            }
            result["standalone_stage_timings_note"] = (
                "Each root is timed with its own cold-L2 flush; these diagnostic "
                "latencies are not additive."
            )
        if include_configs:
            result["standalone_configs"] = standalone_configs
        results.append(result)
    return {
        "workload": "Gemma 4 26B-A4B MoE, matched eight-root boundary",
        "device": torch.cuda.get_device_name(),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=(2, 4, 8, 16))
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup-ms", type=int, default=10_000)
    parser.add_argument(
        "--standalone-config", choices=("default", "matched"), default="matched"
    )
    parser.add_argument(
        "--scheduler-ablation",
        action="store_true",
        help="also compile an otherwise identical persistent kernel with the global list proposer disabled",
    )
    parser.add_argument("--benchmark-stages", action="store_true")
    parser.add_argument("--include-configs", action="store_true")
    parser.add_argument("--persistent-multipliers", type=int, nargs="+", default=(4,))
    parser.add_argument("--persistent-num-warps", type=int, default=4)
    parser.add_argument("--persistent-maxnreg", type=int, default=128)
    parser.add_argument("--persistent-block-sizes", type=int, nargs=7)
    parser.add_argument("--persistent-gate-stages", type=int, nargs="+", default=(3,))
    parser.add_argument("--persistent-down-stages", type=int, nargs="+", default=(5,))
    parser.add_argument("--persistent-gate-unrolls", type=int, nargs="+", default=(0,))
    parser.add_argument("--persistent-down-unrolls", type=int, nargs="+", default=(0,))
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                tuple(args.batch),
                repetitions=args.repetitions,
                warmup_ms=args.warmup_ms,
                standalone_config=args.standalone_config,
                scheduler_ablation=args.scheduler_ablation,
                benchmark_stages=args.benchmark_stages,
                include_configs=args.include_configs,
                persistent_multipliers=tuple(args.persistent_multipliers),
                persistent_num_warps=args.persistent_num_warps,
                persistent_maxnreg=args.persistent_maxnreg,
                persistent_block_sizes=(
                    tuple(args.persistent_block_sizes)
                    if args.persistent_block_sizes is not None
                    else None
                ),
                persistent_gate_stages=tuple(args.persistent_gate_stages),
                persistent_down_stages=tuple(args.persistent_down_stages),
                persistent_gate_unrolls=tuple(args.persistent_gate_unrolls),
                persistent_down_unrolls=tuple(args.persistent_down_unrolls),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
