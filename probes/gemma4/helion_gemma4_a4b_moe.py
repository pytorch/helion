# ruff: noqa: ANN001, ANN201, ANN202
"""Standalone Helion benchmark for the Gemma 4 26B-A4B MoE sub-layer.

The extraction covers the nine ops listed in ``gemma4_a4b_moe_common.MOE_STAGES``:
everything between the post-attention residual and the normalized MoE branch.
At ``--batch 1`` the kernels and their boundaries are exactly the ones the full
A4B layer benchmark uses, so this file is a faithful cut-out rather than a
re-implementation.

Two launch paths are benchmarked, matching the full-layer benchmark:

``matched``    nine kernels, one per op.
``optimized``  five kernels; the router pre-norm/projection/top-k collapse into
               ``router_projection_topk``, the expert gate/up GEMV absorbs its
               GeGLU, and the expert down GEMV absorbs the routing-weighted
               reduction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from probes.common import benchmark_cache_mode
from probes.gemma4.gemma4_26b_a4b_common import benchmark_interleaved
from probes.gemma4.gemma4_26b_a4b_common import capture
from probes.gemma4.gemma4_26b_a4b_common import require_idle_visible_gpu
from probes.gemma4.gemma4_26b_a4b_common import visible_gpu_pids
from probes.gemma4.gemma4_a4b_moe_common import MOE_STAGES
from probes.gemma4.gemma4_a4b_moe_common import Gemma4A4BMoEShape
from probes.gemma4.gemma4_a4b_moe_common import _gelu_tanh
from probes.gemma4.gemma4_a4b_moe_common import align_experts_reference
from probes.gemma4.gemma4_a4b_moe_common import allocate_moe
from probes.gemma4.gemma4_a4b_moe_common import expert_weight_traffic_bytes
from probes.gemma4.gemma4_a4b_moe_common import max_aligned_tiles
from probes.gemma4.gemma4_a4b_moe_common import moe_reference
from probes.gemma4.gemma4_a4b_moe_common import routing_histogram
from probes.gemma4.gemma4_a4b_moe_common import tiles_per_expert as tiles_per_expert_of
from probes.gemma4.helion_gemma4_26b_a4b_layer import expert_down
from probes.gemma4.helion_gemma4_26b_a4b_layer import expert_down_reduce
from probes.gemma4.helion_gemma4_26b_a4b_layer import expert_gate_up
from probes.gemma4.helion_gemma4_26b_a4b_layer import expert_geglu_projection
from probes.gemma4.helion_gemma4_26b_a4b_layer import gemma4_route_topk
from probes.gemma4.helion_gemma4_26b_a4b_layer import router_mm_fp32
from probes.gemma4.helion_gemma4_26b_a4b_layer import router_norm_scale
from probes.gemma4.helion_gemma4_26b_a4b_layer import router_projection_topk
from probes.gemma4.helion_gemma4_26b_a4b_layer import weighted_expert_reduce
from probes.gemma4.helion_gemma4_e4b_layer import geglu
from probes.gemma4.helion_gemma4_e4b_layer import rms_norm

import helion
import helion.language as hl

DEFAULT_CONFIG_PATH = str(Path(__file__).with_name("gemma4_a4b_moe_b200_configs.json"))


# ---------------------------------------------------------------------------
# Expert-grouped kernels.
#
# The batch-1 kernels above process one (token, slot) assignment per row, which
# reloads an expert's weights once per assignment.  That is exact at batch 1,
# where every assignment already has a distinct expert, but at larger batches it
# multiplies the DRAM floor by the number of tokens sharing an expert.  The
# kernels below instead group assignments by expert so each expert's weights are
# streamed once, which is what makes the per-expert token counts ragged.
# ---------------------------------------------------------------------------


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def moe_expert_tiles(
    topk_ids: torch.Tensor,
    tile_tokens: int,
    tiles_per_expert: int,
    num_experts: int,
    max_active_tiles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-expert token counts plus the packed list of occupied expert tiles.

    ``tile_end`` is the inclusive prefix sum of each expert's tile count, so
    ``tile_end[num_experts - 1]`` is the number of occupied tiles and
    ``tile_end - tiles`` is where each expert's tiles start in ``active_tiles``.
    """
    batch, top_k = topk_ids.size()
    assignments = batch * top_k
    tile_tokens = hl.specialize(tile_tokens)
    tiles_per_expert = hl.specialize(tiles_per_expert)
    num_experts = hl.specialize(num_experts)
    device = topk_ids.device
    counts_out = torch.empty((num_experts,), dtype=torch.int32, device=device)
    tile_end_out = torch.empty((num_experts,), dtype=torch.int32, device=device)
    # Only the first ``tile_end[-1]`` entries of ``active_out`` are written, and
    # which entries those are is a routing outcome.  Fill the tail with a
    # sentinel in the wrapper rather than leaving it uninitialized: consumers
    # already guard on the count, but an uninitialized output makes the kernel
    # non-deterministic and therefore impossible to autotune.
    active_out = torch.full((max_active_tiles,), -1, dtype=torch.int32, device=device)
    flat = topk_ids.view(assignments)
    for _ in hl.grid(1):
        experts = hl.arange(num_experts)
        counts = hl.zeros([num_experts], dtype=torch.int32)
        for tile_a in hl.tile(assignments):
            ids = flat[tile_a].to(torch.int32)
            counts = counts + torch.sum(
                (ids[None, :] == experts[:, None]).to(torch.int32), dim=-1
            ).to(torch.int32)
        tiles = (counts + (tile_tokens - 1)) // tile_tokens
        tile_end = torch.cumsum(tiles, dim=0).to(torch.int32)
        hl.store(counts_out, [experts], counts)
        hl.store(tile_end_out, [experts], tile_end)
        tile_start = tile_end - tiles
        for local in range(tiles_per_expert):
            hl.store(
                active_out,
                [tile_start + local],
                (experts * tiles_per_expert + local).to(torch.int32),
                extra_mask=local < tiles,
            )
    return counts_out, tile_end_out, active_out


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def moe_assignment_order(
    topk_ids: torch.Tensor,
    tile_tokens: int,
    tiles_per_expert: int,
    num_experts: int,
) -> torch.Tensor:
    """Place every (token, slot) assignment in stable expert-major order.

    Slot ``expert * tiles_per_expert * tile_tokens + rank`` holds the assignment
    whose rank among the assignments choosing that expert is ``rank``.  The rank
    is counted directly rather than accumulated with atomics so the placement is
    reproducible across replays.
    """
    batch, top_k = topk_ids.size()
    assignments = batch * top_k
    tile_tokens = hl.specialize(tile_tokens)
    tiles_per_expert = hl.specialize(tiles_per_expert)
    num_experts = hl.specialize(num_experts)
    stride = tiles_per_expert * tile_tokens
    # Each expert owns a fixed-stride span but fills only ``counts[e]`` of it,
    # so the padding is sentinel-filled here for the same reason as in
    # ``moe_expert_tiles``.
    order = torch.full(
        (num_experts * stride,), -1, dtype=torch.int32, device=topk_ids.device
    )
    flat = topk_ids.view(assignments)
    for tile_a in hl.tile(assignments):
        mine = flat[tile_a].to(torch.int32)
        rank = hl.zeros([tile_a], dtype=torch.int32)
        for tile_b in hl.tile(assignments):
            other = flat[tile_b].to(torch.int32)
            earlier = tile_b.index[None, :] < tile_a.index[:, None]
            same = other[None, :] == mine[:, None]
            rank = rank + torch.sum((earlier & same).to(torch.int32), dim=-1).to(
                torch.int32
            )
        hl.store(order, [mine * stride + rank], tile_a.index.to(torch.int32))
    return order


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def grouped_expert_geglu_projection(
    expert_input: torch.Tensor,
    expert_weight: torch.Tensor,
    active_tiles: torch.Tensor,
    tile_end: torch.Tensor,
    expert_counts: torch.Tensor,
    order: torch.Tensor,
    tile_tokens: int,
    tiles_per_expert: int,
    top_k: int,
) -> torch.Tensor:
    """Expert-grouped gate/up projection with its GeGLU epilogue."""
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
    expert_stride = tiles_per_expert * tile_tokens
    flat_weight = expert_weight.view(num_experts * twice_intermediate, hidden)
    output = torch.empty(
        (batch * top_k, intermediate),
        dtype=expert_input.dtype,
        device=expert_input.device,
    )
    for tile_t, tile_i in hl.tile([max_active, intermediate], block_size=[1, None]):
        if tile_t.id < hl.load(tile_end, [num_experts - 1]):
            group = hl.load(active_tiles, [tile_t.id])
            expert = group // tiles_per_expert
            local = group - expert * tiles_per_expert
            local_rows = local * tile_tokens + hl.arange(tile_tokens)
            row_valid = local_rows < hl.load(expert_counts, [expert])
            assignment = hl.load(
                order,
                [expert * expert_stride + local_rows],
                extra_mask=row_valid,
            )
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
                [assignment, tile_i.index],
                _gelu_tanh(gate).to(up.dtype) * up,
                extra_mask=row_valid[:, None],
            )
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def grouped_expert_down(
    activation: torch.Tensor,
    expert_weight: torch.Tensor,
    active_tiles: torch.Tensor,
    tile_end: torch.Tensor,
    expert_counts: torch.Tensor,
    order: torch.Tensor,
    tile_tokens: int,
    tiles_per_expert: int,
) -> torch.Tensor:
    """Expert-grouped down projection, one row per (token, slot) assignment."""
    assignments, intermediate = activation.size()
    num_experts, hidden, weight_intermediate = expert_weight.size()
    assert intermediate == weight_intermediate
    max_active = active_tiles.size(0)
    tile_tokens = hl.specialize(tile_tokens)
    tiles_per_expert = hl.specialize(tiles_per_expert)
    hl.specialize(num_experts)
    hl.specialize(hidden)
    expert_stride = tiles_per_expert * tile_tokens
    flat_weight = expert_weight.view(num_experts * hidden, intermediate)
    output = torch.empty(
        (assignments, hidden), dtype=activation.dtype, device=activation.device
    )
    for tile_t, tile_n in hl.tile([max_active, hidden], block_size=[1, None]):
        if tile_t.id < hl.load(tile_end, [num_experts - 1]):
            group = hl.load(active_tiles, [tile_t.id])
            expert = group // tiles_per_expert
            local = group - expert * tiles_per_expert
            local_rows = local * tile_tokens + hl.arange(tile_tokens)
            row_valid = local_rows < hl.load(expert_counts, [expert])
            assignment = hl.load(
                order,
                [expert * expert_stride + local_rows],
                extra_mask=row_valid,
            )
            weight_row = expert * hidden + tile_n.index
            acc = hl.zeros([tile_tokens, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(intermediate):
                values = hl.load(
                    activation,
                    [assignment, tile_k.index],
                    extra_mask=row_valid[:, None],
                )
                acc = torch.addmm(acc, values, flat_weight[weight_row, tile_k].T)
            hl.store(
                output,
                [assignment, tile_n.index],
                acc.to(output.dtype),
                extra_mask=row_valid[:, None],
            )
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def batched_expert_reduce(
    expert_outputs: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Routing-weighted reduction over each token's top-k expert outputs."""
    batch, top_k = topk_weights.size()
    _, hidden = expert_outputs.size()
    top_k = hl.specialize(top_k)
    grouped = expert_outputs.view(batch, top_k, hidden)
    output = torch.empty(
        (batch, hidden), dtype=expert_outputs.dtype, device=expert_outputs.device
    )
    for tile_b, tile_n in hl.tile([batch, hidden], block_size=[1, None]):
        values = grouped[tile_b, :, tile_n].to(torch.float32)
        weights = topk_weights[tile_b, :]
        output[tile_b, tile_n] = torch.sum(values * weights[:, :, None], dim=1).to(
            output.dtype
        )
    return output


def compile_default(kernel, kernel_args):
    bound = kernel.bind(kernel_args)
    config = bound.config_spec.default_config()
    return config, bound.compile_config(config)


def compile_config(kernel, kernel_args, config_dict):
    bound = kernel.bind(kernel_args)
    config = helion.Config.from_dict(config_dict)
    bound.config_spec.normalize(config.config)
    return config, bound.compile_config(config)


def tune_kernel(name, kernel, kernel_args, configs, config_path):
    print(f"autotune_start {name}", flush=True)
    started = time.perf_counter()
    bound = kernel.bind(kernel_args)
    config = bound.autotune(kernel_args, force=True)
    elapsed = time.perf_counter() - started
    configs[name] = dict(config)
    if config_path is not None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(configs, indent=2, sort_keys=True) + "\n")
    print(
        "autotune_result",
        json.dumps(
            {"name": name, "seconds": elapsed, "config": dict(config)},
            sort_keys=True,
        ),
        flush=True,
    )
    return config, bound.compile_config(config)


def _assert_close(name, actual, expected, *, atol=8e-2, rtol=4e-2):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    max_abs = float((actual.float() - expected.float()).abs().max().item())
    print(f"correctness {name} max_abs={max_abs:.6f}", flush=True)


def build_moe(args, tensors, shape, configs, config_path):
    """Instantiate every extracted kernel and return the two launch paths."""
    tune_set = set(args.tune or [])
    tune_all = "all" in tune_set
    selected_configs = {}
    prefix = f"moe_b{shape.batch}"

    def build(name, kernel, kernel_args):
        qualified = f"{prefix}_{name}"
        if tune_all or name in tune_set or qualified in tune_set:
            config, compiled = tune_kernel(
                qualified, kernel, kernel_args, configs, config_path
            )
        elif qualified in configs:
            config, compiled = compile_config(kernel, kernel_args, configs[qualified])
        else:
            config, compiled = compile_default(kernel, kernel_args)
        selected_configs[qualified] = dict(config)
        return compiled

    residual = tensors["residual"]
    root_size = tensors["root_size"]
    gathered = shape.batch == 1

    # 1. pre-MoE RMSNorm.
    expert_input_args = (residual, tensors["pre_ff_norm_weight_2"], shape.eps)
    expert_input_kernel = build("expert_pre_norm", rms_norm, expert_input_args)
    expert_input = expert_input_kernel(*expert_input_args)

    # 2-4. router preprocessing, projection, and top-k, unfused.
    router_norm_args = (residual, tensors["router_scale"], root_size, shape.eps)
    router_norm_kernel = build("router_norm_scale", router_norm_scale, router_norm_args)
    router_hidden = router_norm_kernel(*router_norm_args)
    router_mm_args = (router_hidden, tensors["router_weight"])
    router_mm_kernel = build("router_mm_fp32", router_mm_fp32, router_mm_args)
    router_logits = router_mm_kernel(*router_mm_args)
    route_args = (router_logits, tensors["per_expert_scale"], shape.top_k)
    route_kernel = build("route_topk", gemma4_route_topk, route_args)
    topk_weights, topk_ids = route_kernel(*route_args)

    # 2-4 fused.
    router_fused_args = (
        residual,
        tensors["router_scale"],
        root_size,
        tensors["router_weight"],
        tensors["per_expert_scale"],
        shape.top_k,
        shape.eps,
    )
    router_fused_kernel = build(
        "router_projection_topk", router_projection_topk, router_fused_args
    )
    router_fused_kernel(*router_fused_args)

    # 5-8, gathered: one GEMV per (token, slot) assignment.  Exact only at
    # batch 1, where no two assignments share an expert.
    gate_up_output = None
    activation = None
    expert_outputs = None
    if gathered:
        gate_up_args = (expert_input, tensors["expert_gate_up_weight"], topk_ids)
        gate_up_kernel = build("expert_gate_up", expert_gate_up, gate_up_args)
        gate_up_output = gate_up_kernel(*gate_up_args)
        geglu_args = (gate_up_output,)
        geglu_kernel = build("expert_geglu", geglu, geglu_args)
        activation = geglu_kernel(*geglu_args)
        fused_gate_up_args = (expert_input, tensors["expert_gate_up_weight"], topk_ids)
        fused_gate_up_kernel = build(
            "expert_geglu_projection", expert_geglu_projection, fused_gate_up_args
        )
        fused_gate_up_kernel(*fused_gate_up_args)

        down_args = (activation, tensors["expert_down_weight"], topk_ids)
        down_kernel = build("expert_down", expert_down, down_args)
        expert_outputs = down_kernel(*down_args)
        reduce_args = (expert_outputs, topk_weights)
        reduce_kernel = build("expert_reduce", weighted_expert_reduce, reduce_args)
        moe_down = reduce_kernel(*reduce_args)
        down_reduce_args = (
            activation,
            tensors["expert_down_weight"],
            topk_ids,
            topk_weights,
        )
        down_reduce_kernel = build(
            "expert_down_reduce", expert_down_reduce, down_reduce_args
        )
        down_reduce_kernel(*down_reduce_args)

    # 5-8, grouped: assignments sorted expert-major so each expert's weights are
    # streamed once.  This is the formulation whose per-expert tile counts are
    # ragged, and it is the one the megakernel schedules.
    tile_tokens = args.tile_tokens
    tiles_per_expert = tiles_per_expert_of(shape, tile_tokens)
    max_active = max_aligned_tiles(shape, tile_tokens)
    tiles_args = (
        topk_ids,
        tile_tokens,
        tiles_per_expert,
        shape.num_experts,
        max_active,
    )
    tiles_kernel = build(f"expert_tiles_t{tile_tokens}", moe_expert_tiles, tiles_args)
    expert_counts, tile_end, active_tiles = tiles_kernel(*tiles_args)
    order_args = (topk_ids, tile_tokens, tiles_per_expert, shape.num_experts)
    order_kernel = build(
        f"assignment_order_t{tile_tokens}", moe_assignment_order, order_args
    )
    order = order_kernel(*order_args)
    grouped_gate_up_args = (
        expert_input,
        tensors["expert_gate_up_weight"],
        active_tiles,
        tile_end,
        expert_counts,
        order,
        tile_tokens,
        tiles_per_expert,
        shape.top_k,
    )
    grouped_gate_up_kernel = build(
        f"grouped_expert_geglu_projection_t{tile_tokens}",
        grouped_expert_geglu_projection,
        grouped_gate_up_args,
    )
    grouped_activation = grouped_gate_up_kernel(*grouped_gate_up_args)
    grouped_down_args = (
        grouped_activation,
        tensors["expert_down_weight"],
        active_tiles,
        tile_end,
        expert_counts,
        order,
        tile_tokens,
        tiles_per_expert,
    )
    grouped_down_kernel = build(
        f"grouped_expert_down_t{tile_tokens}", grouped_expert_down, grouped_down_args
    )
    grouped_outputs = grouped_down_kernel(*grouped_down_args)
    batched_reduce_args = (grouped_outputs, topk_weights)
    batched_reduce_kernel = build(
        "batched_expert_reduce", batched_expert_reduce, batched_reduce_args
    )
    grouped_moe_down = batched_reduce_kernel(*batched_reduce_args)
    if not gathered:
        moe_down = grouped_moe_down

    # 9. post-MoE RMSNorm.
    post_norm_args = (moe_down, tensors["post_ff_norm_weight_2"], shape.eps)
    post_norm_kernel = build("moe_post_norm", rms_norm, post_norm_args)
    moe_branch = post_norm_kernel(*post_norm_args)
    torch.cuda.synchronize()

    def launch_grouped():
        local_weights, local_ids = router_fused_kernel(*router_fused_args)
        local_expert_input = expert_input_kernel(*expert_input_args)
        local_counts, local_end, local_active = tiles_kernel(
            local_ids, tile_tokens, tiles_per_expert, shape.num_experts, max_active
        )
        local_order = order_kernel(
            local_ids, tile_tokens, tiles_per_expert, shape.num_experts
        )
        local_activation = grouped_gate_up_kernel(
            local_expert_input,
            tensors["expert_gate_up_weight"],
            local_active,
            local_end,
            local_counts,
            local_order,
            tile_tokens,
            tiles_per_expert,
            shape.top_k,
        )
        local_outputs = grouped_down_kernel(
            local_activation,
            tensors["expert_down_weight"],
            local_active,
            local_end,
            local_counts,
            local_order,
            tile_tokens,
            tiles_per_expert,
        )
        local_moe_down = batched_reduce_kernel(local_outputs, local_weights)
        return post_norm_kernel(
            local_moe_down, tensors["post_ff_norm_weight_2"], shape.eps
        )

    launches = {"launch_grouped": launch_grouped}
    stage_calls = {
        f"{prefix}_expert_pre_norm": lambda: expert_input_kernel(*expert_input_args),
        f"{prefix}_router_norm_scale": lambda: router_norm_kernel(*router_norm_args),
        f"{prefix}_router_mm_fp32": lambda: router_mm_kernel(*router_mm_args),
        f"{prefix}_route_topk": lambda: route_kernel(*route_args),
        f"{prefix}_router_projection_topk": lambda: router_fused_kernel(
            *router_fused_args
        ),
        f"{prefix}_expert_tiles": lambda: tiles_kernel(*tiles_args),
        f"{prefix}_assignment_order": lambda: order_kernel(*order_args),
        f"{prefix}_grouped_expert_geglu_projection": lambda: grouped_gate_up_kernel(
            *grouped_gate_up_args
        ),
        f"{prefix}_grouped_expert_down": lambda: grouped_down_kernel(
            *grouped_down_args
        ),
        f"{prefix}_batched_expert_reduce": lambda: batched_reduce_kernel(
            *batched_reduce_args
        ),
        f"{prefix}_moe_post_norm": lambda: post_norm_kernel(*post_norm_args),
    }
    stage_outputs = {
        "expert_input": expert_input,
        "router_logits": router_logits,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "grouped_activation": grouped_activation,
        "grouped_outputs": grouped_outputs,
        "grouped_moe_down": grouped_moe_down,
        "expert_counts": expert_counts,
        "tile_end": tile_end,
        "active_tiles": active_tiles,
        "order": order,
        "moe_down": moe_down,
        "moe_branch": moe_branch,
    }

    if gathered:

        def launch_matched():
            local_expert_input = expert_input_kernel(*expert_input_args)
            local_router_hidden = router_norm_kernel(*router_norm_args)
            local_logits = router_mm_kernel(
                local_router_hidden, tensors["router_weight"]
            )
            local_weights, local_ids = route_kernel(
                local_logits, tensors["per_expert_scale"], shape.top_k
            )
            local_gate_up = gate_up_kernel(
                local_expert_input, tensors["expert_gate_up_weight"], local_ids
            )
            local_activation = geglu_kernel(local_gate_up)
            local_outputs = down_kernel(
                local_activation, tensors["expert_down_weight"], local_ids
            )
            local_moe_down = reduce_kernel(local_outputs, local_weights)
            return post_norm_kernel(
                local_moe_down, tensors["post_ff_norm_weight_2"], shape.eps
            )

        def launch_optimized():
            local_weights, local_ids = router_fused_kernel(*router_fused_args)
            local_expert_input = expert_input_kernel(*expert_input_args)
            local_activation = fused_gate_up_kernel(
                local_expert_input, tensors["expert_gate_up_weight"], local_ids
            )
            local_moe_down = down_reduce_kernel(
                local_activation,
                tensors["expert_down_weight"],
                local_ids,
                local_weights,
            )
            return post_norm_kernel(
                local_moe_down, tensors["post_ff_norm_weight_2"], shape.eps
            )

        launches["launch_matched"] = launch_matched
        launches["launch_optimized"] = launch_optimized
        stage_outputs["expert_gate_up"] = gate_up_output
        stage_outputs["expert_activation"] = activation
        stage_outputs["expert_outputs"] = expert_outputs
        stage_calls.update(
            {
                f"{prefix}_expert_gate_up": lambda: gate_up_kernel(*gate_up_args),
                f"{prefix}_expert_geglu": lambda: geglu_kernel(*geglu_args),
                f"{prefix}_expert_geglu_projection": lambda: fused_gate_up_kernel(
                    *fused_gate_up_args
                ),
                f"{prefix}_expert_down": lambda: down_kernel(*down_args),
                f"{prefix}_expert_reduce": lambda: reduce_kernel(*reduce_args),
                f"{prefix}_expert_down_reduce": lambda: down_reduce_kernel(
                    *down_reduce_args
                ),
            }
        )

    return {
        **launches,
        "configs": selected_configs,
        "tiling": {
            "tile_tokens": tile_tokens,
            "tiles_per_expert": tiles_per_expert,
            "max_active_tiles": max_active,
        },
        "stage_outputs": stage_outputs,
        "stage_calls": stage_calls,
    }


def _check_alignment(stages, reference, shape, tiling):
    """Confirm the device-side expert compaction matches the host model."""
    expected = align_experts_reference(
        reference["topk_ids"],
        shape.num_experts,
        tiling["tile_tokens"],
        tiling["tiles_per_expert"],
    )
    num_active = expected["num_active_tiles"]
    torch.testing.assert_close(stages["expert_counts"], expected["expert_counts"])
    torch.testing.assert_close(stages["tile_end"], expected["tile_end"])
    torch.testing.assert_close(
        stages["active_tiles"][:num_active], expected["active_tiles"]
    )
    valid = expected["order"] >= 0
    torch.testing.assert_close(stages["order"][valid], expected["order"][valid])
    print(
        f"correctness alignment num_active_tiles={num_active} "
        f"max_active_tiles={tiling['max_active_tiles']}",
        flush=True,
    )


def resolve_tile_tokens(requested, batch):
    """Largest power-of-two token tile that does not exceed the batch."""
    if requested:
        return requested
    tile = 1
    while tile * 2 <= min(batch, 16):
        tile *= 2
    return tile


def run_batch(args, batch, configs, config_path):
    shape = Gemma4A4BMoEShape(batch=batch)
    args.tile_tokens = resolve_tile_tokens(args.requested_tile_tokens, batch)
    tensors = allocate_moe(shape, args.seed, route_skew=args.route_skew)
    reference = moe_reference(tensors, shape)
    built = build_moe(args, tensors, shape, configs, config_path)
    stages = built["stage_outputs"]

    _assert_close("expert_input", stages["expert_input"], reference["expert_input"])
    _assert_close(
        "router_logits", stages["router_logits"], reference["router_logits"], atol=0.1
    )
    torch.testing.assert_close(stages["topk_ids"], reference["topk_ids"])
    _assert_close(
        "topk_weights", stages["topk_weights"], reference["topk_weights"], atol=2e-3
    )
    if "expert_gate_up" in stages:
        _assert_close(
            "expert_gate_up",
            stages["expert_gate_up"].reshape(reference["expert_gate_up"].shape),
            reference["expert_gate_up"],
            atol=0.2,
            rtol=0.08,
        )
    _assert_close(
        "grouped_activation",
        stages["grouped_activation"].reshape(reference["expert_activation"].shape),
        reference["expert_activation"],
        atol=0.2,
        rtol=0.08,
    )
    _assert_close(
        "grouped_moe_down",
        stages["grouped_moe_down"],
        reference["moe_down"],
        atol=0.25,
        rtol=0.1,
    )
    _assert_close(
        "moe_down", stages["moe_down"], reference["moe_down"], atol=0.25, rtol=0.1
    )
    _assert_close(
        "moe_branch", stages["moe_branch"], reference["moe_branch"], atol=0.3, rtol=0.1
    )
    _check_alignment(stages, reference, shape, built["tiling"])

    eager = {
        name.removeprefix("launch_"): built[name]()
        for name in ("launch_matched", "launch_optimized", "launch_grouped")
        if name in built
    }
    torch.cuda.synchronize()
    for name, value in eager.items():
        _assert_close(
            f"{name}_eager", value, reference["moe_branch"], atol=0.3, rtol=0.1
        )

    result = {
        "batch": batch,
        "route_skew": args.route_skew,
        "benchmark_mode": benchmark_cache_mode(),
        "tiling": built["tiling"],
        "routing": routing_histogram(reference["topk_ids"], shape.num_experts),
        "weight_traffic": expert_weight_traffic_bytes(reference["topk_ids"], shape),
    }
    if args.include_configs:
        result["configs"] = built["configs"]
    if args.smoke and not args.benchmark:
        result["status"] = "smoke_ok"
        return result

    graphs = {}
    for name in ("launch_matched", "launch_optimized", "launch_grouped"):
        if name not in built:
            continue
        short = name.removeprefix("launch_")
        graph, output = capture(built[name])
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(
            f"{short}_graph", output, reference["moe_branch"], atol=0.3, rtol=0.1
        )
        graphs[f"helion_moe_b{batch}_{short}"] = graph.replay

    benchmark_pids = visible_gpu_pids()
    timings = benchmark_interleaved(graphs, args.repeats, args.batch_replays)
    if args.benchmark_stages:
        stage_graphs = {
            stage: capture(call)[0].replay
            for stage, call in built["stage_calls"].items()
        }
        timings.update(
            benchmark_interleaved(stage_graphs, args.repeats, args.batch_replays)
        )
    if visible_gpu_pids() != benchmark_pids:
        raise RuntimeError("GPU process set changed during benchmark")
    result["timings"] = timings
    return result


def run(args):
    require_idle_visible_gpu()
    config_path = Path(args.config_path) if args.config_path else None
    configs = (
        json.loads(config_path.read_text())
        if config_path is not None and config_path.exists()
        else {}
    )
    if args.list_stages:
        for name, description, fused_into in MOE_STAGES:
            print(f"stage {name:22s} {description:58s} optimized={fused_into}")
    results = [run_batch(args, batch, configs, config_path) for batch in args.batch]
    print(
        "RESULT_JSON",
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "helion_module": helion.__file__,
                "batches": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, nargs="+", default=[1])
    parser.add_argument("--route-skew", type=float, default=0.0)
    parser.add_argument(
        "--tile-tokens", dest="requested_tile_tokens", type=int, default=0
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch-replays", type=int, default=20)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tune", nargs="*", default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-stages", action="store_true")
    parser.add_argument("--include-configs", action="store_true")
    parser.add_argument("--list-stages", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
