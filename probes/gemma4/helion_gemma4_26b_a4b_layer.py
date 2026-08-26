# ruff: noqa: ANN001, ANN201, ANN202
"""Helion/Triton Gemma 4 26B-A4B production decode-layer benchmark.

The A4B implementation lives separately from the earlier E4B work.  It reuses
only shape-generic Helion primitives (RMSNorm, dense GEMV, RoPE/cache, and
paged attention) and adds A4B-specific router and top-8 expert kernels here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from probes.common import benchmark_cache_mode
from probes.gemma4.gemma4_26b_a4b_common import A4B_LAYER_COUNTS
from probes.gemma4.gemma4_26b_a4b_common import A4B_REPRESENTATIVE_LAYERS
from probes.gemma4.gemma4_26b_a4b_common import Gemma4A4BShape
from probes.gemma4.gemma4_26b_a4b_common import allocate_layer
from probes.gemma4.gemma4_26b_a4b_common import benchmark_interleaved
from probes.gemma4.gemma4_26b_a4b_common import capture
from probes.gemma4.gemma4_26b_a4b_common import layer_reference
from probes.gemma4.gemma4_26b_a4b_common import require_idle_visible_gpu
from probes.gemma4.gemma4_26b_a4b_common import variant_name
from probes.gemma4.gemma4_26b_a4b_common import visible_gpu_pids
from probes.gemma4.helion_gemma4_e4b_layer import bf16_mm
from probes.gemma4.helion_gemma4_e4b_layer import geglu
from probes.gemma4.helion_gemma4_e4b_layer import geglu_projection
from probes.gemma4.helion_gemma4_e4b_layer import merge_attention
from probes.gemma4.helion_gemma4_e4b_layer import paged_attention
from probes.gemma4.helion_gemma4_e4b_layer import paged_attention_split
from probes.gemma4.helion_gemma4_e4b_layer import post_attention_residual_pre_ff_norm
from probes.gemma4.helion_gemma4_e4b_layer import qkv_norm_rope
from probes.gemma4.helion_gemma4_e4b_layer import qkv_norm_rope_cache
from probes.gemma4.helion_gemma4_e4b_layer import reshape_and_cache
from probes.gemma4.helion_gemma4_e4b_layer import rms_norm
from probes.gemma4.helion_gemma4_e4b_layer import rms_qkv_mm

import helion
import helion.language as hl


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    coefficient = 0.7978845608028654
    return 0.5 * x * (1.0 + torch.tanh(coefficient * (x + 0.044715 * x * x * x)))


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def router_norm_scale(
    hidden: torch.Tensor,
    scale: torch.Tensor,
    root_size: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Gemma4Router's unweighted RMSNorm, root-size, and learned scale."""
    m, n = hidden.size()
    hl.specialize(n)
    output = torch.empty_like(hidden)
    for tile_m in hl.tile(m, block_size=1):
        values = hidden[tile_m, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(values * values, dim=-1) + eps)
        normalized = (values * inv_rms[:, None]).to(hidden.dtype)
        root = hl.load(root_size, [])
        output[tile_m, :] = normalized * root * scale[None, :]
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def router_mm_fp32(
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """BF16 router projection with an FP32 output, matching GateLinear."""
    m, k = hidden.size()
    n, weight_k = weight.size()
    assert k == weight_k
    output = torch.empty((m, n), dtype=torch.float32, device=hidden.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                hidden[tile_m, tile_k],
                weight[tile_n, tile_k].T,
            )
        output[tile_m, tile_n] = acc
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def gemma4_route_topk(
    logits: torch.Tensor,
    per_expert_scale: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Gemma4 top-k routing and selected-probability renormalization."""
    m, _ = logits.size()
    top_k = hl.specialize(top_k)
    weights = torch.empty((m, top_k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((m, top_k), dtype=torch.int32, device=logits.device)
    for tile_m in hl.tile(m, block_size=1):
        top_values, top_ids = torch.topk(logits[tile_m, :], top_k, dim=-1, largest=True)
        shifted = top_values - torch.amax(top_values, dim=-1, keepdim=True)
        raw_weights = torch.exp(shifted)
        normalized = raw_weights / torch.sum(raw_weights, dim=-1, keepdim=True)
        weights[tile_m, :] = normalized * per_expert_scale[top_ids].to(torch.float32)
        ids[tile_m, :] = top_ids.to(torch.int32)
    return weights, ids


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def router_projection_topk(
    hidden: torch.Tensor,
    scale: torch.Tensor,
    root_size: torch.Tensor,
    router_weight: torch.Tensor,
    per_expert_scale: torch.Tensor,
    top_k: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse router preprocessing, projection, top-k, and route weighting."""
    m, k = hidden.size()
    num_experts, weight_k = router_weight.size()
    assert k == weight_k
    top_k = hl.specialize(top_k)
    hl.specialize(k)
    hl.specialize(num_experts)
    weights = torch.empty((m, top_k), dtype=torch.float32, device=hidden.device)
    ids = torch.empty((m, top_k), dtype=torch.int32, device=hidden.device)
    for tile_m in hl.tile(m, block_size=1):
        squared_sum = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(k):
            values = hidden[tile_m, tile_k].to(torch.float32)
            squared_sum = squared_sum + torch.sum(values * values, dim=-1)
        inv_rms = torch.rsqrt(squared_sum * (1.0 / k) + eps)
        logits = hl.zeros([tile_m, num_experts], dtype=torch.float32)
        root = hl.load(root_size, [])
        for tile_k in hl.tile(k):
            values = hidden[tile_m, tile_k].to(torch.float32)
            normalized = (values * inv_rms[:, None]).to(hidden.dtype)
            router_input = normalized * root * scale[tile_k]
            logits = torch.addmm(logits, router_input, router_weight[:, tile_k].T)
        top_values, top_ids = torch.topk(logits, top_k, dim=-1, largest=True)
        shifted = top_values - torch.amax(top_values, dim=-1, keepdim=True)
        raw_weights = torch.exp(shifted)
        normalized_weights = raw_weights / torch.sum(raw_weights, dim=-1, keepdim=True)
        weights[tile_m, :] = normalized_weights * per_expert_scale[top_ids].to(
            torch.float32
        )
        ids[tile_m, :] = top_ids.to(torch.int32)
    return weights, ids


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def expert_gate_up(
    hidden: torch.Tensor,
    expert_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather eight experts and execute their fused gate/up GEMVs."""
    m, hidden_size = hidden.size()
    num_experts, twice_intermediate, weight_hidden = expert_weight.size()
    assert m == 1
    assert hidden_size == weight_hidden
    top_k = topk_ids.size(1)
    hl.specialize(num_experts)
    hl.specialize(twice_intermediate)
    flattened_weight = expert_weight.view(num_experts * twice_intermediate, hidden_size)
    output = torch.empty(
        (top_k * twice_intermediate, 1),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    flat_ids = topk_ids.view(top_k)
    for tile_row in hl.tile(top_k * twice_intermediate):
        expert_slot = tile_row.index // twice_intermediate
        expert_row = tile_row.index % twice_intermediate
        selected_expert = flat_ids[expert_slot]
        selected_row = selected_expert * twice_intermediate + expert_row
        acc = hl.zeros([tile_row, 1], dtype=torch.float32)
        for tile_k in hl.tile(hidden_size):
            acc = torch.addmm(
                acc,
                flattened_weight[selected_row, tile_k],
                hidden[:, tile_k].T,
            )
        output[tile_row, :] = acc.to(output.dtype)
    return output.view(top_k, twice_intermediate)


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def expert_geglu_projection(
    hidden: torch.Tensor,
    expert_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Fuse gathered expert gate/up GEMVs with their GeGLU epilogue."""
    m, hidden_size = hidden.size()
    num_experts, twice_intermediate, weight_hidden = expert_weight.size()
    assert m == 1
    assert hidden_size == weight_hidden
    intermediate = twice_intermediate // 2
    top_k = topk_ids.size(1)
    hl.specialize(num_experts)
    hl.specialize(intermediate)
    flattened_weight = expert_weight.view(num_experts * twice_intermediate, hidden_size)
    output = torch.empty(
        (top_k * intermediate, 1), dtype=hidden.dtype, device=hidden.device
    )
    flat_ids = topk_ids.view(top_k)
    for tile_row in hl.tile(top_k * intermediate):
        expert_slot = tile_row.index // intermediate
        expert_row = tile_row.index % intermediate
        selected_expert = flat_ids[expert_slot]
        gate_row = selected_expert * twice_intermediate + expert_row
        up_row = gate_row + intermediate
        gate_acc = hl.zeros([tile_row, 1], dtype=torch.float32)
        up_acc = hl.zeros([tile_row, 1], dtype=torch.float32)
        for tile_k in hl.tile(hidden_size):
            gate_acc = torch.addmm(
                gate_acc,
                flattened_weight[gate_row, tile_k],
                hidden[:, tile_k].T,
            )
            up_acc = torch.addmm(
                up_acc,
                flattened_weight[up_row, tile_k],
                hidden[:, tile_k].T,
            )
        gate = gate_acc.to(torch.bfloat16).to(torch.float32)
        up = up_acc.to(torch.bfloat16)
        output[tile_row, :] = _gelu_tanh(gate).to(up.dtype) * up
    return output.view(top_k, intermediate)


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def expert_down(
    activation: torch.Tensor,
    expert_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Gathered expert down projections before routing-weight reduction."""
    top_k, intermediate = activation.size()
    num_experts, hidden_size, weight_intermediate = expert_weight.size()
    assert intermediate == weight_intermediate
    hl.specialize(num_experts)
    hl.specialize(hidden_size)
    flat_ids = topk_ids.view(top_k)
    output = torch.empty(
        (top_k, hidden_size), dtype=activation.dtype, device=activation.device
    )
    for tile_n in hl.tile(hidden_size):
        selected_experts = flat_ids[:]
        acc = hl.zeros([top_k, 1, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(intermediate):
            lhs = activation[:, tile_k].view(top_k, 1, tile_k)
            rhs = expert_weight[
                selected_experts[:, None, None],
                tile_n.index[None, None, :],
                tile_k.index[None, :, None],
            ]
            acc = torch.baddbmm(
                acc,
                lhs,
                rhs,
            )
        output[:, tile_n] = acc.squeeze(1).to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def weighted_expert_reduce(
    expert_output: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    top_k, hidden_size = expert_output.size()
    output = torch.empty(
        (1, hidden_size), dtype=expert_output.dtype, device=expert_output.device
    )
    flat_weights = topk_weights.view(top_k)
    for tile_n in hl.tile(hidden_size):
        values = expert_output[:, tile_n].to(torch.float32)
        weights = flat_weights[:].view(top_k, 1)
        output[:, tile_n] = torch.sum(values * weights, dim=0, keepdim=True).to(
            output.dtype
        )
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def expert_down_reduce(
    activation: torch.Tensor,
    expert_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Fuse gathered expert down GEMVs with the weighted expert reduction."""
    top_k, intermediate = activation.size()
    num_experts, hidden_size, weight_intermediate = expert_weight.size()
    assert intermediate == weight_intermediate
    hl.specialize(num_experts)
    hl.specialize(hidden_size)
    flat_ids = topk_ids.view(top_k)
    flat_weights = topk_weights.view(top_k)
    output = torch.empty(
        (1, hidden_size), dtype=activation.dtype, device=activation.device
    )
    for tile_n in hl.tile(hidden_size):
        selected_experts = flat_ids[:]
        expert_acc = hl.zeros([top_k, 1, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(intermediate):
            lhs = activation[:, tile_k].view(top_k, 1, tile_k)
            rhs = expert_weight[
                selected_experts[:, None, None],
                tile_n.index[None, None, :],
                tile_k.index[None, :, None],
            ]
            expert_acc = torch.baddbmm(expert_acc, lhs, rhs)
        weighted = expert_acc.squeeze(1) * flat_weights[:][:, None]
        output[:, tile_n] = torch.sum(weighted, dim=0, keepdim=True).to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def add_branches(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(a)
    m, n = a.size()
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        output[tile_m, tile_n] = a[tile_m, tile_n] + b[tile_m, tile_n]
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def residual_scale(
    post_ff: torch.Tensor,
    residual: torch.Tensor,
    layer_scalar: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(post_ff)
    m, n = post_ff.size()
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        scalar = hl.load(layer_scalar, [])
        output[tile_m, tile_n] = (
            post_ff[tile_m, tile_n] + residual[tile_m, tile_n]
        ) * scalar
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def final_dense_moe_residual(
    dense_down: torch.Tensor,
    moe_down: torch.Tensor,
    dense_norm_weight: torch.Tensor,
    moe_norm_weight: torch.Tensor,
    post_ff_norm_weight: torch.Tensor,
    residual: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse the three terminal RMSNorms, branch add, residual, and scalar."""
    m, hidden_size = dense_down.size()
    hl.specialize(hidden_size)
    output = torch.empty_like(dense_down)
    for tile_m in hl.tile(m, block_size=1):
        dense = dense_down[tile_m, :].to(torch.float32)
        dense_inv = torch.rsqrt(torch.mean(dense * dense, dim=-1) + eps)
        dense_branch = (dense * dense_inv[:, None]).to(dense_down.dtype)
        dense_branch = dense_branch * dense_norm_weight[None, :]

        moe = moe_down[tile_m, :].to(torch.float32)
        moe_inv = torch.rsqrt(torch.mean(moe * moe, dim=-1) + eps)
        moe_branch = (moe * moe_inv[:, None]).to(moe_down.dtype)
        moe_branch = moe_branch * moe_norm_weight[None, :]

        combined = dense_branch + moe_branch
        combined_float = combined.to(torch.float32)
        final_inv = torch.rsqrt(
            torch.mean(combined_float * combined_float, dim=-1) + eps
        )
        normalized = (combined_float * final_inv[:, None]).to(combined.dtype)
        scalar = hl.load(layer_scalar, [])
        output[tile_m, :] = (
            normalized * post_ff_norm_weight[None, :] + residual[tile_m, :]
        ) * scalar
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def qk_norm_rope_cache_k_eq_v(
    qk: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    position: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    eps: float,
) -> None:
    """Global A4B Q/K normalization/cache path when the K projection is V."""
    num_tokens = qk.shape[0]
    total_heads = num_q_heads + num_kv_heads
    rotary_dim = cos_sin.shape[-1]
    half = rotary_dim // 2
    hl.specialize(qk.shape[1])
    hl.specialize(num_q_heads)
    hl.specialize(num_kv_heads)
    hl.specialize(head_dim)
    hl.specialize(rotary_dim)
    hl.specialize(block_size)
    packed = qk.view(num_tokens, total_heads, head_dim)

    for tile_m, tile_h, tile_d in hl.tile(
        [num_tokens, total_heads, head_dim],
        block_size=[1, None, head_dim],
    ):
        values = packed[tile_m, tile_h, tile_d].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(values * values, dim=-1) + eps)
        is_q = (tile_h.index < num_q_heads)[None, :, None]
        is_k = (tile_h.index >= num_q_heads)[None, :, None]
        learned_weight = torch.where(
            is_q,
            q_weight[None, None, tile_d],
            k_weight[None, None, tile_d],
        )
        normalized = (values * inv_rms[:, :, None]).to(qk.dtype) * learned_weight
        value_normalized = (values * inv_rms[:, :, None]).to(qk.dtype)
        packed[tile_m, tile_h, tile_d] = normalized

        slot = slot_mapping[tile_m.index]
        cache_block = (slot // block_size)[:, None, None]
        cache_offset = (slot % block_size)[:, None, None]
        d = tile_d.index[None, None, :]
        hl.store(
            kv_cache,
            [
                cache_block,
                cache_offset,
                (tile_h.index - num_q_heads)[None, :, None],
                d + head_dim,
            ],
            value_normalized,
            extra_mask=is_k,
        )

        rotary_offset = hl.arange(half)
        x1 = packed[tile_m, tile_h, rotary_offset]
        x2 = packed[tile_m, tile_h, rotary_offset + half]
        pos = position[tile_m]
        cos = cos_sin[pos, rotary_offset]
        sin = cos_sin[pos, rotary_offset + half]
        o1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        o2 = x2 * cos[:, None, :] + x1 * sin[:, None, :]
        packed[tile_m, tile_h, rotary_offset] = o1
        packed[tile_m, tile_h, rotary_offset + half] = o2
        hl.store(
            kv_cache,
            [
                cache_block,
                cache_offset,
                (tile_h.index - num_q_heads)[None, :, None],
                rotary_offset[None, None, :],
            ],
            o1,
            extra_mask=is_k,
        )
        hl.store(
            kv_cache,
            [
                cache_block,
                cache_offset,
                (tile_h.index - num_q_heads)[None, :, None],
                (rotary_offset + half)[None, None, :],
            ],
            o2,
            extra_mask=is_k,
        )


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


def build_layer(args, tensors, shape, geometry, configs, config_path):
    tune_set = set(args.tune or [])
    tune_all = "all" in tune_set
    selected_configs = {}

    def build(name, kernel, kernel_args):
        if tune_all or name in tune_set:
            config, compiled = tune_kernel(
                name, kernel, kernel_args, configs, config_path
            )
        elif name in configs:
            config, compiled = compile_config(kernel, kernel_args, configs[name])
        else:
            config, compiled = compile_default(kernel, kernel_args)
        selected_configs[name] = dict(config)
        return compiled

    q_width = shape.q_heads * geometry.head_dim
    kv_width = geometry.kv_heads * geometry.head_dim
    splits = args.full_splits if geometry.layer_type == "full" else args.sliding_splits
    prefix = f"a4b_{geometry.layer_type}"

    input_norm_args = (
        tensors["hidden_states"],
        tensors["input_norm_weight"],
        shape.eps,
    )
    input_norm_kernel = build(f"{prefix}_rms_hidden", rms_norm, input_norm_args)
    input_norm = input_norm_kernel(*input_norm_args)
    qkv_mm_args = (input_norm, tensors["qkv_weight"])
    qkv_mm_kernel = build(f"{prefix}_qkv_mm", bf16_mm, qkv_mm_args)
    qkv = qkv_mm_kernel(*qkv_mm_args)

    norm_args = (
        qkv,
        tensors["q_norm_weight"],
        tensors["k_norm_weight"],
        tensors["cos_sin"],
        tensors["position"],
        shape.q_heads,
        geometry.kv_heads,
        geometry.head_dim,
        shape.eps,
        True,
    )
    norm_kernel = build(f"{prefix}_qkv_norm_rope", qkv_norm_rope, norm_args)
    norm_kernel(*norm_args)
    query = qkv[:, :q_width].view(1, shape.q_heads, geometry.head_dim)
    key = qkv[:, q_width : q_width + kv_width].view(
        1, geometry.kv_heads, geometry.head_dim
    )
    value = qkv[:, q_width + kv_width :].view(1, geometry.kv_heads, geometry.head_dim)
    cache_args = (
        key,
        value,
        tensors["kv_cache"],
        tensors["slot_mapping"],
        shape.block_size,
    )
    cache_kernel = build(f"{prefix}_kv_cache", reshape_and_cache, cache_args)
    cache_kernel(*cache_args)

    fused_qkv_args = (
        tensors["hidden_states"],
        tensors["input_norm_weight"],
        tensors["qkv_weight"],
        shape.eps,
    )
    fused_qkv_kernel = build(f"{prefix}_rms_qkv_mm", rms_qkv_mm, fused_qkv_args)
    fused_qkv = fused_qkv_kernel(*fused_qkv_args)
    fused_norm_cache_args = (
        fused_qkv,
        tensors["q_norm_weight"],
        tensors["k_norm_weight"],
        tensors["cos_sin"],
        tensors["position"],
        tensors["kv_cache"],
        tensors["slot_mapping"],
        shape.q_heads,
        geometry.kv_heads,
        geometry.head_dim,
        shape.block_size,
        shape.eps,
    )
    fused_norm_cache_kernel = build(
        f"{prefix}_qkv_norm_rope_cache",
        qkv_norm_rope_cache,
        fused_norm_cache_args,
    )
    fused_norm_cache_kernel(*fused_norm_cache_args)

    qk_mm_kernel = None
    qk_mm_args = None
    qk_norm_cache_kernel = None
    qk_norm_cache_args = None
    if geometry.k_eq_v:
        qk_mm_args = (input_norm, tensors["qk_weight"])
        qk_mm_kernel = build(f"{prefix}_qk_mm", bf16_mm, qk_mm_args)
        qk = qk_mm_kernel(*qk_mm_args)
        qk_norm_cache_args = (
            qk,
            tensors["q_norm_weight"],
            tensors["k_norm_weight"],
            tensors["cos_sin"],
            tensors["position"],
            tensors["kv_cache"],
            tensors["slot_mapping"],
            shape.q_heads,
            geometry.kv_heads,
            geometry.head_dim,
            shape.block_size,
            shape.eps,
        )
        qk_norm_cache_kernel = build(
            f"{prefix}_qk_norm_rope_cache_k_eq_v",
            qk_norm_rope_cache_k_eq_v,
            qk_norm_cache_args,
        )
        qk_norm_cache_kernel(*qk_norm_cache_args)

    attention_args = (
        query,
        tensors["kv_cache"],
        tensors["block_table"],
        shape.context,
        geometry.attention_context,
        shape.block_size,
        shape.q_heads // geometry.kv_heads,
        splits,
    )
    attention_kernel = build(
        f"{prefix}_attention_s{splits}", paged_attention_split, attention_args
    )
    partial_out, partial_lse = attention_kernel(*attention_args)
    merge_args = (partial_out, partial_lse)
    merge_kernel = build(
        f"{prefix}_attention_merge_s{splits}", merge_attention, merge_args
    )
    attention = merge_kernel(*merge_args)
    direct_attention_args = None
    direct_attention_kernel = None
    direct_attention = attention
    if geometry.layer_type == "sliding":
        direct_attention_args = attention_args[:-1]
        direct_attention_kernel = build(
            f"{prefix}_attention_direct", paged_attention, direct_attention_args
        )
        direct_attention = direct_attention_kernel(*direct_attention_args)

    o_args = (attention.view(1, q_width), tensors["o_weight"])
    o_kernel = build(f"{prefix}_o_mm", bf16_mm, o_args)
    attention_out = o_kernel(*o_args)
    post_attention_args = (
        attention_out,
        tensors["hidden_states"],
        tensors["post_attention_norm_weight"],
        tensors["pre_ff_norm_weight"],
        shape.eps,
    )
    post_attention_kernel = build(
        f"{prefix}_post_attention_residual_pre_ff_norm",
        post_attention_residual_pre_ff_norm,
        post_attention_args,
    )
    residual, dense_input = post_attention_kernel(*post_attention_args)

    dense_gate_up_args = (dense_input, tensors["gate_up_weight"])
    dense_gate_up_kernel = build(
        f"{prefix}_dense_gate_up_mm", bf16_mm, dense_gate_up_args
    )
    dense_gate_up = dense_gate_up_kernel(*dense_gate_up_args)
    dense_geglu_args = (dense_gate_up,)
    dense_geglu_kernel = build(f"{prefix}_dense_geglu", geglu, dense_geglu_args)
    dense_activation = dense_geglu_kernel(*dense_geglu_args)
    dense_fused_args = (dense_input, tensors["gate_up_weight"])
    dense_fused_kernel = build(
        f"{prefix}_dense_geglu_projection", geglu_projection, dense_fused_args
    )
    dense_fused_kernel(*dense_fused_args)
    dense_down_args = (dense_activation, tensors["down_weight"])
    dense_down_kernel = build(f"{prefix}_dense_down_mm", bf16_mm, dense_down_args)
    dense_down = dense_down_kernel(*dense_down_args)
    dense_post_args = (
        dense_down,
        tensors["post_ff_norm_weight_1"],
        shape.eps,
    )
    dense_post_kernel = build(f"{prefix}_dense_post_norm", rms_norm, dense_post_args)
    dense_branch = dense_post_kernel(*dense_post_args)

    expert_input_args = (
        residual,
        tensors["pre_ff_norm_weight_2"],
        shape.eps,
    )
    expert_input_kernel = build(
        f"{prefix}_expert_pre_norm", rms_norm, expert_input_args
    )
    expert_input = expert_input_kernel(*expert_input_args)
    root_size = torch.tensor(shape.hidden**-0.5, device="cuda", dtype=torch.bfloat16)
    router_norm_args = (
        residual,
        tensors["router_scale"],
        root_size,
        shape.eps,
    )
    router_norm_kernel = build(
        f"{prefix}_router_norm_scale", router_norm_scale, router_norm_args
    )
    router_hidden = router_norm_kernel(*router_norm_args)
    router_mm_args = (router_hidden, tensors["router_weight"])
    router_mm_kernel = build(f"{prefix}_router_mm_fp32", router_mm_fp32, router_mm_args)
    router_logits = router_mm_kernel(*router_mm_args)
    route_args = (router_logits, tensors["per_expert_scale"], shape.top_k)
    route_kernel = build(f"{prefix}_route_topk", gemma4_route_topk, route_args)
    topk_weights, topk_ids = route_kernel(*route_args)
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
        f"{prefix}_router_projection_topk",
        router_projection_topk,
        router_fused_args,
    )
    router_fused_kernel(*router_fused_args)

    expert_gate_up_args = (
        expert_input,
        tensors["expert_gate_up_weight"],
        topk_ids,
    )
    expert_gate_up_kernel = build(
        f"{prefix}_expert_gate_up", expert_gate_up, expert_gate_up_args
    )
    expert_gate_up_output = expert_gate_up_kernel(*expert_gate_up_args)
    expert_geglu_args = (expert_gate_up_output,)
    expert_geglu_kernel = build(f"{prefix}_expert_geglu", geglu, expert_geglu_args)
    expert_activation = expert_geglu_kernel(*expert_geglu_args)
    expert_fused_args = (
        expert_input,
        tensors["expert_gate_up_weight"],
        topk_ids,
    )
    expert_fused_kernel = build(
        f"{prefix}_expert_geglu_projection",
        expert_geglu_projection,
        expert_fused_args,
    )
    expert_fused_kernel(*expert_fused_args)
    expert_down_args = (
        expert_activation,
        tensors["expert_down_weight"],
        topk_ids,
    )
    expert_down_kernel = build(f"{prefix}_expert_down", expert_down, expert_down_args)
    expert_outputs = expert_down_kernel(*expert_down_args)
    expert_reduce_args = (expert_outputs, topk_weights)
    expert_reduce_kernel = build(
        f"{prefix}_expert_reduce", weighted_expert_reduce, expert_reduce_args
    )
    moe_down = expert_reduce_kernel(*expert_reduce_args)
    expert_down_reduce_args = (
        expert_activation,
        tensors["expert_down_weight"],
        topk_ids,
        topk_weights,
    )
    expert_down_reduce_kernel = build(
        f"{prefix}_expert_down_reduce",
        expert_down_reduce,
        expert_down_reduce_args,
    )
    expert_down_reduce_kernel(*expert_down_reduce_args)
    moe_post_args = (
        moe_down,
        tensors["post_ff_norm_weight_2"],
        shape.eps,
    )
    moe_post_kernel = build(f"{prefix}_moe_post_norm", rms_norm, moe_post_args)
    moe_branch = moe_post_kernel(*moe_post_args)
    branch_add_args = (dense_branch, moe_branch)
    branch_add_kernel = build(f"{prefix}_branch_add", add_branches, branch_add_args)
    combined = branch_add_kernel(*branch_add_args)
    post_ff_args = (
        combined,
        tensors["post_ff_norm_weight"],
        shape.eps,
    )
    post_ff_kernel = build(f"{prefix}_post_ff_norm", rms_norm, post_ff_args)
    post_ff = post_ff_kernel(*post_ff_args)
    residual_scale_args = (post_ff, residual, tensors["layer_scalar"])
    residual_scale_kernel = build(
        f"{prefix}_residual_scale", residual_scale, residual_scale_args
    )
    residual_scale_kernel(*residual_scale_args)
    final_fused_args = (
        dense_down,
        moe_down,
        tensors["post_ff_norm_weight_1"],
        tensors["post_ff_norm_weight_2"],
        tensors["post_ff_norm_weight"],
        residual,
        tensors["layer_scalar"],
        shape.eps,
    )
    final_fused_kernel = build(
        f"{prefix}_final_dense_moe_residual",
        final_dense_moe_residual,
        final_fused_args,
    )
    final_fused_kernel(*final_fused_args)
    torch.cuda.synchronize()

    def launch_attention(local_query):
        if args.optimized_direct_attention:
            assert direct_attention_kernel is not None
            assert direct_attention_args is not None
            return direct_attention_kernel(local_query, *direct_attention_args[1:])
        local_partials, local_lse = attention_kernel(local_query, *attention_args[1:])
        return merge_kernel(local_partials, local_lse)

    def launch_matched():
        local_input_norm = input_norm_kernel(*input_norm_args)
        local_qkv = qkv_mm_kernel(local_input_norm, tensors["qkv_weight"])
        norm_kernel(local_qkv, *norm_args[1:])
        local_query = local_qkv[:, :q_width].view(1, shape.q_heads, geometry.head_dim)
        local_key = local_qkv[:, q_width : q_width + kv_width].view(
            1, geometry.kv_heads, geometry.head_dim
        )
        local_value = local_qkv[:, q_width + kv_width :].view(
            1, geometry.kv_heads, geometry.head_dim
        )
        cache_kernel(
            local_key,
            local_value,
            tensors["kv_cache"],
            tensors["slot_mapping"],
            shape.block_size,
        )
        local_partials, local_lse = attention_kernel(local_query, *attention_args[1:])
        local_attention = merge_kernel(local_partials, local_lse)
        local_attention_out = o_kernel(
            local_attention.view(1, q_width), tensors["o_weight"]
        )
        local_residual, local_dense_input = post_attention_kernel(
            local_attention_out, *post_attention_args[1:]
        )
        local_dense_gate_up = dense_gate_up_kernel(
            local_dense_input, tensors["gate_up_weight"]
        )
        local_dense_activation = dense_geglu_kernel(local_dense_gate_up)
        local_dense_down = dense_down_kernel(
            local_dense_activation, tensors["down_weight"]
        )
        local_dense_branch = dense_post_kernel(
            local_dense_down, tensors["post_ff_norm_weight_1"], shape.eps
        )
        local_expert_input = expert_input_kernel(
            local_residual, tensors["pre_ff_norm_weight_2"], shape.eps
        )
        local_router_hidden = router_norm_kernel(
            local_residual, tensors["router_scale"], root_size, shape.eps
        )
        local_logits = router_mm_kernel(local_router_hidden, tensors["router_weight"])
        local_weights, local_ids = route_kernel(
            local_logits, tensors["per_expert_scale"], shape.top_k
        )
        local_expert_gate_up = expert_gate_up_kernel(
            local_expert_input, tensors["expert_gate_up_weight"], local_ids
        )
        local_expert_activation = expert_geglu_kernel(local_expert_gate_up)
        local_expert_outputs = expert_down_kernel(
            local_expert_activation, tensors["expert_down_weight"], local_ids
        )
        local_moe_down = expert_reduce_kernel(local_expert_outputs, local_weights)
        local_moe_branch = moe_post_kernel(
            local_moe_down, tensors["post_ff_norm_weight_2"], shape.eps
        )
        local_combined = branch_add_kernel(local_dense_branch, local_moe_branch)
        local_post_ff = post_ff_kernel(
            local_combined, tensors["post_ff_norm_weight"], shape.eps
        )
        return residual_scale_kernel(
            local_post_ff, local_residual, tensors["layer_scalar"]
        )

    def launch_optimized():
        if geometry.k_eq_v and args.elide_duplicate_v_projection:
            local_input_norm = input_norm_kernel(*input_norm_args)
            assert qk_mm_kernel is not None
            assert qk_norm_cache_kernel is not None
            local_qk = qk_mm_kernel(local_input_norm, tensors["qk_weight"])
            qk_norm_cache_kernel(local_qk, *qk_norm_cache_args[1:])
            local_query = local_qk[:, :q_width].view(
                1, shape.q_heads, geometry.head_dim
            )
        else:
            local_qkv = fused_qkv_kernel(*fused_qkv_args)
            fused_norm_cache_kernel(local_qkv, *fused_norm_cache_args[1:])
            local_query = local_qkv[:, :q_width].view(
                1, shape.q_heads, geometry.head_dim
            )
        local_attention = launch_attention(local_query)
        local_attention_out = o_kernel(
            local_attention.view(1, q_width), tensors["o_weight"]
        )
        local_residual, local_dense_input = post_attention_kernel(
            local_attention_out, *post_attention_args[1:]
        )
        local_dense_activation = dense_fused_kernel(
            local_dense_input, tensors["gate_up_weight"]
        )
        local_dense_down = dense_down_kernel(
            local_dense_activation, tensors["down_weight"]
        )
        local_weights, local_ids = router_fused_kernel(
            local_residual,
            tensors["router_scale"],
            root_size,
            tensors["router_weight"],
            tensors["per_expert_scale"],
            shape.top_k,
            shape.eps,
        )
        local_expert_input = expert_input_kernel(
            local_residual, tensors["pre_ff_norm_weight_2"], shape.eps
        )
        local_expert_activation = expert_fused_kernel(
            local_expert_input,
            tensors["expert_gate_up_weight"],
            local_ids,
        )
        local_moe_down = expert_down_reduce_kernel(
            local_expert_activation,
            tensors["expert_down_weight"],
            local_ids,
            local_weights,
        )
        return final_fused_kernel(
            local_dense_down,
            local_moe_down,
            tensors["post_ff_norm_weight_1"],
            tensors["post_ff_norm_weight_2"],
            tensors["post_ff_norm_weight"],
            local_residual,
            tensors["layer_scalar"],
            shape.eps,
        )

    return {
        "launch_matched": launch_matched,
        "launch_optimized": launch_optimized,
        "configs": selected_configs,
        "stage_outputs": {
            "query": query,
            "key": key,
            "value": value,
            "attention": attention,
            "direct_attention": direct_attention,
            "residual": residual,
            "dense_down": dense_down,
            "router_logits": router_logits,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "expert_gate_up": expert_gate_up_output,
            "expert_outputs": expert_outputs,
            "moe_down": moe_down,
        },
        "stage_calls": {
            f"{prefix}_qkv_mm": lambda: qkv_mm_kernel(*qkv_mm_args),
            f"{prefix}_attention_s{splits}": lambda: attention_kernel(*attention_args),
            f"{prefix}_attention_merge_s{splits}": lambda: merge_kernel(*merge_args),
            **(
                {
                    f"{prefix}_attention_direct": lambda: direct_attention_kernel(
                        *direct_attention_args
                    )
                }
                if direct_attention_kernel is not None
                else {}
            ),
            f"{prefix}_dense_gate_up_mm": lambda: dense_gate_up_kernel(
                *dense_gate_up_args
            ),
            f"{prefix}_dense_geglu_projection": lambda: dense_fused_kernel(
                *dense_fused_args
            ),
            f"{prefix}_dense_down_mm": lambda: dense_down_kernel(*dense_down_args),
            f"{prefix}_router_mm_fp32": lambda: router_mm_kernel(*router_mm_args),
            f"{prefix}_route_topk": lambda: route_kernel(*route_args),
            f"{prefix}_router_projection_topk": lambda: router_fused_kernel(
                *router_fused_args
            ),
            f"{prefix}_expert_gate_up": lambda: expert_gate_up_kernel(
                *expert_gate_up_args
            ),
            f"{prefix}_expert_geglu_projection": lambda: expert_fused_kernel(
                *expert_fused_args
            ),
            f"{prefix}_expert_down": lambda: expert_down_kernel(*expert_down_args),
            f"{prefix}_expert_down_reduce": lambda: expert_down_reduce_kernel(
                *expert_down_reduce_args
            ),
            f"{prefix}_final_dense_moe_residual": lambda: final_fused_kernel(
                *final_fused_args
            ),
        },
    }


def run_layer(args, layer_idx, configs, config_path):
    configured_shape = Gemma4A4BShape(context=args.context, block_size=args.block_size)
    geometry = configured_shape.layer_geometry(layer_idx)
    effective_block_size = (
        args.block_size
        if args.disable_hybrid_page_promotion
        else configured_shape.effective_block_size(geometry)
    )
    shape = Gemma4A4BShape(context=args.context, block_size=effective_block_size)
    geometry = shape.layer_geometry(layer_idx)
    tensors = allocate_layer(shape, geometry, args.seed)
    reference = layer_reference(tensors, shape, geometry)
    built = build_layer(args, tensors, shape, geometry, configs, config_path)
    stages = built["stage_outputs"]
    _assert_close("query", stages["query"], reference["query"])
    _assert_close("key", stages["key"], reference["key"])
    _assert_close("value", stages["value"], reference["value"])
    _assert_close(
        "attention", stages["attention"], reference["attention"], atol=0.15, rtol=0.06
    )
    _assert_close(
        "direct_attention",
        stages["direct_attention"],
        reference["attention"],
        atol=0.15,
        rtol=0.06,
    )
    _assert_close("residual", stages["residual"], reference["residual"], atol=0.15)
    _assert_close(
        "router_logits", stages["router_logits"], reference["router_logits"], atol=0.1
    )
    torch.testing.assert_close(stages["topk_ids"], reference["topk_ids"])
    _assert_close(
        "topk_weights", stages["topk_weights"], reference["topk_weights"], atol=2e-3
    )
    _assert_close(
        "expert_gate_up",
        stages["expert_gate_up"],
        reference["expert_gate_up"],
        atol=0.2,
        rtol=0.08,
    )
    _assert_close(
        "moe_down", stages["moe_down"], reference["moe_down"], atol=0.25, rtol=0.1
    )

    matched_eager = built["launch_matched"]()
    optimized_eager = built["launch_optimized"]()
    torch.cuda.synchronize()
    _assert_close(
        "matched_eager_output",
        matched_eager,
        reference["output"],
        atol=0.3,
        rtol=0.1,
    )
    _assert_close(
        "optimized_eager_output",
        optimized_eager,
        reference["output"],
        atol=0.3,
        rtol=0.1,
    )
    result = {
        "layer_idx": layer_idx,
        "variant": variant_name(geometry),
        "head_dim": geometry.head_dim,
        "kv_heads": geometry.kv_heads,
        "k_eq_v": geometry.k_eq_v,
        "context": shape.context,
        "configured_block_size": args.block_size,
        "effective_kernel_block_size": shape.block_size,
        "attention_context": geometry.attention_context,
        "benchmark_mode": benchmark_cache_mode(),
        "splits": args.full_splits
        if geometry.layer_type == "full"
        else args.sliding_splits,
    }
    if args.include_configs:
        result["configs"] = built["configs"]
    if args.smoke and not args.benchmark:
        result["status"] = "smoke_ok"
        return result

    matched_graph, matched_output = capture(built["launch_matched"])
    matched_graph.replay()
    optimized_graph, optimized_output = capture(built["launch_optimized"])
    optimized_graph.replay()
    torch.cuda.synchronize()
    _assert_close(
        "matched_graph_output",
        matched_output,
        reference["output"],
        atol=0.3,
        rtol=0.1,
    )
    _assert_close(
        "optimized_graph_output",
        optimized_output,
        reference["output"],
        atol=0.3,
        rtol=0.1,
    )
    benchmark_pids = visible_gpu_pids()
    name = variant_name(geometry)
    timings = benchmark_interleaved(
        {
            f"helion_a4b_{name}_matched": matched_graph.replay,
            f"helion_a4b_{name}_optimized": optimized_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
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
    layers = A4B_REPRESENTATIVE_LAYERS if args.all_variants else (args.layer,)
    results = [run_layer(args, layer_idx, configs, config_path) for layer_idx in layers]
    if args.all_variants and all("timings" in result for result in results):
        weighted = {}
        for path in ("matched", "optimized"):
            total = 0.0
            for result in results:
                key = f"helion_a4b_{result['variant']}_{path}"
                total += (
                    result["timings"][key]["median_us"]
                    * A4B_LAYER_COUNTS[result["variant"]]
                )
            weighted[f"helion_a4b_30_layer_{path}_us"] = total
        summary = {"layers": results, "weighted_model_layer_sum": weighted}
    else:
        summary = {"layers": results}
    print(
        "RESULT_JSON",
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "helion_module": helion.__file__,
                **summary,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--disable-hybrid-page-promotion", action="store_true")
    parser.add_argument("--sliding-splits", type=int, default=16)
    parser.add_argument("--full-splits", type=int, default=64)
    parser.add_argument("--optimized-direct-attention", action="store_true")
    parser.add_argument(
        "--no-elide-duplicate-v-projection",
        dest="elide_duplicate_v_projection",
        action="store_false",
    )
    parser.set_defaults(elide_duplicate_v_projection=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch-replays", type=int, default=20)
    parser.add_argument(
        "--config-path",
        default=str(Path(__file__).with_name("gemma4_26b_a4b_b200_configs.json")),
    )
    parser.add_argument("--tune", nargs="*", default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-stages", action="store_true")
    parser.add_argument("--include-configs", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
