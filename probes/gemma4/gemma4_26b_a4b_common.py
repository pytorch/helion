"""Shared data and references for the Gemma 4 26B-A4B layer benchmarks.

This module is deliberately separate from the E4B benchmark.  It models the
TP=1 decode geometry of ``google/gemma-4-26B-A4B``: 25 sliding layers and five
global layers, with a dense MLP and a parallel top-8-of-128 MoE in every layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import statistics
import subprocess
from typing import Callable

import torch

from probes.common import make_l2_cache_clearer


@dataclass(frozen=True)
class LayerGeometry:
    layer_idx: int
    layer_type: str
    head_dim: int
    rotary_dim: int
    rope_theta: float
    kv_heads: int
    attention_context: int
    k_eq_v: bool


@dataclass(frozen=True)
class Gemma4A4BShape:
    """Production Gemma 4 26B-A4B TP=1 text-layer geometry."""

    hidden: int = 2816
    intermediate: int = 2112
    moe_intermediate: int = 704
    num_experts: int = 128
    top_k: int = 8
    q_heads: int = 16
    sliding_kv_heads: int = 8
    full_kv_heads: int = 2
    context: int = 8192
    block_size: int = 16
    sliding_window: int = 1024
    eps: float = 1e-6

    def layer_geometry(self, layer_idx: int) -> LayerGeometry:
        if not 0 <= layer_idx < 30:
            raise ValueError(
                f"Gemma 4 26B-A4B layer index must be in [0, 30), got {layer_idx}"
            )
        is_full = layer_idx % 6 == 5
        head_dim = 512 if is_full else 256
        return LayerGeometry(
            layer_idx=layer_idx,
            layer_type="full" if is_full else "sliding",
            head_dim=head_dim,
            rotary_dim=head_dim // 4 if is_full else head_dim,
            rope_theta=1_000_000.0 if is_full else 10_000.0,
            kv_heads=self.full_kv_heads if is_full else self.sliding_kv_heads,
            attention_context=self.context
            if is_full
            else min(self.context, self.sliding_window),
            k_eq_v=is_full,
        )

    def effective_block_size(self, geometry: LayerGeometry) -> int:
        """Mirror vLLM hybrid KV-page byte equalization."""
        bytes_per_token = geometry.kv_heads * geometry.head_dim * 2
        max_bytes_per_token = self.sliding_kv_heads * 256 * 2
        if max_bytes_per_token % bytes_per_token:
            raise ValueError("hybrid KV page sizes are not integral")
        return self.block_size * max_bytes_per_token // bytes_per_token


A4B_REPRESENTATIVE_LAYERS = (0, 5)
A4B_LAYER_COUNTS = {"sliding": 25, "full": 5}


def variant_name(geometry: LayerGeometry) -> str:
    return geometry.layer_type


def make_cos_sin(
    max_position: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    device: str,
) -> torch.Tensor:
    """Build vLLM's Gemma4 proportional/default RoPE cache exactly."""
    half = head_dim // 2
    rope_angles = rotary_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
    )
    if rope_angles < half:
        inv_freq = torch.cat(
            (inv_freq, torch.zeros(half - rope_angles, dtype=torch.float32))
        )
    freqs = torch.outer(torch.arange(max_position, dtype=torch.float32), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(
        device=device, dtype=torch.bfloat16
    )


def _linear_weight(rows: int, columns: int, device: str) -> torch.Tensor:
    weight = torch.empty((rows, columns), device=device, dtype=torch.bfloat16)
    return weight.normal_(std=0.5 / math.sqrt(columns))


def allocate_layer(
    shape: Gemma4A4BShape,
    geometry: LayerGeometry,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Allocate deterministic tensors for one steady-state decode layer."""
    torch.manual_seed(seed + geometry.layer_idx)
    device = "cuda"
    q_width = shape.q_heads * geometry.head_dim
    kv_width = geometry.kv_heads * geometry.head_dim
    logical_blocks = math.ceil(shape.context / shape.block_size)
    physical_blocks = math.ceil(logical_blocks * 1.25)
    block_table = torch.randperm(physical_blocks, device=device, dtype=torch.int64)[
        :logical_blocks
    ].to(torch.int32)[None]
    final_logical_block = (shape.context - 1) // shape.block_size
    final_offset = (shape.context - 1) % shape.block_size
    final_physical_block = int(block_table[0, final_logical_block].item())

    def norm_weight(size: int) -> torch.Tensor:
        weight = torch.empty((size,), device=device, dtype=torch.bfloat16)
        return weight.normal_(mean=1.0, std=0.05)

    q_weight = _linear_weight(q_width, shape.hidden, device)
    k_weight = _linear_weight(kv_width, shape.hidden, device)
    if geometry.k_eq_v:
        qkv_weight = torch.cat((q_weight, k_weight, k_weight.clone()), dim=0)
    else:
        v_weight = _linear_weight(kv_width, shape.hidden, device)
        qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)

    expert_gate_up = _linear_weight(
        shape.num_experts * 2 * shape.moe_intermediate,
        shape.hidden,
        device,
    ).view(shape.num_experts, 2 * shape.moe_intermediate, shape.hidden)
    expert_down = _linear_weight(
        shape.num_experts * shape.hidden,
        shape.moe_intermediate,
        device,
    ).view(shape.num_experts, shape.hidden, shape.moe_intermediate)
    hidden_states = torch.randn((1, shape.hidden), device=device, dtype=torch.bfloat16)
    router_weight = _linear_weight(shape.num_experts, shape.hidden, device)
    # Add a small rank-one component so the synthetic token has non-degenerate
    # top-k gaps. This makes routing-boundary checks robust to legitimate GEMM
    # accumulation-order differences without materially changing weight scale.
    direction = hidden_states.float()
    direction = direction / torch.linalg.vector_norm(direction)
    routing_offsets = torch.linspace(
        -0.25,
        0.25,
        shape.num_experts,
        device=device,
        dtype=torch.float32,
    )
    router_weight.add_((routing_offsets[:, None] * direction).to(torch.bfloat16))

    return {
        "hidden_states": hidden_states,
        "input_norm_weight": norm_weight(shape.hidden),
        "post_attention_norm_weight": norm_weight(shape.hidden),
        "pre_ff_norm_weight": norm_weight(shape.hidden),
        "post_ff_norm_weight": norm_weight(shape.hidden),
        "post_ff_norm_weight_1": norm_weight(shape.hidden),
        "pre_ff_norm_weight_2": norm_weight(shape.hidden),
        "post_ff_norm_weight_2": norm_weight(shape.hidden),
        "q_norm_weight": norm_weight(geometry.head_dim),
        "k_norm_weight": norm_weight(geometry.head_dim),
        "qkv_weight": qkv_weight,
        "qk_weight": torch.cat((q_weight, k_weight), dim=0),
        "o_weight": _linear_weight(shape.hidden, q_width, device),
        "gate_up_weight": _linear_weight(2 * shape.intermediate, shape.hidden, device),
        "down_weight": _linear_weight(shape.hidden, shape.intermediate, device),
        "router_scale": norm_weight(shape.hidden),
        "router_weight": router_weight,
        "per_expert_scale": norm_weight(shape.num_experts),
        "expert_gate_up_weight": expert_gate_up,
        "expert_down_weight": expert_down,
        "layer_scalar": torch.tensor(0.95, device=device, dtype=torch.bfloat16),
        "position": torch.tensor([shape.context - 1], device=device, dtype=torch.int64),
        "cos_sin": make_cos_sin(
            max(shape.context, 4096),
            geometry.head_dim,
            geometry.rotary_dim,
            geometry.rope_theta,
            device,
        ),
        "kv_cache": torch.randn(
            (
                physical_blocks,
                shape.block_size,
                geometry.kv_heads,
                2 * geometry.head_dim,
            ),
            device=device,
            dtype=torch.bfloat16,
        ),
        "block_table": block_table,
        "slot_mapping": torch.tensor(
            [final_physical_block * shape.block_size + final_offset],
            device=device,
            dtype=torch.int64,
        ),
    }


def rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    values = x.float()
    values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    values = values.to(x.dtype)
    if weight is not None:
        values = values * weight
    return values


def apply_neox_rope_reference(
    x: torch.Tensor,
    cos_sin: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    rotary_dim = cos_sin.shape[-1]
    half = rotary_dim // 2
    cache = cos_sin[position]
    cos = cache[..., :half][:, None, :]
    sin = cache[..., half:][:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:rotary_dim]
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


def update_cache_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    slot = int(slot_mapping[0].item())
    block = slot // block_size
    offset = slot % block_size
    head_dim = key.shape[-1]
    kv_cache[block, offset, :, :head_dim].copy_(key[0])
    kv_cache[block, offset, :, head_dim:].copy_(value[0])


def paged_attention_reference(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    attention_context: int,
    block_size: int,
    q_per_kv: int,
) -> torch.Tensor:
    start = context - attention_context
    logical_indices = torch.arange(start, context, device=query.device)
    physical_blocks = block_table[0, logical_indices // block_size].long()
    offsets = logical_indices % block_size
    logical = kv_cache[physical_blocks, offsets]
    head_dim = query.shape[-1]
    key = logical[..., :head_dim].permute(1, 0, 2).float()
    value = logical[..., head_dim:].permute(1, 0, 2).float()
    key = key.repeat_interleave(q_per_kv, dim=0)
    value = value.repeat_interleave(q_per_kv, dim=0)
    scores = torch.einsum("hqd,hkd->hqk", query[0, :, None].float(), key)
    probabilities = torch.softmax(scores, dim=-1)
    return (
        torch.einsum("hqk,hkd->hqd", probabilities, value)
        .squeeze(1)[None]
        .to(torch.bfloat16)
    )


def routing_reference(
    logits: torch.Tensor,
    top_k: int,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    topk_weights = probabilities.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * per_expert_scale[topk_ids].float()
    return topk_weights.float(), topk_ids.to(torch.int32)


def layer_reference(
    tensors: dict[str, torch.Tensor],
    shape: Gemma4A4BShape,
    geometry: LayerGeometry,
) -> dict[str, torch.Tensor]:
    """Torch reference with vLLM's materialization and routing boundaries."""
    hidden = tensors["hidden_states"]
    input_norm = rms_norm_reference(hidden, tensors["input_norm_weight"], shape.eps)
    qkv = torch.nn.functional.linear(input_norm, tensors["qkv_weight"])
    q_width = shape.q_heads * geometry.head_dim
    kv_width = geometry.kv_heads * geometry.head_dim
    q = qkv[:, :q_width].view(1, shape.q_heads, geometry.head_dim)
    k = qkv[:, q_width : q_width + kv_width].view(
        1, geometry.kv_heads, geometry.head_dim
    )
    v = qkv[:, q_width + kv_width :].view(1, geometry.kv_heads, geometry.head_dim)
    q = rms_norm_reference(q, tensors["q_norm_weight"], shape.eps)
    k = rms_norm_reference(k, tensors["k_norm_weight"], shape.eps)
    v = rms_norm_reference(v, None, shape.eps)
    q = apply_neox_rope_reference(q, tensors["cos_sin"], tensors["position"])
    k = apply_neox_rope_reference(k, tensors["cos_sin"], tensors["position"])
    cache = tensors["kv_cache"].clone()
    update_cache_reference(k, v, cache, tensors["slot_mapping"], shape.block_size)
    attention = paged_attention_reference(
        q,
        cache,
        tensors["block_table"],
        shape.context,
        geometry.attention_context,
        shape.block_size,
        shape.q_heads // geometry.kv_heads,
    )
    attention_out = torch.nn.functional.linear(
        attention.view(1, q_width), tensors["o_weight"]
    )
    post_attention = rms_norm_reference(
        attention_out, tensors["post_attention_norm_weight"], shape.eps
    )
    residual = post_attention + hidden

    dense_input = rms_norm_reference(residual, tensors["pre_ff_norm_weight"], shape.eps)
    dense_gate_up = torch.nn.functional.linear(dense_input, tensors["gate_up_weight"])
    dense_gate, dense_up = dense_gate_up.chunk(2, dim=-1)
    dense_activation = (
        torch.nn.functional.gelu(dense_gate, approximate="tanh") * dense_up
    )
    dense_down = torch.nn.functional.linear(dense_activation, tensors["down_weight"])
    dense_branch = rms_norm_reference(
        dense_down, tensors["post_ff_norm_weight_1"], shape.eps
    )

    expert_input = rms_norm_reference(
        residual, tensors["pre_ff_norm_weight_2"], shape.eps
    )
    router_hidden = rms_norm_reference(residual, None, shape.eps)
    root_size = torch.tensor(
        shape.hidden**-0.5, device=hidden.device, dtype=hidden.dtype
    )
    router_hidden = router_hidden * root_size
    router_hidden = router_hidden * tensors["router_scale"]
    router_logits = torch.mm(
        router_hidden,
        tensors["router_weight"].T,
        out_dtype=torch.float32,
    )
    topk_weights, topk_ids = routing_reference(
        router_logits, shape.top_k, tensors["per_expert_scale"]
    )
    expert_ids = topk_ids[0].long()
    selected_gate_up = tensors["expert_gate_up_weight"][expert_ids]
    expert_gate_up = torch.bmm(
        selected_gate_up,
        expert_input.expand(shape.top_k, -1).unsqueeze(-1),
    ).squeeze(-1)
    expert_gate, expert_up = expert_gate_up.chunk(2, dim=-1)
    expert_activation = (
        torch.nn.functional.gelu(expert_gate, approximate="tanh") * expert_up
    )
    selected_down = tensors["expert_down_weight"][expert_ids]
    expert_outputs = torch.bmm(selected_down, expert_activation.unsqueeze(-1)).squeeze(
        -1
    )
    moe_down = (
        (expert_outputs.float() * topk_weights[0, :, None])
        .sum(dim=0, keepdim=True)
        .to(torch.bfloat16)
    )
    moe_branch = rms_norm_reference(
        moe_down, tensors["post_ff_norm_weight_2"], shape.eps
    )

    combined = dense_branch + moe_branch
    post_ff = rms_norm_reference(combined, tensors["post_ff_norm_weight"], shape.eps)
    output = (post_ff + residual) * tensors["layer_scalar"]
    return {
        "input_norm": input_norm,
        "qkv": qkv,
        "query": q,
        "key": k,
        "value": v,
        "kv_cache": cache,
        "attention": attention,
        "attention_out": attention_out,
        "residual": residual,
        "dense_input": dense_input,
        "dense_gate_up": dense_gate_up,
        "dense_activation": dense_activation,
        "dense_down": dense_down,
        "dense_branch": dense_branch,
        "expert_input": expert_input,
        "router_hidden": router_hidden,
        "router_logits": router_logits,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "expert_gate_up": expert_gate_up,
        "expert_activation": expert_activation,
        "expert_outputs": expert_outputs,
        "moe_down": moe_down,
        "moe_branch": moe_branch,
        "combined": combined,
        "output": output,
    }


def capture(
    fn: Callable[[], torch.Tensor],
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            output = fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output = fn()
    torch.cuda.synchronize()
    return graph, output


def benchmark_interleaved(
    entries: dict[str, Callable[[], object]],
    repeats: int,
    batch_replays: int,
) -> dict[str, dict[str, float]]:
    samples = {name: [] for name in entries}
    names = list(entries)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    clear_l2 = make_l2_cache_clearer()
    for sample in range(repeats):
        order = names[sample % len(names) :] + names[: sample % len(names)]
        for name in order:
            if clear_l2 is None:
                start.record()
                for _ in range(batch_replays):
                    entries[name]()
                end.record()
                end.synchronize()
                elapsed_us = start.elapsed_time(end) * 1000.0 / batch_replays
            else:
                elapsed_us = 0.0
                for _ in range(batch_replays):
                    clear_l2()
                    torch.cuda.synchronize()
                    start.record()
                    entries[name]()
                    end.record()
                    end.synchronize()
                    elapsed_us += start.elapsed_time(end) * 1000.0
                elapsed_us /= batch_replays
            samples[name].append(elapsed_us)
    return {
        name: {
            "median_us": statistics.median(values),
            "mean_us": statistics.fmean(values),
            "p90_us": sorted(values)[min(len(values) - 1, int(0.9 * len(values)))],
        }
        for name, values in samples.items()
    }


def visible_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if "," in visible:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one idle GPU")
    return visible


def visible_gpu_pids() -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            visible_gpu(),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}


def require_idle_visible_gpu() -> None:
    visible = visible_gpu()
    memory_limit = int(os.environ.get("MEGAKERNEL_IDLE_MEMORY_LIMIT_MB", "256"))
    pids = visible_gpu_pids()
    if pids:
        raise RuntimeError(f"GPU {visible} has compute processes {sorted(pids)}")
    state = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            visible,
            "--query-gpu=utilization.gpu,utilization.memory,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gpu_util, memory_util, memory_used = (
        int(field.strip()) for field in state.split(",")
    )
    if gpu_util != 0 or memory_util != 0 or memory_used > memory_limit:
        raise RuntimeError(f"GPU {visible} is not idle: {state}")
