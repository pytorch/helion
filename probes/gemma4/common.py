"""Shared helpers for the Gemma 4 E4B decode-layer benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import statistics
import subprocess
from typing import Callable

import torch


@dataclass(frozen=True)
class Gemma4E4BShape:
    """Production Gemma 4 E4B TP=1 text-layer geometry."""

    hidden: int = 2560
    intermediate: int = 10240
    q_heads: int = 8
    kv_heads: int = 2
    ple: int = 256
    context: int = 8192
    block_size: int = 16
    sliding_window: int = 512
    eps: float = 1e-6

    @property
    def first_kv_shared_layer(self) -> int:
        return 24

    def layer_geometry(self, layer_idx: int) -> LayerGeometry:
        if not 0 <= layer_idx < 42:
            raise ValueError(
                f"Gemma 4 E4B layer index must be in [0, 42), got {layer_idx}"
            )
        is_full = layer_idx % 6 == 5
        head_dim = 512 if is_full else 256
        return LayerGeometry(
            layer_idx=layer_idx,
            layer_type="full" if is_full else "sliding",
            kv_shared=layer_idx >= self.first_kv_shared_layer,
            head_dim=head_dim,
            rotary_dim=head_dim // 4 if is_full else head_dim,
            rope_theta=1_000_000.0 if is_full else 10_000.0,
            attention_context=self.context
            if is_full
            else min(self.context, self.sliding_window),
        )


@dataclass(frozen=True)
class LayerGeometry:
    layer_idx: int
    layer_type: str
    kv_shared: bool
    head_dim: int
    rotary_dim: int
    rope_theta: float
    attention_context: int


E4B_REPRESENTATIVE_LAYERS = (0, 5, 24, 29)
E4B_LAYER_COUNTS = {
    "sliding_nonshared": 20,
    "full_nonshared": 4,
    "sliding_shared": 15,
    "full_shared": 3,
}


def variant_name(geometry: LayerGeometry) -> str:
    suffix = "shared" if geometry.kv_shared else "nonshared"
    return f"{geometry.layer_type}_{suffix}"


def make_cos_sin(
    max_position: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    device: str,
) -> torch.Tensor:
    """Build the exact vLLM Gemma4 RoPE cache.

    Gemma4 proportional RoPE uses ``head_dim`` in the frequency denominator
    and identity-pads inactive angle pairs. Standard sliding RoPE is the
    special case where ``rotary_dim == head_dim``.
    """
    half = head_dim // 2
    rope_angles = rotary_dim // 2
    # vLLM constructs this cache on CPU before moving it to the accelerator.
    # Keeping the same rounding path avoids small phase differences at long
    # positions.
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
    )
    if rope_angles < half:
        inv_freq = torch.cat(
            (
                inv_freq,
                torch.zeros(half - rope_angles, dtype=torch.float32),
            )
        )
    freqs = torch.outer(torch.arange(max_position, dtype=torch.float32), inv_freq)
    assert freqs.shape[-1] == half
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(
        device=device, dtype=torch.bfloat16
    )


def _linear_weight(rows: int, columns: int, device: str) -> torch.Tensor:
    return torch.randn((rows, columns), device=device, dtype=torch.bfloat16) * (
        0.5 / math.sqrt(columns)
    )


def allocate_layer(
    shape: Gemma4E4BShape,
    geometry: LayerGeometry,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Allocate deterministic tensors for one steady-state decode layer."""
    torch.manual_seed(seed + geometry.layer_idx)
    device = "cuda"
    q_width = shape.q_heads * geometry.head_dim
    kv_width = shape.kv_heads * geometry.head_dim
    qkv_width = q_width + 2 * kv_width
    logical_blocks = math.ceil(shape.context / shape.block_size)
    physical_blocks = math.ceil(logical_blocks * 1.25)
    block_table = torch.randperm(physical_blocks, device=device, dtype=torch.int64)[
        :logical_blocks
    ].to(torch.int32)[None]
    final_logical_block = (shape.context - 1) // shape.block_size
    final_offset = (shape.context - 1) % shape.block_size
    final_physical_block = int(block_table[0, final_logical_block].item())

    def norm_weight(size: int) -> torch.Tensor:
        return torch.randn((size,), device=device, dtype=torch.bfloat16) * 0.05 + 1.0

    return {
        "hidden_states": torch.randn(
            (1, shape.hidden), device=device, dtype=torch.bfloat16
        ),
        "per_layer_input": torch.randn(
            (1, shape.ple), device=device, dtype=torch.bfloat16
        ),
        "input_norm_weight": norm_weight(shape.hidden),
        "post_attention_norm_weight": norm_weight(shape.hidden),
        "pre_ff_norm_weight": norm_weight(shape.hidden),
        "post_ff_norm_weight": norm_weight(shape.hidden),
        "post_ple_norm_weight": norm_weight(shape.hidden),
        "q_norm_weight": norm_weight(geometry.head_dim),
        "k_norm_weight": norm_weight(geometry.head_dim),
        "qkv_weight": _linear_weight(qkv_width, shape.hidden, device),
        "o_weight": _linear_weight(shape.hidden, q_width, device),
        "gate_up_weight": _linear_weight(2 * shape.intermediate, shape.hidden, device),
        "down_weight": _linear_weight(shape.hidden, shape.intermediate, device),
        "ple_gate_weight": _linear_weight(shape.ple, shape.hidden, device),
        "ple_proj_weight": _linear_weight(shape.hidden, shape.ple, device),
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
                shape.kv_heads,
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
    rotary = x[..., :rotary_dim]
    x1 = rotary[..., :half]
    x2 = rotary[..., half:]
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


def layer_reference(
    tensors: dict[str, torch.Tensor],
    shape: Gemma4E4BShape,
    geometry: LayerGeometry,
) -> dict[str, torch.Tensor]:
    """Torch reference with the same BF16 materialization boundaries as vLLM."""
    hidden = tensors["hidden_states"]
    input_norm = rms_norm_reference(hidden, tensors["input_norm_weight"], shape.eps)
    qkv = torch.nn.functional.linear(input_norm, tensors["qkv_weight"])
    q_width = shape.q_heads * geometry.head_dim
    kv_width = shape.kv_heads * geometry.head_dim
    q = qkv[:, :q_width].view(1, shape.q_heads, geometry.head_dim)
    k = qkv[:, q_width : q_width + kv_width].view(1, shape.kv_heads, geometry.head_dim)
    v = qkv[:, q_width + kv_width :].view(1, shape.kv_heads, geometry.head_dim)
    q = rms_norm_reference(q, tensors["q_norm_weight"], shape.eps)
    q = apply_neox_rope_reference(q, tensors["cos_sin"], tensors["position"])
    cache = tensors["kv_cache"].clone()
    if not geometry.kv_shared:
        k = rms_norm_reference(k, tensors["k_norm_weight"], shape.eps)
        k = apply_neox_rope_reference(k, tensors["cos_sin"], tensors["position"])
        v = rms_norm_reference(v, None, shape.eps)
        update_cache_reference(k, v, cache, tensors["slot_mapping"], shape.block_size)
    attention = paged_attention_reference(
        q,
        cache,
        tensors["block_table"],
        shape.context,
        geometry.attention_context,
        shape.block_size,
        shape.q_heads // shape.kv_heads,
    )
    attention_out = torch.nn.functional.linear(
        attention.view(1, q_width), tensors["o_weight"]
    )
    post_attention = rms_norm_reference(
        attention_out, tensors["post_attention_norm_weight"], shape.eps
    )
    residual = post_attention + hidden
    ff_input = rms_norm_reference(residual, tensors["pre_ff_norm_weight"], shape.eps)
    gate_up = torch.nn.functional.linear(ff_input, tensors["gate_up_weight"])
    gate, up = gate_up.chunk(2, dim=-1)
    activation = torch.nn.functional.gelu(gate, approximate="tanh") * up
    down = torch.nn.functional.linear(activation, tensors["down_weight"])
    post_ff = rms_norm_reference(down, tensors["post_ff_norm_weight"], shape.eps)
    hidden = post_ff + residual
    ple_gate = torch.nn.functional.linear(hidden, tensors["ple_gate_weight"])
    ple_input = (
        torch.nn.functional.gelu(ple_gate, approximate="tanh")
        * tensors["per_layer_input"]
    )
    ple_projection = torch.nn.functional.linear(ple_input, tensors["ple_proj_weight"])
    ple_projection = rms_norm_reference(
        ple_projection, tensors["post_ple_norm_weight"], shape.eps
    )
    output = (hidden + ple_projection) * tensors["layer_scalar"]
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
        "ff_input": ff_input,
        "gate_up": gate_up,
        "activation": activation,
        "down": down,
        "hidden": hidden,
        "ple_input": ple_input,
        "ple_projection": ple_projection,
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
    for sample in range(repeats):
        order = names[sample % len(names) :] + names[: sample % len(names)]
        for name in order:
            start.record()
            for _ in range(batch_replays):
                entries[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0 / batch_replays)
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
