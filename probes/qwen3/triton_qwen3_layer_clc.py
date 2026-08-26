# ruff: noqa: A001, A002, ANN001, ANN002, ANN003, ANN201, ANN202, E402, RET504
# pyrefly: ignore-errors
"""Standalone Qwen3-8B FP8 decode layer scheduled by Blackwell CLC.

The Helion-generated tile bodies and dependency continuations are preserved.
Cluster Launch Control turns physical CTAs into persistent workers: every CTA
issues an asynchronous cancellation before running its current tile, then a
successful cancellation keeps that CTA alive for one more tile.  Because CLC
does not promise dependency order for canceled CTA IDs, workers use one
monotonic relaxed cursor to claim scheduler-visible tiles in topological
order, while generated dependencies retain acquire/release ordering.  The
tuned path directly chains one ready attention-merge tile and the final
attention reduction, avoiding 160 scheduler commands without reducing fusion.

The instrumented specialization proves both partitions independently:
physical starts plus successful cancellations cover every launch token once,
and the tile cursor covers every scheduler-visible layer tile once.  Timings
use a 256 MiB L2 flush before every CUDA-graph replay.

This file has no dependency on the ``probes`` package. Copy it to a machine
with a compatible Helion checkout or installation plus Triton, PyTorch, and
CUDA Python, then run it directly.
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import inspect
import json
import linecache
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import textwrap
from typing import Callable
from typing import Protocol

from cuda.bindings import driver as cuda_driver
import torch
import triton

if configured_root := os.environ.get("HELION_REPO_ROOT"):
    sys.path.insert(0, configured_root)
for candidate_root in (Path.cwd(), *Path(__file__).resolve().parents):
    if (candidate_root / "helion" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate_root))
        break

import helion
from helion._compiler.cross_loop_scheduler import CROSS_LOOP_NUM_WORKERS_CONFIG
import helion.language as hl


def make_fp8_random(shape: tuple[int, ...], scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * scale).to(
        torch.float8_e4m3fn
    )


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


def capture(fn):
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


def _capture_with_reset(fn, reset):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            reset()
            output = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        reset()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    torch.cuda.synchronize()
    reset()
    torch.cuda.synchronize()
    return graph, output


def _benchmark_graphs_cold_l2(entries, repeats: int):
    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    samples = {name: [] for name in entries}
    names = list(entries)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for sample in range(repeats):
        order = names[sample % len(names) :] + names[: sample % len(names)]
        for name in order:
            replay, reset = entries[name]
            reset()
            triton.runtime.driver.active.clear_cache(cache)
            torch.cuda.synchronize()
            start.record()
            replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0)
    return {
        name: {
            "median_us": statistics.median(values),
            "mean_us": statistics.fmean(values),
            "p90_us": sorted(values)[min(len(values) - 1, int(0.9 * len(values)))],
        }
        for name, values in samples.items()
    }


def _resources(compiled, static_shared_bytes: int) -> dict[str, int]:
    compiled.run  # noqa: B018 - materialize the lazy CUDA function handle
    error, blocks_per_sm = cuda_driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        cuda_driver.CUfunction(int(compiled.function)),
        32,
        int(compiled.metadata.shared),
    )
    if error != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA occupancy query failed: {error}")
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    return {
        "registers": int(compiled.n_regs),
        "spills": int(compiled.n_spills),
        "triton_dynamic_shared_bytes": int(compiled.metadata.shared),
        "clc_static_shared_bytes": static_shared_bytes,
        "total_shared_bytes": int(compiled.metadata.shared) + static_shared_bytes,
        "blocks_per_sm": int(blocks_per_sm),
        "device_blocks": int(blocks_per_sm) * int(sm_count),
    }


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
    }


FP8_MAX = 448.0


FP8_MIN = -448.0


FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


W13_CONFIG = {
    "atomic_indexing": [],
    "block_sizes": [16],
    "indexing": ["pointer"] * 5,
    "l2_groupings": [1],
    "load_eviction_policies": [""] * 4,
    "loop_orders": [[0, 1]],
    "num_stages": 4,
    "num_warps": 1,
    "pid_type": "flat",
    "range_flattens": [None, False],
    "range_multi_buffers": [None, True],
    "range_num_stages": [0, 0],
    "range_unroll_factors": [0, 2],
    "range_warp_specializes": [None, None],
}


ACTIVATION_CONFIG = {
    "atomic_indexing": [],
    "block_sizes": [],
    "indexing": ["pointer"] * 4,
    "l2_groupings": [1],
    "load_eviction_policies": ["", ""],
    "loop_orders": [[0, 1]],
    "num_stages": 1,
    "num_warps": 4,
    "pid_type": "flat",
    "range_flattens": [None],
    "range_multi_buffers": [None],
    "range_num_stages": [0],
    "range_unroll_factors": [0],
    "range_warp_specializes": [None],
}


W2_CONFIG = {
    "atomic_indexing": [],
    "block_sizes": [8],
    "indexing": ["pointer"] * 5,
    "l2_groupings": [1],
    "load_eviction_policies": [""] * 4,
    "loop_orders": [[0, 1]],
    "num_stages": 4,
    "num_warps": 1,
    "pid_type": "flat",
    "range_flattens": [None, True],
    "range_multi_buffers": [None, False],
    "range_num_stages": [0, 4],
    "range_unroll_factors": [0, 4],
    "range_warp_specializes": [None, None],
}


FFN_CONFIGS = {
    "w13": W13_CONFIG,
    "silu_quant": ACTIVATION_CONFIG,
    "w2": W2_CONFIG,
}


QWEN3_B200_CONFIG = json.loads(r"""
{
  "attention_quant": {
    "atomic_indexing": [],
    "block_sizes": [
      16
    ],
    "indexing": [
      "pointer",
      "pointer",
      "pointer"
    ],
    "l2_groupings": [
      1
    ],
    "load_eviction_policies": [
      ""
    ],
    "loop_orders": [
      [
        0,
        1,
        2
      ]
    ],
    "num_stages": 1,
    "num_warps": 4,
    "pid_type": "flat",
    "range_flattens": [
      null
    ],
    "range_multi_buffers": [
      null
    ],
    "range_num_stages": [],
    "range_unroll_factors": [
      0
    ],
    "range_warp_specializes": [
      null
    ]
  },
  "decode_attention_merge": {
    "atomic_indexing": [],
    "block_sizes": [
      1,
      32
    ],
    "flatten_loops": [
      false
    ],
    "indexing": [
      "pointer",
      "tensor_descriptor",
      "tensor_descriptor"
    ],
    "l2_groupings": [
      8
    ],
    "load_eviction_policies": [
      "",
      "first"
    ],
    "loop_orders": [
      [
        0,
        1
      ]
    ],
    "num_stages": 2,
    "num_warps": 8,
    "pid_type": "flat",
    "range_flattens": [
      null,
      false
    ],
    "range_multi_buffers": [
      null,
      true
    ],
    "range_num_stages": [
      0,
      0
    ],
    "range_unroll_factors": [
      0,
      2
    ],
    "range_warp_specializes": [
      null,
      null
    ]
  },
  "decode_attention_split": {
    "atomic_indexing": [],
    "block_sizes": [
      8,
      128
    ],
    "indexing": [
      "pointer",
      "pointer",
      "tensor_descriptor",
      "tensor_descriptor",
      "pointer",
      "pointer",
      "pointer",
      "pointer"
    ],
    "l2_groupings": [
      16
    ],
    "load_eviction_policies": [
      "last",
      "last",
      "",
      "last",
      "",
      "last"
    ],
    "loop_orders": [
      [
        2,
        1,
        0
      ]
    ],
    "num_stages": 2,
    "num_warps": 4,
    "pid_type": "flat",
    "range_flattens": [
      null,
      true
    ],
    "range_multi_buffers": [
      null,
      true
    ],
    "range_num_stages": [
      0,
      3
    ],
    "range_unroll_factors": [
      0,
      0
    ],
    "range_warp_specializes": [
      null,
      null
    ]
  },
  "kv_cache_update": {
    "atomic_indexing": [],
    "block_sizes": [
      4,
      64
    ],
    "indexing": [
      "pointer",
      "pointer",
      "tensor_descriptor",
      "pointer",
      "tensor_descriptor",
      "tensor_descriptor",
      "pointer"
    ],
    "l2_groupings": [
      32
    ],
    "load_eviction_policies": [
      "first",
      "last",
      "first",
      "last",
      "last"
    ],
    "loop_orders": [
      [
        0,
        1,
        2
      ]
    ],
    "num_stages": 4,
    "num_warps": 1,
    "pid_type": "flat",
    "range_flattens": [
      null
    ],
    "range_multi_buffers": [
      null
    ],
    "range_num_stages": [
      0
    ],
    "range_unroll_factors": [
      0
    ],
    "range_warp_specializes": [
      null
    ]
  },
  "o_mm": {
    "atomic_indexing": [],
    "block_sizes": [
      8
    ],
    "indexing": [
      "pointer",
      "pointer",
      "pointer",
      "pointer",
      "pointer"
    ],
    "l2_groupings": [
      4
    ],
    "load_eviction_policies": [
      "",
      "",
      "",
      ""
    ],
    "loop_orders": [
      [
        1,
        0
      ]
    ],
    "num_stages": 4,
    "num_warps": 1,
    "pid_type": "flat",
    "range_flattens": [
      null,
      false
    ],
    "range_multi_buffers": [
      null,
      false
    ],
    "range_num_stages": [
      0,
      0
    ],
    "range_unroll_factors": [
      0,
      2
    ],
    "range_warp_specializes": [
      null,
      null
    ]
  },
  "qk_norm_rope": {
    "atomic_indexing": [],
    "block_sizes": [
      4
    ],
    "indexing": [
      "tensor_descriptor",
      "tensor_descriptor",
      "pointer",
      "pointer",
      "tensor_descriptor",
      "tensor_descriptor",
      "pointer",
      "pointer",
      "pointer",
      "pointer",
      "pointer"
    ],
    "l2_groupings": [
      2
    ],
    "load_eviction_policies": [
      "",
      "last",
      "",
      "last",
      "last",
      "first",
      "first",
      ""
    ],
    "loop_orders": [
      [
        0,
        1,
        2
      ]
    ],
    "num_stages": 3,
    "num_warps": 4,
    "pid_type": "flat",
    "range_flattens": [
      null
    ],
    "range_multi_buffers": [
      null
    ],
    "range_num_stages": [],
    "range_unroll_factors": [
      0
    ],
    "range_warp_specializes": [
      null
    ]
  },
  "qkv_mm": {
    "atomic_indexing": [],
    "block_sizes": [
      16
    ],
    "indexing": [
      "pointer",
      "pointer",
      "pointer",
      "pointer",
      "pointer"
    ],
    "l2_groupings": [
      1
    ],
    "load_eviction_policies": [
      "",
      "",
      "",
      ""
    ],
    "loop_orders": [
      [
        0,
        1
      ]
    ],
    "num_stages": 4,
    "num_warps": 1,
    "pid_type": "flat",
    "range_flattens": [
      null,
      false
    ],
    "range_multi_buffers": [
      null,
      true
    ],
    "range_num_stages": [
      0,
      0
    ],
    "range_unroll_factors": [
      0,
      2
    ],
    "range_warp_specializes": [
      null,
      null
    ]
  },
  "rms_quant": {
    "atomic_indexing": [],
    "block_sizes": [
      4096,
      32
    ],
    "indexing": [
      "pointer",
      "tensor_descriptor",
      "pointer",
      "pointer",
      "tensor_descriptor",
      "tensor_descriptor",
      "tensor_descriptor",
      "pointer"
    ],
    "load_eviction_policies": [
      "",
      "first",
      "last",
      "",
      ""
    ],
    "loop_orders": [
      [
        0,
        1
      ]
    ],
    "num_stages": 1,
    "num_warps": 16,
    "pid_type": "flat",
    "range_flattens": [
      null,
      true,
      true,
      null
    ],
    "range_multi_buffers": [
      null,
      null,
      null,
      null
    ],
    "range_num_stages": [],
    "range_unroll_factors": [
      0,
      0,
      0,
      0
    ],
    "range_warp_specializes": [
      null,
      null,
      null,
      null
    ],
    "static_ranges": [
      false
    ]
  }
}
""")


def rms_quant_baseline(
    result,
    input,
    weight,
    scale,
    epsilon,
    scale_ub,
    residual,
    group_size,
    is_scale_transposed,
):
    del is_scale_transposed
    num_tokens, hidden_size = input.shape
    x = input.float()
    if residual is not None:
        x = x + residual.float()
        residual.copy_(x.to(residual.dtype))
    rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)
    x_norm = (x * rms).to(input.dtype) * weight
    grouped = x_norm.view(num_tokens, hidden_size // group_size, group_size).float()
    s = grouped.abs().amax(-1)
    if scale_ub is not None:
        s = s.clamp(max=scale_ub)
    s = (s / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    scale.copy_(s)
    result.copy_((grouped / s[:, :, None]).clamp(FP8_MIN, FP8_MAX).view_as(result))


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=rms_quant_baseline,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def rms_norm_per_block_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    """Target-shape copy of vLLM's Helion rms_norm_per_block_quant."""
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)
    groups_per_row = scale.shape[1]
    hl.specialize(groups_per_row)
    assert group_size == 128
    assert result.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32

    for tile_m in hl.tile(num_tokens, block_size=1):
        rms = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            rms = rms + x_blk.pow(2).sum(dim=-1)
        rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)

        m_idx = tile_m.begin + hl.arange(tile_m.block_size)
        m_blk = m_idx[:, None, None]
        for tile_gn, tile_n in hl.tile(
            [groups_per_row, group_size], block_size=[None, group_size]
        ):
            gn_idx = tile_gn.index
            n_idx = gn_idx[:, None] * group_size + tile_n.index[None, :]
            n_blk = n_idx[None, :, :]
            mask = (gn_idx < groups_per_row)[None, :, None]
            x_blk = hl.load(input, [m_blk, n_blk], extra_mask=mask).to(torch.float32)
            if residual is not None:
                x_blk = x_blk + hl.load(residual, [m_blk, n_blk], extra_mask=mask)
            w_blk = hl.load(weight, [n_blk], extra_mask=mask)
            x_norm = (x_blk * rms[:, None, None]).to(input.dtype) * w_blk
            s = torch.amax(torch.abs(x_norm), dim=-1).to(torch.float32)
            if scale_ub is not None:
                s = s.clamp(max=hl.load(scale_ub, []))
            s = (s / FP8_MAX).clamp(min=FP8_MIN_SCALE)
            scale[tile_m, tile_gn] = s
            y = (x_norm / s[:, :, None]).clamp(FP8_MIN, FP8_MAX).to(result.dtype)
            hl.store(result, [m_blk, n_blk], y, extra_mask=mask)
            if residual is not None:
                hl.store(
                    residual, [m_blk, n_blk], x_blk.to(residual.dtype), extra_mask=mask
                )


def qk_norm_rope_baseline(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    eps,
    q_weight,
    k_weight,
    cos_sin_cache,
    is_neox,
    position_ids,
    forced_token_heads_per_warp=-1,
):
    del num_heads_v, forced_token_heads_per_warp
    q_size = num_heads_q * head_dim
    kv_size = num_heads_k * head_dim
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
    qh = q.view(-1, num_heads_q, head_dim)
    kh = k.view(-1, num_heads_k, head_dim)
    qh = (
        qh.float() * torch.rsqrt(qh.float().square().mean(-1, keepdim=True) + eps)
    ).to(qkv.dtype) * q_weight
    kh = (
        kh.float() * torch.rsqrt(kh.float().square().mean(-1, keepdim=True) + eps)
    ).to(qkv.dtype) * k_weight
    cache = cos_sin_cache[position_ids]
    embed = cache.shape[-1] // 2
    cos, sin = cache[..., :embed], cache[..., embed:]
    if is_neox:

        def rotate(x):
            x1, x2 = x[..., :embed], x[..., embed : 2 * embed]
            return torch.cat(
                (
                    x1 * cos[:, None] - x2 * sin[:, None],
                    x2 * cos[:, None] + x1 * sin[:, None],
                ),
                dim=-1,
            )
    else:

        def rotate(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = x1 * cos[:, None] - x2 * sin[:, None]
            out[..., 1::2] = x2 * cos[:, None] + x1 * sin[:, None]
            return out

    qkv[:, :q_size].copy_(rotate(qh).reshape_as(q))
    qkv[:, q_size : q_size + kv_size].copy_(rotate(kh).reshape_as(k))


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=qk_norm_rope_baseline,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=5e-2,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int = -1,
) -> None:
    """Exact Helion body used by vLLM's fused_qk_norm_rope."""
    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v
    hl.specialize(qkv.shape[1])
    _, rotary_dim = cos_sin_cache.shape
    hl.specialize(rotary_dim)
    embed_dim = rotary_dim // 2
    hl.specialize(num_heads_q)
    hl.specialize(num_heads_k)
    hl.specialize(num_heads_v)
    hl.specialize(head_dim)
    qk_heads = num_heads_q + num_heads_k
    qkv = qkv.view(num_tokens, total_heads, head_dim)

    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, qk_heads, head_dim], block_size=[1, None, head_dim]
    ):
        x = qkv[tile_m, tile_gn, tile_n].to(torch.float32)
        rms = torch.rsqrt(x.pow(2).sum(-1) * (1.0 / head_dim) + eps)
        use_q = (tile_gn.index < num_heads_q)[None, :, None]
        w = torch.where(
            use_q,
            q_weight[None, None, tile_n],
            k_weight[None, None, tile_n],
        )
        x = (x * rms[:, :, None]).to(qkv.dtype) * w
        qkv[tile_m, tile_gn, tile_n] = x
        pos = position_ids[tile_m]
        cos = cos_sin_cache[pos, hl.arange(embed_dim)]
        sin = cos_sin_cache[pos, hl.arange(embed_dim) + embed_dim]
        if is_neox:
            x1_offset = hl.arange(embed_dim)
            x2_offset = x1_offset + embed_dim
        else:
            x1_offset = hl.arange(embed_dim) * 2
            x2_offset = x1_offset + 1
        x1 = qkv[tile_m, tile_gn, x1_offset]
        x2 = qkv[tile_m, tile_gn, x2_offset]
        qkv[tile_m, tile_gn, x1_offset] = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        qkv[tile_m, tile_gn, x2_offset] = x2 * cos[:, None, :] + x1 * sin[:, None, :]


def group_quant_baseline(
    input,
    output_q,
    output_s,
    group_size,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
    dummy_is_scale_transposed=False,
    dummy_is_tma_aligned=False,
):
    del dummy_is_scale_transposed, dummy_is_tma_aligned
    grouped = input.view(input.shape[0], -1, group_size).float()
    s = grouped.abs().amax(-1).clamp(min=eps) / fp8_max
    if scale_ue8m0:
        s = torch.exp2(torch.ceil(torch.log2(s)))
    output_s.copy_(s)
    output_q.copy_((grouped / s[:, :, None]).clamp(fp8_min, fp8_max).view_as(output_q))


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=group_quant_baseline,
    # FP8 conversion can land on an adjacent representable value even when
    # the FP32 scale agrees. Keep this tolerance local to the FP8 payload.
    autotune_baseline_atol=1.0,
    autotune_baseline_rtol=2e-2,
)
def per_token_group_fp8_quant(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    """Exact Helion body used by vLLM's per_token_group_fp8_quant."""
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)
    groups_per_row = output_s.shape[1]
    hl.specialize(groups_per_row)
    input = input.view(num_tokens, groups_per_row, group_size)
    output_q = output_q.view(num_tokens, groups_per_row, group_size)
    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    ):
        x = input[tile_m, tile_gn, tile_n]
        s = torch.amax(torch.abs(x), dim=-1).clamp(min=eps) / fp8_max
        if scale_ue8m0:
            s = torch.exp2(torch.ceil(torch.log2(s)))
        output_s[tile_m, tile_gn] = s
        output_q[tile_m, tile_gn, tile_n] = (
            (x / s[:, :, None]).clamp(fp8_min, fp8_max).to(output_q.dtype)
        )


@helion.kernel(static_shapes=True, autotune_effort="full")
def block_fp8_mm(
    activation_q: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    m, k = activation_q.size()
    n, weight_k = weight_q.size()
    assert weight_k == k
    assert group_size == 128
    hl.specialize(group_size)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=activation_q.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=group_size):
            partial = hl.dot(
                activation_q[tile_m, tile_k],
                weight_q[tile_n, tile_k].T,
            ).to(torch.float32)
            a_scale = activation_scale[tile_m, tile_k.id].to(torch.float32)
            w_scale = weight_scale[tile_n.index // group_size, tile_k.id].to(
                torch.float32
            )
            acc = acc + partial * a_scale[:, None] * w_scale[None, :]
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(static_shapes=True, autotune_effort="full")
def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Helion equivalent of vLLM's separate FlashAttention cache-update op."""
    num_tokens, num_kv_heads, head_dim = key.shape
    hl.specialize(num_kv_heads)
    hl.specialize(head_dim)
    hl.specialize(block_size)
    for tile_t, tile_h, tile_d in hl.tile(
        [num_tokens, num_kv_heads, head_dim], block_size=[1, None, None]
    ):
        t = tile_t.index
        h = tile_h.index
        d = tile_d.index
        slot = slot_mapping[t]
        block = (slot // block_size)[:, None, None]
        offset = (slot % block_size)[:, None, None]
        h_idx = h[None, :, None]
        d_idx = d[None, None, :]
        key_value = key[tile_t, tile_h, tile_d]
        value_value = value[tile_t, tile_h, tile_d]
        hl.store(kv_cache, [block, offset, h_idx, d_idx], key_value)
        hl.store(
            kv_cache,
            [block, offset, h_idx, d_idx + head_dim],
            value_value,
        )


def paged_gqa_attention_baseline(
    query,
    kv_cache,
    block_table,
    context,
    block_size,
    q_per_kv,
):
    head_dim = query.shape[-1]
    outputs = []
    for token in range(query.shape[0]):
        blocks = block_table[token, : math.ceil(context / block_size)].long()
        logical = kv_cache[blocks].reshape(-1, kv_cache.shape[2], kv_cache.shape[3])[
            :context
        ]
        k = logical[..., :head_dim].permute(1, 0, 2).repeat_interleave(q_per_kv, dim=0)
        v = logical[..., head_dim:].permute(1, 0, 2).repeat_interleave(q_per_kv, dim=0)
        q = query[token].unsqueeze(1)
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(q, k, v).squeeze(1)
        )
    return torch.cat(outputs, dim=0).unsqueeze(0)


def paged_gqa_attention_split_baseline(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    block_size: int,
    q_per_kv: int,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = query.shape[-1]
    num_kv_heads = kv_cache.shape[2]
    num_tokens = query.shape[0]
    split_context = context // splits
    q = query.reshape(num_tokens * num_kv_heads, q_per_kv, head_dim).float()
    partial_out = torch.empty(
        (splits, num_tokens * num_kv_heads, q_per_kv, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (splits, num_tokens * num_kv_heads, q_per_kv),
        device=query.device,
        dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(head_dim)
    for token in range(num_tokens):
        blocks = block_table[token, : math.ceil(context / block_size)].long()
        logical = kv_cache[blocks].reshape(-1, kv_cache.shape[2], kv_cache.shape[3])[
            :context
        ]
        group_begin = token * num_kv_heads
        group_end = group_begin + num_kv_heads
        for split in range(splits):
            begin = split * split_context
            end = begin + split_context
            k = logical[begin:end, :, :head_dim].permute(1, 0, 2).float()
            v = logical[begin:end, :, head_dim:].permute(1, 0, 2).float()
            scores = torch.einsum("gqd,gnd->gqn", q[group_begin:group_end], k) * scale
            partial_lse[split, group_begin:group_end] = torch.logsumexp(
                scores, dim=-1
            ) * math.log2(math.e)
            partial_out[split, group_begin:group_end] = torch.einsum(
                "gqn,gnd->gqd", torch.softmax(scores, dim=-1), v
            )
    return partial_out, partial_lse


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=paged_gqa_attention_split_baseline,
    autotune_baseline_atol=8e-2,
    autotune_baseline_rtol=3e-2,
)
def paged_gqa_decode_attention_split(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    block_size: int,
    q_per_kv: int,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-KV partials for paged M=1 GQA decode attention."""
    num_tokens, num_q_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[2]
    assert num_q_heads == num_kv_heads * q_per_kv
    assert context % splits == 0
    hl.specialize(head_dim)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(context)
    hl.specialize(block_size)
    hl.specialize(splits)
    split_context = context // splits
    token_kv_heads = num_tokens * num_kv_heads
    partial_out = torch.empty(
        (splits, token_kv_heads, q_per_kv, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (splits, token_kv_heads, q_per_kv),
        device=query.device,
        dtype=torch.float32,
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_split, tile_bg, tile_q in hl.tile(
        [splits, token_kv_heads, q_per_kv], block_size=[1, 1, None]
    ):
        m_i = hl.full([tile_bg, tile_q], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_bg, tile_q], 1.0, dtype=torch.float32)
        acc = hl.zeros([tile_bg, tile_q, head_dim], dtype=torch.float32)
        split_idx = tile_split.begin
        token = tile_bg.index // num_kv_heads
        kv_head = tile_bg.index % num_kv_heads
        query_head = kv_head[:, None] * q_per_kv + tile_q.index[None, :]
        q_blk = query[token[:, None], query_head, :]
        q_blk = (q_blk * qk_scale).to(query.dtype)
        for tile_local_n in hl.tile(split_context):
            n = split_idx * split_context + tile_local_n.index
            physical_block = block_table[token[:, None], (n // block_size)[None, :]]
            block_offset = n % block_size
            d = hl.arange(head_dim)
            k = hl.load(
                kv_cache,
                [
                    physical_block[:, :, None],
                    block_offset[None, :, None],
                    kv_head[:, None, None],
                    d[None, None, :],
                ],
            )
            scores = torch.bmm(q_blk, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(scores, -1))
            p = torch.exp2(scores - m_ij[:, :, None])
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            acc = acc * alpha[:, :, None]
            v = hl.load(
                kv_cache,
                [
                    physical_block[:, :, None],
                    block_offset[None, :, None],
                    kv_head[:, None, None],
                    (d + head_dim)[None, None, :],
                ],
            )
            acc = torch.baddbmm(acc, p.to(v.dtype), v)
            m_i = m_ij
        partial_out[tile_split, tile_bg, tile_q, :] = (acc / l_i[:, :, None])[
            None, :, :, :
        ]
        partial_lse[tile_split, tile_bg, tile_q] = (m_i + torch.log2(l_i))[None, :, :]
    return partial_out, partial_lse


def merge_attention_baseline(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    max_lse = partial_lse.amax(dim=0)
    weights = torch.exp2(partial_lse - max_lse[None])
    output = (partial_out * weights[..., None]).sum(dim=0)
    output = output / weights.sum(dim=0)[..., None]
    return output.to(torch.bfloat16).view(1, -1, partial_out.shape[-1])


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    autotune_baseline_fn=merge_attention_baseline,
    autotune_baseline_atol=2e-2,
    autotune_baseline_rtol=2e-2,
)
def merge_attention_splits(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    splits, num_kv_heads, q_per_kv, head_dim = partial_out.shape
    output = torch.empty(
        (num_kv_heads, q_per_kv, head_dim),
        device=partial_out.device,
        dtype=torch.bfloat16,
    )
    for tile_g, tile_q in hl.tile([num_kv_heads, q_per_kv], block_size=[1, None]):
        max_lse = hl.full([tile_g, tile_q], float("-inf"), dtype=torch.float32)
        denominator = hl.zeros([tile_g, tile_q], dtype=torch.float32)
        accumulator = hl.zeros([tile_g, tile_q, head_dim], dtype=torch.float32)
        for tile_split in hl.tile(splits):
            lse = partial_lse[tile_split, tile_g, tile_q]
            next_max = torch.maximum(max_lse, torch.amax(lse, dim=0))
            old_weight = torch.exp2(max_lse - next_max)
            weights = torch.exp2(lse - next_max[None, :, :])
            denominator = denominator * old_weight + torch.sum(weights, dim=0)
            values = partial_out[tile_split, tile_g, tile_q, :]
            accumulator = accumulator * old_weight[:, :, None] + torch.sum(
                values * weights[:, :, :, None], dim=0
            )
            max_lse = next_max
        output[tile_g, tile_q, :] = (accumulator / denominator[:, :, None]).to(
            output.dtype
        )
    return output.view(1, num_kv_heads * q_per_kv, head_dim)


@helion.kernel(static_shapes=True, autotune_effort="full")
def silu_and_mul_per_block_quant(
    gate_up: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, twice_intermediate = gate_up.size()
    intermediate = twice_intermediate // 2
    hl.specialize(group_size)
    groups = intermediate // group_size
    activation_q = torch.empty(
        (m, intermediate), dtype=torch.float8_e4m3fn, device=gate_up.device
    )
    activation_scale = torch.empty(
        (m, groups), dtype=torch.float32, device=gate_up.device
    )
    for tile_m, tile_i in hl.tile([m, intermediate], block_size=[1, group_size]):
        gate = gate_up[tile_m, tile_i].to(torch.float32)
        up = gate_up[tile_m, tile_i + intermediate].to(torch.float32)
        activated = gate * torch.sigmoid(gate) * up
        scale = (torch.amax(torch.abs(activated), dim=-1) / FP8_MAX).clamp(
            min=FP8_MIN_SCALE
        )
        activation_scale[tile_m, tile_i.id] = scale
        activation_q[tile_m, tile_i] = (
            (activated / scale[:, None]).clamp(FP8_MIN, FP8_MAX).to(activation_q.dtype)
        )
    return activation_q, activation_scale


def compile_config(kernel, kernel_args, config_dict):
    bound = kernel.bind(kernel_args)
    config = helion.Config.from_dict(config_dict)
    bound.config_spec.normalize(config.config)
    return config, bound.compile_config(config)


def make_cos_sin(max_position, head_dim, theta, device):
    inv = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(
        torch.arange(max_position, device=device, dtype=torch.float32), inv
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)


def allocate(args):
    torch.manual_seed(args.seed)
    device = "cuda"
    hidden_groups = args.hidden // args.group
    intermediate_groups = args.intermediate // args.group
    qkv_width = (args.q_heads + 2 * args.kv_heads) * args.head_dim
    logical_blocks = math.ceil(args.context / args.block_size)
    batch = args.batch
    physical_blocks = math.ceil(batch * logical_blocks * 1.25)
    block_table = (
        torch.randperm(physical_blocks, device=device, dtype=torch.int64)[
            : batch * logical_blocks
        ]
        .to(torch.int32)
        .view(batch, logical_blocks)
    )
    final_logical_block = (args.context - 1) // args.block_size
    final_block_offset = (args.context - 1) % args.block_size
    final_physical_blocks = block_table[:, final_logical_block].to(torch.int64)
    tensors = {
        "hidden_states": torch.randn(
            (batch, args.hidden), device=device, dtype=torch.bfloat16
        ),
        "residual": torch.randn(
            (batch, args.hidden), device=device, dtype=torch.bfloat16
        ),
        "pre_weight": torch.randn((args.hidden,), device=device, dtype=torch.bfloat16)
        * 0.1
        + 1.0,
        "post_weight": torch.randn((args.hidden,), device=device, dtype=torch.bfloat16)
        * 0.1
        + 1.0,
        "q_weight": torch.randn((args.head_dim,), device=device, dtype=torch.bfloat16)
        * 0.1
        + 1.0,
        "k_weight": torch.randn((args.head_dim,), device=device, dtype=torch.bfloat16)
        * 0.1
        + 1.0,
        "position": torch.full(
            (batch,), args.context - 1, device=device, dtype=torch.int64
        ),
        "cos_sin": make_cos_sin(
            max(args.context, 4096), args.head_dim, args.rope_theta, device
        ),
        "pre_q": torch.empty(
            (batch, args.hidden), device=device, dtype=torch.float8_e4m3fn
        ),
        "pre_scale": torch.empty(
            (batch, hidden_groups), device=device, dtype=torch.float32
        ),
        "qkv_weight_q": make_fp8_random((qkv_width, args.hidden)),
        "qkv_weight_scale": torch.rand(
            (qkv_width // args.group, hidden_groups), device=device
        )
        * 0.01
        + 0.01,
        "kv_cache": torch.randn(
            (physical_blocks, args.block_size, args.kv_heads, 2 * args.head_dim),
            device=device,
            dtype=torch.bfloat16,
        ),
        "block_table": block_table,
        "slot_mapping": final_physical_blocks * args.block_size + final_block_offset,
        "attention_q": torch.empty(
            (batch, args.hidden), device=device, dtype=torch.float8_e4m3fn
        ),
        "attention_scale": torch.empty(
            (batch, hidden_groups), device=device, dtype=torch.float32
        ),
        "o_weight_q": make_fp8_random((args.hidden, args.hidden)),
        "o_weight_scale": torch.rand((hidden_groups, hidden_groups), device=device)
        * 0.01
        + 0.01,
        "ffn_q": torch.empty(
            (batch, args.hidden), device=device, dtype=torch.float8_e4m3fn
        ),
        "ffn_scale": torch.empty(
            (batch, hidden_groups), device=device, dtype=torch.float32
        ),
        "w13_q": make_fp8_random((2 * args.intermediate, args.hidden)),
        "w13_scale": torch.rand((2 * intermediate_groups, hidden_groups), device=device)
        * (0.5 / args.hidden**0.5)
        + (0.75 / args.hidden**0.5),
        "w2_q": make_fp8_random((args.hidden, args.intermediate)),
        "w2_scale": torch.rand((hidden_groups, intermediate_groups), device=device)
        * (0.5 / args.intermediate**0.5)
        + (0.75 / args.intermediate**0.5),
    }
    return tensors


_USE_CANONICAL_ATTENTION_VIEWS = False


_USE_TASK_ALIGNED_ATTENTION = False


@helion.kernel(static_shapes=True, autotune_effort="none")
def tiled_rms_norm_per_block_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    """RMS/quant expressed as source-visible producer, finalize, consumer tiles."""
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)
    groups_per_row = scale.shape[1]
    hl.specialize(groups_per_row)
    assert group_size == 128
    assert result.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32

    rms_partials = torch.empty(
        (num_tokens, groups_per_row), dtype=torch.float32, device=input.device
    )
    unrounded_values = torch.empty_like(input, dtype=torch.float32)

    for partial_m, partial_n in hl.tile(
        [num_tokens, hidden_size], block_size=[1, group_size]
    ):
        partial_values = input[partial_m, partial_n].to(torch.float32)
        if residual is not None:
            partial_values = partial_values + residual[partial_m, partial_n]
            residual[partial_m, partial_n] = partial_values.to(residual.dtype)
        unrounded_values[partial_m, partial_n] = partial_values
        rms_partials[partial_m, partial_n.id] = torch.sum(
            partial_values * partial_values, dim=-1
        )

    for quant_m, quant_g, quant_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, 1, group_size]
    ):
        quant_m_idx = quant_m.begin + hl.arange(quant_m.block_size)
        quant_group_idx = quant_g.index
        quant_n_idx = quant_group_idx[:, None] * group_size + quant_n.index[None, :]
        quant_m_blk = quant_m_idx[:, None, None]
        quant_n_blk = quant_n_idx[None, :, :]
        square_sum = hl.zeros([quant_m], dtype=torch.float32)
        for reduce_g in hl.tile(groups_per_row, block_size=1):
            square_sum = square_sum + torch.sum(rms_partials[quant_m, reduce_g], dim=-1)
        inv_rms = torch.rsqrt(square_sum * (1.0 / hidden_size) + epsilon)
        quant_values = unrounded_values[quant_m_blk, quant_n_blk]
        normalized = (quant_values * inv_rms[:, None, None]).to(
            torch.bfloat16
        ) * weight[quant_n_blk]
        quant_scale = torch.amax(torch.abs(normalized), dim=-1).to(torch.float32)
        if scale_ub is not None:
            quant_scale = quant_scale.clamp(max=hl.load(scale_ub, []))
        quant_scale = (quant_scale / FP8_MAX).clamp(min=FP8_MIN_SCALE)
        scale[quant_m, quant_g] = quant_scale
        result[quant_m_blk, quant_n_blk] = (
            (normalized / quant_scale[:, :, None])
            .clamp(FP8_MIN, FP8_MAX)
            .to(result.dtype)
        )


@helion.kernel(static_shapes=True, autotune_effort="none")
def tiled_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Store one complete K/V head from each logical cache task."""
    num_tokens, num_kv_heads, head_dim = key.shape
    hl.specialize(num_kv_heads)
    hl.specialize(head_dim)
    hl.specialize(block_size)
    for tile_t, tile_h, tile_d in hl.tile(
        [num_tokens, num_kv_heads, head_dim],
        block_size=[1, 1, head_dim],
    ):
        token = tile_t.index
        cache_head = tile_h.index
        dimension = tile_d.index
        key_value = key[tile_t, tile_h, tile_d]
        value_value = value[tile_t, tile_h, tile_d]
        slot = slot_mapping[token]
        block = (slot // block_size)[:, None, None]
        offset = (slot % block_size)[:, None, None]
        hl.store(
            kv_cache,
            [
                block,
                offset,
                cache_head[None, :, None],
                dimension[None, None, :],
            ],
            key_value,
        )
        hl.store(
            kv_cache,
            [
                block,
                offset,
                cache_head[None, :, None],
                (dimension + head_dim)[None, None, :],
            ],
            value_value,
        )


@helion.kernel(static_shapes=True, autotune_effort="none")
def flat_fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int = -1,
) -> None:
    """Apply Q/K normalization and RoPE over contiguous flat head tiles."""
    num_tokens, qkv_width = qkv.shape
    total_heads = num_heads_q + num_heads_k + num_heads_v
    assert qkv_width == total_heads * head_dim
    _, rotary_dim = cos_sin_cache.shape
    hl.specialize(qkv_width)
    hl.specialize(rotary_dim)
    embed_dim = rotary_dim // 2
    hl.specialize(num_heads_q)
    hl.specialize(num_heads_k)
    hl.specialize(num_heads_v)
    hl.specialize(head_dim)
    assert is_neox
    assert rotary_dim == head_dim
    qk_width = (num_heads_q + num_heads_k) * head_dim

    for tile_m, tile_n in hl.tile([num_tokens, qk_width], block_size=[1, head_dim]):
        x = qkv[tile_m, tile_n].to(torch.float32)
        rms = torch.rsqrt(x.pow(2).sum(-1) * (1.0 / head_dim) + eps)
        dimension = tile_n.index - tile_n.begin
        use_q = tile_n.index < num_heads_q * head_dim
        weight = torch.where(use_q, q_weight[dimension], k_weight[dimension])
        x = (x * rms[:, None]).to(qkv.dtype) * weight[None, :]
        position = position_ids[tile_m]
        first_half = dimension < embed_dim
        partner_dimension = torch.where(
            first_half, dimension + embed_dim, dimension - embed_dim
        )
        partner = torch.gather(x, 1, partner_dimension[None, :])
        cos = cos_sin_cache[position, dimension % embed_dim]
        sin = cos_sin_cache[position, dimension % embed_dim + embed_dim]
        qkv[tile_m, tile_n] = x * cos[:, :] + torch.where(
            first_half[None, :], -partner * sin[:, :], partner * sin[:, :]
        )


@helion.kernel(static_shapes=True, autotune_effort="none")
def canonical_paged_gqa_decode_attention_split(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    block_size: int,
    q_per_kv: int,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split attention with partials stored in canonical query-head coordinates."""
    _, num_q_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[2]
    assert num_q_heads == num_kv_heads * q_per_kv
    assert context % splits == 0
    hl.specialize(head_dim)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(context)
    hl.specialize(block_size)
    hl.specialize(splits)
    split_context = context // splits
    q = query.view(num_kv_heads, q_per_kv, head_dim)
    partial_out = torch.empty(
        (splits, num_q_heads, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (splits, num_q_heads),
        device=query.device,
        dtype=torch.float32,
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_split, tile_g, tile_q in hl.tile(
        [splits, num_kv_heads, q_per_kv], block_size=[1, 1, None]
    ):
        m_i = hl.full([tile_g, tile_q], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_g, tile_q], 1.0, dtype=torch.float32)
        acc = hl.zeros([tile_g, tile_q, head_dim], dtype=torch.float32)
        split_idx = tile_split.begin
        q_blk = (q[tile_g, tile_q, :] * qk_scale).to(q.dtype)
        for tile_local_n in hl.tile(split_context):
            n = split_idx * split_context + tile_local_n.index
            physical_block = block_table[0, n // block_size]
            block_offset = n % block_size
            d = hl.arange(head_dim)
            k = hl.load(
                kv_cache,
                [
                    physical_block[None, :, None],
                    block_offset[None, :, None],
                    tile_g.index[:, None, None],
                    d[None, None, :],
                ],
            )
            scores = torch.bmm(q_blk, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(scores, -1))
            p = torch.exp2(scores - m_ij[:, :, None])
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            acc = acc * alpha[:, :, None]
            v = hl.load(
                kv_cache,
                [
                    physical_block[None, :, None],
                    block_offset[None, :, None],
                    tile_g.index[:, None, None],
                    (d + head_dim)[None, None, :],
                ],
            )
            acc = torch.baddbmm(acc, p.to(v.dtype), v)
            m_i = m_ij
        query_head = tile_g.index[:, None] * q_per_kv + tile_q.index[None, :]
        partial_out[tile_split, query_head, :] = (acc / l_i[:, :, None])[None, :, :, :]
        partial_lse[tile_split, query_head] = (m_i + torch.log2(l_i))[None, :, :]
    return partial_out, partial_lse


@helion.kernel(static_shapes=True, autotune_effort="none")
def task_aligned_paged_gqa_decode_attention_split(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    context: int,
    block_size: int,
    q_per_kv: int,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain the baseline layout while exposing the KV-head task coordinate."""
    _, num_q_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[2]
    assert num_q_heads == num_kv_heads * q_per_kv
    assert context % splits == 0
    hl.specialize(head_dim)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(context)
    hl.specialize(block_size)
    hl.specialize(splits)
    split_context = context // splits
    q = query.view(num_kv_heads, q_per_kv, head_dim)
    partial_out = torch.empty(
        (splits, num_kv_heads, q_per_kv, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (splits, num_kv_heads, q_per_kv),
        device=query.device,
        dtype=torch.float32,
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_split, tile_g, tile_q in hl.tile(
        [splits, num_kv_heads, q_per_kv], block_size=[1, 1, None]
    ):
        m_i = hl.full([tile_g, tile_q], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_g, tile_q], 1.0, dtype=torch.float32)
        acc = hl.zeros([tile_g, tile_q, head_dim], dtype=torch.float32)
        split_idx = tile_split.begin
        q_blk = (q[tile_g, tile_q, :] * qk_scale).to(q.dtype)
        for tile_local_n in hl.tile(split_context):
            n = split_idx * split_context + tile_local_n.index
            physical_block = block_table[0, n // block_size]
            block_offset = n % block_size
            d = hl.arange(head_dim)
            k = hl.load(
                kv_cache,
                [
                    physical_block[None, :, None],
                    block_offset[None, :, None],
                    tile_g.index[:, None, None],
                    d[None, None, :],
                ],
            )
            scores = torch.bmm(q_blk, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(scores, -1))
            p = torch.exp2(scores - m_ij[:, :, None])
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + torch.sum(p, -1)
            acc = acc * alpha[:, :, None]
            v = hl.load(
                kv_cache,
                [
                    physical_block[None, :, None],
                    block_offset[None, :, None],
                    tile_g.index[:, None, None],
                    (d + head_dim)[None, None, :],
                ],
            )
            acc = torch.baddbmm(acc, p.to(v.dtype), v)
            m_i = m_ij
        partial_out[tile_split, tile_g, tile_q, :] = (acc / l_i[:, :, None])[
            None, :, :, :
        ]
        partial_lse[tile_split, tile_g, tile_q] = (m_i + torch.log2(l_i))[None, :, :]
    return partial_out, partial_lse


@helion.kernel(static_shapes=True, autotune_effort="none")
def tiled_merge_attention_splits(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    """Two source-visible merge levels matching the Triton overlap probe."""
    splits, num_kv_heads, q_per_kv, head_dim = partial_out.shape
    hl.specialize(splits)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(head_dim)
    query_heads = num_kv_heads * q_per_kv
    hl.specialize(query_heads)
    merge_chunks = 16
    hl.specialize(merge_chunks)
    assert splits % merge_chunks == 0
    splits_per_chunk = splits // merge_chunks
    hl.specialize(splits_per_chunk)

    partial_out_flat = partial_out.view(splits, query_heads, head_dim)
    partial_lse_flat = partial_lse.view(splits, query_heads)
    partial_out_storage = partial_out_flat.view(-1)
    partial_lse_storage = partial_lse_flat.view(-1)
    chunk_out = torch.empty(
        (merge_chunks, query_heads, head_dim),
        dtype=torch.float32,
        device=partial_out.device,
    )
    chunk_lse = torch.empty(
        (merge_chunks, query_heads),
        dtype=torch.float32,
        device=partial_out.device,
    )
    chunk_out_storage = chunk_out.view(-1)
    chunk_lse_storage = chunk_lse.view(-1)
    output = torch.empty(
        (query_heads, head_dim),
        dtype=torch.bfloat16,
        device=partial_out.device,
    )

    for tile_chunk, chunk_head in hl.tile(
        [merge_chunks, query_heads], block_size=[1, 1]
    ):
        chunk_split_idx = (
            tile_chunk.index[:, None] * splits_per_chunk
            + hl.arange(splits_per_chunk)[None, :]
        )
        chunk_lse_offsets = (
            chunk_split_idx[:, :, None] * query_heads + chunk_head.index[None, None, :]
        )
        chunk_lse_values = partial_lse_storage[chunk_lse_offsets]
        chunk_max_lse = torch.amax(chunk_lse_values, dim=1)
        chunk_weights = torch.exp2(chunk_lse_values - chunk_max_lse[:, None, :])
        chunk_value_offsets = (
            chunk_lse_offsets[:, :, :, None] * head_dim
            + hl.arange(head_dim)[None, None, None, :]
        )
        chunk_values = partial_out_storage[chunk_value_offsets]
        chunk_denominator = torch.sum(chunk_weights, dim=1)
        chunk_merged = torch.sum(chunk_values * chunk_weights[:, :, :, None], dim=1)
        chunk_merged = chunk_merged / chunk_denominator[:, :, None]
        chunk_out[tile_chunk, chunk_head, :] = chunk_merged
        chunk_lse[tile_chunk, chunk_head] = chunk_max_lse + torch.log2(
            chunk_denominator
        )

    for final_head in hl.tile(query_heads, block_size=1):
        final_chunk_idx = hl.arange(merge_chunks)
        final_lse_offsets = (
            final_chunk_idx[:, None] * query_heads + final_head.index[None, :]
        )
        final_lse_values = chunk_lse_storage[final_lse_offsets]
        final_max_lse = torch.amax(final_lse_values, dim=0)
        final_weights = torch.exp2(final_lse_values - final_max_lse[None, :])
        final_value_offsets = (
            final_lse_offsets[:, :, None] * head_dim
            + hl.arange(head_dim)[None, None, :]
        )
        final_values = chunk_out_storage[final_value_offsets]
        final_denominator = torch.sum(final_weights, dim=0)
        final_merged = torch.sum(final_values * final_weights[:, :, None], dim=0)
        output[final_head, :] = (final_merged / final_denominator[:, None]).to(
            output.dtype
        )
    return output


@helion.kernel(static_shapes=True, autotune_effort="none")
def task_aligned_per_token_group_fp8_quant(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool,
    dummy_is_tma_aligned: bool,
) -> None:
    """Quantize head-shaped attention without flattening its logical axes."""
    num_kv_heads, q_per_kv, head_dim = input.shape
    assert group_size == head_dim
    assert not scale_ue8m0
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(head_dim)
    hl.specialize(group_size)
    for tile_g, tile_q, tile_n in hl.tile(
        [num_kv_heads, q_per_kv, head_dim],
        block_size=[1, None, group_size],
    ):
        value = input[tile_g, tile_q, tile_n]
        scale = torch.amax(torch.abs(value), dim=-1).clamp(min=eps) / fp8_max
        flat_group = tile_g.index[:, None] * q_per_kv + tile_q.index[None, :]
        output_s[0, flat_group] = scale
        flat_n = flat_group[:, :, None] * group_size + tile_n.index[None, None, :]
        output_q[0, flat_n] = (
            (value / scale[:, :, None]).clamp(fp8_min, fp8_max).to(output_q.dtype)
        )


@helion.kernel(static_shapes=True, autotune_effort="none")
def canonical_tiled_merge_attention_splits(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    """The same merge tiles indexed through canonical multidimensional views."""
    splits, query_heads, head_dim = partial_out.shape
    hl.specialize(splits)
    hl.specialize(query_heads)
    hl.specialize(head_dim)
    merge_chunks = 16
    hl.specialize(merge_chunks)
    assert splits % merge_chunks == 0
    splits_per_chunk = splits // merge_chunks
    hl.specialize(splits_per_chunk)

    chunk_out = torch.empty(
        (merge_chunks, query_heads, head_dim),
        dtype=torch.float32,
        device=partial_out.device,
    )
    chunk_lse = torch.empty(
        (merge_chunks, query_heads),
        dtype=torch.float32,
        device=partial_out.device,
    )
    output = torch.empty(
        (query_heads, head_dim),
        dtype=torch.bfloat16,
        device=partial_out.device,
    )

    for tile_chunk, chunk_head in hl.tile(
        [merge_chunks, query_heads], block_size=[1, 1]
    ):
        chunk_split_idx = (
            tile_chunk.index[:, None] * splits_per_chunk
            + hl.arange(splits_per_chunk)[None, :]
        )
        chunk_lse_values = partial_lse[
            chunk_split_idx[:, :, None],
            chunk_head.index[None, None, :],
        ]
        chunk_max_lse = torch.amax(chunk_lse_values, dim=1)
        chunk_weights = torch.exp2(chunk_lse_values - chunk_max_lse[:, None, :])
        chunk_values = partial_out[
            chunk_split_idx[:, :, None, None],
            chunk_head.index[None, None, :, None],
            hl.arange(head_dim)[None, None, None, :],
        ]
        chunk_denominator = torch.sum(chunk_weights, dim=1)
        chunk_merged = torch.sum(chunk_values * chunk_weights[:, :, :, None], dim=1)
        chunk_merged = chunk_merged / chunk_denominator[:, :, None]
        chunk_out[tile_chunk, chunk_head, :] = chunk_merged
        chunk_lse[tile_chunk, chunk_head] = chunk_max_lse + torch.log2(
            chunk_denominator
        )

    for final_head in hl.tile(query_heads, block_size=1):
        final_chunk_idx = hl.arange(merge_chunks)
        final_lse_values = chunk_lse[
            final_chunk_idx[:, None],
            final_head.index[None, :],
        ]
        final_max_lse = torch.amax(final_lse_values, dim=0)
        final_weights = torch.exp2(final_lse_values - final_max_lse[None, :])
        final_values = chunk_out[
            final_chunk_idx[:, None, None],
            final_head.index[None, :, None],
            hl.arange(head_dim)[None, None, :],
        ]
        final_denominator = torch.sum(final_weights, dim=0)
        final_merged = torch.sum(final_values * final_weights[:, :, None], dim=0)
        output[final_head, :] = (final_merged / final_denominator[:, None]).to(
            output.dtype
        )
    return output.view(1, query_heads, head_dim)


@helion.kernel(static_shapes=True, autotune_effort="none")
def task_aligned_tiled_merge_attention_splits(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    """Express both merge fan-ins as ordinary tile ranges."""
    splits, num_kv_heads, q_per_kv, head_dim = partial_out.shape
    hl.specialize(splits)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(head_dim)
    merge_chunks = 16
    hl.specialize(merge_chunks)
    assert splits % merge_chunks == 0
    splits_per_chunk = splits // merge_chunks
    hl.specialize(splits_per_chunk)

    chunk_out = torch.empty(
        (merge_chunks, num_kv_heads, q_per_kv, head_dim),
        dtype=torch.float32,
        device=partial_out.device,
    )
    chunk_lse = torch.empty(
        (merge_chunks, num_kv_heads, q_per_kv),
        dtype=torch.float32,
        device=partial_out.device,
    )
    output = torch.empty(
        (num_kv_heads, q_per_kv, head_dim),
        dtype=torch.bfloat16,
        device=partial_out.device,
    )

    for tile_split, tile_g, tile_q in hl.tile(
        [splits, num_kv_heads, q_per_kv],
        block_size=[splits_per_chunk, 1, None],
    ):
        chunk_lse_values = partial_lse[tile_split, tile_g, tile_q]
        chunk_max_lse = torch.amax(chunk_lse_values, dim=0)
        chunk_weights = torch.exp2(chunk_lse_values - chunk_max_lse[None, :, :])
        chunk_values = partial_out[tile_split, tile_g, tile_q, :]
        chunk_denominator = torch.sum(chunk_weights, dim=0)
        chunk_merged = torch.sum(chunk_values * chunk_weights[:, :, :, None], dim=0)
        chunk_out[tile_split.id, tile_g, tile_q, :] = (
            chunk_merged / chunk_denominator[:, :, None]
        )
        chunk_lse[tile_split.id, tile_g, tile_q] = chunk_max_lse + torch.log2(
            chunk_denominator
        )

    for tile_chunk, tile_g, tile_q in hl.tile(
        [merge_chunks, num_kv_heads, q_per_kv],
        block_size=[merge_chunks, 1, None],
    ):
        final_lse_values = chunk_lse[tile_chunk, tile_g, tile_q]
        final_max_lse = torch.amax(final_lse_values, dim=0)
        final_weights = torch.exp2(final_lse_values - final_max_lse[None, :, :])
        final_values = chunk_out[tile_chunk, tile_g, tile_q, :]
        final_denominator = torch.sum(final_weights, dim=0)
        final_merged = torch.sum(final_values * final_weights[:, :, :, None], dim=0)
        output[tile_g, tile_q, :] = (final_merged / final_denominator[:, :, None]).to(
            output.dtype
        )
    return output.view(1, num_kv_heads * q_per_kv, head_dim)


def _compile_granular_separate_kernel(kernel, kernel_args, args):
    """Compile an unchanged granular source body as its own Helion launch."""
    bound = kernel.bind(kernel_args)
    values = dict(bound.config_spec.default_config())
    values.update(
        {
            "num_warps": 1,
            "num_stages": args.kernel_stages,
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
        }
    )
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config, bound.compile_config(config)


def _build_helion_reference(args, tensors):
    """Build a separate-launch graph from the exact sources used above."""
    configs = copy.deepcopy(QWEN3_B200_CONFIG)
    initial_residual = tensors["residual"].clone()
    initial_kv_cache = tensors["kv_cache"].clone()
    rms_args = (
        tensors["pre_q"],
        tensors["hidden_states"],
        tensors["pre_weight"],
        tensors["pre_scale"],
        args.eps,
        None,
        tensors["residual"],
        args.group,
        False,
    )
    _, rms = _compile_granular_separate_kernel(
        tiled_rms_norm_per_block_quant, rms_args, args
    )
    rms(*rms_args)
    qkv_args = (
        tensors["pre_q"],
        tensors["pre_scale"],
        tensors["qkv_weight_q"],
        tensors["qkv_weight_scale"],
        args.group,
    )
    _, qkv_mm = compile_config(block_fp8_mm, qkv_args, configs["qkv_mm"])
    qkv = qkv_mm(*qkv_args)
    qk_args = (
        qkv,
        args.q_heads,
        args.kv_heads,
        args.kv_heads,
        args.head_dim,
        args.eps,
        tensors["q_weight"],
        tensors["k_weight"],
        tensors["cos_sin"],
        True,
        tensors["position"],
        -1,
    )
    qk_source = (
        flat_fused_qk_norm_rope if _USE_TASK_ALIGNED_ATTENTION else fused_qk_norm_rope
    )
    if _USE_TASK_ALIGNED_ATTENTION:
        _, qk = _compile_granular_separate_kernel(qk_source, qk_args, args)
        qk(*qk_args)
    else:
        _, qk = compile_config(qk_source, qk_args, configs["qk_norm_rope"])
        qk(*qk_args)
    key_begin = args.q_heads * args.head_dim
    qkv_width = (args.q_heads + 2 * args.kv_heads) * args.head_dim
    query = qkv[:, :key_begin].view(args.batch, args.q_heads, args.head_dim)
    key = qkv[:, key_begin : key_begin + args.kv_heads * args.head_dim].view(
        args.batch, args.kv_heads, args.head_dim
    )
    value = qkv[:, key_begin + args.kv_heads * args.head_dim : qkv_width].view(
        args.batch, args.kv_heads, args.head_dim
    )
    cache_args = (
        key,
        value,
        tensors["kv_cache"],
        tensors["slot_mapping"],
        args.block_size,
    )
    _, cache = _compile_granular_separate_kernel(
        tiled_reshape_and_cache_flash, cache_args, args
    )
    cache(*cache_args)
    split_args = (
        query,
        tensors["kv_cache"],
        tensors["block_table"],
        args.context,
        args.block_size,
        args.q_heads // args.kv_heads,
        args.attention_splits,
    )
    split_source = (
        canonical_paged_gqa_decode_attention_split
        if _USE_CANONICAL_ATTENTION_VIEWS
        else paged_gqa_decode_attention_split
    )
    _, split_kernel = compile_config(
        split_source, split_args, configs["decode_attention_split"]
    )
    partial_out, partial_lse = split_kernel(*split_args)
    merge_args = (partial_out, partial_lse)
    if _USE_CANONICAL_ATTENTION_VIEWS:
        merge_kernel = canonical_tiled_merge_attention_splits
    elif _USE_TASK_ALIGNED_ATTENTION:
        merge_kernel = task_aligned_tiled_merge_attention_splits
    else:
        merge_kernel = tiled_merge_attention_splits
    _, merge = _compile_granular_separate_kernel(merge_kernel, merge_args, args)
    attention = merge(*merge_args)
    quant_args = (
        attention.view(args.batch, args.hidden),
        tensors["attention_q"],
        tensors["attention_scale"],
        args.group,
        1e-10,
        -448.0,
        448.0,
        False,
        False,
        False,
    )
    _, attention_quant = compile_config(
        per_token_group_fp8_quant, quant_args, configs["attention_quant"]
    )
    attention_quant(*quant_args)
    o_args = (
        tensors["attention_q"],
        tensors["attention_scale"],
        tensors["o_weight_q"],
        tensors["o_weight_scale"],
        args.group,
    )
    _, o_mm = compile_config(block_fp8_mm, o_args, configs["o_mm"])
    attention_out = o_mm(*o_args)
    post_args = (
        tensors["ffn_q"],
        attention_out,
        tensors["post_weight"],
        tensors["ffn_scale"],
        args.eps,
        None,
        tensors["residual"],
        args.group,
        False,
    )
    rms(*post_args)
    w13_args = (
        tensors["ffn_q"],
        tensors["ffn_scale"],
        tensors["w13_q"],
        tensors["w13_scale"],
        args.group,
    )
    _, w13 = compile_config(block_fp8_mm, w13_args, FFN_CONFIGS["w13"])
    gate_up = w13(*w13_args)
    silu_args = (gate_up, args.group)
    _, silu = compile_config(
        silu_and_mul_per_block_quant, silu_args, FFN_CONFIGS["silu_quant"]
    )
    activation_q, activation_scale = silu(*silu_args)
    w2_args = (
        activation_q,
        activation_scale,
        tensors["w2_q"],
        tensors["w2_scale"],
        args.group,
    )
    _, w2 = compile_config(block_fp8_mm, w2_args, FFN_CONFIGS["w2"])
    output = w2(*w2_args)
    tensors["residual"].copy_(initial_residual)
    tensors["kv_cache"].copy_(initial_kv_cache)

    def launch():
        rms(*rms_args)
        local_qkv = qkv_mm(*qkv_args)
        qk(local_qkv, *qk_args[1:])
        local_query = local_qkv[:, :key_begin].view(
            args.batch, args.q_heads, args.head_dim
        )
        local_key = local_qkv[
            :, key_begin : key_begin + args.kv_heads * args.head_dim
        ].view(args.batch, args.kv_heads, args.head_dim)
        local_value = local_qkv[
            :, key_begin + args.kv_heads * args.head_dim : qkv_width
        ].view(args.batch, args.kv_heads, args.head_dim)
        cache(
            local_key,
            local_value,
            tensors["kv_cache"],
            tensors["slot_mapping"],
            args.block_size,
        )
        local_partials, local_lse = split_kernel(
            local_query,
            tensors["kv_cache"],
            tensors["block_table"],
            args.context,
            args.block_size,
            args.q_heads // args.kv_heads,
            args.attention_splits,
        )
        local_attention = merge(local_partials, local_lse)
        attention_quant(
            local_attention.view(args.batch, args.hidden),
            tensors["attention_q"],
            tensors["attention_scale"],
            args.group,
            1e-10,
            -448.0,
            448.0,
            False,
            False,
            False,
        )
        local_attention_out = o_mm(*o_args)
        rms(
            tensors["ffn_q"],
            local_attention_out,
            tensors["post_weight"],
            tensors["ffn_scale"],
            args.eps,
            None,
            tensors["residual"],
            args.group,
            False,
        )
        local_gate = w13(*w13_args)
        local_activation_q, local_activation_scale = silu(local_gate, args.group)
        local_output = w2(
            local_activation_q,
            local_activation_scale,
            tensors["w2_q"],
            tensors["w2_scale"],
            args.group,
        )
        return (
            local_output,
            local_qkv,
            local_partials,
            local_lse,
            local_attention,
            local_attention_out,
            local_gate,
            local_activation_q,
            local_activation_scale,
        )

    return (
        launch,
        {
            "output": output,
            "qkv": qkv,
            "partial_out": partial_out,
            "partial_lse": partial_lse,
            "attention": attention,
            "attention_out": attention_out,
            "gate_up": gate_up,
            "activation_q": activation_q,
            "activation_scale": activation_scale,
        },
    )


def _probe_config(bound, args):
    """Map the retained one-warp probe geometry onto the granular source."""
    values = dict(bound.config_spec.default_config())
    values.pop(CROSS_LOOP_NUM_WORKERS_CONFIG, None)
    uses_flat_qk = _USE_TASK_ALIGNED_ATTENTION or _USE_CANONICAL_ATTENTION_VIEWS
    downstream_shift = (
        2
        if _USE_TASK_ALIGNED_ATTENTION
        else -1
        if _USE_CANONICAL_ATTENTION_VIEWS
        else 0
    )
    block_size_by_id = {
        7: 8,  # QKV output tile
        (16 if uses_flat_qk else 17): 4,
        (18 if uses_flat_qk else 19): args.attention_context_block,
        24 + downstream_shift: (
            args.merge_q_block if _USE_TASK_ALIGNED_ATTENTION else 1
        ),
        27 + downstream_shift: 8,  # O output tile
        36 + downstream_shift: 16,  # W13 output tile
        41 + downstream_shift: 8,  # W2 output tile
    }
    if not _USE_TASK_ALIGNED_ATTENTION:
        block_size_by_id[10] = args.qk_head_block
    if _USE_TASK_ALIGNED_ATTENTION:
        block_size_by_id[21] = args.merge_q_block
        block_size_by_id[24] = args.merge_q_block
    values["block_sizes"] = [
        block_size_by_id.get(spec.block_id, default)
        for spec, default in zip(
            bound.config_spec.block_sizes, values["block_sizes"], strict=True
        )
    ]
    loop_orders = [
        [0, 1],
        [0, 1, 2],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2],
        [2, 1, 0],
        [0, 1],
        [0, 1, 2],
        [1, 0],
        [0, 1],
        [0, 1, 2],
        [0, 1],
        [0, 1],
        [0, 1],
    ]
    if uses_flat_qk:
        loop_orders[3] = [0, 1]
    if _USE_TASK_ALIGNED_ATTENTION:
        loop_orders[6] = [0, 1, 2]
        loop_orders.insert(7, [0, 1, 2])
    values["loop_orders"] = loop_orders
    values["l2_groupings"] = [1] * len(bound.config_spec.l2_groupings)

    def by_block_id(specs, choices, default):
        return [
            next(
                (
                    choices[block_id]
                    for block_id in spec.block_ids
                    if block_id in choices
                ),
                default,
            )
            for spec in specs
        ]

    qkv_range = 8
    attention_range = 18 if uses_flat_qk else 19
    projection_ranges = {
        qkv_range: 4,
        28 + downstream_shift: 4,
        37 + downstream_shift: 4,
        42 + downstream_shift: 4,
    }
    values["range_num_stages"] = by_block_id(
        bound.config_spec.range_num_stages, projection_ranges, 0
    )
    values["range_unroll_factors"] = by_block_id(
        bound.config_spec.range_unroll_factors,
        {
            qkv_range: 2,
            28 + downstream_shift: 2,
            37 + downstream_shift: 2,
            42 + downstream_shift: 4,
        },
        0,
    )
    values["range_multi_buffers"] = by_block_id(
        bound.config_spec.range_multi_buffers,
        {
            qkv_range: True,
            attention_range: True,
            28 + downstream_shift: False,
            37 + downstream_shift: True,
            42 + downstream_shift: False,
        },
        None,
    )
    values["range_flattens"] = by_block_id(
        bound.config_spec.range_flattens,
        {
            qkv_range: False,
            attention_range: True,
            28 + downstream_shift: False,
            37 + downstream_shift: False,
            42 + downstream_shift: True,
        },
        None,
    )
    values.update(
        {
            "num_warps": 1,
            "num_stages": args.kernel_stages,
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
        }
    )
    if CROSS_LOOP_NUM_WORKERS_CONFIG in bound.config_spec.user_defined_tunables:
        values[CROSS_LOOP_NUM_WORKERS_CONFIG] = 1024
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


@dataclasses.dataclass(frozen=True)
class _Invocation:
    prefix: str
    kernel: _KernelWithFunction
    arguments: dict[str, str]
    outputs: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class _Bridge:
    source: str


class _KernelWithFunction(Protocol):
    fn: object


class _AssignedNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)


class _RenameNames(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.Name:
        renamed = self.names.get(node.id)
        if renamed is None:
            return node
        return ast.copy_location(ast.Name(id=renamed, ctx=node.ctx), node)


def _kernel_function_ast(kernel: _KernelWithFunction) -> ast.FunctionDef:
    function = kernel.fn
    source = textwrap.dedent(inspect.getsource(function))
    module = ast.parse(source)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    assert len(functions) == 1
    return functions[0]


def _inline_invocation(invocation: _Invocation) -> tuple[list[ast.stmt], list[ast.For]]:
    function = _kernel_function_ast(invocation.kernel)
    parameters = [argument.arg for argument in function.args.args]
    assert set(parameters) == set(invocation.arguments)

    assigned = _AssignedNames()
    for statement in function.body:
        assigned.visit(statement)
    rename = {
        name: invocation.outputs.get(name, f"__td_{invocation.prefix}_{name}")
        for name in set(parameters) | assigned.names
    }
    transformer = _RenameNames(rename)

    preamble: list[ast.stmt] = []
    for parameter in parameters:
        preamble.append(
            ast.Assign(
                targets=[ast.Name(id=rename[parameter], ctx=ast.Store())],
                value=ast.parse(invocation.arguments[parameter], mode="eval").body,
            )
        )

    loops: list[ast.For] = []
    for statement in function.body:
        if isinstance(statement, ast.Return):
            continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        transformed = transformer.visit(ast.fix_missing_locations(statement))
        assert isinstance(transformed, ast.stmt)
        if isinstance(transformed, ast.For):
            loops.append(transformed)
        else:
            preamble.append(transformed)
    assert loops, invocation.prefix
    return preamble, loops


def _compose_qwen3_layer_source() -> str:
    qkv_width = "(q_heads + 2 * kv_heads) * head_dim"
    events: list[_Invocation | _Bridge] = [
        _Invocation(
            "pre",
            rms_norm_per_block_quant,
            {
                "result": "pre_q",
                "input": "hidden_states",
                "weight": "pre_weight",
                "scale": "pre_scale",
                "epsilon": "eps",
                "scale_ub": "None",
                "residual": "residual",
                "group_size": "group",
                "is_scale_transposed": "False",
            },
        ),
        _Invocation(
            "qkv_mm",
            block_fp8_mm,
            {
                "activation_q": "pre_q",
                "activation_scale": "pre_scale",
                "weight_q": "qkv_weight_q",
                "weight_scale": "qkv_weight_scale",
                "group_size": "group",
            },
            {"out": "qkv"},
        ),
        _Bridge(
            f"""
            batch = hidden_states.shape[0]
            query = qkv[:, : q_heads * head_dim].view(batch, q_heads, head_dim)
            key_begin = q_heads * head_dim
            key = qkv[:, key_begin : key_begin + kv_heads * head_dim].view(
                batch, kv_heads, head_dim
            )
            value = qkv[:, key_begin + kv_heads * head_dim : {qkv_width}].view(
                batch, kv_heads, head_dim
            )
            """
        ),
        _Invocation(
            "qk",
            fused_qk_norm_rope,
            {
                "qkv": "qkv",
                "num_heads_q": "q_heads",
                "num_heads_k": "kv_heads",
                "num_heads_v": "kv_heads",
                "head_dim": "head_dim",
                "eps": "eps",
                "q_weight": "q_weight",
                "k_weight": "k_weight",
                "cos_sin_cache": "cos_sin",
                "is_neox": "True",
                "position_ids": "position",
                "forced_token_heads_per_warp": "-1",
            },
        ),
        _Invocation(
            "cache",
            reshape_and_cache_flash,
            {
                "key": "key",
                "value": "value",
                "kv_cache": "kv_cache",
                "slot_mapping": "slot_mapping",
                "block_size": "cache_block",
            },
        ),
        _Invocation(
            "attention_split",
            paged_gqa_decode_attention_split,
            {
                "query": "query",
                "kv_cache": "kv_cache",
                "block_table": "block_table",
                "context": "context",
                "block_size": "cache_block",
                "q_per_kv": "q_heads // kv_heads",
                "splits": "attention_splits",
            },
            {"partial_out": "partial_out", "partial_lse": "partial_lse"},
        ),
        _Invocation(
            "attention_merge",
            merge_attention_splits,
            {"partial_out": "partial_out", "partial_lse": "partial_lse"},
            {"output": "attention"},
        ),
        _Bridge("attention_flat = attention.view(batch, hidden)"),
        _Invocation(
            "attention_quant",
            per_token_group_fp8_quant,
            {
                "input": "attention_flat",
                "output_q": "attention_q",
                "output_s": "attention_scale",
                "group_size": "group",
                "eps": "1e-10",
                "fp8_min": "FP8_MIN",
                "fp8_max": "FP8_MAX",
                "scale_ue8m0": "False",
                "dummy_is_scale_transposed": "False",
                "dummy_is_tma_aligned": "False",
            },
        ),
        _Invocation(
            "o_mm",
            block_fp8_mm,
            {
                "activation_q": "attention_q",
                "activation_scale": "attention_scale",
                "weight_q": "o_weight_q",
                "weight_scale": "o_weight_scale",
                "group_size": "group",
            },
            {"out": "attention_out"},
        ),
        _Invocation(
            "post",
            rms_norm_per_block_quant,
            {
                "result": "ffn_q",
                "input": "attention_out",
                "weight": "post_weight",
                "scale": "ffn_scale",
                "epsilon": "eps",
                "scale_ub": "None",
                "residual": "residual",
                "group_size": "group",
                "is_scale_transposed": "False",
            },
        ),
        _Invocation(
            "w13",
            block_fp8_mm,
            {
                "activation_q": "ffn_q",
                "activation_scale": "ffn_scale",
                "weight_q": "w13_q",
                "weight_scale": "w13_scale",
                "group_size": "group",
            },
            {"out": "gate_up"},
        ),
        _Invocation(
            "activation",
            silu_and_mul_per_block_quant,
            {"gate_up": "gate_up", "group_size": "group"},
            {
                "activation_q": "activation_q",
                "activation_scale": "activation_scale",
            },
        ),
        _Invocation(
            "w2",
            block_fp8_mm,
            {
                "activation_q": "activation_q",
                "activation_scale": "activation_scale",
                "weight_q": "w2_q",
                "weight_scale": "w2_scale",
                "group_size": "group",
            },
            {"out": "output"},
        ),
    ]

    preamble: list[ast.stmt] = []
    loops: list[ast.For] = []
    for event in events:
        if isinstance(event, _Bridge):
            preamble.extend(ast.parse(textwrap.dedent(event.source)).body)
        else:
            event_preamble, event_loops = _inline_invocation(event)
            preamble.extend(event_preamble)
            loops.extend(event_loops)

    arguments = [
        "hidden_states",
        "residual",
        "pre_weight",
        "pre_q",
        "pre_scale",
        "qkv_weight_q",
        "qkv_weight_scale",
        "q_weight",
        "k_weight",
        "cos_sin",
        "position",
        "kv_cache",
        "block_table",
        "slot_mapping",
        "o_weight_q",
        "o_weight_scale",
        "attention_q",
        "attention_scale",
        "post_weight",
        "ffn_q",
        "ffn_scale",
        "w13_q",
        "w13_scale",
        "w2_q",
        "w2_scale",
        "hidden",
        "intermediate",
        "q_heads",
        "kv_heads",
        "head_dim",
        "context",
        "cache_block",
        "attention_splits",
        "group",
        "eps",
    ]
    result_names = [
        "output",
        "pre_q",
        "pre_scale",
        "qkv",
        "partial_out",
        "partial_lse",
        "attention",
        "attention_q",
        "attention_scale",
        "attention_out",
        "ffn_q",
        "ffn_scale",
        "gate_up",
        "activation_q",
        "activation_scale",
        "residual",
    ]
    function = ast.FunctionDef(
        name="qwen3_layer_tile_dependency_source",
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
                    elts=[ast.Name(id=name, ctx=ast.Load()) for name in result_names],
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    return ast.unparse(module) + "\n"


def _build_composite_kernel():
    source = _compose_qwen3_layer_source()
    filename = str(Path(__file__).with_name("_generated_qwen3_tile_dependency.py"))
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = globals()
    exec(compile(source, filename, "exec"), namespace)
    function = namespace["qwen3_layer_tile_dependency_source"]
    return helion.kernel(static_shapes=True, autotune_effort="none")(function), source


def _composite_args(tensors, args):
    return (
        tensors["hidden_states"],
        tensors["residual"],
        tensors["pre_weight"],
        tensors["pre_q"],
        tensors["pre_scale"],
        tensors["qkv_weight_q"],
        tensors["qkv_weight_scale"],
        tensors["q_weight"],
        tensors["k_weight"],
        tensors["cos_sin"],
        tensors["position"],
        tensors["kv_cache"],
        tensors["block_table"],
        tensors["slot_mapping"],
        tensors["o_weight_q"],
        tensors["o_weight_scale"],
        tensors["attention_q"],
        tensors["attention_scale"],
        tensors["post_weight"],
        tensors["ffn_q"],
        tensors["ffn_scale"],
        tensors["w13_q"],
        tensors["w13_scale"],
        tensors["w2_q"],
        tensors["w2_scale"],
        args.hidden,
        args.intermediate,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.context,
        args.block_size,
        args.attention_splits,
        args.group,
        args.eps,
    )


allocate_layer = allocate


def build_helion_reference(
    args, tensors
) -> tuple[Callable[[], object], dict[str, torch.Tensor]]:
    return _build_helion_reference(args, tensors)


OUTPUT_NAMES = (
    "output",
    "pre_q",
    "pre_scale",
    "qkv",
    "partial_out",
    "partial_lse",
    "attention",
    "attention_q",
    "attention_scale",
    "attention_out",
    "ffn_q",
    "ffn_scale",
    "gate_up",
    "activation_q",
    "activation_scale",
    "residual",
)


def _function_arguments(function: ast.FunctionDef) -> set[str]:
    return {argument.arg for argument in function.args.args}


def _attribute_call(statement: ast.AST, attribute: str) -> ast.Call | None:
    for node in ast.walk(statement):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and node.func.attr == attribute
        ):
            return node
    return None


def _direct_call(statement: ast.stmt, function_name: str) -> ast.Call | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if isinstance(call.func, ast.Name) and call.func.id == function_name:
        return call
    return None


def _insert_group_reports(
    statements: list[ast.stmt],
    function_name: str,
    state_name: str,
    epoch_name: str,
    group_expression,
) -> None:
    index = 0
    while index < len(statements):
        statement = statements[index]
        call = _direct_call(statement, function_name)
        if call is not None:
            insert_at = index + 1
            if (
                insert_at < len(statements)
                and _attribute_call(statements[insert_at], "inline_asm_elementwise")
                is not None
            ):
                insert_at += 1
            group = group_expression(call.args[-1])
            statements.insert(
                insert_at,
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(
                            id="_qwen_layer_report_attention_group",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Name(id=state_name, ctx=ast.Load()),
                            ast.Name(id=epoch_name, ctx=ast.Load()),
                            group,
                            ast.Name(id="SCHED_BASE", ctx=ast.Load()),
                        ],
                        keywords=[],
                    )
                ),
            )
            index = insert_at + 1
            continue
        for field in ("body", "orelse"):
            child = getattr(statement, field, None)
            if isinstance(child, list):
                _insert_group_reports(
                    child,
                    function_name,
                    state_name,
                    epoch_name,
                    group_expression,
                )
        index += 1


def _make_qkv_report_helper(
    scheduled: ast.FunctionDef,
    qk_root: str,
    cache_roots: tuple[str, ...],
) -> ast.FunctionDef:
    helper = copy.deepcopy(scheduled)
    helper.name = "_qwen_layer_qkv_report"
    helper.args.args.append(
        ast.arg(
            arg="SCHED_BASE",
            annotation=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="constexpr",
                ctx=ast.Load(),
            ),
        )
    )
    state_name = next(
        argument.arg
        for argument in scheduled.args.args
        if "dependency_state" in argument.arg
    )
    epoch_name = scheduled.args.args[-1].arg

    def qk_group(task: ast.expr) -> ast.expr:
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="where",
                ctx=ast.Load(),
            ),
            args=[
                ast.Compare(
                    left=copy.deepcopy(task),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(value=32)],
                ),
                ast.BinOp(
                    left=copy.deepcopy(task),
                    op=ast.FloorDiv(),
                    right=ast.Constant(value=4),
                ),
                ast.BinOp(
                    left=copy.deepcopy(task),
                    op=ast.Sub(),
                    right=ast.Constant(value=32),
                ),
            ],
            keywords=[],
        )

    _insert_group_reports(
        helper.body,
        qk_root,
        state_name,
        epoch_name,
        qk_group,
    )
    for cache_root in cache_roots:
        _insert_group_reports(
            helper.body,
            cache_root,
            state_name,
            epoch_name,
            copy.deepcopy,
        )
    return helper


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    tolerances = {
        "output": (0.25, 5e-2),
        "qkv": (6e-2, 5e-2),
        "partial_out": (8e-2, 3e-2),
        "partial_lse": (8e-2, 3e-2),
        "attention": (8e-2, 3e-2),
        "attention_out": (0.125, 3e-2),
        "gate_up": (0.125, 3e-2),
        "activation_q": (64.0, 3e-2),
        "activation_scale": (2e-3, 3e-2),
    }
    tensor_name = name.removeprefix("static_").removeprefix("orchestrator_")
    atol, rtol = tolerances.get(tensor_name, (0.0, 0.0))
    actual_float = actual.view_as(expected).float()
    expected_float = expected.float()
    try:
        torch.testing.assert_close(
            actual_float,
            expected_float,
            atol=atol,
            rtol=rtol,
        )
    except AssertionError as error:
        difference = (actual_float - expected_float).abs()
        raise AssertionError(
            f"{name}: max_abs={difference.max().item()}, "
            f"mean_abs={difference.mean().item()}; {error}"
        ) from error


def _build_granular_kernel():
    global merge_attention_splits, reshape_and_cache_flash, rms_norm_per_block_quant

    rms_norm_per_block_quant = tiled_rms_norm_per_block_quant
    reshape_and_cache_flash = tiled_reshape_and_cache_flash
    merge_attention_splits = tiled_merge_attention_splits
    kernel, _ = _build_composite_kernel()
    return helion.kernel(static_shapes=True, autotune_effort="none")(kernel.fn)


def _tuned_config(bound, args):
    base = _probe_config(bound, args)
    values = dict(base)
    for index, spec in enumerate(bound.config_spec.range_num_stages):
        if 37 in spec.block_ids:
            values["range_num_stages"][index] = args.w13_stages
        elif 42 in spec.block_ids:
            values["range_num_stages"][index] = args.w2_stages
    for index, spec in enumerate(bound.config_spec.range_unroll_factors):
        if 37 in spec.block_ids:
            values["range_unroll_factors"][index] = args.w13_unroll
        elif 42 in spec.block_ids:
            values["range_unroll_factors"][index] = args.w2_unroll
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


VISIBLE_TASKS = 4992
ATTENTION_MERGE_KEYS = 128
ATTENTION_FINAL_TASKS = 32
ORIGINAL_STATE_WORDS = 8032
TASK_CURSOR_SLOT = ORIGINAL_STATE_WORDS
STARTED_SLOT = TASK_CURSOR_SLOT + 1
CANCELED_SLOT = STARTED_SLOT + 1
PROCESSED_SLOT = CANCELED_SLOT + 1
GRID_SEEN_BASE = PROCESSED_SLOT + 1
STARTED_SEEN_BASE = GRID_SEEN_BASE + VISIBLE_TASKS
TASK_SEEN_BASE = STARTED_SEEN_BASE + VISIBLE_TASKS
PIPE_BASE = TASK_SEEN_BASE + VISIBLE_TASKS
ATTENTION_GROUP_BASE = PIPE_BASE
MERGE_KEY_BASE = ATTENTION_GROUP_BASE + 8
MERGE_HEAD_BASE = MERGE_KEY_BASE + 128
STATE_WORDS = MERGE_HEAD_BASE + 32
DEFAULT_CLC_SCRATCH_BYTES = 12288

STAGE_TILE_SPECS = (
    ("pre_residual", 32, 0, False),
    ("pre_norm_quant", 32, 32, False),
    ("qkv", 768, 64, True),
    ("attention", 1024, 880, False),
    ("attention_merge", 512, 1904, True),
    ("attention_finalize_quant", 32, 2416, False),
    ("o_projection", 512, 2480, True),
    ("post_norm_quant", 32, 3024, False),
    ("gate_up", 1536, 3056, True),
    ("down", 512, 4688, True),
)

CLC_SOURCE = r"""
@triton.jit
def _qwen3_layer_load_acquire(address):
    return tl.inline_asm_elementwise(
        asm="ld.acquire.gpu.global.u32 $0, [$1];",
        constraints="=r,l",
        args=[address],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _qwen3_layer_sync_warp():
    lanes = tl.arange(0, 32)
    return tl.inline_asm_elementwise(
        asm="bar.warp.sync 0xffffffff; mov.u32 $0, $1;",
        constraints="=r,r",
        args=[lanes],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _qwen3_layer_wait_count(address, target):
    value = _qwen3_layer_load_acquire(address)
    while value < target:
        value = _qwen3_layer_load_acquire(address)
    return _qwen3_layer_sync_warp()


@triton.jit
def _qwen3_layer_report_attention_group(
    tile_dependency_state,
    epoch,
    group,
    PIPE_BASE: tl.constexpr,
):
    tl.atomic_add(
        tile_dependency_state + PIPE_BASE + group,
        1,
        sem="release",
        scope="gpu",
    )


@triton.jit
def _qwen3_layer_issue_first_cancel():
    return tl.inline_asm_elementwise(
        asm=r'''{
            .reg .pred leader;
            .reg .b32 response_addr, mbar_addr;
            .shared .align 16 .b8 qwen3_layer_clc_scratch[__CLC_SCRATCH_BYTES__];

            mov.u32 response_addr, qwen3_layer_clc_scratch;
            add.u32 mbar_addr, response_addr, 16;
            elect.sync _|leader, 0xffffffff;

            @leader mbarrier.init.shared::cta.b64 [mbar_addr], 1;
            @leader clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [response_addr], [mbar_addr];
            @leader mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [mbar_addr], 16;

            mov.u32 $0, response_addr;
        }''',
        constraints='=r',
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _qwen3_layer_reissue_cancel(response_addr):
    return tl.inline_asm_elementwise(
        asm=r'''{
            .reg .pred leader;
            .reg .b32 response_addr, mbar_addr;

            mov.u32 response_addr, $1;
            add.u32 mbar_addr, response_addr, 16;
            elect.sync _|leader, 0xffffffff;

            @leader fence.proxy.async.shared::cta;
            @leader clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [response_addr], [mbar_addr];
            @leader mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [mbar_addr], 16;
            mov.u32 $0, $1;
        }''',
        constraints='=r,r',
        args=[response_addr],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _qwen3_layer_wait_cancel(response_addr, parity):
    return tl.inline_asm_elementwise(
        asm=r'''{
            .reg .pred complete, canceled, leader, waiting;
            .reg .b32 response_addr, mbar_addr, success, canceled_x, parity;
            .reg .b128 response;

            mov.u32 response_addr, $2;
            add.u32 mbar_addr, response_addr, 16;
            mov.u32 parity, $3;
            elect.sync _|leader, 0xffffffff;
            mov.u32 success, 0;
            mov.u32 canceled_x, 0xffffffff;

        QWEN3_LAYER_CLC_WAIT:
            @leader mbarrier.try_wait.parity.relaxed.cta.shared.b64 complete, [mbar_addr], parity;
            and.pred waiting, leader, !complete;
            @waiting bra QWEN3_LAYER_CLC_WAIT;

            bar.warp.sync 0xffffffff;
            ld.shared.b128 response, [response_addr];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response;
            selp.u32 success, 1, 0, canceled;
            @canceled clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 canceled_x, response;
            bar.warp.sync 0xffffffff;

            mov.u32 $0, success;
            mov.u32 $1, canceled_x;
        }''',
        constraints='=r,=r,r,r',
        args=[response_addr, parity],
        dtype=(tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
"""


def _root_calls(statement: ast.stmt) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("tile_dependency_root_")
    ]


def _master_function(module: ast.Module) -> ast.FunctionDef:
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_helion_qwen3_layer_tile_dependency_source")
    )


def _stage_schedule(tile_bundle: int) -> tuple[tuple[object, ...], ...]:
    stages: list[tuple[object, ...]] = []
    command_start = 0
    tile_start = 0
    for name, tile_count, virtual_start, bundleable in STAGE_TILE_SPECS:
        bundle = tile_bundle if bundleable else 1
        if tile_count % bundle:
            raise ValueError(f"{name} tile count is not divisible by bundle {bundle}")
        command_count = tile_count // bundle
        stages.append(
            (
                name,
                command_start,
                command_start + command_count,
                virtual_start,
                tile_start,
                tile_count,
                bundle,
            )
        )
        command_start += command_count
        tile_start += tile_count
    if tile_start != VISIBLE_TASKS:
        raise AssertionError("stage tile counts no longer cover the layer")
    return tuple(stages)


def _attention_chain_commands(chained_merge_heads: int) -> int:
    return (
        VISIBLE_TASKS
        - chained_merge_heads * ATTENTION_MERGE_KEYS
        - ATTENTION_FINAL_TASKS
    )


def _task_statement_parts(
    statement: ast.stmt,
) -> tuple[list[ast.stmt], list[ast.stmt], list[ast.stmt]]:
    if not isinstance(statement, ast.If):
        raise RuntimeError("expected a guarded generated task statement")
    loops = [node for node in statement.body if isinstance(node, ast.For)]
    if len(loops) != 1:
        raise RuntimeError("expected exactly one generated virtual-tile loop")
    loop = loops[0]
    loop_index = statement.body.index(loop)
    prefix = copy.deepcopy(statement.body[:loop_index])
    loop_body = copy.deepcopy(loop.body)
    suffix = copy.deepcopy(statement.body[loop_index + 1 :])
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == "program_id"
        for statement_node in [*prefix, *loop_body, *suffix]
        for node in ast.walk(statement_node)
    ):
        raise RuntimeError("generated task body retained a physical program ID")
    return prefix, loop_body, suffix


class _ReplaceCall(ast.NodeTransformer):
    def __init__(self, old: str, new: str, extra_argument: ast.expr) -> None:
        self.old = old
        self.new = new
        self.extra_argument = extra_argument

    def visit_Call(self, node: ast.Call) -> ast.Call:
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == self.old:
            node.func.id = self.new
            node.args.append(copy.deepcopy(self.extra_argument))
        return node


class _ProgramIdToWorker(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and node.func.attr == "program_id"
        ):
            return ast.copy_location(ast.Name(id="worker", ctx=ast.Load()), node)
        return node


def _make_dispatcher(
    master: ast.FunctionDef,
    stages: tuple[tuple[object, ...], ...],
    qkv_scheduled_name: str,
    qkv_report_name: str,
    attention_order: str,
) -> ast.FunctionDef:
    task_statements = [statement for statement in master.body if _root_calls(statement)]
    if len(task_statements) != len(stages):
        raise RuntimeError(
            f"expected {len(stages)} scheduler-visible stages, "
            f"found {len(task_statements)}"
        )

    branch: ast.If | None = None
    first_branch: ast.If | None = None
    for stage, statement in zip(stages, task_statements, strict=True):
        (
            _,
            command_start,
            command_end,
            virtual_start,
            tile_start,
            _,
            bundle,
        ) = stage
        prefix, loop_body, suffix = _task_statement_parts(statement)
        tile_prefix: list[ast.stmt] = []
        if attention_order != "global" and stage[0] == "qkv":
            loop_body = [
                _ReplaceCall(
                    qkv_scheduled_name,
                    qkv_report_name,
                    ast.Constant(value=ATTENTION_GROUP_BASE),
                ).visit(node)
                for node in loop_body
            ]
        elif attention_order != "global" and stage[0] == "attention":
            prefix = []
            tile_prefix = ast.parse(
                f"""
attention_task = virtual_pid - 880
attention_group = attention_task % 8
_qwen3_layer_wait_count(
    tile_dependency_state + {ATTENTION_GROUP_BASE} + attention_group,
    6,
)
"""
            ).body
            suffix = ast.parse(
                f"""
attention_split = attention_task // 8
attention_chunk = attention_split // 8
merge_key = attention_group * 16 + attention_chunk
_qwen3_layer_sync_warp()
tl.atomic_add(
    tile_dependency_state + {MERGE_KEY_BASE} + merge_key,
    1,
    sem="release",
    scope="gpu",
)
"""
            ).body
        elif attention_order != "global" and stage[0] == "attention_merge":
            prefix = []
            tile_prefix = ast.parse(
                f"""
merge_task = virtual_pid - 1904
merge_chunk = merge_task % 16
merge_head = merge_task // 16
merge_key = (merge_head // 4) * 16 + merge_chunk
_qwen3_layer_wait_count(
    tile_dependency_state + {MERGE_KEY_BASE} + merge_key,
    8,
)
"""
            ).body
            suffix = ast.parse(
                f"""
_qwen3_layer_sync_warp()
tl.atomic_add(
    tile_dependency_state + {MERGE_HEAD_BASE} + merge_head,
    1,
    sem="release",
    scope="gpu",
)
"""
            ).body
        elif attention_order != "global" and stage[0] == "attention_finalize_quant":
            prefix = []
            tile_prefix = ast.parse(
                f"""
attention_head = virtual_pid - 2416
_qwen3_layer_wait_count(
    tile_dependency_state + {MERGE_HEAD_BASE} + attention_head,
    16,
)
"""
            ).body
        if attention_order == "staged-groups" and stage[0] == "qkv":
            assignments = ast.parse(
                f"""
qkv_task = logical_task - {command_start}
qkv_group = qkv_task // 96
qkv_in_group = qkv_task % 96
virtual_pid = tl.where(
    qkv_in_group < 64,
    64 + qkv_group * 64 + qkv_in_group,
    tl.where(
        qkv_in_group < 80,
        576 + qkv_group * 16 + qkv_in_group - 64,
        704 + qkv_group * 16 + qkv_in_group - 80,
    ),
)
logical_tile = virtual_pid
"""
            ).body
        elif attention_order == "staged-groups" and stage[0] == "attention":
            assignments = ast.parse(
                f"""
attention_order_task = logical_task - {command_start}
attention_order_group = attention_order_task // 128
attention_order_split = attention_order_task % 128
virtual_pid = 880 + attention_order_split * 8 + attention_order_group
logical_tile = 832 + attention_order_split * 8 + attention_order_group
"""
            ).body
        else:
            assignments = ast.parse(
                f"""
logical_tile = (
    {tile_start}
    + (logical_task - {command_start}) * {bundle}
    + bundle_offset
)
virtual_pid = (
    {virtual_start}
    + (logical_task - {command_start}) * {bundle}
    + bundle_offset
)
"""
            ).body
        per_tile = assignments
        per_tile.extend(tile_prefix)
        per_tile.extend(loop_body)
        per_tile.extend(suffix)
        per_tile.extend(
            ast.parse(
                f"""
if RECORD_STATS:
    tl.atomic_add(
        tile_dependency_state + {TASK_SEEN_BASE} + logical_tile,
        1,
        sem="relaxed",
        scope="gpu",
    )
    tl.atomic_add(
        tile_dependency_state + {PROCESSED_SLOT},
        1,
        sem="relaxed",
        scope="gpu",
    )
"""
            ).body
        )
        bundle_loop = ast.For(
            target=ast.Name(id="bundle_offset", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="tl", ctx=ast.Load()),
                    attr="static_range",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=0), ast.Constant(value=bundle)],
                keywords=[],
            ),
            body=per_tile,
            orelse=[],
            type_comment=None,
        )
        next_branch = ast.If(
            test=ast.Compare(
                left=ast.Name(id="logical_task", ctx=ast.Load()),
                ops=[ast.Lt()],
                comparators=[ast.Constant(value=command_end)],
            ),
            body=[*prefix, bundle_loop],
            orelse=[],
        )
        if first_branch is None:
            first_branch = next_branch
        else:
            assert branch is not None
            branch.orelse = [next_branch]
        branch = next_branch
    assert first_branch is not None

    arguments = copy.deepcopy(master.args)
    arguments.args.extend(
        [
            ast.arg(arg="logical_task"),
            ast.arg(arg="tile_dependency_epoch"),
            ast.arg(
                arg="RECORD_STATS",
                annotation=ast.Attribute(
                    value=ast.Name(id="tl", ctx=ast.Load()),
                    attr="constexpr",
                    ctx=ast.Load(),
                ),
            ),
        ]
    )
    return ast.FunctionDef(
        name="_qwen3_layer_run_logical_task",
        args=arguments,
        body=[first_branch],
        decorator_list=[
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="triton", ctx=ast.Load()),
                    attr="jit",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[ast.keyword(arg="noinline", value=ast.Constant(value=True))],
            )
        ],
        returns=None,
        type_comment=None,
    )


def _make_grouped_attention_dispatcher(
    master: ast.FunctionDef,
    qkv_scheduled_name: str,
    qkv_report_name: str,
) -> ast.FunctionDef:
    task_statements = [statement for statement in master.body if _root_calls(statement)]
    if len(task_statements) != len(STAGE_TILE_SPECS):
        raise RuntimeError("unexpected generated Qwen3 stage layout")

    def execution(
        stage_name: str,
        statement: ast.stmt,
        virtual_expression: str,
        logical_expression: str,
    ) -> list[ast.stmt]:
        prefix, loop_body, suffix = _task_statement_parts(statement)
        tile_prefix: list[ast.stmt] = []
        if stage_name == "qkv":
            loop_body = [
                _ReplaceCall(
                    qkv_scheduled_name,
                    qkv_report_name,
                    ast.Constant(value=ATTENTION_GROUP_BASE),
                ).visit(node)
                for node in loop_body
            ]
        elif stage_name == "attention":
            prefix = []
            tile_prefix = ast.parse(
                f"""
attention_task = virtual_pid - 880
attention_group = attention_task % 8
_qwen3_layer_wait_count(
    tile_dependency_state + {ATTENTION_GROUP_BASE} + attention_group,
    6,
)
"""
            ).body
            suffix = ast.parse(
                f"""
attention_split = attention_task // 8
attention_chunk = attention_split // 8
merge_key = attention_group * 16 + attention_chunk
_qwen3_layer_sync_warp()
tl.atomic_add(
    tile_dependency_state + {MERGE_KEY_BASE} + merge_key,
    1,
    sem="release",
    scope="gpu",
)
"""
            ).body
        elif stage_name == "attention_merge":
            prefix = []
            tile_prefix = ast.parse(
                f"""
merge_task = virtual_pid - 1904
merge_chunk = merge_task % 16
merge_head = merge_task // 16
merge_key = (merge_head // 4) * 16 + merge_chunk
_qwen3_layer_wait_count(
    tile_dependency_state + {MERGE_KEY_BASE} + merge_key,
    8,
)
"""
            ).body
            suffix = ast.parse(
                f"""
_qwen3_layer_sync_warp()
tl.atomic_add(
    tile_dependency_state + {MERGE_HEAD_BASE} + merge_head,
    1,
    sem="release",
    scope="gpu",
)
"""
            ).body
        elif stage_name == "attention_finalize_quant":
            prefix = []
            tile_prefix = ast.parse(
                f"""
attention_head = virtual_pid - 2416
_qwen3_layer_wait_count(
    tile_dependency_state + {MERGE_HEAD_BASE} + attention_head,
    16,
)
"""
            ).body
        assignments = ast.parse(
            f"""
logical_tile = {logical_expression}
virtual_pid = {virtual_expression}
"""
        ).body
        stats = ast.parse(
            f"""
if RECORD_STATS:
    tl.atomic_add(
        tile_dependency_state + {TASK_SEEN_BASE} + logical_tile,
        1,
        sem="relaxed",
        scope="gpu",
    )
    tl.atomic_add(
        tile_dependency_state + {PROCESSED_SLOT},
        1,
        sem="relaxed",
        scope="gpu",
    )
"""
        ).body
        return [*prefix, *assignments, *tile_prefix, *loop_body, *suffix, *stats]

    grouped_prefix = ast.parse(
        """
group = (logical_task - 64) // 292
group_task = (logical_task - 64) % 292
"""
    ).body
    qkv_mapping = ast.parse(
        """
qkv_in_group = group_task
qkv_virtual_pid = tl.where(
    qkv_in_group < 64,
    64 + group * 64 + qkv_in_group,
    tl.where(
        qkv_in_group < 80,
        576 + group * 16 + qkv_in_group - 64,
        704 + group * 16 + qkv_in_group - 80,
    ),
)
"""
    ).body
    grouped_qkv_body = [
        *qkv_mapping,
        *execution(
            "qkv",
            task_statements[2],
            "qkv_virtual_pid",
            "qkv_virtual_pid",
        ),
    ]
    grouped_attention_body = execution(
        "attention",
        task_statements[3],
        "880 + (group_task - 96) * 8 + group",
        "832 + (group_task - 96) * 8 + group",
    )
    grouped_merge_body = execution(
        "attention_merge",
        task_statements[4],
        "1904 + (group * 4 + (group_task - 224) // 16) * 16 + (group_task - 224) % 16",
        "1856 + (group * 4 + (group_task - 224) // 16) * 16 + (group_task - 224) % 16",
    )
    grouped_final_body = execution(
        "attention_finalize_quant",
        task_statements[5],
        "2416 + group * 4 + group_task - 288",
        "2368 + group * 4 + group_task - 288",
    )
    grouped_branch = ast.If(
        test=ast.Compare(
            left=ast.Name(id="group_task", ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[ast.Constant(value=96)],
        ),
        body=grouped_qkv_body,
        orelse=[
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="group_task", ctx=ast.Load()),
                    ops=[ast.Lt()],
                    comparators=[ast.Constant(value=224)],
                ),
                body=grouped_attention_body,
                orelse=[
                    ast.If(
                        test=ast.Compare(
                            left=ast.Name(id="group_task", ctx=ast.Load()),
                            ops=[ast.Lt()],
                            comparators=[ast.Constant(value=288)],
                        ),
                        body=grouped_merge_body,
                        orelse=grouped_final_body,
                    )
                ],
            )
        ],
    )

    branch_specs = (
        (
            32,
            execution(
                "pre_residual", task_statements[0], "logical_task", "logical_task"
            ),
        ),
        (
            64,
            execution(
                "pre_norm_quant", task_statements[1], "logical_task", "logical_task"
            ),
        ),
        (2400, [*grouped_prefix, grouped_branch]),
        (
            2912,
            execution(
                "o_projection",
                task_statements[6],
                "logical_task + 80",
                "logical_task",
            ),
        ),
        (
            2944,
            execution(
                "post_norm_quant",
                task_statements[7],
                "logical_task + 112",
                "logical_task",
            ),
        ),
        (
            4480,
            execution(
                "gate_up",
                task_statements[8],
                "logical_task + 112",
                "logical_task",
            ),
        ),
        (
            4992,
            execution(
                "down",
                task_statements[9],
                "logical_task + 208",
                "logical_task",
            ),
        ),
    )
    first_branch: ast.If | None = None
    branch: ast.If | None = None
    for end, body in branch_specs:
        next_branch = ast.If(
            test=ast.Compare(
                left=ast.Name(id="logical_task", ctx=ast.Load()),
                ops=[ast.Lt()],
                comparators=[ast.Constant(value=end)],
            ),
            body=body,
            orelse=[],
        )
        if first_branch is None:
            first_branch = next_branch
        else:
            assert branch is not None
            branch.orelse = [next_branch]
        branch = next_branch
    assert first_branch is not None

    arguments = copy.deepcopy(master.args)
    constexpr = ast.Attribute(
        value=ast.Name(id="tl", ctx=ast.Load()),
        attr="constexpr",
        ctx=ast.Load(),
    )
    arguments.args.extend(
        [
            ast.arg(arg="logical_task"),
            ast.arg(arg="tile_dependency_epoch"),
            ast.arg(arg="RECORD_STATS", annotation=constexpr),
        ]
    )
    return ast.FunctionDef(
        name="_qwen3_layer_run_logical_task",
        args=arguments,
        body=[first_branch],
        decorator_list=[
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="triton", ctx=ast.Load()),
                    attr="jit",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[ast.keyword(arg="noinline", value=ast.Constant(value=True))],
            )
        ],
        returns=None,
        type_comment=None,
    )


def _make_attention_chain_dispatcher(
    master: ast.FunctionDef,
    qkv_scheduled_name: str,
    qkv_report_name: str,
    chained_merge_heads: int,
) -> ast.FunctionDef:
    if not 1 <= chained_merge_heads <= 2:
        raise ValueError("chained merge heads must be 1 or 2")
    task_statements = [statement for statement in master.body if _root_calls(statement)]
    if len(task_statements) != len(STAGE_TILE_SPECS):
        raise RuntimeError("unexpected generated Qwen3 stage layout")

    def stats(logical_expression: str) -> list[ast.stmt]:
        return ast.parse(
            f"""
if RECORD_STATS:
    tl.atomic_add(
        tile_dependency_state + {TASK_SEEN_BASE} + ({logical_expression}),
        1,
        sem="relaxed",
        scope="gpu",
    )
    tl.atomic_add(
        tile_dependency_state + {PROCESSED_SLOT},
        1,
        sem="relaxed",
        scope="gpu",
    )
"""
        ).body

    def simple_execution(
        stage_index: int,
        virtual_expression: str,
        logical_expression: str,
        replace_qkv: bool = False,
    ) -> list[ast.stmt]:
        prefix, loop_body, suffix = _task_statement_parts(task_statements[stage_index])
        if replace_qkv:
            loop_body = [
                _ReplaceCall(
                    qkv_scheduled_name,
                    qkv_report_name,
                    ast.Constant(value=ATTENTION_GROUP_BASE),
                ).visit(node)
                for node in loop_body
            ]
        assignments = ast.parse(
            f"""
virtual_pid = {virtual_expression}
logical_tile = {logical_expression}
"""
        ).body
        return [
            *prefix,
            *assignments,
            *loop_body,
            *suffix,
            *stats("logical_tile"),
        ]

    _, attention_body, _ = _task_statement_parts(task_statements[3])
    _, merge_body, _ = _task_statement_parts(task_statements[4])
    _, final_body, _ = _task_statement_parts(task_statements[5])
    attention_execution = [
        *ast.parse(
            f"""
logical_tile = logical_task
virtual_pid = logical_task + 48
attention_task = virtual_pid - 880
attention_group = attention_task % 8
_qwen3_layer_wait_count(
    tile_dependency_state + {ATTENTION_GROUP_BASE} + attention_group,
    6,
)
"""
        ).body,
        *attention_body,
        *ast.parse(
            f"""
attention_split = attention_task // 8
attention_chunk = attention_split // 8
merge_key = attention_group * 16 + attention_chunk
_qwen3_layer_sync_warp()
merge_previous = tl.atomic_add(
    tile_dependency_state + {MERGE_KEY_BASE} + merge_key,
    1,
    sem="acq_rel",
    scope="gpu",
)
"""
        ).body,
    ]
    chained_merge_body = [
        *ast.parse(
            """
merge_head = attention_group * 4 + head_in_group
virtual_pid = 1904 + merge_head * 16 + attention_chunk
merge_logical_tile = 1856 + merge_head * 16 + attention_chunk
"""
        ).body,
        *merge_body,
        *ast.parse(
            f"""
_qwen3_layer_sync_warp()
head_previous = tl.atomic_add(
    tile_dependency_state + {MERGE_HEAD_BASE} + merge_head,
    1,
    sem="acq_rel",
    scope="gpu",
)
"""
        ).body,
    ]
    final_if_body = [
        *ast.parse(
            """
virtual_pid = 2416 + merge_head
final_logical_tile = 2368 + merge_head
"""
        ).body,
        *final_body,
        *stats("final_logical_tile"),
    ]
    chained_merge_body.append(
        ast.If(
            test=ast.Compare(
                left=ast.Name(id="head_previous", ctx=ast.Load()),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=15)],
            ),
            body=final_if_body,
            orelse=[],
        )
    )
    chained_merge_body.extend(stats("merge_logical_tile"))
    if chained_merge_heads == 1:
        chained_attention_body = [
            *ast.parse("head_in_group = 0").body,
            *chained_merge_body,
        ]
    else:
        chained_attention_body = [
            ast.For(
                target=ast.Name(id="head_in_group", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="tl", ctx=ast.Load()),
                        attr="static_range",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Constant(value=0),
                        ast.Constant(value=chained_merge_heads),
                    ],
                    keywords=[],
                ),
                body=chained_merge_body,
                orelse=[],
                type_comment=None,
            )
        ]
    attention_execution.append(
        ast.If(
            test=ast.Compare(
                left=ast.Name(id="merge_previous", ctx=ast.Load()),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=7)],
            ),
            body=chained_attention_body,
            orelse=[],
        )
    )
    attention_execution.extend(stats("logical_tile"))

    queued_merge_heads = 4 - chained_merge_heads
    queued_merge_tasks = ATTENTION_MERGE_KEYS * queued_merge_heads
    queued_merge_end = 1856 + queued_merge_tasks
    queued_merge_execution = [
        *ast.parse(
            f"""
queued_merge_task = logical_task - 1856
attention_group = queued_merge_task // {queued_merge_heads * 16}
merge_in_group = queued_merge_task % {queued_merge_heads * 16}
merge_head = attention_group * 4 + {chained_merge_heads} + merge_in_group // 16
attention_chunk = merge_in_group % 16
merge_key = attention_group * 16 + attention_chunk
_qwen3_layer_wait_count(
    tile_dependency_state + {MERGE_KEY_BASE} + merge_key,
    8,
)
virtual_pid = 1904 + merge_head * 16 + attention_chunk
merge_logical_tile = 1856 + merge_head * 16 + attention_chunk
"""
        ).body,
        *copy.deepcopy(merge_body),
        *ast.parse(
            f"""
_qwen3_layer_sync_warp()
head_previous = tl.atomic_add(
    tile_dependency_state + {MERGE_HEAD_BASE} + merge_head,
    1,
    sem="acq_rel",
    scope="gpu",
)
"""
        ).body,
        ast.If(
            test=ast.Compare(
                left=ast.Name(id="head_previous", ctx=ast.Load()),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=15)],
            ),
            body=copy.deepcopy(final_if_body),
            orelse=[],
        ),
        *stats("merge_logical_tile"),
    ]

    removed_tasks = chained_merge_heads * ATTENTION_MERGE_KEYS + ATTENTION_FINAL_TASKS
    o_projection_start = queued_merge_end
    post_norm_start = o_projection_start + 512
    gate_up_start = post_norm_start + 32
    down_start = gate_up_start + 1536
    total_commands = down_start + 512

    branch_specs = (
        (32, simple_execution(0, "logical_task", "logical_task")),
        (64, simple_execution(1, "logical_task", "logical_task")),
        (832, simple_execution(2, "logical_task", "logical_task", True)),
        (1856, attention_execution),
        (queued_merge_end, queued_merge_execution),
        (
            post_norm_start,
            simple_execution(
                6,
                f"logical_task + {2480 - o_projection_start}",
                f"logical_task + {removed_tasks}",
            ),
        ),
        (
            gate_up_start,
            simple_execution(
                7,
                f"logical_task + {3024 - post_norm_start}",
                f"logical_task + {removed_tasks}",
            ),
        ),
        (
            down_start,
            simple_execution(
                8,
                f"logical_task + {3056 - gate_up_start}",
                f"logical_task + {removed_tasks}",
            ),
        ),
        (
            total_commands,
            simple_execution(
                9,
                f"logical_task + {4688 - down_start}",
                f"logical_task + {removed_tasks}",
            ),
        ),
    )
    first_branch: ast.If | None = None
    branch: ast.If | None = None
    for end, body in branch_specs:
        next_branch = ast.If(
            test=ast.Compare(
                left=ast.Name(id="logical_task", ctx=ast.Load()),
                ops=[ast.Lt()],
                comparators=[ast.Constant(value=end)],
            ),
            body=body,
            orelse=[],
        )
        if first_branch is None:
            first_branch = next_branch
        else:
            assert branch is not None
            branch.orelse = [next_branch]
        branch = next_branch
    assert first_branch is not None

    arguments = copy.deepcopy(master.args)
    constexpr = ast.Attribute(
        value=ast.Name(id="tl", ctx=ast.Load()),
        attr="constexpr",
        ctx=ast.Load(),
    )
    arguments.args.extend(
        [
            ast.arg(arg="logical_task"),
            ast.arg(arg="tile_dependency_epoch"),
            ast.arg(arg="RECORD_STATS", annotation=constexpr),
        ]
    )
    return ast.FunctionDef(
        name="_qwen3_layer_run_logical_task",
        args=arguments,
        body=[first_branch],
        decorator_list=[
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="triton", ctx=ast.Load()),
                    attr="jit",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[ast.keyword(arg="noinline", value=ast.Constant(value=True))],
            )
        ],
        returns=None,
        type_comment=None,
    )


def _make_clc_kernel(
    master: ast.FunctionDef,
    total_commands: int,
    task_claim: str,
) -> ast.FunctionDef:
    arguments = copy.deepcopy(master.args)
    constexpr = ast.Attribute(
        value=ast.Name(id="tl", ctx=ast.Load()),
        attr="constexpr",
        ctx=ast.Load(),
    )
    arguments.args.append(
        ast.arg(arg="RECORD_STATS", annotation=copy.deepcopy(constexpr))
    )
    argument_names = [argument.arg for argument in master.args.args]
    dispatch_call = ast.Expr(
        value=ast.Call(
            func=ast.Name(id="_qwen3_layer_run_logical_task", ctx=ast.Load()),
            args=[
                *[ast.Name(id=name, ctx=ast.Load()) for name in argument_names],
                ast.Name(id="logical_task", ctx=ast.Load()),
                ast.Name(id="tile_dependency_epoch", ctx=ast.Load()),
                ast.Name(id="RECORD_STATS", ctx=ast.Load()),
            ],
            keywords=[],
        )
    )
    if task_claim == "prefix":
        initial_claim = "logical_task = launch_token.to(tl.int32)"
    else:
        initial_claim = f"""logical_task = tl.atomic_add(
    tile_dependency_state + {TASK_CURSOR_SLOT},
    1,
    sem=\"relaxed\",
    scope=\"gpu\",
).to(tl.int32)"""
    template = ast.parse(
        f"""
launch_token = tl.program_id(0).to(tl.uint32)
if RECORD_STATS:
    tl.atomic_add(
        tile_dependency_state + {STARTED_SLOT},
        1,
        sem="relaxed",
        scope="gpu",
    )
    tl.atomic_add(
        tile_dependency_state + {GRID_SEEN_BASE} + launch_token,
        1,
        sem="relaxed",
        scope="gpu",
    )
    tl.atomic_add(
        tile_dependency_state + {STARTED_SEEN_BASE} + launch_token,
        1,
        sem="relaxed",
        scope="gpu",
    )
response_addr = _qwen3_layer_issue_first_cancel()
phase = tl.full([], 0, tl.uint32)
active = tl.full([], True, tl.int1)
tile_dependency_epoch = tl.full([], 1, tl.uint32)
{initial_claim}
while active:
    if logical_task < {total_commands}:
        pass
    success, canceled_token = _qwen3_layer_wait_cancel(response_addr, phase)
    phase = 1 - phase
    if success != 0:
        if RECORD_STATS:
            tl.atomic_add(
                tile_dependency_state + {CANCELED_SLOT},
                1,
                sem="relaxed",
                scope="gpu",
            )
            tl.atomic_add(
                tile_dependency_state + {GRID_SEEN_BASE} + canceled_token,
                1,
                sem="relaxed",
                scope="gpu",
            )
        _qwen3_layer_reissue_cancel(response_addr)
        logical_task = tl.atomic_add(
            tile_dependency_state + {TASK_CURSOR_SLOT},
            1,
            sem="relaxed",
            scope="gpu",
        ).to(tl.int32)
    else:
        active = False
"""
    ).body
    task_guard = next(
        node
        for node in template
        if isinstance(node, ast.While)
        for node in node.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(child, ast.Name) and child.id == "logical_task"
            for child in ast.walk(node.test)
        )
    )
    task_guard.body[0] = dispatch_call
    return ast.FunctionDef(
        name="qwen3_layer_clc_kernel",
        args=arguments,
        body=template,
        decorator_list=[
            ast.Attribute(
                value=ast.Name(id="triton", ctx=ast.Load()),
                attr="jit",
                ctx=ast.Load(),
            )
        ],
        returns=None,
        type_comment=None,
    )


def _generated_namespace(
    lowered: str,
    output: Path,
    clc_scratch_bytes: int,
    stages: tuple[tuple[object, ...], ...],
    attention_order: str,
    chained_merge_heads: int,
    task_claim: str,
) -> dict[str, object]:
    lowered_module = ast.parse(lowered)
    master = _master_function(lowered_module)
    functions = {
        node.name: node
        for node in lowered_module.body
        if isinstance(node, ast.FunctionDef)
    }
    scheduled = [
        function
        for function in functions.values()
        if function.name.endswith("_scheduled_task")
    ]
    qkv_scheduled = next(
        function
        for function in scheduled
        if {"qkv_weight_q", "kv_cache"} <= _function_arguments(function)
    )
    qkv_called_roots = {
        call.func.id
        for call in ast.walk(qkv_scheduled)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id.startswith("tile_dependency_root_")
    }
    qk_root = next(
        name
        for name in qkv_called_roots
        if {"q_weight", "k_weight"} <= _function_arguments(functions[name])
    )
    cache_roots = tuple(
        name
        for name in qkv_called_roots
        if {"slot_mapping", "kv_cache"} <= _function_arguments(functions[name])
    )
    qkv_report = _make_qkv_report_helper(qkv_scheduled, qk_root, cache_roots)
    qkv_report = _ReplaceCall(
        "_qwen_layer_report_attention_group",
        "_qwen3_layer_report_attention_group",
        ast.Constant(value=0),
    ).visit(qkv_report)
    for node in ast.walk(qkv_report):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_qwen3_layer_report_attention_group"
        ):
            node.args.pop()
    if attention_order == "chain":
        dispatcher = _make_attention_chain_dispatcher(
            master,
            qkv_scheduled.name,
            qkv_report.name,
            chained_merge_heads,
        )
        total_commands = _attention_chain_commands(chained_merge_heads)
    elif attention_order == "groups":
        dispatcher = _make_grouped_attention_dispatcher(
            master,
            qkv_scheduled.name,
            qkv_report.name,
        )
        total_commands = int(stages[-1][2])
    else:
        dispatcher = _make_dispatcher(
            master,
            stages,
            qkv_scheduled.name,
            qkv_report.name,
            attention_order,
        )
        total_commands = int(stages[-1][2])
    dispatcher_functions = [
        dispatcher,
        _make_clc_kernel(master, total_commands, task_claim),
    ]
    generated_module = ast.fix_missing_locations(
        ast.Module(
            body=[
                qkv_report,
                *dispatcher_functions,
            ],
            type_ignores=[],
        )
    )
    clc_source = CLC_SOURCE.replace(
        "__CLC_SCRATCH_BYTES__",
        str(clc_scratch_bytes),
    )
    generated_source = clc_source + "\n\n" + ast.unparse(generated_module) + "\n"
    combined_source = lowered + "\n\n" + generated_source
    filename = str(output.resolve())
    linecache.cache[filename] = (
        len(combined_source),
        None,
        combined_source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {"__name__": "_generated_qwen3_layer_clc"}
    exec(compile(combined_source, filename, "exec"), namespace)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(combined_source)
    return namespace


def _as_outputs(outputs) -> dict[str, torch.Tensor]:
    return dict(zip(OUTPUT_NAMES, outputs, strict=True))


def _clc_launcher(
    kernel,
    state: torch.Tensor,
    record_stats: bool,
    holder: dict,
    total_commands: int,
):
    def launch(
        _original_kernel,
        grid,
        *kernel_args,
        _persistent_state_specs=(),
        _minimum_resident_programs=0,
        **kwargs,
    ):
        compiled = kernel[(total_commands,)](
            *kernel_args,
            state,
            RECORD_STATS=record_stats,
            num_warps=1,
            num_stages=2,
            num_ctas=1,
            launch_pdl=True,
        )
        holder[record_stats] = compiled
        return compiled

    return launch


def _validate_stats(
    state: torch.Tensor,
    total_commands: int,
    stages: tuple[tuple[object, ...], ...],
    expected_first_wave: int,
    require_prefix: bool,
) -> dict[str, object]:
    cursor = int(state[TASK_CURSOR_SLOT].item())
    started = int(state[STARTED_SLOT].item())
    canceled = int(state[CANCELED_SLOT].item())
    processed = int(state[PROCESSED_SLOT].item())
    grid_seen = state[GRID_SEEN_BASE : GRID_SEEN_BASE + total_commands].to(torch.int32)
    started_seen = state[STARTED_SEEN_BASE:TASK_SEEN_BASE].to(torch.int32)
    task_seen = state[TASK_SEEN_BASE : TASK_SEEN_BASE + VISIBLE_TASKS].to(torch.int32)
    grid_missing = int((grid_seen == 0).sum().item())
    grid_duplicated = int((grid_seen != 1).sum().item()) - grid_missing
    task_missing = int((task_seen == 0).sum().item())
    task_duplicated = int((task_seen != 1).sum().item()) - task_missing
    started_prefix_missing = int((started_seen[:expected_first_wave] != 1).sum().item())
    unexpected_late_starts = int(started_seen[expected_first_wave:].sum().item())
    if started + canceled != total_commands:
        raise AssertionError("physical starts plus CLC cancellations != launch grid")
    if processed != VISIBLE_TASKS:
        raise AssertionError("CLC worker loop did not execute every command")
    if cursor != total_commands:
        raise AssertionError("CLC command cursor did not cover the logical grid")
    if grid_missing or grid_duplicated:
        raise AssertionError("CLC launch-token partition is invalid")
    if task_missing or task_duplicated:
        raise AssertionError("CLC logical-task partition is invalid")
    if require_prefix and (
        started != expected_first_wave
        or started_prefix_missing
        or unexpected_late_starts
    ):
        raise AssertionError("CLC physical first wave is not the expected grid prefix")
    return {
        "physically_started_ctas": started,
        "successful_cancellations": canceled,
        "launch_partition_total": started + canceled,
        "task_cursor": cursor,
        "processed_tasks": processed,
        "grid_missing": grid_missing,
        "grid_duplicated": grid_duplicated,
        "task_missing": task_missing,
        "task_duplicated": task_duplicated,
        "started_prefix_missing": started_prefix_missing,
        "unexpected_late_starts": unexpected_late_starts,
        "stage_tasks": {
            name: int(task_seen[start:end].sum().item())
            for name, _, _, _, start, count, _ in stages
            for end in (start + count,)
        },
    }


def _load_lowered(args, composite_args) -> tuple[str, object | None]:
    if args.lowered_input is not None:
        return Path(args.lowered_input).read_text(), None
    granular_kernel = _build_granular_kernel()
    bound = granular_kernel.bind(composite_args)
    config = _tuned_config(bound, args)
    return bound.to_triton_code(config, output_origin_lines=True), bound


def run(args: argparse.Namespace) -> None:
    if not args.allow_busy:
        require_idle_visible_gpu()
    if args.batch != 1:
        raise ValueError("the whole-layer CLC probe supports --batch 1")
    expected_shape = (4096, 12288, 32, 8, 128, 128)
    actual_shape = (
        args.hidden,
        args.intermediate,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.attention_splits,
    )
    if actual_shape != expected_shape:
        raise ValueError(
            f"the CLC task map is pinned to {expected_shape}, got {actual_shape}"
        )
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.clc_scratch_bytes < 32 or args.clc_scratch_bytes % 16:
        raise ValueError("--clc-scratch-bytes must be a multiple of 16 and at least 32")
    if (
        args.attention_order in {"groups", "staged-groups", "chain"}
        and args.tile_bundle != 1
    ):
        raise ValueError(
            "grouped attention ordering currently requires --tile-bundle 1"
        )
    clc_tensors = allocate_layer(args)
    clc_args = _composite_args(clc_tensors, args)
    lowered, bound = _load_lowered(args, clc_args)
    stages = _stage_schedule(args.tile_bundle)
    if args.attention_order == "chain":
        total_commands = _attention_chain_commands(args.chained_merge_heads)
    else:
        total_commands = int(stages[-1][2])
    if not 0 < args.expected_first_wave <= total_commands:
        raise ValueError("--expected-first-wave must fit in the CLC launch grid")
    output_path = Path(args.lowered_output)
    namespace = _generated_namespace(
        lowered,
        output_path,
        args.clc_scratch_bytes,
        stages,
        args.attention_order,
        args.chained_merge_heads,
        args.task_claim,
    )
    clc_kernel = namespace["qwen3_layer_clc_kernel"]
    layer_wrapper = namespace["qwen3_layer_tile_dependency_source"]

    compiled_holder: dict[bool, object] = {}
    state = torch.zeros(STATE_WORDS, device="cuda", dtype=torch.uint32)

    def reset_state() -> None:
        state.zero_()
        if args.task_claim == "prefix":
            state[TASK_CURSOR_SLOT] = args.expected_first_wave

    reset_state()
    clc_outputs = _as_outputs(
        layer_wrapper(
            *clc_args,
            _launcher=_clc_launcher(
                clc_kernel,
                state,
                True,
                compiled_holder,
                total_commands,
            ),
        )
    )
    torch.cuda.synchronize()
    clc_stats = _validate_stats(
        state,
        total_commands,
        stages,
        args.expected_first_wave,
        args.task_claim == "prefix",
    )

    static_tensors = allocate_layer(args)
    static_args = _composite_args(static_tensors, args)
    static_outputs = _as_outputs(layer_wrapper(*static_args))

    reference_tensors = allocate_layer(args)
    reference_launch, reference_outputs = build_helion_reference(
        args,
        reference_tensors,
    )
    reference_launch()
    torch.cuda.synchronize()

    for name, expected in reference_outputs.items():
        _assert_close(f"static_{name}", static_outputs[name], expected)
        _assert_close(f"orchestrator_{name}", clc_outputs[name], expected)
    for name in OUTPUT_NAMES:
        if name in {"residual", "pre_q", "pre_scale", "ffn_q", "ffn_scale"}:
            continue
        torch.testing.assert_close(
            clc_outputs[name],
            static_outputs[name],
            atol=0,
            rtol=0,
            msg=f"clc_{name}_vs_static",
        )

    benchmark_clc_tensors = allocate_layer(args)
    benchmark_clc_args = _composite_args(benchmark_clc_tensors, args)
    reset_state()
    benchmark_outputs = _as_outputs(
        layer_wrapper(
            *benchmark_clc_args,
            _launcher=_clc_launcher(
                clc_kernel,
                state,
                False,
                compiled_holder,
                total_commands,
            ),
        )
    )
    torch.cuda.synchronize()
    for name in OUTPUT_NAMES:
        if name in {"residual", "pre_q", "pre_scale", "ffn_q", "ffn_scale"}:
            continue
        torch.testing.assert_close(
            benchmark_outputs[name],
            static_outputs[name],
            atol=0,
            rtol=0,
            msg=f"uninstrumented_clc_{name}_vs_static",
        )

    static_graph, _ = capture(lambda: layer_wrapper(*static_args))
    clc_graph, _ = _capture_with_reset(
        lambda: layer_wrapper(
            *benchmark_clc_args,
            _launcher=_clc_launcher(
                clc_kernel,
                state,
                False,
                compiled_holder,
                total_commands,
            ),
        ),
        reset_state,
    )

    comparison_args = argparse.Namespace(**vars(args))
    comparison_args.attention_splits = args.helion_comparison_splits
    comparison_tensors = allocate_layer(comparison_args)
    reference_benchmark_launch, _ = build_helion_reference(
        comparison_args,
        comparison_tensors,
    )
    reference_graph, _ = capture(reference_benchmark_launch)

    benchmark_pids = visible_gpu_pids()
    if not args.allow_busy and (foreign_pids := benchmark_pids - {os.getpid()}):
        raise RuntimeError(
            f"GPU gained foreign compute processes {sorted(foreign_pids)}"
        )
    timings = _benchmark_graphs_cold_l2(
        {
            "static_schedule": (static_graph.replay, lambda: None),
            "clc_tile_scheduler": (clc_graph.replay, reset_state),
            "standalone_helion_graph": (reference_graph.replay, lambda: None),
        },
        args.repeats,
    )
    if not args.allow_busy and visible_gpu_pids() != benchmark_pids:
        raise RuntimeError("GPU process set changed during benchmark")

    clc_us = float(timings["clc_tile_scheduler"]["median_us"])
    static_us = float(timings["static_schedule"]["median_us"])
    standalone_us = float(timings["standalone_helion_graph"]["median_us"])
    compiled = compiled_holder[False]
    instrumented = compiled_holder[True]
    ptx = compiled.asm["ptx"]
    result = {
        "workload": "Qwen3-8B FP8 complete decode layer CLC megakernel",
        "device": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "context": args.context,
            "attention_splits": args.attention_splits,
            "helion_comparison_splits": args.helion_comparison_splits,
        },
        "schedule": {
            "scheduler_visible_tasks": VISIBLE_TASKS,
            "scheduler_commands": total_commands,
            "scheduler_granularity": "tile",
            "tile_bundle": args.tile_bundle,
            "expected_first_wave": args.expected_first_wave,
            "generated_fused_continuation_tasks": 5200 - VISIBLE_TASKS,
            "directly_chained_attention_tasks": (
                VISIBLE_TASKS - _attention_chain_commands(args.chained_merge_heads)
                if args.attention_order == "chain"
                else 0
            ),
            "chained_merge_heads_per_attention_key": (
                args.chained_merge_heads if args.attention_order == "chain" else 0
            ),
            "task_order": [stage[0] for stage in stages],
            "clc_scratch_bytes": args.clc_scratch_bytes,
            "launch_pdl": True,
            "task_claim": args.task_claim,
            "dependency_handoffs": "generated acquire/release counters",
            "attention_order": args.attention_order,
        },
        "clc": clc_stats,
        "correctness": {
            name: _error_stats(clc_outputs[name], reference_outputs[name])
            for name in reference_outputs
        },
        "cold_l2": {
            "flush_bytes": 256 * 1024 * 1024,
            "timings_us": timings,
            "speedup_vs_static": static_us / clc_us,
            "speedup_vs_standalone_helion_graph": standalone_us / clc_us,
        },
        "resources": _resources(compiled, args.clc_scratch_bytes),
        "instrumented_resources": _resources(
            instrumented,
            args.clc_scratch_bytes,
        ),
        "ptx_checks": {
            "contains_clc": "clusterlaunchcontrol.try_cancel" in ptx,
            "contains_clc_reissue": ptx.count("clusterlaunchcontrol.try_cancel") >= 2,
            "contains_static_scratch": (
                ".shared .align 16 .b8 qwen3_layer_clc_scratch"
                f"[{args.clc_scratch_bytes}]" in ptx
            ),
            "contains_triton_global_smem": "global_smem" in ptx,
            "contains_acquire": "ld.acquire.gpu.global.u32" in ptx,
            "contains_release": "atom.global.gpu.release" in ptx,
        },
        "lowered_input": (
            None if args.lowered_input is None else str(args.lowered_input)
        ),
        "lowered_output": str(output_path),
        "live_bound": bound is not None,
    }
    result_path = Path(args.output)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    result_path.with_suffix(".ptx").write_text(ptx)
    print("RESULT", json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=12288)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--attention-splits", type=int, default=128)
    parser.add_argument("--helion-comparison-splits", type=int, default=32)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--projection-stages", type=int, default=4)
    parser.add_argument("--w13-stages", type=int, default=4)
    parser.add_argument("--w13-unroll", type=int, default=2)
    parser.add_argument("--w2-stages", type=int, default=4)
    parser.add_argument("--w2-unroll", type=int, default=4)
    parser.add_argument("--kernel-stages", type=int, default=2)
    parser.add_argument("--worker-multiplier", type=int, default=8)
    parser.add_argument("--cross-loop-workers", type=int, default=1024)
    parser.add_argument("--merge-split-block", type=int, default=32)
    parser.add_argument("--merge-q-block", type=int, default=4)
    parser.add_argument("--attention-context-block", type=int, default=32)
    parser.add_argument("--qk-head-block", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--tile-bundle", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument(
        "--attention-order",
        choices=("global", "pipeline", "staged-groups", "groups", "chain"),
        default="chain",
    )
    parser.add_argument("--chained-merge-heads", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--clc-scratch-bytes",
        type=int,
        default=DEFAULT_CLC_SCRATCH_BYTES,
    )
    parser.add_argument("--expected-first-wave", type=int, default=1036)
    parser.add_argument(
        "--task-claim",
        choices=("prefix", "ticket"),
        default="prefix",
    )
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--lowered-input", type=Path)
    parser.add_argument(
        "--lowered-output",
        type=Path,
        default=Path("/tmp/qwen3_layer_clc_lowered.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/triton_qwen3_layer_clc_result.json"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
