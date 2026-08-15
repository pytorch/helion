from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from dataclasses import dataclass
import gc
import importlib
import math
import os
import threading
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import cast
from unittest.mock import patch
import weakref

import pytest
import torch

import helion
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion.exc import BackendUnsupported
from helion.exc import CuteBackendUnavailable
from helion.exc import InvalidConfig
import helion.language as hl
import helion.runtime as helion_runtime
from helion.runtime import _cute_cluster_shape
from helion.runtime import _cute_cluster_shape_from_wrapper_plans
from helion.runtime import _ensure_cute_dsl_arch_env
from helion.runtime import _get_compiled_cute_launcher
from helion.runtime import default_cute_launcher

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Sequence

    from helion.autotuner.config_spec import ConfigSpec

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")

get_cute_mma_support = importlib.import_module(
    "helion._compiler.cute.mma_support"
).get_cute_mma_support
_cute_grouped_reduce_shared_tree = importlib.import_module(
    "helion._compiler.cute.reduce_helpers"
)._cute_grouped_reduce_shared_tree
_cute_flash = importlib.import_module("helion._compiler.cute.cute_flash")
flash_fa4_shared_storage = importlib.import_module(
    "helion._compiler.cute._flash_runtime"
).flash_fa4_shared_storage
resolve_flash_config = _cute_flash.resolve_flash_config


def _batched_tcgen05_two_cta_config(*, cluster_n: int = 1) -> helion.Config:
    return helion.Config(
        block_sizes=[1, 256, 256, 64],
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=cluster_n,
        pid_type="persistent_blocked",
        tcgen05_persistence_model="static_persistent",
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
    )


def _leading_tcgen05_direct_entry_config() -> helion.Config:
    return helion.Config(
        block_sizes=[1, 256, 256, 64],
        indexing=["tensor_descriptor"] * 3,
        l2_groupings=[1],
        num_warps=8,
        num_stages=4,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=6,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=4,
        tcgen05_num_epi_warps=4,
        tcgen05_l2_swizzle_size=1,
        tcgen05_persistence_model="static_persistent",
        tcgen05_layout_strategy="explicit_epi_tile",
        tcgen05_layout_overrides_epi_tile_m=128,
        tcgen05_layout_overrides_epi_tile_n=32,
        tcgen05_layout_overrides_d_store_box_n=32,
        tcgen05_flat_role_coordinates=True,
        tcgen05_tvm_ffi_launch=True,
    )


@helion.kernel(backend="cute")
def cute_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="cute")
def cute_add3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile] + z[tile]
    return out


@helion.kernel(backend="cute")
def cute_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * y[tile]
    return out


@helion.kernel(backend="cute")
def cute_relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.relu(x[tile])
    return out


@helion.kernel(backend="cute")
def cute_sin(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sin(x[tile])
    return out


@helion.kernel(backend="cute")
def cute_sigmoid(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sigmoid(x[tile])
    return out


@helion.kernel(backend="cute")
def cute_pointwise_chain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.sigmoid(torch.sin(torch.relu(x[tile] * y[tile])))
    return out


@helion.kernel(backend="cute")
def cute_minimum_maximum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = torch.minimum(torch.maximum(x[tile], y[tile]), x[tile])
    return out


@helion.kernel(backend="cute", autotune_effort="none")
def cute_affine_scalar_args(
    x: torch.Tensor,
    scale: int,
    bias: float,
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] * scale + bias
    return out


@helion.kernel(backend="cute")
def cute_device_loop_add_one(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + 1
    return out


@helion.kernel(backend="cute")
def cute_flattened_device_loop_add_one(x: torch.Tensor) -> torch.Tensor:
    b, m, n = x.size()
    out = torch.empty_like(x)
    for tile_b in hl.tile(b):
        for tile_m, tile_n in hl.tile([m, n]):
            out[tile_b, tile_m, tile_n] = x[tile_b, tile_m, tile_n] + 1
    return out


@helion.kernel(backend="cute")
def cute_row_sum(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty([n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n, :].sum(-1)
    return out


@helion.kernel(backend="cute")
def cute_normalize_by_sum(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        row_sum = x[tile_n, :].sum(-1)
        out[tile_n, :] = x[tile_n, :] / row_sum[:, None]
    return out


@helion.kernel(backend="cute")
def cute_normalize_by_sum_fp32_cast(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        vals = x[tile_n, :].to(torch.float32)
        row_sum = vals.sum(-1)
        out[tile_n, :] = (vals / row_sum[:, None]).to(x.dtype)
    return out


@helion.kernel(backend="cute")
def cute_row_centered(x: torch.Tensor) -> torch.Tensor:
    n, m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        row_sum = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m):
            row_sum = row_sum + x[tile_n, tile_m].to(torch.float32).sum(dim=1)
        row_mean = row_sum / m
        for tile_m in hl.tile(m):
            vals = x[tile_n, tile_m].to(torch.float32)
            out[tile_n, tile_m] = (vals - row_mean[:, None]).to(x.dtype)
    return out


@helion.kernel(backend="cute", autotune_effort="none")
def cute_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    n, m = x.size()
    out = torch.empty_like(x)
    hl.specialize(m)
    for tile_n in hl.tile(n):
        vals = x[tile_n, :].to(torch.float32)
        mean_sq = torch.mean(vals * vals, dim=-1)
        inv_rms = torch.rsqrt(mean_sq + eps)
        out[tile_n, :] = (vals * inv_rms[:, None] * weight[:].to(torch.float32)).to(
            x.dtype
        )
    return out


@helion.kernel(backend="cute")
def cute_row_max(x: torch.Tensor) -> torch.Tensor:
    n, m = x.size()
    out = torch.empty([n], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        row_max = hl.full([tile_n], float("-inf"), dtype=torch.float32)
        for tile_m in hl.tile(m):
            vals = x[tile_n, tile_m].to(torch.float32)
            row_max = torch.maximum(row_max, torch.amax(vals, dim=1))
        out[tile_n] = row_max
    return out


@helion.kernel(backend="cute")
def cute_row_min(x: torch.Tensor) -> torch.Tensor:
    n, m = x.size()
    out = torch.empty([n], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        row_min = hl.full([tile_n], float("inf"), dtype=torch.float32)
        for tile_m in hl.tile(m):
            vals = x[tile_n, tile_m].to(torch.float32)
            row_min = torch.minimum(row_min, torch.amin(vals, dim=1))
        out[tile_n] = row_min
    return out


@helion.kernel(backend="cute")
def cute_row_prod(x: torch.Tensor) -> torch.Tensor:
    n, m = x.size()
    out = torch.empty([n], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        row_prod = hl.full([tile_n], 1.0, dtype=torch.float32)
        for tile_m in hl.tile(m):
            vals = x[tile_n, tile_m].to(torch.float32)
            row_prod = row_prod * torch.prod(vals, dim=1)
        out[tile_n] = row_prod
    return out


@cute.kernel
def cute_shared_tree_reduce_max(inp, out):
    lane = cutlass.Int32(cute.arch.thread_idx()[0]) + cutlass.Int32(
        cute.arch.thread_idx()[1]
    ) * cutlass.Int32(3)
    lane_in_group = lane % 48
    lane_mod_pre = lane_in_group % 3
    reduce_idx = lane_in_group // 3
    result = _cute_grouped_reduce_shared_tree(
        inp[lane_mod_pre, reduce_idx],
        "max",
        cutlass.Float32(float("-inf")),
        lane,
        lane_in_group,
        lane_mod_pre,
        pre=3,
        group_span=48,
        num_threads=48,
        group_count=1,
    )
    if lane_in_group < 3:
        out[lane_in_group] = result


@cute.kernel
def cute_shared_tree_reduce_min(inp, out):
    lane = cutlass.Int32(cute.arch.thread_idx()[0]) + cutlass.Int32(
        cute.arch.thread_idx()[1]
    ) * cutlass.Int32(3)
    lane_in_group = lane % 48
    lane_mod_pre = lane_in_group % 3
    reduce_idx = lane_in_group // 3
    result = _cute_grouped_reduce_shared_tree(
        inp[lane_mod_pre, reduce_idx],
        "min",
        cutlass.Float32(float("inf")),
        lane,
        lane_in_group,
        lane_mod_pre,
        pre=3,
        group_span=48,
        num_threads=48,
        group_count=1,
    )
    if lane_in_group < 3:
        out[lane_in_group] = result


@cute.kernel
def cute_shared_tree_reduce_prod(inp, out):
    lane = cutlass.Int32(cute.arch.thread_idx()[0]) + cutlass.Int32(
        cute.arch.thread_idx()[1]
    ) * cutlass.Int32(3)
    lane_in_group = lane % 48
    lane_mod_pre = lane_in_group % 3
    reduce_idx = lane_in_group // 3
    result = _cute_grouped_reduce_shared_tree(
        inp[lane_mod_pre, reduce_idx],
        "prod",
        cutlass.Float32(1.0),
        lane,
        lane_in_group,
        lane_mod_pre,
        pre=3,
        group_span=48,
        num_threads=48,
        group_count=1,
    )
    if lane_in_group < 3:
        out[lane_in_group] = result


@cute.kernel
def cute_shared_tree_matmul_sum(lhs, rhs, out):
    lane = cutlass.Int32(cute.arch.thread_idx()[0]) + cutlass.Int32(
        cute.arch.thread_idx()[1]
    ) * cutlass.Int32(3)
    lane_in_group = lane % 48
    row = lane_in_group % 3
    reduce_idx = lane_in_group // 3
    product = lhs[row, reduce_idx] * rhs[reduce_idx, cutlass.Int32(0)]
    result = _cute_grouped_reduce_shared_tree(
        product,
        "sum",
        cutlass.Float32(0.0),
        lane,
        lane_in_group,
        row,
        pre=3,
        group_span=48,
        num_threads=48,
        group_count=1,
    )
    if lane_in_group < 3:
        out[row, cutlass.Int32(0)] = result


@helion.kernel(backend="cute")
def cute_matmul_addmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def cute_matmul_addmm_shifted_operands(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k] + 1, y[tile_k, tile_n] + 1)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def cute_nested_grid_addmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def cute_addmm_same_iteration_relu_consumer(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            mm = torch.addmm(
                hl.zeros([tile_m, tile_n], dtype=torch.float32),
                x[tile_m, tile_k],
                y[tile_k, tile_n],
            )
            acc = acc + torch.relu(mm)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def cute_dot_acc_dynamic_bf16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def cute_matmul_direct(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
        out[tile_m, tile_n] = torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
    return out


@helion.kernel(backend="cute")
def cute_matmul_addmm_direct(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=bias.dtype, device=x.device)
    for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
        out[tile_m, tile_n] = torch.addmm(
            bias[tile_m, tile_n],
            x[tile_m, tile_k],
            y[tile_k, tile_n],
        )
    return out


@helion.kernel(backend="cute")
def cute_matmul_addmm_shifted_direct(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=bias.dtype, device=x.device)
    for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
        out[tile_m, tile_n] = torch.addmm(
            bias[tile_m, tile_n],
            x[tile_m, tile_k] + 1,
            y[tile_k, tile_n] + 1,
        )
    return out


@helion.kernel(backend="cute")
def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def cute_matmul_mma_fp8(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # fp8 (e4m3) inputs, f32 accumulate, bf16 output -- the tcgen05 MMA atom
    # for fp8 is MmaF8F6F4Op (MMA-K=32 vs 16 for bf16/fp16).
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def cute_matmul_mma_fp8_rowvec_scale(
    x: torch.Tensor, y: torch.Tensor, scale_n: torch.Tensor
) -> torch.Tensor:
    # fp8 GEMM with a fused per-column (rowvec) scale in the epilogue.
    # Exercises the rowvec aux chain on the tcgen05 fp8 path (and, for
    # TMA-store configs, the register-hoist of the rowvec load).
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = (acc * scale_n[tile_n]).to(torch.bfloat16)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def cute_matmul_mma_fp8_colvec_scale(
    x: torch.Tensor, y: torch.Tensor, scale_m: torch.Tensor
) -> torch.Tensor:
    # fp8 GEMM with a fused per-row (column-vector ``scale_m[m]``) scale.
    # Exercises the colvec aux chain (``broadcast_axis == 2``) on the tcgen05
    # fp8 path, including the scalar fast-path / dense-materialize selection.
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = (acc * scale_m[tile_m, tile_n]).to(torch.bfloat16)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def cute_matmul_mma_fp8_rowwise_colwise_scale(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_m: torch.Tensor,
    scale_n: torch.Tensor,
) -> torch.Tensor:
    # fp8 GEMM with BOTH a per-row (colvec ``scale_m[m]``) and per-column
    # (rowvec ``scale_n[n]``) fused scale in the epilogue -- the rowwise x
    # rowwise scaling used by vLLM-style fp8 W8A8 GEMMs. Exercises both
    # broadcast-aux directions in a single tcgen05 epilogue chain.
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        acc = acc * scale_m[tile_m, tile_n] * scale_n[tile_n]
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def cute_matmul_mma_epilogue_f32_bias(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    # fp8 GEMM with an exact-shape (full (m, n), non-broadcast) fused bias
    # add in the epilogue, f32 accumulate -> bf16 out. Exercises the
    # exact-shape aux load path (``broadcast_axis is None``).
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = (acc + bias[tile_m, tile_n]).to(torch.bfloat16)
    return out


@helion.kernel(backend="cute")
def cute_matmul_mma_epilogue(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + bias[tile_n]).to(x.dtype)
    return out


@helion.kernel(backend="cute")
def cute_matmul_mma_with_bias_acc(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = bias[tile_m, tile_n].to(torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def cute_matmul_mma_mixed_k_loop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        extra = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            extra = extra + x[tile_m, tile_k].to(torch.float32).sum(dim=1, keepdim=True)
        out[tile_m, tile_n] = acc + extra
    return out


@helion.kernel(backend="cute")
def cute_matmul_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def cute_matmul_dot_direct(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)
    for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
        out[tile_m, tile_n] = hl.dot(
            x[tile_m, tile_k],
            y[tile_k, tile_n],
            out_dtype=torch.float16,
        )
    return out


@helion.kernel(backend="cute")
def cute_matmul_dot_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="cute")
def cute_matmul_dot_out_dtype(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                x[tile_m, tile_k],
                y[tile_k, tile_n],
                acc=acc,
                out_dtype=torch.float16,
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute", static_shapes=False)
def cute_matmul_packed_rhs_bfloat16(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> None:
    m, k = a.shape
    _, n = b.shape
    block_size_k = hl.register_block_size(k // 2)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=a.dtype)
        for tile_k in hl.tile(k // 2, block_size=block_size_k):
            lhs = a[
                tile_m,
                tile_k.begin * 2 : tile_k.begin * 2 + tile_k.block_size * 2,
            ]
            packed = b[tile_k, tile_n]
            rhs = torch.stack([packed, packed], dim=1).reshape(
                tile_k.block_size * 2, tile_n.block_size
            )
            acc = torch.addmm(acc, lhs, rhs)
        c[tile_m, tile_n] = acc


@helion.kernel(backend="cute")
def cute_baddbmm(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = bias[tile_b, tile_m, tile_n].to(torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc,
                x[tile_b, tile_m, tile_k],
                y[tile_b, tile_k, tile_n],
            )
        out[tile_b, tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def cute_batched_baddbmm_tcgen05(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc,
                x[tile_b, tile_m, tile_k],
                y[tile_b, tile_k, tile_n],
            )
        out[tile_b, tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def cute_batched_baddbmm_rowvec_bias_tcgen05(
    x: torch.Tensor,
    y: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # Leading-batch matmul with a trailing-axis (rowvec) bias fused into
    # the epilogue. The rank-3 carrier ``[1, BM, BN]`` has a block-size-1
    # batch-passthrough leading axis; the aux classifier strips it so
    # ``acc + bias[tile_n]`` classifies as the (M, N)-tile rowvec form and
    # splices into the tcgen05 epilogue instead of hitting the backstop.
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc,
                x[tile_b, tile_m, tile_k],
                y[tile_b, tile_k, tile_n],
            )
        out[tile_b, tile_m, tile_n] = (acc + bias[tile_n]).to(torch.bfloat16)
    return out


@helion.kernel(backend="cute")
def cute_transformed_dot_tcgen05(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[(tile_m + 1) % m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute")
def cute_batched_dot_residual_tcgen05(
    x: torch.Tensor,
    y: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                x[tile_b, tile_m, tile_k],
                y[tile_b, tile_k, tile_n],
                acc=acc,
            )
        out[tile_b, tile_m, tile_n] = (acc + residual[tile_b, tile_m, tile_n]).to(
            torch.bfloat16
        )
    return out


@helion.kernel(backend="cute")
def cute_batched_dot_tcgen05(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                x[tile_b, tile_m, tile_k],
                y[tile_b, tile_k, tile_n],
                acc=acc,
            )
        out[tile_b, tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute")
def cute_casted_batched_dot_tcgen05(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                x[tile_b, tile_m, tile_k].to(torch.bfloat16),
                y[tile_b, tile_k, tile_n].to(torch.bfloat16),
                acc=acc,
            )
        out[tile_b, tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute")
def cute_batched_dot_unsupported_epilogue_tcgen05(
    x: torch.Tensor,
    y: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                x[tile_b, tile_m, tile_k],
                y[tile_b, tile_k, tile_n],
                acc=acc,
            )
        out[tile_b, tile_m, tile_n] = (
            acc + torch.sin(residual[tile_b, tile_m, tile_n])
        ).to(torch.bfloat16)
    return out


@helion.kernel(backend="cute")
def cute_bare_bmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=x.dtype, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=x.dtype)
        for tile_k in hl.tile(k):
            acc = torch.bmm(
                x[tile_b, tile_m, tile_k],
                y[tile_b, tile_k, tile_n],
            )
        out[tile_b, tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def cute_bare_bmm_dtype(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.ops.aten.bmm.dtype(
                x[tile_b, tile_m, tile_k],
                y[tile_b, tile_k, tile_n],
                torch.float32,
            )
        out[tile_b, tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def cute_bare_batched_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.float32, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                x[tile_b, tile_m, tile_k],
                y[tile_b, tile_k, tile_n],
                out_dtype=torch.float32,
            )
        out[tile_b, tile_m, tile_n] = acc
    return out


@helion.kernel(backend="cute")
def cute_repeated_2d_dot(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_marker: torch.Tensor,
) -> torch.Tensor:
    b = batch_marker.size(0)
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_b, tile_m, tile_n] = acc.unsqueeze(0).to(torch.bfloat16)
    return out


@helion.kernel(backend="cute")
def cute_shifted_batched_dot_tcgen05(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                x[tile_b + 1, tile_m, tile_k],
                y[tile_b + 1, tile_k, tile_n],
                acc=acc,
            )
        out[tile_b, tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute")
def cute_permuted_store_batched_dot_tcgen05(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    b, m, k = x.size()
    _, _, n = y.size()
    out = torch.empty([b, n, m], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(
                x[tile_b, tile_m, tile_k],
                y[tile_b, tile_k, tile_n],
                acc=acc,
            )
        out[tile_b, tile_n, tile_m] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute")
def cute_dynamic_row_sum(x: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    out = x.new_empty([x.size(0)])
    bs = hl.register_block_size(x.size(1))
    for tile0 in hl.tile(x.size(0)):
        acc = hl.zeros([tile0, bs])
        for tile1 in hl.tile(end[0], block_size=bs):
            acc += x[tile0, tile1]
        out[tile0] = acc.sum(-1)
    return out


@helion.kernel(backend="cute")
def cute_mixed_rank_batched_dot_tcgen05(
    x: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    b, m, k = x.size()
    _, n = w.size()
    out = torch.empty([b, m, n], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_b, tile_m, tile_k], w[tile_k, tile_n], acc=acc)
        out[tile_b, tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute")
def cute_rhs_batched_dot_tcgen05(w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = w.size()
    b, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.bfloat16, device=w.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(w[tile_m, tile_k], y[tile_b, tile_k, tile_n], acc=acc)
        out[tile_b, tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute")
def cute_transposed_operand_batched_dot_tcgen05(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # A rank-3 hl.dot whose LHS operand is TRANSPOSED (a permute in the
    # load->operand chain). _trace_to_load does not trace permutes, so
    # _analyze_mma_operands bails and codegen cannot lower this as batched MMA.
    # It is still rank-3, so it exercises the F2 guarantee: batched tcgen05
    # search enablement is gated on the same structural analyzer codegen uses
    # (analyze_cute_mma_node), NOT on operand rank alone -- so this kernel
    # must NOT shape the batched search surface.
    b, k, m = x.size()
    _, _, n = y.size()
    out = torch.empty([b, m, n], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            xt = x[tile_b, tile_k, tile_m].transpose(-1, -2)
            acc = hl.dot(xt, y[tile_b, tile_k, tile_n], acc=acc)
        out[tile_b, tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute")
def cute_permute_transpose(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n].permute(1, 0)
    return out


@helion.kernel(backend="cute")
def cute_permute_store_then_read(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n].permute(1, 0)
        out[tile_m, tile_n] = out[tile_m, tile_n] + 1
    return out


@helion.kernel(backend="cute")
def cute_reduction_with_nested_tiles(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """RMS-norm-backward-shaped kernel: a `.mean(-1)` reduction plus nested
    non-reduction M tiling (register_block_size + inner hl.tile)."""
    m, n = x.size()
    out = torch.empty_like(x)
    block_m = hl.register_block_size(m)
    for tile_cta in hl.tile(m, block_size=block_m):
        for tile_m in hl.tile(tile_cta.begin, tile_cta.end):
            row = x[tile_m, :].to(torch.float32)
            mean_sq = (row * row).mean(-1)
            out[tile_m, :] = (
                row * torch.rsqrt(mean_sq[:, None] + 1e-6) * w[None, :]
            ).to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_with_lse(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    lse = torch.empty([q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        lse[tile_b, tile_m] = m_i + torch.log2(l_i)
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), lse.view(q_in.size()[:-1])


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_v_loaded_before_k(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            vt = v_view[tile_b, tile_n, :]
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_unscaled_qk(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_fp16_qk(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float16)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1).to(torch.float32))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1).to(torch.float32)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_post_center_scale(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = (qk - m_ij[:, :, None]) * 2.0
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_shifted_q(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm((qt + 1.0) * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_shifted_v(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n + 1, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_shifted_k(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n + 1, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_shifted_q_and_out(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m + 1, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m + 1, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_no_final_divide(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_no_alpha_rescale(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            l_i = l_i + l_ij
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_post_l_update(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            l_i = l_i + 1.0
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_post_acc_update(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            acc = acc + 1.0
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_with_aux(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    aux = torch.empty([q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        aux[tile_b, tile_m] = torch.zeros_like(l_i)
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), aux.view(q_in.size()[:-1])


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_with_lse_and_aux(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    lse = torch.empty([q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32)
    aux = torch.empty([q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        lse[tile_b, tile_m] = m_i + torch.log2(l_i)
        aux[tile_b, tile_m] = torch.zeros_like(l_i)
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), lse.view(q_in.size()[:-1]), aux.view(q_in.size()[:-1])


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_with_log_aux(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    aux = torch.empty([q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        aux[tile_b, tile_m] = torch.log2(l_i) + 1.0
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), aux.view(q_in.size()[:-1])


@helion.kernel(backend="cute", static_shapes=True)
def cute_dense_attention_with_3d_aux(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    aux = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = (acc / l_i[:, :, None]).to(out.dtype)
        aux[tile_b, tile_m, :] = acc
        out[tile_b, tile_m, :] = acc
    return out.view(q_in.size()), aux.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_causal_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    lse = torch.empty([q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        lse[tile_b, tile_m] = m_i + torch.log2(l_i)
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), lse.view(q_in.size()[:-1])


@helion.kernel(backend="cute", static_shapes=True)
def cute_shifted_causal_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = torch.where(
                tile_m.index[None, :, None] - tile_n.index[None, None, :] + 1 >= 0,
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_biased_attention(q_in, k_in, v_in, bias):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    bias_view = bias.reshape([-1, m_dim, n_dim])
    out = torch.empty_like(q_view)
    qk_scale = 1.0 / math.sqrt(head_dim)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = qk + bias_view[tile_b, tile_m, tile_n]
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_biased_attention_with_lse(q_in, k_in, v_in, bias):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    bias_view = bias.reshape([-1, m_dim, n_dim])
    out = torch.empty_like(q_view)
    lse = torch.empty([q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32)
    qk_scale = 1.0 / math.sqrt(head_dim)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = qk + bias_view[tile_b, tile_m, tile_n]
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        lse[tile_b, tile_m] = m_i + torch.log(l_i)
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), lse.view(q_in.size()[:-1])


@helion.kernel(backend="cute", static_shapes=True)
def cute_causal_biased_attention(q_in, k_in, v_in, bias):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    bias_view = bias.reshape([-1, m_dim, n_dim])
    out = torch.empty_like(q_view)
    qk_scale = 1.0 / math.sqrt(head_dim)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = qk + bias_view[tile_b, tile_m, tile_n]
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_relative_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = qk + (tile_m.index[None, :, None] - tile_n.index[None, None, :]) * 0.01
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_alibi_attention(q_in, k_in, v_in, slopes):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    heads = hl.specialize(q_in.size(1))
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            q_idx = tile_m.index[None, :, None]
            kv_idx = tile_n.index[None, None, :]
            qk = qk + (kv_idx - q_idx) * slopes[tile_b.index % heads]
            qk = torch.where(
                q_idx >= kv_idx,
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_sliding_window_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            delta = tile_m.index[None, :, None] - tile_n.index[None, None, :]
            qk = torch.where((delta >= 0) & (delta <= 64), qk, float("-inf"))
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_duplicate_window_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            delta = tile_m.index[None, :, None] - tile_n.index[None, None, :]
            qk = torch.where(
                (delta >= 0) & (delta <= 32) & (delta <= 64),
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_prefix_lm_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            prefix = tile_n.index[None, None, :] < 64
            causal = tile_m.index[None, :, None] >= tile_n.index[None, None, :]
            qk = torch.where(prefix | causal, qk, float("-inf"))
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_prefix_lm_attention_long_prefix(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            prefix = tile_n.index[None, None, :] < 192
            causal = tile_m.index[None, :, None] >= tile_n.index[None, None, :]
            qk = torch.where(prefix | causal, qk, float("-inf"))
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_document_mask_attention(q_in, k_in, v_in, document_ids):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    heads = hl.specialize(q_in.size(1))
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    document_view = document_ids.reshape([-1, m_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            doc_batch = tile_b.index // heads
            doc_q = document_view[doc_batch, tile_m]
            doc_k = document_view[doc_batch, tile_n]
            causal = tile_m.index[None, :, None] >= tile_n.index[None, None, :]
            same_doc = doc_q[:, :, None] == doc_k[:, None, :]
            qk = torch.where(causal & same_doc, qk, float("-inf"))
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_duplicate_document_mask_attention(q_in, k_in, v_in, document_ids):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    heads = hl.specialize(q_in.size(1))
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    document_view = document_ids.reshape([-1, m_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            doc_batch = tile_b.index // heads
            doc_q = document_view[doc_batch, tile_m]
            doc_k = document_view[doc_batch, tile_n]
            causal = tile_m.index[None, :, None] >= tile_n.index[None, None, :]
            same_doc = doc_q[:, :, None] == doc_k[:, None, :]
            qk = torch.where(causal & same_doc & same_doc, qk, float("-inf"))
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.kernel(backend="cute", static_shapes=True)
def cute_softcap_attention(q_in, k_in, v_in):
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        qt = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            kt = k_view[tile_b, tile_n, :]
            qk = torch.bmm(qt * qk_scale, kt.transpose(1, 2), torch.float32)
            qk = 2.0 * torch.tanh(qk / 2.0)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            vt = v_view[tile_b, tile_n, :]
            acc = torch.baddbmm(acc, p.to(vt.dtype), vt)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


def _flash_fired(code: str) -> bool:
    return (
        "_helion_flash_rt" in code
        or "_flash_scale_log2" in code
        or "helion_small_biased_attention" in code
    )


def _assert_score_modified_reductions(test_case: TestCase, code: str) -> None:
    test_case.assertTrue("fmax_reduce_packed" in code or "_fmax_reduce_chunk" in code)
    test_case.assertTrue(
        "fadd_reduce_packed" in code
        or "_disc_chunk_rowsum" in code
        or "fa4_exp2_convert_rowsum" in code
        or "fa4_disc_exp_convert_store" in code
    )


def _attention_from_log2_scores(
    scores_log2: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    probs = torch.softmax(scores_log2.float() * math.log(2.0), dim=-1)
    return torch.matmul(probs.to(v.dtype), v)


def _launch_entry(
    schema: tuple[tuple[object, ...], ...],
    launch_args: tuple[object, ...],
    owned_tensors: tuple[torch.Tensor, ...] = (),
) -> helion_runtime._CuteLaunchArgCacheEntry:
    return helion_runtime._CuteLaunchArgCacheEntry(
        schema, launch_args, (), owned_tensors
    )


def _runtime_identity_kernel() -> Any:
    @helion.kernel(backend="cute")
    def identity(x: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            out[tile] = x[tile]
        return out

    return identity


def _runtime_layout(bound: Any) -> torch.Tensor:
    return cast("torch.Tensor", bound.env.runtime_arg_values_by_name["layout"])


@onlyBackends(["cute"])
class TestCuteBackend(TestCase):
    def test_pointwise_add(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(cute_add, args)
        x, y = args
        torch.testing.assert_close(out, x + y)

    def test_reduction_with_nested_tiles_registers_vec_slots_eagerly(self) -> None:
        """Regression: a cute reduction kernel with its own non-reduction tiling
        (rms_norm backward) registered the tile's cute_vector_widths slot lazily
        during codegen, growing the config spec after the autotuner snapshotted
        it -> IndexError.  Assert the slots are registered eagerly instead.
        """
        x = torch.randn(512, 4096, device=DEVICE, dtype=HALF_DTYPE)
        w = torch.randn(4096, device=DEVICE, dtype=HALF_DTYPE)
        bound = cute_reduction_with_nested_tiles.bind((x, w))
        tile_block_ids = {
            bs.block_id for bs in bound.env.block_sizes if not bs.reduction
        }
        registered = set(bound.config_spec.cute_vector_widths.valid_block_ids())
        self.assertTrue(tile_block_ids, "kernel should expose non-reduction tiles")
        missing = tile_block_ids - registered
        self.assertFalse(
            missing,
            f"non-reduction tile blocks {sorted(missing)} were not registered in "
            f"cute_vector_widths during device-IR analysis (registered: "
            f"{sorted(registered)}); they would be appended lazily during codegen "
            f"and grow the config spec mid-autotune",
        )

    def test_flash_attention_fa4_persistent_power2_decode_uses_shift_mask(self) -> None:
        code = _cute_flash._flash_fa4_wrap(
            "if warp_idx == 14:",
            "    cute.arch.setmaxregister_decrease(40)",
            "        flash_sink = flash_m_pair + flash_bh",
            persistent=True,
            prelude="decode",
            total_tiles=32768,
            num_m_pairs=512,
        )
        self.assertIn(
            "flash_grid_bh_delta = (flash_grid_dim >> 9)",
            code,
        )
        self.assertIn(
            "flash_grid_m_pairs_delta = (flash_grid_dim & cutlass.Int32(511))",
            code,
        )
        self.assertIn(
            "flash_m_pair = (flash_tile_id & cutlass.Int32(511))",
            code,
        )
        self.assertIn("flash_bh = (flash_tile_id >> 9)", code)
        self.assertNotIn("flash_grid_dim // 512", code)
        self.assertNotIn("flash_tile_id % 512", code)
        self.assertNotIn("flash_tile_id // 512", code)

    def test_flash_attention_fa4_no_prelude_uses_counted_loop_for_mid_dense(
        self,
    ) -> None:
        code = _cute_flash._flash_fa4_wrap(
            "if warp_idx == 0:",
            "    cute.arch.setmaxregister_increase(200)",
            "        flash_sink = flash_sink + flash_tile_id",
            persistent=True,
            prelude="none",
            total_tiles=8192,
            num_m_pairs=128,
        )
        self.assertIn("flash_tile_count = cutlass.Int32(0)", code)
        self.assertIn(
            "for flash_tile_iter in cutlass.range(flash_tile_count, unroll=1):\n"
            "        flash_sink = flash_sink + flash_tile_id\n"
            "        flash_tile_id = flash_tile_id + flash_grid_dim",
            code,
        )
        self.assertNotIn("while flash_tile_id < 8192", code)

    def test_flash_attention_fa4_no_prelude_omits_dead_counted_tile_id_advance(
        self,
    ) -> None:
        code = _cute_flash._flash_fa4_wrap(
            "if warp_idx == 0:",
            "    cute.arch.setmaxregister_increase(200)",
            "        flash_sink = flash_sink + cutlass.Int32(1)",
            persistent=True,
            prelude="none",
            total_tiles=8192,
            num_m_pairs=128,
        )
        self.assertIn("for flash_tile_iter in cutlass.range", code)
        self.assertNotIn("flash_tile_id = flash_tile_id + flash_grid_dim", code)

    def test_flash_attention_fa4_no_prelude_keeps_while_loop_for_long_sequence(
        self,
    ) -> None:
        code = _cute_flash._flash_fa4_wrap(
            "if warp_idx == 0:",
            "    cute.arch.setmaxregister_increase(200)",
            "        flash_sink = flash_sink + cutlass.Int32(1)",
            persistent=True,
            prelude="none",
            total_tiles=8192,
            num_m_pairs=1024,
        )
        self.assertIn("while flash_tile_id < 8192", code)
        self.assertNotIn("flash_tile_count = cutlass.Int32(0)", code)

    def test_flash_attention_fa4_first_load_order_variants(self) -> None:
        def order(first_load_order: int) -> list[str]:
            return _cute_flash._flash_fa4_load_prologue_for_order(
                first_load_order, "Q0", "K0", "Q1", "V0"
            ).splitlines()

        self.assertEqual(order(0), ["Q0", "K0", "Q1", "V0"])
        self.assertEqual(order(1), ["K0", "V0", "Q0", "Q1"])
        self.assertEqual(order(2), ["Q0", "Q1", "K0", "V0"])
        self.assertEqual(order(3), ["K0", "Q0", "V0", "Q1"])
        self.assertEqual(order(4), ["K0", "Q0", "Q1", "V0"])

    def test_flash_attention_fires_and_matches_sdpa(self) -> None:
        """With the gate default-on, square fp16 attention at [1,128,128] lowers
        to the fused tcgen05 flash kernel and matches SDPA for head_dim 64/128."""
        for head_dim in (64, 128):
            with (
                self.subTest(head_dim=head_dim),
                patch(
                    "helion._compiler.cute.backend._detect_specialized_mma_loop",
                    return_value=True,
                ),
            ):
                q, k, v = (
                    torch.randn(2, 8, 256, head_dim, dtype=torch.float16, device=DEVICE)
                    for _ in range(3)
                )
                code, out = code_and_output(
                    cute_dense_attention, (q, k, v), block_sizes=[1, 128, 128]
                )
                self.assertTrue(_flash_fired(code))
                self.assertIn("flash_s0_corr_full_ptr", code)
                self.assertNotIn("flash_s0_corr_prod", code)
                self.assertIn("flash_kv_prod", code)
                self.assertNotIn("flash_v_prod", code)
                self.assertIn("flash_kv_prod.tail()", code)
                self.assertIn("flash_q_prod.tail()", code)
                if "flash_grid_m_pairs_delta" in code:
                    self.assertIn("flash_grid_m_pairs_delta", code)
                    self.assertIn("flash_tmem_dealloc_ptr", code)
                    self.assertIn("flash_tmem_user_bar.arrive_and_wait()", code)
                    self.assertIn("flash_tmem_user_bar.arrive()", code)
                    self.assertNotIn("mbarrier_wait(flash_tmem_dealloc_ptr, 0)", code)
                    self.assertNotIn("mbarrier_arrive(flash_tmem_dealloc_ptr)", code)
                    self.assertNotIn("cute.arch.barrier()", code)
                    self.assertNotIn("_flash_total_tiles // _flash_num_bh", code)
                    self.assertNotIn("flash_tile_id % flash_num_m_pairs", code)
                    self.assertNotIn("flash_tile_id // flash_num_m_pairs", code)
                    self.assertNotIn("_flash_num_bh", code)
                    self.assertNotIn("_flash_total_tiles", code)
                    self.assertNotIn(
                        "\n            flash_m_pair = flash_tile_id % flash_num_m_pairs",
                        code,
                    )
                if head_dim == 64:
                    self.assertIn("fa4_disc_exp_convert_store_pipe", code)
                    self.assertNotIn("_flash_tma_o", code)
                    self.assertIn("flash_scale_t", code)
                    self.assertNotIn("storage.alpha0", code)
                    self.assertNotIn("storage.alpha1", code)
                    self.assertNotIn("storage.rowsum0", code)
                    self.assertNotIn("flash_rowsum0_t", code)
                else:
                    self.assertIn("fa4_disc_exp_convert_store_pipe", code)
                    self.assertIn("flash_corr_epi_full_ptr", code)
                    self.assertIn("_flash_tma_o", code)
                    self.assertIn("sO = storage.sO.get_tensor", code)
                    self.assertIn("cp_async_bulk_wait_group(1, read=True)", code)
                    self.assertNotIn("recast_ptr(sQ.iterator, _flash_osl.inner", code)
                    self.assertIn("flash_scale_t", code)
                    self.assertNotIn("storage.alpha0", code)
                    self.assertNotIn("storage.alpha1", code)
                    self.assertNotIn("storage.rowsum0", code)
                    self.assertNotIn("flash_rowsum0_t", code)
                    self.assertNotIn(
                        "flash_s_corr_prod_phase = cutlass.Int32(0)\n"
                        "        flash_corr_epi_empty_phase",
                        code,
                    )
                    self.assertNotIn(
                        "flash_corr_epi_empty_phase ^= 1\n            flash_row_max",
                        code,
                    )
                expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_fa4_deep_one_cta_matches_sdpa(self) -> None:
        cases = (
            (torch.float16, 64, (2, 3, 4)),
            (torch.float16, 128, (2,)),
            (torch.bfloat16, 64, (3,)),
        )
        for dtype, head_dim, kv_stages in cases:
            q, k, v = (
                torch.randn(1, 2, 512, head_dim, dtype=dtype, device=DEVICE)
                for _ in range(3)
            )
            expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            for kv_stage in kv_stages:
                with self.subTest(
                    dtype=str(dtype), head_dim=head_dim, kv_stage=kv_stage
                ):
                    code, out = code_and_output(
                        cute_dense_attention,
                        (q, k, v),
                        block_sizes=[1, 128, 128],
                        cute_flash_pipeline_family="fa4_deep_1cta",
                        cute_flash_kv_stage=kv_stage,
                    )
                    self.assertIn("sV = storage.sV.get_tensor", code)
                    self.assertIn("barrier_storage=storage.k_mbar_ptr.data_ptr()", code)
                    self.assertIn("barrier_storage=storage.v_mbar_ptr.data_ptr()", code)
                    self.assertIn(
                        "barrier_storage=storage.k_mbar_ptr.data_ptr(), "
                        "defer_sync=True",
                        code,
                    )
                    self.assertIn(
                        f"assert _flash_storage_cls.size_in_bytes() == "
                        f"{flash_fa4_shared_storage(head_dim, kv_stage, separate_kv=True).size_in_bytes()}",
                        code,
                    )
                    self.assertIn("flash_k_prod.tail()", code)
                    self.assertIn("flash_v_prod.tail()", code)
                    self.assertNotIn("flash_kv_prod", code)
                    self.assertNotIn("is_two_cta=True", code)
                    self.assertNotIn("PipelineClcFetchAsync", code)
                    self.assertNotIn("sO = storage.sO.get_tensor", code)
                    tol = 3e-2 if dtype is torch.bfloat16 else 1e-2
                    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)

    def test_flash_attention_fa4_sync_schedule_controls_match_sdpa(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                q, k, v = (
                    torch.randn(1, 1, 512, 64, dtype=dtype, device=DEVICE)
                    for _ in range(3)
                )
                code, (out, _lse) = code_and_output(
                    cute_causal_attention,
                    (q, k, v),
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4",
                    cute_flash_mma_interleave=False,
                    cute_flash_wait_hint=0,
                    cute_flash_exp2_packet="8x2",
                )
                steady_start = code.index(
                    "for flash_i in cutlass.range(flash_num_active_kv - 1"
                )
                steady_end = code.index("flash_pfor_phase ^= 1", steady_start)
                steady = code[steady_start:steady_end]
                self.assertLess(
                    steady.index("gemm_ptx_precomputed_pv_ts(flash_o1_addr"),
                    steady.index("flash_k_full ="),
                )
                self.assertIn("pair_batch=8, emu_batch=2", code)
                self.assertIn("flash_pfor_phase, 0)", code)
                self.assertIn("wait_hint=0", code)
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )
                tol = 3e-2 if dtype is torch.bfloat16 else 1e-2
                torch.testing.assert_close(out, expected, atol=tol, rtol=tol)

    def test_flash_attention_single_stat_handoff_persistent_modifier_lse(self) -> None:
        q, k, v = (
            torch.randn(1, 64, 768, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bias = torch.randn(1, 64, 768, 768, dtype=torch.float16, device=DEVICE) * 0.25
        code, (out, lse) = code_and_output(
            cute_biased_attention_with_lse,
            (q, k, v, bias),
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_stat_transport="single",
        )
        self.assertIn("while flash_tile_id < 192", code)
        self.assertNotIn("flash_s_corr_prod_phase", code)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias
        )
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
        self.assertTrue(torch.isfinite(lse).all())

    def test_flash_attention_fa4_deep_one_cta_causal_split_matches_sdpa(
        self,
    ) -> None:
        q, k, v = (
            torch.randn(1, 1, 512, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(64)
        causal_mask = torch.ones(512, 512, dtype=torch.bool, device=DEVICE).tril()
        expected_lse = torch.logsumexp(
            scores.masked_fill(~causal_mask, -torch.inf), dim=-1
        ) * math.log2(math.e)
        for kv_stage in (2, 3, 4):
            with self.subTest(kv_stage=kv_stage):
                code, (out, lse) = code_and_output(
                    cute_causal_attention,
                    (q, k, v),
                    block_sizes=[1, 128, 128],
                    cute_flash_pipeline_family="fa4_deep_1cta",
                    cute_flash_kv_stage=kv_stage,
                    cute_flash_causal_kv_order="descending",
                    cute_flash_causal_loop_split=True,
                )
                self.assertIn("flash_num_active_kv", code)
                self.assertIn("for flash_kv_mask_iter in cutlass.range(", code)
                self.assertIn("for flash_kv_unmask_iter in cutlass.range(", code)
                self.assertIn("flash_k_prod.tail()", code)
                self.assertIn("flash_v_prod.tail()", code)
                self.assertNotIn("flash_kv_prod", code)
                torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
                torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)

    def test_flash_attention_fa4_deep_one_cta_persistent_boundary(self) -> None:
        q, k, v = (
            torch.randn(1, 1, 38400, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_dense_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_deep_1cta",
            cute_flash_kv_stage=2,
            cute_flash_persistent=True,
            cute_flash_persistent_ctas_per_sm=1,
        )
        self.assertIn("while flash_tile_id < 150", code)
        self.assertIn("flash_k_prod.tail()", code)
        self.assertIn("flash_v_prod.tail()", code)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_fa4_causal_two_cta_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 512, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(64)
        causal_mask = torch.ones(512, 512, dtype=torch.bool, device=DEVICE).tril()
        expected_lse = torch.logsumexp(
            scores.masked_fill(~causal_mask, -torch.inf), dim=-1
        ) * math.log2(math.e)

        code, (out, lse) = code_and_output(
            cute_causal_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_2cta_causal",
            cute_flash_kv_stage=2,
            cute_flash_causal_kv_order="descending",
            cute_flash_causal_loop_split=True,
        )

        self.assertIn("cute.arch.cluster_idx()[0]", code)
        self.assertIn("flash_q_mma_tile1 + cutlass.Int32(1)", code)
        self.assertIn("cute_tcgen05_flash.CtaGroup.TWO", code)
        self.assertIn("is_two_cta=True", code)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)

    def test_flash_attention_fa4_causal_two_cta_modifier_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 1, 512, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bias = torch.randn(1, 1, 512, 512, dtype=torch.float16, device=DEVICE) * 0.25
        code, out = code_and_output(
            cute_causal_biased_attention,
            (q, k, v, bias),
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_2cta_causal",
        )
        self.assertIn("add_score_bias_t2r", code)
        self.assertIn("flash_q_mma_tile0 * cutlass.Int32(2)", code)
        causal_mask = torch.ones(512, 512, dtype=torch.bool, device=DEVICE).tril()
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias.masked_fill(~causal_mask, -torch.inf),
        )
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_fa4_causal_two_cta_tma_epilogue_matches_sdpa(
        self,
    ) -> None:
        q, k, v = (
            torch.randn(1, 1, 4096, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, (out, _lse) = code_and_output(
            cute_causal_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4_2cta_causal",
        )
        self.assertIn("_flash_tma_o", code)
        self.assertIn("flash_mma_tile_coord_v", code)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_fa4_zero_threshold_publishes_computed_alpha(
        self,
    ) -> None:
        q = torch.ones(1, 1, 256, 64, dtype=torch.float16, device=DEVICE)
        k = torch.zeros_like(q)
        k[:, :, 128:, :] = math.log(2.0) / 8.0
        v = torch.zeros_like(q)
        v[:, :, :128, :] = 1.0
        code, out = code_and_output(
            cute_dense_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
            cute_flash_topology="fa4",
            cute_flash_softmax_disc=False,
            cute_flash_exp2_impl="split",
            cute_flash_rescale_threshold=0.0,
        )

        exp_call = code.index("flash_p_sum = _helion_flash_rt.fa4_sp_exp_convert_store")
        alpha_compute = code.index("flash_alpha = cute.math.exp2(", exp_call)
        rowsum_update = code.index(
            "flash_row_sum = flash_row_sum * flash_alpha", alpha_compute
        )
        alpha_publish = code.index(" = flash_alpha", alpha_compute, rowsum_update)
        self.assertLess(exp_call, alpha_compute)
        self.assertLess(alpha_compute, alpha_publish)

        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_fa4_fp16_rescale_threshold_stays_finite(self) -> None:
        q = torch.ones(1, 1, 256, 64, dtype=torch.float16, device=DEVICE)
        k = torch.zeros_like(q)
        k[:, :, 128:, :] = 20.0 * math.log(2.0) / 8.0
        v = torch.zeros_like(q)
        v[:, :, :128, :] = 1.0
        with patch.dict(
            os.environ,
            {"HELION_CUTE_FLASH_RESCALE_THRESHOLD": "32"},
            clear=False,
        ):
            code, out = code_and_output(
                cute_dense_attention,
                (q, k, v),
                block_sizes=[1, 128, 128],
                cute_flash_topology="fa4",
                cute_flash_kv_order="ascending",
            )

        self.assertIn("flash_acc_log >= -8.0", code)
        self.assertTrue(torch.isfinite(out).all())
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_fa4_dense_128k_clc_seed_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(2, 32, 131072, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_dense_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
        )

        self.assertIn(
            "flash_clc_pipeline = cutlass_pipeline_flash."
            "PipelineClcFetchAsync.create("
            "barrier_storage=storage.clc_mbar_ptr.data_ptr(), num_stages=2,",
            code,
        )
        self.assertIn("problem_shape_ntile_mnl=(512, 32, 2)", code)
        self.assertIn(
            "flash_m_pair = cutlass.Int32(512 - 1) - "
            "cutlass.Int32(flash_clc_work.tile_idx[0])",
            code,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_fa4_dense_64k_role_chain_seed_matches_sdpa(
        self,
    ) -> None:
        q, k, v = (
            torch.randn(1, 1, 65536, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_dense_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
        )

        self.assertIn("elif warp_idx == 13:", code)
        self.assertIn("elif warp_idx == 14:", code)
        self.assertIn("mbar_ptr=flash_pfor2_ptr + 0", code)
        self.assertIn("fa4_store_o_smem_to_gmem", code)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_fa4_two_cta_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(2, 16, 1024, 128, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        for epi_tma in (True, False):
            with self.subTest(epi_tma=epi_tma):
                code, out = code_and_output(
                    cute_dense_attention,
                    (q, k, v),
                    block_sizes=[1, 128, 128],
                    cute_flash_topology="fa4",
                    cute_flash_persistent=True,
                    cute_flash_use_2cta=True,
                    cute_flash_epi_tma=epi_tma,
                )

                self.assertIn("cta_group=2", code)
                self.assertIn("is_two_cta=True", code)
                self.assertIn("smem_offset=-2048", code)
                self.assertIn("flash_q_mma_tile0 * 2", code)
                self.assertIn("'use_2cta_instrs': True", code)
                self.assertIn(
                    "flash_grid_dim = cutlass.Int32(cute.arch.cluster_dim()[0])",
                    code,
                )
                self.assertIn(f"'epi_tma': {epi_tma}", code)
                if epi_tma:
                    self.assertIn("fa4_correction_epilogue_to_smem_scoped_2cta", code)
                else:
                    self.assertNotIn(
                        "fa4_correction_epilogue_to_smem_scoped_2cta", code
                    )
                torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_fa4_two_cta_clamps_pair_epilogue(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = resolve_flash_config(
                128,
                8,
                {
                    "cute_flash_topology": "fa4",
                    "cute_flash_use_2cta": True,
                    "cute_flash_epi_tma": False,
                    "cute_flash_epi_stg": True,
                    "cute_flash_epi_stg_gmem": "pair",
                },
            )

        self.assertTrue(cfg.use_2cta_instrs)
        self.assertTrue(cfg.epi_stg)
        self.assertEqual(cfg.epi_stg_gmem, "stage")

    def test_flash_attention_fa4_dense_hd64_two_cta_matches_sdpa(self) -> None:
        for seq_len in (32768, 65536, 131072):
            with self.subTest(seq_len=seq_len):
                q, k, v = (
                    torch.randn(1, 1, seq_len, 64, dtype=torch.float16, device=DEVICE)
                    for _ in range(3)
                )
                code, out = code_and_output(
                    cute_dense_attention,
                    (q, k, v),
                    block_sizes=[1, 128, 128],
                    cute_flash_topology="fa4",
                    cute_flash_persistent=True,
                    cute_flash_use_2cta=True,
                    cute_flash_epi_tma=True,
                )

                self.assertIn("cta_group=2", code)
                self.assertIn("is_two_cta=True", code)
                self.assertIn("elif warp_idx < 4:", code)
                self.assertIn("mbarrier_init(flash_pfor_ptr + flash_st, 512)", code)
                self.assertIn("mbarrier_init(flash_pfor2_ptr + flash_st, 256)", code)
                self.assertIn("fa4_correction_epilogue_to_smem_scoped_2cta", code)
                self.assertIn("'use_2cta_instrs': True", code)
                self.assertIn("gemm_ptx_precomputed_qk_static", code)
                self.assertIn(
                    "early_split_publish=True, pair_batch=8, emu_batch=2", code
                )

                expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_clc_preserves_explicit_flattened_geometry(self) -> None:
        q, k, v = (
            torch.randn(2, 32, 8192, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_dense_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
            cute_flash_topology="fa4",
            cute_flash_persistent=True,
            cute_flash_clc=True,
            cute_flash_clc_heads_per_batch=64,
        )

        self.assertIn("problem_shape_ntile_mnl=(32, 64, 1)", code)
        self.assertIn(
            "flash_clc_consumer_state.index * cutlass.Int32(4)).align(16)", code
        )
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_late_rejection_is_safe(self) -> None:
        q, k, v = (
            torch.randn(2, 8, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        with (
            patch(
                "helion._compiler.cute.backend._detect_specialized_mma_loop",
                return_value=True,
            ),
            patch.object(_cute_flash, "codegen_attention_flash", return_value=False),
            self.assertRaisesRegex(BackendUnsupported, "failed late validation"),
        ):
            bound = cute_dense_attention.bind((q, k, v))
            bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))

    def test_flash_attention_bfloat16_fires_and_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(2, 8, 256, 128, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_dense_attention, (q, k, v), block_sizes=[1, 128, 128]
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("cutlass.BFloat16", code)
        self.assertIn("_flash_tma_o", code)
        self.assertIn("sO = storage.sO.get_tensor", code)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)

    def test_flash_attention_causal_fires_and_matches_sdpa(self) -> None:
        for head_dim in (64, 128):
            with (
                self.subTest(head_dim=head_dim),
                patch.dict(
                    os.environ,
                    {"HELION_CUTE_FLASH_TOPOLOGY": "fa4"},
                    clear=False,
                ),
            ):
                q, k, v = (
                    torch.randn(2, 8, 256, head_dim, dtype=torch.float16, device=DEVICE)
                    for _ in range(3)
                )
                code, (out, lse) = code_and_output(
                    cute_causal_attention, (q, k, v), block_sizes=[1, 128, 128]
                )
                self.assertTrue(_flash_fired(code))
                self.assertIn("fa4_disc_rowmax_causal_balanced", code)
                self.assertIn("flash_lpt_group", code)
                self.assertIn("flash_s0_corr_full_ptr", code)
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=True,
                )
                scores = torch.matmul(
                    q.float(), k.float().transpose(-1, -2)
                ) / math.sqrt(head_dim)
                causal_mask = torch.ones(
                    256,
                    256,
                    dtype=torch.bool,
                    device=DEVICE,
                ).tril()
                expected_lse = torch.logsumexp(
                    scores.masked_fill(~causal_mask, -torch.inf), dim=-1
                ) * math.log2(math.e)
                torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
                torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)

    def test_flash_attention_fa4_sload16_balanced_rowmax_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 512, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_dense_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
            cute_flash_pipeline_family="fa4",
            cute_flash_softmax_disc=True,
            cute_flash_p_store_rep=16,
            cute_flash_s_load_rep=16,
        )
        self.assertIn("fa4_disc_rowmax_balanced", code)
        self.assertIn("fa4_disc_exp_convert_store_sload16_pair_pipe", code)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_causal_packed_reduce_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 512, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, (out, _lse) = code_and_output(
            cute_causal_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
            cute_flash_topology="ws_overlap",
            cute_flash_packed_reduce=True,
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("fmax_reduce_packed", code)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_causal_fa4_lpt_residual_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 257, 512, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        with patch.dict(
            os.environ,
            {"HELION_CUTE_FLASH_TOPOLOGY": "fa4"},
            clear=False,
        ):
            code, (out, _lse) = code_and_output(
                cute_causal_attention, (q, k, v), block_sizes=[1, 128, 128]
            )
        self.assertTrue(_flash_fired(code))
        self.assertIn("flash_lpt_group < 1", code)
        self.assertIn("flash_num_active_kv", code)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_causal_fa4_descending_loops_match_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 1, 512, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(64)
        causal_mask = torch.ones(512, 512, dtype=torch.bool, device=DEVICE).tril()
        expected_lse = torch.logsumexp(
            scores.masked_fill(~causal_mask, -torch.inf), dim=-1
        ) * math.log2(math.e)
        boundary_rows = torch.tensor(
            [0, 127, 128, 255, 256, 383, 384, 511],
            device=DEVICE,
        )
        for loop_split in (True, False):
            with self.subTest(loop_split=loop_split):
                code, (out, lse) = code_and_output(
                    cute_causal_attention,
                    (q, k, v),
                    block_sizes=[1, 128, 128],
                    cute_flash_topology="fa4",
                    cute_flash_causal_kv_order="descending",
                    cute_flash_causal_loop_split=loop_split,
                    cute_flash_masked_e2e_schedule="16/4",
                    cute_flash_e2e_schedule="8/2",
                    cute_flash_e2e_offset=0,
                    cute_flash_e2e_offset0=1,
                    cute_flash_disc_pipe=4,
                    cute_flash_role_map="fa4",
                    cute_flash_epi_tma=True,
                    cute_flash_rescale_chunk_cols=16,
                    cute_flash_softmax_regs=200,
                )
                self.assertTrue(_flash_fired(code))
                self.assertIn("fa4_disc_zero_store", code)
                if loop_split:
                    self.assertIn(
                        "for flash_kv_mask_iter in cutlass.range(",
                        code,
                    )
                    self.assertIn(
                        "for flash_kv_unmask_iter in cutlass.range(",
                        code,
                    )
                    self.assertNotIn(
                        "range_constexpr(flash_m_tile",
                        code,
                    )
                    masked_start = code.index(
                        "for flash_kv_mask_iter in cutlass.range("
                    )
                    unmasked_start = code.index(
                        "for flash_kv_unmask_iter in cutlass.range(", masked_start
                    )
                    masked_stage0 = code[masked_start:unmasked_start]
                    alpha_store = masked_stage0.index(
                        "flash_scale_t[flash_s_corr_prod_index, 0, "
                        "flash_local_tidx] = flash_alpha"
                    )
                    alpha_publish = masked_stage0.index(
                        "cute.arch.barrier_arrive(", alpha_store
                    )
                    self.assertEqual(masked_stage0.count("flash_minus_max_scale ="), 1)
                    minus_scale = masked_stage0.index("flash_minus_max_scale =")
                    probability_pass = masked_stage0.index(
                        "flash_p_sum = cutlass.Float32(0.0)", minus_scale
                    )
                    self.assertLess(alpha_store, alpha_publish)
                    self.assertLess(alpha_publish, minus_scale)
                    self.assertLess(minus_scale, probability_pass)
                else:
                    self.assertIn("for flash_kv_iter in", code)
                    self.assertNotIn("flash_kv_mask_iter", code)
                torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
                torch.testing.assert_close(
                    lse[:, :, boundary_rows],
                    expected_lse[:, :, boundary_rows],
                    atol=2e-2,
                    rtol=2e-2,
                )

    def test_flash_attention_causal_fa4_split_bfloat16_boundaries(self) -> None:
        q, k, v = (
            torch.randn(1, 1, 256, 64, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(3)
        )
        code, (out, lse) = code_and_output(
            cute_causal_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
            cute_flash_topology="fa4",
            cute_flash_causal_kv_order="descending",
            cute_flash_causal_loop_split=True,
            cute_flash_softmax_disc=True,
        )
        self.assertIn("for flash_kv_unmask_iter in cutlass.range(", code)
        self.assertNotIn("range_constexpr(flash_m_tile", code)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(64)
        causal_mask = torch.ones(256, 256, dtype=torch.bool, device=DEVICE).tril()
        expected_lse = torch.logsumexp(
            scores.masked_fill(~causal_mask, -torch.inf), dim=-1
        ) * math.log2(math.e)
        boundary_rows = torch.tensor([0, 127, 128, 255], device=DEVICE)
        torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(
            lse[:, :, boundary_rows],
            expected_lse[:, :, boundary_rows],
            atol=4e-2,
            rtol=4e-2,
        )

    def test_flash_attention_causal_single_warpgroup_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 512, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, (out, _lse) = code_and_output(
            cute_causal_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
            cute_flash_s_stage=1,
            cute_flash_topology="ws_overlap",
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("flash_kv >= flash_m_tile", code)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_dense_degree1_packet_matches_sdpa(self) -> None:
        cases = (
            (32768, "deg1_8x2_corr10", 5, 1),
            (65536, "deg1_8x2_corr10", 2, 1),
            (131072, "deg1_8x2_corr10", 2, 1),
            (196608, "deg1_8x2_corr10", 2, 1),
            (262144, "deg1_16x8", 0, 10),
            (262656, "deg1_16x8", 0, 10),
        )
        for sequence_length, packet, offset, offset0 in cases:
            with self.subTest(sequence_length=sequence_length):
                torch.manual_seed(1000 + sequence_length)
                q, k, v = (
                    torch.randn(
                        1,
                        1,
                        sequence_length,
                        64,
                        dtype=torch.float16,
                        device=DEVICE,
                    )
                    for _ in range(3)
                )
                config: dict[str, object] = {
                    "block_sizes": [1, 128, 128],
                    "cute_flash_pipeline_family": "fa4_2cta",
                    "cute_flash_exp2_packet": packet,
                    "cute_flash_e2e_offset": offset,
                    "cute_flash_e2e_offset0": offset0,
                }
                if packet == "deg1_16x8":
                    config["cute_flash_kv_stage"] = 2
                code, out = code_and_output(
                    cute_dense_attention,
                    (q, k, v),
                    **config,
                )
                self.assertIn("is_two_cta=True", code)
                self.assertIn("degree1=True", code)
                expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                torch.testing.assert_close(out, expected, atol=0.05, rtol=0.02)

    def test_flash_attention_dense_degree1_seed_requires_standard_output(self) -> None:
        target = torch.empty(1, 1, 262_144, 64, dtype=torch.float16, device=DEVICE)
        cases = (
            (cute_dense_attention, True),
            (cute_dense_attention_with_lse, False),
            (cute_softcap_attention, False),
        )
        for kernel, expected_standard in cases:
            with self.subTest(kernel=kernel.__name__):
                bound = kernel.bind((target, target, target))
                self.assertEqual(
                    bound.config_spec._cute_flash_standard_dense_output,
                    expected_standard,
                )
                has_degree1_seed = any(
                    seed.config.get(_cute_flash.FLASH_EXP2_PACKET_KEY)
                    in {"deg1_8x2_corr10", "deg1_16x8"}
                    for seed in bound.config_spec.compiler_seed_configs
                )
                self.assertEqual(has_degree1_seed, expected_standard)

    def test_flash_attention_hybrid_seed_excludes_auxiliary_outputs(self) -> None:
        target = torch.empty(1, 1, 65_536, 64, dtype=torch.float16, device=DEVICE)
        slopes = torch.empty(1, dtype=torch.float32, device=DEVICE)
        cases = (
            (cute_causal_attention, (target, target, target)),
            (cute_alibi_attention, (target, target, target, slopes)),
        )
        for kernel, args in cases:
            with self.subTest(kernel=kernel.__name__):
                bound = kernel.bind(args)
                self.assertFalse(bound.config_spec._cute_flash_standard_causal_output)
                self.assertFalse(
                    any(
                        seed.config.get(_cute_flash.FLASH_EXP2_PACKET_KEY)
                        == "hybrid_deg1_16x8"
                        for seed in bound.config_spec.compiler_seed_configs
                    )
                )

    def test_flash_attention_bias_fires_and_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 128, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bias = torch.randn(1, 2, 128, 128, dtype=torch.float16, device=DEVICE) * 0.25
        bound = cute_biased_attention.bind((q, k, v, bias))
        code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
        self.assertTrue(_flash_fired(code))
        self.assertIn("helion_small_biased_attention", code)
        self.assertNotIn("add_score_bias_t2r", code)
        self.assertNotIn("_flash_mBias", code)
        self.assertNotIn("flash_shared_storage", code)
        self.assertNotIn("cute.gemm", code)
        self.assertNotIn("_helion_cute_disable_bake_tensor_shapes", code)
        self.assertNotIn("layout.stride", code)
        self.assertNotIn("for flash_j in cutlass.range_constexpr(flash_n)", code)
        self.assertNotIn("flash_m_pair", code)
        packed_code = bound.to_triton_code(
            helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_packed_reduce=True,
            )
        )
        self.assertIn("helion_small_biased_attention", packed_code)
        self.assertNotIn("fmax_reduce_packed", packed_code)
        self.assertNotIn("fadd_reduce_packed", packed_code)
        generic_code = bound.to_triton_code(
            helion.Config(
                block_sizes=[1, 128, 128],
                cute_flash_small_biased=False,
            )
        )
        self.assertTrue(_flash_fired(generic_code))
        self.assertNotIn("helion_small_biased_attention", generic_code)
        self.assertIn("add_score_bias_t2r", generic_code)
        _code, out = code_and_output(
            cute_biased_attention,
            (q, k, v, bias),
            block_sizes=[1, 128, 128],
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
        )
        # The 64-thread small path uses a different fp32 reduction order than SDPA.
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
        _generic_code, generic_out = code_and_output(
            cute_biased_attention,
            (q, k, v, bias),
            block_sizes=[1, 128, 128],
            cute_flash_small_biased=False,
        )
        torch.testing.assert_close(generic_out, expected, atol=2e-2, rtol=2e-2)

    def test_flash_attention_bias_all_inf_row_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 128, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bias = torch.randn(1, 2, 128, 128, dtype=torch.float16, device=DEVICE) * 0.25
        bias[:, :, 7, :] = -torch.inf
        code, out = code_and_output(
            cute_biased_attention,
            (q, k, v, bias),
            block_sizes=[1, 128, 128],
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("helion_small_biased_attention", code)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
        )
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(out[:, :, 7, :], torch.zeros_like(out[:, :, 7, :]))

    def test_flash_attention_bias_generic_fires_and_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bias = torch.randn(1, 2, 256, 256, dtype=torch.float16, device=DEVICE) * 0.25
        bound = cute_biased_attention.bind((q, k, v, bias))
        code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
        self.assertTrue(_flash_fired(code))
        self.assertIn("add_score_bias_t2r", code)
        self.assertIn("_flash_mBias", code)
        self.assertIn("flash_fa4_shared_storage", code)
        _assert_score_modified_reductions(self, code)
        self.assertNotIn("helion_small_biased_attention", code)
        _code, out = code_and_output(
            cute_biased_attention,
            (q, k, v, bias),
            block_sizes=[1, 128, 128],
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
        )
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_stage_local_softmax_bias_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bias = torch.randn(1, 2, 256, 256, dtype=torch.float16, device=DEVICE) * 0.25
        with patch.dict(
            os.environ,
            {"HELION_CUTE_FLASH_STAGE_LOCAL_SOFTMAX_SETUP": "1"},
            clear=False,
        ):
            code, out = code_and_output(
                cute_biased_attention,
                (q, k, v, bias),
                block_sizes=[1, 128, 128],
                cute_flash_small_biased=False,
            )
        self.assertTrue(_flash_fired(code))
        self.assertIn("flash_tiled_ld_coord", code)
        self.assertIn("add_score_bias_t2r", code)
        self.assertNotIn("helion_small_biased_attention", code)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
        )
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_bias_with_lse_fires_and_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 128, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bias = torch.randn(1, 2, 128, 128, dtype=torch.float16, device=DEVICE) * 0.25
        bound = cute_biased_attention_with_lse.bind((q, k, v, bias))
        flash_fragments = bound.config_spec._flat_fields()
        self.assertEqual(
            flash_fragments[_cute_flash.FLASH_SMALL_BIASED_KEY].search_choices,
            (True,),
        )
        code, (out, lse) = code_and_output(
            cute_biased_attention_with_lse,
            (q, k, v, bias),
            block_sizes=[1, 128, 128],
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("add_score_bias_t2r", code)
        self.assertIn("exp2_split_inplace", code)
        self.assertNotIn("for flash_j in cutlass.range_constexpr(flash_n)", code)
        self.assertIn("0.6931471805599453", code)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
        )
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(64)
        expected_lse = torch.logsumexp(scores + bias.float(), dim=-1)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)

    def test_flash_attention_causal_bias_fires_and_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bias = torch.randn(1, 2, 256, 256, dtype=torch.float16, device=DEVICE) * 0.25
        code, out = code_and_output(
            cute_causal_biased_attention,
            (q, k, v, bias),
            block_sizes=[1, 128, 128],
            cute_flash_topology="fa4",
            cute_flash_causal_kv_order="descending",
            cute_flash_causal_loop_split=True,
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("add_score_bias_t2r", code)
        self.assertIn("causal_mask_t2r", code)
        self.assertIn("flash_fa4_shared_storage", code)
        self.assertNotIn("flash_kv_unmask_iter", code)
        _assert_score_modified_reductions(self, code)
        modifier_loop = code.index("for flash_kv_iter in cutlass.range(")
        minus_scale = code.index("flash_minus_max_scale =", modifier_loop)
        alpha_store = code.index(
            "flash_scale_t[flash_s_corr_prod_index, 0, flash_local_tidx] = flash_alpha",
            modifier_loop,
        )
        self.assertLess(minus_scale, alpha_store)
        causal_mask = torch.ones(256, 256, dtype=torch.bool, device=DEVICE).tril()
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias.masked_fill(~causal_mask, -torch.inf),
        )
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_causal_bias_ws_overlap_matches_sdpa(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bias = torch.randn(1, 2, 256, 256, dtype=torch.float16, device=DEVICE) * 0.25
        code, out = code_and_output(
            cute_causal_biased_attention,
            (q, k, v, bias),
            block_sizes=[1, 128, 128],
            cute_flash_topology="ws_overlap",
            cute_flash_packed_reduce=True,
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("flash_shared_storage", code)
        self.assertNotIn("flash_fa4_shared_storage", code)
        self.assertIn("add_score_bias_t2r", code)
        self.assertIn("causal_mask_t2r", code)
        self.assertIn("fmax_reduce_packed", code)
        causal_mask = torch.ones(256, 256, dtype=torch.bool, device=DEVICE).tril()
        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias.masked_fill(~causal_mask, -torch.inf),
        )
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_relative_bias_fires_and_matches_reference(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_relative_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("add_relative_bias_t2r", code)
        self.assertIn("flash_fa4_shared_storage", code)
        _assert_score_modified_reductions(self, code)
        row = torch.arange(256, device=DEVICE)[:, None]
        col = torch.arange(256, device=DEVICE)[None, :]
        scores = (
            torch.matmul(q.float(), k.float().transpose(-1, -2))
            * (math.log2(math.e) / math.sqrt(64))
            + (row - col) * 0.01
        )
        expected = _attention_from_log2_scores(scores, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_relative_bias_ws_overlap_matches_reference(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_relative_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
            cute_flash_topology="ws_overlap",
            cute_flash_packed_reduce=True,
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("flash_shared_storage", code)
        self.assertNotIn("flash_fa4_shared_storage", code)
        self.assertIn("add_relative_bias_t2r", code)
        self.assertIn("fmax_reduce_packed", code)
        row = torch.arange(256, device=DEVICE)[:, None]
        col = torch.arange(256, device=DEVICE)[None, :]
        scores = (
            torch.matmul(q.float(), k.float().transpose(-1, -2))
            * (math.log2(math.e) / math.sqrt(64))
            + (row - col) * 0.01
        )
        expected = _attention_from_log2_scores(scores, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_alibi_fires_and_matches_reference(self) -> None:
        q, k, v = (
            torch.randn(2, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        slopes = torch.tensor([0.01, 0.03], dtype=torch.float32, device=DEVICE)
        code, out = code_and_output(
            cute_alibi_attention,
            (q, k, v, slopes),
            block_sizes=[1, 128, 128],
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("add_alibi_bias_t2r", code)
        self.assertIn("causal_mask_t2r", code)
        self.assertIn("flash_fa4_shared_storage", code)
        _assert_score_modified_reductions(self, code)
        row = torch.arange(256, device=DEVICE)[:, None]
        col = torch.arange(256, device=DEVICE)[None, :]
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (
            math.log2(math.e) / math.sqrt(64)
        )
        scores = scores + (col - row) * slopes.view(1, 2, 1, 1)
        scores = scores.masked_fill(row < col, -torch.inf)
        expected = _attention_from_log2_scores(scores, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_declines_alibi_mod_divisor_mismatch(self) -> None:
        q, k, v = (
            torch.randn(2, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        slopes = torch.tensor([0.01, 0.02, 0.03, 0.04], device=DEVICE)
        bound = cute_alibi_attention.bind((q, k, v, slopes))
        code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
        self.assertFalse(_flash_fired(code))

    def test_flash_attention_sliding_window_fires_and_matches_reference(self) -> None:
        q, k, v = (
            torch.randn(1, 1, 768, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        with patch.dict(
            os.environ,
            {"HELION_CUTE_FLASH_PERSISTENT": "1"},
            clear=False,
        ):
            code, out = code_and_output(
                cute_sliding_window_attention,
                (q, k, v),
                block_sizes=[1, 128, 128],
            )
        self.assertTrue(_flash_fired(code))
        self.assertIn("sliding_window_mask_t2r", code)
        self.assertIn("fmax_reduce_packed", code)
        self.assertIn("while flash_tile_id < _flash_total_tiles", code)
        self.assertIn("flash_first_kv", code)
        self.assertIn(
            "for flash_active_kv in cutlass.range(flash_active_count, unroll=1)",
            code,
        )
        self.assertIn("flash_kv + cutlass.Int32(4)", code)
        row = torch.arange(768, device=DEVICE)[:, None]
        col = torch.arange(768, device=DEVICE)[None, :]
        delta = row - col
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (
            math.log2(math.e) / math.sqrt(64)
        )
        scores = scores.masked_fill((delta < 0) | (delta > 64), -torch.inf)
        expected = _attention_from_log2_scores(scores, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_prefix_lm_long_prefix_prunes_range(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 384, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_prefix_lm_attention_long_prefix,
            (q, k, v),
            block_sizes=[1, 128, 128],
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("prefix_lm_mask_t2r", code)
        self.assertIn("fmax_reduce_packed", code)
        self.assertIn("cutlass.max(flash_m_tile, cutlass.Int32(1))", code)
        row = torch.arange(384, device=DEVICE)[:, None]
        col = torch.arange(384, device=DEVICE)[None, :]
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (
            math.log2(math.e) / math.sqrt(64)
        )
        scores = scores.masked_fill(~((col < 192) | (row >= col)), -torch.inf)
        expected = _attention_from_log2_scores(scores, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_declines_shifted_index_mask(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bound = cute_shifted_causal_attention.bind((q, k, v))
        code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
        self.assertFalse(_flash_fired(code))

    def test_flash_attention_declines_duplicate_window_mask(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bound = cute_duplicate_window_attention.bind((q, k, v))
        code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
        self.assertFalse(_flash_fired(code))

    def test_flash_attention_prefix_lm_fires_and_matches_reference(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_prefix_lm_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("prefix_lm_mask_t2r", code)
        self.assertIn("fmax_reduce_packed", code)
        row = torch.arange(256, device=DEVICE)[:, None]
        col = torch.arange(256, device=DEVICE)[None, :]
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (
            math.log2(math.e) / math.sqrt(64)
        )
        scores = scores.masked_fill(~((col < 64) | (row >= col)), -torch.inf)
        expected = _attention_from_log2_scores(scores, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_document_mask_fires_and_matches_reference(self) -> None:
        q, k, v = (
            torch.randn(2, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        document_ids = torch.arange(256, device=DEVICE, dtype=torch.int32).div(
            64,
            rounding_mode="floor",
        )
        document_ids = document_ids.expand(2, 256).contiguous()
        with patch.dict(
            os.environ,
            {"HELION_CUTE_FLASH_PERSISTENT": "1"},
            clear=False,
        ):
            code, out = code_and_output(
                cute_document_mask_attention,
                (q, k, v, document_ids),
                block_sizes=[1, 128, 128],
            )
        self.assertTrue(_flash_fired(code))
        self.assertIn("document_mask_t2r", code)
        self.assertIn("fmax_reduce_packed", code)
        self.assertIn("while flash_tile_id < _flash_total_tiles", code)
        self.assertIn("flash_active_count", code)
        row = torch.arange(256, device=DEVICE)[:, None]
        col = torch.arange(256, device=DEVICE)[None, :]
        doc = document_ids
        same_doc = doc[:, None, :, None] == doc[:, None, None, :]
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (
            math.log2(math.e) / math.sqrt(64)
        )
        scores = scores.masked_fill(~((row >= col) & same_doc), -torch.inf)
        expected = _attention_from_log2_scores(scores, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_document_mask_doc_id_collisions_match_reference(
        self,
    ) -> None:
        q, k, v = (
            torch.randn(1, 2, 384, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        document_ids = torch.arange(384, device=DEVICE, dtype=torch.int64)
        document_ids = document_ids.expand(1, 384).contiguous()
        code, out = code_and_output(
            cute_document_mask_attention,
            (q, k, v, document_ids),
            block_sizes=[1, 128, 128],
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("_document_tile_bits_warp", code)
        self.assertIn("fmax_reduce_packed", code)
        torch.testing.assert_close(out, v, atol=1e-2, rtol=1e-2)

    def test_flash_attention_declines_document_floordiv_mismatch(self) -> None:
        q, k, v = (
            torch.randn(2, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        document_ids = torch.arange(256, device=DEVICE, dtype=torch.int32).div(
            64,
            rounding_mode="floor",
        )
        document_ids = document_ids.expand(4, 256).contiguous()
        bound = cute_document_mask_attention.bind((q, k, v, document_ids))
        code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
        self.assertFalse(_flash_fired(code))

    def test_flash_attention_declines_duplicate_document_mask(self) -> None:
        q, k, v = (
            torch.randn(2, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        document_ids = torch.arange(256, device=DEVICE, dtype=torch.int32).div(
            64,
            rounding_mode="floor",
        )
        document_ids = document_ids.expand(2, 256).contiguous()
        bound = cute_duplicate_document_mask_attention.bind((q, k, v, document_ids))
        code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
        self.assertFalse(_flash_fired(code))

    def test_flash_attention_softcap_fires_and_matches_reference(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_softcap_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("softcap_t2r", code)
        self.assertIn("flash_fa4_shared_storage", code)
        _assert_score_modified_reductions(self, code)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (
            math.log2(math.e) / math.sqrt(64)
        )
        scores = 2.0 * torch.tanh(scores / 2.0)
        expected = _attention_from_log2_scores(scores, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_softcap_ws_overlap_matches_reference(self) -> None:
        q, k, v = (
            torch.randn(1, 2, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_softcap_attention,
            (q, k, v),
            block_sizes=[1, 128, 128],
            cute_flash_topology="ws_overlap",
            cute_flash_packed_reduce=True,
        )
        self.assertTrue(_flash_fired(code))
        self.assertIn("flash_shared_storage", code)
        self.assertNotIn("flash_fa4_shared_storage", code)
        self.assertIn("softcap_t2r", code)
        self.assertIn("fmax_reduce_packed", code)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (
            math.log2(math.e) / math.sqrt(64)
        )
        scores = 2.0 * torch.tanh(scores / 2.0)
        expected = _attention_from_log2_scores(scores, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_causal_ws_generated_bodies_parse(self) -> None:
        io_dtype = _cute_flash._flash_io_dtype_str(torch.float16)
        for num_kv in (1, 2, 4):
            with self.subTest(num_kv=num_kv):
                cfg = resolve_flash_config(
                    64,
                    num_kv,
                    {_cute_flash.FLASH_TOPOLOGY_KEY: "ws_overlap"},
                    is_causal=True,
                )
                self.assertFalse(cfg.persistent)
                ast.parse(
                    "if True:\n"
                    + _cute_flash._flash_ws_producer_body(
                        num_kv,
                        cfg.kv_stage,
                        64,
                        score_plan=causal_score_plan(64),
                    )
                )
                ast.parse(
                    "if True:\n"
                    + _cute_flash._flash_ws_consumer_body(
                        64,
                        num_kv,
                        cfg,
                        io_dtype=io_dtype,
                        score_plan=causal_score_plan(64),
                    )
                )

    def test_flash_attention_tuple_output_matches_lse(self) -> None:
        q, k, v = (
            torch.randn(2, 8, 512, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(64)
        expected_lse = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
        expected_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        for topology in ("ws_overlap", "fa4"):
            with (
                self.subTest(topology=topology),
                patch.dict(
                    os.environ,
                    {"HELION_CUTE_FLASH_TOPOLOGY": topology},
                    clear=False,
                ),
            ):
                code, (out, lse) = code_and_output(
                    cute_dense_attention_with_lse,
                    (q, k, v),
                    block_sizes=[1, 128, 128],
                )
                self.assertTrue(_flash_fired(code))
                if topology == "fa4":
                    self.assertNotIn("flash_lse_m_pair", code)
                torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
                torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)

    def test_flash_attention_fa4_clamps_aliased_kv_ring_min_depth(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_KV_STAGE": "1",
            },
            clear=False,
        ):
            cfg = resolve_flash_config(64, 2)
        self.assertEqual(cfg.topology, "fa4")
        self.assertEqual(cfg.kv_stage, 2)

        cfg = resolve_flash_config(
            64,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_kv_stage": 1,
            },
        )
        self.assertEqual(cfg.topology, "fa4")
        self.assertEqual(cfg.kv_stage, 2)

        cfg = resolve_flash_config(64, 2, is_causal=True)
        self.assertEqual(cfg.topology, "fa4")
        self.assertFalse(cfg.persistent)
        self.assertEqual(cfg.kv_stage, 2)
        cfg = resolve_flash_config(
            64,
            2,
            {"cute_flash_topology": "fa4"},
            is_causal=True,
        )
        self.assertEqual(cfg.topology, "fa4")
        self.assertFalse(cfg.persistent)
        self.assertEqual(cfg.kv_stage, 2)

    def test_flash_attention_deep_one_cta_resource_normalization(self) -> None:
        self.assertEqual(
            [
                flash_fa4_shared_storage(64, stage, separate_kv=True).size_in_bytes()
                for stage in (2, 3, 4)
            ],
            [101_376, 134_144, 166_912],
        )
        self.assertEqual(
            flash_fa4_shared_storage(128, 2, separate_kv=True).size_in_bytes(),
            199_680,
        )
        family = {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_deep_1cta"}
        for requested, expected in ((1, 2), (2, 2), (3, 3), (4, 4), (8, 4)):
            with self.subTest(head_dim=64, requested=requested):
                cfg = resolve_flash_config(
                    64,
                    512,
                    {
                        **family,
                        _cute_flash.FLASH_KV_STAGE_KEY: requested,
                        _cute_flash.FLASH_EPI_TMA_KEY: True,
                        _cute_flash.FLASH_EPI_STG_KEY: True,
                    },
                )
                self.assertEqual(cfg.pipeline_family, "fa4_deep_1cta")
                self.assertTrue(cfg.separate_kv_rings)
                self.assertEqual(cfg.kv_stage, expected)
                self.assertEqual(cfg.s_stage, 2)
                self.assertFalse(cfg.epi_tma)
                self.assertFalse(cfg.epi_stg)
                self.assertFalse(cfg.use_2cta_instrs)
                self.assertFalse(cfg.use_cga2_local_cta)
                self.assertFalse(cfg.use_clc_scheduler)
                self.assertFalse(cfg.local_tma_partition)
                self.assertFalse(cfg.tensor_4d_tma)

        hd128 = resolve_flash_config(
            128,
            512,
            {**family, _cute_flash.FLASH_KV_STAGE_KEY: 4},
        )
        self.assertEqual(hd128.pipeline_family, "fa4_deep_1cta")
        self.assertEqual(hd128.kv_stage, 2)

    def test_flash_attention_unsupported_family_variants_normalize_atomically(
        self,
    ) -> None:
        unsupported = (
            resolve_flash_config(
                64,
                512,
                {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: ("fa4_cga2_causal_multicast")},
                is_causal=True,
            ),
            resolve_flash_config(
                64,
                512,
                {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta_causal"},
                dtype=torch.bfloat16,
                is_causal=True,
            ),
            resolve_flash_config(
                64,
                512,
                {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta_causal_deep"},
                is_causal=True,
            ),
        )
        for cfg in unsupported:
            self.assertEqual(cfg.pipeline_family, "fa4")
            self.assertFalse(cfg.separate_kv_rings)
            self.assertFalse(cfg.causal_two_cta)
            self.assertFalse(cfg.use_2cta_instrs)
            self.assertFalse(cfg.use_cga2_local_cta)

    def test_flash_attention_fa4_disc_pipe_defaults_by_head_dim(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
            },
            clear=False,
        ):
            self.assertEqual(resolve_flash_config(64, 2).disc_pipe_depth, 4)
            self.assertEqual(
                resolve_flash_config(64, 2, is_causal=True).disc_pipe_depth, 2
            )
            self.assertEqual(resolve_flash_config(128, 2).disc_pipe_depth, 2)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_DISC_PIPE": "1",
            },
            clear=False,
        ):
            self.assertEqual(resolve_flash_config(64, 2).disc_pipe_depth, 1)

        cfg = resolve_flash_config(
            128,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_disc_pipe": 3,
            },
        )
        self.assertEqual(cfg.disc_pipe_depth, 3)

    def test_flash_attention_fa4_e2e_schedule_defaults_by_head_dim(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
            },
            clear=True,
        ):
            cfg64 = resolve_flash_config(64, 2)
            cfg128 = resolve_flash_config(128, 2)
        self.assertEqual((cfg64.e2e_freq, cfg64.e2e_res), (16, 4))
        self.assertEqual(cfg64.e2e_schedule, "16/4")
        self.assertEqual(cfg64.masked_e2e_schedule, "inherit")
        self.assertEqual((cfg64.masked_e2e_freq, cfg64.masked_e2e_res), (16, 4))
        self.assertEqual(cfg64.e2e_offset, 2)
        self.assertEqual((cfg128.e2e_freq, cfg128.e2e_res), (8, 2))
        self.assertEqual(cfg128.e2e_schedule, "8/2")
        self.assertEqual(cfg128.e2e_offset, 0)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_E2E_SCHEDULE": "xu",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(64, 2)
        self.assertEqual(cfg.exp2_impl, "xu")
        self.assertEqual((cfg.e2e_freq, cfg.e2e_res), (8, 0))
        self.assertEqual(cfg.e2e_schedule, "xu")
        self.assertEqual(cfg.masked_e2e_schedule, "inherit")
        self.assertEqual((cfg.masked_e2e_freq, cfg.masked_e2e_res), (8, 0))
        self.assertEqual(cfg.e2e_offset, 0)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_E2E_FREQ": "8",
                "HELION_CUTE_FLASH_E2E_RES": "2",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(64, 2)
        self.assertEqual((cfg.e2e_freq, cfg.e2e_res), (8, 2))
        self.assertEqual(cfg.e2e_schedule, "8/2")

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_EXP2_IMPL": "xu",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(64, 2)
        self.assertEqual(cfg.exp2_impl, "xu")
        self.assertEqual(cfg.e2e_res, 0)
        self.assertEqual(cfg.e2e_schedule, "xu")
        self.assertEqual(cfg.e2e_offset, 0)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_E2E_SCHEDULE": "xu",
                "HELION_CUTE_FLASH_EXP2_IMPL": "split",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(64, 2)
        self.assertEqual((cfg.exp2_impl, cfg.e2e_freq, cfg.e2e_res), ("split", 16, 4))
        self.assertEqual(cfg.e2e_schedule, "16/4")
        self.assertEqual(cfg.e2e_offset, 2)

        cfg = resolve_flash_config(
            128,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_e2e_schedule": "16/4",
            },
        )
        self.assertEqual((cfg.e2e_freq, cfg.e2e_res), (16, 4))
        self.assertEqual(cfg.e2e_schedule, "16/4")
        self.assertEqual(cfg.e2e_offset, 0)

        cfg = resolve_flash_config(
            64,
            64,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_e2e_schedule": "16/4",
                "cute_flash_masked_e2e_schedule": "xu",
            },
            is_causal=True,
        )
        self.assertEqual(cfg.e2e_schedule, "16/4")
        self.assertEqual((cfg.e2e_freq, cfg.e2e_res), (16, 4))
        self.assertEqual(cfg.masked_e2e_schedule, "xu")
        self.assertEqual((cfg.masked_e2e_freq, cfg.masked_e2e_res), (8, 0))

        cfg = resolve_flash_config(
            64,
            64,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_e2e_schedule": "xu",
                "cute_flash_masked_e2e_schedule": "16/4",
                "cute_flash_e2e_offset": 15,
                "cute_flash_e2e_offset0": 14,
            },
            is_causal=True,
        )
        self.assertEqual(cfg.e2e_schedule, "xu")
        self.assertEqual(cfg.masked_e2e_schedule, "16/4")
        self.assertEqual(cfg.e2e_offset, 15)
        self.assertEqual(cfg.e2e_offset0, 14)

        cfg = resolve_flash_config(
            64,
            64,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_e2e_schedule": "8/2",
                "cute_flash_masked_e2e_schedule": "16/4",
                "cute_flash_e2e_offset": 15,
            },
            is_causal=True,
        )
        self.assertEqual(cfg.e2e_schedule, "8/2")
        self.assertEqual(cfg.masked_e2e_schedule, "16/4")
        self.assertEqual(cfg.e2e_offset, 15)

        cfg = resolve_flash_config(
            64,
            64,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_masked_e2e_schedule": "xu",
            },
        )
        self.assertEqual(cfg.masked_e2e_schedule, "inherit")

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_EXP2_IMPL": "xu",
                "HELION_CUTE_FLASH_E2E_FREQ": "8",
                "HELION_CUTE_FLASH_E2E_RES": "2",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(
                128,
                2,
                {
                    "cute_flash_topology": "fa4",
                    "cute_flash_e2e_schedule": "16/4",
                },
            )
        self.assertEqual((cfg.exp2_impl, cfg.e2e_freq, cfg.e2e_res), ("split", 16, 4))
        self.assertEqual(cfg.e2e_schedule, "16/4")
        self.assertEqual(cfg.e2e_offset, 0)

        cfg = resolve_flash_config(
            64,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_e2e_offset": 4,
            },
        )
        self.assertEqual(cfg.e2e_offset, 4)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_E2E_OFFSET": "12",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(64, 2)
        self.assertEqual(cfg.e2e_offset, 12)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_E2E_OFFSET": "-1",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(64, 2)
        self.assertEqual(cfg.e2e_offset, 2)

        cfg = resolve_flash_config(
            64,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_e2e_offset": -1,
            },
        )
        self.assertEqual(cfg.e2e_offset, 2)

        cfg = resolve_flash_config(
            64,
            64,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_e2e_schedule": "8/2",
                "cute_flash_e2e_offset": -1,
            },
            is_causal=True,
        )
        self.assertEqual(cfg.e2e_offset, 0)

        cfg = resolve_flash_config(
            64,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_e2e_schedule": "xu",
                "cute_flash_exp2_impl": "split",
            },
        )
        self.assertEqual((cfg.exp2_impl, cfg.e2e_freq, cfg.e2e_res), ("split", 16, 4))
        self.assertEqual(cfg.e2e_schedule, "16/4")
        self.assertEqual(cfg.e2e_offset, 2)

        cfg = resolve_flash_config(
            64,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_e2e_freq": 0,
                "cute_flash_e2e_res": 4,
            },
        )
        self.assertEqual((cfg.exp2_impl, cfg.e2e_freq, cfg.e2e_res), ("split", 16, 4))
        self.assertEqual(cfg.e2e_schedule, "16/4")
        self.assertEqual(cfg.e2e_offset, 2)

        cfg = resolve_flash_config(
            128,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_e2e_freq": 16,
                "cute_flash_e2e_res": 4,
            },
        )
        self.assertEqual((cfg.e2e_freq, cfg.e2e_res), (16, 4))
        self.assertEqual(cfg.e2e_schedule, "16/4")
        self.assertEqual(cfg.e2e_offset, 0)

    def test_flash_attention_fa4_epi_tma_defaults_by_head_dim(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
            },
            clear=False,
        ):
            self.assertFalse(resolve_flash_config(64, 2).epi_tma)
            self.assertTrue(resolve_flash_config(128, 2).epi_tma)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_EPI_TMA": "0",
            },
            clear=False,
        ):
            self.assertFalse(resolve_flash_config(128, 2).epi_tma)

        cfg = resolve_flash_config(
            64,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_epi_tma": True,
            },
        )
        self.assertTrue(cfg.epi_tma)

    def test_flash_attention_fa4_rescale_threshold_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
            },
            clear=True,
        ):
            self.assertEqual(resolve_flash_config(64, 2).rescale_threshold, 8.0)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_RESCALE_THRESHOLD": "12",
            },
            clear=True,
        ):
            self.assertEqual(resolve_flash_config(64, 2).rescale_threshold, 12.0)

        cfg = resolve_flash_config(
            64,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_rescale_threshold": 4.0,
            },
        )
        self.assertEqual(cfg.rescale_threshold, 4.0)

        fp16_cfg = resolve_flash_config(
            64,
            2,
            {"cute_flash_rescale_threshold": 32.0},
            dtype=torch.float16,
        )
        bf16_cfg = resolve_flash_config(
            64,
            2,
            {"cute_flash_rescale_threshold": 32.0},
            dtype=torch.bfloat16,
        )
        self.assertEqual(fp16_cfg.rescale_threshold, 8.0)
        self.assertEqual(bf16_cfg.rescale_threshold, 32.0)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_RESCALE_THRESHOLD": "16",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(
                64,
                2,
                {
                    "cute_flash_rescale_threshold": 0.0,
                },
            )
        self.assertEqual(cfg.rescale_threshold, 0.0)

    def test_flash_attention_fa4_rescale_chunk_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
            },
            clear=True,
        ):
            self.assertEqual(resolve_flash_config(64, 2).rescale_chunk_cols, 32)
            self.assertEqual(resolve_flash_config(128, 2).rescale_chunk_cols, 16)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_RESCALE_CHUNK_COLS": "64",
            },
            clear=True,
        ):
            self.assertEqual(resolve_flash_config(64, 2).rescale_chunk_cols, 64)

        cfg = resolve_flash_config(
            64,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_rescale_chunk_cols": 16,
            },
        )
        self.assertEqual(cfg.rescale_chunk_cols, 16)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_RESCALE_CHUNK_COLS": "64",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(
                64,
                2,
                {
                    "cute_flash_rescale_chunk_cols": 32,
                },
            )
        self.assertEqual(cfg.rescale_chunk_cols, 32)

        cfg = resolve_flash_config(
            64,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_rescale_chunk_cols": 48,
            },
        )
        self.assertEqual(cfg.rescale_chunk_cols, 32)

        cfg = resolve_flash_config(
            64,
            3,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_rescale_chunk_cols": 64,
            },
        )
        self.assertEqual(cfg.topology, "ws_overlap")
        self.assertEqual(cfg.rescale_chunk_cols, 32)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "ws_overlap",
                "HELION_CUTE_FLASH_RESCALE_CHUNK_COLS": "bad",
            },
            clear=True,
        ):
            self.assertEqual(resolve_flash_config(64, 2).rescale_chunk_cols, 32)

    def test_flash_attention_fa4_register_budget_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(64, 2)
        self.assertEqual(cfg.softmax_regs, 200)
        self.assertEqual(cfg.corr_regs, 64)
        self.assertEqual(cfg.other_regs, 48)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_SOFTMAX_REGS": "192",
                "HELION_CUTE_FLASH_CORR_REGS": "80",
                "HELION_CUTE_FLASH_OTHER_REGS": "32",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(64, 2)
        self.assertEqual(cfg.softmax_regs, 192)
        self.assertEqual(cfg.corr_regs, 80)
        self.assertEqual(cfg.other_regs, 32)

        cfg = resolve_flash_config(
            64,
            2,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_softmax_regs": 184,
                "cute_flash_corr_regs": 88,
                "cute_flash_other_regs": 40,
            },
        )
        self.assertEqual(cfg.softmax_regs, 184)
        self.assertEqual(cfg.corr_regs, 88)
        self.assertEqual(cfg.other_regs, 40)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_SOFTMAX_REGS": "192",
                "HELION_CUTE_FLASH_CORR_REGS": "80",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(
                64,
                2,
                {
                    "cute_flash_softmax_regs": 196,
                    "cute_flash_corr_regs": 72,
                    "cute_flash_other_regs": 44,
                },
            )
        self.assertEqual(cfg.softmax_regs, 200)
        self.assertEqual(cfg.corr_regs, 72)
        self.assertEqual(cfg.other_regs, 48)

        cfg = resolve_flash_config(
            64,
            3,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_softmax_regs": "bad",
                "cute_flash_corr_regs": "bad",
            },
        )
        self.assertEqual(cfg.topology, "ws_overlap")
        self.assertEqual(cfg.softmax_regs, 200)
        self.assertEqual(cfg.corr_regs, 64)
        self.assertEqual(cfg.other_regs, 48)

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "ws_overlap",
                "HELION_CUTE_FLASH_SOFTMAX_REGS": "bad",
                "HELION_CUTE_FLASH_CORR_REGS": "bad",
            },
            clear=True,
        ):
            cfg = resolve_flash_config(64, 2)
        self.assertEqual(cfg.softmax_regs, 200)
        self.assertEqual(cfg.corr_regs, 64)
        self.assertEqual(cfg.other_regs, 48)

    def test_flash_attention_fa4_register_budget_validation(self) -> None:
        q, k, v = (
            torch.empty(1, 1, 8192, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bound = cute_dense_attention.bind((q, k, v))
        self.assertTrue(bound.config_spec.cute_flash_search_enabled)

        bad = helion.Config(
            block_sizes=[1, 128, 128],
            cute_flash_topology="fa4",
            cute_flash_softmax_regs=200,
            cute_flash_corr_regs=80,
            cute_flash_other_regs=40,
        )
        with self.assertRaisesRegex(InvalidConfig, "FA4 register budget exceeds 512"):
            bound.config_spec.normalize(bad)

        fixed = helion.Config.from_dict(bad.config)
        bound.config_spec.normalize(fixed, _fix_invalid=True)
        total = (
            2 * fixed.config["cute_flash_softmax_regs"]
            + fixed.config["cute_flash_corr_regs"]
            + fixed.config["cute_flash_other_regs"]
        )
        self.assertLessEqual(total, 512)

    def test_flash_attention_dense_hd64_corr_reg_seed_buckets(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg_32k = resolve_flash_config(64, 256)
            cfg_64k = resolve_flash_config(64, 512)
            cfg_64k_bf16 = resolve_flash_config(64, 512, dtype=torch.bfloat16)
            cfg_128k = resolve_flash_config(64, 1024)
            cfg_256k = resolve_flash_config(64, 2048)

        self.assertEqual(cfg_32k.corr_regs, 64)
        self.assertEqual(cfg_64k.corr_regs, 80)
        self.assertEqual(cfg_128k.corr_regs, 72)
        self.assertEqual(cfg_256k.corr_regs, 80)
        self.assertEqual(cfg_32k.e2e_schedule, "8/2")
        self.assertEqual(cfg_64k.e2e_schedule, "8/2")
        self.assertEqual(cfg_128k.e2e_schedule, "16/4")
        self.assertEqual(cfg_256k.e2e_schedule, "16/4")
        self.assertEqual(cfg_32k.first_load_order, 0)
        self.assertEqual(cfg_64k.first_load_order, 4)
        self.assertEqual(cfg_128k.first_load_order, 0)
        self.assertEqual(cfg_256k.first_load_order, 4)
        self.assertFalse(cfg_32k.precompute_qk_desc)
        self.assertTrue(cfg_64k.precompute_qk_desc)
        self.assertFalse(cfg_128k.precompute_qk_desc)
        self.assertFalse(cfg_256k.precompute_qk_desc)
        self.assertEqual(cfg_64k.rescale_threshold, 8.0)
        self.assertEqual(cfg_64k_bf16.rescale_threshold, 8.0)
        self.assertTrue(cfg_64k.split_p_arrive)
        self.assertTrue(cfg_128k.use_clc_scheduler)
        self.assertEqual(cfg_128k.clc_stages, 2)
        with patch.dict(
            os.environ,
            {"HELION_CUTE_FLASH_CLC_STAGES": "1"},
            clear=True,
        ):
            self.assertEqual(resolve_flash_config(64, 1024).clc_stages, 2)
        self.assertEqual(cfg_64k.corr_tile_size, 16)
        self.assertEqual(cfg_64k.role_map, "fa4")
        self.assertTrue(
            _cute_flash._flash_role_chain_default(
                hd=64,
                num_kv=512,
                is_causal=False,
                role_map=cfg_64k.role_map,
            )
        )
        self.assertFalse(
            _cute_flash._flash_role_chain_default(
                hd=64,
                num_kv=1024,
                is_causal=False,
                role_map=cfg_128k.role_map,
            )
        )
        self.assertFalse(
            _cute_flash._flash_role_chain_default(
                hd=64,
                num_kv=512,
                is_causal=True,
                role_map=cfg_64k.role_map,
            )
        )

    def test_flash_attention_dense_hd64_epi_tma_seed_buckets(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg_32k = resolve_flash_config(64, 256)
            cfg_64k = resolve_flash_config(64, 512)
            cfg_128k = resolve_flash_config(64, 1024)
            cfg_256k = resolve_flash_config(64, 2048)

        self.assertTrue(cfg_32k.epi_tma)
        self.assertFalse(cfg_32k.epi_stg)
        self.assertFalse(cfg_64k.epi_tma)
        self.assertTrue(cfg_64k.epi_stg)
        self.assertTrue(cfg_128k.epi_tma)
        self.assertFalse(cfg_128k.epi_stg)
        self.assertFalse(cfg_256k.epi_tma)
        self.assertTrue(cfg_256k.epi_stg)

    def test_flash_attention_dense_hd64_two_cta_eligibility(self) -> None:
        force_two_cta = {
            "cute_flash_topology": "fa4",
            "cute_flash_use_2cta": True,
            "cute_flash_precompute_qk_desc": True,
            "cute_flash_epi_tma": False,
            "cute_flash_epi_stg": True,
            "cute_flash_epi_stg_gmem": "pair",
            "cute_flash_p_store_rep": 32,
            "cute_flash_s_load_rep": 16,
            "cute_flash_softmax_disc": True,
            "cute_flash_split_p_arrive": False,
        }
        with patch.dict(os.environ, {}, clear=True):
            dense_64k = resolve_flash_config(64, 512, force_two_cta)
            dense_32k = resolve_flash_config(64, 256, force_two_cta)
            dense_128k = resolve_flash_config(64, 1024, force_two_cta)
            dense_unaligned = resolve_flash_config(64, 258, force_two_cta)
            dense_256k = resolve_flash_config(64, 2048, force_two_cta)
            dense_midrange = resolve_flash_config(64, 1536, force_two_cta)
            dense_long = resolve_flash_config(64, 2052, force_two_cta)
            dense_long_unaligned = resolve_flash_config(64, 2050, force_two_cta)
            dense_long_seed = next(
                seed
                for seed in _cute_flash.flash_attention_seed_configs(
                    64,
                    2052,
                    standard_dense_output=True,
                )
                if seed.config.get(_cute_flash.FLASH_EXP2_PACKET_KEY) == "deg1_16x8"
            )
            dense_long_degree1 = resolve_flash_config(
                64,
                2052,
                dense_long_seed.config,
                standard_dense_output=True,
            )
            dense_bf16 = resolve_flash_config(
                64, 512, force_two_cta, dtype=torch.bfloat16
            )
            causal = resolve_flash_config(64, 512, force_two_cta, is_causal=True)
            dense_hd128 = resolve_flash_config(128, 512, force_two_cta)

        self.assertTrue(dense_64k.use_2cta_instrs)
        self.assertEqual(dense_64k.pipeline_family, "fa4_2cta")
        self.assertTrue(dense_64k.precompute_qk_desc)
        self.assertTrue(dense_64k.epi_tma)
        self.assertFalse(dense_64k.epi_stg)
        self.assertEqual(dense_64k.epi_stg_gmem, "stage")
        self.assertEqual(dense_64k.p_store_repetition, 16)
        self.assertEqual(dense_64k.s_load_repetition, 32)
        self.assertFalse(dense_64k.softmax_disc)
        self.assertTrue(dense_64k.split_p_arrive)
        self.assertTrue(dense_32k.use_2cta_instrs)
        self.assertTrue(dense_128k.use_2cta_instrs)
        self.assertFalse(dense_unaligned.use_2cta_instrs)
        self.assertEqual(dense_unaligned.pipeline_family, "fa4")
        self.assertTrue(dense_256k.use_2cta_instrs)
        self.assertEqual(dense_256k.pipeline_family, "fa4_2cta")
        self.assertTrue(dense_midrange.use_2cta_instrs)
        self.assertEqual(dense_midrange.pipeline_family, "fa4_2cta")
        self.assertTrue(dense_long.use_2cta_instrs)
        self.assertEqual(dense_long.pipeline_family, "fa4_2cta")
        self.assertTrue(dense_long_degree1.use_2cta_instrs)
        self.assertEqual(dense_long_degree1.pipeline_family, "fa4_2cta")
        self.assertEqual(dense_long_degree1.exp2_packet, "deg1_16x8")
        self.assertTrue(
            _cute_flash._flash_disc_exp2_codegen_params(
                dense_long_degree1.exp2_packet,
                dense_long_degree1.e2e_freq,
                dense_long_degree1.e2e_res,
            ).degree1_unmasked
        )
        self.assertFalse(dense_long_unaligned.use_2cta_instrs)
        self.assertEqual(dense_long_unaligned.pipeline_family, "fa4")
        self.assertFalse(dense_bf16.use_2cta_instrs)
        self.assertEqual(dense_bf16.pipeline_family, "fa4")
        self.assertFalse(causal.use_2cta_instrs)
        self.assertEqual(causal.pipeline_family, "fa4")
        self.assertTrue(dense_hd128.use_2cta_instrs)
        self.assertFalse(dense_hd128.precompute_qk_desc)

        fragments = _cute_flash.flash_autotune_fragments(64, 512)
        family = fragments[_cute_flash.FLASH_PIPELINE_FAMILY_KEY]
        epi_tma = fragments[_cute_flash.FLASH_EPI_TMA_KEY]
        self.assertEqual(
            set(family.search_choices or ()),
            {"fa4", "ws_overlap", "fa4_2cta"},
        )
        self.assertEqual(epi_tma.search_choices, (False,))

        eligible_32k = _cute_flash.flash_autotune_fragments(64, 256)
        self.assertIn(
            "fa4_2cta",
            eligible_32k[_cute_flash.FLASH_PIPELINE_FAMILY_KEY].search_choices or (),
        )
        eligible_128k = _cute_flash.flash_autotune_fragments(64, 1024)
        self.assertIn(
            "fa4_2cta",
            eligible_128k[_cute_flash.FLASH_PIPELINE_FAMILY_KEY].search_choices or (),
        )
        ineligible_unaligned = _cute_flash.flash_autotune_fragments(64, 258)
        self.assertNotIn(
            "fa4_2cta",
            ineligible_unaligned[_cute_flash.FLASH_PIPELINE_FAMILY_KEY].search_choices
            or (),
        )
        eligible_256k = _cute_flash.flash_autotune_fragments(64, 2048)
        self.assertIn(
            "fa4_2cta",
            eligible_256k[_cute_flash.FLASH_PIPELINE_FAMILY_KEY].search_choices or (),
        )
        eligible_midrange = _cute_flash.flash_autotune_fragments(64, 1536)
        self.assertIn(
            "fa4_2cta",
            eligible_midrange[_cute_flash.FLASH_PIPELINE_FAMILY_KEY].search_choices
            or (),
        )
        eligible_long = _cute_flash.flash_autotune_fragments(64, 2052)
        self.assertIn(
            "fa4_2cta",
            eligible_long[_cute_flash.FLASH_PIPELINE_FAMILY_KEY].search_choices or (),
        )
        ineligible_long_unaligned = _cute_flash.flash_autotune_fragments(64, 2050)
        self.assertNotIn(
            "fa4_2cta",
            ineligible_long_unaligned[
                _cute_flash.FLASH_PIPELINE_FAMILY_KEY
            ].search_choices
            or (),
        )

    def test_flash_attention_pipeline_family_resolution(self) -> None:
        expected_flags = {
            "ws_overlap": (
                "ws_overlap",
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ),
            "fa4": ("fa4", False, False, False, False, False, False, False),
            "fa4_deep_1cta": ("fa4", True, False, False, False, False, False, False),
            "fa4_2cta_causal": ("fa4", False, True, True, False, False, False, False),
            "fa4_tma_4d": ("fa4", False, False, False, False, False, False, True),
            "fa4_local_tma": ("fa4", False, False, False, False, False, True, False),
            "fa4_local_tma_4d": ("fa4", False, False, False, False, False, True, True),
            "fa4_cga2_local": ("fa4", False, False, False, True, False, False, False),
            "fa4_cga2_local_tma_4d": (
                "fa4",
                False,
                False,
                False,
                True,
                False,
                False,
                True,
            ),
            "fa4_2cta": ("fa4", False, False, True, False, False, False, False),
            "fa4_2cta_tma_4d": ("fa4", False, False, True, False, False, False, True),
            "fa4_clc": ("fa4", False, False, False, False, True, False, False),
            "fa4_clc_tma_4d": ("fa4", False, False, False, False, True, False, True),
            "fa4_clc_local_tma": ("fa4", False, False, False, False, True, True, False),
            "fa4_clc_local_tma_4d": (
                "fa4",
                False,
                False,
                False,
                False,
                True,
                True,
                True,
            ),
        }
        self.assertEqual(set(expected_flags), set(_cute_flash.FLASH_PIPELINE_FAMILIES))

        with patch.dict(os.environ, {}, clear=True):
            for family_name, expected in expected_flags.items():
                with self.subTest(family=family_name):
                    cfg = resolve_flash_config(
                        64,
                        512,
                        {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: family_name},
                        dtype=torch.float16,
                        is_causal="causal" in family_name,
                    )
                    self.assertEqual(cfg.pipeline_family, family_name)
                    self.assertEqual(
                        (
                            cfg.topology,
                            cfg.separate_kv_rings,
                            cfg.causal_two_cta,
                            cfg.use_2cta_instrs,
                            cfg.use_cga2_local_cta,
                            cfg.use_clc_scheduler,
                            cfg.local_tma_partition,
                            cfg.tensor_4d_tma,
                        ),
                        expected,
                    )

            legacy_by_family = {
                "ws_overlap": {_cute_flash.FLASH_TOPOLOGY_KEY: "ws_overlap"},
                "fa4_2cta_tma_4d": {
                    _cute_flash.FLASH_TOPOLOGY_KEY: "fa4",
                    _cute_flash.FLASH_USE_2CTA_KEY: True,
                    _cute_flash.FLASH_TENSOR_4D_TMA_KEY: True,
                },
                "fa4_cga2_local_tma_4d": {
                    _cute_flash.FLASH_TOPOLOGY_KEY: "fa4",
                    _cute_flash.FLASH_CGA2_LOCAL_KEY: True,
                    _cute_flash.FLASH_TENSOR_4D_TMA_KEY: True,
                },
                "fa4_clc_local_tma_4d": {
                    _cute_flash.FLASH_TOPOLOGY_KEY: "fa4",
                    _cute_flash.FLASH_CLC_KEY: True,
                    _cute_flash.FLASH_LOCAL_TMA_PARTITION_KEY: True,
                    _cute_flash.FLASH_TENSOR_4D_TMA_KEY: True,
                },
            }
            for family_name, legacy_config in legacy_by_family.items():
                with self.subTest(legacy_family=family_name):
                    family_cfg = resolve_flash_config(
                        64,
                        512,
                        {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: family_name},
                        dtype=torch.float16,
                    )
                    legacy_cfg = resolve_flash_config(
                        64,
                        512,
                        legacy_config,
                        dtype=torch.float16,
                    )
                    self.assertEqual(legacy_cfg, family_cfg)

            family_wins = resolve_flash_config(
                64,
                512,
                {
                    _cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
                    _cute_flash.FLASH_TOPOLOGY_KEY: "ws_overlap",
                    _cute_flash.FLASH_USE_2CTA_KEY: True,
                    _cute_flash.FLASH_CGA2_LOCAL_KEY: True,
                    _cute_flash.FLASH_CLC_KEY: True,
                    _cute_flash.FLASH_LOCAL_TMA_PARTITION_KEY: True,
                    _cute_flash.FLASH_TENSOR_4D_TMA_KEY: True,
                },
                dtype=torch.float16,
            )
            self.assertEqual(family_wins.pipeline_family, "fa4")

            odd_kv = resolve_flash_config(
                64,
                511,
                {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: ("fa4_clc_local_tma_4d")},
                dtype=torch.float16,
            )
            self.assertEqual(odd_kv.pipeline_family, "ws_overlap")
            bf16_2cta = resolve_flash_config(
                64,
                512,
                {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta_tma_4d"},
                dtype=torch.bfloat16,
            )
            self.assertEqual(bf16_2cta.pipeline_family, "fa4")
            causal_clc = resolve_flash_config(
                64,
                512,
                {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: ("fa4_clc_local_tma_4d")},
                dtype=torch.float16,
                is_causal=True,
            )
            self.assertEqual(causal_clc.pipeline_family, "fa4")

            legacy_clc = resolve_flash_config(
                64,
                1024,
                {
                    _cute_flash.FLASH_TOPOLOGY_KEY: "fa4",
                    _cute_flash.FLASH_CLC_KEY: True,
                },
            )
            family_clc = resolve_flash_config(
                64,
                1024,
                {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_clc"},
            )
            self.assertEqual(legacy_clc, family_clc)
            self.assertEqual(family_clc.clc_heads_per_batch, 32)
            self.assertEqual(family_clc.clc_stages, 2)
            clc_fragments = _cute_flash.flash_autotune_fragments(
                64,
                1024,
                pipeline_family_override="fa4_clc",
            )
            self.assertEqual(
                clc_fragments[_cute_flash.FLASH_CLC_HEADS_PER_BATCH_KEY].default(),
                32,
            )

            base = resolve_flash_config(
                64,
                512,
                {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"},
            )
            with_legacy_values = resolve_flash_config(
                64,
                512,
                {
                    _cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
                    _cute_flash.FLASH_Q_TILE_COUNT_KEY: 17,
                    _cute_flash.FLASH_MMA_INTERLEAVE_KEY: 9,
                },
            )
            self.assertEqual(with_legacy_values, base)
            field_names = {field.name for field in dataclasses.fields(base)}
            self.assertIn("q_tile_count", field_names)
            self.assertIn("mma_interleave", field_names)
            self.assertEqual(base.q_tile_count, 2)
            self.assertTrue(base.mma_interleave)

    def test_flash_attention_sync_and_softmax_controls_normalize(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            fa4 = resolve_flash_config(
                64,
                512,
                {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"},
            )
            self.assertEqual(fa4.q_tile_count, 2)
            self.assertTrue(fa4.mma_interleave)
            self.assertEqual(fa4.wait_hint, 10_000_000)
            self.assertEqual(fa4.exp2_packet, "1x1")
            self.assertEqual(fa4.stat_transport, "single")

            ws = resolve_flash_config(
                64,
                512,
                {
                    _cute_flash.FLASH_PIPELINE_FAMILY_KEY: "ws_overlap",
                    _cute_flash.FLASH_Q_TILE_COUNT_KEY: 17,
                    _cute_flash.FLASH_MMA_INTERLEAVE_KEY: True,
                    _cute_flash.FLASH_WAIT_HINT_KEY: 0,
                    _cute_flash.FLASH_EXP2_PACKET_KEY: "8x2",
                    _cute_flash.FLASH_STAT_TRANSPORT_KEY: "single",
                },
            )
            self.assertEqual(ws.q_tile_count, 1)
            self.assertFalse(ws.mma_interleave)
            self.assertEqual(ws.wait_hint, 10_000_000)
            self.assertEqual(ws.exp2_packet, "1x1")
            self.assertEqual(ws.stat_transport, "ring2")

            two_cta = resolve_flash_config(
                64,
                512,
                {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"},
            )
            self.assertEqual(two_cta.exp2_packet, "8x2")
            hd128_two_cta = resolve_flash_config(
                128,
                512,
                {_cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4_2cta"},
            )
            self.assertEqual(hd128_two_cta.exp2_packet, "1x1")

            tuned = resolve_flash_config(
                64,
                512,
                {
                    _cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
                    _cute_flash.FLASH_MMA_INTERLEAVE_KEY: False,
                    _cute_flash.FLASH_WAIT_HINT_KEY: 0,
                    _cute_flash.FLASH_EXP2_PACKET_KEY: "8x2",
                    _cute_flash.FLASH_STAT_TRANSPORT_KEY: "ring2",
                },
            )
            self.assertFalse(tuned.mma_interleave)
            self.assertEqual(tuned.wait_hint, 0)
            self.assertEqual(tuned.exp2_packet, "8x2")
            self.assertEqual(tuned.stat_transport, "ring2")

            causal = resolve_flash_config(
                64,
                512,
                {
                    _cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
                    _cute_flash.FLASH_STAT_TRANSPORT_KEY: "single",
                },
                is_causal=True,
            )
            self.assertEqual(causal.stat_transport, "ring2")

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_Q_TILE_COUNT": "legacy-nonnumeric",
                "HELION_CUTE_FLASH_MMA_PTX": "0",
            },
            clear=True,
        ):
            inactive = resolve_flash_config(
                64,
                512,
                {
                    _cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
                    _cute_flash.FLASH_Q_TILE_COUNT_KEY: "also-ignored",
                    _cute_flash.FLASH_MMA_INTERLEAVE_KEY: False,
                },
            )
        self.assertEqual(inactive.q_tile_count, 2)
        self.assertTrue(inactive.mma_interleave)

    def test_flash_attention_fa4_persistent_config_overrides_env(self) -> None:
        with patch.dict(os.environ, {"HELION_CUTE_FLASH_PERSISTENT": "1"}, clear=True):
            cfg = resolve_flash_config(
                64,
                512,
                {
                    "cute_flash_topology": "fa4",
                    "cute_flash_persistent": False,
                },
            )
        self.assertFalse(cfg.persistent)

    def test_flash_config_from_config_forwards_shape_context(self) -> None:
        config = {"cute_flash_topology": "fa4"}

        with patch.dict(os.environ, {}, clear=True):
            causal_cfg = _cute_flash.flash_config_from_config(
                config,
                64,
                64,
                is_causal=True,
            )
            self.assertFalse(causal_cfg.persistent)
            self.assertEqual(causal_cfg.causal_lpt_swizzle, 8)

            dense_cfg = _cute_flash.flash_config_from_config(
                config,
                64,
                64,
                is_causal=False,
            )
            self.assertTrue(dense_cfg.persistent)
            self.assertEqual(dense_cfg.causal_lpt_swizzle, 0)

            fp32_cfg = _cute_flash.flash_config_from_config(
                config,
                64,
                64,
                dtype=torch.float32,
            )
            self.assertEqual(fp32_cfg.rescale_threshold, 0.0)

    def test_flash_attention_sparse_prefers_packed_reduce(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            sparse_default = resolve_flash_config(
                64,
                2,
                prefer_packed_reduce=True,
            )
        self.assertTrue(sparse_default.packed_reduce)

        with patch.dict(
            os.environ,
            {"HELION_CUTE_FLASH_PACKED_REDUCE": "0"},
            clear=True,
        ):
            sparse_env_override = resolve_flash_config(
                64,
                2,
                prefer_packed_reduce=True,
            )
        self.assertTrue(sparse_env_override.packed_reduce)

        sparse_config_override = resolve_flash_config(
            64,
            2,
            {"cute_flash_packed_reduce": False},
            prefer_packed_reduce=True,
        )
        self.assertTrue(sparse_config_override.packed_reduce)

    def test_flash_attention_small_biased_config_overrides(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(resolve_flash_config(64, 1).small_biased)

        cfg = resolve_flash_config(
            64,
            1,
            {_cute_flash.FLASH_SMALL_BIASED_KEY: False},
        )
        self.assertFalse(cfg.small_biased)

    def test_flash_attention_single_kv_defaults_to_one_kv_stage(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = resolve_flash_config(64, 1, {"cute_flash_topology": "ws_overlap"})
        self.assertEqual(cfg.s_stage, 1)
        self.assertEqual(cfg.kv_stage, 1)

    def test_flash_attention_fa4_causal_lpt_swizzle_overrides(self) -> None:
        self.assertEqual(resolve_flash_config(64, 64).causal_lpt_swizzle, 0)
        short_causal = resolve_flash_config(64, 2, is_causal=True)
        self.assertEqual(short_causal.e2e_offset, 2)
        self.assertTrue(short_causal.packed_reduce)
        self.assertEqual(short_causal.causal_lpt_swizzle, 0)
        self.assertEqual(
            resolve_flash_config(64, 64, is_causal=True).causal_lpt_swizzle,
            8,
        )
        self.assertEqual(
            resolve_flash_config(64, 512, is_causal=True).causal_lpt_swizzle,
            1,
        )

        with patch.dict(
            os.environ,
            {
                "HELION_CUTE_FLASH_TOPOLOGY": "fa4",
                "HELION_CUTE_FLASH_CAUSAL_LPT_SWIZZLE": "8",
            },
            clear=True,
        ):
            self.assertEqual(
                resolve_flash_config(64, 64, is_causal=True).causal_lpt_swizzle,
                8,
            )
            self.assertEqual(resolve_flash_config(64, 64).causal_lpt_swizzle, 0)

        cfg = resolve_flash_config(
            64,
            64,
            {
                "cute_flash_topology": "fa4",
                "cute_flash_causal_lpt_swizzle": 16,
            },
            is_causal=True,
        )
        self.assertEqual(cfg.causal_lpt_swizzle, 16)

    def test_flash_attention_binds_qkv_by_graph_operands(self) -> None:
        q, k, v = (
            torch.randn(2, 8, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        code, out = code_and_output(
            cute_dense_attention_v_loaded_before_k,
            (q, k, v),
            block_sizes=[1, 128, 128],
        )
        self.assertTrue(_flash_fired(code))
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_declines_noncanonical_score_dataflow(self) -> None:
        q, k, v = (
            torch.randn(2, 8, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        for kernel in (
            cute_dense_attention_unscaled_qk,
            cute_dense_attention_fp16_qk,
            cute_dense_attention_post_center_scale,
            cute_dense_attention_shifted_q,
            cute_dense_attention_shifted_v,
            cute_dense_attention_shifted_k,
            cute_dense_attention_shifted_q_and_out,
        ):
            with self.subTest(kernel=kernel.fn.__name__):
                bound = kernel.bind((q, k, v))
                code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
                self.assertFalse(_flash_fired(code))

    def test_flash_attention_declines_noncanonical_online_recurrence(self) -> None:
        q, k, v = (
            torch.randn(2, 8, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        for kernel in (
            cute_dense_attention_no_final_divide,
            cute_dense_attention_no_alpha_rescale,
            cute_dense_attention_post_l_update,
            cute_dense_attention_post_acc_update,
        ):
            with self.subTest(kernel=kernel.fn.__name__):
                bound = kernel.bind((q, k, v))
                code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
                self.assertFalse(_flash_fired(code))

    def test_flash_attention_declines_empty_batch(self) -> None:
        q, k, v = (
            torch.empty(0, 8, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bound = cute_dense_attention.bind((q, k, v))
        code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
        self.assertFalse(_flash_fired(code))

    def test_flash_attention_declines_unrelated_fp32_tile_output(self) -> None:
        q, k, v = (
            torch.randn(2, 8, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bound = cute_dense_attention_with_aux.bind((q, k, v))
        code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
        self.assertFalse(_flash_fired(code))
        _code, (out, aux) = code_and_output(
            cute_dense_attention_with_aux,
            (q, k, v),
            block_sizes=[1, 128, 128],
        )
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(aux, torch.zeros_like(aux))

    def test_flash_attention_declines_lse_plus_unrelated_fp32_output(self) -> None:
        q, k, v = (
            torch.randn(2, 8, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bound = cute_dense_attention_with_lse_and_aux.bind((q, k, v))
        code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
        self.assertFalse(_flash_fired(code))

    def test_flash_attention_declines_log_aux_output(self) -> None:
        q, k, v = (
            torch.randn(2, 8, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bound = cute_dense_attention_with_log_aux.bind((q, k, v))
        code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
        self.assertFalse(_flash_fired(code))

    def test_flash_attention_declines_3d_aux_output(self) -> None:
        q, k, v = (
            torch.randn(2, 8, 256, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bound = cute_dense_attention_with_3d_aux.bind((q, k, v))
        code = bound.to_triton_code(helion.Config(block_sizes=[1, 128, 128]))
        self.assertFalse(_flash_fired(code))
        _code, (out, aux) = code_and_output(
            cute_dense_attention_with_3d_aux,
            (q, k, v),
            block_sizes=[1, 128, 128],
        )
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(aux, expected, atol=1e-2, rtol=1e-2)

    def test_flash_attention_declines_unsafe_configs(self) -> None:
        """The detector must NOT fire flash for configs the dense tensor-core
        kernel cannot honor -- otherwise a default-on gate silently miscomputes.
        Each case below produced a WRONG result when it (previously) fired."""

        def fired(kernel, args, **cfg):
            code, _ = code_and_output(kernel, args, **cfg)
            return _flash_fired(code)

        f16 = {"dtype": torch.float16, "device": DEVICE}

        def sq(seq, hd=64):
            return tuple(torch.randn(2, 8, seq, hd, **f16) for _ in range(3))

        # fp32 operands (kernel hardcodes fp16).
        fp32 = tuple(
            torch.randn(2, 8, 256, 64, dtype=torch.float32, device=DEVICE)
            for _ in range(3)
        )
        self.assertFalse(
            fired(cute_dense_attention, fp32, block_sizes=[1, 128, 128]),
            "fp32 must not fire flash",
        )
        # Non-square (cross-attention): num_kv would use the query length.
        nonsq = (
            torch.randn(2, 8, 256, 64, **f16),
            torch.randn(2, 8, 128, 64, **f16),
            torch.randn(2, 8, 128, 64, **f16),
        )
        self.assertFalse(
            fired(cute_dense_attention, nonsq, block_sizes=[1, 128, 128]),
            "non-square must not fire flash",
        )
        # Non-128 tiles (outside the validated 128x128 envelope).
        self.assertFalse(
            fired(cute_dense_attention, sq(256), block_sizes=[1, 64, 64]),
            "non-128 tiles must not fire flash",
        )
        self.assertFalse(
            fired(
                cute_dense_attention,
                sq(256),
                block_sizes=[1, 128, 128],
                loop_orders=[[1, 0]],
            ),
            "non-default loop order must not fire flash",
        )
        self.assertFalse(
            fired(
                cute_dense_attention,
                sq(256),
                block_sizes=[1, 128, 128],
                cute_vector_widths=[1, 2],
            ),
            "non-1 vector widths must not fire flash",
        )
        # Persistent / interleaved pid remaps the program grid.
        self.assertFalse(
            fired(
                cute_dense_attention,
                sq(256),
                block_sizes=[1, 128, 128],
                pid_type="persistent_interleaved",
            ),
            "persistent pid must not fire flash",
        )
        # L2 grouping reorders program ids (flat pid, so this exercises the
        # l2_grouping guard specifically, not the pid guard).
        self.assertFalse(
            fired(
                cute_dense_attention,
                sq(256),
                block_sizes=[1, 128, 128],
                l2_grouping=2,
            ),
            "l2_grouping must not fire flash",
        )

    def test_pointwise_add_three_inputs(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(cute_add3, args)
        x, y, z = args
        torch.testing.assert_close(out, x + y + z)

    def test_pointwise_mul(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(cute_mul, args)
        x, y = args
        torch.testing.assert_close(out, x * y)

    def test_pointwise_relu(self) -> None:
        args = (torch.randn(65, 23, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(cute_relu, args)
        (x,) = args
        torch.testing.assert_close(out, torch.relu(x))

    def test_pointwise_sin(self) -> None:
        args = (torch.randn(65, 23, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(cute_sin, args)
        (x,) = args
        torch.testing.assert_close(out, torch.sin(x))

    def test_pointwise_sigmoid(self) -> None:
        args = (torch.randn(65, 23, device=DEVICE, dtype=HALF_DTYPE),)
        code, out = code_and_output(cute_sigmoid, args)
        (x,) = args
        torch.testing.assert_close(out, torch.sigmoid(x), rtol=1e-3, atol=1e-3)

    def test_pointwise_chain(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(cute_pointwise_chain, args)
        x, y = args
        expected = torch.sigmoid(torch.sin(torch.relu(x * y)))
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_pointwise_minimum_maximum_cutlass_45_fallback(self) -> None:
        cases = (
            (
                torch.tensor(
                    [[float("nan"), -0.0, 1.0]], device=DEVICE, dtype=torch.float32
                ),
                torch.tensor(
                    [[2.0, 0.0, float("nan")]], device=DEVICE, dtype=torch.float32
                ),
            ),
            (
                torch.tensor(
                    [[2**25 + 1, -(2**25 + 1)]], device=DEVICE, dtype=torch.int32
                ),
                torch.tensor([[2**25, -(2**25)]], device=DEVICE, dtype=torch.int32),
            ),
        )
        for args in cases:
            with self.subTest(dtype=str(args[0].dtype)):
                with patch(
                    "helion._compiler.inductor_lowering.cute_math_min_max_available",
                    return_value=False,
                ):
                    code, out = code_and_output(cute_minimum_maximum, args)
                x, y = args
                torch.testing.assert_close(
                    out, torch.minimum(torch.maximum(x, y), x), equal_nan=True
                )
                if x.is_floating_point():
                    torch.testing.assert_close(
                        torch.signbit(out),
                        torch.signbit(torch.minimum(torch.maximum(x, y), x)),
                    )
                self.assertNotIn("cute.arch.fmax", code)
                self.assertNotIn("cute.arch.fmin", code)
                self.assertNotIn("cute.math.max", code)
                self.assertNotIn("cute.math.min", code)

    def test_rms_norm_uses_native_rsqrt(self) -> None:
        x = torch.randn(8, 32, device=DEVICE, dtype=torch.float32)
        weight = torch.randn(32, device=DEVICE, dtype=torch.float32)
        eps = 1e-5
        code, out = code_and_output(cute_rms_norm, (x, weight, eps), block_size=4)
        x_sq = x * x
        inv_rms = torch.rsqrt(x_sq.mean(dim=-1) + eps)
        expected = x * inv_rms[:, None] * weight
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        self.assertIn("cute.math.rsqrt", code)
        self.assertNotIn("cute.math.sqrt", code)
        self.assertNotRegex(code, r"1\.0\s*/\s*v_\d+")

    def test_scalar_args_int_and_float(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            3,
            1.25,
        )
        code, out = code_and_output(cute_affine_scalar_args, args)
        x, scale, bias = args
        torch.testing.assert_close(out, x * scale + bias, rtol=1e-5, atol=1e-5)

    def test_kwargs_dispatch(self) -> None:
        x = torch.randn(65, 23, device=DEVICE, dtype=torch.float32)
        out = cute_affine_scalar_args(bias=0.5, scale=2, x=x)
        torch.testing.assert_close(out, x * 2 + 0.5, rtol=1e-5, atol=1e-5)

        normalized_args = cute_affine_scalar_args.normalize_args(
            bias=0.5,
            scale=2,
            x=x,
        )
        code, out_from_positional = code_and_output(
            cute_affine_scalar_args,
            normalized_args,
        )
        torch.testing.assert_close(out_from_positional, out)

    def test_oversized_nd_block_auto_threads_into_lane_loops(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(cute_add, args, block_sizes=[64, 32])
        x, y = args
        torch.testing.assert_close(out, x + y)
        self.assertIn("for lane_", code)

    def test_nd_num_threads(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(
            cute_add,
            args,
            block_sizes=[64, 32],
            num_threads=[32, 16],
        )
        x, y = args
        torch.testing.assert_close(out, x + y)

    def test_nd_num_threads_not_divisor_raises(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            "block size must be divisible by num_threads",
        ):
            # block_size=32 is not divisible by num_threads=64
            code_and_output(
                cute_add,
                args,
                block_sizes=[32, 32],
                num_threads=[64, 16],
            )

    def test_flattened_num_threads(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(
            cute_add,
            args,
            block_sizes=[64, 32],
            flatten_loop=True,
            num_threads=[32, 16],
        )
        x, y = args
        torch.testing.assert_close(out, x + y)
        self.assertIn("block=(512, 1, 1)", code)

    def test_device_loop_num_threads(self) -> None:
        args = (torch.randn(65, 23, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(
            cute_device_loop_add_one,
            args,
            block_sizes=[64, 32],
            num_threads=[32, 16],
        )
        (x,) = args
        torch.testing.assert_close(out, x + 1)
        self.assertIn("for lane_", code)

    def test_flattened_device_loop_num_threads(self) -> None:
        args = (torch.randn(8, 65, 23, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(
            cute_flattened_device_loop_add_one,
            args,
            block_sizes=[1, 64, 32],
            flatten_loops=[True],
            num_threads=[1, 32, 16],
        )
        (x,) = args
        torch.testing.assert_close(out, x + 1)
        self.assertIn("for lane_", code)

    def test_oversized_flattened_block_caps_threads(self) -> None:
        """When num_threads is auto and block_size > 1024, the CuTe backend
        falls back to a 1024-thread lane loop rather than raising."""

        @helion.kernel(backend="cute", autotune_effort="none")
        def cute_flattened_identity(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                out[tile] = x[tile]
            return out

        args = (torch.randn(2048, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(cute_flattened_identity, args, block_size=2048)
        torch.testing.assert_close(out, args[0])
        # block_size 2048 with auto threads now lowers to a 1024-thread lane
        # loop (each thread owns two elements).
        self.assertIn("for lane_", code)

    def test_oversized_flattened_block_raises_when_threads_explicit(self) -> None:
        """When num_threads is explicit and exceeds the 1024-per-CTA cap,
        the backend still raises rather than silently downsizing."""

        @helion.kernel(backend="cute", autotune_effort="none")
        def cute_flattened_identity(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                out[tile] = x[tile]
            return out

        args = (torch.randn(2048, device=DEVICE, dtype=torch.float32),)
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported, "thread block too large for cute kernel"
        ):
            code_and_output(
                cute_flattened_identity, args, block_size=2048, num_threads=[2048]
            )

    def test_reduction_num_threads(self) -> None:
        args = (torch.randn(129, 130, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(
            cute_row_sum,
            args,
            block_sizes=[64],
            num_threads=[32],
        )
        (x,) = args
        torch.testing.assert_close(out, x.sum(-1), rtol=1e-4, atol=1e-4)
        self.assertIn("for lane_", code)

    def test_looped_reduction_num_threads(self) -> None:
        args = (torch.randn(129, 130, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(
            cute_row_sum,
            args,
            block_sizes=[64],
            reduction_loop=16,
            num_threads=[32],
        )
        (x,) = args
        torch.testing.assert_close(out, x.sum(-1), rtol=1e-4, atol=1e-4)
        self.assertIn("for lane_", code)

    def test_looped_reduction_uses_per_thread_lanes(self) -> None:
        args = (torch.randn(16, 4096, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(
            cute_row_sum,
            args,
            block_sizes=[1],
            reduction_loop=2048,
            num_warps=4,
        )
        (x,) = args
        torch.testing.assert_close(out, x.sum(-1), rtol=1e-4, atol=1e-4)
        self.assertIn("_REDUCTION_BLOCK_1 = 2048", code)
        self.assertIn("for reduction_lane_1 in range(2)", code)
        self.assertIn("_cute_grouped_reduce_shared_two_stage", code)
        self.assertIn("group_span=1024", code)
        self.assertIn("block=(1024, 1, 1)", code)

    def test_cute_vector_widths_partitions_lane_extent(self) -> None:
        """cute_vector_widths=[V] partitions the lane extent into
        outer x inner=V, so the consume sweep walks each V-chunk via a
        constexpr V-loop and the per-thread base stride becomes V."""
        args = (torch.randn(2, 16384, device=DEVICE, dtype=torch.float32) + 2.0,)
        code_v1, _ = code_and_output(
            cute_normalize_by_sum,
            args,
            block_sizes=[1],
            reduction_loop=8192,
        )
        code_v4, _ = code_and_output(
            cute_normalize_by_sum,
            args,
            block_sizes=[1],
            reduction_loop=8192,
            cute_vector_widths=[4],
        )
        # V=1 baseline: no constexpr V-loop, no per-thread V-stride.
        self.assertNotIn("cutlass.range_constexpr(4)", code_v1)
        self.assertNotIn("thread_idx()[0]) * 4", code_v1)
        # V=4: the consume sweep emits a constexpr V-loop and the per-thread
        # base index is offset by ``thread_idx * V``.
        self.assertIn("cutlass.range_constexpr(4)", code_v4)
        self.assertIn("thread_idx()[0]) * 4", code_v4)

    def test_bf16_unroll_mode_emits_uint16_vec_load_and_bitcast(self) -> None:
        """For a bf16 reduction with an explicit fp32 cast, the 'unroll' vec
        mode loads each V-chunk as a Uint16 vector and bitcasts each lane
        back to bf16 via cutlass.Uint16(...).bitcast(cutlass.BFloat16)."""
        args = (torch.randn(2, 16384, device=DEVICE, dtype=torch.bfloat16) + 2.0,)
        code, out = code_and_output(
            cute_normalize_by_sum_fp32_cast,
            args,
            block_sizes=[1],
            reduction_loop=8192,
            cute_vector_widths=[4],
        )
        (x,) = args
        expected = (x.float() / x.float().sum(-1, keepdim=True)).to(x.dtype)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
        self.assertIn("ir.VectorType.get([4], cutlass.Uint16.mlir_type)", code)
        self.assertIn(".bitcast(cutlass.BFloat16)", code)

    def test_two_pass_load_fusion_shape_b_wide_chunk(self) -> None:
        """Shape B: V=1 wide-chunk reduction emits a lane loop inside the
        outer offset loop, and the fuser caches loaded x values across the
        reduce and consume sweeps."""
        args = (torch.randn(2, 16384, device=DEVICE, dtype=torch.float32) + 2.0,)
        code, out = code_and_output(
            cute_normalize_by_sum,
            args,
            block_sizes=[1],
            reduction_loop=8192,
        )
        (x,) = args
        expected = x / x.sum(-1, keepdim=True)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        # The fuser allocates a fragment and rewrites the consume sweep's
        # load to read from the cache.
        self.assertIn("cute.make_rmem_tensor", code)
        self.assertIn("_fuse_cache_0", code)

    def test_two_pass_load_fusion_shape_c_vec_unroll(self) -> None:
        """Shape C: V>1 unroll mode hoists a Uint16 vec load above the
        constexpr V-loop; the fuser recognises the vec hoist and caches
        cache_size * V scalar slots across the two sweeps."""
        args = (torch.randn(2, 16384, device=DEVICE, dtype=torch.bfloat16) + 2.0,)
        code, out = code_and_output(
            cute_normalize_by_sum_fp32_cast,
            args,
            block_sizes=[1],
            reduction_loop=8192,
            cute_vector_widths=[4],
        )
        (x,) = args
        expected = (x.float() / x.float().sum(-1, keepdim=True)).to(x.dtype)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
        self.assertIn("cute.make_rmem_tensor", code)
        self.assertIn("_fuse_cache_0", code)

    def test_strided_threaded_block_reduction(self) -> None:
        args = (torch.randn(4, 16, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(cute_row_centered, args, block_sizes=[2, 8, 8])
        (x,) = args
        expected = x - x.mean(dim=1, keepdim=True)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        self.assertIn("block=(2, 8, 1)", code)

    def test_strided_threaded_block_reduction_non_sum(self) -> None:
        args = (torch.rand(4, 16, device=DEVICE, dtype=torch.float32) + 0.5,)
        (x,) = args
        cases = [
            (cute_row_max, torch.amax(x.to(torch.float32), dim=1)),
            (cute_row_min, torch.amin(x.to(torch.float32), dim=1)),
            (cute_row_prod, torch.prod(x.to(torch.float32), dim=1)),
        ]
        for kernel, expected in cases:
            with self.subTest(kernel=kernel.__name__):
                _code, out = code_and_output(kernel, args, block_sizes=[2, 8])
                torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_direct_shared_tree_reduce_helpers_non_sum(self) -> None:
        x = torch.rand(3, 16, device=DEVICE, dtype=torch.float32) + 0.5
        cases = [
            (
                cute_shared_tree_reduce_max,
                torch.amax(x.to(torch.float32), dim=1),
            ),
            (
                cute_shared_tree_reduce_min,
                torch.amin(x.to(torch.float32), dim=1),
            ),
            (
                cute_shared_tree_reduce_prod,
                torch.prod(x.to(torch.float32), dim=1),
            ),
        ]
        for kernel, expected in cases:
            with self.subTest(kernel=kernel.__name__):
                out = torch.empty_like(expected)
                default_cute_launcher(kernel, (1,), x, out, block=(3, 16, 1))
                torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_permute_transposes_tile_values(self) -> None:
        """Permute should shuffle scalar values between threads."""

        x = torch.arange(16, device=DEVICE, dtype=torch.float32).reshape(4, 4)
        _, out = code_and_output(cute_permute_transpose, (x,), block_sizes=[4, 4])
        torch.testing.assert_close(out, x.transpose(0, 1))

    def test_permute_transposes_tile_values_with_lane_loops(self) -> None:
        x = torch.arange(16, device=DEVICE, dtype=torch.float32).reshape(4, 4)
        code, out = code_and_output(
            cute_permute_transpose,
            (x,),
            block_sizes=[4, 4],
            num_threads=[2, 2],
        )
        torch.testing.assert_close(out, x.transpose(0, 1))
        self.assertIn("for lane_", code)

    def test_permute_store_then_read_preserves_program_order_with_lane_loops(
        self,
    ) -> None:
        x = torch.arange(16, device=DEVICE, dtype=torch.float32).reshape(4, 4)
        code, out = code_and_output(
            cute_permute_store_then_read,
            (x,),
            block_sizes=[4, 4],
            num_threads=[2, 2],
        )
        torch.testing.assert_close(out, x.transpose(0, 1) + 1)
        self.assertIn("x[indices_1, indices_0]", code)

    def test_matmul_mma(self) -> None:
        """Test MMA tensor core matmul with float16 inputs."""
        args = (
            torch.randn(16, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 8, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(cute_matmul_mma, args, block_sizes=[16, 8, 16])
        torch.testing.assert_close(out, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertIn("cute.gemm", code)
        self.assertIn("cute.nvgpu.warp.MmaF16BF16Op", code)
        self.assertNotIn("cute.arch.warp_reduction_sum", code)

    def test_matmul_mma_unit_m_dimension(self) -> None:
        args = (
            torch.randn(1, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 8, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(
            cute_matmul_mma,
            args,
            block_sizes=[1, 8, 16],
            num_threads=[1, 8, 1],
        )
        torch.testing.assert_close(out, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertNotIn("cute.arch.warp_reduction_sum", code)
        self.assertNotIn("cute.gemm", code)

    def test_matmul_mma_epilogue(self) -> None:
        """Test MMA matmul with epilogue (bias add + dtype cast)."""
        args = (
            torch.randn(16, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 8, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(8, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(
            cute_matmul_mma_epilogue, args, block_sizes=[16, 8, 16]
        )
        x, y, bias = args
        expected = (x.float() @ y.float() + bias.float()).to(HALF_DTYPE)
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)
        self.assertIn("cute.gemm", code)
        self.assertIn("cute.nvgpu.warp.MmaF16BF16Op", code)
        self.assertNotIn("cute.arch.warp_reduction_sum", code)

    def test_matmul_dot_mma(self) -> None:
        """Test hl.dot MMA path with float16 inputs."""
        args = (
            torch.randn(16, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 8, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(cute_matmul_dot_mma, args, block_sizes=[16, 8, 16])
        torch.testing.assert_close(out, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertIn("cute.gemm", code)
        self.assertIn("cute.nvgpu.warp.MmaF16BF16Op", code)
        self.assertNotIn("cute.arch.warp_reduction_sum", code)

    def test_matmul_mma_tcgen05(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        args = (
            torch.randn(64, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 8, device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            code, out = code_and_output(cute_matmul_mma, args, block_sizes=[64, 8, 16])
        torch.testing.assert_close(out, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertIn("cutlass.utils.blackwell_helpers.make_trivial_tiled_mma", code)
        self.assertIn("cute.nvgpu.tcgen05", code)
        self.assertIn("cute.gemm(", code)
        # ``tcgen05_acc_pipeline_arrive_count`` / ``tcgen05_ab_pipeline_arrive_count``
        # are no longer materialized as named compile-time constants -- they
        # were always literal ints, so codegen now passes the values inline.
        # Pin the inline form instead: the acc consumer group must be sized to
        # the epi warp count (4) and the AB pipeline still uses one TMA arriver.
        self.assertIn(
            "cutlass.pipeline.CooperativeGroup("
            "cutlass.pipeline.Agent.Thread, cutlass.Int32(4))",
            code,
        )
        self.assertIn(
            "cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 1)",
            code,
        )
        self.assertIn("cutlass.pipeline.NamedBarrier(barrier_id=1", code)

    def test_over_budget_tcgen05_falls_back_to_scalar(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        args = (
            torch.randn(128, 256, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(256, 128, device=DEVICE, dtype=HALF_DTYPE),
        )
        config = helion.Config(
            block_sizes=[128, 128, 256],
            tcgen05_ab_stages=2,
        )
        with (
            patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
            patch(
                "helion._compiler.cute.cute_mma.CuteTcgen05Config."
                "per_cta_smem_budget_bytes",
                return_value=200 * 1024,
            ),
        ):
            bound = cute_matmul_mma.bind(args)
            code = bound.to_triton_code(config)
            out = bound.compile_config(config)(*args)
            torch.cuda.synchronize()
        torch.testing.assert_close(out, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertNotIn("cute.gemm(", code)

    def test_batched_baddbmm_mma_tcgen05(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        args = (
            torch.randn(2, 64, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 64, 8, device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            bound = cute_batched_baddbmm_tcgen05.bind(args)
            config = helion.Config(
                block_sizes=[1, 64, 8, 16],
                tcgen05_ab_stages=2,
                tcgen05_acc_stages=2,
                tcgen05_c_stages=2,
            )
            code = bound.to_triton_code(config)
            out = bound.compile_config(config)(*args)
            torch.cuda.synchronize()
        expected = torch.bmm(args[0].float(), args[1].float())
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)
        self.assertIn("cutlass.utils.blackwell_helpers.make_trivial_tiled_mma", code)
        self.assertIn("cute.nvgpu.tcgen05", code)
        self.assertIn("cute.gemm(", code)
        self.assertIn("'lhs_tma_order': (1, 2, 0)", code)
        self.assertIn("'rhs_tma_order': (2, 1, 0)", code)
        self.assertIn("'d_leading_passthrough': True", code)
        self.assertIn("cpasync.tma_partition", code)
        self.assertIn("tcgen05_tma_store_atom", code)
        self.assertNotIn("cute.copy(tcgen05_simt_atom", code)

    def test_batched_baddbmm_mma_tcgen05_two_cta(self) -> None:
        # A leading-batch matmul composes with the CtaGroup.TWO cluster
        # (cluster_m=2, cluster_n=1, 256-row tile): the 2-CTA MMA and TMA
        # multicast run within each (m, n) tile while the batch axis only
        # offsets the per-tile TMA source. Single-CTA tcgen05 maxes at a
        # 128-row M tile, so bm=256 requires the 2-CTA path.
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        args = (
            torch.randn(2, 256, 128, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 128, 256, device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            bound = cute_batched_baddbmm_tcgen05.bind(args)
            config = _batched_tcgen05_two_cta_config()
            code = bound.to_triton_code(config)
            out = bound.compile_config(config)(*args)
            torch.cuda.synchronize()
        expected = torch.bmm(args[0].float(), args[1].float())
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)
        self.assertIn("make_trivial_tiled_mma", code)
        self.assertIn("cute.gemm(", code)
        self.assertIn("CtaGroup.TWO", code)
        self.assertIn("mcast_mask", code)
        self.assertIn("'lhs_tma_order': (1, 2, 0)", code)
        self.assertIn("'rhs_tma_order': (2, 1, 0)", code)

    def test_leading_matmul_accepts_explicit_deep_direct_entry_config(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        args = (
            torch.randn(2, 256, 128, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(2, 128, 256, device=DEVICE, dtype=torch.bfloat16),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            bound = cute_batched_dot_tcgen05.bind(args)
            config = _leading_tcgen05_direct_entry_config()
            code = bound.to_triton_code(config)
            out = bound.compile_config(config)(*args)
            torch.cuda.synchronize()

        expected = torch.bmm(args[0].float(), args[1].float()).to(out.dtype)
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)
        self.assertIn("cute.gemm(", code)
        self.assertIn("CtaGroup.TWO", code)
        self.assertIn("'ab_stage_count': 6", code)

    def test_batched_baddbmm_rowvec_bias_fused_mma_tcgen05_two_cta(self) -> None:
        # A trailing-axis (rowvec) bias fuses into the CtaGroup.TWO batched
        # matmul epilogue. The rank-3 carrier ``[1, BM, BN]`` has a
        # block-size-1 batch-passthrough leading axis; the epilogue
        # analyzer strips it so ``acc + bias[tile_n]`` classifies as the
        # (M, N)-tile rowvec form and splices into the tcgen05 epilogue.
        # Without the strip this store dropped to the loud-failure backstop
        # (``BackendUnsupported``), so a successful compile-and-match here
        # is itself the proof that the bias is fused (not a separate pass).
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        args = (
            torch.randn(2, 256, 128, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 128, 256, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(256, device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            bound = cute_batched_baddbmm_rowvec_bias_tcgen05.bind(args)
            config = _batched_tcgen05_two_cta_config()
            code = bound.to_triton_code(config)
            out = bound.compile_config(config)(*args)
            torch.cuda.synchronize()
        expected = (torch.bmm(args[0].float(), args[1].float()) + args[2].float()).to(
            torch.bfloat16
        )
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)
        self.assertIn("cute.gemm(", code)
        self.assertIn("CtaGroup.TWO", code)
        # Bias folded into the matmul epilogue -> a single device kernel,
        # no separate elementwise bias pass.
        self.assertEqual(code.count("@cute.kernel"), 1)

    def test_batched_exact_residual_fused_mma_tcgen05_two_cta(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        args = (
            torch.randn(2, 256, 128, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 128, 256, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 256, 256, device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            bound = cute_batched_dot_residual_tcgen05.bind(args)
            config = _batched_tcgen05_two_cta_config()
            code = bound.to_triton_code(config)
            out = bound.compile_config(config)(*args)
            torch.cuda.synchronize()
        expected = (torch.bmm(args[0].float(), args[1].float()) + args[2].float()).to(
            torch.bfloat16
        )
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)
        self.assertIn("cute.gemm(", code)
        self.assertEqual(code.count("@cute.kernel"), 1)

    def test_batched_dot_enables_tcgen05_search_and_uses_mma(self) -> None:
        # An accumulating 3-D hl.dot must enable the leading-passthrough
        # tcgen05 search so it can autotune into cute.gemm without a hand-forced
        # config. Bare dot uses the same collective K reduction when eligible.
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        cute_batched_dot_tcgen05.reset()
        args = (
            torch.randn(4, 256, 64, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(4, 64, 256, device=DEVICE, dtype=torch.bfloat16),
        )
        with (
            patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
            patch(
                "helion.language.matmul_ops._cuda_num_sms_or_zero",
                return_value=0,
            ),
        ):
            bound = cute_batched_dot_tcgen05.bind(args)
            # The batched search surface is enabled (this is what lets the
            # autotuner reach tcgen05 without a hand-forced config).
            self.assertTrue(bound.config_spec.cute_tcgen05_search_enabled)
            self.assertEqual(
                bound.config_spec._tcgen05_cluster_m_search_choices,
                (1, 2),
            )
            self.assertEqual(
                [spec.min_size for spec in bound.config_spec.block_sizes],
                [1, 128, 8, 16],
            )
            self.assertEqual(
                [spec.max_size for spec in bound.config_spec.block_sizes],
                [1, 256, 256, 64],
            )
            self.assertEqual(len(bound.config_spec.matmul_facts), 1)
            seed_layouts = {
                seed.config.get("tcgen05_layout_strategy")
                for seed in bound.config_spec.compiler_seed_configs
                if seed.block_sizes == [1, 256, 256, 64]
            }
            self.assertEqual(seed_layouts, {None, "explicit_epi_tile"})

            projected = {
                "block_sizes": [1, 128, 128, 64],
                "pid_type": "flat",
                "tcgen05_cluster_m": 2,
            }
            bound.config_spec.normalize(projected, _fix_invalid=True)
            # Stage 2b made a generic DEFAULT-layout bn=128 cluster_m=2 tile
            # searchable for the batched leading-passthrough family, so the
            # sampled block_n=128 (index 2) now SURVIVES at 128 instead of
            # snapping to the canonical 256. block_m (index 1) is still pinned
            # to 256.
            self.assertEqual(projected["block_sizes"], [1, 256, 128, 64])
            self.assertEqual(projected["pid_type"], "persistent_interleaved")

            def search_spec(m: int, n: int, k: int) -> ConfigSpec:
                operands = (
                    torch.empty(2, m, k, device=DEVICE, dtype=HALF_DTYPE),
                    torch.empty(2, k, n, device=DEVICE, dtype=HALF_DTYPE),
                )
                return cute_batched_dot_tcgen05.bind(operands).config_spec

            # Leading-axis tcgen05 codegen requires full M/N/K tiles. Shapes
            # without a legal full tile stay on the generic search surface.
            self.assertFalse(search_spec(160, 256, 128).cute_tcgen05_search_enabled)
            divisor_spec = search_spec(320, 256, 128)
            self.assertTrue(divisor_spec.cute_tcgen05_search_enabled)
            self.assertEqual(divisor_spec.block_sizes[1].max_size, 64)

            # BMM's N slot is index 2, not the rank-2 index 1. Clamp against
            # the analyzed axis so a one-cluster N dimension cannot keep 8.
            swizzle_spec = search_spec(2048, 256, 128)
            swizzle_config = {
                "block_sizes": [1, 128, 256, 64],
                "tcgen05_l2_swizzle_size": 8,
            }
            swizzle_spec.normalize(swizzle_config, _fix_invalid=True)
            self.assertEqual(swizzle_config["tcgen05_l2_swizzle_size"], 1)

            config = _batched_tcgen05_two_cta_config()
            code = bound.to_triton_code(config)
            out = bound.compile_config(config)(*args)
            torch.cuda.synchronize()
        expected = torch.bmm(args[0].float(), args[1].float())
        torch.testing.assert_close(out.float(), expected, atol=1e-1, rtol=1e-2)
        self.assertIn("cute.gemm(", code)
        self.assertIn("CtaGroup.TWO", code)

    def test_batched_two_cta_search_counts_batch_clusters(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        def cluster_choices(batch: int) -> tuple[int, ...] | None:
            cute_batched_dot_tcgen05.reset()
            args = (
                torch.empty(batch, 256, 64, device=DEVICE, dtype=HALF_DTYPE),
                torch.empty(batch, 64, 256, device=DEVICE, dtype=HALF_DTYPE),
            )
            return cute_batched_dot_tcgen05.bind(
                args
            ).config_spec._tcgen05_cluster_m_search_choices

        with patch(
            "helion.language.matmul_ops._cuda_num_sms_or_zero",
            return_value=148,
        ):
            # The wave-quant gate multiplies the per-batch output-cluster count
            # by the batch (``leading_work_multiplier``); cluster_m=2 search is
            # only admitted once that product clears ``num_sms // 4`` (= 37).
            # Stage 2b made the batched leading-passthrough family search a
            # bn=128 cluster_m=2 tile, so the count is taken at the narrow tile
            # (256x128 -> 256//256 * 256//128 = 2 clusters per batch), moving the
            # threshold to batch >= 19 (2 * 19 = 38 >= 37). A small batch that
            # underfills stays cluster_m=1; a large batch enables cluster_m=2.
            # If the batch factor were NOT counted the product would be a
            # batch-independent 2 and cluster_m=2 would never enable -- so this
            # boundary is what proves batch clusters are counted.
            self.assertEqual(cluster_choices(16), (1,))
            self.assertEqual(cluster_choices(32), (1, 2))

    def test_batched_direct_entry_seeds_match_epilogue_support(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        cases = (
            (
                cute_batched_dot_residual_tcgen05,
                (
                    torch.empty(2, 256, 64, device=DEVICE, dtype=torch.bfloat16),
                    torch.empty(2, 64, 256, device=DEVICE, dtype=torch.bfloat16),
                    torch.empty(2, 256, 256, device=DEVICE, dtype=torch.bfloat16),
                ),
                {None},
            ),
            (
                cute_batched_baddbmm_rowvec_bias_tcgen05,
                (
                    torch.empty(2, 256, 64, device=DEVICE, dtype=torch.bfloat16),
                    torch.empty(2, 64, 256, device=DEVICE, dtype=torch.bfloat16),
                    torch.empty(256, device=DEVICE, dtype=torch.bfloat16),
                ),
                {None, "explicit_epi_tile"},
            ),
            (
                cute_rhs_batched_dot_tcgen05,
                (
                    torch.empty(64, 256, device=DEVICE, dtype=torch.bfloat16).T,
                    torch.empty(2, 64, 256, device=DEVICE, dtype=torch.bfloat16),
                ),
                {None},
            ),
        )
        with patch(
            "helion.language.matmul_ops._cuda_num_sms_or_zero",
            return_value=0,
        ):
            for kernel, args, expected_layouts in cases:
                with self.subTest(kernel=kernel.name):
                    kernel.reset()
                    bound = kernel.bind(args)
                    layouts = {
                        seed.config.get("tcgen05_layout_strategy")
                        for seed in bound.config_spec.compiler_seed_configs
                        if seed.block_sizes == [1, 256, 256, 64]
                    }
                    self.assertEqual(layouts, expected_layouts)
                    config_generation = bound.config_spec.create_config_generation()
                    normalized_layouts = {
                        (
                            seed.config.get("tcgen05_layout_strategy"),
                            seed.config.get("tcgen05_tvm_ffi_launch") is True,
                        )
                        for _, seed in config_generation.seed_flat_config_pairs()
                        if seed.block_sizes == [1, 256, 256, 64]
                    }
                    self.assertEqual(
                        normalized_layouts,
                        {
                            (
                                layout or "default",
                                layout == "explicit_epi_tile",
                            )
                            for layout in expected_layouts
                        },
                    )

    def test_bare_batched_matmuls_use_collective_k_reduction(self) -> None:
        args = (
            torch.randn(2, 64, 32, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 32, 8, device=DEVICE, dtype=HALF_DTYPE),
        )
        expected = torch.bmm(args[0], args[1])
        expected_by_kernel = {
            cute_bare_bmm: expected,
            cute_bare_bmm_dtype: torch.bmm(
                args[0].float(),
                args[1].float(),
            ),
            cute_bare_batched_dot: torch.bmm(
                args[0].float(),
                args[1].float(),
            ),
        }
        config = helion.Config(block_sizes=[1, 64, 8, 16])
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            for kernel in (
                cute_bare_bmm,
                cute_bare_bmm_dtype,
                cute_bare_batched_dot,
            ):
                with self.subTest(kernel=kernel.name):
                    bound = kernel.bind(args)
                    self.assertTrue(bound.config_spec.cute_tcgen05_search_enabled)
                    code = bound.to_triton_code(config)
                    out = bound.compile_config(config)(*args)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        out,
                        expected_by_kernel[kernel],
                        atol=1e-1,
                        rtol=1e-2,
                    )
                    self.assertIn("cute.gemm(", code)

    def test_unrelated_root_axis_does_not_specialize_2d_dot(self) -> None:
        args = (
            torch.randn(64, 32, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(32, 8, device=DEVICE, dtype=HALF_DTYPE),
            torch.empty(2, device=DEVICE),
        )
        config = helion.Config(block_sizes=[1, 64, 8, 16])
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            bound = cute_repeated_2d_dot.bind(args)
            self.assertFalse(bound.config_spec.cute_tcgen05_search_enabled)
            code = bound.to_triton_code(config)
            out = bound.compile_config(config)(*args)
            torch.cuda.synchronize()
        expected = (args[0].float() @ args[1].float()).to(torch.bfloat16)
        torch.testing.assert_close(
            out,
            expected.unsqueeze(0).expand(2, -1, -1),
            atol=1e-1,
            rtol=1e-2,
        )
        self.assertNotIn("cute.gemm(", code)

    def test_batched_universal_matmul_falls_back_from_collective(self) -> None:
        cases = (
            (torch.float32, 16, 8),
            (HALF_DTYPE, 16, 16),
        )
        for dtype, block_m, block_n in cases:
            with self.subTest(dtype=str(dtype), block_n=block_n):
                args = (
                    torch.randn(2, block_m, 32, device=DEVICE, dtype=dtype),
                    torch.randn(2, 32, block_n, device=DEVICE, dtype=dtype),
                )
                code, out = code_and_output(
                    cute_batched_baddbmm_tcgen05,
                    args,
                    block_sizes=[1, block_m, block_n, 16],
                    num_threads=[1, block_m, block_n, 1],
                )
                torch.testing.assert_close(
                    out,
                    torch.bmm(args[0].float(), args[1].float()),
                    atol=1e-1,
                    rtol=1e-2,
                )
                self.assertNotIn("cute.gemm(", code)

    def test_batched_strided_matrix_layouts_fall_back(self) -> None:
        contiguous_x = torch.randn(2, 64, 32, device=DEVICE, dtype=HALF_DTYPE)
        contiguous_y = torch.randn(2, 32, 8, device=DEVICE, dtype=HALF_DTYPE)
        cases = (
            (
                torch.randn(2, 32, 64, device=DEVICE, dtype=HALF_DTYPE).transpose(
                    -1, -2
                ),
                contiguous_y,
            ),
            (
                contiguous_x,
                torch.randn(2, 8, 32, device=DEVICE, dtype=HALF_DTYPE).transpose(
                    -1, -2
                ),
            ),
        )
        config = helion.Config(block_sizes=[1, 64, 8, 16])
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            for args in cases:
                with self.subTest(
                    lhs_stride=args[0].stride(), rhs_stride=args[1].stride()
                ):
                    bound = cute_batched_baddbmm_tcgen05.bind(args)
                    self.assertFalse(bound.config_spec.cute_tcgen05_search_enabled)
                    code = bound.to_triton_code(config)
                    out = bound.compile_config(config)(*args)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        out,
                        torch.bmm(args[0].float(), args[1].float()),
                        atol=1e-1,
                        rtol=1e-2,
                    )
                    self.assertNotIn("cute.gemm(", code)

    def test_mixed_rank_batched_dot_uses_tcgen05(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        cases = (
            (
                cute_mixed_rank_batched_dot_tcgen05,
                (
                    torch.randn(4, 256, 64, device=DEVICE, dtype=HALF_DTYPE),
                    torch.randn(64, 256, device=DEVICE, dtype=HALF_DTYPE),
                ),
                "'lhs_tma_order': (1, 2, 0)",
            ),
            (
                cute_rhs_batched_dot_tcgen05,
                (
                    torch.randn(256, 64, device=DEVICE, dtype=HALF_DTYPE),
                    torch.randn(4, 64, 256, device=DEVICE, dtype=HALF_DTYPE),
                ),
                "'rhs_tma_order': (2, 1, 0)",
            ),
            (
                cute_mixed_rank_batched_dot_tcgen05,
                (
                    torch.randn(4, 256, 64, device=DEVICE, dtype=HALF_DTYPE),
                    torch.randn(256, 64, device=DEVICE, dtype=HALF_DTYPE).T,
                ),
                "'lhs_tma_order': (1, 2, 0)",
            ),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            for kernel, args, plan_marker in cases:
                with self.subTest(kernel=kernel.name):
                    kernel.reset()
                    bound = kernel.bind(args)
                    self.assertTrue(bound.config_spec.cute_tcgen05_search_enabled)
                    code = bound.to_triton_code(_batched_tcgen05_two_cta_config())
                    out = bound.compile_config(_batched_tcgen05_two_cta_config())(*args)
                    torch.testing.assert_close(
                        out.float(),
                        torch.matmul(args[0].float(), args[1].float()),
                        atol=1e-1,
                        rtol=1e-2,
                    )
                    self.assertIn("cute.gemm(", code)
                    self.assertIn(plan_marker, code)

    def test_casted_batched_dot_falls_back_correctly(self) -> None:
        args = (
            torch.randn(2, 128, 64, device=DEVICE, dtype=torch.float16),
            torch.randn(2, 64, 8, device=DEVICE, dtype=torch.float16),
        )
        config = helion.Config(block_sizes=[1, 128, 8, 64], pid_type="flat")
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            bound = cute_casted_batched_dot_tcgen05.bind(args)
            self.assertFalse(bound.config_spec.cute_tcgen05_search_enabled)
            code = bound.to_triton_code(config)
            out = bound.compile_config(config)(*args)
        expected = torch.bmm(
            args[0].to(torch.bfloat16).float(),
            args[1].to(torch.bfloat16).float(),
        )
        self.assertNotIn("cute.gemm(", code)
        torch.testing.assert_close(out.float(), expected, atol=1e-1, rtol=1e-2)

    def test_unsupported_batched_epilogue_falls_back_correctly(self) -> None:
        args = (
            torch.randn(2, 128, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 64, 8, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 128, 8, device=DEVICE),
        )
        config = helion.Config(block_sizes=[1, 128, 8, 64], pid_type="flat")
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            bound = cute_batched_dot_unsupported_epilogue_tcgen05.bind(args)
            self.assertFalse(bound.config_spec.cute_tcgen05_search_enabled)
            code = bound.to_triton_code(config)
            out = bound.compile_config(config)(*args)
        expected = torch.bmm(args[0].float(), args[1].float()) + torch.sin(args[2])
        self.assertNotIn("cute.gemm(", code)
        torch.testing.assert_close(out.float(), expected, atol=1e-1, rtol=1e-2)

    def test_batched_dot_codegen_rejected_does_not_shape_search(self) -> None:
        # F2 regression: batched tcgen05 search enablement for hl.dot must be
        # gated on the SAME structural analyzer codegen uses
        # (analyze_cute_mma_node), not on operand rank alone. A rank-3 dot
        # that codegen cannot lower (here: a transposed LHS operand) must NOT
        # shape the batched search surface -- otherwise the autotuner tunes a
        # config family for a kernel codegen later rejects. A clean batched dot
        # of the same rank/shape stays enabled, proving the gate is structural,
        # not rank-based.
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        clean = (
            torch.randn(4, 256, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(4, 64, 256, device=DEVICE, dtype=HALF_DTYPE),
        )
        rejected = (
            torch.randn(4, 64, 256, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(4, 64, 256, device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            clean_bound = cute_batched_dot_tcgen05.bind(clean)
            rejected_bound = cute_transposed_operand_batched_dot_tcgen05.bind(rejected)
            shifted_bound = cute_shifted_batched_dot_tcgen05.bind(clean)
            # Same rank (3-D dot), opposite enablement -> the gate is structural.
            self.assertTrue(clean_bound.config_spec.cute_tcgen05_search_enabled)
            self.assertFalse(rejected_bound.config_spec.cute_tcgen05_search_enabled)
            self.assertFalse(shifted_bound.config_spec.cute_tcgen05_search_enabled)

    def test_transformed_2d_dot_does_not_shape_tcgen05_search(self) -> None:
        args = (
            torch.randn(256, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 256, device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            bound = cute_transformed_dot_tcgen05.bind(args)
        self.assertFalse(bound.config_spec.cute_tcgen05_search_enabled)

    def test_permuted_output_store_falls_back_correctly(self) -> None:
        args = (
            torch.randn(2, 128, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 64, 8, device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            bound = cute_permuted_store_batched_dot_tcgen05.bind(args)
            self.assertFalse(bound.config_spec.cute_tcgen05_search_enabled)
            config = helion.Config(block_sizes=[1, 128, 8, 64], pid_type="flat")
            code = bound.to_triton_code(config)
            out = bound.compile_config(config)(*args)
        expected = torch.bmm(args[0].float(), args[1].float()).transpose(-1, -2)
        self.assertNotIn("cute.gemm(", code)
        torch.testing.assert_close(out, expected.to(out.dtype), atol=1e-1, rtol=1e-2)

    def test_batched_two_cta_partial_edge_tiles_rejected(self) -> None:
        # A batched CtaGroup.TWO matmul with ANY partial tile -- M edge, N
        # edge, OR a K tail -- must be rejected loudly: the output-edge
        # scheduler linearizes the virtual pid across the batch axis and the
        # K-tail reduction is batch-unaware, so both would silently miscompute
        # (only static full tiles are validated for batched 2-CTA). Each
        # (M, K) x (K, N) below makes exactly one axis partial vs the
        # 256x256x64 blocks (plus the combined case).
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        config = _batched_tcgen05_two_cta_config()
        # (M, N, K): M-edge, N-edge, K-tail (M/N full), double-edge + K-tail.
        partial_shapes = [
            (300, 256, 128),
            (256, 300, 128),
            (256, 256, 100),
            (300, 300, 100),
        ]

        def _bmm_args(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.randn(2, m, k, device=DEVICE, dtype=HALF_DTYPE),
                torch.randn(2, k, n, device=DEVICE, dtype=HALF_DTYPE),
            )

        for m, n, k in partial_shapes:
            args = _bmm_args(m, n, k)
            with patch.dict(
                os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False
            ):
                bound = cute_batched_baddbmm_tcgen05.bind(args)
                self.assertFalse(bound.config_spec.cute_tcgen05_search_enabled)
                with self.assertRaisesRegex(
                    helion.exc.InvalidConfig,
                    "only supported for tcgen05-enabled CuTe matmul kernels",
                    msg=f"2-CTA partial M={m} N={n} K={k} not rejected",
                ):
                    bound.to_triton_code(config)

    def test_batched_one_cta_partial_tiles_fall_back_correctly(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        # Leading-passthrough MMA is validated only for static full M/N/K
        # tiles. Partial configs must use the scalar fallback.
        cases = [
            ("N edge", (2, 64, 32), (2, 32, 10)),
            ("K tail", (2, 64, 35), (2, 35, 8)),
        ]
        for label, lhs_shape, rhs_shape in cases:
            for mma_impl in ("tcgen05", "universal"):
                args = (
                    torch.randn(*lhs_shape, device=DEVICE, dtype=HALF_DTYPE),
                    torch.randn(*rhs_shape, device=DEVICE, dtype=HALF_DTYPE),
                )
                with (
                    self.subTest(case=label, mma_impl=mma_impl),
                    patch.dict(
                        os.environ,
                        {"HELION_CUTE_MMA_IMPL": mma_impl},
                        clear=False,
                    ),
                ):
                    bound = cute_batched_baddbmm_tcgen05.bind(args)
                    self.assertFalse(bound.config_spec.cute_tcgen05_search_enabled)
                    config = helion.Config(
                        block_sizes=[1, 64, 8, 16],
                        pid_type="flat",
                    )
                    code = bound.to_triton_code(config)
                    out = bound.compile_config(config)(*args)
                    expected = torch.bmm(args[0].float(), args[1].float())
                    self.assertNotIn("cute.gemm(", code)
                    torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)

    def test_batched_tcgen05_cluster_n_rejected_cleanly(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        args = (
            torch.randn(2, 256, 128, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 128, 256, device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            bound = cute_batched_baddbmm_tcgen05.bind(args)
            with self.assertRaisesRegex(
                helion.exc.BackendUnsupported,
                "leading passthrough axis does not support",
            ):
                bound.to_triton_code(_batched_tcgen05_two_cta_config(cluster_n=2))

    def test_matmul_mma_tcgen05_128x8_uses_full_cta_barrier(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        args = (
            torch.randn(128, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 8, device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            code, out = code_and_output(cute_matmul_mma, args, block_sizes=[128, 8, 16])
        torch.testing.assert_close(out, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertIn("cute.nvgpu.tcgen05", code)
        # Pin the inline arrive-count form (cf. ``test_matmul_mma_tcgen05``).
        self.assertIn(
            "cutlass.pipeline.CooperativeGroup("
            "cutlass.pipeline.Agent.Thread, cutlass.Int32(4))",
            code,
        )
        self.assertIn("cutlass.pipeline.NamedBarrier(barrier_id=1", code)

    def test_matmul_mma_tcgen05_fp8(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")

        torch.manual_seed(0)
        x = (torch.randn(256, 128, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(128, 128, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        code, out = code_and_output(
            cute_matmul_mma_fp8, (x, y), block_sizes=[128, 128, 128]
        )
        ref = x.float() @ y.float()
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=1e-1)
        # fp8 routes through the tcgen05 F8F6F4 MMA atom (MMA-K=32).
        self.assertIn("cutlass.utils.blackwell_helpers.make_trivial_tiled_mma", code)
        self.assertIn("cutlass.Float8E4M3FN", code)
        self.assertIn("cute.nvgpu.tcgen05", code)
        self.assertIn("cute.gemm(", code)

    def test_matmul_mma_tcgen05_fp8_col_major_b(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")

        torch.manual_seed(0)
        x = (torch.randn(256, 128, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        # Column-major (K-contiguous) B. Helion must emit a K-major B operand
        # (OperandMajorMode.K for B) and a matching K-major B SMEM layout,
        # rather than forcing the slow non-TMA fallback.
        y = (torch.randn(128, 128, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = y.T.contiguous().T
        self.assertFalse(y.is_contiguous())
        code, out = code_and_output(
            cute_matmul_mma_fp8, (x, y), block_sizes=[128, 128, 128]
        )
        ref = x.float() @ y.float()
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=1e-1)
        # B is emitted K-major: both A and B operand major modes are K, so
        # OperandMajorMode.K appears at least twice (A + B); the MN-major B
        # spelling must be absent.
        self.assertIn("cutlass.Float8E4M3FN", code)
        self.assertIn("cute.nvgpu.tcgen05", code)
        self.assertGreaterEqual(code.count("cute.nvgpu.OperandMajorMode.K"), 2)
        self.assertNotIn("cute.nvgpu.OperandMajorMode.MN", code)

    def test_matmul_mma_tcgen05_fp8_rowvec_scale(self) -> None:
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")

        torch.manual_seed(0)
        x = (torch.randn(256, 128, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(128, 128, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        scale_n = torch.rand(128, device=DEVICE) + 0.5
        code, out = code_and_output(
            cute_matmul_mma_fp8_rowvec_scale,
            (x, y, scale_n),
            block_sizes=[128, 128, 128],
        )
        ref = (x.float() @ y.float()) * scale_n.float()
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=1e-1)
        self.assertIn("cutlass.Float8E4M3FN", code)
        self.assertIn("cute.nvgpu.tcgen05", code)

    def test_matmul_mma_tcgen05_fp8_rowvec_scale_block_m64(self) -> None:
        # block_m=64 is below the T2R atom's M extent, so each thread's
        # per-subtile epilogue fragment spans MULTIPLE M rows. A rowvec aux
        # (``scale_n[n]``) partitions with a stride-0 M mode that does not
        # coalesce to the accumulator carrier's flat profile, so a plain
        # ``.load()`` used to raise ``profile of input tuples doesn't match:
        # (8, (2, 2, 2))`` at trace time. Regression guard for the
        # broadcast-aux dense-materialization path. All existing rowvec
        # tests use block_m>=128, where the broadcast mode is size 1 and the
        # bug is invisible.
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")

        torch.manual_seed(0)
        x = (torch.randn(64, 256, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(256, 256, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        scale_n = torch.rand(256, device=DEVICE) + 0.5
        code, out = code_and_output(
            cute_matmul_mma_fp8_rowvec_scale,
            (x, y, scale_n),
            block_sizes=[64, 128, 32],
            tcgen05_strategy="role_local_monolithic",
            tcgen05_persistence_model="non_persistent",
            pid_type="flat",
        )
        ref = (x.float() @ y.float()) * scale_n.float()
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=1e-1)
        self.assertFalse(out.float().isnan().any().item())
        self.assertIn("cute.nvgpu.tcgen05", code)

    def test_matmul_mma_tcgen05_fp8_rowwise_colwise_scale_block_m64(self) -> None:
        # Both broadcast-aux directions (per-row colvec ``scale_m[m]`` AND
        # per-column rowvec ``scale_n[n]``) in one epilogue chain at
        # block_m=64. The colvec scalar fast-path (a single T2R read at
        # ``(0,0,0,subtile)``) is only valid when each thread's fragment lies
        # within a single M row; if it spans multiple rows, applying row 0's
        # scale everywhere silently corrupts the output. This is the fp8
        # rowwise-x-rowwise pattern that the M=512/M=64 fp8_gemm dashboard
        # shapes hit via the autotuner's default config.
        #
        # Uses a STRONGLY row-dependent ``scale_m`` (row i -> i+1) so a
        # wrong-row read is off by a large factor, not masked by a loose
        # tolerance on a near-uniform random scale.
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")

        torch.manual_seed(0)
        m, k, n = 64, 256, 256
        x = (torch.randn(m, k, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(k, n, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        # Per-row scale fed as a broadcast (m, 1) -> (m, n) view, matching the
        # fp8 rowwise GEMM operator (``scale_a.reshape(-1, 1).expand(m, n)``):
        # this classifies as a colvec (``broadcast_axis == 2``) aux.
        scale_m = (
            (torch.arange(m, device=DEVICE, dtype=torch.float32) + 1.0)
            .reshape(m, 1)
            .expand(m, n)
        )
        scale_n = torch.rand(n, device=DEVICE) + 0.5
        code, out = code_and_output(
            cute_matmul_mma_fp8_rowwise_colwise_scale,
            (x, y, scale_m, scale_n),
            block_sizes=[64, 128, 32],
            tcgen05_strategy="role_local_monolithic",
            tcgen05_persistence_model="non_persistent",
            pid_type="flat",
        )
        ref = (x.float() @ y.float()) * scale_m.float() * scale_n.float().reshape(1, -1)
        # Relative check: a row-broadcast bug would scale row i by 1 instead of
        # (i+1), diverging by up to m x on the later rows.
        torch.testing.assert_close(out.float(), ref, atol=2.0, rtol=5e-2)
        self.assertFalse(out.float().isnan().any().item())
        self.assertIn("cute.nvgpu.tcgen05", code)

    def _run_colvec_scale_row_dependent(
        self,
        block_sizes: list[int],
        *,
        m: int = 64,
        n: int = 256,
        **config_kwargs: object,
    ) -> str:
        # Shared body: per-row colvec scale with a STRONGLY row-dependent
        # ``scale_m`` (row i -> i+1). The colvec scalar fast-path is only valid
        # when each thread's epilogue fragment lies within one M row; if the
        # ``tcgen05_colvec_fragment_single_m_row`` predicate (epi_tile_m >= the
        # 128-lane TMEM datapath) were wrong, a thread spanning multiple rows
        # would read row 0's scale for every row and diverge by up to m x.
        # This is the runnable form of the per-thread M-extent check: a passing
        # numeric assertion proves the fragment was single-M-row wherever the
        # scalar arm was emitted. Returns the generated code for optional
        # inspection.
        k = 256
        x = (torch.randn(m, k, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(k, n, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        scale_m = (
            (torch.arange(m, device=DEVICE, dtype=torch.float32) + 1.0)
            .reshape(m, 1)
            .expand(m, n)
        )
        kwargs: dict[str, object] = {
            "tcgen05_strategy": "role_local_monolithic",
            "tcgen05_persistence_model": "non_persistent",
            "pid_type": "flat",
        }
        kwargs.update(config_kwargs)
        code, out = code_and_output(
            cute_matmul_mma_fp8_colvec_scale,
            (x, y, scale_m),
            block_sizes=block_sizes,
            **kwargs,
        )
        ref = (x.float() @ y.float()) * scale_m.float()
        torch.testing.assert_close(out.float(), ref, atol=2.0, rtol=5e-2)
        self.assertFalse(out.float().isnan().any().item())
        return code

    def test_matmul_mma_tcgen05_fp8_colvec_scale_block_m64_row_dependent(self) -> None:
        # block_m=64: epi_tile_m=64 < 128, so a thread's fragment spans
        # multiple M rows and the predicate selects the dense materialize.
        # A wrong predicate (scalar read here) would scale row i by 1 instead
        # of (i+1); the row-dependent reference catches that.
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")
        torch.manual_seed(0)
        self._run_colvec_scale_row_dependent([64, 128, 32])

    def test_matmul_mma_tcgen05_fp8_colvec_scale_block_m128_row_dependent(
        self,
    ) -> None:
        # block_m=128: epi_tile_m=128, single-M-row fragment -- the #2742
        # scalar fast-path regime. Confirms the per-row value stays correct
        # under a row-dependent scale.
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")
        torch.manual_seed(0)
        self._run_colvec_scale_row_dependent([128, 128, 32])

    def test_matmul_mma_tcgen05_fp8_colvec_scale_2cta_m256_row_dependent(
        self,
    ) -> None:
        # block_m=256 + cluster_m=2 (2-CTA bm=128 family): the per-CTA epilogue
        # tile M is bm // 2 = 128, so the fragment is single-M-row and the
        # scalar fast-path is valid -- but only because the predicate uses
        # epi_tile_m (bm // 2), not bm. This is the branch of
        # ``tcgen05_colvec_fragment_single_m_row`` a plain ``bm >= 128`` would
        # also get right but a ``bm // 2``-unaware test would not exercise;
        # the row-dependent scale catches a wrong per-CTA tile-M derivation.
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")
        torch.manual_seed(0)
        self._run_colvec_scale_row_dependent(
            [256, 256, 64],
            m=256,
            n=256,
            tcgen05_cluster_m=2,
            pid_type="persistent_blocked",
            tcgen05_persistence_model="static_persistent",
        )

    def test_matmul_mma_tcgen05_epilogue_exact_aux_block_m64(self) -> None:
        # Exact-shape (full (m, n), non-broadcast) fused aux at block_m=64.
        # The per-thread fragment spans multiple M rows, so the plain
        # ``.load()`` profile no longer matches the coalesced accumulator
        # carrier and the chain add hit ``profile of input tuples doesn't
        # match: (8, (2, 2, 2))``. Regression guard for the exact-shape arm
        # of the dense-materialization fix (distinct from the broadcast
        # rowvec/colvec arms).
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")

        torch.manual_seed(0)
        m, k, n = 64, 256, 256
        x = (torch.randn(m, k, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(k, n, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        bias = torch.randn(m, n, device=DEVICE)
        code, out = code_and_output(
            cute_matmul_mma_epilogue_f32_bias,
            (x, y, bias),
            block_sizes=[64, 128, 32],
            tcgen05_strategy="role_local_monolithic",
            tcgen05_persistence_model="non_persistent",
            pid_type="flat",
        )
        ref = (x.float() @ y.float()) + bias.float()
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=1e-1)
        self.assertFalse(out.float().isnan().any().item())
        self.assertIn("cute.nvgpu.tcgen05", code)

    def test_matmul_mma_tcgen05_fp8_cluster_m2_persistent(self) -> None:
        """Test FP8 E4M3 with cluster_m=2 persistent scheduling."""
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")

        torch.manual_seed(0)
        x = (torch.randn(512, 2048, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(2048, 2048, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)

        # Use block_m=256 to enable is_two_cta (required for cluster_m=2 role-local)
        code, out = code_and_output(
            cute_matmul_mma_fp8,
            (x, y),
            block_sizes=[256, 256, 64],
            tcgen05_cluster_m=2,
            pid_type="persistent_blocked",
        )
        ref = x.float() @ y.float()
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=1e-1)

        # Verify FP8 dtype, tcgen05 backend, cluster_m=2, and persistent scheduler
        self.assertIn("cutlass.Float8E4M3FN", code)
        self.assertIn("cute.nvgpu.tcgen05", code)
        self.assertIn("(2, 1, 1)", code)  # cluster_m=2
        self.assertIn("StaticPersistentTileScheduler", code)

    def test_matmul_mma_tcgen05_fp8_two_cta_m128_codegen_and_correctness(
        self,
    ) -> None:
        """bm=128 + cluster_m=2 on fp8 selects the 2-CTA MMA (CTA tile 64xbn).

        The epilogue must use the per-CTA tile convention throughout:
        ``compute_epilogue_tile_shape((64, bn), True, ...)`` (whose tile is
        N-mode permuted), a kernel_desc with ``cta_tile_shape_mnk`` of
        ``(64, bn, bk)``, and a host TMA store atom built from the same
        expression via the ``epi_tile_raw_expr`` wrapper-plan key. A plain
        ``(m, n)`` tile on any side silently permutes the output.
        """
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")

        torch.manual_seed(0)
        x = (torch.randn(256, 512, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(512, 384, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        code, out = code_and_output(
            cute_matmul_mma_fp8,
            (x, y),
            block_sizes=[128, 128, 128],
            tcgen05_cluster_m=2,
            pid_type="persistent_blocked",
        )
        ref = x.float() @ y.float()
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=1e-1)
        self.assertFalse(out.float().isnan().any().item())
        # 2-CTA MMA at the (128, bn) MMA tiler.
        self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
        self.assertNotIn("cute.nvgpu.tcgen05.CtaGroup.ONE", code)
        # Per-CTA epilogue tile convention: (64, bn) + use_2cta=True, and the
        # kernel_desc carries the per-CTA tile.
        self.assertIn(
            "compute_epilogue_tile_shape((64, 128), True",
            code,
        )
        self.assertIn("'cta_tile_shape_mnk': (64, 128, 128)", code)
        self.assertIn("get_tmem_load_op((64, 128, 128)", code)
        # Host TMA store atom is built from the device-exact tile expression.
        self.assertIn("'epi_tile_raw_expr'", code)
        # The resolved CtaGroup decision is recorded for the host wrapper.
        self.assertIn("'use_2cta_instrs': True", code)

    def test_matmul_mma_tcgen05_fp8_two_cta_m128_rowvec_scale(self) -> None:
        """Fused rowvec-scale epilogue on the bm=128 2-CTA family.

        The rowvec aux fragment is partitioned through the same N-mode
        permuted epilogue tile as the accumulator; a convention mismatch
        shows up as scrambled (not just scaled-wrong) output.
        """
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")

        torch.manual_seed(0)
        x = (torch.randn(256, 512, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(512, 256, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        scale_n = torch.rand(256, device=DEVICE) + 0.5
        code, out = code_and_output(
            cute_matmul_mma_fp8_rowvec_scale,
            (x, y, scale_n),
            block_sizes=[128, 128, 128],
            tcgen05_cluster_m=2,
            pid_type="persistent_blocked",
        )
        ref = (x.float() @ y.float()) * scale_n.float()
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=1e-1)
        self.assertFalse(out.float().isnan().any().item())
        self.assertIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)
        self.assertIn("compute_epilogue_tile_shape((64, 128), True", code)

    def test_matmul_mma_tcgen05_fp8_two_cta_m128_rowvec_prewait_hoist(self) -> None:
        """The bm=128 2-CTA family pre-hoists rowvec aux above the acc wait.

        One whole-fragment ``autovec_copy`` into registers is emitted in the
        per-tile setup (before the accumulator ``consumer_wait``) so the
        rowvec GMEM latency hides under the MMA wait; the per-subtile loop
        slices the register tensor instead of issuing per-subtile LDGs.
        bm=256 must keep the per-subtile GMEM load (the whole-tile hoist
        historically caused register spills there).
        """
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")

        torch.manual_seed(0)
        x = (torch.randn(256, 512, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(512, 256, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        scale_n = torch.rand(256, device=DEVICE) + 0.5
        code, out = code_and_output(
            cute_matmul_mma_fp8_rowvec_scale,
            (x, y, scale_n),
            block_sizes=[128, 128, 128],
            tcgen05_cluster_m=2,
            pid_type="persistent_blocked",
            tcgen05_aux_load_placement="pre_acc_wait",
        )
        ref = (x.float() @ y.float()) * scale_n.float()
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=1e-1)
        # Whole-fragment register hoist present...
        self.assertIn("tcgen05_aux_rmem_full_", code)
        hoist_pos = code.index("cute.autovec_copy(tcgen05_tTR_gAux_grouped_")
        # ...and emitted before the accumulator consumer_wait.
        acc_wait_pos = code.index(".consumer_wait(tcgen05_acc_consumer_state)")
        self.assertLess(hoist_pos, acc_wait_pos)
        # The subtile loop reads the register tensor, not per-subtile GMEM.
        self.assertNotIn("tcgen05_tTR_gAux_subtile_", code)

        # bm=256 keeps the per-subtile GMEM load (no whole-fragment hoist).
        code256 = cute_matmul_mma_fp8_rowvec_scale.bind((x, y, scale_n)).to_triton_code(
            helion.Config(
                block_sizes=[256, 128, 128],
                tcgen05_cluster_m=2,
                pid_type="persistent_blocked",
                tcgen05_aux_load_placement="pre_acc_wait",
            )
        )
        self.assertNotIn("tcgen05_aux_rmem_full_", code256)

    def test_matmul_mma_tcgen05_f16_m128_cluster_m2_keeps_cta_group_one(
        self,
    ) -> None:
        """f16/bf16 bm=128 + cluster_m=2 stays on the legacy CTA-local family.

        That config point is owned by the guarded CtaGroup.ONE diagnostic
        bridge and the multi-tile runtime guard; the fp8-only gate on the
        bm=128 2-CTA family must not change f16 codegen.
        """
        support = get_cute_mma_support()
        if not support.tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

        torch.manual_seed(0)
        x = torch.randn(256, 64, device=DEVICE, dtype=torch.float16)
        y = torch.randn(64, 256, device=DEVICE, dtype=torch.float16)
        code = cute_matmul_mma.bind((x, y)).to_triton_code(
            helion.Config(
                block_sizes=[128, 128, 16],
                tcgen05_cluster_m=2,
                pid_type="persistent_blocked",
            )
        )
        self.assertNotIn("cute.nvgpu.tcgen05.CtaGroup.TWO", code)

    def test_matmul_mma_tcgen05_fp8_deep_ab_staging_6(self) -> None:
        """Test FP8 with ab_stages=6 (mid-depth staging)."""
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")

        torch.manual_seed(0)
        x = (torch.randn(512, 1024, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(1024, 1024, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        # cluster_m=2 requires a persistent pid_type; block_m=256 engages the
        # validated two-CTA role-local path.
        code, out = code_and_output(
            cute_matmul_mma_fp8,
            (x, y),
            block_sizes=[256, 128, 64],
            tcgen05_ab_stages=6,
            tcgen05_cluster_m=2,
            pid_type="persistent_blocked",
        )
        ref = x.float() @ y.float()
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=1e-1)
        # Verify deep staging config is in generated code
        self.assertIn("cutlass.Float8E4M3FN", code)
        self.assertIn("cute.nvgpu.tcgen05", code)

    def test_matmul_mma_tcgen05_fp8_deep_ab_staging_8(self) -> None:
        """Test FP8 with ab_stages=8 (sweet spot from benchmarks)."""
        support = get_cute_mma_support()
        if not support.tcgen05_f8:
            self.skipTest("tcgen05 FP8 MMA is not supported on this machine")

        torch.manual_seed(0)
        x = (torch.randn(256, 1024, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        y = (torch.randn(1024, 1024, device=DEVICE) * 0.4).to(torch.float8_e4m3fn)
        # cluster_m=2 requires a persistent pid_type; block_m=256 engages the
        # validated two-CTA role-local path.
        code, out = code_and_output(
            cute_matmul_mma_fp8,
            (x, y),
            block_sizes=[256, 128, 64],
            tcgen05_ab_stages=8,
            tcgen05_cluster_m=2,
            pid_type="persistent_blocked",
        )
        ref = x.float() @ y.float()
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=1e-1)
        # Verify deep staging is used
        self.assertIn("cutlass.Float8E4M3FN", code)
        self.assertIn("cute.nvgpu.tcgen05", code)

    def test_matmul_dot_out_dtype_falls_back_from_mma(self) -> None:
        args = (
            torch.randn(16, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 8, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(
            cute_matmul_dot_out_dtype, args, block_sizes=[16, 8, 16]
        )
        x, y = args
        expected = (x[:, :, None] * y[None, :, :]).to(torch.float32).sum(dim=1)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
        self.assertNotIn("cute.gemm", code)
        self.assertNotIn("cute.nvgpu.MmaUniversalOp", code)

    def test_matmul_packed_rhs_bfloat16(self) -> None:
        m, k, n = 32, 64, 32
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(k // 2, n, device=DEVICE, dtype=torch.bfloat16)
        c = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)

        code, _ = code_and_output(cute_matmul_packed_rhs_bfloat16, (a, b, c))
        b_unpacked = torch.stack([b, b], dim=1).reshape(k, n)
        expected = a @ b_unpacked

        torch.testing.assert_close(c, expected, atol=2e-1, rtol=2e-2)

    def test_matmul_mma_preserves_incoming_accumulator(self) -> None:
        args = (
            torch.randn(16, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 8, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(16, 8, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(
            cute_matmul_mma_with_bias_acc,
            args,
            block_sizes=[16, 8, 16],
        )
        x, y, bias = args
        expected = x.float() @ y.float() + bias
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)
        self.assertIn("cute.gemm", code)
        self.assertNotIn("cute.arch.warp_reduction_sum", code)

    def test_addmm_rejects_alpha_beta_kwargs(self) -> None:
        @helion.kernel(backend="cute")
        def cute_addmm_alpha_beta(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            m, k = x.size()
            _, n = y.size()
            out = torch.empty([m, n], dtype=bias.dtype, device=x.device)
            for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
                out[tile_m, tile_n] = torch.addmm(
                    bias[tile_m, tile_n],
                    x[tile_m, tile_k],
                    y[tile_k, tile_n],
                    beta=0.5,
                    alpha=2.0,
                )
            return out

        args = (
            torch.randn(16, 16, device=DEVICE, dtype=torch.float32),
            torch.randn(16, 16, device=DEVICE, dtype=torch.float32),
            torch.randn(16, 16, device=DEVICE, dtype=torch.float32),
        )
        with self.assertRaises(AssertionError):
            code_and_output(cute_addmm_alpha_beta, args, block_sizes=[16, 16, 16])

    def test_matmul_mma_mixed_loop_falls_back_cleanly(self) -> None:
        args = (
            torch.randn(16, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 8, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(
            cute_matmul_mma_mixed_k_loop,
            args,
            block_sizes=[16, 8, 16],
        )
        x, y = args
        extra = x.float().sum(dim=1, keepdim=True).expand(-1, y.size(1))
        expected = x.float() @ y.float() + extra
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)
        self.assertNotIn("cute.gemm", code)

    def test_matmul_mma_with_lane_loops(self) -> None:
        args = (
            torch.randn(32, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 16, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(
            cute_matmul_mma,
            args,
            block_sizes=[32, 16, 16],
            num_threads=[16, 8, 1],
        )
        torch.testing.assert_close(out, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertNotIn("cute.gemm", code)

    def test_batched_baddbmm_bias_acc_init_falls_back(self) -> None:
        # A nonzero incoming accumulator stays on the scalar path: tcgen05
        # cannot seed its fragment from the existing per-element value.
        args = (
            torch.randn(2, 16, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 64, 8, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 16, 8, device=DEVICE, dtype=torch.float32),
        )
        self.assertFalse(
            cute_baddbmm.bind(args).config_spec.cute_tcgen05_search_enabled
        )
        code, out = code_and_output(
            cute_baddbmm,
            args,
            block_sizes=[1, 16, 8, 16],
            num_threads=[1, 16, 8, 1],
        )
        x, y, bias = args
        expected = torch.baddbmm(bias, x.float(), y.float())
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)
        self.assertNotIn("cute.gemm", code)

    def test_batched_nonzero_acc_does_not_plan_persistent_tcgen05(self) -> None:
        args = (
            torch.randn(2, 256, 128, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 128, 256, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 256, 256, device=DEVICE, dtype=torch.float32),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            bound = cute_baddbmm.bind(args)
            self.assertFalse(bound.config_spec.cute_tcgen05_search_enabled)
            code = bound.to_triton_code(
                helion.Config(
                    block_sizes=[1, 256, 256, 64],
                    pid_type="persistent_blocked",
                )
            )
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                "tcgen05_cluster_m",
            ):
                bound.to_triton_code(_batched_tcgen05_two_cta_config())
        self.assertNotIn("CtaGroup.TWO", code)

    def test_matmul_mma_non_divisible(self) -> None:
        """Test MMA with non-divisible matrix dimensions (masking)."""
        args = (
            torch.randn(13, 37, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(37, 7, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(cute_matmul_mma, args, block_sizes=[16, 8, 16])
        torch.testing.assert_close(out, args[0] @ args[1], atol=1e-1, rtol=1e-2)
        self.assertIn("cute.gemm", code)
        self.assertIn("cute.nvgpu.warp.MmaF16BF16Op", code)
        self.assertNotIn("cute.arch.warp_reduction_sum", code)

    def test_matmul_addmm(self) -> None:
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(
            cute_matmul_addmm,
            args,
            block_sizes=[4, 4, 16],
            num_threads=[4, 4, 1],
        )
        torch.testing.assert_close(out, args[0] @ args[1], atol=1e-1, rtol=1e-2)

    def test_matmul_direct_full_k_tile_falls_back_correctly(self) -> None:
        args = (
            torch.randn(4, 4, device=DEVICE, dtype=torch.float32),
            torch.randn(4, 4, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(
            cute_matmul_direct,
            args,
            block_sizes=[1, 1, 4],
            num_threads=[1, 1, 4],
        )
        torch.testing.assert_close(out, args[0] @ args[1], atol=1e-5, rtol=1e-5)
        self.assertIn("cute.arch.warp_reduction_sum", code)
        self.assertNotIn("cute.gemm", code)

    def test_direct_shared_tree_sum_matches_matmul_lane_mapping(self) -> None:
        lhs = torch.randn(3, 16, device=DEVICE, dtype=torch.float32)
        rhs = torch.randn(16, 1, device=DEVICE, dtype=torch.float32)
        out = torch.empty(3, 1, device=DEVICE, dtype=torch.float32)
        default_cute_launcher(
            cute_shared_tree_matmul_sum, (1,), lhs, rhs, out, block=(3, 16, 1)
        )
        torch.testing.assert_close(out, lhs @ rhs, atol=1e-5, rtol=1e-5)

    def test_addmm_direct_full_k_tile_falls_back_correctly(self) -> None:
        args = (
            torch.randn(4, 4, device=DEVICE, dtype=torch.float32),
            torch.randn(4, 4, device=DEVICE, dtype=torch.float32),
            torch.randn(4, 4, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(
            cute_matmul_addmm_shifted_direct,
            args,
            block_sizes=[1, 1, 4],
            num_threads=[1, 1, 4],
        )
        x, y, bias = args
        expected = torch.addmm(bias, x + 1, y + 1)
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
        self.assertIn("cute.arch.warp_reduction_sum", code)
        self.assertNotIn("cute.gemm", code)

    def test_matmul_addmm_shifted_operands_falls_back_cleanly(self) -> None:
        args = (
            torch.randn(32, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 32, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(
            cute_matmul_addmm_shifted_operands,
            args,
            block_sizes=[16, 16, 16],
        )
        x, y = args
        expected = (x.cpu().float() + 1) @ (y.cpu().float() + 1)
        torch.testing.assert_close(out.cpu(), expected, atol=1e-1, rtol=1e-2)
        self.assertNotIn("cute.gemm", code)

    def test_nested_grid_addmm_falls_back_correctly(self) -> None:
        torch.manual_seed(0)
        args = (
            torch.randn(16, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(64, 8, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(
            cute_nested_grid_addmm,
            args,
            block_sizes=[16, 8, 16],
            num_threads=[1, 1, 4],
        )
        expected = args[0].float() @ args[1].float()
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)
        self.assertNotIn("cute.gemm", code)

    def test_addmm_same_iteration_consumer_falls_back_cleanly(self) -> None:
        args = (
            torch.randn(16, 1, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(1, 8, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(
            cute_addmm_same_iteration_relu_consumer,
            args,
            block_sizes=[16, 8, 1],
        )
        expected = torch.relu(args[0].float() @ args[1].float())
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)
        self.assertNotIn("cute.gemm", code)

    def test_matmul_direct_grouped_n_uses_mma(self) -> None:
        @helion.kernel(
            backend="cute",
            config=helion.Config(block_sizes=[32], indexing="block_ptr"),
            static_shapes=True,
        )
        def grouped_n_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, _n = x.size()
            out = torch.empty([m, y.size(1)], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] @ y[:, :]
            return out

        args = (
            torch.randn(256, 128, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(128, 128, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(grouped_n_matmul, args)
        expected = args[0].float() @ args[1].float()
        torch.testing.assert_close(out, expected.to(out.dtype), atol=1e-1, rtol=1e-2)
        self.assertIn("cute.gemm", code)
        self.assertIn("cute.nvgpu.warp.MmaF16BF16Op", code)
        self.assertNotIn("dot_serial_result", code)

    def test_matmul_direct_grouped_n_slice_operands_use_mma(self) -> None:
        @helion.kernel(
            backend="cute",
            config=helion.Config(block_sizes=[32], indexing="block_ptr"),
            static_shapes=True,
        )
        def grouped_n_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, _n = x.size()
            out = torch.empty([m, 128], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, 16:144] @ y[16:144, :]
            return out

        args = (
            torch.randn(256, 160, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(160, 128, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(grouped_n_matmul, args)
        expected = args[0][:, 16:144].float() @ args[1][16:144, :].float()
        torch.testing.assert_close(out, expected.to(out.dtype), atol=1e-1, rtol=1e-2)
        self.assertIn("cute.gemm", code)
        self.assertNotIn("dot_serial_result", code)

    def test_matmul_direct_grouped_n_rhs_offset_uses_mma(self) -> None:
        @helion.kernel(
            backend="cute",
            config=helion.Config(block_sizes=[32], indexing="block_ptr"),
            static_shapes=True,
        )
        def grouped_n_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, _n = x.size()
            out = torch.empty([m, 128], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] @ y[:, 16:144]
            return out

        args = (
            torch.randn(256, 128, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(128, 160, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(grouped_n_matmul, args)
        expected = args[0].float() @ args[1][:, 16:144].float()
        torch.testing.assert_close(out, expected.to(out.dtype), atol=1e-1, rtol=1e-2)
        self.assertIn("cute.gemm", code)
        self.assertNotIn("dot_serial_result", code)

    def test_matmul_direct_grouped_n_noncontiguous_operands_reject_cleanly(
        self,
    ) -> None:
        @helion.kernel(
            backend="cute",
            config=helion.Config(block_sizes=[32], indexing="block_ptr"),
            static_shapes=True,
        )
        def grouped_n_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, _n = x.size()
            out = torch.empty([m, 64], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, 16:144:2] @ y[16:144:2, :]
            return out

        args = (
            torch.randn(256, 160, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(160, 64, device=DEVICE, dtype=HALF_DTYPE),
        )
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            "strided slices .* are not supported",
        ):
            code_and_output(grouped_n_matmul, args)

    def test_matmul_direct_grouped_n_negative_rhs_offset_rejects_cleanly(
        self,
    ) -> None:
        @helion.kernel(
            backend="cute",
            config=helion.Config(block_sizes=[32], indexing="block_ptr"),
            static_shapes=True,
        )
        def grouped_n_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, _n = x.size()
            out = torch.empty([m, 128], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] @ y[:, -144:-16]
            return out

        args = (
            torch.randn(256, 128, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(128, 160, device=DEVICE, dtype=HALF_DTYPE),
        )
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            "CuTe direct mm without an active K tile only supports contiguous direct-load operands",
        ):
            code_and_output(grouped_n_matmul, args)

    def test_matmul_direct_grouped_n_multiple_mms_fall_back_cleanly(self) -> None:
        @helion.kernel(
            backend="cute",
            config=helion.Config(block_sizes=[32], indexing="block_ptr"),
            static_shapes=True,
        )
        def grouped_n_two_matmuls(
            x1: torch.Tensor,
            y1: torch.Tensor,
            x2: torch.Tensor,
            y2: torch.Tensor,
        ) -> torch.Tensor:
            m, _n = x1.size()
            out = torch.empty([m, 128], dtype=x1.dtype, device=x1.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x1[tile_m, 16:144] @ y1[16:144, :]
                out[tile_m, :] += x2[tile_m, 16:144] @ y2[16:144, :]
            return out

        args = (
            torch.randn(256, 160, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(160, 128, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(256, 160, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(160, 128, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(grouped_n_two_matmuls, args)
        expected = (
            args[0][:, 16:144].float() @ args[1][16:144, :].float()
            + args[2][:, 16:144].float() @ args[3][16:144, :].float()
        )
        torch.testing.assert_close(out, expected.to(out.dtype), atol=1e-1, rtol=1e-2)
        self.assertNotIn("cute.nvgpu.warp.MmaF16BF16Op", code)
        self.assertIn("dot_serial_result", code)

    def test_matmul_direct_grouped_n_respects_mma_override(self) -> None:
        @helion.kernel(
            backend="cute",
            config=helion.Config(block_sizes=[32], indexing="block_ptr"),
            static_shapes=True,
        )
        def grouped_n_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, _n = x.size()
            out = torch.empty([m, y.size(1)], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] @ y[:, :]
            return out

        args = (
            torch.randn(256, 128, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(128, 128, device=DEVICE, dtype=HALF_DTYPE),
        )
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "universal"}, clear=False):
            code, out = code_and_output(grouped_n_matmul, args)
        expected = args[0].float() @ args[1].float()
        torch.testing.assert_close(out, expected.to(out.dtype), atol=1e-1, rtol=1e-2)
        self.assertNotIn("cute.gemm", code)

    def test_matmul_direct_grouped_n_mismatched_threads_falls_back(self) -> None:
        @helion.kernel(
            backend="cute",
            config=helion.Config(
                block_sizes=[64],
                num_threads=[32],
                indexing="block_ptr",
            ),
            static_shapes=True,
        )
        def grouped_n_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, _n = x.size()
            out = torch.empty([m, y.size(1)], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m, :] = x[tile_m, :] @ y[:, :]
            return out

        args = (
            torch.randn(256, 128, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(128, 128, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(grouped_n_matmul, args)
        expected = args[0].float() @ args[1].float()
        torch.testing.assert_close(out, expected.to(out.dtype), atol=1e-1, rtol=1e-2)
        self.assertNotIn("cute.gemm", code)

    def test_dot_acc_dynamic_shape_uses_mma(self) -> None:
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(64, 64, device=DEVICE, dtype=torch.bfloat16),
        )
        cute_dot_acc_dynamic_bf16.settings.static_shapes = False
        cute_dot_acc_dynamic_bf16.reset()
        code, out = code_and_output(
            cute_dot_acc_dynamic_bf16,
            args,
            block_sizes=[16, 16, 16],
        )
        expected = args[0].float() @ args[1].float()
        torch.testing.assert_close(out, expected, atol=1e-1, rtol=1e-2)
        self.assertNotIn("cute.arch.warp_reduction_sum", code)
        self.assertNotIn("cute.gemm", code)

    def test_cute_dsl_arch_env_tracks_launch_device(self) -> None:
        tensor = torch.empty(1, device=DEVICE)
        major, minor = torch.cuda.get_device_capability(tensor.device)
        suffix = "a" if major >= 9 else ""
        expected = f"sm_{major}{minor}{suffix}"
        with patch.dict(os.environ, {"CUTE_DSL_ARCH": "sm_00"}, clear=False):
            _ensure_cute_dsl_arch_env((tensor,))
            self.assertEqual(os.environ["CUTE_DSL_ARCH"], expected)

    def test_cute_launcher_cache_key_includes_compile_options(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        schema_key = (("tensor", 2, "float32"),)
        block = (32, 1, 1)
        created: list[str] = []

        def make_wrapper(*_args: object, **_kwargs: object) -> str:
            created.append("wrapper")
            return f"wrapper-{len(created)}"

        with patch(
            "helion.runtime.cute.launcher._create_cute_wrapper",
            side_effect=make_wrapper,
        ):
            wrapper_default = _get_compiled_cute_launcher(
                cute_kernel,
                schema_key,
                block,
            )
            wrapper_lineinfo = _get_compiled_cute_launcher(
                cute_kernel,
                schema_key,
                block,
                compile_options="--generate-line-info",
            )

        self.assertNotEqual(wrapper_default, wrapper_lineinfo)

    def test_cute_launcher_reuses_compiled_wrapper(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        schema_key = (("tensor", 2, "float32"),)
        block = (32, 1, 1)
        compiled_calls: list[tuple[object, tuple[object, ...], str | None]] = []
        launched_args: list[tuple[object, ...]] = []

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                launched_args.append(args)
                return ("launched", args)

        def fake_compile(
            jit_func: object,
            *args: object,
            options: str | None = None,
        ) -> FakeCompiled:
            compiled_calls.append((jit_func, args, options))
            return FakeCompiled()

        with (
            patch(
                "helion.runtime.cute.launcher._create_cute_wrapper",
                return_value="jit-wrapper",
            ),
            patch("cutlass.cute.compile", side_effect=fake_compile),
        ):
            launcher = _get_compiled_cute_launcher(cute_kernel, schema_key, block)
            first = launcher(1, 2, 3)
            second = launcher(4, 5, 6)

        self.assertEqual(
            compiled_calls, [("jit-wrapper", (1, 2, 3), "--enable-tvm-ffi")]
        )
        self.assertEqual(launched_args, [(1, 2, 3), (4, 5, 6)])
        self.assertEqual(first, ("launched", (1, 2, 3)))
        self.assertEqual(second, ("launched", (4, 5, 6)))

    def test_cute_launcher_passes_compile_options(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        schema_key = (("tensor", 2, "float32"),)
        block = (32, 1, 1)
        compiled_calls: list[tuple[object, tuple[object, ...], str | None]] = []

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                return ("launched", args)

        def fake_compile(
            jit_func: object,
            *args: object,
            options: str | None = None,
        ) -> FakeCompiled:
            compiled_calls.append((jit_func, args, options))
            return FakeCompiled()

        with (
            patch(
                "helion.runtime.cute.launcher._create_cute_wrapper",
                return_value="jit-wrapper",
            ),
            patch("cutlass.cute.compile", side_effect=fake_compile),
        ):
            launcher = _get_compiled_cute_launcher(
                cute_kernel,
                schema_key,
                block,
                compile_options="--generate-line-info",
            )
            result = launcher(1, 2, 3)

        # The runtime merges ``--enable-tvm-ffi`` into any caller-provided
        # compile_options so the generic launcher always benefits from
        # the FFI bridge (e.g. when the autotuner selects
        # ``tcgen05_cubin_lineinfo=True``).
        self.assertEqual(
            compiled_calls,
            [("jit-wrapper", (1, 2, 3), "--generate-line-info --enable-tvm-ffi")],
        )
        self.assertEqual(result, ("launched", (1, 2, 3)))

    def test_cute_launcher_reuses_launch_args_for_stable_scalar_signature(
        self,
    ) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        build_calls: list[tuple[tuple[object, ...], tuple[int, int, int]]] = []
        launched_args: list[tuple[object, ...]] = []

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                launched_args.append(args)
                return ("launched", args)

        def fake_build(
            _cute_kernel: object,
            args: tuple[object, ...],
            grid: tuple[int, int, int],
        ) -> helion_runtime._CuteLaunchArgCacheEntry:
            build_calls.append((args, grid))
            return _launch_entry((("scalar", "int"),), ("launch-arg", *args, *grid))

        with (
            patch(
                "helion.runtime.cute.launcher._build_cute_schema_and_args",
                side_effect=fake_build,
            ),
            patch(
                "helion.runtime.cute.launcher._cute_current_stream",
                return_value="stream",
            ),
            patch(
                "helion.runtime.cute.launcher._get_compiled_cute_launcher",
                return_value=FakeCompiled(),
            ),
        ):
            first = default_cute_launcher(cute_kernel, (2,), 7, block=(32, 1, 1))
            second = default_cute_launcher(cute_kernel, (2,), 7, block=(32, 1, 1))
            third = default_cute_launcher(cute_kernel, (2,), 8, block=(32, 1, 1))

        self.assertEqual(build_calls, [((7,), (2, 1, 1)), ((8,), (2, 1, 1))])
        # The stream is appended fresh per launch (never cached), so it trails
        # the cached launch args on every call.
        self.assertEqual(
            launched_args,
            [
                ("launch-arg", 7, 2, 1, 1, "stream"),
                ("launch-arg", 7, 2, 1, 1, "stream"),
                ("launch-arg", 8, 2, 1, 1, "stream"),
            ],
        )
        self.assertEqual(first, ("launched", ("launch-arg", 7, 2, 1, 1, "stream")))
        self.assertEqual(second, first)
        self.assertEqual(third, ("launched", ("launch-arg", 8, 2, 1, 1, "stream")))

    def test_cute_launcher_samples_stream_fresh_per_launch(self) -> None:
        # The CUDA stream must NOT be cached: on a launch-arg cache HIT the
        # stream still has to be re-sampled, otherwise a kernel launched during
        # CUDA graph capture would run on a stale (eager) stream and the graph
        # would capture no work (empty-graph / no-op replay). Here the build is
        # cached after the first call (same signature), but the current stream
        # changes between launches and must be reflected each time.
        cute_kernel = type("DummyCuteKernel", (), {})()
        build_calls: list[tuple[object, ...]] = []
        launched_args: list[tuple[object, ...]] = []
        streams = iter(["stream-A", "stream-B", "stream-C"])
        owned_tensors = (torch.empty(1),)

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                launched_args.append(args)
                return ("launched", args)

        def fake_build(
            _cute_kernel: object,
            args: tuple[object, ...],
            _grid: tuple[int, int, int],
        ) -> helion_runtime._CuteLaunchArgCacheEntry:
            build_calls.append(args)
            # Note: no stream baked into the returned launch args.
            return _launch_entry((("scalar", "int"),), ("launch-arg",), owned_tensors)

        with (
            patch(
                "helion.runtime.cute.launcher._build_cute_schema_and_args",
                side_effect=fake_build,
            ),
            patch(
                "helion.runtime.cute.launcher._cute_current_stream",
                side_effect=lambda: next(streams),
            ),
            patch(
                "helion.runtime.cute.launcher._get_compiled_cute_launcher",
                return_value=FakeCompiled(),
            ),
            patch(
                "helion.runtime.cute.launcher._record_cute_owned_launch_tensors"
            ) as record_owned,
        ):
            first = default_cute_launcher(cute_kernel, (1,), 7, block=(32, 1, 1))
            second = default_cute_launcher(cute_kernel, (1,), 7, block=(32, 1, 1))
            third = default_cute_launcher(cute_kernel, (1,), 7, block=(32, 1, 1))

        # Build (and thus the cached args) happens once; the stream is appended
        # fresh on each of the three launches.
        self.assertEqual(build_calls, [(7,)])
        self.assertEqual(record_owned.call_count, 3)
        record_owned.assert_called_with(owned_tensors)
        self.assertEqual(
            launched_args,
            [
                ("launch-arg", "stream-A"),
                ("launch-arg", "stream-B"),
                ("launch-arg", "stream-C"),
            ],
        )
        self.assertEqual(first, ("launched", ("launch-arg", "stream-A")))
        self.assertEqual(second, ("launched", ("launch-arg", "stream-B")))
        self.assertEqual(third, ("launched", ("launch-arg", "stream-C")))

    def test_cute_owned_launch_tensor_survives_cross_stream_lru_eviction(
        self,
    ) -> None:
        if DEVICE.type != "cuda":
            self.skipTest("CuTe launch ownership test needs CUDA")
        cuda_runtime = importlib.import_module("cuda.bindings.runtime")
        producer_stream = torch.cuda.Stream(device=DEVICE)
        launch_stream = torch.cuda.Stream(device=DEVICE)
        tensor_size = 4 * 1024 * 1024

        with torch.cuda.stream(producer_stream):
            owned = torch.ones(tensor_size, dtype=torch.uint8, device=DEVICE)
        producer_stream.synchronize()
        owned_ref = weakref.ref(owned)
        output = torch.empty_like(owned)
        launch = _launch_entry((), (), (owned,))

        with torch.cuda.stream(launch_stream):
            # Keep the raw-pointer read pending while the launch-argument entry
            # is evicted. Unlike a torch op, this memcpy does not retain or
            # register the source tensor with the caching allocator.
            torch.cuda._sleep(500_000_000)
            copy_result = cuda_runtime.cudaMemcpyAsync(
                output.data_ptr(),
                owned.data_ptr(),
                tensor_size,
                cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                cuda_runtime.cudaStream_t(launch_stream.cuda_stream),
            )
            self.assertEqual(copy_result[0], cuda_runtime.cudaError_t.cudaSuccess)
            helion_runtime._record_cute_owned_launch_tensors(launch.owned_tensors)

        cache = {0: launch}
        cache.update(
            (index, _launch_entry((), ()))
            for index in range(1, helion_runtime._CUTE_LAUNCH_ARG_CACHE_LIMIT + 1)
        )
        cache.pop(next(iter(cache)))
        self.assertNotIn(0, cache)
        del launch, owned
        gc.collect()
        self.assertIsNone(owned_ref())

        with torch.cuda.stream(producer_stream):
            replacement = torch.full(
                (tensor_size,), 7, dtype=torch.uint8, device=DEVICE
            )
        producer_stream.synchronize()
        launch_stream.synchronize()

        self.assertTrue(torch.all(replacement == 7).item())
        self.assertTrue(torch.all(output == 1).item())

    def test_grouped_dynamic_tensormap_workspace_isolated_by_stream(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("dynamic TensorMap workspace test needs CUDA")
        runtime_mod = importlib.import_module("helion.runtime")
        cute_kernel = type("DummyCuteKernel", (), {})()
        cute_kernel._helion_cute_wrapper_plans = [
            {
                "kind": "tcgen05_grouped_static_persistent",
                "layout_idx": 0,
                "dynamic_ab_tensormaps": True,
            }
        ]
        layout = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        streams = [
            torch.cuda.Stream(device=DEVICE)
            for _ in range(
                runtime_mod._TCGEN05_DYNAMIC_TENSORMAP_WORKSPACE_CACHE_LIMIT + 1
            )
        ]
        stream_a, stream_b = streams[:2]

        with torch.cuda.stream(stream_a):
            context_a = runtime_mod._cute_dynamic_tensormap_contexts(
                cute_kernel, (layout,)
            )
            workspace_a = runtime_mod._tcgen05_grouped_dynamic_tensormap_workspace(
                cute_kernel, device=layout.device, tensormap_count=2
            )
            workspace_a_again = (
                runtime_mod._tcgen05_grouped_dynamic_tensormap_workspace(
                    cute_kernel, device=layout.device, tensormap_count=2
                )
            )
        with torch.cuda.stream(stream_b):
            context_b = runtime_mod._cute_dynamic_tensormap_contexts(
                cute_kernel, (layout,)
            )
            workspace_b = runtime_mod._tcgen05_grouped_dynamic_tensormap_workspace(
                cute_kernel, device=layout.device, tensormap_count=2
            )

        self.assertIs(workspace_a_again, workspace_a)
        self.assertIsNot(workspace_b, workspace_a)
        self.assertNotEqual(workspace_b.data_ptr(), workspace_a.data_ptr())
        self.assertNotEqual(context_a, context_b)
        self.assertIsNone(context_a[0][3])
        self.assertIsNone(context_b[0][3])

        for stream in streams[2:]:
            with torch.cuda.stream(stream):
                runtime_mod._tcgen05_grouped_dynamic_tensormap_workspace(
                    cute_kernel, device=layout.device, tensormap_count=2
                )
        cache = cute_kernel._helion_tcgen05_dynamic_tensormap_workspace_cache
        self.assertEqual(
            len(cache), runtime_mod._TCGEN05_DYNAMIC_TENSORMAP_WORKSPACE_CACHE_LIMIT
        )
        self.assertFalse(any(cached is workspace_a for cached in cache.values()))
        self.assertTrue(any(cached is workspace_b for cached in cache.values()))

    def test_grouped_static_capture_query_fails_closed(self) -> None:
        if DEVICE.type != "cuda":
            self.skipTest("grouped capture context test needs CUDA")
        cute_kernel = type("DummyCuteKernel", (), {})()
        cute_kernel._helion_cute_wrapper_plans = [
            {"kind": "tcgen05_grouped_static_persistent", "layout_idx": 0}
        ]
        layout = torch.zeros(1, dtype=torch.int32, device=DEVICE)

        with (
            patch(
                "helion.runtime.cute.launcher._cuda_stream_capture_context",
                side_effect=BackendUnsupported("cute", "capture query failed"),
            ),
            self.assertRaisesRegex(BackendUnsupported, "capture query failed"),
        ):
            helion_runtime._cute_grouped_launch_contexts(cute_kernel, (layout,))

    def test_grouped_dynamic_tensormap_workspace_retained_per_capture(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("dynamic TensorMap workspace test needs CUDA")
        runtime_mod = importlib.import_module("helion.runtime")
        cute_kernel = type("DummyCuteKernel", (), {})()
        cute_kernel._helion_cute_wrapper_plans = [
            {
                "kind": "tcgen05_grouped_static_persistent",
                "layout_idx": 0,
                "dynamic_ab_tensormaps": True,
            }
        ]
        layout = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        args = (layout,)
        grid = (1, 1, 1)
        capture_stream = torch.cuda.Stream(device=DEVICE)
        graphs: list[torch.cuda.CUDAGraph] = []
        contexts: list[tuple[int, int | None]] = []
        workspaces: list[torch.Tensor] = []
        launch_keys: list[tuple[object, ...]] = []
        guards: list[Any] = []
        replay_aliases: list[Callable[[], None]] = []

        runtime_mod.get_num_sm(DEVICE)
        runtime_mod._cuda_stream_capture_context(DEVICE)
        torch.empty(1, device=DEVICE).fill_(0)
        torch.cuda.synchronize(DEVICE)
        for value in range(1, 10):
            with runtime_mod.cute_cuda_graph(stream=capture_stream) as graph:
                replay_alias = graph.replay
                context = runtime_mod._cuda_stream_capture_context(DEVICE)
                launch_key = runtime_mod._cute_launch_arg_cache_key(
                    cute_kernel, args, grid
                )
                if guards:
                    self.assertFalse(guards[-1].matches(cute_kernel, args, grid))
                guard = runtime_mod._cute_last_launch_arg_guard(cute_kernel, args, grid)
                self.assertTrue(guard.matches(cute_kernel, args, grid))
                workspace = runtime_mod._tcgen05_grouped_dynamic_tensormap_workspace(
                    cute_kernel, device=DEVICE, tensormap_count=2
                )
                workspace.fill_(value)
            graphs.append(graph)
            contexts.append(context)
            workspaces.append(workspace)
            launch_keys.append(launch_key)
            guards.append(guard)
            replay_aliases.append(replay_alias)

        self.assertEqual({context[0] for context in contexts}, {contexts[0][0]})
        self.assertNotIn(None, {context[1] for context in contexts})
        self.assertEqual(len({context[1] for context in contexts}), len(contexts))
        self.assertEqual(len({workspace.data_ptr() for workspace in workspaces}), 9)
        self.assertEqual(len(set(launch_keys)), 9)
        cache = cute_kernel._helion_tcgen05_dynamic_tensormap_workspace_cache
        self.assertEqual(len(cache), 9)
        for workspace in workspaces:
            self.assertTrue(any(cached is workspace for cached in cache.values()))

        replay_aliases[0]()
        replay_aliases[-1]()
        torch.cuda.synchronize(DEVICE)
        self.assertEqual(workspaces[0].flatten()[0].item(), 1)
        self.assertEqual(workspaces[-1].flatten()[0].item(), 9)

        replay_alias = replay_aliases.pop(0)
        first_graph = graphs.pop(0)
        graph_refs = [
            weakref.ref(first_graph),
            *(weakref.ref(graph) for graph in graphs),
        ]
        del first_graph
        gc.collect()
        self.assertIsNotNone(graph_refs[0]())
        self.assertEqual(len(cache), 9)
        replay_alias()
        torch.cuda.synchronize(DEVICE)
        self.assertEqual(workspaces[0].flatten()[0].item(), 1)

        graphs.clear()
        replay_aliases.clear()
        del graph, replay_alias
        gc.collect()
        self.assertTrue(all(graph_ref() is None for graph_ref in graph_refs))
        self.assertEqual(cache, {})

    def test_grouped_dynamic_tensormap_context_guards_launcher_caches(
        self,
    ) -> None:
        runtime_mod = importlib.import_module("helion.runtime")
        cute_kernel = type("DummyCuteKernel", (), {})()
        context_a = (("cuda", 0, 101, None),)
        context_b = (("cuda", 0, 202, 7),)
        current_context: list[tuple[tuple[str, int | None, int, int | None], ...]] = [
            context_a
        ]
        build_calls: list[tuple[tuple[str, int | None, int, int | None], ...]] = []
        launched_args: list[tuple[object, ...]] = []
        streams = iter(["stream-A", "stream-B", "stream-C"])
        generic_matcher = runtime_mod._cute_last_launch_cache_entry

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                launched_args.append(args)
                return ("launched", args)

        def fake_contexts(
            _cute_kernel: object,
            _args: tuple[object, ...],
        ) -> tuple[tuple[str, int | None, int, int | None], ...]:
            return current_context[0]

        def fake_build(
            _cute_kernel: object,
            _args: tuple[object, ...],
            _grid: tuple[int, int, int],
        ) -> helion_runtime._CuteLaunchArgCacheEntry:
            build_calls.append(current_context[0])
            return _launch_entry((("scalar", "int"),), (current_context[0],))

        with (
            patch(
                "helion.runtime.cute.launcher._cute_dynamic_tensormap_contexts",
                side_effect=fake_contexts,
            ),
            patch(
                "helion.runtime.cute.launcher._build_cute_schema_and_args",
                side_effect=fake_build,
            ),
            patch(
                "helion.runtime.cute.launcher._cute_current_stream",
                side_effect=lambda: next(streams),
            ),
            patch(
                "helion.runtime.cute.launcher._get_compiled_cute_launcher",
                return_value=FakeCompiled(),
            ),
            patch(
                "helion.runtime.cute.launcher._cute_last_launch_cache_entry",
                wraps=generic_matcher,
            ) as generic_match,
        ):
            first = default_cute_launcher(cute_kernel, (1,), 7)
            current_context[0] = context_b
            second = default_cute_launcher(cute_kernel, (1,), 7)
            third = default_cute_launcher(cute_kernel, (1,), 7)

        self.assertEqual(build_calls, [context_a, context_b])
        self.assertEqual(generic_match.call_count, 3)
        self.assertEqual(
            launched_args,
            [
                (context_a, "stream-A"),
                (context_b, "stream-B"),
                (context_b, "stream-C"),
            ],
        )
        self.assertEqual(first, ("launched", (context_a, "stream-A")))
        self.assertEqual(second, ("launched", (context_b, "stream-B")))
        self.assertEqual(third, ("launched", (context_b, "stream-C")))

    def test_grouped_static_metadata_match_filters_device_plans(self) -> None:
        launcher = importlib.import_module("helion.runtime.cute.launcher")
        host_plan = {"kind": "tcgen05_grouped_static_persistent"}
        device_plan = {
            "kind": "tcgen05_grouped_static_persistent",
            "device_split_sizes": True,
        }
        unrelated_plan = {"kind": "unrelated"}
        cute_kernel = type("DummyCuteKernel", (), {})()

        class Entry:
            has_m_tail = False
            has_n_tail = False

            def __init__(self, matches: bool = True) -> None:
                self._matches = matches

            def matches(self, *_args: object) -> bool:
                return self._matches

        def metadata_matches(entries: tuple[Entry, ...]) -> bool:
            return launcher._cute_grouped_static_metadata_matches(
                entries,
                cute_kernel,
                (),
            )

        with (
            patch.object(
                launcher,
                "_tcgen05_grouped_static_layout_arg",
                return_value=object(),
            ),
            patch.object(
                launcher,
                "_tcgen05_grouped_static_size_arg",
                return_value=None,
            ),
        ):
            for plans in ([], [unrelated_plan], [device_plan]):
                cute_kernel._helion_cute_wrapper_plans = plans
                self.assertTrue(metadata_matches(()))

            for plans in (
                [host_plan, device_plan],
                [device_plan, unrelated_plan, host_plan],
            ):
                cute_kernel._helion_cute_wrapper_plans = plans
                self.assertTrue(metadata_matches((Entry(),)))
                self.assertFalse(metadata_matches(()))
                self.assertFalse(metadata_matches((Entry(matches=False),)))

    def test_grouped_capture_retains_generated_tensors_beyond_launch_lru(
        self,
    ) -> None:
        if not torch.cuda.is_available():
            self.skipTest("grouped capture ownership test needs CUDA")
        runtime_mod = importlib.import_module("helion.runtime")
        cute_kernel = type("DummyCuteKernel", (), {})()
        plan = {
            "kind": "tcgen05_grouped_static_persistent",
            "layout_idx": 0,
            "n_sizes_idx": 1,
            "group_count": 2,
            "bm": 128,
            "bn": 64,
            "bk": 128,
            "n_size": 256,
            "k_total_size": 128,
            "problem_sizes_arg": "tcgen05_grouped_problem_sizes_0",
            "starts_arg": "tcgen05_grouped_starts_0",
            "total_clusters_arg": "tcgen05_grouped_total_clusters_0",
        }
        cute_kernel._helion_cute_wrapper_plans = [plan]

        def make_args(signature: int) -> tuple[torch.Tensor, torch.Tensor]:
            first_group_size = 128 * (signature + 1)
            layout = torch.cat(
                (
                    torch.zeros(first_group_size, dtype=torch.int32, device=DEVICE),
                    torch.ones(128, dtype=torch.int32, device=DEVICE),
                )
            )
            n_offset = 64 * (signature % 4)
            n_sizes = torch.tensor(
                (64 + n_offset, 256 - n_offset),
                dtype=torch.int32,
                device=DEVICE,
            )
            return layout, n_sizes

        def fake_imports() -> tuple[object, object, object]:
            def make_ptr(
                _dtype: object,
                data_ptr: int,
                _space: object,
                *,
                assumed_align: int,
            ) -> int:
                self.assertEqual(assumed_align, 16)
                return data_ptr

            return object(), make_ptr, object()

        first_args = make_args(0)
        grid = (1, 1, 1)
        # Metadata must be ready before capture, but leave the launch-argument
        # cache cold so capture exercises the new-build ownership path.
        runtime_mod._build_tcgen05_grouped_static_metadata(
            cute_kernel, plan, first_args
        )
        runtime_mod._cuda_stream_capture_context(DEVICE)
        torch.empty(1, device=DEVICE).fill_(0)
        torch.cuda.synchronize(DEVICE)

        capture_stream = torch.cuda.Stream(device=DEVICE)
        with (
            patch(
                "helion.runtime.cute.launcher._get_cute_launcher_imports",
                side_effect=fake_imports,
            ),
            patch(
                "helion.runtime.cute.launcher._torch_dtype_to_cutlass", side_effect=str
            ),
        ):
            with runtime_mod.cute_cuda_graph(stream=capture_stream) as graph:
                runtime_mod._build_cached_cute_schema_and_args(
                    cute_kernel, first_args, grid
                )
                capture_owned = cute_kernel._helion_cute_capture_owned_launch_tensors
                self.assertEqual(len(capture_owned), 1)
                capture_context = next(iter(capture_owned))
                capture_tensors = capture_owned[capture_context]
                problem_tensor = next(
                    tensor for tensor in capture_tensors.values() if tensor.ndim == 2
                )
                problem_ref = weakref.ref(problem_tensor)
                problem_ptr = problem_tensor.data_ptr()

                # Drop one promoted tensor, then ensure the cache-hit path
                # promotes it again from the bounded launch ownership cache.
                capture_tensors.pop(id(problem_tensor))
                runtime_mod._build_cached_cute_schema_and_args(
                    cute_kernel, first_args, grid
                )
                self.assertIs(capture_tensors[id(problem_tensor)], problem_tensor)
                problem_tensor.fill_(17)

            torch.cuda.synchronize(DEVICE)
            problem_tensor.zero_()
            torch.cuda.synchronize(DEVICE)
            del capture_owned, capture_tensors, problem_tensor

            other_args = [make_args(signature) for signature in range(1, 9)]
            for args in other_args:
                runtime_mod._build_cached_cute_schema_and_args(cute_kernel, args, grid)

        self.assertEqual(len(cute_kernel._helion_cute_launch_arg_cache), 8)
        self.assertTrue(
            all(
                entry.owned_tensors
                for entry in cute_kernel._helion_cute_launch_arg_cache.values()
            )
        )
        self.assertEqual(
            len(cute_kernel._helion_tcgen05_grouped_static_metadata_cache), 8
        )
        first_problem = problem_ref()
        self.assertIsNotNone(first_problem)
        assert first_problem is not None
        self.assertFalse(
            any(
                tensor is first_problem
                for entry in cute_kernel._helion_cute_launch_arg_cache.values()
                for tensor in entry.owned_tensors
            )
        )
        self.assertFalse(
            any(
                value is first_problem
                for entry in cute_kernel._helion_tcgen05_grouped_static_metadata_cache.values()
                for value in entry.result.tensors()
            )
        )
        del first_problem

        gc.collect()
        allocator_churn = [
            torch.empty((2, 4), dtype=torch.int32, device=DEVICE) for _ in range(32)
        ]
        graph.replay()
        torch.cuda.synchronize(DEVICE)

        retained_problem = problem_ref()
        self.assertIsNotNone(retained_problem)
        assert retained_problem is not None
        self.assertEqual(retained_problem.data_ptr(), problem_ptr)
        self.assertTrue(torch.all(retained_problem == 17).item())
        self.assertEqual(len(allocator_churn), 32)

        capture_owned = cute_kernel._helion_cute_capture_owned_launch_tensors
        self.assertEqual(len(capture_owned), 1)
        del retained_problem, graph
        gc.collect()
        self.assertEqual(capture_owned, {})
        self.assertIsNone(problem_ref())

    def test_grouped_worklist_metadata_cache_guards_source_extents(self) -> None:
        if DEVICE.type != "cuda":
            self.skipTest("grouped worklist metadata test needs CUDA")
        runtime_mod = importlib.import_module("helion.runtime")
        plan = {
            "kind": "tcgen05_grouped_static_persistent",
            "layout_idx": 2,
            "lhs_idx": 0,
            "rhs_idx": 1,
            "group_count": 1,
            "bm": 128,
            "bn": 64,
            "bk": 64,
            "n_size": 64,
            "k_total_size": 64,
            "worklist_metadata": True,
            "dynamic_ab_tensormaps": True,
            "problem_sizes_arg": "problem_sizes",
            "starts_arg": "starts",
            "real_groups_arg": "real_groups",
            "total_clusters_arg": "total_clusters",
        }
        cute_kernel = type("DummyCuteKernel", (), {})()
        cute_kernel._helion_cute_wrapper_plans = [plan]
        layout = torch.tensor(((0, 0, 128, 0),), dtype=torch.int32, device=DEVICE)
        rhs = torch.empty((1, 64, 64), dtype=torch.bfloat16, device=DEVICE)

        runtime_mod._build_tcgen05_grouped_static_metadata(
            cute_kernel,
            plan,
            (torch.empty((128, 64), dtype=torch.bfloat16, device=DEVICE), rhs, layout),
        )
        with self.assertRaisesRegex(BackendUnsupported, "row exceeds A extent"):
            runtime_mod._build_tcgen05_grouped_static_metadata(
                cute_kernel,
                plan,
                (
                    torch.empty((64, 64), dtype=torch.bfloat16, device=DEVICE),
                    rhs,
                    layout,
                ),
            )
        with self.assertRaisesRegex(BackendUnsupported, "outside B_grouped"):
            runtime_mod._build_tcgen05_grouped_static_metadata(
                cute_kernel,
                plan,
                (
                    torch.empty((128, 64), dtype=torch.bfloat16, device=DEVICE),
                    torch.empty((0, 64, 64), dtype=torch.bfloat16, device=DEVICE),
                    layout,
                ),
            )

    def test_grouped_device_split_cluster_bound_uses_source_profile(self) -> None:
        launcher = importlib.import_module("helion.runtime.cute.launcher")
        common_plan = {
            "group_count": 8,
            "m_size": 2048,
            "n_size": 512,
        }

        self.assertEqual(
            launcher._tcgen05_grouped_device_split_total_clusters(
                {**common_plan, "source_m_tile": 224}
            ),
            160,
        )
        self.assertEqual(
            launcher._tcgen05_grouped_device_split_total_clusters(
                {**common_plan, "source_m_tile": 256}
            ),
            128,
        )

    def test_grouped_device_split_dimensions_must_fit_int32(self) -> None:
        launcher = importlib.import_module("helion.runtime.cute.launcher")
        plan = {
            "group_count": 1,
            "orientation": "nm",
            "bk": 128,
            "source_m_tile": 256,
            "m_size": torch.iinfo(torch.int32).max + 1,
            "n_size": 512,
            "k_total_size": 128,
            "dynamic_ab_tensormaps": True,
            "dynamic_ab_tensormap_rank": 2,
            "dynamic_d_tensormap": True,
        }

        with self.assertRaisesRegex(BackendUnsupported, "positive signed Int32"):
            launcher._validate_tcgen05_grouped_device_split_sizes(
                plan,
                torch.empty(1, dtype=torch.int32),
            )

    def test_grouped_static_problem_shapes_runtime_guard(self) -> None:
        if DEVICE.type != "cuda":
            self.skipTest("grouped static problem-shape guard needs CUDA")
        runtime_mod = importlib.import_module("helion.runtime")
        base_plan = {
            "kind": "tcgen05_grouped_static_persistent",
            "layout_idx": 0,
            "n_sizes_idx": 1,
            "group_count": 2,
            "bm": 128,
            "bn": 64,
            "bk": 128,
            "n_size": 256,
            "k_total_size": 128,
            "problem_sizes_arg": "problem_sizes",
            "starts_arg": "starts",
            "total_clusters_arg": "total_clusters",
        }
        layout = torch.cat(
            (
                torch.zeros(128, dtype=torch.int32, device=DEVICE),
                torch.ones(128, dtype=torch.int32, device=DEVICE),
            )
        )
        n_sizes = torch.tensor((64, 256), dtype=torch.int32, device=DEVICE)
        args = (layout, n_sizes)
        actual_shapes = ((128, 64, 128), (128, 256, 128))

        matching_plan = {**base_plan, "static_problem_shapes": actual_shapes}
        matching_kernel = type("MatchingDummyCuteKernel", (), {})()
        matching_kernel._helion_cute_wrapper_plans = [matching_plan]
        runtime_mod._build_tcgen05_grouped_static_metadata(
            matching_kernel, matching_plan, args
        )

        mismatches = (
            ((64, 64, 128), (128, 256, 128)),
            ((128, 64, 128), (128, 192, 128)),
            ((128, 64, 64), (128, 256, 128)),
        )
        for expected_shapes in mismatches:
            with self.subTest(expected_shapes=expected_shapes):
                plan = {**base_plan, "static_problem_shapes": expected_shapes}
                cute_kernel = type("MismatchedDummyCuteKernel", (), {})()
                cute_kernel._helion_cute_wrapper_plans = [plan]
                with self.assertRaisesRegex(
                    BackendUnsupported,
                    "static problem-shape specialization",
                ):
                    runtime_mod._build_tcgen05_grouped_static_metadata(
                        cute_kernel, plan, args
                    )

    def test_grouped_inference_metadata_tracks_value_mutation(self) -> None:
        if DEVICE.type != "cuda":
            self.skipTest("grouped inference metadata test needs CUDA")
        runtime_mod = importlib.import_module("helion.runtime")
        kernel_mod = importlib.import_module("helion.runtime.kernel")
        plan = {
            "kind": "tcgen05_grouped_static_persistent",
            "layout_idx": 0,
            "n_sizes_idx": 1,
            "group_count": 2,
            "bm": 128,
            "bn": 64,
            "bk": 128,
            "n_size": 256,
            "k_total_size": 128,
            "problem_sizes_arg": "problem_sizes",
            "starts_arg": "starts",
            "total_clusters_arg": "total_clusters",
        }
        cute_kernel = type("DummyCuteKernel", (), {})()
        cute_kernel._helion_cute_wrapper_plans = [plan]

        with torch.inference_mode():
            layout = torch.cat(
                (
                    torch.zeros(128, dtype=torch.int32, device=DEVICE),
                    torch.ones(128, dtype=torch.int32, device=DEVICE),
                )
            )
            n_sizes = torch.tensor((64, 256), dtype=torch.int32, device=DEVICE)
            args = (layout, n_sizes)
            self.assertTrue(torch.is_inference(layout))

            first = runtime_mod._build_tcgen05_grouped_static_metadata(
                cute_kernel, plan, args
            )
            self.assertIs(
                runtime_mod._build_tcgen05_grouped_static_metadata(
                    cute_kernel, plan, args
                ),
                first,
            )
            first_problem_sizes = first.result.problem_sizes.cpu().clone()
            first_launch_key = runtime_mod._cute_launch_arg_cache_key(
                cute_kernel, args, (1, 1, 1)
            )
            first_guard = runtime_mod._cute_last_launch_arg_guard(
                cute_kernel, args, (1, 1, 1)
            )
            values_cache = kernel_mod.WeakIdKeyDictionary()
            self.assertEqual(
                kernel_mod._cute_int_1d_tensor_values(n_sizes, values_cache),
                (64, 256),
            )

            n_sizes.copy_(torch.tensor((128, 192), dtype=torch.int32, device=DEVICE))

            self.assertFalse(first_guard.matches(cute_kernel, args, (1, 1, 1)))
            self.assertNotEqual(
                runtime_mod._cute_launch_arg_cache_key(cute_kernel, args, (1, 1, 1)),
                first_launch_key,
            )
            self.assertEqual(
                kernel_mod._cute_int_1d_tensor_values(n_sizes, values_cache),
                (128, 192),
            )
            updated = runtime_mod._build_tcgen05_grouped_static_metadata(
                cute_kernel, plan, args
            )

        self.assertIsNot(updated, first)
        self.assertFalse(
            torch.equal(updated.result.problem_sizes.cpu(), first_problem_sizes)
        )
        with (
            patch(
                "helion.runtime.cute.launcher._cuda_stream_capture_context",
                return_value=(123, 456),
            ),
            self.assertRaisesRegex(BackendUnsupported, "inference tensor.*capture"),
        ):
            runtime_mod._tcgen05_grouped_tensor_mutation_key(n_sizes)

    def test_grouped_metadata_rejects_mixed_cuda_devices(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest("mixed-device grouped metadata test needs two CUDA devices")
        runtime_mod = importlib.import_module("helion.runtime")
        device0 = torch.device(DEVICE.type, 0)
        device1 = torch.device(DEVICE.type, 1)
        layout = torch.zeros(128, dtype=torch.int32, device=device0)
        other = torch.empty(1, device=device1)

        with self.assertRaisesRegex(
            BackendUnsupported, "every tensor argument.*mismatched argument indices"
        ):
            runtime_mod._build_tcgen05_grouped_static_metadata(
                object(), {"layout_idx": 0}, (layout, other)
            )

    def test_cute_build_schema_excludes_stream_from_cached_args(self) -> None:
        # The stream must never be part of the cached launch args produced by
        # ``_build_cute_schema_and_args`` (it is appended per launch instead).
        # Patch the cute imports so the builder runs without a real cutlass
        # install; the sentinel stream must NOT appear in the returned args.
        sentinel_stream = object()

        def fake_imports() -> tuple[object, ...]:
            gmem = object()

            def make_ptr(*_a: object, **_k: object) -> str:
                return "ptr"

            def current_stream() -> object:
                return sentinel_stream

            return (gmem, make_ptr, current_stream)

        def cute_kernel(alpha: int) -> None:
            pass

        with (
            patch(
                "helion.runtime.cute.launcher._get_cute_launcher_imports",
                side_effect=fake_imports,
            ),
            patch(
                "helion.runtime.cute.launcher._cute_kernel_param_is_constexpr",
                return_value=(False,),
            ),
        ):
            launch = helion_runtime._build_cute_schema_and_args(
                cute_kernel, (7,), (2, 1, 1)
            )

        # Args end with the grid; the stream is not baked in.
        self.assertEqual(launch.launch_args, (7, 2, 1, 1))
        self.assertNotIn(sentinel_stream, launch.launch_args)
        self.assertEqual(launch.schema, (("scalar", "int"),))

    def test_cute_launcher_launch_arg_cache_distinguishes_signed_zero(
        self,
    ) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        build_calls: list[tuple[object, ...]] = []
        launched_args: list[tuple[object, ...]] = []

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                launched_args.append(args)
                return ("launched", args)

        def fake_build(
            _cute_kernel: object,
            args: tuple[object, ...],
            _grid: tuple[int, int, int],
        ) -> helion_runtime._CuteLaunchArgCacheEntry:
            build_calls.append(args)
            return _launch_entry((("scalar", "float"),), (f"float-{len(build_calls)}",))

        with (
            patch(
                "helion.runtime.cute.launcher._build_cute_schema_and_args",
                side_effect=fake_build,
            ),
            patch(
                "helion.runtime.cute.launcher._cute_current_stream",
                return_value="stream",
            ),
            patch(
                "helion.runtime.cute.launcher._get_compiled_cute_launcher",
                return_value=FakeCompiled(),
            ),
        ):
            positive = default_cute_launcher(cute_kernel, (1,), 0.0)
            negative = default_cute_launcher(cute_kernel, (1,), -0.0)

        self.assertEqual(build_calls, [(0.0,), (-0.0,)])
        self.assertEqual(launched_args, [("float-1", "stream"), ("float-2", "stream")])
        self.assertEqual(positive, ("launched", ("float-1", "stream")))
        self.assertEqual(negative, ("launched", ("float-2", "stream")))

    def test_cute_launcher_last_launch_distinguishes_scalar_kinds(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        build_calls: list[tuple[object, ...]] = []
        launched_args: list[tuple[object, ...]] = []

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                launched_args.append(args)
                return ("launched", args)

        def fake_build(
            _cute_kernel: object,
            args: tuple[object, ...],
            _grid: tuple[int, int, int],
        ) -> helion_runtime._CuteLaunchArgCacheEntry:
            build_calls.append(args)
            scalar_kind = type(args[0]).__name__
            return _launch_entry(
                (("scalar", scalar_kind),), (f"{scalar_kind}-{len(build_calls)}",)
            )

        with (
            patch(
                "helion.runtime.cute.launcher._build_cute_schema_and_args",
                side_effect=fake_build,
            ),
            patch(
                "helion.runtime.cute.launcher._cute_current_stream",
                return_value="stream",
            ),
            patch(
                "helion.runtime.cute.launcher._get_compiled_cute_launcher",
                return_value=FakeCompiled(),
            ),
        ):
            bool_launch = default_cute_launcher(cute_kernel, (1,), True)
            bool_hit = default_cute_launcher(cute_kernel, (1,), True)
            int_miss = default_cute_launcher(cute_kernel, (1,), 1)
            float_miss = default_cute_launcher(cute_kernel, (1,), 1.0)

        self.assertEqual(build_calls, [(True,), (1,), (1.0,)])
        self.assertEqual(bool_launch, bool_hit)
        self.assertEqual(int_miss, ("launched", ("int-2", "stream")))
        self.assertEqual(float_miss, ("launched", ("float-3", "stream")))
        self.assertEqual(
            launched_args,
            [
                ("bool-1", "stream"),
                ("bool-1", "stream"),
                ("int-2", "stream"),
                ("float-3", "stream"),
            ],
        )

    def test_cute_launcher_sets_arch_env_only_before_first_compile(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                return ("launched", args)

        with (
            patch(
                "helion.runtime.cute.launcher._build_cute_schema_and_args",
                return_value=_launch_entry((("scalar", "int"),), ("launch-arg",)),
            ),
            patch(
                "helion.runtime.cute.launcher._cute_current_stream",
                return_value="stream",
            ),
            patch(
                "helion.runtime.cute.launcher._create_cute_wrapper",
                return_value="jit-wrapper",
            ),
            patch(
                "helion.runtime.cute.launcher._ensure_cute_dsl_arch_env"
            ) as ensure_arch,
            patch("cutlass.cute.compile", return_value=FakeCompiled()),
        ):
            first = default_cute_launcher(cute_kernel, (1,), 7, block=(32, 1, 1))
            second = default_cute_launcher(cute_kernel, (1,), 7, block=(32, 1, 1))

        self.assertEqual(ensure_arch.call_count, 1)
        self.assertEqual(first, ("launched", ("launch-arg", "stream")))
        self.assertEqual(second, first)

    def test_cute_launcher_constexpr_float_cache_distinguishes_signed_zero(
        self,
    ) -> None:
        def cute_kernel(alpha: cutlass.Constexpr) -> None:
            pass

        created_schema_keys: list[tuple[tuple[object, ...], ...]] = []

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                return ("launched", args)

        def fake_create_wrapper(
            _cute_kernel: object,
            schema_key: tuple[tuple[object, ...], ...],
            _block: tuple[int, int, int],
            **_kwargs: object,
        ) -> str:
            created_schema_keys.append(schema_key)
            return f"jit-wrapper-{len(created_schema_keys)}"

        with (
            patch(
                "helion.runtime.cute.launcher._create_cute_wrapper",
                side_effect=fake_create_wrapper,
            ),
            patch(
                "helion.runtime.cute.launcher._cute_current_stream",
                return_value="stream",
            ),
            patch("helion.runtime.cute.launcher._ensure_cute_dsl_arch_env"),
            patch("cutlass.cute.compile", return_value=FakeCompiled()),
        ):
            positive = default_cute_launcher(cute_kernel, (1,), 0.0)
            negative = default_cute_launcher(cute_kernel, (1,), -0.0)

        self.assertEqual(len(created_schema_keys), 2)
        self.assertNotEqual(created_schema_keys[0], created_schema_keys[1])
        self.assertEqual(positive[0], "launched")
        self.assertEqual(positive[1][:3], (1, 1, 1))
        # Stream is appended fresh as the trailing launch arg.
        self.assertEqual(positive[1][-1], "stream")
        self.assertEqual(negative, positive)

    def test_cute_launcher_launch_arg_cache_misses_on_tensor_pointer_change(
        self,
    ) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        build_calls: list[int] = []
        launched_args: list[tuple[object, ...]] = []
        tensor = torch.empty(2, device=DEVICE)
        other_tensor = torch.empty(2, device=DEVICE)
        self.assertNotEqual(tensor.data_ptr(), other_tensor.data_ptr())

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                launched_args.append(args)
                return ("launched", args)

        def fake_build(
            _cute_kernel: object,
            args: tuple[object, ...],
            _grid: tuple[int, int, int],
        ) -> helion_runtime._CuteLaunchArgCacheEntry:
            build_calls.append(cast("torch.Tensor", args[0]).data_ptr())
            return _launch_entry(
                (("tensor", "torch.float32", 1),), (f"ptr-{len(build_calls)}",)
            )

        with (
            patch(
                "helion.runtime.cute.launcher._build_cute_schema_and_args",
                side_effect=fake_build,
            ),
            patch(
                "helion.runtime.cute.launcher._cute_current_stream",
                return_value="stream",
            ),
            patch(
                "helion.runtime.cute.launcher._get_compiled_cute_launcher",
                return_value=FakeCompiled(),
            ),
        ):
            first = default_cute_launcher(cute_kernel, (1,), tensor)
            second = default_cute_launcher(cute_kernel, (1,), tensor)
            third = default_cute_launcher(cute_kernel, (1,), other_tensor)

        self.assertEqual(build_calls, [tensor.data_ptr(), other_tensor.data_ptr()])
        self.assertEqual(
            launched_args,
            [("ptr-1", "stream"), ("ptr-1", "stream"), ("ptr-2", "stream")],
        )
        self.assertEqual(first, second)
        self.assertEqual(third, ("launched", ("ptr-2", "stream")))

    def test_cute_launcher_last_launch_guards_launch_parameters(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        build_calls: list[tuple[object, ...]] = []
        compiled_calls: list[tuple[tuple[int, int, int], str | None]] = []
        launched: list[tuple[int, tuple[object, ...]]] = []

        class FakeCompiled:
            def __init__(self, index: int) -> None:
                self.index = index

            def __call__(self, *args: object) -> tuple[int, tuple[object, ...]]:
                launched.append((self.index, args))
                return (self.index, args)

        def fake_build(
            _cute_kernel: object,
            args: tuple[object, ...],
            grid: tuple[int, int, int],
        ) -> helion_runtime._CuteLaunchArgCacheEntry:
            build_calls.append((*args, *grid))
            return _launch_entry((("scalar", "int"),), ("launch", *args, *grid))

        def fake_get(
            _cute_kernel: object,
            _schema_key: tuple[tuple[object, ...], ...],
            block: tuple[int, int, int],
            *,
            compile_options: str | None = None,
            arch_args: tuple[object, ...] | None = None,
        ) -> FakeCompiled:
            self.assertIsNotNone(arch_args)
            compiled_calls.append((block, compile_options))
            return FakeCompiled(len(compiled_calls))

        with (
            patch(
                "helion.runtime.cute.launcher._build_cute_schema_and_args",
                side_effect=fake_build,
            ),
            patch(
                "helion.runtime.cute.launcher._cute_current_stream",
                return_value="stream",
            ),
            patch(
                "helion.runtime.cute.launcher._get_compiled_cute_launcher",
                side_effect=fake_get,
            ),
        ):
            first = default_cute_launcher(cute_kernel, (2,), 7, block=(32, 1, 1))
            second = default_cute_launcher(cute_kernel, (2,), 7, block=(32, 1, 1))
            options_miss = default_cute_launcher(
                cute_kernel,
                (2,),
                7,
                block=(32, 1, 1),
                cute_compile_options="--generate-line-info",
            )
            block_miss = default_cute_launcher(
                cute_kernel,
                (2,),
                7,
                block=(64, 1, 1),
                cute_compile_options="--generate-line-info",
            )
            grid_miss = default_cute_launcher(
                cute_kernel,
                (3,),
                7,
                block=(64, 1, 1),
                cute_compile_options="--generate-line-info",
            )

        self.assertEqual(first[0], 1)
        self.assertEqual(second[0], 1)
        self.assertEqual(options_miss[0], 2)
        self.assertEqual(block_miss[0], 3)
        self.assertEqual(grid_miss[0], 4)
        self.assertEqual(len(compiled_calls), 4)
        self.assertEqual(
            [entry[0] for entry in launched],
            [1, 1, 2, 3, 4],
        )
        self.assertEqual(
            build_calls,
            [
                (7, 2, 1, 1),
                (7, 3, 1, 1),
            ],
        )

    def test_cute_launcher_last_launch_guards_tensor_pointer_and_schema(
        self,
    ) -> None:
        if DEVICE.type != "cuda":
            self.skipTest("CuTe launcher tensor guard test needs CUDA")
        cute_kernel = type("DummyCuteKernel", (), {})()
        tensor = torch.empty(4, device=DEVICE)
        same_storage_view = tensor.view_as(tensor)
        self.assertEqual(tensor.data_ptr(), same_storage_view.data_ptr())
        self.assertNotEqual(id(tensor), id(same_storage_view))
        build_calls: list[tuple[int, bool]] = []
        compiled_calls: list[int] = []

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                return ("launched", args)

        def fake_build(
            _cute_kernel: object,
            args: tuple[object, ...],
            _grid: tuple[int, int, int],
        ) -> helion_runtime._CuteLaunchArgCacheEntry:
            build_calls.append(
                (
                    id(args[0]),
                    bool(
                        getattr(
                            _cute_kernel,
                            "_helion_cute_disable_bake_tensor_shapes",
                            False,
                        )
                    ),
                )
            )
            return _launch_entry(
                (("tensor", "torch.float32", 1),), (f"ptr-{len(build_calls)}",)
            )

        def fake_get(*_args: object, **_kwargs: object) -> FakeCompiled:
            compiled_calls.append(len(compiled_calls) + 1)
            return FakeCompiled()

        with (
            patch(
                "helion.runtime.cute.launcher._build_cute_schema_and_args",
                side_effect=fake_build,
            ),
            patch(
                "helion.runtime.cute.launcher._cute_current_stream",
                return_value="stream",
            ),
            patch(
                "helion.runtime.cute.launcher._get_compiled_cute_launcher",
                side_effect=fake_get,
            ),
        ):
            first = default_cute_launcher(cute_kernel, (1,), tensor)
            second = default_cute_launcher(cute_kernel, (1,), tensor)
            view_hit = default_cute_launcher(cute_kernel, (1,), same_storage_view)
            cute_kernel._helion_cute_disable_bake_tensor_shapes = True
            schema_miss = default_cute_launcher(cute_kernel, (1,), same_storage_view)

        self.assertEqual(first, ("launched", ("ptr-1", "stream")))
        self.assertEqual(second, first)
        self.assertEqual(view_hit, first)
        self.assertEqual(schema_miss, ("launched", ("ptr-2", "stream")))
        self.assertEqual(
            build_calls,
            [
                (id(tensor), False),
                (id(same_storage_view), True),
            ],
        )
        self.assertEqual(len(compiled_calls), 2)

    def test_cute_launcher_last_launch_misses_on_same_tensor_metadata_mutation(
        self,
    ) -> None:
        if DEVICE.type != "cuda":
            self.skipTest("CuTe launcher tensor guard test needs CUDA")
        cute_kernel = type("DummyCuteKernel", (), {})()
        base = torch.empty(32, device=DEVICE)
        tensor = base.as_strided((4, 2), (2, 1), 0)
        tensor_id = id(tensor)
        build_calls: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = []

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                return ("launched", args)

        def fake_build(
            _cute_kernel: object,
            args: tuple[object, ...],
            _grid: tuple[int, int, int],
        ) -> helion_runtime._CuteLaunchArgCacheEntry:
            tensor_arg = cast("torch.Tensor", args[0])
            build_calls.append(
                (
                    tuple(int(size) for size in tensor_arg.shape),
                    tuple(int(stride) for stride in tensor_arg.stride()),
                    int(tensor_arg.storage_offset()),
                    int(tensor_arg.data_ptr()),
                )
            )
            return _launch_entry(
                (("tensor", "torch.float32", 2),), (f"ptr-{len(build_calls)}",)
            )

        with (
            patch(
                "helion.runtime.cute.launcher._build_cute_schema_and_args",
                side_effect=fake_build,
            ),
            patch(
                "helion.runtime.cute.launcher._cute_current_stream",
                return_value="stream",
            ),
            patch(
                "helion.runtime.cute.launcher._get_compiled_cute_launcher",
                return_value=FakeCompiled(),
            ),
        ):
            first = default_cute_launcher(cute_kernel, (1,), tensor)
            second = default_cute_launcher(cute_kernel, (1,), tensor)
            tensor.as_strided_((4, 2), (1, 4), 0)
            stride_miss = default_cute_launcher(cute_kernel, (1,), tensor)
            tensor.as_strided_((4, 2), (2, 1), 1)
            offset_miss = default_cute_launcher(cute_kernel, (1,), tensor)

        self.assertEqual(id(tensor), tensor_id)
        self.assertEqual(first, second)
        self.assertEqual(stride_miss, ("launched", ("ptr-2", "stream")))
        self.assertEqual(offset_miss, ("launched", ("ptr-3", "stream")))
        self.assertEqual(
            [(shape, stride, offset) for shape, stride, offset, _ in build_calls],
            [
                ((4, 2), (2, 1), 0),
                ((4, 2), (1, 4), 0),
                ((4, 2), (2, 1), 1),
            ],
        )
        self.assertEqual(build_calls[0][3], build_calls[1][3])
        self.assertNotEqual(build_calls[1][3], build_calls[2][3])

    def test_kernel_late_cute_specialization_rekeys_bound_kernel(self) -> None:
        identity = _runtime_identity_kernel()
        x = torch.empty(1)
        layout = torch.zeros(128, dtype=torch.int64)
        args = (x, layout)
        bound = identity.bind(args)
        signature = identity._base_specialization_key(args)
        plan: dict[str, object] = {
            "kind": "tcgen05_grouped_static_persistent",
            "layout_bind_idx": 1,
            "group_count": 1,
            "bm": 128,
        }
        kernel_mod = importlib.import_module("helion.runtime.kernel")
        self.assertEqual(
            kernel_mod._cute_grouped_static_tail_extra_descriptors([plan]), ()
        )
        plan.update(
            grouped_static_has_m_tail=True,
            grouped_static_has_n_tail=False,
        )
        bound.env.cute_resolved_wrapper_plans = [plan]
        self.assertEqual(bound._base_spec_key, signature)
        with bound.env, bound._runtime_arg_values_for_codegen():
            self.assertEqual(
                bound.env.runtime_arg_values_by_name,
                {"x": x, "layout": layout},
            )
            bound._register_cute_grouped_static_tail_specializations()

        key0 = identity._fast_dispatch_key(args)
        self.assertIs(identity.bind(args), bound)
        layout[64:] = -1
        key1 = identity._fast_dispatch_key(args)

        self.assertNotEqual(key0, key1)
        self.assertIsNot(identity.bind(args), bound)

    def test_kernel_growing_late_specialization_evicts_all_stale_keys(self) -> None:
        identity = _runtime_identity_kernel()
        x = torch.empty(1)
        layout_a = torch.tensor([0, 0], dtype=torch.int64)
        args_a = (x, layout_a)
        signature = identity._base_specialization_key(args_a)
        bound_a = identity.bind(args_a)

        def first(values: Sequence[object]) -> Hashable:
            return int(cast("torch.Tensor", values[1])[0].item())

        def last(values: Sequence[object]) -> Hashable:
            return int(cast("torch.Tensor", values[1])[-1].item())

        identity._extend_bound_kernel_specializations(
            bound_a, signature, [first], args_a
        )
        args_b = (x, torch.tensor([1, 1], dtype=torch.int64))
        bound_b = identity.bind(args_b)
        self.assertIsNot(bound_b, bound_a)
        identity._dispatch_cache.update(old_a=bound_a, old_b=bound_b)

        identity._extend_bound_kernel_specializations(
            bound_a, signature, [last], args_a
        )

        signature_entries = [
            (key, cached_bound)
            for key, cached_bound in identity._bound_kernels.items()
            if key.specialization_key == signature
        ]
        self.assertEqual(len(signature_entries), 1)
        self.assertEqual(signature_entries[0][0].extra_results, (0, 0))
        self.assertIs(signature_entries[0][1], bound_a)
        self.assertEqual(identity._dispatch_cache, {})

    def test_bound_kernel_runtime_args_are_scoped_to_codegen(self) -> None:
        @helion.kernel()
        def identity(
            x: torch.Tensor, layout: torch.Tensor, n_sizes: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                out[tile] = x[tile]
            return out

        x = torch.empty(16, device=DEVICE)
        layout = torch.tensor([0], device=DEVICE, dtype=torch.int64)
        n_sizes = torch.tensor([128], device=DEVICE, dtype=torch.int64)
        bound = identity.bind((x, layout, n_sizes))
        layout_ref = weakref.ref(layout)

        self.assertEqual(bound.env.runtime_arg_values_by_name, {})
        with bound._runtime_arg_values_for_codegen():
            self.assertEqual(
                bound.env.runtime_arg_values_by_name,
                {"x": x, "layout": layout, "n_sizes": n_sizes},
            )
        self.assertEqual(bound.env.runtime_arg_values_by_name, {})

        del layout
        gc.collect()
        self.assertIsNone(layout_ref())

    def test_bound_kernel_codegen_uses_stable_bind_and_current_call_args(self) -> None:
        identity = _runtime_identity_kernel()
        x = torch.empty(1)
        layout_a = torch.zeros(1, dtype=torch.int64)
        layout_b = torch.ones(1, dtype=torch.int64)
        bound = identity.bind((x, layout_a))
        self.assertIs(identity.bind((x, layout_b)), bound)
        delayed_codegen_layouts: list[torch.Tensor] = []

        def fake_generate_ast(*_args: object, **_kwargs: object) -> ast.Module:
            delayed_codegen_layouts.append(_runtime_layout(bound))
            return ast.parse("pass")

        with patch("helion.runtime.kernel.generate_ast", side_effect=fake_generate_ast):
            bound.to_code(helion.Config(block_sizes=[1]))

        self.assertEqual(delayed_codegen_layouts, [layout_a])
        call_codegen_layouts: list[torch.Tensor] = []

        def fake_ensure(_args: tuple[object, ...]) -> None:
            with bound._runtime_arg_values_for_codegen():
                call_codegen_layouts.append(_runtime_layout(bound))
            bound._run = lambda _x, layout: layout

        with patch.object(bound, "ensure_config_exists", side_effect=fake_ensure):
            result = bound(x, layout_b)

        self.assertEqual(call_codegen_layouts, [layout_b])
        self.assertIs(result, layout_b)

    def test_concurrent_first_compile_uses_each_calls_runtime_args(self) -> None:
        identity = _runtime_identity_kernel()
        x = torch.empty(1)
        layouts = {
            "a": torch.zeros(1, dtype=torch.int64),
            "b": torch.ones(1, dtype=torch.int64),
        }
        bind_lock = identity._bind_lock
        both_binds_waiting = threading.Event()
        bind_attempts: list[None] = []

        class TrackingBindLock:
            def __enter__(self) -> None:
                bind_attempts.append(None)
                if len(bind_attempts) == 2:
                    both_binds_waiting.set()
                bind_lock.acquire()

            def __exit__(self, *_args: object) -> None:
                bind_lock.release()

        identity._bind_lock = TrackingBindLock()
        with ThreadPoolExecutor(max_workers=2) as executor:
            bind_lock.acquire()
            try:
                bound_a = executor.submit(identity.bind, (x, layouts["a"]))
                bound_b = executor.submit(identity.bind, (x, layouts["b"]))
                self.assertTrue(both_binds_waiting.wait(timeout=5))
            finally:
                bind_lock.release()
            bound = bound_a.result(timeout=5)
            self.assertIs(bound_b.result(timeout=5), bound)
        identity._bind_lock = bind_lock

        first_compile_entered = threading.Event()
        second_compile_waiting = threading.Event()
        release_first_compile = threading.Event()
        seen_layouts: list[torch.Tensor] = []
        compile_lock = bound._first_compile_lock

        class TrackingCompileLock:
            def __enter__(self) -> None:
                if first_compile_entered.is_set():
                    second_compile_waiting.set()
                compile_lock.acquire()

            def __exit__(self, *_args: object) -> None:
                compile_lock.release()

        bound._first_compile_lock = cast("Any", TrackingCompileLock())

        def layout_value(values: tuple[object, ...]) -> int:
            return int(cast("torch.Tensor", values[1]).item())

        def fake_ensure(current_bound: Any, call_args: tuple[object, ...]) -> None:
            with current_bound._runtime_arg_values_for_codegen():
                seen_layouts.append(_runtime_layout(current_bound))
            if current_bound is bound:
                first_compile_entered.set()
                self.assertTrue(release_first_compile.wait(timeout=5))
                identity._extend_bound_kernel_specializations(
                    current_bound,
                    current_bound._base_spec_key,
                    [layout_value],
                    call_args,
                )
            current_bound._run = lambda _x, layout: layout

        with (
            patch(
                "helion.runtime.kernel.BoundKernel.ensure_config_exists",
                new=fake_ensure,
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            future_a = executor.submit(identity, x, layouts["a"])
            self.assertTrue(first_compile_entered.wait(timeout=5))
            future_b = executor.submit(identity, x, layouts["b"])
            try:
                self.assertTrue(second_compile_waiting.wait(timeout=5))
            finally:
                release_first_compile.set()
            output_a = future_a.result(timeout=5)
            output_b = future_b.result(timeout=5)

        self.assertEqual(seen_layouts, [layouts["a"], layouts["b"]])
        self.assertIs(output_a, layouts["a"])
        self.assertIs(output_b, layouts["b"])
        rebound_b = identity.bind((x, layouts["b"]))
        self.assertIsNot(rebound_b, bound)
        fast_key_b = identity._fast_dispatch_key((x, layouts["b"]))
        self.assertIsNotNone(fast_key_b)
        assert fast_key_b is not None
        self.assertIs(identity._dispatch_cache[fast_key_b], rebound_b)

        def stale_run(*_args: object) -> torch.Tensor:
            raise AssertionError("fast dispatch used the pre-specialization kernel")

        bound._run = stale_run
        self.assertIs(identity(x, layouts["b"]), layouts["b"])

    def test_compile_cache_lookup_normalizes_config_copy(self) -> None:
        identity = _runtime_identity_kernel()
        bound = identity.bind((torch.empty(16), torch.empty(1)))
        config = helion.Config(block_sizes=[16])
        normalized = bound._normalized_config_copy(config)
        self.assertNotEqual(config, normalized)
        compiled = cast("Any", lambda x: x)
        bound._compile_cache[normalized] = compiled

        self.assertIs(bound.compile_config(config), compiled)
        self.assertEqual(config, helion.Config(block_sizes=[16]))

    def test_bound_kernel_ignores_non_tensor_runtime_args(self) -> None:
        @dataclass(frozen=True)
        class RuntimeMetadata:
            layout: torch.Tensor
            label: str

        @helion.kernel()
        def identity(x: torch.Tensor, metadata: RuntimeMetadata) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                out[tile] = x[tile]
            return out

        x = torch.empty(16, device=DEVICE)
        layout = torch.tensor([0], device=DEVICE, dtype=torch.int64)
        metadata = RuntimeMetadata(layout=layout, label="grouped-layout")
        metadata_ref = weakref.ref(metadata)
        layout_ref = weakref.ref(layout)
        bound = identity.bind((x, metadata))

        with bound._runtime_arg_values_for_codegen():
            self.assertEqual(bound.env.runtime_arg_values_by_name, {"x": x})

        del metadata, layout
        gc.collect()
        self.assertIsNone(metadata_ref())
        self.assertIsNone(layout_ref())

    def test_cute_launcher_tensor_shape_baking(self) -> None:
        tensor = torch.empty((2, 128, 64), device=DEVICE, dtype=torch.float16)
        full_tcgen05_plan = {
            "kind": "tcgen05_ab_tma",
            "m_size": 128,
            "n_size": 128,
            "k_total_size": 64,
            "bm": 128,
            "bn": 128,
            "bk": 64,
        }
        cases = (
            ({"kind": "helion_small_biased_attention"}, False, True),
            (full_tcgen05_plan, False, True),
            ({**full_tcgen05_plan, "n_size": 136}, False, False),
            ({"kind": "helion_flash"}, False, False),
            (full_tcgen05_plan, True, False),
        )
        for plan, disable_bake, expected_baked in cases:
            with self.subTest(plan=plan, disable_bake=disable_bake):
                kernel = type("DummyCuteKernel", (), {})()
                kernel._helion_cute_wrapper_plans = [plan]
                kernel._helion_cute_disable_bake_tensor_shapes = disable_bake
                launch = helion_runtime._build_cute_schema_and_args(
                    kernel, (tensor,), (128, 2, 1)
                )
                if expected_baked:
                    self.assertEqual(
                        launch.schema,
                        (
                            (
                                "tensor",
                                "torch.float16",
                                3,
                                (2, 128, 64),
                                (8192, 64, 1),
                            ),
                        ),
                    )
                    self.assertEqual(len(launch.launch_args), 4)
                else:
                    self.assertEqual(launch.schema, (("tensor", "torch.float16", 3),))
                    self.assertEqual(len(launch.launch_args), 10)

    def test_cute_cluster_shape_from_wrapper_plans(self) -> None:
        self.assertIsNone(_cute_cluster_shape_from_wrapper_plans([]))
        self.assertIsNone(
            _cute_cluster_shape_from_wrapper_plans(
                [{"kind": "tcgen05_ab_tma", "cluster_m": 1, "cluster_n": 1}]
            )
        )
        self.assertEqual(
            _cute_cluster_shape_from_wrapper_plans(
                [
                    {
                        "kind": "tcgen05_ab_tma",
                        "cluster_m": 2,
                        "cluster_n": 1,
                    }
                ]
            ),
            (2, 1, 1),
        )

    def test_cute_cluster_shape_prefers_explicit_kernel_metadata(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        cute_kernel._helion_cute_cluster_shape = (2, 1, 1)
        self.assertEqual(
            _cute_cluster_shape(
                cute_kernel,
                [{"kind": "tcgen05_ab_tma", "cluster_m": 1, "cluster_n": 1}],
            ),
            (2, 1, 1),
        )

    def test_addmm_direct_full_k_tile_static_shapes_falls_back_correctly(self) -> None:
        args = (
            torch.randn(4, 4, device=DEVICE, dtype=torch.float32),
            torch.randn(4, 4, device=DEVICE, dtype=torch.float32),
            torch.randn(4, 4, device=DEVICE, dtype=torch.float32),
        )
        old_static_shapes = cute_matmul_addmm_direct.settings.static_shapes
        cute_matmul_addmm_direct.settings.static_shapes = True
        cute_matmul_addmm_direct.reset()
        try:
            code, out = code_and_output(
                cute_matmul_addmm_direct,
                args,
                block_sizes=[1, 1, 4],
                num_threads=[1, 1, 4],
            )
        finally:
            cute_matmul_addmm_direct.settings.static_shapes = old_static_shapes
            cute_matmul_addmm_direct.reset()
        x, y, bias = args
        expected = torch.addmm(bias, x, y)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
        self.assertIn("cute.arch.warp_reduction_sum", code)
        self.assertNotIn("cute.gemm", code)

    def test_matmul_direct_threaded_k_uses_fp32_accumulation(self) -> None:
        torch.manual_seed(0)
        args = (
            torch.randn(4, 256, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(256, 4, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(
            cute_matmul_direct,
            args,
            block_sizes=[1, 1, 256],
            num_threads=[1, 1, 16],
        )
        expected = torch.matmul(*args)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
        self.assertIn("cute.arch.warp_reduction_sum", code)
        self.assertNotIn("cute.gemm", code)

    def test_matmul_dot(self) -> None:
        args = (
            torch.randn(64, 64, device=DEVICE, dtype=torch.float32),
            torch.randn(64, 64, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(
            cute_matmul_dot,
            args,
            block_sizes=[4, 4, 16],
            num_threads=[4, 4, 1],
        )
        torch.testing.assert_close(out, args[0] @ args[1], atol=1e-1, rtol=1e-2)

    def test_matmul_dot_direct_full_k_tile_falls_back_correctly(self) -> None:
        args = (
            torch.randn(4, 4, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(4, 4, device=DEVICE, dtype=HALF_DTYPE),
        )
        code, out = code_and_output(
            cute_matmul_dot_direct,
            args,
            block_sizes=[1, 1, 4],
            num_threads=[1, 1, 4],
        )
        expected = torch.mm(args[0], args[1], out_dtype=torch.float16)
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
        self.assertIn("cute.arch.warp_reduction_sum", code)
        self.assertNotIn("cute.gemm", code)

    def test_strided_threaded_reduction_uses_warp_per_row(self) -> None:
        """With ``block_sizes=[32, 32]`` and the default
        ``num_threads=[32, 32]`` the warp-per-row plan (P15) swaps the
        thread-axis assignment so each warp owns one M-row.  The
        ``acc.sum(-1)`` then lowers to a per-warp reduction (each warp
        sums its row across the 32 lanes) instead of routing through
        the cross-warp ``_cute_grouped_reduce_shared_two_stage`` SMEM
        path.  The launch dim stays ``(32, 32, 1)`` (N on axis 0, M on
        axis 1) so the joint thread count still fits the budget.
        """
        args = (
            torch.randn(512, 512, device=DEVICE, dtype=torch.float32),
            torch.tensor([200], device=DEVICE, dtype=torch.int64),
        )
        code, out = code_and_output(cute_dynamic_row_sum, args, block_sizes=[32, 32])
        x, end = args
        expected = x[:, : end.item()].sum(dim=1)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        self.assertIn("block=(32, 32, 1)", code)
        # Each warp reduces its own row via ``_cute_grouped_reduce_warp``
        # with ``group_span=32``; no shared-memory two-stage reduce.
        self.assertIn("_cute_grouped_reduce_warp", code)
        self.assertIn("group_span=32", code)
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)

    def test_branch_free_arange_reuses_reduction_thread_axis(self) -> None:
        """A free ``hl.arange`` in a grid branch must reuse the thread axis a
        reduction claimed in a mutually-exclusive sibling branch, not claim a
        fresh one.

        Branches ``pid==0`` / ``pid==1`` reduce over a free ``hl.arange`` (their
        lane dim binds to reduction thread axis 0); the branch-only ``pid==2``
        uses a free ``hl.arange`` with no reduction. Because the three branches
        are mutually exclusive, ``pid==2`` can reuse axis 0. If it instead grabs
        a second thread axis, the launch block becomes 2D (e.g. ``(64, 16, 1)``)
        and the 16 extra lanes re-run ``pid==1``'s single-axis shared-memory
        reduction redundantly, racing on the same SMEM slots and producing
        intermittently wrong output (uninitialized memory leaks through on the
        first launch). Assert the deterministic codegen decision -- a 1-D launch
        block and the ``pid==2`` store indexing thread axis 0 -- which catches the
        race at compile time without depending on the timing-sensitive failure.
        """
        t = 4
        a = torch.randn(t, 8, 32, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(t, 8, 64, device=DEVICE, dtype=torch.bfloat16)
        c = torch.randn(t, 12, 16, device=DEVICE, dtype=torch.bfloat16)
        code, (out_a, out_b, out_c) = code_and_output(
            cute_branch_free_arange_reduction, (a, b, c)
        )

        # Correctness: every output row must be written (a dropped/raced store
        # leaves uninitialized memory that diverges from the reference).
        a_ref = a.float()
        a_scale = torch.rsqrt(
            torch.sum(a_ref * a_ref, dim=-1, keepdim=True) / 32 + 1e-6
        )
        b_ref = b.float()
        b_scale = torch.amax(torch.abs(b_ref), dim=-1, keepdim=True)
        torch.testing.assert_close(out_a, a_ref * a_scale, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(out_b, b_ref / b_scale, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(out_c, c.float() + 1.0, rtol=1e-2, atol=1e-2)

        # Deterministic codegen guard: the branch-only free arange reused the
        # reduction's thread axis, so the launch block stays 1-D and every store
        # indexes thread axis 0. A regression re-introduces a second thread axis
        # (a 2-D ``block=(.., N, 1)`` with ``N > 1``) and a ``thread_idx()[1]``
        # store index.
        import re as _re

        block_match = _re.search(r"block=\((\d+),\s*(\d+),\s*(\d+)\)", code)
        self.assertIsNotNone(block_match, f"no launch block found in:\n{code}")
        assert block_match is not None
        bx, by, bz = (int(g) for g in block_match.groups())
        self.assertEqual(
            (by, bz),
            (1, 1),
            f"expected a 1-D launch block (free arange reused reduction axis 0) "
            f"but got block=({bx}, {by}, {bz}); the branch-only arange claimed a "
            f"spurious second thread axis, racing the single-axis reduction",
        )
        store_lines = "\n".join(line for line in code.splitlines() if ".store(" in line)
        self.assertNotIn(
            "thread_idx()[1]",
            store_lines,
            "a store indexes thread axis 1; the branch-only free arange must "
            "reuse the reduction's axis 0 in mutually-exclusive branches",
        )

        # Lane-bound guard: the launch block is sized to the widest branch
        # (pid==1, db=64), so the narrower branches must mask their surplus
        # thread lanes to their own dim size. Without this, pid==0's reduction
        # store (da=32) and pid==2's free-arange store (dc=16) run lanes 32..63 /
        # 16..63 out of bounds, corrupting the sibling pid==1 reduction's output.
        self.assertIn(
            "cute.arch.thread_idx()[0]) < 32",
            code,
            "pid==0's per-lane access (reduction dim da=32) is not bounded to its "
            "lane extent on the shared 64-wide axis -- lanes 32..63 go OOB",
        )
        self.assertIn(
            "cute.arch.thread_idx()[0]) < 16",
            code,
            "pid==2's per-lane free-arange access (dc=16) is not bounded to its "
            "lane extent on the shared 64-wide axis -- lanes 16..63 go OOB",
        )

    def test_branch_noncanonical_free_arange_lane_bound(self) -> None:
        """A non-canonical free ``hl.arange`` (non-zero start / non-unit step)
        that reuses a wider sibling's thread axis must still mask its surplus
        lanes to its own length.

        ``pid==0`` reduces over a 64-wide free arange (claims thread axis 0);
        ``pid==1`` uses ``hl.arange(8, 24)`` -- length 16, start 8 -- which reuses
        axis 0 (the branches are mutually exclusive). The launch block is sized to
        the wider branch (64), so ``pid==1``'s lanes 16..63 are surplus and must be
        masked. The bound is on the arange's *lane position* (``thread_idx()[axis]
        < length``), independent of the start offset -- ``< 16``, not the dim size
        (32) and not the addressed value range (8..23). The earlier canonical-only
        masking emitted no bound here, so those surplus lanes went out of bounds.
        """
        t = 4
        a = torch.randn(t, 8, 32, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(t, 8, 64, device=DEVICE, dtype=torch.bfloat16)
        code, (out_a, out_b) = code_and_output(
            cute_branch_noncanonical_free_arange, (a, b)
        )

        # Correctness: the written slice (positions 8..23) matches the reference,
        # and the sibling reduction output is intact (surplus lanes did not race).
        torch.testing.assert_close(
            out_a[:, :, 8:24], a.float()[:, :, 8:24] + 1.0, rtol=1e-2, atol=1e-2
        )
        b_ref = b.float()
        b_scale = torch.amax(torch.abs(b_ref), dim=-1, keepdim=True)
        torch.testing.assert_close(out_b, b_ref / b_scale, rtol=1e-2, atol=1e-2)

        # The branch reused axis 0, so the launch block stays 1-D.
        import re as _re

        block_match = _re.search(r"block=\((\d+),\s*(\d+),\s*(\d+)\)", code)
        self.assertIsNotNone(block_match, f"no launch block found in:\n{code}")
        assert block_match is not None
        bx, by, bz = (int(g) for g in block_match.groups())
        self.assertEqual((by, bz), (1, 1), f"expected 1-D block, got {(bx, by, bz)}")

        # The non-canonical arange's surplus lanes are bounded to its length (16),
        # NOT the dim size (32). Without the generalized bound (canonical-only),
        # this access emitted no lane mask and lanes 16..63 went out of bounds.
        self.assertIn(
            "cute.arch.thread_idx()[0]) < 16",
            code,
            "pid==1's non-canonical free arange (hl.arange(8, 24), length 16) is "
            "not bounded to its lane extent on the shared 64-wide axis",
        )
        self.assertNotIn(
            "cute.arch.thread_idx()[0]) < 32",
            code,
            "the non-canonical arange must be bounded by its length (16), not the "
            "dim size (32)",
        )


@helion.kernel(backend="cute", static_shapes=False)
def cute_branch_free_arange_reduction(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = a.size(0)
    ha = hl.specialize(a.shape[1])
    hc = hl.specialize(c.shape[1])
    hmax = hl.specialize(max(a.shape[1], c.shape[1]))
    da = hl.specialize(a.shape[2])
    db = hl.specialize(b.shape[2])
    dc = hl.specialize(c.shape[2])
    out_a = torch.empty_like(a, dtype=torch.float32)
    out_b = torch.empty_like(b, dtype=torch.float32)
    out_c = torch.empty_like(c, dtype=torch.float32)
    for pid, tile_t, tile_h in hl.grid([3, t, hmax]):
        if pid == 0:
            if tile_h < ha:
                ao = hl.arange(0, da)
                av = a[tile_t, tile_h, ao].to(torch.float32)
                asc = torch.rsqrt(torch.sum(av * av, dim=-1) / da + 1.0e-6)
                out_a[tile_t, tile_h, ao] = av * asc
        elif pid == 1:
            if tile_h < ha:
                bo = hl.arange(0, db)
                bv = b[tile_t, tile_h, bo].to(torch.float32)
                bsc = torch.amax(torch.abs(bv), dim=-1)
                out_b[tile_t, tile_h, bo] = bv / bsc
        elif pid == 2:
            if tile_h < hc:
                co = hl.arange(0, dc)
                cv = c[tile_t, tile_h, co].to(torch.float32)
                out_c[tile_t, tile_h, co] = cv + 1.0
    return out_a, out_b, out_c


@helion.kernel(backend="cute", static_shapes=False)
def cute_branch_noncanonical_free_arange(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    t = a.size(0)
    ha = hl.specialize(a.shape[1])
    hmax = hl.specialize(max(a.shape[1], b.shape[1]))
    db = hl.specialize(b.shape[2])
    out_a = torch.empty_like(a, dtype=torch.float32)
    out_b = torch.empty_like(b, dtype=torch.float32)
    for pid, tile_t, tile_h in hl.grid([2, t, hmax]):
        if pid == 0:
            if tile_h < ha:
                bo = hl.arange(0, db)
                bv = b[tile_t, tile_h, bo].to(torch.float32)
                bsc = torch.amax(torch.abs(bv), dim=-1)
                out_b[tile_t, tile_h, bo] = bv / bsc
        elif pid == 1:
            if tile_h < ha:
                # Non-canonical free arange: start=8, length=16 (positions 8..23).
                ao = hl.arange(8, 24)
                av = a[tile_t, tile_h, ao].to(torch.float32)
                out_a[tile_t, tile_h, ao] = av + 1.0
    return out_a, out_b


@helion.kernel(backend="cute")
def _cute_2d_tile_reduction_kernel(x: torch.Tensor) -> torch.Tensor:
    """2D reduction kernel: outer M-grid tile + inner N-reduction tile.

    Used by the thread-budget rejection tests and the warp-reduce
    heuristic registration test below.
    """
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _cute_fp8_gemm_skinny_m(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Skinny-M fp8 GEMM: full M kept resident, grid over N, reduce over K.

    Used by the thread-budget rejection tests: an explicit ``num_threads``
    split on the K (contraction) axis whose joint thread count exceeds the
    1024-thread CTA budget must be rejected rather than silently miscompiled.
    """
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_n in hl.tile(n):
        acc = hl.zeros([m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[:, tile_k], y[tile_k, tile_n], acc=acc)
        acc = acc * scale_a[:, tile_n] * scale_b[tile_n]
        out[:, tile_n] = acc.to(torch.bfloat16)
    return out


@onlyBackends(["cute"])
class TestCuteThreadBudgetRejection(TestCase):
    """The CuTe launcher raises ``BackendUnsupported`` when a config
    would force the launcher to silently truncate the joint thread
    count below what codegen committed to.

    The original bug: codegen for ``block_sizes=[8, 1024], num_threads=
    [0, 256]`` commits to an 8 * 256 = 2048-thread layout, but the
    launcher caps at MAX_THREADS_PER_BLOCK = 1024 → an axis is silently
    dropped and the kernel writes nan.  The guard (in
    ``CuteBackend.launcher_keyword_args``) rejects such configs cleanly
    so the autotuner doesn't record them as "fast but wrong".
    """

    def test_joint_thread_overflow_rejected(self) -> None:
        """A 2048-thread codegen budget on a launcher capped at 1024
        MUST raise ``BackendUnsupported`` instead of silently truncating.
        """
        x = torch.randn(4096, 1024, device=DEVICE, dtype=HALF_DTYPE)
        with pytest.raises(BackendUnsupported):
            code_and_output(
                _cute_2d_tile_reduction_kernel,
                (x,),
                block_sizes=[8, 1024],
                num_threads=[0, 256],
                cute_vector_widths=[1, 4],
            )

    def test_in_budget_multi_row_passes(self) -> None:
        """A multi-row config that DOES fit in 1024 threads must still
        compile and run cleanly — the rejection must be precise, not
        over-broad.
        """
        x = torch.randn(4096, 256, device=DEVICE, dtype=HALF_DTYPE)
        _, out = code_and_output(
            _cute_2d_tile_reduction_kernel,
            (x,),
            block_sizes=[2, 256],
            num_threads=[1, 32],  # 2 * 32 = 64 threads — within budget
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_skinny_fp8_gemm_overbudget_k_threads_rejected(self) -> None:
        """A skinny-M fp8 GEMM whose ``num_threads`` splits the K
        (contraction) axis so the joint CTA thread count exceeds 1024 MUST
        raise ``BackendUnsupported`` instead of silently miscompiling.

        Regression for the skinny-M fp8 miscompile: a config like
        ``block_sizes=[4, 16384], num_threads=[0, 1024]`` (block_n=4,
        K threaded by 1024) commits the grouped K-reduction to a
        4 * 1024 = 4096-thread span, but the launcher caps at 1024 and
        silently drops the K thread axis — the reduction then reads phantom
        lanes and the output came out ~1e4x too large. The truncation guard
        (in ``CuteBackend.launcher_keyword_args``) now also fires for matmul
        kernels that lowered ``hl.dot`` to the scalar grouped-reduce path
        (no ``cute.gemm`` intrinsic), rejecting such configs cleanly. The same
        over-budget decomposition is reachable with an in-range block_k too.
        """
        torch.manual_seed(0)
        m, k, n = 16, 4096, 512
        x = torch.randn(m, k, device=DEVICE).to(torch.float8_e4m3fn)
        y = torch.randn(k, n, device=DEVICE).to(torch.float8_e4m3fn)
        scale_a = torch.ones(m, n, device=DEVICE, dtype=torch.float32)
        scale_b = torch.ones(n, device=DEVICE, dtype=torch.float32)
        with pytest.raises(BackendUnsupported):
            code_and_output(
                _cute_fp8_gemm_skinny_m,
                (x, y, scale_a, scale_b),
                block_sizes=[4, 16384],
                cute_vector_widths=[8, 1],
                epilogue_subtile=2,
                num_threads=[0, 1024],
            )

    def test_skinny_fp8_gemm_in_budget_is_correct(self) -> None:
        """A valid skinny-M fp8 GEMM config (joint threads within budget)
        must compile and produce numerically correct output.

        Uses identity scales and range-filling fp8 inputs so the reference
        ``x.float() @ y.float()`` is O(1) and any miscompile is visible (the
        original bug was masked by degenerate near-zero benchmark inputs).
        """
        torch.manual_seed(0)
        m, k, n = 16, 4096, 512
        x = torch.randn(m, k, device=DEVICE).to(torch.float8_e4m3fn)
        y = torch.randn(k, n, device=DEVICE).to(torch.float8_e4m3fn)
        scale_a = torch.ones(m, n, device=DEVICE, dtype=torch.float32)
        scale_b = torch.ones(n, device=DEVICE, dtype=torch.float32)
        _, out = code_and_output(
            _cute_fp8_gemm_skinny_m,
            (x, y, scale_a, scale_b),
            block_sizes=[256, 64],
        )
        ref = x.float() @ y.float()
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=1e-1)


@onlyBackends(["cute"])
class TestCuteTileVecWarpReduceHeuristic(TestCase):
    """Pins the ``CuteTileVecWarpReduceHeuristic`` autotuner seed:
    ``block_sizes=[1, V*32]``, ``num_threads=[0, 32]``,
    ``cute_vector_widths=[1, V]`` — the warp-reduce config family that
    is the picked default for 2D reduction kernels with no rolled
    reduction.
    """

    def test_seed_compiles_to_warp_reduction(self) -> None:
        """The seed config must produce a working kernel that uses
        ``cute.arch.warp_reduction_*`` and not the shared-memory
        two-stage reduce.
        """
        x = torch.randn(4096, 6400, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _cute_2d_tile_reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertIn("cute.arch.warp_reduction_max", code)
        self.assertIn("cute.arch.warp_reduction_sum", code)
        # Block of 32 threads on the reduction axis — exactly one warp.
        self.assertIn("block=(32, 1, 1)", code)
        # Should NOT use the shared-memory two-stage reduce at this size.
        self.assertNotIn("_cute_grouped_reduce_shared_two_stage", code)

    def test_heuristic_class_is_registered(self) -> None:
        """The class must be discoverable and registered for the cute
        backend so the autotuner can use it as a seed.
        """
        from helion._compiler.autotuner_heuristics import HEURISTICS_BY_BACKEND
        from helion._compiler.autotuner_heuristics.cute import (
            CuteTileVecWarpReduceHeuristic,
        )

        self.assertIn(
            CuteTileVecWarpReduceHeuristic, HEURISTICS_BY_BACKEND.get("cute", ())
        )


@helion.kernel(backend="cute", autotune_effort="none")
def cute_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


class TestCuteConfigValuePriors(TestCase):
    """The cute backend supplies per-key value priors (the learned distribution
    that replaces the old hardcoded per-shape seeds); they bias the random half
    of the initial population toward the known-good 2-CTA matmul family."""

    def _assert_prior_choices(
        self,
        prior: Callable[[Any, int], object],
        fragment: Any,
        expected_values: Any,
    ) -> None:
        captured_values: tuple[object, ...] | None = None

        def fake_choices(values, *, weights, k):
            nonlocal captured_values
            self.assertEqual(k, 1)
            captured_values = tuple(values)
            return [values[0]]

        with patch("random.choices", side_effect=fake_choices):
            value = prior(fragment, 0)
        self.assertIsNotNone(captured_values)
        self.assertEqual(set(captured_values), expected_values)
        choices = fragment.search_choices or fragment.choices
        self.assertIn(value, choices)
        for candidate in captured_values:
            self.assertIn(candidate, choices)

    def test_priors_cover_the_template_keys(self) -> None:
        from helion._compiler.backend import CuteBackend

        priors = CuteBackend().config_value_priors(cast("Any", None))
        for key in (
            "indexing",
            "pid_type",
            "tcgen05_cluster_m",
            "tcgen05_ab_stages",
            "tcgen05_acc_stages",
            "tcgen05_c_stages",
            "tcgen05_num_epi_warps",
            "tcgen05_strategy",
            "tcgen05_persistence_model",
            "tcgen05_tvm_ffi_launch",
        ):
            self.assertIn(key, priors)

    def test_priors_wired_and_sampling_valid_for_matmul(self) -> None:
        from helion.autotuner.config_generation import ConfigGeneration

        x = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)
        gen = ConfigGeneration(cute_matmul.bind((x, x)).config_spec)
        # The cute priors engage on real matmul knobs for this kernel.
        engaged = set(gen._config_value_priors) & set(gen._key_to_flat_indices)
        self.assertIn("tcgen05_cluster_m", engaged)
        self.assertFalse(any(key.startswith("cute_flash_") for key in engaged))
        # Biased sampling must still produce only valid configs.
        self.assertEqual(len(gen.random_population(8)), 8)

    def test_priors_bias_indexing_toward_tma(self) -> None:
        from helion.autotuner.config_generation import ConfigGeneration

        x = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)
        gen = ConfigGeneration(cute_matmul.bind((x, x)).config_spec)
        (idx_slot,), _ = gen._key_to_flat_indices["indexing"]
        # ``indexing`` is one ListOf slot whose inner EnumFragment holds the
        # per-dimension choice; bias should favor tensor_descriptor per element.
        inner = getattr(gen.flat_spec[idx_slot], "inner", None)
        if "tensor_descriptor" not in getattr(inner, "choices", ()):
            self.skipTest("tensor_descriptor indexing not available for this spec")
        tma = total = 0
        for _ in range(40):
            for value in gen.biased_random_flat()[idx_slot]:
                total += 1
                tma += value == "tensor_descriptor"
        # Prior weights tensor_descriptor 4:1 over pointer; a strict majority of
        # the biased indexing slots should pick TMA.
        self.assertGreater(tma, total // 2)

    def test_flash_priors_bias_dense_hd64_fa4_values(self) -> None:
        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.cute_flash import FLASH_CLC_HEADS_PER_BATCH_KEY
        from helion._compiler.cute.cute_flash import FLASH_CLC_PDL_KEY
        from helion._compiler.cute.cute_flash import FLASH_CLC_STAGES_KEY
        from helion._compiler.cute.cute_flash import FLASH_CORR_REGS_KEY
        from helion._compiler.cute.cute_flash import FLASH_CORR_TILE_SIZE_KEY
        from helion._compiler.cute.cute_flash import FLASH_DISC_PIPE_KEY
        from helion._compiler.cute.cute_flash import FLASH_E2E_OFFSET0_KEY
        from helion._compiler.cute.cute_flash import FLASH_E2E_OFFSET_KEY
        from helion._compiler.cute.cute_flash import FLASH_E2E_SCHEDULE_KEY
        from helion._compiler.cute.cute_flash import FLASH_EPI_STG_GMEM_KEY
        from helion._compiler.cute.cute_flash import FLASH_EPI_STG_KEY
        from helion._compiler.cute.cute_flash import FLASH_EPI_STG_STORE_KEY
        from helion._compiler.cute.cute_flash import FLASH_EPI_TMA_KEY
        from helion._compiler.cute.cute_flash import FLASH_FIRST_LOAD_ORDER_KEY
        from helion._compiler.cute.cute_flash import FLASH_KV_ORDER_KEY
        from helion._compiler.cute.cute_flash import FLASH_KV_STAGE_KEY
        from helion._compiler.cute.cute_flash import FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS
        from helion._compiler.cute.cute_flash import FLASH_MASKED_E2E_SCHEDULE_KEY
        from helion._compiler.cute.cute_flash import FLASH_OTHER_REGS_KEY
        from helion._compiler.cute.cute_flash import FLASH_P_STORE_REP_KEY
        from helion._compiler.cute.cute_flash import FLASH_PACKED_REDUCE_KEY
        from helion._compiler.cute.cute_flash import FLASH_PERSISTENT_CTAS_PER_SM_KEY
        from helion._compiler.cute.cute_flash import FLASH_PERSISTENT_KEY
        from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
        from helion._compiler.cute.cute_flash import FLASH_PRECOMPUTE_QK_DESC_KEY
        from helion._compiler.cute.cute_flash import FLASH_RESCALE_CHUNK_COLS_KEY
        from helion._compiler.cute.cute_flash import FLASH_RESCALE_THRESHOLD_KEY
        from helion._compiler.cute.cute_flash import FLASH_S_LOAD_REP_KEY
        from helion._compiler.cute.cute_flash import FLASH_S_STAGE_KEY
        from helion._compiler.cute.cute_flash import FLASH_SKIP_RESCALE_STATS_KEY
        from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY
        from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_REGS_KEY
        from helion.autotuner.config_generation import ConfigGeneration

        q, k, v = (
            torch.empty(1, 1, 8192, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bound = cute_dense_attention.bind((q, k, v))
        self.assertTrue(bound.config_spec.cute_flash_search_enabled)
        bound.config_spec.compiler_seed_configs = []
        bound.config_spec.compiler_default_config = None
        priors = CuteBackend().config_value_priors(bound.config_spec)
        fragments = bound.config_spec._flat_fields()

        expected = {
            FLASH_S_STAGE_KEY: {2},
            FLASH_PIPELINE_FAMILY_KEY: {"fa4", "ws_overlap"},
            FLASH_PERSISTENT_KEY: {True},
            FLASH_PERSISTENT_CTAS_PER_SM_KEY: {1, 2, 3, 4},
            FLASH_E2E_SCHEDULE_KEY: {"8/2", "16/4"},
            FLASH_SOFTMAX_DISC_KEY: {False, True},
            FLASH_DISC_PIPE_KEY: {2, 3, 4},
            FLASH_P_STORE_REP_KEY: {16, 32},
            FLASH_S_LOAD_REP_KEY: {16, 32},
            FLASH_PRECOMPUTE_QK_DESC_KEY: {False, True},
            FLASH_FIRST_LOAD_ORDER_KEY: {0, 1, 2, 3, 4},
            FLASH_KV_ORDER_KEY: {"ascending", "descending"},
            FLASH_RESCALE_THRESHOLD_KEY: {8.0},
            FLASH_SKIP_RESCALE_STATS_KEY: {False, True},
            FLASH_RESCALE_CHUNK_COLS_KEY: {16, 32},
            FLASH_CORR_REGS_KEY: {64},
            FLASH_OTHER_REGS_KEY: {32, 40, 48, 56, 64, 80},
            FLASH_CORR_TILE_SIZE_KEY: {8, 16, 32},
            FLASH_EPI_TMA_KEY: {False, True},
            FLASH_EPI_STG_KEY: {False, True},
            FLASH_EPI_STG_STORE_KEY: {"slice", "whole"},
            FLASH_EPI_STG_GMEM_KEY: {"stage", "pair"},
            FLASH_SOFTMAX_REGS_KEY: {184, 200},
            FLASH_KV_STAGE_KEY: {2, 3},
            FLASH_PACKED_REDUCE_KEY: {True},
            FLASH_CLC_HEADS_PER_BATCH_KEY: {0, 32},
            FLASH_CLC_PDL_KEY: {False, True},
            FLASH_CLC_STAGES_KEY: {2, 3},
            FLASH_E2E_OFFSET_KEY: {0, 1, 2, 3},
            FLASH_E2E_OFFSET0_KEY: {0, 1, 2, 3},
        }
        for key, values in expected.items():
            self.assertIn(key, priors)
            self._assert_prior_choices(priors[key], fragments[key], values)

        bound.config_spec._cute_flash_num_kv = 256
        very_long_priors = CuteBackend().config_value_priors(bound.config_spec)
        very_long_fragments = bound.config_spec._flat_fields()
        self._assert_prior_choices(
            very_long_priors[FLASH_RESCALE_THRESHOLD_KEY],
            very_long_fragments[FLASH_RESCALE_THRESHOLD_KEY],
            {8.0, 12.0},
        )
        bound.config_spec._cute_flash_num_kv = 384
        bound.config_spec._cute_flash_dtype = torch.bfloat16
        bf16_priors = CuteBackend().config_value_priors(bound.config_spec)
        bf16_fragments = bound.config_spec._flat_fields()
        self._assert_prior_choices(
            bf16_priors[FLASH_RESCALE_THRESHOLD_KEY],
            bf16_fragments[FLASH_RESCALE_THRESHOLD_KEY],
            {8.0, 16.0, 32.0},
        )
        for key, values in {
            FLASH_PIPELINE_FAMILY_KEY: {"fa4", "ws_overlap", "fa4_2cta"},
            FLASH_E2E_SCHEDULE_KEY: {"8/2", "16/4"},
            FLASH_E2E_OFFSET_KEY: {0},
            FLASH_E2E_OFFSET0_KEY: {1},
            FLASH_KV_STAGE_KEY: {2, 3},
            FLASH_PERSISTENT_CTAS_PER_SM_KEY: {1},
            FLASH_SOFTMAX_DISC_KEY: {False},
            FLASH_DISC_PIPE_KEY: {1},
            FLASH_P_STORE_REP_KEY: {16},
            FLASH_S_LOAD_REP_KEY: {32},
            FLASH_PRECOMPUTE_QK_DESC_KEY: {False},
            FLASH_FIRST_LOAD_ORDER_KEY: {0},
            FLASH_KV_ORDER_KEY: {"descending"},
            FLASH_SKIP_RESCALE_STATS_KEY: {False},
            FLASH_RESCALE_CHUNK_COLS_KEY: {8},
            FLASH_SOFTMAX_REGS_KEY: {192, 200},
            FLASH_CORR_REGS_KEY: {64},
            FLASH_OTHER_REGS_KEY: {40},
            FLASH_CORR_TILE_SIZE_KEY: {8, 16},
            FLASH_EPI_TMA_KEY: {True},
            FLASH_EPI_STG_KEY: {False},
            FLASH_EPI_STG_STORE_KEY: {"slice"},
            FLASH_EPI_STG_GMEM_KEY: {"stage"},
            FLASH_PACKED_REDUCE_KEY: {True},
            FLASH_CLC_HEADS_PER_BATCH_KEY: {0},
            FLASH_CLC_PDL_KEY: {False},
            FLASH_CLC_STAGES_KEY: {1},
        }.items():
            self._assert_prior_choices(
                very_long_priors[key],
                very_long_fragments[key],
                values,
            )
        bound.config_spec._cute_flash_num_kv = 512
        staged_priors = CuteBackend().config_value_priors(bound.config_spec)
        staged_fragments = bound.config_spec._flat_fields()
        for key, values in {
            FLASH_EPI_TMA_KEY: {False},
            FLASH_EPI_STG_KEY: {True},
            FLASH_EPI_STG_STORE_KEY: {"slice"},
            FLASH_EPI_STG_GMEM_KEY: {"stage"},
        }.items():
            self._assert_prior_choices(
                staged_priors[key], staged_fragments[key], values
            )
        bound.config_spec._cute_flash_num_kv = 1024
        tma_priors = CuteBackend().config_value_priors(bound.config_spec)
        tma_fragments = bound.config_spec._flat_fields()
        for key, values in {
            FLASH_EPI_TMA_KEY: {True},
            FLASH_EPI_STG_KEY: {False},
            FLASH_EPI_STG_STORE_KEY: {"slice"},
            FLASH_EPI_STG_GMEM_KEY: {"stage"},
        }.items():
            self._assert_prior_choices(tma_priors[key], tma_fragments[key], values)
        bound.config_spec._cute_flash_num_kv = 2048
        bound.config_spec._cute_flash_dtype = torch.float16
        two_cta_priors = CuteBackend().config_value_priors(bound.config_spec)
        two_cta_fragments = bound.config_spec._flat_fields()
        self._assert_prior_choices(
            two_cta_priors[FLASH_PIPELINE_FAMILY_KEY],
            two_cta_fragments[FLASH_PIPELINE_FAMILY_KEY],
            {"fa4", "ws_overlap", "fa4_2cta"},
        )
        bound.config_spec._cute_flash_num_kv = 64

        gen = ConfigGeneration(bound.config_spec)
        with patch(
            "random.choices", side_effect=lambda values, weights, k: [values[0]]
        ):
            biased_config = gen.unflatten(gen.biased_random_flat()).config
        self.assertEqual(biased_config[FLASH_PIPELINE_FAMILY_KEY], "fa4")
        self.assertEqual(biased_config[FLASH_E2E_SCHEDULE_KEY], "8/2")
        self.assertEqual(biased_config[FLASH_MASKED_E2E_SCHEDULE_KEY], "inherit")
        self.assertEqual(biased_config[FLASH_KV_STAGE_KEY], 2)
        self.assertEqual(biased_config[FLASH_PERSISTENT_CTAS_PER_SM_KEY], 1)
        self.assertEqual(biased_config[FLASH_E2E_OFFSET_KEY], 2)
        self.assertEqual(biased_config[FLASH_E2E_OFFSET0_KEY], 2)
        self.assertTrue(biased_config[FLASH_SOFTMAX_DISC_KEY])
        self.assertEqual(biased_config[FLASH_P_STORE_REP_KEY], 16)
        self.assertEqual(biased_config[FLASH_S_LOAD_REP_KEY], 32)
        self.assertFalse(biased_config[FLASH_PRECOMPUTE_QK_DESC_KEY])
        self.assertEqual(biased_config[FLASH_FIRST_LOAD_ORDER_KEY], 0)
        self.assertEqual(biased_config[FLASH_KV_ORDER_KEY], "ascending")
        self.assertEqual(biased_config[FLASH_OTHER_REGS_KEY], 48)
        self.assertTrue(biased_config[FLASH_EPI_TMA_KEY])
        self.assertFalse(biased_config[FLASH_EPI_STG_KEY])
        self.assertEqual(biased_config[FLASH_CORR_TILE_SIZE_KEY], 16)
        self.assertFalse(biased_config[FLASH_SKIP_RESCALE_STATS_KEY])
        self.assertEqual(biased_config[FLASH_RESCALE_CHUNK_COLS_KEY], 16)
        for legacy_key in FLASH_LEGACY_STRUCTURAL_CONFIG_KEYS:
            self.assertNotIn(legacy_key, biased_config)

    def test_flash_priors_bias_causal_hd64_shape_family_values(self) -> None:
        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.cute_flash import FLASH_CAUSAL_KV_ORDER_KEY
        from helion._compiler.cute.cute_flash import FLASH_CAUSAL_LOOP_SPLIT_KEY
        from helion._compiler.cute.cute_flash import FLASH_CAUSAL_LPT_SWIZZLE_KEY
        from helion._compiler.cute.cute_flash import FLASH_DISC_PIPE_KEY
        from helion._compiler.cute.cute_flash import FLASH_E2E_OFFSET0_KEY
        from helion._compiler.cute.cute_flash import FLASH_E2E_OFFSET_KEY
        from helion._compiler.cute.cute_flash import FLASH_E2E_SCHEDULE_KEY
        from helion._compiler.cute.cute_flash import FLASH_EPI_TMA_KEY
        from helion._compiler.cute.cute_flash import FLASH_KV_STAGE_KEY
        from helion._compiler.cute.cute_flash import FLASH_MASKED_E2E_SCHEDULE_KEY
        from helion._compiler.cute.cute_flash import FLASH_PACKED_REDUCE_KEY
        from helion._compiler.cute.cute_flash import FLASH_PERSISTENT_KEY
        from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
        from helion._compiler.cute.cute_flash import FLASH_ROLE_MAP_KEY
        from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_REGS_KEY
        from helion.autotuner.config_generation import ConfigGeneration

        q, k, v = (
            torch.empty(1, 1, 8192, 64, dtype=torch.float16, device=DEVICE)
            for _ in range(3)
        )
        bound = cute_causal_attention.bind((q, k, v))
        self.assertTrue(bound.config_spec.cute_flash_search_enabled)
        bound.config_spec.compiler_seed_configs = []
        bound.config_spec.compiler_default_config = None
        # Exercise the causal shape family through the recorded flash fact without
        # allocating a separate long-sequence Q/K/V triple for every case.
        for num_kv in (32, 64, 128, 256, 512):
            with self.subTest(num_kv=num_kv):
                bound.config_spec._cute_flash_num_kv = num_kv
                priors = CuteBackend().config_value_priors(bound.config_spec)
                fragments = bound.config_spec._flat_fields()
                expected = {
                    FLASH_PIPELINE_FAMILY_KEY: {"fa4", "ws_overlap"},
                    FLASH_PERSISTENT_KEY: {False},
                    FLASH_KV_STAGE_KEY: {2, 3, 4, 6, 8, 10},
                    FLASH_E2E_SCHEDULE_KEY: {"16/4", "8/2", "xu"},
                    FLASH_MASKED_E2E_SCHEDULE_KEY: {
                        "inherit",
                        "xu",
                        "16/4",
                        "8/2",
                    },
                    FLASH_PACKED_REDUCE_KEY: {True},
                    FLASH_EPI_TMA_KEY: {False, True},
                    FLASH_ROLE_MAP_KEY: {"fa4", "helion"},
                    FLASH_CAUSAL_LOOP_SPLIT_KEY: {False, True},
                    FLASH_CAUSAL_LPT_SWIZZLE_KEY: {0, 1, 2, 4, 8, 16},
                    FLASH_CAUSAL_KV_ORDER_KEY: {"ascending", "descending"},
                    FLASH_DISC_PIPE_KEY: {2, 3, 4},
                    FLASH_SOFTMAX_REGS_KEY: {184, 192, 200},
                    FLASH_E2E_OFFSET_KEY: set(range(16)),
                    FLASH_E2E_OFFSET0_KEY: {0, 1, 2, 3, 4, 8, 11},
                }
                for key, values in expected.items():
                    self.assertIn(key, priors)
                    self._assert_prior_choices(priors[key], fragments[key], values)

        bound.config_spec._cute_flash_num_kv = 96
        unsupported_priors = CuteBackend().config_value_priors(bound.config_spec)
        self.assertNotIn(FLASH_CAUSAL_LPT_SWIZZLE_KEY, unsupported_priors)
        self.assertNotIn(FLASH_E2E_OFFSET_KEY, unsupported_priors)

        bound.config_spec._cute_flash_num_kv = 512
        gen = ConfigGeneration(bound.config_spec)
        with patch(
            "random.choices", side_effect=lambda values, weights, k: [values[0]]
        ):
            population = [
                gen.unflatten(flat).config for flat in gen.random_population_flat(4)
            ]
        self.assertEqual(len(population), 4)
        biased_config = next(
            config
            for config in population
            if config[FLASH_CAUSAL_LPT_SWIZZLE_KEY] == 1
            and config[FLASH_CAUSAL_KV_ORDER_KEY] == "descending"
            and config[FLASH_MASKED_E2E_SCHEDULE_KEY] == "16/4"
            and config[FLASH_ROLE_MAP_KEY] == "helion"
        )
        self.assertEqual(biased_config[FLASH_CAUSAL_LPT_SWIZZLE_KEY], 1)
        self.assertEqual(biased_config[FLASH_CAUSAL_KV_ORDER_KEY], "descending")
        self.assertEqual(biased_config[FLASH_MASKED_E2E_SCHEDULE_KEY], "16/4")
        self.assertEqual(biased_config[FLASH_ROLE_MAP_KEY], "helion")
        self.assertFalse(biased_config[FLASH_EPI_TMA_KEY])
        self.assertEqual(biased_config[FLASH_E2E_OFFSET_KEY], 9)
        self.assertEqual(biased_config[FLASH_E2E_OFFSET0_KEY], 3)
        self.assertEqual(biased_config[FLASH_DISC_PIPE_KEY], 4)
        self.assertEqual(biased_config[FLASH_SOFTMAX_REGS_KEY], 184)


class TestCuteBackendRequirements(TestCase):
    """The cute backend hard-requires CuTe DSL >= 4.5.1, apache-tvm-ffi, and
    CUDA >= 13, enforced up front via ``CuteBackend.validate_environment``.
    This module is ``importorskip``-gated on cutlass, so the environment under
    test already satisfies the requirements (the gate must pass here).
    """

    def test_requirements_satisfied_in_this_environment(self) -> None:
        from helion._compiler.cute.cutedsl_compat import _cute_backend_requirement_error

        self.assertIsNone(_cute_backend_requirement_error())

    def test_check_does_not_raise_when_satisfied(self) -> None:
        from helion._compiler.cute.cutedsl_compat import check_cute_backend_requirements

        check_cute_backend_requirements()  # must not raise in this environment

    def test_validate_environment_passes(self) -> None:
        from helion._compiler.backend import CuteBackend

        CuteBackend().validate_environment()  # must not raise in this environment

    def test_math_min_max_version_guard(self) -> None:
        from helion._compiler.cute import cutedsl_compat

        self.addCleanup(cutedsl_compat.cute_math_min_max_available.cache_clear)
        for version, expected in (("4.5.1", False), ("4.6.0", True), ("4.6.1", True)):
            with self.subTest(version=version):
                cutedsl_compat.cute_math_min_max_available.cache_clear()
                with patch.object(
                    cutedsl_compat.importlib.metadata,
                    "version",
                    return_value=version,
                ):
                    self.assertEqual(
                        cutedsl_compat.cute_math_min_max_available(), expected
                    )

    def test_unmet_requirement_raises_with_actionable_message(self) -> None:
        from helion._compiler.cute import cutedsl_compat

        with (
            patch.object(
                cutedsl_compat,
                "_cute_backend_requirement_error",
                return_value="the apache-tvm-ffi package is required (simulated)",
            ),
            self.assertRaises(CuteBackendUnavailable) as ctx,
        ):
            cutedsl_compat.check_cute_backend_requirements()
        message = str(ctx.exception)
        self.assertIn("apache-tvm-ffi package is required (simulated)", message)
        # The fixed tail names all three requirements so the user knows the set.
        self.assertIn("nvidia-cutlass-dsl >= 4.5.1", message)
        self.assertIn("CUDA >= 13", message)
