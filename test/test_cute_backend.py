from __future__ import annotations

import importlib
import os
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion.exc import BackendUnsupported
from helion.exc import CuteBackendUnavailable
import helion.language as hl
from helion.runtime import _cute_cluster_shape
from helion.runtime import _cute_cluster_shape_from_wrapper_plans
from helion.runtime import _ensure_cute_dsl_arch_env
from helion.runtime import _get_compiled_cute_launcher
from helion.runtime import default_cute_launcher

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")

get_cute_mma_support = importlib.import_module(
    "helion._compiler.cute.mma_support"
).get_cute_mma_support
_cute_grouped_reduce_shared_tree = importlib.import_module(
    "helion._compiler.cute.reduce_helpers"
)._cute_grouped_reduce_shared_tree


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

    def test_baddbmm_falls_back_from_mma(self) -> None:
        args = (
            torch.randn(2, 16, 64, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 64, 8, device=DEVICE, dtype=HALF_DTYPE),
            torch.randn(2, 16, 8, device=DEVICE, dtype=torch.float32),
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
            "index type: <class 'slice'>",
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

    def test_cute_launcher_cache_key_includes_wrapper_plans(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        schema_key = (("tensor", 2, "float32"),)
        block = (32, 1, 1)
        created: list[str] = []

        def make_wrapper(*_args: object) -> str:
            created.append("wrapper")
            return f"wrapper-{len(created)}"

        with patch("helion.runtime._create_cute_wrapper", side_effect=make_wrapper):
            cute_kernel._helion_cute_wrapper_plans = [{"kind": "plan-a"}]
            wrapper_a0 = _get_compiled_cute_launcher(cute_kernel, schema_key, block)
            wrapper_a1 = _get_compiled_cute_launcher(cute_kernel, schema_key, block)
            cute_kernel._helion_cute_wrapper_plans = [{"kind": "plan-b"}]
            wrapper_b = _get_compiled_cute_launcher(cute_kernel, schema_key, block)

        self.assertEqual(wrapper_a0, wrapper_a1)
        self.assertNotEqual(wrapper_a0, wrapper_b)

    def test_cute_launcher_cache_key_includes_cluster_shape(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        schema_key = (("tensor", 2, "float32"),)
        block = (32, 1, 1)
        created: list[str] = []

        def make_wrapper(*_args: object) -> str:
            created.append("wrapper")
            return f"wrapper-{len(created)}"

        with patch("helion.runtime._create_cute_wrapper", side_effect=make_wrapper):
            cute_kernel._helion_cute_wrapper_plans = [{"kind": "plan-a"}]
            cute_kernel._helion_cute_cluster_shape = (1, 1, 1)
            wrapper_a = _get_compiled_cute_launcher(cute_kernel, schema_key, block)
            cute_kernel._helion_cute_cluster_shape = (2, 1, 1)
            wrapper_b = _get_compiled_cute_launcher(cute_kernel, schema_key, block)

        self.assertNotEqual(wrapper_a, wrapper_b)

    def test_cute_launcher_cache_key_includes_compile_options(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        schema_key = (("tensor", 2, "float32"),)
        block = (32, 1, 1)
        created: list[str] = []

        def make_wrapper(*_args: object) -> str:
            created.append("wrapper")
            return f"wrapper-{len(created)}"

        with patch("helion.runtime._create_cute_wrapper", side_effect=make_wrapper):
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
            patch("helion.runtime._create_cute_wrapper", return_value="jit-wrapper"),
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
            patch("helion.runtime._create_cute_wrapper", return_value="jit-wrapper"),
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
        ) -> tuple[tuple[tuple[object, ...], ...], tuple[object, ...]]:
            build_calls.append((args, grid))
            return (("scalar", "int"),), ("launch-arg", *args, *grid)

        with (
            patch("helion.runtime._build_cute_schema_and_args", side_effect=fake_build),
            patch(
                "helion.runtime._get_compiled_cute_launcher",
                return_value=FakeCompiled(),
            ),
        ):
            first = default_cute_launcher(cute_kernel, (2,), 7, block=(32, 1, 1))
            second = default_cute_launcher(cute_kernel, (2,), 7, block=(32, 1, 1))
            third = default_cute_launcher(cute_kernel, (2,), 8, block=(32, 1, 1))

        self.assertEqual(build_calls, [((7,), (2, 1, 1)), ((8,), (2, 1, 1))])
        self.assertEqual(
            launched_args,
            [
                ("launch-arg", 7, 2, 1, 1),
                ("launch-arg", 7, 2, 1, 1),
                ("launch-arg", 8, 2, 1, 1),
            ],
        )
        self.assertEqual(first, ("launched", ("launch-arg", 7, 2, 1, 1)))
        self.assertEqual(second, first)
        self.assertEqual(third, ("launched", ("launch-arg", 8, 2, 1, 1)))

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
        ) -> tuple[tuple[tuple[object, ...], ...], tuple[object, ...]]:
            build_calls.append(args)
            return (("scalar", "float"),), (f"float-{len(build_calls)}",)

        with (
            patch("helion.runtime._build_cute_schema_and_args", side_effect=fake_build),
            patch(
                "helion.runtime._get_compiled_cute_launcher",
                return_value=FakeCompiled(),
            ),
        ):
            positive = default_cute_launcher(cute_kernel, (1,), 0.0)
            negative = default_cute_launcher(cute_kernel, (1,), -0.0)

        self.assertEqual(build_calls, [(0.0,), (-0.0,)])
        self.assertEqual(launched_args, [("float-1",), ("float-2",)])
        self.assertEqual(positive, ("launched", ("float-1",)))
        self.assertEqual(negative, ("launched", ("float-2",)))

    def test_cute_launcher_sets_arch_env_only_before_first_compile(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                return ("launched", args)

        with (
            patch(
                "helion.runtime._build_cute_schema_and_args",
                return_value=((("scalar", "int"),), ("launch-arg",)),
            ),
            patch("helion.runtime._create_cute_wrapper", return_value="jit-wrapper"),
            patch("helion.runtime._ensure_cute_dsl_arch_env") as ensure_arch,
            patch("cutlass.cute.compile", return_value=FakeCompiled()),
        ):
            first = default_cute_launcher(cute_kernel, (1,), 7, block=(32, 1, 1))
            second = default_cute_launcher(cute_kernel, (1,), 7, block=(32, 1, 1))

        self.assertEqual(ensure_arch.call_count, 1)
        self.assertEqual(first, ("launched", ("launch-arg",)))
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
        ) -> str:
            created_schema_keys.append(schema_key)
            return f"jit-wrapper-{len(created_schema_keys)}"

        with (
            patch(
                "helion.runtime._create_cute_wrapper", side_effect=fake_create_wrapper
            ),
            patch("helion.runtime._ensure_cute_dsl_arch_env"),
            patch("cutlass.cute.compile", return_value=FakeCompiled()),
        ):
            positive = default_cute_launcher(cute_kernel, (1,), 0.0)
            negative = default_cute_launcher(cute_kernel, (1,), -0.0)

        self.assertEqual(len(created_schema_keys), 2)
        self.assertNotEqual(created_schema_keys[0], created_schema_keys[1])
        self.assertEqual(positive[0], "launched")
        self.assertEqual(positive[1][:3], (1, 1, 1))
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
        ) -> tuple[tuple[tuple[object, ...], ...], tuple[object, ...]]:
            build_calls.append(cast("torch.Tensor", args[0]).data_ptr())
            return (("tensor", "torch.float32", 1),), (f"ptr-{len(build_calls)}",)

        with (
            patch("helion.runtime._build_cute_schema_and_args", side_effect=fake_build),
            patch(
                "helion.runtime._get_compiled_cute_launcher",
                return_value=FakeCompiled(),
            ),
        ):
            first = default_cute_launcher(cute_kernel, (1,), tensor)
            second = default_cute_launcher(cute_kernel, (1,), tensor)
            third = default_cute_launcher(cute_kernel, (1,), other_tensor)

        self.assertEqual(build_calls, [tensor.data_ptr(), other_tensor.data_ptr()])
        self.assertEqual(launched_args, [("ptr-1",), ("ptr-1",), ("ptr-2",)])
        self.assertEqual(first, second)
        self.assertEqual(third, ("launched", ("ptr-2",)))

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
