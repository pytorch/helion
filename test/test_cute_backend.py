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
from helion.runtime import _create_cute_direct_entry
from helion.runtime import _cute_cluster_shape
from helion.runtime import _cute_cluster_shape_from_wrapper_plans
from helion.runtime import _direct_entry_clustered_grid_k
from helion.runtime import _ensure_cute_dsl_arch_env
from helion.runtime import _get_compiled_cute_direct_entry_launcher
from helion.runtime import _get_compiled_cute_launcher
from helion.runtime import _validate_target1_direct_entry_args
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

    def test_cute_launcher_uses_target1_direct_entry_plan(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        cute_kernel._helion_cute_direct_entry_plans = [
            {
                "kind": "tcgen05_target1_direct_entry",
                "lhs_idx": 0,
                "rhs_idx": 1,
                "d_idx": 2,
            }
        ]
        x = torch.empty((1024, 1024), device=DEVICE, dtype=torch.bfloat16)
        y = torch.empty((1024, 4096), device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty((1024, 4096), device=DEVICE, dtype=torch.bfloat16)
        launched_args: list[tuple[object, ...]] = []

        class FakeDirectLauncher:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                launched_args.append(args)
                return ("direct", args)

        with (
            patch(
                "helion.runtime._get_compiled_cute_direct_entry_launcher",
                return_value=FakeDirectLauncher(),
            ) as direct_launcher,
            patch("helion.runtime._build_cached_cute_schema_and_args") as build_schema,
            patch("helion.runtime._get_compiled_cute_launcher") as wrapper_launcher,
        ):
            result = default_cute_launcher(
                cute_kernel,
                (2, 1, 64),
                x,
                y,
                out,
                block=(256, 1, 1),
                cute_compile_options="--enable-tvm-ffi",
            )

        direct_launcher.assert_called_once()
        build_schema.assert_not_called()
        wrapper_launcher.assert_not_called()
        self.assertEqual(launched_args, [(x, y, out)])
        self.assertEqual(result, ("direct", (x, y, out)))

    def test_cute_launcher_direct_entry_requires_tvm_ffi_option(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        cute_kernel._helion_cute_direct_entry_plans = [
            {
                "kind": "tcgen05_target1_direct_entry",
                "lhs_idx": 0,
                "rhs_idx": 1,
                "d_idx": 2,
            }
        ]
        x = torch.empty((1024, 1024), device=DEVICE, dtype=torch.bfloat16)
        y = torch.empty((1024, 4096), device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty((1024, 4096), device=DEVICE, dtype=torch.bfloat16)

        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            "explicit CuTe compile options",
        ):
            default_cute_launcher(
                cute_kernel,
                (2, 1, 64),
                x,
                y,
                out,
                block=(256, 1, 1),
            )

    def test_cute_direct_entry_launcher_compiles_with_fake_tensors(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        direct_plan: dict[str, object] = {
            "kind": "tcgen05_target1_direct_entry",
            "lhs_idx": 0,
            "rhs_idx": 1,
            "d_idx": 2,
            "bm": 256,
            "bn": 256,
            "bk": 64,
            "cluster_m": 2,
            "cluster_n": 1,
            "ab_stage_count": 3,
            "c_stage_count": 2,
            "input_dtype": "cutlass.BFloat16",
            "output_dtype": "cutlass.BFloat16",
            "validated_shape": [1024, 4096, 1024],
        }
        x = torch.empty((1024, 1024), device=DEVICE, dtype=torch.bfloat16)
        y = torch.empty((1024, 4096), device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty((1024, 4096), device=DEVICE, dtype=torch.bfloat16)
        fake_args = ("fake-x", "fake-y", "fake-out")
        compiled_calls: list[tuple[object, tuple[object, ...], str | None]] = []
        launched_args: list[tuple[object, ...]] = []

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                launched_args.append(args)
                return ("direct", args)

        def fake_compile(
            jit_func: object,
            *args: object,
            options: str | None = None,
        ) -> FakeCompiled:
            compiled_calls.append((jit_func, args, options))
            return FakeCompiled()

        with (
            patch("helion.runtime._ensure_cute_dsl_arch_env"),
            patch(
                "helion.runtime._create_cute_direct_entry",
                return_value="jit-direct-entry",
            ),
            patch(
                "helion.runtime._make_cute_direct_entry_fake_tensor",
                side_effect=fake_args,
            ),
            patch("cutlass.cute.compile", side_effect=fake_compile),
        ):
            launcher = _get_compiled_cute_direct_entry_launcher(
                cute_kernel,
                direct_plan,
                (x, y, out),
                (2, 1, 64),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
            first = launcher(x, y, out)
            second = launcher(x, y, out)

        self.assertEqual(
            compiled_calls,
            [("jit-direct-entry", fake_args, "--enable-tvm-ffi")],
        )
        self.assertEqual(launched_args, [(x, y, out), (x, y, out)])
        self.assertEqual(first, ("direct", (x, y, out)))
        self.assertEqual(second, first)

    def test_cute_direct_entry_launcher_uses_generated_entry(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        direct_plan: dict[str, object] = {
            "kind": "tcgen05_target1_direct_entry",
            "lhs_idx": 0,
            "rhs_idx": 1,
            "d_idx": 2,
            "bm": 256,
            "bn": 256,
            "bk": 64,
            "cluster_m": 2,
            "cluster_n": 1,
            "ab_stage_count": 3,
            "c_stage_count": 2,
            "input_dtype": "cutlass.BFloat16",
            "output_dtype": "cutlass.BFloat16",
            "validated_shape": [1024, 4096, 1024],
        }
        cute_kernel._helion_cute_generated_direct_entry = "generated-direct-entry"
        x = torch.empty((1024, 1024), device=DEVICE, dtype=torch.bfloat16)
        y = torch.empty((1024, 4096), device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty((1024, 4096), device=DEVICE, dtype=torch.bfloat16)
        fake_args = ("fake-x", "fake-y", "fake-out")
        compiled_calls: list[tuple[object, tuple[object, ...], str | None]] = []

        class FakeCompiled:
            def __call__(self, *args: object) -> tuple[str, tuple[object, ...]]:
                return ("direct", args)

        def fake_compile(
            jit_func: object,
            *args: object,
            options: str | None = None,
        ) -> FakeCompiled:
            compiled_calls.append((jit_func, args, options))
            return FakeCompiled()

        with (
            patch("helion.runtime._ensure_cute_dsl_arch_env"),
            patch(
                "helion.runtime._patch_cutlass_jit_shutdown_unload"
            ) as patch_shutdown,
            patch(
                "helion.runtime._create_cute_direct_entry",
                side_effect=AssertionError("runtime direct entry fallback used"),
            ),
            patch(
                "helion.runtime._make_cute_direct_entry_fake_tensor",
                side_effect=fake_args,
            ),
            patch("cutlass.cute.compile", side_effect=fake_compile),
        ):
            launcher = _get_compiled_cute_direct_entry_launcher(
                cute_kernel,
                direct_plan,
                (x, y, out),
                (128, 1, 1),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
            result = launcher(x, y, out)

        patch_shutdown.assert_called_once()
        self.assertEqual(
            compiled_calls,
            [("generated-direct-entry", fake_args, "--enable-tvm-ffi")],
        )
        self.assertEqual(result, ("direct", (x, y, out)))

    def test_cute_direct_entry_cache_key_includes_cluster_shape(self) -> None:
        cute_kernel = type("DummyCuteKernel", (), {})()
        direct_plan: dict[str, object] = {
            "kind": "tcgen05_target1_direct_entry",
            "lhs_idx": 0,
            "rhs_idx": 1,
            "d_idx": 2,
            "bm": 256,
            "bn": 256,
            "bk": 64,
            "cluster_m": 2,
            "cluster_n": 1,
            "ab_stage_count": 3,
            "c_stage_count": 2,
            "input_dtype": "cutlass.BFloat16",
            "output_dtype": "cutlass.BFloat16",
            "validated_shape": [1024, 4096, 1024],
        }
        cute_kernel._helion_cute_wrapper_plans = [
            {
                "kind": "tcgen05_ab_tma",
                "kernel_args": [
                    "tma_atom_a",
                    "tma_tensor_a",
                    "tma_atom_b",
                    "tma_tensor_b",
                ],
            },
            {
                "kind": "tcgen05_d_tma",
                "kernel_args": [
                    "tma_store_atom",
                    "tma_store_tensor",
                ],
            },
        ]
        x = torch.empty((1024, 1024), device=DEVICE, dtype=torch.bfloat16)
        y = torch.empty((1024, 4096), device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty((1024, 4096), device=DEVICE, dtype=torch.bfloat16)
        fake_args = ("fake-x", "fake-y", "fake-out")
        created: list[object] = []

        def make_direct_entry(*_args: object) -> object:
            entry = f"jit-direct-entry-{len(created)}"
            created.append(entry)
            return entry

        with (
            patch("helion.runtime._ensure_cute_dsl_arch_env"),
            patch(
                "helion.runtime._create_cute_direct_entry",
                side_effect=make_direct_entry,
            ),
            patch(
                "helion.runtime._make_cute_direct_entry_fake_tensor",
                side_effect=fake_args * 3,
            ),
        ):
            cute_kernel._helion_cute_cluster_shape = (1, 1, 1)
            launcher_a0 = _get_compiled_cute_direct_entry_launcher(
                cute_kernel,
                direct_plan,
                (x, y, out),
                (2, 1, 64),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
            launcher_a1 = _get_compiled_cute_direct_entry_launcher(
                cute_kernel,
                direct_plan,
                (x, y, out),
                (2, 1, 64),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
            cute_kernel._helion_cute_cluster_shape = (2, 1, 1)
            launcher_b = _get_compiled_cute_direct_entry_launcher(
                cute_kernel,
                direct_plan,
                (x, y, out),
                (2, 1, 64),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )

        self.assertIs(launcher_a0, launcher_a1)
        self.assertIsNot(launcher_a0, launcher_b)
        self.assertEqual(created, ["jit-direct-entry-0", "jit-direct-entry-1"])

    def test_cute_direct_entry_rejects_stale_wrapper_metadata(self) -> None:
        direct_plan: dict[str, object] = {
            "kind": "tcgen05_target1_direct_entry",
            "lhs_idx": 0,
            "rhs_idx": 1,
            "d_idx": 2,
            "bm": 256,
            "bn": 256,
            "bk": 64,
            "cluster_m": 2,
            "cluster_n": 1,
            "ab_stage_count": 3,
            "c_stage_count": 2,
            "input_dtype": "cutlass.BFloat16",
            "output_dtype": "cutlass.BFloat16",
            "ab_kernel_args": [
                "tma_atom_a",
                "tma_tensor_a",
                "tma_atom_b",
                "tma_tensor_b",
            ],
            "d_kernel_args": [
                "tma_store_atom",
                "tma_store_tensor",
            ],
        }
        ab_plan: dict[str, object] = {
            "kind": "tcgen05_ab_tma",
            "lhs_idx": 0,
            "rhs_idx": 1,
            "bm": 256,
            "bn": 256,
            "bk": 64,
            "cluster_m": 2,
            "cluster_n": 1,
            "ab_stage_count": 3,
            "input_dtype": "cutlass.BFloat16",
            "acc_dtype": "cutlass.Float32",
            "kernel_args": [
                "tma_atom_a",
                "tma_tensor_a",
                "tma_atom_b",
                "tma_tensor_b",
            ],
        }
        d_plan: dict[str, object] = {
            "kind": "tcgen05_d_tma",
            "d_idx": 2,
            "bm": 256,
            "bn": 256,
            "c_stage_count": 2,
            "output_dtype": "cutlass.BFloat16",
            "epi_tile_m": 128,
            "epi_tile_n": 32,
            "d_store_box_n": 32,
            "kernel_args": [
                "tma_store_atom",
                "tma_store_tensor",
            ],
        }
        cute_kernel = type("DummyCuteKernel", (), {})()
        cute_kernel._helion_cute_wrapper_plans = [ab_plan, d_plan]

        direct_entry = _create_cute_direct_entry(
            cute_kernel,
            direct_plan,
            (2, 1, 64),
            (256, 1, 1),
        )
        self.assertTrue(direct_entry.__name__.startswith("_helion_cute_direct_entry_"))

        stale_index_kernel = type("DummyCuteKernel", (), {})()
        stale_index_kernel._helion_cute_wrapper_plans = [
            {**ab_plan, "lhs_idx": 1},
            d_plan,
        ]
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            "wrapper A/B plan mismatch for lhs_idx",
        ):
            _create_cute_direct_entry(
                stale_index_kernel,
                direct_plan,
                (2, 1, 64),
                (256, 1, 1),
            )

        stale_acc_kernel = type("DummyCuteKernel", (), {})()
        stale_acc_kernel._helion_cute_wrapper_plans = [
            {**ab_plan, "acc_dtype": "cutlass.Float16"},
            d_plan,
        ]
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            "wrapper A/B plan mismatch for acc_dtype",
        ):
            _create_cute_direct_entry(
                stale_acc_kernel,
                direct_plan,
                (2, 1, 64),
                (256, 1, 1),
            )

        stale_shape_kernel = type("DummyCuteKernel", (), {})()
        stale_shape_kernel._helion_cute_wrapper_plans = [
            ab_plan,
            {**d_plan, "epi_tile_n": 64},
        ]
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            "wrapper D plan mismatch for epi_tile_n",
        ):
            _create_cute_direct_entry(
                stale_shape_kernel,
                direct_plan,
                (2, 1, 64),
                (256, 1, 1),
            )

        stale_layout_kernel = type("DummyCuteKernel", (), {})()
        stale_layout_kernel._helion_cute_wrapper_plans = [
            {**ab_plan, "smem_swizzle_a": 8},
            d_plan,
        ]
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            "default A/B SMEM wrapper layouts",
        ):
            _create_cute_direct_entry(
                stale_layout_kernel,
                direct_plan,
                (2, 1, 64),
                (256, 1, 1),
            )

    def test_cute_direct_entry_validator_rejects_shape_bk_mismatch(self) -> None:
        """Cycle-2 P1 + cycle-3 B1: tensor shape envelope must be tied to
        both the plan's ``bk`` AND the plan's recorded ``validated_shape``.

        The cycle-3 review identified that T4 and T5 share ``bk=128``
        and the same ``(ab,c)`` stage tuple, so the bk-keyed shape-set
        alone cannot tell a T4 plan apart from a T5 plan at the
        validator boundary. The plan now carries its
        ``validated_shape=(M,N,K)`` envelope so the validator
        dispatches on it directly.
        """
        t1_plan: dict[str, object] = {
            "kind": "tcgen05_target1_direct_entry",
            "lhs_idx": 0,
            "rhs_idx": 1,
            "d_idx": 2,
            "bm": 256,
            "bn": 256,
            "bk": 64,
            "cluster_m": 2,
            "cluster_n": 1,
            "ab_stage_count": 3,
            "c_stage_count": 2,
            "input_dtype": "cutlass.BFloat16",
            "output_dtype": "cutlass.BFloat16",
            "validated_shape": [1024, 4096, 1024],
        }
        # T2, T3, T4, T5, and T6 share bk=128 but carry distinct
        # validated_shape triples so the validator can tell them apart
        # even when the bk-keyed shape-set table contains all five. T2
        # and T6 additionally carry ``bias_idx=3`` so they require a
        # 4th rank-1 arg; T3/T4/T5 keep the 3-tensor (lhs/rhs/output)
        # signature.
        t2_plan: dict[str, object] = {
            **t1_plan,
            "bk": 128,
            "validated_shape": [4096, 2048, 2048],
            "bias_idx": 3,
        }
        t3_plan: dict[str, object] = {
            **t1_plan,
            "bk": 128,
            "validated_shape": [2048, 4096, 2048],
        }
        t4_plan: dict[str, object] = {
            **t1_plan,
            "bk": 128,
            "validated_shape": [8192, 1024, 1024],
        }
        t5_plan: dict[str, object] = {
            **t1_plan,
            "bk": 128,
            "validated_shape": [1024, 8192, 1024],
        }
        t6_plan: dict[str, object] = {
            **t1_plan,
            "bk": 128,
            "validated_shape": [8192, 2048, 2048],
            "bias_idx": 3,
        }
        t7_plan: dict[str, object] = {
            **t1_plan,
            "bk": 128,
            "validated_shape": [2048, 8192, 2048],
        }
        t1_tensors = (
            torch.empty((1024, 1024), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((1024, 4096), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((1024, 4096), device=DEVICE, dtype=torch.bfloat16),
        )
        t2_tensors = (
            torch.empty((4096, 2048), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((2048, 2048), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((4096, 2048), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((2048,), device=DEVICE, dtype=torch.bfloat16),
        )
        t3_tensors = (
            torch.empty((2048, 2048), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((2048, 4096), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((2048, 4096), device=DEVICE, dtype=torch.bfloat16),
        )
        t4_tensors = (
            torch.empty((8192, 1024), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((1024, 1024), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((8192, 1024), device=DEVICE, dtype=torch.bfloat16),
        )
        t5_tensors = (
            torch.empty((1024, 1024), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((1024, 8192), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((1024, 8192), device=DEVICE, dtype=torch.bfloat16),
        )
        t6_tensors = (
            torch.empty((8192, 2048), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((2048, 2048), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((8192, 2048), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((2048,), device=DEVICE, dtype=torch.bfloat16),
        )
        t7_tensors = (
            torch.empty((2048, 2048), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((2048, 8192), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((2048, 8192), device=DEVICE, dtype=torch.bfloat16),
        )
        # T1 plan + T1 tensors accepted.
        _validate_target1_direct_entry_args(
            t1_plan,
            t1_tensors,
            (2, 1, 64),
            (256, 1, 1),
            "--enable-tvm-ffi",
        )
        # T4 plan + T4 tensors accepted.
        _validate_target1_direct_entry_args(
            t4_plan,
            t4_tensors,
            (2, 1, 74),
            (256, 1, 1),
            "--enable-tvm-ffi",
        )
        # T5 plan + T5 tensors accepted (same bk=128 accept set; the
        # validator now also matches validated_shape against tensors).
        _validate_target1_direct_entry_args(
            t5_plan,
            t5_tensors,
            (2, 1, 74),
            (256, 1, 1),
            "--enable-tvm-ffi",
        )
        # T3 plan + T3 tensors accepted. T3's runtime clustered grid is
        # ``(2, 1, 74)`` (M_tiles*N_tiles = 8*16 = 128, capped to
        # min(128, num_sms // cluster_m = 74) on B200) — same as
        # T4/T5's grid value on the bk=128 stage tuple plus T3's shape
        # envelope.
        _validate_target1_direct_entry_args(
            t3_plan,
            t3_tensors,
            (2, 1, 74),
            (256, 1, 1),
            "--enable-tvm-ffi",
        )
        # T2 plan + T2 tensors accepted (4 args: lhs, rhs, output, bias).
        # T2 shares bk=128 with T3/T4/T5/T6 but adds a 4th rank-1 bias arg.
        _validate_target1_direct_entry_args(
            t2_plan,
            t2_tensors,
            (2, 1, 74),
            (256, 1, 1),
            "--enable-tvm-ffi",
        )
        # T6 plan + T6 tensors accepted (4 args). T6 shares the 4-arg
        # signature with T2 but has a different validated_shape. T6's
        # runtime clustered grid is ``(2, 1, 74)`` (M_tiles*N_tiles =
        # 32*8 = 256, capped to min(256, num_sms // cluster_m = 74) on
        # B200).
        _validate_target1_direct_entry_args(
            t6_plan,
            t6_tensors,
            (2, 1, 74),
            (256, 1, 1),
            "--enable-tvm-ffi",
        )
        # T7 plan + T7 tensors accepted (3 args). T7 shares bk=128 and
        # the 3-tensor signature with T3/T4/T5 but has a different
        # validated_shape. T7's runtime clustered grid is ``(2, 1, 74)``
        # (M_tiles*N_tiles = 8*32 = 256, capped to min(256,
        # num_sms // cluster_m = 74) on B200) — same as T6's grid.
        _validate_target1_direct_entry_args(
            t7_plan,
            t7_tensors,
            (2, 1, 74),
            (256, 1, 1),
            "--enable-tvm-ffi",
        )
        # T1 plan + T4 tensors rejected.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(1024, 4096, 1024\) \(bk=64\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t1_plan,
                t4_tensors,
                (2, 1, 64),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T1 plan + T5 tensors rejected (bk=64 requires T1 shape).
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(1024, 4096, 1024\) \(bk=64\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t1_plan,
                t5_tensors,
                (2, 1, 64),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T4 plan + T1 tensors rejected.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(8192, 1024, 1024\) \(bk=128\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t4_plan,
                t1_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # B1 (cycle-3 review): T4 plan + T5 tensors must be rejected.
        # Both share bk=128, ab=3, c=2, and the clustered grid (2,1,74),
        # so only the per-plan validated_shape can tell them apart.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(8192, 1024, 1024\) \(bk=128\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t4_plan,
                t5_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # B1 (cycle-3 review): T5 plan + T4 tensors must be rejected.
        # Symmetric to the above; this is the defense-in-depth case
        # cycle 3 was missing.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(1024, 8192, 1024\) \(bk=128\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t5_plan,
                t4_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T3 plan + T4 tensors must be rejected (same bk, different
        # validated_shape).
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(2048, 4096, 2048\) \(bk=128\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t3_plan,
                t4_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T4 plan + T3 tensors must be rejected (same bk, different
        # validated_shape).
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(8192, 1024, 1024\) \(bk=128\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t4_plan,
                t3_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T1 plan + T3 tensors rejected (bk=64 requires T1 shape).
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(1024, 4096, 1024\) \(bk=64\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t1_plan,
                t3_tensors,
                (2, 1, 64),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T2 plan + 3 args rejected: T2's plan carries bias_idx=3, so the
        # validator requires exactly 4 args.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"lhs/rhs/output/bias args",
        ):
            _validate_target1_direct_entry_args(
                t2_plan,
                t2_tensors[:3],
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T3 plan + T2 tensors (4-tensor call) rejected: T3 lacks
        # bias_idx so the validator requires exactly 3 args.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"lhs/rhs/output args",
        ):
            _validate_target1_direct_entry_args(
                t3_plan,
                t2_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T2 plan + T3 tensors (with a bf16 rank-1 bias appended) rejected:
        # same bk=128 but different validated_shape, so the shape
        # envelope cross-check catches it.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(4096, 2048, 2048\) \(bk=128\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t2_plan,
                (
                    *t3_tensors,
                    torch.empty((4096,), device=DEVICE, dtype=torch.bfloat16),
                ),
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T2 plan + rank-2 4th tensor rejected: the bias must be rank-1
        # with shape (N,).
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"contiguous rank-1 bf16 tensor of shape \(2048,\)",
        ):
            _validate_target1_direct_entry_args(
                t2_plan,
                (
                    *t2_tensors[:3],
                    torch.empty((1, 2048), device=DEVICE, dtype=torch.bfloat16),
                ),
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T2 plan + rank-1 4th tensor with wrong N-extent rejected: the
        # bias's N must match the validated_shape's N.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"contiguous rank-1 bf16 tensor of shape \(2048,\)",
        ):
            _validate_target1_direct_entry_args(
                t2_plan,
                (
                    *t2_tensors[:3],
                    torch.empty((4096,), device=DEVICE, dtype=torch.bfloat16),
                ),
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T6 plan + 3 args rejected: T6's plan carries bias_idx=3, so
        # the validator requires exactly 4 args.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"lhs/rhs/output/bias args",
        ):
            _validate_target1_direct_entry_args(
                t6_plan,
                t6_tensors[:3],
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T6 plan + T2 tensors rejected: same bk=128 and 4-arg signature,
        # but different validated_shape.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(8192, 2048, 2048\) \(bk=128\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t6_plan,
                t2_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T2 plan + T6 tensors rejected: same bk=128 and 4-arg signature,
        # but different validated_shape.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(4096, 2048, 2048\) \(bk=128\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t2_plan,
                t6_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T6 plan + T3 tensors (3 args) rejected: T6 plan requires 4 args.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"lhs/rhs/output/bias args",
        ):
            _validate_target1_direct_entry_args(
                t6_plan,
                t3_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T3 plan + T6 tensors (4 args) rejected: T3 plan lacks bias_idx.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"lhs/rhs/output args",
        ):
            _validate_target1_direct_entry_args(
                t3_plan,
                t6_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T6 plan + rank-1 4th tensor with wrong N-extent rejected: the
        # bias's N must match the validated_shape's N (= 2048 for T6).
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"contiguous rank-1 bf16 tensor of shape \(2048,\)",
        ):
            _validate_target1_direct_entry_args(
                t6_plan,
                (
                    *t6_tensors[:3],
                    torch.empty((8192,), device=DEVICE, dtype=torch.bfloat16),
                ),
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T7 plan + T3 tensors rejected: same bk=128 + 3-tensor signature
        # but different validated_shape.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(2048, 8192, 2048\) \(bk=128\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t7_plan,
                t3_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T3 plan + T7 tensors rejected: symmetric to above.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(2048, 4096, 2048\) \(bk=128\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t3_plan,
                t7_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T7 plan + T5 tensors rejected: T5 (1024x8192x1024) and T7
        # (2048x8192x2048) share N=8192 but differ on M and K, so only
        # the validated_shape cross-check catches the mismatch.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(2048, 8192, 2048\) \(bk=128\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t7_plan,
                t5_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T5 plan + T7 tensors rejected: symmetric to above.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(1024, 8192, 1024\) \(bk=128\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t5_plan,
                t7_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T7 plan + T6 tensors (4 args) rejected: T7 plan lacks
        # bias_idx so the validator requires exactly 3 args.
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"lhs/rhs/output args",
        ):
            _validate_target1_direct_entry_args(
                t7_plan,
                t6_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T6 plan + T7 tensors (3 args) rejected: T6 plan requires 4
        # args (with bias_idx).
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"lhs/rhs/output/bias args",
        ):
            _validate_target1_direct_entry_args(
                t6_plan,
                t7_tensors,
                (2, 1, 74),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # T1 plan + T7 tensors rejected (bk=64 requires T1 shape).
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated_shape=\(1024, 4096, 1024\) \(bk=64\) requires shapes",
        ):
            _validate_target1_direct_entry_args(
                t1_plan,
                t7_tensors,
                (2, 1, 64),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )
        # Unvalidated clustered grid K is also rejected (V1 tightening).
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            r"validated cluster_m=2 launch geometry",
        ):
            _validate_target1_direct_entry_args(
                t4_plan,
                t4_tensors,
                (2, 1, 73),
                (256, 1, 1),
                "--enable-tvm-ffi",
            )

    def test_cute_direct_entry_clustered_grid_accepts_target_total_clusters(
        self,
    ) -> None:
        """A1 (cycle-7 review): the clustered ``grid[2]`` accept set must
        include every validated ``total_clusters`` value once SM-count is
        large enough.

        On B200 (148 SMs, cluster_m=2 cap = 74) all validated
        ``total_clusters`` collapse to the cap. The validator must keep
        emitting the unclamped accept-set values on a hypothetical
        larger SKU where ``num_sms // cluster_m >= 256`` so T6's
        runtime ``grid[2] = 256`` is in the accept set (and T2/T3/T4/T5
        keep their ``grid[2] = 128``, T1 its ``64``).
        """
        from unittest.mock import patch

        FakeDeviceProps = type("FakeDeviceProps", (), {"multi_processor_count": 0})

        # B200-like (148 SMs, cap=74): all targets collapse to 74.
        b200_props = FakeDeviceProps()
        b200_props.multi_processor_count = 148
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=b200_props),
        ):
            b200_accepted = _direct_entry_clustered_grid_k(torch.device("cuda:0"))
        # ``{74} ∪ {min(64, 74), min(128, 74), min(256, 74)}`` = {64, 74}
        # because every total_clusters >= 74 clamps to 74.
        self.assertEqual(b200_accepted, (64, 74))

        # Hypothetical larger SKU (520 SMs, cap=260): all four validated
        # total_clusters fit under the cap, so the unclamped 64, 128, and
        # 256 values are all in the accept set alongside the cap.
        large_props = FakeDeviceProps()
        large_props.multi_processor_count = 520
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=large_props),
        ):
            large_accepted = _direct_entry_clustered_grid_k(torch.device("cuda:0"))
        self.assertEqual(large_accepted, (64, 128, 256, 260))
        # T6's runtime ``grid[2] = 256`` is in the accept set on the
        # hypothetical larger SKU; that's the load-bearing check this
        # test guards against in case ``TCGEN05_DIRECT_ENTRY_TOTAL_WORK_CLUSTERS``
        # were to drop 256.
        self.assertIn(256, large_accepted)

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
