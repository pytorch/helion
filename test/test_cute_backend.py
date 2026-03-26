from __future__ import annotations

import os
from unittest.mock import patch

import torch

import helion
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


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

    def test_oversized_flattened_block_raises(self) -> None:
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
            code_and_output(cute_flattened_identity, args, block_size=2048)

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
        self.assertIn("cute.arch.warp_reduction_sum", code)
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
        self.assertIn("cute.nvgpu.tcgen05.make_umma_smem_desc", code)
        self.assertIn("cute.mma_atom_call", code)
        self.assertIn(
            "tcgen05_pipeline_arrive_count = cutlass.Int32(4)",
            code,
        )
        self.assertIn(
            "cutlass.pipeline.NamedBarrier(barrier_id=1, num_threads=512)",
            code,
        )
        self.assertIn(
            "cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, tcgen05_pipeline_arrive_count)",
            code,
        )

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
        self.assertIn(
            "tcgen05_pipeline_arrive_count = cutlass.Int32(8)",
            code,
        )
        self.assertIn(
            "cutlass.pipeline.NamedBarrier(barrier_id=1, num_threads=1024)",
            code,
        )
        self.assertIn(
            "cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, tcgen05_pipeline_arrive_count)",
            code,
        )

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
        self.assertIn("cute.arch.warp_reduction_sum", code)
        self.assertNotIn("cute.gemm", code)

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

    def test_strided_threaded_reduction_cross_warp_shared_memory(self) -> None:
        args = (
            torch.randn(512, 512, device=DEVICE, dtype=torch.float32),
            torch.tensor([200], device=DEVICE, dtype=torch.int64),
        )
        code, out = code_and_output(cute_dynamic_row_sum, args, block_sizes=[32, 32])
        x, end = args
        expected = x[:, : end.item()].sum(dim=1)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        self.assertIn("block=(32, 32, 1)", code)
        self.assertIn("cute.arch.alloc_smem", code)
        self.assertIn("cute.arch.sync_threads()", code)
