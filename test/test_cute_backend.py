from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessCuteAvailable
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
def cute_dynamic_row_sum(x: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    out = x.new_empty([x.size(0)])
    bs = hl.register_block_size(x.size(1))
    for tile0 in hl.tile(x.size(0)):
        acc = hl.zeros([tile0, bs])
        for tile1 in hl.tile(end[0], block_size=bs):
            acc += x[tile0, tile1]
        out[tile0] = acc.sum(-1)
    return out


@onlyBackends(["triton", "cute"])
@skipUnlessCuteAvailable("requires CUTLASS CuTe DSL")
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
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
        args = (torch.randn(65, 23, device=DEVICE, dtype=torch.float16),)
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

    def test_oversized_nd_block_raises(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported, "thread block too large for cute kernel"
        ):
            code_and_output(cute_add, args, block_sizes=[64, 32])

    def test_nd_elements_per_thread(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(
            cute_add,
            args,
            block_sizes=[64, 32],
            elements_per_thread=[2, 2],
        )
        x, y = args
        torch.testing.assert_close(out, x + y)

    def test_nd_elements_per_thread_exceeds_block_raises(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        with self.assertRaisesRegex(
            helion.exc.BackendUnsupported,
            "elements_per_thread must divide block size",
        ):
            code_and_output(
                cute_add,
                args,
                block_sizes=[32, 32],
                elements_per_thread=[64, 1],
            )

    def test_flattened_elements_per_thread(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(
            cute_add,
            args,
            block_sizes=[64, 32],
            flatten_loop=True,
            elements_per_thread=[2, 2],
        )
        x, y = args
        torch.testing.assert_close(out, x + y)
        self.assertIn("block=(512, 1, 1)", code)

    def test_device_loop_elements_per_thread(self) -> None:
        args = (torch.randn(65, 23, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(
            cute_device_loop_add_one,
            args,
            block_sizes=[64, 32],
            elements_per_thread=[2, 2],
        )
        (x,) = args
        torch.testing.assert_close(out, x + 1)
        self.assertIn("for lane_", code)

    def test_flattened_device_loop_elements_per_thread(self) -> None:
        args = (torch.randn(8, 65, 23, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(
            cute_flattened_device_loop_add_one,
            args,
            block_sizes=[1, 64, 32],
            flatten_loops=[True],
            elements_per_thread=[1, 2, 2],
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

    def test_reduction_elements_per_thread(self) -> None:
        args = (torch.randn(129, 130, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(
            cute_row_sum,
            args,
            block_sizes=[64],
            elements_per_thread=[2],
        )
        (x,) = args
        torch.testing.assert_close(out, x.sum(-1), rtol=1e-4, atol=1e-4)
        self.assertIn("for lane_", code)

    def test_looped_reduction_elements_per_thread(self) -> None:
        args = (torch.randn(129, 130, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(
            cute_row_sum,
            args,
            block_sizes=[64],
            reduction_loop=16,
            elements_per_thread=[2],
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

    def test_strided_threaded_reduction_cross_warp_shared_memory(self) -> None:
        args = (
            torch.randn(512, 512, device=DEVICE, dtype=torch.float32),
            torch.tensor([200], device=DEVICE, dtype=torch.int64),
        )
        code, out = code_and_output(cute_dynamic_row_sum, args, block_sizes=[32, 32])
        x, end = args
        expected = x[:, : end.item()].sum(dim=1)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        self.assertIn("cute.arch.alloc_smem", code)
        self.assertIn("cute.arch.sync_threads()", code)
