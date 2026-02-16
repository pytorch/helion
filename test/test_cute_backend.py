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
        self.assertExpectedJournal(code)

    def test_pointwise_add_three_inputs(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(cute_add3, args)
        x, y, z = args
        torch.testing.assert_close(out, x + y + z)
        self.assertExpectedJournal(code)

    def test_pointwise_mul(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(cute_mul, args)
        x, y = args
        torch.testing.assert_close(out, x * y)
        self.assertExpectedJournal(code)

    def test_pointwise_relu(self) -> None:
        args = (torch.randn(65, 23, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(cute_relu, args)
        (x,) = args
        torch.testing.assert_close(out, torch.relu(x))
        self.assertExpectedJournal(code)

    def test_pointwise_sin(self) -> None:
        args = (torch.randn(65, 23, device=DEVICE, dtype=torch.float32),)
        code, out = code_and_output(cute_sin, args)
        (x,) = args
        torch.testing.assert_close(out, torch.sin(x))
        self.assertExpectedJournal(code)

    def test_pointwise_sigmoid(self) -> None:
        args = (torch.randn(65, 23, device=DEVICE, dtype=torch.float16),)
        code, out = code_and_output(cute_sigmoid, args)
        (x,) = args
        torch.testing.assert_close(out, torch.sigmoid(x), rtol=1e-3, atol=1e-3)
        self.assertExpectedJournal(code)

    def test_pointwise_chain(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
        )
        code, out = code_and_output(cute_pointwise_chain, args)
        x, y = args
        expected = torch.sigmoid(torch.sin(torch.relu(x * y)))
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        self.assertExpectedJournal(code)

    def test_scalar_args_int_and_float(self) -> None:
        args = (
            torch.randn(65, 23, device=DEVICE, dtype=torch.float32),
            3,
            1.25,
        )
        code, out = code_and_output(cute_affine_scalar_args, args)
        x, scale, bias = args
        torch.testing.assert_close(out, x * scale + bias, rtol=1e-5, atol=1e-5)
        self.assertExpectedJournal(code)

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
        self.assertExpectedJournal(code)
