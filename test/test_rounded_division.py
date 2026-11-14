"""Truncating and flooring float division needs the correctly rounded quotient.

The Triton backend compiles an fp32 ``/`` to ``div.full.f32`` (a reciprocal
and a multiply, within 2 ulp), which gives 33 / 33 just below 1: truncated or
floored, that is 0.  A quotient its node rounds to an integer is ``div.rn``
instead (``GenerateASTFromInductor.truediv`` and ``div_rn``), and so is
``addcdiv``'s without ``fast_math``.
"""

from __future__ import annotations

from contextlib import ExitStack
import functools
from typing import Any
import unittest
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing import skipIfCute
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cuda_fake
import helion.language as hl

# The source checks bind on CPU; the tests that compare results with eager's
# are marked cuda and run in the GPU lane.
pytestmark = pytest.mark.cpu_only

# How the backend spells the round-to-nearest division, the approximate
# division of ``fast_math``, and a quotient rounded to fp16 and widened again.
DIV_RN = "tl.div_rn("
FAST_DIVISION = "fast_dividef("
ROUNDED_TO_FP16 = ", tl.float16), tl.float32)"


def _truncated(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.div(x[tile], y[tile], rounding_mode="trunc")
    return out


def _floored(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.div(x[tile], y[tile], rounding_mode="floor")
    return out


def _floor_divided(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] // y[tile]
    return out


def _plain(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] / y[tile]
    return out


def _truncated_in_place(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        quotient = x[tile]
        quotient.div_(y[tile], rounding_mode="trunc")
        out[tile] = quotient
    return out


def _floored_in_place(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        quotient = x[tile]
        quotient.div_(y[tile], rounding_mode="floor")
        out[tile] = quotient
    return out


def _divided_in_place(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        quotient = x[tile]
        quotient.div_(y[tile])
        out[tile] = quotient
    return out


def _truncated_by_a_scalar(x: torch.Tensor, divisor: float) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.div(x[tile], divisor, rounding_mode="trunc")
    return out


def _addcdiv(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.addcdiv(x[tile], a[tile], b[tile], value=0.5)
    return out


@functools.cache
def _multiples() -> tuple[torch.Tensor, torch.Tensor]:
    """``q * b`` and ``b`` for every factor pair up to 128, a third of the
    products negated: each quotient is exactly ``q`` or ``-q``."""
    factor = torch.arange(1, 129, device=DEVICE, dtype=torch.float32)
    q, b = torch.meshgrid(factor, factor, indexing="ij")
    x, y = (q * b).reshape(-1), b.reshape(-1).contiguous()
    sign = torch.where(torch.arange(x.numel(), device=DEVICE) % 3 == 0, -1.0, 1.0)
    return (x * sign).contiguous(), y


@functools.cache
def _addcdiv_operands() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    x = torch.randn(1 << 14, device=DEVICE, generator=generator)
    a = torch.randn(1 << 14, device=DEVICE, generator=generator)
    b = torch.rand(1 << 14, device=DEVICE, generator=generator) + 0.5
    return x, a, b


@functools.cache
def _run(fn: Any, dtype: torch.dtype = torch.float32) -> tuple[str, torch.Tensor]:
    """``fn``'s generated code and its result, compiled once per module."""
    operands = _addcdiv_operands() if fn is _addcdiv else _multiples()
    args = tuple(operand.to(dtype) for operand in operands)
    return code_and_output(helion.kernel(fn, static_shapes=True), args)


def _render(
    fn: Any,
    dtype: torch.dtype = torch.float32,
    divisor_dtype: torch.dtype | None = None,
    **settings: object,
) -> str:
    """``fn``'s generated code for a dividend of ``dtype`` and a divisor of
    ``divisor_dtype`` (the same dtype by default), bound on CPU."""
    x = torch.ones(64, dtype=dtype)
    y = torch.ones(64, dtype=divisor_dtype or dtype)
    args: tuple[object, ...] = (x, y)
    if fn is _truncated_by_a_scalar:
        args = (x, 3.0)
    with ExitStack() as stack:
        if fn is _addcdiv:
            # Inductor takes addcdiv's div_rn path on a CUDA device only: the
            # inputs are traced as ``cuda:0`` tensors, still without a CUDA
            # context (``cpu_render_target`` re-homes them for CuTe).
            args = (x, x, y)
            stack.enter_context(cpu_render_target(fake_device_sm_count=148))
            stack.enter_context(
                patch(
                    "helion._compiler.triton.backend.TritonBackend"
                    ".normalize_input_fake_tensor",
                    new=cuda_fake,
                )
            )
        else:
            stack.enter_context(cpu_render_target())
        bound = cpu_bind(helion.kernel(fn, static_shapes=True, **settings), args)
        return bound.to_code(bound.config_spec.default_config())


@onlyBackends(["triton"])
@skipIfTileIR("TileIR keeps `/` for a rounded quotient (tl.div_rn was not tried there)")
class TestRoundedDivision(TestCase):
    @pytest.mark.cuda
    def test_multiples_divide_to_their_factor(self) -> None:
        x, y = _multiples()
        for fn, mode in (
            (_truncated, "trunc"),
            (_floored, "floor"),
            (_floor_divided, "floor"),
        ):
            _, out = _run(fn)
            torch.testing.assert_close(
                out, torch.div(x, y, rounding_mode=mode), rtol=0, atol=0
            )

    @pytest.mark.cuda
    def test_a_16_bit_quotient_truncates_as_eager_does(self) -> None:
        # The products past 2048 are not fp16 values, so their quotients are
        # not whole: eager rounds each to its dtype before truncating it
        # (2.9998 is 3 in fp16), and truncating the fp32 quotient gave 2.
        # XPU eager does not round an fp16 quotient, and neither does the
        # kernel there (test_an_fp16_quotient_is_not_rounded_on_xpu).
        for dtype in (torch.float16, torch.bfloat16):
            x, y = (operand.to(dtype) for operand in _multiples())
            _, out = _run(_truncated, dtype)
            torch.testing.assert_close(
                out, torch.div(x, y, rounding_mode="trunc"), rtol=0, atol=0
            )

    @skipIfCute("XPU kernels are Triton's")
    @skipIfRefEager("checks the generated code")
    def test_an_fp16_quotient_is_not_rounded_on_xpu(self) -> None:
        # XPU eager's fp16 kernel truncates the unrounded quotient (2144 / 65
        # is 32 there, and 33 where 32.98 is rounded to fp16 first); its bf16
        # kernel rounds first, as on CUDA.  Rendered for an XPU device
        # without one.
        for dtype, cast in (
            (torch.float16, "tl.float16"),
            (torch.bfloat16, "tl.bfloat16"),
        ):
            operand = torch.ones(64, dtype=dtype)
            with cpu_render_target():
                bound = cpu_bind(
                    helion.kernel(_truncated, static_shapes=True), (operand, operand)
                )
                with patch.object(bound.env, "device", torch.device("xpu")):
                    code = bound.to_code(bound.config_spec.default_config())
            self.assertIn("tl.div_rn(", code)
            rounded = f", {cast}), tl.float32)" in code
            self.assertEqual(rounded, dtype == torch.bfloat16)

    @pytest.mark.cuda
    def test_addcdiv_divides_as_eager_does(self) -> None:
        """Inductor's ``addcdiv`` asks for ``div_rn`` so that its quotient
        rounds as eager's before the multiply-add; with ``/`` 14% of fp32
        results differed, by up to 0.6%."""
        x, a, b = _addcdiv_operands()
        _, out = _run(_addcdiv)
        torch.testing.assert_close(
            out, torch.addcdiv(x, a, b, value=0.5), rtol=0, atol=0
        )

    @skipIfRefEager("checks the generated code")
    @unittest.skipUnless(
        torch.backends.cuda.is_built(),
        "traces addcdiv on a fake CUDA device, which this torch build lacks",
    )
    def test_addcdiv_quotient_is_div_rn_without_fast_math(self) -> None:
        self.assertIn(DIV_RN, _render(_addcdiv))
        # Nothing rounds addcdiv's quotient to an integer, so ``fast_math``
        # leaves it the backend's ``/``.
        self.assertNotIn(DIV_RN, _render(_addcdiv, fast_math=True))

    @skipIfRefEager("checks the generated code")
    def test_the_quotient_is_div_rn(self) -> None:
        for fn, dtype in (
            (_truncated, torch.float32),
            (_truncated, torch.float16),
            (_floored, torch.float32),
            (_floor_divided, torch.float32),
        ):
            self.assertIn(DIV_RN, _render(fn, dtype))
        # A plain quotient keeps the backend's ``/``.
        self.assertNotIn(DIV_RN, _render(_plain))

    @skipIfRefEager("checks the generated code")
    def test_a_16_bit_quotient_is_rounded_where_eager_rounds_it(self) -> None:
        # Eager's tensor-by-tensor kernel divides in the dtype, so the
        # quotient is rounded to it before it is truncated.  With a host
        # scalar for a divisor it truncates an fp32 product: rounding that
        # quotient first gave 3 for 1999 / 666.5 in fp16, where eager has 2.
        self.assertIn(ROUNDED_TO_FP16, _render(_truncated, torch.float16))
        scalar = _render(_truncated_by_a_scalar, torch.float16)
        self.assertIn(DIV_RN, scalar)
        self.assertNotIn(ROUNDED_TO_FP16, scalar)
        # A float32 divisor promotes an in-place fp16 division to fp32.
        mixed = _render(_truncated_in_place, torch.float16, torch.float32)
        self.assertIn(DIV_RN, mixed)
        self.assertNotIn(ROUNDED_TO_FP16, mixed)

    @skipIfRefEager("checks the generated code")
    def test_a_float64_divisor_divides_in_float64(self) -> None:
        # ``x.div_(y)`` keeps the dtype of ``x``, but a float64 ``y`` promotes
        # the division: its operands must not be narrowed to fp32 for
        # ``tl.div_rn``, and float64 ``/`` is correctly rounded already.
        for fn in (_truncated_in_place, _floored_in_place):
            self.assertIn(DIV_RN, _render(fn))
            code = _render(fn, divisor_dtype=torch.float64)
            self.assertNotIn(DIV_RN, code)
            self.assertRegex(code, r" = \w+ / \w+\n")
        # Nor for the approximate division of ``fast_math``.
        self.assertIn(FAST_DIVISION, _render(_divided_in_place, fast_math=True))
        code = _render(_divided_in_place, divisor_dtype=torch.float64, fast_math=True)
        self.assertNotIn(FAST_DIVISION, code)
        self.assertRegex(code, r" = \w+ / \w+\n")

    @skipIfRefEager("checks the generated code")
    def test_fast_math_keeps_the_rounded_quotient(self) -> None:
        # ``fast_math`` approximates a plain quotient only: one that is then
        # truncated or floored is still correctly rounded.
        for fn in (_truncated, _floored, _floor_divided):
            self.assertIn(DIV_RN, _render(fn, fast_math=True))


if __name__ == "__main__":
    unittest.main()
