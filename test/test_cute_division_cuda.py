"""The CuTe backend's division on a GPU: bit for bit the Triton backend's.

Both compile a 32-bit or 16-bit float ``/`` to ``div.full.f32`` (within 2 ulp
over the full range, no slow path), so their quotients are the same bits for
any operands; ``Settings.cute_exact_division`` gives eager's IEEE quotient
instead, and a truncated quotient is exact in either mode.
``test_cute_division.py`` checks the generated code.
"""

from __future__ import annotations

import functools
from typing import Any

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing.cute_sass import cute_launcher_sass
import helion.language as hl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CONFIG = helion.Config(block_sizes=[128])


def _divide(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] / y[tile]
    return out


def _reciprocal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.reciprocal(y[tile])
    return out


def _truncated(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = torch.div(x[tile], y[tile], rounding_mode="trunc")
    return out


def _floor_divided(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] // y[tile]
    return out


@functools.cache
def _kernel(fn: Any, backend: str, exact: bool = False) -> helion.Kernel:
    settings: dict[str, Any] = {"cute_exact_division": True} if exact else {}
    return helion.kernel(
        fn,
        backend=backend,
        config=CONFIG,
        static_shapes=True,
        ignore_warnings=[exc.FastMathUnused],
        **settings,
    )


@functools.cache
def _operands() -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 operands of every kind: random bit patterns (subnormals, huge
    values, infinities and NaNs among them), then every pair of a few
    special values."""
    generator = torch.Generator(device=DEVICE).manual_seed(0)

    def bits() -> torch.Tensor:
        words = torch.randint(
            -(2**31), 2**31 - 1, (1 << 16,), device=DEVICE, generator=generator
        )
        return words.to(torch.int32).view(torch.float32)

    tiny = [1e-45, 1.1754942e-38, 1.17549435e-38, 2e-38]
    huge = [8.5e37, 8.6e37, 3.4028235e38, float("inf"), -float("inf")]
    specials = torch.tensor(
        [0.0, -0.0, 1.0, -3.0, 33.0, *tiny, *huge, float("nan")], device=DEVICE
    )
    left, right = torch.meshgrid(specials, specials, indexing="ij")
    x = torch.cat([bits(), left.reshape(-1)])
    y = torch.cat([bits(), right.reshape(-1)])
    pad = -x.numel() % 128
    return (
        torch.cat([x, x.new_ones(pad)]).contiguous(),
        torch.cat([y, y.new_ones(pad)]).contiguous(),
    )


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    integer = torch.int32 if a.element_size() == 4 else torch.int16
    return (a.view(integer) == b.view(integer)) | (a.isnan() & b.isnan())


@pytest.mark.parametrize(
    ("fn", "dtype"),
    [
        (_divide, torch.float32),
        (_divide, torch.bfloat16),  # divided in fp32 and rounded back
        (_reciprocal, torch.float32),
    ],
    ids=["divide", "divide-bf16", "reciprocal"],
)
def test_quotients_are_the_triton_backends(fn: Any, dtype: torch.dtype) -> None:
    x, y = (operand.to(dtype) for operand in _operands())
    cute = _kernel(fn, "cute")(x, y)
    triton = _kernel(fn, "triton")(x, y)
    assert bool(_same_bits(cute, triton).all())
    if dtype == torch.float32:
        # Not the IEEE quotient: about a quarter of them differ from eager's
        # in the last bits (16-bit quotients round to the same value).
        eager = x / y if fn is _divide else torch.reciprocal(y)
        assert not bool(_same_bits(cute, eager).all())


def test_quotients_are_within_two_ulp() -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    x = torch.randn(1 << 14, device=DEVICE, generator=generator) * 100.0
    y = torch.randn(1 << 14, device=DEVICE, generator=generator) + 3.0
    got = _kernel(_divide, "cute")(x, y)
    exact = x.double() / y.double()
    ulp = torch.ldexp(torch.ones_like(exact), exact.abs().frexp().exponent - 24)
    assert bool(((got.double() - exact).abs() <= 2 * ulp).all())


def test_exact_division_is_eagers_quotient() -> None:
    x, y = _operands()
    got = _kernel(_divide, "cute", exact=True)(x, y)
    assert bool(_same_bits(got, x / y).all())


def test_a_truncated_quotient_is_exact() -> None:
    """Every multiple of a divisor divides to its factor: the approximate
    quotient of 33 / 33 is just below 1 and would truncate to 0."""
    factor = torch.arange(1, 129, device=DEVICE, dtype=torch.float32)
    q, b = torch.meshgrid(factor, factor, indexing="ij")
    x, y = (q * b).reshape(-1).contiguous(), b.reshape(-1).contiguous()
    for exact in (False, True):
        got = _kernel(_truncated, "cute", exact=exact)(x, y)
        assert torch.equal(got, q.reshape(-1))
    # The approximate quotient the kernel's other divisions use is not.
    plain = _kernel(_divide, "cute")(x, y)
    assert not torch.equal(plain.trunc(), q.reshape(-1))


def test_a_16_bit_quotient_floors_to_its_factor() -> None:
    """``//`` of two bf16 tensors: the lowering's zero and NaN selects take
    the fp32 quotient next to their 16-bit operands."""
    factor = torch.arange(1, 16, device=DEVICE, dtype=torch.bfloat16)
    q, b = torch.meshgrid(factor, factor, indexing="ij")
    x, y = (q * b).reshape(-1).contiguous(), b.reshape(-1).contiguous()
    got = _kernel(_floor_divided, "cute")(x, y)
    assert torch.equal(got, q.reshape(-1))
    assert torch.equal(got, x // y)


def test_the_default_division_has_no_slow_path() -> None:
    """ptxas expands ``div.rn.f32`` into a fast path, a range check (FCHK)
    and a slow-path call; ``div.full.f32`` is a reciprocal and a multiply."""
    x, y = _operands()
    for exact in (False, True):
        kernel = _kernel(_divide, "cute", exact=exact)
        kernel(x, y)
        compiled = kernel.bind((x, y)).compile_config(CONFIG)
        sass = cute_launcher_sass(compiled)
        assert "MUFU.RCP" in sass
        assert ("FCHK" in sass) is exact
