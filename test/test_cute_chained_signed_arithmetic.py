from __future__ import annotations

import operator
import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    patch.object(
        torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
    ).start()

from .test_cute_chained_tcgen05 import _compile
from .test_cute_chained_tcgen05_guards import _code
import helion
from helion._compiler.cute.chained_matmul import _Expression
from helion._compiler.cute.chained_matmul import _signed_floor_adjustment
from helion._compiler.cute.chained_matmul import _signed_remainder_adjustment
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _signed_arithmetic(
    a: torch.Tensor,
    b: torch.Tensor,
    integer_dtype: torch.dtype,
    divisor: int,
    mode: hl.constexpr,
) -> torch.Tensor:
    divisor = hl.specialize(divisor)
    batch, rows, reduction = a.shape
    columns = b.shape[2]
    out = torch.empty((batch, rows, columns), dtype=a.dtype, device=a.device)
    for bt, row, col in hl.tile([batch, rows, columns], block_size=[1, None, None]):
        if mode == "scalar_index":
            bi = (bt.begin - 2) % 3
        elif mode == "scalar_floor_index":
            bi = (bt.begin - 2) // divisor + 1
        elif mode == "scalar_positive_floor_index":
            bi = bt.begin // divisor
        else:
            bi = bt.begin
        kk = hl.arange(reduction)
        first = hl.dot((a[bi, row, kk].float() * 0.5).to(a.dtype), b[bi, kk, col])
        numerator = row.index.to(integer_dtype) - 64
        if mode in (
            "scalar_index",
            "scalar_floor_index",
            "scalar_positive_floor_index",
        ):
            value = numerator[:, None] * 0
        elif mode == "floor":
            value = torch.div(numerator, divisor, rounding_mode="floor")[:, None]
        elif mode == "floor_op":
            value = torch.floor_divide(numerator, divisor)[:, None]
        elif mode == "remainder":
            value = torch.remainder(numerator, divisor)[:, None]
        else:
            denominator = (col.index.to(integer_dtype) % 2) * 6 - 3
            if mode == "scalar_left":
                value = torch.remainder(-7, denominator)[None, :]
            elif mode == "floor_tensor":
                value = torch.div(
                    numerator[:, None], denominator[None, :], rounding_mode="floor"
                )
            elif mode == "floor_scalar_left":
                value = torch.div(-7, denominator, rounding_mode="floor")[None, :]
            else:
                value = torch.remainder(numerator[:, None], denominator[None, :])
        out[bt.begin, row, col] = (first + value.float()).to(a.dtype)
    return out


def _values(device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((2, 128, 64), dtype=torch.bfloat16, device=device),
        torch.zeros((2, 64, 64), dtype=torch.bfloat16, device=device),
    )


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_signed_remainder_correction_matches_torch(dtype: torch.dtype) -> None:
    limits = torch.iinfo(dtype)
    dividends = [limits.min, limits.min + 1, -11, -6, -1, 0, 1, 6, 11, limits.max]
    divisors = [limits.min, -7, -3, -1, 1, 3, 7, limits.max]
    expression = _signed_remainder_adjustment("remainder", "divisor")
    for dividend in dividends:
        for divisor in divisors:
            if dividend == limits.min and divisor == -1:
                continue  # The native signed division overflow is unchanged.
            quotient = abs(dividend) // abs(divisor)
            if (dividend < 0) != (divisor < 0):
                quotient = -quotient
            remainder = dividend - quotient * divisor
            actual = eval(expression, {}, {"remainder": remainder, "divisor": divisor})
            expected = torch.remainder(
                torch.tensor(dividend, dtype=dtype), torch.tensor(divisor, dtype=dtype)
            ).item()
            assert actual == expected


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_signed_floor_exact_quotient_correction_matches_torch(
    dtype: torch.dtype,
) -> None:
    limits = torch.iinfo(dtype)
    dividends = [limits.min, limits.min + 1, -11, -6, -1, 0, 1, 6, 11, limits.max]
    divisors = [limits.min, -7, -3, -1, 1, 3, 7, limits.max]
    expression = _signed_floor_adjustment("quotient", "remainder", "divisor")
    for dividend in dividends:
        for divisor in divisors:
            if dividend == limits.min and divisor == -1:
                continue
            quotient = abs(dividend) // abs(divisor)
            if (dividend < 0) != (divisor < 0):
                quotient = -quotient
            remainder = dividend - quotient * divisor
            exact_dividend = dividend - remainder
            assert limits.min <= exact_dividend <= limits.max
            assert exact_dividend % divisor == 0
            actual = eval(
                expression,
                {},
                {
                    "quotient": exact_dividend // divisor,
                    "remainder": remainder,
                    "divisor": divisor,
                },
            )
            expected = torch.div(
                torch.tensor(dividend, dtype=dtype),
                torch.tensor(divisor, dtype=dtype),
                rounding_mode="floor",
            ).item()
            assert actual == expected


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "mode",
    [
        "floor",
        "floor_op",
        "floor_tensor",
        "floor_scalar_left",
        "remainder",
        "tensor",
        "scalar_left",
    ],
)
@pytest.mark.parametrize("divisor", [-3, 3])
def test_signed_arithmetic_codegen(dtype: torch.dtype, mode: str, divisor: int) -> None:
    code = _code((*_values("cpu"), dtype, divisor, mode), None, _signed_arithmetic)
    assert "chain_0_mma" in code
    assert "!= 0" in code
    if mode.startswith("floor"):
        assert " // " in code and " % " in code and " - 1 if " in code


def test_negative_symint_remainder_codegen() -> None:
    values = (
        torch.empty((4, 128, 64), dtype=torch.bfloat16),
        torch.empty((4, 64, 64), dtype=torch.bfloat16),
    )
    code = _code((*values, torch.int32, 3, "scalar_index"), None, _signed_arithmetic)
    assert "!= 0" in code
    assert "_async_pointer" not in code


@pytest.mark.parametrize("divisor", [-3, 3])
def test_negative_symint_floor_codegen(divisor: int) -> None:
    values = (
        torch.empty((4, 128, 64), dtype=torch.bfloat16),
        torch.empty((4, 64, 64), dtype=torch.bfloat16),
    )
    code = _code(
        (*values, torch.int32, divisor, "scalar_floor_index"), None, _signed_arithmetic
    )
    assert " - 1 if " in code and " % " in code
    assert "_async_pointer" not in code


def test_nonnegative_symint_floor_preserves_index_fast_path() -> None:
    code = _code(
        (*_values("cpu"), torch.int32, 3, "scalar_positive_floor_index"),
        None,
        _signed_arithmetic,
    )
    assert "_async_pointer" in code
    assert " - 1 if " not in code


def test_scalar_float_floor_keeps_existing_lowering() -> None:
    graph = torch.fx.Graph()
    node = graph.call_function(operator.floordiv, (1.5, -3.0))
    node.meta["val"] = -1.0
    expression = _Expression.__new__(_Expression)
    assert expression.scalar(node) == "(1.5 // -3.0)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["scalar_index", "scalar_floor_index"])
@pytest.mark.parametrize("divisor", [-3, 3])
def test_negative_symint_remainder_runtime_exact(mode: str, divisor: int) -> None:
    values = (
        torch.arange(1, 5, device=DEVICE, dtype=torch.bfloat16)[:, None, None]
        .expand(4, 128, 64)
        .contiguous(),
        torch.eye(64, device=DEVICE, dtype=torch.bfloat16)[None, :, :]
        .expand(4, 64, 64)
        .contiguous(),
    )
    saved = tuple(value.clone() for value in values)
    function = _signed_arithmetic._bind_isolated(
        (*values, torch.int32, divisor, mode)
    ).compile_config(
        helion.Config(
            block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        )
    )
    selected = (
        torch.tensor([1, 2, 0, 1], device=DEVICE)
        if mode == "scalar_index"
        else torch.div(
            torch.arange(4, device=DEVICE) - 2, divisor, rounding_mode="floor"
        )
        + 1
    )
    expected = values[0].index_select(0, selected) * 0.5
    actual = function(*values, torch.int32, divisor, mode)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    repeated = function(*values, torch.int32, divisor, mode)
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    assert repeated.data_ptr() != actual.data_ptr()
    for value, before in zip(values, saved, strict=True):
        torch.testing.assert_close(value, before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "mode", ["floor", "floor_tensor", "scalar_floor_index", "remainder"]
)
def test_signed_arithmetic_real_cpu_compile(mode: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", __name__, mode],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "CUTE_DSL_ARCH": "sm_103a",
            "CUTE_DSL_KEEP_PTX": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "mode",
    [
        "floor",
        "floor_op",
        "floor_tensor",
        "floor_scalar_left",
        "remainder",
        "tensor",
        "scalar_left",
    ],
)
@pytest.mark.parametrize("divisor", [-3, 3])
def test_signed_arithmetic_runtime_exact(
    dtype: torch.dtype, mode: str, divisor: int
) -> None:
    values = _values(DEVICE)
    function = _signed_arithmetic._bind_isolated(
        (*values, dtype, divisor, mode)
    ).compile_config(
        helion.Config(
            block_sizes=[128, 64], num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"
        )
    )
    numerator = torch.arange(128, dtype=dtype) - 64
    denominator = (torch.arange(64, dtype=dtype) % 2) * 6 - 3
    if mode in ("floor", "floor_op"):
        expected = torch.div(numerator, divisor, rounding_mode="floor")[:, None]
    elif mode == "floor_tensor":
        expected = torch.div(
            numerator[:, None], denominator[None, :], rounding_mode="floor"
        )
    elif mode == "floor_scalar_left":
        expected = torch.div(-7, denominator, rounding_mode="floor")[None, :]
    elif mode == "remainder":
        expected = torch.remainder(numerator, divisor)[:, None]
    elif mode == "scalar_left":
        expected = torch.remainder(-7, denominator)[None, :]
    else:
        expected = torch.remainder(numerator[:, None], denominator[None, :])
    expected = expected.expand(2, 128, 64).to(device=DEVICE, dtype=torch.bfloat16)
    actual = function(*values, dtype, divisor, mode)
    repeated = function(*values, dtype, divisor, mode)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    assert actual.data_ptr() != repeated.data_ptr()
    for value in values:
        assert torch.count_nonzero(value) == 0


if __name__ == "__main__":
    args = (*_values("cpu"), torch.int64, -3, sys.argv[1])
    code = _code(args, None, _signed_arithmetic)
    ptx = _compile(code, args, None, "_signed_arithmetic")
    assert "tcgen05.mma" in ptx
    assert not torch.cuda.is_initialized()
