"""The CuTe backend's ``FastMathUnused`` warning on CPU renders: which
exponentials and divisions count, by dtype.  The warning's tests that run
kernels are in ``test_fast_math_warning.py``."""

from __future__ import annotations

import contextlib
import io

import pytest
import torch

import helion
from helion._testing import skipIfRefEager
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _exp(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.exp(x[tile])
    return out


def _exp2(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.exp2(x[tile])
    return out


def _divide(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] / y[tile]
    return out


@skipIfRefEager("Warning emitted in compile mode only")
def test_cute_float64_exponentials_do_not_count() -> None:
    """The CuTe backend computes a float64 exp or exp2 the same way with or
    without fast_math (there is no float64 ``ex2.approx``), so a kernel whose
    only exponential is float64 is not told to set it; a float32 one still
    counts."""
    for fn, name in ((_exp, "exp"), (_exp2, "exp2")):
        for dtype in (torch.float64, torch.float32):
            buffer = io.StringIO()
            with contextlib.redirect_stderr(buffer):
                cpu_bind(
                    helion.kernel(fn, backend="cute", static_shapes=True),
                    (torch.empty([64, 128], dtype=dtype),),
                )
            text = buffer.getvalue()
            if dtype == torch.float64:
                assert "FastMathUnused" not in text, name
            else:
                assert f"WARNING[FastMathUnused]: This kernel computes {name}," in text


@skipIfRefEager("Warning emitted in compile mode only")
def test_cute_divisions_count_with_an_fp32_result_only() -> None:
    """The CuTe backend divides approximately under fast_math only for an
    fp32 result; a float64 or bf16 true division stays IEEE, so a kernel
    whose only division is one of those is not told to set it."""
    for dtype in (torch.float64, torch.bfloat16, torch.float32):
        x = torch.empty([64, 128], dtype=dtype)
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            cpu_bind(helion.kernel(_divide, backend="cute", static_shapes=True), (x, x))
        text = buffer.getvalue()
        with cpu_target():
            fast = helion.kernel(
                _divide, backend="cute", static_shapes=True, fast_math=True
            ).bind((x, x))
            code = fast.to_code(fast.config_spec.default_config())
        assert ("approx=True" in code) is (dtype == torch.float32), dtype
        if dtype == torch.float32:
            assert (
                "WARNING[FastMathUnused]: This kernel computes fp32 division," in text
            )
        else:
            assert "FastMathUnused" not in text, dtype
