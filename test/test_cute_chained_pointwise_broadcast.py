from __future__ import annotations

from typing import Any

import pytest
import torch

from .test_cute_chained_pointwise import _args
from .test_cute_chained_pointwise import _code
from .test_cute_chained_pointwise import _config
from .test_cute_chained_pointwise import _pointwise_dot
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def _broadcast_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    a, b, scale, bias, transpose = args
    return a, b[:, :1, :].expand_as(b), scale, bias, transpose


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pointwise_zero_outer_stride_codegen(dtype: torch.dtype) -> None:
    args = _broadcast_args(_args("cpu", "dense", dtype))
    assert args[1].stride()[-2:] == (0, 1)
    code = _code(args)
    assert "chain_0_b_pointwise_copy =" in code
    assert "stride=(0, 1)" in code
    assert "_pointwise_leaf_0_pointer.toint() % 16 == 0" in code
    assert ".layout.stride[1] == 0" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("compile_broadcast", [False, True])
def test_pointwise_zero_outer_stride_runtime(
    dtype: torch.dtype, compile_broadcast: bool
) -> None:
    a, b, scale, bias, transpose = _args(DEVICE, "dense", dtype)
    # A unit factor keeps an incorrect B row visible despite the +1 epilogue.
    dense = a, b, torch.zeros_like(scale), torch.ones_like(bias), transpose
    broadcast = _broadcast_args(dense)
    compile_args = broadcast if compile_broadcast else dense
    run = _pointwise_dot._bind_isolated(compile_args).compile_config(_config())
    for args in (broadcast, dense, broadcast):
        a, b, scale, bias, _ = args
        frozen = tuple(value.clone() for value in args[:-1])
        left = (a.float() + 1.0).to(dtype)
        right = (
            b.float() * (scale.float().exp() * bias.float()[:, None])[:, :, None] + 1.0
        ).to(dtype)
        expected = (left.float() @ right.float()).to(dtype)
        actual = run(*args)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
        repeated = run(*args)
        assert actual.data_ptr() != repeated.data_ptr()
        torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
        for before, value in zip(frozen, args[:-1], strict=True):
            torch.testing.assert_close(before, value, atol=0, rtol=0)
