"""A host constant fill of a kernel output's unwritten remainder is written by the kernel."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.cute.host_remainder_fills import prove_host_remainder_fills
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = skipUnlessBackends("cute")


def _rows_plus_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0) + 1, x.size(1)], dtype=x.dtype, device=x.device)
    for i, j in hl.tile([x.size(0), x.size(1)]):
        out[i, j] = x[i, j] + 1
    out[-1, :] = 0
    return out


def _rows_plus_one_zeros(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros([x.size(0) + 1, x.size(1)], dtype=x.dtype, device=x.device)
    for i, j in hl.tile([x.size(0), x.size(1)]):
        out[i, j] = x[i, j] + 1
    out[-1, :] = 0
    return out


def _cols_tail_int(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), x.size(1) + 3], dtype=torch.int32, device=x.device)
    for i, j in hl.tile([x.size(0), x.size(1)]):
        out[i, j] = x[i, j].to(torch.int32)
    out[:, -3:] = 7
    return out


def _three_dim_tail(x: torch.Tensor) -> torch.Tensor:
    b, length, v = x.shape
    out = torch.empty([b, length + 1, v], dtype=x.dtype, device=x.device)
    for i, j in hl.tile([b, length]):
        for k in hl.tile(v):
            out[i, j, k] = x[i, j, k] * 2
    out[:, -1, :] = 1.5
    return out


def _read_between(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty([x.size(0) + 1, x.size(1)], dtype=x.dtype, device=x.device)
    for i, j in hl.tile([x.size(0), x.size(1)]):
        out[i, j] = x[i, j] + 1
    other = out * 2
    out[-1, :] = 0
    return out, other


def _kernel_loads_output(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0) + 1, x.size(1)], dtype=x.dtype, device=x.device)
    for i, j in hl.tile([x.size(0), x.size(1)]):
        out[i, j] = x[i, j] + 1
        out[i, j] = out[i, j] * 2
    out[-1, :] = 0
    return out


def _inexact_value(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0) + 1, x.size(1)], dtype=torch.int32, device=x.device)
    for i, j in hl.tile([x.size(0), x.size(1)]):
        out[i, j] = x[i, j].to(torch.int32)
    out[-1, :] = 2.5
    return out


def _wrong_remainder(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0) + 2, x.size(1)], dtype=x.dtype, device=x.device)
    for i, j in hl.tile([x.size(0), x.size(1)]):
        out[i, j] = x[i, j] + 1
    out[-1, :] = 0
    return out


def _kernel(fn: Callable[..., Any], **settings: Any) -> Kernel:
    return helion.kernel(
        fn,
        backend="cute",
        static_shapes=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
        **settings,
    )


def _code(fn: Callable[..., Any], *args: torch.Tensor) -> tuple[BoundKernel, str]:
    bound = _cpu_bind(_kernel(fn, autotune_effort="full"), args)
    return bound, bound.to_code(bound.config_spec.default_config())


def _defs(code: str) -> tuple[ast.FunctionDef, ast.FunctionDef]:
    """``(kernel, host)``: the decorated device function and the host function."""
    module = ast.parse(code)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    kernel = next(node for node in functions if node.decorator_list)
    host = next(node for node in functions if not node.decorator_list)
    return kernel, host


def _host(code: str) -> str:
    return ast.unparse(_defs(code)[1])


@skipIfRefEager("generates code")
@pytest.mark.parametrize(
    "fn,fill,call",
    [
        (
            _rows_plus_one,
            "out[-1, :] = 0",
            "_cute_remainder_fill(out, dim=0, remainder=1, value=0, is_float=True)",
        ),
        (
            _cols_tail_int,
            "out[:, -3:] = 7",
            "_cute_remainder_fill(out, dim=1, remainder=3, value=7, is_float=False)",
        ),
        (
            _three_dim_tail,
            "out[:, -1, :] = 1.5",
            "_cute_remainder_fill(out, dim=1, remainder=1, value=1.5, is_float=True)",
        ),
    ],
)
def test_fill_moves_into_the_kernel(
    fn: Callable[..., Any], fill: str, call: str
) -> None:
    with _mock_cuda_unavailable():
        shape = (2, 16, 64) if fn is _three_dim_tail else (64, 128)
        bound, code = _code(fn, torch.empty(shape))
        assert bound.host_function is not None
        with bound.env:
            proof = prove_host_remainder_fills(
                bound.host_function, bound.env, bound.config_spec.default_config()
            )
        assert [
            (site.name, site.dim, site.remainder, site.value) for site in proof
        ] == [
            (
                "out",
                *[
                    int(part)
                    for part in call.split("dim=")[1]
                    .split(", value")[0]
                    .split(", remainder=")
                ],
                ast.literal_eval(call.split("value=")[1].split(",")[0]),
            )
        ]
        assert fill not in _host(code)
        kernel, _host_def = _defs(code)
        # The device fill is the last statement of the kernel body.
        assert ast.unparse(kernel.body[-1]) == call
        assert ast.unparse(kernel).count("_cute_remainder_fill") == 1


@skipIfRefEager("generates code")
def test_zero_factory_and_remainder_fill_both_leave_the_host() -> None:
    with _mock_cuda_unavailable():
        _bound, code = _code(_rows_plus_one_zeros, torch.empty((64, 128)))
        host = _host(code)
        assert "torch.empty(" in host and "torch.zeros(" not in host
        assert "out[-1, :] = 0" not in host
        assert (
            "_cute_remainder_fill(out, dim=0, remainder=1, value=0, is_float=True)"
            in code
        )


@skipIfRefEager("generates code")
@pytest.mark.parametrize(
    "fn,fill",
    [
        (_read_between, "out[-1, :] = 0"),
        (_kernel_loads_output, "out[-1, :] = 0"),
        (_inexact_value, "out[-1, :] = 2.5"),
        (_wrong_remainder, "out[-1, :] = 0"),
    ],
    ids=[
        "host read between",
        "kernel loads the tensor",
        "inexact value",
        "rows left unwritten",
    ],
)
def test_unproven_fills_stay_on_the_host(fn: Callable[..., Any], fill: str) -> None:
    with _mock_cuda_unavailable():
        bound, code = _code(fn, torch.empty((64, 128)))
        assert bound.host_function is not None
        with bound.env:
            assert not prove_host_remainder_fills(
                bound.host_function, bound.env, bound.config_spec.default_config()
            )
        assert fill in _host(code)
        assert "_cute_remainder_fill" not in code


def _reference(fn: Callable[..., Any], x: torch.Tensor) -> torch.Tensor:
    if fn in (_rows_plus_one, _rows_plus_one_zeros):
        return torch.cat([x + 1, torch.zeros_like(x[:1])])
    if fn is _cols_tail_int:
        tail = torch.full((x.size(0), 3), 7, dtype=torch.int32, device=x.device)
        return torch.cat([x.to(torch.int32), tail], dim=1)
    assert fn is _three_dim_tail
    tail = torch.full_like(x[:, :1], 1.5)
    return torch.cat([x * 2, tail], dim=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "fn,shape,dtype",
    [
        (_rows_plus_one, (64, 128), torch.float32),
        (_rows_plus_one, (33, 1000), torch.bfloat16),
        (_rows_plus_one_zeros, (5, 7), torch.float16),
        (_cols_tail_int, (64, 128), torch.float32),
        (_three_dim_tail, (2, 16, 64), torch.float32),
        (_three_dim_tail, (1, 64, 1024), torch.bfloat16),
    ],
)
def test_device_fill_matches_the_reference(
    fn: Callable[..., Any], shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    torch.manual_seed(len(shape))
    x = torch.randn(shape, device=DEVICE).to(dtype)
    kernel = _kernel(fn, autotune_effort="none")
    expected = _reference(fn, x)
    for _ in range(2):
        result = kernel(x)
        assert result.dtype == expected.dtype and result.shape == expected.shape
        assert torch.equal(result, expected)
