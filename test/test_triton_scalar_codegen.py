from __future__ import annotations

import ast
from contextlib import ExitStack
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import sympy
import torch
from torch.utils._sympy.functions import PowByNatural
from torch.utils._sympy.value_ranges import ValueRanges

import helion
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion._utils import triton_is_available
import helion.language as hl
from helion.runtime.settings import _get_backend

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.skipif(
    not triton_is_available(), reason="Triton is required for source generation"
)


@pytest.fixture
def cpu_codegen() -> Iterator[None]:
    with ExitStack() as stack:
        for name, value in (
            ("torch.cuda.is_available", False),
            ("helion._compat._supports_maxnreg", False),
            ("helion._compat._supports_tensor_descriptor", False),
            ("helion._compat._is_hip", False),
            ("helion.language.loops.use_tileir_tunables", False),
            ("helion.autotuner.config_spec.num_compute_units", 128),
            ("helion.runtime.kernel.target_device_capability", (9, 0)),
            ("helion._compiler.compile_environment.target_device_capability", (9, 0)),
            ("helion.runtime.get_num_sm", 132),
        ):
            stack.enter_context(patch(name, return_value=value))
        stack.enter_context(
            patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden"))
        )
        yield


def _device_function(code: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )


def _native_kernel(kernel):
    return helion.kernel(
        kernel.fn,
        backend=_get_backend(),
        static_shapes=True,
        autotune_effort="none",
    )


@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def _config_derived_iota(x: torch.Tensor, stepped: hl.constexpr):
    block = hl.register_block_size(x.size(1))
    count = (x.size(1) + block - 1) // block
    scratch = torch.zeros((x.size(0), count), device=x.device, dtype=x.dtype)
    out = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
    for row in hl.tile(x.size(0)):
        values = scratch[row, :]
        if stepped:
            indices = hl.arange(3, 3 + 2 * values.size(-1), 2, dtype=torch.int64)
        else:
            indices = hl.arange(values.size(-1))
        out[row] = (values + indices[None, :].float()).sum(-1)
    return out


def _iota_config(bound, block, reduction_block):
    config = bound.config_spec.default_config()
    config["block_sizes"][0] = block
    config.config["reduction_loops"] = [reduction_block]
    return config


@pytest.mark.parametrize("stepped", [False, True])
@pytest.mark.parametrize("reduction_block", [None, 2])
@pytest.mark.parametrize("block", [16, 32])
@skipIfRefEager("inspects generated device code")
def test_config_derived_iota_codegen(cpu_codegen, block, reduction_block, stepped):
    bound = _config_derived_iota._bind_isolated((torch.ones(3, 65), stepped))
    code = bound.to_code(_iota_config(bound, block, reduction_block))
    tree = ast.parse(code)
    function = _device_function(code)
    constexprs = {
        arg.arg
        for arg in function.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation) == "tl.constexpr"
    }
    constexprs.update(
        node.targets[0].id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "tl.constexpr"
    )
    for call in ast.walk(function):
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "tl.arange":
            end = call.args[1]
            assert isinstance(end, ast.Constant) or (
                isinstance(end, ast.Name) and end.id in constexprs
            ), ast.unparse(call)
    assert "< count" in code
    if reduction_block is not None:
        assert "rindex_" in code and "roffset_" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipUnlessBackends(["triton", "tileir"])
@pytest.mark.parametrize("stepped", [False, True])
@pytest.mark.parametrize("reduction_block", [None, 2])
@pytest.mark.parametrize("block", [16, 32])
@skipIfRefEager("requires explicit block and reduction configurations")
def test_config_derived_iota_values(block, reduction_block, stepped):
    x = torch.zeros((3, 65), device=DEVICE)
    kernel = _native_kernel(_config_derived_iota)
    bound = kernel._bind_isolated((x, stepped))
    config = _iota_config(bound, block, reduction_block)
    indices = torch.arange((65 + block - 1) // block, device=x.device)
    if stepped:
        indices = 3 + 2 * indices
    expected = indices.sum().float().expand(3)
    torch.testing.assert_close(bound.compile_config(config)(x, stepped), expected)


@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def _scalar_integer_powers(
    x: torch.Tensor,
    base: hl.constexpr,
    begin: hl.constexpr,
    end: hl.constexpr,
    step: hl.constexpr,
):
    out = torch.empty((x.numel(), 64), device=x.device, dtype=torch.int64)
    for row in hl.tile(x.numel()):
        for exponent in range(begin, end, step):
            out[row, exponent] = x[row] + base**exponent
    return out


@pytest.mark.parametrize(
    "base,begin,end,step",
    [(2, 0, 63, 1), (2, 3, 63, 2), (2, 62, -1, -1), (4, 0, 32, 1), (8, 0, 21, 1)],
)
@skipIfRefEager("inspects generated device code")
def test_scalar_integer_power_codegen(cpu_codegen, base, begin, end, step):
    bound = _scalar_integer_powers._bind_isolated(
        (torch.ones(5, dtype=torch.int64), base, begin, end, step)
    )
    code = bound.to_code(bound.config_spec.default_config())
    shifts = [
        node
        for node in ast.walk(_device_function(code))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.LShift)
    ]
    assert shifts, code
    for shift in shifts:
        assert ast.unparse(shift.left) == "tl.full((), 1, tl.int64)"
        names = {
            node.id for node in ast.walk(shift.right) if isinstance(node, ast.Name)
        }
        assert len(names) == 1
        name = names.pop()
        expression = compile(ast.Expression(shift), "<generated shift>", "eval")
        for exponent in range(begin, end, step):
            actual = eval(
                expression,
                {"__builtins__": {}},
                {
                    name: exponent,
                    "tl": SimpleNamespace(
                        full=lambda shape, fill, dtype: dtype(fill), int64=int
                    ),
                },
            )
            assert actual == base**exponent
            assert isinstance(actual, int)


@pytest.mark.parametrize(
    "base,bounds",
    [
        (3, ValueRanges(0, 10)),
        (2, ValueRanges(0, 63)),
        (2, ValueRanges(-1, 30)),
        (2, ValueRanges.unknown_int()),
    ],
)
def test_unproved_integer_powers_keep_existing_expression(base, bounds):
    from helion._compiler.integer_power import lower_integer_powers

    exponent = sympy.Symbol("exponent", integer=True)
    expression = PowByNatural(base, exponent)
    assert (
        lower_integer_powers(expression, {exponent: bounds}, backend="triton")
        is expression
    )


@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def _host_integer_power(x: torch.Tensor, bits: int):
    value = 1 << bits
    out = torch.empty_like(x)
    for row in hl.tile(x.numel()):
        out[row] = x[row] + value
    return out


@pytest.mark.parametrize("bits", [5, 31, 62])
@skipIfRefEager("inspects generated device code")
def test_host_integer_power_keeps_scalar_origin(cpu_codegen, bits):
    bound = _host_integer_power._bind_isolated((torch.ones(3, dtype=torch.int64), bits))
    code = bound.to_code(bound.config_spec.default_config())
    device = _device_function(code)
    args = [arg.arg for arg in device.args.args]
    assert "value" in args and "bits" not in args
    assert "value = 1 << bits" in code
    assert not any(isinstance(node, ast.LShift) for node in ast.walk(device))


@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def _integer_bitmasks(x: torch.Tensor):
    out = torch.empty((x.numel(), 31), device=x.device, dtype=torch.int32)
    for row in hl.tile(x.numel()):
        for bit in range(31):
            out[row, bit] = x[row] | (1 << (30 - bit))
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipUnlessBackends(["triton", "tileir"])
def test_integer_power_preserves_int64_values():
    x = torch.arange(17, dtype=torch.int64, device=DEVICE)
    expected = (
        x[:, None]
        + torch.tensor(
            [1 << exponent for exponent in range(63)],
            dtype=torch.int64,
            device=x.device,
        )[None, :]
    )
    actual = _native_kernel(_scalar_integer_powers)(x, 2, 0, 63, 1)
    torch.testing.assert_close(actual[:, :63], expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@skipUnlessBackends(["triton", "tileir"])
def test_integer_power_supports_bitwise_consumer():
    x = torch.arange(17, dtype=torch.int32, device=DEVICE)
    expected = (
        x[:, None]
        | torch.tensor(
            [1 << (30 - bit) for bit in range(31)], dtype=torch.int32, device=x.device
        )[None, :]
    )
    torch.testing.assert_close(
        _native_kernel(_integer_bitmasks)(x), expected, rtol=0, atol=0
    )
