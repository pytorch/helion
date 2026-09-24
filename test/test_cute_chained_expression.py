from __future__ import annotations

import ast
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])
CPU_DEVICE = torch.device("cpu")
_SCHEDULES = ("coalesced", "cp_async", "cp_async_register")
_OPERATIONS = ("left_add", "right_add", "both_add", "bias", "exp")


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _expression_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    a_bias: torch.Tensor,
    b_bias: torch.Tensor,
    operation: hl.constexpr,
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, 16, 32]):
        bi = batch.begin
        kk = hl.arange(k)
        qq = hl.arange(q)
        left = a[bi, row, kk].float()
        right = b[bi, qq, kk].float()
        if operation == "left_add" or operation == "both_add":
            left = left + 1.0
        if operation == "right_add" or operation == "both_add":
            right = right + 1.0
        if operation == "bias":
            left = left + a_bias[bi, row][:, None].float()
            right = right + b_bias[bi, qq][:, None].float()
        if operation == "exp":
            left = torch.exp(left)
            right = torch.exp(right)
        first = hl.dot(left.to(a.dtype), right.to(a.dtype).T)
        # Both operands resurrect a padded zero at the second contraction too.
        # This exercises the register bridge as well as shared-memory staging.
        second = hl.dot(
            (first + 1.0).to(a.dtype), (v[bi, qq, col].float() + 1.0).to(a.dtype)
        )
        out[bi, row, col] = second.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _negative_view_chain(
    a: torch.Tensor, b: torch.Tensor, v: torch.Tensor, view: hl.constexpr
) -> torch.Tensor:
    batches, m, k = a.shape
    q, n = v.shape[1:]
    out = torch.empty((batches, m, n), device=a.device, dtype=a.dtype)
    for batch, row, col in hl.tile([batches, m, n], block_size=[1, 16, 32]):
        bi = batch.begin
        kk = hl.arange(k)
        qq = hl.arange(q)
        left = a[bi, row, kk]
        right = b[bi, qq, kk]
        if view == "identity":
            left = left.permute(-2, -1)
            right = right.T
        elif view == "permute":
            right = right.permute(-1, -2)
        else:
            right = right.transpose(-1, -2)
        first = hl.dot(left, right)
        out[bi, row, col] = hl.dot(first.to(a.dtype), v[bi, qq, col]).to(a.dtype)
    return out


def _config(schedule: str) -> helion.Config:
    return helion.Config(num_warps=4, cute_chained_mma_schedule=schedule)


def _cpu_code(kernel: Any, args: tuple[Any, ...], schedule: str) -> str:
    with (
        patch_cute_mma_support(),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 3),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        kernel.reset()
        return kernel.bind(args).to_code(_config(schedule))


def _inputs(device: Any, *, seed: int | None = None) -> tuple[torch.Tensor, ...]:
    shapes = ((2, 35, 49), (2, 49, 49), (2, 49, 24))
    if seed is None:
        values = tuple(
            torch.zeros(s, device=device, dtype=torch.bfloat16) for s in shapes
        )
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        values = tuple(
            torch.randn(s, device=device, dtype=torch.bfloat16, generator=generator)
            * 0.03
            for s in shapes
        )
    return (
        *values,
        torch.ones((2, 35), device=device, dtype=torch.bfloat16),
        torch.ones((2, 49), device=device, dtype=torch.bfloat16),
    )


def _reference(args: tuple[torch.Tensor, ...], operation: str) -> torch.Tensor:
    a, b, v, a_bias, b_bias = args
    left, right = a.float(), b.float()
    if operation in ("left_add", "both_add"):
        left = left + 1.0
    if operation in ("right_add", "both_add"):
        right = right + 1.0
    if operation == "bias":
        left = left + a_bias.float().unsqueeze(-1)
        right = right + b_bias.float().unsqueeze(-1)
    if operation == "exp":
        left, right = left.exp(), right.exp()
    first = (
        left.to(a.dtype).double() @ right.to(a.dtype).double().transpose(-1, -2)
    ).float()
    middle = (first + 1.0).to(a.dtype).double()
    last = (v.float() + 1.0).to(a.dtype).double()
    return (middle @ last).to(a.dtype)


def _check(
    fn: Any, args: tuple[torch.Tensor, ...], mode: str, expected: torch.Tensor
) -> None:
    originals = tuple(x.clone() for x in args)
    output = fn(*args, mode)
    repeated = fn(*args, mode)
    assert output.dtype is torch.bfloat16
    torch.testing.assert_close(output, expected, atol=0.01, rtol=0.01)
    torch.testing.assert_close(repeated, output, atol=0, rtol=0)
    assert output.data_ptr() != repeated.data_ptr()
    for x, original in zip(args, originals, strict=True):
        torch.testing.assert_close(x, original, atol=0, rtol=0)


def _predicate_value(condition: ast.expr, **coordinates: int) -> bool:
    values = {
        node.id: 0
        for node in ast.walk(condition)
        if isinstance(node, ast.Name) and node.id.startswith("chain_origin_")
    }
    values.update(coordinates)
    # Only the emitted integer-coordinate predicate is evaluated, never GPU
    # operations or generated expression values.
    return bool(
        eval(compile(ast.Expression(condition), "<operand-domain>", "eval"), {}, values)
    )


@pytest.mark.parametrize("schedule", _SCHEDULES)
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_expression_codegen_masks_after_pointwise(
    operation: str, schedule: str
) -> None:
    code = _cpu_code(_expression_chain, (*_inputs("cpu"), operation), schedule)
    assert "chain_0_mma" in code and "chain_1_mma" in code
    assignments = {
        target.value.id: node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
    }
    for name in ("chain_0_a", "chain_0_b", "chain_1_b"):
        value = assignments[name]
        assert isinstance(value, ast.IfExp), (name, ast.unparse(value))
        # The final dtype conversion, not merely each global load, is masked.
        assert isinstance(value.body, ast.Call)
        assert ast.unparse(value.orelse) == "cutlass.BFloat16(0)"
        assert "< 49" in ast.unparse(value.test)
        load = "chain_0_load" if name.startswith("chain_0") else "chain_1_load"
        valid = 48 if name != "chain_1_b" else 48 * 32
        invalid = 49 if name != "chain_1_b" else 49 * 32
        assert _predicate_value(value.test, **{load: valid})
        assert not _predicate_value(value.test, **{load: invalid})
    name = "chain_1_a_bridge_values" if schedule == "cp_async_register" else "chain_1_a"
    value = assignments[name]
    assert isinstance(value, ast.IfExp)
    assert ast.unparse(value.orelse) == "cutlass.BFloat16(0)"
    assert "< 49" in ast.unparse(value.test)
    assert "< 35" in ast.unparse(value.test)
    if schedule == "cp_async_register":
        assert _predicate_value(
            value.test, chain_1_a_bridge_row=0, chain_1_a_bridge_col=48
        )
        assert not _predicate_value(
            value.test, chain_1_a_bridge_row=0, chain_1_a_bridge_col=49
        )
    else:
        assert _predicate_value(value.test, chain_1_load=48)
        assert not _predicate_value(value.test, chain_1_load=49)
    assert ("chain_0_c_ptr" not in code) is (schedule == "cp_async_register")


@pytest.mark.parametrize("schedule", _SCHEDULES)
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_expression_padded_reduction_lanes(operation: str, schedule: str) -> None:
    args = _inputs(DEVICE)
    _expression_chain.reset()
    fn = _expression_chain.bind((*args, operation)).compile_config(_config(schedule))
    # Exact zeros make the unmasked-lane error deterministic and very large:
    # (49 + 1) * 49 rather than (64 + 1) * 64 for both_add/bias/exp.
    _check(fn, args, operation, _reference(args, operation))
    for seed in range(5):
        args = _inputs(DEVICE, seed=seed)
        _check(fn, args, operation, _reference(args, operation))


@pytest.mark.parametrize("schedule", _SCHEDULES)
@pytest.mark.parametrize("view", ["identity", "permute", "transpose"])
def test_negative_view_codegen(view: str, schedule: str) -> None:
    code = _cpu_code(_negative_view_chain, (*_inputs("cpu")[:3], view), schedule)
    assert "chain_0_mma" in code and "chain_1_mma" in code


@pytest.mark.parametrize("schedule", _SCHEDULES)
@pytest.mark.parametrize("view", ["identity", "permute", "transpose"])
def test_negative_view_operands(view: str, schedule: str) -> None:
    args = _inputs(DEVICE, seed=0)[:3]
    _negative_view_chain.reset()
    fn = _negative_view_chain.bind((*args, view)).compile_config(_config(schedule))
    for seed in range(5):
        a, b, v = _inputs(DEVICE, seed=seed)[:3]
        first = (a.double() @ b.double().transpose(-1, -2)).to(a.dtype)
        expected = (first.double() @ v.double()).to(a.dtype)
        _check(fn, (a, b, v), view, expected)


@pytest.mark.parametrize("schedule", ["cp_async", "cp_async_register"])
def test_async_codegen_large_offsets_promote_before_multiply(schedule: str) -> None:
    # Fake CPU storage avoids allocating three multi-gigabyte tensors or CUDA
    # initialization. The first dimension needs Int64 byte/element offsets.
    with FakeTensorMode():
        args = tuple(
            torch.empty((1048577, 64, 64), device=CPU_DEVICE, dtype=torch.bfloat16)
            for _ in range(3)
        )
        code = _cpu_code(_negative_view_chain, (*args, "identity"), schedule)
    pointers = [
        node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id.endswith("_async_pointer")
            for target in node.targets
        )
    ]
    assert len(pointers) == 3
    for pointer in pointers:
        products = [
            node
            for node in ast.walk(pointer)
            if isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mult)
            and any(
                isinstance(x, ast.Constant) and x.value == 4096
                for x in (node.left, node.right)
            )
        ]
        assert products, ast.unparse(pointer)
        for product in products:
            operand = (
                product.right
                if isinstance(product.left, ast.Constant)
                else product.left
            )
            assert isinstance(operand, ast.Call), ast.unparse(product)
            assert ast.unparse(operand.func) == "cutlass.Int64", ast.unparse(product)
