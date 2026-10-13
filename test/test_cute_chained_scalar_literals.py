from __future__ import annotations

import ast
import importlib
import math
import struct
from unittest.mock import patch

import cutlass
import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_row_registers import _args
from .test_cute_chained_row_registers import _config
from .test_cute_chained_row_registers import _poisoned_sequence
from helion._compiler.cute.chained_matmul import _Expression

ir = importlib.import_module("cutlass._mlir.ir")
func = importlib.import_module("cutlass._mlir.dialects.func")


def _literal(value: float) -> str:
    # Literal rendering must not depend on a current emitter or CUDA context.
    return _Expression.__new__(_Expression).scalar(value)


@pytest.mark.parametrize(
    "value", [False, True, 0, -1, 1 << 1024, 0.0, -0.0, 0.125, -127.25]
)
def test_finite_scalar_literal_source_is_unchanged(value: float) -> None:
    assert _literal(value) == repr(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.0])
@pytest.mark.parametrize("dtype", ["Float16", "BFloat16", "Float32"])
def test_nonfinite_scalar_literal_lowers_to_actual_cute_constant(
    value: float, dtype: str
) -> None:
    before = torch.cuda.is_initialized()
    expression = f"cutlass.{dtype}({_literal(value)})"
    assert not any(
        isinstance(node, ast.Name) and node.id in ("nan", "inf")
        for node in ast.walk(ast.parse(expression, mode="eval"))
    )
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp("constant", ([], []))
        with ir.InsertionPoint(function.add_entry_block()):
            result = eval(expression, {"cutlass": cutlass}).ir_value()
            assert result.owner.name == "arith.constant"
            actual = ir.FloatAttr(result.owner.attributes["value"]).value
            if math.isnan(value):
                assert math.isnan(actual)
            else:
                assert struct.pack("d", actual) == struct.pack("d", value)
            func.ReturnOp([])
        assert module.operation.verify()
    assert torch.cuda.is_initialized() is before


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("schedule", ["shared", "warp_rows"])
def test_true_poison_kernel_has_bound_nonfinite_literals_in_both_schedules(
    dtype: torch.dtype, schedule: str
) -> None:
    with _cpu_codegen():
        bound = _poisoned_sequence._bind_isolated(_args(dtype))
        source = bound.to_code(_config(schedule))
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Name) and node.id in ("nan", "inf")
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "nan"
        for node in ast.walk(tree)
    )
    assert ("chain_row_register" in source) is (schedule == "warp_rows")
