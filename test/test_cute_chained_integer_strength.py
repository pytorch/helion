"""Signed literal strength reduction, without changing the scalar-index path."""

from __future__ import annotations

import ast
from functools import cache
import importlib
import os
import subprocess
import sys
from unittest.mock import patch

import cutlass
import pytest
import sympy
import torch

from test.test_cute_chained_tcgen05 import _signed_arithmetic
from test.test_cute_chained_tcgen05 import _signed_values
from test.test_cute_chained_tcgen05 import _tcgen_compile
from test.test_cute_chained_tcgen05 import _tcgen_guard_code
from test.test_cute_chained_tcgen05 import (
    test_signed_arithmetic_runtime_exact as _signed_runtime_oracle,
)

from helion._compiler.cute.chained_matmul import _power_of_two_divisor_shift

_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)
_MODES = ("floor", "floor_op", "remainder")
_DIVISORS = (1, 2, 8, 64)
ir = importlib.import_module("cutlass._mlir.ir")
func = importlib.import_module("cutlass._mlir.dialects.func")
passmanager = importlib.import_module("cutlass._mlir.passmanager")


@pytest.mark.parametrize("dtype", _DTYPES)
def test_all_in_range_literal_powers_and_signed_cast_boundaries(dtype):
    bits = torch.iinfo(dtype).bits
    for shift in range(bits - 1):
        assert _power_of_two_divisor_shift(1 << shift, dtype) == shift
    # These are powers of two, but the original signed operand cast would
    # make them negative or wrap. They must not select the positive rewrite.
    for divisor in (1 << (bits - 1), 1 << bits, 1 << (bits + 1)):
        assert _power_of_two_divisor_shift(divisor, dtype) is None


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "divisor",
    (True, False, 1.0, 8.0, "8", None, 0, -1, -2, -8, 3, 6, 127, sympy.Integer(8)),
)
def test_nonliteral_nonpositive_and_nonpower_divisors_are_rejected(dtype, divisor):
    assert _power_of_two_divisor_shift(divisor, dtype) is None


@pytest.mark.parametrize("dtype", _DTYPES)
def test_integer_subclass_and_tensor_divisor_do_not_become_literals(dtype):
    class IntegerSubclass(int):
        pass

    for divisor in (IntegerSubclass(8), torch.tensor(8, dtype=dtype)):
        assert _power_of_two_divisor_shift(divisor, dtype) is None
    graph = torch.fx.Graph()
    node = graph.placeholder("divisor")
    node.meta["val"] = torch.tensor(8, dtype=dtype)
    assert _power_of_two_divisor_shift(node, dtype) is None


@pytest.mark.parametrize("dtype", _DTYPES)
def test_exact_torch_boundary_proof_for_every_in_range_power(dtype):
    limits = torch.iinfo(dtype)
    for shift in range(limits.bits - 1):
        divisor = 1 << shift
        candidates = {
            limits.min,
            limits.min + 1,
            limits.min + divisor - 1,
            limits.min + divisor,
            limits.max - divisor,
            limits.max - 1,
            limits.max,
            -1,
            0,
            1,
        }
        candidates.update(
            multiple * divisor + delta
            for multiple in (-2, -1, 1, 2)
            for delta in (-1, 0, 1)
        )
        values = sorted(v for v in candidates if limits.min <= v <= limits.max)
        tensor = torch.tensor(values, dtype=dtype)
        expected_floor = torch.div(tensor, divisor, rounding_mode="floor")
        expected_remainder = torch.remainder(tensor, divisor)
        torch.testing.assert_close(tensor >> shift, expected_floor, atol=0, rtol=0)
        torch.testing.assert_close(
            tensor & (divisor - 1), expected_remainder, atol=0, rtol=0
        )
        assert expected_floor.tolist() == [value >> shift for value in values]
        assert expected_remainder.tolist() == [
            value & (divisor - 1) for value in values
        ]


@cache
def _source(dtype, mode, divisor):
    return _tcgen_guard_code(
        (*_signed_values("cpu"), dtype, divisor, mode), None, _signed_arithmetic
    )


def _reduced_expression(dtype, mode, divisor):
    """Select the actual emitted value, not an unrelated address bit operation."""
    expected_op = ast.BitAnd if mode == "remainder" else ast.RShift
    expected_rhs = divisor - 1 if mode == "remainder" else divisor.bit_length() - 1
    matches = []
    for assignment in ast.walk(ast.parse(_source(dtype, mode, divisor))):
        if not isinstance(assignment, ast.Assign):
            continue
        expression = assignment.value
        binary = (
            expression.args[0]
            if isinstance(expression, ast.Call) and expression.args
            else expression
        )
        if (
            isinstance(binary, ast.BinOp)
            and isinstance(binary.op, expected_op)
            and isinstance(binary.left, ast.Name)
            and binary.left.id.startswith("chain_value_")
            and isinstance(binary.right, ast.Constant)
            and binary.right.value == expected_rhs
        ):
            matches.append((expression, binary.left.id))
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("divisor", _DIVISORS)
def test_literal_codegen_preserves_operand_and_result_dtype(dtype, mode, divisor):
    code = _source(dtype, mode, divisor)
    assert "chain_0_mma" in code
    assert " - 1 if " not in code
    expression, left = _reduced_expression(dtype, mode, divisor)
    dtype_name = f"cutlass.Int{torch.iinfo(dtype).bits}"
    assert isinstance(expression, ast.Call)
    assert ast.unparse(expression.func) == dtype_name
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    operand = assignments[left]
    assert isinstance(operand, ast.Call)
    assert ast.unparse(operand.func) == dtype_name
    # The original RHS cast is retained even though its result is now unused.
    assert f"{dtype_name}({divisor})" in code


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("divisor", (-8, -3, 3, 7))
def test_negative_or_nonpower_codegen_keeps_signed_correction(dtype, mode, divisor):
    code = _source(dtype, mode, divisor)
    assert "chain_0_mma" in code and "!= 0" in code and " % " in code
    if mode != "remainder":
        assert " // " in code and " - 1 if " in code
    else:
        assert " + " in code and " if " in code


@pytest.mark.parametrize("mode", ("floor_tensor", "tensor"))
def test_tensor_divisor_keeps_original_correction(mode):
    code = _source(torch.int64, mode, 8)
    assert "!= 0" in code and " % " in code
    if mode == "floor_tensor":
        assert " - 1 if " in code


def test_positive_literal_does_not_rewrite_scalar_symint_floor():
    values = (
        torch.empty((4, 128, 64), dtype=torch.bfloat16),
        torch.empty((4, 64, 64), dtype=torch.bfloat16),
    )
    code = _tcgen_guard_code(
        (*values, torch.int64, 8, "scalar_floor_index"), None, _signed_arithmetic
    )
    assert " - 1 if " in code and " % " in code
    assert "_async_pointer" not in code


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("divisor", _DIVISORS)
def test_actual_cute_ir_is_signed_and_preserves_result_dtype(dtype, mode, divisor):
    expression, left = _reduced_expression(dtype, mode, divisor)
    bits = torch.iinfo(dtype).bits
    cute_dtype = {
        8: cutlass.Int8,
        16: cutlass.Int16,
        32: cutlass.Int32,
        64: cutlass.Int64,
    }[bits]
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            scalar_type = ir.IntegerType.get_signless(bits)
            function = func.FuncOp("signed_value", ([scalar_type], [scalar_type]))
        with ir.InsertionPoint(function.add_entry_block()):
            operand = cute_dtype(function.entry_block.arguments[0])
            result = eval(ast.unparse(expression), {"cutlass": cutlass, left: operand})
            assert type(result) is cute_dtype
            func.ReturnOp([result.ir_value()])
        assert module.operation.verify()
        source = str(module)
        assert ("arith.andi" if mode == "remainder" else "arith.shrsi") in source
        assert "arith.shrui" not in source
        if bits < 32:
            # CuTe bitops with Python literals promote narrow operands. The
            # emitted final cast must undo that promotion before the next node.
            assert "arith.extsi" in source and "arith.trunci" in source
    assert torch.cuda.is_initialized() is before


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("divisor", (1, 2, 8, 64, 1 << 62))
def test_actual_cute_constant_fold_of_int64_minimum(mode, divisor):
    expression, left = _reduced_expression(torch.int64, mode, divisor)
    minimum = torch.iinfo(torch.int64).min
    expected = minimum % divisor if mode == "remainder" else minimum // divisor
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp("minimum", ([], [ir.IntegerType.get_signless(64)]))
        with ir.InsertionPoint(function.add_entry_block()):
            result = eval(
                ast.unparse(expression),
                {"cutlass": cutlass, left: cutlass.Int64(minimum)},
            )
            assert type(result) is cutlass.Int64
            func.ReturnOp([result.ir_value()])
        passmanager.PassManager.parse("builtin.module(canonicalize)").run(
            module.operation
        )
        assert module.operation.verify()
        returned = list(function.entry_block.operations)[-1].operation.operands[0]
        assert returned.owner.name == "arith.constant"
        assert returned.owner.attributes["value"].value == expected
    assert torch.cuda.is_initialized() is before


@pytest.mark.parametrize("mode", _MODES)
def test_full_generated_kernel_cpu_ptx_compile(mode):
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
@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("divisor", _DIVISORS)
def test_gpu_literal_arithmetic_original_oracle(dtype, mode, divisor):
    # Reuse the existing native-precision oracle unchanged: exact result,
    # repeated output, distinct output allocation and input immutability.
    _signed_runtime_oracle(dtype, mode, divisor)


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    mode = sys.argv[1]
    assert mode in _MODES
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")):
        arguments = (*_signed_values("cpu"), torch.int64, 8, mode)
        code = _tcgen_guard_code(arguments, None, _signed_arithmetic)
        ptx = _tcgen_compile(code, arguments, None, "_signed_arithmetic")
        assert "tcgen05.mma" in ptx
    assert not torch.cuda.is_initialized()
