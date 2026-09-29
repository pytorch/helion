from __future__ import annotations

import ast
import operator
import struct
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from helion._compiler.cute.scalar_recipe_rounding import preserve_fp32_multiply_rounding


def _rewrite(source: str) -> str:
    tree = ast.parse(source)
    statements, result = preserve_fp32_multiply_rounding(
        cast("list[ast.Assign]", tree.body), ast.Name(id="result", ctx=ast.Load())
    )
    module = ast.Module(body=[*statements], type_ignores=[])
    # The original recipe is still available to other consumers/proofs.
    assert ast.unparse(tree) == ast.unparse(ast.parse(source))
    assert isinstance(result, ast.Name) and result.id == "result"
    return ast.unparse(ast.fix_missing_locations(module))


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def test_separate_products_keep_cancellation_before_bf16_rounding() -> None:
    source = _rewrite(
        "left = cutlass.Float32(a)\n"
        "right = cutlass.Float32(b)\n"
        "scale = 1.44269504\n"
        "left_scaled = left * scale\n"
        "right_scaled = operator.mul(right, scale)\n"
        "difference = left_scaled - right_scaled\n"
        "decay = cute.math.exp2(difference)\n"
        "weighted = decay * -1.25\n"
        "result = cutlass.BFloat16(weighted * 0.953125)\n"
    )
    assert source.count("mul.rn.f32") == 4

    def multiply(args, *, asm, constraints, dtype, is_pure):
        assert asm == "mul.rn.f32 $0, $1, $2;"
        assert constraints == "=f,f,f"
        assert dtype is _f32 and is_pure
        return _f32(args[0] * args[1])

    namespace = {
        "a": -12.125,
        "b": -12.125,
        "cutlass": SimpleNamespace(
            Float32=_f32,
            BFloat16=lambda value: float(torch.tensor(value, dtype=torch.bfloat16)),
        ),
        "cute": SimpleNamespace(math=SimpleNamespace(exp2=lambda value: 2.0**value)),
        "operator": operator,
        "_cute_inline_asm_elementwise": multiply,
    }
    exec(compile(source, "<rounding-test>", "exec"), namespace)
    assert namespace["difference"] == 0.0
    assert namespace["result"] == -1.1875
    # A newly contracted multiply/subtract changes a halfway BF16 rounding.
    scale = _f32(1.44269504)
    fused_difference = _f32(-12.125 * scale - _f32(-12.125 * scale))
    assert fused_difference != 0.0
    assert (
        float(
            torch.tensor(
                _f32(_f32(2.0**fused_difference * -1.25) * 0.953125),
                dtype=torch.bfloat16,
            )
        )
        == -1.1953125
    )


@pytest.mark.parametrize(
    "source",
    [
        "x = cutlass.Int32(a)\nresult = x * 1.44269504",
        "x = cutlass.Float16(a)\nresult = x * 1.44269504",
        "x = cutlass.BFloat16(a)\nresult = x * 1.44269504",
        "x = cutlass.Float64(a)\nresult = x * 1.44269504",
        "x = cutlass.Float32(a)\ny = cutlass.Float64(b)\nresult = x * y",
        "x = cutlass.Float32(a)\nresult = x * unknown",
        "x = cutlass.Float32(a)\nresult = x * 1099511627777",
        "x = cutlass.Float32(a)\nx = unknown\nresult = x * 1.44269504",
        "x = 1.44269504\nresult = x * 2.0",
        "x = cutlass.Float32(a)\nresult = cute.math.fma(x, 1.44269504, x)",
    ],
)
def test_unproven_other_dtypes_and_explicit_fma_stay_unchanged(source: str) -> None:
    assert _rewrite(source) == ast.unparse(ast.parse(source))


def test_explicit_fma_survives_with_a_separately_rounded_product_input() -> None:
    source = _rewrite(
        "x = cutlass.Float32(a)\nresult = cute.math.fma(x * 1.44269504, x, x)"
    )
    assert source.count("mul.rn.f32") == 1
    assert "cute.math.fma(" in source
