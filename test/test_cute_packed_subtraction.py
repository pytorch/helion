from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.factor_affine_reductions import pack_fp32_constexpr_loops


def _source(expression: str, extra_use: bool = False) -> str:
    return "\n".join(
        (
            "result = []",
            "for element in cutlass.range_constexpr(2):",
            "    prediction = cutlass.Float32(values[element])",
            "    product = prediction * scale",
            "    alias = product",
            "    raw = cutlass.Float32(inputs[element])",
            f"    residual = {expression}",
            "    result.append(residual + alias)"
            if extra_use
            else "    result.append(residual)",
            "return result[0], result[1]",
        )
    )


def _pack(
    source: str, *, fast_math: bool = True, target: tuple[int, int] = (10, 3)
) -> str:
    result = pack_fp32_constexpr_loops(
        ast.parse(source).body,
        fast_math=fast_math,
        target_device_capability=target,
        float_scalar_names={"scale"},
    )
    return ast.unparse(ast.Module(body=result, type_ignores=[]))


@pytest.mark.parametrize(
    "expression,negative",
    (("raw - alias", "-scale"), ("alias - raw", "-_factored_raw")),
)
def test_subtractive_fma_preserves_negative_operand(
    expression: str, negative: str
) -> None:
    result = _pack(_source(expression))
    assert result.count("cute.arch.fma_packed_f32x2") == 1
    assert "cute.arch.mul_packed_f32x2" not in result
    assert "cute.arch.sub_packed_f32x2" not in result
    assert negative in result
    assert "_factored_alias" not in result
    assert "_factored_product" not in result


def test_reused_product_keeps_its_original_rounding() -> None:
    result = _pack(_source("raw - alias", extra_use=True))
    assert "cute.arch.fma_packed_f32x2" not in result
    assert result.count("cute.arch.mul_packed_f32x2") == 1
    assert result.count("cute.arch.sub_packed_f32x2") == 1


@pytest.mark.parametrize("fast_math,target", ((False, (10, 3)), (True, (9, 0))))
def test_subtractive_fma_requires_existing_fast_math_authority(
    fast_math: bool, target: tuple[int, int]
) -> None:
    source = _source("raw - alias")
    assert _pack(source, fast_math=fast_math, target=target) == ast.unparse(
        ast.parse(source)
    )
