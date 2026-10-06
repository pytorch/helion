from __future__ import annotations

import ast
from typing import cast

import numpy as np
import pytest

from test.test_cute_staged_matmul_reductions import _RENAMES
from test.test_cute_staged_matmul_reductions import _body
from test.test_cute_staged_matmul_reductions import _execute
from test.test_cute_staged_matmul_reductions import _loop
from test.test_cute_staged_matmul_reductions import _lower
from test.test_cute_staged_matmul_reductions import _marker
from test.test_cute_staged_matmul_reductions import _source

import helion
from helion._compiler import tile_strategy as lanes
from helion._compiler.cute import matmul_fallback
from helion._compiler.generate_ast import GenerateAST


def _bare_body() -> str:
    return (
        _body()
        .replace("alpha = cute.math.exp2(old_high - high_value)\n", "")
        .replace("scaled_mass = old_mass * alpha\n", "")
        .replace(
            "next_mass = scaled_mass + weight_sum", "next_mass = old_mass + weight_sum"
        )
        .replace("scaled_total = old_total * alpha\n", "")
        .replace(
            "next_total = scaled_total + product_sum",
            "next_total = old_total + product_sum",
        )
    )


def _classify(prefix: str, product: str, owner: str = "lane") -> bool:
    # Only the compiler's current statement buffer is needed by this source
    # predicate; no tracing, environment, binding or device object is created.
    cg = object.__new__(GenerateAST)
    cg.statements_stack = [cast("list[ast.AST]", ast.parse(prefix).body)]
    before = ast.dump(ast.parse(prefix), include_attributes=False)
    result = matmul_fallback._cute_product_uses_owned_lane_reduction(
        cg, ast.parse(product, mode="eval").body, owner
    )
    assert (
        ast.dump(
            ast.Module(body=cg.statements_stack[-1], type_ignores=[]),
            include_attributes=False,
        )
        == before
    )
    return result


@pytest.mark.parametrize("direction", ["left", "right"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_owned_reduction_dependency_is_proved_in_both_operands(
    direction: str, wrapped: bool
) -> None:
    prefix = _bare_body().split("product =", 1)[0]
    weight = "cutlass.Float16(weight)" if wrapped else "weight"
    product = f"{weight} * value" if direction == "left" else f"value * {weight}"
    assert _classify(prefix, product)


@pytest.mark.parametrize(
    "fault",
    [
        "no_marker",
        "unowned",
        "wrong_owner",
        "duplicate",
        "forward",
        "self_add",
        "unknown_call",
        "conditional",
        "store",
        "unrelated_product",
    ],
)
def test_unproved_definition_or_ownership_keeps_old_route(fault: str) -> None:
    prefix = _bare_body().split("product =", 1)[0]
    product = "weight * value"
    if fault == "no_marker":
        prefix = prefix.replace(_marker("score", "max"), "score")
    elif fault == "unowned":
        prefix = prefix.replace(
            _marker("score", "max"),
            lanes._lane_reduce_marker_expr(
                "score", "max", "cutlass.Float32(float('-inf'))", 1
            ),
        )
    elif fault == "wrong_owner":
        prefix = prefix.replace("'lane'", "'other_lane'")
    elif fault == "duplicate":
        prefix += "weight = 1\n"
    elif fault == "forward":
        prefix = prefix.replace(
            "weight = cute.math.exp2(score - high_value)",
            "weight = later_weight\nlater_weight = cute.math.exp2(score - high_value)",
        )
    elif fault == "self_add":
        prefix += "extra = extra + value\n"
    elif fault == "unknown_call":
        prefix = prefix.replace(
            "cute.math.exp2(score - high_value)", "unknown_call(score - high_value)"
        )
    elif fault == "conditional":
        prefix = prefix.replace(
            "weight = cute.math.exp2(score - high_value)",
            "if flag:\n    weight = cute.math.exp2(score - high_value)",
        )
    elif fault == "store":
        prefix += "values[index] = weight\n"
    else:
        product = "score_input * value"
    assert not _classify(prefix, product)


def test_plain_product_without_owned_reductions_is_unchanged() -> None:
    assert not _classify("left = values[lane]\nright = other[lane]\n", "left * right")


@pytest.mark.parametrize("extent", [2, 5, 16, 128])
@pytest.mark.parametrize("initial_high", [np.float32(-np.inf), np.float32(1)])
def test_bare_carry_uses_complete_FP32_products_once_per_tile(
    extent: int, initial_high: np.float32
) -> None:
    outer = ast.For(
        target=ast.Name(id="offset", ctx=ast.Store()),
        iter=ast.parse(f"range(0, {3 * extent}, {extent})", mode="eval").body,
        body=[_loop(extent, _bare_body())],
        orelse=[],
    )
    ast.fix_missing_locations(outer)
    lowered = lanes.split_lane_loop_reductions([outer], rename_groups=_RENAMES)
    scores = np.resize(np.array([-2, -1, 0, 1, 2], np.float32), 3 * extent)
    scores[extent : 2 * extent] += 4
    scores[2 * extent :] += 8
    values = np.resize(np.array([13, -11, 7, -3, 0.25], np.float16), 3 * extent)
    high, mass, total = initial_high, np.float32(2), np.float32(17)
    wrongly_rescaled_total = total
    for offset in range(0, scores.size, extent):
        next_high = np.maximum(high, scores[offset : offset + extent].max())
        alpha = np.float32(np.exp2(high - next_high))
        weight_sum = np.float32(0)
        product_sum = np.float32(0)
        for lane in range(extent):
            weight = np.float32(np.exp2(scores[offset + lane] - next_high))
            weight_sum = np.float32(weight_sum + weight)
            product = np.float32(np.float16(weight)) * np.float32(values[offset + lane])
            product_sum = np.float32(product_sum + product)
        total = np.float32(total + product_sum)
        mass = np.float32(mass + weight_sum)
        wrongly_rescaled_total = np.float32(
            wrongly_rescaled_total * alpha + product_sum
        )
        high = next_high
    actual, calls = _execute(
        lowered,
        scores=scores,
        values=values,
        high=initial_high,
        mass=np.float32(2),
        total=np.float32(17),
    )
    for name, expected in (("high", high), ("mass", mass), ("total", total)):
        assert actual[name].tobytes() == expected.tobytes()
    assert actual["total"].tobytes() != wrongly_rescaled_total.tobytes()
    assert calls == 3 * extent
    code = _source(lowered)
    assert "_helion_lane_reduce" not in code and "alpha" not in code
    assert code.count("next_total = old_total + product_sum") == 1
    assert code.count("next_mass = old_mass + weight_sum") == 1


def test_raw_running_sum_with_marker_dependency_still_rejects() -> None:
    body = (
        _bare_body()
        .replace(
            f"product_sum = {_marker('product', 'sum', product=True)}",
            "dot_acc = dot_acc + product",
        )
        .replace(
            "next_total = old_total + product_sum", "next_total = old_total + dot_acc"
        )
    )
    with pytest.raises(helion.exc.BackendUnsupported, match="unproved body effect"):
        lanes.split_lane_loop_reductions(
            [_loop(source=body)], rename_groups=_RENAMES, running_sums={"dot_acc"}
        )


@pytest.mark.parametrize(
    "replacement",
    [
        "next_total = old_total + product_sum + product",
        "next_total = old_total + product_sum\nnext_total = next_total + 1",
        "if flag:\n    next_total = old_total + product_sum",
        "next_total = old_total + product_sum\nvalues[index] = next_total",
    ],
)
def test_bare_carry_staging_preserves_strict_effect_and_carry_checks(
    replacement: str,
) -> None:
    source = _bare_body().replace("next_total = old_total + product_sum", replacement)
    with pytest.raises(helion.exc.BackendUnsupported):
        _lower(_loop(source=source))
