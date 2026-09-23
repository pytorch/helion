"""Typed address proofs and bounded analysis for vectorized operand loads."""

from __future__ import annotations

import ast
from typing import cast
from unittest.mock import patch

import pytest

from helion._compiler.cute.contiguous_copy import CopyTensorFacts
from helion._compiler.cute.contiguous_copy import _Analysis
from helion._compiler.cute.contiguous_copy import plan_contiguous_copy


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _load_plan(source: str, dtype: str, width: int):
    return plan_contiguous_copy(
        cast("list[ast.Assign]", ast.parse(source).body),
        _expr("loaded"),
        coordinate="k",
        tensors={"A": CopyTensorFacts(dtype, (24, 1), 16)},
        aligned_names={"k": width},
    )


@pytest.mark.parametrize(
    "dtype,width",
    [("cutlass.Float16", 8), ("cutlass.BFloat16", 8), ("cutlass.Float32", 4)],
)
@pytest.mark.parametrize(
    "offset",
    [
        "cutlass.Float32(0.125) * 8",
        "0.125 * 8",
        "(scales.iterator + 2).load() * 8",
        "cutlass.Float32(-0.125) * 8",
    ],
)
@pytest.mark.parametrize("predicate", ["True", "k < 11", "False"])
def test_precast_float_arithmetic_declines_contiguous_copy(
    dtype: str, width: int, offset: str, predicate: str
) -> None:
    source = f"""
extra = cutlass.Int32({offset})
loaded = (A.iterator + k + extra).load() if {predicate} else {dtype}(0.0)
value = {dtype}(loaded * 0.5)
"""
    assert _load_plan(source, dtype, width) is None


@pytest.mark.parametrize(
    "offset",
    [
        "cutlass.Int32(cutlass.Float32(0.125)) * 8",
        "cutlass.Int32((indices.iterator + m).load()) * 8",
    ],
)
def test_integer_arithmetic_after_cast_retains_contiguous_copy_plan(
    offset: str,
) -> None:
    source = f"""
extra = {offset}
loaded = (A.iterator + k + extra).load()
value = cutlass.Float16(loaded * 0.5)
"""
    assert _load_plan(source, "cutlass.Float16", 8) is not None


def _doubling_recipe(depth: int) -> str:
    statements = ["q0 = (k + cutlass.Int32(m) * 8) - k"]
    statements.extend(f"q{i} = q{i - 1} + q{i - 1}" for i in range(1, depth + 1))
    statements.extend(
        [
            f"loaded = (A.iterator + k + q{depth}).load()",
            "value = cutlass.Float16(loaded * 0.5)",
        ]
    )
    return "\n".join(statements)


def test_small_affine_dag_retains_contiguous_copy_plan() -> None:
    source = _doubling_recipe(4)
    assert _load_plan(source, "cutlass.Float16", 8) is not None


@pytest.mark.parametrize("depth", [8, 12, 40, 300])
def test_large_affine_dag_declines_without_expansion(depth: int) -> None:
    source = _doubling_recipe(depth)
    assert _load_plan(source, "cutlass.Float16", 8) is None


def _predicate_recipe(depth: int) -> str:
    statements = ["p0 = k < 16"]
    statements.extend(f"p{i} = p{i - 1} and p{i - 1}" for i in range(1, depth + 1))
    statements.extend(
        [
            f"loaded = (A.iterator + k).load() if p{depth} else cutlass.Float16(0)",
            "value = cutlass.Float16(loaded * 0.5)",
        ]
    )
    return "\n".join(statements)


def test_small_predicate_dag_retains_masked_copy_plan() -> None:
    source = _predicate_recipe(2)
    assert _load_plan(source, "cutlass.Float16", 8) is not None


@pytest.mark.parametrize("depth", [12, 24])
def test_large_predicate_dag_has_bounded_proof_work(depth: int) -> None:
    source = _predicate_recipe(depth)
    interval = _Analysis.interval
    calls = 0
    limit = 128 * len(ast.parse(source).body)

    def limited_interval(self: _Analysis, value: ast.expr) -> bool:
        nonlocal calls
        calls += 1
        # A deterministic work cap catches exponential traversal before an
        # adversarial test can itself hang. It does not require a specific
        # memoization layout or where a conservative rejection happens.
        assert calls <= limit
        return interval(self, value)

    with patch.object(_Analysis, "interval", limited_interval):
        assert _load_plan(source, "cutlass.Float16", 8) is None
