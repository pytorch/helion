from __future__ import annotations

from dataclasses import replace
import gc
from unittest.mock import patch
import weakref

import pytest

from helion._compiler.cute.chained_coordinates import MAX_NODES
from helion._compiler.cute.chained_coordinates import CoordinateDomain
from helion._compiler.cute.chained_coordinates import IndexExpr
from helion._compiler.cute.chained_coordinates import _ValueCache
from helion._compiler.cute.chained_coordinates import _values
from helion._compiler.cute.chained_coordinates import binary
from helion._compiler.cute.chained_coordinates import constant
from helion._compiler.cute.chained_coordinates import original_coordinate
from helion._compiler.cute.chained_operand_coordinates import OperandCoordinates


def test_shared_subtree_values_survive_bounded_and_affine_checks() -> None:
    context = OperandCoordinates(CoordinateDomain("r", "c", (0, 31), (0, 31)))
    flat = binary(
        "add", binary("mul", IndexExpr("row"), constant(32)), IndexExpr("column")
    )
    original = context.bounded_values(flat)
    assert original == tuple(range(1024))
    assert context._values_cache.entries[id(flat)][1] is original
    result = context.normalize(binary("mod", flat, constant(32)), "flat % 32")
    assert str(result) == "flat % 32"
    assert original_coordinate(result) == "flat % 32"
    assert context._values_cache.entries[id(flat)][1] is original
    assert context.in_bounds(result, 32)
    assert not context.in_bounds(result, 31)


@pytest.mark.parametrize("kind", ["add", "sub", "mul", "div", "mod"])
def test_cached_values_match_integer_arithmetic_at_every_owned_point(kind: str) -> None:
    points = CoordinateDomain("r", "c", (1, 15), (0, 15)).points()
    assert points is not None
    left = binary(
        "add", binary("mul", IndexExpr("row"), constant(16)), IndexExpr("column")
    )
    expression = binary(kind, left, constant(3))
    cache = _ValueCache()
    assert _values(left, points, cache=cache) == tuple(r * 16 + c for r, c in points)
    expected = {
        "add": lambda x: x + 3,
        "sub": lambda x: x - 3,
        "mul": lambda x: x * 3,
        "div": lambda x: x // 3,
        "mod": lambda x: x % 3,
    }[kind]
    assert _values(expression, points, cache=cache) == tuple(
        expected(r * 16 + c) for r, c in points
    )


@pytest.mark.parametrize(
    "case",
    [
        "negative",
        "overflow",
        "cancelled_overflow",
        "zero_divisor",
        "variable_divisor",
        "unknown",
        "invalid_arity",
    ],
)
def test_warm_valid_subtree_does_not_admit_invalid_intermediates(case: str) -> None:
    points = ((0, 0), (1, 1), (31, 31))
    row = IndexExpr("row")
    cache = _ValueCache()
    assert _values(row, points, cache=cache) == (0, 1, 31)
    overflow = binary("mul", row, constant(2**30))
    expression = {
        "negative": binary("sub", row, constant(1)),
        "overflow": overflow,
        "cancelled_overflow": binary("div", overflow, constant(2**30)),
        "zero_divisor": binary("mod", row, constant(0)),
        "variable_divisor": binary("div", row, IndexExpr("column")),
        "unknown": IndexExpr("runtime_load", (row,)),
        "invalid_arity": IndexExpr("add", (row,)),
    }[case]
    assert _values(expression, points, cache=cache) is None
    assert id(expression) not in cache.entries


def test_cached_subtree_cannot_bypass_whole_expression_budget() -> None:
    expression = constant(0)
    cache = _ValueCache()
    for _ in range(MAX_NODES - 1):
        expression = binary("add", expression, expression)
    assert _values(expression, ((0, 0),), cache=cache) == (0,)
    too_large = binary("add", expression, expression)
    assert _values(too_large, ((0, 0),), cache=cache) is None


def test_point_domain_change_recomputes_before_reuse() -> None:
    expression = binary("add", IndexExpr("row"), IndexExpr("column"))
    cache = _ValueCache()
    assert _values(expression, ((0, 0), (1, 2)), cache=cache) == (0, 3)
    first = cache.entries[id(expression)][1]
    assert _values(expression, ((2, 3), (3, 4)), cache=cache) == (5, 7)
    assert cache.entries[id(expression)][1] is not first


def test_cache_retains_identity_owners_and_releases_them_at_capacity() -> None:
    cache = _ValueCache()
    expression = constant(7)
    reference = weakref.ref(expression)
    assert _values(expression, ((0, 0),), cache=cache) == (7,)
    expression = None
    gc.collect()
    assert reference() is not None
    for value in range(MAX_NODES):
        assert _values(constant(value), ((0, 0),), cache=cache) == (value,)
        assert len(cache.entries) <= MAX_NODES
    gc.collect()
    assert reference() is None


def test_evaluator_never_hashes_expression_dag_but_proof_keys_stay_structural() -> None:
    expression = binary("add", IndexExpr("row"), constant(0))
    with patch.object(
        IndexExpr, "__hash__", side_effect=AssertionError("recursive hash")
    ):
        assert _values(expression, ((1, 2),), cache=_ValueCache()) == (1,)
    context = OperandCoordinates(CoordinateDomain("row", "column", (0, 15), (0, 15)))
    result = context.normalize(expression, "row + 0")
    proof = context.proofs[expression]
    assert proof is not None
    assert context.proofs.get(replace(expression)) == proof
    assert proof.original is expression
    assert original_coordinate(result) == "row + 0"


def test_contexts_never_share_value_owners() -> None:
    first = OperandCoordinates(CoordinateDomain("r", "c", (0, 3), (0, 3)))
    second = OperandCoordinates(CoordinateDomain("r", "c", (4, 7), (4, 7)))
    expression = IndexExpr("row")
    assert first.bounded_values(expression) == (0,) * 4 + (1,) * 4 + (2,) * 4 + (3,) * 4
    assert (
        second.bounded_values(expression) == (4,) * 4 + (5,) * 4 + (6,) * 4 + (7,) * 4
    )
    assert first._values_cache is not second._values_cache
