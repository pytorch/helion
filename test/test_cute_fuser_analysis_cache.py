"""Epoch-local fusion analysis retains prefix and alias semantics."""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from helion._compiler.ast_read_writes import ReadWrites
from helion._compiler.cute.fuse_two_pass_loads import _AnalysisCache
from helion._compiler.cute.fuse_two_pass_loads import _CuteFuseTwoPassLoads


def _definitions(values: dict[str, ast.expr]) -> dict[str, str]:
    return {name: ast.dump(value) for name, value in values.items()}


def _prefix_statements() -> list[ast.stmt]:
    return ast.parse(
        """
a = source + 1
b = a * 2
if predicate:
    a = 3
    destination[0] = b
c = pointer.load()
d = _cute_atomic_add(pointer, 1)
e = b + 7
for lane in range(4):
    b = lane
    holder.field = e
f = e
"""
    ).body


@pytest.mark.parametrize("capacity", [2, 512])
def test_prefix_cache_preserves_inherited_and_nonmonotone_queries(capacity: int):
    statements = _prefix_statements()
    cache = _AnalysisCache()
    cache._CAPACITY = capacity
    inherited = {
        name: ast.Name(id=f"outer_{name}")
        for name in ("a", "b", "c", "d", "destination", "holder", "untouched")
    }
    for stop in [0, len(statements), 2, 5, 1, 7, 3, len(statements)]:
        for entry in ({}, inherited, {"a": ast.Constant(17)}):
            expected = _CuteFuseTwoPassLoads._pure_definitions_before(
                statements, stop, entry
            )
            actual = cache.pure_definitions_before(statements, stop, entry)
            assert _definitions(actual) == _definitions(expected)
            actual.clear()
    assert max(map(len, (cache._effects, cache._scopes, cache._prefixes))) <= capacity
    final = cache.pure_definitions_before(statements, len(statements), inherited)
    assert "untouched" in final and "f" in final
    assert not {"a", "b", "c", "d", "destination"} & final.keys()
    assert final["holder"] is inherited["holder"]


def test_statement_effects_are_analyzed_once_per_epoch():
    statements = _prefix_statements()
    cache = _AnalysisCache()
    with patch.object(ReadWrites, "from_ast", wraps=ReadWrites.from_ast) as effects:
        for stop in range(len(statements) + 1):
            cache.pure_definitions_before(statements, stop, {})
        for stop in reversed(range(len(statements) + 1)):
            cache.pure_definitions_before(statements, stop, {"outer": ast.Name("v")})
    assert effects.call_count == len(statements)


def test_epoch_clear_observes_nested_write_and_pure_value_mutation():
    statements = ast.parse("a = 1\nfor lane in range(4):\n    temporary = lane").body
    cache = _AnalysisCache()
    assert "a" in cache.pure_definitions_before(statements, 2, {})
    loop = statements[1]
    assert isinstance(loop, ast.For)
    loop.body.append(ast.parse("a = pointer.load()").body[0])
    cache.clear()
    assert "a" not in cache.pure_definitions_before(statements, 2, {})
    assignment = statements[0]
    assert isinstance(assignment, ast.Assign)
    assignment.value = ast.parse("pointer.load()", mode="eval").body
    cache.clear()
    assert not cache.pure_definitions_before(statements, 1, {})


@pytest.mark.parametrize("capacity", [1, 512])
def test_loop_alias_summary_retains_first_breadth_first_match(capacity: int):
    first, second = ast.parse(
        """
for offset in range(8):
    reduction_lane_base_1 = offset
    lane_base_1 = offset
    if predicate:
        reduction_lane_base_deeper = offset
        reduction_vec_lane_1 = offset
    reduction_lane_base_later = offset
    for vec_lane_1 in range(4):
        pass
for offset in range(8):
    reduction_lane_base_2 = offset
    lane_base_2 = offset
    reduction_vec_lane_2 = offset
    for vec_lane_2 in range(4):
        pass
"""
    ).body
    assert isinstance(first, ast.For) and isinstance(second, ast.For)
    cache = _AnalysisCache()
    cache._CAPACITY = capacity
    fuser = _CuteFuseTwoPassLoads()
    expected = fuser._build_second_alias(first, second)
    for _ in range(3):
        assert fuser._build_second_alias(first, second, analysis=cache) == expected
    assert expected["reduction_lane_base_2"] == "reduction_lane_base_1"
    assert len(cache._aliases) <= capacity
    target = second.body[0]
    assert isinstance(target, ast.Assign)
    target.targets[0] = ast.Name("reduction_lane_base_fresh", ctx=ast.Store())
    cache.clear()
    assert fuser._build_second_alias(first, second, analysis=cache) == (
        fuser._build_second_alias(first, second)
    )
