"""The statement scans the CuTe codegen passes use to skip work."""

from __future__ import annotations

import ast

import pytest

from helion._compiler.ast_extension import iter_statements
from helion._compiler.cute.thread_budget import referenced_thread_axes
from helion._compiler.tile_strategy import has_lane_reduce_markers

pytestmark = pytest.mark.cpu_only


def _body(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


def test_referenced_thread_axes_finds_literal_subscripts_at_any_depth() -> None:
    body = _body(
        """
x = cute.arch.thread_idx()[0]
for i in range(4):
    if flag:
        y = (lane * 32 + cute.arch.thread_idx()[1]) * 4
    else:
        z = cute.arch.thread_idx()[2] + cute.arch.thread_idx()[1]
"""
    )
    assert sorted(referenced_thread_axes(body)) == [0, 1, 1, 2]


def test_referenced_thread_axes_ignores_text_and_non_literal_indices() -> None:
    body = _body(
        """
label = "cute.arch.thread_idx()[0]"
a = cute.arch.thread_idx()[axis]
b = cute.arch.thread_idx(1)[0]
c = cute.arch.thread_idx()[3]
d = other.thread_idx()[1]
"""
    )
    assert referenced_thread_axes(body) == []


def test_iter_statements_walks_nested_blocks_only() -> None:
    body = _body(
        """
a = f(lambda: g())
for i in range(2):
    b = 1
    if a:
        c = 2
    else:
        d = 3
try:
    e = 4
except ValueError:
    f_ = 5
finally:
    g_ = 6
with ctx():
    h = 7
"""
    )
    kinds = [type(stmt).__name__ for stmt in iter_statements(body)]
    assert kinds == [
        "Assign",
        "For",
        "Assign",
        "If",
        "Assign",
        "Assign",
        "Try",
        "Assign",
        "Assign",
        "Assign",
        "With",
        "Assign",
    ]


def test_has_lane_reduce_markers_needs_the_marker_name() -> None:
    assert not has_lane_reduce_markers(_body("acc = acc + x\nfor i in r: y = f(i)"))
    assert has_lane_reduce_markers(
        _body("for i in r:\n    acc = _helion_lane_reduce(acc, 'sum', i)")
    )
