"""Dominance summaries keep large generated loops cheap to analyze."""

from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING

import pytest

from helion._compiler.cute.hoist_loop_invariant_recip import _collect_assigns_in_body

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("nested", [False, True])
def test_large_body_read_analysis_has_linear_walk_work(
    monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    original_walk = ast.walk
    visited = 0

    def counted_walk(node: ast.AST) -> Iterator[ast.AST]:
        nonlocal visited
        for child in original_walk(node):
            visited += 1
            yield child

    monkeypatch.setattr(ast, "walk", counted_walk)
    work = []
    for count in (128, 256):
        source = "\n".join(f"v{i} = external + {i}" for i in range(count))
        source += "\nconsume(" + ", ".join(f"v{i}" for i in range(count)) + ")"
        if nested:
            source = "for step in steps:\n" + textwrap.indent(source, "    ")
        body = ast.parse(source).body
        visited = 0
        assignments, _ = _collect_assigns_in_body(body)
        assert set(assignments) == {f"v{i}" for i in range(count)}
        work.append(visited)
    # Doubling the input doubles linear work; the old per-definition scan
    # quadruples it. This is independent of timing and permits more linear
    # passes without loosening the asymptotic complexity requirement.
    assert work[0] > 0
    assert work[1] < 3 * work[0]


@pytest.mark.parametrize(
    "source,accepted",
    [
        ("a = external\nconsume(a)", True),
        ("consume(a)\na = external", False),
        ("a = a + external\nconsume(a)", False),
        ("for i in steps:\n    a = external\n    consume(a)", True),
        ("for i in steps:\n    a = external\nconsume(a)", False),
        ("for i in steps:\n    a = external", True),
    ],
)
def test_reads_must_follow_assignment_in_its_own_body(
    source: str, accepted: bool
) -> None:
    assignments, _ = _collect_assigns_in_body(ast.parse(source).body)
    assert ("a" in assignments) == accepted
