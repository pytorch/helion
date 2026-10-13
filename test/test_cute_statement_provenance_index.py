from __future__ import annotations

import ast
from collections import Counter
from unittest.mock import patch

import pytest

from helion._compiler import tile_strategy as lanes
from helion._compiler.ast_read_writes import ReadWrites


def _original(
    body: list[ast.AST],
    thread_axes: dict[str, frozenset[int]],
    definitions: dict[str, ast.AST],
) -> tuple[dict[int, dict[str, frozenset[int]]], dict[int, dict[str, ast.AST]]]:
    before_axes: dict[int, dict[str, frozenset[int]]] = {}
    before_defs: dict[int, dict[str, ast.AST]] = {}
    local_axes = dict(thread_axes)
    local_defs = dict(definitions)
    for stmt in body:
        before_axes[id(stmt)] = dict(local_axes)
        before_defs[id(stmt)] = dict(local_defs)
        lanes._update_thread_axis_names(stmt, local_axes)
        lanes._update_scalar_definitions(stmt, local_defs)
    return before_axes, before_defs


def _compare(
    body: list[ast.AST], definitions: dict[str, ast.AST]
) -> dict[int, dict[str, ast.AST]]:
    axes = {"row": frozenset({1}), "key": frozenset({0})}
    initial = list(definitions.items())
    expected_axes, expected_defs = _original(body, axes, definitions)
    actual_axes, actual_defs = lanes._statement_provenance_before(
        body, axes, definitions
    )
    assert actual_axes == expected_axes
    assert list(actual_defs) == list(expected_defs)
    for stmt in body:
        # AST identity and insertion order are part of the original snapshots.
        assert list(actual_defs[id(stmt)].items()) == list(
            expected_defs[id(stmt)].items()
        )
    assert list(definitions.items()) == initial
    assert axes == {"row": frozenset({1}), "key": frozenset({0})}
    return actual_defs


def _expr(text: str) -> ast.AST:
    return ast.parse(text, mode="eval").body


def test_transitive_cycles_self_references_and_branch_rebinding() -> None:
    definitions: dict[str, ast.AST] = {
        "tail": _expr("middle + 1"),
        "middle": _expr("external + row"),
        "self": _expr("self + 1"),
        "cycle_a": _expr("cycle_b + 1"),
        "cycle_b": _expr("cycle_a + key"),
        "view": _expr("tensor[index]"),
        "stable": _expr("constant + 1"),
    }
    body: list[ast.AST] = list(
        ast.parse(
            """
record(tail)
if condition:
    external = 9
record(tail)
self = self + 2
saved = self + stable
self += 1
record(saved)
cycle_a = 7
record(cycle_b)
tensor[index] = 3
record(view)
"""
        ).body
    )
    snapshots = _compare(body, definitions)
    assert list(snapshots[id(body[2])]) == [
        "self",
        "cycle_a",
        "cycle_b",
        "view",
        "stable",
    ]
    assert list(snapshots[id(body[6])]) == ["cycle_a", "cycle_b", "view", "stable"]
    assert list(snapshots[id(body[8])]) == ["view", "stable", "cycle_a"]
    assert list(snapshots[id(body[10])]) == ["stable", "cycle_a"]


@pytest.mark.parametrize(
    "change",
    [
        "root = root + 1",
        "root += delta",
        "root, other = pair",
        "if cond:\n    root = next_value\nelse:\n    other = 0",
        "for lane in values:\n    root = lane",
        "while cond:\n    root += 1",
        "try:\n    root = next_value\nfinally:\n    other = 0",
        "root[index] = value",
        "hl.store(root, index, value)",
        "hl.atomic_add(root, index, value)",
        "other = root + 1",
        "record(root)",
    ],
)
def test_original_effect_semantics_and_definition_order(change: str) -> None:
    definitions: dict[str, ast.AST] = {
        "dependent": _expr("indirect * 2"),
        "indirect": _expr("root + other"),
        "root": _expr("input_value"),
        "other": _expr("unrelated"),
    }
    body: list[ast.AST] = list(
        ast.parse(change + "\nrecord(dependent)\nroot = 2\nrecord(root)").body
    )
    _compare(body, definitions)


def test_dependencies_are_rebuilt_after_ast_mutation_between_calls() -> None:
    expression = _expr("old_value + 1")
    definitions: dict[str, ast.AST] = {"result": expression}
    body: list[ast.AST] = list(ast.parse("new_value = 3\nrecord(result)").body)
    assert "result" in _compare(body, definitions)[id(body[1])]
    assert isinstance(expression, ast.BinOp)
    assert isinstance(expression.left, ast.Name)
    expression.left.id = "new_value"
    assert "result" not in _compare(body, definitions)[id(body[1])]


def test_unchanged_rhs_is_analyzed_once_per_epoch() -> None:
    definitions: dict[str, ast.AST] = {
        f"entry{i}": _expr(f"external{i} + 1") for i in range(20)
    }
    body: list[ast.AST] = list(
        ast.parse("\n".join(f"local{i} = entry0 + {i}" for i in range(25))).body
    )
    rhs_ids = {id(value) for value in definitions.values()}
    rhs_ids.update(id(stmt.value) for stmt in body if isinstance(stmt, ast.Assign))
    original = ReadWrites.from_ast
    counts: Counter[int] = Counter()

    def count(node: ast.AST) -> ReadWrites:
        if id(node) in rhs_ids:
            counts[id(node)] += 1
        return original(node)

    with patch.object(ReadWrites, "from_ast", side_effect=count):
        _original(body, {}, definitions)
    assert sum(counts.values()) > 500
    counts.clear()
    with patch.object(ReadWrites, "from_ast", side_effect=count):
        lanes._statement_provenance_before(body, {}, definitions)
    assert set(counts) == rhs_ids
    assert set(counts.values()) == {1}
