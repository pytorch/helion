from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.hoist_loop_invariant_recip import hoist_loop_invariant_recips


@pytest.mark.parametrize(
    "updates",
    [
        "a = a_copy + b_copy\nb = b_copy + 3",
        "a = b_copy\nb = a_copy + 3",
        "a += 10\nb = a_copy + b_copy",
        "if enabled:\n    a = 10\nb = a_copy + b_copy",
        "for i in range(2):\n    a += i\nb = a_copy + b_copy",
        "a, b = 10, 20\nb = a_copy + b_copy",
        "while a_copy < 3:\n    a += 1\n    if a == 5:\n        break",
    ],
)
def test_parallel_snapshots_preserve_evaluation_order(updates: str) -> None:
    source = (
        "a = 2\nb = 5\nenabled = True\n"
        "a_incoming = a\nb_incoming = b\n"
        "a_copy = a_incoming\nb_copy = b_incoming\n" + updates + "\nresult = (a, b)\n"
    )
    expected = {}
    exec(source, expected)
    tree = ast.parse(source)
    tree.body = hoist_loop_invariant_recips(
        tree.body,
        snapshot_names={"a_incoming", "b_incoming", "a_copy", "b_copy"},
    )
    actual = {}
    exec(compile(ast.fix_missing_locations(tree), "<snapshots>", "exec"), actual)
    assert actual["result"] == expected["result"]


def test_snapshot_chain_keeps_early_value_across_root_write() -> None:
    tree = ast.parse(
        "value = 2\nfirst = value\nvalue = 7\nsecond = first\nresult = second\n"
    )
    tree.body = hoist_loop_invariant_recips(
        tree.body, snapshot_names={"first", "second"}
    )
    actual = {}
    exec(compile(ast.fix_missing_locations(tree), "<snapshots>", "exec"), actual)
    assert actual["result"] == 2


def test_final_read_in_root_update_eliminates_snapshot() -> None:
    tree = ast.parse("value = 2\nsnapshot = value\nvalue = snapshot + 1\n")
    tree.body = hoist_loop_invariant_recips(tree.body, snapshot_names={"snapshot"})
    assert "snapshot" not in ast.unparse(tree)
    actual = {}
    exec(compile(ast.fix_missing_locations(tree), "<snapshots>", "exec"), actual)
    assert actual["value"] == 3


def test_nested_loop_target_is_not_invariant() -> None:
    # The enclosing loop must not hoist ``8 - inner`` out of the inner loop.
    source = (
        "result = []\n"
        "for outer in range(2):\n"
        "    for inner in range(4):\n"
        "        result.append((8 - inner) * 2.0)\n"
    )
    expected = {}
    exec(source, expected)
    tree = ast.parse(source)
    tree.body = hoist_loop_invariant_recips(tree.body)
    actual = {}
    exec(compile(ast.fix_missing_locations(tree), "<nested-loops>", "exec"), actual)
    assert actual["result"] == expected["result"]


@pytest.mark.parametrize("expression", ["(8 - outer) * 2.0", "8.0 / outer"])
def test_tuple_loop_targets_are_not_invariant(expression: str) -> None:
    source = (
        "result = []\n"
        "for outer, other in [(1, 2), (2, 3)]:\n"
        f"    result.append({expression})\n"
    )
    expected = {}
    exec(source, expected)
    tree = ast.parse(source)
    tree.body = hoist_loop_invariant_recips(tree.body)
    actual = {}
    exec(compile(ast.fix_missing_locations(tree), "<tuple-loop>", "exec"), actual)
    assert actual["result"] == expected["result"]
