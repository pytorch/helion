from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.fuse_two_pass_loads import _LexicalMemoryOrder


@pytest.mark.parametrize(
    "store",
    [
        "(x.iterator + i).store(value)",
        "cute.arch.store(x.iterator + i, value)",
        "_cute_store_scalar(x.iterator + i, value)",
        "_cute_atomic_add(x.iterator + i, value)",
        "cute.arch.atomic_add(x.iterator + i, value)",
        "x.__setitem__(i, value)",
        "_helion_persistent_branch_vec_store(0, 1, 2, x.iterator, 4, 5)",
    ],
)
@pytest.mark.parametrize("disjoint", [False, True])
def test_repeated_queries_keep_destination_aliases(store: str, disjoint: bool) -> None:
    body = ast.parse(f"for i in range(8):\n    if enabled:\n        {store}").body
    pairs = {frozenset(("x", "y"))} if disjoint else set()
    order = _LexicalMemoryOrder(body, {"x", "y"}, pairs)
    for roots, expected in (
        (frozenset(("y",)), not disjoint),
        (frozenset(("x",)), True),
        (None, True),
        (frozenset(("x", "y")), True),
        (frozenset(("y",)), not disjoint),
    ):
        assert order.has_aliasing_store_in(body[0], roots) is expected


@pytest.mark.parametrize(
    "store",
    [
        "unknown.store(value)",
        "_cute_store_scalar()",
        "cute.arch.atomic_add(unknown, value)",
        "(x.iterator + other.iterator).store(value)",
    ],
)
def test_unknown_destination_remains_conservative(store: str) -> None:
    body = ast.parse(store).body
    order = _LexicalMemoryOrder(body, {"x", "y"}, {frozenset(("x", "y"))})
    for roots in (frozenset(("y",)), frozenset(("x",)), None):
        assert order.has_aliasing_store_in(body[0], roots)


def test_empty_and_nested_domains_remain_separate() -> None:
    root = ast.parse(
        """
for i in range(8):
    value = (x.iterator + i).load()
    if enabled:
        (y.iterator + i).store(value)
else:
    (z.iterator).store(value)
"""
    )
    loop = root.body[0]
    assert isinstance(loop, ast.For)
    pairs = {frozenset(pair) for pair in (("x", "y"), ("x", "z"), ("y", "z"))}
    order = _LexicalMemoryOrder(root.body, {"x", "y", "z"}, pairs)
    for _ in range(3):
        assert not order.has_aliasing_store_in(loop.body[0], None)
        assert not order.has_aliasing_store_in(root, frozenset(("x",)))
        assert order.has_aliasing_store_in(root, frozenset(("y",)))
        assert order.has_aliasing_store_in(loop, frozenset(("z",)))
        assert not order.has_aliasing_store_in(loop.body[1], frozenset(("z",)))


def test_rebuilt_group_observes_store_added_after_previous_analysis() -> None:
    body = ast.parse("for i in range(8):\n    value = (x.iterator + i).load()").body
    loop = body[0]
    assert isinstance(loop, ast.For)
    old_order = _LexicalMemoryOrder(body, {"x"}, set())
    assert not old_order.has_aliasing_store_in(loop, frozenset(("x",)))
    loop.body.append(ast.parse("(x.iterator + i).store(value)").body[0])
    new_order = _LexicalMemoryOrder(body, {"x"}, set())
    assert new_order.has_aliasing_store_in(loop, frozenset(("x",)))


def test_store_summary_keeps_multiple_known_roots() -> None:
    body = ast.parse("(x.iterator + y.iterator).store(value)").body
    pairs = {frozenset(pair) for pair in (("x", "z"), ("y", "z"))}
    order = _LexicalMemoryOrder(body, {"x", "y", "z"}, pairs)
    for name in ("x", "y", "z", "x"):
        assert order.has_aliasing_store_in(body[0], frozenset((name,))) is (name != "z")
