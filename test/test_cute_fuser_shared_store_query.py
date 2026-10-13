from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest

from helion._compiler.cute import fuse_two_pass_loads as fuser

if TYPE_CHECKING:
    from collections.abc import Iterator


def _shared_expression(leaf: ast.expr, depth: int) -> ast.expr:
    for _ in range(depth):
        leaf = ast.BinOp(left=leaf, op=ast.Add(), right=leaf)
    return leaf


@pytest.mark.parametrize(
    "store", [None, "x.iterator.store(value)", "unknown.store(value)"]
)
@pytest.mark.parametrize("root", [None, "x", "y"])
def test_deep_shared_store_presence_has_bounded_work(
    monkeypatch: pytest.MonkeyPatch, store: str | None, root: str | None
) -> None:
    leaf = ast.parse(store or "value", mode="eval").body
    expression = _shared_expression(leaf, 60)
    visited: list[ast.AST] = []
    original = ast.iter_child_nodes

    def observe(node: ast.AST) -> Iterator[ast.AST]:
        visited.append(node)
        assert len(visited) < 150, "presence query expanded shared syntax"
        return original(node)

    # Isolate the occurrence-independent query. Lexical execution-position
    # analysis intentionally remains occurrence-sensitive and is tested below.
    order = fuser._LexicalMemoryOrder([], {"x", "y"}, {frozenset(("x", "y"))})
    with monkeypatch.context() as patch:
        patch.setattr(ast, "iter_child_nodes", observe)
        roots = None if root is None else frozenset((root,))
        expected = store is not None and ("unknown" in store or root != "y")
        assert order.has_aliasing_store_in(expression, roots) is expected
        assert len(order._stores_in[expression]) == int(store is not None)
        count = len(visited)
        assert order.has_aliasing_store_in(expression, roots) is expected
        assert len(visited) == count  # Repeated queries reuse the existing snapshot.


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
        "unknown.store(value)",
        "_cute_store_scalar()",
        "(x.iterator + y.iterator).store(value)",
    ],
)
@pytest.mark.parametrize("depth", [0, 4])
def test_shared_and_tree_queries_match_occurrence_expansion(
    store: str, depth: int
) -> None:
    expression = _shared_expression(ast.parse(store, mode="eval").body, depth)
    root = ast.Module(
        body=[ast.Expr(value=expression), *ast.parse("z.iterator.store(value)").body],
        type_ignores=[],
    )
    pairs = {frozenset(pair) for pair in (("x", "y"), ("x", "z"), ("y", "z"))}
    order = fuser._LexicalMemoryOrder([], {"x", "y", "z"}, pairs)
    # The old query enumerates every occurrence. Its answer must remain exact
    # even when a shared call is reached through many different parents.
    stores = [
        fuser._store_tensor_roots(node, {"x", "y", "z"})
        for node in ast.walk(root)
        if isinstance(node, ast.Call) and fuser._is_store_call(node)
    ]
    for roots in (None, frozenset(("x",)), frozenset(("y",)), frozenset(("z",))):
        expected = any(fuser._tensor_roots_may_alias(s, roots, pairs) for s in stores)
        assert order.has_aliasing_store_in(root, roots) is expected


@pytest.mark.parametrize("middle", [False, True])
@pytest.mark.parametrize("disjoint", [False, True])
def test_repeated_store_objects_keep_distinct_execution_positions(
    middle: bool, disjoint: bool
) -> None:
    producer, consumer = ast.parse("first = x.iterator.load()\nlast = first").body
    assert isinstance(producer, ast.Assign) and isinstance(consumer, ast.Assign)
    store = ast.parse("y.iterator.store(value)").body[0]
    body = [store, producer, *([store] if middle else []), consumer, store]
    pairs = {frozenset(("x", "y"))} if disjoint else set()
    order = fuser._LexicalMemoryOrder(body, {"x", "y"}, pairs)
    assert len(order.stores) == (3 if middle else 2)
    positions = [position for position, _ in order.stores]
    assert len(set(positions)) == len(positions)
    assert order.has_aliasing_store_between(producer, consumer, frozenset(("x",))) is (
        middle and not disjoint
    )
    assert order.has_aliasing_store_between(consumer, producer, frozenset(("x",)))


def test_query_snapshot_is_retired_after_syntax_changes() -> None:
    leaf = ast.parse("value", mode="eval").body
    expression = _shared_expression(leaf, 5)
    old = fuser._LexicalMemoryOrder([], {"x"}, set())
    assert not old.has_aliasing_store_in(expression, None)
    assert isinstance(expression, ast.BinOp)
    expression.right = ast.parse("x.iterator.store(value)", mode="eval").body
    current = fuser._LexicalMemoryOrder([], {"x"}, set())
    assert current.has_aliasing_store_in(expression, frozenset(("x",)))


@pytest.mark.parametrize("form", ["member", "arch", "atomic"])
@pytest.mark.parametrize("unknown", [False, True])
def test_shared_store_pointer_queries_also_have_bounded_work(
    monkeypatch: pytest.MonkeyPatch, form: str, unknown: bool
) -> None:
    pointer = _shared_expression(
        ast.parse(
            "x.iterator + other.iterator" if unknown else "x.iterator", mode="eval"
        ).body,
        60,
    )
    value = ast.Name(id="value", ctx=ast.Load())
    if form == "member":
        store = ast.Call(
            func=ast.Attribute(value=pointer, attr="store", ctx=ast.Load()),
            args=[value],
            keywords=[],
        )
    else:
        func = ast.parse(
            "cute.arch.store" if form == "arch" else "_cute_atomic_add", mode="eval"
        ).body
        store = ast.Call(func=func, args=[pointer, value], keywords=[])
    original = ast.iter_child_nodes
    visits = 0

    def observe(node: ast.AST) -> Iterator[ast.AST]:
        nonlocal visits
        visits += 1
        assert visits < 500, "alias-root query expanded shared syntax"
        return original(node)

    order = fuser._LexicalMemoryOrder([], {"x", "y"}, {frozenset(("x", "y"))})
    with monkeypatch.context() as patch:
        patch.setattr(ast, "iter_child_nodes", observe)
        assert order.has_aliasing_store_in(store, frozenset(("y",))) is unknown
        assert order.has_aliasing_store_in(store, frozenset(("x",)))
        assert order.has_aliasing_store_in(store, None)
