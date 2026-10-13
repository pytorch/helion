from __future__ import annotations

import ast
import copy

import pytest

from helion._compiler.cute import proven_loop_bounds as bounds
from helion._compiler.cute.simplify_proven_bounds import _assigned_names


@pytest.mark.parametrize(
    "source",
    [
        "x = 1",
        "x, y = opaque()",
        "a[index] = opaque()",
        "x += opaque()",
        "for i in range(4):\n x = i\nelse:\n y = 2",
        "if flag:\n x = 1\nelse:\n y = 2",
        "with scope() as (x, y):\n z = opaque()",
        "import opaque as alias\nfrom opaque import name",
        "opaque(x)\npass",
        "for i in range(4):\n with scope() as x:\n  if flag:\n   y = i\n  else:\n   z = x",
    ],
)
def test_target_walk_equals_full_walk(source: str) -> None:
    body = ast.parse(source).body
    assert bounds._supported(body)
    summary = bounds._statement_assigned_names(body)
    assert summary is not None
    names, count = summary
    module = ast.Module(body=body, type_ignores=[])
    assert names == _assigned_names(module)
    assert count == sum(
        isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
        for node in ast.walk(module)
    )


def test_queries_use_current_targets_and_surviving_branches() -> None:
    loop = ast.parse("for i in range(4):\n x = 1\nelse:\n y = 2").body[0]
    proof = bounds._Proof((128, 1, 1), statement_bindings_only=True)
    assert proof.assigned_names(loop) == {"i", "x", "y"}
    assert isinstance(loop, ast.For)
    loop.orelse.clear()
    loop.target = ast.Name(id="changed", ctx=ast.Store())
    assert proof.assigned_names(loop) == {"changed", "x"}
    assignment = loop.body[0]
    assert isinstance(assignment, ast.Assign)
    assignment.targets[0] = ast.Name(id="new", ctx=ast.Store())
    assert proof.assigned_names(loop) == {"changed", "new"}


@pytest.mark.parametrize(
    "source",
    [
        "x = (y := 1)",
        "x = [i for i in range(4)]",
        "x = {i for i in range(4)}",
        "x = {i: i for i in range(4)}",
        "x = (i for i in range(4))",
        "x = lambda: 1",
        "yield 1",
        "yield from values",
        "await task()",
        "x = y = 1",
        "while flag:\n x = 1",
        "try:\n x = 1\nexcept Exception:\n y = 2",
        "def function():\n return 1",
        "class C:\n pass",
        "from opaque import *",
    ],
)
def test_unsupported_forms_are_not_rewritten(source: str) -> None:
    body = ast.parse(source).body
    before = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=True)
    output = bounds.simplify_proven_loop_bounds(
        body,
        enabled=True,
        thread_block_dims=(128, 1, 1),
        constexpr_values={},
    )
    assert output is body
    assert (
        ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=True)
        == before
    )


@pytest.mark.parametrize("shared", [False, True])
def test_unusual_expression_binding_keeps_full_query(
    shared: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = ast.parse("for i in range(4):\n x = opaque()\n record(x)").body
    loop = body[0]
    assert isinstance(loop, ast.For)
    assignment = loop.body[0]
    assert isinstance(assignment, ast.Assign)
    name = assignment.targets[0] if shared else ast.Name(id="foreign", ctx=ast.Del())
    loop.body.append(ast.Expr(value=name))
    flags: list[bool] = []
    original = bounds._Proof.assigned_names

    def observe(proof: bounds._Proof, node: ast.AST) -> set[str]:
        flags.append(proof.statement_bindings_only)
        return original(proof, node)

    monkeypatch.setattr(bounds._Proof, "assigned_names", observe)
    bounds.simplify_proven_loop_bounds(
        copy.deepcopy(body),
        enabled=True,
        thread_block_dims=(128, 1, 1),
        constexpr_values={},
    )
    assert flags and not any(flags)


def test_shared_target_occurrences_remain_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = ast.parse("for i in range(4):\n x = 1\n y = 2").body
    loop = body[0]
    assert isinstance(loop, ast.For)
    first, second = loop.body
    assert isinstance(first, ast.Assign) and isinstance(second, ast.Assign)
    second.targets[0] = first.targets[0]
    flags: list[bool] = []
    original = bounds._Proof.assigned_names

    def observe(proof: bounds._Proof, node: ast.AST) -> set[str]:
        flags.append(proof.statement_bindings_only)
        return original(proof, node)

    monkeypatch.setattr(bounds._Proof, "assigned_names", observe)
    bounds.simplify_proven_loop_bounds(
        body, enabled=True, thread_block_dims=(128, 1, 1), constexpr_values={}
    )
    assert flags and all(flags)


def _noncanonical_body(kind: str) -> list[ast.stmt]:
    body = ast.parse("y = 0\nfor i in range(4):\n opaque()\nif y < 2:\n record(y)").body
    loop = body[1]
    assert isinstance(loop, ast.For)
    if kind == "for_subclass":

        class CustomFor(ast.For):
            _fields = ("iter", "body", "orelse", "extra_binding")

        replacement = CustomFor(**loop.__dict__)
        replacement.__dict__["extra_binding"] = ast.Name(id="y", ctx=ast.Store())
        body[1] = replacement
    elif kind == "for_fields":
        loop.__dict__["_fields"] = ("iter", "body", "orelse", "extra_binding")
        loop.__dict__["extra_binding"] = ast.Name(id="y", ctx=ast.Store())
    elif kind == "duplicate_fields":
        loop.__dict__["_fields"] = (*loop._fields, "target")
    elif kind == "tuple_root":
        return tuple(body)  # type: ignore[return-value]
    elif kind in {"tuple_body", "tuple_orelse"}:
        field = kind.removeprefix("tuple_")
        setattr(loop, field, tuple(ast.parse("y = 1").body))
        loop.iter = ast.Name(id="foreign", ctx=ast.Store())
    elif kind == "tuple_targets":
        assignment = ast.Assign(
            targets=(ast.Name(id="y", ctx=ast.Store()),),  # type: ignore[arg-type]
            value=ast.Name(id="foreign", ctx=ast.Store()),
        )
        loop.body = [assignment]
    elif kind == "if_fields":
        branch = ast.parse("if flag:\n y = 1").body[0]
        branch.__dict__["_fields"] = ("test",)
        loop.body = [branch]
        loop.iter = ast.Name(id="foreign", ctx=ast.Store())
    else:
        scope = ast.parse("with scope() as y:\n pass").body[0]
        assert isinstance(scope, ast.With)
        loop.body = [scope]
        if kind == "with_fields":
            scope.__dict__["_fields"] = ("body", "type_comment")
        elif kind == "withitem_fields":
            scope.items[0].__dict__["_fields"] = ("context_expr",)
        elif kind == "tuple_items":
            scope.items = tuple(scope.items)  # type: ignore[assignment]
        elif kind == "tuple_with_body":
            scope.body = tuple(ast.parse("y = 1").body)  # type: ignore[assignment]
        else:
            raise AssertionError(kind)
        loop.iter = ast.Name(id="foreign", ctx=ast.Store())
    return body


@pytest.mark.parametrize(
    "kind",
    [
        "for_subclass",
        "for_fields",
        "duplicate_fields",
        "tuple_root",
        "tuple_body",
        "tuple_orelse",
        "tuple_targets",
        "if_fields",
        "with_fields",
        "withitem_fields",
        "tuple_items",
        "tuple_with_body",
    ],
)
def test_noncanonical_schemas_keep_original_queries(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _noncanonical_body(kind)
    assert bounds._supported(body)
    assert bounds._statement_assigned_names(body) is None

    def run() -> list[ast.stmt]:
        return bounds.simplify_proven_loop_bounds(
            copy.deepcopy(body),
            enabled=True,
            thread_block_dims=(128, 1, 1),
            constexpr_values={},
        )

    fast = run()
    monkeypatch.setattr(bounds, "_statement_assigned_names", lambda body: None)
    original = run()

    # ast.dump renders malformed tuple fields with address-bearing reprs.
    # Compare their complete stored contents, preserving tuple/list distinction.
    def fingerprint(value: object) -> object:
        if isinstance(value, ast.AST):
            return type(value), tuple(
                (key, fingerprint(item)) for key, item in sorted(value.__dict__.items())
            )
        if isinstance(value, (list, tuple)):
            return type(value), tuple(fingerprint(item) for item in value)
        return value

    assert fingerprint(fast) == fingerprint(original)


@pytest.mark.parametrize("kind", ["for_fields", "tuple_body", "withitem_fields"])
def test_live_schema_change_falls_back(kind: str) -> None:
    loop = _noncanonical_body(kind)[1]
    proof = bounds._Proof((128, 1, 1), statement_bindings_only=True)
    assert proof.assigned_names(loop) == _assigned_names(loop)
