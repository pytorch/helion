from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.proven_loop_bounds import simplify_proven_loop_bounds

MARKER = "_cute_proven_loop_bounds_0_abi_version"


def _rewrite(
    body: list[ast.stmt], constants: dict[str, int] | None = None
) -> list[ast.stmt]:
    return simplify_proven_loop_bounds(
        body,
        enabled=True,
        thread_block_dims=(128, 1, 1),
        constexpr_values={} if constants is None else constants,
    )


def _first_marker(body: list[ast.stmt]) -> str:
    first = body[0]
    assert isinstance(first, ast.Assign)
    assert isinstance(first.targets[0], ast.Name)
    return first.targets[0].id


@pytest.mark.parametrize(
    "schema",
    [
        "if_fields",
        "tuple_for_body",
        "ifexp_fields",
        "ifexp_tuple_body",
        "boolop_fields",
        "tuple_boolop_values",
        "nested_boolop_values",
        "unrelated_ifexp",
        "unrelated_boolop",
    ],
)
def test_rewrite_preserves_marker_name_hidden_by_nonstandard_schema(
    schema: str,
) -> None:
    if schema == "if_fields":
        body = ast.parse(f"for i in range(4):\n if i < 4:\n  record({MARKER})").body
        loop = body[0]
        assert isinstance(loop, ast.For)
        branch = loop.body[0]
        assert isinstance(branch, ast.If)
        branch.__dict__["_fields"] = ("test", "orelse")
        expected = [37] * 4
    elif schema == "tuple_for_body":
        body = ast.parse(f"for i in range(4):\n record({MARKER})\n record(i < 4)").body
        loop = body[0]
        assert isinstance(loop, ast.For)
        loop.body = tuple(loop.body)  # type: ignore[assignment]
        expected = [37, True] * 4
    else:
        value: ast.expr
        if schema in ("ifexp_fields", "ifexp_tuple_body", "unrelated_ifexp"):
            conditional = ast.IfExp(
                test=ast.Constant(value=True),
                body=ast.Name(id=MARKER, ctx=ast.Load()),
                orelse=ast.Constant(value=False),
            )
            if schema == "ifexp_fields":
                conditional.__dict__["_fields"] = ("test", "orelse")
                value = conditional
            elif schema == "ifexp_tuple_body":
                conditional.body = (conditional.body,)  # type: ignore[assignment]
                value = conditional
            else:

                class IfExp(ast.expr):
                    _fields = ("test", "orelse")

                value = IfExp(
                    test=conditional.test,
                    body=conditional.body,
                    orelse=conditional.orelse,
                )
        else:
            boolean = ast.BoolOp(
                op=ast.And(),
                values=[ast.Constant(value=True), ast.Name(id=MARKER, ctx=ast.Load())],
            )
            if schema == "boolop_fields":
                boolean.__dict__["_fields"] = ("op",)
            elif schema == "tuple_boolop_values":
                boolean.values = tuple(boolean.values)  # type: ignore[assignment]
            elif schema == "nested_boolop_values":
                boolean.values[1] = [boolean.values[1]]  # type: ignore[call-overload]
            if schema == "unrelated_boolop":

                class BoolOp(ast.expr):
                    _fields = ("op",)

                value = BoolOp(op=boolean.op, values=boolean.values)
            else:
                value = boolean
        body = [
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="record", ctx=ast.Load()),
                    args=[value],
                    keywords=[],
                )
            )
        ]
        expected = [37]
    assert not any(
        isinstance(node, ast.Name) and node.id == MARKER
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
    )
    output = _rewrite(body)
    assert _first_marker(output) == "_cute_proven_loop_bounds_1_abi_version"
    recorded: list[object] = []
    namespace = {MARKER: 37, "record": recorded.append}
    exec(
        compile(ast.Module(body=output, type_ignores=[]), "<marker-proof>", "exec"),
        namespace,
    )
    assert recorded == expected


def test_last_marker_load_removed_by_proof_leaves_original_ordinal_free() -> None:
    body = ast.parse(f"for i in range(4):\n record(i < {MARKER})").body
    output = _rewrite(body, {MARKER: 4})
    assert _first_marker(output) == MARKER
    assert "record(True)" in ast.unparse(ast.Module(body=output, type_ignores=[]))


def test_import_marker_stays_reserved_when_its_branch_is_removed() -> None:
    body = ast.parse(
        f"for i in range(4):\n if i < 4:\n  record()\n else:\n  import opaque as {MARKER}"
    ).body
    output = _rewrite(body)
    assert _first_marker(output) == "_cute_proven_loop_bounds_1_abi_version"
    assert "import opaque" not in ast.unparse(ast.Module(body=output, type_ignores=[]))
