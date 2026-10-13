from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from helion._compiler.cute import fuse_two_pass_loads as fuser


@pytest.mark.parametrize(
    "source",
    [
        "a = source + 1\na += 1\ndel a",
        "for lane in range(8):\n a = lane\nelse:\n b = 0",
        "with resource() as token:\n token = value",
        "a = (captured := index)\nb = [x for x in values]",
        "def fn(arg):\n local = arg\n return local",
        "import x as imported\nfrom x import y as another",
    ],
)
def test_store_summary_matches_complete_original_walk(source):
    body = ast.parse(source).body
    expected = fuser._scalar_bindings(ast.Module(body=body, type_ignores=[]))
    names = {"discard_old_output"}
    with patch.object(
        fuser, "_scalar_bindings", side_effect=AssertionError("extra walk")
    ):
        result = fuser._own_statement_occurrences(body, scalar_bindings=names)
    assert result is body
    assert names == expected


def test_unknown_fields_and_nested_store_contexts_are_not_skipped():
    class CustomExpr(ast.expr):
        _fields = ("hidden",)

    expression = CustomExpr()
    expression.hidden = ast.Name(id="hidden_write", ctx=ast.Store())
    body = [ast.Expr(value=expression)]
    names = set()
    assert fuser._own_statement_occurrences(body, scalar_bindings=names) is body
    assert names == {"hidden_write"}


def test_shared_statements_rescan_the_actual_cloned_body():
    statement = ast.parse("x = source").body[0]
    body = [statement, statement, ast.parse("last = source").body[0]]
    names = {"discard_old_output"}
    with patch.object(fuser, "_scalar_bindings", wraps=fuser._scalar_bindings) as query:
        result = fuser._own_statement_occurrences(body, scalar_bindings=names)
    assert result is not body and result[0] is not result[1]
    query.assert_called_once()
    assert query.call_args.args[0].body is result
    assert names == {"x", "last"}
