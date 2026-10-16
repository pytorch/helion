from __future__ import annotations

import ast

import pytest

from helion._compiler.ast_extension import create
from helion._compiler.ast_extension import expr_from_string
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.ast_extension import statements_from_string
from helion._compiler.ast_extension import unparse
from helion._compiler.source_location import SourceLocation


def test_statements_from_string_parses_statement_blocks() -> None:
    stmts = statements_from_string("x = 1\ny = x + 1\nif y:\n    z = 2")
    assert isinstance(stmts[0], ast.Assign)
    assert isinstance(stmts[1], ast.Assign)
    assert isinstance(stmts[2], ast.If)

    # Placeholders are replaced in every statement of the block.
    placeholder = expr_from_string("q")
    stmts = statements_from_string("a = {v}\nb = a", v=placeholder)
    assert isinstance(stmts[0].value, ast.Name)
    assert stmts[0].value.id == "q"

    # The single-statement variant enforces its exactly-one contract:
    # a two-statement block does not unpack.
    with pytest.raises(ValueError):
        statement_from_string("x = 1\ny = 2")


def test_unparse_inserts_source_location_comments(tmp_path) -> None:
    src_file = tmp_path / "snippet.py"
    src_lines = ["x = 1\n", "y = x + 1\n"]
    src_file.write_text("".join(src_lines))

    location = SourceLocation(
        lineno=2,
        colno=0,
        end_lineno=2,
        end_colno=len(src_lines[1].rstrip("\n")),
        name="snippet",
        filename=str(src_file),
    )

    with location:
        assign_stmt = statement_from_string("result = y")

    module = create(ast.Module, body=[assign_stmt], type_ignores=[])
    rendered = unparse(module)

    line_comment = f"# src[{src_file.name}:2]: y = x + 1"
    assert line_comment in rendered
    assert "result = y" in rendered


def test_duplicate_locations_not_duplicated(tmp_path) -> None:
    src_file = tmp_path / "snippet.py"
    src_file.write_text("a = 1\nb = 2\n")

    location = SourceLocation(
        lineno=2,
        colno=0,
        end_lineno=2,
        end_colno=1,
        name="snippet",
        filename=str(src_file),
    )

    with location:
        stmt1 = statement_from_string("first = b")
        stmt2 = statement_from_string("second = b")

    module = create(ast.Module, body=[stmt1, stmt2], type_ignores=[])
    rendered = unparse(module)

    line_comment = f"# src[{src_file.name}:2]: b = 2"
    assert rendered.count(line_comment) == 1
    assert "first = b" in rendered
    assert "second = b" in rendered
