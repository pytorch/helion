from __future__ import annotations

import ast

from helion._compiler.ast_extension import create
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.ast_extension import unparse
from helion._compiler.source_location import SourceLocation


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
