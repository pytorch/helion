"""Startup TMA copies require a full participating warp, not an outer election."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


def _issue_lines() -> Callable[[list[SimpleNamespace]], list[str]]:
    # Source-only collection: this test must not import CuTe or initialize CUDA.
    source = (
        Path(__file__).resolve().parents[1] / "helion/_compiler/cute/chained_startup.py"
    )
    with source.open() as stream:
        module = ast.parse(stream.read())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "issue_lines"
    )
    namespace: dict[str, object] = {
        "math": math,
        "cast": cast,
        "StartupInput": SimpleNamespace,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return cast(
        "Callable[[list[SimpleNamespace]], list[str]]", namespace["issue_lines"]
    )


@pytest.mark.parametrize("height,width", [(64, 32), (64, 128), (128, 64), (128, 256)])
@pytest.mark.parametrize("inner", [0, 1])
@pytest.mark.parametrize("roles", [("a",), ("b",), ("a", "b")])
def test_startup_tma_copy_is_warp_collective(height, width, inner, roles):
    transfers = [
        SimpleNamespace(
            role=role,
            shape=(height, width),
            inner=inner,
            wrapper={"tile": (height, width)},
            row="origin_row",
            col="origin_col",
        )
        for role in roles
    ]
    lines = _issue_lines()(transfers)
    tree = ast.parse("\n".join(lines))
    assert not any(isinstance(node, ast.With) for node in ast.walk(tree))
    issues = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "chain_warp == 0"
    ]
    assert len(issues) == len(roles)
    for role, issue in zip(roles, issues, strict=True):
        assert len(issue.body) == 1 and not issue.orelse
        statement = issue.body[0]
        assert isinstance(statement, ast.Expr)
        assert ast.unparse(statement.value) == (
            f"cute.copy(chain_start_{role}_atom, chain_start_{role}_partition, "
            f"chain_start_{role}_shared, tma_bar_ptr=chain_start_bar)"
        )
    expected_bytes = len(roles) * height * width * 2
    assert (
        f"    cute.arch.mbarrier_arrive_and_expect_tx(chain_start_bar, {expected_bytes})"
        in lines
    )
    assert lines.index("cute.arch.sync_threads()") < next(
        i for i, line in enumerate(lines) if line == "if chain_warp == 0:"
    )
