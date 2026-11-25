"""Keep a full-width unsigned grid index out of ordinary signed scalar arithmetic."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .. import exc

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping


def narrow_uint64_grid_uses(
    kernel: ast.FunctionDef,
    axes: set[str],
    narrow: Callable[[ast.expr], ast.AST],
    reducing_calls: Mapping[str, int],
    backend: str,
) -> None:
    # A range coordinate is a Python integer. There is no single signed type for
    # all UInt64 coordinates and their potential negative intermediate results.
    # A positive remainder or a quotient by at least two is within signed Int64;
    # once reduced, its consumers must see that signed result.
    def raw(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in axes

    def positive(node: ast.AST, minimum: int) -> bool:
        return (
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and minimum <= node.value < 2**63
        )

    class Narrow(ast.NodeTransformer):
        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
            if raw(node.left) and (
                (isinstance(node.op, ast.FloorDiv) and positive(node.right, 2))
                or (isinstance(node.op, ast.Mod) and positive(node.right, 1))
            ):
                return narrow(node)
            return self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> ast.AST:
            if (
                len(node.args) == 2
                and (minimum := reducing_calls.get(ast.unparse(node.func))) is not None
                and raw(node.args[0])
                and positive(node.args[1], minimum)
            ):
                return narrow(node)
            return self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and raw(node):
                raise exc.InvalidConfig(
                    f"{backend} cannot use an unbounded UInt64 grid coordinate in signed "
                    "arithmetic before a bounded quotient or remainder"
                )
            return node

    Narrow().visit(kernel)
