from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import NoReturn

from .ast_extension import create

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .host_function import HostFunction


class CannotUnrollLoop(Exception):
    pass


class StaticLoopUnroller(ast.NodeTransformer):
    """
    A compiler optimization pass that unrolls static for loops.

    TODO(oulgen): This pass is primitive, does not handle for.orelse, break, continue etc
    """

    def visit_For(self, node: ast.For) -> ast.AST | list[ast.AST]:
        # Generic visit to handle nested loops
        node = self.generic_visit(  # pyright: ignore[reportAssignmentType]
            node
        )

        # Check if this is a static loop that can be unrolled
        if static_values := self._extract_static_values(node.iter):
            return self._unroll_loop(node, static_values)

        return node

    def visit_Break(self, node: ast.Break) -> NoReturn:
        raise CannotUnrollLoop

    def visit_Continue(self, node: ast.Continue) -> NoReturn:
        raise CannotUnrollLoop

    def _extract_static_values(self, iter_node: ast.expr) -> list[ast.expr] | None:
        """
        Check if iterator is static, and if so extract those values
        """
        if isinstance(iter_node, (ast.List, ast.Tuple)):
            return iter_node.elts
        return None

    def _unroll_loop(
        self, loop_node: ast.For, static_values: Sequence[ast.AST]
    ) -> ast.AST | list[ast.AST]:
        unrolled_statements = []

        for value in static_values:
            assignment = create(
                ast.Assign,
                targets=[loop_node.target],
                value=value,
            )
            unrolled_statements.append(assignment)

            # TODO(oulgen): Should we deepcopy these to avoid reference issues?
            unrolled_statements.extend(loop_node.body)

        if loop_node.orelse:
            raise CannotUnrollLoop
        return unrolled_statements


def unroll_static_loops(func: HostFunction) -> None:
    new_body = []
    for stmt in func.body:
        try:
            unrolled_stmts = StaticLoopUnroller().visit(stmt)
        except CannotUnrollLoop:
            new_body.append(stmt)
        else:
            assert isinstance(unrolled_stmts, ast.stmt)
            new_body.append(unrolled_stmts)
    func.body = new_body
