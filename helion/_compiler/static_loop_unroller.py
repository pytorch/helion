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

    def __init__(self, allow_range: bool) -> None:
        self.allow_range = allow_range

    def visit_For(self, node: ast.For) -> ast.AST | list[ast.AST]:
        # Generic visit to handle nested loops
        node = self.generic_visit(node)  # pyre-ignore[9]

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
        if (
            self.allow_range
            and isinstance(iter_node, ast.Call)
            and isinstance(iter_node.func, ast.Name)
            and iter_node.func.id == "range"
        ):
            range_values = self._extract_range_values(iter_node)
            if range_values is not None:
                return [create(ast.Constant, value=val) for val in range_values]

        return None

    def _extract_range_values(self, range_call: ast.Call) -> list[int] | None:
        """
        Extract values from a range() call if all arguments are constants.
        """
        args = range_call.args

        for arg in args:
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, int):
                return None

        if len(args) == 1:
            return list(range(args[0].value))  # pyre-ignore[16]
        if len(args) == 2:
            return list(range(args[0].value, args[1].value))
        if len(args) == 3:
            return list(range(args[0].value, args[1].value, args[2].value))

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


def unroll_loop(*, node: ast.AST, allow_range: bool) -> ast.AST | list[ast.AST]:
    try:
        return StaticLoopUnroller(allow_range).visit(node)
    except CannotUnrollLoop:
        return node


def unroll_static_loops(*, func: HostFunction, allow_range: bool) -> None:
    new_body: list[ast.stmt] = []
    for stmt in func.body:
        maybe_unrolled = unroll_loop(node=stmt, allow_range=allow_range)
        assert isinstance(maybe_unrolled, ast.stmt)
        new_body.append(maybe_unrolled)
    func.body = new_body
