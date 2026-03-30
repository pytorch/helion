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
        # pyrefly: ignore [bad-assignment]
        node = self.generic_visit(node)
        node.body = self.unroll_counted_whiles(node.body)
        node.orelse = self.unroll_counted_whiles(node.orelse)

        # Check if this is a static loop that can be unrolled
        if static_values := self._extract_static_values(node.iter):
            return self._unroll_loop(node, static_values)

        return node

    def visit_Break(self, node: ast.Break) -> NoReturn:
        raise CannotUnrollLoop

    def visit_Continue(self, node: ast.Continue) -> NoReturn:
        raise CannotUnrollLoop

    def visit_While(self, node: ast.While) -> ast.AST | list[ast.AST]:
        visited = self.generic_visit(node)
        assert isinstance(visited, ast.While)
        node = visited
        node.body = self.unroll_counted_whiles(node.body)
        node.orelse = self.unroll_counted_whiles(node.orelse)
        return node

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.AST]:
        visited = self.generic_visit(node)
        assert isinstance(visited, ast.If)
        node = visited
        node.body = self.unroll_counted_whiles(node.body)
        node.orelse = self.unroll_counted_whiles(node.orelse)
        return node

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
            unrolled_statements.extend(
                loop_node.body  # pyrefly: ignore[bad-argument-type]
            )

        if loop_node.orelse:
            raise CannotUnrollLoop
        return unrolled_statements  # pyrefly: ignore[bad-return]

    def unroll_counted_whiles(
        self, statements: list[ast.stmt], known_scalars: dict[str, int] | None = None
    ) -> list[ast.stmt]:
        env = {} if known_scalars is None else known_scalars
        result: list[ast.stmt] = []
        for stmt in statements:
            try:
                transformed = self.visit(stmt)
            except CannotUnrollLoop:
                transformed = stmt
            stmt_list = transformed if isinstance(transformed, list) else [transformed]
            for item in stmt_list:
                if isinstance(item, ast.While):
                    unrolled = self._unroll_counted_while(item, env)
                    if unrolled is not None:
                        result.extend(unrolled)
                        continue
                result.append(item)
                self._update_known_scalars(env, item)
        return result

    def _unroll_counted_while(
        self, node: ast.While, env: dict[str, int]
    ) -> list[ast.stmt] | None:
        if node.orelse:
            return None
        loop_info = self._extract_counted_while(node, env)
        if loop_info is None:
            return None
        var_name, current, limit, delta = loop_info
        assert isinstance(node.test, ast.Compare)
        trip_count = 0
        probe = current
        while self._compare_counted_while(probe, node.test.ops[0], limit):
            probe += delta
            trip_count += 1
            if trip_count > 10000:
                return None
        env[var_name] = probe
        unrolled: list[ast.stmt] = []
        local_env = dict(env)
        local_env[var_name] = current
        for _ in range(trip_count):
            unrolled.extend(self.unroll_counted_whiles(node.body, local_env))
        return unrolled

    def _extract_counted_while(
        self, node: ast.While, env: dict[str, int]
    ) -> tuple[str, int, int, int] | None:
        test = node.test
        if (
            not isinstance(test, ast.Compare)
            or len(test.ops) != 1
            or len(test.comparators) != 1
            or not isinstance(test.left, ast.Name)
            or test.left.id not in env
        ):
            return None
        limit = self._literal_int(test.comparators[0])
        if limit is None:
            return None
        delta = self._extract_induction_delta(node.body, test.left.id)
        if delta in (None, 0):
            return None
        return test.left.id, env[test.left.id], limit, delta

    def _extract_induction_delta(
        self, body: list[ast.stmt], var_name: str
    ) -> int | None:
        delta: int | None = None
        for stmt in body:
            if self._has_nested_scalar_update(stmt, var_name):
                return None
            if isinstance(stmt, ast.Assign):
                if (
                    len(stmt.targets) != 1
                    or not isinstance(stmt.targets[0], ast.Name)
                    or stmt.targets[0].id != var_name
                ):
                    continue
                value = stmt.value
                if (
                    not isinstance(value, ast.BinOp)
                    or not isinstance(value.left, ast.Name)
                    or value.left.id != var_name
                ):
                    return None
                step = self._literal_int(value.right)
                if step is None:
                    return None
                if delta is not None:
                    return None
                if isinstance(value.op, ast.Add):
                    delta = step
                elif isinstance(value.op, ast.Sub):
                    delta = -step
                else:
                    return None
            elif isinstance(stmt, ast.AugAssign):
                if not isinstance(stmt.target, ast.Name) or stmt.target.id != var_name:
                    continue
                step = self._literal_int(stmt.value)
                if step is None:
                    return None
                if delta is not None:
                    return None
                if isinstance(stmt.op, ast.Add):
                    delta = step
                elif isinstance(stmt.op, ast.Sub):
                    delta = -step
                else:
                    return None
        return delta

    def _has_nested_scalar_update(self, stmt: ast.stmt, var_name: str) -> bool:
        for child in ast.walk(stmt):
            if child is stmt:
                continue
            if isinstance(child, ast.Assign):
                if (
                    len(child.targets) == 1
                    and isinstance(child.targets[0], ast.Name)
                    and child.targets[0].id == var_name
                ):
                    return True
            elif isinstance(child, ast.AugAssign):
                if isinstance(child.target, ast.Name) and child.target.id == var_name:
                    return True
        return False

    def _compare_counted_while(self, current: int, op: ast.cmpop, limit: int) -> bool:
        if isinstance(op, ast.Lt):
            return current < limit
        if isinstance(op, ast.LtE):
            return current <= limit
        if isinstance(op, ast.Gt):
            return current > limit
        if isinstance(op, ast.GtE):
            return current >= limit
        return False

    def _literal_int(self, node: ast.AST) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "zeros"
                and node.args
                and isinstance(node.args[0], ast.List)
                and len(node.args[0].elts) == 0
            ):
                return 0
        return None

    def _update_known_scalars(self, env: dict[str, int], stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                return
            value = self._literal_int(stmt.value)
            if value is None:
                env.pop(stmt.targets[0].id, None)
            else:
                env[stmt.targets[0].id] = value
        elif isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name) or stmt.target.id not in env:
                return
            value = self._literal_int(stmt.value)
            if value is None:
                env.pop(stmt.target.id, None)
                return
            if isinstance(stmt.op, ast.Add):
                env[stmt.target.id] += value
            elif isinstance(stmt.op, ast.Sub):
                env[stmt.target.id] -= value
            else:
                env.pop(stmt.target.id, None)


def unroll_static_loops(func: HostFunction) -> None:
    unroller = StaticLoopUnroller()
    new_body = []
    for stmt in func.body:
        try:
            unrolled_stmts = unroller.visit(stmt)
        except CannotUnrollLoop:
            new_body.append(stmt)
        else:
            if isinstance(unrolled_stmts, list):
                new_body.extend(unrolled_stmts)
            else:
                new_body.append(unrolled_stmts)
    func.body = unroller.unroll_counted_whiles(new_body)
