from __future__ import annotations

import ast
import builtins
import operator
import types
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import TypeVar
from typing import cast
import weakref

from .. import language as language_module
from ..language._decorators import is_api_func
from .ast_extension import ExtendedAST
from .ast_extension import create
from .compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from collections.abc import Callable
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


_CONSTANT_OPERATORS: dict[type[ast.AST], Callable[..., object]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Not: operator.not_,
    ast.USub: operator.neg,
}
_NOT_CONSTANT = object()
_A = TypeVar("_A", bound=ast.AST)


class _HostStaticRangeUnroller:
    """Unrolls host-level ``hl.static_range`` loops and folds host ``if``s
    whose condition is a constant.

    Each copy of a loop body sees the loop variable as an int literal, so an
    ``if r == 3:`` in it keeps one branch per copy.  The body of a folded host
    loop can then hold a static period of different layer types::

        for p in range(n):
            for r in hl.static_range(4):
                if r == 3:
                    ...  # top-level device loops of one layer type
                else:
                    ...  # another type, indexing w[3 * p + r]

    Device loops are left alone: ``hl.static_range`` inside them is unrolled
    by codegen.
    """

    def __init__(self, func: HostFunction) -> None:
        self.func = func
        assigned = {
            node.id
            for stmt in func.body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)
        }
        self.local_names = assigned | {arg.arg for arg in func.args.args}
        # ``hl.constexpr`` arguments the body never reassigns.
        self.constants = {
            name: value
            for name, value in func.constexpr_args.items()
            if name not in assigned
        }

    def statements(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for stmt in statements:
            # Only ``if``s around device loops: type propagation resolves the
            # others, and their host code stays as written.
            if isinstance(stmt, ast.If) and self._holds_device_loop(stmt):
                test = self._constant(stmt.test)
                if test is not _NOT_CONSTANT:
                    result.extend(self.statements(stmt.body if test else stmt.orelse))
                    continue
            if isinstance(stmt, ast.For) and isinstance(stmt.iter, ast.Call):
                function = self._resolve(stmt.iter.func)
                if function is language_module.static_range:
                    unrolled = self._unroll(stmt, stmt.iter)
                    if unrolled is not None:
                        result.extend(self.statements(unrolled))
                        continue
                elif self._is_device_loop(function):
                    result.append(stmt)
                    continue
            if isinstance(stmt, ast.For):
                stmt.body = self.statements(stmt.body)
            result.append(stmt)
        return result

    @staticmethod
    def _is_device_loop(function: object) -> bool:
        return is_api_func(function) and function._is_device_loop

    def _holds_device_loop(self, node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.For)
            and isinstance(child.iter, ast.Call)
            and self._is_device_loop(self._resolve(child.iter.func))
            for child in ast.walk(node)
        )

    def _unroll(self, node: ast.For, call: ast.Call) -> list[ast.stmt] | None:
        """One copy of the body per value, or ``None`` if the loop is not a
        static ``for <name> in hl.static_range(<ints>)``."""
        bounds = [self._int(arg) for arg in call.args]
        steps = [self._int(kw.value) for kw in call.keywords if kw.arg == "step"]
        if (
            node.orelse
            or not isinstance(node.target, ast.Name)
            or len(steps) != len(call.keywords)
            or not 1 <= len(bounds) <= len(bounds) + len(steps) <= 3
        ):
            return None
        args = [0, *bounds, *steps] if len(bounds) == 1 else [*bounds, *steps]
        name = node.target.id
        if any(
            isinstance(child, ast.Name)
            and child.id == name
            and not isinstance(child.ctx, ast.Load)
            for stmt in node.body
            for child in ast.walk(stmt)
        ) or not all(isinstance(arg, int) for arg in args):
            return None
        # pyrefly: ignore [no-matching-overload]
        values = range(*args)
        return [
            _substitute(stmt, name, value) for value in values for stmt in node.body
        ]

    def _resolve(self, node: ast.expr) -> object:
        """The global object that a ``name`` or ``module.attr`` chain names."""
        if isinstance(node, ast.Name) and node.id not in self.local_names:
            # pyrefly: ignore [missing-attribute]
            scope = self.func.fn.__globals__
            return scope[node.id] if node.id in scope else vars(builtins).get(node.id)
        if isinstance(node, ast.Attribute):
            module = self._resolve(node.value)
            if isinstance(module, types.ModuleType):
                return vars(module).get(node.attr)
        return None

    def _int(self, node: ast.expr) -> int | None:
        value = self._constant(node)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def _constant(self, node: ast.expr) -> object:
        """The value of an expression of literals, ``hl.constexpr`` arguments
        and arithmetic, comparison and boolean operators, else
        ``_NOT_CONSTANT``."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.constants.get(node.id, _NOT_CONSTANT)
        if isinstance(node, ast.UnaryOp):
            children, ops = [node.operand], [node.op]
        elif isinstance(node, ast.BinOp):
            children, ops = [node.left, node.right], [node.op]
        elif isinstance(node, ast.Compare):
            children, ops = [node.left, *node.comparators], node.ops
        elif isinstance(node, ast.BoolOp):
            children, ops = node.values, []
        else:
            return _NOT_CONSTANT
        values = [self._constant(child) for child in children]
        # The nodes are ExtendedAST subclasses of the ``ast`` operator types.
        functions = [
            next(
                (fn for cls, fn in _CONSTANT_OPERATORS.items() if isinstance(op, cls)),
                None,
            )
            for op in ops
        ]
        if any(value is _NOT_CONSTANT for value in values) or None in functions:
            return _NOT_CONSTANT
        if isinstance(node, ast.BoolOp):
            return (all if isinstance(node.op, ast.And) else any)(values)
        if isinstance(node, ast.Compare):
            return all(
                # pyrefly: ignore [not-callable]
                fn(left, right)
                for fn, left, right in zip(
                    functions, values[:-1], values[1:], strict=True
                )
            )
        # pyrefly: ignore [not-callable]
        return functions[0](*values)


def _substitute(node: _A, name: str, value: int) -> _A:
    """A copy of ``node`` with the loads of ``name`` replaced by ``value``."""
    assert isinstance(node, ExtendedAST)
    if isinstance(node, ast.Name) and node.id == name:
        with node._location:
            return cast("_A", create(ast.Constant, value=value, kind=None))
    fields: dict[str, object] = {}
    for field, child in node.fields().items():
        if isinstance(child, list):
            fields[field] = [
                _substitute(item, name, value) if isinstance(item, ast.AST) else item
                for item in child
            ]
        elif isinstance(child, ast.AST):
            fields[field] = _substitute(child, name, value)
        else:
            fields[field] = child
    return cast("_A", node.new(fields))


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
    # Before the host ``hl.static_range`` unroll, so a collective in a static
    # loop body is one site whose buffers every copy of the body shares.
    from .collective_expansion import expand_collectives

    expand_collectives(func)
    if CompileEnvironment.current().backend.supports_folded_host_loops:
        func.body = _HostStaticRangeUnroller(func).statements(func.body)


# Host ``if``s that compiler passes emit to choose code by shape: folded
# after type propagation like the ``if``s around device loops.
_static_host_ifs: weakref.WeakSet[ast.If] = weakref.WeakSet()


def mark_static_host_if(node: ast.stmt) -> ast.stmt:
    assert isinstance(node, ast.If)
    _static_host_ifs.add(node)
    return node


def fold_static_host_ifs(func: HostFunction) -> None:
    """Keep only the taken branch of each host ``if`` around device loops (or
    marked by ``mark_static_host_if``) whose condition type propagation
    resolved, e.g. from the shapes: lowering would trace both branches, but
    type propagation only visited one."""
    unroller = _HostStaticRangeUnroller(func)

    def fold(statements: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for stmt in statements:
            if isinstance(stmt, ast.If) and (
                stmt in _static_host_ifs or unroller._holds_device_loop(stmt)
            ):
                type_info = getattr(stmt.test, "_type_info", None)
                try:
                    taken = None if type_info is None else type_info.truth_value()
                except NotImplementedError:
                    taken = None
                if taken is not None:
                    result.extend(fold(stmt.body if taken else stmt.orelse))
                    continue
            if isinstance(stmt, ast.For) and not (
                isinstance(stmt.iter, ast.Call)
                and unroller._is_device_loop(unroller._resolve(stmt.iter.func))
            ):
                stmt.body = fold(stmt.body)
            result.append(stmt)
        return result

    func.body = fold(func.body)
