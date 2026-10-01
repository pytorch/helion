"""CSE identical max collectives in a proved one-tile private-cache sweep.

An abstract interpreter executes only bounded constexpr loops. Scalar values
are immutable DAG nodes, so loop carries, assignment order and every typed
operation remain visible without expanding expressions exponentially. Raw-load
identities come exclusively from the preceding immutable-load cache proof.
Only a duplicate collective call is replaced; masks, casts, carries, cache
writes and first-tile nonfinite handling remain in their original positions.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import create
from .bounded_loop_cache import CachedLoops
from .bounded_loop_cache import private_fragment_accesses

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Mapping

_MAX_HELPERS = frozenset(
    {
        "_cute_grouped_reduce_warp",
        "_cute_grouped_reduce_shared_two_stage",
        "_cute_grouped_reduce_shared_serial",
        "_cute_grouped_reduce_shared_tree",
    }
)
_SCALARS = frozenset(
    {
        "cutlass.Boolean",
        "cutlass.Int32",
        "cutlass.Int64",
        "cutlass.Uint32",
        "cutlass.Uint64",
        "cutlass.Float16",
        "cutlass.BFloat16",
        "cutlass.Float32",
    }
)


class _Decline(Exception):
    pass


@dataclass(frozen=True)
class _Value:
    node: int
    dtype: str
    integer: int | bool | None = None


class _Interpreter:
    def __init__(
        self,
        cached: CachedLoops,
        arguments: Collection[str],
        constants: Mapping[str, int],
        renames: Mapping[str, str],
    ) -> None:
        self.nodes: dict[tuple[object, ...], int] = {}
        self.values: dict[str, _Value] = {}
        self.renames = renames
        self.fragments = {fragment.name: fragment for fragment in cached.fragments}
        self.origins = {id(origin.statement): origin for origin in cached.origins}
        self.storage: dict[tuple[str, int], _Value] = {}
        self.reductions: dict[int, tuple[str, _Value]] = {}
        self.rewrites: dict[int, str] = {}
        self.visits = 0
        self.writes = Counter(
            self.name(node.id)
            for statement in cached.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
        )
        for name in arguments:
            self.values[self.name(name)] = self.node("argument", (name,), "unknown")
        for name, value in constants.items():
            if self.writes[self.name(name)]:
                raise _Decline
            self.values[self.name(name)] = self.constant(value)

    def name(self, name: str) -> str:
        return self.renames.get(name, name)

    def visit(self) -> None:
        self.visits += 1
        if self.visits > 32768:
            raise _Decline

    def node(
        self,
        operation: str,
        operands: tuple[object, ...],
        dtype: str,
        integer: int | bool | None = None,
    ) -> _Value:
        key = operation, operands, dtype
        index = self.nodes.setdefault(key, len(self.nodes))
        return _Value(index, dtype, integer)

    def constant(self, value: object, dtype: str = "literal") -> _Value:
        # repr gives stable identities for NaNs and distinguishes signed zero.
        integer = cast("int | bool", value) if type(value) in {int, bool} else None
        return self.node(
            "constant", (type(value).__name__, repr(value)), dtype, integer
        )

    def expression(self, expression: ast.expr) -> _Value:
        self.visit()
        if isinstance(expression, ast.Name):
            value = self.values.get(self.name(expression.id))
            if value is None:
                raise _Decline
            return value
        if isinstance(expression, ast.Constant):
            return self.constant(expression.value)
        if isinstance(expression, ast.Subscript):
            index = self.expression(expression.slice)
            if (
                isinstance(expression.value, ast.Name)
                and expression.value.id in self.fragments
            ):
                if type(index.integer) is not int:
                    raise _Decline
                value = self.storage.get((expression.value.id, index.integer))
                if value is None:
                    raise _Decline
                return value
            value = self.expression(expression.value)
            if (
                value.dtype != "index_tuple"
                or type(index.integer) is not int
                or not 0 <= index.integer < 3
            ):
                raise _Decline
            return self.node("subscript", (value.node, index.node), "cutlass.Int32")
        if isinstance(expression, ast.IfExp):
            test = self.expression(expression.test)
            if type(test.integer) is bool:
                return self.expression(
                    expression.body if test.integer else expression.orelse
                )
            yes = self.expression(expression.body)
            no = self.expression(expression.orelse)
            return self.node(
                "if",
                (test.node, yes.node, no.node),
                yes.dtype if yes.dtype == no.dtype else "unknown",
            )
        if isinstance(expression, ast.Compare):
            if len(expression.ops) != 1 or len(expression.comparators) != 1:
                raise _Decline
            left = self.expression(expression.left)
            right = self.expression(expression.comparators[0])
            op = expression.ops[0]
            if left.integer is not None and right.integer is not None:
                if isinstance(op, ast.Eq):
                    return self.constant(
                        left.integer == right.integer, "cutlass.Boolean"
                    )
                if isinstance(op, ast.NotEq):
                    return self.constant(
                        left.integer != right.integer, "cutlass.Boolean"
                    )
            return self.node(
                type(op).__name__, (left.node, right.node), "cutlass.Boolean"
            )
        if isinstance(expression, ast.BoolOp):
            values = tuple(self.expression(value) for value in expression.values)
            if any(
                value.dtype != "cutlass.Boolean" and type(value.integer) is not bool
                for value in values
            ):
                raise _Decline
            return self.node(
                type(expression.op).__name__,
                tuple(value.node for value in values),
                "cutlass.Boolean",
            )
        if isinstance(expression, ast.BinOp):
            left = self.expression(expression.left)
            right = self.expression(expression.right)
            # Only Python constexpr index arithmetic is evaluated here.
            # Typed arithmetic keeps its complete operation/width in the DAG.
            left_constant, right_constant = left.integer, right.integer
            if (
                left.dtype == right.dtype == "literal"
                and type(left_constant) is int
                and type(right_constant) is int
            ):
                if isinstance(expression.op, ast.Add):
                    return self.constant(left_constant + right_constant)
                if isinstance(expression.op, ast.Sub):
                    return self.constant(left_constant - right_constant)
                if isinstance(expression.op, ast.Mult):
                    return self.constant(left_constant * right_constant)
            return self.node(
                type(expression.op).__name__,
                (left.node, right.node),
                left.dtype if left.dtype == right.dtype else "unknown",
            )
        if isinstance(expression, ast.UnaryOp):
            value = self.expression(expression.operand)
            if value.dtype == "literal" and type(value.integer) is int:
                if isinstance(expression.op, ast.USub):
                    return self.constant(-value.integer)
                if isinstance(expression.op, ast.UAdd):
                    return value
            return self.node(type(expression.op).__name__, (value.node,), value.dtype)
        if isinstance(expression, ast.Call):
            function = ast.unparse(expression.func)
            if (
                function in _SCALARS
                and len(expression.args) == 1
                and not expression.keywords
            ):
                value = self.expression(expression.args[0])
                if value.dtype == function:
                    return value
                if function == "cutlass.Boolean" and type(value.integer) is bool:
                    return self.constant(value.integer, function)
                if (
                    function
                    in {
                        "cutlass.Int32",
                        "cutlass.Int64",
                        "cutlass.Uint32",
                        "cutlass.Uint64",
                    }
                    and value.integer is not None
                ):
                    width = 32 if function.endswith("32") else 64
                    number = int(value.integer) % (1 << width)
                    if ".Int" in function and number >= 1 << (width - 1):
                        number -= 1 << width
                    return self.constant(number, function)
                return self.node(function, (value.node,), function)
            if (
                function == "float"
                and len(expression.args) == 1
                and not expression.keywords
                and isinstance(expression.args[0], ast.Constant)
                and expression.args[0].value in {"-inf", "inf", "nan"}
            ):
                return self.constant(float(expression.args[0].value))
            if (
                function in {"cute.arch.thread_idx", "cute.arch.block_idx"}
                and not expression.args
                and not expression.keywords
            ):
                return self.node(function, (), "index_tuple")
            if (
                function in {"cute.arch.fmax", "cute.math.max"}
                and len(expression.args) == 2
            ):
                values = tuple(self.expression(arg) for arg in expression.args)
                keywords = tuple(
                    (kw.arg, self.expression(kw.value).node)
                    for kw in expression.keywords
                )
                if any(kw.arg != "propagate_nan" for kw in expression.keywords):
                    raise _Decline
                dtype = (
                    values[0].dtype if values[0].dtype == values[1].dtype else "unknown"
                )
                return self.node(
                    function, (*[value.node for value in values], keywords), dtype
                )
        raise _Decline

    def index(self, name: str, expression: ast.expr) -> int:
        value = self.expression(expression)
        if (
            type(value.integer) is not int
            or not 0 <= value.integer < self.fragments[name].elements
        ):
            raise _Decline
        return value.integer

    def assignment(self, statement: ast.Assign, depth: int) -> None:
        if len(statement.targets) != 1:
            raise _Decline
        target = statement.targets[0]
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id in self.fragments
        ):
            name = target.value.id
            self.storage[(name, self.index(name, target.slice))] = self.expression(
                statement.value
            )
            return
        if not isinstance(target, ast.Name):
            raise _Decline
        name = self.name(target.id)
        if target.id in self.fragments:
            # The caller has just checked exact unique private allocations.
            return
        origin = self.origins.get(id(statement))
        if origin is not None:
            if ast.dump(statement.value, include_attributes=False) != origin.expression:
                raise _Decline
            index = self.index(origin.fragment, origin.index)
            value = self.node(
                "immutable_cache_input", (origin.fragment, index), origin.dtype
            )
        else:
            call = statement.value
            while (
                isinstance(call, ast.Call)
                and ast.unparse(call.func) == "cutlass.Float32"
                and len(call.args) == 1
                and not call.keywords
            ):
                call = call.args[0]
            if isinstance(call, ast.Call) and ast.unparse(call.func) in _MAX_HELPERS:
                if (
                    depth != 1
                    or self.writes[name] != 1
                    or len(call.args) < 3
                    or not isinstance(call.args[1], ast.Constant)
                    or call.args[1].value != "max"
                ):
                    raise _Decline
                values = tuple(self.expression(arg) for arg in call.args)
                if values[0].dtype != "cutlass.Float32":
                    raise _Decline
                keywords = tuple(
                    (kw.arg, self.expression(kw.value).node) for kw in call.keywords
                )
                value = self.node(
                    ast.unparse(call.func),
                    (*[item.node for item in values], keywords),
                    "cutlass.Float32",
                )
                previous = self.reductions.get(value.node)
                if previous is not None:
                    self.rewrites[id(call)] = previous[0]
                    value = previous[1]
                else:
                    self.reductions[value.node] = (target.id, value)
            else:
                value = self.expression(statement.value)
        self.values[name] = value

    def statement(self, statement: ast.stmt, depth: int = 0) -> None:
        self.visit()
        if isinstance(statement, ast.Assign):
            self.assignment(statement, depth)
            return
        if isinstance(statement, ast.For):
            iterator = statement.iter
            if (
                statement.orelse
                or not isinstance(statement.target, ast.Name)
                or not isinstance(iterator, ast.Call)
                or ast.unparse(iterator.func) != "cutlass.range_constexpr"
                or len(iterator.args) != 1
                or iterator.keywords
            ):
                raise _Decline
            count = self.expression(iterator.args[0]).integer
            if (
                type(count) is not int
                or not 1 <= count <= 128
                or (depth == 0 and count != 1)
            ):
                raise _Decline
            for iteration in range(count):
                self.values[self.name(statement.target.id)] = self.constant(iteration)
                for child in statement.body:
                    self.statement(child, depth + 1)
            return
        raise _Decline


def deduplicate_bounded_maxima(
    cached: CachedLoops,
    *,
    argument_names: Collection[str],
    constexpr_values: Mapping[str, int],
    rename_groups: Mapping[str, str],
) -> list[ast.stmt]:
    """Replace only proven identical, dominating FP32 max collective calls."""
    if not private_fragment_accesses(cached.body, cached.fragments):
        return cached.body
    try:
        interpreter = _Interpreter(
            cached, argument_names, constexpr_values, rename_groups
        )
        for statement in cached.body:
            interpreter.statement(statement)
            if interpreter.rewrites:
                # A complete one-trip sweep was validated before any edit.
                replacements = interpreter.rewrites

                class Rewrite(ast.NodeTransformer):
                    def visit_Call(
                        self,
                        node: ast.Call,
                        replacements: dict[int, str] = replacements,
                    ) -> ast.AST:
                        name = replacements.get(id(node))
                        if name is not None:
                            return create(ast.Name, id=name, ctx=ast.Load())
                        return self.generic_visit(node)

                return [cast("ast.stmt", Rewrite().visit(item)) for item in cached.body]
    except _Decline:
        return cached.body
    return cached.body
