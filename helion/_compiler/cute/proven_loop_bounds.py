"""Remove integer predicates implied by a complete positive-step loop tile.

The proof uses immutable scalar value snapshots, exact launch dimensions, and
nonnegative signed-int32 intervals. A loop's induction value is at most end-1;
when (end-start) is divisible by its step it is at most end-step. No memory
contents or runtime tensor size hints are proof inputs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import field
import math
from typing import cast

from .simplify_proven_bounds import _assigned_names
from .simplify_proven_bounds import _axis_index
from .simplify_proven_bounds import _call_path
from .simplify_proven_bounds import _checked_interval
from .simplify_proven_bounds import _remove_true_and_operands

_LIMIT = (1 << 31) - 1
_MAX_TERMS = 32


@dataclass(frozen=True)
class _Form:
    constant: int = 0
    terms: tuple[tuple[int, int], ...] = ()

    def add(self, other: _Form, factor: int = 1) -> _Form | None:
        terms = dict(self.terms)
        for symbol, coefficient in other.terms:
            terms[symbol] = terms.get(symbol, 0) + factor * coefficient
        result = tuple(sorted((s, c) for s, c in terms.items() if c))
        if len(result) > _MAX_TERMS:
            return None
        return _Form(self.constant + factor * other.constant, result)

    def scale(self, factor: int) -> _Form:
        return _Form(
            self.constant * factor,
            tuple(
                (symbol, coefficient * factor)
                for symbol, coefficient in self.terms
                if coefficient * factor
            ),
        )

    def divisible(self, divisor: int) -> bool:
        return self.constant % divisor == 0 and all(
            coefficient % divisor == 0 for symbol, coefficient in self.terms
        )


@dataclass(frozen=True)
class _Value:
    low: int
    high: int
    form: _Form


@dataclass
class _Context:
    values: dict[str, _Value] = field(default_factory=dict)
    upper: dict[int, _Form] = field(default_factory=dict)

    def copy(self) -> _Context:
        return _Context(dict(self.values), dict(self.upper))


@dataclass
class _Proof:
    threads: tuple[int, int, int]
    grid: tuple[int, int, int] | None = None
    symbols: dict[object, int] = field(default_factory=dict)
    intervals: dict[int, tuple[int, int]] = field(default_factory=dict)
    changed: int = 0
    ordinal: int = 0

    def symbol(self, key: object, low: int, high: int) -> _Form:
        symbol = self.symbols.setdefault(key, len(self.symbols))
        previous = self.intervals.get(symbol, (low, high))
        self.intervals[symbol] = min(low, previous[0]), max(high, previous[1])
        return _Form(terms=((symbol, 1),))

    def value(
        self, node: ast.expr, context: _Context, budget: int = 128
    ) -> _Value | None:
        if budget <= 0:
            return None
        if isinstance(node, ast.Constant) and type(node.value) is int:
            if 0 <= node.value <= _LIMIT:
                return _Value(node.value, node.value, _Form(node.value))
            return None
        if isinstance(node, ast.Name):
            return context.values.get(node.id)
        axis = _axis_index(node, "thread_idx")
        if axis is not None:
            high = self.threads[axis] - 1
            return _Value(0, high, self.symbol(("thread", axis), 0, high))
        axis = _axis_index(node, "block_idx")
        if axis is not None:
            # Architectural CUDA grid limits; no shape or launch-size hint.
            high = (
                self.grid[axis] - 1
                if self.grid is not None
                else (_LIMIT - 1 if axis == 0 else 65534)
            )
            return _Value(0, high, self.symbol(("block", axis), 0, high))
        if isinstance(node, ast.Call):
            if (
                _call_path(node.func)
                not in {
                    ("cutlass", name) for name in ("Int32", "Int64", "Uint32", "Uint64")
                }
                or len(node.args) != 1
                or node.keywords
            ):
                return None
            return self.value(node.args[0], context, budget - 1)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return self.value(node.operand, context, budget - 1)
        if not isinstance(node, ast.BinOp):
            return None
        left = self.value(node.left, context, budget - 1)
        right = self.value(node.right, context, budget - 1)
        if left is None or right is None:
            return None
        form: _Form | None
        if isinstance(node.op, ast.Add):
            low, high = left.low + right.low, left.high + right.high
            form = left.form.add(right.form)
        elif isinstance(node.op, ast.Sub):
            low, high = left.low - right.high, left.high - right.low
            form = left.form.add(right.form, -1)
        elif isinstance(node.op, ast.Mult):
            low, high = left.low * right.low, left.high * right.high
            if left.low == left.high:
                form = right.form.scale(left.low)
            elif right.low == right.high:
                form = left.form.scale(right.low)
            else:
                form = self.symbol(("mul", left.form, right.form), low, high)
        elif (
            isinstance(node.op, (ast.FloorDiv, ast.Mod)) and right.low == right.high > 0
        ):
            divisor = right.low
            if isinstance(node.op, ast.FloorDiv):
                low, high = left.low // divisor, left.high // divisor
                if left.form.divisible(divisor):
                    form = _Form(
                        left.form.constant // divisor,
                        tuple((s, c // divisor) for s, c in left.form.terms),
                    )
                else:
                    form = self.symbol(("div", left.form, divisor), low, high)
            else:
                modulus = math.gcd(
                    divisor, *(coefficient for symbol, coefficient in left.form.terms)
                )
                low = left.form.constant % modulus
                high = low + (min(left.high, divisor - 1) - low) // modulus * modulus
                form = self.symbol(("mod", left.form, divisor), low, high)
        else:
            return None
        if form is None or _checked_interval(low, high) is None:
            return None
        if low == high:
            form = _Form(low)
        return _Value(low, high, form)

    def minimum(self, form: _Form, context: _Context) -> int | None:
        # Substitute only negative induction coefficients. Each upper bound
        # uses values captured before its loop, so the graph is acyclic.
        for _iteration in range(16):
            selected = next(
                (
                    (symbol, coefficient)
                    for symbol, coefficient in form.terms
                    if coefficient < 0 and symbol in context.upper
                ),
                None,
            )
            if selected is None:
                return form.constant + sum(
                    coefficient * self.intervals[symbol][0 if coefficient >= 0 else 1]
                    for symbol, coefficient in form.terms
                )
            symbol, coefficient = selected
            without = _Form(
                form.constant, tuple((s, c) for s, c in form.terms if s != symbol)
            )
            replacement = without.add(context.upper[symbol], coefficient)
            if replacement is None:
                return None
            form = replacement
        return None

    def true_comparison(self, node: ast.Compare, context: _Context) -> bool:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return False
        left = self.value(node.left, context)
        right = self.value(node.comparators[0], context)
        if left is None or right is None:
            return False
        operation = node.ops[0]
        if isinstance(operation, (ast.Gt, ast.GtE)):
            left, right = right, left
        elif not isinstance(operation, (ast.Lt, ast.LtE)):
            return False
        difference = right.form.add(left.form, -1)
        if difference is None:
            return False
        minimum = self.minimum(difference, context)
        return minimum is not None and minimum >= (
            1 if isinstance(operation, (ast.Lt, ast.Gt)) else 0
        )

    def loop_value(
        self, statement: ast.For, context: _Context
    ) -> tuple[_Value, int, _Form] | None:
        iterator = statement.iter
        if not isinstance(iterator, ast.Call) or _call_path(iterator.func) not in {
            ("range",),
            ("cutlass", "range"),
            ("cutlass", "range_constexpr"),
        }:
            return None
        if not 1 <= len(iterator.args) <= 3:
            return None
        if iterator.keywords and not (
            _call_path(iterator.func) == ("cutlass", "range")
            and all(
                keyword.arg == "unroll"
                and isinstance(keyword.value, ast.Constant)
                and type(keyword.value.value) is int
                and keyword.value.value >= 0
                for keyword in iterator.keywords
            )
        ):
            return None
        args = list(iterator.args)
        if len(args) == 1:
            args.insert(0, ast.Constant(value=0))
        if len(args) == 2:
            args.append(ast.Constant(value=1))
        values = [self.value(argument, context) for argument in args]
        if any(value is None for value in values):
            return None
        start, stop, step = cast("list[_Value]", values)
        if (
            step.low != step.high
            or step.low <= 0
            or stop.high <= start.low
            or stop.high + step.low - 1 > _LIMIT
        ):
            return None
        span = stop.form.add(start.form, -1)
        distance = step.low if span is not None and span.divisible(step.low) else 1
        upper = stop.form.add(_Form(-distance))
        assert upper is not None
        self.ordinal += 1
        form = self.symbol(("loop", self.ordinal), start.low, stop.high - 1)
        symbol = form.terms[0][0]
        return _Value(start.low, stop.high - 1, form), symbol, upper

    def expression(self, expression: ast.expr, context: _Context) -> ast.expr:
        proof = self

        class Rewrite(ast.NodeTransformer):
            def visit_Compare(self, node: ast.Compare) -> ast.AST:
                if proof.true_comparison(node, context):
                    proof.changed += 1
                    return ast.copy_location(ast.Constant(value=True), node)
                return self.generic_visit(node)

            def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
                node = cast("ast.BoolOp", self.generic_visit(node))
                if isinstance(node.op, ast.And):
                    remaining = _remove_true_and_operands(node.values)
                    if len(remaining) != len(node.values):
                        proof.changed += 1
                        if not remaining:
                            return ast.copy_location(ast.Constant(value=True), node)
                        if len(remaining) == 1:
                            return remaining[0]
                        node.values = remaining
                return node

            def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
                node = cast("ast.IfExp", self.generic_visit(node))
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    proof.changed += 1
                    return node.body
                return node

        return cast("ast.expr", Rewrite().visit(expression))

    def block(self, statements: list[ast.stmt], context: _Context) -> list[ast.stmt]:
        output: list[ast.stmt] = []
        for statement in statements:
            if isinstance(statement, ast.Assign):
                statement.value = self.expression(statement.value, context)
                if isinstance(statement.targets[0], ast.Name):
                    target = statement.targets[0].id
                    value = self.value(statement.value, context)
                    context.values.pop(target, None)
                    if value is not None:
                        context.values[target] = value
                else:
                    # Tuple results (for example packed arithmetic) need no
                    # interval inference, but every rebound name must forget
                    # its previous proof before subsequent predicates are read.
                    for target in _assigned_names(statement.targets[0]):
                        context.values.pop(target, None)
            elif isinstance(statement, ast.For):
                induction = self.loop_value(statement, context)
                local = context.copy()
                for name in _assigned_names(statement):
                    local.values.pop(name, None)
                if induction is not None:
                    value, symbol, upper = induction
                    assert isinstance(statement.target, ast.Name)
                    local.values[statement.target.id] = value
                    local.upper[symbol] = upper
                statement.body = self.block(statement.body, local)
                for name in _assigned_names(statement):
                    context.values.pop(name, None)
                statement.orelse = self.block(statement.orelse, context.copy())
            elif isinstance(statement, ast.If):
                statement.test = self.expression(statement.test, context)
                if (
                    isinstance(statement.test, ast.Constant)
                    and statement.test.value is True
                ):
                    self.changed += 1
                    output.extend(self.block(statement.body, context))
                    continue
                statement.body = self.block(statement.body, context.copy())
                statement.orelse = self.block(statement.orelse, context.copy())
                for name in _assigned_names(statement):
                    context.values.pop(name, None)
            elif isinstance(statement, ast.With):
                local = context.copy()
                for item in statement.items:
                    if item.optional_vars is not None:
                        for name in _assigned_names(item.optional_vars):
                            local.values.pop(name, None)
                statement.body = self.block(statement.body, local)
                for name in _assigned_names(statement):
                    context.values.pop(name, None)
            elif isinstance(statement, ast.Expr):
                statement.value = self.expression(statement.value, context)
            output.append(statement)
        return output


def _supported(statements: list[ast.stmt]) -> bool:
    for statement in statements:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1:
                return False
            target = statement.targets[0]
            if not (
                isinstance(target, (ast.Name, ast.Subscript))
                or (
                    isinstance(target, ast.Tuple)
                    and all(isinstance(value, ast.Name) for value in target.elts)
                )
            ):
                return False
        elif isinstance(statement, (ast.If, ast.For)):
            if isinstance(statement, ast.For) and not isinstance(
                statement.target, ast.Name
            ):
                return False
            if not _supported(statement.body) or not _supported(statement.orelse):
                return False
        elif isinstance(statement, ast.With):
            if not _supported(statement.body):
                return False
        elif not isinstance(statement, (ast.Expr, ast.Pass)):
            return False
    return True


def simplify_proven_loop_bounds(
    body: list[ast.stmt],
    *,
    enabled: bool,
    thread_block_dims: tuple[int, int, int] | None,
    constexpr_values: dict[str, int],
    block_grid_dims: tuple[int, int, int] | None = None,
) -> list[ast.stmt]:
    """Keep unknown arithmetic, partial tiles, and all memory-derived bounds."""
    if not enabled or thread_block_dims is None or len(thread_block_dims) != 3:
        return body
    if any(type(size) is not int or not 0 < size <= 1024 for size in thread_block_dims):
        return body
    if thread_block_dims[0] * thread_block_dims[1] * thread_block_dims[2] > 1024:
        return body
    if block_grid_dims is not None and (
        len(block_grid_dims) != 3
        or any(
            type(size) is not int or not 0 < size <= limit
            for size, limit in zip(block_grid_dims, (_LIMIT, 65535, 65535), strict=True)
        )
    ):
        return body
    if any(
        not name.isidentifier() or type(value) is not int or not 0 <= value <= _LIMIT
        for name, value in constexpr_values.items()
    ):
        return body
    module = ast.Module(body=body, type_ignores=[])
    if _assigned_names(module) & (set(constexpr_values) | {"cutlass", "cute", "range"}):
        return body
    if not _supported(body) or any(
        isinstance(
            node,
            (
                ast.NamedExpr,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
            ),
        )
        for node in ast.walk(module)
    ):
        return body
    proof = _Proof(thread_block_dims, block_grid_dims)
    context = _Context(
        {
            name: _Value(value, value, _Form(value))
            for name, value in constexpr_values.items()
        }
    )
    output = proof.block(body, context)
    if proof.changed:
        occupied = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}
        ordinal = 0
        while f"_cute_proven_loop_bounds_{ordinal}_abi_version" in occupied:
            ordinal += 1
        output.insert(
            0,
            ast.Assign(
                targets=[
                    ast.Name(
                        id=f"_cute_proven_loop_bounds_{ordinal}_abi_version",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Constant(value=1),
            ),
        )
        ast.fix_missing_locations(ast.Module(body=output, type_ignores=[]))
    return output


def exact_static_grid(
    expression: ast.AST, constexpr_values: dict[str, int]
) -> tuple[int, int, int] | None:
    """Read a constant grid expression; never evaluate host code or size hints."""
    if not isinstance(expression, ast.Tuple) or not 1 <= len(expression.elts) <= 3:
        return None
    if any(
        isinstance(node, ast.Call)
        and _call_path(node.func)
        not in {("cutlass", name) for name in ("Int32", "Int64", "Uint32", "Uint64")}
        for node in ast.walk(expression)
    ):
        return None
    context = _Context(
        {
            name: _Value(value, value, _Form(value))
            for name, value in constexpr_values.items()
            if type(value) is int and 0 <= value <= _LIMIT
        }
    )
    proof = _Proof((1, 1, 1))
    result = [1, 1, 1]
    for axis, element in enumerate(expression.elts):
        value = proof.value(element, context)
        if (
            value is None
            or value.low != value.high
            or not 0 < value.low <= (_LIMIT if axis == 0 else 65535)
        ):
            return None
        result[axis] = value.low
    return result[0], result[1], result[2]
