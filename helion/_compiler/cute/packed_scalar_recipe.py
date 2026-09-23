"""Pack independent, explicitly rounded scalar products without reassociation."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import expr_from_string
from .scalar_recipe import _MATH_CALLS
from .scalar_recipe import _clone
from .scalar_recipe import _path
from .scalar_recipe import _read_names
from .scalar_recipe_rounding import is_rounded_fp32_multiply

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

_CASTS = {
    "BFloat16",
    "Boolean",
    "Float16",
    "Float32",
    "Float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Uint8",
    "Uint16",
    "Uint32",
    "Uint64",
}
_OPERATORS = {
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "and_",
    "or_",
    "xor",
    "lshift",
    "rshift",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "neg",
    "pos",
    "invert",
    "not_",
}


def _expr(source: str) -> ast.expr:
    return cast("ast.expr", expr_from_string(source))


def _pure(expression: ast.expr) -> bool:
    if isinstance(expression, ast.Name):
        return True
    if isinstance(expression, ast.Constant):
        return type(expression.value) in (bool, int, float)
    if isinstance(expression, ast.Subscript):
        return isinstance(expression.value, ast.Name) and _pure(expression.slice)
    if isinstance(expression, ast.BinOp):
        return _pure(expression.left) and _pure(expression.right)
    if isinstance(expression, ast.UnaryOp):
        return _pure(expression.operand)
    if isinstance(expression, ast.BoolOp):
        return all(_pure(value) for value in expression.values)
    if isinstance(expression, ast.Compare):
        return _pure(expression.left) and all(
            _pure(value) for value in expression.comparators
        )
    if isinstance(expression, ast.IfExp):
        return all(
            _pure(value)
            for value in (expression.test, expression.body, expression.orelse)
        )
    if not isinstance(expression, ast.Call):
        return False
    if is_rounded_fp32_multiply(expression):
        return all(_pure(value) for value in cast("ast.Tuple", expression.args[0]).elts)
    path = _path(expression.func)
    if path is None:
        return False
    allowed = (
        len(path) == 2
        and (
            (path[0] == "cutlass" and path[1] in _CASTS)
            or (path[0] == "operator" and path[1] in _OPERATORS)
        )
    ) or (len(path) == 3 and path[:2] == ("cute", "math") and path[2] in _MATH_CALLS)
    return (
        allowed
        and all(_pure(argument) for argument in expression.args)
        and all(
            keyword.arg is not None and isinstance(keyword.value, ast.Constant)
            for keyword in expression.keywords
        )
    )


class _Lane(ast.NodeTransformer):
    def __init__(self, lane: str, index: ast.expr, names: dict[str, str]) -> None:
        self.lane = lane
        self.index = index
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id == self.lane and isinstance(node.ctx, ast.Load):
            return _clone(self.index)
        if node.id in self.names:
            node.id = self.names[node.id]
        return node


def rounded_recipe_loop(
    statements: Sequence[ast.stmt],
    *,
    lane: str,
    width: int,
    register_outputs: frozenset[str],
    fresh_name: Callable[[str], str],
    target_device_capability: tuple[int, int] | None,
) -> ast.For:
    """Pair only RN products in an independently owned register packet.

    The caller supplies fresh, unaliased register outputs of exactly ``width``
    elements, and scalar recipe locals that are dead outside this loop. Inputs
    are read-only for the recipe. This helper additionally requires straight
    SSA definitions, pure scalar expressions, direct lane stores, and no read
    of any output. It never changes a lane's arithmetic DAG, fuses operations,
    pairs a reduction, or changes an existing narrowing/rounding boundary.
    """
    original = ast.For(
        ast.Name(lane, ast.Store()),
        _expr(f"cutlass.range_constexpr({width})"),
        list(statements),
        [],
    )
    if (
        target_device_capability is None
        or target_device_capability < (10, 0)
        or width < 2
        or width % 2
        or not register_outputs
    ):
        return original
    definitions: dict[str, ast.expr] = {}
    stores: set[str] = set()
    rounded: set[str] = set()
    for statement in statements:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return original
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            if (
                target.id in definitions
                or target.id in register_outputs
                or target.id == lane
            ):
                return original
            definitions[target.id] = statement.value
            if is_rounded_fp32_multiply(statement.value):
                rounded.add(target.id)
        elif (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id in register_outputs
            and isinstance(target.slice, ast.Name)
            and target.slice.id == lane
        ):
            if target.value.id in stores:
                return original
            stores.add(target.value.id)
        else:
            return original
        if (
            not _pure(statement.value)
            or _read_names(statement.value) & register_outputs
        ):
            return original
    if not rounded or stores != register_outputs:
        return original
    seen: set[str] = set()
    for statement in statements:
        assert isinstance(statement, ast.Assign)
        if _read_names(statement.value) & (definitions.keys() - seen):
            return original
        if isinstance(statement.targets[0], ast.Name):
            seen.add(statement.targets[0].id)

    pair = fresh_name("rounded_pair")
    lo_names = {name: fresh_name("rounded_lo") for name in definitions}
    hi_names = {name: fresh_name("rounded_hi") for name in definitions}
    lo = _Lane(lane, _expr(f"{pair} * 2"), lo_names)
    hi = _Lane(lane, _expr(f"{pair} * 2 + 1"), hi_names)
    body: list[ast.stmt] = []
    for statement in statements:
        assert isinstance(statement, ast.Assign)
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in rounded:
            call = cast("ast.Call", statement.value)
            operands = cast("ast.Tuple", call.args[0]).elts
            body.append(
                ast.Assign(
                    [
                        ast.Tuple(
                            [
                                ast.Name(lo_names[target.id], ast.Store()),
                                ast.Name(hi_names[target.id], ast.Store()),
                            ],
                            ast.Store(),
                        )
                    ],
                    ast.Call(
                        _expr("cute.arch.mul_packed_f32x2"),
                        [
                            ast.Tuple(
                                [lo.visit(_clone(operand)), hi.visit(_clone(operand))],
                                ast.Load(),
                            )
                            for operand in operands
                        ],
                        [
                            ast.keyword("rnd", ast.Constant("rn")),
                            ast.keyword("ftz", ast.Constant(False)),
                        ],
                    ),
                )
            )
        else:
            body.extend(
                (
                    cast("ast.stmt", lo.visit(_clone(statement))),
                    cast("ast.stmt", hi.visit(_clone(statement))),
                )
            )
    return ast.For(
        ast.Name(pair, ast.Store()),
        _expr(f"cutlass.range_constexpr({width // 2})"),
        body,
        [],
    )
