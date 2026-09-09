"""Small static cost model shared by CuTe loop-invariant-code motion passes."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


# Moving a scalar result across a loop extends its live range.  Require enough
# statically visible work to pay for that pressure, as well as a small absolute
# floor so a two-iteration loop containing only casts is left alone.  These are
# instruction-cost units, not shape- or kernel-specific thresholds.
_MIN_SAVED_WORK = 8
_LIVE_SCALAR_COST = 4


def _call_path(node: ast.expr) -> str | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _constexpr_extent(loop: ast.For) -> int | None:
    call = loop.iter
    if (
        not isinstance(call, ast.Call)
        or _call_path(call.func) not in {"range", "cutlass.range_constexpr"}
        or len(call.args) != 1
        or call.keywords
        or not isinstance(call.args[0], ast.Constant)
        or not isinstance(call.args[0].value, int)
        or isinstance(call.args[0].value, bool)
        or call.args[0].value <= 0
    ):
        return None
    return call.args[0].value


def _expression_work(expression: ast.AST) -> int:
    """Estimate scalar instruction work while avoiding target-specific tables."""
    work = 0
    for node in ast.walk(expression):
        if isinstance(node, ast.Call):
            path = _call_path(node.func)
            # A warp reduction expands to multiple shuffle/combine operations.
            # Transcendentals and memory operations also deserve more weight
            # than a scalar cast or ordinary helper call.
            if (
                path is not None and path.startswith("cute.arch.warp_reduction_")
            ) or path in {
                "cute.arch.load",
                "cute.math.exp",
                "cute.math.exp2",
                "cute.math.log",
                "cute.math.log2",
                "cute.math.rcp",
                "cute.math.rsqrt",
                "cute.math.sqrt",
            }:
                work += 4
            else:
                work += 1
        elif isinstance(
            node,
            (
                ast.BinOp,
                ast.BoolOp,
                ast.Compare,
                ast.IfExp,
                ast.Subscript,
                ast.UnaryOp,
            ),
        ):
            work += 1
    return work


def _statement_work(statement: ast.stmt) -> int:
    if isinstance(statement, ast.For):
        extent = _constexpr_extent(statement)
        if extent is None:
            return 0
        return _expression_work(statement.iter) + extent * sum(
            _statement_work(child) for child in statement.body
        )
    if isinstance(statement, ast.If):
        return _expression_work(statement.test) + max(
            sum(_statement_work(child) for child in statement.body),
            sum(_statement_work(child) for child in statement.orelse),
        )
    if isinstance(statement, ast.Assign):
        return _expression_work(statement.value)
    if isinstance(statement, ast.AnnAssign) and statement.value is not None:
        return _expression_work(statement.value)
    if isinstance(statement, ast.AugAssign):
        return _expression_work(statement.value) + 1
    return 0


def repeated_work_is_profitable(
    statements: Iterable[ast.stmt],
    *,
    repeat_extent: int,
    live_scalars: int = 1,
    fixed_work: int = 0,
) -> bool:
    """Return whether statically saved work pays for longer scalar liveness.

    Safety and invariance are proved by the calling transformation.  This
    helper only replaces fixed outer-loop extent gates with a structural cost
    test: at least two iterations, statically countable work, and enough work
    removed to cover both a small absolute floor and the number of results
    kept live across the remaining loop.
    """
    if repeat_extent <= 1 or live_scalars < 0 or fixed_work < 0:
        return False
    work = sum(_statement_work(statement) for statement in statements)
    saved_work = (repeat_extent - 1) * work
    return saved_work >= (
        max(_MIN_SAVED_WORK, live_scalars * _LIVE_SCALAR_COST) + fixed_work
    )
