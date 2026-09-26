"""Integer-kind proofs for scalar expressions in generated CuTe code."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet


_INTEGER_CASTS = frozenset(
    f"cutlass.{name}" for name in ("Int32", "Int64", "Uint32", "Uint64")
)


def _integer_expression(node: ast.expr, lane: str, known: AbstractSet[str]) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, ast.Name):
        return node.id == lane or node.id in known
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and type(node.slice.value) is int
    ):
        value = node.value
        if (
            isinstance(value, ast.Call)
            and ast.unparse(value.func)
            in {"cute.arch.block_idx", "cute.arch.thread_idx"}
            and not value.args
            and not value.keywords
        ):
            return 0 <= node.slice.value < 3
        return (
            isinstance(value, ast.Attribute)
            and value.attr == "stride"
            and isinstance(value.value, ast.Attribute)
            and value.value.attr == "layout"
            and isinstance(value.value.value, ast.Name)
            and node.slice.value >= 0
        )
    if (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) in _INTEGER_CASTS
        and len(node.args) == 1
        and not node.keywords
    ):
        # A direct scalar conversion is exact replay. Arithmetic before the
        # cast must already be integral: float rounding can destroy adjacency
        # or change a rematerialized predicate through contraction.
        return isinstance(node.args[0], ast.Name) or _integer_expression(
            node.args[0], lane, known
        )
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub, ast.Invert)
    ):
        return _integer_expression(node.operand, lane, known)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.FloorDiv,
            ast.Mod,
            ast.LShift,
            ast.RShift,
            ast.BitAnd,
            ast.BitOr,
            ast.BitXor,
        ),
    ):
        return _integer_expression(node.left, lane, known) and _integer_expression(
            node.right, lane, known
        )
    return False
