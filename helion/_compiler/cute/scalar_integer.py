"""Integer-kind proofs for scalar expressions in generated CuTe code."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet


_INTEGER_CASTS = frozenset(
    f"cutlass.{name}" for name in ("Int32", "Int64", "Uint32", "Uint64")
)


def _integer_expression(
    node: ast.expr,
    lane: str,
    known: AbstractSet[str],
    *,
    integer_tensors: AbstractSet[str] = frozenset(),
) -> bool:
    # Tensor facts are for kind-only analysis of an existing value. They do
    # not prove that re-evaluating a load is safe; replay proofs leave them empty.
    if integer_tensors and isinstance(node, ast.IfExp):
        # A predicated load retains integer kind when both possible values
        # are integral. The original predicate and its evaluation stay in place.
        return all(
            _integer_expression(value, lane, known, integer_tensors=integer_tensors)
            for value in (node.body, node.orelse)
        )
    if (
        integer_tensors
        and isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and not node.args
        and not node.keywords
    ):
        pointer = node.func.value
        while isinstance(pointer, ast.BinOp) and isinstance(pointer.op, ast.Add):
            pointer = pointer.left
        return (
            isinstance(pointer, ast.Attribute)
            and pointer.attr == "iterator"
            and isinstance(pointer.value, ast.Name)
            and pointer.value.id in integer_tensors
        )
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
            node.args[0], lane, known, integer_tensors=integer_tensors
        )
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub, ast.Invert)
    ):
        return _integer_expression(
            node.operand, lane, known, integer_tensors=integer_tensors
        )
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
        return _integer_expression(
            node.left, lane, known, integer_tensors=integer_tensors
        ) and _integer_expression(
            node.right, lane, known, integer_tensors=integer_tensors
        )
    return False
