"""Scalar temporaries local to one evaluation of a lazy fragment expression."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..generate_ast import GenerateAST


class FragmentExpression:
    """Hash-cons pure integer coordinates and memoize typed producer scalars.

    The caller creates a fresh instance for each outermost pointwise read in a
    single emitted lexical body. Nothing survives a completed read, a store, or
    a transition into another body. Shared/global loads are not memoized here.
    """

    def __init__(self, cg: GenerateAST) -> None:
        self.cg = cg
        self.body = cg.statements_stack[-1]
        self.keys: dict[tuple[object, ...], int] = {}
        self.aliases: dict[str, int] = {}
        self.coordinates: dict[int, str] = {}
        self.values: dict[tuple[object, tuple[int, ...]], str] = {}

    def intern(self, key: tuple[object, ...]) -> int:
        if key not in self.keys:
            self.keys[key] = len(self.keys)
        return self.keys[key]

    def coordinate_key(self, node: ast.AST) -> int:
        if isinstance(node, ast.Name) and node.id in self.aliases:
            return self.aliases[node.id]
        if isinstance(node, ast.BinOp):
            left, right = (
                self.coordinate_key(node.left),
                self.coordinate_key(node.right),
            )
            zero, one = self.intern(("Constant", 0)), self.intern(("Constant", 1))
            # Coordinates are pure integers. Preserve their arithmetic, casts,
            # signedness and order except for these exact neutral identities.
            if (isinstance(node.op, (ast.FloorDiv, ast.Mult)) and right == one) or (
                isinstance(node.op, (ast.Add, ast.Sub)) and right == zero
            ):
                return left
            if isinstance(node.op, ast.Mult) and left == one:
                return right
            if isinstance(node.op, ast.Add) and left == zero:
                return right
            if isinstance(node.op, ast.Mod) and right == one:
                return zero
            return self.intern(("BinOp", type(node.op).__name__, left, right))
        if isinstance(node, ast.Constant):
            return self.intern(("Constant", node.value))
        # Intern child IDs, not expanded child trees: keys stay linear in the
        # scalar coordinate DAG even when the same expression has many users.
        fields: list[object] = [type(node).__name__]
        for name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                value = self.coordinate_key(value)
            elif isinstance(value, list):
                value = tuple(
                    self.coordinate_key(item) if isinstance(item, ast.AST) else item
                    for item in value
                )
            fields.extend((name, value))
        return self.intern(tuple(fields))

    def coordinate(self, text: str) -> str:
        expression = expr_from_string(text)
        key = self.coordinate_key(expression)
        if key not in self.coordinates:
            if isinstance(expression, (ast.Name, ast.Constant)):
                result = text
            else:
                result = self.cg.lift(expression, prefix="fragment_coordinate").id
                self.aliases[result] = key
            self.coordinates[key] = result
        return self.coordinates[key]

    def read(
        self,
        producer: object,
        coordinates: tuple[str, ...],
        evaluate: Callable[[], str],
    ) -> str:
        key = (
            producer,
            tuple(self.coordinate_key(expr_from_string(x)) for x in coordinates),
        )
        if key not in self.values:
            # evaluate() returns the producer's logical dtype cast. Name that
            # cast as well, preserving rounding/truncation at every DSL node.
            result = self.cg.lift(
                expr_from_string(evaluate()), prefix="fragment_value"
            ).id
            self.values[key] = result
        return self.values[key]
