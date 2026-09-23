"""Keep CuTe guard preprocessing linear using dominating Boolean type facts.

The SDK duplicates its accumulated left operand when folding a flat ``and``.
Reassociate only pure Boolean operands, preserving their values, evaluation
order and short-circuit boundaries. Definitions supply types, never substituted
expressions. Run after variable renaming so aliases cannot conceal a write.
"""

from __future__ import annotations

import ast
from collections import Counter
import copy
from typing import TypeVar
from typing import cast

from ..ast_extension import ExtendedAST
from .affine_vector_io import _pure_boolean_guard

_AST = TypeVar("_AST", bound=ast.AST)
_ANALYSIS_BUDGET = 32768
_OPAQUE_BINDINGS = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.NamedExpr,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFor,
    ast.Match,
    ast.Yield,
    ast.YieldFrom,
    ast.Return,
    ast.Raise,
    ast.Break,
    ast.Continue,
)


def _copy_node(node: _AST) -> _AST:
    result = (
        cast("_AST", node.copy()) if isinstance(node, ExtendedAST) else copy.copy(node)
    )
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            setattr(result, field, list(value))
    return result


def _single_write_bindings(body: list[ast.stmt]) -> frozenset[str] | None:
    writes: Counter[str] = Counter()
    has_flat_conjunction = False
    for index, node in enumerate(ast.walk(ast.Module(body=body, type_ignores=[]))):
        if index >= _ANALYSIS_BUDGET or isinstance(node, _OPAQUE_BINDINGS):
            return None
        has_flat_conjunction |= (
            isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.And)
            and len(node.values) > 2
        )
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            writes[node.id] += 1
        elif isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            root = node.value
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name):
                writes[root.id] += 1
    if not has_flat_conjunction:
        return None
    return frozenset(name for name, count in writes.items() if count == 1)


class _Reassociate(ast.NodeTransformer):
    def __init__(self, boolean_names: frozenset[str]) -> None:
        self.boolean_names = boolean_names

    def visit(self, node: _AST) -> _AST:
        # Generated ASTs can share nodes across scopes. Own the node and its
        # child lists before visiting, so a local proof never rewrites another
        # occurrence where those Boolean facts do not dominate.
        return super().visit(_copy_node(node))

    def visit_Lambda(self, node: ast.Lambda) -> ast.expr:
        # Reduction callbacks have their own parameter scope. Keep the entire
        # expression, including defaults, unchanged instead of importing facts
        # about identically named outer bindings into that scope.
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        self.generic_visit(node)
        if not (
            isinstance(node.op, ast.And)
            and len(node.values) > 2
            and all(
                _pure_boolean_guard(value, boolean_names=self.boolean_names)
                for value in node.values
            )
        ):
            return node
        result = node.values[-1]
        for value in reversed(node.values[:-1]):
            result = ast.copy_location(
                ast.BoolOp(op=ast.And(), values=[value, result]), node
            )
        return result


def reassociate_boolean_guards(body: list[ast.stmt]) -> list[ast.stmt]:
    """Use definite, single-write Boolean values without expanding definitions.

    Each branch/loop receives its dominating facts, then learns its own local
    definitions in statement order. New facts never escape a conditional or
    potentially zero-trip loop. Counting every syntactic write first excludes
    all rebindings, including conditional/loop-carried and through-value writes.
    Lambdas remain opaque and unchanged; other binding scopes and oversized
    inputs decline before any mutation.
    """
    single_write = _single_write_bindings(body)
    if single_write is None:
        return body

    def visit(statements: list[ast.stmt], incoming: frozenset[str]) -> None:
        boolean_names = set(incoming)
        for index, original in enumerate(statements):
            statement = _copy_node(original)
            statements[index] = statement
            rewriter = _Reassociate(frozenset(boolean_names))
            if isinstance(statement, ast.Assign):
                statement.value = rewriter.visit(statement.value)
                if (
                    len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id in single_write
                    and _pure_boolean_guard(
                        statement.value, boolean_names=frozenset(boolean_names)
                    )
                ):
                    boolean_names.add(statement.targets[0].id)
            elif isinstance(statement, ast.If):
                statement.test = rewriter.visit(statement.test)
                visit(statement.body, frozenset(boolean_names))
                visit(statement.orelse, frozenset(boolean_names))
            elif isinstance(statement, (ast.For, ast.While)):
                if isinstance(statement, ast.For):
                    statement.iter = rewriter.visit(statement.iter)
                else:
                    statement.test = rewriter.visit(statement.test)
                visit(statement.body, frozenset(boolean_names))
                visit(statement.orelse, frozenset(boolean_names))
            elif isinstance(statement, ast.Expr):
                statement.value = rewriter.visit(statement.value)

    visit(body, frozenset())
    return body
