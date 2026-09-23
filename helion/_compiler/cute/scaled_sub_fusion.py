"""Make the existing distributed-scale FP32 contraction choice explicit.

The reciprocal/scale hoister already changes ``(a - b) * c`` into the
FMA-friendly ``a * c - scaled_b``. Leaving the latter as ordinary arithmetic
makes ptxas contraction depend on intervening masked loads or register reuse.
Only nodes marked by that existing transformation are handled here, under
the existing non-matmul ``fuse_fma`` policy (which permits contraction with
either fast-math setting). No division, approximation, integer arithmetic,
unmarked product, or explicit user FMA is changed.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Literal

from ..ast_extension import create
from ..ast_extension import expr_from_string
from .scalar_recipe_rounding import _kind

if TYPE_CHECKING:
    from collections.abc import Mapping

SCALED_SUBTRACTION_ATTR = "_helion_cute_distributed_scale_subtraction"
_Kind = Literal["fp32", "literal"] | None


def _simple_operand(node: ast.expr) -> bool:
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cutlass"
        and node.func.attr == "Float32"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], (ast.Name, ast.Constant))
    )


class _Contraction:
    def __init__(self, rename_groups: Mapping[str, str]) -> None:
        self.rename_groups = rename_groups

    def canonical(self, name: str) -> str:
        return self.rename_groups.get(name, name)

    def writes(self, node: ast.AST) -> set[str]:
        return {
            self.canonical(item.id)
            for item in ast.walk(node)
            if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del))
        }

    def kind(self, expression: ast.expr, types: Mapping[str, _Kind]) -> _Kind:
        names = {
            node.id: types.get(self.canonical(node.id))
            for node in ast.walk(expression)
            if isinstance(node, ast.Name)
        }
        return _kind(expression, names)

    def expression(self, expression: ast.expr, types: Mapping[str, _Kind]) -> ast.expr:
        owner = self

        class Rewrite(ast.NodeTransformer):
            def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
                if not (
                    getattr(node, SCALED_SUBTRACTION_ATTR, False)
                    and isinstance(node.op, ast.Sub)
                    and owner.kind(node, types) == "fp32"
                ):
                    return self.generic_visit(node)
                if isinstance(node.left, ast.BinOp) and isinstance(
                    node.left.op, ast.Mult
                ):
                    product, addend, subtract_product = node.left, node.right, False
                elif isinstance(node.right, ast.BinOp) and isinstance(
                    node.right.op, ast.Mult
                ):
                    product, addend, subtract_product = node.right, node.left, True
                else:
                    return self.generic_visit(node)
                if (
                    owner.kind(product, types) != "fp32"
                    or owner.kind(addend, types) != "fp32"
                ):
                    return self.generic_visit(node)
                left, right = product.left, product.right
                if not (
                    _simple_operand(left)
                    and isinstance(right, ast.Constant)
                    and type(right.value) is float
                    and isinstance(addend, ast.Name)
                ):
                    return self.generic_visit(node)
                if subtract_product:
                    left = create(ast.UnaryOp, op=ast.USub(), operand=left)
                else:
                    addend = create(ast.UnaryOp, op=ast.USub(), operand=addend)
                result = expr_from_string(
                    "cute.math.fma({a}, {b}, {c})", a=left, b=right, c=addend
                )
                assert isinstance(result, ast.expr)
                return ast.copy_location(result, node)

        result = Rewrite().visit(expression)
        assert isinstance(result, ast.expr)
        return result

    @staticmethod
    def intersection(
        left: Mapping[str, _Kind], right: Mapping[str, _Kind]
    ) -> dict[str, _Kind]:
        return {
            name: kind
            for name, kind in left.items()
            if kind is not None and right.get(name) == kind
        }

    def body(
        self, body: list[ast.stmt], incoming: Mapping[str, _Kind]
    ) -> dict[str, _Kind]:
        types = dict(incoming)
        for statement in body:
            if isinstance(statement, ast.Assign):
                kind = self.kind(statement.value, types)
                statement.value = self.expression(statement.value, types)
                written = self.writes(statement)
                for name in written:
                    types.pop(name, None)
                if len(statement.targets) == 1 and isinstance(
                    statement.targets[0], ast.Name
                ):
                    types[self.canonical(statement.targets[0].id)] = kind
            elif isinstance(statement, ast.If):
                statement.test = self.expression(statement.test, types)
                yes = self.body(statement.body, types)
                no = self.body(statement.orelse, types)
                types = self.intersection(yes, no)
            elif isinstance(statement, ast.For) and not statement.orelse:
                written = self.writes(statement)
                # A repeated body cannot inherit a prior iteration's type
                # from an initializer. Re-establish modified names only from
                # definite local definitions, and retain the zero-trip path.
                entry = {
                    name: kind for name, kind in types.items() if name not in written
                }
                if any(
                    isinstance(node, (ast.Break, ast.Continue, ast.Return, ast.Raise))
                    for node in ast.walk(statement)
                ):
                    for name in written:
                        types.pop(name, None)
                    continue
                after = self.body(statement.body, entry)
                for name in written:
                    if types.get(name) != after.get(name):
                        types.pop(name, None)
            elif isinstance(statement, ast.Expr):
                statement.value = self.expression(statement.value, types)
            elif isinstance(statement, ast.Pass):
                continue
            else:
                # Opaque control flow contributes no reaching type facts and
                # keeps its original arithmetic. The caller's ordinary FMA
                # pass remains responsible for its pre-existing patterns.
                types.clear()
        return types


def contract_distributed_scale_subtractions(
    body: list[ast.stmt], rename_groups: Mapping[str, str]
) -> list[ast.stmt]:
    """Contract only marked FP32 expressions with definite reaching types."""
    marked = False
    for index, node in enumerate(
        node for statement in body for node in ast.walk(statement)
    ):
        if index >= 32768 or isinstance(node, (ast.NamedExpr, ast.Lambda)):
            return body
        marked |= bool(getattr(node, SCALED_SUBTRACTION_ATTR, False))
    if not marked:
        return body
    _Contraction(rename_groups).body(body, {})
    return body
