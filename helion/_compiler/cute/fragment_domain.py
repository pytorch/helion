"""Coordinate provenance for computed-fragment ownership predicates.

Rendered predicates can be equal after broadcasting while referring to different
axes. Keep those identities separate until a contraction projects its free axes.
The expressions remain ordinary scalar predicates at the code-generation edge.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import itertools
from typing import TYPE_CHECKING
from typing import cast

from ... import exc
from .proven_loop_bounds import _Context
from .proven_loop_bounds import _Proof
from .proven_loop_bounds import _Value

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping


@dataclass(frozen=True)
class BroadcastAxis:
    name: str


CoordinateOrigin = str | BroadcastAxis


@dataclass(frozen=True)
class OwnershipTerm:
    expression: str
    coordinates: frozenset[CoordinateOrigin]


OwnershipMask = str | OwnershipTerm
_symbol_ids = itertools.count()


def coordinate_names(expression: str) -> frozenset[CoordinateOrigin]:
    names: set[CoordinateOrigin] = set()

    class Collect(ast.NodeVisitor):
        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
            # Proof-only broadcast aliases are distinct from every ordinary
            # variable, even a caller variable with the exact same spelling.
            assert isinstance(node.target, ast.Name)
            names.add(BroadcastAxis(node.target.id))

        def visit_Name(self, node: ast.Name) -> None:
            names.add(node.id)

    Collect().visit(ast.parse(expression, mode="eval"))
    return frozenset(names)


def ownership_terms(masks: Iterable[OwnershipMask]) -> tuple[OwnershipTerm, ...]:
    result = []
    for mask in masks:
        if isinstance(mask, OwnershipTerm):
            result.append(mask)
            continue
        pending = [ast.parse(mask, mode="eval").body]
        while pending:
            node = pending.pop()
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                pending.extend(reversed(node.values))
            else:
                expression = ast.unparse(node)
                result.append(OwnershipTerm(expression, coordinate_names(expression)))
    return tuple(dict.fromkeys(result))


def render_ownership(mask: OwnershipMask) -> str:
    return mask.expression if isinstance(mask, OwnershipTerm) else mask


def rewrite_ownership(
    term: OwnershipTerm,
    values: Mapping[CoordinateOrigin, str],
    coordinates: Mapping[CoordinateOrigin, str] | None = None,
) -> OwnershipTerm:
    """Rewrite rendered coordinates and, separately, their ownership origins."""
    if coordinates is None:
        coordinates = values
    replacements = {
        name: ast.parse(value, mode="eval").body for name, value in values.items()
    }

    class Rewrite(ast.NodeTransformer):
        def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.expr:
            assert isinstance(node.target, ast.Name)
            origin = BroadcastAxis(node.target.id)
            return copy.deepcopy(replacements.get(origin, node))

        def visit_Name(self, node: ast.Name) -> ast.expr:
            return copy.deepcopy(replacements.get(node.id, node))

    expression = Rewrite().visit(ast.parse(term.expression, mode="eval").body)
    dependencies: set[CoordinateOrigin] = set()
    for name in term.coordinates:
        dependencies.update(
            coordinate_names(coordinates[name]) if name in coordinates else (name,)
        )
    return OwnershipTerm(ast.unparse(expression), frozenset(dependencies))


def broadcast_ownership(
    domain: Callable[[tuple[str, ...]], tuple[OwnershipTerm, ...]],
    shape: tuple[int, ...],
    indices: tuple[str, ...],
) -> tuple[OwnershipTerm, ...]:
    """Fold singleton values without erasing which caller axis they came from."""
    indices = indices[-len(shape) :] if shape else ()
    # A synthetic named-expression node keeps an axis token distinct from a
    # real variable of the same spelling. These nodes are never evaluated:
    # every token is substituted out of both fields before returning.
    aliases = tuple(
        BroadcastAxis(f"__helion_ownership_axis_{next(_symbol_ids)}") for _ in shape
    )
    values: dict[CoordinateOrigin, str] = {
        name: "0" if size == 1 else index
        for name, size, index in zip(aliases, shape, indices, strict=True)
    }
    coordinates: dict[CoordinateOrigin, str] = dict(zip(aliases, indices, strict=True))
    return tuple(
        rewrite_ownership(term, values, coordinates)
        for term in domain(tuple(f"({alias.name} := 0)" for alias in aliases))
    )


def project_contracted_ownership(
    terms: tuple[OwnershipTerm, ...],
    sizes: Mapping[str, int],
    contracted: str,
) -> tuple[OwnershipTerm, ...]:
    """Project a rectangular K domain using one common monotone endpoint.

    A reshape can combine free and contracted coordinates in one predicate.
    Keep that predicate's existential meaning rather than discarding it. The
    bounded integer proof rejects opaque arithmetic, wrapping and predicates
    needing different K witnesses. An empty contraction still owns its output.
    """
    free = set(sizes) - {contracted}
    retained = tuple(
        term
        for term in terms
        if contracted not in term.coordinates and term.coordinates & free
    )
    mixed = tuple(
        term
        for term in terms
        if contracted in term.coordinates and term.coordinates & free
    )
    if not mixed or sizes[contracted] == 0:
        return retained
    if any(size > (1 << 31) - 1 for size in sizes.values()):
        raise exc.InvalidConfig("mixed fragment ownership exceeds bounded coordinates")

    # Ownership supplies coordinate extents, not a CUDA launch shape.
    proof = _Proof(None)
    context = _Context()
    symbols: dict[str, int] = {}
    for name, size in sizes.items():
        high = max(size - 1, 0)
        form = proof.symbol(("ownership", name), 0, high)
        context.values[name] = _Value(0, high, form)
        symbols[name] = form.terms[0][0]
    base_symbols = set(symbols.values())

    class Simplify(ast.NodeTransformer):
        def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
            node = cast("ast.BinOp", self.generic_visit(node))
            if isinstance(node.right, ast.Constant) and type(node.right.value) is int:
                divisor = node.right.value
                if isinstance(node.op, ast.FloorDiv) and divisor == 1:
                    return node.left
                if isinstance(node.op, ast.Mod) and divisor > 0:
                    value = proof.value(node.left, context)
                    if value is not None and 0 <= value.low <= value.high < divisor:
                        return node.left
            return node

    opposites: dict[type[ast.cmpop], type[ast.cmpop]] = {
        ast.Lt: ast.GtE,
        ast.LtE: ast.Gt,
        ast.Gt: ast.LtE,
        ast.GtE: ast.Lt,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
    }

    def truth(node: ast.expr) -> bool | None:
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return node.value
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            return None
        if proof.true_comparison(node, context):
            return True
        opposite = opposites.get(type(node.ops[0]))
        if opposite is not None and proof.true_comparison(
            ast.Compare(node.left, [opposite()], node.comparators), context
        ):
            return False
        return None

    predicates = [
        (term, Simplify().visit(ast.parse(term.expression, mode="eval").body))
        for term in mixed
    ]
    predicates = [(term, node) for term, node in predicates if truth(node) is not True]
    if not predicates:
        return retained

    # K-only conditions must describe either this whole physical interval or
    # no K at all. A partial/unknown interval needs a different projection.
    contraction = [
        truth(Simplify().visit(ast.parse(term.expression, mode="eval").body))
        for term in terms
        if contracted in term.coordinates and not term.coordinates & free
    ]
    if False in contraction:
        return retained
    if None in contraction:
        raise exc.InvalidConfig("mixed fragment ownership needs a proved K interval")

    endpoint: int | None = None
    projected = []
    for term, node in predicates:
        if truth(node) is False:
            projected.append(OwnershipTerm("False", term.coordinates - {contracted}))
            continue
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            raise exc.InvalidConfig("mixed fragment ownership is not affine")
        left = proof.value(node.left, context)
        right = proof.value(node.comparators[0], context)
        difference = left.form.add(right.form, -1) if left and right else None
        if difference is None or any(
            symbol not in base_symbols for symbol, _ in difference.terms
        ):
            raise exc.InvalidConfig("mixed fragment ownership is not bounded affine")
        coefficient = dict(difference.terms).get(symbols[contracted], 0)
        if coefficient:
            if isinstance(node.ops[0], (ast.Gt, ast.GtE)):
                coefficient = -coefficient
            elif not isinstance(node.ops[0], (ast.Lt, ast.LtE)):
                raise exc.InvalidConfig("mixed fragment ownership is not monotone")
            candidate = 0 if coefficient > 0 else sizes[contracted] - 1
            if endpoint is not None and endpoint != candidate:
                raise exc.InvalidConfig(
                    "mixed fragment ownership has no common K endpoint"
                )
            endpoint = candidate
        projected.append(OwnershipTerm(ast.unparse(node), term.coordinates))
    return (
        *retained,
        *(
            rewrite_ownership(term, {contracted: str(endpoint or 0)})
            for term in projected
        ),
    )
