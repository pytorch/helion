"""Conditional rectangular raw-leaf coordinates, without scheduling or emission."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .chained_vector_leaf import _DTYPES
from .chained_vector_leaf import _INTEGER_CASTS
from .chained_vector_leaf import _expand
from .chained_vector_leaf import _NotProven

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet

    import torch


_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1
_I64_MAX = (1 << 63) - 1
_RELATIONS = {
    "operator.eq": "eq",
    "operator.ne": "ne",
    "operator.lt": "lt",
    "operator.le": "le",
    "operator.gt": "gt",
    "operator.ge": "ge",
}
_NEGATE = {"eq": "ne", "ne": "eq", "lt": "ge", "le": "gt", "gt": "le", "ge": "lt"}


@dataclass(frozen=True)
class RectangularLeafPlan:
    """A guarded exact source rectangle, not authority to issue a transaction.

    ``origin_indices`` retain the original typed index expressions at (0,0).
    ``base`` and ``origin`` use safe widened guard arithmetic; they agree with
    those exact indices only when ``guard`` is true. Origins are element
    coordinates, not tile-grid indices. The caller owns descriptor legality,
    pointer alignment, uniform ownership, alias/lifetime and scalar fallback.
    """

    indices: tuple[str, ...]
    mask: str | None
    row: str
    column: str
    tile_shape: tuple[int, int]
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    element_bytes: int
    pitch: int
    view_shape: tuple[int, int]
    origin_indices: tuple[str, ...]
    base: str
    origin: tuple[str, str]
    guard: str


@dataclass(frozen=True)
class _Affine:
    row: int = 0
    column: int = 0
    constant: int = 0
    atoms: tuple[tuple[str, int], ...] = ()

    def scaled(self, factor: int) -> _Affine:
        return _Affine(
            self.row * factor,
            self.column * factor,
            self.constant * factor,
            tuple((atom, coefficient * factor) for atom, coefficient in self.atoms),
        )

    def plus(self, other: _Affine) -> _Affine:
        atoms = dict(self.atoms)
        for atom, coefficient in other.atoms:
            atoms[atom] = atoms.get(atom, 0) + coefficient
        return _Affine(
            self.row + other.row,
            self.column + other.column,
            self.constant + other.constant,
            tuple(
                (atom, coefficient)
                for atom, coefficient in atoms.items()
                if coefficient
            ),
        )


def _literal(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal(node.operand)
        if value is not None:
            return -value if isinstance(node.op, ast.USub) else value
    return None


def _integer(node: ast.expr) -> bool:
    """Admit defined opaque integer syntax; never simplify its typed arithmetic."""
    if isinstance(node, ast.Constant):
        return type(node.value) is int and abs(node.value) <= _I64_MAX
    if isinstance(node, ast.Name):
        return True  # Expansion already checked explicit caller ownership.
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)) and _integer(
            node.operand
        )
    if isinstance(node, ast.Call):
        return ast.unparse(node.func) in _INTEGER_CASTS and _integer(node.args[0])
    if not isinstance(node, ast.BinOp) or not (
        _integer(node.left) and _integer(node.right)
    ):
        return False
    if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
        divisor = _literal(node.right)
        return divisor is not None and 0 < divisor <= _I32_MAX
    if isinstance(node.op, (ast.LShift, ast.RShift)):
        shift = _literal(node.right)
        return shift is not None and 0 <= shift < 32
    return isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
    )


def _join(parts: Sequence[str], operator: str = "&") -> str:
    return f" {operator} ".join(f"({part})" for part in parts) if parts else "True"


class _Proof:
    def __init__(self, row: str, column: str, tile: tuple[int, int]) -> None:
        self.row, self.column, self.tile = row, column, tile
        self.atoms: dict[str, None] = {}
        self.intermediates: list[_Affine] = []

    def bounded(self, value: _Affine) -> _Affine:
        # This L1 envelope bounds every generated Int64 product and partial
        # sum, including guard-failure values, without short-circuit reliance.
        norm = (
            abs(value.constant)
            + abs(value.row) * (self.tile[0] - 1)
            + abs(value.column) * (self.tile[1] - 1)
            + (1 << 31) * sum(abs(coefficient) for _, coefficient in value.atoms)
        )
        if norm > _I64_MAX:
            raise _NotProven
        return value

    def affine(self, node: ast.expr) -> _Affine:
        varies = any(
            isinstance(part, ast.Name) and part.id in (self.row, self.column)
            for part in ast.walk(node)
        )
        if not varies:
            if not _integer(node):
                raise _NotProven
            literal = _literal(node)
            if literal is not None:
                return self.bounded(_Affine(constant=literal))
            text = ast.unparse(node)
            self.atoms[text] = None
            return _Affine(atoms=((text, 1),))
        if isinstance(node, ast.Name):
            value = _Affine(
                row=int(node.id == self.row), column=int(node.id == self.column)
            )
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = self.affine(node.operand).scaled(
                -1 if isinstance(node.op, ast.USub) else 1
            )
        elif isinstance(node, ast.Call) and ast.unparse(node.func) in _INTEGER_CASTS:
            value = self.affine(node.args[0])
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            value = self.affine(node.left).plus(
                self.affine(node.right).scaled(
                    -1 if isinstance(node.op, ast.Sub) else 1
                )
            )
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left, right = _literal(node.left), _literal(node.right)
            if any(
                value is not None and abs(value) > _I32_MAX for value in (left, right)
            ):
                raise _NotProven
            if left is not None:
                value = self.affine(node.right).scaled(left)
            elif right is not None:
                value = self.affine(node.left).scaled(right)
            else:
                raise _NotProven
        else:
            raise _NotProven
        self.intermediates.append(self.bounded(value))
        return value

    def expression(
        self, value: _Affine, *, upper: bool = False, origin: bool = False
    ) -> str:
        self.bounded(value)
        constant = value.constant
        if not origin:
            select = max if upper else min
            constant += select(0, value.row * (self.tile[0] - 1))
            constant += select(0, value.column * (self.tile[1] - 1))
        parts = [f"cutlass.Int64({constant})"]
        parts.extend(
            f"(cutlass.Int64(cutlass.Int32({atom})) * {coefficient})"
            for atom, coefficient in value.atoms
        )
        return "(" + " + ".join(parts) + ")"

    def comparison(
        self, left: ast.expr, right: ast.expr, relation: str, negate: bool
    ) -> str:
        difference = self.affine(left).plus(self.affine(right).scaled(-1))
        low, high = self.expression(difference), self.expression(difference, upper=True)
        relation = _NEGATE[relation] if negate else relation
        return {
            "eq": f"({low} == 0) & ({high} == 0)",
            "ne": f"({high} < 0) | ({low} > 0)",
            "lt": f"{high} < 0",
            "le": f"{high} <= 0",
            "gt": f"{low} > 0",
            "ge": f"{low} >= 0",
        }[relation]

    def mask(self, node: ast.expr, negate: bool = False) -> str:
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return str(node.value != negate)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.mask(node.operand, not negate)
        if isinstance(node, ast.BoolOp) or (
            isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr))
        ):
            parts = (
                node.values if isinstance(node, ast.BoolOp) else (node.left, node.right)
            )
            conjunction = isinstance(node.op, (ast.And, ast.BitAnd)) != negate
            # OR is deliberately sufficient, not complete: one whole-tile
            # disjunct must hold. Do not infer per-cell mask coverage.
            return _join(
                [self.mask(part, negate) for part in parts], "&" if conjunction else "|"
            )
        if isinstance(node, ast.Call) and ast.unparse(node.func) in _RELATIONS:
            return self.comparison(
                node.args[0], node.args[1], _RELATIONS[ast.unparse(node.func)], negate
            )
        if isinstance(node, ast.Compare):
            names: dict[type[ast.cmpop], str] = {
                ast.Eq: "eq",
                ast.NotEq: "ne",
                ast.Lt: "lt",
                ast.LtE: "le",
                ast.Gt: "gt",
                ast.GtE: "ge",
            }
            operands = [node.left, *node.comparators]
            if any(type(op) not in names for op in node.ops):
                raise _NotProven
            return _join(
                [
                    self.comparison(left, right, names[type(op)], negate)
                    for left, right, op in zip(
                        operands[:-1], operands[1:], node.ops, strict=True
                    )
                ],
                "|" if negate else "&",
            )
        raise _NotProven


def prove_rectangular_leaf(
    indices: Sequence[str],
    definitions: Mapping[str, str],
    *,
    row: str,
    column: str,
    uniform_names: AbstractSet[str],
    tile_shape: tuple[int, int],
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    dtype: torch.dtype,
    mask: str | None = None,
) -> RectangularLeafPlan | None:
    """Prove an exact full-mask rectangle conditionally, with safe eager guards.

    Uniform names must denote owned, defined signed Int32/Int64 scalar values.
    Opaque uniform subexpressions retain their original typed semantics; only
    positive constant division/modulo and fixed shifts below 32 are admitted.
    The caller must retain original scalar loads on every guard failure and
    independently validate the current source pointer/descriptor and aliasing.
    """
    if (
        dtype not in _DTYPES
        or not indices
        or row == column
        or len(indices) != len(shape)
        or len(shape) != len(strides)
        or len(tile_shape) != 2
        or any(
            type(value) is not int or not 0 < value < _I32_MAX
            for value in (*shape, *strides, *tile_shape)
        )
        or math.prod(shape) >= _I32_MAX
    ):
        return None
    expected = 1
    for stride, extent in sorted(
        (stride, extent)
        for stride, extent in zip(strides, shape, strict=True)
        if extent != 1
    ):
        if stride != expected:
            return None
        expected *= extent
    try:
        expanded = tuple(
            _expand(index, definitions, row, uniform_names | {column})
            for index in indices
        )
        predicate = (
            _expand(mask, definitions, row, uniform_names | {column})
            if mask is not None
            else None
        )
        proof = _Proof(row, column, tile_shape)
        axes = tuple(proof.affine(node) for node in expanded)
        flat = _Affine()
        for axis, stride in zip(axes, strides, strict=True):
            flat = proof.bounded(flat.plus(axis.scaled(stride)))
        pitch = flat.row
        dtype_name, element_bytes = _DTYPES[dtype]
        total = math.prod(shape)
        if (
            flat.column != 1
            or pitch < tile_shape[1]
            or total % pitch
            or pitch * element_bytes % 16
        ):
            return None
        view_shape = (total // pitch, pitch)
        if tile_shape[0] > view_shape[0]:
            return None
        mask_guard = proof.mask(predicate) if predicate is not None else "True"
        base = proof.expression(flat, origin=True)
        origin = (f"({base} // {pitch})", f"({base} % {pitch})")
        guards = [
            f"cutlass.Int64({atom}) == cutlass.Int64(cutlass.Int32({atom}))"
            for atom in proof.atoms
        ]
        for value in proof.intermediates:
            guards.extend(
                (
                    f"{proof.expression(value)} >= {_I32_MIN}",
                    f"{proof.expression(value, upper=True)} <= {_I32_MAX}",
                )
            )
        for value, extent in zip(axes, shape, strict=True):
            guards.extend(
                (
                    f"{proof.expression(value)} >= 0",
                    f"{proof.expression(value, upper=True)} < {extent}",
                )
            )
        guards.extend(
            (
                f"{origin[0]} >= 0",
                f"{origin[0]} <= {view_shape[0] - tile_shape[0]}",
                f"{origin[1]} <= {pitch - tile_shape[1]}",
                f"{origin[1]} % {16 // element_bytes} == 0",
                mask_guard,
            )
        )

        class AtOrigin(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                return ast.Constant(0) if node.id in (row, column) else node

        origins = tuple(
            ast.unparse(
                AtOrigin().visit(ast.parse(ast.unparse(node), mode="eval").body)
            )
            for node in expanded
        )
        return RectangularLeafPlan(
            tuple(ast.unparse(node) for node in expanded),
            ast.unparse(predicate) if predicate is not None else None,
            row,
            column,
            tile_shape,
            shape,
            strides,
            dtype_name,
            element_bytes,
            pitch,
            view_shape,
            origins,
            base,
            origin,
            _join(guards),
        )
    except (_NotProven, SyntaxError):
        return None
