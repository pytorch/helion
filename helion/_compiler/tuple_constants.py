"""Recover original immutable tuple constants without replacing runtime tuples."""

from __future__ import annotations

import ast
import dis
import struct
from typing import TYPE_CHECKING

from .ast_extension import ExtendedAST

if TYPE_CHECKING:
    import types


def _constant_key(value: object) -> object:
    kind = type(value)
    if type(value) is tuple:
        return (tuple, tuple(_constant_key(item) for item in value))
    if type(value) is float:
        return (float, struct.pack("!d", value))
    if type(value) is complex:
        return (complex, struct.pack("!dd", value.real, value.imag))
    if kind in (type(None), bool, int, str, bytes, type(Ellipsis)):
        return (kind, value)
    raise TypeError("not an immutable code constant")


class TupleConstants:
    """A single original function's LOAD_CONST positions and constant objects."""

    def __init__(self, code: types.CodeType) -> None:
        self.code = code
        self.positions: dict[tuple[int, int, int, int], tuple[object, ...] | None] = {}
        self.has_positions = False
        for instruction in dis.get_instructions(code):
            position = getattr(instruction, "positions", None)
            if position is None or any(item is None for item in position):
                continue
            self.has_positions = True
            if instruction.opname == "LOAD_CONST" and type(instruction.argval) is tuple:
                key = (
                    position.lineno,
                    position.col_offset,
                    position.end_lineno,
                    position.end_col_offset,
                )
                if (
                    key in self.positions
                    and self.positions[key] is not instruction.argval
                ):
                    self.positions[key] = None
                else:
                    self.positions[key] = instruction.argval

    def resolve(
        self, node: ast.Tuple, values: tuple[object, ...]
    ) -> tuple[object, ...] | None:
        if not isinstance(node, ExtendedAST) or node._compiler_inserted:
            return None
        location = node._location
        if location.filename != self.code.co_filename or location.lineno is None:
            return None
        try:
            key = _constant_key(values)
        except TypeError:
            return None
        if self.has_positions:
            constant = self.positions.get(
                (
                    location.lineno,
                    location.colno,
                    location.end_lineno,
                    location.end_colno,
                )
            )
            if constant is not None and _constant_key(constant) == key:
                return constant
            return None

        # Older interpreters, or code compiled without debug ranges, lack
        # positions. Only a literal-only AST and a unique, fully typed match
        # can establish the original object; equal runtime values cannot.
        try:
            if _constant_key(ast.literal_eval(node)) != key:
                return None
        except (ValueError, TypeError):
            return None
        matches: dict[int, tuple[object, ...]] = {}

        def visit(value: object) -> None:
            if type(value) is tuple:
                if _constant_key(value) == key:
                    matches[id(value)] = value
                for item in value:
                    visit(item)

        for constant in self.code.co_consts:
            visit(constant)
        return next(iter(matches.values())) if len(matches) == 1 else None
