"""Bounds on the row loops used by CuTe reduction schedules."""

from __future__ import annotations

import ast
from collections import Counter
import dataclasses
from typing import TYPE_CHECKING

from ..device_function import TensorSizeArg
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import call_path_str
from .affine_vector_io import _i32_offset

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..device_function import DeviceFunction


def minimum_terms(value: ast.expr) -> tuple[ast.expr, ast.expr] | None:
    if (
        isinstance(value, ast.IfExp)
        and isinstance(value.test, ast.Compare)
        and len(value.test.ops) == len(value.test.comparators) == 1
        and isinstance(value.test.ops[0], (ast.Lt, ast.LtE))
        and ast.dump(value.test.left) == ast.dump(value.body)
        and ast.dump(value.test.comparators[0]) == ast.dump(value.orelse)
    ):
        return value.body, value.orelse
    return None


def clipped_tile_width(
    start: ast.expr,
    end: ast.expr,
    tensor_size_names: frozenset[str],
    int32_names: frozenset[str] = frozenset(),
) -> int | None:
    # Match the native-width tile edge from GenerateAST.tile_end_expr:
    # min(start, size - span) + span, where span = min(size, width).
    # A tensor size is nonnegative, so the span is in [0, width] and
    # size - span cannot underflow. The clipped start is an Int32 no
    # greater than start; a signed wrap of the sum can only shorten the loop.
    def is_i32_value(expr: ast.expr) -> bool:
        if not isinstance(expr, ast.BinOp):
            return _i32_offset(expr, int32_names)
        # Int32(pid) * 32 is a native Int32 even if it wraps. Two Python
        # literals, however, multiply without narrowing (65536 * 65536).
        return (
            isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult))
            and not (
                isinstance(expr.left, ast.Constant)
                and isinstance(expr.right, ast.Constant)
            )
            and is_i32_value(expr.left)
            and is_i32_value(expr.right)
        )

    while (
        isinstance(end, ast.Call)
        and call_path_str(end.func) == "cutlass.Int32"
        and len(end.args) == 1
        and not end.keywords
    ):
        end = end.args[0]
    if not (
        is_i32_value(start)
        and isinstance(end, ast.BinOp)
        and isinstance(end.op, ast.Add)
    ):
        return None
    clip = minimum_terms(end.left)
    span = minimum_terms(end.right)
    if clip is None or span is None:
        return None
    first, last = clip
    size, width = span
    while (
        ast.dump(start) != ast.dump(first)
        and isinstance(start, ast.Call)
        and call_path_str(start.func) == "cutlass.Int32"
        and len(start.args) == 1
        and is_i32_value(start.args[0])
        and not start.keywords
    ):
        start = start.args[0]
    if (
        isinstance(size, ast.Name)
        and size.id in tensor_size_names
        and isinstance(width, ast.Constant)
        and type(width.value) is int
        and 0 < width.value < 2**31
        and ast.dump(start) == ast.dump(first)
        and isinstance(last, ast.BinOp)
        and isinstance(last.op, ast.Sub)
        and ast.dump(last.left) == ast.dump(size)
        and ast.dump(last.right) == ast.dump(end.right)
    ):
        return width.value
    return None


@dataclasses.dataclass(frozen=True)
class SerialRangeFacts:
    constexpr: Callable[[str], int | None] | None = None
    tensor_sizes: frozenset[str] = frozenset()
    int32_names: frozenset[str] = frozenset()

    @classmethod
    def for_device_function(cls, fn: DeviceFunction) -> SerialRangeFacts:
        sizes = {arg.name for arg in fn.arguments if isinstance(arg, TensorSizeArg)}
        if not sizes:
            return cls(fn.resolved_int_constexpr)
        # Only unconditional definitions before the loops dominate every
        # interchange. A later rebind invalidates even an enclosing definition.
        writes = Counter(
            node.id
            for stmt in fn.body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
        )
        i32: set[str] = set()

        def native(expr: ast.expr) -> bool:
            return (
                isinstance(expr, ast.Name)
                and expr.id in i32
                or isinstance(expr, ast.Call)
                and _i32_offset(expr)
            )

        def fits(expr: ast.expr) -> bool:
            if isinstance(expr, ast.Name) and not writes[expr.id]:
                value = fn.resolved_int_constexpr(expr.id)
                return value is not None and -(2**31) <= value < 2**31
            return native(expr) or isinstance(expr, ast.Constant) and _i32_offset(expr)

        for stmt in fn.body:
            if not isinstance(stmt, ast.Assign):
                break
            name = assigned_name(stmt)
            if name is None or writes[name] != 1:
                continue
            value = stmt.value
            # A product of two Python constexprs need not fit, even if each
            # one does. Require a device Int32 operand for native arithmetic.
            if native(value) or (
                isinstance(value, ast.BinOp)
                and isinstance(value.op, (ast.Add, ast.Sub, ast.Mult))
                and (native(value.left) or native(value.right))
                and fits(value.left)
                and fits(value.right)
            ):
                i32.add(name)
        return cls(
            fn.resolved_int_constexpr, frozenset(sizes - writes.keys()), frozenset(i32)
        )
