"""Shared recognition helpers for MXFP8 scale expressions."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import cast

import sympy
import torch

from ..language import _tracing_ops
from ..language import memory_ops
from ..language import tile_ops
from ..language import view_ops


@dataclass(frozen=True)
class PackedScaleMatch:
    """A proved DeepGEMM-style packed E8M0 scale extraction."""

    tensor: torch.Tensor
    row: object
    word_index: torch.fx.Node
    mask: object | None


def host_tensor(node: object) -> torch.Tensor | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not _tracing_ops._host_tensor
    ):
        return None
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) and value.ndim == 2 else None


def tile_load(
    node: object, *, allow_mask: bool = False
) -> tuple[torch.Tensor, list[object]] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not memory_ops.load
        or len(node.args) != 4
        or node.args[3] is not None
        or (not allow_mask and node.args[2] is not None)
        or node.kwargs
        or not isinstance(node.args[1], (list, tuple))
        or len(node.args[1]) != 2
    ):
        return None
    tensor = host_tensor(node.args[0])
    return (tensor, list(node.args[1])) if tensor is not None else None


def packed_group_expression(
    node: object, groups: int, *, budget: int = 128
) -> sympy.Expr | None:
    """Prove a broadcasted full group iota and its bounded integer arithmetic."""
    remaining = budget
    coordinate = sympy.Symbol("group", integer=True, nonnegative=True)
    if groups <= 0 or groups >= 1 << 26:
        return None
    quotient = cast(
        "sympy.Expr",
        sympy.floor(sympy.Mul(coordinate, sympy.Rational(1, 4))),
    )
    remainder = sympy.Mod(coordinate, 4)
    if not isinstance(remainder, sympy.Expr):
        return None

    def visit(value: object) -> sympy.Expr | None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            return None
        if type(value) is int:
            return sympy.Integer(value)
        if not isinstance(value, torch.fx.Node) or value.op != "call_function":
            return None
        if (
            value.target is _tracing_ops._new_var
            and len(value.args) == 1
            and not value.kwargs
        ):
            return visit(value.args[0])
        if value.target is tile_ops.tile_index:
            tile_value = value.meta.get("val")
            if (
                isinstance(tile_value, torch.Tensor)
                and tile_value.ndim == 1
                and (
                    not isinstance(tile_value.shape[0], int)
                    or tile_value.shape[0] == groups
                )
            ):
                return coordinate
            return None
        if value.target in (memory_ops.load, view_ops.subscript):
            if len(value.args) not in (2, 4) or value.kwargs:
                return None
            if len(value.args) == 4 and value.args[2:] != (None, None):
                return None
            if not isinstance(value.args[1], (list, tuple)) or any(
                item is not None and item != slice(None) for item in value.args[1]
            ):
                return None
            return visit(value.args[0])
        if value.target is torch.ops.aten.unsqueeze.default and len(value.args) == 2:
            return visit(value.args[0])
        if value.target is torch.ops.prims.convert_element_type.default:
            if len(value.args) != 2 or value.args[1] not in (torch.int32, torch.int64):
                return None
            return visit(value.args[0])
        if value.target in (
            torch.ops.aten.arange.default,
            torch.ops.aten.arange.start,
            torch.ops.aten.arange.start_step,
        ):
            if value.kwargs.get("dtype", torch.int64) not in (
                torch.int32,
                torch.int64,
            ):
                return None
            values = tuple(value.args)
            if values not in ((groups,), (0, groups), (0, groups, 1)):
                return None
            return coordinate
        if value.target is torch.ops.prims.iota.default:
            if (
                value.args != (groups,)
                or value.kwargs.get("start", 0) != 0
                or value.kwargs.get("step", 1) != 1
                or value.kwargs.get("dtype", torch.int64)
                not in (torch.int32, torch.int64)
            ):
                return None
            return coordinate
        if len(value.args) != 2:
            return None
        left, right = (visit(item) for item in value.args)
        if left is None or right is None:
            return None
        if value.target in (
            operator.floordiv,
            torch.ops.aten.div.Tensor_mode,
        ):
            if value.target is torch.ops.aten.div.Tensor_mode:
                if value.kwargs != {"rounding_mode": "floor"}:
                    return None
            elif value.kwargs:
                return None
            return quotient if left == coordinate and right == 4 else None
        if value.kwargs:
            return None
        if value.target in (
            operator.mod,
            torch.ops.aten.remainder.Scalar,
            torch.ops.aten.remainder.Tensor,
        ):
            return remainder if left == coordinate and right == 4 else None
        if value.target in (operator.mul, torch.ops.aten.mul.Tensor):
            if (left == remainder and right == 8) or (right == remainder and left == 8):
                return sympy.Mul(remainder, 8)
        return None

    return visit(node)


def match_packed_scale(
    node: object,
    row: object | None,
    groups: int | None = None,
    *,
    allow_mask: bool = False,
) -> PackedScaleMatch | None:
    """Match ``uint8((load(int32) >> ((group % 4) * 8)) & 255)``."""
    while (
        isinstance(node, torch.fx.Node)
        and node.target is _tracing_ops._new_var
        and len(node.args) == 1
        and not node.kwargs
    ):
        node = node.args[0]
    if (
        not isinstance(node, torch.fx.Node)
        or node.target is not torch.ops.prims.convert_element_type.default
        or node.args[1:] != (torch.uint8,)
        or node.kwargs
    ):
        return None
    masked = node.args[0]
    if (
        not isinstance(masked, torch.fx.Node)
        or masked.target not in (operator.and_, torch.ops.aten.bitwise_and.Scalar)
        or len(masked.args) != 2
        or masked.args[1] != 255
        or masked.kwargs
    ):
        return None
    shifted = masked.args[0]
    if (
        not isinstance(shifted, torch.fx.Node)
        or shifted.target
        not in (
            operator.rshift,
            torch.ops.aten.bitwise_right_shift.Tensor,
            torch.ops.aten.__rshift__.Tensor,
        )
        or len(shifted.args) != 2
        or shifted.kwargs
    ):
        return None
    words = tile_load(shifted.args[0], allow_mask=allow_mask)
    if (
        words is None
        or words[0].dtype != torch.int32
        or (row is not None and words[1][0] != row)
        or not isinstance(words[1][1], torch.fx.Node)
    ):
        return None
    if groups is None:
        columns = words[0].shape[1]
        if not isinstance(columns, int):
            return None
        groups = columns * 4
    index_value = words[1][1].meta.get("val")
    shift_value = (
        shifted.args[1].meta.get("val")
        if isinstance(shifted.args[1], torch.fx.Node)
        else None
    )
    if (
        not isinstance(index_value, torch.Tensor)
        or index_value.ndim != 1
        or not isinstance(shift_value, torch.Tensor)
        or not (
            shift_value.ndim == 1
            or (
                shift_value.ndim == 2
                and type(shift_value.shape[0]) is int
                and shift_value.shape[0] == 1
            )
        )
    ):
        return None
    group = sympy.Symbol("group", integer=True, nonnegative=True)
    remainder = sympy.Mod(group, 4)
    if not isinstance(remainder, sympy.Expr):
        return None
    word_expr = packed_group_expression(words[1][1], groups)
    shift_expr = packed_group_expression(shifted.args[1], groups)
    if word_expr != sympy.floor(
        sympy.Mul(group, sympy.Rational(1, 4))
    ) or shift_expr != sympy.Mul(remainder, 8):
        return None
    load_node = shifted.args[0]
    assert isinstance(load_node, torch.fx.Node)
    return PackedScaleMatch(words[0], words[1][0], words[1][1], load_node.args[2])
