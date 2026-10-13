"""Coordinate lowering for static, typed split/join contraction views."""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import view_ops
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from .chained_matmul import _Expression


def _tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, Node):
        value = value.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return None
    if any(type(size) is not int or size <= 0 for size in value.shape):
        return None
    return value


def _same(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)


def _split(node: Node) -> bool:
    if node.target is not view_ops.split or len(node.args) != 1 or node.kwargs:
        return False
    source = _tensor(node.args[0])
    result = node.meta.get("val")
    if (
        source is None
        or source.ndim == 0
        or source.shape[-1] != 2
        or not isinstance(result, tuple)
        or len(result) != 2
    ):
        return False
    for value in result:
        value = _tensor(value)
        if (
            value is None
            or value.dtype != source.dtype
            or tuple(value.shape) != tuple(source.shape[:-1])
        ):
            return False
    return True


def supports(node: Node) -> bool:
    """Keep broadcast joins, dynamic shapes and arbitrary tuple operations out."""
    if node.op != "call_function" or node.target not in (
        view_ops.split,
        view_ops.join,
        operator.getitem,
        torch.ops.aten.full.default,
    ):
        return False
    if node.target is view_ops.split:
        return _split(node) and all(
            user.target is operator.getitem
            and len(user.args) == 2
            and user.args[0] is node
            and type(user.args[1]) is int
            and user.args[1] in (0, 1)
            and not user.kwargs
            for user in node.users
        )
    result = _tensor(node)
    if result is None:
        return False
    if node.target is operator.getitem:
        if (
            len(node.args) != 2
            or node.kwargs
            or not isinstance(node.args[0], Node)
            or type(node.args[1]) is not int
            or node.args[1] not in (0, 1)
            or not _split(node.args[0])
        ):
            return False
        source = node.args[0].meta["val"][node.args[1]]
        return _same(source, result)
    if node.target is view_ops.join:
        if len(node.args) != 2 or node.kwargs:
            return False
        left, right = map(_tensor, node.args)
        return (
            left is not None
            and right is not None
            and _same(left, right)
            and result.dtype == left.dtype
            and tuple(result.shape) == (*left.shape, 2)
        )
    if node.target is torch.ops.aten.full.default:
        if len(node.args) != 2:
            return False
        shape, fill = node.args
        return (
            isinstance(shape, (list, tuple))
            and all(type(size) is int for size in shape)
            and tuple(shape) == tuple(result.shape)
            and type(fill) in (int, float, bool)
            and (not isinstance(fill, float) or math.isfinite(fill))
        )
    return False


def value(
    expression: _Expression, node: Node, coordinates: tuple[str, ...]
) -> str | None:
    if not supports(node):
        return None
    if node.target is operator.getitem:
        split = cast("Node", node.args[0])
        return expression.value(
            cast("Node", split.args[0]), (*coordinates, str(node.args[1]))
        )
    if node.target is view_ops.join:
        left, right = (
            expression.value(cast("Node", arg), coordinates[:-1]) for arg in node.args
        )
        return expression.bind(f"({left} if ({coordinates[-1]}) == 0 else {right})")
    if node.target is torch.ops.aten.full.default:
        dtype = CompileEnvironment.current().backend.dtype_str(node.meta["val"].dtype)
        return f"{dtype}({node.args[1]!r})"
    return None
