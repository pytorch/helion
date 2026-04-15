from __future__ import annotations

import dataclasses
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language.tile_ops import tile_begin

if TYPE_CHECKING:
    import ast


@dataclasses.dataclass(frozen=True)
class CuteAffineRangeIndex:
    base: object
    factor: int
    start: object
    length: object
    step: object
    dtype: torch.dtype


@dataclasses.dataclass(frozen=True)
class CutePackedAffineLoad:
    terms: tuple[ast.AST, ...]


@dataclasses.dataclass(frozen=True)
class CutePackedTerms:
    terms: tuple[ast.AST, ...]


@dataclasses.dataclass(frozen=True)
class CuteShapeChainView:
    node: Node


_CUTE_SHAPE_CHAIN_TARGETS = frozenset(
    {
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.view.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
    }
)


def is_cute_shape_chain_target(target: object) -> bool:
    return target in _CUTE_SHAPE_CHAIN_TARGETS


def _match_constant_multiple(value: object) -> tuple[object, int] | None:
    if (
        isinstance(value, Node)
        and value.op == "call_function"
        and value.target
        in (
            operator.mul,
            torch.ops.aten.mul.Tensor,
        )
    ):
        lhs, rhs = value.args
        if isinstance(lhs, int) and lhs > 0:
            return rhs, lhs
        if isinstance(rhs, int) and rhs > 0:
            return lhs, rhs
    return value, 1


def match_cute_affine_range_iota(node: Node) -> CuteAffineRangeIndex | None:
    if node.op != "call_function" or node.target is not torch.ops.prims.iota.default:
        return None

    start = node.kwargs.get("start", 0)
    step = node.kwargs.get("step", 1)
    dtype = node.kwargs.get("dtype")
    if not isinstance(dtype, torch.dtype):
        return None
    if step != 1:
        return None

    (length_arg,) = node.args
    length_base_factor = _match_constant_multiple(length_arg)
    start_base_factor = _match_constant_multiple(start)
    if length_base_factor is None or start_base_factor is None:
        return None
    length_base, length_factor = length_base_factor
    start_base, start_factor = start_base_factor
    if length_factor <= 1 or length_factor != start_factor:
        return None
    if not (
        isinstance(start_base, Node)
        and start_base.op == "call_function"
        and start_base.target is tile_begin
        and len(start_base.args) == 1
        and start_base.args[0] == length_base
    ):
        return None

    return CuteAffineRangeIndex(
        base=length_base,
        factor=length_factor,
        start=start,
        length=length_arg,
        step=step,
        dtype=dtype,
    )


def unwrap_cute_affine_range_index(value: object) -> CuteAffineRangeIndex | None:
    if isinstance(value, CuteAffineRangeIndex):
        return value
    return None


def match_cute_stack_reshape_rhs(node: Node) -> tuple[tuple[Node, ...], int] | None:
    current = node
    while current.op == "call_function" and current.target in (
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.view.default,
    ):
        source = current.args[0]
        if not isinstance(source, Node):
            return None
        current = source

    if (
        current.op != "call_function"
        or current.target is not torch.ops.aten.stack.default
    ):
        return None

    tensors = current.args[0]
    dim = current.args[1] if len(current.args) > 1 else current.kwargs.get("dim", 0)
    if not isinstance(tensors, (list, tuple)) or not isinstance(dim, int):
        return None
    if len(tensors) <= 1 or not all(isinstance(tensor, Node) for tensor in tensors):
        return None

    first = cast("Node", tensors[0])
    first_val = first.meta.get("val")
    expected_dim = 1
    if isinstance(first_val, torch.Tensor):
        expected_dim = first_val.ndim - 1
        dim %= first_val.ndim + 1
    if dim != expected_dim:
        return None
    return tuple(cast("Node", tensor) for tensor in tensors), len(tensors)


def match_cute_duplicate_stack_reshape_rhs(node: Node) -> tuple[Node, int] | None:
    matched = match_cute_stack_reshape_rhs(node)
    if matched is None:
        return None
    tensors, factor = matched
    first = tensors[0]
    if not all(tensor == first for tensor in tensors[1:]):
        return None
    return first, factor
