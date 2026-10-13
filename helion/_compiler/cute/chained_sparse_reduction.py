"""Prune positive-zero reduction branches while retaining the live FP32 tree.

Only an original local integer iota compared with one static valid coordinate
is admitted. This is not a sum-to-load rewrite: the initial addition and all
five warp-tree additions remain, with their original left/right operands.
Source readiness, disjoint output storage and publication remain caller-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import view_ops
from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_collectives import classify_collective
from .chained_execution import ChainedExecution

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan


@dataclass(frozen=True)
class SparseReduction:
    node: Node
    source: Node
    condition: Node
    selected: Node
    axis: int
    extent: int
    position: int
    output_shape: tuple[int, ...]
    zero_sides: tuple[str, ...]


def _shape(node: Node) -> tuple[int, ...]:
    if CompileEnvironment.has_current():
        return chain._shape(node)
    value = cast("torch.Tensor", node.meta["val"])
    return tuple(value.shape)


def _positive_zero(value: object) -> bool:
    return (
        type(value) in (int, float)
        and value == 0
        and math.copysign(1, cast("float", value)) > 0
    )


def _literal(node: object, dtype: torch.dtype) -> object:
    if (
        isinstance(node, Node)
        and node.target is torch.ops.aten.scalar_tensor.default
        and len(node.args) == 1
        and isinstance(value := node.meta.get("val"), torch.Tensor)
        and value.ndim == 0
        and value.dtype == dtype
    ):
        return node.args[0]
    return None


def _broadcast(
    source: tuple[int, ...], target: tuple[int, ...], axes: tuple[int | None, ...]
) -> tuple[int | None, ...] | None:
    if len(source) > len(target):
        return None
    offset = len(target) - len(source)
    if any(size not in (1, target[i + offset]) for i, size in enumerate(source)):
        return None
    return tuple(
        None if size == 1 else axes[i + offset] for i, size in enumerate(source)
    )


def _views(
    node: Node, axes: tuple[int | None, ...]
) -> tuple[Node, tuple[int | None, ...]]:
    """Map only exact identity/broadcast/axis views, never indices or casts."""
    while node.args and isinstance(node.args[0], Node):
        source = node.args[0]
        if not isinstance(source.meta.get("val"), torch.Tensor):
            break
        old, new = _shape(source), _shape(node)
        mapped = None
        if node.kwargs or source.meta["val"].dtype != node.meta["val"].dtype:
            break
        if node.target is _tracing_ops._new_var and old == new:
            mapped = axes
        elif node.target is view_ops.subscript and len(node.args) == 2:
            selectors = node.args[1]
            if (
                isinstance(selectors, (tuple, list))
                and len(selectors) == len(axes)
                and all(index is None or index == slice(None) for index in selectors)
            ):
                mapped = tuple(
                    axis
                    for axis, index in zip(axes, selectors, strict=True)
                    if index is not None
                )
        elif node.target is torch.ops.aten.unsqueeze.default and len(node.args) == 2:
            dimension = node.args[1]
            if type(dimension) is int and -len(new) <= dimension < len(new):
                dimension %= len(new)
                if new[dimension] == 1:
                    mapped = axes[:dimension] + axes[dimension + 1 :]
        elif node.target is torch.ops.aten.expand.default:
            mapped = _broadcast(old, new, axes)
        elif node.target is torch.ops.aten.permute.default and len(node.args) == 2:
            permutation = node.args[1]
            if (
                isinstance(permutation, (tuple, list))
                and all(type(axis) is int for axis in permutation)
                and sorted(cast("tuple[int, ...]", permutation))
                == list(range(len(old)))
            ):
                mapped = tuple(axes[permutation.index(i)] for i in range(len(old)))
        if mapped is None or len(mapped) != len(old):
            break
        node, axes = source, mapped
    return node, axes


def plan_sparse_reduction(
    plan: ChainedMatmulPlan, node: Node
) -> SparseReduction | None:
    """Prove one static live coordinate in a single-part FP32 warp reduction."""
    region = plan.region
    operation = classify_collective(node)
    if (
        region is None
        or node not in region.reductions
        or operation is None
        or operation.kind != "sum"
        or node.graph is not region.graph
        or any(
            item.graph is not region.graph or item not in region.nodes
            for item in chain._ancestors(node)
        )
    ):
        return None
    source = operation.source
    shape, output_shape = _shape(source), _shape(node)
    axis = operation.axis
    extent = shape[axis]
    keepdim = node.args[2] if len(node.args) == 3 else node.kwargs.get("keepdim", False)
    if type(keepdim) is not bool:
        return None
    expected = (
        tuple(1 if i == axis else size for i, size in enumerate(shape))
        if keepdim
        else (shape[1 - axis],)
    )
    if (
        not 1 <= extent <= 32
        or shape[1 - axis] <= 0
        or output_shape != expected
        or node.kwargs.get("dtype") not in (None, torch.float32)
    ):
        return None
    selection = source
    while selection.target is _tracing_ops._mask_to:
        if (
            len(selection.args) != 2
            or selection.kwargs
            or not _positive_zero(selection.args[1])
        ):
            return None
        inner = selection.args[0]
        if (
            not isinstance(inner, Node)
            or _shape(inner) != shape
            or inner.meta["val"].dtype != torch.float32
        ):
            return None
        selection = inner
    if (
        selection.target is not torch.ops.aten.where.self
        or len(selection.args) != 3
        or selection.kwargs
    ):
        return None
    condition, selected, false = selection.args
    if (
        not isinstance(condition, Node)
        or not isinstance(selected, Node)
        or condition.meta["val"].dtype != torch.bool
        or selected.meta["val"].dtype != torch.float32
        or not _positive_zero(_literal(false, torch.float32))
    ):
        return None
    axes = _broadcast(_shape(condition), shape, (0, 1))
    if axes is None:
        return None
    equality, axes = _views(condition, axes)
    if (
        equality.target not in (torch.ops.aten.eq.Scalar, torch.ops.aten.eq.Tensor)
        or len(equality.args) != 2
        or equality.kwargs
    ):
        return None
    for index, position in (equality.args, equality.args[::-1]):
        if isinstance(position, Node):
            integer = _literal(position, torch.int64)
            position = _literal(position, torch.int32) if integer is None else integer
        if (
            not isinstance(index, Node)
            or not isinstance(index.meta.get("val"), torch.Tensor)
            or type(position) is not int
            or not 0 <= position < extent
        ):
            continue
        index_axes = _broadcast(_shape(index), _shape(equality), axes)
        if index_axes is None:
            continue
        iota, index_axes = _views(index, index_axes)
        if (
            iota.target is torch.ops.prims.iota.default
            and iota.meta["val"].dtype in (torch.int32, torch.int64)
            and _shape(iota) == (extent,)
            and (index_axes == (axis,) or extent == 1 and index_axes == (None,))
            and len(iota.args) == 1
            and type(iota.args[0]) is int
            and iota.args[0] == extent
            and type(iota.kwargs.get("start", 0)) is int
            and iota.kwargs.get("start", 0) == 0
            and type(iota.kwargs.get("step", 1)) is int
            and iota.kwargs.get("step", 1) == 1
        ):
            return SparseReduction(
                node,
                source,
                condition,
                selected,
                axis,
                extent,
                position,
                output_shape,
                tuple(
                    "left" if position & offset else "right"
                    for offset in (16, 8, 4, 2, 1)
                ),
            )
    return None


def emit_sparse_reduction(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    node: Node,
    name: str,
    *,
    execution: ChainedExecution | None = None,
) -> list[str] | None:
    """Emit only the proven live FP32 path; caller retains its final barrier."""
    proof = plan_sparse_reduction(plan, node)
    if proof is None:
        return None
    execution = execution or ChainedExecution(plan.threads)
    vector, accumulator = f"{name}_vector", f"{name}_acc"
    coordinates = (
        (str(proof.position), vector)
        if proof.axis == 0
        else (vector, str(proof.position))
    )
    expression = chain._Expression(cg, plan, boundaries)
    expression.coordinate_names.add(vector)
    try:
        # Retain the original where and _mask_to wrappers. The proven equality
        # is constant at these coordinates; source/domain masks remain dynamic.
        value = expression.value(proof.source, coordinates)
        domain = chain._operand_domain(cg, proof.source, coordinates, plan)
    except chain._UnsupportedChain:
        return None
    value = chain._masked_operand(value, "cutlass.Float32", domain)
    output_coordinates = (
        (("0", vector) if proof.axis == 0 else (vector, "0"))
        if len(proof.output_shape) == 2
        else (vector,)
    )
    vectors = _shape(proof.source)[1 - proof.axis]
    body = [*expression.lines, f"{accumulator} = cutlass.Float32(0) + {value}"]
    for side in proof.zero_sides:
        body.append(
            f"{accumulator} = cutlass.Float32(0) + {accumulator}"
            if side == "left"
            else f"{accumulator} = {accumulator} + cutlass.Float32(0)"
        )
    body.append(f"{name}[{', '.join(output_coordinates)}] = {accumulator}")
    return [
        f"for {name}_vector_step in cutlass.range_constexpr({(vectors + execution.threads - 1) // execution.threads}):",
        f"    {vector} = {execution.thread} + {name}_vector_step * {execution.threads}",
        f"    if {vector} < {vectors}:",
        chain._indent(body, 8),
    ]
