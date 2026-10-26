"""Bounded same-warp value chains from reductions through global fetch-adds."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.ir import Reduction
from torch.fx import Node

from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import memory_ops
from ...language import view_ops
from ..indexing_strategy import SubscriptIndexing
from ..inductor_lowering import ReductionLowering

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment


def static_sizes(sizes: object, env: CompileEnvironment) -> tuple[int, ...] | None:
    result = []
    for size in cast("tuple[int | torch.SymInt, ...]", sizes):
        expr = env.specialize_expr(
            cast("sympy.Expr", size._sympy_())
            if isinstance(size, torch.SymInt)
            else sympy.Integer(size)
        )
        if not expr.is_Integer or int(expr) <= 0:
            return None
        result.append(int(expr))
    return tuple(result)


def static_shape(value: object, env: CompileEnvironment) -> tuple[int, ...] | None:
    return static_sizes(value.shape, env) if isinstance(value, torch.Tensor) else None


def broadcast_preserves_warp(
    source: tuple[int, ...], target: tuple[int, ...], groups: int
) -> bool:
    """Prove every broadcasted source belongs to the target's physical warp.

    The domain is at most one value per lane, bounded by the CTA worker limit.
    Enumeration avoids assumptions about singleton placement or equal extents.
    """
    if len(source) > len(target):
        return False
    source_size, target_size = math.prod(source), math.prod(target)
    if source_size not in (groups, groups * 32) or target_size not in (
        groups,
        groups * 32,
    ):
        return False
    offset = len(target) - len(source)
    if any(a not in (1, b) for a, b in zip(source, target[offset:], strict=True)):
        return False
    for index in range(target_size):
        coordinates = [
            index // math.prod(target[axis + 1 :]) % size
            for axis, size in enumerate(target)
        ]
        source_index = sum(
            (0 if size == 1 else coordinates[offset + axis])
            * math.prod(source[axis + 1 :])
            for axis, size in enumerate(source)
        )
        if source_index // (source_size // groups) != index // (target_size // groups):
            return False
    return True


def warp_result_chain(
    node: Node, env: CompileEnvironment, max_threads: int
) -> dict[Node, int]:
    """Admit a complete lexical chain, or retain ordinary shared storage.

    Only Int32 sum changes placement here; its existing serial/parallel
    algorithm remains independent. Reshapes preserve logical linear order,
    pointwise broadcasts preserve the proved warp, and side effects keep their
    existing CTA ordering barriers. Carries, cross-warp indexing, reductions,
    scans, opaque code and unproved views are deliberately excluded.
    """
    lowering = node.meta.get("lowering")
    value = node.meta.get("val")
    shape = static_shape(value, env)
    if (
        not isinstance(lowering, ReductionLowering)
        or lowering.reduction_type != "sum"
        or not isinstance(value, torch.Tensor)
        or value.dtype != torch.int32
        or shape is None
        or not node.users
    ):
        return {}
    reduction = lowering.buffer.data
    if not isinstance(reduction, Reduction):
        return {}
    ranges = reduction.reduction_ranges
    if len(ranges) != 1 or env.specialize_expr(sympy.sympify(ranges[0])) != 32:
        return {}
    source = node.args[0]
    input_shape = (
        static_shape(source.meta.get("val"), env) if isinstance(source, Node) else None
    )
    dims = node.args[1] if len(node.args) > 1 else None
    if (
        input_shape is None
        or not isinstance(dims, (list, tuple))
        or len(dims) != 1
        or not isinstance(dims[0], int)
        or dims[0] % len(input_shape) != len(input_shape) - 1
        or input_shape[-1] != 32
    ):
        return {}
    groups = math.prod(shape)
    if groups > max_threads // 32:
        return {}
    layouts = {node: shape}
    pending = [node]
    used_atomic = False
    while pending:
        source = pending.pop()
        for user in source.users:
            if user.graph is not node.graph:
                return {}
            target = user.target
            view = target in (
                _tracing_ops._new_var,
                torch.ops.aten.alias.default,
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten.unsqueeze.default,
                torch.ops.aten.squeeze.default,
                torch.ops.aten.squeeze.dim,
                torch.ops.aten.squeeze.dims,
            )
            if target in (memory_ops.load, view_ops.subscript):
                if user.args[0] is not source or (
                    target is memory_ops.load and user.args[2] is not None
                ):
                    return {}
                if any(
                    index is not None
                    and not (
                        isinstance(index, slice)
                        and index.start is None
                        and index.stop is None
                        and index.step is None
                    )
                    for index in cast("list[object]", user.args[1])
                ):
                    return {}
                view = True
            output_shape = static_shape(user.meta.get("val"), env)
            if target in (memory_ops.store, atomic_ops.atomic_add):
                if user.args[0] is source:
                    return {}
                host = cast("Node", user.args[0])
                if target is memory_ops.store:
                    if host.target is not _tracing_ops._host_tensor:
                        return {}
                    indices = [
                        index.meta["val"] if isinstance(index, Node) else index
                        for index in cast("list[object]", user.args[1])
                    ]
                    destination = host.meta["val"]
                    sizes = SubscriptIndexing.compute_shape(destination, indices)
                    # Preserve symbolic proof rather than relying on tracing hints.
                    output_shape = static_sizes(sizes, env)
                else:
                    if user.args[3] != "relaxed":
                        return {}
                    if user.users:
                        if (
                            host.target is not _tracing_ops._host_tensor
                            or host.meta["val"].dtype != torch.int32
                        ):
                            return {}
                        if output_shape is None or math.prod(output_shape) != groups:
                            return {}
                        used_atomic = True
            elif not view and not (
                isinstance(target, torch._ops.OpOverload)
                and torch.Tag.pointwise in target.tags
                and not target._schema.is_mutable
            ):
                return {}
            if output_shape is None or math.prod(output_shape) not in (
                groups,
                groups * 32,
            ):
                return {}
            if view:
                if math.prod(layouts[source]) != math.prod(output_shape):
                    return {}
            elif not broadcast_preserves_warp(layouts[source], output_shape, groups):
                return {}
            if user in layouts:
                continue
            layouts[user] = output_shape
            if user.users:
                pending.append(user)
            elif target not in (memory_ops.store, atomic_ops.atomic_add):
                return {}
    return dict.fromkeys(layouts, groups) if used_atomic else {}
