"""Distribute sum reductions over affine updates of a large operand (fast math).

A linear recurrence (delta rule, linear attention, SSM scan) updates a large
state and reduces it in the same step::

    s1 = s * a[:, :, None]  # decay, invariant along the lanes
    p = sum(s1 * k[:, None, :], -1)
    s2 = s1 + d[:, :, None] * k[:, None, :]
    o = sum(s2 * q[:, None, :], -1)

Two identities move the reductions onto the step's input ``s``::

    sum(x * a_b * w, D)          ->  a * sum(x * w, D)
    sum((x + c_b * y) * w, D)    ->  sum(x * w, D) + c * sum(y * w, D)

where ``a_b`` and ``c_b`` broadcast along every reduced dim ``D``.  Every
reduction of the step then reads ``s``, and ``o`` no longer waits for ``s2``.
On TPU a state wider than the vreg file is a VMEM sweep per pass, so the
reductions share one sweep and the update is the second (instead of three
sweeps back to back), and ``sum(y * w, D)`` (``k . q`` here) is computed at
the operands' broadcast shape, which is small.

Both identities reassociate floating-point arithmetic, so the pass only runs
under ``Settings.fast_math``.  It matches fp32 sums over explicit
``x[..., None]`` broadcast views only and leaves anything else unchanged.  It
runs before ``prepare_graph_lowerings``, so the reductions it creates get the
usual lowering and masks.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.fx.experimental import proxy_tensor

from ...language import view_ops

if TYPE_CHECKING:
    from collections.abc import Callable

_ADD = torch.ops.aten.add.Tensor
_MUL = torch.ops.aten.mul.Tensor
_SUM = torch.ops.aten.sum.dim_IntList
# How many rewrites may feed one reduction before distribution stops.  Each
# rewrite moves the reduction back past one op of a recurrence (factoring the
# decay of ``s1 = s * a`` reads ``s``; distributing over ``s2 = s1 + c * y``
# reads ``s1``).  At depth 1, a reduction written on a step's output moves to
# the step's input and one written on a decayed state stays on the state.
# Deeper levels lead to the chunked form, all reductions on the first state,
# whose small cross terms grow quadratically with the depth.
_MAX_DEPTH = 1
_DEPTH = "helion_distribute_depth"


def _val(node: object) -> torch.Tensor | None:
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _binary(node: object, target: object) -> tuple[torch.fx.Node, torch.fx.Node] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target is not target
        or len(node.args) != 2
        or node.kwargs
    ):
        return None
    lhs, rhs = node.args
    if _val(lhs) is None or _val(rhs) is None:
        return None
    assert isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)
    return lhs, rhs


def _sum_dims(node: torch.fx.Node) -> tuple[int, ...] | None:
    """Sorted non-negative dims of a keepdim=False fp32 ``sum.dim_IntList``."""
    if node.op != "call_function" or node.target is not _SUM:
        return None
    if len(node.args) != 2 or node.kwargs:
        return None
    source, dims = node.args
    value, source_value = _val(node), _val(source)
    if value is None or source_value is None or value.dtype != torch.float32:
        return None
    if source_value.dtype != torch.float32 or source_value.ndim == 0:
        return None
    if not isinstance(dims, (list, tuple)):
        return None
    ints = [dim for dim in dims if isinstance(dim, int)]
    if not dims or len(ints) != len(dims):
        return None
    rank = source_value.ndim
    normalized = tuple(sorted({dim % rank for dim in ints}))
    return normalized if len(normalized) == len(dims) else None


def _squeezed_view(
    node: torch.fx.Node, rank: int, dims: tuple[int, ...]
) -> tuple[torch.fx.Node, list[object]] | None:
    """``(source, index)`` such that ``source[index]`` is ``node`` without the
    reduced ``dims``, when ``node`` is a ``subscript`` view of rank ``rank``
    that inserts ``None`` at every reduced dim (so it is invariant along them).
    """
    if (
        node.op != "call_function"
        or node.target is not view_ops.subscript
        or len(node.args) != 2
        or node.kwargs
    ):
        return None
    source, index = node.args
    value = _val(node)
    if _val(source) is None or value is None or value.ndim != rank:
        return None
    if not isinstance(index, (list, tuple)) or len(index) != rank:
        return None
    full = slice(None)
    if not all(entry is None or entry == full for entry in index):
        return None
    if any(index[dim] is not None for dim in dims):
        return None
    assert isinstance(source, torch.fx.Node)
    return source, [entry for dim, entry in enumerate(index) if dim not in dims]


class _Builder:
    def __init__(
        self, graph: torch.fx.Graph, template: torch.fx.Node, depth: int
    ) -> None:
        self.graph = graph
        self.template = template
        self.depth = depth

    def call(
        self,
        target: object,
        args: tuple[object, ...],
        compute: Callable[..., torch.Tensor],
    ) -> torch.fx.Node:
        vals = [_val(arg) if isinstance(arg, torch.fx.Node) else arg for arg in args]
        with proxy_tensor.disable_proxy_modes_tracing():
            value = compute(*vals)
        # pyrefly: ignore [bad-argument-type]
        node = self.graph.call_function(target, args, {})
        node.meta = {**self.template.meta, "val": value, _DEPTH: self.depth}
        return node

    def mul(self, lhs: torch.fx.Node, rhs: torch.fx.Node) -> torch.fx.Node:
        return self.call(_MUL, (lhs, rhs), torch.mul)

    def add(self, lhs: torch.fx.Node, rhs: torch.fx.Node) -> torch.fx.Node:
        return self.call(_ADD, (lhs, rhs), torch.add)

    def sum(self, source: torch.fx.Node, dims: tuple[int, ...]) -> torch.fx.Node:
        return self.call(_SUM, (source, list(dims)), torch.sum)

    def view(self, source: torch.fx.Node, index: list[object]) -> torch.fx.Node:
        if all(entry == slice(None) for entry in index):
            return source
        return self.call(view_ops.subscript, (source, index), lambda x, i: x[tuple(i)])


def _factor_scale(graph: torch.fx.Graph, reduction: torch.fx.Node) -> bool:
    """``sum(x * a_b * w, D) -> a * sum(x * w, D)``."""
    dims = _sum_dims(reduction)
    if dims is None:
        return False
    product = reduction.args[0]
    outer = _binary(product, _MUL)
    if outer is None:
        return False
    assert isinstance(product, torch.fx.Node)
    rank = product.meta["val"].ndim
    for scaled, weight in (outer, outer[::-1]):
        inner = _binary(scaled, _MUL)
        if inner is None:
            continue
        for x, scale in (inner, inner[::-1]):
            view = _squeezed_view(scale, rank, dims)
            x_val = _val(x)
            if (
                view is None
                or x_val is None
                or x_val.shape != product.meta["val"].shape
            ):
                continue
            with graph.inserting_before(reduction):
                build = _Builder(graph, reduction, reduction.meta.get(_DEPTH, 0) + 1)
                rest = build.sum(build.mul(x, weight), dims)
                result = build.mul(build.view(*view), rest)
            reduction.replace_all_uses_with(result)
            return True
    return False


def _distribute_update(graph: torch.fx.Graph, reduction: torch.fx.Node) -> bool:
    """``sum((x + c_b * y) * w, D) -> sum(x * w, D) + c * sum(y * w, D)`` when
    ``y * w`` is smaller than the product (both broadcast somewhere)."""
    depth = reduction.meta.get(_DEPTH, 0)
    dims = _sum_dims(reduction)
    if dims is None or depth >= _MAX_DEPTH:
        return False
    product = reduction.args[0]
    outer = _binary(product, _MUL)
    if outer is None:
        return False
    assert isinstance(product, torch.fx.Node)
    product_val = product.meta["val"]
    rank = product_val.ndim
    for update, weight in (outer, outer[::-1]):
        terms = _binary(update, _ADD)
        if terms is None:
            continue
        for x, affine in (terms, terms[::-1]):
            factors = _binary(affine, _MUL)
            if factors is None:
                continue
            for coef, y in (factors, factors[::-1]):
                view = _squeezed_view(coef, rank, dims)
                y_val, w_val = _val(y), _val(weight)
                assert y_val is not None and w_val is not None
                if view is None or y_val.ndim != rank or w_val.ndim != rank:
                    continue
                dot_shape = torch.broadcast_shapes(y_val.shape, w_val.shape)
                if math.prod(dot_shape) >= math.prod(product_val.shape):
                    continue
                with graph.inserting_before(reduction):
                    build = _Builder(graph, reduction, depth + 1)
                    base = build.sum(build.mul(x, weight), dims)
                    dot = build.sum(build.mul(y, weight), dims)
                    result = build.add(base, build.mul(build.view(*view), dot))
                reduction.replace_all_uses_with(result)
                return True
    return False


def distribute_affine_reductions(graph: torch.fx.Graph, *, fast_math: bool) -> int:
    """Apply both identities to a fixed point; return the number of rewrites."""
    if not fast_math:
        return 0
    changed = 0
    progress = True
    while progress:
        progress = False
        for node in list(graph.nodes):
            if _distribute_update(graph, node) or _factor_scale(graph, node):
                graph.erase_node(node)
                changed += 1
                progress = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed
