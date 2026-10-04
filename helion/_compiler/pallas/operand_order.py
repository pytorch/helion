"""Operand order of broadcasting elementwise ops for Mosaic layout inference."""

from __future__ import annotations

from itertools import starmap

import torch

from ..compile_environment import CompileEnvironment

_COMMUTATIVE_TARGETS = (torch.ops.aten.mul.Tensor, torch.ops.aten.add.Tensor)
_REDUCTION_TARGETS = (
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.mean.dim,
    torch.ops.aten.amax.default,
    torch.ops.aten.amin.default,
)


def _has_shape(value: torch.Tensor, shape: torch.Size) -> bool:
    env = CompileEnvironment.current()
    return value.ndim == len(shape) and all(
        starmap(env.known_equal, zip(value.shape, shape, strict=True))
    )


def _tensor(node: object) -> torch.Tensor | None:
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _reduces_lanes(node: torch.fx.Node) -> bool:
    """A reduction that drops the last (lane) dim of its source and keeps a
    non-scalar result."""
    if node.target not in _REDUCTION_TARGETS or len(node.args) < 2:
        return False
    source, dims = node.args[0], node.args[1]
    keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim")
    source_val, value = _tensor(source), _tensor(node)
    if keepdim or source_val is None or value is None or value.ndim == 0:
        return False
    if not isinstance(dims, (list, tuple)):
        return False
    return any(
        isinstance(d, int) and d % source_val.ndim == source_val.ndim - 1 for d in dims
    )


def put_full_operand_first(graph: torch.fx.Graph) -> None:
    """Order the operands of commutative broadcasts for Mosaic layouts.

    Mosaic gives an elementwise op the vector layout of its first operand and
    relayouts the other operands to it.  A lane reduction yields values laid
    out down the sublanes: cheap for the reduced tensor, but a full-size
    result in that layout takes up to 128 times the vregs and spills.  So the
    first operand is

    * one that is not lane-reduced (a lane reduction, or a pointwise op
      whose first operand is lane-reduced, so it has that layout), then
    * the full-shape one.

    In ``k[:, None, :] * state`` with ``k`` lane-reduced, ``state`` goes
    first, and in ``sum(state * k, -1) * g[:, None]`` (a ``[6, 128]`` sum
    whose 128 values sit in 16 sublane vregs per head) the broadcast ``g``
    goes first, so the sum is relayouted once to the compact layout.
    ``a * b`` and ``a + b`` are exactly commutative, so results do not
    change.  A 0-d operand is a layout-free splat and stays in place.
    """
    lane_reduced: set[torch.fx.Node] = set()
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if _reduces_lanes(node):
            lane_reduced.add(node)
            continue
        if (
            node.target in _COMMUTATIVE_TARGETS
            and not node.kwargs
            and len(node.args) == 2
        ):
            lhs, rhs = node.args
            out, lhs_val, rhs_val = _tensor(node), _tensor(lhs), _tensor(rhs)
            if (
                out is not None
                and lhs_val is not None
                and rhs_val is not None
                and lhs_val.ndim > 0
                and rhs_val.ndim > 0
            ):
                lhs_key = (lhs in lane_reduced, not _has_shape(lhs_val, out.shape))
                rhs_key = (rhs in lane_reduced, not _has_shape(rhs_val, out.shape))
                if rhs_key < lhs_key:
                    node.args = (rhs, lhs)
        if isinstance(node.target, torch._ops.OpOverload) and (
            torch.Tag.pointwise in node.target.tags
        ):
            first = next((a for a in node.args if _tensor(a) is not None), None)
            if first in lane_reduced:
                lane_reduced.add(node)
