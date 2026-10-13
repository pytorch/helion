"""Logical-axis and readonly load invariance, independent of a kernel name."""

from __future__ import annotations

import dataclasses
from typing import cast

import torch
from torch.fx import Node

from ...language import memory_ops
from ...language import view_ops
from .horizontal_state_update import InapplicableStateUpdate
from .horizontal_state_update import PointwiseStateRegion
from .horizontal_state_update import _ancestors
from .horizontal_state_update import _pointwise
from .horizontal_state_update import _proven_int


@dataclasses.dataclass(frozen=True)
class FeatureInvariance:
    region: PointwiseStateRegion = dataclasses.field(repr=False)
    axes: tuple[tuple[Node, tuple[str | None, ...]], ...]
    invariant: frozenset[Node]

    def check(self) -> None:
        self.region.check_readonly_owners()
        if self != feature_invariance(self.region):
            raise InapplicableStateUpdate("feature-invariant graph or strides changed")

    def coordinates(self, node: Node, coordinates: tuple[str, ...]) -> tuple[str, ...]:
        if node not in self.invariant:
            return coordinates
        axes = dict(self.axes)[node]
        if len(axes) != len(coordinates):
            raise InapplicableStateUpdate("invariant coordinate rank changed")
        return tuple(
            "0" if axis == "feature" else c
            for axis, c in zip(axes, coordinates, strict=True)
        )


def feature_invariance(region: PointwiseStateRegion) -> FeatureInvariance:
    """Prove same loaded bits over the original feature domain, then pure DAG.

    Zero stride is an address fact only. A varying dynamic index predicate or
    explicit mask prevents hoisting even if its address stride is zero.
    """
    region.check_readonly_owners()
    axes: dict[Node, tuple[str | None, ...]] = {}
    invariant: set[Node] = set()
    feature = region.feature_selector
    row = region.row_selector
    for node in region.root.graph.nodes:
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            continue
        if node is feature:
            axes[node] = ("feature",)
            continue
        if node.target is memory_ops.load:
            selectors = node.args[1]
            source = node.args[0].meta["val"]
            dims: list[str | None] = []
            fixed = node is not region.state_load
            for dim, index in enumerate(selectors):
                if index is row:
                    dims.append("row")
                elif index is feature:
                    dims.append("feature")
                    if (
                        _proven_int(source.stride(dim)) != 0
                        or _proven_int(source.shape[dim]) != feature.args[0]
                    ):
                        fixed = False
                elif isinstance(index, Node):
                    if feature in _ancestors(index):
                        fixed = False
                    item = index.meta.get("val")
                    if isinstance(item, torch.Tensor) and item.ndim:
                        raise InapplicableStateUpdate("unproved logical index axes")
            extra = node.args[2]
            if isinstance(extra, Node) and feature in _ancestors(extra):
                fixed = False
            axes[node] = tuple(dims)
            if len(dims) != value.ndim:
                raise InapplicableStateUpdate("memory axes do not match logical rank")
            if fixed:
                invariant.add(node)
            continue
        if node.target is view_ops.subscript:
            source, selectors = node.args
            if source not in axes:
                raise InapplicableStateUpdate("unproved view source axes")
            pending = iter(axes[source])
            dims = [None if x is None else next(pending) for x in selectors]
            if next(pending, "end") != "end" or len(dims) != value.ndim:
                raise InapplicableStateUpdate("view axes do not preserve dimensions")
            axes[node] = tuple(dims)
            if source in invariant:
                invariant.add(node)
            continue
        if node is region.reduction:
            source = cast("Node", node.args[0])
            if axes[source] != ("row", "feature"):
                raise InapplicableStateUpdate(
                    "sum does not reduce original feature axis"
                )
            axes[node] = ("row",)
            continue
        if _pointwise(node):
            dims: list[str | None] = [None] * value.ndim
            fixed = True
            for arg in node.all_input_nodes:
                if not isinstance(arg.meta.get("val"), torch.Tensor):
                    if feature in _ancestors(arg):
                        fixed = False
                    continue
                if arg not in axes:
                    raise InapplicableStateUpdate("unproved pointwise source axes")
                offset = value.ndim - len(axes[arg])
                if offset < 0:
                    raise InapplicableStateUpdate("pointwise broadcast rank")
                for i, axis in enumerate(axes[arg]):
                    if axis is not None and dims[offset + i] not in (None, axis):
                        raise InapplicableStateUpdate("pointwise mixes logical axes")
                    if axis is not None:
                        dims[offset + i] = axis
                fixed &= arg in invariant
            axes[node] = tuple(dims)
            if fixed:
                invariant.add(node)
    return FeatureInvariance(region, tuple(axes.items()), frozenset(invariant))
