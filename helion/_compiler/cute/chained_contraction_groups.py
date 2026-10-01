"""Physical common-left grouping without changing source arithmetic.

Independent contractions A@B and A@D can issue as A@[B,D]. Each member keeps
its own explicit accumulator and logical result view. This is concatenation,
not reassociation: no reduction dimension or source narrowing is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import _tracing_ops
from . import chained_matmul as chain
from .chained_tcgen_stage import StageGeometry

if TYPE_CHECKING:
    from torch.fx import Node

    from .chained_matmul import ChainedMatmulPlan


@dataclass(frozen=True)
class ContractionGroup:
    stages: tuple[int, ...]
    geometries: tuple[StageGeometry, ...]

    @property
    def offsets(self) -> tuple[int, ...]:
        offset = 0
        result = []
        for geometry in self.geometries:
            result.append(offset)
            offset += geometry.physical[1]
        return tuple(result)

    @property
    def physical(self) -> tuple[int, int, int]:
        m, _, k = self.geometries[0].physical
        return m, sum(item.physical[1] for item in self.geometries), k


def _physical_left(
    node: Node, geometry: StageGeometry
) -> tuple[Node, bool, tuple[torch.dtype, ...]]:
    """Commute casts with rank-two views while retaining every rounding step."""
    operand = cast("Node", node.args[1 if geometry.transpose else 0])
    transpose = geometry.transpose
    casts: list[torch.dtype] = []
    while True:
        if operand.target is _tracing_ops._new_var:
            operand = cast("Node", operand.args[0])
        elif operand.target is torch.ops.prims.convert_element_type.default:
            if (
                cast("Node", operand.args[0]).meta["val"].dtype
                != operand.meta["val"].dtype
            ):
                casts.append(operand.meta["val"].dtype)
            operand = cast("Node", operand.args[0])
        elif (
            operand.target is torch.ops.aten.t.default
            or (
                operand.target is torch.ops.aten.permute.default
                and operand.args[1] in ((1, 0), [1, 0])
            )
            or (
                operand.target is torch.ops.aten.transpose.int
                and cast("int", operand.args[1]) % 2 != cast("int", operand.args[2]) % 2
            )
        ):
            operand = cast("Node", operand.args[0])
            transpose = not transpose
        else:
            return operand, transpose, tuple(casts)


def contraction_groups(
    plan: ChainedMatmulPlan, geometries: tuple[StageGeometry, ...]
) -> tuple[ContractionGroup, ...]:
    """Group adjacent independent dots with the exact same physical A image."""
    assert len(plan.dots) == len(geometries)
    assert plan.region is not None
    positions = {node: index for index, node in enumerate(plan.region.nodes)}
    collectives = (*plan.region.scans, *plan.region.reductions)
    groups: list[ContractionGroup] = []
    for stage, geometry in enumerate(geometries):
        singleton = ContractionGroup((stage,), (geometry,))
        if not groups:
            groups.append(singleton)
            continue
        previous = groups[-1]
        first = previous.stages[0]
        begin, end = positions[plan.dots[first]], positions[plan.dots[stage]]
        group_dots = {plan.dots[index] for index in previous.stages}
        if (
            plan.operand_dtype(first) != plan.operand_dtype(stage)
            or bool(group_dots & chain._ancestors(plan.dots[stage]))
            or any(begin < positions[node] < end for node in collectives)
        ):
            groups.append(singleton)
            continue
        # A singleton may exchange its physical M/N ownership to expose a
        # common left image. Previously grouped members already share one
        # view, so retain that choice when extending their concatenation.
        previous_choices = (
            tuple(
                ContractionGroup(previous.stages, (item,))
                for item in _orientations(previous.geometries[0])
            )
            if len(previous.stages) == 1
            else (previous,)
        )
        combined = None
        for choice in previous_choices:
            for candidate in _orientations(geometry):
                m, n, k = choice.physical
                next_m, next_n, next_k = candidate.physical
                if (
                    (m, k) == (next_m, next_k)
                    and n + next_n <= 256
                    and _physical_left(plan.dots[first], choice.geometries[0])
                    == _physical_left(plan.dots[stage], candidate)
                ):
                    combined = ContractionGroup(
                        (*previous.stages, stage), (*choice.geometries, candidate)
                    )
                    break
            if combined is not None:
                break
        if combined is None:
            groups.append(singleton)
        else:
            groups[-1] = combined
    return tuple(groups)


def _orientations(geometry: StageGeometry) -> tuple[StageGeometry, ...]:
    alternative = StageGeometry(geometry.logical, not geometry.transpose)
    m, n, _ = geometry.logical
    rows, columns = (n, m) if alternative.transpose else (m, n)
    return (geometry, alternative) if rows <= 128 and columns <= 256 else (geometry,)
