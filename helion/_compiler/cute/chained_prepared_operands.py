"""Exact prepared images that already constitute a complete TCgen operand.

This proof changes neither frame allocation nor action order. A future emitter
may publish an admitted image through its native logical view and consume that
same view after slot readiness. The image remains live until whole-slot release;
its allocation must never become recurrence staging scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
from typing import Literal

import torch

from ...language.matmul_ops import dot
from .chained_contraction_groups import ContractionGroup
from .chained_late_rhs import _layout_bytes
from .chained_tcgen05 import _layout
from .chained_tcgen_stage import stage_geometry
from .contraction_region import _domain

if TYPE_CHECKING:
    from torch.fx import Node

    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_frame import PreparationBuffer
    from .chained_preparation_frame import PreparationFrame
    from .chained_recurrence_workspace import RecurrenceWorkspace
    from .chained_tcgen_stage import StageGeometry
    from .contraction_region import ContractionSpec
    from .warp_specialized_plan import SharedBufferRegion


@dataclass(frozen=True)
class PreparedOperand:
    buffer: PreparationBuffer
    region: SharedBufferRegion
    role: Literal["b"]
    stage: int
    operand_index: int
    geometry: StageGeometry
    physical_shape: tuple[int, int]
    native_layout: tuple[str, ...]


def _matches_contraction(spec: ContractionSpec) -> bool:
    # Recheck exact retained facts, not symbolic-domain equivalences that were
    # already admitted inside the original compiler environment.
    node = spec.node
    if (
        node.op != "call_function"
        or node.target is not dot
        or node.args != (spec.lhs, spec.rhs, spec.accumulator, spec.requested_out_dtype)
        or node.kwargs
    ):
        return False
    if any(
        not isinstance(operand.meta.get("val"), torch.Tensor)
        or operand.meta["val"].dtype != dtype
        or _domain(operand.meta["val"]) != domain
        for operand, dtype, domain in (
            (spec.lhs, spec.operand_dtypes[0], spec.lhs_domain),
            (spec.rhs, spec.operand_dtypes[1], spec.rhs_domain),
            (node, spec.result_dtype, spec.result_domain),
        )
    ):
        return False
    return spec.accumulator is None or (
        isinstance(spec.accumulator.meta.get("val"), torch.Tensor)
        and spec.accumulator.meta["val"].dtype == spec.accumulator_dtype
    )


def plan_prepared_operands(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    recurrence: RecurrenceWorkspace,
) -> tuple[PreparedOperand, ...] | None:
    """Prove complete singleton B images without consulting device context.

    Return ``None`` for inconsistent input plans, and an empty tuple when the
    admitted plans have no supported image. Initial capability requires the
    logical image to have exactly the physical (row,K) coordinates; transpose
    views, padded operands, casts and concatenated groups remain ordinary fills.
    Native-view statements come from the existing stage layout policy; merely
    returning them does not enable an emitter or modify the input plans.
    """
    region, loop, cut = plan.region, plan.loop, frame.cut
    if (
        region is None
        or loop is None
        or recurrence.cut is not cut
        or cut.region.graph is not region.graph
        or loop.region.graph is not region.graph
        or cut.region.nodes != region.nodes
        or loop.region.nodes != region.nodes
        or cut.region.contractions != region.contractions
        or loop.region.contractions != region.contractions
        or cut.carries != region.carries
        or loop.region.carries != region.carries
        or region.nodes != tuple(region.graph.nodes)
        or loop.storage_key != cut.storage_proof_key
        or plan.strategy != "tcgen05_tmem"
        or plan.dots != tuple(spec.node for spec in region.contractions)
        or len(plan.shapes) != len(plan.dots)
        or not all(_matches_contraction(spec) for spec in region.contractions)
        or not frame.actions
        or frame.actions[-1].kind != "ready"
    ):
        return None
    groups = plan.contraction_groups
    if groups is None:
        geometries = tuple(stage_geometry(shape) for shape in plan.shapes)
        if any(geometry is None for geometry in geometries):
            return None
        groups = tuple(
            ContractionGroup((index,), (geometry,))
            for index, geometry in enumerate(geometries)
            if geometry is not None
        )
    if tuple(index for group in groups for index in group.stages) != tuple(
        range(len(plan.dots))
    ) or any(
        not group.stages
        or len(group.stages) != len(group.geometries)
        or any(
            geometry.logical != plan.shapes[index]
            or stage_geometry(geometry.logical) is None
            for index, geometry in zip(group.stages, group.geometries, strict=True)
        )
        for group in groups
    ):
        return None
    recurrent = set(cut.recurrence)
    expected = tuple(
        group for group in groups if plan.dots[group.stages[0]] in recurrent
    )
    if tuple(stage.group for stage in recurrence.stages) != expected or any(
        set(stage.group.stages) & plan.warp_mma_stages
        or stage.read_event != 2 * stage.group.stages[0]
        or stage.publication_event != stage.read_event + 1
        or any(plan.dots[index] not in recurrent for index in stage.group.stages)
        for stage in recurrence.stages
    ):
        return None
    images = {image.node: image for image in cut.images}
    buffers = {
        buffer.node: buffer for buffer in frame.buffers if buffer.kind == "frontier"
    }
    regions = {item.name: item for item in frame.layout.regions}
    bindings = dict(recurrence.bindings)
    if (
        len(images) != len(cut.images)
        or set(buffers) != set(images)
        or len(buffers) != sum(buffer.kind == "frontier" for buffer in frame.buffers)
        or len(regions) != len(frame.layout.regions)
    ):
        return None
    for node, buffer in buffers.items():
        assert node is not None
        value, image = node.meta.get("val"), images[node]
        allocation = regions.get(buffer.name)
        actions = tuple(
            action for action in frame.actions if buffer.name in action.writes
        )
        if (
            not isinstance(value, torch.Tensor)
            or buffer.dtype != image.dtype
            or value.dtype != image.dtype
            or _domain(value) != image.logical_domain
            or len(buffer.shape) != value.ndim
            or any(type(size) is not int or size <= 0 for size in buffer.shape)
            or any(
                type(old) is int and old != new
                for old, new in zip(value.shape, buffer.shape, strict=True)
            )
            or bindings.get(node) != buffer.name
            or allocation is None
            or allocation.alignment < 128
            or allocation.byte_offset < 0
            or allocation.byte_offset % 128
            or allocation.byte_end > frame.layout.allocated_bytes
            or allocation.byte_size < math.prod(buffer.shape) * buffer.dtype.itemsize
            or allocation.live_until != frame.actions[-1].publication_event
            or len(actions) != 1
            or actions[0].kind != "frontier"
            or actions[0].nodes != (node,)
            or actions[0].writes != (buffer.name,)
            # A grouped producer may reserve its destination before the
            # original action. Reservation is not publication: the original
            # action and ready cut below still establish the readable image.
            or not 0 <= allocation.live_from <= actions[0].event
            or actions[0].publication_event > frame.actions[-1].event
            or any(
                other.name != allocation.name
                and other.overlaps_lifetime(allocation)
                and other.overlaps_storage(allocation)
                for other in frame.layout.regions
            )
        ):
            return None
    candidates: dict[Node, list[PreparedOperand]] = {}
    incompatible: set[Node] = set()
    for stage in recurrence.stages:
        group = stage.group
        for index, geometry in zip(group.stages, group.geometries, strict=True):
            spec = region.contractions[index]
            for role in ("a", "b"):
                operand_index, coordinates = geometry.operand(role, "row", "k")
                operand = (spec.lhs, spec.rhs)[operand_index]
                buffer = buffers.get(operand)
                if buffer is None:
                    continue
                m, n, k = group.physical
                shape = (n, k)
                size = _layout_bytes(shape, 1)
                if (
                    role != "b"
                    or len(group.stages) != 1
                    or coordinates != ("row", "k")
                    or buffer.dtype not in (torch.bfloat16, torch.float16)
                    or spec.operand_dtypes != (buffer.dtype, buffer.dtype)
                    or buffer.shape != shape
                    or size is None
                    or size != math.prod(buffer.shape) * buffer.dtype.itemsize
                    or size != regions[buffer.name].byte_size
                ):
                    incompatible.add(operand)
                    continue
                dtype = (
                    "cutlass.BFloat16"
                    if buffer.dtype == torch.bfloat16
                    else "cutlass.Float16"
                )
                candidate = PreparedOperand(
                    buffer,
                    regions[buffer.name],
                    "b",
                    index,
                    operand_index,
                    geometry,
                    shape,
                    tuple(_layout(buffer.name, shape, 1, dtype)),
                )
                previous = candidates.setdefault(operand, [])
                if previous and previous[0].native_layout != candidate.native_layout:
                    incompatible.add(operand)
                previous.append(candidate)
    return tuple(
        sorted(
            (
                item
                for image, uses in candidates.items()
                if image not in incompatible
                for item in uses
            ),
            key=lambda item: item.stage,
        )
    )
