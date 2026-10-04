"""Early geometry/lifetime proof for row sums followed by operand preparation.

This does not authorize fusion: the caller must still reject an ordinary sparse
reduction schedule and prove useful original-node/exact-coordinate retention,
typed expressions, domains, native targets and full-warp participation. The
original frame is never changed. All affected writes (including later sums)
start early, and external reads remain live through the complete fused fill.
The caller revalidates this candidate against the same frame before emission
and preserves every publication fence and whole-slot ownership boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
from typing import TYPE_CHECKING

import torch

from .chained_collectives import Collective
from .chained_collectives import classify_collective
from .chained_mma_selection import warp_mma_shape

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_frame import PreparationBuffer
    from .chained_preparation_frame import PreparationFrame
    from .chained_preparation_frame import PreparationStage
    from .warp_specialized_plan import SharedBufferRegion


@dataclass(frozen=True)
class RowCollectiveRegion:
    original: SharedBufferRegion
    required: SharedBufferRegion


@dataclass(frozen=True)
class RowCollectiveGroup:
    first_event: int
    stop_event: int
    collectives: tuple[Collective, ...]
    buffers: tuple[PreparationBuffer, ...]
    stage: PreparationStage
    shape: tuple[int, int]
    affected_regions: tuple[RowCollectiveRegion, ...]
    read_regions: tuple[RowCollectiveRegion, ...]


def _shape(
    node: Node, shapes: Mapping[Node, tuple[int, ...]]
) -> tuple[int, ...] | None:
    value = node.meta.get("val")
    shape = shapes.get(node)
    if (
        not isinstance(value, torch.Tensor)
        or type(shape) is not tuple
        or len(shape) != value.ndim
        or any(type(size) is not int or size <= 0 for size in shape)
        or any(
            type(old) is int and old != new
            for old, new in zip(value.shape, shape, strict=True)
        )
    ):
        return None
    return shape


def plan_row_collective_group(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    first_event: int,
    shapes: Mapping[Node, tuple[int, ...]],
) -> RowCollectiveGroup | None:
    """Find adjacent axis-one FP32 sums and a complete, equal-row warp fill.

    Shape resolution is explicit: no DeviceFunction or CUDA context is needed.
    Sparse/dense selection is deliberately a late proof, not inferred from the
    sum operator. Returning a candidate neither replaces actions nor changes
    their arithmetic, reduction tree, masks, casts, stores or barriers.
    """
    # These modules also depend on stage geometry; defer imports so the stage
    # emitter can use this planner without a module-initialization cycle.
    from .chained_matmul import _ancestors
    from .chained_prepared_groups import _valid_frame

    if (
        type(first_event) is not int
        or not 0 <= first_event < len(frame.actions)
        or not _valid_frame(frame)
        or plan.region is None
        or plan.region.graph is not frame.cut.region.graph
        or plan.region.nodes != frame.cut.region.nodes
        or plan.region.nodes != tuple(plan.region.graph.nodes)
        or tuple(item.node for item in frame.cut.region.contractions) != plan.dots
    ):
        return None
    by_name = {buffer.name: buffer for buffer in frame.buffers}
    regions = {region.name: region for region in frame.layout.regions}
    collectives: list[Collective] = []
    buffers = []
    shape = None
    event = first_event
    while event < len(frame.actions):
        action = frame.actions[event]
        if action.kind != "collective":
            break
        if len(action.nodes) != 1 or len(action.writes) != 1 or action.stages:
            return None
        operation = classify_collective(action.nodes[0])
        if operation is None or operation.kind != "sum" or operation.axis != 1:
            return None
        source_shape = _shape(operation.source, shapes)
        result_shape = _shape(operation.node, shapes)
        if (
            source_shape is None
            or len(source_shape) != 2
            or result_shape not in ((source_shape[0],), (source_shape[0], 1))
            or shape is not None
            and source_shape != shape
            or operation.node not in frame.cut.preparation
            or operation.node not in plan.region.reductions
        ):
            return None
        buffer = by_name[action.writes[0]]
        if (
            buffer.kind != "collective"
            or buffer.node is not operation.node
            or buffer.dtype != torch.float32
            or buffer.shape != result_shape
            or regions[buffer.name].live_from != event
            or regions[buffer.name].byte_size < math.prod(buffer.shape) * 4
        ):
            return None
        collectives.append(operation)
        buffers.append(buffer)
        shape = source_shape
        event += 1
    if not collectives or shape is None or event == len(frame.actions):
        return None
    nodes = {item.node for item in collectives}
    if len(nodes) != len(collectives) or any(
        nodes.intersection(_ancestors(item.source)) for item in collectives
    ):
        return None
    fill = frame.actions[event]
    stages = tuple(stage for stage in frame.stages if stage.group.stages == fill.stages)
    if fill.kind != "fill" or len(stages) != 1:
        return None
    stage = stages[0]
    group = stage.group
    if (
        not group.stages
        or len(group.stages) != len(group.geometries)
        or len(set(group.stages)) != len(group.stages)
        or any(
            type(index) is not int or not 0 <= index < len(plan.dots)
            for index in group.stages
        )
        or group.stages[0] not in plan.warp_mma_stages
        or fill.nodes != tuple(plan.dots[index] for index in group.stages)
        or any(
            node not in frame.cut.preparation or node.args[2] is not None
            for node in fill.nodes
        )
        or any(
            geometry.logical != plan.shapes[index]
            for index, geometry in zip(group.stages, group.geometries, strict=True)
        )
        or stage.shape != warp_mma_shape(group.geometries[0], group)
        or (stage.shape[0], stage.shape[2]) != shape
        or any(
            (geometry.physical[1], geometry.physical[2]) != shape
            for geometry in group.geometries
        )
        or fill.writes != (stage.a.name, stage.b.name)
        or event + 1 >= len(frame.actions)
    ):
        return None
    mma = frame.actions[event + 1]
    if (
        mma.kind != "mma"
        or mma.nodes != fill.nodes
        or mma.stages != fill.stages
        or set(mma.reads) != set(fill.writes)
    ):
        return None
    dtype = plan.operand_dtype(group.stages[0])
    if dtype not in (torch.bfloat16, torch.float16) or any(
        plan.operand_dtype(index) != dtype for index in group.stages
    ):
        return None
    for name, kind, target_shape in (
        (stage.a.name, "a", (stage.shape[0], stage.shape[2])),
        (stage.b.name, "b", (stage.shape[1], stage.shape[2])),
    ):
        buffer = by_name[name]
        region = regions[name]
        if (
            buffer.kind != kind
            or buffer.node is not None
            or buffer.dtype != dtype
            or buffer.shape != target_shape
            or region.byte_size < math.prod(target_shape) * dtype.itemsize
            or region.live_from != event
            or region.live_until < mma.publication_event
        ):
            return None
    stop = event + 1
    actions = frame.actions[first_event:stop]
    write_names = tuple(name for action in actions for name in action.writes)
    if len(set(write_names)) != len(write_names):
        return None
    collective_names = {buffer.name for buffer in buffers}
    external_reads = {
        name
        for action in actions
        for name in action.reads
        if name not in collective_names
    }
    if any(regions[name].live_from >= first_event for name in external_reads):
        return None
    affected = tuple(
        RowCollectiveRegion(
            regions[name], replace(regions[name], live_from=first_event)
        )
        for name in write_names
    )
    reads = tuple(
        RowCollectiveRegion(
            region, replace(region, live_until=max(region.live_until, stop))
        )
        for region in frame.layout.regions
        if region.name in external_reads
    )
    required = {item.original.name: item.required for item in (*affected, *reads)}
    extended = tuple(
        required.get(region.name, region) for region in frame.layout.regions
    )
    if any(
        left.overlaps_storage(right) and left.overlaps_lifetime(right)
        for index, left in enumerate(extended)
        for right in extended[index + 1 :]
    ):
        return None
    return RowCollectiveGroup(
        first_event,
        stop,
        tuple(collectives),
        tuple(buffers),
        stage,
        (shape[0], shape[1]),
        affected,
        reads,
    )
