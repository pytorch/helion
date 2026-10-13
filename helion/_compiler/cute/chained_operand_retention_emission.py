"""Late expression proofs and early-write reservations for typed operand reuse.

No source, alias, allocation or synchronization is installed here. An accepted
image is still published only after its original complete fill/MMA action and
role fence. The caller must rebind all native/island/frontier plans after packing
the returned reservations; a reserved lifetime never proves publication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from .chained_contraction_groups import ContractionGroup
from .chained_frontier_groups import plan_frontier_group
from .chained_matmul import _Expression
from .chained_matmul import _operand_domain
from .chained_matmul import _shape
from .chained_matmul import _UnsupportedChain
from .chained_mma_selection import warp_mma_shape
from .chained_operand_retention import discover_operand_retention
from .chained_pointwise_residency import _finite_consumers
from .chained_prepared_groups import _valid_frame
from .chained_register_islands import plan_register_islands
from .chained_tcgen_stage import stage_geometry
from .warp_specialized_plan import SharedBufferRequest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_operand_retention import OperandRetentionCandidate
    from .chained_preparation_frame import PreparationFrame
    from .chained_preparation_frame import PreparationStage
    from .chained_register_islands import RegisterIsland


def _islands(plan: ChainedMatmulPlan, enabled: bool) -> tuple[RegisterIsland, ...]:
    if type(enabled) is not bool:
        raise TypeError("register-island selection must be bool")
    if not enabled or plan.region is None:
        return ()
    groups = plan.contraction_groups
    if groups is None:
        geometries = tuple(stage_geometry(shape) for shape in plan.shapes)
        if any(geometry is None for geometry in geometries):
            return ()
        groups = tuple(
            ContractionGroup((index,), (geometry,))
            for index, geometry in enumerate(geometries)
            if geometry is not None
        )
    return plan_register_islands(
        plan.region,
        groups,
        {
            node: _shape(node)
            for node in plan.region.nodes
            if isinstance(node.meta.get("val"), torch.Tensor)
        },
        fast_math=True,
        entry_boundaries=(
            frozenset(entry.node for entry in plan.pointwise_cache.entries)
            if plan.pointwise_cache is not None
            else frozenset()
        ),
    )


def operand_retention_reservations(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    *,
    register_islands: bool,
    frontier_groups: bool,
) -> tuple[SharedBufferRequest, ...]:
    """Reserve every potentially earlier write, even if late CSE falls back.

    ``register_islands`` denotes the caller's effective request (including its
    original fast-math policy). Discovery alone is not permission to emit it.
    Frontier prefixes must be safe in the ORIGINAL frame, including actual
    publication of their sources; equal shapes alone do not make a group.
    """
    if type(frontier_groups) is not bool or type(register_islands) is not bool:
        raise TypeError("retention schedule selections must be bool")
    if not _valid_frame(frame) or plan.region != frame.cut.region:
        raise _UnsupportedChain("invalid operand-retention reservation frame")
    requests: dict[str, SharedBufferRequest] = {}

    def reserve(name: str, first: int) -> None:
        region = frame.layout.region(name)
        old = requests.get(name)
        requests[name] = SharedBufferRequest(
            name,
            region.byte_size,
            region.alignment,
            min(first, region.live_from, old.live_from if old else first),
            region.live_until,
        )

    for island in _islands(plan, register_islands):
        starts = tuple(
            action.event
            for action in frame.actions
            if action.kind == "fill" and action.stages == island.groups[0].stages
        )
        if not starts:
            continue  # A root/recurrence island has no preparation writes.
        if len(starts) != 1:
            raise _UnsupportedChain("ambiguous preparation-island start")
        for node in island.exports:
            buffers = tuple(
                buffer
                for buffer in frame.buffers
                if buffer.kind == "c" and buffer.node is node
            )
            if len(buffers) != 1:
                raise _UnsupportedChain("missing preparation-island export")
            reserve(buffers[0].name, starts[0])
    if frontier_groups:
        stop = 0
        for action in frame.actions:
            if action.event < stop or action.kind != "frontier":
                continue
            group = plan_frontier_group(frame, action.event)
            if group is not None:
                for buffer in group.buffers:
                    reserve(buffer.name, group.first_event)
                stop = group.stop_event
    return tuple(requests.values())


def _publications(frame: PreparationFrame) -> dict[int, dict[Node, str]]:
    buffers = {buffer.name: buffer for buffer in frame.buffers}
    published: dict[Node, str] = {}
    result = {}
    for action in frame.actions:
        result[action.event] = {
            node: name
            for node, name in published.items()
            if frame.layout.region(name).live_from
            <= action.event
            < frame.layout.region(name).live_until
        }
        if action.kind in ("mma", "collective", "cache", "leaf"):
            for name in action.writes:
                node = buffers[name].node
                if node is not None:
                    published[node] = name
    return result


class _References(_Expression):
    def __init__(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        *,
        retained: Node,
    ) -> None:
        super().__init__(cg, plan, boundaries)
        self.retained = retained
        self.coordinates: list[tuple[str, ...]] = []

    def value(self, node: Node, coordinates: tuple[str, ...]) -> str:
        if node is self.retained:
            self.coordinates.append(coordinates)
        return super().value(node, coordinates)


def _in_bounds(
    coordinates: tuple[str, ...],
    shape: tuple[int, ...],
    extents: dict[str, int],
) -> bool:
    # No simplification of fixed-width arithmetic, gather, reshape, or nonlinear
    # index expressions. Original axis exchange/broadcast retain these strings.
    return len(coordinates) == len(shape) and all(
        extent > 0
        and (
            coordinate == "0"
            or coordinate in extents
            and 0 < extents[coordinate] <= extent
        )
        for coordinate, extent in zip(coordinates, shape, strict=True)
    )


def _operands(
    plan: ChainedMatmulPlan, stage: PreparationStage
) -> Iterator[tuple[Node, tuple[str, ...], dict[str, int]]]:
    m, _, k = stage.shape
    for role in ("a", "b"):
        members = tuple(zip(stage.group.stages, stage.group.geometries, strict=True))
        for index, geometry in members[:1] if role == "a" else members:
            operand_index, coordinates = geometry.operand(role, "row", "column")
            yield (
                cast("Node", plan.dots[index].args[operand_index]),
                coordinates,
                {"row": m if role == "a" else geometry.physical[1], "column": k},
            )


def admit_operand_retention(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    candidates: tuple[OperandRetentionCandidate, ...],
    *,
    register_islands: bool = False,
) -> tuple[OperandRetentionCandidate, ...]:
    """Prove exact, in-bounds typed reuse through original scalar expressions.

    The native producer must cover the complete logical image without operand
    padding/domain zero-fill. Leaf masks remain part of the original expression
    and are not rejected or removed. All consumer reads must map directly into
    this image and have an empty original operand domain. Collective/cache
    consumers need a separate schedule proof and currently decline. This does
    not authorize a writer skipped by an island.
    """
    if type(register_islands) is not bool:
        raise TypeError("register-island selection must be bool")
    if plan.region is None or not candidates:
        return ()
    if any(
        item.revision.plan is not plan or item.revision.frame is not frame
        for item in candidates
    ):
        return ()
    shapes = {
        node: _shape(node)
        for node in plan.region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    available = discover_operand_retention(plan, frame, shapes)
    if available is None or candidates != tuple(
        x for x in available if x in candidates
    ):
        return ()
    excluded = {
        stage
        for island in _islands(plan, register_islands)
        for group in island.groups
        for stage in group.stages
    }
    published = _publications(frame)
    stages = {stage.group.stages: stage for stage in frame.stages}
    accepted = []
    boundaries = {*plan.dots, *plan.region.scans, *plan.region.reductions}
    if plan.pointwise_cache is not None:
        boundaries.update(entry.node for entry in plan.pointwise_cache.entries)
    for item in candidates:
        stage = stages[item.group.stages]
        if (
            excluded.intersection(item.group.stages)
            or stage.shape != warp_mma_shape(stage.group.geometries[0], stage.group)
            or not _finite_consumers(
                item.node, boundaries, within=frozenset(frame.cut.preparation)
            )
        ):
            continue
        fill = next(
            action
            for action in frame.actions
            if action.kind == "fill" and action.stages == item.group.stages
        )
        row, column = "row", "column"
        _, coordinates = item.geometry.operand(item.role, row, column)
        expected = (row, column) if item.logical_modes == (0, 1) else (column, row)
        try:
            producer = _References(cg, plan, published[fill.event], retained=item.node)
            producer.coordinate_names.update((row, column))
            producer.value(item.operand, coordinates)
            if (
                not producer.coordinates
                or any(coords != expected for coords in producer.coordinates)
                or _operand_domain(cg, item.operand, coordinates, plan)
            ):
                continue
            valid = True
            for event in item.consumer_events:
                action = frame.actions[event]
                if action.kind == "frontier":
                    buffer = next(
                        b for b in frame.buffers if b.name == action.writes[0]
                    )
                    output = action.nodes[0]
                    coords = tuple(f"coord_{axis}" for axis in range(len(buffer.shape)))
                    roots = (
                        (output, coords, dict(zip(coords, buffer.shape, strict=True))),
                    )
                elif action.kind == "fill":
                    roots = tuple(_operands(plan, stages[action.stages]))
                else:
                    valid = False
                    break
                visited = False
                for node, coords, extents in roots:
                    expression = _References(
                        cg,
                        plan,
                        {**published[event], item.node: "chain_retained_proof"},
                        retained=item.node,
                    )
                    expression.coordinate_names.update(extents)
                    expression.value(node, coords)
                    visited |= bool(expression.coordinates)
                    # Original domains are not replaced by boundary extents.
                    # Keep the first capability deliberately stricter than
                    # value equivalence under a surrounding padding predicate.
                    if expression.coordinates and _operand_domain(
                        cg, node, coords, plan
                    ):
                        valid = False
                        break
                    if any(
                        not _in_bounds(read, item.logical_shape, extents)
                        for read in expression.coordinates
                    ):
                        valid = False
                        break
                if not valid or not visited:
                    valid = False
                    break
            if valid:
                accepted.append(item)
        except _UnsupportedChain:
            continue
    return tuple(accepted)
