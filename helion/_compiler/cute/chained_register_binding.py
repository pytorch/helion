"""Late binding of register-island candidates to an owned preparation frame.

No allocation, emission, slot release or publication is performed here. The
caller invokes this immediately before the first fill of the selected span,
after all native/TMA frame rebinding. Existing entry publication and whole-role
barriers remain mandatory; neither graph support nor a frame offset alone is
permission to read a register value at different logical coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import TYPE_CHECKING

import torch

from ..compile_environment import CompileEnvironment
from .chained_matmul import _Expression
from .chained_matmul import _operand_domain
from .chained_matmul import _shape
from .chained_matmul import _UnsupportedChain
from .chained_prepared_groups import _valid_frame
from .chained_register_islands import RegisterImage
from .chained_register_islands import plan_register_islands
from .chained_register_islands import register_island_matches
from .chained_tcgen_stage import stage_geometry

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_island_publication import IslandConsumerPublication
    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_frame import PreparationFrame
    from .chained_register_islands import RegisterIsland
    from .warp_specialized_plan import SharedBufferRegion


@dataclass(frozen=True)
class RegisterExport:
    node: Node
    name: str
    shape: tuple[int, int]
    region: SharedBufferRegion


@dataclass(frozen=True)
class RegisterOrigin:
    node: Node
    # Original logical row and column: base + stride * component index.
    axes: tuple[tuple[int, int], tuple[int, int]]
    image: RegisterImage


def _image_facts(
    island: RegisterIsland, origins: tuple[RegisterOrigin, ...]
) -> tuple[object, ...]:
    return (
        tuple(
            (
                value.node,
                value.dtype,
                value.shape,
                value.defined_at,
                value.last_use,
                value.consumers,
                value.support_rows,
                tuple((image.node, image.tiles) for image in value.images),
            )
            for value in island.values
        ),
        tuple(
            (item.node, item.axes, item.image.node, item.image.tiles)
            for item in origins
        ),
    )


def _frame_facts(frame: PreparationFrame) -> tuple[object, ...]:
    return (
        frame.cut,
        frame.layout,
        frame.buffers,
        frame.actions,
        frame.stages,
        frame.frontier_order,
        frame.peak_live_bytes,
    )


@dataclass(frozen=True)
class BoundPreparationIsland:
    island: RegisterIsland
    plan: ChainedMatmulPlan
    execution: ChainedExecution
    frame: PreparationFrame
    first_event: int
    stop_event: int
    exports: tuple[RegisterExport, ...]
    origins: tuple[RegisterOrigin, ...]
    published: tuple[tuple[Node, str], ...]
    frame_facts: tuple[object, ...]
    image_facts: tuple[object, ...]
    island_input: IslandConsumerPublication | None = None
    input_identity: int | None = None

    def coordinates(self, node: Node | RegisterImage, prefix: str) -> tuple[str, str]:
        """Shared bind/emission spelling: differing origins remain observable."""
        matches = tuple(
            item
            for item in self.origins
            if (
                item.image == node
                if isinstance(node, RegisterImage)
                else item.node is node
            )
        )
        if len(matches) != 1:
            raise _UnsupportedChain("missing or ambiguous register image coordinates")
        origin = matches[0]
        row, column = origin.axes
        return (
            f"({row[0]} + {row[1]} * {prefix}_warp + {prefix}_coords[{prefix}_index][0])",
            f"({column[0]} + {column[1]} * {prefix}_warp + {prefix}_coords[{prefix}_index][1])",
        )

    def operand_image(self, stage: int, argument: int) -> RegisterImage:
        selected = tuple(
            tuple(issue for issue in component.issues if issue.stage == stage)
            for component in self.island.components
        )
        if (
            argument not in (0, 1)
            or not selected
            or any(len(items) != 1 for items in selected)
        ):
            raise _UnsupportedChain("missing or invalid register operand use")
        issues = tuple(items[0] for items in selected)
        node = issues[0].node.args[argument]
        axes = (0, 2) if argument == 0 else (2, 1)
        tiles = tuple(
            (index, (issue.origins[axes[0]], issue.origins[axes[1]]))
            for index, issue in enumerate(issues)
        )
        matches = tuple(
            item.image
            for item in self.origins
            if item.node is node and item.image.tiles == tiles
        )
        if len(matches) != 1:
            raise _UnsupportedChain("missing or ambiguous register operand image")
        return matches[0]

    def matches(
        self,
        plan: ChainedMatmulPlan,
        frame: PreparationFrame,
        execution: ChainedExecution,
        boundaries: dict[Node, str],
    ) -> bool:
        """Check the same late binding before emitting, without changing state."""
        if (
            plan is not self.plan
            or frame is not self.frame
            or execution != self.execution
            or _frame_facts(frame) != self.frame_facts
            or dict(self.published) != boundaries
            or not CompileEnvironment.has_current()
            or CompileEnvironment.current().settings.fast_math is not True
            or plan.region is not self.island.revision.region
            or self.image_facts != _image_facts(self.island, self.origins)
            or self.input_identity
            != (id(self.island_input) if self.island_input is not None else None)
            or self.island_input is not None
            and not self.island_input.matches()
        ):
            return False
        shapes = {node: _shape(node) for node, _ in self.island.revision.shapes}
        return register_island_matches(
            self.island,
            plan.region,
            self.island.revision.groups,
            shapes,
            entry_boundaries=self.island.revision.entry_boundaries,
        )


def bind_preparation_island(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    island: RegisterIsland,
    execution: ChainedExecution,
    boundaries: dict[Node, str],
    *,
    island_input: IslandConsumerPublication | None = None,
) -> BoundPreparationIsland | None:
    """Prove one complete ordered replacement; retain all original shared views.

    Selected fill A/B and nonexport C storage will be unused by this replacement.
    Every other interval live during the span is protected against early export
    zeroing, including entries, unrelated workspaces, and outstanding readers.
    The frame's existing action publication fences must already have completed
    for each published entry. This function cannot establish runtime completion.
    """
    if (
        not CompileEnvironment.has_current()
        or CompileEnvironment.current().settings.fast_math is not True
        or plan.region is not island.revision.region
        or plan.dots != tuple(spec.node for spec in island.revision.region.contractions)
        # The cut is recollected by the existing planner. Its record can be a
        # distinct object, but graph/node identities and every port/contract
        # must agree, not merely shapes or a graph number.
        or frame.cut.region != plan.region
        or plan.loop is None
        or not _valid_frame(frame)
        or not island.groups
        or not island.components
        or len(island.components) * 32 > execution.threads
    ):
        return None
    groups = island.revision.groups
    if plan.contraction_groups is not None:
        if plan.contraction_groups != groups:
            return None
    elif any(
        group.stages != (index,)
        or group.geometries != (stage_geometry(plan.shapes[index]),)
        for index, group in enumerate(groups)
    ):
        return None
    shapes = {node: _shape(node) for node, _ in island.revision.shapes}
    if not register_island_matches(
        island,
        plan.region,
        groups,
        shapes,
        entry_boundaries=island.revision.entry_boundaries,
    ) or island not in plan_register_islands(
        plan.region,
        groups,
        shapes,
        fast_math=True,
        entry_boundaries=island.revision.entry_boundaries,
        multi_image=island.revision.multi_image,
    ):
        return None
    sequence = tuple(issue.stage for issue in island.components[0].issues)
    if any(
        tuple(issue.stage for issue in component.issues) != sequence
        for component in island.components
    ):
        return None
    origins = []
    for value in island.values:
        for image in value.images:
            tiles = dict(image.tiles)
            if (
                image.node is not value.node
                or len(value.shape) != 2
                or tuple(tiles) != tuple(range(len(island.components)))
                or len(tiles) != len(image.tiles)
            ):
                return None
            axes = []
            for axis in range(2):
                base = tiles[0][axis]
                stride = tiles[1][axis] - base if len(tiles) > 1 else 0
                if any(tiles[index][axis] != base + stride * index for index in tiles):
                    return None
                axes.append((base, stride))
            origins.append(RegisterOrigin(value.node, (axes[0], axes[1]), image))
    starts = [
        index
        for index, action in enumerate(frame.actions)
        if action.kind == "fill" and action.stages == island.groups[0].stages
    ]
    if len(starts) != 1:
        return None
    start = starts[0]
    stop = start + 2 * len(island.groups)
    if island_input is not None:
        item = island_input.candidate
        retained = item.retained_input
        retention = item.pipeline.operand_retention
        if (
            not island_input.matches()
            or island_input.consumed
            or item.cg is not cg
            or item.plan is not plan
            or item.bound.frame is not frame
            or item.bound.execution != execution
            or item.bound.stop_event != start
            or item.consumer != sequence[0]
            or len(sequence) != 2
            or any(
                sum(source is item.operand for source in issue.node.args[:2]) != 1
                for issue in island.components[0].issues
            )
            or item.operand not in island.entries
            or boundaries is not item.boundary_owner
            or tuple(boundaries.items()) != island_input.outputs
            or retained is None
            or retention is None
            or retained[1].publication_event != start + 2
            or not start + 2 < stop
            or any(
                start < other.publication_event <= stop and other is not retained[1]
                for other in retention.candidates
            )
        ):
            return None
        # A genuine published operand can itself be an MMA port. The graph
        # planner correctly stops at this boundary; bind its actual per-use
        # images without traversing the omitted producer expressions again.
        for argument in (0, 1):
            for ordinal, issue in enumerate(island.components[0].issues):
                if issue.node.args[argument] is not item.operand:
                    continue
                axes = (0, 2) if argument == 0 else (2, 1)
                image = RegisterImage(
                    item.operand,
                    tuple(
                        (
                            index,
                            (
                                component.issues[ordinal].origins[axes[0]],
                                component.issues[ordinal].origins[axes[1]],
                            ),
                        )
                        for index, component in enumerate(island.components)
                    ),
                )
                if any(origin.image == image for origin in origins):
                    continue
                tiles = dict(image.tiles)
                affine = tuple(
                    (
                        tiles[0][axis],
                        tiles[1][axis] - tiles[0][axis] if len(tiles) > 1 else 0,
                    )
                    for axis in range(2)
                )
                if any(
                    tiles[index][axis] != affine[axis][0] + affine[axis][1] * index
                    for index in tiles
                    for axis in range(2)
                ):
                    return None
                origins.append(
                    RegisterOrigin(item.operand, (affine[0], affine[1]), image)
                )
        if (
            len({origin.image for origin in origins if origin.node is item.operand})
            != 2
        ):
            return None
    actions = frame.actions[start:stop]
    if len(actions) != 2 * len(island.groups):
        return None
    removed = set()
    for index, group in enumerate(island.groups):
        stage = next((stage for stage in frame.stages if stage.group == group), None)
        fill, mma = actions[index * 2 : index * 2 + 2]
        nodes = tuple(plan.dots[stage] for stage in group.stages)
        if (
            stage is None
            or group.stages[0] not in plan.warp_mma_stages
            or any(node not in frame.cut.preparation for node in nodes)
            or fill.kind != "fill"
            or mma.kind != "mma"
            or fill.stages != group.stages
            or mma.stages != group.stages
            or fill.nodes != nodes
            or mma.nodes != nodes
            or fill.source_stage != group.stages[0]
            or mma.source_stage != group.stages[0]
            or fill.writes != (stage.a.name, stage.b.name)
            or set(mma.reads) != set(fill.writes)
            or set(mma.writes)
            != {
                buffer.name
                for buffer in frame.buffers
                if buffer.kind == "c" and buffer.node in nodes
            }
        ):
            return None
        removed.update(fill.writes)
        removed.update(mma.writes)
    if any(plan.dots[stage] in boundaries for stage in sequence):
        return None
    for entry in island.entries:
        name = boundaries.get(entry)
        published = island_input is not None and entry is island_input.candidate.operand
        buffer = next(
            (
                buffer
                for buffer in frame.buffers
                if buffer.name == name
                and (
                    buffer.node is entry
                    or island_input is not None
                    and published
                    and frame.layout.region(buffer.name)
                    is (
                        island_input.candidate.stage.a
                        if island_input.candidate.role == "a"
                        else island_input.candidate.stage.b
                    )
                )
            ),
            None,
        )
        if buffer is None:
            return None
        region = frame.layout.region(buffer.name)
        if not published and not region.live_from < start < region.live_until:
            return None
        if published:
            assert island_input is not None
            # The earlier actual publication is the only authority for a native
            # A/B view whose frame buffer intentionally has no graph Node.
            # Its physical read span is checked again by the existing finalizer.
            if (
                buffer.shape != island_input.candidate.shape
                or buffer.dtype is not entry.meta["val"].dtype
            ):
                return None
            removed.discard(buffer.name)
    exports = []
    for node in island.exports:
        buffers = tuple(
            buffer
            for buffer in frame.buffers
            if buffer.node is node and buffer.kind == "c"
        )
        if len(buffers) != 1:
            return None
        buffer = buffers[0]
        region = frame.layout.region(buffer.name)
        if (
            buffer.dtype is not torch.float32
            or len(buffer.shape) != 2
            or buffer.shape != shapes[node]
            or region.byte_size < math.prod(buffer.shape) * 4
        ):
            return None
        exports.append(
            RegisterExport(
                node, buffer.name, (buffer.shape[0], buffer.shape[1]), region
            )
        )
    if any(
        left.region.overlaps_storage(right.region)
        for left, right in combinations(exports, 2)
    ):
        return None
    export_names = {export.name for export in exports}
    protected = tuple(
        region
        for region in frame.layout.regions
        if region.name not in removed | export_names
        and region.live_from < stop
        and start < region.live_until
    )
    if any(
        export.region.overlaps_storage(region)
        for export in exports
        for region in protected
    ):
        return None
    bound = BoundPreparationIsland(
        island,
        plan,
        execution,
        frame,
        start,
        stop,
        tuple(exports),
        tuple(origins),
        tuple(boundaries.items()),
        _frame_facts(frame),
        _image_facts(island, tuple(origins)),
        island_input,
        id(island_input) if island_input is not None else None,
    )
    fragments = []
    selected = {issue.node for issue in island.components[0].issues}
    try:
        for index, (value, image) in enumerate(
            (item, image) for item in island.values for image in item.images
        ):
            coordinates = bound.coordinates(
                image if value.additional_images else value.node, "chain_island"
            )
            if _operand_domain(cg, value.node, coordinates, plan):
                return None
            if value.node not in selected:
                expression = _Expression(cg, plan, dict(boundaries))
                for source, coords, name in fragments:
                    if source is not value.node:
                        expression.bind_fragment_image(source, coords, name)
                expression.value(value.node, coordinates)
                if expression.global_accesses:
                    return None
            fragments.append(
                (
                    value.node,
                    coordinates,
                    f"chain_island_value_{index}[chain_island_index]",
                )
            )
    except _UnsupportedChain:
        return None
    return bound
