"""Published, unchanged native images available to vector expressions.

Retained operands require a late-admitted retention plan; raw preparation leaves
require their own completed publication and effective scan-phase witness. These
are distinct authorities, not interchangeable candidate types. ``published`` is
the caller's receipt from actual completed actions, not a proposed alias map.
This component adds no allocation, movement or copy. The vector reader still
proves exact coordinate/payload mapping and preserves original bounds and zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from .chained_frontier_groups import FrontierGroup
    from .chained_matmul import ChainedMatmulPlan
    from .chained_operand_retention import OperandRetentionPlan
    from .chained_preparation_frame import PreparationFrame
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_scan_producer import ScanProducer


def _matches_retention(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    retention: OperandRetentionPlan,
) -> bool:
    original = retention.revision.plan
    # The pipeline replaces only recurrence storage after accepting retention.
    # All original graph, schedule, shape and selected-producer facts still apply.
    return (
        retention.frame is frame
        and replace(plan, loop_workspace=original.loop_workspace) == original
        and retention.matches(
            original,
            retention.revision.frame,
            dict(retention.revision.shapes),
            prepared_groups=retention.original_groups,
            reservations=retention.reservations,
        )
    )


@dataclass(frozen=True)
class _Selection:
    plan: ChainedMatmulPlan
    retention: OperandRetentionPlan
    values: tuple[object, ...]

    def matches(self, plan: ChainedMatmulPlan) -> bool:
        return _matches_retention(plan, self.retention.frame, self.retention)


@dataclass(frozen=True)
class _LeafSelection:
    plan: ChainedMatmulPlan
    pipeline: PreparationPipeline
    scan: ScanProducer
    shapes: tuple[tuple[Node, tuple[int, ...]], ...]
    values: tuple[object, ...]
    facts: tuple[object, ...]

    def matches(self, plan: ChainedMatmulPlan) -> bool:
        return self.facts == _leaf_facts(
            plan, self.pipeline, self.scan, dict(self.shapes)
        )


def _leaf_facts(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    scan: ScanProducer,
    shapes: Mapping[Node, tuple[int, ...]],
) -> tuple[object, ...] | None:
    """Snapshot actual frame/proof values, including mutable leaf wrappers."""
    from .chained_pipeline_storage import _freeze

    original = scan.revision.plan
    if (
        pipeline.scan_producer is not scan
        or replace(plan, loop_workspace=original.loop_workspace) != original
        or not scan.matches(original, pipeline, shapes)
    ):
        return None
    frame = pipeline.frame
    return (
        frame.layout.allocated_bytes,
        tuple(_freeze(vars(region)) for region in frame.layout.regions),
        tuple(_freeze(vars(region)) for region in scan.regions),
        tuple(_freeze(vars(buffer)) for buffer in frame.buffers),
        tuple(_freeze(vars(action)) for action in frame.actions),
        tuple(
            (
                stage.shape,
                _freeze(vars(stage.a)),
                _freeze(vars(stage.b)),
                stage.group.stages,
                stage.group.offsets,
                stage.group.physical,
                tuple(_freeze(vars(geometry)) for geometry in stage.group.geometries),
            )
            for stage in frame.stages
        ),
        scan.first_event,
        scan.stop_event,
        scan.phases,
        scan.overrides,
        tuple(
            (
                leaf.node,
                leaf.name,
                _freeze(vars(leaf.proof)),
                _freeze(leaf.wrapper),
                leaf.first_event,
                leaf.last_event,
                leaf.read_events,
            )
            for leaf in pipeline.prepared_leaves
        ),
    )


@dataclass(frozen=True)
class NativeReadInput:
    """An already-offset alias of the full native owner, never a dense subview.

    ``index`` is the ordinal in the original selected retention or leaf tuple,
    including candidates which cannot use native vector reads. Only a factory
    installs the private witness; manually constructed records have no proof.
    Source and destination layout/copy selection remains the reader's job.
    """

    node: Node
    tensor: str
    full_shape: tuple[int, int]
    shape: tuple[int, int]
    row_offset: int
    dtype: torch.dtype
    index: int
    _selection: _Selection | _LeafSelection | None = field(
        default=None, repr=False, compare=False
    )

    def matches(self, plan: ChainedMatmulPlan, boundaries: Mapping[Node, str]) -> bool:
        selection = self._selection
        return (
            selection is not None
            and plan is selection.plan
            and type(self.tensor) is str
            and type(self.index) is int
            and type(self.row_offset) is int
            and type(self.full_shape) is tuple
            and type(self.shape) is tuple
            and all(type(size) is int for size in (*self.full_shape, *self.shape))
            and selection.values
            == (
                self.node,
                self.tensor,
                self.full_shape,
                self.shape,
                self.row_offset,
                self.dtype,
                self.index,
            )
            and boundaries.get(self.node) == self.tensor
            and selection.matches(plan)
        )


def bind_preparation_leaf_native_inputs(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    scan: ScanProducer,
    published: Mapping[Node, str],
    shapes: Mapping[Node, tuple[int, ...]],
) -> tuple[NativeReadInput, ...]:
    """Bind completed raw leaves at an already-admitted scan/fill phase.

    ``published`` is the emitter's completed-leaf boundary map, after the
    original prelude leaf waits and role joins. Only each leaf's exact original
    native name is accepted, never a caller-invented alias. The current scan
    proof supplies the effective phase order and full-owner leases; original
    action positions alone cannot authorize a reordered read. The caller still
    emits every original mask/cast and activates only after successful emission.
    """
    facts = _leaf_facts(plan, pipeline, scan, shapes)
    if facts is None:
        return ()
    frame = pipeline.frame
    fills = tuple(
        action
        for action in frame.actions
        if action.kind == "fill" and action.stages == scan.stage.group.stages
    )
    if len(fills) != 1:
        return ()
    fill = fills[0]
    phase = scan.phases[scan.first_event]
    if scan.phases[fill.event] != phase:
        return ()
    consumers = (frame.actions[scan.first_event], fill)
    reads = {name for action in consumers for name in action.reads}
    regions = {region.name: region for region in scan.regions}
    buffers = {buffer.name: buffer for buffer in frame.buffers}
    # These are the complete physical owners, including every grouped-B member.
    destinations = (scan.stage.a, scan.stage.b)
    result = []
    for index, leaf in enumerate(pipeline.prepared_leaves):
        if published.get(leaf.node) != leaf.name or leaf.name not in reads:
            continue
        actions = tuple(
            action for action in frame.actions if action.writes == (leaf.name,)
        )
        if len(actions) != 1:
            continue
        publication = actions[0]
        if (
            publication.kind != "leaf"
            or publication.nodes != (leaf.node,)
            or scan.phases[publication.event] >= phase
            or publication.event >= scan.first_event
            and publication not in scan.prelude
            or not any(action.event in leaf.read_events for action in consumers)
        ):
            continue
        region = regions.get(leaf.name)
        buffer = buffers.get(leaf.name)
        metadata = leaf.node.meta.get("val")
        if not isinstance(metadata, torch.Tensor):
            continue
        dtype = metadata.dtype
        shape = shapes.get(leaf.node)
        if (
            type(shape) is not tuple
            or len(shape) != 2
            or any(type(size) is not int or size <= 0 for size in shape)
            or dtype not in (torch.bfloat16, torch.float16)
            or shape != leaf.proof.tile_shape
            or shape != scan.shape
            or shape[0] % 8
            or shape[1] % 64
            or leaf.proof.element_bytes != dtype.itemsize
            or leaf.proof.dtype
            != ("cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16")
            or leaf.wrapper.get("tile") != shape
            or leaf.wrapper.get("dtype") != str(dtype).removeprefix("torch.")
            or region is None
            or buffer is None
            or buffer.kind != "leaf"
            or buffer.node is not leaf.node
            or buffer.dtype != dtype
            or buffer.shape != shape
            or region.alignment != 128
            or region.byte_offset % 128
            or region.byte_size < math.prod(shape) * dtype.itemsize
            or not region.live_from <= phase < region.live_until
            or any(region.overlaps_storage(destination) for destination in destinations)
        ):
            continue
        values = (leaf.node, leaf.name, shape, shape, 0, dtype, index)
        result.append(
            NativeReadInput(
                leaf.node,
                leaf.name,
                shape,
                shape,
                0,
                dtype,
                index,
                _LeafSelection(
                    plan, pipeline, scan, tuple(shapes.items()), values, facts
                ),
            )
        )
    return tuple(result)


def bind_retained_native_inputs(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    retention: OperandRetentionPlan,
    published: Mapping[Node, str],
    first_event: int,
    stop_event: int,
    writes: tuple[str, ...],
    *,
    frontier_group: FrontierGroup | None = None,
) -> tuple[NativeReadInput, ...]:
    """Restrict late-admitted, actually published images to original consumers.

    Intervals are half open. Every supplied destination must exactly correspond
    to an original action write, including destinations of an early grouped
    frontier publisher. A prepared-group member charges its entire native owner,
    not just the logical member bytes. Source storage must remain live and
    disjoint throughout the whole interval, even if its own consumer is last.
    An exact, freshly rechecked FrontierGroup instead executes all original
    reads/publications during its first action. Only that existing schedule
    permits a source lease ending before the group's original stop event.

    Only identity logical axes on the existing K-major SW128 half layout are
    supported. A row-offset alias keeps its complete owner's layout and offset;
    this function never invents a member layout or changes a source cast.
    """
    if (
        type(first_event) is not int
        or type(stop_event) is not int
        or not 0 <= first_event < stop_event <= len(frame.actions)
        or type(writes) is not tuple
        or any(type(name) is not str for name in writes)
        or len(set(writes)) != len(writes)
        or not _matches_retention(plan, frame, retention)
    ):
        return ()
    actions = frame.actions[first_event:stop_event]
    if any(action.kind not in ("frontier", "fill") for action in actions) or set(
        writes
    ) != {name for action in actions for name in action.writes}:
        return ()
    read_stop = stop_event
    if frontier_group is not None:
        # Frontier emitters consume this module; avoid an import cycle.
        from .chained_frontier_groups import plan_frontier_group

        if (
            frontier_group.first_event != first_event
            or frontier_group.stop_event != stop_event
            or tuple(buffer.name for buffer in frontier_group.buffers) != writes
            or frontier_group != plan_frontier_group(frame, first_event)
        ):
            return ()
        read_stop = first_event + 1

    regions = {region.name: region for region in frame.layout.regions}
    buffers = {buffer.name: buffer for buffer in frame.buffers}
    # A target view may be one member of a native grouped-B allocation. Copy
    # partitions and swizzled indexing use that whole owner, not a dense member.
    destinations = []
    for name in writes:
        region = regions.get(name)
        if region is None:
            return ()
        span = (region.byte_offset, region.byte_end)
        for binding in retention.prepared_groups:
            if any(member.buffer.name == name for member in binding.candidate.members):
                span = (
                    binding.byte_offset,
                    binding.byte_offset + binding.candidate.byte_size,
                )
                break
        destinations.append(span)

    result = []
    for index, candidate in enumerate(retention.candidates):
        tensor = published.get(candidate.node)
        if (
            type(tensor) is not str
            or not tensor.isidentifier()
            or candidate.logical_modes != (0, 1)
            or candidate.dtype not in (torch.bfloat16, torch.float16)
            or candidate.major_mode != "k"
            or candidate.full_shape[0] % 8
            or candidate.full_shape[1] % 64
            or candidate.logical_shape[1] != candidate.full_shape[1]
            or candidate.row_offset < 0
            or candidate.row_offset + candidate.logical_shape[0]
            > candidate.full_shape[0]
            or candidate.publication_event > first_event
            or not any(
                action.event in candidate.consumer_events
                and candidate.owner in action.reads
                for action in actions
            )
        ):
            continue
        owner = regions.get(candidate.owner)
        buffer = buffers.get(candidate.owner)
        if (
            owner is None
            or buffer is None
            or buffer.kind != candidate.role
            or buffer.dtype != candidate.dtype
            or buffer.shape != candidate.full_shape
            or owner.byte_size
            < math.prod(candidate.full_shape) * candidate.dtype.itemsize
            or owner.alignment != 128
            or owner.byte_offset % 128
            or not owner.live_from <= first_event < read_stop <= owner.live_until
            or any(
                owner.byte_offset < end and start < owner.byte_end
                for start, end in destinations
            )
        ):
            continue
        # The immutable retained frame includes the exact physical stage owner.
        # Do not accept a similarly named buffer or a different grouped layout.
        if not any(
            stage.group == candidate.group
            and (stage.a if candidate.role == "a" else stage.b) == owner
            and (stage.shape[0 if candidate.role == "a" else 1], stage.shape[2])
            == candidate.full_shape
            for stage in frame.stages
        ):
            continue
        values = (
            candidate.node,
            tensor,
            candidate.full_shape,
            candidate.logical_shape,
            candidate.row_offset,
            candidate.dtype,
            index,
        )
        result.append(
            NativeReadInput(
                candidate.node,
                tensor,
                candidate.full_shape,
                candidate.logical_shape,
                candidate.row_offset,
                candidate.dtype,
                index,
                _Selection(plan, retention, values),
            )
        )
    return tuple(result)
