"""A warp prefix retained through one native operand producer.

The original frame is immutable. This proof describes the *effective* order
inside one action span, including every earlier write and prolonged read. A
prefix keeps its original full logical view; only proven residual rows occupy
shared storage. The emitter must additionally prove the exact coordinate reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from .chained_collectives import Collective
from .chained_collectives import classify_collective
from .chained_mma_selection import warp_mma_shape
from .chained_operand_retention import _InvalidRetention
from .chained_operand_retention import _Revision
from .chained_operand_retention import _revision
from .chained_sparse_reduction import plan_sparse_reduction

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_frame import PreparationAction
    from .chained_preparation_frame import PreparationBuffer
    from .chained_preparation_frame import PreparationStage
    from .chained_preparation_pipeline import PreparationPipeline
    from .warp_specialized_plan import SharedBufferRegion


@dataclass(frozen=True)
class ScanProducer:
    revision: _Revision
    first_event: int
    stop_event: int
    scan: Collective
    buffer: PreparationBuffer
    stage: PreparationStage
    shape: tuple[int, int]
    prelude: tuple[PreparationAction, ...]
    deferred: tuple[PreparationAction, ...]
    residual_rows: tuple[int, ...]
    phases: tuple[int, ...]
    regions: tuple[SharedBufferRegion, ...]
    overrides: tuple[tuple[str, int], ...]
    scratch_mode: str
    _context: tuple[object, ...] = field(repr=False)
    _selection: tuple[object, ...] = field(repr=False)

    def _fields(self) -> tuple[object, ...]:
        return (
            self.revision,
            self.first_event,
            self.stop_event,
            self.scan,
            self.buffer,
            self.stage,
            self.shape,
            self.prelude,
            self.deferred,
            self.residual_rows,
            self.phases,
            self.regions,
            self.overrides,
            self.scratch_mode,
            self._context,
        )

    def matches(
        self,
        plan: ChainedMatmulPlan,
        pipeline: PreparationPipeline,
        shapes: Mapping[Node, tuple[int, ...]],
    ) -> bool:
        try:
            return (
                self.revision.plan is plan
                and self.revision.frame is pipeline.frame
                and self.revision == _revision(plan, pipeline.frame, shapes)
                and self._context == _context(pipeline)
                and self._selection == self._fields()
            )
        except _InvalidRetention:
            return False


def _context(pipeline: PreparationPipeline) -> tuple[object, ...]:
    return (
        pipeline.prepared_operands,
        pipeline.prepared_groups,
        pipeline.prepared_leaves,
        pipeline.operand_retention,
        pipeline.cohorts,
        pipeline.slots,
        pipeline.preparation_threads,
    )


def _collides(left: SharedBufferRegion, right: SharedBufferRegion) -> bool:
    return left.overlaps_storage(right) and left.overlaps_lifetime(right)


def _overlay(
    pipeline: PreparationPipeline,
    first: int,
    fill: PreparationAction,
    prelude: tuple[PreparationAction, ...],
    deferred: tuple[PreparationAction, ...],
    scan_name: str,
    rows: tuple[int, ...],
    width: int,
) -> (
    tuple[tuple[int, ...], tuple[SharedBufferRegion, ...], tuple[tuple[str, int], ...]]
    | None
):
    """Conservatively preserve reservations while reordering complete phases.

    Both supported FP32 layouts keep an entire logical row inside the same
    physical row span. Reserving complete rows therefore also covers every XOR
    lane permutation, without inventing a dense rebased prefix view.
    """
    frame = pipeline.frame
    scale = len(frame.actions) + 2
    phases = [action.event * scale for action in frame.actions]
    for index, action in enumerate(prelude):
        phases[action.event] = first * scale + index * 2
    fused = first * scale + len(prelude) * 2
    phases[first] = phases[fill.event] = fused
    for index, action in enumerate(deferred):
        phases[action.event] = fused + 2 + index * 2
    if deferred and phases[deferred[-1].event] + 1 >= phases[fill.event + 1]:
        return None
    writes = {
        name: tuple(action.event for action in frame.actions if name in action.writes)
        for name in (region.name for region in frame.layout.regions)
    }
    reads = {
        name: tuple(action.event for action in frame.actions if name in action.reads)
        for name in writes
    }
    regions = []
    relocated = {
        name
        for action in prelude
        if action.kind == "collective"
        for name in action.writes
    }
    for original in frame.layout.regions:
        own_writes, own_reads = writes[original.name], reads[original.name]
        if len(own_writes) != 1:
            return None
        write = own_writes[0]
        last = max((write, *own_reads)) + 1
        # A lease beginning before the original writer is an authoritative
        # early-publication reservation, not an ordinary action start.
        begin = phases[write]
        if original.live_from < write:
            begin = min(begin, original.live_from * scale)
        end = max(phases[event] + 1 for event in (write, *own_reads))
        if original.live_until > last:
            end = max(end, original.live_until * scale)
        region = replace(original, live_from=begin, live_until=end)
        if original.name == scan_name:
            # The complete scan is never stored. These original full-view rows
            # are the only shared outputs; the fill reads its own registers.
            end = max(phases[action.event] + 1 for action in deferred)
            regions.extend(
                replace(
                    region,
                    name=f"{scan_name}:row:{row}",
                    byte_offset=original.byte_offset + row * width * 4,
                    byte_size=width * 4,
                    live_until=end,
                )
                for row in rows
            )
        elif original.name not in relocated:
            regions.append(region)
    # Original grouped-native members can share a full layout. Retain their
    # COMPLETE union as a reservation, not just individual dense member spans.
    unions = []
    for binding in pipeline.prepared_groups:
        members = {member.buffer.name for member in binding.candidate.members}
        selected = [region for region in regions if region.name in members]
        if len(selected) != len(members):
            return None
        unions.append(
            (
                members,
                replace(
                    selected[0],
                    name=binding.candidate.name,
                    byte_offset=binding.byte_offset,
                    byte_size=binding.candidate.byte_size,
                    live_from=min(
                        *(region.live_from for region in selected),
                        binding.candidate.live_from * scale,
                    ),
                    live_until=max(
                        *(region.live_until for region in selected),
                        binding.candidate.live_until * scale,
                    ),
                ),
            )
        )
    if any(
        _collides(left, right)
        for i, left in enumerate(regions)
        for right in regions[i + 1 :]
    ):
        return None
    if any(
        _collides(union, region)
        for members, union in unions
        for region in regions
        if region.name not in members
    ):
        return None
    overrides = []
    for action in prelude:
        if action.kind != "collective":
            continue
        for name in action.writes:
            original = frame.layout.region(name)
            write = writes[name][0]
            begin = phases[write]
            if original.live_from < write:
                begin = min(begin, original.live_from * scale)
            end = max(phases[event] + 1 for event in (write, *reads[name]))
            if original.live_until > max((write, *reads[name])) + 1:
                end = max(end, original.live_until * scale)
            proposal = replace(original, live_from=begin, live_until=end)
            offsets = (
                original.byte_offset,
                *range(0, frame.layout.allocated_bytes - original.byte_size + 1, 128),
            )
            for offset in dict.fromkeys(offsets):
                candidate = replace(proposal, byte_offset=offset)
                if not any(
                    _collides(candidate, other)
                    for other in (*regions, *(union for _, union in unions))
                ):
                    break
            else:
                return None
            regions.append(candidate)
            if candidate.byte_offset != original.byte_offset:
                overrides.append((name, candidate.byte_offset))
    if any(region.byte_end > frame.layout.allocated_bytes for region in regions):
        return None
    return tuple(phases), tuple(regions), tuple(overrides)


def plan_scan_producer(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    shapes: Mapping[Node, tuple[int, ...]],
    *,
    scratch_mode: str,
) -> ScanProducer | None:
    """Select one rank-two single-warp scan and its complete next operand fill.

    The bounded initial copy geometry is 32 rows, eight values per thread,
    one 128-thread role. Longer/hierarchical scans and partial copy tails keep
    their original path. This planner never substitutes a mathematical formula.
    """
    from .chained_matmul import _ancestors

    frame = pipeline.frame
    threads = (
        pipeline.preparation_threads
        if pipeline.cohorts is None
        else pipeline.cohorts.cohort_threads
    )
    if threads != 128 or scratch_mode not in ("row_major", "xor"):
        return None
    try:
        revision = _revision(plan, frame, shapes)
    except _InvalidRetention:
        return None
    buffers = {buffer.name: buffer for buffer in frame.buffers}
    for first in frame.actions:
        if (
            first.kind != "collective"
            or len(first.nodes) != 1
            or len(first.writes) != 1
        ):
            continue
        scan = classify_collective(first.nodes[0])
        if scan is None or scan.kind != "scan" or scan.axis != 0:
            continue
        shape = shapes.get(scan.node)
        if (
            shape is None
            or len(shape) != 2
            or shape != shapes.get(scan.source)
            or shape[0] != 32
            or shape[1] % 32
        ):
            continue
        width = shape[1]
        if width <= 0 or scratch_mode == "xor" and width & (width - 1):
            continue
        buffer = buffers[first.writes[0]]
        if (
            buffer.node is not scan.node
            or buffer.kind != "collective"
            or buffer.shape != shape
            or buffer.dtype != torch.float32
            or frame.layout.region(buffer.name).byte_size < 4 * shape[0] * shape[1]
        ):
            continue
        following = frame.actions[first.event + 1 :]
        fill = next(
            (
                action
                for action in following
                if action.kind not in ("leaf", "collective")
            ),
            None,
        )
        if fill is None or fill.kind != "fill" or fill.event + 1 >= len(frame.actions):
            continue
        stages = [stage for stage in frame.stages if stage.group.stages == fill.stages]
        if len(stages) != 1:
            continue
        stage = stages[0]
        group = stage.group
        if (
            group.stages[0] not in plan.warp_mma_stages
            or stage.shape != warp_mma_shape(group.geometries[0], group)
            or (stage.shape[0], stage.shape[2]) != shape
            or any(
                (geometry.physical[1], geometry.physical[2]) != shape
                for geometry in group.geometries
            )
            or any(
                plan.operand_dtype(index) not in (torch.bfloat16, torch.float16)
                for index in group.stages
            )
            or any(plan.dots[index].args[2] is not None for index in group.stages)
            or fill.writes != (stage.a.name, stage.b.name)
        ):
            continue
        mma = frame.actions[fill.event + 1]
        if (
            mma.kind != "mma"
            or mma.stages != fill.stages
            or set(mma.reads) != set(fill.writes)
        ):
            continue
        prelude, deferred, rows = [], [], []
        valid = True
        for action in frame.actions[first.event + 1 : fill.event]:
            if action.kind == "collective" and (
                len(action.nodes) != 1
                or len(action.writes) != 1
                or buffers[action.writes[0]].node is not action.nodes[0]
                or buffers[action.writes[0]].dtype != torch.float32
                or buffers[action.writes[0]].shape != shapes.get(action.nodes[0])
            ):
                valid = False
                break
            if buffer.name in action.reads:
                sparse = (
                    plan_sparse_reduction(plan, action.nodes[0])
                    if action.kind == "collective" and len(action.nodes) == 1
                    else None
                )
                if (
                    sparse is None
                    or sparse.axis != 0
                    or sparse.extent != 32
                    or shapes.get(sparse.source) != shape
                ):
                    valid = False
                    break
                deferred.append(action)
                rows.append(sparse.position)
            else:
                # No source of an early action may transitively depend on the
                # scan or a delayed result, even through a nonmaterialized view.
                if any(scan.node in _ancestors(node) for node in action.nodes):
                    valid = False
                    break
                if action.kind == "collective" and any(
                    (operation := classify_collective(node)) is None
                    or operation.kind != "sum"
                    for node in action.nodes
                ):
                    valid = False
                    break
                prelude.append(action)
        if not valid or not deferred or not prelude:
            continue
        consumers = {
            action.event for action in frame.actions if buffer.name in action.reads
        }
        if consumers != {fill.event, *(action.event for action in deferred)}:
            continue
        delayed_names = {name for action in deferred for name in action.writes}
        if any(delayed_names.intersection(action.reads) for action in (*prelude, fill)):
            continue
        # Check read readiness in the actual reordered schedule, not just the
        # original action sequence. The original scan source is read at fill.
        available = {
            name for action in frame.actions[: first.event] for name in action.writes
        }
        for action in prelude:
            if not set(action.reads) <= available:
                valid = False
                break
            available.update(action.writes)
        if (
            not valid
            or not set(first.reads) <= available
            or not set(fill.reads) - {buffer.name} <= available
        ):
            continue
        available.update((buffer.name, *fill.writes))
        for action in deferred:
            if not set(action.reads) <= available:
                valid = False
                break
            available.update(action.writes)
        if not valid:
            continue
        overlay = _overlay(
            pipeline,
            first.event,
            fill,
            tuple(prelude),
            tuple(deferred),
            buffer.name,
            tuple(sorted(set(rows))),
            width,
        )
        if overlay is None:
            continue
        phases, regions, overrides = overlay
        candidate = ScanProducer(
            revision,
            first.event,
            fill.event + 1,
            scan,
            buffer,
            stage,
            (shape[0], shape[1]),
            tuple(prelude),
            tuple(deferred),
            tuple(sorted(set(rows))),
            phases,
            regions,
            overrides,
            scratch_mode,
            _context(pipeline),
            (),
        )
        return replace(candidate, _selection=candidate._fields())
    return None
