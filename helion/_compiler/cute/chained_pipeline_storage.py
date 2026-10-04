"""Finalize pipeline storage only after accepted, same-attempt transports.

Capture a revision after final frame/native/TMA rebinding, BEFORE late expression
proofs. Select stage transports once, then pass that exact selection to both this
planner and emission. Pure candidates do not authorize any storage discount.

Endpoint carry views may share the whole frame slab only under the existing CTA
joins: all uploads finish before any preparation, and every role/async operation
finishes before downloads. This includes zero-trip loops. The result does not
authorize an earlier slot release or a new barrier schedule. No arithmetic,
typed layout, frame action, or surviving recurrence lifetime is changed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Literal

import torch

from .chained_loop_tmem_bridges import plan_loop_tmem_bridges
from .chained_loop_tmem_carry import plan_loop_tmem_carry
from .chained_loop_tmem_carry_transport import LoopTmemCarryTransport
from .chained_loop_tmem_slots import plan_tmem_operand_slots
from .chained_loop_tmem_transport import LoopTmemTransport
from .chained_prepared_groups import _valid_frame
from .chained_prepared_groups import prepared_group_candidates
from .chained_prepared_operands import plan_prepared_operands
from .chained_recurrence_workspace import plan_recurrence_workspace
from .chained_workspace import _align
from .chained_workspace import _reuse_requests
from .contraction_region import _domain
from .warp_specialized_plan import VALID_TMEM_COLUMNS
from .warp_specialized_plan import SharedBufferRequest
from .warp_specialized_plan import packed_input_tmem_columns

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from .chained_completed_store import CompletedStorePlan
    from .chained_contraction_groups import ContractionGroup
    from .chained_matmul import ChainedMatmulPlan
    from .chained_output_lease import OutputLeaseSnapshot
    from .chained_preparation_cohorts import PreparationCohorts
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_preparation_storage import BoundPreparationStorage
    from .chained_prepared_groups import PreparedGroupBinding
    from .chained_prepared_operands import PreparedOperand
    from .chained_recurrence_workspace import RecurrenceWorkspace
    from .chained_tmem_accumulator import TmemAccumulatorResidency
    from .chained_tmem_transport import TmemOperandBinding
    from .contraction_region import ContractionCarry


@dataclass(frozen=True)
class StorageRevision:
    """Local compiler-attempt token, not serializable/cacheable proof authority."""

    plan: ChainedMatmulPlan
    pipeline: PreparationPipeline
    shapes: tuple[tuple[Node, tuple[int, ...]], ...]
    facts: tuple[object, ...]
    aliases: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class StageTransports:
    group: ContractionGroup
    tmem_input: TmemOperandBinding | None
    prepared_operand: PreparedOperand | None
    prepared_group: PreparedGroupBinding | None
    tmem_output: LoopTmemTransport | None
    residency: TmemAccumulatorResidency | None
    tmem_accumulator: str | None
    tmem_carry: LoopTmemCarryTransport | None


@dataclass(frozen=True)
class CarryStorageView:
    carry: ContractionCarry
    name: str
    pool: Literal["recurrence", "frames", "endpoints"]
    byte_offset: int
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass(frozen=True)
class PipelineStorage:
    recurrence: RecurrenceWorkspace
    carry_views: tuple[CarryStorageView, ...]
    allocations: tuple[tuple[str, int], ...]
    omitted_results: tuple[Node, ...]
    stages: tuple[StageTransports, ...]
    output_lease: OutputLeaseSnapshot | None = None
    preparation: BoundPreparationStorage | None = None
    completed_store: CompletedStorePlan | None = None

    @property
    def charged_bytes(self) -> int:
        return sum(_align(size) for _, size in self.allocations)


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return (dict, tuple((_freeze(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_freeze(item) for item in value))
    if isinstance(value, slice):
        return (slice, _freeze((value.start, value.stop, value.step)))
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return (type(value), value.node.expr)
    return (type(value), value)


def _facts(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    assert plan.region is not None and plan.loop is not None
    nodes = tuple(
        dict.fromkeys((*plan.region.graph.nodes, *plan.loop.root.graph.nodes))
    )
    return (
        tuple(
            (
                node,
                node.op,
                node.target,
                _freeze(node.args),
                _freeze(node.kwargs),
                (
                    (value.dtype, _domain(value), _freeze(value.stride()))
                    if isinstance(value := node.meta.get("val"), torch.Tensor)
                    else _freeze(value)
                ),
                node.meta.get("lowering"),
            )
            for node in nodes
        ),
    )


def capture_storage_revision(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    shapes: Mapping[Node, tuple[int, ...]],
) -> StorageRevision | None:
    """Record exact semantic facts before late proof, with resolved body shapes.

    The original admitted loop's runtime storage/alias proof remains required.
    This token cannot replace it or turn an early candidate into a late proof.
    """
    if plan.region is None or plan.loop is None:
        return None
    resolved = tuple(
        (node, shapes.get(node))
        for node in plan.region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    )
    if any(
        type(shape) is not tuple
        # Host tensors captured by a zero-trip loop may have empty dimensions.
        # This is a semantic revision, not an allocation request: frame and
        # recurrence planners independently require positive physical storage.
        or any(type(size) is not int or size < 0 for size in shape)
        for _, shape in resolved
    ):
        return None
    return StorageRevision(
        plan,
        pipeline,
        tuple((node, shape) for node, shape in resolved if shape is not None),
        _facts(plan),
        tuple(plan.tensor_aliases.items()),
    )


def select_stage_transports(
    pipeline: PreparationPipeline,
    bridges: tuple[LoopTmemTransport, ...],
    carry: LoopTmemCarryTransport | None = None,
) -> tuple[StageTransports, ...] | None:
    """Mirror current stage priority, using ONLY accepted late-proof objects.

    ``pipeline`` must contain the final domain-filtered singleton/group images.
    Failed domain/packing proofs must be absent, retaining ordinary staging.
    The returned records, not a second independent selection, drive emission.
    """
    if (
        any(not isinstance(item, LoopTmemTransport) for item in bridges)
        or carry is not None
        and not isinstance(carry, LoopTmemCarryTransport)
        or len({item.slot.candidate.source_stage for item in bridges}) != len(bridges)
        or len({item.slot.candidate.destination_group for item in bridges})
        != len(bridges)
        or len({item.stage for item in pipeline.prepared_operands})
        != len(pipeline.prepared_operands)
        or len({item.candidate.group for item in pipeline.prepared_groups})
        != len(pipeline.prepared_groups)
    ):
        return None
    result = []
    for stage in pipeline.recurrence.stages:
        group = stage.group
        first = group.stages[0]
        result.append(
            StageTransports(
                group,
                (None if carry is None else carry.operand(group))
                or next(
                    (
                        item.operand
                        for item in bridges
                        if item.slot.candidate.destination_group == group
                    ),
                    None,
                ),
                next(
                    (
                        item
                        for item in pipeline.prepared_operands
                        if item.stage == first
                    ),
                    None,
                ),
                next(
                    (
                        item
                        for item in pipeline.prepared_groups
                        if item.candidate.group == group
                    ),
                    None,
                ),
                next(
                    (
                        item
                        for item in bridges
                        if item.slot.candidate.source_stage == first
                    ),
                    None,
                ),
                pipeline.residency,
                None if carry is None else carry.accumulator(first),
                carry
                if carry is not None and carry.candidate.final_group == group
                else None,
            )
        )
    if sum(item.tmem_output is not None for item in result) != len(bridges):
        return None
    return tuple(result)


def _accepted_transports(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    stages: tuple[StageTransports, ...],
) -> bool:
    groups = tuple(stage.group for stage in pipeline.recurrence.stages)
    if tuple(item.group for item in stages) != groups:
        return False
    bridges = tuple(item.tmem_output for item in stages if item.tmem_output is not None)
    carries = tuple(item.tmem_carry for item in stages if item.tmem_carry is not None)
    if len(carries) > 1:
        return False
    carry = carries[0] if carries else None
    if select_stage_transports(pipeline, bridges, carry) != stages:
        return False
    candidates = plan_loop_tmem_bridges(plan, groups)
    if candidates is None or any(
        item.slot.candidate not in candidates for item in bridges
    ):
        return False
    slots = plan_tmem_operand_slots(
        tuple(item.slot.candidate for item in bridges),
        max(group.physical[1] for group in groups),
    )
    if slots is None or slots.slots != tuple(item.slot for item in bridges):
        return False
    for item in bridges:
        first = item.slot.candidate.source_stage
        stage = next(entry for entry in stages if first in entry.group.stages)
        if len(stage.group.stages) != 1 or (
            pipeline.residency is not None and first == pipeline.residency.source_stage
        ):
            return False
        dtype = (
            "cutlass.BFloat16"
            if item.slot.candidate.dtype == torch.bfloat16
            else "cutlass.Float16"
        )
        if item.dtype != dtype:
            return False
    if carry is not None:
        expected = plan_loop_tmem_carry(plan, groups, pipeline.residency)
        arena = (slots.required_columns + 31) // 32 * 32
        if expected is None or carry.candidate != expected:
            return False
        snapshot = (arena + expected.arena_columns + 31) // 32 * 32
        end = snapshot + packed_input_tmem_columns(expected.snapshot_shape[1])
        if (
            carry.arena_offset != arena
            or carry.snapshot_offset != snapshot
            or carry.required_columns != end
            or end > VALID_TMEM_COLUMNS[-1]
            or carry.dtype
            != (
                "cutlass.BFloat16"
                if expected.snapshot_dtype == torch.bfloat16
                else "cutlass.Float16"
            )
        ):
            return False
    available = plan_prepared_operands(plan, pipeline.frame, pipeline.recurrence)
    grouped = prepared_group_candidates(plan, pipeline.frame, pipeline.recurrence)
    if (
        available is None
        or grouped is None
        or any(item not in available for item in pipeline.prepared_operands)
    ):
        return False
    for binding in pipeline.prepared_groups:
        candidate = binding.candidate
        if candidate not in grouped or any(
            pipeline.frame.layout.region(member.buffer.name).byte_offset
            != binding.byte_offset + member.byte_offset
            for member in candidate.members
        ):
            return False
    return all(
        not (item.prepared_operand is not None and item.prepared_group is not None)
        for item in stages
    )


def finalize_pipeline_storage(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    stages: tuple[StageTransports, ...],
    *,
    revision: StorageRevision,
    capacity_bytes: int,
    cohorts: PreparationCohorts | None = None,
    output_lease: OutputLeaseSnapshot | None = None,
    preparation: BoundPreparationStorage | None = None,
    completed_store: CompletedStorePlan | None = None,
) -> PipelineStorage | None:
    """Repack actual publications; never discount an unproved transport.

    ``None`` means no finalization was admitted. A caller may retain the old
    path only if its COMPLETE unchanged allocation still fits its own quota.
    No-finalization/default source generation is outside this pure component.
    """
    from .chained_leaf_sets import LeafSetProtocol

    if preparation is not None and not preparation.matches(plan, pipeline):
        return None
    if completed_store is not None and (
        not completed_store.matches()
        or completed_store.plan is not plan
        or completed_store.pipeline is not pipeline
        or completed_store.stages is not stages
        or completed_store.preparation is not preparation
        or output_lease is not None
        or dict(pipeline.recurrence.bindings).get(completed_store.candidate.node)
        != completed_store.c_name
        or pipeline.recurrence.layout.region(completed_store.c_name)
        is not completed_store.request
    ):
        return None
    if pipeline.scan_producer is not None and not pipeline.scan_producer.matches(
        plan, pipeline, dict(revision.shapes)
    ):
        return None
    if output_lease is not None and not output_lease.matches(
        plan, pipeline, stages, revision
    ):
        return None
    if (
        type(capacity_bytes) is not int
        or capacity_bytes < 0
        or plan.region is None
        or plan.loop is None
        or revision.plan is not plan
        or revision.pipeline is not pipeline
        or revision.facts != _facts(plan)
        # _Expression registers additional host-tensor names during late proof.
        # Appending names is not a graph change; rebinding an old name is.
        or any(
            plan.tensor_aliases.get(name) != alias for name, alias in revision.aliases
        )
        or not _valid_frame(pipeline.frame)
        or not pipeline.recurrence.stages
        or plan_recurrence_workspace(
            plan, pipeline.frame.cut, dict(revision.shapes), pipeline.residency
        )
        != pipeline.recurrence
        or not _accepted_transports(plan, pipeline, stages)
    ):
        return None
    if cohorts is None:
        if type(pipeline.slots) is not int or pipeline.slots != 2:
            return None
        slots = pipeline.slots
    else:
        from .chained_preparation_cohorts import plan_preparation_cohorts

        if (
            type(pipeline.slots) is not int
            or pipeline.slots != cohorts.slots
            or pipeline.preparation_threads != cohorts.preparation_threads
            or cohorts
            != plan_preparation_cohorts(
                plan.threads,
                pipeline.recurrence_threads,
                cohorts.count,
                has_tma=bool(pipeline.prepared_leaves),
                named_barrier_base=cohorts.named_barrier_base,
            )
        ):
            return None
        slots = cohorts.slots
    slot_bars = (
        LeafSetProtocol(
            slots, len(pipeline.prepared_leaves), cohorts is not None
        ).barrier_count
        if pipeline.prepared_leaves
        else 2 * slots
    )
    assert plan.loop is not None
    omitted = set()
    resident = []
    a_bytes = b_bytes = 0
    for item in stages:
        m, n, k = item.group.physical
        if item.tmem_input is None:
            a_bytes = max(a_bytes, _align(2 * m * k))
        if item.prepared_operand is None and item.prepared_group is None:
            b_bytes = max(b_bytes, _align(2 * n * k))
        if item.tmem_output is not None or (
            item.residency is not None
            and item.group.stages[0] == item.residency.source_stage
        ):
            omitted.update(plan.dots[index] for index in item.group.stages)
        if item.tmem_carry is not None:
            carry = item.tmem_carry.candidate.carry
            omitted.add(carry.output)
            resident.append(carry)
    original = pipeline.recurrence
    if completed_store is not None:
        if completed_store.candidate.node in omitted:
            return None
        omitted.add(completed_store.candidate.node)
    bindings = dict(original.bindings)
    removed = omitted | {carry.input for carry in resident}
    names = {bindings[node] for node in removed if node in bindings}
    requests = tuple(
        SharedBufferRequest(
            region.name,
            region.byte_size,
            region.alignment,
            region.live_from,
            region.live_until,
        )
        for region in original.layout.regions
        if region.name not in names
    )
    if any(request.alignment != 128 for request in requests):
        return None
    layout = _reuse_requests(requests)
    endpoint_requests = tuple(
        SharedBufferRequest(
            bindings[carry.input],
            original.layout.region(bindings[carry.input]).byte_size,
            128,
        )
        for carry in resident
    )
    endpoints = _reuse_requests(endpoint_requests)
    frame_bytes = slots * (
        pipeline.frame.layout.allocated_bytes
        if preparation is None
        else preparation.stride
    )
    overlay = endpoints.allocated_bytes <= frame_bytes
    views = []
    shapes = dict(revision.shapes)
    for carry in plan.loop.region.carries:
        name = bindings[carry.input]
        pool: Literal["recurrence", "frames", "endpoints"] = "recurrence"
        region = (
            layout.region(name) if carry not in resident else endpoints.region(name)
        )
        if carry in resident:
            pool = "frames" if overlay else "endpoints"
        views.append(
            CarryStorageView(
                carry,
                name,
                pool,
                region.byte_offset,
                torch.float32,
                shapes[carry.input],
            )
        )
    events = {request.live_from for request in requests}
    peak = max(
        (
            sum(
                _align(request.byte_size)
                for request in requests
                if request.live_from <= event < request.live_until
            )
            for event in events
        ),
        default=0,
    )
    recurrence = replace(
        original,
        layout=layout,
        a_bytes=a_bytes,
        b_bytes=b_bytes,
        peak_live_bytes=peak,
        bindings=tuple(
            (node, name) for node, name in original.bindings if node not in removed
        ),
    )
    allocations = tuple(
        (name, size)
        for name, size in (
            ("a", a_bytes),
            ("b", b_bytes),
            ("recurrence", layout.allocated_bytes),
            ("frames", frame_bytes),
            ("endpoints", 0 if overlay else endpoints.allocated_bytes),
            ("stage_barriers", len(plan.dots) * 8),
            ("tmem_address", 4),
            ("slot_barriers", slot_bars * 8),
            (
                "output_snapshot",
                0 if output_lease is None else output_lease.layout.allocated_bytes,
            ),
        )
        if size
    )
    result = PipelineStorage(
        recurrence,
        tuple(views),
        allocations,
        tuple(node for node in plan.dots if node in omitted),
        stages,
        output_lease,
        preparation,
        completed_store,
    )
    if completed_store is not None and not completed_store.matches_storage(result):
        return None
    if result.charged_bytes > capacity_bytes:
        return None
    if preparation is not None and not preparation._finalize(result):
        return None
    return result
