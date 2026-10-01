"""Two-role preparation ahead of an ordered, graph-owned contraction loop.

Slots contain exact typed images, not a model-specific algorithm. One role
executes the independent DAG; the other owns every recurrent update and store.
A slot is reusable only after all of its consumer reads have completed. Every
inner synchronization is role-local; allocation and the final join are CTA-wide.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from .chained_execution import ChainedExecution
from .chained_leaf_sets import LeafSetProtocol
from .chained_pointwise_residency import _finite_consumers
from .chained_preparation_frame import plan_preparation_frame
from .chained_recurrence_workspace import plan_recurrence_workspace

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import TypedDict

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_broadcast_retention import BroadcastRetentionAttempt
    from .chained_cache_layout import PointwiseCacheLayouts
    from .chained_leaf_schedule_emission import LeafEmissionCapture
    from .chained_matmul import ChainedMatmulPlan
    from .chained_operand_retention import OperandRetentionPlan
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_preparation_actions import _PreparationRecorder
    from .chained_preparation_cohorts import PreparationCohorts
    from .chained_preparation_cut import PreparationCut
    from .chained_preparation_frame import PreparationFrame
    from .chained_preparation_leaves import PreparationLeaf
    from .chained_prepared_groups import PreparedGroupBinding
    from .chained_prepared_operands import PreparedOperand
    from .chained_recurrence_workspace import RecurrenceWorkspace
    from .chained_scan_producer import ScanProducer
    from .chained_scratch_layout import ScratchLayouts
    from .chained_seed_tiles import SeedTiling
    from .chained_tmem_accumulator import TmemAccumulatorResidency
    from .chained_vector_native import NativeReadInputs
    from .chained_vector_stage import VectorStageOperand
    from .chained_vector_stage import VectorStaging

    class _ScanExpressionOptions(TypedDict, total=False):
        native_inputs: NativeReadInputs
        broadcast: BroadcastRetentionAttempt

    class _AcceptedPreparationOptions(TypedDict, total=False):
        leaf_capture: LeafEmissionCapture
        broadcast_retention: bool
        island_consumers: bool
        fragment_epilogues: bool


@dataclass(frozen=True)
class PreparationPipeline:
    frame: PreparationFrame
    recurrence: RecurrenceWorkspace
    residency: TmemAccumulatorResidency | None
    slots: int
    preparation_threads: int
    recurrence_threads: int
    protocol_bytes: int
    prepared_operands: tuple[PreparedOperand, ...] = ()
    prepared_groups: tuple[PreparedGroupBinding, ...] = ()
    prepared_leaves: tuple[PreparationLeaf, ...] = ()
    cohorts: PreparationCohorts | None = None
    operand_retention: OperandRetentionPlan | None = None
    scan_producer: ScanProducer | None = None

    @property
    def slot_barrier_count(self) -> int:
        if not self.prepared_leaves:
            return 2 * self.slots
        return LeafSetProtocol(
            self.slots, len(self.prepared_leaves), self.cohorts is not None
        ).barrier_count

    @property
    def shared_bytes(self) -> int:
        return (
            self.slots * self.frame.layout.allocated_bytes
            + self.recurrence.shared_bytes
            + self.protocol_bytes
        )


def validate_preparation_pipeline(
    plan: ChainedMatmulPlan | None, requested: object
) -> None:
    if requested is False:
        return
    if requested is not True or plan is None or plan.preparation_pipeline is None:
        raise exc.BackendUnsupported(
            "cute",
            "preparation pipeline requires a supported graph cut, role schedule, and shared-memory plan",
        )


def plan_preparation_pipeline(
    plan: ChainedMatmulPlan,
    cut: PreparationCut,
    shapes: Mapping[Node, tuple[int, ...]],
    capacity: int,
    slots: int = 2,
    *,
    consumer_warps: int = 4,
    cohort_count: int = 1,
    has_tma: bool = False,
) -> PreparationPipeline | None:
    """Choose one bounded schedule after proving dependencies and all storage."""
    from .chained_preparation_cohorts import plan_preparation_cohorts
    from .chained_prepared_operands import plan_prepared_operands
    from .chained_tmem_accumulator import plan_tmem_accumulator_residency

    if (
        type(slots) is not int
        or slots != 2
        or type(consumer_warps) is not int
        or consumer_warps not in (4, 8, 16)
        or plan.threads < 32 * consumer_warps + 128
        or plan.threads % 128
        or type(cohort_count) is not int
        or cohort_count < 1
        or type(has_tma) is not bool
    ):
        return None
    cohorts = plan_preparation_cohorts(
        plan.threads, 32 * consumer_warps, cohort_count, has_tma=has_tma
    )
    if cohort_count != 1 and cohorts is None:
        return None
    if cohorts is not None:
        slots = cohorts.slots
    # The ordered role owns TMEM; only the preparation role uses warp MMA.
    # Small uninitialized recurrent dots still belong to the ordered TCgen
    # sequence, even when they also meet the preparation row threshold.
    preparation = set(cut.preparation)
    plan = replace(
        plan,
        warp_mma_stages=frozenset(
            stage for stage in plan.warp_mma_stages if plan.dots[stage] in preparation
        ),
    )
    # Materialized pointwise values have finite logical bounds. Arbitrary
    # gathers can otherwise turn inline exp(0) outside the domain into zero.
    boundaries = {*plan.dots, *cut.region.scans, *cut.region.reductions}
    if not all(
        _finite_consumers(image.node, boundaries, within=frozenset(cut.recurrence))
        for image in cut.images
    ):
        return None
    frame = plan_preparation_frame(plan, cut, shapes)
    recurrence = plan_recurrence_workspace(plan, cut, shapes)
    if frame is None or recurrence is None:
        return None
    residency = plan_tmem_accumulator_residency(
        plan, tuple(stage.group for stage in recurrence.stages)
    )
    if residency is not None:
        recurrence = plan_recurrence_workspace(plan, cut, shapes, residency)
        if recurrence is None:
            return None
    prepared_operands = plan_prepared_operands(plan, frame, recurrence)
    if prepared_operands is None:
        return None
    # Individual allocations align independently: stage completion barriers,
    # TMEM address holding, and the slot ready/empty barriers respectively.
    slot_bytes = 128 if cohorts is None else cohorts.slot_mbarrier_allocated_bytes
    protocol_bytes = ((len(plan.dots) * 8 + 127) // 128 + 1) * 128 + slot_bytes
    pipeline = PreparationPipeline(
        frame,
        recurrence,
        residency,
        slots,
        plan.threads - 32 * consumer_warps,
        32 * consumer_warps,
        protocol_bytes,
        prepared_operands,
        cohorts=cohorts,
    )
    # A cohort request is provisional until SAME-ATTEMPT late transports are
    # accepted and the full allocation is finalized. Never use this upper
    # bound as a discounted allocation or launch-capacity proof.
    if cohorts is not None:
        return pipeline
    return pipeline if pipeline.shared_bytes <= capacity else None


def _frame_planning_capacity(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    *,
    compact_preparation: bool = False,
) -> int:
    """Bound semantic construction, not the eventual physical allocation.

    Compact cohorts first retain original logical images, then bind accepted
    actions into a separate physical table. Dividing this provisional bound by
    slots would reject those images before their actual storage is known. The
    finalizer still charges every physical slot and all other CTA allocations.
    Ordinary schedules and the original single-team path keep their old bound.
    """
    from .tcgen05_config import CuteTcgen05Config

    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(
        plan.dots[0].meta["val"].device
    )
    divisor = (
        1 if compact_preparation and pipeline.cohorts is not None else pipeline.slots
    )
    return capacity // divisor // 128 * 128


def _retain_operands(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    *,
    compact_preparation: bool = False,
) -> PreparationPipeline:
    """Admit exact native images and atomically replace every frame binding."""
    from ..compile_environment import CompileEnvironment
    from .chained_matmul import _shape
    from .chained_operand_retention import discover_operand_retention
    from .chained_operand_retention import plan_operand_retention_frame
    from .chained_operand_retention_emission import admit_operand_retention
    from .chained_operand_retention_emission import operand_retention_reservations
    from .chained_prepared_groups import prepared_group_candidates
    from .chained_prepared_operands import plan_prepared_operands

    frame = pipeline.frame
    shapes = {
        node: _shape(node)
        for node in frame.cut.region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    candidates = discover_operand_retention(plan, frame, shapes)
    register_islands = (
        cg.device_function.config.config.get("cute_chained_register_islands") is True
        and CompileEnvironment.current().settings.fast_math is True
    )
    candidates = (
        admit_operand_retention(
            cg, plan, frame, candidates, register_islands=register_islands
        )
        if candidates
        else ()
    )
    reservations = operand_retention_reservations(
        plan,
        frame,
        register_islands=register_islands,
        frontier_groups=(
            cg.device_function.config.config.get("cute_chained_vector_group") is True
            and cg.device_function.config.config.get("cute_chained_pointwise_vectorize")
            is True
        ),
    )
    retained = plan_operand_retention_frame(
        plan,
        frame,
        candidates,
        shapes,
        prepared_groups=pipeline.prepared_groups,
        reservations=reservations,
        # Retaining an operand can grow a small frame even while eliminating
        # recomputation. This is only a placement ceiling; the mandatory
        # same-attempt finalizer still charges recurrence and protocol storage.
        capacity_bytes=_frame_planning_capacity(
            plan, pipeline, compact_preparation=compact_preparation
        ),
    )
    if retained is None or not retained.matches(
        plan,
        frame,
        shapes,
        prepared_groups=pipeline.prepared_groups,
        reservations=reservations,
    ):
        raise exc.BackendUnsupported(
            "cute",
            "operand retention requires exact typed values and a complete frame proof",
        )
    operands = plan_prepared_operands(plan, retained.frame, pipeline.recurrence)
    groups = prepared_group_candidates(plan, retained.frame, pipeline.recurrence)
    if (
        operands is None
        or groups is None
        or any(binding.candidate not in groups for binding in retained.prepared_groups)
    ):
        raise exc.BackendUnsupported(
            "cute", "retained operand views could not be rebound"
        )
    # Keep the original late domain selection; rebinding a region does not
    # promote a previously rejected native transport.
    accepted = {operand.buffer.name for operand in pipeline.prepared_operands}
    reads = dict(retained.leaf_reads)
    if any(not reads.get(leaf.node) for leaf in pipeline.prepared_leaves):
        raise exc.BackendUnsupported(
            "cute", "operand retention cannot erase a selected TMA leaf's read schedule"
        )
    return replace(
        pipeline,
        frame=retained.frame,
        prepared_operands=tuple(
            item for item in operands if item.buffer.name in accepted
        ),
        prepared_groups=retained.prepared_groups,
        prepared_leaves=tuple(
            replace(
                leaf,
                first_event=reads[leaf.node][0],
                last_event=reads[leaf.node][-1],
                read_events=reads[leaf.node],
            )
            for leaf in pipeline.prepared_leaves
        ),
        operand_retention=retained,
    )


def _emit_scan_producer_stage(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    boundaries: dict[Node, str],
    execution: ChainedExecution,
    unroll: BoundedProducerUnroll,
    vector: VectorStaging,
    *,
    recorder: _PreparationRecorder | None = None,
    native_inputs: NativeReadInputs | None = None,
) -> list[str]:
    """Consume a checked macro span without exposing its partial prefix."""
    from .chained_collectives import collective_bindings
    from .chained_collectives import emit_collectives_before
    from .chained_matmul import _shape
    from .chained_matmul import _UnsupportedChain
    from .chained_native_read_inputs import bind_preparation_leaf_native_inputs
    from .chained_scan_producer_emission import emit_scan_producer
    from .chained_tcgen_stage import emit_stage

    candidate = pipeline.scan_producer
    assert candidate is not None
    names = collective_bindings(plan)
    if names.get(candidate.scan.node) != candidate.buffer.name or any(
        tuple(names.get(node) for node in action.nodes) != action.writes
        for action in (*candidate.prelude, *candidate.deferred)
        if action.kind == "collective"
    ):
        raise exc.BackendUnsupported(
            "cute", "scan producer publication binding changed"
        )
    original = candidate.revision.plan
    if replace(
        plan, loop_workspace=original.loop_workspace
    ) != original or not candidate.matches(
        original,
        pipeline,
        {
            node: _shape(node)
            for node in pipeline.frame.cut.region.nodes
            if isinstance(node.meta.get("val"), torch.Tensor)
        },
    ):
        raise exc.BackendUnsupported("cute", "scan producer revision changed")
    trial = dict(boundaries)
    lines = []
    capture = None if recorder is None else recorder.leaf_capture
    for action in candidate.prelude:
        segment_start = len(lines)
        segment_before = dict(trial) if capture is not None else None
        if action.kind == "leaf":
            ordinal, leaf = next(
                (index, leaf)
                for index, leaf in enumerate(pipeline.prepared_leaves)
                if action.nodes == (leaf.node,)
            )
            protocol = LeafSetProtocol(
                pipeline.slots,
                len(pipeline.prepared_leaves),
                pipeline.cohorts is not None,
            )
            lines.extend(
                leaf.emit(
                    cg,
                    plan,
                    trial,
                    execution,
                    protocol.leaf_pointer(ordinal),
                    protocol.phase,
                )
                if recorder is None
                else recorder.emit_leaf(
                    leaf,
                    action,
                    trial,
                    execution,
                    protocol.leaf_pointer(ordinal),
                    protocol.phase,
                )
            )
            trial[leaf.node] = leaf.name
        else:
            assert action.kind == "collective" and action.source_stage is not None
            lines.extend(
                emit_collectives_before(
                    cg,
                    plan,
                    trial,
                    action.source_stage,
                    execution=execution,
                    selected=frozenset(action.nodes),
                )
            )
            if capture is not None:
                assert recorder is not None and segment_before is not None
                capture.prelude(
                    len(recorder.actions),
                    action,
                    segment_before,
                    trial,
                    lines[segment_start:],
                )
    published = dict(trial) if recorder is not None else None
    core_start = len(lines)
    broadcast_first = 0 if vector.broadcast is None else vector.broadcast.checkpoint()
    if native_inputs is not None:
        # Each prelude leaf has completed its original wait and whole-role
        # join. The compact recorder accepts only relocatable emission here;
        # its separate physical transfer binder still gates body installation.
        native_inputs.inputs = bind_preparation_leaf_native_inputs(
            plan,
            pipeline,
            candidate,
            trial,
            {
                node: _shape(node)
                for node in pipeline.frame.cut.region.nodes
                if isinstance(node.meta.get("val"), torch.Tensor)
            },
        )
    stage = candidate.stage
    local = replace(
        execution,
        a_workspace=(
            recorder.workspace(stage, "a")
            if recorder is not None
            else f"cute.recast_ptr(chain_frame + {stage.a.byte_offset}, dtype=cutlass.BFloat16)"
        ),
        b_workspace=(
            recorder.workspace(stage, "b")
            if recorder is not None
            else f"cute.recast_ptr(chain_frame + {stage.b.byte_offset}, dtype=cutlass.BFloat16)"
        ),
    )

    emissions = []

    def producer(operands: tuple[VectorStageOperand, ...]) -> list[str] | None:
        options: _ScanExpressionOptions = {}
        if native_inputs is not None:
            options["native_inputs"] = native_inputs
        if vector.broadcast is not None:
            options["broadcast"] = vector.broadcast
        emission = emit_scan_producer(
            cg,
            plan,
            candidate,
            trial,
            operands,
            execution=local,
            producer_unroll=unroll,
            **options,
        )
        if emission is None:
            return None
        emissions.append(emission)
        return list(emission.lines)

    try:
        stage_lines = emit_stage(
            cg,
            plan,
            trial,
            stage.group.stages[0],
            stage.group.geometries[0],
            "0",
            stage.group,
            producer_unroll=unroll,
            execution=local,
            prepared_shape=stage.shape,
            operand_producer=producer,
            broadcast_capture=vector.broadcast,
            member_fragment=None if recorder is None else recorder.fragment,
        )
        if recorder is not None and recorder.fragment is not None:
            assert recorder.scan_attempt is not None
            recorder.fragment.accept_stage(
                stage_lines, recorder.scan_attempt.body_first + len(lines)
            )
        broadcast_placement = (
            None
            if vector.broadcast is None
            else vector.broadcast.place(broadcast_first, stage_lines, core_start)
        )
        lines.extend(stage_lines)
    except _UnsupportedChain as error:
        raise exc.BackendUnsupported(
            "cute", "scan producer coordinate/native proof failed"
        ) from error
    trial.update(
        (node, name)
        for action in candidate.deferred
        for node, name in zip(action.nodes, action.writes, strict=True)
    )
    # Do not publish a full logical prefix backed by only residual rows.
    trial.pop(candidate.scan.node, None)
    assert len(emissions) == 1
    if recorder is not None:
        assert published is not None
        recorder.complete_scan(published, trial, local, emissions[0], lines)
        if capture is not None:
            capture.scan_core(
                len(recorder.actions),
                candidate.first_event,
                published,
                trial,
                lines[core_start:],
            )
    boundaries.update(trial)
    if vector.group_enabled and emissions[0].shared_outputs:
        vector.group_activated = True
    if native_inputs is not None and emissions[0].native_reads:
        native_inputs.activated = True
    if vector.broadcast is not None:
        vector.broadcast.enclose(broadcast_first, lines, (broadcast_placement,))
    return lines


def _emit_row_collective_stage(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    first_event: int,
    boundaries: dict[Node, str],
    execution: ChainedExecution,
    unroll: BoundedProducerUnroll,
) -> tuple[list[str], int] | None:
    """Try a complete row-local producer without publishing a partial attempt."""
    from .chained_matmul import _shape
    from .chained_matmul import _UnsupportedChain
    from .chained_row_collective_emission import emit_row_collective_group
    from .chained_row_collectives import plan_row_collective_group
    from .chained_tcgen_stage import emit_stage

    assert plan.region is not None
    candidate = plan_row_collective_group(
        plan,
        frame,
        first_event,
        {
            node: _shape(node)
            for node in plan.region.nodes
            if isinstance(node.meta.get("val"), torch.Tensor)
        },
    )
    if candidate is None:
        return None
    stage = candidate.stage
    local = replace(
        execution,
        a_workspace=f"cute.recast_ptr(chain_frame + {stage.a.byte_offset}, dtype=cutlass.BFloat16)",
        b_workspace=f"cute.recast_ptr(chain_frame + {stage.b.byte_offset}, dtype=cutlass.BFloat16)",
    )
    trial_boundaries = dict(boundaries)

    def producer(operands: tuple[VectorStageOperand, ...]) -> list[str] | None:
        return emit_row_collective_group(
            cg,
            plan,
            frame,
            candidate,
            trial_boundaries,
            operands,
            execution=local,
            producer_unroll=unroll,
        )

    try:
        lines = emit_stage(
            cg,
            plan,
            trial_boundaries,
            stage.group.stages[0],
            stage.group.geometries[0],
            "0",
            stage.group,
            producer_unroll=unroll,
            execution=local,
            prepared_shape=stage.shape,
            operand_producer=producer,
        )
    except _UnsupportedChain:
        return None
    trial_boundaries.update(
        (operation.node, buffer.name)
        for operation, buffer in zip(
            candidate.collectives, candidate.buffers, strict=True
        )
    )
    boundaries.update(trial_boundaries)
    return lines, candidate.stop_event


def _prepare(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    execution: ChainedExecution,
    vector: VectorStaging,
    unroll: BoundedProducerUnroll,
    scratch: ScratchLayouts,
    *,
    recorder: _PreparationRecorder | None = None,
) -> list[str]:
    from .chained_body_program import emit_body_program

    return emit_body_program(
        cg, plan, pipeline, execution, vector, unroll, scratch, recorder=recorder
    )


def emit_preparation_pipeline(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    prologue: Sequence[str],
    scratch: ScratchLayouts,
    vector: VectorStaging,
    unroll: BoundedProducerUnroll,
    seed_tiling: SeedTiling | None = None,
    cache_layouts: PointwiseCacheLayouts | None = None,
    *,
    compact_preparation: bool = False,
    leaf_issue_batching: bool = False,
    broadcast_retention: bool = False,
    completed_member_store: bool = False,
    island_consumers: bool = False,
    fragment_epilogues: bool = False,
) -> list[str]:
    """Emit role branches from reusable stage/expression/loop components."""
    from .chained_body_program import emit_body_program
    from .chained_leaf_sets import capture_leaf_set_witness
    from .chained_leaf_sets import extend_preparation_leaves
    from .chained_loop import advance_carries
    from .chained_loop import initialize_loop
    from .chained_loop_tmem_carry_transport import prepare_loop_tmem_carry
    from .chained_loop_tmem_transport import prepare_loop_tmem_transports
    from .chained_matmul import _emit_store
    from .chained_matmul import _indent
    from .chained_matmul import _operand_domain
    from .chained_matmul import _resolved_extent
    from .chained_matmul import _shape
    from .chained_matmul import _UnsupportedChain
    from .chained_pipeline_storage import capture_storage_revision
    from .chained_pipeline_storage import finalize_pipeline_storage
    from .chained_pipeline_storage import select_stage_transports
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_preparation_leaves import insert_preparation_leaf
    from .chained_preparation_leaves import preparation_leaf_candidates
    from .chained_prepared_groups import coallocate_prepared_groups
    from .chained_prepared_groups import prepared_group_candidates
    from .chained_prepared_operands import plan_prepared_operands
    from .chained_prepared_values import bind_frame_buffers
    from .chained_prepared_values import frontier_bindings
    from .chained_recurrence_body import bind_recurrence_body
    from .chained_tcgen_stage import allocate_tmem_resources
    from .chained_tcgen_stage import free_stages
    from .chained_tmem_drains import DrainTiling
    from .tcgen05_config import CuteTcgen05Config

    loop = plan.loop
    assert loop is not None
    drain_tiling = DrainTiling(
        cast(
            "int",
            cg.device_function.config.config.get("cute_chained_drain_tile_columns", 0),
        )
    )
    if type(compact_preparation) is not bool:
        raise ValueError("compact preparation must be an explicit boolean")
    if (
        type(fragment_epilogues) is not bool
        or fragment_epilogues
        and not compact_preparation
    ):
        raise exc.BackendUnsupported(
            "cute", "fragment epilogues require explicit compact preparation"
        )
    if type(island_consumers) is not bool or (
        island_consumers
        and (
            not compact_preparation
            or cg.device_function.config.config.get("cute_chained_register_islands")
            is not True
        )
    ):
        raise exc.BackendUnsupported(
            "cute", "island consumers require compact register-island preparation"
        )
    if (
        type(broadcast_retention) is not bool
        or broadcast_retention
        and not compact_preparation
    ):
        raise exc.BackendUnsupported(
            "cute", "broadcast retention requires accepted physical preparation"
        )
    if type(completed_member_store) is not bool:
        raise ValueError("completed member store must be an explicit boolean")
    if type(leaf_issue_batching) is not bool or (
        leaf_issue_batching and not compact_preparation
    ):
        raise exc.BackendUnsupported(
            "cute", "leaf issue batching requires compact preparation"
        )
    frame, recurrence = pipeline.frame, pipeline.recurrence
    slots, prep_threads = pipeline.slots, pipeline.preparation_threads
    cohorts = pipeline.cohorts
    plan = replace(
        plan,
        loop_workspace=recurrence.layout,
        warp_mma_stages=frozenset(stage.group.stages[0] for stage in frame.stages),
    )
    if plan_prepared_operands(plan, frame, recurrence) != pipeline.prepared_operands:
        raise ValueError("prepared operand proof does not match the current pipeline")
    candidates = prepared_group_candidates(plan, frame, recurrence)
    if candidates is None:
        raise ValueError("prepared group proof does not match the current pipeline")
    admitted = []
    for candidate in candidates:
        for member in candidate.members:
            node = member.buffer.node
            assert node is not None
            _, coords = member.geometry.operand(
                "b", "chain_prepared_row", "chain_prepared_k"
            )
            try:
                if _operand_domain(cg, node, coords, plan):
                    break
            except _UnsupportedChain:
                break
        else:
            admitted.append(candidate)
    # Keep the existing resource envelope. The union lifetime may increase
    # fragmentation, so failure simply retains all ordinary member fills.
    placement = coallocate_prepared_groups(
        plan,
        frame,
        recurrence,
        tuple(admitted),
        frame_capacity_bytes=frame.layout.allocated_bytes,
    )
    if placement is not None and placement.groups:
        frame = placement.frame
        # Coallocation moves unrelated regions too; never retain old native
        # operand handles or preparation-stage A/B offsets after repacking.
        prepared_operands = plan_prepared_operands(plan, frame, recurrence)
        assert prepared_operands is not None
        pipeline = replace(
            pipeline,
            frame=frame,
            prepared_operands=prepared_operands,
            prepared_groups=placement.groups,
        )
    if (
        cg.device_function.config.config.get("cute_chained_leaf_pipeline")
        == "rectangular_tma"
    ):
        leaves = tuple(
            sorted(
                preparation_leaf_candidates(cg, plan, frame),
                key=lambda item: (item.first_event, item.name),
            )
        )
        leaf_count = cast(
            "int", cg.device_function.config.config.get("cute_chained_leaf_count", 1)
        )
        witness = (
            capture_leaf_set_witness(
                plan, frame, leaves, prepared_groups=pipeline.prepared_groups
            )
            if leaf_count > 1
            else None
        )
        for leaf in leaves:
            extended = insert_preparation_leaf(
                frame,
                leaf,
                capacity=frame.layout.allocated_bytes,
                prepared_groups=pipeline.prepared_groups,
            )
            if extended is None:
                continue
            frame = extended
            # Existing byte offsets do not move, but every action/lifetime and
            # all region-bearing proofs must refer to the extended frame.
            prepared_operands = plan_prepared_operands(plan, frame, recurrence)
            groups = prepared_group_candidates(plan, frame, recurrence)
            assert prepared_operands is not None and groups is not None
            rebound = {candidate.group: candidate for candidate in groups}
            pipeline = replace(
                pipeline,
                frame=frame,
                prepared_leaves=(leaf,),
                prepared_operands=prepared_operands,
                prepared_groups=tuple(
                    replace(binding, candidate=rebound[binding.candidate.group])
                    for binding in pipeline.prepared_groups
                ),
            )
            break
        else:
            raise _UnsupportedChain(
                "rectangular TMA requires a proved preparation leaf and free frame interval"
            )
        if leaf_count > 1:
            # This is only an append-placement bound. The complete allocation,
            # including all recurrence and protocol storage, is proved below
            # after the same attempt's late transport decisions.
            capacity = _frame_planning_capacity(
                plan, pipeline, compact_preparation=compact_preparation
            )
            extended_set = (
                extend_preparation_leaves(
                    plan,
                    witness,
                    frame,
                    recurrence,
                    pipeline.prepared_leaves,
                    pipeline.prepared_operands,
                    pipeline.prepared_groups,
                    max_count=leaf_count,
                    capacity=capacity,
                )
                if witness is not None
                else None
            )
            if extended_set is None or len(extended_set.leaves) < 2:
                raise exc.BackendUnsupported(
                    "cute",
                    "multiple preparation leaves require proved repeated raw loads and frame capacity",
                )
            frame = extended_set.frame
            pipeline = replace(
                pipeline,
                frame=frame,
                prepared_leaves=extended_set.leaves,
                prepared_operands=extended_set.prepared_operands,
                prepared_groups=extended_set.prepared_groups,
                protocol_bytes=(
                    ((len(plan.dots) * 8 + 127) // 128 + 1) * 128
                    + LeafSetProtocol(
                        slots, len(extended_set.leaves), cohorts is not None
                    ).allocated_bytes
                ),
            )
        for leaf in pipeline.prepared_leaves:
            cg.cute_wrapper_plans.append(leaf.wrapper)
            cg.device_function.wrapper_only_params.extend(
                [f"{leaf.name}_atom", f"{leaf.name}_tensor"]
            )
    # Physical coverage does not prove inherited logical padding is absent.
    # Ordinary operand fills apply that domain mask after evaluating an image;
    # changing its stored value could affect other (non-MMA) consumers. Keep
    # the original view and fill for every image with an unproved domain.
    rejected_images = set()
    for operand in pipeline.prepared_operands:
        node = operand.buffer.node
        assert node is not None
        try:
            domain = _operand_domain(
                cg, node, ("chain_prepared_row", "chain_prepared_k"), plan
            )
        except _UnsupportedChain:
            rejected_images.add(operand.buffer.name)
        else:
            if domain:
                rejected_images.add(operand.buffer.name)
    pipeline = replace(
        pipeline,
        prepared_operands=tuple(
            operand
            for operand in pipeline.prepared_operands
            if operand.buffer.name not in rejected_images
        ),
    )
    if cg.device_function.config.config.get("cute_chained_operand_retention") is True:
        pipeline = _retain_operands(
            cg, plan, pipeline, compact_preparation=compact_preparation
        )
        frame = pipeline.frame
    crops = None
    if compact_preparation:
        from .chained_preparation_actions import extend_native_crop_owners

        pipeline, crops = extend_native_crop_owners(plan, pipeline)
        frame = pipeline.frame
    if (
        cg.device_function.config.config.get("cute_chained_scan_producer_retention")
        is True
    ):
        from .chained_scan_producer import plan_scan_producer

        scan = plan_scan_producer(
            plan,
            pipeline,
            {
                node: _shape(node)
                for node in frame.cut.region.nodes
                if isinstance(node.meta.get("val"), torch.Tensor)
            },
            scratch_mode=scratch.mode,
        )
        if scan is None:
            raise exc.BackendUnsupported(
                "cute", "scan producer has no complete phase/lease proof"
            )
        pipeline = replace(pipeline, scan_producer=scan)
    frontier = frontier_bindings(frame)
    revision = None
    output_lease_requested = (
        cg.device_function.config.config.get("cute_chained_output_lease_snapshot")
        is True
    )
    needs_finalization = (
        compact_preparation
        or completed_member_store
        or cohorts is not None
        or len(pipeline.prepared_leaves) > 1
        or pipeline.operand_retention is not None
        or output_lease_requested
        or pipeline.scan_producer is not None
    )
    preparation_storage = None
    compact_prep = None
    compact_unroll = None
    if compact_preparation:
        from .chained_leaf_schedule import plan_leaf_schedule
        from .chained_leaf_schedule_emission import LeafEmissionCapture
        from .chained_leaf_schedule_emission import seal_leaf_schedule
        from .chained_preparation_actions import build_accepted_preparation
        from .chained_preparation_storage import bind_preparation_storage
        from .chained_preparation_storage import plan_accepted_preparation_storage
        from .chained_preparation_storage import plan_scheduled_preparation_storage
        from .chained_prepared_image_emission import bind_raw_widening_transfers
        from .chained_prepared_image_transfers import discover_raw_widening_transfers

        if cg.device_function.config.config.get("cute_chained_collective_retention"):
            # That independent callback still uses numeric stage workspaces;
            # it cannot be installed into this symbolic physical table yet.
            raise exc.BackendUnsupported(
                "cute", "compact preparation does not bind row-collective workspaces"
            )
        shapes = {
            node: _shape(node)
            for node in frame.cut.region.nodes
            if isinstance(node.meta.get("val"), torch.Tensor)
        }
        raw_candidates = discover_raw_widening_transfers(plan, frame, shapes)
        if raw_candidates is None:
            raise exc.BackendUnsupported("cute", "raw preparation revision changed")
        raw = (
            bind_raw_widening_transfers(plan, frame, shapes, raw_candidates)
            if raw_candidates
            else None
        )
        if raw_candidates and raw is None:
            raise exc.BackendUnsupported("cute", "raw preparation cannot be bound")
        if raw is not None and output_lease_requested:
            raise exc.BackendUnsupported(
                "cute", "compact raw preparation needs a compatible output lease proof"
            )
        compact_prep = (
            cohorts.preparation_execution()
            if cohorts is not None
            else ChainedExecution(
                prep_threads,
                thread="chain_prep_thread",
                warp="chain_prep_warp",
                sync="chain_prep_barrier.arrive_and_wait()",
            )
        )
        compact_factor = cast(
            "int",
            cg.device_function.config.config.get("cute_chained_preparation_unroll", 0),
        )
        compact_unroll = (
            BoundedProducerUnroll(compact_factor) if compact_factor else unroll
        )
        leaf_capture = LeafEmissionCapture() if leaf_issue_batching else None
        preparation_options: _AcceptedPreparationOptions = {}
        if leaf_capture is not None:
            preparation_options["leaf_capture"] = leaf_capture
        if broadcast_retention:
            preparation_options["broadcast_retention"] = True
        if island_consumers:
            preparation_options["island_consumers"] = True
        if fragment_epilogues:
            preparation_options["fragment_epilogues"] = True
        accepted = build_accepted_preparation(
            cg,
            plan,
            pipeline,
            compact_prep,
            vector,
            compact_unroll,
            scratch,
            raw=raw,
            crops=crops,
            **preparation_options,
        )
        physical = plan_accepted_preparation_storage(
            plan,
            pipeline,
            accepted,
            capacity_bytes=CuteTcgen05Config.per_cta_smem_capacity_bytes(
                plan.dots[0].meta["val"].device
            ),
        )
        if leaf_capture is not None:
            recorded = leaf_capture.seal(accepted)
            schedule = (
                plan_leaf_schedule(plan, pipeline, physical)
                if physical is not None
                else None
            )
            scheduled = (
                seal_leaf_schedule(plan, pipeline, recorded, schedule)
                if schedule is not None
                else None
            )
            physical = (
                plan_scheduled_preparation_storage(plan, pipeline, scheduled)
                if scheduled is not None
                else None
            )
            if physical is None:
                raise exc.BackendUnsupported(
                    "cute",
                    "leaf issue batching lacks a completed schedule and extended leases",
                )
        preparation_storage = (
            bind_preparation_storage(
                plan, pipeline, physical, cache_layouts=cache_layouts
            )
            if physical is not None
            else None
        )
        if preparation_storage is None:
            raise exc.BackendUnsupported(
                "cute", "accepted preparation has no complete physical binding"
            )
    if needs_finalization:
        revision = capture_storage_revision(
            plan,
            pipeline,
            {
                node: _shape(node)
                for node in frame.cut.region.nodes
                if isinstance(node.meta.get("val"), torch.Tensor)
            },
        )
        if revision is None:
            raise exc.BackendUnsupported(
                "cute", "preparation pipeline storage revision is invalid"
            )
    transport_plan, transport_frontier = plan, frontier
    if (
        preparation_storage is not None
        and preparation_storage.physical.accepted.raw is not None
    ):
        transport_raw = preparation_storage.physical.accepted.raw
        transport_plan = replace(plan, prepared_widenings=transport_raw)
        transport_frontier = transport_raw.recurrence_boundaries(frontier)
    # These proofs retain emitted original-expression statements. They must
    # capture the same typed boundary map that their recurrence consumers use.
    transports, columns = prepare_loop_tmem_transports(
        cg,
        transport_plan,
        transport_frontier,
        tuple(stage.group for stage in recurrence.stages),
    )
    carry = prepare_loop_tmem_carry(
        cg,
        transport_plan,
        transport_frontier,
        tuple(stage.group for stage in recurrence.stages),
        pipeline.residency,
        columns,
        drain_tile_columns=drain_tiling.columns,
        snapshot_cut=(
            frame.cut
            if cg.device_function.config.config.get(
                "cute_chained_snapshot_tile_columns"
            )
            == 32
            and seed_tiling is not None
            and seed_tiling.max_columns == 32
            else None
        ),
    )
    snapshot_tile_columns = cast(
        "int",
        cg.device_function.config.config.get("cute_chained_snapshot_tile_columns", 0),
    )
    if snapshot_tile_columns and carry is None:
        raise exc.BackendUnsupported(
            "cute", "streamed snapshot requires a proven resident carry"
        )
    if carry is not None:
        columns = carry.required_columns
    shared_allocations = (
        f"chain_a_workspace = cute.arch.alloc_smem(cutlass.BFloat16, {recurrence.a_bytes // 2}, alignment=128)",
        f"chain_b_workspace = cute.arch.alloc_smem(cutlass.BFloat16, {recurrence.b_bytes // 2}, alignment=128)",
        f"chain_c_workspace = cute.arch.alloc_smem(cutlass.Float32, {recurrence.layout.allocated_bytes // 4}, alignment=128)",
        f"chain_frames = cute.arch.alloc_smem(cutlass.Uint8, {slots * frame.layout.allocated_bytes}, alignment=128)",
        f"chain_slot_bars = cute.arch.alloc_smem(cutlass.Int64, {pipeline.slot_barrier_count}, alignment=128)",
    )
    selected_transports = None
    carry_pointers = None
    storage = None
    if needs_finalization:
        assert revision is not None
        selected_transports = select_stage_transports(pipeline, transports, carry)
        if selected_transports is None:
            raise exc.BackendUnsupported(
                "cute", "preparation pipeline transports are inconsistent"
            )
        output_lease_proof = None
        if output_lease_requested:
            from .chained_output_lease import plan_output_lease_snapshot

            output_lease_proof = plan_output_lease_snapshot(
                plan, pipeline, selected_transports, revision=revision
            )
            if output_lease_proof is None:
                raise exc.BackendUnsupported(
                    "cute",
                    "output lease snapshot has no complete terminal frame read proof",
                )
        completed_store_plan = None
        if completed_member_store and not output_lease_requested:
            from .chained_completed_store import prepare_completed_store_plan

            completed_store_plan = prepare_completed_store_plan(
                cg,
                plan,
                transport_plan,
                pipeline,
                selected_transports,
                transport_frontier,
                ChainedExecution(
                    pipeline.recurrence_threads,
                    thread="chain_recurrence_thread",
                    warp="chain_recurrence_warp",
                    sync="chain_recurrence_barrier.arrive_and_wait()",
                ),
                preparation_storage,
                carry,
            )
        storage = finalize_pipeline_storage(
            plan,
            pipeline,
            selected_transports,
            revision=revision,
            capacity_bytes=CuteTcgen05Config.per_cta_smem_capacity_bytes(
                plan.dots[0].meta["val"].device
            ),
            cohorts=cohorts,
            output_lease=output_lease_proof,
            completed_store=completed_store_plan,
            preparation=preparation_storage if compact_preparation else None,
        )
        if storage is None:
            raise exc.BackendUnsupported(
                "cute",
                "preparation pipeline requires a proved complete post-transport allocation",
            )
        recurrence = storage.recurrence
        plan = replace(plan, loop_workspace=recurrence.layout)
        pool_types = {
            "a": ("chain_a_workspace", "cutlass.BFloat16", 2),
            "b": ("chain_b_workspace", "cutlass.BFloat16", 2),
            "recurrence": ("chain_c_workspace", "cutlass.Float32", 4),
            "frames": ("chain_frames", "cutlass.Uint8", 1),
            "endpoints": ("chain_endpoints", "cutlass.Float32", 4),
            "slot_barriers": ("chain_slot_bars", "cutlass.Int64", 8),
            "output_snapshot": ("chain_output_snapshots", "cutlass.Uint8", 1),
        }
        shared_allocations = tuple(
            f"{name} = cute.arch.alloc_smem({dtype}, {size // width}, alignment=128)"
            for pool, size in storage.allocations
            if pool in pool_types
            for name, dtype, width in (pool_types[pool],)
        )
        carry_pointers = {
            view.name: (
                f"cute.recast_ptr(chain_frames + {view.byte_offset}, dtype=cutlass.Float32)"
                if view.pool == "frames"
                else f"{'chain_c_workspace' if view.pool == 'recurrence' else 'chain_endpoints'} + {view.byte_offset // 4}"
            )
            for view in storage.carry_views
        }
    lines = [
        *prologue,
        *allocate_tmem_resources(
            columns,
            len(plan.dots),
            plan.threads,
            shared_allocations=shared_allocations,
        ),
    ]
    if carry_pointers is None:
        lines.extend(initialize_loop(cg, plan, scratch, recurrence.layout))
    else:
        lines.extend(
            initialize_loop(
                cg, plan, scratch, recurrence.layout, carry_pointers=carry_pointers
            )
        )
    regions = {region.name: region for region in recurrence.layout.regions}
    if transports or carry is not None:
        lines.append(
            "chain_tmem_barrier = chain_pipeline.NamedBarrier(barrier_id=4, num_threads=128)"
        )
    carry_endpoints = None
    if carry is not None:
        if completed_member_store:
            from .chained_completed_store import emit_completed_carry_entry

            assert storage is not None
            entry_lines, carry_endpoints = emit_completed_carry_entry(
                plan, pipeline, storage, carry
            )
            if tuple(entry_lines) != carry_endpoints.entry:
                raise _UnsupportedChain("completed carry upload body changed")
            lines.extend(entry_lines)
        else:
            lines.extend([*carry.view(), *carry.shared_transfer(upload=True)])
    for stage, shape in enumerate(plan.shapes):
        name = f"chain_{stage}_c"
        if name in regions:
            lines.append(
                f"{name} = cute.make_tensor(chain_c_workspace + {regions[name].byte_offset // 4}, {scratch.layout(name, shape[:2])})"
            )
    lines.extend(
        [
            "from helion._compiler.cute import warp_specialized_primitives as chain_sync",
            *(
                [
                    f"chain_prep_barrier = chain_pipeline.NamedBarrier(barrier_id=2, num_threads={prep_threads})"
                ]
                if cohorts is None
                else []
            ),
            f"chain_recurrence_barrier = chain_pipeline.NamedBarrier(barrier_id=3, num_threads={pipeline.recurrence_threads})",
            "if chain_thread == 0:",
            *(
                f"    cute.arch.mbarrier_init(chain_slot_bars + {index}, 1)"
                for index in range(pipeline.slot_barrier_count)
            ),
            "cute.arch.mbarrier_init_fence()",
            "cute.arch.sync_threads()",
        ]
    )
    prep = (
        compact_prep
        if compact_prep is not None
        else (
            cohorts.preparation_execution()
            if cohorts is not None
            else ChainedExecution(
                prep_threads,
                thread="chain_prep_thread",
                warp="chain_prep_warp",
                sync="chain_prep_barrier.arrive_and_wait()",
            )
        )
    )
    consumer = ChainedExecution(
        pipeline.recurrence_threads,
        thread="chain_recurrence_thread",
        warp="chain_recurrence_warp",
        sync="chain_recurrence_barrier.arrive_and_wait()",
    )
    output_lease = None
    if output_lease_requested:
        from .chained_output_lease import bind_output_lease_snapshot

        assert storage is not None
        output_lease = bind_output_lease_snapshot(
            plan, pipeline, storage, frontier, consumer
        )
        if output_lease is None:
            raise exc.BackendUnsupported(
                "cute",
                "output lease snapshot storage or runtime binding is unsupported",
            )
        lines.extend(output_lease.setup)
    step = cg.device_function.resolved_block_size(loop.block_id)
    assert step is not None
    step = _resolved_extent(step)
    assert step > 0
    iteration = f"((chain_loop_index - chain_loop_begin) // {step})"
    header = f"for chain_loop_index in cutlass.range(chain_loop_begin, chain_loop_end, {step}, unroll=1):"
    producer_header = header if cohorts is None else cohorts.producer_header(step)
    preparation_unroll = unroll
    preparation_factor = cast(
        "int",
        cg.device_function.config.config.get("cute_chained_preparation_unroll", 0),
    )
    if preparation_factor:
        preparation_unroll = BoundedProducerUnroll(preparation_factor)
    if compact_unroll is not None:
        preparation_unroll = compact_unroll
    # Both the arena base and every slot stride are 128-byte aligned. Dynamic
    # pointer arithmetic loses that proof in CuTe; restore it before creating
    # typed views used by shared-memory matrix loads.
    frame_stride = (
        frame.layout.allocated_bytes
        if preparation_storage is None
        else preparation_storage.stride
    )
    assert frame_stride % 128 == 0
    common = [
        f"chain_iteration = {iteration}",
        f"chain_slot = chain_iteration % {slots}",
        f"chain_generation = chain_iteration // {slots}",
        f"chain_frame_address = chain_frames + chain_slot * {frame_stride}",
        "chain_frame = cute.make_ptr(cutlass.Uint8, chain_frame_address.toint(), cute.AddressSpace.smem, assumed_align=128)",
    ]
    if scratch.read_buffers is not None:
        scratch.read_buffers |= frozenset(frontier.values())
    if preparation_storage is None:
        views = bind_frame_buffers(
            frame,
            "chain_frame",
            scratch,
            prepared_operands=pipeline.prepared_operands,
            prepared_groups=pipeline.prepared_groups,
            prepared_leaves=pipeline.prepared_leaves,
            cache_layouts=cache_layouts,
        )
        producer_views = (
            views
            if pipeline.scan_producer is None
            else bind_frame_buffers(
                frame,
                "chain_frame",
                scratch,
                prepared_operands=pipeline.prepared_operands,
                prepared_groups=pipeline.prepared_groups,
                prepared_leaves=pipeline.prepared_leaves,
                cache_layouts=cache_layouts,
                scan_producer=pipeline.scan_producer,
            )
        )
        prepared_lines = _prepare(
            cg, plan, pipeline, prep, vector, preparation_unroll, scratch
        )
    else:
        views = bind_frame_buffers(
            frame,
            "chain_frame",
            scratch,
            prepared_operands=pipeline.prepared_operands,
            prepared_groups=pipeline.prepared_groups,
            prepared_leaves=pipeline.prepared_leaves,
            cache_layouts=cache_layouts,
            preparation_storage=preparation_storage,
            scan_producer=pipeline.scan_producer,
        )
        producer_views = views
        assert storage is not None
        prepared_lines = preparation_storage.consume_body(
            storage, vector, preparation_unroll
        )
    producer_body = [
        *common,
        f"if chain_iteration >= {slots}:",
        f"    cute.arch.mbarrier_wait(chain_slot_bars + {slots} + chain_slot, (chain_generation - 1) & 1)",
        *producer_views,
        *prepared_lines,
        "if chain_prep_warp == 0:",
        "    chain_sync.arrive_mbarrier(chain_slot_bars + chain_slot)",
    ]
    if preparation_unroll is not unroll:
        preparation_unroll.validate()
    consumer_body = [
        *common,
        "cute.arch.mbarrier_wait(chain_slot_bars + chain_slot, chain_generation & 1)",
        *views,
    ]
    if (
        preparation_storage is not None
        and preparation_storage.physical.accepted.raw is not None
    ):
        # Finalization may change only loop-workspace placement. Rebind the
        # original half/widening graph against that exact recurrence plan.
        from .chained_prepared_image_emission import bind_raw_widening_transfers
        from .chained_prepared_image_transfers import discover_raw_widening_transfers

        shapes = dict(preparation_storage.physical.accepted.revision.shapes)
        transfers = discover_raw_widening_transfers(plan, frame, shapes)
        raw = (
            bind_raw_widening_transfers(plan, frame, shapes, transfers)
            if transfers
            else None
        )
        original_raw = preparation_storage.physical.accepted.raw
        if raw is None or tuple(item.buffer for item in raw.bindings) != tuple(
            item.buffer for item in original_raw.bindings
        ):
            raise exc.BackendUnsupported("cute", "final raw widening proof changed")
        frontier = raw.recurrence_boundaries(frontier)
        plan = replace(plan, prepared_widenings=raw)
    if carry is not None:
        consumer_body.extend(
            carry.snapshot(consumer, tile_columns=snapshot_tile_columns)
            if snapshot_tile_columns
            else carry.snapshot(consumer)
        )
    body = bind_recurrence_body(
        cg,
        plan,
        pipeline,
        recurrence,
        frontier,
        consumer,
        selected_transports=selected_transports,
        storage=storage,
        transports=transports,
        carry=carry,
        seed_tiling=seed_tiling,
        output_lease=output_lease,
        carry_endpoints=carry_endpoints,
        completed_member_store=completed_member_store,
    )
    stage_lines = emit_body_program(
        cg,
        plan,
        None,
        consumer,
        vector,
        unroll,
        scratch,
        recurrence_body=body,
        drain_tiling=drain_tiling,
    )
    body.validate_return(stage_lines)
    consumer_body.extend(stage_lines)
    completed_action = body.completed_action
    completed_lines = body.completed_lines
    epilogue = (
        frontier
        if output_lease is None
        else output_lease.epilogue_boundaries(plan, frontier)
    )
    for index, store in enumerate(loop.region.stores):
        if completed_action is not None and store is completed_action.candidate.store:
            consumer_body.extend(
                completed_action.consume(
                    plan,
                    store,
                    epilogue,
                    consumer,
                    f"chain_store_{index}",
                    completed_lines,
                )
            )
            continue
        consumer_body.extend(
            _emit_store(
                cg,
                plan,
                store,
                epilogue,
                [],
                [],
                f"chain_store_{index}",
                execution=consumer,
            )
        )
    if completed_member_store:
        if completed_action is None:
            raise _UnsupportedChain("completed member store was not emitted")
        completed_action.validate_consumed()
    consumer_body.extend(
        advance_carries(
            cg,
            plan,
            epilogue,
            scratch,
            execution=consumer,
            resident_carries=(
                frozenset()
                if carry is None
                else frozenset((carry.candidate.carry_index,))
            ),
        )
    )
    if output_lease is None:
        consumer_body.extend(
            [
                "if chain_recurrence_warp == 0:",
                f"    chain_sync.arrive_mbarrier(chain_slot_bars + {slots} + chain_slot)",
            ]
        )
    lines.extend(
        [
            f"if chain_thread < {prep_threads}:",
            *(
                [
                    "    chain_prep_thread = chain_thread",
                    "    chain_prep_warp = chain_warp",
                ]
                if cohorts is None
                else [f"    {line}" for line in cohorts.preparation_bindings()]
            ),
            _indent([producer_header, _indent(producer_body)]),
            "else:",
            f"    chain_recurrence_thread = chain_thread - {prep_threads}",
            f"    chain_recurrence_warp = chain_warp - {prep_threads // 32}",
            _indent([header, _indent(consumer_body)]),
            *([] if carry_endpoints is not None else ["cute.arch.sync_threads()"]),
        ]
    )
    if carry_endpoints is not None:
        assert carry is not None
        drain_lines = carry_endpoints.drain(plan, carry)
        carry_endpoints.validate_drained(plan, drain_lines)
        lines.extend(drain_lines)
        assert completed_action is not None
        completed_action.validate_consumed()
    elif carry is not None:
        lines.extend(carry.shared_transfer(upload=False))
    if carry is not None and carry.drain_panels is not None:
        drain_tiling.activated = True
    drain_tiling.check(
        cg.device_function.config.config.get("cute_chained_drain_tile_columns", 0)
    )
    drain_tiling.validate()
    for index, store in enumerate(loop.final_stores):
        lines.extend(
            _emit_store(cg, plan, store, {}, [], [], f"chain_final_store_{index}")
        )
    lines.extend(free_stages())
    return lines
