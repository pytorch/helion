"""Ordered body actions, independent of a root or preparation-ring envelope.

Root resources are the original separately allocated images, never a synthetic
preparation frame. The preparation adapter keeps its real frame and receipts.
Both execute fill/MMA spans through the same ordered interpreter below.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import torch

from ... import exc
from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_execution import ChainedExecution
from .chained_leaf_sets import LeafSetProtocol
from .chained_preparation_frame import PreparationAction
from .chained_recurrence_workspace import RecurrenceStage
from .chained_root_stage import RootStageAction
from .chained_root_stage import _execution_fields
from .chained_root_stage import _pair_revision
from .chained_root_stage import _startup_value
from .chained_root_stage import root_stage_progress
from .chained_root_stage import root_stage_state
from .chained_root_warp_stage import RootWarpStageAction
from .chained_warp_bridge import RootWarpBridgeAction
from .chained_warp_bridge import RootWarpBridgeSequence
from .chained_warp_bridge import _staged_facts
from .chained_warp_stage import CompletedWarpStage
from .chained_warp_stage import _snapshot
from .chained_warp_stage import warp_revision
from .prepared_continuation import ContinuationAction
from .prepared_continuation import PreparedBodyLowering
from .prepared_continuation import PreparedContinuation
from .prepared_continuation import action_facts as continuation_action_facts
from .prepared_epoch_body import EpochBranch
from .prepared_epoch_body import EpochRange
from .prepared_epoch_body import EpochSourceBody
from .prepared_epoch_issue import DescriptorIssue
from .prepared_epoch_memory import EpochMemoryAction
from .prepared_epoch_protocol import EpochEvent
from .prepared_epoch_protocol import EpochSynchronizationAction
from .prepared_epoch_setup import EpochSetup
from .prepared_epoch_state import StateArrival
from .prepared_state_body import RootStateBody
from .prepared_state_body import StateEffect

if TYPE_CHECKING:
    from typing import TypedDict

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pointwise_cache import PointwiseReadCache
    from .chained_pointwise_inplace import PointwiseInplace
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_pointwise_unroll import PointwiseUnroll
    from .chained_preparation_actions import _PreparationRecorder
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_recurrence_body import RecurrenceBody
    from .chained_root_stage import RootRhsCompletion
    from .chained_root_stage import RootSeededInputs
    from .chained_root_stage import RootStageCompletion
    from .chained_root_stage import RootStageSequence
    from .chained_scratch_layout import ScratchLayouts
    from .chained_stage_operands import DirectStageOperands
    from .chained_tcgen_stage import StageGeometry
    from .chained_tmem_drains import DrainTiling
    from .chained_vector_native import NativeReadInputs
    from .chained_vector_stage import VectorStaging
    from .chunk_recurrence import CuteChunkRecurrencePlan
    from .prepared_graph_schedule import ContractionGraph
    from .prepared_graph_schedule import GroupedContractionSchedule

    class _ScanProducerOptions(TypedDict, total=False):
        recorder: _PreparationRecorder
        native_inputs: NativeReadInputs


@dataclass(frozen=True)
class RootDeferredRhsEnqueue:
    inputs: RootSeededInputs


@dataclass(frozen=True)
class RootDeferredSpan:
    action: RootDeferredRhsEnqueue
    completion: RootRhsCompletion
    start: int
    lines: tuple[str, ...]

    @property
    def stop(self) -> int:
        return self.start + len(self.lines)

    def facts(self) -> object:
        return (
            self.action,
            self.action.inputs,
            self.completion,
            self.completion.queued,
            self.completion.before,
            self.completion.progress,
            self.start,
            self.lines,
        )


@dataclass(frozen=True)
class BodyProgram:
    actions: tuple[
        PreparationAction
        | RootStageAction
        | RootDeferredRhsEnqueue
        | RootWarpStageAction
        | RootWarpBridgeAction
        | RecurrenceStage
        | ContinuationAction
        | StateEffect
        | StateArrival
        | DescriptorIssue
        | EpochEvent
        | EpochSynchronizationAction
        | EpochSetup
        | EpochMemoryAction
        | EpochBranch
        | EpochRange,
        ...,
    ]

    def facts(self) -> object:
        return _startup_value(
            tuple(
                continuation_action_facts(a)
                if isinstance(a, ContinuationAction)
                else a.facts()
                if isinstance(
                    a,
                    (
                        StateEffect,
                        StateArrival,
                        DescriptorIssue,
                        EpochEvent,
                        EpochSynchronizationAction,
                        EpochSetup,
                        EpochMemoryAction,
                        EpochBranch,
                        EpochRange,
                    ),
                )
                else (a.sequence, a.stage)
                if isinstance(a, RootStageAction)
                else (a.inputs, a.inputs.lines)
                if isinstance(a, RootDeferredRhsEnqueue)
                else (id(a), a.facts())
                if isinstance(a, RootWarpStageAction)
                else (id(a), id(a.sequence), a.sequence.facts(), a.stage)
                if isinstance(a, RootWarpBridgeAction)
                else (id(a), a.group, a.read_event, a.publication_event)
                if isinstance(a, RecurrenceStage)
                else (
                    a.event,
                    a.kind,
                    a.nodes,
                    a.stages,
                    a.source_stage,
                    a.reads,
                    a.writes,
                )
                for a in self.actions
            )
        )


def preparation_body_program(pipeline: PreparationPipeline) -> BodyProgram:
    return BodyProgram(pipeline.frame.actions)


@dataclass
class RootWarpBody:
    """Complete original warp actions; no workspace or readiness substitute."""

    codegen: GenerateAST
    plan: ChainedMatmulPlan
    owner: RootWarpStageAction | RootWarpBridgeSequence
    program: BodyProgram
    boundaries: dict[Node, str]
    staged: list[chain._StagedInput]
    scratch: ScratchLayouts
    prefix: list[str]
    execution: ChainedExecution
    context: object
    accepted_prefix: tuple[str, ...]
    cursor: int = 0
    pending: object = None
    publication: CompletedWarpStage | None = None
    consumed: bool = False

    def facts(self) -> object:
        return (
            id(self),
            id(self.codegen),
            id(self.plan),
            id(self.owner),
            id(self.program),
            self.program.facts(),
            id(self.boundaries),
            id(self.staged),
            id(self.scratch),
            self.scratch.mode,
            self.scratch.read_buffers,
            id(self.prefix),
            id(self.execution),
            _execution_fields(self.execution),
            warp_revision(self.plan, aliases=False),
            _snapshot(self.codegen.device_function.config.config),
        )

    def check(self, *, finished: bool = False) -> None:
        if (
            self.context != self.facts()
            or self.owner._body.check(self.owner) is not self
            or not self.owner._body.accepted
            or self.consumed is not finished
            or tuple(self.prefix) != self.accepted_prefix
        ):
            raise chain._UnsupportedChain("root warp body context or prefix changed")

    def begin(
        self,
        owner: RootWarpStageAction | RootWarpBridgeSequence,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        staged: list[chain._StagedInput],
        scratch: ScratchLayouts,
        prefix: list[str],
        stage: int,
    ) -> None:
        self.check()
        if (
            owner is not self.owner
            or cg is not self.codegen
            or plan is not self.plan
            or boundaries is not self.boundaries
            or staged is not self.staged
            or scratch is not self.scratch
            or prefix is not self.prefix
            or self.pending is not None
            or stage != self.cursor
            or stage >= len(self.program.actions)
            or (
                self.publication is not None
                and not self.publication.matches(cg, plan, boundaries, prefix)
            )
        ):
            raise chain._UnsupportedChain("root warp body call changed or repeated")
        self.pending = self.program.actions[stage]

    def accept(
        self, action: RootWarpStageAction | RootWarpBridgeAction, lines: list[str]
    ) -> None:
        self.check()
        owner = self.owner
        if isinstance(owner, RootWarpStageAction):
            completion = owner._completion
            publication = None if completion is None else completion._state.publication
            complete = owner._consumed is True
            staged = owner._staged
        else:
            publication = owner._state.publication
            complete = owner._next_stage == owner._state.next_stage == self.cursor + 1
            staged = owner._state.staged
        if (
            self.pending is not action
            or self.cursor >= len(self.program.actions)
            or self.program.actions[self.cursor] is not action
            or not complete
            or publication is None
            or publication is self.publication
            or staged != _staged_facts(self.staged)
            or not publication.matches(
                self.codegen, self.plan, self.boundaries, (*self.prefix, *lines)
            )
        ):
            raise chain._UnsupportedChain("root warp body missing original publication")
        self.publication = publication
        self.accepted_prefix = publication.prefix
        self.cursor += 1
        self.pending = None

    def finish(self) -> None:
        self.check()
        if (
            self.pending is not None
            or self.cursor != len(self.program.actions)
            or self.publication is None
            or not self.publication.matches(
                self.codegen, self.plan, self.boundaries, self.prefix
            )
        ):
            raise chain._UnsupportedChain("incomplete root warp body")
        self.consumed = True

    def validate_join(
        self,
        owner: RootWarpBridgeSequence,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        staged: list[chain._StagedInput],
        prefix: list[str],
    ) -> None:
        self.validate_return(prefix)
        if (
            owner is not self.owner
            or cg is not self.codegen
            or plan is not self.plan
            or boundaries is not self.boundaries
            or staged is not self.staged
            or prefix is not self.prefix
        ):
            raise chain._UnsupportedChain("root warp body joined context changed")

    def validate_return(self, prefix: list[str]) -> None:
        """The original root caller receives this exact completed prefix object."""
        self.check(finished=True)
        if (
            prefix is not self.prefix
            or self.pending is not None
            or self.cursor != len(self.program.actions)
            or self.publication is None
            or _staged_facts(self.staged)
            != (
                self.owner._staged
                if isinstance(self.owner, RootWarpStageAction)
                else self.owner._state.staged
            )
            or not self.publication.matches(
                self.codegen, self.plan, self.boundaries, prefix
            )
        ):
            raise chain._UnsupportedChain("root warp body return changed")


def bind_root_warp_body(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    owner: RootWarpStageAction | RootWarpBridgeSequence,
    boundaries: dict[Node, str],
    staged: list[chain._StagedInput],
    scratch: ScratchLayouts,
    prefix: list[str],
) -> RootWarpBody:
    if isinstance(owner, RootWarpStageAction):
        if not owner.matches(cg, plan, boundaries) or staged:
            raise chain._UnsupportedChain("root warp action changed before body")
        actions = (owner,)
    else:
        owner.check(cg, plan, boundaries, staged, 0)
        actions = tuple(RootWarpBridgeAction(owner, stage) for stage in (0, 1))
    binding = owner._body
    if binding.check(owner) is not None or binding.accepted:
        raise chain._UnsupportedChain("root warp body already selected")
    body = RootWarpBody(
        cg,
        plan,
        owner,
        BodyProgram(actions),
        boundaries,
        staged,
        scratch,
        prefix,
        ChainedExecution(plan.threads),
        None,
        tuple(prefix),
    )
    body.context = body.facts()
    binding.body = body
    binding.accepted = True
    return body


def _completed_span(
    plan: ChainedMatmulPlan, stage: int, policy: str, lines: list[str]
) -> tuple[tuple[int, str, tuple[str, ...]], tuple[tuple[str, str], ...]]:
    return (stage, policy, tuple(lines)), tuple(plan.tensor_aliases.items())


def _matches_span(
    plan: ChainedMatmulPlan,
    completion: tuple[int, str, tuple[str, ...]] | None,
    aliases: tuple[tuple[str, str], ...] | None,
    stage: int,
    policy: str,
    lines: list[str],
) -> bool:
    return (completion, aliases) == _completed_span(plan, stage, policy, lines)


def supports_materialized_root(cg: GenerateAST, plan: ChainedMatmulPlan) -> bool:
    from .chained_collectives import uses_general_collectives
    from .chained_plain_root import _ordinary_root_options
    from .chained_tcgen05 import _prefetch_final_b
    from .chained_tcgen05 import supported_plan

    return (
        plan.strategy == "tcgen05_tmem"
        and plan.loop is None
        and len(plan.dots) == 2
        and all(s[0] == 128 for s in plan.shapes)
        and all(dot.args[2] is None for dot in plan.dots)
        and plan.initialized_accumulator is None
        and plan.late_rhs_reuse is None
        and not plan.direct_output
        and plan.k_schedule is None
        and plan.pointwise_cache is None
        and not plan.warp_mma_stages
        and plan.preparation_pipeline is None
        and not uses_general_collectives(plan)
        and not plan.scans
        and not plan.scan_exports
        and not _prefetch_final_b(plan)
        and _ordinary_root_options(cg)
        and cg.device_function.config.config.get(
            "cute_chained_snapshot_tile_columns", 0
        )
        == 0
        and supported_plan(plan)
    )


@dataclass
class RootBody:
    """One accepted body attempt and its private, completed publication ledger."""

    codegen: GenerateAST
    plan: ChainedMatmulPlan
    program: BodyProgram
    axes: tuple[tuple[int, int], ...]
    scratch: ScratchLayouts
    unroll: PointwiseUnroll | None
    cache: PointwiseReadCache | None
    inplace: PointwiseInplace | None
    direct: DirectStageOperands | None
    early_release: bool
    execution: ChainedExecution
    revision: object
    options: object
    action_facts: object
    context: tuple[object, ...]
    boundaries: dict[Node, str] = field(default_factory=dict)
    staged: list[chain._StagedInput] = field(default_factory=list)
    next_stage: int = 0
    consumed: bool = False
    aliases: tuple[tuple[str, str], ...] = ()
    completion: tuple[int, str, tuple[str, ...]] | None = None
    completed_aliases: tuple[tuple[str, str], ...] | None = None
    operand_count: int = 0

    def context_facts(self) -> tuple[object, ...]:
        direct = self.direct
        return (
            self.codegen,
            self.plan,
            self.plan.store,
            self.plan.strategy,
            self.plan.dtype,
            self.plan.loop,
            self.plan.initialized_accumulator,
            self.plan.late_rhs_reuse,
            self.plan.direct_output,
            self.plan.k_schedule,
            self.plan.pointwise_cache,
            self.plan.warp_mma_stages,
            self.plan.preparation_pipeline,
            self.plan.scan_exports,
            self.axes,
            self.early_release,
            _execution_fields(self.execution),
            None
            if direct is None
            else (
                direct.stage,
                direct.nodes,
                direct.shape,
                direct.dtype,
                direct.a,
                direct.b,
            ),
            self.scratch.mode,
            self.scratch.read_buffers,
            None if self.unroll is None else self.unroll.factor,
            None if self.cache is None else self.cache.enabled,
            None if self.inplace is None else self.inplace.enabled,
        )

    def check(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        stage: int,
    ) -> None:
        expected = {
            plan.dots[i]: f"chain_{i}_c" for i in range(min(stage, len(plan.dots) - 1))
        }
        if (
            cg is not self.codegen
            or plan is not self.plan
            or self.consumed
            or stage != self.next_stage
            or self.revision != _pair_revision(plan)
            or self.options != _startup_value(cg.device_function.config.config)
            or self.action_facts != self.program.facts()
            or self.context != self.context_facts()
            or tuple(plan.tensor_aliases.items())[: len(self.aliases)] != self.aliases
            or boundaries != expected
        ):
            raise chain._UnsupportedChain("ordered root body changed before completion")

    def stage(self, ordinal: int) -> RootBodyStage:
        return RootBodyStage(
            self,
            ordinal,
            "terminal" if ordinal == len(self.plan.dots) - 1 else "allocated",
        )

    def accept_stage(self, policy: RootBodyStage, lines: list[str]) -> None:
        self.check(self.codegen, self.plan, self.boundaries, policy.ordinal)
        if not _matches_span(
            self.plan,
            self.completion,
            self.completed_aliases,
            policy.ordinal,
            policy.result,
            lines,
        ):
            raise chain._UnsupportedChain("missing or changed completed body span")
        self.completion = None
        self.completed_aliases = None
        self.operand_count = 0
        if policy.result == "allocated":
            self.boundaries[self.plan.dots[policy.ordinal]] = (
                f"chain_{policy.ordinal}_c"
            )
        self.aliases = tuple(self.plan.tensor_aliases.items())
        self.next_stage += 1


@dataclass(frozen=True)
class RootBodyStage:
    body: RootBody
    ordinal: int
    result: Literal["allocated", "terminal"]

    def record_completion(self, lines: list[str]) -> None:
        if self.body.completion is not None:
            raise chain._UnsupportedChain("ordered root stage completed twice")
        if self.body.operand_count != (0 if self.body.direct is not None else 2):
            raise chain._UnsupportedChain("ordered root operands incomplete")
        self.body.completion, self.body.completed_aliases = _completed_span(
            self.body.plan, self.ordinal, self.result, lines
        )

    @property
    def major_modes(self) -> tuple[str, str]:
        a, b = self.body.axes[self.ordinal]
        return ("K" if a else "MN", "K" if b else "MN")

    def validate(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        stage: int,
        geometry: StageGeometry,
        execution: ChainedExecution,
    ) -> None:
        self.body.check(cg, plan, boundaries, stage)
        if (
            stage != self.ordinal
            or geometry.logical != plan.shapes[stage]
            or geometry.physical != geometry.logical
            or geometry.transpose
            or _execution_fields(execution) != _execution_fields(self.body.execution)
            or self.result
            != ("terminal" if stage == len(plan.dots) - 1 else "allocated")
        ):
            raise chain._UnsupportedChain(
                "ordered root stage geometry or policy changed"
            )

    def operand_lines(
        self, cg: GenerateAST, boundaries: dict[Node, str], role: str, dtype: str
    ) -> list[str]:
        from .chained_tcgen05 import _layout
        from .chained_tcgen05 import _stage

        body, stage = self.body, self.ordinal
        body.check(cg, body.plan, boundaries, stage)
        if (
            body.operand_count not in (0, 1)
            or role != ("a", "b")[body.operand_count]
            or dtype
            != CompileEnvironment.current().backend.dtype_str(
                body.plan.operand_dtype(stage)
            )
        ):
            raise chain._UnsupportedChain("ordered root operand role or dtype changed")
        assert (
            body.unroll is not None
            and body.cache is not None
            and body.inplace is not None
        )
        m, n, k = body.plan.shapes[stage]
        shape = (m if role == "a" else n, k)
        inner = body.axes[stage][0 if role == "a" else 1]
        prefix = f"chain_{stage}"
        # Preserve the original producer call before layout construction and
        # terminal staged-input discovery (including its naming side effects).
        producer = _stage(
            cg,
            body.plan,
            boundaries,
            [],
            stage,
            role,
            inner,
            dtype,
            body.unroll,
            body.cache,
            body.inplace,
        )
        lines = [
            f"{prefix}_{role}_ptr = chain_{role}_workspace",
            *_layout(f"{prefix}_{role}", shape, inner, dtype),
            *producer,
        ]
        if stage == len(body.plan.dots) - 1:
            cached = chain._stage_input(
                cg,
                body.plan,
                cast("Node", body.plan.dots[stage].args[0 if role == "a" else 1]),
                f"{prefix}_{role}",
                role,
                shape,
            )
            if cached is not None:
                body.staged.append(cached)
        body.operand_count += 1
        return lines


def plan_root_body(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    scratch: ScratchLayouts,
    *,
    inner_axes: dict[tuple[int, str], int],
    bridges: dict[int, chain._RegisterBridge],
    early_release: bool,
    direct: DirectStageOperands | None = None,
    unroll: PointwiseUnroll | None = None,
    cache: PointwiseReadCache | None = None,
    inplace: PointwiseInplace | None = None,
) -> RootBody | None:
    from .chained_plain_root import supports_plain_root
    from .chained_tcgen_stage import stage_geometry
    from .physical_use_frontier import PhysicalPublication
    from .physical_use_frontier import PhysicalReadPoint
    from .physical_use_frontier import PhysicalUseFrontier

    if bridges or not (
        supports_plain_root(cg, plan)
        if direct is not None
        else supports_materialized_root(cg, plan)
    ):
        return None
    axes = tuple(
        (inner_axes[i, "a"], inner_axes[i, "b"]) for i in range(len(plan.dots))
    )
    if any(type(x) is not int or x not in (0, 1) for pair in axes for x in pair):
        return None
    if type(early_release) is not bool:
        return None
    geometries = tuple(stage_geometry(shape) for shape in plan.shapes)
    if plan.region is None or any(geometry is None for geometry in geometries):
        return None
    schedule = _root_group_schedule(
        plan, tuple(cast("StageGeometry", geometry) for geometry in geometries)
    )
    actions = []
    use_frontier = PhysicalUseFrontier(plan.region.nodes)
    publications: dict[Node, PhysicalPublication] = {}
    names = {node: f"chain_{i}_c" for i, node in enumerate(plan.dots)}
    for scheduled in schedule.groups:
        (i,) = scheduled.group.stages
        node = plan.dots[i]
        sources = use_frontier.resolve(
            node.all_input_nodes,
            publications,
            external=frozenset(),
            required=frozenset(plan.dots),
            traversable=frozenset(plan.region.nodes),
        )
        point = PhysicalReadPoint(plan, scheduled.read_event, scheduled.read_event + 1)
        for publication in sources:
            use_frontier.read(publication.owner, point)
        read_nodes = {value.node for value in sources}
        reads = tuple(names[node] for node in plan.dots if node in read_nodes)
        actions.extend(
            (
                PreparationAction(
                    "fill",
                    scheduled.read_event,
                    (node,),
                    (i,),
                    i,
                    reads,
                    ("chain_a_workspace", "chain_b_workspace"),
                ),
                PreparationAction(
                    "mma",
                    scheduled.publication_event,
                    (node,),
                    (i,),
                    i,
                    ("chain_a_workspace", "chain_b_workspace"),
                    (f"chain_{i}_c",) if i < len(plan.dots) - 1 else (),
                ),
            )
        )
        if i < len(plan.dots) - 1:
            # The existing allocated C policy owns this complete FP32 result.
            # Actual publication still requires RootBody.accept_stage's original
            # complete native span; planning cannot install a boundary.
            use_frontier.register_owner(node, plan, scheduled.publication_event)
            publications[node] = PhysicalPublication(
                node, node, torch.float32, plan.shapes[i][:2]
            )
    use_frontier.check()
    program = BodyProgram(tuple(actions))
    execution = ChainedExecution(128)
    result = RootBody(
        cg,
        plan,
        program,
        axes,
        scratch,
        unroll,
        cache,
        inplace,
        direct,
        early_release,
        execution,
        _pair_revision(plan),
        _startup_value(cg.device_function.config.config),
        program.facts(),
        (),
    )
    result.context = result.context_facts()
    result.aliases = tuple(plan.tensor_aliases.items())
    return result


@dataclass
class RootActionBody:
    """Traversal of an existing accepted sequence, not another readiness plan."""

    codegen: GenerateAST
    plan: ChainedMatmulPlan
    sequence: RootStageSequence
    boundaries: dict[Node, str]
    _boundary_owner: dict[Node, str]
    program: BodyProgram
    actions: tuple[RootStageAction, ...]
    execution: ChainedExecution
    initial_boundaries: tuple[tuple[Node, str], ...]
    revision: object
    options: object
    context: object
    aliases: tuple[tuple[str, str], ...]
    deferred: RootDeferredRhsEnqueue | None = None
    continuation: PreparedContinuation | None = None
    event: int = 0
    cursor: int = 0
    consumed: bool = False
    pending: RootStageAction | None = None
    completion: tuple[int, str, tuple[str, ...]] | None = None
    completed_aliases: tuple[tuple[str, str], ...] | None = None
    completed_staged: object = None
    progress: object = None
    completed_progress: object = None
    completed_action: RootStageCompletion | None = None
    last_action: RootStageCompletion | None = None
    _deferred_span: RootDeferredSpan | None = None
    span_facts: object = None
    accepted_lines: tuple[str, ...] = ()
    finished_lines: tuple[str, ...] = ()
    finished_state: object = None

    @property
    def deferred_span(self) -> RootDeferredSpan | None:
        return self._deferred_span if self.consumed else None

    def context_facts(self) -> object:
        s = self.sequence
        return _startup_value(
            (
                s.plan,
                s.facts,
                tuple((g.logical, g.transpose, g.native_rows) for g in s.geometries),
                s.inner_axes,
                tuple(vars(x) for x in s.scans),
                s.scan_lines,
                s.early_release,
                s.independent,
                s.pair_inputs,
                s.pair_selection,
                s.pair_staging,
                s.input_readiness,
                s.startup,
                s.startup_issued,
                s.snapshot,
                s.early_scan,
                tuple(vars(x) for x in s.early_cached),
                None
                if s.initialized is None
                else (id(s.initialized), vars(s.initialized)),
                None
                if s.initialized is None or s.initialized.paired is None
                else (
                    id(s.initialized.paired),
                    vars(s.initialized.paired),
                    id(s.initialized.paired.transfer),
                    id(s.initialized.paired.transfer.wrapper),
                ),
                None
                if s.seeded_inputs is None
                else (id(s.seeded_inputs), vars(s.seeded_inputs)),
                s.pointwise_unroll.factor,
                s.pointwise_cache.enabled,
                s.pointwise_inplace.enabled,
                _execution_fields(self.execution),
                None
                if self.continuation is None
                else (id(self.continuation), self.continuation._current()),
            )
        )

    def progress_facts(self) -> object:
        return root_stage_progress(self.sequence)

    def check(self, *, completed: bool = False) -> None:
        s = self.sequence
        if self.continuation is not None:
            self.continuation.check()
        expected = dict(self.initial_boundaries)
        expected.update(
            (self.plan.dots[i], f"chain_{i}_c")
            for i in range(self.cursor)
            if self.actions[i].result == "logical_fragment"
        )
        # Existing selection matchers consume the original pre-stage map. The
        # actual callback still receives this body's exact live map object.
        selection_boundaries = dict(self.initial_boundaries)
        ordinal = min(self.cursor, len(self.actions) - 1)
        selection_boundaries.update(
            (self.plan.dots[i], f"chain_{i}_c")
            for i in range(ordinal)
            if self.actions[i].result == "logical_fragment"
        )
        valid_selection = (
            (
                s.independent is None
                or s.independent.matches(
                    self.codegen, self.plan, selection_boundaries, s.inner_axes
                )
            )
            and (
                s.pair_selection is None
                or s.pair_selection.matches(
                    self.codegen, self.plan, selection_boundaries, s.inner_axes, ordinal
                )
            )
            and (
                s.startup is None
                or s.startup.matches(self.codegen, self.plan, s.inner_axes)
            )
            and (
                s.snapshot is None
                or s.snapshot.matches(
                    self.codegen, self.plan, selection_boundaries, s.scans
                )
            )
            and (
                s.initialized is None
                or s.initialized.matches(
                    self.codegen, self.plan, selection_boundaries, s.inner_axes, s.scans
                )
            )
            and (
                s.seeded_inputs is None
                or s.seeded_inputs.result is s.initialized
                and s.seeded_inputs.matches(self.plan)
            )
        )
        program = (
            (self.actions[0], self.deferred, *self.actions[1:])
            if self.deferred is not None
            else self.actions
        )
        self.actions[ordinal]._check_grouped_state()
        current_aliases = tuple(self.plan.tensor_aliases.items())
        if (
            self.consumed
            or not 0 <= self.cursor <= len(self.actions)
            or s.next_stage != self.cursor + int(completed)
            or self.revision != _pair_revision(self.plan)
            or self.options
            != _startup_value(self.codegen.device_function.config.config)
            or self.context != self.context_facts()
            or (self.pending is None and self.progress != self.progress_facts())
            or not valid_selection
            or self.boundaries is not self._boundary_owner
            or self.boundaries != expected
            or self.event != self.cursor + int(self._deferred_span is not None)
            or len(self.program.actions) != len(program)
            or any(
                a is not b for a, b in zip(self.program.actions, program, strict=True)
            )
            or any(
                a.sequence is not s or a.stage != i for i, a in enumerate(self.actions)
            )
            or self.deferred is not None
            and self.deferred.inputs is not s.seeded_inputs
            or self._deferred_span is not None
            and (
                self._deferred_span.facts() != self.span_facts
                or self._deferred_span.completion is not s._rhs_completion
                or self._deferred_span.completion.queued is not s.queued_rhs
                or self._deferred_span.completion.queued.inputs is not s.seeded_inputs
                or self._deferred_span.completion.queued.seed is not s.seed_completion
            )
            or (
                current_aliases[: len(self.aliases)] != self.aliases
                if self.pending is not None
                else current_aliases != self.aliases
            )
        ):
            raise chain._UnsupportedChain(
                "ordered root action provenance or readiness changed"
            )

    def validate_stage(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        action: RootStageAction | None,
        stage: int,
        geometry: StageGeometry,
        execution: ChainedExecution,
        *,
        completed: bool = False,
    ) -> None:
        self.check(completed=completed)
        if (
            cg is not self.codegen
            or plan is not self.plan
            or boundaries is not self.boundaries
            or stage != self.cursor
            or action is not self.actions[self.cursor]
            or action is not self.program.actions[self.event]
            or geometry != self.sequence.geometries[stage]
            or _execution_fields(execution) != _execution_fields(self.execution)
            or (self.pending is not action if completed else self.pending is not None)
        ):
            raise chain._UnsupportedChain(
                "ordered root action call provenance or readiness changed"
            )

    def begin_stage(self, action: RootStageAction) -> None:
        """Called only after the original stage policy validation succeeds."""
        if self.pending is not None or action is not self.actions[self.cursor]:
            raise chain._UnsupportedChain("ordered root action call changed")
        self.pending = action

    def record_completion(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        action: RootStageAction,
        stage: int,
        geometry: StageGeometry,
        execution: ChainedExecution,
        lines: list[str],
    ) -> None:
        self.validate_stage(
            cg, plan, boundaries, action, stage, geometry, execution, completed=True
        )
        if self.completion is not None:
            raise chain._UnsupportedChain("ordered root action completed twice")
        transition = action._completion
        if transition is None or not transition.matches(action):
            raise chain._UnsupportedChain("missing or changed root action transition")
        self.completion, self.completed_aliases = _completed_span(
            plan, stage, "action", lines
        )
        self.completed_staged = _startup_value(
            tuple(vars(x) for x in self.sequence.staged)
        )
        self.completed_progress = self.progress_facts()
        self.completed_action = transition

    def accept(self, action: RootStageAction, lines: list[str]) -> None:
        self.check(completed=True)
        if (
            self.pending is not action
            or action is not self.actions[self.cursor]
            or self.completed_action is None
            or not self.completed_action.matches(action)
            or not _matches_span(
                self.plan,
                self.completion,
                self.completed_aliases,
                self.cursor,
                "action",
                lines,
            )
            or self.completed_staged
            != _startup_value(tuple(vars(x) for x in self.sequence.staged))
            or self.completed_progress != self.progress_facts()
        ):
            raise chain._UnsupportedChain("missing or changed root action completion")
        if action.result == "logical_fragment":
            self.boundaries[self.plan.dots[self.cursor]] = f"chain_{self.cursor}_c"
        self.aliases = tuple(self.plan.tensor_aliases.items())
        self.cursor += 1
        self.event += 1
        self.pending = None
        self.completion = None
        self.completed_aliases = None
        self.completed_staged = None
        self.completed_progress = None
        self.last_action = self.completed_action
        self.completed_action = None
        self.progress = self.progress_facts()
        self.accepted_lines += tuple(lines)

    def enqueue(self, action: RootDeferredRhsEnqueue, lines: list[str]) -> None:
        self.check()
        if (
            action is not self.deferred
            or action is not self.program.actions[self.event]
            or self.cursor != 1
            or self.pending is not None
            or self._deferred_span is not None
            or self.last_action is None
            or not self.last_action.matches(self.actions[0])
            or tuple(lines) != self.accepted_lines
        ):
            raise chain._UnsupportedChain("ordered RHS enqueue changed")
        start = len(lines)
        lines.extend(action.inputs.lines)
        self.sequence.enqueue_rhs(list(action.inputs.lines))
        completion = self.sequence._rhs_completion
        if (
            completion is None
            or not completion.matches(self.sequence)
            or completion.before != self.progress
        ):
            raise chain._UnsupportedChain("missing or changed RHS enqueue transition")
        self._deferred_span = RootDeferredSpan(
            action, completion, start, action.inputs.lines
        )
        self.span_facts = self._deferred_span.facts()
        self.event += 1
        self.progress = self.progress_facts()
        self.accepted_lines += action.inputs.lines

    def deferred_position(
        self, lines: list[str], body_start: int, deferred_rhs: list[str]
    ) -> int:
        """Validate the original inserted body span at the naming-only prepass."""
        span = self.deferred_span
        if (
            not self.consumed
            or self.deferred is None
            or span is None
            or span.action is not self.deferred
            or span.facts() != self.span_facts
            or span.completion is not self.sequence._rhs_completion
            or span.completion.queued is not self.sequence.queued_rhs
            or span.completion.queued.inputs is not self.sequence.seeded_inputs
            or span.completion.queued.seed is not self.sequence.seed_completion
            or not span.action.inputs.matches(self.plan)
            or self.sequence.initialized is not None
            and self.sequence.initialized.paired is not None
            and not self.sequence.initialized.paired.matches_finalized()
            or tuple(self.plan.tensor_aliases.items())[: len(self.aliases)]
            != self.aliases
            or self.finished_state != root_stage_state(self.sequence)
            or self.context != self.context_facts()
            or self.revision != _pair_revision(self.plan)
            or self.options
            != _startup_value(self.codegen.device_function.config.config)
            or tuple(deferred_rhs) != span.lines
            or tuple(lines[body_start : body_start + len(self.finished_lines)])
            != self.finished_lines
            or tuple(lines[body_start + span.start : body_start + span.stop])
            != span.lines
        ):
            raise chain._UnsupportedChain("completed RHS insertion span changed")
        return body_start + span.start


def _root_group_schedule(
    plan: ChainedMatmulPlan,
    geometries: tuple[StageGeometry, ...],
    graph: ContractionGraph | None = None,
) -> GroupedContractionSchedule:
    from .chained_contraction_groups import ContractionGroup
    from .prepared_graph_schedule import ContractionGraph
    from .prepared_graph_schedule import schedule_contraction_groups

    if (
        plan.region is None
        or tuple(spec.node for spec in plan.region.contractions) != plan.dots
    ):
        raise chain._UnsupportedChain("root grouped schedule lost original region")
    schedule = schedule_contraction_groups(
        graph
        if graph is not None
        else ContractionGraph(
            plan.region,
            ()
            if plan.initialized_accumulator is None
            else (plan.initialized_accumulator,),
        ),
        tuple(
            ContractionGroup((i,), (geometry,)) for i, geometry in enumerate(geometries)
        ),
        plan.shapes,
        partitions=(frozenset(plan.dots),),
        materializations=frozenset((*plan.region.scans, *plan.region.reductions)),
    )
    # Original root completion consumes consecutive logical stages. Retain that
    # physical authority; the shared pass must decide before actions are built.
    if tuple(item.group.stages[0] for item in schedule.groups) != tuple(
        range(len(plan.dots))
    ):
        raise chain._UnsupportedChain(
            "graph schedule conflicts with original root binding"
        )
    return schedule


def plan_root_action_body(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    sequence: RootStageSequence | None,
    boundaries: dict[Node, str],
    *,
    deferred_rhs: list[str],
    prepared_continuation: bool = False,
) -> RootActionBody | None:
    """Consume complete original actions without new discovery."""
    if sequence is None or not all(sequence.handles(i) for i in range(len(plan.dots))):
        return None
    if type(prepared_continuation) is not bool:
        raise chain._UnsupportedChain("invalid private continuation choice")
    if sequence.plan is not plan or sequence.next_stage != 0:
        raise chain._UnsupportedChain("root action selection already consumed")
    selected = sequence.seeded_inputs
    deferred = (
        RootDeferredRhsEnqueue(selected)
        if selected is not None and selected.mode == "deferred"
        else None
    )
    if tuple(deferred_rhs) != (() if deferred is None else deferred.inputs.lines):
        raise chain._UnsupportedChain("original deferred RHS program changed")
    continuation = None
    if prepared_continuation:
        from .prepared_continuation import continuation_for_nodes

        if (
            sequence.initialized is None
            or plan.initialized_accumulator is None
            or plan.region is None
            or len(plan.dots) != 2
        ):
            return None
        continuation = continuation_for_nodes(
            plan.region,
            plan.dots[0],
            plan.dots[1],
            transformed=plan.initialized_accumulator,
        )
    schedule = _root_group_schedule(
        plan,
        sequence.geometries,
        continuation.graph if continuation is not None else None,
    )
    actions = tuple(sequence.action(item.group.stages[0]) for item in schedule.groups)
    program = BodyProgram(
        (actions[0], deferred, *actions[1:]) if deferred is not None else actions
    )
    result = RootActionBody(
        cg,
        plan,
        sequence,
        boundaries,
        boundaries,
        program,
        actions,
        ChainedExecution(128),
        tuple(boundaries.items()),
        _pair_revision(plan),
        _startup_value(cg.device_function.config.config),
        (),
        tuple(plan.tensor_aliases.items()),
        deferred=deferred,
        continuation=continuation,
    )
    result.context = result.context_facts()
    result.progress = result.progress_facts()
    result.check()
    return result


class _PreparationBody:
    def __init__(
        self, cg: GenerateAST, plan: ChainedMatmulPlan, pipeline: PreparationPipeline
    ) -> None:
        from ..compile_environment import CompileEnvironment
        from .chained_contraction_groups import ContractionGroup
        from .chained_frontier_groups import FrontierStores
        from .chained_frontier_ownership import FrontierOwnership
        from .chained_matmul import _shape
        from .chained_register_islands import plan_register_islands
        from .chained_tcgen_stage import stage_geometry

        self.frame = pipeline.frame
        self.boundaries: dict[Node, str] = {}
        self.native_requested = (
            cg.device_function.config.config.get("cute_chained_native_vector_reads")
            is True
        )
        self.native_activated = False
        self.published_native: dict[Node, str] = {}
        tile_columns = cast(
            "int",
            cg.device_function.config.config.get(
                "cute_chained_frontier_tile_columns", 0
            ),
        )
        self.frontier_ownership = (
            FrontierOwnership(tile_columns) if tile_columns else None
        )
        self.frontier_stores = (
            FrontierStores(True)
            if cg.device_function.config.config.get("cute_chained_frontier_stmatrix")
            is True
            else None
        )
        self.retention = pipeline.operand_retention
        self.retained_nodes = set()
        if self.retention is not None:
            original_plan = self.retention.revision.plan
            if (
                self.retention.frame is not self.frame
                or replace(plan, loop_workspace=original_plan.loop_workspace)
                != original_plan
                or not self.retention.matches(
                    original_plan,
                    self.retention.revision.frame,
                    {
                        node: _shape(node)
                        for node in self.frame.cut.region.nodes
                        if isinstance(node.meta.get("val"), torch.Tensor)
                    },
                    prepared_groups=self.retention.original_groups,
                    reservations=self.retention.reservations,
                )
            ):
                raise exc.BackendUnsupported(
                    "cute", "operand retention revision changed"
                )
            self.retained_nodes = {item.node for item in self.retention.candidates}

        self.stages = {stage.group.stages[0]: stage for stage in self.frame.stages}
        self.buffers = {buffer.name: buffer for buffer in self.frame.buffers}
        self.transposed_images = {
            member.buffer.name
            for binding in pipeline.prepared_groups
            for member in binding.candidate.members
            if member.logical_modes != (0, 1)
        }
        self.register_requested = (
            cg.device_function.config.config.get("cute_chained_register_islands")
            is True
            and CompileEnvironment.current().settings.fast_math is True
        )
        self.islands = {}
        if self.register_requested:
            assert plan.region is not None
            groups = plan.contraction_groups
            if groups is None:
                geometries = tuple(stage_geometry(shape) for shape in plan.shapes)
                assert all(geometry is not None for geometry in geometries)
                groups = tuple(
                    ContractionGroup((index,), (geometry,))
                    for index, geometry in enumerate(geometries)
                    if geometry is not None
                )
            self.islands = {
                island.groups[0].stages[0]: island
                for island in plan_register_islands(
                    plan.region,
                    groups,
                    {
                        node: _shape(node)
                        for node in plan.region.nodes
                        if isinstance(node.meta.get("val"), torch.Tensor)
                    },
                    fast_math=True,
                    # Multi-image value transport does not yet replace an
                    # original direct-publication or retained-input event.
                    # Keep those planning/emission paths single-image before
                    # any selected island emits or changes a boundary.
                    multi_image=(
                        cg.device_function.config.config.get(
                            "cute_chained_island_consumers"
                        )
                        is not True
                        and not (
                            self.retention is not None and self.retention.candidates
                        )
                    ),
                    entry_boundaries=(
                        frozenset(entry.node for entry in plan.pointwise_cache.entries)
                        if plan.pointwise_cache is not None
                        else frozenset()
                    ),
                )
            }
        self.skip_until = 0
        self.used_island = False
        self.row_requested = (
            cg.device_function.config.config.get("cute_chained_collective_retention")
            is True
        )
        self.used_rows = False

    def publish_retained(
        self, first: int, stop: int, recorder: _PreparationRecorder | None = None
    ) -> list[str]:
        if self.retention is None:
            return []
        before = tuple(self.boundaries.items()) if recorder is not None else ()
        aliases = []
        for index, item in enumerate(self.retention.candidates):
            if first < item.publication_event <= stop:
                name = f"chain_retained_operand_{index}"
                # The frame owns prep_* byte regions; emit_stage constructs
                # separate chain_* full native tensors over those regions.
                native = f"chain_{item.group.stages[0]}_{item.role}"
                aliases.extend(item.alias_lines(native, name))
                self.boundaries[item.node] = name
                # This receipt is installed only after the original stage
                # completed its fill/MMA and published this exact native alias.
                self.published_native[item.node] = name
                self.retained_nodes.remove(item.node)
        if recorder is not None:
            recorder.complete_retained(
                first, stop, before, self.boundaries, tuple(aliases)
            )
        return aliases


def emit_body_program(
    cg: GenerateAST,
    plan: ChainedMatmulPlan | CuteChunkRecurrencePlan,
    pipeline: PreparationPipeline | None,
    execution: ChainedExecution,
    vector: VectorStaging | None,
    unroll: BoundedProducerUnroll | None,
    scratch: ScratchLayouts | None,
    *,
    recorder: _PreparationRecorder | None = None,
    root: RootBody | None = None,
    root_actions: RootActionBody | None = None,
    root_warp: RootWarpBody | None = None,
    recurrence_body: RecurrenceBody | None = None,
    drain_tiling: DrainTiling | None = None,
    prepared_body: PreparedBodyLowering | None = None,
    state_body: RootStateBody | None = None,
    epoch_body: EpochSourceBody | None = None,
) -> list[str]:
    from .chained_collectives import emit_collectives_before
    from .chained_frontier_groups import emit_frontier_group
    from .chained_frontier_groups import plan_frontier_group
    from .chained_native_read_inputs import bind_retained_native_inputs
    from .chained_pointwise_residency import emit_pointwise_cache_before
    from .chained_preparation_pipeline import _emit_row_collective_stage
    from .chained_preparation_pipeline import _emit_scan_producer_stage
    from .chained_preparation_reads import PreparationFrontierCompletion
    from .chained_prepared_values import emit_prepared_value
    from .chained_register_binding import bind_preparation_island
    from .chained_register_emission import emit_register_island
    from .chained_tcgen_stage import emit_stage
    from .chained_tcgen_stage import stage_geometry
    from .chained_vector_native import NativeReadInputs

    state = None
    if epoch_body is not None:
        if (
            any(
                value is not None
                for value in (
                    pipeline,
                    recorder,
                    root,
                    root_actions,
                    root_warp,
                    recurrence_body,
                    drain_tiling,
                    prepared_body,
                    state_body,
                    vector,
                    unroll,
                    scratch,
                )
            )
            or cg is not epoch_body.codegen
            or plan is not epoch_body.plan
            or execution.threads != epoch_body.plan.threads
        ):
            raise chain._UnsupportedChain("epoch body has a foreign envelope")
        epoch_body.check()

        def walk(program: BodyProgram, scope: tuple[object, ...]) -> list[str]:
            lines = []
            for action in program.actions:
                if isinstance(
                    action,
                    (
                        EpochSetup,
                        EpochMemoryAction,
                        StateEffect,
                        StateArrival,
                        DescriptorIssue,
                        EpochEvent,
                        EpochSynchronizationAction,
                    ),
                ):
                    lines.extend(epoch_body.step(action, scope))
                elif isinstance(action, EpochBranch):
                    lines.extend(
                        (
                            f"if {action.condition}:",
                            chain._indent(
                                walk(action.body, (*scope, (action.predicate, True)))
                            ),
                        )
                    )
                    if action.otherwise.actions:
                        lines.extend(
                            (
                                "else:",
                                chain._indent(
                                    walk(
                                        action.otherwise,
                                        (*scope, (action.predicate, False)),
                                    )
                                ),
                            )
                        )
                elif isinstance(action, EpochRange):
                    lines.extend(
                        (
                            f"for {action.target} in {action.iterator}:",
                            chain._indent(walk(action.body, (*scope, action.domain))),
                        )
                    )
                else:
                    raise chain._UnsupportedChain("unknown epoch source action")
            return lines

        epoch_lines = walk(epoch_body.program, ())
        epoch_body.finish(epoch_lines)
        return epoch_lines
    if state_body is not None:
        if any(
            value is not None
            for value in (
                pipeline,
                recorder,
                root,
                root_actions,
                root_warp,
                recurrence_body,
                drain_tiling,
                prepared_body,
            )
        ):
            raise chain._UnsupportedChain("state body cannot own another envelope")
        state_body.check()
        if (
            cg is not state_body.owner.codegen
            or plan is not state_body.owner.plan
            or execution is not state_body.execution
            or vector is not None
            or unroll is not None
            or scratch is not None
        ):
            raise chain._UnsupportedChain("state body execution changed")
        state_lines = []
        for effect in state_body.program.actions:
            if not isinstance(effect, StateEffect):
                raise chain._UnsupportedChain("unknown state body action")
            state_lines.extend(state_body.emit(effect))
        state_body.finish(state_lines)
        return state_lines
    if prepared_body is not None:
        if (
            pipeline is not None
            or recorder is not None
            or root is not None
            or root_actions is not None
            or root_warp is not None
            or recurrence_body is not None
            or drain_tiling is not None
        ):
            raise chain._UnsupportedChain("prepared body cannot own another envelope")
        prepared_body.check()
        if cg is not prepared_body.codegen or plan is not prepared_body.owner.plan:
            raise chain._UnsupportedChain("prepared body execution changed")
        if prepared_body.root_action is not None:
            assert isinstance(prepared_body.owner, RootActionBody)
            if _execution_fields(execution) != _execution_fields(
                prepared_body.owner.execution
            ):
                raise chain._UnsupportedChain("prepared root execution changed")
        elif execution.threads != plan.threads:
            raise chain._UnsupportedChain("prepared external execution changed")
        program = prepared_body.program
    elif recurrence_body is not None:
        if (
            pipeline is not None
            or recorder is not None
            or root is not None
            or root_actions is not None
            or root_warp is not None
        ):
            raise chain._UnsupportedChain("recurrence body cannot own another envelope")
        recurrence_body.check()
        if (
            cg is not recurrence_body.codegen
            or plan is not recurrence_body.plan
            or execution is not recurrence_body.execution
        ):
            raise chain._UnsupportedChain("recurrence body execution changed")
        program = recurrence_body.program
    elif root_warp is not None:
        if (
            root is not None
            or root_actions is not None
            or pipeline is not None
            or recorder is not None
        ):
            raise chain._UnsupportedChain("root warp body cannot own another envelope")
        root_warp.check()
        if (
            cg is not root_warp.codegen
            or plan is not root_warp.plan
            or execution is not root_warp.execution
            or scratch is not root_warp.scratch
        ):
            raise chain._UnsupportedChain("root warp body execution changed")
        program = root_warp.program
    elif root_actions is not None:
        if root is not None or pipeline is not None or recorder is not None:
            raise chain._UnsupportedChain(
                "root action body cannot own another envelope"
            )
        root_actions.check()
        if (
            cg is not root_actions.codegen
            or plan is not root_actions.plan
            or _execution_fields(execution) != _execution_fields(root_actions.execution)
        ):
            raise chain._UnsupportedChain("root action execution changed")
        program = root_actions.program
    elif root is not None:
        if pipeline is not None or recorder is not None:
            raise chain._UnsupportedChain("root body cannot own preparation receipts")
        assert isinstance(plan, chain.ChainedMatmulPlan)
        root.check(cg, plan, root.boundaries, 0)
        if _execution_fields(execution) != _execution_fields(root.execution):
            raise chain._UnsupportedChain("ordered body execution changed")
        program = root.program
    else:
        assert pipeline is not None
        assert isinstance(plan, chain.ChainedMatmulPlan)
        state = _PreparationBody(cg, plan, pipeline)
        program = preparation_body_program(pipeline)
    lines = root_warp.prefix if root_warp is not None else []
    for action in program.actions:
        if prepared_body is not None:
            if not isinstance(action, ContinuationAction):
                raise chain._UnsupportedChain("invalid prepared body action")
            prepared_body.lower(action)
            continue
        assert isinstance(plan, chain.ChainedMatmulPlan) and scratch is not None
        if recurrence_body is not None:
            if not isinstance(action, RecurrenceStage):
                raise chain._UnsupportedChain("invalid recurrence body action")
            group = action.group
            selection = recurrence_body.begin(action)
            recurrence_body.prepare_store(action, selection)
            if selection is None:
                selection = recurrence_body.unfinalized_selection(action)
            replacement = emit_stage(
                cg,
                plan,
                recurrence_body.boundaries,
                group.stages[0],
                group.geometries[0],
                "chain_iteration & 1",
                group,
                vector,
                unroll,
                execution=execution,
                seed_tiling=recurrence_body.seed_tiling,
                residency=selection.residency,
                prepared_operand=selection.prepared_operand,
                prepared_group=selection.prepared_group,
                tmem_input=selection.tmem_input,
                tmem_output=selection.tmem_output,
                tmem_accumulator=selection.tmem_accumulator,
                tmem_carry=selection.tmem_carry,
                output_lease=(
                    recurrence_body.output_lease
                    if recurrence_body.output_lease is not None
                    and group == recurrence_body.output_lease.proof.final_group
                    else None
                ),
                completed_store=recurrence_body.completed_action,
                drain_tiling=drain_tiling,
            )
            recurrence_body._completed_boundaries = tuple(
                recurrence_body.boundaries.items()
            )
            completion, aliases = _completed_span(
                plan, group.stages[0], "recurrence", replacement
            )
            recurrence_body.accept(action, replacement, completion, aliases)
            lines.extend(replacement)
            continue
        if root_warp is not None:
            root_warp.check()
            if not isinstance(action, (RootWarpStageAction, RootWarpBridgeAction)):
                raise chain._UnsupportedChain("invalid root warp body action")
            replacement = action.emit(
                cg, plan, root_warp.boundaries, root_warp.staged, scratch, lines
            )
            root_warp.accept(action, replacement)
            lines.extend(replacement)
            continue
        if root_actions is not None:
            if isinstance(action, RootDeferredRhsEnqueue):
                root_actions.enqueue(action, lines)
                continue
            assert isinstance(action, RootStageAction)
            ordinal = action.stage
            sequence = root_actions.sequence
            replacement = emit_stage(
                cg,
                plan,
                root_actions.boundaries,
                ordinal,
                sequence.geometries[ordinal],
                "0",
                root_actions=action,
                pending_allocation=sequence.early_release if ordinal == 0 else None,
                terminal_fragment=ordinal == len(plan.dots) - 1,
                tmem_input=action.tmem_input,
                tmem_accumulator=action.accumulator,
                execution=root_actions.execution,
                root_action_body=root_actions,
            )
            root_actions.accept(action, replacement)
            lines.extend(replacement)
            continue
        assert isinstance(action, PreparationAction)
        if root is not None:
            if action.kind == "fill":
                ordinal = action.stages[0]
                geometry = stage_geometry(plan.shapes[ordinal])
                assert geometry is not None
                policy = root.stage(ordinal)
                replacement = emit_stage(
                    cg,
                    plan,
                    root.boundaries,
                    ordinal,
                    geometry,
                    "0",
                    execution=root.execution,
                    direct_operands=root.direct,
                    pending_allocation=root.early_release if ordinal == 0 else None,
                    terminal_fragment=policy.result == "terminal",
                    tmem_accumulator="chain_tptr + 0",
                    root_body=policy,
                    drain_tiling=drain_tiling,
                )
                root.accept_stage(policy, replacement)
                lines.extend(replacement)
            elif action.kind != "mma" or action.stages[0] != root.next_stage - 1:
                raise chain._UnsupportedChain("invalid ordered root completion")
            continue
        assert pipeline is not None and state is not None
        assert vector is not None and unroll is not None
        if action.event < state.skip_until:
            continue
        segment_start = len(lines)
        postlude: list[str] = []
        frontier_completion = (
            PreparationFrontierCompletion()
            if action.kind == "frontier"
            and recorder is not None
            and recorder.island_consumers
            else None
        )
        before = dict(state.boundaries) if recorder is not None else None
        if (
            recorder is not None
            and recorder.broadcast_retention
            and not (action.kind == "mma" and recorder.cursor == action.event + 1)
        ):
            vector.broadcast = recorder.begin_broadcast(action.event)
        materializations = (
            recorder.frontier_materializations(action, state.boundaries)
            if recorder is not None
            else None
        )
        if recorder is not None and materializations is None:
            image = recorder.emit_image(action, state.boundaries, vector, unroll)
            if image is not None:
                lines.extend(image)
                continue
        if (
            pipeline.scan_producer is not None
            and action.event == pipeline.scan_producer.first_event
        ):
            scan_inputs = NativeReadInputs(()) if state.native_requested else None
            scan_options: _ScanProducerOptions = {}
            if recorder is not None:
                scan_options["recorder"] = recorder
            if scan_inputs is not None:
                scan_options["native_inputs"] = scan_inputs
            if recorder is not None:
                assert before is not None
                recorder.begin_scan(action.event, before, body_first=len(lines))
            lines.extend(
                _emit_scan_producer_stage(
                    cg,
                    plan,
                    pipeline,
                    state.boundaries,
                    execution,
                    unroll,
                    vector,
                    **scan_options,
                )
            )
            state.native_activated |= scan_inputs is not None and scan_inputs.activated
            state.skip_until = pipeline.scan_producer.stop_event
            publication = state.frame.actions[state.skip_until]
            assert publication.kind == "mma"
            postlude = state.publish_retained(
                action.event, publication.publication_event
            )
            lines.extend(postlude)
            if recorder is not None:
                assert before is not None
                recorder.record(
                    action,
                    before,
                    state.boundaries,
                    kind="scan",
                    proof=pipeline.scan_producer,
                    stop=publication.publication_event,
                    emitted=lines[segment_start:],
                    body_first=segment_start,
                    postlude=postlude,
                )
            continue
        if state.row_requested and action.kind == "collective":
            replacement = _emit_row_collective_stage(
                cg, plan, state.frame, action.event, state.boundaries, execution, unroll
            )
            if replacement is not None:
                body, state.skip_until = replacement
                lines.extend(body)
                # The row producer emits the complete fill+MMA stage, while
                # its action span leaves the following MMA bookkeeping no-op.
                if state.retention is not None:
                    publication = state.frame.actions[state.skip_until]
                    assert publication.kind == "mma"
                    lines.extend(
                        state.publish_retained(
                            action.event, publication.publication_event
                        )
                    )
                state.used_rows = True
                if recorder is not None:
                    assert before is not None
                    recorder.record(
                        action,
                        before,
                        state.boundaries,
                        stop=state.skip_until + 1,
                        emitted=lines[segment_start:],
                        body_first=segment_start,
                    )
                continue
        input_bound = (
            recorder.input_island(action.event, state.boundaries)
            if action.kind == "fill"
            and recorder is not None
            and recorder.island_consumers
            else None
        )
        if action.kind == "fill" and (
            input_bound is not None or action.stages[0] in state.islands
        ):
            bound = (
                input_bound
                if input_bound is not None
                else bind_preparation_island(
                    cg,
                    plan,
                    state.frame,
                    state.islands[action.stages[0]],
                    execution,
                    state.boundaries,
                )
            )
            if bound is not None:
                replacement = (
                    recorder.emit_island(
                        bound, state.boundaries, vector, body_first=len(lines)
                    )
                    if recorder is not None and recorder.island_consumers
                    else emit_register_island(cg, plan, bound, state.boundaries)
                )
                if replacement is not None:
                    assert bound.first_event == action.event
                    if state.retention is not None and any(
                        action.event < item.publication_event <= bound.stop_event
                        and not (
                            bound.island_input is not None
                            and bound.island_input.candidate.retained_input is not None
                            and item is bound.island_input.candidate.retained_input[1]
                        )
                        for item in state.retention.candidates
                    ):
                        raise exc.BackendUnsupported(
                            "cute",
                            "register island cannot publish an unwritten native operand",
                        )
                    if bound.island_input is not None:
                        publication = bound.island_input
                        retained = publication.candidate.retained_input
                        assert (
                            retained is not None
                            and publication.island_completion is not None
                        )
                        state.published_native[retained[1].node] = (
                            publication.boundary_name(bound.stop_event)
                        )
                        state.retained_nodes.remove(retained[1].node)
                    lines.extend(replacement)
                    state.skip_until = bound.stop_event
                    state.used_island = True
                    if recorder is not None:
                        assert before is not None
                        recorder.record(
                            action,
                            before,
                            state.boundaries,
                            kind="island",
                            proof=bound,
                            stop=bound.stop_event,
                            emitted=lines[segment_start:],
                            body_first=segment_start,
                        )
                    continue
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
                    state.boundaries,
                    execution,
                    protocol.leaf_pointer(ordinal),
                    protocol.phase,
                )
                if recorder is None
                else recorder.emit_leaf(
                    leaf,
                    action,
                    state.boundaries,
                    execution,
                    protocol.leaf_pointer(ordinal),
                    protocol.phase,
                )
            )
            state.boundaries[leaf.node] = leaf.name
        elif action.kind == "collective":
            assert action.source_stage is not None
            lines.extend(
                emit_collectives_before(
                    cg,
                    plan,
                    state.boundaries,
                    action.source_stage,
                    execution=execution,
                    selected=frozenset(action.nodes),
                )
            )
        elif action.kind == "cache":
            assert plan.pointwise_cache is not None and action.source_stage is not None
            fragment = None if recorder is None else recorder.fragment
            if fragment is not None and action.event == fragment.cache_event:
                lines.extend(fragment.emit_cache(action, state.boundaries, len(lines)))
            else:
                lines.extend(
                    emit_pointwise_cache_before(
                        cg,
                        plan,
                        state.boundaries,
                        plan.pointwise_cache,
                        action.source_stage,
                        execution=execution,
                        selected=frozenset(action.nodes),
                    )
                )
        elif action.kind == "fill":
            stage = state.stages[action.stages[0]]
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
            # The stage emitter performs this adjacent fill+MMA pair, including
            # its publication barrier. A/B stay live through C publication.
            if recorder is not None:
                lines.extend(
                    recorder.emit_stage(
                        stage,
                        state.boundaries,
                        local,
                        vector,
                        unroll,
                        first=action.event,
                        body_first=len(lines),
                    )
                )
            else:
                lines.extend(
                    emit_stage(
                        cg,
                        plan,
                        state.boundaries,
                        stage.group.stages[0],
                        stage.group.geometries[0],
                        "0",
                        stage.group,
                        vector,
                        unroll,
                        execution=local,
                        prepared_shape=stage.shape,
                    )
                )
            postlude = state.publish_retained(action.event, action.event + 2, recorder)
            lines.extend(postlude)
        elif action.kind == "frontier":
            native_inputs = None
            group = (
                plan_frontier_group(state.frame, action.event)
                if state.native_requested
                else None
            )
            if group is not None and state.retention is not None:
                native_inputs = NativeReadInputs(
                    bind_retained_native_inputs(
                        plan,
                        state.frame,
                        state.retention,
                        state.published_native,
                        group.first_event,
                        group.stop_event,
                        tuple(buffer.name for buffer in group.buffers),
                        frontier_group=group,
                    )
                )
            grouped = (
                emit_frontier_group(
                    cg,
                    plan,
                    state.frame,
                    action.event,
                    state.boundaries,
                    execution,
                    vector,
                    unroll,
                    scalar_targets=scratch.xor_buffers | state.transposed_images,
                    materialized=state.retention is not None,
                    ownership=state.frontier_ownership,
                    prepared_operands=pipeline.prepared_operands,
                    prepared_groups=pipeline.prepared_groups,
                    native_inputs=native_inputs,
                    stores=state.frontier_stores,
                    completion=frontier_completion,
                    **(
                        {"materializations": materializations}
                        if materializations is not None
                        else {}
                    ),
                )
                if recorder is None
                or materializations is not None
                or recorder.permits_frontier_group(action.event)
                else None
            )
            if grouped is not None:
                replacement, state.skip_until = grouped
                lines.extend(replacement)
                state.native_activated |= (
                    native_inputs is not None and native_inputs.activated
                )
                if recorder is not None:
                    assert before is not None
                    recorder.record(
                        action,
                        before,
                        state.boundaries,
                        kind="frontier",
                        proof=(
                            materializations.receipt
                            if materializations is not None
                            else plan_frontier_group(state.frame, action.event)
                        ),
                        stop=state.skip_until,
                        emitted=lines[segment_start:],
                        body_first=segment_start,
                        frontier_completion=frontier_completion,
                    )
                continue
            if recorder is not None and materializations is not None:
                image = recorder.emit_image(action, state.boundaries, vector, unroll)
                if image is not None:
                    lines.extend(image)
                    continue
            # A rejected group has no authority over the single-output path.
            # Rebind to the original singleton event and its exact write set.
            native_inputs = None
            if state.native_requested and state.retention is not None:
                native_inputs = NativeReadInputs(
                    bind_retained_native_inputs(
                        plan,
                        state.frame,
                        state.retention,
                        state.published_native,
                        action.event,
                        action.event + 1,
                        action.writes,
                    )
                )
            lines.extend(
                emit_prepared_value(
                    cg,
                    plan,
                    state.buffers[action.writes[0]],
                    state.boundaries,
                    execution,
                    vector=vector,
                    producer_unroll=unroll,
                    vector_store=state.buffers[action.writes[0]].name
                    not in scratch.xor_buffers | state.transposed_images,
                    raw_boundaries={
                        leaf.node: leaf.name
                        for leaf in pipeline.prepared_leaves
                        if len(pipeline.prepared_leaves) > 1
                        and len(leaf.read_events) > 1
                        and state.boundaries.get(leaf.node) == leaf.name
                        and leaf.name in action.reads
                        and state.frame.layout.region(leaf.name).live_from
                        <= action.event
                        < state.frame.layout.region(leaf.name).live_until
                    },
                    native_inputs=native_inputs,
                    completion=frontier_completion,
                    shared_sink=scratch.vector_sinks.get(
                        state.buffers[action.writes[0]].name
                    ),
                )
            )
            state.native_activated |= (
                native_inputs is not None and native_inputs.activated
            )
        elif action.kind == "ready":
            if pipeline.prepared_operands or pipeline.prepared_groups:
                # The original writers publish their generic shared stores to
                # the asynchronous MMA proxy before handing the slot over.
                lines.append("cute.arch.fence_view_async_shared()")
            lines.append(execution.sync)
        else:
            assert action.kind == "mma"
        if recorder is not None:
            assert before is not None
            recorder.record(
                action,
                before,
                state.boundaries,
                emitted=lines[segment_start:],
                body_first=segment_start,
                frontier_completion=frontier_completion,
                broadcast_postlude=postlude,
            )
    if prepared_body is not None:
        return prepared_body.finish()
    assert isinstance(plan, chain.ChainedMatmulPlan)
    if root_warp is not None:
        root_warp.finish()
        return lines
    if root_actions is not None:
        root_actions.check()
        if (
            root_actions.cursor != len(root_actions.actions)
            or root_actions.event != len(root_actions.program.actions)
            or root_actions.pending is not None
            or tuple(lines) != root_actions.accepted_lines
        ):
            raise chain._UnsupportedChain("incomplete root action program")
        root_actions.consumed = True
        root_actions.finished_lines = tuple(lines)
        root_actions.finished_state = root_stage_state(root_actions.sequence)
        return lines
    if root is not None:
        if root.next_stage != len(plan.dots) or root.consumed:
            raise chain._UnsupportedChain("incomplete or repeated ordered body")
        root.check(cg, plan, root.boundaries, root.next_stage)
        root.consumed = True
        return lines
    if recurrence_body is not None:
        recurrence_body.finish(lines)
        return lines
    assert state is not None
    if state.frontier_ownership is not None:
        state.frontier_ownership.validate()
    if state.frontier_stores is not None:
        state.frontier_stores.validate()
    if state.native_requested and not state.native_activated:
        raise exc.BackendUnsupported(
            "cute",
            "native vector reads require an original published operand and a proved vector consumer",
        )
    if state.retained_nodes:
        raise exc.BackendUnsupported(
            "cute", "retained native operand was not published"
        )
    if state.register_requested and not state.used_island:
        raise exc.BackendUnsupported(
            "cute",
            "register islands require a proved typed warp region and final frame lifetimes",
        )
    if state.row_requested and not state.used_rows:
        raise exc.BackendUnsupported(
            "cute",
            "collective retention requires a proved row-local sum and complete operand fill",
        )
    return lines
