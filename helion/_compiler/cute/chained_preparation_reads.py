"""Completed, same-attempt reads of independently owned preparation exports.

A nominal action end is not completion authority. Warp and frontier readers
retain the original emitter's successful completion, never a recertification
of text supplied by a caller after that emitter returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .physical_use_frontier import PhysicalPublication
from .physical_use_frontier import PhysicalUseFrontier

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TypedDict

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_island_publication import IslandConsumerPublication
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_preparation_actions import AcceptedPreparation
    from .chained_preparation_actions import AcceptedPreparationAction
    from .chained_preparation_frame import PreparationBuffer
    from .chained_preparation_frame import PreparationFrame
    from .chained_preparation_frame import PreparationStage
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_register_binding import BoundPreparationIsland
    from .chained_register_binding import RegisterExport
    from .chained_vector_stage import VectorStaging
    from .chained_warp_stage import CompletedWarpStage

    class _IslandStageOptions(TypedDict, total=False):
        island_input: IslandConsumerPublication
        preparation_completion: PreparationWarpCompletion


@dataclass
class PreparationWarpCompletion:
    """Output slot for the existing executor token, not a new completion mint."""

    publication: CompletedWarpStage | None = None
    _publication: CompletedWarpStage | None = None

    def record(
        self,
        publication: CompletedWarpStage,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        lines: list[str],
    ) -> None:
        from .chained_matmul import _UnsupportedChain

        if (
            self._publication is not None
            or self.publication is not None
            or not publication.matches(cg, plan, boundaries, lines)
        ):
            raise _UnsupportedChain("preparation warp completion changed before return")
        self.publication = self._publication = publication

    def result(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        lines: list[str],
    ) -> CompletedWarpStage | None:
        from .chained_matmul import _UnsupportedChain

        publication = self.publication
        if publication is not self._publication or (
            publication is not None
            and not publication.matches(cg, plan, boundaries, lines)
        ):
            raise _UnsupportedChain("preparation warp completion changed after return")
        return publication


@dataclass(frozen=True)
class CompletedPreparationFrontier:
    cg: GenerateAST
    plan: ChainedMatmulPlan
    buffers: tuple[PreparationBuffer, ...]
    execution: ChainedExecution
    inputs: tuple[tuple[Node, str], ...]
    outputs: tuple[tuple[Node, str], ...]
    lines: tuple[str, ...]
    slot: PreparationFrontierCompletion
    _selection: object

    def facts(self) -> object:
        from .chained_warp_stage import _snapshot

        return _snapshot(
            (
                id(self),
                self.cg,
                self.plan,
                tuple(vars(buffer) for buffer in self.buffers),
                vars(self.execution),
                self.inputs,
                self.outputs,
                self.lines,
                id(self.slot),
                self.cg.device_function.config.config,
            )
        )

    def matches(
        self,
        plan: ChainedMatmulPlan,
        inputs: tuple[tuple[Node, str], ...],
        outputs: tuple[tuple[Node, str], ...],
        lines: tuple[str, ...],
        *,
        consumed: bool,
    ) -> bool:
        return (
            self.slot.publication is self
            and self.slot._publication is self
            and self.slot.consumed is consumed
            and self._selection == self.facts()
            and self.plan is plan
            and self.inputs == inputs
            and self.outputs == outputs
            and self.lines == lines
        )


@dataclass
class PreparationFrontierCompletion:
    """One original frontier return, consumed once by the existing recorder."""

    publication: CompletedPreparationFrontier | None = None
    _publication: CompletedPreparationFrontier | None = None
    consumed: bool = False

    def record(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        buffers: tuple[PreparationBuffer, ...],
        execution: ChainedExecution,
        inputs: tuple[tuple[Node, str], ...],
        boundaries: dict[Node, str],
        lines: list[str],
    ) -> None:
        from .chained_matmul import _UnsupportedChain

        if (
            self.publication is not None
            or self._publication is not None
            or self.consumed
        ):
            raise _UnsupportedChain("duplicate frontier completion")
        publication = CompletedPreparationFrontier(
            cg,
            plan,
            buffers,
            execution,
            inputs,
            tuple(boundaries.items()),
            tuple(lines),
            self,
            None,
        )
        object.__setattr__(publication, "_selection", publication.facts())
        self.publication = self._publication = publication

    def consume(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        execution: ChainedExecution,
        inputs: tuple[tuple[Node, str], ...],
        outputs: tuple[tuple[Node, str], ...],
        lines: tuple[str, ...],
    ) -> CompletedPreparationFrontier:
        from .chained_matmul import _UnsupportedChain

        publication = self.publication
        if (
            publication is None
            or publication.cg is not cg
            or publication.execution != execution
            or not publication.matches(plan, inputs, outputs, lines, consumed=False)
        ):
            raise _UnsupportedChain("frontier completion changed after return")
        self.consumed = True
        return publication


class PreparationReadFrontier:
    """One original emission attempt's typed publication environment.

    This adapts complete semantic owners, not their eventual storage offsets.
    Existing accepted/scheduled transfer binders consume the returned owner
    names with their original phases and completion proofs.
    """

    def __init__(
        self, pipeline: PreparationPipeline, shapes: Mapping[Node, tuple[int, ...]]
    ) -> None:
        self.pipeline = pipeline
        self.shapes = dict(shapes)
        self.frontier = PhysicalUseFrontier(pipeline.frame.cut.region.nodes)
        self.buffers = {buffer.name: buffer for buffer in pipeline.frame.buffers}
        self.retained = (
            {}
            if pipeline.operand_retention is None
            else {
                f"chain_retained_operand_{index}": item
                for index, item in enumerate(pipeline.operand_retention.candidates)
            }
        )
        self.publications: dict[tuple[Node, str], PhysicalPublication] = {}
        self.islands: list[IslandConsumerPublication] = []
        for buffer in self.buffers.values():
            self.frontier.register_owner(
                buffer,
                pipeline.frame,
                pipeline.frame.layout.region(buffer.name).live_from,
            )
        self._facts = self.facts()

    def facts(self) -> object:
        return (
            id(self.pipeline),
            id(self.pipeline.frame),
            id(self.pipeline.frame.cut),
            id(self.pipeline.frame.cut.region),
            id(self.pipeline.operand_retention),
            tuple(id(buffer) for buffer in self.pipeline.frame.buffers),
            ()
            if self.pipeline.operand_retention is None
            else tuple(id(item) for item in self.pipeline.operand_retention.candidates),
            tuple(self.shapes.items()),
            tuple(
                (name, id(b), b.node, b.kind, b.dtype, b.shape)
                for name, b in self.buffers.items()
            ),
            tuple(
                (
                    name,
                    id(c),
                    c.node,
                    c.owner,
                    c.logical_shape,
                    c.full_shape,
                    c.logical_modes,
                    c.row_offset,
                    c.dtype,
                    c.view_path,
                    c.publication_event,
                )
                for name, c in self.retained.items()
            ),
        )

    def check(self, islands: tuple[IslandConsumerPublication, ...] = ()) -> None:
        if (
            self.facts() != self._facts
            or len(self.islands) != len(islands)
            or any(
                left is not right
                for left, right in zip(self.islands, islands, strict=True)
            )
        ):
            raise ValueError("late preparation publication environment changed")
        if any(not item.matches() for item in self.islands):
            raise ValueError("late preparation native publication changed")
        self.frontier.check()

    def publication(self, node: Node, name: str) -> PhysicalPublication | None:
        retained = self.retained.get(name)
        owner = self.buffers.get(name if retained is None else retained.owner)
        if owner is None:
            return None
        if node not in self.shapes:
            raise ValueError("late publication is outside the original typed graph")
        shape = self.shapes[node]
        if retained is not None:
            if (
                retained.node is not node
                or shape != retained.logical_shape
                or owner.shape != retained.full_shape
                or owner.dtype != retained.dtype
            ):
                raise ValueError("late retained publication lost its full owner/view")
        elif owner.node is not node or owner.shape != shape:
            # A native island publication is a different SSA value in this
            # original complete stage owner. Only its real successful token
            # proves that relation; buffer names/shape coincidence do not.
            matches = tuple(
                item
                for item in self.islands
                if item.candidate.operand is node
                and dict(item.outputs).get(node) == name
            )
            if len(matches) != 1:
                raise ValueError("late publication lost its original typed value")
            item = matches[0]
            candidate = item.candidate
            if (
                # Full graph/receipt validation belongs to check() at the
                # enclosing acceptance/finalizer boundary, not every leaf.
                item._published != (candidate, item.body_first, item.lines)
                or candidate._selection != candidate._fields()
                or candidate.pipeline is not self.pipeline
                or candidate.owner != owner.name
                or candidate.shape != shape
                or candidate.shape != owner.shape
                or candidate.dtype != owner.dtype
            ):
                raise ValueError("late publication lost its actual native owner/view")
        key = node, name
        value = self.publications.get(key)
        if value is None:
            value = PhysicalPublication(node, owner, owner.dtype, shape)
            self.publications[key] = value
        elif (
            value.node is not node
            or value.owner is not owner
            or value.dtype != owner.dtype
            or value.shape != shape
        ):
            raise ValueError("late publication cache lost original owner/view")
        return value


def resolved_preparation_reads(
    pipeline: PreparationPipeline,
    nodes: tuple[Node, ...],
    boundaries: Mapping[Node, str],
    *,
    use_frontier: PreparationReadFrontier,
    published: tuple[tuple[Node, str], ...],
    fragments: BoundPreparationIsland | None = None,
) -> tuple[str, ...]:
    """Original dependency closure stopped at actual fragment/published SSA.

    Fragment stops mirror the original expression emitter's precedence over
    shared boundaries. Callers must separately prove their coordinate ownership;
    this graph walk does not authorize a register binding or storage release.
    """
    if use_frontier.pipeline is not pipeline or tuple(boundaries.items()) != published:
        raise ValueError("late reads lost original completed publication inventory")
    stops = frozenset()
    if fragments is not None:
        if not fragments.exports or not fragments.matches(
            fragments.plan,
            pipeline.frame,
            fragments.execution,
            dict(boundaries),
        ):
            raise ValueError("late reads lack original fragment binding")
        coordinates = fragments.coordinates(fragments.exports[0].node, "chain_read")
        if any(
            fragments.coordinates(item.node, "chain_read") != coordinates
            for item in fragments.exports
        ):
            raise ValueError("late reads cross fragment coordinates")
        stops = frozenset(item.node for item in fragments.exports)
    publications = {}
    for node, name in published:
        if node not in stops:
            value = use_frontier.publication(node, name)
            if value is not None:
                publications[node] = value
            else:
                raise ValueError("unowned published input")
    resolved = use_frontier.frontier.resolve(
        nodes,
        publications,
        external=stops,
        required=frozenset(node for node, _ in published),
        traversable=frozenset(pipeline.frame.cut.region.nodes),
    )
    names = {id(buffer): buffer.name for buffer in pipeline.frame.buffers}
    return tuple(sorted({names[id(value.owner)] for value in resolved}))


@dataclass(frozen=True)
class CompletedPreparationStage:
    plan: ChainedMatmulPlan
    pipeline: PreparationPipeline
    stage: PreparationStage
    execution: ChainedExecution
    first: int
    stop: int
    inputs: tuple[tuple[Node, str], ...]
    outputs: tuple[tuple[Node, str], ...]
    roots: tuple[Node, ...]
    reads: tuple[str, ...]
    body_first: int
    lines: tuple[str, ...]
    _selection: tuple[object, ...]
    island_input: IslandConsumerPublication | None = None
    warp_completion: CompletedWarpStage | None = None
    warp_revision: object = None
    warp_aliases: tuple[tuple[str, str], ...] = ()

    def _fields(self) -> tuple[object, ...]:
        result = (
            self.plan,
            self.pipeline,
            self.stage,
            self.execution,
            self.first,
            self.stop,
            self.inputs,
            self.outputs,
            self.roots,
            self.reads,
            self.body_first,
            self.lines,
        )
        if self.island_input is not None:
            result = (*result, self.island_input)
        if self.warp_completion is not None:
            result = (
                *result,
                self.warp_completion,
                self.warp_completion._selection,
                self.warp_revision,
                self.warp_aliases,
            )
        return result

    def original_warp_matches(self) -> bool:
        """Preserve the already-validated token while later aliases accumulate.

        Keep the existing executor's complete non-alias revision current and its
        original alias prefix intact; only later appended aliases are permitted.
        """
        from .chained_warp_stage import _execution
        from .chained_warp_stage import _snapshot
        from .chained_warp_stage import warp_revision

        publication = self.warp_completion
        if publication is None:
            return False
        prepared = publication.prepared
        completion = prepared.completion
        return (
            publication._selection == publication.facts()
            and completion._state.publication is publication
            and completion._state.prepared is prepared
            and completion._state.consumed is True
            and completion._consumed is True
            and prepared._selection == prepared.facts()
            and completion._selection == completion.fields()
            and completion.plan is self.plan
            and self.warp_revision == warp_revision(self.plan, aliases=False)
            and tuple(self.plan.tensor_aliases.items())[: len(self.warp_aliases)]
            == self.warp_aliases
            and completion.execution == self.execution
            and completion._context == _execution(self.execution)
            and completion._config
            == _snapshot(completion.codegen.device_function.config.config)
            and completion.boundaries == self.inputs
            and publication.boundaries == self.outputs
            and publication.prefix == self.lines
        )

    def matches(
        self, accepted: AcceptedPreparation, action: AcceptedPreparationAction
    ) -> bool:
        frame = accepted.pipeline.frame
        if (
            self._selection != self._fields()
            or self.plan is not accepted.revision.plan
            or self.pipeline is not accepted.pipeline
            or action.proof is not self
            or action.kind != "ordinary"
            or (action.first, action.stop) != (self.first, self.stop)
            or action.inputs != self.inputs
            or not (
                self.island_input.stage_outputs_match(self, action)
                if self.island_input is not None
                else all(
                    dict(action.outputs).get(node) == name
                    for node, name in self.outputs
                )
            )
            or action.reads != self.reads
            or self.stage not in frame.stages
            or self.execution.threads != accepted.execution.threads
            or self.execution.thread != accepted.execution.thread
            or self.execution.sync != accepted.execution.sync
            or type(self.body_first) is not int
            or self.body_first < 0
            or accepted.lines[self.body_first : self.body_first + len(self.lines)]
            != self.lines
            or not _complete_span(frame, self.stage, self.first, self.stop)
            or self.warp_completion is not None
            and not self.original_warp_matches()
        ):
            return False
        roots = tuple(
            source
            for index in self.stage.group.stages
            for source in self.plan.dots[index].all_input_nodes
        )
        try:
            if self.island_input is not None:
                item = self.island_input
                if (
                    not item.matches()
                    or not item.consumed
                    or self.first != item.candidate.bound.stop_event
                ):
                    return False
                roots = tuple(
                    source for source in roots if source is not item.candidate.operand
                )
                reads = tuple(
                    sorted(
                        {
                            *resolved_preparation_reads(
                                self.pipeline,
                                roots,
                                dict(self.inputs),
                                use_frontier=accepted.read_frontier,
                                published=action.inputs,
                            ),
                            item.candidate.owner,
                        }
                    )
                )
                return self.roots == roots and self.reads == reads
            return self.roots == roots and self.reads == resolved_preparation_reads(
                self.pipeline,
                roots,
                dict(self.inputs),
                use_frontier=accepted.read_frontier,
                published=action.inputs,
            )
        except ValueError:
            return False


def _complete_span(
    frame: PreparationFrame, stage: PreparationStage, first: int, stop: int
) -> bool:
    if not 0 <= first < stop <= len(frame.actions) or stop != first + 2:
        return False
    fill, mma = frame.actions[first:stop]
    return (
        fill.kind == "fill"
        and mma.kind == "mma"
        and fill.stages == mma.stages == stage.group.stages
        and fill.nodes == mma.nodes
        and stage in frame.stages
    )


def emit_completed_preparation_stage(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    stage: PreparationStage,
    boundaries: dict[Node, str],
    execution: ChainedExecution,
    vector: VectorStaging,
    unroll: BoundedProducerUnroll,
    *,
    first: int,
    body_first: int,
    use_frontier: PreparationReadFrontier,
    published: tuple[tuple[Node, str], ...],
    island_input: IslandConsumerPublication | None = None,
    capture_warp: bool = False,
) -> tuple[list[str], CompletedPreparationStage]:
    """Invoke the concrete complete stage once; no callback or async enqueue."""
    from .chained_tcgen_stage import emit_stage

    stop = first + 2
    if not _complete_span(pipeline.frame, stage, first, stop):
        raise ValueError("preparation stage lacks its complete original action span")
    inputs = tuple(boundaries.items())
    roots = tuple(
        source
        for index in stage.group.stages
        for source in plan.dots[index].all_input_nodes
    )
    if island_input is not None:
        roots = tuple(
            source for source in roots if source is not island_input.candidate.operand
        )
    reads = resolved_preparation_reads(
        pipeline, roots, boundaries, use_frontier=use_frontier, published=published
    )
    if island_input is not None:
        reads = tuple(sorted({*reads, island_input.candidate.owner}))
    # This exact existing protocol completes the MMA and its result publication
    # join before returning. No direct/overlap/root/alternate producer is passed.
    options: _IslandStageOptions = {}
    if island_input is not None:
        options["island_input"] = island_input
    completion = PreparationWarpCompletion() if capture_warp else None
    if completion is not None:
        options["preparation_completion"] = completion
    lines = emit_stage(
        cg,
        plan,
        boundaries,
        stage.group.stages[0],
        stage.group.geometries[0],
        "0",
        stage.group,
        vector,
        unroll,
        execution=execution,
        prepared_shape=stage.shape,
        **options,
    )
    warp = (
        completion.result(cg, plan, boundaries, lines)
        if completion is not None
        else None
    )
    fields = (
        plan,
        pipeline,
        stage,
        execution,
        first,
        stop,
        inputs,
        tuple(boundaries.items()),
        roots,
        reads,
        body_first,
        tuple(lines),
    )
    selection = fields if island_input is None else (*fields, island_input)
    if warp is not None:
        from .chained_warp_stage import warp_revision

        revision = warp_revision(plan, aliases=False)
        aliases = tuple(plan.tensor_aliases.items())
        selection = (*selection, warp, warp._selection, revision, aliases)
    else:
        revision, aliases = None, ()
    return lines, CompletedPreparationStage(
        *fields, selection, island_input, warp, revision, aliases
    )


@dataclass(frozen=True)
class IslandExportLease:
    export: RegisterExport
    producer: int
    readers: tuple[CompletedPreparationStage, ...]
    original_start: int
    original_stop: int
    stop: int


def _escapes_preparation(accepted: AcceptedPreparation, node: Node) -> bool:
    cut = accepted.pipeline.frame.cut
    pending = [*cut.recurrence, *cut.region.stores]
    pending.extend(carry.output for carry in cut.carries)
    boundaries = {image.node for image in cut.images} | set(cut.shared_inputs)
    visited = set()
    while pending:
        current = pending.pop()
        if current is node:
            return True
        if current in visited or current in boundaries:
            continue
        visited.add(current)
        pending.extend(current.all_input_nodes)
    return False


def plan_island_export_leases(
    accepted: AcceptedPreparation,
) -> tuple[IslandExportLease, ...] | None:
    """Shorten only original standalone FP32 exports with completed readers.

    Unsupported reader protocols preserve the original lease. A stale sealed
    stage receipt rejects the physical plan rather than silently falling back.
    """
    from .chained_register_binding import BoundPreparationIsland

    plan, pipeline = accepted.revision.plan, accepted.pipeline
    if not accepted.matches(plan, pipeline, dict(accepted.revision.shapes)):
        return None
    for action in accepted.actions:
        if isinstance(
            action.proof, CompletedPreparationStage
        ) and not action.proof.matches(accepted, action):
            return None
    frame = pipeline.frame
    scan = pipeline.scan_producer
    effective = frame.layout.regions if scan is None else scan.regions
    regions = {region.name: region for region in effective}
    scale = len(frame.actions) + 2 if scan is not None else 1

    def completion(stop: int) -> int:
        return (
            scan.phases[stop]
            if scan is not None and stop < len(scan.phases)
            else stop * scale
        )

    result = []
    for producer, action in enumerate(accepted.actions):
        if action.kind != "island" or not isinstance(
            action.proof, BoundPreparationIsland
        ):
            continue
        for export in action.proof.exports:
            buffers = tuple(
                buffer for buffer in frame.buffers if buffer.node is export.node
            )
            if (
                len(buffers) != 1
                or buffers[0].name != export.name
                or buffers[0].kind != "c"
                or buffers[0].dtype != torch.float32
                or export.name not in regions
                or export.name not in action.writes
                or export.name in action.reads
                or _escapes_preparation(accepted, export.node)
                or any(
                    item.buffer.node is export.node
                    for item in pipeline.prepared_operands
                )
                or any(
                    member.buffer.node is export.node
                    for binding in pipeline.prepared_groups
                    for member in binding.candidate.members
                )
                or pipeline.operand_retention is not None
                and any(
                    item.node is export.node or item.owner == export.name
                    for item in pipeline.operand_retention.candidates
                )
                or any(
                    leaf.node is export.node or leaf.name == export.name
                    for leaf in pipeline.prepared_leaves
                )
            ):
                continue
            readers = tuple(
                item for item in accepted.actions if export.name in item.reads
            )
            if (
                not readers
                or any(
                    item.first < action.stop
                    or not isinstance(item.proof, CompletedPreparationStage)
                    for item in readers
                )
                or any(
                    export.name in item.writes
                    for item in accepted.actions[producer + 1 :]
                )
            ):
                continue
            receipts = tuple(
                item.proof
                for item in readers
                if isinstance(item.proof, CompletedPreparationStage)
            )
            region = regions[export.name]
            stop = max(completion(item.stop) for item in receipts)
            if region.live_from < stop < region.live_until:
                result.append(
                    IslandExportLease(
                        export,
                        producer,
                        receipts,
                        region.live_from,
                        region.live_until,
                        stop,
                    )
                )
    return tuple(result)
