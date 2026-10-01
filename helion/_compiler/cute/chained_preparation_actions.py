"""Same-attempt preparation choices accepted by the original emitters.

The mathematical body is built once. Only the private recorder used by that
walk can seal its choices; candidates alone never authorize an omission.
Physical rebinding and complete pipeline finalization are still required.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Literal
from typing import NoReturn

import torch

from ... import exc
from .chained_frontier_groups import FrontierGroup
from .chained_frontier_groups import plan_frontier_group
from .chained_frontier_materialization import FrontierMaterializationAttempt
from .chained_frontier_materialization import FrontierTransferReceipts
from .chained_frontier_materialization import plan_frontier_materializations
from .chained_operand_retention import _InvalidRetention
from .chained_operand_retention import _Revision
from .chained_operand_retention import _revision
from .chained_preparation_reads import CompletedPreparationStage
from .chained_preparation_reads import PreparationReadFrontier
from .chained_preparation_reads import emit_completed_preparation_stage
from .chained_preparation_reads import resolved_preparation_reads
from .chained_prepared_image_emission import RawImageBinding
from .chained_prepared_image_emission import emit_raw_image
from .chained_prepared_image_transfers import NativeIdentityCrop
from .chained_register_binding import BoundPreparationIsland
from .chained_scan_producer import ScanProducer
from .chained_scan_transfers import ScanTransferAttempt

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_broadcast_retention import BroadcastActionBody
    from .chained_broadcast_retention import BroadcastRetentionAttempt
    from .chained_broadcast_retention import BroadcastRetentionReceipt
    from .chained_execution import ChainedExecution
    from .chained_island_publication import IslandConsumerPublication
    from .chained_island_publication import IslandPublicationRead
    from .chained_leaf_schedule_emission import LeafEmissionCapture
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_preparation_fragment import PreparationFragment
    from .chained_preparation_frame import PreparationAction
    from .chained_preparation_frame import PreparationStage
    from .chained_preparation_leaves import PreparationLeaf
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_preparation_reads import PreparationFrontierCompletion
    from .chained_prepared_image_emission import BoundRawWidenings
    from .chained_prepared_image_transfers import NativeCropPlan
    from .chained_scan_producer_emission import ScanProducerEmission
    from .chained_scan_transfers import ScanTransferReceipt
    from .chained_scratch_layout import ScratchLayouts
    from .chained_vector_stage import VectorStaging


ActionKind = Literal["ordinary", "island", "scan", "frontier", "raw", "crop"]
ActionProof = (
    BoundPreparationIsland
    | ScanProducer
    | FrontierGroup
    | RawImageBinding
    | NativeIdentityCrop
    | FrontierTransferReceipts
    | CompletedPreparationStage
    | None
)


@dataclass(frozen=True)
class AcceptedPreparationAction:
    first: int
    stop: int
    kind: ActionKind
    proof: ActionProof
    inputs: tuple[tuple[Node, str], ...]
    outputs: tuple[tuple[Node, str], ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    omitted: tuple[str, ...]
    scan_transfer: ScanTransferReceipt | None = None
    broadcasts: tuple[BroadcastRetentionReceipt, ...] = ()
    island_publication: IslandConsumerPublication | None = None
    broadcast_body: BroadcastActionBody | None = None
    fragment: PreparationFragment | None = None


@dataclass(frozen=True)
class AcceptedPreparation:
    revision: _Revision
    pipeline: PreparationPipeline
    execution: ChainedExecution
    raw: BoundRawWidenings | None
    crops: NativeCropPlan | None
    actions: tuple[AcceptedPreparationAction, ...]
    lines: tuple[str, ...]
    workspaces: tuple[str, ...]
    vector_state: tuple[bool, bool, bool, bool, bool, bool]
    unroll_state: tuple[int, bool, bool]
    scratch_mode: str
    transport_facts: tuple[object, ...]
    scan_transfers: tuple[ScanTransferReceipt, ...]
    read_frontier: PreparationReadFrontier
    _selection: tuple[object, ...]
    broadcasts: tuple[BroadcastRetentionReceipt, ...] = ()
    broadcast_context: tuple[object, ...] = ()
    island_publications: tuple[IslandConsumerPublication, ...] = ()
    island_reads: tuple[IslandPublicationRead, ...] = ()
    fragments: tuple[PreparationFragment, ...] = ()

    def _fields(self) -> tuple[object, ...]:
        fields = (
            self.revision,
            self.pipeline,
            self.execution,
            self.raw,
            self.crops,
            self.actions,
            self.lines,
            self.workspaces,
            self.vector_state,
            self.unroll_state,
            self.scratch_mode,
            self.transport_facts,
            self.scan_transfers,
            self.read_frontier,
        )
        result = (
            fields
            if not self.broadcast_context
            else (*fields, self.broadcasts, self.broadcast_context)
        )
        result = (
            result
            if not self.island_publications
            else (*result, self.island_publications, self.island_reads)
        )
        return (*result, self.fragments) if self.fragments else result

    def matches(
        self,
        plan: ChainedMatmulPlan,
        pipeline: PreparationPipeline,
        shapes: Mapping[Node, tuple[int, ...]],
    ) -> bool:
        if (
            self._selection != self._fields()
            or any(not item.accepted(self) for item in self.fragments)
            or self.revision.plan is not plan
            or self.pipeline is not pipeline
            or self.revision.frame is not pipeline.frame
            or self.transport_facts != _transport_facts(pipeline)
            or self.island_publications
            != tuple(
                action.island_publication
                for action in self.actions
                if action.island_publication is not None
            )
            or any(
                action.island_publication is not None
                and not action.island_publication.accepted(self, action)
                for action in self.actions
            )
            or self.scan_transfers
            != tuple(
                action.scan_transfer
                for action in self.actions
                if action.scan_transfer is not None
            )
            or self.broadcasts
            != tuple(item for action in self.actions for item in action.broadcasts)
            or self.broadcast_context
            and self.broadcast_context != _broadcast_context(self.actions)
            or any(
                not item.matches(self, action)
                for action in self.actions
                for item in action.broadcasts
            )
        ):
            return False
        try:
            self.read_frontier.check(self.island_publications)
            return self.revision == _revision(plan, pipeline.frame, shapes)
        except (_InvalidRetention, ValueError):
            return False


def _broadcast_context(
    actions: tuple[AcceptedPreparationAction, ...],
) -> tuple[object, ...]:
    return tuple(
        (
            action.first,
            action.stop,
            action.kind,
            action.proof,
            action.inputs,
            action.outputs,
            action.reads,
            action.writes,
            action.omitted,
            action.scan_transfer,
            action.broadcasts,
            action.broadcast_body,
        )
        for action in actions
    )


def _transport_facts(pipeline: PreparationPipeline) -> tuple[object, ...]:
    """Freeze descriptor inputs used by the body, not just their owners.

    Native bindings and role records are immutable, but each TMA wrapper owns
    mutable lists/dictionaries. The same object identity does not preserve the
    descriptor arguments, guard or layout emitted during this attempt.
    """
    from .chained_pipeline_storage import _freeze

    return (
        pipeline.slots,
        pipeline.preparation_threads,
        pipeline.recurrence_threads,
        pipeline.protocol_bytes,
        pipeline.recurrence,
        pipeline.residency,
        pipeline.cohorts,
        pipeline.prepared_operands,
        pipeline.prepared_groups,
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


def workspace_name(name: str) -> str:
    return f"chain_preparation_workspace_{name}"


class _PreparationRecorder:
    """Private companion of the one original `_prepare` emission walk."""

    def __init__(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        pipeline: PreparationPipeline,
        execution: ChainedExecution,
        revision: _Revision,
        raw: BoundRawWidenings | None,
        crops: NativeCropPlan | None,
        leaf_capture: LeafEmissionCapture | None = None,
        broadcast_retention: bool = False,
        island_consumers: bool = False,
        fragment_epilogues: bool = False,
    ) -> None:
        self.cg = cg
        self.plan = plan
        self.pipeline = pipeline
        self.execution = execution
        self.revision = revision
        self.read_frontier = PreparationReadFrontier(pipeline, dict(revision.shapes))
        from .chained_preparation_fragment import plan_preparation_fragment

        self.fragment = (
            plan_preparation_fragment(cg, plan, pipeline, execution, self.read_frontier)
            if fragment_epilogues
            else None
        )
        if fragment_epilogues and self.fragment is None:
            self.fail(
                "fragment epilogues require a grouped result at a proved cache cut"
            )
        self.transport_facts = _transport_facts(pipeline)
        self.raw = raw
        self.crops = crops
        self.leaf_capture = leaf_capture
        self.broadcast_retention = broadcast_retention
        self.island_consumers = island_consumers
        self._publication_receipts: tuple[IslandConsumerPublication, ...] = ()
        self._publication_reads: tuple[IslandPublicationRead, ...] = ()
        self.broadcast_attempt: BroadcastRetentionAttempt | None = None
        self._broadcasts: tuple[BroadcastRetentionReceipt, ...] = ()
        self.actions: list[AcceptedPreparationAction] = []
        self.workspaces: list[str] = []
        self.completed_stages: dict[int, CompletedPreparationStage] = {}
        self.scan_attempt: ScanTransferAttempt | None = None
        self._scan_receipts: tuple[ScanTransferReceipt, ...] = ()
        self.cursor = 0
        self.sealed = False
        self.special: dict[str, RawImageBinding | NativeIdentityCrop] = (
            {item.transfer.buffer.name: item for item in raw.bindings}
            if raw is not None
            else {}
        )
        if crops is not None:
            if self.special.keys() & {item.buffer.name for item in crops.crops}:
                self.fail("overlapping typed image choices")
            self.special.update((item.buffer.name, item) for item in crops.crops)

    @staticmethod
    def fail(reason: str) -> NoReturn:
        raise exc.BackendUnsupported("cute", f"accepted preparation: {reason}")

    def begin_broadcast(self, first: int) -> BroadcastRetentionAttempt | None:
        from .chained_broadcast_retention import BroadcastRetentionAttempt

        if self.broadcast_attempt is not None or self.sealed or first != self.cursor:
            self.fail("unfinished broadcast producer")
        if self.broadcast_retention:
            self.broadcast_attempt = BroadcastRetentionAttempt(first, self.execution)
        return self.broadcast_attempt

    def workspace(self, stage: PreparationStage, role: Literal["a", "b"]) -> str:
        if self.sealed:
            self.fail("workspace requested after sealing")
        region = stage.a if role == "a" else stage.b
        if region != self.pipeline.frame.layout.region(region.name):
            self.fail("stage workspace changed")
        if region.name not in self.workspaces:
            self.workspaces.append(region.name)
        return workspace_name(region.name)

    def permits_frontier_group(self, first: int) -> bool:
        group = plan_frontier_group(self.pipeline.frame, first)
        return group is None or not any(
            buffer.name in self.special for buffer in group.buffers
        )

    def frontier_materializations(
        self, action: PreparationAction, boundaries: Mapping[Node, str]
    ) -> FrontierMaterializationAttempt | None:
        if action.kind != "frontier":
            return None
        group = plan_frontier_group(self.pipeline.frame, action.event)
        if group is None or any(
            isinstance(self.special.get(buffer.name), RawImageBinding)
            for buffer in group.buffers
        ):
            return None
        selected = plan_frontier_materializations(
            self.cg,
            self.plan,
            self.pipeline,
            self.revision,
            self.crops,
            action.event,
            boundaries,
        )
        return None if selected is None else FrontierMaterializationAttempt(selected)

    def _reads(
        self, nodes: tuple[Node, ...], boundaries: Mapping[Node, str]
    ) -> tuple[str, ...]:
        try:
            return resolved_preparation_reads(
                self.pipeline,
                nodes,
                boundaries,
                use_frontier=self.read_frontier,
                published=self.actions[-1].outputs if self.actions else (),
            )
        except ValueError as error:
            self.fail(str(error))

    def emit_stage(
        self,
        stage: PreparationStage,
        boundaries: dict[Node, str],
        execution: ChainedExecution,
        vector: VectorStaging,
        unroll: BoundedProducerUnroll,
        *,
        first: int,
        body_first: int,
    ) -> list[str]:
        if self.sealed or first != self.cursor or first in self.completed_stages:
            self.fail("duplicate or reordered completed stage")
        publication = next(
            (
                item
                for item in self._publication_receipts
                if item.candidate.bound.stop_event == first
            ),
            None,
        )
        options = {} if publication is None else {"island_input": publication}
        if publication is not None:
            self.check_island_action(first, vector)
        lines, receipt = emit_completed_preparation_stage(
            self.cg,
            self.plan,
            self.pipeline,
            stage,
            boundaries,
            execution,
            vector,
            unroll,
            first=first,
            body_first=body_first,
            use_frontier=self.read_frontier,
            published=self.actions[-1].outputs if self.actions else (),
            capture_warp=self.island_consumers,
            **options,
        )
        self.completed_stages[first] = receipt
        return lines

    def complete_retained(
        self,
        first: int,
        stop: int,
        before: tuple[tuple[Node, str], ...],
        boundaries: dict[Node, str],
        lines: tuple[str, ...],
    ) -> None:
        for publication in self._publication_receipts:
            item = publication.candidate.retained_input
            if item is None or not first < item[1].publication_event <= stop:
                continue
            completion = self.completed_stages.get(first)
            if self.sealed or completion is None or completion.stop != stop:
                self.fail("island retained alias has no pending original stage")
            publication.record_retained(completion, before, boundaries, lines)

    def check_island_action(self, first: int, vector: VectorStaging) -> None:
        """Use the actual original-action captures, never global flag approval."""
        if (
            self.sealed
            or first != self.cursor
            or vector.broadcast is not self.broadcast_attempt
            or self.broadcast_retention != (self.broadcast_attempt is not None)
        ):
            self.fail("island action lost its original broadcast capture")
        if self.broadcast_attempt is not None and (
            self.broadcast_attempt.first != first
            or self.broadcast_attempt.execution != self.execution
            or self.broadcast_attempt.result()
        ):
            self.fail("island action has stale broadcast emissions")
        if self.leaf_capture is not None and (
            self.leaf_capture.sealed
            or self.leaf_capture.pending
            or self.leaf_capture.core is not None
        ):
            self.fail("island action crosses an unfinished leaf/scan segment")

    def emit_island(
        self,
        bound: BoundPreparationIsland,
        boundaries: dict[Node, str],
        vector: VectorStaging,
        *,
        body_first: int,
    ) -> list[str] | None:
        from .chained_island_publication import IslandConsumerPublication
        from .chained_island_publication import plan_island_consumer
        from .chained_register_emission import emit_register_island

        if self.sealed or bound.first_event != self.cursor:
            self.fail("island publication action reordered")
        self.check_island_action(bound.first_event, vector)
        if bound.island_input is not None:
            if not any(
                bound.island_input is item for item in self._publication_receipts
            ):
                self.fail("island input belongs to a different preparation attempt")
            lines = emit_register_island(
                self.cg, self.plan, bound, boundaries, body_first=body_first
            )
            if lines is None or bound.island_input.island_completion is None:
                self.fail("selected published-input island did not complete")
            return lines
        candidate = plan_island_consumer(
            self.cg,
            self.plan,
            self.pipeline,
            self.revision,
            bound,
            boundaries,
            vector,
            use_frontier=self.read_frontier,
            published=self.actions[-1].outputs if self.actions else (),
        )
        if candidate is None:
            return emit_register_island(self.cg, self.plan, bound, boundaries)
        retained = self.pipeline.operand_retention
        if retained is not None and any(
            bound.first_event < item.publication_event <= bound.stop_event
            for item in retained.candidates
        ):
            self.fail("island cannot replace an original retained operand publication")
        publication = IslandConsumerPublication(candidate, body_first)
        lines = emit_register_island(
            self.cg, self.plan, bound, boundaries, publication=publication
        )
        if lines is None:
            self.fail("selected island consumer did not emit")
        self.workspace(candidate.stage, candidate.role)
        self._publication_receipts = (*self._publication_receipts, publication)
        return lines

    def input_island(
        self, first: int, boundaries: dict[Node, str]
    ) -> BoundPreparationIsland | None:
        """Discover only behind an actual completed native publication cut.

        A declined candidate leaves the original consuming stage unchanged.
        After selecting and emitting an island, failure is never a fallback.
        """
        from .chained_register_binding import bind_preparation_island
        from .chained_register_islands import plan_register_islands

        publications = tuple(
            item
            for item in self._publication_receipts
            if item.candidate.bound.stop_event == first and not item.consumed
        )
        if not publications:
            return None
        if self.sealed or self.cursor != first or len(publications) != 1:
            self.fail("published-input island has a foreign action position")
        publication = publications[0]
        item = publication.candidate
        if not publication.matches() or boundaries is not item.boundary_owner:
            self.fail("published-input island lost its actual full owner")
        if item.retained_input is None:
            return None
        revision = item.bound.island.revision
        candidates = plan_register_islands(
            revision.region,
            revision.groups,
            dict(revision.shapes),
            fast_math=True,
            entry_boundaries=revision.entry_boundaries | {item.operand},
            multi_image=True,
        )
        selected = tuple(
            candidate
            for candidate in candidates
            if candidate.groups[0].stages[0] == item.consumer
        )
        if len(selected) != 1:
            return None
        return bind_preparation_island(
            self.cg,
            self.plan,
            self.pipeline.frame,
            selected[0],
            self.execution,
            boundaries,
            island_input=publication,
        )

    def begin_scan(
        self,
        first: int,
        before: Mapping[Node, str],
        *,
        body_first: int,
    ) -> None:
        scan = self.pipeline.scan_producer
        if (
            self.sealed
            or self.scan_attempt is not None
            or first != self.cursor
            or scan is None
            or first != scan.first_event
        ):
            self.fail("duplicate or reordered scan transfer")
        self.scan_attempt = ScanTransferAttempt(
            self.plan,
            self.pipeline,
            tuple(before.items()),
            body_first,
            _transport_facts(self.pipeline),
            fragment=self.fragment,
        )

    def complete_scan(
        self,
        published: Mapping[Node, str],
        after: Mapping[Node, str],
        execution: ChainedExecution,
        emission: ScanProducerEmission,
        lines: list[str],
    ) -> None:
        if self.sealed or self.scan_attempt is None:
            self.fail("scan transfer has no pending emission")
        try:
            self.scan_attempt.complete(published, after, execution, emission, lines)
        except ValueError as error:
            self.fail(str(error))
        if self.scan_attempt.receipt is not None:
            self._scan_receipts = (*self._scan_receipts, self.scan_attempt.receipt)

    def record(
        self,
        action: PreparationAction,
        before: Mapping[Node, str],
        after: Mapping[Node, str],
        *,
        kind: ActionKind = "ordinary",
        proof: ActionProof = None,
        stop: int | None = None,
        emitted: Sequence[str] = (),
        postlude: Sequence[str] = (),
        broadcast_postlude: Sequence[str] = (),
        body_first: int | None = None,
        frontier_completion: PreparationFrontierCompletion | None = None,
    ) -> None:
        frame = self.pipeline.frame
        # Ordinary stage/scan callbacks emit the following MMA themselves.
        # Its existing bookkeeping action remains a no-op, not a second issue.
        if action.kind == "mma" and self.cursor == action.event + 1:
            return
        if self.sealed or action.event != self.cursor:
            self.fail("missing, duplicate, or reordered action")
        if stop is None:
            stop = action.event + (2 if action.kind == "fill" else 1)
        if not action.event < stop <= len(frame.actions):
            self.fail("invalid action span")
        if tuple(before.items()) != (self.actions[-1].outputs if self.actions else ()):
            self.fail("action lost original completed publication inventory")
        span = frame.actions[action.event : stop]
        writes = tuple(dict.fromkeys(name for item in span for name in item.writes))
        fragment = self.fragment
        fragment_action = (
            fragment
            if fragment is not None
            and (
                action.event <= fragment.source_event < stop
                or action.event == fragment.cache_event
            )
            else None
        )
        if (
            fragment_action is not None
            and action.event <= fragment_action.source_event < stop
            and not fragment_action.keep_shared
        ):
            writes = tuple(
                name for name in writes if name != fragment_action.source_name
            )
        omitted: tuple[str, ...] = ()
        scan_transfer = None
        island_publication = None
        if kind == "island":
            island_publication = next(
                (
                    item
                    for item in self._publication_receipts
                    if item.candidate.bound is proof
                ),
                None,
            )
            if (
                not isinstance(proof, BoundPreparationIsland)
                or not proof.matches(self.plan, frame, self.execution, dict(before))
                or (action.event, stop) != (proof.first_event, proof.stop_event)
                or (
                    island_publication is None
                    and any(after.get(item.node) != item.name for item in proof.exports)
                )
            ):
                self.fail("island completion proof changed")
            exports = {item.name for item in proof.exports}
            omitted = tuple(name for name in writes if name not in exports)
            if proof.island_input is not None:
                completion = proof.island_input.island_completion
                if (
                    completion is None
                    or completion.bound is not proof
                    or not completion.current()
                    or completion.body_first != body_first
                    or completion.lines != tuple(emitted)
                    or completion.outputs != tuple(after.items())
                ):
                    self.fail("published-input island lacks exact actual completion")
                # The earlier action really wrote this full native owner.
                # Replacing its later fill is not permission to omit storage.
                omitted = tuple(
                    name
                    for name in omitted
                    if name != proof.island_input.candidate.owner
                )
            writes = tuple(item.name for item in proof.exports)
            reads = self._reads(proof.island.entries, before)
            if island_publication is not None:
                if (
                    not island_publication.matches()
                    or tuple(after.items()) != island_publication.outputs
                ):
                    self.fail("island publication changed before recording")
                omitted = tuple(name for item in span for name in item.writes)
                writes = (island_publication.candidate.owner,)
                reads = tuple(sorted({*reads, *island_publication.candidate.reads}))
        elif kind == "scan":
            if (
                not isinstance(proof, ScanProducer)
                or proof is not self.pipeline.scan_producer
                or not proof.matches(
                    self.plan, self.pipeline, dict(self.revision.shapes)
                )
                or action.event != proof.first_event
                or stop != proof.stop_event + 1
            ):
                self.fail("scan completion proof changed")
            attempt = self.scan_attempt
            if (
                attempt is None
                or not attempt.completed
                or attempt.inputs != tuple(before.items())
            ):
                self.fail("scan transfer lacks complete emission")
            try:
                scan_transfer = attempt.result()
            except ValueError as error:
                self.fail(str(error))
            self.scan_attempt = None
            reads = tuple(sorted({name for item in span for name in item.reads}))
        elif kind == "frontier":
            if isinstance(proof, FrontierTransferReceipts):
                selected = proof.materializations
                if (
                    not proof.matches()
                    or selected.pipeline is not self.pipeline
                    or selected.revision.plan is not self.plan
                    or selected.group.first_event != action.event
                    or selected.group.stop_event != stop
                    or selected.boundaries != tuple(before.items())
                    or any(
                        self.special.get(crop.buffer.name) is not crop
                        for _, crop in selected.aliases
                    )
                ):
                    self.fail("materialized frontier completion changed")
                group = selected.group
                writes = tuple(
                    group.buffers[index].name for index in selected.emitted_ordinals
                )
                omitted = tuple(crop.buffer.name for _, crop in selected.aliases)
            elif (
                not isinstance(proof, FrontierGroup)
                or proof != plan_frontier_group(frame, action.event)
                or stop != proof.stop_event
                or not self.permits_frontier_group(action.event)
            ):
                self.fail("frontier completion span changed")
            else:
                group = proof
            reads = self._reads(
                tuple(item.node for item in group.buffers if item.node is not None),
                before,
            )
        elif kind == "raw":
            if (
                not isinstance(proof, RawImageBinding)
                or self.special.get(proof.transfer.buffer.name) is not proof
            ):
                self.fail("raw image choice changed")
            reads = self._reads((proof.transfer.source,), before)
        elif kind == "crop":
            if (
                not isinstance(proof, NativeIdentityCrop)
                or self.special.get(proof.buffer.name) is not proof
            ):
                self.fail("native crop choice changed")
            omitted = (proof.buffer.name,)
            writes = ()
            reads = (proof.source.owner,)
        else:
            if proof is not None:
                self.fail("ordinary action has replacement authority")
            roots = tuple(
                node
                for item in span
                for node in (
                    tuple(
                        source for dot in item.nodes for source in dot.all_input_nodes
                    )
                    if item.kind == "fill"
                    else item.nodes
                )
            )
            publication = next(
                (
                    item
                    for item in self._publication_receipts
                    if item.candidate.bound.stop_event == action.event
                ),
                None,
            )
            if publication is not None:
                roots = tuple(
                    node for node in roots if node is not publication.candidate.operand
                )
            if (
                fragment_action is not None
                and action.event == fragment_action.cache_event
            ):
                state = fragment_action._state
                if (
                    state.cache_inputs != tuple(before.items())
                    or state.cache_outputs != tuple(after.items())
                    or state.cache_lines != tuple(emitted)
                    or state.shared_reads is None
                ):
                    self.fail("fragment cache returned body or boundary changed")
                reads = state.shared_reads
            else:
                reads = self._reads(roots, before)
            if publication is not None:
                reads = tuple(sorted({*reads, publication.candidate.owner}))
            if action.kind == "mma":
                self.fail("standalone MMA has no completed stage")
            if action.kind == "fill":
                proof = self.completed_stages.pop(action.event, None)
                if (
                    proof is None
                    or proof.stop != stop
                    or proof.inputs != tuple(before.items())
                    or proof.reads != reads
                ):
                    self.fail("ordinary stage lacks actual completed emission")
        broadcasts = (
            () if self.broadcast_attempt is None else self.broadcast_attempt.result()
        )
        broadcast_body = (
            None
            if self.broadcast_attempt is None
            else self.broadcast_attempt.action_body(
                action.event,
                stop,
                before,
                after,
                emitted,
                body_first,
                postlude if kind == "scan" else broadcast_postlude,
            )
        )
        self._broadcasts = (*self._broadcasts, *broadcasts)
        self.broadcast_attempt = None
        completed_frontier = (
            frontier_completion.consume(
                self.cg,
                self.plan,
                self.execution,
                tuple(before.items()),
                tuple(after.items()),
                tuple(emitted),
            )
            if frontier_completion is not None
            else None
        )
        accepted_action = AcceptedPreparationAction(
            action.event,
            stop,
            kind,
            proof,
            tuple(before.items()),
            tuple(after.items()),
            reads,
            writes,
            omitted,
            scan_transfer,
            broadcasts,
            island_publication,
            broadcast_body,
            fragment_action,
        )
        for publication in self._publication_receipts:
            item = publication.candidate
            if action.event < item.bound.stop_event:
                continue
            if after.get(item.operand) != publication.boundary_name(stop) or before.get(
                item.operand
            ) != publication.boundary_name(action.event):
                self.fail("published island operand alias was replaced")
            # READY's closure is bookkeeping, not a memory-reading producer.
            # Preserve its original conservative charge without manufacturing
            # an expression-completion receipt for its barrier-only action.
            if item.owner in reads and action.kind != "ready":
                from .chained_island_publication import record_island_read

                self._publication_reads = (
                    *self._publication_reads,
                    record_island_read(
                        publication,
                        accepted_action,
                        body_first,
                        tuple(emitted),
                        frontier=completed_frontier,
                    ),
                )
        if self.leaf_capture is not None:
            self.leaf_capture.action(
                len(self.actions), accepted_action, emitted, postlude
            )
        self.actions.append(accepted_action)
        if island_publication is not None:
            self.read_frontier.islands.append(island_publication)
        self.cursor = stop

    def emit_leaf(
        self,
        leaf: PreparationLeaf,
        action: PreparationAction,
        boundaries: dict[Node, str],
        execution: ChainedExecution,
        barrier: str,
        phase: str,
    ) -> list[str]:
        if self.leaf_capture is None:
            return leaf.emit(self.cg, self.plan, boundaries, execution, barrier, phase)
        return self.leaf_capture.leaf(
            len(self.actions),
            action,
            leaf,
            self.cg,
            self.plan,
            boundaries,
            execution,
            barrier,
            phase,
        )

    def emit_image(
        self,
        action: PreparationAction,
        boundaries: dict[Node, str],
        vector: VectorStaging,
        unroll: BoundedProducerUnroll,
    ) -> list[str] | None:
        if action.kind != "frontier" or len(action.writes) != 1:
            return None
        proof = self.special.get(action.writes[0])
        if proof is None:
            return None
        before = dict(boundaries)
        if isinstance(proof, RawImageBinding):
            assert self.raw is not None
            lines = emit_raw_image(
                self.cg,
                self.plan,
                self.raw,
                proof,
                boundaries,
                self.execution,
                vector=vector,
                producer_unroll=unroll,
            )
            if lines is None:
                self.fail("selected raw image did not emit")
            self.record(
                action, before, boundaries, kind="raw", proof=proof, emitted=lines
            )
            return lines
        from .chained_matmul import _operand_domain
        from .chained_matmul import _UnsupportedChain

        assert isinstance(proof, NativeIdentityCrop)
        retained = self.pipeline.operand_retention
        if retained is None or proof.source not in retained.candidates:
            self.fail("native crop lacks original retained producer")
        index = retained.candidates.index(proof.source)
        if (
            boundaries.get(proof.source.node) != f"chain_retained_operand_{index}"
            or proof.source.publication_event > action.event
            or self.pipeline.frame.layout.region(proof.source.owner).live_until
            < len(self.pipeline.frame.actions)
        ):
            self.fail("native crop precedes completed whole-owner publication")
        try:
            if _operand_domain(
                self.cg,
                proof.source.node,
                ("chain_crop_row", "chain_crop_k"),
                self.plan,
            ):
                self.fail("native crop changes original operand domain")
        except _UnsupportedChain as error:
            raise exc.BackendUnsupported(
                "cute", "native crop domain is not proved"
            ) from error
        # The physical binder creates this same alias for the consumer. The
        # producer's full tensor already exists at its actual completion point.
        full = f"chain_{proof.source.group.stages[0]}_{proof.source.role}"
        lines = [*proof.alias_lines(full, proof.buffer.name), self.execution.sync]
        self.record(action, before, boundaries, kind="crop", proof=proof, emitted=lines)
        return lines

    def seal(
        self,
        lines: list[str],
        vector: VectorStaging,
        unroll: BoundedProducerUnroll,
        scratch: ScratchLayouts,
    ) -> AcceptedPreparation:
        if (
            self.sealed
            or self.cursor != len(self.pipeline.frame.actions)
            or self.completed_stages
            or self.scan_attempt is not None
            or self.broadcast_attempt is not None
        ):
            self.fail("incomplete or duplicate seal")
        if self.transport_facts != _transport_facts(self.pipeline):
            self.fail("preparation transports changed during body emission")
        if self._publication_receipts != tuple(
            action.island_publication
            for action in self.actions
            if action.island_publication is not None
        ):
            self.fail("successful island publication inventory changed before sealing")
        if self.island_consumers and not self._publication_receipts:
            self.fail("island consumer publication has no admitted original operand")
        if self._scan_receipts != tuple(
            action.scan_transfer
            for action in self.actions
            if action.scan_transfer is not None
        ):
            self.fail("successful scan transfer inventory changed before sealing")
        if self._broadcasts != tuple(
            item for action in self.actions for item in action.broadcasts
        ):
            self.fail("successful broadcast inventory changed before sealing")
        if self.broadcast_retention and not self._broadcasts:
            self.fail("broadcast retention has no admitted original subtree")
        seen = {
            action.proof.buffer.name
            if isinstance(action.proof, NativeIdentityCrop)
            else action.proof.transfer.buffer.name
            for action in self.actions
            if isinstance(action.proof, (NativeIdentityCrop, RawImageBinding))
        }
        seen.update(
            crop.buffer.name
            for action in self.actions
            if isinstance(action.proof, FrontierTransferReceipts)
            for _, crop in action.proof.materializations.aliases
        )
        if seen != self.special.keys():
            self.fail("a selected image was swallowed by another action")
        fields = (
            self.revision,
            self.pipeline,
            self.execution,
            self.raw,
            self.crops,
            tuple(self.actions),
            tuple(lines),
            tuple(self.workspaces),
            (
                vector.enabled,
                vector.activated,
                vector.group_enabled,
                vector.group_activated,
                vector.async_enabled,
                vector.async_activated,
            ),
            (unroll.factor, unroll.activated, unroll.eliminated),
            scratch.mode,
            self.transport_facts,
            self._scan_receipts,
            self.read_frontier,
        )
        context = (
            _broadcast_context(tuple(self.actions)) if self.broadcast_retention else ()
        )
        selection = (*fields, self._broadcasts, context) if context else fields
        if self._publication_receipts:
            selection = (
                *selection,
                self._publication_receipts,
                self._publication_reads,
            )
        accepted = AcceptedPreparation(
            *fields,
            selection,
            self._broadcasts,
            context,
            self._publication_receipts,
            self._publication_reads,
            () if self.fragment is None else (self.fragment,),
        )
        if accepted.fragments:
            object.__setattr__(accepted, "_selection", (*selection, accepted.fragments))
            if any(not item.accepted(accepted) for item in accepted.fragments):
                self.fail("fragment body lacks exact completion and cache publication")
        if any(
            action.island_publication is not None
            and not action.island_publication.accepted(accepted, action)
            for action in self.actions
        ):
            self.fail("island publication lacks its exact completed consumer")
        if any(
            not item.matches(accepted, action)
            for action in self.actions
            for item in action.broadcasts
        ):
            self.fail("broadcast publication or typed scope changed before sealing")
        if any(
            isinstance(action.proof, CompletedPreparationStage)
            and not action.proof.matches(accepted, action)
            for action in self.actions
        ):
            self.fail("completed stage segment changed before sealing")
        if any(
            action.scan_transfer is not None
            and not action.scan_transfer.matches(accepted, action)
            for action in self.actions
        ):
            self.fail("scan transfer segment changed before sealing")
        self.read_frontier.check(self._publication_receipts)
        self.sealed = True
        return accepted


def build_accepted_preparation(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    execution: ChainedExecution,
    vector: VectorStaging,
    unroll: BoundedProducerUnroll,
    scratch: ScratchLayouts,
    *,
    raw: BoundRawWidenings | None = None,
    crops: NativeCropPlan | None = None,
    leaf_capture: LeafEmissionCapture | None = None,
    broadcast_retention: bool = False,
    island_consumers: bool = False,
    fragment_epilogues: bool = False,
) -> AcceptedPreparation:
    """Build the actual body once on private publication/activation state.

    No caller is activated by default. The returned body must not be installed
    until its physical views and whole pipeline allocation have been sealed.
    """
    from .chained_matmul import _shape
    from .chained_preparation_pipeline import _prepare
    from .chained_prepared_values import bind_frame_buffers

    if type(island_consumers) is not bool:
        raise TypeError("island consumer selection must be an explicit bool")
    shapes = {
        node: _shape(node)
        for node in pipeline.frame.cut.region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    try:
        revision = _revision(plan, pipeline.frame, shapes)
    except _InvalidRetention as error:
        raise exc.BackendUnsupported(
            "cute", "accepted preparation revision is invalid"
        ) from error
    if raw is not None and (raw.frame is not pipeline.frame or not raw.matches(plan)):
        _PreparationRecorder.fail("raw transfer does not match the semantic frame")
    if crops is not None and (
        pipeline.operand_retention is not crops.retention
        or crops.retention.frame is not pipeline.frame
        or not crops.matches(plan, crops.retention.revision.frame, shapes)
    ):
        _PreparationRecorder.fail("crop owner extensions are not the semantic frame")
    local_vector, local_unroll = replace(vector), replace(unroll)
    local_scratch = replace(
        scratch, xor_buffers=set(scratch.xor_buffers), vector_sinks={}
    )
    # Establish the original layout/target categories before expression
    # emission, just as the ordinary caller does. These numeric view lines are
    # deliberately not installed; the physical binder owns the final preamble.
    bind_frame_buffers(
        pipeline.frame,
        "chain_frame",
        local_scratch,
        prepared_operands=pipeline.prepared_operands,
        prepared_groups=pipeline.prepared_groups,
        prepared_leaves=pipeline.prepared_leaves,
        scan_producer=pipeline.scan_producer,
    )
    recorder = _PreparationRecorder(
        cg,
        plan,
        pipeline,
        execution,
        revision,
        raw,
        crops,
        leaf_capture,
        broadcast_retention,
        island_consumers,
        fragment_epilogues,
    )
    lines = _prepare(
        cg,
        plan,
        pipeline,
        execution,
        local_vector,
        local_unroll,
        local_scratch,
        recorder=recorder,
    )
    return recorder.seal(lines, local_vector, local_unroll, local_scratch)


def extend_native_crop_owners(
    plan: ChainedMatmulPlan, pipeline: PreparationPipeline
) -> tuple[PreparationPipeline, NativeCropPlan | None]:
    """Rebind complete retained owners before choosing any crop omission.

    The provisional semantic frame remains fully charged. Only the subsequent
    accepted-action planner can omit successfully replaced publications.
    """
    from .chained_prepared_groups import prepared_group_candidates
    from .chained_prepared_image_transfers import discover_native_identity_crops
    from .chained_prepared_image_transfers import plan_native_identity_crops
    from .chained_prepared_operands import plan_prepared_operands

    retained = pipeline.operand_retention
    if retained is None:
        return pipeline, None
    original = retained.revision.frame
    shapes = dict(retained.revision.shapes)
    available = discover_native_identity_crops(plan, original, shapes)
    if available is None:
        _PreparationRecorder.fail("native crop discovery changed")
    selected = tuple(item for item in available if item.source in retained.candidates)
    if not selected:
        return pipeline, None
    crops = plan_native_identity_crops(
        plan,
        original,
        shapes,
        retained.candidates,
        selected,
        prepared_groups=retained.original_groups,
        reservations=retained.reservations,
        # This is a semantic-analysis ceiling, not a launch allocation. Every
        # original owner remains charged until actual choices are sealed.
        capacity_bytes=sum(region.byte_size for region in original.layout.regions),
    )
    if crops is None:
        _PreparationRecorder.fail("complete crop owner leases cannot be rebound")
    retained = crops.retention
    operands = plan_prepared_operands(plan, retained.frame, pipeline.recurrence)
    groups = prepared_group_candidates(plan, retained.frame, pipeline.recurrence)
    if (
        operands is None
        or groups is None
        or any(binding.candidate not in groups for binding in retained.prepared_groups)
    ):
        _PreparationRecorder.fail("native crop changed an admitted layout")
    accepted = {item.buffer.name for item in pipeline.prepared_operands}
    reads = dict(retained.leaf_reads)
    if any(not reads.get(leaf.node) for leaf in pipeline.prepared_leaves):
        _PreparationRecorder.fail("native crop erased a selected TMA leaf read")
    return (
        replace(
            pipeline,
            frame=retained.frame,
            prepared_operands=tuple(
                operand for operand in operands if operand.buffer.name in accepted
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
        ),
        crops,
    )
