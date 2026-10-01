"""One original emission walk, followed by a separately sealed leaf schedule.

Original-order strings prove segment provenance only. The scheduled body has
its own pending/completed replay and is installed only through final storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

from .chained_leaf_schedule import LeafTransferEmission
from .chained_leaf_schedule import emit_leaf_transfer
from .chained_leaf_sets import LeafSetProtocol

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_leaf_schedule import LeafSchedule
    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_actions import AcceptedPreparation
    from .chained_preparation_actions import AcceptedPreparationAction
    from .chained_preparation_frame import PreparationAction
    from .chained_preparation_leaves import PreparationLeaf
    from .chained_preparation_pipeline import PreparationPipeline


@dataclass(frozen=True)
class PreparationSegment:
    key: tuple[int, int, str]
    inputs: tuple[tuple[Node, str], ...]
    outputs: tuple[tuple[Node, str], ...]
    lines: tuple[str, ...]
    leaf: PreparationLeaf | None = None
    transfer: LeafTransferEmission | None = None
    protocol: tuple[str, str] | None = None


@dataclass(frozen=True)
class RecordedPreparationSegments:
    original: AcceptedPreparation
    pieces: tuple[PreparationSegment, ...]
    _selection: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        # Store values, not a reference to the mutable contents of a frozen
        # dataclass; same-object descriptor/segment mutation must also fail.
        return (
            self.original,
            tuple(
                (
                    piece.key,
                    piece.inputs,
                    piece.outputs,
                    piece.lines,
                    piece.leaf,
                    None
                    if piece.transfer is None
                    else (piece.transfer.issue, piece.transfer.completion),
                    piece.protocol,
                )
                for piece in self.pieces
            ),
        )

    def matches(self, plan: ChainedMatmulPlan, pipeline: PreparationPipeline) -> bool:
        if not (
            self._selection == self._fields()
            and self.original.matches(
                plan, pipeline, dict(self.original.revision.shapes)
            )
            and len({piece.key for piece in self.pieces}) == len(self.pieces)
            and tuple(line for piece in self.pieces for line in piece.lines)
            == self.original.lines
        ):
            return False
        for index, action in enumerate(self.original.actions):
            pieces = tuple(piece for piece in self.pieces if piece.key[0] == index)
            if (
                not pieces
                or pieces[0].inputs != action.inputs
                or pieces[-1].outputs != action.outputs
            ):
                return False
            if any(left.outputs != right.inputs for left, right in pairwise(pieces)):
                return False
        return True


class LeafEmissionCapture:
    """Private subsegment capture driven by the actual preparation recorder."""

    def __init__(self) -> None:
        self.pieces: list[PreparationSegment] = []
        self.pending: list[PreparationSegment] = []
        self.core: PreparationSegment | None = None
        self.sealed = False

    def leaf(
        self,
        index: int,
        action: PreparationAction,
        leaf: PreparationLeaf,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: Mapping[Node, str],
        execution: ChainedExecution,
        barrier: str,
        phase: str,
    ) -> list[str]:
        if self.sealed or action.kind != "leaf" or action.nodes != (leaf.node,):
            raise ValueError("leaf segment is not an original input action")
        transfer = emit_leaf_transfer(
            leaf, cg, plan, dict(boundaries), execution, barrier, phase
        )
        after = dict(boundaries)
        after[leaf.node] = leaf.name
        self.pending.append(
            PreparationSegment(
                (index, action.event, "leaf"),
                tuple(boundaries.items()),
                tuple(after.items()),
                tuple(transfer.lines()),
                leaf,
                transfer,
                (barrier, phase),
            )
        )
        return transfer.lines()

    def prelude(
        self,
        index: int,
        action: PreparationAction,
        before: Mapping[Node, str],
        after: Mapping[Node, str],
        lines: Sequence[str],
    ) -> None:
        if self.sealed or action.kind != "collective":
            raise ValueError("unsupported scan prelude segment")
        self.pending.append(
            PreparationSegment(
                (index, action.event, "action"),
                tuple(before.items()),
                tuple(after.items()),
                tuple(lines),
            )
        )

    def scan_core(
        self,
        index: int,
        event: int,
        before: Mapping[Node, str],
        after: Mapping[Node, str],
        lines: Sequence[str],
    ) -> None:
        if self.sealed or self.core is not None:
            raise ValueError("duplicate scan core segment")
        self.core = PreparationSegment(
            (index, event, "scan_body"),
            tuple(before.items()),
            tuple(after.items()),
            tuple(lines),
        )

    def action(
        self,
        index: int,
        action: AcceptedPreparationAction,
        emitted: Sequence[str],
        postlude: Sequence[str],
    ) -> None:
        if self.sealed:
            raise ValueError("segments already sealed")
        pieces = self.pending
        if action.kind == "scan":
            core = self.core
            if core is None or core.key != (index, action.first, "scan_body"):
                raise ValueError("scan lacks its original complete core")
            pieces = [
                *pieces,
                PreparationSegment(
                    core.key, core.inputs, action.outputs, (*core.lines, *postlude)
                ),
            ]
        elif pieces:
            if len(pieces) != 1 or pieces[0].key != (index, action.first, "leaf"):
                raise ValueError("unconsumed leaf/prelude segment")
        else:
            if self.core is not None or postlude:
                raise ValueError("unexpected macro fragment")
            pieces = [
                PreparationSegment(
                    (index, action.first, "action"),
                    action.inputs,
                    action.outputs,
                    tuple(emitted),
                )
            ]
        if tuple(line for piece in pieces for line in piece.lines) != tuple(emitted):
            raise ValueError("original emitted subsegments do not reassemble")
        self.pieces.extend(pieces)
        self.pending = []
        self.core = None

    def seal(self, original: AcceptedPreparation) -> RecordedPreparationSegments:
        if self.sealed or self.pending or self.core is not None:
            raise ValueError("incomplete or repeated segment seal")
        pieces = tuple(self.pieces)
        result = RecordedPreparationSegments(original, pieces, ())
        result = RecordedPreparationSegments(original, pieces, result._fields())
        if not result.matches(original.revision.plan, original.pipeline):
            raise ValueError("original segment provenance changed")
        self.sealed = True
        return result


@dataclass(frozen=True)
class AcceptedLeafSchedule:
    recorded: RecordedPreparationSegments
    schedule: LeafSchedule
    lines: tuple[str, ...]
    _selection: tuple[object, ...]

    def matches(self, plan: ChainedMatmulPlan, pipeline: PreparationPipeline) -> bool:
        if self._selection != (self.recorded, self.schedule, self.lines):
            return False
        fresh = seal_leaf_schedule(plan, pipeline, self.recorded, self.schedule)
        return fresh is not None and fresh.lines == self.lines


def seal_leaf_schedule(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    recorded: RecordedPreparationSegments,
    schedule: LeafSchedule,
) -> AcceptedLeafSchedule | None:
    """Replay actual reads; an incidental old before-map is not readiness."""
    if (
        not recorded.matches(plan, pipeline)
        or schedule.physical.accepted is not recorded.original
        or not schedule.matches(plan, pipeline)
        or schedule.max_pending < 2
    ):
        return None
    pieces = {piece.key: piece for piece in recorded.pieces}
    expected = tuple(
        (action.accepted_index, action.event, action.kind)
        for action in schedule.actions
    )
    if tuple(piece.key for piece in recorded.pieces) != expected:
        return None
    protocol = LeafSetProtocol(
        pipeline.slots, len(pipeline.prepared_leaves), pipeline.cohorts is not None
    )
    leaf_pieces = []
    for selected in schedule.leaves:
        action = schedule.actions[selected.original_action]
        piece = pieces[(action.accepted_index, action.event, "leaf")]
        if (
            piece.leaf is not selected.leaf
            or piece.transfer is None
            or piece.lines != tuple(piece.transfer.lines())
            or piece.protocol
            != (protocol.leaf_pointer(selected.ordinal), protocol.phase)
            or action.reads
        ):
            return None
        leaf_pieces.append(piece)
    retained = pipeline.operand_retention
    aliases = (
        {
            f"chain_retained_operand_{index}": item.owner
            for index, item in enumerate(retained.candidates)
        }
        if retained is not None
        else {}
    )
    completed: set[str] = set()
    pending: set[int] = set()
    issued: set[int] = set()
    bindings: dict[Node, str] = {}
    lines: list[str] = []
    for step in schedule.steps:
        if step.kind in ("issue", "complete"):
            piece = leaf_pieces[step.index]
            assert piece.transfer is not None and piece.leaf is not None
            if step.kind == "issue":
                if step.index in issued:
                    return None
                issued.add(step.index)
                pending.add(step.index)
                lines.extend(piece.transfer.issue)
            else:
                if step.index not in pending:
                    return None
                pending.remove(step.index)
                completed.add(piece.leaf.name)
                bindings[piece.leaf.node] = piece.leaf.name
                lines.extend(piece.transfer.completion)
            continue
        action = schedule.actions[step.index]
        piece = pieces[(action.accepted_index, action.event, action.kind)]
        before, after = dict(piece.inputs), dict(piece.outputs)
        if not set(action.reads) <= completed or any(
            aliases.get(name, name) in action.reads and bindings.get(node) != name
            for node, name in before.items()
        ):
            return None
        if pipeline.frame.actions[action.event].kind == "ready" and pending:
            return None
        lines.extend(piece.lines)
        completed.update(action.writes)
        for node in before.keys() - after.keys():
            bindings.pop(node, None)
        for node, name in after.items():
            if before.get(node) != name:
                bindings[node] = name
                completed.add(aliases.get(name, name))
    if (
        pending
        or len(issued) != len(schedule.leaves)
        or bindings != dict(recorded.original.actions[-1].outputs)
    ):
        return None
    result = tuple(lines)
    return AcceptedLeafSchedule(
        recorded, schedule, result, (recorded, schedule, result)
    )
