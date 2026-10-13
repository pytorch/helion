"""Same-attempt broadcast emission and final physical read-scope binding.

The expression helper proves typed scalar equivalence. These records add the
original completed producer and its complete write interval; neither a request
nor a successful expression probe is permission to install a kernel body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import TypedDict

from .chained_broadcast_expressions import BroadcastExpressionEmission
from .chained_broadcast_expressions import emit_broadcast_expressions
from .chained_broadcast_expressions import plan_broadcast_expressions
from .chained_preparation_reads import resolved_preparation_reads
from .chained_preparation_transfers import BoundPreparationTransfers
from .chained_preparation_transfers import bind_preparation_transfer_span

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _Expression
    from .chained_preparation_actions import AcceptedPreparation
    from .chained_preparation_actions import AcceptedPreparationAction
    from .chained_preparation_storage import AcceptedPreparationStorage
    from .chained_vector_ownership import VectorOwnership


class BroadcastOptions(TypedDict, total=False):
    broadcast: BroadcastRetentionAttempt


def _emission_fields(emission: BroadcastExpressionEmission) -> tuple[object, ...]:
    return (
        emission.candidate._fields(),
        emission.before_steps,
        emission.per_step,
        emission.replacements,
        tuple(
            (read.node, read.coordinates, read.tensor, read.shape, read.dtype)
            for read in emission.reads
        ),
    )


@dataclass(frozen=True)
class BroadcastRetentionReceipt:
    first: int
    boundaries: tuple[tuple[Node, str], ...]
    execution: ChainedExecution
    emission: BroadcastExpressionEmission
    lines: tuple[str, ...]
    _selection: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        return (
            id(self),
            self.first,
            self.boundaries,
            tuple(vars(self.execution).items()),
            _emission_fields(self.emission),
            self.lines,
        )

    def matches(
        self, accepted: AcceptedPreparation, action: AcceptedPreparationAction
    ) -> bool:
        if (
            self._selection != self._fields()
            or self not in action.broadcasts
            or self.first != action.first
            or action.broadcast_body is None
            or not action.broadcast_body.matches(accepted, action)
            or not self.lines
            or self.execution.threads != accepted.execution.threads
            or self.execution.thread != accepted.execution.thread
            or self.execution.sync != accepted.execution.sync
            or not self.emission.candidate.matches(
                accepted.revision.plan, dict(self.boundaries)
            )
        ):
            return False
        published = dict(action.inputs)
        if action.kind == "scan":
            scan = accepted.pipeline.scan_producer
            if action.proof is not scan or scan is None:
                return False
            for item in scan.prelude:
                published.update(zip(item.nodes, item.writes, strict=True))
        elif action.kind not in ("ordinary", "frontier"):
            return False
        # The current scan prefix is an in-register version, never a stable
        # shared input. Other boundary entries need not be used by the probe.
        return all(
            published.get(read.node) == read.tensor
            and dict(self.boundaries).get(read.node) == read.tensor
            for read in self.emission.reads
        )


@dataclass(frozen=True)
class BroadcastPlacement:
    body: BroadcastBody
    first: int
    indent: int = 0

    def lines(self) -> tuple[str, ...]:
        from .chained_matmul import _indent

        return (
            self.body.lines
            if not self.indent
            else (_indent(self.body.lines, self.indent),)
        )


@dataclass(frozen=True)
class BroadcastBody:
    """Original producer or original enclosing emitter, sealed before return."""

    first: int
    stop: int
    lines: tuple[str, ...]
    receipts: tuple[BroadcastRetentionReceipt, ...]
    children: tuple[BroadcastPlacement, ...]
    _selection: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        return (
            id(self),
            self.first,
            self.stop,
            self.lines,
            tuple(id(receipt) for receipt in self.receipts),
            tuple((id(item.body), item.first, item.indent) for item in self.children),
        )

    def matches(self) -> bool:
        if (
            self._selection != self._fields()
            or not self.lines
            or len(self.receipts) != self.stop - self.first
            or any(receipt._selection != receipt._fields() for receipt in self.receipts)
        ):
            return False
        if not self.children:
            return len(self.receipts) == 1 and self.lines == self.receipts[0].lines
        children_receipts = tuple(
            receipt for child in self.children for receipt in child.body.receipts
        )
        if len(children_receipts) != len(self.receipts) or any(
            left is not right
            for left, right in zip(children_receipts, self.receipts, strict=True)
        ):
            return False
        cursor = self.first
        end = 0
        for item in self.children:
            if (
                not item.body.matches()
                or item.body.first != cursor
                or type(item.first) is not int
                or type(item.indent) is not int
                or item.first < end
                or item.indent < 0
                or self.lines[item.first : item.first + len(item.lines())]
                != item.lines()
            ):
                return False
            cursor = item.body.stop
            end = item.first + len(item.lines())
        return cursor == self.stop


@dataclass(frozen=True)
class BroadcastActionBody:
    body: BroadcastBody
    body_first: int
    first: int
    stop: int
    inputs: tuple[tuple[Node, str], ...]
    outputs: tuple[tuple[Node, str], ...]
    lines: tuple[str, ...]
    receipts: tuple[BroadcastRetentionReceipt, ...]
    _selection: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        return (
            id(self),
            id(self.body),
            self.body_first,
            self.first,
            self.stop,
            self.inputs,
            self.outputs,
            self.lines,
            tuple(id(receipt) for receipt in self.receipts),
        )

    def matches(
        self, accepted: AcceptedPreparation, action: AcceptedPreparationAction
    ) -> bool:
        return (
            self._selection == self._fields()
            and self.body.matches()
            and self.body.first == 0
            and self.body.stop == len(self.receipts)
            and len(self.body.receipts) == len(self.receipts)
            and all(
                left is right
                for left, right in zip(self.body.receipts, self.receipts, strict=True)
            )
            and self.first == action.first
            and self.stop == action.stop
            and self.inputs == action.inputs
            and self.outputs == action.outputs
            and len(self.receipts) == len(action.broadcasts)
            and all(
                left is right
                for left, right in zip(self.receipts, action.broadcasts, strict=True)
            )
            and self.lines[: len(self.body.lines)] == self.body.lines
            and accepted.lines[self.body_first : self.body_first + len(self.lines)]
            == self.lines
        )


class BroadcastRetentionAttempt:
    """Private per-original-action capture, with no lifetime assertion."""

    def __init__(self, first: int, execution: ChainedExecution) -> None:
        self.first = first
        self.execution = execution
        self.receipts: tuple[BroadcastRetentionReceipt, ...] = ()
        self._completed: tuple[BroadcastRetentionReceipt, ...] = ()
        self._selection = first, tuple(vars(execution).items())
        self._bodies: tuple[BroadcastBody, ...] = ()

    def emit(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        probes: Sequence[_Expression],
        ownership: VectorOwnership,
        *,
        tag: str,
        row: str,
        base: str,
        element: str,
        execution: ChainedExecution,
    ) -> BroadcastExpressionEmission | None:
        if (
            self._selection != (self.first, tuple(vars(self.execution).items()))
            or execution.threads != self.execution.threads
            or execution.thread != self.execution.thread
            or execution.sync != self.execution.sync
        ):
            return None
        candidate = plan_broadcast_expressions(
            probes, ownership, row=row, base=base, element=element
        )
        return (
            None
            if candidate is None
            else emit_broadcast_expressions(
                cg, plan, boundaries, candidate, tag=tag, execution=execution
            )
        )

    def complete(
        self,
        emission: BroadcastExpressionEmission,
        boundaries: Mapping[Node, str],
        execution: ChainedExecution,
        lines: Sequence[str],
    ) -> None:
        if (
            self._selection != (self.first, tuple(vars(self.execution).items()))
            or self.receipts != self._completed
            or execution.threads != self.execution.threads
            or execution.thread != self.execution.thread
            or execution.sync != self.execution.sync
            or not lines
            or not emission.candidate.matches(emission.candidate.plan, boundaries)
        ):
            raise ValueError("broadcast producer changed before completion")
        receipt = BroadcastRetentionReceipt(
            self.first, tuple(boundaries.items()), execution, emission, tuple(lines), ()
        )
        object.__setattr__(receipt, "_selection", receipt._fields())
        self.receipts = (*self.receipts, receipt)
        self._completed = self.receipts
        body = BroadcastBody(
            len(self.receipts) - 1,
            len(self.receipts),
            receipt.lines,
            (receipt,),
            (),
            (),
        )
        object.__setattr__(body, "_selection", body._fields())
        self._bodies = (*self._bodies, body)

    def checkpoint(self) -> int:
        receipts = self.result()
        if tuple(
            index for body in self._bodies for index in range(body.first, body.stop)
        ) != tuple(range(len(receipts))):
            raise ValueError("broadcast body inventory changed")
        body_receipts = tuple(
            receipt for body in self._bodies for receipt in body.receipts
        )
        if len(body_receipts) != len(receipts) or any(
            left is not right
            for left, right in zip(body_receipts, receipts, strict=True)
        ):
            raise ValueError("broadcast body belongs to another completion")
        return len(receipts)

    def place(
        self, first: int, lines: Sequence[str], offset: int, *, indent: int = 0
    ) -> BroadcastPlacement | None:
        """Validate a returned child at the original, explicit insertion site."""
        self.checkpoint()
        if first == len(self.receipts):
            return None
        bodies = tuple(body for body in self._bodies if body.first >= first)
        if (
            len(bodies) != 1
            or bodies[0].first != first
            or bodies[0].stop != len(self.receipts)
            or tuple(lines) != bodies[0].lines
            or type(offset) is not int
            or offset < 0
            or type(indent) is not int
            or indent < 0
        ):
            raise ValueError("broadcast returned producer or placement changed")
        return BroadcastPlacement(bodies[0], offset, indent)

    def enclose(
        self,
        first: int,
        lines: Sequence[str],
        placements: Sequence[BroadcastPlacement | None],
    ) -> None:
        """Seal the actual original assembly, before its enclosing emitter returns."""
        self.checkpoint()
        children = tuple(item for item in placements if item is not None)
        bodies = tuple(body for body in self._bodies if body.first >= first)
        if not bodies and not children:
            return
        if (
            not bodies
            or bodies[0].first != first
            or len(bodies) != len(children)
            or any(
                body is not item.body
                for body, item in zip(bodies, children, strict=True)
            )
        ):
            raise ValueError(
                "broadcast enclosing body dropped or duplicated a producer"
            )
        body = BroadcastBody(
            first, len(self.receipts), tuple(lines), self.receipts[first:], children, ()
        )
        object.__setattr__(body, "_selection", body._fields())
        if not body.matches():
            raise ValueError("broadcast structural placement changed")
        self._bodies = (*self._bodies[: -len(bodies)], body)

    def action_body(
        self,
        first: int,
        stop: int,
        before: Mapping[Node, str],
        after: Mapping[Node, str],
        lines: Sequence[str],
        body_first: int | None,
        postlude: Sequence[str],
    ) -> BroadcastActionBody | None:
        receipts = self.result()
        if not receipts:
            return None
        self.checkpoint()
        if (
            len(self._bodies) != 1
            or self._bodies[0].first != 0
            or self._bodies[0].stop != len(receipts)
            or type(body_first) is not int
            or body_first < 0
            or first != self.first
            or tuple(lines) != (*self._bodies[0].lines, *postlude)
        ):
            raise ValueError("broadcast original action body changed before acceptance")
        body = BroadcastActionBody(
            self._bodies[0],
            body_first,
            first,
            stop,
            tuple(before.items()),
            tuple(after.items()),
            tuple(lines),
            receipts,
            (),
        )
        object.__setattr__(body, "_selection", body._fields())
        return body

    def result(self) -> tuple[BroadcastRetentionReceipt, ...]:
        if (
            self._selection != (self.first, tuple(vars(self.execution).items()))
            or self.receipts != self._completed
            or any(item._selection != item._fields() for item in self.receipts)
            or any(not body.matches() for body in self._bodies)
        ):
            raise ValueError("broadcast completion inventory changed")
        return self.receipts


@dataclass(frozen=True)
class BoundBroadcastRetention:
    receipt: BroadcastRetentionReceipt
    transfer: BoundPreparationTransfers


def bind_broadcast_retention(
    physical: AcceptedPreparationStorage,
) -> tuple[BoundBroadcastRetention, ...] | None:
    """Bind actual read subtrees against every current full-owner write.

    Ordinary fills/frontiers read within their original first action. A fused
    scan reads after its complete prelude and before deferred reductions/MMA.
    Scheduled asynchronous leaf writes already extend the SAME physical table;
    their live intervals participate in the complete temporal interference test.
    No old nominal lease or old body receipt authorizes a scheduled read.
    """
    accepted = physical.accepted
    frame = accepted.pipeline.frame
    scan = accepted.pipeline.scan_producer
    phases = scan.phases if scan is not None else tuple(range(len(frame.actions)))
    result = []
    for action in accepted.actions:
        if not action.broadcasts:
            continue
        original = frame.actions[action.first]
        phase = phases[action.first]
        if action.kind == "scan":
            if scan is None or action.proof is not scan:
                return None
            fill = frame.actions[scan.stop_event - 1]
            if fill.kind != "fill" or phases[fill.event] != phase:
                return None
            writes = (*original.writes, *fill.writes)
        elif action.kind == "frontier":
            writes = action.writes
        elif action.kind == "ordinary" and original.kind in ("fill", "frontier"):
            writes = original.writes
        else:
            return None
        for receipt in action.broadcasts:
            if not receipt.matches(accepted, action):
                return None
            try:
                reads = resolved_preparation_reads(
                    accepted.pipeline,
                    tuple(read.node for read in receipt.emission.reads),
                    dict(receipt.boundaries),
                    use_frontier=accepted.read_frontier,
                    published=receipt.boundaries,
                )
            except ValueError:
                return None
            transfer = bind_preparation_transfer_span(
                physical, phase, phase + 1, reads, writes
            )
            if transfer is None or not transfer.sources:
                return None
            # Include every live physical owner, not a supplied destination or
            # omission list. Inactive aliases are allowed; simultaneous writes
            # (including pending TMA destinations) cannot alias a retained read.
            if any(
                source.name != region.name
                and region.live_from <= phase < region.live_until
                and source.overlaps_storage(region)
                for source in transfer.sources
                for region in physical.layout.regions
            ):
                return None
            result.append(BoundBroadcastRetention(receipt, transfer))
    return tuple(result)
