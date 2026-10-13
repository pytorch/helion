"""Bounded original async copies within one independently published frame.

Enqueue evidence never grants shared visibility. Only an emitted wait settles
the covered groups, and READY requires every original group to be settled.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import cast

from ... import exc
from .chained_warp_stage import _snapshot

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_frontier_layout import PacketXorVectorSink
    from .chained_independent_publication import IndependentPublication
    from .chained_matmul import ChainedMatmulPlan
    from .chained_native_vector_sink import NativeVectorSink
    from .chained_preparation_frame import PreparationAction
    from .chained_preparation_frame import PreparationBuffer
    from .chained_preparation_reads import PreparationFrontierEvidence
    from .chained_rectangular_frontier import RectangularFrontier
    from .chained_vector_leaf import DenseVectorSink
    from .chained_vector_leaf import VectorLeafEmission
    from .chained_vector_leaf import VectorLeafPlan
    from .chained_vector_ownership import VectorOwnership


# PTX cp.async has at most eight committed groups per issuing thread. Empty
# guarded groups count too: accounting is static, never predicate-dependent.
MAX_ASYNC_GROUPS = 8
ASYNC_GROUP_LIMITS = (1, 2, 4, MAX_ASYNC_GROUPS)
COMMIT = "cute.arch.cp_async_commit_group()"
WAIT = "cute.arch.cp_async_wait_group(0)"


def _reject(reason: str) -> NoReturn:
    raise exc.BackendUnsupported("cute", f"async preparation groups {reason}")


@dataclass(frozen=True)
class EnqueuedAsyncCopy:
    owner: AsyncPreparationGroups
    ordinal: int
    buffer: PreparationBuffer
    execution: ChainedExecution
    inputs: tuple[tuple[Node, str], ...]
    ownership: VectorOwnership
    sink: DenseVectorSink | NativeVectorSink | PacketXorVectorSink
    leaf: VectorLeafPlan
    emission: VectorLeafEmission
    body: tuple[str, ...]
    prefix: tuple[str, ...]
    rectangle: RectangularFrontier | None
    _selection: object = field(repr=False)

    @property
    def lines(self) -> tuple[str, ...]:
        return (*self.prefix, *self.body, COMMIT)

    def facts(self) -> object:
        return _snapshot(
            (
                id(self.owner),
                self.ordinal,
                vars(self.buffer),
                vars(self.execution),
                self.inputs,
                self.ownership,
                self.sink,
                self.leaf,
                self.emission,
                self.body,
                self.prefix,
                self.rectangle,
                self.owner.publication.cg.device_function.config.config,
            )
        )

    def matches(self) -> bool:
        if self.rectangle is not None:
            self.rectangle.check()
        return (
            0 <= self.ordinal < len(self.owner.enqueued)
            and self.owner.enqueued[self.ordinal] is self
            and self.facts() == self._selection
        )


@dataclass(frozen=True)
class SettledAsyncCopies:
    """A wait at this lexical boundary settles exactly these earlier commits."""

    groups: tuple[EnqueuedAsyncCopy, ...]
    before: int
    lines: tuple[str, ...] = (WAIT,)


@dataclass
class AsyncCopyAttempt:
    owner: AsyncPreparationGroups
    buffer: PreparationBuffer
    inputs: tuple[tuple[Node, str], ...]
    enqueued: EnqueuedAsyncCopy | None = None

    def emit(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        node: Node,
        execution: ChainedExecution,
        boundaries: Mapping[Node, str],
        ownership: VectorOwnership,
        sink: DenseVectorSink | NativeVectorSink | PacketXorVectorSink,
        leaf: VectorLeafPlan,
        emission: VectorLeafEmission,
        lines: list[str],
        *,
        rectangle: RectangularFrontier | None = None,
    ) -> list[str]:
        """Called only after the original guarded vector emitter succeeds."""
        owner = self.owner
        publication = owner.publication
        role = publication.execution
        copy_role = replace(role, threads=1 << (role.threads.bit_length() - 1))
        if (
            owner.active is not self
            or self.enqueued is not None
            or cg is not publication.cg
            or plan is not publication.plan
            or node is not self.buffer.node
            or self.inputs != tuple(boundaries.items())
            or execution != copy_role
            or len(self.buffer.shape) != 2
            or not ownership.matches(
                cast("tuple[int, int]", self.buffer.shape), execution.threads
            )
            or sink.target != self.buffer.name
            or sink.shape != self.buffer.shape
            or leaf.dtype != sink.dtype
            or not emission.lines
            or not emission.vectorized
            or not lines
        ):
            _reject("lost an original copy owner")
        if rectangle is not None:
            rectangle.check()
            if (
                rectangle.node is not node
                or rectangle.execution != execution
                or rectangle.ownership != ownership
                or rectangle.sink != sink
                or rectangle.lines() != lines
            ):
                _reject("lost complete rectangle evidence")
        prefix = owner.settle() if len(owner.pending) == owner.limit else ()
        receipt = EnqueuedAsyncCopy(
            owner,
            len(owner.enqueued),
            self.buffer,
            execution,
            self.inputs,
            ownership,
            sink,
            leaf,
            emission,
            tuple(lines),
            prefix,
            rectangle,
            None,
        )
        object.__setattr__(receipt, "_selection", receipt.facts())
        self.enqueued = receipt
        owner.enqueued.append(receipt)
        owner.pending.append(receipt)
        if len(owner.pending) > owner.limit:
            _reject("exceeded the selected committed-group limit")
        return list(receipt.lines)


class AsyncPreparationGroups:
    """One finalized, independently owned preparation walk, never a loop cache."""

    def __init__(self, publication: IndependentPublication, limit: int) -> None:
        if type(limit) is not int or limit not in ASYNC_GROUP_LIMITS[1:]:
            _reject("require a nondefault limit of 2, 4 or 8")
        self.publication = publication
        self.limit = limit
        self._limit = limit
        self.active: AsyncCopyAttempt | None = None
        self.enqueued: list[EnqueuedAsyncCopy] = []
        self.pending: list[EnqueuedAsyncCopy] = []
        self.accepted: list[EnqueuedAsyncCopy] = []
        self.settled: list[SettledAsyncCopies] = []
        self.ready_lines: tuple[str, ...] | None = None
        self.finished = False

    def check(self) -> None:
        if (
            self.limit != self._limit
            or self.limit not in ASYNC_GROUP_LIMITS[1:]
            or self.publication.cg.device_function.config.config.get(
                "cute_chained_async_group_limit", 1
            )
            != self.limit
        ):
            _reject("changed the selected group limit")

    def begin(
        self, action: PreparationAction, before: Mapping[Node, str]
    ) -> AsyncCopyAttempt:
        publication = self.publication
        self.check()
        frame = publication.pipeline.frame
        if (
            self.active is not None
            or self.ready_lines is not None
            or self.finished
            or action.event != publication.cursor
            or action is not frame.actions[action.event]
            or action.kind != "frontier"
            or len(action.writes) != 1
        ):
            _reject("lost lexical frontier order")
        buffer = next(b for b in frame.buffers if b.name == action.writes[0])
        self.active = AsyncCopyAttempt(self, buffer, tuple(before.items()))
        return self.active

    def accept(self, completed: PreparationFrontierEvidence) -> None:
        from .chained_preparation_reads import EnqueuedPreparationFrontier

        active = self.active
        self.check()
        if active is None or self.ready_lines is not None or self.finished:
            _reject("have no active original frontier")
        if isinstance(completed, EnqueuedPreparationFrontier):
            receipt = completed.copy
            if (
                receipt is not active.enqueued
                or not receipt.matches()
                or receipt.ordinal != len(self.accepted)
                or completed.buffers != (active.buffer,)
                or completed.inputs != active.inputs
            ):
                _reject("received stale or reordered enqueue evidence")
            expected = list(receipt.lines)
            role = self.publication.execution
            if receipt.execution.threads != role.threads:
                from .chained_matmul import _indent

                expected = [
                    f"if {role.thread} < {receipt.execution.threads}:",
                    _indent(expected, 4),
                ]
            expected.append(role.sync)
            if completed.lines != tuple(expected):
                _reject("changed the original enqueued frontier")
            self.accepted.append(receipt)
        elif active.enqueued is not None:
            _reject("treated a pending copy as a completed frontier")
        self.active = None

    def settle(self) -> tuple[str, ...]:
        self.check()
        if (
            not self.pending
            or any(not group.matches() for group in self.pending)
            or any(group not in self.accepted for group in self.pending)
            or self.finished
        ):
            _reject("cannot settle unaccepted or stale copies")
        receipt = SettledAsyncCopies(tuple(self.pending), len(self.enqueued))
        self.settled.append(receipt)
        self.pending.clear()
        return receipt.lines

    def ready(self, action: PreparationAction) -> list[str]:
        self.check()
        if (
            self.active is not None
            or self.ready_lines is not None
            or self.finished
            or action is not self.publication.pipeline.frame.actions[-1]
            or action.event != self.publication.cursor
            or len(self.enqueued) < 2
        ):
            _reject("require at least two original copies and one final settlement")
        self.ready_lines = self.settle()
        return list(self.ready_lines)

    def finish(self) -> None:
        self.check()
        if (
            self.finished
            or self.active is not None
            or self.pending
            or self.ready_lines != (WAIT,)
            or self.accepted != self.enqueued
            or any(not group.matches() for group in self.enqueued)
        ):
            _reject("lost final copy readiness")
        cursor = 0
        for index, settlement in enumerate(self.settled):
            stop = cursor + len(settlement.groups)
            final = index == len(self.settled) - 1
            if (
                settlement.lines != (WAIT,)
                or stop > len(self.enqueued)
                or not final
                and stop >= len(self.enqueued)
                or not 1 <= len(settlement.groups) <= self.limit
                or not final
                and len(settlement.groups) != self.limit
                or settlement.groups != tuple(self.enqueued[cursor:stop])
                or settlement.before != stop
                or not final
                and self.enqueued[stop].prefix != settlement.lines
                or final
                and stop != len(self.enqueued)
            ):
                _reject("changed committed-group settlement order")
            cursor = stop
        if cursor != len(self.enqueued):
            _reject("left original groups unsettled")
        self.finished = True
