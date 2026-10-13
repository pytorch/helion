"""Overlap an independent resident-carry seed with an earlier native issue.

An early store is only enqueued. The original aggregate TMEM store fence and
role join, after every other final-group seed, remain its completion authority.
No allocation, MMA, original expression, or snapshot transport changes here.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from ...language import memory_ops
from . import chained_matmul as chain
from .chained_completed_store import _records
from .chained_loop_tmem_carry import plan_loop_tmem_carry
from .chained_pipeline_storage import _facts
from .chained_preparation_cut import _known_effects
from .chained_preparation_cut import _shape_input
from .warp_specialized_plan import packed_input_tmem_columns

if TYPE_CHECKING:
    from torch.fx import Node

    from .chained_contraction_groups import ContractionGroup
    from .chained_execution import ChainedExecution
    from .chained_loop_tmem_carry_transport import LoopTmemCarryTransport
    from .chained_recurrence_body import RecurrenceBody


KEY = "cute_chained_early_carry_seed"
_PHASE = "chain_iteration & 1"


@dataclass(frozen=True)
class CarrySnapshotReady:
    carry: LoopTmemCarryTransport
    execution: ChainedExecution
    lines: tuple[str, ...]
    facts: object


def emit_snapshot(
    carry: LoopTmemCarryTransport,
    execution: ChainedExecution,
    tile_columns: int,
) -> tuple[list[str], CarrySnapshotReady]:
    """The existing snapshot renderer and its completed participant join."""
    lines = (
        carry.snapshot(execution, tile_columns=tile_columns)
        if tile_columns
        else carry.snapshot(execution)
    )
    if not lines or lines[-1] != execution.sync:
        raise chain._UnsupportedChain("early carry requires completed snapshot")
    return lines, CarrySnapshotReady(
        carry, execution, tuple(lines), _records((carry, execution, tuple(lines)))
    )


def disjoint(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[1] <= right[0] or right[1] <= left[0]


def ready_accumulator(body: RecurrenceBody) -> bool:
    """Independent seed inputs must already cross this exact READY frame.

    Do not mistake a pure-looking global load, another contraction, or another
    carry for a ready value. Shape-only queries are the existing proven ones.
    The original expression binder still owns casts, indices and masks.
    """
    carry = body.carry
    assert carry is not None
    cut = body.pipeline.frame.cut
    ready = {
        node for node in body.pipeline.frame.frontier_order if node in body.boundaries
    }
    pending = [carry.candidate.accumulator]
    seen: set[Node] = set()
    forbidden = {
        *body.plan.dots,
        *cut.region.scans,
        *cut.region.reductions,
        *cut.region.stores,
        *(
            item.input
            for item in cut.carries
            if item.input is not carry.candidate.input
        ),
    }
    while pending:
        node = pending.pop()
        if node in seen or node is carry.candidate.input or node in ready:
            continue
        seen.add(node)
        if _shape_input(node):
            continue
        if (
            node in forbidden
            or node.target in (memory_ops.load, memory_ops.store)
            or not _known_effects(node)
        ):
            return False
        if node.op == "placeholder":
            if node not in cut.captures or isinstance(
                node.meta.get("val"), torch.Tensor
            ):
                return False
        else:
            pending.extend(node.all_input_nodes)
    return True


@dataclass(frozen=True)
class EarlyCarryCandidate:
    predecessor: ContractionGroup
    destination: ContractionGroup
    member: int
    seed_window: tuple[int, int]
    snapshot_window: tuple[int, int]
    working_windows: tuple[tuple[int, int], ...]
    ready: CarrySnapshotReady
    facts: object


def _facts_for(body: RecurrenceBody) -> object:
    assert body.plan.region is not None
    return (
        _facts(body.plan),
        tuple((node, tuple(node.users)) for node in body.plan.region.graph.nodes),
        _records((body.carry, body.selected_transports, body.pipeline.frame)),
    )


def bind_early_carry_seed(
    body: RecurrenceBody, snapshot: CarrySnapshotReady | None
) -> EarlyCarrySeed | None:
    """A bounded ordinary full-K protocol, after final storage selection."""
    carry, selections = body.carry, body.selected_transports
    config = body.codegen.device_function.config.config
    if (
        carry is None
        or snapshot is None
        or snapshot.carry is not carry
        or snapshot.execution is not body.execution
        or snapshot.facts != _records((carry, body.execution, snapshot.lines))
        or snapshot.lines[-1:] != (body.execution.sync,)
        or body.execution.threads != 128
        or body.execution.tmem != "chain_tptr"
        or body.storage is None
        or selections is not body.storage.stages
        or selections is None
        or len(selections) < 2
        or body.plan.warp_mma_stages
        or body.output_lease is not None
        or body.grouped_fragments
        or carry.snapshot_seed is not None
        or carry.candidate.residency is not None
        or body.seed_tiling is not None
        and body.seed_tiling.max_columns
        or any(
            config.get(key)
            for key in (
                "cute_chained_drain_tile_columns",
                "cute_chained_snapshot_tile_columns",
                "cute_chained_snapshot_seed",
                "cute_chained_vector_group",
            )
        )
    ):
        return None
    groups = tuple(item.group for item in selections)
    candidate = carry.candidate
    if (
        plan_loop_tmem_carry(body.plan, groups) != candidate
        or groups[-1] is not candidate.final_group
        or not ready_accumulator(body)
        or selections[-1].tmem_carry is not carry
        or selections[-1].tmem_accumulator != carry.arena
        or selections[-1].tmem_input is not None
        or selections[-1].residency is not None
        or selections[-1].tmem_output is not None
    ):
        return None
    seed = (
        carry.arena_offset + candidate.member_offset,
        carry.arena_offset + candidate.member_offset + candidate.geometry.physical[1],
    )
    snapshot_window = (
        carry.snapshot_offset,
        carry.snapshot_offset + packed_input_tmem_columns(candidate.snapshot_shape[1]),
    )
    if not disjoint(seed, snapshot_window):
        return None
    windows = []
    for selection in selections[:-1]:
        # Working C is the unchanged zero-based arena. Reject other producers
        # and residency protocols instead of inferring pointer-expression ranges.
        if (
            selection.tmem_accumulator is not None
            or selection.tmem_output is not None
            or selection.residency is not None
            or selection.tmem_carry is not None
        ):
            return None
        working = (0, selection.group.physical[1])
        if working[1] > carry.arena_offset or not disjoint(seed, working):
            return None
        windows.append(working)
        operand = selection.tmem_input
        expected = carry.operand(selection.group)
        if operand != expected:
            return None
        if operand is not None and (
            operand.base != "chain_tptr"
            or operand.column_offset != carry.snapshot_offset
            or operand.physical_shape != candidate.snapshot_shape
        ):
            return None
    proof = EarlyCarryCandidate(
        groups[0],
        groups[-1],
        candidate.member,
        seed,
        snapshot_window,
        tuple(windows),
        snapshot,
        _facts_for(body),
    )
    return EarlyCarrySeed(body, proof)


def _span(lines: list[str], expected: tuple[str, ...]) -> int:
    matches = [
        index
        for index in range(len(lines) - len(expected) + 1)
        if tuple(lines[index : index + len(expected)]) == expected
    ]
    if not expected or len(matches) != 1:
        raise chain._UnsupportedChain("early carry temporal source changed")
    return matches[0]


@dataclass(frozen=True)
class NativeCarryIssue:
    """Exact ordinary issue block, bound before its first emitted instruction."""

    owner: EarlyCarrySeed
    group: ContractionGroup
    stage: int
    prefix: str
    execution: ChainedExecution
    phase: str
    lines: tuple[str, ...]


@dataclass
class EarlyCarrySeed:
    body: RecurrenceBody
    candidate: EarlyCarryCandidate
    state: str = "bound"
    early_lines: tuple[str, ...] = ()
    late_lines: tuple[str, ...] = ()
    _binding: tuple[int, int] = field(init=False)
    pending_issue: NativeCarryIssue | None = None
    committed_issue: NativeCarryIssue | None = None
    issues: list[tuple[NativeCarryIssue, tuple[str, ...]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._binding = id(self.body), id(self.candidate)

    def check(self) -> None:
        if (
            self._binding != (id(self.body), id(self.candidate))
            or self.candidate.facts != _facts_for(self.body)
            or self.candidate.ready.facts
            != _records(
                (self.body.carry, self.body.execution, self.candidate.ready.lines)
            )
        ):
            raise chain._UnsupportedChain("early carry graph or transport changed")

    def check_stage(
        self, group: ContractionGroup | None, execution: ChainedExecution, phase: str
    ) -> None:
        self.check()
        if (
            self.body._pending is None
            or self.body._pending.group is not group
            or execution is not self.body.execution
            or phase != _PHASE
        ):
            raise chain._UnsupportedChain("early carry action or role changed")

    def hoisted_setup(self, group: ContractionGroup | None) -> bool:
        if group is not self.candidate.destination:
            return False
        if self.state != "enqueued":
            raise chain._UnsupportedChain("early carry missing prior enqueue")
        return True

    def before_issue(
        self,
        prefix: str,
        stage: int,
        phase: str,
        execution: ChainedExecution,
        lines: tuple[str, ...],
    ) -> NativeCarryIssue:
        """Only the original full-K emitter can authorize this issue interval."""
        action = self.body._pending
        if action is None:
            raise chain._UnsupportedChain("early carry issue lacks current action")
        self.check_stage(action.group, execution, phase)
        if (
            self.pending_issue is not None
            or self.committed_issue is not None
            or stage != action.group.stages[0]
            or prefix != f"chain_{stage}"
            or any(issue.group is action.group for issue, _ in self.issues)
            or not lines
        ):
            raise chain._UnsupportedChain("early carry native issue repeated or stale")
        if action.group is self.candidate.destination and self.state != "settled":
            raise chain._UnsupportedChain("early carry final issue before settlement")
        issue = NativeCarryIssue(
            self, action.group, stage, prefix, execution, phase, lines
        )
        self.pending_issue = issue
        return issue

    def after_commit(self, issue: NativeCarryIssue) -> list[str]:
        from .chained_tcgen_stage import _native_accumulator_setup
        from .chained_tcgen_stage import _native_mma_definition
        from .chained_tcgen_stage import _seed_member

        if (
            issue is not self.pending_issue
            or issue.owner is not self
            or self.committed_issue is not None
        ):
            raise chain._UnsupportedChain("early carry commit lacks owned native issue")
        self.check_stage(issue.group, issue.execution, issue.phase)
        self.committed_issue = issue
        if issue.group is not self.candidate.predecessor:
            return []
        execution = issue.execution
        if self.state != "bound":
            raise chain._UnsupportedChain("early carry repeated enqueue")
        body, destination = self.body, self.candidate.destination
        assert body.carry is not None and body.plan.region is not None
        selected = body.selected_transports
        assert selected is not None
        final = selected[-1]
        prefix = f"chain_{destination.stages[0]}"
        from ..compile_environment import CompileEnvironment

        dtype = CompileEnvironment.current().backend.dtype_str(
            body.plan.region.contractions[destination.stages[0]].operand_dtypes[0]
        )
        m, n, _ = destination.physical
        lines = [
            _native_mma_definition(
                prefix,
                dtype,
                m,
                n,
                "K",
                "K",
                "TMEM" if final.tmem_input is not None else "SMEM",
            ),
            *_native_accumulator_setup(prefix, m, n, body.carry.arena),
            *_seed_member(
                body.codegen,
                body.plan,
                body.boundaries,
                prefix,
                (m, n),
                self.candidate.member,
                body.carry.candidate.geometry,
                body.carry.candidate.member_offset,
                execution,
                tmem_carry=body.carry,
            ),
        ]
        self.early_lines = tuple(lines)
        self.state = "enqueued"
        return lines

    def complete_issue(self, issue: NativeCarryIssue, lines: list[str]) -> None:
        if issue is not self.pending_issue or issue is not self.committed_issue:
            raise chain._UnsupportedChain("early carry completed an unowned issue")
        expected = (
            *issue.lines,
            *(self.early_lines if issue.group is self.candidate.predecessor else ()),
            f"cute.arch.mbarrier_wait({issue.execution.barriers} + {issue.stage}, {issue.phase})",
        )
        if tuple(lines) != expected:
            raise chain._UnsupportedChain("early carry native issue source changed")
        self.issues.append((issue, expected))
        self.pending_issue = self.committed_issue = None

    def skip_member(self, member: int) -> bool:
        if member != self.candidate.member:
            return False
        self.check()
        if (
            self.state != "enqueued"
            or self.body._pending is None
            or self.body._pending.group is not self.candidate.destination
        ):
            raise chain._UnsupportedChain("early carry seed consumed out of order")
        self.state = "consumed"
        return True

    def settle(self, lines: list[str]) -> None:
        self.check()
        if self.state != "consumed" or lines[-2:] != [
            "cute.arch.fence_view_async_tmem_store()",
            self.body.execution.sync,
        ]:
            raise chain._UnsupportedChain("early carry lacks late aggregate fence")
        self.late_lines = tuple(lines)
        self.state = "settled"

    def verify_stage(self, group: ContractionGroup, lines: list[str]) -> None:
        self.check()
        owned = [(issue, span) for issue, span in self.issues if issue.group is group]
        if len(owned) != 1 or owned[0][0].owner is not self:
            raise chain._UnsupportedChain("early carry lacks completed native issue")
        issue, span = owned[0]
        beginning = _span(lines, span)
        if group is self.candidate.predecessor and self.state != "enqueued":
            raise chain._UnsupportedChain("early carry enqueue receipt changed")
        if group is self.candidate.destination:
            start = _span(lines, self.late_lines)
            if self.state != "settled" or start + len(self.late_lines) > beginning:
                raise chain._UnsupportedChain(
                    "early carry final issue precedes settlement"
                )

    def finish(self) -> None:
        if (
            self.state != "settled"
            or not self.early_lines
            or not self.late_lines
            or self.pending_issue is not None
            or self.committed_issue is not None
            or tuple(issue.group for issue, _ in self.issues)
            != tuple(stage.group for stage in self.body.recurrence.stages)
        ):
            raise chain._UnsupportedChain("incomplete early carry schedule")
