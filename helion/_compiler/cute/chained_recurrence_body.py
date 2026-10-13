"""Same-attempt traversal context for the original recurrence stage actions.

This owns emission order, not async readiness or storage selection. The common
body interpreter calls emit_stage directly; the original envelope owns stores,
carry transfers and READY/EMPTY. No expression is lowered during binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from ... import exc
from . import chained_matmul as chain
from .chained_completed_store import _physical_facts
from .chained_completed_store import _records
from .chained_pipeline_storage import StageTransports
from .chained_pipeline_storage import _facts
from .chained_root_stage import _execution_fields

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_body_program import BodyProgram
    from .chained_completed_store import CompletedCarryEndpoints
    from .chained_completed_store import CompletedStoreAction
    from .chained_early_carry_seed import CarrySnapshotReady
    from .chained_early_carry_seed import EarlyCarrySeed
    from .chained_execution import ChainedExecution
    from .chained_fragment_reuse import FragmentReuse
    from .chained_grouped_fragment import GroupedFragmentBridge
    from .chained_loop_tmem_carry_transport import LoopTmemCarryTransport
    from .chained_loop_tmem_transport import LoopTmemTransport
    from .chained_matmul import ChainedMatmulPlan
    from .chained_output_lease import BoundOutputLease
    from .chained_pipeline_storage import PipelineStorage
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_recurrence_workspace import RecurrenceStage
    from .chained_recurrence_workspace import RecurrenceWorkspace
    from .chained_seed_tiles import SeedTiling


@dataclass
class RecurrenceBody:
    codegen: GenerateAST
    plan: ChainedMatmulPlan
    pipeline: PreparationPipeline
    recurrence: RecurrenceWorkspace
    boundaries: dict[Node, str]
    execution: ChainedExecution
    selected_transports: tuple[StageTransports, ...] | None
    storage: PipelineStorage | None
    transports: tuple[LoopTmemTransport, ...]
    carry: LoopTmemCarryTransport | None
    seed_tiling: SeedTiling | None
    output_lease: BoundOutputLease | None
    carry_endpoints: CompletedCarryEndpoints | None
    completed_member_store: bool
    program: BodyProgram
    grouped_fragments: tuple[GroupedFragmentBridge, ...] = ()
    fragment_reuse: FragmentReuse | None = None
    early_carry: EarlyCarrySeed | None = None
    _context: object = None
    _maps: tuple[tuple[dict[str, str], tuple[tuple[str, str], ...]], ...] = ()
    _boundaries: tuple[tuple[Node, str], ...] = ()
    _completed_boundaries: tuple[tuple[Node, str], ...] | None = None
    _cursor: int = 0
    _pending: RecurrenceStage | None = None
    _finished: tuple[str, ...] | None = None
    _lines: list[str] = field(default_factory=list)
    completed_action: CompletedStoreAction | None = None
    _completed_lines: tuple[str, ...] = ()
    _completed: CompletedStoreAction | None = None

    @property
    def completed_lines(self) -> tuple[str, ...]:
        return self._completed_lines

    def _facts(
        self,
    ) -> tuple[object, tuple[tuple[dict[str, str], tuple[tuple[str, str], ...]], ...]]:
        aliases: dict[int, tuple[dict[str, str], tuple[tuple[str, str], ...]]] = {}
        lease = self.output_lease
        records = _records(
            (
                self.plan,
                self.pipeline,
                self.recurrence,
                self.selected_transports,
                self.transports,
                self.carry,
                self.grouped_fragments,
                self.fragment_reuse,
                None if self.early_carry is None else self.early_carry.candidate,
                # Its existing activation bit advances during seed emission;
                # only the selected geometry policy is immutable here.
                None if self.seed_tiling is None else self.seed_tiling.max_columns,
                # Only the existing completion ledger may advance.
                None
                if lease is None
                else (lease.proof, lease.sources, lease.setup, lease.copies),
            ),
            aliases=aliases,
        )
        physical = (
            None
            if self.storage is None
            else _physical_facts(self.pipeline, self.storage)
        )
        if physical is not None:
            aliases.update(
                (id(mapping), (mapping, pairs)) for mapping, pairs in physical[1]
            )
        context = (
            tuple(
                id(value)
                for value in (
                    self,
                    self.codegen,
                    self.plan,
                    self.pipeline,
                    self.recurrence,
                    self.boundaries,
                    self.plan.tensor_aliases,
                    self.execution,
                    self.selected_transports,
                    self.storage,
                    self.transports,
                    self.carry,
                    self.seed_tiling,
                    lease,
                    self.carry_endpoints,
                    self.program,
                    self.plan.prepared_widenings,
                )
            ),
            tuple(id(stage) for stage in self.recurrence.stages),
            tuple(id(stage.group) for stage in self.recurrence.stages),
            None
            if self.selected_transports is None
            else tuple((id(item), id(item.group)) for item in self.selected_transports),
            tuple(id(stage) for stage in self.program.actions),
            _facts(self.plan),
            records,
            None if physical is None else physical[0],
            _records(self.codegen.device_function.config.config),
            _execution_fields(self.execution),
            self.completed_member_store,
        )
        return context, tuple(aliases.values())

    def check(self, *, after_stage: bool = False, finished: bool = False) -> None:
        if self.fragment_reuse is not None:
            self.fragment_reuse.check(self.plan)
        if (
            self.completed_action is self._completed
            and self._completed is not None
            and not self._completed.matches(self.plan)
        ):
            raise chain._UnsupportedChain("completed store binding changed")
        context, maps = self._facts()
        if (
            context != self._context
            or self.plan.loop_workspace is not self.recurrence.layout
            or self.storage is None
            and self.recurrence is not self.pipeline.recurrence
            or self.storage is not None
            and (
                self.recurrence is not self.storage.recurrence
                or self.selected_transports is not self.storage.stages
            )
            or (self._finished is not None) is not finished
            or self.completed_action is not self._completed
            or len(maps) != len(self._maps)
            or any(
                old is not new
                or (
                    tuple(new.items())[: len(pairs)]
                    if after_stage
                    else tuple(new.items())
                )
                != pairs
                for (old, pairs), (new, _) in zip(self._maps, maps, strict=True)
            )
            or (
                tuple(self.boundaries.items())[: len(self._boundaries)]
                if after_stage
                else tuple(self.boundaries.items())
            )
            != self._boundaries
        ):
            raise chain._UnsupportedChain(
                "recurrence body context or publication changed"
            )

    def begin(self, action: RecurrenceStage) -> StageTransports | None:
        self.check()
        if (
            self._pending is not None
            or self._cursor >= len(self.recurrence.stages)
            or self.recurrence.stages[self._cursor] is not action
        ):
            raise chain._UnsupportedChain("recurrence action reordered or repeated")
        self._pending = action
        # Preserve the old lookup at this action, not a new global selector.
        return (
            None
            if self.selected_transports is None
            else next(
                item for item in self.selected_transports if item.group == action.group
            )
        )

    def unfinalized_selection(self, action: RecurrenceStage) -> StageTransports:
        """Original lazy argument order; no additional transport admission."""
        group, pipeline, carry, transports = (
            action.group,
            self.pipeline,
            self.carry,
            self.transports,
        )
        return StageTransports(
            group=group,
            residency=pipeline.residency,
            prepared_operand=next(
                (
                    operand
                    for operand in pipeline.prepared_operands
                    if operand.stage == group.stages[0]
                ),
                None,
            ),
            prepared_group=next(
                (
                    binding
                    for binding in pipeline.prepared_groups
                    if binding.candidate.group == group
                ),
                None,
            ),
            tmem_input=(None if carry is None else carry.operand(group))
            or next(
                (
                    item.operand
                    for item in transports
                    if item.slot.candidate.destination_group == group
                ),
                None,
            ),
            tmem_output=next(
                (
                    item
                    for item in transports
                    if item.slot.candidate.source_stage == group.stages[0]
                ),
                None,
            ),
            tmem_accumulator=None
            if carry is None
            else carry.accumulator(group.stages[0]),
            tmem_carry=carry
            if carry is not None and carry.candidate.final_group == group
            else None,
        )

    def prepare_store(
        self, stage: RecurrenceStage, selection: StageTransports | None
    ) -> None:
        if not self.completed_member_store or stage is not self.recurrence.stages[-1]:
            return
        from .chained_completed_members import plan_completed_member_store
        from .chained_completed_store import prepare_completed_store

        plan, pipeline, storage = self.plan, self.pipeline, self.storage
        group, loop = stage.group, plan.loop
        assert storage is not None and selection is not None and loop is not None
        shapes = {
            node: chain._shape(node)
            for node in loop.region.nodes
            if isinstance(node.meta.get("val"), torch.Tensor)
        }
        choices = [
            (index, candidate)
            for index, store in enumerate(loop.region.stores)
            for member in group.stages
            if (
                candidate := plan_completed_member_store(
                    plan, group, member, store, shapes
                )
            )
            is not None
        ]
        if len(choices) != 1:
            raise chain._UnsupportedChain(
                "completed member store lacks one exclusive result"
            )
        index, candidate = choices[0]
        self.completed_action = prepare_completed_store(
            plan,
            pipeline,
            storage,
            candidate,
            self.boundaries,
            self.execution,
            prefix=f"chain_store_{index}",
            phase="chain_iteration & 1",
            output_lease=self.output_lease,
            carry_endpoints=self.carry_endpoints,
        )
        if self.completed_action is None:
            raise chain._UnsupportedChain(
                "completed member store lacks final storage proof"
            )
        self._completed = self.completed_action

    def accept(
        self,
        stage: RecurrenceStage,
        lines: list[str],
        completion: tuple[int, str, tuple[str, ...]],
        aliases: tuple[tuple[str, str], ...],
    ) -> None:
        """Called only by the interpreter immediately after its direct stage call."""
        from .chained_body_program import _matches_span

        self.check(after_stage=True)
        if (
            self._pending is not stage
            or self.program.actions[self._cursor] is not stage
            or tuple(self.boundaries.items()) != self._completed_boundaries
            or not _matches_span(
                self.plan,
                completion,
                aliases,
                stage.group.stages[0],
                "recurrence",
                lines,
            )
        ):
            raise chain._UnsupportedChain("recurrence stage completion changed")
        if self.early_carry is not None:
            self.early_carry.verify_stage(stage.group, lines)
        self._lines.extend(lines)
        self._boundaries = tuple(self.boundaries.items())
        self._completed_boundaries = None
        self._maps = self._facts()[1]
        if self.completed_action is not None:
            self._completed_lines = tuple(lines)
        self._cursor += 1
        self._pending = None

    def finish(self, lines: list[str]) -> None:
        self.check()
        if (
            self._pending is not None
            or self._cursor != len(self.program.actions)
            or lines != self._lines
            or (self.completed_action is not None) != self.completed_member_store
        ):
            raise chain._UnsupportedChain("incomplete recurrence body")
        if self.early_carry is not None:
            self.early_carry.finish()
        self._finished = tuple(lines)

    def validate_return(self, lines: list[str]) -> None:
        self.check(finished=True)
        if tuple(lines) != self._finished or tuple(self._lines) != self._finished:
            raise chain._UnsupportedChain("recurrence returned source changed")


def bind_recurrence_body(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    recurrence: RecurrenceWorkspace,
    boundaries: dict[Node, str],
    execution: ChainedExecution,
    *,
    selected_transports: tuple[StageTransports, ...] | None,
    storage: PipelineStorage | None,
    transports: tuple[LoopTmemTransport, ...],
    carry: LoopTmemCarryTransport | None,
    seed_tiling: SeedTiling | None,
    output_lease: BoundOutputLease | None,
    carry_endpoints: CompletedCarryEndpoints | None,
    completed_member_store: bool,
    carry_snapshot: CarrySnapshotReady | None = None,
) -> RecurrenceBody:
    from .chained_body_program import BodyProgram

    if (
        not recurrence.stages
        or (storage is None) != (selected_transports is None)
        or plan.loop_workspace is not recurrence.layout
        or storage is None
        and recurrence is not pipeline.recurrence
        or (
            storage is not None
            and (
                recurrence is not storage.recurrence
                or selected_transports is not storage.stages
            )
        )
        or (
            selected_transports is not None
            and (
                len(selected_transports) != len(recurrence.stages)
                or any(
                    item.group != stage.group
                    for item, stage in zip(
                        selected_transports, recurrence.stages, strict=True
                    )
                )
            )
        )
    ):
        raise chain._UnsupportedChain("recurrence body lost final transport selection")
    body = RecurrenceBody(
        cg,
        plan,
        pipeline,
        recurrence,
        boundaries,
        execution,
        selected_transports,
        storage,
        transports,
        carry,
        seed_tiling,
        output_lease,
        carry_endpoints,
        completed_member_store,
        BodyProgram(tuple(recurrence.stages)),
    )
    if cg.device_function.config.config.get("cute_chained_grouped_fragment_bridge"):
        from .chained_grouped_fragment import bind_grouped_fragment_bridges

        config = cg.device_function.config.config
        if (
            storage is None
            or selected_transports is None
            or seed_tiling is not None
            and seed_tiling.max_columns
            or config.get("cute_chained_drain_tile_columns", 0)
            or config.get("cute_chained_vector_group", False)
            or output_lease is not None
            or completed_member_store
        ):
            raise exc.BackendUnsupported(
                "cute", "grouped fragment conflicts with stage policy"
            )
        body.grouped_fragments = bind_grouped_fragment_bridges(
            cg, plan, boundaries, selected_transports, execution
        )
        if not body.grouped_fragments:
            raise exc.BackendUnsupported(
                "cute", "no complete adjacent grouped fragment bridge"
            )
    if cg.device_function.config.config.get("cute_chained_fragment_reuse"):
        from .chained_fragment_reuse import bind_fragment_reuse

        config = cg.device_function.config.config
        if (
            storage is None
            or selected_transports is None
            or seed_tiling is not None
            and seed_tiling.max_columns
            or config.get("cute_chained_drain_tile_columns", 0)
            or config.get("cute_chained_vector_group", False)
            or output_lease is not None
            or body.grouped_fragments
        ):
            raise exc.BackendUnsupported(
                "cute", "fragment reuse conflicts with stage policy"
            )
        body.fragment_reuse = bind_fragment_reuse(
            cg, plan, boundaries, selected_transports, execution
        )
        if body.fragment_reuse is None:
            raise exc.BackendUnsupported("cute", "no complete fragment reuse path")
        if completed_member_store:
            if storage.completed_store is None:
                raise exc.BackendUnsupported(
                    "cute", "fragment terminal export lacks storage proof"
                )
            body.fragment_reuse.check_terminal(storage.completed_store.candidate)
    if cg.device_function.config.config.get("cute_chained_early_carry_seed"):
        from .chained_early_carry_seed import bind_early_carry_seed

        body.early_carry = bind_early_carry_seed(body, carry_snapshot)
        if body.early_carry is None:
            raise exc.BackendUnsupported("cute", "no independent early carry seed")
    body._context, body._maps = body._facts()
    body._boundaries = tuple(boundaries.items())
    return body
