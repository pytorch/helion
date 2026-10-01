"""Same-attempt original stores consuming a completed common-stage member.

The concrete stage owns issue/load/join completion. This adapter neither emits
those operations nor treats an allocated C view as a published boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from dataclasses import replace
import math
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from . import chained_matmul as chain
from .chained_completed_members import plan_completed_member_map
from .chained_pipeline_storage import _accepted_transports
from .chained_pipeline_storage import _freeze
from .chained_store_expression import UnboundStoreTarget
from .chained_store_expression import lower_store_point
from .chained_store_expression import materialized_store_read
from .contraction_region import _domain

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..generate_ast import GenerateAST
    from .chained_completed_members import CompletedMemberCandidate
    from .chained_execution import ChainedExecution
    from .chained_loop_tmem_carry_transport import LoopTmemCarryTransport
    from .chained_matmul import ChainedMatmulPlan
    from .chained_output_lease import BoundOutputLease
    from .chained_pipeline_storage import CarryStorageView
    from .chained_pipeline_storage import PipelineStorage
    from .chained_pipeline_storage import StageTransports
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_preparation_storage import BoundPreparationStorage
    from .chained_store_expression import StorePoint
    from .warp_specialized_plan import SharedBufferRegion


def _records(
    value: object,
    seen: set[int] | None = None,
    aliases: dict[int, tuple[dict[str, str], tuple[tuple[str, str], ...]]]
    | None = None,
) -> object:
    """Snapshot nested physical records, not references to mutable leaf dicts."""
    if isinstance(value, Node):
        return value
    if isinstance(value, torch.Tensor):
        return value.dtype, _domain(value), _freeze(value.stride())
    if seen is None:
        seen = set()
    if id(value) in seen:
        return type(value), id(value)
    seen.add(id(value))
    if isinstance(value, chain.ChainedMatmulPlan) and aliases is not None:
        mapping = value.tensor_aliases
        aliases[id(mapping)] = (mapping, tuple(mapping.items()))
        return type(value), tuple(
            (field.name, _records(getattr(value, field.name), seen, aliases))
            for field in fields(value)
            if field.name != "tensor_aliases"
        )
    if is_dataclass(value) and not isinstance(value, type):
        return type(value), tuple(
            (field.name, _records(getattr(value, field.name), seen, aliases))
            for field in fields(value)
        )
    if isinstance(value, dict):
        return dict, tuple(
            (_records(key, seen, aliases), _records(item, seen, aliases))
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_records(item, seen, aliases) for item in value)
    if isinstance(value, (set, frozenset)):
        return type(value), frozenset(_records(item, seen, aliases) for item in value)
    return _freeze(value)


def _physical_facts(
    pipeline: PreparationPipeline, storage: PipelineStorage
) -> tuple[object, tuple[tuple[dict[str, str], tuple[tuple[str, str], ...]], ...]]:
    # Consumption/completion ledgers legitimately advance. All layouts, views,
    # original actions and selected transfers remain independently captured.
    preparation = storage.preparation
    lease = storage.output_lease
    aliases: dict[int, tuple[dict[str, str], tuple[tuple[str, str], ...]]] = {}
    records = _records(
        (
            pipeline.frame,
            pipeline.recurrence,
            pipeline.slots,
            pipeline.recurrence_threads,
            pipeline.preparation_threads,
            pipeline.prepared_operands,
            pipeline.prepared_groups,
            pipeline.prepared_leaves,
            pipeline.scan_producer,
            storage.recurrence,
            storage.carry_views,
            storage.allocations,
            storage.omitted_results,
            storage.stages,
            None if storage.completed_store is None else id(storage.completed_store),
            # The original lease revision deliberately permits new host output
            # aliases registered by store lowering. Preserve its explicit
            # captured aliases/facts, not the plan's growing alias dictionary.
            None
            if lease is None
            else (
                lease.revision.facts,
                lease.revision.aliases,
                lease.revision.shapes,
                lease.stages,
                lease.final_group,
                lease.roots,
                lease.images,
                lease.result_reads,
                lease.carry_reads,
                lease.resident_carries,
                lease.layout,
            ),
            None if preparation is None else preparation.physical,
        ),
        aliases=aliases,
    )
    return records, tuple(aliases.values())


def _physical_matches(
    expected: tuple[
        object, tuple[tuple[dict[str, str], tuple[tuple[str, str], ...]], ...]
    ],
    pipeline: PreparationPipeline,
    storage: PipelineStorage,
) -> bool:
    current = _physical_facts(pipeline, storage)
    # Original lowering may register new host output aliases. Every previously
    # captured mapping object and key/value must remain exact; no existing alias
    # may disappear or change. Nested semantic revisions share this rule.
    return (
        expected[0] == current[0]
        and len(expected[1]) == len(current[1])
        and all(
            old is new and all(old.get(name) == alias for name, alias in pairs)
            for (old, pairs), (new, _) in zip(expected[1], current[1], strict=True)
        )
    )


@dataclass(frozen=True)
class _Prepared:
    point: StorePoint
    boundaries: tuple[tuple[Node, str], ...]
    reads: tuple[tuple[Node, str], ...]
    lines: tuple[str, ...] | None


def _prepare_point(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    candidate: CompletedMemberCandidate,
    epilogue: Mapping[Node, str],
    c_name: str,
    prefix: str,
    *,
    defer_target: bool = False,
) -> _Prepared:
    """Lower the original store once; the owning action chooses its placement."""
    stage_prefix = f"chain_{candidate.group.stages[0]}"
    index = f"{prefix}_register"
    n = candidate.native.logical_shape[1]
    coords = (f"{prefix} // {n}", f"{prefix} % {n}")
    trial = dict(epilogue)
    trial[candidate.node] = c_name
    replacement = materialized_store_read(
        plan, trial, candidate.node, coords, f"{stage_prefix}_values[{index}]"
    )
    point = lower_store_point(
        cg,
        plan,
        candidate.store,
        trial,
        [],
        [],
        coordinates=coords,
        coordinate_names=(prefix,),
        completed_read=replacement,
        defer_target=defer_target,
    )
    reads = tuple(
        dict.fromkeys(
            (node, epilogue[node])
            for node, _ in point.reads
            if node is not candidate.node
        )
    )
    lines = None if defer_target else _point_lines(candidate, point, prefix)
    return _Prepared(point, tuple(epilogue.items()), reads, lines)


def _point_lines(
    candidate: CompletedMemberCandidate, point: StorePoint, prefix: str
) -> tuple[str, ...]:
    if not isinstance(point.target, str):
        raise chain._UnsupportedChain("completed store target is not bound")
    stage_prefix = f"chain_{candidate.group.stages[0]}"
    index = f"{prefix}_register"
    row, column = f"{prefix}_row", f"{prefix}_column"
    m, n = candidate.native.logical_shape
    offset, width = candidate.native.offset, candidate.native.width
    geometry = candidate.group.geometries[
        candidate.group.stages.index(candidate.member)
    ]
    logical = geometry.result_coordinates(row, f"({column} - {offset})")
    return (
        f"for {index} in cutlass.range_constexpr(cute.size({stage_prefix}_values)):",
        f"    {row}, {column} = {stage_prefix}_coords[{index}]",
        f"    if ({column} >= {offset}) & ({column} < {offset + width}) & ({logical[0]} < {m}) & ({logical[1]} < {n}):",
        f"        {prefix} = {logical[0]} * {n} + {logical[1]}",
        chain._indent(point.lines, 8),
        f"        if {' and '.join(point.bounds)}:",
        f"            {point.target}[{', '.join(point.indices)}] = {point.value}",
    )


def _metadata_region(
    pipeline: PreparationPipeline,
    preparation: BoundPreparationStorage | None,
    node: Node,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> SharedBufferRegion | None:
    """Static original-image ownership; no completion or endpoint permission."""
    frame = pipeline.frame
    buffers = tuple(
        buffer
        for buffer in frame.buffers
        if buffer.node is node and buffer.name == name and buffer.kind == "frontier"
    )
    if len(buffers) != 1:
        return None
    buffer = buffers[0]
    native = {item.buffer.name for item in pipeline.prepared_operands} | {
        member.buffer.name
        for binding in pipeline.prepared_groups
        for member in binding.candidate.members
    }
    if name in native:
        return None
    region = frame.layout.region(name)
    if region.live_until != frame.actions[-1].publication_event:
        return None
    if preparation is not None:
        physical = preparation.physical
        views = tuple(
            view for view in physical.views if view.original.semantic is buffer
        )
        if len(views) != 1 or views[0].crop is not None:
            return None
        view = views[0]
        if (
            view.original.dtype != buffer.dtype
            or view.original.stored_node is not node
            or view.original.shape != buffer.shape
            or view.owner != name
        ):
            return None
        region = physical.layout.region(view.owner)
        scan = pipeline.scan_producer
        ready = frame.actions[-1]
        ready_end = (
            ready.publication_event if scan is None else scan.phases[ready.event] + 1
        )
        if region.live_until < ready_end:
            return None
    if (
        region.byte_size < math.prod(buffer.shape) * buffer.dtype.itemsize
        or node.meta["val"].dtype != buffer.dtype
        or (chain._shape(node) if shape is None else shape) != buffer.shape
    ):
        return None
    return region


def _endpoint_view(
    storage: PipelineStorage,
    carry: LoopTmemCarryTransport,
    view: CarryStorageView,
    shape: tuple[int, ...] | None = None,
) -> bool:
    """The endpoint-only view; actual upload/drain remain separate obligations."""
    return (
        view in storage.carry_views
        and view.pool == "frames"
        and view.carry == carry.candidate.carry
        and view.dtype == torch.float32
        and view.shape == (chain._shape(view.carry.input) if shape is None else shape)
        and view.carry.output in storage.omitted_results
        and view.byte_offset >= 0
        and view.byte_offset + math.prod(view.shape) * view.dtype.itemsize
        <= dict(storage.allocations)["frames"]
    )


@dataclass(frozen=True)
class CompletedStorePlan:
    """One original lowered store, not yet completed or installed in the body."""

    plan: ChainedMatmulPlan
    pipeline: PreparationPipeline
    stages: tuple[StageTransports, ...]
    candidate: CompletedMemberCandidate
    execution: ChainedExecution
    prefix: str
    c_name: str
    request: SharedBufferRegion
    prepared: _Prepared
    preparation: BoundPreparationStorage | None
    carry: LoopTmemCarryTransport | None
    _facts: object
    _aliases: tuple[tuple[dict[str, str], tuple[tuple[str, str], ...]], ...]
    _selection: tuple[object, ...] = ()

    def _fields(self) -> tuple[object, ...]:
        return (
            self.plan,
            self.pipeline,
            self.stages,
            self.candidate,
            self.execution,
            self.prefix,
            self.c_name,
            self.request,
            self.prepared,
            None
            if self.preparation is None
            else (id(self.preparation), self.preparation.physical),
            self.carry,
        )

    def matches(self) -> bool:
        aliases: dict[int, tuple[dict[str, str], tuple[tuple[str, str], ...]]] = {}
        facts = _records(self._fields(), aliases=aliases)
        current_aliases = tuple(aliases.values())
        return (
            self._selection == (self._fields(), self._facts, self._aliases)
            and self._facts is self._selection[-2]
            and self._aliases is self._selection[-1]
            and self._facts == facts
            and len(self._aliases) == len(current_aliases)
            and all(
                old is new and all(old.get(k) == v for k, v in pairs)
                for (old, pairs), (new, _) in zip(
                    self._aliases, current_aliases, strict=True
                )
            )
            and self.candidate.matches(self.candidate.plan, dict(self.candidate.shapes))
            and _accepted_transports(self.plan, self.pipeline, self.stages)
            and (
                self.preparation is None
                or self.preparation.matches(self.plan, self.pipeline)
            )
        )

    def matches_storage(self, storage: PipelineStorage) -> bool:
        if (
            not self.matches()
            or storage.completed_store is not self
            or storage.stages is not self.stages
            or storage.preparation is not self.preparation
            or storage.output_lease is not None
            or self.candidate.node not in storage.omitted_results
            or self.candidate.node in dict(storage.recurrence.bindings)
            or any(
                region.name == self.c_name
                for region in storage.recurrence.layout.regions
            )
            or tuple(
                item.tmem_carry for item in self.stages if item.tmem_carry is not None
            )
            != (() if self.carry is None else (self.carry,))
        ):
            return False
        shapes = dict(self.candidate.shapes)
        for node, name in self.prepared.reads:
            region = _metadata_region(
                self.pipeline, self.preparation, node, name, shapes[node]
            )
            if region is None or any(
                view.pool == "frames"
                and max(region.byte_offset, view.byte_offset)
                < min(
                    region.byte_end,
                    view.byte_offset + math.prod(view.shape) * view.dtype.itemsize,
                )
                and not (
                    self.carry is not None
                    and _endpoint_view(
                        storage, self.carry, view, shapes[view.carry.input]
                    )
                )
                for view in storage.carry_views
            ):
                return False
        return True

    def binds(
        self,
        plan: ChainedMatmulPlan,
        storage: PipelineStorage,
        candidate: CompletedMemberCandidate,
        boundaries: Mapping[Node, str],
        execution: ChainedExecution,
        prefix: str,
        carry_endpoints: CompletedCarryEndpoints | None,
    ) -> bool:
        raw = self.candidate.plan.prepared_widenings
        rebound = plan.prepared_widenings
        return (
            self.matches_storage(storage)
            and replace(
                plan,
                loop_workspace=self.plan.loop_workspace,
                prepared_widenings=self.plan.prepared_widenings,
            )
            == self.plan
            and (
                raw is None
                and rebound is None
                or raw is not None
                and rebound is not None
                and rebound.matches(plan)
                and raw.frame is rebound.frame
                and raw.shapes == rebound.shapes
                and tuple(item.buffer for item in raw.bindings)
                == tuple(item.buffer for item in rebound.bindings)
            )
            and candidate.node is self.candidate.node
            and candidate.store is self.candidate.store
            and candidate.group is self.candidate.group
            and candidate.member == self.candidate.member
            and candidate.native == self.candidate.native
            and candidate.facts == self.candidate.facts
            and candidate.shapes == self.candidate.shapes
            and execution == self.execution
            and prefix == self.prefix
            and all(
                boundaries.get(node) == name for node, name in self.prepared.boundaries
            )
            and (
                self.carry is None
                and carry_endpoints is None
                or self.carry is not None
                and carry_endpoints is not None
                and carry_endpoints.carry is self.carry
                and carry_endpoints.storage is storage
                and carry_endpoints.matches(plan)
            )
        )


def prepare_completed_store_plan(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    lowering_plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    stages: tuple[StageTransports, ...],
    boundaries: Mapping[Node, str],
    execution: ChainedExecution,
    preparation: BoundPreparationStorage | None,
    carry: LoopTmemCarryTransport | None,
) -> CompletedStorePlan:
    from .chained_completed_members import plan_completed_member_store
    from .chained_preparation_cut import _shape_input

    if (
        plan.loop is None
        or not stages
        or not _accepted_transports(plan, pipeline, stages)
        or replace(lowering_plan, prepared_widenings=plan.prepared_widenings) != plan
        or lowering_plan.prepared_widenings is not None
        and not lowering_plan.prepared_widenings.matches(lowering_plan)
    ):
        raise chain._UnsupportedChain("completed store allocation lost transports")
    group, selection = stages[-1].group, stages[-1]
    shapes = {
        node: chain._shape(node)
        for node in plan.loop.region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    }
    choices = [
        (index, candidate)
        for index, store in enumerate(plan.loop.region.stores)
        for member in group.stages
        if (
            candidate := plan_completed_member_store(
                lowering_plan, group, member, store, shapes
            )
        )
        is not None
    ]
    if len(choices) != 1:
        raise chain._UnsupportedChain(
            "completed member store lacks one exclusive result"
        )
    index, candidate = choices[0]
    if (
        execution.threads != 128
        or execution.tmem != "chain_tptr"
        or pipeline.recurrence_threads != 128
        or selection.tmem_output is not None
        or selection.residency is not None
        and selection.residency.source_stage == group.stages[0]
        or selection.tmem_accumulator is not None
        and (
            selection.tmem_carry is None
            or selection.tmem_accumulator != selection.tmem_carry.arena
        )
        or candidate.node in boundaries
        or plan_completed_member_map(
            candidate.native.full_shape,
            candidate.native.logical_shape,
            candidate.native.offset,
            candidate.native.width,
            candidate.native.transpose,
            candidate.native.input_dtype,
            operand_source="SMEM" if selection.tmem_input is None else "TMEM",
        )
        != candidate.native
    ):
        raise chain._UnsupportedChain(
            "completed store allocation lacks common native ownership"
        )
    c_name = dict(pipeline.recurrence.bindings).get(candidate.node)
    if c_name is None or c_name != f"chain_{candidate.member}_c":
        raise chain._UnsupportedChain(
            "completed store allocation lacks original C request"
        )
    request = pipeline.recurrence.layout.region(c_name)
    if request.byte_size < math.prod(candidate.native.logical_shape) * 4:
        raise chain._UnsupportedChain("completed store original C request is too small")
    # No missing recurrence result may be silently recomputed by early lowering.
    pending = [
        node
        for node in candidate.store.all_input_nodes
        if node is not candidate.store.args[0]
    ]
    seen = set()
    while pending:
        node = pending.pop()
        if (
            node in seen
            or node is candidate.node
            or node in boundaries
            or _shape_input(node)
        ):
            continue
        seen.add(node)
        if node in plan.dots or any(
            node in (item.input, item.output) for item in plan.loop.region.carries
        ):
            raise chain._UnsupportedChain(
                f"completed store allocation has another recurrence read: {node.format_node()}"
            )
        pending.extend(node.all_input_nodes)
    prefix = f"chain_store_{index}"
    alias_container = lowering_plan.tensor_aliases
    before_aliases = tuple(lowering_plan.tensor_aliases.items())
    argument_order = tuple(id(arg) for arg in cg.device_function.arguments)
    prepared = _prepare_point(
        cg, lowering_plan, candidate, boundaries, c_name, prefix, defer_target=True
    )
    if (
        lowering_plan.tensor_aliases is not alias_container
        or tuple(lowering_plan.tensor_aliases.items()) != before_aliases
        or tuple(id(arg) for arg in cg.device_function.arguments) != argument_order
    ):
        raise chain._UnsupportedChain("early completed store registered a host alias")
    if prepared.point.host_reads or any(
        _metadata_region(pipeline, preparation, node, name) is None
        for node, name in prepared.reads
    ):
        raise chain._UnsupportedChain("completed store has unproved metadata reads")
    result = CompletedStorePlan(
        plan,
        pipeline,
        stages,
        candidate,
        execution,
        prefix,
        c_name,
        request,
        prepared,
        preparation,
        carry,
        None,
        (),
    )
    aliases: dict[int, tuple[dict[str, str], tuple[tuple[str, str], ...]]] = {}
    result = replace(
        result,
        _facts=_records(result._fields(), aliases=aliases),
        _aliases=tuple(aliases.values()),
    )
    return replace(
        result, _selection=(result._fields(), result._facts, result._aliases)
    )


@dataclass
class _Progress:
    token: object
    history: tuple[object, ...]
    prepared: _Prepared | None = None
    prepared_facts: object = None
    stage: tuple[str, ...] = ()


@dataclass
class _EndpointProgress:
    token: object
    history: tuple[object, ...]
    lines: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CompletedCarryEndpoints:
    """Concrete original upload/drain, outside the entire ordered loop.

    Only this adapter's original upload establishes the entry receipt. The
    matching drain includes the original post-loop CTA join and must be emitted
    exactly once before the enclosing pipeline can return its source.
    """

    pipeline: PreparationPipeline
    storage: PipelineStorage
    carry: LoopTmemCarryTransport
    entry: tuple[str, ...]
    _aliases: tuple[tuple[str, str], ...]
    _facts: tuple[
        object, tuple[tuple[dict[str, str], tuple[tuple[str, str], ...]], ...]
    ]
    _tokens: tuple[object, object]
    _progress: _EndpointProgress
    _binding: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        return (
            self.pipeline,
            self.storage,
            self.carry,
            self.entry,
            self._aliases,
            self._facts,
            self._tokens,
            self._progress,
        )

    def matches(self, plan: ChainedMatmulPlan) -> bool:
        return (
            self._fields() == self._binding
            and self._progress is self._binding[-1]
            and self._progress.token in self._tokens
            and all(
                plan.tensor_aliases.get(name) == alias for name, alias in self._aliases
            )
            and self._progress.history
            == self._tokens[: self._tokens.index(self._progress.token) + 1]
            and (self._progress.lines is None)
            == (self._progress.token is self._tokens[0])
            and _physical_matches(self._facts, self.pipeline, self.storage)
            and _accepted_transports(plan, self.pipeline, self.storage.stages)
            and tuple(
                stage.tmem_carry
                for stage in self.storage.stages
                if stage.tmem_carry is not None
            )
            == (self.carry,)
        )

    def permits(self, plan: ChainedMatmulPlan, view: CarryStorageView) -> bool:
        return (
            self.matches(plan)
            and self._progress.token is self._tokens[0]
            and _endpoint_view(self.storage, self.carry, view)
        )

    def drain(
        self, plan: ChainedMatmulPlan, carry: LoopTmemCarryTransport
    ) -> list[str]:
        if (
            not self.matches(plan)
            or carry is not self.carry
            or self._progress.token is not self._tokens[0]
        ):
            raise chain._UnsupportedChain("completed carry endpoint proof changed")
        lines = ["cute.arch.sync_threads()", *carry.shared_transfer(upload=False)]
        self._progress.lines = tuple(lines)
        self._progress.token = self._tokens[1]
        self._progress.history += (self._tokens[1],)
        return lines

    def validate_drained(self, plan: ChainedMatmulPlan, lines: Sequence[str]) -> None:
        if (
            not self.matches(plan)
            or self._progress.token is not self._tokens[1]
            or tuple(lines) != self._progress.lines
        ):
            raise chain._UnsupportedChain("completed carry drain is missing")


def emit_completed_carry_entry(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    storage: PipelineStorage,
    carry: LoopTmemCarryTransport,
) -> tuple[list[str], CompletedCarryEndpoints]:
    if not _accepted_transports(plan, pipeline, storage.stages) or tuple(
        stage.tmem_carry for stage in storage.stages if stage.tmem_carry is not None
    ) != (carry,):
        raise chain._UnsupportedChain("completed carry upload is unproved")
    lines = [*carry.view(), *carry.shared_transfer(upload=True)]
    tokens = (object(), object())
    values = (
        pipeline,
        storage,
        carry,
        tuple(lines),
        tuple(plan.tensor_aliases.items()),
        _physical_facts(pipeline, storage),
        tokens,
        _EndpointProgress(tokens[0], (tokens[0],)),
    )
    receipt = CompletedCarryEndpoints(*values, values)
    return lines, receipt


@dataclass(frozen=True)
class CompletedStoreAction:
    plan: ChainedMatmulPlan
    pipeline: PreparationPipeline
    storage: PipelineStorage
    candidate: CompletedMemberCandidate
    execution: ChainedExecution
    _execution: object
    prefix: str
    phase: str
    output_lease: BoundOutputLease | None
    carry_endpoints: CompletedCarryEndpoints | None
    inputs: tuple[tuple[Node, str], ...]
    c_name: str
    _aliases: tuple[tuple[str, str], ...]
    _facts: tuple[
        object, tuple[tuple[dict[str, str], tuple[tuple[str, str], ...]], ...]
    ]
    _tokens: tuple[object, ...]
    _progress: _Progress
    _selection: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        return (
            self.plan,
            self.pipeline,
            self.storage,
            self.candidate,
            self.execution,
            self._execution,
            self.prefix,
            self.phase,
            self.output_lease,
            self.carry_endpoints,
            self.inputs,
            self.c_name,
            self._aliases,
            self._facts,
            self._tokens,
            self._progress,
        )

    def matches(self, plan: ChainedMatmulPlan) -> bool:
        prepared = self._progress.prepared
        return (
            self._selection == self._fields()
            and self._execution == _records(self.execution)
            and self._progress is self._selection[-1]
            and plan is self.plan
            and all(
                plan.tensor_aliases.get(name) == alias for name, alias in self._aliases
            )
            and self.candidate.matches(plan, dict(self.candidate.shapes))
            and _physical_matches(self._facts, self.pipeline, self.storage)
            and (
                self.storage.completed_store is None
                or self.storage.completed_store.matches_storage(self.storage)
            )
            and plan.loop_workspace is self.storage.recurrence.layout
            and (
                self.carry_endpoints is None
                or self.carry_endpoints.matches(plan)
                and self.carry_endpoints.storage is self.storage
                and self.carry_endpoints.pipeline is self.pipeline
            )
            and self._progress.token in self._tokens
            and self._progress.history
            == self._tokens[: self._tokens.index(self._progress.token) + 1]
            and (
                prepared is None
                and self._progress.prepared_facts is None
                or prepared is not None
                and self._progress.prepared_facts == _records(prepared)
            )
            and (
                self.output_lease is None
                or self.output_lease.matches(plan)
                and self.output_lease.storage is self.storage
            )
        )

    def begin_stage(
        self,
        plan: ChainedMatmulPlan,
        selection: StageTransports,
        execution: ChainedExecution,
        boundaries: Mapping[Node, str],
        phase: str,
    ) -> None:
        if (
            not self.matches(plan)
            or self._progress.token is not self._tokens[0]
            or selection != self.storage.stages[-1]
            or execution != self.execution
            or phase != self.phase
            or tuple(boundaries.items()) != self.inputs
            or self.candidate.node in boundaries
        ):
            raise chain._UnsupportedChain("completed store stage binding changed")
        self._progress.token = self._tokens[1]
        self._progress.history += (self._tokens[1],)

    def prepare_point(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: Mapping[Node, str],
    ) -> None:
        """Called after the actual full load, before any publication omission."""
        if (
            not self.matches(plan)
            or self._progress.token is not self._tokens[1]
            or tuple(boundaries.items()) != self.inputs
        ):
            raise chain._UnsupportedChain("completed store preparation changed")
        epilogue = (
            dict(boundaries)
            if self.output_lease is None
            else self.output_lease.epilogue_boundaries(plan, boundaries)
        )
        contract = self.storage.completed_store
        prepared = (
            _prepare_point(cg, plan, self.candidate, epilogue, self.c_name, self.prefix)
            if contract is None
            else contract.prepared
        )
        if contract is not None:
            target = prepared.point.target
            if not isinstance(target, UnboundStoreTarget) or prepared.lines is not None:
                raise chain._UnsupportedChain(
                    "completed store target preparation changed"
                )
            point = replace(
                prepared.point,
                target=target.bind(cg, plan, self.candidate.store, epilogue),
            )
            prepared = replace(
                prepared,
                point=point,
                lines=_point_lines(self.candidate, point, self.prefix),
            )
        if prepared.point.host_reads or any(
            epilogue.get(node) != name or not self._metadata_read(node, name)
            for node, name in prepared.reads
        ):
            raise chain._UnsupportedChain("completed store has unproved metadata reads")
        self._progress.prepared = replace(prepared, boundaries=tuple(epilogue.items()))
        self._progress.prepared_facts = _records(self._progress.prepared)
        self._progress.token = self._tokens[2]
        self._progress.history += (self._tokens[2],)

    def _metadata_read(self, node: Node, name: str) -> bool:
        if self.output_lease is not None:
            images = tuple(
                image
                for image in self.output_lease.proof.images
                if image.buffer.node is node and image.name == name
            )
            return len(images) == 1 and (
                self.output_lease.proof.layout.region(name).byte_size
                >= math.prod(images[0].buffer.shape) * images[0].buffer.dtype.itemsize
            )
        region = _metadata_region(self.pipeline, self.storage.preparation, node, name)
        return region is not None and not any(
            carry.pool == "frames"
            and not (
                self.carry_endpoints is not None
                and self.carry_endpoints.permits(self.plan, carry)
            )
            and max(region.byte_offset, carry.byte_offset)
            < min(
                region.byte_end,
                carry.byte_offset + math.prod(carry.shape) * carry.dtype.itemsize,
            )
            for carry in self.storage.carry_views
        )

    def suppresses(self, member: int) -> bool:
        if not self.matches(self.plan) or self._progress.token is not self._tokens[2]:
            raise chain._UnsupportedChain("completed store publication is not prepared")
        return member == self.candidate.member

    def complete_stage(
        self,
        plan: ChainedMatmulPlan,
        boundaries: Mapping[Node, str],
        lines: Sequence[str],
    ) -> None:
        """Only the concrete full-load/publication/join branch calls this."""
        if (
            not self.matches(plan)
            or self._progress.token is not self._tokens[2]
            or self.candidate.node in boundaries
            or any(boundaries.get(node) != name for node, name in self.inputs)
        ):
            raise chain._UnsupportedChain("completed store completion changed")
        self._progress.stage = tuple(lines)
        self._progress.token = self._tokens[3]
        self._progress.history += (self._tokens[3],)

    def consume(
        self,
        plan: ChainedMatmulPlan,
        store: Node,
        boundaries: Mapping[Node, str],
        execution: ChainedExecution,
        prefix: str,
        stage_lines: Sequence[str],
    ) -> list[str]:
        prepared = self._progress.prepared
        if (
            not self.matches(plan)
            or self._progress.token is not self._tokens[3]
            or prepared is None
            or prepared.lines is None
            or not isinstance(prepared.point.target, str)
            or store is not self.candidate.store
            or execution != self.execution
            or prefix != self.prefix
            or tuple(stage_lines) != self._progress.stage
            or self.candidate.node in boundaries
            or any(boundaries.get(node) != name for node, name in prepared.reads)
            or any(not self._metadata_read(node, name) for node, name in prepared.reads)
        ):
            raise chain._UnsupportedChain("completed store consumption changed")
        lines = list(prepared.lines)
        self._progress.token = self._tokens[4]
        self._progress.history += (self._tokens[4],)
        return lines

    def validate_consumed(self) -> None:
        if not self.matches(self.plan) or self._progress.token is not self._tokens[4]:
            raise chain._UnsupportedChain(
                "completed store was not consumed exactly once"
            )


def prepare_completed_store(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    storage: PipelineStorage,
    candidate: CompletedMemberCandidate,
    boundaries: Mapping[Node, str],
    execution: ChainedExecution,
    *,
    prefix: str,
    phase: str,
    output_lease: BoundOutputLease | None = None,
    carry_endpoints: CompletedCarryEndpoints | None = None,
) -> CompletedStoreAction | None:
    from .chained_loop import loop_storage_is_proven
    from .chained_loop import loop_storage_matches_runtime

    contract = storage.completed_store
    if (
        plan.loop is None
        or not candidate.matches(plan, dict(candidate.shapes))
        or not storage.stages
        or storage.stages[-1].group != candidate.group
        or candidate.store not in plan.loop.region.stores
        or execution.threads != 128
        or execution.tmem != "chain_tptr"
        or pipeline.recurrence_threads != 128
        or plan.loop_workspace is not storage.recurrence.layout
        or not loop_storage_is_proven(plan.loop)
        or not loop_storage_matches_runtime(plan.loop)
        or not _accepted_transports(plan, pipeline, storage.stages)
        or candidate.node in boundaries
        or contract is None
        and candidate.node in storage.omitted_results
        or storage.output_lease is not None
        and output_lease is None
        or output_lease is not None
        and not output_lease.matches(plan)
    ):
        return None
    preparation = storage.preparation
    if preparation is not None and (
        not preparation.matches(preparation.physical.accepted.revision.plan, pipeline)
        or preparation._state.finalized is not storage
        or not preparation._state.consumed
        or dict(storage.allocations).get("frames")
        != pipeline.slots * preparation.stride
    ):
        return None
    selection = storage.stages[-1]
    if (
        selection.tmem_output is not None
        or selection.residency is not None
        and selection.residency.source_stage == candidate.group.stages[0]
        or selection.tmem_accumulator is not None
        and (
            selection.tmem_carry is None
            or selection.tmem_accumulator != selection.tmem_carry.arena
        )
    ):
        return None
    c_name = (
        dict(storage.recurrence.bindings).get(candidate.node)
        if contract is None
        else contract.c_name
    )
    regions = {region.name: region for region in storage.recurrence.layout.regions}
    if (
        c_name is None
        or c_name != f"chain_{candidate.member}_c"
        or contract is None
        and (
            c_name not in regions
            or regions[c_name].byte_size < math.prod(candidate.native.logical_shape) * 4
            or regions[c_name].byte_end > storage.recurrence.layout.allocated_bytes
        )
        or contract is not None
        and not contract.binds(
            plan, storage, candidate, boundaries, execution, prefix, carry_endpoints
        )
        or plan_completed_member_map(
            candidate.native.full_shape,
            candidate.native.logical_shape,
            candidate.native.offset,
            candidate.native.width,
            candidate.native.transpose,
            candidate.native.input_dtype,
            operand_source="SMEM" if selection.tmem_input is None else "TMEM",
        )
        != candidate.native
    ):
        return None
    tokens = tuple(object() for _ in range(5))
    fields = (
        plan,
        pipeline,
        storage,
        candidate,
        execution,
        _records(execution),
        prefix,
        phase,
        output_lease,
        carry_endpoints,
        tuple(boundaries.items()),
        c_name,
        tuple(plan.tensor_aliases.items()),
        _physical_facts(pipeline, storage),
        tokens,
        _Progress(tokens[0], (tokens[0],)),
    )
    result = CompletedStoreAction(*fields, fields)
    return result if result.matches(plan) else None
