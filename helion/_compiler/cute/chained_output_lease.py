"""Typed terminal frame reads detached before ordered-consumer publication.

The plan is not permission to release a frame. Same-attempt storage, original
runtime alias guards, and the final asynchronous completion must still bind it.
No frame allocation or original expression is shortened or rewritten here.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
import math
from typing import TYPE_CHECKING

import torch

from .chained_pipeline_storage import _accepted_transports
from .chained_pipeline_storage import _facts
from .chained_preparation_cut import _known_effects
from .chained_preparation_cut import _shape_input
from .chained_prepared_groups import _valid_frame
from .chained_workspace import _align
from .chained_workspace import _reuse_requests
from .contraction_region import _domain
from .warp_specialized_plan import SharedBufferRequest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from .chained_contraction_groups import ContractionGroup
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pipeline_storage import PipelineStorage
    from .chained_pipeline_storage import StageTransports
    from .chained_pipeline_storage import StorageRevision
    from .chained_preparation_frame import PreparationBuffer
    from .chained_preparation_pipeline import PreparationPipeline
    from .contraction_region import LogicalDomain
    from .warp_specialized_plan import SharedBufferRegion
    from .warp_specialized_plan import SharedMemoryLayoutPlan


@dataclass(frozen=True)
class OutputLeaseImage:
    """One original initialized image, not a recomputed equivalent expression."""

    buffer: PreparationBuffer
    source: SharedBufferRegion
    domain: LogicalDomain
    name: str


@dataclass(frozen=True)
class OutputLeaseSnapshot:
    revision: StorageRevision
    stages: tuple[StageTransports, ...]
    final_group: ContractionGroup
    roots: tuple[Node, ...]
    images: tuple[OutputLeaseImage, ...]
    result_reads: tuple[Node, ...]
    carry_reads: tuple[Node, ...]
    resident_carries: frozenset[int]
    layout: SharedMemoryLayoutPlan
    _selection: tuple[object, ...] = field(repr=False)

    def matches(
        self,
        plan: ChainedMatmulPlan,
        pipeline: PreparationPipeline,
        stages: tuple[StageTransports, ...],
        revision: StorageRevision,
    ) -> bool:
        return (
            self.revision is revision
            and revision.plan is plan
            and revision.pipeline is pipeline
            and self.stages == stages
            and self._selection
            == (
                self.revision,
                self.stages,
                self.final_group,
                self.roots,
                self.images,
                self.result_reads,
                self.carry_reads,
                self.resident_carries,
                self.layout,
            )
            and plan.region is not None
            and plan.loop is not None
            and revision.facts == _facts(plan)
            and all(
                plan.tensor_aliases.get(name) == alias
                for name, alias in revision.aliases
            )
            and _valid_frame(pipeline.frame)
            and _accepted_transports(plan, pipeline, stages)
        )


def _terminal_reads(
    roots: tuple[Node, ...],
    *,
    boundaries: Mapping[Node, PreparationBuffer],
    preparation: frozenset[Node],
    results: frozenset[Node],
    carries: frozenset[Node],
    omitted: frozenset[Node],
) -> tuple[frozenset[Node], frozenset[Node], frozenset[Node]] | None:
    pending = list(roots)
    visited: set[Node] = set()
    images: set[Node] = set()
    result_reads: set[Node] = set()
    carry_reads: set[Node] = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        if node in boundaries:
            images.add(node)
        elif _shape_input(node):
            # Existing exact shape-query proof, not arbitrary metadata taint.
            continue
        elif node in omitted:
            return None
        elif node in results:
            result_reads.add(node)
        elif node in carries:
            carry_reads.add(node)
        elif node in preparation or not _known_effects(node):
            return None
        else:
            # all_input_nodes includes indices, masks, explicit accumulators
            # and kwargs; no tensor-valued side path may evade this cut.
            pending.extend(node.all_input_nodes)
    return frozenset(images), frozenset(result_reads), frozenset(carry_reads)


def plan_output_lease_snapshot(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    stages: tuple[StageTransports, ...],
    *,
    revision: StorageRevision,
) -> OutputLeaseSnapshot | None:
    """Select complete ordinary frontier images used after the final MMA wait.

    Native/retained/leaf images require different initialization/view proofs
    and are deliberately not admitted by this first terminal-snapshot path.
    The snapshot occupies independent simultaneous storage; no byte discount
    or role-overlap authority is granted by this pure result.
    """
    if (
        plan.region is None
        or plan.loop is None
        or revision.plan is not plan
        or revision.pipeline is not pipeline
        or revision.facts != _facts(plan)
        or not _valid_frame(pipeline.frame)
        or not stages
        or not _accepted_transports(plan, pipeline, stages)
        or stages[-1].group != pipeline.recurrence.stages[-1].group
        or stages[-1].tmem_output is not None
        or stages[-1].residency is not None
        and stages[-1].residency.source_stage == stages[-1].group.stages[0]
    ):
        return None
    frame, region = pipeline.frame, plan.loop.region
    if (
        frame.cut.region.graph is not plan.region.graph
        or frame.cut.region.nodes != plan.region.nodes
        or region.graph is not plan.region.graph
        or frame.cut.storage_proof_key != plan.loop.storage_key
        or tuple(item.group for item in stages)
        != tuple(item.group for item in pipeline.recurrence.stages)
    ):
        return None
    resident = frozenset(
        item.tmem_carry.candidate.carry_index
        for item in stages
        if item.tmem_carry is not None
    )
    roots = (
        *region.stores,
        *(
            carry.output
            for carry in region.carries
            if carry.input_index not in resident
        ),
    )
    boundaries = {
        buffer.node: buffer
        for buffer in frame.buffers
        if buffer.kind == "frontier" and buffer.node is not None
    }
    omitted = frozenset(
        plan.dots[index]
        for item in stages
        for index in item.group.stages
        if item.tmem_output is not None
        or item.residency is not None
        and item.residency.source_stage == item.group.stages[0]
    ) | frozenset(
        carry.output for carry in region.carries if carry.input_index in resident
    )
    reads = _terminal_reads(
        roots,
        boundaries=boundaries,
        preparation=frozenset(frame.cut.preparation),
        results=frozenset(plan.dots),
        carries=frozenset(carry.input for carry in region.carries),
        omitted=omitted,
    )
    if reads is None or not reads[0]:
        return None
    selected, result_reads, carry_reads = reads
    native = {operand.buffer.name for operand in pipeline.prepared_operands}
    native.update(
        member.buffer.name
        for binding in pipeline.prepared_groups
        for member in binding.candidate.members
    )
    shapes = dict(revision.shapes)
    images = []
    requests = []
    for node in region.nodes:
        if node not in selected:
            continue
        buffer = boundaries[node]
        value = node.meta.get("val")
        source = frame.layout.region(buffer.name)
        writes = tuple(
            action for action in frame.actions if buffer.name in action.writes
        )
        if (
            buffer.name in native
            or not isinstance(value, torch.Tensor)
            or value.dtype != buffer.dtype
            or buffer.dtype
            not in (
                torch.bool,
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.bfloat16,
                torch.float16,
                torch.float32,
            )
            or shapes.get(node) != buffer.shape
            or any(type(size) is not int or size <= 0 for size in buffer.shape)
            or source.byte_size
            < _align(math.prod(buffer.shape) * buffer.dtype.itemsize)
            or source.live_until != frame.actions[-1].publication_event
            or len(writes) != 1
            or writes[0].kind != "frontier"
            or writes[0].nodes != (node,)
            or writes[0].publication_event > frame.actions[-1].event
        ):
            return None
        name = f"chain_output_snapshot_{len(images)}"
        images.append(OutputLeaseImage(buffer, source, _domain(value), name))
        requests.append(
            SharedBufferRequest(
                name, math.prod(buffer.shape) * buffer.dtype.itemsize, 128
            )
        )
    layout = _reuse_requests(tuple(requests))
    selection = (
        revision,
        stages,
        stages[-1].group,
        tuple(roots),
        tuple(images),
        tuple(node for node in region.nodes if node in result_reads),
        tuple(node for node in region.nodes if node in carry_reads),
        resident,
        layout,
    )
    candidate = OutputLeaseSnapshot(*selection, selection)
    return candidate if candidate.matches(plan, pipeline, stages, revision) else None


@dataclass
class _CompletionLedger:
    receipt: object | None = None


@dataclass(frozen=True)
class BoundOutputLease:
    """Same-attempt, detached-storage completion action for one final group.

    The stage may invoke this only after its original async wait. This is not
    a callback: the only permitted operations copy the proved typed images,
    join their whole consumer team, and return the original EMPTY token.
    """

    proof: OutputLeaseSnapshot
    plan: ChainedMatmulPlan
    pipeline: PreparationPipeline
    storage: PipelineStorage
    execution: ChainedExecution
    sources: tuple[tuple[Node, str], ...]
    setup: tuple[str, ...]
    copies: tuple[str, ...]
    _completion: _CompletionLedger = field(repr=False)
    _issued_receipt: object = field(repr=False)
    _binding: tuple[object, ...] = field(repr=False)

    def matches(self, plan: ChainedMatmulPlan) -> bool:
        original = self.proof.revision.plan
        return (
            self.plan is plan
            and replace(plan, loop_workspace=original.loop_workspace) == original
            and plan.loop_workspace is self.storage.recurrence.layout
            and self.storage.output_lease is self.proof
            and (
                self._completion.receipt is None
                or self._completion.receipt is self._issued_receipt
            )
            and self.proof.matches(
                original,
                self.pipeline,
                self.storage.stages,
                self.proof.revision,
            )
            and self._binding
            == (
                self.proof,
                self.plan,
                self.pipeline,
                self.storage,
                self.execution,
                self.sources,
                self.setup,
                self.copies,
                self._completion,
                self._issued_receipt,
            )
        )

    def post_issue_lines(
        self,
        plan: ChainedMatmulPlan,
        selection: StageTransports,
        execution: ChainedExecution,
        boundaries: Mapping[Node, str],
    ) -> list[str]:
        from .chained_matmul import _UnsupportedChain

        if (
            not self.matches(plan)
            or self._completion.receipt is not None
            or selection != self.storage.stages[-1]
            or execution != self.execution
            or any(boundaries.get(node) != name for node, name in self.sources)
        ):
            raise _UnsupportedChain("output lease completion proof is stale")
        lines = [
            *self.copies,
            execution.sync,
            f"if {execution.warp} == 0:",
            f"    chain_sync.arrive_mbarrier(chain_slot_bars + {self.pipeline.slots} + chain_slot)",
        ]
        self._completion.receipt = self._issued_receipt
        return lines

    def epilogue_boundaries(
        self, plan: ChainedMatmulPlan, boundaries: Mapping[Node, str]
    ) -> dict[Node, str]:
        from .chained_matmul import _UnsupportedChain

        if (
            not self.matches(plan)
            or self._completion.receipt is not self._issued_receipt
            or any(boundaries.get(node) != name for node, name in self.sources)
        ):
            raise _UnsupportedChain("output lease epilogue proof is stale")
        result = dict(boundaries)
        for image in self.proof.images:
            assert image.buffer.node is not None
            result[image.buffer.node] = image.name
        return result


def bind_output_lease_snapshot(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    storage: PipelineStorage,
    boundaries: Mapping[Node, str],
    execution: ChainedExecution,
) -> BoundOutputLease | None:
    """Bind accepted allocation and runtime alias proof, without re-evaluation.

    Ordinary frontier producers initialize every logical cell, including their
    original masked values. Typed scalar copies preserve those exact bits; the
    original expression reader still owns bounds/Boolean restoration at every
    epilogue use. Preparation uses only its unchanged frame slab. Independent
    C/TMEM storage and the original final CTA carry-drain join are untouched.
    """
    from .chained_loop import loop_storage_is_proven
    from .chained_loop import loop_storage_matches_runtime
    from .chained_prepared_values import storage_dtype

    proof = storage.output_lease
    if (
        proof is None
        or plan.loop is None
        or not proof.matches(
            proof.revision.plan, pipeline, storage.stages, proof.revision
        )
        or replace(plan, loop_workspace=proof.revision.plan.loop_workspace)
        != proof.revision.plan
        or plan.loop_workspace is not storage.recurrence.layout
        or dict(storage.allocations).get("output_snapshot")
        != proof.layout.allocated_bytes
        or execution.threads != pipeline.recurrence_threads
        or not loop_storage_is_proven(plan.loop)
        or not loop_storage_matches_runtime(plan.loop)
        or any(
            view.carry.input in proof.carry_reads and view.pool == "frames"
            for view in storage.carry_views
        )
        or any(
            node not in dict(storage.recurrence.bindings) for node in proof.result_reads
        )
    ):
        return None
    sources = []
    setup = []
    copies = []
    for image in proof.images:
        buffer, name = image.buffer, image.name
        node = buffer.node
        assert node is not None
        if boundaries.get(node) != buffer.name:
            return None
        sources.append((node, buffer.name))
        region = proof.layout.region(name)
        dtype = storage_dtype(buffer.dtype)
        shape = buffer.shape or (1,)
        strides = tuple(math.prod(shape[index + 1 :]) for index in range(len(shape)))
        setup.append(
            f"{name} = cute.make_tensor(cute.recast_ptr(chain_output_snapshots + {region.byte_offset}, dtype={dtype}), cute.make_layout({shape!r}, stride={strides!r}))"
        )
        size = math.prod(shape)
        index = f"{name}_index"
        coords = ", ".join(
            f"({index} // {stride}) % {extent}"
            for stride, extent in zip(strides, shape, strict=True)
        )
        copies.extend(
            [
                f"for {name}_part in cutlass.range_constexpr({(size + execution.threads - 1) // execution.threads}):",
                f"    {index} = {execution.thread} + {name}_part * {execution.threads}",
                f"    if {index} < {size}:",
                f"        {name}[{coords}] = {buffer.name}[{coords}]",
            ]
        )
    selection = (
        proof,
        plan,
        pipeline,
        storage,
        execution,
        tuple(sources),
        tuple(setup),
        tuple(copies),
        _CompletionLedger(),
        object(),
    )
    result = BoundOutputLease(*selection, selection)
    return result if result.matches(plan) else None
