"""Actual scan-native copies, rebound through the accepted physical table.

The original ScanProducer remains the schedule proof. This separate receipt
records only successful transfers inside its one completed emitted macro. Raw
loads are live during the fused scan/fill phase, not the later MMA; every other
original action retains its own effective phase and complete owner coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
from typing import TYPE_CHECKING

import torch

from .chained_native_read_inputs import bind_preparation_leaf_native_inputs
from .chained_native_reads import NativeVectorRead
from .chained_native_reads import plan_native_vector_read
from .chained_preparation_transfers import BoundPreparationTransfers
from .chained_preparation_transfers import bind_preparation_transfer_span
from .chained_preparation_transfers import preparation_transfer_owner
from .chained_scratch_layout import xor_swizzle

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_native_read_inputs import NativeReadInput
    from .chained_preparation_actions import AcceptedPreparation
    from .chained_preparation_actions import AcceptedPreparationAction
    from .chained_preparation_fragment import PreparationFragment
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_preparation_storage import AcceptedPreparationStorage
    from .chained_scan_producer import ScanProducer
    from .chained_scan_producer_emission import ScanProducerEmission


@dataclass(frozen=True)
class ScanTransferReceipt:
    plan: ChainedMatmulPlan
    pipeline: PreparationPipeline
    candidate: ScanProducer
    inputs: tuple[tuple[Node, str], ...]
    published: tuple[tuple[Node, str], ...]
    outputs: tuple[tuple[Node, str], ...]
    execution: ChainedExecution
    transport_facts: tuple[object, ...]
    body_first: int
    lines: tuple[str, ...]
    emission: ScanProducerEmission
    reads: tuple[tuple[NativeReadInput, NativeVectorRead], ...]
    _selection: tuple[object, ...]
    fragment: PreparationFragment | None = None

    def _fields(self) -> tuple[object, ...]:
        result = (
            self.plan,
            self.pipeline,
            self.candidate,
            self.inputs,
            self.published,
            self.outputs,
            self.execution,
            self.transport_facts,
            self.body_first,
            self.lines,
            self.emission,
            self.reads,
        )
        return (*result, self.fragment) if self.fragment is not None else result

    def _valid(self) -> bool:
        from .chained_preparation_actions import _transport_facts
        from .chained_preparation_actions import workspace_name

        scan, emission = self.candidate, self.emission
        if (
            self._selection != self._fields()
            or self.pipeline.scan_producer is not scan
            or self.transport_facts != _transport_facts(self.pipeline)
            or not scan.matches(self.plan, self.pipeline, dict(scan.revision.shapes))
            or type(self.body_first) is not int
            or self.body_first < 0
            or not self.lines
            or not emission.lines
            or emission.native_reads is not True
            or not self.reads
            or tuple(source for source, _ in self.reads) != emission.native_sources
            or emission.ownership is None
            or self.execution.threads != 128
            or self.execution.a_workspace != workspace_name(scan.stage.a.name)
            or self.execution.b_workspace != workspace_name(scan.stage.b.name)
            or emission.ownership.threads != self.execution.threads
            or emission.ownership.thread_order != "column_major"
            or not emission.ownership.matches(scan.shape, self.execution.threads)
        ):
            return False
        published = dict(self.inputs)
        for action in scan.prelude:
            published.update(zip(action.nodes, action.writes, strict=True))
        if published != dict(self.published):
            return False
        expected = dict(published)
        for action in scan.deferred:
            expected.update(zip(action.nodes, action.writes, strict=True))
        mma = self.pipeline.frame.actions[scan.stop_event]
        if mma.kind != "mma" or mma.stages != scan.stage.group.stages:
            return False
        expected.update(zip(mma.nodes, mma.writes, strict=True))
        if self.fragment is not None:
            if (
                self.fragment.pipeline is not self.pipeline
                or self.fragment.source_event != mma.event
            ):
                return False
            expected = self.fragment.shared_outputs(expected)
        expected.pop(scan.scan.node, None)
        if expected != dict(self.outputs):
            return False
        eligible = bind_preparation_leaf_native_inputs(
            self.plan,
            self.pipeline,
            scan,
            published,
            dict(scan.revision.shapes),
        )
        return (
            len({source.node for source, _ in self.reads}) == len(self.reads)
            and len({source.index for source, _ in self.reads}) == len(self.reads)
            and all(
                source in eligible
                and source.matches(self.plan, published)
                and geometry
                == plan_native_vector_read(
                    source.full_shape,
                    source.shape,
                    source.row_offset,
                    source.dtype,
                    emission.ownership,
                )
                for source, geometry in self.reads
            )
        )

    def matches(
        self, accepted: AcceptedPreparation, action: AcceptedPreparationAction
    ) -> bool:
        return (
            self._valid()
            and self.plan is accepted.revision.plan
            and self.pipeline is accepted.pipeline
            and action.proof is self.candidate
            and action.scan_transfer is self
            and action.fragment is self.fragment
            and action.kind == "scan"
            and (action.first, action.stop)
            == (self.candidate.first_event, self.candidate.stop_event + 1)
            and action.inputs == self.inputs
            and all(
                dict(action.outputs).get(node) == name for node, name in self.outputs
            )
            and self.execution.thread == accepted.execution.thread
            and self.execution.sync == accepted.execution.sync
            and self.execution.threads == accepted.execution.threads
            and accepted.lines[self.body_first : self.body_first + len(self.lines)]
            == self.lines
        )


@dataclass
class ScanTransferAttempt:
    """Private pending completion, created before the original prelude runs."""

    plan: ChainedMatmulPlan
    pipeline: PreparationPipeline
    inputs: tuple[tuple[Node, str], ...]
    body_first: int
    transport_facts: tuple[object, ...]
    fragment: PreparationFragment | None = None
    completed: bool = False
    receipt: ScanTransferReceipt | None = None
    _selection: tuple[object, ...] = field(init=False, repr=False)
    _completion: tuple[ScanProducerEmission, ScanTransferReceipt | None] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._selection = self._fields()

    def _fields(self) -> tuple[object, ...]:
        result = (
            self.plan,
            self.pipeline,
            self.inputs,
            self.body_first,
            self.transport_facts,
        )
        return (*result, self.fragment) if self.fragment is not None else result

    def result(self) -> ScanTransferReceipt | None:
        """A successful native emission cannot lose its pending receipt."""
        if (
            self._selection != self._fields()
            or self.completed is not True
            or self._completion is None
            or self.receipt is not self._completion[1]
            or self._completion[0].native_reads != (self.receipt is not None)
            or self.receipt is not None
            and not self.receipt._valid()
        ):
            raise ValueError("scan completion receipt changed")
        return self.receipt

    def complete(
        self,
        published: Mapping[Node, str],
        after: Mapping[Node, str],
        execution: ChainedExecution,
        emission: ScanProducerEmission,
        lines: list[str],
    ) -> None:
        from .chained_preparation_actions import _transport_facts

        if self.completed:
            raise ValueError("scan transfer completed twice")
        if (
            self._selection != self._fields()
            or self._completion is not None
            or self.receipt is not None
            or self.transport_facts != _transport_facts(self.pipeline)
        ):
            raise ValueError("scan transports changed during emission")
        if not emission.native_reads:
            if emission.native_sources or emission.ownership is not None:
                raise ValueError("unemitted scan native sources")
            self._completion = (emission, None)
            self.completed = True
            return
        scan = self.pipeline.scan_producer
        if scan is None or emission.ownership is None:
            raise ValueError("emitted scan lacks native ownership")
        reads = []
        for source in emission.native_sources:
            geometry = plan_native_vector_read(
                source.full_shape,
                source.shape,
                source.row_offset,
                source.dtype,
                emission.ownership,
            )
            if geometry is None:
                raise ValueError("emitted scan native geometry changed")
            reads.append((source, geometry))
        fields = (
            self.plan,
            self.pipeline,
            scan,
            self.inputs,
            tuple(published.items()),
            tuple(after.items()),
            execution,
            self.transport_facts,
            self.body_first,
            tuple(lines),
            emission,
            tuple(reads),
        )
        selection = (*fields, self.fragment) if self.fragment is not None else fields
        receipt = ScanTransferReceipt(*fields, selection, self.fragment)
        if not receipt._valid():
            raise ValueError("completed scan transfer changed")
        self.receipt = receipt
        self._completion = (emission, receipt)
        self.completed = True


@dataclass(frozen=True)
class BoundScanTransfers:
    receipt: ScanTransferReceipt
    phases: tuple[BoundPreparationTransfers, ...]


def bind_scan_transfers(
    physical: AcceptedPreparationStorage, receipt: ScanTransferReceipt
) -> BoundScanTransfers | None:
    """Revalidate every original access against relocated complete owners.

    No semantic native input is reused as physical authority. Each original
    phase resolves its own sources/destinations, including scalar norms and
    deferred prefix-row reads. Neither allocation nor lifetime is changed here.
    """
    accepted = physical.accepted
    actions = tuple(
        action for action in accepted.actions if action.scan_transfer is receipt
    )
    if len(actions) != 1 or not receipt.matches(accepted, actions[0]):
        return None
    scan, frame = receipt.candidate, accepted.pipeline.frame
    fill = frame.actions[scan.stop_event - 1]
    mma = frame.actions[scan.stop_event]
    if fill.kind != "fill" or fill.stages != scan.stage.group.stages:
        return None
    phase = scan.phases[scan.first_event]
    if scan.phases[fill.event] != phase:
        return None
    spans = [
        (scan.phases[action.event], action.reads, action.writes)
        for action in scan.prelude
        # A scheduled leaf is pending over its actual issue-to-completion
        # span, bound separately against the same physical table.
        if physical.scheduled_body is None or action.kind != "leaf"
    ]
    scan_action = frame.actions[scan.first_event]
    spans.append(
        (
            phase,
            tuple(sorted({*scan_action.reads, *fill.reads} - {scan.buffer.name})),
            (*scan_action.writes, *fill.writes),
        )
    )
    spans.extend(
        (scan.phases[action.event], action.reads, action.writes)
        for action in (*scan.deferred, mma)
    )
    bound = []
    for first, reads, writes in spans:
        transfer = bind_preparation_transfer_span(
            physical, first, first + 1, reads, writes
        )
        if transfer is None:
            return None
        bound.append(transfer)

    views = {view.original.semantic.name: view for view in physical.views}
    buffers = {buffer.name: buffer for buffer in frame.buffers}
    accessed = {name for _, reads, writes in spans for name in (*reads, *writes)}
    for name in accessed:
        view = views.get(name)
        if view is None:
            return None
        original = view.original
        buffer = buffers[name]
        owner = physical.layout.region(view.owner)
        if (
            original.semantic != buffer
            or original.stored_node is not buffer.node
            or original.dtype != buffer.dtype
            or original.shape != buffer.shape
            or view.crop is not None
            or view.owner != original.owner
            or view.declared_bytes != original.byte_size
        ):
            return None
        if name != scan.buffer.name and (
            view.byte_offset != owner.byte_offset + original.member_byte_offset
            or owner.byte_size != frame.layout.region(name).byte_size
            or view.accesses != ((view.byte_offset, original.byte_size),)
        ):
            return None

    # Residual rows retain the original complete logical view and row-preserving
    # XOR mapping, never a densely rebased row mistaken for the original Node.
    residual = views[scan.buffer.name]
    owner = physical.layout.region(residual.owner)
    row_bytes = scan.shape[1] * torch.float32.itemsize
    low, high = min(scan.residual_rows), max(scan.residual_rows)
    if (
        scan.buffer.dtype != torch.float32
        or scan.buffer.shape != scan.shape
        or accepted.scratch_mode != scan.scratch_mode
        or scan.scratch_mode not in ("row_major", "xor")
        or scan.scratch_mode == "xor"
        and xor_swizzle(scan.shape) is None
        or residual.byte_offset + low * row_bytes != owner.byte_offset
        or owner.byte_size != (high - low + 1) * row_bytes
        or residual.accesses != ((owner.byte_offset, owner.byte_size),)
    ):
        return None
    for source, geometry in receipt.reads:
        leaf = accepted.pipeline.prepared_leaves[source.index]
        view = views.get(leaf.name)
        region = preparation_transfer_owner(physical, leaf.name, phase, phase + 1)
        if (
            view is None
            or region is None
            or view.owner != leaf.name
            or view.original.native_group is not None
            or source.tensor != leaf.name
            or source.node is not leaf.node
            or source.row_offset != 0
            or source.full_shape != source.shape
            or source.shape != leaf.proof.tile_shape
            or region.byte_size < math.prod(source.full_shape) * source.dtype.itemsize
            or geometry.ownership != receipt.emission.ownership
            or view.byte_offset != region.byte_offset
            or view.byte_offset % 128
        ):
            return None
    m, n, k = scan.stage.shape
    for allocation, role, shape in (
        (scan.stage.a, "a", (m, k)),
        (scan.stage.b, "b", (n, k)),
    ):
        view = views.get(allocation.name)
        region = preparation_transfer_owner(
            physical, allocation.name, phase, scan.phases[mma.event] + 1
        )
        if (
            view is None
            or region is None
            or view.owner != allocation.name
            or view.original.semantic.kind != role
            or view.original.shape != shape
            or view.original.dtype
            != receipt.plan.operand_dtype(scan.stage.group.stages[0])
            or view.original.native_group is not None
            or region.byte_size != allocation.byte_size
            or view.byte_offset != region.byte_offset
            or view.byte_offset % 128
        ):
            return None
    return BoundScanTransfers(receipt, tuple(bound))
