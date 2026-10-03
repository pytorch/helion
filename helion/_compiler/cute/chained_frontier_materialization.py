"""Accepted identity-copy omissions within an original frontier group.

Semantic transfer proofs permit one relocatable body. Separate physical binding
must validate that body's complete owners and effective spans after packing.
Neither record permits replaying math or discounting an unaccepted operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .chained_frontier_groups import FrontierGroup
from .chained_frontier_groups import plan_frontier_group
from .chained_native_reads import NativeVectorRead
from .chained_native_reads import plan_native_vector_read
from .chained_operand_retention import _InvalidRetention
from .chained_operand_retention import _Revision
from .chained_operand_retention import _revision
from .chained_preparation_transfers import bind_preparation_transfer_span
from .chained_preparation_transfers import preparation_transfer_owner

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_native_read_inputs import NativeReadInput
    from .chained_native_stores import NativeStMatrixStore
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_preparation_storage import AcceptedPreparationStorage
    from .chained_prepared_image_transfers import NativeCropPlan
    from .chained_prepared_image_transfers import NativeIdentityCrop
    from .chained_vector_ownership import VectorOwnership
    from .warp_specialized_plan import SharedBufferRegion


@dataclass(frozen=True)
class FrontierMaterializations:
    revision: _Revision
    pipeline: PreparationPipeline
    group: FrontierGroup
    crops: NativeCropPlan
    boundaries: tuple[tuple[Node, str], ...]
    emitted_ordinals: tuple[int, ...]
    aliases: tuple[tuple[int, NativeIdentityCrop], ...]
    _selection: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        return (
            self.revision,
            self.pipeline,
            self.group,
            self.crops,
            self.boundaries,
            self.emitted_ordinals,
            self.aliases,
        )

    def matches(self) -> bool:
        if (
            self._selection != self._fields()
            or self.group
            != plan_frontier_group(self.pipeline.frame, self.group.first_event)
            or self.revision.frame is not self.pipeline.frame
            or not self.crops.matches(
                self.revision.plan,
                self.crops.retention.revision.frame,
                dict(self.revision.shapes),
            )
        ):
            return False
        try:
            return self.revision == _revision(
                self.revision.plan, self.pipeline.frame, dict(self.revision.shapes)
            )
        except _InvalidRetention:
            return False


def plan_frontier_materializations(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    revision: _Revision,
    crops: NativeCropPlan | None,
    first_event: int,
    boundaries: Mapping[Node, str],
) -> FrontierMaterializations | None:
    """Keep the original group and omit only a proved original typed identity."""
    from .chained_matmul import _operand_domain
    from .chained_matmul import _UnsupportedChain

    group = plan_frontier_group(pipeline.frame, first_event)
    if crops is None or group is None or revision.plan is not plan:
        return None
    aliases = tuple(
        (ordinal, crop)
        for ordinal, buffer in enumerate(group.buffers)
        for crop in crops.crops
        if crop.buffer == buffer
    )
    omitted = {ordinal for ordinal, _ in aliases}
    emitted = tuple(
        index for index in range(len(group.buffers)) if index not in omitted
    )
    if not aliases or len(omitted) != len(aliases) or len(emitted) < 2:
        return None
    for _, crop in aliases:
        if (
            crop.source.node not in boundaries
            or crop.source.publication_event > first_event
            or crop not in crops.crops
        ):
            return None
        try:
            if _operand_domain(
                cg, crop.source.node, ("chain_crop_row", "chain_crop_k"), plan
            ):
                return None
        except _UnsupportedChain:
            return None
    fields = (
        revision,
        pipeline,
        group,
        crops,
        tuple(boundaries.items()),
        emitted,
        aliases,
    )
    result = FrontierMaterializations(*fields, fields)
    return result if result.matches() else None


@dataclass(frozen=True)
class FrontierTransferReceipts:
    materializations: FrontierMaterializations
    ownership: VectorOwnership
    reads: tuple[tuple[NativeReadInput, NativeVectorRead], ...]
    stores: tuple[tuple[int, NativeStMatrixStore], ...]
    _selection: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        return self.materializations, self.ownership, self.reads, self.stores

    def matches(self) -> bool:
        selected = self.materializations
        if self._selection != self._fields() or not selected.matches():
            return False
        shape = selected.group.buffers[0].shape
        if not self.ownership.matches((shape[0], shape[1]), self.ownership.threads):
            return False
        if len({source.node for source, _ in self.reads}) != len(self.reads):
            return False
        if any(
            not source.matches(selected.revision.plan, dict(selected.boundaries))
            or geometry
            != plan_native_vector_read(
                source.full_shape,
                source.shape,
                source.row_offset,
                source.dtype,
                self.ownership,
            )
            for source, geometry in self.reads
        ):
            return False
        ordinals = tuple(ordinal for ordinal, _ in self.stores)
        return (
            len(set(ordinals)) == len(ordinals)
            and all(
                type(ordinal) is int and ordinal in selected.emitted_ordinals
                for ordinal in ordinals
            )
            and all(
                store.matches() and store.ownership == self.ownership
                for _, store in self.stores
            )
        )


@dataclass
class FrontierMaterializationAttempt:
    """A local result slot, populated only after complete original emission."""

    selection: FrontierMaterializations
    receipt: FrontierTransferReceipts | None = None

    def complete(
        self,
        ownership: VectorOwnership,
        reads: tuple[NativeReadInput, ...],
        stores: tuple[tuple[int, NativeStMatrixStore], ...],
    ) -> None:
        if self.receipt is not None:
            raise ValueError("frontier materialization completed twice")
        geometries = tuple(
            plan_native_vector_read(
                source.full_shape,
                source.shape,
                source.row_offset,
                source.dtype,
                ownership,
            )
            for source in reads
        )
        if any(geometry is None for geometry in geometries):
            raise ValueError("emitted native read geometry changed")
        selected = tuple(
            (source, geometry)
            for source, geometry in zip(reads, geometries, strict=True)
            if geometry is not None
        )
        fields = self.selection, ownership, selected, stores
        receipt = FrontierTransferReceipts(*fields, fields)
        if not receipt.matches():
            raise ValueError("completed frontier transfers changed")
        self.receipt = receipt


@dataclass(frozen=True)
class BoundFrontierTransfers:
    receipt: FrontierTransferReceipts
    first: int
    stop: int
    sources: tuple[SharedBufferRegion, ...]
    destinations: tuple[SharedBufferRegion, ...]
    aliases: tuple[tuple[str, str, int], ...]


def bind_frontier_transfers(
    physical: AcceptedPreparationStorage,
    receipt: FrontierTransferReceipts,
) -> BoundFrontierTransfers | None:
    """Independently prove relocated owners, not semantic-witness reuse.

    The body has already been emitted. Every original logical layout and
    complete native owner must survive, and all reads/writes must be disjoint
    throughout the conservative complete original group span.
    """
    selected = receipt.materializations
    pipeline = physical.accepted.pipeline
    if not receipt.matches() or selected.pipeline is not pipeline:
        return None
    actions = tuple(
        action for action in physical.accepted.actions if action.proof is receipt
    )
    if len(actions) != 1:
        return None
    action = actions[0]
    group = selected.group
    if (action.first, action.stop) != (group.first_event, group.stop_event):
        return None
    views = {view.original.semantic.name: view for view in physical.views}
    scan = pipeline.scan_producer
    phases = tuple(range(len(pipeline.frame.actions))) if scan is None else scan.phases
    first, stop = phases[action.first], max(phases[action.first : action.stop]) + 1

    def owner(name: str) -> SharedBufferRegion | None:
        return preparation_transfer_owner(physical, name, first, stop)

    transfer = bind_preparation_transfer_span(
        physical, first, stop, action.reads, action.writes
    )
    if transfer is None:
        return None
    sources, destinations = transfer.sources, transfer.destinations
    retention = pipeline.operand_retention
    for source, geometry in receipt.reads:
        if retention is None or not 0 <= source.index < len(retention.candidates):
            return None
        candidate = retention.candidates[source.index]
        view = views.get(candidate.owner)
        region = owner(candidate.owner)
        if (
            candidate.node is not source.node
            or candidate.logical_modes != (0, 1)
            or candidate.major_mode != "k"
            or candidate.full_shape != source.full_shape
            or candidate.logical_shape != source.shape
            or candidate.row_offset != source.row_offset
            or candidate.dtype != source.dtype
            or view is None
            or region is None
            or region not in sources
            or view.crop is not None
            or view.byte_offset != region.byte_offset
            or region.byte_offset % 128
            or region.alignment != 128
            or region.byte_size != math.prod(source.full_shape) * source.dtype.itemsize
            or view.original.semantic.shape != source.full_shape
            or view.original.semantic.dtype != source.dtype
            or view.original.semantic.kind != candidate.role
            or geometry.ownership != receipt.ownership
        ):
            return None
    for ordinal, geometry in receipt.stores:
        buffer = group.buffers[ordinal]
        bindings = tuple(
            (binding, member)
            for binding in pipeline.prepared_groups
            for member in binding.candidate.members
            if member.buffer == buffer
        )
        if len(bindings) != 1:
            return None
        binding, member = bindings[0]
        candidate = binding.candidate
        view, region = views.get(buffer.name), owner(buffer.name)
        if (
            view is None
            or region is None
            or region not in destinations
            or view.owner != candidate.name
            or view.crop is not None
            or member.logical_modes != (1, 0)
            or geometry.full_shape != candidate.physical_shape
            or geometry.shape != buffer.shape
            or geometry.dtype != buffer.dtype
            or geometry.row_offset != member.row_offset
            or region.byte_offset % 128
            or region.alignment != 128
            or region.byte_size != candidate.byte_size
            or view.byte_offset - region.byte_offset != member.byte_offset
        ):
            return None
    aliases = []
    for ordinal, crop in selected.aliases:
        buffer = group.buffers[ordinal]
        view, region = views.get(buffer.name), owner(crop.source.owner)
        if (
            view is None
            or view.crop != crop
            or region is None
            or view.owner != crop.source.owner
            or region not in sources
            or view.byte_offset != region.byte_offset
            or region.byte_size
            != math.prod(crop.source.full_shape) * crop.source.dtype.itemsize
        ):
            return None
        aliases.append((buffer.name, region.name, crop.source.row_offset))
    return BoundFrontierTransfers(
        receipt, first, stop, tuple(sources), tuple(destinations), tuple(aliases)
    )
