"""Physical preparation-image planning without changing the semantic frame.

This first bounded core only substitutes already-bound original raw-half images.
All original action intervals survive; admitted native groups are indivisible
owners. The result is not emission or allocation authority: a later integration
must rebind every view, commit the same selected emissions, and run the complete
pipeline finalizer. No caller currently installs this plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
from typing import TYPE_CHECKING

from .chained_frontier_materialization import FrontierTransferReceipts
from .chained_frontier_ownership import _group_matches
from .chained_operand_retention import _InvalidRetention
from .chained_operand_retention import _Revision
from .chained_operand_retention import _revision
from .chained_workspace import _reuse_requests
from .warp_specialized_plan import SharedBufferRegion
from .warp_specialized_plan import SharedBufferRequest
from .warp_specialized_plan import SharedMemoryLayoutPlan

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    from torch.fx import Node

    from .chained_broadcast_retention import BoundBroadcastRetention
    from .chained_cache_layout import PointwiseCacheLayouts
    from .chained_frontier_materialization import BoundFrontierTransfers
    from .chained_leaf_schedule_emission import AcceptedLeafSchedule
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pipeline_storage import PipelineStorage
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_preparation_actions import AcceptedPreparation
    from .chained_preparation_frame import PreparationBuffer
    from .chained_preparation_frame import PreparationFrame
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_preparation_reads import IslandExportLease
    from .chained_preparation_transfers import BoundPreparationTransfers
    from .chained_prepared_groups import PreparedGroupBinding
    from .chained_prepared_image_emission import BoundRawWidenings
    from .chained_prepared_image_transfers import NativeIdentityCrop
    from .chained_scan_transfers import BoundScanTransfers
    from .chained_scratch_layout import ScratchLayouts
    from .chained_vector_stage import VectorStaging


@dataclass(frozen=True)
class PreparationStorageView:
    """Original semantic image and its explicitly typed physical binding.

    Native members retain their complete owner and member byte offset; this
    record does not construct a dense member layout. Raw images identify the
    original half Node separately from the semantic Float32 widening Node.
    """

    semantic: PreparationBuffer
    stored_node: Node | None
    dtype: torch.dtype
    shape: tuple[int, ...]
    owner: str
    byte_offset: int
    byte_size: int
    live_from: int
    live_until: int
    native_group: PreparedGroupBinding | None
    member_byte_offset: int


@dataclass(frozen=True)
class PreparationStorage:
    revision: _Revision
    raw: BoundRawWidenings | None
    prepared_groups: tuple[PreparedGroupBinding, ...]
    layout: SharedMemoryLayoutPlan
    views: tuple[PreparationStorageView, ...]
    capacity_bytes: int
    _selection: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        return (
            self.revision,
            self.raw,
            self.prepared_groups,
            self.layout,
            self.views,
            self.capacity_bytes,
        )

    def matches(
        self,
        plan: ChainedMatmulPlan,
        frame: PreparationFrame,
        shapes: Mapping[Node, tuple[int, ...]],
    ) -> bool:
        """Recheck this planning result, not permission to emit discounted code."""
        if (
            self._selection != self._fields()
            or self.revision.plan is not plan
            or self.revision.frame is not frame
        ):
            return False
        fresh = plan_preparation_storage(
            plan,
            frame,
            shapes,
            raw=self.raw,
            prepared_groups=self.prepared_groups,
            capacity_bytes=self.capacity_bytes,
        )
        return fresh is not None and fresh._fields() == self._fields()


def plan_preparation_storage(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    shapes: Mapping[Node, tuple[int, ...]],
    *,
    raw: BoundRawWidenings | None = None,
    prepared_groups: tuple[PreparedGroupBinding, ...] = (),
    capacity_bytes: int,
) -> PreparationStorage | None:
    """Plan original intervals with typed raw images and whole native owners.

    ``prepared_groups`` must be the caller's complete, already-admitted set;
    this function revalidates it but cannot discover missing late admissions.
    Empty raw selection preserves original offsets and allocation exactly;
    native member requests are represented by their complete union owner.
    No scan phase, island omission, crop elimination, shorter lease, or protocol
    is inferred.
    A capacity failure returns None, never the original discounted frame.
    """
    if type(capacity_bytes) is not int or capacity_bytes < 0:
        return None
    try:
        revision = _revision(plan, frame, shapes)
    except _InvalidRetention:
        return None
    if raw is not None and (
        raw.frame is not frame or not raw.bindings or not raw.matches(plan)
    ):
        return None
    regions = {region.name: region for region in frame.layout.regions}
    buffers = {buffer.name: buffer for buffer in frame.buffers}
    if any(
        any(type(size) is not int or size <= 0 for size in buffer.shape)
        or regions[name].byte_size < math.prod(buffer.shape) * buffer.dtype.itemsize
        for name, buffer in buffers.items()
    ):
        return None
    grouped = {}
    group_names = set()
    for binding in prepared_groups:
        if (
            not _group_matches(frame, binding)
            or binding.candidate.group not in (plan.contraction_groups or ())
            or binding.candidate.name in group_names
            or binding.candidate.name in regions
        ):
            return None
        group_names.add(binding.candidate.name)
        for member in binding.candidate.members:
            if member.buffer.name in grouped:
                return None
            grouped[member.buffer.name] = (binding, member)
    replacements = (
        {}
        if raw is None
        else {item.transfer.buffer.name: item for item in raw.bindings}
    )
    if (
        len(replacements) != (0 if raw is None else len(raw.bindings))
        or len({item.buffer.name for item in replacements.values()})
        != len(replacements)
        or replacements.keys() & grouped.keys()
        or any(
            item.buffer.name in regions or item.buffer.name in group_names
            for item in replacements.values()
        )
    ):
        return None
    requests = []
    requested_groups = set()
    # The relation stores original semantic owner -> physical owner + offset.
    relation = {}
    for region in frame.layout.regions:
        name = region.name
        if name in grouped:
            binding, member = grouped[name]
            candidate = binding.candidate
            if candidate.name not in requested_groups:
                requests.append(
                    SharedBufferRequest(
                        candidate.name,
                        candidate.byte_size,
                        128,
                        candidate.live_from,
                        candidate.live_until,
                    )
                )
                requested_groups.add(candidate.name)
            relation[name] = (candidate.name, member.byte_offset)
        else:
            replacement = replacements.get(name)
            owner = name if replacement is None else replacement.buffer.name
            size = (
                region.byte_size
                if replacement is None
                else (
                    math.prod(replacement.buffer.shape)
                    * replacement.buffer.dtype.itemsize
                    + 127
                )
                // 128
                * 128
            )
            requests.append(
                SharedBufferRequest(
                    owner, size, 128, region.live_from, region.live_until
                )
            )
            relation[name] = (owner, 0)
    if raw is None:
        # No raw replacement means no new physical plan is selected. Preserve
        # the original offsets and owner-sized accounting, including groups.
        original_offsets = {
            request.name: (
                next(
                    group.byte_offset
                    for group in prepared_groups
                    if group.candidate.name == request.name
                )
                if request.name in group_names
                else regions[request.name].byte_offset
            )
            for request in requests
        }
        layout = SharedMemoryLayoutPlan(
            tuple(
                SharedBufferRegion(
                    request.name,
                    original_offsets[request.name],
                    request.byte_size,
                    request.live_from,
                    request.live_until,
                    request.alignment,
                )
                for request in requests
            ),
            frame.layout.allocated_bytes,
        )
        if not prepared_groups:
            layout = frame.layout
    else:
        layout = _reuse_requests(tuple(requests))
    if layout.allocated_bytes > capacity_bytes or any(
        left.overlaps_storage(right) and left.overlaps_lifetime(right)
        for index, left in enumerate(layout.regions)
        for right in layout.regions[index + 1 :]
    ):
        return None
    views = []
    for buffer in frame.buffers:
        name = buffer.name
        owner, member_offset = relation[name]
        allocation = layout.region(owner)
        original = regions[name]
        replacement = replacements.get(name)
        stored = buffer if replacement is None else replacement.buffer
        size = original.byte_size if replacement is None else allocation.byte_size
        if (
            member_offset < 0
            or member_offset + size > allocation.byte_size
            or allocation.live_from > original.live_from
            or allocation.live_until < original.live_until
        ):
            return None
        views.append(
            PreparationStorageView(
                buffer,
                stored.node,
                stored.dtype,
                stored.shape,
                owner,
                allocation.byte_offset + member_offset,
                size,
                original.live_from,
                original.live_until,
                grouped[name][0] if name in grouped else None,
                member_offset,
            )
        )
    values = tuple(views)
    fields = (revision, raw, prepared_groups, layout, values, capacity_bytes)
    return PreparationStorage(
        revision, raw, prepared_groups, layout, values, capacity_bytes, fields
    )


@dataclass(frozen=True)
class PhysicalPreparationView:
    """A declared logical view and the actual allocated byte-access spans.

    Residual scans keep their full logical layout, with a separately checked
    origin. Native crops retain their complete source owner, not a dense slice.
    """

    original: PreparationStorageView
    owner: str
    byte_offset: int
    declared_bytes: int
    accesses: tuple[tuple[int, int], ...]
    crop: NativeIdentityCrop | None = None


@dataclass(frozen=True)
class AcceptedPreparationStorage:
    base: PreparationStorage
    accepted: AcceptedPreparation
    layout: SharedMemoryLayoutPlan
    views: tuple[PhysicalPreparationView, ...]
    omitted: tuple[str, ...]
    capacity_bytes: int
    _selection: tuple[object, ...]
    export_leases: tuple[IslandExportLease, ...] = ()
    scheduled_body: AcceptedLeafSchedule | None = None

    def _fields(self) -> tuple[object, ...]:
        fields = (
            self.base,
            self.accepted,
            self.layout,
            self.views,
            self.omitted,
            self.capacity_bytes,
            self.export_leases,
        )
        return fields if self.scheduled_body is None else (*fields, self.scheduled_body)

    def matches(
        self,
        plan: ChainedMatmulPlan,
        pipeline: PreparationPipeline,
        shapes: Mapping[Node, tuple[int, ...]],
    ) -> bool:
        if (
            self._selection != self._fields()
            or not self.accepted.matches(plan, pipeline, shapes)
            or not self.base.matches(plan, pipeline.frame, shapes)
        ):
            return False
        fresh = (
            plan_accepted_preparation_storage(
                plan, pipeline, self.accepted, capacity_bytes=self.capacity_bytes
            )
            if self.scheduled_body is None
            else plan_scheduled_preparation_storage(plan, pipeline, self.scheduled_body)
        )
        return fresh is not None and fresh._fields() == self._fields()


def _place_active_requests(
    requests: tuple[SharedBufferRequest, ...],
    minimum_offsets: Mapping[str, int],
    declared_ends: Mapping[str, int] | None = None,
) -> SharedMemoryLayoutPlan:
    from .chained_preparation_placement import place_preparation_storage

    return place_preparation_storage(
        requests, minimum_offsets, {} if declared_ends is None else declared_ends
    )


def plan_accepted_preparation_storage(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    accepted: AcceptedPreparation,
    *,
    capacity_bytes: int,
) -> AcceptedPreparationStorage | None:
    """Pack ONLY the exact successful body, preserving all remaining leases.

    This is still not executable binding authority. The physical binder must
    revalidate every native/typed view before the complete finalizer charges it.
    """
    from .chained_preparation_actions import AcceptedPreparation
    from .chained_preparation_reads import plan_island_export_leases
    from .chained_prepared_image_transfers import NativeIdentityCrop
    from .chained_register_binding import BoundPreparationIsland
    from .chained_scan_producer import ScanProducer

    if (
        type(capacity_bytes) is not int
        or capacity_bytes < 0
        or not isinstance(accepted, AcceptedPreparation)
    ):
        return None
    frame = pipeline.frame
    shapes = dict(accepted.revision.shapes)
    if not accepted.matches(plan, pipeline, shapes):
        return None
    export_leases = plan_island_export_leases(accepted)
    if export_leases is None:
        return None
    leases = {lease.export.name: lease for lease in export_leases}
    base = plan_preparation_storage(
        plan,
        frame,
        shapes,
        raw=accepted.raw,
        prepared_groups=pipeline.prepared_groups,
        capacity_bytes=max(
            frame.layout.allocated_bytes,
            sum(region.byte_size for region in frame.layout.regions),
        ),
    )
    if base is None:
        return None
    scans = tuple(
        action.proof
        for action in accepted.actions
        if isinstance(action.proof, ScanProducer)
    )
    if len(scans) > 1 or (
        pipeline.scan_producer is not None and scans != (pipeline.scan_producer,)
    ):
        return None
    scan = scans[0] if scans else None
    scale = len(frame.actions) + 2 if scan is not None else 1
    phases = scan.phases if scan is not None else tuple(range(len(frame.actions)))
    effective = frame.layout.regions if scan is None else scan.regions
    intervals = {
        region.name: (region.live_from, region.live_until) for region in effective
    }
    originals = {view.semantic.name: view for view in base.views}
    omitted = {name for action in accepted.actions for name in action.omitted}
    crops = {
        action.proof.buffer.name: action.proof
        for action in accepted.actions
        if isinstance(action.proof, NativeIdentityCrop)
    }
    crops.update(
        (crop.buffer.name, crop)
        for action in accepted.actions
        if isinstance(action.proof, FrontierTransferReceipts)
        for _, crop in action.proof.materializations.aliases
    )
    # Crop materialization disappears, but its original frontier still has a
    # typed view of the COMPLETE retained owner. Other omitted values may not
    # escape the accepted register island.
    if any(
        view.semantic.node in {image.node for image in frame.cut.images}
        for name, view in originals.items()
        if name in omitted and name not in crops
    ):
        return None
    omitted -= crops.keys()
    owner_for = {name: view.owner for name, view in originals.items()}
    owner_for.update(
        (name, originals[crop.source.owner].owner) for name, crop in crops.items()
    )
    removed_owners = {originals[name].owner for name in (*omitted, *crops)}
    active_owners = {
        view.owner
        for name, view in originals.items()
        if name not in omitted and name not in crops
    }
    if removed_owners & active_owners:
        # No partial omission of an admitted full-native union.
        return None
    required_reads: dict[str, int] = {}
    early_writes: dict[str, int] = {}
    published_owners = {item.candidate.owner for item in accepted.island_publications}
    for action in accepted.actions:
        first, last = (
            phases[action.first],
            max(phases[index] for index in range(action.first, action.stop)) + 1,
        )
        for name in action.reads:
            if name in omitted or name not in owner_for:
                return None
            owner = owner_for[name]
            if action.kind in ("island", "frontier") or name in published_owners:
                required_reads[owner] = max(required_reads.get(owner, 0), last)
        for name in action.writes:
            if name in omitted or name not in owner_for:
                return None
            owner = owner_for[name]
            if action.kind in ("island", "frontier"):
                early_writes[owner] = min(early_writes.get(owner, first), first)
        if isinstance(action.proof, BoundPreparationIsland):
            expected = (
                {export.name for export in action.proof.exports}
                if action.island_publication is None
                else {action.island_publication.candidate.owner}
            )
            if expected != set(action.writes):
                return None
    group_owners = {
        binding.candidate.name: binding for binding in pipeline.prepared_groups
    }
    requests = []
    minimum_offsets = {}
    origin_bias = {}
    for region in base.layout.regions:
        if region.name in removed_owners:
            continue
        names = tuple(
            name for name, view in originals.items() if view.owner == region.name
        )
        rows = None
        if scan is not None and scan.buffer.name in names:
            rows = tuple(
                item
                for item in effective
                if item.name.startswith(f"{scan.buffer.name}:row:")
            )
            if len(rows) != len(scan.residual_rows) or not rows:
                return None
            low, high = min(scan.residual_rows), max(scan.residual_rows)
            width = scan.shape[1] * 4
            size = (high - low + 1) * width
            origin_bias[region.name] = -low * width
            minimum_offsets[region.name] = low * width
            first = min(item.live_from for item in rows)
            last = max(item.live_until for item in rows)
        else:
            if any(name not in intervals for name in names):
                return None
            size = region.byte_size
            first = min(intervals[name][0] for name in names)
            last = max(intervals[name][1] for name in names)
        if region.name in group_owners:
            group = group_owners[region.name].candidate
            first = min(first, group.live_from * scale)
            last = max(last, group.live_until * scale)
        first = min(first, early_writes.get(region.name, first))
        last = max(last, required_reads.get(region.name, last))
        if any(owner_for[name] == region.name for name in crops):
            # Initial crop policy retains the entire native image until the
            # original READY/EMPTY ownership transfer, never just MMA issue.
            last = max(last, len(frame.actions) * scale)
        if region.name in leases:
            lease = leases[region.name]
            if (
                names != (lease.export.name,)
                or region.name in group_owners
                or required_reads.get(region.name, 0) > lease.stop
                or first >= lease.stop
            ):
                return None
            # Keep the complete original/accepted early-zero reservation. Only
            # the last reader's proved full-stage completion can reduce its end.
            last = min(last, lease.stop)
        requests.append(SharedBufferRequest(region.name, size, 128, first, last))
    declared_ends: dict[str, int] = {}
    for name, view in originals.items():
        if name not in omitted and name not in crops:
            owner = owner_for[name]
            declared_ends[owner] = max(
                declared_ends.get(owner, 0),
                view.member_byte_offset + origin_bias.get(owner, 0) + view.byte_size,
            )
    layout = _place_active_requests(tuple(requests), minimum_offsets, declared_ends)
    views = []
    declared_end = layout.allocated_bytes
    for name, view in originals.items():
        if name in omitted:
            continue
        crop = crops.get(name)
        owner = owner_for[name]
        allocation = layout.region(owner)
        if crop is not None:
            offset = allocation.byte_offset
            size = allocation.byte_size
            accesses = ((offset, size),)
        else:
            offset = (
                allocation.byte_offset
                + view.member_byte_offset
                + origin_bias.get(owner, 0)
            )
            size = view.byte_size
            if owner in origin_bias:
                size = frame.layout.region(name).byte_size
                accesses = ((allocation.byte_offset, allocation.byte_size),)
            else:
                accesses = ((offset, size),)
        if offset < 0 or any(
            start < allocation.byte_offset or start + length > allocation.byte_end
            for start, length in accesses
        ):
            return None
        declared_end = max(declared_end, offset + size)
        views.append(PhysicalPreparationView(view, owner, offset, size, accesses, crop))
    layout = replace(layout, allocated_bytes=(declared_end + 127) // 128 * 128)
    if layout.allocated_bytes > capacity_bytes:
        return None
    values = tuple(views)
    removed = tuple(name for name in originals if name in omitted)
    fields = (base, accepted, layout, values, removed, capacity_bytes, export_leases)
    result = AcceptedPreparationStorage(
        base, accepted, layout, values, removed, capacity_bytes, fields, export_leases
    )
    if accepted.island_publications:
        from .chained_island_publication import validate_island_physical

        if not validate_island_physical(result):
            return None
    return result


@dataclass
class _BodyConsumption:
    consumed: bool = False
    finalized: PipelineStorage | None = None


def plan_scheduled_preparation_storage(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    body: AcceptedLeafSchedule,
) -> AcceptedPreparationStorage | None:
    """Use the actual extended leases, without changing any physical byte view."""
    original = body.schedule.physical
    if original.scheduled_body is not None or not body.matches(plan, pipeline):
        return None
    layout = replace(original.layout, regions=body.schedule.regions)
    result = replace(original, layout=layout, scheduled_body=body)
    return replace(result, _selection=result._fields())


@dataclass(frozen=True)
class BoundPreparationStorage:
    """One physical table for views, charging, and an already-emitted body.

    Binding does not install the body. ``consume_body`` additionally requires
    the complete pipeline finalizer to reference this exact binding. The body
    has no second lowering or fallback after the successful choices were sealed.
    """

    physical: AcceptedPreparationStorage
    cache_layouts: PointwiseCacheLayouts | None
    cache_mode: str | None
    _state: _BodyConsumption
    _selection: tuple[object, ...]
    frontier_transfers: tuple[BoundFrontierTransfers, ...] = ()
    scan_transfers: tuple[BoundScanTransfers, ...] = ()
    leaf_transfers: tuple[BoundPreparationTransfers, ...] = ()
    broadcast_transfers: tuple[BoundBroadcastRetention, ...] = ()

    @property
    def stride(self) -> int:
        return self.physical.layout.allocated_bytes

    def matches(self, plan: ChainedMatmulPlan, pipeline: PreparationPipeline) -> bool:
        return (
            self._selection
            == (
                self.physical,
                self.cache_layouts,
                self.cache_mode,
                self._state,
                self.frontier_transfers,
                self.scan_transfers,
                self.leaf_transfers,
                self.broadcast_transfers,
            )
            and self._state is self._selection[3]
            and (None if self.cache_layouts is None else self.cache_layouts.mode)
            == self.cache_mode
            and type(self._state.consumed) is bool
            and self.physical.matches(
                plan, pipeline, dict(self.physical.accepted.revision.shapes)
            )
            and _frontier_bindings(self.physical) == self.frontier_transfers
            and _scan_bindings(self.physical) == self.scan_transfers
            and _leaf_bindings(self.physical) == self.leaf_transfers
            and _broadcast_bindings(self.physical) == self.broadcast_transfers
        )

    def view_lines(self, byte_pointer: str, scratch: ScratchLayouts) -> list[str]:
        """Bind every surviving image using its original typed logical view."""
        from .chained_preparation_actions import workspace_name
        from .chained_prepared_values import buffer_layout
        from .chained_prepared_values import storage_dtype
        from .chained_tcgen05 import _layout

        accepted = self.physical.accepted
        plan, pipeline = accepted.revision.plan, accepted.pipeline
        if not self.matches(plan, pipeline) or scratch.mode != accepted.scratch_mode:
            raise ValueError("physical preparation binding changed")
        views = {view.original.semantic.name: view for view in self.physical.views}
        lines = []
        for name in accepted.workspaces:
            view = views[name]
            if view.original.semantic.kind not in ("a", "b") or view.crop is not None:
                raise ValueError("symbolic workspace is not an original operand arena")
            lines.append(
                f"{workspace_name(name)} = cute.recast_ptr({byte_pointer} + "
                f"{view.byte_offset}, dtype=cutlass.BFloat16)"
            )
        grouped = {}
        for binding in pipeline.prepared_groups:
            candidate = binding.candidate
            if any(member.buffer.name not in views for member in candidate.members):
                raise ValueError("physical plan omitted part of a native group")
            region = self.physical.layout.region(candidate.name)
            lines.extend(
                [
                    f"{candidate.name}_ptr = cute.recast_ptr({byte_pointer} + {region.byte_offset}, dtype={storage_dtype(candidate.dtype)})",
                    *candidate.native_layout,
                ]
            )
            grouped.update((member.buffer.name, member) for member in candidate.members)
        native = {item.buffer.name: item for item in pipeline.prepared_operands}
        raw = (
            {}
            if accepted.raw is None
            else {item.transfer.buffer.name: item for item in accepted.raw.bindings}
        )
        crop_owners = set()
        for buffer in pipeline.frame.buffers:
            view = views.get(buffer.name)
            if view is None or buffer.kind in ("a", "b"):
                continue
            crop = view.crop
            if crop is not None:
                source = crop.source
                prefix = f"chain_preparation_owner_{view.owner}"
                if view.owner not in crop_owners:
                    dtype = storage_dtype(source.dtype)
                    lines.extend(
                        [
                            f"{prefix}_ptr = cute.recast_ptr({byte_pointer} + {view.byte_offset}, dtype={dtype})",
                            *_layout(prefix, source.full_shape, 1, dtype),
                        ]
                    )
                    crop_owners.add(view.owner)
                lines.extend(crop.alias_lines(prefix, buffer.name))
                continue
            item = raw.get(buffer.name)
            if item is not None:
                # The original FP32 Node never names a half-precision tensor.
                # Only its original raw-half predecessor gets this new view.
                lines.extend(
                    replace(
                        item,
                        region=replace(
                            item.region,
                            byte_offset=view.byte_offset,
                            byte_size=view.declared_bytes,
                        ),
                    ).view_lines(byte_pointer)
                )
                continue
            pointer = f"cute.recast_ptr({byte_pointer} + {view.byte_offset}, dtype={storage_dtype(buffer.dtype)})"
            if buffer.kind == "leaf":
                leaf = next(
                    item
                    for item in pipeline.prepared_leaves
                    if item.node is buffer.node
                )
                lines.extend(leaf.view(byte_pointer, view.byte_offset))
            elif buffer.name in grouped:
                lines.extend(
                    [
                        f"{buffer.name}_ptr = {pointer}",
                        *grouped[buffer.name].native_layout,
                    ]
                )
            elif buffer.name in native:
                lines.extend(
                    [
                        f"{buffer.name}_ptr = {pointer}",
                        *native[buffer.name].native_layout,
                    ]
                )
            else:
                lines.append(
                    f"{buffer.name} = cute.make_tensor({pointer}, "
                    f"{buffer_layout(buffer, scratch, self.cache_layouts)})"
                )
        return lines

    def consume_body(
        self,
        storage: PipelineStorage,
        vector: VectorStaging,
        unroll: BoundedProducerUnroll,
    ) -> list[str]:
        """Commit the successful body once, after full same-table charging."""
        accepted = self.physical.accepted
        if (
            not self.matches(accepted.revision.plan, accepted.pipeline)
            or self._state.finalized is not storage
            or storage.preparation is not self
            or dict(storage.allocations).get("frames")
            != accepted.pipeline.slots * self.stride
            or self._state.consumed
            or (vector.enabled, vector.group_enabled, vector.async_enabled)
            != (
                accepted.vector_state[0],
                accepted.vector_state[2],
                accepted.vector_state[4],
            )
            or unroll.factor != accepted.unroll_state[0]
        ):
            raise ValueError("preparation body lacks its unconsumed final allocation")
        scheduled = self.physical.scheduled_body
        lines = list(accepted.lines if scheduled is None else scheduled.lines)
        vector.activated |= accepted.vector_state[1]
        vector.group_activated |= accepted.vector_state[3]
        vector.async_activated |= accepted.vector_state[5]
        unroll.activated |= accepted.unroll_state[1]
        unroll.eliminated |= accepted.unroll_state[2]
        self._state.consumed = True
        return lines

    def _finalize(self, storage: PipelineStorage) -> bool:
        """Called only by the complete allocator after its capacity check."""
        if (
            self._state.finalized is not None
            or self._state.consumed
            or storage.preparation is not self
        ):
            return False
        self._state.finalized = storage
        return True


def _frontier_bindings(
    physical: AcceptedPreparationStorage,
) -> tuple[BoundFrontierTransfers, ...] | None:
    from .chained_frontier_materialization import FrontierTransferReceipts
    from .chained_frontier_materialization import bind_frontier_transfers

    result = []
    for action in physical.accepted.actions:
        if isinstance(action.proof, FrontierTransferReceipts):
            bound = bind_frontier_transfers(physical, action.proof)
            if bound is None:
                return None
            result.append(bound)
    return tuple(result)


def _scan_bindings(
    physical: AcceptedPreparationStorage,
) -> tuple[BoundScanTransfers, ...] | None:
    from .chained_scan_transfers import bind_scan_transfers

    result = []
    for action in physical.accepted.actions:
        if action.scan_transfer is not None:
            bound = bind_scan_transfers(physical, action.scan_transfer)
            if bound is None:
                return None
            result.append(bound)
    return tuple(result)


def _leaf_bindings(
    physical: AcceptedPreparationStorage,
) -> tuple[BoundPreparationTransfers, ...] | None:
    from .chained_preparation_transfers import bind_preparation_transfer_span

    body = physical.scheduled_body
    if body is None:
        return ()
    result = []
    for leaf in body.schedule.leaves:
        first = body.schedule.actions[leaf.issue_before].phase
        stop = body.schedule.actions[leaf.complete_before].phase + 1
        bound = bind_preparation_transfer_span(
            physical, first, stop, (), (leaf.leaf.name,)
        )
        if bound is None:
            return None
        result.append(bound)
    return tuple(result)


def _broadcast_bindings(
    physical: AcceptedPreparationStorage,
) -> tuple[BoundBroadcastRetention, ...] | None:
    from .chained_broadcast_retention import bind_broadcast_retention

    return bind_broadcast_retention(physical)


def bind_preparation_storage(
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    physical: AcceptedPreparationStorage,
    *,
    cache_layouts: PointwiseCacheLayouts | None = None,
) -> BoundPreparationStorage | None:
    """Validate original layout owners before exposing one physical binding."""
    if not physical.matches(plan, pipeline, dict(physical.accepted.revision.shapes)):
        return None
    original = {buffer.name: buffer for buffer in pipeline.frame.buffers}
    views = {view.original.semantic.name: view for view in physical.views}
    for operand in pipeline.prepared_operands:
        if (
            operand.buffer != original[operand.buffer.name]
            or operand.region != pipeline.frame.layout.region(operand.buffer.name)
            or operand.buffer.name not in views
        ):
            return None
    for leaf in pipeline.prepared_leaves:
        view = views.get(leaf.name)
        if view is None or view.original.semantic.node is not leaf.node:
            return None
    if any(
        name not in views or views[name].byte_offset % 128
        for name in physical.accepted.workspaces
    ):
        return None
    if any(
        view.crop is not None
        and (
            view.crop.source.major_mode != "k"
            or view.crop.source.logical_modes != (0, 1)
            or math.prod(view.crop.source.full_shape) * view.crop.source.dtype.itemsize
            != physical.layout.region(view.owner).byte_size
        )
        for view in physical.views
    ):
        return None
    frontier_transfers = _frontier_bindings(physical)
    if frontier_transfers is None:
        return None
    scan_transfers = _scan_bindings(physical)
    if scan_transfers is None:
        return None
    leaf_transfers = _leaf_bindings(physical)
    if leaf_transfers is None:
        return None
    broadcast_transfers = _broadcast_bindings(physical)
    if broadcast_transfers is None:
        return None
    state = _BodyConsumption()
    mode = None if cache_layouts is None else cache_layouts.mode
    return BoundPreparationStorage(
        physical,
        cache_layouts,
        mode,
        state,
        (
            physical,
            cache_layouts,
            mode,
            state,
            frontier_transfers,
            scan_transfers,
            leaf_transfers,
            broadcast_transfers,
        ),
        frontier_transfers,
        scan_transfers,
        leaf_transfers,
        broadcast_transfers,
    )
