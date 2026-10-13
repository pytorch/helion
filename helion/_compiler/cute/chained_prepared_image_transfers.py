"""Typed alternatives to materializing an original preparation image.

These records prove graph identity and storage requirements, not asynchronous
publication. Native crops keep their complete native owner alive. Raw transfers
keep the original half-precision load as the stored value and the original
Float32 conversion as a recurrence expression. Neither changes the semantic
frame, omits an allocation, nor licenses a producer/consumer overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...language import memory_ops
from .chained_operand_retention import OperandRetentionCandidate
from .chained_operand_retention import OperandRetentionPlan
from .chained_operand_retention import _InvalidRetention
from .chained_operand_retention import _Revision
from .chained_operand_retention import _revision
from .chained_operand_retention import discover_operand_retention
from .chained_operand_retention import plan_operand_retention_frame
from .contraction_region import _domain
from .warp_specialized_plan import SharedBufferRequest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_frame import PreparationBuffer
    from .chained_preparation_frame import PreparationFrame
    from .chained_prepared_groups import PreparedGroupBinding


@dataclass(frozen=True)
class NativeIdentityCrop:
    """An exact frontier SSA image already present in a full native owner."""

    buffer: PreparationBuffer
    source: OperandRetentionCandidate
    consumers: tuple[int, ...]

    def alias_lines(self, full_tensor: str, name: str) -> tuple[str, ...]:
        """Crop the full layout, including its outer K-panel stride.

        This method is geometry emission only. Use an unmodified, matching
        NativeCropPlan and revalidate producer/domain/readiness proofs before
        replacing the original publication or issuing any native descriptor.
        """
        rows, columns = self.source.logical_shape
        return (
            (
                f"{name} = cute.local_tile({full_tensor}, ({rows}, {columns}), "
                f"({self.source.row_offset // rows}, 0))"
            ),
        )


def discover_native_identity_crops(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    shapes: Mapping[Node, tuple[int, ...]],
) -> tuple[NativeIdentityCrop, ...] | None:
    """Find exact, untransformed singleton physical-B recurrence images.

    Same-valued expressions are not identity. An image with even one other
    recurrence consumer remains on the ordinary path. This bounded policy does
    not strip a mask, cast, or view from the recurrence operand.
    """
    available = discover_operand_retention(plan, frame, shapes)
    if available is None:
        return None
    region = plan.region
    assert region is not None
    groups = plan.contraction_groups
    if groups is None:
        return ()
    if tuple(index for group in groups for index in group.stages) != tuple(
        range(len(plan.dots))
    ) or any(len(group.stages) != len(group.geometries) for group in groups):
        return None
    recurrent = set(frame.cut.recurrence)
    images = {image.node: image for image in frame.cut.images}
    result = []
    for source in available:
        buffers = tuple(
            buffer
            for buffer in frame.buffers
            if buffer.kind == "frontier" and buffer.node is source.node
        )
        image = images.get(source.node)
        if len(buffers) != 1 or image is None:
            continue
        buffer = buffers[0]
        rows, columns = source.logical_shape
        if (
            source.logical_modes != (0, 1)
            or rows <= 0
            or columns <= 0
            or source.row_offset < 0
            or source.row_offset % rows
            or source.full_shape[0] % rows
            or source.row_offset + rows > source.full_shape[0]
            or source.full_shape[1] != columns
            or buffer.shape != source.logical_shape
            or buffer.dtype != source.dtype
            or image.dtype != source.dtype
            or image.logical_domain != _domain(source.node.meta["val"])
        ):
            continue
        users = tuple(user for user in source.node.users if user in recurrent)
        if not users or image.consumers != users:
            continue
        consumers = []
        for group in groups:
            if len(group.stages) != 1:
                continue
            index = group.stages[0]
            spec = region.contractions[index]
            geometry = group.geometries[0]
            operand_index, coords = geometry.operand("b", "row", "k")
            if (
                spec.node in users
                and (spec.lhs, spec.rhs)[operand_index] is source.node
                and coords == ("row", "k")
                and geometry.physical[1:] == source.logical_shape
                and spec.operand_dtypes == (source.dtype, source.dtype)
                and index not in plan.warp_mma_stages
                and spec.accumulator is not source.node
                and (spec.lhs, spec.rhs)[1 - operand_index] is not source.node
            ):
                consumers.append(index)
        if tuple(region.contractions[i].node for i in consumers) != users:
            continue
        result.append(NativeIdentityCrop(buffer, source, tuple(consumers)))
    return tuple(result)


@dataclass(frozen=True)
class NativeCropPlan:
    """Conservatively rebound frame; no original owner is removed or shrunk."""

    retention: OperandRetentionPlan
    crops: tuple[NativeIdentityCrop, ...]
    _selection: tuple[object, ...]

    def matches(
        self,
        plan: ChainedMatmulPlan,
        frame: PreparationFrame,
        shapes: Mapping[Node, tuple[int, ...]],
    ) -> bool:
        if self._selection != (self.retention, self.crops):
            return False
        retained = self.retention
        available = discover_native_identity_crops(plan, frame, shapes)
        return (
            retained.matches(
                plan,
                frame,
                shapes,
                prepared_groups=retained.original_groups,
                reservations=retained.reservations,
            )
            and available is not None
            and self.crops == tuple(item for item in available if item in self.crops)
            and all(
                retained.frame.layout.region(item.source.owner).live_until
                == len(frame.actions)
                for item in self.crops
            )
        )


def plan_native_identity_crops(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    shapes: Mapping[Node, tuple[int, ...]],
    candidates: tuple[OperandRetentionCandidate, ...],
    crops: tuple[NativeIdentityCrop, ...],
    *,
    prepared_groups: tuple[PreparedGroupBinding, ...] = (),
    reservations: tuple[SharedBufferRequest, ...] = (),
    capacity_bytes: int,
) -> NativeCropPlan | None:
    """Extend complete owners through READY with the existing retention packer.

    The caller supplies its late-admitted retention candidates and every
    authoritative early-write/group reservation. A capacity failure is a real
    failure, never an excuse to return the original discounted frame.
    """
    available = discover_native_identity_crops(plan, frame, shapes)
    if (
        available is None
        or not crops
        or crops != tuple(item for item in available if item in crops)
        or any(item.source not in candidates for item in crops)
    ):
        return None
    owners = tuple(dict.fromkeys(item.source.owner for item in crops))
    extensions = tuple(
        SharedBufferRequest(
            owner,
            frame.layout.region(owner).byte_size,
            frame.layout.region(owner).alignment,
            frame.layout.region(owner).live_from,
            len(frame.actions),
        )
        for owner in owners
    )
    retained = plan_operand_retention_frame(
        plan,
        frame,
        candidates,
        shapes,
        prepared_groups=prepared_groups,
        reservations=(*reservations, *extensions),
        capacity_bytes=capacity_bytes,
    )
    if retained is None:
        return None
    return NativeCropPlan(retained, crops, (retained, crops))


@dataclass(frozen=True)
class RawWideningTransfer:
    """Store source bits, then execute the original same-coordinate widening."""

    buffer: PreparationBuffer
    source: Node
    widening: Node
    consumer: Node
    dtype: torch.dtype
    shape: tuple[int, ...]
    publication_event: int
    revision: _Revision

    def matches(
        self,
        plan: ChainedMatmulPlan,
        frame: PreparationFrame,
        shapes: Mapping[Node, tuple[int, ...]],
    ) -> bool:
        available = discover_raw_widening_transfers(plan, frame, shapes)
        return available is not None and self in available


def discover_raw_widening_transfers(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    shapes: Mapping[Node, tuple[int, ...]],
) -> tuple[RawWideningTransfer, ...] | None:
    """Recognize exclusive masked-load → Float32 frontier conversion chains.

    The load itself (including its complete indices and mask) is retained. No
    mask/cast/view is stripped. Multi-use and preparation-side widening uses are
    deliberately unsupported in this first slice.
    """
    try:
        revision = _revision(plan, frame, shapes)
    except _InvalidRetention:
        return None
    images = {image.node: image for image in frame.cut.images}
    recurrent = set(frame.cut.recurrence)
    result = []
    for buffer in frame.buffers:
        widening = buffer.node
        if (
            buffer.kind != "frontier"
            or widening is None
            or widening not in images
            or widening not in frame.cut.preparation
            or widening.target is not torch.ops.prims.convert_element_type.default
            or len(widening.args) != 2
            or widening.args[1] is not torch.float32
            or widening.kwargs
        ):
            continue
        source = widening.args[0]
        if (
            not isinstance(source, Node)
            or source.target is not memory_ops.load
            or source not in frame.cut.preparation
            or len(source.args) != 4
            or source.kwargs
        ):
            continue
        value = source.meta.get("val")
        host, indices, mask, _eviction = source.args
        if (
            not isinstance(host, Node)
            or not isinstance(host.meta.get("val"), torch.Tensor)
            or not isinstance(indices, (tuple, list))
            or (
                mask is not None
                and (
                    not isinstance(mask, Node)
                    or not isinstance(mask.meta.get("val"), torch.Tensor)
                    or mask.meta["val"].dtype != torch.bool
                )
            )
        ):
            continue
        users = tuple(widening.users)
        image = images[widening]
        actions = tuple(
            action for action in frame.actions if buffer.name in action.writes
        )
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype not in (torch.bfloat16, torch.float16)
            or host.meta["val"].dtype != value.dtype
            or tuple(source.users) != (widening,)
            or len(users) != 1
            or users[0] not in recurrent
            or image.consumers != users
            or buffer.dtype != torch.float32
            or widening.meta["val"].dtype != torch.float32
            or image.dtype != torch.float32
            or image.logical_domain != _domain(value)
            or _domain(widening.meta["val"]) != _domain(value)
            or shapes[source] != buffer.shape
            or shapes[widening] != buffer.shape
            or len(buffer.shape) != 2
            or any(size <= 0 for size in buffer.shape)
            or len(actions) != 1
            or actions[0].kind != "frontier"
            or actions[0].nodes != (widening,)
            or actions[0].writes != (buffer.name,)
            # Slice 1 preserves the original FP32 allocation, not merely the
            # smaller raw-half payload. Physical compaction needs a separate
            # accepted-action storage proof.
            or frame.layout.region(buffer.name).byte_size
            < math.prod(buffer.shape) * torch.float32.itemsize
            or frame.layout.region(buffer.name).live_until != len(frame.actions)
        ):
            continue
        result.append(
            RawWideningTransfer(
                buffer,
                source,
                widening,
                users[0],
                value.dtype,
                buffer.shape,
                actions[0].publication_event,
                revision,
            )
        )
    return tuple(result)
