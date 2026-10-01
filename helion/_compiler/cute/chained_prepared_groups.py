"""Complete grouped-B images and conservative preparation-frame coallocation.

This is an early graph/storage proof, not permission to emit a native operand.
Every member still needs the original expression's late coordinate/domain proof
before any ordinary B fill is omitted. Producers retain their original nodes,
casts, masks and logical coordinates. Consumers retain whole-slot ownership
through MMA completion; none of these records grants an early release.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
from typing import TYPE_CHECKING

import torch

from .chained_late_rhs import _layout_bytes
from .chained_prepared_operands import plan_prepared_operands
from .chained_tcgen05 import _layout
from .chained_tcgen_stage import stage_geometry
from .chained_workspace import _reuse_requests
from .warp_specialized_plan import SharedBufferRequest
from .warp_specialized_plan import SharedMemoryLayoutPlan

if TYPE_CHECKING:
    from .chained_contraction_groups import ContractionGroup
    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_frame import PreparationBuffer
    from .chained_preparation_frame import PreparationFrame
    from .chained_recurrence_workspace import RecurrenceWorkspace
    from .chained_tcgen_stage import StageGeometry


@dataclass(frozen=True)
class PreparedGroupMember:
    buffer: PreparationBuffer
    stage: int
    operand_index: int
    geometry: StageGeometry
    row_offset: int
    byte_offset: int
    physical_shape: tuple[int, int]
    logical_modes: tuple[int, int]
    native_layout: tuple[str, ...]


@dataclass(frozen=True)
class PreparedGroupCandidate:
    group: ContractionGroup
    name: str
    members: tuple[PreparedGroupMember, ...]
    physical_shape: tuple[int, int]
    dtype: torch.dtype
    byte_size: int
    live_from: int
    live_until: int
    native_layout: tuple[str, ...]


@dataclass(frozen=True)
class PreparedGroupBinding:
    candidate: PreparedGroupCandidate
    byte_offset: int


@dataclass(frozen=True)
class PreparedGroupPlacement:
    frame: PreparationFrame
    groups: tuple[PreparedGroupBinding, ...]
    reservation_peak_bytes: int


def _valid_frame(frame: PreparationFrame) -> bool:
    """Check physical records and publication cuts without device context."""
    regions = frame.layout.regions
    by_name = {region.name: region for region in regions}
    if (
        len(by_name) != len(regions)
        or len({buffer.name for buffer in frame.buffers}) != len(frame.buffers)
        or set(by_name) != {buffer.name for buffer in frame.buffers}
        or frame.layout.allocated_bytes <= 0
        or frame.layout.allocated_bytes % 128
        or not frame.actions
        or frame.actions[-1].kind != "ready"
        or tuple(action.event for action in frame.actions)
        != tuple(range(len(frame.actions)))
        or any(
            region.alignment != 128
            or region.byte_offset < 0
            or region.byte_offset % 128
            or region.byte_size <= 0
            or region.byte_size % 128
            or region.byte_end > frame.layout.allocated_bytes
            or not 0 <= region.live_from < region.live_until <= len(frame.actions)
            for region in regions
        )
        or any(
            left.overlaps_lifetime(right) and left.overlaps_storage(right)
            for index, left in enumerate(regions)
            for right in regions[index + 1 :]
        )
    ):
        return False
    published: set[str] = set()
    for action in frame.actions:
        for name in (*action.reads, *action.writes):
            region = by_name.get(name)
            if (
                region is None
                or not region.live_from <= action.event < region.live_until
            ):
                return False
            if name in action.reads and (
                region.live_from >= action.event or name not in published
            ):
                return False
        # An early byte reservation is not a completed producer. Even when
        # a grouped writer starts earlier, ordinary action reads must retain
        # their original publication ordering.
        published.update(action.writes)
    return all(
        stage.a == by_name.get(stage.a.name) and stage.b == by_name.get(stage.b.name)
        for stage in frame.stages
    )


def _member_layout(
    name: str, shape: tuple[int, int], dtype: str, modes: tuple[int, int]
) -> tuple[str, ...]:
    lines = _layout(name, shape, 1, dtype)
    if modes == (1, 0):
        # Only the logical view changes. The native atom policy and backing
        # bytes are the same as the common TCgen operand layout.
        lines[1] = (
            f"{name} = cute.make_tensor(cute.recast_ptr({name}_ptr, "
            f"{name}_layout.inner, dtype={dtype}), "
            f"cute.select({name}_layout.outer, mode=[1, 0]))"
        )
    return tuple(lines)


def _native_row_subview(
    whole: tuple[int, int], member: tuple[int, int], row_offset: int
) -> bool:
    """Prove a dense, phase-preserving slice of the common K-major policy.

    With one complete K atom, ``_layout``'s unswizzled outer map is
    ``row * K + column``. Multiple K panels instead interleave all group rows
    before the next panel, and cannot use a dense member allocation.

    Native tensor indexing applies S<B,4,3>, B=log2(K/8), to BYTE addresses:
    ``swizzle(base_bytes + 2 * outer(row, column))``. The full and member
    views present the exact same absolute byte address to the same swizzle.
    XOR destination bits are below bit 7, so the existing 128-byte-aligned
    member boundaries retain dense disjoint spans at every frame/slot phase.
    Whole 16-row members have byte extents divisible by 128. Logical axis
    exchange only permutes coordinates. A composed-layout evaluation in
    element units is NOT this native-pointer address proof.
    """
    rows, k = whole
    member_rows, member_k = member
    if (
        _layout_bytes(whole, 1) is None
        or _layout_bytes(member, 1) is None
        or member_k != k
        or rows % 16
        or member_rows % 16
        or row_offset % 16
        or not 0 <= row_offset < row_offset + member_rows <= rows
    ):
        return False
    width_bytes = 2 * k
    atom_bytes = min(128, width_bytes & -width_bytes)
    return width_bytes == atom_bytes


def prepared_group_candidates(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    recurrence: RecurrenceWorkspace,
) -> tuple[PreparedGroupCandidate, ...] | None:
    """Find complete ordered groups of exact, exclusive typed frontier images.

    Invalid retained plans return ``None``; noneligible groups remain ordinary
    staging and contribute no candidate. The initial capability requires each
    distinct image to have only its original B-member consumer. No padding,
    duplicate image, cast bypass, partial group, or extra consumer is admitted.
    """
    if (
        not _valid_frame(frame)
        or plan_prepared_operands(plan, frame, recurrence) is None
    ):
        return None
    assert plan.region is not None
    buffers = {
        buffer.node: buffer for buffer in frame.buffers if buffer.kind == "frontier"
    }
    regions = {region.name: region for region in frame.layout.regions}
    result = []
    for stage in recurrence.stages:
        group = stage.group
        if len(group.stages) < 2:
            continue
        _, n, k = group.physical
        total_bytes = _layout_bytes((n, k), 1)
        if total_bytes is None or stage_geometry(group.physical) is None:
            continue
        members = []
        dtype = None
        for index, geometry, row_offset in zip(
            group.stages, group.geometries, group.offsets, strict=True
        ):
            spec = plan.region.contractions[index]
            operand_index, coordinates = geometry.operand("b", "row", "k")
            operand = (spec.lhs, spec.rhs)[operand_index]
            buffer = buffers.get(operand)
            rows = geometry.physical[1]
            shape = (rows, k)
            modes = (0, 1) if coordinates == ("row", "k") else (1, 0)
            logical_shape = shape if modes == (0, 1) else shape[::-1]
            size = _layout_bytes(shape, 1)
            if (
                buffer is None
                or any(member.buffer.node is operand for member in members)
                or set(operand.users) != {spec.node}
                or sum(arg is operand for arg in spec.node.args[:3]) != 1
                or coordinates not in (("row", "k"), ("k", "row"))
                or buffer.dtype not in (torch.bfloat16, torch.float16)
                or dtype is not None
                and buffer.dtype != dtype
                or spec.operand_dtypes != (buffer.dtype, buffer.dtype)
                or geometry.physical[::2] != group.physical[::2]
                or any(
                    type(old) is int and old != configured
                    for old, configured in zip(
                        spec.shape, geometry.logical, strict=True
                    )
                )
                or buffer.shape != logical_shape
                or size is None
                or size != math.prod(buffer.shape) * buffer.dtype.itemsize
                or size != regions[buffer.name].byte_size
                or row_offset * k * buffer.dtype.itemsize % 128
                or not _native_row_subview((n, k), shape, row_offset)
            ):
                break
            dtype = buffer.dtype
            dtype_name = (
                "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
            )
            members.append(
                PreparedGroupMember(
                    buffer,
                    index,
                    operand_index,
                    geometry,
                    row_offset,
                    row_offset * k * dtype.itemsize,
                    shape,
                    modes,
                    _member_layout(buffer.name, shape, dtype_name, modes),
                )
            )
        if len(members) != len(group.stages) or dtype is None:
            continue
        if total_bytes != sum(
            regions[member.buffer.name].byte_size for member in members
        ):
            continue
        name = f"chain_prepared_group_{group.stages[0]}"
        if name in regions:
            return None
        dtype_name = (
            "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
        )
        result.append(
            PreparedGroupCandidate(
                group,
                name,
                tuple(members),
                (n, k),
                dtype,
                total_bytes,
                min(regions[member.buffer.name].live_from for member in members),
                max(regions[member.buffer.name].live_until for member in members),
                tuple(_layout(name, (n, k), 1, dtype_name)),
            )
        )
    return tuple(result)


def coallocate_prepared_groups(
    plan: ChainedMatmulPlan,
    frame: PreparationFrame,
    recurrence: RecurrenceWorkspace,
    candidates: tuple[PreparedGroupCandidate, ...],
    *,
    frame_capacity_bytes: int,
) -> PreparedGroupPlacement | None:
    """Place selected early candidates without changing any materialization.

    The caller supplies an actual per-frame quota after accounting for other
    resources and slot count. Overflow returns ``None`` for ordinary fallback.
    Aggregate requests reserve the union lifetime, then expand back into the
    original disjoint member regions. Every region and preparation A/B handle
    is rebound. Recompute downstream native bindings against the returned frame;
    old region-bearing proofs must not be transplanted to the new placement.
    Late domain proof is still mandatory before adopting or emitting a group.
    The actual frame base and whole-slot stride must retain the existing
    128-byte alignment. Native tensor swizzles act on absolute BYTE addresses.
    """
    available = prepared_group_candidates(plan, frame, recurrence)
    if (
        available is None
        or type(frame_capacity_bytes) is not int
        or frame_capacity_bytes <= 0
        or len(set(candidates)) != len(candidates)
        or candidates
        != tuple(candidate for candidate in available if candidate in candidates)
    ):
        return None
    if not candidates:
        return (
            PreparedGroupPlacement(frame, (), frame.peak_live_bytes)
            if frame.layout.allocated_bytes <= frame_capacity_bytes
            else None
        )
    owners = {
        member.buffer.name: (candidate, member)
        for candidate in candidates
        for member in candidate.members
    }
    if len(owners) != sum(len(candidate.members) for candidate in candidates):
        return None
    requests = tuple(
        SharedBufferRequest(
            region.name,
            region.byte_size,
            region.alignment,
            region.live_from,
            region.live_until,
        )
        for region in frame.layout.regions
        if region.name not in owners
    ) + tuple(
        SharedBufferRequest(
            candidate.name,
            candidate.byte_size,
            128,
            candidate.live_from,
            candidate.live_until,
        )
        for candidate in candidates
    )
    packed = _reuse_requests(requests)
    if packed.allocated_bytes > frame_capacity_bytes:
        return None
    regions = []
    for original in frame.layout.regions:
        if original.name in owners:
            candidate, member = owners[original.name]
            regions.append(
                replace(
                    original,
                    byte_offset=packed.region(candidate.name).byte_offset
                    + member.byte_offset,
                )
            )
        else:
            regions.append(packed.region(original.name))
    layout = SharedMemoryLayoutPlan(tuple(regions), packed.allocated_bytes)
    rebound = replace(
        frame,
        layout=layout,
        stages=tuple(
            replace(stage, a=layout.region(stage.a.name), b=layout.region(stage.b.name))
            for stage in frame.stages
        ),
    )
    if not _valid_frame(rebound):
        return None
    peak = max(
        sum(
            region.byte_size
            for region in packed.regions
            if region.live_from <= action.event < region.live_until
        )
        for action in frame.actions
    )
    return PreparedGroupPlacement(
        rebound,
        tuple(
            PreparedGroupBinding(candidate, packed.region(candidate.name).byte_offset)
            for candidate in candidates
        ),
        peak,
    )
