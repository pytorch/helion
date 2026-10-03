# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Workload-neutral resource plans for warp-specialized CuTe kernels.

The objects in this module describe physical resources only.  They deliberately
contain no tensor names, arithmetic policies, or application-specific roles.
They are suitable for validating both generated kernels and specialized
schedules before either path emits CuTe DSL.
"""

from __future__ import annotations

import dataclasses

from ... import exc
from .thread_budget import MAX_THREADS_PER_BLOCK

WARP_SIZE = 32
REGISTER_FILE_WORDS = 65_536
MAX_MBAR_ARRIVALS = (1 << 20) - 1
VALID_TMEM_COLUMNS = (0, 32, 64, 128, 256, 512)


def _power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclasses.dataclass(frozen=True)
class WarpRole:
    """A contiguous set of warps with one dynamic register budget."""

    name: str
    first_warp: int
    warp_count: int
    registers_per_thread: int

    @property
    def last_warp(self) -> int:
        return self.first_warp + self.warp_count


@dataclasses.dataclass(frozen=True)
class MBarrierRegion:
    """A packed array of mbarriers sharing one arrival count."""

    name: str
    byte_offset: int
    stages: int
    arrivals: int

    @property
    def byte_size(self) -> int:
        return self.stages * 8


@dataclasses.dataclass(frozen=True)
class SharedBufferRegion:
    """One shared-memory interval with an abstract half-open lifetime."""

    name: str
    byte_offset: int
    byte_size: int
    live_from: int
    live_until: int
    alignment: int = 16

    @property
    def byte_end(self) -> int:
        return self.byte_offset + self.byte_size

    def overlaps_storage(self, other: SharedBufferRegion) -> bool:
        return self.byte_offset < other.byte_end and other.byte_offset < self.byte_end

    def overlaps_lifetime(self, other: SharedBufferRegion) -> bool:
        return self.live_from < other.live_until and other.live_from < self.live_until


@dataclasses.dataclass(frozen=True)
class SharedBufferRequest:
    """A shared-memory allocation request independent of tensor semantics."""

    name: str
    byte_size: int
    alignment: int = 16
    live_from: int = 0
    live_until: int = 1


@dataclasses.dataclass(frozen=True)
class SharedMemoryLayoutPlan:
    """Resolved shared-memory regions and aligned allocation size."""

    regions: tuple[SharedBufferRegion, ...]
    allocated_bytes: int

    def region(self, name: str) -> SharedBufferRegion:
        matches = tuple(region for region in self.regions if region.name == name)
        if len(matches) != 1:
            raise ValueError(f"invalid shared-memory region {name!r}")
        return matches[0]


@dataclasses.dataclass(frozen=True)
class TensorMemoryRegionRequest:
    """A sequential TMEM allocation request measured in columns."""

    name: str
    columns: int
    stages: int = 1


@dataclasses.dataclass(frozen=True)
class TensorMemoryRegion:
    """One resolved, contiguous TMEM region."""

    name: str
    column_offset: int
    columns: int
    stages: int

    @property
    def column_end(self) -> int:
        return self.column_offset + self.columns * self.stages


@dataclasses.dataclass(frozen=True)
class TensorMemoryLayoutPlan:
    """Sequential TMEM regions and their hardware allocation envelope."""

    regions: tuple[TensorMemoryRegion, ...]
    required_columns: int
    allocated_columns: int

    def region(self, name: str) -> TensorMemoryRegion:
        matches = tuple(region for region in self.regions if region.name == name)
        if len(matches) != 1:
            raise ValueError(f"invalid TMEM region {name!r}")
        return matches[0]


@dataclasses.dataclass(frozen=True)
class WarpSpecializedPipelinePlan:
    """Validated physical resource contract for one CTA pipeline."""

    threads: int
    shared_bytes: int
    max_shared_bytes_per_block: int
    shared_bytes_per_mp: int
    tmem_columns: int
    min_blocks_per_mp: int
    max_threads_per_mp: int
    max_blocks_per_mp: int
    roles: tuple[WarpRole, ...]
    barriers: tuple[MBarrierRegion, ...] = ()
    shared_buffers: tuple[SharedBufferRegion, ...] = ()

    @property
    def warps(self) -> int:
        return self.threads // WARP_SIZE

    @property
    def register_words(self) -> int:
        return sum(
            role.warp_count * WARP_SIZE * role.registers_per_thread
            for role in self.roles
        )

    def role(self, name: str) -> WarpRole:
        """Return one named role, rejecting ambiguous or missing plans."""

        matches = tuple(role for role in self.roles if role.name == name)
        if len(matches) != 1:
            raise exc.BackendUnsupported("cute", f"invalid warp role {name!r}")
        return matches[0]

    def barrier(self, name: str) -> MBarrierRegion:
        """Return one named barrier region."""

        matches = tuple(region for region in self.barriers if region.name == name)
        if len(matches) != 1:
            raise exc.BackendUnsupported("cute", f"invalid mbarrier region {name!r}")
        return matches[0]

    def shared_buffer(self, name: str) -> SharedBufferRegion:
        """Return one named shared-memory region."""

        matches = tuple(region for region in self.shared_buffers if region.name == name)
        if len(matches) != 1:
            raise exc.BackendUnsupported(
                "cute", f"invalid shared-buffer region {name!r}"
            )
        return matches[0]

    def validate(self) -> None:
        """Reject resource layouts that cannot describe one legal CUDA CTA."""

        if (
            type(self.threads) is not int
            or self.threads <= 0
            or self.threads % WARP_SIZE
            or self.threads > MAX_THREADS_PER_BLOCK
            or type(self.shared_bytes) is not int
            or self.shared_bytes < 0
            or type(self.max_shared_bytes_per_block) is not int
            or self.max_shared_bytes_per_block <= 0
            or self.shared_bytes > self.max_shared_bytes_per_block
            or type(self.shared_bytes_per_mp) is not int
            or self.shared_bytes_per_mp <= 0
            or type(self.tmem_columns) is not int
            or self.tmem_columns not in VALID_TMEM_COLUMNS
            or type(self.min_blocks_per_mp) is not int
            or self.min_blocks_per_mp <= 0
            or self.shared_bytes * self.min_blocks_per_mp > self.shared_bytes_per_mp
            or type(self.max_threads_per_mp) is not int
            or self.max_threads_per_mp <= 0
            or self.threads * self.min_blocks_per_mp > self.max_threads_per_mp
            or type(self.max_blocks_per_mp) is not int
            or self.max_blocks_per_mp <= 0
            or self.min_blocks_per_mp > self.max_blocks_per_mp
        ):
            raise exc.BackendUnsupported("cute", "invalid warp-specialized resources")

        names: set[str] = set()
        covered_warps: set[int] = set()
        for role in self.roles:
            if (
                not role.name
                or role.name in names
                or type(role.first_warp) is not int
                or type(role.warp_count) is not int
                or role.first_warp < 0
                or role.warp_count <= 0
                or role.first_warp % 4
                or role.warp_count % 4
                or role.last_warp > self.warps
                or type(role.registers_per_thread) is not int
                or not 24 <= role.registers_per_thread <= 256
                or role.registers_per_thread % 8
            ):
                raise exc.BackendUnsupported("cute", "invalid warp role")
            names.add(role.name)
            role_warps = set(range(role.first_warp, role.last_warp))
            if covered_warps.intersection(role_warps):
                raise exc.BackendUnsupported("cute", "overlapping warp roles")
            covered_warps.update(role_warps)
        if covered_warps != set(range(self.warps)):
            raise exc.BackendUnsupported("cute", "warp roles must cover the CTA")
        if self.register_words * self.min_blocks_per_mp > REGISTER_FILE_WORDS:
            raise exc.BackendUnsupported("cute", "warp roles exceed register budget")
        if self.tmem_columns * self.min_blocks_per_mp > 512:
            raise exc.BackendUnsupported("cute", "pipeline exceeds TMEM capacity")

        barrier_names: set[str] = set()
        barrier_spans: list[tuple[int, int]] = []
        for region in self.barriers:
            if (
                not region.name
                or region.name in barrier_names
                or type(region.byte_offset) is not int
                or region.byte_offset < 0
                or region.byte_offset % 8
                or type(region.stages) is not int
                or region.stages <= 0
                or type(region.arrivals) is not int
                or region.arrivals <= 0
                or region.arrivals > MAX_MBAR_ARRIVALS
            ):
                raise exc.BackendUnsupported("cute", "invalid mbarrier region")
            end = region.byte_offset + region.byte_size
            if end > self.shared_bytes or any(
                region.byte_offset < other_end and other_start < end
                for other_start, other_end in barrier_spans
            ):
                raise exc.BackendUnsupported("cute", "invalid mbarrier region")
            barrier_names.add(region.name)
            barrier_spans.append((region.byte_offset, end))

        buffer_names: set[str] = set()
        for index, region in enumerate(self.shared_buffers):
            if (
                not region.name
                or region.name in buffer_names
                or type(region.byte_offset) is not int
                or type(region.byte_size) is not int
                or region.byte_offset < 0
                or region.byte_size <= 0
                or type(region.alignment) is not int
                or not _power_of_two(region.alignment)
                or region.byte_offset % region.alignment
                or region.byte_end > self.shared_bytes
                or type(region.live_from) is not int
                or type(region.live_until) is not int
                or region.live_from < 0
                or region.live_until <= region.live_from
                or any(
                    region.byte_offset < barrier_end and barrier_start < region.byte_end
                    for barrier_start, barrier_end in barrier_spans
                )
            ):
                raise exc.BackendUnsupported("cute", "invalid shared-buffer region")
            buffer_names.add(region.name)
            for other in self.shared_buffers[:index]:
                if region.overlaps_storage(other) and region.overlaps_lifetime(other):
                    raise exc.BackendUnsupported(
                        "cute", "overlapping live shared-buffer regions"
                    )


def accumulator_tmem_columns(n_dim: int) -> int:
    """Return TMEM columns for an FP32 ``[128, n_dim]`` accumulator."""

    if type(n_dim) is not int or n_dim <= 0 or n_dim % 8:
        raise ValueError(f"n_dim must be a positive multiple of 8, got {n_dim!r}")
    return n_dim


def packed_input_tmem_columns(k_dim: int) -> int:
    """Return TMEM columns for a packed 16-bit ``[128, k_dim]`` input."""

    if type(k_dim) is not int or k_dim <= 0 or k_dim % 2:
        raise ValueError(f"k_dim must be a positive even integer, got {k_dim!r}")
    return k_dim // 2


def allocated_tmem_columns(required_columns: int) -> int:
    """Round a requirement to a legal tcgen05 allocation size."""

    if type(required_columns) is not int or required_columns <= 0:
        raise ValueError(
            f"required_columns must be a positive integer, got {required_columns!r}"
        )
    for columns in VALID_TMEM_COLUMNS[1:]:
        if required_columns <= columns:
            return columns
    raise ValueError(f"required_columns must be <= 512, got {required_columns}")


def allocate_tmem_regions(
    requests: tuple[TensorMemoryRegionRequest, ...],
) -> TensorMemoryLayoutPlan:
    """Lay out named TMEM regions sequentially and select a legal allocation."""

    names: set[str] = set()
    regions: list[TensorMemoryRegion] = []
    cursor = 0
    for request in requests:
        if (
            not request.name
            or request.name in names
            or type(request.columns) is not int
            or request.columns <= 0
            or type(request.stages) is not int
            or request.stages <= 0
        ):
            raise ValueError("invalid TMEM region request")
        names.add(request.name)
        region = TensorMemoryRegion(
            request.name,
            cursor,
            request.columns,
            request.stages,
        )
        regions.append(region)
        cursor = region.column_end
    return TensorMemoryLayoutPlan(
        regions=tuple(regions),
        required_columns=cursor,
        allocated_columns=allocated_tmem_columns(cursor),
    )


def allocate_shared_regions(
    requests: tuple[SharedBufferRequest, ...],
    *,
    final_alignment: int = 16,
) -> SharedMemoryLayoutPlan:
    """Lay out shared buffers sequentially with explicit alignment/lifetimes."""

    if type(final_alignment) is not int or not _power_of_two(final_alignment):
        raise ValueError(f"invalid final alignment {final_alignment!r}")
    names: set[str] = set()
    regions: list[SharedBufferRegion] = []
    cursor = 0
    for request in requests:
        if (
            not request.name
            or request.name in names
            or type(request.byte_size) is not int
            or request.byte_size <= 0
            or type(request.alignment) is not int
            or not _power_of_two(request.alignment)
            or final_alignment % request.alignment
            or type(request.live_from) is not int
            or type(request.live_until) is not int
            or request.live_from < 0
            or request.live_until <= request.live_from
        ):
            raise ValueError("invalid shared-memory region request")
        names.add(request.name)
        cursor = (
            (cursor + request.alignment - 1) // request.alignment * request.alignment
        )
        region = SharedBufferRegion(
            request.name,
            cursor,
            request.byte_size,
            request.live_from,
            request.live_until,
            alignment=request.alignment,
        )
        regions.append(region)
        cursor = region.byte_end
    allocated_bytes = (
        (cursor + final_alignment - 1) // final_alignment * final_alignment
    )
    return SharedMemoryLayoutPlan(tuple(regions), allocated_bytes)


def chained_recurrence_tmem_layout(
    *,
    state_width: int,
    step_width: int,
    factor_input_stages: int = 2,
    auxiliary_accumulator_stages: int = 2,
) -> TensorMemoryLayoutPlan:
    """Plan TMEM for a resident state and a chain of local contractions.

    The two primary accumulator images are separated by the auxiliary
    accumulator ring.  This permits a consumer to retain one primary result
    while the independent auxiliary chain advances, without encoding any
    workload-specific operand names or arithmetic.
    """

    state_columns = accumulator_tmem_columns(state_width)
    step_columns = accumulator_tmem_columns(step_width)
    return allocate_tmem_regions(
        (
            TensorMemoryRegionRequest("state", state_columns),
            TensorMemoryRegionRequest(
                "state_input", packed_input_tmem_columns(state_width)
            ),
            TensorMemoryRegionRequest(
                "factor_input",
                packed_input_tmem_columns(step_width),
                factor_input_stages,
            ),
            TensorMemoryRegionRequest("primary_accumulator", step_columns),
            TensorMemoryRegionRequest(
                "auxiliary_accumulator",
                step_columns,
                auxiliary_accumulator_stages,
            ),
            TensorMemoryRegionRequest("secondary_accumulator", step_columns),
        )
    )


__all__ = [
    "MBarrierRegion",
    "SharedBufferRegion",
    "SharedBufferRequest",
    "SharedMemoryLayoutPlan",
    "TensorMemoryLayoutPlan",
    "TensorMemoryRegion",
    "TensorMemoryRegionRequest",
    "WarpRole",
    "WarpSpecializedPipelinePlan",
    "accumulator_tmem_columns",
    "allocate_shared_regions",
    "allocate_tmem_regions",
    "allocated_tmem_columns",
    "chained_recurrence_tmem_layout",
    "packed_input_tmem_columns",
]
