from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import Literal

from .strategies import TCGEN05_LEGAL_L2_SWIZZLE_SIZES
from .tcgen05_constants import TCGEN05_CONSUMER_REGS_CHOICES
from .tcgen05_constants import TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS
from .tcgen05_constants import TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE
from .tcgen05_constants import TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES

if TYPE_CHECKING:
    from .grouped_worklist import GroupedWorklistRows

GroupedBMajor = Literal["k", "n"]
GroupedWorklistHardwareIdentity = tuple[str, str, str]


@dataclasses.dataclass(frozen=True, slots=True)
class GroupedWorklistWorkload:
    """Exact semantic workload for one measured grouped-worklist seed.

    ``num_sm`` is an observed execution-resource constraint, not a product
    identifier. The enclosing policy is selected by exact hardware identity;
    keeping the SM count here protects the measured wave geometry on reduced-SM
    or partitioned instances of that product.
    """

    groups: int
    n: int
    k: int
    b_major: GroupedBMajor
    source_m_tile: int
    source_tiles: int
    num_sm: int
    reviewed_worklist_rows: GroupedWorklistRows | None = None

    def __post_init__(self) -> None:
        positive_fields = (
            self.groups,
            self.n,
            self.k,
            self.source_m_tile,
            self.source_tiles,
            self.num_sm,
        )
        if any(type(value) is not int or value <= 0 for value in positive_fields):
            raise ValueError(
                "grouped-worklist workload values must be positive integers"
            )
        if self.b_major not in ("k", "n"):
            raise ValueError("grouped-worklist B major must be 'k' or 'n'")
        if self.source_m_tile not in TCGEN05_GROUPED_WORKLIST_SOURCE_M_TILE_CHOICES:
            raise ValueError("unsupported grouped-worklist source M tile")
        if self.n % 32:
            raise ValueError("grouped-worklist N must be divisible by 32")
        rows = self.reviewed_worklist_rows
        if rows is None:
            return
        if type(rows) is not tuple or len(rows) != self.groups:
            raise ValueError("reviewed worklist rows must match the group count")
        expected_start = 0
        for expected_group, row in enumerate(rows):
            if (
                type(row) is not tuple
                or len(row) != 4
                or any(type(value) is not int for value in row)
            ):
                raise ValueError("reviewed worklist rows must be integer 4-tuples")
            group, start, valid_m, stored_m = row
            if (
                group != expected_group
                or start != expected_start
                or valid_m <= 0
                or valid_m > stored_m
                or stored_m % self.source_m_tile
            ):
                raise ValueError("reviewed worklist rows are not normalized")
            expected_start += stored_m
        if expected_start // self.source_m_tile != self.source_tiles:
            raise ValueError("reviewed worklist rows do not match source tiles")

    def matches(
        self,
        *,
        groups: int,
        n: int,
        k: int,
        b_major: GroupedBMajor,
        source_m_tile: int,
        source_tiles: int,
        num_sm: int,
        worklist_rows: GroupedWorklistRows | None,
    ) -> bool:
        return (
            groups == self.groups
            and n == self.n
            and k == self.k
            and b_major == self.b_major
            and source_m_tile == self.source_m_tile
            and source_tiles == self.source_tiles
            and num_sm == self.num_sm
            and worklist_rows == self.reviewed_worklist_rows
        )

    def overlaps(self, other: GroupedWorklistWorkload) -> bool:
        """Whether two policies could select the same observed workload."""
        return (
            self.groups == other.groups
            and self.n == other.n
            and self.k == other.k
            and self.b_major == other.b_major
            and self.source_m_tile == other.source_m_tile
            and self.source_tiles == other.source_tiles
            and self.num_sm == other.num_sm
            and (
                self.reviewed_worklist_rows is None
                or other.reviewed_worklist_rows is None
                or self.reviewed_worklist_rows == other.reviewed_worklist_rows
            )
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GroupedWorklistTuning:
    """One measured seed for an exact grouped-worklist workload."""

    workload: GroupedWorklistWorkload
    consumer_regs: int
    l2_swizzle_size: int | None
    block_k: int = 64
    ab_stages: int = 6
    runtime_direct: bool = True
    reserved_sms: int | None = None
    clc: bool = True

    def __post_init__(self) -> None:
        if type(self.workload) is not GroupedWorklistWorkload:
            raise ValueError("grouped-worklist tuning requires a workload")
        if (
            type(self.block_k) is not int
            or self.block_k not in TCGEN05_GROUPED_WORKLIST_BLOCK_K_CHOICES
        ):
            raise ValueError("unsupported grouped-worklist block K")
        if self.workload.k % self.block_k:
            raise ValueError("grouped-worklist block K must divide workload K")
        if (
            type(self.consumer_regs) is not int
            or self.consumer_regs not in TCGEN05_CONSUMER_REGS_CHOICES
        ):
            raise ValueError("unsupported grouped-worklist consumer register count")
        if type(self.ab_stages) is not int or not 1 <= self.ab_stages <= 7:
            raise ValueError("grouped-worklist AB stages must be between 1 and 7")
        if self.l2_swizzle_size is not None and (
            type(self.l2_swizzle_size) is not int
            or self.l2_swizzle_size not in TCGEN05_LEGAL_L2_SWIZZLE_SIZES
        ):
            raise ValueError("unsupported grouped-worklist L2 swizzle")
        if type(self.runtime_direct) is not bool or type(self.clc) is not bool:
            raise ValueError("grouped-worklist scheduler flags must be booleans")
        if self.reserved_sms is not None and (
            type(self.reserved_sms) is not int
            or self.reserved_sms <= 0
            or self.reserved_sms > TCGEN05_GROUPED_STATIC_RESERVED_SMS_MAX
            or self.reserved_sms > self.workload.num_sm - 2
        ):
            raise ValueError("unsupported grouped-worklist reserved SM count")
        if (
            self.l2_swizzle_size is not None
            and self.l2_swizzle_size > 1
            and not self.runtime_direct
        ):
            raise ValueError(
                "grouped-worklist panel swizzles require runtime_direct=True"
            )
        if not self.clc:
            return
        if not self.runtime_direct:
            raise ValueError("grouped worklist CLC requires runtime_direct=True")
        if self.reserved_sms is not None:
            raise ValueError("grouped worklist CLC cannot reserve SMs")
        if self.workload.source_m_tile == TCGEN05_GROUPED_WORKLIST_SMALL_SOURCE_M_TILE:
            raise ValueError("grouped worklist CLC requires a two-CTA source M tile")
        if self.workload.n % 256:
            raise ValueError("grouped worklist CLC requires N divisible by 256")
        n_tiles = self.workload.n // 256
        clusters = self.workload.source_tiles * n_tiles
        if clusters < self.workload.num_sm:
            raise ValueError("grouped worklist CLC requires at least one device wave")
        if clusters > TCGEN05_GROUPED_RUNTIME_DIRECT_CLC_MAX_CLUSTERS:
            raise ValueError("grouped worklist CLC exceeds the runtime grid limit")

    def matches(
        self,
        *,
        groups: int,
        n: int,
        k: int,
        b_major: GroupedBMajor,
        source_m_tile: int,
        source_tiles: int,
        num_sm: int,
        worklist_rows: GroupedWorklistRows | None,
    ) -> bool:
        return self.workload.matches(
            groups=groups,
            n=n,
            k=k,
            b_major=b_major,
            source_m_tile=source_m_tile,
            source_tiles=source_tiles,
            num_sm=num_sm,
            worklist_rows=worklist_rows,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GroupedWorklistTargetPolicy:
    """Measured grouped-worklist seed overrides for one exact target."""

    tunings: tuple[GroupedWorklistTuning, ...] = ()

    def __post_init__(self) -> None:
        if type(self.tunings) is not tuple or any(
            type(tuning) is not GroupedWorklistTuning for tuning in self.tunings
        ):
            raise ValueError(
                "grouped-worklist target tunings must be an immutable tuple"
            )
        for index, tuning in enumerate(self.tunings):
            for prior in self.tunings[:index]:
                if tuning.workload.overlaps(prior.workload):
                    raise ValueError("grouped-worklist target tunings must not overlap")

    def tuning_for(
        self,
        *,
        groups: int,
        n: int,
        k: int,
        b_major: GroupedBMajor,
        source_m_tile: int,
        source_tiles: int,
        num_sm: int,
        worklist_rows: GroupedWorklistRows | None,
    ) -> GroupedWorklistTuning | None:
        return next(
            (
                tuning
                for tuning in self.tunings
                if tuning.matches(
                    groups=groups,
                    n=n,
                    k=k,
                    b_major=b_major,
                    source_m_tile=source_m_tile,
                    source_tiles=source_tiles,
                    num_sm=num_sm,
                    worklist_rows=worklist_rows,
                )
            ),
            None,
        )

    def reviewed_worklist_rows(self) -> frozenset[GroupedWorklistRows]:
        """Return exact measured row signatures eligible for policy overrides."""
        return frozenset(
            tuning.workload.reviewed_worklist_rows
            for tuning in self.tunings
            if tuning.workload.reviewed_worklist_rows is not None
        )


def _reviewed_worklist_rows(actual_ms: tuple[int, ...]) -> GroupedWorklistRows:
    """Build the exact normalized source-256 worklist used by the campaign."""
    rows: list[tuple[int, int, int, int]] = []
    start = 0
    source_m_tile = TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE
    for group, valid_m in enumerate(actual_ms):
        stored_m = (valid_m + source_m_tile - 1) // source_m_tile * source_m_tile
        rows.append((group, start, valid_m, stored_m))
        start += stored_m
    return tuple(rows)


def _gb300_workload(
    groups: int,
    n: int,
    k: int,
    b_major: GroupedBMajor,
    actual_ms: tuple[int, ...],
) -> GroupedWorklistWorkload:
    """Build one exact GB300 source-256/152-SM workload."""
    rows = _reviewed_worklist_rows(actual_ms)
    source_m_tile = TCGEN05_GROUPED_WORKLIST_LARGE_SOURCE_M_TILE
    source_tiles = sum(row[3] for row in rows) // source_m_tile
    return GroupedWorklistWorkload(
        groups=groups,
        n=n,
        k=k,
        b_major=b_major,
        source_m_tile=source_m_tile,
        source_tiles=source_tiles,
        num_sm=152,
        reviewed_worklist_rows=rows,
    )


_GENERIC_GROUPED_WORKLIST_TARGET_POLICY = GroupedWorklistTargetPolicy()
_GROUPED_WORKLIST_TARGET_POLICIES: dict[
    GroupedWorklistHardwareIdentity, GroupedWorklistTargetPolicy
] = {
    ("cuda", "NVIDIA B200", "sm100"): GroupedWorklistTargetPolicy(),
    ("cuda", "NVIDIA GB300", "sm103"): GroupedWorklistTargetPolicy(
        tunings=(
            GroupedWorklistTuning(
                workload=_gb300_workload(
                    4,
                    6144,
                    7168,
                    "k",
                    (9884, 9459, 7801, 7007),
                ),
                consumer_regs=256,
                l2_swizzle_size=8,
            ),
            GroupedWorklistTuning(
                workload=_gb300_workload(
                    4,
                    7168,
                    3072,
                    "k",
                    (8247, 7724, 9586, 7225),
                ),
                consumer_regs=232,
                l2_swizzle_size=32,
            ),
            GroupedWorklistTuning(
                workload=_gb300_workload(
                    8,
                    7168,
                    3072,
                    "k",
                    (4027, 3114, 3934, 4368, 5111, 5242, 4039, 4993),
                ),
                consumer_regs=240,
                l2_swizzle_size=8,
            ),
            GroupedWorklistTuning(
                workload=_gb300_workload(
                    4,
                    4096,
                    4096,
                    "k",
                    (8076, 8601, 10197, 8215),
                ),
                consumer_regs=224,
                l2_swizzle_size=16,
            ),
            GroupedWorklistTuning(
                workload=_gb300_workload(
                    4,
                    4096,
                    2048,
                    "k",
                    (7119, 9449, 8773, 6965),
                ),
                consumer_regs=240,
                l2_swizzle_size=1,
            ),
            GroupedWorklistTuning(
                workload=_gb300_workload(
                    8,
                    6144,
                    7168,
                    "k",
                    (5102, 5282, 4858, 5084, 3629, 4660, 5076, 4548),
                ),
                consumer_regs=240,
                l2_swizzle_size=8,
            ),
            GroupedWorklistTuning(
                workload=_gb300_workload(
                    8,
                    4096,
                    4096,
                    "k",
                    (3507, 4845, 4215, 2901, 4635, 3847, 4894, 4509),
                ),
                consumer_regs=256,
                l2_swizzle_size=32,
            ),
            GroupedWorklistTuning(
                workload=_gb300_workload(
                    8,
                    4096,
                    2048,
                    "k",
                    (2870, 4080, 4999, 3466, 3666, 5006, 3336, 4261),
                ),
                consumer_regs=256,
                l2_swizzle_size=1,
            ),
        ),
    ),
}


def get_grouped_worklist_target_policy(
    hardware_identity: GroupedWorklistHardwareIdentity | None,
) -> GroupedWorklistTargetPolicy:
    """Return tuning measured on one exact device/product/architecture."""
    if hardware_identity is None:
        return _GENERIC_GROUPED_WORKLIST_TARGET_POLICY
    return _GROUPED_WORKLIST_TARGET_POLICIES.get(
        hardware_identity,
        _GENERIC_GROUPED_WORKLIST_TARGET_POLICY,
    )


def grouped_worklist_target_identities() -> frozenset[GroupedWorklistHardwareIdentity]:
    """Return exact hardware identities with validated worklist policies."""
    return frozenset(_GROUPED_WORKLIST_TARGET_POLICIES)
