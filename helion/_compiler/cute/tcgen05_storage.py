"""Disjoint static shared-memory packing for a dormant native composition.

All allocations remain live together. This planner reorders aligned storage;
it never aliases buffers, shortens rings, or changes resource capacity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NativeSharedAllocation:
    name: str
    bytes: int
    alignment: int

    def __post_init__(self) -> None:
        if not self.name or type(self.bytes) is not int or self.bytes <= 0:
            raise ValueError("shared allocation requires a name and positive byte size")
        if (
            type(self.alignment) is not int
            or self.alignment <= 0
            or self.alignment > (1 << 30)
            or self.alignment & (self.alignment - 1)
        ):
            raise ValueError("shared alignment must be a positive power of two")


@dataclass(frozen=True)
class NativeSharedPlacement:
    allocation: NativeSharedAllocation
    offset: int

    @property
    def end(self) -> int:
        return self.offset + self.allocation.bytes


@dataclass(frozen=True)
class NativeSharedStoragePlan:
    placements: tuple[NativeSharedPlacement, ...]

    def __post_init__(self) -> None:
        if not self.placements:
            raise ValueError("shared storage must have at least one placement")
        names: set[str] = set()
        end = 0
        base_alignment = self.placements[0].allocation.alignment
        for placement in self.placements:
            allocation = placement.allocation
            if (
                allocation.name in names
                or type(placement.offset) is not int
                or placement.offset < end
                or placement.offset % allocation.alignment
                or allocation.alignment > base_alignment
                or placement.end > (1 << 31) - 1
            ):
                raise ValueError("invalid or overlapping shared placement")
            names.add(allocation.name)
            end = placement.end

    @classmethod
    def pack(
        cls, allocations: tuple[NativeSharedAllocation, ...]
    ) -> NativeSharedStoragePlan:
        if not allocations or len({a.name for a in allocations}) != len(allocations):
            raise ValueError("shared allocations must be nonempty and uniquely named")
        # Stable tie ordering preserves the emitter's order among equal-aligned
        # objects. Each boundary is rounded up; no byte has multiple owners.
        ordered = sorted(allocations, key=lambda item: -item.alignment)
        offset = 0
        placements = []
        for allocation in ordered:
            offset = (offset + allocation.alignment - 1) & -allocation.alignment
            placement = NativeSharedPlacement(allocation, offset)
            if placement.end > (1 << 31) - 1:
                raise ValueError("shared placement exceeds the typed address domain")
            placements.append(placement)
            offset = placement.end
        return cls(tuple(placements))

    def append(self, allocation: NativeSharedAllocation) -> NativeSharedStoragePlan:
        """Append live storage without moving existing objects or their base.

        The first placement still determines the allocation's base alignment.
        A suffix cannot require a stronger base, even if its offset is aligned.
        ``pack`` retains its original descending-alignment packing policy.
        """
        if allocation.alignment > self.alignment:
            raise ValueError("shared suffix exceeds the existing base alignment")
        offset = (self.bytes + allocation.alignment - 1) & -allocation.alignment
        return type(self)((*self.placements, NativeSharedPlacement(allocation, offset)))

    @property
    def bytes(self) -> int:
        return self.placements[-1].end

    @property
    def alignment(self) -> int:
        return self.placements[0].allocation.alignment

    @property
    def cache_identity(self) -> tuple[tuple[str, int, int, int], ...]:
        return tuple(
            (p.allocation.name, p.allocation.bytes, p.allocation.alignment, p.offset)
            for p in self.placements
        )

    def fits(self, *, capacity: int, reserved: int) -> bool:
        return (
            capacity > 0
            and reserved >= 0
            and self.alignment <= capacity
            and self.bytes + reserved <= capacity
        )
