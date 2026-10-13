"""Physical owner checks shared by accepted preparation transports.

Callers supply a separately proved effective access interval. These checks do
not recognize a graph, accept an emission, shorten a lease, or authorize an
in-place copy. In particular, a semantic native-layout proof is not evidence
that its original pointer remains safe after storage placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chained_preparation_storage import AcceptedPreparationStorage
    from .warp_specialized_plan import SharedBufferRegion


def preparation_transfer_owner(
    physical: AcceptedPreparationStorage,
    name: str,
    first: int,
    stop: int,
) -> SharedBufferRegion | None:
    """Resolve a complete owner and every effective access, not a dense crop."""
    if type(first) is not int or type(stop) is not int or not 0 <= first < stop:
        return None
    views = tuple(
        view for view in physical.views if view.original.semantic.name == name
    )
    if len(views) != 1:
        return None
    view = views[0]
    owners = tuple(
        region for region in physical.layout.regions if region.name == view.owner
    )
    if len(owners) != 1:
        return None
    region = owners[0]
    if (
        not region.live_from <= first < stop <= region.live_until
        or region.alignment != 128
        or region.byte_offset < 0
        or region.byte_offset % 128
        or region.byte_size <= 0
        or region.byte_end > physical.layout.allocated_bytes
        or not view.accesses
        or any(
            length <= 0
            or start < region.byte_offset
            or start + length > region.byte_end
            for start, length in view.accesses
        )
    ):
        return None
    return region


@dataclass(frozen=True)
class BoundPreparationTransfers:
    first: int
    stop: int
    sources: tuple[SharedBufferRegion, ...]
    destinations: tuple[SharedBufferRegion, ...]


def bind_preparation_transfer_span(
    physical: AcceptedPreparationStorage,
    first: int,
    stop: int,
    reads: tuple[str, ...],
    writes: tuple[str, ...],
) -> BoundPreparationTransfers | None:
    """Check disjoint complete owners over one actual transfer interval.

    Several members may refer to the same native union. Keep that full union once;
    the caller still proves each member's original layout and byte offset.
    """
    if type(first) is not int or type(stop) is not int or not 0 <= first < stop:
        return None
    sources: list[SharedBufferRegion] = []
    destinations: list[SharedBufferRegion] = []
    for names, target in ((reads, sources), (writes, destinations)):
        for name in names:
            region = preparation_transfer_owner(physical, name, first, stop)
            if region is None:
                return None
            if region not in target:
                target.append(region)
    if any(
        source.overlaps_storage(target) for source in sources for target in destinations
    ):
        return None
    return BoundPreparationTransfers(first, stop, tuple(sources), tuple(destinations))
