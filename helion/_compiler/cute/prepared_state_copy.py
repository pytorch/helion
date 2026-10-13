"""Plan bit-preserving external-to-TMEM copies at existing state cuts.

The physical adapter owns precision, view interpretation and scratch lifetime.
This planner aggregates disjoint native panels and validates their resources.
It cannot move a read across a cut or create a consumer readiness event.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Protocol

from . import chained_matmul as chain

if TYPE_CHECKING:
    from .prepared_state_planner import StateTransferPlan
    from .startup_shared_reuse import StartupSharedReuse
    from .warp_specialized_plan import MBarrierRegion
    from .warp_specialized_plan import SharedBufferRegion
    from .warp_specialized_plan import WarpSpecializedPipelinePlan


def raw_state_copy_eligible(
    shape: tuple[int, ...], strides: tuple[int, ...], alignment: int
) -> bool:
    """Static rank-four raw32 TMA coordinates; other layouts retain LDG/STTM."""
    return (
        len(shape) == len(strides) == 4
        and all(type(size) is int and 0 < size < 2**31 for size in shape)
        and shape[2:] == (128, 128)
        and all(type(stride) is int for stride in strides)
        and strides[3] == 1
        and all(0 < stride < 2**38 and stride % 4 == 0 for stride in strides[:3])
        and alignment >= 16
    )


class StateCopyLease(Protocol):
    def facts(self, plan: RawStateCopyPlan) -> object:
        """Check original raw FP32 views and the scratch release/use boundaries."""
        ...


@dataclass(frozen=True)
class RawStateCopyPlan:
    schedule: StateTransferPlan
    lease: StateCopyLease
    pipeline: WarpSpecializedPipelinePlan
    scratch: SharedBufferRegion
    barriers: MBarrierRegion
    tmem_column: int
    rows: int = 128
    startup_reuse: StartupSharedReuse | None = None
    planes_per_transfer: int = 1
    _facts: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_facts", self._current())

    def _current(self) -> object:
        self.schedule.check()
        lease = self.lease.facts(self)
        requests = self.schedule.requests
        if not requests or len(self.schedule.cuts) != 1:
            raise chain._UnsupportedChain("raw copy requires one original state cut")
        cut = self.schedule.cuts[0]
        first = requests[0]
        ordered = sorted(requests, key=lambda item: item.source_view.offset)
        for panel, request in enumerate(ordered):
            if (
                request.read is not request.value
                or request.source is not first.source
                or request.result is not first.result
                or request.transform_before is not cut
                or request.store_before is not cut
                or request.complete_before is not cut
                or request.read_before is not None
                or request.source_view.offset != panel * 32
                or request.destination_view.offset != panel * 32
                or request.source_view.width != 32
                or request.destination_view.width != 32
                or replace(request.source_view, offset=first.source_view.offset).facts()
                != first.source_view.facts()
                or replace(
                    request.destination_view, offset=first.destination_view.offset
                ).facts()
                != first.destination_view.facts()
            ):
                raise chain._UnsupportedChain("raw copy changed original state panels")
        columns = 32 * len(requests)
        if (
            type(self.rows) is not int
            or self.rows != 128
            or type(self.tmem_column) is not int
            or self.tmem_column < 0
            or self.tmem_column % 32
            or self.tmem_column + columns > self.pipeline.tmem_columns
            or type(self.planes_per_transfer) is not int
            or self.planes_per_transfer not in (1, 2)
            or columns % (32 * self.planes_per_transfer)
            or self.scratch.byte_size != self.rows * 32 * self.planes_per_transfer * 4
            or (self.planes_per_transfer == 2 and self.startup_reuse is None)
            or self.scratch.alignment < 1024
            or self.barriers.stages
            != (
                4
                if self.startup_reuse is not None and self.planes_per_transfer == 1
                else 3
            )
            or self.barriers.arrivals != 1
        ):
            raise chain._UnsupportedChain("unsupported raw32 copy resources")
        # The lease maps graph boundaries to these half-open physical lifetimes.
        # The allocator still checks all other live buffers and barrier regions.
        pipeline = self.pipeline
        scratch = (self.scratch,)
        if self.startup_reuse is not None:
            pipeline, borrowed = self.startup_reuse.resources(pipeline)
            if borrowed.byte_size != self.scratch.byte_size:
                raise chain._UnsupportedChain("unequal raw copy scratch slots")
            if self.planes_per_transfer == 2:
                # A planar transfer uses only the borrowed contiguous region.
                if self.scratch != borrowed:
                    raise chain._UnsupportedChain("foreign planar copy scratch")
            else:
                scratch = (*scratch, borrowed)
        replace(
            pipeline,
            shared_buffers=(*pipeline.shared_buffers, *scratch),
            barriers=(*pipeline.barriers, self.barriers),
        ).validate()
        return (
            lease,
            tuple(request.facts() for request in requests),
            self.pipeline,
            self.scratch,
            self.barriers,
            self.tmem_column,
            self.rows,
            self.startup_reuse,
            self.planes_per_transfer,
        )

    def check(self) -> None:
        if self._current() != self._facts:
            raise chain._UnsupportedChain("raw state copy decision changed")

    def payload(self) -> tuple[int, ...]:
        self.check()
        result = (
            3
            if self.planes_per_transfer == 2
            else (1 if self.startup_reuse is None else 2),
            self.rows,
            32 * len(self.schedule.requests),
            32,
            self.scratch.byte_offset,
            self.tmem_column,
            self.barriers.byte_offset,
        )
        if self.startup_reuse is not None:
            reuse = self.startup_reuse
            _, borrowed = reuse.resources(self.pipeline)
            result += (
                self.planes_per_transfer
                if self.planes_per_transfer == 2
                else borrowed.byte_offset,
                reuse.first_warp,
                reuse.last_warp,
            )
        return result
