"""Shared scratch borrowed from one producer group before its first access.

The physical adapter must bind the original owner and emit a completion wait
after that group's register release and before any of its buffer accesses.
This object validates the resource subdivision; it does not infer that wait.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from . import chained_matmul as chain
from .warp_specialized_plan import SharedBufferRegion
from .warp_specialized_plan import WarpRole
from .warp_specialized_plan import WarpSpecializedPipelinePlan


@dataclass(frozen=True)
class StartupSharedReuse:
    owner: SharedBufferRegion
    role: WarpRole
    groups: int
    group: int
    byte_size: int

    @property
    def first_warp(self) -> int:
        return self.role.first_warp + self.group * (self.role.warp_count // self.groups)

    @property
    def last_warp(self) -> int:
        return self.first_warp + self.role.warp_count // self.groups

    def resources(
        self, pipeline: WarpSpecializedPipelinePlan
    ) -> tuple[WarpSpecializedPipelinePlan, SharedBufferRegion]:
        """Split all owner bytes without changing unrelated groups' lifetimes."""
        pipeline.validate()
        if (
            type(self.groups) is not int
            or self.groups <= 0
            or type(self.group) is not int
            or not 0 <= self.group < self.groups
            or type(self.byte_size) is not int
            or self.byte_size <= 0
            or pipeline.shared_buffer(self.owner.name) != self.owner
            or pipeline.role(self.role.name) != self.role
            or self.owner.byte_size % self.groups
            or self.role.warp_count % self.groups
            or self.role.first_warp % 4
            or (self.role.warp_count // self.groups) % 4
            or self.owner.live_from + 1 >= self.owner.live_until
        ):
            raise chain._UnsupportedChain("invalid startup shared owner subdivision")
        stride = self.owner.byte_size // self.groups
        begin = self.owner.byte_offset + self.group * stride
        if self.byte_size > stride or begin % 1024:
            raise chain._UnsupportedChain("invalid startup scratch extent")
        scratch = SharedBufferRegion(
            "startup_copy_scratch",
            begin,
            self.byte_size,
            self.owner.live_from,
            self.owner.live_from + 1,
            1024,
        )
        parts = []
        for group in range(self.groups):
            parts.append(
                replace(
                    self.owner,
                    name=f"{self.owner.name}_{group}",
                    byte_offset=self.owner.byte_offset + group * stride,
                    byte_size=stride,
                    live_from=self.owner.live_from + (group == self.group),
                )
            )
        revised = replace(
            pipeline,
            shared_buffers=tuple(
                part
                for region in pipeline.shared_buffers
                for part in (parts if region == self.owner else (region,))
            ),
        )
        replace(revised, shared_buffers=(*revised.shared_buffers, scratch)).validate()
        return revised, scratch
