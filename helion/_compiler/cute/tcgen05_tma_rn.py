"""An explicit TF32 MMA approximation with conversion performed by TMA.

This policy does not implement an explicit RNA cast. Admission belongs to the
typed direct-load contraction proof and mapped ``dot_precision='tf32'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

CONVERSION_KEY = "cute_mma_f32_conversion"
AUTO = "auto"
TMA_RN = "tma_rn"
WARP_RAW = "warp_raw"


@dataclass(frozen=True)
class Tcgen05TmaRnRoles:
    """One-CTA TMA-to-UMMA pipeline, with no SIMT conversion consumers."""

    epilogue_warp_ids: ClassVar[tuple[int, ...]] = (0, 1, 2, 3)
    mma_warp: ClassVar[int] = 4
    raw_tma_warp: ClassVar[int] = 5
    converter_warps: ClassVar[int] = 0
    converter_warp_ids: ClassVar[tuple[int, ...]] = ()
    scheduler_warp: ClassVar[int] = 6
    padding_warp_ids: ClassVar[tuple[int, ...]] = (7,)
    threads_per_cta: ClassVar[int] = 256
    scheduler_self_consumer: ClassVar[bool] = False
    scheduler_consumer_warp_ids: ClassVar[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)
    scheduler_empty_warp_arrivals: ClassVar[int] = 6

    def cache_identity(self) -> tuple[object, ...]:
        return (
            "tcgen05_tma_rn_roles_v1",
            self.epilogue_warp_ids,
            self.mma_warp,
            self.raw_tma_warp,
            self.scheduler_warp,
            self.padding_warp_ids,
            self.threads_per_cta,
            self.scheduler_consumer_warp_ids,
        )
