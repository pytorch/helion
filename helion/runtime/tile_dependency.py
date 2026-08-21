from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True, repr=False)
class TileDependencySchedule:
    """Opt in to cross-grid scheduling of dependent top-level tile loops.

    Each source tile body remains opaque: lowering may change only its physical
    worker assignment and add waits/publications at existing tile or ordered-loop
    boundaries. It may not rewrite arithmetic, split stores, or reassociate a
    reduction. The compiler derives legal task groups and worker cohorts from
    dependency/access maps. These optional values select among compiler-generated
    legal schedules; ``None`` asks the compiler to choose its derived default.
    """

    epoch_replicas: int | None = None
    tile_dependency_stages: int | None = None
    continuation_split: int | None = None
    producer_order: Literal["physical", "consumer_major"] | None = None

    def __post_init__(self) -> None:
        for name in (
            "epoch_replicas",
            "tile_dependency_stages",
            "continuation_split",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None, got {value!r}")
        if self.producer_order not in (None, "physical", "consumer_major"):
            raise ValueError(
                "producer_order must be 'physical', 'consumer_major', or None, "
                f"got {self.producer_order!r}"
            )

    def __repr__(self) -> str:
        values = (
            ("epoch_replicas", self.epoch_replicas),
            ("tile_dependency_stages", self.tile_dependency_stages),
            ("continuation_split", self.continuation_split),
            ("producer_order", self.producer_order),
        )
        arguments = ", ".join(
            f"{name}={value!r}" for name, value in values if value is not None
        )
        return f"helion.TileDependencySchedule({arguments})"
