from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, repr=False)
class TileDependencySchedule:
    """Opt in to cross-grid scheduling of dependent top-level tile loops.

    Each source tile body remains opaque: lowering may change only its physical
    worker assignment and add waits/publications at existing tile or ordered-loop
    boundaries. It may not rewrite arithmetic, split stores, or reassociate a
    reduction. The compiler derives legal task groups, worker cohorts, and
    publication layout from dependency/access maps.
    """

    def __repr__(self) -> str:
        return "helion.TileDependencySchedule()"
