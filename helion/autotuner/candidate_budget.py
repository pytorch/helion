"""Global candidate-attempt accounting for autotuner searches."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .rag.instrumentation import InstrumentationCollector

_RANDOM_REPLACEMENT_DRAW_MULTIPLIER = 20
_MIN_RANDOM_REPLACEMENT_DRAW_CAP = 100


def random_replacement_draw_cap(population_size: int) -> int:
    """Return the shared frozen replacement-draw cap for a population size."""
    return max(
        _MIN_RANDOM_REPLACEMENT_DRAW_CAP,
        population_size * _RANDOM_REPLACEMENT_DRAW_MULTIPLIER,
    )


class AttemptCategory(enum.Enum):
    """Mutually exclusive outcomes for one candidate attempt."""

    INITIAL_POPULATION = "initial_population"
    INVALID = "invalid"
    DUPLICATE = "duplicate"
    GENERATION = "generation"
    LLM_PROPOSED = "llm_proposed"


class AttemptBudget:
    """A category counter with an optional immutable global attempt limit."""

    def __init__(self, limit: int | None) -> None:
        if limit is not None and limit <= 0:
            raise ValueError(f"candidate attempt limit must be positive, got {limit}")
        self._limit = limit
        self._counts: dict[AttemptCategory, int] = dict.fromkeys(AttemptCategory, 0)

    @property
    def limit(self) -> int | None:
        return self._limit

    def record(self, category: AttemptCategory, n: int = 1) -> bool:
        """Record all ``n`` attempts, or return False without overshooting."""
        if n <= 0:
            raise ValueError(f"candidate attempt count must be positive, got {n}")
        remaining = self.remaining()
        if remaining is not None and n > remaining:
            return False
        self._counts[category] += n
        return True

    def spent(self) -> int:
        return sum(self._counts.values())

    def remaining(self) -> int | None:
        if self._limit is None:
            return None
        return self._limit - self.spent()

    @property
    def exhausted(self) -> bool:
        return self._limit is not None and self.spent() >= self._limit

    def spent_by(self, category: AttemptCategory) -> int:
        return self._counts[category]

    def by_category(self) -> dict[AttemptCategory, int]:
        return dict(self._counts)


class SharedAttemptState:
    """Run-level candidate accounting shared across a composed search's stages.

    A two-stage search -- e.g. an LLM seed stage followed by an LFBO stage --
    must charge a single global attempt ceiling, deduplicate candidates across
    both stages, and record one continuous instrumentation trajectory. This
    bundles the one :class:`AttemptBudget`, the shared normalized-config dedup
    set, and the optional shared instrumentation collector so both stages
    reference the same objects. Per-category accounting already lives on the
    budget (:meth:`AttemptBudget.by_category`).

    Adopt it on each stage with ``BaseSearch.adopt_shared_attempt_state`` before
    that stage records any attempt. Single-stage tuners never build one and keep
    their per-search budget/dedup set unchanged.
    """

    def __init__(
        self,
        budget: AttemptBudget,
        *,
        collector: InstrumentationCollector | None = None,
    ) -> None:
        self.budget = budget
        self.attempted_configs: set[Config] = set()
        self.collector = collector


class CandidatePopulationUnderfilled(RuntimeError):
    """A capped arm could not construct its frozen initial population size."""

    def __init__(self, *, requested: int, realized: int) -> None:
        self.requested = requested
        self.realized = realized
        super().__init__(
            f"initial candidate population underfilled: {realized}/{requested}"
        )
