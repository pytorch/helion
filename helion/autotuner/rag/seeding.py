"""Fixed-size LFBO seeded-population builder and global attempt budget (§3).

Both pieces are pure and dependency-injected so they need no autotuner, CUDA, or
retrieval dependency: side-effecting operations (validation, normalization,
random draws) are passed in as callables. This lets the equal-budget accounting
be tested on CPU with fakes and reused by the RAG execution wrapper, which wires
these callables to the live ``base_search`` path.

Equal-budget means an identical global unique-candidate attempt limit and an
identical fixed initial-population size ``N`` -- not identical successful
benchmark counts. Compilation and validation success are treatment *outcomes*,
so the builder reports realized attempted / valid / duplicate counts separately
and never silently shrinks a population: if ``N`` unique valid configs cannot be
reached within the frozen draw cap (or before the global budget is exhausted) the
arm repetition fails with :class:`PopulationUnderfilled`.

See ``docs/rag_autotuning_experiment.md``.
"""

from __future__ import annotations

import dataclasses
from typing import Callable
from typing import Hashable
from typing import Sequence
from typing import TypeVar

from ..candidate_budget import AttemptBudget
from ..candidate_budget import AttemptCategory

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class SeedingResult:
    """Realized counts from one seeded-population build, reported separately (§3).

    ``attempted`` equals ``valid + duplicates + invalid`` and equals the number
    of attempts this build charged to the :class:`AttemptBudget`. ``random_draws``
    is the subset of attempts that came from ``draw_random`` rather than seeds.
    """

    attempted: int
    valid: int
    duplicates: int
    invalid: int
    random_draws: int


class PopulationUnderfilled(Exception):
    """Raised when ``N`` unique valid configs cannot be reached (§3).

    The arm repetition fails rather than silently shrinking the population. The
    attached fields record why the build stopped short.
    """

    def __init__(
        self,
        *,
        requested: int,
        realized: int,
        draw_cap: int,
        budget_exhausted: bool,
    ) -> None:
        self.requested = requested
        self.realized = realized
        self.draw_cap = draw_cap
        self.budget_exhausted = budget_exhausted
        reason = "global attempt budget exhausted" if budget_exhausted else "draw cap"
        super().__init__(
            f"seeded population underfilled: {realized}/{requested} unique valid "
            f"configs ({reason}, draw_cap={draw_cap})"
        )


def build_seeded_population(
    seeds: Sequence[T],
    *,
    n: int,
    num_neighbors_cap: int,
    draw_random: Callable[[], T],
    is_valid: Callable[[T], bool],
    flatten_key: Callable[[T], Hashable],
    budget: AttemptBudget,
    draw_cap: int,
) -> tuple[list[T], SeedingResult]:
    """Build a fixed-size initial population of exactly ``n`` unique valid configs.

    Injected seeds are capped to ``num_neighbors_cap`` before construction, then
    validated and deduplicated first; fresh random candidates from ``draw_random``
    fill the remainder until exactly ``n`` unique valid configs exist or the frozen
    ``draw_cap`` random-draw limit is reached (or the global ``budget`` is
    exhausted). Every considered candidate -- accepted seed, accepted draw,
    invalid, or duplicate -- is charged once to ``budget``.

    Args:
        seeds: Candidate configs from retrieval, most-preferred first.
        n: Exact target population size.
        num_neighbors_cap: Max injected seeds admitted before construction.
        draw_random: Draws one fresh random candidate.
        is_valid: True if a candidate is a valid config (invalid ones dropped).
        flatten_key: Hashable normalized dedup key for a valid candidate.
        budget: Global attempt counter charged for every considered candidate.
        draw_cap: Max number of ``draw_random`` calls.

    Returns:
        The list of exactly ``n`` unique valid configs and a :class:`SeedingResult`.

    Raises:
        PopulationUnderfilled: If ``n`` cannot be reached within the caps.
    """
    population: list[T] = []
    seen: set[Hashable] = set()
    valid = 0
    duplicates = 0
    invalid = 0
    random_draws = 0

    def admit(candidate: T) -> None:
        nonlocal valid, duplicates, invalid
        if not is_valid(candidate):
            invalid += 1
            budget.record(AttemptCategory.INVALID)
            return
        key = flatten_key(candidate)
        if key in seen:
            duplicates += 1
            budget.record(AttemptCategory.DUPLICATE)
            return
        seen.add(key)
        population.append(candidate)
        valid += 1
        budget.record(AttemptCategory.INITIAL_POPULATION)

    # Cap injected seeds before construction, then validate + dedup them first.
    for seed in list(seeds)[:num_neighbors_cap]:
        if len(population) >= n or budget.exhausted:
            break
        admit(seed)

    # Draw fresh random candidates until exactly N unique valid configs exist.
    while len(population) < n and random_draws < draw_cap and not budget.exhausted:
        random_draws += 1
        admit(draw_random())

    result = SeedingResult(
        attempted=valid + duplicates + invalid,
        valid=valid,
        duplicates=duplicates,
        invalid=invalid,
        random_draws=random_draws,
    )
    if len(population) != n:
        raise PopulationUnderfilled(
            requested=n,
            realized=len(population),
            draw_cap=draw_cap,
            budget_exhausted=budget.exhausted,
        )
    return population, result
