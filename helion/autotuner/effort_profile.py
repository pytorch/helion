from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import TYPE_CHECKING
from typing import Literal

if TYPE_CHECKING:
    from .config_generation import ConfigGeneration

AutotuneEffort = Literal["none", "quick", "full", "auto"]

log = logging.getLogger(__name__)
InitialPopulation = Literal["from_random", "from_default", "from_best_available"]


@dataclass(frozen=True)
class PatternSearchConfig:
    initial_population: int
    copies: int
    max_generations: int
    initial_population_strategy: InitialPopulation = "from_random"
    compile_timeout_lower_bound: float = 30.0
    compile_timeout_quantile: float = 0.9


@dataclass(frozen=True)
class DifferentialEvolutionConfig:
    population_size: int
    max_generations: int
    initial_population_strategy: InitialPopulation = "from_random"
    compile_timeout_lower_bound: float = 30.0
    compile_timeout_quantile: float = 0.9


@dataclass(frozen=True)
class RandomSearchConfig:
    count: int


# Default values for each algorithm (single source of truth)
PATTERN_SEARCH_DEFAULTS = PatternSearchConfig(
    initial_population=100,
    copies=5,
    max_generations=20,
)

DIFFERENTIAL_EVOLUTION_DEFAULTS = DifferentialEvolutionConfig(
    population_size=40,
    max_generations=40,
)

RANDOM_SEARCH_DEFAULTS = RandomSearchConfig(
    count=1000,
)


@dataclass(frozen=True)
class AutotuneEffortProfile:
    pattern_search: PatternSearchConfig | None
    lfbo_pattern_search: PatternSearchConfig | None
    differential_evolution: DifferentialEvolutionConfig | None
    random_search: RandomSearchConfig | None
    finishing_rounds: int = 0
    rebenchmark_threshold: float = 1.5


_PROFILES: dict[AutotuneEffort, AutotuneEffortProfile] = {
    "none": AutotuneEffortProfile(
        pattern_search=None,
        lfbo_pattern_search=None,
        differential_evolution=None,
        random_search=None,
    ),
    "quick": AutotuneEffortProfile(
        pattern_search=PatternSearchConfig(
            initial_population=30,
            copies=2,
            max_generations=5,
            initial_population_strategy="from_default",
        ),
        lfbo_pattern_search=PatternSearchConfig(
            initial_population=30,
            copies=2,
            max_generations=5,
            initial_population_strategy="from_default",
        ),
        differential_evolution=DifferentialEvolutionConfig(
            population_size=20,
            max_generations=8,
            initial_population_strategy="from_default",
        ),
        random_search=RandomSearchConfig(
            count=100,
        ),
        finishing_rounds=0,
        rebenchmark_threshold=0.9,  # <1.0 effectively disables rebenchmarking
    ),
    "full": AutotuneEffortProfile(
        pattern_search=PATTERN_SEARCH_DEFAULTS,
        lfbo_pattern_search=PATTERN_SEARCH_DEFAULTS,
        differential_evolution=DIFFERENTIAL_EVOLUTION_DEFAULTS,
        random_search=RANDOM_SEARCH_DEFAULTS,
    ),
}


def get_effort_profile(effort: AutotuneEffort) -> AutotuneEffortProfile:
    return _PROFILES[effort]


# Thresholds for automatic effort recommendation.
# A space with fewer configs than _QUICK_THRESHOLD can be explored
# quickly; spaces beyond _FULL_THRESHOLD benefit from full search.
_QUICK_THRESHOLD = 50_000
_FULL_THRESHOLD = 500_000


def recommend_effort(config_gen: ConfigGeneration) -> AutotuneEffort:
    """Recommend an autotuning effort level based on search space analysis.

    Examines the kernel's configuration space (block sizes, loop orders,
    warps, etc.) and returns ``"quick"`` or ``"full"`` depending on how
    large and complex the space is.  ``"none"`` is never recommended
    because the caller explicitly asked for automatic selection.

    Heuristic factors:

    * **Search space size**: product of per-fragment cardinalities.
      Small spaces (< 50 k configs) → ``"quick"``.
      Large spaces (> 500 k) → ``"full"``.
    * **Permutation dimensions**: loop orderings grow factorially and
      benefit from longer surrogate-assisted search.
    * **Number of block-size dimensions**: more tiling dimensions
      create harder optimization landscapes.

    The recommendation is logged so users can see why a particular
    effort was chosen and override it if desired.
    """
    space_size = config_gen.search_space_size()
    num_block_dims = len(config_gen.block_size_indices)
    num_fragments = len(config_gen.flat_spec)

    # Detect presence of permutation fragments (factorial growth)
    from .config_fragment import PermutationFragment

    has_permutations = any(
        isinstance(spec, PermutationFragment) for spec in config_gen.flat_spec
    )

    # Base decision on space size
    if space_size < _QUICK_THRESHOLD:
        effort: AutotuneEffort = "quick"
    elif space_size > _FULL_THRESHOLD:
        effort = "full"
    else:
        # Middle ground — use heuristics to break the tie
        if has_permutations or num_block_dims >= 3 or num_fragments >= 10:
            effort = "full"
        else:
            effort = "quick"

    log.info(
        "recommend_effort: space_size=%s, fragments=%d, block_dims=%d, "
        "has_permutations=%s → %r",
        f"{space_size:,}",
        num_fragments,
        num_block_dims,
        has_permutations,
        effort,
    )
    log_space_size = math.log10(max(space_size, 1))
    log.debug(
        "recommend_effort: log10(space)=%.1f, quick_threshold=%s, full_threshold=%s",
        log_space_size,
        f"{_QUICK_THRESHOLD:,}",
        f"{_FULL_THRESHOLD:,}",
    )
    return effort
