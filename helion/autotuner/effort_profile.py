from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AutotuneEffort = Literal["none", "quick", "full"]
InitialPopulation = Literal["from_random", "from_best_available"]

DEFAULT_LLM_MODEL = "gpt-5-2"
DEFAULT_LLM_CONFIGS_PER_ROUND = 15
DEFAULT_LLM_MAX_ROUNDS = 4
DEFAULT_LLM_INITIAL_RANDOM_CONFIGS = 10
DEFAULT_LLM_COMPILE_TIMEOUT_S: int | None = 15


@dataclass(frozen=True)
class PatternSearchConfig:
    initial_population: int
    copies: int
    max_generations: int
    initial_population_strategy: InitialPopulation = "from_random"
    best_available_pad_random: bool = True
    compile_timeout_lower_bound: float = 30.0
    compile_timeout_quantile: float = 0.9


@dataclass(frozen=True)
class DifferentialEvolutionConfig:
    population_size: int
    max_generations: int
    initial_population_strategy: InitialPopulation = "from_random"
    best_available_pad_random: bool = True
    compile_timeout_lower_bound: float = 30.0
    compile_timeout_quantile: float = 0.9


@dataclass(frozen=True)
class RandomSearchConfig:
    count: int


@dataclass(frozen=True)
class LLMSearchConfig:
    model: str = DEFAULT_LLM_MODEL
    configs_per_round: int = DEFAULT_LLM_CONFIGS_PER_ROUND
    max_rounds: int = DEFAULT_LLM_MAX_ROUNDS
    initial_random_configs: int = DEFAULT_LLM_INITIAL_RANDOM_CONFIGS
    compile_timeout_s: int | None = DEFAULT_LLM_COMPILE_TIMEOUT_S


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

# Full guided-search defaults used by direct LLMGuidedSearch construction.
LLM_SEARCH_DEFAULTS = LLMSearchConfig()

# Quick LLM preset used by the quick effort profile and the hybrid seed stage.
QUICK_LLM_SEARCH_DEFAULTS = LLMSearchConfig(
    max_rounds=1,
)


@dataclass(frozen=True)
class AutotuneEffortProfile:
    pattern_search: PatternSearchConfig | None
    lfbo_pattern_search: PatternSearchConfig | None
    differential_evolution: DifferentialEvolutionConfig | None
    random_search: RandomSearchConfig | None
    llm_search: LLMSearchConfig | None = None
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
            initial_population_strategy="from_best_available",
            best_available_pad_random=False,
        ),
        lfbo_pattern_search=PatternSearchConfig(
            initial_population=30,
            copies=2,
            max_generations=5,
            initial_population_strategy="from_best_available",
            best_available_pad_random=False,
        ),
        differential_evolution=DifferentialEvolutionConfig(
            population_size=20,
            max_generations=8,
            initial_population_strategy="from_best_available",
            best_available_pad_random=False,
        ),
        random_search=RandomSearchConfig(
            count=100,
        ),
        llm_search=QUICK_LLM_SEARCH_DEFAULTS,
        finishing_rounds=0,
        rebenchmark_threshold=0.9,  # <1.0 effectively disables rebenchmarking
    ),
    "full": AutotuneEffortProfile(
        pattern_search=PATTERN_SEARCH_DEFAULTS,
        lfbo_pattern_search=PATTERN_SEARCH_DEFAULTS,
        differential_evolution=DIFFERENTIAL_EVOLUTION_DEFAULTS,
        random_search=RANDOM_SEARCH_DEFAULTS,
        llm_search=LLM_SEARCH_DEFAULTS,
    ),
}


def get_effort_profile(effort: AutotuneEffort) -> AutotuneEffortProfile:
    return _PROFILES[effort]
