from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AutotuneEffort = Literal["none", "quick", "full"]


@dataclass(frozen=True)
class PatternSearchConfig:
    initial_population: int
    copies: int
    max_generations: int


@dataclass(frozen=True)
class DifferentialEvolutionConfig:
    population_size: int
    max_generations: int


@dataclass(frozen=True)
class RandomSearchConfig:
    count: int


@dataclass(frozen=True)
class MultiFidelityBOConfig:
    n_low_fidelity: int
    n_medium_fidelity: int
    n_high_fidelity: int
    n_ultra_fidelity: int
    fidelity_low: int
    fidelity_medium: int
    fidelity_high: int
    fidelity_ultra: int


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

MULTIFIDELITY_BO_DEFAULTS = MultiFidelityBOConfig(
    n_low_fidelity=200,
    n_medium_fidelity=30,
    n_high_fidelity=10,
    n_ultra_fidelity=3,
    fidelity_low=5,
    fidelity_medium=15,
    fidelity_high=50,
    fidelity_ultra=500,
)


@dataclass(frozen=True)
class AutotuneEffortProfile:
    pattern_search: PatternSearchConfig | None
    differential_evolution: DifferentialEvolutionConfig | None
    random_search: RandomSearchConfig | None
    multifidelity_bo: MultiFidelityBOConfig | None = None
    rebenchmark_threshold: float = 1.5


_PROFILES: dict[AutotuneEffort, AutotuneEffortProfile] = {
    "none": AutotuneEffortProfile(
        pattern_search=None,
        differential_evolution=None,
        random_search=None,
        multifidelity_bo=None,
    ),
    "quick": AutotuneEffortProfile(
        pattern_search=PatternSearchConfig(
            initial_population=30,
            copies=2,
            max_generations=5,
        ),
        differential_evolution=DifferentialEvolutionConfig(
            population_size=20,
            max_generations=8,
        ),
        random_search=RandomSearchConfig(
            count=100,
        ),
        multifidelity_bo=MultiFidelityBOConfig(
            n_low_fidelity=50,
            n_medium_fidelity=10,
            n_high_fidelity=3,
            n_ultra_fidelity=1,
            fidelity_low=5,
            fidelity_medium=15,
            fidelity_high=50,
            fidelity_ultra=200,
        ),
        rebenchmark_threshold=0.9,  # <1.0 effectively disables rebenchmarking
    ),
    "full": AutotuneEffortProfile(
        pattern_search=PATTERN_SEARCH_DEFAULTS,
        differential_evolution=DIFFERENTIAL_EVOLUTION_DEFAULTS,
        random_search=RANDOM_SEARCH_DEFAULTS,
        multifidelity_bo=MULTIFIDELITY_BO_DEFAULTS,
    ),
}


def get_effort_profile(effort: AutotuneEffort) -> AutotuneEffortProfile:
    return _PROFILES[effort]
