from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING

from .. import exc
from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .base_search import performance
from .effort_profile import PATTERN_SEARCH_DEFAULTS

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ..autotuner.effort_profile import AutotuneEffortProfile
    from ..runtime.config import Config
    from ..runtime.settings import Settings
    from .base_search import _AutotunableKernel
    from .config_generation import FlatConfig


class InitialPopulationStrategy(enum.Enum):
    """Strategy for generating the initial population for search algorithms."""

    FROM_RANDOM = "from_random"
    """Generate a random population of configurations."""

    FROM_BEST_AVAILABLE = "from_best_available"
    """Start from default config plus up to 20 best matching cached configs from previous runs."""


class PatternSearch(PopulationBasedSearch):
    """Search that explores single-parameter perturbations around the current best."""

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        initial_population: int = PATTERN_SEARCH_DEFAULTS.initial_population,
        copies: int = PATTERN_SEARCH_DEFAULTS.copies,
        max_generations: int = PATTERN_SEARCH_DEFAULTS.max_generations,
        min_improvement_delta: float = 0.001,
        initial_population_strategy: InitialPopulationStrategy | None = None,
        best_available_pad_random: bool = PATTERN_SEARCH_DEFAULTS.best_available_pad_random,
        num_neighbors_cap: int = -1,
        finishing_rounds: int = 0,
        compile_timeout_lower_bound: float = PATTERN_SEARCH_DEFAULTS.compile_timeout_lower_bound,
        compile_timeout_quantile: float = PATTERN_SEARCH_DEFAULTS.compile_timeout_quantile,
    ) -> None:
        """
        Create a PatternSearch autotuner.

        Args:
            kernel: The kernel to be autotuned.
            args: The arguments to be passed to the kernel.
            initial_population: The number of random configurations to generate for the initial population.
            copies: Count of top Configs to run pattern search on.
            max_generations: The maximum number of generations to run.
            min_improvement_delta: Relative stop threshold; stop if abs(best/current - 1) < this.
            initial_population_strategy: Strategy for generating the initial population.
                FROM_RANDOM generates initial_population random configs.
                FROM_BEST_AVAILABLE uses cached configs from prior runs, and fills the
                remainder with random configs when best_available_pad_random is True.
                Can be overridden by HELION_AUTOTUNER_INITIAL_POPULATION env var (handled in default_autotuner_fn).
                If None is passed, defaults to FROM_RANDOM.
            best_available_pad_random: When True and using FROM_BEST_AVAILABLE, pad the
                cached configs with random configs to reach initial_population size.
                When False, use only the default and cached configs (no random padding).
            num_neighbors_cap: Maximum number of neighbors to explore per generation. -1 means no cap.
                Set HELION_CAP_AUTOTUNE_NUM_NEIGHBORS=N to override.
            finishing_rounds: Number of finishing rounds to run after the main search.
            compile_timeout_lower_bound: Lower bound for adaptive compile timeout in seconds.
            compile_timeout_quantile: Quantile of compile times to use for adaptive timeout.
        """
        super().__init__(kernel, args, finishing_rounds=finishing_rounds)
        if initial_population_strategy is None:
            initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        self.initial_population_strategy = initial_population_strategy
        self.best_available_pad_random = best_available_pad_random
        self.copies = copies
        self.max_generations = max_generations
        self.min_improvement_delta = min_improvement_delta
        self.initial_population = initial_population
        self.num_neighbors_cap = num_neighbors_cap
        self.compile_timeout_lower_bound = compile_timeout_lower_bound
        self.compile_timeout_quantile = compile_timeout_quantile

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        from ..runtime.settings import _env_get_int
        from ..runtime.settings import _get_initial_population_strategy

        assert profile.pattern_search is not None
        strategy = _get_initial_population_strategy(
            profile.pattern_search.initial_population_strategy,
            settings.autotune_initial_population_strategy,
        )
        return {
            "initial_population": profile.pattern_search.initial_population,
            "copies": profile.pattern_search.copies,
            "max_generations": profile.pattern_search.max_generations,
            "initial_population_strategy": strategy,
            "best_available_pad_random": profile.pattern_search.best_available_pad_random,
            "num_neighbors_cap": _env_get_int("HELION_CAP_AUTOTUNE_NUM_NEIGHBORS", -1),
            **super().get_kwargs_from_profile(profile, settings),
        }

    def _generate_initial_population_flat(self) -> list[FlatConfig]:
        """
        Generate the initial population of flat configurations based on the strategy.

        Returns:
            A list of flat configurations for the initial population.
        """
        if (
            self.initial_population_strategy
            == InitialPopulationStrategy.FROM_BEST_AVAILABLE
        ):
            pop = self._generate_best_available_population_flat()
            if self.best_available_pad_random:
                n_random = max(0, self.initial_population - len(pop))
                pop.extend(self.config_gen.random_flat() for _ in range(n_random))
            return pop
        return self.config_gen.random_population_flat(self.initial_population)

    def _autotune(self) -> Config:
        initial_population_name = self.initial_population_strategy.name
        self.log(
            f"Starting PatternSearch with initial_population={initial_population_name}, copies={self.copies}, max_generations={self.max_generations}"
        )
        visited: set[Config] = set()
        self.population = []
        for flat_config in self._generate_initial_population_flat():
            member = self.make_unbenchmarked(flat_config)
            if member.config not in visited:
                visited.add(member.config)
                self.population.append(member)
        self.benchmark_population(self.population, desc="Initial population")

        # Compute adaptive compile timeout based on initial population compile times
        self.set_adaptive_compile_timeout(
            self.population,
            min_seconds=self.compile_timeout_lower_bound,
            quantile=self.compile_timeout_quantile,
        )

        # again with higher accuracy
        self.rebenchmark_population(self.population, desc="Verifying initial results")
        self.population.sort(key=performance)
        starting_points = []
        for member in self.population[: self.copies]:
            if math.isfinite(member.perf):  # filter failed compiles
                starting_points.append(member)
        self.log(
            f"Initial random population of {len(self.population)}, {len(starting_points)} starting points:",
            self.statistics,
        )
        if not starting_points:
            raise exc.NoConfigFound

        search_copies = [self._pattern_search_from(m, visited) for m in starting_points]
        for generation in range(1, self.max_generations + 1):
            prior_best = self.best
            new_population = {id(prior_best): prior_best}
            num_neighbors = 0
            num_active = 0
            for search_copy in search_copies:
                added = next(search_copy, ())
                if added:
                    assert len(added) > 1
                    num_active += 1
                    num_neighbors += len(added) - 1
                    for member in added:
                        new_population[id(member)] = member
            if num_active == 0:
                break

            # Log generation header before compiling/benchmarking
            self.log(
                f"Generation {generation} starting: {num_neighbors} neighbors, {num_active} active search path(s)"
            )

            self.population = [*new_population.values()]
            # compile any unbenchmarked members in parallel
            unbenchmarked = [m for m in self.population if len(m.perfs) == 0]
            if unbenchmarked:
                self.set_generation(generation)
                self.benchmark_population(
                    unbenchmarked, desc=f"Generation {generation}:"
                )
            # higher-accuracy rebenchmark
            self.rebenchmark_population(
                self.population, desc=f"Generation {generation}: verifying top configs"
            )
            # Log final statistics for this generation
            self.log(f"Generation {generation} complete:", self.statistics)

        # Run finishing phase to simplify the best configuration
        best = self.run_finishing_phase(self.best, self.finishing_rounds)
        return best.config

    def _pattern_search_from(
        self, current: PopulationMember, visited: set[Config]
    ) -> Iterator[list[PopulationMember]]:
        """
        Run a single copy of pattern search from the given starting point.

        We use a generator and yield the new population at each generation so that we can
        run multiple copies of pattern search in parallel.
        """
        for _ in range(self.max_generations):
            candidates = [current]
            for flat_config in self._generate_neighbors(current.flat_values):
                new_member = self.make_unbenchmarked(flat_config)
                if new_member.config not in visited:
                    visited.add(new_member.config)
                    candidates.append(new_member)
            if len(candidates) <= 1:
                return  # no new candidates, stop searching
            yield candidates  # yield new population to benchmark in parallel
            # update search copy and check early stopping criteria
            best = min(candidates, key=performance)
            if self._check_early_stopping(best, current):
                return
            current = best

    def _check_early_stopping(
        self, best: PopulationMember, current: PopulationMember
    ) -> bool:
        """
        Check if early stopping criteria are met for the search copy

        Early stops if either the best config has not changed or if
        the relative improvement is smaller than a user-specified delta

        Returns:
            True the search copy is terminated, False otherwise.
        """
        if best is current:
            return True  # no improvement, stop searching
        # Stop if the relative improvement is smaller than a user-specified delta
        return bool(
            self.min_improvement_delta > 0.0
            and math.isfinite(best.perf)
            and math.isfinite(current.perf)
            and current.perf != 0.0
            and abs(best.perf / current.perf - 1.0) < self.min_improvement_delta
        )

    def shrink_neighbors(self, neighbors: list[FlatConfig]) -> list[FlatConfig]:
        if self.num_neighbors_cap > 0:
            return neighbors[: self.num_neighbors_cap]
        return neighbors

    def _generate_neighbors(self, base: FlatConfig) -> list[FlatConfig]:
        """
        Generate neighboring configurations by changing one or two parameters at a time.
        """
        overridden = self.config_gen.overridden_flat_indices
        candidates_by_index = [
            spec.pattern_neighbors(base[index]) if index not in overridden else []
            for index, spec in enumerate(self.config_gen.flat_spec)
        ]
        assert len(candidates_by_index) == len(base)
        neighbors: list[FlatConfig] = []

        # Add all single-parameter changes
        for index, candidates in enumerate(candidates_by_index):
            for candidate_value in candidates:
                new_flat = [*base]
                new_flat[index] = candidate_value
                neighbors.append(new_flat)

        # Block sizes are important enough to try pairs of changes at a time
        block_indices = [
            i for i in self.config_gen.block_size_indices if i not in overridden
        ]
        for i_pos, first in enumerate(block_indices):
            first_candidates = candidates_by_index[first]
            if not first_candidates:
                continue
            for second in block_indices[i_pos + 1 :]:
                second_candidates = candidates_by_index[second]
                if not second_candidates:
                    continue
                for first_value in first_candidates:
                    for second_value in second_candidates:
                        new_flat = [*base]
                        new_flat[first] = first_value
                        new_flat[second] = second_value
                        neighbors.append(new_flat)

        return self.shrink_neighbors(neighbors)
