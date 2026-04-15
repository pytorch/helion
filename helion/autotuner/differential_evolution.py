from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .base_search import performance
from .base_search import population_statistics
from .effort_profile import DIFFERENTIAL_EVOLUTION_DEFAULTS
from .pattern_search import InitialPopulationStrategy
from helion._dist_utils import sync_seed

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ..autotuner.effort_profile import AutotuneEffortProfile
    from ..runtime.config import Config
    from ..runtime.settings import Settings
    from .base_search import _AutotunableKernel
    from .config_generation import FlatConfig


class DifferentialEvolutionSearch(PopulationBasedSearch):
    """
    A search strategy that uses differential evolution to find the best config.
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        population_size: int = DIFFERENTIAL_EVOLUTION_DEFAULTS.population_size,
        max_generations: int = DIFFERENTIAL_EVOLUTION_DEFAULTS.max_generations,
        crossover_rate: float = 0.8,
        immediate_update: bool | None = None,
        min_improvement_delta: float | None = None,
        patience: int | None = None,
        initial_population_strategy: InitialPopulationStrategy | None = None,
        best_available_pad_random: bool = DIFFERENTIAL_EVOLUTION_DEFAULTS.best_available_pad_random,
        finishing_rounds: int = 0,
        compile_timeout_lower_bound: float = DIFFERENTIAL_EVOLUTION_DEFAULTS.compile_timeout_lower_bound,
        compile_timeout_quantile: float = DIFFERENTIAL_EVOLUTION_DEFAULTS.compile_timeout_quantile,
    ) -> None:
        """
        Create a DifferentialEvolutionSearch autotuner.

        Args:
            kernel: The kernel to be autotuned.
            args: The arguments to be passed to the kernel.
            population_size: The size of the population.
            max_generations: The maximum number of generations to run.
            crossover_rate: The crossover rate for mutation.
            immediate_update: Whether to update population immediately after each evaluation.
            min_improvement_delta: Relative improvement threshold for early stopping.
                If None (default), early stopping is disabled.
            patience: Number of generations without improvement before stopping.
                If None (default), early stopping is disabled.
            initial_population_strategy: Strategy for generating the initial population.
                FROM_RANDOM generates a random population.
                FROM_BEST_AVAILABLE uses cached configs from prior runs, and fills the
                remainder with random configs when best_available_pad_random is True.
                Can be overridden by HELION_AUTOTUNER_INITIAL_POPULATION env var (handled in default_autotuner_fn).
                If None is passed, defaults to FROM_RANDOM.
            best_available_pad_random: When True and using FROM_BEST_AVAILABLE, pad the
                cached configs with random configs to reach 2x population size.
                When False, use only the default and cached configs (no random padding).
            finishing_rounds: Number of finishing rounds to run after the main search.
            compile_timeout_lower_bound: Lower bound for adaptive compile timeout in seconds.
            compile_timeout_quantile: Quantile of compile times to use for adaptive timeout.
        """
        super().__init__(kernel, args, finishing_rounds=finishing_rounds)
        if immediate_update is None:
            immediate_update = not bool(kernel.settings.autotune_precompile)
        if initial_population_strategy is None:
            initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        self.initial_population_strategy = initial_population_strategy
        self.best_available_pad_random = best_available_pad_random
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.immediate_update = immediate_update
        self.min_improvement_delta = min_improvement_delta
        self.patience = patience
        self.compile_timeout_lower_bound = compile_timeout_lower_bound
        self.compile_timeout_quantile = compile_timeout_quantile

        # Early stopping state
        self.best_perf_history: list[float] = []
        self.generations_without_improvement = 0

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        from ..runtime.settings import _get_initial_population_strategy

        assert profile.differential_evolution is not None
        strategy = _get_initial_population_strategy(
            profile.differential_evolution.initial_population_strategy,
            settings.autotune_initial_population_strategy,
        )
        return {
            "population_size": profile.differential_evolution.population_size,
            "max_generations": profile.differential_evolution.max_generations,
            "initial_population_strategy": strategy,
            "best_available_pad_random": profile.differential_evolution.best_available_pad_random,
            **super().get_kwargs_from_profile(profile, settings),
        }

    def mutate(self, x_index: int) -> FlatConfig:
        with sync_seed(process_group_name=self.kernel.env.process_group_name):
            a, b, c, *_ = [
                self.population[p]
                for p in random.sample(range(len(self.population)), 4)
                if p != x_index
            ]
            return self.config_gen.differential_mutation(
                self.population[x_index].flat_values,
                a.flat_values,
                b.flat_values,
                c.flat_values,
                self.crossover_rate,
            )

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
                target = self.population_size * 2
                n_random = max(0, target - len(pop))
                pop.extend(self.config_gen.random_flat() for _ in range(n_random))
                return pop[:target]
            return pop

        return self.config_gen.random_population_flat(self.population_size * 2)

    def initial_two_generations(self) -> None:
        # The initial population is 2x larger so we can throw out the slowest half and give the tuning process a head start
        initial_population_name = self.initial_population_strategy.name
        oversized_population = sorted(
            self.parallel_benchmark_flat(
                self._generate_initial_population_flat(),
            ),
            key=performance,
        )
        self.log(
            f"Initial population (initial_population={initial_population_name}):",
            lambda: population_statistics(oversized_population),
        )
        self.population = oversized_population[: self.population_size]

    def _benchmark_mutation_batch(
        self, indices: Sequence[int]
    ) -> list[PopulationMember]:
        if not indices:
            return []
        flat_configs = [self.mutate(i) for i in indices]
        return self.parallel_benchmark_flat(flat_configs)

    def iter_candidates(self) -> Iterator[tuple[int, PopulationMember]]:
        if self.immediate_update:
            for i in range(len(self.population)):
                candidates = self._benchmark_mutation_batch([i])
                if not candidates:
                    continue
                yield i, candidates[0]
        else:
            indices = list(range(len(self.population)))
            candidates = self._benchmark_mutation_batch(indices)
            yield from zip(indices, candidates, strict=True)

    def evolve_population(self) -> int:
        replaced = 0
        for i, candidate in self.iter_candidates():
            if self.compare(candidate, self.population[i]) < 0:
                self.population[i] = candidate
                replaced += 1
        return replaced

    def check_early_stopping(self) -> bool:
        """
        Check if early stopping criteria are met and update state.

        This method updates best_perf_history and generations_without_improvement,
        and returns whether the optimization should stop.

        Returns:
            True if optimization should stop early, False otherwise.
        """
        # Update history
        current_best = self.best.perf
        self.best_perf_history.append(current_best)

        if self.patience is None or len(self.best_perf_history) <= self.patience:
            return False

        # Check improvement over last patience generations
        past_best = self.best_perf_history[-self.patience - 1]

        if not (
            math.isfinite(current_best)
            and math.isfinite(past_best)
            and past_best != 0.0
        ):
            return False

        relative_improvement = abs(current_best / past_best - 1.0)

        if (
            self.min_improvement_delta is not None
            and relative_improvement < self.min_improvement_delta
        ):
            # No significant improvement
            self.generations_without_improvement += 1
            if self.generations_without_improvement >= self.patience:
                self.log(
                    f"Early stopping at generation {self._autotune_metrics.num_generations}: "
                    f"no improvement >{self.min_improvement_delta:.1%} for {self.patience} generations"
                )
                return True
            return False

        # Significant improvement - reset counter
        self.generations_without_improvement = 0
        return False

    def _autotune(self) -> Config:
        early_stopping_enabled = (
            self.min_improvement_delta is not None and self.patience is not None
        )
        initial_population_name = self.initial_population_strategy.name

        self.log(
            lambda: (
                f"Starting DifferentialEvolutionSearch with population={self.population_size}, "
                f"generations={self.max_generations}, crossover_rate={self.crossover_rate}, "
                f"initial_population={initial_population_name}, "
                f"early_stopping=(delta={self.min_improvement_delta}, patience={self.patience})"
            )
        )

        self.initial_two_generations()

        # Compute adaptive compile timeout based on initial population compile times
        self.set_adaptive_compile_timeout(
            self.population,
            min_seconds=self.compile_timeout_lower_bound,
            quantile=self.compile_timeout_quantile,
        )

        # Initialize early stopping tracking
        if early_stopping_enabled:
            self.best_perf_history = [self.best.perf]
            self.generations_without_improvement = 0

        for i in range(2, self.max_generations):
            self.set_generation(i)
            self.log(f"Generation {i} starting")
            replaced = self.evolve_population()
            self.log(f"Generation {i} complete: replaced={replaced}", self.statistics)

            # Check for convergence (only if early stopping enabled)
            if early_stopping_enabled and self.check_early_stopping():
                break

        self.rebenchmark_population()

        # Run finishing phase to simplify the best configuration
        self.best = self.run_finishing_phase(self.best, self.finishing_rounds)
        return self.best.config
