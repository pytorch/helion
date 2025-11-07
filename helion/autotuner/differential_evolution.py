from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from .base_search import FlatConfig
from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .base_search import performance
from .base_search import population_statistics
from .effort_profile import DIFFERENTIAL_EVOLUTION_DEFAULTS

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel


class DifferentialEvolutionSearch(PopulationBasedSearch):
    """
    A search strategy that uses differential evolution to find the best config.
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        population_size: int = DIFFERENTIAL_EVOLUTION_DEFAULTS.population_size,
        max_generations: int = DIFFERENTIAL_EVOLUTION_DEFAULTS.max_generations,
        crossover_rate: float = 0.8,
        immediate_update: bool | None = None,
        min_improvement_delta: float = 0.001,
        patience: int = 3,
    ) -> None:
        super().__init__(kernel, args)
        if immediate_update is None:
            immediate_update = not bool(kernel.settings.autotune_precompile)
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.immediate_update = immediate_update
        self.min_improvement_delta = min_improvement_delta
        self.patience = patience

    def mutate(self, x_index: int) -> FlatConfig:
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

    def initial_two_generations(self) -> None:
        # The initial population is 2x larger so we can throw out the slowest half and give the tuning process a head start
        self.set_generation(0)
        oversized_population = sorted(
            self.parallel_benchmark_flat(
                self.config_gen.random_population_flat(self.population_size * 2),
            ),
            key=performance,
        )
        self.log(
            "Initial population:",
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

    def _autotune(self) -> Config:
        self.log(
            lambda: (
                f"Starting DifferentialEvolutionSearch with population={self.population_size}, "
                f"generations={self.max_generations}, crossover_rate={self.crossover_rate}, "
                f"early_stopping=(delta={self.min_improvement_delta}, patience={self.patience})"
            )
        )
        self.initial_two_generations()

        # Early stopping tracking
        best_perf_history = [self.best.perf]
        generations_without_improvement = 0

        for i in range(2, self.max_generations):
            self.set_generation(i)
            self.log(f"Generation {i} starting")
            replaced = self.evolve_population()
            self.log(f"Generation {i} complete: replaced={replaced}", self.statistics)

            # Check for convergence
            current_best = self.best.perf
            best_perf_history.append(current_best)

            if len(best_perf_history) > self.patience:
                # Check improvement over last patience generations
                past_best = best_perf_history[-self.patience - 1]

                if math.isfinite(current_best) and math.isfinite(past_best) and past_best != 0.0:
                    relative_improvement = abs(current_best / past_best - 1.0)

                    if relative_improvement < self.min_improvement_delta:
                        generations_without_improvement += 1
                        if generations_without_improvement >= self.patience:
                            self.log(
                                f"Early stopping at generation {i}: "
                                f"no improvement >{self.min_improvement_delta:.1%} for {self.patience} generations"
                            )
                            break
                    else:
                        generations_without_improvement = 0

        self.rebenchmark_population()
        return self.best.config
