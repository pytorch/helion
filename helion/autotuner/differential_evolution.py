from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .base_search import FlatConfig
from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .base_search import performance
from .base_search import population_statistics

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
        # pyre-fixme[11]: BoundKernel undefined?
        kernel: BoundKernel,
        args: Sequence[object],
        population_size: int = 40,
        num_generations: int = 20,
        crossover_rate: float = 0.8,
        immediate_update: bool | None = None,
    ) -> None:
        super().__init__(kernel, args)
        if immediate_update is None:
            immediate_update = not kernel.settings.autotune_precompile
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.immediate_update = immediate_update

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

    def iter_candidates(self) -> Iterator[tuple[int, PopulationMember]]:
        if self.immediate_update:
            for i in range(len(self.population)):
                yield i, self.benchmark_flat(self.mutate(i))
        else:
            yield from enumerate(
                self.parallel_benchmark_flat(
                    [self.mutate(i) for i in range(len(self.population))]
                )
            )

    def evolve_population(self) -> int:
        replaced = 0
        for i, candidate in self.iter_candidates():
            candidate = self.benchmark_flat(self.mutate(i))
            if candidate.perf < self.population[i].perf:
                self.population[i] = candidate
                replaced += 1
        return replaced

    def _autotune(self) -> Config:
        self.log(
            lambda: (
                f"Starting DifferentialEvolutionSearch with population={self.population_size}, "
                f"generations={self.num_generations}, crossover_rate={self.crossover_rate}"
            )
        )
        self.initial_two_generations()
        for i in range(2, self.num_generations):
            replaced = self.evolve_population()
            self.log(f"Generation {i}: replaced={replaced}", self.statistics)
        return self.best.config
