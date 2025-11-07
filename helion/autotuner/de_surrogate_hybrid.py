"""
Differential Evolution with Surrogate-Assisted Selection (DE-SAS).

This hybrid approach combines the robust exploration of Differential Evolution
with the sample efficiency of surrogate models. It's designed to beat standard DE
by making smarter decisions about which candidates to evaluate.

Key idea:
- Use DE's mutation/crossover to generate candidates (good exploration)
- Use a Random Forest surrogate to predict which candidates are promising
- Only evaluate the most promising candidates (sample efficiency)
- Periodically re-fit the surrogate model

This is inspired by recent work on surrogate-assisted evolutionary algorithms,
which have shown 2-5× speedups over standard EAs on expensive optimization problems.

References:
- Jin, Y. (2011). "Surrogate-assisted evolutionary computation: Recent advances and future challenges."
- Sun, C., et al. (2019). "A surrogate-assisted DE with an adaptive local search"

Author: Francisco Geiman Thiesen
Date: 2025-11-05
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base_search import PopulationBasedSearch, PopulationMember
from .config_encoding import ConfigEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.kernel import BoundKernel
    from .config_generation import FlatConfig


class DESurrogateHybrid(PopulationBasedSearch):
    """
    Hybrid Differential Evolution with Surrogate-Assisted Selection.

    This algorithm uses DE for exploration but adds a surrogate model to intelligently
    select which candidates to actually evaluate, avoiding wasting evaluations on
    poor candidates.

    Args:
        kernel: The bound kernel to tune
        args: Arguments for the kernel
        population_size: Size of the DE population
        max_generations: Maximum number of generations
        crossover_rate: Crossover probability (default: 0.8)
        surrogate_threshold: Use surrogate after this many evaluations (default: 100)
        candidate_ratio: Generate this many× candidates per slot (default: 3)
        refit_frequency: Refit surrogate every N generations (default: 5)
        n_estimators: Number of trees in Random Forest (default: 50)
        min_improvement_delta: Relative stop threshold (default: 0.001 = 0.1%)
        patience: Stop if no improvement for this many generations (default: 3)
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        population_size: int = 40,
        max_generations: int = 40,
        crossover_rate: float = 0.8,
        surrogate_threshold: int = 100,
        candidate_ratio: int = 3,
        refit_frequency: int = 5,
        n_estimators: int = 50,
        min_improvement_delta: float = 0.001,
        patience: int = 3,
    ) -> None:
        super().__init__(kernel, args)

        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.surrogate_threshold = surrogate_threshold
        self.candidate_ratio = candidate_ratio
        self.refit_frequency = refit_frequency
        self.n_estimators = n_estimators
        self.min_improvement_delta = min_improvement_delta
        self.patience = patience

        # Config encoder for surrogate model
        self.encoder = ConfigEncoder(self.config_gen)

        # Surrogate model
        self.surrogate: RandomForestRegressor | None = None

        # Track all evaluations for surrogate training
        self.all_observations: list[tuple[FlatConfig, float]] = []

    def _autotune(self):
        """
        Run DE with surrogate-assisted selection.

        Returns:
            Best configuration found
        """
        self.log("=" * 70)
        self.log("Differential Evolution with Surrogate-Assisted Selection")
        self.log("=" * 70)
        self.log(f"Population: {self.population_size}")
        self.log(f"Generations: {self.max_generations}")
        self.log(f"Crossover rate: {self.crossover_rate}")
        self.log(f"Surrogate activation: after {self.surrogate_threshold} evals")
        self.log(f"Candidate oversampling: {self.candidate_ratio}× per slot")
        self.log(f"Early stopping: delta={self.min_improvement_delta}, patience={self.patience}")
        self.log("=" * 70)

        # Initialize population
        self._initialize_population()

        # Early stopping tracking
        best_perf_history = [min(m.perf for m in self.population)]
        generations_without_improvement = 0

        # Evolution loop
        for gen in range(2, self.max_generations + 1):
            self._evolve_generation(gen)

            # Track best performance
            current_best = min(m.perf for m in self.population)
            best_perf_history.append(current_best)

            # Check for convergence
            if len(best_perf_history) > self.patience:
                past_best = best_perf_history[-self.patience - 1]

                if math.isfinite(current_best) and math.isfinite(past_best) and past_best != 0.0:
                    relative_improvement = abs(current_best / past_best - 1.0)

                    if relative_improvement < self.min_improvement_delta:
                        generations_without_improvement += 1
                        if generations_without_improvement >= self.patience:
                            self.log(
                                f"Early stopping at generation {gen}: "
                                f"no improvement >{self.min_improvement_delta:.1%} for {self.patience} generations"
                            )
                            break
                    else:
                        generations_without_improvement = 0

        # Return best config
        best = min(self.population, key=lambda m: m.perf)
        self.log("=" * 70)
        self.log(f"✓ Best configuration: {best.perf:.4f} ms")
        self.log(f"Total evaluations: {len(self.all_observations)}")
        self.log("=" * 70)

        return best.config

    def _initialize_population(self) -> None:
        """Initialize population with random configs."""
        self.log(f"\nInitializing population ({self.population_size*2} configs)")

        # Generate initial population (2× size for good coverage)
        configs = [self.config_gen.random_flat() for _ in range(self.population_size * 2)]
        members = self.parallel_benchmark_flat(configs)

        # Track observations
        for member in members:
            if member.perf != float("inf"):
                self.all_observations.append((member.flat_values, member.perf))

        # Keep top population_size members
        valid_members = [m for m in members if m.perf != float("inf")]
        valid_members.sort(key=lambda m: m.perf)
        self.population = valid_members[: self.population_size]

        # Pad with random if needed
        while len(self.population) < self.population_size:
            config = self.config_gen.random_flat()
            member = self.benchmark_flat(config)
            if member.perf != float("inf"):
                self.population.append(member)
                self.all_observations.append((member.flat_values, member.perf))

        best_perf = min(m.perf for m in self.population)
        self.log(
            f"Population initialized: "
            f"best={best_perf:.4f} ms, size={len(self.population)}"
        )

    def _evolve_generation(self, generation: int) -> None:
        """Run one generation of DE with surrogate assistance."""

        # Refit surrogate periodically
        use_surrogate = len(self.all_observations) >= self.surrogate_threshold
        if use_surrogate and (generation % self.refit_frequency == 0):
            self._fit_surrogate()

        # Generate candidates using DE mutation/crossover
        if use_surrogate:
            # Generate more candidates and use surrogate to select best
            n_candidates = self.population_size * self.candidate_ratio
            candidates = self._generate_de_candidates(n_candidates)
            selected_candidates = self._surrogate_select(candidates, self.population_size)
        else:
            # Standard DE: generate and evaluate all
            selected_candidates = self._generate_de_candidates(self.population_size)

        # Evaluate selected candidates
        new_members = self.parallel_benchmark_flat(selected_candidates)

        # Track observations
        for member in new_members:
            if member.perf != float("inf"):
                self.all_observations.append((member.flat_values, member.perf))

        # Selection: keep better of old vs new for each position
        replacements = 0
        for i, new_member in enumerate(new_members):
            if new_member.perf < self.population[i].perf:
                self.population[i] = new_member
                replacements += 1

        # Log progress
        best_perf = min(m.perf for m in self.population)
        surrogate_status = "SURROGATE" if use_surrogate else "STANDARD"
        self.log(
            f"Gen {generation}: {surrogate_status} | "
            f"best={best_perf:.4f} ms | replaced={replacements}/{self.population_size} | "
            f"total_evals={len(self.all_observations)}"
        )

    def _generate_de_candidates(self, n_candidates: int) -> list[FlatConfig]:
        """Generate candidates using standard DE mutation/crossover."""
        candidates = []

        for _ in range(n_candidates):
            # Select four distinct individuals: x (base), and a, b, c for mutation
            x, a, b, c = random.sample(self.population, 4)

            # Differential mutation: x + F(a - b + c)
            trial = self.config_gen.differential_mutation(
                x.flat_values, a.flat_values, b.flat_values, c.flat_values, crossover_rate=self.crossover_rate
            )

            candidates.append(trial)

        return candidates

    def _fit_surrogate(self) -> None:
        """Fit Random Forest surrogate model on all observations."""
        if len(self.all_observations) < 10:
            return  # Need minimum data

        # Encode configs to numeric arrays
        X = []
        y = []

        for config, perf in self.all_observations:
            try:
                encoded = self.encoder.encode(config)
                X.append(encoded)
                y.append(perf)
            except Exception:
                continue

        if len(X) < 10:
            return

        X_array = np.array(X)
        y_array = np.array(y)

        # Fit Random Forest
        self.surrogate = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        self.surrogate.fit(X_array, y_array)

    def _surrogate_select(self, candidates: list[FlatConfig], n_select: int) -> list[FlatConfig]:
        """
        Use surrogate model to select most promising candidates.

        Args:
            candidates: Pool of candidate configurations
            n_select: Number of candidates to select

        Returns:
            Selected candidates predicted to be best
        """
        if self.surrogate is None:
            # Fallback: random selection
            return random.sample(candidates, min(n_select, len(candidates)))

        # Predict performance for all candidates
        predictions = []

        for config in candidates:
            try:
                encoded = self.encoder.encode(config)
                pred = self.surrogate.predict([encoded])[0]
                predictions.append((config, pred))
            except Exception:
                # Skip encoding failures
                predictions.append((config, float("inf")))

        # Sort by predicted performance (lower is better)
        predictions.sort(key=lambda x: x[1])

        # Select top n_select candidates
        selected = [config for config, pred in predictions[:n_select]]

        return selected

    def __repr__(self) -> str:
        return (
            f"DESurrogateHybrid(pop={self.population_size}, "
            f"gen={self.max_generations}, "
            f"cr={self.crossover_rate}, "
            f"surrogate_threshold={self.surrogate_threshold})"
        )
