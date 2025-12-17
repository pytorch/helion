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
import operator
import random
from typing import TYPE_CHECKING
from typing import Any

from .differential_evolution import DifferentialEvolutionSearch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.kernel import BoundKernel
    from .config_generation import Config
    from .config_generation import FlatConfig
    from .pattern_search import InitialPopulationStrategy

try:
    import numpy as np  # type: ignore[import-not-found]
    from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-not-found]

    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    np = None  # type: ignore[assignment]
    RandomForestRegressor = None  # type: ignore[assignment,misc]


class DESurrogateHybrid(DifferentialEvolutionSearch):
    """
    Hybrid Differential Evolution with Surrogate-Assisted Selection.

    This algorithm uses DE for exploration but adds a surrogate model to intelligently
    select which candidates to actually evaluate, avoiding wasting evaluations on
    poor candidates.

    Args:
        kernel: The bound kernel to tune
        args: Arguments for the kernel
        population_size: Size of the DE population (default: 40)
        max_generations: Maximum number of generations (default: 40)
        crossover_rate: Crossover probability (default: 0.8)
        surrogate_threshold: Use surrogate after this many evaluations (default: 100)
        candidate_ratio: Generate this many× candidates per slot (default: 3)
        refit_frequency: Refit surrogate every N generations (default: 5)
        n_estimators: Number of trees in Random Forest (default: 50)
        min_improvement_delta: Relative improvement threshold for early stopping.
            Default: 0.001 (0.1%). Early stopping enabled by default.
        patience: Number of generations without improvement before stopping.
            Default: 3. Early stopping enabled by default.
        initial_population_strategy: Strategy for generating the initial population.
            FROM_RANDOM generates a random population.
            FROM_DEFAULT starts from the default configuration.
            Can be overridden by HELION_AUTOTUNER_INITIAL_POPULATION env var.
            If not set via env var and None is passed, defaults to FROM_RANDOM.
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
        initial_population_strategy: InitialPopulationStrategy | None = None,
    ) -> None:
        if not HAS_ML_DEPS:
            raise ImportError(
                "DESurrogateHybrid requires numpy and scikit-learn. "
                "Install them with: pip install helion[surrogate]"
            )

        # Initialize parent with early stopping and initial population strategy parameters
        super().__init__(
            kernel,
            args,
            population_size=population_size,
            max_generations=max_generations,
            crossover_rate=crossover_rate,
            min_improvement_delta=min_improvement_delta,
            patience=patience,
            initial_population_strategy=initial_population_strategy,
        )

        self.surrogate_threshold = surrogate_threshold
        self.candidate_ratio = candidate_ratio
        self.refit_frequency = refit_frequency
        self.n_estimators = n_estimators

        # Surrogate model
        self.surrogate: Any = None

        # Track all evaluations for surrogate training
        self.all_observations: list[tuple[FlatConfig, float]] = []

    def _autotune(self) -> Config:
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
        self.log(
            f"Early stopping: delta={self.min_improvement_delta}, patience={self.patience}"
        )
        self.log("=" * 70)

        # Initialize population
        self.set_generation(0)
        self.initial_two_generations()

        # Track initial observations for surrogate
        for member in self.population:
            if math.isfinite(member.perf):
                self.all_observations.append((member.flat_values, member.perf))

        # Initialize early stopping tracking
        self.best_perf_history = [self.best.perf]
        self.generations_without_improvement = 0

        # Evolution loop
        for gen in range(2, self.max_generations + 1):
            self.set_generation(gen)
            self._evolve_generation(gen)

            # Check for convergence
            if self.check_early_stopping():
                break

        # Return best config
        best = min(self.population, key=lambda m: m.perf)
        self.log("=" * 70)
        self.log(f"✓ Best configuration: {best.perf:.4f} ms")
        self.log(f"Total evaluations: {len(self.all_observations)}")
        self.log("=" * 70)

        return best.config

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
            selected_candidates = self._surrogate_select(
                candidates, self.population_size
            )
        else:
            # Standard DE: generate and evaluate all
            selected_candidates = self._generate_de_candidates(self.population_size)

        # Evaluate selected candidates
        new_members = self.parallel_benchmark_flat(selected_candidates)

        # Track observations
        for member in new_members:
            if math.isfinite(member.perf):
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
                x.flat_values,
                a.flat_values,
                b.flat_values,
                c.flat_values,
                crossover_rate=self.crossover_rate,
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
                encoded = self.config_gen.encode_config(config)
                X.append(encoded)
                y.append(perf)
            except Exception:
                continue

        if len(X) < 10:
            return

        X_array = np.array(X)  # type: ignore[union-attr]
        y_array = np.array(y)  # type: ignore[union-attr]

        # Fit Random Forest
        surrogate = RandomForestRegressor(  # type: ignore[misc]
            n_estimators=self.n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        surrogate.fit(X_array, y_array)
        self.surrogate = surrogate

    def _surrogate_select(
        self, candidates: list[FlatConfig], n_select: int
    ) -> list[FlatConfig]:
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
                encoded = self.config_gen.encode_config(config)
                pred = self.surrogate.predict([encoded])[0]
                predictions.append((config, pred))
            except Exception:
                # Skip encoding failures
                predictions.append((config, float("inf")))

        # Sort by predicted performance (lower is better)
        predictions.sort(key=operator.itemgetter(1))

        # Select top n_select candidates
        return [config for config, pred in predictions[:n_select]]

    def __repr__(self) -> str:
        return (
            f"DESurrogateHybrid(pop={self.population_size}, "
            f"gen={self.max_generations}, "
            f"cr={self.crossover_rate}, "
            f"surrogate_threshold={self.surrogate_threshold})"
        )
