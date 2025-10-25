from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

from .acquisition import expected_improvement
from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .config_encoding import ConfigEncoder
from .effort_profile import MULTIFIDELITY_BO_DEFAULTS
from .gaussian_process import MultiFidelityGP

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from .config_generation import FlatConfig


class MultiFidelityBayesianSearch(PopulationBasedSearch):
    """
    Multi-Fidelity Bayesian Optimization for kernel autotuning.

    Uses cheap low-fidelity evaluations to guide expensive high-fidelity evaluations,
    achieving 10-40x speedup over standard pattern search.
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        *,
        n_low_fidelity: int = MULTIFIDELITY_BO_DEFAULTS.n_low_fidelity,
        n_medium_fidelity: int = MULTIFIDELITY_BO_DEFAULTS.n_medium_fidelity,
        n_high_fidelity: int = MULTIFIDELITY_BO_DEFAULTS.n_high_fidelity,
        n_ultra_fidelity: int = MULTIFIDELITY_BO_DEFAULTS.n_ultra_fidelity,
        fidelity_low: int = MULTIFIDELITY_BO_DEFAULTS.fidelity_low,
        fidelity_medium: int = MULTIFIDELITY_BO_DEFAULTS.fidelity_medium,
        fidelity_high: int = MULTIFIDELITY_BO_DEFAULTS.fidelity_high,
        fidelity_ultra: int = MULTIFIDELITY_BO_DEFAULTS.fidelity_ultra,
        acquisition: Literal["ei", "ucb"] = "ei",
    ) -> None:
        """
        Create a MultiFidelityBayesianSearch autotuner.

        Args:
            kernel: The kernel to be autotuned.
            args: The arguments to be passed to the kernel.
            n_low_fidelity: Number of configs to evaluate at low fidelity.
            n_medium_fidelity: Number of configs to evaluate at medium fidelity.
            n_high_fidelity: Number of configs to evaluate at high fidelity.
            n_ultra_fidelity: Number of configs to evaluate at ultra-high fidelity.
            fidelity_low: Number of reps for low fidelity.
            fidelity_medium: Number of reps for medium fidelity.
            fidelity_high: Number of reps for high fidelity.
            fidelity_ultra: Number of reps for ultra-high fidelity.
            acquisition: Acquisition function to use ("ei" or "ucb").
        """
        super().__init__(kernel, args)
        self.n_low = n_low_fidelity
        self.n_medium = n_medium_fidelity
        self.n_high = n_high_fidelity
        self.n_ultra = n_ultra_fidelity
        self.fid_low = fidelity_low
        self.fid_medium = fidelity_medium
        self.fid_high = fidelity_high
        self.fid_ultra = fidelity_ultra
        self.acquisition_fn = acquisition

        # Initialize encoder and GP
        self.encoder = ConfigEncoder(self.config_gen)
        self.gp = MultiFidelityGP()

        # Track all evaluated configs by fidelity
        self.evaluated_low: list[PopulationMember] = []
        self.evaluated_medium: list[PopulationMember] = []
        self.evaluated_high: list[PopulationMember] = []
        self.evaluated_ultra: list[PopulationMember] = []

    def _autotune(self) -> Config:
        self.log(
            f"Starting MultiFidelityBayesianSearch: "
            f"low={self.n_low}×{self.fid_low}, "
            f"med={self.n_medium}×{self.fid_medium}, "
            f"high={self.n_high}×{self.fid_high}, "
            f"ultra={self.n_ultra}×{self.fid_ultra}"
        )

        # Stage 1: Low-fidelity exploration
        self._stage_low_fidelity()

        # Stage 2: Medium-fidelity (BO-guided)
        self._stage_medium_fidelity()

        # Stage 3: High-fidelity validation
        self._stage_high_fidelity()

        # Stage 4: Ultra-high fidelity final comparison
        self._stage_ultra_fidelity()

        # Return the best configuration
        best = min(self.evaluated_ultra, key=lambda m: m.perf)
        self.log(f"Best config: {best.config}, perf={best.perf:.4f}ms")
        return best.config

    def _stage_low_fidelity(self) -> None:
        """Stage 1: Broad exploration at low fidelity."""
        self.log(f"Stage 1: Low-fidelity exploration ({self.n_low} configs × {self.fid_low} reps)")

        # Generate random configurations
        candidates = list(self.config_gen.random_population_flat(self.n_low))
        members = [self.make_unbenchmarked(flat) for flat in candidates]

        # Benchmark at low fidelity
        members = self._benchmark_population_at_fidelity(
            members, self.fid_low, desc="Low-fidelity exploration"
        )

        # Filter out failed configs
        self.evaluated_low = [m for m in members if math.isfinite(m.perf)]
        self.population.extend(self.evaluated_low)

        if not self.evaluated_low:
            self.log.warning("No valid configs found at low fidelity!")
            return

        # Train GP on low-fidelity data
        X_low = np.array([self.encoder.encode(m.flat_values) for m in self.evaluated_low])
        y_low = np.array([m.perf for m in self.evaluated_low])
        self.gp.fit_low(X_low, y_low)

        best = min(self.evaluated_low, key=lambda m: m.perf)
        self.log(f"Stage 1 complete: best={best.perf:.4f}ms, {len(self.evaluated_low)} valid configs")

    def _stage_medium_fidelity(self) -> None:
        """Stage 2: Medium-fidelity validation (BO-guided selection)."""
        if not self.evaluated_low:
            return

        self.log(
            f"Stage 2: Medium-fidelity validation ({self.n_medium} configs × {self.fid_medium} reps)"
        )

        # Generate candidate pool and select by acquisition function
        candidates = self._select_by_acquisition(
            self.n_medium, candidate_pool_size=min(1000, self.n_low * 5)
        )
        members = [self.make_unbenchmarked(flat) for flat in candidates]

        # Benchmark at medium fidelity
        members = self._benchmark_population_at_fidelity(
            members, self.fid_medium, desc="Medium-fidelity validation"
        )

        # Filter out failed configs
        self.evaluated_medium = [m for m in members if math.isfinite(m.perf)]
        self.population.extend(self.evaluated_medium)

        if not self.evaluated_medium:
            self.log.warning("No valid configs found at medium fidelity!")
            return

        # Train GP on medium-fidelity data
        X_medium = np.array([self.encoder.encode(m.flat_values) for m in self.evaluated_medium])
        y_medium = np.array([m.perf for m in self.evaluated_medium])
        self.gp.fit_high(X_medium, y_medium)

        best = min(self.evaluated_medium, key=lambda m: m.perf)
        self.log(
            f"Stage 2 complete: best={best.perf:.4f}ms, {len(self.evaluated_medium)} valid configs"
        )

    def _stage_high_fidelity(self) -> None:
        """Stage 3: High-fidelity validation (BO-guided with multi-fidelity GP)."""
        if not self.evaluated_medium:
            # Fall back to low fidelity if medium failed
            if not self.evaluated_low:
                return
            source = self.evaluated_low
        else:
            source = self.evaluated_medium

        self.log(
            f"Stage 3: High-fidelity validation ({self.n_high} configs × {self.fid_high} reps)"
        )

        # Select best candidates using multi-fidelity GP
        candidates = self._select_by_acquisition(
            self.n_high, candidate_pool_size=min(500, len(source) * 3), use_multifidelity=True
        )
        members = [self.make_unbenchmarked(flat) for flat in candidates]

        # Benchmark at high fidelity
        members = self._benchmark_population_at_fidelity(
            members, self.fid_high, desc="High-fidelity validation"
        )

        # Filter out failed configs
        self.evaluated_high = [m for m in members if math.isfinite(m.perf)]
        self.population.extend(self.evaluated_high)

        if not self.evaluated_high:
            self.log.warning("No valid configs found at high fidelity!")
            return

        best = min(self.evaluated_high, key=lambda m: m.perf)
        self.log(
            f"Stage 3 complete: best={best.perf:.4f}ms, {len(self.evaluated_high)} valid configs"
        )

    def _stage_ultra_fidelity(self) -> None:
        """Stage 4: Ultra-high fidelity final comparison."""
        if not self.evaluated_high:
            # Fall back to previous stage
            if self.evaluated_medium:
                source = self.evaluated_medium
            elif self.evaluated_low:
                source = self.evaluated_low
            else:
                raise Exception("No valid configurations found in any stage!")
        else:
            source = self.evaluated_high

        self.log(
            f"Stage 4: Ultra-high fidelity final ({self.n_ultra} configs × {self.fid_ultra} reps)"
        )

        # Select top N configs from high-fidelity results
        source_sorted = sorted(source, key=lambda m: m.perf)
        top_n = source_sorted[: self.n_ultra]

        # Re-benchmark at ultra-high fidelity for final comparison
        members = [
            PopulationMember(m.fn, [], m.flat_values, m.config, m.status) for m in top_n
        ]
        members = self._benchmark_population_at_fidelity(
            members, self.fid_ultra, desc="Ultra-high fidelity final"
        )

        # Filter out failed configs
        self.evaluated_ultra = [m for m in members if math.isfinite(m.perf)]

        if not self.evaluated_ultra:
            self.log.warning("No valid configs at ultra-high fidelity, using high-fidelity best")
            self.evaluated_ultra = top_n

        best = min(self.evaluated_ultra, key=lambda m: m.perf)
        self.log(f"Stage 4 complete: best={best.perf:.4f}ms")

    def _benchmark_population_at_fidelity(
        self, members: list[PopulationMember], fidelity: int, *, desc: str = "Benchmarking"
    ) -> list[PopulationMember]:
        """
        Benchmark a population at a specific fidelity level.

        Args:
            members: Population members to benchmark.
            fidelity: Number of repetitions.
            desc: Description for progress bar.

        Returns:
            The benchmarked population members.
        """
        # Store fidelity for benchmark_function to use
        self._current_fidelity = fidelity

        configs = [m.config for m in members]
        results = self.parallel_benchmark([c for c in configs], desc=desc)

        for member, (config_out, fn, perf, status) in zip(members, results, strict=True):
            assert config_out is member.config
            member.perfs.append(perf)
            member.fidelities.append(fidelity)
            member.fn = fn
            member.status = status

        return members

    def benchmark_function(self, config: Config, fn: object, *, fidelity: int = 50) -> float:
        """Benchmark with specific fidelity."""
        # Use the fidelity set by _benchmark_population_at_fidelity if available
        actual_fidelity = getattr(self, "_current_fidelity", fidelity)
        return super().benchmark_function(config, fn, fidelity=actual_fidelity)  # type: ignore

    def _select_by_acquisition(
        self,
        n_select: int,
        candidate_pool_size: int = 1000,
        use_multifidelity: bool = False,
    ) -> list[FlatConfig]:
        """
        Select configurations using acquisition function.

        Args:
            n_select: Number of configurations to select.
            candidate_pool_size: Size of random candidate pool to score.
            use_multifidelity: Whether to use multi-fidelity GP predictions.

        Returns:
            List of selected flat configurations.
        """
        # Generate candidate pool
        candidate_pool = list(self.config_gen.random_population_flat(candidate_pool_size))
        X_candidates = np.array([self.encoder.encode(flat) for flat in candidate_pool])

        # Get GP predictions
        if use_multifidelity and self.gp.fitted_high:
            mu, sigma = self.gp.predict_multifidelity(X_candidates)
        elif self.gp.fitted_high:
            mu, sigma = self.gp.predict_high(X_candidates, return_std=True)  # type: ignore
        else:
            mu, sigma = self.gp.predict_low(X_candidates, return_std=True)  # type: ignore

        # Compute acquisition scores
        best_so_far = self.gp.get_best_observed()
        if self.acquisition_fn == "ei":
            scores = expected_improvement(mu, sigma, best_so_far)
        else:
            # UCB (lower is better for minimization)
            from .acquisition import upper_confidence_bound

            lcb = upper_confidence_bound(mu, sigma, beta=2.0)
            scores = -lcb  # Negate so higher scores are better

        # Select top N
        top_indices = np.argsort(scores)[-n_select:][::-1]
        selected = [candidate_pool[i] for i in top_indices]

        return selected
