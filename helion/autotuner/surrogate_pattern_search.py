from __future__ import annotations

import math
import operator
import random
from typing import TYPE_CHECKING

from .. import exc
from .base_search import FlatConfig
from .base_search import PopulationMember
from .base_search import performance
from .config_fragment import PowerOfTwoFragment
from .effort_profile import PATTERN_SEARCH_DEFAULTS
from .pattern_search import InitialPopulationStrategy
from .pattern_search import PatternSearch

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    HAS_ML_DEPS = True
except ImportError as e:
    HAS_ML_DEPS = False
    _IMPORT_ERROR = e


class LFBOPatternSearch(PatternSearch):
    """
    Likelihood-Free Bayesian Optimization (LFBO) Pattern Search.

    This algorithm enhances PatternSearch by using a Random Forest classifier as a surrogate
    model to select which configurations to benchmark, reducing the number of
    kernel compilations and runs needed to find optimal configurations.

    Algorithm Overview:
        1. Generate an initial population (random or default) and benchmark all configurations
        2. Fit a Random Forest classifier to predict "good" vs "bad" configurations:
           - Configs with performance < quantile threshold are labeled as "good" (class 1)
           - Configs with performance >= quantile threshold are labeled as "bad" (class 0)
           - Weighted classification emphasize configs that are much better than the threshold
        3. For each generation:
           - Generate random neighbors around the current best configurations
           - Score all neighbors using the classifier's predicted probability of being "good"
           - Benchmark only the top frac_selected fraction of neighbors
           - Retrain the classifier on all observed data (not incremental)
           - Update search trajectories based on new results

    The weighted classification model learns to identify which configs maximize
    expected improvement over the current best config. Compared to fitting a surrogate
    to fit the config performances themselves, since this method is based on classification,
    it can also learn from configs that timeout or have unacceptable accuracy.

    References:
    - Song, J., et al. (2022). "A General Recipe for Likelihood-free Bayesian Optimization."

    Args:
        kernel: The kernel to be autotuned.
        args: The arguments to be passed to the kernel during benchmarking.
        initial_population: Number of random configurations in initial population.
            Default from PATTERN_SEARCH_DEFAULTS. Ignored when using DEFAULT strategy.
        copies: Number of top configurations to run pattern search from.
            Default from PATTERN_SEARCH_DEFAULTS.
        max_generations: Maximum number of search iterations per copy.
            Default from PATTERN_SEARCH_DEFAULTS.
        min_improvement_delta: Early stopping threshold. Search stops if the relative
            improvement abs(best/current - 1) < min_improvement_delta.
            Default: 0.001 (0.1% improvement threshold).
        frac_selected: Fraction of generated neighbors to actually benchmark, after
            filtering by classifier score. Range: (0, 1]. Lower values reduce benchmarking
            cost but may miss good configurations. Default: 0.15.
        num_neighbors: Number of random neighbor configurations to generate around
            each search point per generation. Default: 300.
        radius: Maximum perturbation distance in configuration space. For power-of-two
            parameters, this is the max change in log2 space. For other parameters,
            this limits how many parameters can be changed. Default: 2.
        quantile: Threshold for labeling configs as "good" (class 1) vs "bad" (class 0).
            Configs with performance below this quantile are labeled as good.
            Range: (0, 1). Lower values create a more selective definition of "good".
            Default: 0.3 (top 30% are considered good).
        patience: Number of generations without improvement before stopping
            the search copy. Default: 2.
        initial_population_strategy: Strategy for generating the initial population.
            FROM_RANDOM generates initial_population random configs.
            FROM_DEFAULT starts from only the default configuration.
            Can be overridden by HELION_AUTOTUNER_INITIAL_POPULATION env var ("from_random" or "from_default").
    """

    def __init__(
        self,
        kernel: BoundKernel,
        args: Sequence[object],
        *,
        initial_population: int = PATTERN_SEARCH_DEFAULTS.initial_population,
        copies: int = PATTERN_SEARCH_DEFAULTS.copies,
        max_generations: int = PATTERN_SEARCH_DEFAULTS.max_generations,
        min_improvement_delta: float = 0.001,
        frac_selected: float = 0.10,
        num_neighbors: int = 300,
        radius: int = 2,
        quantile: float = 0.1,
        patience: int = 1,
        initial_population_strategy: InitialPopulationStrategy | None = None,
    ) -> None:
        if not HAS_ML_DEPS:
            raise exc.AutotuneError(
                "LFBOPatternSearch requires numpy and scikit-learn."
                "Install them with: pip install helion[surrogate]"
            ) from _IMPORT_ERROR

        super().__init__(
            kernel=kernel,
            args=args,
            initial_population=initial_population,
            copies=copies,
            max_generations=max_generations,
            min_improvement_delta=min_improvement_delta,
            initial_population_strategy=initial_population_strategy,
        )

        # Number of neighbors and how many to evalaute
        self.num_neighbors = num_neighbors
        self.radius = radius
        self.frac_selected = frac_selected
        self.patience = patience

        # Save training data
        self.train_x = []
        self.train_y = []
        self.quantile = quantile

    def _fit_surrogate(self) -> None:
        train_x = np.array(self.train_x)
        train_y = np.array(self.train_y)
        self.log(f"Fitting surrogate: {len(train_x)} points, {len(train_y)} targets")

        # Compute labels based on quantile threshold
        finite_mask = ~np.isinf(train_y)
        if finite_mask.any():
            # Compute quantile among finite performance values
            train_y_quantile = np.quantile(train_y[finite_mask], self.quantile)
            pos_mask = train_y <= train_y_quantile
            train_labels = 1.0 * (pos_mask)

            # Sample weights to emphasize configs that are much better than the threshold
            # Clip this difference to a small number (e.g. 1e-5) so that in the case that all perfs
            # are equal (and train_y_quantile - train_y = 0) we avoid dividing by zero.
            # Instead, we will have all sample weights = 1 for all positive points.
            pos_weights = np.maximum(1e-5, train_y_quantile - train_y) * train_labels
            normalizing_factor = np.mean(pos_weights[pos_mask])
            # Normalize weights so on average they are 1.0
            pos_weights = pos_weights / normalizing_factor
            # Weights for negative labels are 1.0
            sample_weight = np.where(pos_mask, pos_weights, 1.0)
        else:
            # If all targets are inf, then all labels are 0 (except the first one)
            train_labels = np.zeros(len(train_y))
            sample_weight = np.ones(len(train_y))

        # Ensure we have at least 2 classes for the classifier
        # If all labels are the same, we need to handle this case
        if np.all(train_labels == train_labels[0]):
            if len(train_labels) == 1:
                # With only one data point, we need to duplicate it with opposite label
                # to give the classifier two classes to learn from
                train_x = np.vstack([train_x, train_x[0]])
                train_labels = np.array([train_labels[0], 1.0 - train_labels[0]])
                sample_weight = np.array([sample_weight[0], sample_weight[0]])
                self.log(
                    "Only one training point, duplicating with opposite label for LFBO."
                )
            else:
                # Multiple points but all same label - flip the first one
                train_labels[0] = 1.0 - train_labels[0]
                self.log("All LFBO train labels are identical, flip the first bit.")

        self.surrogate = RandomForestClassifier(
            criterion="log_loss",
            random_state=42,
            n_estimators=100,
            n_jobs=-1,
        )
        self.surrogate.fit(train_x, train_labels, sample_weight=sample_weight)
        assert len(self.surrogate.classes_) == 2

    def _surrogate_select(
        self, candidates: list[PopulationMember], n_sorted: int
    ) -> list[PopulationMember]:
        # Score candidates
        candidate_X = np.array(
            [self.config_gen.encode_config(member.flat_values) for member in candidates]
        )
        scores = self.surrogate.predict_proba(candidate_X)
        scores = scores[:, 1]  # type: ignore[index]

        # sort candidates by score
        candidates_sorted = sorted(
            zip(candidates, scores, strict=True),
            key=operator.itemgetter(1),
            reverse=True,  # higher scores are better
        )[:n_sorted]

        self.log.debug(
            f"Scoring {len(candidate_X)} neighbors, selecting {(n_sorted / len(candidate_X)) * 100:.0f}% neighbors: {len(candidates_sorted)}"
        )

        return [member for member, score in candidates_sorted]

    def _autotune(self) -> Config:
        initial_population_name = self.initial_population_strategy.name
        self.log(
            f"Starting LFBOPatternSearch with initial_population={initial_population_name}, copies={self.copies}, max_generations={self.max_generations}"
        )
        visited: set[Config] = set()
        self.population = []
        for flat_config in self._generate_initial_population_flat():
            member = self.make_unbenchmarked(flat_config)
            if member.config not in visited:
                visited.add(member.config)
                self.population.append(member)
        self.parallel_benchmark_population(self.population, desc="Initial population")
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

        # Save to training data
        for member in self.population:
            self.train_x.append(self.config_gen.encode_config(member.flat_values))
            self.train_y.append(member.perf)

        # Fit model
        self._fit_surrogate()

        search_copies = [
            self._pruned_pattern_search_from(m, visited) for m in starting_points
        ]
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
                self.parallel_benchmark_population(
                    unbenchmarked, desc=f"Generation {generation}:"
                )
            # higher-accuracy rebenchmark
            self.rebenchmark_population(
                self.population, desc=f"Generation {generation}: verifying top configs"
            )
            # Log final statistics for this generation
            self.log(f"Generation {generation} complete:", self.statistics)

            # Update training data
            for member in self.population:
                self.train_x.append(self.config_gen.encode_config(member.flat_values))
                self.train_y.append(member.perf)

            # Fit model
            self._fit_surrogate()

        return self.best.config

    def _random_log2_neighbor(
        self, current_val: int, radius: int, low: int, high: int
    ) -> int:
        # Log the current value
        current_log = int(math.log2(current_val))
        # Random log perturbation
        delta = random.randint(-radius, radius)
        new_log = current_log + delta
        # Clamp to valid range
        min_log = int(math.log2(low))
        max_log = int(math.log2(high))
        new_log = max(min_log, min(new_log, max_log))
        return int(2**new_log)

    def _generate_neighbors(self, base: FlatConfig) -> list[FlatConfig]:
        """
        Generate neighboring configurations randomly within a specified radius.

        Strategy:
        1. Sample one block size index and change it by at most radius (in log2 space)
        2. Sample the num_warps index and change it by at most radius (in log2 space)
        3. For at most radius remaining indices, randomly select pattern neighbors

        Args:
            base: The base configuration to generate neighbors from

        Returns:
            A list of neighboring configurations
        """
        neighbors: list[FlatConfig] = []

        # Generate num_neighbors random neighbors
        for _ in range(self.num_neighbors):
            new_flat = [*base]  # Copy the base configuration
            modified_indices = set()

            # 1. Sample a block size index and change it by at most radius
            if self.config_gen.block_size_indices:
                block_idx = random.choice(self.config_gen.block_size_indices)
                modified_indices.add(block_idx)

                block_spec = self.config_gen.flat_spec[block_idx]
                current_val = base[block_idx]
                assert isinstance(current_val, int)

                if isinstance(block_spec, PowerOfTwoFragment):
                    # Change by at most radius in log2 space
                    new_flat[block_idx] = self._random_log2_neighbor(
                        current_val,
                        radius=self.radius,
                        low=block_spec.low,
                        high=block_spec.high,
                    )
                else:
                    raise ValueError("BlockSize should be PowerOfTwoFragment")

            # 2. Sample the num_warps index and change it by at most radius
            if self.config_gen.num_warps_index:
                warp_idx = self.config_gen.num_warps_index
                modified_indices.add(warp_idx)

                warp_spec = self.config_gen.flat_spec[warp_idx]
                current_val = base[warp_idx]
                assert isinstance(current_val, int)

                if isinstance(warp_spec, PowerOfTwoFragment):
                    # Change by at most self.radius in log2 space
                    new_flat[warp_idx] = self._random_log2_neighbor(
                        current_val,
                        radius=self.radius,
                        low=warp_spec.low,
                        high=warp_spec.high,
                    )
                else:
                    raise ValueError("NumWarps should be PowerOfTwoFragment")

            # 3. For at most radius remaining indices, use pattern neighbors
            # Exclude the already-modified block size and warp indices

            # Collect available pattern neighbors for remaining indices
            remaining_pattern_neighbors = []
            for index, spec in enumerate(self.config_gen.flat_spec):
                if index not in modified_indices:
                    pattern_neighbors = spec.pattern_neighbors(base[index])
                    if pattern_neighbors:
                        remaining_pattern_neighbors.append((index, pattern_neighbors))

            # Randomly select at most radius indices to change
            if remaining_pattern_neighbors:
                num_to_change = random.randint(
                    0, min(self.radius, len(remaining_pattern_neighbors))
                )
                if num_to_change > 0:
                    indices_to_change = random.sample(
                        remaining_pattern_neighbors, num_to_change
                    )
                    for idx, pattern_neighbors in indices_to_change:
                        new_flat[idx] = random.choice(pattern_neighbors)

            # Only add if it's different from the base
            if new_flat != base:
                neighbors.append(new_flat)

        return neighbors

    def _pruned_pattern_search_from(
        self,
        current: PopulationMember,
        visited: set[Config],
    ) -> Iterator[list[PopulationMember]]:
        """
        Run a single copy of pattern search from the given starting point.

        We use a generator and yield the new population at each generation so that we can
        run multiple copies of pattern search in parallel.

        Only keep self.frac_selected of the neighbors generated from the current
        search_copy using _surrogate_select.

        Args:
            current: The current best configuration.
            visited: A set of visited configurations.

        Returns:
            A generator that yields the new population at each generation.
        """
        patience = self.patience
        for _ in range(self.max_generations):
            candidates: list[PopulationMember] = [current]
            all_neighbors = self._generate_neighbors(current.flat_values)
            for flat_config in all_neighbors:
                new_member = self.make_unbenchmarked(flat_config)
                if new_member.config not in visited:
                    candidates.append(new_member)
                    visited.add(new_member.config)

            # score candidates
            n_sorted = int(len(candidates) * self.frac_selected)
            candidates = self._surrogate_select(candidates, n_sorted)

            if len(candidates) <= 1:
                return  # no new candidates, stop searching
            yield candidates  # yield new population to benchmark in parallel
            best = min(candidates, key=performance)
            if self._check_early_stopping(best, current):
                if patience > 0:
                    patience -= 1
                else:
                    return
            current = best
