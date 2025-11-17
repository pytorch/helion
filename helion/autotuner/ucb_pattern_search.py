from __future__ import annotations

from itertools import accumulate
import math
import operator
import random
from typing import TYPE_CHECKING

import torch

from .. import exc
from .base_search import FlatConfig
from .base_search import PopulationMember
from .base_search import performance
from .config_fragment import PowerOfTwoFragment
from .effort_profile import PATTERN_SEARCH_DEFAULTS
from .pattern_search import PatternSearch

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel

try:
    from botorch.acquisition import (
        qUpperConfidenceBound,  # type: ignore[import-not-found]
    )
    from botorch.fit import fit_gpytorch_mll  # type: ignore[import-not-found]
    from botorch.models import MixedSingleTaskGP  # type: ignore[import-not-found]
    from gpytorch.mlls import (
        ExactMarginalLogLikelihood,  # type: ignore[import-not-found]
    )

    HAS_BO_DEPS = True
except ImportError as e:
    HAS_BO_DEPS = False
    _IMPORT_ERROR = e


class UCBPatternSearch(PatternSearch):
    """
    Upper Confidence Bound (UCB) Pattern Search - A Bayesian optimization-guided autotuner.

    This algorithm enhances PatternSearch by using Gaussian Process surrogate models
    with Upper Confidence Bound (UCB) acquisition to intelligently select which
    configurations to benchmark, reducing the number of kernel compilations and runs
    needed to find optimal configurations.

    Algorithm Overview:
        1. Generate an initial random population and benchmark all configurations
        2. Fit a Gaussian Process (GP) model on the benchmarked data
        3. For each generation:
           - Generate random neighbors around the current best configurations
           - Score all neighbors using UCB acquisition function
           - Benchmark only the top frac_selected fraction of neighbors
           - Condition the GP on new observations (rather than refitting)
           - Update search trajectories based on new results

    Key Differences from PatternSearch:
        - Generates num_neighbors random neighbors (within radius) instead of
          systematic single-parameter perturbations
        - Uses GP+UCB to filter which neighbors to actually benchmark, significantly
          reducing compilation/benchmark overhead
        - Supports both continuous (power-of-two) and categorical parameters via
          MixedSingleTaskGP from BoTorch

    Args:
        kernel: The kernel to be autotuned.
        args: The arguments to be passed to the kernel during benchmarking.
        initial_population: Number of random configurations in initial population.
            Default from PATTERN_SEARCH_DEFAULTS.
        copies: Number of top configurations to run pattern search from.
            Default from PATTERN_SEARCH_DEFAULTS.
        max_generations: Maximum number of search iterations per copy.
            Default from PATTERN_SEARCH_DEFAULTS.
        min_improvement_delta: Early stopping threshold. Search stops if the relative
            improvement abs(best/current - 1) < min_improvement_delta.
            Default: 0.0005 (0.05% improvement threshold).
        frac_selected: Fraction of generated neighbors to actually benchmark, after
            filtering by UCB score. Range: (0, 1]. Lower values reduce benchmarking
            cost but may miss good configurations. Default: 0.3.
        num_neighbors: Number of random neighbor configurations to generate around
            each search point per generation. Default: 100.
        radius: Maximum perturbation distance in configuration space. For power-of-two
            parameters, this is the max change in log2 space. For other parameters,
            this limits how many parameters can be changed. Default: 2.
        ucb_beta: Exploration/exploitation trade-off parameter for UCB acquisition.
            Higher values favor exploration of uncertain regions. Typical range: [1, 5].
            Default: 2.0.
        use_greedy_batch: If True, use greedy batch acquisition where points are
            selected sequentially, conditioning the GP on each selected point before
            choosing the next. This produces more diverse batches but is slower.
            If False, all points are scored independently (standard UCB).
            Default: False.
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
        frac_selected: float = 0.1,
        num_neighbors: int = 300,
        radius: int = 2,
        ucb_beta: float = 2.0,
    ) -> None:
        if not HAS_BO_DEPS:
            raise exc.AutotuneError(
                "UCBPatternSearch requires botorch. Install before using."
            ) from _IMPORT_ERROR

        super().__init__(
            kernel=kernel,
            args=args,
            initial_population=initial_population,
            copies=copies,
            max_generations=max_generations,
            min_improvement_delta=min_improvement_delta,
        )
        # Storage for BO
        self.num_neighbors = num_neighbors
        self.radius = radius
        self.ucb_beta = ucb_beta

        # Initialize config encoder
        self.frac_selected = frac_selected

        # compute offsets from the flat_spec
        dim_sizes = [spec.dim() for spec in self.config_gen.flat_spec]
        offsets = [0, *list(accumulate(dim_sizes))]

        self.cat_dims = [
            idx
            for i, spec in enumerate(self.config_gen.flat_spec)
            if spec.is_categorical()
            for idx in range(offsets[i], offsets[i + 1])
        ]

    def _fit_gp(
        self, train_X: torch.Tensor, train_Y: torch.Tensor, cat_dims: list
    ) -> MixedSingleTaskGP:
        # Filter out rows where train_Y contains inf or nan

        gp = MixedSingleTaskGP(  # type: ignore[misc]
            train_X,
            -train_Y.unsqueeze(-1),
            cat_dims,
        )

        with torch.enable_grad():
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)  # type: ignore[misc]
            fit_gpytorch_mll(mll)  # type: ignore[misc]

        return gp

    def _optimize_batch_acq(
        self,
        candidates: list[PopulationMember],
        gp: MixedSingleTaskGP,
        num_select: int,
    ) -> list[PopulationMember]:
        """
        Greedily optimize the set-valued UCB acquisition function.

        This treats the acquisition function as set-valued: it evaluates the value
        of acquiring a batch of points together. We greedily build up the batch by:
        1. Start with empty selected set
        2. For each candidate, evaluate acq_fun(selected_set ∪ {candidate})
        3. Select the candidate that maximizes this set value
        4. Add it to selected set and repeat

        This encourages diversity since adding a point near already-selected points
        typically yields lower marginal gain.

        Args:
            candidates: List of candidate configurations to select from
            gp: The Gaussian Process model
            num_select: Number of candidates to select

        Returns:
            List of selected candidates (in order of selection)
        """
        selected: list[PopulationMember] = []
        selected_indices: list[int] = []
        remaining_indices = list(range(len(candidates)))

        acq_fn = qUpperConfidenceBound(gp, beta=self.ucb_beta)  # type: ignore[misc]

        candidate_X = torch.stack(
            [
                torch.tensor(self.config_gen.encode_config(member.flat_values))
                for member in candidates
            ]
        )

        for _ in range(num_select):
            if not remaining_indices:
                break

            # Batch evaluate all remaining candidates at once
            if selected_indices:
                # Build batch: for each remaining, create [selected + remaining[i]]
                # Shape: [num_remaining, num_selected + 1, D]
                num_remaining = len(remaining_indices)

                # Expand selected points to [num_remaining, num_selected, D]
                selected_X = candidate_X[selected_indices]  # [num_selected, D]
                expanded_selected = selected_X.unsqueeze(0).expand(
                    num_remaining, -1, -1
                )

                # Get remaining candidates as [num_remaining, 1, D]
                remaining_X = candidate_X[remaining_indices].unsqueeze(1)

                # Concatenate to get [num_remaining, num_selected+1, D]
                batch_X = torch.cat([expanded_selected, remaining_X], dim=1)

                # Evaluate all sets at once: [num_remaining]
                set_values = acq_fn(batch_X.to(dtype=torch.float64))  # [num_remaining]
            else:
                # First selection: evaluate each candidate independently
                remaining_X = candidate_X[remaining_indices].unsqueeze(
                    1
                )  # [num_remaining, 1, D]
                set_values = acq_fn(
                    remaining_X.to(dtype=torch.float64)
                )  # [num_remaining]

            # Select the best
            best_idx_in_remaining = int(set_values.argmax())
            best_idx = remaining_indices[best_idx_in_remaining]

            selected.append(candidates[best_idx])
            selected_indices.append(best_idx)
            remaining_indices.pop(best_idx_in_remaining)

        return selected

    def get_train_data_from_pop(
        self, population: list[PopulationMember]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        train_X = []
        train_Y = []
        for member in population:
            train_X.append(
                torch.tensor(self.config_gen.encode_config(member.flat_values))
            )
            train_Y.append(member.perf)

        train_X = torch.stack(train_X)
        train_Y = torch.tensor(train_Y)

        valid_mask = torch.isfinite(train_Y)
        train_X_filtered = train_X[valid_mask].to(dtype=torch.float64)
        train_Y_filtered = train_Y[valid_mask].to(dtype=torch.float64)

        return train_X_filtered, train_Y_filtered

    def _autotune(self) -> Config:
        self.log(
            f"Starting UCBPatternSearch with initial_population={self.initial_population}, copies={self.copies}, max_generations={self.max_generations}"
        )
        visited = set()
        self.population = []
        for flat_config in self.config_gen.random_population_flat(
            self.initial_population
        ):
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
        train_X, train_Y = self.get_train_data_from_pop(self.population)

        # Fit GP
        self.log(f"Fitting GP: {len(train_X)} points, {len(train_Y)} targets")
        self.gp = self._fit_gp(
            train_X,
            train_Y,
            self.cat_dims,
        )

        search_copies = [
            self._pruned_pattern_search_from(m, visited, self.gp)
            for m in starting_points
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

            # Save to training data
            train_X, train_Y = self.get_train_data_from_pop(self.population)

            self.log(
                f"Conditioning on new data: {len(train_X)} points, {len(train_Y)} targets"
            )
            self.gp = self.gp.condition_on_observations(train_X, -train_Y.unsqueeze(1))

        return self.best.config

    def random_log2_neighbor(
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

            # 1. Sample a block size index and change it by at most 1
            if self.config_gen.block_size_indices:
                block_idx = random.choice(self.config_gen.block_size_indices)
                modified_indices.add(block_idx)

                block_spec = self.config_gen.flat_spec[block_idx]
                current_val = base[block_idx]
                assert type(current_val) is int

                if isinstance(block_spec, PowerOfTwoFragment):
                    # Change by at most 1 in log2 space
                    new_flat[block_idx] = self.random_log2_neighbor(
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
                assert type(current_val) is int

                if isinstance(warp_spec, PowerOfTwoFragment):
                    # Change by at most self.radius in log2 space
                    new_flat[warp_idx] = self.random_log2_neighbor(
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
        gp: MixedSingleTaskGP,
    ) -> Iterator[list[PopulationMember]]:
        """
        Run a single copy of pattern search from the given starting point.

        We use a generator and yield the new population at each generation so that we can
        run multiple copies of pattern search in parallel.

        Only keep self.frac_selected of the neighbors generated from the current
        search_copy. Filter them using the GaussianProcess + UCB acqusition function.
        """
        for _ in range(self.max_generations):
            candidates = [current]
            all_neighbors = self._generate_neighbors(current.flat_values)
            self.log(f"Number of all candidate neighbors: {len(all_neighbors)}")
            for flat_config in all_neighbors:
                new_member = self.make_unbenchmarked(flat_config)
                if new_member.config not in visited:
                    candidates.append(new_member)
                    visited.add(new_member.config)

            # Select candidates using greedy batch or standard independent scoring
            num_neighbors = len(candidates)
            num_to_select = int(self.frac_selected * num_neighbors)
            candidates = self._optimize_batch_acq(candidates, gp, num_to_select)

            self.log(
                f"Scoring {num_neighbors} neighbors, selecting {self.frac_selected * 100}% neighbors: {len(candidates)}"
            )

            if len(candidates) <= 1:
                return  # no new candidates, stop searching
            yield candidates  # yield new population to benchmark in parallel
            # update search copy and check early stopping criteria
            best = min(candidates, key=performance)
            if self._check_early_stopping(best, current):
                return
            current = best
