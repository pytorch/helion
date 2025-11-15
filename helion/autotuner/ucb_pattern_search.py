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
    from botorch.acquisition import UpperConfidenceBound
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import MixedSingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    HAS_BO_DEPS = True
except ImportError as e:
    HAS_BO_DEPS = False
    _IMPORT_ERROR = e


class UCBPatternSearch(PatternSearch):
    """
    Upper Confidence Bound (UCB) Pattern Search

    Modifies PatternSearch to (1) generate random neighbors from each search copy
    within a set radius, (2) filter the neighbors to benchmark using a fitted GaussianProcess
    with the UCB acquisition function.

    Uses the MixedSingleTaskGP model from botorch, which supports continuous
    and categorical variables. It only fits the GP once to avoid long runtimes.
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
        frac_selected: float = 0.3,
        num_neighbors: int = 100,
        radius: int = 2,
        ucb_beta: float = 2.0,
    ) -> None:
        if not HAS_BO_DEPS:
            raise exc.MissingDependency(
                "UCBPatternSearch requires botorch>=0.16.0.Install before using."
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

    def fit_gp(
        self, train_X: torch.Tensor, train_Y: torch.Tensor, cat_dims: list
    ) -> MixedSingleTaskGP:
        # Filter out rows where train_Y contains inf or nan

        gp = MixedSingleTaskGP(
            train_X,
            -train_Y.unsqueeze(-1),
            cat_dims,
        )

        with torch.enable_grad():
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

        return gp

    def acq_fun(self, X: torch.Tensor, gp: MixedSingleTaskGP) -> torch.Tensor:
        orig_dtype = X.dtype

        acq_fun = UpperConfidenceBound(gp, beta=self.ucb_beta)
        return (
            acq_fun(X.unsqueeze(1).to(dtype=torch.float64))
            .detach()
            .to(dtype=orig_dtype)
        )

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
        gp = self.fit_gp(
            train_X,
            train_Y,
            self.cat_dims,
        )

        search_copies = [
            self._pruned_pattern_search_from(m, visited, gp) for m in starting_points
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
            gp = gp.condition_on_observations(train_X, -train_Y.unsqueeze(1))

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

            # score candidates
            candidate_X = torch.stack(
                [
                    torch.tensor(self.config_gen.encode_config(member.flat_values))
                    for member in candidates
                ]
            )
            scores = self.acq_fun(candidate_X, gp)

            # filter candidates by score
            candidates_sorted = sorted(
                zip(candidates, scores, strict=True),
                key=operator.itemgetter(1),
                reverse=True,
            )[: int(self.frac_selected * len(candidates))]
            candidates = [member for member, score in candidates_sorted]
            visited.update([member.config for member in candidates])

            self.log(
                f"Scoring {len(candidate_X)} neighbors, selecting {self.frac_selected * 100}% neighbors: {len(candidates)}"
            )

            if len(candidates) <= 1:
                return  # no new candidates, stop searching
            yield candidates  # yield new population to benchmark in parallel
            best = min(candidates, key=performance)
            if best is current:
                return  # no improvement, stop searching
            # Stop if the relative improvement is smaller than a user-specified delta
            if (
                self.min_improvement_delta > 0.0
                and math.isfinite(best.perf)
                and math.isfinite(current.perf)
                and current.perf != 0.0
                and abs(best.perf / current.perf - 1.0) < self.min_improvement_delta
            ):
                return
            current = best
