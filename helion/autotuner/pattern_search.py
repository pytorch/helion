from __future__ import annotations

import dataclasses
import enum
import math
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from torch.utils._ordered_set import OrderedSet

from .. import exc
from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .base_search import performance
from .effort_profile import PATTERN_SEARCH_DEFAULTS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .base_search import _AutotunableKernel
    from .config_generation import FlatConfig


class InitialPopulationStrategy(enum.Enum):
    """Strategy for generating the initial population for search algorithms."""

    FROM_RANDOM = "from_random"
    """Generate a random population of configurations."""

    FROM_DEFAULT = "from_default"
    """Start from only the default configuration."""


@dataclasses.dataclass
class PatternSearchCopy:
    """
    Represents one copy of the pattern search.

    Each copy explores from a different starting point. The `copies` parameter
    controls how many of these run in parallel.
    """

    # The current best member for this search copy.
    current: PopulationMember

    # The number of generations this copy has run.
    generation: int = 0

    # Whether this search copy has stopped (no more candidates or early stopping).
    stopped: bool = False

    def generate_candidates(
        self, parent: PatternSearch, visited: OrderedSet[Config]
    ) -> list[PopulationMember] | None:
        """
        Generate candidates for this search copy.

        Args:
            parent: The parent PatternSearch instance.
            visited: Set of already-visited configs (shared across copies).

        Returns:
            List of candidates to benchmark (including current), or None if stopped.
        """
        if self.stopped:
            return None

        candidates = [self.current]
        for flat_config in parent._generate_neighbors(self.current.flat_values):
            new_member = parent.make_unbenchmarked(flat_config)
            if new_member.config not in visited:
                visited.add(new_member.config)
                candidates.append(new_member)

        if len(candidates) <= 1:
            self.stopped = True
            return None

        return candidates

    def to_dict(self, member_id_to_idx: dict[int, int]) -> dict[str, Any]:
        """Serialize this search copy to a dict."""
        return {
            "current_index": member_id_to_idx[id(self.current)],
            "generation": self.generation,
            "stopped": self.stopped,
        }

    @classmethod
    def from_dict(
        cls, state_data: dict[str, Any], current: PopulationMember
    ) -> PatternSearchCopy:
        """Create a search copy from serialized data."""
        return cls(
            current=current,
            generation=state_data["generation"],
            stopped=state_data["stopped"],
        )


class PatternSearch(PopulationBasedSearch):
    """Search that explores single-parameter perturbations around the current best."""

    # Keys that this class contributes to state_dict for checkpointing.
    _checkpoint_state_dict_keys: ClassVar[set[str]] = {
        "initial_population_strategy",
        "copies",
        "max_generations",
        "min_improvement_delta",
        "initial_population",
        "compile_timeout_lower_bound",
        "compile_timeout_quantile",
        "visited",
        "search_copies",
    }

    # Instance attributes that are intentionally NOT checkpointed.
    _checkpoint_excluded_attrs: ClassVar[set[str]] = set()

    search_copy_class: ClassVar[type[PatternSearchCopy]] = PatternSearchCopy

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
        compile_timeout_lower_bound: float = PATTERN_SEARCH_DEFAULTS.compile_timeout_lower_bound,
        compile_timeout_quantile: float = PATTERN_SEARCH_DEFAULTS.compile_timeout_quantile,
    ) -> None:
        """
        Create a PatternSearch autotuner.

        Args:
            kernel: The kernel to be autotuned.
            args: The arguments to be passed to the kernel.
            initial_population: The number of random configurations to generate for the initial population.
                When using FROM_DEFAULT strategy, this is ignored (always 1).
            copies: Count of top Configs to run pattern search on.
            max_generations: The maximum number of generations to run.
            min_improvement_delta: Relative stop threshold; stop if abs(best/current - 1) < this.
            initial_population_strategy: Strategy for generating the initial population.
                FROM_RANDOM generates initial_population random configs.
                FROM_DEFAULT starts from only the default configuration.
                Can be overridden by HELION_AUTOTUNER_INITIAL_POPULATION env var (handled in default_autotuner_fn).
                If None is passed, defaults to FROM_RANDOM.
            compile_timeout_lower_bound: Lower bound for adaptive compile timeout in seconds.
            compile_timeout_quantile: Quantile of compile times to use for adaptive timeout.
        """
        super().__init__(kernel, args)
        if initial_population_strategy is None:
            initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        self.initial_population_strategy = initial_population_strategy
        self.copies = copies
        self.max_generations = max_generations
        self.min_improvement_delta = min_improvement_delta
        self.initial_population = initial_population
        self.compile_timeout_lower_bound = compile_timeout_lower_bound
        self.compile_timeout_quantile = compile_timeout_quantile
        self.visited: OrderedSet[Config] = OrderedSet()
        self.search_copies: list[PatternSearchCopy] = []

    def _generate_initial_population_flat(self) -> list[FlatConfig]:
        """
        Generate the initial population of flat configurations based on the strategy.

        Returns:
            A list of flat configurations for the initial population.
        """
        if self.initial_population_strategy == InitialPopulationStrategy.FROM_DEFAULT:
            return [self.config_gen.default_flat()] * self.initial_population
        return self.config_gen.random_population_flat(self.initial_population)

    def _init_search(self) -> None:
        """Initialize PatternSearch state for a fresh run."""
        initial_population_name = self.initial_population_strategy.name
        self.log(
            f"Starting PatternSearch with initial_population={initial_population_name}, copies={self.copies}, max_generations={self.max_generations}"
        )

        # Initialize population from flat configs, filtering duplicates
        self.visited.clear()
        self.population = []
        for flat_config in self._generate_initial_population_flat():
            member = self.make_unbenchmarked(flat_config)
            if member.config not in self.visited:
                self.visited.add(member.config)
                self.population.append(member)

        # Benchmark initial population
        self.set_generation(0)
        self.parallel_benchmark_population(self.population, desc="Initial population")

        # Compute adaptive compile timeout based on initial population compile times
        self.set_adaptive_compile_timeout(
            self.population,
            min_seconds=self.compile_timeout_lower_bound,
            quantile=self.compile_timeout_quantile,
        )

        # again with higher accuracy
        self.rebenchmark_population(self.population, desc="Verifying initial results")

        # Get starting points (top performers with finite perf)
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

        # Initialize search states from starting points
        self.search_copies = [
            self.search_copy_class(current=m) for m in starting_points
        ]

        # Set to first generation so loop starts correctly for both fresh and restored runs
        self.set_generation(1)

    def _autotune(self) -> Config:
        for generation in range(self._current_generation, self.max_generations + 1):
            self.set_generation(generation)
            prior_best = self.best
            new_population = {id(prior_best): prior_best}
            num_neighbors = 0
            num_active = 0

            # Collect candidates from all active search copies
            active_copies: list[tuple[PatternSearchCopy, list[PopulationMember]]] = []
            for search_copy in self.search_copies:
                candidates = search_copy.generate_candidates(self, self.visited)
                if candidates:
                    num_active += 1
                    num_neighbors += len(candidates) - 1
                    for member in candidates:
                        new_population[id(member)] = member
                    active_copies.append((search_copy, candidates))

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

            # Update each search copy after rebenchmarking (uses refined perf values)
            for search_copy, candidates in active_copies:
                best = min(candidates, key=performance)
                if self._check_early_stopping(best, search_copy.current):
                    search_copy.stopped = True
                if not search_copy.stopped:
                    search_copy.current = best
                search_copy.generation += 1

            # Log final statistics for this generation
            self.log(f"Generation {generation} complete:", self.statistics)
        return self.best.config

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

    def _generate_neighbors(self, base: FlatConfig) -> list[FlatConfig]:
        """
        Generate neighboring configurations by changing one or two parameters at a time.
        """
        candidates_by_index = [
            spec.pattern_neighbors(base[index])
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
        block_indices = self.config_gen.block_size_indices
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

        return neighbors

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state including PatternSearch-specific fields."""
        state = super().state_dict()
        state.update(
            {
                "initial_population_strategy": self.initial_population_strategy.value,
                "copies": self.copies,
                "max_generations": self.max_generations,
                "min_improvement_delta": self.min_improvement_delta,
                "initial_population": self.initial_population,
                "compile_timeout_lower_bound": self.compile_timeout_lower_bound,
                "compile_timeout_quantile": self.compile_timeout_quantile,
                "visited": self.visited,
            }
        )

        # Serialize only non-stopped search_copies. Stopped copies are excluded because:
        # 1. They won't contribute to future search (generate_candidates returns None)
        # 2. Their current member may not be in population (it's not being explored)
        # 3. On restore, we only need active copies that will continue searching
        member_id_to_idx = {id(m): i for i, m in enumerate(self.population)}
        state["search_copies"] = [
            sc.to_dict(member_id_to_idx) for sc in self.search_copies if not sc.stopped
        ]

        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore PatternSearch-specific state."""
        super().load_state_dict(state)

        # Restore PatternSearch-specific fields
        self.initial_population_strategy = InitialPopulationStrategy(
            state["initial_population_strategy"]
        )
        self.copies = state["copies"]
        self.max_generations = state["max_generations"]
        self.min_improvement_delta = state["min_improvement_delta"]
        self.initial_population = state["initial_population"]
        self.compile_timeout_lower_bound = state["compile_timeout_lower_bound"]
        self.compile_timeout_quantile = state["compile_timeout_quantile"]
        self.visited = state["visited"]

        # Restore search_copies
        self.search_copies = [
            self.search_copy_class.from_dict(sd, self.population[sd["current_index"]])
            for sd in state["search_copies"]
        ]
