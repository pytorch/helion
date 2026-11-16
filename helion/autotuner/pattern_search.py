from __future__ import annotations

import enum
import math
import statistics
from typing import TYPE_CHECKING

from .. import exc
from ..runtime.config_cache import ConfigSet
from .base_search import PopulationBasedSearch
from .base_search import PopulationMember
from .base_search import performance
from .benchmarking import KERNEL_TRACE_TICK_US
from .compiler_coverage import coverage_policy
from .effort_profile import PATTERN_SEARCH_DEFAULTS

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import MutableSet
    from collections.abc import Sequence

    from ..autotuner.effort_profile import AutotuneEffortProfile
    from ..runtime.config import Config
    from ..runtime.settings import Settings
    from . import ConfigSpec
    from .base_search import _AutotunableKernel
    from .config_generation import ConfigGeneration
    from .config_generation import FlatConfig


# A search that reads its candidates from the CUDA activity trace
# (``PopulationBasedSearch.traces_candidates``) moves on from an incumbent
# only for a step beyond this many spreads of the population's re-readings
# (``rereading_spread``): the best of a few hundred readings of one config
# sits about three spreads of a re-reading below another reading of it.
_TIE_REREADING_SPREADS = 3.0
# ... and never for less than this fraction: two steady windows of one config
# read 0.3-0.5% apart (``base_search``'s final-selection tolerance) ...
_TIE_FLOOR = 0.005
# ... or than this many steps of the trace's clock (``KERNEL_TRACE_TICK_US``).
# A window's median sits on a tick or half-way between two, and a config that
# sits near a tick's edge reads one tick or the next from window to window
# (1.6% of a 2 us kernel): of 24 re-readings of an unchanged 1.98 us kernel,
# 3 read one tick below the other 21. A step of one tick is therefore a tie
# however many re-readings repeat to the bit, and one of two ticks is not.
_TIE_TICKS = 1.5
# Fewer re-read members than this say nothing about the spread.
_TIE_MIN_REREADINGS = 16
# The spread is the deviation that this share of the re-readings stay within,
# in units of the normal deviation with that coverage.
_SPREAD_COVERAGE = 0.9
_SPREAD_COVERAGE_DEVIATIONS = statistics.NormalDist().inv_cdf(
    0.5 + _SPREAD_COVERAGE / 2.0
)


def rereading_spread(members: Iterable[PopulationMember]) -> float:
    """How far a config's reading moves when it is taken again, as a
    fraction of the reading.

    A pattern search's generations re-time the members near their best
    (``rebenchmark_population``), and a differential-evolution comparison
    re-times a parent and child that both read within the re-timing
    threshold of the best (``PopulationBasedSearch.compare``), so those
    members hold two readings of one config: the spread is taken from
    last / previous - 1 around its median,
    which takes out what moved every re-reading alike (a longer window, a
    warmer device). It is the deviation that ``_SPREAD_COVERAGE`` of them
    stay within, scaled to a normal distribution's standard deviation: the
    tenth beyond it (bimodal outliers) does not count, and readings that sit
    on a clock's ticks neither zero it nor inflate it. (A median absolute
    deviation is zero as soon as half of the re-readings repeat to the bit,
    and a tick and a half once more than half of them move by one.) ``0.0``
    when fewer than ``_TIE_MIN_REREADINGS`` members were read twice (always
    in the surrogate hybrid of differential evolution, whose generations
    re-time no member).

    Only two readings the member took itself count
    (``PopulationMember.timer_readings_from``). A config settled with a copy
    of an earlier candidate's reading holds that copy first, which pairs
    with nothing; and no member holds readings of two timers: a search that
    moves to the trace timer replaces its members' CUDA-event readings
    (``PopulationBasedSearch._read_on_settled_timer``), which sit 4-5 us
    above the trace readings of one kernel and, left in, put the spread at
    tens of percent.
    """
    steps = [
        member.perfs[-1] / member.perfs[-2] - 1.0
        for member in members
        if len(member.perfs) - member.timer_readings_from >= 2
        and math.isfinite(member.perfs[-1])
        and math.isfinite(member.perfs[-2])
        and member.perfs[-2] > 0.0
    ]
    if len(steps) < _TIE_MIN_REREADINGS:
        return 0.0
    center = statistics.median(steps)
    deviations = sorted(abs(step - center) for step in steps)
    position = _SPREAD_COVERAGE * (len(deviations) - 1)
    below = math.floor(position)
    above = min(below + 1, len(deviations) - 1)
    covered = deviations[below] + (position - below) * (
        deviations[above] - deviations[below]
    )
    return covered / _SPREAD_COVERAGE_DEVIATIONS


def trace_tie_tolerance(search: PopulationBasedSearch, perf: float) -> float:
    """The relative step from a reading of ``perf`` that the readings of a
    search timed from the CUDA activity trace
    (``PopulationBasedSearch.traces_candidates``) repeat to: the largest of
    ``_TIE_REREADING_SPREADS`` spreads of the current population's
    re-readings (``rereading_spread``, over its kernels: a source alias
    repeats its kernel's re-reading), ``_TIE_FLOOR`` and ``_TIE_TICKS`` steps
    of the trace's clock at ``perf``. A step within it is a tie for the
    pattern paths and the polish descent (``PatternSearch._reading_tolerance``)
    and no improvement for the early stop of the differential-evolution
    searches (``DifferentialEvolutionSearch._stop_tolerance``)."""
    ticks = 0.0
    if math.isfinite(perf) and perf > 0.0:
        ticks = _TIE_TICKS * KERNEL_TRACE_TICK_US / (perf * 1000.0)
    kernels = [
        member
        for member in search.population
        if search._kernel_holder(member, search.population) is member
    ]
    return max(
        _TIE_FLOOR,
        ticks,
        _TIE_REREADING_SPREADS * rereading_spread(kernels),
    )


class InitialPopulationStrategy(enum.Enum):
    """Strategy for generating the initial population for search algorithms."""

    FROM_RANDOM = "from_random"
    """Generate a random population of configurations."""

    FROM_BEST_AVAILABLE = "from_best_available"
    """Start from default config plus up to 20 best matching cached configs from previous runs."""


def random_fallback_population_target(
    strategy: InitialPopulationStrategy,
    pad_random: bool,
    config_spec: ConfigSpec,
    population_size: int,
) -> int | None:
    """Population size to pad to when every seed/default/cache config fails.

    Only a seed-only population (FROM_BEST_AVAILABLE without random padding)
    needs the fallback. Random populations already prove the search space
    broken when they fail entirely, and CuTe flash populations are
    structurally designed rather than seeded; both keep the immediate
    compile-error re-raise.
    """
    if (
        strategy != InitialPopulationStrategy.FROM_BEST_AVAILABLE
        or pad_random
        or config_spec.cute_flash_search_enabled
        or population_size <= 0
    ):
        return None
    return population_size


class PatternSearch(PopulationBasedSearch):
    """Search that explores single-parameter perturbations around the current best."""

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
        best_available_pad_random: bool = PATTERN_SEARCH_DEFAULTS.best_available_pad_random,
        num_neighbors_cap: int = -1,
        finishing_rounds: int = 0,
        compile_timeout_lower_bound: float = PATTERN_SEARCH_DEFAULTS.compile_timeout_lower_bound,
        compile_timeout_quantile: float = PATTERN_SEARCH_DEFAULTS.compile_timeout_quantile,
    ) -> None:
        """
        Create a PatternSearch autotuner.

        Args:
            kernel: The kernel to be autotuned.
            args: The arguments to be passed to the kernel.
            initial_population: The number of random configurations to generate for the initial population.
            copies: Count of top Configs to run pattern search on.
            max_generations: The maximum number of generations to run.
            min_improvement_delta: Relative stop threshold; a path stops if
                abs(best/current - 1) < this. A search that times its
                candidates from the CUDA activity trace stops on the larger
                of this and what its readings repeat to
                (``_reading_tolerance``), so 0 no longer takes every step
                there.
            initial_population_strategy: Strategy for generating the initial population.
                FROM_RANDOM generates initial_population random configs.
                FROM_BEST_AVAILABLE uses cached configs from prior runs, and fills the
                remainder with random configs when best_available_pad_random is True.
                Can be overridden by HELION_AUTOTUNER_INITIAL_POPULATION env var (handled in default_autotuner_fn).
                If None is passed, defaults to FROM_RANDOM.
            best_available_pad_random: When True and using FROM_BEST_AVAILABLE, pad the
                cached configs with random configs to reach initial_population size.
                When False, use only the default and cached configs; random configs
                are added only as a fallback when every one of them fails to compile
                or run (see PopulationBasedSearch.benchmark_initial_population).
            num_neighbors_cap: Maximum number of neighbors to explore per generation. -1 means no cap.
                Set HELION_CAP_AUTOTUNE_NUM_NEIGHBORS=N to override.
            finishing_rounds: Number of finishing rounds to run after the main search.
            compile_timeout_lower_bound: Lower bound for adaptive compile timeout in seconds.
            compile_timeout_quantile: Quantile of compile times to use for adaptive timeout.
        """
        super().__init__(kernel, args, finishing_rounds=finishing_rounds)
        if initial_population_strategy is None:
            initial_population_strategy = InitialPopulationStrategy.FROM_RANDOM
        self.initial_population_strategy = initial_population_strategy
        self.best_available_pad_random = best_available_pad_random
        self.copies = copies
        self.max_generations = max_generations
        self.min_improvement_delta = min_improvement_delta
        # A CuTe flash surface measures one parent row per structural leaf;
        # a profile population below that count (``quick`` on the widest
        # small-grid surface) is raised here rather than for every kernel.
        self.initial_population = self.config_gen.extension.population_floor(
            initial_population
        )
        self.num_neighbors_cap = num_neighbors_cap
        self.compile_timeout_lower_bound = compile_timeout_lower_bound
        self.compile_timeout_quantile = compile_timeout_quantile

    def _algorithm_cache_policy(self) -> dict[str, object]:
        result: dict[str, object] = {
            # 2: ListOf.pattern_neighbors also proposes uniform lists.
            "pattern_version": 2,
            "initial_population": self.initial_population,
            "copies": self.copies,
            "max_generations": self.max_generations,
            "min_improvement_delta": self.min_improvement_delta,
            "initial_population_strategy": self.initial_population_strategy,
            "best_available_pad_random": self.best_available_pad_random,
            "num_neighbors_cap": self.num_neighbors_cap,
            "finishing_rounds": self.finishing_rounds,
            "compile_timeout_lower_bound": self.compile_timeout_lower_bound,
            "compile_timeout_quantile": self.compile_timeout_quantile,
        }
        policy = coverage_policy(self.config_spec.compiler_coverage_groups)
        if policy is not None:
            result["compiler_coverage"] = policy
        return result

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        from ..runtime.settings import _env_get_int
        from ..runtime.settings import _get_initial_population_strategy

        assert profile.pattern_search is not None
        strategy = _get_initial_population_strategy(
            profile.pattern_search.initial_population_strategy,
            settings.autotune_initial_population_strategy,
        )
        return {
            "initial_population": profile.pattern_search.initial_population,
            "copies": profile.pattern_search.copies,
            "max_generations": profile.pattern_search.max_generations,
            "initial_population_strategy": strategy,
            "best_available_pad_random": profile.pattern_search.best_available_pad_random,
            "num_neighbors_cap": _env_get_int("HELION_CAP_AUTOTUNE_NUM_NEIGHBORS", -1),
            **super().get_kwargs_from_profile(profile, settings),
        }

    def _generate_initial_population_flat(self) -> list[FlatConfig]:
        # Empty registries retain the exact old builder, cache lookup and draws.
        if not self.config_gen.config_spec.compiler_coverage_groups:
            population = self._generate_initial_population_base()
        else:
            generation = self.config_gen.initial_population_view()
            population = self._generate_initial_population_base(generation=generation)
            if (
                self.initial_population > 0
                and self.config_gen.compiler_coverage_enabled
            ):
                population = self._append_compiler_coverage(
                    population,
                    use_cache=self.initial_population_strategy
                    == InitialPopulationStrategy.FROM_BEST_AVAILABLE,
                )
        # Both additions follow every old return, including padding/truncation.
        # Nothing may trim or reorder the nominal/coverage prefix afterward.
        return self._append_requested_compiler_seeds(population)

    def _append_requested_compiler_seeds(
        self, population: list[FlatConfig]
    ) -> list[FlatConfig]:
        spec = self.config_gen.config_spec
        if (
            not getattr(spec, "compiler_seed_overflow_requested", False)
            or not spec.compiler_seed_configs
            or self.initial_population <= 0
        ):
            return population
        # The existing transfer path owns normalization, scalar overrides and
        # invalid-seed handling. It performs no sampling or cache lookup.
        pairs = self.config_gen.seed_flat_config_pairs(log_func=self.log)
        self.pin_finalist_configs([config for _, config in pairs])
        seen: MutableSet[Config] = ConfigSet()
        for flat in population:
            try:
                _, config = self.config_gen.canonicalize_flat(flat)
            except (exc.InvalidConfig, ValueError, TypeError, KeyError, AssertionError):
                continue
            seen.add(config)
        result = list(population)
        for flat, config in pairs:
            if config not in seen:
                result.append(flat)
                seen.add(config)
        self.log(
            f"Compiler seed overflow: appended {len(result) - len(population)} "
            f"normalized seeds after {len(population)} nominal/coverage rows"
        )
        return result

    def _generate_initial_population_base(
        self, *, generation: ConfigGeneration | None = None
    ) -> list[FlatConfig]:
        """
        Generate the initial population of flat configurations based on the strategy.

        Returns:
            A list of flat configurations for the initial population.
        """
        config_gen = self.config_gen if generation is None else generation
        if (
            self.initial_population_strategy
            == InitialPopulationStrategy.FROM_BEST_AVAILABLE
        ):
            pop = (
                self._generate_best_available_population_flat()
                if generation is None
                else self._generate_best_available_population_flat(
                    generation=config_gen
                )
            )
            if config_gen.config_spec.cute_flash_search_enabled:
                design = config_gen.extension.deterministic_population_configs()
                population_target = max(0, self.initial_population)
                budget = min(
                    population_target,
                    config_gen.extension.structural_population_budget(
                        population_target
                    ),
                    len(design),
                )
                qualification_count = min(
                    budget,
                    config_gen.extension.structural_qualification_prefix_count(),
                )
                pinned: list[FlatConfig] = []
                optional: list[FlatConfig] = []
                pinned_configs = self._pinned_finalist_configs
                for flat in pop:
                    try:
                        canonical_flat, config = config_gen.canonicalize_flat(flat)
                    except exc.InvalidConfig:
                        continue
                    (pinned if config in pinned_configs else optional).append(
                        canonical_flat
                    )

                # Structural qualification, pinned seeds/defaults, and a bounded
                # exact space are required. Canonicalize and deduplicate them
                # before limiting optional cache rows, so an alias cannot consume
                # a nominal slot.
                required = [
                    *(
                        config_gen.flatten(config)
                        for config in design[:qualification_count]
                    ),
                    *pinned,
                    *(
                        config_gen.flatten(config)
                        for config in design[qualification_count:budget]
                    ),
                ]
                exact_space = None
                if population_target > 0:
                    exact_space = (
                        config_gen.extension.exact_effective_search_space_configs(
                            population_target
                        )
                    )
                    if exact_space is not None:
                        required.extend(
                            config_gen.flatten(config) for config in exact_space
                        )
                ordered: list[FlatConfig] = []
                seen: MutableSet[Config] = ConfigSet()

                def append_unique(flat: FlatConfig) -> None:
                    try:
                        canonical_flat, config = config_gen.canonicalize_flat(flat)
                    except exc.InvalidConfig:
                        return
                    if config in seen:
                        return
                    seen.add(config)
                    ordered.append(canonical_flat)

                for flat in required:
                    append_unique(flat)
                if population_target <= 0:
                    for flat in optional:
                        append_unique(flat)
                else:
                    for flat in optional:
                        if len(ordered) >= population_target:
                            break
                        append_unique(flat)
                if self.best_available_pad_random and exact_space is None:
                    if generation is not None:
                        return self._pad_initial_population_with_unique_random(
                            ordered, population_target, generation=config_gen
                        )
                    return self._pad_initial_population_with_unique_random(
                        ordered, population_target
                    )
                return ordered
            if self.best_available_pad_random:
                n_random = max(0, self.initial_population - len(pop))
                pop.extend(config_gen.random_flat() for _ in range(n_random))
            return pop
        population = config_gen.random_population_flat(
            self.initial_population,
            user_seed_configs=self._autotune_seed_configs(),
            log_func=self.log,
        )
        # Pin the seed/default configs into final verification (mirroring the
        # FROM_BEST_AVAILABLE path): a seed's single in-search reading is often
        # burst-inflated, and without the pin a 2-6% real winner dies to a
        # noisy rival before the steady final rebenchmark can arbitrate.
        pinned_seed_configs = [
            config
            for _flat, config in (
                *config_gen.user_seed_flat_config_pairs(self._autotune_seed_configs()),
                *config_gen.seed_flat_config_pairs(),
            )
        ]
        pinned_seed_configs.append(config_gen.unflatten(config_gen.default_flat()))
        self.pin_finalist_configs(pinned_seed_configs)
        if self.config_gen.config_spec.cute_flash_search_enabled:
            return population
        return self._replace_backend_rejected_initial_configs(
            population, self.initial_population
        )

    def _random_fallback_population_target(self) -> int | None:
        return random_fallback_population_target(
            self.initial_population_strategy,
            self.best_available_pad_random,
            self.config_spec,
            self.initial_population,
        )

    def _autotune(self) -> Config:
        initial_population_name = self.initial_population_strategy.name
        self.log(
            f"Starting PatternSearch with initial_population={initial_population_name}, copies={self.copies}, max_generations={self.max_generations}"
        )
        visited: MutableSet[Config] = ConfigSet()
        self.population = []
        for flat_config in self._generate_initial_population_flat():
            member = self.make_unbenchmarked(flat_config)
            if member is not None and member.config not in visited:
                visited.add(member.config)
                self.population.append(member)
        self.benchmark_initial_population(
            self.population,
            random_fallback_target=self._random_fallback_population_target(),
            visited=visited,
        )

        # Compute adaptive compile timeout based on initial population compile times
        self.set_adaptive_compile_timeout(
            self.population,
            min_seconds=self.compile_timeout_lower_bound,
            quantile=self.compile_timeout_quantile,
        )

        # again with higher accuracy
        self.rebenchmark_population(self.population, desc="Verifying initial results")
        # Snapshot compiler-seeded members so they survive the search-loop
        # pruning into the final-pick verification candidate pool.
        self.capture_compiler_seed_members(self.population)
        self.population.sort(key=performance)
        starting_points = []
        # One path per kernel: an initial population keeps its source aliases.
        for member in self._one_member_per_kernel(self.population)[: self.copies]:
            if math.isfinite(member.perf):  # filter failed compiles
                starting_points.append(member)
        self.log(
            f"Initial random population of {len(self.population)}, {len(starting_points)} starting points:",
            self.statistics,
        )
        if not starting_points:
            raise exc.NoConfigFound

        search_copies = [self._pattern_search_from(m, visited) for m in starting_points]
        for generation in self._budgeted_range(1, self.max_generations + 1):
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
                self.set_generation(generation)
                self.benchmark_population(
                    unbenchmarked, desc=f"Generation {generation}:"
                )
                # A neighbor that repeats earlier code is not a new member.
                self.population = self._without_source_aliases(
                    self.population, unbenchmarked
                )
            # higher-accuracy rebenchmark
            self.rebenchmark_population(
                self.population, desc=f"Generation {generation}: verifying top configs"
            )
            # Log final statistics for this generation
            self.log(f"Generation {generation} complete:", self.statistics)

        # Final verification, finishing phase, and (TPU-only) final-pick re-rank.
        return self._finalize()

    def _pattern_search_from(
        self, current: PopulationMember, visited: MutableSet[Config]
    ) -> Iterator[list[PopulationMember]]:
        """
        Run a single copy of pattern search from the given starting point.

        We use a generator and yield the new population at each generation so that we can
        run multiple copies of pattern search in parallel.
        """
        for _ in range(self.max_generations):
            candidates = [current]
            for flat_config in self._generate_neighbors(current.flat_values):
                new_member = self.make_unbenchmarked(flat_config)
                if new_member is not None and new_member.config not in visited:
                    visited.add(new_member.config)
                    candidates.append(new_member)
            if len(candidates) <= 1:
                return  # no new candidates, stop searching
            yield candidates  # yield new population to benchmark in parallel
            # update search copy and check early stopping criteria; a neighbor
            # that repeats earlier code is no step for this path
            candidates = self._without_source_aliases(candidates, candidates[1:])
            best = min(candidates, key=performance)
            if self._check_early_stopping(best, current):
                return
            current = best

    def _check_early_stopping(
        self, best: PopulationMember, current: PopulationMember
    ) -> bool:
        """
        Check if early stopping criteria are met for the search copy

        Early stops if either the best config has not changed or if the
        relative improvement is a tie at the incumbent's reading
        (``_reading_tolerance``: the user-specified delta, and in a search
        timed from the CUDA activity trace also what its readings repeat
        to); a tie beyond the delta is logged

        Returns:
            True the search copy is terminated, False otherwise.
        """
        if best is current:
            return True  # no improvement, stop searching
        # Stop if the relative improvement is smaller than a user-specified
        # delta, or than the incumbent's readings repeat to. (A search
        # without a tolerance takes every step, and a reading of zero has no
        # relative step.)
        tolerance = self._reading_tolerance(current.perf)
        if not (
            tolerance > 0.0
            and current.perf != 0.0
            and self._reads_level(best.perf, current.perf, tolerance)
        ):
            return False
        step = abs(best.perf / current.perf - 1.0)
        if step >= self.min_improvement_delta:
            self.log(
                f"A step of {step:.2%} from {self.format_performance(current.perf)} "
                f"is a tie: readings of one config repeat to {tolerance:.2%}"
            )
        return True

    def _reads_level(
        self, perf: float, reference: float, tolerance: float | None = None
    ) -> bool:
        """Whether a reading of ``perf`` is a tie with ``reference``: equal
        to it, or within ``tolerance`` of it in either direction
        (``_reading_tolerance`` at ``reference`` unless given: a caller that
        compares many readings with one reference computes it once).  It is
        the step ``_check_early_stopping`` does not count as an improvement
        and the one the polish descent crosses to new code, by one
        tolerance."""
        if not math.isfinite(perf) or not math.isfinite(reference):
            return False
        if perf == reference:
            return True
        if tolerance is None:
            tolerance = self._reading_tolerance(reference)
        return (
            tolerance > 0.0
            and reference != 0.0
            and abs(perf / reference - 1.0) < tolerance
        )

    def _reading_tolerance(self, perf: float) -> float:
        """The relative step from a reading of ``perf`` below which two
        readings are a tie: the one definition of an improvement for the
        search paths and for the polish descent, its level moves and its
        patience included.

        ``min_improvement_delta``, except in a search that reads its
        candidates from the CUDA activity trace (``traces_candidates``; a
        search timed with CUDA events keeps the fixed delta whatever its
        backend): that timer resolves steps far smaller than it repeats to,
        so a path that took every such step as an improvement would walk on
        noise through all its generations and polish rounds. There a step
        also has to exceed what the search's readings repeat to at ``perf``
        (``trace_tie_tolerance``).
        """
        if not self.traces_candidates:
            return self.min_improvement_delta
        return max(self.min_improvement_delta, trace_tie_tolerance(self, perf))

    def shrink_neighbors(self, neighbors: list[FlatConfig]) -> list[FlatConfig]:
        if self.num_neighbors_cap > 0:
            return neighbors[: self.num_neighbors_cap]
        return neighbors

    def _generate_neighbors(self, base: FlatConfig) -> list[FlatConfig]:
        """
        Generate neighboring configurations by changing one or two parameters at a time.
        """
        overridden = self.config_gen.overridden_flat_indices
        candidates_by_index = [
            spec.pattern_neighbors(base[index]) if index not in overridden else []
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
        block_indices = [
            i for i in self.config_gen.block_size_indices if i not in overridden
        ]
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

        return self.shrink_neighbors(
            [self.config_gen.conditional_neighbor(base, row) for row in neighbors]
        )
