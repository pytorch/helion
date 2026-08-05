"""Run a two-stage hybrid autotuner that seeds a local search with an LLM pass.

High-level flow:
1. Run ``LLMGuidedSearch`` for ``llm_max_rounds`` rounds and keep its best
   config. The hybrid defaults to 1 LLM round.
2. Run a second-stage non-LLM search, ``LFBOTreeSearch`` by default.
3. If the second stage supports best-available seeding, force
   ``FROM_BEST_AVAILABLE`` and inject the LLM best config so stage 2 can refine
   it instead of starting cold.
4. Report per-stage timing and config-count metrics, plus aggregated hybrid
   totals.

Setting ``llm_max_rounds=0`` skips the LLM stage and runs only the second
stage.
"""

from __future__ import annotations

import math
import operator
import os
import time
from typing import TYPE_CHECKING
from typing import cast

from .. import exc
from .base_search import BaseSearch
from .base_search import PopulationBasedSearch
from .candidate_budget import AttemptBudget
from .candidate_budget import CandidatePopulationUnderfilled
from .candidate_budget import SharedAttemptState
from .effort_profile import QUICK_LLM_SEARCH_DEFAULTS
from .llm.transport import DEFAULT_REQUEST_TIMEOUT_S
from .llm_search import LLMGuidedSearch
from .llm_search import guided_search_kwargs_from_config
from .pattern_search import InitialPopulationStrategy

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.settings import Settings
    from .base_search import _AutotunableKernel
    from .effort_profile import AutotuneEffortProfile


_DISALLOWED_SECOND_STAGE_ALGORITHMS = {
    "LLMGuidedSearch",
    "LLMSeededSearch",
    "LLMSeededLFBOTreeSearch",
}
_AGGREGATED_METRIC_FIELDS = (
    "num_configs_tested",
    "num_compile_failures",
    "num_accuracy_failures",
    "num_generations",
)


def _resolve_second_stage_algorithm(name: str) -> type[BaseSearch]:
    """Resolve and validate the non-LLM search used in stage 2."""
    from . import search_algorithms

    search_cls = search_algorithms.get(name)
    if search_cls is None:
        raise ValueError(
            f"Unknown hybrid second-stage algorithm: {name}. "
            f"Valid options are: {', '.join(search_algorithms.keys())}"
        )
    if name in _DISALLOWED_SECOND_STAGE_ALGORITHMS:
        raise ValueError(
            f"Invalid hybrid second-stage algorithm: {name}. "
            "The second stage must be a non-LLM search algorithm."
        )
    return search_cls


def _supports_best_available_handoff(search_cls: type[BaseSearch]) -> bool:
    """Return whether the second stage supports FROM_BEST_AVAILABLE seeding."""
    from .differential_evolution import DifferentialEvolutionSearch
    from .pattern_search import PatternSearch

    return issubclass(search_cls, (PatternSearch, DifferentialEvolutionSearch))


class LLMSeededSearch(BaseSearch):
    """
    Generic hybrid autotuner that seeds a second-stage search with LLM proposals.

    The algorithm runs in two stages:
    1. Run ``LLMGuidedSearch`` for ``llm_max_rounds`` rounds and capture its best
       config in memory.
    2. Run the configured second-stage search algorithm. If the algorithm
       supports best-available seeding, it is switched to
       ``FROM_BEST_AVAILABLE`` so it can start from the LLM seed config.

    Setting ``llm_max_rounds=0`` disables the seed stage and runs only the
    second-stage search.
    """

    default_second_stage_algorithm = "LFBOTreeSearch"
    allow_second_stage_env_override = True
    hybrid_stage_breakdown: dict[str, object] | None
    _llm_stage_search: LLMGuidedSearch | None
    _second_stage_search: BaseSearch | None
    _shared_attempt_state: SharedAttemptState | None
    _best_overall_config: Config | None

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        second_stage_algorithm: str | None = None,
        second_stage_kwargs: dict[str, object] | None = None,
        best_available_pad_random: bool = False,
        llm_provider: str | None = None,
        llm_model: str = QUICK_LLM_SEARCH_DEFAULTS.model,
        llm_configs_per_round: int = QUICK_LLM_SEARCH_DEFAULTS.configs_per_round,
        llm_max_rounds: int = QUICK_LLM_SEARCH_DEFAULTS.max_rounds,
        llm_initial_random_configs: int = QUICK_LLM_SEARCH_DEFAULTS.initial_random_configs,
        llm_compile_timeout_s: int | None = QUICK_LLM_SEARCH_DEFAULTS.compile_timeout_s,
        llm_api_base: str | None = None,
        llm_api_key: str | None = None,
        llm_request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        llm_effort_level: str | None = None,
        llm_fast_mode: bool = False,
    ) -> None:
        super().__init__(kernel, args)
        if llm_max_rounds < 0:
            raise ValueError("LLMSeededSearch llm_max_rounds must be >= 0")
        self.second_stage_algorithm = (
            second_stage_algorithm or type(self).default_second_stage_algorithm
        )
        self._second_stage_search_cls = _resolve_second_stage_algorithm(
            self.second_stage_algorithm
        )
        self._second_stage_supports_best_available_handoff = (
            _supports_best_available_handoff(self._second_stage_search_cls)
        )
        self.second_stage_kwargs = dict(second_stage_kwargs or {})
        self.best_available_pad_random = best_available_pad_random

        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_configs_per_round = llm_configs_per_round
        self.llm_max_rounds = llm_max_rounds
        self.llm_initial_random_configs = llm_initial_random_configs
        self.llm_compile_timeout_s = llm_compile_timeout_s
        self.llm_api_base = llm_api_base
        self.llm_api_key = llm_api_key
        self.llm_request_timeout_s = llm_request_timeout_s
        self.llm_effort_level = llm_effort_level
        self.llm_fast_mode = llm_fast_mode

        self.hybrid_stage_breakdown = None
        self._llm_stage_search = None
        self._second_stage_search = None
        self._shared_attempt_state = None
        self._best_overall_config = None

    @property
    def config_gen(self) -> object:
        """Delegate config normalization to a stage search.

        The composed search never builds its own ``config_gen`` (only
        ``PopulationBasedSearch`` does), so canonical event emission borrows a
        stage's -- both derive from the same ``config_spec`` and normalize
        identically. Raises until a stage has run.
        """
        for source in (self._llm_stage_search, self._second_stage_search):
            generator = getattr(source, "config_gen", None) if source else None
            if generator is not None:
                return generator
        raise AttributeError(f"{type(self).__name__} has no config_gen yet")

    @classmethod
    def _get_default_second_stage_algorithm(cls) -> str:
        """Read the default stage-2 algorithm, optionally from env."""
        if (
            cls.allow_second_stage_env_override
            and (value := os.environ.get("HELION_HYBRID_SECOND_STAGE_ALGORITHM"))
            is not None
        ):
            return value
        return cls.default_second_stage_algorithm

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        """Combine shared LLM defaults with the chosen second-stage profile."""
        second_stage_algorithm = cls._get_default_second_stage_algorithm()
        second_stage_cls = _resolve_second_stage_algorithm(second_stage_algorithm)

        # The hybrid uses a quick LLM seed stage by default, even under full effort.
        guided_kwargs = guided_search_kwargs_from_config(
            QUICK_LLM_SEARCH_DEFAULTS, settings
        )
        llm_kwargs: dict[str, object] = {
            f"llm_{k}": v for k, v in guided_kwargs.items()
        }

        kwargs = {
            **super().get_kwargs_from_profile(profile, settings),
            "second_stage_algorithm": second_stage_algorithm,
            "second_stage_kwargs": second_stage_cls.get_kwargs_from_profile(
                profile, settings
            ),
            **llm_kwargs,
            "best_available_pad_random": False,
        }

        if (value := os.environ.get("HELION_HYBRID_LLM_MAX_ROUNDS")) is not None:
            kwargs["llm_max_rounds"] = int(value)
        return kwargs

    def aggregate_token_usage(self) -> dict[str, int | None]:
        """Delegate provider token accounting to the LLM seed stage (0 if none).

        The LFBO stage issues no provider requests, so the hybrid's provider
        usage is exactly the LLM stage's.
        """
        if self._llm_stage_search is not None:
            return self._llm_stage_search.aggregate_token_usage()
        return {
            "requests": 0,
            "input_tokens": None,
            "cached_input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
        }

    def aggregate_provider_identity(self) -> dict[str, str | None]:
        """Project the LLM stage's per-request identities into one identity."""
        if self._llm_stage_search is not None:
            return self._llm_stage_search.aggregate_provider_identity()
        return {"request_id": None, "response_id": None, "cache_state": None}

    @property
    def provider_replay_identities(self) -> tuple[tuple[str, str | None], ...]:
        """Return the LLM stage's ordered request/response replay identities."""
        if self._llm_stage_search is not None:
            return self._llm_stage_search.provider_replay_identities
        return ()

    def _make_llm_search(self) -> LLMGuidedSearch:
        """Construct the stage-1 guided search from llm_* settings."""
        return LLMGuidedSearch(
            self.kernel,
            self.args,
            finishing_rounds=0,
            provider=self.llm_provider,
            model=self.llm_model,
            configs_per_round=self.llm_configs_per_round,
            max_rounds=self.llm_max_rounds,
            initial_random_configs=self.llm_initial_random_configs,
            compile_timeout_s=self.llm_compile_timeout_s,
            api_base=self.llm_api_base,
            api_key=self.llm_api_key,
            request_timeout_s=self.llm_request_timeout_s,
            effort_level=self.llm_effort_level,
            fast_mode=self.llm_fast_mode,
        )

    def _second_stage_search_kwargs(
        self, *, seeded: bool, remaining: int | None = None
    ) -> dict[str, object]:
        """Build the stage-2 kwargs, forcing best-available seeding when supported."""
        kwargs = dict(self.second_stage_kwargs)
        # Clamp the stage-2 initial population to the attempts the LLM stage left
        # so the shared ceiling is never exceeded while building it: a capped
        # initial population refills to its target size and would otherwise
        # overrun the remaining budget (or raise CandidatePopulationUnderfilled).
        if remaining is not None:
            configured = kwargs.get("initial_population")
            if isinstance(configured, int) and configured > remaining:
                kwargs["initial_population"] = max(1, remaining)
        if not seeded:
            return kwargs

        if not self._second_stage_supports_best_available_handoff:
            self.log(
                f"Second-stage algorithm {self.second_stage_algorithm} "
                "does not support FROM_BEST_AVAILABLE initialization; "
                "the LLM seed may not influence the next stage."
            )
            return kwargs

        kwargs["initial_population_strategy"] = (
            InitialPopulationStrategy.FROM_BEST_AVAILABLE
        )
        kwargs["best_available_pad_random"] = self.best_available_pad_random
        return kwargs

    def _make_second_stage_search(
        self, *, seeded: bool, remaining: int | None = None
    ) -> BaseSearch:
        """Construct stage 2 and enable best-available seeding when supported."""
        factory = cast("Callable[..., BaseSearch]", self._second_stage_search_cls)
        return factory(
            self.kernel,
            self.args,
            **self._second_stage_search_kwargs(seeded=seeded, remaining=remaining),
        )

    def _inject_seed_into_second_stage(
        self,
        second_stage_search: BaseSearch,
        llm_seed_config: Config,
        llm_search: LLMGuidedSearch | None = None,
    ) -> None:
        """Pass the best LLM config into searches that expose the seed hook.

        For LFBO stage 2, also seed the surrogate's training set so LFBO
        learns from the LLM's exploration, not just the single best config.
        """
        if not self._second_stage_supports_best_available_handoff:
            return
        seeded_search = cast("PopulationBasedSearch", second_stage_search)
        seeded_search.set_best_available_seed_configs([llm_seed_config])

        from .surrogate_pattern_search import LFBOPatternSearch

        if llm_search is not None and isinstance(seeded_search, LFBOPatternSearch):
            results = llm_search._all_benchmark_results
            seeded_search.seed_training_data(results)
            self.log(
                f"Seeded LFBO surrogate with {len(results)} (config, perf) pairs "
                "from the LLM stage."
            )

    @staticmethod
    def _finite_perf(search: BaseSearch | None) -> float | None:
        """Return a search's best perf when finite, else None for reporting."""
        if search is None or not math.isfinite(search.best_perf_so_far):
            return None
        return search.best_perf_so_far

    def _run_llm_seed_stage(
        self,
        shared: SharedAttemptState,
    ) -> tuple[LLMGuidedSearch | None, Config | None, float]:
        """Run the optional stage-1 LLM search under the shared attempt state."""
        if self.llm_max_rounds <= 0:
            return None, None, 0.0

        self.log(
            "Hybrid stage 1/2: "
            f"LLMGuidedSearch for {self.llm_max_rounds} round(s) "
            f"with {self.llm_configs_per_round} configs/round"
        )
        llm_search = self._make_llm_search()
        # Share the run-level budget, cross-stage dedup set, and collector before
        # the LLM stage records any attempt.
        llm_search.adopt_shared_attempt_state(shared)
        self._llm_stage_search = llm_search
        llm_start = time.perf_counter()
        llm_seed_config = llm_search.autotune(skip_cache=True)
        llm_wall_time = time.perf_counter() - llm_start
        return llm_search, llm_seed_config, llm_wall_time

    def _run_second_stage(
        self,
        shared: SharedAttemptState,
        llm_seed_config: Config | None,
        llm_search: LLMGuidedSearch | None = None,
    ) -> tuple[BaseSearch | None, Config | None, float]:
        """Run stage 2 on the budget the LLM stage left, seeded from its best."""
        remaining = shared.budget.remaining()
        if remaining is not None and remaining <= 0:
            # The LLM stage consumed the whole shared ceiling; its config is final.
            self.log(
                "Hybrid stage 2/2: skipped -- the LLM stage consumed the full "
                f"{shared.budget.limit}-attempt budget."
            )
            return None, llm_seed_config, 0.0

        seeded = llm_seed_config is not None
        self.log(
            "Hybrid stage 2/2: "
            + (
                f"running {self.second_stage_algorithm} from best available seed"
                if seeded
                else f"running {self.second_stage_algorithm} without LLM seed"
            )
        )
        second_stage_search = self._make_second_stage_search(
            seeded=seeded, remaining=remaining
        )
        self._second_stage_search = second_stage_search
        # Share the same budget/dedup/collector so stage 2 continues -- rather
        # than restarts -- the run-level accounting and trajectory.
        second_stage_search.adopt_shared_attempt_state(shared)
        if llm_seed_config is not None:
            self._inject_seed_into_second_stage(
                second_stage_search, llm_seed_config, llm_search
            )
        second_stage_start = time.perf_counter()
        try:
            best_config = second_stage_search.autotune()
        except (CandidatePopulationUnderfilled, exc.NoConfigFound) as error:
            # The remaining budget/space could not seed a stage-2 population; keep
            # the LLM result rather than failing the whole hybrid run.
            self.log(
                f"Hybrid stage 2/2 could not run ({type(error).__name__}); "
                "keeping the LLM stage result."
            )
            return (
                second_stage_search,
                llm_seed_config,
                time.perf_counter() - second_stage_start,
            )
        second_stage_wall_time = time.perf_counter() - second_stage_start
        return second_stage_search, best_config, second_stage_wall_time

    def _select_best_overall(
        self,
        llm_search: LLMGuidedSearch | None,
        llm_seed_config: Config | None,
        second_stage_search: BaseSearch | None,
        second_stage_best_config: Config | None,
    ) -> tuple[float, Config | None]:
        """Pick the globally best (perf, config) across both stages.

        Cross-stage dedup can keep the LLM's best config out of the LFBO
        population, so the LFBO stage may end worse than the handoff incumbent;
        return the better of the two so the hybrid never regresses below its seed.
        """
        candidates: list[tuple[float, Config | None]] = []
        if llm_search is not None and math.isfinite(llm_search.best_perf_so_far):
            candidates.append((llm_search.best_perf_so_far, llm_seed_config))
        if second_stage_search is not None and math.isfinite(
            second_stage_search.best_perf_so_far
        ):
            candidates.append(
                (second_stage_search.best_perf_so_far, second_stage_best_config)
            )
        if candidates:
            return min(candidates, key=operator.itemgetter(0))
        return math.inf, second_stage_best_config or llm_seed_config

    def _finalize_stage_metrics(
        self,
        shared: SharedAttemptState,
        llm_attempts: int,
        llm_search: LLMGuidedSearch | None,
        llm_seed_config: Config | None,
        llm_wall_time: float,
        second_stage_search: BaseSearch | None,
        second_stage_best_config: Config | None,
        second_stage_wall_time: float,
    ) -> None:
        """Merge per-stage timing/metrics into the hybrid summary and assert fairness."""

        llm_metrics = llm_search._autotune_metrics if llm_search else None
        second_stage_metrics = (
            second_stage_search._autotune_metrics if second_stage_search else None
        )
        second_stage_tested = (
            second_stage_metrics.num_configs_tested if second_stage_metrics else 0
        )
        provider_requests = llm_search._provider_requests if llm_search else 0
        handoff_perf = self._finite_perf(llm_search)
        final_perf, final_config = self._select_best_overall(
            llm_search, llm_seed_config, second_stage_search, second_stage_best_config
        )
        counts = shared.budget.by_category()

        total_attempts = shared.budget.spent()
        self.hybrid_stage_breakdown = {
            "used_llm_seed": llm_search is not None,
            "candidate_attempt_limit": shared.budget.limit,
            "total_attempts": total_attempts,
            "llm_attempts": llm_attempts,
            "lfbo_attempts": total_attempts - llm_attempts,
            "attempts_by_category": {
                category.value: count for category, count in counts.items()
            },
            "provider_requests": provider_requests,
            "provider_tokens": (
                llm_search.aggregate_token_usage() if llm_search else None
            ),
            "llm_seed_perf_ms": handoff_perf,
            "best_perf_at_handoff_ms": handoff_perf,
            "llm_seed_time_s": llm_wall_time,
            "llm_seed_configs_tested": (
                llm_metrics.num_configs_tested if llm_metrics else 0
            ),
            "llm_seed_config": (
                dict(llm_seed_config) if llm_seed_config is not None else None
            ),
            "second_stage_algorithm": self.second_stage_algorithm,
            "second_stage_ran": second_stage_search is not None,
            "second_stage_perf_ms": self._finite_perf(second_stage_search),
            "second_stage_time_s": second_stage_wall_time,
            "second_stage_configs_tested": second_stage_tested,
            "final_perf_ms": final_perf if math.isfinite(final_perf) else None,
        }

        # Aggregate metrics from both stages
        for field in _AGGREGATED_METRIC_FIELDS:
            setattr(
                self._autotune_metrics,
                field,
                (getattr(llm_metrics, field) if llm_metrics else 0)
                + (getattr(second_stage_metrics, field) if second_stage_metrics else 0),
            )

        self.best_perf_so_far = final_perf
        self._best_overall_config = final_config

        # Fairness invariants: one shared ceiling and at most one provider call.
        limit = shared.budget.limit
        spent = shared.budget.spent()
        if limit is not None and spent > limit:
            raise AssertionError(
                f"hybrid attempts {spent} exceed the shared limit {limit}"
            )
        if provider_requests > 1:
            raise AssertionError(
                f"hybrid issued {provider_requests} provider requests; expected <= 1"
            )

    def _autotune(self) -> Config:
        """Run the optional LLM seed stage, then the configured second stage."""
        self.log(
            f"Starting {type(self).__name__} with "
            f"second_stage_algorithm={self.second_stage_algorithm}, "
            f"llm_max_rounds={self.llm_max_rounds}, "
            f"llm_configs_per_round={self.llm_configs_per_round}, "
            f"best_available_pad_random={self.best_available_pad_random}"
        )

        # One run-level attempt state shared by both stages: a single ceiling,
        # one cross-stage dedup set, and (when the experiment attached one to
        # this hybrid) one continuous instrumentation collector so the LFBO
        # trajectory continues from the LLM stage rather than restarting at zero.
        shared = SharedAttemptState(
            AttemptBudget(self.settings.autotune_candidate_attempt_limit),
            collector=self._attempt_instrumentation,
        )
        self._shared_attempt_state = shared
        # Expose the shared budget on the hybrid so unified event emission reports
        # run-level totals rather than the hybrid's unused per-search budget.
        self.set_candidate_attempt_budget(shared.budget)

        # Stage 1: run the LLM seed search when enabled and keep its best config.
        llm_search, llm_seed_config, llm_wall_time = self._run_llm_seed_stage(shared)
        # Attempts charged by stage 1 so the breakdown can partition the shared total.
        llm_attempts = shared.budget.spent()
        # Stage 2: run the follow-up search on the remaining budget, seeded when
        # stage 1 found a config.
        second_stage_search, second_stage_best_config, second_stage_wall_time = (
            self._run_second_stage(shared, llm_seed_config, llm_search)
        )

        self._finalize_stage_metrics(
            shared,
            llm_attempts,
            llm_search,
            llm_seed_config,
            llm_wall_time,
            second_stage_search,
            second_stage_best_config,
            second_stage_wall_time,
        )
        if self._best_overall_config is not None:
            return self._best_overall_config
        if second_stage_best_config is not None:
            return second_stage_best_config
        if llm_seed_config is not None:
            return llm_seed_config
        raise exc.NoConfigFound


class LLMSeededLFBOTreeSearch(LLMSeededSearch):
    """Convenience wrapper for the common LLM-seeded LFBO tree search pipeline.

    LFBO-specific stage-2 settings should be passed through ``second_stage_kwargs``.
    """

    allow_second_stage_env_override = False

    @classmethod
    def get_kwargs_from_profile(
        cls, profile: AutotuneEffortProfile, settings: Settings
    ) -> dict[str, object]:
        """Drop the explicit stage-2 algorithm knob from the LFBO convenience API."""
        kwargs = super().get_kwargs_from_profile(profile, settings)
        kwargs.pop("second_stage_algorithm", None)
        return kwargs

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        *,
        second_stage_kwargs: dict[str, object] | None = None,
        best_available_pad_random: bool = False,
        llm_provider: str | None = None,
        llm_model: str = QUICK_LLM_SEARCH_DEFAULTS.model,
        llm_configs_per_round: int = QUICK_LLM_SEARCH_DEFAULTS.configs_per_round,
        llm_max_rounds: int = QUICK_LLM_SEARCH_DEFAULTS.max_rounds,
        llm_initial_random_configs: int = QUICK_LLM_SEARCH_DEFAULTS.initial_random_configs,
        llm_compile_timeout_s: int | None = QUICK_LLM_SEARCH_DEFAULTS.compile_timeout_s,
        llm_api_base: str | None = None,
        llm_api_key: str | None = None,
        llm_request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        llm_effort_level: str | None = None,
        llm_fast_mode: bool = False,
    ) -> None:
        super().__init__(
            kernel,
            args,
            second_stage_algorithm="LFBOTreeSearch",
            second_stage_kwargs=second_stage_kwargs,
            best_available_pad_random=best_available_pad_random,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_configs_per_round=llm_configs_per_round,
            llm_max_rounds=llm_max_rounds,
            llm_initial_random_configs=llm_initial_random_configs,
            llm_compile_timeout_s=llm_compile_timeout_s,
            llm_api_base=llm_api_base,
            llm_api_key=llm_api_key,
            llm_request_timeout_s=llm_request_timeout_s,
            llm_effort_level=llm_effort_level,
            llm_fast_mode=llm_fast_mode,
        )
