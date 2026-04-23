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
import os
import time
from typing import TYPE_CHECKING
from typing import cast

from .base_search import BaseSearch
from .base_search import PopulationBasedSearch
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

        self.hybrid_stage_breakdown = None

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
        )

    def _second_stage_search_kwargs(self, *, seeded: bool) -> dict[str, object]:
        """Build the stage-2 kwargs, forcing best-available seeding when supported."""
        kwargs = dict(self.second_stage_kwargs)
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

    def _make_second_stage_search(self, *, seeded: bool) -> BaseSearch:
        """Construct stage 2 and enable best-available seeding when supported."""
        factory = cast("Callable[..., BaseSearch]", self._second_stage_search_cls)
        return factory(
            self.kernel,
            self.args,
            **self._second_stage_search_kwargs(seeded=seeded),
        )

    def _inject_seed_into_second_stage(
        self,
        second_stage_search: BaseSearch,
        llm_seed_config: Config,
    ) -> None:
        """Pass the best LLM config into searches that expose the seed hook."""
        if not self._second_stage_supports_best_available_handoff:
            return
        seeded_search = cast("PopulationBasedSearch", second_stage_search)
        seeded_search.set_best_available_seed_configs([llm_seed_config])

    @staticmethod
    def _finite_perf(search: BaseSearch | None) -> float | None:
        """Return a search's best perf when finite, else None for reporting."""
        if search is None or not math.isfinite(search.best_perf_so_far):
            return None
        return search.best_perf_so_far

    def _run_llm_seed_stage(
        self,
    ) -> tuple[LLMGuidedSearch | None, Config | None, float]:
        """Run the optional stage-1 LLM search and return its best config."""
        if self.llm_max_rounds <= 0:
            return None, None, 0.0

        self.log(
            "Hybrid stage 1/2: "
            f"LLMGuidedSearch for {self.llm_max_rounds} round(s) "
            f"with {self.llm_configs_per_round} configs/round"
        )
        llm_search = self._make_llm_search()
        llm_start = time.perf_counter()
        llm_seed_config = llm_search.autotune(skip_cache=True)
        llm_wall_time = time.perf_counter() - llm_start
        return llm_search, llm_seed_config, llm_wall_time

    def _run_second_stage(
        self,
        llm_seed_config: Config | None,
    ) -> tuple[BaseSearch, Config, float]:
        """Run stage 2, optionally seeded from the stage-1 best config."""
        seeded = llm_seed_config is not None
        self.log(
            "Hybrid stage 2/2: "
            + (
                f"running {self.second_stage_algorithm} from best available seed"
                if seeded
                else f"running {self.second_stage_algorithm} without LLM seed"
            )
        )
        second_stage_search = self._make_second_stage_search(seeded=seeded)
        if llm_seed_config is not None:
            self._inject_seed_into_second_stage(second_stage_search, llm_seed_config)
        second_stage_start = time.perf_counter()
        best_config = second_stage_search.autotune()
        second_stage_wall_time = time.perf_counter() - second_stage_start
        return second_stage_search, best_config, second_stage_wall_time

    def _finalize_stage_metrics(
        self,
        llm_search: LLMGuidedSearch | None,
        llm_seed_config: Config | None,
        llm_wall_time: float,
        second_stage_search: BaseSearch,
        second_stage_wall_time: float,
    ) -> None:
        """Merge per-stage timing and autotune metrics into the hybrid summary."""

        llm_metrics = llm_search._autotune_metrics if llm_search else None
        second_stage_metrics = second_stage_search._autotune_metrics
        second_stage_tested = second_stage_metrics.num_configs_tested

        self.hybrid_stage_breakdown = {
            "used_llm_seed": llm_search is not None,
            "llm_seed_perf_ms": self._finite_perf(llm_search),
            "llm_seed_time_s": llm_wall_time,
            "llm_seed_configs_tested": (
                llm_metrics.num_configs_tested if llm_metrics else 0
            ),
            "llm_seed_config": (
                dict(llm_seed_config) if llm_seed_config is not None else None
            ),
            "second_stage_algorithm": self.second_stage_algorithm,
            "second_stage_perf_ms": self._finite_perf(second_stage_search),
            "second_stage_time_s": second_stage_wall_time,
            "second_stage_configs_tested": second_stage_tested,
        }

        # Aggregate metrics from both stages
        for field in _AGGREGATED_METRIC_FIELDS:
            setattr(
                self._autotune_metrics,
                field,
                (getattr(llm_metrics, field) if llm_metrics else 0)
                + getattr(second_stage_metrics, field),
            )

        candidate_best = [
            stage.best_perf_so_far
            for stage in (llm_search, second_stage_search)
            if stage is not None and math.isfinite(stage.best_perf_so_far)
        ]
        self.best_perf_so_far = min(candidate_best) if candidate_best else math.inf

    def _autotune(self) -> Config:
        """Run the optional LLM seed stage, then the configured second stage."""
        self.log(
            f"Starting {type(self).__name__} with "
            f"second_stage_algorithm={self.second_stage_algorithm}, "
            f"llm_max_rounds={self.llm_max_rounds}, "
            f"llm_configs_per_round={self.llm_configs_per_round}, "
            f"best_available_pad_random={self.best_available_pad_random}"
        )

        # Stage 1: run the LLM seed search when enabled and keep its best config.
        llm_search, llm_seed_config, llm_wall_time = self._run_llm_seed_stage()
        # Stage 2: run the configured follow-up search, seeded when stage 1 found a config.
        second_stage_search, best_config, second_stage_wall_time = (
            self._run_second_stage(llm_seed_config)
        )

        self._finalize_stage_metrics(
            llm_search,
            llm_seed_config,
            llm_wall_time,
            second_stage_search,
            second_stage_wall_time,
        )
        return best_config


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
        )
