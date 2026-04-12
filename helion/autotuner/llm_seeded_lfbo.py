"""Seed a second-stage autotuner with configs from an LLM search pass."""

from __future__ import annotations

import inspect
import math
import os
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from .base_search import BaseSearch
from .effort_profile import PATTERN_SEARCH_DEFAULTS
from .effort_profile import QUICK_LLM_SEARCH_DEFAULTS
from .llm.transport import DEFAULT_REQUEST_TIMEOUT_S
from .llm_search import LLMGuidedSearch
from .llm_search import guided_search_kwargs_from_config
from .pattern_search import InitialPopulationStrategy

if TYPE_CHECKING:
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


def _parse_env_bool(value: str) -> bool:
    """Parse the small env-var bool dialect used by the hybrid overrides."""
    return value.strip().lower() not in {"", "0", "false"}


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


def _supports_init_parameter(search_cls: type[BaseSearch], name: str) -> bool:
    """Check whether a second-stage search accepts a particular kwarg."""
    return name in inspect.signature(search_cls.__init__).parameters


class LLMSeededSearch(BaseSearch):
    """
    Generic hybrid autotuner that seeds a second-stage search with LLM proposals.

    The algorithm runs in two stages:
    1. Run ``LLMGuidedSearch`` for ``llm_max_rounds`` rounds and capture its best
       config in memory.
    2. Run the configured second-stage search algorithm. If the algorithm
       supports ``initial_population_strategy``, it is switched to
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
        _resolve_second_stage_algorithm(self.second_stage_algorithm)
        self.second_stage_kwargs = dict(second_stage_kwargs or {})
        self.best_available_pad_random = best_available_pad_random

        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_configs_per_round = llm_configs_per_round
        self.llm_max_rounds = llm_max_rounds
        self.llm_initial_random_configs = llm_initial_random_configs
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
        if (
            value := os.environ.get("HELION_HYBRID_BEST_AVAILABLE_PAD_RANDOM")
        ) is not None:
            kwargs["best_available_pad_random"] = _parse_env_bool(value)
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
            api_base=self.llm_api_base,
            api_key=self.llm_api_key,
            request_timeout_s=self.llm_request_timeout_s,
        )

    def _make_second_stage_search(self, *, seeded: bool) -> BaseSearch:
        """Construct stage 2 and enable best-available seeding when supported."""
        second_stage_cls = _resolve_second_stage_algorithm(self.second_stage_algorithm)
        kwargs = dict(self.second_stage_kwargs)

        if seeded:
            if _supports_init_parameter(
                second_stage_cls, "initial_population_strategy"
            ):
                kwargs["initial_population_strategy"] = (
                    InitialPopulationStrategy.FROM_BEST_AVAILABLE
                )
                if _supports_init_parameter(
                    second_stage_cls, "best_available_pad_random"
                ):
                    kwargs["best_available_pad_random"] = self.best_available_pad_random
            else:
                self.log(
                    f"Second-stage algorithm {self.second_stage_algorithm} "
                    "does not support FROM_BEST_AVAILABLE initialization; "
                    "the LLM seed may not influence the next stage."
                )

        return cast(
            "BaseSearch",
            cast("Any", second_stage_cls)(self.kernel, self.args, **kwargs),
        )

    def _inject_seed_into_second_stage(
        self,
        second_stage_search: BaseSearch,
        llm_seed_config: Config,
    ) -> None:
        """Pass the best LLM config into searches that expose the seed hook."""
        setter = getattr(second_stage_search, "set_best_available_seed_configs", None)
        if setter is None:
            return
        setter([llm_seed_config])

    def _finalize_stage_metrics(
        self,
        llm_search: LLMGuidedSearch | None,
        llm_seed_config: Config | None,
        llm_wall_time: float,
        second_stage_search: BaseSearch,
        second_stage_wall_time: float,
    ) -> None:
        """Merge per-stage timing and autotune metrics into the hybrid summary."""

        def _finite_perf(search: BaseSearch | None) -> float | None:
            if search is None or not math.isfinite(search.best_perf_so_far):
                return None
            return search.best_perf_so_far

        llm_metrics = llm_search._autotune_metrics if llm_search else None
        second_stage_metrics = second_stage_search._autotune_metrics
        second_stage_tested = second_stage_metrics.num_configs_tested

        self.hybrid_stage_breakdown = {
            "used_llm_seed": llm_search is not None,
            "llm_seed_perf_ms": _finite_perf(llm_search),
            "llm_seed_time_s": llm_wall_time,
            "llm_seed_configs_tested": (
                llm_metrics.num_configs_tested if llm_metrics else 0
            ),
            "llm_seed_config": (
                dict(llm_seed_config) if llm_seed_config is not None else None
            ),
            "second_stage_algorithm": self.second_stage_algorithm,
            "second_stage_perf_ms": _finite_perf(second_stage_search),
            "second_stage_time_s": second_stage_wall_time,
            "second_stage_configs_tested": second_stage_tested,
        }
        if self.second_stage_algorithm == "LFBOTreeSearch":
            self.hybrid_stage_breakdown.update(
                {
                    "lfbo_stage_perf_ms": _finite_perf(second_stage_search),
                    "lfbo_stage_time_s": second_stage_wall_time,
                    "lfbo_stage_configs_tested": second_stage_tested,
                }
            )

        # Aggregate metrics from both stages
        for field in (
            "num_configs_tested",
            "num_compile_failures",
            "num_accuracy_failures",
            "num_generations",
        ):
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

        # Stage 1: LLM seed search
        llm_search: LLMGuidedSearch | None = None
        llm_seed_config: Config | None = None
        llm_wall_time = 0.0

        if self.llm_max_rounds > 0:
            self.log(
                "Hybrid stage 1/2: "
                f"LLMGuidedSearch for {self.llm_max_rounds} round(s) "
                f"with {self.llm_configs_per_round} configs/round"
            )
            llm_search = self._make_llm_search()
            llm_start = time.perf_counter()
            llm_seed_config = llm_search.autotune(skip_cache=True)
            llm_wall_time = time.perf_counter() - llm_start

        # Stage 2: second-stage search (optionally seeded)
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

        self._finalize_stage_metrics(
            llm_search,
            llm_seed_config,
            llm_wall_time,
            second_stage_search,
            second_stage_wall_time,
        )
        return best_config


class LLMSeededLFBOTreeSearch(LLMSeededSearch):
    """Convenience wrapper for the common LLM-seeded LFBO tree search pipeline."""

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
        num_neighbors: int = 200,
        frac_selected: float = 0.10,
        radius: int = 2,
        initial_population: int = PATTERN_SEARCH_DEFAULTS.initial_population,
        copies: int = PATTERN_SEARCH_DEFAULTS.copies,
        max_generations: int = PATTERN_SEARCH_DEFAULTS.max_generations,
        min_improvement_delta: float = 0.001,
        quantile: float = 0.1,
        patience: int = 1,
        similarity_penalty: float = 1.0,
        initial_population_strategy: InitialPopulationStrategy | None = None,
        best_available_pad_random: bool = False,
        finishing_rounds: int = 0,
        compile_timeout_lower_bound: float = PATTERN_SEARCH_DEFAULTS.compile_timeout_lower_bound,
        compile_timeout_quantile: float = PATTERN_SEARCH_DEFAULTS.compile_timeout_quantile,
        llm_provider: str | None = None,
        llm_model: str = QUICK_LLM_SEARCH_DEFAULTS.model,
        llm_configs_per_round: int = QUICK_LLM_SEARCH_DEFAULTS.configs_per_round,
        llm_max_rounds: int = QUICK_LLM_SEARCH_DEFAULTS.max_rounds,
        llm_initial_random_configs: int = QUICK_LLM_SEARCH_DEFAULTS.initial_random_configs,
        llm_api_base: str | None = None,
        llm_api_key: str | None = None,
        llm_request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> None:
        # Build LFBO second-stage kwargs from individual params or passthrough
        computed_second_stage_kwargs: dict[str, object]
        if second_stage_kwargs is not None:
            computed_second_stage_kwargs = dict(second_stage_kwargs)
        else:
            computed_second_stage_kwargs = {
                "num_neighbors": num_neighbors,
                "frac_selected": frac_selected,
                "radius": radius,
                "initial_population": initial_population,
                "copies": copies,
                "max_generations": max_generations,
                "min_improvement_delta": min_improvement_delta,
                "quantile": quantile,
                "patience": patience,
                "similarity_penalty": similarity_penalty,
                "initial_population_strategy": initial_population_strategy,
                "finishing_rounds": finishing_rounds,
                "compile_timeout_lower_bound": compile_timeout_lower_bound,
                "compile_timeout_quantile": compile_timeout_quantile,
            }

        super().__init__(
            kernel,
            args,
            second_stage_algorithm="LFBOTreeSearch",
            second_stage_kwargs=computed_second_stage_kwargs,
            best_available_pad_random=best_available_pad_random,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_configs_per_round=llm_configs_per_round,
            llm_max_rounds=llm_max_rounds,
            llm_initial_random_configs=llm_initial_random_configs,
            llm_api_base=llm_api_base,
            llm_api_key=llm_api_key,
            llm_request_timeout_s=llm_request_timeout_s,
        )
