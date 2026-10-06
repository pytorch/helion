"""Native coordinate proposals scored exclusively by a whole-pipeline objective."""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
from dataclasses import field
import functools
import inspect
import math
import os
import time
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

from .. import exc
from .base_search import BaseSearch
from .base_search import normalize_autotune_seed_configs
from .benchmark_provider import PipelineBenchmarkProvider
from .benchmark_provider import _not_an_isolated_stage
from .effort_profile import get_effort_profile
from .llm_search import LLMGuidedSearch
from .llm_seeded_lfbo import LLMSeededLFBOTreeSearch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from typing import Any

    from ..runtime.config import Config
    from ..runtime.pipeline import PipelineStage
    from ..runtime.settings import Settings
    from .base_search import _AutotunableKernel


@dataclass(frozen=True)
class CoordinateSearchResult:
    """Search statistics; the pipeline controller retains measured candidates."""

    method: str
    elapsed_seconds: float
    evaluations: int
    budget_exhausted: bool
    hybrid_metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "elapsed_seconds": self.elapsed_seconds,
            "evaluations": self.evaluations,
            "budget_exhausted": self.budget_exhausted,
            "hybrid_metadata": copy.deepcopy(self.hybrid_metadata),
        }


class _CoordinateBudgetExhausted(BaseException):
    """Unwind native search cleanup even when candidate errors are caught."""


def _unique_configs(configs: Sequence[Config]) -> tuple[Config, ...]:
    return tuple(dict.fromkeys(copy.deepcopy(config) for config in configs))


@dataclass
class _Objective:
    evaluate: Callable[[Config], float]
    max_evaluations: int
    evaluations: int = 0
    exhausted: bool = False
    measurements_by_phase: dict[str, int] = field(default_factory=dict)

    def measure(self, config: Config, *, phase: str = "incumbent") -> float:
        if self.evaluations >= self.max_evaluations:
            self.exhausted = True
            raise _CoordinateBudgetExhausted
        self.evaluations += 1
        self.measurements_by_phase[phase] = self.measurements_by_phase.get(phase, 0) + 1
        # Do not let the callback mutate native search proposals.
        value = self.evaluate(copy.deepcopy(config))
        if isinstance(value, bool):
            raise TypeError("The pipeline objective must return a latency in ms")
        perf = float(value)
        if not math.isfinite(perf) or perf <= 0:
            perf = math.inf
        return perf


class _KernelView:
    """Preserve stage code/spec while isolating temporary search settings."""

    def __init__(self, bound: _AutotunableKernel, settings: Settings) -> None:
        self._bound = bound
        self.settings = settings

    def __getattr__(self, name: str) -> object:
        return getattr(self._bound, name)

    compile_config = staticmethod(_not_an_isolated_stage)
    bench_compile_config = staticmethod(_not_an_isolated_stage)


def _attach_objective(search: BaseSearch, objective: _Objective) -> None:
    search._benchmark_provider_cls = functools.partial(
        PipelineBenchmarkProvider,
        objective=objective,
        search_name=type(search).__name__,
    )
    # A stage cache describes a different objective. Explicit seeds remain
    # available; do not scan local/remote isolated-stage warm-start caches.
    search._find_similar_cached_configs = lambda max_configs, *, config_spec_hash=None: []


class _CoordinateHybrid(LLMSeededLFBOTreeSearch):
    objective: _Objective

    def _make_llm_search(self) -> LLMGuidedSearch:
        search = super()._make_llm_search()
        _attach_objective(search, self.objective)
        return search

    def _make_second_stage_search(self, *, seeded: bool) -> BaseSearch:
        search = super()._make_second_stage_search(seeded=seeded)
        _attach_objective(search, self.objective)
        return search


def _resolve_algorithm(algorithm: str | None) -> str:
    """Use the normal default/environment selection for pipeline proposals."""
    if algorithm is None:
        algorithm = os.environ.get("HELION_AUTOTUNER") or "LFBOTreeSearch"
    supported = (
        "finite",
        "LFBOTreeSearch",
        "LFBOPatternSearch",
        "LLMGuidedSearch",
        "LLMSeededLFBOTreeSearch",
    )
    if algorithm not in supported:
        raise ValueError(
            f"Pipeline search supports {', '.join(supported)}; got {algorithm!r}"
        )
    return algorithm


def search_coordinate(
    stage: PipelineStage,
    evaluate: Callable[[Config], float],
    *,
    seed_configs: Sequence[Config],
    max_evaluations: int,
    algorithm: str | None = None,
    effort: str = "full",
) -> CoordinateSearchResult:
    """Propose stage configs using whole-operation scores for every search step.

    ``evaluate`` must realize a coherent pipeline bundle, validate every input,
    and return its positive aggregate latency in milliseconds (or infinity for
    a rejected candidate). It owns graph/eager timing and repetition policy.
    Every call, including rechecks, consumes one evaluation from the shared
    coordinate budget. No candidate compiles or benchmarks outside this callback.

    The incumbent is evaluated first. ``finite`` measures only the incumbent
    and distinct explicit/settings seeds; it never invokes an LLM or LFBO.
    Native algorithms use the normal ConfigSpec, effort profile and environment
    controls. The default follows HELION_AUTOTUNER, or LFBOTreeSearch when unset.
    Effort ``none`` measures only the incumbent.
    """
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int):
        raise TypeError("max_evaluations must be an integer")
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    algorithm = _resolve_algorithm(algorithm)
    if effort not in {"none", "quick", "full"}:
        raise ValueError(f"Unknown autotune effort: {effort}")
    started = time.perf_counter()
    settings = copy.copy(stage.bound.settings)
    seeds = _unique_configs(
        (stage.config, *seed_configs, *normalize_autotune_seed_configs(settings))
    )
    settings.autotune_seed_configs = seeds
    settings.autotune_effort = cast("Literal['none', 'quick', 'full']", effort)
    objective = _Objective(evaluate, max_evaluations)
    search: BaseSearch | None = None
    try:
        objective.measure(seeds[0])
        if effort != "none" and algorithm == "finite":
            for config in seeds[1:]:
                objective.measure(config, phase="finite")
        elif effort != "none":
            from . import search_algorithms

            view = cast("_AutotunableKernel", _KernelView(stage.bound, settings))
            native_cls = search_algorithms[algorithm]
            parameters = inspect.signature(native_cls.__init__).parameters
            kwargs = native_cls.get_kwargs_from_profile(
                get_effort_profile(settings.autotune_effort), settings
            )
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}
            if algorithm == "LLMSeededLFBOTreeSearch":
                search = _CoordinateHybrid(
                    view, stage.args, **cast("dict[str, Any]", kwargs)
                )
                search.objective = objective
            else:
                search = native_cls(view, stage.args, **cast("dict[str, Any]", kwargs))
            _attach_objective(search, objective)
            # The controller retains the incumbent and all viable candidates.
            with contextlib.suppress(exc.NoConfigFound):
                search.autotune(skip_cache=True)
    except _CoordinateBudgetExhausted:
        pass
    metadata: dict[str, object] = {
        "measurements_by_phase": dict(objective.measurements_by_phase),
    }
    if isinstance(search, LLMSeededLFBOTreeSearch):
        metadata["hybrid_stage_breakdown"] = search.hybrid_stage_breakdown
        metadata["llm_model"] = search.llm_model
        metadata["llm_provider"] = search.llm_provider
        metadata["llm_effort_level"] = search.llm_effort_level
        metadata["llm_fast_mode"] = search.llm_fast_mode
    elif isinstance(search, LLMGuidedSearch):
        metadata["llm_model"] = search.model
        metadata["llm_provider"] = search.provider
        metadata["llm_effort_level"] = search.effort_level
        metadata["llm_fast_mode"] = search.fast_mode
    return CoordinateSearchResult(
        method=algorithm,
        elapsed_seconds=time.perf_counter() - started,
        evaluations=objective.evaluations,
        budget_exhausted=objective.exhausted,
        hybrid_metadata=metadata,
    )
