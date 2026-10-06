"""Native coordinate proposals scored exclusively by a whole-pipeline objective."""

from __future__ import annotations

from contextlib import ExitStack
import copy
from dataclasses import dataclass
from dataclasses import field
import functools
import math
import operator
import statistics
import time
from types import MethodType
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

from ..runtime.config import Config
from .base_search import BaseSearch
from .base_search import PopulationBasedSearch
from .base_search import normalize_autotune_seed_configs
from .benchmark_provider import BenchmarkProvider
from .benchmark_provider import BenchmarkResult
from .benchmarking import MirroredBenchmarkTrace
from .effort_profile import get_effort_profile
from .llm_seeded_lfbo import LLMSeededLFBOTreeSearch
from .logger import AutotuneLogEntry

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..runtime.pipeline import PipelineStage
    from ..runtime.settings import Settings
    from .base_search import PopulationMember
    from .base_search import _AutotunableKernel
    from .config_spec import ConfigSpec
    from .llm_search import LLMGuidedSearch
    from .logger import AutotuningLogger
    from .metrics import AutotuneMetrics


@dataclass(frozen=True)
class CoordinateSearchResult:
    """Distinct valid candidates, ordered by their latest pipeline score."""

    configs: tuple[Config, ...]
    scores: tuple[float, ...]
    method: str
    elapsed_seconds: float
    evaluations: int
    budget_exhausted: bool
    hybrid_metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "configs": [copy.deepcopy(dict(config)) for config in self.configs],
            "scores": list(self.scores),
            "method": self.method,
            "elapsed_seconds": self.elapsed_seconds,
            "evaluations": self.evaluations,
            "budget_exhausted": self.budget_exhausted,
            "hybrid_metadata": copy.deepcopy(self.hybrid_metadata),
        }


class _CoordinateBudgetExhausted(BaseException):
    """Unwind native search cleanup even when candidate errors are caught."""


def _not_an_isolated_stage(*args: object, **kwargs: object) -> object:
    raise RuntimeError("Pipeline proposals cannot execute an isolated stage")


def _unique_configs(configs: Sequence[Config]) -> tuple[Config, ...]:
    return tuple(dict.fromkeys(Config(**copy.deepcopy(dict(c))) for c in configs))


@dataclass
class _Objective:
    evaluate: Callable[[Config], float]
    max_evaluations: int
    evaluations: int = 0
    exhausted: bool = False
    latest: dict[Config, float] = field(default_factory=dict)
    measurements_by_phase: dict[str, int] = field(default_factory=dict)
    successful_llm_proposals: int = 0

    def measure(self, config: Config, *, phase: str = "incumbent") -> float:
        if self.evaluations >= self.max_evaluations:
            self.exhausted = True
            raise _CoordinateBudgetExhausted
        self.evaluations += 1
        self.measurements_by_phase[phase] = self.measurements_by_phase.get(phase, 0) + 1
        # Config is mutable despite being hashable. Keep the ledger independent
        # of callers and callbacks, which must not mutate search proposals.
        saved = Config(**copy.deepcopy(dict(config)))
        value = self.evaluate(Config(**copy.deepcopy(dict(config))))
        if isinstance(value, bool):
            raise TypeError("The pipeline objective must return a latency in ms")
        perf = float(value)
        if not math.isfinite(perf) or perf <= 0:
            perf = math.inf
        self.latest[saved] = perf
        if math.isfinite(perf) and phase.startswith("LLMGuidedSearch:Round "):
            suffix = phase.removeprefix("LLMGuidedSearch:Round ")
            if suffix == "0 LLM" or suffix.isdecimal():
                self.successful_llm_proposals += 1
        return perf


class _PipelineBenchmarkProvider(BenchmarkProvider):
    """No compile or timer here: the callback owns realization and measurement."""

    def __init__(
        self,
        kernel: _AutotunableKernel,
        settings: Settings,
        config_spec: ConfigSpec,
        args: Sequence[object],
        log: AutotuningLogger,
        autotune_metrics: AutotuneMetrics,
        *,
        objective: _Objective,
        search_name: str = "coordinate",
    ) -> None:
        self.objective = objective
        self.log = log
        self.metrics = autotune_metrics
        self.search_name = search_name
        self.mutated_arg_indices = ()

    def setup(self) -> None:
        pass

    def cleanup(self) -> None:
        # BaseSearch installs a bound method here. Reset it so a finished
        # provider does not retain its owning search and tensor arguments.
        vars(self).pop("budget_exceeded_fn", None)

    def benchmark(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[BenchmarkResult]:
        results = []
        for config in configs:
            perf = self.objective.measure(config, phase=f"{self.search_name}:{desc}")
            status: Literal["ok", "error"] = "ok" if math.isfinite(perf) else "error"
            self.metrics.num_configs_tested += 1
            if status == "ok":
                self.metrics.num_successful_candidate_measurements += 1
            self.log.record_autotune_entry(
                AutotuneLogEntry(
                    self.metrics.num_generations,
                    status,
                    perf if status == "ok" else None,
                    None,
                    f"pipeline-{self.objective.evaluations}",
                    config,
                )
            )
            results.append(
                BenchmarkResult(config, _not_an_isolated_stage, perf, status, None)
            )
        return results


class _KernelView:
    """Preserve stage code/spec while isolating temporary search settings."""

    def __init__(self, bound: _AutotunableKernel, settings: Settings) -> None:
        self._bound = bound
        self.settings = settings

    def __getattr__(self, name: str) -> object:
        return getattr(self._bound, name)

    compile_config = staticmethod(_not_an_isolated_stage)
    bench_compile_config = staticmethod(_not_an_isolated_stage)


def _rebenchmark(
    search: PopulationBasedSearch,
    members: list[PopulationMember],
    *,
    desc: str = "Rebenchmarking",
    target_ms: float = 200.0,
    use_isolated: bool = True,
    confirm_suspicious: bool = True,
    use_interleaved: bool = True,
    candidate_private_args: bool = False,
) -> None:
    results = search.benchmark_provider.benchmark(
        [member.config for member in members], desc=desc
    )
    for member, result in zip(members, results, strict=True):
        member.status = result.status
        member.fn = result.fn
    search._apply_rebenchmark_timings(members, [result.perf for result in results])


def _mirrored_rebenchmark(
    search: PopulationBasedSearch,
    members: list[PopulationMember],
    *,
    desc: str,
    target_ms: float = 200.0,
) -> MirroredBenchmarkTrace:
    # Each callback already performs the requested full-pipeline timer protocol.
    # Mirror candidate order, not host callback execution or individual stages.
    orders = [list(range(len(members))), list(reversed(range(len(members))))]
    samples: list[list[float]] = [[] for _ in members]
    elapsed: list[list[float]] = []
    for order in orders:
        results = search.benchmark_provider.benchmark(
            [members[index].config for index in order], desc=desc
        )
        elapsed.append([result.perf for result in results])
        for index, result in zip(order, results, strict=True):
            samples[index].append(result.perf)
            members[index].status = result.status
            members[index].fn = result.fn
    medians = [statistics.median(values) for values in samples]
    search._apply_rebenchmark_timings(members, medians)
    provider = cast("_PipelineBenchmarkProvider", search.benchmark_provider)
    for member, perf in zip(members, medians, strict=True):
        if not math.isfinite(perf):
            member.status = "error"
        provider.objective.latest[member.config] = perf
    return MirroredBenchmarkTrace(orders, elapsed, medians, sweep_count=2)


_MISSING_OVERRIDE = object()


def _restore_override(target: object, name: str, previous: object) -> None:
    if previous is _MISSING_OVERRIDE:
        vars(target).pop(name, None)
    else:
        setattr(target, name, previous)


def _override(stack: ExitStack, target: object, name: str, value: object) -> None:
    previous = vars(target).get(name, _MISSING_OVERRIDE)
    setattr(target, name, value)
    stack.callback(_restore_override, target, name, previous)


def _release_budget_hook(search: BaseSearch) -> None:
    provider = vars(search).get("benchmark_provider")
    if isinstance(provider, _PipelineBenchmarkProvider):
        vars(provider).pop("budget_exceeded_fn", None)


def _attach_objective(
    search: BaseSearch, objective: _Objective, stack: ExitStack
) -> None:
    stack.callback(_release_budget_hook, search)
    _override(
        stack,
        search,
        "_benchmark_provider_cls",
        functools.partial(
            _PipelineBenchmarkProvider,
            objective=objective,
            search_name=type(search).__name__,
        ),
    )
    # A stage cache describes a different objective. Explicit seeds remain
    # available; do not scan local/remote isolated-stage warm-start caches.
    _override(stack, search, "_find_similar_cached_configs", lambda max_configs: [])
    if isinstance(search, PopulationBasedSearch):
        # Native normal, suspicious and final verification all dispatch through
        # these methods. Do not return a callable that a stock timer could time.
        _override(stack, search, "rebenchmark", MethodType(_rebenchmark, search))
        _override(
            stack,
            search,
            "mirrored_rebenchmark",
            MethodType(_mirrored_rebenchmark, search),
        )


class _CoordinateHybrid(LLMSeededLFBOTreeSearch):
    objective: _Objective
    seeds: tuple[Config, ...]
    overrides: ExitStack
    objective_context: str
    llm_stage_started = False
    second_stage_started = False
    llm_seed_handed_off = False

    def _make_llm_search(self) -> LLMGuidedSearch:
        self.llm_stage_started = True
        search = super()._make_llm_search()
        _attach_objective(search, self.objective, self.overrides)
        build_seeds = search._build_seed_configs
        seeds = self.seeds
        _override(
            self.overrides,
            search,
            "_build_seed_configs",
            lambda: list(_unique_configs((*seeds, *build_seeds()))),
        )
        build_prompt = search._build_initial_prompt
        objective_context = self.objective_context
        guidance = (
            "\n\n## Whole-pipeline objective\n"
            "All reported timings are complete-pipeline GPU graph replay latencies "
            "aggregated by geomean or max over every representative input, as "
            "specified in the pipeline context. They are not isolated-stage "
            "latencies. Changing this stage can change downstream tensor shapes, "
            "stage count, and costs. Optimize the whole operation, including "
            "downstream work, for the provided aggregation.\n"
        )
        _override(
            self.overrides,
            search,
            "_build_initial_prompt",
            lambda: build_prompt() + guidance + objective_context,
        )
        return search

    def _make_second_stage_search(self, *, seeded: bool) -> BaseSearch:
        self.second_stage_started = True
        self.llm_seed_handed_off = seeded
        search = super()._make_second_stage_search(seeded=seeded)
        _attach_objective(search, self.objective, self.overrides)
        return search


def search_coordinate(
    stage: PipelineStage,
    evaluate: Callable[[Config], float],
    *,
    seed_configs: Sequence[Config],
    max_evaluations: int,
    algorithm: str = "LLMSeededLFBOTreeSearch",
    effort: str = "full",
    objective_context: str = "",
) -> CoordinateSearchResult:
    """Propose stage configs using whole-operation scores for every search step.

    ``evaluate`` must realize a coherent pipeline bundle, validate every input,
    and return its positive aggregate latency in milliseconds (or infinity for
    a rejected candidate). It owns graph/eager timing and repetition policy.
    Every call, including rechecks, consumes one evaluation from the shared
    coordinate budget. No candidate compiles or benchmarks outside this callback.

    The incumbent is evaluated first. ``finite`` measures only the incumbent
    and distinct explicit/settings seeds; it never invokes an LLM or LFBO.
    The default invokes the existing hybrid algorithm, ConfigSpec and effort
    profile, including the existing LLM environment/settings controls.
    ``objective_context`` supplies pipeline source, traces, other configs, and
    aggregation details for the LLM prompt; it does not change the search space.
    """
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int):
        raise TypeError("max_evaluations must be an integer")
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    if algorithm not in {"finite", "LLMSeededLFBOTreeSearch"}:
        raise ValueError(f"Unsupported pipeline coordinate algorithm: {algorithm}")
    if effort not in {"none", "quick", "full"}:
        raise ValueError(f"Unknown autotune effort: {effort}")
    if not isinstance(objective_context, str):
        raise TypeError("objective_context must be a string")
    started = time.perf_counter()
    settings = copy.copy(stage.bound.settings)
    seeds = _unique_configs(
        (stage.config, *seed_configs, *normalize_autotune_seed_configs(settings))
    )
    settings.autotune_seed_configs = seeds
    settings.autotune_effort = cast("Literal['none', 'quick', 'full']", effort)
    objective = _Objective(evaluate, max_evaluations)
    hybrid: _CoordinateHybrid | None = None
    with ExitStack() as overrides:
        try:
            objective.measure(seeds[0])
            if algorithm == "finite":
                for config in seeds[1:]:
                    objective.measure(config, phase="finite")
            else:
                view = cast("_AutotunableKernel", _KernelView(stage.bound, settings))
                kwargs = LLMSeededLFBOTreeSearch.get_kwargs_from_profile(
                    get_effort_profile(settings.autotune_effort), settings
                )
                # BaseSearch's shared profile helper also emits this at the outer
                # level, while the hybrid constructor accepts it only in its child
                # kwargs (already populated by the same profile).
                kwargs.pop("max_generations", None)
                hybrid = _CoordinateHybrid(view, stage.args, **kwargs)
                hybrid.objective = objective
                hybrid.seeds = seeds
                hybrid.overrides = overrides
                hybrid.objective_context = objective_context
                _attach_objective(hybrid, objective, overrides)
                hybrid.autotune(skip_cache=True)
        except _CoordinateBudgetExhausted:
            pass
    ranked = sorted(
        (
            (config, score)
            for config, score in objective.latest.items()
            if math.isfinite(score)
        ),
        key=operator.itemgetter(1),
    )
    metadata: dict[str, object] = {
        "objective": "whole_pipeline_callback_ms",
        "effort": effort,
        "coordinate_final_verification": "whole_pipeline_callback",
        "rebenchmark_timing_policy": "evaluator_owned",
        "cache_reuse": False,
        "isolated_stage_cache_seeds": False,
        "search_completed": not objective.exhausted,
        "measurements_by_phase": dict(objective.measurements_by_phase),
        "successful_llm_proposals": objective.successful_llm_proposals,
        "objective_context_provided": bool(objective_context),
    }
    if hybrid is not None:
        metadata["hybrid_stage_breakdown"] = hybrid.hybrid_stage_breakdown
        metadata["llm_model"] = hybrid.llm_model
        metadata["llm_provider"] = hybrid.llm_provider
        metadata["llm_effort_level"] = hybrid.llm_effort_level
        metadata["llm_fast_mode"] = hybrid.llm_fast_mode
        metadata["llm_stage_started"] = hybrid.llm_stage_started
        metadata["second_stage_started"] = hybrid.second_stage_started
        metadata["llm_seed_handed_off"] = hybrid.llm_seed_handed_off
    return CoordinateSearchResult(
        configs=tuple(config for config, _score in ranked),
        scores=tuple(score for _config, score in ranked),
        method=algorithm,
        elapsed_seconds=time.perf_counter() - started,
        evaluations=objective.evaluations,
        budget_exhausted=objective.exhausted,
        hybrid_metadata=metadata,
    )
