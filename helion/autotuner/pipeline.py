"""Joint configuration search with complete-pipeline, multi-input objectives."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from ..runtime.pipeline import PipelineBudgetExhausted
from ..runtime.pipeline import PipelineConfig
from ..runtime.pipeline import PipelineInitialConfig
from ..runtime.pipeline import PipelineKeyFunction
from ..runtime.pipeline import PipelineStage
from .pipeline_benchmark import PipelineEvaluator
from .pipeline_benchmark import PipelineMeasurement
from .pipeline_proposals import _resolve_algorithm
from .pipeline_proposals import search_coordinate

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..runtime.config import Config


@dataclass
class PipelineTuningResult:
    """Selected coherent bundle plus graph measurements and search provenance."""

    config: PipelineConfig
    timings_ms: tuple[float, ...]
    aggregate_ms: float
    incumbent_timings_ms: tuple[float, ...]
    incumbent_aggregate_ms: float
    traces: tuple[tuple[str, ...], ...]
    metadata: dict[str, Any]
    history: list[dict[str, Any]]

    @property
    def per_input_speedups(self) -> tuple[float, ...]:
        return tuple(
            old / new
            for old, new in zip(self.incumbent_timings_ms, self.timings_ms, strict=True)
        )

    @property
    def regressed_input_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, speedup in enumerate(self.per_input_speedups)
            if speedup < 1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "timings_ms": list(self.timings_ms),
            "aggregate_ms": self.aggregate_ms,
            "incumbent_timings_ms": list(self.incumbent_timings_ms),
            "incumbent_aggregate_ms": self.incumbent_aggregate_ms,
            "per_input_speedups": list(self.per_input_speedups),
            "regressed_input_indices": list(self.regressed_input_indices),
            "traces": [list(trace) for trace in self.traces],
            "metadata": self.metadata,
            "history": self.history,
        }

    def save(self, path: str | Path) -> None:
        """Save a pipeline-specific receipt without editing any stage AOT cache."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")


class _PipelineSearch:
    def __init__(
        self,
        evaluator: PipelineEvaluator,
        *,
        config_candidates: Callable[[PipelineStage], Sequence[Config]] | None,
        algorithm: str,
        effort: str,
        max_evaluations: int,
        max_seconds: float | None,
        coordinate_evaluations: int,
        beam_width: int,
        topology_refinements: int,
        rounds: int,
        final_top_k: int,
    ) -> None:
        self.evaluator = evaluator
        self.config_candidates = config_candidates
        self.algorithm = algorithm
        self.effort = effort
        self.max_evaluations = max_evaluations
        self.max_seconds = max_seconds
        self.coordinate_evaluations = coordinate_evaluations
        self.beam_width = beam_width
        self.topology_refinements = topology_refinements
        self.rounds = rounds
        self.final_top_k = final_top_k
        self.started = time.perf_counter()
        self.evaluations = 0
        self.confirmations = 0
        self.history: list[dict[str, Any]] = []
        self.coordinate_history: list[dict[str, Any]] = []
        self.pool: list[PipelineMeasurement] = []

    def remaining(self) -> int:
        if (
            self.max_seconds is not None
            and time.perf_counter() - self.started >= self.max_seconds
        ):
            return 0
        return self.max_evaluations - self.evaluations

    def measure(self, bundle: PipelineConfig, *, phase: str) -> PipelineMeasurement:
        if phase == "confirmation":
            self.confirmations += 1
        else:
            if phase != "incumbent" and self.remaining() <= 0:
                raise PipelineBudgetExhausted("Pipeline search budget exhausted")
            self.evaluations += 1
        result = self.evaluator.evaluate(bundle)
        self.history.append({"phase": phase, **result.to_dict()})
        return result

    def keep(self, result: PipelineMeasurement) -> None:
        if result.status != "ok" or not math.isfinite(result.aggregate_ms):
            return
        candidates = {item.bundle.digest(): item for item in [*self.pool, result]}
        ranked = sorted(candidates.values(), key=lambda item: item.aggregate_ms)
        # Retain the best measured schedules plus a bounded set of genuinely
        # different topologies. Only this small pool retains live stage inputs.
        selected = ranked[: self.final_top_k]
        seen = {item.traces for item in selected}
        for item in ranked:
            if len(selected) >= self.final_top_k + self.beam_width:
                break
            if item.traces not in seen:
                selected.append(item)
                seen.add(item.traces)
        self.pool = selected

    def beam(self) -> list[PipelineMeasurement]:
        ranked = sorted(self.pool, key=lambda item: item.aggregate_ms)
        selected: list[PipelineMeasurement] = []
        seen = set()
        for item in ranked:
            if item.traces not in seen:
                selected.append(item)
                seen.add(item.traces)
            if len(selected) == self.beam_width:
                return selected
        for item in ranked:
            if all(item is not old for old in selected):
                selected.append(item)
            if len(selected) == self.beam_width:
                break
        return selected

    def seeds(self, stage: PipelineStage) -> list[Config]:
        candidates = [stage.config]
        if self.config_candidates is not None:
            candidates.extend(self.config_candidates(stage))
        else:
            candidates.extend(
                [
                    stage.bound.config_spec.default_config(),
                    stage.bound.config_spec.autotune_reference_config(),
                ]
            )
        unique: dict[str, Config] = {}
        for config in candidates:
            unique.setdefault(config.to_json(), config)
        return list(unique.values())

    def refine_new_stages(
        self, result: PipelineMeasurement, previous_keys: set[str]
    ) -> PipelineMeasurement:
        if result.status != "ok":
            return result
        # New branches get downstream trials before their first raw score can
        # eliminate them. Later coordinate rounds revisit upstream decisions.
        new_keys = [key for key in result.stages if key not in previous_keys]
        best = result
        for key in new_keys[: self.topology_refinements]:
            if self.remaining() <= 0:
                break
            if key not in best.stages:
                continue
            stage = best.stages[key]
            for config in self.seeds(stage)[1 : self.coordinate_evaluations]:
                if self.remaining() <= 0:
                    break
                proposal = self.measure(
                    best.bundle.with_config(key, config), phase="topology_refinement"
                )
                if (
                    proposal.status == "ok"
                    and proposal.aggregate_ms < best.aggregate_ms
                ):
                    best = proposal
        return best

    def run_coordinates(self) -> None:
        if self.effort == "none":
            return
        for round_index in range(self.rounds):
            for base in self.beam():
                for key in base.stages:
                    if self.remaining() <= 0:
                        return
                    if key not in base.stages:
                        continue
                    stage = base.stages[key]
                    best_for_coordinate = base

                    def evaluate(
                        config: Config,
                        base: PipelineMeasurement = base,
                        key: str = key,
                    ) -> float:
                        nonlocal best_for_coordinate
                        result = self.measure(
                            base.bundle.with_config(key, config), phase="coordinate"
                        )
                        refined = self.refine_new_stages(result, set(base.stages))
                        self.keep(refined)
                        if (
                            refined.status == "ok"
                            and refined.aggregate_ms < best_for_coordinate.aggregate_ms
                        ):
                            best_for_coordinate = refined
                        return refined.aggregate_ms

                    try:
                        result = search_coordinate(
                            stage,
                            evaluate,
                            seed_configs=self.seeds(stage),
                            max_evaluations=min(
                                self.coordinate_evaluations, self.remaining()
                            ),
                            algorithm=self.algorithm,
                            effort=self.effort,
                        )
                        details = result.to_dict()
                    except PipelineBudgetExhausted:
                        details = {"status": "global_budget_exhausted"}
                    self.coordinate_history.append(
                        {
                            "round": round_index,
                            "stage_key": key,
                            **details,
                        }
                    )
                    # Advance within this branch. Jumping to the global winner
                    # here would abandon slower topologies before their later
                    # downstream coordinates had a chance to improve.
                    base = best_for_coordinate


def autotune_pipeline(
    pipeline_fn: Callable[..., object],
    arg_sets: Sequence[Sequence[object]],
    *,
    reference: Callable[..., object],
    check: Callable[[object, object], None],
    benchmark: Callable[[Callable[[], Any]], float] | None = None,
    aggregation: Literal["geomean", "max"] = "geomean",
    initial_bundle: PipelineConfig | None = None,
    initial_config: PipelineInitialConfig | None = None,
    key_fn: PipelineKeyFunction | None = None,
    config_candidates: Callable[[PipelineStage], Sequence[Config]] | None = None,
    algorithm: str | None = None,
    effort: str = "full",
    max_evaluations: int = 512,
    max_seconds: float | None = None,
    coordinate_evaluations: int = 256,
    beam_width: int = 2,
    topology_refinements: int = 2,
    rounds: int = 2,
    final_top_k: int = 3,
) -> PipelineTuningResult:
    """Optimize all stages using complete GPU-graph latency on every input.

    Each coordinate uses its own ConfigSpec and a real Helion search algorithm:
    LFBO, LLM-guided search, or LLM-seeded LFBO. The default follows
    ``HELION_AUTOTUNER``, falling back to ``LFBOTreeSearch`` like normal autotuning.
    Candidate scores are always whole-pipeline geomeans (or maxima), including
    downstream stages introduced by a topology change. ``algorithm='finite'``
    provides bounded, explicit candidate comparisons without an LLM request.

    Search budgets cover initial discovery and candidate evaluations. Final
    confirmation, up to ``final_top_k + 1`` complete measurements, is separate
    and always includes the original incumbent. Time limits are checked between
    complete evaluations and never interrupt GPU work. Inputs must be immutable
    and replay-safe; internal scratch and shape-controlled host loops are allowed.

    Custom benchmark callbacks receive only a captured graph's replay callable
    and return milliseconds. Saved bundles can be passed as ``initial_bundle``
    and are validated and measured again.
    """
    if (
        min(max_evaluations, coordinate_evaluations, beam_width, rounds, final_top_k)
        < 1
    ):
        raise ValueError("Pipeline search budgets and widths must be positive")
    if topology_refinements < 0 or (max_seconds is not None and max_seconds <= 0):
        raise ValueError("Invalid pipeline refinement or time budget")
    algorithm = _resolve_algorithm(algorithm)
    if effort not in ("none", "quick", "full"):
        raise ValueError("Pipeline effort must be 'none', 'quick', or 'full'")
    evaluator = PipelineEvaluator(
        pipeline_fn,
        arg_sets,
        reference=reference,
        check=check,
        benchmark=benchmark,
        aggregation=aggregation,
        key_fn=key_fn,
        initial_config=initial_config,
    )
    search = _PipelineSearch(
        evaluator,
        config_candidates=config_candidates,
        algorithm=algorithm,
        effort=effort,
        max_evaluations=max_evaluations,
        max_seconds=max_seconds,
        coordinate_evaluations=coordinate_evaluations,
        beam_width=beam_width,
        topology_refinements=topology_refinements,
        rounds=rounds,
        final_top_k=final_top_k,
    )
    incumbent = search.measure(initial_bundle or PipelineConfig(), phase="incumbent")
    if incumbent.status != "ok":
        raise ValueError(
            f"Initial pipeline config failed validation: {incumbent.error}"
        )
    search.keep(incumbent)
    search.run_coordinates()
    ranked = sorted(search.pool, key=lambda item: item.aggregate_ms)
    finalists = [incumbent]
    seen = {incumbent.bundle.digest()}
    for candidate in ranked[:final_top_k]:
        if candidate.bundle.digest() not in seen:
            finalists.append(candidate)
            seen.add(candidate.bundle.digest())
    confirmed_incumbent = search.measure(incumbent.bundle, phase="confirmation")
    if confirmed_incumbent.status != "ok":
        raise RuntimeError("Incumbent failed final pipeline confirmation")
    winner = confirmed_incumbent
    for candidate in finalists[1:]:
        measured = search.measure(candidate.bundle, phase="confirmation")
        if measured.status == "ok" and measured.aggregate_ms < winner.aggregate_ms:
            winner = measured
    active_keys = {key for trace in winner.traces for key in trace}
    selected = PipelineConfig(
        {key: winner.bundle.configs[key] for key in active_keys},
        key_fn=winner.bundle._key_fn,
    )
    return PipelineTuningResult(
        config=selected,
        timings_ms=winner.timings_ms,
        aggregate_ms=winner.aggregate_ms,
        incumbent_timings_ms=confirmed_incumbent.timings_ms,
        incumbent_aggregate_ms=confirmed_incumbent.aggregate_ms,
        traces=winner.traces,
        metadata={
            "search_method": algorithm,
            "effort": effort,
            "aggregation": aggregation,
            "input_count": len(arg_sets),
            "autotune_seconds": time.perf_counter() - search.started,
            "search_evaluations": search.evaluations,
            "confirmation_evaluations": search.confirmations,
            "search_budget": max_evaluations,
            "time_budget_seconds": max_seconds,
            "coordinate_searches": search.coordinate_history,
        },
        history=search.history,
    )
