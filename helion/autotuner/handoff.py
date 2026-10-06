"""Choose where configuration search should hand off to source optimization."""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
import copy
import dataclasses
import hashlib
import math
import statistics
import time
from typing import TYPE_CHECKING
from typing import Literal

from .. import exc
from .search_space_logger import canonical_config_id
from helion._dist_utils import all_gather_object
from helion._dist_utils import sync_object

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..runtime.config import Config
    from .base_cache import AutotuneCacheBase
    from .base_search import BaseSearch
    from .benchmark_provider import BenchmarkResult


@dataclasses.dataclass(frozen=True)
class HandoffPolicy:
    """Stop at the first enabled trigger, or when the search completes.

    Trial/time limits are checked after complete benchmark batches, so they
    may be exceeded by one batch. Final confirmation happens after stopping
    and can extend the total run time beyond the limit.
    """

    # Stop after this many search trials, including rejected candidates.
    # None disables this limit.
    after_trials: int | None = None
    # Stop after this many elapsed search seconds.
    # None disables this limit.
    after_seconds: float | None = None
    # Optional predicate on batch progress; return True to request handoff.
    callback: Callable[[HandoffProgress], bool] | None = None
    # Top candidates to remeasure, deduplicated by source when available.
    # The returned config may add one extra candidate.
    finalists: int = 5
    # Fresh confirmation passes per finalist (at least 3); rank by their median.
    repetitions: int = 3

    def __post_init__(self) -> None:
        for name, value in (
            ("after_trials", self.after_trials),
            ("finalists", self.finalists),
        ):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.after_seconds is not None and (
            not math.isfinite(self.after_seconds) or self.after_seconds <= 0
        ):
            raise ValueError("after_seconds must be finite and positive")
        if not isinstance(self.repetitions, int) or self.repetitions < 3:
            raise ValueError("repetitions must be at least 3")


@dataclasses.dataclass(frozen=True)
class HandoffProgress:
    """Search progress; confirmation measurements do not count as trials."""

    trials: int
    unique_sources: int
    elapsed_seconds: float
    best_perf: float | None


@dataclasses.dataclass(frozen=True)
class HandoffMeasurement:
    """A detached observation, including failures and confirmation samples."""

    phase: Literal["search", "confirmation"]
    algorithm: str
    config: Config
    perf: float | None
    status: str
    source_hash: str | None
    elapsed_seconds: float
    per_shape: tuple[float, ...] = ()
    per_shape_statuses: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class HandoffCandidate:
    """A finalist that passed every confirmation; performance is the median.

    ``noise`` is the median absolute deviation of the repeated measurements,
    in the same objective units as ``perf`` (not a confidence interval).
    """

    config: Config
    source_hash: str | None
    samples: tuple[float, ...]
    perf: float
    noise: float


@dataclasses.dataclass(frozen=True)
class HandoffPoint:
    """Selected starting kernel and the evidence for a later source optimizer.

    ``fn`` is the anchor kernel callable for multi-shape searches; measurements
    cover every shape and retain the search's aggregate objective.
    """

    reason: str
    config: Config
    fn: Callable[..., object]
    progress: HandoffProgress
    finalists: tuple[HandoffCandidate, ...]
    measurements: tuple[HandoffMeasurement, ...]
    objective_unit: str
    reference_latencies: tuple[float, ...] | None


class _HandoffStop(BaseException):
    # Search implementations may recover from Exception. This control-flow
    # signal must unwind all nested stages, including their provider cleanup.
    pass


_active_handoff: ContextVar[_HandoffSession | None] = ContextVar(
    "helion_autotune_handoff", default=None
)


def _observe_handoff(search: BaseSearch, results: list[BenchmarkResult]) -> None:
    session = _active_handoff.get()
    if session is not None:
        session.observe(search, results)


class _HandoffSession:
    def __init__(self, search: BaseSearch, policy: HandoffPolicy) -> None:
        self.search = search
        self.policy = policy
        self.start = time.perf_counter()
        self.trials = 0
        self.sources: set[str] = set()
        self.measurements: list[HandoffMeasurement] = []
        self.candidates: dict[str, HandoffMeasurement] = {}
        self.reason: str | None = None
        self.callback_error: str | None = None

    def progress(self) -> HandoffProgress:
        values = [m.perf for m in self.candidates.values() if m.perf is not None]
        return HandoffProgress(
            self.trials,
            len(self.sources),
            time.perf_counter() - self.start,
            min(values, default=None),
        )

    def _measurement(
        self,
        search: BaseSearch,
        result: BenchmarkResult,
        phase: Literal["search", "confirmation"],
    ) -> HandoffMeasurement:
        from .benchmark_provider import _materialize_multi_shape_config
        from .benchmark_provider import _MultiShapeAutotuneArgs

        valid = (
            result.status in ("ok", "deduplicated")
            and math.isfinite(result.perf)
            and result.perf > 0
        )
        per_shape: tuple[float, ...] = ()
        per_shape_statuses: tuple[str, ...] = ()
        config = result.config
        cases = [search.kernel]
        if isinstance(search.args, _MultiShapeAutotuneArgs):
            cases = [kernel for kernel, _ in search.args.cases]
            # Invalid configs have no materialized measurement to retrieve.
            with contextlib.suppress(exc.InvalidConfig):
                config = _materialize_multi_shape_config(search.config_spec, config)
            if (measured := search.args.measurements.get(repr(config))) is not None:
                per_shape, _, per_shape_statuses = measured
        source_hash = None
        if valid:
            hashes = []
            for kernel in cases:
                fn = (
                    result.fn
                    if kernel is search.kernel
                    else kernel.bench_compile_config(config, allow_print=False)
                )
                key = kernel.config_spec.backend.generated_source_hash(fn)
                if key is None:
                    source = kernel.to_code(config)
                    if source is None:
                        break
                    key = hashlib.sha256(source.encode()).hexdigest()
                hashes.append(key)
            else:
                source_hash = hashlib.sha256(repr(hashes).encode()).hexdigest()
        measurement = HandoffMeasurement(
            phase,
            type(search).__name__,
            copy.deepcopy(result.config),
            result.perf if valid else None,
            result.status,
            source_hash,
            (
                result.completed_at
                if result.completed_at is not None
                else time.perf_counter()
            )
            - self.start,
            per_shape,
            per_shape_statuses,
        )
        self.measurements.append(measurement)
        if phase == "confirmation":
            self.search.log.record_handoff_measurement(
                measurement,
                search.performance_unit,
                completed_at=result.completed_at,
            )
        return measurement

    def observe(self, search: BaseSearch, results: list[BenchmarkResult]) -> None:
        for result in results:
            measurement = self._measurement(search, result, "search")
            self.trials += 1
            key = canonical_config_id(result.config)
            if measurement.perf is not None:
                self.candidates[key] = measurement
                if measurement.source_hash is not None:
                    self.sources.add(measurement.source_hash)
            else:
                self.candidates.pop(key, None)

        # Agree on callbacks and time limits only after the whole batch has
        # completed its collectives. Propagate callback errors to every rank.
        reason = None
        error = None
        try:
            progress = self.progress()
            policy = self.policy
            if (
                policy.after_trials is not None
                and progress.trials >= policy.after_trials
            ):
                reason = "trials"
            elif (
                policy.after_seconds is not None
                and progress.elapsed_seconds >= policy.after_seconds
            ):
                reason = "time"
            elif policy.callback is not None and policy.callback(progress):
                reason = "callback"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        group = search.kernel.env.process_group_name
        errors = all_gather_object(error, group)
        if any(errors):
            self.callback_error = f"Handoff callback failed: {errors}"
            raise _HandoffStop
        reason = sync_object(reason, group)
        if reason is not None:
            self.reason = reason
            raise _HandoffStop

    def confirm(self, extra: Config | None = None) -> tuple[HandoffCandidate, ...]:
        from .benchmark_provider import MultiShapeBenchmarkProvider
        from .benchmark_provider import _MultiShapeAutotuneArgs
        from .metrics import AutotuneMetrics

        search = self.search
        ranked = sorted(
            self.candidates.values(),
            key=lambda m: m.perf if m.perf is not None else math.inf,
        )
        configs: list[Config] = []
        seen: set[str] = set()
        for measurement in ranked:
            key = measurement.source_hash or canonical_config_id(measurement.config)
            if key not in seen:
                seen.add(key)
                configs.append(measurement.config)
            if len(configs) == self.policy.finalists:
                break
        # Include the returned config even if its search timing did not place
        # it in the top K.
        if extra is not None and extra not in configs:
            configs.append(copy.deepcopy(extra))
        configs = sync_object(configs, search.kernel.env.process_group_name)
        if not configs:
            return ()
        samples: dict[Config, list[HandoffMeasurement]] = {c: [] for c in configs}
        settings = copy.copy(search.settings)
        settings.autotune_accuracy_check = True
        confirmation_log = copy.copy(search.log)
        # Keep text diagnostics, but record these as confirmation events instead
        # of adding more trials to the config search's CSV/dataset/trace.
        confirmation_log._log_sink = None
        confirmation_log._trace_sink = None
        provider_cls = (
            MultiShapeBenchmarkProvider
            if isinstance(search.args, _MultiShapeAutotuneArgs)
            else search._benchmark_provider_cls
        )
        for repetition in range(self.policy.repetitions):
            # Fresh providers avoid reusing deduplicated search timings. Rotate
            # order across passes without perturbing the search's random state.
            offset = repetition % len(configs)
            ordered = configs[offset:] + configs[:offset]
            provider = provider_cls(
                kernel=search.kernel,
                settings=settings,
                config_spec=search.config_spec,
                args=search.args,
                log=confirmation_log,
                autotune_metrics=AutotuneMetrics(),
            )
            if isinstance(provider, MultiShapeBenchmarkProvider):
                for child in provider.children:
                    child.settings = copy.copy(child.settings)
                    child.settings.autotune_accuracy_check = True
            try:
                provider.setup()
                for result in provider.benchmark(ordered, desc="Confirming handoff"):
                    measurement = self._measurement(search, result, "confirmation")
                    samples[result.config].append(measurement)
            finally:
                provider.cleanup()
        confirmed = []
        for config, measurements in samples.items():
            values = tuple(m.perf for m in measurements if m.perf is not None)
            key = canonical_config_id(config)
            if len(values) != self.policy.repetitions:
                self.candidates.pop(key, None)
                continue
            perf = statistics.median(values)
            candidate = HandoffCandidate(
                copy.deepcopy(config),
                measurements[-1].source_hash,
                values,
                perf,
                statistics.median(abs(value - perf) for value in values),
            )
            confirmed.append(candidate)
            self.candidates[key] = dataclasses.replace(measurements[-1], perf=perf)
        # A finalist must succeed on every rank. Share one ranking so every
        # rank selects the same kernel, even with noisy timers.
        valid_ids = {canonical_config_id(c.config) for c in confirmed}
        invalid_ids = {canonical_config_id(c) for c in configs} - valid_ids
        group = search.kernel.env.process_group_name
        for invalid in all_gather_object(invalid_ids, group):
            invalid_ids = invalid_ids | invalid
        for key in invalid_ids:
            self.candidates.pop(key, None)
        return sync_object(
            tuple(
                sorted(
                    (
                        c
                        for c in confirmed
                        if canonical_config_id(c.config) not in invalid_ids
                    ),
                    key=lambda candidate: candidate.perf,
                )
            ),
            group,
        )


def find_handoff(
    autotuner: BaseSearch | AutotuneCacheBase,
    policy: HandoffPolicy | None = None,
    *,
    skip_cache: bool = False,
) -> HandoffPoint:
    """Run any search (or cache-wrapped search) up to a source handoff point.

    This opt-in API selects and remeasures a starting kernel; it does not launch
    an agent or install source edits. Ordinary ``autotune()`` is unchanged.
    Cache hits are revalidated. Normal cache behavior is preserved when a search
    completes; early handoff unwinds before the cache write.
    An internal stop unwinds nested hybrid/best-of-K searches at batch boundaries.

    If ``autotune_baseline_fn`` triggers autotuning of another Helion kernel,
    those trials share this handoff session and can interfere with selection.
    """
    from .base_cache import AutotuneCacheBase
    from .benchmark_provider import _MultiShapeAutotuneArgs

    search = (
        autotuner.autotuner if isinstance(autotuner, AutotuneCacheBase) else autotuner
    )
    session = _HandoffSession(search, policy or HandoffPolicy())
    token = _active_handoff.set(session)
    try:
        with search.log.autotune_tracing("find_handoff"):
            returned = None
            with contextlib.suppress(_HandoffStop):
                returned = autotuner.autotune(skip_cache=skip_cache)
            if session.callback_error is not None:
                raise RuntimeError(session.callback_error)
            reason = session.reason or "completed"
            finalists = session.confirm(returned)
            if not finalists:
                raise exc.NoConfigFound
            best = finalists[0]
            point = HandoffPoint(
                reason,
                copy.deepcopy(best.config),
                search.kernel.compile_config(best.config, allow_print=False),
                session.progress(),
                finalists,
                tuple(session.measurements),
                search.performance_unit,
                search.args.reference_latencies
                if isinstance(search.args, _MultiShapeAutotuneArgs)
                else None,
            )
            search.log.record_handoff(point)
            return point
    finally:
        _active_handoff.reset(token)
