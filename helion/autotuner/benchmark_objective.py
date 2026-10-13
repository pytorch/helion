"""Candidate-private workloads for an explicitly selected benchmark objective.

This module does not select a search algorithm or a timer. An immutable objective
owns the complete workload boundary, and a session applies that same boundary to
every requested candidate. The ordinary single-kernel path does not use it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
from typing import TYPE_CHECKING
from typing import Protocol

from ..runtime.config import Config
from .kernel_args import _clone_args

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence


class BenchmarkObjectiveError(RuntimeError):
    """The requested workload could not retain its declared timing boundary."""


def _encode(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclasses.dataclass(frozen=True)
class BenchmarkRequest:
    config_json: str
    compiled: Callable[..., object]
    source_hash: str

    @classmethod
    def create(
        cls, config: Config, compiled: Callable[..., object], source_hash: str
    ) -> BenchmarkRequest:
        return cls(_encode(config.config), compiled, source_hash)

    @property
    def config(self) -> Config:
        # A factory cannot mutate the caller's config through this request.
        return Config.from_json(self.config_json)

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _encode((self.config_json, self.source_hash)).encode()
        ).hexdigest()


@dataclasses.dataclass(frozen=True)
class BenchmarkWorkload:
    run: Callable[[], object]
    reset: Callable[[], None]
    validate: Callable[[object], None]
    close: Callable[[], None]


class BenchmarkObjective(Protocol):
    """A fixed pipeline tuple and its raw-input/oracle/timing contract.

    ``cache_record`` must identify all partner sources/configs, raw input and
    alias ABI, input recipe, reset/oracle/tolerances, and timing/device policy.
    Changing any of those requires a fresh session and search. ``bind`` receives
    private physical input storage and must retain it for the workload lifetime.
    It substitutes the candidate directly, without invoking nested autotuning.
    Non-tensor raw argument leaves are immutable, as in the existing kernel
    argument cloning contract. ``close`` releases any factory-owned resources.
    """

    @property
    def raw_args(self) -> Sequence[object]: ...

    def cache_record(self) -> Mapping[str, object]: ...

    def bind(
        self, request: BenchmarkRequest, private_args: Sequence[object]
    ) -> BenchmarkWorkload: ...


@dataclasses.dataclass(frozen=True)
class WorkloadMeasurement:
    epoch: str
    request_digest: str
    samples_ms: tuple[float, ...]

    def validate(self, epoch: str, request: BenchmarkRequest) -> None:
        if self.epoch != epoch or self.request_digest != request.digest:
            raise BenchmarkObjectiveError("Measurement belongs to another workload")
        if not self.samples_ms or any(
            isinstance(sample, bool) or not math.isfinite(sample) or sample <= 0
            for sample in self.samples_ms
        ):
            raise BenchmarkObjectiveError("Workload timing is not finite and positive")


@dataclasses.dataclass(frozen=True)
class WorkloadResult:
    request: BenchmarkRequest
    measurement: WorkloadMeasurement | None
    accuracy_error: str | None = None


class WorkloadTimer(Protocol):
    """Return samples in workload order after all timed work is complete.

    Each workload must be invoked through its supplied run callback. A captured
    graph keeps that callback's returned output buffers stable across replays.
    """

    def __call__(
        self, workloads: Sequence[BenchmarkWorkload]
    ) -> Sequence[Sequence[float]]: ...


@dataclasses.dataclass
class _TrackedWorkload:
    workload: BenchmarkWorkload
    last_output: object = None
    invoked: bool = False

    def run(self) -> object:
        output = self.workload.run()
        self.last_output = output
        self.invoked = True
        return output


class BenchmarkObjectiveSession:
    """Own one immutable objective epoch across candidate scoring phases."""

    def __init__(self, objective: BenchmarkObjective) -> None:
        self.objective = objective
        self._record = _encode(dict(objective.cache_record()))
        self._epoch = hashlib.sha256(self._record.encode()).hexdigest()
        self._raw_args = _clone_args(objective.raw_args, None)

    @property
    def epoch(self) -> str:
        return self._epoch

    def validate_epoch(self) -> None:
        record = _encode(dict(self.objective.cache_record()))
        if (
            record != self._record
            or hashlib.sha256(record.encode()).hexdigest() != self._epoch
        ):
            raise BenchmarkObjectiveError(
                "Benchmark objective changed; start a fresh search epoch"
            )

    def extra_cache_key(self, default_key: str) -> str:
        self.validate_epoch()
        return _encode(("helion.benchmark_objective.v1", default_key, self.epoch))

    def measurement_key(self, request: BenchmarkRequest) -> tuple[str, str]:
        self.validate_epoch()
        return self.epoch, request.digest

    @contextlib.contextmanager
    def prepare(self, request: BenchmarkRequest) -> Iterator[BenchmarkWorkload]:
        self.validate_epoch()
        private_args = _clone_args(self._raw_args, None)
        workload = self.objective.bind(request, private_args)
        try:
            self.validate_epoch()
            yield workload
            self.validate_epoch()
        finally:
            workload.close()

    def measure(
        self, requests: Sequence[BenchmarkRequest], timer: WorkloadTimer
    ) -> list[WorkloadResult]:
        """Validate the full boundary before and after timing each workload.

        ``timer`` must reset before every setup/capture/warmup/sample invocation,
        outside the measured boundary. Reset is deliberately separate from run.
        Final validation reads the retained last timed/captured output without
        resetting or rerunning, so it cannot conceal the final timed mutation.
        Only an oracle AssertionError is classified as a candidate rejection;
        launch, capture, identity, and resource failures propagate to the caller.
        """
        self.validate_epoch()
        results: list[WorkloadResult | None] = [None] * len(requests)
        accepted: list[tuple[int, BenchmarkRequest, _TrackedWorkload]] = []
        with contextlib.ExitStack() as stack:
            for index, request in enumerate(requests):
                workload = stack.enter_context(self.prepare(request))
                workload.reset()
                output = workload.run()
                try:
                    workload.validate(output)
                except AssertionError as error:
                    results[index] = WorkloadResult(request, None, str(error))
                else:
                    accepted.append((index, request, _TrackedWorkload(workload)))
                self.validate_epoch()
            if accepted:
                samples = timer(
                    [
                        dataclasses.replace(tracked.workload, run=tracked.run)
                        for _, _, tracked in accepted
                    ]
                )
                if len(samples) != len(accepted):
                    raise BenchmarkObjectiveError("Timer omitted a candidate workload")
                self.validate_epoch()
                for (index, request, tracked), values in zip(
                    accepted, samples, strict=True
                ):
                    measurement = WorkloadMeasurement(
                        self.epoch, request.digest, tuple(values)
                    )
                    measurement.validate(self.epoch, request)
                    if not tracked.invoked:
                        raise BenchmarkObjectiveError(
                            "Timer did not invoke a candidate workload"
                        )
                    try:
                        tracked.workload.validate(tracked.last_output)
                    except AssertionError as error:
                        results[index] = WorkloadResult(request, None, str(error))
                    else:
                        results[index] = WorkloadResult(request, measurement)
                    self.validate_epoch()
        self.validate_epoch()
        assert all(result is not None for result in results)
        return [result for result in results if result is not None]
