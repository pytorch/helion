"""Private in-process pipeline objective routing; ordinary timers are unchanged."""

from __future__ import annotations

from dataclasses import dataclass
import os
import statistics
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import TypedDict

from .benchmark_objective import BenchmarkObjectiveError
from .benchmark_objective import BenchmarkObjectiveSession
from .benchmark_objective import BenchmarkRequest
from .workload_timer import PipelineWorkload
from .workload_timer import benchmark_workloads

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import SupportsIndex

    from ..runtime.config import Config
    from ..runtime.settings import Settings
    from .base_search import _AutotunableKernel
    from .benchmark_objective import BenchmarkObjective
    from .benchmark_objective import BenchmarkWorkload
    from .benchmark_objective import WorkloadMeasurement
    from .benchmark_objective import WorkloadResult
    from .workload_timer import GraphPolicy


class _SampleOptions(TypedDict, total=False):
    samples: int


@dataclass(frozen=True)
class ObjectiveTiming:
    """Explicit shared flush and graph policy, with a caller-owned cache identity.

    cache_tag identifies the physical device, flush implementation and policy.
    The callback and captured resources must remain unchanged during the search.
    Initial/search timing retains its millisecond targets unless search_samples
    is a positive sample budget. This includes nonfinal polish remeasurements.
    Warmup remains millisecond-driven in either mode. Final verification
    uses 120 samples by default so an expensive reset/flush is not repeated
    millions of times based on a short kernel-only event estimate. Set
    final_samples=None to retain millisecond-driven final verification.
    """

    clear_cache: Callable[[], None]
    graph_policy: GraphPolicy
    cache_tag: str
    final_samples: int | None = 120
    search_samples: int | None = None

    @property
    def search_benchmark_samples(self) -> int | None:
        # Preserve custom measure overrides and their legacy call signature.
        if type(self).measure is not ObjectiveTiming.measure:
            return None
        return self.search_samples

    @property
    def final_rebenchmark_samples(self) -> int | None:
        # Existing custom measure implementations keep their original call
        # signature and timing semantics. Only the built-in timer opts in.
        if type(self).measure is not ObjectiveTiming.measure:
            return None
        return self.final_samples

    def cache_record(self) -> dict[str, object]:
        if self.graph_policy not in ("required", "disabled") or not self.cache_tag:
            raise BenchmarkObjectiveError(
                "An explicit graph/flush identity is required"
            )
        if self.final_samples is not None and (
            type(self.final_samples) is not int or self.final_samples <= 0
        ):
            raise BenchmarkObjectiveError(
                "final_samples must be a positive int or None"
            )
        if self.search_samples is not None and (
            type(self.search_samples) is not int or self.search_samples <= 0
        ):
            raise BenchmarkObjectiveError(
                "search_samples must be a positive int or None"
            )
        return {
            "timer": "helion.workload_timer95.v1",
            "graph": self.graph_policy,
            "flush": self.cache_tag,
            "initial_warmup_ms": 1,
            "initial_rep_ms": 50,
            "final_policy_version": 1,
            "final_samples": self.final_rebenchmark_samples,
            "search_policy_version": 1,
            "search_samples": self.search_benchmark_samples,
        }

    def measure(
        self,
        workloads: Sequence[BenchmarkWorkload],
        *,
        warmup_ms: float,
        rep_ms: float,
        samples: int | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        options: _SampleOptions = {}
        if samples is not None:
            options["samples"] = samples
        return tuple(
            result.samples_ms
            for result in benchmark_workloads(
                [PipelineWorkload(w.run, w.reset) for w in workloads],
                clear_cache=self.clear_cache,
                graph_policy=self.graph_policy,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
                **options,
            )
        )


@dataclass(frozen=True)
class _TimedObjective:
    factory: BenchmarkObjective
    timing: ObjectiveTiming

    @property
    def raw_args(self) -> Sequence[object]:
        return self.factory.raw_args

    def cache_record(self) -> Mapping[str, object]:
        return {
            "objective": dict(self.factory.cache_record()),
            "timing": self.timing.cache_record(),
        }

    def bind(
        self, request: BenchmarkRequest, private_args: Sequence[object]
    ) -> BenchmarkWorkload:
        return self.factory.bind(request, private_args)


class ObjectiveBenchmark:
    """One immutable session shared by provider, population history and cache."""

    def __init__(self, factory: BenchmarkObjective, timing: ObjectiveTiming) -> None:
        self._timing = timing
        self._session = BenchmarkObjectiveSession(_TimedObjective(factory, timing))
        self.observations: list[tuple[str, WorkloadResult]] = []

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> NoReturn:
        raise BenchmarkObjectiveError(
            "Pipeline objectives cannot serialize an entire search"
        )

    @property
    def epoch(self) -> str:
        return self._session.epoch

    @property
    def final_rebenchmark_samples(self) -> int | None:
        self.validate()
        return self._timing.final_rebenchmark_samples

    @property
    def search_benchmark_samples(self) -> int | None:
        self.validate()
        return self._timing.search_benchmark_samples

    def validate(self) -> None:
        self._session.validate_epoch()

    def validate_settings(self, kernel: _AutotunableKernel, settings: Settings) -> None:
        self.validate()
        if (
            settings.autotune_timing_method != "default"
            or settings.autotune_benchmark_subprocess
            or settings.autotune_benchmark_fn is not None
            or settings.autotune_baseline_fn is not None
            or settings.autotune_baseline_accuracy_check_fn is not None
            or not settings.autotune_accuracy_check
            or settings.autotune_cache
            not in ("LocalAutotuneCache", "StrictLocalAutotuneCache")
            or kernel.config_spec.backend.name != "cute"
            or kernel.config_spec.cute_flash_search_enabled
            or kernel.env.process_group_name is not None
            or kernel.env.device.type != "cuda"
            or kernel.config_spec.backend.get_do_bench() is not None
            or kernel.config_spec.backend.get_paired_device_micros_bench() is not None
        ):
            raise BenchmarkObjectiveError(
                "Pipeline objectives require local in-process CuTe timing and their full-boundary oracle"
            )
        isolation = os.getenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_ISOLATED")
        if isolation is not None and isolation != "0":
            raise BenchmarkObjectiveError(
                "Pipeline objective finalist isolation is unsupported; request in-process timing"
            )

    def extra_cache_key(self, default_key: str) -> str:
        return self._session.extra_cache_key(default_key)

    def request(
        self,
        kernel: _AutotunableKernel,
        config: Config,
        compiled: Callable[..., object],
    ) -> BenchmarkRequest:
        self.validate()
        source = kernel.config_spec.backend.generated_source_hash(compiled)
        if (
            not isinstance(source, str)
            or len(source) != 64
            or any(c not in "0123456789abcdef" for c in source)
        ):
            raise BenchmarkObjectiveError(
                "Pipeline candidate has no exact generated-source hash"
            )
        return BenchmarkRequest.create(config, compiled, source)

    def check(
        self,
        request: BenchmarkRequest,
        measurement: WorkloadMeasurement | None,
        perf: float,
    ) -> None:
        self.validate()
        if measurement is None:
            raise BenchmarkObjectiveError("Pipeline score has no workload evidence")
        measurement.validate(self.epoch, request)
        if statistics.median(measurement.samples_ms) != perf:
            raise BenchmarkObjectiveError("Pipeline score differs from its samples")

    def measure(
        self,
        requests: Sequence[BenchmarkRequest],
        *,
        phase: str,
        warmup_ms: float,
        rep_ms: float,
        samples: int | None = None,
    ) -> list[WorkloadResult]:
        results = self._session.measure(
            requests,
            lambda workloads: self._timing.measure(
                workloads,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
                **({"samples": samples} if samples is not None else {}),
            ),
        )
        self.observations.extend((phase, result) for result in results)
        return results
